import asyncio
import concurrent.futures
import os
from concurrent.futures import ProcessPoolExecutor
from queue import Empty, Queue
from typing import Any, Callable, List, NamedTuple, Optional

import torch

from tensorrt_llm.llmapi.utils import print_colored_debug
from tensorrt_llm.logger import logger

from ..bindings import executor as tllm
from ..disaggregated_params import DisaggregatedParams
from ..llmapi.mpi_session import (MpiCommSession, MpiPoolSession, MpiSession,
                                  RemoteMpiCommSessionClient)
from ..llmapi.utils import print_colored_debug

PERIODICAL_RESP_IN_AWAIT = os.getenv(
    "TLLM_EXECUTOR_PERIODICAL_RESP_IN_AWAIT") == "1"


def get_spawn_proxy_process_ipc_addr_env() -> str | None:
    ''' Get the IPC address for the spawn proxy process dynamically. '''
    return os.getenv("TLLM_SPAWN_PROXY_PROCESS_IPC_ADDR")


def get_spawn_proxy_process_env() -> bool:
    ''' Get the environment variable for the spawn proxy process dynamically. '''
    return os.getenv("TLLM_SPAWN_PROXY_PROCESS") == "1"


if PERIODICAL_RESP_IN_AWAIT:
    logger.info("Using periodical responses in await_responses")


def create_mpi_comm_session(
        n_workers: int) -> RemoteMpiCommSessionClient | MpiPoolSession:
    if get_spawn_proxy_process_env():
        assert get_spawn_proxy_process_ipc_addr_env(
        ), "TLLM_SPAWN_PROXY_PROCESS_IPC_ADDR is not set."
        print_colored_debug(
            f"Using RemoteMpiPoolSessionClient to bind to external MPI processes at {get_spawn_proxy_process_ipc_addr_env()}\n",
            "yellow")
        return RemoteMpiCommSessionClient(
            get_spawn_proxy_process_ipc_addr_env())
    else:
        print_colored_debug(
            f"Using MpiCommSession to bind to external MPI processes\n",
            "yellow")
        return MpiCommSession(n_workers=n_workers)


def has_event_loop() -> bool:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return False
    return True


class RequestError(RuntimeError):
    ''' The error raised when the request is failed. '''


class ProcessPoolExecutorSession(MpiSession):
    # This process pool is introduced for better recoverable exceptions handling.
    # It replaces MpiPoolExecutor for single-gpu case.

    def __init__(self, n_workers: int, **kwargs):
        self.n_workers = n_workers
        self.mpi_pool = ProcessPoolExecutor(max_workers=self.n_workers,
                                            **kwargs)

    def submit(self, task: Callable, *args,
               **kwargs) -> List[concurrent.futures.Future]:
        return [
            self.mpi_pool.submit(task, *args, **kwargs)
            for i in range(self.n_workers)
        ]

    def submit_sync(self, task: Callable, *args, **kwargs) -> List[Any]:
        futures = [
            self.mpi_pool.submit(task, *args, **kwargs)
            for i in range(self.n_workers)
        ]
        return [future.result() for future in futures]

    def shutdown(self):
        self.mpi_pool.shutdown(wait=False)


class ExecutorResponseTensors(NamedTuple):
    output_token_ids: List[List[int]]
    # context_logits is a tensor or a string denoting the path to the shared memory.
    context_logits: Optional[torch.Tensor | str]
    # generation_logits is a tensor or a string denoting the path to the shared memory.
    generation_logits: Optional[torch.Tensor | str]
    log_probs: Optional[list]
    cum_log_probs: Optional[list]


class ErrorResponse(NamedTuple):
    client_id: int
    error_msg: str
    request_id: int


class ExecutorResponse(NamedTuple):
    """ The response from the cpp-executor to the Python main thread. """
    client_id: int
    tensors: Optional[ExecutorResponseTensors]
    finish_reasons: Optional[List[tllm.FinishReason]]
    is_final: Optional[bool]
    sequence_index: Optional[int]
    # There are two types of errors:
    # 1. str for the errors from the cpp-executor.await_responses, this will be dispatched to the user's
    #    generate_async as a per-request error, and won't stop the whole service.
    # 2. Exception for the errors from the background threads/processes, this will be processed in the main thread,
    #    and stop the whole service.
    error: Optional[str | Exception]
    # The timestamp of the creation of the response. We use this to track the IPC overhead.
    timestamp: Optional[float] = None
    # Optional disaggregated serving params needed by the generation instances
    disaggregated_params: Optional[DisaggregatedParams] = None


class IntraProcessQueue:
    ''' A Queue-like container for IPC within the same process. '''

    def __init__(self):
        self.queue = Queue()

    def put(self, obj: Any):
        self.queue.put(obj)

    def get(self, timeout=None) -> Any:
        return self.queue.get(timeout=timeout)

    def close(self):
        pass

    def poll(self, timeout=None) -> bool:
        try:
            # Try to get an item from the queue without blocking
            item = self.queue.get(timeout=timeout)
            # If successful, put the item back to not alter the state
            self.queue.put(item)
            return True
        except Empty:
            # If the queue thread is empty, return False
            return False


class WorkerCommIpcAddrs(NamedTuple):
    ''' IPC addresses for communication with the worker processes. '''
    request_queue_addr: str
    request_error_queue_addr: str
    result_queue_addr: str
    stats_queue_addr: str
    kv_cache_events_queue_addr: str
