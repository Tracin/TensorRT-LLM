# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import weakref
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Callable, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from tensorrt_llm._torch.attention_backend.fp4_mla import (
    FP4_MLA_Q_RESIDUAL_DIM,
    FP4_MLA_TOKENS_PER_BLOCK,
    apply_fp4_mla_rope,
    get_fp4_mla_hp_block_size,
    run_fp4_mla_attention_decode,
    scatter_fp4_mla_kv_cache,
    update_hp_kv_for_fp4_mla,
)
from tensorrt_llm._torch.attention_backend.interface import (
    AttentionForwardArgs,
    AttentionInputType,
    PredefinedAttentionMask,
)
from tensorrt_llm._utils import get_sm_version, is_sm_100f, prefer_pinned
from tensorrt_llm.bindings import DataType
from tensorrt_llm.logger import logger
from tensorrt_llm.quantization.mode import QuantMode

from .fallback import FallbackFmha
from .phased import FmhaParams, PhasedFmha

if TYPE_CHECKING:
    from tensorrt_llm._torch.attention_backend.trtllm import (
        TrtllmAttention,
        TrtllmAttentionMetadata,
    )


FP4_MLA_FP8_CONTEXT_ENV = "TRTLLM_FP4_MLA_FP8_CONTEXT"
_FP8_CONTEXT_SUPPORTED_SMS = {90, 100, 103, 107, 120}
_FP8_CONTEXT_SCRATCH_ATTR = "_fp4_mla_fp8_context_scratch"


def fp4_mla_fp8_context_enabled() -> bool:
    """Use TRT-LLM's FP8 MLA context FMHA on supported architectures."""
    sm = get_sm_version()
    value = os.environ.get(FP4_MLA_FP8_CONTEXT_ENV)
    enabled = True if value is None else value == "1"
    return enabled and sm in _FP8_CONTEXT_SUPPORTED_SMS


def _execute_fp8_context_with_cache_update(
    attention_fn: Callable[[], None],
    cache_update_fn: Callable[[], None],
    aux_stream: Optional[torch.cuda.Stream],
    start_event: Optional[torch.cuda.Event],
    done_event: Optional[torch.cuda.Event],
) -> None:
    """Overlap FP8 context attention with its independent FP4 cache update."""
    if aux_stream is None or start_event is None or done_event is None:
        cache_update_fn()
        attention_fn()
        return

    current_stream = torch.cuda.current_stream()
    if aux_stream == current_stream:
        cache_update_fn()
        attention_fn()
        return

    start_event.record(current_stream)
    attention_fn()
    with torch.cuda.stream(aux_stream):
        aux_stream.wait_event(start_event)
        cache_update_fn()
        done_event.record(aux_stream)
    current_stream.wait_event(done_event)


def _build_fp8_mla_context_block_tables(
    context_lengths: Sequence[int],
    *,
    max_num_sequences: int,
    max_blocks_per_seq: int,
    page_size: int,
    pin_memory: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Build compact page IDs for the disposable FP8 context cache."""
    if len(context_lengths) > max_num_sequences:
        raise ValueError(
            f"FP8 MLA context scratch supports at most {max_num_sequences} sequences, "
            f"got {len(context_lengths)}."
        )

    block_ids = torch.zeros(
        max_num_sequences,
        max_blocks_per_seq,
        dtype=torch.int32,
        device="cpu",
        pin_memory=pin_memory,
    )
    next_block = 0
    for seq_idx, context_length in enumerate(context_lengths):
        context_length = int(context_length)
        if context_length < 0:
            raise ValueError(f"Context length must be non-negative, got {context_length}.")
        num_blocks = (context_length + page_size - 1) // page_size
        if num_blocks > max_blocks_per_seq:
            raise ValueError(
                f"Context sequence requires {num_blocks} FP8 scratch blocks, but the "
                f"page table holds only {max_blocks_per_seq}."
            )
        if num_blocks:
            block_ids[seq_idx, :num_blocks] = torch.arange(
                next_block,
                next_block + num_blocks,
                dtype=torch.int32,
            )
            next_block += num_blocks

    block_offsets = torch.zeros(
        1,
        max_num_sequences,
        2,
        max_blocks_per_seq,
        dtype=torch.int32,
        device="cpu",
        pin_memory=pin_memory,
    )
    block_offsets[0, :, 0].copy_(block_ids)
    block_offsets[0, :, 1].copy_(block_ids)
    return block_offsets, block_ids, next_block


@dataclass
class _Fp8MlaContextScratch:
    pool: torch.Tensor
    block_offsets: torch.Tensor
    block_ids_per_seq: torch.Tensor
    host_pool_pointers: torch.Tensor
    host_pool_mapping: torch.Tensor
    max_num_sequences: int
    max_blocks_per_seq: int
    capacity_blocks: int
    page_size: int
    head_dim: int
    cache_stream: torch.cuda.Stream
    cache_start_event: torch.cuda.Event
    cache_done_event: torch.cuda.Event
    mapping_signature: Optional[Tuple[int, ...]] = None
    host_staging: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

    @classmethod
    def create(
        cls,
        meta: "TrtllmAttentionMetadata",
        *,
        device: torch.device,
        head_dim: int,
    ) -> "_Fp8MlaContextScratch":
        kv_cache_manager = meta.kv_cache_manager
        if kv_cache_manager is None:
            raise RuntimeError("FP8 MLA context scratch requires a KV cache manager.")

        page_size = int(meta.tokens_per_block)
        max_num_sequences = int(meta.max_num_sequences or meta.max_num_requests)
        max_blocks_per_seq = int(kv_cache_manager.max_blocks_per_seq)
        max_num_tokens = int(meta.max_num_tokens)
        max_nonempty_sequences = min(max_num_sequences, max_num_tokens)
        capacity_blocks = max(
            1,
            (max_num_tokens + page_size - 1) // page_size + max(0, max_nonempty_sequences - 1),
        )
        pool = torch.empty(
            capacity_blocks * page_size * head_dim,
            dtype=torch.float8_e4m3fn,
            device=device,
        )
        block_offsets = torch.zeros(
            1,
            max_num_sequences,
            2,
            max_blocks_per_seq,
            dtype=torch.int32,
            device=device,
        )
        block_ids_per_seq = torch.zeros(
            max_num_sequences,
            max_blocks_per_seq,
            dtype=torch.int32,
            device=device,
        )
        host_pool_pointers = torch.tensor(
            [[pool.data_ptr(), 0]],
            dtype=torch.int64,
            device="cpu",
            pin_memory=prefer_pinned(),
        )
        host_pool_mapping = torch.zeros(
            1,
            2,
            dtype=torch.int32,
            device="cpu",
            pin_memory=prefer_pinned(),
        )
        return cls(
            pool=pool,
            block_offsets=block_offsets,
            block_ids_per_seq=block_ids_per_seq,
            host_pool_pointers=host_pool_pointers,
            host_pool_mapping=host_pool_mapping,
            max_num_sequences=max_num_sequences,
            max_blocks_per_seq=max_blocks_per_seq,
            capacity_blocks=capacity_blocks,
            page_size=page_size,
            head_dim=head_dim,
            cache_stream=torch.cuda.Stream(device=device),
            cache_start_event=torch.cuda.Event(),
            cache_done_event=torch.cuda.Event(),
        )

    def matches(
        self,
        meta: "TrtllmAttentionMetadata",
        *,
        device: torch.device,
        head_dim: int,
    ) -> bool:
        kv_cache_manager = meta.kv_cache_manager
        return (
            kv_cache_manager is not None
            and self.pool.device == device
            and self.head_dim == head_dim
            and self.page_size == meta.tokens_per_block
            and self.max_num_sequences >= int(meta.max_num_sequences or meta.max_num_requests)
            and self.max_blocks_per_seq >= int(kv_cache_manager.max_blocks_per_seq)
        )

    def prepare(self, meta: "TrtllmAttentionMetadata") -> None:
        context_lengths = tuple(
            int(length) for length in meta.prompt_lens_cpu_runtime[: meta.num_contexts].tolist()
        )
        if context_lengths == self.mapping_signature:
            return

        host_offsets, host_block_ids, required_blocks = _build_fp8_mla_context_block_tables(
            context_lengths,
            max_num_sequences=self.max_num_sequences,
            max_blocks_per_seq=self.max_blocks_per_seq,
            page_size=self.page_size,
            pin_memory=prefer_pinned(),
        )
        if required_blocks > self.capacity_blocks:
            raise RuntimeError(
                f"FP8 MLA context scratch requires {required_blocks} blocks, but only "
                f"{self.capacity_blocks} were allocated."
            )
        self.block_offsets.copy_(host_offsets, non_blocking=True)
        self.block_ids_per_seq.copy_(host_block_ids, non_blocking=True)
        self.mapping_signature = context_lengths
        self.host_staging = (host_offsets, host_block_ids)


class _Fp8MlaContextAttnProxy:
    """Override cache-only attention attributes while forwarding model config."""

    def __init__(self, attn: "TrtllmAttention") -> None:
        self._attn = weakref.proxy(attn)
        self.quant_mode = int(QuantMode(0).set_fp8_kv_cache())
        self.local_layer_idx = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self._attn, name)


class _Fp8MlaContextMetadataProxy:
    """Route only the mandatory FP8 cache write to disposable storage."""

    def __init__(
        self,
        meta: "TrtllmAttentionMetadata",
        scratch: _Fp8MlaContextScratch,
    ) -> None:
        self._meta = meta
        self.kv_cache_block_offsets = scratch.block_offsets
        self.host_kv_cache_pool_pointers = scratch.host_pool_pointers
        self.host_kv_cache_pool_mapping = scratch.host_pool_mapping
        self.block_ids_per_seq = scratch.block_ids_per_seq

    def __getattr__(self, name: str) -> Any:
        return getattr(self._meta, name)


class Fp4MlaFmha(PhasedFmha):
    """TRTLLM FMHA library for the no-dequant NVFP4 MLA decode kernel."""

    SUPPORTED_Q_DTYPES = {torch.bfloat16, torch.float8_e4m3fn}
    SUPPORTED_CONTEXT_DTYPES = {torch.float16, torch.bfloat16}
    SUPPORTED_OUTPUT_DTYPES = {torch.float16, torch.bfloat16}

    def __init__(self, attn: "TrtllmAttention") -> None:
        super().__init__(attn)
        self._fp8_context_attn_proxy = _Fp8MlaContextAttnProxy(attn)
        self._fp8_context_fmha = FallbackFmha(self._fp8_context_attn_proxy)

    @classmethod
    def is_available(cls, attn: "TrtllmAttention") -> bool:
        if not attn.is_mla_enable:
            logger.debug("FP4 MLA FMHA is unavailable: requires MLA.")
            return False
        if not QuantMode(attn.quant_mode).has_fp4_kv_cache():
            logger.debug("FP4 MLA FMHA is unavailable: requires NVFP4 KV cache quantization.")
            return False
        if attn.attention_chunk_size not in (None, 0):
            logger.debug("FP4 MLA FMHA is unavailable: chunked attention is not supported.")
            return False
        if attn.predicted_tokens_per_seq > get_fp4_mla_hp_block_size():
            logger.debug(
                "FP4 MLA FMHA is unavailable: linear MTP length exceeds "
                "FP4 MLA HP rollback support."
            )
            return False
        if attn.kv_lora_rank is None or attn.qk_rope_head_dim is None:
            logger.debug("FP4 MLA FMHA is unavailable: missing MLA dimensions.")
            return False
        if attn.qk_nope_head_dim is None or attn.v_head_dim is None:
            logger.debug("FP4 MLA FMHA is unavailable: missing MLA context dimensions.")
            return False
        if attn.qk_rope_head_dim != FP4_MLA_Q_RESIDUAL_DIM:
            logger.debug(
                f"FP4 MLA FMHA is unavailable: requires qk_rope_head_dim="
                f"{FP4_MLA_Q_RESIDUAL_DIM}, got {attn.qk_rope_head_dim}."
            )
            return False
        context_head_dim = attn.qk_nope_head_dim + attn.qk_rope_head_dim
        fused_head_dim = attn.kv_lora_rank + attn.qk_rope_head_dim
        if attn.head_dim not in (context_head_dim, fused_head_dim):
            logger.debug(
                "FP4 MLA FMHA is unavailable: head_dim must equal either "
                f"qk_nope_head_dim + qk_rope_head_dim ({context_head_dim}) or "
                f"kv_lora_rank + qk_rope_head_dim ({fused_head_dim})."
            )
            return False
        sm = get_sm_version()
        if not is_sm_100f(sm):
            logger.debug(f"FP4 MLA FMHA is unavailable: requires SM100 or SM103, got SM{sm}.")
            return False
        if not hasattr(torch.ops, "trtllm") or not hasattr(
            torch.ops.trtllm, "fp4_quantize_with_residual"
        ):
            logger.debug("FP4 MLA FMHA is unavailable: missing trtllm FP4 quantization op.")
            return False
        return True

    def is_supported(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        metadata: "TrtllmAttentionMetadata",
        forward_args: AttentionForwardArgs,
    ) -> bool:
        supported, reason = self._is_supported_with_reason(
            q, k, v, self.attn, metadata, forward_args
        )
        if not supported:
            logger.debug(f"FP4 MLA FMHA does not support request: {reason}")
        return supported

    @classmethod
    def _is_supported_with_reason(
        cls,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        attn: "TrtllmAttention",
        meta: "TrtllmAttentionMetadata",
        fwd: AttentionForwardArgs,
    ) -> Tuple[bool, str]:
        if fwd.attention_input_type == AttentionInputType.context_only:
            return cls._is_context_supported_with_reason(q, k, v, attn, meta, fwd)

        if fwd.attention_input_type != AttentionInputType.generation_only:
            return False, "supports generation-only attention."
        if meta.num_generations <= 0:
            return False, "requires generation requests."
        if k is not None or v is not None:
            return False, "expects fused MLA query input."
        if fwd.output is None:
            return False, "requires output."
        if fwd.latent_cache is None:
            return False, "requires latent_cache."
        if q.dtype not in cls.SUPPORTED_Q_DTYPES:
            return False, f"unsupported query dtype {q.dtype}."
        if fwd.output.dtype not in cls.SUPPORTED_OUTPUT_DTYPES:
            return False, f"unsupported output dtype {fwd.output.dtype}."
        if fwd.output_sf is not None:
            return False, "does not support quantized attention output."
        if fwd.attention_mask != PredefinedAttentionMask.CAUSAL:
            return False, "requires causal mask."
        if fwd.attention_mask_data is not None:
            return False, "does not support custom attention masks."
        if fwd.attention_sinks is not None:
            return False, "does not support attention sinks."
        if fwd.sage_attn_num_elts_per_blk_q > 0 or fwd.sage_attn_num_elts_per_blk_k > 0:
            return False, "does not support sage attention."
        if fwd.sage_attn_num_elts_per_blk_v > 0:
            return False, "does not support sage attention."
        sparse = fwd.sparse_prediction
        if (
            (sparse.sparse_kv_indices is not None and sparse.sparse_kv_indices.numel() > 0)
            or (sparse.sparse_attn_indices is not None and sparse.sparse_attn_indices.numel() > 0)
            or meta.num_sparse_topk > 0
        ):
            return False, "does not support sparse attention."
        if meta.helix_position_offsets is not None:
            return False, "does not support helix parallelism."
        if meta.use_spec_decoding and meta.is_spec_dec_tree:
            return False, "does not support speculative decoding trees."
        if meta.kv_cache_manager is None:
            return False, "requires a KV cache manager."
        if meta.kv_cache_manager.dtype != DataType.NVFP4:
            return False, f"requires NVFP4 KV cache storage, got {meta.kv_cache_manager.dtype}."
        if meta.kv_cache_manager.kv_factor != 1:
            return False, "requires MLA SELF-K-only KV cache."
        if meta.kv_cache_block_offsets is None:
            return False, "requires paged KV cache block offsets."
        if meta.high_precision_kv_pool is None:
            return False, "requires high-precision KV pool."
        if meta.fp4_mla_v_scale_pool is None:
            return False, "requires FP4 MLA V-scale pool."
        if meta.batch_indices is None or meta.positions is None:
            return False, "requires FP4 MLA append metadata."
        if (
            meta._paged_kv_indptr is None
            or meta.paged_kv_indptr_decode is None
            or meta._paged_kv_indices is None
        ):
            return False, "requires FP4 MLA page metadata."
        if meta.tokens_per_block != FP4_MLA_TOKENS_PER_BLOCK:
            return (
                False,
                f"requires tokens_per_block={FP4_MLA_TOKENS_PER_BLOCK}, "
                f"got {meta.tokens_per_block}.",
            )
        if fwd.attention_window_size is not None and fwd.attention_window_size < meta.max_seq_len:
            return False, "does not support sliding-window attention."
        if meta.beam_width != 1:
            return False, f"does not support beam search, got beam_width={meta.beam_width}."
        fused_head_dim = attn.kv_lora_rank + attn.qk_rope_head_dim
        if q.shape[-1] != attn.num_heads * fused_head_dim:
            return False, f"unexpected fused query hidden size {q.shape[-1]}."
        if fwd.latent_cache.shape[-1] != fused_head_dim:
            return False, f"unexpected latent_cache hidden size {fwd.latent_cache.shape[-1]}."
        if q.shape[0] != fwd.latent_cache.shape[0]:
            return False, "query and latent_cache token counts do not match."
        if q.shape[0] < meta.num_generations:
            return False, "not enough query tokens for generation batch."
        if q.shape[0] % meta.num_generations != 0:
            return False, "requires uniform linear MTP generation length."

        return True, ""

    @classmethod
    def _is_context_supported_with_reason(
        cls,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        attn: "TrtllmAttention",
        meta: "TrtllmAttentionMetadata",
        fwd: AttentionForwardArgs,
    ) -> Tuple[bool, str]:
        if meta.num_contexts <= 0:
            return False, "requires context requests."
        if getattr(meta, "num_ctx_cached_tokens", 0) != 0:
            return False, "does not support cached-context FP4 MLA prefill."
        if k is None or v is None:
            return False, "requires expanded context K and V tensors."
        if fwd.output is None:
            return False, "requires output."
        if fwd.latent_cache is None:
            return False, "requires latent_cache."
        if q.dtype not in cls.SUPPORTED_CONTEXT_DTYPES:
            return False, f"unsupported context query dtype {q.dtype}."
        if k.dtype != q.dtype or v.dtype != q.dtype:
            return False, "requires matching context q/k/v dtypes."
        if fwd.output.dtype not in cls.SUPPORTED_OUTPUT_DTYPES:
            return False, f"unsupported output dtype {fwd.output.dtype}."
        if fwd.output_sf is not None:
            return False, "does not support quantized context output."
        if fwd.attention_mask != PredefinedAttentionMask.CAUSAL:
            return False, "requires causal mask."
        if fwd.attention_mask_data is not None:
            return False, "does not support custom attention masks."
        if fwd.attention_sinks is not None:
            return False, "does not support attention sinks."
        if fwd.sage_attn_num_elts_per_blk_q > 0 or fwd.sage_attn_num_elts_per_blk_k > 0:
            return False, "does not support sage attention."
        if fwd.sage_attn_num_elts_per_blk_v > 0:
            return False, "does not support sage attention."
        sparse = fwd.sparse_prediction
        if (
            (sparse.sparse_kv_indices is not None and sparse.sparse_kv_indices.numel() > 0)
            or (sparse.sparse_attn_indices is not None and sparse.sparse_attn_indices.numel() > 0)
            or meta.num_sparse_topk > 0
        ):
            return False, "does not support sparse attention."
        if meta.helix_position_offsets is not None:
            return False, "does not support helix parallelism."
        if meta.kv_cache_manager is None:
            return False, "requires a KV cache manager."
        if meta.kv_cache_manager.dtype != DataType.NVFP4:
            return False, f"requires NVFP4 KV cache storage, got {meta.kv_cache_manager.dtype}."
        if meta.kv_cache_manager.kv_factor != 1:
            return False, "requires MLA SELF-K-only KV cache."
        if meta.kv_cache_block_offsets is None:
            return False, "requires paged KV cache block offsets."
        if meta.high_precision_kv_pool is None:
            return False, "requires high-precision KV pool."
        if meta.fp4_mla_v_scale_pool is None:
            return False, "requires FP4 MLA V-scale pool."
        if meta.batch_indices is None or meta.positions is None:
            return False, "requires FP4 MLA append metadata."
        if meta.paged_kv_indptr_decode is None or meta._paged_kv_indices is None:
            return False, "requires FP4 MLA page metadata."
        if meta.tokens_per_block != FP4_MLA_TOKENS_PER_BLOCK:
            return (
                False,
                f"requires tokens_per_block={FP4_MLA_TOKENS_PER_BLOCK}, "
                f"got {meta.tokens_per_block}.",
            )
        if fwd.attention_window_size is not None and fwd.attention_window_size < meta.max_seq_len:
            return False, "does not support sliding-window attention."
        if meta.beam_width != 1:
            return False, f"does not support beam search, got beam_width={meta.beam_width}."
        qk_head_dim = attn.qk_nope_head_dim + attn.qk_rope_head_dim
        if q.shape[-1] != attn.num_heads * qk_head_dim:
            return False, f"unexpected context query hidden size {q.shape[-1]}."
        if k.shape[-1] != attn.num_heads * qk_head_dim:
            return False, f"unexpected context key hidden size {k.shape[-1]}."
        if v.shape[-1] != attn.num_heads * attn.v_head_dim:
            return False, f"unexpected context value hidden size {v.shape[-1]}."
        if fwd.latent_cache.shape[-1] != attn.kv_lora_rank + attn.qk_rope_head_dim:
            return False, f"unexpected latent_cache hidden size {fwd.latent_cache.shape[-1]}."
        if q.shape[0] != meta.num_ctx_tokens:
            return False, "query token count must match num_ctx_tokens."
        if k.shape[0] != q.shape[0] or v.shape[0] != q.shape[0]:
            return False, "context q/k/v token counts do not match."
        if fwd.latent_cache.shape[0] < q.shape[0]:
            return False, "latent_cache does not contain all context tokens."

        return True, ""

    def run_mla_context(self, params: FmhaParams) -> None:
        attn = params.attn
        meta = params.meta
        fwd = params.fwd
        if params.qkv_input is None:
            raise RuntimeError("FP4 MLA context requires q input.")
        if params.k_input is None or params.v_input is None:
            raise RuntimeError("FP4 MLA context requires expanded k/v inputs.")
        if params.context_buf is None:
            raise RuntimeError("FP4 MLA context requires context_buf.")
        if fwd.latent_cache is None:
            raise RuntimeError("FP4 MLA context requires latent_cache.")
        if meta.positions is None:
            raise RuntimeError("FP4 MLA context requires positions.")

        local_layer = attn.get_local_layer_idx(meta)
        kv_lora_rank = attn.kv_lora_rank or 0
        qk_nope_head_dim = attn.qk_nope_head_dim or 0
        qk_rope_head_dim = attn.qk_rope_head_dim or 0
        v_head_dim = attn.v_head_dim or 0
        qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        num_tokens = params.num_tokens

        attn._ensure_rope_table_size(meta.max_seq_len)
        positions = meta.positions[:num_tokens]
        k_pe = fwd.latent_cache[:num_tokens, kv_lora_rank:].unsqueeze(1)
        k_pe = apply_fp4_mla_rope(
            k_pe,
            positions,
            attn.rotary_cos_sin,
            attn.rope_params.max_positions,
            qk_rope_head_dim,
        ).squeeze(1)

        latent_cache = torch.empty_like(fwd.latent_cache[:num_tokens])
        latent_cache[..., :kv_lora_rank].copy_(fwd.latent_cache[:num_tokens, :kv_lora_rank])
        latent_cache[..., kv_lora_rank:].copy_(k_pe)

        def update_fp4_cache() -> None:
            scatter_fp4_mla_kv_cache(
                meta,
                latent_cache,
                attn.layer_idx,
                token_offset=0,
                phase="context",
                local_layer=local_layer,
                v_head_dim=kv_lora_rank,
            )
            update_hp_kv_for_fp4_mla(meta, latent_cache, local_layer, phase="context")

        if fp4_mla_fp8_context_enabled() and not meta.is_cuda_graph:
            # FP8 context FMHA reads its Q/K/V workspace; this cache only absorbs
            # the native RoPE helper's mandatory paged-cache write.
            kv_cache_manager = meta.kv_cache_manager
            if kv_cache_manager is None:
                raise RuntimeError("FP8 MLA context scratch requires a KV cache manager.")
            scratch = getattr(kv_cache_manager, _FP8_CONTEXT_SCRATCH_ATTR, None)
            scratch_head_dim = kv_lora_rank + qk_rope_head_dim
            if not isinstance(scratch, _Fp8MlaContextScratch) or not scratch.matches(
                meta,
                device=params.qkv_input.device,
                head_dim=scratch_head_dim,
            ):
                scratch = _Fp8MlaContextScratch.create(
                    meta,
                    device=params.qkv_input.device,
                    head_dim=scratch_head_dim,
                )
                setattr(kv_cache_manager, _FP8_CONTEXT_SCRATCH_ATTR, scratch)
            scratch.prepare(meta)

            fp8_meta = _Fp8MlaContextMetadataProxy(meta, scratch)
            fp8_fwd = replace(
                fwd,
                kv_scale_orig_quant=attn.kv_scale_orig_quant,
                kv_scale_quant_orig=attn.kv_scale_quant_orig,
            )
            _execute_fp8_context_with_cache_update(
                lambda: self._fp8_context_fmha.forward(
                    params.qkv_input,
                    params.k_input,
                    params.v_input,
                    fp8_meta,
                    fp8_fwd,
                ),
                update_fp4_cache,
                scratch.cache_stream,
                scratch.cache_start_event,
                scratch.cache_done_event,
            )
            return

        update_fp4_cache()
        q_ctx = params.qkv_input.view(num_tokens, attn.num_heads, qk_head_dim)
        k_ctx = params.k_input.view(num_tokens, attn.num_heads, qk_head_dim)
        v_ctx = params.v_input.view(num_tokens, attn.num_heads, v_head_dim)
        q_nope = q_ctx[..., :qk_nope_head_dim]
        q_pe = q_ctx[..., qk_nope_head_dim:]
        k_nope = k_ctx[..., :qk_nope_head_dim]
        q_pe = apply_fp4_mla_rope(
            q_pe,
            positions,
            attn.rotary_cos_sin,
            attn.rope_params.max_positions,
            qk_rope_head_dim,
        )

        q_ctx = torch.cat((q_nope, q_pe), dim=-1)
        k_ctx = torch.cat((k_nope, k_pe.unsqueeze(1).expand(-1, attn.num_heads, -1)), dim=-1)
        output = params.context_buf.view(num_tokens, attn.num_heads, v_head_dim)
        sm_scale = 1.0 / (attn.q_scaling * qk_head_dim**0.5)

        host_context_lengths = meta.prompt_lens_cpu_runtime[: meta.num_contexts].tolist()
        token_offset = 0
        for context_len in host_context_lengths:
            if context_len == 0:
                continue
            next_offset = token_offset + int(context_len)
            q_seq = q_ctx[token_offset:next_offset].transpose(0, 1).unsqueeze(0)
            k_seq = k_ctx[token_offset:next_offset].transpose(0, 1).unsqueeze(0)
            v_seq = v_ctx[token_offset:next_offset].transpose(0, 1).unsqueeze(0)
            out_seq = F.scaled_dot_product_attention(
                q_seq,
                k_seq,
                v_seq,
                is_causal=True,
                scale=sm_scale,
            )
            output[token_offset:next_offset].copy_(out_seq.squeeze(0).transpose(0, 1))
            token_offset = next_offset

    def run_mla_generation(self, params: FmhaParams) -> None:
        attn = params.attn
        meta = params.meta
        fwd = params.fwd
        if params.qkv_input is None:
            raise RuntimeError("FP4 MLA generation requires qkv_input.")
        if params.context_buf is None:
            raise RuntimeError("FP4 MLA generation requires context_buf.")
        if fwd.latent_cache is None:
            raise RuntimeError("FP4 MLA generation requires latent_cache.")

        local_layer = attn.get_local_layer_idx(meta)
        kv_lora_rank = attn.kv_lora_rank or 0
        qk_rope_head_dim = attn.qk_rope_head_dim or 0
        fused_head_dim = kv_lora_rank + qk_rope_head_dim

        cache_scattered = bool(getattr(meta, "_fp4_mla_generation_cache_scattered", False))
        if cache_scattered:
            meta._fp4_mla_generation_cache_scattered = False
            hp_pool_updated = True
        else:
            hp_pool_updated = scatter_fp4_mla_kv_cache(
                meta,
                fwd.latent_cache,
                attn.layer_idx,
                token_offset=getattr(meta, "num_ctx_tokens", 0),
                phase="generation",
                local_layer=local_layer,
                v_head_dim=kv_lora_rank,
            )
        if not hp_pool_updated:
            update_hp_kv_for_fp4_mla(meta, fwd.latent_cache, local_layer, phase="generation")

        query = params.qkv_input.view(params.num_tokens, attn.num_heads, fused_head_dim)
        output = params.context_buf.view(params.num_tokens, attn.num_heads, kv_lora_rank)
        sm_scale = 1.0 / (attn.q_scaling * (attn.qk_nope_head_dim + qk_rope_head_dim) ** 0.5)
        run_fp4_mla_attention_decode(
            meta,
            attn.layer_idx,
            local_layer,
            query,
            output,
            sm_scale=sm_scale,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
        )
