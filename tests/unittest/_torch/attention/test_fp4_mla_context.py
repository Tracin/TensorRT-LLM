# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FP4 MLA context FMHA tests."""

import torch

import tensorrt_llm._torch.attention_backend.fmha.fp4_mla as fp4_mla_fmha


def test_fp4_mla_fp8_context_defaults_on_for_supported_architectures(monkeypatch):
    monkeypatch.delenv(fp4_mla_fmha.FP4_MLA_FP8_CONTEXT_ENV, raising=False)
    monkeypatch.setattr(fp4_mla_fmha, "get_sm_version", lambda: 107)
    assert fp4_mla_fmha.fp4_mla_fp8_context_enabled()

    monkeypatch.setenv(fp4_mla_fmha.FP4_MLA_FP8_CONTEXT_ENV, "0")
    assert not fp4_mla_fmha.fp4_mla_fp8_context_enabled()

    monkeypatch.delenv(fp4_mla_fmha.FP4_MLA_FP8_CONTEXT_ENV, raising=False)
    monkeypatch.setattr(fp4_mla_fmha, "get_sm_version", lambda: 100)
    assert fp4_mla_fmha.fp4_mla_fp8_context_enabled()

    monkeypatch.setenv(fp4_mla_fmha.FP4_MLA_FP8_CONTEXT_ENV, "1")
    assert fp4_mla_fmha.fp4_mla_fp8_context_enabled()

    monkeypatch.setattr(fp4_mla_fmha, "get_sm_version", lambda: 80)
    assert not fp4_mla_fmha.fp4_mla_fp8_context_enabled()


def test_fp4_mla_fp8_context_cache_update_overlaps_and_rejoins(monkeypatch):
    operations = []

    class FakeStream:
        def __init__(self, name):
            self.name = name

        def wait_event(self, event):
            operations.append(f"{self.name}:wait:{event.name}")

    class FakeEvent:
        def __init__(self, name):
            self.name = name

        def record(self, stream):
            operations.append(f"{stream.name}:record:{self.name}")

    class FakeStreamContext:
        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            operations.append(f"enter:{self.stream.name}")

        def __exit__(self, *_args):
            operations.append(f"exit:{self.stream.name}")

    current_stream = FakeStream("current")
    aux_stream = FakeStream("aux")
    monkeypatch.setattr(fp4_mla_fmha.torch.cuda, "current_stream", lambda: current_stream)
    monkeypatch.setattr(fp4_mla_fmha.torch.cuda, "stream", FakeStreamContext)

    fp4_mla_fmha._execute_fp8_context_with_cache_update(
        lambda: operations.append("attention"),
        lambda: operations.append("cache"),
        aux_stream,
        FakeEvent("start"),
        FakeEvent("done"),
    )

    assert operations == [
        "current:record:start",
        "attention",
        "enter:aux",
        "aux:wait:start",
        "cache",
        "aux:record:done",
        "exit:aux",
        "current:wait:done",
    ]


def test_fp4_mla_fp8_context_scratch_block_tables():
    block_offsets, block_ids, total_blocks = fp4_mla_fmha._build_fp8_mla_context_block_tables(
        [1, 128, 129],
        max_num_sequences=4,
        max_blocks_per_seq=3,
        page_size=128,
    )

    expected = torch.tensor(
        [[0, 0, 0], [1, 0, 0], [2, 3, 0], [0, 0, 0]],
        dtype=torch.int32,
    )
    assert total_blocks == 4
    torch.testing.assert_close(block_ids, expected)
    torch.testing.assert_close(block_offsets[0, :, 0], expected)
    torch.testing.assert_close(block_offsets[0, :, 1], expected)
