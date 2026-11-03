"""Host-side tests of the FP4 MLA decode path selection (no GPU, no torch)."""

import pytest

from tensorrt_llm._torch.attention.backends.fp4_mla import fp4_mla_cutedsl_4x_host as host_4x
from tensorrt_llm._torch.attention.backends.fp4_mla import fp4_mla_dispatch as dp

SM = 212
ROWS_Q4, ROWS_Q2 = 512, 256


def _choose(num_requests, query_len, kv_capacity, *, live=None, four_x=True, **kw):
    rows = 128 * query_len
    return dp.choose_decode_strategy(
        num_requests=num_requests,
        query_len=query_len,
        num_heads=128,
        rows_per_seq=rows,
        kv_capacity=kv_capacity,
        sm_count=SM,
        envelope_reason=host_4x.envelope_reason_static,
        live_max_kv=live,
        four_x_enabled=four_x,
        **kw,
    )


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in (
        dp.DECODE_PATH_ENV,
        dp.MIN_KV_4X_ENV,
        dp.KV_REGIME_GRID_ENV,
        "TRTLLM_FP4_MLA_TWOSTREAM",
        "TRTLLM_FP4_MLA_TWOSTREAM_R2",
        host_4x.MIN_PAIRS_ENV,
        host_4x.SPLIT_MIN_KV_MULTIWAVE_ENV,
    ):
        monkeypatch.delenv(name, raising=False)
    # the measured table was taken with the internal 2x kernel; pin that calibration (the main tree defaults to 1.2)
    monkeypatch.setenv("TRTLLM_FP4_MLA_TWOSTREAM_T2_SCALE", "1.0")


def test_knob_off_is_2x():
    s = _choose(128, 4, 30080, four_x=False)
    assert s.path == dp.PATH_2X


def test_b128_q4_long_kv_is_two_stream():
    s = _choose(128, 4, 30080)
    assert s.path == dp.PATH_TWO_STREAM and s.r2 == 13 and s.num_requests_2x == 13
    assert s.record_tag == "ts13"
    s16 = _choose(128, 4, 16384)
    assert s16.path == dp.PATH_TWO_STREAM and s16.r2 == 13


def test_b128_q4_short_kv_stays_2x():
    assert _choose(128, 4, 8192).path == dp.PATH_2X
    assert _choose(128, 4, 4096).path == dp.PATH_2X
    # live KV wins over the capacity: an 8K step inside a 160K-capacity graph bucket stays on 2x
    s = _choose(128, 4, 163840, live=8192)
    assert s.path == dp.PATH_2X and s.kv_source == "live" and "kv8192" in s.record_tag


def test_live_kv_long_in_big_capacity_bucket():
    s = _choose(128, 4, 163840, live=30080)
    assert s.path == dp.PATH_TWO_STREAM and s.r2 == 13 and s.kv_len_decision == 30080


@pytest.mark.parametrize(
    "num_requests,query_len,kv", [(64, 4, 30080), (32, 4, 30080), (128, 2, 16384), (46, 2, 30080)]
)
def test_integral_waves_use_plain_4x(num_requests, query_len, kv):
    s = _choose(num_requests, query_len, kv)
    assert s.path == dp.PATH_4X and s.r2 == 0, s


def test_b96_q4_two_stream_four_requests():
    s = _choose(96, 4, 30080)
    assert s.path == dp.PATH_TWO_STREAM and s.r2 == 4


def test_small_batch_follows_envelope_rules():
    # 12 requests x 4 m-tiles: the 2x two-piece plan fits one wave -> envelope keeps 2x (host rule)
    s = _choose(12, 4, 30080)
    assert s.path == dp.PATH_2X and "two" in s.reason


def test_helix_and_stats_stay_2x():
    assert _choose(128, 4, 30080, helix=True).path == dp.PATH_2X
    assert _choose(128, 4, 30080, softmax_stats=True).path == dp.PATH_2X
    assert _choose(128, 4, 30080, v_packed=False).path == dp.PATH_2X


def test_forced_paths(monkeypatch):
    monkeypatch.setenv(dp.DECODE_PATH_ENV, "2x")
    assert _choose(128, 4, 30080).path == dp.PATH_2X
    monkeypatch.setenv(dp.DECODE_PATH_ENV, "4x")
    s = _choose(128, 4, 30080)
    assert s.path == dp.PATH_4X and s.kv_source == "forced"
    monkeypatch.setenv(dp.DECODE_PATH_ENV, "twostream")
    assert _choose(128, 4, 30080).path == dp.PATH_TWO_STREAM
    # a forced two-stream falls back to 4x where no split helps, and to 2x outside the envelope
    assert _choose(64, 4, 30080).path == dp.PATH_4X
    assert _choose(128, 4, 30080, helix=True).path == dp.PATH_2X
    # forcing never bypasses the KV floor only in auto mode: forced 4x at 8K is honoured if the envelope allows it
    monkeypatch.setenv(dp.DECODE_PATH_ENV, "4x")
    assert _choose(128, 4, 8192).path == dp.PATH_4X


def test_min_kv_knob(monkeypatch):
    monkeypatch.setenv(dp.MIN_KV_4X_ENV, "4096")
    assert _choose(128, 4, 8192).path == dp.PATH_TWO_STREAM
    monkeypatch.setenv(dp.MIN_KV_4X_ENV, "40000")
    assert _choose(128, 4, 30080).path == dp.PATH_2X


def test_main_tree_2x_calibration_keeps_plain_4x_at_b128(monkeypatch):
    monkeypatch.setenv("TRTLLM_FP4_MLA_TWOSTREAM_T2_SCALE", "1.2")
    s = _choose(128, 4, 30080)
    assert s.path == dp.PATH_4X and s.r2 == 0
    assert _choose(48, 4, 30080).path == dp.PATH_TWO_STREAM


def test_kv_regime_ceiling():
    assert dp.kv_regime_ceiling(1000, 163840) == 4096
    assert dp.kv_regime_ceiling(8192, 163840) == 8192
    assert dp.kv_regime_ceiling(30080, 163840) == 32768
    assert dp.kv_regime_ceiling(100000, 163840) == 163840
    assert dp.kv_regime_ceiling(30080, 30080) == 30080


def test_kv_regimes_for_graphs(monkeypatch):
    # default floor 12288: one 2x regime ending right under it, every grid point above it, the capacity last
    assert dp.kv_regimes_for_graphs(163840) == (12287, 16384, 32768, 65536, 163840)
    assert dp.kv_regimes_for_graphs(30080) == (12287, 16384, 30080)
    assert dp.kv_regimes_for_graphs(65536) == (12287, 16384, 32768, 65536)
    assert dp.kv_regimes_for_graphs(12288) == (12287, 12288)
    # a capacity under the floor is 2x throughout: a single regime
    assert dp.kv_regimes_for_graphs(8192) == (8192,)
    assert dp.kv_regimes_for_graphs(4000) == (4000,)
    # the floor knob moves the 2x regime; a zero floor keeps the whole grid
    monkeypatch.setenv(dp.MIN_KV_4X_ENV, "40000")
    assert dp.kv_regimes_for_graphs(163840) == (39999, 65536, 163840)
    monkeypatch.setenv(dp.MIN_KV_4X_ENV, "0")
    assert dp.kv_regimes_for_graphs(30080) == (4096, 8192, 12288, 16384, 30080)
    monkeypatch.delenv(dp.MIN_KV_4X_ENV)
    # the grid knob replaces the points above the floor; 0 leaves the 2x regime and the capacity
    monkeypatch.setenv(dp.KV_REGIME_GRID_ENV, "20000,50000")
    assert dp.kv_regimes_for_graphs(163840) == (12287, 20000, 50000, 163840)
    monkeypatch.setenv(dp.KV_REGIME_GRID_ENV, "0")
    assert dp.kv_regimes_for_graphs(163840) == (12287, 163840)
    monkeypatch.delenv(dp.KV_REGIME_GRID_ENV)
    # a forced 2x or 4x path never looks at the KV length; a forced two-stream still plans on it
    monkeypatch.setenv(dp.DECODE_PATH_ENV, "2x")
    assert dp.kv_regimes_for_graphs(163840) == (163840,)
    monkeypatch.setenv(dp.DECODE_PATH_ENV, "4x")
    assert dp.kv_regimes_for_graphs(163840) == (163840,)
    monkeypatch.setenv(dp.DECODE_PATH_ENV, "twostream")
    assert dp.kv_regimes_for_graphs(163840) == (12287, 16384, 32768, 65536, 163840)


def test_kv_regime_for_step():
    ceilings = dp.kv_regimes_for_graphs(163840)
    assert dp.kv_regime_for_step(1, ceilings) == 12287
    assert dp.kv_regime_for_step(12287, ceilings) == 12287
    assert dp.kv_regime_for_step(12288, ceilings) == 16384
    assert dp.kv_regime_for_step(30080, ceilings) == 32768
    assert dp.kv_regime_for_step(100000, ceilings) == 163840
    assert dp.kv_regime_for_step(200000, ceilings) == 163840
    assert dp.kv_regime_for_step(5000, (30080,)) == 30080
    # the graph of a regime decides the way the eager call would for every step in it: 2x for every KV under the
    # floor (the band between the last grid point and the floor included), 4x-eligible from the floor up
    for kv in (1, 5000, 8193, 9000, 12287):
        assert _choose(128, 4, 163840, live=dp.kv_regime_for_step(kv, ceilings)).path == dp.PATH_2X
        assert _choose(128, 4, 163840, live=kv).path == dp.PATH_2X
    for kv in (12288, 16384, 30080):
        assert _choose(128, 4, 163840, live=dp.kv_regime_for_step(kv, ceilings)).path != dp.PATH_2X
        assert _choose(128, 4, 163840, live=kv).path != dp.PATH_2X
