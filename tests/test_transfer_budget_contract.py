from __future__ import annotations

from eligsir.runtime.transfer_budget import (
    CPU_ORIGIN_INSERTION_BUDGET,
    GPU_NATIVE_INSERTION_BUDGET,
    LIVE_SOURCE_FRAME_BUDGET,
    LatencySamples,
    counter_delta,
)


def test_live_frame_budget_is_capped_at_one_packed_h2d_and_zero_d2h():
    assert LIVE_SOURCE_FRAME_BUDGET.evaluate(
        {"h2d_ops": 1, "d2h_ops": 0, "h2d_bytes": 123, "d2h_bytes": 0}
    )[0]
    ok, failures = LIVE_SOURCE_FRAME_BUDGET.evaluate(
        {"h2d_ops": 2, "d2h_ops": 0, "h2d_bytes": 246, "d2h_bytes": 0}
    )
    assert not ok
    assert failures == ("h2d_ops=2>1",)


def test_gpu_native_insertion_regresses_on_any_transfer():
    ok, failures = GPU_NATIVE_INSERTION_BUDGET.evaluate(
        {"h2d_ops": 1, "d2h_ops": 0, "h2d_bytes": 4, "d2h_bytes": 0}
    )
    assert not ok
    assert failures == ("h2d_ops=1>0",)


def test_cpu_origin_insertion_is_capped_at_one_packed_h2d():
    assert CPU_ORIGIN_INSERTION_BUDGET.evaluate({"h2d_ops": 1, "d2h_ops": 0})[0]
    assert not CPU_ORIGIN_INSERTION_BUDGET.evaluate({"h2d_ops": 2, "d2h_ops": 0})[0]


def test_counter_delta_is_boundary_local():
    assert counter_delta(
        {"h2d_ops": 4, "d2h_ops": 2, "h2d_bytes": 40},
        {"h2d_ops": 6, "d2h_ops": 2, "h2d_bytes": 64},
    ) == {"d2h_ops": 0, "h2d_bytes": 24, "h2d_ops": 2}


def test_latency_summary_reports_tail_percentiles():
    samples = LatencySamples()
    samples.extend(range(1, 101))
    summary = samples.summary()
    assert summary["count"] == 100
    assert summary["p50_ms"] == 50.5
    assert summary["p95_ms"] == 95.05
    assert summary["p99_ms"] == 99.01
    assert summary["max_ms"] == 100.0
