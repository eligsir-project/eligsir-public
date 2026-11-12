"""Tests for eligsir.runtime.profiling.pipeline_trace.PipelineTrace."""

import time

import pytest

from eligsir.runtime.profiling import events as events_module
from eligsir.runtime.profiling.pipeline_trace import PipelineTrace


@pytest.fixture(autouse=True)
def _profiling_enabled():
    prior = events_module.PROFILING_ENABLED
    events_module.set_profiling_enabled(True)
    yield
    events_module.set_profiling_enabled(prior)


def test_metric_keys_are_fixed():
    trace = PipelineTrace(window_seconds=1.0)
    metrics = trace.metrics()
    assert set(metrics.keys()) == {
        "optimization_steps_hz",
        "admitted_keyframes_hz",
        "map_commits_hz",
        "map_age_ms",
        "has_map_commit",
        "optimization_step_gpu_ms",
    }


def test_rates_positive_after_recording():
    trace = PipelineTrace(window_seconds=1.0)
    for _ in range(10):
        trace.record_optimization_step(gpu_ms=5.0)
    trace.record_admitted_keyframe()
    trace.record_map_commit()

    metrics = trace.metrics()
    assert metrics["optimization_steps_hz"] > 0.0
    assert metrics["admitted_keyframes_hz"] > 0.0
    assert metrics["map_commits_hz"] > 0.0
    assert metrics["optimization_step_gpu_ms"] == pytest.approx(5.0)
    assert metrics["map_age_ms"] >= 0.0


def test_rate_decays_to_zero_after_window_elapses():
    """Regression guard for the last-N-events window bug class noted in the plan/memory:
    a stalled source must decay the rate to zero, not freeze it at the last value."""
    window_seconds = 0.2
    trace = PipelineTrace(window_seconds=window_seconds)

    for _ in range(20):
        trace.record_optimization_step()

    assert trace.metrics()["optimization_steps_hz"] > 0.0

    # Let the whole rolling window elapse with no new events.
    time.sleep(window_seconds * 3)

    assert trace.metrics()["optimization_steps_hz"] == 0.0
    assert trace.metrics()["admitted_keyframes_hz"] == 0.0
    assert trace.metrics()["map_commits_hz"] == 0.0


def test_map_age_grows_since_last_commit():
    trace = PipelineTrace(window_seconds=5.0)
    trace.record_map_commit()
    age_immediately = trace.metrics()["map_age_ms"]
    time.sleep(0.05)
    age_later = trace.metrics()["map_age_ms"]
    assert age_later > age_immediately


def test_map_age_is_none_before_any_commit():
    """map_age_ms must be JSON-safe (None, not float('inf'), which json.dumps would
    serialize as the invalid-JSON token `Infinity`)."""
    trace = PipelineTrace(window_seconds=5.0)
    metrics = trace.metrics()
    assert metrics["map_age_ms"] is None
    assert metrics["has_map_commit"] is False

    import json

    # Must round-trip through strict JSON (allow_nan=False) without raising.
    json.dumps(metrics, allow_nan=False)


def test_has_map_commit_true_after_first_commit():
    trace = PipelineTrace(window_seconds=5.0)
    trace.record_map_commit()
    metrics = trace.metrics()
    assert metrics["has_map_commit"] is True
    assert metrics["map_age_ms"] is not None
    assert metrics["map_age_ms"] >= 0.0


def test_rate_hz_not_underreported_during_startup():
    """Regression guard: a real 2Hz rate observed for only a fraction of the rolling
    window must not be divided by the full window_seconds (which would under-report
    it), see review Fix 3."""
    trace = PipelineTrace(window_seconds=5.0)
    period = 1.0 / 30.0
    accept_every = 15  # 30fps / 15 = 2Hz accepted
    t0 = time.perf_counter()
    frame_count = 90  # 3 seconds @ 30fps
    for i in range(frame_count):
        target = t0 + i * period
        remaining = target - time.perf_counter()
        if remaining > 0:
            time.sleep(remaining)
        if i % accept_every == 0:
            trace.record_admitted_keyframe()
    rate = trace.metrics()["admitted_keyframes_hz"]
    assert rate == pytest.approx(2.0, abs=0.5)


def test_profiling_disabled_returns_zeroed_metrics_and_skips_recording():
    events_module.set_profiling_enabled(False)
    trace = PipelineTrace(window_seconds=1.0)
    for _ in range(5):
        trace.record_optimization_step(gpu_ms=5.0)
    trace.record_admitted_keyframe()
    trace.record_map_commit()

    metrics = trace.metrics()
    assert metrics == {
        "optimization_steps_hz": 0.0,
        "admitted_keyframes_hz": 0.0,
        "map_commits_hz": 0.0,
        "map_age_ms": 0.0,
        "has_map_commit": False,
        "optimization_step_gpu_ms": 0.0,
    }
    # Internal state must genuinely be untouched, not just masked at report time.
    assert len(trace._optimization_steps._events) == 0
    assert trace._last_map_commit_time is None
