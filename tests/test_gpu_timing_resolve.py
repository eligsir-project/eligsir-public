"""Tests proving GPU span timing does NOT synchronize per-span (W3-01 P0 fix).

Covers both ``eligsir.runtime.profiling.events.GpuSpanRegistry``/``gpu_timed_span``
and ``eligsir.runtime.profiling.gpu_timing.StageTimer``/``StageTimerRegistry``: spans
recorded before ``resolve_pending()`` must not be readable yet, spans recorded after must be
correct, and ``resolve_pending()`` itself must call ``torch.cuda.synchronize()`` at most once
regardless of how many spans are pending.

Requires a real CUDA-capable device (``torch.cuda.Event`` needs one) -- skipped otherwise,
matching this repo's convention of gating GPU-only tests. Run inside the container:
``docker compose run --rm train python -m pytest tests/test_gpu_timing_resolve.py -q``.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

torch = pytest.importorskip("torch")

from eligsir.runtime.profiling import events as events_module
from eligsir.runtime.profiling.events import GpuSpanRegistry, gpu_timed_span
from eligsir.runtime.profiling.gpu_timing import StageTimer, StageTimerRegistry

_CUDA_AVAILABLE = torch.cuda.is_available()

pytestmark = pytest.mark.skipif(not _CUDA_AVAILABLE, reason="requires a real CUDA/ROCm device")


@pytest.fixture(autouse=True)
def _profiling_enabled():
    prior = events_module.PROFILING_ENABLED
    events_module.set_profiling_enabled(True)
    yield
    events_module.set_profiling_enabled(prior)


def _busy_gpu_work():
    """A small amount of real GPU work so the span isn't instantaneous."""
    a = torch.randn(512, 512, device="cuda")
    b = torch.randn(512, 512, device="cuda")
    for _ in range(20):
        a = a @ b


def test_span_registry_not_resolved_until_resolve_pending_called():
    reg = GpuSpanRegistry()
    with gpu_timed_span("test.span", registry=reg):
        _busy_gpu_work()

    # Before resolve_pending(): nothing accumulated yet, even though the span ended.
    snap_before = reg.snapshot()
    assert snap_before.get("test.span", {"count": 0})["count"] == 0

    resolved = reg.resolve_pending()
    assert resolved == 1

    snap_after = reg.snapshot()
    assert snap_after["test.span"]["count"] == 1
    assert snap_after["test.span"]["total_ms"] >= 0.0


def test_span_registry_resolve_pending_calls_synchronize_at_most_once():
    reg = GpuSpanRegistry()
    for _ in range(5):
        with gpu_timed_span("test.span", registry=reg):
            _busy_gpu_work()

    with patch("torch.cuda.synchronize", wraps=torch.cuda.synchronize) as mock_sync:
        resolved = reg.resolve_pending()
        assert resolved == 5
        assert mock_sync.call_count == 1

    snap = reg.snapshot()
    assert snap["test.span"]["count"] == 5


def test_span_registry_resolve_pending_noop_when_nothing_pending():
    reg = GpuSpanRegistry()
    with patch("torch.cuda.synchronize", wraps=torch.cuda.synchronize) as mock_sync:
        resolved = reg.resolve_pending()
        assert resolved == 0
        mock_sync.assert_not_called()


def test_gpu_timed_span_does_not_synchronize_inside_the_context():
    """The context manager itself (entering/exiting the span) must never synchronize --
    only an explicit resolve_pending() call may."""
    reg = GpuSpanRegistry()
    with patch("torch.cuda.synchronize", wraps=torch.cuda.synchronize) as mock_sync:
        with gpu_timed_span("test.span", registry=reg):
            _busy_gpu_work()
        mock_sync.assert_not_called()


def test_stage_timer_not_resolved_until_resolve_pending_called():
    timer = StageTimer("optimization")
    with timer.span():
        _busy_gpu_work()

    stats_before = timer.stats()
    assert stats_before["count"] == 0

    resolved = timer.resolve_pending()
    assert resolved == 1

    stats_after = timer.stats()
    assert stats_after["count"] == 1


def test_stage_timer_span_does_not_synchronize():
    timer = StageTimer("optimization")
    with patch("torch.cuda.synchronize", wraps=torch.cuda.synchronize) as mock_sync:
        with timer.span():
            _busy_gpu_work()
        mock_sync.assert_not_called()


def test_stage_timer_registry_resolve_pending_single_synchronize_across_timers():
    registry = StageTimerRegistry(stage_names=("optimization", "admission", "mutation"))
    with registry.span("optimization"):
        _busy_gpu_work()
    with registry.span("admission"):
        _busy_gpu_work()
    with registry.span("mutation"):
        _busy_gpu_work()

    # Nothing resolved yet on any timer.
    export_before = registry.export()
    assert all(stats["count"] == 0 for stats in export_before.values())

    with patch("torch.cuda.synchronize", wraps=torch.cuda.synchronize) as mock_sync:
        total = registry.resolve_pending()
        assert total == 3
        assert mock_sync.call_count == 1

    export_after = registry.export()
    assert export_after["optimization"]["count"] == 1
    assert export_after["admission"]["count"] == 1
    assert export_after["mutation"]["count"] == 1
