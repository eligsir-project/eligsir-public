"""Unit coverage for the admission-time RGB-D prefetch (issue #279/#1259/#1342).

Real fr3 GPU verification (see docs/architecture/2026-08-31-issue279-admission-prefetch-attempts.md)
confirmed the shipped mechanism never actually schedules a job on the standard causal_replay/tum_rgbd
recipe -- PixelReloadToken is only ever constructed by eligsir/runtime/replay.py, not by that
adapter. This means the mechanism itself was unexercised by that GPU verification; these tests are
what actually exercise it.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np

from eligsir.contracts import SourceIdentity
from eligsir.runtime.production_application import CanonicalMappingRuntime


def _make_runtime(max_entries: int = 8) -> CanonicalMappingRuntime:
    runtime = object.__new__(CanonicalMappingRuntime)
    runtime._admission_prefetch_enabled = True
    runtime._admission_prefetch_requested = True
    runtime._admission_prefetch_reason = "explicit"
    runtime._admission_prefetch_lock = threading.Lock()
    runtime._admission_prefetch_cache = {}
    runtime._admission_prefetch_pending = set()
    runtime._admission_prefetch_protected_frame_ids = set()
    runtime._admission_prefetch_cache_max_entries = max_entries
    runtime._admission_prefetch_executor = ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="test-admission-prefetch"
    )
    runtime._admission_prefetch_scheduled_total = 0
    runtime._admission_prefetch_completed_total = 0
    runtime._admission_prefetch_failed_total = 0
    runtime._admission_prefetch_hits_total = 0
    return runtime


def _fake_decode(token: int) -> tuple[np.ndarray, np.ndarray]:
    rgb = np.full((4, 4, 3), int(token) % 256, dtype=np.uint8)
    depth = np.full((4, 4), float(token), dtype=np.float32)
    return rgb, depth


def test_admission_prefetch_hit_matches_a_fresh_decode(monkeypatch):
    monkeypatch.setattr(
        CanonicalMappingRuntime, "_decode_reload_token", staticmethod(_fake_decode)
    )
    runtime = _make_runtime()
    cold_pixels = SimpleNamespace(rgb=None, depth=None, reload_token=7)

    runtime._maybe_schedule_admission_prefetch(frame_id=1, cold_pixels=cold_pixels)
    runtime._admission_prefetch_executor.shutdown(wait=True)

    assert runtime._admission_prefetch_scheduled_total == 1
    assert runtime._admission_prefetch_completed_total == 1
    assert runtime._admission_prefetch_failed_total == 0

    hit = runtime._pop_admission_prefetch(1)
    assert hit is not None
    rgb, depth = hit
    expected_rgb, expected_depth = _fake_decode(7)
    assert np.array_equal(rgb, expected_rgb)
    assert np.array_equal(depth, expected_depth)
    assert runtime._admission_prefetch_hits_total == 1

    # Consumed exactly once: a second pop for the same frame is a genuine miss,
    # never a stale/duplicate serve.
    assert runtime._pop_admission_prefetch(1) is None
    assert runtime._admission_prefetch_hits_total == 1


def test_admission_prefetch_skips_when_arrays_already_resident():
    runtime = _make_runtime()
    cold_pixels = SimpleNamespace(
        rgb=np.zeros((2, 2, 3), dtype=np.uint8),
        depth=np.zeros((2, 2), dtype=np.float32),
        reload_token=None,
    )

    runtime._maybe_schedule_admission_prefetch(frame_id=2, cold_pixels=cold_pixels)
    runtime._admission_prefetch_executor.shutdown(wait=True)

    # This is the real-world causal_replay/tum_rgbd case: nothing to prefetch,
    # so nothing is scheduled.
    assert runtime._admission_prefetch_scheduled_total == 0
    assert runtime._pop_admission_prefetch(2) is None


def test_admission_prefetch_skips_without_a_reload_token():
    runtime = _make_runtime()
    cold_pixels = SimpleNamespace(rgb=None, depth=None, reload_token=None)

    runtime._maybe_schedule_admission_prefetch(frame_id=3, cold_pixels=cold_pixels)
    runtime._admission_prefetch_executor.shutdown(wait=True)

    assert runtime._admission_prefetch_scheduled_total == 0


def test_admission_prefetch_submits_each_cold_frame_only_once_while_pending():
    """A polling COLD resolver must not queue duplicate decodes behind itself."""

    class _Executor:
        def __init__(self):
            self.submissions = []

        def submit(self, callback, *args):
            self.submissions.append((callback, args))

    runtime = _make_runtime()
    executor = _Executor()
    runtime._admission_prefetch_executor.shutdown(wait=True)
    runtime._admission_prefetch_executor = executor
    cold_pixels = SimpleNamespace(rgb=None, depth=None, reload_token=11)

    for _ in range(3):
        runtime._maybe_schedule_admission_prefetch(frame_id=5, cold_pixels=cold_pixels)

    assert len(executor.submissions) == 1
    assert runtime._admission_prefetch_scheduled_total == 1


def test_admission_prefetch_pop_is_a_no_op_when_uninitialized():
    """A test double built via `object.__new__` (never ran `__init__`) must not raise."""

    runtime = object.__new__(CanonicalMappingRuntime)
    assert runtime._pop_admission_prefetch(1) is None


def test_admission_prefetch_never_raises_on_decode_failure(monkeypatch):
    def _boom(token):
        raise RuntimeError("simulated decode failure")

    monkeypatch.setattr(CanonicalMappingRuntime, "_decode_reload_token", staticmethod(_boom))
    runtime = _make_runtime()
    cold_pixels = SimpleNamespace(rgb=None, depth=None, reload_token=1)

    runtime._maybe_schedule_admission_prefetch(frame_id=4, cold_pixels=cold_pixels)
    runtime._admission_prefetch_executor.shutdown(wait=True)

    assert runtime._admission_prefetch_failed_total == 1
    assert runtime._admission_prefetch_completed_total == 0
    # The miss path remains fully available -- a failed prefetch is invisible
    # to a caller, never an exception surfaced up the tick.
    assert runtime._pop_admission_prefetch(4) is None


def test_admission_prefetch_cache_is_bounded(monkeypatch):
    monkeypatch.setattr(
        CanonicalMappingRuntime, "_decode_reload_token", staticmethod(_fake_decode)
    )
    runtime = _make_runtime(max_entries=2)

    # Submit synchronously (single worker, so these serialize) to make the
    # FIFO-eviction ordering deterministic.
    for frame_id in range(5):
        runtime._maybe_schedule_admission_prefetch(
            frame_id=frame_id,
            cold_pixels=SimpleNamespace(rgb=None, depth=None, reload_token=frame_id),
        )
        runtime._admission_prefetch_executor.shutdown(wait=True)
        runtime._admission_prefetch_executor = ThreadPoolExecutor(max_workers=1)

    assert len(runtime._admission_prefetch_cache) <= 2
    # Only the most recent entries survive; the earliest ones were evicted
    # before ever being consumed -- never served stale, just dropped.
    assert runtime._pop_admission_prefetch(0) is None
    assert runtime._pop_admission_prefetch(4) is not None


def test_exact_prepared_demand_survives_an_unrelated_fifo_completion(monkeypatch):
    """A frozen B4 COLD decode must outlive unrelated speculative prefetch."""
    monkeypatch.setattr(
        CanonicalMappingRuntime, "_decode_reload_token", staticmethod(_fake_decode)
    )
    runtime = _make_runtime(max_entries=1)
    demanded = SourceIdentity("camera", 0, 1, 1_000)
    runtime._historical_frame_ids_by_source = {demanded: 1}
    runtime._set_admission_prefetch_protected_sources(frozenset((demanded,)))

    for frame_id in (1, 2):
        runtime._maybe_schedule_admission_prefetch(
            frame_id=frame_id,
            cold_pixels=SimpleNamespace(rgb=None, depth=None, reload_token=frame_id),
        )
        runtime._admission_prefetch_executor.shutdown(wait=True)
        runtime._admission_prefetch_executor = ThreadPoolExecutor(max_workers=1)

    assert runtime._pop_admission_prefetch(1) is not None
    assert runtime._pop_admission_prefetch(2) is None


def test_telemetry_reports_requested_reason_alongside_effective_enabled():
    """Issue #1543 finding A: the report surface must carry `requested`/
    `reason` next to the existing effective `enabled`/counter fields."""
    runtime = _make_runtime()
    runtime._admission_prefetch_requested = False
    runtime._admission_prefetch_reason = "prepared_b4_required"

    telemetry = runtime.admission_prefetch_telemetry()

    assert telemetry["requested"] is False
    assert telemetry["enabled"] is True
    assert telemetry["reason"] == "prepared_b4_required"
    assert telemetry["scheduled"] == 0
