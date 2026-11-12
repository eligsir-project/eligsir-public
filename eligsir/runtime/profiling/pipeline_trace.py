"""CPU-side monotonic pipeline-stage span/rate tracking.

Terminology is fixed by the Wave 3 plan, section 2.4 ("Compatibility rules") and must not
drift: ``optimization_steps_hz``, ``admitted_keyframes_hz``, ``map_commits_hz``,
``map_age_ms``, ``optimization_step_gpu_ms``.

Rates use a rolling wall-time window (``time.perf_counter()``), not a fixed event-count
window. A fixed-N-event window is a known bug class in this project (see
eligsir/mapping's history of "last-N-events" counters that freeze instead of decaying
to zero when a source stalls) — with a wall-time window, once no new events arrive within
the window, the rate genuinely decays to zero rather than staying frozen at the last
observed value.

Every public method is gated behind ``PROFILING_ENABLED``
(see :mod:`eligsir.runtime.profiling.events`): when disabled, recording methods do
nothing (no deque mutation, no timestamp read) and ``metrics()`` returns a fixed all-zero
snapshot without touching any internal state.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Deque

from eligsir.runtime.profiling.events import is_profiling_enabled

_METRIC_KEYS = (
    "optimization_steps_hz",
    "admitted_keyframes_hz",
    "map_commits_hz",
    "map_age_ms",
    "has_map_commit",
    "optimization_step_gpu_ms",
)


class _RollingRateCounter:
    """Tracks event timestamps within a rolling wall-time window."""

    def __init__(self, window_seconds: float) -> None:
        self._window_seconds = window_seconds
        self._events: Deque[float] = deque()
        #: When this counter was created (or last reset) -- used as a
        #: fallback anchor for the "elapsed since start" denominator below.
        self._start_time = time.perf_counter()
        #: Timestamp of the very first recorded event (never reset by
        #: pruning) -- used to avoid under-reporting the rate during the
        #: startup period before a full window's worth of wall-clock time
        #: has actually elapsed. A real 2Hz rate observed for only 1s of a
        #: 5s window must be divided by 1s, not the full 5s, or it reports
        #: as ~0.4x the true rate.
        self._first_event_time: float | None = None

    def record(self, now: float) -> None:
        if self._first_event_time is None:
            self._first_event_time = now
        self._events.append(now)
        self._prune(now)

    def _prune(self, now: float) -> None:
        cutoff = now - self._window_seconds
        while self._events and self._events[0] < cutoff:
            self._events.popleft()

    def rate_hz(self, now: float) -> float:
        self._prune(now)
        if not self._events:
            return 0.0
        anchor = self._first_event_time if self._first_event_time is not None else self._start_time
        elapsed = now - anchor
        effective_window = min(self._window_seconds, elapsed) if elapsed > 0 else self._window_seconds
        if effective_window <= 0:
            effective_window = self._window_seconds
        return len(self._events) / effective_window

    def reset(self) -> None:
        self._events.clear()
        self._start_time = time.perf_counter()
        self._first_event_time = None


class _RollingValueWindow:
    """Tracks (timestamp, value) pairs within a rolling wall-time window for a mean."""

    def __init__(self, window_seconds: float) -> None:
        self._window_seconds = window_seconds
        self._events: Deque[tuple[float, float]] = deque()

    def record(self, now: float, value: float) -> None:
        self._events.append((now, value))
        self._prune(now)

    def _prune(self, now: float) -> None:
        cutoff = now - self._window_seconds
        while self._events and self._events[0][0] < cutoff:
            self._events.popleft()

    def mean(self, now: float) -> float:
        self._prune(now)
        if not self._events:
            return 0.0
        return sum(v for _, v in self._events) / len(self._events)

    def reset(self) -> None:
        self._events.clear()


class PipelineTrace:
    """Accumulates named pipeline-stage counters/rates over a rolling wall-time window."""

    def __init__(self, window_seconds: float = 5.0) -> None:
        self._window_seconds = window_seconds
        self._lock = threading.Lock()
        self._optimization_steps = _RollingRateCounter(window_seconds)
        self._admitted_keyframes = _RollingRateCounter(window_seconds)
        self._map_commits = _RollingRateCounter(window_seconds)
        self._optimization_step_gpu_ms = _RollingValueWindow(window_seconds)
        self._last_map_commit_time: float | None = None

    # -- recording ---------------------------------------------------------

    def record_optimization_step(self, gpu_ms: float | None = None) -> None:
        if not is_profiling_enabled():
            return
        now = time.perf_counter()
        with self._lock:
            self._optimization_steps.record(now)
            if gpu_ms is not None:
                self._optimization_step_gpu_ms.record(now, gpu_ms)

    def record_admitted_keyframe(self) -> None:
        if not is_profiling_enabled():
            return
        now = time.perf_counter()
        with self._lock:
            self._admitted_keyframes.record(now)

    def record_map_commit(self) -> None:
        if not is_profiling_enabled():
            return
        now = time.perf_counter()
        with self._lock:
            self._map_commits.record(now)
            self._last_map_commit_time = now

    # -- reporting -----------------------------------------------------------

    def metrics(self) -> dict[str, float | bool | None]:
        """Return the current metric snapshot as a plain dict, keyed by the fixed names
        this plan mandates (section 2.4).

        ``map_age_ms`` is ``None`` (not ``float("inf")``) before the first map commit --
        Python's ``json.dumps`` serializes ``inf`` as the non-standard, invalid-JSON token
        ``Infinity``, which would silently corrupt any written experiment report. The
        companion ``has_map_commit`` bool lets callers distinguish "no commit yet"
        (``map_age_ms is None``) from a genuine (if currently improbable) zero age."""
        if not is_profiling_enabled():
            return {
                key: (False if key == "has_map_commit" else 0.0)
                for key in _METRIC_KEYS
            }

        now = time.perf_counter()
        with self._lock:
            has_map_commit = self._last_map_commit_time is not None
            if self._last_map_commit_time is None:
                map_age_ms = None
            else:
                map_age_ms = (now - self._last_map_commit_time) * 1000.0
            return {
                "optimization_steps_hz": self._optimization_steps.rate_hz(now),
                "admitted_keyframes_hz": self._admitted_keyframes.rate_hz(now),
                "map_commits_hz": self._map_commits.rate_hz(now),
                "map_age_ms": map_age_ms,
                "has_map_commit": has_map_commit,
                "optimization_step_gpu_ms": self._optimization_step_gpu_ms.mean(now),
            }

    def reset(self) -> None:
        with self._lock:
            self._optimization_steps.reset()
            self._admitted_keyframes.reset()
            self._map_commits.reset()
            self._optimization_step_gpu_ms.reset()
            self._last_map_commit_time = None
