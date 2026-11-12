"""Named GPU "stage timer" helpers built on top of ``events.py`` primitives.

Combines :func:`eligsir.runtime.profiling.events.gpu_timed_span` with a rolling window
of recent durations per named stage (e.g. ``"optimization"``, ``"admission"``,
``"mutation"``), reporting min/mean/p50/p90/p99/max — exportable to a plain dict for JSON
serialization (``report.json``-style output, see the plan's ``report.py``).

Gated behind ``PROFILING_ENABLED`` exactly like the rest of this package: with profiling
off, :meth:`StageTimer.span` is a bare no-op context manager and :meth:`StageTimer.stats`
returns zeros without touching any window state.

**Why there is no per-span ``torch.cuda.synchronize()``**: see
``eligsir.runtime.profiling.events``'s module docstring -- the same reasoning applies
here verbatim. :meth:`StageTimer.span` records start/end CUDA events without blocking on
them; the elapsed time is only computed once :meth:`StageTimer.resolve_pending` (or the
owning :class:`StageTimerRegistry`'s :meth:`StageTimerRegistry.resolve_pending`) is called,
which performs at most one ``torch.cuda.synchronize()`` for all timers combined. Call this
at report/finalization time, never inside the per-frame/per-iteration hot loop.
"""

from __future__ import annotations

import threading
from collections import deque
from contextlib import contextmanager
from typing import Deque, Iterator

from eligsir.runtime.profiling.events import is_profiling_enabled

_PERCENTILES = (0.50, 0.90, 0.99)


def _percentile(sorted_values: list[float], p: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    idx = p * (len(sorted_values) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = idx - lo
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * frac


class StageTimer:
    """Rolling-window GPU stage timer, keyed by a fixed stage name."""

    def __init__(self, name: str, window_size: int = 256) -> None:
        self.name = name
        self._window_size = window_size
        self._lock = threading.Lock()
        self._durations_ms: Deque[float] = deque(maxlen=window_size)
        self._pending: list[tuple] = []

    @contextmanager
    def span(self) -> Iterator[None]:
        if not is_profiling_enabled():
            yield
            return

        import torch  # local import: keep torch out of the disabled fast path entirely

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        try:
            yield
        finally:
            end.record()
            with self._lock:
                self._pending.append((start, end))

    def _drain_pending(self) -> list:
        """Pop this timer's pending event pairs without synchronizing or resolving them."""
        with self._lock:
            pending = self._pending
            self._pending = []
        return pending

    def _absorb_resolved(self, pending: list) -> None:
        """Given event pairs whose completion has already been guaranteed by a prior
        synchronize (elsewhere), read their elapsed time and append to the window."""
        with self._lock:
            for start, end in pending:
                self._durations_ms.append(start.elapsed_time(end))

    def resolve_pending(self) -> int:
        """Resolve this timer's own pending CUDA event pairs into the rolling window.

        Performs AT MOST ONE ``torch.cuda.synchronize()`` call (skipped if nothing is
        pending). See module docstring for why this is deferred out of the hot loop. Prefer
        :meth:`StageTimerRegistry.resolve_pending` when resolving multiple timers together --
        it performs a single synchronize for all of them instead of one per timer.
        """
        pending = self._drain_pending()
        if not pending:
            return 0

        import torch

        torch.cuda.synchronize()
        self._absorb_resolved(pending)
        return len(pending)

    def record_ms(self, duration_ms: float) -> None:
        if not is_profiling_enabled():
            return
        with self._lock:
            self._durations_ms.append(duration_ms)

    def stats(self) -> dict[str, float]:
        if not is_profiling_enabled():
            return {
                "count": 0,
                "min_ms": 0.0,
                "mean_ms": 0.0,
                "p50_ms": 0.0,
                "p90_ms": 0.0,
                "p99_ms": 0.0,
                "max_ms": 0.0,
            }
        with self._lock:
            values = sorted(self._durations_ms)
        if not values:
            return {
                "count": 0,
                "min_ms": 0.0,
                "mean_ms": 0.0,
                "p50_ms": 0.0,
                "p90_ms": 0.0,
                "p99_ms": 0.0,
                "max_ms": 0.0,
            }
        return {
            "count": len(values),
            "min_ms": values[0],
            "mean_ms": sum(values) / len(values),
            "p50_ms": _percentile(values, 0.50),
            "p90_ms": _percentile(values, 0.90),
            "p99_ms": _percentile(values, 0.99),
            "max_ms": values[-1],
        }

    def reset(self) -> None:
        with self._lock:
            self._durations_ms.clear()
            self._pending.clear()


class StageTimerRegistry:
    """Owns a fixed set of named :class:`StageTimer` instances (e.g. optimization,
    admission, mutation) and exports all of them as one JSON-serializable dict."""

    def __init__(self, stage_names: tuple[str, ...] = ("optimization", "admission", "mutation")) -> None:
        self._timers: dict[str, StageTimer] = {name: StageTimer(name) for name in stage_names}

    def timer(self, name: str) -> StageTimer:
        if name not in self._timers:
            self._timers[name] = StageTimer(name)
        return self._timers[name]

    def resolve_pending(self) -> int:
        """Resolve every timer's pending events with a single synchronize call.

        Drains every timer's pending event pairs first, issues AT MOST ONE
        ``torch.cuda.synchronize()`` for the whole registry (skipped if nothing anywhere is
        pending), then absorbs each timer's now-safe-to-read events -- see module docstring.
        """
        per_timer_pending = {name: timer._drain_pending() for name, timer in self._timers.items()}
        total = sum(len(p) for p in per_timer_pending.values())
        if total == 0:
            return 0

        import torch

        torch.cuda.synchronize()
        for name, pending in per_timer_pending.items():
            self._timers[name]._absorb_resolved(pending)
        return total

    @contextmanager
    def span(self, name: str) -> Iterator[None]:
        with self.timer(name).span():
            yield

    def export(self) -> dict[str, dict[str, float]]:
        return {name: timer.stats() for name, timer in self._timers.items()}

    def reset(self) -> None:
        for timer in self._timers.values():
            timer.reset()
