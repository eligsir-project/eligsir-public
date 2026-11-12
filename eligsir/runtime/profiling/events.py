"""GPU event timing primitives.

This repo's whole GPU backend is ROCm, reached transparently through the CUDA-compatible
``torch.cuda`` API (see e.g. ``train.py``'s ``torch.cuda.Event(enable_timing=True)`` usage) —
no HIP-specific wrapper is needed here.

``PROFILING_ENABLED`` gates every real code path in this module. When disabled (the default),
:func:`gpu_timed_span` and :class:`GpuSpanRegistry` do no CUDA event allocation and no
dict/list mutation beyond a cheap counter increment — this is the plan's "profiler-off path
shows less than 1% overhead" acceptance gate (section 6.6).

**Why there is no per-span ``torch.cuda.synchronize()``** (hard constraint, found in W3-01
review): a synchronize call forces the host to block until every GPU work item queued so far
has completed. Doing that after *every single measured span* serializes the entire GPU
pipeline — it destroys exactly the inter-stream/inter-stage overlap that later Wave 3 work
(W3-03 concurrent admission/optimization streams, W3-12 CUDA graph capture) exists to
introduce and measure. A profiler that syncs per span would not be measuring the real,
overlapped workload; it would be measuring a strictly-serial workload the profiler itself
imposed. Instead, :meth:`GpuSpanRegistry.record` only *records* a pair of already-``.record()``-ed
``torch.cuda.Event`` objects (start/end) without reading their elapsed time — reading
``Event.elapsed_time()`` is what actually requires the end event to have completed, so that
read is deferred to :meth:`GpuSpanRegistry.resolve_pending`, which performs **at most one**
``torch.cuda.synchronize()`` call and then resolves every pending pair recorded since the last
resolve. Callers must invoke ``resolve_pending()`` explicitly at report/finalization time
(after all frames in a benchmark scenario have been processed), never inside the hot loop —
see ``eligsir/optimization/gaussian/contracts.py`` for this repo's convention of
documenting non-obvious constraints like this one directly in the docstring.
"""

from __future__ import annotations

import contextlib
import os
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterator


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


#: Module-level profiling switch. Defaults to off; set ``SPLAT_PROFILE_RUNTIME=1`` to enable.
#: Read once at import for a cheap default, but re-checkable/overridable via
#: :func:`set_profiling_enabled` (used by tests and by callers who toggle at runtime).
PROFILING_ENABLED: bool = _env_flag("SPLAT_PROFILE_RUNTIME", default=False)


def set_profiling_enabled(enabled: bool) -> None:
    """Override the module-level profiling flag (primarily for tests)."""
    global PROFILING_ENABLED
    PROFILING_ENABLED = bool(enabled)


def is_profiling_enabled() -> bool:
    """Check if GPU event profiling is currently enabled.

    Returns:
        True if profiling is active, False otherwise.
    """
    return PROFILING_ENABLED


@dataclass
class _SpanStat:
    count: int = 0
    total_ms: float = 0.0


class GpuSpanRegistry:
    """Accumulates named GPU span durations (ms) across many calls.

    Cheap to construct and update when profiling is disabled: :meth:`record_pending` is a
    no-op early return, no allocation.

    Timings are NOT available immediately after a span ends -- see the module docstring for
    why. :meth:`record_pending` only stores the (name, start-event, end-event) triple;
    :meth:`resolve_pending` must be called (at report/finalization time) to turn pending
    event pairs into accumulated ``_SpanStat`` entries. :meth:`snapshot` only reports
    already-resolved stats -- it never resolves pending events itself, so callers who forget
    to call ``resolve_pending()`` will observe an under-reported (or all-zero) snapshot rather
    than a silently-blocking one.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._stats: dict[str, _SpanStat] = defaultdict(_SpanStat)
        self._pending: list[tuple[str, "torch.cuda.Event", "torch.cuda.Event"]] = []  # type: ignore[name-defined]

    def record_pending(self, name: str, start_event, end_event) -> None:
        """Queue a recorded (but not-yet-elapsed) CUDA event pair for later resolution.

        Does not call ``.elapsed_time()`` or synchronize -- that is deferred to
        :meth:`resolve_pending` so that recording a span never blocks the GPU pipeline.
        """
        if not PROFILING_ENABLED:
            return
        with self._lock:
            self._pending.append((name, start_event, end_event))

    def resolve_pending(self) -> int:
        """Resolve every pending event pair into accumulated stats.

        Performs AT MOST ONE ``torch.cuda.synchronize()`` call (skipped entirely if there is
        nothing pending), then reads ``elapsed_time()`` for each pending pair -- safe to do
        without a synchronize per pair once the single synchronize above has guaranteed every
        queued GPU op, including every pending end-event's ``.record()``, has completed.

        Returns the number of spans resolved.
        """
        with self._lock:
            pending = self._pending
            self._pending = []
        if not pending:
            return 0

        import torch  # local import: avoid importing torch at all on the disabled fast path

        torch.cuda.synchronize()
        with self._lock:
            for name, start_event, end_event in pending:
                duration_ms = start_event.elapsed_time(end_event)
                stat = self._stats[name]
                stat.count += 1
                stat.total_ms += duration_ms
        return len(pending)

    def poll_completed(self) -> int:
        """Resolve only event pairs whose end event is already complete.

        Unlike :meth:`resolve_pending`, this never synchronizes.  It is for
        low-rate observers that can use a completed value but must preserve a
        null measurement while the GPU stage is still in flight.
        """
        with self._lock:
            pending = self._pending
            self._pending = []
        completed = []
        unresolved = []
        for item in pending:
            name, start_event, end_event = item
            if bool(end_event.query()):
                completed.append(item)
            else:
                unresolved.append(item)
        with self._lock:
            for name, start_event, end_event in completed:
                duration_ms = start_event.elapsed_time(end_event)
                stat = self._stats[name]
                stat.count += 1
                stat.total_ms += duration_ms
            self._pending = unresolved + self._pending
        return len(completed)

    def record(self, name: str, duration_ms: float) -> None:
        """Directly accumulate an already-known duration (bypassing the pending-event path).

        Used by callers that already have a resolved ``duration_ms`` (e.g. tests, or a
        non-CUDA-event timing source) rather than a pair of CUDA events.
        """
        if not PROFILING_ENABLED:
            return
        with self._lock:
            stat = self._stats[name]
            stat.count += 1
            stat.total_ms += duration_ms

    def snapshot(self) -> dict[str, dict[str, float]]:
        """Export accumulated (already-resolved) stats as a plain, JSON-serializable dict.

        Does NOT resolve pending events -- call :meth:`resolve_pending` first if there may be
        unresolved spans (e.g. at report/finalization time).
        """
        with self._lock:
            out = {}
            for name, stat in self._stats.items():
                mean = stat.total_ms / stat.count if stat.count else 0.0
                out[name] = {
                    "count": stat.count,
                    "total_ms": stat.total_ms,
                    "mean_ms": mean,
                }
            return out

    def reset(self) -> None:
        """Clear all accumulated stats and pending event pairs.

        Useful for resetting the registry between benchmark scenarios or test runs.
        """
        with self._lock:
            self._stats.clear()
            self._pending.clear()


#: Default process-wide registry. Modules that just want "accumulate this span somewhere"
#: without owning their own registry can import and use this directly.
default_registry = GpuSpanRegistry()


@contextlib.contextmanager
def gpu_timed_span(
    name: str,
    registry: GpuSpanRegistry | None = None,
) -> Iterator[None]:
    """Context manager recording a named GPU span into ``registry``.

    When :data:`PROFILING_ENABLED` is False, this is a bare no-op ``yield`` — no
    ``torch.cuda.Event`` is constructed, no synchronization happens, no registry is touched.

    When enabled, this records start/end CUDA events around the span WITHOUT synchronizing
    or reading their elapsed time — see the module docstring for why. The span's duration is
    only available in ``registry``'s stats after a caller explicitly calls
    ``registry.resolve_pending()``.
    """
    if not PROFILING_ENABLED:
        yield
        return

    import torch  # local import: avoid importing torch at all on the disabled fast path

    reg = registry if registry is not None else default_registry
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    try:
        yield
    finally:
        end.record()
        reg.record_pending(name, start, end)
