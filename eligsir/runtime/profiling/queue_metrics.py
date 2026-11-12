"""Bounded-queue observability wrapper.

Plan section 2.2 (program-wide runtime rule): "Every queue is bounded and exposes queue
depth, oldest-item age, drops, and drop reason." Depth/capacity/drop-count bookkeeping is
cheap integer arithmetic and stays on unconditionally — it's needed for basic runtime
correctness checks (e.g. detecting silent drops), not just profiling.

The one genuinely per-item allocation this wrapper could do — recording an enqueue
timestamp per item, to compute oldest-item age — is gated behind ``PROFILING_ENABLED``
(see :mod:`eligsir.runtime.profiling.events`) so the profiler-off path still does no
timestamp bookkeeping beyond what ``QueueMetrics`` itself is asked to report. When
profiling is off, ``oldest_item_age_ms`` reports 0.0 rather than tracking real ages.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Deque

from eligsir.runtime.profiling.events import is_profiling_enabled


class QueueMetrics:
    """Wraps observability bookkeeping for a bounded FIFO queue.

    This class does not own the queue's storage — callers still push/pop their own
    container. Call :meth:`on_enqueue` / :meth:`on_dequeue` / :meth:`on_drop` at the
    corresponding call sites to keep the metrics in sync. Assumes FIFO ordering (the
    oldest-enqueued item is always the next dequeued one), matching every bounded queue
    this plan introduces (``eligsir/runtime/queues/bounded_queue.py``).
    """

    def __init__(self, max_depth: int = 0) -> None:
        self.max_depth = max_depth
        self._lock = threading.Lock()
        self._depth = 0
        self._peak_depth = 0
        self._enqueue_times: Deque[float] = deque()
        self._cumulative_drops = 0
        self._last_drop_reason: str | None = None

    def on_enqueue(self) -> None:
        with self._lock:
            self._depth += 1
            self._peak_depth = max(self._peak_depth, self._depth)
            if is_profiling_enabled():
                self._enqueue_times.append(time.perf_counter())

    def on_dequeue(self) -> None:
        with self._lock:
            if self._depth > 0:
                self._depth -= 1
            if self._enqueue_times:
                self._enqueue_times.popleft()

    def on_drop(self, reason: str) -> None:
        with self._lock:
            self._cumulative_drops += 1
            self._last_drop_reason = reason

    @property
    def depth(self) -> int:
        with self._lock:
            return self._depth

    @property
    def cumulative_drops(self) -> int:
        with self._lock:
            return self._cumulative_drops

    @property
    def last_drop_reason(self) -> str | None:
        with self._lock:
            return self._last_drop_reason

    def oldest_item_age_ms(self) -> float:
        if not is_profiling_enabled():
            return 0.0
        with self._lock:
            if not self._enqueue_times:
                return 0.0
            return (time.perf_counter() - self._enqueue_times[0]) * 1000.0

    def snapshot(self) -> dict[str, object]:
        return {
            "depth": self.depth,
            "capacity": self.max_depth,
            "max_depth": self.max_depth,
            "peak_depth": self._peak_depth,
            "oldest_item_age_ms": self.oldest_item_age_ms(),
            "cumulative_drops": self.cumulative_drops,
            "last_drop_reason": self.last_drop_reason,
        }

    def reset(self) -> None:
        with self._lock:
            self._depth = 0
            self._peak_depth = 0
            self._enqueue_times.clear()
            self._cumulative_drops = 0
            self._last_drop_reason = None
