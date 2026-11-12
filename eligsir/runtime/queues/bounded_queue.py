"""A bounded FIFO queue with non-blocking offer semantics."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Deque, Generic, TypeVar

from .metrics import QueueDropReason, QueueMetrics

T = TypeVar("T")


class QueueOfferResult(str, Enum):
    """Outcome of a non-blocking queue offer."""

    ENQUEUED = "enqueued"
    DROPPED_QUEUE_FULL = "dropped_queue_full"
    DROPPED_QUEUE_CLOSED = "dropped_queue_closed"
    DROPPED_QUEUE_FAILED = "dropped_queue_failed"


class QueuePollState(str, Enum):
    """State observed by a non-blocking poll."""

    ITEM = "item"
    EMPTY = "empty"
    CLOSED = "closed"
    FAILED = "failed"


@dataclass(frozen=True)
class QueuePollResult(Generic[T]):
    """An item, if one was immediately available to a polling consumer."""

    state: QueuePollState
    item: T | None = None
    error: BaseException | None = None


class BoundedQueue(Generic[T]):
    """A non-blocking single-producer/single-consumer bounded FIFO.

    The producer exclusively calls :meth:`try_offer`, :meth:`close`, and
    :meth:`fail`; the optimizer-side consumer exclusively calls :meth:`try_poll`.
    CPython's deque end operations run atomically under the GIL, so this design avoids
    a mutex that could otherwise make an empty optimizer poll wait for producer work.
    It intentionally is not a multi-producer or multi-consumer queue.
    """

    def __init__(
        self, *, capacity: int, clock_ns: Callable[[], int] = time.monotonic_ns
    ) -> None:
        if capacity < 1:
            raise ValueError("capacity must be at least 1")
        self._capacity = capacity
        self._clock_ns = clock_ns
        self._items: Deque[tuple[T, int]] = deque()
        self._closed = False
        self._failure: BaseException | None = None
        self._enqueued_count = 0
        self._dequeued_count = 0
        self._dropped_count = 0
        self._drop_counts: dict[QueueDropReason, int] = {}

    def try_offer(self, item: T) -> QueueOfferResult:
        """Offer ``item`` without waiting for capacity."""
        if self._failure is not None:
            self._record_drop(QueueDropReason.QUEUE_FAILED)
            return QueueOfferResult.DROPPED_QUEUE_FAILED
        if self._closed:
            self._record_drop(QueueDropReason.QUEUE_CLOSED)
            return QueueOfferResult.DROPPED_QUEUE_CLOSED
        if len(self._items) == self._capacity:
            self._record_drop(QueueDropReason.QUEUE_FULL)
            return QueueOfferResult.DROPPED_QUEUE_FULL
        self._items.append((item, self._clock_ns()))
        self._enqueued_count += 1
        return QueueOfferResult.ENQUEUED

    def close(self) -> None:
        """Refuse future offers while allowing consumers to drain accepted items."""
        self._closed = True

    def fail(self, error: BaseException) -> None:
        """Discard pending items and make a terminal producer failure visible to pollers."""
        if self._failure is not None:
            return
        self._failure = error
        while True:
            try:
                self._items.popleft()
            except IndexError:
                break
            self._record_drop(QueueDropReason.QUEUE_FAILED)

    def try_poll(self) -> QueuePollResult[T]:
        """Return the next FIFO item when available, without waiting for one."""
        if self._failure is not None:
            return QueuePollResult(
                state=QueuePollState.FAILED,
                error=self._failure,
            )
        try:
            item, _ = self._items.popleft()
        except IndexError:
            if self._failure is not None:
                return QueuePollResult(
                    state=QueuePollState.FAILED,
                    error=self._failure,
                )
            state = QueuePollState.CLOSED if self._closed else QueuePollState.EMPTY
            return QueuePollResult(state=state)
        self._dequeued_count += 1
        return QueuePollResult(state=QueuePollState.ITEM, item=item)

    def _record_drop(self, reason: QueueDropReason) -> None:
        self._dropped_count += 1
        self._drop_counts[reason] = self._drop_counts.get(reason, 0) + 1

    def metrics(self) -> QueueMetrics:
        """Return a non-blocking, potentially concurrent telemetry snapshot."""
        depth = len(self._items)
        oldest_item_age_ms = 0.0
        if depth:
            try:
                oldest_enqueued_ns = self._items[0][1]
            except IndexError:
                depth = 0
            else:
                oldest_item_age_ms = max(0, self._clock_ns() - oldest_enqueued_ns) / 1_000_000
        return QueueMetrics(
            capacity=self._capacity,
            depth=depth,
            oldest_item_age_ms=oldest_item_age_ms,
            enqueued_count=self._enqueued_count,
            dequeued_count=self._dequeued_count,
            dropped_count=self._dropped_count,
            drop_counts=dict(self._drop_counts),
        )
