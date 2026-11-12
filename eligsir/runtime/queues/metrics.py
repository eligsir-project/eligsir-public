"""Snapshots for bounded runtime-queue observability."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class QueueDropReason(str, Enum):
    """Reasons an offered item was not retained by a runtime queue."""

    QUEUE_FULL = "queue_full"
    QUEUE_CLOSED = "queue_closed"
    QUEUE_FAILED = "queue_failed"


@dataclass(frozen=True)
class QueueMetrics:
    """A point-in-time view of a bounded queue's state."""

    capacity: int
    depth: int
    oldest_item_age_ms: float
    enqueued_count: int
    dequeued_count: int
    dropped_count: int
    drop_counts: dict[QueueDropReason, int]
