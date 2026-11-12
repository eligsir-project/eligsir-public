"""Live-camera freshness source policy (plan section 9.4, "Add live
freshness policy").

Live sensing has the opposite correctness requirement from bag replay
(see ``bag_replay.py``): under sustained overload, a stale frame is
strictly less useful than the newest one (staleness compounds -- a queue
is only ever saturated because the consumer is behind, so the oldest
queued item is also always the least current view of the live scene).
This mirrors the reasoning already documented for the candidate-output
queue in ``eligsir.runtime.queues.policies``'s module docstring; this
module applies the same "latest wins" eviction at the SLAM-submission
boundary, but additionally makes every drop **reason-coded and counted**
(deliverable 5), which the generic ``offer_latest_wins`` helper alone does
not: it returns only the evicted item, not *why* eviction was necessary.

Bounded operation under overload: this policy never grows unboundedly and
never blocks -- the queue capacity is the hard ceiling, so a runaway
producer degrades to "always evict the previous frame" rather than
consuming unbounded memory or stalling the live camera's own ingestion
thread.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Generic, Optional, TypeVar

from eligsir.runtime.queues.bounded_queue import BoundedQueue, QueueOfferResult

T = TypeVar("T")


class LiveDropReason(str, Enum):
    """Why a live frame was dropped rather than delivered."""

    SUPERSEDED_BY_FRESHER_FRAME = "superseded_by_fresher_frame"
    QUEUE_UNAVAILABLE = "queue_unavailable"  # closed/failed queue


@dataclass
class LiveFreshnessReport:
    """Cumulative reason-coded drop counters (deliverable 5)."""

    frames_offered: int = 0
    frames_enqueued: int = 0
    drop_counts: dict = field(default_factory=dict)
    last_drop_reason: Optional[LiveDropReason] = None
    last_drop_at_s: Optional[float] = None

    def record_enqueued(self) -> None:
        self.frames_offered += 1
        self.frames_enqueued += 1

    def record_drop(self, reason: LiveDropReason) -> None:
        self.frames_offered += 1
        self.note_eviction(reason)

    def note_eviction(self, reason: LiveDropReason) -> None:
        """Record a reason-coded eviction/drop without counting it against
        ``frames_offered`` -- used when the eviction is a side effect of
        successfully admitting a *different* (the newly offered) item, so
        the offered item itself is not what got dropped."""
        self.drop_counts[reason] = self.drop_counts.get(reason, 0) + 1
        self.last_drop_reason = reason
        self.last_drop_at_s = time.monotonic()

    def as_dict(self) -> dict:
        return {
            "frames_offered": self.frames_offered,
            "frames_enqueued": self.frames_enqueued,
            "drop_counts": {k.value: v for k, v in self.drop_counts.items()},
            "last_drop_reason": self.last_drop_reason.value if self.last_drop_reason else None,
        }


class LiveFreshnessPolicy(Generic[T]):
    """Producer-side "latest wins" policy over a :class:`BoundedQueue`,
    with reason-coded drop accounting.

    Args:
        queue: The bounded queue frames are submitted onto.
    """

    def __init__(self, queue: BoundedQueue[T]) -> None:
        self._queue = queue
        self.report = LiveFreshnessReport()

    def offer(self, item: T) -> LiveDropReason | None:
        """Offer ``item``; evict the oldest queued item first if full.
        Returns the drop reason if ``item`` itself ultimately could not be
        delivered (queue closed/failed even after evicting), else
        ``None`` (delivered, whether or not an older item was evicted to
        make room)."""
        result = self._queue.try_offer(item)
        if result is QueueOfferResult.ENQUEUED:
            self.report.record_enqueued()
            return None

        if result is QueueOfferResult.DROPPED_QUEUE_FULL:
            polled = self._queue.try_poll()
            if polled.state.value == "item":
                # An older frame was evicted to make room -- reason-coded,
                # not a silent drop. This does not count against the
                # *offered* item's own accounting (it is delivered below).
                self.report.note_eviction(LiveDropReason.SUPERSEDED_BY_FRESHER_FRAME)
            retried = self._queue.try_offer(item)
            if retried is QueueOfferResult.ENQUEUED:
                self.report.record_enqueued()
                return None
            self.report.record_drop(LiveDropReason.QUEUE_UNAVAILABLE)
            return LiveDropReason.QUEUE_UNAVAILABLE

        # Closed/failed queue: item itself could not be admitted at all.
        self.report.record_drop(LiveDropReason.QUEUE_UNAVAILABLE)
        return LiveDropReason.QUEUE_UNAVAILABLE
