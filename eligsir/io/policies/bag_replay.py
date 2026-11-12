"""Lossless bag-replay source policy (plan section 9.4, "Add bag-replay
source policy").

Bag replay is a **dev-only evaluation convenience** (see CLAUDE.md's HARD
CONSTRAINT section), but while it is used, every frame in the bag must be
preserved -- CLAUDE.md's "Full trajectory coverage" run contract already
requires every source frame to be ingested at least once, and a silently
dropped frame during replay would violate that contract invisibly. This
module is the SLAM-worker-facing enforcement of that requirement at the
queue boundary introduced by this PR: offering a frame into a bounded
:class:`~eligsir.runtime.queues.bounded_queue.BoundedQueue` when it is
temporarily full must never simply discard the frame the way
``BoundedQueue.try_offer`` alone would.

Backpressure, durable bag references, or spill are all acceptable per the
plan; silent drop is not. This implementation combines the first two: a
non-blocking bounded backpressure retry window, falling back to an
in-memory spill deque (a "durable bag reference" -- the frame is retained,
not decoded away, until it can be delivered) when the queue stays full
past that window. Only if the spill store itself is exhausted (a
configurable hard cap, guarding against unbounded memory growth on a
truly stuck consumer) does a frame actually get lost -- and that is
treated as a hard integrity failure: :attr:`BagReplayIntegrityReport
.mapping_result_valid` flips to ``False`` for the rest of the run,
matching the standing of ``eval_valid == False``/``SUSPECTED_TRUNCATED``/
``map_reinit_invalidated`` already documented in CLAUDE.md's Run
Contracts. This module does not itself write into ``report.json`` --
wiring :meth:`BagReplayIntegrityReport.as_geometry_fields` into the real
per-run report (``utils/partition_eval.py``'s
``compute_trajectory_coverage``, alongside ``stream_end_reason``/
``map_reinit_invalidated``) is a follow-up integration task, deliberately
scoped out of this PR to avoid duplicating or contradicting that existing
machinery without a dedicated, tested change to it.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Deque, Generic, Optional, TypeVar

from eligsir.runtime.queues.bounded_queue import BoundedQueue, QueueOfferResult

T = TypeVar("T")


class BagOfferOutcome(str, Enum):
    """Outcome of a single :meth:`BagReplayPolicy.offer` call."""

    ENQUEUED_DIRECT = "enqueued_direct"
    ENQUEUED_AFTER_BACKPRESSURE = "enqueued_after_backpressure"
    SPILLED = "spilled"
    LOST_SPILL_EXHAUSTED = "lost_spill_exhausted"


@dataclass
class BagReplayIntegrityReport:
    """Cumulative lossless-replay integrity tracker.

    ``mapping_result_valid`` starts ``True`` and is a one-way latch: once
    a real loss happens it stays ``False`` for the rest of the run (a
    lost frame earlier in the bag can silently invalidate coverage
    conclusions drawn from data ingested after it, so there is no safe
    point at which to "recover" validity).
    """

    frames_offered: int = 0
    frames_enqueued_direct: int = 0
    frames_enqueued_after_backpressure: int = 0
    frames_spilled: int = 0
    frames_lost: int = 0
    loss_events: list = field(default_factory=list)

    @property
    def mapping_result_valid(self) -> bool:
        return self.frames_lost == 0

    @property
    def frames_delivered(self) -> int:
        """Frames accepted losslessly by :meth:`BagReplayPolicy.offer` --
        i.e. every offered frame that is now either resident in the queue
        or retained in the spill deque awaiting delivery, excluding only
        the ``LOST_SPILL_EXHAUSTED`` ones that were genuinely lost.

        Issue #861 (2026-08-16): this is the ONLY correct notion of "how
        long is this stream" for a consumer that indexes the queue's own
        store. ``len(queue_store)`` alone undercounts by exactly the
        current spill depth, and a consumer that freezes a stream length
        from it at end-of-stream silently truncates the run (a spilled
        frame is delivered later, by ``flush_spill``, not never)."""
        return (
            self.frames_enqueued_direct
            + self.frames_enqueued_after_backpressure
            + self.frames_spilled
        )

    def record(self, outcome: BagOfferOutcome, *, detail: Optional[str] = None) -> None:
        self.frames_offered += 1
        if outcome is BagOfferOutcome.ENQUEUED_DIRECT:
            self.frames_enqueued_direct += 1
        elif outcome is BagOfferOutcome.ENQUEUED_AFTER_BACKPRESSURE:
            self.frames_enqueued_after_backpressure += 1
        elif outcome is BagOfferOutcome.SPILLED:
            self.frames_spilled += 1
        elif outcome is BagOfferOutcome.LOST_SPILL_EXHAUSTED:
            self.frames_lost += 1
            self.loss_events.append(
                {"reason": "spill_exhausted", "detail": detail, "at_offer_index": self.frames_offered}
            )

    def as_geometry_fields(self) -> dict:
        """Fields shaped to slot into ``report.json["trajectory_coverage"]``
        (top-level, despite this method's legacy name -- issue #504)
        alongside the existing
        ``stream_end_reason``/``map_reinit_invalidated`` fields
        (see module docstring) -- not wired in automatically by this PR."""
        return {
            "mapping_result_valid": self.mapping_result_valid,
            "bag_frames_offered": self.frames_offered,
            "bag_frames_spilled": self.frames_spilled,
            "bag_frames_lost": self.frames_lost,
            "bag_loss_events": list(self.loss_events),
        }


class BagReplayPolicy(Generic[T]):
    """Lossless producer-side policy over a :class:`BoundedQueue`.

    Args:
        queue: The bounded queue frames are ultimately delivered onto.
        backpressure_max_wait_s: Total time :meth:`offer` will
            busy-retry ``try_offer`` before falling back to spill. ``0``
            disables the backpressure phase (spill immediately on a full
            queue).
        backpressure_poll_interval_s: Sleep between backpressure retries.
        max_spill: Hard cap on the in-memory spill deque. Exceeding this
            is the only way :meth:`offer` reports a real loss -- pick a
            cap generous enough that a real bag's consumer stall never
            hits it in practice (memory cost is one frame's numpy buffers
            per spilled item, so this should track available RAM, not
            bag length).
    """

    def __init__(
        self,
        queue: BoundedQueue[T],
        *,
        backpressure_max_wait_s: float = 0.25,
        backpressure_poll_interval_s: float = 0.005,
        max_spill: int = 4096,
    ) -> None:
        self._queue = queue
        self._backpressure_max_wait_s = backpressure_max_wait_s
        self._backpressure_poll_interval_s = backpressure_poll_interval_s
        self._max_spill = max_spill
        self._spill: Deque[T] = deque()
        # Issue #861 (2026-08-16): the spill DRAIN is the one operation two
        # threads can genuinely run at once -- the producer's own
        # ``offer()`` (which drains first) and the consumer's
        # ``flush_spill()`` (called from release_upto, and from get_frame's
        # block-wait loop). Its "peek head -> try_offer -> popleft" sequence
        # is not atomic: two concurrent drains could both peek the same head
        # and both enqueue it (a duplicated frame) while popping two
        # distinct entries (an undelivered one). Guard the drain only --
        # deliberately NOT the sleeping backpressure window, so a consumer's
        # flush_spill/release_upto is never blocked for the backpressure
        # budget. Lock order is always this lock -> the queue's own lock
        # (via try_offer); nothing takes them the other way round.
        self._drain_lock = threading.Lock()
        self.report = BagReplayIntegrityReport()

    def offer(self, item: T) -> BagOfferOutcome:
        """Deliver ``item`` losslessly: drain any existing spill first (in
        FIFO order, so a temporarily-stuck-then-recovered consumer still
        sees frames in bag order), then attempt this item."""
        self._drain_spill_into_queue()

        if not self._spill:
            result = self._queue.try_offer(item)
            if result is QueueOfferResult.ENQUEUED:
                outcome = BagOfferOutcome.ENQUEUED_DIRECT
                self.report.record(outcome)
                return outcome

            # Queue full (or closed/failed) -- try short backpressure retry.
            outcome = self._try_backpressure(item)
            if outcome is not None:
                self.report.record(outcome)
                return outcome

        # Still can't deliver: spill (durable in-memory reference).
        if len(self._spill) < self._max_spill:
            self._spill.append(item)
            outcome = BagOfferOutcome.SPILLED
            self.report.record(outcome)
            return outcome

        outcome = BagOfferOutcome.LOST_SPILL_EXHAUSTED
        self.report.record(outcome, detail=f"spill cap {self._max_spill} exceeded")
        return outcome

    def flush_spill(self, *, max_items: Optional[int] = None) -> int:
        """Manually drain as much spill as currently fits in the queue.
        Returns the number of items moved. Safe to call from a caller's
        own idle loop to proactively shrink the spill backlog."""
        return self._drain_spill_into_queue(max_items=max_items)

    @property
    def spill_depth(self) -> int:
        return len(self._spill)

    # -- internal -------------------------------------------------------

    def _try_backpressure(self, item: T) -> Optional[BagOfferOutcome]:
        if self._backpressure_max_wait_s <= 0:
            return None
        deadline = time.monotonic() + self._backpressure_max_wait_s
        while time.monotonic() < deadline:
            time.sleep(self._backpressure_poll_interval_s)
            result = self._queue.try_offer(item)
            if result is QueueOfferResult.ENQUEUED:
                return BagOfferOutcome.ENQUEUED_AFTER_BACKPRESSURE
            if result in (QueueOfferResult.DROPPED_QUEUE_CLOSED, QueueOfferResult.DROPPED_QUEUE_FAILED):
                return None
        return None

    def _drain_spill_into_queue(self, *, max_items: Optional[int] = None) -> int:
        moved = 0
        with self._drain_lock:
            while self._spill and (max_items is None or moved < max_items):
                head = self._spill[0]
                result = self._queue.try_offer(head)
                if result is not QueueOfferResult.ENQUEUED:
                    break
                self._spill.popleft()
                moved += 1
        return moved
