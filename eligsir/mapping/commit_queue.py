"""Pending commit queue (plan section 8.4, "Introduce a pending commit
queue").

An accepted :class:`~eligsir.mapping.candidates.contracts.
PreparedFrameCandidate` becomes a :class:`CommitRequest`: everything the
optimizer thread needs to materialize the full-resolution training image
and insert/relocate Gaussians, still without a GPU tensor attached (that
happens at commit time, on the optimizer thread, for an ACCEPTED candidate
only -- see ``candidates/contracts.py``'s docstring on deliverable #3).

The queue exists so accepted-frame map mutations can be batched at
explicit barriers (``drain``) instead of being applied one at a time,
interleaved with arbitrary numerical optimizer steps, the moment each one
arrives -- the failure mode described in plan section 8.4's rationale for
this file. Unlike the candidate-output queue in
``eligsir.runtime.workers.frame_preparation_worker`` (latest-wins,
staleness tolerated), a commit must never be silently evicted: dropping an
admitted frame's map mutation breaks the full-coverage run contract in
CLAUDE.md. ``submit`` therefore uses ``BoundedQueue``'s plain
reject-when-full behavior, and a caller that sees
``QueueOfferResult.DROPPED_QUEUE_FULL`` must treat it as real backpressure
(e.g. widen the drain barrier's ``max_items``, or slow admission) rather
than silently discard the candidate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, List, Optional, TypeVar

from eligsir.mapping.candidates.contracts import PreparedFrameCandidate
from eligsir.runtime.queues.bounded_queue import BoundedQueue, QueueOfferResult, QueuePollState
from eligsir.runtime.queues.metrics import QueueMetrics

T = TypeVar("T")


@dataclass(frozen=True)
class CommitRequest:
    """One accepted candidate, queued for a batched map mutation."""

    candidate: PreparedFrameCandidate
    slam_map_id: int


class PendingCommitQueue(Generic[T]):
    """Thin, typed wrapper over :class:`BoundedQueue` for accepted-frame
    commits. Not itself a new queue primitive -- composes
    ``BoundedQueue.try_offer``/``try_poll`` (Subagent A's contract)
    directly, adding only the barrier-drain convenience this file's
    rationale calls for.
    """

    def __init__(self, *, capacity: int) -> None:
        self._queue: BoundedQueue[T] = BoundedQueue(capacity=capacity)

    def submit(self, item: T) -> QueueOfferResult:
        """Non-blocking offer. See module docstring: a
        ``DROPPED_QUEUE_FULL`` here is real backpressure, not something to
        paper over with an eviction policy."""
        return self._queue.try_offer(item)

    def drain(self, *, max_items: Optional[int] = None) -> List[T]:
        """Non-blocking full (or capped) drain, for a caller applying a
        batch at an explicit barrier (e.g. once per mapping cycle). Stops
        early on an empty/closed/failed poll; does not raise on failure --
        callers that need to observe a producer failure should also poll
        ``metrics()``/call ``try_poll`` directly if they need the error."""
        drained: List[T] = []
        while max_items is None or len(drained) < max_items:
            polled = self._queue.try_poll()
            if polled.state is not QueuePollState.ITEM:
                break
            drained.append(polled.item)  # type: ignore[arg-type]
        return drained

    def close(self) -> None:
        self._queue.close()

    def fail(self, error: BaseException) -> None:
        self._queue.fail(error)

    def metrics(self) -> QueueMetrics:
        return self._queue.metrics()
