"""Tests for eligsir.mapping.commit_queue.PendingCommitQueue (PR W3-03)."""

from __future__ import annotations

from eligsir.mapping.commit_queue import PendingCommitQueue
from eligsir.runtime.queues.bounded_queue import QueueOfferResult
from eligsir.runtime.queues.metrics import QueueDropReason


def test_submit_and_drain_round_trip_fifo_order():
    q: PendingCommitQueue[str] = PendingCommitQueue(capacity=4)
    assert q.submit("a") is QueueOfferResult.ENQUEUED
    assert q.submit("b") is QueueOfferResult.ENQUEUED
    assert q.drain() == ["a", "b"]
    assert q.drain() == []


def test_submit_rejects_without_evicting_when_full():
    """A commit must never be silently dropped in favor of a newer one --
    see the module docstring's contrast with the candidate queue's
    latest-wins policy."""
    q: PendingCommitQueue[str] = PendingCommitQueue(capacity=1)
    assert q.submit("first") is QueueOfferResult.ENQUEUED
    assert q.submit("second") is QueueOfferResult.DROPPED_QUEUE_FULL
    assert q.metrics().drop_counts == {QueueDropReason.QUEUE_FULL: 1}
    assert q.drain() == ["first"]


def test_drain_respects_max_items_cap():
    q: PendingCommitQueue[int] = PendingCommitQueue(capacity=4)
    for i in range(4):
        q.submit(i)
    assert q.drain(max_items=2) == [0, 1]
    assert q.drain() == [2, 3]


def test_close_then_submit_is_rejected_but_drain_still_returns_retained_items():
    q: PendingCommitQueue[str] = PendingCommitQueue(capacity=2)
    q.submit("retained")
    q.close()
    assert q.submit("after-close") is QueueOfferResult.DROPPED_QUEUE_CLOSED
    assert q.drain() == ["retained"]


def test_fail_discards_pending_and_future_submits_are_rejected():
    q: PendingCommitQueue[str] = PendingCommitQueue(capacity=2)
    q.submit("lost")
    q.fail(RuntimeError("producer died"))
    assert q.drain() == []
    assert q.submit("after-failure") is QueueOfferResult.DROPPED_QUEUE_FAILED
