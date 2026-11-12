"""Contract tests for the bounded runtime queue used by the Wave 3 path."""

from __future__ import annotations

import importlib
import threading


def test_full_queue_rejects_without_growing_and_records_reason_coded_drop():
    """Removing the capacity check would silently turn source pressure into memory growth."""
    queues = importlib.import_module("eligsir.runtime.queues")
    queue = queues.BoundedQueue[str](capacity=1)

    assert queue.try_offer("first") is queues.QueueOfferResult.ENQUEUED
    assert queue.try_offer("second") is queues.QueueOfferResult.DROPPED_QUEUE_FULL

    metrics = queue.metrics()
    assert metrics.capacity == 1
    assert metrics.depth == 1
    assert metrics.dropped_count == 1
    assert metrics.drop_counts == {queues.QueueDropReason.QUEUE_FULL: 1}


def test_poll_is_non_blocking_fifo_and_reports_oldest_item_age():
    """Replacing FIFO timestamps with a last-in timestamp hides a growing backlog."""
    from eligsir.runtime.queues import BoundedQueue, QueuePollState

    now_ns = 1_000_000_000

    def clock_ns() -> int:
        return now_ns

    queue = BoundedQueue[str](capacity=2, clock_ns=clock_ns)
    queue.try_offer("oldest")
    now_ns += 7_000_000
    queue.try_offer("newest")

    assert queue.metrics().oldest_item_age_ms == 7.0

    first = queue.try_poll()
    assert first.state is QueuePollState.ITEM
    assert first.item == "oldest"
    assert queue.metrics().oldest_item_age_ms == 0.0

    second = queue.try_poll()
    assert second.state is QueuePollState.ITEM
    assert second.item == "newest"

    assert queue.try_poll().state is QueuePollState.EMPTY


def test_close_drains_retained_items_then_reports_closed_and_counts_new_offer():
    """Closing before a consumer drains must not discard an already accepted candidate."""
    from eligsir.runtime.queues import (
        BoundedQueue,
        QueueDropReason,
        QueueOfferResult,
        QueuePollState,
    )

    queue = BoundedQueue[str](capacity=2)
    assert queue.try_offer("retained") is QueueOfferResult.ENQUEUED

    queue.close()

    assert queue.try_offer("after-close") is QueueOfferResult.DROPPED_QUEUE_CLOSED
    assert queue.try_poll().item == "retained"
    assert queue.try_poll().state is QueuePollState.CLOSED
    assert queue.metrics().drop_counts == {QueueDropReason.QUEUE_CLOSED: 1}


def test_failure_discards_backlog_and_reports_terminal_cause_without_waiting():
    """A failed producer must not leave consumers polling an unrecoverable backlog forever."""
    from eligsir.runtime.queues import (
        BoundedQueue,
        QueueDropReason,
        QueueOfferResult,
        QueuePollState,
    )

    queue = BoundedQueue[str](capacity=3)
    queue.try_offer("lost-1")
    queue.try_offer("lost-2")
    cause = RuntimeError("SLAM worker exited")

    queue.fail(cause)

    terminal = queue.try_poll()
    assert terminal.state is QueuePollState.FAILED
    assert terminal.error is cause
    assert queue.try_offer("after-failure") is QueueOfferResult.DROPPED_QUEUE_FAILED

    metrics = queue.metrics()
    assert metrics.depth == 0
    assert metrics.dropped_count == 3
    assert metrics.drop_counts == {QueueDropReason.QUEUE_FAILED: 3}


def test_metrics_count_each_retained_item_at_its_enqueue_and_dequeue_boundary():
    """Counting only depth loses the throughput evidence needed to diagnose a stalled stage."""
    from eligsir.runtime.queues import BoundedQueue

    queue = BoundedQueue[int](capacity=2)
    queue.try_offer(1)
    queue.try_offer(2)
    queue.try_poll()

    metrics = queue.metrics()
    assert metrics.enqueued_count == 2
    assert metrics.dequeued_count == 1
    assert metrics.depth == 1


def test_poll_returns_while_producer_is_mid_offer():
    """A contended producer must not make an optimizer-side poll wait for its lock."""
    from eligsir.runtime.queues import BoundedQueue, QueuePollState

    offer_entered = threading.Event()
    release_offer = threading.Event()
    poll_done = threading.Event()
    polled = []

    def blocking_clock_ns() -> int:
        offer_entered.set()
        assert release_offer.wait(timeout=1.0)
        return 0

    queue = BoundedQueue[str](capacity=1, clock_ns=blocking_clock_ns)
    offer_thread = threading.Thread(target=lambda: queue.try_offer("candidate"))
    offer_thread.start()
    assert offer_entered.wait(timeout=1.0)

    def poll() -> None:
        polled.append(queue.try_poll())
        poll_done.set()

    poll_thread = threading.Thread(target=poll)
    poll_thread.start()
    try:
        assert poll_done.wait(timeout=0.1), "try_poll waited on producer contention"
        assert polled[0].state is QueuePollState.EMPTY
    finally:
        release_offer.set()
        offer_thread.join(timeout=1.0)
        poll_thread.join(timeout=1.0)
