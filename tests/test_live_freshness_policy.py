"""Tests for LiveFreshnessPolicy (PR W3-04): reason-coded, counted drops
under live-camera overload. Covers deliverable 5."""

from __future__ import annotations

from eligsir.io.policies.live_freshness import LiveDropReason, LiveFreshnessPolicy
from eligsir.runtime.queues.bounded_queue import BoundedQueue


def test_enqueue_when_room_available():
    q = BoundedQueue(capacity=4)
    policy = LiveFreshnessPolicy(q)
    result = policy.offer("frame-0")
    assert result is None
    assert policy.report.frames_enqueued == 1
    assert policy.report.drop_counts == {}


def test_full_queue_evicts_oldest_and_reason_codes_the_drop():
    q = BoundedQueue(capacity=1)
    policy = LiveFreshnessPolicy(q)
    policy.offer("frame-0")
    result = policy.offer("frame-1")  # queue full -> evict frame-0, admit frame-1

    assert result is None  # frame-1 itself was delivered
    assert policy.report.drop_counts.get(LiveDropReason.SUPERSEDED_BY_FRESHER_FRAME) == 1
    assert q.try_poll().item == "frame-1"  # newest wins


def test_bounded_operation_under_sustained_overload():
    q = BoundedQueue(capacity=2)
    policy = LiveFreshnessPolicy(q)
    for i in range(50):
        policy.offer(f"frame-{i}")

    # Never grows past capacity, regardless of overload duration.
    assert q.metrics().depth <= 2
    assert policy.report.frames_offered == 50
    assert policy.report.drop_counts[LiveDropReason.SUPERSEDED_BY_FRESHER_FRAME] == 48


def test_closed_queue_reason_codes_as_queue_unavailable():
    q = BoundedQueue(capacity=2)
    q.close()
    policy = LiveFreshnessPolicy(q)
    result = policy.offer("frame-0")
    assert result is LiveDropReason.QUEUE_UNAVAILABLE
    assert policy.report.drop_counts[LiveDropReason.QUEUE_UNAVAILABLE] == 1


def test_as_dict_serializes_reason_values():
    q = BoundedQueue(capacity=1)
    policy = LiveFreshnessPolicy(q)
    policy.offer("frame-0")
    policy.offer("frame-1")
    d = policy.report.as_dict()
    assert d["last_drop_reason"] == LiveDropReason.SUPERSEDED_BY_FRESHER_FRAME.value
    assert "superseded_by_fresher_frame" in d["drop_counts"]
