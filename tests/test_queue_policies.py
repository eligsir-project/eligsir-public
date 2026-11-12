"""Tests for eligsir.runtime.queues.policies (PR W3-03)."""

from __future__ import annotations

from eligsir.runtime.queues.bounded_queue import BoundedQueue, QueueOfferResult
from eligsir.runtime.queues.metrics import QueueMetrics
from eligsir.runtime.queues.policies import (
    BackpressurePolicyConfig,
    QueuePressureLevel,
    offer_latest_wins,
    pressure_level,
)


def _metrics(capacity: int, depth: int) -> QueueMetrics:
    return QueueMetrics(
        capacity=capacity, depth=depth, oldest_item_age_ms=0.0,
        enqueued_count=depth, dequeued_count=0, dropped_count=0, drop_counts={},
    )


def test_pressure_level_ok_below_high_watermark():
    assert pressure_level(_metrics(10, 5)) is QueuePressureLevel.OK


def test_pressure_level_high_at_watermark():
    cfg = BackpressurePolicyConfig(high_watermark=0.75, critical_watermark=1.0)
    assert pressure_level(_metrics(4, 3), cfg) is QueuePressureLevel.HIGH


def test_pressure_level_critical_when_full():
    assert pressure_level(_metrics(4, 4)) is QueuePressureLevel.CRITICAL


def test_offer_latest_wins_enqueues_directly_when_not_full():
    q = BoundedQueue[str](capacity=2)
    result = offer_latest_wins(q, "a")
    assert result.result is QueueOfferResult.ENQUEUED
    assert result.evicted is None
    assert q.metrics().depth == 1


def test_offer_latest_wins_evicts_oldest_when_full():
    q = BoundedQueue[str](capacity=1)
    q.try_offer("stale")
    result = offer_latest_wins(q, "fresh")
    assert result.result is QueueOfferResult.ENQUEUED
    assert result.evicted == "stale"
    polled = q.try_poll()
    assert polled.item == "fresh"


def test_offer_latest_wins_never_grows_queue_beyond_capacity():
    q = BoundedQueue[int](capacity=3)
    for i in range(10):
        offer_latest_wins(q, i)
        assert q.metrics().depth <= 3
