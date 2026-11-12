"""Admission/backpressure policies layered on top of :class:`BoundedQueue`
(plan section 8.4, "Add bounded candidate and commit queues" /
"Introduce non-blocking queue polling").

``BoundedQueue`` itself only knows one backpressure behavior: reject the
new item when full (``DROPPED_QUEUE_FULL``). That is the right default for
a commit queue -- an accepted frame's map mutation must never be silently
discarded in favor of a newer one, or full-coverage (CLAUDE.md's Run
Contracts) breaks. It is the WRONG default for a candidate queue feeding
admission scoring on a live stream, where a stale candidate is strictly
less useful than the newest one once the queue is saturated (staleness
compounds: a queue is only ever saturated when the consumer is behind,
so the oldest item is also always the least current view of the scene).

These policies never wait and never touch ``BoundedQueue`` internals --
each is just a small, explicit protocol built from ``try_offer``/
``try_poll``, so this module composes with Subagent A's queue without
reimplementing it.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Generic, TypeVar

from .bounded_queue import BoundedQueue, QueueOfferResult
from .metrics import QueueDropReason, QueueMetrics

T = TypeVar("T")


class QueuePressureLevel(str, Enum):
    """Coarse classification of a queue's current fullness, for callers
    that want to react to backpressure (e.g. the preparation worker
    slowing down, or the prefilter tightening its thresholds) without
    hand-rolling a depth/capacity ratio at every call site."""

    OK = "ok"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass(frozen=True)
class BackpressurePolicyConfig:
    """Fullness ratios (``depth / capacity``) at which
    :func:`pressure_level` reports HIGH/CRITICAL."""

    high_watermark: float = 0.75
    critical_watermark: float = 1.0


def pressure_level(
    metrics: QueueMetrics, config: BackpressurePolicyConfig | None = None
) -> QueuePressureLevel:
    """Classify ``metrics`` into a :class:`QueuePressureLevel`."""
    cfg = config or BackpressurePolicyConfig()
    if metrics.capacity <= 0:
        return QueuePressureLevel.OK
    ratio = metrics.depth / metrics.capacity
    if ratio >= cfg.critical_watermark:
        return QueuePressureLevel.CRITICAL
    if ratio >= cfg.high_watermark:
        return QueuePressureLevel.HIGH
    return QueuePressureLevel.OK


@dataclass(frozen=True)
class LatestWinsOfferResult(Generic[T]):
    """Outcome of :func:`offer_latest_wins`: the underlying
    :class:`QueueOfferResult`, plus the evicted item (if any) so a caller
    can attribute the eviction to a specific stale candidate rather than a
    bare drop count."""

    result: QueueOfferResult
    evicted: T | None = None


def offer_latest_wins(queue: BoundedQueue[T], item: T) -> LatestWinsOfferResult[T]:
    """Offer ``item``; if the queue is full, evict the oldest item first
    (via a single non-blocking ``try_poll``) and retry once.

    Appropriate for a candidate-output queue feeding a live admission
    consumer that only ever cares about the freshest evidence (see module
    docstring). Never used for a commit queue -- accepted work must never
    be silently evicted, only ``BoundedQueue``'s default reject-newest
    behavior (``try_offer`` directly) is safe there.

    Non-blocking: at most one ``try_poll`` plus one ``try_offer``, no
    waiting, so this is safe to call from a producer thread without risking
    contention with the consumer's own ``try_poll`` (see ``BoundedQueue``'s
    single-producer/single-consumer contract).
    """
    result = queue.try_offer(item)
    if result is not QueueOfferResult.DROPPED_QUEUE_FULL:
        return LatestWinsOfferResult(result=result, evicted=None)

    polled = queue.try_poll()
    evicted = polled.item if polled.state.value == "item" else None
    retried = queue.try_offer(item)
    return LatestWinsOfferResult(result=retried, evicted=evicted)


__all__ = [
    "BackpressurePolicyConfig",
    "LatestWinsOfferResult",
    "QueueDropReason",
    "QueuePressureLevel",
    "offer_latest_wins",
    "pressure_level",
]
