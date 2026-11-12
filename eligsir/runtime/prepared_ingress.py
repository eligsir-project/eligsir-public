"""Bounded accepted-source handoff for prepared mapping ingress.

The handoff is deliberately transport-neutral. ``MappingSession`` is the
single producer; a final-item sink may durably commit semantic state before the
item crosses to the asynchronous preparation consumer.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
import numbers
import threading
from types import MappingProxyType
from typing import Callable, Deque, Mapping, Optional

from eligsir.contracts.control import ObservationDisposition
from eligsir.contracts.runtime import SourceIdentity, SensorPacket, TrackingObservation


@dataclass(frozen=True, slots=True)
class PreparedIngressControlToken:
    sequence: int
    source: SourceIdentity
    snapshot_id: int | None
    disposition: Optional[ObservationDisposition]

    def __post_init__(self) -> None:
        if isinstance(self.sequence, bool) or not isinstance(self.sequence, numbers.Integral):
            raise TypeError("prepared control sequence must be an integer")
        if int(self.sequence) < 0:
            raise ValueError("prepared control sequence must be non-negative")
        if not isinstance(self.source, SourceIdentity):
            raise TypeError("prepared control source must be SourceIdentity")
        if self.snapshot_id is not None:
            if isinstance(self.snapshot_id, bool) or not isinstance(self.snapshot_id, numbers.Integral):
                raise TypeError("prepared control snapshot_id must be an integer or None")
            if int(self.snapshot_id) < 0:
                raise ValueError("prepared control snapshot_id must be non-negative")
            object.__setattr__(self, "snapshot_id", int(self.snapshot_id))
        if self.disposition is not None:
            if not isinstance(self.disposition, ObservationDisposition):
                raise TypeError("prepared control disposition must be ObservationDisposition")
            if self.disposition.source != self.source:
                raise ValueError("prepared control disposition and source identities differ")
            if self.snapshot_id != self.disposition.snapshot_id:
                raise ValueError("prepared control snapshot_id must match disposition")
        object.__setattr__(self, "sequence", int(self.sequence))


@dataclass(frozen=True, slots=True)
class PreparedIngressItem:
    sequence: int
    packet: SensorPacket
    observation: TrackingObservation
    disposition: Optional[ObservationDisposition] = None
    control_token: Optional[PreparedIngressControlToken] = None

    def __post_init__(self) -> None:
        if isinstance(self.sequence, bool) or not isinstance(self.sequence, numbers.Integral):
            raise TypeError("prepared ingress sequence must be an integer")
        if int(self.sequence) < 0:
            raise ValueError("prepared ingress sequence must be non-negative")
        if not isinstance(self.packet, SensorPacket):
            raise TypeError("prepared ingress packet must be SensorPacket")
        if not isinstance(self.observation, TrackingObservation):
            raise TypeError("prepared ingress observation must be TrackingObservation")
        if self.packet.source != self.observation.source:
            raise ValueError("prepared ingress packet and observation identities differ")
        if self.disposition is not None:
            if not isinstance(self.disposition, ObservationDisposition):
                raise TypeError("prepared ingress disposition must be ObservationDisposition")
            if self.disposition.source != self.packet.source:
                raise ValueError("prepared ingress disposition and packet identities differ")
        if self.control_token is not None:
            if not isinstance(self.control_token, PreparedIngressControlToken):
                raise TypeError("prepared ingress control_token must be PreparedIngressControlToken")
            if self.control_token.sequence != int(self.sequence):
                raise ValueError("prepared ingress control token sequence differs")
            if self.control_token.source != self.packet.source:
                raise ValueError("prepared ingress control token and packet identities differ")
            if self.control_token.disposition is not self.disposition:
                raise ValueError("prepared ingress control token must retain the item disposition")
        object.__setattr__(self, "sequence", int(self.sequence))


class PreparedIngressOfferResult(str, Enum):
    ENQUEUED = "enqueued"
    DROPPED_QUEUE_FULL = "queue_full"
    DROPPED_DUPLICATE_SOURCE = "duplicate_source"
    DROPPED_OUT_OF_ORDER = "out_of_order"
    DROPPED_QUEUE_CLOSED = "queue_closed"


class PreparedIngressDeferred(RuntimeError):
    """A source remains pending because retention cannot commit safely."""


@dataclass(frozen=True, slots=True)
class PreparedIngressMetrics:
    capacity: int
    depth: int
    enqueued_count: int
    dequeued_count: int
    dropped_count: int
    dropped_by_reason: Mapping[str, int]


class PreparedIngressReservation:
    def __init__(
        self,
        handoff: "PreparedSourceIngressHandoff",
        producer_owner,
        item: PreparedIngressItem,
    ) -> None:
        self._handoff = handoff
        self._producer_owner = producer_owner
        self._item = item
        self._resolved = False

    @property
    def item(self) -> PreparedIngressItem:
        """Return the reserved ingress item."""
        return self._item

    def commit(self, item: PreparedIngressItem) -> PreparedIngressOfferResult:
        """Commit the reserved item to the handoff with optional replacement."""
        if self._resolved:
            raise RuntimeError("prepared ingress reservation is already resolved")
        result = self._handoff._commit_reservation(self, self._producer_owner, item)
        self._resolved = True
        return result

    def cancel(self) -> None:
        """Cancel the reservation without committing the item."""
        if self._resolved:
            return
        self._handoff._cancel_reservation(self, self._producer_owner)
        self._resolved = True


class PreparedSourceIngressHandoff:
    """Exclusive ordered SPSC handoff with an optional producer-side commit sink."""

    def __init__(
        self,
        *,
        capacity: int,
        commit_sink: Callable[[PreparedIngressItem], object] | None = None,
        admission_preflight: Callable[[SensorPacket], bool] | None = None,
    ) -> None:
        if isinstance(capacity, bool) or not isinstance(capacity, numbers.Integral):
            raise TypeError("capacity must be a positive integer")
        capacity = int(capacity)
        if capacity < 1:
            raise ValueError("capacity must be a positive integer")
        if commit_sink is not None and not callable(commit_sink):
            raise TypeError("commit_sink must be callable or None")
        if admission_preflight is not None and not callable(admission_preflight):
            raise TypeError("admission_preflight must be callable or None")
        self._capacity = capacity
        self._commit_sink = commit_sink
        self._admission_preflight = admission_preflight
        self._items: Deque[PreparedIngressItem] = deque()
        self._reservation: PreparedIngressReservation | None = None
        self._producer_owner = None
        self._consumer_owner = None
        self._closed = False
        self._last_sequence: int | None = None
        self._last_source = None
        self._enqueued_count = 0
        self._dequeued_count = 0
        self._dropped_count = 0
        self._drop_counts: dict[str, int] = {}
        self._progress_callback: Callable[[], None] | None = None

    @staticmethod
    def _same_owner(left, right) -> bool:
        """Test ownership equality, returning False on exception."""
        if left is right:
            return True
        try:
            return bool(left == right)
        except Exception:
            return False

    def _claim(self, role: str, owner=None):
        """Claim exclusive ownership of a role and return its token."""
        token = threading.get_ident() if owner is None else owner
        attribute = "_producer_owner" if role == "producer" else "_consumer_owner"
        current = getattr(self, attribute)
        if current is None:
            setattr(self, attribute, token)
        elif not self._same_owner(current, token):
            raise RuntimeError(f"prepared ingress {role} ownership is exclusive")
        return token

    def claim_producer(self, owner=None) -> "PreparedIngressProducer":
        """Claim the only producer role and return its operation handle."""
        return PreparedIngressProducer(self, self._claim("producer", owner))

    def claim_consumer(self, owner=None) -> "PreparedIngressConsumer":
        """Claim the only consumer role and return its operation handle."""
        return PreparedIngressConsumer(self, self._claim("consumer", owner))

    def set_progress_callback(self, callback: Callable[[], None] | None) -> None:
        """Register a callback to notify when items are committed to the handoff.

        The callback is responsible for incrementing progress generation and notifying
        waiters under the appropriate lock. This enables event-driven wakeup of the
        prep worker without exposing internal synchronization primitives.
        """
        if callback is not None and not callable(callback):
            raise TypeError("progress_callback must be callable or None")
        self._progress_callback = callback

    def _drop(self, result: PreparedIngressOfferResult) -> PreparedIngressOfferResult:
        """Record a drop event and update drop counters."""
        self._dropped_count += 1
        reason = result.value
        self._drop_counts[reason] = self._drop_counts.get(reason, 0) + 1
        return result

    def _offer_status(self, item: PreparedIngressItem) -> PreparedIngressOfferResult | None:
        """Check whether an item can be admitted or classify the drop reason."""
        if self._closed:
            return PreparedIngressOfferResult.DROPPED_QUEUE_CLOSED
        if self._reservation is not None:
            return PreparedIngressOfferResult.DROPPED_QUEUE_FULL
        if self._last_sequence is not None and item.sequence <= self._last_sequence:
            return (
                PreparedIngressOfferResult.DROPPED_DUPLICATE_SOURCE
                if item.packet.source == self._last_source
                else PreparedIngressOfferResult.DROPPED_OUT_OF_ORDER
            )
        if any(existing.packet.source == item.packet.source for existing in self._items):
            return PreparedIngressOfferResult.DROPPED_DUPLICATE_SOURCE
        if self._last_source is not None and item.packet.source.ordering_key() <= self._last_source.ordering_key():
            return PreparedIngressOfferResult.DROPPED_OUT_OF_ORDER
        if len(self._items) >= self._capacity:
            return PreparedIngressOfferResult.DROPPED_QUEUE_FULL
        return None

    def preflight_retention(self, packet: SensorPacket) -> bool:
        """Check packet retention before tracking or any semantic side effect."""
        if not isinstance(packet, SensorPacket):
            raise TypeError("prepared ingress retention preflight requires SensorPacket")
        return self._admission_preflight is None or bool(self._admission_preflight(packet))

    def _commit(self, item: PreparedIngressItem) -> None:
        """Commit an item to the handoff queue and invoke callbacks."""
        if self._commit_sink is not None:
            committed = self._commit_sink(item)
            if isinstance(committed, PreparedIngressItem):
                if (
                    committed.sequence != item.sequence
                    or committed.packet.source != item.packet.source
                    or committed.observation is not item.observation
                    or committed.disposition is not item.disposition
                    or committed.control_token is not item.control_token
                ):
                    raise ValueError("prepared ingress commit sink changed frozen control state")
                item = committed
        self._items.append(item)
        self._last_sequence = item.sequence
        self._last_source = item.packet.source
        self._enqueued_count += 1
        if self._progress_callback is not None:
            self._progress_callback()

    def _reserve(self, producer_owner, item: PreparedIngressItem) -> (
        PreparedIngressReservation | PreparedIngressOfferResult
    ):
        """Reserve space for an item before commitment."""
        self._claim("producer", producer_owner)
        if not isinstance(item, PreparedIngressItem):
            raise TypeError("prepared ingress item must be PreparedIngressItem")
        status = self._offer_status(item)
        if status is not None:
            return self._drop(status)
        reservation = PreparedIngressReservation(self, producer_owner, item)
        self._reservation = reservation
        return reservation

    def reserve(
        self, item: PreparedIngressItem
    ) -> PreparedIngressReservation | PreparedIngressOfferResult:
        """Reserve space for an item before commitment on calling thread."""
        return self._reserve(threading.get_ident(), item)

    def _commit_reservation(
        self,
        reservation: PreparedIngressReservation,
        producer_owner,
        item: PreparedIngressItem,
    ) -> PreparedIngressOfferResult:
        """Commit a reserved item to the handoff queue."""
        self._claim("producer", producer_owner)
        if self._reservation is not reservation:
            raise RuntimeError("prepared ingress reservation is not active")
        if not isinstance(item, PreparedIngressItem):
            raise TypeError("prepared ingress item must be PreparedIngressItem")
        reserved = reservation.item
        if item.sequence != reserved.sequence or item.packet.source != reserved.packet.source:
            raise ValueError("prepared ingress commit must retain reserved source and sequence")
        self._commit(item)
        self._reservation = None
        return PreparedIngressOfferResult.ENQUEUED

    def _cancel_reservation(
        self,
        reservation: PreparedIngressReservation,
        producer_owner,
    ) -> None:
        """Cancel a reservation without committing the item."""
        self._claim("producer", producer_owner)
        if self._reservation is not reservation:
            raise RuntimeError("prepared ingress reservation is not active")
        self._reservation = None

    def can_accept(self, item: PreparedIngressItem) -> PreparedIngressOfferResult:
        """Check whether an item can be accepted without actually enqueuing."""
        self._claim("producer", threading.get_ident())
        if not isinstance(item, PreparedIngressItem):
            raise TypeError("prepared ingress item must be PreparedIngressItem")
        status = self._offer_status(item)
        return PreparedIngressOfferResult.ENQUEUED if status is None else self._drop(status)

    def _offer(self, producer_owner, item: PreparedIngressItem) -> PreparedIngressOfferResult:
        """Attempt to enqueue an item on the calling thread."""
        self._claim("producer", producer_owner)
        if not isinstance(item, PreparedIngressItem):
            raise TypeError("prepared ingress item must be PreparedIngressItem")
        status = self._offer_status(item)
        if status is not None:
            return self._drop(status)
        self._commit(item)
        return PreparedIngressOfferResult.ENQUEUED

    def try_offer(self, item: PreparedIngressItem) -> PreparedIngressOfferResult:
        """Attempt to enqueue an item on the calling thread."""
        return self._offer(threading.get_ident(), item)

    offer = try_offer

    def _poll(self, consumer_owner) -> PreparedIngressItem | None:
        """Dequeue one pending item or return None if empty."""
        self._claim("consumer", consumer_owner)
        try:
            item = self._items.popleft()
        except IndexError:
            return None
        self._dequeued_count += 1
        return item

    def try_poll(self) -> PreparedIngressItem | None:
        """Dequeue one pending item on the calling thread or return None."""
        return self._poll(threading.get_ident())

    poll = try_poll

    def _close(self, producer_owner) -> None:
        """Close the handoff to new items."""
        self._claim("producer", producer_owner)
        self._closed = True

    def close(self) -> None:
        """Close the handoff to new items on the calling thread."""
        self._close(threading.get_ident())

    @property
    def capacity(self) -> int:
        """Return the fixed queue capacity."""
        return self._capacity

    @property
    def has_retention_preflight(self) -> bool:
        """Whether this handoff can defer a source before semantic commit."""
        return self._admission_preflight is not None

    @property
    def depth(self) -> int:
        """Return current queue depth including reserved items."""
        return len(self._items) + int(self._reservation is not None)

    @property
    def dropped_by_reason(self) -> Mapping[str, int]:
        """Return drop counts mapped by reason string."""
        return MappingProxyType(dict(self._drop_counts))

    def metrics(self) -> PreparedIngressMetrics:
        """Return a snapshot of all metrics."""
        return PreparedIngressMetrics(
            capacity=self._capacity,
            depth=self.depth,
            enqueued_count=self._enqueued_count,
            dequeued_count=self._dequeued_count,
            dropped_count=self._dropped_count,
            dropped_by_reason=MappingProxyType(dict(self._drop_counts)),
        )


class PreparedIngressProducer:
    """Exclusive producer handle."""

    def __init__(self, handoff: PreparedSourceIngressHandoff, token) -> None:
        self._handoff = handoff
        self._token = token

    def try_offer(self, item: PreparedIngressItem) -> PreparedIngressOfferResult:
        """Attempt to enqueue an item."""
        return self._handoff._offer(self._token, item)

    offer = try_offer

    def reserve(
        self, item: PreparedIngressItem
    ) -> PreparedIngressReservation | PreparedIngressOfferResult:
        """Reserve space for an item before commitment."""
        return self._handoff._reserve(self._token, item)

    def close(self) -> None:
        """Close the handoff to new items."""
        self._handoff._close(self._token)


class PreparedIngressConsumer:
    """Exclusive consumer handle."""

    def __init__(self, handoff: PreparedSourceIngressHandoff, token) -> None:
        self._handoff = handoff
        self._token = token

    def try_poll(self) -> PreparedIngressItem | None:
        """Dequeue one pending item or return None."""
        return self._handoff._poll(self._token)

    poll = try_poll


__all__ = [
    "PreparedIngressConsumer",
    "PreparedIngressControlToken",
    "PreparedIngressDeferred",
    "PreparedIngressItem",
    "PreparedIngressMetrics",
    "PreparedIngressOfferResult",
    "PreparedIngressReservation",
    "PreparedIngressProducer",
    "PreparedSourceIngressHandoff",
]
