"""Bounded, non-blocking ownership for prepared source work.

The owner in this module is deliberately independent of map, topology, and
parameter versions.  It only decides whether a source preparation is ready
to be handed to a later JIT transaction-binding stage.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, replace
from enum import Enum, IntEnum
import numbers
import threading
from typing import Any, Mapping


from eligsir.contracts import SourceIdentity
from eligsir.runtime.prepared_source import PreparedSourceLease


class ReadySourceLane(str, Enum):
    """The two bounded source lanes owned by :class:`ReadySourceOwner`."""

    FRESH = "fresh"
    REPLAY = "replay"


class ReplayPriority(IntEnum):
    """Deterministic replay priority; lower values are selected first."""

    DEMANDED = 0
    NORMAL = 1


@dataclass(frozen=True, slots=True)
class ReadySourceTerminalRecord:
    """Record of a retired source identity with reason and control token."""

    source: SourceIdentity
    reason: str
    control_token: Any


@dataclass(frozen=True, slots=True)
class ReadySourceCounters:
    """Immutable bounded counters for diagnostics and reserve assertions."""

    fresh_pending: int
    replay_pending: int
    in_flight: int
    offered: int
    selected: int
    released: int
    full_drop_count: int
    full_not_resident_count: int
    duplicate_drop_count: int
    stale_generation_drop_count: int
    not_ready_count: int
    ready_error_count: int
    reserve_underflow_count: int
    fresh_ready_count: int = 0
    replay_ready_count: int = 0
    ready_bytes_current: int = 0
    ready_bytes_peak: int = 0
    replay_evicted_bytes: int = 0


@dataclass(slots=True)
class _ReadyEntry:
    """Internal entry tracking a prepared source in a READY lane."""

    lease: PreparedSourceLease
    lane: str
    priority: int
    arrival_ordinal: int
    enqueue_sequence: int
    # An in-flight B4 may have a deterministic post-compute return lane.  It
    # reserves that lane before dispatch, so producer pressure cannot make the
    # completion requeue overflow after device work has borrowed the payload.
    reserved_return_lanes: frozenset[str] = frozenset()
    ready_latched: bool = False
    # Cached resident byte counts (computed once at admission, unchanged across
    # requeue/select). Keyed by (tier, payload_kind). Deduped by storage identity.
    resident_bytes: tuple[tuple[str, str, int], ...] = ()

    @property
    def source(self) -> SourceIdentity:
        """Return the source identity for this entry."""
        return self.lease.source_identity


@dataclass(frozen=True, slots=True)
class _GenerationRecord:
    generation: int
    lifetime_token: Any


@dataclass(frozen=True, slots=True)
class _ReservationEntry:
    """Capacity claimed before a producer starts device preparation."""

    source: SourceIdentity
    lane: str
    arrival_ordinal: int
    priority: int
    generation: int
    # Capacity is not the only thing reserved before H2D.  The exact lifetime
    # object is the ABA fence for the borrowed payload published later.
    lifetime_token: Any
    control_token: Any


class ReadySourceReservation:
    """One-shot READY-lane capacity claim owned by the preparation producer.

    A reservation deliberately holds *only* bounded residency capacity and
    source identity metadata.  It never owns host staging or device payloads,
    so a cache-full result can leave a durably retained source COLD without
    submitting an H2D transfer.
    """

    def __init__(self, owner: "ReadySourceOwner", producer_owner: Any, entry: _ReservationEntry) -> None:
        self._owner = owner
        self._producer_owner = producer_owner
        self._entry = entry
        self._resolved = False

    @property
    def source(self) -> SourceIdentity:
        """Return the source identity for this reservation."""
        return self._entry.source

    def commit(self, lease: PreparedSourceLease) -> bool:
        """Commit a prepared lease to the reserved capacity."""
        return self._owner._commit_reservation(self, lease)

    def abort(self) -> bool:
        """Abort this reservation and return the capacity."""
        return self._owner._abort_reservation(self)


class ProvisionalReplayEviction:
    """One idle-replay eviction reserved ahead of compute dispatch.

    Reclaiming a return lane for a selected B4 may require evicting an
    unrelated idle replay source before canonical compute is entered.  This
    token defers finalizing that eviction until the transaction's outcome is
    known: resolve it through :meth:`ReadySourceOwner.resolve_replay_eviction`
    with ``commit=True`` only once compute execution was actually issued, or
    ``commit=False`` to restore the exact evicted identity and order -- a
    transaction that never commits must leave READY membership/order/capacity
    identical to its pre-attempt state.
    """

    __slots__ = ("_owner", "_entry")

    def __init__(self, owner: "ReadySourceOwner", entry: "_ReadyEntry") -> None:
        self._owner = owner
        self._entry = entry

    @property
    def lease(self) -> PreparedSourceLease:
        """Return the lease of the evicted replay entry."""
        return self._entry.lease


class _CompositeReadyEvent:
    """Preserve readiness when idle; replace it after a new device use."""

    @staticmethod
    def extend(left: Any, right: Any) -> Any:
        """Merge left and right readiness events, preferring right when non-None."""
        return left if right is None else right


class ReadySourceOwner:
    """Bounded fresh/replay lanes with explicit SPSC producer/consumer roles.

    The source-preparation producer calls :meth:`claim_producer` and offers
    immutable leases.  The GPU owner calls :meth:`claim_consumer` and selects
    already-complete leases.  No method waits for preparation; a delayed
    completion simply remains pending while another ready source may be
    selected.  The implementation is intentionally SPSC: use one producer
    combiner when several decode workers exist.  The bounded history rejects
    stale generation-zero ingress, while a scheduler page-in uses its newer
    prepared generation to rebuild a durable COLD identity.
    """

    def __init__(
        self,
        *,
        max_fresh: int,
        max_replay: int,
        replay_reserve: int = 0,
    ) -> None:
        self._max_fresh = self._positive_int(max_fresh, "max_fresh")
        self._max_replay = self._positive_int(max_replay, "max_replay")
        if isinstance(replay_reserve, bool) or not isinstance(replay_reserve, numbers.Integral):
            raise TypeError("replay_reserve must be a non-negative integer")
        self._replay_reserve = int(replay_reserve)
        if self._replay_reserve < 0:
            raise ValueError("replay_reserve must be a non-negative integer")
        if self._replay_reserve > self._max_replay:
            raise ValueError("replay_reserve cannot exceed max_replay")

        self._fresh: list[_ReadyEntry] = []
        self._replay: list[_ReadyEntry] = []
        self._in_flight: dict[SourceIdentity, _ReadyEntry] = {}
        self._reservations: dict[SourceIdentity, ReadySourceReservation] = {}
        self._state_lock = threading.Lock()
        self._generation_by_source: OrderedDict[SourceIdentity, _GenerationRecord] = OrderedDict()
        self._retired_frontier: tuple[int, int, int, str] | None = None
        self._minimum_stream_epoch = 0
        self._generation_history_limit = self._max_fresh + self._max_replay + 1
        self._producer_owner: Any = None
        self._consumer_owner: Any = None
        self._scheduler_owner: Any = None
        self._enqueue_sequence = 0
        self._offered = 0
        self._selected = 0
        self._released = 0
        self._full_drop_count = 0
        self._full_not_resident_count = 0
        self._duplicate_drop_count = 0
        self._stale_generation_drop_count = 0
        self._not_ready_count = 0
        self._ready_error_count = 0
        self._fresh_ready_count = 0
        self._replay_ready_count = 0
        self._reserve_underflow_count = 0
        self._ready_bytes_current = 0
        self._ready_bytes_peak = 0
        self._replay_evicted_bytes = 0
        # Cumulative bytes by (tier, payload_kind): ("native", "rgb"), ("native", "depth"), etc.
        self._ready_bytes_by_tier_kind: dict[tuple[str, str], int] = {}
        # Cumulative bytes by lane: ReadySourceLane.FRESH or ReadySourceLane.REPLAY
        self._ready_bytes_by_lane: dict[str, int] = {}
        self._terminal_records: OrderedDict[SourceIdentity, ReadySourceTerminalRecord] = OrderedDict()
        self._progress_callback: Callable[[], None] | None = None
        self._signal_progress_pending = False

    def set_progress_callback(self, callback: Callable[[], None] | None) -> None:
        """Register a callback to be invoked when READY-publication occurs.

        The callback is invoked AFTER releasing the owner's state lock on each
        READY transition (+1 lease becoming READY). Never call out under _state_lock
        to avoid deadlock with the consumer's shared condition lock.
        """
        if callback is not None and not callable(callback):
            raise TypeError("callback must be callable or None")
        self._progress_callback = callback

    @staticmethod
    def _positive_int(value: int, name: str) -> int:
        """Validate and return a positive integer or raise TypeError/ValueError."""
        if isinstance(value, bool) or not isinstance(value, numbers.Integral):
            raise TypeError(f"{name} must be a positive integer")
        value = int(value)
        if value < 1:
            raise ValueError(f"{name} must be a positive integer")
        return value

    @staticmethod
    def _owner_equal(left: Any, right: Any) -> bool:
        """Test ownership equality, returning False on exception."""
        if left is right:
            return True
        try:
            return bool(left == right)
        except Exception:
            return False

    def _claim(self, role: str, owner: Any | None) -> Any:
        """Claim exclusive ownership of a role and return its ownership token."""
        token = threading.get_ident() if owner is None else owner
        attributes = {
            "producer": "_producer_owner",
            "consumer": "_consumer_owner",
            "scheduler": "_scheduler_owner",
        }
        try:
            attribute = attributes[role]
        except KeyError as error:
            raise ValueError(f"unknown ready-source role {role!r}") from error
        current = getattr(self, attribute)
        if current is None:
            setattr(self, attribute, token)
        elif not self._owner_equal(current, token):
            raise RuntimeError(f"{role} ownership is exclusive")
        return token

    def claim_producer(self, owner: Any | None = None) -> "ReadySourceProducer":
        """Claim the only producer role and return its operation handle."""

        token = self._claim("producer", owner)
        return ReadySourceProducer(self, token)

    def claim_consumer(self, owner: Any | None = None) -> "ReadySourceConsumer":
        """Claim the only GPU-consumer role and return its operation handle."""

        token = self._claim("consumer", owner)
        return ReadySourceConsumer(self, token)

    def claim_scheduler(self, owner: Any | None = None) -> "ReadySourceScheduler":
        """Claim the bounded cache-eviction role without GPU selection rights.

        The scheduler may reclaim only idle replay residency for a demanded
        COLD page-in.  It cannot inspect or release a selected lease, keeping
        source preparation and B4 ownership disjoint.
        """

        token = self._claim("scheduler", owner)
        return ReadySourceScheduler(self, token)

    def offer(
        self,
        lease: PreparedSourceLease,
        *,
        lane: str = ReadySourceLane.REPLAY,
        arrival_ordinal: int | None = None,
        priority: int | str | ReplayPriority | None = None,
    ) -> bool:
        """Convenience producer operation for the calling thread."""

        return self._offer(
            threading.get_ident(),
            lease,
            lane=lane,
            arrival_ordinal=arrival_ordinal,
            priority=priority,
        )

    def poll(self, lane: str | None = None) -> PreparedSourceLease | None:
        """Convenience consumer operation for the calling thread."""

        return self._poll(threading.get_ident(), lane)

    def select(
        self,
        batch_size: int,
        *,
        fresh_limit: int | None = None,
        fresh_sources: frozenset[SourceIdentity] | None = None,
        replay_sources: frozenset[SourceIdentity] | None = None,
        exact_sources: tuple[SourceIdentity, ...] | None = None,
    ) -> tuple[PreparedSourceLease, ...]:
        """Select a ready fresh-first batch without waiting."""

        return self._select(
            threading.get_ident(),
            batch_size,
            fresh_limit=fresh_limit,
            fresh_sources=fresh_sources,
            replay_sources=replay_sources,
            exact_sources=exact_sources,
        )

    def release(self, lease: PreparedSourceLease) -> None:
        """Release a lease selected by the calling consumer thread."""

        self._release(threading.get_ident(), lease)

    def selected_lane(self, lease: PreparedSourceLease) -> ReadySourceLane:
        """Return the immutable lane assigned to an in-flight selected lease.

        This is intentionally read-only and consumer-owned.  It lets a JIT
        transaction preserve fresh/replay provenance without examining the
        owner internals or inferring lane from completion order.
        """

        self._claim("consumer", threading.get_ident())
        if not isinstance(lease, PreparedSourceLease):
            raise TypeError("lease must be PreparedSourceLease")
        with self._state_lock:
            entry = self._in_flight.get(lease.source_identity)
            if entry is None or entry.lease is not lease:
                raise ValueError("lease is not currently owned by the consumer")
            return ReadySourceLane(entry.lane)

    def ready_replay_sources(self) -> tuple[SourceIdentity, ...]:
        """Return the current READY replay identities without selecting them."""

        self._claim("consumer", threading.get_ident())
        with self._state_lock:
            return tuple(
                entry.source
                for entry in sorted(self._replay, key=self._key)
                if self._ready(entry, record=False)
            )

    def replay_resident_sources(self) -> tuple[SourceIdentity, ...]:
        """Return replay identities whose device residency cannot disappear.

        This includes queued replay leases regardless of their completion
        event, plus selected leases that already promise a replay return.  It
        deliberately excludes producer reservations and COLD identities: a
        reservation has no payload, while COLD has no device residency.
        """

        self._claim("consumer", threading.get_ident())
        with self._state_lock:
            sources = {entry.source for entry in self._replay}
            sources.update(
                entry.source
                for entry in self._in_flight.values()
                if entry.lane == ReadySourceLane.REPLAY
                or ReadySourceLane.REPLAY in entry.reserved_return_lanes
            )
            return tuple(sorted(sources, key=SourceIdentity.ordering_key))

    def exact_readiness(
        self,
        *,
        fresh_sources: frozenset[SourceIdentity],
        replay_sources: frozenset[SourceIdentity],
        exact_sources: tuple[SourceIdentity, ...],
    ) -> tuple[tuple[SourceIdentity, str], ...]:
        """Classify a frozen plan without selecting or changing queue order."""

        self._claim("consumer", threading.get_ident())
        with self._state_lock:
            fresh = {entry.source: entry for entry in self._fresh}
            replay = {entry.source: entry for entry in self._replay}
            states = []
            for source in exact_sources:
                lane = (
                    ReadySourceLane.FRESH
                    if source in fresh_sources
                    else ReadySourceLane.REPLAY
                )
                entries = fresh if lane is ReadySourceLane.FRESH else replay
                entry = entries.get(source)
                if entry is None:
                    states.append((source, f"ABSENT_{lane.value.upper()}"))
                    continue
                ready, errored = self._query_and_latch(entry)
                state = "EVENT_QUERY_ERROR" if errored else ("READY" if ready else "EVENT_PENDING")
                states.append((source, state))
            return tuple(states)

    def requeue(
        self, lease: PreparedSourceLease, *, ready_event: Any, lane: str | None = None
    ) -> PreparedSourceLease:
        """Atomically return an in-flight lease, optionally changing its lane."""

        return self._requeue(
            threading.get_ident(), lease, ready_event=ready_event, lane=lane
        )

    def evict_replay(
        self,
        *,
        count: int = 1,
        protected_sources: frozenset[SourceIdentity] = frozenset(),
        target_source: SourceIdentity | None = None,
    ) -> tuple[PreparedSourceLease, ...]:
        """Drop only idle replay residency under the GPU consumer's authority.

        This intentionally does not retire the durable source identity.  A
        later scheduler page-in must use a strictly newer prepared generation;
        the old borrowed device payload can never re-enter after eviction.
        In-flight leases are kept in a separate table and are therefore never
        candidates for this cache-only operation.
        """

        return self._evict_replay(
            threading.get_ident(),
            count=count,
            protected_sources=protected_sources,
            target_source=target_source,
        )

    def evict_replay_provisional(
        self, *, count: int = 1
    ) -> tuple[ProvisionalReplayEviction, ...]:
        """Reserve idle replay capacity without finalizing the eviction.

        Use this instead of :meth:`evict_replay` when the caller must
        reclaim return-lane capacity before compute dispatch but may not
        ultimately commit the transaction (for example, a pre-execute B4
        reservation attempt).  Resolve every returned token exactly once
        through :meth:`resolve_replay_eviction`.
        """

        return self._evict_replay_provisional(threading.get_ident(), count=count)

    def _evict_replay_provisional(
        self, consumer_owner: Any, *, count: int
    ) -> tuple[ProvisionalReplayEviction, ...]:
        """Reserve eviction capacity without finalizing until explicit commit."""
        self._claim("consumer", consumer_owner)
        if isinstance(count, bool) or not isinstance(count, numbers.Integral):
            raise TypeError("count must be a positive integer")
        count = int(count)
        if count < 1:
            raise ValueError("count must be a positive integer")
        with self._state_lock:
            selected = tuple(self._replay)[:count]
            if not selected:
                return ()
            selected_ids = {id(entry) for entry in selected}
            self._replay[:] = [
                entry for entry in self._replay if id(entry) not in selected_ids
            ]
            for entry in selected:
                if entry.ready_latched:
                    self._adjust_ready_count(entry.lane, -1)
                self._adjust_resident_bytes(entry, -1)
            return tuple(ProvisionalReplayEviction(self, entry) for entry in selected)

    def resolve_replay_eviction(
        self, evictions: tuple[ProvisionalReplayEviction, ...], *, commit: bool
    ) -> None:
        """Finalize (``commit=True``) or roll back (``commit=False``) prior evictions.

        ``commit=True`` retires the evicted identities exactly as
        :meth:`evict_replay` already does.  ``commit=False`` restores each
        evicted replay entry to its exact original identity and relative
        order -- unless a strictly newer generation, reservation, or
        in-flight claim has since superseded that source identity, in which
        case the stale evicted payload must never re-enter (the same ABA
        contract :meth:`evict_replay` already documents).
        """

        self._resolve_replay_eviction(threading.get_ident(), evictions, commit=commit)

    def _resolve_replay_eviction(
        self,
        consumer_owner: Any,
        evictions: tuple[ProvisionalReplayEviction, ...],
        *,
        commit: bool,
    ) -> None:
        """Finalize or rollback provisional evictions based on transaction outcome."""
        self._claim("consumer", consumer_owner)
        evictions = tuple(evictions)
        if not all(
            isinstance(eviction, ProvisionalReplayEviction) and eviction._owner is self
            for eviction in evictions
        ):
            raise TypeError(
                "evictions must be ProvisionalReplayEviction values from this owner"
            )
        if not evictions:
            return
        with self._state_lock:
            if commit:
                for eviction in evictions:
                    self._replay_evicted_bytes += sum(
                        nbytes for _, nbytes in eviction._entry.resident_bytes
                    )
                    self._terminal(eviction._entry.lease, "evicted_replay_residency")
                should_signal = False
            else:
                self._signal_progress_pending = False
                restorable = []
                for eviction in evictions:
                    entry = eviction._entry
                    source = entry.source
                    prior = self._generation_by_source.get(source)
                    superseded = (
                        prior is None
                        or prior.generation != entry.lease.generation
                        or prior.lifetime_token is not entry.lease.lifetime_token
                    )
                    already_present = (
                        source in self._reservations
                        or source in self._in_flight
                        or any(item.source == source for item in (*self._fresh, *self._replay))
                    )
                    if superseded or already_present:
                        continue
                    restorable.append(entry)
                # Provisional evictions are always taken from the front of the
                # replay queue in oldest-first order (see
                # ``_evict_replay_provisional``), so re-inserting them at the
                # front in the same relative order reconstructs the exact
                # pre-eviction queue -- membership, identity, and order alike.
                self._replay[:0] = restorable
                for entry in restorable:
                    if entry.ready_latched:
                        self._adjust_ready_count(entry.lane, +1)
                    self._adjust_resident_bytes(entry, +1)
                should_signal = self._signal_progress_pending
        # Invoke callback outside the lock
        if should_signal:
            self._progress_callback()

    def _normalize_lane(self, lane: str) -> str:
        """Validate and return a normalized lane string."""
        lane = getattr(lane, "value", lane)
        if lane not in (ReadySourceLane.FRESH, ReadySourceLane.REPLAY):
            raise ValueError(f"unknown ready-source lane {lane!r}")
        return str(lane)

    def _evict_replay(
        self,
        consumer_owner: Any,
        *,
        count: int,
        protected_sources: frozenset[SourceIdentity] = frozenset(),
        target_source: SourceIdentity | None = None,
    ) -> tuple[PreparedSourceLease, ...]:
        """Evict idle replay sources under consumer authority."""
        self._claim("consumer", consumer_owner)
        return self._evict_replay_locked(
            count=count,
            protected_sources=protected_sources,
            target_source=target_source,
        )

    def _scheduler_evict_replay(
        self,
        scheduler_owner: Any,
        *,
        count: int,
        protected_sources: frozenset[SourceIdentity] = frozenset(),
    ) -> tuple[PreparedSourceLease, ...]:
        """Evict idle replay sources while protecting demanded sources."""
        self._claim("scheduler", scheduler_owner)
        return self._evict_replay_locked(
            count=count,
            protected_sources=protected_sources,
        )

    def _scheduler_evict_fresh(
        self,
        scheduler_owner: Any,
        *,
        protected_sources: frozenset[SourceIdentity],
    ) -> tuple[PreparedSourceLease, ...]:
        """Evict one idle fresh lease while protecting demanded sources."""
        self._claim("scheduler", scheduler_owner)
        if not isinstance(protected_sources, frozenset) or not all(
            isinstance(source, SourceIdentity) for source in protected_sources
        ):
            raise TypeError("protected_sources must be a frozenset of SourceIdentity")
        with self._state_lock:
            for index, entry in enumerate(self._fresh):
                if entry.source not in protected_sources:
                    del self._fresh[index]
                    self._adjust_resident_bytes(entry, -1)
                    self._terminal(entry.lease, "evicted_fresh_residency")
                    return (entry.lease,)
        return ()

    def _scheduler_has_source(
        self, scheduler_owner: Any, source: SourceIdentity, *, ready: bool = False
    ) -> bool:
        """Check whether a source exists, optionally filtering to READY status."""
        self._claim("scheduler", scheduler_owner)
        if not isinstance(source, SourceIdentity):
            raise TypeError("source must be SourceIdentity")
        with self._state_lock:
            if ready:
                return any(
                    entry.source == source and self._ready(entry)
                    for entry in (*self._fresh, *self._replay)
                )
            return (
                source in self._in_flight
                or source in self._reservations
                or any(entry.source == source for entry in self._fresh)
                or any(entry.source == source for entry in self._replay)
            )

    def _scheduler_next_generation(
        self, scheduler_owner: Any, source: SourceIdentity
    ) -> int:
        """Return the next safe prepared generation for a source."""
        self._claim("scheduler", scheduler_owner)
        if not isinstance(source, SourceIdentity):
            raise TypeError("source must be SourceIdentity")
        with self._state_lock:
            prior = self._generation_by_source.get(source)
            return 1 if prior is None else prior.generation + 1

    def _scheduler_replay_residency_depth(self, scheduler_owner: Any) -> int:
        """Return count of in-device replay residency."""
        self._claim("scheduler", scheduler_owner)
        with self._state_lock:
            return self._lane_occupancy_locked(ReadySourceLane.REPLAY)

    def _evict_replay_locked(
        self,
        *,
        count: int,
        protected_sources: frozenset[SourceIdentity] = frozenset(),
        target_source: SourceIdentity | None = None,
    ) -> tuple[PreparedSourceLease, ...]:
        """Evict replay sources after validating count and acquiring the lock."""
        if isinstance(count, bool) or not isinstance(count, numbers.Integral):
            raise TypeError("count must be a positive integer")
        count = int(count)
        if count < 1:
            raise ValueError("count must be a positive integer")
        if not isinstance(protected_sources, frozenset) or not all(
            isinstance(source, SourceIdentity) for source in protected_sources
        ):
            raise TypeError("protected_sources must be a frozenset of SourceIdentity")
        if target_source is not None and not isinstance(target_source, SourceIdentity):
            raise TypeError("target_source must be SourceIdentity or None")
        with self._state_lock:
            # Pending replay order is deterministic.  No in-flight source is
            # present in this list, so eviction cannot invalidate a launched
            # B4 borrow or its completion requeue reservation.
            selected = tuple(
                entry
                for entry in self._replay
                if entry.source not in protected_sources
                and (target_source is None or entry.source == target_source)
            )[:count]
            if not selected:
                return ()
            selected_ids = {id(entry) for entry in selected}
            self._replay[:] = [
                entry for entry in self._replay if id(entry) not in selected_ids
            ]
            for entry in selected:
                if entry.ready_latched:
                    self._adjust_ready_count(entry.lane, -1)
                self._adjust_resident_bytes(entry, -1)
                self._replay_evicted_bytes += sum(
                    nbytes for _, nbytes in entry.resident_bytes
                )
                self._terminal(entry.lease, "evicted_replay_residency")
            return tuple(entry.lease for entry in selected)

    @staticmethod
    def _normalize_priority(priority: int | str | ReplayPriority | None) -> int:
        """Validate and normalize a replay priority value."""
        if priority is None:
            return int(ReplayPriority.NORMAL)
        if isinstance(priority, str):
            aliases = {
                "demanded": ReplayPriority.DEMANDED,
                "demanded_replay": ReplayPriority.DEMANDED,
                "normal": ReplayPriority.NORMAL,
                "normal_replay": ReplayPriority.NORMAL,
            }
            if priority not in aliases:
                raise ValueError(f"unknown replay priority {priority!r}")
            return int(aliases[priority])
        if isinstance(priority, bool) or not isinstance(priority, numbers.Integral):
            raise TypeError("priority must be an integer or known string")
        priority = int(priority)
        if priority < 0:
            raise ValueError("priority must be non-negative")
        return priority

    def _offer(
        self,
        producer_owner: Any,
        lease: PreparedSourceLease,
        *,
        lane: str,
        arrival_ordinal: int | None,
        priority: int | str | ReplayPriority | None,
        full_is_nonterminal: bool = False,
    ) -> bool:
        """Offer a prepared lease to a lane with callback signaling."""
        with self._state_lock:
            self._signal_progress_pending = False
            result = self._offer_locked(
                producer_owner,
                lease,
                lane=lane,
                arrival_ordinal=arrival_ordinal,
                priority=priority,
                full_is_nonterminal=full_is_nonterminal,
            )
            should_signal = self._signal_progress_pending
        # Invoke callback outside the lock
        if should_signal:
            self._progress_callback()
        return result

    def _offer_locked(
        self,
        producer_owner: Any,
        lease: PreparedSourceLease,
        *,
        lane: str,
        arrival_ordinal: int | None,
        priority: int | str | ReplayPriority | None,
        full_is_nonterminal: bool,
    ) -> bool:
        """Perform lease admission checks and queue management under the state lock."""
        self._claim("producer", producer_owner)
        if not isinstance(lease, PreparedSourceLease):
            raise TypeError("lease must be PreparedSourceLease")
        lane = self._normalize_lane(lane)
        if arrival_ordinal is None:
            arrival_ordinal = lease.source_identity.source_ordinal
        if isinstance(arrival_ordinal, bool) or not isinstance(arrival_ordinal, numbers.Integral):
            raise TypeError("arrival_ordinal must be a non-negative integer")
        arrival_ordinal = int(arrival_ordinal)
        if arrival_ordinal < 0:
            raise ValueError("arrival_ordinal must be a non-negative integer")
        priority_value = self._normalize_priority(priority)
        source = lease.source_identity
        if source.stream_epoch < self._minimum_stream_epoch:
            self._stale_generation_drop_count += 1
            self._terminal(lease, "stale_generation")
            return False
        lifetime_token = lease.lifetime_token
        prior = self._generation_by_source.get(source)
        terminal = self._terminal_records.get(source)
        if prior is not None:
            if lease.generation < prior.generation or (
                lease.generation == prior.generation
                and (
                    lease.lifetime_token is not prior.lifetime_token
                    or (
                        terminal is not None
                        and terminal.reason == "evicted_replay_residency"
                    )
                )
            ):
                self._stale_generation_drop_count += 1
                self._terminal(lease, "stale_generation")
                return False
        else:
            retired_frontier = self._retired_frontier
            if (
                retired_frontier is not None
                and lease.generation == 0
                and self._source_order(source) <= retired_frontier
            ):
                self._stale_generation_drop_count += 1
                self._terminal(lease, "stale_generation")
                return False
        if (
            source in self._reservations
            or source in self._in_flight
            or any(item.source == source for item in (*self._fresh, *self._replay))
        ):
            self._duplicate_drop_count += 1
            self._terminal(lease, "duplicate_source")
            return False
        if not self._lane_has_capacity_locked(lane):
            if not full_is_nonterminal:
                self._full_drop_count += 1
                self._terminal(lease, "ready_queue_full")
            else:
                self._full_not_resident_count += 1
            return False
        queue = self._fresh if lane == ReadySourceLane.FRESH else self._replay
        self._generation_by_source[source] = _GenerationRecord(
            generation=lease.generation,
            lifetime_token=lifetime_token,
        )
        self._generation_by_source.move_to_end(source)
        self._trim_generation_history()
        # Compute resident bytes once, cache on entry.
        resident_bytes = self._compute_lease_resident_bytes(lease)
        entry = _ReadyEntry(
            lease=lease,
            lane=lane,
            priority=priority_value,
            arrival_ordinal=arrival_ordinal,
            enqueue_sequence=self._enqueue_sequence,
            ready_latched=lease.ready_event is None,
            resident_bytes=resident_bytes,
        )
        self._enqueue_sequence += 1
        queue.append(entry)
        if entry.ready_latched:
            self._adjust_ready_count(lane, +1)
        self._adjust_resident_bytes(entry, +1)
        self._offered += 1
        return True

    def _reserve(
        self,
        producer_owner: Any,
        source: SourceIdentity,
        *,
        lane: str,
        arrival_ordinal: int | None,
        priority: int | str | ReplayPriority | None,
        generation: int,
        lifetime_token: Any,
        full_is_nonterminal: bool,
        control_token: Any = None,
    ) -> ReadySourceReservation | None:
        """Claim READY capacity before allocating/staging a source payload."""

        self._claim("producer", producer_owner)
        if not isinstance(source, SourceIdentity):
            raise TypeError("source must be SourceIdentity")
        if isinstance(generation, bool) or not isinstance(generation, numbers.Integral):
            raise TypeError("generation must be a non-negative integer")
        generation = int(generation)
        if generation < 0:
            raise ValueError("generation must be a non-negative integer")
        if not isinstance(full_is_nonterminal, bool):
            raise TypeError("full_is_nonterminal must be bool")
        lane = self._normalize_lane(lane)
        if arrival_ordinal is None:
            arrival_ordinal = source.source_ordinal
        if isinstance(arrival_ordinal, bool) or not isinstance(arrival_ordinal, numbers.Integral):
            raise TypeError("arrival_ordinal must be a non-negative integer")
        arrival_ordinal = int(arrival_ordinal)
        if arrival_ordinal < 0:
            raise ValueError("arrival_ordinal must be a non-negative integer")
        priority_value = self._normalize_priority(priority)

        with self._state_lock:
            if source.stream_epoch < self._minimum_stream_epoch:
                self._stale_generation_drop_count += 1
                self._terminal_source(source, "stale_generation", control_token)
                return None
            prior = self._generation_by_source.get(source)
            if prior is not None and (
                generation < prior.generation
                or (
                    generation == prior.generation
                    and lifetime_token is not prior.lifetime_token
                )
            ):
                self._stale_generation_drop_count += 1
                self._terminal_source(source, "stale_generation", control_token)
                return None
            if prior is None:
                retired_frontier = self._retired_frontier
                if (
                    retired_frontier is not None
                    and generation == 0
                    and self._source_order(source) <= retired_frontier
                ):
                    self._stale_generation_drop_count += 1
                    self._terminal_source(source, "stale_generation", control_token)
                    return None
            if (
                source in self._reservations
                or source in self._in_flight
                or any(item.source == source for item in (*self._fresh, *self._replay))
            ):
                self._duplicate_drop_count += 1
                self._terminal_source(source, "duplicate_source", control_token)
                return None
            if not self._lane_has_capacity_locked(lane):
                if full_is_nonterminal:
                    self._full_not_resident_count += 1
                else:
                    self._full_drop_count += 1
                    self._terminal_source(source, "ready_queue_full", control_token)
                return None
            entry = _ReservationEntry(
                source=source,
                lane=lane,
                arrival_ordinal=arrival_ordinal,
                priority=priority_value,
                generation=generation,
                lifetime_token=lifetime_token,
                control_token=control_token,
            )
            reservation = ReadySourceReservation(self, producer_owner, entry)
            self._reservations[source] = reservation
            return reservation

    def _commit_reservation(
        self, reservation: ReadySourceReservation, lease: PreparedSourceLease
    ) -> bool:
        """Turn one pre-preparation claim into a resident immutable lease."""

        if not isinstance(reservation, ReadySourceReservation):
            raise TypeError("reservation must be ReadySourceReservation")
        if not isinstance(lease, PreparedSourceLease):
            raise TypeError("lease must be PreparedSourceLease")
        self._claim("producer", reservation._producer_owner)
        should_signal = False
        with self._state_lock:
            self._signal_progress_pending = False
            current = self._reservations.get(reservation.source)
            if current is not reservation or reservation._resolved:
                raise RuntimeError("ready-source reservation is not active")
            entry = reservation._entry
            if lease.source_identity != entry.source or lease.generation != entry.generation:
                # A reservation must never remain live after a caller tries
                # to publish a different borrowed lifetime.  Free the slot so
                # a valid later preparation cannot be stranded behind it.
                self._reservations.pop(entry.source)
                reservation._resolved = True
                raise ValueError("lease does not match ready-source reservation")
            if lease.lifetime_token is not entry.lifetime_token:
                self._reservations.pop(entry.source)
                reservation._resolved = True
                raise ValueError("lease lifetime token does not match ready-source reservation")
            prior = self._generation_by_source.get(entry.source)
            if prior is not None and (
                lease.generation < prior.generation
                or (
                    lease.generation == prior.generation
                    and lease.lifetime_token is not prior.lifetime_token
                )
            ):
                self._reservations.pop(entry.source)
                reservation._resolved = True
                raise ValueError("lease generation is stale")
            if not self._lane_has_capacity_locked(
                entry.lane, exclude_reservations=frozenset((entry.source,))
            ):
                self._reservations.pop(entry.source)
                reservation._resolved = True
                raise RuntimeError("ready-source reservation capacity was lost")
            queue = self._fresh if entry.lane == ReadySourceLane.FRESH else self._replay
            self._generation_by_source[entry.source] = _GenerationRecord(
                generation=lease.generation,
                lifetime_token=lease.lifetime_token,
            )
            self._generation_by_source.move_to_end(entry.source)
            # Compute resident bytes once, cache on entry.
            resident_bytes = self._compute_lease_resident_bytes(lease)
            ready_entry = _ReadyEntry(
                    lease=lease,
                    lane=entry.lane,
                    priority=entry.priority,
                    arrival_ordinal=entry.arrival_ordinal,
                    enqueue_sequence=self._enqueue_sequence,
                    ready_latched=lease.ready_event is None,
                    resident_bytes=resident_bytes,
                )
            queue.append(ready_entry)
            if ready_entry.ready_latched:
                self._adjust_ready_count(entry.lane, +1)
            self._adjust_resident_bytes(ready_entry, +1)
            self._enqueue_sequence += 1
            self._offered += 1
            self._reservations.pop(entry.source)
            reservation._resolved = True
            self._trim_generation_history()
            should_signal = self._signal_progress_pending
        # Invoke callback outside the lock
        if should_signal:
            self._progress_callback()
        return True

    def _lane_occupancy_locked(
        self,
        lane: str,
        *,
        exclude_in_flight: frozenset[SourceIdentity] = frozenset(),
        exclude_reservations: frozenset[SourceIdentity] = frozenset(),
    ) -> int:
        """Return every claim against one bounded READY lane.

        Pending leases, producer reservations, and selected leases with a
        promised completion lane all consume the same finite residency budget.
        Every mutator uses this one accounting rule so a future H2D cannot win
        capacity that a selected B4 already needs to return safely.
        """

        lane = self._normalize_lane(lane)
        queue = self._fresh if lane == ReadySourceLane.FRESH else self._replay
        pending = len(queue)
        producer_claims = sum(
            1
            for source, reservation in self._reservations.items()
            if source not in exclude_reservations and reservation._entry.lane == lane
        )
        in_flight_claims = sum(
            1
            for source, entry in self._in_flight.items()
            if source not in exclude_in_flight
            and (lane == entry.lane or lane in entry.reserved_return_lanes)
        )
        return pending + producer_claims + in_flight_claims

    def _lane_has_capacity_locked(
        self,
        lane: str,
        *,
        incoming: int = 1,
        exclude_in_flight: frozenset[SourceIdentity] = frozenset(),
        exclude_reservations: frozenset[SourceIdentity] = frozenset(),
    ) -> bool:
        """Check whether a lane has capacity for incoming claims."""
        if isinstance(incoming, bool) or not isinstance(incoming, numbers.Integral):
            raise TypeError("incoming must be a non-negative integer")
        incoming = int(incoming)
        if incoming < 0:
            raise ValueError("incoming must be a non-negative integer")
        lane = self._normalize_lane(lane)
        limit = self._max_fresh if lane == ReadySourceLane.FRESH else self._max_replay
        return self._lane_occupancy_locked(
            lane,
            exclude_in_flight=exclude_in_flight,
            exclude_reservations=exclude_reservations,
        ) + incoming <= limit

    def _abort_reservation(self, reservation: ReadySourceReservation) -> bool:
        """Cancel a producer reservation and free its capacity."""
        if not isinstance(reservation, ReadySourceReservation):
            raise TypeError("reservation must be ReadySourceReservation")
        self._claim("producer", reservation._producer_owner)
        with self._state_lock:
            current = self._reservations.get(reservation.source)
            if current is not reservation or reservation._resolved:
                return False
            self._reservations.pop(reservation.source)
            reservation._resolved = True
            return True

    def _terminal(self, lease: PreparedSourceLease, reason: str) -> None:
        """Record a lease termination with its reason."""
        self._terminal_source(lease.source_identity, reason, lease.control_token)

    def _terminal_source(self, source: SourceIdentity, reason: str, control_token: Any) -> None:
        """Record source termination and maintain bounded history."""
        self._terminal_records[source] = ReadySourceTerminalRecord(
            source=source,
            reason=reason,
            control_token=control_token,
        )
        self._terminal_records.move_to_end(source)
        while len(self._terminal_records) > self._generation_history_limit:
            self._terminal_records.popitem(last=False)

    def _abandon_pending(self, producer_owner: Any, *, reason: str) -> None:
        """Abandon all pending sources and release their capacity."""
        self._claim("producer", producer_owner)
        with self._state_lock:
            # Decrement bytes for all queued entries (fresh + replay).
            # In-flight entries are left alone (they remain resident).
            for entry in (*self._fresh, *self._replay):
                self._adjust_resident_bytes(entry, -1)
                self._terminal(entry.lease, reason)
            self._fresh.clear()
            self._replay.clear()
            self._fresh_ready_count = 0
            self._replay_ready_count = 0
            for reservation in self._reservations.values():
                reservation._resolved = True
            self._reservations.clear()

    def _trim_generation_history(self) -> None:
        """Keep ABA protection bounded without evicting active sources."""

        while len(self._generation_by_source) > self._generation_history_limit:
            evicted = False
            for source in tuple(self._generation_by_source):
                if source not in self._in_flight and not any(
                    entry.source == source
                    for entry in (*self._fresh, *self._replay)
                ) and source not in self._reservations:
                    retired = self._source_order(source)
                    if self._retired_frontier is None:
                        self._retired_frontier = retired
                    else:
                        self._retired_frontier = max(self._retired_frontier, retired)
                    del self._generation_by_source[source]
                    evicted = True
                    break
            if not evicted:
                return

    @staticmethod
    def _source_order(source: SourceIdentity) -> tuple[int, int, int, str]:
        """Extract ordering key from a source identity."""
        return (
            source.stream_epoch,
            source.source_ordinal,
            source.timestamp_ns,
            source.source_id,
        )

    def _reset_generation_history(
        self,
        producer_owner: Any,
        *,
        next_stream_epoch: int,
    ) -> None:
        """Clear generation history and advance the minimum stream epoch."""
        self._claim("producer", producer_owner)
        with self._state_lock:
            if self._fresh or self._replay or self._in_flight or self._reservations:
                raise RuntimeError("generation history reset requires an idle owner")
            if (
                isinstance(next_stream_epoch, bool)
                or not isinstance(next_stream_epoch, numbers.Integral)
                or int(next_stream_epoch) <= self._minimum_stream_epoch
            ):
                raise ValueError(
                    "next_stream_epoch must be an integer greater than the current epoch"
                )
            self._minimum_stream_epoch = int(next_stream_epoch)
            self._generation_by_source.clear()
            self._retired_frontier = None

    def _adjust_ready_count(self, lane: str, delta: int) -> None:
        """Adjust per-lane READY count (fresh or replay) by delta (+1/-1).

        When delta > 0 (READY publication), sets _signal_progress_pending=True
        so the progress callback is invoked by the wrapping method AFTER releasing the state lock.
        """
        if delta > 0 and self._progress_callback is not None:
            self._signal_progress_pending = True

        if lane == ReadySourceLane.REPLAY:
            self._replay_ready_count += delta
        elif lane == ReadySourceLane.FRESH:
            self._fresh_ready_count += delta

    @staticmethod
    def _compute_lease_resident_bytes(
        lease: PreparedSourceLease,
    ) -> tuple[tuple[str, str, int], ...]:
        """Compute resident bytes for a lease, deduped by storage identity.

        Returns a tuple of (tier, payload_kind, nbytes) sorted by (tier, payload_kind).
        Deduplication by storage pointer prevents double-counting tensors that
        appear in both borrowed_device_payloads and tier_payloads.
        """
        seen_ptrs: set[int] = set()
        breakdown: dict[tuple[str, str], int] = {}

        # Count borrowed device payloads (the core residence).
        for payload_name, payload_ref in lease.borrowed_device_payloads.items():
            data = getattr(payload_ref, "data", None)
            if data is None or not hasattr(data, "untyped_storage"):
                continue
            ptr = data.untyped_storage().data_ptr()
            if ptr in seen_ptrs:
                continue
            seen_ptrs.add(ptr)
            nbytes = int(data.numel()) * int(data.element_size())
            # Borrowed payloads are not tied to a tier, so use a generic tier key.
            key = ("base", payload_name)
            breakdown[key] = breakdown.get(key, 0) + nbytes

        # Count tier-specific payloads, deduped against borrowed.
        for tier_name, tier_payload in lease.tier_payloads.items():
            for field_name in ("rgb", "depth", "camera_intrinsics", "depth_confidence",
                               "projection_matrix", "full_proj_transform"):
                payload_ref = getattr(tier_payload, field_name, None)
                if payload_ref is None:
                    continue
                data = getattr(payload_ref, "data", None)
                if data is None or not hasattr(data, "untyped_storage"):
                    continue
                ptr = data.untyped_storage().data_ptr()
                if ptr in seen_ptrs:
                    continue
                seen_ptrs.add(ptr)
                nbytes = int(data.numel()) * int(data.element_size())
                key = (tier_name, field_name)
                breakdown[key] = breakdown.get(key, 0) + nbytes

        return tuple(sorted(breakdown.items(), key=lambda kv: kv[0]))

    def _adjust_resident_bytes(self, entry: _ReadyEntry, sign: int) -> None:
        """Adjust resident byte counters by the entry's cached breakdown.

        Sign should be +1 for admission, -1 for departure. Updates current,
        peak, and per-lane/per-tier breakdowns.
        """
        if not entry.resident_bytes:
            return
        total_delta = 0
        for (tier, kind), nbytes in entry.resident_bytes:
            delta = sign * nbytes
            key = (tier, kind)
            self._ready_bytes_by_tier_kind[key] = (
                self._ready_bytes_by_tier_kind.get(key, 0) + delta
            )
            total_delta += delta

        # Update per-lane breakdown.
        self._ready_bytes_by_lane[entry.lane] = (
            self._ready_bytes_by_lane.get(entry.lane, 0) + total_delta
        )

        # Update current and peak.
        self._ready_bytes_current = max(0, self._ready_bytes_current + total_delta)
        self._ready_bytes_peak = max(self._ready_bytes_peak, self._ready_bytes_current)

    def _query_and_latch(self, entry: _ReadyEntry) -> tuple[bool, bool]:
        """Query lease readiness once, latching a successful observation.

        Returns ``(ready, errored)``. Shared by ``_ready()`` (the real
        selection path, which also maintains ``_not_ready_count``/
        ``_ready_error_count``) and ``exact_readiness()`` (an observational
        probe that must not double-count those selection-path stats, but
        still needs to promote a successful query into the same monotonic
        ``ready_latched``/per-lane ready counts -- issue #1575A: a
        candidate repeatedly inspected but not yet selected was re-querying
        an already-completed event on every tick).
        """

        if entry.ready_latched:
            return True, False
        try:
            ready = entry.lease.is_ready()
        except Exception:
            return False, True
        if ready:
            entry.ready_latched = True
            self._adjust_ready_count(entry.lane, +1)
        return ready, False

    def _ready(self, entry: _ReadyEntry, *, record: bool = True) -> bool:
        """Check lease readiness and optionally record diagnostics."""
        ready, errored = self._query_and_latch(entry)
        if record:
            if errored:
                self._ready_error_count += 1
            elif not ready:
                self._not_ready_count += 1
        return ready

    @staticmethod
    def _key(entry: _ReadyEntry) -> tuple[Any, ...]:
        """Generate sorting key for a ready entry."""
        if entry.lane == ReadySourceLane.REPLAY:
            # Replay is a deterministic FIFO cycle: a completed/requeued B4
            # receives a new enqueue sequence and moves behind waiting peers.
            return (entry.priority, entry.enqueue_sequence, entry.source.ordering_key())
        # Fresh work keeps arrival-order determinism even when preparation
        # completion races alter the order in which it became ready.
        return (-1, entry.arrival_ordinal, entry.source.ordering_key(), entry.enqueue_sequence)

    def _select(
        self,
        consumer_owner: Any,
        batch_size: int,
        *,
        fresh_limit: int | None = None,
        lane: str | None = None,
        fresh_sources: frozenset[SourceIdentity] | None = None,
        replay_sources: frozenset[SourceIdentity] | None = None,
        exact_sources: tuple[SourceIdentity, ...] | None = None,
    ) -> tuple[PreparedSourceLease, ...]:
        """Select READY leases from fresh/replay lanes without waiting."""
        self._claim("consumer", consumer_owner)
        should_signal = False
        with self._state_lock:
            self._signal_progress_pending = False
            if isinstance(batch_size, bool) or not isinstance(batch_size, numbers.Integral):
                raise TypeError("batch_size must be a positive integer")
            batch_size = int(batch_size)
            if batch_size < 1:
                raise ValueError("batch_size must be a positive integer")
            if fresh_limit is None:
                fresh_limit = batch_size
            if isinstance(fresh_limit, bool) or not isinstance(fresh_limit, numbers.Integral):
                raise TypeError("fresh_limit must be a non-negative integer")
            fresh_limit = int(fresh_limit)
            if fresh_limit < 0:
                raise ValueError("fresh_limit must be a non-negative integer")
            if lane is not None:
                lane = self._normalize_lane(lane)
            for name, sources in (
                ("fresh_sources", fresh_sources),
                ("replay_sources", replay_sources),
            ):
                if sources is not None and not isinstance(sources, frozenset):
                    raise TypeError(f"{name} must be frozenset or None")
            if exact_sources is not None:
                if (
                    not isinstance(exact_sources, tuple)
                    or not exact_sources
                    or not all(isinstance(source, SourceIdentity) for source in exact_sources)
                ):
                    raise TypeError("exact_sources must be a non-empty tuple of SourceIdentity")
                if len(set(exact_sources)) != len(exact_sources):
                    raise ValueError("exact_sources must be unique")
                if (
                    batch_size != len(exact_sources)
                    or fresh_limit != batch_size
                    or lane is not None
                ):
                    raise ValueError("exact selection requires one unrestricted matching batch")
                if (
                    fresh_sources is None
                    or replay_sources is None
                    or fresh_sources & replay_sources
                    or fresh_sources | replay_sources != frozenset(exact_sources)
                ):
                    raise ValueError("exact sources must have one authorized lane")
                fresh_by_source = {entry.source: entry for entry in self._fresh}
                replay_by_source = {entry.source: entry for entry in self._replay}
                selected_entries = []
                for source in exact_sources:
                    entry = (
                        fresh_by_source.get(source)
                        if source in fresh_sources
                        else replay_by_source.get(source)
                    )
                    if entry is None or not self._ready(entry, record=False):
                        return ()
                    selected_entries.append(entry)
            else:
                fresh_ready = []
                if lane in (None, ReadySourceLane.FRESH):
                    fresh_ready = sorted(
                        (
                            entry
                            for entry in self._fresh
                            if (fresh_sources is None or entry.source in fresh_sources)
                            and self._ready(entry)
                        ),
                        key=self._key,
                    )[: min(batch_size, fresh_limit)]
                selected_entries = list(fresh_ready)
                remaining = batch_size - len(selected_entries)
                if remaining and lane in (None, ReadySourceLane.REPLAY):
                    replay_ready = sorted(
                        (
                            entry
                            for entry in self._replay
                            if (replay_sources is None or entry.source in replay_sources)
                            and self._ready(entry)
                        ),
                        key=self._key,
                    )[:remaining]
                    selected_entries.extend(replay_ready)

            selected = {id(entry) for entry in selected_entries}
            if selected:
                self._fresh[:] = [entry for entry in self._fresh if id(entry) not in selected]
                self._replay[:] = [entry for entry in self._replay if id(entry) not in selected]
                for entry in selected_entries:
                    self._in_flight[entry.source] = entry
                    if entry.ready_latched:
                        self._adjust_ready_count(entry.lane, -1)
                self._selected += len(selected_entries)
            replay_depth = self._replay_ready_count
            if (
                len(selected_entries) < batch_size
                or replay_depth < self._replay_reserve
            ) and self._replay_reserve > 0:
                self._reserve_underflow_count += 1
            should_signal = self._signal_progress_pending
            result = tuple(entry.lease for entry in selected_entries)
        # Invoke callback outside the lock
        if should_signal:
            self._progress_callback()
        return result

    def _poll(
        self,
        consumer_owner: Any,
        lane: str | None = None,
    ) -> PreparedSourceLease | None:
        """Poll for one READY lease without changing selection order."""
        self._claim("consumer", consumer_owner)
        if lane is None:
            selected = self._select(consumer_owner, 1)
        else:
            lane = self._normalize_lane(lane)
            selected = self._select(
                consumer_owner,
                1,
                fresh_limit=1 if lane == ReadySourceLane.FRESH else 0,
                lane=lane,
            )
        return None if not selected else selected[0]

    def _release(self, consumer_owner: Any, lease: PreparedSourceLease) -> None:
        """Release a selected lease and free its capacity."""
        self._claim("consumer", consumer_owner)
        with self._state_lock:
            if not isinstance(lease, PreparedSourceLease):
                raise TypeError("lease must be PreparedSourceLease")
            entry = self._in_flight.pop(lease.source_identity, None)
            if entry is None or entry.lease is not lease:
                raise ValueError("lease is not currently owned by the consumer")
            self._adjust_resident_bytes(entry, -1)
            self._released += 1

    def _requeue(
        self,
        consumer_owner: Any,
        lease: PreparedSourceLease,
        *,
        ready_event: Any,
        lane: str | None = None,
    ) -> PreparedSourceLease:
        """Return an in-flight lease to a queue, optionally changing its lane."""
        self._claim("consumer", consumer_owner)
        with self._state_lock:
            return self._requeue_locked(lease, ready_event=ready_event, lane=lane)

    def requeue_batch(
        self,
        leases: tuple[PreparedSourceLease, ...],
        *,
        ready_event: Any,
        lane: str | None = None,
        lane_by_source: Mapping[SourceIdentity, str] | None = None,
    ) -> tuple[PreparedSourceLease, ...]:
        """Atomically requeue a selected batch or leave every lease in flight."""

        return self._requeue_batch(
            threading.get_ident(),
            leases,
            ready_event=ready_event,
            lane=lane,
            lane_by_source=lane_by_source,
        )

    def _requeue_batch(
        self,
        consumer_owner: Any,
        leases: tuple[PreparedSourceLease, ...],
        *,
        ready_event: Any,
        lane: str | None,
        lane_by_source: Mapping[SourceIdentity, str] | None,
    ) -> tuple[PreparedSourceLease, ...]:
        """Return a batch of selected leases to queues with optional per-source lanes."""
        self._claim("consumer", consumer_owner)
        leases = tuple(leases)
        if lane is not None and lane_by_source is not None:
            raise ValueError("lane and lane_by_source are mutually exclusive")
        if lane_by_source is not None and not isinstance(lane_by_source, Mapping):
            raise TypeError("lane_by_source must be a mapping or None")
        if ready_event is not None and not callable(getattr(ready_event, "query", None)):
            raise TypeError("ready_event must provide a callable query() method")
        if len({lease.source_identity for lease in leases}) != len(leases):
            raise ValueError("batch requeue leases must have unique sources")
        with self._state_lock:
            entries = []
            for lease in leases:
                if not isinstance(lease, PreparedSourceLease):
                    raise TypeError("leases must contain PreparedSourceLease values")
                entry = self._in_flight.get(lease.source_identity)
                prior = self._generation_by_source.get(lease.source_identity)
                if entry is None or entry.lease is not lease or prior is None or (
                    prior.generation != lease.generation
                    or prior.lifetime_token is not lease.lifetime_token
                ):
                    raise ValueError("batch requeue contains stale or unowned lease")
                if lane_by_source is None:
                    target = entry.lane if lane is None else self._normalize_lane(lane)
                else:
                    target = self._normalize_lane(
                        lane_by_source.get(lease.source_identity, entry.lane)
                    )
                entries.append((lease, entry, target))
            for target in (ReadySourceLane.FRESH, ReadySourceLane.REPLAY):
                incoming = sum(1 for _lease, _entry, chosen in entries if chosen == target)
                selected_sources = frozenset(
                    lease.source_identity for lease, _entry, _target in entries
                )
                if not self._lane_has_capacity_locked(
                    target,
                    incoming=incoming,
                    exclude_in_flight=selected_sources,
                ):
                    raise RuntimeError("ready-source target lane is full")
            if ready_event is None and lane is None and lane_by_source is None:
                for lease, entry, _target in entries:
                    self._in_flight.pop(lease.source_identity)
                    queue = (
                        self._fresh
                        if entry.lane == ReadySourceLane.FRESH
                        else self._replay
                    )
                    restored = replace(entry, reserved_return_lanes=frozenset())
                    queue.append(restored)
                    if restored.ready_latched:
                        self._adjust_ready_count(restored.lane, +1)
                self._fresh.sort(key=lambda entry: entry.enqueue_sequence)
                self._replay.sort(key=lambda entry: entry.enqueue_sequence)
                self._selected -= len(entries)
                return leases
            requeued = []
            for lease, entry, target in entries:
                event = _CompositeReadyEvent.extend(lease.ready_event, ready_event)
                replacement = replace(
                    entry,
                    lease=replace(lease, ready_event=event),
                    lane=target,
                    enqueue_sequence=self._enqueue_sequence,
                    reserved_return_lanes=frozenset(),
                    ready_latched=ready_event is None and entry.ready_latched,
                )
                self._enqueue_sequence += 1
                self._in_flight.pop(lease.source_identity)
                queue = (
                    self._fresh
                    if target == ReadySourceLane.FRESH
                    else self._replay
                )
                queue.append(replacement)
                if replacement.ready_latched:
                    self._adjust_ready_count(replacement.lane, +1)
                requeued.append(replacement.lease)
            return tuple(requeued)

    def _requeue_locked(
        self,
        lease: PreparedSourceLease,
        *,
        ready_event: Any,
        lane: str | None,
    ) -> PreparedSourceLease:
        """Requeue a single lease under the state lock."""
        if not isinstance(lease, PreparedSourceLease):
            raise TypeError("lease must be PreparedSourceLease")
        if ready_event is not None and not callable(getattr(ready_event, "query", None)):
            raise TypeError("ready_event must provide a callable query() method")
        entry = self._in_flight.get(lease.source_identity)
        if entry is None or entry.lease is not lease:
            raise ValueError("lease is not currently owned by the consumer")
        prior = self._generation_by_source.get(lease.source_identity)
        if (
            prior is None
            or prior.generation != lease.generation
            or prior.lifetime_token is not lease.lifetime_token
        ):
            raise ValueError("lease generation is stale")

        target_lane = entry.lane if lane is None else self._normalize_lane(lane)
        target_queue = self._fresh if target_lane == ReadySourceLane.FRESH else self._replay
        if not self._lane_has_capacity_locked(
            target_lane,
            exclude_in_flight=frozenset((lease.source_identity,)),
        ):
            raise RuntimeError("ready-source target lane is full")
        event = _CompositeReadyEvent.extend(lease.ready_event, ready_event)
        requeued = replace(lease, ready_event=event)
        replacement = replace(
            entry,
            lease=requeued,
            lane=target_lane,
            enqueue_sequence=self._enqueue_sequence,
            reserved_return_lanes=frozenset(),
            ready_latched=ready_event is None and entry.ready_latched,
        )
        self._enqueue_sequence += 1
        self._in_flight.pop(lease.source_identity)
        target_queue.append(replacement)
        if replacement.ready_latched:
            self._adjust_ready_count(replacement.lane, +1)
        return requeued

    def reserve_return_lanes(
        self,
        leases: tuple[PreparedSourceLease, ...],
        *,
        lane_by_source: Mapping[SourceIdentity, str],
    ) -> None:
        """Atomically reserve every possible post-dispatch destination lane.

        The original selected lane is always reserved.  ``lane_by_source``
        adds an alternate eventual lane (fresh-to-replay after a successful
        first supervision).  This operation happens before any GPU handoff;
        if it cannot preserve bounds, no transaction is issued.
        """

        self._reserve_return_lanes(
            threading.get_ident(), leases, lane_by_source=lane_by_source
        )

    def _reserve_return_lanes(
        self,
        consumer_owner: Any,
        leases: tuple[PreparedSourceLease, ...],
        *,
        lane_by_source: Mapping[SourceIdentity, str],
    ) -> None:
        """Reserve post-dispatch destination lanes with capacity validation."""
        self._claim("consumer", consumer_owner)
        if not isinstance(lane_by_source, Mapping):
            raise TypeError("lane_by_source must be a mapping")
        leases = tuple(leases)
        with self._state_lock:
            entries: list[tuple[PreparedSourceLease, _ReadyEntry, str]] = []
            for lease in leases:
                if not isinstance(lease, PreparedSourceLease):
                    raise TypeError("leases must contain PreparedSourceLease values")
                entry = self._in_flight.get(lease.source_identity)
                if entry is None or entry.lease is not lease:
                    raise ValueError("return-lane reservation contains stale or unowned lease")
                target = self._normalize_lane(lane_by_source.get(lease.source_identity, entry.lane))
                entries.append((lease, entry, target))
            proposed: dict[SourceIdentity, frozenset[str]] = {
                lease.source_identity: entry.reserved_return_lanes | frozenset((target,))
                for lease, entry, target in entries
            }
            for target in (ReadySourceLane.FRESH, ReadySourceLane.REPLAY):
                in_flight_claims = sum(
                    1
                    for source, entry in self._in_flight.items()
                    if target == entry.lane
                    or target in proposed.get(source, entry.reserved_return_lanes)
                )
                pending_and_producer = self._lane_occupancy_locked(
                    target,
                    exclude_in_flight=frozenset(self._in_flight),
                )
                limit = self._max_fresh if target == ReadySourceLane.FRESH else self._max_replay
                if pending_and_producer + in_flight_claims > limit:
                    raise RuntimeError("ready-source return lane capacity is unavailable")
            for lease, entry, _target in entries:
                self._in_flight[lease.source_identity] = replace(
                    entry,
                    reserved_return_lanes=proposed[lease.source_identity],
                )

    @property
    def pending_count(self) -> int:
        """Return total count of pending (queued but not yet selected) sources."""
        with self._state_lock:
            return len(self._fresh) + len(self._replay)

    @property
    def ready_count(self) -> int:
        """Return count of sources ready for selection."""
        with self._state_lock:
            return sum(
                self._ready(entry, record=False)
                for entry in (*self._fresh, *self._replay)
            )

    @property
    def fresh_ready_count(self) -> int:
        """O(1) latched READY count for the fresh lane."""

        with self._state_lock:
            return self._fresh_ready_count

    @property
    def replay_ready_count(self) -> int:
        """O(1) latched READY count for the replay lane."""

        with self._state_lock:
            return self._replay_ready_count

    @property
    def ready_bytes_current(self) -> int:
        """O(1) current resident bytes for all READY sources (fresh + replay + in-flight)."""
        with self._state_lock:
            return self._ready_bytes_current

    @property
    def ready_bytes_peak(self) -> int:
        """O(1) peak resident bytes ever observed."""
        with self._state_lock:
            return self._ready_bytes_peak

    @property
    def ready_bytes_by_tier_kind(self) -> dict[tuple[str, str], int]:
        """Copy of current bytes breakdown by (tier, payload_kind)."""
        with self._state_lock:
            return dict(self._ready_bytes_by_tier_kind)

    @property
    def ready_bytes_by_lane(self) -> dict[str, int]:
        """Copy of current bytes breakdown by lane (fresh or replay)."""
        with self._state_lock:
            return dict(self._ready_bytes_by_lane)

    @property
    def replay_reserve_depth(self) -> int:
        """Return current number of reserved replay lanes."""
        with self._state_lock:
            return self._replay_ready_count

    @property
    def reserve_underflow_count(self) -> int:
        """Return count of reserve underflow occurrences."""
        with self._state_lock:
            return self._reserve_underflow_count

    @property
    def reserve_underflow_fault(self) -> bool:
        """Return whether any reserve underflow has occurred."""
        with self._state_lock:
            return self._reserve_underflow_count > 0

    @property
    def replay_reserve_target(self) -> int:
        """Return the configured replay reserve capacity."""
        return self._replay_reserve

    @property
    def generation_history_depth(self) -> int:
        """Return current generation history entry count."""
        with self._state_lock:
            return len(self._generation_by_source)

    @property
    def generation_history_limit(self) -> int:
        """Return maximum generation history entries."""
        return self._generation_history_limit

    @property
    def pending_leases(self) -> tuple[PreparedSourceLease, ...]:
        """Return all currently pending (queued) leases."""
        with self._state_lock:
            return tuple(entry.lease for entry in (*self._fresh, *self._replay))

    @property
    def replay_capacity(self) -> int:
        """Fixed bounded replay-residency capacity (metadata-only read)."""

        return self._max_replay

    @property
    def terminal_records(self) -> tuple[ReadySourceTerminalRecord, ...]:
        """Return all terminal records for retired sources."""
        with self._state_lock:
            return tuple(self._terminal_records.values())

    @property
    def counters(self) -> ReadySourceCounters:
        """Return a snapshot of all diagnostic counters."""
        with self._state_lock:
            return ReadySourceCounters(
                fresh_pending=len(self._fresh),
                replay_pending=len(self._replay),
                in_flight=len(self._in_flight),
                offered=self._offered,
                selected=self._selected,
                released=self._released,
                full_drop_count=self._full_drop_count,
                full_not_resident_count=self._full_not_resident_count,
                duplicate_drop_count=self._duplicate_drop_count,
                stale_generation_drop_count=self._stale_generation_drop_count,
                not_ready_count=self._not_ready_count,
                ready_error_count=self._ready_error_count,
                reserve_underflow_count=self._reserve_underflow_count,
                fresh_ready_count=self._fresh_ready_count,
                replay_ready_count=self._replay_ready_count,
                ready_bytes_current=self._ready_bytes_current,
                ready_bytes_peak=self._ready_bytes_peak,
                replay_evicted_bytes=self._replay_evicted_bytes,
            )

    @property
    def not_ready_count(self) -> int:
        """Return count of offers rejected due to readiness-check failures."""
        with self._state_lock:
            return self._not_ready_count

    @property
    def duplicate_drop_count(self) -> int:
        """Return count of duplicate-source drops."""
        with self._state_lock:
            return self._duplicate_drop_count

    @property
    def full_drop_count(self) -> int:
        """Return count of full-lane drops."""
        with self._state_lock:
            return self._full_drop_count

    @property
    def full_not_resident_count(self) -> int:
        """Count cache-pressure offers retained outside READY residency."""

        with self._state_lock:
            return self._full_not_resident_count

    @property
    def stale_generation_drop_count(self) -> int:
        """Return count of stale-generation drops."""
        with self._state_lock:
            return self._stale_generation_drop_count


class ReadySourceProducer:
    """Exclusive source-preparation producer handle."""

    def __init__(self, owner: ReadySourceOwner, token: Any) -> None:
        self._owner = owner
        self._token = token

    def offer(
        self,
        lease: PreparedSourceLease,
        *,
        lane: str = ReadySourceLane.REPLAY,
        arrival_ordinal: int | None = None,
        priority: int | str | ReplayPriority | None = None,
        full_is_nonterminal: bool = False,
    ) -> bool:
        """Offer a prepared lease to the specified lane."""
        return self._owner._offer(
            self._token,
            lease,
            lane=lane,
            arrival_ordinal=arrival_ordinal,
            priority=priority,
            full_is_nonterminal=full_is_nonterminal,
        )

    def reset_generation_history(self, *, next_stream_epoch: int) -> None:
        """Start a new source-identity epoch explicitly when idle."""
        self._owner._reset_generation_history(
            self._token,
            next_stream_epoch=next_stream_epoch,
        )

    def abandon_pending(self, *, reason: str) -> None:
        """Abandon all pending sources with the given terminal reason."""
        if not isinstance(reason, str) or not reason:
            raise ValueError("reason must be non-empty")
        self._owner._abandon_pending(self._token, reason=reason)

    def reserve(
        self,
        source: SourceIdentity,
        *,
        lane: str = ReadySourceLane.REPLAY,
        arrival_ordinal: int | None = None,
        priority: int | str | ReplayPriority | None = None,
        generation: int = 0,
        lifetime_token: Any,
        full_is_nonterminal: bool = False,
        control_token: Any = None,
    ) -> ReadySourceReservation | None:
        """Reserve bounded READY residency before source H2D/preparation."""

        return self._owner._reserve(
            self._token,
            source,
            lane=lane,
            arrival_ordinal=arrival_ordinal,
            priority=priority,
            generation=generation,
            lifetime_token=lifetime_token,
            full_is_nonterminal=full_is_nonterminal,
            control_token=control_token,
        )


class ReadySourceConsumer:
    """Exclusive GPU-consumer handle."""

    def __init__(self, owner: ReadySourceOwner, token: Any) -> None:
        self._owner = owner
        self._token = token

    def poll(self, lane: str | None = None) -> PreparedSourceLease | None:
        """Poll for one READY lease without changing selection order."""
        return self._owner._poll(self._token, lane)

    def select(
        self,
        batch_size: int,
        *,
        fresh_limit: int | None = None,
        fresh_sources: frozenset[SourceIdentity] | None = None,
        replay_sources: frozenset[SourceIdentity] | None = None,
        exact_sources: tuple[SourceIdentity, ...] | None = None,
    ) -> tuple[PreparedSourceLease, ...]:
        """Select a ready fresh-first batch without waiting."""
        return self._owner._select(
            self._token,
            batch_size,
            fresh_limit=fresh_limit,
            fresh_sources=fresh_sources,
            replay_sources=replay_sources,
            exact_sources=exact_sources,
        )

    def release(self, lease: PreparedSourceLease) -> None:
        """Release a lease selected by this consumer thread."""
        self._owner._release(self._token, lease)

    def requeue(
        self, lease: PreparedSourceLease, *, ready_event: Any, lane: str | None = None
    ) -> PreparedSourceLease:
        """Return an in-flight lease, optionally changing its lane."""
        return self._owner._requeue(
            self._token, lease, ready_event=ready_event, lane=lane
        )

    def evict_replay(
        self,
        *,
        count: int = 1,
        protected_sources: frozenset[SourceIdentity] = frozenset(),
        target_source: SourceIdentity | None = None,
    ) -> tuple[PreparedSourceLease, ...]:
        """Reclaim bounded idle replay residency for a scheduler page-in."""

        return self._owner._evict_replay(
            self._token,
            count=count,
            protected_sources=protected_sources,
            target_source=target_source,
        )

    def requeue_batch(
        self,
        leases: tuple[PreparedSourceLease, ...],
        *,
        ready_event: Any,
        lane: str | None = None,
        lane_by_source: Mapping[SourceIdentity, str] | None = None,
    ) -> tuple[PreparedSourceLease, ...]:
        """Return a complete selected batch through this consumer owner."""

        return self._owner._requeue_batch(
            self._token,
            leases,
            ready_event=ready_event,
            lane=lane,
            lane_by_source=lane_by_source,
        )

    def reserve_return_lanes(
        self,
        leases: tuple[PreparedSourceLease, ...],
        *,
        lane_by_source: Mapping[SourceIdentity, str],
    ) -> None:
        """Reserve post-dispatch lanes through the exclusive GPU handle."""

        self._owner._reserve_return_lanes(
            self._token,
            leases,
            lane_by_source=lane_by_source,
        )


class ReadySourceScheduler:
    """Bounded page-in/eviction handle; intentionally not a B4 consumer."""

    def __init__(self, owner: ReadySourceOwner, token: Any) -> None:
        self._owner = owner
        self._token = token

    def evict_replay(
        self,
        *,
        count: int = 1,
        protected_sources: frozenset[SourceIdentity] = frozenset(),
    ) -> tuple[PreparedSourceLease, ...]:
        """Reclaim idle replay without invalidating a frozen exact B4."""

        return self._owner._scheduler_evict_replay(
            self._token,
            count=count,
            protected_sources=protected_sources,
        )

    def evict_fresh(
        self, *, protected_sources: frozenset[SourceIdentity]
    ) -> tuple[PreparedSourceLease, ...]:
        """Reclaim one idle fresh lease without invalidating an exact B4."""

        return self._owner._scheduler_evict_fresh(
            self._token, protected_sources=protected_sources
        )

    def has_source(self, source: SourceIdentity, *, ready: bool = False) -> bool:
        """Return whether identity exists, or whether it is READY when requested."""

        return self._owner._scheduler_has_source(self._token, source, ready=ready)

    def next_generation(self, source: SourceIdentity) -> int:
        """Return the next safe prepared generation for a durable source."""

        return self._owner._scheduler_next_generation(self._token, source)

    @property
    def replay_pending(self) -> int:
        """Return count of queued (not yet selected) replay sources."""
        return self._owner.counters.replay_pending

    @property
    def replay_residency_depth(self) -> int:
        """Return count of in-device replay residency."""
        return self._owner._scheduler_replay_residency_depth(self._token)

    @property
    def replay_capacity(self) -> int:
        """Return fixed bounded replay-residency capacity."""
        return self._owner.replay_capacity


SourceReadyRegistry = ReadySourceOwner


__all__ = [
    "ProvisionalReplayEviction",
    "ReadySourceConsumer",
    "ReadySourceCounters",
    "ReadySourceLane",
    "ReadySourceOwner",
    "ReadySourceProducer",
    "ReadySourceReservation",
    "ReadySourceScheduler",
    "ReadySourceTerminalRecord",
    "ReplayPriority",
    "SourceReadyRegistry",
]
