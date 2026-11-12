from __future__ import annotations

import dataclasses
import threading

import pytest
import torch

from eligsir.contracts import (
    BufferOwnership,
    BufferRef,
    BufferSpec,
    MemoryDomain,
    ModalityEligibility,
    SourceIdentity,
)
from eligsir.runtime.prepared_source import PreparedSourceLease
import eligsir.runtime.ready_source as ready_source_module
from eligsir.runtime.ready_source import ReadySourceLane, ReadySourceOwner, ReplayPriority


class Gate:
    def __init__(self, ready: bool = False) -> None:
        self.ready = ready
        self.queries = 0

    def query(self) -> bool:
        self.queries += 1
        return self.ready


class BrokenGate:
    def query(self) -> bool:
        raise RuntimeError("synthetic readiness failure")


class HandoffGate(Gate):
    def __init__(self, name: str) -> None:
        super().__init__(True)
        self.name = name

    def handoff_to_compute(self, owner) -> None:
        owner.handoffs.append(self.name)


class HandoffOwner:
    def __init__(self) -> None:
        self.handoffs: list[object] = []

    def handoff_to_compute(self, event) -> None:
        self.handoffs.append(event)


class RecordingLock:
    """Lock probe used to assert producer/consumer operation overlap."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.offer_entered = threading.Event()

    def __enter__(self):
        self._lock.acquire()
        if threading.current_thread().name == "offer":
            self.offer_entered.set()
        return self

    def __exit__(self, *_args) -> None:
        self._lock.release()


class BlockingReplaceList(list):
    def __init__(self, values, entered: threading.Event, resume: threading.Event) -> None:
        super().__init__(values)
        self._entered = entered
        self._resume = resume

    def __setitem__(self, index, value):
        if isinstance(index, slice):
            self._entered.set()
            self._resume.wait(timeout=1)
        super().__setitem__(index, value)


class BlockingPopDict(dict):
    def __init__(self, values, entered: threading.Event, resume: threading.Event) -> None:
        super().__init__(values)
        self._entered = entered
        self._resume = resume

    def pop(self, *args):
        self._entered.set()
        self._resume.wait(timeout=1)
        return super().pop(*args)


def identity(ordinal: int, stream_epoch: int = 0) -> SourceIdentity:
    return SourceIdentity("camera", stream_epoch, ordinal, ordinal * 1_000)


def lease(
    ordinal: int,
    *,
    event=None,
    generation: int = 0,
    lifetime_token=None,
    stream_epoch: int = 0,
) -> PreparedSourceLease:
    payload_token = lifetime_token if lifetime_token is not None else object()
    payload = BufferRef(
        data=torch.zeros(2),
        spec=BufferSpec((2,), "float32", MemoryDomain.OTHER_DEVICE, "tensor", device="cpu"),
        ownership=BufferOwnership.BORROWED,
        owner_token=payload_token,
    )
    return PreparedSourceLease(
        source_identity=identity(ordinal, stream_epoch),
        modality_eligibility=ModalityEligibility(rgb_supervision=True),
        source_metadata={"width": 8, "height": 6},
        tier_preparation_available=("native",),
        borrowed_device_payloads={"rgb": payload},
        device="cpu",
        ready_event=event,
        generation=generation,
        lifetime_token=lifetime_token,
    )


def test_delayed_event_is_not_dequeued_and_ready_replay_fills_selection():
    gate = Gate()
    owner = ReadySourceOwner(max_fresh=2, max_replay=2, replay_reserve=1)
    producer = owner.claim_producer("decode")
    consumer = owner.claim_consumer("gpu")
    delayed = lease(8, event=gate)
    replay = lease(2)
    assert producer.offer(delayed, lane="fresh", arrival_ordinal=8)
    assert producer.offer(replay, lane="replay", arrival_ordinal=2)

    selected = consumer.select(batch_size=1)

    assert selected == (replay,)
    assert owner.pending_count == 1
    assert owner.not_ready_count >= 1
    assert delayed in owner.pending_leases

    consumer.release(replay)
    gate.ready = True
    assert consumer.select(batch_size=1) == (delayed,)


def test_latched_replay_readiness_avoids_a_second_event_sweep():
    owner = ReadySourceOwner(max_fresh=1, max_replay=8, replay_reserve=4)
    producer = owner.claim_producer("decode")
    consumer = owner.claim_consumer("gpu")
    gates = tuple(Gate(True) for _ in range(8))
    for ordinal, gate in enumerate(gates):
        assert producer.offer(lease(ordinal, event=gate), lane=ReadySourceLane.REPLAY)

    selected = consumer.select(4, fresh_limit=0)
    assert owner.replay_reserve_depth == 4
    first_queries = sum(gate.queries for gate in gates)
    assert first_queries == 8
    consumer.requeue_batch(selected, ready_event=None)

    assert consumer.select(4, fresh_limit=0)
    assert sum(gate.queries for gate in gates) == first_queries


def test_ready_replay_sources_is_read_only_and_excludes_pending_leases():
    owner = ReadySourceOwner(max_fresh=1, max_replay=3)
    producer = owner.claim_producer("decode")
    pending = Gate(False)
    assert producer.offer(lease(3, event=pending), lane="replay", arrival_ordinal=3)
    assert producer.offer(lease(2), lane="replay", arrival_ordinal=2)
    assert producer.offer(lease(1), lane="replay", arrival_ordinal=1)

    assert owner.ready_replay_sources() == (identity(2), identity(1))
    assert tuple(item.source_identity for item in owner.pending_leases) == (
        identity(3),
        identity(2),
        identity(1),
    )
    assert owner.counters.selected == 0


def test_replay_resident_sources_include_selected_return_claims():
    """A scheduler may author against replay that is only temporarily in flight."""

    owner = ReadySourceOwner(max_fresh=1, max_replay=4)
    producer = owner.claim_producer("decode")
    for ordinal in range(1, 5):
        assert producer.offer(lease(ordinal), lane=ReadySourceLane.REPLAY)

    selected = owner.select(batch_size=4, fresh_limit=0)
    owner.reserve_return_lanes(
        selected,
        lane_by_source={item.source_identity: ReadySourceLane.REPLAY for item in selected},
    )

    assert owner.replay_resident_sources() == tuple(
        identity(ordinal) for ordinal in range(1, 5)
    )


def test_scheduler_distinguishes_present_source_from_ready_source():
    owner = ReadySourceOwner(max_fresh=1, max_replay=1)
    producer = owner.claim_producer("decode")
    scheduler = owner.claim_scheduler("scheduler")
    gate = Gate(False)
    source = lease(3, event=gate)
    assert producer.offer(source, lane="fresh", arrival_ordinal=3)

    assert scheduler.has_source(source.source_identity)
    assert not scheduler.has_source(source.source_identity, ready=True)

    gate.ready = True
    assert scheduler.has_source(source.source_identity, ready=True)


def test_immediate_partial_requeues_do_not_nest_ready_fences():
    """A bootstrap retry must preserve, not recursively wrap, readiness."""

    owner = ReadySourceOwner(max_fresh=1, max_replay=1)
    producer = owner.claim_producer("decode")
    consumer = owner.claim_consumer("gpu")
    source = lease(1, event=Gate(True))
    assert producer.offer(source, lane="fresh", arrival_ordinal=1)

    for _ in range(1_100):
        selected = consumer.select(batch_size=1)
        assert len(selected) == 1
        consumer.requeue_batch(selected, ready_event=None)

    assert consumer.select(batch_size=1)[0].source_identity == source.source_identity
    assert owner.counters.ready_error_count == 0


def test_completion_order_does_not_change_arrival_order():
    owner = ReadySourceOwner(max_fresh=4, max_replay=1)
    producer = owner.claim_producer("decode")
    consumer = owner.claim_consumer("gpu")
    third = lease(30)
    first = lease(10)
    second = lease(20)
    for item in (third, first, second):
        assert producer.offer(item, lane="fresh", arrival_ordinal=item.source_identity.source_ordinal)

    assert tuple(item.source_identity.source_ordinal for item in consumer.select(3)) == (10, 20, 30)


def test_fresh_precedes_replay_and_replay_priority_is_deterministic():
    owner = ReadySourceOwner(max_fresh=2, max_replay=2)
    producer = owner.claim_producer("decode")
    consumer = owner.claim_consumer("gpu")
    fresh = lease(99)
    demanded = lease(20)
    normal = lease(1)
    assert producer.offer(normal, lane=ReadySourceLane.REPLAY, arrival_ordinal=1)
    assert producer.offer(demanded, lane="replay", arrival_ordinal=20, priority="demanded_replay")
    assert producer.offer(fresh, lane="fresh", arrival_ordinal=99)

    selected = consumer.select(batch_size=3)

    assert selected == (fresh, demanded, normal)


def test_exact_selection_of_an_incomplete_b4_is_residency_and_order_neutral():
    """A 3/4 frozen plan must not churn READY just to discover it is incomplete."""

    owner = ReadySourceOwner(max_fresh=4, max_replay=4)
    producer = owner.claim_producer("decode")
    consumer = owner.claim_consumer("gpu")
    sources = tuple(lease(ordinal) for ordinal in (10, 20, 30, 40))
    for item in sources[:3]:
        assert producer.offer(item, lane=ReadySourceLane.REPLAY)

    before = owner.pending_leases
    counters = owner.counters
    for _ in range(2_000):
        assert consumer.select(
            4,
            fresh_sources=frozenset(),
            replay_sources=frozenset(item.source_identity for item in sources),
            exact_sources=tuple(item.source_identity for item in sources),
        ) == ()

    assert owner.pending_leases == before
    assert owner.counters == counters

    assert producer.offer(sources[3], lane=ReadySourceLane.REPLAY)
    assert consumer.select(
        4,
        fresh_sources=frozenset(),
        replay_sources=frozenset(item.source_identity for item in sources),
        exact_sources=tuple(item.source_identity for item in sources),
    ) == sources


def test_exact_readiness_classifies_missing_pending_and_error_without_selection():
    """A trace probe must explain a frozen B4 without changing READY ownership."""

    owner = ReadySourceOwner(max_fresh=1, max_replay=4)
    producer = owner.claim_producer("decode")
    fresh = lease(10, event=Gate(False))
    ready = lease(20)
    failed = lease(30, event=BrokenGate())
    missing = lease(40)
    assert producer.offer(fresh, lane=ReadySourceLane.FRESH)
    assert producer.offer(ready, lane=ReadySourceLane.REPLAY)
    assert producer.offer(failed, lane=ReadySourceLane.REPLAY)

    before = owner.pending_leases

    assert owner.exact_readiness(
        fresh_sources=frozenset((fresh.source_identity,)),
        replay_sources=frozenset(
            (ready.source_identity, failed.source_identity, missing.source_identity)
        ),
        exact_sources=(
            fresh.source_identity,
            ready.source_identity,
            failed.source_identity,
            missing.source_identity,
        ),
    ) == (
        (fresh.source_identity, "EVENT_PENDING"),
        (ready.source_identity, "READY"),
        (failed.source_identity, "EVENT_QUERY_ERROR"),
        (missing.source_identity, "ABSENT_REPLAY"),
    )


def test_exact_readiness_latches_and_stops_requerying_after_first_ready():
    """Issue #1575A: a candidate repeatedly inspected via exact_readiness()
    (protection/telemetry probes) but not yet selected must not re-query an
    already-completed event on every call once it transitions READY -- the
    same monotonic latch _select()/_ready() already give a real selection."""

    owner = ReadySourceOwner(max_fresh=1, max_replay=4)
    producer = owner.claim_producer("decode")
    gate = Gate(False)
    replay = lease(50, event=gate)
    assert producer.offer(replay, lane=ReadySourceLane.REPLAY)

    probe = dict(
        fresh_sources=frozenset(),
        replay_sources=frozenset((replay.source_identity,)),
        exact_sources=(replay.source_identity,),
    )
    assert owner.exact_readiness(**probe) == ((replay.source_identity, "EVENT_PENDING"),)
    assert gate.queries == 1
    assert owner.replay_reserve_depth == 0

    gate.ready = True
    assert owner.exact_readiness(**probe) == ((replay.source_identity, "READY"),)
    assert gate.queries == 2
    assert owner.replay_reserve_depth == 1

    for _ in range(1000):
        assert owner.exact_readiness(**probe) == ((replay.source_identity, "READY"),)

    assert gate.queries == 2
    assert owner.replay_reserve_depth == 1


def test_noncomputing_exact_selection_restores_replay_fifo_and_counters():
    owner = ReadySourceOwner(max_fresh=1, max_replay=4)
    producer = owner.claim_producer("decode")
    consumer = owner.claim_consumer("gpu")
    sources = tuple(lease(ordinal) for ordinal in (10, 20, 30, 40))
    for item in sources:
        assert producer.offer(item, lane=ReadySourceLane.REPLAY)

    before = owner.counters
    selected = consumer.select(
        4,
        fresh_sources=frozenset(),
        replay_sources=frozenset(item.source_identity for item in sources),
        exact_sources=tuple(item.source_identity for item in reversed(sources)),
    )
    consumer.requeue_batch(selected, ready_event=None)

    assert owner.counters == before
    assert consumer.select(4) == sources


def test_explicit_lane_poll_never_falls_back_to_the_other_lane():
    owner = ReadySourceOwner(max_fresh=1, max_replay=1)
    producer = owner.claim_producer("decode")
    consumer = owner.claim_consumer("gpu")
    replay = lease(2)
    assert producer.offer(replay, lane="replay", arrival_ordinal=2)

    assert consumer.poll(lane="fresh") is None
    assert consumer.poll(lane="replay") is replay


def test_lane_bounds_and_duplicate_identity_are_nonblocking_drops():
    owner = ReadySourceOwner(max_fresh=1, max_replay=1)
    producer = owner.claim_producer("decode")
    first = lease(1)
    duplicate = dataclasses.replace(first)
    assert producer.offer(first, lane="fresh", arrival_ordinal=1)
    assert not producer.offer(duplicate, lane="fresh", arrival_ordinal=1)
    assert not producer.offer(lease(2), lane="fresh", arrival_ordinal=2)
    assert owner.duplicate_drop_count == 1
    assert owner.full_drop_count == 1


def test_in_flight_fresh_selection_reserves_its_lane_until_requeue():
    """Producer pressure cannot make a later same-lane return exceed capacity."""

    owner = ReadySourceOwner(max_fresh=4, max_replay=4)
    producer = owner.claim_producer("decode")
    consumer = owner.claim_consumer("gpu")
    selected_source = tuple(lease(ordinal) for ordinal in range(4))
    for item in selected_source:
        assert producer.offer(item, lane=ReadySourceLane.FRESH)

    assert consumer.select(4) == selected_source
    for ordinal in range(4, 8):
        assert not producer.offer(lease(ordinal), lane=ReadySourceLane.FRESH)

    assert owner.counters.fresh_pending == 0
    assert owner.counters.in_flight == 4
    assert consumer.requeue_batch(selected_source, ready_event=Gate(True))
    assert owner.counters.fresh_pending == 4
    assert owner.counters.in_flight == 0
    assert owner.pending_count == 4


def test_full_drop_does_not_poison_a_source_generation_for_a_later_retry():
    owner = ReadySourceOwner(max_fresh=1, max_replay=1)
    producer = owner.claim_producer("decode")
    assert producer.offer(lease(1), lane="fresh", arrival_ordinal=1)
    blocked = lease(2, generation=1, lifetime_token=object())
    assert not producer.offer(blocked, lane="fresh", arrival_ordinal=2)

    consumer = owner.claim_consumer("gpu")
    selected = consumer.poll(lane="fresh")
    assert selected is not None
    consumer.release(selected)
    retried = lease(2, generation=1, lifetime_token=object())
    assert producer.offer(retried, lane="fresh", arrival_ordinal=2)


def test_producer_and_consumer_claims_are_exclusive():
    owner = ReadySourceOwner(max_fresh=1, max_replay=1)
    owner.claim_producer("decode")
    owner.claim_consumer("gpu")
    with pytest.raises(RuntimeError, match="producer"):
        owner.claim_producer("other-decode")
    with pytest.raises(RuntimeError, match="consumer"):
        owner.claim_consumer("other-gpu")


def test_generation_rejects_stale_replacement_but_allows_same_lease_reuse():
    owner = ReadySourceOwner(max_fresh=2, max_replay=2)
    producer = owner.claim_producer("decode")
    consumer = owner.claim_consumer("gpu")
    token = object()
    old = lease(4, generation=1, lifetime_token=token)
    newer = lease(4, generation=2, lifetime_token=object())
    stale = lease(4, generation=1, lifetime_token=object())
    assert producer.offer(old, lane="replay", arrival_ordinal=4)
    checked_out = consumer.poll()
    assert checked_out is old
    consumer.release(old)
    assert producer.offer(old, lane="replay", arrival_ordinal=4)
    assert not producer.offer(newer, lane="replay", arrival_ordinal=4)
    assert owner.stale_generation_drop_count == 0
    consumer.poll()
    consumer.release(old)
    assert producer.offer(newer, lane="replay", arrival_ordinal=4)
    assert not producer.offer(stale, lane="replay", arrival_ordinal=4)
    assert owner.stale_generation_drop_count == 1


def test_generation_identity_rejects_deterministic_aba_token_id_reuse(monkeypatch):
    owner = ReadySourceOwner(max_fresh=2, max_replay=2)
    producer = owner.claim_producer("decode")
    consumer = owner.claim_consumer("gpu")
    monkeypatch.setattr(ready_source_module, "id", lambda _token: 17, raising=False)
    first = lease(4, generation=1, lifetime_token=object())
    replacement = lease(4, generation=1, lifetime_token=object())
    assert producer.offer(first, lane="replay", arrival_ordinal=4)
    assert consumer.poll() is first
    consumer.release(first)

    assert not producer.offer(replacement, lane="replay", arrival_ordinal=4)
    assert owner.stale_generation_drop_count == 1


def test_generation_history_is_bounded_after_replay_leases_are_released():
    owner = ReadySourceOwner(max_fresh=1, max_replay=1)
    producer = owner.claim_producer("decode")
    consumer = owner.claim_consumer("gpu")
    for ordinal in range(20):
        item = lease(ordinal)
        assert producer.offer(item, lane="replay", arrival_ordinal=ordinal)
        assert consumer.poll() is item
        consumer.release(item)

    assert owner.generation_history_depth <= owner.generation_history_limit


def test_released_source_cannot_reenter_after_history_pressure_without_reset():
    owner = ReadySourceOwner(max_fresh=1, max_replay=1)
    producer = owner.claim_producer("decode")
    consumer = owner.claim_consumer("gpu")
    retired = lease(0)
    assert producer.offer(retired, lane="replay", arrival_ordinal=0)
    assert consumer.poll() is retired
    consumer.release(retired)
    for ordinal in range(1, 20):
        item = lease(ordinal)
        assert producer.offer(item, lane="replay", arrival_ordinal=ordinal)
        assert consumer.poll() is item
        consumer.release(item)

    assert not producer.offer(retired, lane="replay", arrival_ordinal=0)
    assert owner.stale_generation_drop_count >= 1


def test_scheduler_pagein_reenters_durable_source_after_history_pressure():
    owner = ReadySourceOwner(max_fresh=1, max_replay=1)
    producer = owner.claim_producer("decode")
    consumer = owner.claim_consumer("gpu")
    scheduler = owner.claim_scheduler("scheduler")
    durable = identity(0)
    original = lease(0)
    assert producer.offer(original, lane="replay", arrival_ordinal=0)
    assert consumer.poll() is original
    consumer.release(original)
    for ordinal in range(1, 20):
        item = lease(ordinal)
        assert producer.offer(item, lane="replay", arrival_ordinal=ordinal)
        assert consumer.poll() is item
        consumer.release(item)

    generation = scheduler.next_generation(durable)
    assert producer.offer(
        lease(0, generation=generation), lane="replay", arrival_ordinal=0
    )


def test_generation_history_reset_requires_idle_owner_and_is_explicit():
    owner = ReadySourceOwner(max_fresh=1, max_replay=1)
    producer = owner.claim_producer("decode")
    consumer = owner.claim_consumer("gpu")
    item = lease(0)
    assert producer.offer(item, lane="replay", arrival_ordinal=0)
    with pytest.raises(RuntimeError, match="idle"):
        producer.reset_generation_history(next_stream_epoch=1)
    assert consumer.poll() is item
    consumer.release(item)
    producer.reset_generation_history(next_stream_epoch=1)
    assert not producer.offer(item, lane="replay", arrival_ordinal=0)
    assert producer.offer(lease(0, stream_epoch=1), lane="replay", arrival_ordinal=0)


def test_selection_reports_bounded_replay_reserve_underflow_without_waiting():
    owner = ReadySourceOwner(max_fresh=1, max_replay=2, replay_reserve=2)
    producer = owner.claim_producer("decode")
    consumer = owner.claim_consumer("gpu")
    assert producer.offer(lease(1), lane="replay", arrival_ordinal=1)

    selected = consumer.select(batch_size=2)

    assert len(selected) == 1
    assert owner.reserve_underflow_count == 1
    assert owner.reserve_underflow_fault


def test_consumer_eviction_reclaims_only_idle_replay_residency():
    """A scheduler-owned page-in may reclaim an idle replay lease, never B4 work."""

    owner = ReadySourceOwner(max_fresh=1, max_replay=2)
    producer = owner.claim_producer("decode")
    consumer = owner.claim_consumer("gpu")
    first, second = lease(1), lease(2)
    assert producer.offer(first, lane="replay", arrival_ordinal=1)
    assert producer.offer(second, lane="replay", arrival_ordinal=2)

    selected = consumer.poll(lane="replay")
    assert selected is first
    # The selected source is in flight and cannot be reclaimed.  The next
    # deterministic idle replay entry is evicted instead.
    assert consumer.evict_replay(count=1) == (second,)
    assert owner.pending_leases == ()
    assert owner.counters.in_flight == 1

    assert consumer.evict_replay(count=1) == ()
    consumer.release(selected)


def test_consumer_replay_eviction_protects_exact_sources_and_allows_new_generation_reload():
    """A completed turnover removes only its nominated idle replay payload."""

    owner = ReadySourceOwner(max_fresh=1, max_replay=3)
    producer = owner.claim_producer("decode")
    consumer = owner.claim_consumer("gpu")
    scheduler = owner.claim_scheduler("scheduler")
    in_flight, victim, protected = lease(1), lease(2), lease(3)
    for item in (in_flight, victim, protected):
        assert producer.offer(
            item, lane="replay", arrival_ordinal=item.source_identity.source_ordinal
        )
    assert consumer.poll(lane="replay") is in_flight
    bytes_before = owner.ready_bytes_current

    assert consumer.evict_replay(
        protected_sources=frozenset((protected.source_identity,))
    ) == (victim,)
    assert owner.ready_bytes_current < bytes_before
    assert owner.pending_leases == (protected,)
    assert owner.terminal_records[-1].source == victim.source_identity
    assert not producer.offer(victim, lane="replay", arrival_ordinal=2)
    reloaded = lease(2, generation=scheduler.next_generation(victim.source_identity))
    assert producer.offer(
        reloaded,
        lane="replay",
        arrival_ordinal=2,
    )


def test_targeted_replay_eviction_removes_only_the_nominated_idle_source():
    owner = ReadySourceOwner(max_fresh=1, max_replay=3)
    producer = owner.claim_producer("decode")
    consumer = owner.claim_consumer("gpu")
    earlier_unready = lease(1, event=Gate())
    victim = lease(2)
    later = lease(3)
    for item in (earlier_unready, victim, later):
        assert producer.offer(
            item, lane="replay", arrival_ordinal=item.source_identity.source_ordinal
        )
    before_bytes = owner.ready_bytes_current

    assert consumer.evict_replay(target_source=victim.source_identity) == (victim,)
    assert owner.pending_leases == (earlier_unready, later)
    assert owner.counters.replay_evicted_bytes == before_bytes - owner.ready_bytes_current
    assert not producer.offer(victim, lane="replay", arrival_ordinal=2)
    scheduler = owner.claim_scheduler("scheduler")
    assert producer.offer(
        lease(2, generation=scheduler.next_generation(victim.source_identity)),
        lane="replay",
        arrival_ordinal=2,
    )


def test_targeted_replay_eviction_never_substitutes_or_touches_inflight_sources():
    owner = ReadySourceOwner(max_fresh=1, max_replay=2)
    producer = owner.claim_producer("decode")
    consumer = owner.claim_consumer("gpu")
    in_flight, idle = lease(1), lease(2)
    for item in (in_flight, idle):
        assert producer.offer(
            item, lane="replay", arrival_ordinal=item.source_identity.source_ordinal
        )
    assert consumer.poll(lane="replay") is in_flight
    consumer.reserve_return_lanes(
        (in_flight,), lane_by_source={in_flight.source_identity: ReadySourceLane.REPLAY}
    )
    before = owner.pending_leases

    assert consumer.evict_replay(target_source=identity(99)) == ()
    assert consumer.evict_replay(
        target_source=idle.source_identity,
        protected_sources=frozenset((idle.source_identity,)),
    ) == ()
    assert consumer.evict_replay(target_source=in_flight.source_identity) == ()
    with pytest.raises(TypeError, match="target_source"):
        consumer.evict_replay(target_source="not-a-source")
    assert owner.pending_leases == before
    assert owner.counters.replay_evicted_bytes == 0


def test_owner_replay_eviction_accepts_protected_sources():
    owner = ReadySourceOwner(max_fresh=1, max_replay=2)
    producer = owner.claim_producer("decode")
    victim, protected = lease(1), lease(2)
    assert producer.offer(victim, lane="replay", arrival_ordinal=1)
    assert producer.offer(protected, lane="replay", arrival_ordinal=2)

    assert owner.evict_replay(
        target_source=victim.source_identity,
        protected_sources=frozenset((protected.source_identity,)),
    ) == (victim,)


def test_provisional_replay_eviction_restores_exact_identity_and_order_on_rollback():
    """A rolled-back provisional eviction must reconstruct the pre-eviction queue."""

    owner = ReadySourceOwner(max_fresh=1, max_replay=3)
    producer = owner.claim_producer("decode")
    first, second, third = lease(1), lease(2), lease(3)
    for item in (first, second, third):
        assert producer.offer(
            item, lane="replay", arrival_ordinal=item.source_identity.source_ordinal
        )

    before = owner.pending_leases
    evictions = owner.evict_replay_provisional(count=1)
    assert tuple(eviction.lease for eviction in evictions) == (first,)
    assert owner.pending_leases == (second, third)
    assert owner.terminal_records == ()

    owner.resolve_replay_eviction(evictions, commit=False)

    assert owner.pending_leases == before
    assert owner.terminal_records == ()


def test_provisional_replay_eviction_commit_retires_exactly_like_evict_replay():
    """A committed provisional eviction must finalize identically to evict_replay."""

    owner = ReadySourceOwner(max_fresh=1, max_replay=3)
    producer = owner.claim_producer("decode")
    first, second = lease(1), lease(2)
    for item in (first, second):
        assert producer.offer(
            item, lane="replay", arrival_ordinal=item.source_identity.source_ordinal
        )

    evictions = owner.evict_replay_provisional(count=1)
    assert tuple(eviction.lease for eviction in evictions) == (first,)

    owner.resolve_replay_eviction(evictions, commit=True)

    assert owner.pending_leases == (second,)
    records = owner.terminal_records
    assert len(records) == 1
    assert records[0].source == first.source_identity
    assert records[0].reason == "evicted_replay_residency"


def test_provisional_replay_eviction_rollback_never_resurrects_a_superseded_generation():
    """A strictly newer generation offered during the rollback window must win."""

    owner = ReadySourceOwner(max_fresh=1, max_replay=3)
    producer = owner.claim_producer("decode")
    stale = lease(1, generation=0)
    assert producer.offer(stale, lane="replay", arrival_ordinal=1)

    evictions = owner.evict_replay_provisional(count=1)
    assert tuple(eviction.lease for eviction in evictions) == (stale,)

    fresher = lease(1, generation=1)
    assert producer.offer(fresher, lane="replay", arrival_ordinal=1)

    owner.resolve_replay_eviction(evictions, commit=False)

    # The stale evicted payload must never re-enter once a strictly newer
    # generation has superseded it (the same ABA contract evict_replay
    # already documents) -- rollback is a safe no-op for that entry.
    assert owner.pending_leases == (fresher,)


def test_scheduler_fresh_eviction_keeps_exact_plan_sources_resident():
    owner = ReadySourceOwner(max_fresh=3, max_replay=1)
    producer = owner.claim_producer("decode")
    scheduler = owner.claim_scheduler("scheduler")
    first, protected, third = lease(1), lease(2), lease(3)
    for item in (first, protected, third):
        assert producer.offer(item, lane="fresh", arrival_ordinal=item.source_identity.source_ordinal)

    assert scheduler.evict_fresh(
        protected_sources=frozenset((protected.source_identity,))
    ) == (first,)
    assert tuple(item.source_identity for item in owner.pending_leases) == (
        protected.source_identity,
        third.source_identity,
    )


def test_scheduler_replay_eviction_keeps_exact_plan_sources_resident():
    """COLD page-in cannot rotate a frozen B4 hole through replay residency."""

    owner = ReadySourceOwner(max_fresh=1, max_replay=3)
    producer = owner.claim_producer("decode")
    scheduler = owner.claim_scheduler("scheduler")
    protected, candidate, other = lease(1), lease(2), lease(3)
    for item in (protected, candidate, other):
        assert producer.offer(
            item,
            lane="replay",
            arrival_ordinal=item.source_identity.source_ordinal,
        )

    assert scheduler.evict_replay(
        protected_sources=frozenset((protected.source_identity,))
    ) == (candidate,)
    assert tuple(item.source_identity for item in owner.pending_leases) == (
        protected.source_identity,
        other.source_identity,
    )


def test_scheduler_fresh_eviction_requires_a_new_generation_for_reentry():
    owner = ReadySourceOwner(max_fresh=1, max_replay=1)
    producer = owner.claim_producer("decode")
    scheduler = owner.claim_scheduler("scheduler")
    original = lease(1, generation=1)
    assert producer.offer(original, lane="fresh", arrival_ordinal=1)
    assert scheduler.evict_fresh(protected_sources=frozenset()) == (original,)

    assert producer.offer(lease(1, generation=1), lane="fresh", arrival_ordinal=1) is False
    assert producer.offer(
        lease(1, generation=scheduler.next_generation(identity(1))),
        lane="fresh",
        arrival_ordinal=1,
    )


def test_invalid_ready_event_is_rejected_at_lease_construction():
    with pytest.raises(TypeError, match="query"):
        lease(1, event=object())


def test_requeue_keeps_the_in_flight_lifetime_lane_priority_and_waits_for_replacement():
    owner = ReadySourceOwner(max_fresh=1, max_replay=1)
    producer = owner.claim_producer("decode")
    consumer = owner.claim_consumer("gpu")
    token = object()
    original = lease(4, generation=7, lifetime_token=token)
    assert producer.offer(
        original,
        lane=ReadySourceLane.REPLAY,
        arrival_ordinal=4,
        priority=ReplayPriority.DEMANDED,
    )
    assert consumer.poll() is original
    completion = Gate(False)

    requeued = consumer.requeue(original, ready_event=completion)

    assert owner.counters.in_flight == 0
    assert owner.counters.released == 0
    assert owner.pending_count == 1
    assert consumer.poll() is None
    completion.ready = True
    selected = consumer.poll()
    assert selected is requeued
    assert selected.generation == 7
    assert selected.lifetime_token is token
    consumer.release(selected)


def test_requeue_rejects_stale_or_foreign_in_flight_lease():
    owner = ReadySourceOwner(max_fresh=1, max_replay=1)
    producer = owner.claim_producer("decode")
    consumer = owner.claim_consumer("gpu")
    original = lease(4, generation=7, lifetime_token=object())
    assert producer.offer(original, lane=ReadySourceLane.FRESH, arrival_ordinal=4)
    assert consumer.poll() is original

    with pytest.raises(ValueError, match="currently owned"):
        consumer.requeue(lease(4, generation=7, lifetime_token=object()), ready_event=Gate())


def test_batch_requeue_prevalidation_keeps_every_lease_in_flight_on_failure():
    owner = ReadySourceOwner(max_fresh=2, max_replay=2)
    producer = owner.claim_producer("decode")
    first, second = lease(1), lease(2)
    assert producer.offer(first, lane=ReadySourceLane.FRESH, arrival_ordinal=1)
    assert producer.offer(second, lane=ReadySourceLane.FRESH, arrival_ordinal=2)
    selected = owner.select(2)
    assert selected == (first, second)

    with pytest.raises(ValueError, match="batch requeue"):
        owner.requeue_batch((first, lease(3)), ready_event=Gate(True))

    assert owner.counters.in_flight == 2
    assert owner.pending_count == 0
    assert owner.requeue_batch(selected, ready_event=Gate(True))
    assert owner.counters.in_flight == 0
    assert owner.pending_count == 2


def test_requeued_completion_replaces_prior_preparation_handoff():
    owner = ReadySourceOwner(max_fresh=1, max_replay=1)
    producer = owner.claim_producer("decode")
    consumer = owner.claim_consumer("gpu")
    prepared = HandoffGate("prepared")
    original = lease(4, event=prepared)
    assert producer.offer(original, lane=ReadySourceLane.FRESH, arrival_ordinal=4)
    assert consumer.poll() is original
    completion = HandoffGate("completion")

    requeued = consumer.requeue(original, ready_event=completion)
    selected = consumer.poll()
    assert selected is requeued
    transfer_owner = HandoffOwner()

    selected.handoff_to_compute(transfer_owner)

    assert transfer_owner.handoffs == ["completion"]


def test_requeue_publish_is_atomic_against_a_concurrent_duplicate_offer():
    owner = ReadySourceOwner(max_fresh=1, max_replay=1)
    producer = owner.claim_producer("decode")
    consumer = owner.claim_consumer("gpu")
    original = lease(4)
    duplicate = dataclasses.replace(original)
    assert producer.offer(original, lane=ReadySourceLane.FRESH, arrival_ordinal=4)
    assert consumer.poll() is original
    requeue_entered = threading.Event()
    resume_requeue = threading.Event()
    duplicate_result: list[bool] = []

    class RequeueGateLock:
        def __init__(self) -> None:
            self._lock = threading.Lock()

        def __enter__(self):
            self._lock.acquire()
            if threading.current_thread().name == "requeue":
                requeue_entered.set()
                assert resume_requeue.wait(timeout=1)
            return self

        def __exit__(self, *_args) -> None:
            self._lock.release()

    owner._state_lock = RequeueGateLock()
    requeue_thread = threading.Thread(
        name="requeue",
        target=lambda: consumer.requeue(original, ready_event=Gate(True)),
    )
    requeue_thread.start()
    assert requeue_entered.wait(timeout=1)
    offer_thread = threading.Thread(
        target=lambda: duplicate_result.append(
            producer.offer(duplicate, lane=ReadySourceLane.FRESH, arrival_ordinal=4)
        )
    )
    offer_thread.start()
    resume_requeue.set()
    requeue_thread.join(timeout=1)
    offer_thread.join(timeout=1)

    assert not requeue_thread.is_alive()
    assert not offer_thread.is_alive()
    assert duplicate_result == [False]
    assert owner.pending_leases[0].source_identity == original.source_identity


def test_select_replacement_is_atomic_against_a_concurrent_offer():
    owner = ReadySourceOwner(max_fresh=2, max_replay=1)
    producer = owner.claim_producer("decode")
    consumer = owner.claim_consumer("gpu")
    first = lease(1)
    second = lease(2)
    assert producer.offer(first, lane=ReadySourceLane.FRESH, arrival_ordinal=1)

    replacement_entered = threading.Event()
    resume_replacement = threading.Event()
    owner._fresh = BlockingReplaceList(owner._fresh, replacement_entered, resume_replacement)
    state_lock = RecordingLock()
    owner._state_lock = state_lock
    selected: list[tuple[PreparedSourceLease, ...]] = []
    select_thread = threading.Thread(
        name="select",
        target=lambda: selected.append(consumer.select(1)),
    )
    select_thread.start()
    assert replacement_entered.wait(timeout=1)

    offered: list[bool] = []
    offer_thread = threading.Thread(
        name="offer",
        target=lambda: offered.append(
            producer.offer(second, lane=ReadySourceLane.FRESH, arrival_ordinal=2)
        ),
    )
    offer_thread.start()
    try:
        assert not state_lock.offer_entered.wait(timeout=0.1)
    finally:
        resume_replacement.set()
    select_thread.join(timeout=1)
    offer_thread.join(timeout=1)

    assert not select_thread.is_alive()
    assert not offer_thread.is_alive()
    assert selected == [(first,)]
    assert offered == [True]
    assert owner.pending_leases == (second,)
    assert owner.counters.fresh_pending <= owner._max_fresh


def test_release_is_atomic_against_a_concurrent_offer():
    owner = ReadySourceOwner(max_fresh=1, max_replay=1)
    producer = owner.claim_producer("decode")
    consumer = owner.claim_consumer("gpu")
    first = lease(1)
    second = lease(2)
    assert producer.offer(first, lane=ReadySourceLane.FRESH, arrival_ordinal=1)
    assert consumer.poll() is first

    removal_entered = threading.Event()
    resume_removal = threading.Event()
    owner._in_flight = BlockingPopDict(owner._in_flight, removal_entered, resume_removal)
    state_lock = RecordingLock()
    owner._state_lock = state_lock
    release_thread = threading.Thread(name="release", target=lambda: consumer.release(first))
    release_thread.start()
    assert removal_entered.wait(timeout=1)

    offered: list[bool] = []
    offer_thread = threading.Thread(
        name="offer",
        target=lambda: offered.append(
            producer.offer(second, lane=ReadySourceLane.FRESH, arrival_ordinal=2)
        ),
    )
    offer_thread.start()
    try:
        assert not state_lock.offer_entered.wait(timeout=0.1)
    finally:
        resume_removal.set()
    release_thread.join(timeout=1)
    offer_thread.join(timeout=1)

    assert not release_thread.is_alive()
    assert not offer_thread.is_alive()
    assert offered == [True]
    assert owner.pending_leases == (second,)
    assert owner.counters.fresh_pending <= owner._max_fresh


def test_pending_reservation_consumes_capacity_before_another_offer():
    """A producer claim must block every later same-lane publication."""

    owner = ReadySourceOwner(max_fresh=1, max_replay=1)
    producer = owner.claim_producer("decode")
    reservation = producer.reserve(
        identity(1),
        lane=ReadySourceLane.FRESH,
        generation=0,
        lifetime_token=object(),
    )

    assert reservation is not None
    assert not producer.offer(lease(2), lane=ReadySourceLane.FRESH)
    assert owner.counters.fresh_pending == 0
    assert owner.counters.in_flight == 0


def test_two_b4_reservoir_allows_pagein_while_one_b4_is_in_flight():
    """A B4 must not consume the capacity needed to prepare its successor."""

    owner = ReadySourceOwner(max_fresh=1, max_replay=16)
    producer = owner.claim_producer("decode")
    consumer = owner.claim_consumer("gpu")
    for ordinal in range(12):
        assert producer.offer(lease(ordinal), lane=ReadySourceLane.REPLAY)

    assert len(consumer.select(batch_size=4)) == 4
    reservations = [
        producer.reserve(
            identity(ordinal),
            lane=ReadySourceLane.REPLAY,
            generation=0,
            lifetime_token=object(),
        )
        for ordinal in range(12, 16)
    ]

    assert all(reservation is not None for reservation in reservations)
    scheduler = owner.claim_scheduler("pagein")
    assert scheduler.replay_residency_depth == 16


def test_pending_reservation_blocks_an_in_flight_return_lane_claim():
    """Return reservations account for producer capacity claims too."""

    owner = ReadySourceOwner(max_fresh=1, max_replay=1)
    producer = owner.claim_producer("decode")
    consumer = owner.claim_consumer("gpu")
    selected = lease(2)
    assert producer.offer(selected, lane=ReadySourceLane.FRESH)
    assert consumer.poll(lane=ReadySourceLane.FRESH) is selected
    reservation = producer.reserve(
        identity(1),
        lane=ReadySourceLane.REPLAY,
        generation=0,
        lifetime_token=object(),
    )
    assert reservation is not None

    with pytest.raises(RuntimeError, match="return lane capacity"):
        consumer.reserve_return_lanes(
            (selected,),
            lane_by_source={selected.source_identity: ReadySourceLane.REPLAY},
        )
    assert owner.counters.in_flight == 1
    reservation.abort()
    consumer.release(selected)


def test_reservation_rejects_same_generation_different_lifetime_token():
    """The reservation's token is the ABA fence, not just its generation."""

    owner = ReadySourceOwner(max_fresh=1, max_replay=1)
    producer = owner.claim_producer("decode")
    reserved_token = object()
    reservation = producer.reserve(
        identity(1),
        lane=ReadySourceLane.FRESH,
        generation=3,
        lifetime_token=reserved_token,
    )
    assert reservation is not None

    with pytest.raises(ValueError, match="lifetime"):
        reservation.commit(lease(1, generation=3, lifetime_token=object()))

    assert producer.reserve(
        identity(1),
        lane=ReadySourceLane.FRESH,
        generation=3,
        lifetime_token=object(),
    ) is not None


def test_requeue_capacity_failure_leaves_selected_lease_in_flight():
    """A failed lane switch must not orphan a selected source."""

    owner = ReadySourceOwner(max_fresh=1, max_replay=1)
    producer = owner.claim_producer("decode")
    consumer = owner.claim_consumer("gpu")
    selected = lease(2)
    assert producer.offer(selected, lane=ReadySourceLane.FRESH)
    assert consumer.poll(lane=ReadySourceLane.FRESH) is selected
    reservation = producer.reserve(
        identity(1),
        lane=ReadySourceLane.REPLAY,
        generation=0,
        lifetime_token=object(),
    )
    assert reservation is not None

    with pytest.raises(RuntimeError, match="target lane is full"):
        consumer.requeue(selected, ready_event=Gate(True), lane=ReadySourceLane.REPLAY)
    assert owner.counters.in_flight == 1
    assert owner.pending_count == 0

    reservation.abort()
    assert consumer.requeue(selected, ready_event=Gate(True), lane=ReadySourceLane.REPLAY)
    assert owner.counters.in_flight == 0
    assert owner.counters.replay_pending == 1
