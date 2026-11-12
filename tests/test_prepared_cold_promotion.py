"""CPU contracts for runtime-resolved durable COLD -> READY page-in."""

from __future__ import annotations

import threading
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from eligsir.contracts import BufferRef, CameraIntrinsics, SensorPacket, SourceIdentity
from eligsir.contracts.control import (
    DispositionKind,
    HistoricalEligibility,
    ModalityEligibility,
    ObservationDisposition,
    ResidencyIntent,
    SemanticAction,
)
from eligsir.mapping.admission.observation_owner import HistoricalPromotionPermit
from eligsir.mapping.observation_db import ColdPixelHandle, FrameObservationDB
from eligsir.mapping.scheduler.historical_candidate_index import (
    HistoricalCandidateEvidence,
    HistoricalReplayCandidate,
)
from eligsir.runtime.prepared_ingress import (
    PreparedIngressControlToken,
    PreparedIngressItem,
    PreparedSourceIngressHandoff,
)
from eligsir.runtime.prepared_runtime_bridge import PreparedRuntimeBridge
from eligsir.runtime.prepared_source_producer import PreparedSourceProducer
from eligsir.runtime.prepared_source_producer import PreparedSourceProduceResult
from eligsir.runtime.ready_source import ReadySourceOwner
from eligsir.runtime.transfers import DeviceTransferOwner
from eligsir.tracking.providers import observation_from_pose_matrix


def _candidate(frame_id: int) -> HistoricalReplayCandidate:
    return HistoricalReplayCandidate(
        frame_id=frame_id,
        weight=1.0,
        evidence=HistoricalCandidateEvidence(
            lanes=("focus",), region_id=0, focus_weight=1.0, snapshot_id=7, epoch=3
        ),
    )


def _item(frame_id: int, *, action: SemanticAction, depth: bool = True):
    source = SourceIdentity("cold", 0, frame_id, frame_id * 1_000)
    intrinsics = CameraIntrinsics(
        fx=2.0, fy=2.0, cx=1.0, cy=1.0, width=3, height=2, depth_scale=1000.0
    )
    observation = observation_from_pose_matrix(source, np.eye(4), provider="test")
    commit = action is SemanticAction.COMMIT
    disposition = ObservationDisposition(
        source=source,
        snapshot_id=3,
        kind=DispositionKind.ADMIT if commit else DispositionKind.SKIP_REDUNDANT,
        observe=True,
        mapping_candidate=commit,
        admit_supervision=commit,
        allow_geometry_insertion=False,
        residency=ResidencyIntent.HOT if commit else ResidencyIntent.COLD,
        reason=action.value,
        modality_eligibility=ModalityEligibility(
            rgb_supervision=True, depth_supervision=depth
        ),
        semantic_action=action,
        historical_eligibility=HistoricalEligibility.ELIGIBLE,
    )
    packet = SensorPacket(
        source=source,
        rgb=BufferRef.from_numpy(np.zeros((2, 3, 3), np.uint8), encoding="rgb8"),
        depth=(
            BufferRef.from_numpy(
                np.ones((2, 3), np.uint16), encoding="16UC1", units="mm"
            )
            if depth
            else None
        ),
        intrinsics=intrinsics,
        tracking=observation,
    )
    token = PreparedIngressControlToken(frame_id, source, 3, disposition)
    return PreparedIngressItem(
        frame_id, packet, observation, disposition=disposition, control_token=token
    )


def _resolution(frame_id: int, *, action: SemanticAction, depth: bool = True):
    from eligsir.runtime.prepared_cold_promotion import PreparedColdPromotionResolution

    item = _item(frame_id, action=action, depth=depth)
    permit = (
        None
        if action is SemanticAction.COMMIT
        else HistoricalPromotionPermit(item.packet.source, 3, 4, 5, 6)
    )
    return PreparedColdPromotionResolution.ready(item, promotion_permit=permit)


class _Bridge:
    def __init__(self, result=PreparedSourceProduceResult.ENQUEUED):
        self.result = result
        self.max_items_per_tick = 4
        self.calls = []
        self.batches = []

    def submit_promoted(self, item, *, promotion_permit, generation=0):
        self.calls.append(("defer", item, promotion_permit, generation))
        return self.result

    def submit_retained_commit(self, item, *, generation=0):
        self.calls.append(("commit", item, None, generation))
        return self.result

    def submit_promoted_batch(self, items, *, promotion_permits, generations):
        self.batches.append(tuple(items))
        return tuple(
            self.submit_promoted(item, promotion_permit=permit, generation=generation)
            for item, permit, generation in zip(items, promotion_permits, generations)
        )

    def submit_retained_commit_batch(self, items, *, generations):
        self.batches.append(tuple(items))
        return tuple(
            self.submit_retained_commit(item, generation=generation)
            for item, generation in zip(items, generations)
        )


def _owner(bridge, resolve, *, retire_frame=None):
    from eligsir.runtime.prepared_cold_promotion import PreparedColdPromotionOwner

    ready = ReadySourceOwner(max_fresh=1, max_replay=2)
    owner = PreparedColdPromotionOwner(
        bridge,
        ready_scheduler=ready.claim_scheduler("historical-scheduler"),
        resolve=resolve,
        retire_frame=retire_frame,
    )
    owner._promotion_fence = (4, 5, 6)
    return owner


def test_schedule_resolves_on_runtime_thread_and_worker_never_calls_resolver():
    bridge = _Bridge()
    resolution_thread = []

    def resolve(candidate):
        resolution_thread.append(threading.get_ident())
        return _resolution(candidate.frame_id, action=SemanticAction.DEFER)

    promotion = _owner(bridge, resolve)
    caller = threading.get_ident()
    assert promotion.schedule_shortlist((_candidate(17),), promotion_fence=(4, 5, 6)) == 1
    assert resolution_thread == [caller]

    promotion._resolve = lambda _candidate: (_ for _ in ()).throw(
        AssertionError("prep worker re-entered runtime resolver")
    )
    assert promotion.pump_once() is True
    assert bridge.calls[0][0] == "defer"
    assert bridge.calls[0][3] == 1
    assert promotion.pump_once() is False


def test_pump_once_submits_available_compatible_cold_requests_as_one_batch():
    bridge = _Bridge()
    promotion = _owner(
        bridge, lambda candidate: _resolution(candidate.frame_id, action=SemanticAction.DEFER)
    )
    assert promotion.schedule_shortlist(
        (_candidate(71), _candidate(72)), promotion_fence=(4, 5, 6)
    ) == 2
    promotion._resolve = lambda _candidate: pytest.fail("worker must not resolve")

    assert promotion.pump_once() is True

    assert [[item.packet.source for item in batch] for batch in bridge.batches] == [[
        _item(71, action=SemanticAction.DEFER).packet.source,
        _item(72, action=SemanticAction.DEFER).packet.source,
    ]]


def test_compatible_batch_drops_requests_when_promotion_fence_changes_before_submit():
    """A worker-taken cohort must not submit after its DEFER permit expires."""
    bridge = _Bridge()
    promotion = _owner(
        bridge, lambda candidate: _resolution(candidate.frame_id, action=SemanticAction.DEFER)
    )
    assert promotion.schedule_shortlist(
        (_candidate(81), _candidate(82)), promotion_fence=(4, 5, 6)
    ) == 2
    promotion._promotion_fence = (7, 8, 9)

    assert promotion.pump_once() is True
    assert bridge.batches == []


def test_ready_cohort_member_does_not_skip_valid_sibling_submission():
    """A preflight success must not short-circuit the remaining compatible request."""
    bridge = _Bridge()
    promotion = _owner(
        bridge, lambda candidate: _resolution(candidate.frame_id, action=SemanticAction.DEFER)
    )
    already_ready = _item(83, action=SemanticAction.DEFER).packet.source
    promotion._ready_scheduler = SimpleNamespace(
        has_source=lambda source, ready=False: source == already_ready,
        next_generation=lambda _source: 1,
    )
    assert promotion.schedule_shortlist(
        (_candidate(83), _candidate(84)), promotion_fence=(4, 5, 6)
    ) == 2

    assert promotion.pump_once() is True
    assert [call[1].packet.source for call in bridge.calls] == [
        _item(84, action=SemanticAction.DEFER).packet.source
    ]


def test_retry_identity_does_not_suppress_new_generation_after_stale_queue_entry():
    """A stale equal-shaped request cannot suppress the currently scheduled retry."""
    from eligsir.runtime.prepared_cold_promotion import _PromotionRequest

    bridge = _Bridge()
    promotion = _owner(
        bridge, lambda candidate: _resolution(candidate.frame_id, action=SemanticAction.DEFER)
    )
    resolution = _resolution(85, action=SemanticAction.DEFER)
    stale = _PromotionRequest(85, resolution, frozenset())
    current = _PromotionRequest(85, resolution, frozenset())
    promotion._queue.append(stale)
    promotion._scheduled[85] = current

    promotion._retry(current)

    assert tuple(promotion._queue) == (stale, current)


def test_compatible_cold_prefix_uses_bridge_source_work_budget_not_transfer_slots():
    bridge = _Bridge()
    bridge.max_items_per_tick = 2
    promotion = _owner(
        bridge, lambda candidate: _resolution(candidate.frame_id, action=SemanticAction.DEFER)
    )
    assert promotion.schedule_shortlist(
        (_candidate(73), _candidate(74), _candidate(75)), promotion_fence=(4, 5, 6)
    ) == 3

    assert promotion.pump_once() is True
    assert [item.packet.source for item in bridge.batches[0]] == [
        _item(73, action=SemanticAction.DEFER).packet.source,
        _item(74, action=SemanticAction.DEFER).packet.source,
    ]
    assert promotion.pump_once() is True
    assert bridge.calls[-1][1].packet.source == _item(75, action=SemanticAction.DEFER).packet.source


def test_partial_group_capacity_repair_protects_all_group_members():
    """Evicting for a retained sibling must not evict a newly enqueued peer."""
    bridge = _Bridge()
    results = iter((PreparedSourceProduceResult.ENQUEUED, PreparedSourceProduceResult.RETAINED_NOT_RESIDENT))
    bridge.submit_promoted_batch = lambda items, **_kwargs: tuple(next(results) for _item in items)
    promotion = _owner(
        bridge, lambda candidate: _resolution(candidate.frame_id, action=SemanticAction.DEFER)
    )
    evictions = []
    promotion._ready_scheduler = SimpleNamespace(
        has_source=lambda _source, ready=False: False,
        next_generation=lambda _source: 1,
        replay_residency_depth=2,
        replay_capacity=2,
        evict_replay=lambda **kwargs: evictions.append(kwargs) or (object(),),
    )
    assert promotion.schedule_shortlist(
        (_candidate(91), _candidate(92)), promotion_fence=(4, 5, 6)
    ) == 2

    assert promotion.pump_once() is True
    assert {
        _item(91, action=SemanticAction.DEFER).packet.source,
        _item(92, action=SemanticAction.DEFER).packet.source,
    }.issubset(evictions[0]["protected_sources"])


def test_group_submit_exception_does_not_duplicate_an_already_retried_request():
    """A batch failure must leave at most one retry entry per scheduled source."""
    bridge = _Bridge()
    def raise_submit(*_args, **_kwargs):
        raise RuntimeError("boom")

    bridge.submit_promoted = raise_submit
    bridge.submit_promoted_batch = raise_submit
    promotion = _owner(
        bridge, lambda candidate: _resolution(candidate.frame_id, action=SemanticAction.DEFER)
    )
    pending = _item(93, action=SemanticAction.DEFER).packet.source
    promotion._ready_scheduler = SimpleNamespace(
        has_source=lambda source, ready=False: source == pending and not ready,
        next_generation=lambda _source: 1,
    )
    assert promotion.schedule_shortlist(
        (_candidate(93), _candidate(94)), promotion_fence=(4, 5, 6)
    ) == 2

    with pytest.raises(RuntimeError, match="boom"):
        promotion.pump_once()
    assert [request.frame_id for request in promotion._queue].count(93) == 1
    assert [request.frame_id for request in promotion._queue].count(94) == 1


def test_pending_resolution_is_not_queued_and_retries_on_next_runtime_shortlist():
    from eligsir.runtime.prepared_cold_promotion import PreparedColdPromotionResolution

    calls = 0

    def resolve(candidate):
        nonlocal calls
        calls += 1
        return (
            PreparedColdPromotionResolution.pending()
            if calls == 1
            else _resolution(candidate.frame_id, action=SemanticAction.DEFER)
        )

    bridge = _Bridge()
    promotion = _owner(bridge, resolve)
    candidate = _candidate(8)
    assert promotion.schedule_shortlist((candidate,), promotion_fence=(4, 5, 6)) == 0
    assert promotion.pump_once() is False
    assert promotion.schedule_shortlist((candidate,), promotion_fence=(4, 5, 6)) == 1
    assert promotion.pump_once() is True
    assert len(bridge.calls) == 1


def test_retirement_runs_during_runtime_scheduling_not_worker_pump():
    from eligsir.runtime.prepared_cold_promotion import PreparedColdPromotionResolution

    retired = []
    caller = threading.get_ident()
    promotion = _owner(
        _Bridge(),
        lambda _candidate: PreparedColdPromotionResolution.retire(),
        retire_frame=lambda frame_id: retired.append((frame_id, threading.get_ident())),
    )
    assert promotion.schedule_shortlist((_candidate(5),), promotion_fence=(4, 5, 6)) == 0
    assert retired == [(5, caller)]
    assert promotion.pump_once() is False


def test_retained_commit_uses_fresh_route_without_historical_permit():
    bridge = _Bridge()
    promotion = _owner(
        bridge, lambda candidate: _resolution(candidate.frame_id, action=SemanticAction.COMMIT)
    )
    assert promotion.schedule_shortlist((_candidate(61),), promotion_fence=(4, 5, 6)) == 1
    assert promotion.pump_once() is True
    assert bridge.calls[0][0] == "commit"


def test_retained_commit_reclaims_only_unprotected_fresh_residency():
    bridge = _Bridge(PreparedSourceProduceResult.RETAINED_NOT_RESIDENT)
    promotion = _owner(
        bridge, lambda candidate: _resolution(candidate.frame_id, action=SemanticAction.COMMIT)
    )
    evicted = [object()]
    protected = _item(99, action=SemanticAction.COMMIT).packet.source
    promotion._ready_scheduler = SimpleNamespace(
        has_source=lambda _source: False,
        next_generation=lambda _source: 1,
        evict_fresh=lambda *, protected_sources: (
            tuple(evicted.pop() for _ in range(1))
            if protected_sources == frozenset((protected,)) and evicted
            else ()
        ),
    )

    assert promotion.schedule_shortlist(
        (_candidate(61),),
        promotion_fence=(4, 5, 6),
        protected_sources=frozenset((protected,)),
    ) == 1
    assert promotion.pump_once() is True
    assert bridge.calls[0][0] == "commit"


def test_retained_commit_refreshes_exact_plan_protection_before_pagein():
    bridge = _Bridge(PreparedSourceProduceResult.RETAINED_NOT_RESIDENT)
    promotion = _owner(
        bridge, lambda candidate: _resolution(candidate.frame_id, action=SemanticAction.COMMIT)
    )
    plan_a = _item(98, action=SemanticAction.COMMIT).packet.source
    plan_b = _item(99, action=SemanticAction.COMMIT).packet.source
    protected_calls = []
    promotion._ready_scheduler = SimpleNamespace(
        has_source=lambda _source: False,
        next_generation=lambda _source: 1,
        evict_fresh=lambda *, protected_sources: (
            protected_calls.append(protected_sources) or (object(),)
        ),
    )

    assert promotion.schedule_shortlist(
        (_candidate(62),),
        promotion_fence=(4, 5, 6),
        protected_sources=frozenset((plan_a,)),
    ) == 1
    # The same durable COMMIT remains demanded, but the exact plan changed
    # before the prep worker got to it. Its eventual fresh eviction must use
    # the current plan guard, not the obsolete one.
    assert promotion.schedule_shortlist(
        (_candidate(62),), promotion_fence=(4, 5, 6), protected_sources=frozenset((plan_b,))
    ) == 0
    assert promotion.pump_once() is True
    assert protected_calls == [frozenset((plan_b,))]


def test_inflight_retained_commit_uses_refreshed_exact_plan_protection():
    started = threading.Event()
    release = threading.Event()

    class _BlockingBridge(_Bridge):
        def submit_retained_commit(self, item, *, generation=0):
            self.calls.append(("commit", item, None, generation))
            started.set()
            assert release.wait(timeout=1.0)
            return self.result

    bridge = _BlockingBridge(PreparedSourceProduceResult.RETAINED_NOT_RESIDENT)
    promotion = _owner(
        bridge, lambda candidate: _resolution(candidate.frame_id, action=SemanticAction.COMMIT)
    )
    plan_a = _item(98, action=SemanticAction.COMMIT).packet.source
    plan_b = _item(99, action=SemanticAction.COMMIT).packet.source
    protected_calls = []
    promotion._ready_scheduler = SimpleNamespace(
        has_source=lambda _source: False,
        next_generation=lambda _source: 1,
        evict_fresh=lambda *, protected_sources: (
            protected_calls.append(protected_sources) or (object(),)
        ),
    )
    candidate = _candidate(62)
    assert promotion.schedule_shortlist(
        (candidate,), promotion_fence=(4, 5, 6), protected_sources=frozenset((plan_a,))
    ) == 1
    worker = threading.Thread(target=promotion.pump_once)
    worker.start()
    assert started.wait(timeout=1.0)
    assert promotion.schedule_shortlist(
        (candidate,), promotion_fence=(4, 5, 6), protected_sources=frozenset((plan_b,))
    ) == 0
    release.set()
    worker.join(timeout=1.0)

    assert not worker.is_alive()
    assert protected_calls == [frozenset((plan_b,))]


def test_ready_owner_drop_is_terminal_authority_rejection():
    bridge = _Bridge(PreparedSourceProduceResult.DROPPED_READY_OWNER)
    promotion = _owner(
        bridge, lambda candidate: _resolution(candidate.frame_id, action=SemanticAction.DEFER)
    )
    assert promotion.schedule_shortlist((_candidate(9),), promotion_fence=(4, 5, 6)) == 1
    assert promotion.pump_once() is True
    assert promotion.pump_once() is False
    assert [call[3] for call in bridge.calls] == [1]


def test_present_but_unready_source_keeps_promotion_pending_until_ready():
    """A pre-existing preparation must not be mistaken for READY residency."""

    bridge = _Bridge()
    promotion = _owner(
        bridge, lambda candidate: _resolution(candidate.frame_id, action=SemanticAction.COMMIT)
    )
    ready_gate = [False]
    promotion._ready_scheduler = SimpleNamespace(
        has_source=lambda _source, *, ready=False: ready_gate[0] if ready else True,
        next_generation=lambda _source: 1,
    )

    assert promotion.schedule_shortlist((_candidate(63),), promotion_fence=(4, 5, 6)) == 1
    assert promotion.pump_once() is False
    assert bridge.calls == []

    ready_gate[0] = True
    assert promotion.pump_once() is True
    assert bridge.calls == []


def test_nonresident_without_replay_saturation_retries_but_yields():
    """A nonresident result is not itself evidence that READY replay is full."""

    bridge = _Bridge(PreparedSourceProduceResult.RETAINED_NOT_RESIDENT)
    promotion = _owner(
        bridge, lambda candidate: _resolution(candidate.frame_id, action=SemanticAction.DEFER)
    )
    promotion._ready_scheduler = SimpleNamespace(
        has_source=lambda _source: False,
        next_generation=lambda _source: 1,
        replay_residency_depth=0,
        replay_capacity=2,
        evict_replay=lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("unsaturated replay must not evict")
        ),
    )

    assert promotion.schedule_shortlist((_candidate(19),), promotion_fence=(4, 5, 6)) == 1
    assert promotion.pump_once() is False
    bridge.result = PreparedSourceProduceResult.ENQUEUED
    assert promotion.pump_once() is True
    assert [call[3] for call in bridge.calls] == [1, 1]


def test_replay_eviction_counts_as_progress_before_nonresident_retry():
    bridge = _Bridge(PreparedSourceProduceResult.RETAINED_NOT_RESIDENT)
    promotion = _owner(
        bridge, lambda candidate: _resolution(candidate.frame_id, action=SemanticAction.DEFER)
    )
    evicted = [object()]
    promotion._ready_scheduler = SimpleNamespace(
        has_source=lambda _source: False,
        next_generation=lambda _source: 1,
        replay_residency_depth=2,
        replay_capacity=2,
        evict_replay=lambda **_kwargs: tuple(evicted.pop() for _ in range(1)) if evicted else (),
    )
    promotion.schedule_shortlist((_candidate(20),), promotion_fence=(4, 5, 6))
    assert promotion.pump_once() is True
    assert promotion.pump_once() is False


def test_full_ready_replay_batch_evicts_then_publishes_each_compatible_cold_source():
    """A full READY lane must not strand a compatible retained COLD cohort."""
    from tests.test_ready_source_owner import lease
    from eligsir.runtime.prepared_cold_promotion import PreparedColdPromotionOwner

    ready = ReadySourceOwner(max_fresh=1, max_replay=2)
    resident_producer = ready.claim_producer("cold-decode")
    assert resident_producer.offer(lease(1), lane="replay", arrival_ordinal=1)
    assert resident_producer.offer(lease(2), lane="replay", arrival_ordinal=2)
    bridge = PreparedRuntimeBridge(
        PreparedSourceIngressHandoff(capacity=4),
        PreparedSourceProducer(
            ready,
            transfer_owner=DeviceTransferOwner("cpu", max_inflight=4, max_pinned_bytes=4096),
            producer_owner="cold-decode",
        ),
        ready_owner=ready,
        consumer_owner="cold-bridge",
        max_items_per_tick=4,
    )
    promotion = PreparedColdPromotionOwner(
        bridge,
        ready_scheduler=ready.claim_scheduler("cold-scheduler"),
        resolve=lambda candidate: _resolution(candidate.frame_id, action=SemanticAction.DEFER),
    )
    requested = (_candidate(101), _candidate(102))
    assert promotion.schedule_shortlist(requested, promotion_fence=(4, 5, 6)) == 2

    assert promotion.pump_once() is True
    assert promotion.pump_once() is True
    assert {
        source.source_ordinal for source in ready.ready_replay_sources()
    } == {101, 102}


def test_replay_pagein_preserves_the_frozen_exact_plan_sources():
    bridge = _Bridge(PreparedSourceProduceResult.RETAINED_NOT_RESIDENT)
    promotion = _owner(
        bridge, lambda candidate: _resolution(candidate.frame_id, action=SemanticAction.DEFER)
    )
    protected = _item(41, action=SemanticAction.DEFER).packet.source
    seen = []
    promotion._ready_scheduler = SimpleNamespace(
        has_source=lambda _source: False,
        next_generation=lambda _source: 1,
        replay_residency_depth=2,
        replay_capacity=2,
        evict_replay=lambda *, count, protected_sources: (
            seen.append((count, protected_sources)) or (object(),)
        ),
    )

    assert promotion.schedule_shortlist(
        (_candidate(42),),
        promotion_fence=(4, 5, 6),
        protected_sources=frozenset((protected,)),
    ) == 1
    assert promotion.pump_once() is True
    assert seen == [(1, frozenset((protected,)))]


def test_rgb_only_history_is_resolved_but_never_evicts_rgbd_ready():
    bridge = _Bridge(PreparedSourceProduceResult.CAPABILITY_UNAVAILABLE)
    promotion = _owner(
        bridge,
        lambda candidate: _resolution(
            candidate.frame_id, action=SemanticAction.DEFER, depth=False
        ),
    )
    assert promotion.schedule_shortlist((_candidate(33),), promotion_fence=(4, 5, 6)) == 1
    promotion._ready_scheduler = SimpleNamespace(
        has_source=lambda _source: False,
        replay_residency_depth=2,
        replay_capacity=2,
        evict_replay=lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("RGB-only history must not evict READY replay")
        ),
    )
    assert promotion.pump_once() is True
    assert [call[0] for call in bridge.calls] == ["defer"]
    assert promotion.pump_once() is False


def test_db_resolver_rebuilds_exact_deferred_packet():
    from eligsir.runtime.prepared_cold_promotion import (
        FrameObservationPreparedColdResolver,
        PreparedColdPromotionContext,
        PreparedColdPromotionState,
    )

    item = _item(31, action=SemanticAction.DEFER)
    source = item.packet.source
    intrinsics = item.packet.intrinsics
    db = FrameObservationDB(device="cpu")
    record = db.record_prepared_historical_candidate(
        source_identity=source,
        pose_c2w=np.eye(4),
        timestamp=source.timestamp_seconds,
        intrinsics=intrinsics,
        cold_pixels=ColdPixelHandle(
            source_identity=source.fingerprint(),
            rgb=np.full((2, 3, 3), 127, dtype=np.uint8),
            depth=np.full((2, 3), 1000, dtype=np.uint16),
            intrinsics=intrinsics,
            rgb_encoding="rgb8",
            depth_encoding="16UC1",
            depth_units="mm",
        ),
        modality_eligibility=item.disposition.modality_eligibility,
        disposition_snapshot_id=3,
        semantic_action="defer",
    )
    context = PreparedColdPromotionContext(
        observation=item.observation,
        disposition=item.disposition,
        control_token=item.control_token,
        promotion_permit=HistoricalPromotionPermit(source, 3, 4, 5, 6),
    )
    resolver = FrameObservationPreparedColdResolver(
        db, context_for_source=lambda key: context if key == source.fingerprint() else None
    )
    resolution = resolver(_candidate(record.frame_id))
    assert resolution.state is PreparedColdPromotionState.READY
    assert resolution.item.packet.source is source
    assert resolution.item.packet.depth.spec.units == "mm"


def test_cold_resolver_retains_motion_facts_for_prepared_confidence():
    from eligsir.runtime.prepared_cold_promotion import (
        FrameObservationPreparedColdResolver,
        PreparedColdPromotionContext,
        PreparedColdPromotionState,
    )
    from eligsir.runtime.prepared_source_producer import PreparedSourceProducer
    from eligsir.runtime.transfers import DeviceTransferOwner

    item = _item(35, action=SemanticAction.DEFER)
    source = item.packet.source
    transform = np.eye(4, dtype=np.float32)
    transform[0, 3] = 0.5
    transform.setflags(write=False)
    observation = replace(
        item.observation,
        diagnostics={
            "motion_depth_confidence_source": "offline_tum",
            "motion_depth_confidence_reason": "interpolated",
            "motion_depth_confidence_transform": transform,
        },
    )
    item = PreparedIngressItem(
        item.sequence,
        replace(item.packet, tracking=observation),
        observation,
        disposition=item.disposition,
        control_token=item.control_token,
    )
    db = FrameObservationDB(device="cpu")
    record = db.record_prepared_historical_candidate(
        source_identity=source,
        pose_c2w=np.eye(4),
        timestamp=source.timestamp_seconds,
        intrinsics=item.packet.intrinsics,
        cold_pixels=ColdPixelHandle(
            source_identity=source.fingerprint(),
            rgb=np.zeros((2, 3, 3), dtype=np.uint8),
            depth=np.array(((1000, 2000, 4000), (1000, 2000, 4000)), dtype=np.uint16),
            intrinsics=item.packet.intrinsics,
            rgb_encoding="rgb8", depth_encoding="16UC1", depth_units="mm",
        ),
        modality_eligibility=item.disposition.modality_eligibility,
        disposition_snapshot_id=3, semantic_action="defer",
    )
    context = PreparedColdPromotionContext(
        observation=observation, disposition=item.disposition, control_token=item.control_token,
        promotion_permit=HistoricalPromotionPermit(source, 3, 4, 5, 6),
    )
    resolution = FrameObservationPreparedColdResolver(
        db, context_for_source=lambda key: context if key == source.fingerprint() else None
    )(_candidate(record.frame_id))

    assert resolution.state is PreparedColdPromotionState.READY
    assert resolution.item.packet.tracking.diagnostics["motion_depth_confidence_reason"] == "interpolated"
    assert np.array_equal(resolution.item.packet.tracking.diagnostics["motion_depth_confidence_transform"], transform)
    ready = ReadySourceOwner(max_fresh=1, max_replay=1)
    producer = PreparedSourceProducer(
        ready, transfer_owner=DeviceTransferOwner("cpu", max_pinned_bytes=4096),
        producer_owner="test", motion_depth_confidence_mode="motion",
    )
    producer.submit(resolution.item.packet, modality_eligibility=item.disposition.modality_eligibility)
    lease = ready.claim_consumer("consumer").poll()
    assert lease is not None
    assert not np.allclose(lease.tier_payload("native").depth_confidence.data.numpy(), 1.0)


def test_db_resolver_marks_unmaterializable_record_for_runtime_retirement():
    from eligsir.runtime.prepared_cold_promotion import (
        FrameObservationPreparedColdResolver,
        PreparedColdPromotionState,
    )

    item = _item(32, action=SemanticAction.DEFER)
    source = item.packet.source
    db = FrameObservationDB(device="cpu")
    record = db.record_prepared_historical_candidate(
        source_identity=source,
        pose_c2w=np.eye(4),
        timestamp=source.timestamp_seconds,
        intrinsics=item.packet.intrinsics,
        cold_pixels=ColdPixelHandle(
            source_identity=source.fingerprint(),
            rgb=None,
            depth=None,
            intrinsics=item.packet.intrinsics,
            payload_evicted=True,
        ),
        modality_eligibility=item.disposition.modality_eligibility,
        disposition_snapshot_id=3,
        semantic_action="defer",
    )
    resolver = FrameObservationPreparedColdResolver(
        db, context_for_source=lambda _key: None
    )
    assert resolver(_candidate(record.frame_id)).state is PreparedColdPromotionState.RETIRE
