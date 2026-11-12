"""Regression contracts for durable prepared-COMMIT first supervision."""

from __future__ import annotations

from collections import OrderedDict, deque
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from eligsir.contracts import (
    BufferOwnership,
    BufferRef,
    BufferSpec,
    CameraIntrinsics,
    MemoryDomain,
    SensorPacket,
)
from eligsir.contracts import SourceIdentity
from eligsir.contracts.control import (
    ControlLimits,
    ControlProposal,
    ControlSnapshot,
    DispositionKind,
    FreshnessMode,
    ModalityEligibility,
    ObservationDisposition,
    ResidencyIntent,
)
from eligsir.mapping.observation_db import ColdPixelHandle, FrameObservationDB
from eligsir.mapping.scheduler.historical_candidate_index import (
    HistoricalCandidateEvidence,
    HistoricalReplayCandidate,
)
from eligsir.mapping.scheduler.work_budget import WorkBudgetOwner
from eligsir.runtime.config import CanonicalOptimizationPolicy, MappingRuntimeConfig
from eligsir.runtime.prepared_ingress import (
    PreparedIngressControlToken,
    PreparedIngressItem,
    PreparedSourceIngressHandoff,
)
from eligsir.runtime.prepared_source import PreparedSourceLease
from eligsir.runtime.prepared_source_producer import PreparedSourceProducer
from eligsir.runtime.production_application import CanonicalMappingRuntime
from eligsir.runtime.ready_source import ReadySourceLane, ReadySourceOwner
from eligsir.runtime.transfers import DeviceTransferOwner
from eligsir.tracking.providers import observation_from_pose_matrix


BATCH_SIZE = 4
MODALITY = ModalityEligibility(rgb_supervision=True)

_DISABLED_REPLAY_CHALLENGER_POLICY = CanonicalOptimizationPolicy(
    {
        "streaming_replay_challenger_fraction": 0.0,
        "streaming_replay_challenger_demotion_cadence": 0,
        "streaming_replay_challenger_demotion_limit": 0,
        "streaming_replay_challenger_lease_ticks": 100,
    }
)


class _ResolveSpy:
    def __init__(self):
        self.kwargs = None

    def resolve(self, snapshot, **kwargs):
        self.kwargs = kwargs
        return object()


def _prepared_config(enabled: bool) -> MappingRuntimeConfig:
    return MappingRuntimeConfig(
        model_layout="gsplat",
        renderer="gsplat",
        numerical_mode="fp32",
        static_options={
            "population_shadow_enabled": False,
            "population_live_enabled": False,
            "prepared_ready_source_insertion": enabled,
        },
    )


def _source(ordinal: int) -> SourceIdentity:
    return SourceIdentity("prepared-commit", 0, ordinal, ordinal * 1_000)


def _lease(source: SourceIdentity, *, ready_event=None) -> PreparedSourceLease:
    token = object()
    payload = BufferRef(
        data=torch.zeros(2),
        spec=BufferSpec(
            (2,), "float32", MemoryDomain.OTHER_DEVICE, "tensor", device="cpu"
        ),
        ownership=BufferOwnership.BORROWED,
        owner_token=token,
    )
    return PreparedSourceLease(
        source_identity=source,
        modality_eligibility=MODALITY,
        source_metadata={"width": 2, "height": 2},
        tier_preparation_available=("native",),
        borrowed_device_payloads={"rgb": payload},
        device="cpu",
        ready_event=ready_event,
        lifetime_token=token,
    )


def _snapshot(source: SourceIdentity) -> ControlSnapshot:
    return ControlSnapshot(
        snapshot_id=source.source_ordinal + 1,
        control_version=1,
        source=source,
        tracking=None,
        pending_sources=(source,),
        newest_frontier=source,
        freshness_mode=FreshnessMode.HEALTHY,
        pending_oldest_age_ns=0,
        ingress_queue_depth=0,
        candidate_queue_depth=0,
        transfer_queue_depth=0,
        gpu_queue_depth=0,
        map_content_version=1,
        topology_version=1,
        parameter_version=1,
        gaussian_count=0,
        optimizer_steps_completed=0,
        limits=ControlLimits(BATCH_SIZE, BATCH_SIZE, 1, 0, 0, 0),
    )


def _disposition(source: SourceIdentity, snapshot: ControlSnapshot) -> ObservationDisposition:
    return ObservationDisposition(
        source=source,
        snapshot_id=snapshot.snapshot_id,
        kind=DispositionKind.ADMIT,
        observe=True,
        mapping_candidate=True,
        admit_supervision=True,
        allow_geometry_insertion=False,
        residency=ResidencyIntent.HOT,
        reason="prepared COMMIT",
        modality_eligibility=MODALITY,
    )


def _durably_retain_commit(
    db: FrameObservationDB,
    source: SourceIdentity,
    snapshot: ControlSnapshot,
    obligation: object,
    intrinsics: object | None = None,
    modality: ModalityEligibility = MODALITY,
) -> None:
    intrinsics = object() if intrinsics is None else intrinsics
    db.record_prepared_historical_candidate(
        source_identity=source,
        pose_c2w=np.eye(4, dtype=np.float32),
        timestamp=source.timestamp_seconds,
        intrinsics=intrinsics,
        cold_pixels=ColdPixelHandle(
            source_identity=source.fingerprint(),
            rgb=np.zeros(
                (int(getattr(intrinsics, "height", 2)), int(getattr(intrinsics, "width", 2)), 3),
                dtype=np.uint8,
            ),
            depth=None,
            intrinsics=intrinsics,
        ),
        modality_eligibility=modality,
        disposition_snapshot_id=snapshot.snapshot_id,
        semantic_action="commit",
    )
    db.record_prepared_commit_obligation(source, obligation)


def _runtime_with_unconsumed_commits():
    runtime = object.__new__(CanonicalMappingRuntime)
    runtime._optimization_policy = _DISABLED_REPLAY_CHALLENGER_POLICY
    runtime._observation_db = FrameObservationDB(device="cpu")
    runtime._latest_control_packet = None
    runtime._prepared_ready_candidate_work = None
    runtime._prepared_ready_frozen_work = None
    runtime._prepared_ready_plan_for_execution = lambda: None
    runtime._prepared_execution_contexts_by_source = OrderedDict()
    runtime._prepared_execution_context_capacity = 64
    runtime._dispositions_since_tick = deque(maxlen=64)
    runtime._replay_modalities = OrderedDict()
    runtime._replay_weights = {}
    runtime._pending_supervision_modalities = OrderedDict()
    runtime._pending_supervision_capacity = BATCH_SIZE - 1
    runtime._work_budget_owner = None
    runtime._work_clock_owner_if_available = lambda: None
    runtime._admitted_observations_total = 0
    runtime._services = None
    runtime._admitted_clock_ns = 0

    ready = ReadySourceOwner(max_fresh=BATCH_SIZE, max_replay=BATCH_SIZE - 1)
    producer = ready.claim_producer("prepared-commit-test")
    commits = tuple(_source(ordinal) for ordinal in range(1, BATCH_SIZE + 1))
    proposals = (
        ControlProposal("work", "supervision_slots", 1, "fresh"),
        ControlProposal("work", "replay_slots", BATCH_SIZE - 1, "ready replay"),
        ControlProposal("work", "optimizer_updates", 1, "one B4"),
    )
    for source in commits:
        snapshot = _snapshot(source)
        runtime._latest_control_outputs = (snapshot, proposals)
        runtime.record_prepared_disposition(_disposition(source, snapshot))
        context = runtime._prepared_execution_contexts_by_source[source]
        _durably_retain_commit(
            runtime._observation_db,
            source,
            snapshot,
            (object(), context.disposition, context.control_token, context),
        )
        assert producer.offer(_lease(source), lane=ReadySourceLane.FRESH)

    for ordinal in range(101, 101 + BATCH_SIZE - 1):
        source = _source(ordinal)
        runtime._replay_modalities[source] = MODALITY
        assert producer.offer(_lease(source), lane=ReadySourceLane.REPLAY)

    runtime._ready_source_owner = ready
    runtime._work_budget_owner = WorkBudgetOwner(
        requested_batch_size=BATCH_SIZE,
        maximum_camera_count=BATCH_SIZE,
        constellation_policy="fixed",
    )
    runtime._native_geometry_evidence_purpose = lambda: None
    runtime._historical_candidate_index = None
    runtime.initialized = True
    runtime.gaussians = SimpleNamespace(get_xyz=torch.empty((0, 3)))
    runtime.optimizer_steps = 0
    runtime._supervision_views_completed = 0
    runtime._control_tick_epoch = 1
    runtime._versions = lambda: (1, 1, 1)
    return runtime, commits


def test_bounded_pending_eviction_rearms_all_durable_commits_in_admission_order():
    """Removing durable-first selection would strand an admitted COMMIT."""

    runtime, commits = _runtime_with_unconsumed_commits()

    assert tuple(source.source_ordinal for source in runtime._pending_supervision_modalities) == (
        2,
        3,
        4,
    )
    for source in commits:
        snapshot, plan = runtime._rearm_prepared_pending_supervision()

        assert snapshot.supervision_candidates == (source,)
        assert tuple(work.source for work in plan.supervision_work) == (source,)
        assert runtime._observation_db.prepared_commit_obligation_for_source(source) is not None
        assert runtime._observation_db.clear_prepared_commit_obligation(source)

    assert runtime._rearm_prepared_pending_supervision() is None


@pytest.mark.parametrize("replay_count", (31, 0, 1, 2))
def test_source_insertion_b32_rearm_authorizes_one_fresh_b4_with_reserved_replay(replay_count):
    """One durable COMMIT may use the existing B4 camera contract, not B32."""

    runtime, commits = _runtime_with_unconsumed_commits()
    for committed in commits:
        assert runtime._observation_db.clear_prepared_commit_obligation(committed)
    source = _source(900)
    snapshot = replace(
        _snapshot(source),
        limits=ControlLimits(32, 32, 1, 0, 0, 0),
        supervision_candidates=(source,),
    )
    proposals = (
        ControlProposal("work", "supervision_slots", 31, "fresh"),
        ControlProposal("work", "replay_slots", 1, "resident replay"),
        ControlProposal("work", "optimizer_updates", 1, "one transaction"),
    )
    runtime._latest_control_outputs = (snapshot, proposals)
    runtime.record_prepared_disposition(_disposition(source, snapshot))
    context = runtime._prepared_execution_contexts_by_source[source]
    _durably_retain_commit(
        runtime._observation_db,
        source,
        snapshot,
        (object(), context.disposition, context.control_token, context),
    )

    replay = tuple(_source(ordinal) for ordinal in range(1_000, 1_000 + replay_count))
    modalities = OrderedDict(
        (
            candidate,
            ModalityEligibility(
                rgb_supervision=True,
                depth_supervision=bool(candidate.source_ordinal % 2),
            ),
        )
        for candidate in replay
    )
    reserved = replay[-1] if replay else None
    runtime._shortlisted_replay_view = lambda **_kwargs: (modalities, reserved)
    runtime._ready_source_owner = SimpleNamespace(
        replay_resident_sources=lambda: tuple(modalities)
    )
    runtime._replay_weights = {
        candidate: 1.0 + candidate.source_ordinal for candidate in replay
    }
    runtime._batch_size = 32
    runtime._work_budget_owner = WorkBudgetOwner(
        requested_batch_size=32,
        maximum_camera_count=32,
        constellation_policy="fixed",
    )
    runtime.config = _prepared_config(True)

    rearmed = runtime._rearm_prepared_pending_supervision()
    if replay_count < 3:
        assert rearmed is None
        assert runtime._observation_db.prepared_commit_obligation_for_source(source) is not None
        return

    rearmed, plan = rearmed
    assert rearmed.supervision_candidates == (source,)
    if replay_count >= 3:
        assert rearmed.replay_candidates == (replay[0], replay[1], reserved)
        assert plan.supervision_sources == (source,)
        assert len(plan.replay_sources) == 3
        assert reserved in plan.replay_sources
        assert plan.supervision_constellation.realized_batch_size == 4
        assert plan.supervision_constellation.realized_camera_count == 4
    assert rearmed.replay_work_modalities == tuple(
        modalities[candidate] for candidate in rearmed.replay_candidates
    )
    assert rearmed.replay_candidate_weights == tuple(
        runtime._replay_weights[candidate] for candidate in rearmed.replay_candidates
    )
    assert {
        work.source: work.modality_eligibility for work in plan.replay_work
    } == {candidate: modalities[candidate] for candidate in plan.replay_sources}
    assert plan.replay_selected_weights == tuple(
        runtime._replay_weights[candidate] for candidate in plan.replay_sources
    )
    assert runtime._observation_db.prepared_commit_obligation_for_source(source) is not None


def test_source_insertion_rearms_b4_when_a_third_replay_becomes_ready():
    """A shortage successor is retried by the normal READY freezer poll."""
    from eligsir.runtime.post_refinement import PostRefinementPhase

    runtime, commits = _runtime_with_unconsumed_commits()
    for committed in commits:
        assert runtime._observation_db.clear_prepared_commit_obligation(committed)
    source = _source(900)
    snapshot = replace(
        _snapshot(source),
        limits=ControlLimits(32, 32, 1, 0, 0, 0),
        supervision_candidates=(source,),
    )
    proposals = (
        ControlProposal("work", "supervision_slots", 31, "fresh"),
        ControlProposal("work", "replay_slots", 1, "resident replay"),
        ControlProposal("work", "optimizer_updates", 1, "one transaction"),
    )
    runtime._latest_control_outputs = (snapshot, proposals)
    runtime.record_prepared_disposition(_disposition(source, snapshot))
    context = runtime._prepared_execution_contexts_by_source[source]
    _durably_retain_commit(
        runtime._observation_db,
        source,
        snapshot,
        (object(), context.disposition, context.control_token, context),
    )
    replay = tuple(_source(1_000 + ordinal) for ordinal in range(32))
    shortlist = OrderedDict((candidate, MODALITY) for candidate in replay)
    owner = ReadySourceOwner(max_fresh=32, max_replay=64)
    producer = owner.claim_producer("b32-shortage")
    assert producer.offer(_lease(source), lane=ReadySourceLane.FRESH)
    for candidate in replay:
        assert producer.offer(_lease(candidate), lane=ReadySourceLane.REPLAY)
    runtime._ready_source_owner = owner
    runtime._shortlisted_replay_view = lambda **_kwargs: (shortlist, None)
    runtime._replay_weights = {candidate: 1.0 for candidate in replay}
    runtime._batch_size = 32
    runtime._work_budget_owner = WorkBudgetOwner(
        requested_batch_size=32, maximum_camera_count=32, constellation_policy="fixed"
    )
    runtime._native_geometry_evidence_purpose = lambda: None
    runtime.config = _prepared_config(True)
    runtime._prepared_ready_plan_for_execution = (
        CanonicalMappingRuntime._prepared_ready_plan_for_execution.__get__(runtime)
    )
    runtime._fresh_idle_snapshot = lambda value: value
    runtime._post_refinement_phase_owner = SimpleNamespace(phase=PostRefinementPhase.MAPPING)
    runtime._replay_depth_due = lambda: False

    rearmed_snapshot, rearmed_b4 = runtime._rearm_prepared_pending_supervision()
    full_b32 = runtime._idle_replay_plan(rearmed_snapshot, rearmed_b4)
    assert full_b32.supervision_constellation.realized_batch_size == 32
    runtime._prepared_ready_frozen_work = (rearmed_snapshot, full_b32)
    runtime._prepared_ready_candidate_work = (rearmed_snapshot, full_b32)

    shortlist = OrderedDict((candidate, MODALITY) for candidate in replay[:2])
    runtime._advance_prepared_ready_plan(full_b32)
    scarce_snapshot, scarce_plan = runtime._prepared_ready_candidate_work
    assert scarce_snapshot is rearmed_snapshot
    assert scarce_plan.supervision_constellation.realized_batch_size == 2
    assert runtime._observation_db.prepared_commit_obligation_for_source(source) is not None

    shortlist = OrderedDict((candidate, MODALITY) for candidate in replay[:3])
    assert runtime._prepared_ready_plan_for_execution() is not scarce_plan
    recovered_snapshot, recovered_plan = runtime._prepared_ready_candidate_work
    assert recovered_snapshot.supervision_candidates == (source,)
    assert recovered_plan.supervision_constellation.realized_batch_size == 4
    assert recovered_plan.supervision_constellation.realized_camera_count == 4
    assert runtime._prepared_ready_frozen_work == (recovered_snapshot, recovered_plan)
    assert runtime._observation_db.prepared_commit_obligation_for_source(source) is not None

    assert runtime._observation_db.clear_prepared_commit_obligation(source)
    shortlist = OrderedDict((candidate, MODALITY) for candidate in replay[:4])
    runtime._advance_prepared_ready_plan(recovered_plan)
    assert runtime._prepared_ready_plan_for_execution() is None
    assert runtime._prepared_ready_candidate_work[1].supervision_sources == ()
    assert runtime._prepared_ready_candidate_work[1].supervision_constellation.realized_batch_size == 4

    shortlist = OrderedDict((candidate, MODALITY) for candidate in replay)
    ordinary_b32 = runtime._prepared_ready_plan_for_execution()
    assert ordinary_b32.supervision_sources == ()
    assert ordinary_b32.supervision_constellation.realized_batch_size == 32
    assert ordinary_b32.supervision_constellation.realized_camera_count == 32


def test_two_b32_source_debts_complete_through_b4_then_return_to_b32_replay(monkeypatch):
    """The adapter clears exact debts, while the no-debt successor is ordinary B32 replay."""
    from eligsir.runtime.prepared_runtime_bridge import PreparedRuntimeBridge
    from eligsir.runtime.ready_b4_streaming_adapter import (
        ReadyB4StreamingOptimizationAdapter,
        ReadyB4StreamingStepInputs,
    )
    from eligsir.runtime import ready_b4_streaming_adapter as adapter_module
    from eligsir.runtime.post_refinement import PostRefinementPhase

    runtime, original = _runtime_with_unconsumed_commits()
    for source in original:
        assert runtime._observation_db.clear_prepared_commit_obligation(source)
    committed = tuple(_source(900 + index) for index in range(2))
    proposals = (
        ControlProposal("work", "supervision_slots", 31, "fresh"),
        ControlProposal("work", "replay_slots", 1, "resident replay"),
        ControlProposal("work", "optimizer_updates", 1, "one transaction"),
    )
    for source in committed:
        snapshot = replace(
            _snapshot(source),
            limits=ControlLimits(32, 32, 1, 0, 0, 0),
            supervision_candidates=(source,),
        )
        runtime._latest_control_outputs = (snapshot, proposals)
        runtime.record_prepared_disposition(_disposition(source, snapshot))
        context = runtime._prepared_execution_contexts_by_source[source]
        _durably_retain_commit(
            runtime._observation_db,
            source,
            snapshot,
            (object(), context.disposition, context.control_token, context),
        )

    replay = tuple(_source(1_000 + index) for index in range(32))
    runtime._replay_modalities = OrderedDict((source, MODALITY) for source in replay)
    runtime._replay_weights = {source: 1.0 for source in replay}
    owner = ReadySourceOwner(max_fresh=32, max_replay=64)
    transfer = DeviceTransferOwner("cpu", max_inflight=32, max_pinned_bytes=4096)
    producer = PreparedSourceProducer(owner, transfer_owner=transfer, producer_owner="b32")
    bridge = PreparedRuntimeBridge(
        PreparedSourceIngressHandoff(capacity=32),
        producer,
        ready_owner=owner,
        consumer_owner="b32",
        retained_commit_clearer=runtime.clear_retained_prepared_commit,
    )
    for source in committed:
        context = runtime._prepared_execution_contexts_by_source[source]
        packet = _packet(source)
        item = PreparedIngressItem(
            source.source_ordinal,
            packet,
            packet.tracking,
            context.disposition,
            context.control_token,
        )
        assert bridge.submit_retained_commit(item).value == "enqueued"
    ready_producer = owner.claim_producer("b32")
    for source in replay:
        assert ready_producer.offer(_lease(source), lane=ReadySourceLane.REPLAY)

    runtime._ready_source_owner = owner
    runtime._shortlisted_replay_view = lambda **_kwargs: (
        OrderedDict((source, MODALITY) for source in replay), replay[-1]
    )
    runtime._batch_size = 32
    runtime._work_budget_owner = WorkBudgetOwner(
        requested_batch_size=32, maximum_camera_count=32, constellation_policy="fixed"
    )
    runtime._native_geometry_evidence_purpose = lambda: None
    runtime.config = _prepared_config(True)
    runtime._prepared_ready_plan_for_execution = (
        CanonicalMappingRuntime._prepared_ready_plan_for_execution.__get__(runtime)
    )
    runtime._fresh_idle_snapshot = lambda snapshot: snapshot
    runtime._post_refinement_phase_owner = SimpleNamespace(phase=PostRefinementPhase.MAPPING)
    runtime._replay_depth_due = lambda: False
    runtime.tick = lambda *, ready_transaction: SimpleNamespace(
        executed=True,
        optimizer_steps=1,
        supervision_views=4,
        successful_supervision_sources=tuple(ready_transaction.plan.supervision_sources),
    )
    monkeypatch.setattr(
        adapter_module,
        "bind_ready_camera",
        lambda binding, *, tier: SimpleNamespace(
            source_identity=binding.lease.source_identity, camera=object()
        ),
    )
    monkeypatch.setattr(adapter_module, "_compute_completion", lambda: None)
    adapter = ReadyB4StreamingOptimizationAdapter(
        bridge,
        runtime=runtime,
        transfer_owner=transfer,
        batch_size=32,
        source_insertion=True,
    )

    completed = []
    for index, source in enumerate(committed):
        if index == 0:
            snapshot, plan = runtime._rearm_prepared_pending_supervision()
            runtime._prepared_ready_candidate_work = (snapshot, plan)
        else:
            snapshot, plan = runtime._prepared_ready_candidate_work
        assert plan.supervision_constellation.realized_batch_size == 4
        assert plan.supervision_constellation.realized_camera_count == 4
        assert runtime._prepared_ready_plan_for_execution() is plan
        if not completed:
            versions = runtime._versions
            runtime._versions = lambda: (99, 99, 99)
            deferred = adapter.tick(ReadyB4StreamingStepInputs(snapshot, plan))
            assert deferred.state.value == "stale"
            assert runtime._observation_db.prepared_commit_obligation_for_source(source) is not None
            assert runtime._prepared_ready_frozen_work == (snapshot, plan)
            runtime._versions = versions
        outcome = adapter.tick(ReadyB4StreamingStepInputs(snapshot, plan))
        assert outcome.completed_supervision_sources == (source,)
        assert runtime._observation_db.prepared_commit_obligation_for_source(source) is None
        completed.append(source)

    successor = runtime._prepared_ready_candidate_work[1]
    assert completed == list(committed)
    assert runtime._observation_db.next_prepared_commit_obligation_source() is None
    assert successor.supervision_sources == ()
    assert len(successor.replay_sources) == 32
    assert successor.supervision_constellation.realized_batch_size == 32
    assert successor.supervision_constellation.realized_camera_count == 32


@pytest.mark.parametrize(
    ("native_purpose", "live_population_authorization"),
    ((object(), False), (None, True)),
)
def test_source_insertion_b32_rearm_keeps_full_service_for_e0_or_live_population(
    native_purpose, live_population_authorization
):
    """E0 and live policy keep their existing configured-B32 authority."""

    runtime, commits = _runtime_with_unconsumed_commits()
    for committed in commits:
        assert runtime._observation_db.clear_prepared_commit_obligation(committed)
    source = _source(900)
    snapshot = replace(
        _snapshot(source),
        limits=ControlLimits(32, 32, 1, 0, 0, 0),
        supervision_candidates=(source,),
    )
    proposals = (
        ControlProposal("work", "supervision_slots", 31, "fresh"),
        ControlProposal("work", "replay_slots", 1, "resident replay"),
        ControlProposal("work", "optimizer_updates", 1, "one transaction"),
        *(
            (
                ControlProposal(
                    "population",
                    "live_population_authorization",
                    True,
                    "retain live policy authority",
                ),
            )
            if live_population_authorization
            else ()
        ),
    )
    runtime._latest_control_outputs = (snapshot, proposals)
    runtime.record_prepared_disposition(_disposition(source, snapshot))
    context = runtime._prepared_execution_contexts_by_source[source]
    _durably_retain_commit(
        runtime._observation_db,
        source,
        snapshot,
        (object(), context.disposition, context.control_token, context),
    )
    replay = tuple(_source(ordinal) for ordinal in range(1_000, 1_031))
    modalities = OrderedDict((candidate, MODALITY) for candidate in replay)
    runtime._shortlisted_replay_view = lambda **_kwargs: (modalities, replay[-1])
    runtime._ready_source_owner = SimpleNamespace(
        replay_resident_sources=lambda: tuple(modalities)
    )
    runtime._replay_weights = {candidate: 1.0 for candidate in replay}
    runtime._batch_size = 32
    runtime._work_budget_owner = WorkBudgetOwner(
        requested_batch_size=32,
        maximum_camera_count=32,
        constellation_policy="fixed",
    )
    runtime._native_geometry_evidence_purpose = lambda: native_purpose
    runtime.config = _prepared_config(True)

    rearmed, plan = runtime._rearm_prepared_pending_supervision()

    assert rearmed.replay_candidates == replay
    assert len(plan.replay_sources) == 31
    assert plan.supervision_constellation.realized_batch_size == 32
    assert plan.supervision_constellation.realized_camera_count == 32


def test_record_prepared_disposition_forwards_native_depth_requirement_from_typed_config():
    runtime, _commits = _runtime_with_unconsumed_commits()
    spy = _ResolveSpy()
    runtime._work_budget_owner = spy
    runtime.config = _prepared_config(True)
    source = _source(900)
    snapshot = _snapshot(source)
    runtime._latest_control_outputs = (
        snapshot,
        (ControlProposal("work", "optimizer_updates", 1, "one"),),
    )

    runtime.record_prepared_disposition(_disposition(source, snapshot))

    assert spy.kwargs["source_depth_requires_native"] is True


def test_rearm_prepared_pending_supervision_forwards_native_depth_requirement_from_typed_config():
    runtime, commits = _runtime_with_unconsumed_commits()
    spy = _ResolveSpy()
    runtime._work_budget_owner = spy
    runtime.config = _prepared_config(True)
    runtime._shortlisted_replay_view = lambda **_kwargs: (OrderedDict(), None)

    runtime._rearm_prepared_pending_supervision()

    assert spy.kwargs["source_depth_requires_native"] is True


def test_rgb_withdrawal_retires_unserviceable_commit_before_rearm():
    """The installed RGB retirement dispatcher must advance durable service."""

    from eligsir.runtime.historical_retirement import (
        install_modality_aware_historical_retirement,
    )

    runtime, commits = _runtime_with_unconsumed_commits()
    withdrawn, successor = commits[:2]
    frame_id = runtime._observation_db.frame_id_for_source(withdrawn)
    record = runtime._observation_db.get_frame(frame_id)
    record.modality_eligibility = ModalityEligibility(
        rgb_supervision=False,
        depth_supervision=True,
        depth_insertion=True,
    )
    runtime._historical_sources_by_frame_id = {frame_id: withdrawn}
    runtime._historical_frame_ids_by_source = {withdrawn: frame_id}
    active_snapshot, active_plan = runtime._rearm_prepared_pending_supervision()
    runtime._prepared_ready_frozen_work = (active_snapshot, active_plan)
    runtime._batch_size = BATCH_SIZE
    runtime._fresh_idle_snapshot = lambda snapshot: snapshot
    runtime._shortlisted_replay_view = lambda **_kwargs: (OrderedDict(), None)
    runtime._replay_depth_due = lambda: False
    install_modality_aware_historical_retirement(runtime)

    assert runtime._observation_db.historical_candidate_for_source(withdrawn) is not None
    assert (
        runtime._observation_db.historical_candidate_for_source(
            withdrawn, require_rgb=True
        )
        is None
    )
    assert runtime._retire_historical_identity(frame_id) is True
    assert runtime._observation_db.historical_candidate_for_source(withdrawn) is not None
    assert runtime._observation_db.prepared_commit_obligation_for_source(withdrawn) is None
    successor_plan = runtime._prepared_ready_candidate_work[1]
    assert withdrawn not in (*successor_plan.supervision_sources, *successor_plan.replay_sources)

    snapshot, plan = runtime._rearm_prepared_pending_supervision()

    assert snapshot.supervision_candidates == (successor,)
    assert tuple(work.source for work in plan.supervision_work) == (successor,)


def test_event_pending_commit_stays_durable_until_actual_service():
    """A pending READY event is a defer, never a synthetic first supervision."""

    class Gate:
        ready = False

        def query(self):
            return self.ready

    source = _source(9)
    snapshot = _snapshot(source)
    db = FrameObservationDB(device="cpu")
    _durably_retain_commit(
        db,
        source,
        snapshot,
        (object(), _disposition(source, snapshot), object(), object()),
    )
    gate = Gate()
    owner = ReadySourceOwner(max_fresh=1, max_replay=1)
    assert owner.claim_producer("event-pending").offer(
        _lease(source, ready_event=gate), lane=ReadySourceLane.FRESH
    )
    probe = dict(
        fresh_sources=frozenset((source,)), replay_sources=frozenset(), exact_sources=(source,)
    )

    assert owner.exact_readiness(**probe) == ((source, "EVENT_PENDING"),)
    assert db.prepared_commit_obligation_for_source(source) is not None
    gate.ready = True
    assert owner.exact_readiness(**probe) == ((source, "READY"),)
    assert db.prepared_commit_obligation_for_source(source) is not None


def _packet(source: SourceIdentity) -> SensorPacket:
    intrinsics = CameraIntrinsics(
        fx=2.0, fy=2.0, cx=1.0, cy=1.0, width=3, height=2, depth_scale=1000.0
    )
    observation = observation_from_pose_matrix(source, np.eye(4), provider="test")
    return SensorPacket(
        source=source,
        rgb=BufferRef.from_numpy(np.zeros((2, 3, 3), dtype=np.uint8), encoding="rgb8"),
        depth=None,
        intrinsics=intrinsics,
        tracking=observation,
    )


def test_evicted_commit_debt_drains_through_cold_promotion_and_successors(monkeypatch, tmp_path):
    """Only adapter-reported FRESH service advances every durable COMMIT."""

    from eligsir.runtime.prepared_cold_promotion import (
        FrameObservationPreparedColdResolver,
        PreparedColdPromotionOwner,
    )
    from eligsir.runtime.prepared_runtime_bridge import PreparedRuntimeBridge
    from eligsir.runtime.ready_b4_streaming_adapter import (
        ReadyB4StreamingOptimizationAdapter,
        ReadyB4StreamingStepInputs,
    )

    class Gate:
        ready = False

        def query(self):
            return self.ready

        def wait_on_current_stream(self, _device):
            return None

    runtime = object.__new__(CanonicalMappingRuntime)
    runtime._observation_db = FrameObservationDB(
        device="cpu",
        max_cold_payload_frames=1,
        cold_payload_spill_root=tmp_path / "prepared-cold-payloads",
    )
    runtime._latest_control_packet = None
    runtime._prepared_ready_candidate_work = None
    runtime._prepared_ready_frozen_work = None
    runtime._prepared_execution_contexts_by_source = OrderedDict()
    runtime._prepared_execution_context_capacity = BATCH_SIZE - 1
    runtime._prepared_execution_contexts = OrderedDict()
    runtime._dispositions_since_tick = deque(maxlen=64)
    runtime._replay_modalities = OrderedDict()
    runtime._replay_weights = {}
    runtime._pending_supervision_modalities = OrderedDict()
    runtime._pending_supervision_capacity = BATCH_SIZE - 1
    runtime._work_budget_owner = None
    runtime._work_clock_owner_if_available = lambda: None
    runtime._admitted_observations_total = 0
    runtime._services = None
    runtime._admitted_clock_ns = 0
    runtime._historical_candidate_index = None
    runtime._retired_replay_sources = OrderedDict()
    runtime._batch_size = BATCH_SIZE
    runtime._native_geometry_evidence_purpose = lambda: None
    runtime.initialized = True
    runtime.gaussians = SimpleNamespace(get_xyz=torch.empty((0, 3)))
    runtime.optimizer_steps = 0
    runtime._supervision_views_completed = 0
    runtime._control_tick_epoch = 1
    runtime._versions = lambda: (1, 1, 1)
    runtime._fresh_idle_snapshot = lambda snapshot: snapshot
    runtime._idle_replay_plan = lambda _snapshot, plan, **_kwargs: plan

    owner = ReadySourceOwner(max_fresh=BATCH_SIZE, max_replay=BATCH_SIZE)
    transfer = DeviceTransferOwner("cpu", max_inflight=BATCH_SIZE, max_pinned_bytes=4096)
    producer = PreparedSourceProducer(owner, transfer_owner=transfer, producer_owner="test")
    bridge = PreparedRuntimeBridge(
        PreparedSourceIngressHandoff(capacity=BATCH_SIZE),
        producer,
        ready_owner=owner,
        consumer_owner="test",
        retained_commit_clearer=runtime.clear_retained_prepared_commit,
        cold_payload_spill_writer=runtime._observation_db.write_pending_cold_payload_spills,
    )
    runtime._prepared_cold_payload_spill_bridge = bridge
    runtime._ready_source_owner = owner
    proposals = (
        ControlProposal("work", "supervision_slots", 1, "fresh"),
        ControlProposal("work", "replay_slots", BATCH_SIZE - 1, "ready replay"),
        ControlProposal("work", "optimizer_updates", 1, "one B4"),
    )
    commits = tuple(_source(ordinal) for ordinal in range(51, 55))
    retained = {}
    cold = commits[0]
    source_rgb = np.full((2, 3, 3), 51, dtype=np.uint8)
    source_depth = np.full((2, 3), 1051, dtype=np.uint16)
    corrected_pose = np.eye(4, dtype=np.float64)
    corrected_pose[0, 3] = 0.51
    for source in commits:
        snapshot = _snapshot(source)
        disposition = _disposition(source, snapshot)
        packet = _packet(source)
        if source == cold:
            observation = observation_from_pose_matrix(
                source, corrected_pose, provider="accepted-test"
            )
            packet = SensorPacket(
                source=source,
                rgb=BufferRef.from_numpy(
                    source_rgb,
                    encoding="rgb8",
                    ownership=BufferOwnership.BORROWED,
                    owner_token=object(),
                ),
                depth=BufferRef.from_numpy(
                    source_depth,
                    encoding="16UC1",
                    units="mm",
                    ownership=BufferOwnership.BORROWED,
                    owner_token=object(),
                ),
                intrinsics=packet.intrinsics,
                tracking=observation,
            )
        token = PreparedIngressControlToken(
            source.source_ordinal, source, snapshot.snapshot_id, disposition
        )
        item = PreparedIngressItem(source.source_ordinal, packet, packet.tracking, disposition, token)
        runtime._latest_control_outputs = (snapshot, proposals)
        runtime.record_prepared_disposition(disposition)
        if not runtime.can_accept_prepared_cold_payload(packet):
            assert bridge.drain().submitted == 0
            runtime._commit_prepared_cold_payload_spill_receipts()
        assert runtime.can_accept_prepared_cold_payload(packet)
        assert runtime.retain_prepared_source(item, lane=ReadySourceLane.FRESH)
        runtime.record_retained_prepared_commit(item)
        retained[source] = item

    assert bridge.drain().submitted == 0
    runtime._commit_prepared_cold_payload_spill_receipts()
    cold_frame_id = runtime._observation_db.frame_id_for_source(cold)
    cold_handle = runtime._observation_db.get_cold_pixels(cold_frame_id)
    assert cold_handle is not None and cold_handle.reload_token is not None
    source_rgb.fill(0)
    source_depth.fill(0)
    prefetched = {
        cold_frame_id: runtime._observation_db.load_cold_payload_from_token(
            cold_handle.reload_token
        )
    }

    runtime._work_budget_owner = WorkBudgetOwner(
        requested_batch_size=BATCH_SIZE,
        maximum_camera_count=BATCH_SIZE,
        constellation_policy="fixed",
    )
    replay = tuple(_source(ordinal) for ordinal in range(61, 64))
    for source in replay:
        runtime._replay_modalities[source] = MODALITY
        assert producer.submit(
            _packet(source), modality_eligibility=MODALITY, lane=ReadySourceLane.REPLAY
        ).value == "enqueued"
    runtime._shortlisted_replay_view = lambda **_kwargs: (
        OrderedDict((source, MODALITY) for source in replay),
        None,
    )
    for source in commits[1:]:
        assert bridge.submit_retained_commit(retained[source]).value == "enqueued"

    executed = []

    def tick(*, ready_transaction):
        fresh = tuple(ready_transaction.plan.supervision_sources)
        executed.append(fresh)
        return SimpleNamespace(
            executed=True,
            optimizer_steps=1,
            supervision_views=BATCH_SIZE,
            successful_supervision_sources=fresh,
        )

    runtime.tick = tick
    monkeypatch.setattr(
        "eligsir.runtime.ready_b4_streaming_adapter.bind_ready_camera",
        lambda binding, *, tier: SimpleNamespace(
            source_identity=binding.lease.source_identity, camera=object()
        ),
    )
    monkeypatch.setattr(
        "eligsir.runtime.ready_b4_streaming_adapter._compute_completion", lambda: None
    )
    adapter = ReadyB4StreamingOptimizationAdapter(
        bridge, runtime=runtime, transfer_owner=transfer
    )
    candidate = HistoricalReplayCandidate(
        frame_id=runtime._observation_db.frame_id_for_source(cold),
        weight=1.0,
        evidence=HistoricalCandidateEvidence(
            lanes=("local",), region_id=0, focus_weight=1.0, snapshot_id=1, epoch=1
        ),
    )
    resolved = []
    resolver = FrameObservationPreparedColdResolver(
        runtime._observation_db,
        context_for_source=runtime.prepared_cold_promotion_context_for,
        reload_pixels=lambda frame_id, _handle: prefetched.pop(frame_id, None),
    )
    promotion = PreparedColdPromotionOwner(
        bridge,
        ready_scheduler=owner.claim_scheduler("promotion"),
        resolve=lambda historical: (resolved.append(resolver(historical)) or resolved[-1]),
    )
    snapshot, plan = runtime._rearm_prepared_pending_supervision()
    assert snapshot.supervision_candidates == (cold,)
    runtime._prepared_ready_candidate_work = (snapshot, plan)
    before_promotion = transfer.accounting().h2d_ops
    assert promotion.schedule_shortlist((candidate,), promotion_fence=(1, 1, 1)) == 1
    assert transfer.accounting().h2d_ops == before_promotion
    assert promotion.pump_once()
    assert transfer.accounting().h2d_ops == before_promotion + 1
    assert np.array_equal(
        resolved[0].item.packet.rgb.data, np.full((2, 3, 3), 51, dtype=np.uint8)
    )
    assert np.array_equal(
        resolved[0].item.packet.depth.data, np.full((2, 3), 1051, dtype=np.uint16)
    )
    assert np.allclose(resolved[0].item.observation.pose.matrix, corrected_pose)
    assert runtime._prepared_ready_plan_for_execution() is plan

    gate = Gate()
    lease = owner.select(1, fresh_sources=frozenset((cold,)))[0]
    owner.requeue(lease, ready_event=gate)
    deferred = adapter.tick(ReadyB4StreamingStepInputs(snapshot, plan))
    assert not deferred.executed
    assert runtime._observation_db.prepared_commit_obligation_for_source(cold) is not None

    gate.ready = True
    for expected in commits:
        frozen = runtime._prepared_ready_plan_for_execution()
        if frozen is None:
            frozen = plan
        active_snapshot = runtime._prepared_ready_frozen_work[0]
        outcome = adapter.tick(ReadyB4StreamingStepInputs(active_snapshot, frozen))
        assert outcome.executed
        assert outcome.completed_supervision_sources == (expected,)
        assert runtime._observation_db.prepared_commit_obligation_for_source(expected) is None

    assert executed == [(source,) for source in commits]
    assert runtime._observation_db.next_prepared_commit_obligation_source() is None
