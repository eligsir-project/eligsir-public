from __future__ import annotations

from collections import OrderedDict, deque
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from eligsir.contracts import SourceIdentity
from eligsir.contracts.control import (
    ControlLimits,
    ControlSnapshot,
    DispositionKind,
    FreshnessMode,
    HistoricalEligibility,
    ModalityEligibility,
    ObservationDisposition,
    ResidencyIntent,
    SemanticAction,
)
from eligsir.mapping.admission.observation_owner import HistoricalPromotionPermit
from eligsir.mapping.observation_db import ColdPixelHandle, FrameObservationDB
from eligsir.mapping.scheduler.historical_candidate_index import (
    HistoricalCandidateIndex,
    HistoricalFrameMetadata,
)
from eligsir.contracts import BufferRef, CameraIntrinsics, SensorPacket
from eligsir.runtime.prepared_ingress import PreparedIngressControlToken
from eligsir.runtime.prepared_ingress import PreparedIngressItem
from eligsir.runtime.prepared_execution_context import (
    PreparedControlAxes,
    PreparedExecutionContext,
)
from eligsir.runtime.production_application import CanonicalMappingRuntime
from eligsir.runtime.ready_source import ReadySourceLane, ReplayPriority


def _source(ordinal: int) -> SourceIdentity:
    return SourceIdentity("camera", 0, ordinal, ordinal * 1_000)


def _snapshot(source: SourceIdentity, *, snapshot_id: int = 8) -> ControlSnapshot:
    return ControlSnapshot(
        snapshot_id=snapshot_id,
        control_version=4,
        source=source,
        tracking=None,
        pending_sources=(),
        newest_frontier=None,
        freshness_mode=FreshnessMode.HEALTHY,
        pending_oldest_age_ns=0,
        ingress_queue_depth=0,
        candidate_queue_depth=0,
        transfer_queue_depth=0,
        gpu_queue_depth=0,
        map_content_version=101,
        topology_version=102,
        parameter_version=103,
        gaussian_count=0,
        optimizer_steps_completed=0,
        limits=ControlLimits(1, 1, 1, 0, 0, 0),
    )


def _admit(source: SourceIdentity) -> ObservationDisposition:
    return ObservationDisposition(
        source=source,
        snapshot_id=8,
        kind=DispositionKind.ADMIT,
        observe=True,
        mapping_candidate=True,
        admit_supervision=True,
        allow_geometry_insertion=False,
        residency=ResidencyIntent.HOT,
        reason="admit",
        modality_eligibility=ModalityEligibility(rgb_supervision=True),
    )


def _defer(source: SourceIdentity) -> ObservationDisposition:
    return ObservationDisposition(
        source=source,
        snapshot_id=8,
        kind=DispositionKind.SKIP_REDUNDANT,
        observe=True,
        mapping_candidate=False,
        admit_supervision=False,
        allow_geometry_insertion=False,
        residency=ResidencyIntent.COLD,
        reason="defer",
        modality_eligibility=ModalityEligibility(rgb_supervision=True),
        semantic_action=SemanticAction.DEFER,
        historical_eligibility=HistoricalEligibility.ELIGIBLE,
    )


def _reject(source: SourceIdentity) -> ObservationDisposition:
    return ObservationDisposition(
        source=source,
        snapshot_id=8,
        kind=DispositionKind.SKIP_CAPACITY,
        observe=True,
        mapping_candidate=False,
        admit_supervision=False,
        allow_geometry_insertion=False,
        residency=ResidencyIntent.METADATA_ONLY,
        reason="reject",
        modality_eligibility=ModalityEligibility(),
    )


def _runtime(source: SourceIdentity, disposition: ObservationDisposition) -> CanonicalMappingRuntime:
    runtime = object.__new__(CanonicalMappingRuntime)
    runtime._latest_control_packet = None
    runtime._latest_control_outputs = (_snapshot(source), ())
    runtime._prepared_execution_contexts = {}
    runtime._prepared_execution_contexts_by_source = {}
    runtime._prepared_execution_context_capacity = 4
    runtime._dispositions_since_tick = deque(maxlen=16)
    runtime._replay_modalities = OrderedDict()
    runtime._pending_supervision_modalities = OrderedDict()
    runtime._pending_supervision_capacity = 2
    runtime._replay_weights = {}
    runtime._admitted_observations_total = 0
    runtime._admitted_clock_ns = 0
    runtime._services = None
    runtime._optimization_runtime = None
    runtime._work_clock_owner = None
    runtime.record_prepared_disposition(disposition)
    return runtime


def test_retire_completed_candidates_keeps_unsuccessful_prepared_commit_pending():
    """A partial B4 may only retire identities proven successful by the result."""

    fresh = _source(41)
    replay = _source(42)
    runtime = object.__new__(CanonicalMappingRuntime)
    runtime._pending_supervision_modalities = OrderedDict(
        (
            (fresh, ModalityEligibility(rgb_supervision=True)),
            (replay, ModalityEligibility(rgb_supervision=True)),
        )
    )
    plan = SimpleNamespace(
        supervision_sources=(fresh,),
        replay_sources=(replay,),
    )

    runtime._retire_completed_candidates(successful_sources=(replay,))

    assert tuple(runtime._pending_supervision_modalities) == (fresh,)


def test_prepared_admit_context_is_durable_and_records_catalogs_once():
    source = _source(1)
    disposition = _admit(source)
    runtime = _runtime(source, disposition)
    token = PreparedIngressControlToken(3, source, 8, disposition)

    context = runtime.prepared_execution_context_for(token)
    runtime.record_prepared_disposition(disposition)

    assert context.source == source
    assert context.disposition is disposition
    assert context.lane is ReadySourceLane.FRESH
    assert context.replay_priority is ReplayPriority.NORMAL
    assert context.control_axes.snapshot_id == 8
    assert context.snapshot is runtime._latest_control_outputs[0]
    assert not hasattr(context.control_axes, "map_content_version")
    assert runtime._admitted_observations_total == 1
    assert tuple(runtime._pending_supervision_modalities) == (source,)
    assert tuple(runtime._replay_modalities) == (source,)


def test_successful_prepared_defer_activates_its_admission_frontier_once():
    source = _source(2)
    disposition = _defer(source)
    runtime = _runtime(source, disposition)
    token = PreparedIngressControlToken(4, source, 8, disposition)
    assert runtime.prepared_execution_context_for(
        token,
        promotion_permit=HistoricalPromotionPermit(source, 8, 9, 5, 6),
    ) is not None
    admitted = []
    runtime._services = SimpleNamespace(
        four_clocks=SimpleNamespace(record_admitted_keyframe=lambda: admitted.append(source))
    )
    runtime._optimization_runtime = SimpleNamespace(
        note_admitted_source=lambda *, optimizer_updates_completed: None
    )
    runtime._work_clock_owner = SimpleNamespace(
        snapshot=lambda: SimpleNamespace(optimizer_updates=7)
    )

    runtime._activate_successful_prepared_deferred_sources((source,))
    runtime._activate_successful_prepared_deferred_sources((source,))

    assert runtime._admitted_clock_ns == source.timestamp_ns
    assert admitted == [source]
    assert source not in runtime._prepared_execution_contexts_by_source


def test_prepared_admission_preserves_an_exact_rearmed_b4_candidate():
    """A later single-source admission cannot erase executable READY authority."""

    original_sources = tuple(_source(index) for index in range(10, 14))
    exact_plan = SimpleNamespace(
        supervision_sources=original_sources[:1],
        replay_sources=original_sources[1:],
        supervision_constellation=SimpleNamespace(realized_batch_size=4, realized_camera_count=4),
    )
    later_source = _source(14)
    later_disposition = _admit(later_source)
    later_snapshot = _snapshot(later_source)
    incomplete_plan = SimpleNamespace(
        supervision_sources=(),
        replay_sources=(),
        supervision_constellation=SimpleNamespace(realized_batch_size=0, realized_camera_count=0),
    )

    runtime = _runtime(_source(1), _admit(_source(1)))
    runtime._batch_size = 4
    runtime._prepared_ready_candidate_work = (_snapshot(original_sources[0]), exact_plan)
    runtime._prepared_ready_frozen_work = None
    runtime._latest_control_packet = None
    runtime._latest_control_outputs = (later_snapshot, ())
    runtime._native_geometry_evidence_purpose = lambda: None
    runtime._work_budget_owner = SimpleNamespace(
        resolve=lambda *_args, **_kwargs: incomplete_plan
    )

    runtime.record_prepared_disposition(later_disposition)

    assert runtime._prepared_ready_frozen_work == (
        _snapshot(original_sources[0]),
        exact_plan,
    )
    assert runtime._prepared_ready_candidate_work == (
        _snapshot(original_sources[0]),
        exact_plan,
    )


def test_prepared_reject_never_becomes_ready_context():
    source = _source(2)
    disposition = _reject(source)
    runtime = _runtime(source, disposition)

    assert runtime.prepared_execution_context_for(
        PreparedIngressControlToken(4, source, 8, disposition)
    ) is None


def test_prepared_defer_requires_promotion_permit_and_uses_replay_lane():
    source = _source(3)
    disposition = _defer(source)
    runtime = _runtime(source, disposition)
    token = PreparedIngressControlToken(5, source, 8, disposition)

    assert runtime.prepared_execution_context_for(token) is None
    permit = HistoricalPromotionPermit(source, 8, 9, 5, 6)
    context = runtime.prepared_execution_context_for(token, promotion_permit=permit)

    assert context.lane is ReadySourceLane.REPLAY
    assert context.replay_priority is ReplayPriority.DEMANDED


@pytest.mark.parametrize("factory", (_defer, _reject, _admit))
def test_absent_nonpending_context_remains_a_none_lookup(factory):
    """Only a present malformed COMMIT debt is an invariant violation."""

    source = _source(35)
    disposition = factory(source)
    runtime = _runtime(source, disposition)
    runtime._prepared_execution_contexts.clear()
    runtime._prepared_execution_contexts_by_source.clear()
    runtime._observation_db = SimpleNamespace(
        prepared_commit_obligation_for_source=lambda _source: None
    )

    token = PreparedIngressControlToken(35, source, disposition.snapshot_id, disposition)

    assert runtime.prepared_execution_context_for(token) is None


def test_present_malformed_commit_context_fails_closed():
    source = _source(36)
    disposition = _admit(source)
    runtime = _runtime(source, disposition)
    runtime._prepared_execution_contexts.clear()
    runtime._prepared_execution_contexts_by_source.clear()
    runtime._observation_db = SimpleNamespace(
        prepared_commit_obligation_for_source=lambda _source: (object(),)
    )

    token = PreparedIngressControlToken(36, source, disposition.snapshot_id, disposition)

    with pytest.raises(RuntimeError, match="lacks execution context"):
        runtime.prepared_execution_context_for(token)


def test_retained_commit_is_shortlisted_for_fresh_cold_rebuild():
    """A full FRESH READY cache cannot erase a terminal COMMIT's first use."""

    from eligsir.runtime.prepared_cold_promotion import PreparedColdPromotionContext
    from eligsir.tracking.providers import observation_from_pose_matrix

    source = _source(41)
    disposition = _admit(source)
    intrinsics = CameraIntrinsics(
        fx=2.0, fy=2.0, cx=1.0, cy=1.0, width=3, height=2, depth_scale=1000.0
    )
    packet = SensorPacket(
        source=source,
        rgb=BufferRef.from_numpy(np.full((2, 3, 3), 128, dtype=np.uint8), encoding="rgb8"),
        depth=None,
        intrinsics=intrinsics,
        tracking=observation_from_pose_matrix(source, np.eye(4), provider="test"),
    )
    token = PreparedIngressControlToken(41, source, 8, disposition)
    item = PreparedIngressItem(41, packet, packet.tracking, disposition, token)
    runtime = object.__new__(CanonicalMappingRuntime)
    runtime._prepared_execution_contexts = {}
    runtime._prepared_execution_contexts_by_source = {}
    runtime._prepared_execution_context_capacity = 4
    runtime._observation_db = FrameObservationDB(device="cpu")
    runtime._historical_candidate_index = HistoricalCandidateIndex()
    runtime._historical_sources_by_frame_id = {}
    runtime._historical_frame_ids_by_source = {}
    runtime._hot_cold_challenger_arena = None
    runtime.iteration = 0
    runtime._latest_control_packet = None
    runtime._dispositions_since_tick = deque(maxlen=16)
    runtime._replay_modalities = OrderedDict()
    runtime._pending_supervision_modalities = OrderedDict()
    runtime._pending_supervision_capacity = 4
    runtime._replay_weights = {}
    runtime._admitted_observations_total = 0
    runtime._admitted_clock_ns = 0
    runtime._services = None
    runtime._optimization_runtime = None
    runtime._work_clock_owner = None
    runtime._latest_control_outputs = (_snapshot(source), ())
    runtime._prepared_execution_contexts_by_source[source] = PreparedExecutionContext(
        source=source,
        control_token=token,
        disposition=disposition,
        control_axes=PreparedControlAxes(
            snapshot_id=8, control_version=4, freshness_mode=FreshnessMode.HEALTHY,
            limits=_snapshot(source).limits, proposals=(),
        ),
        snapshot=_snapshot(source),
        lane=ReadySourceLane.FRESH,
        replay_priority=ReplayPriority.NORMAL,
    )

    assert runtime.retain_prepared_source(item, lane=ReadySourceLane.FRESH)
    runtime.record_retained_prepared_commit(item)
    record = runtime._observation_db.historical_candidate_for_source(source)
    assert record is not None
    context = runtime.prepared_cold_promotion_context_for(source.fingerprint())
    assert isinstance(context, PreparedColdPromotionContext)
    assert context.initial_fresh is True
    assert context.promotion_permit is None
    shortlist = runtime.prepared_cold_promotion_shortlist()
    assert tuple(candidate.frame_id for candidate in shortlist) == (record.frame_id,)

    runtime.clear_retained_prepared_commit(item)

    replay_context = runtime.prepared_cold_promotion_context_for(source.fingerprint())
    assert isinstance(replay_context, PreparedColdPromotionContext)
    assert replay_context.initial_fresh is False
    assert replay_context.promotion_permit is None
    assert tuple(candidate.frame_id for candidate in runtime.prepared_cold_promotion_shortlist()) == (
        record.frame_id,
    )


def test_prepared_cold_shortlist_only_pages_frozen_authorized_sources():
    """A frozen exact B4 never pages unrelated historical candidates."""

    sources = tuple(_source(60 + index) for index in range(4))
    frame_ids = tuple(160 + index for index in range(4))
    generic_source = _source(99)
    generic_ids = (260, 261)
    captured = {}
    records = {
        frame_id: SimpleNamespace(
            semantic_action=SemanticAction.COMMIT.value,
            prepared_commit_obligation=object(),
        )
        for frame_id in frame_ids
    }
    records.update(
        {
            frame_id: SimpleNamespace(
                semantic_action=SemanticAction.DEFER.value,
                prepared_defer_context=object(),
                source_identity=generic_source,
            )
            for frame_id in generic_ids
        }
    )

    def shortlist(**kwargs):
        captured.update(kwargs)
        return tuple(
            SimpleNamespace(frame_id=frame_id) for frame_id in (*generic_ids, *frame_ids)
        )

    runtime = object.__new__(CanonicalMappingRuntime)
    runtime._historical_candidate_index = SimpleNamespace(shortlist=shortlist)
    runtime._latest_control_outputs = (_snapshot(sources[0]), ())
    frozen_snapshot = _snapshot(sources[0], snapshot_id=9)
    runtime._prepared_ready_frozen_work = (
        frozen_snapshot,
        SimpleNamespace(supervision_sources=sources[:2], replay_sources=sources[2:]),
    )
    runtime._historical_frame_ids_by_source = dict(zip(sources, frame_ids, strict=True))
    runtime._observation_db = SimpleNamespace(get_frame=records.get)

    shortlist = runtime.prepared_cold_promotion_shortlist()

    assert captured["local_ids"] == frame_ids
    assert captured["snapshot_id"] == frozen_snapshot.snapshot_id
    assert tuple(candidate.frame_id for candidate in shortlist) == frame_ids


def test_prepared_cold_shortlist_keeps_pending_candidate_with_replay_fallback():
    """A pending trial augments rather than cancels active fallback promotion demand."""

    candidate_sources = tuple(_source(70 + index) for index in range(4))
    fallback_sources = tuple(_source(80 + index) for index in range(4))
    candidate_ids = tuple(170 + index for index in range(4))
    fallback_ids = tuple(180 + index for index in range(4))
    captured = {}
    records = {
        frame_id: SimpleNamespace(
            semantic_action=SemanticAction.COMMIT.value,
            prepared_commit_obligation=object(),
        )
        for frame_id in (*candidate_ids, *fallback_ids)
    }

    def shortlist(**kwargs):
        captured.update(kwargs)
        return tuple(SimpleNamespace(frame_id=frame_id) for frame_id in fallback_ids)

    runtime = object.__new__(CanonicalMappingRuntime)
    runtime._historical_candidate_index = SimpleNamespace(
        shortlist=shortlist,
        metadata_for=lambda frame_id: SimpleNamespace(
            base_weight=1.0, region_id=f"region-{frame_id}"
        ),
    )
    runtime._latest_control_outputs = (_snapshot(candidate_sources[0]), ())
    snapshot = _snapshot(candidate_sources[0], snapshot_id=10)
    runtime._prepared_ready_candidate_work = None
    runtime._prepared_ready_frozen_work = (
        snapshot,
        SimpleNamespace(supervision_sources=(), replay_sources=fallback_sources),
    )
    runtime._pending_replay_trial_sources = candidate_sources
    runtime._historical_frame_ids_by_source = dict(
        zip((*candidate_sources, *fallback_sources), (*candidate_ids, *fallback_ids), strict=True)
    )
    runtime._observation_db = SimpleNamespace(get_frame=records.get)

    shortlist = runtime.prepared_cold_promotion_shortlist()

    assert captured["local_ids"] == fallback_ids
    assert tuple(candidate.frame_id for candidate in shortlist) == (*fallback_ids, *candidate_ids)


def test_prepared_cold_shortlist_keeps_active_fresh_demand_with_pending_trial():
    """A pending trial must augment, never cancel, an active fresh page-in request."""

    fresh, trial = _source(90), _source(91)
    fresh_id, trial_id = 190, 191
    captured = {}
    records = {
        frame_id: SimpleNamespace(
            semantic_action=SemanticAction.COMMIT.value,
            prepared_commit_obligation=object(),
        )
        for frame_id in (fresh_id, trial_id)
    }

    def shortlist(**kwargs):
        captured.update(kwargs)
        return tuple(SimpleNamespace(frame_id=frame_id) for frame_id in (fresh_id, trial_id))

    runtime = object.__new__(CanonicalMappingRuntime)
    runtime._historical_candidate_index = SimpleNamespace(
        shortlist=shortlist,
        metadata_for=lambda frame_id: SimpleNamespace(base_weight=1.0, region_id=0),
    )
    snapshot = _snapshot(fresh, snapshot_id=11)
    runtime._latest_control_outputs = (snapshot, ())
    runtime._prepared_ready_candidate_work = (
        snapshot,
        SimpleNamespace(supervision_sources=(fresh,), replay_sources=()),
    )
    runtime._prepared_ready_frozen_work = None
    runtime._pending_replay_trial_sources = (trial,)
    runtime._historical_frame_ids_by_source = {fresh: fresh_id, trial: trial_id}
    runtime._observation_db = SimpleNamespace(get_frame=records.get)

    selected = runtime.prepared_cold_promotion_shortlist()

    assert captured["local_ids"] == (fresh_id,)
    assert tuple(candidate.frame_id for candidate in selected) == (fresh_id, trial_id)


def test_prepared_cold_shortlist_appends_pending_trial_outside_full_local_lane():
    """A B32 active plan leaves its exact pending trial outside the index lane."""

    active_sources = tuple(_source(100 + offset) for offset in range(32))
    trial = _source(132)
    active_ids = tuple(range(300, 332))
    trial_id = 332
    index = HistoricalCandidateIndex()
    for frame_id in (*active_ids, trial_id):
        index.register(HistoricalFrameMetadata(frame_id=frame_id, region_id=0))
    records = {
        frame_id: SimpleNamespace(
            semantic_action=SemanticAction.COMMIT.value,
            prepared_commit_obligation=object(),
        )
        for frame_id in (*active_ids, trial_id)
    }

    runtime = object.__new__(CanonicalMappingRuntime)
    runtime._historical_candidate_index = index
    snapshot = _snapshot(active_sources[0], snapshot_id=12)
    runtime._latest_control_outputs = (snapshot, ())
    runtime._prepared_ready_candidate_work = (
        snapshot,
        SimpleNamespace(supervision_sources=active_sources, replay_sources=()),
    )
    runtime._prepared_ready_frozen_work = None
    runtime._pending_replay_trial_sources = (trial,)
    runtime._historical_frame_ids_by_source = dict(
        zip((*active_sources, trial), (*active_ids, trial_id), strict=True)
    )
    runtime._observation_db = SimpleNamespace(get_frame=records.get)

    selected = runtime.prepared_cold_promotion_shortlist()

    assert {candidate.frame_id for candidate in selected} == {*active_ids, trial_id}


def test_prepared_retention_and_ready_record_preserve_full_intrinsics_identity():
    """A retained prepared source must bind with the same calibration record."""

    from eligsir.runtime.prepared_source_producer import PreparedSourceProducer
    from eligsir.runtime.ready_camera_binder import bind_ready_camera
    from eligsir.runtime.ready_only_tick import ReadySourceBinding, ReadySourceVersions
    from eligsir.runtime.ready_source import ReadySourceOwner
    from eligsir.runtime.transfers import TransferEvent
    from eligsir.tracking.providers import observation_from_pose_matrix

    class TransferOwner:
        device = "cpu"

        def prepare_source(self, arrays, **_kwargs):
            rgb, depth = arrays
            return (
                (torch.from_numpy(rgb).float().div(255.0), torch.from_numpy(depth).float().div(1000.0)),
                TransferEvent(),
                TransferEvent(),
            )

        def prepare_ready_calibration(self, *, pose, camera_intrinsics):
            return (
                (torch.tensor(pose), torch.tensor(camera_intrinsics)),
                TransferEvent(),
            )

        def prepare_source_tiers(self, rgb, depth, camera_intrinsics, **_kwargs):
            return {
                name: (rgb, depth, camera_intrinsics, None)
                for name in ("native", "half", "quarter")
            }, TransferEvent()

        def prepare_source_camera_geometry(self, c2w, tiers, *, w2c=None):
            if w2c is None:
                w2c = torch.linalg.inv(c2w)
            world_view = w2c.transpose(0, 1).contiguous()
            center = c2w[:3, 3].contiguous()
            common = (world_view, center, world_view.transpose(0, 1), center.view(1, 3))
            geometry = {
                name: (torch.eye(4), torch.eye(4))
                for name in tiers
            }
            return common, geometry, TransferEvent()

        def handoff_to_compute(self, *_events):
            pass

    source = _source(43)
    intrinsics = CameraIntrinsics(
        fx=2.0,
        fy=2.5,
        cx=1.0,
        cy=0.75,
        width=3,
        height=2,
        depth_fx=1.7,
        depth_fy=1.8,
        depth_cx=0.9,
        depth_cy=0.8,
        depth_width=3,
        depth_height=2,
        depth_scale=1000.0,
        intrinsics_source="streaming_frame",
    )
    tracking = observation_from_pose_matrix(source, np.eye(4), provider="test")
    packet = SensorPacket(
        source=source,
        rgb=BufferRef.from_numpy(np.full((2, 3, 3), 128, dtype=np.uint8), encoding="rgb8"),
        depth=BufferRef.from_numpy(np.full((2, 3), 1000, dtype=np.uint16), encoding="16UC1", units="mm"),
        intrinsics=intrinsics,
        tracking=tracking,
    )
    disposition = _admit(source)
    token = PreparedIngressControlToken(43, source, 8, disposition)
    item = PreparedIngressItem(43, packet, tracking, disposition, token)
    ready = ReadySourceOwner(max_fresh=1, max_replay=1)
    producer = PreparedSourceProducer(ready, transfer_owner=TransferOwner(), producer_owner="producer")
    producer.submit(packet, modality_eligibility=disposition.modality_eligibility, control_token=token)
    lease = ready.claim_consumer("consumer").poll()
    assert lease is not None
    assert len(lease.source_metadata["camera_intrinsics"]) == 18
    binding = bind_ready_camera(ReadySourceBinding(lease, ReadySourceVersions(1, 2, 3)))
    assert binding.metadata.intrinsics == intrinsics

    runtime = object.__new__(CanonicalMappingRuntime)
    runtime._observation_db = FrameObservationDB(device="cpu")
    runtime._register_historical_record = lambda *_args, **_kwargs: None
    assert runtime.retain_prepared_source(item, lane=ReadySourceLane.FRESH)
    durable = runtime._observation_db.historical_candidate_for_source(source)
    assert durable is not None
    durable_record = runtime._observation_db.get_frame(durable.frame_id)
    assert durable_record is not None
    assert durable_record.intrinsics == intrinsics

    runtime._record_ready_scene_candidate(
        type("Camera", (), {})(), binding, is_train=True, is_holdout=False
    )

    record = runtime._observation_db.get_frame(
        runtime._observation_db.frame_id_for_source(source)
    )
    assert record is not None
    assert record.intrinsics == intrinsics


def test_retained_commit_records_frozen_execution_context():
    """The pre-handoff sink records durable COMMIT debt without a frozen plan."""

    from eligsir.tracking.providers import observation_from_pose_matrix

    source = _source(42)
    disposition = _admit(source)
    intrinsics = CameraIntrinsics(
        fx=2.0, fy=2.0, cx=1.0, cy=1.0, width=3, height=2, depth_scale=1000.0
    )
    tracking = observation_from_pose_matrix(source, np.eye(4), provider="test")
    packet = SensorPacket(
        source=source,
        rgb=BufferRef.from_numpy(
            np.full((2, 3, 3), 128, dtype=np.uint8), encoding="rgb8"
        ),
        depth=None,
        intrinsics=intrinsics,
        tracking=tracking,
    )
    token = PreparedIngressControlToken(42, source, 8, disposition)
    item = PreparedIngressItem(42, packet, tracking, disposition, token)

    class DurableDB:
        def __init__(self):
            self.calls = []

        def record_prepared_commit_obligation(self, source_identity, obligation):
            self.calls.append((source_identity, obligation))

    runtime = object.__new__(CanonicalMappingRuntime)
    runtime._observation_db = DurableDB()
    context = object()
    runtime._prepared_execution_contexts_by_source = {source: context}

    # The async handoff preserves the source-local authority that later
    # re-entry needs after the bounded execution-context cache rolls.
    runtime.record_retained_prepared_commit(item)

    assert runtime._observation_db.calls == [
        (source, (tracking, disposition, token, context))
    ]


def test_retained_commit_obligation_survives_execution_context_capacity():
    """An admitted COMMIT remains page-in eligible after the context cache rolls.

    The prepared execution context table is intentionally bounded.  A retained
    COMMIT is different: it is an unresolved first-supervision obligation, so
    it must live with its durable historical record until it is either rebuilt
    or explicitly retired.  It must never be silently evicted just because
    later source-local execution contexts arrived.
    """

    from eligsir.runtime.prepared_cold_promotion import PreparedColdPromotionContext
    from eligsir.tracking.providers import observation_from_pose_matrix

    intrinsics = CameraIntrinsics(
        fx=2.0, fy=2.0, cx=1.0, cy=1.0, width=3, height=2, depth_scale=1000.0
    )
    runtime = object.__new__(CanonicalMappingRuntime)
    runtime._prepared_execution_contexts = OrderedDict()
    runtime._prepared_execution_contexts_by_source = OrderedDict()
    runtime._prepared_execution_context_capacity = 4
    runtime._observation_db = FrameObservationDB(device="cpu")
    runtime._historical_candidate_index = HistoricalCandidateIndex()
    runtime._historical_sources_by_frame_id = {}
    runtime._historical_frame_ids_by_source = {}
    runtime._hot_cold_challenger_arena = None
    runtime.iteration = 0
    runtime._latest_control_packet = None
    runtime._dispositions_since_tick = deque(maxlen=16)
    runtime._replay_modalities = OrderedDict()
    runtime._pending_supervision_modalities = OrderedDict()
    runtime._pending_supervision_capacity = 4
    runtime._replay_weights = {}
    runtime._admitted_observations_total = 0
    runtime._admitted_clock_ns = 0
    runtime._services = None
    runtime._optimization_runtime = None
    runtime._work_clock_owner = None

    items = []
    for ordinal in range(100, 5101):
        source = _source(ordinal)
        disposition = _admit(source)
        tracking = observation_from_pose_matrix(source, np.eye(4), provider="test")
        packet = SensorPacket(
            source=source,
            rgb=BufferRef.from_numpy(
                np.full((2, 3, 3), ordinal, dtype=np.uint8), encoding="rgb8"
            ),
            depth=None,
            intrinsics=intrinsics,
            tracking=tracking,
        )
        token = PreparedIngressControlToken(ordinal, source, 8, disposition)
        item = PreparedIngressItem(ordinal, packet, tracking, disposition, token)
        runtime._latest_control_packet = packet
        runtime._latest_control_outputs = (_snapshot(source), ())
        runtime.record_prepared_disposition(disposition)
        assert runtime.retain_prepared_source(item, lane=ReadySourceLane.FRESH)
        runtime.record_retained_prepared_commit(item)
        items.append(item)

    first = items[0]
    assert first.packet.source not in runtime._prepared_execution_contexts_by_source
    context = runtime.prepared_cold_promotion_context_for(first.packet.source.fingerprint())

    assert isinstance(context, PreparedColdPromotionContext)
    assert context.initial_fresh is True
    assert context.control_token == first.control_token
    assert context.disposition == first.disposition
    rebuilt = runtime.prepared_execution_context_for(first.control_token)
    assert rebuilt is not None
    assert rebuilt.source == first.packet.source
    assert rebuilt.lane is ReadySourceLane.FRESH
    shortlist_sources = {
        runtime._observation_db.get_frame(candidate.frame_id).source_identity
        for candidate in runtime.prepared_cold_promotion_shortlist()
    }
    assert first.packet.source.fingerprint() in shortlist_sources

    runtime.clear_retained_prepared_commit(first)

    replay_context = runtime.prepared_cold_promotion_context_for(
        first.packet.source.fingerprint()
    )
    assert isinstance(replay_context, PreparedColdPromotionContext)
    assert replay_context.initial_fresh is False


def test_prepared_defer_retains_durable_cold_before_it_can_leave_ready_residency():
    """A terminal DEFER cannot disappear merely because READY is bounded."""

    source = _source(30)
    disposition = _defer(source)
    runtime = object.__new__(CanonicalMappingRuntime)
    packet = type("Packet", (), {"source": source, "tracking": "tracking"})()
    runtime._latest_control_packet = packet
    runtime._latest_control_outputs = (_snapshot(source), ())
    runtime._prepared_execution_contexts = {}
    runtime._prepared_execution_contexts_by_source = {}
    runtime._prepared_execution_context_capacity = 4
    runtime._dispositions_since_tick = deque(maxlen=16)
    runtime._replay_modalities = OrderedDict()
    runtime._pending_supervision_modalities = OrderedDict()
    runtime._pending_supervision_capacity = 2
    runtime._replay_weights = {}
    runtime._admitted_observations_total = 0
    runtime._admitted_clock_ns = 0
    runtime._services = None
    runtime._optimization_runtime = None
    runtime._work_clock_owner = None
    retained = []

    class _DurableDB:
        def __init__(self):
            self.calls = []

        def record_prepared_defer_context(self, source_identity, context):
            self.calls.append((source_identity, context))

    runtime._observation_db = _DurableDB()

    class _Owner:
        def cold_retention_permit(self, observed):
            assert observed is disposition
            return "cold-permit"

    runtime.observation_owner = _Owner()
    runtime._retain_deferred_observation = lambda *args: retained.append(args)

    runtime.record_prepared_disposition(disposition)

    assert retained == [(disposition, "cold-permit", packet, "tracking")]
    assert runtime._observation_db.calls == [
        (source, ("tracking", disposition, runtime._prepared_execution_contexts_by_source[source].control_token))
    ]
    assert runtime._replay_modalities[source] is disposition.modality_eligibility
    assert runtime._replay_weights[source] == 1.0


def test_prepared_defer_context_survives_more_than_4096_historical_records():
    """DEFER page-in authority is durable rather than an active-context LRU."""

    db = FrameObservationDB(device="cpu")
    intrinsics = CameraIntrinsics(fx=2.0, fy=2.0, cx=1.0, cy=1.0, width=2, height=2)
    first_source = _source(100)
    first_context = ("tracking", "disposition", "token")
    for ordinal in range(100, 5101):
        source = _source(ordinal)
        db.record_historical_candidate(
            source_identity=source,
            pose_c2w=np.eye(4, dtype=np.float32),
            timestamp=float(ordinal),
            intrinsics=intrinsics,
            cold_pixels=ColdPixelHandle(
                source_identity=source.fingerprint(),
                rgb=np.zeros((2, 2, 3), dtype=np.uint8),
                depth=None,
                intrinsics=intrinsics,
            ),
            modality_eligibility=ModalityEligibility(rgb_supervision=True),
            disposition_snapshot_id=ordinal,
        )
        db.record_prepared_defer_context(
            source,
            first_context if source == first_source else (ordinal,),
        )

    assert db.prepared_defer_context_for_source(first_source.fingerprint()) == first_context


def test_cold_defer_context_waits_for_newer_control_snapshot():
    """A retained DEFER is pending at its disposition snapshot, not an error."""

    from eligsir.mapping.admission.observation_owner import ObservationOwner
    from eligsir.tracking.providers import observation_from_pose_matrix

    source = _source(31)
    disposition = _defer(source)
    token = PreparedIngressControlToken(31, source, disposition.snapshot_id, disposition)
    tracking = observation_from_pose_matrix(source, np.eye(4), provider="test")
    candidate = type(
        "Candidate",
        (),
        {
            "source": source,
            "disposition_snapshot_id": disposition.snapshot_id,
            "semantic_action": disposition.semantic_action.value,
        },
    )()
    runtime = object.__new__(CanonicalMappingRuntime)
    runtime._observation_db = type(
        "ObservationDB",
        (),
        {
            "prepared_defer_context_for_source": (
                lambda _self, *_args, **_kwargs: (tracking, disposition, token)
            ),
            "historical_candidate_for_source": lambda _self, *_args, **_kwargs: candidate,
        },
    )()
    runtime.observation_owner = object.__new__(ObservationOwner)
    runtime._latest_control_outputs = (_snapshot(source), ())

    assert runtime.prepared_cold_promotion_context_for(source.fingerprint()) is None

    runtime._latest_control_outputs = (_snapshot(source, snapshot_id=9), ())
    context = runtime.prepared_cold_promotion_context_for(source.fingerprint())

    assert context is not None
    assert context.promotion_permit.source == source
    assert context.promotion_permit.disposition_snapshot_id == disposition.snapshot_id
    assert context.promotion_permit.promotion_snapshot_id == 9


def test_cold_defer_context_uses_newer_frozen_prepared_snapshot_after_eof():
    """Refinement rearm, not the final source snapshot, authorizes page-in."""

    from eligsir.mapping.admission.observation_owner import ObservationOwner
    from eligsir.tracking.providers import observation_from_pose_matrix

    source = _source(32)
    disposition = _defer(source)
    token = PreparedIngressControlToken(32, source, disposition.snapshot_id, disposition)
    tracking = observation_from_pose_matrix(source, np.eye(4), provider="test")
    candidate = type(
        "Candidate",
        (),
        {
            "source": source,
            "disposition_snapshot_id": disposition.snapshot_id,
            "semantic_action": disposition.semantic_action.value,
        },
    )()
    final_snapshot = _snapshot(source)
    refinement_snapshot = _snapshot(source, snapshot_id=9)
    runtime = object.__new__(CanonicalMappingRuntime)
    runtime._observation_db = type(
        "ObservationDB",
        (),
        {
            "prepared_defer_context_for_source": (
                lambda _self, *_args, **_kwargs: (tracking, disposition, token)
            ),
            "historical_candidate_for_source": lambda _self, *_args, **_kwargs: candidate,
        },
    )()
    runtime.observation_owner = object.__new__(ObservationOwner)
    runtime._latest_control_outputs = (final_snapshot, ())
    runtime._prepared_ready_frozen_work = (
        refinement_snapshot,
        SimpleNamespace(supervision_sources=(source,), replay_sources=()),
    )

    context = runtime.prepared_cold_promotion_context_for(source.fingerprint())

    assert context is not None
    assert context.promotion_permit.promotion_snapshot_id == refinement_snapshot.snapshot_id


def test_cold_defer_context_keeps_frozen_work_authority_through_control_churn():
    """Unrelated controls cannot revoke a COLD request for a frozen exact B4."""

    from eligsir.mapping.admission.observation_owner import ObservationOwner
    from eligsir.tracking.providers import observation_from_pose_matrix

    source = _source(33)
    disposition = _defer(source)
    token = PreparedIngressControlToken(33, source, disposition.snapshot_id, disposition)
    tracking = observation_from_pose_matrix(source, np.eye(4), provider="test")
    candidate = type(
        "Candidate",
        (),
        {
            "source": source,
            "disposition_snapshot_id": disposition.snapshot_id,
            "semantic_action": disposition.semantic_action.value,
        },
    )()
    runtime = object.__new__(CanonicalMappingRuntime)
    runtime._observation_db = type(
        "ObservationDB",
        (),
        {
            "prepared_defer_context_for_source": (
                lambda _self, *_args, **_kwargs: (tracking, disposition, token)
            ),
            "historical_candidate_for_source": lambda _self, *_args, **_kwargs: candidate,
        },
    )()
    runtime.observation_owner = object.__new__(ObservationOwner)
    runtime._prepared_ready_frozen_work = (
        _snapshot(source, snapshot_id=12),
        SimpleNamespace(supervision_sources=(source,), replay_sources=()),
    )
    runtime._prepared_ready_candidate_work = (
        _snapshot(source, snapshot_id=13),
        SimpleNamespace(supervision_sources=(source,), replay_sources=()),
    )

    for snapshot_id in range(20, 1_020):
        runtime._latest_control_outputs = (_snapshot(source, snapshot_id=snapshot_id), ())
        context = runtime.prepared_cold_promotion_context_for(source.fingerprint())
        assert context is not None
        assert context.promotion_permit.promotion_snapshot_id == 13



def test_context_rejects_a_foreign_control_token_for_the_same_source():
    source = _source(4)
    disposition = _admit(source)
    runtime = _runtime(source, disposition)

    token = PreparedIngressControlToken(6, source, 8, disposition)
    assert runtime.prepared_execution_context_for(token) is not None

    with pytest.raises(ValueError, match="token differs"):
        runtime.prepared_execution_context_for(
            PreparedIngressControlToken(7, source, 8, disposition)
        )
