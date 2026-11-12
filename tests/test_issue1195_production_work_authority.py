"""Behavioral guards that make MappingWorkPlan production authority."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from collections import OrderedDict
from dataclasses import replace
from pathlib import Path
import threading
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from eligsir.contracts.control import (
    ControlLimits,
    ControlProposal,
    ControlSnapshot,
    DispositionKind,
    FreshnessMode,
    DepthInsertionPlan,
    MappingTransactionClass,
    MappingWorkPlan,
    ModalityEligibility,
    MutationBudget,
    ObservationDisposition,
    ResidencyIntent,
    ResolutionTier,
    SupervisionConstellation,
    SupervisionWork,
    EvidenceFact,
)
from eligsir.contracts.runtime import (
    BufferOwnership,
    BufferRef,
    MappingFrame,
    SensorPacket,
    SourceIdentity,
    TrackingObservation,
    TrackingStatus,
)
from eligsir.contracts.types import CameraIntrinsics, Pose
from eligsir.mapping.scheduler.work_budget import (
    AdaptiveGrowthProposalProvider,
    MutationProposalProvider,
    WorkBudgetOwner,
)
from eligsir.runtime.production_application import (
    CanonicalMappingRuntime as ContinualMappingRuntime,
    depth_insertion_policy_from_args,
    motion_admission_evidence,
    depth_supervision_enabled,
    scheduled_b1_work,
)
import eligsir.runtime.production_application as production_application
from eligsir.runtime.config import CanonicalOptimizationPolicy, DepthInsertionPolicy
from eligsir.contracts.work_clocks import RuntimeWorkClockOwner
from eligsir.mapping.scheduler.mcmc_cadence import MCMCCadenceOwner
from eligsir.mapping.observation_db import (
    BufferStage,
    CacheHandle,
    ColdPixelHandle,
    FrameObservationDB,
    FrameRecord,
)
from eligsir.map.gpu_transactions import (
    BatchExecutionMode,
    FixedTopologyBatchBuilder,
    GpuWorkEnvelope,
)
from eligsir.runtime.post_refinement import (
    PostRefinementPhase,
    PostRefinementPhaseOwner,
)
from eligsir.runtime.session import MappingSession
from eligsir.runtime.streaming_observation import StreamingObservation
from eligsir.runtime.prepared_ingress import PreparedIngressDeferred
from eligsir.runtime.application import ComposedMappingApplication, MappingApplicationOptions
from eligsir.tracking.providers import ExternalTrackingProvider


_DISABLED_REPLAY_CHALLENGER_POLICY = CanonicalOptimizationPolicy(
    {
        "streaming_replay_challenger_fraction": 0.0,
        "streaming_replay_challenger_demotion_cadence": 0,
        "streaming_replay_challenger_demotion_limit": 0,
        "streaming_replay_challenger_lease_ticks": 100,
    }
)


def _source(ordinal: int) -> SourceIdentity:
    return SourceIdentity("camera", 0, ordinal, 1_000 + ordinal)


def _admitted_modalities() -> ModalityEligibility:
    return ModalityEligibility(True, "qualified", True, "qualified", True)


def test_depth_insertion_policy_resolves_static_device_semantics_at_boundary():
    policy = depth_insertion_policy_from_args(
        SimpleNamespace(
            streaming_min_depth=0.2,
            streaming_max_depth=6.0,
            streaming_depth_edge_threshold=0.03,
            streaming_max_view_angle=65.0,
            streaming_depth_filter_enabled=True,
            streaming_depth_filter_erode_px=3,
            streaming_depth_filter_window=7,
            streaming_depth_filter_min_valid_ratio=0.8,
            streaming_depth_filter_max_range=0.09,
            streaming_depth_filter_median_thresh=0.05,
            streaming_depth_filter_bottom_margin=0.1,
            streaming_depth_temporal_check=True,
            streaming_depth_temporal_thresh=0.08,
            streaming_depth_temporal_require_for_insert=True,
            streaming_depth_temporal_require_for_sensor_closer=True,
            streaming_depth_consistency_thresh=0.06,
            streaming_insert_alpha_threshold=0.4,
            streaming_insert_score_mode="weighted_topk",
            streaming_sparse_seed_voxel_size=0.03,
            streaming_sparse_seed_min_confidence=0.25,
        )
    )
    assert policy.stability_filter_enabled
    assert policy.stability_window_px == 7
    assert policy.temporal_check_enabled
    assert policy.require_temporal_stability
    assert policy.temporal_tolerance_m == 0.08
    assert policy.occupancy_distance_m == 0.06
    assert policy.score_mode == "weighted_topk"
    assert policy.minimum_confidence == 0.25


def test_canonical_device_planner_has_no_depth_or_insertion_args_controls():
    import ast

    source = Path("eligsir/mapping/integration/device_integration_planner.py").read_text()
    module = ast.parse(source)
    executor = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "insert_gaussians_from_frame_device_native"
    )
    assert "args" not in {argument.arg for argument in executor.args.args}
    assert "modality_eligibility" in {
        argument.arg for argument in executor.args.kwonlyargs
    }


def test_candidate_voxel_is_independent_from_canonical_occupancy_index():
    from eligsir.mapping.canonical_scene import CanonicalStreamingScene
    from eligsir.utils.torch_geometry import pack_voxel_keys_torch

    existing = torch.tensor([[0.01, 0.0, 0.0]], dtype=torch.float32)
    scene = object.__new__(CanonicalStreamingScene)
    scene.occupancy_voxel_size = 0.12
    scene.gaussians = SimpleNamespace(get_xyz=existing)
    scene._occupancy_keys_gpu = torch.unique(
        pack_voxel_keys_torch(existing, scene.occupancy_voxel_size), sorted=True
    )

    occupied = scene.check_occupancy_torch(
        torch.tensor([[0.02, 0.0, 0.0], [1.0, 1.0, 1.0]]),
        0.05,
        check_neighbors=True,
    )

    assert occupied.tolist() == [True, False]


def test_motion_admission_evidence_uses_translation_or_rotation_threshold():
    assert motion_admission_evidence(0.05, 0.0, min_translation_m=0.05, min_rotation_deg=5.0)
    assert motion_admission_evidence(0.0, 5.0, min_translation_m=0.05, min_rotation_deg=5.0)
    assert not motion_admission_evidence(0.049, 4.9, min_translation_m=0.05, min_rotation_deg=5.0)


def _snapshot(source: SourceIdentity) -> ControlSnapshot:
    return ControlSnapshot(
        snapshot_id=17,
        control_version=1,
        source=source,
        tracking=None,
        pending_sources=(source,),
        newest_frontier=source,
        freshness_mode=FreshnessMode.HEALTHY,
        pending_oldest_age_ns=0,
        ingress_queue_depth=0,
        candidate_queue_depth=1,
        transfer_queue_depth=0,
        gpu_queue_depth=0,
        map_content_version=3,
        topology_version=2,
        parameter_version=3,
        gaussian_count=10,
        optimizer_steps_completed=4,
        limits=ControlLimits(1, 0, 1, 7, 0, 1024),
        supervision_candidates=(source,),
    )


def _proposals() -> tuple[ControlProposal, ...]:
    return (
        ControlProposal("test", "supervision_slots", 1, "one view"),
        ControlProposal("test", "optimizer_updates", 1, "one update"),
        ControlProposal("test", "insertion_enabled", True, "enabled"),
        ControlProposal("test", "insertion_seeds", 7, "bounded insertion"),
        ControlProposal("test", "allow_growth", False, "fail closed"),
        ControlProposal("test", "allow_relocation", False, "fail closed"),
        ControlProposal("test", "mutation_budget", 0, "fail closed"),
    )


def _publish_snapshot_outputs(runtime, snapshot) -> None:
    runtime._latest_control_snapshot = snapshot
    runtime._latest_control_outputs = (snapshot, _proposals())


def _plan(source: SourceIdentity, *, insertion_seeds: int = 7) -> MappingWorkPlan:
    return MappingWorkPlan(
        plan_id="test-plan",
        snapshot_id=17,
        control_version=1,
        supervision_sources=(source,),
        replay_sources=(),
        resolution_tier="full",
        optimizer_updates=1,
        insertion=DepthInsertionPlan(
            source=source if insertion_seeds else None,
            enabled=bool(insertion_seeds),
            sample_stride=1,
            candidate_budget=insertion_seeds,
            seed_budget=insertion_seeds,
            voxel_size_m=0.02,
            minimum_confidence=0.0,
        ),
        allow_growth=False,
        allow_relocation=False,
        mutation_budget=MutationBudget(0),
        freshness_mode=FreshnessMode.HEALTHY,
        map_content_version=3,
        topology_version=2,
        parameter_version=3,
        supervision_work=(
            SupervisionWork(source, "native", _admitted_modalities()),
        ),
    )


def test_disposition_and_plan_share_the_exact_snapshot():
    source = _source(3)
    runtime = object.__new__(ContinualMappingRuntime)
    runtime._dispositions_since_tick = __import__("collections").deque(maxlen=4096)
    runtime._latest_tick_work = None
    runtime._latest_report_work = None
    runtime._active_materialization_work = None
    _publish_snapshot_outputs(runtime, _snapshot(source))
    runtime._work_budget_owner = WorkBudgetOwner()
    disposition = ObservationDisposition(
        source=source,
        snapshot_id=17,
        kind=DispositionKind.ADMIT,
        observe=True,
        mapping_candidate=True,
        admit_supervision=True,
        allow_geometry_insertion=True,
        residency=ResidencyIntent.HOT,
        reason="admitted",
        modality_eligibility=_admitted_modalities(),
    )

    runtime.record_disposition(disposition)

    snapshot, plan = runtime._latest_tick_work
    assert snapshot is runtime._latest_control_snapshot
    assert snapshot.snapshot_id == plan.snapshot_id == disposition.snapshot_id
    assert plan.supervision_sources == (source,)
    assert plan.supervision_work[0].source == source
    assert runtime._replay_modalities[source] == disposition.modality_eligibility
    assert runtime._active_materialization_work == (snapshot, plan)
    assert runtime._pending_supervision_modalities == {
        source: disposition.modality_eligibility
    }


def test_admitted_disposition_with_zero_slot_proposal_keeps_materialized_source_in_plan():
    source = _source(3)
    runtime = object.__new__(ContinualMappingRuntime)
    runtime._dispositions_since_tick = __import__("collections").deque(maxlen=4096)
    runtime._latest_tick_work = None
    runtime._latest_report_work = None
    runtime._active_materialization_work = None
    snapshot = _snapshot(source)
    runtime._latest_control_snapshot = snapshot
    runtime._latest_control_outputs = (
        snapshot,
        (
            ControlProposal("safety", "supervision_slots", 0, "temporary ceiling"),
            ControlProposal("optimizer", "optimizer_updates", 1, "one update"),
        ),
    )
    runtime._work_budget_owner = WorkBudgetOwner()
    disposition = ObservationDisposition(
        source=source,
        snapshot_id=snapshot.snapshot_id,
        kind=DispositionKind.ADMIT,
        observe=True,
        mapping_candidate=True,
        admit_supervision=True,
        allow_geometry_insertion=True,
        residency=ResidencyIntent.HOT,
        reason="admitted",
        modality_eligibility=_admitted_modalities(),
    )

    runtime.record_disposition(disposition)

    active_snapshot, active_plan = runtime._active_materialization_work
    assert active_snapshot is snapshot
    assert active_plan.requested_supervision_slots == 0
    assert active_plan.supervision_sources == (source,)
    assert active_plan.supervision_work[0].source == source


def test_pending_admitted_candidates_keep_identity_modalities_until_execution():
    first, newest = _source(4), _source(5)
    runtime = object.__new__(ContinualMappingRuntime)
    runtime._dispositions_since_tick = __import__("collections").deque(maxlen=4096)
    runtime._latest_tick_work = None
    runtime._latest_report_work = None
    runtime._active_materialization_work = None
    runtime._pending_supervision_modalities = __import__("collections").OrderedDict()
    runtime._pending_supervision_capacity = 3
    runtime._replay_modalities = __import__("collections").OrderedDict()
    runtime._replay_weights = {}
    runtime._work_budget_owner = WorkBudgetOwner()

    first_snapshot = _snapshot(first)
    _publish_snapshot_outputs(runtime, first_snapshot)
    first_disposition = ObservationDisposition(
        source=first,
        snapshot_id=first_snapshot.snapshot_id,
        kind=DispositionKind.ADMIT,
        observe=True,
        mapping_candidate=True,
        admit_supervision=True,
        allow_geometry_insertion=False,
        residency=ResidencyIntent.HOT,
        reason="admitted",
        modality_eligibility=ModalityEligibility(
            True, "qualified", False, "missing", False
        ),
    )
    runtime.record_disposition(first_disposition)

    newest_snapshot = replace(
        _snapshot(newest),
        snapshot_id=18,
        pending_sources=(first, newest),
        supervision_candidates=(first, newest),
        limits=ControlLimits(3, 3, 1, 7, 0, 1024),
        replay_candidates=(first,),
        replay_work_modalities=(first_disposition.modality_eligibility,),
    )
    newest_disposition = replace(
        first_disposition,
        source=newest,
        snapshot_id=18,
        allow_geometry_insertion=True,
        modality_eligibility=_admitted_modalities(),
    )
    runtime._latest_control_snapshot = newest_snapshot
    runtime._latest_control_outputs = (
        newest_snapshot,
        (
            ControlProposal("constellation", "supervision_slots", 2, "pending drain"),
            ControlProposal("constellation", "replay_slots", 2, "C4 fill"),
            ControlProposal("optimizer", "optimizer_updates", 1, "true C4"),
        ),
    )

    runtime.record_disposition(newest_disposition)
    plan = runtime._latest_tick_work[1]

    assert plan.supervision_sources == (first, newest)
    assert plan.supervision_work[0].modality_eligibility == (
        first_disposition.modality_eligibility
    )
    assert plan.supervision_work[1].modality_eligibility == (
        newest_disposition.modality_eligibility
    )
    assert tuple(runtime._pending_supervision_modalities) == (first, newest)

    runtime._retire_completed_candidates(
        set((*plan.supervision_sources, *plan.replay_sources)),
    )

    assert not runtime._pending_supervision_modalities


def test_depth_supervision_eligibility_gates_the_optimizer_request():
    source = _source(3)
    disposition = ObservationDisposition(
        source=source,
        snapshot_id=17,
        kind=DispositionKind.ADMIT,
        observe=True,
        mapping_candidate=True,
        admit_supervision=True,
        allow_geometry_insertion=False,
        residency=ResidencyIntent.HOT,
        reason="rgb only",
        modality_eligibility=ModalityEligibility(True, "qualified", False, "rejected", False),
    )
    plan = replace(
        _plan(source),
        supervision_work=(
            SupervisionWork(source, "native", disposition.modality_eligibility),
        ),
    )
    assert not depth_supervision_enabled(plan.supervision_work[0], depth_loss_configured=True)


def test_b1_schedule_executes_current_then_identity_bound_replay():
    current, replay = _source(3), _source(2)
    plan = _plan(current)
    plan = MappingWorkPlan(
        **{
            field: getattr(plan, field)
            for field in plan.__dataclass_fields__
        }
        | {
            "optimizer_updates": 2,
            "replay_sources": (replay,),
            "replay_work": (SupervisionWork(replay, "native", _admitted_modalities()),),
        }
    )

    assert tuple(item.source for item in scheduled_b1_work(plan)) == (current, replay)


def test_transaction_work_preserves_legacy_b1_updates_without_adaptive_padding():
    from eligsir.runtime.production_application import _scheduled_transaction_work

    current = _source(3)
    legacy = replace(
        _plan(current, insertion_seeds=0),
        optimizer_updates=2,
    )
    adaptive = replace(
        legacy,
        supervision_constellation=SupervisionConstellation(
            requested_batch_size=2,
            requested_camera_count=1,
            realized_batch_size=1,
            realized_camera_count=1,
            unique_eligible_count=1,
            fill_ratio=0.5,
            microbatch_count=1,
            mode="sequential_accumulation",
            reason="risk_shrunk",
            hysteresis_state="risk_shrunk",
        ),
    )

    assert tuple(
        work.source
        for work in _scheduled_transaction_work(legacy, configured_batch_size=1)
    ) == (current, current)
    assert tuple(
        work.source
        for work in _scheduled_transaction_work(adaptive, configured_batch_size=2)
    ) == (current,)


def test_source_idle_plan_uses_only_admitted_replay_and_refreshes_fences():
    current, historical = _source(3), _source(2)
    active = replace(
        _plan(current),
        optimizer_updates=2,
        replay_sources=(historical,),
        replay_work=(
            SupervisionWork(historical, "native", _admitted_modalities()),
        ),
    )
    runtime = object.__new__(ContinualMappingRuntime)
    runtime._optimization_policy = _DISABLED_REPLAY_CHALLENGER_POLICY
    runtime._batch_size = 1
    runtime._versions = lambda: (9, 10, 11)
    runtime._replay_modalities = {historical: _admitted_modalities()}
    runtime._shortlisted_replay_view = lambda **_kwargs: (
        {
            historical: _admitted_modalities(),
            current: _admitted_modalities(),
        },
        current,
    )

    idle = runtime._idle_replay_plan(
        _snapshot(current),
        active,
        ready_replay_sources=frozenset((historical,)),
    )

    assert idle.supervision_sources == ()
    assert idle.supervision_work == ()
    assert len(idle.replay_sources) == 1
    assert idle.replay_sources == (historical,)
    assert tuple(work.source for work in idle.replay_work) == idle.replay_sources
    assert idle.optimizer_updates == 2
    assert not idle.insertion.enabled
    assert not idle.allow_growth
    assert not idle.allow_relocation
    assert idle.mutation_budget.max_total == 0
    assert (idle.map_content_version, idle.topology_version, idle.parameter_version) == (
        9,
        10,
        11,
    )
    assert "source_idle_replay_only" in idle.reason_trace


def test_ready_idle_plan_uses_full_ready_replay_reservoir_not_stale_shortlist():
    current = _source(9)
    stale = _source(8)
    ready = tuple(_source(index) for index in range(4))
    captured = {}

    class Owner:
        def forces_native_post_refinement(self, _fraction):
            return False

        def resolve_idle_replay(self, snapshot, active_plan, **kwargs):
            captured["snapshot"] = snapshot
            captured["modalities"] = kwargs["replay_modalities"]
            return active_plan

    runtime = object.__new__(ContinualMappingRuntime)
    runtime._optimization_policy = _DISABLED_REPLAY_CHALLENGER_POLICY
    runtime._batch_size = 4
    runtime._versions = lambda: (3, 2, 3)
    runtime._shortlisted_replay_view = lambda **_kwargs: (
        {stale: _admitted_modalities()},
        stale,
    )
    runtime._replay_modalities = {
        source: _admitted_modalities() for source in ready
    }
    runtime._replay_weights = {source: 1.0 for source in ready}
    runtime._work_budget_owner = Owner()
    runtime._replay_depth_due = lambda: False

    runtime._idle_replay_plan(
        _snapshot(current),
        _plan(current),
        ready_replay_sources=frozenset(ready),
    )

    assert tuple(captured["modalities"]) == ready
    assert captured["snapshot"].replay_candidates == ready
    assert captured["snapshot"].reserved_global_replay_candidate is None


def test_post_refinement_focus_forces_a_rotating_global_replay_slot():
    """Focused B4 refinement keeps one non-focused history source present."""

    current = _source(9)
    global_source, focused = _source(1), _source(2)
    captured = {}

    class Owner:
        def forces_native_post_refinement(self, _fraction):
            return False

        def resolve_idle_replay(self, snapshot, active_plan, **kwargs):
            captured["snapshot"] = snapshot
            captured["kwargs"] = kwargs
            return active_plan

    runtime = object.__new__(ContinualMappingRuntime)
    runtime._optimization_policy = _DISABLED_REPLAY_CHALLENGER_POLICY
    runtime._batch_size = 4
    runtime.optimizer_steps = 41
    runtime._versions = lambda: (3, 2, 3)
    runtime._post_refinement_phase_owner = SimpleNamespace(
        phase=PostRefinementPhase.REFINEMENT
    )
    runtime._post_refinement_replay_catalog = lambda: {
        global_source: _admitted_modalities(),
        focused: _admitted_modalities(),
    }
    runtime._shortlisted_replay_view = lambda **_kwargs: ({}, global_source)
    runtime._replay_weights = {global_source: 1.0, focused: 10.0}
    runtime._work_budget_owner = Owner()
    runtime._replay_depth_due = lambda: False

    runtime._idle_replay_plan(_snapshot(current), _plan(current))

    assert captured["snapshot"].reserved_global_replay_candidate is global_source
    assert captured["kwargs"]["force_reserved_global"] is True


def test_post_refinement_idle_plan_does_not_authorize_unproven_mcmc_mutation():
    """The quality-critical replay tail refines a completed topology by default."""

    current = _source(9)
    captured = {}

    class Owner:
        def forces_native_post_refinement(self, _fraction):
            return False

        def resolve_idle_replay(self, _snapshot, active_plan, **kwargs):
            captured["proposals"] = kwargs["proposals"]
            return active_plan

    runtime = object.__new__(ContinualMappingRuntime)
    runtime._optimization_policy = _DISABLED_REPLAY_CHALLENGER_POLICY
    runtime._batch_size = 1
    runtime.optimizer_steps = 41
    runtime._versions = lambda: (3, 2, 3)
    runtime._post_refinement_phase_owner = SimpleNamespace(
        phase=PostRefinementPhase.REFINEMENT
    )
    runtime._post_refinement_replay_catalog = lambda: {current: _admitted_modalities()}
    runtime._replay_weights = {current: 1.0}
    runtime._work_budget_owner = Owner()
    runtime._mutation_policy_provider = MutationProposalProvider()
    runtime._replay_depth_due = lambda: False

    runtime._idle_replay_plan(_snapshot(current), _plan(current))

    assert not any(
        proposal.provider == "MutationProposalProvider"
        for proposal in captured["proposals"]
    )


def test_source_idle_plan_reuses_latest_admitted_view_before_history_exists():
    current = _source(3)
    active = _plan(current)
    runtime = object.__new__(ContinualMappingRuntime)
    runtime._optimization_policy = _DISABLED_REPLAY_CHALLENGER_POLICY
    runtime._batch_size = 1
    runtime._versions = lambda: (3, 2, 3)

    idle = runtime._idle_replay_plan(_snapshot(current), active)

    assert idle.supervision_sources == ()
    assert idle.replay_sources == (current,)
    assert idle.replay_work == active.supervision_work


def test_source_idle_c4_fills_every_lane_from_admitted_views():
    current = _source(4)
    historical = (_source(1), _source(2), _source(3))
    replay_work = tuple(
        SupervisionWork(source, "native", _admitted_modalities())
        for source in historical
    )
    active = replace(
        _plan(current),
        optimizer_updates=1,
        replay_sources=historical,
        replay_work=replay_work,
    )
    runtime = object.__new__(ContinualMappingRuntime)
    runtime._optimization_policy = _DISABLED_REPLAY_CHALLENGER_POLICY
    runtime._batch_size = 4
    runtime._versions = lambda: (3, 2, 3)
    runtime.optimizer_steps = 10
    runtime._work_budget_owner = WorkBudgetOwner()
    runtime._replay_modalities = __import__("collections").OrderedDict(
        (work.source, work.modality_eligibility)
        for work in (*replay_work, *active.supervision_work)
    )
    runtime._replay_weights = {
        work.source: 1.0 for work in (*replay_work, *active.supervision_work)
    }

    idle = runtime._idle_replay_plan(_snapshot(current), active)

    assert idle.supervision_sources == ()
    assert set(idle.replay_sources) == {*historical, current}
    assert len(idle.replay_work) == 4
    assert idle.optimizer_updates == 1


def test_source_idle_resolution_is_actuated_from_one_policy_proposal():
    current = _source(4)
    historical = (_source(1), _source(2), _source(3))
    replay_work = tuple(
        SupervisionWork(source, "eighth", _admitted_modalities())
        for source in historical
    )
    active = replace(
        _plan(current),
        resolution_tier="eighth",
        supervision_work=(
            SupervisionWork(current, "eighth", _admitted_modalities()),
        ),
        replay_sources=historical,
        replay_work=replay_work,
    )

    class IdleResolutionProvider:
        def propose_idle(self, snapshot, **kwargs):
            assert snapshot.source == current
            assert kwargs["batch_size"] == 4
            assert kwargs["mapping_cost_ms"] == 5.0
            return ControlProposal(
                "IdleResolutionProvider",
                "replay_resolution_tier",
                "quarter",
                "idle GPU budget",
            )

    runtime = object.__new__(ContinualMappingRuntime)
    runtime._optimization_policy = _DISABLED_REPLAY_CHALLENGER_POLICY
    runtime._batch_size = 4
    runtime._versions = lambda: (3, 2, 3)
    runtime.optimizer_steps = 10
    runtime._work_budget_owner = WorkBudgetOwner()
    runtime._resolution_policy_provider = IdleResolutionProvider()
    runtime._latest_mapping_step_wall_ms = 5.0
    runtime._replay_modalities = __import__("collections").OrderedDict(
        (work.source, work.modality_eligibility)
        for work in (*replay_work, *active.supervision_work)
    )
    runtime._replay_weights = {
        work.source: 1.0 for work in (*replay_work, *active.supervision_work)
    }

    idle = runtime._idle_replay_plan(_snapshot(current), active)

    assert idle.resolution_tier.value == "quarter"
    assert {work.resolution_tier.value for work in idle.replay_work} == {"quarter"}
    assert any("IdleResolutionProvider" in reason for reason in idle.reason_trace)


def test_source_completion_alone_retains_live_idle_resolution_authority():
    """Bug: EOF must not grant the durable post-refinement phase."""

    current = _source(4)
    active = _plan(current)

    class LiveResolutionProvider:
        def propose_idle(self, _snapshot, **_kwargs):
            return ControlProposal(
                "live", "replay_resolution_tier", "quarter", "live ingress policy"
            )

    runtime = object.__new__(ContinualMappingRuntime)
    runtime._optimization_policy = _DISABLED_REPLAY_CHALLENGER_POLICY
    runtime._batch_size = 1
    runtime._versions = lambda: (3, 2, 3)
    runtime.optimizer_steps = 10
    runtime._work_budget_owner = WorkBudgetOwner()
    runtime._resolution_policy_provider = LiveResolutionProvider()
    runtime._latest_mapping_step_wall_ms = 500.0
    runtime._source_complete = False
    runtime._post_refinement_phase_owner = PostRefinementPhaseOwner()

    runtime.mark_source_complete()
    idle = runtime._idle_replay_plan(_snapshot(current), active)

    assert runtime._source_complete is True
    assert runtime._post_refinement_phase_owner.phase is PostRefinementPhase.SOURCE_COMPLETE
    assert idle.resolution_tier.value == "quarter"
    assert {work.resolution_tier.value for work in idle.replay_work} == {"quarter"}
    assert "post_refinement" not in idle.reason_trace


def test_source_idle_growth_requires_adaptive_quality_owner_approval():
    current = _source(4)
    active = _plan(current)
    provider = AdaptiveGrowthProposalProvider(
        enabled=True,
        marginal_utility_brake=-1.0e-5,
        marginal_utility_release=1.0e-5,
        hysteresis=2,
        minimum_population_delta=200,
        utility_ema=0.5,
        regression_tolerance_db=0.25,
    )
    for iteration, psnr, population, drop in (
        (100, 24.0, 10_000, 0.0),
        (200, 23.8, 10_500, 0.8),
        (300, 23.7, 11_000, 0.9),
    ):
        provider.observe_training_spatial_quality(
            iteration=iteration,
            aggregate_psnr_db=psnr,
            gaussian_count=population,
            worst_drop_from_peak_db=drop,
        )
    runtime = object.__new__(ContinualMappingRuntime)
    runtime._optimization_policy = _DISABLED_REPLAY_CHALLENGER_POLICY
    runtime._batch_size = 1
    runtime._versions = lambda: (3, 2, 3)
    runtime.optimizer_steps = 10
    runtime._work_budget_owner = WorkBudgetOwner()
    runtime._mutation_policy_provider = MutationProposalProvider()
    runtime._adaptive_growth_provider = provider
    snapshot = _snapshot(current)
    snapshot = replace(
        snapshot, limits=replace(snapshot.limits, max_mutations=1)
    )

    braked = runtime._idle_replay_plan(snapshot, active)

    assert braked.allow_growth is False
    assert braked.allow_relocation is True
    assert braked.mutation_budget.max_total == 1
    assert any("AdaptiveGrowthProposalProvider" in reason for reason in braked.reason_trace)

    provider.observe_training_spatial_quality(
        iteration=400, aggregate_psnr_db=23.9, gaussian_count=11_000,
        worst_drop_from_peak_db=0.2,
    )
    provider.observe_training_spatial_quality(
        iteration=500, aggregate_psnr_db=24.0, gaussian_count=11_000,
        worst_drop_from_peak_db=0.0,
    )

    released = runtime._idle_replay_plan(snapshot, active)
    assert released.allow_growth is True
    assert any("released" in reason for reason in released.reason_trace)


def test_source_idle_c4_rotates_weighted_random_samples_across_full_cache():
    current = _source(40)
    active = _plan(current)
    runtime = object.__new__(ContinualMappingRuntime)
    runtime._optimization_policy = _DISABLED_REPLAY_CHALLENGER_POLICY
    runtime._batch_size = 4
    runtime._versions = lambda: (3, 2, 3)
    runtime._work_budget_owner = WorkBudgetOwner()
    modalities = _admitted_modalities()
    runtime._replay_modalities = __import__("collections").OrderedDict(
        (_source(index), modalities) for index in range(1, 41)
    )
    runtime._replay_weights = {_source(index): 1.0 for index in range(1, 41)}

    runtime.optimizer_steps = 100
    first = runtime._idle_replay_plan(_snapshot(current), active)
    runtime.optimizer_steps = 101
    second = runtime._idle_replay_plan(_snapshot(current), active)

    assert len(first.replay_sources) == len(second.replay_sources) == 4
    assert set(first.replay_sources).issubset(set(runtime._replay_modalities))
    assert set(second.replay_sources).issubset(set(runtime._replay_modalities))
    assert first.replay_sources != second.replay_sources
    assert first.replay_population_size == second.replay_population_size == 40


def test_runtime_accepts_focus_weights_only_for_full_cache_identities():
    runtime = object.__new__(ContinualMappingRuntime)
    admitted = (_source(1), _source(2), _source(3))
    runtime._replay_modalities = __import__("collections").OrderedDict(
        (source, _admitted_modalities()) for source in admitted
    )
    runtime._replay_weights = {source: 1.0 for source in admitted}

    runtime.publish_replay_focus_weights({admitted[1]: 7.5, admitted[2]: 0.0})

    assert runtime._replay_weights == {
        admitted[0]: 1.0,
        admitted[1]: 7.5,
        admitted[2]: 0.0,
    }
    with pytest.raises(ValueError, match="outside the cache"):
        runtime.publish_replay_focus_weights({_source(99): 2.0})


def test_runtime_rejects_unbounded_focus_publication_before_processing_items():
    runtime = object.__new__(ContinualMappingRuntime)
    runtime._replay_modalities = __import__("collections").OrderedDict()
    runtime._replay_weights = {}

    class OversizedFocus(dict):
        def items(self):
            raise AssertionError("oversized publication must fail before item processing")

    oversized = OversizedFocus(
        (_source(index), 2.0) for index in range(33)
    )

    with pytest.raises(ValueError, match="at most 32"):
        runtime.publish_replay_focus_weights(oversized)


def test_runtime_focus_publication_promotes_old_cold_identity_into_next_shortlist():
    """Publishing a weak old source must affect shortlist membership, not only weight."""

    from collections import OrderedDict
    from eligsir.mapping.scheduler.historical_candidate_index import (
        HistoricalCandidateIndex,
        HistoricalFrameMetadata,
    )

    runtime = object.__new__(ContinualMappingRuntime)
    runtime.initialized = True
    runtime.optimizer_steps = 9
    runtime._work_budget_owner = WorkBudgetOwner()
    runtime._historical_candidate_index = HistoricalCandidateIndex(seed=17)
    runtime._historical_sources_by_frame_id = {}
    runtime._historical_frame_ids_by_source = {}
    runtime._replay_modalities = OrderedDict()
    runtime._replay_weights = {}
    for frame_id in range(700):
        source = _source(frame_id)
        runtime._historical_candidate_index.register(
            HistoricalFrameMetadata(frame_id=frame_id, region_id=frame_id % 7)
        )
        runtime._historical_sources_by_frame_id[frame_id] = source
        runtime._historical_frame_ids_by_source[source] = frame_id
        runtime._replay_modalities[source] = _admitted_modalities()
        runtime._replay_weights[source] = 1.0

    baseline = runtime._shortlisted_replay_modalities(
        snapshot_id=31, control_tick_epoch=5
    )
    old_source = next(
        _source(frame_id)
        for frame_id in range(1, 500)
        if _source(frame_id) not in baseline
    )

    requested_focus = {
        _source(frame_id): float(frame_id - 498)
        for frame_id in range(508, 539)
    }
    requested_focus[old_source] = 100.0
    runtime.publish_replay_focus_weights(requested_focus)
    focused = runtime._shortlisted_replay_modalities(
        snapshot_id=32, control_tick_epoch=5
    )

    assert old_source in focused
    assert len(focused) <= 128
    retained_focus_ids = {
        frame_id for frame_id, _weight in runtime._replay_focus_frame_weights
    }
    assert len(retained_focus_ids) == 32
    assert retained_focus_ids == {old_source.source_ordinal, *range(508, 539)}


def test_shortlisted_replay_excludes_prepared_commit_awaiting_first_supervision():
    """A prepared COMMIT stays FRESH until its first successful supervision."""

    from collections import OrderedDict
    from eligsir.mapping.scheduler.historical_candidate_index import (
        HistoricalCandidateIndex,
        HistoricalFrameMetadata,
    )

    fresh, replay = _source(1), _source(2)
    runtime = object.__new__(ContinualMappingRuntime)
    runtime.initialized = True
    runtime.optimizer_steps = 0
    runtime._work_budget_owner = WorkBudgetOwner()
    runtime._historical_candidate_index = HistoricalCandidateIndex(seed=17)
    runtime._historical_sources_by_frame_id = {1: fresh, 2: replay}
    runtime._replay_modalities = OrderedDict(
        ((fresh, _admitted_modalities()), (replay, _admitted_modalities()))
    )
    runtime._observation_db = SimpleNamespace(
        prepared_commit_obligation_for_source=lambda source: (
            object() if source is fresh else None
        )
    )
    for frame_id in (1, 2):
        runtime._historical_candidate_index.register(
            HistoricalFrameMetadata(frame_id=frame_id, region_id=frame_id)
        )

    shortlisted = runtime._shortlisted_replay_modalities(
        snapshot_id=31, control_tick_epoch=5
    )

    assert tuple(shortlisted) == (replay,)


def test_b1_idle_replay_reserves_global_history_on_bounded_cadence():
    """High focus weights must not starve the reserved global identity at B1."""

    current = _source(100)
    focused = _source(99)
    owner = WorkBudgetOwner(global_history_supervision_cadence=8)
    active = _plan(current)
    modalities = {
        _source(index): _admitted_modalities() for index in range(1, 101)
    }
    weights = {source: 0.0 for source in modalities}
    weights[focused] = 1_000.0

    realized = []
    # The first realizable replay plan in each cadence bucket owns the slot;
    # correctness cannot depend on landing on the exact modulo boundary.
    for optimizer_step, reserved_ordinal in ((1, 1), (9, 2), (17, 3)):
        reserved = _source(reserved_ordinal)
        snapshot = replace(
            _snapshot(current),
            snapshot_id=100 + optimizer_step,
            optimizer_steps_completed=optimizer_step,
            replay_candidates=tuple(modalities),
            replay_work_modalities=tuple(modalities.values()),
            replay_candidate_weights=tuple(weights.values()),
            reserved_global_replay_candidate=reserved,
        )
        plan = owner.resolve_idle_replay(
            snapshot,
            active,
            replay_modalities=modalities,
            replay_weights=weights,
            batch_size=1,
            optimizer_steps_completed=optimizer_step,
            versions=(3, 2, 3),
        )
        assert plan.replay_sources == (reserved,)
        assert "reserved_global_history_replay" in plan.reason_trace
        realized.extend(plan.replay_sources)

    assert realized == [_source(1), _source(2), _source(3)]


def test_post_refinement_focus_keeps_one_global_source_in_each_b4():
    """Focused refinement must retain a broad replay constraint per B4."""

    current = _source(100)
    candidates = tuple(_source(index) for index in range(1, 6))
    reserved = candidates[0]
    modalities = {source: _admitted_modalities() for source in candidates}
    weights = {
        source: (10_000.0 if source is candidates[-1] else 1.0)
        for source in candidates
    }
    snapshot = replace(
        _snapshot(current),
        replay_candidates=candidates,
        replay_work_modalities=tuple(modalities.values()),
        replay_candidate_weights=tuple(weights.values()),
        reserved_global_replay_candidate=reserved,
    )
    owner = WorkBudgetOwner(
        requested_batch_size=4,
        maximum_camera_count=4,
        constellation_policy="fixed",
    )

    plan = owner.resolve_idle_replay(
        snapshot,
        _plan(current),
        replay_modalities=modalities,
        replay_weights=weights,
        batch_size=4,
        optimizer_steps_completed=41,
        versions=(3, 2, 3),
        post_refinement=True,
        force_reserved_global=True,
    )

    assert len(plan.replay_sources) == 4
    assert reserved in plan.replay_sources


def test_b1_admitted_frame_reserves_replay_only_transaction_and_retains_current():
    """An active B1 slot must realize global replay without losing admission."""

    from collections import OrderedDict, deque
    import eligsir.runtime.production_application as production_application
    from eligsir.runtime.streaming_observation import StreamingObservation

    production_application.StreamingObservation = StreamingObservation

    current = _source(700)
    history = tuple(_source(index) for index in range(600))
    reserved = history[17]
    modalities = _admitted_modalities()
    snapshot = replace(
        _snapshot(current),
        snapshot_id=80,
        optimizer_steps_completed=8,
        replay_candidates=history,
        replay_work_modalities=(modalities,) * len(history),
        replay_candidate_weights=(1.0,) * len(history),
        reserved_global_replay_candidate=reserved,
        limits=ControlLimits(1, 1, 1, 7, 0, 1024),
    )
    runtime = object.__new__(ContinualMappingRuntime)
    runtime._dispositions_since_tick = deque(maxlen=4096)
    runtime._latest_control_snapshot = snapshot
    runtime._latest_control_outputs = (snapshot, _proposals())
    runtime._latest_tick_work = None
    runtime._latest_report_work = None
    runtime._active_materialization_work = None
    runtime._pending_supervision_modalities = OrderedDict()
    runtime._pending_supervision_capacity = 1
    runtime._replay_modalities = OrderedDict(
        (source, modalities) for source in history
    )
    runtime._replay_weights = {source: 1.0 for source in history}
    runtime._work_budget_owner = WorkBudgetOwner(
        requested_batch_size=1,
        global_history_supervision_cadence=8,
    )
    runtime._closed = False
    runtime.initialized = False
    runtime.frames_seen = 0
    runtime.bootstrap_frames = 2
    runtime._bootstrap = []
    disposition = ObservationDisposition(
        source=current,
        snapshot_id=snapshot.snapshot_id,
        kind=DispositionKind.ADMIT,
        observe=True,
        mapping_candidate=True,
        admit_supervision=True,
        allow_geometry_insertion=True,
        residency=ResidencyIntent.HOT,
        reason="admitted",
        modality_eligibility=modalities,
    )

    runtime.record_disposition(disposition)
    plan = runtime._latest_tick_work[1]
    frame = MappingFrame(
        source=current,
        rgb=BufferRef.from_numpy(
            np.zeros((2, 2, 3), dtype=np.uint8), encoding="rgb8"
        ),
        depth=BufferRef.from_numpy(
            np.ones((2, 2), dtype=np.float32), encoding="32FC1", units="m"
        ),
        intrinsics=CameraIntrinsics(
            fx=2.0,
            fy=2.0,
            cx=1.0,
            cy=1.0,
            width=2,
            height=2,
            intrinsics_source="test",
        ),
        device="cpu",
        pose=Pose(np.eye(4, dtype=np.float32)),
    )
    runtime.map_frame(frame, None)

    assert plan.materialization_sources == (current,)
    assert plan.supervision_sources == ()
    assert plan.replay_sources == (reserved,)
    assert plan.optimizer_updates == 1
    assert "reserved_global_history_replay" in plan.reason_trace
    assert "admitted_source_retained_for_later_replay" in plan.reason_trace
    assert current in runtime._replay_modalities
    assert current in runtime._pending_supervision_modalities


def test_active_b1_realizes_one_global_replay_transaction_per_update_bucket():
    owner = WorkBudgetOwner(
        requested_batch_size=1,
        global_history_supervision_cadence=8,
    )
    modalities = _admitted_modalities()
    history = tuple(_source(index) for index in range(32))
    replay_steps = []

    for optimizer_step in range(16):
        current = _source(1_000 + optimizer_step)
        reserved = history[optimizer_step // 8]
        snapshot = replace(
            _snapshot(current),
            snapshot_id=200 + optimizer_step,
            optimizer_steps_completed=optimizer_step,
            replay_candidates=history,
            replay_work_modalities=(modalities,) * len(history),
            replay_candidate_weights=(1.0,) * len(history),
            reserved_global_replay_candidate=reserved,
            limits=ControlLimits(1, 1, 1, 7, 0, 1024),
        )
        disposition = ObservationDisposition(
            source=current,
            snapshot_id=snapshot.snapshot_id,
            kind=DispositionKind.ADMIT,
            observe=True,
            mapping_candidate=True,
            admit_supervision=True,
            allow_geometry_insertion=True,
            residency=ResidencyIntent.HOT,
            reason="admitted",
            modality_eligibility=modalities,
        )

        plan = owner.resolve(
            snapshot,
            disposition=disposition,
            proposals=_proposals(),
        )

        assert plan.materialization_sources == (current,)
        if plan.replay_sources:
            replay_steps.append(optimizer_step)
            assert plan.replay_sources == (reserved,)
            assert plan.supervision_sources == ()
        else:
            assert plan.supervision_sources == (current,)

    assert replay_steps == [0, 8]


def test_tick_result_publishes_post_disposition_catalog_population(monkeypatch):
    """Decision traces must receive catalog evidence from the completed tick."""

    from collections import OrderedDict, deque
    import eligsir.runtime.production_application as production_application
    from eligsir.runtime.telemetry import MapStateTelemetry

    monkeypatch.setattr(
        production_application,
        "MapStateTelemetry",
        MapStateTelemetry,
        raising=False,
    )

    runtime = object.__new__(ContinualMappingRuntime)
    runtime._poll_gpu_telemetry = lambda: None
    runtime._dispositions_since_tick = deque((object(), object()))
    runtime._replay_modalities = OrderedDict(
        (_source(index), _admitted_modalities()) for index in range(3)
    )
    runtime._versions = lambda: (0, 0, 0)
    runtime._work_clock_owner = SimpleNamespace(snapshot=lambda: object())
    runtime.iteration = 0
    runtime.optimizer_steps = 0
    runtime._supervision_views_completed = 0
    runtime._transfer_owner = None
    runtime._telemetry_inputs = lambda _snapshot: object()
    runtime._latest_gpu_transaction = None
    runtime._telemetry = SimpleNamespace(
        publish_tick=lambda *_args, **_kwargs: SimpleNamespace(
            map_state=MapStateTelemetry(3, 0, 0, 0)
        )
    )

    result = runtime._tick_result(
        False,
        "test",
        snapshot=None,
        plan=None,
    )

    assert result.retained_catalog_population_post_dispositions == 3
    assert len(result.dispositions) == 2
    assert not runtime._dispositions_since_tick


def test_tick_result_carries_real_insertion_filter_receipt_to_decision_trace(monkeypatch):
    from collections import deque
    import eligsir.runtime.production_application as production_application
    from eligsir.runtime.causal_replay import decision_trace_row
    from eligsir.runtime.telemetry import MapStateTelemetry

    monkeypatch.setattr(
        production_application,
        "MapStateTelemetry",
        MapStateTelemetry,
        raising=False,
    )
    source = _source(9)
    runtime = object.__new__(ContinualMappingRuntime)
    runtime._poll_gpu_telemetry = lambda: None
    runtime._dispositions_since_tick = deque()
    runtime._replay_modalities = {}
    runtime._versions = lambda: (0, 0, 0)
    runtime._work_clock_owner = SimpleNamespace(snapshot=lambda: object())
    runtime.iteration = runtime.optimizer_steps = runtime._supervision_views_completed = 0
    runtime._transfer_owner = None
    runtime._telemetry_inputs = lambda _snapshot: object()
    runtime._latest_gpu_transaction = None
    runtime._population_selection_trace_enabled = True
    runtime._decision_trace_insertion_filter = {
        "source": source.fingerprint(),
        "stats": {"raw_candidates": 0, "after_grazing_filter": 0},
    }
    runtime._telemetry = SimpleNamespace(
        publish_tick=lambda *_args, **_kwargs: SimpleNamespace(
            map_state=MapStateTelemetry(3, 0, 0, 0)
        )
    )

    result = runtime._tick_result(False, "test", snapshot=None, plan=None)

    assert decision_trace_row(result)["insertion_filter"] == {
        "source": source.fingerprint(),
        "stats": {"raw_candidates": 0, "after_grazing_filter": 0},
    }


def test_tick_result_does_not_attach_future_gpu_packet_to_frozen_snapshot(monkeypatch):
    """An asynchronous packet belongs to a later control snapshot, not this tick."""

    from collections import deque
    import eligsir.runtime.production_application as production_application
    from eligsir.runtime.telemetry import (
        GpuTransactionTelemetry,
        MapStateTelemetry,
        TransactionType,
    )

    monkeypatch.setattr(
        production_application,
        "MapStateTelemetry",
        MapStateTelemetry,
        raising=False,
    )

    runtime = object.__new__(ContinualMappingRuntime)
    runtime._poll_gpu_telemetry = lambda: None
    runtime._dispositions_since_tick = deque()
    runtime._replay_modalities = {}
    runtime._versions = lambda: (0, 0, 0)
    runtime._work_clock_owner = SimpleNamespace(snapshot=lambda: object())
    runtime.iteration = 0
    runtime.optimizer_steps = 0
    runtime._supervision_views_completed = 0
    runtime._transfer_owner = None
    runtime._telemetry_inputs = lambda _snapshot: object()
    runtime._latest_gpu_transaction = GpuTransactionTelemetry(
        transaction_id=2,
        snapshot_id=2,
        transaction_type=TransactionType.FRAME_UPLOAD,
        map_content_version=0,
        topology_version=0,
        parameter_version=0,
        batch_size=0,
        pixels=0,
        rasterizer_calls=0,
        optimizer_updates=0,
        candidates=0,
        duration_ms=0.0,
        mutation_type="none",
        inserted=0,
        pruned=0,
        relocated=0,
        corrected=0,
    )
    captured = {}
    runtime._telemetry = SimpleNamespace(
        publish_tick=lambda *_args, **kwargs: (
            captured.update(kwargs)
            or SimpleNamespace(map_state=MapStateTelemetry(0, 0, 0, 0))
        )
    )

    runtime._tick_result(
        False,
        "test",
        snapshot=SimpleNamespace(snapshot_id=1),
        plan=None,
    )

    assert captured["gpu_transaction"] is None


def test_tick_result_does_not_join_pre_rollback_gpu_packet_to_live_map(monkeypatch):
    """A rejected branch attempt remains historical after live map rollback."""

    from collections import deque
    import eligsir.runtime.production_application as production_application
    from eligsir.runtime.telemetry import (
        GpuTransactionTelemetry,
        MapStateTelemetry,
        TransactionType,
    )

    monkeypatch.setattr(
        production_application,
        "MapStateTelemetry",
        MapStateTelemetry,
        raising=False,
    )

    runtime = object.__new__(ContinualMappingRuntime)
    runtime._poll_gpu_telemetry = lambda: None
    runtime._dispositions_since_tick = deque()
    runtime._replay_modalities = {}
    # Live rollback restores accepted values, then mints a generation beyond
    # the rejected GPU attempt.  The historical timing cannot describe this
    # new map state even though its control snapshot is still current.
    runtime._versions = lambda: (9, 4, 11)
    runtime._work_clock_owner = SimpleNamespace(snapshot=lambda: object())
    runtime.iteration = 0
    runtime.optimizer_steps = 0
    runtime._supervision_views_completed = 0
    runtime._transfer_owner = None
    runtime._telemetry_inputs = lambda _snapshot: object()
    runtime._latest_gpu_transaction = GpuTransactionTelemetry(
        transaction_id=3,
        snapshot_id=1,
        transaction_type=TransactionType.OPTIMIZATION,
        map_content_version=8,
        topology_version=3,
        parameter_version=10,
        batch_size=4,
        pixels=64,
        rasterizer_calls=1,
        optimizer_updates=1,
        candidates=0,
        duration_ms=1.0,
        mutation_type="none",
        inserted=0,
        pruned=0,
        relocated=0,
        corrected=0,
    )
    captured = {}
    runtime._telemetry = SimpleNamespace(
        publish_tick=lambda *_args, **kwargs: (
            captured.update(kwargs)
            or SimpleNamespace(map_state=MapStateTelemetry(3, 9, 4, 11))
        )
    )

    runtime._tick_result(
        False,
        "test",
        snapshot=SimpleNamespace(snapshot_id=1),
        plan=None,
    )

    assert captured["gpu_transaction"] is None


def test_mismatched_disposition_is_rejected_before_publication():
    source = _source(3)
    runtime = object.__new__(ContinualMappingRuntime)
    runtime._dispositions_since_tick = __import__("collections").deque(maxlen=4096)
    runtime._latest_tick_work = None
    runtime._latest_report_work = None
    runtime._active_materialization_work = None
    _publish_snapshot_outputs(runtime, _snapshot(source))
    runtime._work_budget_owner = WorkBudgetOwner()
    disposition = ObservationDisposition(
        source=source,
        snapshot_id=18,
        kind=DispositionKind.ADMIT,
        observe=True,
        mapping_candidate=True,
        admit_supervision=True,
        allow_geometry_insertion=True,
        residency=ResidencyIntent.HOT,
        reason="foreign snapshot",
        modality_eligibility=_admitted_modalities(),
    )

    with pytest.raises(RuntimeError, match="no matching control snapshot"):
        runtime.record_disposition(disposition)

    assert tuple(runtime._dispositions_since_tick) == ()
    assert runtime._latest_tick_work is None
    assert runtime._latest_report_work is None
    assert runtime._active_materialization_work is None


def test_foreign_source_disposition_is_rejected_before_publication():
    source = _source(3)
    runtime = object.__new__(ContinualMappingRuntime)
    runtime._dispositions_since_tick = __import__("collections").deque(maxlen=4096)
    runtime._latest_tick_work = None
    runtime._latest_report_work = None
    runtime._active_materialization_work = None
    _publish_snapshot_outputs(runtime, _snapshot(source))
    runtime._work_budget_owner = WorkBudgetOwner()
    disposition = ObservationDisposition(
        source=_source(4),
        snapshot_id=17,
        kind=DispositionKind.ADMIT,
        observe=True,
        mapping_candidate=True,
        admit_supervision=True,
        allow_geometry_insertion=True,
        residency=ResidencyIntent.HOT,
        reason="foreign source",
        modality_eligibility=_admitted_modalities(),
    )

    with pytest.raises(ValueError, match="source must match"):
        runtime.record_disposition(disposition)

    assert tuple(runtime._dispositions_since_tick) == ()
    assert runtime._latest_tick_work is None
    assert runtime._latest_report_work is None
    assert runtime._active_materialization_work is None


def test_disposition_is_not_published_when_work_plan_resolution_fails():
    source = _source(3)
    runtime = object.__new__(ContinualMappingRuntime)
    runtime._dispositions_since_tick = __import__("collections").deque(maxlen=4096)
    runtime._latest_tick_work = None
    runtime._latest_report_work = None
    runtime._active_materialization_work = None
    _publish_snapshot_outputs(runtime, _snapshot(source))

    class _RejectingWorkBudgetOwner:
        def resolve(self, _snapshot, **_outputs):
            raise RuntimeError("no valid work plan")

    runtime._work_budget_owner = _RejectingWorkBudgetOwner()
    disposition = ObservationDisposition(
        source=source,
        snapshot_id=17,
        kind=DispositionKind.ADMIT,
        observe=True,
        mapping_candidate=True,
        admit_supervision=True,
        allow_geometry_insertion=True,
        residency=ResidencyIntent.HOT,
        reason="admitted",
        modality_eligibility=_admitted_modalities(),
    )

    with pytest.raises(RuntimeError, match="no valid work plan"):
        runtime.record_disposition(disposition)

    assert tuple(runtime._dispositions_since_tick) == ()
    assert runtime._latest_tick_work is None
    assert runtime._latest_report_work is None
    assert runtime._active_materialization_work is None


def test_live_work_lane_is_bounded_size_one_and_newest_wins():
    runtime = object.__new__(ContinualMappingRuntime)
    runtime._dispositions_since_tick = __import__("collections").deque(maxlen=4096)
    runtime._latest_tick_work = None
    runtime._latest_report_work = None
    runtime._active_materialization_work = None
    runtime._work_budget_owner = WorkBudgetOwner()

    for ordinal in range(10_000):
        source = _source(ordinal)
        _publish_snapshot_outputs(runtime, _snapshot(source))
        disposition = ObservationDisposition(
            source=source,
            snapshot_id=17,
            kind=DispositionKind.ADMIT,
            observe=True,
            mapping_candidate=True,
            admit_supervision=True,
            allow_geometry_insertion=True,
            residency=ResidencyIntent.HOT,
            reason="admitted",
            modality_eligibility=_admitted_modalities(),
        )
        runtime.record_disposition(disposition)

    assert len(runtime._dispositions_since_tick) == 4096
    assert runtime._latest_tick_work[1].supervision_sources == (_source(9_999),)


def test_full_admitted_replay_catalog_outlives_the_legacy_candidate_capacity():
    runtime = object.__new__(ContinualMappingRuntime)
    runtime._dispositions_since_tick = __import__("collections").deque(maxlen=4096)
    runtime._latest_tick_work = None
    runtime._latest_report_work = None
    runtime._active_materialization_work = None
    runtime._pending_supervision_modalities = __import__("collections").OrderedDict()
    runtime._pending_supervision_capacity = 1
    runtime._replay_modalities = __import__("collections").OrderedDict()
    runtime._replay_weights = {}
    runtime._replay_candidate_capacity = 2
    runtime._work_budget_owner = WorkBudgetOwner()
    sources = tuple(_source(ordinal) for ordinal in range(6))

    for snapshot_id, source in enumerate(sources, start=20):
        snapshot = replace(_snapshot(source), snapshot_id=snapshot_id)
        _publish_snapshot_outputs(runtime, snapshot)
        runtime.record_disposition(
            ObservationDisposition(
                source=source,
                snapshot_id=snapshot_id,
                kind=DispositionKind.ADMIT,
                observe=True,
                mapping_candidate=True,
                admit_supervision=True,
                allow_geometry_insertion=True,
                residency=ResidencyIntent.HOT,
                reason="admitted",
                modality_eligibility=_admitted_modalities(),
            )
        )

    assert tuple(runtime._replay_modalities) == sources
    assert set(runtime._replay_weights) == set(sources)


def test_retired_planned_replay_payload_is_rebuilt_from_cache_without_source_decode(
    tmp_path, monkeypatch
):
    source = _source(8)
    intrinsics = CameraIntrinsics(
        fx=4.0, fy=5.0, cx=1.5, cy=1.0, width=4, height=3
    )
    rgb = np.full((3, 4, 3), 128, dtype=np.uint8)
    depth = np.full((3, 4), 1.5, dtype=np.float32)
    db = FrameObservationDB(
        device="cpu",
        max_cold_payload_frames=1,
        cold_payload_spill_root=tmp_path / "cold-payloads",
    )
    record = db.record_prepared_historical_candidate(
        source_identity=source,
        pose_c2w=np.eye(4, dtype=np.float32),
        timestamp=source.timestamp_seconds,
        intrinsics=intrinsics,
        cold_pixels=ColdPixelHandle(
            source_identity=source.fingerprint(),
            rgb=rgb,
            depth=depth,
            intrinsics=intrinsics,
            rgb_encoding="rgb8",
            depth_encoding="depth_float32",
            depth_units="m",
        ),
        modality_eligibility=_admitted_modalities(),
        disposition_snapshot_id=8,
        semantic_action="commit",
    )
    assert not db.can_accept_prepared_cold_payload(payload_bytes=1)
    assert db.commit_cold_payload_spill_receipts(
        db.write_pending_cold_payload_spills(max_items=1)
    ) == 1
    cold = db.get_cold_pixels(record.frame_id)
    assert cold is not None and cold.reload_token is not None
    rgb.fill(0)
    depth.fill(0)
    runtime = object.__new__(ContinualMappingRuntime)
    runtime.scene = SimpleNamespace(_payload_residency_dirty=False)
    runtime._observation_db = db
    runtime._admission_prefetch_lock = threading.Lock()
    runtime._admission_prefetch_cache = {}
    runtime._admission_prefetch_pending = set()
    runtime._admission_prefetch_protected_frame_ids = set()
    runtime._admission_prefetch_cache_max_entries = 1
    runtime._admission_prefetch_scheduled_total = 0
    runtime._admission_prefetch_completed_total = 0
    runtime._admission_prefetch_failed_total = 0
    runtime._admission_prefetch_hits_total = 0
    runtime._admission_prefetch_terminal_failures = OrderedDict()
    runtime._admission_prefetch_executor = ThreadPoolExecutor(max_workers=1)
    monkeypatch.setattr(
        runtime,
        "_decode_reload_token",
        lambda _token: pytest.fail("cache replay must not reopen the original source"),
    )
    camera = SimpleNamespace(
        _mapping_frame=None,
        _streaming_frame=None,
        _live_payload_retired=True,
        original_image=None,
        _sensor_depth_cache=None,
        _canonical_native_resolution_state=None,
        image_width=1,
        image_height=1,
        width=1,
        height=1,
        fx=1.0,
        fy=1.0,
        cx=0.0,
        cy=0.0,
        gsplat=object(),
    )

    assert runtime._restore_camera_payload_from_cold(source, camera) is False
    runtime._admission_prefetch_executor.shutdown(wait=True)
    restored = runtime._restore_camera_payload_from_cold(source, camera)

    assert restored is True
    assert camera.original_image.device.type == "cpu"
    assert camera.original_image.shape == (3, 3, 4)
    assert camera.original_image.dtype == torch.float32
    assert camera.original_image.mean().item() == pytest.approx(128 / 255.0)
    assert camera._sensor_depth_cache.device.type == "cpu"
    assert camera._sensor_depth_cache.shape == (3, 4)
    assert (camera.image_width, camera.image_height) == (4, 3)
    assert (camera.fx, camera.fy, camera.cx, camera.cy) == (4.0, 5.0, 1.5, 1.0)
    assert camera._live_payload_retired is False
    assert camera.gsplat is None
    assert runtime.scene._payload_residency_dirty is True


def test_default_b4_off_session_catalogs_captured_packet_before_source_reuse(
    tmp_path, monkeypatch
):
    """Ordinary composition catalogues the owned packet through its disposition sink."""
    source = _source(8)
    intrinsics = CameraIntrinsics(
        fx=4.0, fy=5.0, cx=1.5, cy=1.0, width=4, height=3
    )
    db = FrameObservationDB(
        device="cpu",
        max_cold_payload_frames=1,
        cold_payload_spill_root=tmp_path / "cold-payloads",
    )
    runtime = object.__new__(ContinualMappingRuntime)
    monkeypatch.setattr(
        production_application, "StreamingObservation", StreamingObservation, raising=False
    )
    runtime._closed = False
    runtime.initialized = True
    runtime.frames_seen = 0
    runtime._active_materialization_work = None
    runtime._last_materialized_work = None
    runtime._latest_tick_work = None
    runtime._latest_report_work = None
    runtime._latest_control_packet = None
    runtime._latest_control_outputs = None
    runtime._dispositions_since_tick = __import__("collections").deque(maxlen=16)
    runtime._observation_db = db
    runtime._work_budget_owner = SimpleNamespace(resolve=lambda *_args, **_kwargs: _plan(source, insertion_seeds=0))
    runtime._native_geometry_evidence_purpose = lambda: None
    runtime._work_clock_owner_if_available = lambda: None
    runtime._pending_supervision_modalities = OrderedDict()
    runtime._pending_supervision_capacity = 1
    runtime._replay_modalities = OrderedDict()
    runtime._replay_weights = {}
    runtime._admitted_observations_total = 0
    runtime._admitted_clock_ns = 0
    runtime._source_clock_ns = 0
    runtime._latest_admitted_optimizer_update = 0
    runtime._services = SimpleNamespace(four_clocks=None)
    runtime._register_historical_record = lambda *_args, **_kwargs: None
    runtime._maybe_schedule_admission_prefetch = lambda *_args, **_kwargs: None
    runtime._maybe_prematerialize_admitted_supervision = lambda *_args, **_kwargs: None

    camera = SimpleNamespace(c2w=np.eye(4, dtype=np.float32), _is_bootstrap=False)

    def ingest(local, *, record_fn):
        record_fn(camera, local, True, False)
        return camera, local, True

    runtime.scene = SimpleNamespace(
        ingest_canonical_frame=ingest,
        retire_live_admitted_payloads=lambda: None,
    )
    rgb = np.full((3, 4, 3), 128, dtype=np.uint8)
    depth = np.full((3, 4), 1.5, dtype=np.float32)
    packet = SensorPacket(
        source=source,
        rgb=BufferRef.from_numpy(rgb, encoding="rgb8", ownership=BufferOwnership.BORROWED),
        depth=BufferRef.from_numpy(
            depth, encoding="32FC1", units="m", ownership=BufferOwnership.BORROWED
        ),
        intrinsics=intrinsics,
        tracking=TrackingObservation(
            source=source,
            pose=Pose(np.eye(4, dtype=np.float32)),
            status=TrackingStatus.VALID,
            provider="test",
        ),
    )
    disposition = ObservationDisposition(
        source=source,
        snapshot_id=17,
        kind=DispositionKind.ADMIT,
        observe=True,
        mapping_candidate=True,
        admit_supervision=True,
        allow_geometry_insertion=True,
        residency=ResidencyIntent.HOT,
        reason="admitted",
        modality_eligibility=_admitted_modalities(),
    )

    def snapshot_provider(packet, observation, _telemetry):
        snapshot = replace(_snapshot(source), tracking=observation)
        runtime._latest_control_packet = packet
        runtime._latest_control_outputs = (snapshot, _proposals())
        return snapshot

    application = ComposedMappingApplication(
        MappingApplicationOptions(
            tracking_provider=ExternalTrackingProvider(),
            mapper_sink=runtime.map_frame,
            optimizer_tick=runtime.tick,
            observation_owner=SimpleNamespace(
                decide=lambda _snapshot: disposition,
                permit=lambda _disposition: object(),
            ),
            control_snapshot_provider=snapshot_provider,
            disposition_sink=runtime.record_disposition,
            packet_capture=runtime.capture_packet_for_history,
            source_consumed_observer=lambda _packet: (rgb.fill(0), depth.fill(0)),
            runtime=runtime,
        )
    )
    session = application.session
    outcome = session.submit(packet)
    record = db.get_frame(db.frame_id_for_source(source))

    assert outcome.accepted
    assert rgb.max() == 0 and depth.max() == 0
    assert record is not None and record.historical_eligible
    cold = db.get_cold_pixels(record.frame_id)
    assert cold is not None
    assert cold.rgb[0, 0, 0] == 128
    assert cold.depth[0, 0] == pytest.approx(1.5)

    runtime._admission_prefetch_executor = ThreadPoolExecutor(max_workers=1)
    runtime._cold_payload_spill_future = None
    runtime._batch_size = 1
    later = SensorPacket(
        source=_source(9),
        rgb=BufferRef.from_numpy(np.full((3, 4, 3), 9, dtype=np.uint8), encoding="rgb8"),
        depth=BufferRef.from_numpy(
            np.full((3, 4), 1.0, dtype=np.float32), encoding="32FC1", units="m"
        ),
        intrinsics=intrinsics,
    )
    with pytest.raises(PreparedIngressDeferred):
        session.submit(later)
    assert db.pending_cold_payload_spill_count == 1
    runtime._poll_cold_payload_spill()
    runtime._admission_prefetch_executor.shutdown(wait=True)
    runtime._poll_cold_payload_spill()
    cold = db.get_cold_pixels(record.frame_id)
    assert cold is not None and cold.reload_token is not None

    class _Event:
        def __init__(self, **_kwargs):
            self.recorded = False

        def record(self):
            self.recorded = True

    for name, value in {
        "_mapping_frame": None,
        "_streaming_frame": None,
        "_live_payload_retired": True,
        "original_image": None,
        "_sensor_depth_cache": None,
        "_canonical_native_resolution_state": None,
        "_canonical_resolution_binding_native_state": None,
        "_canonical_resolution_tier": None,
        "image_width": 1,
        "image_height": 1,
        "width": 1,
        "height": 1,
        "fx": 1.0,
        "fy": 1.0,
        "cx": 0.0,
        "cy": 0.0,
        "gsplat": object(),
    }.items():
        setattr(camera, name, value)
    runtime._control_tick_epoch = 0
    runtime._work_clock_owner = RuntimeWorkClockOwner()
    runtime._last_materialized_work = None
    runtime._pending_integration = None
    runtime._pending_integration_since_ns = None
    runtime.gaussians = SimpleNamespace(get_xyz=torch.zeros((1, 3)), active_sh_degree=0)
    runtime._transfer_owner = SimpleNamespace(prefetch_cameras=lambda _cameras: None)
    runtime.config = SimpleNamespace(static_options={})
    runtime._runtime_policy = SimpleNamespace(random_background=False, sh_update_interval=1)
    runtime._background = torch.zeros(3)
    runtime._depth_loss_policy = SimpleNamespace(enabled=False, weight=0.0)
    runtime._optimization_runtime = object()
    runtime._engine = SimpleNamespace(session=object())
    runtime.backend = SimpleNamespace(map_version=3, topology_version=2)
    runtime.pipe = object()
    runtime.iteration = 0
    runtime._next_gpu_transaction_id = 0
    runtime._source_arrival_ns = {}
    runtime._first_supervision_latency_ms = []
    runtime._geometry_commit_latency_ms = []
    runtime._map_clock_ns = 0
    runtime._supervision_views_completed = 0
    runtime.optimizer_steps = 0
    runtime._admission_prefetch_cache = {}
    runtime._admission_prefetch_pending = set()
    runtime._admission_prefetch_protected_frame_ids = set()
    runtime._admission_prefetch_cache_max_entries = 1
    runtime._admission_prefetch_scheduled_total = 0
    runtime._admission_prefetch_completed_total = 0
    runtime._admission_prefetch_failed_total = 0
    runtime._admission_prefetch_hits_total = 0
    runtime._admission_prefetch_terminal_failures = OrderedDict()
    runtime._admission_prefetch_lock = threading.Lock()
    runtime._admission_prefetch_executor = ThreadPoolExecutor(max_workers=1)
    runtime.scene.train_cameras = [camera]
    runtime.scene.current_camera = None
    runtime.scene.hot_camera_ids = lambda: ()
    runtime.scene._payload_residency_dirty = False
    runtime._residency_ledger = SimpleNamespace(release=lambda _ids: None)
    runtime._commit_prepared_cold_payload_spill_receipts = lambda: None
    runtime._assert_plan_version_fence = lambda _plan: None
    runtime._versions = lambda: (3, 2, 3)
    runtime._population_live_requested = lambda: False
    runtime._note_scene_camera_page_in = lambda *_args: None
    runtime._begin_optimization_gpu_envelope = lambda *_args, **_kwargs: 0
    runtime._stamp_completed_result_map_version_after = lambda _result: None
    runtime._record_completed_optimization_result = lambda *_args, **_kwargs: SimpleNamespace(
        optimizer_updates=1, supervised_rgb_pixels=12
    )
    runtime._observe_population_policy_inputs_after_step = lambda *_args, **_kwargs: None
    runtime._maybe_raise_sh_degree = lambda **_kwargs: None
    runtime._run_due_lifecycle = lambda *_args, **_kwargs: None
    runtime._observe_replay_evidence = lambda *_args, **_kwargs: None
    runtime._supervision_work_clocks_or_create = lambda: SimpleNamespace(observe=lambda _result: None)
    runtime._post_step_structural_mutation = lambda *_args, **_kwargs: (False, None)
    runtime._record_completed_mcmc_commit = lambda _mutation: None
    runtime._record_successful_replay_supervision = lambda *_args, **_kwargs: None
    completed = []
    runtime.record_successful_supervision_sources = lambda sources: completed.extend(sources)
    runtime._retire_completed_candidates = lambda _sources: None
    runtime._sample_primary_training_psnr = lambda *_args: None
    runtime._sample_primary_training_spatial = lambda *_args: None
    runtime._submit_gpu_timing = lambda *_args, **_kwargs: None
    runtime._commit_mutation_barrier = lambda *_args, **_kwargs: None
    runtime._record_completed_depth_mutation = lambda _mutation: None
    runtime._services.poll_map_activity = lambda: None
    runtime._record_transaction_cost = lambda *_args, **_kwargs: None
    runtime._tick_result = lambda executed, reason, **kwargs: SimpleNamespace(
        executed=executed, reason=reason, **kwargs
    )
    monkeypatch.setattr(
        runtime,
        "_decode_reload_token",
        lambda _token: pytest.fail("cache replay must not reopen the original source"),
    )
    monkeypatch.setattr(
        production_application, "torch", SimpleNamespace(cuda=SimpleNamespace(Event=_Event))
    )
    monkeypatch.setattr(
        production_application,
        "execute_fixed_topology_transaction",
        lambda *_args, **_kwargs: SimpleNamespace(
            optimizer_step_count=1,
            supervision_view_count=1,
            supervised_pixel_count=12,
            rasterizer_call_count=1,
            microbatch_sizes=(1,),
        ),
        raising=False,
    )
    monkeypatch.setattr(production_application, "materialize_resolution_tier", lambda *_args, **_kwargs: None, raising=False)
    monkeypatch.setattr(production_application, "CacheHandle", CacheHandle, raising=False)
    monkeypatch.setattr(production_application, "BatchExecutionMode", BatchExecutionMode, raising=False)
    monkeypatch.setattr(production_application, "FixedTopologyBatchBuilder", FixedTopologyBatchBuilder, raising=False)
    monkeypatch.setattr(production_application, "GpuWorkEnvelope", GpuWorkEnvelope, raising=False)
    monkeypatch.setattr(production_application, "GpuTelemetryDescriptor", lambda **kwargs: kwargs, raising=False)
    monkeypatch.setattr(production_application, "TransactionType", SimpleNamespace(OPTIMIZATION="optimization"), raising=False)
    monkeypatch.setattr(
        __import__("scene.cameras", fromlist=["prepare_camera_for_render"]),
        "prepare_camera_for_render",
        lambda _camera, **_kwargs: None,
    )

    prefetch_gate = threading.Event()
    del runtime._maybe_schedule_admission_prefetch
    decode_worker = runtime._prefetch_decode_worker

    def blocked_decode(*args):
        assert prefetch_gate.wait(timeout=1)
        return decode_worker(*args)

    monkeypatch.setattr(runtime, "_prefetch_decode_worker", blocked_decode)

    pending = application.tick()
    assert (pending.executed, pending.reason) == (False, "cold_payload_pending")
    prefetch_gate.set()
    runtime._admission_prefetch_executor.shutdown(wait=True)
    served = application.tick()

    assert served.executed is True
    assert completed == [source]
    assert camera.original_image.mean().item() == pytest.approx(128 / 255.0)
    assert camera._sensor_depth_cache.mean().item() == pytest.approx(1.5)

def test_exact_planned_source_is_resolved_without_random_substitution():
    wanted = _source(8)
    other = _source(7)
    wanted_camera = SimpleNamespace(
        _mapping_frame=SimpleNamespace(source=wanted)
    )
    runtime = object.__new__(ContinualMappingRuntime)
    runtime._source_identities_by_ordinal = {}
    runtime.scene = SimpleNamespace(
        train_cameras=[SimpleNamespace(_mapping_frame=SimpleNamespace(source=other)), wanted_camera],
        current_camera=None,
    )

    assert runtime._camera_for_source(wanted) is wanted_camera


def test_cold_camera_keeps_compact_source_identity_after_mapping_frame_retirement():
    wanted = _source(8)
    wanted_camera = SimpleNamespace(
        _mapping_frame=None,
        _source_identity=wanted,
    )
    runtime = object.__new__(ContinualMappingRuntime)
    runtime._source_identities_by_ordinal = {}
    runtime.scene = SimpleNamespace(
        train_cameras=[wanted_camera],
        current_camera=None,
    )

    assert runtime._camera_for_source(wanted) is wanted_camera


def test_attached_full_identity_conflict_cannot_fall_back_to_ordinal_registry():
    import pytest

    wanted = _source(8)
    runtime = object.__new__(ContinualMappingRuntime)
    runtime._source_identities_by_ordinal = {8: wanted}
    runtime.scene = SimpleNamespace(
        train_cameras=[
            SimpleNamespace(
                uid=8,
                _mapping_frame=SimpleNamespace(source=SourceIdentity("other", 1, 8, 1008)),
            )
        ],
        current_camera=None,
    )

    with pytest.raises(RuntimeError, match="did not resolve"):
        runtime._camera_for_source(wanted)


def test_integration_uses_the_plan_insertion_cap(monkeypatch):
    source = _source(5)
    recorded = {}

    class Candidates:
        pts = SimpleNamespace(device="cuda")

        def __len__(self):
            return 1

    runtime = object.__new__(ContinualMappingRuntime)
    runtime.scene = object()
    runtime.args = SimpleNamespace(
        streaming_insert_from_depth=False,
        streaming_max_new_gaussians_per_frame=100,
    )
    runtime.gaussians = object()
    runtime.pipe = object()
    runtime._background = object()
    runtime.coordinator = object()
    runtime.backend = object()
    runtime._prev_insert_frame = None
    runtime._prev_insert_depth_m = None
    runtime._transfer_owner = None
    runtime.frames_seen = 1
    runtime.iteration = 1
    runtime._work_clock_owner = RuntimeWorkClockOwner()
    runtime._depth_insertion_policy = DepthInsertionPolicy()
    monkeypatch.setattr(
        "eligsir.runtime.production_application.render",
        lambda *_args, **_kwargs: {"alpha": None, "rendered_depth": None},
    )
    monkeypatch.setattr(
        "eligsir.runtime.production_application.insertion_proposal_from_plan",
        lambda plan, candidates, **kwargs: (plan, candidates, kwargs),
    )
    monkeypatch.setattr(
        "eligsir.runtime.production_application.normalize_pending_candidates",
        lambda pending, **_kwargs: pending[0],
    )
    monkeypatch.setattr(
        "eligsir.runtime.production_application.insert_gaussians_from_frame_device_native",
        lambda *_args, **kwargs: (
            recorded.update(kwargs) or Candidates(),
            {},
        ),
    )
    frame = SimpleNamespace(_depth_np=np.ones((2, 2), dtype=np.float32))

    proposal = runtime._prepare_insertion_proposal(
        object(), frame, source, _plan(source, insertion_seeds=7)
    )

    assert recorded["insertion_plan"].candidate_budget == 7
    assert recorded["modality_eligibility"].depth_insertion
    assert proposal is not None
    assert proposal[2]["iteration"] == 0


def test_record_disposition_uses_frozen_snapshot_freshness_not_disposition_trace():
    source = _source(3)
    runtime = object.__new__(ContinualMappingRuntime)
    runtime._dispositions_since_tick = __import__("collections").deque(maxlen=4096)
    runtime._latest_tick_work = None
    runtime._latest_report_work = None
    runtime._active_materialization_work = None
    _publish_snapshot_outputs(runtime, _snapshot(source))
    runtime._work_budget_owner = WorkBudgetOwner()
    disposition = ObservationDisposition(
        source=source,
        snapshot_id=17,
        kind=DispositionKind.ADMIT,
        observe=True,
        mapping_candidate=True,
        admit_supervision=True,
        allow_geometry_insertion=True,
        residency=ResidencyIntent.HOT,
        reason="admitted",
        evidence_trace=(EvidenceFact("backpressure.pressure_state", "critical"),),
        modality_eligibility=_admitted_modalities(),
    )

    runtime.record_disposition(disposition)

    assert runtime._latest_tick_work[1].freshness_mode is FreshnessMode.HEALTHY


def test_control_evidence_rejects_cross_domain_name_collisions():
    from eligsir.runtime.production_application import combine_control_evidence

    with pytest.raises(ValueError, match="cross-domain control evidence collision"):
        combine_control_evidence(
            (EvidenceFact("mapping_service_ratio", 9.0),),
            (EvidenceFact("mapping_service_ratio", 1.0),),
        )


def test_zero_mutation_budget_never_calls_strategy():
    source = _source(1)
    runtime = object.__new__(ContinualMappingRuntime)
    runtime._mcmc_strategy = SimpleNamespace(
        step_post_backward=lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("mutation strategy was called")
        )
    )

    runtime._mutate_after_step(SimpleNamespace(backward_ran=True), _plan(source))


def test_positive_mutation_budget_fails_closed_on_new_frame_plan():
    source = _source(1)
    runtime = object.__new__(ContinualMappingRuntime)
    plan = _plan(source)
    plan = MappingWorkPlan(
        **{
            **{
                field: getattr(plan, field)
                for field in plan.__dataclass_fields__
            },
            "allow_growth": True,
            "mutation_budget": MutationBudget(1),
        }
    )

    import pytest

    with pytest.raises(RuntimeError, match="replay-only"):
        runtime._mutate_after_step(SimpleNamespace(backward_ran=True), plan)


def test_idle_positive_mutation_budget_builds_one_delegated_proposal():
    source = _source(1)
    runtime = object.__new__(ContinualMappingRuntime)
    proposal = object()
    runtime._mcmc_strategy = SimpleNamespace(
        delegated_post_backward_proposal=lambda **kwargs: (
            setattr(runtime, "built_mcmc_kwargs", kwargs) or proposal
        )
    )
    runtime._mcmc_cfg = __import__(
        "utils.mcmc_schedule", fromlist=["MCMCScheduleConfig"]
    ).MCMCScheduleConfig(start_iter=0)
    runtime._runtime_policy = SimpleNamespace(cap_max=100)
    runtime._optimization_policy = SimpleNamespace()
    runtime._use_energy_mcmc = False
    runtime.iteration = 600
    runtime.gaussians = SimpleNamespace(get_xyz=np.zeros((10, 3)))
    runtime._versions = lambda: (3, 2, 3)
    commits = []
    runtime._commit_mutation_barrier = lambda proposals, **kwargs: (
        commits.append((proposals, kwargs)) or "committed"
    )
    plan = replace(
        _plan(source, insertion_seeds=0),
        supervision_sources=(),
        supervision_work=(),
        replay_sources=(source,),
        replay_work=(SupervisionWork(source, "native", _admitted_modalities()),),
        allow_growth=True,
        allow_relocation=True,
        mutation_budget=MutationBudget(1),
    )
    result = SimpleNamespace(
        backward_ran=True,
        render_outputs=SimpleNamespace(payload={}),
        evidence=None,
        xyz_lr=0.01,
    )

    committed = runtime._mutate_after_step(result, plan)

    assert committed == "committed"
    assert commits[0][0] == (proposal,)
    assert commits[0][1]["allow_delegated"] is True
    assert runtime.built_mcmc_kwargs["sched"]["allow_growth"] is True
    assert runtime.built_mcmc_kwargs["sched"]["allow_position_noise"] is False


def test_post_refinement_mcmc_reenables_position_noise():
    source = _source(1)
    runtime = object.__new__(ContinualMappingRuntime)
    proposal = object()
    runtime._mcmc_strategy = SimpleNamespace(
        delegated_post_backward_proposal=lambda **kwargs: (
            setattr(runtime, "built_mcmc_kwargs", kwargs) or proposal
        )
    )
    runtime._mcmc_cfg = __import__(
        "utils.mcmc_schedule", fromlist=["MCMCScheduleConfig"]
    ).MCMCScheduleConfig(start_iter=0)
    runtime._runtime_policy = SimpleNamespace(cap_max=100)
    runtime._optimization_policy = SimpleNamespace()
    runtime._use_energy_mcmc = False
    runtime._source_complete = True
    runtime.iteration = 600
    runtime.gaussians = SimpleNamespace(get_xyz=np.zeros((10, 3)))
    runtime._versions = lambda: (3, 2, 3)
    runtime._commit_mutation_barrier = lambda proposals, **_kwargs: "committed"
    plan = replace(
        _plan(source, insertion_seeds=0),
        supervision_sources=(),
        supervision_work=(),
        replay_sources=(source,),
        replay_work=(SupervisionWork(source, "native", _admitted_modalities()),),
        allow_growth=True,
        allow_relocation=True,
        mutation_budget=MutationBudget(1),
    )

    runtime._mutate_after_step(
        SimpleNamespace(
            backward_ran=True,
            render_outputs=SimpleNamespace(payload={}),
            evidence=None,
            xyz_lr=0.01,
        ),
        plan,
    )

    assert runtime.built_mcmc_kwargs["sched"]["allow_position_noise"] is True


def test_mcmc_position_noise_phase_authority_rejects_non_bool_state():
    source = _source(1)
    runtime = object.__new__(ContinualMappingRuntime)
    runtime._mcmc_strategy = SimpleNamespace(
        delegated_post_backward_proposal=lambda **kwargs: (
            setattr(runtime, "built_mcmc_kwargs", kwargs) or object()
        )
    )
    runtime._mcmc_cfg = __import__(
        "utils.mcmc_schedule", fromlist=["MCMCScheduleConfig"]
    ).MCMCScheduleConfig(start_iter=0)
    runtime._runtime_policy = SimpleNamespace(cap_max=100)
    runtime._optimization_policy = SimpleNamespace()
    runtime._use_energy_mcmc = False
    runtime._source_complete = 1
    runtime.iteration = 600
    runtime.gaussians = SimpleNamespace(get_xyz=np.zeros((10, 3)))
    runtime._versions = lambda: (3, 2, 3)
    runtime._commit_mutation_barrier = lambda proposals, **_kwargs: "committed"
    plan = replace(
        _plan(source, insertion_seeds=0),
        supervision_sources=(),
        supervision_work=(),
        replay_sources=(source,),
        replay_work=(SupervisionWork(source, "native", _admitted_modalities()),),
        allow_growth=True,
        allow_relocation=True,
        mutation_budget=MutationBudget(1),
    )

    with pytest.raises(TypeError, match="source_complete"):
        runtime._mutate_after_step(
            SimpleNamespace(
                backward_ran=True,
                render_outputs=SimpleNamespace(payload={}),
                evidence=None,
                xyz_lr=0.01,
            ),
            plan,
        )


def test_canonical_mcmc_vram_brake_suppresses_growth_at_mutation_barrier(monkeypatch):
    import eligsir.runtime.production_application as production_application

    source = _source(1)
    runtime = object.__new__(ContinualMappingRuntime)
    proposal = object()
    runtime._mcmc_strategy = SimpleNamespace(
        delegated_post_backward_proposal=lambda **kwargs: (
            setattr(runtime, "built_mcmc_kwargs", kwargs) or proposal
        )
    )
    runtime._mcmc_cfg = __import__(
        "utils.mcmc_schedule", fromlist=["MCMCScheduleConfig"]
    ).MCMCScheduleConfig(start_iter=0)
    runtime._runtime_policy = SimpleNamespace(cap_max=100)
    runtime._optimization_policy = SimpleNamespace(
        streaming_vram_brake_fraction=0.85,
        # Legacy sampling intervals cannot skip a proposed semantic growth
        # event; VRAM safety is checked on every such event.
        streaming_vram_brake_check_interval=1_000,
    )
    runtime._use_energy_mcmc = False
    runtime.iteration = 600
    runtime._work_clock_owner = RuntimeWorkClockOwner()
    runtime._work_clock_owner.record_optimization(1, 1, 100, 0)
    runtime._mcmc_cadence_owner = MCMCCadenceOwner(
        grow_interval_pixels=100,
        relocate_interval_pixels=100,
        min_optimizer_updates_after_structural_mutation=0,
    )
    runtime.gaussians = SimpleNamespace(get_xyz=np.zeros((10, 3)))
    runtime._versions = lambda: (3, 2, 3)
    runtime._commit_mutation_barrier = lambda proposals, **_kwargs: SimpleNamespace(
        change=SimpleNamespace(topology_version=2, relocated=0),
        inserted=0,
    )
    monkeypatch.setattr(
        production_application,
        "torch",
        SimpleNamespace(
            cuda=SimpleNamespace(
                is_available=lambda: True,
                memory_reserved=lambda: 90,
                get_device_properties=lambda _device: SimpleNamespace(
                    total_memory=100
                ),
            )
        ),
    )
    plan = replace(
        _plan(source, insertion_seeds=0),
        supervision_sources=(),
        supervision_work=(),
        replay_sources=(source,),
        replay_work=(SupervisionWork(source, "native", _admitted_modalities()),),
        allow_growth=True,
        allow_relocation=True,
        mutation_budget=MutationBudget(1),
    )
    result = SimpleNamespace(
        backward_ran=True,
        render_outputs=SimpleNamespace(payload={}),
        evidence=None,
        xyz_lr=0.01,
    )

    runtime._mutate_after_step(result, plan)

    assert runtime.built_mcmc_kwargs["sched"]["allow_growth"] is False
    assert runtime.built_mcmc_kwargs["sched"]["allow_relocation"] is True
    assert runtime.built_mcmc_kwargs["optimizer_updates"] == 1
    assert runtime.built_mcmc_kwargs["iteration"] == 1


def test_canonical_mcmc_rejects_stale_topology_before_strategy_or_utility_compute():
    source = _source(1)
    runtime = object.__new__(ContinualMappingRuntime)
    runtime._mcmc_strategy = SimpleNamespace(
        delegated_post_backward_proposal=lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("strategy compute ran before topology fence")
        )
    )
    runtime._mcmc_cfg = __import__(
        "utils.mcmc_schedule", fromlist=["MCMCScheduleConfig"]
    ).MCMCScheduleConfig(start_iter=0)
    runtime._runtime_policy = SimpleNamespace(cap_max=100)
    runtime._optimization_policy = SimpleNamespace()
    runtime._use_energy_mcmc = True
    runtime.iteration = 600
    runtime.gaussians = SimpleNamespace(get_xyz=np.zeros((10, 3)))
    runtime._versions = lambda: (3, 3, 3)
    runtime._work_clock_owner = RuntimeWorkClockOwner()
    runtime._work_clock_owner.record_optimization(1, 1, 100, 0)
    runtime._mcmc_cadence_owner = MCMCCadenceOwner(
        grow_interval_pixels=100,
        relocate_interval_pixels=100,
        min_optimizer_updates_after_structural_mutation=0,
    )
    plan = replace(
        _plan(source, insertion_seeds=0),
        supervision_sources=(),
        supervision_work=(),
        replay_sources=(source,),
        replay_work=(SupervisionWork(source, "native", _admitted_modalities()),),
        allow_growth=True,
        allow_relocation=True,
        mutation_budget=MutationBudget(1),
    )

    with pytest.raises(RuntimeError, match="stale topology before MCMC utility compute"):
        runtime._mutate_after_step(
            SimpleNamespace(backward_ran=True),
            plan,
        )


def test_canonical_mcmc_not_due_skips_utility_and_strategy_compute():
    source = _source(1)
    runtime = object.__new__(ContinualMappingRuntime)
    runtime._mcmc_strategy = SimpleNamespace(
        delegated_post_backward_proposal=lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("strategy ran without semantic mutation debt")
        )
    )
    runtime._mcmc_cfg = __import__(
        "utils.mcmc_schedule", fromlist=["MCMCScheduleConfig"]
    ).MCMCScheduleConfig(start_iter=0)
    runtime._runtime_policy = SimpleNamespace(cap_max=100)
    runtime._optimization_policy = SimpleNamespace(
        streaming_vram_brake_fraction=0.0,
    )
    runtime._use_energy_mcmc = True
    runtime.iteration = 1
    runtime.gaussians = SimpleNamespace(get_xyz=np.zeros((10, 3)))
    runtime._versions = lambda: (3, 2, 3)
    runtime._work_clock_owner = RuntimeWorkClockOwner()
    runtime._work_clock_owner.record_optimization(1, 1, 1, 0)
    runtime._mcmc_cadence_owner = MCMCCadenceOwner(
        grow_interval_pixels=100,
        relocate_interval_pixels=100,
        min_optimizer_updates_after_structural_mutation=0,
    )
    runtime._compute_post_step_utility = lambda *_args, **_kwargs: (
        _ for _ in ()
    ).throw(AssertionError("utility ran without semantic mutation debt"))
    plan = replace(
        _plan(source, insertion_seeds=0),
        supervision_sources=(),
        supervision_work=(),
        replay_sources=(source,),
        replay_work=(SupervisionWork(source, "native", _admitted_modalities()),),
        allow_growth=True,
        allow_relocation=True,
        mutation_budget=MutationBudget(1),
    )

    committed = runtime._mutate_after_step(
        SimpleNamespace(backward_ran=True),
        plan,
    )

    assert committed is None
    assert runtime._mcmc_cadence_owner.state_dict()[
        "observed_supervised_rgb_pixels"
    ] == 1


def test_idle_replay_plan_can_actuate_one_mcmc_epoch_without_depth_insertion():
    source = _source(1)
    snapshot = replace(
        _snapshot(source),
        limits=ControlLimits(1, 4, 1, 7, 1, 1024),
    )
    active = _plan(source)
    provider = MutationProposalProvider()

    idle = WorkBudgetOwner().resolve_idle_replay(
        snapshot,
        active,
        replay_modalities={source: _admitted_modalities()},
        replay_weights={source: 1.0},
        batch_size=1,
        optimizer_steps_completed=5,
        versions=(3, 2, 3),
        proposals=provider.propose_idle(snapshot),
    )

    assert idle.supervision_sources == ()
    assert idle.insertion.enabled is False
    assert idle.allow_growth is True
    assert idle.allow_relocation is True
    assert idle.mutation_budget.max_total == 1


def test_source_complete_idle_plan_schedules_replay_depth_from_successful_replay_views():
    source = _source(1)
    snapshot = replace(
        _snapshot(source),
        limits=ControlLimits(
            1, 4, 1, 7, 1, 1024, max_replay_insertion_seeds=64
        ),
    )
    active = _plan(source)
    runtime = object.__new__(ContinualMappingRuntime)
    runtime._optimization_policy = _DISABLED_REPLAY_CHALLENGER_POLICY
    runtime._batch_size = 1
    runtime._replay_modalities = __import__("collections").OrderedDict(
        ((source, _admitted_modalities()),)
    )
    runtime._replay_weights = {source: 1.0}
    runtime._work_budget_owner = WorkBudgetOwner()
    runtime._resolution_policy_provider = None
    runtime._mutation_policy_provider = MutationProposalProvider()
    runtime._source_complete = True
    runtime._replay_depth_max_per_event = 64
    runtime._replay_depth_interval = 50
    runtime._replay_supervision_views_completed = 99
    runtime._next_replay_depth_supervision_view = 100
    runtime.optimizer_steps = runtime.iteration = 10_000
    runtime._versions = lambda: (3, 2, 3)

    before = runtime._idle_replay_plan(snapshot, active)
    runtime._replay_supervision_views_completed = 100
    due = runtime._idle_replay_plan(snapshot, active)

    assert before.replay_insertion is None
    assert due.replay_insertion is not None
    assert due.replay_insertion.source == source
    assert due.replay_insertion.seed_budget == 64
    assert due.mutation_budget.max_total == 0
    assert not due.allow_growth and not due.allow_relocation


def test_replay_depth_deadline_advances_from_current_successful_views_without_burst():
    runtime = object.__new__(ContinualMappingRuntime)
    runtime._replay_depth_interval = 50
    runtime._next_replay_depth_supervision_view = 100
    runtime._replay_supervision_views_completed = 237

    runtime._consume_replay_depth_deadline()

    assert runtime._next_replay_depth_supervision_view == 287


def test_failed_optimizer_attempts_never_advance_replay_depth_cadence():
    runtime = object.__new__(ContinualMappingRuntime)
    runtime._source_complete = True
    runtime._replay_depth_max_per_event = 64
    runtime._replay_depth_interval = 50
    runtime._replay_supervision_views_completed = 49
    runtime._next_replay_depth_supervision_view = 50
    runtime.iteration = 100_000
    runtime.optimizer_steps = 100_000

    assert runtime._replay_depth_due() is False


def test_only_successful_replay_identities_advance_replay_depth_clock():
    new_source = _source(1)
    replay_source = _source(2)
    runtime = object.__new__(ContinualMappingRuntime)
    runtime._replay_supervision_views_completed = 7
    plan = replace(
        _plan(new_source),
        replay_sources=(replay_source,),
        replay_work=(
            replace(_plan(replay_source).supervision_work[0]),
        ),
    )

    runtime._record_successful_replay_supervision(
        plan,
        successful_sources={new_source, replay_source},
        unique_sources_known=True,
    )
    runtime._record_successful_replay_supervision(
        plan,
        successful_sources={replay_source},
        unique_sources_known=False,
    )

    assert runtime._replay_supervision_views_completed == 8


def test_primary_psnr_cadence_uses_successful_optimizer_updates_not_attempts():
    runtime = object.__new__(ContinualMappingRuntime)
    runtime.optimizer_steps = 49
    runtime.iteration = 100_000
    runtime._primary_psnr_interval = 50
    runtime._next_primary_psnr_optimizer_update = 50

    assert runtime._primary_psnr_due() is False
    runtime.iteration += 100_000
    assert runtime._primary_psnr_due() is False
    runtime.optimizer_steps = 50
    assert runtime._primary_psnr_due() is True

    runtime._consume_primary_psnr_deadline()
    assert runtime._next_primary_psnr_optimizer_update == 100


def test_spatial_quality_cadence_uses_successful_replay_views_without_catchup():
    runtime = object.__new__(ContinualMappingRuntime)
    runtime._source_complete = True
    runtime.iteration = 500_000
    runtime._spatial_replay_focus_enabled = True
    runtime._spatial_replay_focus_interval = 25
    runtime._replay_supervision_views_completed = 74
    runtime._next_spatial_quality_replay_view = 75

    assert runtime._spatial_quality_due() is False
    runtime._replay_supervision_views_completed = 100
    assert runtime._spatial_quality_due() is True

    runtime._consume_spatial_quality_deadline()
    assert runtime._next_spatial_quality_replay_view == 125


def test_spatial_quality_cadence_is_available_during_online_mapping():
    runtime = object.__new__(ContinualMappingRuntime)
    runtime._source_complete = False
    runtime._spatial_replay_focus_enabled = True
    runtime._spatial_replay_focus_interval = 25
    runtime._replay_supervision_views_completed = 100
    runtime._next_spatial_quality_replay_view = 75

    assert runtime._spatial_quality_due() is True


def test_canonical_runtime_promotes_supported_old_geometry_lifecycle():
    runtime = object.__new__(ContinualMappingRuntime)
    runtime._optimization_policy = SimpleNamespace(
        streaming_lifecycle_enabled=True,
        streaming_utility_ema_beta=0.95,
        streaming_mature_age_frames=15,
        streaming_mature_min_utility=0.0,
        streaming_min_support_views=2,
        streaming_freeze_age_frames=-1,
    )
    runtime.frames_seen = 20_000
    runtime.optimizer_steps = 20
    runtime._versions = lambda: (3, 7, 4)
    runtime.gaussians = SimpleNamespace(
        lifecycle_state=torch.tensor([1, 1, 0, 0], dtype=torch.int8),
        birth_frame=torch.tensor([0, 10, 0, 0], dtype=torch.int32),
        birth_iter=torch.tensor([0, 10, 0, 0], dtype=torch.int32),
        provisional=torch.tensor([False, False, True, True]),
        support_count=torch.tensor([2, 2, 2, 1], dtype=torch.int16),
        utility_ema=torch.zeros(4, dtype=torch.float32),
    )

    runtime._advance_lifecycle(
        utility=torch.ones(4),
        optimizer_updates=20,
        expected_topology_version=7,
    )

    assert runtime.gaussians.provisional.tolist() == [False, False, False, True]
    assert runtime.gaussians.lifecycle_state.tolist() == [2, 1, 2, 0]
    assert runtime._lifecycle_transition_calls == 1


def test_lifecycle_cadence_uses_optimizer_updates_without_catch_up_bursts():
    runtime = object.__new__(ContinualMappingRuntime)
    runtime._lifecycle_interval_updates = 8
    runtime._next_lifecycle_optimizer_update = 8

    assert runtime._lifecycle_due(7) is False
    assert runtime._lifecycle_due(8) is True

    runtime._consume_lifecycle_deadline(80)

    assert runtime._next_lifecycle_optimizer_update == 88
    assert runtime._lifecycle_due(87) is False
    assert runtime._lifecycle_due(88) is True


def test_lifecycle_age_ignores_source_frames_and_uses_successful_updates():
    runtime = object.__new__(ContinualMappingRuntime)
    runtime._optimization_policy = SimpleNamespace(
        streaming_lifecycle_enabled=True,
        streaming_utility_ema_beta=0.0,
        streaming_mature_age_frames=15,
        streaming_mature_min_utility=0.0,
        streaming_min_support_views=2,
        streaming_freeze_age_frames=-1,
    )
    runtime.frames_seen = 1_000_000
    runtime._versions = lambda: (3, 7, 4)
    runtime.gaussians = SimpleNamespace(
        lifecycle_state=torch.tensor([1], dtype=torch.int8),
        birth_frame=torch.tensor([0], dtype=torch.int32),
        birth_iter=torch.tensor([0], dtype=torch.int32),
        provisional=torch.tensor([False]),
        support_count=torch.tensor([2], dtype=torch.int16),
        utility_ema=torch.zeros(1),
    )

    runtime._advance_lifecycle(
        utility=torch.ones(1),
        optimizer_updates=14,
        expected_topology_version=7,
    )

    assert runtime.gaussians.lifecycle_state.tolist() == [1]


def test_lifecycle_fails_before_compute_on_stale_topology_or_invalid_utility():
    runtime = object.__new__(ContinualMappingRuntime)
    runtime._optimization_policy = SimpleNamespace(streaming_lifecycle_enabled=True)
    runtime._versions = lambda: (3, 8, 4)
    runtime.gaussians = SimpleNamespace(
        lifecycle_state=torch.tensor([1], dtype=torch.int8),
        birth_frame=torch.tensor([0], dtype=torch.int32),
        birth_iter=torch.tensor([0], dtype=torch.int32),
        provisional=torch.tensor([False]),
        utility_ema=torch.zeros(1),
    )

    with pytest.raises(RuntimeError, match="stale topology before lifecycle"):
        runtime._advance_lifecycle(
            utility=torch.ones(1),
            optimizer_updates=20,
            expected_topology_version=7,
        )

    runtime._versions = lambda: (3, 7, 4)
    with pytest.raises(ValueError, match="finite per-Gaussian"):
        runtime._advance_lifecycle(
            utility=torch.tensor([float("nan")]),
            optimizer_updates=20,
            expected_topology_version=7,
        )


def test_stale_plan_fails_before_any_tick_compute():
    runtime = object.__new__(ContinualMappingRuntime)
    runtime.backend = SimpleNamespace(
        map_content_version=4,
        topology_version=2,
        parameter_version=4,
    )

    import pytest

    with pytest.raises(RuntimeError, match="stale mapping work plan"):
        runtime._assert_plan_version_fence(_plan(_source(1)))


def test_versions_read_the_optimizer_session_parameter_clock():
    runtime = object.__new__(ContinualMappingRuntime)
    runtime.backend = SimpleNamespace(
        map_content_version=9,
        topology_version=4,
    )
    runtime._engine = SimpleNamespace(session=SimpleNamespace(parameter_version=3))

    assert runtime._versions() == (9, 4, 3)


def test_control_evidence_reports_motion_from_last_admitted_pose():
    runtime = object.__new__(ContinualMappingRuntime)
    runtime._last_admitted_pose = np.eye(4, dtype=np.float32)
    runtime._adaptive_growth_provider = SimpleNamespace(
        observe_training_spatial_quality=lambda **_kwargs: None
    )
    runtime._adaptive_quality_risk_tolerance_db = 0.25
    runtime.publish_growth_quality_evidence(
        iteration=10,
        aggregate_psnr_db=20.0,
        gaussian_count=100,
        worst_drop_from_peak_db=0.8,
    )
    moved = np.eye(4, dtype=np.float32)
    moved[0, 3] = 0.12
    observation = TrackingObservation(
        source=_source(2),
        pose=Pose(moved),
        status=TrackingStatus.VALID,
        provider="test",
        confidence=0.8,
        diagnostics={
            "overlap": 0.4,
            "coverage_gain": 0.3,
            "reconstruction_residual": 0.2,
            "rgb_quality": 0.9,
            "depth_quality_score": 0.7,
        },
    )

    packet = SimpleNamespace(
        rgb=BufferRef.from_numpy(
            np.full((2, 2, 3), 128, dtype=np.uint8), encoding="rgb8"
        ),
        depth=None,
    )
    evidence = {
        fact.name: fact.value
        for fact in runtime._control_evidence(packet, observation)
    }

    assert evidence["translation_m"] == pytest.approx(0.12)
    assert evidence["rotation_deg"] == pytest.approx(0.0)
    assert evidence["quality"] == pytest.approx(0.8)
    assert evidence["overlap"] == pytest.approx(0.4)
    assert evidence["coverage_gain"] == pytest.approx(0.3)
    assert evidence["new_region"] is True
    assert evidence["partition_fade"] is True
    assert evidence["quality_risk"] is True
    assert evidence["parallax_deg"] is None
    assert evidence["rgb_luma"] is None
    assert evidence["reconstruction_residual"] == pytest.approx(0.2)
    assert {
        "source_rate_hz",
        "admitted_rate_hz",
        "commit_rate_hz",
        "mapping_service_ratio",
        "admission_ratio",
        "replay_ratio",
        "frontier_age_ms",
        "frontier_age_slope_ms_per_s",
        "ingress_queue_depth",
        "candidate_queue_depth",
        "transfer_queue_depth",
        "gpu_queue_depth",
        "gpu_cost_ms",
        "gpu_cost_per_update_ms",
        "observation_value",
        "admitted_observations_total",
    } <= evidence.keys()
    assert evidence["observation_value"] == pytest.approx(0.8)


def test_idle_snapshot_refreshes_published_quality_risk_evidence():
    source = _source(2)
    runtime = object.__new__(ContinualMappingRuntime)
    runtime._versions = lambda: (3, 2, 4)
    runtime._control_snapshot_id = 18
    runtime._control_tick_epoch = 6
    runtime.optimizer_steps = 12
    runtime._supervision_views_completed = 48
    runtime._adaptive_supervision_quality_evidence = (True, True)
    runtime._requested_source_fps_for_pacing = None

    refreshed = runtime._fresh_idle_snapshot(_snapshot(source))
    evidence = {fact.name: fact.value for fact in refreshed.evidence}

    assert evidence["partition_fade"] is True
    assert evidence["quality_risk"] is True


def test_post_mapping_rearm_uses_current_barrier_without_advancing_control_tick(
    tmp_path, monkeypatch
):
    from collections import OrderedDict
    from eligsir.mapping.canonical_services import CanonicalFourClocks

    source = _source(2)
    retained_snapshot = replace(
        _snapshot(source),
        snapshot_id=17,
        control_tick_epoch=2,
        source_release_rate_ratio=0.0,
    )
    now = [0.0]
    monkeypatch.setattr(
        "eligsir.mapping.canonical_services.time.perf_counter", lambda: now[0]
    )
    clocks = CanonicalFourClocks()
    clocks.record_source_frame()
    now[0] = 1.0 / 30.0
    clocks.record_source_frame()
    now[0] = 5.1
    retained_plan = _plan(source, insertion_seeds=0)
    runtime = object.__new__(ContinualMappingRuntime)
    runtime._optimization_policy = _DISABLED_REPLAY_CHALLENGER_POLICY
    runtime._last_materialized_work = (retained_snapshot, retained_plan)
    runtime._versions = lambda: (3, 2, 4)
    runtime._control_snapshot_id = 18
    runtime._control_tick_epoch = 6
    runtime.optimizer_steps = 12
    runtime._supervision_views_completed = 48
    runtime._batch_size = 4
    runtime._services = SimpleNamespace(four_clocks=clocks)
    runtime._requested_source_fps_for_pacing = 30.0
    exhausted = [True]
    runtime._source_input_exhausted_provider = lambda: exhausted[0]
    runtime._source_complete = False
    runtime._replay_modalities = OrderedDict(
        ((source, retained_plan.supervision_work[0].modality_eligibility),)
    )
    runtime._replay_weights = {source: 1.0}
    runtime._work_budget_owner = WorkBudgetOwner(
        requested_batch_size=4,
        maximum_camera_count=4,
        constellation_policy="fixed",
        refinement_resolution_policy="bounded",
    )
    for tier, wall_ms in (
        (ResolutionTier.NATIVE, 150.0),
        (ResolutionTier.HALF, 90.0),
        (ResolutionTier.QUARTER, 60.0),
    ):
        runtime._work_budget_owner.record_completed_transaction(
            transaction_class=MappingTransactionClass.REPLAY_ONLY,
            tier=tier,
            gaussian_count=retained_snapshot.gaussian_count,
            realized_camera_count=4,
            prepare_ms=0.5,
            optimization_ms=wall_ms,
            transaction_wall_ms=wall_ms,
            kernel_ms=wall_ms,
        )
    runtime._resolution_policy_provider = None
    runtime._mutation_policy_provider = None
    runtime._adaptive_growth_provider = None
    runtime._observation_db = FrameObservationDB(device="cpu")
    runtime._historical_sources_by_frame_id = {2: source}
    runtime._observation_db.add_frame(
        FrameRecord(
            frame_id=2,
            pose_c2w=np.eye(4, dtype=np.float32),
            timestamp=2.0,
            submap_id=0,
            buffer_stage=BufferStage.COLD_CACHE,
            visible_gaussian_uids=torch.zeros(0, dtype=torch.int64),
            source_identity=source.fingerprint(),
            source_ordinal=source.source_ordinal,
            historical_eligible=True,
            modality_eligibility=retained_plan.supervision_work[0].modality_eligibility,
        )
    )
    runtime._observation_db.store_cold_pixels(
        2,
        ColdPixelHandle(
            source_identity=source.fingerprint(),
            rgb=np.zeros((2, 2, 3), dtype=np.uint8),
            depth=np.ones((2, 2), dtype=np.float32),
            intrinsics=CameraIntrinsics(
                fx=1.0, fy=1.0, cx=1.0, cy=1.0, width=2, height=2
            ),
        ),
    )
    runtime._post_refinement_phase_owner = PostRefinementPhaseOwner()
    point_cloud = tmp_path / "point_cloud.ply"
    checkpoint = tmp_path / "checkpoint.pth"
    point_cloud.write_bytes(b"ply\n")
    checkpoint.write_bytes(b"checkpoint")
    runtime.mark_source_complete()
    runtime.enter_post_mapping_refinement(
        {"point_cloud": str(point_cloud), "checkpoint": str(checkpoint)}
    )

    # EOF draining has already resolved idle work at the current barrier, while
    # the last materialized frame still carries an older source-time snapshot.
    current_snapshot = replace(
        retained_snapshot,
        snapshot_id=18,
        control_tick_epoch=runtime._control_tick_epoch,
    )
    runtime._idle_replay_plan(current_snapshot, retained_plan)
    source_timestamp_ns = retained_snapshot.source.timestamp_ns
    runtime._work_budget_owner._budget_relax_factor = 2.0

    runtime.rearm_post_mapping_work()

    first_snapshot, _first_plan = runtime._latest_tick_work
    assert first_snapshot.control_tick_epoch == 6
    assert runtime._control_tick_epoch == 6
    assert first_snapshot.source.timestamp_ns == source_timestamp_ns
    assert first_snapshot.source_release_rate_ratio is None
    assert runtime._work_budget_owner._budget_relax_factor > 1.0

    # One subsequent application tick owns the next barrier. Rearm consumes
    # that barrier but must not advance it a second time.
    runtime._control_tick_epoch = 7
    exhausted[0] = False
    runtime._work_budget_owner._budget_relax_factor = 2.0
    runtime.rearm_post_mapping_work()

    second_snapshot, _second_plan = runtime._latest_tick_work
    assert second_snapshot.control_tick_epoch == 7
    assert runtime._control_tick_epoch == 7
    assert second_snapshot.source.timestamp_ns == source_timestamp_ns
    assert second_snapshot.source_release_rate_ratio == 0.0
    assert runtime._work_budget_owner._budget_relax_factor == 1.0


def test_completed_transaction_cost_reaches_the_work_budget_owner():
    """The estimator's only production feed must actually deliver evidence."""

    from eligsir.mapping.scheduler.tier_cost_model import FORWARD_BACKWARD

    source = _source(3)
    plan = _plan(source, insertion_seeds=0)
    owner = WorkBudgetOwner(device_id="test-device")
    runtime = object.__new__(ContinualMappingRuntime)
    runtime._work_budget_owner = owner
    runtime.initialized = False

    runtime._record_transaction_cost(
        plan,
        tick_started_ns=0,
        prepare_end_ns=2_000_000,
        optimization_end_ns=9_000_000,
        transaction_wall_ms=11.0,
    )

    sample = owner.tier_costs.last_sample
    assert sample is not None
    assert sample.tier is ResolutionTier.NATIVE
    assert sample.transaction_class is MappingTransactionClass.FRESH
    assert sample.prepare_ms == pytest.approx(2.0)
    assert sample.optimization_ms == pytest.approx(7.0)
    assert sample.transaction_wall_ms == pytest.approx(11.0)
    assert sample.context.bucket() == (
        "test-device",
        "g<25k",
        "c1",
        "sequential_accumulation",
        FORWARD_BACKWARD,
    )
    assert owner.tier_costs.state(
        ResolutionTier.NATIVE, context=sample.context
    ).completed_samples == 1


def test_a_non_finite_gpu_kernel_timing_does_not_refuse_the_measurement():
    """ROCm event timings can come back non-finite; wall cost still counts."""

    source = _source(3)
    plan = _plan(source, insertion_seeds=0)
    owner = WorkBudgetOwner(device_id="test-device")
    runtime = object.__new__(ContinualMappingRuntime)
    runtime._work_budget_owner = owner
    runtime.initialized = False
    runtime._latest_optimization_gpu_ms = float("nan")

    runtime._record_transaction_cost(
        plan,
        tick_started_ns=0,
        prepare_end_ns=2_000_000,
        optimization_end_ns=9_000_000,
        transaction_wall_ms=11.0,
    )

    sample = owner.tier_costs.last_sample
    assert sample is not None
    assert sample.kernel_ms is None
    assert sample.transaction_wall_ms == pytest.approx(11.0)
