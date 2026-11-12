from collections import OrderedDict
from types import SimpleNamespace
import json

import numpy as np
import pytest
import torch

from eligsir.contracts.control import (
    DepthInsertionPlan,
    FreshnessMode,
    MappingWorkPlan,
    ModalityEligibility,
    MutationBudget,
    ResolutionTier,
    SupervisionWork,
    ControlLimits,
    ControlSnapshot,
)
from eligsir.contracts.runtime import SourceIdentity
from eligsir.contracts.work_clocks import RuntimeWorkClockOwner
from eligsir.contracts.types import CameraIntrinsics
from eligsir.mapping.observation_db import ColdPixelHandle, FrameObservationDB
from eligsir.mapping.scheduler.work_budget import WorkBudgetOwner
from eligsir.runtime.post_refinement import PostRefinementPhaseOwner
from eligsir.runtime.config import DepthLossPolicy
from eligsir.runtime.production_application import CanonicalMappingRuntime
from eligsir.runtime.artifact_paths import (
    point_cloud_artifact_paths,
    publish_legacy_iteration_alias,
)


def _plan(
    source: SourceIdentity, optimizer_updates: int = 1, extra_work: bool = False
) -> MappingWorkPlan:
    modality = ModalityEligibility(rgb_supervision=True, rgb_quality="qualified")
    sources = (
        (source, SourceIdentity("smoke", 0, 1, 2)) if extra_work else (source,)
    )
    return MappingWorkPlan(
        plan_id="mapping-1",
        snapshot_id=1,
        control_version=1,
        supervision_sources=sources,
        replay_sources=(),
        resolution_tier=ResolutionTier.NATIVE,
        optimizer_updates=optimizer_updates,
        insertion=DepthInsertionPlan(
            source=None,
            enabled=False,
            sample_stride=1,
            candidate_budget=0,
            seed_budget=0,
            voxel_size_m=0.01,
            minimum_confidence=0.0,
        ),
        allow_growth=False,
        allow_relocation=False,
        mutation_budget=MutationBudget(max_total=0),
        freshness_mode=FreshnessMode.HEALTHY,
        map_content_version=0,
        topology_version=0,
        parameter_version=0,
        supervision_work=(
            *(SupervisionWork(item, ResolutionTier.NATIVE, modality) for item in sources),
        ),
        materialization_sources=(source,),
    )


def _enter_durable_refinement(runtime, tmp_path, sources, modality) -> None:
    """Give a legacy CPU seam the same typed endpoint boundary as production."""

    if not hasattr(runtime, "_work_clock_owner"):
        runtime._work_clock_owner = RuntimeWorkClockOwner()
        runtime._work_clock_owner.record_optimization(
            int(runtime.iteration), int(runtime.iteration) * 4,
            int(runtime.iteration) * 4, 0,
        )
    runtime._pending_optimization_transaction_timings = []
    runtime._observation_db = FrameObservationDB(device="cpu")
    runtime._historical_sources_by_frame_id = {}
    runtime._replay_modalities = OrderedDict()
    runtime._replay_weights = {}
    intrinsics = CameraIntrinsics(fx=1.0, fy=1.0, cx=1.0, cy=1.0, width=2, height=2)
    for ordinal, source in enumerate(sources):
        record = runtime._observation_db.record_historical_candidate(
            source_identity=source,
            pose_c2w=np.eye(4, dtype=np.float32),
            timestamp=float(ordinal),
            intrinsics=intrinsics,
            cold_pixels=ColdPixelHandle(
                source_identity=source.fingerprint(),
                rgb=np.zeros((2, 2, 3), dtype=np.uint8),
                depth=np.ones((2, 2), dtype=np.float32),
                intrinsics=intrinsics,
            ),
            modality_eligibility=modality,
            disposition_snapshot_id=ordinal,
        )
        runtime._register_historical_record(record, source)
        runtime._replay_modalities[source] = modality
        runtime._replay_weights[source] = 1.0
    runtime._control_tick_epoch = getattr(runtime, "_control_tick_epoch", 0)
    runtime._post_refinement_phase_owner = PostRefinementPhaseOwner()
    # This deliberately constructed CPU seam exercises post-mapping rearming,
    # not the full telemetry publisher.  Mark the source boundary directly so
    # the test does not accidentally depend on unrelated GPU-event fields that
    # normal runtime construction installs.
    runtime._source_complete = True
    runtime.mark_source_complete = lambda: None
    runtime._post_refinement_phase_owner.mark_source_complete()
    point_cloud = tmp_path / "point_cloud.ply"
    checkpoint = tmp_path / "checkpoint.pth"
    point_cloud.write_bytes(b"ply\n")
    checkpoint.write_bytes(b"checkpoint")
    runtime.enter_post_mapping_refinement(
        {"point_cloud": str(point_cloud), "checkpoint": str(checkpoint)}
    )


def test_refinement_loss_context_overrides_only_tail_ticks_and_restores_on_error():
    """The causal owner scopes the optional tail weight to one optimizer tick."""

    runtime = CanonicalMappingRuntime.__new__(CanonicalMappingRuntime)
    runtime._optimization_policy = SimpleNamespace(
        streaming_refinement_depth_loss_weight=0.2
    )
    runtime._depth_loss_policy = DepthLossPolicy(enabled=False, weight=0.0)
    runtime._optimization_runtime = SimpleNamespace(
        loss_assembly_kwargs={"depth_loss_weight": 0.0},
        fixed_shape_mcmc=True,
    )
    loss_kwargs = runtime._optimization_runtime.loss_assembly_kwargs

    for _ in range(2):
        with runtime.post_mapping_refinement_loss_context():
            assert loss_kwargs["depth_loss_weight"] == 0.2
            assert runtime._depth_loss_policy == DepthLossPolicy(enabled=True, weight=0.2)
            assert runtime._optimization_runtime.fixed_shape_mcmc is True
        assert loss_kwargs["depth_loss_weight"] == 0.0
        assert runtime._depth_loss_policy == DepthLossPolicy(enabled=False, weight=0.0)

    with pytest.raises(RuntimeError, match="tail failure"):
        with runtime.post_mapping_refinement_loss_context():
            raise RuntimeError("tail failure")
    assert loss_kwargs["depth_loss_weight"] == 0.0
    assert runtime._depth_loss_policy == DepthLossPolicy(enabled=False, weight=0.0)

    runtime._optimization_policy.streaming_refinement_depth_loss_weight = -1.0
    with runtime.post_mapping_refinement_loss_context():
        assert loss_kwargs["depth_loss_weight"] == 0.0
        assert runtime._depth_loss_policy == DepthLossPolicy(enabled=False, weight=0.0)


@pytest.mark.parametrize("invalid", (-2.0, float("nan"), float("inf"), float("-inf")))
def test_refinement_loss_context_rejects_invalid_override(invalid):
    """Only the -1 sentinel or a finite non-negative tail weight is valid."""

    runtime = CanonicalMappingRuntime.__new__(CanonicalMappingRuntime)
    runtime._optimization_policy = SimpleNamespace(
        streaming_refinement_depth_loss_weight=invalid
    )

    with pytest.raises(ValueError, match="refinement depth-loss weight"):
        with runtime.post_mapping_refinement_loss_context():
            pass


def test_rearm_uses_explicit_optimizer_permission_beyond_mapping_budget(tmp_path):
    source = SourceIdentity("smoke", 0, 0, 1)
    snapshot = ControlSnapshot(
        snapshot_id=1,
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
        map_content_version=0,
        topology_version=0,
        parameter_version=0,
        gaussian_count=1,
        optimizer_steps_completed=10,
        supervision_views_completed=37,
        limits=ControlLimits(1, 0, 1, 0, 0, 0),
        supervision_candidates=(source,),
    )
    runtime = CanonicalMappingRuntime.__new__(CanonicalMappingRuntime)
    runtime._last_admitted_control = (snapshot, (), None, _plan(source))
    runtime._control_snapshot_id = 2
    runtime._batch_size = 1
    runtime.iteration = 10
    runtime.optimizer_steps = 10
    runtime._work_clock_owner = RuntimeWorkClockOwner()
    runtime._work_clock_owner.record_optimization(10, 37, 37, 0)
    runtime.opt = SimpleNamespace(iterations=10)
    runtime._optimization_runtime = SimpleNamespace()
    runtime._versions = lambda: (0, 0, 0)
    _enter_durable_refinement(
        runtime,
        tmp_path,
        (source,),
        _plan(source).supervision_work[0].modality_eligibility,
    )

    runtime.rearm_post_mapping_work()

    assert runtime._latest_tick_work[1].optimizer_updates == 1
    assert runtime._latest_tick_work[0].supervision_views_completed == 37
    assert not hasattr(runtime._optimization_runtime, "optimizer_update_permitted")

    runtime.config = object()
    with pytest.raises(RuntimeError, match="supervision work clock"):
        runtime.rearm_post_mapping_work()


def test_rearm_refreshes_true_camera_batch_from_full_replay_cache(tmp_path):
    current = SourceIdentity("smoke", 0, 0, 1)
    replay = SourceIdentity("smoke", 0, 1, 2)
    modality = ModalityEligibility(rgb_supervision=True, rgb_quality="qualified")
    snapshot = ControlSnapshot(
        snapshot_id=1, control_version=1, source=current, tracking=None,
        pending_sources=(current,), newest_frontier=current,
        freshness_mode=FreshnessMode.HEALTHY, pending_oldest_age_ns=0,
        ingress_queue_depth=0, candidate_queue_depth=0, transfer_queue_depth=0,
        gpu_queue_depth=0, map_content_version=0, topology_version=0,
        parameter_version=0, gaussian_count=1, optimizer_steps_completed=10,
        limits=ControlLimits(1, 1, 1, 0, 0, 0),
        supervision_candidates=(current,), replay_candidates=(replay,),
        replay_work_modalities=(modality,),
    )
    previous_plan = MappingWorkPlan(
        plan_id="mapping-c2", snapshot_id=1, control_version=1,
        supervision_sources=(current,), replay_sources=(replay,),
        resolution_tier=ResolutionTier.NATIVE, optimizer_updates=1,
        insertion=DepthInsertionPlan(
            source=None, enabled=False, sample_stride=1, candidate_budget=0,
            seed_budget=0, voxel_size_m=0.01, minimum_confidence=0.0,
        ),
        allow_growth=False, allow_relocation=False,
        mutation_budget=MutationBudget(max_total=0),
        freshness_mode=FreshnessMode.HEALTHY, map_content_version=0,
        topology_version=0, parameter_version=0,
        supervision_work=(SupervisionWork(current, ResolutionTier.NATIVE, modality),),
        replay_work=(SupervisionWork(replay, ResolutionTier.NATIVE, modality),),
        replay_selected_weights=(2.5,), replay_population_size=2,
        requested_supervision_slots=1, requested_replay_slots=1,
    )
    runtime = CanonicalMappingRuntime.__new__(CanonicalMappingRuntime)
    runtime._last_admitted_control = (snapshot, (), None, previous_plan)
    runtime._last_materialized_work = (snapshot, previous_plan)
    runtime._control_snapshot_id = 2
    runtime._batch_size = 2
    runtime.iteration = 10
    runtime.optimizer_steps = 10
    runtime._work_clock_owner = RuntimeWorkClockOwner()
    runtime._work_clock_owner.record_optimization(10, 40, 40, 0)
    runtime.opt = SimpleNamespace(iterations=10)
    runtime._optimization_runtime = SimpleNamespace(optimizer_iteration_limit=None)
    runtime._versions = lambda: (0, 0, 0)
    runtime._work_budget_owner = WorkBudgetOwner(
        requested_batch_size=2,
        maximum_camera_count=2,
    )
    replay_sources = tuple(
        SourceIdentity("smoke", 0, index, index + 1) for index in range(4)
    )
    runtime._replay_modalities = OrderedDict(
        (source, modality) for source in replay_sources
    )
    runtime._replay_weights = {
        source: float(index + 1) for index, source in enumerate(replay_sources)
    }
    _enter_durable_refinement(runtime, tmp_path, replay_sources, modality)

    runtime.rearm_post_mapping_work(optimizer_updates=1)

    first = runtime._latest_tick_work[1]
    assert first.supervision_sources == ()
    assert len(first.replay_sources) == 2
    assert set(first.replay_sources).issubset(replay_sources)
    assert len(first.replay_selected_weights) == 2
    assert "source_idle_replay_only" in first.reason_trace

    runtime.optimizer_steps = 11
    runtime.rearm_post_mapping_work(optimizer_updates=1)
    second = runtime._latest_tick_work[1]

    assert second.supervision_sources == ()
    assert len(second.replay_sources) == 2
    assert set(second.replay_sources).issubset(replay_sources)
    assert len(second.replay_selected_weights) == 2
    assert second.replay_sources != first.replay_sources


def test_refinement_realization_caps_last_tick_to_remaining_budget(tmp_path):
    source = SourceIdentity("smoke", 0, 0, 1)
    snapshot = ControlSnapshot(
        snapshot_id=1, control_version=1, source=source, tracking=None,
        pending_sources=(source,), newest_frontier=source,
        freshness_mode=FreshnessMode.HEALTHY, pending_oldest_age_ns=0,
        ingress_queue_depth=0, candidate_queue_depth=0, transfer_queue_depth=0,
        gpu_queue_depth=0, map_content_version=0, topology_version=0,
        parameter_version=0, gaussian_count=1, optimizer_steps_completed=10,
        limits=ControlLimits(1, 0, 2, 0, 0, 0),
        supervision_candidates=(source,),
    )
    runtime = CanonicalMappingRuntime.__new__(CanonicalMappingRuntime)
    runtime._last_admitted_control = (
        snapshot, (), None, _plan(source, optimizer_updates=2, extra_work=True)
    )
    runtime._control_snapshot_id = 2
    runtime._batch_size = 1
    runtime.iteration = 10
    runtime.optimizer_steps = 10
    runtime.opt = SimpleNamespace(iterations=10)
    runtime._optimization_runtime = SimpleNamespace(optimizer_iteration_limit=None)
    runtime._versions = lambda: (0, 0, 0)
    _enter_durable_refinement(
        runtime,
        tmp_path,
        (source,),
        _plan(source).supervision_work[0].modality_eligibility,
    )
    runtime.backend = SimpleNamespace(
        checkpoint_state=lambda: SimpleNamespace(
            payload={"weights": "test"},
            layout="gsplat",
            map_version=0,
            topology_version=0,
            coordinate_frame_version=0,
            schema_version=1,
        )
    )

    class Application:
        def __init__(self):
            self.realized = []
            self.scheduled = []

        def flush(self):
            work = runtime._work_clock_owner.snapshot()
            paths = point_cloud_artifact_paths(
                tmp_path,
                optimizer_updates=int(work.optimizer_updates),
                supervised_views=int(work.supervised_rgb_views),
            )
            paths.directory.mkdir(parents=True, exist_ok=True)
            paths.point_cloud.write_bytes(b"ply")
            publish_legacy_iteration_alias(paths)

        def tick(self):
            from eligsir.runtime.config_boundary import scheduled_b1_work

            plan = runtime._latest_tick_work[1]
            self.scheduled.append(scheduled_b1_work(plan))
            updates = plan.optimizer_updates
            self.realized.append(updates)
            runtime.iteration += updates
            runtime.optimizer_steps += updates
            runtime._work_clock_owner.record_optimization(
                updates, updates * 4, updates * 4, 0
            )
            return SimpleNamespace(optimizer_steps=updates)

    initial_paths = point_cloud_artifact_paths(
        tmp_path, optimizer_updates=10, supervised_views=40
    )
    initial_paths.directory.mkdir(parents=True)
    initial_paths.point_cloud.write_bytes(b"ply")
    publish_legacy_iteration_alias(initial_paths)
    application = Application()

    from eligsir.runtime.causal_entry import run_causal_post_mapping_refinement

    metadata = run_causal_post_mapping_refinement(
        application, runtime, source_frame_count=20, requested_steps=3,
        model_path=tmp_path, mapping_wall_time_s=1.0,
    )

    assert application.realized == [2, 1]
    assert [len(work) for work in application.scheduled] == [2, 1]
    assert metadata["refinement_optimizer_steps"] == 3
    assert json.loads((tmp_path / "streaming_phase_metadata.json").read_text())[
        "refinement_optimizer_steps"
    ] == 3
