from pathlib import Path

import numpy as np
import pytest
import torch

from eligsir.contracts.control import (
    ControlLimits,
    ControlSnapshot,
    DepthInsertionPlan,
    FreshnessMode,
    MappingWorkPlan,
    ModalityEligibility,
    MutationBudget,
    ResolutionTier,
    SupervisionWork,
)
from eligsir.contracts.runtime import SourceIdentity
from eligsir.contracts.types import CameraIntrinsics
from eligsir.mapping.observation_db import BufferStage, ColdPixelHandle, FrameRecord
from eligsir.runtime.post_refinement import (
    PostRefinementPhase,
    PostRefinementPhaseOwner,
)
from eligsir.runtime.production_application import (
    CanonicalMappingRuntime,
    create_production_mapping_application,
)


def _endpoint(tmp_path: Path) -> dict[str, str]:
    point_cloud = tmp_path / "mapping_endpoint" / "point_cloud.ply"
    checkpoint = tmp_path / "mapping_endpoint" / "chkpnt4.pth"
    point_cloud.parent.mkdir(parents=True, exist_ok=True)
    point_cloud.write_bytes(b"ply\n")
    checkpoint.write_bytes(b"checkpoint")
    return {"point_cloud": str(point_cloud), "checkpoint": str(checkpoint)}


def test_post_refinement_phase_rejects_premature_and_undurable_endpoint_transition(tmp_path):
    """A mapping endpoint must exist before EOF can authorize refinement."""

    owner = PostRefinementPhaseOwner()

    with pytest.raises(RuntimeError, match="source completion"):
        owner.enter_post_refinement(_endpoint(tmp_path))

    owner.mark_source_complete()
    with pytest.raises(RuntimeError, match="durable"):
        owner.enter_post_refinement({"point_cloud": str(tmp_path / "missing.ply")})

    owner.enter_post_refinement(_endpoint(tmp_path))

    assert owner.phase is PostRefinementPhase.REFINEMENT


def test_post_refinement_phase_rejects_corrupt_untyped_phase_authority():
    """A truthy string or integer must not grant the refinement phase."""

    owner = PostRefinementPhaseOwner()
    owner._phase = "refinement"  # type: ignore[assignment]

    with pytest.raises(TypeError, match="exact PostRefinementPhase"):
        _ = owner.phase


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="requires CUDA/ROCm production runtime",
)
def test_production_runtime_post_refinement_empty_executable_catalog_never_falls_back_to_active_work(
    tmp_path,
):
    """Bug: zero executable history must not resurrect an unavailable active camera."""

    application = create_production_mapping_application(model_path=str(tmp_path / "empty"))
    runtime = application._options.mapper_sink.__self__
    assert isinstance(runtime, CanonicalMappingRuntime)
    source = SourceIdentity("camera", 0, 1, 2)
    modality = ModalityEligibility(rgb_supervision=True, rgb_quality="qualified")
    unavailable = FrameRecord(
        frame_id=1,
        pose_c2w=np.eye(4, dtype=np.float32),
        timestamp=1.0,
        submap_id=0,
        buffer_stage=BufferStage.EVICTED_REDUNDANT,
        visible_gaussian_uids=torch.zeros(0, dtype=torch.int64),
        source_identity=source.fingerprint(),
        source_ordinal=source.source_ordinal,
        historical_eligible=True,
        modality_eligibility=modality,
    )
    try:
        runtime._observation_db.add_frame(unavailable)
        runtime._register_historical_record(unavailable, source)
        runtime._replay_modalities[source] = modality
        runtime._replay_weights[source] = 1.0
        snapshot = ControlSnapshot(
            snapshot_id=1,
            control_version=1,
            source=source,
            tracking=None,
            pending_sources=(source,),
            newest_frontier=source,
            freshness_mode=FreshnessMode.CATCH_UP,
            pending_oldest_age_ns=1,
            ingress_queue_depth=1,
            candidate_queue_depth=1,
            transfer_queue_depth=1,
            gpu_queue_depth=1,
            map_content_version=0,
            topology_version=0,
            parameter_version=0,
            gaussian_count=1,
            optimizer_steps_completed=0,
            limits=ControlLimits(4, 4, 1, 0, 0, 0),
        )
        active_plan = MappingWorkPlan(
            plan_id="unavailable-active",
            snapshot_id=1,
            control_version=1,
            supervision_sources=(source,),
            replay_sources=(),
            resolution_tier=ResolutionTier.QUARTER,
            optimizer_updates=1,
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
            freshness_mode=FreshnessMode.CATCH_UP,
            map_content_version=0,
            topology_version=0,
            parameter_version=0,
            supervision_work=(
                SupervisionWork(source, ResolutionTier.QUARTER, modality),
            ),
        )
        runtime._last_materialized_work = (snapshot, active_plan)

        runtime.mark_source_complete()
        runtime.enter_post_mapping_refinement(_endpoint(tmp_path))
        runtime.rearm_post_mapping_work()
        _rearm_snapshot, rearm_plan = runtime._latest_tick_work
        tick = runtime.tick()

        assert runtime._replay_mapping_frame(source) is None
        assert rearm_plan.supervision_sources == ()
        assert rearm_plan.replay_sources == ()
        assert rearm_plan.replay_work == ()
        assert tick.work_plan is not None
        assert tick.work_plan.replay_sources == ()
    finally:
        application.close()


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="requires CUDA/ROCm production runtime",
)
@pytest.mark.parametrize("policy", ("fixed", "adaptive"))
def test_production_runtime_post_refinement_materializes_b256_from_full_retained_catalog(
    tmp_path, policy
):
    """Bug: a 128-entry live shortlist or evicted handle must not plan B256 work."""

    application = create_production_mapping_application(
        model_path=str(tmp_path / policy),
        overrides={
            "streaming_optimization_batch_size": 256,
            "streaming_optimization_camera_count_max": 64,
            "streaming_optimization_constellation_policy": policy,
            "streaming_mapping_resolution_tier": "quarter",
            "streaming_live_cold_payload_max_frames": 300,
            "streaming_live_cold_payload_max_bytes": 10_000_000,
        },
    )
    runtime = application._options.mapper_sink.__self__
    assert isinstance(runtime, CanonicalMappingRuntime)
    modality = ModalityEligibility(rgb_supervision=True, rgb_quality="qualified")
    intrinsics = CameraIntrinsics(fx=3.0, fy=2.0, cx=1.5, cy=1.0, width=3, height=2)
    sources = []
    try:
        for ordinal in range(257):
            source = SourceIdentity("camera", 0, ordinal, ordinal + 1)
            record = runtime._observation_db.record_historical_candidate(
                source_identity=source,
                pose_c2w=np.eye(4, dtype=np.float32),
                timestamp=float(ordinal),
                intrinsics=intrinsics,
                cold_pixels=ColdPixelHandle(
                    source_identity=source.fingerprint(),
                    rgb=np.full((2, 3, 3), ordinal % 255, dtype=np.uint8),
                    depth=np.full((2, 3), 1.0, dtype=np.float32),
                    intrinsics=intrinsics,
                ),
                modality_eligibility=modality,
                disposition_snapshot_id=ordinal,
            )
            runtime._register_historical_record(record, source)
            runtime._replay_modalities[source] = modality
            runtime._replay_weights[source] = 1.0
            sources.append(source)
        unavailable = SourceIdentity("camera", 0, 257, 258)
        unavailable_record = FrameRecord(
            frame_id=257,
            pose_c2w=np.eye(4, dtype=np.float32),
            timestamp=257.0,
            submap_id=0,
            buffer_stage=BufferStage.EVICTED_REDUNDANT,
            visible_gaussian_uids=torch.zeros(0, dtype=torch.int64),
            source_identity=unavailable.fingerprint(),
            source_ordinal=257,
            historical_eligible=True,
            modality_eligibility=modality,
        )
        runtime._observation_db.add_frame(unavailable_record)
        runtime._register_historical_record(unavailable_record, unavailable)
        runtime._replay_modalities[unavailable] = modality
        runtime._replay_weights[unavailable] = 1_000_000_000.0

        snapshot = ControlSnapshot(
            snapshot_id=1,
            control_version=1,
            source=sources[-1],
            tracking=None,
            pending_sources=(),
            newest_frontier=None,
            freshness_mode=FreshnessMode.CATCH_UP,
            pending_oldest_age_ns=1,
            ingress_queue_depth=1,
            candidate_queue_depth=1,
            transfer_queue_depth=1,
            gpu_queue_depth=1,
            map_content_version=0,
            topology_version=0,
            parameter_version=0,
            gaussian_count=1,
            optimizer_steps_completed=0,
            limits=ControlLimits(256, 256, 1, 0, 0, 0),
        )
        active_plan = MappingWorkPlan(
            plan_id="mapping",
            snapshot_id=1,
            control_version=1,
            supervision_sources=(sources[0],),
            replay_sources=(),
            resolution_tier=ResolutionTier.QUARTER,
            optimizer_updates=1,
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
            freshness_mode=FreshnessMode.CATCH_UP,
            map_content_version=0,
            topology_version=0,
            parameter_version=0,
            supervision_work=(
                SupervisionWork(sources[0], ResolutionTier.QUARTER, modality),
            ),
        )
        runtime._last_materialized_work = (snapshot, active_plan)

        runtime.mark_source_complete()
        runtime.enter_post_mapping_refinement(_endpoint(tmp_path))
        runtime.rearm_post_mapping_work()
        first_snapshot, first_plan = runtime._latest_tick_work
        first_tick = runtime.tick()
        runtime.rearm_post_mapping_work()
        second_snapshot, second_plan = runtime._latest_tick_work
        second_tick = runtime.tick()

        for plan in (first_plan, second_plan):
            assert len(plan.replay_work) == 256
            assert len({work.source for work in plan.replay_work}) == 256
            assert max(work.source.source_ordinal for work in plan.replay_work) > 128
            assert unavailable not in {work.source for work in plan.replay_work}
            assert plan.resolution_tier is ResolutionTier.NATIVE
            assert {work.resolution_tier for work in plan.replay_work} == {ResolutionTier.NATIVE}
            assert plan.supervision_constellation.realized_camera_count == 64
            assert "post_refinement_constellation_phase_resolved" in plan.reason_trace
            for work in plan.replay_work:
                replay = runtime._replay_mapping_frame(work.source)
                assert replay is not None
                assert replay.source == work.source
        assert first_snapshot.freshness_mode is FreshnessMode.CATCH_UP
        assert second_snapshot.control_tick_epoch > first_snapshot.control_tick_epoch
        assert first_tick.reason == second_tick.reason == "bootstrap_pending"
    finally:
        application.close()
