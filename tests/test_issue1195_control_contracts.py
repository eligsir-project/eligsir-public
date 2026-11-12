from dataclasses import FrozenInstanceError, fields

import numpy as np
import pytest

from eligsir.contracts.control import (
    CompactGpuTelemetry,
    ControlLimits,
    ControlProposal,
    ControlSnapshot,
    DepthInsertionPlan,
    DispositionKind,
    EvidenceFact,
    FreshnessMode,
    MappingWorkPlan,
    MutationBudget,
    ModalityEligibility,
    ObservationDisposition,
    ResidencyIntent,
    ResolutionTier,
    SupervisionWork,
)
from eligsir.contracts.runtime import SourceIdentity, TrackingObservation, TrackingStatus
from eligsir.contracts.types import Pose


def _source(ordinal: int) -> SourceIdentity:
    return SourceIdentity(
        source_id="camera",
        stream_epoch=2,
        source_ordinal=ordinal,
        timestamp_ns=ordinal * 1_000_000,
    )


def _disabled_insertion() -> DepthInsertionPlan:
    return DepthInsertionPlan(
        source=None,
        enabled=False,
        sample_stride=1,
        candidate_budget=0,
        seed_budget=0,
        voxel_size_m=0.02,
        minimum_confidence=0.0,
    )


def _limits() -> ControlLimits:
    return ControlLimits(
        max_supervision_slots=2,
        max_replay_slots=1,
        max_optimizer_updates=3,
        max_insertion_seeds=64,
        max_mutations=16,
        max_transfer_bytes=1_000_000,
    )


def _snapshot() -> ControlSnapshot:
    pending = (_source(7), _source(8), _source(9))
    return ControlSnapshot(
        snapshot_id=11,
        control_version=1,
        source=pending[-1],
        tracking=None,
        pending_sources=pending,
        newest_frontier=pending[-1],
        freshness_mode=FreshnessMode.CATCH_UP,
        pending_oldest_age_ns=20_000_000,
        ingress_queue_depth=1,
        candidate_queue_depth=3,
        transfer_queue_depth=0,
        gpu_queue_depth=1,
        map_content_version=17,
        topology_version=4,
        parameter_version=23,
        gaussian_count=500,
        optimizer_steps_completed=41,
        limits=_limits(),
        supervision_candidates=(pending[-1], pending[-2]),
        replay_candidates=(pending[0],),
        replay_work_modalities=(
            ModalityEligibility(True, "qualified", False, "missing", False),
        ),
        latest_gpu_telemetry=CompactGpuTelemetry(
            transaction_id=10,
            map_content_version=17,
            topology_version=4,
            parameter_version=23,
            facts=(EvidenceFact("gpu_ms", 3.5),),
        ),
        evidence=(EvidenceFact("valid_depth_fraction", 0.8),),
    )


def test_control_snapshot_is_frozen_scalar_metadata_only():
    snapshot = _snapshot()

    with pytest.raises(FrozenInstanceError):
        snapshot.snapshot_id = 12
    with pytest.raises(AttributeError):
        snapshot.evidence.append(EvidenceFact("bad", 1))

    forbidden = {"packet", "frame", "rgb", "depth", "buffer", "tensor", "pixels"}
    assert not forbidden.intersection(field.name for field in fields(ControlSnapshot))
    assert "current_disposition" not in {field.name for field in fields(ControlSnapshot)}
    assert "proposals" not in {field.name for field in fields(ControlSnapshot)}


def test_control_snapshot_owns_immutable_pose_evidence():
    snapshot = _snapshot()
    original_matrix = np.eye(4, dtype=np.float32)
    tracking = TrackingObservation(
        source=snapshot.source,
        pose=Pose(original_matrix),
        status=TrackingStatus.VALID,
        provider="tracker",
    )
    snapshot = ControlSnapshot(
        **{
            **{field.name: getattr(snapshot, field.name) for field in fields(snapshot)},
            "tracking": tracking,
        }
    )

    original_matrix[0, 3] = 42.0
    tracking.pose.matrix[1, 3] = 23.0

    assert snapshot.tracking is not tracking
    assert snapshot.tracking.pose.matrix[0, 3] == 0.0
    assert snapshot.tracking.pose.matrix[1, 3] == 0.0
    with pytest.raises(ValueError, match="read-only"):
        snapshot.tracking.pose.matrix[2, 3] = 7.0
    with pytest.raises(FrozenInstanceError):
        snapshot.tracking.pose.matrix = np.eye(4, dtype=np.float32)


def test_control_snapshot_recursively_owns_tracking_diagnostics():
    snapshot = _snapshot()
    nested_array = np.array([1.0, 2.0])
    nested_list = [{"samples": nested_array}]
    tracking = TrackingObservation(
        source=snapshot.source,
        pose=Pose.identity(),
        status=TrackingStatus.VALID,
        provider="tracker",
        diagnostics={"nested": nested_list},
    )
    snapshot = ControlSnapshot(
        **{
            **{field.name: getattr(snapshot, field.name) for field in fields(snapshot)},
            "tracking": tracking,
        }
    )

    nested_array[0] = 99.0
    nested_list[0]["samples"] = np.array([88.0])

    frozen = snapshot.tracking.diagnostics["nested"][0]["samples"]
    assert frozen.tolist() == [1.0, 2.0]
    with pytest.raises(ValueError, match="read-only"):
        frozen[0] = 3.0
    with pytest.raises(TypeError):
        snapshot.tracking.diagnostics["nested"][0]["samples"] = frozen


def test_control_snapshot_requires_full_ordered_frontier_identity():
    snapshot = _snapshot()
    with pytest.raises(ValueError, match="newest_frontier"):
        ControlSnapshot(
            **{
                **{field.name: getattr(snapshot, field.name) for field in fields(snapshot)},
                "newest_frontier": _source(8),
            }
        )


def test_control_snapshot_carries_bounded_identity_candidates_not_frames():
    snapshot = _snapshot()
    assert snapshot.supervision_candidates == (_source(9), _source(8))
    assert snapshot.replay_candidates == (_source(7),)
    with pytest.raises(ValueError, match="replay_candidates must be unique"):
        ControlSnapshot(
            **{
                **{field.name: getattr(snapshot, field.name) for field in fields(snapshot)},
                "replay_candidates": (_source(7), _source(7)),
            }
        )


@pytest.mark.parametrize(
    "kind",
    [
        DispositionKind.METADATA_ONLY,
        DispositionKind.COALESCE,
        DispositionKind.SKIP_REDUNDANT,
        DispositionKind.SKIP_CAPACITY,
        DispositionKind.SKIP_QUALITY,
    ],
)
def test_declined_disposition_cannot_materialize_or_insert(kind):
    kwargs = {}
    if kind is DispositionKind.COALESCE:
        kwargs["representative"] = _source(10)
    disposition = ObservationDisposition(
        source=_source(9),
        snapshot_id=11,
        kind=kind,
        observe=True,
        mapping_candidate=False,
        admit_supervision=False,
        allow_geometry_insertion=False,
        residency=ResidencyIntent.METADATA_ONLY,
        reason=kind.value,
        **kwargs,
    )

    assert disposition.terminal
    assert not disposition.materialization_permitted


def test_only_admit_disposition_can_issue_materialization_permission():
    with pytest.raises(ValueError, match="only ADMIT"):
        ObservationDisposition(
            source=_source(9),
            snapshot_id=11,
            kind=DispositionKind.SKIP_CAPACITY,
            observe=True,
            mapping_candidate=True,
            admit_supervision=True,
            allow_geometry_insertion=False,
            residency=ResidencyIntent.HOT,
            reason="invalid resurrection",
            modality_eligibility=ModalityEligibility(
                rgb_supervision=True,
                rgb_quality="qualified",
                depth_supervision=False,
                depth_quality="missing",
                depth_insertion=False,
            ),
        )


def test_coalesce_requires_a_distinct_representative():
    with pytest.raises(ValueError, match="distinct representative"):
        ObservationDisposition(
            source=_source(9),
            snapshot_id=11,
            kind=DispositionKind.COALESCE,
            observe=True,
            mapping_candidate=False,
            admit_supervision=False,
            allow_geometry_insertion=False,
            residency=ResidencyIntent.METADATA_ONLY,
            reason="coalesced",
            representative=_source(9),
        )


def test_mapping_work_plan_is_identity_bearing_and_version_fenced():
    snapshot = _snapshot()
    plan = MappingWorkPlan(
        plan_id="snapshot-11-v1",
        snapshot_id=snapshot.snapshot_id,
        control_version=snapshot.control_version,
        supervision_sources=(snapshot.newest_frontier,),
        replay_sources=(),
        supervision_work=(
            SupervisionWork(
                source=snapshot.newest_frontier,
                resolution_tier=ResolutionTier.HALF,
                modality_eligibility=ModalityEligibility(
                    rgb_supervision=True,
                    rgb_quality="qualified",
                    depth_supervision=False,
                    depth_quality="missing",
                    depth_insertion=False,
                ),
            ),
        ),
        resolution_tier="half",
        optimizer_updates=1,
        insertion=_disabled_insertion(),
        allow_growth=False,
        allow_relocation=False,
        mutation_budget=MutationBudget(max_total=0),
        freshness_mode=FreshnessMode.CRITICAL,
        map_content_version=snapshot.map_content_version,
        topology_version=snapshot.topology_version,
        parameter_version=snapshot.parameter_version,
        reason_trace=("critical_newest_frontier",),
    )

    assert plan.supervision_sources == (_source(9),)
    assert plan.replay_sources == ()
    assert not plan.supervision_work[0].modality_eligibility.depth_supervision
    with pytest.raises(FrozenInstanceError):
        plan.optimizer_updates = 2


def test_work_plan_rejects_duplicate_or_overlapping_identities():
    common = dict(
        plan_id="p",
        snapshot_id=1,
        control_version=1,
        resolution_tier="full",
        optimizer_updates=1,
        insertion=_disabled_insertion(),
        allow_growth=False,
        allow_relocation=False,
        mutation_budget=MutationBudget(max_total=0),
        freshness_mode=FreshnessMode.HEALTHY,
        map_content_version=1,
        topology_version=1,
        parameter_version=1,
        supervision_work=(),
    )
    with pytest.raises(ValueError, match="unique"):
        MappingWorkPlan(
            supervision_sources=(_source(1), _source(1)), replay_sources=(), **common
        )
    with pytest.raises(ValueError, match="both supervision and replay"):
        MappingWorkPlan(
            supervision_sources=(_source(1),), replay_sources=(_source(1),), **common
        )


@pytest.mark.parametrize(
    ("field", "value", "error"),
    (
        ("freshness_mode", "critical", "FreshnessMode"),
        ("allow_growth", 1, "allow_growth"),
        ("allow_relocation", 0, "allow_relocation"),
    ),
)
def test_mapping_work_plan_rejects_noncanonical_mode_and_mutation_flags(field, value, error):
    common = dict(
        plan_id="p",
        snapshot_id=1,
        control_version=1,
        supervision_sources=(_source(1),),
        replay_sources=(),
        resolution_tier="full",
        optimizer_updates=1,
        insertion=_disabled_insertion(),
        allow_growth=False,
        allow_relocation=False,
        mutation_budget=MutationBudget(max_total=0),
        freshness_mode=FreshnessMode.HEALTHY,
        map_content_version=1,
        topology_version=1,
        parameter_version=1,
        supervision_work=(
            SupervisionWork(
                _source(1),
                "native",
                ModalityEligibility(True, "qualified", False, "missing", False),
            ),
        ),
    )
    common[field] = value

    with pytest.raises(TypeError, match=error):
        MappingWorkPlan(**common)


@pytest.mark.parametrize(
    ("field", "value", "error"),
    (
        ("plan_id", 7, "plan_id"),
        ("resolution_tier", True, "resolution_tier"),
        ("reason_trace", "ok", "reason_trace"),
    ),
)
def test_mapping_work_plan_rejects_noncanonical_text_fields(field, value, error):
    common = dict(
        plan_id="p",
        snapshot_id=1,
        control_version=1,
        supervision_sources=(_source(1),),
        replay_sources=(),
        resolution_tier="full",
        optimizer_updates=1,
        insertion=_disabled_insertion(),
        allow_growth=False,
        allow_relocation=False,
        mutation_budget=MutationBudget(max_total=0),
        freshness_mode=FreshnessMode.HEALTHY,
        map_content_version=1,
        topology_version=1,
        parameter_version=1,
        supervision_work=(
            SupervisionWork(
                _source(1),
                "native",
                ModalityEligibility(True, "qualified", False, "missing", False),
            ),
        ),
        reason_trace=("ok",),
    )
    common[field] = value

    with pytest.raises(TypeError, match=error):
        MappingWorkPlan(**common)
