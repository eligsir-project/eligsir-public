"""Contract tests for deferred, cold-retained observations."""

from __future__ import annotations

from dataclasses import replace

import pytest

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
from eligsir.contracts.runtime import SourceIdentity
from eligsir.mapping.admission.keyframe_provider import KeyframeAdmissionProposal
from eligsir.mapping.admission.observation_owner import ObservationOwner
from eligsir.mapping.scheduler.backpressure import (
    BackpressureAction,
    BackpressureDecision,
)


def _source(ordinal: int) -> SourceIdentity:
    return SourceIdentity(
        source_id="camera",
        stream_epoch=0,
        source_ordinal=ordinal,
        timestamp_ns=ordinal * 1_000_000,
    )


def _snapshot(source: SourceIdentity, *, snapshot_id: int = 7) -> ControlSnapshot:
    return ControlSnapshot(
        snapshot_id=snapshot_id,
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
        map_content_version=3,
        topology_version=2,
        parameter_version=5,
        gaussian_count=0,
        optimizer_steps_completed=0,
        limits=ControlLimits(1, 0, 0, 0, 0, 0),
    )


class _SemanticProvider:
    def __init__(self, *, admitted: bool, reason: str) -> None:
        self._proposal = KeyframeAdmissionProposal(
            provider="KeyframeAdmitter",
            admitted=admitted,
            reason=reason,
        )

    def propose(self, _snapshot: ControlSnapshot) -> KeyframeAdmissionProposal:
        return self._proposal


def _backpressure_admit(snapshot: ControlSnapshot) -> BackpressureDecision:
    assert snapshot.source is not None
    return BackpressureDecision(
        frame_id=snapshot.source.source_ordinal,
        action=BackpressureAction.ADMIT,
        reason="soft_all_capacity_available",
        admit_supervision=True,
        allow_geometry_insertion=True,
    )


def test_defer_retains_pixels_without_authorizing_immediate_mapping_work():
    disposition = ObservationDisposition(
        source=_source(4),
        snapshot_id=7,
        kind=DispositionKind.SKIP_REDUNDANT,
        semantic_action=SemanticAction.DEFER,
        historical_eligibility=HistoricalEligibility.ELIGIBLE,
        observe=True,
        mapping_candidate=False,
        admit_supervision=False,
        allow_geometry_insertion=False,
        residency=ResidencyIntent.COLD,
        reason="no_motion",
        modality_eligibility=ModalityEligibility(
            rgb_supervision=True,
            rgb_quality="qualified",
            depth_supervision=True,
            depth_quality="qualified",
            depth_insertion=True,
        ),
    )

    assert disposition.cold_retention_permitted
    assert disposition.historical_promotion_eligible
    assert not disposition.materialization_permitted


def test_reject_cannot_request_cold_retention_or_historical_eligibility():
    with pytest.raises(ValueError, match="REJECT dispositions cannot be historically eligible"):
        ObservationDisposition(
            source=_source(4),
            snapshot_id=7,
            kind=DispositionKind.SKIP_QUALITY,
            semantic_action=SemanticAction.REJECT,
            historical_eligibility=HistoricalEligibility.ELIGIBLE,
            observe=True,
            mapping_candidate=False,
            admit_supervision=False,
            allow_geometry_insertion=False,
            residency=ResidencyIntent.METADATA_ONLY,
            reason="blur_reject",
        )

    with pytest.raises(ValueError, match="REJECT dispositions cannot request pixel residency"):
        ObservationDisposition(
            source=_source(4),
            snapshot_id=7,
            kind=DispositionKind.SKIP_QUALITY,
            semantic_action=SemanticAction.REJECT,
            historical_eligibility=HistoricalEligibility.NONE,
            observe=True,
            mapping_candidate=False,
            admit_supervision=False,
            allow_geometry_insertion=False,
            residency=ResidencyIntent.COLD,
            reason="blur_reject",
        )


def test_no_motion_is_deferred_to_cold_history_without_an_immediate_permit():
    source = _source(10)
    owner = ObservationOwner(
        _backpressure_admit,
        semantic_providers=(
            _SemanticProvider(admitted=False, reason="no_motion"),
        ),
    )

    disposition = owner.decide(_snapshot(source))

    assert disposition.kind is DispositionKind.SKIP_REDUNDANT
    assert disposition.semantic_action is SemanticAction.DEFER
    assert disposition.residency is ResidencyIntent.COLD
    assert disposition.historical_eligibility is HistoricalEligibility.ELIGIBLE
    assert owner.permit(disposition) is None
    cold_permit = owner.cold_retention_permit(disposition)
    assert cold_permit is not None
    assert cold_permit.source == source
    assert cold_permit.snapshot_id == 7


def test_quality_rejection_remains_metadata_only_and_cannot_be_promoted():
    source = _source(11)
    owner = ObservationOwner(
        _backpressure_admit,
        semantic_providers=(
            _SemanticProvider(admitted=False, reason="blur_quality_reject"),
        ),
    )

    disposition = owner.decide(_snapshot(source))

    assert disposition.semantic_action is SemanticAction.REJECT
    assert disposition.residency is ResidencyIntent.METADATA_ONLY
    assert disposition.historical_eligibility is HistoricalEligibility.NONE
    assert owner.cold_retention_permit(disposition) is None
    assert owner.promotion_permit(disposition, _snapshot(_source(12), snapshot_id=8)) is None


def test_unknown_semantic_rejection_fails_closed_instead_of_becoming_history():
    source = _source(12)
    owner = ObservationOwner(
        _backpressure_admit,
        semantic_providers=(
            _SemanticProvider(admitted=False, reason="invalid_pose_evidence"),
        ),
    )

    disposition = owner.decide(_snapshot(source))

    assert disposition.semantic_action is SemanticAction.REJECT
    assert disposition.kind is DispositionKind.SKIP_QUALITY
    assert disposition.residency is ResidencyIntent.METADATA_ONLY
    assert owner.cold_retention_permit(disposition) is None


def test_genuine_rejection_takes_precedence_over_a_deferred_filter():
    source = _source(13)
    owner = ObservationOwner(
        _backpressure_admit,
        semantic_providers=(
            _SemanticProvider(admitted=False, reason="no_motion"),
            _SemanticProvider(admitted=False, reason="blur_quality_reject"),
        ),
    )

    disposition = owner.decide(_snapshot(source))

    assert disposition.semantic_action is SemanticAction.REJECT
    assert disposition.kind is DispositionKind.SKIP_QUALITY
    assert disposition.residency is ResidencyIntent.METADATA_ONLY
    assert disposition.reason == "blur_quality_reject"
    assert owner.cold_retention_permit(disposition) is None


def test_deferred_promotion_is_a_separate_newer_snapshot_fenced_capability():
    source = _source(14)
    owner = ObservationOwner(
        _backpressure_admit,
        semantic_providers=(
            _SemanticProvider(admitted=False, reason="no_motion"),
        ),
    )
    disposition = owner.decide(_snapshot(source, snapshot_id=7))

    with pytest.raises(ValueError, match="newer control snapshot"):
        owner.promotion_permit(
            disposition,
            replace(_snapshot(source, snapshot_id=7), pending_sources=(), newest_frontier=None),
        )

    promotion_snapshot = replace(
        _snapshot(_source(14), snapshot_id=8),
        source=None,
        pending_sources=(),
        newest_frontier=None,
        control_version=2,
        topology_version=4,
    )
    permit = owner.promotion_permit(disposition, promotion_snapshot)

    assert permit is not None
    assert permit.source == source
    assert permit.disposition_snapshot_id == 7
    assert permit.promotion_snapshot_id == 8
    assert permit.control_version == 2
    assert permit.topology_version == 4
    assert owner.decide(_snapshot(source, snapshot_id=9)) is disposition
