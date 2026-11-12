"""Tests for issue #822 compatibility adapters (Wave-3 to canonical evidence migration).

Tests that legacy InputEvidence types from both evidence.contracts and
candidates.contracts convert correctly to canonical FrameEvidence via
adapters, with explicit availability tracking (NOT_REQUESTED for missing
signals, never zero).
"""

from __future__ import annotations

import json
from dataclasses import asdict

import pytest

from eligsir.mapping.candidates.contracts import (
    InputEvidence as CandidateInputEvidence,
)
from eligsir.mapping.evidence.adapters import (
    from_candidate_input_evidence,
    from_mapping_input_evidence,
)
from eligsir.mapping.evidence.canonical import (
    EvidenceAvailability,
    FrameEvidence,
)
from eligsir.mapping.evidence.contracts import (
    EvidenceMeta,
    InputEvidence as MappingInputEvidence,
)


class TestMappingInputEvidenceAdapter:
    """Tests for from_mapping_input_evidence() adapter."""

    def test_converts_pose_confidence_to_tracking_confidence(self):
        """pose_confidence from mapping.evidence maps to tracking_confidence."""
        legacy = MappingInputEvidence(
            pose_confidence=0.95,
            image_quality=0.8,
            depth_valid_fraction=0.7,
            geometry_reliability_mean=0.6,
        )
        evidence = from_mapping_input_evidence(legacy, source_ordinal=42)

        assert evidence.source.tracking_confidence.value == pytest.approx(0.95)
        assert evidence.source.tracking_confidence.availability == EvidenceAvailability.AVAILABLE

    def test_converts_depth_valid_fraction(self):
        """depth_valid_fraction maps directly."""
        legacy = MappingInputEvidence(
            pose_confidence=0.9,
            image_quality=0.8,
            depth_valid_fraction=0.65,
            geometry_reliability_mean=0.6,
        )
        evidence = from_mapping_input_evidence(legacy)

        assert evidence.source.depth_valid_fraction.value == pytest.approx(0.65)
        assert evidence.source.depth_valid_fraction.availability == EvidenceAvailability.AVAILABLE

    def test_missing_fields_marked_not_requested(self):
        """Fields not in legacy type are marked NOT_REQUESTED, not zero."""
        legacy = MappingInputEvidence(
            pose_confidence=0.9,
            image_quality=0.8,
            depth_valid_fraction=0.7,
            geometry_reliability_mean=0.6,
        )
        evidence = from_mapping_input_evidence(legacy)

        # Image quality metrics are not in mapping.evidence.InputEvidence
        assert evidence.source.blur_laplacian.availability == EvidenceAvailability.NOT_REQUESTED
        assert evidence.source.blur_laplacian.value is None
        assert evidence.source.exposure_luma.availability == EvidenceAvailability.NOT_REQUESTED
        assert evidence.source.white_clip_fraction.availability == EvidenceAvailability.NOT_REQUESTED
        assert evidence.source.black_clip_fraction.availability == EvidenceAvailability.NOT_REQUESTED

        # Motion signals not in legacy type
        assert evidence.source.translation_delta_m.availability == EvidenceAvailability.NOT_REQUESTED
        assert evidence.source.rotation_delta_deg.availability == EvidenceAvailability.NOT_REQUESTED
        assert evidence.source.translation_velocity_m_per_s.availability == EvidenceAvailability.NOT_REQUESTED
        assert evidence.source.rotation_velocity_deg_per_s.availability == EvidenceAvailability.NOT_REQUESTED

    def test_preserves_meta_versioning(self):
        """EvidenceMeta (map/topology versions) is threaded through."""
        meta = EvidenceMeta(
            frame_id=10,
            map_version=5,
            topology_version=3,
            resolution_bucket="high",
        )
        legacy = MappingInputEvidence(
            pose_confidence=0.9,
            image_quality=0.8,
            depth_valid_fraction=0.7,
            geometry_reliability_mean=0.6,
            meta=meta,
        )
        evidence = from_mapping_input_evidence(legacy, source_ordinal=10)

        assert evidence.identity.map_version == 5
        assert evidence.identity.topology_version == 3
        assert evidence.identity.resolution_bucket == "high"

    def test_depth_eligibility_gated_by_valid_fraction(self):
        """depth_eligible is False when depth_valid_fraction is zero."""
        legacy_no_depth = MappingInputEvidence(
            pose_confidence=0.9,
            image_quality=0.8,
            depth_valid_fraction=0.0,
            geometry_reliability_mean=0.6,
        )
        evidence = from_mapping_input_evidence(legacy_no_depth)
        assert evidence.depth_eligible is False

        legacy_with_depth = MappingInputEvidence(
            pose_confidence=0.9,
            image_quality=0.8,
            depth_valid_fraction=0.1,
            geometry_reliability_mean=0.6,
        )
        evidence = from_mapping_input_evidence(legacy_with_depth)
        assert evidence.depth_eligible is True

    def test_no_rejection_reasons_in_mapping_type(self):
        """Mapping InputEvidence has no reject_reason field."""
        legacy = MappingInputEvidence(
            pose_confidence=0.9,
            image_quality=0.8,
            depth_valid_fraction=0.7,
            geometry_reliability_mean=0.6,
        )
        evidence = from_mapping_input_evidence(legacy)

        assert evidence.hard_reject_reasons == ()
        assert evidence.soft_reject_reasons == ()


class TestCandidateInputEvidenceAdapter:
    """Tests for from_candidate_input_evidence() adapter."""

    def test_converts_image_quality_metrics(self):
        """Candidate InputEvidence quality metrics map to SourceReliabilityEvidence."""
        legacy = CandidateInputEvidence(
            blur_laplacian=42.5,
            exposure_luma=0.55,
            white_clip_ratio=0.02,
            black_clip_ratio=0.01,
            valid_depth_fraction=0.8,
            quality_weight=0.9,
        )
        evidence = from_candidate_input_evidence(legacy, source_index=5, timestamp_ns=1234567890)

        assert evidence.source.blur_laplacian.value == pytest.approx(42.5)
        assert evidence.source.exposure_luma.value == pytest.approx(0.55)
        assert evidence.source.white_clip_fraction.value == pytest.approx(0.02)
        assert evidence.source.black_clip_fraction.value == pytest.approx(0.01)
        assert evidence.source.depth_valid_fraction.value == pytest.approx(0.8)

    def test_all_converted_fields_are_available(self):
        """Converted fields are marked AVAILABLE."""
        legacy = CandidateInputEvidence(
            blur_laplacian=40.0,
            exposure_luma=0.5,
            white_clip_ratio=0.03,
            black_clip_ratio=0.02,
            valid_depth_fraction=0.75,
            quality_weight=0.85,
        )
        evidence = from_candidate_input_evidence(legacy)

        assert evidence.source.blur_laplacian.availability == EvidenceAvailability.AVAILABLE
        assert evidence.source.exposure_luma.availability == EvidenceAvailability.AVAILABLE
        assert evidence.source.white_clip_fraction.availability == EvidenceAvailability.AVAILABLE
        assert evidence.source.black_clip_fraction.availability == EvidenceAvailability.AVAILABLE
        assert evidence.source.depth_valid_fraction.availability == EvidenceAvailability.AVAILABLE

    def test_missing_fields_marked_not_requested(self):
        """Fields not in candidate.InputEvidence are marked NOT_REQUESTED."""
        legacy = CandidateInputEvidence(
            blur_laplacian=40.0,
            exposure_luma=0.5,
            white_clip_ratio=0.03,
            black_clip_ratio=0.02,
            valid_depth_fraction=0.75,
            quality_weight=0.85,
        )
        evidence = from_candidate_input_evidence(legacy)

        # Motion/tracking signals not in candidate type
        assert evidence.source.translation_delta_m.availability == EvidenceAvailability.NOT_REQUESTED
        assert evidence.source.rotation_delta_deg.availability == EvidenceAvailability.NOT_REQUESTED
        assert evidence.source.translation_velocity_m_per_s.availability == EvidenceAvailability.NOT_REQUESTED
        assert evidence.source.rotation_velocity_deg_per_s.availability == EvidenceAvailability.NOT_REQUESTED
        assert evidence.source.tracking_confidence.availability == EvidenceAvailability.NOT_REQUESTED

    def test_reject_reason_threaded_to_hard_rejects(self):
        """reject_reason field becomes hard_reject_reasons."""
        legacy = CandidateInputEvidence(
            blur_laplacian=40.0,
            exposure_luma=0.5,
            white_clip_ratio=0.03,
            black_clip_ratio=0.02,
            valid_depth_fraction=0.75,
            quality_weight=0.85,
            reject_reason="rgb_decode_failed",
        )
        evidence = from_candidate_input_evidence(legacy)

        assert evidence.hard_reject_reasons == ("rgb_decode_failed",)
        assert evidence.rgb_eligible is False

    def test_no_reject_reason_means_no_hard_rejects(self):
        """When reject_reason is None, no hard rejects."""
        legacy = CandidateInputEvidence(
            blur_laplacian=40.0,
            exposure_luma=0.5,
            white_clip_ratio=0.03,
            black_clip_ratio=0.02,
            valid_depth_fraction=0.75,
            quality_weight=0.85,
            reject_reason=None,
        )
        evidence = from_candidate_input_evidence(legacy)

        assert evidence.hard_reject_reasons == ()
        assert evidence.rgb_eligible is True

    def test_identity_captures_source_index_and_timestamp(self):
        """Identity carries source_index and timestamp_ns."""
        legacy = CandidateInputEvidence(
            blur_laplacian=40.0,
            exposure_luma=0.5,
            white_clip_ratio=0.03,
            black_clip_ratio=0.02,
            valid_depth_fraction=0.75,
            quality_weight=0.85,
        )
        evidence = from_candidate_input_evidence(
            legacy,
            source_index=99,
            timestamp_ns=9876543210,
            source_id="my-source",
        )

        assert evidence.identity.source_ordinal == 99
        assert evidence.identity.timestamp_ns == 9876543210
        assert evidence.identity.source_id == "my-source"

    def test_depth_eligibility_gated_by_valid_fraction_and_rejects(self):
        """depth_eligible is False when depth_valid_fraction is zero or hard rejects exist."""
        # No depth
        legacy_no_depth = CandidateInputEvidence(
            blur_laplacian=40.0,
            exposure_luma=0.5,
            white_clip_ratio=0.03,
            black_clip_ratio=0.02,
            valid_depth_fraction=0.0,
            quality_weight=0.85,
        )
        evidence = from_candidate_input_evidence(legacy_no_depth)
        assert evidence.depth_eligible is False

        # Has depth
        legacy_with_depth = CandidateInputEvidence(
            blur_laplacian=40.0,
            exposure_luma=0.5,
            white_clip_ratio=0.03,
            black_clip_ratio=0.02,
            valid_depth_fraction=0.1,
            quality_weight=0.85,
        )
        evidence = from_candidate_input_evidence(legacy_with_depth)
        assert evidence.depth_eligible is True

        # Has reject reason
        legacy_rejected = CandidateInputEvidence(
            blur_laplacian=40.0,
            exposure_luma=0.5,
            white_clip_ratio=0.03,
            black_clip_ratio=0.02,
            valid_depth_fraction=0.9,
            quality_weight=0.85,
            reject_reason="decode_failed",
        )
        evidence = from_candidate_input_evidence(legacy_rejected)
        assert evidence.depth_eligible is False


class TestAdapterRoundTrip:
    """Tests for serialization round-trip consistency."""

    def test_candidate_evidence_json_roundtrip(self):
        """FrameEvidence from candidate adapter serializes and deserializes consistently."""
        legacy = CandidateInputEvidence(
            blur_laplacian=42.5,
            exposure_luma=0.55,
            white_clip_ratio=0.02,
            black_clip_ratio=0.01,
            valid_depth_fraction=0.8,
            quality_weight=0.9,
        )
        evidence = from_candidate_input_evidence(legacy, source_index=3, timestamp_ns=1000)

        # Serialize to JSON-compatible dict using asdict (handles slots=True)
        source_dict = asdict(evidence.source)

        # Verify critical fields are present
        assert source_dict["blur_laplacian"]["value"] == pytest.approx(42.5)
        assert source_dict["blur_laplacian"]["availability"] == "available"
        assert source_dict["exposure_luma"]["value"] == pytest.approx(0.55)

    def test_mapping_evidence_json_roundtrip(self):
        """FrameEvidence from mapping adapter serializes consistently."""
        legacy = MappingInputEvidence(
            pose_confidence=0.95,
            image_quality=0.8,
            depth_valid_fraction=0.7,
            geometry_reliability_mean=0.6,
        )
        evidence = from_mapping_input_evidence(legacy, source_ordinal=2)

        # Verify NOT_REQUESTED fields serialize properly using asdict (handles slots=True)
        source_dict = asdict(evidence.source)

        assert source_dict["tracking_confidence"]["value"] == pytest.approx(0.95)
        assert source_dict["tracking_confidence"]["availability"] == "available"
        assert source_dict["blur_laplacian"]["value"] is None
        assert source_dict["blur_laplacian"]["availability"] == "not_requested"


class TestFieldMappingTable:
    """Table-driven tests verifying every legacy field maps correctly."""

    # Mapping table: (legacy_type, legacy_field, canonical_path, availability)
    MAPPING_ADAPTER_FIELDS = [
        ("MappingInputEvidence", "pose_confidence", "source.tracking_confidence", "AVAILABLE"),
        ("MappingInputEvidence", "depth_valid_fraction", "source.depth_valid_fraction", "AVAILABLE"),
        # Missing fields in MappingInputEvidence
        ("MappingInputEvidence", "blur_laplacian", "source.blur_laplacian", "NOT_REQUESTED"),
        ("MappingInputEvidence", "exposure_luma", "source.exposure_luma", "NOT_REQUESTED"),
        ("MappingInputEvidence", "white_clip_ratio", "source.white_clip_fraction", "NOT_REQUESTED"),
        ("MappingInputEvidence", "black_clip_ratio", "source.black_clip_fraction", "NOT_REQUESTED"),
    ]

    CANDIDATE_ADAPTER_FIELDS = [
        ("CandidateInputEvidence", "blur_laplacian", "source.blur_laplacian", "AVAILABLE"),
        ("CandidateInputEvidence", "exposure_luma", "source.exposure_luma", "AVAILABLE"),
        ("CandidateInputEvidence", "white_clip_ratio", "source.white_clip_fraction", "AVAILABLE"),
        ("CandidateInputEvidence", "black_clip_ratio", "source.black_clip_fraction", "AVAILABLE"),
        ("CandidateInputEvidence", "valid_depth_fraction", "source.depth_valid_fraction", "AVAILABLE"),
        # Missing fields in CandidateInputEvidence
        ("CandidateInputEvidence", "pose_confidence", "source.tracking_confidence", "NOT_REQUESTED"),
        ("CandidateInputEvidence", "translation_delta", "source.translation_delta_m", "NOT_REQUESTED"),
    ]

    @pytest.mark.parametrize("legacy_type,legacy_field,canonical_path,availability", CANDIDATE_ADAPTER_FIELDS)
    def test_candidate_field_mapping(self, legacy_type, legacy_field, canonical_path, availability):
        """Each candidate legacy field maps to correct canonical path with right availability."""
        legacy = CandidateInputEvidence(
            blur_laplacian=42.0,
            exposure_luma=0.5,
            white_clip_ratio=0.02,
            black_clip_ratio=0.01,
            valid_depth_fraction=0.8,
            quality_weight=0.9,
        )
        evidence = from_candidate_input_evidence(legacy)

        # Navigate canonical_path (e.g., "source.blur_laplacian")
        obj = evidence
        for attr in canonical_path.split("."):
            obj = getattr(obj, attr)

        expected_avail = EvidenceAvailability(availability.lower())
        assert obj.availability == expected_avail

    @pytest.mark.parametrize("legacy_type,legacy_field,canonical_path,availability", MAPPING_ADAPTER_FIELDS)
    def test_mapping_field_mapping(self, legacy_type, legacy_field, canonical_path, availability):
        """Each mapping legacy field maps to correct canonical path with right availability."""
        legacy = MappingInputEvidence(
            pose_confidence=0.95,
            image_quality=0.8,
            depth_valid_fraction=0.7,
            geometry_reliability_mean=0.6,
        )
        evidence = from_mapping_input_evidence(legacy)

        # Navigate canonical_path (e.g., "source.depth_valid_fraction")
        obj = evidence
        for attr in canonical_path.split("."):
            obj = getattr(obj, attr)

        expected_avail = EvidenceAvailability(availability.lower())
        assert obj.availability == expected_avail
