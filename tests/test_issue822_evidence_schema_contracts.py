"""Round-trip, missing≠zero, and schema-version contracts for canonical evidence schema (#822)."""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

import pytest

from eligsir.mapping.evidence.canonical import (
    SCHEMA_VERSION,
    EvidenceAvailability,
    EvidenceIdentity,
    FrameEvidence,
    GaussianProbeEvidence,
    ScalarEvidence,
    SourceReliabilityEvidence,
)


class TestScalarEvidenceRoundTrip:
    """ScalarEvidence JSON serialization contracts."""

    def test_available_scalar_with_value_round_trips(self) -> None:
        """AVAILABLE scalar with a value survives JSON round-trip."""
        original = ScalarEvidence(
            value=42.5,
            unit="m",
            availability=EvidenceAvailability.AVAILABLE,
            reason=None,
        )
        serialized = json.dumps(asdict(original))
        data = json.loads(serialized)
        reconstructed = self._reconstruct_scalar(data)

        assert reconstructed == original
        assert reconstructed.value == 42.5
        assert reconstructed.unit == "m"
        assert reconstructed.availability == EvidenceAvailability.AVAILABLE

    def test_missing_scalar_round_trips(self) -> None:
        """Missing scalar with reason survives JSON round-trip."""
        original = ScalarEvidence.missing(
            EvidenceAvailability.NOT_REQUESTED,
            unit="s",
            reason="not collected",
        )
        serialized = json.dumps(asdict(original))
        data = json.loads(serialized)
        reconstructed = self._reconstruct_scalar(data)

        assert reconstructed == original
        assert reconstructed.value is None
        assert reconstructed.unit == "s"
        assert reconstructed.availability == EvidenceAvailability.NOT_REQUESTED
        assert reconstructed.reason == "not collected"

    def test_zero_value_differs_from_missing(self) -> None:
        """Zero value is NOT the same as missing evidence."""
        zero_value = ScalarEvidence(
            value=0.0,
            unit="unitless",
            availability=EvidenceAvailability.AVAILABLE,
        )
        missing = ScalarEvidence.missing(
            EvidenceAvailability.NOT_REQUESTED,
            unit="unitless",
        )

        # They are not equal
        assert zero_value != missing

        # Availability flags differ
        assert zero_value.availability == EvidenceAvailability.AVAILABLE
        assert missing.availability == EvidenceAvailability.NOT_REQUESTED

        # Value semantics differ
        assert zero_value.value == 0.0
        assert missing.value is None

    def test_all_availability_states_survive_round_trip(self) -> None:
        """Every EvidenceAvailability member survives round-trip."""
        for availability in EvidenceAvailability:
            if availability == EvidenceAvailability.AVAILABLE:
                # AVAILABLE requires a value, test separately
                original = ScalarEvidence(
                    value=1.0,
                    unit="test",
                    availability=availability,
                    reason=None,
                )
            else:
                original = ScalarEvidence.missing(
                    availability,
                    unit="test",
                    reason=f"test_{availability.value}",
                )
            serialized = json.dumps(asdict(original))
            data = json.loads(serialized)
            reconstructed = self._reconstruct_scalar(data)

            assert reconstructed.availability == availability

    def test_missing_cannot_be_available(self) -> None:
        """missing() constructor rejects AVAILABLE availability."""
        with pytest.raises(ValueError, match="missing evidence cannot be AVAILABLE"):
            ScalarEvidence.missing(EvidenceAvailability.AVAILABLE)

    def test_scalar_with_none_unit_uses_default(self) -> None:
        """Omitting unit uses unitless default."""
        original = ScalarEvidence(value=1.0)
        assert original.unit == "unitless"

    def test_different_units_are_distinguishable(self) -> None:
        """Scalars with different units are not equal."""
        meters = ScalarEvidence(1.0, unit="m")
        seconds = ScalarEvidence(1.0, unit="s")
        assert meters != seconds

    @staticmethod
    def _reconstruct_scalar(data: dict[str, Any]) -> ScalarEvidence:
        """Reconstruct ScalarEvidence from dict, handling enums."""
        availability_str = data["availability"]
        availability = EvidenceAvailability(availability_str)
        return ScalarEvidence(
            value=data["value"],
            unit=data["unit"],
            availability=availability,
            reason=data["reason"],
        )


class TestEvidenceIdentityRoundTrip:
    """EvidenceIdentity JSON serialization contracts."""

    def test_minimal_identity_round_trips(self) -> None:
        """Minimal EvidenceIdentity survives JSON round-trip."""
        original = EvidenceIdentity(
            source_id="scene-0",
            source_ordinal=42,
            timestamp_ns=1_000_000_000,
        )
        serialized = json.dumps(asdict(original))
        data = json.loads(serialized)
        reconstructed = EvidenceIdentity(**data)

        assert reconstructed == original
        assert reconstructed.source_id == "scene-0"
        assert reconstructed.source_ordinal == 42
        assert reconstructed.timestamp_ns == 1_000_000_000

    def test_fully_populated_identity_round_trips(self) -> None:
        """EvidenceIdentity with all fields survives round-trip."""
        original = EvidenceIdentity(
            source_id="kitchen1",
            source_ordinal=1234,
            timestamp_ns=5_000_000_000,
            map_version=7,
            topology_version=2,
            coordinate_frame_version=1,
            optimization_step=50000,
            objective_version="v2_loss",
            resolution_bucket="multires_4",
            schema_version=SCHEMA_VERSION,
        )
        serialized = json.dumps(asdict(original))
        data = json.loads(serialized)
        reconstructed = EvidenceIdentity(**data)

        assert reconstructed == original
        assert reconstructed.schema_version == SCHEMA_VERSION

    def test_identity_defaults_match_module_schema_version(self) -> None:
        """New EvidenceIdentity instances use current SCHEMA_VERSION."""
        identity = EvidenceIdentity(
            source_id="test",
            source_ordinal=0,
            timestamp_ns=0,
        )
        assert identity.schema_version == SCHEMA_VERSION


class TestSourceReliabilityEvidenceRoundTrip:
    """SourceReliabilityEvidence JSON serialization contracts."""

    def test_empty_source_reliability_round_trips(self) -> None:
        """SourceReliabilityEvidence with all defaults survives round-trip."""
        original = SourceReliabilityEvidence()
        serialized = json.dumps(asdict(original))
        data = json.loads(serialized)
        reconstructed = self._reconstruct_source_reliability(data)

        assert reconstructed == original
        # All fields should be NOT_REQUESTED by default
        assert original.source_elapsed_s.availability == EvidenceAvailability.NOT_REQUESTED
        assert original.translation_delta_m.availability == EvidenceAvailability.NOT_REQUESTED

    def test_populated_source_reliability_round_trips(self) -> None:
        """SourceReliabilityEvidence with values survives round-trip."""
        original = SourceReliabilityEvidence(
            source_elapsed_s=ScalarEvidence(0.033, unit="s"),
            translation_delta_m=ScalarEvidence(0.15, unit="m"),
            rotation_delta_deg=ScalarEvidence(5.0, unit="deg"),
            translation_velocity_m_per_s=ScalarEvidence(4.5, unit="m/s"),
            rotation_velocity_deg_per_s=ScalarEvidence(150.0, unit="deg/s"),
            blur_laplacian=ScalarEvidence(45.0),
            exposure_luma=ScalarEvidence(0.6),
            white_clip_fraction=ScalarEvidence(0.02),
            black_clip_fraction=ScalarEvidence(0.05),
            depth_valid_fraction=ScalarEvidence(0.88),
            tracking_confidence=ScalarEvidence(0.95),
        )
        serialized = json.dumps(asdict(original))
        data = json.loads(serialized)
        reconstructed = self._reconstruct_source_reliability(data)

        assert reconstructed == original
        assert reconstructed.translation_delta_m.value == 0.15
        assert reconstructed.tracking_confidence.value == 0.95

    def test_mixed_available_and_missing_source_evidence_round_trips(self) -> None:
        """SourceReliabilityEvidence with mixed states survives round-trip."""
        original = SourceReliabilityEvidence(
            translation_delta_m=ScalarEvidence(0.10, unit="m"),
            rotation_delta_deg=ScalarEvidence.missing(
                EvidenceAvailability.NOT_SUPPORTED,
                unit="deg",
                reason="no rotation sensor",
            ),
            blur_laplacian=ScalarEvidence.missing(
                EvidenceAvailability.INVALID_SOURCE,
                reason="corrupted frame",
            ),
        )
        serialized = json.dumps(asdict(original))
        data = json.loads(serialized)
        reconstructed = self._reconstruct_source_reliability(data)

        assert reconstructed == original
        assert reconstructed.translation_delta_m.value == 0.10
        assert reconstructed.rotation_delta_deg.availability == EvidenceAvailability.NOT_SUPPORTED
        assert reconstructed.blur_laplacian.availability == EvidenceAvailability.INVALID_SOURCE

    @staticmethod
    def _reconstruct_source_reliability(data: dict[str, Any]) -> SourceReliabilityEvidence:
        """Reconstruct SourceReliabilityEvidence from dict."""
        # Reconstruct each ScalarEvidence field
        scalar_fields = {}
        for field_name, field_data in data.items():
            if isinstance(field_data, dict) and "availability" in field_data:
                scalar_fields[field_name] = TestScalarEvidenceRoundTrip._reconstruct_scalar(field_data)
            else:
                scalar_fields[field_name] = field_data
        return SourceReliabilityEvidence(**scalar_fields)


class TestGaussianProbeEvidenceRoundTrip:
    """GaussianProbeEvidence JSON serialization contracts."""

    def test_empty_probe_evidence_round_trips(self) -> None:
        """GaussianProbeEvidence with all defaults survives round-trip."""
        original = GaussianProbeEvidence()
        serialized = json.dumps(asdict(original))
        data = json.loads(serialized)
        reconstructed = self._reconstruct_probe_evidence(data)

        assert reconstructed == original

    def test_populated_rgb_probe_evidence_round_trips(self) -> None:
        """GaussianProbeEvidence with RGB metrics survives round-trip."""
        original = GaussianProbeEvidence(
            rgb_l1_mean=ScalarEvidence(0.08),
            rgb_mse_mean=ScalarEvidence(0.012),
            rgb_ssim=ScalarEvidence(0.85),
            rgb_objective=ScalarEvidence(0.095),
            rgb_residual_normalized=ScalarEvidence(0.15),
        )
        serialized = json.dumps(asdict(original))
        data = json.loads(serialized)
        reconstructed = self._reconstruct_probe_evidence(data)

        assert reconstructed == original
        assert reconstructed.rgb_l1_mean.value == 0.08
        assert reconstructed.rgb_ssim.value == 0.85

    def test_populated_depth_probe_evidence_round_trips(self) -> None:
        """GaussianProbeEvidence with depth metrics survives round-trip."""
        original = GaussianProbeEvidence(
            depth_l1_mean_m=ScalarEvidence(0.025, unit="m"),
            depth_rmse_m=ScalarEvidence(0.035, unit="m"),
            depth_valid_pixel_fraction=ScalarEvidence(0.92),
            depth_valid_pixel_count=ScalarEvidence(240_000, unit="pixels"),
        )
        serialized = json.dumps(asdict(original))
        data = json.loads(serialized)
        reconstructed = self._reconstruct_probe_evidence(data)

        assert reconstructed == original
        assert reconstructed.depth_l1_mean_m.value == 0.025
        assert reconstructed.depth_valid_pixel_count.value == 240_000

    def test_populated_coverage_probe_evidence_round_trips(self) -> None:
        """GaussianProbeEvidence with coverage metrics survives round-trip."""
        original = GaussianProbeEvidence(
            alpha_mean=ScalarEvidence(0.78),
            supported_pixel_fraction=ScalarEvidence(0.82),
            unexplained_pixel_fraction=ScalarEvidence(0.18),
            well_explained_pixel_fraction=ScalarEvidence(0.65),
            probe_pixel_count=ScalarEvidence(260_000, unit="pixels"),
        )
        serialized = json.dumps(asdict(original))
        data = json.loads(serialized)
        reconstructed = self._reconstruct_probe_evidence(data)

        assert reconstructed == original
        assert reconstructed.supported_pixel_fraction.value == 0.82

    def test_populated_gradient_probe_evidence_round_trips(self) -> None:
        """GaussianProbeEvidence with gradient metrics survives round-trip (#803)."""
        original = GaussianProbeEvidence(
            gradient_norm=ScalarEvidence(1.5),
            gradient_energy=ScalarEvidence(2.3),
            gradient_touched_gaussian_fraction=ScalarEvidence(0.45),
            gradient_energy_means=ScalarEvidence(0.8),
            gradient_energy_opacity=ScalarEvidence(0.6),
            gradient_energy_scale=ScalarEvidence(0.5),
            gradient_energy_rotation=ScalarEvidence(0.4),
            gradient_energy_color=ScalarEvidence(1.2),
        )
        serialized = json.dumps(asdict(original))
        data = json.loads(serialized)
        reconstructed = self._reconstruct_probe_evidence(data)

        assert reconstructed == original
        assert reconstructed.gradient_energy.value == 2.3

    def test_populated_residual_structure_probe_evidence_round_trips(self) -> None:
        """GaussianProbeEvidence with residual structure survives round-trip."""
        original = GaussianProbeEvidence(
            residual_entropy=ScalarEvidence(0.72),
            residual_top_decile_mass_fraction=ScalarEvidence(0.35),
        )
        serialized = json.dumps(asdict(original))
        data = json.loads(serialized)
        reconstructed = self._reconstruct_probe_evidence(data)

        assert reconstructed == original
        assert reconstructed.residual_entropy.value == 0.72

    def test_populated_gaussian_support_probe_evidence_round_trips(self) -> None:
        """GaussianProbeEvidence with Gaussian support metrics survives round-trip."""
        original = GaussianProbeEvidence(
            visible_gaussian_fraction=ScalarEvidence(0.68),
            visible_gaussian_count=ScalarEvidence(1850, unit="gaussians"),
        )
        serialized = json.dumps(asdict(original))
        data = json.loads(serialized)
        reconstructed = self._reconstruct_probe_evidence(data)

        assert reconstructed == original
        assert reconstructed.visible_gaussian_count.value == 1850

    def test_populated_timing_probe_evidence_round_trips(self) -> None:
        """GaussianProbeEvidence with timing metrics survives round-trip."""
        original = GaussianProbeEvidence(
            probe_gpu_ms=ScalarEvidence(2.4, unit="ms"),
            probe_wall_ms=ScalarEvidence(3.1, unit="ms"),
        )
        serialized = json.dumps(asdict(original))
        data = json.loads(serialized)
        reconstructed = self._reconstruct_probe_evidence(data)

        assert reconstructed == original
        assert reconstructed.probe_gpu_ms.value == 2.4

    @staticmethod
    def _reconstruct_probe_evidence(data: dict[str, Any]) -> GaussianProbeEvidence:
        """Reconstruct GaussianProbeEvidence from dict."""
        scalar_fields = {}
        for field_name, field_data in data.items():
            if isinstance(field_data, dict) and "availability" in field_data:
                scalar_fields[field_name] = TestScalarEvidenceRoundTrip._reconstruct_scalar(field_data)
            else:
                scalar_fields[field_name] = field_data
        return GaussianProbeEvidence(**scalar_fields)


class TestFrameEvidenceRoundTrip:
    """FrameEvidence JSON serialization contracts."""

    def test_minimal_frame_evidence_round_trips(self) -> None:
        """Minimal FrameEvidence survives JSON round-trip."""
        original = FrameEvidence(
            identity=EvidenceIdentity(
                source_id="scene",
                source_ordinal=10,
                timestamp_ns=100,
            ),
        )
        serialized = json.dumps(asdict(original))
        data = json.loads(serialized)
        reconstructed = self._reconstruct_frame_evidence(data)

        assert reconstructed == original

    def test_fully_populated_frame_evidence_round_trips(self) -> None:
        """FrameEvidence with all populated fields survives round-trip."""
        original = FrameEvidence(
            identity=EvidenceIdentity(
                source_id="kitchen1",
                source_ordinal=1234,
                timestamp_ns=5_000_000_000,
                map_version=7,
                topology_version=2,
                coordinate_frame_version=1,
            ),
            source=SourceReliabilityEvidence(
                translation_delta_m=ScalarEvidence(0.15, unit="m"),
                rotation_delta_deg=ScalarEvidence(5.0, unit="deg"),
            ),
            probe=GaussianProbeEvidence(
                rgb_l1_mean=ScalarEvidence(0.08),
                unexplained_pixel_fraction=ScalarEvidence(0.18),
            ),
            rgb_eligible=True,
            depth_eligible=True,
            hard_reject_reasons=("motion_blur",),
            soft_reject_reasons=("low_confidence",),
            escalations=("manual_override",),
        )
        serialized = json.dumps(asdict(original))
        data = json.loads(serialized)
        reconstructed = self._reconstruct_frame_evidence(data)

        assert reconstructed == original
        assert reconstructed.identity.map_version == 7
        assert reconstructed.source.translation_delta_m.value == 0.15
        assert reconstructed.probe.rgb_l1_mean.value == 0.08
        assert reconstructed.hard_reject_reasons == ("motion_blur",)

    def test_frame_evidence_with_multiple_reject_reasons_round_trips(self) -> None:
        """FrameEvidence with multiple reject reasons survives round-trip."""
        original = FrameEvidence(
            identity=EvidenceIdentity(
                source_id="test",
                source_ordinal=0,
                timestamp_ns=0,
            ),
            hard_reject_reasons=("reason_1", "reason_2", "reason_3"),
            soft_reject_reasons=("soft_1", "soft_2"),
            escalations=("esc_1", "esc_2", "esc_3"),
        )
        serialized = json.dumps(asdict(original))
        data = json.loads(serialized)
        reconstructed = self._reconstruct_frame_evidence(data)

        assert reconstructed == original
        assert len(reconstructed.hard_reject_reasons) == 3
        assert reconstructed.hard_reject_reasons[1] == "reason_2"

    def test_frame_evidence_eligibility_flags_round_trip(self) -> None:
        """FrameEvidence eligibility flags survive round-trip."""
        original_rgb_only = FrameEvidence(
            identity=EvidenceIdentity(
                source_id="test",
                source_ordinal=0,
                timestamp_ns=0,
            ),
            rgb_eligible=True,
            depth_eligible=False,
        )
        serialized = json.dumps(asdict(original_rgb_only))
        data = json.loads(serialized)
        reconstructed = self._reconstruct_frame_evidence(data)

        assert reconstructed == original_rgb_only
        assert reconstructed.rgb_eligible is True
        assert reconstructed.depth_eligible is False

    @staticmethod
    def _reconstruct_frame_evidence(data: dict[str, Any]) -> FrameEvidence:
        """Reconstruct FrameEvidence from dict."""
        # Reconstruct identity
        identity_data = data["identity"]
        identity = EvidenceIdentity(**identity_data)

        # Reconstruct source
        source_data = data["source"]
        source_scalar_fields = {}
        for field_name, field_data in source_data.items():
            if isinstance(field_data, dict) and "availability" in field_data:
                source_scalar_fields[field_name] = TestScalarEvidenceRoundTrip._reconstruct_scalar(field_data)
        source = SourceReliabilityEvidence(**source_scalar_fields)

        # Reconstruct probe
        probe_data = data["probe"]
        probe_scalar_fields = {}
        for field_name, field_data in probe_data.items():
            if isinstance(field_data, dict) and "availability" in field_data:
                probe_scalar_fields[field_name] = TestScalarEvidenceRoundTrip._reconstruct_scalar(field_data)
        probe = GaussianProbeEvidence(**probe_scalar_fields)

        return FrameEvidence(
            identity=identity,
            source=source,
            probe=probe,
            rgb_eligible=data["rgb_eligible"],
            depth_eligible=data["depth_eligible"],
            hard_reject_reasons=tuple(data["hard_reject_reasons"]),
            soft_reject_reasons=tuple(data["soft_reject_reasons"]),
            escalations=tuple(data["escalations"]),
        )


class TestEvidenceSchemaVersion:
    """SCHEMA_VERSION backward-compat contracts."""

    def test_schema_version_is_present_in_module(self) -> None:
        """SCHEMA_VERSION constant is defined."""
        assert isinstance(SCHEMA_VERSION, int)
        assert SCHEMA_VERSION > 0

    def test_new_identity_has_current_schema_version(self) -> None:
        """New EvidenceIdentity instances carry current SCHEMA_VERSION."""
        identity = EvidenceIdentity(
            source_id="test",
            source_ordinal=0,
            timestamp_ns=0,
        )
        assert identity.schema_version == SCHEMA_VERSION

    def test_schema_version_survives_serialization(self) -> None:
        """SCHEMA_VERSION is present and unchanged in serialized form."""
        identity = EvidenceIdentity(
            source_id="test",
            source_ordinal=0,
            timestamp_ns=0,
            schema_version=SCHEMA_VERSION,
        )
        serialized = json.dumps(asdict(identity))
        data = json.loads(serialized)

        assert "schema_version" in data
        assert data["schema_version"] == SCHEMA_VERSION

    def test_schema_version_validation_helper(self) -> None:
        """assert_schema_version detects mismatches."""
        # Valid version
        valid_data = {"schema_version": SCHEMA_VERSION}
        assert_schema_version(valid_data)  # Should not raise

        # Missing version
        with pytest.raises(ValueError, match="schema_version"):
            assert_schema_version({})

        # Unknown version
        with pytest.raises(ValueError, match="Unknown schema_version"):
            assert_schema_version({"schema_version": SCHEMA_VERSION + 999})

    def test_frame_evidence_identity_schema_version_is_preserved(self) -> None:
        """FrameEvidence's nested identity preserves schema_version."""
        frame = FrameEvidence(
            identity=EvidenceIdentity(
                source_id="test",
                source_ordinal=0,
                timestamp_ns=0,
                schema_version=SCHEMA_VERSION,
            ),
        )
        serialized = json.dumps(asdict(frame))
        data = json.loads(serialized)

        assert data["identity"]["schema_version"] == SCHEMA_VERSION


class TestMissingNotEqualZeroSemantics:
    """Missing ≠ zero semantics validation."""

    def test_explicit_zero_is_available(self) -> None:
        """Explicit zero value has AVAILABLE availability."""
        zero = ScalarEvidence(0.0)
        assert zero.availability == EvidenceAvailability.AVAILABLE
        assert zero.value == 0.0

    def test_missing_is_not_available(self) -> None:
        """Missing scalar has non-AVAILABLE availability."""
        missing = ScalarEvidence.missing(EvidenceAvailability.NOT_REQUESTED)
        assert missing.availability != EvidenceAvailability.AVAILABLE
        assert missing.value is None

    def test_zero_and_missing_are_distinct_in_serialization(self) -> None:
        """Zero and missing are distinguishable after JSON round-trip."""
        zero = ScalarEvidence(0.0)
        missing = ScalarEvidence.missing(EvidenceAvailability.NOT_REQUESTED)

        zero_data = json.loads(json.dumps(asdict(zero)))
        missing_data = json.loads(json.dumps(asdict(missing)))

        assert zero_data["value"] == 0.0
        assert zero_data["availability"] == "available"

        assert missing_data["value"] is None
        assert missing_data["availability"] == "not_requested"

    def test_consumer_can_distinguish_zero_from_missing(self) -> None:
        """Consumers using availability flag can treat zero and missing differently."""

        def treat_conservative(evidence: ScalarEvidence) -> float:
            """A conservative policy: unknown values treated as 1.0 (worst case)."""
            if evidence.availability != EvidenceAvailability.AVAILABLE:
                return 1.0  # Treat missing as worst-case
            return evidence.value or 0.0

        zero = ScalarEvidence(0.0)
        missing = ScalarEvidence.missing(EvidenceAvailability.NOT_REQUESTED)

        assert treat_conservative(zero) == 0.0
        assert treat_conservative(missing) == 1.0

    def test_frame_evidence_probe_missing_vs_zero(self) -> None:
        """Frame evidence probe can contain both zero and missing values."""
        frame = FrameEvidence(
            identity=EvidenceIdentity(
                source_id="test",
                source_ordinal=0,
                timestamp_ns=0,
            ),
            probe=GaussianProbeEvidence(
                rgb_l1_mean=ScalarEvidence(0.0),  # Zero: good performance
                rgb_mse_mean=ScalarEvidence.missing(  # Missing: not evaluated
                    EvidenceAvailability.NOT_REQUESTED,
                ),
            ),
        )

        assert frame.probe.rgb_l1_mean.value == 0.0
        assert frame.probe.rgb_l1_mean.availability == EvidenceAvailability.AVAILABLE

        assert frame.probe.rgb_mse_mean.value is None
        assert frame.probe.rgb_mse_mean.availability == EvidenceAvailability.NOT_REQUESTED

        # These are distinct
        assert frame.probe.rgb_l1_mean != frame.probe.rgb_mse_mean


class TestEvidenceAvailabilityCompleteness:
    """All EvidenceAvailability states are handled correctly."""

    def test_all_availability_members_are_represented(self) -> None:
        """Every enum member exists and is a string."""
        for member in EvidenceAvailability:
            assert isinstance(member, EvidenceAvailability)
            assert isinstance(member.value, str)

    def test_all_availability_members_serialize_correctly(self) -> None:
        """Every EvidenceAvailability member serializes and deserializes."""
        for availability in EvidenceAvailability:
            evidence = ScalarEvidence.missing(availability) if availability != EvidenceAvailability.AVAILABLE else ScalarEvidence(1.0)

            serialized = json.dumps(asdict(evidence))
            data = json.loads(serialized)

            reconstructed_availability = EvidenceAvailability(data["availability"])
            assert reconstructed_availability == availability

    def test_specific_availability_states(self) -> None:
        """Each specific availability state has expected semantics."""
        # AVAILABLE: value is meaningful
        available = ScalarEvidence(42.0)
        assert available.availability == EvidenceAvailability.AVAILABLE

        # NOT_REQUESTED: data was never collected
        not_requested = ScalarEvidence.missing(EvidenceAvailability.NOT_REQUESTED)
        assert not_requested.availability == EvidenceAvailability.NOT_REQUESTED
        assert not_requested.value is None

        # NOT_SUPPORTED: this measurement is unsupported for this sensor
        not_supported = ScalarEvidence.missing(EvidenceAvailability.NOT_SUPPORTED, reason="no depth sensor")
        assert not_supported.availability == EvidenceAvailability.NOT_SUPPORTED

        # UNEVALUABLE_NO_MAP_SUPPORT: measurement infrastructure not ready
        no_map = ScalarEvidence.missing(EvidenceAvailability.UNEVALUABLE_NO_MAP_SUPPORT)
        assert no_map.availability == EvidenceAvailability.UNEVALUABLE_NO_MAP_SUPPORT

        # STALE: was available but is now out-of-date
        stale = ScalarEvidence.missing(EvidenceAvailability.STALE, reason="topology changed")
        assert stale.availability == EvidenceAvailability.STALE

        # INVALID_SOURCE: source data was corrupted/invalid
        invalid = ScalarEvidence.missing(EvidenceAvailability.INVALID_SOURCE, reason="frame corrupted")
        assert invalid.availability == EvidenceAvailability.INVALID_SOURCE


def assert_schema_version(data: dict[str, Any]) -> None:
    """Helper: validate that serialized dict has matching schema_version.

    Raises ValueError if schema_version is missing or unknown.
    """
    if "schema_version" not in data:
        raise ValueError("schema_version missing from serialized data")

    version = data["schema_version"]
    if version != SCHEMA_VERSION:
        raise ValueError(
            f"Unknown schema_version {version}; current is {SCHEMA_VERSION}"
        )
