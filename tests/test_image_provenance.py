"""Tests for utils/image_provenance.py (issue #644).

Tests the image/depth provenance contract: tracking transformations,
classifying them as lossless/declared/visualization-only, and guarding
metric entry points against lossy/unknown data.

Covers:
- Canonical float tensors (gold standard for metrics)
- Lossless representation changes (HWC<->CHW, BGR<->RGB, device transfer)
- Declared score transforms (resize with explicit intent)
- Visualization-only artifacts (JPEG, PNG, MP4, contact sheets)
- Rejection of undeclared lossy transforms
- Multires domain mismatch detection
- Serialization to report format
"""

from __future__ import annotations

import pytest

from eligsir.utils.image_provenance import (
    CanonicalScoreDomain,
    ChannelLayout,
    ColorOrder,
    EncodingFormat,
    ImageProvenance,
    ImageSourceDescriptor,
    ScoringEligibility,
    TransformStep,
    TransformType,
    assert_scorable,
    detect_multires_domain_mismatch,
    is_score_eligible,
    provenance_for_canonical_float,
    provenance_for_visualization_artifact,
    provenance_to_report_dict,
)


class TestTransformType:
    """Test TransformType enum and classification."""

    def test_transform_type_values(self):
        """Verify all expected transform types exist."""
        assert TransformType.LOSSLESS_REPRESENTATION_CHANGE.value == "lossless_representation_change"
        assert TransformType.DECLARED_SCORE_TRANSFORM.value == "declared_score_transform"
        assert TransformType.VISUALIZATION_ONLY.value == "visualization_only"

    def test_transform_type_distinctness(self):
        """Ensure types are distinct."""
        types = [
            TransformType.LOSSLESS_REPRESENTATION_CHANGE,
            TransformType.DECLARED_SCORE_TRANSFORM,
            TransformType.VISUALIZATION_ONLY,
        ]
        assert len(types) == len(set(types))


class TestImageSourceDescriptor:
    """Test ImageSourceDescriptor creation and immutability."""

    def test_source_descriptor_creation(self):
        """Create a source descriptor."""
        descriptor = ImageSourceDescriptor(
            dtype="float32",
            shape=(3, 1080, 1920),
            layout=ChannelLayout.CHW,
            color_order=ColorOrder.RGB,
            encoding=EncodingFormat.RAW,
            resolution=(1080, 1920),
            domain="float32_0_1",
        )
        assert descriptor.dtype == "float32"
        assert descriptor.shape == (3, 1080, 1920)
        assert descriptor.layout == ChannelLayout.CHW
        assert descriptor.resolution == (1080, 1920)

    def test_source_descriptor_immutable(self):
        """Verify source descriptor is frozen."""
        descriptor = ImageSourceDescriptor(
            dtype="float32",
            shape=(3, 1080, 1920),
            layout=ChannelLayout.CHW,
            color_order=ColorOrder.RGB,
            encoding=EncodingFormat.RAW,
            resolution=(1080, 1920),
            domain="float32_0_1",
        )
        with pytest.raises(Exception):  # FrozenInstanceError
            descriptor.dtype = "uint8"


class TestCanonicalScoreDomain:
    """Test CanonicalScoreDomain creation."""

    def test_canonical_domain_creation(self):
        """Create a canonical score domain."""
        domain = CanonicalScoreDomain(
            dtype="float32",
            domain="float32_0_1",
            resolution=(1080, 1920),
            is_declared_transform=False,
        )
        assert domain.dtype == "float32"
        assert domain.domain == "float32_0_1"
        assert domain.resolution == (1080, 1920)
        assert domain.is_declared_transform is False

    def test_canonical_domain_with_declared_transform(self):
        """Create a canonical domain for a declared transform."""
        domain = CanonicalScoreDomain(
            dtype="float32",
            domain="float32_0_1",
            resolution=(540, 960),
            is_declared_transform=True,
        )
        assert domain.is_declared_transform is True


class TestTransformStep:
    """Test TransformStep creation and validation."""

    def test_lossless_step(self):
        """Create a lossless representation change step."""
        step = TransformStep(
            transform_type=TransformType.LOSSLESS_REPRESENTATION_CHANGE,
            description="HWC to CHW permutation",
        )
        assert step.transform_type == TransformType.LOSSLESS_REPRESENTATION_CHANGE
        assert step.description == "HWC to CHW permutation"
        assert step.target_resolution is None
        assert step.target_domain is None

    def test_declared_transform_requires_target(self):
        """DECLARED_SCORE_TRANSFORM must specify target_resolution or target_domain."""
        with pytest.raises(ValueError, match="must specify target_resolution"):
            TransformStep(
                transform_type=TransformType.DECLARED_SCORE_TRANSFORM,
                description="resize without target",
            )

    def test_declared_transform_with_resolution(self):
        """Create a declared transform step with target resolution."""
        step = TransformStep(
            transform_type=TransformType.DECLARED_SCORE_TRANSFORM,
            description="bilinear resize to half resolution",
            target_resolution=(540, 960),
        )
        assert step.target_resolution == (540, 960)

    def test_declared_transform_with_domain(self):
        """Create a declared transform step with target domain."""
        step = TransformStep(
            transform_type=TransformType.DECLARED_SCORE_TRANSFORM,
            description="clamp to [0,1]",
            target_domain="float32_0_1",
        )
        assert step.target_domain == "float32_0_1"

    def test_visualization_step(self):
        """Create a visualization-only step."""
        step = TransformStep(
            transform_type=TransformType.VISUALIZATION_ONLY,
            description="JPEG encode quality 92",
        )
        assert step.transform_type == TransformType.VISUALIZATION_ONLY


class TestImageProvenance:
    """Test ImageProvenance creation and mutation."""

    def test_provenance_creation(self):
        """Create a provenance with source and canonical domain."""
        source = ImageSourceDescriptor(
            dtype="float32",
            shape=(3, 1080, 1920),
            layout=ChannelLayout.CHW,
            color_order=ColorOrder.RGB,
            encoding=EncodingFormat.RAW,
            resolution=(1080, 1920),
            domain="float32_0_1",
        )
        canonical = CanonicalScoreDomain(
            dtype="float32",
            domain="float32_0_1",
            resolution=(1080, 1920),
            is_declared_transform=False,
        )
        provenance = ImageProvenance(
            source=source,
            canonical_score_domain=canonical,
            steps=(),
        )
        assert provenance.source == source
        assert provenance.canonical_score_domain == canonical
        assert len(provenance.steps) == 0

    def test_add_step(self):
        """Add a step to a provenance."""
        provenance = provenance_for_canonical_float(
            "float32", (3, 1080, 1920), (1080, 1920)
        )
        step = TransformStep(
            transform_type=TransformType.LOSSLESS_REPRESENTATION_CHANGE,
            description="CHW to HWC permutation",
        )
        new_provenance = provenance.add_step(step)
        assert len(provenance.steps) == 0  # Original unchanged
        assert len(new_provenance.steps) == 1
        assert new_provenance.steps[0] == step

    def test_add_multiple_steps(self):
        """Add multiple steps to a provenance."""
        provenance = provenance_for_canonical_float(
            "float32", (3, 1080, 1920), (1080, 1920)
        )
        step1 = TransformStep(
            transform_type=TransformType.LOSSLESS_REPRESENTATION_CHANGE,
            description="CHW to HWC",
        )
        step2 = TransformStep(
            transform_type=TransformType.LOSSLESS_REPRESENTATION_CHANGE,
            description="CPU to GPU",
        )
        new_provenance = provenance.add_steps([step1, step2])
        assert len(new_provenance.steps) == 2
        assert new_provenance.steps[0] == step1
        assert new_provenance.steps[1] == step2


class TestScoringEligibility:
    """Test ScoringEligibility result object."""

    def test_eligible_provenance(self):
        """Create an eligible scoring result."""
        result = ScoringEligibility(is_eligible=True)
        assert result.is_eligible is True
        assert len(result.reasons) == 0
        assert len(result.warnings) == 0
        assert bool(result) is True

    def test_ineligible_provenance(self):
        """Create an ineligible scoring result."""
        result = ScoringEligibility(
            is_eligible=False,
            reasons=("Contains VISUALIZATION_ONLY step",),
        )
        assert result.is_eligible is False
        assert len(result.reasons) == 1
        assert bool(result) is False

    def test_eligible_with_warnings(self):
        """Eligible result can have warnings."""
        result = ScoringEligibility(
            is_eligible=True,
            reasons=(),
            warnings=("Domain differs from source (but declared)",),
        )
        assert result.is_eligible is True
        assert len(result.warnings) == 1


class TestCanonicalFloatProvenance:
    """Test creation of canonical float provenances."""

    def test_float32_rgb(self):
        """Create canonical provenance for float32 RGB."""
        provenance = provenance_for_canonical_float(
            "float32", (3, 1080, 1920), (1080, 1920)
        )
        assert provenance.source.dtype == "float32"
        assert provenance.source.layout == ChannelLayout.CHW
        assert provenance.source.color_order == ColorOrder.RGB
        assert provenance.canonical_score_domain.dtype == "float32"
        assert len(provenance.steps) == 0

    def test_float64_scalar(self):
        """Create canonical provenance for float64 scalar (e.g., depth)."""
        provenance = provenance_for_canonical_float(
            "float64", (1080, 1920), (1080, 1920)
        )
        assert provenance.source.dtype == "float64"
        assert provenance.source.layout == ChannelLayout.SCALAR
        assert provenance.source.color_order == ColorOrder.GRAYSCALE
        assert len(provenance.steps) == 0

    def test_invalid_dtype(self):
        """Reject non-float dtypes."""
        with pytest.raises(ValueError, match="must be float32 or float64"):
            provenance_for_canonical_float("uint8", (3, 1080, 1920), (1080, 1920))


class TestVisualizationArtifactProvenance:
    """Test creation of visualization artifact provenances."""

    def test_jpeg_artifact(self):
        """Create provenance for a JPEG artifact."""
        provenance = provenance_for_visualization_artifact(
            "JPEG encode quality 92", "float32", (1080, 1920)
        )
        assert len(provenance.steps) == 1
        assert provenance.steps[0].transform_type == TransformType.VISUALIZATION_ONLY
        assert "JPEG" in provenance.steps[0].description

    def test_mp4_artifact(self):
        """Create provenance for an MP4 frame."""
        provenance = provenance_for_visualization_artifact(
            "MP4 H.264 encode", "uint8", (1080, 1920)
        )
        assert len(provenance.steps) == 1
        assert provenance.steps[0].transform_type == TransformType.VISUALIZATION_ONLY

    def test_contact_sheet_artifact(self):
        """Create provenance for a contact sheet PNG."""
        provenance = provenance_for_visualization_artifact(
            "contact_sheet.png", "uint8", (4320, 3840)
        )
        assert len(provenance.steps) == 1
        assert provenance.steps[0].transform_type == TransformType.VISUALIZATION_ONLY


class TestScoringEligibilityCheck:
    """Test is_score_eligible() guard function."""

    def test_none_provenance_ineligible(self):
        """None provenance is ineligible."""
        result = is_score_eligible(None)
        assert result.is_eligible is False
        assert len(result.reasons) > 0

    def test_canonical_float_eligible(self):
        """Canonical float provenance is eligible."""
        provenance = provenance_for_canonical_float(
            "float32", (3, 1080, 1920), (1080, 1920)
        )
        result = is_score_eligible(provenance)
        assert result.is_eligible is True
        assert len(result.reasons) == 0

    def test_visualization_artifact_ineligible(self):
        """Visualization artifact is ineligible."""
        provenance = provenance_for_visualization_artifact(
            "JPEG encode", "float32", (1080, 1920)
        )
        result = is_score_eligible(provenance)
        assert result.is_eligible is False
        assert len(result.reasons) > 0
        assert "VISUALIZATION_ONLY" in result.reasons[0]

    def test_lossless_steps_eligible(self):
        """Provenance with only lossless steps is eligible."""
        provenance = provenance_for_canonical_float(
            "float32", (3, 1080, 1920), (1080, 1920)
        )
        step1 = TransformStep(
            transform_type=TransformType.LOSSLESS_REPRESENTATION_CHANGE,
            description="CHW to HWC",
        )
        step2 = TransformStep(
            transform_type=TransformType.LOSSLESS_REPRESENTATION_CHANGE,
            description="BGR to RGB",
        )
        provenance = provenance.add_steps([step1, step2])
        result = is_score_eligible(provenance)
        assert result.is_eligible is True

    def test_declared_transform_eligible(self):
        """Provenance with declared transform is eligible."""
        source = ImageSourceDescriptor(
            dtype="float32",
            shape=(3, 1080, 1920),
            layout=ChannelLayout.CHW,
            color_order=ColorOrder.RGB,
            encoding=EncodingFormat.RAW,
            resolution=(1080, 1920),
            domain="float32_0_1",
        )
        canonical = CanonicalScoreDomain(
            dtype="float32",
            domain="float32_0_1",
            resolution=(540, 960),  # Declared lower resolution
            is_declared_transform=True,
        )
        provenance = ImageProvenance(source=source, canonical_score_domain=canonical)
        step = TransformStep(
            transform_type=TransformType.DECLARED_SCORE_TRANSFORM,
            description="bilinear resize to eval resolution",
            target_resolution=(540, 960),
        )
        provenance = provenance.add_step(step)
        result = is_score_eligible(provenance)
        assert result.is_eligible is True

    def test_mixed_visualization_and_lossless_ineligible(self):
        """Visualization step mixed with lossless steps makes it ineligible."""
        provenance = provenance_for_canonical_float(
            "float32", (3, 1080, 1920), (1080, 1920)
        )
        step1 = TransformStep(
            transform_type=TransformType.LOSSLESS_REPRESENTATION_CHANGE,
            description="CHW to HWC",
        )
        step2 = TransformStep(
            transform_type=TransformType.VISUALIZATION_ONLY,
            description="uint8 quantization",
        )
        provenance = provenance.add_steps([step1, step2])
        result = is_score_eligible(provenance)
        assert result.is_eligible is False
        assert "VISUALIZATION_ONLY" in result.reasons[0]

    def test_undeclared_resolution_change_is_ineligible(self):
        """A changed score grid is never accepted as an implicit conversion."""
        source = ImageSourceDescriptor(
            dtype="float32",
            shape=(3, 1080, 1920),
            layout=ChannelLayout.CHW,
            color_order=ColorOrder.RGB,
            encoding=EncodingFormat.RAW,
            resolution=(1080, 1920),
            domain="float32_0_1",
        )
        canonical = CanonicalScoreDomain(
            dtype="float32",
            domain="float32_0_1",
            resolution=(540, 960),  # Different resolution, NOT declared
            is_declared_transform=False,
        )
        provenance = ImageProvenance(source=source, canonical_score_domain=canonical)
        result = is_score_eligible(provenance)
        assert result.is_eligible is False
        assert "resolution" in result.reasons[0].lower()

    def test_declared_resolution_requires_a_matching_transform_step(self):
        """A boolean declaration alone cannot hide an untracked resize."""
        source = ImageSourceDescriptor(
            dtype="float32",
            shape=(3, 1080, 1920),
            layout=ChannelLayout.CHW,
            color_order=ColorOrder.RGB,
            encoding=EncodingFormat.RAW,
            resolution=(1080, 1920),
            domain="float32_0_1",
        )
        canonical = CanonicalScoreDomain(
            dtype="float32",
            domain="float32_0_1",
            resolution=(540, 960),
            is_declared_transform=True,
        )
        provenance = ImageProvenance(source=source, canonical_score_domain=canonical)

        result = is_score_eligible(provenance)

        assert result.is_eligible is False
        assert "matching declared transform" in result.reasons[0].lower()

    @pytest.mark.parametrize(
        "encoding",
        [
            EncodingFormat.JPEG,
            EncodingFormat.PNG,
            EncodingFormat.MP4,
            EncodingFormat.H264,
            EncodingFormat.YUV420,
            EncodingFormat.WEBP,
        ],
    )
    def test_encoded_presentation_source_is_ineligible_without_an_extra_step(self, encoding):
        """Lossy/presentation provenance fails closed even if a step was omitted."""
        source = ImageSourceDescriptor(
            dtype="float32",
            shape=(3, 16, 24),
            layout=ChannelLayout.CHW,
            color_order=ColorOrder.RGB,
            encoding=encoding,
            resolution=(16, 24),
            domain="float32_0_1",
        )
        canonical = CanonicalScoreDomain(
            dtype="float32",
            domain="float32_0_1",
            resolution=(16, 24),
        )

        result = is_score_eligible(
            ImageProvenance(source=source, canonical_score_domain=canonical)
        )

        assert result.is_eligible is False
        assert "encoding" in result.reasons[0].lower()


class TestAssertScorableGuard:
    """Test assert_scorable() guard function."""

    def test_assert_scorable_passes(self):
        """Assert passes for eligible provenance."""
        provenance = provenance_for_canonical_float(
            "float32", (3, 1080, 1920), (1080, 1920)
        )
        assert_scorable(provenance, context="test_metric")  # No exception

    def test_assert_scorable_rejects_visualization(self):
        """Assert rejects visualization artifact."""
        provenance = provenance_for_visualization_artifact(
            "JPEG", "float32", (1080, 1920)
        )
        with pytest.raises(ValueError, match="not eligible for scientific metrics"):
            assert_scorable(provenance, context="compute_psnr")

    def test_assert_scorable_rejects_none(self):
        """Assert rejects None provenance."""
        with pytest.raises(ValueError, match="not eligible for scientific metrics"):
            assert_scorable(None, context="test")

    def test_assert_scorable_includes_context(self):
        """Assert error message includes context."""
        provenance = provenance_for_visualization_artifact(
            "JPEG", "float32", (1080, 1920)
        )
        with pytest.raises(ValueError, match="compute_psnr"):
            assert_scorable(provenance, context="compute_psnr")


class TestMultiresDomainMismatch:
    """Test detect_multires_domain_mismatch() function."""

    def test_matching_resolutions_no_mismatch(self):
        """No mismatch when resolutions match."""
        gt_prov = provenance_for_canonical_float(
            "float32", (3, 1080, 1920), (1080, 1920)
        )
        render_prov = provenance_for_canonical_float(
            "float32", (3, 1080, 1920), (1080, 1920)
        )
        mismatch = detect_multires_domain_mismatch(gt_prov, render_prov)
        assert mismatch is None

    def test_same_resolution_with_different_declared_grid_transforms_is_a_mismatch(self):
        """A crop and a resize are not the same score domain by shape alone."""
        source = ImageSourceDescriptor(
            dtype="float32",
            shape=(3, 8, 10),
            layout=ChannelLayout.CHW,
            color_order=ColorOrder.RGB,
            encoding=EncodingFormat.RAW,
            resolution=(8, 10),
            domain="float32_0_1",
        )
        canonical = CanonicalScoreDomain(
            dtype="float32",
            domain="float32_0_1",
            resolution=(4, 5),
            is_declared_transform=True,
        )
        gt = ImageProvenance(source=source, canonical_score_domain=canonical).add_step(
            TransformStep(
                transform_type=TransformType.DECLARED_SCORE_TRANSFORM,
                description="bilinear resize",
                target_resolution=(4, 5),
            )
        )
        render = ImageProvenance(source=source, canonical_score_domain=canonical).add_step(
            TransformStep(
                transform_type=TransformType.DECLARED_SCORE_TRANSFORM,
                description="center crop",
                target_resolution=(4, 5),
            )
        )

        mismatch = detect_multires_domain_mismatch(gt, render)

        assert mismatch is not None
        assert "Transform mismatch" in mismatch

    def test_resolution_mismatch(self):
        """Detect resolution mismatch."""
        gt_prov = provenance_for_canonical_float(
            "float32", (3, 1080, 1920), (1080, 1920)
        )
        render_prov = provenance_for_canonical_float(
            "float32", (3, 540, 960), (540, 960)
        )
        mismatch = detect_multires_domain_mismatch(gt_prov, render_prov)
        assert mismatch is not None
        assert "Resolution mismatch" in mismatch

    def test_transform_status_mismatch(self):
        """Detect transform status mismatch (declared vs native)."""
        source = ImageSourceDescriptor(
            dtype="float32",
            shape=(3, 1080, 1920),
            layout=ChannelLayout.CHW,
            color_order=ColorOrder.RGB,
            encoding=EncodingFormat.RAW,
            resolution=(1080, 1920),
            domain="float32_0_1",
        )
        # GT at native resolution
        gt_canonical = CanonicalScoreDomain(
            dtype="float32",
            domain="float32_0_1",
            resolution=(1080, 1920),
            is_declared_transform=False,
        )
        gt_prov = ImageProvenance(source=source, canonical_score_domain=gt_canonical)

        # Render at declared-transform resolution
        render_canonical = CanonicalScoreDomain(
            dtype="float32",
            domain="float32_0_1",
            resolution=(1080, 1920),
            is_declared_transform=True,
        )
        render_prov = ImageProvenance(source=source, canonical_score_domain=render_canonical)

        mismatch = detect_multires_domain_mismatch(gt_prov, render_prov)
        assert mismatch is not None
        assert "Transform mismatch" in mismatch

    def test_numeric_domain_mismatch(self):
        """Detect numeric domain mismatch."""
        source = ImageSourceDescriptor(
            dtype="float32",
            shape=(3, 1080, 1920),
            layout=ChannelLayout.CHW,
            color_order=ColorOrder.RGB,
            encoding=EncodingFormat.RAW,
            resolution=(1080, 1920),
            domain="float32_0_1",
        )
        gt_canonical = CanonicalScoreDomain(
            dtype="float32",
            domain="float32_0_1",
            resolution=(1080, 1920),
            is_declared_transform=False,
        )
        render_canonical = CanonicalScoreDomain(
            dtype="float32",
            domain="float32_0_255",  # Different domain
            resolution=(1080, 1920),
            is_declared_transform=False,
        )
        gt_prov = ImageProvenance(source=source, canonical_score_domain=gt_canonical)
        render_prov = ImageProvenance(source=source, canonical_score_domain=render_canonical)

        mismatch = detect_multires_domain_mismatch(gt_prov, render_prov)
        assert mismatch is not None
        assert "domain" in mismatch.lower()


class TestProvenanceReportSerialization:
    """Test provenance_to_report_dict() serialization."""

    def test_canonical_float_serialization(self):
        """Serialize canonical float provenance."""
        provenance = provenance_for_canonical_float(
            "float32", (3, 1080, 1920), (1080, 1920)
        )
        report = provenance_to_report_dict(provenance)

        assert "source" in report
        assert "canonical_score_domain" in report
        assert "steps" in report

        assert report["source"]["dtype"] == "float32"
        assert report["source"]["layout"] == "chw"
        assert report["canonical_score_domain"]["dtype"] == "float32"
        assert report["canonical_score_domain"]["resolution"] == (1080, 1920)
        assert len(report["steps"]) == 0

    def test_with_steps_serialization(self):
        """Serialize provenance with transformation steps."""
        provenance = provenance_for_canonical_float(
            "float32", (3, 1080, 1920), (1080, 1920)
        )
        step1 = TransformStep(
            transform_type=TransformType.LOSSLESS_REPRESENTATION_CHANGE,
            description="CHW to HWC",
        )
        step2 = TransformStep(
            transform_type=TransformType.DECLARED_SCORE_TRANSFORM,
            description="resize to eval resolution",
            target_resolution=(540, 960),
        )
        provenance = provenance.add_steps([step1, step2])
        report = provenance_to_report_dict(provenance)

        assert len(report["steps"]) == 2
        assert report["steps"][0]["type"] == "lossless_representation_change"
        assert report["steps"][0]["description"] == "CHW to HWC"
        assert report["steps"][1]["type"] == "declared_score_transform"
        assert report["steps"][1]["target_resolution"] == (540, 960)

    def test_visualization_serialization(self):
        """Serialize visualization artifact provenance."""
        provenance = provenance_for_visualization_artifact(
            "contact_sheet.png", "uint8", (1080, 1920)
        )
        report = provenance_to_report_dict(provenance)

        assert len(report["steps"]) == 1
        assert report["steps"][0]["type"] == "visualization_only"


class TestNegativeCases:
    """Test negative cases: JPEG-sourced GT, MP4, contact sheet, undeclared resize, float->uint8."""

    def test_jpeg_sourced_gt_rejected(self):
        """JPEG-sourced GT (recompressed) is rejected."""
        # Simulate a GT that was loaded from JPEG
        source = ImageSourceDescriptor(
            dtype="uint8",
            shape=(3, 1080, 1920),
            layout=ChannelLayout.HWC,
            color_order=ColorOrder.RGB,
            encoding=EncodingFormat.JPEG,  # Key: JPEG encoding
            resolution=(1080, 1920),
            domain="uint8_0_255",
        )
        canonical = CanonicalScoreDomain(
            dtype="float32",
            domain="float32_0_1",
            resolution=(1080, 1920),
            is_declared_transform=False,
        )
        provenance = ImageProvenance(source=source, canonical_score_domain=canonical)
        # Add a step to convert uint8 to float
        step = TransformStep(
            transform_type=TransformType.VISUALIZATION_ONLY,
            description="uint8 to float32 after JPEG decode",
        )
        provenance = provenance.add_step(step)

        result = is_score_eligible(provenance)
        assert result.is_eligible is False

    def test_mp4_frame_rejected(self):
        """MP4 frame is rejected."""
        provenance = provenance_for_visualization_artifact(
            "MP4 H.264 decode frame 1234", "uint8", (1080, 1920)
        )
        result = is_score_eligible(provenance)
        assert result.is_eligible is False

    def test_contact_sheet_png_rejected(self):
        """Contact sheet PNG is rejected."""
        provenance = provenance_for_visualization_artifact(
            "test_contact_sheet.png", "uint8", (2160, 3840)
        )
        result = is_score_eligible(provenance)
        assert result.is_eligible is False

    def test_undeclared_resize_with_visualization_rejected(self):
        """Undeclared resize followed by visualization is rejected."""
        provenance = provenance_for_canonical_float(
            "float32", (3, 1080, 1920), (1080, 1920)
        )
        # Add a resize WITHOUT declaring it
        step1 = TransformStep(
            transform_type=TransformType.DECLARED_SCORE_TRANSFORM,
            description="bilinear resize",
            target_resolution=(540, 960),
        )
        # Add visualization (this is what makes it clearly ineligible)
        step2 = TransformStep(
            transform_type=TransformType.VISUALIZATION_ONLY,
            description="uint8 quantization",
        )
        provenance = provenance.add_steps([step1, step2])

        result = is_score_eligible(provenance)
        assert result.is_eligible is False
        assert "VISUALIZATION_ONLY" in result.reasons[0]

    def test_float_to_uint8_cast_with_visualization(self):
        """Silent float->uint8 cast as visualization is rejected."""
        provenance = provenance_for_canonical_float(
            "float32", (3, 1080, 1920), (1080, 1920)
        )
        # Add uint8 quantization as visualization step
        step = TransformStep(
            transform_type=TransformType.VISUALIZATION_ONLY,
            description="float32 to uint8 quantization (irreversible)",
        )
        provenance = provenance.add_step(step)

        result = is_score_eligible(provenance)
        assert result.is_eligible is False
        assert len(result.reasons) > 0
