"""Image/depth provenance tracking and scoring eligibility contract.

Issue #644: Track image transforms/provenance and forbid lossy artifacts as metric inputs.

This module defines the architectural contract for image/depth data lineage:
- Source descriptor (dtype, shape, layout, color order, encoding, resolution)
- Canonical score domain (dtype, domain, resolution)
- Ordered transformation steps (classified as lossless, declared transform, or visualization-only)
- Guard APIs that reject visualization-only data and undeclared sample-changing operations

The provenance model is torch-free at import time. It accepts duck-typed shape/dtype
descriptors and can work with numpy arrays or other numeric containers.

Key classes:
- ImageSourceDescriptor: Captures source metadata
- CanonicalScoreDomain: Defines the target for evaluation
- TransformStep: One operation in the lineage, classified by type
- ImageProvenance: Complete lineage from source to current state
- ScoringEligibility: Result of an eligibility check with reasons

Key functions:
- assert_scorable(tensor_provenance): Raises if data cannot be used for scientific metrics
- is_score_eligible(tensor_provenance): Returns ScoringEligibility without raising
- provenance_for_canonical_float(dtype, shape, resolution): Create provenance for a
  canonical float tensor (the gold standard for scientific use)
"""

from __future__ import annotations

import dataclasses
import enum
from typing import Any, Optional, Sequence


class TransformType(enum.Enum):
    """Classification of a transformation step in image lineage.

    Each operation is exactly one type. The classification is by operation kind,
    not by whether individual samples happen to be preserved; e.g., nearest-neighbor
    resize is a SAMPLE_CHANGING_TRANSFORM even though surviving pixels are unchanged.
    """

    # Representation change that does not alter sample values or sample count.
    # Examples: HWC<->CHW permutation, BGR<->RGB channel swap, CPU<->GPU transfer,
    # contiguous copy, int->float with exact recoverability.
    LOSSLESS_REPRESENTATION_CHANGE = "lossless_representation_change"

    # An explicitly declared transformation to a different sample domain.
    # Must be accompanied by metadata about the target resolution/domain.
    # Examples: resize (any method), crop, depth reprojection, alpha compositing,
    # clamping to [0,1] when it changes samples.
    # This is used when an experiment intentionally changes the score domain.
    DECLARED_SCORE_TRANSFORM = "declared_score_transform"

    # Transformation for presentation/visualization only, never eligible for metrics.
    # Examples: uint8 quantization, JPEG/PNG encoding, MP4/H.264 video encoding,
    # contact sheets, web-viewer frames, any lossy recompression.
    VISUALIZATION_ONLY = "visualization_only"


class ColorOrder(enum.Enum):
    """Color channel ordering convention."""

    RGB = "rgb"
    BGR = "bgr"
    GRAYSCALE = "grayscale"


class ChannelLayout(enum.Enum):
    """Spatial arrangement of channels and height/width."""

    CHW = "chw"  # Channels first: (C, H, W)
    HWC = "hwc"  # Channels last: (H, W, C)
    SCALAR = "scalar"  # Single channel or scalar


class EncodingFormat(enum.Enum):
    """Data encoding or compression format."""

    RAW = "raw"
    JPEG = "jpeg"
    PNG = "png"
    MP4 = "mp4"
    H264 = "h264"
    YUV420 = "yuv420"
    WEBP = "webp"


@dataclasses.dataclass(frozen=True)
class ImageSourceDescriptor:
    """Metadata describing the source of an image/depth tensor.

    Attributes:
        dtype: Data type string (e.g., 'float32', 'uint8', 'uint16')
        shape: Shape tuple (int, int, int) representing dimensions
        layout: ChannelLayout enum (CHW, HWC, SCALAR)
        color_order: ColorOrder enum (RGB, BGR, GRAYSCALE)
        encoding: EncodingFormat enum (RAW, JPEG, PNG, etc.)
        resolution: (height, width) tuple in pixels
        domain: Numeric domain descriptor, e.g., 'uint8_0_255', 'float32_0_1'
    """

    dtype: str
    shape: tuple[int, ...]
    layout: ChannelLayout
    color_order: ColorOrder
    encoding: EncodingFormat
    resolution: tuple[int, int]  # (height, width)
    domain: str  # e.g., "uint8_0_255", "float32_0_1", "float32_0_inf"


@dataclasses.dataclass(frozen=True)
class CanonicalScoreDomain:
    """Definition of the canonical score domain for a tensor.

    Scientific metrics (PSNR, SSIM, LPIPS) must operate on tensors in a
    well-defined canonical domain. If score-bearing data has been transformed,
    this records the transformation intent.

    Attributes:
        dtype: Target data type (e.g., 'float32')
        domain: Numeric domain (e.g., 'float32_0_1')
        resolution: (height, width) in pixels
        is_declared_transform: True if this domain was explicitly requested
            by the user (e.g., via a multires evaluation flag). False if this
            is the native/source resolution.
    """

    dtype: str
    domain: str  # numeric range, e.g., "float32_0_1"
    resolution: tuple[int, int]  # (height, width)
    is_declared_transform: bool = False


@dataclasses.dataclass(frozen=True)
class TransformStep:
    """A single transformation in an image lineage.

    Attributes:
        transform_type: TransformType enum value
        description: Human-readable description of the operation
        target_resolution: If this is a resize/crop, the result resolution
        target_domain: If this is a domain change, the numeric domain
    """

    transform_type: TransformType
    description: str
    target_resolution: Optional[tuple[int, int]] = None
    target_domain: Optional[str] = None

    def __post_init__(self):
        # Validate consistency: DECLARED_SCORE_TRANSFORM should have target resolution/domain
        if self.transform_type == TransformType.DECLARED_SCORE_TRANSFORM:
            if self.target_resolution is None and self.target_domain is None:
                raise ValueError(
                    f"DECLARED_SCORE_TRANSFORM must specify target_resolution "
                    f"or target_domain: {self.description}"
                )


@dataclasses.dataclass(frozen=True)
class ImageProvenance:
    """Complete provenance/lineage from source to current state.

    Attributes:
        source: ImageSourceDescriptor of the original data
        canonical_score_domain: CanonicalScoreDomain for metric use
        steps: Ordered sequence of TransformStep applied
    """

    source: ImageSourceDescriptor
    canonical_score_domain: CanonicalScoreDomain
    steps: tuple[TransformStep, ...] = ()

    def add_step(self, step: TransformStep) -> ImageProvenance:
        """Return a new provenance with an additional step appended."""
        return dataclasses.replace(
            self, steps=self.steps + (step,)
        )

    def add_steps(self, steps: Sequence[TransformStep]) -> ImageProvenance:
        """Return a new provenance with additional steps appended."""
        return dataclasses.replace(
            self, steps=self.steps + tuple(steps)
        )


@dataclasses.dataclass(frozen=True)
class ScoringEligibility:
    """Result of checking whether data is eligible for scientific metrics.

    Attributes:
        is_eligible: True if data can safely be used for scoring
        reasons: List of rejection reasons (empty if eligible)
        warnings: List of non-blocking concerns (eligible but worth noting)
    """

    is_eligible: bool
    reasons: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def __bool__(self) -> bool:
        """Allow if is_eligible: checks in boolean context."""
        return self.is_eligible


def is_score_eligible(provenance: ImageProvenance) -> ScoringEligibility:
    """Check if a provenance chain is eligible for scientific metrics.

    Fails closed: returns ineligible for absent/unknown provenance or any
    visualization-only step. Undeclared sample-changing transforms are also
    rejected.

    Args:
        provenance: ImageProvenance instance to check

    Returns:
        ScoringEligibility with is_eligible and list of reasons
    """
    reasons = []
    # Missing provenance entirely
    if provenance is None:
        return ScoringEligibility(
            is_eligible=False,
            reasons=("Provenance is None; cannot verify eligibility",),
        )

    if not isinstance(provenance, ImageProvenance):
        return ScoringEligibility(
            is_eligible=False,
            reasons=("Provenance has an unknown type; cannot verify eligibility",),
        )

    if provenance.source.encoding is not EncodingFormat.RAW:
        encoding = provenance.source.encoding
        encoding_name = (
            encoding.value if isinstance(encoding, EncodingFormat) else repr(encoding)
        )
        reasons.append(
            f"Source encoding {encoding_name} is a serialized/"
            "presentation artifact and cannot enter metric code"
        )

    source = provenance.source
    if source.layout is ChannelLayout.HWC:
        described_resolution = tuple(source.shape[:2])
    else:
        described_resolution = tuple(source.shape[-2:])
    if described_resolution != tuple(source.resolution):
        reasons.append(
            "Source shape/layout does not match its declared source resolution"
        )
    if not source.domain.startswith(f"{source.dtype}_"):
        reasons.append("Source dtype and numeric-domain descriptor disagree")
    canonical = provenance.canonical_score_domain
    if not canonical.domain.startswith(f"{canonical.dtype}_"):
        reasons.append("Canonical dtype and numeric-domain descriptor disagree")

    # Check for visualization-only steps
    for i, step in enumerate(provenance.steps):
        if step.transform_type == TransformType.VISUALIZATION_ONLY:
            reasons.append(
                f"Step {i}: {step.description} (VISUALIZATION_ONLY "
                "cannot enter metric code)"
            )

    # If any visualization-only found, fail immediately
    if reasons:
        return ScoringEligibility(is_eligible=False, reasons=tuple(reasons))

    # Check for undeclared sample-changing transforms
    # (DECLARED_SCORE_TRANSFORM is OK because it was explicit)
    for i, step in enumerate(provenance.steps):
        if step.transform_type not in (
            TransformType.LOSSLESS_REPRESENTATION_CHANGE,
            TransformType.DECLARED_SCORE_TRANSFORM,
            TransformType.VISUALIZATION_ONLY,  # already filtered above
        ):
            reasons.append(
                f"Step {i}: {step.description} has unknown/invalid transform type"
            )

    if reasons:
        return ScoringEligibility(is_eligible=False, reasons=tuple(reasons))

    declared_steps = tuple(
        step
        for step in provenance.steps
        if step.transform_type is TransformType.DECLARED_SCORE_TRANSFORM
    )
    resolution_changed = provenance.source.resolution != canonical.resolution
    domain_changed = provenance.source.domain != canonical.domain

    if (resolution_changed or domain_changed) and not canonical.is_declared_transform:
        changed = []
        if resolution_changed:
            changed.append("resolution")
        if domain_changed:
            changed.append("numeric domain")
        reasons.append(
            f"Canonical {' and '.join(changed)} differs from the source but is not "
            "marked as a declared score transform"
        )
    elif canonical.is_declared_transform:
        resolution_targets = tuple(
            step.target_resolution
            for step in declared_steps
            if step.target_resolution is not None
        )
        domain_targets = tuple(
            step.target_domain for step in declared_steps if step.target_domain is not None
        )
        has_matching_resolution = bool(
            resolution_targets and resolution_targets[-1] == canonical.resolution
        )
        has_matching_domain = bool(domain_targets and domain_targets[-1] == canonical.domain)
        if resolution_changed and not has_matching_resolution:
            reasons.append(
                "Canonical resolution differs from the source without a matching "
                "declared transform step"
            )
        if domain_changed and not has_matching_domain:
            reasons.append(
                "Canonical numeric domain differs from the source without a matching "
                "declared transform step"
            )
        if not resolution_changed and not domain_changed and not declared_steps:
            reasons.append(
                "Canonical domain is marked transformed without a declared transform step"
            )

    return ScoringEligibility(
        is_eligible=not reasons,
        reasons=tuple(reasons),
        warnings=(),
    )


def assert_scorable(provenance: ImageProvenance, context: str = "") -> None:
    """Assert that provenance is eligible for scientific metrics.

    Raises ValueError if provenance is not eligible. Use this as a guard
    at metric entry points.

    Args:
        provenance: ImageProvenance to check
        context: Optional description of where this check occurs

    Raises:
        ValueError: If provenance is not eligible for scoring
    """
    eligibility = is_score_eligible(provenance)
    if not eligibility.is_eligible:
        reasons_str = "\n  ".join(eligibility.reasons)
        ctx_str = f" ({context})" if context else ""
        raise ValueError(
            f"Tensor is not eligible for scientific metrics{ctx_str}.\n"
            f"Reasons:\n  {reasons_str}"
        )


def provenance_for_canonical_float(
    dtype: str, shape: tuple[int, ...], resolution: tuple[int, int]
) -> ImageProvenance:
    """Create provenance for a canonical float tensor (gold standard).

    This is used for source data that is already in the canonical scoring
    domain: float32/float64, no lossy transforms applied, exact data.

    Args:
        dtype: Data type, should be 'float32' or 'float64'
        shape: Tensor shape
        resolution: (height, width) in pixels

    Returns:
        ImageProvenance for a canonical float tensor with no transforms
    """
    if dtype not in ("float32", "float64"):
        raise ValueError(f"Canonical float must be float32 or float64, got {dtype}")

    channels = int(shape[0]) if len(shape) == 3 else 1
    source = ImageSourceDescriptor(
        dtype=dtype,
        shape=shape,
        layout=ChannelLayout.CHW if len(shape) == 3 else ChannelLayout.SCALAR,
        color_order=ColorOrder.RGB if channels == 3 else ColorOrder.GRAYSCALE,
        encoding=EncodingFormat.RAW,
        resolution=resolution,
        domain=f"{dtype}_0_1",
    )

    canonical = CanonicalScoreDomain(
        dtype=dtype,
        domain=f"{dtype}_0_1",
        resolution=resolution,
        is_declared_transform=False,
    )

    return ImageProvenance(source=source, canonical_score_domain=canonical, steps=())


def provenance_for_visualization_artifact(
    name: str, original_dtype: str, original_resolution: tuple[int, int]
) -> ImageProvenance:
    """Create provenance for a visualization artifact (JPEG, MP4, contact sheet).

    These provenances are explicitly marked unsuitable for metrics.

    Args:
        name: Descriptive name (e.g., 'contact_sheet_png', 'mp4_frame')
        original_dtype: Original data type before visualization
        original_resolution: Original resolution before visualization

    Returns:
        ImageProvenance with a single VISUALIZATION_ONLY step
    """
    source = ImageSourceDescriptor(
        dtype=original_dtype,
        shape=(original_resolution[0], original_resolution[1], 3),
        layout=ChannelLayout.HWC,
        color_order=ColorOrder.RGB,
        encoding=EncodingFormat.RAW,
        resolution=original_resolution,
        domain=f"{original_dtype}_0_1",
    )

    canonical = CanonicalScoreDomain(
        dtype=original_dtype,
        domain=f"{original_dtype}_0_1",
        resolution=original_resolution,
        is_declared_transform=False,
    )

    provenance = ImageProvenance(source=source, canonical_score_domain=canonical)

    # Add the visualization step
    step = TransformStep(
        transform_type=TransformType.VISUALIZATION_ONLY,
        description=name,
    )

    return provenance.add_step(step)


def detect_multires_domain_mismatch(
    gt_provenance: ImageProvenance, render_provenance: ImageProvenance
) -> Optional[str]:
    """Detect if GT and render are in differently transformed domains.

    This guards against the multires hazard: if GT is at native resolution
    and render is at a declared-transform resolution (or vice versa), metrics
    will be incomparable.

    Args:
        gt_provenance: Provenance of ground-truth image
        render_provenance: Provenance of rendered image

    Returns:
        None if domains match, otherwise a description of the mismatch
    """
    # Check resolution
    if gt_provenance.canonical_score_domain.resolution != render_provenance.canonical_score_domain.resolution:
        gt_res = gt_provenance.canonical_score_domain.resolution
        render_res = render_provenance.canonical_score_domain.resolution
        return (
            f"Resolution mismatch: GT {gt_res} vs render {render_res}. "
            "Metrics are incomparable."
        )

    # Check if transform-status differs
    gt_is_transform = gt_provenance.canonical_score_domain.is_declared_transform
    render_is_transform = render_provenance.canonical_score_domain.is_declared_transform
    if gt_is_transform != render_is_transform:
        return (
            f"Transform mismatch: GT is_declared_transform={gt_is_transform}, "
            f"render is_declared_transform={render_is_transform}. "
            "Cannot compare metrics across different transform domains."
        )

    def _grid_transform_signature(provenance: ImageProvenance):
        return tuple(
            (step.description, step.target_resolution)
            for step in provenance.steps
            if step.transform_type is TransformType.DECLARED_SCORE_TRANSFORM
            and step.target_resolution is not None
        )

    gt_grid_transforms = _grid_transform_signature(gt_provenance)
    render_grid_transforms = _grid_transform_signature(render_provenance)
    if gt_grid_transforms != render_grid_transforms:
        return (
            "Transform mismatch: GT and render use different declared score-grid "
            f"transforms ({gt_grid_transforms} vs {render_grid_transforms})."
        )

    # Check numeric domain
    if gt_provenance.canonical_score_domain.domain != render_provenance.canonical_score_domain.domain:
        gt_domain = gt_provenance.canonical_score_domain.domain
        render_domain = render_provenance.canonical_score_domain.domain
        return (
            f"Numeric domain mismatch: GT {gt_domain} vs render {render_domain}. "
            "Metrics may be incomparable."
        )

    return None


def provenance_to_report_dict(provenance: ImageProvenance) -> dict[str, Any]:
    """Serialize provenance to a report-compatible dictionary.

    Suitable for embedding in report.json["image_provenance"] or similar.

    Args:
        provenance: ImageProvenance to serialize

    Returns:
        Dictionary with source, canonical domain, and step information
    """
    steps_list = []
    for step in provenance.steps:
        step_dict = {
            "type": step.transform_type.value,
            "description": step.description,
        }
        if step.target_resolution is not None:
            step_dict["target_resolution"] = step.target_resolution
        if step.target_domain is not None:
            step_dict["target_domain"] = step.target_domain
        steps_list.append(step_dict)

    return {
        "source": {
            "dtype": provenance.source.dtype,
            "shape": provenance.source.shape,
            "layout": provenance.source.layout.value,
            "color_order": provenance.source.color_order.value,
            "encoding": provenance.source.encoding.value,
            "resolution": provenance.source.resolution,
            "domain": provenance.source.domain,
        },
        "canonical_score_domain": {
            "dtype": provenance.canonical_score_domain.dtype,
            "domain": provenance.canonical_score_domain.domain,
            "resolution": provenance.canonical_score_domain.resolution,
            "is_declared_transform": provenance.canonical_score_domain.is_declared_transform,
        },
        "steps": steps_list,
    }
