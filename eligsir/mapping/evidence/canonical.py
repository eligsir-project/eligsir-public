"""Canonical versioned frame-evidence vocabulary for admission and replay."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

#: Bumped to 2 by issue #803 (batched candidate probe): ``GaussianProbeEvidence``
#: gained additional forward-only and gradient-probe scalars. Every added field
#: is defaulted to ``NOT_REQUESTED``, so a version-1 producer's payload remains
#: constructible and readable unchanged -- the bump records that a version-2
#: envelope can carry strictly more named signals, not that anything moved.
SCHEMA_VERSION = 2


class EvidenceAvailability(str, Enum):
    """Reason-coded availability state for an evidence quantity."""

    AVAILABLE = "available"
    NOT_REQUESTED = "not_requested"
    NOT_SUPPORTED = "not_supported"
    UNEVALUABLE_NO_MAP_SUPPORT = "unevaluable_no_map_support"
    STALE = "stale"
    INVALID_SOURCE = "invalid_source"


@dataclass(frozen=True, slots=True)
class EvidenceIdentity:
    """Versioned source and map identity at which evidence was measured."""

    source_id: str
    source_ordinal: int
    timestamp_ns: int
    map_version: int = 0
    topology_version: int = 0
    coordinate_frame_version: int = 0
    optimization_step: int = 0
    objective_version: str = "default"
    resolution_bucket: str = "default"
    schema_version: int = SCHEMA_VERSION


@dataclass(frozen=True, slots=True)
class ScalarEvidence:
    """One optional scalar together with units and availability provenance."""

    value: Optional[float]
    unit: str = "unitless"
    availability: EvidenceAvailability = EvidenceAvailability.AVAILABLE
    reason: Optional[str] = None

    @classmethod
    def missing(cls, availability: EvidenceAvailability, *, unit: str = "unitless", reason: str | None = None):
        """Construct an explicitly unavailable scalar evidence value."""

        if availability is EvidenceAvailability.AVAILABLE:
            raise ValueError("missing evidence cannot be AVAILABLE")
        return cls(None, unit=unit, availability=availability, reason=reason)


@dataclass(frozen=True, slots=True)
class SourceReliabilityEvidence:
    """Transport, motion, image, depth, and tracking reliability evidence."""

    source_elapsed_s: ScalarEvidence = field(default_factory=lambda: ScalarEvidence.missing(EvidenceAvailability.NOT_REQUESTED, unit="s"))
    translation_delta_m: ScalarEvidence = field(default_factory=lambda: ScalarEvidence.missing(EvidenceAvailability.NOT_REQUESTED, unit="m"))
    rotation_delta_deg: ScalarEvidence = field(default_factory=lambda: ScalarEvidence.missing(EvidenceAvailability.NOT_REQUESTED, unit="deg"))
    translation_velocity_m_per_s: ScalarEvidence = field(default_factory=lambda: ScalarEvidence.missing(EvidenceAvailability.NOT_REQUESTED, unit="m/s"))
    rotation_velocity_deg_per_s: ScalarEvidence = field(default_factory=lambda: ScalarEvidence.missing(EvidenceAvailability.NOT_REQUESTED, unit="deg/s"))
    blur_laplacian: ScalarEvidence = field(default_factory=lambda: ScalarEvidence.missing(EvidenceAvailability.NOT_REQUESTED))
    exposure_luma: ScalarEvidence = field(default_factory=lambda: ScalarEvidence.missing(EvidenceAvailability.NOT_REQUESTED))
    white_clip_fraction: ScalarEvidence = field(default_factory=lambda: ScalarEvidence.missing(EvidenceAvailability.NOT_REQUESTED))
    black_clip_fraction: ScalarEvidence = field(default_factory=lambda: ScalarEvidence.missing(EvidenceAvailability.NOT_REQUESTED))
    depth_valid_fraction: ScalarEvidence = field(default_factory=lambda: ScalarEvidence.missing(EvidenceAvailability.NOT_REQUESTED))
    tracking_confidence: ScalarEvidence = field(default_factory=lambda: ScalarEvidence.missing(EvidenceAvailability.NOT_REQUESTED))


def _absent(unit: str = "unitless"):
    """Shorthand default factory for a NOT_REQUESTED scalar of the given unit."""

    return field(
        default_factory=lambda: ScalarEvidence.missing(
            EvidenceAvailability.NOT_REQUESTED, unit=unit
        )
    )


@dataclass(frozen=True, slots=True)
class GaussianProbeEvidence:
    """Read-only rendered-map evidence for one candidate frame.

    This is the SINGLE canonical vocabulary for read-only render/loss evidence
    about one camera measured against one fixed Gaussian/topology snapshot. It
    is shared, by construction, between the pre-commit candidate probe
    (issue #803) and post-admission repeated-exposure evidence (issue #292):
    the two differ in *when* and *how often* a quantity is measured, never in
    what a quantity means. Do not introduce a second definition of residual,
    visibility, or gradient energy elsewhere.

    None of these fields is a quality or utility score. A high residual can
    mean "this view would teach the map a lot" or "this view is garbage"; the
    combination that decides is issue #500's, and deliberately does not live
    here (issue #803: "Do not label raw high residual as quality or utility").
    """

    # --- RGB objective components (per view, never batch-reduced) ---------
    rgb_l1_mean: ScalarEvidence = _absent()
    rgb_mse_mean: ScalarEvidence = _absent()
    #: Structural similarity of render vs. ground truth (1.0 = identical).
    rgb_ssim: ScalarEvidence = _absent()
    #: ``(1 - lambda_dssim) * L1 + lambda_dssim * (1 - SSIM)`` -- the canonical
    #: photometric objective term, evaluated forward-only for this one view.
    rgb_objective: ScalarEvidence = _absent()
    #: ``mean|render - gt| / (mean|gt| + eps)``: residual normalized by the
    #: view's own image energy, so a dark frame and a bright frame with the
    #: same relative error compare equally.
    rgb_residual_normalized: ScalarEvidence = _absent()

    # --- Sensor-depth residual (where valid) -------------------------------
    depth_l1_mean_m: ScalarEvidence = _absent("m")
    depth_rmse_m: ScalarEvidence = _absent("m")
    depth_valid_pixel_fraction: ScalarEvidence = _absent()
    depth_valid_pixel_count: ScalarEvidence = _absent("pixels")

    # --- Alpha / visibility coverage ---------------------------------------
    alpha_mean: ScalarEvidence = _absent()
    supported_pixel_fraction: ScalarEvidence = _absent()
    unexplained_pixel_fraction: ScalarEvidence = _absent()
    #: Fraction of pixels whose absolute RGB residual is already below the
    #: probe's ``well_explained_residual_threshold`` -- "area already explained".
    well_explained_pixel_fraction: ScalarEvidence = _absent()
    #: Total pixels this probe actually evaluated (probe resolution, not native).
    probe_pixel_count: ScalarEvidence = _absent("pixels")

    # --- Residual spatial structure ----------------------------------------
    #: Normalized Shannon entropy of the residual mass distribution over pixels
    #: (1.0 = perfectly uniform residual, 0.0 = all residual in one pixel).
    residual_entropy: ScalarEvidence = _absent()
    #: Share of total residual mass carried by the worst 10% of pixels.
    residual_top_decile_mass_fraction: ScalarEvidence = _absent()

    # --- Visible Gaussian support ------------------------------------------
    visible_gaussian_fraction: ScalarEvidence = _absent()
    visible_gaussian_count: ScalarEvidence = _absent("gaussians")

    # --- Gradient probe (issue #803 Phase 3, opt-in channel) ---------------
    #: L2 norm of the exact per-view gradient of the probe objective, taken via
    #: an isolated ``torch.autograd.grad`` -- NEVER an aggregate batch gradient
    #: attributed to one view.
    gradient_norm: ScalarEvidence = _absent()
    gradient_energy: ScalarEvidence = _absent()
    gradient_touched_gaussian_fraction: ScalarEvidence = _absent()
    gradient_energy_means: ScalarEvidence = _absent()
    gradient_energy_opacity: ScalarEvidence = _absent()
    gradient_energy_scale: ScalarEvidence = _absent()
    gradient_energy_rotation: ScalarEvidence = _absent()
    gradient_energy_color: ScalarEvidence = _absent()

    # --- Timing ------------------------------------------------------------
    probe_gpu_ms: ScalarEvidence = _absent("ms")
    probe_wall_ms: ScalarEvidence = _absent("ms")


@dataclass(frozen=True, slots=True)
class FrameEvidence:
    """Canonical evidence envelope consumed by admission and replay policy."""

    identity: EvidenceIdentity
    source: SourceReliabilityEvidence = field(default_factory=SourceReliabilityEvidence)
    probe: GaussianProbeEvidence = field(default_factory=GaussianProbeEvidence)
    rgb_eligible: bool = True
    depth_eligible: bool = True
    hard_reject_reasons: tuple[str, ...] = ()
    soft_reject_reasons: tuple[str, ...] = ()
    escalations: tuple[str, ...] = ()

    def stale_against(self, *, map_version: int, topology_version: int, coordinate_frame_version: int | None = None) -> bool:
        """Return whether this evidence predates any supplied map version."""

        if self.identity.map_version != int(map_version) or self.identity.topology_version != int(topology_version):
            return True
        return coordinate_frame_version is not None and self.identity.coordinate_frame_version != int(coordinate_frame_version)
