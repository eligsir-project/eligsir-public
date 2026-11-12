"""Evidence dataclasses (Wave 3, PR W3-05, plan section 10.3).

The four dataclasses below (``InputEvidence``, ``RenderEvidence``,
``GaussianEvidence``, ``EvidenceSummary``) are reproduced **field-for-field**
from the plan's section 10.3 — later PRs (W3-06 shadow-frame replay, W3-07
admission evidence, W3-08 population-policy evidence, W3-09 learning) import
these exact field names/types, so they are not renamed or reshaped here.

The plan's prose right after the code block ("All evidence carries frame
ID, map version, topology version, and resolution bucket") is not itself
part of the literal dataclass listing, so it is implemented as one
additional field, ``meta: EvidenceMeta``, appended last (with a default) to
each of the four dataclasses — this satisfies the versioning requirement
without touching any of the originally-listed field names, order, or types.
``meta`` is what :mod:`eligsir.mapping.evidence.gaussian_history` /
``frame_history`` use to detect stale (topology-version-mismatched)
per-Gaussian evidence — see those modules' docstrings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True)
class EvidenceMeta:
    """Versioning/identity stamp carried by every evidence record.

    ``frame_id`` identifies which arrived frame this evidence was produced
    from (``-1`` sentinel for evidence not tied to a specific frame, e.g. a
    rolling aggregate). ``map_version``/``topology_version`` mirror
    :class:`eligsir.map.views.GaussianMapReadView`'s own versioning
    convention (same meaning: ``topology_version`` bumps on any structural
    mutation — insert/prune/relocate — ``map_version`` bumps on any change
    including in-place parameter updates). ``resolution_bucket`` mirrors
    :class:`eligsir.optimization.gaussian.contracts.OptimizationView`'s
    field of the same name.
    """

    frame_id: int = -1
    map_version: int = 0
    topology_version: int = 0
    resolution_bucket: str = "default"


@dataclass(frozen=True)
class InputEvidence:
    """Per-frame input/observation quality metrics.

    Characterizes the quality and reliability of the raw input data
    (camera pose, RGB image, depth data, derived geometry estimates).
    """
    pose_confidence: float
    image_quality: float
    depth_valid_fraction: float
    geometry_reliability_mean: float
    meta: EvidenceMeta = field(default_factory=EvidenceMeta)


@dataclass
class RenderEvidence:
    """Per-frame rendering quality metrics from differentiable rendering.

    Carries residual maps and intermediate renders used to assess how well
    the current Gaussian map explains the input (RGB/depth) and how render
    pass transparency is distributed.
    """
    rgb_residual: "torch.Tensor"
    depth_residual: Optional["torch.Tensor"]
    alpha: "torch.Tensor"
    rendered_depth: Optional["torch.Tensor"]
    transmittance: Optional["torch.Tensor"]
    meta: EvidenceMeta = field(default_factory=EvidenceMeta)


@dataclass
class GaussianEvidence:
    """Per-Gaussian reconstruction quality and health indicators.

    Carries per-Gaussian metrics tracking visibility, shape, contribution to
    the render, gradient flow, support confidence, geometric consistency, and
    lifetime in the map. Used to assess individual Gaussian utility for
    mutation decisions (growth, relocation, pruning).
    """
    visibility: "torch.Tensor"
    projected_radius: "torch.Tensor"
    contribution: "torch.Tensor"
    xyz_grad_ema: "torch.Tensor"
    opacity_grad_ema: "torch.Tensor"
    support: "torch.Tensor"
    geometry_confidence: "torch.Tensor"
    free_space_conflict: "torch.Tensor"
    age: "torch.Tensor"
    meta: EvidenceMeta = field(default_factory=EvidenceMeta)


@dataclass(frozen=True)
class EvidenceSummary:
    """Aggregated frame-level evidence summary for policy decisions.

    Summarizes per-frame and per-Gaussian evidence into scalar metrics that
    characterize the current map quality, training progress, and cost. Used
    to feed keyframe selection, admission scoring, and other policy decisions
    that require a high-level view of map state.
    """
    new_surface_ratio: float
    persistent_residual: float
    under_supported_mass: float
    expected_gradient_value: float
    predicted_cost_ms: float
    meta: EvidenceMeta = field(default_factory=EvidenceMeta)
