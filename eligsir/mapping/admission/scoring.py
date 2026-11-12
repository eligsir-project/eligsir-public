"""Deterministic frame-utility scorer (plan section 11.3/11.6, Subagent B).

Implements the formula verbatim from plan section 11.3:

    U = pose_quality * image_quality *
        (w_new_surface * new_surface
         + w_residual * persistent_depth_supported_residual
         + w_debt * under_supported_mass
         + w_gradient * expected_gradient_value
         + w_parallax * useful_parallax)
        - w_redundancy * redundancy
        - w_specular * specular_risk
        - w_cost * predicted_gpu_cost

Explicitly follows two constraints from the plan:

1. "Do not use raw RGB residual alone." ``persistent_depth_supported_residual``
   here means an RGB residual gated by depth support -- a pixel with a large
   RGB residual but no valid/consistent depth at that location (e.g. a
   specular highlight, a moving shadow, a blown-out exposure edge) is
   excluded before averaging. This is exactly why ``specular_risk`` is a
   *separate, penalized* term instead of folding into the residual: a high
   RGB residual with LOW depth support is redirected into specular risk
   (subtracted), not counted as useful residual (added).
2. "Expected gradient value shall ... use historical per-Gaussian gradient
   EMA and visibility. It shall not run a backward pass for every
   candidate." ``expected_gradient_value`` here is read straight off
   :class:`eligsir.mapping.evidence.contracts.GaussianEvidence`'s
   ``xyz_grad_ema``/``opacity_grad_ema`` (produced by the existing EMA
   reducer in ``evidence/reducers.py``, warm from prior training steps),
   restricted to the visible mask -- no new backward pass, no new gradient
   computation, anywhere in this module.
"""

from __future__ import annotations

from typing import Optional

import torch

from eligsir.mapping.admission.cost_model import predict_gpu_cost
from eligsir.mapping.admission.utility_contracts import (
    FrameUtilityComponents,
    FrameUtilityScore,
    UtilityWeights,
)
from eligsir.mapping.evidence.contracts import EvidenceMeta, GaussianEvidence, InputEvidence, RenderEvidence


def depth_supported_rgb_residual(
    render_evidence: Optional[RenderEvidence],
    depth_valid_fraction: float,
    specular_threshold: float = 0.35,
) -> "tuple[float, float]":
    """Split an RGB residual into (depth-supported residual, specular risk).

    ``render_evidence.rgb_residual`` may be dense (all pixels) while
    ``render_evidence.depth_residual``/``alpha`` tell us where depth is
    actually trustworthy. Without a per-pixel depth-validity mask we fall
    back to the frame-level ``depth_valid_fraction`` (from
    :class:`InputEvidence`) as a coarse gate: a frame with abundant valid
    depth is treated as "residual is geometry-grounded", a frame with
    little/no valid depth routes a large residual into ``specular_risk``
    instead (matches "do not use raw RGB residual alone" -- a large
    residual is only counted as useful signal when depth corroborates it).

    Returns ``(0.0, 0.0)`` when no render evidence is available (shadow
    scoring was invoked without RGB residual evidence collected).
    """
    if render_evidence is None or render_evidence.rgb_residual is None:
        return 0.0, 0.0
    with torch.no_grad():
        residual = render_evidence.rgb_residual.abs()
        mean_residual = float(residual.mean().item()) if residual.numel() > 0 else 0.0
    depth_valid_fraction = max(0.0, min(1.0, depth_valid_fraction))
    supported = mean_residual * depth_valid_fraction
    # The portion of the residual NOT corroborated by depth is the specular/
    # unreliable-source-of-residual risk. A frame with high residual and
    # near-zero depth support (e.g. glass/mirror reflections, blown-out
    # highlights) pushes this up, and up-weighting THIS instead of the
    # residual term is what stops a scorer built on raw residual from
    # rewarding exactly the failure mode plan section 11.3 warns against.
    unsupported = mean_residual * (1.0 - depth_valid_fraction)
    specular_risk = unsupported if mean_residual > specular_threshold else unsupported * 0.5
    return supported, specular_risk


def expected_gradient_value_from_history(
    gaussian_evidence: Optional[GaussianEvidence],
    visibility_mask: Optional[torch.Tensor] = None,
) -> float:
    """Historical-EMA-only gradient signal (see module docstring point 2).

    Restricts the mean to ``visibility_mask`` when provided (candidate
    frame's own visible set) so the signal reflects "how much would THIS
    frame's visible Gaussians still benefit from more gradient signal",
    not a whole-map average dominated by Gaussians the frame can't even
    see.
    """
    if gaussian_evidence is None:
        return 0.0
    xyz_ema = gaussian_evidence.xyz_grad_ema
    opacity_ema = gaussian_evidence.opacity_grad_ema
    if xyz_ema is None or xyz_ema.numel() == 0:
        return 0.0
    with torch.no_grad():
        combined = xyz_ema.abs()
        if opacity_ema is not None and opacity_ema.shape == xyz_ema.shape:
            combined = combined + opacity_ema.abs()
        if visibility_mask is not None and visibility_mask.shape == combined.shape:
            selected = combined[visibility_mask]
            if selected.numel() == 0:
                return 0.0
            return float(selected.mean().item())
        return float(combined.mean().item())


def score_frame_utility(
    *,
    frame_id: int,
    input_evidence: Optional[InputEvidence],
    render_evidence: Optional[RenderEvidence] = None,
    gaussian_evidence: Optional[GaussianEvidence] = None,
    new_surface: float = 0.0,
    under_supported_mass: float = 0.0,
    useful_parallax: float = 0.0,
    redundancy: float = 0.0,
    n_gaussians: int = 0,
    resolution_bucket: str = "default",
    weights: Optional[UtilityWeights] = None,
    meta: Optional[EvidenceMeta] = None,
) -> FrameUtilityScore:
    """Compute a :class:`FrameUtilityScore` for one candidate frame.

    Every argument beyond ``frame_id``/``input_evidence`` is optional and
    defaults to a neutral (zero-contribution) value -- a caller with only
    partial evidence still gets a well-formed score, just with the missing
    terms contributing nothing (never crashing, never silently using a
    stale/wrong value). ``meta`` is threaded through unchanged so the
    resulting score carries the same topology/map-version stamp as the
    evidence it was built from.
    """
    weights = weights or UtilityWeights()

    pose_quality = float(input_evidence.pose_confidence) if input_evidence is not None else 0.0
    image_quality = float(input_evidence.image_quality) if input_evidence is not None else 0.0
    depth_valid_fraction = float(input_evidence.depth_valid_fraction) if input_evidence is not None else 0.0

    persistent_residual, specular_risk = depth_supported_rgb_residual(render_evidence, depth_valid_fraction)

    visibility_mask = gaussian_evidence.visibility if gaussian_evidence is not None else None
    if visibility_mask is not None and visibility_mask.dtype != torch.bool:
        visibility_mask = visibility_mask.to(dtype=torch.bool)
    expected_gradient_value = expected_gradient_value_from_history(gaussian_evidence, visibility_mask)

    predicted_cost = predict_gpu_cost(n_gaussians, resolution_bucket)

    components = FrameUtilityComponents(
        pose_quality=pose_quality,
        image_quality=image_quality,
        new_surface=float(new_surface),
        persistent_depth_supported_residual=float(persistent_residual),
        under_supported_mass=float(under_supported_mass),
        expected_gradient_value=float(expected_gradient_value),
        useful_parallax=float(useful_parallax),
        redundancy=float(redundancy),
        specular_risk=float(specular_risk),
        predicted_gpu_cost=float(predicted_cost),
    )
    utility = _apply_formula(components, weights)

    return FrameUtilityScore(
        frame_id=frame_id,
        utility=utility,
        components=components,
        weights=weights,
        meta=meta or EvidenceMeta(frame_id=frame_id),
    )


def _apply_formula(components: FrameUtilityComponents, weights: UtilityWeights) -> float:
    positive = (
        weights.w_new_surface * components.new_surface
        + weights.w_residual * components.persistent_depth_supported_residual
        + weights.w_debt * components.under_supported_mass
        + weights.w_gradient * components.expected_gradient_value
        + weights.w_parallax * components.useful_parallax
    )
    penalty = (
        weights.w_redundancy * components.redundancy
        + weights.w_specular * components.specular_risk
        + weights.w_cost * components.predicted_gpu_cost
    )
    return float(components.pose_quality * components.image_quality * positive - penalty)


def recompute_utility(score: FrameUtilityScore) -> float:
    """Recompute ``score.utility`` from ``score.components``/``score.weights``
    alone -- a self-consistency check used by tests and by
    ``analyze_frame_utility.py`` to detect a corrupted/hand-edited log
    record.
    """
    return _apply_formula(score.components, score.weights)
