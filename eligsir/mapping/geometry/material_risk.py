"""Material-risk heuristics (plan section 13.4/13.6 Subagent D).

Cheap, per-pixel ``[0,1]`` geometry-reliability signal and a boolean
"material risk" mask (glass/specular/uncertain) built from it. This is a
first-pass implementation of the reliability-map inputs listed in plan
section 13.4:

- depth validity (implemented — from the tri-state classification);
- local depth discontinuity (implemented);
- saturation and extreme intensity (implemented);
- temporal RGB consistency (NOT implemented here — needs a previous-frame
  reference, left as a caller-supplied optional term);
- temporal depth consistency (NOT implemented here, same reason);
- view-dependent appearance change (NOT implemented — needs multi-view
  history, out of this module's single-frame scope);
- multi-view support (implemented — delegates to the
  ``FreeSpaceConsensusIndex`` vote count / the classifier's
  ``CERTIFIED_FREE_SPACE`` vs. ``UNKNOWN`` split);
- pose confidence (implemented — scalar passed straight through).

The reliability map feeds depth-loss weighting, insertion confidence, and
free-space evidence per section 13.4; it is intentionally a plain float
tensor (not a new evidence dataclass) so callers can multiply it directly
into existing loss terms without a shape/field migration.
"""

from __future__ import annotations

import torch


def geometry_reliability_map(
    kind_map: torch.Tensor,
    sensor_depth: torch.Tensor,
    rgb: torch.Tensor,
    *,
    pose_confidence: float = 1.0,
    saturation_thresh: float = 0.98,
    depth_discontinuity_thresh_m: float = 0.15,
) -> torch.Tensor:
    """Return a ``[H, W]`` float tensor in ``[0, 1]``.

    ``kind_map`` is the ``RayClassificationResult.kind_map`` int8 tensor
    (0=SURFACE_HIT, 1=CERTIFIED_FREE_SPACE, 2=UNKNOWN — see
    ``ray_classifier.CODE_TO_KIND``). ``sensor_depth`` is the raw ``[H,W]``
    depth map (same convention as ``ray_classifier.classify_rays``).
    ``rgb`` is ``[3, H, W]`` in ``[0, 1]``.
    """
    h, w = sensor_depth.shape[-2:]
    device = sensor_depth.device
    reliability = torch.zeros((h, w), dtype=torch.float32, device=device)

    surface_hit = kind_map == 0
    certified_free = kind_map == 1
    reliability[surface_hit] = 1.0
    reliability[certified_free] = 0.7  # corroborated, but not a direct hit
    # UNKNOWN stays 0.0 unless later penalized further below.

    # Local depth discontinuity: a valid pixel whose 4-neighbour depth jumps
    # by more than the threshold is a depth-edge — lower reliability even
    # though it counted as SURFACE_HIT (occlusion boundaries are exactly
    # where per-pixel depth is least trustworthy).
    if h > 2 and w > 2:
        valid_f = (sensor_depth > 0).float()
        d = sensor_depth.clamp(min=0)
        dx = (d[:, 1:] - d[:, :-1]).abs()
        dy = (d[1:, :] - d[:-1, :]).abs()
        edge = torch.zeros((h, w), dtype=torch.bool, device=device)
        edge[:, 1:] |= dx > depth_discontinuity_thresh_m
        edge[:, :-1] |= dx > depth_discontinuity_thresh_m
        edge[1:, :] |= dy > depth_discontinuity_thresh_m
        edge[:-1, :] |= dy > depth_discontinuity_thresh_m
        edge &= valid_f.bool()
        reliability[edge] *= 0.5

    # Saturation / extreme intensity: near-pure-white or near-pure-black
    # pixels are unreliable for both appearance and (if it happened to be
    # SURFACE_HIT) geometry confidence, since these are the classic
    # specular-highlight / underexposed-shadow failure modes.
    intensity = rgb.mean(dim=0)
    extreme = (intensity > saturation_thresh) | (intensity < (1.0 - saturation_thresh))
    reliability[extreme] *= 0.3

    # Pose confidence scales everything (a frame we don't trust the pose of
    # can't be trusted for geometry regardless of per-pixel signal).
    reliability = reliability * float(max(0.0, min(1.0, pose_confidence)))

    return reliability.clamp(0.0, 1.0)


def material_risk_mask(
    kind_map: torch.Tensor,
    reliability: torch.Tensor,
    *,
    reliability_thresh: float = 0.35,
) -> torch.Tensor:
    """Boolean ``[H, W]`` mask: True where geometry should not be trusted but
    an appearance-only update may still be worthwhile (UNKNOWN classification
    AND low reliability — a pixel could be UNKNOWN purely from an
    under-observed region rather than a true material-risk failure, so both
    conditions are required)."""
    unknown = kind_map == 2
    return unknown & (reliability < float(reliability_thresh))
