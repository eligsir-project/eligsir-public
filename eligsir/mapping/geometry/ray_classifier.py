"""Per-pixel tri-state ray classification (plan section 13.3/13.6 Subagent B).

:func:`classify_rays` is the actual classifier: given one frame's sensor
depth map and camera pose, it produces a per-pixel
:class:`~eligsir.mapping.geometry.ray_semantics.RayEvidenceKind` map
plus inspectable/counted reasons (deliverable 2), and records this frame's
real (``SURFACE_HIT``) free-space evidence into the shared
:class:`~eligsir.mapping.geometry.free_space_consensus.FreeSpaceConsensusIndex`
so later frames' invalid-depth pixels can query it.

**Deliverable 1, made concrete here:** a pixel with invalid depth is
classified purely from ``consensus_index`` — a *different* structure this
same call never writes to for its own invalid pixels. A single frame,
called in isolation with an empty/fresh consensus index, can therefore never
produce ``CERTIFIED_FREE_SPACE`` for any of its own invalid-depth pixels;
see ``tests/test_ray_classifier.py::test_single_frame_invalid_depth_stays_unknown``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from eligsir.mapping.geometry.free_space_consensus import FreeSpaceConsensusIndex
from eligsir.mapping.geometry.ray_semantics import ClassificationCounts, RayEvidenceKind
from eligsir.utils.rgbd_frames import backproject_depth_pixels_torch

# Encoded as small ints so the per-pixel map is a compact int8 tensor rather
# than a Python-object array; RayEvidenceKind stays the canonical enum for
# all non-tensor call sites.
_KIND_TO_CODE = {
    RayEvidenceKind.SURFACE_HIT: 0,
    RayEvidenceKind.CERTIFIED_FREE_SPACE: 1,
    RayEvidenceKind.UNKNOWN: 2,
}
CODE_TO_KIND = {v: k for k, v in _KIND_TO_CODE.items()}


@dataclass
class RayClassificationResult:
    kind_map: torch.Tensor  # [H, W] int8, values in {0,1,2} per _KIND_TO_CODE
    counts: ClassificationCounts

    def mask_for(self, kind: RayEvidenceKind) -> torch.Tensor:
        return self.kind_map == _KIND_TO_CODE[kind]


def classify_rays(
    sensor_depth: torch.Tensor,
    *,
    c2w: torch.Tensor,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    consensus_index: FreeSpaceConsensusIndex,
    min_views: int,
    max_depth_m: float,
    surface_margin_m: float = 0.05,
    view_uid: int = 0,
    pose_confidence: float = 1.0,
    min_pose_confidence: float = 0.0,
    record_votes: bool = True,
    ray_sample_stride_px: int = 8,
    ray_samples_per_pixel: int = 4,
    ray_sample_near_m: float = 0.2,
) -> RayClassificationResult:
    """Classify every pixel of one depth frame into SURFACE_HIT /
    CERTIFIED_FREE_SPACE / UNKNOWN.

    Parameters
    ----------
    sensor_depth : [H, W] float tensor. Invalid pixels are ``<= 0`` or
        non-finite (NaN/inf) — the same convention already used throughout
        this repo's streaming depth pipeline.
    c2w : [4, 4] camera-to-world transform for this frame.
    fx, fy, cx, cy : depth-camera intrinsics.
    consensus_index : the shared, run-lifetime free-space consensus store.
        Read (to certify invalid pixels) AND written (to record this
        frame's own valid-depth free-space evidence for *future* frames) —
        never written to certify *this* frame's own invalid pixels, which
        is exactly what keeps deliverable 1 true.
    min_views : corroborating-view threshold (mirrors
        ``RaySemanticsPolicy.min_views`` / ``--streaming_ray_semantics_min_views``).
    max_depth_m : sensor documented max usable range.
    surface_margin_m : kept clear of the measured hit when voting the
        traversed-before-hit segment as free space (avoid voting the
        surface voxel itself).
    view_uid : unique id for this frame/view (dedup key in the consensus
        store, and — via the store's baseline gate — the granularity at
        which "distinct view" is judged).
    pose_confidence : this frame's pose-confidence signal (``InputEvidence
        .pose_confidence`` when available; ``1.0`` = trust unconditionally,
        matching current behaviour for sources that don't populate it yet).
    min_pose_confidence : frames below this confidence contribute no votes
        and classify their own invalid pixels as UNKNOWN
        (reason ``low_pose_confidence``) regardless of consensus state.
    record_votes : set False to classify without mutating
        ``consensus_index`` (e.g. for a dry-run/report-only evaluation pass).
    ray_sample_stride_px : only every Nth row/column of SURFACE_HIT pixels
        is used as a ray-segment source (cost control -- see below).
    ray_samples_per_pixel : number of points sampled along each selected
        ray's traversed-before-hit segment (near range to just before the
        hit), not just the single point closest to the hit. This is what
        makes ``ray_semantics.py`` rule 1 ("every point strictly closer than
        the hit is real evidence") actually reachable by a later frame's
        query: a query at ``max_depth_m`` can only be corroborated if some
        earlier vote landed in the SAME coarse voxel, and a single
        near-the-hit point per ray makes that a near-zero-probability
        coincidence for realistic corridor geometry (verified empirically --
        see ``tests/test_ray_semantics_classification.py::test_realistic_corridor_requires_segment_sampling``).
        Sampling the segment turns "another view happened to measure almost
        exactly this point" into "another view's ray passed anywhere near
        this point on its way to a farther hit", which is the actual claim
        rule 1 makes.
    ray_sample_near_m : nearest sample distance along the ray (avoids
        voting degenerate near-zero-depth points).
    """
    device = sensor_depth.device
    h, w = sensor_depth.shape[-2:]
    kind_map = torch.full((h, w), _KIND_TO_CODE[RayEvidenceKind.UNKNOWN], dtype=torch.int8, device=device)
    counts = ClassificationCounts()

    valid = torch.isfinite(sensor_depth) & (sensor_depth > 0) & (sensor_depth <= float(max_depth_m))
    kind_map[valid] = _KIND_TO_CODE[RayEvidenceKind.SURFACE_HIT]
    counts.surface_hit = int(valid.sum().item())

    low_conf = pose_confidence < float(min_pose_confidence)

    ys, xs = torch.meshgrid(
        torch.arange(h, device=device, dtype=torch.float32),
        torch.arange(w, device=device, dtype=torch.float32),
        indexing="ij",
    )

    if valid.any() and not low_conf and record_votes:
        ys_valid = ys[valid]
        xs_valid = xs[valid]
        z_hit = sensor_depth[valid]
        stride = max(1, int(ray_sample_stride_px))
        if stride > 1:
            stride_mask = (ys_valid.long() % stride == 0) & (xs_valid.long() % stride == 0)
            ys_valid = ys_valid[stride_mask]
            xs_valid = xs_valid[stride_mask]
            z_hit = z_hit[stride_mask]
        if z_hit.numel() > 0:
            k = max(1, int(ray_samples_per_pixel))
            z_max = (z_hit - float(surface_margin_m)).clamp(min=0)
            # k evenly-spaced fractions of the traversed-before-hit segment,
            # excluding the very near/far ends (0 and 1) -- see docstring:
            # this is what makes rule 1 ("every point before the hit is real
            # evidence") actually corroborable by a later frame's single
            # far-range query point, instead of only the single
            # closest-to-the-hit point.
            fracs = torch.linspace(1.0 / (k + 1), k / (k + 1), steps=k, device=device)
            z_samples = z_max.unsqueeze(1) * fracs.unsqueeze(0)  # [n_sel, k]
            keep = z_samples > float(ray_sample_near_m)
            if keep.any():
                rep_x = xs_valid.unsqueeze(1).expand(-1, k)[keep]
                rep_y = ys_valid.unsqueeze(1).expand(-1, k)[keep]
                z_flat = z_samples[keep]
                pts_free, _ = backproject_depth_pixels_torch(
                    z_flat, rep_x, rep_y, fx, fy, cx, cy, c2w,
                    torch.zeros((h, w, 3), device=device), (h, w),
                )
                cam_center = c2w[:3, 3]
                consensus_index.vote_free(pts_free, view_uid=view_uid, camera_center=cam_center)

    invalid = ~valid
    n_invalid = int(invalid.sum().item())
    if n_invalid:
        if low_conf:
            counts.unknown = n_invalid
            counts.unknown_reason_low_pose_confidence = n_invalid
        else:
            z_far = torch.full((n_invalid,), float(max_depth_m), device=device)
            pts_far, _ = backproject_depth_pixels_torch(
                z_far, xs[invalid], ys[invalid], fx, fy, cx, cy, c2w,
                torch.zeros((h, w, 3), device=device), (h, w),
            )
            vote_counts = consensus_index.vote_counts(pts_far)
            certified = vote_counts >= int(min_views)
            uncorroborated = vote_counts == 0
            below_min = (~certified) & (~uncorroborated)

            flat_kind = kind_map[invalid]
            flat_kind[certified] = _KIND_TO_CODE[RayEvidenceKind.CERTIFIED_FREE_SPACE]
            kind_map[invalid] = flat_kind

            counts.certified_free_space = int(certified.sum().item())
            counts.unknown = int((~certified).sum().item())
            counts.unknown_reason_invalid_depth_uncorroborated = int(uncorroborated.sum().item())
            counts.unknown_reason_below_min_views = int(below_min.sum().item())

    return RayClassificationResult(kind_map=kind_map, counts=counts)
