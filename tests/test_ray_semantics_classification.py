"""Unit tests for tri-state ray semantics (Wave 3, PR W3-08, plan section 13).

Pure-tensor / CPU logic only (no GPU, no training loop) — mirrors the style
of ``tests/test_floater_consensus.py``.

Deliverable 1 ("invalid depth is never automatically certified free space")
is the load-bearing assertion here: :func:`test_single_frame_invalid_depth_stays_unknown`
constructs a fresh, empty consensus index and shows that classifying one
frame's own invalid-depth pixel — with NO other frame's evidence recorded —
can never produce ``CERTIFIED_FREE_SPACE``, regardless of how the pixel is
positioned. :func:`test_multiview_corroboration_certifies_free_space` then
shows the *positive* case: once >= ``min_views`` distinct, adequately
separated frames have directly measured that same 3-D region as free space
(a real ``SURFACE_HIT`` ray traversal, not another frame's invalid-depth
guess), the region legitimately becomes ``CERTIFIED_FREE_SPACE`` — this is
the floor2-corridor fix CLAUDE.md's "streaming_carve_invalid_depth vs.
long-corridor scenes" section describes as missing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eligsir.mapping.geometry.free_space_consensus import FreeSpaceConsensusIndex  # noqa: E402
from eligsir.mapping.geometry.ray_classifier import classify_rays  # noqa: E402
from eligsir.mapping.geometry.ray_semantics import RayEvidenceKind  # noqa: E402

_FX = _FY = 50.0
_CX = _CY = 0.5  # single-pixel image, principal point == the only pixel


def _c2w_at_z(z: float) -> torch.Tensor:
    """Camera translated along world +z (down a corridor), identity rotation."""
    m = torch.eye(4)
    m[2, 3] = z
    return m


def _fresh_index(voxel_size_m: float = 0.5, min_baseline_m: float = 0.5) -> FreeSpaceConsensusIndex:
    return FreeSpaceConsensusIndex(voxel_size_m=voxel_size_m, min_baseline_m=min_baseline_m)


def test_single_frame_invalid_depth_stays_unknown():
    """DELIVERABLE 1. An empty consensus index (no other frame's evidence)
    must classify an invalid-depth pixel as UNKNOWN, never CERTIFIED_FREE_SPACE,
    no matter the geometry."""
    idx = _fresh_index()
    sensor = torch.zeros(1, 1)  # invalid depth
    result = classify_rays(
        sensor, c2w=_c2w_at_z(0.0), fx=_FX, fy=_FY, cx=_CX, cy=_CY,
        consensus_index=idx, min_views=1, max_depth_m=5.0, view_uid=0,
    )
    assert result.counts.certified_free_space == 0
    assert result.counts.unknown == 1
    assert result.kind_map[0, 0].item() == 2  # UNKNOWN code
    assert result.mask_for(RayEvidenceKind.CERTIFIED_FREE_SPACE).sum().item() == 0


def test_single_frame_carving_would_have_certified_but_does_not():
    """Same as above but repeated across a larger invalid-depth region and a
    higher min_views — reproduces the shape of the real floor2 bug (a big
    contiguous invalid-depth patch) at unit-test scale."""
    idx = _fresh_index()
    sensor = torch.zeros(4, 4)
    result = classify_rays(
        sensor, c2w=_c2w_at_z(0.0), fx=_FX, fy=_FY, cx=_CX, cy=_CY,
        consensus_index=idx, min_views=3, max_depth_m=5.0, view_uid=0,
    )
    assert result.counts.certified_free_space == 0
    assert result.counts.unknown == 16
    assert result.counts.unknown_reason_invalid_depth_uncorroborated == 16


def test_multiview_corroboration_certifies_free_space():
    """POSITIVE case: >= min_views distinct, baseline-separated frames whose
    own valid depth directly measured the same 3-D region as free space
    (real SURFACE_HIT evidence, not another invalid-depth guess) certify it."""
    idx = _fresh_index(voxel_size_m=0.5, min_baseline_m=0.5)
    sensor_a = torch.zeros(1, 1)

    # Before any corroboration: UNKNOWN (deliverable 1, again).
    before = classify_rays(
        sensor_a, c2w=_c2w_at_z(0.0), fx=_FX, fy=_FY, cx=_CX, cy=_CY,
        consensus_index=idx, min_views=2, max_depth_m=5.0, view_uid=0,
    )
    assert before.counts.certified_free_space == 0

    # Two other keyframes further down the corridor directly measure a real
    # surface beyond frame A's own 5m range, each voting the (same) traversed
    # free-space voxel just before their own hit.
    margin = 0.05
    target_world_z = 5.25
    for cam_z, uid in ((4.0, 1), (4.6, 2)):
        hit_local = (target_world_z - cam_z) + margin
        sensor_hit = torch.full((1, 1), float(hit_local))
        classify_rays(
            sensor_hit, c2w=_c2w_at_z(cam_z), fx=_FX, fy=_FY, cx=_CX, cy=_CY,
            consensus_index=idx, min_views=2, max_depth_m=5.0, view_uid=uid,
        )

    after = classify_rays(
        sensor_a, c2w=_c2w_at_z(0.0), fx=_FX, fy=_FY, cx=_CX, cy=_CY,
        consensus_index=idx, min_views=2, max_depth_m=5.0, view_uid=3,
    )
    assert after.counts.certified_free_space == 1
    assert after.mask_for(RayEvidenceKind.CERTIFIED_FREE_SPACE)[0, 0].item() is True


def test_below_min_views_stays_unknown_with_reason():
    """A single corroborating vote (below min_views=2) must NOT certify —
    and the reason must be distinguishable from "no votes at all"
    (deliverable 2: classification reason is inspectable and counted)."""
    idx = _fresh_index(voxel_size_m=0.5, min_baseline_m=0.5)
    sensor_a = torch.zeros(1, 1)
    margin = 0.05
    target_world_z = 5.25
    hit_local = (target_world_z - 4.0) + margin
    classify_rays(
        torch.full((1, 1), float(hit_local)), c2w=_c2w_at_z(4.0),
        fx=_FX, fy=_FY, cx=_CX, cy=_CY,
        consensus_index=idx, min_views=2, max_depth_m=5.0, view_uid=1,
    )
    result = classify_rays(
        sensor_a, c2w=_c2w_at_z(0.0), fx=_FX, fy=_FY, cx=_CX, cy=_CY,
        consensus_index=idx, min_views=2, max_depth_m=5.0, view_uid=2,
    )
    assert result.counts.certified_free_space == 0
    assert result.counts.unknown_reason_below_min_views == 1
    assert result.counts.unknown_reason_invalid_depth_uncorroborated == 0


def test_baseline_gate_rejects_near_duplicate_viewpoints():
    """Two votes from nearly the same camera position must not double-count
    toward min_views (mirrors streaming_floater_consensus_baseline_gate's
    rationale — mm-apart cameras can't independently disambiguate depth)."""
    idx = _fresh_index(voxel_size_m=0.5, min_baseline_m=0.5)
    margin = 0.05
    target_world_z = 5.25
    for cam_z, uid in ((4.0, 1), (4.05, 2)):  # only 5cm apart, below 0.5m gate
        hit_local = (target_world_z - cam_z) + margin
        classify_rays(
            torch.full((1, 1), float(hit_local)), c2w=_c2w_at_z(cam_z),
            fx=_FX, fy=_FY, cx=_CX, cy=_CY,
            consensus_index=idx, min_views=2, max_depth_m=5.0, view_uid=uid,
        )
    result = classify_rays(
        torch.zeros(1, 1), c2w=_c2w_at_z(0.0), fx=_FX, fy=_FY, cx=_CX, cy=_CY,
        consensus_index=idx, min_views=2, max_depth_m=5.0, view_uid=3,
    )
    assert result.counts.certified_free_space == 0
    assert result.counts.unknown_reason_below_min_views == 1


def test_low_pose_confidence_frame_never_votes_or_gets_certified():
    idx = _fresh_index()
    result = classify_rays(
        torch.zeros(1, 1), c2w=_c2w_at_z(0.0), fx=_FX, fy=_FY, cx=_CX, cy=_CY,
        consensus_index=idx, min_views=1, max_depth_m=5.0, view_uid=0,
        pose_confidence=0.1, min_pose_confidence=0.5,
    )
    assert result.counts.unknown_reason_low_pose_confidence == 1
    assert result.counts.certified_free_space == 0
    assert len(idx) == 0


def test_realistic_corridor_requires_segment_sampling():
    """Discriminator test (added after code review): the earlier
    single-point-per-ray version of ``FreeSpaceConsensusIndex.vote_free``
    voted only ``z_hit - surface_margin_m`` -- one point right before each
    ray's own hit. That makes corroboration a near-zero-probability
    coincidence on REALISTIC corridor geometry (two different rays' single
    sample points would have to land in the exact same coarse voxel), even
    though the earlier hand-solved test (`test_multiview_corroboration_
    certifies_free_space`, which computes ``hit_local`` by SOLVING for the
    exact depth that makes the voxels coincide) passed regardless.

    This test uses geometry that is NOT back-solved from the query voxel:
    three cameras at plausible, evenly-spaced corridor positions, each
    measuring the SAME plausible local hit distance (as they would
    approaching the same far wall), and checks whether classify_rays's
    actual (segment-sampling) ray_samples_per_pixel > 1 default corroborates
    frame A's query -- and that reducing to ray_samples_per_pixel=1
    (the old behaviour) does NOT, on this same non-contrived geometry.
    """
    fx = fy = 500.0
    cx = cy = 0.5

    def c2w(z):
        m = torch.eye(4)
        m[2, 3] = z
        return m

    hit_local = 2.5  # plausible local hit distance, NOT solved from a target voxel
    cam_positions = (3.0, 3.6, 4.2)  # evenly spaced, not back-solved either

    # --- Segment sampling (this PR's actual default): corroborates. -------
    idx_seg = FreeSpaceConsensusIndex(voxel_size_m=0.5, min_baseline_m=0.5)
    for uid, cam_z in enumerate(cam_positions, start=1):
        classify_rays(
            torch.full((1, 1), hit_local), c2w=c2w(cam_z), fx=fx, fy=fy, cx=cx, cy=cy,
            consensus_index=idx_seg, min_views=2, max_depth_m=5.0, view_uid=uid,
            ray_samples_per_pixel=4, ray_sample_stride_px=1,
        )
    res_seg = classify_rays(
        torch.zeros(1, 1), c2w=c2w(0.0), fx=fx, fy=fy, cx=cx, cy=cy,
        consensus_index=idx_seg, min_views=2, max_depth_m=5.0, view_uid=99,
        ray_samples_per_pixel=4, ray_sample_stride_px=1,
    )
    assert res_seg.counts.certified_free_space == 1, (
        "segment sampling should corroborate this realistic (non-contrived) corridor geometry"
    )

    # --- Single point per ray (the old, inert behaviour): does NOT. -------
    idx_single = FreeSpaceConsensusIndex(voxel_size_m=0.5, min_baseline_m=0.5)
    for uid, cam_z in enumerate(cam_positions, start=1):
        classify_rays(
            torch.full((1, 1), hit_local), c2w=c2w(cam_z), fx=fx, fy=fy, cx=cx, cy=cy,
            consensus_index=idx_single, min_views=2, max_depth_m=5.0, view_uid=uid,
            ray_samples_per_pixel=1, ray_sample_stride_px=1,
        )
    res_single = classify_rays(
        torch.zeros(1, 1), c2w=c2w(0.0), fx=fx, fy=fy, cx=cx, cy=cy,
        consensus_index=idx_single, min_views=2, max_depth_m=5.0, view_uid=99,
        ray_samples_per_pixel=1, ray_sample_stride_px=1,
    )
    assert res_single.counts.certified_free_space == 0, (
        "single-point-per-ray voting should NOT corroborate on this geometry -- "
        "demonstrates why segment sampling is necessary, not just cosmetic"
    )


def test_surface_hit_valid_depth_classified_and_no_consensus_needed():
    idx = _fresh_index()
    result = classify_rays(
        torch.full((2, 2), 1.5), c2w=_c2w_at_z(0.0), fx=_FX, fy=_FY, cx=_CX, cy=_CY,
        consensus_index=idx, min_views=99, max_depth_m=5.0, view_uid=0,
    )
    assert result.counts.surface_hit == 4
    assert result.counts.unknown == 0
    assert (result.kind_map == 0).all()
