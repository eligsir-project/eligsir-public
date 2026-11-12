"""Tests for scene-vs-trajectory extent metrics (utils/scene_geometry_metrics).

Pure-numpy + open3d, no GPU/torch required.
"""

from __future__ import annotations

import numpy as np

from eligsir.utils.scene_geometry_metrics import (
    compute_voxel_occupancy, compute_trajectory_extent, compare_scene_vs_trajectory,
    compute_rendered_extent_stats,
)


def _grid_cube(n=10, size=2.0):
    xs = np.linspace(0.0, size, n)
    X, Y, Z = np.meshgrid(xs, xs, xs)
    return np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1).astype(np.float32)


def test_compute_voxel_occupancy_known_bbox():
    pts = _grid_cube(n=10, size=2.0)
    r = compute_voxel_occupancy(pts, voxel_size=0.5)
    assert r["n_points"] == pts.shape[0]
    assert np.allclose(r["bbox_min"], [0.0, 0.0, 0.0], atol=1e-5)
    assert np.allclose(r["bbox_max"], [2.0, 2.0, 2.0], atol=1e-5)
    assert abs(r["bbox_volume_m3"] - 8.0) < 1e-3
    assert r["n_voxels_occupied"] > 0
    assert r["occupied_volume_m3"] > 0.0


def test_compute_voxel_occupancy_empty():
    r = compute_voxel_occupancy(np.zeros((0, 3)), voxel_size=0.1)
    assert r["n_points"] == 0
    assert r["n_voxels_occupied"] == 0


def test_compute_trajectory_extent():
    traj = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 1]], dtype=np.float32)
    r = compute_trajectory_extent(traj)
    assert r["n_positions"] == 4
    assert np.allclose(r["bbox_min"], [0, 0, 0])
    assert np.allclose(r["bbox_max"], [1, 1, 1])
    assert abs(r["bbox_volume_m3"] - 1.0) < 1e-6
    assert np.allclose(r["centroid"], [0.5, 0.5, 0.25])


def test_compare_scene_vs_trajectory_no_plume():
    # Gaussians tightly clustered around the trajectory -- low outside-mass.
    rng = np.random.default_rng(0)
    traj = np.stack([np.linspace(0, 5, 50), np.zeros(50), np.zeros(50)], axis=1)
    pts = traj[rng.integers(0, 50, size=2000)] + rng.normal(scale=0.05, size=(2000, 3))
    opacity = np.full(pts.shape[0], 0.8, dtype=np.float32)

    r = compare_scene_vs_trajectory(pts, opacity, traj, margin_m=1.0)
    assert r["valid"]
    assert r["frac_mass_outside_trajectory_bbox"] < 0.05
    assert r["max_distance_from_trajectory_m"] < 1.0


def test_compare_scene_vs_trajectory_detects_plume():
    rng = np.random.default_rng(1)
    traj = np.stack([np.linspace(0, 5, 50), np.zeros(50), np.zeros(50)], axis=1)
    near = traj[rng.integers(0, 50, size=1000)] + rng.normal(scale=0.05, size=(1000, 3))
    # A far-away plume cluster, well outside the trajectory bbox + margin.
    plume = rng.normal(loc=[100.0, 100.0, 100.0], scale=1.0, size=(200, 3))
    pts = np.concatenate([near, plume], axis=0)
    opacity = np.full(pts.shape[0], 0.8, dtype=np.float32)

    r = compare_scene_vs_trajectory(pts, opacity, traj, margin_m=1.0)
    assert r["valid"]
    # 200/1200 points are plume -> roughly matching opacity-weighted mass fraction.
    assert r["frac_mass_outside_trajectory_bbox"] > 0.1
    assert r["max_distance_from_trajectory_m"] > 50.0
    assert r["bbox_volume_ratio"] > 1.0


def test_compare_scene_vs_trajectory_empty_inputs():
    r = compare_scene_vs_trajectory(np.zeros((0, 3)), np.zeros((0,)), np.zeros((0, 3)))
    assert r["valid"] is False


def test_compare_scene_vs_trajectory_ignores_dead_outlier():
    # A single near-dead (opacity << threshold) outlier far away must NOT
    # blow up max_distance_from_trajectory_m/bbox_volume_ratio -- this is the
    # module's live-mass comparability contract (see module docstring).
    rng = np.random.default_rng(2)
    traj = np.stack([np.linspace(0, 5, 50), np.linspace(0, 1, 50), np.linspace(0, 1, 50)], axis=1)
    near = traj[rng.integers(0, 50, size=1000)] + rng.normal(scale=0.05, size=(1000, 3))
    dead_outlier = np.array([[1000.0, 1000.0, 1000.0]])
    pts = np.concatenate([near, dead_outlier], axis=0)
    opacity = np.concatenate([np.full(1000, 0.8, dtype=np.float32), np.array([0.001], dtype=np.float32)])

    r = compare_scene_vs_trajectory(pts, opacity, traj, margin_m=1.0, opacity_threshold=0.05)
    assert r["valid"]
    assert r["n_total"] == 1001
    assert r["n_live"] == 1000
    assert r["max_distance_from_trajectory_m"] < 1.0
    assert r["bbox_volume_ratio"] < 10.0


# ---------------------------------------------------------------------------
# compute_rendered_extent_stats (#137: scale/quaternion-aware rendered extent)
# ---------------------------------------------------------------------------

_IDENTITY_QUAT = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)  # wxyz


def _uniform_gaussians(n, scale, rng, box=5.0):
    """n identity-rotation Gaussians of a fixed isotropic scale, random centres."""
    means = rng.uniform(-box, box, size=(n, 3)).astype(np.float32)
    scales = np.full((n, 3), scale, dtype=np.float32)
    quats = np.tile(_IDENTITY_QUAT, (n, 1))
    opacities = np.full(n, 0.9, dtype=np.float32)
    return means, scales, quats, opacities


def test_rendered_extent_stats_well_scaled_matches_hand_computation():
    rng = np.random.default_rng(0)
    scale = 0.05
    n = 500
    means, scales, quats, opacities = _uniform_gaussians(n, scale, rng)

    r = compute_rendered_extent_stats(means, scales, quats, opacities, sigma=3.0)
    assert r["valid"]
    assert r["n_total"] == n
    assert r["n_live"] == n

    expected_extent = 3.0 * scale  # 3-sigma extent along the (identical) longest axis
    for key in ("extent_p50_m", "extent_p90_m", "extent_p99_m", "extent_p99.9_m",
                "extent_max_m", "extent_min_m", "extent_mass_weighted_m"):
        assert abs(r[key] - expected_extent) < 1e-5, (key, r[key])

    assert abs(r["median_scale_m"] - scale) < 1e-6
    # threshold defaults: max(1.0m abs, 5x median=0.25m) == 1.0m -> nothing flagged
    assert r["physical_scale_threshold_used_m"] == 1.0
    assert r["count_above_physical_scale_threshold"] == 0

    # Union AABB: identity-rotation isotropic Gaussians -> world half-extent on
    # each axis is exactly `expected_extent`, independent of quaternion.
    for i in range(3):
        half = (r["union_aabb_max"][i] - r["union_aabb_min"][i]) / 2.0
        centre_span_half = (means[:, i].max() - means[:, i].min()) / 2.0
        # union half-size = centre half-span + per-Gaussian half-extent
        assert abs(half - (centre_span_half + expected_extent)) < 1e-4


def test_rendered_extent_stats_oversized_splat_visible_but_typical_preserved():
    rng = np.random.default_rng(1)
    scale = 0.05
    n_normal = 999
    means, scales, quats, opacities = _uniform_gaussians(n_normal, scale, rng)

    # One catastrophically oversized, fully opaque splat (matches the real
    # ScanNet++ finding: ~40m scale opaque Gaussian invisible to a centre bbox).
    outlier_mean = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    outlier_scale = np.array([[40.0, 40.0, 40.0]], dtype=np.float32)
    outlier_quat = _IDENTITY_QUAT.reshape(1, 4)
    outlier_opacity = np.array([0.99], dtype=np.float32)

    means_all = np.concatenate([means, outlier_mean], axis=0)
    scales_all = np.concatenate([scales, outlier_scale], axis=0)
    quats_all = np.concatenate([quats, outlier_quat], axis=0)
    opacities_all = np.concatenate([opacities, outlier_opacity], axis=0)

    r = compute_rendered_extent_stats(means_all, scales_all, quats_all, opacities_all, sigma=3.0)
    assert r["valid"]
    assert r["n_total"] == n_normal + 1

    typical_extent = 3.0 * scale
    outlier_extent = 3.0 * 40.0

    # The typical population's shape must stay visible/representative --
    # p50/p90 must NOT be dragged up by the single outlier.
    assert abs(r["extent_p50_m"] - typical_extent) < 1e-3
    assert abs(r["extent_p90_m"] - typical_extent) < 1e-3
    assert abs(r["median_scale_m"] - scale) < 1e-3

    # But the outlier must be visible somewhere in the robust stats: max and
    # p99.9 (and the union AABB) must reflect it.
    assert abs(r["extent_max_m"] - outlier_extent) < 1e-3
    # p99.9 with n=1000 interpolates between the 2nd-largest and largest
    # sample -- it is pulled meaningfully toward the outlier (unlike p50/p90
    # above) without claiming to equal it exactly.
    assert r["extent_p99.9_m"] > r["extent_p99_m"]
    assert r["extent_p99.9_m"] > 1.5 * typical_extent

    # Union AABB (the scale/quat-aware analogue of a centre-only bbox) is
    # dominated by the one outlier, exactly the failure mode #137 describes:
    # a centre-only bbox over these same means would be ~10m (box=5 -> [-5,5]).
    assert r["union_aabb_max_dim_m"] > 2 * outlier_extent - 1.0

    # Explicit threshold-count check: exactly the one injected splat is flagged.
    assert r["count_above_physical_scale_threshold"] == 1
    assert r["mass_above_physical_scale_threshold"] > 0.0

    # Mass-weighted extent should sit between the typical value and the
    # outlier's own extent -- it is pulled up but not equal to either extreme
    # (opacity-weighted average over ~1000 Gaussians dominated in count by the
    # normal population but with a fully-opaque, huge-extent outlier).
    assert typical_extent < r["extent_mass_weighted_m"] < outlier_extent


def test_rendered_extent_stats_empty_and_all_dead():
    r_empty = compute_rendered_extent_stats(np.zeros((0, 3)), np.zeros((0, 3)),
                                              np.zeros((0, 4)), np.zeros((0,)))
    assert r_empty["valid"] is False
    assert r_empty["n_total"] == 0

    n = 10
    means = np.zeros((n, 3), dtype=np.float32)
    scales = np.full((n, 3), 0.05, dtype=np.float32)
    quats = np.tile(_IDENTITY_QUAT, (n, 1))
    opacities = np.full(n, 0.001, dtype=np.float32)  # all below default threshold
    r_dead = compute_rendered_extent_stats(means, scales, quats, opacities)
    assert r_dead["valid"] is False
    assert r_dead["n_total"] == n
    assert r_dead["n_live"] == 0


def test_rendered_extent_stats_rotation_changes_world_aabb_not_physical_extent():
    """A 45-degree rotation about Z changes the world-space AABB footprint of
    an anisotropic Gaussian but must NOT change its rotation-invariant
    per-Gaussian physical extent (extent = sigma * longest semi-axis)."""
    means = np.zeros((1, 3), dtype=np.float32)
    scales = np.array([[1.0, 0.1, 0.1]], dtype=np.float32)  # long along local x
    opacities = np.array([0.9], dtype=np.float32)

    r_identity = compute_rendered_extent_stats(means, scales, _IDENTITY_QUAT.reshape(1, 4), opacities)

    theta = np.pi / 4.0
    quat_z45 = np.array([[np.cos(theta / 2), 0.0, 0.0, np.sin(theta / 2)]], dtype=np.float32)  # wxyz
    r_rot = compute_rendered_extent_stats(means, scales, quat_z45, opacities)

    # Rotation-invariant physical stats are identical.
    assert abs(r_identity["extent_max_m"] - r_rot["extent_max_m"]) < 1e-5
    assert abs(r_identity["median_scale_m"] - r_rot["median_scale_m"]) < 1e-5

    # World-axis-aligned union AABB footprint differs: identity puts the full
    # 3*1.0=3.0 half-extent on X and 3*0.1=0.3 on Y; a 45-degree Z rotation
    # spreads it more evenly across X and Y.
    ax_id = r_identity["union_aabb_size_m"][0]
    ax_rot = r_rot["union_aabb_size_m"][0]
    ay_id = r_identity["union_aabb_size_m"][1]
    ay_rot = r_rot["union_aabb_size_m"][1]
    assert ax_rot < ax_id - 1e-3
    assert ay_rot > ay_id - 1e-3 and ay_rot != ay_id
