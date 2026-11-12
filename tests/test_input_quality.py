"""Tests for the input-quality metric: sensor-noise vs tracking-error split.

Decomposition identity: surface_thickness^2 ~ sensor_noise^2 + tracking_thickness^2.
"""

from __future__ import annotations

import numpy as np

from eligsir.utils.scene_geometry_metrics import input_quality_stats


def test_tracking_unavailable_for_gt_poses():
    out = input_quality_stats(sensor_noise_m=0.004, sensor_n=10000,
                              odom_stats=None, surface_thickness_m=0.005,
                              complexity_dim=2.02, pose_source="precomputed")
    assert out["tracking_available"] is False
    assert out["tracking_inlier_rmse_m"] is None
    assert out["quality_verdict"] == "clean"  # dim_excess 0.02 < 0.15


def test_tracking_aggregation_from_odom_stats():
    odom = [
        {"valid": True, "inlier_rmse": 0.02, "fitness": 0.8},
        {"valid": True, "inlier_rmse": 0.04, "fitness": 0.6},
        {"valid": False, "inlier_rmse": None, "fitness": 0.0},
    ]
    out = input_quality_stats(sensor_noise_m=0.005, sensor_n=5000, odom_stats=odom,
                              surface_thickness_m=0.05, complexity_dim=2.6,
                              pose_source="open3d_odometry_live")
    assert out["tracking_available"] is True
    assert abs(out["tracking_inlier_rmse_m"] - 0.03) < 1e-9        # mean(0.02,0.04)
    assert abs(out["tracking_fail_frac"] - 1.0 / 3.0) < 1e-9
    assert out["tracking_n_edges"] == 3


def test_decomposition_identity():
    out = input_quality_stats(sensor_noise_m=0.03, surface_thickness_m=0.05,
                              complexity_dim=2.5)
    # tracking_thickness = sqrt(0.05^2 - 0.03^2) = 0.04
    assert abs(out["tracking_thickness_m"] - 0.04) < 1e-6
    assert abs(out["dim_excess"] - 0.5) < 1e-9


def test_verdict_sensor_vs_tracking_limited():
    odom = [{"valid": True, "inlier_rmse": 0.02, "fitness": 0.7}]
    tracking = input_quality_stats(sensor_noise_m=0.01, surface_thickness_m=0.05,
                                   complexity_dim=2.6, odom_stats=odom)  # nst=.049 >> sensor
    assert tracking["quality_verdict"] == "tracking-limited"

    sensor = input_quality_stats(sensor_noise_m=0.05, surface_thickness_m=0.051,
                                 complexity_dim=2.6, odom_stats=odom)   # nst small, sensor dominates
    assert sensor["quality_verdict"] == "sensor-limited"


def test_gt_poses_never_tracking_limited():
    # No odometry (GT/precomputed): even with high non-sensor thickening (curvature),
    # the verdict must not be 'tracking-limited' — there is no tracking by construction.
    out = input_quality_stats(sensor_noise_m=0.01, surface_thickness_m=0.05,
                              complexity_dim=2.6, odom_stats=None, pose_source="precomputed")
    assert out["tracking_available"] is False
    assert out["quality_verdict"] == "sensor-limited"
    assert out["nonsensor_thickness_m"] is not None  # still reported (upper bound)


def test_thickness_le_sensor_gives_zero_tracking():
    out = input_quality_stats(sensor_noise_m=0.05, surface_thickness_m=0.03,
                              complexity_dim=2.4)
    assert out["tracking_thickness_m"] == 0.0  # clamped, not negative/NaN


def test_pca_residual_tracks_sensor_noise():
    """A clean frontal plane -> tiny plane-fit residual; adding depth jitter -> larger."""
    import torch  # noqa: F401  (PCA path uses torch)
    from geometry.normals import NormalEstimationConfig, estimate_depth_normals_pca

    H = W = 80
    fx = fy = 100.0
    cx = cy = W / 2.0
    z0 = np.full((H, W), 2.0, np.float32)  # frontal plane at 2 m
    cfg = NormalEstimationConfig(mode="pca_plane", window=5, max_depth_jump=0.2,
                                 min_depth=0.1, max_depth=8.0)

    _, _, res_clean = estimate_depth_normals_pca(z0, fx, fy, cx, cy, cfg, return_residual=True)
    rng = np.random.default_rng(0)
    z_noisy = (z0 + rng.normal(0, 0.02, z0.shape)).astype(np.float32)
    _, _, res_noisy = estimate_depth_normals_pca(z_noisy, fx, fy, cx, cy, cfg, return_residual=True)

    mc = np.nanmean(res_clean)
    mn = np.nanmean(res_noisy)
    assert mc < 0.005                 # clean plane: near-zero residual
    assert mn > mc and mn > 0.005     # noise inflates the residual
