"""Tests for utils/map_comparison.py -- Gaussian map vs. independent
point-cloud-only reference map comparison.

Requires torch (for compute_offsurface_metrics) and open3d (for ICP/voxel
occupancy) -- both hard project dependencies, run inside the container.
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
o3d = pytest.importorskip("open3d")

from eligsir.utils.map_comparison import (
    align_point_clouds,
    compare_gaussian_vs_reference_map,
    render_map_overlay,
)
from eligsir.utils.scene_geometry_metrics import compute_offsurface_metrics


class _FakeGaussians:
    def __init__(self, xyz: np.ndarray, opacity: np.ndarray):
        self._xyz = torch.from_numpy(xyz).float()
        self._opacity = torch.from_numpy(opacity).float()

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_opacity(self):
        return self._opacity


def _grid_surface(n=20, size=2.0, z=0.0):
    xs = np.linspace(0.0, size, n)
    X, Y = np.meshgrid(xs, xs)
    Z = np.full_like(X, z)
    return np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1).astype(np.float32)


def _pcd(points: np.ndarray):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    return pcd


def test_matching_clouds_report_low_offsurface_mass():
    pts = _grid_surface(n=25, size=2.0)
    gaussians = _FakeGaussians(pts, opacity=np.full(pts.shape[0], 0.9, dtype=np.float32))
    reference = _pcd(pts + np.random.default_rng(0).normal(scale=0.005, size=pts.shape))

    report = compare_gaussian_vs_reference_map(gaussians, reference, align=False, tau_m=0.05)

    assert report["valid"]
    assert report["gaussian_to_reference"]["offsurface_mass_frac"] < 0.05
    assert report["reference_to_gaussian"]["offsurface_mass_frac"] < 0.05
    assert report["symmetric_chamfer_m"] < 0.02


def test_gaussian_plume_flagged_gaussian_to_reference_only():
    surface = _grid_surface(n=25, size=2.0)
    plume = surface[:50].copy()
    plume[:, 2] += 10.0  # far-away floater cluster, low opacity
    gaussian_pts = np.concatenate([surface, plume], axis=0)
    opacity = np.concatenate([
        np.full(surface.shape[0], 0.9, dtype=np.float32),
        np.full(plume.shape[0], 0.9, dtype=np.float32),
    ])
    gaussians = _FakeGaussians(gaussian_pts, opacity)
    reference = _pcd(surface)

    report = compare_gaussian_vs_reference_map(gaussians, reference, align=False, tau_m=0.05)

    assert report["valid"]
    # Gaussian mass includes the plume, which has no nearby reference point -> flagged.
    assert report["gaussian_to_reference"]["offsurface_mass_frac"] > 0.05
    # Every reference point still has a nearby Gaussian on the real surface -> not flagged.
    assert report["reference_to_gaussian"]["offsurface_mass_frac"] < 0.05


def test_reference_missing_region_flagged_reference_to_gaussian_only():
    full_surface = _grid_surface(n=25, size=2.0)
    partial_surface = full_surface[full_surface[:, 0] < 1.0]  # reference misses half the room
    gaussians = _FakeGaussians(full_surface, opacity=np.full(full_surface.shape[0], 0.9, dtype=np.float32))
    reference = _pcd(partial_surface)

    report = compare_gaussian_vs_reference_map(gaussians, reference, align=False, tau_m=0.05)

    assert report["valid"]
    assert report["gaussian_to_reference"]["offsurface_mass_frac"] > 0.1
    assert report["reference_to_gaussian"]["offsurface_mass_frac"] < 0.05


def test_empty_inputs_reported_invalid():
    gaussians = _FakeGaussians(np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.float32))
    reference = _pcd(_grid_surface(n=5))
    report = compare_gaussian_vs_reference_map(gaussians, reference, align=False)
    assert report["valid"] is False


def test_map_overlay_is_a_vector_svg_without_embedded_raster(tmp_path):
    points = _grid_surface(n=3)
    requested = tmp_path / "map_comparison.png"
    output = tmp_path / "map_comparison.svg"

    render_map_overlay(str(requested), points, points + 0.01)

    text = output.read_text()
    assert "<svg" in text
    assert "<image" not in text
    assert "<path" in text
    assert not requested.exists()


def test_align_point_clouds_recovers_near_identity_on_shifted_copy():
    rng = np.random.default_rng(1)
    base = _grid_surface(n=20, size=2.0) + rng.normal(scale=0.001, size=(400, 3))
    shift = np.array([0.3, -0.2, 0.1], dtype=np.float32)
    shifted = base + shift

    transform, fitness, rmse = align_point_clouds(shifted, base, voxel_size=0.05)

    assert fitness > 0.8
    assert rmse < 0.05
    # align_point_clouds(source, target) registers source ONTO target, so the
    # recovered translation undoes the shift we applied to build `shifted`.
    recovered_shift = transform[:3, 3]
    assert np.allclose(recovered_shift, -shift, atol=0.05)


def test_align_point_clouds_reports_poor_fitness_on_unrelated_clouds():
    rng = np.random.default_rng(2)
    # Same centroid (translation alone can't hide the mismatch) but wildly different
    # spatial extent, so after centroid-init few points fall within ICP's correspondence
    # radius regardless of how the (structureless) point sets happen to align.
    cloud_a = rng.uniform(-1, 1, size=(300, 3)).astype(np.float32)
    cloud_b = rng.uniform(-25, 25, size=(300, 3)).astype(np.float32)

    _, fitness, _ = align_point_clouds(cloud_a, cloud_b, voxel_size=0.05)

    assert fitness < 0.5


def test_f_score_curve_high_at_loose_tau_low_at_tight_tau_for_matching_clouds():
    pts = _grid_surface(n=25, size=2.0)
    gaussians = _FakeGaussians(pts, opacity=np.full(pts.shape[0], 0.9, dtype=np.float32))
    reference = _pcd(pts + np.random.default_rng(0).normal(scale=0.03, size=pts.shape))

    report = compare_gaussian_vs_reference_map(gaussians, reference, align=False, tau_m=0.05)

    curve = report["f_score_curve"]
    # Loose threshold (0.2m) comfortably covers the 0.03 noise -> near-perfect F-score.
    assert curve["0.2"]["f_score"] > 0.95
    # Tight threshold (0.02m) is smaller than the noise scale -> F-score degrades.
    assert curve["0.02"]["f_score"] < curve["0.2"]["f_score"]


def test_f_score_curve_precision_drops_recall_stays_for_gaussian_plume():
    surface = _grid_surface(n=25, size=2.0)
    plume = surface[:50].copy()
    plume[:, 2] += 10.0
    gaussian_pts = np.concatenate([surface, plume], axis=0)
    opacity = np.full(gaussian_pts.shape[0], 0.9, dtype=np.float32)
    gaussians = _FakeGaussians(gaussian_pts, opacity)
    reference = _pcd(surface)

    report = compare_gaussian_vs_reference_map(gaussians, reference, align=False, tau_m=0.05)

    curve = report["f_score_curve"]
    # Plume drags precision down (extra unmatched Gaussian mass) but every real
    # reference point is still covered by a real-surface Gaussian -> recall stays high.
    assert curve["0.05"]["precision"] < 0.95
    assert curve["0.05"]["recall"] > 0.95


def test_floater_volume_captures_plume_not_matched_region():
    surface = _grid_surface(n=25, size=2.0)
    plume = surface[:50].copy()
    plume[:, 2] += 10.0
    gaussian_pts = np.concatenate([surface, plume], axis=0)
    opacity = np.full(gaussian_pts.shape[0], 0.9, dtype=np.float32)
    gaussians = _FakeGaussians(gaussian_pts, opacity)
    reference = _pcd(surface)

    report = compare_gaussian_vs_reference_map(gaussians, reference, align=False, tau_m=0.05, voxel_size=0.1)

    fv = report["floater_volume"]
    assert fv["n_floater_gaussians"] == 50
    assert fv["floater_volume_m3"] > 0.0
    # Matched-region (real surface) voxel count from occupancy should not absorb the plume.
    assert fv["floater_occupancy"]["n_points"] == 50


def test_compute_offsurface_metrics_return_distances_is_backward_compatible():
    pts = _grid_surface(n=10, size=1.0)
    xyz = torch.from_numpy(pts).float()
    opacity = torch.full((pts.shape[0],), 0.9, dtype=torch.float32)

    default = compute_offsurface_metrics(xyz, opacity, xyz, tau=0.05)
    with_distances = compute_offsurface_metrics(xyz, opacity, xyz, tau=0.05, return_distances=True)

    assert set(default.keys()) == {
        "valid", "tau_m", "offsurface_mass_frac", "offsurface_count_frac",
        "nn_dist_p50_m", "nn_dist_p90_m", "nn_dist_p99_m", "nn_dist_mean_m",
        "n_gaussians", "n_sensor_points",
    }
    assert set(default.keys()) < set(with_distances.keys())
    assert {"_xyz", "_alpha", "_nn_dist"} <= set(with_distances.keys())
    assert with_distances["_nn_dist"].shape[0] == pts.shape[0]
    for k in default:
        assert default[k] == with_distances[k]
