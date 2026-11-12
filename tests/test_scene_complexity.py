"""Tests for scene-expressiveness (SV) and Gaussian-efficiency metrics.

SV = unique surface-cells (voxel x normal-bin) of the observed surface. It normalises
Gaussian count so efficiency (gaussians_per_surfel = n_live/SV) is size/growth-invariant.
These tests are pure-numpy (no GPU).
"""

from __future__ import annotations

import numpy as np

from eligsir.utils.scene_geometry_metrics import scene_complexity_stats, expressiveness_metrics


def _grid_plane(n=200, size=1.0):
    xs = np.linspace(0.0, size, n)
    ys = np.linspace(0.0, size, n)
    X, Y = np.meshgrid(xs, ys)
    pts = np.stack([X.ravel(), Y.ravel(), np.zeros(n * n)], axis=1).astype(np.float32)
    nrm = np.tile(np.array([0.0, 0.0, 1.0], np.float32), (pts.shape[0], 1))
    return pts, nrm


def test_flat_plane_dim_approx_2():
    pts, nrm = _grid_plane()
    r = scene_complexity_stats(pts, nrm, voxel_size=0.05, scales=(1, 2, 4))
    assert r["sv"] > 0
    # A 2D surface scales as SV ~ delta^-2 -> box-counting dimension ~ 2.
    assert abs(r["complexity_dim"] - 2.0) < 0.4


def test_halving_voxel_increases_sv():
    pts, nrm = _grid_plane()
    coarse = scene_complexity_stats(pts, nrm, voxel_size=0.10)["sv"]
    fine = scene_complexity_stats(pts, nrm, voxel_size=0.05)["sv"]
    assert fine > coarse


def test_rough_surface_more_complex_than_flat():
    pts, nrm = _grid_plane()
    rng = np.random.default_rng(0)
    rough_n = rng.normal(size=nrm.shape).astype(np.float32)  # normals span many bins
    flat = scene_complexity_stats(pts, nrm, voxel_size=0.05)["sv"]
    rough = scene_complexity_stats(pts, rough_n, voxel_size=0.05)["sv"]
    assert rough > flat


def test_sv_invariant_to_duplication_and_order():
    pts, nrm = _grid_plane(n=50)
    base = scene_complexity_stats(pts, nrm, voxel_size=0.05)["sv"]
    dup_p = np.concatenate([pts, pts], axis=0)
    dup_n = np.concatenate([nrm, nrm], axis=0)
    perm = np.random.default_rng(1).permutation(dup_p.shape[0])
    dup = scene_complexity_stats(dup_p[perm], dup_n[perm], voxel_size=0.05)["sv"]
    assert dup == base  # unique cells: duplication/reordering must not change SV


def test_empty_cloud():
    r = scene_complexity_stats(np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32),
                               voxel_size=0.05)
    assert r["sv"] == 0 and r["n_points"] == 0


def test_efficiency_arithmetic_and_gating():
    m = expressiveness_metrics(sv=1000, n_live=2000, n_total=4000, opacity_sum=1500,
                               mean_test_psnr=20.0, eval_valid=True)
    assert abs(m["gaussians_per_surfel"] - 2.0) < 1e-6
    assert abs(m["n_eff_frac"] - 0.375) < 1e-6
    assert abs(m["psnr_per_kgaussian"] - 10.0) < 1e-6  # 20 / (2000/1000)

    # Invalid / degenerate eval -> no efficiency PSNR (avoid bogus numbers).
    m2 = expressiveness_metrics(sv=1000, n_live=2000, mean_test_psnr=20.0, eval_valid=False)
    assert "psnr_per_kgaussian" not in m2

    # No SV -> gaussians_per_surfel undefined (None), not a divide-by-zero.
    m3 = expressiveness_metrics(sv=0, n_live=2000)
    assert m3["gaussians_per_surfel"] is None


def test_gaussians_per_surfel_is_size_invariant():
    # Double the scene (2x SV) with proportionally more Gaussians -> same efficiency.
    small = expressiveness_metrics(sv=1000, n_live=1500)["gaussians_per_surfel"]
    big = expressiveness_metrics(sv=2000, n_live=3000)["gaussians_per_surfel"]
    assert abs(small - big) < 1e-6
