"""WS4: snapshot_and_transform_for_rvizsplat_torch (gather + torch transform
+ single download) must match the existing two-step numpy pipeline
(snapshot_gaussians_for_rvizsplat -> transform_snapshot_basis) exactly.
CPU-only (no GPU required)."""

from __future__ import annotations

import numpy as np
import torch

from eligsir.viewers.rvizsplat_export import (
    sh_band_rotation_matrices,
    snapshot_and_transform_for_rvizsplat_torch,
    snapshot_gaussians_for_rvizsplat,
    transform_snapshot_basis,
)


class _RandGaussians:
    active_sh_degree = 2

    def __init__(self, n: int = 17, seed: int = 0):
        g = torch.Generator().manual_seed(seed)
        self._xyz = torch.randn(n, 3, generator=g)
        self._scales = torch.rand(n, 3, generator=g) * 0.1
        self._quats = torch.nn.functional.normalize(torch.randn(n, 4, generator=g), dim=1)
        self._opacities = torch.rand(n, 1, generator=g)
        self.params = {
            "sh0": torch.randn(n, 1, 3, generator=g),
            "shN": torch.randn(n, 8, 3, generator=g),
        }

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_scaling(self):
        return self._scales

    @property
    def get_rotation(self):
        return self._quats

    @property
    def get_opacity(self):
        return self._opacities

    @property
    def get_features(self):
        return torch.cat((self.params["sh0"], self.params["shN"]), dim=1)


def _random_rotation_basis(seed):
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((3, 3))
    q, r = np.linalg.qr(a)
    d = np.sign(np.diag(r))
    q = q * d
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q.astype(np.float32)


def _assert_snapshots_equal(a, b):
    np.testing.assert_allclose(a.positions, b.positions, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(a.scales, b.scales, rtol=1e-6, atol=1e-6)
    # Quaternion sign ambiguity.
    diff_pos = np.abs(a.quaternions - b.quaternions).max(axis=1)
    diff_neg = np.abs(-a.quaternions - b.quaternions).max(axis=1)
    assert np.minimum(diff_pos, diff_neg).max() < 1e-3
    np.testing.assert_allclose(a.opacities, b.opacities, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(a.spherical_harmonics, b.spherical_harmonics, rtol=1e-4, atol=1e-4)
    assert a.sh_degree == b.sh_degree


def test_combined_matches_two_step_numpy_pipeline():
    g = _RandGaussians(n=17)
    basis = _random_rotation_basis(seed=1)
    sh_bands = sh_band_rotation_matrices(basis, degree=2)

    ref_snap = snapshot_gaussians_for_rvizsplat(g)
    ref_xform = transform_snapshot_basis(ref_snap, basis)

    combined = snapshot_and_transform_for_rvizsplat_torch(g, basis, sh_bands)

    _assert_snapshots_equal(combined, ref_xform)


def test_combined_matches_with_subsampling():
    g = _RandGaussians(n=50)
    basis = _random_rotation_basis(seed=2)
    sh_bands = sh_band_rotation_matrices(basis, degree=2)

    ref_snap = snapshot_gaussians_for_rvizsplat(g, max_gaussians=10)
    ref_xform = transform_snapshot_basis(ref_snap, basis)

    combined = snapshot_and_transform_for_rvizsplat_torch(g, basis, sh_bands, max_gaussians=10)

    assert combined.positions.shape[0] == 10
    _assert_snapshots_equal(combined, ref_xform)


def test_combined_matches_with_identity_basis():
    g = _RandGaussians(n=12, seed=3)
    identity = np.eye(3, dtype=np.float32)
    sh_bands = sh_band_rotation_matrices(identity, degree=2)

    ref_snap = snapshot_gaussians_for_rvizsplat(g)
    ref_xform = transform_snapshot_basis(ref_snap, identity)

    combined = snapshot_and_transform_for_rvizsplat_torch(g, identity, sh_bands)

    _assert_snapshots_equal(combined, ref_xform)
