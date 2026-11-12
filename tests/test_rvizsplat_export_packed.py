"""WS4: snapshot_gaussians_for_rvizsplat_packed must exactly match the
existing five-download snapshot_gaussians_for_rvizsplat -- single-D2H
packing only, no behavior change. CPU-only (no GPU required)."""

from __future__ import annotations

import numpy as np
import torch

from eligsir.viewers.rvizsplat_export import (
    snapshot_gaussians_for_rvizsplat,
    snapshot_gaussians_for_rvizsplat_packed,
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
            "shN": torch.randn(n, 8, 3, generator=g),  # degree-2 has 9 coeffs total (1 dc + 8 rest)
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


def _assert_snapshots_equal(a, b):
    np.testing.assert_allclose(a.positions, b.positions, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(a.scales, b.scales, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(a.quaternions, b.quaternions, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(a.opacities, b.opacities, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(a.spherical_harmonics, b.spherical_harmonics, rtol=1e-6, atol=1e-6)
    assert a.sh_degree == b.sh_degree
    assert a.positions.dtype == b.positions.dtype == np.float32
    assert a.opacities.shape == b.opacities.shape


def test_packed_matches_unpacked_full_model():
    g = _RandGaussians(n=17)
    unpacked = snapshot_gaussians_for_rvizsplat(g)
    packed = snapshot_gaussians_for_rvizsplat_packed(g)
    _assert_snapshots_equal(unpacked, packed)


def test_packed_matches_unpacked_with_subsampling():
    g = _RandGaussians(n=50)
    unpacked = snapshot_gaussians_for_rvizsplat(g, max_gaussians=10)
    packed = snapshot_gaussians_for_rvizsplat_packed(g, max_gaussians=10)
    _assert_snapshots_equal(unpacked, packed)
    assert unpacked.positions.shape[0] == 10


def test_packed_arrays_are_contiguous():
    g = _RandGaussians(n=5)
    packed = snapshot_gaussians_for_rvizsplat_packed(g)
    assert packed.positions.flags["C_CONTIGUOUS"]
    assert packed.scales.flags["C_CONTIGUOUS"]
    assert packed.quaternions.flags["C_CONTIGUOUS"]
    assert packed.opacities.flags["C_CONTIGUOUS"]
    assert packed.spherical_harmonics.flags["C_CONTIGUOUS"]
