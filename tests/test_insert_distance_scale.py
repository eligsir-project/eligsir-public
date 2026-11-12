"""LEVER 1 — distance-aware in-plane Gaussian scale (pure-math core)."""

from __future__ import annotations

import numpy as np

from geometry.insertion import distance_aware_inplane_scale


def test_scale_grows_linearly_with_depth():
    fx = fy = 500.0
    z = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    tx, ty = distance_aware_inplane_scale(z, fx, fy, coeff=1.0, scale_max=10.0)
    # tx = z / f, strictly linear in z (no clamping at scale_max=10).
    np.testing.assert_allclose(tx, z / fx, rtol=1e-5)
    np.testing.assert_allclose(ty, z / fy, rtol=1e-5)
    # ratios match depth ratios → linearity
    assert tx[1] / tx[0] == np.float32(2.0)
    assert tx[2] / tx[0] == np.float32(3.0)


def test_coeff_scales_output():
    z = np.array([2.0], dtype=np.float32)
    a, _ = distance_aware_inplane_scale(z, 100.0, 100.0, coeff=1.0, scale_max=10.0)
    b, _ = distance_aware_inplane_scale(z, 100.0, 100.0, coeff=2.0, scale_max=10.0)
    np.testing.assert_allclose(b, 2.0 * a, rtol=1e-5)


def test_clamped_at_scale_max():
    z = np.array([1.0, 100.0], dtype=np.float32)
    tx, ty = distance_aware_inplane_scale(z, 10.0, 10.0, coeff=1.0, scale_max=0.05)
    # z=100, f=10 -> raw 10.0, must clamp to 0.05
    assert np.all(tx <= 0.05 + 1e-7)
    assert tx[1] == np.float32(0.05)
    assert ty[1] == np.float32(0.05)


def test_clamped_at_scale_min():
    z = np.array([1e-6], dtype=np.float32)
    tx, _ = distance_aware_inplane_scale(z, 1000.0, 1000.0, coeff=1.0, scale_min=1e-4)
    assert tx[0] == np.float32(1e-4)
