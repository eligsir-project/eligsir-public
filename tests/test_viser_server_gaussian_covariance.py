"""Unit tests for the scale/quat -> covariance construction in
``eligsir/viewers/sinks/viser_server.py`` (issue #1042 follow-up: real
oriented Gaussian ellipsoids via viser's ``add_gaussian_splats``, not a
uniform-dot point cloud).

Deliberately does NOT require ``viser`` to be installed: the module under
test only imports ``viser`` inside ``run_viser_server`` (the sidecar entry
point), not at module scope, so these pure-numpy helper functions are
importable and testable in any environment (including CI without the
sidecar-only ``viser`` dependency).
"""
from __future__ import annotations

import numpy as np
import pytest

from eligsir.viewers.sinks.viser_server import (
    _covariances_from_scale_quat,
    _default_covariances,
    _quat_wxyz_to_rotmat,
)


def test_identity_quat_gives_identity_rotation():
    quats = np.array([[1.0, 0.0, 0.0, 0.0]])  # wxyz identity
    R = _quat_wxyz_to_rotmat(quats)
    assert R.shape == (1, 3, 3)
    assert np.allclose(R[0], np.eye(3), atol=1e-6)


def test_covariance_from_identity_quat_is_diag_scale_squared():
    scales = np.array([[1.0, 2.0, 3.0]])
    quats = np.array([[1.0, 0.0, 0.0, 0.0]])
    cov = _covariances_from_scale_quat(scales, quats)
    assert cov.shape == (1, 3, 3)
    expected = np.diag([1.0, 4.0, 9.0])
    assert np.allclose(cov[0], expected, atol=1e-4)


def test_covariance_is_symmetric_and_psd():
    rng = np.random.default_rng(0)
    n = 50
    scales = rng.uniform(0.01, 0.5, size=(n, 3))
    quats = rng.normal(size=(n, 4))
    quats /= np.linalg.norm(quats, axis=1, keepdims=True)
    cov = _covariances_from_scale_quat(scales, quats)
    assert cov.shape == (n, 3, 3)
    # symmetric
    assert np.allclose(cov, cov.transpose(0, 2, 1), atol=1e-4)
    # positive semi-definite: all eigenvalues >= 0
    eigvals = np.linalg.eigvalsh(cov.astype(np.float64))
    assert np.all(eigvals >= -1e-6)


def test_covariance_eigenvalues_equal_squared_scales_rotation_invariant():
    # A rotation must not change the *eigenvalues* of the covariance (only
    # its orientation) -- the physically-meaningful invariant that proves
    # the ellipsoid shape (not just its existence) is real and correct.
    rng = np.random.default_rng(1)
    n = 20
    scales = rng.uniform(0.05, 1.0, size=(n, 3))
    quats = rng.normal(size=(n, 4))
    quats /= np.linalg.norm(quats, axis=1, keepdims=True)
    cov = _covariances_from_scale_quat(scales, quats)
    eigvals = np.sort(np.linalg.eigvalsh(cov.astype(np.float64)), axis=1)
    expected = np.sort(scales**2, axis=1)
    assert np.allclose(eigvals, expected, atol=1e-3)


def test_covariance_batch_is_non_degenerate_and_non_uniform():
    # The specific regression this whole follow-up exists to fix: distinct
    # Gaussians must get distinct, non-degenerate covariances (real
    # ellipsoids), not a single uniform-size dot repeated N times.
    rng = np.random.default_rng(2)
    n = 100
    scales = rng.uniform(0.01, 0.3, size=(n, 3))
    quats = rng.normal(size=(n, 4))
    quats /= np.linalg.norm(quats, axis=1, keepdims=True)
    cov = _covariances_from_scale_quat(scales, quats)
    eigvals = np.linalg.eigvalsh(cov.astype(np.float64))
    assert eigvals.min() > 0.0  # no degenerate (zero-volume) ellipsoid
    assert eigvals.std() > 0.0  # not all identical -- real shape variety


def test_default_covariances_are_small_isotropic_fallback():
    cov = _default_covariances(5, radius=0.02)
    assert cov.shape == (5, 3, 3)
    for i in range(5):
        assert np.allclose(cov[i], np.eye(3) * (0.02**2), atol=1e-8)


# -- MapSnapshot (coordinator follow-up: typed contract, not an ad-hoc dict) --

from eligsir.viewers.sinks.viser_server import MapSnapshot


def test_map_snapshot_from_cmd_parses_full_payload():
    n = 6
    cmd = {
        "positions": np.arange(n * 3, dtype=np.float32).reshape(n, 3),
        "colors": np.full((n, 3), 0.7, dtype=np.float32),
        "scales": np.full((n, 3), 0.02, dtype=np.float32),
        "quats": np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (n, 1)),
        "opacities": np.full((n, 1), 0.9, dtype=np.float32),
        "n_total_gaussians": 1000,
        "iteration": 42,
        "timestamp": 3.5,
    }
    snap = MapSnapshot.from_cmd(cmd)
    assert snap.centers.shape == (n, 3)
    assert snap.n_sampled == n
    assert snap.gaussian_count == 1000
    assert snap.is_complete is False
    assert snap.has_shape is True
    assert snap.iteration == 42
    assert snap.timestamp == pytest.approx(3.5)
    assert np.allclose(snap.colors, 0.7)
    assert np.allclose(snap.opacities, 0.9)


def test_map_snapshot_from_cmd_applies_defaults_when_optional_fields_absent():
    n = 3
    cmd = {"positions": np.zeros((n, 3), dtype=np.float32)}
    snap = MapSnapshot.from_cmd(cmd)
    assert snap.scales is None
    assert snap.rotations is None
    assert snap.opacities is None
    assert snap.colors is None
    assert snap.has_shape is False
    assert snap.gaussian_count == n  # defaults to n_sampled when absent
    assert snap.is_complete is True
    assert snap.iteration == 0
    assert snap.timestamp == 0.0


def test_map_snapshot_opacities_reshaped_to_column_vector():
    n = 4
    cmd = {
        "positions": np.zeros((n, 3), dtype=np.float32),
        "opacities": np.full(n, 0.42, dtype=np.float32),  # flat (N,), not (N,1)
    }
    snap = MapSnapshot.from_cmd(cmd)
    assert snap.opacities.shape == (n, 1)
    assert np.allclose(snap.opacities, 0.42)
