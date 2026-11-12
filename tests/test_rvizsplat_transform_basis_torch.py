"""WS4: torch port of transform_snapshot_basis's per-Gaussian math (positions/
quaternions/SH) must match the numpy reference exactly. CPU tensors only --
no GPU required for this parity check (matches the pattern used for WS2's
torch dense-mask ports)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from eligsir.viewers.rvizsplat_export import (
    RVizSplatSnapshot,
    sh_band_rotation_matrices,
    transform_snapshot_basis,
    transform_snapshot_basis_torch,
)


def _random_unit_quats(n, seed):
    rng = np.random.default_rng(seed)
    q = rng.standard_normal((n, 4)).astype(np.float32)
    q /= np.linalg.norm(q, axis=1, keepdims=True) + 1e-12
    return q


def _random_rotation_basis(seed):
    rng = np.random.default_rng(seed)
    # QR decomposition of a random matrix gives a random orthonormal basis.
    a = rng.standard_normal((3, 3))
    q, r = np.linalg.qr(a)
    # Fix determinant sign to +1 (proper rotation, not a reflection).
    d = np.sign(np.diag(r))
    q = q * d
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q.astype(np.float32)


def _make_snapshot(n, sh_degree, seed=0):
    rng = np.random.default_rng(seed)
    coeff_count = (sh_degree + 1) ** 2
    return RVizSplatSnapshot(
        positions=rng.standard_normal((n, 3)).astype(np.float32),
        quaternions=_random_unit_quats(n, seed + 1),
        scales=rng.random((n, 3)).astype(np.float32) * 0.1,
        opacities=rng.random(n).astype(np.float32),
        spherical_harmonics=rng.standard_normal((n, coeff_count * 3)).astype(np.float32),
        sh_degree=sh_degree,
        sh_rotations=(),
    )


@pytest.mark.parametrize("sh_degree", [0, 1, 2, 3])
@pytest.mark.parametrize("n", [0, 1, 5, 37])
def test_torch_matches_numpy_reference(sh_degree, n):
    snap = _make_snapshot(n, sh_degree, seed=sh_degree * 100 + n)
    basis = _random_rotation_basis(seed=sh_degree * 100 + n + 1)

    ref = transform_snapshot_basis(snap, basis)

    sh_bands = ()
    if sh_degree >= 1 and n > 0:
        sh_bands = sh_band_rotation_matrices(basis, degree=min(3, sh_degree))

    positions_t = torch.from_numpy(snap.positions)
    quats_t = torch.from_numpy(snap.quaternions)
    sh_t = torch.from_numpy(snap.spherical_harmonics)

    pos_out, quat_out, sh_out = transform_snapshot_basis_torch(
        positions_t, quats_t, sh_t, sh_degree, basis, sh_bands,
    )

    assert pos_out.shape == ref.positions.shape
    assert quat_out.shape == ref.quaternions.shape
    assert sh_out.shape == ref.spherical_harmonics.shape

    if n == 0:
        return

    np.testing.assert_allclose(pos_out.numpy(), ref.positions, rtol=1e-4, atol=1e-5)
    # Quaternion sign is ambiguous (q and -q represent the same rotation);
    # compare via the closer of the two signs.
    q_np = quat_out.numpy()
    diff_pos = np.abs(q_np - ref.quaternions).max(axis=1)
    diff_neg = np.abs(-q_np - ref.quaternions).max(axis=1)
    worst = np.minimum(diff_pos, diff_neg).max()
    assert worst < 1e-3, f"quaternion mismatch: {worst}"
    np.testing.assert_allclose(sh_out.numpy(), ref.spherical_harmonics, rtol=1e-4, atol=1e-4)


def test_identity_basis_is_noop_on_positions_and_dc():
    snap = _make_snapshot(10, sh_degree=2, seed=7)
    identity = np.eye(3, dtype=np.float32)
    sh_bands = sh_band_rotation_matrices(identity, degree=2)

    positions_t = torch.from_numpy(snap.positions)
    quats_t = torch.from_numpy(snap.quaternions)
    sh_t = torch.from_numpy(snap.spherical_harmonics)

    pos_out, quat_out, sh_out = transform_snapshot_basis_torch(
        positions_t, quats_t, sh_t, 2, identity, sh_bands,
    )
    np.testing.assert_allclose(pos_out.numpy(), snap.positions, atol=1e-5)
    # DC band (first 3 coeffs, channel-major) must be unchanged.
    np.testing.assert_allclose(sh_out.numpy()[:, :3], snap.spherical_harmonics[:, :3], atol=1e-4)
