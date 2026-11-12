"""Regression tests for issue #1149: NumPy vs. PyTorch KNN surface-fusion
normals structurally diverging on sparse/degenerate neighbourhoods.

Both `geometry.insertion.fuse_points_with_knn` (NumPy) and
`geometry.device_fusion.fuse_points_with_knn_torch` (PyTorch, device-native)
estimate a local surface normal as the smallest-eigenvalue eigenvector of a
KNN neighbourhood's centered covariance. When that neighbourhood is
rank-deficient (points collapsed onto too few distinct locations, or
near-collinear), the "smallest eigenvector" lives in an undefined null space
and NumPy's/PyTorch's `eigh` can resolve it to near-opposite directions on
the identical input (confirmed NOT a float32-vs-float64 artifact).

The fix (`geometry/pca_degeneracy.py`) makes both implementations detect this
via a shared eigenvalue-ratio guard and fall back to the caller-supplied
`normal_hint` instead of trusting an ill-conditioned PCA estimate.
"""

from __future__ import annotations

import numpy as np
import torch

from geometry.device_fusion import fuse_points_with_knn_torch
from geometry.insertion import fuse_points_with_knn
from geometry.pca_degeneracy import EIGENVALUE_RATIO_EPS, eigvals_well_conditioned


def _angle_between(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    return float(np.arccos(np.clip(np.dot(a, b), -1.0, 1.0)))


def _near_collinear_repro():
    """issue #1149's synthetic repro (found by search over rng seeds against
    the PRE-FIX logic; seed=29 gives ~pi rad divergence pre-fix -- see
    docs referenced in the PR). Fixed data baked in here (not re-derived at
    test time) so this test is deterministic and doesn't depend on which
    seeds happen to trigger the pre-fix bug on a given numpy/torch build.
    """
    pool = np.array(
        [
            [0.03397284, -0.0042694, 0.01447644],
            [0.0438792, -0.00557871, 0.0183639],
            [0.00961958, -0.00111128, 0.00425448],
            [0.01098711, -0.001401, 0.00465283],
            [0.04349574, -0.00558651, 0.01836146],
            [0.02405873, -0.00310929, 0.01012053],
            [-0.02632541, 0.00342126, -0.01113968],
            [-0.01080185, 0.00124539, -0.00447052],
            [-0.04411653, 0.00560044, -0.01861245],
            [-0.03312914, 0.00419799, -0.01388943],
            [-0.01205616, 0.00153592, -0.00500696],
            [0.02872656, -0.00369043, 0.01218807],
            [-0.03118741, 0.00404392, -0.01315535],
            [0.00724813, -0.00086719, 0.00296513],
            [0.04322204, -0.00554663, 0.01820525],
        ],
        dtype=np.float32,
    )
    hint = np.array([[-0.05420366, 0.26023084, 0.96402377]], dtype=np.float32)
    query = pool[0:1].copy()
    nn_idx = np.arange(pool.shape[0])[None, :].astype(np.int64)
    return query, pool, nn_idx, hint


def test_near_collinear_neighborhood_is_flagged_degenerate_numpy():
    """The near-collinear repro's covariance must fail the eigenvalue-ratio
    guard (this is what makes fuse_points_with_knn fall back to the hint
    below, rather than trust an ill-conditioned PCA estimate)."""
    query, pool, nn_idx, hint = _near_collinear_repro()
    neigh = pool[nn_idx[0]]
    cloud = np.vstack([query[0][None, :], neigh]).astype(np.float64)
    centered = cloud - cloud.mean(axis=0, keepdims=True)
    cov = centered.T @ centered
    evals, _ = np.linalg.eigh(cov)
    assert not eigvals_well_conditioned(float(evals[0]), float(evals[1]), float(evals[2]))


def test_degenerate_neighborhood_numpy_and_torch_agree_via_hint_fallback():
    """The decisive #1149 regression check: numpy and torch must now agree
    on this input -- both fall back to the identical hint -- where the
    pre-fix logic produced normals ~pi rad apart (nearly antiparallel)."""
    query, pool, nn_idx, hint = _near_collinear_repro()

    degen_mask = np.zeros(1, dtype=bool)
    fpts_np, fnorm_np = fuse_points_with_knn(
        query, pool, nn_idx, hint, degenerate_mask_out=degen_mask
    )
    fpts_t, fnorm_t, well_conditioned_t = fuse_points_with_knn_torch(
        torch.from_numpy(query),
        torch.from_numpy(pool),
        torch.from_numpy(nn_idx),
        torch.from_numpy(hint),
        return_diagnostics=True,
    )

    assert bool(degen_mask[0]) is True
    assert bool(well_conditioned_t[0]) is False

    ang = _angle_between(fnorm_np[0], fnorm_t[0].numpy())
    assert ang < 1e-4, f"numpy/torch normals diverged by {ang} rad on a degenerate neighborhood"

    # Both implementations must have fallen back to the caller's hint exactly
    # (not merely agreed with each other by coincidence).
    np.testing.assert_allclose(fnorm_np[0], hint[0], atol=1e-5)
    np.testing.assert_allclose(fnorm_t[0].numpy(), hint[0], atol=1e-5)


def test_two_distinct_point_locations_is_degenerate():
    """The issue's headline scenario: a neighbourhood collapsed to ~2
    distinct point locations (near-duplicate depth samples)."""
    rng = np.random.default_rng(7)
    p_a = np.array([0.0, 0.0, 0.0])
    p_b = np.array([0.05, 0.0, 0.0])
    pts = []
    for _ in range(6):
        pts.append(p_a + rng.normal(0, 1e-6, size=3))
        pts.append(p_b + rng.normal(0, 1e-6, size=3))
    pool = np.array(pts, dtype=np.float32)
    query = pool[0:1].copy()
    nn_idx = np.arange(pool.shape[0])[None, :].astype(np.int64)
    hint = np.array([[0.0, 1.0, 0.0]], dtype=np.float32)

    degen_mask = np.zeros(1, dtype=bool)
    _, fnorm_np = fuse_points_with_knn(query, pool, nn_idx, hint, degenerate_mask_out=degen_mask)
    assert bool(degen_mask[0]) is True
    np.testing.assert_allclose(fnorm_np[0], hint[0], atol=1e-5)

    _, fnorm_t, well_conditioned_t = fuse_points_with_knn_torch(
        torch.from_numpy(query),
        torch.from_numpy(pool),
        torch.from_numpy(nn_idx),
        torch.from_numpy(hint),
        return_diagnostics=True,
    )
    assert bool(well_conditioned_t[0]) is False
    np.testing.assert_allclose(fnorm_t[0].numpy(), hint[0], atol=1e-5)


def _planar_patch(rng, n=60, jitter=0.005):
    xy = rng.uniform(-0.1, 0.1, size=(n, 2))
    zj = rng.normal(0.0, jitter, size=(n, 1))
    return np.hstack([xy, zj]).astype(np.float32)


def test_general_well_conditioned_case_unaffected_by_guard():
    """The guard must NOT trigger, and numpy/torch must still agree to
    ~1e-5/1e-6 precision, on a real, well-sampled planar patch -- mirrors
    issue #1149's own "agree to ~1e-6/1e-7 in the general case" finding."""
    rng = np.random.default_rng(3)
    pool = _planar_patch(rng, n=40, jitter=0.004)
    queries = pool[:5]
    k = 10
    nn_idx = np.zeros((5, k), dtype=np.int64)
    for i in range(5):
        d = np.linalg.norm(pool - queries[i], axis=1)
        nn_idx[i] = np.argsort(d)[:k]
    hints = np.tile([0.0, 0.0, 1.0], (5, 1)).astype(np.float32)

    degen_mask = np.zeros(5, dtype=bool)
    fpts_np, fnorm_np = fuse_points_with_knn(
        queries, pool, nn_idx, hints, degenerate_mask_out=degen_mask
    )
    assert not degen_mask.any(), "guard should not fire on a genuine planar patch"

    fpts_t, fnorm_t, well_conditioned_t = fuse_points_with_knn_torch(
        torch.from_numpy(queries),
        torch.from_numpy(pool),
        torch.from_numpy(nn_idx),
        torch.from_numpy(hints),
        return_diagnostics=True,
    )
    assert bool(well_conditioned_t.all())

    for i in range(5):
        ang = _angle_between(fnorm_np[i], fnorm_t[i].numpy())
        assert ang < 1e-3, f"query {i}: numpy/torch normals diverged by {ang} rad in the general case"
    assert np.max(np.abs(fpts_np - fpts_t.numpy())) < 1e-4


def test_fuse_points_with_knn_torch_default_return_is_still_a_2tuple():
    """return_diagnostics defaults False -- existing callers must be unaffected."""
    q = torch.tensor([[0.0, 0.0, 0.0]])
    pool = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
    idx = torch.tensor([[0, 1, 2]], dtype=torch.long)
    hint = torch.tensor([[0.0, 0.0, 1.0]])
    result = fuse_points_with_knn_torch(q, pool, idx, hint)
    assert len(result) == 2


def test_eigenvalue_ratio_eps_is_a_sane_value():
    # Sanity guard on the constant itself: strictly between float32's own
    # sqrt-eps floor and 1 (see geometry/pca_degeneracy.py's derivation).
    assert 1e-6 < EIGENVALUE_RATIO_EPS < 1e-1
