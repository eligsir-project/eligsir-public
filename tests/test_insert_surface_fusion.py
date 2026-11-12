"""LEVER 2 — KNN surface-fusion of voxel-dedup representatives (pure-math core)."""

from __future__ import annotations

import numpy as np

from geometry.insertion import fuse_point_with_neighbors, fuse_points_with_knn


def _planar_patch(rng, n=20, jitter=0.02):
    """Points on the z=0 plane (normal +z) with Gaussian jitter along z."""
    xy = rng.uniform(-0.1, 0.1, size=(n, 2))
    zj = rng.normal(0.0, jitter, size=(n, 1))
    return np.hstack([xy, zj]).astype(np.float32)


def test_position_denoises_toward_plane():
    rng = np.random.default_rng(0)
    patch = _planar_patch(rng, n=40, jitter=0.03)
    # representative is a single noisy point; neighbours are the patch
    rep = np.array([0.0, 0.0, 0.05], dtype=np.float32)  # noisy off-plane
    fused_pos, _ = fuse_point_with_neighbors(rep, patch)
    # fused z should be much closer to the plane (0) than the noisy rep
    assert abs(fused_pos[2]) < abs(rep[2])
    assert abs(fused_pos[2]) < 0.02


def test_pca_normal_recovers_plane_normal_up_to_sign():
    rng = np.random.default_rng(1)
    patch = _planar_patch(rng, n=60, jitter=0.005)
    rep = patch[0]
    _, normal = fuse_point_with_neighbors(rep, patch[1:])
    # plane normal is +/- z
    assert abs(abs(normal[2]) - 1.0) < 0.05
    np.testing.assert_allclose(np.linalg.norm(normal), 1.0, atol=1e-5)


def test_normal_oriented_to_hint():
    rng = np.random.default_rng(2)
    patch = _planar_patch(rng, n=60, jitter=0.005)
    rep = patch[0]
    hint = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    _, normal = fuse_point_with_neighbors(rep, patch[1:], normal_hint=hint)
    # must agree with hint sign (negative z hemisphere)
    assert normal[2] < 0.0


def test_degenerate_few_points_uses_hint():
    rep = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    neigh = np.array([[1.0, 1.0, 1.0]], dtype=np.float32)  # only 2 distinct -> <3
    hint = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    pos, normal = fuse_point_with_neighbors(rep, neigh, normal_hint=hint)
    np.testing.assert_allclose(normal, hint, atol=1e-6)


def test_vectorised_wrapper_shapes_and_orientation():
    rng = np.random.default_rng(3)
    pool = _planar_patch(rng, n=30, jitter=0.004)
    queries = pool[:5]
    # each query's 8 nearest in pool (brute force for the test)
    k = 8
    nn_idx = np.zeros((5, k), dtype=np.int64)
    for i in range(5):
        d = np.linalg.norm(pool - queries[i], axis=1)
        nn_idx[i] = np.argsort(d)[:k]
    hints = np.tile([0.0, 0.0, 1.0], (5, 1)).astype(np.float32)
    fpts, fnorm = fuse_points_with_knn(queries, pool, nn_idx, hints)
    assert fpts.shape == (5, 3)
    assert fnorm.shape == (5, 3)
    # all normals oriented to +z hint and roughly planar
    assert np.all(fnorm[:, 2] > 0.0)
    assert np.all(np.abs(np.abs(fnorm[:, 2]) - 1.0) < 0.1)


def test_edge_aware_radius_excludes_far_neighbors():
    """A neighbour across a depth edge (far away) must NOT drag the fused point."""
    import numpy as np
    from geometry.insertion import fuse_point_with_neighbors
    point = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    # 3 same-surface neighbours near the point + 1 far "edge" neighbour at z=3
    near = np.array([[0.01, 0.0, 1.0], [0.0, 0.01, 1.0], [-0.01, 0.0, 1.0]], dtype=np.float32)
    far = np.array([[0.0, 0.0, 3.0]], dtype=np.float32)
    neigh = np.vstack([near, far])
    # No filter: far neighbour drags z up
    fp_nofilter, _ = fuse_point_with_neighbors(point, neigh)
    assert fp_nofilter[2] > 1.2, "without filter the far edge neighbour should raise z"
    # Edge-aware: 0.1 m radius excludes the z=3 neighbour → fused z stays ~1.0
    fp_filter, _ = fuse_point_with_neighbors(point, neigh, max_radius=0.1)
    assert abs(fp_filter[2] - 1.0) < 0.02, "edge-aware filter should keep the point on-surface"
