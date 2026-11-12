from __future__ import annotations

import numpy as np

from geometry.normals import NormalEstimationConfig, estimate_depth_normals


def test_finite_difference_normals_shape_and_confidence():
    depth = np.ones((8, 8), dtype=np.float32)
    normals, conf = estimate_depth_normals(depth, 100.0, 100.0, 4.0, 4.0)
    assert normals.shape == (3, 8, 8)
    assert conf.shape == (8, 8)
    assert np.allclose(normals[:, 4, 4], np.array([0.0, 0.0, 1.0], dtype=np.float32))


def test_pca_normals_runs_on_plane():
    depth = np.ones((6, 6), dtype=np.float32)
    normals, conf = estimate_depth_normals(
        depth,
        100.0,
        100.0,
        3.0,
        3.0,
        NormalEstimationConfig(mode="pca_plane", window=3, min_valid_ratio=0.5),
    )
    assert normals.shape == (3, 6, 6)
    assert conf[3, 3] > 0
