from __future__ import annotations

import numpy as np

from geometry.surface_cells import SurfaceCellIndex, surface_cell_keys


def test_surface_cell_keys_include_normal_direction():
    pts = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32)
    normals = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]], dtype=np.float32)
    keys = surface_cell_keys(pts, normals, voxel_size=0.1)
    assert keys[0] != keys[1]


def test_surface_cell_index_detects_occupied():
    idx = SurfaceCellIndex(voxel_size=0.1)
    pts = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    normals = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
    idx.add(pts, normals)
    assert bool(idx.occupied(pts, normals)[0])
