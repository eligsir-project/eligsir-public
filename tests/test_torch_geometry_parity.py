"""CPU parity tests: torch geometry/occupancy ports vs numpy references.

These run on CPU torch tensors only (no GPU device requested). They assert
bit-exact parity between:

  * utils.torch_geometry.voxel_downsample_indices_torch  vs the numpy
    _voxel_downsample_indices reference (surviving point set + first indices),
  * utils.torch_geometry.pack_voxel_keys_torch  vs StreamingScene._xyz_to_keys,
  * StreamingScene.check_occupancy_torch  vs StreamingScene.check_occupancy
    (with and without 26-neighbour expansion).
"""
import numpy as np
import torch

from eligsir.utils.torch_geometry import (
    pack_voxel_keys_torch,
    voxel_downsample_indices_torch,
)
from scene.streaming_scene import StreamingScene


# --------------------------------------------------------------------------
# numpy references (copied from the production code, kept here so the test is
# self-contained and pins the exact behaviour we mirror).
# --------------------------------------------------------------------------
def _voxel_downsample_indices_np(points: np.ndarray, voxel_size: float):
    if points.shape[0] == 0:
        return points, np.array([], dtype=np.int64)
    mins = points.min(axis=0)
    cell = np.floor((points - mins) / max(voxel_size, 1e-6)).astype(np.int64)
    maxc = cell.max(axis=0) + 1
    stride = np.array([maxc[1] * maxc[2], maxc[2], 1], dtype=np.int64)
    keys = cell @ stride
    _, first = np.unique(keys, return_index=True)
    return points[first], first


def _xyz_to_keys_np(xyz_np: np.ndarray, voxel_size: float) -> np.ndarray:
    offset = 1000.0
    coords = np.floor((xyz_np + offset) / max(voxel_size, 1e-6)).astype(np.int64)
    return (coords[:, 0] << 42) | (coords[:, 1] << 21) | coords[:, 2]


def _make_scene_stub():
    """A bare StreamingScene instance carrying only the occupancy attributes
    used by the torch occupancy methods (no heavy __init__)."""
    scene = object.__new__(StreamingScene)
    from eligsir.mapping.stores.occupancy_index import OccupancyIndex
    scene._occupancy_index = OccupancyIndex(voxel_size=0.02)
    scene._occupancy_keys_gpu = None
    scene._occupancy_keys = np.empty(0, dtype=np.int64)
    return scene


def _random_points(rng, n, scale, with_negative=True):
    pts = rng.uniform(-1.0, 1.0, size=(n, 3)) * scale
    if not with_negative:
        pts = np.abs(pts)
    return pts.astype(np.float64)


# --------------------------------------------------------------------------
# voxel_downsample_indices_torch parity
# --------------------------------------------------------------------------
def test_voxel_downsample_parity_varied():
    rng = np.random.default_rng(1234)
    configs = [
        (0, 0.05, 1.0),      # empty
        (1, 0.05, 1.0),      # single point
        (50, 0.1, 0.5),
        (500, 0.05, 2.0),
        (1000, 0.2, 5.0),
        (300, 0.01, 0.1),    # fine voxels, sparse survivors
        (300, 10.0, 1.0),    # coarse voxels, heavy collapse
    ]
    for n, vs, scale in configs:
        pts = _random_points(rng, n, scale)
        # Inject explicit duplicates so multiple rows share a cell key.
        if n >= 10:
            pts[5] = pts[2]
            pts[7] = pts[2]
            pts[9] = pts[4]
        np_ds, np_first = _voxel_downsample_indices_np(pts, vs)

        t_pts = torch.from_numpy(pts)
        t_ds, t_first = voxel_downsample_indices_torch(t_pts, vs)
        t_first_np = t_first.cpu().numpy()
        t_ds_np = t_ds.cpu().numpy()

        # First-index array must be bit-exact (both sorted by ascending key,
        # both take the lowest original index per cell -> stable tie-break).
        assert np.array_equal(t_first_np, np_first), (
            f"first-idx mismatch n={n} vs={vs}: {t_first_np} vs {np_first}"
        )
        # Surviving points follow directly from first indices.
        assert np.array_equal(t_ds_np, np_ds), f"survivor mismatch n={n} vs={vs}"

        # Surviving SET equality (sort-insensitive), as an independent check.
        np_set = set(map(tuple, np_ds.tolist()))
        t_set = set(map(tuple, t_ds_np.tolist()))
        assert np_set == t_set


def test_voxel_downsample_all_duplicate():
    # All rows identical -> single survivor at index 0.
    pts = np.tile(np.array([[0.3, -0.7, 1.1]]), (20, 1))
    np_ds, np_first = _voxel_downsample_indices_np(pts, 0.05)
    t_ds, t_first = voxel_downsample_indices_torch(torch.from_numpy(pts), 0.05)
    assert np.array_equal(t_first.cpu().numpy(), np_first)
    assert int(np_first[0]) == 0
    assert np.array_equal(t_ds.cpu().numpy(), np_ds)


# --------------------------------------------------------------------------
# pack_voxel_keys_torch parity (bit-for-bit int64)
# --------------------------------------------------------------------------
def test_pack_voxel_keys_bit_exact():
    rng = np.random.default_rng(999)
    for vs in (0.02, 0.05, 0.1, 1.0):
        for scale in (0.1, 1.0, 10.0, 100.0):
            xyz = _random_points(rng, 777, scale)
            np_keys = _xyz_to_keys_np(xyz, vs)
            t_keys = pack_voxel_keys_torch(torch.from_numpy(xyz), vs).cpu().numpy()
            assert t_keys.dtype == np.int64
            assert np.array_equal(t_keys, np_keys), f"key mismatch vs={vs} scale={scale}"


def test_pack_voxel_keys_empty():
    xyz = np.empty((0, 3), dtype=np.float64)
    np_keys = _xyz_to_keys_np(xyz, 0.02)
    t_keys = pack_voxel_keys_torch(torch.from_numpy(xyz), 0.02).cpu().numpy()
    assert np.array_equal(t_keys, np_keys)
    assert t_keys.shape == (0,)


# --------------------------------------------------------------------------
# check_occupancy_torch parity
# --------------------------------------------------------------------------
def _build_occupancy(scene, occ_xyz, vs):
    """Populate both numpy and torch occupancy stores from occ_xyz."""
    scene.occupancy_voxel_size = vs
    keys = _xyz_to_keys_np(occ_xyz, vs)
    scene._occupancy_keys = np.sort(np.unique(keys))
    scene._occupancy_keys_gpu = torch.from_numpy(
        np.sort(np.unique(keys))
    )


def test_check_occupancy_parity_with_and_without_neighbors():
    rng = np.random.default_rng(2024)
    for vs in (0.02, 0.1, 0.5):
        for scale in (1.0, 5.0):
            scene = _make_scene_stub()
            occ = _random_points(rng, 400, scale)
            _build_occupancy(scene, occ, vs)

            # Mixed queries: some exact hits, some near-misses, some far.
            q_hits = occ[: 50]
            q_near = occ[50:100] + vs * 0.5  # likely neighbour cells
            q_far = _random_points(rng, 50, scale * 3.0) + 50.0
            query = np.concatenate([q_hits, q_near, q_far], axis=0)

            for check_neighbors in (False, True):
                np_res = scene.check_occupancy(query, vs, check_neighbors=check_neighbors)
                t_res = scene.check_occupancy_torch(
                    torch.from_numpy(query), vs, check_neighbors=check_neighbors
                ).cpu().numpy()
                assert np_res.dtype == bool
                assert np.array_equal(t_res, np_res), (
                    f"occupancy mismatch vs={vs} scale={scale} "
                    f"neighbors={check_neighbors}"
                )


def test_check_occupancy_empty_hash():
    scene = _make_scene_stub()
    scene.occupancy_voxel_size = 0.02
    scene._occupancy_keys = np.empty(0, dtype=np.int64)
    scene._occupancy_keys_gpu = torch.empty(0, dtype=torch.int64)
    query = np.random.default_rng(0).uniform(-1, 1, size=(20, 3))
    np_res = scene.check_occupancy(query, 0.02, check_neighbors=True)
    t_res = scene.check_occupancy_torch(
        torch.from_numpy(query), 0.02, check_neighbors=True
    ).cpu().numpy()
    assert not np_res.any()
    assert np.array_equal(t_res, np_res)


def test_check_occupancy_empty_query():
    scene = _make_scene_stub()
    occ = np.random.default_rng(1).uniform(-1, 1, size=(30, 3))
    _build_occupancy(scene, occ, 0.05)
    query = np.empty((0, 3), dtype=np.float64)
    np_res = scene.check_occupancy(query, 0.05, check_neighbors=True)
    t_res = scene.check_occupancy_torch(
        torch.from_numpy(query), 0.05, check_neighbors=True
    ).cpu().numpy()
    assert np_res.shape == (0,)
    assert np.array_equal(t_res, np_res)
