"""Wiring tests for the cohort-telemetry debug side channel added to
``flush_insertion_batch`` (Task 3 of the evidence-driven-mutation-cadence
experiment plan).

Uses lightweight stubs for ``gaussians``/``streaming_scene`` rather than
real model objects -- flush_insertion_batch's own core dedup/subsample
math is what's under test here, not the real Gaussian model.
"""

from types import SimpleNamespace

import numpy as np
import torch

from eligsir.mapping.integration.insertion_committer import flush_insertion_batch


class _StubGaussians:
    def __init__(self):
        self.added_calls = []
        self.get_xyz = torch.zeros((0, 3))

    # Parameter names must match the real primitive on both models
    # (scene/gaussian_model.py, scene/gsplat_model.py): since insertion is
    # routed through a mutation transaction, the backend calls this with
    # `points=`/`colors=` keywords, so a stub taking `pts`/`cols` fails with
    # "missing 2 required positional arguments" rather than anything that
    # names the real drift.
    def add_points_as_gaussians(self, points, colors, **kwargs):
        self.added_calls.append(points.shape[0])
        self.get_xyz = torch.cat([self.get_xyz, points], dim=0)
        return points.shape[0]


class _StubStreamingScene:
    def _xyz_to_keys(self, xyz_np, voxel_size):
        # One key per point -- deterministic, collision-free for these tests
        # (no two test points coincide), so dedup never trims anything.
        return np.arange(xyz_np.shape[0])

    def add_to_occupancy_hash(self, xyz):
        pass

    def add_to_surface_cells(self, xyz, normals):
        pass


def _make_candidate(n_pts: int, birth_frame: int, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    return {
        "pts": rng.normal(size=(n_pts, 3)).astype(np.float32),
        "cols": rng.uniform(size=(n_pts, 3)).astype(np.float32),
        "log_scales": np.zeros((n_pts, 3), dtype=np.float32),
        "q_world": np.tile(np.array([1, 0, 0, 0], dtype=np.float32), (n_pts, 1)),
        "birth_frame": birth_frame,
        "init_opacity": 0.5,
    }


def _base_args():
    return SimpleNamespace(
        streaming_insert_voxel_size=0.02,
        streaming_max_new_gaussians_per_frame=100000,
        streaming_surface_cell_dedup=False,
        # Read bare by flush_insertion_batch (PR #1142) to size the
        # GaussianMutationArbiter capacity ceiling. Real parser default;
        # every candidate in this module is far below it, so no clamping.
        cap_max=500000,
    )


class TestCohortDebugOutDefaultOff:
    def test_none_by_default_is_a_strict_noop(self):
        gaussians = _StubGaussians()
        scene = _StubStreamingScene()
        pending = [_make_candidate(5, birth_frame=10, seed=1)]
        added = flush_insertion_batch(pending, gaussians, scene, iteration=100, args=_base_args())
        assert added == 5
        # No exception, no cohort_debug_out param touched -- nothing to assert
        # beyond "ran exactly like before this parameter existed".


class TestCohortDebugOutPopulated:
    def test_single_candidate_frame_index_matches_birth_frame(self):
        gaussians = _StubGaussians()
        scene = _StubStreamingScene()
        pending = [_make_candidate(5, birth_frame=42, seed=2)]
        debug_out = []
        added = flush_insertion_batch(
            pending, gaussians, scene, iteration=100, args=_base_args(),
            cohort_debug_out=debug_out,
        )
        assert added == 5
        assert len(debug_out) == 1
        frame_idx = debug_out[0]["frame_index"]
        xyz = debug_out[0]["xyz"]
        assert frame_idx.shape[0] == 5
        assert xyz.shape == (5, 3)
        assert (frame_idx == 42).all()
        assert debug_out[0]["n_before_dedup"] == 5
        assert debug_out[0]["n_deduped"] == 0

    def test_multi_candidate_frame_index_is_per_point_not_collapsed(self):
        # Unlike birth_frame_val (the production single-scalar-per-batch
        # value), the debug frame_index array must preserve each candidate's
        # own birth_frame per point -- this is the exact provenance gap
        # Task 2 flagged.
        gaussians = _StubGaussians()
        scene = _StubStreamingScene()
        pending = [
            _make_candidate(3, birth_frame=10, seed=3),
            _make_candidate(4, birth_frame=20, seed=4),
            _make_candidate(2, birth_frame=30, seed=5),
        ]
        debug_out = []
        added = flush_insertion_batch(
            pending, gaussians, scene, iteration=200, args=_base_args(),
            cohort_debug_out=debug_out,
        )
        assert added == 9
        frame_idx = debug_out[0]["frame_index"]
        assert frame_idx.shape[0] == 9
        assert (frame_idx[:3] == 10).all()
        assert (frame_idx[3:7] == 20).all()
        assert (frame_idx[7:9] == 30).all()

    def test_subsample_trims_frame_index_in_lockstep_with_xyz(self):
        gaussians = _StubGaussians()
        scene = _StubStreamingScene()
        pending = [_make_candidate(50, birth_frame=7, seed=6)]
        args = _base_args()
        args.streaming_max_new_gaussians_per_frame = 10  # forces subsampling
        debug_out = []
        added = flush_insertion_batch(
            pending, gaussians, scene, iteration=300, args=args,
            cohort_debug_out=debug_out,
        )
        assert added == 10
        frame_idx = debug_out[0]["frame_index"]
        xyz = debug_out[0]["xyz"]
        assert frame_idx.shape[0] == 10
        assert xyz.shape[0] == 10
        assert (frame_idx == 7).all()
        assert debug_out[0]["n_before_dedup"] == 50
        # dedup didn't trim anything here (stub _xyz_to_keys is collision-free),
        # so all 40 removed points are attributed to the subsample step alone.
        assert debug_out[0]["n_deduped"] == 40


def _single_voxel_candidate(n_pts: int, birth_frame: int, voxel_size: float) -> dict:
    """Candidate whose points all fall inside ONE voxel of ``voxel_size``.

    Dedup is no longer driven by the streaming scene's ``_xyz_to_keys``: since
    insertion moved into the committer, ``flush_insertion_batch`` deduplicates
    with the torch-native ``deterministic_voxel_first_indices`` over the
    candidate points themselves. A stub scene that returns a constant key is
    therefore inert, and the collision has to be real geometry -- points
    spread over a fraction of one voxel, so ``floor(p / voxel_size)`` is
    identical for all of them.
    """
    coords = np.linspace(0.0, voxel_size * 0.4, n_pts, dtype=np.float32)
    pts = np.stack([coords, coords, coords], axis=1)
    return {
        "pts": pts,
        "cols": np.full((n_pts, 3), 0.5, dtype=np.float32),
        "log_scales": np.zeros((n_pts, 3), dtype=np.float32),
        "q_world": np.tile(np.array([1, 0, 0, 0], dtype=np.float32), (n_pts, 1)),
        "birth_frame": birth_frame,
        "init_opacity": 0.5,
    }


class TestNBeforeDedupReflectsDedupLoss:
    def test_dedup_loss_is_attributed_correctly(self):
        gaussians = _StubGaussians()
        scene = _StubStreamingScene()
        pending = [_single_voxel_candidate(6, birth_frame=1, voxel_size=0.02)]
        debug_out = []
        added = flush_insertion_batch(
            pending, gaussians, scene, iteration=400, args=_base_args(),
            cohort_debug_out=debug_out,
        )
        assert added == 1  # all 6 points collide into a single voxel key
        assert debug_out[0]["n_before_dedup"] == 6
        assert debug_out[0]["n_deduped"] == 5
        assert debug_out[0]["frame_index"].shape[0] == 1
