"""C1 regression: the DEFAULT (non-batched) source-depth insertion path must
stamp the real training ``iteration`` into ``birth_iter``.

Why this file exists separately from
``tests/test_insertion_committer_birth_iter.py``: that test only source-greps
``flush_insertion_batch``, which is *unreachable under default flags*
(``streaming_insertion_batch_frames=1`` / ``streaming_insertion_batch_steps=0``
=> ``_use_batch=False`` => ``_collect_only=False`` in ``train_streaming.py``,
so ``insert_gaussians_from_frame`` commits directly and never batches). A
source-grep of the batching path is therefore blind to whether the live default
path is fixed at all -- exactly how the original bug survived its own test.

Two layers here:
 1. ``test_default_flags_make_the_direct_path_live`` -- CPU-only. Pins the
    invariant that makes this file necessary: under stock defaults the direct
    (non-``_collect_only``) branch is the one that runs.
 2. behavioural tests -- GPU. Call ``insert_gaussians_from_frame`` for real on
    both of its commit branches (direct ``add_points_as_gaussians`` and
    ``map_backend``/``GaussianInsertionBatch``) and assert the newly inserted
    rows carry ``birth_iter == iteration``, not 0.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest
import torch


def test_default_flags_make_the_direct_path_live():
    """Stock defaults must leave batching OFF, so the direct commit branch of
    ``insert_gaussians_from_frame`` (not ``flush_insertion_batch``) is the one
    a default run actually exercises. Mirrors train_streaming.py's
    ``_use_batch = _batch_frames > 1 or _batch_steps > 0``.
    """
    from argparse import ArgumentParser

    from arguments import StreamingParams

    parser = ArgumentParser()
    sp = StreamingParams(parser)
    args = parser.parse_args([])
    extracted = sp.extract(args)

    batch_frames = int(getattr(extracted, "streaming_insertion_batch_frames"))
    batch_steps = int(getattr(extracted, "streaming_insertion_batch_steps"))
    use_batch = batch_frames > 1 or batch_steps > 0
    assert not use_batch, (
        "Defaults now enable batched insertion; this file's premise (and the "
        "reachability of the direct path) must be re-checked."
    )


def test_insert_gaussians_from_frame_accepts_iteration_and_uses_it_for_birth_iter():
    """Source-level: the commit path must pass ``birth_iter=iteration``.

    Kept alongside the behavioural tests below so the invariant is still
    checked on machines without a GPU.

    This used to require TWO occurrences, one per commit branch. PR #1142
    removed the direct ``add_points_as_gaussians`` fallback, so there is now
    exactly one commit path (a backend mutation transaction, against an
    ephemeral backend when no map_backend was supplied) and therefore exactly
    one occurrence. The invariant being protected -- new Gaussians record the
    REAL training iteration, not the source-frame index -- is unchanged, and
    the behavioural tests below check it directly rather than by counting
    string matches.
    """
    from eligsir.mapping.integration.integration_planner import (
        insert_gaussians_from_frame,
    )

    sig = inspect.signature(insert_gaussians_from_frame)
    assert "iteration" in sig.parameters, (
        "insert_gaussians_from_frame must accept an `iteration` parameter so "
        "source-depth Gaussians can record their real birth iteration."
    )

    source = inspect.getsource(insert_gaussians_from_frame)
    assert source.count("birth_iter=iteration") >= 1, (
        "The commit path must pass birth_iter=iteration to "
        "GaussianInsertionBatch."
    )
    assert "gaussians.add_points_as_gaussians(" not in source, (
        "PR #1142 removed the planner's direct population writer: every "
        "insertion must commit through a backend mutation transaction. A "
        "reintroduced direct call would bypass the mutation owner (and its "
        "version counters) entirely."
    )


# ---------------------------------------------------------------------------
# Behavioural (GPU) -- GsplatGaussianModel allocates CUDA tensors eagerly.
# ---------------------------------------------------------------------------

gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU")


def _make_model(n_points: int = 1):
    from argparse import Namespace

    from scene.gsplat_model import GsplatGaussianModel
    from eligsir.utils.graphics_utils import BasicPointCloud

    torch.manual_seed(0)
    model = GsplatGaussianModel(sh_degree=0)
    points = torch.rand(n_points, 3, device="cuda")
    colors = torch.rand(n_points, 3, device="cuda")
    pcd = BasicPointCloud(
        points=points.cpu().numpy(), colors=colors.cpu().numpy(),
        normals=np.zeros((n_points, 3)),
    )
    model.create_from_pcd(pcd, spatial_lr_scale=1.0)
    model.training_setup(Namespace(
        iterations=100,
        position_lr_init=0.00016, position_lr_final=0.0000016,
        position_lr_delay_mult=0.01, position_lr_max_steps=20000,
        feature_lr=0.0025, feature_lr_final=0.00025,
        opacity_lr=0.05, opacity_lr_final=0.001,
        scaling_lr=0.005, scaling_lr_final=0.0005,
        rotation_lr=0.001, rotation_lr_final=0.0001,
        percent_dense=0.01, lambda_dssim=0.2,
        densification_interval=100, opacity_reset_interval=30000,
        densify_from_iter=500, densify_until_iter=15000,
        densify_grad_threshold=0.0002, random_background=False,
    ))
    return model


def _make_frame():
    from eligsir.utils.streaming_frames import StreamingRGBDFrame

    h, w = 48, 64
    rng = np.random.default_rng(0)
    # Smooth fronto-parallel plane at 2m: random depth produces wild per-pixel
    # normals that the view-angle filter rejects, yielding 0 insertions.
    # float32 depth is interpreted as already-in-metres (see depth_to_meters).
    depth_np = np.full((h, w), 2.0, dtype=np.float32)
    rgb_np = (rng.uniform(0, 255, size=(h, w, 3))).astype(np.uint8)
    return StreamingRGBDFrame(
        index=0, timestamp=0.0, rgb_path="", depth_path="dummy.png",
        c2w=np.eye(4, dtype=np.float32), fx=500.0, fy=500.0,
        cx=w / 2.0, cy=h / 2.0, width=w, height=h,
        depth_scale=1000.0, _rgb_np=rgb_np, _depth_np=depth_np,
    )


class _Args:
    streaming_depth_stride = 4
    streaming_min_depth = 0.1
    streaming_max_depth = 8.0
    streaming_insert_voxel_size = 0.02
    streaming_max_new_gaussians_per_frame = 2000
    streaming_insert_opacity = 0.05
    streaming_depth_edge_threshold = 0.02
    streaming_max_view_angle = 70.0
    streaming_insert_knn_scale = False
    streaming_insertion_debug = True
    gpu_frame_prep = False
    gpu_frame_resident = False
    streaming_depth_filter_enabled = False
    streaming_depth_temporal_check = False
    # issue #408 / #368: integration_planner reads these bare (no getattr
    # fallback), so every partial-args fixture reaching insert_gaussians_from_frame
    # must declare them. The two temporal require_* flags are at the real parser
    # default (False, arguments/__init__.py:1049-1050) and are only *read* when
    # streaming_depth_temporal_check is on -- which it is not here -- so they are
    # behaviour-neutral. streaming_insert_scale_mode is pinned to "footprint",
    # the exact literal PR #394/#404 removed from the getattr (commit ab9834f),
    # NOT the parser default "knn": "knn" takes the distCUDA2 branch and would
    # change this fixture's scale math on GPU.
    streaming_depth_temporal_require_for_insert = False
    streaming_depth_temporal_require_for_sensor_closer = False
    streaming_insert_scale_mode = "footprint"
    streaming_surface_cell_dedup = False
    streaming_carve_invalid_depth = False


class _Scene:
    def check_occupancy(self, xyz_np, voxel_size, check_neighbors=True):
        return np.zeros(xyz_np.shape[0], dtype=bool)

    def check_surface_cells(self, xyz_np, normals_np):
        return np.zeros(xyz_np.shape[0], dtype=bool)

    def add_to_occupancy_hash(self, xyz):
        pass

    def add_to_surface_cells(self, xyz, normals):
        pass


ITER = 4321


@gpu
def test_direct_branch_stamps_real_birth_iter():
    from eligsir.mapping.integration.integration_planner import (
        insert_gaussians_from_frame,
    )

    model = _make_model()
    added, _ = insert_gaussians_from_frame(
        model, _make_frame(), _Args(), _Scene(),
        current_frame_idx=7, iteration=ITER,
    )
    assert added > 0, "fixture must actually insert something"
    new_birth_iter = model.birth_iter[-added:].cpu()
    assert torch.equal(new_birth_iter, torch.full((added,), ITER, dtype=new_birth_iter.dtype)), (
        f"direct path stamped birth_iter={new_birth_iter.unique().tolist()}, expected {ITER}"
    )
    # birth_frame keeps its own (frame-index) meaning, unchanged.
    assert int(model.birth_frame[-added:].min().item()) == 7


@gpu
def test_map_backend_branch_stamps_real_birth_iter():
    from eligsir.map.adapters.gsplat import GsplatMapBackend
    from eligsir.mapping.integration.integration_planner import (
        insert_gaussians_from_frame,
    )

    model = _make_model()
    backend = GsplatMapBackend(model)
    added, _ = insert_gaussians_from_frame(
        model, _make_frame(), _Args(), _Scene(),
        current_frame_idx=7, iteration=ITER, map_backend=backend,
    )
    assert added > 0
    new_birth_iter = model.birth_iter[-added:].cpu()
    assert torch.equal(new_birth_iter, torch.full((added,), ITER, dtype=new_birth_iter.dtype))


@gpu
def test_default_iteration_still_zero_for_legacy_callers():
    """Back-compat: ``iteration`` is optional, so pre-existing callers that
    don't pass it keep the old behaviour rather than crashing."""
    from eligsir.mapping.integration.integration_planner import (
        insert_gaussians_from_frame,
    )

    model = _make_model()
    added, _ = insert_gaussians_from_frame(
        model, _make_frame(), _Args(), _Scene(), current_frame_idx=0,
    )
    assert added > 0
    assert int(model.birth_iter[-added:].max().item()) == 0
