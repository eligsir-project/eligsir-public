"""Tier B parity (Wave 2 backend-ownership pass, 2026-08-06): the insertion
hot path's two call sites --
``eligsir.mapping.integration.insertion_committer.flush_insertion_batch``
and ``eligsir.mapping.integration.rgbd_integrator_impl.insert_gaussians_from_frame``'s
direct (non-``_collect_only``) branch -- must produce byte-identical
resulting model state whether committed the old way
(``gaussians.add_points_as_gaussians(...)`` directly) or the new way
(``map_backend.begin_mutation()`` / ``GaussianInsertionBatch`` /
``map_backend.commit_mutation(...)``).

Real-GPU fixture pattern matches ``tests/test_optimization_engine_wiring.py``/
``tests/test_integration_services_delegate.py``.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from argparse import Namespace

from scene.gsplat_model import GsplatGaussianModel
from eligsir.map.adapters.gsplat import GsplatMapBackend
from eligsir.utils.graphics_utils import BasicPointCloud

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU")


def _make_training_args():
    return Namespace(
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
    )


def _make_model(n_points: int = 10) -> GsplatGaussianModel:
    torch.manual_seed(0)
    model = GsplatGaussianModel(sh_degree=0)
    points = torch.rand(n_points, 3, device="cuda")
    colors = torch.rand(n_points, 3, device="cuda")
    pcd = BasicPointCloud(
        points=points.cpu().numpy(), colors=colors.cpu().numpy(),
        normals=np.zeros((n_points, 3)),
    )
    model.create_from_pcd(pcd, spatial_lr_scale=1.0)
    model.training_setup(_make_training_args())
    return model


def _model_state(model) -> dict:
    return {
        "means": model.get_xyz.detach().clone(),
        "scales": model.get_scaling.detach().clone(),
        "opacities": model.get_opacity.detach().clone(),
        "rotations": model.get_rotation.detach().clone(),
        "n": model.get_xyz.shape[0],
    }


def _assert_states_equal(a: dict, b: dict) -> None:
    assert a["n"] == b["n"]
    for key in ("means", "scales", "opacities", "rotations"):
        torch.testing.assert_close(a[key], b[key], atol=0, rtol=0)


class _FakeStreamingScene:
    def _xyz_to_keys(self, pts_np, voxel_size):
        return np.arange(pts_np.shape[0])

    def add_to_occupancy_hash(self, xyz):
        pass

    def add_to_surface_cells(self, xyz, normals):
        pass


def _new_insert_points(n=5, seed=1):
    rng = np.random.default_rng(seed)
    pts = rng.uniform(-1, 1, size=(n, 3)).astype(np.float32)
    cols = rng.uniform(0, 1, size=(n, 3)).astype(np.float32)
    log_scales = np.log(np.full((n, 3), 0.01, dtype=np.float32))
    q_world = np.tile(np.array([1, 0, 0, 0], dtype=np.float32), (n, 1))
    normals = np.tile(np.array([0, 0, 1], dtype=np.float32), (n, 1))
    confidence = np.full((n,), 0.8, dtype=np.float32)
    source = np.zeros((n,), dtype=np.int16)
    return pts, cols, log_scales, q_world, normals, confidence, source


def test_flush_insertion_batch_backend_parity():
    from eligsir.mapping.integration.insertion_committer import flush_insertion_batch

    pts, cols, log_scales, q_world, normals, confidence, source = _new_insert_points()
    pending = [{
        "pts": pts, "cols": cols, "log_scales": log_scales, "q_world": q_world,
        "normals": normals, "confidence": confidence, "source": source,
        "birth_frame": 0, "init_opacity": 0.5, "opacities_raw": None,
    }]

    scene = _FakeStreamingScene()

    class _Args:
        # flush_insertion_batch (insertion_committer.py) reads this bare, no
        # getattr fallback (issue #363 residue -- see that function's own
        # docstring comment). Real parser default, arguments/__init__.py:853.
        streaming_max_new_gaussians_per_frame = 1000
        # Also read bare (PR #1142): flush_insertion_batch sizes the
        # GaussianMutationArbiter's capacity ceiling from it. Real parser
        # default, arguments/__init__.py -- far above this fixture's row
        # count, so it cannot clamp and the parity assertions are unmoved.
        cap_max = 500000

    args_ns = _Args()

    # Path A: direct (no map_backend)
    model_a = _make_model()
    added_a = flush_insertion_batch(list(pending), model_a, scene, iteration=10, args=args_ns)
    state_a = _model_state(model_a)

    # Path B: through GaussianMapBackend
    model_b = _make_model()
    backend_b = GsplatMapBackend(model_b)
    added_b = flush_insertion_batch(list(pending), model_b, scene, iteration=10, args=args_ns, map_backend=backend_b)
    state_b = _model_state(model_b)

    assert added_a == added_b
    _assert_states_equal(state_a, state_b)
    # Backend path must have actually advanced the version counters. Read the
    # PUBLIC contract properties: the private fields this used to reach into
    # were renamed (_map_version -> _map_content_version) and the assertion
    # started raising AttributeError instead of comparing anything, so the
    # counters went unverified while the test merely looked red for an
    # unrelated-looking reason.
    assert backend_b.map_version == 1
    assert backend_b.map_content_version == 1
    assert backend_b.topology_version == 1


def test_insert_gaussians_from_frame_backend_parity():
    from eligsir.mapping.integration.rgbd_integrator_impl import insert_gaussians_from_frame
    from eligsir.utils.streaming_frames import StreamingRGBDFrame

    h, w = 48, 64
    rng = np.random.default_rng(0)
    depth_np = (rng.uniform(1.0, 3.0, size=(h, w))).astype(np.float32) * 1000.0
    rgb_np = (rng.uniform(0, 255, size=(h, w, 3))).astype(np.uint8)
    c2w = np.eye(4, dtype=np.float32)
    frame = StreamingRGBDFrame(
        index=0, timestamp=0.0, rgb_path="", depth_path="dummy.png",
        c2w=c2w, fx=500.0, fy=500.0, cx=w / 2.0, cy=h / 2.0, width=w, height=h,
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
        # fallback), so every partial-args fixture reaching
        # insert_gaussians_from_frame must declare them. The two temporal
        # require_* flags are at the real parser default (False,
        # arguments/__init__.py:1049-1050) and are only *read* when
        # streaming_depth_temporal_check is on -- which it is not here -- so
        # they are behaviour-neutral. streaming_insert_scale_mode is pinned to
        # "footprint", the exact literal PR #394/#404 removed from the getattr
        # (commit ab9834f), NOT the parser default "knn": "knn" takes the
        # distCUDA2 branch and would change this fixture's scale math on GPU.
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

    torch.manual_seed(0)
    model_a = _make_model(n_points=1)
    added_a, _stats_a = insert_gaussians_from_frame(
        model_a, frame, _Args(), _Scene(), current_frame_idx=0,
    )
    state_a = _model_state(model_a)

    torch.manual_seed(0)
    model_b = _make_model(n_points=1)
    backend_b = GsplatMapBackend(model_b)
    added_b, _stats_b = insert_gaussians_from_frame(
        model_b, frame, _Args(), _Scene(), current_frame_idx=0, map_backend=backend_b,
    )
    state_b = _model_state(model_b)

    assert added_a == added_b
    _assert_states_equal(state_a, state_b)
    if added_b > 0:
        assert backend_b._map_version == 1
