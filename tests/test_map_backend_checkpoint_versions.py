"""Wave 2 backend-ownership pass (2026-08-06): ``GaussianMapBackend.checkpoint_state()``/
``restore_checkpoint()`` must round-trip ``map_version``/``topology_version``,
not just the underlying model's own ``capture()``/``restore()`` payload —
previously both were silently dropped (a restored backend always reported
0/0 regardless of real history). Also covers
``_save_streaming_checkpoint``'s optional ``map_backend=`` parameter, which
threads the same round-trip through the on-disk checkpoint format.

Real-GPU fixture pattern matches ``tests/test_optimization_engine_wiring.py``.
"""

from __future__ import annotations

import tempfile
from argparse import Namespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from scene.gsplat_model import GsplatGaussianModel
from eligsir.map.adapters.gsplat import GsplatMapBackend
from eligsir.map.contracts import CheckpointBlob
from eligsir.map.mutation import GaussianInsertionBatch
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


def _make_model(n_points: int = 20) -> GsplatGaussianModel:
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


def test_checkpoint_state_restore_round_trips_versions():
    model = _make_model()
    backend = GsplatMapBackend(model)

    # Advance the backend's version counters via a real mutation.
    txn = backend.begin_mutation()
    batch = GaussianInsertionBatch(
        points=torch.rand(3, 3, device="cuda"),
        colors=torch.rand(3, 3, device="cuda"),
        init_scale=0.01,
    )
    txn.insert(batch)
    backend.commit_mutation(txn)
    assert backend._map_version == 1
    assert backend._topology_version == 1

    blob = backend.checkpoint_state()
    assert blob.map_version == 1
    assert blob.topology_version == 1

    # The canonical owner is idempotent per live model: constructing another
    # adapter cannot reset version history before restore.
    fresh_backend = GsplatMapBackend(model)
    assert fresh_backend is backend
    assert fresh_backend._map_version == 1
    fresh_backend.restore_checkpoint(blob, _make_training_args())
    assert fresh_backend._map_version == 1
    assert fresh_backend._topology_version == 1


def test_checkpoint_blob_defaults_versions_to_zero():
    blob = CheckpointBlob(layout="gsplat", payload="anything")
    assert blob.map_version == 0
    assert blob.topology_version == 0


def test_save_streaming_checkpoint_uses_backend_when_given():
    from eligsir.mapping.integration.rgbd_integrator_impl import _save_streaming_checkpoint

    save_worker = MagicMock()
    gaussians = MagicMock()

    fake_backend = MagicMock()
    fake_backend.checkpoint_state.return_value = CheckpointBlob(
        layout="gsplat", payload="backend_state", map_version=7, topology_version=3,
    )

    captured = {}

    def _fake_enqueue(fn):
        orig_save = torch.save
        try:
            def _capture_and_save(payload, path):
                captured.setdefault("payload", payload)
                return orig_save(payload, path)
            torch.save = _capture_and_save
            fn()
        finally:
            torch.save = orig_save

    save_worker.enqueue.side_effect = _fake_enqueue

    with tempfile.TemporaryDirectory() as tmpdir:
        _save_streaming_checkpoint(
            save_worker, gaussians, tmpdir, iteration=100,
            n_frames_ingested=10, last_ingest_iter=99, source_idx=10,
            map_backend=fake_backend,
        )

    state, _iteration, streaming_meta = captured["payload"]
    assert state == "backend_state"
    assert streaming_meta["map_version"] == 7
    assert streaming_meta["topology_version"] == 3
    gaussians.capture.assert_not_called()


def test_save_streaming_checkpoint_falls_back_without_backend():
    from eligsir.mapping.integration.rgbd_integrator_impl import _save_streaming_checkpoint

    save_worker = MagicMock()
    gaussians = MagicMock()
    gaussians.capture.return_value = "direct_state"

    captured = {}

    def _fake_enqueue(fn):
        orig_save = torch.save
        try:
            def _capture_and_save(payload, path):
                captured.setdefault("payload", payload)
                return orig_save(payload, path)
            torch.save = _capture_and_save
            fn()
        finally:
            torch.save = orig_save

    save_worker.enqueue.side_effect = _fake_enqueue

    with tempfile.TemporaryDirectory() as tmpdir:
        _save_streaming_checkpoint(
            save_worker, gaussians, tmpdir, iteration=100,
            n_frames_ingested=10, last_ingest_iter=99, source_idx=10,
        )

    state, _iteration, streaming_meta = captured["payload"]
    assert state == "direct_state"
    assert streaming_meta["map_version"] == 0
    assert streaming_meta["topology_version"] == 0
