"""Regression coverage for immutable async checkpoints (issue #1172)."""

from __future__ import annotations

import threading
import types

import pytest
import torch
from torch import nn

from scene.gsplat_model import (
    GsplatGaussianModel,
    OptimizerDictProxy,
    _immutable_tensor_snapshot,
)
from eligsir.runtime.save_worker import AsyncSaveWorker


def _source_model():
    model = object.__new__(GsplatGaussianModel)
    param = nn.Parameter(torch.tensor([1.0, 2.0]))
    model.params = nn.ParameterDict({"means": param})
    # Checkpoint immutability/state restoration is optimizer-API generic. Using
    # stock CPU Adam keeps this regression on the CPU-Torch CI surface instead
    # of coupling it to gsplat's device-oriented SelectiveAdam implementation.
    optimizer = torch.optim.Adam(
        [{"params": [param], "name": "means"}], eps=1e-15, betas=(0.9, 0.999)
    )
    optimizer.state[param] = {
        "step": torch.tensor(5.0),
        "exp_avg": torch.tensor([3.0, 3.0]),
        "exp_avg_sq": torch.tensor([4.0, 4.0]),
    }
    model.optimizers = {"means": optimizer}
    model.optimizer = OptimizerDictProxy(model.optimizers)
    model.active_sh_degree = 2
    model._next_gaussian_uid = 10
    model._seed_uid_range = (8, 10)
    model.spatial_lr_scale = 1.5
    model.visibility_ema = torch.tensor([[6.0], [6.0]])
    model.birth_frame = torch.tensor([7, 7], dtype=torch.int32)
    model.root_uid = torch.tensor([8, 9], dtype=torch.int64)
    return model, optimizer, param


def _restore_target(device="cpu"):
    model = object.__new__(GsplatGaussianModel)
    model.params = nn.ParameterDict(
        {"means": nn.Parameter(torch.zeros(2, device=device))}
    )
    model.gaussian_uid = torch.empty(0, dtype=torch.int64, device=device)

    def training_setup(self, _args):
        param = self.params["means"]
        optimizer = torch.optim.Adam(
            [{"params": [param], "name": "means"}],
            eps=1e-15,
            betas=(0.9, 0.999),
        )
        self.optimizers = {"means": optimizer}
        self.optimizer = OptimizerDictProxy(self.optimizers)

    model.training_setup = types.MethodType(training_setup, model)
    model._ensure_geometry_streaming_buffers = types.MethodType(lambda self: None, model)
    model._ensure_lineage_after_restore = types.MethodType(
        lambda self, streaming_state, skipped=frozenset(): False, model
    )
    model._restore_seed_uid_range = types.MethodType(lambda self, *args, **kwargs: None, model)
    return model


def _mutate_live_state(model, optimizer, param):
    state = optimizer.state[param]
    with torch.no_grad():
        for tensor in (param, state["step"], state["exp_avg"], state["exp_avg_sq"]):
            tensor.add_(10)
        model.visibility_ema.add_(10)
        model.birth_frame.add_(10)
        model.root_uid.add_(10)


def test_recursive_snapshot_copies_containers_and_rejects_meta():
    tensor = torch.tensor([1.0])
    live = {"list": [tensor], "tuple": (tensor,)}

    captured = _immutable_tensor_snapshot(live)
    tensor.add_(10)

    assert captured is not live
    assert captured["list"] is not live["list"]
    assert captured["tuple"] is not live["tuple"]
    assert torch.equal(captured["list"][0], torch.tensor([1.0]))
    assert torch.equal(captured["tuple"][0], torch.tensor([1.0]))
    assert captured["list"][0].data_ptr() != tensor.data_ptr()
    with pytest.raises(ValueError, match="meta tensor"):
        _immutable_tensor_snapshot(torch.empty(1, device="meta"))


def test_delayed_async_save_serializes_capture_time_state(tmp_path):
    from eligsir.mapping.integration.checkpoint_io import _save_streaming_checkpoint

    model, optimizer, param = _source_model()
    worker = AsyncSaveWorker()
    blocker_started = threading.Event()
    release = threading.Event()

    def block_worker():
        blocker_started.set()
        release.wait(timeout=10)

    worker.enqueue(block_worker)
    assert blocker_started.wait(timeout=5)
    checkpoint = _save_streaming_checkpoint(
        worker,
        model,
        str(tmp_path),
        iteration=123,
        n_frames_ingested=4,
        last_ingest_iter=120,
        source_idx=5,
    )
    _mutate_live_state(model, optimizer, param)
    release.set()
    worker.shutdown()

    state, iteration, metadata = torch.load(checkpoint, weights_only=False)
    optimizer_state = state["optimizers"]["means"]["state"][0]
    assert iteration == 123
    assert metadata["source_idx"] == 5
    assert torch.equal(state["params"]["means"], torch.tensor([1.0, 2.0]))
    assert set(optimizer_state) == {"step", "exp_avg", "exp_avg_sq"}
    assert torch.equal(optimizer_state["step"], torch.tensor(5.0))
    assert torch.equal(optimizer_state["exp_avg"], torch.tensor([3.0, 3.0]))
    assert torch.equal(optimizer_state["exp_avg_sq"], torch.tensor([4.0, 4.0]))
    assert torch.equal(state["visibility_ema"], torch.tensor([[6.0], [6.0]]))
    assert torch.equal(
        state["streaming_state"]["birth_frame"],
        torch.tensor([7, 7], dtype=torch.int32),
    )
    assert torch.equal(
        state["streaming_state"]["root_uid"],
        torch.tensor([8, 9], dtype=torch.int64),
    )
    assert all(
        tensor.device.type == "cpu"
        for tensor in (
            state["params"]["means"],
            optimizer_state["exp_avg"],
            state["visibility_ema"],
            state["streaming_state"]["birth_frame"],
        )
    )


def test_cpu_snapshot_restores_full_state_on_existing_target():
    source, _optimizer, _param = _source_model()
    snapshot = source.capture()
    restored = _restore_target()

    restored.restore(snapshot, object())

    restored_param = restored.params["means"]
    restored_state = restored.optimizers["means"].state[restored_param]
    assert restored_param.device.type == "cpu"
    assert torch.equal(restored_param, torch.tensor([1.0, 2.0]))
    assert set(restored_state) == {"step", "exp_avg", "exp_avg_sq"}
    assert torch.equal(restored_state["step"], torch.tensor(5.0))
    assert torch.equal(restored_state["exp_avg"], torch.tensor([3.0, 3.0]))
    assert torch.equal(restored_state["exp_avg_sq"], torch.tensor([4.0, 4.0]))
    assert torch.equal(restored.visibility_ema, torch.tensor([[6.0], [6.0]]))
    assert torch.equal(restored.birth_frame, torch.tensor([7, 7], dtype=torch.int32))
    assert torch.equal(restored.root_uid, torch.tensor([8, 9], dtype=torch.int64))


def test_restore_prefers_existing_runtime_target_over_saved_device():
    source, _optimizer, _param = _source_model()
    snapshot = source.capture()
    snapshot["optimizers"] = {}
    restored = _restore_target(device="meta")

    restored.restore(snapshot, object())

    assert restored.params["means"].device.type == "meta"
    assert restored.visibility_ema.device.type == "meta"
    assert restored.birth_frame.device.type == "meta"
    assert restored.root_uid.device.type == "meta"
