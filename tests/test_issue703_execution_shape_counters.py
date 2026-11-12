"""Issue #703: execution-shape counters for detecting sequential vs vectorized work.

Tests verify that renderer_camera_batch_sizes, loss_objective_call_count,
loss_views_per_call, and backward_call_count accurately reflect the execution
pattern and can distinguish SEQUENTIAL_GRADIENT_ACCUMULATION from
TRUE_CAMERA_BATCH patterns.
"""

from __future__ import annotations

from argparse import Namespace
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from _prepared_loss_fakes import patch_assemble_loss

from eligsir.mapping.renderer.true_camera_batch import TrueCameraBatchResult
from eligsir.contracts.runtime import SourceIdentity
from eligsir.optimization.gaussian import engine as engine_mod
from eligsir.optimization.gaussian import true_batch_executor as executor_mod
from eligsir.optimization.gaussian.contracts import (
    EvidenceRequirements,
    GradientPolicyContext,
    OptimizationRuntimeContext,
    OptimizationStepRequest,
    OptimizationStepResult,
    OptimizationView,
    RandomStateToken,
    RenderOutputs,
)
from eligsir.optimization.gaussian.engine import GaussianOptimizationEngine
from eligsir.optimization.gaussian.loss_breakdown import LossBreakdown
from eligsir.optimization.gaussian.true_batch_executor import (
    run_true_camera_batched_step,
)


N = 6


class _FakeModel:
    def __init__(self):
        self.provisional = torch.zeros(N, dtype=torch.bool)
        self.last_supervised_iter = torch.zeros(N, dtype=torch.int32)

    def normalize_rotation_params(self, mask=None):
        pass


class _FakeSession:
    parameter_version = 0

    def __init__(self, model):
        self.model = model
        self.steps = []

    def update_learning_rate(self, iteration):
        return 0.001

    def step(self, visibility=None):
        self.steps.append(visibility)

    def zero_grad(self):
        pass


class _FakeBackend:
    def __init__(self, session):
        self._session = session

    def begin_optimization(self):
        return self._session


class _FakeBreakdown:
    def __init__(self, tensor, finite=True, depth_valid_sample_count=0):
        self.total_tensor = tensor
        self.total = tensor.detach()
        self.components = {"Ll1": tensor.detach()}
        self.loss_finite = finite
        self.depth_valid_sample_count = depth_valid_sample_count


def _visibility(*indices):
    mask = torch.zeros(N, dtype=torch.bool)
    for i in indices:
        mask[i] = True
    return mask


def _request(iteration=5, camera="cam", source_index=None, render_depth=False):
    return OptimizationStepRequest(
        view=OptimizationView(
            camera=camera,
            background=None,
            render_depth=render_depth,
            source_index=source_index,
        ),
        iteration=iteration,
        map_version=1,
        topology_version=2,
        resolution_bucket="default",
        evidence_requirements=EvidenceRequirements(),
        random_state_token=RandomStateToken.capture(),
    )


@pytest.fixture
def harness(monkeypatch):
    """Wire an engine over fakes to track execution-shape behavior."""
    model = _FakeModel()
    session = _FakeSession(model)
    engine = GaussianOptimizationEngine(_FakeBackend(session))

    rec = {
        "renders": [],
        "backward_calls": [],
        "loss_calls": [],
        "visibility": {},
        "finite": {},
        "loss_value": {},
    }

    def _fake_invoke_render(gaussians, camera, pipe, bg, **kwargs):
        rec["renders"].append(camera)
        return RenderOutputs(payload={
            "render": torch.zeros(3, 2, 2),
            "visibility_filter": rec["visibility"].get(camera, _visibility(0)),
        })

    def _fake_assemble_loss(inputs):
        rec["loss_calls"].append(inputs.viewpoint_cam)
        cam = inputs.viewpoint_cam
        value = rec["loss_value"].get(cam, 1.0)
        finite = rec["finite"].get(cam, True)
        tensor = torch.tensor(float("nan") if not finite else value, requires_grad=True)
        return _FakeBreakdown(tensor, finite=finite)

    def _fake_render_and_backward(gaussians, args, iteration, strategy, pkg, loss):
        rec["backward_calls"].append(float(loss.detach()))

    monkeypatch.setattr(engine_mod, "invoke_render", _fake_invoke_render)
    monkeypatch.setattr(engine_mod._objective, "assemble_loss", _fake_assemble_loss)
    monkeypatch.setattr(engine, "render_and_backward", _fake_render_and_backward)
    monkeypatch.setattr(engine, "run_gradient_policies", lambda ctx: [])

    def _runtime():
        return OptimizationRuntimeContext(
            session=session, gaussians=model,
            args=Namespace(), opt=Namespace(iterations=10_000),
            pipe=Namespace(gsplat_sparse_grad=False),
            mcmc_strategy=None,
            gradient_policy_context=GradientPolicyContext(
                gaussians=model, args=Namespace(), iteration=5, n_frames_ingested=0,
            ),
            loss_assembly_kwargs={},
        )

    return engine, session, rec, _runtime


def test_b1_single_view_execution_shape(harness):
    """B=1: one render, one loss call, one backward call."""
    engine, _session, rec, runtime = harness
    rec["loss_value"]["cam"] = 0.75

    result = engine.run_batched_step([_request()], runtime())

    # Execution shape for single view
    assert result.renderer_camera_batch_sizes == (1,)
    assert result.loss_objective_call_count == 1
    assert result.loss_views_per_call == (1,)
    assert result.backward_call_count == 1


def test_b4_sequential_gradient_accumulation_execution_shape(harness):
    """B=4 sequential: four renders, four loss calls, four backward calls."""
    engine, _session, rec, runtime = harness
    for i in range(4):
        rec["loss_value"][f"cam{i}"] = 1.0 / 4.0
        rec["visibility"][f"cam{i}"] = _visibility(0)

    requests = [_request(camera=f"cam{i}") for i in range(4)]
    result = engine.run_batched_step(requests, runtime())

    # Sequential gradient accumulation signature
    assert result.renderer_camera_batch_sizes == (1, 1, 1, 1)
    assert result.loss_objective_call_count == 4
    assert result.loss_views_per_call == (1, 1, 1, 1)
    assert result.backward_call_count == 4
    assert result.supervision_view_count == 4


def test_b4_skipped_view_does_not_increment_counters(harness):
    """B=4 with second view non-finite: three renders/losses/backards, one loss call for failed view."""
    engine, _session, rec, runtime = harness
    rec["loss_value"]["cam0"] = 0.5
    rec["loss_value"]["cam1"] = float("nan")  # Will be non-finite
    rec["finite"]["cam1"] = False
    rec["loss_value"]["cam2"] = 0.5
    rec["loss_value"]["cam3"] = 0.5
    for i in range(4):
        rec["visibility"][f"cam{i}"] = _visibility(0)

    requests = [_request(camera=f"cam{i}") for i in range(4)]
    result = engine.run_batched_step(requests, runtime())

    # cam0 succeeds, cam1 fails, cam2 and cam3 succeed
    # Loss call happens for all four (assembly doesn't know if it will fail)
    # But backward only on successful views
    assert result.renderer_camera_batch_sizes == (1, 1, 1, 1)
    assert result.loss_objective_call_count == 4
    assert result.loss_views_per_call == (1, 1, 1, 1)
    assert result.backward_call_count == 3  # Only successful views
    assert result.supervision_view_count == 3
    assert result.skipped_view_count == 1


def test_report_block_ratios_at_zero_steps():
    """With zero optimization steps, execution-shape ratios are not included."""
    from eligsir.utils.supervision_work import supervision_work_report_block

    block = supervision_work_report_block(
        optimization_step_count=0,
        supervision_view_count=0,
        rasterizer_call_count=0,
        supervised_pixel_count=0,
        loss_objective_call_count=4,
        backward_call_count=4,
    )

    # With zero steps, no ratios should be computed
    assert "loss_objective_calls_per_step" not in block
    assert "backward_calls_per_step" not in block


def test_report_block_ratios_with_valid_steps():
    """With valid steps, execution-shape ratios are computed."""
    from eligsir.utils.supervision_work import supervision_work_report_block

    block = supervision_work_report_block(
        optimization_step_count=10,
        supervision_view_count=40,
        rasterizer_call_count=40,
        supervised_pixel_count=40 * 4 * 4,  # 4x4 images
        loss_objective_call_count=40,
        backward_call_count=40,
        renderer_camera_batch_size_sum=40,
        renderer_camera_batch_size_max=1,
        loss_views_per_call_sum=40,
        loss_views_per_call_max=1,
    )

    # Sequential gradient accumulation: B=4, so 40 views / 10 steps = 4
    assert block["loss_objective_calls_per_step"] == 4.0
    assert block["backward_calls_per_step"] == 4.0
    assert block["renderer_camera_batch_size_mean"] == 1.0
    assert block["renderer_camera_batch_size_max"] == 1
    assert block["loss_views_per_call_mean"] == 1.0
    assert block["loss_views_per_call_max"] == 1
    assert block["rasterizer_calls_per_step"] == 4.0


def test_report_block_with_partial_execution_shape_params():
    """Report block accepts partial execution-shape params gracefully."""
    from eligsir.utils.supervision_work import supervision_work_report_block

    block = supervision_work_report_block(
        optimization_step_count=5,
        supervision_view_count=5,
        rasterizer_call_count=5,
        supervised_pixel_count=5 * 4 * 4,
        loss_objective_call_count=5,
        # backward_call_count and batch size params omitted
    )

    assert block["loss_objective_calls_per_step"] == 1.0
    assert "backward_calls_per_step" not in block
    assert "renderer_camera_batch_size_max" not in block


def test_report_block_preserves_exposure_at_zero_steps_with_camera_counts():
    """With zero steps but camera counts, exposure distribution is returned."""
    from eligsir.utils.supervision_work import supervision_work_report_block

    camera_counts = {"cam0": 0, "cam1": 0}
    block = supervision_work_report_block(
        optimization_step_count=0,
        supervision_view_count=0,
        rasterizer_call_count=0,
        supervised_pixel_count=0,
        per_camera_supervision_counts=camera_counts,
        loss_objective_call_count=0,
    )

    # Exposure distribution should be present
    assert "supervision_exposures_per_view_mean" in block
    assert "never_supervised_view_count" in block
    assert block["never_supervised_view_count"] == 2


def test_metric_registry_has_execution_shape_metrics():
    """All new execution-shape metrics are registered."""
    from eligsir.utils.metric_registry import CANONICAL_METRICS

    required_metrics = [
        "loss_objective_call_count",
        "backward_call_count",
        "loss_objective_calls_per_step",
        "backward_calls_per_step",
        "renderer_camera_batch_size_mean",
        "renderer_camera_batch_size_max",
        "loss_views_per_call_mean",
        "loss_views_per_call_max",
    ]

    for metric_name in required_metrics:
        assert metric_name in CANONICAL_METRICS, f"Metric {metric_name} not in registry"
        entry = CANONICAL_METRICS[metric_name]
        assert entry.name == metric_name
        assert entry.semantic_version >= 1
        assert entry.producer, f"{metric_name} has no producer"


# ===========================================================================
# TRUE CAMERA BATCH TESTS (C=4 execution-shape coverage)
# ===========================================================================

class _TrueBatchCamera:
    """Mock camera for true batch testing."""
    image_width = 2
    image_height = 2

    def __init__(self, uid: int):
        self.uid = uid


class _TrueBatchModel:
    """Mock model for true batch testing."""
    def __init__(self):
        self.get_xyz = torch.zeros((3, 3))
        self.provisional = torch.zeros(3, dtype=torch.bool)


class _TrueBatchSession:
    """Mock session for true batch testing."""
    parameter_version = 7

    def __init__(self, model):
        self.model = model

    def update_learning_rate(self, _iteration):
        return 0.01


class _TrueBatchEngine:
    """Mock engine for true batch testing."""
    def __init__(self, model):
        self.session = _TrueBatchSession(model)
        self._optimizer_step_calls = 0

    def run_post_backward_step(self, *_args, loss_breakdown, render_outputs, **_kwargs):
        self._optimizer_step_calls += 1
        return OptimizationStepResult(
            loss_breakdown=loss_breakdown,
            render_outputs=render_outputs,
            evidence=None,
            optimization_gpu_ms=None,
            map_version_before=11,
            map_version_after=11,
            topology_version=13,
        )

    def run_gradient_policies(self, context):
        pass


def _true_batch_request(camera: _TrueBatchCamera) -> OptimizationStepRequest:
    """Create a request for true batch testing."""
    return OptimizationStepRequest(
        view=OptimizationView(
            camera=camera,
            source_index=camera.uid,
            source_identity=SourceIdentity("test", 0, camera.uid, camera.uid),
            resolution_bucket="native",
        ),
        iteration=17,
        map_version=11,
        topology_version=13,
        resolution_bucket="native",
        evidence_requirements=EvidenceRequirements(want_supervision_evidence=False),
        random_state_token=RandomStateToken.capture(),
    )


def _true_batch_runtime(
    engine: _TrueBatchEngine, model: _TrueBatchModel
) -> OptimizationRuntimeContext:
    """Create runtime context for true batch testing."""
    return OptimizationRuntimeContext(
        session=engine.session,
        gaussians=model,
        args=SimpleNamespace(streaming_objective_step_regularizers=False, streaming_objective_batched_pixel_terms=False),
        opt=SimpleNamespace(iterations=100),
        pipe=SimpleNamespace(),
        gradient_policy_context=GradientPolicyContext(
            gaussians=model,
            args=SimpleNamespace(streaming_objective_step_regularizers=False, streaming_objective_batched_pixel_terms=False),
            iteration=17,
            n_frames_ingested=2,
            optimizer_update_id=19,
        ),
        n_frames_ingested=2,
        optimizer_update_id=19,
        optimizer_update_permitted=True,
    )


def test_true_batch_c4_single_group_execution_shape(monkeypatch):
    """C=4 in one render group: one render with C=4, four loss calls, one backward."""
    monkeypatch.setattr(
        executor_mod, "supports_true_camera_batch", lambda _pipe, _cameras: (True, "ok")
    )

    def _render(cameras, *_args, **_kwargs):
        return TrueCameraBatchResult(
            renders=torch.zeros((4, 3, 2, 2)),
            alphas=torch.ones((4, 1, 2, 2)),
            rendered_depth=None,
            rendered_depth_ed=None,
            meta={
                "camera_ids": torch.tensor([0, 1, 2, 3]),
                "gaussian_ids": torch.tensor([0, 1, 2, 0]),
            },
            camera_count=4,
            sparse_grad=True,
        )

    def _loss(inputs):
        loss = torch.tensor(float(inputs.viewpoint_cam.uid + 1), requires_grad=True)
        return LossBreakdown(
            total=loss.detach(),
            total_tensor=loss,
            components={"Ll1": loss.detach()},
            loss_finite=True,
        )

    monkeypatch.setattr(executor_mod, "render_true_camera_batch", _render)
    patch_assemble_loss(monkeypatch, executor_mod._objective, _loss)

    model = _TrueBatchModel()
    engine = _TrueBatchEngine(model)
    requests = tuple(_true_batch_request(_TrueBatchCamera(i)) for i in range(4))
    result = run_true_camera_batched_step(engine, requests, _true_batch_runtime(engine, model))

    # TRUE_CAMERA_BATCH_RENDER_ONLY shape: one render with C=4, loss loop, aggregated backward
    assert result.renderer_camera_batch_sizes == (4,)
    assert result.loss_objective_call_count == 4
    assert result.loss_views_per_call == (1, 1, 1, 1)  # Current main: per-view loss
    assert result.backward_call_count == 1  # One aggregated backward


def test_true_batch_c2_two_groups_execution_shape(monkeypatch):
    """C=2 in two groups (microbatch_size=2): two renders, four loss calls, two backwds."""
    monkeypatch.setattr(
        executor_mod, "supports_true_camera_batch", lambda _pipe, _cameras: (True, "ok")
    )

    render_calls = []

    def _render(cameras, *_args, **_kwargs):
        render_calls.append(len(cameras))
        camera_count = len(cameras)
        return TrueCameraBatchResult(
            renders=torch.zeros((camera_count, 3, 2, 2)),
            alphas=torch.ones((camera_count, 1, 2, 2)),
            rendered_depth=None,
            rendered_depth_ed=None,
            meta={
                "camera_ids": torch.arange(camera_count),
                "gaussian_ids": torch.arange(camera_count) % 3,
            },
            camera_count=camera_count,
            sparse_grad=True,
        )

    def _loss(inputs):
        loss = torch.tensor(float(inputs.viewpoint_cam.uid + 1), requires_grad=True)
        return LossBreakdown(
            total=loss.detach(),
            total_tensor=loss,
            components={"Ll1": loss.detach()},
            loss_finite=True,
        )

    monkeypatch.setattr(executor_mod, "render_true_camera_batch", _render)
    patch_assemble_loss(monkeypatch, executor_mod._objective, _loss)

    model = _TrueBatchModel()
    engine = _TrueBatchEngine(model)
    requests = tuple(_true_batch_request(_TrueBatchCamera(i)) for i in range(4))
    result = run_true_camera_batched_step(
        engine,
        requests,
        _true_batch_runtime(engine, model),
        camera_microbatch_size=2,
    )

    # Two groups of 2: two renders at C=2 each, four loss calls, two aggregated backwds
    assert result.renderer_camera_batch_sizes == (2, 2)
    assert result.loss_objective_call_count == 4
    assert result.loss_views_per_call == (1, 1, 1, 1)
    assert result.backward_call_count == 2  # One per group
