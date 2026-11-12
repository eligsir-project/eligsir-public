from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from _prepared_loss_fakes import patch_assemble_loss

from eligsir.mapping.renderer.true_camera_batch import TrueCameraBatchResult
from eligsir.optimization.gaussian.contracts import (
    EvidenceRequirements,
    GradientPolicyContext,
    OptimizationRuntimeContext,
    OptimizationStepRequest,
    OptimizationStepResult,
    OptimizationView,
    RenderOutputs,
    RandomStateToken,
)
from eligsir.optimization.gaussian.loss_breakdown import LossBreakdown
from eligsir.optimization.gaussian.true_batch_executor import (
    run_true_camera_batched_step,
)


class _Camera:
    image_width = 2
    image_height = 2

    def __init__(self, uid: int):
        self.uid = uid


class _Model:
    def __init__(self):
        self.get_xyz = torch.zeros((3, 3))
        self.provisional = torch.zeros(3, dtype=torch.bool)
        self.post_render_mutations = 0


class _Session:
    parameter_version = 0

    def __init__(self, model):
        self.model = model

    def update_learning_rate(self, _iteration):
        return 0.01


class _Engine:
    def __init__(self, model):
        self.session = _Session(model)
        self._optimizer_step_calls = 0
        self.visible = None
        self.policy_optimizer_update_ids = []

    def run_post_backward_step(
        self,
        _gaussians,
        _args,
        _pipe,
        context,
        *,
        visible,
        loss_breakdown,
        render_outputs,
        **_kwargs,
    ):
        self.visible = visible.clone()
        self.policy_optimizer_update_ids.append(context.optimizer_update_id)
        self._optimizer_step_calls += 1
        return OptimizationStepResult(
            loss_breakdown=loss_breakdown,
            render_outputs=render_outputs,
            evidence=None,
            optimization_gpu_ms=None,
            map_version_before=0,
            map_version_after=0,
            topology_version=0,
        )


def _request(camera: _Camera, *, render_depth: bool = False) -> OptimizationStepRequest:
    return OptimizationStepRequest(
        view=OptimizationView(
            camera=camera,
            source_index=camera.uid,
            background=None,
            update_sh_rest=True,
            render_depth=render_depth,
            resolution_bucket="native",
        ),
        iteration=1,
        map_version=0,
        topology_version=0,
        resolution_bucket="native",
        evidence_requirements=EvidenceRequirements(),
        random_state_token=RandomStateToken.capture(),
    )


def _patch_successful_true_c(monkeypatch):
    from eligsir.optimization.gaussian import true_batch_executor as executor

    monkeypatch.setattr(
        executor,
        "supports_true_camera_batch",
        lambda _pipe, _cams: (True, "ok"),
    )
    monkeypatch.setattr(
        executor,
        "render_true_camera_batch",
        lambda group, *_args, **_kwargs: TrueCameraBatchResult(
            renders=torch.zeros((len(group), 3, 2, 2)),
            alphas=torch.ones((len(group), 1, 2, 2)),
            rendered_depth=None,
            rendered_depth_ed=None,
            meta={"radii": torch.ones((len(group), 3))},
            camera_count=len(group),
            sparse_grad=True,
        ),
    )
    patch_assemble_loss(monkeypatch, executor._objective, lambda _inputs: LossBreakdown(
            total=torch.tensor(1.0),
            total_tensor=torch.tensor(1.0, requires_grad=True),
            components={},
            loss_finite=True,
        ),
    )


def test_two_true_c2_microbatches_accumulate_then_step_once(monkeypatch):
    from eligsir.optimization.gaussian import true_batch_executor as executor

    cameras = tuple(_Camera(index) for index in range(4))
    model = _Model()
    engine = _Engine(model)
    render_groups = []
    losses = []

    monkeypatch.setattr(executor, "supports_true_camera_batch", lambda _pipe, _cams: (True, "ok"))

    def _render(group, *_args, **_kwargs):
        render_groups.append(tuple(camera.uid for camera in group))
        count = len(group)
        return TrueCameraBatchResult(
            renders=torch.zeros((count, 3, 2, 2)),
            alphas=torch.ones((count, 1, 2, 2)),
            rendered_depth=None,
            rendered_depth_ed=None,
            meta={"radii": torch.ones((count, 3))},
            camera_count=count,
            sparse_grad=True,
        )

    def _loss(_inputs):
        tensor = torch.tensor(1.0, requires_grad=True)
        losses.append(tensor)
        return LossBreakdown(
            total=tensor.detach(),
            total_tensor=tensor,
            components={"photometric": tensor.detach()},
            loss_finite=True,
        )

    monkeypatch.setattr(executor, "render_true_camera_batch", _render)
    patch_assemble_loss(monkeypatch, executor._objective, _loss)
    runtime = OptimizationRuntimeContext(
        session=engine.session,
        gaussians=model,
        args=SimpleNamespace(streaming_objective_step_regularizers=False, streaming_objective_batched_pixel_terms=False),
        opt=SimpleNamespace(iterations=100),
        pipe=SimpleNamespace(),
        mcmc_strategy=None,
        gradient_policy_context=GradientPolicyContext(
            gaussians=model,
            args=SimpleNamespace(streaming_objective_step_regularizers=False, streaming_objective_batched_pixel_terms=False),
            iteration=1,
            n_frames_ingested=4,
        ),
        n_frames_ingested=4,
    )

    result = run_true_camera_batched_step(
        engine,
        tuple(_request(camera) for camera in cameras),
        runtime,
        camera_microbatch_size=2,
    )

    assert render_groups == [(0, 1), (2, 3)]
    assert [loss.grad.item() for loss in losses] == pytest.approx([0.25] * 4)
    assert result.optimizer_step_count == 1
    assert result.rasterizer_call_count == 2
    assert result.camera_batch_rasterizer_calls == 2
    assert result.microbatch_sizes == (2, 2)
    assert result.supervision_view_count == 4
    assert torch.equal(engine.visible, torch.ones(3, dtype=torch.bool))


@pytest.mark.parametrize(
    ("camera_microbatch_size", "expected_sizes"),
    [(1, (1,) * 16), (4, (4,) * 4)],
)
def test_b16_existing_executor_partitions_c1_or_c4_and_steps_once(
    monkeypatch, camera_microbatch_size, expected_sizes
):
    """B16/C1 and B16/C4 use existing accumulation and one optimizer boundary."""
    cameras = tuple(_Camera(index) for index in range(16))
    model = _Model()
    engine = _Engine(model)
    _patch_successful_true_c(monkeypatch)
    if camera_microbatch_size == 1:
        from eligsir.optimization.gaussian import true_batch_executor as executor

        monkeypatch.setattr(
            executor,
            "invoke_render",
            lambda *_args, **_kwargs: RenderOutputs(
                payload={
                    "render": torch.zeros((3, 2, 2)),
                    "visibility_filter": torch.ones(3, dtype=torch.bool),
                }
            ),
        )
    runtime = OptimizationRuntimeContext(
        session=engine.session,
        gaussians=model,
        args=SimpleNamespace(),
        opt=SimpleNamespace(iterations=100),
        pipe=SimpleNamespace(),
        gradient_policy_context=GradientPolicyContext(
            gaussians=model,
            args=SimpleNamespace(),
            iteration=1,
            n_frames_ingested=16,
        ),
        n_frames_ingested=16,
    )

    result = run_true_camera_batched_step(
        engine,
        tuple(_request(camera) for camera in cameras),
        runtime,
        camera_microbatch_size=camera_microbatch_size,
    )

    assert result.optimizer_step_count == 1
    assert result.microbatch_sizes == expected_sizes
    assert result.supervision_view_count == 16
    assert engine._optimizer_step_calls == 1


def test_true_c16_two_groups_updates_fixed_mcmc_visibility_once_from_union(monkeypatch):
    """Two C8 renders still produce one C16 committed-visibility EMA update."""
    from eligsir.optimization.gaussian import true_batch_executor as executor

    cameras = tuple(_Camera(index) for index in range(16))
    model = _Model()
    model.visibility_ema = torch.full((3, 1), 0.5)
    engine = _Engine(model)
    render_groups = []
    loss_active_masks = []
    monkeypatch.setattr(executor, "supports_true_camera_batch", lambda _pipe, _cams: (True, "ok"))

    def _render(group, *_args, **_kwargs):
        render_groups.append(tuple(camera.uid for camera in group))
        radii = torch.zeros((len(group), 3))
        for index, camera in enumerate(group):
            if camera.uid == 0:
                radii[index, 0] = 1
            if camera.uid == 1:
                radii[index, 1] = 1
            if camera.uid == 8:
                radii[index, 2] = 1
        return TrueCameraBatchResult(
            renders=torch.zeros((len(group), 3, 2, 2)),
            alphas=torch.ones((len(group), 1, 2, 2)),
            rendered_depth=None,
            rendered_depth_ed=None,
            meta={"radii": radii},
            camera_count=len(group),
            sparse_grad=True,
        )

    monkeypatch.setattr(executor, "render_true_camera_batch", _render)
    class _Strategy:
        def step_pre_backward(self, **_kwargs):
            pass

        def inject_noise(self, **_kwargs):
            pass

    context = GradientPolicyContext(
        gaussians=model, args=SimpleNamespace(streaming_objective_step_regularizers=False, streaming_objective_batched_pixel_terms=False), iteration=1,
        n_frames_ingested=16, optimizer_update_id=7,
    )
    runtime = OptimizationRuntimeContext(
        session=engine.session,
        gaussians=model,
        args=SimpleNamespace(streaming_objective_step_regularizers=False, streaming_objective_batched_pixel_terms=False),
        opt=SimpleNamespace(iterations=100),
        pipe=SimpleNamespace(),
        mcmc_strategy=_Strategy(),
        gradient_policy_context=context,
        n_frames_ingested=16,
        optimizer_update_id=7,
        optimizer_update_permitted=True,
        fixed_shape_mcmc=True,
        post_render_hook=lambda _out: None,
    )

    def _loss(inputs):
        loss_active_masks.append(context.active_set.visible_mask.clone())
        tensor = torch.tensor(1.0, requires_grad=True)
        return LossBreakdown(
            total=tensor.detach(), total_tensor=tensor, components={},
            loss_finite=inputs.viewpoint_cam.uid != 1,
        )

    patch_assemble_loss(monkeypatch, executor._objective, _loss)

    result = run_true_camera_batched_step(
        engine,
        tuple(_request(camera) for camera in cameras),
        runtime,
        camera_microbatch_size=8,
    )

    assert render_groups == [tuple(range(8)), tuple(range(8, 16))]
    assert result.optimizer_step_count == 1
    assert result.skipped_view_count == 1
    assert torch.equal(loss_active_masks[0], torch.tensor([True, True, False]))
    assert torch.equal(
        model.last_supervised_iter,
        torch.tensor([7, 0, 7], dtype=torch.int32),
    )
    torch.testing.assert_close(
        model.visibility_ema,
        torch.tensor([[0.51], [0.49], [0.51]]),
        rtol=0,
        atol=1e-7,
    )


def test_true_c_nonfinite_secondary_does_not_contribute_to_visibility_ema(monkeypatch):
    from eligsir.optimization.gaussian import true_batch_executor as executor

    cameras = (_Camera(0), _Camera(1))
    model = _Model()
    model.visibility_ema = torch.full((3, 1), 0.5)
    engine = _Engine(model)
    _patch_successful_true_c(monkeypatch)
    monkeypatch.setattr(
        executor,
        "render_true_camera_batch",
        lambda group, *_args, **_kwargs: TrueCameraBatchResult(
            renders=torch.zeros((len(group), 3, 2, 2)),
            alphas=torch.ones((len(group), 1, 2, 2)),
            rendered_depth=None,
            rendered_depth_ed=None,
            meta={"radii": torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])},
            camera_count=len(group),
            sparse_grad=True,
        ),
    )

    def _loss(inputs):
        tensor = torch.tensor(1.0, requires_grad=True)
        return LossBreakdown(
            total=tensor.detach(), total_tensor=tensor, components={},
            loss_finite=inputs.viewpoint_cam.uid == 0,
        )

    patch_assemble_loss(monkeypatch, executor._objective, _loss)

    class _Strategy:
        def step_pre_backward(self, **_kwargs):
            pass

        def inject_noise(self, **_kwargs):
            pass

    runtime = OptimizationRuntimeContext(
        session=engine.session,
        gaussians=model,
        args=SimpleNamespace(streaming_objective_step_regularizers=False, streaming_objective_batched_pixel_terms=False),
        opt=SimpleNamespace(iterations=100),
        pipe=SimpleNamespace(),
        mcmc_strategy=_Strategy(),
        gradient_policy_context=GradientPolicyContext(
            gaussians=model, args=SimpleNamespace(streaming_objective_step_regularizers=False, streaming_objective_batched_pixel_terms=False), iteration=1,
            n_frames_ingested=2, optimizer_update_id=7,
        ),
        n_frames_ingested=2,
        optimizer_update_id=7,
        optimizer_update_permitted=True,
        fixed_shape_mcmc=True,
    )

    result = run_true_camera_batched_step(
        engine, tuple(_request(camera) for camera in cameras), runtime
    )

    assert result.optimizer_step_count == 1
    assert result.skipped_view_count == 1
    assert torch.equal(
        model.last_supervised_iter,
        torch.tensor([7, 0, 0], dtype=torch.int32),
    )
    torch.testing.assert_close(
        model.visibility_ema,
        torch.tensor([[0.51], [0.49], [0.49]]),
        rtol=0,
        atol=1e-7,
    )


def test_true_c_fixed_mcmc_without_optimizer_commit_leaves_visibility_ema_unchanged(
    monkeypatch,
):
    cameras = (_Camera(0), _Camera(1))
    model = _Model()
    model.visibility_ema = torch.full((3, 1), 0.5)
    engine = _Engine(model)
    _patch_successful_true_c(monkeypatch)
    runtime = OptimizationRuntimeContext(
        session=engine.session,
        gaussians=model,
        args=SimpleNamespace(streaming_objective_step_regularizers=False, streaming_objective_batched_pixel_terms=False),
        opt=SimpleNamespace(iterations=100),
        pipe=SimpleNamespace(),
        gradient_policy_context=GradientPolicyContext(
            gaussians=model, args=SimpleNamespace(streaming_objective_step_regularizers=False, streaming_objective_batched_pixel_terms=False), iteration=1,
            n_frames_ingested=2, optimizer_update_id=7,
        ),
        n_frames_ingested=2,
        optimizer_update_id=7,
        optimizer_update_permitted=False,
        fixed_shape_mcmc=True,
    )

    result = run_true_camera_batched_step(
        engine, tuple(_request(camera) for camera in cameras), runtime
    )

    assert result.optimizer_step_count == 0
    torch.testing.assert_close(model.visibility_ema, torch.full((3, 1), 0.5))


def test_existing_true_c_path_remains_one_render_when_microbatching_is_disabled(monkeypatch):
    from eligsir.optimization.gaussian import true_batch_executor as executor

    cameras = tuple(_Camera(index) for index in range(4))
    model = _Model()
    engine = _Engine(model)
    renders = []
    monkeypatch.setattr(executor, "supports_true_camera_batch", lambda _pipe, _cams: (True, "ok"))

    def _render(group, *_args, **_kwargs):
        renders.append(tuple(group))
        return TrueCameraBatchResult(
            renders=torch.zeros((4, 3, 2, 2)),
            alphas=torch.ones((4, 1, 2, 2)),
            rendered_depth=None,
            rendered_depth_ed=None,
            meta={"radii": torch.ones((4, 3))},
            camera_count=4,
            sparse_grad=True,
        )

    monkeypatch.setattr(executor, "render_true_camera_batch", _render)
    patch_assemble_loss(monkeypatch, executor._objective, lambda _inputs: LossBreakdown(
            total=torch.tensor(1.0),
            total_tensor=torch.tensor(1.0, requires_grad=True),
            components={},
            loss_finite=True,
        ),
    )
    runtime = OptimizationRuntimeContext(
        session=engine.session,
        gaussians=model,
        args=SimpleNamespace(streaming_objective_step_regularizers=False, streaming_objective_batched_pixel_terms=False),
        opt=SimpleNamespace(iterations=100),
        pipe=SimpleNamespace(),
        gradient_policy_context=GradientPolicyContext(
            gaussians=model, args=SimpleNamespace(streaming_objective_step_regularizers=False, streaming_objective_batched_pixel_terms=False), iteration=1, n_frames_ingested=4
        ),
        n_frames_ingested=4,
    )

    result = run_true_camera_batched_step(
        engine, tuple(_request(camera) for camera in cameras), runtime
    )

    assert len(renders) == 1
    assert result.rasterizer_call_count == 1
    assert result.camera_batch_rasterizer_calls == 1
    assert result.optimizer_step_count == 1


def test_true_c_denied_then_permitted_has_no_gradient_contamination(monkeypatch):
    from eligsir.optimization.gaussian import true_batch_executor as executor

    cameras = (_Camera(0), _Camera(1))
    model = _Model()
    engine = _Engine(model)
    losses = []
    monkeypatch.setattr(executor, "supports_true_camera_batch", lambda _pipe, _cams: (True, "ok"))
    monkeypatch.setattr(
        executor,
        "render_true_camera_batch",
        lambda group, *_args, **_kwargs: TrueCameraBatchResult(
            renders=torch.zeros((len(group), 3, 2, 2)),
            alphas=torch.ones((len(group), 1, 2, 2)),
            rendered_depth=None,
            rendered_depth_ed=None,
            meta={"radii": torch.ones((len(group), 3))},
            camera_count=len(group),
            sparse_grad=True,
        ),
    )

    def _loss(_inputs):
        tensor = torch.tensor(1.0, requires_grad=True)
        losses.append(tensor)
        return LossBreakdown(
            total=tensor.detach(), total_tensor=tensor, components={}, loss_finite=True
        )

    patch_assemble_loss(monkeypatch, executor._objective, _loss)
    def _runtime(target_engine, target_model, *, permitted):
        def _post_render(_outputs):
            target_model.post_render_mutations += 1

        return OptimizationRuntimeContext(
            session=target_engine.session,
            gaussians=target_model,
            args=SimpleNamespace(streaming_objective_step_regularizers=False, streaming_objective_batched_pixel_terms=False),
            opt=SimpleNamespace(iterations=100),
            pipe=SimpleNamespace(),
            gradient_policy_context=GradientPolicyContext(
                gaussians=target_model,
                args=SimpleNamespace(streaming_objective_step_regularizers=False, streaming_objective_batched_pixel_terms=False),
                iteration=999,
                n_frames_ingested=2,
                optimizer_update_id=7,
            ),
            n_frames_ingested=2,
            optimizer_update_id=7,
            optimizer_update_permitted=permitted,
            post_render_hook=_post_render,
        )

    runtime = _runtime(engine, model, permitted=False)

    denied = run_true_camera_batched_step(
        engine, tuple(_request(camera) for camera in cameras), runtime
    )
    denied_losses = tuple(losses)
    assert model.post_render_mutations == 0
    runtime.optimizer_update_permitted = True
    permitted = run_true_camera_batched_step(
        engine, tuple(_request(camera) for camera in cameras), runtime
    )
    retried_grads = tuple(loss.grad.detach().clone() for loss in losses[2:])

    clean_model = _Model()
    clean_engine = _Engine(clean_model)
    clean_runtime = _runtime(clean_engine, clean_model, permitted=True)
    clean = run_true_camera_batched_step(
        clean_engine, tuple(_request(camera) for camera in cameras), clean_runtime
    )
    clean_grads = tuple(loss.grad.detach().clone() for loss in losses[4:])

    assert denied.optimizer_step_count == 0
    assert denied.backward_ran is False
    assert all(loss.grad is None for loss in denied_losses)
    assert permitted.optimizer_step_count == clean.optimizer_step_count == 1
    assert model.post_render_mutations == clean_model.post_render_mutations == 1
    torch.testing.assert_close(retried_grads, clean_grads, rtol=0, atol=0)
    assert [grad.item() for grad in retried_grads] == pytest.approx([0.5, 0.5])


def test_true_c_synthesized_gradient_context_uses_runtime_optimizer_update_id(
    monkeypatch,
):
    cameras = (_Camera(0), _Camera(1))
    model = _Model()
    engine = _Engine(model)
    _patch_successful_true_c(monkeypatch)
    runtime = OptimizationRuntimeContext(
        session=engine.session,
        gaussians=model,
        args=SimpleNamespace(streaming_objective_step_regularizers=False, streaming_objective_batched_pixel_terms=False),
        opt=SimpleNamespace(iterations=100),
        pipe=SimpleNamespace(),
        n_frames_ingested=1,
        optimizer_update_id=7,
        optimizer_update_permitted=True,
    )

    run_true_camera_batched_step(
        engine, tuple(_request(camera) for camera in cameras), runtime
    )

    assert engine.policy_optimizer_update_ids == [7]


def test_true_c_supplied_gradient_context_must_match_runtime_optimizer_update_id(
    monkeypatch,
):
    cameras = (_Camera(0), _Camera(1))
    model = _Model()
    engine = _Engine(model)
    _patch_successful_true_c(monkeypatch)
    runtime = OptimizationRuntimeContext(
        session=engine.session,
        gaussians=model,
        args=SimpleNamespace(streaming_objective_step_regularizers=False, streaming_objective_batched_pixel_terms=False),
        opt=SimpleNamespace(iterations=100),
        pipe=SimpleNamespace(),
        gradient_policy_context=GradientPolicyContext(
            gaussians=model,
            args=SimpleNamespace(streaming_objective_step_regularizers=False, streaming_objective_batched_pixel_terms=False),
            iteration=1,
            n_frames_ingested=1,
            optimizer_update_id=8,
        ),
        n_frames_ingested=1,
        optimizer_update_id=7,
        optimizer_update_permitted=True,
    )

    with pytest.raises(ValueError, match="optimizer_update_id must match"):
        run_true_camera_batched_step(
            engine, tuple(_request(camera) for camera in cameras), runtime
        )


def test_true_c_rejects_hybrid_canonical_and_legacy_optimizer_clock(monkeypatch):
    cameras = (_Camera(0), _Camera(1))
    model = _Model()
    engine = _Engine(model)
    _patch_successful_true_c(monkeypatch)
    runtime = OptimizationRuntimeContext(
        session=engine.session,
        gaussians=model,
        args=SimpleNamespace(streaming_objective_step_regularizers=False, streaming_objective_batched_pixel_terms=False),
        opt=SimpleNamespace(iterations=100),
        pipe=SimpleNamespace(),
        gradient_policy_context=GradientPolicyContext(
            gaussians=model,
            args=SimpleNamespace(streaming_objective_step_regularizers=False, streaming_objective_batched_pixel_terms=False),
            iteration=1,
            n_frames_ingested=1,
            optimizer_update_id=7,
        ),
        n_frames_ingested=1,
    )
    runtime.optimizer_update_id = 7

    with pytest.raises(ValueError, match="must either both be None or both be set"):
        run_true_camera_batched_step(
            engine, tuple(_request(camera) for camera in cameras), runtime
        )


def test_true_c_mcmc_schedules_use_semantic_update_id(monkeypatch):
    from eligsir.optimization.gaussian import true_batch_executor as executor

    cameras = (_Camera(0), _Camera(1))
    model = _Model()
    engine = _Engine(model)
    calls = {"pre": [], "noise": []}
    monkeypatch.setattr(executor, "supports_true_camera_batch", lambda _pipe, _cams: (True, "ok"))
    monkeypatch.setattr(
        executor,
        "render_true_camera_batch",
        lambda group, *_args, **_kwargs: TrueCameraBatchResult(
            renders=torch.zeros((len(group), 3, 2, 2)),
            alphas=torch.ones((len(group), 1, 2, 2)),
            rendered_depth=None,
            rendered_depth_ed=None,
            meta={"radii": torch.ones((len(group), 3))},
            camera_count=len(group),
            sparse_grad=True,
        ),
    )
    patch_assemble_loss(monkeypatch, executor._objective, lambda _inputs: LossBreakdown(
            total=torch.tensor(1.0),
            total_tensor=torch.tensor(1.0, requires_grad=True),
            components={},
            loss_finite=True,
        ),
    )

    class _Strategy:
        def step_pre_backward(self, **kwargs):
            calls["pre"].append(kwargs["iteration"])

        def inject_noise(self, **kwargs):
            calls["noise"].append(kwargs["iteration"])

    runtime = OptimizationRuntimeContext(
        session=engine.session,
        gaussians=model,
        args=SimpleNamespace(streaming_objective_step_regularizers=False, streaming_objective_batched_pixel_terms=False),
        opt=SimpleNamespace(iterations=100),
        pipe=SimpleNamespace(),
        mcmc_strategy=_Strategy(),
        gradient_policy_context=GradientPolicyContext(
            gaussians=model, args=SimpleNamespace(streaming_objective_step_regularizers=False, streaming_objective_batched_pixel_terms=False), iteration=10_000,
            n_frames_ingested=2, optimizer_update_id=7,
        ),
        n_frames_ingested=2,
        optimizer_update_id=7,
        optimizer_update_permitted=True,
        fixed_shape_mcmc=True,
    )

    result = run_true_camera_batched_step(
        engine, tuple(_request(camera) for camera in cameras), runtime
    )

    assert result.optimizer_step_count == 1
    assert calls == {"pre": [7, 7], "noise": [7]}


def test_true_camera_depth_samples_sum_only_finite_views_in_the_update(monkeypatch):
    from eligsir.optimization.gaussian import true_batch_executor as executor

    cameras = tuple(_Camera(index) for index in range(3))
    model = _Model()
    engine = _Engine(model)
    monkeypatch.setattr(
        executor, "supports_true_camera_batch", lambda _pipe, _cams: (True, "ok")
    )

    def _render(group, *_args, **_kwargs):
        count = len(group)
        return TrueCameraBatchResult(
            renders=torch.zeros((count, 3, 2, 2)),
            alphas=torch.ones((count, 1, 2, 2)),
            rendered_depth=torch.ones((count, 1, 2, 2)),
            rendered_depth_ed=torch.ones((count, 1, 2, 2)),
            meta={"radii": torch.ones((count, 3))},
            camera_count=count,
            sparse_grad=True,
        )

    def _loss(inputs):
        counts = {0: 2, 1: 100, 2: 5}
        finite = inputs.viewpoint_cam.uid != 1
        tensor = torch.tensor(1.0, requires_grad=True)
        return LossBreakdown(
            total=tensor.detach(),
            total_tensor=tensor,
            components={},
            loss_finite=finite,
            depth_valid_sample_count=counts[inputs.viewpoint_cam.uid],
        )

    monkeypatch.setattr(executor, "render_true_camera_batch", _render)
    patch_assemble_loss(monkeypatch, executor._objective, _loss)
    runtime = OptimizationRuntimeContext(
        session=engine.session,
        gaussians=model,
        args=SimpleNamespace(streaming_objective_step_regularizers=False, streaming_objective_batched_pixel_terms=False),
        opt=SimpleNamespace(iterations=100),
        pipe=SimpleNamespace(),
        gradient_policy_context=GradientPolicyContext(
            gaussians=model, args=SimpleNamespace(streaming_objective_step_regularizers=False, streaming_objective_batched_pixel_terms=False), iteration=1, n_frames_ingested=3
        ),
        n_frames_ingested=3,
    )

    result = run_true_camera_batched_step(
        engine,
        tuple(_request(camera, render_depth=True) for camera in cameras),
        runtime,
    )

    assert result.optimizer_step_count == 1
    assert result.depth_valid_sample_count == 7
