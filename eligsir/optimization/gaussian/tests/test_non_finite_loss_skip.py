"""Regression test for PR #39 review's P0 finding: a non-finite loss must
SKIP backward()/optimizer.step() entirely, not just get logged.

``train_streaming.py``'s original inline loop guards the entire
backward+post-backward-step sequence on
``if not _placement_only and _loss_finite:`` -- one bad live frame
producing a NaN/Inf loss must never touch parameters or optimizer state.
A first version of the extracted ``GaussianOptimizationEngine.run_step()``
dropped this guard: ``objective.assemble_loss()`` detected the non-finite
loss and zero_grad()'d defensively, but ``run_step()`` still called
``render_and_backward()`` (running ``loss.backward()`` on a NaN tensor,
poisoning every parameter's ``.grad``) and then ``run_post_backward_step()``
(running the optimizer step on those poisoned grads).

Fix: ``LossBreakdown.loss_finite`` now carries the signal end-to-end, and
both ``run_step()`` (before calling ``render_and_backward``) and
``run_post_backward_step()`` (defense-in-depth, since it's a public
"granular method" a caller may invoke directly) check it and skip.
"""

from __future__ import annotations

import copy

import pytest

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="needs a real CUDA/ROCm device for the gsplat rasterizer",
)

from eligsir.optimization.gaussian.tests.test_gpu_parity_harness import (  # noqa: E402
    _build_args,
    _build_camera,
    _build_gaussians,
    _bg,
    _snapshot_adam_state,
    _snapshot_grads,
    _snapshot_params,
)


def _tensors_equal(a, b) -> bool:
    if a is None or b is None:
        return a is b
    return bool(torch.equal(a, b))


def test_non_finite_loss_skips_backward_and_optimizer_step():
    from eligsir.map.contracts import GaussianOptimizationSession
    from eligsir.optimization.gaussian import objective as objective_mod
    from eligsir.optimization.gaussian.contracts import (
        EvidenceRequirements, GradientPolicyContext, OptimizationRuntimeContext, OptimizationStepRequest,
        OptimizationView, RandomStateToken,
    )
    from eligsir.optimization.gaussian.engine import GaussianOptimizationEngine
    from eligsir.optimization.gaussian.loss_breakdown import LossBreakdown

    torch.manual_seed(0)
    gaussians = _build_gaussians(seed=0)
    cam = _build_camera(seed=1)
    args, opt, pipe, _ns = _build_args()

    class _FakeSession(GaussianOptimizationSession):
        def zero_grad(self):
            gaussians.optimizer.zero_grad(set_to_none=True)

        def step(self, visibility=None):
            gaussians.optimizer.step()

        def update_learning_rate(self, iteration):
            return None

    class _FakeBackend:
        def begin_optimization(self):
            return _FakeSession()

    engine = GaussianOptimizationEngine(_FakeBackend())

    class _NoOpMcmcStrategy:
        def step_pre_backward(self, **kwargs):
            pass

    params_before = _snapshot_params(gaussians)
    grads_before = _snapshot_grads(gaussians)
    adam_before = _snapshot_adam_state(gaussians)

    # Inject a non-finite loss: monkeypatch assemble_loss to return the
    # exact shape a real non-finite-loss iteration would produce (a NaN
    # total_tensor with requires_grad=True, loss_finite=False), matching
    # what objective.assemble_loss() itself constructs on that path.
    real_assemble_loss = objective_mod.assemble_loss

    def _fake_assemble_loss(inputs):
        nan_loss = gaussians.get_xyz.sum() * float("nan")
        return LossBreakdown(
            total=float("nan"), components={}, total_tensor=nan_loss,
            loss_finite=False,
        )

    objective_mod.assemble_loss = _fake_assemble_loss
    try:
        context = GradientPolicyContext(
            gaussians=gaussians, args=args, iteration=1, n_frames_ingested=0,
        )
        result = engine.run_step(
            OptimizationStepRequest(
                view=OptimizationView(camera=cam, background=_bg()), iteration=1,
                map_version=0, topology_version=0, resolution_bucket="default",
                evidence_requirements=EvidenceRequirements(),
                random_state_token=RandomStateToken.capture(),
            ),
            OptimizationRuntimeContext(
                gaussians=gaussians, args=args, opt=opt, pipe=pipe,
                mcmc_strategy=_NoOpMcmcStrategy(), gradient_policy_context=context,
            ),
        )
    finally:
        objective_mod.assemble_loss = real_assemble_loss

    assert result.step_allowed is False
    assert result.skip_reason == "non_finite_loss"
    assert result.loss_breakdown.loss_finite is False

    params_after = _snapshot_params(gaussians)
    grads_after = _snapshot_grads(gaussians)
    adam_after = _snapshot_adam_state(gaussians)

    for name in params_before:
        assert _tensors_equal(params_before[name], params_after[name]), (
            f"param {name!r} changed despite a non-finite loss -- "
            "optimizer.step() must not have run"
        )
    # zero_grad() was called (mirrors the original inline guard's own
    # defensive zero_grad on the non-finite path), so every grad must be
    # None both before (fresh model) and after.
    for name in grads_before:
        assert grads_after[name] is None, (
            f"param {name!r} has a non-None grad after a non-finite-loss "
            "step -- backward() must not have run, and zero_grad() should "
            "have cleared anything stale"
        )

    for name in adam_before:
        before = adam_before[name]
        after = adam_after[name]
        assert before["step"] == after["step"], (
            f"Adam step counter for {name!r} advanced despite a "
            "non-finite loss -- optimizer.step() must not have run"
        )
        assert _tensors_equal(before["exp_avg"], after["exp_avg"])
        assert _tensors_equal(before["exp_avg_sq"], after["exp_avg_sq"])


def test_run_post_backward_step_guards_directly_against_non_finite_loss_breakdown():
    """Defense-in-depth: run_post_backward_step() is a public "granular
    method" per run_step()'s own docstring -- a caller invoking it directly
    (not through run_step()) with a non-finite loss_breakdown must also be
    refused, not just the composed end-to-end wrapper."""
    from eligsir.map.contracts import GaussianOptimizationSession
    from eligsir.optimization.gaussian.contracts import GradientPolicyContext
    from eligsir.optimization.gaussian.engine import GaussianOptimizationEngine
    from eligsir.optimization.gaussian.loss_breakdown import LossBreakdown

    gaussians = _build_gaussians(seed=2)
    args, _opt, pipe, _ns = _build_args()

    class _FakeSession(GaussianOptimizationSession):
        def zero_grad(self):
            gaussians.optimizer.zero_grad(set_to_none=True)

        def step(self, visibility=None):
            gaussians.optimizer.step()

        def update_learning_rate(self, iteration):
            return gaussians.update_learning_rate(iteration)

    class _FakeBackend:
        def begin_optimization(self):
            return _FakeSession()

    engine = GaussianOptimizationEngine(_FakeBackend())
    context = GradientPolicyContext(
        gaussians=gaussians, args=args, iteration=1, n_frames_ingested=0,
    )
    params_before = _snapshot_params(gaussians)

    non_finite_breakdown = LossBreakdown(
        total=float("nan"), components={}, total_tensor=None, loss_finite=False,
    )
    result = engine.run_post_backward_step(
        gaussians, args, pipe, context,
        loss_breakdown=non_finite_breakdown, render_outputs=None,
    )

    assert result.step_allowed is False
    assert result.skip_reason == "non_finite_loss"
    params_after = _snapshot_params(gaussians)
    for name in params_before:
        assert _tensors_equal(params_before[name], params_after[name])
