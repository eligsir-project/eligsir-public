"""Executor wiring for ``--streaming_objective_step_regularizers`` (issue
#1670 Phase A -- see ``.superpowers/sdd/2026-09-03-throughput-recovery-
tracks-1-2/step-regularizers-brief.md``).

``run_true_camera_batched_step`` (``eligsir/optimization/gaussian/
true_batch_executor.py``) is the true-camera-batch (B4) executor. When the
flag is on AND the group has more than one view AND
``runtime.sparse_active_set`` AND the differentiable-energy-loss path is
off, it should call ``losses.assemble_step_regularizers`` exactly ONCE for
the whole group (instead of relying on ``assemble_loss``'s own per-view
loop), pass ``skip_gaussian_regularizers=True`` into every view's
``LossAssemblyInputs``, and add the (already 1/B-scaled) result to the
group's aggregate total + components. Flag off (default), a B=1 group, or
either ineligibility condition must fall back to today's behavior exactly
-- ``assemble_step_regularizers`` never called, ``skip_gaussian_regularizers``
always False.

Mirrors the fakes/pattern in ``tests/test_true_camera_microbatch_accumulation.py``.

Divisor-contract fix (see ``.superpowers/sdd/2026-09-03-throughput-
recovery-tracks-1-2/step-regularizers-fix-brief.md``): the executor MUST
pass ``batch_size=requested_b`` (the call's TRUE global divisor, including
any ``skipped_view_count`` or other groups folded into the same call) to
``assemble_step_regularizers`` -- NOT ``len(group)``, which under
microbatching/skipped views can be strictly smaller and silently
over-weights the term by ``requested_b / len(group)`` (reviewer reproduced
2.0000x with two 2-view groups). The tests below that exercise a
multi-group or ``skipped_view_count > 0`` call use a mock proportional to
``weights.sum()`` (rather than a constant) specifically so an incorrect
``batch_size`` at the call site changes the asserted total, not just the
recorded argument.
"""

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
    def __init__(self, n: int = 3):
        self.get_xyz = torch.zeros((n, 3))
        self.get_opacity = torch.sigmoid(torch.randn(n, 1))
        self.get_scaling = torch.rand(n, 3) + 0.1
        self.provisional = torch.zeros(n, dtype=torch.bool)


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

    def run_post_backward_step(self, _gaussians, _args, _pipe, context, *, visible, loss_breakdown, render_outputs, **_kwargs):
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


def _request(camera: _Camera) -> OptimizationStepRequest:
    return OptimizationStepRequest(
        view=OptimizationView(
            camera=camera,
            source_index=camera.uid,
            background=None,
            update_sh_rest=True,
            render_depth=False,
            resolution_bucket="native",
        ),
        iteration=1,
        map_version=0,
        topology_version=0,
        resolution_bucket="native",
        evidence_requirements=EvidenceRequirements(),
        random_state_token=RandomStateToken.capture(),
    )


def _patch_render(monkeypatch, executor, *, population: int):
    monkeypatch.setattr(executor, "supports_true_camera_batch", lambda _pipe, _cams: (True, "ok"))

    def _render(group, *_args, **_kwargs):
        count = len(group)
        return TrueCameraBatchResult(
            renders=torch.zeros((count, 3, 2, 2)),
            alphas=torch.ones((count, 1, 2, 2)),
            rendered_depth=None,
            rendered_depth_ed=None,
            meta={"radii": torch.ones((count, population))},
            camera_count=count,
            sparse_grad=True,
        )

    monkeypatch.setattr(executor, "render_true_camera_batch", _render)


def _patch_invoke_render(monkeypatch, executor, *, population: int):
    """For a trailing B=1 group, the executor calls `invoke_render` (the
    single-camera path), not `render_true_camera_batch`."""
    from eligsir.optimization.gaussian.contracts import RenderOutputs

    def _invoke(_gaussians, _camera, _pipe, _bg, **_kwargs):
        return RenderOutputs(payload={
            "render": torch.zeros((3, 2, 2)),
            "visibility_filter": torch.ones(population, dtype=torch.bool),
        })

    monkeypatch.setattr(executor, "invoke_render", _invoke)


def _patch_assemble_loss(monkeypatch, executor, *, skip_flags: list):
    def _loss(inputs):
        skip_flags.append(bool(inputs.skip_gaussian_regularizers))
        tensor = torch.tensor(1.0, requires_grad=True)
        return LossBreakdown(
            total=tensor.detach(), total_tensor=tensor,
            components={"photometric": tensor.detach()}, loss_finite=True,
        )

    patch_assemble_loss(monkeypatch, executor._objective, _loss)


def _runtime(engine, model, *, flag: bool, sparse_active_set: bool = True, energy_loss_enabled: bool = False):
    return OptimizationRuntimeContext(
        session=engine.session,
        gaussians=model,
        args=SimpleNamespace(
            streaming_objective_step_regularizers=flag,
            streaming_objective_batched_pixel_terms=False,
            energy_loss_enabled=energy_loss_enabled,
        ),
        opt=SimpleNamespace(iterations=100),
        pipe=SimpleNamespace(),
        sparse_active_set=sparse_active_set,
        gradient_policy_context=GradientPolicyContext(
            gaussians=model, args=SimpleNamespace(
                streaming_objective_step_regularizers=False,
                streaming_objective_batched_pixel_terms=False,
            ), iteration=1, n_frames_ingested=4,
        ),
        n_frames_ingested=4,
    )


def test_flag_on_b4_calls_step_regularizers_once_and_skips_every_view(monkeypatch):
    from eligsir.optimization.gaussian import true_batch_executor as executor

    cameras = tuple(_Camera(i) for i in range(4))
    model = _Model()
    engine = _Engine(model)
    _patch_render(monkeypatch, executor, population=3)
    skip_flags: list = []
    _patch_assemble_loss(monkeypatch, executor, skip_flags=skip_flags)

    calls = []

    def _step_reg(gaussians, args, optimizer_update_id, weights, *, batch_size):
        calls.append((gaussians is model, batch_size, tuple(weights.shape)))
        return torch.tensor(0.4, requires_grad=True)

    monkeypatch.setattr(executor._losses, "assemble_step_regularizers", _step_reg)

    runtime = _runtime(engine, model, flag=True)
    result = run_true_camera_batched_step(
        engine, tuple(_request(c) for c in cameras), runtime,
    )

    assert skip_flags == [True, True, True, True]
    assert len(calls) == 1
    assert calls[0] == (True, 4, (3,))
    assert result.loss_breakdown.components["active_reg+young"].item() == pytest.approx(0.4)
    # 4 views x total 1.0, scaled by batch_scale=1/4 => 1.0, plus reg_step 0.4.
    assert result.loss_breakdown.total.item() == pytest.approx(1.4)


def test_flag_off_never_calls_step_regularizers(monkeypatch):
    from eligsir.optimization.gaussian import true_batch_executor as executor

    cameras = tuple(_Camera(i) for i in range(4))
    model = _Model()
    engine = _Engine(model)
    _patch_render(monkeypatch, executor, population=3)
    skip_flags: list = []
    _patch_assemble_loss(monkeypatch, executor, skip_flags=skip_flags)

    called = []
    monkeypatch.setattr(
        executor._losses, "assemble_step_regularizers",
        lambda *a, **kw: called.append(1) or torch.tensor(0.0),
    )

    runtime = _runtime(engine, model, flag=False)
    result = run_true_camera_batched_step(
        engine, tuple(_request(c) for c in cameras), runtime,
    )

    assert skip_flags == [False, False, False, False]
    assert called == []
    assert "active_reg+young" not in result.loss_breakdown.components


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(sparse_active_set=False),
        dict(energy_loss_enabled=True),
    ],
)
def test_ineligible_configurations_fall_back(monkeypatch, kwargs):
    from eligsir.optimization.gaussian import true_batch_executor as executor

    cameras = tuple(_Camera(i) for i in range(4))
    model = _Model()
    engine = _Engine(model)
    _patch_render(monkeypatch, executor, population=3)
    skip_flags: list = []
    _patch_assemble_loss(monkeypatch, executor, skip_flags=skip_flags)

    called = []
    monkeypatch.setattr(
        executor._losses, "assemble_step_regularizers",
        lambda *a, **kw: called.append(1) or torch.tensor(0.0),
    )

    runtime = _runtime(engine, model, flag=True, **kwargs)
    run_true_camera_batched_step(engine, tuple(_request(c) for c in cameras), runtime)

    assert called == []
    assert skip_flags == [False, False, False, False]


def _proportional_step_reg(calls: list):
    """A step-regularizer stand-in proportional to ``weights.sum()`` (NOT a
    constant): with the ``_patch_render`` fixture's all-visible radii, a
    group of size ``k`` produces ``weights.sum() == k`` regardless of what
    ``batch_size`` the executor passes in -- so ``weights.sum() /
    batch_size`` changes value depending on whether the call site passes
    the group's own size (the pre-fix bug) or the call's true
    ``requested_b`` (the fix), unlike a constant mock which cannot tell the
    two apart."""

    def _step_reg(gaussians, args, optimizer_update_id, weights, *, batch_size):
        calls.append(batch_size)
        return weights.sum() / float(batch_size)

    return _step_reg


def test_trailing_b1_group_falls_back_while_leading_group_uses_step_regularizers(monkeypatch):
    from eligsir.optimization.gaussian import true_batch_executor as executor

    cameras = tuple(_Camera(i) for i in range(3))
    model = _Model()
    engine = _Engine(model)
    _patch_render(monkeypatch, executor, population=3)
    _patch_invoke_render(monkeypatch, executor, population=3)
    skip_flags: list = []
    _patch_assemble_loss(monkeypatch, executor, skip_flags=skip_flags)

    calls: list = []
    monkeypatch.setattr(executor._losses, "assemble_step_regularizers", _proportional_step_reg(calls))

    runtime = _runtime(engine, model, flag=True)
    result = run_true_camera_batched_step(
        engine, tuple(_request(c) for c in cameras), runtime,
        camera_microbatch_size=2,
    )

    # Groups are (0,1) and (2,) -- only the size-2 group is eligible.
    # `requested_b = len(requests) + skipped_view_count = 3 + 0 = 3`; the
    # executor MUST pass that (not the group's own size, 2) as `batch_size`
    # -- pre-fix this recorded [2], not [3].
    assert calls == [3]
    assert skip_flags == [True, True, False]
    # weights.sum() == 2 for the size-2 group (both views fully visible),
    # so the mock returns 2 / requested_b(=3) = 2/3.
    reg_term = 2.0 / 3.0
    # Multi-group recombination (len(group_aggregates) == 2 here) must not
    # silently drop the size-2 group's step-regularizer term from the FINAL
    # reported total/components (see `_combine_group_aggregates`): total =
    # [(1.0+1.0)/3 + reg_term] (group 0, B=2, step reg added) + [1.0/3]
    # (group 1, B=1, falls back, no step reg).
    assert result.loss_breakdown.components["active_reg+young"].item() == pytest.approx(reg_term)
    assert result.loss_breakdown.total.item() == pytest.approx(2.0 / 3.0 + reg_term + 1.0 / 3.0)


def test_multiple_eligible_groups_sum_their_step_terms_in_the_final_result(monkeypatch):
    """Two full B4-sized groups in one call (e.g. a true B8 request split by
    camera_microbatch_size) -- each group's step-regularizer contribution
    must both survive into the final combined result, not just the
    single-group fast path's `group_aggregates[0]`. Both groups must also
    be scaled by the SAME `requested_b`, not their own size -- pre-fix this
    over-weighted each group's term by `requested_b / len(group) = 2x`."""
    from eligsir.optimization.gaussian import true_batch_executor as executor

    cameras = tuple(_Camera(i) for i in range(4))
    model = _Model()
    engine = _Engine(model)
    _patch_render(monkeypatch, executor, population=3)
    skip_flags: list = []
    _patch_assemble_loss(monkeypatch, executor, skip_flags=skip_flags)

    calls: list = []
    monkeypatch.setattr(executor._losses, "assemble_step_regularizers", _proportional_step_reg(calls))

    runtime = _runtime(engine, model, flag=True)
    result = run_true_camera_batched_step(
        engine, tuple(_request(c) for c in cameras), runtime,
        camera_microbatch_size=2,
    )

    # Groups are (0,1) and (2,3), both eligible (B=2 > 1).
    # `requested_b = 4 + 0 = 4`; both groups must be scaled by it, not by
    # their own size (2) -- pre-fix this recorded [2, 2].
    assert calls == [4, 4]
    assert skip_flags == [True, True, True, True]
    # weights.sum() == 2 per group (both views fully visible per group), so
    # each mocked reg term is 2 / requested_b(=4) = 0.5.
    reg_term = 0.5
    assert result.loss_breakdown.components["active_reg+young"].item() == pytest.approx(2 * reg_term)
    # Each group: (1.0+1.0)/4 [global batch_scale = 1/4] + its own reg term
    # (0.5, not the pre-fix 1.0 = weights.sum()/len(group) = 2/2).
    assert result.loss_breakdown.total.item() == pytest.approx(0.5 + 0.5 + reg_term + reg_term)


def test_skipped_view_count_scales_the_step_term_by_requested_b(monkeypatch):
    """`skipped_view_count` folds extra views into `requested_b` without
    ever producing a group for them -- the single present group (all 3
    requests) must still be scaled by the inflated `requested_b=4`, not by
    its own size (3)."""
    from eligsir.optimization.gaussian import true_batch_executor as executor

    cameras = tuple(_Camera(i) for i in range(3))
    model = _Model()
    engine = _Engine(model)
    _patch_render(monkeypatch, executor, population=3)
    skip_flags: list = []
    _patch_assemble_loss(monkeypatch, executor, skip_flags=skip_flags)

    calls: list = []
    monkeypatch.setattr(executor._losses, "assemble_step_regularizers", _proportional_step_reg(calls))

    runtime = _runtime(engine, model, flag=True)
    result = run_true_camera_batched_step(
        engine, tuple(_request(c) for c in cameras), runtime,
        skipped_view_count=1,
    )

    # requested_b = len(requests) + skipped_view_count = 3 + 1 = 4; the
    # single group (all 3 requests, no microbatching) must be scaled by
    # that, not by its own size (3) -- pre-fix this recorded [3].
    assert calls == [4]
    assert skip_flags == [True, True, True]
    # weights.sum() == 3 (all 3 views fully visible), so the mocked reg
    # term is 3 / requested_b(=4) = 0.75.
    reg_term = 0.75
    assert result.loss_breakdown.components["active_reg+young"].item() == pytest.approx(reg_term)
    # 3 views x total 1.0, scaled by batch_scale = 1/requested_b = 1/4,
    # plus the reg term.
    assert result.loss_breakdown.total.item() == pytest.approx(0.75 + reg_term)


def test_non_finite_step_term_is_dropped_and_reported(monkeypatch):
    from eligsir.optimization.gaussian import true_batch_executor as executor

    cameras = tuple(_Camera(i) for i in range(4))
    model = _Model()
    engine = _Engine(model)
    _patch_render(monkeypatch, executor, population=3)
    skip_flags: list = []
    _patch_assemble_loss(monkeypatch, executor, skip_flags=skip_flags)

    monkeypatch.setattr(
        executor._losses, "assemble_step_regularizers",
        lambda *a, **kw: torch.tensor(float("nan"), requires_grad=True),
    )
    reported = []
    monkeypatch.setattr(executor, "_nan_term_report", lambda name, iteration: reported.append((name, iteration)))

    runtime = _runtime(engine, model, flag=True)
    result = run_true_camera_batched_step(
        engine, tuple(_request(c) for c in cameras), runtime,
    )

    assert reported == [("active_reg+young", 1)]
    assert "active_reg+young" not in result.loss_breakdown.components
    # Only the 4 photometric views remain: 4 x 1.0 x (1/4) = 1.0.
    assert result.loss_breakdown.total.item() == pytest.approx(1.0)


def test_non_finite_secondary_aborts_step_when_step_regularizers_are_shared(monkeypatch):
    from eligsir.optimization.gaussian import true_batch_executor as executor

    cameras = tuple(_Camera(i) for i in range(2))
    model = _Model()
    engine = _Engine(model)
    _patch_render(monkeypatch, executor, population=3)

    def _loss(inputs):
        total = torch.tensor(1.0, requires_grad=True)
        return LossBreakdown(
            total=total.detach(),
            total_tensor=total,
            components={},
            loss_finite=inputs.viewpoint_cam.uid == 0,
        )

    patch_assemble_loss(monkeypatch, executor._objective, _loss)
    monkeypatch.setattr(
        executor._losses,
        "assemble_step_regularizers",
        lambda *_args, **_kwargs: torch.tensor(0.1, requires_grad=True),
    )

    result = run_true_camera_batched_step(
        engine, tuple(_request(camera) for camera in cameras), _runtime(engine, model, flag=True)
    )

    assert result.step_allowed is False
    assert result.skip_reason == "non_finite_loss"
    assert engine._optimizer_step_calls == 0
