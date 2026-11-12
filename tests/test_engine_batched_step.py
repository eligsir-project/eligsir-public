"""Issue #106: ``GaussianOptimizationEngine.run_batched_step`` orchestration.

CPU-only. These tests deliberately fake the two heavy leaves the batched step
calls into (``invoke_render`` and ``objective.assemble_loss``) and exercise the
*orchestration* around them, which is where the batching-specific correctness
risk actually lives:

* ``batch_size=1`` must take the same branches the pre-#106 single-view
  ``run_step`` took -- in particular NO loss scaling at all.
* the optimizer must step against the UNION of the batch's visibility filters
  (the specific defect that would silently discard B-1 views' gradients on
  rows only an earlier view made visible, while still "working");
* ``post_render_hook`` must fire once per optimizer step, not once per view
  (it drives per-iteration cadences, including a 100-iteration sweep);
* the non-finite policy must abort on a bad PRIMARY view and skip-only on a
  bad SECONDARY view.

Real GPU gradient parity for the ``B=1`` path is covered end to end by the
engine-vs-inline GPU A/B recorded in
``docs/architecture/2026-08-12-issue106-batched-render-loss-implementation.md``.
"""

from __future__ import annotations

from argparse import Namespace
from dataclasses import replace

import pytest
import torch

from eligsir.optimization.gaussian import engine as engine_mod
from eligsir.optimization.gaussian.contracts import (
    EvidenceRequirements,
    GradientPolicyContext,
    OptimizationRuntimeContext,
    OptimizationStepRequest,
    OptimizationView,
    RandomStateToken,
    RenderOutputs,
)
from eligsir.contracts import ModalityEligibility
from eligsir.optimization.gaussian.supervision_evidence import SupervisionModality
from eligsir.optimization.gaussian.engine import GaussianOptimizationEngine


N = 6


class _FakeModel:
    def __init__(self):
        self.provisional = torch.zeros(N, dtype=torch.bool)
        self.last_supervised_iter = torch.zeros(N, dtype=torch.int32)
        self.post_render_mutations = 0

    def normalize_rotation_params(self, mask=None):
        pass


class _FakeSession:
    parameter_version = 0

    def __init__(self, model):
        self.model = model
        self.steps = []
        self.zero_grads = 0
        self.learning_rate_updates = []
        self.current_lr = 0.01

    def update_learning_rate(self, iteration):
        self.learning_rate_updates.append(iteration)
        self.current_lr = 0.001 / max(1, iteration)
        return self.current_lr

    def step(self, visibility=None):
        self.steps.append(visibility)

    def zero_grad(self):
        self.zero_grads += 1


class _FakeBackend:
    def __init__(self, session):
        self._session = session

    def begin_optimization(self):
        return self._session


class _FakeBreakdown:
    def __init__(self, tensor, finite=True, depth_valid_sample_count=0, depth=None):
        self.total_tensor = tensor
        self.total = tensor.detach()
        self.components = {"Ll1": tensor.detach()}
        if depth is not None:
            self.components["depth"] = torch.tensor(depth)
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
    """Wire an engine over fakes and record what each leaf was called with."""
    model = _FakeModel()
    session = _FakeSession(model)
    engine = GaussianOptimizationEngine(_FakeBackend(session))

    rec = {
        "renders": [],          # cameras rendered, in order
        "hook_calls": 0,        # post_render_hook invocations
        "backward_losses": [],  # loss VALUES handed to backward, in order
        "visibility": {},       # camera -> visibility filter to report
        "finite": {},           # camera -> loss_finite to report
        "loss_value": {},       # camera -> raw (unscaled) loss value
        "depth_valid_samples": {},  # camera -> exact objective depth mask count
        "depth_value": {},  # camera -> detached raw accepted depth data term
        "loss_optimizer_update_ids": [],
        "backward_iterations": [],
        "rgb_loss_enabled": [],
    }

    def _fake_invoke_render(gaussians, camera, pipe, bg, **kwargs):
        rec["renders"].append(camera)
        return RenderOutputs(payload={
            "render": torch.zeros(3, 2, 2),
            "visibility_filter": rec["visibility"].get(camera, _visibility(0)),
        })

    def _fake_assemble_loss(inputs):
        rec["loss_optimizer_update_ids"].append(inputs.optimizer_update_id)
        rec["rgb_loss_enabled"].append(inputs.rgb_loss_enabled)
        cam = inputs.viewpoint_cam
        value = rec["loss_value"].get(cam, 1.0)
        finite = rec["finite"].get(cam, True)
        tensor = torch.tensor(float("nan") if not finite else value, requires_grad=True)
        return _FakeBreakdown(
            tensor,
            finite=finite,
            depth_valid_sample_count=rec["depth_valid_samples"].get(cam, 0),
            depth=rec["depth_value"].get(cam),
        )

    def _fake_render_and_backward(gaussians, args, iteration, strategy, pkg, loss):
        rec["backward_iterations"].append(iteration)
        rec["backward_losses"].append(float(loss.detach()))

    monkeypatch.setattr(engine_mod, "invoke_render", _fake_invoke_render)
    monkeypatch.setattr(engine_mod._objective, "assemble_loss", _fake_assemble_loss)
    monkeypatch.setattr(engine, "render_and_backward", _fake_render_and_backward)
    # Gradient policies touch a lot of unrelated state; the batching questions
    # under test are all upstream of them.
    monkeypatch.setattr(engine, "run_gradient_policies", lambda ctx: [])

    def _runtime():
        def _hook(_outputs):
            rec["hook_calls"] += 1
            model.post_render_mutations += 1

        return OptimizationRuntimeContext(
            session=session, gaussians=model,
            args=Namespace(), opt=Namespace(iterations=10_000),
            pipe=Namespace(gsplat_sparse_grad=False),
            mcmc_strategy=None,
            gradient_policy_context=GradientPolicyContext(
                gaussians=model, args=Namespace(), iteration=5, n_frames_ingested=0,
            ),
            post_render_hook=_hook,
            loss_assembly_kwargs={},
        )

    return engine, session, rec, _runtime


def test_batch_size_one_applies_no_loss_scaling_and_reports_unit_clocks(harness):
    engine, _session, rec, runtime = harness
    rec["loss_value"]["cam"] = 0.75

    result = engine.run_batched_step([_request()], runtime())

    # Exactly the raw loss reached backward -- no 1/1 division, no multiply.
    assert rec["backward_losses"] == [0.75]
    assert result.batch_loss_scale == 1.0
    assert result.batch_size == 1
    assert result.supervision_view_count == 1
    assert result.skipped_view_count == 0


def test_run_step_delegates_to_the_batched_path(harness):
    engine, _session, rec, runtime = harness
    rec["loss_value"]["cam"] = 0.5

    result = engine.run_step(_request(), runtime())

    assert rec["backward_losses"] == [0.5]
    assert result.batch_size == 1


@pytest.mark.parametrize("request_count", [1, 2], ids=["b1_c1", "b_greater_than_1_c1"])
@pytest.mark.parametrize(
    "supervision_modality",
    [None, SupervisionModality(rgb_loss_enabled=True)],
    ids=["production_shape", "contradictory_explicit_intent"],
)
def test_engine_intersects_explicit_rgb_intent_with_production_eligibility(
    harness, request_count, supervision_modality,
):
    """Would catch a C=1 path widening RGB work beyond its admission contract."""
    engine, _session, rec, runtime = harness
    requests = [
        OptimizationStepRequest(
            view=OptimizationView(
                camera=f"cam-{index}",
                modality_eligibility=ModalityEligibility(rgb_supervision=False),
                supervision_modality=supervision_modality,
            ),
            iteration=5,
            map_version=1,
            topology_version=2,
            resolution_bucket="default",
            evidence_requirements=EvidenceRequirements(),
            random_state_token=RandomStateToken.capture(),
        )
        for index in range(request_count)
    ]

    if request_count == 1:
        engine.run_step(requests[0], runtime())
    else:
        engine.run_batched_step(requests, runtime())

    assert rec["rgb_loss_enabled"] == [False] * request_count


def test_each_view_backwards_its_loss_divided_by_the_batch_size(harness):
    engine, _session, rec, runtime = harness
    rec["loss_value"].update({"a": 1.0, "b": 2.0, "c": 3.0})

    result = engine.run_batched_step(
        [_request(camera="a"), _request(camera="b"), _request(camera="c")], runtime(),
    )

    assert rec["renders"] == ["a", "b", "c"]
    assert rec["backward_losses"] == pytest.approx([1 / 3, 2 / 3, 3 / 3])
    assert result.supervision_view_count == 3
    assert result.batch_loss_scale == pytest.approx(1 / 3)


def test_optimizer_steps_against_the_union_of_the_batch_visibility_filters(harness):
    """The defect this guards: stepping with only one view's mask silently
    drops the other views' real, already-accumulated gradients."""
    engine, session, rec, runtime = harness
    rec["visibility"].update({
        "a": _visibility(0, 1),
        "b": _visibility(2),
        "c": _visibility(4, 5),
    })
    ctx = runtime()
    ctx.sparse_active_set = True
    ctx.optimizer_type = "adam"

    engine.run_batched_step(
        [_request(camera="a"), _request(camera="b"), _request(camera="c")], ctx,
    )

    union = _visibility(0, 1, 2, 4, 5)
    assert torch.equal(
        ctx.gradient_policy_context.active_set.visible_mask, union,
    )


def test_post_render_hook_fires_once_per_optimizer_step_not_once_per_view(harness):
    """It drives per-ITERATION cadences (incl. a 100-iteration prune sweep);
    firing it per view would run them B x faster."""
    engine, _session, rec, runtime = harness

    engine.run_batched_step(
        [_request(camera="a"), _request(camera="b"), _request(camera="c")], runtime(),
    )

    assert rec["hook_calls"] == 1


def test_non_finite_primary_view_aborts_the_batch_before_any_extra_render(harness):
    engine, _session, rec, runtime = harness
    rec["finite"]["a"] = False

    result = engine.run_batched_step(
        [_request(camera="a"), _request(camera="b")], runtime(),
    )

    assert rec["renders"] == ["a"]          # "b" was never rendered
    assert rec["backward_losses"] == []
    assert result.step_allowed is False
    assert result.skip_reason == "non_finite_loss"
    assert result.supervision_view_count == 0


def test_failed_retry_reuses_semantic_update_id_without_supervision_age(harness):
    """Artifact attempts may advance; LR and recency must stay on update 7."""
    engine, session, rec, runtime = harness
    rec["finite"]["bad"] = False
    ctx = runtime()
    ctx.optimizer_update_id = 7
    ctx.optimizer_update_permitted = True
    ctx.gradient_policy_context.optimizer_update_id = 7
    before_lr = session.current_lr

    failed = engine.run_batched_step([_request(iteration=90, camera="bad")], ctx)
    assert session.current_lr == before_lr
    assert session.learning_rate_updates == []
    retried = engine.run_batched_step([_request(iteration=91, camera="good")], ctx)

    assert failed.optimizer_step_count == 0
    assert retried.optimizer_step_count == 1
    assert session.learning_rate_updates == [7]
    assert rec["loss_optimizer_update_ids"] == [7, 7]
    assert torch.equal(
        session.model.last_supervised_iter,
        torch.tensor([7, 0, 0, 0, 0, 0], dtype=torch.int32),
    )


def test_successful_batch_writes_union_visibility_at_optimizer_update_id(harness):
    engine, session, rec, runtime = harness
    ctx = runtime()
    ctx.optimizer_update_id = 7
    ctx.optimizer_update_permitted = True
    ctx.gradient_policy_context.optimizer_update_id = 7
    rec["visibility"].update({
        "a": _visibility(0, 1),
        "b": _visibility(3),
    })

    result = engine.run_batched_step(
        [_request(iteration=500, camera="a"), _request(iteration=500, camera="b")],
        ctx,
    )

    assert result.optimizer_step_count == 1
    assert session.learning_rate_updates == [7]
    assert rec["loss_optimizer_update_ids"] == [7, 7]
    assert torch.equal(
        session.model.last_supervised_iter,
        torch.tensor([7, 7, 0, 7, 0, 0], dtype=torch.int32),
    )


@pytest.mark.parametrize("camera_count", (1, 8))
def test_fixed_shape_mcmc_updates_visibility_ema_once_from_completed_union(
    harness, camera_count
):
    """C1 and accumulated C8 use one committed union, never one EMA per view."""
    engine, session, rec, runtime = harness

    class _Strategy:
        def inject_noise(self, **_kwargs):
            pass

    session.model.visibility_ema = torch.full((N, 1), 0.5)
    context = runtime()
    context.optimizer_update_id = 7
    context.optimizer_update_permitted = True
    context.gradient_policy_context.optimizer_update_id = 7
    context.fixed_shape_mcmc = True
    context.mcmc_strategy = _Strategy()
    cameras = tuple(f"camera-{index}" for index in range(camera_count))
    for index, camera in enumerate(cameras):
        rec["visibility"][camera] = _visibility(*(0, 2, 4) if index == 0 else ())

    result = engine.run_batched_step(
        [_request(iteration=500, camera=camera) for camera in cameras], context
    )

    assert result.optimizer_step_count == 1
    assert torch.equal(
        session.model.last_supervised_iter,
        torch.tensor([7, 0, 7, 0, 7, 0], dtype=torch.int32),
    )
    torch.testing.assert_close(
        session.model.visibility_ema,
        torch.tensor([[0.51], [0.49], [0.51], [0.49], [0.51], [0.49]]),
        rtol=0,
        atol=1e-7,
    )


def test_fixed_shape_mcmc_excludes_nonfinite_secondary_visibility_from_ema(harness):
    engine, session, rec, runtime = harness

    class _Strategy:
        def inject_noise(self, **_kwargs):
            pass

    session.model.visibility_ema = torch.full((N, 1), 0.5)
    context = runtime()
    context.optimizer_update_id = 7
    context.optimizer_update_permitted = True
    context.gradient_policy_context.optimizer_update_id = 7
    context.fixed_shape_mcmc = True
    context.mcmc_strategy = _Strategy()
    rec["visibility"].update({"good": _visibility(0), "bad": _visibility(1)})
    rec["finite"]["bad"] = False

    result = engine.run_batched_step(
        [_request(iteration=500, camera="good"), _request(iteration=500, camera="bad")],
        context,
    )

    assert result.optimizer_step_count == 1
    assert result.skipped_view_count == 1
    assert torch.equal(
        session.model.last_supervised_iter,
        torch.tensor([7, 0, 0, 0, 0, 0], dtype=torch.int32),
    )
    torch.testing.assert_close(
        session.model.visibility_ema,
        torch.tensor([[0.51], [0.49], [0.49], [0.49], [0.49], [0.49]]),
        rtol=0,
        atol=1e-7,
    )


def test_legacy_or_live_population_runtime_leaves_visibility_ema_unchanged(harness):
    engine, session, _rec, runtime = harness
    session.model.visibility_ema = torch.full((N, 1), 0.5)

    legacy = engine.run_batched_step([_request(iteration=500)], runtime())
    assert legacy.optimizer_step_count == 1
    torch.testing.assert_close(session.model.visibility_ema, torch.full((N, 1), 0.5))

    context = runtime()
    context.optimizer_update_id = 7
    context.optimizer_update_permitted = True
    context.gradient_policy_context.optimizer_update_id = 7
    live_population = engine.run_batched_step([_request(iteration=501)], context)
    assert live_population.optimizer_step_count == 1
    torch.testing.assert_close(session.model.visibility_ema, torch.full((N, 1), 0.5))


def test_exact_work_plan_permission_replaces_iteration_ceiling_and_fails_closed(harness):
    engine, session, _rec, runtime = harness
    denied = runtime()
    denied.optimizer_update_id = 1
    denied.optimizer_update_permitted = False
    denied.gradient_policy_context.optimizer_update_id = 1
    denied.opt.iterations = 1
    denied_result = engine.run_batched_step([_request(iteration=50)], denied)

    assert session.model.post_render_mutations == 0
    denied.optimizer_update_permitted = True
    permitted_result = engine.run_batched_step([_request(iteration=51)], denied)

    assert denied_result.optimizer_step_count == 0
    assert denied_result.skip_reason == "work_plan_endpoint"
    assert denied_result.backward_ran is False
    assert permitted_result.optimizer_step_count == 1
    assert session.model.post_render_mutations == 1
    assert _rec["backward_losses"] == [1.0]


def test_canonical_mcmc_pre_backward_uses_semantic_update_id(harness):
    engine, _session, rec, runtime = harness
    context = runtime()
    context.optimizer_update_id = 7
    context.optimizer_update_permitted = True
    context.gradient_policy_context.optimizer_update_id = 7

    result = engine.run_batched_step([_request(iteration=10_000)], context)

    assert result.optimizer_step_count == 1
    assert rec["backward_iterations"] == [7]


def test_synthesized_gradient_context_uses_runtime_optimizer_update_id(
    harness, monkeypatch
):
    from contextlib import contextmanager
    import eligsir.utils.seed_lineage_freeze as seed_freeze

    engine, _session, _rec, runtime = harness
    guarded_iterations = []

    @contextmanager
    def _guard(*_args, iteration, **_kwargs):
        guarded_iterations.append(iteration)
        yield

    monkeypatch.setattr(seed_freeze, "seed_lineage_optimizer_guard", _guard)
    context = runtime()
    context.gradient_policy_context = None
    context.optimizer_update_id = 7
    context.optimizer_update_permitted = True

    engine.run_batched_step([_request(iteration=10_000)], context)

    assert guarded_iterations == [7]


def test_supplied_gradient_context_must_match_runtime_optimizer_update_id(harness):
    engine, _session, _rec, runtime = harness
    context = runtime()
    context.optimizer_update_id = 7
    context.optimizer_update_permitted = True
    context.gradient_policy_context.optimizer_update_id = 8

    with pytest.raises(ValueError, match="optimizer_update_id must match"):
        engine.run_batched_step([_request(iteration=10_000)], context)


def test_engine_rejects_hybrid_canonical_and_legacy_optimizer_clock(harness):
    engine, _session, _rec, runtime = harness
    context = runtime()
    context.optimizer_update_id = 7
    context.gradient_policy_context.optimizer_update_id = 7

    with pytest.raises(ValueError, match="must either both be None or both be set"):
        engine.run_batched_step([_request(iteration=10_000)], context)


@pytest.mark.parametrize(
    "clock_fields",
    [
        {"optimizer_update_id": 7},
        {"optimizer_update_permitted": True},
    ],
)
def test_runtime_context_rejects_hybrid_optimizer_clock_at_construction(
    clock_fields,
):
    with pytest.raises(ValueError, match="must either both be None or both be set"):
        OptimizationRuntimeContext(**clock_fields)


def test_canonical_seed_freeze_and_noise_use_semantic_update_id(harness, monkeypatch):
    from contextlib import contextmanager
    import eligsir.utils.seed_lineage_freeze as seed_freeze

    engine, _session, _rec, runtime = harness
    calls = {"seed": [], "noise": []}

    @contextmanager
    def _guard(*_args, iteration, **_kwargs):
        calls["seed"].append(iteration)
        yield

    class _Strategy:
        def inject_noise(self, **kwargs):
            calls["noise"].append(kwargs["iteration"])

    monkeypatch.setattr(seed_freeze, "seed_lineage_optimizer_guard", _guard)
    context = runtime()
    context.optimizer_update_id = 7
    context.gradient_policy_context.optimizer_update_id = 7
    context.optimizer_update_permitted = True
    context.fixed_shape_mcmc = True
    context.mcmc_strategy = _Strategy()

    result = engine.run_batched_step([_request(iteration=10_000)], context)

    assert result.optimizer_step_count == 1
    assert calls == {"seed": [7], "noise": [7]}


def test_non_finite_secondary_view_is_skipped_and_the_batch_continues(harness):
    engine, _session, rec, runtime = harness
    rec["loss_value"].update({"a": 1.0, "c": 3.0})
    rec["finite"]["b"] = False

    result = engine.run_batched_step(
        [_request(camera="a"), _request(camera="b"), _request(camera="c")], runtime(),
    )

    # The divisor is NOT renormalized to 1/2: "a" already backwarded at 1/3.
    assert rec["backward_losses"] == pytest.approx([1 / 3, 3 / 3])
    assert result.supervision_view_count == 2
    assert result.skipped_view_count == 1


def test_batch_requires_one_fixed_topology_for_every_view(harness):
    engine, _session, _rec, runtime = harness
    mismatched = replace(_request(camera="b"), topology_version=99)

    with pytest.raises(ValueError, match="fixed topology"):
        engine.run_batched_step([_request(camera="a"), mismatched], runtime())


def test_empty_request_sequence_is_rejected(harness):
    engine, _session, _rec, runtime = harness
    with pytest.raises(ValueError):
        engine.run_batched_step([], runtime())


# ---------------------------------------------------------------------------
# Review of PR #528: views the CALLER dropped before dispatch.
#
# The defect: issue #333's tier-homogeneity skip drops an extra camera at
# COLLECTION time, before dispatch, so `batch_size = len(requests)` shrank with
# it. At B=2 with the single extra view unusable, `extra_cameras` is empty, the
# dispatch gate falls through to `run_step`, and the loss is divided by nothing
# at all -- a 2x effective-LR step on exactly the iterations where a bind
# failed, and the opposite of what the PR body claimed ("leaves
# _opt_batch_size as the divisor"). These pin the divisor numerically.
# ---------------------------------------------------------------------------


def test_caller_declared_skips_keep_the_requested_batch_size_as_the_divisor(harness):
    """B=2 with the one extra view dropped by the caller still divides by 2."""
    engine, _session, rec, runtime = harness
    rec["loss_value"]["cam"] = 1.0

    result = engine.run_batched_step([_request()], runtime(), skipped_view_count=1)

    assert rec["backward_losses"] == [0.5]
    assert result.batch_loss_scale == 0.5
    assert result.batch_size == 2
    # batch_size == supervised + skipped, whoever did the skipping.
    assert result.supervision_view_count == 1
    assert result.skipped_view_count == 1


def test_caller_skips_and_non_finite_skips_share_one_divisor_rule(harness):
    """B=3: one view dropped by the caller, one skipped for a non-finite loss.
    Both are 1/3 of the divisor and neither renormalizes the other."""
    engine, _session, rec, runtime = harness
    rec["loss_value"].update({"a": 1.0, "b": 2.0})
    rec["finite"]["b"] = False

    result = engine.run_batched_step(
        [_request(camera="a"), _request(camera="b")], runtime(), skipped_view_count=1,
    )

    assert rec["backward_losses"] == pytest.approx([1 / 3])
    assert result.batch_loss_scale == pytest.approx(1 / 3)
    assert result.batch_size == 3
    assert result.supervision_view_count == 1
    assert result.skipped_view_count == 2


def test_a_nominally_batched_step_with_no_extra_request_leaves_visible_mask_alone(harness):
    """`skipped_view_count` must change the divisor and nothing else: with no
    second view there is no visibility filter to union, so the single-view
    path's mask handling has to stay exactly as it was."""
    engine, _session, rec, runtime = harness
    rec["visibility"]["cam"] = _visibility(1, 3)
    ctx = runtime()
    sentinel = _visibility(5)
    ctx.gradient_policy_context.active_set.visible_mask = sentinel

    engine.run_batched_step([_request()], ctx, skipped_view_count=1)

    assert ctx.gradient_policy_context.active_set.visible_mask is sentinel


def test_non_finite_primary_with_a_declared_skip_still_backwards_nothing(harness):
    """The `batch_size == supervised + skipped` identity is scoped to steps
    whose PRIMARY view backwarded. A non-finite primary aborts before any
    view is supervised, so it reports 0 supervised against a nonzero
    batch_size -- pre-existing behavior a declared caller-skip must not
    disturb (in particular it must not fake a supervision view)."""
    engine, _session, rec, runtime = harness
    rec["finite"]["cam"] = False

    result = engine.run_batched_step([_request()], runtime(), skipped_view_count=1)

    assert rec["backward_losses"] == []
    assert result.step_allowed is False
    assert result.skip_reason == "non_finite_loss"
    assert result.batch_size == 2
    assert result.supervision_view_count == 0
    assert result.skipped_view_count == 1


def test_negative_skipped_view_count_is_rejected(harness):
    engine, _session, _rec, runtime = harness
    with pytest.raises(ValueError, match="skipped_view_count"):
        engine.run_batched_step([_request()], runtime(), skipped_view_count=-1)


def test_dispatch_routes_an_all_skipped_batch_through_the_batched_path(harness):
    """THE discriminating case, end to end through the real dispatch helper.

    At B=2 with the single extra view tier-skipped, ``extra_cameras`` is EMPTY.
    A dispatch gate that tests only ``extra_cameras`` sends this to
    ``run_step`` (divisor 1) no matter what the engine does with a declared
    skip count, so the gate is part of the fix and is pinned here.
    """
    from eligsir.mapping.optimization_runtime import run_streaming_optimization_step

    engine, _session, rec, runtime = harness
    rec["loss_value"]["cam"] = 1.0
    ctx = runtime()

    class _FakeRuntime:
        random_state_token = None

        def build_step(self, **kwargs):
            return _request(camera=kwargs.get("camera", "cam")), ctx

        def commit_step(self, _context, _result=None):
            self.committed = True

    result = run_streaming_optimization_step(
        _FakeRuntime(), engine,
        extra_cameras=[],            # every extra view was dropped by the caller
        skipped_extra_view_count=1,
        camera="cam",
    )

    assert rec["backward_losses"] == [0.5]
    assert result.batch_size == 2
    assert result.batch_loss_scale == 0.5


def test_b1_dispatch_reuses_its_normal_render_for_e0(harness):
    """B1 E0 must be attached to the completed normal renderer result."""

    from eligsir.contracts.runtime import SourceIdentity
    from eligsir.mapping.optimization_runtime import run_streaming_optimization_step
    from eligsir.optimization.gaussian.supervision_evidence import SupervisionModality

    engine, _session, rec, runtime = harness
    rec["depth_value"]["cam"] = 2.5
    rec["depth_valid_samples"]["cam"] = 7
    context = runtime()
    context.evidence_transaction_id = 11
    context.coordinate_frame_version = context.schema_version = 0
    context.surface_evidence_version = context.policy_version = 0
    context.optimizer_update_id = 7
    context.optimizer_update_permitted = True
    context.gradient_policy_context.optimizer_update_id = 7
    source = SourceIdentity("test", 0, 7, 7)
    request = _request(camera="cam", source_index=7, render_depth=True)
    request = replace(
        request,
        evidence_requirements=EvidenceRequirements(want_supervision_evidence=True),
        view=replace(
            request.view,
            source_identity=source,
            supervision_modality=SupervisionModality(
                depth_loss_enabled=True, depth_evidence_enabled=True
            ),
        ),
    )

    class _Runtime:
        random_state_token = None

        def build_step(self, **_kwargs):
            return request, context

        def commit_step(self, _context, _result=None):
            pass

    result = run_streaming_optimization_step(_Runtime(), engine, camera="cam")

    assert rec["renders"] == ["cam"]
    assert result.rasterizer_call_count == 1
    assert result.evidence.source_identities == (source,)
    assert result.evidence.modalities == (
        SupervisionModality(depth_loss_enabled=True, depth_evidence_enabled=True),
    )
    assert torch.equal(result.evidence.depth_residual_summary, torch.tensor([2.5]))
    assert torch.equal(result.evidence.depth_valid_pixels, torch.tensor([7]))


def test_dispatch_falls_back_to_sequential_for_dense_true_camera_batch(harness, monkeypatch):
    """Dense C>1 parity is unqualified, so it must not reach one-call gsplat."""
    from eligsir.mapping.optimization_runtime import run_streaming_optimization_step
    from eligsir.optimization.gaussian import true_batch_executor

    engine, _session, rec, runtime = harness
    ctx = runtime()
    ctx.pipe = Namespace(sh_impl="gsplat", gsplat_sparse_grad=False)

    class _Camera:
        image_width = 2
        image_height = 2

    primary, secondary = _Camera(), _Camera()
    rec["loss_value"].update({primary: 1.0, secondary: 1.0})

    class _FakeRuntime:
        pipe = ctx.pipe
        random_state_token = None

        def build_step(self, **kwargs):
            return _request(camera=kwargs["camera"]), ctx

        def commit_step(self, _context, _result=None):
            self.committed = True

    def _unqualified_one_call(*_args, **_kwargs):
        pytest.fail("dense C>1 must fall back before the one-call executor")

    monkeypatch.setattr(
        true_batch_executor,
        "run_true_camera_batched_step",
        _unqualified_one_call,
    )

    result = run_streaming_optimization_step(
        _FakeRuntime(),
        engine,
        extra_cameras=[secondary],
        camera=primary,
    )

    assert rec["renders"] == [primary, secondary]
    assert result.true_camera_batch is False
    assert result.true_camera_batch_fallback_reason == "planned_sequential_accumulation"


# ---------------------------------------------------------------------------
# Issue #703: rasterizer call count and supervised pixel accounting.
# ---------------------------------------------------------------------------


def test_batch_size_one_rasterizer_calls_and_pixels(harness):
    """B=1: 1 rasterizer call, pixels from one 2x2 render."""
    engine, _session, rec, runtime = harness
    rec["loss_value"]["cam"] = 0.75

    result = engine.run_batched_step([_request(source_index=101)], runtime())

    # One render of a 2x2 image = 4 pixels (the fake render is 3x2x2)
    assert result.rasterizer_call_count == 1
    assert result.supervised_pixel_count == 4
    assert result.successful_supervision_source_indices == (101,)


def test_batch_size_four_sequential_rasterizer_calls_and_pixels(harness):
    """B=4 sequential: 4 rasterizer calls, 4*4 supervised pixels."""
    engine, _session, rec, runtime = harness
    rec["loss_value"].update({"a": 1.0, "b": 2.0, "c": 3.0, "d": 4.0})

    result = engine.run_batched_step(
        [
            _request(camera="a", source_index=10),
            _request(camera="b", source_index=20),
            _request(camera="c", source_index=30),
            _request(camera="d", source_index=40),
        ],
        runtime(),
    )

    # 4 renders, each 2x2 = 4 pixels per render, 16 total
    assert result.rasterizer_call_count == 4
    assert result.supervised_pixel_count == 16
    assert result.supervision_view_count == 4
    assert result.successful_supervision_source_indices == (10, 20, 30, 40)


def test_successful_exposure_indices_preserve_duplicate_camera_draws(harness):
    """Sampling is with replacement, so each successful backward is exposure."""
    engine, _session, rec, runtime = harness
    rec["loss_value"].update({"a": 1.0, "b": 2.0, "c": 3.0})

    result = engine.run_batched_step(
        [
            _request(camera="a", source_index=10),
            _request(camera="b", source_index=20),
            _request(camera="a", source_index=10),
            _request(camera="c", source_index=30),
        ],
        runtime(),
    )

    assert result.successful_supervision_source_indices == (10, 20, 10, 30)


def test_declared_tier_skip_does_not_fabricate_an_extra_exposure(harness):
    """A caller-dropped tier-mismatched view has no request and no exposure."""
    engine, _session, rec, runtime = harness
    rec["loss_value"]["primary"] = 1.0

    result = engine.run_batched_step(
        [_request(camera="primary", source_index=10)],
        runtime(),
        skipped_view_count=1,
    )

    assert result.skipped_view_count == 1
    assert result.successful_supervision_source_indices == (10,)


def test_rasterizer_count_skips_non_finite_primary(harness):
    """Non-finite primary: 1 rasterizer call, 0 supervised pixels, 0 views."""
    engine, _session, rec, runtime = harness
    rec["finite"]["a"] = False

    result = engine.run_batched_step([_request(camera="a")], runtime())

    # Primary render happens, but backward doesn't, so no supervised pixels
    assert result.rasterizer_call_count == 1
    assert result.supervised_pixel_count == 0
    assert result.supervision_view_count == 0


def test_rasterizer_count_skips_non_finite_secondary_but_continues(harness):
    """Non-finite secondary: renders all, counts only successful supervisions."""
    engine, _session, rec, runtime = harness
    rec["loss_value"].update({"a": 1.0, "b": 2.0, "c": 3.0})
    rec["finite"]["b"] = False  # middle view is non-finite

    result = engine.run_batched_step(
        [
            _request(camera="a", source_index=10),
            _request(camera="b", source_index=20),
            _request(camera="c", source_index=30),
        ],
        runtime(),
    )

    # All 3 renders happen, but only 2 backward (a and c)
    assert result.rasterizer_call_count == 3
    assert result.supervised_pixel_count == 8  # 2 successful renders * 4 pixels each
    assert result.supervision_view_count == 2
    assert result.skipped_view_count == 1
    assert result.successful_supervision_source_indices == (10, 30)


def test_depth_samples_sum_only_depth_enabled_views_in_the_accepted_update(harness):
    engine, _session, rec, runtime = harness
    rec["depth_valid_samples"].update({"a": 3, "b": 101, "c": 5, "d": 97})
    rec["finite"]["b"] = False

    result = engine.run_batched_step(
        [
            _request(camera="a", render_depth=True),
            _request(camera="b", render_depth=True),
            _request(camera="c", render_depth=True),
            _request(camera="d", render_depth=False),
        ],
        runtime(),
    )

    assert result.optimizer_step_count == 1
    assert result.depth_valid_sample_count == 8


def test_depth_samples_are_zero_when_backward_does_not_reach_an_update(harness):
    engine, _session, rec, runtime = harness
    rec["depth_valid_samples"]["cam"] = 13
    ctx = runtime()
    ctx.opt.iterations = 5

    result = engine.run_batched_step(
        [_request(iteration=5, render_depth=True)], ctx
    )

    assert result.backward_ran is True
    assert result.optimizer_step_count == 0
    assert result.depth_valid_sample_count == 0


def test_different_render_resolutions_accumulate_correctly(harness):
    """Pixel count works if different views have different resolutions."""
    engine, _session, _rec, runtime = harness

    # Monkey patch invoke_render to return different-sized images per camera
    render_sizes = {"a": (3, 2, 2), "b": (3, 3, 4), "c": (3, 5, 5)}

    def _fake_invoke_render_variable_size(gaussians, camera, pipe, bg, **kwargs):
        size = render_sizes.get(camera, (3, 2, 2))
        return RenderOutputs(payload={
            "render": torch.zeros(*size),
            "visibility_filter": torch.zeros(N, dtype=torch.bool),
        })

    import eligsir.optimization.gaussian.engine as engine_mod
    original_invoke = engine_mod.invoke_render
    engine_mod.invoke_render = _fake_invoke_render_variable_size

    try:
        result = engine.run_batched_step(
            [
                _request(camera="a"),
                _request(camera="b"),
                _request(camera="c"),
            ],
            runtime(),
        )

        # a: 2x2=4, b: 3x4=12, c: 5x5=25 = 41 total
        assert result.rasterizer_call_count == 3
        assert result.supervised_pixel_count == 41
    finally:
        engine_mod.invoke_render = original_invoke


def test_placement_only_step_has_no_supervised_pixels(harness):
    """Placement-only step: render happens but no backward, no supervised pixels."""
    engine, _session, rec, runtime = harness
    rec["loss_value"]["cam"] = 1.0
    ctx = runtime()
    ctx.placement_only = True

    result = engine.run_batched_step([_request()], ctx)

    assert result.rasterizer_call_count == 1
    assert result.supervised_pixel_count == 0
    assert result.supervision_view_count == 0
    assert result.step_allowed is False


def test_iteration_limit_has_rasterizer_call_but_no_supervised_pixels(harness):
    """A backward without an accepted update contributes no canonical exposure."""
    engine, _session, rec, runtime = harness
    rec["loss_value"]["cam"] = 1.0
    ctx = runtime()
    ctx.opt.iterations = 5  # Current iteration is 5 (from _request default)

    result = engine.run_batched_step([_request(iteration=5)], ctx)

    # Terminal semantics: backward ran, but optimizer step did not
    # (per the engine docstring "backward_ran=True with step_allowed=False")
    # The render/backward remain diagnostic work, but canonical supervision
    # exposure is committed only together with the accepted optimizer update.
    assert result.rasterizer_call_count == 1
    assert result.supervised_pixel_count == 0
    assert result.supervision_view_count == 0
    assert result.backward_ran is True
    assert result.step_allowed is False


def test_accepted_update_rejects_a_positive_view_count_without_pixels(
    harness, monkeypatch
):
    engine, _session, _rec, runtime = harness

    def _zero_area_render(gaussians, camera, pipe, bg, **kwargs):
        return RenderOutputs(
            payload={
                "render": torch.zeros(3, 2, 0),
                "visibility_filter": torch.zeros(N, dtype=torch.bool),
            }
        )

    monkeypatch.setattr(engine_mod, "invoke_render", _zero_area_render)

    with pytest.raises(ValueError, match="positive supervised_pixel_count"):
        engine.run_batched_step([_request()], runtime())


# Issue #809: optimizer_step_count real behavioral tests
# (not AST-only structural checks). These verify the engine reports the
# correct NUMBER of optimizer steps, not just that the code exists.

def test_issue809_single_view_step_reports_one_optimizer_step(harness):
    """B=1: one optimizer step (the critical baseline)."""
    engine, _session, rec, runtime = harness
    rec["loss_value"]["cam"] = 0.75

    result = engine.run_batched_step([_request()], runtime())

    # One render of a 2x2 image = 4 pixels; one optimizer step
    assert result.optimizer_step_count == 1, (
        "B=1 step must report exactly one optimizer step"
    )
    assert len(_session.steps) == 1, "Session step() must be called exactly once"


def test_issue809_batched_four_views_one_optimizer_step(harness):
    """B=4 sequential: ONE optimizer step (not 4).

    This is the critical assertion that catches per-view bugs: if the counter
    were incremented per view instead of per step, it would report 4 here.
    """
    engine, _session, rec, runtime = harness
    rec["loss_value"].update({"a": 1.0, "b": 2.0, "c": 3.0, "d": 4.0})

    result = engine.run_batched_step(
        [
            _request(camera="a"),
            _request(camera="b"),
            _request(camera="c"),
            _request(camera="d"),
        ],
        runtime(),
    )

    # 4 renders, but ONE optimizer step against the union of all visibility
    assert result.optimizer_step_count == 1, (
        "B=4 batch must report exactly one optimizer step, not 4. "
        "Per-view increment would incorrectly report 4."
    )
    assert result.rasterizer_call_count == 4
    assert result.supervision_view_count == 4
    assert len(_session.steps) == 1, "Session step() called once for the batch"


def test_issue809_non_finite_primary_zero_optimizer_steps(harness):
    """Non-finite primary: no backward, no optimizer step."""
    engine, _session, rec, runtime = harness
    rec["finite"]["a"] = False  # Primary view will not backward

    result = engine.run_batched_step([_request(camera="a")], runtime())

    # Primary render happens, but backward and optimizer step do not
    assert result.optimizer_step_count == 0, (
        "Non-finite primary must skip optimizer step"
    )
    assert result.rasterizer_call_count == 1
    assert result.supervision_view_count == 0
    assert len(_session.steps) == 0, "Session step() must not be called"


def test_issue809_iteration_limit_terminal_zero_optimizer_steps(harness):
    """Iteration limit (terminal iteration): backward runs, but NO optimizer step."""
    engine, _session, rec, runtime = harness
    rec["loss_value"]["cam"] = 1.0
    ctx = runtime()
    ctx.opt.iterations = 5  # Current iteration is 5 (from _request default) = limit

    result = engine.run_batched_step([_request(iteration=5)], ctx)

    # Terminal semantics: backward ran, optimizer step did not
    assert result.optimizer_step_count == 0, (
        "Terminal iteration (iteration >= opt.iterations) must not run optimizer step"
    )
    assert result.backward_ran is True, "Backward must still run at terminal"
    assert result.step_allowed is False, "step_allowed gate reflects no step"
    assert len(_session.steps) == 0, "Session step() must not be called"
