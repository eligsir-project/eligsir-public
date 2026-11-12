"""Executor wiring for ``--streaming_objective_batched_pixel_terms`` (issue
#1670 Phase B -- see ``.superpowers/sdd/2026-09-03-throughput-recovery-
tracks-1-2/batched-pixel-terms-brief.md``).

``run_true_camera_batched_step`` (``eligsir/optimization/gaussian/
true_batch_executor.py``) is the true-camera-batch (B4) executor. When the
flag is on AND a group is eligible (``len(group) > 1``, every view
``rgb_loss_enabled``, no free-space/appearance-only/tile-probe flag active,
TB scalar logging not due this iteration), it should build the group's
``renders``/``gts`` stacks once, call
``batched_pixel_terms.assemble_batched_pixel_terms`` exactly ONCE for the
whole group, and pass each view's own ``PixelTermsView`` into
``LossAssemblyInputs.precomputed_pixel_terms``. Flag off (default), a B=1
group, or an ineligible group must fall back to today's behavior exactly --
``assemble_batched_pixel_terms`` never called, ``precomputed_pixel_terms``
always ``None``.

Mirrors the fakes/pattern in ``tests/test_step_regularizers_executor_wiring.py``.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from _prepared_loss_fakes import patch_assemble_loss

from eligsir.mapping.renderer.true_camera_batch import TrueCameraBatchResult
from eligsir.optimization.gaussian.batched_pixel_terms import PixelTermsView
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
from eligsir.optimization.gaussian.supervision_evidence import SupervisionModality
from eligsir.optimization.gaussian.true_batch_executor import (
    run_true_camera_batched_step,
)


class _Camera:
    image_width = 2
    image_height = 2

    def __init__(self, uid: int, *, original_image=None):
        self.uid = uid
        self.original_image = (
            torch.rand(3, 2, 2) if original_image is None else original_image
        )


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


def _request(camera: _Camera, *, rgb_loss_enabled: bool = True, depth_loss_enabled: bool = False) -> OptimizationStepRequest:
    return OptimizationStepRequest(
        view=OptimizationView(
            camera=camera,
            source_index=camera.uid,
            background=None,
            update_sh_rest=True,
            render_depth=False,
            resolution_bucket="native",
            supervision_modality=SupervisionModality(
                rgb_loss_enabled=rgb_loss_enabled,
                depth_loss_enabled=depth_loss_enabled,
            ),
        ),
        iteration=1,
        map_version=0,
        topology_version=0,
        resolution_bucket="native",
        evidence_requirements=EvidenceRequirements(),
        random_state_token=RandomStateToken.capture(),
    )


def _patch_render(monkeypatch, executor, *, population: int, count: int | None = None):
    monkeypatch.setattr(executor, "supports_true_camera_batch", lambda _pipe, _cams: (True, "ok"))

    def _render(group, *_args, **_kwargs):
        n = count if count is not None else len(group)
        return TrueCameraBatchResult(
            renders=torch.rand((n, 3, 2, 2)),
            alphas=torch.ones((n, 1, 2, 2)),
            rendered_depth=None,
            rendered_depth_ed=None,
            meta={"radii": torch.ones((n, population))},
            camera_count=n,
            sparse_grad=True,
        )

    monkeypatch.setattr(executor, "render_true_camera_batch", _render)


def _patch_render_with_depth(monkeypatch, executor, *, population: int):
    """Like ``_patch_render`` but returns real (non-None) rendered_depth/
    rendered_depth_ed, so depth eligibility can actually be reached."""
    monkeypatch.setattr(executor, "supports_true_camera_batch", lambda _pipe, _cams: (True, "ok"))

    def _render(group, *_args, **_kwargs):
        n = len(group)
        return TrueCameraBatchResult(
            renders=torch.rand((n, 3, 2, 2)),
            alphas=torch.ones((n, 1, 2, 2)),
            rendered_depth=torch.rand((n, 1, 2, 2)).clamp_min(0.1),
            rendered_depth_ed=torch.rand((n, 1, 2, 2)).clamp_min(0.1),
            meta={"radii": torch.ones((n, population))},
            camera_count=n,
            sparse_grad=True,
        )

    monkeypatch.setattr(executor, "render_true_camera_batch", _render)


def _patch_invoke_render(monkeypatch, executor, *, population: int):
    from eligsir.optimization.gaussian.contracts import RenderOutputs

    def _invoke(_gaussians, _camera, _pipe, _bg, **_kwargs):
        return RenderOutputs(payload={
            "render": torch.zeros((3, 2, 2)),
            "visibility_filter": torch.ones(population, dtype=torch.bool),
        })

    monkeypatch.setattr(executor, "invoke_render", _invoke)


def _patch_assemble_loss(monkeypatch, executor, *, captured: list):
    def _loss(inputs):
        captured.append(inputs.precomputed_pixel_terms)
        tensor = torch.tensor(1.0, requires_grad=True)
        return LossBreakdown(
            total=tensor.detach(), total_tensor=tensor,
            components={"Ll1": tensor.detach()}, loss_finite=True,
        )

    patch_assemble_loss(monkeypatch, executor._objective, _loss)


def _fake_terms(b: int) -> tuple[PixelTermsView, ...]:
    return tuple(
        PixelTermsView(
            Ll1=torch.tensor(float(i)), ssim=torch.tensor(0.5),
            depth=None, depth_valid_count=None,
        )
        for i in range(b)
    )


def _patch_batched_pixel_terms(monkeypatch, executor, *, calls: list, terms_fn=_fake_terms):
    def _fake(**kwargs):
        calls.append(kwargs)
        return terms_fn(kwargs["renders"].shape[0])

    monkeypatch.setattr(executor, "assemble_batched_pixel_terms", _fake)


def _runtime(engine, model, *, flag: bool, tb_writer=None, loss_assembly_kwargs: dict | None = None, **args_kwargs):
    defaults = dict(
        streaming_objective_step_regularizers=False,
        streaming_objective_batched_pixel_terms=flag,
        streaming_free_space_loss_weight=0.0,
        streaming_appearance_only_enabled=False,
        supervision_tile_probe_grid=0,
        surface_confidence_depth_loss=False,
        streaming_depth_loss_type="l1",
        streaming_depth_loss_quantity="d",
        streaming_depth_loss_alpha_gate=0.0,
    )
    defaults.update(args_kwargs)
    return OptimizationRuntimeContext(
        session=engine.session,
        gaussians=model,
        args=SimpleNamespace(**defaults),
        opt=SimpleNamespace(iterations=100),
        pipe=SimpleNamespace(),
        tb_writer=tb_writer,
        loss_assembly_kwargs=loss_assembly_kwargs or {},
        gradient_policy_context=GradientPolicyContext(
            gaussians=model, args=SimpleNamespace(
                streaming_objective_step_regularizers=False,
                streaming_objective_batched_pixel_terms=False,
            ), iteration=1, n_frames_ingested=4,
        ),
        n_frames_ingested=4,
    )


def test_flag_on_calls_batched_pixel_terms_once_and_routes_each_view_its_own_terms(monkeypatch):
    from eligsir.optimization.gaussian import true_batch_executor as executor

    cameras = tuple(_Camera(i) for i in range(4))
    model = _Model()
    engine = _Engine(model)
    _patch_render(monkeypatch, executor, population=3)
    captured: list = []
    _patch_assemble_loss(monkeypatch, executor, captured=captured)
    calls: list = []
    _patch_batched_pixel_terms(monkeypatch, executor, calls=calls)

    runtime = _runtime(engine, model, flag=True)
    run_true_camera_batched_step(engine, tuple(_request(c) for c in cameras), runtime)

    assert len(calls) == 1
    assert len(captured) == 4
    expected_terms = _fake_terms(4)
    for got, want in zip(captured, expected_terms):
        assert got is not None
        torch.testing.assert_close(got.Ll1, want.Ll1)


def test_flag_off_never_calls_batched_pixel_terms_and_passes_none(monkeypatch):
    from eligsir.optimization.gaussian import true_batch_executor as executor

    cameras = tuple(_Camera(i) for i in range(4))
    model = _Model()
    engine = _Engine(model)
    _patch_render(monkeypatch, executor, population=3)
    captured: list = []
    _patch_assemble_loss(monkeypatch, executor, captured=captured)
    calls: list = []
    _patch_batched_pixel_terms(monkeypatch, executor, calls=calls)

    runtime = _runtime(engine, model, flag=False)
    run_true_camera_batched_step(engine, tuple(_request(c) for c in cameras), runtime)

    assert calls == []
    assert captured == [None, None, None, None]


def test_b1_group_falls_back_while_leading_group_batches(monkeypatch):
    from eligsir.optimization.gaussian import true_batch_executor as executor

    cameras = tuple(_Camera(i) for i in range(3))
    model = _Model()
    engine = _Engine(model)
    _patch_render(monkeypatch, executor, population=3)
    _patch_invoke_render(monkeypatch, executor, population=3)
    captured: list = []
    _patch_assemble_loss(monkeypatch, executor, captured=captured)
    calls: list = []
    _patch_batched_pixel_terms(monkeypatch, executor, calls=calls)

    runtime = _runtime(engine, model, flag=True)
    run_true_camera_batched_step(
        engine, tuple(_request(c) for c in cameras), runtime,
        camera_microbatch_size=2,
    )

    # Groups are (0,1) [size 2, eligible] and (2,) [size 1, always falls back].
    assert len(calls) == 1
    assert captured[0] is not None
    assert captured[1] is not None
    assert captured[2] is None


def test_mixed_rgb_loss_enabled_falls_back_the_whole_group(monkeypatch):
    from eligsir.optimization.gaussian import true_batch_executor as executor

    cameras = tuple(_Camera(i) for i in range(4))
    model = _Model()
    engine = _Engine(model)
    _patch_render(monkeypatch, executor, population=3)
    captured: list = []
    _patch_assemble_loss(monkeypatch, executor, captured=captured)
    calls: list = []
    _patch_batched_pixel_terms(monkeypatch, executor, calls=calls)

    requests = tuple(
        _request(c, rgb_loss_enabled=(i != 2)) for i, c in enumerate(cameras)
    )
    runtime = _runtime(engine, model, flag=True)
    run_true_camera_batched_step(engine, requests, runtime)

    assert calls == []
    assert captured == [None, None, None, None]


def test_mismatched_gt_image_shapes_falls_back(monkeypatch):
    from eligsir.optimization.gaussian import true_batch_executor as executor

    cameras = list(_Camera(i) for i in range(4))
    cameras[1] = _Camera(1, original_image=torch.rand(3, 2, 3))  # mismatched width
    model = _Model()
    engine = _Engine(model)
    _patch_render(monkeypatch, executor, population=3)
    captured: list = []
    _patch_assemble_loss(monkeypatch, executor, captured=captured)
    calls: list = []
    _patch_batched_pixel_terms(monkeypatch, executor, calls=calls)

    runtime = _runtime(engine, model, flag=True)
    run_true_camera_batched_step(engine, tuple(_request(c) for c in cameras), runtime)

    assert calls == []
    assert captured == [None, None, None, None]


def test_mixed_depth_modality_still_batches_rgb_with_depth_none(monkeypatch):
    from eligsir.optimization.gaussian import true_batch_executor as executor

    cameras = tuple(_Camera(i) for i in range(4))
    model = _Model()
    engine = _Engine(model)
    _patch_render(monkeypatch, executor, population=3)
    captured: list = []
    _patch_assemble_loss(monkeypatch, executor, captured=captured)
    calls: list = []
    _patch_batched_pixel_terms(monkeypatch, executor, calls=calls)

    requests = tuple(
        _request(c, depth_loss_enabled=(i < 2)) for i, c in enumerate(cameras)
    )
    runtime = _runtime(engine, model, flag=True)
    run_true_camera_batched_step(engine, requests, runtime)

    assert len(calls) == 1
    assert calls[0]["rendered_depth"] is None
    assert calls[0]["sensor_depth"] is None
    assert len(captured) == 4
    assert all(term is not None for term in captured)


def test_free_space_flag_falls_back_the_whole_group(monkeypatch):
    from eligsir.optimization.gaussian import true_batch_executor as executor

    cameras = tuple(_Camera(i) for i in range(4))
    model = _Model()
    engine = _Engine(model)
    _patch_render(monkeypatch, executor, population=3)
    captured: list = []
    _patch_assemble_loss(monkeypatch, executor, captured=captured)
    calls: list = []
    _patch_batched_pixel_terms(monkeypatch, executor, calls=calls)

    runtime = _runtime(engine, model, flag=True, streaming_free_space_loss_weight=0.1)
    run_true_camera_batched_step(engine, tuple(_request(c) for c in cameras), runtime)

    assert calls == []
    assert captured == [None, None, None, None]


def test_tb_scalar_logging_due_this_iteration_falls_back(monkeypatch):
    from eligsir.optimization.gaussian import true_batch_executor as executor

    cameras = tuple(_Camera(i) for i in range(4))
    model = _Model()
    engine = _Engine(model)
    _patch_render(monkeypatch, executor, population=3)
    captured: list = []
    _patch_assemble_loss(monkeypatch, executor, captured=captured)
    calls: list = []
    _patch_batched_pixel_terms(monkeypatch, executor, calls=calls)

    # iteration=1 in `_request`, scalar_log_interval=1 -> always "due".
    runtime = _runtime(
        engine, model, flag=True,
        tb_writer=SimpleNamespace(add_scalar=lambda *a, **kw: None),
        loss_assembly_kwargs={"scalar_log_interval": 1},
    )
    run_true_camera_batched_step(engine, tuple(_request(c) for c in cameras), runtime)

    assert calls == []
    assert captured == [None, None, None, None]


def test_depth_loss_type_is_read_from_loss_assembly_kwargs_not_args(monkeypatch):
    """Regression for a real bug: `_resolve_batched_pixel_terms` must read
    `depth_loss_type` from `runtime.loss_assembly_kwargs` -- the SAME dict
    `LossAssemblyInputs` is built from via `**runtime.loss_assembly_kwargs`
    -- not from `args.streaming_depth_loss_type` directly.
    `LossAssemblyInputs.depth_loss_type` defaults to `"l1"` while
    `args.streaming_depth_loss_type` defaults to `"huber"`
    (`arguments/__init__.py`); a resolver that read `args` would silently
    compute a DIFFERENT depth-error formula on the batched path than the
    per-view fallback for the exact same run whenever a caller left
    `loss_assembly_kwargs["depth_loss_type"]` unset. Deliberately sets
    `args.streaming_depth_loss_type` and `loss_assembly_kwargs
    ["depth_loss_type"]` to DIFFERENT values and asserts the
    `loss_assembly_kwargs` one is what reaches `assemble_batched_pixel_terms`."""
    from eligsir.optimization.gaussian import true_batch_executor as executor

    cameras = tuple(_Camera(i) for i in range(4))
    model = _Model()
    engine = _Engine(model)
    _patch_render_with_depth(monkeypatch, executor, population=3)
    monkeypatch.setattr(
        executor, "_get_sensor_depth",
        lambda cam, h, w: torch.rand(1, h, w).clamp_min(0.1),
    )
    captured: list = []
    _patch_assemble_loss(monkeypatch, executor, captured=captured)
    calls: list = []
    _patch_batched_pixel_terms(monkeypatch, executor, calls=calls)

    requests = tuple(_request(c, depth_loss_enabled=True) for c in cameras)
    runtime = _runtime(
        engine, model, flag=True,
        streaming_depth_loss_type="huber",  # args -- must NOT be read
        loss_assembly_kwargs={"depth_loss_type": "l1"},  # must be read
    )
    run_true_camera_batched_step(engine, requests, runtime)

    assert len(calls) == 1
    assert calls[0]["depth_loss_type"] == "l1"
    assert calls[0]["rendered_depth"] is not None
    assert calls[0]["sensor_depth"] is not None


@pytest.mark.parametrize(
    "bad_gt",
    [
        lambda: torch.empty(3, 2, 2, device="meta"),
        lambda: torch.empty(3, 2, 2, dtype=torch.float64),
    ],
    ids=["nonresident", "wrong_dtype"],
)
def test_nonresident_or_mismatched_gt_falls_back_before_stack_or_ssim(monkeypatch, bad_gt):
    from eligsir.optimization.gaussian import true_batch_executor as executor

    cameras = (
        _Camera(0, original_image=bad_gt()),
        *(_Camera(index) for index in range(1, 4)),
    )
    model = _Model()
    engine = _Engine(model)
    _patch_render(monkeypatch, executor, population=3)
    captured: list = []
    _patch_assemble_loss(monkeypatch, executor, captured=captured)
    calls: list = []
    _patch_batched_pixel_terms(monkeypatch, executor, calls=calls)
    monkeypatch.setattr(
        executor._objective,
        "_gt_image_validated",
        lambda *_: pytest.fail("nonresident GT reached batched validation"),
    )

    run_true_camera_batched_step(engine, tuple(_request(camera) for camera in cameras), _runtime(engine, model, flag=True))

    assert calls == []
    assert captured == [None, None, None, None]
