"""Unused RGB math is absent, not a measured zero, on canonical true-C work."""

from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")

from eligsir.optimization.gaussian import batched_pixel_terms, objective
from eligsir.optimization.gaussian.supervision_evidence import SupervisionModality
from test_group_loss_status import _case, _request, executor
from test_objective_precomputed_pixel_terms_integration import _fixture, _inputs
from test_true_batch_supervision_evidence import _request as _evidence_request


def _canonical_case(monkeypatch, *, batched=False, size=32):
    engine, runtime, cameras, renders = _case(monkeypatch, batch_size=size)
    runtime.optimizer_update_id = runtime.gradient_policy_context.optimizer_update_id = 1
    runtime.optimizer_update_permitted = True
    runtime.args.streaming_objective_batched_pixel_terms = batched
    return engine, runtime, cameras, renders


def _forbid(*_args, **_kwargs):
    raise AssertionError("unrequested RGB term was evaluated")


@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("camera_count", [32, 8])
@pytest.mark.parametrize("rgb_enabled", [False, True])
def test_canonical_rgb_demand_skips_dead_math_and_preserves_gradient(
    monkeypatch, batched, camera_count, rgb_enabled,
):
    engine, runtime, cameras, renders = _canonical_case(monkeypatch, batched=batched)
    runtime.opt.lambda_dssim = 0.0
    model = runtime.gaussians
    reference = []
    for camera, image in zip(cameras, renders):
        inputs = _inputs(runtime.args, runtime.opt, runtime.pipe, camera, model, image)
        reference.append(objective.assemble_loss(replace(inputs, rgb_loss_enabled=rgb_enabled)))
    expected = executor._aggregate_loss(reference, 1 / 32)
    expected.total_tensor.backward()
    expected_gradients = [p.grad.clone() for p in (renders, model.get_scaling, model.get_opacity)]
    model.optimizer.zero_grad(set_to_none=True)

    monkeypatch.setattr(objective, "ssim", _forbid)
    monkeypatch.setattr(batched_pixel_terms, "ssim_per_view", _forbid)
    if not rgb_enabled:
        monkeypatch.setattr(objective, "l1_loss", _forbid)
    readbacks = []
    original_tolist = torch.Tensor.tolist
    def tolist(tensor):
        readbacks.append(tensor.shape)
        return original_tolist(tensor)
    monkeypatch.setattr(torch.Tensor, "tolist", tolist)
    monkeypatch.setattr(torch.Tensor, "item", _forbid)
    result = executor.run_true_camera_batched_step(
        engine, tuple(_request(c, rgb_loss_enabled=rgb_enabled) for c in cameras), runtime,
        camera_microbatch_size=camera_count,
    )
    assert result.optimizer_step_count == 1
    assert len(readbacks) == 32 // camera_count
    assert "ssim" not in result.loss_breakdown.components
    assert ("Ll1" in result.loss_breakdown.components) is rgb_enabled
    torch.testing.assert_close(result.loss_breakdown.total, expected.total, rtol=1e-6, atol=1e-7)
    for parameter, gradient in zip((renders, model.get_scaling, model.get_opacity), expected_gradients):
        torch.testing.assert_close(parameter.grad, gradient, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("weight", [0.0, 1.0])
def test_b1_default_preserves_both_legacy_component_measurements(weight):
    args, opt, pipe, camera, model, image = _fixture()
    opt.lambda_dssim = weight
    result = objective.assemble_loss(_inputs(args, opt, pipe, camera, model, image))
    assert torch.is_tensor(result.components["Ll1"])
    assert torch.is_tensor(result.components["ssim"])


def test_rgb_disabled_evidence_keeps_real_l1_and_modality_alignment(monkeypatch):
    engine, runtime, cameras, renders = _canonical_case(monkeypatch, size=4)
    runtime.evidence_transaction_id = 31
    runtime.coordinate_frame_version = runtime.schema_version = 0
    runtime.surface_evidence_version = runtime.policy_version = 0
    engine.session.parameter_version = 0
    modalities = tuple(SupervisionModality(rgb_loss_enabled=False, rgb_evidence_enabled=i != 1) for i in range(4))
    requests = tuple(_evidence_request(c, evidence=True, modality=m) for c, m in zip(cameras, modalities))
    expected = (renders.detach() - cameras[0].original_image).abs().mean((1, 2, 3))
    expected[1] = 0.0  # Existing explicit ineligible-row padding, not a missing observation.
    monkeypatch.setattr(objective, "ssim", _forbid)
    result = executor.run_true_camera_batched_step(engine, requests, runtime)
    assert result.backward_ran
    assert torch.count_nonzero(renders.grad) == 0
    assert result.evidence.source_identities == tuple(r.view.source_identity for r in requests)
    assert result.evidence.modalities == modalities
    torch.testing.assert_close(result.evidence.rgb_residual_summary, expected)
    assert "ssim" not in result.loss_breakdown.components


@pytest.mark.parametrize("camera_count", [4, 2])
def test_partial_rgb_measurement_is_not_reported_as_zero_padded_mean(monkeypatch, camera_count):
    engine, runtime, cameras, _ = _canonical_case(monkeypatch, size=4)
    requests = tuple(_request(c, rgb_loss_enabled=c.uid < 2) for c in cameras)
    result = executor.run_true_camera_batched_step(engine, requests, runtime, camera_microbatch_size=camera_count)
    assert result.optimizer_step_count == 1
    assert "Ll1" not in result.loss_breakdown.components
    assert "ssim" not in result.loss_breakdown.components


def test_disabled_rgb_nan_keeps_existing_skip_and_reports_unmeasured(monkeypatch, capsys):
    engine, runtime, cameras, renders = _canonical_case(monkeypatch, size=4)
    with torch.no_grad():
        renders[0].fill_(float("nan"))
    monkeypatch.setattr(objective, "ssim", _forbid)
    monkeypatch.setattr(objective, "l1_loss", _forbid)
    result = executor.run_true_camera_batched_step(
        engine, tuple(_request(c, rgb_loss_enabled=False) for c in cameras), runtime,
    )
    assert result.skip_reason == "non_finite_loss"
    assert result.optimizer_step_count == 0
    assert "Ll1=not_measured ssim=not_measured" in capsys.readouterr().out


@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("weight,required", [(0.0, (True, False)), (1.0, (False, True)), (0.2, (True, True))])
def test_requested_pixel_terms_match_independent_formula_and_gradient(monkeypatch, batched, weight, required):
    args, opt, pipe, camera, model, image = _fixture()
    opt.lambda_dssim = weight
    l1 = objective.l1_loss(image, camera.original_image)
    ssim = objective.ssim(image, camera.original_image)
    expected = (1 - weight) * l1 + weight * (1 - ssim)
    expected_gradient, = torch.autograd.grad(expected, image)
    if not required[0]:
        monkeypatch.setattr(objective, "l1_loss", _forbid)
        monkeypatch.setattr(torch.Tensor, "abs", _forbid)
    if not required[1]:
        monkeypatch.setattr(objective, "ssim", _forbid)
        monkeypatch.setattr(batched_pixel_terms, "ssim_per_view", _forbid)
    terms = None
    if batched:
        terms, = batched_pixel_terms.assemble_batched_pixel_terms(
            renders=image.unsqueeze(0), gts=camera.original_image.unsqueeze(0), required_rgb=required,
        )
    inputs = replace(
        _inputs(args, opt, pipe, camera, model, image, precomputed_pixel_terms=terms),
        rgb_measurements=(False, False), skip_gaussian_regularizers=True,
    )
    actual = objective.assemble_loss(inputs)
    gradient, = torch.autograd.grad(actual.total_tensor, image)
    torch.testing.assert_close(actual.total, expected.detach(), rtol=0, atol=0)
    torch.testing.assert_close(gradient, expected_gradient, rtol=0, atol=0)
    assert set(actual.components) == {name for name, wanted in zip(("Ll1", "ssim"), required) if wanted}


def test_diagnostic_only_nonfinite_ssim_does_not_poison_zero_weight_objective(monkeypatch):
    args, opt, pipe, camera, model, image = _fixture()
    opt.lambda_dssim = 0.0
    monkeypatch.setattr(objective, "ssim", lambda *_: image.new_tensor(float("nan")))
    actual = objective.assemble_loss(_inputs(args, opt, pipe, camera, model, image))
    assert actual.loss_finite
    assert torch.isnan(actual.components["ssim"])
    actual.total_tensor.backward()
    assert torch.isfinite(image.grad).all()


def test_precomputed_missing_requested_measurement_rejects_without_recomputation(monkeypatch):
    args, opt, pipe, camera, model, image = _fixture()
    opt.lambda_dssim = 0.0
    terms, = batched_pixel_terms.assemble_batched_pixel_terms(
        renders=image.unsqueeze(0), gts=camera.original_image.unsqueeze(0), required_rgb=(True, False),
    )
    monkeypatch.setattr(objective, "ssim", _forbid)
    with pytest.raises(ValueError, match="requested RGB"):
        objective.assemble_loss(_inputs(args, opt, pipe, camera, model, image, precomputed_pixel_terms=terms))


@pytest.mark.parametrize("camera_count", [4, 2])
def test_disabled_rgb_keeps_depth_gradients_counts_and_requested_b(monkeypatch, camera_count):
    engine, runtime, cameras, renders = _canonical_case(monkeypatch, size=4)
    for camera in cameras:
        camera._sensor_depth_cache = torch.ones(1, 4, 4)
    runtime.args.surface_confidence_depth_loss = False
    runtime.loss_assembly_kwargs = {"depth_loss_weight": 0.25}
    depths = torch.full((4, 1, 4, 4), 2.0, requires_grad=True)
    with torch.no_grad():
        depths[1].fill_(float("inf"))
    original_render = executor.render_true_camera_batch
    def render(group, *args, **kwargs):
        return replace(original_render(group, *args, **kwargs), rendered_depth=depths[[c.uid for c in group]])
    monkeypatch.setattr(executor, "render_true_camera_batch", render)
    monkeypatch.setattr(objective, "l1_loss", _forbid)
    monkeypatch.setattr(objective, "ssim", _forbid)
    result = executor.run_true_camera_batched_step(
        engine, tuple(_request(c, rgb_loss_enabled=False, depth_loss_enabled=True) for c in cameras), runtime,
        camera_microbatch_size=camera_count,
    )
    assert result.optimizer_step_count == 1
    assert result.depth_valid_sample_count == 48
    assert result.batch_loss_scale == 0.25
    assert result.supervision_view_count == 4
    assert torch.count_nonzero(renders.grad) == 0
    assert torch.count_nonzero(depths.grad[1]) == 0
    torch.testing.assert_close(depths.grad[[0, 2, 3]], torch.full((3, 1, 4, 4), 0.25 / 4 / 16))
