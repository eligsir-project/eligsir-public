"""CUDA-only qualifier for the batched pixel objective; not a mapper benchmark."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a CUDA-capable training container"
)

from eligsir.optimization.gaussian.batched_pixel_terms import assemble_batched_pixel_terms
from eligsir.optimization.gaussian.depth_loss_weighting import (
    combine_depth_loss_weights,
    masked_depth_error,
    weighted_depth_error_mean,
)
from eligsir.utils.loss_utils import l1_loss, set_ssim_mode, ssim


def _inputs(batch_size: int):
    torch.manual_seed(batch_size)
    device = "cuda"
    gt = torch.rand(batch_size, 3, 16, 16, device=device)
    render = (gt + 0.05 * torch.randn_like(gt)).clamp(0, 1).detach().requires_grad_()
    sensor = torch.rand(batch_size, 1, 16, 16, device=device) * 3
    sensor[0] = 0
    sensor[0, :, 0, 0] = 1  # deliberately disparate valid-depth populations
    sensor[1, :, :8] = 0
    depth = (sensor + 0.03 * torch.randn_like(sensor)).clamp_min(0).detach().requires_grad_()
    alpha = torch.rand_like(sensor)
    confidence = torch.rand_like(sensor).clamp_min(0.1)
    return render, gt, depth, sensor, alpha, confidence


def _reference(render, gt, depth, sensor, alpha, confidence):
    terms = []
    for index in range(render.shape[0]):
        valid = (sensor[index] > 0) & (depth[index].detach() > 0) & (alpha[index] > 0.35)
        error = masked_depth_error(depth[index], sensor[index], valid, loss_type="huber")
        weighted = weighted_depth_error_mean(
            error, combine_depth_loss_weights(confidence[index]), valid
        )
        terms.append(0.8 * l1_loss(render[index], gt[index]) + 0.2 * (1 - ssim(render[index], gt[index])) + 0.05 * weighted)
    return sum(terms) / len(terms)


@pytest.mark.parametrize("batch_size", [8, 16, 32])
@pytest.mark.parametrize("requested_mode", ["ref", "fused", "compiled"])
def test_batched_pixel_terms_match_cuda_forward_and_gradients(batch_size, requested_mode):
    from eligsir.utils import loss_utils

    previous_mode = loss_utils._SSIM_MODE
    try:
        actual_mode = set_ssim_mode(requested_mode)
        if actual_mode != requested_mode:
            pytest.skip(f"{requested_mode} SSIM backend is unavailable")
        batched_inputs = _inputs(batch_size)
        render, gt, depth, sensor, alpha, confidence = batched_inputs
        views = assemble_batched_pixel_terms(
            renders=render, gts=gt, rendered_depth=depth, sensor_depth=sensor,
            alpha=alpha, alpha_gate=0.35, depth_loss_type="huber", weight_maps=(confidence,),
        )
        batched = sum(0.8 * view.Ll1 + 0.2 * (1 - view.ssim) + 0.05 * view.depth for view in views) / batch_size
        batched_grads = torch.autograd.grad(batched, (render, depth))

        reference_inputs = _inputs(batch_size)
        reference = _reference(*reference_inputs)
        reference_grads = torch.autograd.grad(reference, (reference_inputs[0], reference_inputs[2]))
        torch.testing.assert_close(batched, reference, atol=5e-4, rtol=5e-4)
        for actual, expected in zip(batched_grads, reference_grads):
            torch.testing.assert_close(actual, expected, atol=5e-4, rtol=5e-4)
    finally:
        set_ssim_mode(previous_mode)


def test_batched_pixel_terms_hot_path_has_no_host_to_device_copy():
    inputs = _inputs(8)
    assemble_batched_pixel_terms(
        renders=inputs[0], gts=inputs[1], rendered_depth=inputs[2], sensor_depth=inputs[3],
        alpha=inputs[4], alpha_gate=0.35, depth_loss_type="huber", weight_maps=(inputs[5],),
    )  # warm the SSIM window cache before profiling
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as profile:
        views = assemble_batched_pixel_terms(
            renders=inputs[0], gts=inputs[1], rendered_depth=inputs[2], sensor_depth=inputs[3],
            alpha=inputs[4], alpha_gate=0.35, depth_loss_type="huber", weight_maps=(inputs[5],),
        )
        torch.autograd.grad(sum(view.Ll1 + view.ssim + view.depth for view in views), (inputs[0], inputs[2]))
    assert not any("htod" in event.key.lower() for event in profile.key_averages())
