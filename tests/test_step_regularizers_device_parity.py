"""CUDA-only forward/gradient qualifier for once-per-step regularizers."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a CUDA-capable training container"
)

from eligsir.optimization.gaussian.losses import (
    _streaming_young_opacity_reg,
    assemble_step_regularizers,
)
from eligsir.utils.compiled_kernels import active_reg_core_masked


def _case(batch_size: int, population: int = 4096):
    torch.manual_seed(batch_size)
    raw_opacity = torch.randn(population, 1, device="cuda", requires_grad=True)
    raw_scale = (torch.rand(population, 3, device="cuda") + 0.1).detach().requires_grad_()
    model = SimpleNamespace(
        get_opacity=torch.sigmoid(raw_opacity),
        get_scaling=raw_scale,
        birth_iter=torch.randint(0, 900, (population,), device="cuda"),
        support_count=torch.randint(0, 20, (population,), device="cuda").float(),
        last_supervised_iter=torch.randint(0, 1000, (population,), device="cuda"),
    )
    args = SimpleNamespace(
        opacity_reg=0.05, scale_reg=0.02, anisotropy_reg=0.1,
        streaming_young_reg_exempt_iters=50, streaming_maturity_reg_ramp=True,
        streaming_recency_reg_horizon=100.0, streaming_debt_replay_target_support=10,
        streaming_maturity_reg_ramp_min=0.2, streaming_recency_reg_ramp_min=0.2,
    )
    masks = torch.rand(batch_size, population, device="cuda") > 0.5
    masks[:, 0] = True
    weights = (masks.float() / masks.sum(1, keepdim=True)).sum(0)
    return model, args, masks, weights, raw_opacity, raw_scale


@pytest.mark.parametrize("batch_size", [8, 16, 32])
def test_step_regularizers_match_cuda_forward_and_gradients(batch_size):
    model, args, masks, weights, raw_opacity, raw_scale = _case(batch_size)
    batched = assemble_step_regularizers(model, args, 1000, weights, batch_size=batch_size)
    batched_grads = torch.autograd.grad(batched, (raw_opacity, raw_scale), retain_graph=True)
    reference = sum(
        active_reg_core_masked(model.get_opacity, model.get_scaling, mask, w_opacity=0.0, w_scale=args.scale_reg, w_aniso=args.anisotropy_reg)
        + _streaming_young_opacity_reg(model, args, 1000, mask, args.streaming_young_reg_exempt_iters, args.streaming_maturity_reg_ramp)
        for mask in masks
    ) / batch_size
    reference_grads = torch.autograd.grad(reference, (raw_opacity, raw_scale))
    torch.testing.assert_close(batched, reference, atol=3e-5, rtol=3e-5)
    for actual, expected in zip(batched_grads, reference_grads):
        torch.testing.assert_close(actual, expected, atol=3e-5, rtol=3e-5)


def test_step_regularizers_hot_path_has_no_host_to_device_copy():
    model, args, _masks, weights, raw_opacity, raw_scale = _case(8)
    assemble_step_regularizers(model, args, 1000, weights, batch_size=8)  # warm kernels
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as profile:
        loss = assemble_step_regularizers(model, args, 1000, weights, batch_size=8)
        torch.autograd.grad(loss, (raw_opacity, raw_scale))
    assert not any("htod" in event.key.lower() for event in profile.key_averages())
