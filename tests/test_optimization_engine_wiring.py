"""Tier B parity: ``GaussianOptimizationEngine.render_and_backward`` must
produce byte-identical gradients to the inline
``mcmc_strategy.step_pre_backward(...) -> loss.backward()`` sequence it
replaces in ``train_streaming.py``'s main training loop (Wave 2 step 1).

Follows the exact fixture pattern from
``tests/test_integration_services_delegate.py``/``tests/test_gaussian_uid.py``
(real GPU, minimal ``GsplatGaussianModel`` built via ``create_from_pcd``).
"""

from __future__ import annotations

from argparse import Namespace

import numpy as np
import pytest
import torch

from scene.gsplat_model import GsplatGaussianModel
from eligsir.map.adapters.gsplat import GsplatMapBackend
from eligsir.optimization.gaussian.engine import GaussianOptimizationEngine
from eligsir.utils.graphics_utils import BasicPointCloud

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU")


def _make_training_args():
    return Namespace(
        iterations=100,
        position_lr_init=0.00016,
        position_lr_final=0.0000016,
        position_lr_delay_mult=0.01,
        position_lr_max_steps=20000,
        feature_lr=0.0025,
        feature_lr_final=0.00025,
        opacity_lr=0.05,
        opacity_lr_final=0.001,
        scaling_lr=0.005,
        scaling_lr_final=0.0005,
        rotation_lr=0.001,
        rotation_lr_final=0.0001,
        percent_dense=0.01,
        lambda_dssim=0.2,
        densification_interval=100,
        opacity_reset_interval=30000,
        densify_from_iter=500,
        densify_until_iter=15000,
        densify_grad_threshold=0.0002,
        random_background=False,
    )


def _make_model(seed: int, n_points: int = 20) -> GsplatGaussianModel:
    torch.manual_seed(seed)
    model = GsplatGaussianModel(sh_degree=0)
    points = torch.rand(n_points, 3, device="cuda")
    colors = torch.rand(n_points, 3, device="cuda")
    pcd = BasicPointCloud(
        points=points.cpu().numpy(), colors=colors.cpu().numpy(),
        normals=np.zeros((n_points, 3)),
    )
    model.create_from_pcd(pcd, spatial_lr_scale=1.0)
    model.training_setup(_make_training_args())
    return model


class _NoOpMcmcStrategy:
    """Real no-op ``step_pre_backward`` shape matching
    ``utils.strategies.mcmc_strategy``'s base-strategy default (``**_kwargs``,
    does nothing) -- exercised here as the pre-backward hook the engine
    method is required to call before ``loss.backward()``."""

    def __init__(self):
        self.calls = []

    def step_pre_backward(self, **kwargs):
        self.calls.append(kwargs)


def _fixed_loss(model: GsplatGaussianModel) -> torch.Tensor:
    """Deterministic scalar loss over the model's learnable params, standing
    in for the real (much larger) loss-assembly block -- this test verifies
    the engine's backward wiring, not the loss math itself."""
    means = model.get_xyz
    opac = model.get_opacity
    scales = model.get_scaling
    return (means.pow(2).sum() + opac.pow(2).sum() + scales.pow(2).sum())


def test_render_and_backward_matches_inline_sequence():
    render_pkg = {"visibility_filter": torch.ones(20, dtype=torch.bool, device="cuda")}
    iteration = 100

    # --- Path A: inline sequence (as it exists today in train_streaming.py) ---
    model_a = _make_model(seed=0)
    strategy_a = _NoOpMcmcStrategy()
    loss_a = _fixed_loss(model_a)
    strategy_a.step_pre_backward(
        gaussians=model_a, args=None, iteration=iteration, render_pkg=render_pkg, loss=loss_a,
    )
    loss_a.backward()

    # --- Path B: through GaussianOptimizationEngine.render_and_backward ---
    model_b = _make_model(seed=0)
    backend_b = GsplatMapBackend(model_b)
    engine = GaussianOptimizationEngine(backend_b)
    strategy_b = _NoOpMcmcStrategy()
    loss_b = _fixed_loss(model_b)
    result = engine.render_and_backward(
        model_b, None, iteration, strategy_b, render_pkg, loss_b,
    )

    assert result is None
    assert torch.allclose(loss_a, loss_b, atol=0)
    assert torch.allclose(model_a.means_param.grad, model_b.means_param.grad, atol=0)
    assert torch.allclose(model_a.raw_scales.grad, model_b.raw_scales.grad, atol=0)
    assert torch.allclose(model_a.raw_opacities.grad, model_b.raw_opacities.grad, atol=0)

    # engine's call must reach the pre-backward hook with the same kwargs
    # shape as the inline call site.
    assert len(strategy_b.calls) == 1
    assert strategy_b.calls[0]["iteration"] == iteration
    assert strategy_b.calls[0]["render_pkg"] is render_pkg
    assert strategy_b.calls[0]["loss"] is loss_b
