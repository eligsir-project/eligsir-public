"""Phase 1 acceptance test: GaussianEnergyPolicy output matches legacy API.

Verifies that ``GaussianEnergyPolicy.compute_utility()`` and
``compute_dead_mask()`` produce numerically equivalent results to the direct
calls to ``utils.energy_mcmc.compute_gaussian_utility`` and
``utils.energy_mcmc.compute_dead_mask``.

Runs on CPU with a tiny synthetic Gaussian model (~100 splats).
"""

from __future__ import annotations

import types

import pytest
import torch

from eligsir.utils.compiled_kernels import configure_torch_compile
from eligsir.utils.energy_mcmc import compute_dead_mask, compute_gaussian_utility
from eligsir.utils.gaussian_energy_policy import GaussianEnergyConfig, GaussianEnergyPolicy


@pytest.fixture(scope="session", autouse=True)
def _register_kernels():
    """Register compiled kernels in eager mode before any energy test."""
    import types
    _args = types.SimpleNamespace(
        compile_mode="off",
        compile_energy=False,
        compile_utility=False,
        compile_sh=False,
        compile_dynamic=False,
        compile_after_iter=0,
        compile_growth_margin=0,
    )
    configure_torch_compile(_args)


def _make_fake_gaussians(n: int = 100, device="cpu"):
    """Build a minimal synthetic Gaussian model with all expected attributes.

    Args:
        n: Number of Gaussians.
        device: Torch device string.

    Returns:
        A SimpleNamespace acting as a Gaussian model.
    """
    g = types.SimpleNamespace()
    g.get_xyz = torch.randn(n, 3, device=device)
    g.get_xyz.grad = torch.randn(n, 3, device=device) * 0.01

    # Raw opacity (pre-sigmoid), get_opacity returns sigmoid(raw)
    raw_opacity = torch.randn(n, 1, device=device) * 2.0
    g._opacity = raw_opacity
    g._opacity.grad = torch.randn(n, 1, device=device) * 0.001
    g.get_opacity = torch.sigmoid(raw_opacity)

    raw_scale = torch.randn(n, 3, device=device) * 0.5
    g._scaling = raw_scale
    g._scaling.grad = torch.randn(n, 3, device=device) * 0.001
    # Clamp to positive: exp of raw
    g.get_scaling = torch.exp(raw_scale.clamp(-4, 4))

    g.get_rotation = torch.nn.functional.normalize(
        torch.randn(n, 4, device=device), dim=-1
    )

    g.visibility_ema = torch.rand(n, 1, device=device)

    # gsplat layout params (for _named_grad support)
    g.params = {
        "means": g.get_xyz,
        "opacities": g._opacity,
        "scales": g._scaling,
    }
    g.params["means"].grad = g.get_xyz.grad
    g.params["opacities"].grad = g._opacity.grad
    g.params["scales"].grad = g._scaling.grad

    return g


def _make_render_pkg(n: int = 100, device="cpu"):
    """Build a minimal synthetic render package.

    Args:
        n: Number of Gaussians.
        device: Torch device string.

    Returns:
        Dict with ``visibility_filter`` key.
    """
    return {
        "visibility_filter": torch.rand(n, device=device) > 0.3,
    }


@pytest.fixture
def fake_model():
    return _make_fake_gaussians(100)


@pytest.fixture
def fake_render(fake_model):
    return _make_render_pkg(fake_model.get_xyz.shape[0])


class TestEnergyPolicyParity:
    """Verify GaussianEnergyPolicy output matches legacy functions exactly."""

    def test_utility_parity(self, fake_model, fake_render):
        """Utility scores from policy == legacy compute_gaussian_utility."""
        cfg = GaussianEnergyConfig()
        policy = GaussianEnergyPolicy(cfg)

        legacy = compute_gaussian_utility(
            fake_model, fake_render, iteration=500,
            w_alpha=cfg.w_alpha, w_vis=cfg.w_vis, w_grad=cfg.w_grad,
            w_scale=cfg.w_scale, w_dead=cfg.w_dead, w_support=cfg.w_support,
            beta_opacity=cfg.beta_opacity, beta_scale=cfg.beta_scale,
            alpha_dead=cfg.alpha_dead, w_aniso_utility=cfg.w_aniso_utility,
            aniso_soft_limit=cfg.aniso_soft_limit,
        )

        new_utility = policy.compute_utility(fake_model, fake_render, iteration=500)

        assert new_utility.shape == legacy.shape, "Shape mismatch"
        assert torch.allclose(new_utility, legacy, atol=1e-5), (
            f"Utility mismatch: max_diff={( new_utility - legacy).abs().max():.2e}"
        )

    def test_dead_mask_parity_no_utility(self, fake_model):
        """Dead mask (no utility) from policy == legacy compute_dead_mask."""
        cfg = GaussianEnergyConfig()
        policy = GaussianEnergyPolicy(cfg)

        legacy = compute_dead_mask(
            fake_model, utility=None,
            opacity_threshold=cfg.alpha_dead,
            support_threshold=cfg.support_threshold,
            utility_quantile=cfg.utility_quantile,
            use_utility_quantile=False,
            scale_kill_threshold=cfg.scale_kill_threshold,
            scale_moderate_threshold=cfg.scale_moderate_threshold,
            scale_moderate_opacity_mult=cfg.scale_moderate_opacity_mult,
            scale_dead_support_mult=cfg.scale_dead_support_mult,
            anisotropy_kill_ratio=cfg.anisotropy_kill_ratio,
        )

        new_cfg = GaussianEnergyConfig()
        new_cfg.use_utility_quantile = False
        new_policy = GaussianEnergyPolicy(new_cfg)
        new_mask = new_policy.compute_dead_mask(fake_model, utility=None)

        assert new_mask.shape == legacy.shape
        assert torch.equal(new_mask, legacy), (
            f"Dead mask mismatch: {(new_mask ^ legacy).sum()} bits differ"
        )

    def test_dead_mask_parity_with_utility(self, fake_model, fake_render):
        """Dead mask (with utility) from policy == legacy compute_dead_mask."""
        cfg = GaussianEnergyConfig()
        policy = GaussianEnergyPolicy(cfg)

        utility = compute_gaussian_utility(
            fake_model, fake_render, iteration=500,
            w_alpha=cfg.w_alpha, w_vis=cfg.w_vis, w_grad=cfg.w_grad,
            w_scale=cfg.w_scale, w_dead=cfg.w_dead, w_support=cfg.w_support,
            beta_opacity=cfg.beta_opacity, beta_scale=cfg.beta_scale,
            alpha_dead=cfg.alpha_dead, w_aniso_utility=cfg.w_aniso_utility,
            aniso_soft_limit=cfg.aniso_soft_limit,
        )

        legacy = compute_dead_mask(
            fake_model, utility=utility,
            opacity_threshold=cfg.alpha_dead,
            support_threshold=cfg.support_threshold,
            utility_quantile=cfg.utility_quantile,
            use_utility_quantile=True,
            scale_kill_threshold=cfg.scale_kill_threshold,
            scale_moderate_threshold=cfg.scale_moderate_threshold,
            scale_moderate_opacity_mult=cfg.scale_moderate_opacity_mult,
            scale_dead_support_mult=cfg.scale_dead_support_mult,
            anisotropy_kill_ratio=cfg.anisotropy_kill_ratio,
        )

        new_mask = policy.compute_dead_mask(fake_model, utility=utility)
        assert torch.equal(new_mask, legacy), (
            f"Dead mask (with utility) mismatch: {(new_mask ^ legacy).sum()} bits differ"
        )

    def test_terms_shape(self, fake_model, fake_render):
        """compute_terms() returns tensors with correct shape [N]."""
        policy = GaussianEnergyPolicy()
        terms = policy.compute_terms(fake_model, fake_render, iteration=100)
        n = fake_model.get_xyz.shape[0]
        assert terms.alpha.shape == (n,)
        assert terms.support.shape == (n,)
        assert terms.max_scale.shape == (n,)
        assert terms.anisotropy.shape == (n,)

    def test_terms_detach_state(self, fake_model, fake_render):
        """State tensors (support, age_frac) are detached when detach_state=True."""
        policy = GaussianEnergyPolicy(GaussianEnergyConfig(detach_state=True))
        terms = policy.compute_terms(fake_model, fake_render, iteration=100)
        assert not terms.support.requires_grad, "support should be detached"

    def test_utility_parity_with_legacy_args(self, fake_model, fake_render):
        """Phase 3: compute_gaussian_utility with legacy args == without args.

        Asserts that passing args with energy_mode='legacy_utility' produces
        bit-identical results to calling without args (the no-args legacy path).
        """
        import types as _t
        mock_legacy_args = _t.SimpleNamespace(energy_mode="legacy_utility")

        legacy_no_args = compute_gaussian_utility(fake_model, fake_render, iteration=500)
        legacy_with_args = compute_gaussian_utility(
            fake_model, fake_render, iteration=500, args=mock_legacy_args
        )
        assert torch.allclose(legacy_no_args, legacy_with_args, atol=0.0), (
            "Legacy default with args must be bit-identical to no-args call"
        )

    def test_utility_population_energy_mode_shape(self, fake_model, fake_render):
        """Phase 3: energy_mode='population_energy' returns valid [N] utility tensor."""
        import types as _t
        pop_args = _t.SimpleNamespace(
            energy_mode="population_energy",
            energy_w_opacity=1.0,
            energy_w_anisotropy=0.3,
            energy_w_conflict=0.5,
            energy_w_age=0.2,
            energy_alpha_dead=0.005,
            energy_w_grad=3.0,
            energy_w_support=2.0,
            energy_w_scale=0.5,
            energy_w_dead=1.0,
        )
        result = compute_gaussian_utility(fake_model, fake_render, iteration=500, args=pop_args)
        n = fake_model.get_xyz.shape[0]
        assert result.shape == (n,), f"Expected [{n}], got {result.shape}"
        assert result.dtype == torch.float32, "Expected float32"
        # Population energy utility is negated, so it can range widely
        assert torch.isfinite(result).all(), "All values must be finite"

    def test_terms_with_stale_visibility_ema_falls_back_instead_of_crashing(self, fake_render):
        """Regression: a visibility_ema left over from before an external
        growth event (e.g. GsplatDefaultStrategy's clone/split, which
        doesn't extend this MCMC-strategy-specific buffer) must not crash
        compute_terms() with a shape-mismatch RuntimeError -- it should fall
        back to uniform support, matching how the sibling optional buffers
        (anchor_confidence, surface_state, anchor_normal) already behave in
        this same function. Found live 2026-08-12: resuming an
        externally-grown 854830-Gaussian checkpoint with a 97862-sized
        visibility_ema crashed training outright.
        """
        n = 100
        model = _make_fake_gaussians(n)
        model.visibility_ema = torch.rand(37, 1)  # stale size, real bug had 97862 vs 854830
        render_pkg = _make_render_pkg(n)

        policy = GaussianEnergyPolicy()
        terms = policy.compute_terms(model, render_pkg, iteration=100)
        assert terms.support.shape == (n,)
        assert torch.isfinite(terms.support).all()

        energy = policy.compute_population_energy(terms)
        assert energy.total.shape == (n,)
        assert torch.isfinite(energy.total).all()

    def test_dead_mask_with_stale_visibility_ema_falls_back_instead_of_crashing(self):
        """Same regression as above, for the legacy-clauses dead-mask path
        (utils/strategies/mcmc/energy_policy.py's second visibility_ema
        read site)."""
        n = 100
        model = _make_fake_gaussians(n)
        model.visibility_ema = torch.rand(37, 1)

        policy = GaussianEnergyPolicy()
        mask = policy.compute_dead_mask(model, current_iter=100)
        assert mask.shape == (n,)

    def test_terms_with_stale_lineage_and_tracking_buffers_falls_back(self, fake_render):
        """Regression (round 2 of the same class of bug, found immediately
        after the visibility_ema fix landed and this exact scenario crashed
        again on a real GPU run): birth_iter, anchor_xyz, depth_conflict_count
        and provisional are all read in compute_terms() the same
        hasattr()-only way visibility_ema was -- any one of them stale after
        an external growth event (e.g. GsplatDefaultStrategy's clone/split)
        crashes compute_population_energy() the same way. All four must
        fall back gracefully when their size doesn't match N.
        """
        n = 100
        model = _make_fake_gaussians(n)
        model.birth_iter = torch.zeros(37)
        model.anchor_xyz = torch.zeros(37, 3)
        model.depth_conflict_count = torch.zeros(37)
        model.provisional = torch.zeros(37, dtype=torch.bool)

        policy = GaussianEnergyPolicy()
        terms = policy.compute_terms(model, fake_render, iteration=100)
        assert terms.alpha.shape == (n,)
        assert terms.age_frac is None
        assert terms.has_anchor is None
        assert terms.has_depth_conflict is None
        assert terms.provisional is None

        energy = policy.compute_population_energy(terms)
        assert energy.total.shape == (n,)
        assert torch.isfinite(energy.total).all()
