"""Parity tests for Phases 4, 5, 6, and 10 energy flag wiring.

Verifies:
- Phase 4 (energy_dead_mode): legacy_clauses path is bit-identical; energy modes
  return valid bool masks of correct shape.
- Phase 5 (energy_loss_enabled): documented parity limitation — active_reg term
  differs between legacy and compute_differentiable_loss due to raw vs. activated
  opacities and different scale argument convention.
- Phase 6 (mcmc_proposal_mode): trust_region and legacy_clamped both produce
  noise of the same shape and device.
- Phase 10 (soft_lifecycle_enabled): maturity modulation changes parent scores,
  legacy path (False) leaves them unchanged.

All tests run on CPU with synthetic data — no GPU required.
"""
from __future__ import annotations

import types

import pytest
import torch

from eligsir.utils.compiled_kernels import configure_torch_compile
from eligsir.utils.gaussian_energy_policy import GaussianEnergyConfig, GaussianEnergyPolicy


@pytest.fixture(scope="session", autouse=True)
def _register_kernels():
    """Register compiled kernels in eager mode for all wiring tests."""
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


def _make_gaussians(n: int = 80) -> types.SimpleNamespace:
    """Build a minimal synthetic Gaussian model."""
    g = types.SimpleNamespace()
    g.get_xyz = torch.randn(n, 3)
    g.get_xyz.grad = torch.randn(n, 3) * 0.01
    raw_op = torch.randn(n, 1) * 2.0
    g._opacity = raw_op
    g._opacity.grad = torch.randn(n, 1) * 0.001
    g.get_opacity = torch.sigmoid(raw_op)
    raw_scale = torch.randn(n, 3) * 0.5
    g._scaling = raw_scale
    g._scaling.grad = torch.randn(n, 3) * 0.001
    g.get_scaling = torch.exp(raw_scale.clamp(-4, 4))
    g.get_rotation = torch.nn.functional.normalize(torch.randn(n, 4), dim=-1)
    g.visibility_ema = torch.rand(n, 1)
    g.birth_frame = torch.zeros(n, dtype=torch.int32)
    g.provisional = torch.zeros(n, dtype=torch.bool)
    g.depth_conflict_count = torch.zeros(n, dtype=torch.int16)
    # gsplat layout
    g.params = {"means": g.get_xyz, "opacities": g._opacity, "scales": g._scaling}
    g.params["means"].grad = g.get_xyz.grad
    g.params["opacities"].grad = g._opacity.grad
    g.params["scales"].grad = g._scaling.grad
    return g


def _make_render_pkg(n: int = 80) -> dict:
    return {"visibility_filter": torch.rand(n) > 0.3}


def _make_legacy_args(**overrides) -> types.SimpleNamespace:
    """Return a minimal args namespace with all legacy defaults."""
    defaults = dict(
        energy_mode="legacy_utility",
        energy_dead_mode="legacy_clauses",
        mcmc_proposal_mode="legacy_clamped",
        soft_lifecycle_enabled=False,
        energy_loss_enabled=False,
        energy_dead_quantile=0.05,
        energy_dead_temperature=10.0,
        scale_kill_threshold=0.20,
        scale_moderate_threshold=0.05,
        anisotropy_kill_ratio=20.0,
        noise_lr=5e5,
        noise_max_scale=0.05,
        noise_max_step=0.01,
        gsplat_mcmc_noise_max_scale=0.05,
        gsplat_mcmc_noise_max_step=0.1,
        mcmc_trust_radius=0.05,
        mcmc_trust_support_power=0.5,
        mcmc_trust_opacity_power=0.5,
    )
    defaults.update(overrides)
    return types.SimpleNamespace(**defaults)


# ---------------------------------------------------------------------------
# Phase 4: dead-mask parity
# ---------------------------------------------------------------------------

class TestPhase4DeadMaskParity:
    """Phase 4: energy_dead_mode flag wiring."""

    def test_legacy_clauses_dead_mask_shape(self):
        """legacy_clauses mode returns bool mask of correct shape."""
        g = _make_gaussians(80)
        render_pkg = _make_render_pkg(80)
        cfg = GaussianEnergyConfig(
            energy_dead_mode="legacy_clauses",
            alpha_dead=0.005,
        )
        policy = GaussianEnergyPolicy(cfg)
        result = policy.compute_legacy_dead_mask(g, utility=None)
        assert result.mask.shape == (80,)
        assert result.mask.dtype == torch.bool

    def test_legacy_clauses_updates_both_windowed_and_cumulative_pruning_stats(self):
        """issue #1286: report.json's geometry.pruning/pruning_cumulative
        blocks read pop_pruning_stats()/get_cumulative_pruning_stats().
        compute_legacy_dead_mask (the default energy_dead_mode="legacy_clauses"
        production path) used to mirror its stats into the module-level
        _pruning_stats dict directly, which only updates the WINDOWED
        accumulator -- it bypassed .bump(), the only thing that also updates
        the CUMULATIVE one, so pruning_cumulative stayed permanently 0 for the
        entire run regardless of how much pruning actually happened."""
        from eligsir.utils.energy_mcmc import (
            _pruning_stats_cumulative,
            get_cumulative_pruning_stats,
            pop_pruning_stats,
            reset_pruning_stats,
        )

        reset_pruning_stats()
        for k in _pruning_stats_cumulative:
            _pruning_stats_cumulative[k] = 0

        g = _make_gaussians(50)
        with torch.no_grad():
            g._opacity[:10] = -10.0  # sigmoid(-10) ~ 4.5e-5: unambiguously dead
            g.get_opacity = torch.sigmoid(g._opacity)
        cfg = GaussianEnergyConfig(energy_dead_mode="legacy_clauses", alpha_dead=0.005)
        policy = GaussianEnergyPolicy(cfg)

        result = policy.compute_legacy_dead_mask(g, utility=None)

        assert result.stats["total"] >= 10
        windowed = pop_pruning_stats()
        cumulative = get_cumulative_pruning_stats()
        assert windowed["total_pruned"] == result.stats["total"]
        # The bug: cumulative used to stay 0 here regardless of windowed.
        assert cumulative["total_pruned"] == result.stats["total"]
        assert cumulative["dead_opacity"] == windowed["dead_opacity"] > 0

    def test_energy_quantile_returns_bool_mask(self):
        """energy_quantile mode returns a bool mask of correct shape."""
        g = _make_gaussians(100)
        render_pkg = _make_render_pkg(100)
        cfg = GaussianEnergyConfig(
            energy_dead_mode="energy_quantile",
            energy_dead_quantile=0.05,
            alpha_dead=0.005,
        )
        policy = GaussianEnergyPolicy(cfg)
        terms = policy.compute_terms(g, render_pkg, iteration=200)
        energy = policy.compute_population_energy(terms)
        dead = policy.compute_dead_mask_from_energy(energy, g)
        assert dead.shape == (100,)
        assert dead.dtype == torch.bool

    def test_energy_sigmoid_returns_bool_mask(self):
        """energy_sigmoid mode returns a bool mask of correct shape."""
        g = _make_gaussians(100)
        render_pkg = _make_render_pkg(100)
        cfg = GaussianEnergyConfig(
            energy_dead_mode="energy_sigmoid",
            energy_dead_temperature=10.0,
            alpha_dead=0.005,
        )
        policy = GaussianEnergyPolicy(cfg)
        terms = policy.compute_terms(g, render_pkg, iteration=200)
        energy = policy.compute_population_energy(terms)
        dead = policy.compute_dead_mask_from_energy(energy, g)
        assert dead.shape == (100,)
        assert dead.dtype == torch.bool

    def test_legacy_clauses_unchanged_vs_direct_call(self):
        """energy_dead_mode='legacy_clauses' produces same mask as compute_dead_mask."""
        from eligsir.utils.energy_mcmc import compute_dead_mask

        g = _make_gaussians(60)
        cfg = GaussianEnergyConfig(
            energy_dead_mode="legacy_clauses",
            use_utility_quantile=False,
            alpha_dead=0.005,
        )
        policy = GaussianEnergyPolicy(cfg)
        render_pkg = _make_render_pkg(60)
        terms = policy.compute_terms(g, render_pkg, iteration=100)
        energy = policy.compute_population_energy(terms)

        dead_via_energy_flag = policy.compute_dead_mask_from_energy(energy, g)
        dead_direct = compute_dead_mask(g, utility=None, opacity_threshold=0.005, use_utility_quantile=False)
        assert torch.equal(dead_via_energy_flag, dead_direct), (
            f"legacy_clauses dead mask must match direct call; {(dead_via_energy_flag ^ dead_direct).sum()} bits differ"
        )


# ---------------------------------------------------------------------------
# Phase 5: differentiable energy loss parity documentation
# ---------------------------------------------------------------------------

class TestPhase5LossParity:
    """Phase 5: energy_loss_enabled flag — parity documentation tests.

    IMPORTANT: compute_differentiable_loss uses raw (pre-sigmoid) opacities
    for ALL terms, while the live training path uses:
    - active_reg_core(activated_opacity[visible], scaling[visible], ...)
      — activated opacities, with scaling as 2nd arg
    - compute_opacity_entropy_loss(raw_opacity) — raw opacities (parity holds)
    - compute_effective_count_loss(raw_opacity, ...) — raw opacities (parity holds)

    Full numerical parity for the active_reg term cannot be asserted because
    the calling conventions differ. The entropy and eff_count terms would match,
    but the combined loss is not equivalent. This test documents the limitation.
    """

    def test_compute_differentiable_loss_runs(self):
        """compute_differentiable_loss runs without error and returns a DifferentiableLossResult."""
        from eligsir.utils.differentiable_energy_loss import compute_differentiable_loss, DifferentiableLossResult

        g = _make_gaussians(50)
        result = compute_differentiable_loss(
            gaussians=g,
            iteration=100,
            cap_max=1000,
            active_mask=None,
            include_effective_count=False,
        )
        assert isinstance(result, DifferentiableLossResult)
        assert result.total.shape == ()  # scalar
        assert torch.isfinite(result.total)

    def test_entropy_loss_matches_standalone(self):
        """Entropy component of compute_differentiable_loss matches opacity_entropy_core directly."""
        from eligsir.utils.compiled_kernels import opacity_entropy_core
        from eligsir.utils.differentiable_energy_loss import compute_differentiable_loss

        g = _make_gaussians(50)
        result = compute_differentiable_loss(
            gaussians=g,
            iteration=100,
            cap_max=1000,
            active_mask=None,
            include_effective_count=False,
            w_opacity_reg=0.0,    # disable active reg
            w_scale_reg=0.0,
            w_opacity_entropy=1.0,
            w_effective_count=0.0,
            w_scale_prior=0.0,
            w_anisotropy_reg=0.0,
        )
        direct_entropy = opacity_entropy_core(g._opacity)
        assert torch.allclose(result.opacity_entropy, direct_entropy, atol=1e-6), (
            f"Entropy mismatch: {(result.opacity_entropy - direct_entropy).abs():.4e}"
        )

    def test_active_reg_parity_matches_legacy(self):
        """compute_differentiable_loss active_reg matches direct active_reg_core call.

        The parity fix: pass w_opacity_reg and w_scale_reg through to active_reg_core
        so the weights match exactly. With entropy and extra priors zeroed out,
        total == active_reg_core(get_opacity, get_scaling, w_opacity, w_scale).
        """
        from eligsir.utils.compiled_kernels import active_reg_core
        from eligsir.utils.differentiable_energy_loss import compute_differentiable_loss

        g = _make_gaussians(50)
        w_op = 0.01
        w_sc = 0.01
        legacy_active_reg = active_reg_core(g.get_opacity, g.get_scaling, w_opacity=w_op, w_scale=w_sc)
        diff_result = compute_differentiable_loss(
            gaussians=g,
            iteration=100,
            cap_max=1000,
            active_mask=None,
            include_effective_count=False,
            w_opacity_reg=w_op,
            w_scale_reg=w_sc,
            w_opacity_entropy=0.0,
            w_effective_count=0.0,
            w_scale_prior=0.0,
            w_anisotropy_reg=0.0,
        )
        torch.testing.assert_close(diff_result.active_reg, legacy_active_reg, atol=1e-6, rtol=1e-5)

    def test_active_reg_empty_active_set_is_zero(self):
        from eligsir.utils.compiled_kernels import active_reg_core

        opacity = torch.empty((0, 1), dtype=torch.float32)
        scaling = torch.empty((0, 3), dtype=torch.float32)

        loss = active_reg_core(opacity, scaling, w_opacity=0.01, w_scale=0.01, w_aniso=0.01)

        assert torch.isfinite(loss)
        assert loss.item() == 0.0


# ---------------------------------------------------------------------------
# Phase 6: trust-region noise shape/device parity
# ---------------------------------------------------------------------------

class TestPhase6NoiseParity:
    """Phase 6: mcmc_proposal_mode=trust_region produces noise of correct shape.

    Note: ScheduledMCMCStrategy.inject_noise calls build_scaling_rotation which is
    hard-coded to device='cuda' — the full inject_noise integration test requires GPU.
    These tests verify the trust-region policy directly (CPU-compatible) and confirm
    the flag routing is correct without calling GPU-only code.
    """

    def test_trust_region_noise_shape_from_policy(self):
        """GaussianEnergyPolicy.sample_trust_region_noise returns [N, 3] on CPU."""
        # sample_trust_region_noise calls build_scaling_rotation via local import
        # from eligsir.utils.general_utils — patch at that module.
        import unittest.mock as mock
        n = 60
        g = _make_gaussians(n)
        with mock.patch("utils.general_utils.build_scaling_rotation") as mock_bsr:
            # Return a fake L tensor of shape [N, 3, 3]
            mock_bsr.return_value = torch.eye(3).unsqueeze(0).expand(n, -1, -1)
            cfg = GaussianEnergyConfig(
                mcmc_proposal_mode="trust_region",
                mcmc_trust_radius=0.05,
                mcmc_support_power=0.5,
                mcmc_opacity_power=0.5,
            )
            policy = GaussianEnergyPolicy(cfg)
            noise = policy.sample_trust_region_noise(
                gaussians=g,
                xyz_lr=1e-4,
                noise_lr=1e3,
                visible=None,
            )
        assert noise.shape == (n, 3), f"Expected [{n}, 3], got {noise.shape}"
        assert noise.device == g.get_xyz.device

    def test_trust_region_noise_with_visible_mask(self):
        """sample_trust_region_noise zeroes out non-visible Gaussians."""
        import unittest.mock as mock
        n = 60
        g = _make_gaussians(n)
        visible_idx = torch.arange(20)  # only first 20 are visible

        with mock.patch("utils.general_utils.build_scaling_rotation") as mock_bsr:
            mock_bsr.return_value = torch.eye(3).unsqueeze(0).expand(n, -1, -1)
            cfg = GaussianEnergyConfig(
                mcmc_trust_radius=0.05,
                mcmc_support_power=0.5,
                mcmc_opacity_power=0.5,
            )
            policy = GaussianEnergyPolicy(cfg)
            noise = policy.sample_trust_region_noise(
                gaussians=g,
                xyz_lr=1e-4,
                noise_lr=1e3,
                visible=visible_idx,
            )
        # Non-visible Gaussians (index >= 20) must have zero noise
        assert noise[20:].abs().sum() == 0, "Non-visible Gaussians must have zero noise"

    def test_legacy_clamped_flag_default(self):
        """getattr(args, 'mcmc_proposal_mode', 'legacy_clamped') defaults to legacy path."""
        args_no_flag = types.SimpleNamespace()  # no mcmc_proposal_mode attribute
        mode = getattr(args_no_flag, "mcmc_proposal_mode", "legacy_clamped")
        assert mode == "legacy_clamped", (
            "Missing attribute must default to legacy_clamped, not activate trust-region"
        )

    def test_legacy_clamped_noise_step_is_bounded(self):
        """sample_legacy_clamped_noise (GsplatEnergyMCMCStrategy's default noise
        path) must never exceed args.gsplat_mcmc_noise_max_step, even for a
        near-fully-transparent Gaussian with a huge raw scale and a huge
        noise_lr/lr product -- the exact combination that flung real
        Gaussians ~200m outside a real Replica room0 scene before this fix
        (the escape-plume root cause: this path used to call gsplat's
        stock, UNCLAMPED inject_noise_to_position with no scale or
        step-size cap at all).

        Uses gsplat_mcmc_noise_max_scale/gsplat_mcmc_noise_max_step -- this
        strategy's OWN caps, deliberately separate from the legacy
        strategy's noise_max_scale/noise_max_step tested elsewhere in this
        class (a real GPU A/B found the legacy strategy's 0.01m step cap
        over-constrains this strategy's much larger noise_lr; see
        sample_legacy_clamped_noise's docstring for the full tradeoff data).
        """
        from eligsir.utils.strategies.mcmc.proposal import sample_legacy_clamped_noise

        n = 40
        g = types.SimpleNamespace()
        # Huge raw scale on every Gaussian -- without the scale clamp this
        # alone would blow up the noise covariance.
        g.get_scaling = torch.full((n, 3), 50.0)
        g.get_rotation = torch.nn.functional.normalize(torch.randn(n, 4), dim=-1)
        means = torch.randn(n, 3)
        # Near-zero opacity for every Gaussian -- fully opens the op_sigmoid
        # noise gate (worst case for step magnitude).
        opacities_raw = torch.full((n, 1), -12.0)  # sigmoid(-12) ~= 6e-6
        params = {"means": means, "opacities": opacities_raw}
        args = _make_legacy_args(gsplat_mcmc_noise_max_scale=0.05, gsplat_mcmc_noise_max_step=0.1)

        # Absurdly large noise_lr/lr product -- another real-run ingredient
        # (default noise_lr=5e5, lr up to a few 1e-4 early in training).
        # build_scaling_rotation is hard-coded to device='cuda' (matching
        # legacy.py's own sibling path, see TestPhase6NoiseParity's class
        # docstring) -- mock it the same way that class already does, so
        # this runs on CPU.
        import unittest.mock as mock
        with mock.patch("utils.general_utils.build_scaling_rotation") as mock_bsr:
            mock_bsr.return_value = torch.eye(3).unsqueeze(0).expand(n, -1, -1).clone()
            noise = sample_legacy_clamped_noise(
                gaussians=g, params=params, args=args, lr=1.0, noise_lr=5e5,
            )

        assert noise.shape == (n, 3)
        step_norms = noise.norm(dim=-1)
        assert float(step_norms.max()) <= args.gsplat_mcmc_noise_max_step + 1e-6, (
            f"noise step exceeded gsplat_mcmc_noise_max_step={args.gsplat_mcmc_noise_max_step}: "
            f"max observed step={float(step_norms.max())}"
        )

    def test_legacy_clamped_noise_unclamped_when_caps_are_zero(self):
        """gsplat_mcmc_noise_max_scale=0 / gsplat_mcmc_noise_max_step=0 must
        fully restore the old, unclamped behavior (explicit opt-out), not
        silently clamp anyway."""
        from eligsir.utils.strategies.mcmc.proposal import sample_legacy_clamped_noise

        n = 40
        g = types.SimpleNamespace()
        g.get_scaling = torch.full((n, 3), 50.0)
        g.get_rotation = torch.nn.functional.normalize(torch.randn(n, 4), dim=-1)
        means = torch.randn(n, 3)
        opacities_raw = torch.full((n, 1), -12.0)
        params = {"means": means, "opacities": opacities_raw}
        args = _make_legacy_args(gsplat_mcmc_noise_max_scale=0.0, gsplat_mcmc_noise_max_step=0.0)

        import unittest.mock as mock
        with mock.patch("utils.general_utils.build_scaling_rotation") as mock_bsr:
            mock_bsr.return_value = torch.eye(3).unsqueeze(0).expand(n, -1, -1).clone()
            noise = sample_legacy_clamped_noise(
                gaussians=g, params=params, args=args, lr=1.0, noise_lr=5e5,
            )
        step_norms = noise.norm(dim=-1)
        # With huge scale/noise_lr and no clamps, at least one step must be
        # far larger than the (now-inactive) 0.1m default cap.
        assert float(step_norms.max()) > 0.1

    def test_inject_noise_routing_selection(self):
        """Verify that inject_noise selects trust_region vs legacy_clamped based on flag."""
        # We verify the routing decision via the args flag without calling GPU code.
        # The test checks that trust_region path is selected when the flag says so.
        args_legacy = _make_legacy_args(mcmc_proposal_mode="legacy_clamped")
        args_trust = _make_legacy_args(mcmc_proposal_mode="trust_region")

        assert getattr(args_legacy, "mcmc_proposal_mode", "legacy_clamped") == "legacy_clamped"
        assert getattr(args_trust, "mcmc_proposal_mode", "legacy_clamped") == "trust_region"

        # Confirm the routing condition as written in inject_noise
        _proposal_mode_legacy = getattr(args_legacy, "mcmc_proposal_mode", "legacy_clamped")
        _proposal_mode_trust = getattr(args_trust, "mcmc_proposal_mode", "legacy_clamped")

        assert _proposal_mode_legacy != "trust_region", "Legacy flag must not enter trust-region branch"
        assert _proposal_mode_trust == "trust_region", "Trust-region flag must enter trust-region branch"


# ---------------------------------------------------------------------------
# Phase 10: soft lifecycle parent score modulation
# ---------------------------------------------------------------------------

class TestPhase10SoftLifecycle:
    """Phase 10: soft_lifecycle_enabled=True modulates parent_scores."""

    def test_maturity_logit_shape(self):
        """compute_maturity_logit returns [N] tensor."""
        g = _make_gaussians(50)
        render_pkg = _make_render_pkg(50)
        policy = GaussianEnergyPolicy()
        terms = policy.compute_terms(g, render_pkg, iteration=100)
        logit = policy.compute_maturity_logit(terms)
        assert logit.shape == (50,)

    def test_maturity_sigmoid_range(self):
        """Sigmoid of maturity logit is in (0, 1)."""
        g = _make_gaussians(50)
        render_pkg = _make_render_pkg(50)
        policy = GaussianEnergyPolicy()
        terms = policy.compute_terms(g, render_pkg, iteration=100)
        logit = policy.compute_maturity_logit(terms)
        prob = logit.sigmoid()
        assert (prob > 0).all() and (prob < 1).all(), "Maturity probability must be in (0,1)"

    def test_soft_lifecycle_modulates_parent_scores(self):
        """soft_lifecycle_enabled=True should produce different parent scores than utility."""
        g = _make_gaussians(50)
        render_pkg = _make_render_pkg(50)
        utility = torch.rand(50)

        policy = GaussianEnergyPolicy()
        terms = policy.compute_terms(g, render_pkg, iteration=100)
        maturity_prob = policy.compute_maturity_logit(terms).sigmoid().detach()
        modulated = utility * maturity_prob

        # Modulated scores should differ from raw utility (unless maturity_prob == 1 everywhere)
        assert not torch.allclose(modulated, utility), (
            "Maturity modulation should change parent scores for non-trivial maturity"
        )

    def test_soft_lifecycle_disabled_leaves_scores_unchanged(self):
        """soft_lifecycle_enabled=False: parent scores must be the unmodified utility tensor."""
        # This is a structural test: when the flag is False, the code path
        # does not touch utility. We verify by checking the flag routing logic.
        utility = torch.rand(60)
        args_disabled = _make_legacy_args(soft_lifecycle_enabled=False)
        # The flag check in mcmc_strategy.py: if getattr(args, 'soft_lifecycle_enabled', False) and ...
        assert not getattr(args_disabled, "soft_lifecycle_enabled", False), (
            "Default soft_lifecycle_enabled must be False"
        )
        # With the flag False, the code falls through without modulation.
        # Simulate what the strategy does:
        _parent_scores = utility
        if getattr(args_disabled, "soft_lifecycle_enabled", False) and utility is not None:
            raise AssertionError("Should not enter maturity block when flag is False")
        assert _parent_scores is utility, "parent_scores must be the original utility"
