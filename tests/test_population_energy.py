"""Tests for population energy (Phase 3) and energy-based death (Phase 4)."""

from __future__ import annotations

import types

import pytest
import torch

from eligsir.utils.compiled_kernels import configure_torch_compile
from eligsir.utils.gaussian_energy_policy import GaussianEnergyConfig, GaussianEnergyPolicy


@pytest.fixture(scope="session", autouse=True)
def _register_kernels():
    """Register compiled kernels in eager mode."""
    import types as _t
    _args = _t.SimpleNamespace(
        compile_mode="off",
        compile_energy=False,
        compile_utility=False,
        compile_sh=False,
        compile_dynamic=False,
        compile_after_iter=0,
        compile_growth_margin=0,
    )
    configure_torch_compile(_args)


def _make_gaussians(n: int = 50, seed: int = 0) -> types.SimpleNamespace:
    # Seeded generator: these tests assert mathematical invariants (e.g.
    # "geometry support must not raise energy") that should hold for any
    # random draw, but leaving this on torch's global (unseeded) RNG made
    # the result depend on whatever ran earlier in a full-suite pytest
    # session -- passing in isolation, occasionally failing at full-suite
    # random states. Fixed seed makes the test deterministic and reproducible.
    gen = torch.Generator().manual_seed(seed)
    g = types.SimpleNamespace()
    g.get_xyz = torch.randn(n, 3, generator=gen)
    g.get_xyz.grad = torch.randn(n, 3, generator=gen) * 0.01
    raw_op = torch.randn(n, 1, generator=gen) * 2.0
    g._opacity = raw_op
    g._opacity.grad = torch.randn(n, 1, generator=gen) * 0.001
    g.get_opacity = torch.sigmoid(raw_op)
    raw_scale = torch.randn(n, 3, generator=gen) * 0.5
    g._scaling = raw_scale
    g._scaling.grad = torch.randn(n, 3, generator=gen) * 0.001
    g.get_scaling = torch.exp(raw_scale.clamp(-4, 4))
    g.get_rotation = torch.nn.functional.normalize(torch.randn(n, 4, generator=gen), dim=-1)
    g.visibility_ema = torch.rand(n, 1, generator=gen)
    g.birth_frame = torch.zeros(n, dtype=torch.int32)
    g.provisional = torch.zeros(n, dtype=torch.bool)
    g.depth_conflict_count = torch.zeros(n, dtype=torch.int16)
    g.params = {
        "means": g.get_xyz, "opacities": g._opacity, "scales": g._scaling,
    }
    g.params["means"].grad = g.get_xyz.grad
    g.params["opacities"].grad = g._opacity.grad
    g.params["scales"].grad = g._scaling.grad
    return g


class TestPopulationEnergy:
    """Phase 3: population energy tests."""

    def test_compute_population_energy_shape(self):
        """All output tensors have shape [N]."""
        g = _make_gaussians(50)
        render_pkg = {"visibility_filter": torch.rand(50) > 0.3}
        cfg = GaussianEnergyConfig(energy_mode="population_energy")
        policy = GaussianEnergyPolicy(cfg)
        terms = policy.compute_terms(g, render_pkg, iteration=100)
        result = policy.compute_population_energy(terms)
        assert result.total.shape == (50,)
        assert result.E_opacity.shape == (50,)
        assert result.E_support.shape == (50,)

    def test_population_energy_nonneg(self):
        """Penalty energy is non-negative when the protective geometry term is off.

        Non-negativity only holds for the penalty (death-signal) components.
        ``E_geometry_support`` is intentionally *subtractive* (it protects
        geometry-supported Gaussians) and can drive the total negative; that is
        covered by ``test_geometry_support_is_protective``.
        """
        g = _make_gaussians(50)
        render_pkg = {"visibility_filter": torch.ones(50, dtype=torch.bool)}
        policy = GaussianEnergyPolicy(GaussianEnergyConfig(energy_w_geometry_support=0.0))
        terms = policy.compute_terms(g, render_pkg, iteration=100)
        result = policy.compute_population_energy(terms)
        assert (result.total >= 0).all(), "Penalty energy must be non-negative without geometry support"

    def test_geometry_support_is_protective(self):
        """Geometry support lowers death energy (protect-by-subtraction)."""
        g = _make_gaussians(50)
        g.anchor_confidence = torch.ones(50)  # fully trusted geometry
        render_pkg = {"visibility_filter": torch.ones(50, dtype=torch.bool)}

        base = GaussianEnergyPolicy(GaussianEnergyConfig(energy_w_geometry_support=0.0))
        prot = GaussianEnergyPolicy(GaussianEnergyConfig(energy_w_geometry_support=1.0))
        t_base = base.compute_terms(g, render_pkg, iteration=100)
        t_prot = prot.compute_terms(g, render_pkg, iteration=100)
        total_base = base.compute_population_energy(t_base).total
        total_prot = prot.compute_population_energy(t_prot).total

        # Support must reduce energy everywhere, and with full confidence +
        # unit weight it must be able to push the total below zero.
        assert (total_prot <= total_base + 1e-6).all(), "geometry support must not raise energy"
        assert (total_prot < total_base).any(), "geometry support must lower energy somewhere"
        assert (total_prot < 0).any(), "full-confidence support should be able to go negative"

    def test_overflowing_scale_does_not_poison_whole_population(self):
        """Regression for the NaN-poisoning bug (found live 2026-08-13, real
        300k-iteration run): a single Gaussian's scale overflowing to Inf
        (plausible after many compounding MCMC relocation cycles on the same
        surviving Gaussian) previously made `_norm`'s `t.max()` become Inf,
        driving every OTHER Gaussian's normalized term to 0 and the
        offending entry itself to Inf/Inf=NaN. `total` then had one NaN
        entry, and `torch.quantile(total, ...)` -- used by both
        `energy_dead_mode=energy_quantile` and `energy_sigmoid` -- returns
        NaN for the WHOLE population once ANY entry is NaN. That silently
        and permanently disabled dead-mask selection for the rest of a real
        training run (relocated=0 for ~249k of a planned 300k iterations).

        `total` must stay fully finite even with one Inf-scale Gaussian, and
        the dead mask must still be computable (non-NaN quantile) and still
        flag the genuinely weak (low-opacity) Gaussians as dead.
        """
        n, n_weak = 50, 5
        g = _make_gaussians(n)
        with torch.no_grad():
            raw_op = g.params["opacities"]
            raw_op[:n_weak] = -6.0  # sigmoid(-6) ~ 0.0025: clearly weak
            raw_op[n_weak:] = 3.0   # sigmoid(3) ~ 0.9526: clearly healthy
            g.get_opacity = torch.sigmoid(raw_op)
            # Simulate one Gaussian's scale overflowing after many
            # relocation cycles -- bypass the fixture's exp(clamp(-4,4))
            # scaling so this test exercises the real failure mode.
            scaling = g.get_scaling.clone()
            scaling[-1, 0] = float("inf")
            g.get_scaling = scaling

        render_pkg = {"visibility_filter": torch.ones(n, dtype=torch.bool)}
        for mode in ("energy_quantile", "energy_sigmoid"):
            cfg = GaussianEnergyConfig(energy_dead_mode=mode, energy_dead_quantile=0.10)
            policy = GaussianEnergyPolicy(cfg)
            terms = policy.compute_terms(g, render_pkg, iteration=200)
            energy = policy.compute_population_energy(terms)
            assert torch.isfinite(energy.total).all(), f"{mode}: total must stay finite despite an Inf-scale Gaussian"

            dead = policy.compute_dead_mask_from_energy(energy, g)
            assert dead.sum() > 0, f"{mode}: dead mask must not be silently empty (the NaN-poisoning symptom)"
            dead_idx = dead.nonzero(as_tuple=True)[0]
            assert (dead_idx < n_weak).all(), f"{mode}: only the genuinely weak tail should be flagged dead"

    def test_tensorboard_dict_keys(self):
        """to_tensorboard_dict returns expected keys."""
        g = _make_gaussians(30)
        render_pkg = {"visibility_filter": torch.ones(30, dtype=torch.bool)}
        policy = GaussianEnergyPolicy()
        terms = policy.compute_terms(g, render_pkg, iteration=50)
        result = policy.compute_population_energy(terms)
        tb = result.to_tensorboard_dict()
        assert "energy/population_mean" in tb
        assert "energy/population_p95" in tb
        assert "energy/E_opacity_mean" in tb

    def test_legacy_dead_mask_result_type(self):
        """compute_legacy_dead_mask returns DeadMaskResult with stats."""
        from eligsir.utils.gaussian_energy_policy import DeadMaskResult
        g = _make_gaussians(50)
        policy = GaussianEnergyPolicy()
        result = policy.compute_legacy_dead_mask(g, utility=None)
        assert isinstance(result, DeadMaskResult)
        assert "very_low_opacity" in result.stats
        assert "total" in result.stats
        assert result.mask.shape == (50,)
        assert result.stats["total"] == int(result.mask.sum())


class TestEnergyBasedDeath:
    """Phase 4: energy-based dead mask tests."""

    def test_energy_quantile_death(self):
        """energy_quantile mode returns a bool mask of correct shape."""
        g = _make_gaussians(100)
        render_pkg = {"visibility_filter": torch.ones(100, dtype=torch.bool)}
        cfg = GaussianEnergyConfig(energy_dead_mode="energy_quantile")
        policy = GaussianEnergyPolicy(cfg)
        terms = policy.compute_terms(g, render_pkg, iteration=200)
        energy = policy.compute_population_energy(terms)
        dead = policy.compute_dead_mask_from_energy(energy, g)
        assert dead.shape == (100,)
        assert dead.dtype == torch.bool

    def _make_gaussians_with_weak_tail(self, n=200, n_weak=20, seed=0):
        """A population with a clear, deliberately weak (low-opacity) tail and
        an otherwise healthy body -- used to assert energy_quantile/energy_sigmoid
        actually flag the WEAK Gaussians as dead, not the healthy majority.
        """
        g = _make_gaussians(n, seed=seed)
        with torch.no_grad():
            raw_op = g.params["opacities"]
            raw_op[:n_weak] = -6.0  # sigmoid(-6) ~ 0.0025: clearly weak
            raw_op[n_weak:] = 3.0   # sigmoid(3) ~ 0.9526: clearly healthy
            g.get_opacity = torch.sigmoid(raw_op)
        return g

    def test_energy_quantile_death_selects_weak_gaussians_not_healthy(self):
        """Regression for the quantile-direction inversion (root-caused
        2026-08-12): `total` is a DEATH-energy score (higher = more deserving
        of death -- see test_geometry_support_is_protective), so the dead
        quantile must select the TOP `energy_dead_quantile` fraction. The
        previous code selected `total < quantile(energy_dead_quantile)` --
        the HEALTHIEST tail -- which almost never also has low opacity, so
        the population-energy dead path silently produced an empty mask on
        any real population (observed: relocated=0 for 30k+ iterations on a
        real 854830-Gaussian checkpoint under --energy_dead_mode
        energy_quantile).
        """
        n, n_weak = 200, 20
        g = self._make_gaussians_with_weak_tail(n=n, n_weak=n_weak)
        render_pkg = {"visibility_filter": torch.ones(n, dtype=torch.bool)}
        cfg = GaussianEnergyConfig(energy_dead_mode="energy_quantile", energy_dead_quantile=0.05)
        policy = GaussianEnergyPolicy(cfg)
        terms = policy.compute_terms(g, render_pkg, iteration=200)
        energy = policy.compute_population_energy(terms)
        dead = policy.compute_dead_mask_from_energy(energy, g)

        assert dead.sum() > 0, "quantile mode must flag SOME Gaussians dead on a population with a weak tail"
        dead_idx = dead.nonzero(as_tuple=True)[0]
        assert (dead_idx < n_weak).all(), "every flagged-dead Gaussian must come from the weak tail, not the healthy body"

    def test_energy_sigmoid_death_selects_weak_gaussians_not_healthy(self):
        """Same direction regression as above, for energy_sigmoid mode: dead
        probability must rise with `total` (higher death-energy), not fall
        with it.
        """
        n, n_weak = 200, 20
        g = self._make_gaussians_with_weak_tail(n=n, n_weak=n_weak)
        render_pkg = {"visibility_filter": torch.ones(n, dtype=torch.bool)}
        cfg = GaussianEnergyConfig(energy_dead_mode="energy_sigmoid", energy_dead_temperature=10.0)
        policy = GaussianEnergyPolicy(cfg)
        terms = policy.compute_terms(g, render_pkg, iteration=200)
        energy = policy.compute_population_energy(terms)
        dead = policy.compute_dead_mask_from_energy(energy, g)

        assert dead.sum() > 0, "sigmoid mode must flag SOME Gaussians dead on a population with a weak tail"
        dead_idx = dead.nonzero(as_tuple=True)[0]
        assert (dead_idx < n_weak).all(), "every flagged-dead Gaussian must come from the weak tail, not the healthy body"

    def test_legacy_clauses_matches_default(self):
        """energy_dead_mode=legacy_clauses matches GaussianEnergyPolicy.compute_dead_mask."""
        g = _make_gaussians(80)
        cfg = GaussianEnergyConfig(energy_dead_mode="legacy_clauses", use_utility_quantile=False)
        policy = GaussianEnergyPolicy(cfg)
        render_pkg = {"visibility_filter": torch.ones(80, dtype=torch.bool)}
        terms = policy.compute_terms(g, render_pkg, iteration=100)
        energy = policy.compute_population_energy(terms)
        dead_from_energy = policy.compute_dead_mask_from_energy(energy, g)
        dead_direct = policy.compute_dead_mask(g, utility=None)
        assert torch.equal(dead_from_energy, dead_direct)

    def test_compute_gaussian_utility_population_energy_shape(self):
        """Phase 3: compute_gaussian_utility routes to population energy and returns [N]."""
        import types as _t
        from eligsir.utils.energy_mcmc import compute_gaussian_utility

        g = _make_gaussians(60)
        render_pkg = {"visibility_filter": torch.rand(60) > 0.3}
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
        result = compute_gaussian_utility(g, render_pkg, iteration=100, args=pop_args)
        assert result.shape == (60,), f"Expected [60], got {result.shape}"
        assert torch.isfinite(result).all(), "All utility values must be finite"
