"""Merge A (CapacityPolicy) tests: streaming_target_capacity routing + parity.

These tests prove that the new soft TARGET-N knob (``target_capacity`` /
``streaming_target_capacity``) is default-behaviour-preserving when unset (0)
and that, when set, it scales the count-loss / growth-deficit N_target curve to
the requested target while leaving cap_max as the hard ceiling.
"""

from __future__ import annotations

import math

import torch

import pytest

from eligsir.utils.energy_mcmc import compute_effective_count_loss
from eligsir.utils.mcmc_schedule import MCMCScheduleConfig, get_mcmc_schedule


@pytest.fixture(autouse=True)
def _register_eager_kernels():
    # Outside a training run the compiled-kernel REGISTRY has no eager fn
    # registered, and its torch.compile attempt fails on a CPU-only box. Register
    # the eager implementation directly so the count-loss kernel runs natively.
    from eligsir.utils import compiled_kernels as ck

    ck.REGISTRY.register(
        "effective_count_loss_core", ck._effective_count_loss_core, enabled=False
    )
    yield


# --- analytic reference for the count-loss N_target curve ------------------

def _expected_n_target(iteration, max_iterations, q_start, q_end, tau_N, target_scale):
    u = min(iteration / float(max(1, max_iterations)), 1.0)
    tau = max(tau_N, 1e-6)
    denom = 1.0 - math.exp(-1.0 / tau)
    alpha_q = (1.0 - math.exp(-u / tau)) / denom
    q = q_start + (q_end - q_start) * alpha_q
    return target_scale * q


def _call_count_loss(target_capacity, cap_max=500000, target_splat_end=150000):
    opac = torch.zeros((1000, 1), dtype=torch.float32)
    _, _, n_target = compute_effective_count_loss(
        opac,
        iteration=15000,
        cap_max=cap_max,
        max_iterations=30000,
        target_splat_end=target_splat_end,
        target_capacity=target_capacity,
    )
    return n_target


def test_count_loss_default_capacity_is_parity():
    # target_capacity=0 (default) -> target_scale falls through to
    # target_splat_end (150000), byte-identical to pre-Merge-A behaviour.
    got = _call_count_loss(target_capacity=0)
    expected = _expected_n_target(15000, 30000, 0.05, 0.85, 0.45, 150000)
    assert math.isclose(got, expected, rel_tol=0, abs_tol=1e-6)


def test_count_loss_capacity_scales_target_curve():
    got = _call_count_loss(target_capacity=35000)
    expected = _expected_n_target(15000, 30000, 0.05, 0.85, 0.45, 35000)
    assert math.isclose(got, expected, rel_tol=0, abs_tol=1e-6)
    # And it must differ from the cap_max / splat_end based value.
    assert got < _call_count_loss(target_capacity=0)


def test_count_loss_capacity_overrides_splat_end():
    # target_capacity wins over target_splat_end when both are set.
    got = _call_count_loss(target_capacity=40000, target_splat_end=150000)
    expected = _expected_n_target(15000, 30000, 0.05, 0.85, 0.45, 40000)
    assert math.isclose(got, expected, rel_tol=0, abs_tol=1e-6)


# --- schedule (growth deficit) routing -------------------------------------

def test_schedule_default_capacity_is_parity():
    base = MCMCScheduleConfig(use_target_deficit=True, target_splat_end=150000)
    withzero = MCMCScheduleConfig(
        use_target_deficit=True, target_splat_end=150000, target_capacity=0
    )
    a = get_mcmc_schedule(8000, current_n=20000, cap_max=500000, cfg=base)
    b = get_mcmc_schedule(8000, current_n=20000, cap_max=500000, cfg=withzero)
    assert a == b


def test_schedule_capacity_overrides_splat_end_in_deficit():
    # With a small target_capacity the deficit vanishes once current_n exceeds
    # the capacity-scaled N_target, so growth_factor collapses to 1.0 -> no
    # growth. With the big target_splat_end the deficit is large -> growth.
    cfg_cap = MCMCScheduleConfig(
        use_target_deficit=True, target_splat_end=150000, target_capacity=10000
    )
    cfg_big = MCMCScheduleConfig(
        use_target_deficit=True, target_splat_end=150000, target_capacity=0
    )
    s_cap = get_mcmc_schedule(8000, current_n=40000, cap_max=500000, cfg=cfg_cap)
    s_big = get_mcmc_schedule(8000, current_n=40000, cap_max=500000, cfg=cfg_big)
    assert s_cap["growth_factor"] == 1.0
    assert s_big["growth_factor"] > 1.0


def test_target_deficit_can_be_deferred_until_source_completion():
    """Live ingress must not pay the post-mapping growth policy's cost."""
    cfg = MCMCScheduleConfig(
        growth_factor_start=1.05,
        use_target_deficit=True,
        target_capacity=40_000,
    )

    during_ingress = get_mcmc_schedule(
        100,
        current_n=1_000,
        cap_max=2_000,
        cfg=cfg,
        grow_due=True,
        relocate_due=False,
        target_deficit_enabled=False,
    )
    after_source_complete = get_mcmc_schedule(
        100,
        current_n=1_000,
        cap_max=2_000,
        cfg=cfg,
        grow_due=True,
        relocate_due=False,
        target_deficit_enabled=True,
    )

    assert during_ingress["growth_factor"] < after_source_complete["growth_factor"]
    assert after_source_complete["allow_growth"]


# --- PRIMARY GOAL: bound N in online mode without crushing the schedule ----

def _online_n_target(target_capacity, anneal_horizon, iteration):
    """N_target as the live count-loss would compute it in online mode."""
    from eligsir.core.stop_policy import apply_online_schedule_overrides
    from types import SimpleNamespace

    opt = SimpleNamespace(
        iterations=30_000,
        densify_until_iter=25_000,
        mcmc_noise_stop_iter=30_000,
        mcmc_stop_growth_iter=12_000,
        position_lr_max_steps=30_000,
    )
    apply_online_schedule_overrides(
        opt, anneal_horizon=anneal_horizon if anneal_horizon > 0 else None
    )
    opac = torch.zeros((1000, 1), dtype=torch.float32)
    _, _, n_target = compute_effective_count_loss(
        opac,
        iteration=iteration,
        cap_max=500000,
        max_iterations=opt._anneal_clock.horizon,
        target_splat_end=150000,
        target_capacity=target_capacity,
    )
    return n_target


def test_online_default_pins_target_low_documents_pathology():
    # With the unbounded loop bound and NO anneal horizon, u~=0 -> q~=q_start
    # -> N_target ~= 0.05 * target. This is the pre-existing online pathology
    # (target never ramps); setting target_capacity alone does NOT bound to it.
    n = _online_n_target(target_capacity=35000, anneal_horizon=0, iteration=20000)
    assert n < 0.06 * 35000  # pinned near q_start, not near 35000


def test_online_finite_horizon_ramps_target_to_capacity_band():
    # PRIMARY GOAL: a finite anneal horizon lets target_capacity actually bound
    # N. Late in the horizon, N_target reaches q_end * capacity (~0.85*35000),
    # inside the 25-40k band.
    n = _online_n_target(target_capacity=35000, anneal_horizon=20000, iteration=20000)
    assert 25000 <= n <= 40000


def test_online_finite_horizon_leaves_dead_opacity_ramp_frozen():
    # "Bound WITHOUT over-prune": the dead-opacity ramp keys off the schedule's
    # u_reloc, which is still on the unbounded horizon (stop_reloc_iter=2^31),
    # so it stays at its start value -> no horizon-driven over-prune.
    from eligsir.utils.mcmc_schedule import MCMCScheduleConfig, get_mcmc_schedule
    from eligsir.core.stop_policy import ONLINE_UNBOUNDED_ITERATIONS

    cfg = MCMCScheduleConfig(
        stop_growth_iter=ONLINE_UNBOUNDED_ITERATIONS,
        stop_reloc_iter=ONLINE_UNBOUNDED_ITERATIONS,
        target_capacity=35000,
    )
    s = get_mcmc_schedule(20000, current_n=35000, cap_max=500000, cfg=cfg)
    # dead-opacity threshold pinned at its start value (ramp frozen by u_reloc~=0)
    assert math.isclose(s["dead_opacity_threshold"], cfg.dead_opacity_start, abs_tol=1e-6)


def test_schedule_rho_still_reads_cap_max_not_capacity():
    # rho (hard-ceiling pressure) must keep using cap_max, unaffected by
    # target_capacity: the reported rho equals current_n / cap_max.
    cfg = MCMCScheduleConfig(use_target_deficit=True, target_capacity=10000)
    s = get_mcmc_schedule(8000, current_n=50000, cap_max=500000, cfg=cfg)
    assert math.isclose(s["rho"], 50000 / 500000, abs_tol=1e-9)
