"""Tests for the online-mode schedule-horizon helpers."""

from __future__ import annotations

from types import SimpleNamespace

import eligsir.core as core
import eligsir.core.stop_policy as stop_policy_mod
import eligsir.utils.stop_policy as compat_mod
from eligsir.core.stop_policy import (
    ONLINE_UNBOUNDED_ITERATIONS,
    apply_online_schedule_overrides,
)


def test_utils_stop_policy_reexports_core_helpers():
    assert compat_mod.apply_online_schedule_overrides is apply_online_schedule_overrides
    assert compat_mod.ONLINE_UNBOUNDED_ITERATIONS == ONLINE_UNBOUNDED_ITERATIONS


def test_stop_policy_classes_stay_deleted():
    """Regression guard for issue #1075.

    ``OfflineStopPolicy``/``OnlineStopPolicy`` were unwired dead code that
    predated the 2026-07-04 unified rolling-budget termination rule and could
    not express it -- they CONSUMED a termination verdict (``input_done: bool``)
    rather than computing one, so nothing could hand them ``_last_ingest_iter``.
    Deleted 2026-08-23. The two real termination decisions live in
    ``eligsir/io/sources/ros2_live.py`` (labeled ``stream_end_reason``) and
    ``train_streaming.py`` (the rolling-budget rule); see
    ``docs/architecture/2026-08-23-issue1075-stop-policy-reconcile.md`` before
    reintroducing any stop-policy abstraction here.
    """
    for name in ("OfflineStopPolicy", "OnlineStopPolicy"):
        assert not hasattr(stop_policy_mod, name), (
            f"{name} was reintroduced into eligsir/core/stop_policy.py -- "
            "read docs/architecture/2026-08-23-issue1075-stop-policy-reconcile.md first"
        )
        assert not hasattr(compat_mod, name)
        assert not hasattr(core, name)
        assert name not in stop_policy_mod.__dict__
        assert name not in compat_mod.__all__
        assert name not in core.__all__


def test_apply_online_schedule_overrides_neutralizes_global_stops():
    opt = SimpleNamespace(
        iterations=30_000,
        densify_until_iter=25_000,
        mcmc_noise_stop_iter=30_000,
        mcmc_stop_growth_iter=12_000,
        position_lr_max_steps=30_000,
    )
    apply_online_schedule_overrides(opt)
    assert opt.iterations == ONLINE_UNBOUNDED_ITERATIONS
    assert opt.densify_until_iter == ONLINE_UNBOUNDED_ITERATIONS
    assert opt.mcmc_noise_stop_iter == ONLINE_UNBOUNDED_ITERATIONS
    assert opt.mcmc_stop_growth_iter == ONLINE_UNBOUNDED_ITERATIONS
    assert opt.position_lr_max_steps == ONLINE_UNBOUNDED_ITERATIONS


def test_noise_stop_horizon_defaults_to_unbounded():
    """Unset noise_stop_horizon preserves today's unbounded-noise behaviour."""
    opt = SimpleNamespace(
        iterations=30_000,
        densify_until_iter=25_000,
        mcmc_noise_stop_iter=30_000,
        mcmc_stop_growth_iter=12_000,
        position_lr_max_steps=30_000,
    )
    apply_online_schedule_overrides(opt)
    assert opt.mcmc_noise_stop_iter == ONLINE_UNBOUNDED_ITERATIONS


def test_noise_stop_horizon_is_finite_and_stops_growth_too_but_not_densify():
    """A finite noise_stop_horizon is a pure-refinement switch (2026-07-06b):
    noise AND growth stop together at the same point, but densify/dead-opacity
    (driven by schedule_horizon/opt.iterations) stay unbounded, avoiding the
    schedule_horizon over-prune risk. Live testing found noise-stop ALONE does
    not reduce floater accumulation (new Gaussians kept entering from
    newly-ingested frames) -- coupling growth-stop to the same horizon closes
    that gap."""
    opt = SimpleNamespace(
        iterations=30_000,
        densify_until_iter=25_000,
        mcmc_noise_stop_iter=30_000,
        mcmc_stop_growth_iter=12_000,
        position_lr_max_steps=30_000,
    )
    apply_online_schedule_overrides(opt, noise_stop_horizon=40_000)
    assert opt.mcmc_noise_stop_iter == 40_000
    assert opt.mcmc_stop_growth_iter == 40_000
    assert opt._streaming_insertion_stop_iter == 40_000
    assert opt.iterations == ONLINE_UNBOUNDED_ITERATIONS
    assert opt.densify_until_iter == ONLINE_UNBOUNDED_ITERATIONS
    assert opt.position_lr_max_steps == ONLINE_UNBOUNDED_ITERATIONS


def test_noise_stop_horizon_independent_of_schedule_horizon_value():
    """noise_stop_horizon (now the pure-refinement horizon: noise + growth +
    streaming insertion) does not have to match schedule_horizon (densify/
    dead-opacity-ramp) -- they're genuinely decoupled clocks, not one deriving
    from the other."""
    opt = SimpleNamespace(
        iterations=30_000,
        densify_until_iter=25_000,
        mcmc_noise_stop_iter=30_000,
        mcmc_stop_growth_iter=12_000,
        position_lr_max_steps=30_000,
    )
    sched = apply_online_schedule_overrides(
        opt, schedule_horizon=20_000, noise_stop_horizon=5_000
    )
    assert sched == 20_000
    assert opt.densify_until_iter == 20_000
    assert opt.mcmc_stop_growth_iter == 5_000
    assert opt.mcmc_noise_stop_iter == 5_000
    assert opt._streaming_insertion_stop_iter == 5_000


def test_noise_stop_horizon_unset_preserves_legacy_unbounded_behaviour():
    """noise_stop_horizon=None (default/unset) must leave growth-stop coupled
    to `sched` exactly as before this feature existed -- no behaviour change
    for anyone not opting in."""
    opt = SimpleNamespace(
        iterations=30_000,
        densify_until_iter=25_000,
        mcmc_noise_stop_iter=30_000,
        mcmc_stop_growth_iter=12_000,
        position_lr_max_steps=30_000,
    )
    apply_online_schedule_overrides(opt)
    assert opt.mcmc_noise_stop_iter == ONLINE_UNBOUNDED_ITERATIONS
    assert opt.mcmc_stop_growth_iter == ONLINE_UNBOUNDED_ITERATIONS
    assert opt._streaming_insertion_stop_iter is None


# --- growth_stop_horizon, decoupled from noise_stop_horizon (2026-07-18) ----
#
# Real-GPU testing on floor3 found coupling growth-stop to the SAME (early,
# mapping-end-ish) horizon as noise-stop starves the model of capacity: MCMC
# growth is an ongoing densification mechanism independent of new-frame
# arrival, so it typically needs to keep running far longer than noise/
# insertion should. growth_stop_horizon lets a caller set that independently.

def test_growth_stop_horizon_independent_of_noise_stop_horizon():
    opt = SimpleNamespace(
        iterations=30_000,
        densify_until_iter=25_000,
        mcmc_noise_stop_iter=30_000,
        mcmc_stop_growth_iter=12_000,
        position_lr_max_steps=30_000,
    )
    apply_online_schedule_overrides(
        opt, noise_stop_horizon=8_000, growth_stop_horizon=55_000,
    )
    assert opt.mcmc_noise_stop_iter == 8_000
    assert opt.mcmc_stop_growth_iter == 55_000
    assert opt._streaming_insertion_stop_iter == 55_000


def test_growth_stop_horizon_unset_falls_back_to_noise_stop_horizon():
    """Preserves the prior coupled behaviour exactly when growth_stop_horizon
    isn't passed -- no behaviour change for any caller not yet opting in."""
    opt = SimpleNamespace(
        iterations=30_000,
        densify_until_iter=25_000,
        mcmc_noise_stop_iter=30_000,
        mcmc_stop_growth_iter=12_000,
        position_lr_max_steps=30_000,
    )
    apply_online_schedule_overrides(opt, noise_stop_horizon=8_000)
    assert opt.mcmc_noise_stop_iter == 8_000
    assert opt.mcmc_stop_growth_iter == 8_000
    assert opt._streaming_insertion_stop_iter == 8_000


def test_growth_stop_horizon_alone_without_noise_stop_horizon():
    """growth_stop_horizon can be set even when noise_stop_horizon is unset --
    noise stays unbounded (coupled to sched) while growth stops independently."""
    opt = SimpleNamespace(
        iterations=30_000,
        densify_until_iter=25_000,
        mcmc_noise_stop_iter=30_000,
        mcmc_stop_growth_iter=12_000,
        position_lr_max_steps=30_000,
    )
    sched = apply_online_schedule_overrides(opt, growth_stop_horizon=55_000)
    assert opt.mcmc_noise_stop_iter == sched  # unbounded, coupled to sched (no noise horizon)
    assert opt.mcmc_stop_growth_iter == 55_000
    assert opt._streaming_insertion_stop_iter == 55_000


def test_growth_stop_horizon_zero_or_none_is_neutral():
    for val in (0, None, -1):
        opt = SimpleNamespace(
            iterations=30_000,
            densify_until_iter=25_000,
            mcmc_noise_stop_iter=30_000,
            mcmc_stop_growth_iter=12_000,
            position_lr_max_steps=30_000,
        )
        apply_online_schedule_overrides(opt, noise_stop_horizon=8_000, growth_stop_horizon=val)
        assert opt.mcmc_stop_growth_iter == 8_000


# --- growth-schedule normalization horizon, decoupled from the hard stop ----
# (2026-07-18): mcmc_stop_growth_iter conflated the HARD growth-stop gate
# (utils/strategies/mcmc_strategy.py's refine_stop check) with the growth-RATE
# schedule's normalization denominator (utils/mcmc_schedule.py's u_growth).
# Real-GPU testing found a near-term hard-stop fed into the schedule
# denominator throttles growth RATE for the whole run, not just after the
# stop point (floor3: final N ~30k instead of a healthy ~220k). The schedule
# denominator must stay pinned to the unbounded loop horizon regardless of
# the hard-stop value.

def test_mcmc_growth_schedule_horizon_stays_unbounded_regardless_of_hard_stop():
    opt = SimpleNamespace(
        iterations=30_000,
        densify_until_iter=25_000,
        mcmc_noise_stop_iter=30_000,
        mcmc_stop_growth_iter=12_000,
        position_lr_max_steps=30_000,
    )
    apply_online_schedule_overrides(opt, growth_stop_horizon=55_000)
    # Hard-stop gate reflects the requested near-term value...
    assert opt.mcmc_stop_growth_iter == 55_000
    # ...but the schedule-rate normalization denominator stays at the
    # unbounded loop horizon, unaffected by growth_stop_horizon.
    assert opt._mcmc_growth_schedule_horizon == ONLINE_UNBOUNDED_ITERATIONS


def test_mcmc_growth_schedule_horizon_matches_loop_horizon_even_when_finite():
    opt = SimpleNamespace(
        iterations=30_000,
        densify_until_iter=25_000,
        mcmc_noise_stop_iter=30_000,
        mcmc_stop_growth_iter=12_000,
        position_lr_max_steps=30_000,
    )
    apply_online_schedule_overrides(
        opt, horizon=100_000, growth_stop_horizon=8_000, noise_stop_horizon=8_000,
    )
    assert opt.mcmc_stop_growth_iter == 8_000
    assert opt._mcmc_growth_schedule_horizon == 100_000
