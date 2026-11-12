"""Merge B (AnnealClock) tests: progress-parity + default-neutral wiring."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from eligsir.core.stop_policy import (
    ONLINE_UNBOUNDED_ITERATIONS,
    AnnealClock,
    apply_online_schedule_overrides,
    should_apply_adaptive_schedule_horizons,
)
from eligsir.utils.mcmc_schedule import progress as schedule_progress


def _count_loss_u(iteration, max_iterations):
    # The verbatim inline clamp used by compute_effective_count_loss.
    return min(iteration / float(max(1, max_iterations)), 1.0)


@pytest.mark.parametrize("horizon", [1, 30000, 12000, ONLINE_UNBOUNDED_ITERATIONS])
@pytest.mark.parametrize("iteration", [0, 1, 500, 6000, 30000, 60000, 2_000_000_000])
def test_progress_matches_count_loss_inline(horizon, iteration):
    clock = AnnealClock(horizon)
    assert math.isclose(
        clock.progress(iteration), _count_loss_u(iteration, horizon), rel_tol=0, abs_tol=1e-12
    )


@pytest.mark.parametrize("start", [0, 500])
@pytest.mark.parametrize("iteration", [0, 1, 500, 6000, 30000, 60000])
def test_progress_matches_schedule_progress(start, iteration):
    # With start>=0 the clock equals utils.mcmc_schedule.progress(start, stop).
    horizon = 30000
    clock = AnnealClock(horizon)
    assert math.isclose(
        clock.progress(iteration, start=start),
        schedule_progress(iteration, start, horizon),
        rel_tol=0,
        abs_tol=1e-12,
    )


def test_apply_overrides_stashes_clock_neutral_horizon():
    opt = SimpleNamespace(
        iterations=30_000,
        densify_until_iter=25_000,
        mcmc_noise_stop_iter=30_000,
        mcmc_stop_growth_iter=12_000,
        position_lr_max_steps=30_000,
    )
    apply_online_schedule_overrides(opt)
    # Default-neutral: clock horizon equals the (unbounded) loop bound, so
    # reading it yields the same u as max_iterations=opt.iterations did before.
    assert opt._anneal_clock.horizon == ONLINE_UNBOUNDED_ITERATIONS
    assert opt._anneal_clock.horizon == opt.iterations


def test_apply_overrides_finite_anneal_horizon_decouples_loop_bound():
    opt = SimpleNamespace(
        iterations=30_000,
        densify_until_iter=25_000,
        mcmc_noise_stop_iter=30_000,
        mcmc_stop_growth_iter=12_000,
        position_lr_max_steps=30_000,
    )
    apply_online_schedule_overrides(opt, anneal_horizon=30_000)
    # Loop bound stays unbounded...
    assert opt.iterations == ONLINE_UNBOUNDED_ITERATIONS
    # ...but the anneal progresses over the finite horizon.
    assert opt._anneal_clock.horizon == 30_000
    assert opt._anneal_clock.progress(30_000) == 1.0
    assert opt._anneal_clock.progress(15_000) == 0.5


# --- Adaptive schedule horizons (2026-07-18) --------------------------------

def test_adaptive_horizons_off_never_fires():
    assert not should_apply_adaptive_schedule_horizons(
        adaptive_on=False, already_applied=False,
        last_ingest_iter=100, first_iter=0, iteration=5000, threshold=1000,
    )


def test_adaptive_horizons_does_not_refire_once_applied():
    assert not should_apply_adaptive_schedule_horizons(
        adaptive_on=True, already_applied=True,
        last_ingest_iter=100, first_iter=0, iteration=5000, threshold=1000,
    )


def test_adaptive_horizons_never_fires_before_any_ingestion():
    # last_ingest_iter == first_iter means no frame has ever been ingested yet
    # (the initial value, not a real ingest event) -- must not fire.
    assert not should_apply_adaptive_schedule_horizons(
        adaptive_on=True, already_applied=False,
        last_ingest_iter=0, first_iter=0, iteration=5000, threshold=1000,
    )


def test_adaptive_horizons_fires_exactly_at_threshold():
    # gap = iteration - last_ingest_iter must be >= threshold (inclusive).
    assert not should_apply_adaptive_schedule_horizons(
        adaptive_on=True, already_applied=False,
        last_ingest_iter=5000, first_iter=0, iteration=5999, threshold=1000,
    )
    assert should_apply_adaptive_schedule_horizons(
        adaptive_on=True, already_applied=False,
        last_ingest_iter=5000, first_iter=0, iteration=6000, threshold=1000,
    )


def test_adaptive_horizons_fires_well_past_threshold_too():
    assert should_apply_adaptive_schedule_horizons(
        adaptive_on=True, already_applied=False,
        last_ingest_iter=5000, first_iter=0, iteration=50000, threshold=1000,
    )
