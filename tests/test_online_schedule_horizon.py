"""A6: finite online MCMC-annealing horizon, decoupled from the loop bound.

online_mode historically pinned every progress-normalized MCMC schedule field to
2^31, freezing the schedule at u≈0 (growth at MAX, intervals at MIN, dead-opacity at
its floor) -> perpetual low-opacity fog. A finite schedule_horizon must anneal the
schedule WITHOUT bounding the (unbounded) training loop.
"""
from eligsir.core.stop_policy import (
    apply_online_schedule_overrides,
    enforce_growth_stop,
    validate_fixed_topology_contract,
    ONLINE_UNBOUNDED_ITERATIONS,
)


class _Opt:
    pass


def test_finite_horizon_decouples_loop_from_schedule():
    opt = _Opt()
    sched = apply_online_schedule_overrides(opt, schedule_horizon=30000)
    assert sched == 30000
    # loop bound stays unbounded
    assert opt.iterations == ONLINE_UNBOUNDED_ITERATIONS
    # every schedule field anneals on the finite horizon
    assert opt.densify_until_iter == 30000
    assert opt.mcmc_stop_growth_iter == 30000
    assert opt.mcmc_noise_stop_iter == 30000
    assert opt.position_lr_max_steps == 30000
    # count-loss consumer reads the same clock
    assert opt._online_schedule_horizon == 30000


def test_zero_horizon_is_legacy_unbounded():
    for val in (0, None, -1):
        opt = _Opt()
        apply_online_schedule_overrides(opt, schedule_horizon=val)
        assert opt.iterations == ONLINE_UNBOUNDED_ITERATIONS
        assert opt.densify_until_iter == ONLINE_UNBOUNDED_ITERATIONS
        assert opt.mcmc_stop_growth_iter == ONLINE_UNBOUNDED_ITERATIONS
        assert opt._online_schedule_horizon == ONLINE_UNBOUNDED_ITERATIONS


def test_global_maintenance_cannot_reenable_growth_after_hard_stop():
    schedule = {"allow_growth": True, "allow_relocation": True}

    enforce_growth_stop(schedule, iteration=20_000, stop_growth_iter=1)

    assert schedule["allow_growth"] is False
    assert schedule["allow_relocation"] is True


def test_fixed_topology_contract_rejects_raw_stop_without_explicit_horizon():
    errors = validate_fixed_topology_contract(
        online_mode=True,
        requested_growth_stop=1,
        explicit_growth_stop=None,
        resolved_growth_stop=ONLINE_UNBOUNDED_ITERATIONS,
        insertion_enabled=False,
        explicit_insertion_stop=None,
    )

    assert any("streaming_growth_stop_horizon" in error for error in errors)


def test_fixed_topology_contract_requires_matching_resolved_stops():
    errors = validate_fixed_topology_contract(
        online_mode=False,
        requested_growth_stop=1,
        explicit_growth_stop=1,
        resolved_growth_stop=1,
        insertion_enabled=False,
        explicit_insertion_stop=1,
    )

    assert errors == []
