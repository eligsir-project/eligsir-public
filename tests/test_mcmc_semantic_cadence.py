from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest

from eligsir.contracts.work_clocks import RuntimeWorkClockSnapshot


def _cadence_types():
    module = importlib.import_module(
        "eligsir.mapping.scheduler.mcmc_cadence"
    )
    return module.MCMCCadenceOwner, module.MCMCCadenceDecision


def _snapshot(
    *,
    optimizer_updates: int,
    supervised_rgb_pixels: int,
    depth_valid_samples: int = 0,
    source_complete: bool = False,
) -> RuntimeWorkClockSnapshot:
    return RuntimeWorkClockSnapshot(
        optimizer_updates=optimizer_updates,
        supervised_rgb_views=int(supervised_rgb_pixels > 0),
        supervised_rgb_pixels=supervised_rgb_pixels,
        depth_valid_samples=depth_valid_samples,
        source_complete=source_complete,
    )


def _first_due_exposure(chunks: list[int], *, pixels_per_view: int) -> int:
    owner_type, _ = _cadence_types()
    owner = owner_type(
        grow_interval_pixels=800,
        relocate_interval_pixels=800,
        min_optimizer_updates_after_structural_mutation=0,
    )
    pixels = 0
    updates = 0
    for chunk in chunks:
        pixels += chunk * pixels_per_view
        updates += 1
        decision = owner.decide(
            _snapshot(
                optimizer_updates=updates,
                supervised_rgb_pixels=pixels,
            ),
            topology_version=0,
        )
        if decision.grow_due:
            return pixels
    raise AssertionError("growth never became due")


def test_mcmc_due_point_depends_on_pixels_not_transactions():
    assert _first_due_exposure([1] * 8, pixels_per_view=100) == 800
    assert _first_due_exposure([4, 4], pixels_per_view=100) == 800


def test_population_authorization_is_neutral_and_accepts_depth_opportunity():
    """A live scheduler token must not encode a legacy action choice.

    Regresses a live-policy bridge that copied only RGB grow/relocate due
    lanes: geometry evidence would then never authorize policy work until the
    unrelated RGB clock crossed its MCMC threshold.
    """

    owner_type, _ = _cadence_types()
    owner = owner_type(
        grow_interval_pixels=100,
        relocate_interval_pixels=100,
        min_optimizer_updates_after_structural_mutation=0,
    )

    token = owner.authorize_population_mutation(
        _snapshot(
            optimizer_updates=4,
            supervised_rgb_pixels=0,
            depth_valid_samples=1,
        ),
        topology_version=0,
        max_actions=2,
        geometry_opportunity=True,
    )

    assert token.epoch == 4
    assert token.units == 2
    assert token.geometry_opportunity is True
    assert not hasattr(token, "grow_due")
    assert not hasattr(token, "relocate_due")


def test_population_authorization_without_any_due_opportunity_is_an_abstention_budget():
    """A live policy gets a zero budget instead of an RGB-only action lane."""

    owner_type, _ = _cadence_types()
    owner = owner_type(
        grow_interval_pixels=100,
        relocate_interval_pixels=100,
        min_optimizer_updates_after_structural_mutation=0,
    )

    token = owner.authorize_population_mutation(
        _snapshot(optimizer_updates=4, supervised_rgb_pixels=0),
        topology_version=0,
        max_actions=2,
        geometry_opportunity=False,
    )

    assert token.epoch == 4
    assert token.units == 0
    assert token.geometry_opportunity is False


def test_population_commit_consumes_neutral_authorization_until_fresh_work_arrives():
    """An applied live plan cannot reuse an already-consumed RGB due debt."""

    owner_type, _ = _cadence_types()
    owner = owner_type(
        grow_interval_pixels=100,
        relocate_interval_pixels=100,
        min_optimizer_updates_after_structural_mutation=2,
    )
    due = _snapshot(optimizer_updates=2, supervised_rgb_pixels=100)
    assert owner.authorize_population_mutation(
        due, topology_version=0, max_actions=1, geometry_opportunity=False
    ).units == 1

    owner.record_population_mutation(due, topology_version_after=1)

    assert owner.authorize_population_mutation(
        _snapshot(optimizer_updates=3, supervised_rgb_pixels=100),
        topology_version=1, max_actions=1, geometry_opportunity=False,
    ).units == 0
    assert owner.authorize_population_mutation(
        _snapshot(optimizer_updates=4, supervised_rgb_pixels=100),
        topology_version=1, max_actions=1, geometry_opportunity=False,
    ).units == 0
    assert owner.authorize_population_mutation(
        _snapshot(optimizer_updates=4, supervised_rgb_pixels=200),
        topology_version=1, max_actions=1, geometry_opportunity=False,
    ).units == 1


def test_overdue_exposure_emits_one_epoch_without_catchup():
    owner_type, _ = _cadence_types()
    owner = owner_type(
        grow_interval_pixels=100,
        relocate_interval_pixels=1_000,
        min_optimizer_updates_after_structural_mutation=2,
    )
    decision = owner.decide(
        _snapshot(optimizer_updates=2, supervised_rgb_pixels=550),
        topology_version=0,
    )
    assert decision.grow_due

    owner.record_structural_mutation(
        _snapshot(optimizer_updates=2, supervised_rgb_pixels=550),
        topology_version_after=1,
        grew=True,
        relocated=False,
    )

    assert not owner.decide(
        _snapshot(optimizer_updates=3, supervised_rgb_pixels=550),
        topology_version=1,
    ).grow_due


def test_quality_brake_denies_growth_without_consuming_relocation_debt():
    owner_type, _ = _cadence_types()
    owner = owner_type(
        grow_interval_pixels=100,
        relocate_interval_pixels=100,
        min_optimizer_updates_after_structural_mutation=0,
    )
    braked = owner.decide(
        _snapshot(optimizer_updates=1, supervised_rgb_pixels=100),
        topology_version=0,
        growth_allowed=False,
    )
    assert not braked.grow_due
    assert braked.relocate_due

    released = owner.decide(
        _snapshot(optimizer_updates=1, supervised_rgb_pixels=100),
        topology_version=0,
        growth_allowed=True,
    )
    assert released.grow_due
    assert released.relocate_due


def test_source_completion_does_not_reset_exposure_debt():
    owner_type, _ = _cadence_types()
    owner = owner_type(
        grow_interval_pixels=100,
        relocate_interval_pixels=100,
        min_optimizer_updates_after_structural_mutation=0,
    )
    before = owner.decide(
        _snapshot(optimizer_updates=1, supervised_rgb_pixels=90),
        topology_version=0,
    )
    after = owner.decide(
        _snapshot(
            optimizer_updates=1,
            supervised_rgb_pixels=100,
            source_complete=True,
        ),
        topology_version=0,
    )
    assert before.exposure_debt_pixels == 90
    assert after.grow_due and after.relocate_due
    assert after.exposure_debt_pixels == 100


def test_source_completion_switches_growth_interval_without_resetting_debt():
    """Catch a runtime that keeps native refinement on the ingestion cadence."""

    owner_type, _ = _cadence_types()
    owner = owner_type(
        grow_interval_pixels=100,
        post_ingestion_grow_interval_pixels=1_000,
        relocate_interval_pixels=1_000,
        min_optimizer_updates_after_structural_mutation=0,
    )
    before = owner.decide(
        _snapshot(optimizer_updates=1, supervised_rgb_pixels=100),
        topology_version=0,
    )
    after = owner.decide(
        _snapshot(
            optimizer_updates=1,
            supervised_rgb_pixels=100,
            source_complete=True,
        ),
        topology_version=0,
    )

    assert before.grow_due
    assert before.active_grow_interval_pixels == 100
    assert not after.grow_due
    assert after.active_grow_interval_pixels == 1_000
    assert after.exposure_debt_pixels == 100

    due = owner.decide(
        _snapshot(
            optimizer_updates=2,
            supervised_rgb_pixels=1_000,
            source_complete=True,
        ),
        topology_version=0,
    )
    assert due.grow_due
    assert due.active_grow_interval_pixels == 1_000


def test_zero_post_ingestion_growth_interval_preserves_existing_cadence():
    """Catch a compatibility regression when the new override is disabled."""

    owner_type, _ = _cadence_types()
    owner = owner_type(
        grow_interval_pixels=100,
        post_ingestion_grow_interval_pixels=0,
        relocate_interval_pixels=1_000,
        min_optimizer_updates_after_structural_mutation=0,
    )
    decision = owner.decide(
        _snapshot(
            optimizer_updates=1,
            supervised_rgb_pixels=100,
            source_complete=True,
        ),
        topology_version=0,
    )

    assert decision.grow_due
    assert decision.active_grow_interval_pixels == 100


def test_source_completion_switches_relocation_interval_without_resetting_debt():
    """Catch native refinement that retains the fast ingestion relocation cadence."""

    owner_type, _ = _cadence_types()
    owner = owner_type(
        grow_interval_pixels=1_000,
        relocate_interval_pixels=100,
        post_ingestion_relocate_interval_pixels=1_000,
        min_optimizer_updates_after_structural_mutation=0,
    )
    before = owner.decide(
        _snapshot(optimizer_updates=1, supervised_rgb_pixels=100),
        topology_version=0,
    )
    after = owner.decide(
        _snapshot(
            optimizer_updates=1,
            supervised_rgb_pixels=100,
            source_complete=True,
        ),
        topology_version=0,
    )

    assert before.relocate_due
    assert before.active_relocate_interval_pixels == 100
    assert not after.relocate_due
    assert after.active_relocate_interval_pixels == 1_000
    assert after.exposure_debt_pixels == 100


def test_growth_commit_uses_the_active_post_ingestion_interval():
    """Catch commits that bypass the slower post-ingestion cadence fence."""

    owner_type, _ = _cadence_types()
    owner = owner_type(
        grow_interval_pixels=100,
        post_ingestion_grow_interval_pixels=1_000,
        relocate_interval_pixels=1_000,
        min_optimizer_updates_after_structural_mutation=0,
    )
    snapshot = _snapshot(
        optimizer_updates=1,
        supervised_rgb_pixels=500,
        source_complete=True,
    )
    assert not owner.decide(snapshot, topology_version=0).grow_due

    with pytest.raises(ValueError, match="growth had no due exposure debt"):
        owner.record_structural_mutation(
            snapshot,
            topology_version_after=1,
            grew=True,
        )


def test_external_topology_epoch_resets_only_optimizer_settling_baseline():
    owner_type, _ = _cadence_types()
    owner = owner_type(
        grow_interval_pixels=100,
        relocate_interval_pixels=100,
        min_optimizer_updates_after_structural_mutation=2,
    )
    owner.decide(
        _snapshot(optimizer_updates=2, supervised_rgb_pixels=100),
        topology_version=0,
    )
    unsettled = owner.decide(
        _snapshot(optimizer_updates=2, supervised_rgb_pixels=100),
        topology_version=1,
    )
    assert not unsettled.optimizer_settled
    assert not unsettled.grow_due
    assert unsettled.exposure_debt_pixels == 100

    settled = owner.decide(
        _snapshot(optimizer_updates=4, supervised_rgb_pixels=100),
        topology_version=1,
    )
    assert settled.optimizer_settled
    assert settled.grow_due and settled.relocate_due


def test_denied_or_failed_work_retains_debt_until_successful_commit():
    owner_type, _ = _cadence_types()
    owner = owner_type(
        grow_interval_pixels=100,
        relocate_interval_pixels=100,
        min_optimizer_updates_after_structural_mutation=0,
    )
    due = owner.decide(
        _snapshot(optimizer_updates=1, supervised_rgb_pixels=100),
        topology_version=0,
    )
    assert due.grow_due and due.relocate_due
    assert owner.decide(
        _snapshot(optimizer_updates=2, supervised_rgb_pixels=100),
        topology_version=0,
    ).grow_due


def test_successful_commit_consumes_only_the_effect_that_actually_applied():
    owner_type, _ = _cadence_types()
    owner = owner_type(
        grow_interval_pixels=100,
        relocate_interval_pixels=100,
        min_optimizer_updates_after_structural_mutation=0,
    )
    snapshot = _snapshot(optimizer_updates=1, supervised_rgb_pixels=100)
    due = owner.decide(snapshot, topology_version=0)
    assert due.grow_due and due.relocate_due

    owner.record_structural_mutation(
        snapshot,
        topology_version_after=1,
        grew=True,
        relocated=False,
    )

    after = owner.decide(snapshot, topology_version=1)
    assert not after.grow_due
    assert after.relocate_due


def test_depth_structural_mutation_resets_settling_without_consuming_debt():
    owner_type, _ = _cadence_types()
    owner = owner_type(
        grow_interval_pixels=100,
        relocate_interval_pixels=100,
        min_optimizer_updates_after_structural_mutation=2,
    )
    snapshot = _snapshot(optimizer_updates=2, supervised_rgb_pixels=100)
    assert owner.decide(snapshot, topology_version=0).grow_due

    owner.record_structural_mutation(snapshot, topology_version_after=1)

    unsettled = owner.decide(snapshot, topology_version=1)
    assert not unsettled.optimizer_settled
    assert unsettled.exposure_debt_pixels == 100
    settled = owner.decide(
        _snapshot(optimizer_updates=4, supervised_rgb_pixels=100),
        topology_version=1,
    )
    assert settled.grow_due and settled.relocate_due


def test_state_round_trip_is_strict_and_rejects_counter_or_version_regression():
    owner_type, _ = _cadence_types()
    owner = owner_type(
        grow_interval_pixels=100,
        relocate_interval_pixels=50,
        min_optimizer_updates_after_structural_mutation=2,
    )
    owner.decide(
        _snapshot(optimizer_updates=4, supervised_rgb_pixels=200),
        topology_version=3,
    )
    payload = owner.state_dict()

    restored = owner_type(
        grow_interval_pixels=100,
        relocate_interval_pixels=50,
        min_optimizer_updates_after_structural_mutation=2,
    )
    restored.load_state_dict(payload)
    assert restored.state_dict() == payload

    with pytest.raises(ValueError, match="regressed"):
        restored.decide(
            _snapshot(optimizer_updates=3, supervised_rgb_pixels=200),
            topology_version=3,
        )
    with pytest.raises(ValueError, match="regressed"):
        restored.decide(
            _snapshot(optimizer_updates=4, supervised_rgb_pixels=200),
            topology_version=2,
        )

    with pytest.raises(ValueError, match="incompatible|keys"):
        restored.load_state_dict(payload | {"unknown": 1})

    rollback = owner_type(
        grow_interval_pixels=100,
        relocate_interval_pixels=50,
        min_optimizer_updates_after_structural_mutation=2,
    ).state_dict()
    with pytest.raises(ValueError, match="rollback|regressed"):
        restored.load_state_dict(rollback)


def _owner_with_committed_cadence_baselines():
    owner_type, _ = _cadence_types()
    owner = owner_type(
        grow_interval_pixels=100,
        relocate_interval_pixels=50,
        min_optimizer_updates_after_structural_mutation=2,
    )
    snapshot = _snapshot(optimizer_updates=4, supervised_rgb_pixels=200)
    owner.decide(snapshot, topology_version=0)
    owner.record_structural_mutation(
        snapshot,
        topology_version_after=1,
        grew=True,
        relocated=True,
    )
    return owner


def test_state_load_rejects_committed_grow_baseline_regression():
    owner = _owner_with_committed_cadence_baselines()
    payload = owner.state_dict() | {"last_committed_grow_pixels": 100}

    with pytest.raises(ValueError, match="last_committed_grow_pixels.*regress"):
        owner.load_state_dict(payload)


def test_state_load_rejects_committed_relocate_baseline_regression():
    owner = _owner_with_committed_cadence_baselines()
    payload = owner.state_dict() | {"last_committed_relocate_pixels": 100}

    with pytest.raises(
        ValueError, match="last_committed_relocate_pixels.*regress"
    ):
        owner.load_state_dict(payload)


def test_state_load_rejects_structural_update_baseline_regression():
    owner = _owner_with_committed_cadence_baselines()
    payload = owner.state_dict() | {"last_structural_optimizer_updates": 3}

    with pytest.raises(
        ValueError, match="last_structural_optimizer_updates.*regress"
    ):
        owner.load_state_dict(payload)


def test_explicit_semantic_due_flags_bypass_legacy_iteration_horizons(monkeypatch):
    import eligsir.utils.mcmc_schedule as schedule_module
    from eligsir.utils.mcmc_schedule import MCMCScheduleConfig, get_mcmc_schedule

    monkeypatch.setattr(
        schedule_module,
        "progress",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("legacy iteration progress was evaluated")
        ),
    )

    schedule = get_mcmc_schedule(
        0,
        current_n=10,
        cap_max=100,
        cfg=MCMCScheduleConfig(
            start_iter=500,
            stop_growth_iter=501,
            stop_reloc_iter=501,
        ),
        grow_due=True,
        relocate_due=True,
    )

    assert schedule["allow_growth"] is True
    assert schedule["allow_relocation"] is True
    assert schedule["grow_due"] is True
    assert schedule["relocate_due"] is True
    assert schedule["u_growth"] == 0.0
    assert schedule["u_reloc"] == 0.0
    assert schedule["growth_factor"] == schedule["semantic_growth_factor"]


def test_energy_strategy_explicit_due_flag_never_calls_legacy_due(monkeypatch):
    from eligsir.utils.strategies.mcmc.energy import GsplatEnergyMCMCStrategy

    strategy = GsplatEnergyMCMCStrategy()
    monkeypatch.setattr(
        strategy,
        "_due",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("legacy iteration cadence was consulted")
        ),
    )

    assert strategy._mutation_due(
        "grow",
        {"grow_due": True},
        optimizer_updates=1,
        interval=100,
        start_update=500,
    )


def test_temperature_schedule_is_explicitly_optimizer_update_driven():
    from eligsir.utils.strategies.mcmc.proposal import (
        temperature_from_optimizer_updates,
    )

    args = SimpleNamespace(
        energy_temp_start=10.0,
        energy_temp_min=1.0,
        energy_temp_tau=0.4,
        mcmc_temperature_horizon_optimizer_updates=30_000,
    )
    assert temperature_from_optimizer_updates(args, optimizer_updates=0) == 10.0
    assert temperature_from_optimizer_updates(
        args, optimizer_updates=30_000
    ) == pytest.approx(1.0)
    with pytest.raises(TypeError, match="optimizer_updates"):
        temperature_from_optimizer_updates(args, optimizer_updates=True)


def test_legacy_noise_stop_iteration_cannot_actuate_canonical_strategy():
    from eligsir.utils.strategies.mcmc.energy import _noise_stop_from_policy

    canonical = SimpleNamespace(
        mcmc_noise_stop_optimizer_updates=30_000,
        mcmc_noise_stop_iter=1,
        iterations=2,
    )
    assert _noise_stop_from_policy(
        canonical, semantic_cadence=True
    ) == 30_000

    offline = SimpleNamespace(
        mcmc_noise_stop_optimizer_updates=30_000,
        mcmc_noise_stop_iter=1,
        iterations=2,
    )
    assert _noise_stop_from_policy(offline, semantic_cadence=False) == 1


def test_live_phase_position_noise_gate_overrides_update_horizon():
    from eligsir.utils.strategies.mcmc.energy import _position_noise_allowed

    assert not _position_noise_allowed(
        {"allow_position_noise": False},
        optimizer_updates=10,
        noise_stop_iter=30_000,
    )
    assert _position_noise_allowed(
        {"allow_position_noise": True},
        optimizer_updates=10,
        noise_stop_iter=30_000,
    )
    assert not _position_noise_allowed(
        {"allow_position_noise": True},
        optimizer_updates=30_001,
        noise_stop_iter=30_000,
    )


def test_live_phase_position_noise_gate_requires_exact_bool():
    from eligsir.utils.strategies.mcmc.energy import _position_noise_allowed

    with pytest.raises(TypeError, match="allow_position_noise"):
        _position_noise_allowed(
            {"allow_position_noise": 1},
            optimizer_updates=10,
            noise_stop_iter=30_000,
        )


def test_factory_threads_semantic_cadence_to_direct_and_live_strategies(
    monkeypatch,
):
    import eligsir.utils.strategies.mcmc.registry as registry

    monkeypatch.setattr(
        registry,
        "_owned",
        lambda strategy, _gaussians, _args: strategy,
    )
    gaussians = SimpleNamespace(uses_gsplat_layout=True)

    direct = registry.make_mcmc_strategy(
        "gsplat_energy_mcmc",
        gaussians=gaussians,
        args=SimpleNamespace(),
        semantic_cadence=True,
    )
    live = registry.make_mcmc_strategy(
        "gsplat_live_strategy",
        gaussians=gaussians,
        args=SimpleNamespace(),
        semantic_cadence=True,
    )
    offline = registry.make_mcmc_strategy(
        "gsplat_energy_mcmc",
        gaussians=gaussians,
        args=SimpleNamespace(),
    )

    assert direct.semantic_cadence is True
    assert live._base.semantic_cadence is True
    assert offline.semantic_cadence is False


def test_factory_rejects_non_bool_semantic_cadence():
    from eligsir.utils.strategies.mcmc.registry import make_mcmc_strategy

    with pytest.raises(TypeError, match="semantic_cadence"):
        make_mcmc_strategy(
            "gsplat_energy_mcmc",
            gaussians=SimpleNamespace(uses_gsplat_layout=True),
            args=SimpleNamespace(),
            semantic_cadence=1,
        )
