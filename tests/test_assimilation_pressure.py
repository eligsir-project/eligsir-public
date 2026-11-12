from dataclasses import FrozenInstanceError

import pytest

from eligsir.utils.assimilation_pressure import (
    AssimilationPressureTier,
    classify_assimilation_pressure,
    shadow_decision,
)


def test_all_zero_signals_is_low_pressure():
    tier = classify_assimilation_pressure(
        source_insert_rate=0.0,
        support_debt=0.0,
        residual_pressure=0.0,
        partition_pressure=0.0,
        newborn_pressure=0.0,
        optimization_rate=1.0,
    )
    assert tier == AssimilationPressureTier.LOW


def test_all_signals_saturated_is_critical_pressure():
    tier = classify_assimilation_pressure(
        source_insert_rate=1.0,
        support_debt=1.0,
        residual_pressure=1.0,
        partition_pressure=1.0,
        newborn_pressure=1.0,
        optimization_rate=0.0,
    )
    assert tier == AssimilationPressureTier.CRITICAL


def test_mixed_signals_land_in_medium_or_high():
    tier = classify_assimilation_pressure(
        source_insert_rate=0.5,
        support_debt=0.5,
        residual_pressure=0.5,
        partition_pressure=0.5,
        newborn_pressure=0.5,
        optimization_rate=0.5,
    )
    assert tier in (AssimilationPressureTier.MEDIUM, AssimilationPressureTier.HIGH)


def test_signals_are_clamped_to_valid_range():
    # out-of-range inputs (e.g. a rate signal that can exceed 1.0 before normalization
    # upstream) must not raise or silently produce a nonsense tier. With clamping applied,
    # source_insert_rate=5.0 -> 1.0, support_debt=-1.0 -> 0.0, optimization_rate=2.0 -> 1.0
    # (then inverted to 0.0), giving signals [1.0, 0.0, 0.0, 0.0, 0.0, 0.0], score = 1/6 ~=
    # 0.1667, which is below the 0.25 LOW threshold.
    tier = classify_assimilation_pressure(
        source_insert_rate=5.0,
        support_debt=-1.0,
        residual_pressure=0.0,
        partition_pressure=0.0,
        newborn_pressure=0.0,
        optimization_rate=2.0,
    )
    assert tier == AssimilationPressureTier.LOW


def test_optimization_rate_is_inverted_relative_to_the_other_five():
    # low optimization_rate (the optimizer is starved) should INCREASE pressure,
    # not decrease it, unlike the other five signals where higher = more pressure
    low_opt_rate_tier = classify_assimilation_pressure(
        source_insert_rate=0.2, support_debt=0.2, residual_pressure=0.2,
        partition_pressure=0.2, newborn_pressure=0.2, optimization_rate=0.0,
    )
    high_opt_rate_tier = classify_assimilation_pressure(
        source_insert_rate=0.2, support_debt=0.2, residual_pressure=0.2,
        partition_pressure=0.2, newborn_pressure=0.2, optimization_rate=1.0,
    )
    # low_opt_rate_tier: signals [0.2, 0.2, 0.2, 0.2, 0.2, 1.0], score = 1.8/6 = 0.3 -> MEDIUM
    # high_opt_rate_tier: signals [0.2, 0.2, 0.2, 0.2, 0.2, 0.0], score = 1.0/6 ~= 0.1667 -> LOW
    # A strict > is required here: `>=` cannot distinguish "inversion correctly applied" from
    # "optimization_rate silently ignored" (both would satisfy >= for these inputs).
    assert int(low_opt_rate_tier) > int(high_opt_rate_tier)


def test_shadow_decision_low_tier_activates_no_lever():
    d = shadow_decision(AssimilationPressureTier.LOW, base_steps=100)
    assert d.would_suppress_insertion is False
    assert d.would_reduce_steps is False
    assert d.would_be_steps == 100
    assert d.would_bias_replay_weak is False
    assert d.would_suppress_growth is False


def test_shadow_decision_medium_tier_activates_only_replay_bias():
    d = shadow_decision(AssimilationPressureTier.MEDIUM, base_steps=100)
    assert d.would_bias_replay_weak is True
    assert d.would_suppress_insertion is False
    assert d.would_reduce_steps is False
    assert d.would_suppress_growth is False


def test_shadow_decision_high_tier_activates_insertion_suppression_and_consolidation():
    d = shadow_decision(AssimilationPressureTier.HIGH, base_steps=100)
    assert d.would_bias_replay_weak is True
    assert d.would_suppress_insertion is True
    assert d.would_reduce_steps is True
    assert d.would_be_steps == 150  # 100 * 1.5 consolidation multiplier
    assert d.would_suppress_growth is False  # growth is CRITICAL-only, per #110's own ordering


def test_shadow_decision_critical_tier_activates_every_lever_including_growth():
    d = shadow_decision(AssimilationPressureTier.CRITICAL, base_steps=100)
    assert d.would_bias_replay_weak is True
    assert d.would_suppress_insertion is True
    assert d.would_reduce_steps is True
    assert d.would_be_steps == 150
    assert d.would_suppress_growth is True


def test_shadow_decision_is_never_applied_only_observed():
    # shadow_decision takes no MappingDecision/mutable state and returns a frozen
    # dataclass -- structurally incapable of mutating real training behavior.
    d = shadow_decision(AssimilationPressureTier.CRITICAL, base_steps=50)
    with pytest.raises(FrozenInstanceError):
        d.would_suppress_growth = False


def test_shadow_decision_would_be_steps_echoes_base_when_no_reduction():
    d = shadow_decision(AssimilationPressureTier.MEDIUM, base_steps=37)
    assert d.would_reduce_steps is False
    assert d.would_be_steps == 37
