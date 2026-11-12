"""Regression test: pop_pruning_stats() is windowed (resets each call), while
get_cumulative_pruning_stats() accumulates over the whole run and is never
reset by pop_pruning_stats()/reset_pruning_stats().

Root cause this guards against: geometry_metrics.json's "pruning" block is
read via pop_pruning_stats() on every periodic report, so two runs sampled at
different offsets relative to their own last periodic report can show wildly
different (even zero) windowed pruning counts despite near-identical whole-run
pruning activity -- this looked like a real sh_impl-dependent functional gap
during the 2026-07-16 SH-native-backward-parity diagnostic but was a
telemetry-window artifact. "pruning_cumulative" is the fix: a true run-total
that survives periodic-report resets.
"""
from eligsir.utils.energy_mcmc import (
    _bump_pruning_stat,
    _pruning_stats,
    _pruning_stats_cumulative,
    get_cumulative_pruning_stats,
    pop_pruning_stats,
    reset_pruning_stats,
)


def _reset_all():
    reset_pruning_stats()
    for k in _pruning_stats_cumulative:
        _pruning_stats_cumulative[k] = 0


def test_pop_resets_window_but_cumulative_survives():
    _reset_all()
    _bump_pruning_stat("dead_opacity", 5)
    _bump_pruning_stat("total_pruned", 5)

    window1 = pop_pruning_stats()
    assert window1["dead_opacity"] == 5
    assert window1["total_pruned"] == 5

    # Simulate more pruning happening between periodic reports.
    _bump_pruning_stat("dead_opacity", 3)
    _bump_pruning_stat("total_pruned", 3)

    window2 = pop_pruning_stats()
    assert window2["dead_opacity"] == 3  # windowed: only the second batch
    assert window2["total_pruned"] == 3

    cumulative = get_cumulative_pruning_stats()
    assert cumulative["dead_opacity"] == 8  # cumulative: survives both resets
    assert cumulative["total_pruned"] == 8


def test_cumulative_unaffected_by_reset_pruning_stats():
    _reset_all()
    _bump_pruning_stat("low_support", 7)
    reset_pruning_stats()

    assert pop_pruning_stats()["low_support"] == 0
    assert get_cumulative_pruning_stats()["low_support"] == 7
