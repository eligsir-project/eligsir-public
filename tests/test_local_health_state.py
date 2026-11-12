"""Tests for Level C (issue #358): the reconstruction-health state machine.

CPU-only, no GPU, no torch. Two of these encode defects this project has
already shipped once and must not ship again:

* growth must never be reported as forgetting (PR #375's fake 14 dB drop);
* a scrambled-PSNR history must leave everything PSNR-INDEPENDENT byte-identical
  (PR #375's non-circularity proof, adapted -- see the test's own docstring for
  why the assertion differs here).
"""
import json
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from eligsir.utils.local_health_state import (  # noqa: E402
    ADVERSE_STATES, DEGRADING, ESTABLISHED, FORGOTTEN, NOT_ACTIVE, RECOVERING,
    STABLE, UNESTABLISHED, HealthPolicy, QualitySeries, cell_series,
    classify_all, classify_health, health_records, morton_order_key,
    partition_event_records, partition_snapshot, region_series,
    severe_forgetting_marks, worst_k, write_health_history,
    write_partition_events, write_partition_snapshot,
)
from eligsir.utils.local_quality import cell_key, pooled_psnr_db  # noqa: E402
from eligsir.utils.local_regions import RegionPolicy, derive_region_history  # noqa: E402


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def mk_series(psnr, iters=None, established=None, measured=None, active=None,
              confounded=None, kind="cell", entity_id="1"):
    n = len(psnr)
    iters = iters if iters is not None else [1000 * (i + 1) for i in range(n)]
    return QualitySeries(
        entity_id=entity_id, label=f"E{entity_id}", kind=kind,
        iterations=np.asarray(iters, dtype=np.int64),
        psnr_db=np.asarray(psnr, dtype=np.float64),
        established=np.asarray(established if established is not None
                               else [True] * n, dtype=bool),
        measured=np.asarray(measured if measured is not None
                            else [True] * n, dtype=bool),
        active=np.asarray(active if active is not None else [True] * n, dtype=bool),
        confounded=np.asarray(confounded if confounded is not None
                              else [False] * n, dtype=bool),
    )


def cell_record(iteration, cells):
    """One Level A history row. ``cells`` is ``{key: (psnr, established, n_samples)}``."""
    return {
        "iteration": int(iteration),
        "cell_size_m": 0.5,
        "n_cells_observed": len(cells),
        "n_cells_established": sum(1 for v in cells.values() if v[1]),
        "cells": {
            str(k): {"cell_key": int(k), "psnr_db": float(p),
                     "established": bool(e), "n_eval_samples": int(ns),
                     "n_views": 3}
            for k, (p, e, ns) in cells.items()
        },
    }


# ---------------------------------------------------------------------------
# basic state transitions
# ---------------------------------------------------------------------------
def test_unestablished_never_produces_a_drop():
    s = mk_series([10.0, 12.0, 9.0], established=[False, False, False])
    r = classify_health(s)
    assert r.states == [UNESTABLISHED] * 3
    assert r.drop_from_peak_db == [None] * 3
    assert r.summary["max_drop_from_peak_db"] == 0.0
    assert r.episodes == []


def test_monotonically_improving_never_leaves_the_healthy_states():
    """A region that keeps getting better GREW. It did not forget anything."""
    s = mk_series([18.0, 21.0, 24.0, 26.0, 28.0, 30.0])
    r = classify_health(s)
    assert set(r.states) <= {UNESTABLISHED, ESTABLISHED, STABLE}
    assert r.summary["max_drop_from_peak_db"] == 0.0
    assert r.summary["episode_count"] == 0
    assert r.summary["adverse_at_end"] is False
    assert r.summary["recent_slope_db_per_1k_iters"] > 0


def test_growth_after_a_dip_is_measured_against_the_RUNNING_peak():
    """Comparing early samples against a LATER peak would report initial
    under-reconstruction as forgetting -- Level A/B's running-peak rule."""
    s = mk_series([20.0, 30.0, 25.0, 34.0])
    r = classify_health(s)
    # tick 0 is the peak-so-far; it can never be "5 dB below" the later 30/34.
    assert r.drop_from_peak_db[0] == 0.0
    assert r.drop_from_peak_db[1] == 0.0
    assert r.drop_from_peak_db[2] == pytest.approx(5.0)
    assert r.drop_from_peak_db[3] == 0.0
    assert r.summary["max_drop_from_peak_db"] == pytest.approx(5.0)


def test_degrading_forgotten_recovering_stable_sequence():
    pol = HealthPolicy(stable_band_db=1.0, degrading_db=3.0, forgotten_db=5.0,
                       stable_dwell_ticks=2)
    #        est  est  -3.5  -6.0  back up  back in band  band again
    s = mk_series([30.0, 30.2, 26.7, 24.2, 28.0, 29.5, 30.1])
    r = classify_health(s, pol)
    assert r.states[0] == ESTABLISHED
    assert r.states[1] == STABLE
    assert r.states[2] == DEGRADING
    assert r.states[3] == FORGOTTEN
    assert r.states[4] == RECOVERING     # climbed >=1 dB off the trough
    assert r.states[5] == RECOVERING     # episode closes inside the band
    assert r.states[6] == STABLE
    assert len(r.episodes) == 1
    ep = r.episodes[0]
    assert ep["onset_iteration"] == 3000
    assert ep["trough_iteration"] == 4000
    assert ep["trough_drop_db"] == pytest.approx(6.0)
    assert ep["recovered_iteration"] == 6000
    assert ep["reached_forgotten"] is True
    assert 0.0 < ep["recovered_fraction"] <= 1.0
    assert r.summary["recovered_final"] is True


def test_episodes_are_a_list_not_a_scalar_onset():
    """Issue #282's audit: a scalar `fade_start_iter` silently held the LAST
    episode's onset while a real run had 17 episodes. Never a scalar."""
    s = mk_series([30, 30, 25, 30, 30, 24, 30, 30])
    r = classify_health(s)
    assert r.summary["episode_count"] == 2
    assert r.summary["first_onset_iteration"] == 3000
    assert r.summary["last_onset_iteration"] == 6000
    assert r.summary["first_onset_iteration"] != r.summary["last_onset_iteration"]
    assert [e["onset_iteration"] for e in r.episodes] == [3000, 6000]


def test_run_ending_inside_an_episode_is_not_reported_as_recovered():
    """The sticky-`recovered` latch defect from issue #282, refused here."""
    s = mk_series([30.0, 30.0, 22.0, 23.0])
    r = classify_health(s)
    assert r.summary["episode_count"] == 1
    assert r.summary["episode_count_open_at_end"] == 1
    assert r.episodes[0]["recovered_iteration"] is None
    assert r.episodes[0]["recovered_fraction"] is None
    assert r.summary["recovered_final"] is False
    assert r.summary["adverse_at_end"] is True


# ---------------------------------------------------------------------------
# issue #417: RECOVERING must demote on a sub-threshold dip
#
# The band between `stable_band_db` and `degrading_db` is a real but
# sub-threshold dip: too large to count as a clean tick, too small to open an
# episode. STABLE has always demoted to ESTABLISHED there. RECOVERING did not,
# so it could persist indefinitely across ticks oscillating inside that band --
# never completing the stable dwell, never reopening an episode. These four
# tests pin the whole band, including both sides of the exact boundary.
# ---------------------------------------------------------------------------
def _post_episode_series(tail):
    """[peak, peak, DEGRADING, FORGOTTEN, climbing, episode-closes] + ``tail``.

    Running peak is 30.2 (tick 1). After tick 5 the episode has closed and the
    state is RECOVERING with one clean tick banked (dwell needs 2).
    """
    return mk_series([30.0, 30.2, 26.7, 24.2, 28.0, 29.5] + list(tail))


def test_normal_degradation_from_stable():
    """Control: nothing about the #417 fix may change the plain path."""
    pol = HealthPolicy(stable_dwell_ticks=2)
    r = classify_health(mk_series([30.0, 30.0, 30.0, 28.5, 26.5, 24.0]), pol)
    assert r.states[:3] == [ESTABLISHED, STABLE, STABLE]
    assert r.states[3] == ESTABLISHED     # sub-threshold dip (1.5 dB): demoted
    assert r.states[4] == DEGRADING       # 3.5 dB: episode opens
    assert r.states[5] == FORGOTTEN       # 6.0 dB
    assert r.summary["episode_count"] == 1


def test_recovering_with_sustained_recovery_does_not_demote():
    """A genuinely-recovering series must hold RECOVERING through the dwell and
    then reach STABLE. The #417 fix must not shorten this."""
    pol = HealthPolicy(stable_dwell_ticks=3)
    r = classify_health(_post_episode_series([29.6, 30.1, 30.3]), pol)
    assert r.states[5] == RECOVERING      # episode closes here, clean_run = 1
    assert r.states[6] == RECOVERING      # clean_run = 2, dwell not met
    assert r.states[7] == STABLE          # clean_run = 3
    assert r.states[8] == STABLE


def test_recovering_demotes_on_a_sub_threshold_dip():
    """Issue #417. A 1.7 dB dip is inside (stable_band_db, degrading_db): it
    breaks the recovery streak but opens no episode. RECOVERING must not
    survive it -- otherwise a merely-noisy region displays as 'RECOVERING'
    forever, never completing the dwell and never reopening an episode."""
    pol = HealthPolicy(stable_band_db=1.0, degrading_db=3.0, stable_dwell_ticks=2)
    # tick 6 drop = 30.2 - 28.5 = 1.7 dB; ticks 7/8 oscillate in the same band.
    r = classify_health(_post_episode_series([28.5, 29.0, 28.6]), pol)
    assert r.states[5] == RECOVERING
    assert r.states[6] == ESTABLISHED
    assert r.states[7] == ESTABLISHED
    assert r.states[8] == ESTABLISHED
    # The dip is genuinely sub-threshold: no second episode was opened, and the
    # episode record itself is untouched.
    assert r.summary["episode_count"] == 1
    assert r.episodes[0]["recovered_iteration"] == 6000
    assert DEGRADING not in r.states[6:] and FORGOTTEN not in r.states[6:]
    # Recovery is still reachable afterwards: two clean ticks re-promote.
    r2 = classify_health(_post_episode_series([28.5, 30.1, 30.2]), pol)
    assert r2.states[6] == ESTABLISHED
    assert r2.states[7] == ESTABLISHED    # clean_run = 1, dwell not met
    assert r2.states[8] == STABLE         # clean_run = 2


def test_recovering_at_exactly_the_stable_band_boundary_is_still_clean():
    """`drop <= stable_band_db` is the clean-tick predicate, so a drop of
    EXACTLY 1.0 dB is inside the band and must not demote; 1.0 dB + epsilon
    is outside it and must."""
    pol = HealthPolicy(stable_band_db=1.0, degrading_db=3.0, stable_dwell_ticks=2)
    # 29.2 / 29.1 are chosen against the fixture's 30.2 running peak so the
    # drops land exactly ON and just OVER the 1.0 dB band edge.
    at = classify_health(_post_episode_series([29.2]), pol)      # drop == 1.0
    assert at.drop_from_peak_db[6] == pytest.approx(1.0)
    assert at.states[6] == STABLE       # clean_run 1 -> 2, dwell met

    over = classify_health(_post_episode_series([29.1]), pol)    # drop == 1.1
    assert over.drop_from_peak_db[6] == pytest.approx(1.1)
    assert over.states[6] == ESTABLISHED


# ---------------------------------------------------------------------------
# absence of evidence is not a verdict
# ---------------------------------------------------------------------------
def test_unmeasured_tick_freezes_the_state_and_never_reaches_forgotten():
    s = mk_series([30.0, 30.0, np.nan, np.nan, 30.0],
                  measured=[True, True, False, False, True])
    r = classify_health(s)
    assert r.frozen == [False, False, True, True, False]
    assert r.states[2] == r.states[1] == STABLE
    assert r.states[3] == STABLE
    assert FORGOTTEN not in r.states and DEGRADING not in r.states
    assert r.drop_from_peak_db[2] is None
    assert r.summary["n_ticks_frozen_unmeasured"] == 2


def test_measurement_gap_contributes_no_integration_span_at_all():
    """A gap must contribute ZERO iteration-weight -- not "the interpolant".

    Interpolating a 20000-iteration gap between two 10 dB-below-peak samples
    would manufacture a 20000-iteration deficit out of ticks that were never
    measured. That is a verdict from absence, which this module refuses
    everywhere else, so it must not sneak in through the integrals.
    """
    full = classify_health(mk_series([30.0, 30.0, 30.0, 30.0]))
    assert full.summary["psnr_auc_span_iters"] == pytest.approx(3000.0)

    gap = classify_health(mk_series(
        [30.0, 20.0, np.nan, 20.0], iters=[1000, 2000, 21000, 22000],
        measured=[True, True, False, True]))
    assert gap.summary["n_ticks_measured_established"] == 3
    # Measured intervals: [1000,2000] only. [2000,22000] straddles the gap.
    assert gap.summary["psnr_auc_span_iters"] == pytest.approx(1000.0)
    assert gap.summary["iters_below_peak_gt_5db"] == pytest.approx(500.0)
    assert gap.summary["iters_below_peak_gt_5db"] < 20000.0
    # ... and the deficit AUC likewise gets nothing from the gap.
    assert gap.summary["deficit_auc_db_iters"] == pytest.approx(5000.0)


def test_auc_mean_field_names_state_their_own_window():
    """The AUC mean is over each entity's OWN measured span, so two entities in
    one run are not directly comparable. The field name has to say so."""
    r = classify_health(mk_series([30.0, 30.0]))
    assert "psnr_auc_mean_db_over_own_measured_span" in r.summary
    assert "deficit_auc_mean_db_over_own_measured_span" in r.summary
    assert "psnr_auc_span_iters" in r.summary          # the normaliser
    assert "established_first_iteration" in r.summary


def test_inactive_ticks_are_distinct_from_unmeasured_ones():
    s = mk_series([np.nan, 30.0, 30.0], active=[False, True, True],
                  measured=[False, True, True], established=[False, True, True])
    r = classify_health(s)
    assert r.states[0] == NOT_ACTIVE
    assert r.frozen[0] is False


# ---------------------------------------------------------------------------
# THE regression: growth must never be reported as forgetting
# ---------------------------------------------------------------------------
def test_region_growth_is_not_reported_as_forgetting():
    """PR #375's bug, one level up: an established region absorbing a large
    batch of fresh, poor cells must not read as a collapse.

    Built end-to-end through Level B so the guard actually under test is the
    real one (established-cells-only pooling), not a mock.
    """
    good = [cell_key(np.array([i, 0, 0])) for i in range(10)]
    fresh = [cell_key(np.array([i, 0, 0])) for i in range(10, 40)]
    recs = []
    for t, it in enumerate([1000, 2000, 3000, 4000, 5000]):
        cells = {int(k): (30.0, t >= 2, 5000) for k in good}
        if t >= 3:
            # 30 brand-new, badly-reconstructed cells arrive: pure GROWTH.
            cells.update({int(k): (15.0, False, 5000) for k in fresh})
        recs.append(cell_record(it, cells))

    derived = derive_region_history(recs, policy=RegionPolicy(split_min_cells=10_000))
    results = classify_all(region_series(derived))
    assert results, "region series should not be empty"
    for r in results:
        assert r.summary["max_drop_from_peak_db"] < 1.0, (
            f"{r.label} reported a {r.summary['max_drop_from_peak_db']:.2f} dB drop "
            f"from absorbing fresh unestablished cells -- that is growth, not "
            f"forgetting (PR #375's bug)"
        )
        assert not (set(r.states) & ADVERSE_STATES)

    # ... and the equivalent at cell granularity: the fresh cells are simply
    # UNESTABLISHED, never DEGRADING.
    cres = classify_all(cell_series(recs))
    by_id = {r.entity_id: r for r in cres}
    for k in fresh:
        # NOT_ACTIVE before the cell was first observed, UNESTABLISHED after --
        # never DEGRADING/FORGOTTEN.
        assert set(by_id[str(int(k))].states) <= {NOT_ACTIVE, UNESTABLISHED}


def test_confounded_ticks_are_excluded_from_markers_and_worst_k():
    """A growth-confounded row's drop mixes composition change with real change,
    so it must not light a severe-forgetting marker or take a worst-K slot --
    but the verdict itself is still emitted on the row, flagged."""
    bad = mk_series([30.0, 30.0, 20.0, 20.0], entity_id="1",
                    confounded=[False, False, True, True])
    real = mk_series([30.0, 30.0, 24.0, 24.0], entity_id="2")
    series = [bad, real]
    results = classify_all(series)
    assert FORGOTTEN in results[0].states          # verdict still emitted
    assert results[0].summary["n_ticks_confounded"] == 2
    marks = severe_forgetting_marks(results, series)
    flagged = {e for m in marks for e in m["entities"]}
    assert "E1" not in flagged and "E2" in flagged
    assert [r.entity_id for r in worst_k(results, 5)] == ["2"]


# ---------------------------------------------------------------------------
# PSNR-independence (the PR #375 scramble test, adapted)
# ---------------------------------------------------------------------------
def test_scrambled_psnr_leaves_every_psnr_independent_output_identical():
    """PR #375 scrambled every PSNR and asserted the PARTITION came out
    byte-identical. Here the assertion has to be different: states are supposed
    to depend on PSNR, so asserting identical states would be asserting the
    wrong thing (or passing vacuously).

    What must be invariant is everything the metric claims is PSNR-independent:
    the partition, the lineage, the split/merge event stream, the row set and
    row ORDER, and the UNESTABLISHED / NOT_ACTIVE / frozen classification --
    i.e. exactly the fields that decide *what is measured* as opposed to *how
    good it is*.
    """
    rng = np.random.default_rng(20260813)
    keys = [int(cell_key(np.array([i % 8, i // 8, 0]))) for i in range(60)]
    recs = []
    for t, it in enumerate([1000 * (i + 1) for i in range(8)]):
        cells = {}
        for n, k in enumerate(keys):
            if n > 20 and t < 3:
                continue                      # cells arrive over time
            cells[k] = (20.0 + 5.0 * math.sin(n + t), t >= 2, 4000)
        recs.append(cell_record(it, cells))

    scram = json.loads(json.dumps(recs))
    for r in scram:
        for cr in r["cells"].values():
            cr["psnr_db"] = float(rng.uniform(3.0, 45.0))

    def fingerprint(records):
        derived = derive_region_history(records, policy=RegionPolicy())
        rser = region_series(derived)
        cser = cell_series(records)
        rres = classify_all(rser)
        cres = classify_all(cser)
        return {
            "lineage": derived["lineage"],
            "events": partition_event_records(derived),
            "membership": {str(k): v for k, v in derived["membership"].items()},
            "partition_versions": [t["partition_version"] for t in derived["ticks"]],
            "region_ids": [s.entity_id for s in rser],
            "cell_ids": [s.entity_id for s in cser],
            "region_active": [s.active.tolist() for s in rser],
            "cell_active": [s.active.tolist() for s in cser],
            "region_established": [s.established.tolist() for s in rser],
            "cell_established": [s.established.tolist() for s in cser],
            "region_measured": [s.measured.tolist() for s in rser],
            "cell_measured": [s.measured.tolist() for s in cser],
            "region_frozen": [r.frozen for r in rres],
            "cell_frozen": [r.frozen for r in cres],
            "region_unestablished": [[st == UNESTABLISHED for st in r.states] for r in rres],
            "cell_notactive": [[st == NOT_ACTIVE for st in r.states] for r in cres],
            "morton_order": [morton_order_key(int(s.entity_id)) for s in cser],
        }

    a, b = fingerprint(recs), fingerprint(scram)
    assert json.dumps(a, sort_keys=True, default=str) == \
        json.dumps(b, sort_keys=True, default=str)

    # ... and the control: the states themselves DID change, so the test above
    # is not passing vacuously on a no-op scramble.
    orig = classify_all(cell_series(recs))
    scrm = classify_all(cell_series(scram))
    assert [r.states for r in orig] != [r.states for r in scrm]


# ---------------------------------------------------------------------------
# split invariance (Level B's architectural invariant, enforced)
# ---------------------------------------------------------------------------
def test_split_child_state_series_equals_the_machine_run_over_its_own_cells():
    """A split child must inherit NOTHING from its parent.

    Level B re-derives each child's curve over that child's own membership back
    to tick 0; because ``classify_health`` is a pure function of one series, the
    child's states are therefore identical to running the machine over a history
    containing only that child's cells. Asserted directly.
    """
    # Dumbbell: two lobes joined by a 1-cell neck, large enough to trigger a
    # size split, with one lobe degrading.
    lobe_a = [int(cell_key(np.array([i % 5, i // 5, 0]))) for i in range(25)]
    lobe_b = [int(cell_key(np.array([10 + i % 5, i // 5, 0]))) for i in range(25)]
    neck = [int(cell_key(np.array([x, 2, 0]))) for x in range(5, 10)]

    recs = []
    for t, it in enumerate([1000 * (i + 1) for i in range(10)]):
        cells = {}
        for k in lobe_a:
            cells[k] = (30.0 if t < 5 else 21.0, t >= 2, 6000)   # degrades
        for k in lobe_b:
            cells[k] = (30.0, t >= 2, 6000)                      # healthy
        for k in neck:
            cells[k] = (30.0, t >= 2, 6000)
        recs.append(cell_record(it, cells))

    policy = RegionPolicy(split_min_cells=40, split_hysteresis_ticks=2,
                          erosion_min_degree=3, split_min_lobe_cells=5)
    derived = derive_region_history(recs, policy=policy)
    n_split = sum(1 for t in derived["ticks"] for e in t["events"]
                  if e["event"] == "split")
    assert n_split >= 1, "fixture failed to produce a split"

    series = region_series(derived)
    results = {s.entity_id: r for s, r in zip(series, classify_all(series))}
    children = [s for s in series if s.meta.get("parent_ids")]
    assert children, "no split child in the derived history"

    membership = {int(k): set(v) for k, v in derived["membership"].items()}
    for child in children:
        rid = int(child.entity_id)
        own = membership.get(rid)
        if not own:
            continue
        # Re-derive over a history restricted to this child's own cells.
        sub = [dict(r, cells={k: v for k, v in r["cells"].items() if int(k) in own})
               for r in recs]
        sub_derived = derive_region_history(sub, policy=RegionPolicy(
            split_min_cells=10_000, split_hysteresis_ticks=99))
        sub_series = region_series(sub_derived)
        assert len(sub_series) == 1
        sub_res = classify_health(sub_series[0])
        got = results[child.entity_id]
        # Compare only the ticks the child was active for -- before creation it
        # is NOT_ACTIVE by lineage, which the standalone history has no notion of.
        live = [t for t, st in enumerate(got.states) if st != NOT_ACTIVE]
        assert [got.states[t] for t in live] == [sub_res.states[t] for t in live], (
            f"child {child.label} inherited state from its parent"
        )
        assert got.summary["psnr_peak_db"] == pytest.approx(
            sub_res.summary["psnr_peak_db"])


# ---------------------------------------------------------------------------
# iteration-weighting / cadence invariance
# ---------------------------------------------------------------------------
def test_integrals_are_invariant_to_tick_cadence():
    """Ticks are NOT uniformly spaced in this project (checkpoint cadence).
    Inserting a linearly-interpolated tick must not change any integral."""
    iters = [1000, 5000, 9000, 21000]
    psnr = [30.0, 24.0, 22.0, 29.0]
    coarse = classify_health(mk_series(psnr, iters=iters))

    fine_it, fine_p = [], []
    for a in range(len(iters) - 1):
        fine_it.append(iters[a])
        fine_p.append(psnr[a])
        mid = (iters[a] + iters[a + 1]) / 2.0
        fine_it.append(int(mid))
        fine_p.append((psnr[a] + psnr[a + 1]) / 2.0)
    fine_it.append(iters[-1])
    fine_p.append(psnr[-1])
    fine = classify_health(mk_series(fine_p, iters=fine_it))

    for k in ("psnr_auc_db_iters", "psnr_auc_span_iters", "deficit_auc_db_iters",
              "iters_below_peak_gt_1db", "iters_below_peak_gt_3db",
              "iters_below_peak_gt_5db", "max_drop_from_peak_db"):
        assert coarse.summary[k] == pytest.approx(fine.summary[k], rel=1e-9, abs=1e-6), k
    # And the tick COUNT deliberately is NOT invariant -- which is exactly why
    # the iteration-weighted field is the one to quote.
    assert coarse.summary["ticks_below_peak_gt_3db"] != \
        fine.summary["ticks_below_peak_gt_3db"]


def test_time_below_peak_bands_are_nested_and_iteration_weighted():
    # 20 dB below peak for exactly the interval [2000, 3000].
    s = mk_series([30.0, 30.0, 10.0, 30.0], iters=[1000, 2000, 3000, 4000])
    r = classify_health(s)
    for a, b in (("1db", "3db"), ("3db", "5db")):
        assert r.summary[f"iters_below_peak_gt_{a}"] >= \
            r.summary[f"iters_below_peak_gt_{b}"]
    # Triangle down and back up: the >5 dB region is 2 * (1 - 5/20) * 1000.
    assert r.summary["iters_below_peak_gt_5db"] == pytest.approx(1500.0)


def test_recent_slope_uses_an_iteration_window_not_a_tick_count():
    s = mk_series([10.0, 12.0, 14.0, 30.0, 31.0],
                  iters=[1000, 2000, 3000, 50000, 51000])
    r = classify_health(s, HealthPolicy(slope_window_iters=5000.0))
    # Only the last two ticks fall inside a 5000-iteration trailing window.
    assert r.summary["recent_slope_db_per_1k_iters"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# adapters + artifacts
# ---------------------------------------------------------------------------
def test_cell_series_carries_sticky_establishment_and_marks_absence_unmeasured():
    k = int(cell_key(np.array([1, 2, 3])))
    recs = [
        cell_record(1000, {k: (20.0, False, 500)}),
        cell_record(2000, {k: (25.0, True, 500)}),
        cell_record(3000, {}),                       # cell simply not seen
        cell_record(4000, {k: (26.0, True, 500)}),
    ]
    s = cell_series(recs)[0]
    assert s.active.tolist() == [True, True, True, True]
    assert s.established.tolist() == [False, True, True, True]
    assert s.measured.tolist() == [True, True, False, True]
    r = classify_health(s)
    assert r.frozen == [False, False, True, False]


def test_region_series_gives_parent_and_children_disjoint_active_spans():
    lobe_a = [int(cell_key(np.array([i % 5, i // 5, 0]))) for i in range(25)]
    lobe_b = [int(cell_key(np.array([10 + i % 5, i // 5, 0]))) for i in range(25)]
    neck = [int(cell_key(np.array([x, 2, 0]))) for x in range(5, 10)]
    recs = [cell_record(1000 * (t + 1),
                        {k: (30.0, t >= 2, 6000) for k in lobe_a + lobe_b + neck})
            for t in range(8)]
    derived = derive_region_history(recs, policy=RegionPolicy(
        split_min_cells=40, split_hysteresis_ticks=2))
    series = {s.entity_id: s for s in region_series(derived)}
    for s in series.values():
        for child in s.meta.get("child_ids", []):
            cs = series.get(str(int(child)))
            if cs is None:
                continue
            assert not (s.active & cs.active).any(), (
                f"{s.label} and its child {cs.label} are active at the same tick -- "
                f"a single row that changes meaning mid-history"
            )


def test_partition_artifacts_are_written_and_are_pure_projections(tmp_path):
    k = [int(cell_key(np.array([i, 0, 0]))) for i in range(6)]
    recs = [cell_record(1000 * (t + 1), {kk: (28.0, t >= 2, 4000) for kk in k})
            for t in range(5)]
    derived = derive_region_history(recs, policy=RegionPolicy())
    series = region_series(derived)
    results = classify_all(series)

    pe = write_partition_events(str(tmp_path), derived)
    ps = write_partition_snapshot(str(tmp_path), derived, results)
    hh = write_health_history(str(tmp_path), health_records(results, series))
    assert os.path.basename(pe) == "spatial_partition_events.jsonl"
    assert os.path.basename(ps) == "spatial_partition_snapshot.json"
    assert os.path.basename(hh) == "local_quality_health.jsonl"

    evs = [json.loads(l) for l in open(pe)]
    flat = [e for t in derived["ticks"] for e in t["events"]]
    assert len(evs) == len(flat)
    for row, src in zip(evs, flat):
        for kk, vv in src.items():
            assert row[kk] == vv          # pure projection, nothing recomputed
        assert "partition_version" in row and "iteration" in row

    snap = json.load(open(ps))
    assert snap["cell_size_m"] == 0.5
    assert snap["partition_version"] == derived["ticks"][-1]["partition_version"]
    assert snap["regions"] and all("cell_keys" in r for r in snap["regions"])
    assert snap["absent_clocks"]          # named, never silent nulls
    rows = [json.loads(l) for l in open(hh)]
    assert rows and all("policy" in r and "episodes" in r for r in rows)


def test_health_policy_is_echoed_into_every_record():
    s = mk_series([30.0, 30.0])
    pol = HealthPolicy(stable_band_db=0.5, degrading_db=2.0, forgotten_db=4.0)
    r = classify_health(s, pol)
    assert r.summary["policy"]["degrading_db"] == 2.0
    assert "issue #358" in r.summary["policy"]["bands_source"]
    assert "no second criterion" in r.summary["policy"]["establishment_source"]


def test_morton_order_is_spatially_local_and_handles_negative_indices():
    a = int(cell_key(np.array([-3, -3, -3])))
    b = int(cell_key(np.array([-3, -3, -2])))
    far = int(cell_key(np.array([500, 500, 500])))
    assert morton_order_key(a) < morton_order_key(far)
    assert abs(morton_order_key(a) - morton_order_key(b)) < \
        abs(morton_order_key(a) - morton_order_key(far))


def test_series_length_and_ordering_are_validated():
    with pytest.raises(ValueError):
        QualitySeries("1", "E1", np.array([1, 2]), np.array([1.0]),
                      np.array([True, True]), np.array([True, True]),
                      np.array([True, True]), np.array([False, False]))
    with pytest.raises(ValueError, match="strictly increasing"):
        mk_series([1.0, 2.0, 3.0], iters=[1000, 10000, 9000])


def test_pooled_psnr_roundtrip_matches_level_a():
    """The series this module consumes really is Level A's own pooled PSNR."""
    sse = np.array([1.0, 4.0])
    n = np.array([100.0, 100.0])
    got = pooled_psnr_db(sse, n)
    assert got[0] == pytest.approx(20.0)
    assert got[1] == pytest.approx(13.9794, abs=1e-3)
