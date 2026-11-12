"""Tests for #428 Phase 2 -- CVQ, CVQ curves and CVQ-AUC.

Torch-free by construction (numpy only), so this file collects and runs in
the lean CPU CI image as well as in the training image. Fixtures are
synthetic: hand-placed world points, a 1.0 m cell size so every ``cell_key``
is derivable by hand, and hand-written per-cell accumulator vectors standing
in for what ``CellTickAccumulator.finalize()`` returns. No real scene, no
render, no GPU.

Every causality test is written to be **discriminating**: each asserts both
that the causal quantity is unchanged AND that some naive alternative (the
unmasked pooled PSNR, the unweighted tick mean, the mean over all observed
cells) genuinely differs on the same fixture. A test that only asserted the
first would still pass against a stubbed metric.
"""
from __future__ import annotations

import ast
import hashlib
import math
import os

import numpy as np
import pytest

from eligsir.utils.causal_quality import (
    STATUS_DIAGNOSTIC_ONLY,
    STATUS_INSUFFICIENT_CHECKPOINTS,
    STATUS_INSUFFICIENT_SUPPORT,
    STATUS_MEASUREMENT_INCOMPLETE,
    STATUS_NOT_APPLICABLE,
    STATUS_OK,
    CausalQualityError,
    CausalQualityParams,
    CvqCurve,
    MetricValue,
    TickAccounting,
    TickObservation,
    auc_mean_over_span,
    build_cvq_block,
    build_cvq_curve,
    compute_tick_cvq,
    cvq_axis_auc,
    cvq_progress_auc,
    refinement_tail_delta,
    spec_provenance,
)
from eligsir.utils.causal_reference import (
    SPEC_DOC_RELPATH,
    CausalReferenceParams,
    SourceFrame,
    build_causal_reference,
    build_holdout_membership,
    spec_doc_sha256,
)
from eligsir.utils.local_quality import cell_ijk, cell_key

CELL = 1.0
DECLARED = 100

CELL_A = np.array([0.5, 0.5, 0.5])        # room A
CELL_A2 = np.array([1.5, 0.5, 0.5])       # room A, second cell
CELL_B = np.array([20.5, 0.5, 0.5])       # room B -- observed only late
CELL_B2 = np.array([21.5, 0.5, 0.5])


def key_of(point) -> int:
    return int(cell_key(cell_ijk(np.asarray(point, dtype=float).reshape(1, 3), CELL))[0])


K_A, K_A2, K_B, K_B2 = key_of(CELL_A), key_of(CELL_A2), key_of(CELL_B), key_of(CELL_B2)


def make_frame(index, center, cells, n_pix=16, depth=2.0):
    pts = np.repeat(np.asarray(cells, dtype=float).reshape(-1, 3), n_pix, axis=0)
    return SourceFrame(
        index=index,
        frame_id=f"f{index:05d}",
        camera_center=np.asarray(center, dtype=float),
        points_world=pts,
        point_depth=np.full(pts.shape[0], float(depth)),
        point_incidence_deg=np.zeros(pts.shape[0]),
    )


def ref_params(**kw) -> CausalReferenceParams:
    d = dict(cell_size_m=CELL, m_pix=8, d_min=0.3, d_max=5.0, theta_max_deg=75.0,
             K_vis=2, b_min_m=0.10, alpha_min_deg=5.0)
    d.update(kw)
    return CausalReferenceParams(**d)


def build_ref(frames, *, declared=DECLARED, membership=None, realized=None):
    return build_causal_reference(
        frames,
        scene_id="synthetic",
        n_source_frames_declared=declared,
        params=ref_params(),
        reference_pose_source_id="fixture_gt",
        holdout_membership=membership,
        n_source_frames_realized=realized,
    )


def observation(cells, *, p=None, w=None, u=None, iteration=None, accounting=None):
    """``cells`` is ``{cell_key: (sse, n_samples, n_views)}``."""
    keys = list(cells)
    return TickObservation(
        cell_keys=np.asarray(keys, dtype=np.int64),
        sq_err=np.asarray([cells[k][0] for k in keys], dtype=np.float64),
        n_samples=np.asarray([cells[k][1] for k in keys], dtype=np.float64),
        n_views=np.asarray([cells[k][2] for k in keys], dtype=np.float64),
        source_progress=p,
        wall_clock_s=w,
        supervision_views=u,
        iteration=iteration,
        accounting=accounting or TickAccounting(),
    )


def db(sse, n):
    return -10.0 * math.log10(max(sse / n, 1e-12))


# Room A visible from frame 2 (K_vis=2), room B only from frame 8.
FRAMES_A_ONLY = [
    make_frame(1, [0.0, 0.0, -3.0], [CELL_A, CELL_A2]),
    make_frame(2, [3.0, 0.0, -3.0], [CELL_A, CELL_A2]),
    make_frame(3, [6.0, 0.0, -3.0], [CELL_A, CELL_A2]),
]
FRAMES_B_LATER = [
    make_frame(7, [20.0, 0.0, -3.0], [CELL_B, CELL_B2]),
    make_frame(8, [23.0, 0.0, -3.0], [CELL_B, CELL_B2]),
    make_frame(9, [26.0, 0.0, -3.0], [CELL_B, CELL_B2]),
]

GOOD = (30.0, 3000.0, 4)      # ~20 dB
AWFUL = (2700.0, 3000.0, 4)   # ~0.46 dB


def test_fixture_sanity_room_b_is_late():
    ref = build_ref(FRAMES_A_ONLY + FRAMES_B_LATER)
    assert ref.status == "ok"
    assert ref.t_visible_of(K_A) == 2 and ref.p_visible_of(K_A) == pytest.approx(0.02)
    assert ref.t_visible_of(K_B) == 8 and ref.p_visible_of(K_B) == pytest.approx(0.08)


# ---------------------------------------------------------------------------
# PROPERTY 1 -- unseen regions are excluded from the denominator
# ---------------------------------------------------------------------------
def test_unseen_region_excluded_from_both_numerator_and_denominator():
    """Room B is rendered badly but is not yet causally observable.

    Discriminating in three independent ways: the region-balanced support is
    the eligible count (not the observed count); the value equals room A's
    quality *exactly*; and the unmasked pooled PSNR over the same observation
    is far worse, so a stub that ignored eligibility could not pass.
    """
    ref = build_ref(FRAMES_A_ONLY + FRAMES_B_LATER)
    obs = observation(
        {K_A: GOOD, K_A2: GOOD, K_B: AWFUL, K_B2: AWFUL}, p=0.05
    )
    r = compute_tick_cvq(ref, obs)

    assert r.n_eligible == 2 and r.n_measurable == 2
    assert r.n_observed_ineligible == 2
    assert r.region_balanced.support == 2
    assert r.region_balanced.status == STATUS_OK
    assert r.region_balanced.value == pytest.approx(db(*GOOD[:2]), abs=1e-12)
    assert r.pixel_weighted.value == pytest.approx(db(2 * GOOD[0], 2 * GOOD[1]), abs=1e-12)

    # The motivation (spec §15 S1's third REQUIRE): the unmasked number, which
    # is what a non-causal metric would report, is materially different.
    assert r.unmasked_pooled_psnr_db is not None
    assert r.region_balanced.value - r.unmasked_pooled_psnr_db > 5.0

    # ... and the mean over ALL observed cells is different too, so a stub
    # that averaged everything cannot pass either.
    naive = float(np.mean([db(*GOOD[:2])] * 2 + [db(*AWFUL[:2])] * 2))
    assert abs(naive - r.region_balanced.value) > 5.0


def test_never_observed_cell_is_not_in_the_reference_at_all_and_is_counted():
    ref = build_ref(FRAMES_A_ONLY)                       # room B never sourced
    obs = observation({K_A: GOOD, K_A2: GOOD, K_B: AWFUL}, p=1.0)
    r = compute_tick_cvq(ref, obs)
    assert r.n_observed_not_in_reference == 1            # named, never anonymous
    assert r.n_eligible == 2
    assert r.region_balanced.value == pytest.approx(db(*GOOD[:2]), abs=1e-12)


# ---------------------------------------------------------------------------
# PROPERTY 2 -- a newly-visible region enters the denominator exactly per the
#               frozen rules (eligibility boundary AND the §6.3 measurability
#               gate, which are two different gates)
# ---------------------------------------------------------------------------
def test_new_region_enters_at_exactly_p_visible():
    ref = build_ref(FRAMES_A_ONLY + FRAMES_B_LATER)
    p_vis = ref.p_visible_of(K_B)
    assert p_vis == pytest.approx(0.08)

    cells = {K_A: GOOD, K_A2: GOOD, K_B: AWFUL, K_B2: AWFUL}
    before = compute_tick_cvq(ref, observation(cells, p=p_vis - 1e-6))
    at = compute_tick_cvq(ref, observation(cells, p=p_vis))

    # The boundary is inclusive: `eligible` is `p >= p_visible`.
    assert before.n_eligible == 2 and at.n_eligible == 4
    assert before.n_measurable == 2 and at.n_measurable == 4
    # And it really moves the number -- not a no-op boundary.
    assert at.region_balanced.value < before.region_balanced.value - 5.0


def test_measurability_gate_is_separate_from_eligibility():
    """A newly-eligible cell with too little holdout evidence enters E, not M.

    This is the pairing spec §2.3 refuses to collapse: CVQ-P (pooled over
    E) moves, CVQ-R (mean over M) does not.
    """
    params = CausalQualityParams(min_eval_samples=300, min_views=2)
    ref = build_ref(FRAMES_A_ONLY + FRAMES_B_LATER)

    # Distinctly WORSE than GOOD, so entering E must move the pooled number.
    thin_samples = (180.0, 200.0, 4)   # 200 < min_eval_samples
    one_view = (2700.0, 3000.0, 1)     # 1 < min_views

    base = {K_A: GOOD, K_A2: GOOD}
    r0 = compute_tick_cvq(ref, observation(base, p=0.5), params)
    r1 = compute_tick_cvq(
        ref, observation({**base, K_B: thin_samples}, p=0.5), params
    )
    r2 = compute_tick_cvq(
        ref, observation({**base, K_B: one_view}, p=0.5), params
    )

    for r in (r1, r2):
        assert r.n_eligible == 3           # in E
        assert r.n_measurable == 2         # but NOT in M
        assert r.region_balanced.value == pytest.approx(r0.region_balanced.value, abs=1e-12)
        assert r.region_balanced.support == 2
        # CVQ-P pools over E, so it DID move -- proving the cell was really in E
        # and the two gates are not the same gate.
        assert r.pixel_weighted.value != pytest.approx(r0.pixel_weighted.value, abs=1e-9)
        assert r.pixel_weighted.support == 3


def test_eligible_but_no_measurable_cell_is_null_not_zero():
    """Spec §5 R4 C3's precondition can never be produced by this module."""
    params = CausalQualityParams(min_eval_samples=300, min_views=2)
    ref = build_ref(FRAMES_A_ONLY)
    r = compute_tick_cvq(
        ref, observation({K_A: (1.0, 10.0, 1), K_A2: (1.0, 10.0, 1)}, p=0.5), params
    )
    assert r.n_eligible == 2 and r.n_measurable == 0
    assert r.region_balanced.value is None                     # never 0.0
    assert r.region_balanced.status != STATUS_OK
    assert "eligible_but_no_measurable_cell" in r.region_balanced.notes


# ---------------------------------------------------------------------------
# PROPERTY 3 -- an already-visible region stays eligible
# ---------------------------------------------------------------------------
def test_already_visible_region_never_retracts_even_when_unobserved_or_bad():
    """Eligibility is a pure comparison against t_visible.

    Neither a tick at which the cell produced no holdout evidence, nor a tick
    at which its quality collapses, may remove it from the eligible set -- and
    the unobserved tick is *counted*, not silently dropped.
    """
    ref = build_ref(FRAMES_A_ONLY + FRAMES_B_LATER)
    ps = [0.02, 0.10, 0.40, 0.90, 1.0]

    seen = []
    for p in ps:
        assert ref.eligible(K_A, p) is True
        cells = {K_A2: GOOD, K_B: GOOD, K_B2: GOOD}
        if p == 0.40:
            pass                                   # A produced no evidence here
        elif p == 0.90:
            cells[K_A] = AWFUL                     # A collapsed here
        else:
            cells[K_A] = GOOD
        seen.append(compute_tick_cvq(ref, observation(cells, p=p)))

    # Still eligible at every later p, by the reference itself.
    assert all(ref.eligible(K_A, p) for p in ps)
    # The unobserved tick counts A as eligible-but-unobserved rather than
    # dropping it anonymously.
    assert seen[2].n_eligible_unobserved >= 1
    # The collapsed tick still has A in the denominator (its support includes
    # A) and is therefore dragged down -- forgetting is fully penalised.
    assert seen[3].n_measurable == seen[4].n_measurable
    assert seen[3].region_balanced.value < seen[4].region_balanced.value - 3.0


# ---------------------------------------------------------------------------
# PROPERTY 4 -- future frames cannot affect earlier eligibility (causality)
# ---------------------------------------------------------------------------
def test_truncating_the_source_stream_does_not_change_any_earlier_cvq():
    """Spec §15 S1's fourth REQUIRE, at the metric layer.

    The fixture contains a cell that SPANS the cut (observed both before and
    after), so this cannot pass by trivially-separated construction: a change
    that let later frames rewrite an earlier ``t_visible`` would move the
    spanning cell's eligibility and therefore the earlier CVQ.
    """
    spanning = np.array([2.5, 0.5, 0.5])
    k_span = key_of(spanning)
    frames_full = [
        make_frame(1, [0.0, 0.0, -3.0], [CELL_A, spanning]),
        make_frame(2, [3.0, 0.0, -3.0], [CELL_A, spanning]),
        make_frame(3, [6.0, 0.0, -3.0], [CELL_A]),
        make_frame(9, [20.0, 0.0, -3.0], [CELL_B, spanning]),
        make_frame(10, [23.0, 0.0, -3.0], [CELL_B, spanning]),
    ]
    ref_full = build_ref(frames_full)
    ref_trunc = build_ref(frames_full[:3], realized=3)

    assert ref_full.t_visible_of(k_span) == ref_trunc.t_visible_of(k_span) == 2
    assert ref_full.n_src_of(k_span) > ref_trunc.n_src_of(k_span)   # support did grow

    obs = observation({K_A: GOOD, k_span: (500.0, 3000.0, 3), K_B: AWFUL}, p=0.05)
    a = compute_tick_cvq(ref_full, obs)
    b = compute_tick_cvq(ref_trunc, obs)

    # EXACT equality (spec §15 S1 says "exactly"), not approx.
    assert a.region_balanced.value == b.region_balanced.value
    assert a.pixel_weighted.value == b.pixel_weighted.value
    assert (a.n_eligible, a.n_measurable) == (b.n_eligible, b.n_measurable)
    # Discriminating: the two references are genuinely different objects,
    # differing in cell count and in what is eligible LATER.
    assert ref_full.n_cells != ref_trunc.n_cells
    late_full = compute_tick_cvq(ref_full, obs.__class__(**{
        **{k: getattr(obs, k) for k in
           ("cell_keys", "sq_err", "n_samples", "n_views")},
        "source_progress": 0.5,
    }))
    late_trunc = compute_tick_cvq(ref_trunc, obs.__class__(**{
        **{k: getattr(obs, k) for k in
           ("cell_keys", "sq_err", "n_samples", "n_views")},
        "source_progress": 0.5,
    }))
    assert late_full.n_eligible > late_trunc.n_eligible
    assert late_full.region_balanced.value != late_trunc.region_balanced.value


# ---------------------------------------------------------------------------
# PROPERTY 5 -- identical renders + different FUTURE scene extent => identical
#               past CVQ (spec §15 S1)
# ---------------------------------------------------------------------------
def test_identical_renders_different_future_extent_give_identical_past_cvq():
    """Run A never visits room B; run B visits it later. Same render, same CVQ.

    The observation is byte-identical between the two arms (it is the same
    object), and the holdout sees room B either way -- so the ONLY difference
    is the future extent of the source stream.
    """
    membership = build_holdout_membership(
        [("h0", np.stack([CELL_A, CELL_A2, CELL_B, CELL_B2]))], cell_size_m=CELL
    )
    ref_a = build_ref(FRAMES_A_ONLY, membership=membership)
    ref_b = build_ref(FRAMES_A_ONLY + FRAMES_B_LATER, membership=membership)

    obs = observation({K_A: GOOD, K_A2: GOOD, K_B: AWFUL, K_B2: AWFUL}, p=0.05)
    ra = compute_tick_cvq(ref_a, obs)
    rb = compute_tick_cvq(ref_b, obs)

    assert ra.region_balanced.value == rb.region_balanced.value      # exact
    assert ra.pixel_weighted.value == rb.pixel_weighted.value        # exact
    assert ra.n_measurable == rb.n_measurable == 2

    # Discriminating #1: the two references really do differ.
    assert ref_a.eligibility_hash != ref_b.eligibility_hash
    assert ref_a.n_cells < ref_b.n_cells
    # Discriminating #2: past p is what makes them agree -- at a LATER p they
    # diverge, so the agreement above is not vacuous.
    late = observation({K_A: GOOD, K_A2: GOOD, K_B: AWFUL, K_B2: AWFUL}, p=0.5)
    assert (
        compute_tick_cvq(ref_a, late).region_balanced.value
        != compute_tick_cvq(ref_b, late).region_balanced.value
    )
    # Discriminating #3: the non-causal number does NOT have this property --
    # it is identical between the arms only because it ignores causality, and
    # it differs from CVQ by a wide margin (the motivation).
    assert ra.unmasked_pooled_psnr_db == rb.unmasked_pooled_psnr_db
    assert ra.region_balanced.value - ra.unmasked_pooled_psnr_db > 5.0


def test_holdout_membership_absent_does_not_shrink_the_denominator():
    ref = build_ref(FRAMES_A_ONLY)                     # no membership built
    assert ref.holdout_resolvable is None
    r = compute_tick_cvq(ref, observation({K_A: GOOD, K_A2: GOOD}, p=0.5))
    assert r.n_eligible == 2
    assert "holdout_resolvable_unavailable" in r.notes


def test_unresolvable_cells_are_excluded_and_counted():
    membership = build_holdout_membership(
        [("h0", np.stack([CELL_A]))], cell_size_m=CELL      # A2 never resolvable
    )
    ref = build_ref(FRAMES_A_ONLY, membership=membership)
    r = compute_tick_cvq(ref, observation({K_A: GOOD, K_A2: AWFUL}, p=0.5))
    assert r.n_eligible == 1
    assert r.n_observed_unresolvable == 1
    assert r.region_balanced.value == pytest.approx(db(*GOOD[:2]), abs=1e-12)


# ---------------------------------------------------------------------------
# AUC over IRREGULAR checkpoints
# ---------------------------------------------------------------------------
def test_auc_is_axis_weighted_not_an_unweighted_tick_mean():
    """Clustered sampling of a known ramp: trapezoid == analytic mean.

    Deliberately clustered rather than uniform, and the naive unweighted mean
    of the same values is asserted to DIFFER -- a uniform fixture would pass
    for an implementation that just averaged the tick values.
    """
    axis = [0.0, 0.05, 0.10, 0.15, 0.20, 1.0]      # 5 clustered + 1 far tick
    vals = [10.0 + 10.0 * a for a in axis]         # exact linear ramp
    st = [STATUS_OK] * len(axis)

    mv = auc_mean_over_span(axis, vals, st, min_checkpoints=4)
    assert mv.status == STATUS_OK
    assert mv.value == pytest.approx(15.0, abs=1e-12)          # mean of 10..20
    assert mv.extra["covered_span"] == pytest.approx(1.0)
    assert mv.extra["interval"] == [0.0, 1.0]

    naive = float(np.mean(vals))
    assert abs(naive - mv.value) > 2.0                          # 11.5 vs 15.0


def test_auc_matches_between_uniform_and_irregular_sampling_of_the_same_curve():
    f = lambda a: 12.0 + 8.0 * a
    uniform = [i / 10.0 for i in range(11)]
    irregular = [0.0, 0.01, 0.02, 0.03, 0.5, 0.97, 0.99, 1.0]
    a = auc_mean_over_span(uniform, [f(x) for x in uniform], [STATUS_OK] * len(uniform))
    b = auc_mean_over_span(
        irregular, [f(x) for x in irregular], [STATUS_OK] * len(irregular)
    )
    assert a.value == pytest.approx(b.value, abs=1e-9)


def test_measurement_gap_is_excluded_from_both_integral_and_span():
    """Spec §3.5's bridge rule: a gap is not interpolated and not counted."""
    axis = [0.0, 0.1, 0.2, 0.5, 0.6, 0.7]
    vals = [20.0, 20.0, 20.0, None, 10.0, 10.0]
    st = [STATUS_OK, STATUS_OK, STATUS_OK, STATUS_INSUFFICIENT_SUPPORT,
          STATUS_OK, STATUS_OK]
    mv = auc_mean_over_span(axis, vals, st, min_checkpoints=4)
    # Covered span is 0.2 (0.0->0.2) + 0.1 (0.6->0.7); the 0.2->0.5->0.6 gap
    # contributes nothing to either the integral or the span.
    assert mv.extra["covered_span"] == pytest.approx(0.3)
    assert mv.extra["n_segments"] == 3
    assert mv.value == pytest.approx((20.0 * 0.2 + 10.0 * 0.1) / 0.3)
    # Bridging the gap would give a different (larger span) answer.
    bridged = auc_mean_over_span(
        axis, [20.0, 20.0, 20.0, 15.0, 10.0, 10.0], [STATUS_OK] * 6
    )
    assert bridged.extra["covered_span"] == pytest.approx(0.7)
    assert bridged.value != pytest.approx(mv.value)


def test_all_measured_ticks_isolated_is_zero_span_not_insufficient_checkpoints():
    """A reachable state on sparse-tick runs that the two statuses confuse."""
    axis = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    vals = [20.0, None, 20.0, None, 20.0, None, 20.0, None]
    st = [STATUS_OK, STATUS_INSUFFICIENT_SUPPORT] * 4
    mv = auc_mean_over_span(axis, vals, st, min_checkpoints=4)
    assert mv.value is None
    assert mv.status == STATUS_INSUFFICIENT_SUPPORT
    assert "zero_covered_span" in mv.notes
    assert mv.status != STATUS_INSUFFICIENT_CHECKPOINTS       # the wrong label
    assert mv.extra["n_measured_ticks"] == 4                  # count was fine


def test_too_few_measured_ticks_is_insufficient_checkpoints():
    mv = auc_mean_over_span([0.0, 0.5, 1.0], [10.0, 11.0, 12.0], [STATUS_OK] * 3)
    assert mv.value is None and mv.status == STATUS_INSUFFICIENT_CHECKPOINTS
    assert mv.support == 0


def test_auc_refuses_misaligned_arrays():
    with pytest.raises(CausalQualityError):
        auc_mean_over_span([0.0, 1.0], [1.0], [STATUS_OK, STATUS_OK])


def test_ticks_out_of_axis_order_are_sorted_and_flagged():
    axis = [0.0, 0.5, 0.25, 1.0, 0.75]
    f = lambda a: 10.0 + 10.0 * a
    mv = auc_mean_over_span(axis, [f(a) for a in axis], [STATUS_OK] * 5)
    assert mv.value == pytest.approx(15.0, abs=1e-12)
    assert "axis_not_monotone_in_tick_order" in mv.notes


# ---------------------------------------------------------------------------
# Curves + the progress/wall-clock/work axes
# ---------------------------------------------------------------------------
def _curve_of(ps, values, ws=None, us=None):
    n = len(ps)
    ws = ws if ws is not None else [None] * n
    us = us if us is not None else [None] * n
    st = [STATUS_OK if v is not None else STATUS_INSUFFICIENT_SUPPORT for v in values]
    return CvqCurve(
        axis_progress=tuple(ps),
        axis_wall_clock_s=tuple(ws),
        axis_supervision_views=tuple(us),
        iteration=tuple(range(n)),
        region_balanced_db=tuple(values),
        pixel_weighted_db=tuple(values),
        status=tuple(st),
        status_pixel_weighted=tuple(st),
        support_region_balanced=tuple(1 if v is not None else 0 for v in values),
        support_pixel_weighted=tuple(1 if v is not None else 0 for v in values),
        n_eligible=tuple([1] * n),
        n_measurable=tuple([1] * n),
        n_eligible_unobserved=tuple([0] * n),
        camera_coverage_fraction=tuple([1.0] * n),
        unmasked_pooled_psnr_db=tuple(values),
    )


def test_curve_refuses_misaligned_columns_invariant_I0():
    with pytest.raises(CausalQualityError):
        CvqCurve(
            axis_progress=(0.0, 1.0), axis_wall_clock_s=(0.0,),
            axis_supervision_views=(0.0, 1.0), iteration=(0, 1),
            region_balanced_db=(1.0, 2.0), pixel_weighted_db=(1.0, 2.0),
            status=(STATUS_OK, STATUS_OK),
            status_pixel_weighted=(STATUS_OK, STATUS_OK),
            support_region_balanced=(1, 1), support_pixel_weighted=(1, 1),
            n_eligible=(1, 1), n_measurable=(1, 1), n_eligible_unobserved=(0, 0),
            camera_coverage_fraction=(1.0, 1.0), unmasked_pooled_psnr_db=(1.0, 2.0),
        )


def test_incomplete_run_cannot_emit_an_ok_span_normalized_auc():
    """Spec §12.2 rule 2 / §12.1's gaming vector."""
    curve = _curve_of([0.0, 0.1, 0.2, 0.3, 0.5], [10.0, 12.0, 14.0, 16.0, 18.0])
    out = cvq_progress_auc(curve, covered_progress_fraction=0.5)
    full = out["region_balanced_progress_mean_over_span"]
    prefix = out["region_balanced_progress_prefix"]
    assert full.value is None and full.status != STATUS_OK
    assert "incomplete_run" in full.notes
    assert prefix.value is not None and prefix.status == STATUS_OK
    assert prefix.extra["prefix_end_progress"] == pytest.approx(0.5)
    assert "comparable_only_against_matched_prefix" in prefix.notes

    complete = cvq_progress_auc(curve, covered_progress_fraction=1.0)
    assert complete["region_balanced_progress_mean_over_span"].status == STATUS_OK


def test_progress_denominator_mismatch_is_noted():
    curve = _curve_of([0.0, 0.1, 0.2, 0.3], [10.0, 12.0, 14.0, 16.0])
    out = cvq_progress_auc(curve, progress_denominator_ok=False)
    assert "progress_denominator_mismatch" in out[
        "region_balanced_progress_mean_over_span"
    ].notes


def test_work_axis_is_forced_diagnostic_only_and_wallclock_notes_its_host():
    curve = _curve_of(
        [0.0, 0.25, 0.5, 1.0], [10.0, 12.0, 14.0, 16.0],
        ws=[0.0, 10.0, 20.0, 40.0], us=[0.0, 100.0, 200.0, 400.0],
    )
    work = cvq_axis_auc(curve, "supervision_views")
    assert work.value is not None and work.status == STATUS_DIAGNOSTIC_ONLY

    wall = cvq_axis_auc(curve, "wall_clock_s")
    assert wall.status == STATUS_OK
    assert "same_host_only" in wall.notes and "host_fingerprint_absent" in wall.notes
    wall2 = cvq_axis_auc(curve, "wall_clock_s", host_fingerprint={"host": "amd1"})
    assert wall2.extra["host_fingerprint"] == {"host": "amd1"}
    assert "host_fingerprint_absent" not in wall2.notes

    with pytest.raises(CausalQualityError):
        cvq_axis_auc(curve, "progress")


def test_refinement_tail_is_reported_separately_and_refused_when_undefined():
    curve = _curve_of([0.2, 0.6, 1.0, 1.0], [10.0, 14.0, 18.0, 22.0])
    d = refinement_tail_delta(curve)
    assert d.value == pytest.approx(4.0)

    truncated = _curve_of([0.1, 0.2, 0.3, 0.4], [10.0, 12.0, 14.0, 16.0])
    assert refinement_tail_delta(truncated).status == STATUS_NOT_APPLICABLE
    assert "mapping_end_not_reached" in refinement_tail_delta(truncated).notes


# ---------------------------------------------------------------------------
# No-silent-zero discipline (spec §5 R1/R2/R3/R5)
# ---------------------------------------------------------------------------
def test_metric_value_refuses_invariant_violations():
    with pytest.raises(CausalQualityError):                    # I1
        MetricValue(0.0, "dB", 0, STATUS_OK)
    with pytest.raises(CausalQualityError):                    # I1
        MetricValue(None, "dB", 0, STATUS_OK)
    with pytest.raises(CausalQualityError):                    # I2
        MetricValue(None, "dB", 5, STATUS_OK)
    with pytest.raises(CausalQualityError):                    # closed enum
        MetricValue(1.0, "dB", 1, "totally_fine")
    with pytest.raises(CausalQualityError):                    # non-finite
        MetricValue(float("inf"), "dB", 1, STATUS_OK)
    ok = MetricValue(1.0, "dB", 1, STATUS_OK)
    assert ok.as_dict()["status"] == STATUS_OK


def test_accounting_refuses_impossible_arithmetic_I4_I5():
    with pytest.raises(CausalQualityError):
        TickAccounting(atoms_attempted=10, atoms_succeeded=9, atoms_failed_error=0)
    with pytest.raises(CausalQualityError):
        TickAccounting(views_attempted=4, views_succeeded=3)
    ok = TickAccounting(
        atoms_attempted=10, atoms_succeeded=9, atoms_failed_error=1,
        views_attempted=4, views_succeeded=3, views_skipped_render_failure=1,
    )
    assert ok.camera_coverage_fraction == pytest.approx(0.75)
    assert ok.partial is True


def test_partial_camera_coverage_downgrades_the_tick_status():
    ref = build_ref(FRAMES_A_ONLY)
    acc = TickAccounting(
        views_attempted=4, views_succeeded=3, views_skipped_render_failure=1
    )
    r = compute_tick_cvq(
        ref, observation({K_A: GOOD, K_A2: GOOD}, p=0.5, accounting=acc)
    )
    assert r.region_balanced.value is not None
    assert r.region_balanced.status == STATUS_MEASUREMENT_INCOMPLETE
    assert "measurement_incomplete" in r.region_balanced.notes
    assert r.camera_coverage_fraction == pytest.approx(0.75)


def test_tick_without_a_progress_coordinate_is_unmeasured_not_zero():
    ref = build_ref(FRAMES_A_ONLY)
    r = compute_tick_cvq(ref, observation({K_A: GOOD}, p=None))
    assert r.region_balanced.value is None
    assert r.region_balanced.status != STATUS_OK
    assert "source_progress_unavailable" in r.region_balanced.notes


def test_duplicate_cell_keys_in_one_tick_are_refused():
    with pytest.raises(CausalQualityError):
        TickObservation(
            cell_keys=np.asarray([K_A, K_A], dtype=np.int64),
            sq_err=np.asarray([1.0, 1.0]),
            n_samples=np.asarray([10.0, 10.0]),
            n_views=np.asarray([1.0, 1.0]),
            source_progress=0.5,
        )


def test_join_is_by_key_not_by_row_position():
    """Shuffling the observation's row order changes nothing, bit-for-bit."""
    ref = build_ref(FRAMES_A_ONLY + FRAMES_B_LATER)
    cells = {K_A: GOOD, K_A2: (17.0, 3000.0, 3), K_B: AWFUL, K_B2: (5.0, 900.0, 2)}
    a = compute_tick_cvq(ref, observation(cells, p=0.5))
    shuffled = {k: cells[k] for k in reversed(list(cells))}
    b = compute_tick_cvq(ref, observation(shuffled, p=0.5))
    assert a.region_balanced.value == b.region_balanced.value
    assert a.pixel_weighted.value == b.pixel_weighted.value


# ---------------------------------------------------------------------------
# Spec provenance (#428 hard requirement)
# ---------------------------------------------------------------------------
def test_spec_provenance_carries_version_and_a_verifiable_doc_hash():
    prov = spec_provenance()
    for k in (
        "spec_id", "spec_version", "spec_version_minor",
        "spec_doc_relpath", "spec_doc_sha256", "spec_doc_sha256_available",
    ):
        assert k in prov, k
    assert prov["spec_id"] == "CAUSAL_ONLINE_MAPPING_METRICS"
    assert prov["spec_version"] == "1.1.0-draft"
    assert prov["spec_version_minor"] == "1.1"
    assert prov["spec_doc_relpath"] == SPEC_DOC_RELPATH

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    doc = os.path.join(repo_root, SPEC_DOC_RELPATH)
    if os.path.isfile(doc):
        with open(doc, "rb") as fh:
            expected = hashlib.sha256(fh.read()).hexdigest()
        assert prov["spec_doc_sha256"] == expected
        assert prov["spec_doc_sha256_available"] is True
    else:
        # The lean CPU CI image deliberately does not copy docs/paper/. A
        # missing doc must degrade to a recorded null, never a crash and never
        # a silently absent field.
        assert prov["spec_doc_sha256"] is None
        assert prov["spec_doc_sha256_available"] is False
        assert spec_doc_sha256() is None


def test_emitted_block_records_the_spec_and_the_frozen_parameters():
    ref = build_ref(FRAMES_A_ONLY + FRAMES_B_LATER)
    ticks = [
        compute_tick_cvq(ref, observation({K_A: GOOD, K_A2: GOOD}, p=p, w=10.0 * i,
                                          u=100.0 * i, iteration=100 * i))
        for i, p in enumerate([0.1, 0.3, 0.6, 1.0])
    ]
    block = build_cvq_block(ref, ticks, covered_progress_fraction=1.0)

    assert block["spec"]["spec_version"] == "1.1.0-draft"
    assert block["spec"]["spec_id"] == "CAUSAL_ONLINE_MAPPING_METRICS"
    assert "spec_doc_sha256" in block["spec"]
    assert block["parameters"] == CausalQualityParams().as_dict()
    assert block["identity"]["eligibility_hash"] == ref.eligibility_hash

    auc = block["auc"]["region_balanced_progress_mean_over_span"]
    assert auc["status"] == STATUS_OK and auc["value"] is not None
    # SSIM/LPIPS are DEFERRED, and say so explicitly rather than being absent.
    for k in ("ssim_region_balanced", "lpips_fully_eligible_cameras"):
        assert block["final"][k]["value"] is None
        assert block["final"][k]["status"] == "not_computed"
        assert "deferred_issue428_phase2_scope" in block["final"][k]["notes"]

    # I0 holds across every curve column.
    lengths = {k: len(v) for k, v in block["curve"].items()}
    assert len(set(lengths.values())) == 1


def test_curve_columns_stay_parallel_when_ticks_are_unmeasured():
    ref = build_ref(FRAMES_A_ONLY)
    ticks = [
        compute_tick_cvq(ref, observation({K_A: GOOD, K_A2: GOOD}, p=0.1)),
        compute_tick_cvq(ref, observation({K_A: (1.0, 10.0, 1)}, p=0.2)),  # unmeasured
        compute_tick_cvq(ref, observation({K_A: GOOD, K_A2: GOOD}, p=0.3)),
    ]
    curve = build_cvq_curve(ticks)
    assert len(curve) == 3
    assert curve.region_balanced_db[1] is None          # null, never 0.0
    assert curve.status[1] != STATUS_OK


# ---------------------------------------------------------------------------
# Collectability on the lean CPU CI surface
# ---------------------------------------------------------------------------
def test_module_has_no_module_scope_torch_import():
    """Guards the surface this file's own tests run on.

    A module-scope ``import torch`` in ``utils/causal_quality.py`` would make
    this whole test file uncollectable in the lean CPU CI image (#518's AST
    scanner would exclude it), so it is asserted structurally rather than
    assumed.
    """
    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "utils", "causal_quality.py",
    )
    with open(path, "r", encoding="utf-8") as fh:
        tree = ast.parse(fh.read())
    for node in tree.body:                       # module scope only
        if isinstance(node, ast.Import):
            assert all(not a.name.split(".")[0] == "torch" for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert (node.module or "").split(".")[0] != "torch"


# ---------------------------------------------------------------------------
# R6 -- a not_computed causal reference can never render as a measurement
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "frames,expected_reason",
    [
        ([], "no_source_frames_observed"),
        # Every cell seen from ONE viewpoint only -> nothing reaches K_vis=2.
        ([make_frame(1, [0.0, 0.0, -3.0], [CELL_A, CELL_A2])], "no_cell_reached_K_vis"),
    ],
)
def test_not_computed_reference_propagates_and_never_reads_as_measured(
    frames, expected_reason
):
    """Two distinct Phase 1 failures, both of which used to be mis-rendered.

    The empty-stream case additionally CRASHED the key join with an IndexError
    (the reference's cell array is empty); the K_vis case silently produced a
    clean-looking ``insufficient_support``, i.e. a not-computed *input*
    reading as a measured-nothing *output* -- exactly the class §5 exists to
    prevent.
    """
    ref = build_ref(frames)
    assert ref.status == "not_computed" and ref.reason == expected_reason

    r = compute_tick_cvq(ref, observation({K_A: GOOD, K_A2: GOOD}, p=0.5))
    for mv in (r.region_balanced, r.pixel_weighted):
        assert mv.value is None
        assert mv.status == "not_computed"            # not insufficient_support
        assert expected_reason in mv.notes[0]
    assert r.unmasked_pooled_psnr_db is None

    # R6: the WHOLE block, not a partial one -- no metric values at all.
    block = build_cvq_block(ref, [r])
    assert block["status"] == "not_computed"
    assert expected_reason in block["reason"]
    assert block["missing_inputs"] == ["causal_reference"]
    for k in ("cvq", "curve", "auc", "final"):
        assert k not in block
    assert block["spec"]["spec_version"] == "1.1.0-draft"     # provenance kept


def test_ok_reference_block_declares_its_status():
    ref = build_ref(FRAMES_A_ONLY)
    ticks = [
        compute_tick_cvq(ref, observation({K_A: GOOD, K_A2: GOOD}, p=p))
        for p in (0.1, 0.3, 0.6, 1.0)
    ]
    block = build_cvq_block(ref, ticks, covered_progress_fraction=1.0)
    assert block["status"] == STATUS_OK and block["reason"] is None


# ---------------------------------------------------------------------------
# §5 R5 -- partial ticks stay IN the integral, and say so
# ---------------------------------------------------------------------------
def test_partial_ticks_contribute_to_the_auc_and_degrade_its_status():
    """R5 excludes a partial tick from CROSS-RUN comparison, not from its own
    integral. Dropping it here would silently null the headline AUC of any run
    where one holdout camera fails at every tick.
    """
    axis = [0.0, 0.25, 0.5, 1.0]
    vals = [10.0, 12.0, 14.0, 16.0]
    all_ok = auc_mean_over_span(axis, vals, [STATUS_OK] * 4)
    partial = auc_mean_over_span(
        axis, vals, [STATUS_OK, STATUS_MEASUREMENT_INCOMPLETE, STATUS_OK, STATUS_OK]
    )
    assert partial.value == pytest.approx(all_ok.value)          # same integral
    assert partial.extra["covered_span"] == all_ok.extra["covered_span"]
    assert partial.status == STATUS_MEASUREMENT_INCOMPLETE       # but degraded
    assert "measurement_incomplete" in partial.notes

    every_tick_partial = auc_mean_over_span(
        axis, vals, [STATUS_MEASUREMENT_INCOMPLETE] * 4
    )
    assert every_tick_partial.value is not None                  # not nulled
    assert every_tick_partial.status == STATUS_MEASUREMENT_INCOMPLETE


# ---------------------------------------------------------------------------
# §3.4 -- refining longer offline cannot move the progress-axis AUC
# ---------------------------------------------------------------------------
def test_refinement_tail_cannot_inflate_the_progress_auc():
    """The central anti-gaming property of the headline scalar.

    p == 1.0 at the last ingest by definition, so every refinement-tail tick
    sits at the same axis coordinate and spans zero width. An implementation
    that folded the tail into the integral would fail this loudly.
    """
    mapping = _curve_of([0.2, 0.5, 0.8, 1.0], [10.0, 12.0, 14.0, 16.0])
    with_tail = _curve_of(
        [0.2, 0.5, 0.8, 1.0, 1.0, 1.0, 1.0, 1.0],
        [10.0, 12.0, 14.0, 16.0, 26.0, 26.0, 26.0, 26.0],
    )
    a = cvq_progress_auc(mapping, covered_progress_fraction=1.0)[
        "region_balanced_progress_mean_over_span"
    ]
    b = cvq_progress_auc(with_tail, covered_progress_fraction=1.0)[
        "region_balanced_progress_mean_over_span"
    ]
    assert b.value == a.value                          # bit-identical
    assert b.extra["interval"] == [0.2, 1.0]
    # The tail is real and large -- it is reported separately, not folded in.
    assert refinement_tail_delta(with_tail).value == pytest.approx(10.0)
