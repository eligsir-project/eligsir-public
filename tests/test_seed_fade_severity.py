"""Tests for the seed-fade severity verdict layer (issue #346).

The two calibration anchors are embedded as literals rather than read from
``/data/...`` on purpose: that mount does not exist on
the r9700 satellite machine or in CI, and these two curves ARE the calibration
-- pinning them here is what stops a future threshold edit from silently
regrading the case this issue was opened about.

Anchor A: ``issue278mode_iterbased_spf20`` -- seed max_dip_db=15.56, test PSNR
pinned ~9dB for 75,000 of 100,000 iterations, then a late spike to 21.53.
Anchor B: ``fr3postmerge_champ_asis`` -- seed max_dip_db=2.93 ("a routine
blip"), test PSNR pinned ~9.2dB for ~23,000 of 50,000 iterations, then a spike
to 21.91.

Both booleans-identical in the detector (``detected/recovered/
controller_engaged`` all True on both). The whole point of this layer is that
they must NOT come out with the same verdict.
"""

from __future__ import annotations

import json
import os

from eligsir.utils.seed_fade_severity import (
    classify_curve_shape,
    grade_run_dir,
    grade_seed_fade,
    grade_seed_fade_for_run,
    longest_low_plateau,
    max_drawdown_db,
)

# --- real calibration anchors (verbatim psnr_history.jsonl, 1000-iter ticks) --

ANCHOR_A_VALS = [
    8.94, 8.82, 9.04, 8.88, 8.83, 9.1, 9.35, 9.55, 9.65, 9.69, 9.72, 9.71,
    9.66, 9.69, 9.45, 9.28, 9.18, 9.13, 9.04, 8.9, 8.91, 8.98, 9.01, 8.95,
    8.98, 9.02, 9.05, 9.05, 8.99, 9.05, 9.12, 9.14, 9.13, 9.15, 9.15, 9.1,
    9.07, 9.06, 9.13, 9.12, 9.12, 9.15, 9.16, 9.16, 9.18, 9.19, 9.16, 9.13,
    9.12, 9.16, 9.14, 9.14, 9.16, 9.15, 9.16, 9.13, 9.11, 9.09, 9.1, 9.07,
    9.07, 9.06, 9.06, 9.05, 9.05, 9.02, 9.01, 9.01, 9.01, 9.01, 9.0, 8.99,
    8.99, 8.99, 9.0, 14.74, 16.8, 17.95, 18.59, 19.15, 19.39, 19.7, 20.06,
    20.22, 20.25, 20.58, 20.67, 20.83, 20.89, 20.99, 20.99, 21.2, 21.25,
    21.18, 21.33, 21.42, 21.41, 21.4, 21.49, 21.53,
]
ANCHOR_A_ITERS = [1000 * (i + 1) for i in range(len(ANCHOR_A_VALS))]
ANCHOR_A_SEED_FADE = {
    "detected": True, "max_dip_db": 15.56, "fade_start_iter": 21500,
    "recovered": True, "controller_engaged": True,
}

ANCHOR_B_VALS = [
    11.21, 9.8, 9.41, 9.28, 9.22, 8.96, 9.05, 9.08, 9.11, 9.1, 9.1, 9.17,
    9.2, 9.24, 9.22, 9.24, 9.24, 9.24, 9.2, 9.23, 9.22, 9.2, 9.19, 9.21,
    9.19, 9.21, 16.16, 17.97, 18.87, 19.36, 19.85, 20.15, 20.36, 20.68,
    20.82, 20.87, 20.93, 21.12, 21.41, 21.44, 21.41, 21.29, 21.48, 21.61,
    21.65, 21.76, 21.74, 21.78, 21.85, 21.89, 21.91,
]
ANCHOR_B_ITERS = [1000 * (i + 1) for i in range(len(ANCHOR_B_VALS))]
ANCHOR_B_SEED_FADE = {
    "detected": True, "max_dip_db": 2.93, "fade_start_iter": 28900,
    "recovered": True, "controller_engaged": True,
}

# --- synthetic negative control: a clean, steadily-climbing run --------------

STEADY_CLIMB_VALS = [12.0 + 14.0 * (i / 49.0) + (0.05 if i % 2 else -0.05) for i in range(50)]
STEADY_CLIMB_ITERS = [1000 * (i + 1) for i in range(50)]


# =============================== the headline ===============================


def test_anchors_do_not_grade_the_same():
    """The defect issue #346 names, stated as an assertion."""
    a = grade_seed_fade(ANCHOR_A_SEED_FADE, ANCHOR_A_ITERS, ANCHOR_A_VALS)
    b = grade_seed_fade(ANCHOR_B_SEED_FADE, ANCHOR_B_ITERS, ANCHOR_B_VALS)
    # Detector booleans really are identical -- that is the premise.
    for key in ("detected", "recovered", "controller_engaged"):
        assert a[key] == b[key] is True
    assert a["severity"] != b["severity"]


def test_anchor_a_is_severe():
    a = grade_seed_fade(ANCHOR_A_SEED_FADE, ANCHOR_A_ITERS, ANCHOR_A_VALS)
    assert a["severity"] == "severe"
    assert a["confounded"] is True
    assert a["curve_shape"] == "low_plateau_then_rise"
    # ~75% of the run pinned low.
    assert a["low_plateau_iter_frac"] > 0.7
    assert a["low_plateau_deficit_db"] > 10.0


def test_anchor_b_is_not_severe_but_is_escalated_off_minor():
    """B's 2.93dB dip alone is a blip (base=minor), but its curve spent ~46%
    of the run pinned ~12.7dB below its final value -- that shape escalates it
    to moderate. Distinct from A on both counts."""
    b = grade_seed_fade(ANCHOR_B_SEED_FADE, ANCHOR_B_ITERS, ANCHOR_B_VALS)
    assert b["severity"] == "moderate"
    assert b["confounded"] is True
    assert b["curve_shape"] == "low_plateau_then_rise"
    assert 0.35 < b["low_plateau_iter_frac"] < 0.7
    assert any("max_dip_db=2.93" in r for r in b["severity_reasons"])
    assert any("escalated" in r for r in b["severity_reasons"])


def test_anchor_b_without_curve_is_only_minor():
    """Scalar-only grading (no psnr_history on disk) must still work and must
    grade B's 2.93dB as the blip it is -- this is the graceful-degradation
    path, and it is what distinguishes the magnitude axis from the shape one."""
    b = grade_seed_fade(ANCHOR_B_SEED_FADE)
    assert b["severity"] == "minor"
    assert b["confounded"] is False
    assert b["curve_shape"] == "insufficient_history"


def test_anchor_a_without_curve_is_still_severe():
    a = grade_seed_fade(ANCHOR_A_SEED_FADE)
    assert a["severity"] == "severe"
    assert a["confounded"] is True


# ============================ negative control ==============================


def test_steady_climb_with_no_fade_is_none():
    g = grade_seed_fade(
        {"detected": False, "max_dip_db": 0.0, "fade_start_iter": None,
         "recovered": False, "controller_engaged": False},
        STEADY_CLIMB_ITERS, STEADY_CLIMB_VALS,
    )
    assert g["severity"] == "none"
    assert g["confounded"] is False
    assert g["curve_shape"] == "steady_climb"


def test_steady_climb_with_small_fade_is_minor_and_quotable():
    g = grade_seed_fade(
        {"detected": True, "max_dip_db": 3.2, "recovered": True,
         "controller_engaged": True},
        STEADY_CLIMB_ITERS, STEADY_CLIMB_VALS,
    )
    assert g["severity"] == "minor"
    assert g["confounded"] is False


def test_unrecovered_fade_escalates():
    g = grade_seed_fade(
        {"detected": True, "max_dip_db": 3.2, "recovered": False,
         "controller_engaged": True},
        STEADY_CLIMB_ITERS, STEADY_CLIMB_VALS,
    )
    assert g["severity"] == "moderate"
    assert g["confounded"] is True
    # The escalation reason string changed with the sticky-`recovered` grader
    # (#346/#282): `ended_healthy()` now decides end-of-run health and the
    # escalation is phrased "fade active at end of run". The *behaviour*
    # asserted above (moderate + confounded for detected/unrecovered) is
    # unchanged -- only this string drifted, and no CI job runs this file.
    assert any("fade active at end of run" in r for r in g["severity_reasons"])


def test_severity_is_capped_at_severe():
    g = grade_seed_fade(
        {"detected": True, "max_dip_db": 40.0, "recovered": False},
        ANCHOR_A_ITERS, ANCHOR_A_VALS,
    )
    assert g["severity"] == "severe"


# ============================ shape primitives ==============================


def test_ends_below_peak_beats_plateau():
    vals = [10.0, 20.0, 25.0, 25.1, 24.9, 25.0, 12.0]
    iters = [1000 * (i + 1) for i in range(len(vals))]
    assert classify_curve_shape(iters, vals)["curve_shape"] == "ends_below_peak"


def test_trough_and_recovery():
    vals = [20.0, 21.0, 22.0, 17.0, 18.5, 21.0, 22.5, 23.0]
    iters = [1000 * (i + 1) for i in range(len(vals))]
    out = classify_curve_shape(iters, vals)
    assert out["curve_shape"] == "trough_and_recovery"
    assert out["curve_max_drawdown_db"] >= 3.0


def test_short_curve_is_insufficient_history():
    out = classify_curve_shape([1000, 2000], [9.0, 21.0])
    assert out["curve_shape"] == "insufficient_history"
    assert out["low_plateau_iter_frac"] is None


def test_empty_curve_never_raises():
    assert classify_curve_shape([], [])["curve_shape"] == "insufficient_history"
    assert grade_seed_fade(None)["severity"] == "none"


def test_max_drawdown_db():
    assert max_drawdown_db([10.0, 12.0, 8.0, 11.0]) == 4.0
    assert max_drawdown_db([1.0, 2.0, 3.0]) == 0.0


def test_longest_low_plateau_ignores_a_pure_ramp():
    """A monotone ramp's early points are far below the final value, but the
    span is never FLAT -- the plateau metric must not fire on it."""
    vals = [10.0 + i for i in range(20)]
    iters = [1000 * (i + 1) for i in range(20)]
    span, level, deficit = longest_low_plateau(iters, vals)
    assert span <= 2000.0  # at most a couple of adjacent ticks, not a plateau


def test_grade_never_mutates_input():
    src = dict(ANCHOR_A_SEED_FADE)
    grade_seed_fade(src, ANCHOR_A_ITERS, ANCHOR_A_VALS)
    assert src == ANCHOR_A_SEED_FADE


# ========================= on-disk plumbing =================================


def _write_run(tmpdir, name, seed_fade, iters, vals, final_iter):
    run = os.path.join(tmpdir, name)
    comp = os.path.join(run, "comparison", f"iter_{final_iter}")
    os.makedirs(comp)
    # A deliberately-earlier checkpoint whose directory name sorts AFTER the
    # final one lexically -- iter_2147483647 < iter_7000 as strings.
    early = os.path.join(run, "comparison", "iter_7000")
    os.makedirs(early)
    with open(os.path.join(early, "report.json"), "w") as f:
        json.dump({"mean_test_psnr": 9.0, "seed_fade": {"detected": False}}, f)
    with open(os.path.join(comp, "report.json"), "w") as f:
        json.dump({"mean_test_psnr": vals[-1], "n_test": 315, "eval_valid": True,
                   "seed_fade": seed_fade}, f)
    with open(os.path.join(run, "psnr_history.jsonl"), "w") as f:
        for it, v in zip(iters, vals):
            f.write(json.dumps({"iteration": it, "mean_test_psnr": v}) + "\n")
    return run


def test_grade_run_dir_picks_the_numerically_final_checkpoint(tmp_path):
    """iter_2147483647 (the online-mode sentinel) sorts BEFORE iter_7000 as a
    string. A lexical max here silently grades the wrong, early report -- a
    mistake made repeatedly against these exact directories."""
    run = _write_run(str(tmp_path), "anchorA", ANCHOR_A_SEED_FADE,
                     ANCHOR_A_ITERS, ANCHOR_A_VALS, 2147483647)
    g = grade_run_dir(run)
    assert g["_report_path"].endswith(os.path.join("iter_2147483647", "report.json"))
    assert g["severity"] == "severe"
    assert g["max_dip_db"] == 15.56


def test_grade_seed_fade_for_run_reads_history(tmp_path):
    run = _write_run(str(tmp_path), "anchorB", ANCHOR_B_SEED_FADE,
                     ANCHOR_B_ITERS, ANCHOR_B_VALS, 2147483647)
    g = grade_seed_fade_for_run(ANCHOR_B_SEED_FADE, run)
    assert g["curve_shape"] == "low_plateau_then_rise"
    assert g["severity"] == "moderate"


def test_grade_seed_fade_for_run_missing_path_degrades(tmp_path):
    g = grade_seed_fade_for_run(ANCHOR_A_SEED_FADE, str(tmp_path / "nope"))
    assert g["severity"] == "severe"
    assert g["curve_shape"] == "insufficient_history"


def test_grade_run_dir_returns_none_without_report(tmp_path):
    os.makedirs(str(tmp_path / "empty"))
    assert grade_run_dir(str(tmp_path / "empty")) is None


# =================== downstream consumers (issue #346 wiring) ===============


def test_compare_against_baseline_reports_both_sides_without_blocking(tmp_path):
    """Seed-fade severity must appear on a --baseline_run comparison, but must
    NOT flip ``comparable`` -- doing so would silently invalidate every
    existing baseline comparison, well outside this issue's reporting-only
    scope."""
    from eligsir.utils.baseline_comparison import compare_against_baseline

    baseline = _write_run(str(tmp_path), "base", ANCHOR_A_SEED_FADE,
                          ANCHOR_A_ITERS, ANCHOR_A_VALS, 2147483647)
    with open(os.path.join(baseline, "report.json"), "w") as f:
        json.dump({"mean_test_psnr": ANCHOR_A_VALS[-1], "eval_valid": True,
                   "n_test": 315, "seed_fade": ANCHOR_A_SEED_FADE}, f)
    current = _write_run(str(tmp_path), "cur", ANCHOR_B_SEED_FADE,
                         ANCHOR_B_ITERS, ANCHOR_B_VALS, 2147483647)

    current_summary = {
        "mean_test_psnr": ANCHOR_B_VALS[-1], "n_test": 315,
        "seed_fade": grade_seed_fade_for_run(ANCHOR_B_SEED_FADE, current),
    }
    res = compare_against_baseline(baseline, current_summary, current)
    assert res["comparable"] is True  # NOT blocked by seed fade
    assert res["baseline_seed_fade_severity"] == "severe"
    assert res["seed_fade_severity"] == "moderate"
    assert res["either_side_seed_fade_confounded"] is True


def test_training_curves_verdict_label_is_severity_aware(tmp_path):
    from eligsir.utils.training_curves import _seed_fade_verdict_label

    run = str(tmp_path / "run")
    os.makedirs(run)
    # No report.json at all -> the pre-#346 label is preserved, never lost.
    assert _seed_fade_verdict_label(run) == "SEED-FADE DETECTED"
    with open(os.path.join(run, "report.json"), "w") as f:
        json.dump({"seed_fade": {"detected": True, "severity": "severe"}}, f)
    assert _seed_fade_verdict_label(run) == "SEED-FADE DETECTED (SEVERE)"
    # An older report.json predating grading also falls back cleanly.
    with open(os.path.join(run, "report.json"), "w") as f:
        json.dump({"seed_fade": {"detected": True, "max_dip_db": 3.0}}, f)
    assert _seed_fade_verdict_label(run) == "SEED-FADE DETECTED"


def test_verdict_label_prefers_final_checkpoint_over_stale_root_report(tmp_path):
    """No real run writes <model_path>/report.json; anything there is stale or
    hand-placed, and under the reused-folder/ablator-rerun hazard it can be an
    EARLIER checkpoint. The numerically-final comparison/ report must win."""
    from eligsir.utils.training_curves import _seed_fade_verdict_label

    run = _write_run(str(tmp_path), "stale", ANCHOR_A_SEED_FADE,
                     ANCHOR_A_ITERS, ANCHOR_A_VALS, 2147483647)
    with open(os.path.join(run, "comparison", "iter_2147483647", "report.json")) as f:
        final = json.load(f)
    final["seed_fade"] = dict(ANCHOR_A_SEED_FADE, severity="severe")
    with open(os.path.join(run, "comparison", "iter_2147483647", "report.json"), "w") as f:
        json.dump(final, f)
    # A stale root copy claiming a much milder verdict.
    with open(os.path.join(run, "report.json"), "w") as f:
        json.dump({"seed_fade": {"detected": True, "severity": "minor"}}, f)
    assert _seed_fade_verdict_label(run) == "SEED-FADE DETECTED (SEVERE)"


# ============ mapping-boundary awareness (issue #419) =======================
#
# The discriminating pair below is the whole point of #419: ONE curve, ONE
# detector block, TWO mapping-end locations. Any implementation that ignores
# mapping_end_iter makes both sides identical and fails.
#
# The curve is Anchor A's real fr3 shape (pinned ~9dB to iter 75000, then a
# late spike to 21.5). The detector block is a deliberately MILD one
# (max_dip_db=3.0 -> base grade `minor`, recovered) so the plateau-duration
# escalation is the only thing that can move the grade -- which makes the
# severity/confounded difference, not just the label, observable.

MILD_SEED_FADE = {
    "detected": True, "max_dip_db": 3.0, "fade_start_iter": 21500,
    "recovered": True, "controller_engaged": True,
}

# Anchor A's plateau runs from iter 1000 to iter 75000; the rise starts at
# 76000 and the curve ends at 100000.
MAPPING_END_EARLY = 2000    # ingestion stopped at 2000: the plateau is 73000
                            # iterations of POST-ingest flatline -> pathological
MAPPING_END_LATE = 76000    # ingestion ran to 76000: the plateau IS the
                            # mapping phase, and quality rose the moment it
                            # ended -> the legitimate W3-66 shape


def test_identical_curve_different_mapping_end_does_not_grade_the_same():
    """Issue #419's acceptance criterion, stated as an assertion."""
    early = grade_seed_fade(MILD_SEED_FADE, ANCHOR_A_ITERS, ANCHOR_A_VALS,
                            mapping_end_iter=MAPPING_END_EARLY,
                            mapping_phase_source="live")
    late = grade_seed_fade(MILD_SEED_FADE, ANCHOR_A_ITERS, ANCHOR_A_VALS,
                           mapping_end_iter=MAPPING_END_LATE,
                           mapping_phase_source="live")

    # The premise: same curve, same detector block, same everything but the
    # boundary.
    for key in ("detected", "recovered", "max_dip_db"):
        assert early[key] == late[key]

    # The verdict must differ, on all three of shape / severity / confounded.
    assert early["curve_shape"] == "low_plateau_then_rise"
    assert late["curve_shape"] == "low_plateau_in_mapping"
    assert early["severity"] == "moderate"      # escalated by the plateau
    assert late["severity"] == "minor"          # not escalated -- it was mapping
    assert early["confounded"] is True
    assert late["confounded"] is False


def test_mapping_end_fields_are_reported_for_both_sides():
    early = grade_seed_fade(MILD_SEED_FADE, ANCHOR_A_ITERS, ANCHOR_A_VALS,
                            mapping_end_iter=MAPPING_END_EARLY,
                            mapping_phase_source="backfilled_exact")
    late = grade_seed_fade(MILD_SEED_FADE, ANCHOR_A_ITERS, ANCHOR_A_VALS,
                           mapping_end_iter=MAPPING_END_LATE,
                           mapping_phase_source="backfilled_exact")
    for g, end in ((early, MAPPING_END_EARLY), (late, MAPPING_END_LATE)):
        assert g["mapping_phase_known"] is True
        assert g["mapping_end_iter"] == end
        assert g["mapping_phase_source"] == "backfilled_exact"
        assert g["post_mapping_span_iters"] == ANCHOR_A_ITERS[-1] - end
    # The plateau survives well into the tail on the early boundary and
    # essentially not at all on the late one.
    assert early["post_mapping_plateau_frac"] >= 0.35
    assert late["post_mapping_plateau_frac"] < 0.35
    # The whole-run number is IDENTICAL on both sides -- which is exactly why
    # it cannot be the thing that discriminates them.
    assert early["low_plateau_iter_frac"] == late["low_plateau_iter_frac"]


# --- the legacy fallback, asserted rather than assumed ----------------------


def test_absent_mapping_metadata_reproduces_legacy_grading_exactly():
    """The fallback is the PRIMARY path on today's corpus (0 of 284 reports
    carry last_ingest_iter; the train.log backfill reaches ~11% of run dirs),
    so 'unchanged when absent' is a headline requirement, not an edge case."""
    for block, iters, vals, expect_shape, expect_sev in (
        (ANCHOR_A_SEED_FADE, ANCHOR_A_ITERS, ANCHOR_A_VALS,
         "low_plateau_then_rise", "severe"),
        (ANCHOR_B_SEED_FADE, ANCHOR_B_ITERS, ANCHOR_B_VALS,
         "low_plateau_then_rise", "moderate"),
        (MILD_SEED_FADE, ANCHOR_A_ITERS, ANCHOR_A_VALS,
         "low_plateau_then_rise", "moderate"),
        (ANCHOR_A_SEED_FADE, STEADY_CLIMB_ITERS, STEADY_CLIMB_VALS,
         "steady_climb", "severe"),
    ):
        g = grade_seed_fade(block, iters, vals)          # no mapping_end_iter
        assert g["curve_shape"] == expect_shape
        assert g["severity"] == expect_sev
        assert g["mapping_phase_known"] is False
        assert g["mapping_end_iter"] is None
        assert g["post_mapping_plateau_frac"] is None
        assert "legacy" in " ".join(g["severity_reasons"])


def test_mapping_end_at_or_past_curve_end_falls_back_to_legacy():
    """No observed refinement tail means no evidence either way -- the
    conservative legacy grade stands rather than a free exoneration."""
    for end in (ANCHOR_A_ITERS[-1], ANCHOR_A_ITERS[-1] + 50000):
        g = grade_seed_fade(MILD_SEED_FADE, ANCHOR_A_ITERS, ANCHOR_A_VALS,
                            mapping_end_iter=end)
        assert g["mapping_phase_known"] is False
        assert g["curve_shape"] == "low_plateau_then_rise"
        assert g["severity"] == "moderate"
        assert "no observed refinement tail" in g["mapping_boundary_note"]


def test_mapping_end_before_curve_start_is_all_post_mapping():
    g = grade_seed_fade(MILD_SEED_FADE, ANCHOR_A_ITERS, ANCHOR_A_VALS,
                        mapping_end_iter=0)
    assert g["mapping_phase_known"] is True
    assert g["curve_shape"] == "low_plateau_then_rise"
    assert g["severity"] == "moderate"


def test_short_curve_with_mapping_end_still_degrades_cleanly():
    g = grade_seed_fade(MILD_SEED_FADE, [1000, 2000], [9.0, 21.0],
                        mapping_end_iter=1500)
    assert g["curve_shape"] == "insufficient_history"
    assert g["mapping_phase_known"] is False
    assert g["severity"] == "minor"


def test_longest_low_plateau_interval_reports_bounds():
    from eligsir.utils.seed_fade_severity import longest_low_plateau_interval

    span, level, deficit, start, end = longest_low_plateau_interval(
        ANCHOR_A_ITERS, ANCHOR_A_VALS)
    assert span > 0 and level is not None
    assert start == 1000
    assert 70000 <= end <= 76000
    # The 3-tuple public wrapper must stay identical to the interval form.
    assert longest_low_plateau(ANCHOR_A_ITERS, ANCHOR_A_VALS) == (span, level, deficit)


# --- end-to-end: the boundary really is picked up off disk ------------------


IDLE_STOP_LINE = ("[streaming] no new frame ingested for 50000 optimization steps "
                  "(last at step={n}) — treating source as exhausted/idle; stopping.\n")


def test_grade_seed_fade_for_run_picks_up_boundary_from_train_log(tmp_path):
    """report.json carries no last_ingest_iter on ANY run currently on disk,
    so the train.log backfill is the path that actually fires today."""
    run = _write_run(str(tmp_path), "late", MILD_SEED_FADE,
                     ANCHOR_A_ITERS, ANCHOR_A_VALS, 2147483647)
    with open(os.path.join(run, "train.log"), "w") as f:
        f.write("[RUN START] 2026-08-13 host=x pid=1 model_path=late\n")
        f.write(IDLE_STOP_LINE.format(n=MAPPING_END_LATE))
        f.write("[FINAL STEP 100000]\n")
    g = grade_seed_fade_for_run(MILD_SEED_FADE, run)
    assert g["mapping_phase_known"] is True
    assert g["mapping_end_iter"] == MAPPING_END_LATE
    assert g["mapping_phase_source"] == "backfilled_exact"
    assert g["curve_shape"] == "low_plateau_in_mapping"
    assert g["severity"] == "minor"


def test_grade_seed_fade_for_run_without_train_log_is_legacy(tmp_path):
    run = _write_run(str(tmp_path), "nolog", MILD_SEED_FADE,
                     ANCHOR_A_ITERS, ANCHOR_A_VALS, 2147483647)
    g = grade_seed_fade_for_run(MILD_SEED_FADE, run)
    assert g["mapping_phase_known"] is False
    assert g["mapping_end_iter"] is None
    assert g["curve_shape"] == "low_plateau_then_rise"
    assert g["severity"] == "moderate"


def test_grade_seed_fade_for_run_prefers_report_json_boundary(tmp_path):
    """When a post-PR#359 run DOES carry the field, it wins over the log."""
    run = _write_run(str(tmp_path), "reportwins", MILD_SEED_FADE,
                     ANCHOR_A_ITERS, ANCHOR_A_VALS, 2147483647)
    final = os.path.join(run, "comparison", "iter_2147483647", "report.json")
    with open(final) as f:
        r = json.load(f)
    r["geometry"] = {"trajectory_coverage": {
        "last_ingest_iter": MAPPING_END_LATE,
        "final_iteration": 100000,
        "mapping_phase_source": "live",
    }}
    with open(final, "w") as f:
        json.dump(r, f)
    with open(os.path.join(run, "train.log"), "w") as f:
        f.write(IDLE_STOP_LINE.format(n=MAPPING_END_EARLY))
    g = grade_seed_fade_for_run(MILD_SEED_FADE, run)
    assert g["mapping_end_iter"] == MAPPING_END_LATE
    assert g["mapping_phase_source"] == "live"
    assert g["curve_shape"] == "low_plateau_in_mapping"


def test_explicit_mapping_end_overrides_on_disk_resolution(tmp_path):
    """write_post_training_report passes its own live _last_ingest_iter: at
    report-write time the FINAL report.json does not exist yet, so an on-disk
    resolution would read an earlier checkpoint's value. The explicit
    argument must win over both report.json and train.log."""
    run = _write_run(str(tmp_path), "override", MILD_SEED_FADE,
                     ANCHOR_A_ITERS, ANCHOR_A_VALS, 2147483647)
    with open(os.path.join(run, "train.log"), "w") as f:
        f.write(IDLE_STOP_LINE.format(n=MAPPING_END_EARLY))
    g = grade_seed_fade_for_run(MILD_SEED_FADE, run,
                                mapping_end_iter=MAPPING_END_LATE,
                                mapping_phase_source="live")
    assert g["mapping_end_iter"] == MAPPING_END_LATE
    assert g["mapping_phase_source"] == "live"
    assert g["curve_shape"] == "low_plateau_in_mapping"
    assert g["severity"] == "minor"


def _write_post_training_report_source() -> str:
    """Read + ast-parse utils/comparison_report.py's write_post_training_report
    as TEXT, never importing the module. ``utils.comparison_report`` imports
    torch at module scope, and the CPU smoke image (Dockerfile.ci-cpu) has
    none -- an ``import eligsir.utils.comparison_report`` inside a test body previously
    turned this test into a hard failure (not a skip) there. Same fix shape as
    tests/test_trajectory_coverage_report_location.py's
    ``_write_post_training_report_ast`` (issue #504): parse-as-text is
    torch-free by construction, so this now runs (not skips) on every CI
    smoke run, unlike an importorskip("torch") guard which would silently
    never execute this assertion in CI at all."""
    import ast
    import os

    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "utils", "comparison_report.py",
    )
    with open(path) as f:
        text = f.read()
    tree = ast.parse(text, filename=path)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "write_post_training_report":
            return ast.get_source_segment(text, node)
    raise AssertionError("write_post_training_report not found in utils/comparison_report.py")


def test_write_post_training_report_hands_the_grader_its_live_boundary():
    """A source-level guard on the wiring, not on behaviour -- calling
    write_post_training_report for real needs a trained model and a GPU
    render pass, which no unit test here can do. It still catches the one
    regression that matters: silently dropping the kwarg would make every
    future run fall back to on-disk resolution and read an EARLIER
    checkpoint's boundary."""
    src = _write_post_training_report_source()
    assert "mapping_end_iter=last_ingest_iter" in src
    assert 'mapping_phase_source=("live" if last_ingest_iter is not None else None)' in src


# --- the boundary must never grade WORSE than legacy (review finding) -------

# A curve that legacy grades as a mere trough (whole-run plateau fraction well
# under PLATEAU_FRAC), but whose short post-mapping tail is almost entirely a
# flat depressed stretch. Before the escalation clamp this graded `moderate`
# with a boundary and `minor` without it -- i.e. the boundary INVENTED a
# pathology, the exact opposite of what #419 is for.
_TROUGH_VALS = (
    [12.0 + 9.0 * (i / 77.0) for i in range(78)]   # clean climb 12 -> 21
    + [17.0] * 18                                  # flat, depressed
    + [21.0] * 4                                   # back up
)
_TROUGH_ITERS = [1000 * (i + 1) for i in range(len(_TROUGH_VALS))]
_TROUGH_MAPPING_END = 78000


def test_boundary_never_escalates_above_the_legacy_grade():
    legacy = grade_seed_fade(MILD_SEED_FADE, _TROUGH_ITERS, _TROUGH_VALS)
    withb = grade_seed_fade(MILD_SEED_FADE, _TROUGH_ITERS, _TROUGH_VALS,
                            mapping_end_iter=_TROUGH_MAPPING_END)
    # The tail really does look like a plateau -- the guard is doing work,
    # not passing vacuously.
    assert withb["mapping_phase_known"] is True
    assert withb["post_mapping_plateau_frac"] >= 0.35
    assert withb["low_plateau_iter_frac"] < 0.35
    assert legacy["severity"] == "minor"
    assert withb["severity"] == "minor"


def test_severity_is_monotone_non_worse_with_a_boundary():
    """The documented invariant, over a battery of curves and boundaries."""
    from eligsir.utils.seed_fade_severity import SEVERITY_ORDER

    curves = (
        (ANCHOR_A_ITERS, ANCHOR_A_VALS),
        (ANCHOR_B_ITERS, ANCHOR_B_VALS),
        (STEADY_CLIMB_ITERS, STEADY_CLIMB_VALS),
        (_TROUGH_ITERS, _TROUGH_VALS),
    )
    blocks = (MILD_SEED_FADE, ANCHOR_A_SEED_FADE, ANCHOR_B_SEED_FADE,
              {"detected": True, "max_dip_db": 7.0, "recovered": False})
    for iters, vals in curves:
        legacy = None
        for block in blocks:
            legacy = grade_seed_fade(block, iters, vals)
            for frac in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0, 1.5):
                end = iters[0] + frac * (iters[-1] - iters[0])
                g = grade_seed_fade(block, iters, vals, mapping_end_iter=end)
                assert (SEVERITY_ORDER.index(g["severity"])
                        <= SEVERITY_ORDER.index(legacy["severity"])), (
                    f"boundary at {end} graded {g['severity']} > legacy "
                    f"{legacy['severity']} on curve of {len(vals)} ticks")


# --- provisional mid-run boundaries must not be trusted ---------------------


def test_short_observed_tail_is_not_treated_as_a_known_boundary():
    """Every --test_iterations checkpoint calls write_post_training_report
    with the CURRENT _last_ingest_iter, which mid-run means 'most recent
    ingest so far', not 'mapping ended'. A few-hundred-iteration tail cannot
    support a verdict."""
    # 95% of the way along Anchor A -> 5000 iters / 5 ticks of tail.
    g = grade_seed_fade(MILD_SEED_FADE, ANCHOR_A_ITERS, ANCHOR_A_VALS,
                        mapping_end_iter=95000)
    assert g["mapping_phase_known"] is False
    assert "too short to judge" in g["mapping_boundary_note"]
    assert g["curve_shape"] == "low_plateau_then_rise"   # legacy grade stands
    assert g["severity"] == "moderate"


def test_mapping_end_iter_stays_an_int_when_the_input_is_integral():
    """report.json's sibling trajectory_coverage.last_ingest_iter is an int;
    an equality check across the two must not trip over 76000.0."""
    g = grade_seed_fade(MILD_SEED_FADE, ANCHOR_A_ITERS, ANCHOR_A_VALS,
                        mapping_end_iter=MAPPING_END_LATE)
    assert g["mapping_end_iter"] == MAPPING_END_LATE
    assert isinstance(g["mapping_end_iter"], int)
