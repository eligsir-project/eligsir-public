"""Tests for --baseline_run auto-comparison (utils.comparison_report.
compare_against_baseline / _align_psnr_curves / _classify_curve_shape).

Per CLAUDE.md's "Compare PSNR curves, not just the final number" run
contract, this feature compares the FULL psnr_history curve, not just the
final PSNR -- these tests cover curve-shape classification using synthetic
data modeled on this project's own real fr3 investigation
(docs/architecture/2026-08-10-w3-66-fr3-admission-density-root-cause.md and
its follow-up doc): R1-R5 were superimposable with baseline at every
checkpoint; baseline/R1 showed a sharp late-recovery relative to an
already-healthy run.
"""
from __future__ import annotations

import json
import os
import tempfile

from eligsir.utils.baseline_comparison import (
    compare_against_baseline,
    _align_psnr_curves,
    _classify_curve_shape,
    _compare_phase_relative_psnr,
)


def _write_run(tmpdir_path: str, name: str, *, mean_test_psnr, eval_valid=True,
               n_test=10, holdout_hash=None, psnr_history=None, iteration=52000,
               admission_fingerprint=None) -> str:
    """Real runs never write report.json at <model_path>/report.json -- only
    at <model_path>/comparison/iter_{N}/report.json (see
    _find_latest_report_json's docstring in utils/comparison_report.py).
    Matching that real layout here is what makes these tests exercise the
    actual production code path instead of a fixture-only shortcut."""
    run_dir = os.path.join(tmpdir_path, name)
    comparison_dir = os.path.join(run_dir, "comparison", f"iter_{iteration}")
    os.makedirs(comparison_dir, exist_ok=True)
    report = {
        "mean_test_psnr": mean_test_psnr,
        "eval_valid": eval_valid,
        "n_test": n_test,
    }
    if holdout_hash is not None:
        report["eval_metadata"] = {"holdout_hash": holdout_hash}
    if admission_fingerprint is not None:
        report["admission_fingerprint"] = admission_fingerprint
    with open(os.path.join(comparison_dir, "report.json"), "w") as f:
        json.dump(report, f)
    if psnr_history is not None:
        with open(os.path.join(run_dir, "psnr_history.jsonl"), "w") as f:
            for it, val in psnr_history:
                f.write(json.dumps({"iteration": it, "mean_test_psnr": val}) + "\n")
    return run_dir


# ---------------------------------------------------------------------------
# compare_against_baseline: error handling / behavior-neutral cases
# ---------------------------------------------------------------------------

def test_no_baseline_run_specified():
    result = compare_against_baseline("", {"mean_test_psnr": 20.0}, "/tmp/whatever")
    assert result["comparable"] is False
    assert "no baseline_run" in result["reason"]


def test_baseline_path_does_not_exist():
    result = compare_against_baseline(
        "/nonexistent/path/xyz", {"mean_test_psnr": 20.0}, "/tmp/whatever"
    )
    assert result["comparable"] is False
    assert "does not exist" in result["reason"]


def test_baseline_missing_report_json():
    with tempfile.TemporaryDirectory() as tmpdir:
        baseline_dir = os.path.join(tmpdir, "baseline_no_report")
        os.makedirs(baseline_dir)
        result = compare_against_baseline(baseline_dir, {"mean_test_psnr": 20.0}, tmpdir)
        assert result["comparable"] is False
        assert "no comparison/iter_*/report.json" in result["reason"]


def test_baseline_ignores_stray_top_level_report_json():
    """Regression test for the real bug found 2026-08-11: the first
    implementation looked directly at <baseline_run>/report.json, which no
    real run ever writes -- every --baseline_run comparison this session
    silently reported comparable=False as a result. A stray top-level
    report.json (e.g. left over from unrelated tooling) must NOT be picked
    up; only the real comparison/iter_{N}/report.json location counts."""
    with tempfile.TemporaryDirectory() as tmpdir:
        baseline_dir = os.path.join(tmpdir, "baseline_stray_toplevel")
        os.makedirs(baseline_dir)
        with open(os.path.join(baseline_dir, "report.json"), "w") as f:
            json.dump({"mean_test_psnr": 999.0}, f)  # must be ignored
        result = compare_against_baseline(baseline_dir, {"mean_test_psnr": 20.0}, tmpdir)
        assert result["comparable"] is False
        assert "no comparison/iter_*/report.json" in result["reason"]


def test_baseline_never_raises_on_malformed_report_json():
    with tempfile.TemporaryDirectory() as tmpdir:
        baseline_dir = os.path.join(tmpdir, "baseline_malformed")
        comparison_dir = os.path.join(baseline_dir, "comparison", "iter_1000")
        os.makedirs(comparison_dir)
        with open(os.path.join(comparison_dir, "report.json"), "w") as f:
            f.write("{not valid json")
        result = compare_against_baseline(baseline_dir, {"mean_test_psnr": 20.0}, tmpdir)
        assert result["comparable"] is False
        assert result["reason"] is not None


def test_baseline_skips_bootstrap_views_debug_artifact():
    """iter_{N}_bootstrap_views is a degenerate-split debug artifact (see
    CLAUDE.md's Run Contracts) -- never a real generalization result, must
    never be picked as the baseline report even if it's the only one."""
    with tempfile.TemporaryDirectory() as tmpdir:
        baseline_dir = os.path.join(tmpdir, "baseline_bootstrap_only")
        comparison_dir = os.path.join(baseline_dir, "comparison", "iter_2147483647_bootstrap_views")
        os.makedirs(comparison_dir)
        with open(os.path.join(comparison_dir, "report.json"), "w") as f:
            json.dump({"mean_test_psnr": 999.0}, f)
        result = compare_against_baseline(baseline_dir, {"mean_test_psnr": 20.0}, tmpdir)
        assert result["comparable"] is False
        assert "no comparison/iter_*/report.json" in result["reason"]


def test_baseline_picks_highest_real_iteration_not_mtime():
    """Multiple real iter_N checkpoints exist (normal for any run) -- must
    pick the numerically highest, not whichever happens to be newest by
    mtime or first in glob order."""
    with tempfile.TemporaryDirectory() as tmpdir:
        baseline_dir = os.path.join(tmpdir, "baseline_multi_iter")
        for it, psnr in [(1000, 10.0), (50000, 22.0), (5000, 15.0)]:
            comparison_dir = os.path.join(baseline_dir, "comparison", f"iter_{it}")
            os.makedirs(comparison_dir)
            with open(os.path.join(comparison_dir, "report.json"), "w") as f:
                json.dump({"mean_test_psnr": psnr}, f)
        result = compare_against_baseline(baseline_dir, {"mean_test_psnr": 20.0}, tmpdir)
        assert result["comparable"] is True
        assert result["baseline_mean_test_psnr"] == 22.0


# ---------------------------------------------------------------------------
# holdout_hash comparability contract
# ---------------------------------------------------------------------------

def test_matching_holdout_hash_is_verified():
    with tempfile.TemporaryDirectory() as tmpdir:
        baseline_dir = _write_run(
            tmpdir, "baseline", mean_test_psnr=22.44, holdout_hash="abc123",
            psnr_history=[(1000, 15.0), (2000, 20.0), (3000, 22.44)],
        )
        current_dir = _write_run(
            tmpdir, "current", mean_test_psnr=22.40,
            psnr_history=[(1000, 15.1), (2000, 20.1), (3000, 22.40)],
        )
        current_summary = {
            "mean_test_psnr": 22.40,
            "eval_metadata": {"holdout_hash": "abc123"},
        }
        result = compare_against_baseline(baseline_dir, current_summary, current_dir)
        assert result["comparable"] is True
        assert result["holdout_verified"] is True


def test_mismatched_holdout_hash_refuses_comparison():
    with tempfile.TemporaryDirectory() as tmpdir:
        baseline_dir = _write_run(
            tmpdir, "baseline", mean_test_psnr=22.44, holdout_hash="abc123",
            psnr_history=[(1000, 20.0)],
        )
        current_dir = _write_run(
            tmpdir, "current", mean_test_psnr=22.40,
            psnr_history=[(1000, 20.0)],
        )
        current_summary = {
            "mean_test_psnr": 22.40,
            "eval_metadata": {"holdout_hash": "different_hash"},
        }
        result = compare_against_baseline(baseline_dir, current_summary, current_dir)
        assert result["comparable"] is False
        assert "mismatch" in result["reason"]


def test_missing_holdout_hash_on_either_side_proceeds_but_unverified():
    """Matches this project's own precedent for the historical fr3 champion
    checkpoint, which predates holdout_hash entirely (see
    docs/architecture/2026-08-10-w3-66-fr3-admission-density-root-cause.md):
    treat as a caveated reference point, not block outright."""
    with tempfile.TemporaryDirectory() as tmpdir:
        baseline_dir = _write_run(
            tmpdir, "baseline", mean_test_psnr=22.44,  # no holdout_hash at all
            psnr_history=[(1000, 20.0), (2000, 22.44)],
        )
        current_dir = _write_run(
            tmpdir, "current", mean_test_psnr=22.40,
            psnr_history=[(1000, 20.1), (2000, 22.40)],
        )
        current_summary = {
            "mean_test_psnr": 22.40,
            "eval_metadata": {"holdout_hash": "abc123"},
        }
        result = compare_against_baseline(baseline_dir, current_summary, current_dir)
        assert result["comparable"] is True
        assert result["holdout_verified"] is False


def _admission_fingerprint(
    *,
    nominal_policy="hybrid_keyframe",
    mismatch=False,
    reason="NOMINAL_POLICY_REALIZED",
    owner_shares=None,
    source_fps=1.9,
    gap_p50=15.0,
    controller=None,
):
    return {
        "nominal_policy": nominal_policy,
        "admission_realized_mismatch": mismatch,
        "mismatch_reason": reason,
        "owner_shares": owner_shares or {"admit": {"values": {"pose_delta": 1.0}}},
        "realized_admit_fps": 2.0,
        "realized_admit_fps_source_time": source_fps,
        "gap_percentiles": {"p50": gap_p50},
        "controller": controller or {
            "velocity_adaptive": True,
            "timebase_source": "source_timestamp",
            "controller_target_interval_s": 0.5,
        },
    }


def _admission_comparison(tmpdir, baseline_fingerprint, current_fingerprint):
    baseline_dir = _write_run(
        tmpdir,
        "baseline",
        mean_test_psnr=22.44,
        holdout_hash="h1",
        psnr_history=[(1000, 20.0), (2000, 21.0), (3000, 22.44)],
        admission_fingerprint=baseline_fingerprint,
    )
    current_dir = _write_run(
        tmpdir,
        "current",
        mean_test_psnr=22.40,
        psnr_history=[(1000, 20.1), (2000, 21.1), (3000, 22.40)],
    )
    return compare_against_baseline(
        baseline_dir,
        {
            "mean_test_psnr": 22.40,
            "eval_metadata": {"holdout_hash": "h1"},
            "admission_fingerprint": current_fingerprint,
        },
        current_dir,
    )


def test_admission_outcome_shares_differ_without_suppressing_quality_report():
    """Observed outcomes are shown side by side, not equality-gated.

    An A/A tolerance would need to be declared explicitly; the normal A/B
    report must not invent one from exact owner-share equality.
    """
    baseline_fingerprint = _admission_fingerprint(
        owner_shares={"admit": {"values": {"pose_delta": 0.50, "coverage": 0.50}}},
        gap_p50=15.0,
    )
    current_fingerprint = _admission_fingerprint(
        owner_shares={"admit": {"values": {"pose_delta": 0.51, "coverage": 0.49}}},
        source_fps=1.8,
        gap_p50=16.0,
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        result = _admission_comparison(
            tmpdir, baseline_fingerprint, current_fingerprint
        )

    comparison = result["admission_policy_comparison"]
    assert result["comparable"] is True
    assert result["holdout_quality_comparable"] is True
    assert comparison["current_status"] == "REALIZED"
    assert comparison["baseline_status"] == "REALIZED"
    assert comparison["current"]["semantic_status"] == "REALIZED"
    assert comparison["baseline"]["semantic_status"] == "REALIZED"
    assert comparison["claim_eligibility"]["eligible"] is True
    assert result["admission_policy_claim_eligible"] is True
    assert comparison["current"]["fingerprint"] == current_fingerprint
    assert comparison["baseline"]["fingerprint"] == baseline_fingerprint
    assert comparison["current"]["owner_shares"] == current_fingerprint["owner_shares"]
    assert comparison["baseline"]["realized"]["gap_percentiles"] == {"p50": 15.0}


def test_declared_fixed_stride_vs_hybrid_treatment_remains_reportable():
    """Nominal controller/policy axes are an intended treatment, not a fault."""
    baseline_fingerprint = _admission_fingerprint(
        nominal_policy="fixed_stride",
        controller={
            "velocity_adaptive": False,
            "timebase_source": "source_timestamp",
            "controller_target_interval_s": 1.0,
        },
    )
    current_fingerprint = _admission_fingerprint(
        nominal_policy="hybrid_keyframe",
        controller={
            "velocity_adaptive": True,
            "timebase_source": "source_timestamp",
            "controller_target_interval_s": 0.5,
        },
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        result = _admission_comparison(
            tmpdir, baseline_fingerprint, current_fingerprint
        )

    comparison = result["admission_policy_comparison"]
    assert result["comparable"] is True
    assert comparison["claim_eligibility"]["eligible"] is True
    assert comparison["declared_treatment_axes"]["nominal_policy"] == {
        "current": "hybrid_keyframe",
        "baseline": "fixed_stride",
    }


def test_non_gated_arm_is_not_applicable_not_an_unknown_realization():
    baseline_fingerprint = _admission_fingerprint(
        nominal_policy="all",
        mismatch=False,
        reason="NON_GATED_ADMISSION",
    )
    current_fingerprint = _admission_fingerprint()
    with tempfile.TemporaryDirectory() as tmpdir:
        result = _admission_comparison(
            tmpdir, baseline_fingerprint, current_fingerprint
        )

    comparison = result["admission_policy_comparison"]
    assert comparison["current_status"] == "REALIZED"
    assert comparison["baseline_status"] == "NOT_APPLICABLE"
    assert comparison["claim_eligibility"]["eligible"] is True


def test_identical_force_dominated_arms_block_only_admission_policy_claims():
    fingerprint = _admission_fingerprint(
        mismatch=True,
        reason="FORCE_DOMINATED (80.0% force-owned >= 70%, nominal=hybrid_keyframe)",
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        result = _admission_comparison(tmpdir, fingerprint, fingerprint)

    comparison = result["admission_policy_comparison"]
    assert result["comparable"] is True
    assert result["holdout_quality_comparable"] is True
    assert comparison["current_status"] == "MISMATCH"
    assert comparison["baseline_status"] == "MISMATCH"
    assert comparison["semantic_verdict"] == "MISMATCH"
    assert comparison["claim_eligibility"]["eligible"] is False
    assert comparison["claim_eligibility"]["status"] == "BLOCKED_MISMATCH"
    assert comparison["claim_eligibility"]["blocking_arms"] == ["current", "baseline"]


def test_insufficient_admits_and_missing_fingerprint_stay_unknown():
    insufficient = _admission_fingerprint(
        mismatch=False,
        reason="INSUFFICIENT_ADMITS (5 < 20)",
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        insufficient_result = _admission_comparison(tmpdir, insufficient, insufficient)

    insufficient_comparison = insufficient_result["admission_policy_comparison"]
    assert insufficient_comparison["current_status"] == "UNKNOWN"
    assert insufficient_comparison["baseline_status"] == "UNKNOWN"
    assert insufficient_comparison["semantic_verdict"] == "UNKNOWN"
    assert insufficient_comparison["claim_eligibility"]["eligible"] is False
    assert insufficient_result["comparable"] is True

    with tempfile.TemporaryDirectory() as tmpdir:
        missing_result = _admission_comparison(tmpdir, None, None)

    missing_comparison = missing_result["admission_policy_comparison"]
    assert missing_comparison["current_status"] == "UNKNOWN"
    assert missing_comparison["baseline_status"] == "UNKNOWN"
    assert missing_comparison["semantic_verdict"] == "UNKNOWN"
    assert missing_comparison["claim_eligibility"]["status"] == "UNKNOWN"
    assert missing_result["comparable"] is True


def test_valid_arm_vs_force_dominated_arm_blocks_policy_claim_not_curve_report():
    baseline_fingerprint = _admission_fingerprint()
    current_fingerprint = _admission_fingerprint(
        mismatch=True,
        reason="FORCE_DOMINATED (80.0% force-owned >= 70%, nominal=hybrid_keyframe)",
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        result = _admission_comparison(
            tmpdir, baseline_fingerprint, current_fingerprint
        )

    comparison = result["admission_policy_comparison"]
    assert result["comparable"] is True
    assert comparison["current_status"] == "MISMATCH"
    assert comparison["baseline_status"] == "REALIZED"
    assert comparison["claim_eligibility"]["status"] == "BLOCKED_MISMATCH"


# ---------------------------------------------------------------------------
# issue #199: holdout_hash matching is necessary but not sufficient --
# realized_holdout_hash / n_test fallback layers
# ---------------------------------------------------------------------------

def test_matching_realized_hash_is_verified():
    """Both sides carry realized_holdout_hash and it matches.

    UPDATED for issue #420: the MEANING of 'realized_hash' narrowed. It
    verifies the realized holdout POOL, but says nothing about which subset
    of that pool metrics were actually averaged over (the
    --streaming_report_test_metrics_max cap/subsampling step). It is
    therefore no longer the top level, and `holdout_verified` is now the
    narrower truthy marker 'realized_hash' rather than True -- True is
    reserved for a verified scored subset ('scored_hash'). Neither fixture
    here carries a scored_holdout_hash, so this is the downgraded tier.
    See tests/test_scored_subset_hash.py for the new top level."""
    with tempfile.TemporaryDirectory() as tmpdir:
        baseline_dir = _write_run(
            tmpdir, "baseline", mean_test_psnr=22.44, holdout_hash="abc123",
            psnr_history=[(1000, 20.0), (2000, 22.44)],
        )
        # Patch in realized_holdout_hash on the baseline report written by _write_run.
        report_path = os.path.join(baseline_dir, "comparison", "iter_52000", "report.json")
        with open(report_path) as f:
            report = json.load(f)
        report["eval_metadata"]["realized_holdout_hash"] = "realized_xyz"
        with open(report_path, "w") as f:
            json.dump(report, f)

        current_dir = _write_run(
            tmpdir, "current", mean_test_psnr=22.40,
            psnr_history=[(1000, 20.1), (2000, 22.40)],
        )
        current_summary = {
            "mean_test_psnr": 22.40,
            "n_test": 10,
            "eval_metadata": {"holdout_hash": "abc123", "realized_holdout_hash": "realized_xyz"},
        }
        result = compare_against_baseline(baseline_dir, current_summary, current_dir)
        assert result["comparable"] is True
        assert result["holdout_verified"] == "realized_hash"
        assert result["holdout_verified"] is not True  # issue #420 downgrade
        assert result["verification_level"] == "realized_hash"


def test_same_hash_different_n_test_is_flagged_not_falsely_verified():
    """The exact GitHub issue #199 scenario: identical selection-rule
    holdout_hash, but the realized test sets differ in size (52 vs 102 in
    the real fr3 case). Must be refused, not reported holdout_verified=true."""
    with tempfile.TemporaryDirectory() as tmpdir:
        baseline_dir = _write_run(
            tmpdir, "baseline", mean_test_psnr=16.38, holdout_hash="4751ca2df3d672c3",
            n_test=52, psnr_history=[(1000, 16.0), (4000, 16.38)],
        )
        current_dir = _write_run(
            tmpdir, "current", mean_test_psnr=13.45,
            psnr_history=[(1000, 16.0), (8000, 13.45)],
        )
        current_summary = {
            "mean_test_psnr": 13.45,
            "n_test": 102,
            "eval_metadata": {"holdout_hash": "4751ca2df3d672c3"},
        }
        result = compare_against_baseline(baseline_dir, current_summary, current_dir)
        assert result["comparable"] is False
        assert result["verification_level"] == "n_test_mismatch"
        assert "n_test" in result["reason"]
        assert "52" in result["reason"] and "102" in result["reason"]


def test_legacy_reports_without_realized_hash_degrade_gracefully_when_n_test_matches():
    """Neither side has realized_holdout_hash (both predate this fix). Equal
    n_test + matching holdout_hash should NOT be reported as falsely
    'incomparable' -- it degrades to an honest 'n_test_only' verification
    level, still usable as a caveated reference point."""
    with tempfile.TemporaryDirectory() as tmpdir:
        baseline_dir = _write_run(
            tmpdir, "baseline", mean_test_psnr=22.44, holdout_hash="abc123",
            n_test=62, psnr_history=[(1000, 20.0), (2000, 22.44)],
        )
        current_dir = _write_run(
            tmpdir, "current", mean_test_psnr=22.40,
            psnr_history=[(1000, 20.1), (2000, 22.40)],
        )
        current_summary = {
            "mean_test_psnr": 22.40,
            "n_test": 62,
            "eval_metadata": {"holdout_hash": "abc123"},
        }
        result = compare_against_baseline(baseline_dir, current_summary, current_dir)
        assert result["comparable"] is True
        assert result["verification_level"] == "n_test_only"
        assert result["holdout_verified"] == "n_test_only"


# ---------------------------------------------------------------------------
# issue #745 follow-up: a report predating the #745 fix carries a legacy
# live/bag REALIZED-set hash under the `holdout_hash` field name (see
# scene/streaming_scene.py's eval_metadata() docstring) -- a bare hash
# equality against such a report must never be promoted to
# holdout_verified=True, even when the hashes happen to match.
# ---------------------------------------------------------------------------

def _write_run_with_selection(tmpdir_path: str, name: str, *, mean_test_psnr,
                               holdout_hash, holdout_selection, n_test=10,
                               psnr_history=None, iteration=52000) -> str:
    run_dir = os.path.join(tmpdir_path, name)
    comparison_dir = os.path.join(run_dir, "comparison", f"iter_{iteration}")
    os.makedirs(comparison_dir, exist_ok=True)
    report = {
        "mean_test_psnr": mean_test_psnr,
        "eval_valid": True,
        "eval_metadata": {
            "holdout_hash": holdout_hash,
            "holdout_selection": holdout_selection,
        },
    }
    if n_test is not None:
        report["n_test"] = n_test
    with open(os.path.join(comparison_dir, "report.json"), "w") as f:
        json.dump(report, f)
    if psnr_history is not None:
        with open(os.path.join(run_dir, "psnr_history.jsonl"), "w") as f:
            for it, val in psnr_history:
                f.write(json.dumps({"iteration": it, "mean_test_psnr": val}) + "\n")
    return run_dir


def test_legacy_live_hash_match_is_not_promoted_to_verified():
    """Baseline predates #745 (holdout_selection == the legacy realized-set
    marker); current run's holdout_hash happens to match it byte-for-byte.
    Per the mechanism #745 root-caused, that match is coincidental -- the
    baseline's "holdout_hash" is actually a realized-set fingerprint, not a
    selection-rule fingerprint. Must NOT read as holdout_verified=True/
    "hash_only". Neither side carries `n_test`/`realized_holdout_hash`, so
    this exercises the bottom-of-the-ladder tier where the new
    "legacy_live_realized_set_hash" level applies directly (a case with a
    comparable `n_test` instead falls into the existing "n_test_only" tier
    with `holdout_verified=False` -- same credit-withholding outcome, just
    reported under that tier's existing name; see
    test_legacy_live_hash_downgrades_n_test_only_tier_too below)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        baseline_dir = _write_run_with_selection(
            tmpdir, "baseline_legacy", mean_test_psnr=22.44,
            holdout_hash="coincidental_match",
            holdout_selection="online_every_nth_hashed",
            n_test=None,
            psnr_history=[(1000, 20.0), (2000, 22.44)],
        )
        current_dir = _write_run_with_selection(
            tmpdir, "current_post_fix", mean_test_psnr=22.40,
            holdout_hash="coincidental_match",
            holdout_selection="online_every_nth_ordinal_rule",
            n_test=None,
            psnr_history=[(1000, 20.1), (2000, 22.40)],
        )
        current_summary = {
            "mean_test_psnr": 22.40,
            "eval_metadata": {
                "holdout_hash": "coincidental_match",
                "holdout_selection": "online_every_nth_ordinal_rule",
            },
        }
        result = compare_against_baseline(baseline_dir, current_summary, current_dir)
        assert result["comparable"] is True  # still usable as a caveated reference point
        assert result["holdout_verified"] is False
        assert result["verification_level"] == "legacy_live_realized_set_hash"


def test_legacy_live_hash_downgrades_n_test_only_tier_too():
    """Same coincidental-match scenario as above, but both sides also carry
    a matching `n_test` (the pre-#199 fallback tier). The legacy marker must
    still withhold hash-based credit here: `holdout_verified` stays False
    rather than the truthy `"n_test_only"` marker a genuine (non-legacy)
    hash match at this tier would earn -- see
    test_legacy_reports_without_realized_hash_degrade_gracefully_when_
    n_test_matches above for that non-legacy baseline behavior, which this
    must NOT regress."""
    with tempfile.TemporaryDirectory() as tmpdir:
        baseline_dir = _write_run_with_selection(
            tmpdir, "baseline_legacy_ntest", mean_test_psnr=22.44,
            holdout_hash="coincidental_match", n_test=62,
            holdout_selection="online_every_nth_hashed",
            psnr_history=[(1000, 20.0), (2000, 22.44)],
        )
        current_dir = _write_run_with_selection(
            tmpdir, "current_post_fix_ntest", mean_test_psnr=22.40, n_test=62,
            holdout_hash="coincidental_match",
            holdout_selection="online_every_nth_ordinal_rule",
            psnr_history=[(1000, 20.1), (2000, 22.40)],
        )
        current_summary = {
            "mean_test_psnr": 22.40,
            "n_test": 62,
            "eval_metadata": {
                "holdout_hash": "coincidental_match",
                "holdout_selection": "online_every_nth_ordinal_rule",
            },
        }
        result = compare_against_baseline(baseline_dir, current_summary, current_dir)
        assert result["comparable"] is True
        assert result["verification_level"] == "n_test_only"
        assert result["holdout_verified"] is False  # NOT the truthy "n_test_only" marker


def test_legacy_live_hash_mismatch_still_refuses_outright():
    """A genuine mismatch stays a hard refusal regardless of the legacy
    marker -- the legacy downgrade only withholds CREDIT from a match, it
    never grants leniency to a mismatch."""
    with tempfile.TemporaryDirectory() as tmpdir:
        baseline_dir = _write_run_with_selection(
            tmpdir, "baseline_legacy", mean_test_psnr=22.44,
            holdout_hash="realized_set_hash_A",
            holdout_selection="online_every_nth_hashed",
            psnr_history=[(1000, 20.0)],
        )
        current_dir = _write_run_with_selection(
            tmpdir, "current_post_fix", mean_test_psnr=22.40,
            holdout_hash="rule_hash_B",
            holdout_selection="online_every_nth_ordinal_rule",
            psnr_history=[(1000, 20.0)],
        )
        current_summary = {
            "mean_test_psnr": 22.40,
            "eval_metadata": {
                "holdout_hash": "rule_hash_B",
                "holdout_selection": "online_every_nth_ordinal_rule",
            },
        }
        result = compare_against_baseline(baseline_dir, current_summary, current_dir)
        assert result["comparable"] is False
        assert result["verification_level"] == "hash_mismatch"


def test_legacy_live_hash_does_not_affect_realized_hash_tier():
    """The legacy-marker downgrade only touches the bottom two (hash-only /
    n_test-only) tiers. A report that ALSO carries a matching
    realized_holdout_hash (possible in principle -- a report could predate
    only the holdout_hash rename, not the realized_holdout_hash field,
    though not expected in practice) must still be free to earn credit at
    that stronger tier, since realized_holdout_hash never depended on the
    holdout_hash field's meaning in the first place."""
    with tempfile.TemporaryDirectory() as tmpdir:
        baseline_dir = os.path.join(tmpdir, "baseline_legacy_with_realized")
        comparison_dir = os.path.join(baseline_dir, "comparison", "iter_52000")
        os.makedirs(comparison_dir)
        with open(os.path.join(comparison_dir, "report.json"), "w") as f:
            json.dump({
                "mean_test_psnr": 22.44,
                "eval_valid": True,
                "n_test": 10,
                "eval_metadata": {
                    "holdout_hash": "coincidental_match",
                    "holdout_selection": "online_every_nth_hashed",
                    "realized_holdout_hash": "realized_xyz",
                },
            }, f)
        with open(os.path.join(baseline_dir, "psnr_history.jsonl"), "w") as f:
            f.write(json.dumps({"iteration": 2000, "mean_test_psnr": 22.44}) + "\n")

        current_dir = os.path.join(tmpdir, "current_post_fix")
        os.makedirs(current_dir)
        with open(os.path.join(current_dir, "psnr_history.jsonl"), "w") as f:
            f.write(json.dumps({"iteration": 2000, "mean_test_psnr": 22.40}) + "\n")

        current_summary = {
            "mean_test_psnr": 22.40,
            "n_test": 10,
            "eval_metadata": {
                "holdout_hash": "coincidental_match",
                "holdout_selection": "online_every_nth_ordinal_rule",
                "realized_holdout_hash": "realized_xyz",
            },
        }
        result = compare_against_baseline(baseline_dir, current_summary, current_dir)
        assert result["comparable"] is True
        assert result["verification_level"] == "realized_hash"
        assert result["holdout_verified"] == "realized_hash"


# ---------------------------------------------------------------------------
# issue #745 follow-up, second gap: a ros2://live bag-replay run's
# holdout_hash rule payload cannot discriminate between two DIFFERENT bags
# (args.source_path is the literal constant "ros2://live" for every such
# run) -- scene_fingerprint (a content hash of the actual GT pixels) can,
# and is now consulted as an additional top-level refusal check.
# ---------------------------------------------------------------------------

def _write_run_with_fingerprint(tmpdir_path: str, name: str, *, mean_test_psnr,
                                 holdout_hash, scene_fingerprint, n_test=10,
                                 psnr_history=None, iteration=52000) -> str:
    run_dir = os.path.join(tmpdir_path, name)
    comparison_dir = os.path.join(run_dir, "comparison", f"iter_{iteration}")
    os.makedirs(comparison_dir, exist_ok=True)
    report = {
        "mean_test_psnr": mean_test_psnr,
        "eval_valid": True,
        "n_test": n_test,
        "eval_metadata": {
            "holdout_hash": holdout_hash,
            "holdout_selection": "online_every_nth_ordinal_rule",
        },
        "scene_fingerprint": scene_fingerprint,
    }
    with open(os.path.join(comparison_dir, "report.json"), "w") as f:
        json.dump(report, f)
    if psnr_history is not None:
        with open(os.path.join(run_dir, "psnr_history.jsonl"), "w") as f:
            for it, val in psnr_history:
                f.write(json.dumps({"iteration": it, "mean_test_psnr": val}) + "\n")
    return run_dir


def test_scene_fingerprint_mismatch_refuses_comparison_despite_matching_holdout_hash():
    """Two DIFFERENT bags replayed via ros2://live can coincidentally share
    a holdout_hash rule fingerprint (same --streaming_eval_hold/--streaming_
    eval_guard, both source_path == the literal constant "ros2://live") --
    scene_fingerprint (a content hash of the actual GT pixels) must still
    catch the real scene mismatch and refuse outright, exactly like a
    holdout_hash mismatch would."""
    with tempfile.TemporaryDirectory() as tmpdir:
        baseline_dir = _write_run_with_fingerprint(
            tmpdir, "baseline_floor3", mean_test_psnr=22.44,
            holdout_hash="coincidental_rule_match",
            scene_fingerprint="fp_floor3",
            psnr_history=[(1000, 20.0), (2000, 22.44)],
        )
        current_dir = _write_run_with_fingerprint(
            tmpdir, "current_handheld", mean_test_psnr=13.0,
            holdout_hash="coincidental_rule_match",
            scene_fingerprint="fp_handheld",
            psnr_history=[(1000, 12.0), (2000, 13.0)],
        )
        current_summary = {
            "mean_test_psnr": 13.0,
            "n_test": 10,
            "eval_metadata": {
                "holdout_hash": "coincidental_rule_match",
                "holdout_selection": "online_every_nth_ordinal_rule",
            },
            "scene_fingerprint": "fp_handheld",
        }
        result = compare_against_baseline(baseline_dir, current_summary, current_dir)
        assert result["comparable"] is False
        assert result["verification_level"] == "scene_fingerprint_mismatch"
        assert "different scenes" in result["reason"]


def test_matching_scene_fingerprint_does_not_block_comparison():
    with tempfile.TemporaryDirectory() as tmpdir:
        baseline_dir = _write_run_with_fingerprint(
            tmpdir, "baseline_floor3b", mean_test_psnr=22.44,
            holdout_hash="abc123", scene_fingerprint="fp_floor3",
            psnr_history=[(1000, 20.0), (2000, 22.44)],
        )
        current_dir = _write_run_with_fingerprint(
            tmpdir, "current_floor3b", mean_test_psnr=22.40,
            holdout_hash="abc123", scene_fingerprint="fp_floor3",
            psnr_history=[(1000, 20.1), (2000, 22.40)],
        )
        current_summary = {
            "mean_test_psnr": 22.40,
            "n_test": 10,
            "eval_metadata": {
                "holdout_hash": "abc123",
                "holdout_selection": "online_every_nth_ordinal_rule",
            },
            "scene_fingerprint": "fp_floor3",
        }
        result = compare_against_baseline(baseline_dir, current_summary, current_dir)
        assert result["comparable"] is True
        assert result["verification_level"] != "scene_fingerprint_mismatch"


def test_missing_scene_fingerprint_on_either_side_skips_the_check():
    """Behavior-neutral for every report written before scene_fingerprint
    existed (or a degenerate split, which never emits one) -- absence must
    never itself cause a refusal."""
    with tempfile.TemporaryDirectory() as tmpdir:
        baseline_dir = _write_run(
            tmpdir, "baseline_no_fp", mean_test_psnr=22.44, holdout_hash="abc123",
            psnr_history=[(1000, 20.0), (2000, 22.44)],
        )
        current_dir = _write_run(
            tmpdir, "current_no_fp", mean_test_psnr=22.40,
            psnr_history=[(1000, 20.1), (2000, 22.40)],
        )
        current_summary = {
            "mean_test_psnr": 22.40,
            "eval_metadata": {"holdout_hash": "abc123"},
            "scene_fingerprint": "fp_only_on_this_side",
        }
        result = compare_against_baseline(baseline_dir, current_summary, current_dir)
        assert result["comparable"] is True
        assert result["verification_level"] != "scene_fingerprint_mismatch"


def test_realized_holdout_hash_computed_from_test_cams(monkeypatch):
    """_realized_holdout_hash / write_post_training_report wiring: two
    fake camera lists with the same length but different underlying frame
    indices must hash differently, proving n_test equality alone is not
    what backs this new field."""
    from eligsir.utils.baseline_comparison import _realized_holdout_hash

    class _FakeFrame:
        def __init__(self, index):
            self.index = index

    class _FakeCam:
        def __init__(self, index):
            self._streaming_frame = _FakeFrame(index)

    cams_a = [_FakeCam(i) for i in (5, 15, 25)]
    cams_b = [_FakeCam(i) for i in (5, 15, 35)]  # same count, different realized set
    h_a = _realized_holdout_hash(cams_a)
    h_b = _realized_holdout_hash(cams_b)
    assert h_a is not None and h_b is not None
    assert h_a != h_b
    # Order-independence: sorted internally.
    assert _realized_holdout_hash(list(reversed(cams_a))) == h_a
    assert _realized_holdout_hash([]) is None


# ---------------------------------------------------------------------------
# curve alignment / shape classification -- realistic synthetic data
# ---------------------------------------------------------------------------

def test_superimposable_curves_real_r1_vs_baseline_pattern():
    """Modeled on the real R1-vs-baseline data (superimposable at every
    checkpoint, e.g. iter 8000: 14.85-15.24dB across all arms; iter 40000:
    10.40-11.68dB across all arms) -- deltas stay small throughout."""
    baseline_curve = [(1000, 23.0), (8000, 15.0), (20000, 12.0), (40000, 11.0), (52000, 18.98)]
    r1_curve = [(1000, 22.8), (8000, 15.2), (20000, 12.3), (40000, 11.5), (52000, 19.02)]
    aligned = _align_psnr_curves(
        [it for it, _ in r1_curve], [v for _, v in r1_curve],
        [it for it, _ in baseline_curve], [v for _, v in baseline_curve],
    )
    assert len(aligned) == 5
    assert _classify_curve_shape(aligned) == "superimposable"


def test_converges_curve_real_late_spike_recovery_pattern():
    """Modeled on the real baseline/R1 vs. a healthy-throughout run (e.g.
    R6): baseline sat in an 11-12dB trough through iter 50000 then jumped
    to ~19dB at iter 52000, while a healthy comparator stayed near 20-22dB
    throughout -- large mid-run divergence, but if the LATE baseline value
    is used as "current" against an early-converged "baseline" comparator,
    or vice versa depending on framing, the point is: big max delta, small
    final delta is a real, distinct pattern from "always identical"."""
    healthy_run = [(1000, 20.5), (8000, 21.0), (20000, 21.5), (40000, 22.0), (52000, 22.4)]
    late_spike_run = [(1000, 23.0), (8000, 15.0), (20000, 12.0), (40000, 11.0), (52000, 22.1)]
    aligned = _align_psnr_curves(
        [it for it, _ in late_spike_run], [v for _, v in late_spike_run],
        [it for it, _ in healthy_run], [v for _, v in healthy_run],
    )
    shape = _classify_curve_shape(aligned)
    assert shape == "converges"


def test_diverges_curve_endpoint_meaningfully_different():
    current_curve = [(1000, 15.0), (5000, 16.0), (10000, 17.0)]
    baseline_curve = [(1000, 15.0), (5000, 20.0), (10000, 25.0)]
    aligned = _align_psnr_curves(
        [it for it, _ in current_curve], [v for _, v in current_curve],
        [it for it, _ in baseline_curve], [v for _, v in baseline_curve],
    )
    assert _classify_curve_shape(aligned) == "diverges"


def test_insufficient_overlap_too_few_aligned_points():
    current_curve = [(1000, 15.0)]
    baseline_curve = [(1000, 15.0)]
    aligned = _align_psnr_curves(
        [it for it, _ in current_curve], [v for _, v in current_curve],
        [it for it, _ in baseline_curve], [v for _, v in baseline_curve],
    )
    assert _classify_curve_shape(aligned) == "insufficient_overlap"


def test_alignment_skips_points_beyond_tolerance():
    """A checkpoint with no nearby baseline point within tolerance must not
    be fabricated via interpolation across a large gap -- it's simply
    dropped from the aligned set."""
    current_curve = [(1000, 15.0), (2000, 16.0), (100000, 30.0)]  # 100000 is a huge outlier gap
    baseline_curve = [(1000, 15.0), (2000, 16.0), (3000, 17.0)]
    aligned = _align_psnr_curves(
        [it for it, _ in current_curve], [v for _, v in current_curve],
        [it for it, _ in baseline_curve], [v for _, v in baseline_curve],
    )
    aligned_iters = [it for it, _, _, _ in aligned]
    assert 100000 not in aligned_iters


# ---------------------------------------------------------------------------
# full compare_against_baseline result shape
# ---------------------------------------------------------------------------

def test_full_result_includes_final_delta_and_curve_fields():
    with tempfile.TemporaryDirectory() as tmpdir:
        baseline_dir = _write_run(
            tmpdir, "baseline", mean_test_psnr=22.44, holdout_hash="h1",
            psnr_history=[(1000, 20.0), (2000, 22.0), (3000, 22.44)],
        )
        current_dir = _write_run(
            tmpdir, "current", mean_test_psnr=22.40,
            psnr_history=[(1000, 20.1), (2000, 22.1), (3000, 22.40)],
        )
        current_summary = {
            "mean_test_psnr": 22.40,
            "eval_metadata": {"holdout_hash": "h1"},
        }
        result = compare_against_baseline(baseline_dir, current_summary, current_dir)
        assert result["comparable"] is True
        assert abs(result["final_psnr_delta"] - (22.40 - 22.44)) < 1e-9
        assert result["curve_shape"] == "superimposable"
        assert result["n_aligned_checkpoints"] == 3
        assert "curve_mean_delta" in result
        assert "curve_max_abs_delta" in result
        assert result["baseline_mean_test_psnr"] == 22.44
        assert result["baseline_run"] == baseline_dir


def test_current_run_with_no_psnr_history_yet_falls_back_to_endpoint_only():
    with tempfile.TemporaryDirectory() as tmpdir:
        baseline_dir = _write_run(
            tmpdir, "baseline", mean_test_psnr=22.44, holdout_hash="h1",
            psnr_history=[(1000, 20.0)],
        )
        current_dir = os.path.join(tmpdir, "current_no_history")
        os.makedirs(current_dir)
        current_summary = {
            "mean_test_psnr": 22.40,
            "eval_metadata": {"holdout_hash": "h1"},
        }
        result = compare_against_baseline(baseline_dir, current_summary, current_dir)
        assert result["comparable"] is True
        assert result["curve_shape"] == "insufficient_overlap"
        # Endpoint delta is still computed even without curve data.
        assert result["final_psnr_delta"] is not None


# ---------------------------------------------------------------------------
# Phase-relative curve comparison.  Absolute optimizer iteration is not a
# comparable clock when two online runs finish mapping at different updates.
# ---------------------------------------------------------------------------


def _write_phase_relative_run(
    tmpdir_path: str,
    name: str,
    *,
    mapping_endpoint_iteration: int,
    endpoint_mapping_active_s: float,
    endpoint_supervision_views: int,
    psnr_values: list[float],
    psnr_reduction: str | None = None,
    omit_mapping_active_at: int | None = None,
    throughput_evidence_valid: bool | None = None,
) -> str:
    """Write a complete synthetic streaming run using production file layouts."""
    run_dir = _write_run(
        tmpdir_path,
        name,
        mean_test_psnr=psnr_values[-1],
        holdout_hash="phase-relative-holdout",
        psnr_history=None,
        iteration=mapping_endpoint_iteration + 100 * (len(psnr_values) - 1),
    )
    with open(os.path.join(run_dir, "streaming_phase_metadata.json"), "w") as f:
        json.dump(
            {
                "version": 1,
                "phase": "complete",
                "mapping_end_iteration": mapping_endpoint_iteration,
                "mapping_optimizer_steps": mapping_endpoint_iteration,
                "refinement_optimizer_steps": len(psnr_values) - 1,
            },
            f,
        )
    with open(os.path.join(run_dir, "psnr_history.jsonl"), "w") as f:
        for offset, psnr in enumerate(psnr_values):
            row = {
                "iteration": mapping_endpoint_iteration + 100 * offset,
                "mean_test_psnr": psnr,
                "mapping_active_s": endpoint_mapping_active_s + 10.0 * offset,
                "supervision_views": endpoint_supervision_views + 400 * offset,
            }
            if psnr_reduction is not None:
                row["psnr_reduction"] = psnr_reduction
            if omit_mapping_active_at == offset:
                del row["mapping_active_s"]
            if throughput_evidence_valid is not None:
                row["throughput_evidence_valid"] = throughput_evidence_valid
                row["throughput_evidence_status"] = (
                    "valid_no_mapping_holdout_observer_pause"
                    if throughput_evidence_valid
                    else "invalid_mapping_holdout_observer_pause"
                )
            f.write(json.dumps(row) + "\n")
    return run_dir


def test_baseline_comparison_refuses_cross_run_psnr_reduction_mismatch():
    """A numerical cross-run gain needs one declared PSNR reduction."""
    with tempfile.TemporaryDirectory() as tmpdir:
        baseline_dir = _write_phase_relative_run(
            tmpdir,
            "legacy_baseline",
            mapping_endpoint_iteration=100,
            endpoint_mapping_active_s=10.0,
            endpoint_supervision_views=400,
            psnr_values=[20.0, 21.0, 22.0],
        )
        current_dir = _write_phase_relative_run(
            tmpdir,
            "rgb_current",
            mapping_endpoint_iteration=100,
            endpoint_mapping_active_s=10.0,
            endpoint_supervision_views=400,
            psnr_values=[20.5, 21.5, 22.5],
            psnr_reduction="rgb_per_view_v2",
        )

        result = compare_against_baseline(
            baseline_dir,
            {
                "mean_test_psnr": 22.5,
                "eval_metadata": {"holdout_hash": "phase-relative-holdout"},
            },
            current_dir,
        )

        assert result["comparable"] is False
        assert "PSNR reduction mismatch" in result["reason"]
        assert "final_psnr_delta" not in result


def test_phase_relative_reader_refuses_a_mixed_psnr_history():
    """The raw phase-relative consumer must not bypass reduction validation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        baseline_dir = _write_phase_relative_run(
            tmpdir,
            "baseline",
            mapping_endpoint_iteration=100,
            endpoint_mapping_active_s=10.0,
            endpoint_supervision_views=400,
            psnr_values=[20.0, 21.0, 22.0],
        )
        current_dir = _write_phase_relative_run(
            tmpdir,
            "mixed_current",
            mapping_endpoint_iteration=100,
            endpoint_mapping_active_s=10.0,
            endpoint_supervision_views=400,
            psnr_values=[20.0, 21.0, 22.0],
        )
        with open(os.path.join(current_dir, "psnr_history.jsonl"), "a") as handle:
            handle.write(json.dumps({"iteration": 130, "mean_test_psnr": 23.0, "psnr_reduction": "rgb_per_view_v2"}) + "\n")

        try:
            _compare_phase_relative_psnr(current_dir, baseline_dir)
        except ValueError as error:
            assert "mixed PSNR reductions" in str(error)
        else:
            raise AssertionError("phase-relative comparison accepted a mixed PSNR history")


def test_phase_relative_curves_ignore_shifted_mapping_endpoint_iterations():
    """Regression: identical refinement curves must not look worse merely
    because mapping ended at update 1603 instead of 1143.

    The absolute-iteration overlap compares the treatment's early refinement
    against the baseline's mature refinement and therefore reads as a large
    regression.  Both endpoint-relative physical-work axes are byte-for-byte
    identical and are the authoritative comparison.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        psnr_values = [16.0 + offset for offset in range(11)]
        baseline_dir = _write_phase_relative_run(
            tmpdir,
            "baseline_endpoint_1143",
            mapping_endpoint_iteration=1143,
            endpoint_mapping_active_s=63.0,
            endpoint_supervision_views=4572,
            psnr_values=psnr_values,
        )
        current_dir = _write_phase_relative_run(
            tmpdir,
            "treatment_endpoint_1603",
            mapping_endpoint_iteration=1603,
            endpoint_mapping_active_s=91.5,
            endpoint_supervision_views=6412,
            psnr_values=psnr_values,
        )

        result = compare_against_baseline(
            baseline_dir,
            {
                "mean_test_psnr": 26.0,
                "eval_metadata": {"holdout_hash": "phase-relative-holdout"},
            },
            current_dir,
        )

        primary = result["phase_relative_comparison"]
        assert result["psnr_reduction"] == "legacy_chw_channel_mean_v1"
        assert primary["status"] == "VALID"
        assert primary["authoritative"] is True
        assert primary["mapping_endpoints"]["baseline"]["iteration"] == 1143
        assert primary["mapping_endpoints"]["current"]["iteration"] == 1603
        for axis_name in ("mapping_active_s", "supervision_views"):
            axis = primary["axes"][axis_name]
            assert axis["status"] == "VALID"
            assert axis["curve_shape"] == "superimposable"
            assert axis["sample_counts"] == {
                "current_post_mapping": 11,
                "baseline_post_mapping": 11,
                "aligned": 11,
            }
            assert axis["endpoint_psnr_db"] == {
                "current": 16.0,
                "baseline": 16.0,
                "delta": 0.0,
            }
            assert axis["fast_growth"]["current"]["axis_to_plus_1db"] == (
                10.0 if axis_name == "mapping_active_s" else 400.0
            )
            assert axis["axis_provenance"]["normalization"].startswith(
                "subtract each run's own mapping endpoint"
            )
            assert axis["axis_provenance"]["alignment"] == (
                "piecewise-linear interpolation on the union of observed knots "
                "within shared post-mapping support"
            )

        # Kept for compatibility, but explicitly prevented from masquerading
        # as the primary cross-throughput comparison.
        assert result["curve_shape"] == "diverges"
        assert result["curve_shape_provenance"]["status"] == "LEGACY_NON_PRIMARY"
        assert result["curve_shape_provenance"]["primary"] is False
        assert result["legacy_iteration_comparison"]["curve_shape"] == "diverges"


def test_phase_relative_comparison_fails_closed_when_an_axis_clock_is_missing():
    """A partially logged physical-work clock must not silently fall back to
    absolute optimizer iteration or allow the complete primary block to pass.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        baseline_dir = _write_phase_relative_run(
            tmpdir,
            "baseline_complete_clocks",
            mapping_endpoint_iteration=1143,
            endpoint_mapping_active_s=63.0,
            endpoint_supervision_views=4572,
            psnr_values=[16.0, 17.0, 18.0],
        )
        current_dir = _write_phase_relative_run(
            tmpdir,
            "current_missing_wall_clock",
            mapping_endpoint_iteration=1603,
            endpoint_mapping_active_s=91.5,
            endpoint_supervision_views=6412,
            psnr_values=[16.0, 17.0, 18.0],
            omit_mapping_active_at=1,
        )

        result = compare_against_baseline(
            baseline_dir,
            {
                "mean_test_psnr": 18.0,
                "eval_metadata": {"holdout_hash": "phase-relative-holdout"},
            },
            current_dir,
        )

        primary = result["phase_relative_comparison"]
        assert primary["status"] == "UNAVAILABLE"
        assert primary["authoritative"] is True
        wall_time = primary["axes"]["mapping_active_s"]
        assert wall_time["status"] == "UNAVAILABLE"
        assert "missing mapping_active_s" in wall_time["reason"]
        assert wall_time["curve_shape"] is None
        # The independently complete axis remains readable, but cannot upgrade
        # the two-axis authoritative comparison to VALID.
        assert primary["axes"]["supervision_views"]["status"] == "VALID"
        assert result["curve_shape_provenance"]["status"] == "LEGACY_NON_PRIMARY"


def test_phase_relative_comparison_refuses_contaminated_throughput_axis():
    with tempfile.TemporaryDirectory() as tmpdir:
        baseline_dir = _write_phase_relative_run(
            tmpdir,
            "baseline_clean_throughput",
            mapping_endpoint_iteration=100,
            endpoint_mapping_active_s=10.0,
            endpoint_supervision_views=400,
            psnr_values=[20.0, 21.0],
            throughput_evidence_valid=True,
        )
        current_dir = _write_phase_relative_run(
            tmpdir,
            "current_paused_observer",
            mapping_endpoint_iteration=100,
            endpoint_mapping_active_s=10.0,
            endpoint_supervision_views=400,
            psnr_values=[20.0, 21.0],
            throughput_evidence_valid=False,
        )

        result = compare_against_baseline(
            baseline_dir,
            {
                "mean_test_psnr": 21.0,
                "eval_metadata": {"holdout_hash": "phase-relative-holdout"},
            },
            current_dir,
        )

        comparison = result["phase_relative_comparison"]
        assert comparison["status"] == "UNAVAILABLE"
        assert comparison["axes"]["mapping_active_s"]["status"] == "UNAVAILABLE"
        assert "invalid_mapping_holdout_observer_pause" in (
            comparison["axes"]["mapping_active_s"]["reason"]
        )
        assert comparison["axes"]["supervision_views"]["status"] == "VALID"
