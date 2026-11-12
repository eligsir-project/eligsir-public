"""Tests for utils.run_summary (issue #699, consolidated-run-summary slice).

Torch-free (per module docstring). Uses synthetic report.json fixtures
under temp-dir ``comparison/iter_N/report.json`` layouts -- never reads real
run directories from ``output/`` (real-corpus validation for those is done
manually and recorded in the companion architecture doc, not in this suite).

Three scenario classes, per the task's explicit requirement:
  1. a clean run (all gates present and pass -> overall QUOTABLE)
  2. a run failing several gates (-> overall NOT_QUOTABLE)
  3. a run with keys ABSENT because it predates a fix (-> overall UNKNOWN,
     never silently treated as clean)
"""
from __future__ import annotations

import json
import os

import pytest

from eligsir.utils.run_summary import (
    Gate,
    build_experiment_validity_summary,
    build_holdout_block,
    build_metrics_block,
    build_provenance_block,
    build_quotability_block,
    build_run_summary,
    build_seed_fade_block,
    compute_overall_verdict,
    render_markdown,
)


def _write_report(run_dir: str, report: dict, iteration: int = 5000) -> str:
    comp_dir = os.path.join(run_dir, "comparison", f"iter_{iteration}")
    os.makedirs(comp_dir, exist_ok=True)
    path = os.path.join(comp_dir, "report.json")
    with open(path, "w") as f:
        json.dump(report, f)
    return path


def _write_psnr_history(run_dir: str, iters, vals) -> None:
    path = os.path.join(run_dir, "psnr_history.jsonl")
    with open(path, "w") as f:
        for it, v in zip(iters, vals):
            f.write(json.dumps({"iteration": it, "mean_test_psnr": v}) + "\n")


# ---------------------------------------------------------------------------
# Scenario 1: clean, fully-instrumented run -> QUOTABLE
# ---------------------------------------------------------------------------


def _clean_report() -> dict:
    return {
        "eval_valid": True,
        "n_test": 50,
        "n_test_metrics": 50,
        "n_test_metrics_attempted": 50,
        "n_test_metrics_failed": 0,
        "psnr_history_curve_valid": True,
        "mean_test_psnr": 27.5,
        "min_test_psnr": 25.0,
        "max_test_psnr": 29.0,
        "mean_test_lpips": 0.15,
        "iteration": 5000,
        "eval_metadata": {
            "holdout_hash": "abc123",
            "realized_holdout_hash": "def456",
            "scored_holdout_hash": "ghi789",
        },
        "trajectory_coverage": {
            "verdict": "FULL",
            "n_source_frames": 100,
            "frames_admitted": 100,
            "coverage_fraction": 1.0,
            "final_iteration": 5000,
            "map_reinit_invalidated": False,
            "stream_end_reason": "max_frames_reached",
            "scene_registered": True,
        },
        "seed_fade": {
            "detected": False,
            "indeterminate": False,
            "indeterminate_reasons": [],
            "suppression_confounded": False,
        },
        "experiment_validity": {
            "declared": "FULL_CONTINUAL_VALIDATION",
            "achieved": "FULL_CONTINUAL_VALIDATION",
            "effective": "FULL_CONTINUAL_VALIDATION",
            "negative_conclusions_permitted": True,
            "strong_negative_conclusions_permitted": True,
            "negative_result_status_ceiling": None,
            "mismatch": False,
            "is_final_report": True,
            "warnings": [],
        },
        "git_provenance": {
            "git_commit": "deadbeef",
            "git_branch": "main",
            "git_dirty": False,
        },
        "report_provenance": {"kind": "native"},
    }


def test_clean_run_is_quotable(tmp_path):
    run_dir = str(tmp_path / "clean_run")
    os.makedirs(run_dir, exist_ok=True)
    _write_report(run_dir, _clean_report())
    _write_psnr_history(run_dir, [1000, 2000, 3000, 4000, 5000], [20.0, 22.0, 24.0, 26.0, 27.5])

    summary = build_run_summary(run_dir)
    assert summary.report_found is True
    assert summary.overall["verdict"] == "QUOTABLE"
    assert summary.overall["failing_gates"] == []
    assert summary.overall["unknown_gates"] == []
    assert summary.quotability["eval_valid"] is True
    assert summary.quotability["n_test"] == 50
    assert summary.metrics["headline"]["mean_test_psnr"] == 27.5
    assert summary.metrics["curve_shape"] in ("steady_climb", "insufficient_history")
    assert summary.provenance["git_commit"] == "deadbeef"
    assert summary.experiment_validity["effective"] == "FULL_CONTINUAL_VALIDATION"


def test_incomplete_psnr_history_is_not_quotable(tmp_path):
    run_dir = str(tmp_path / "invalid_curve")
    os.makedirs(run_dir, exist_ok=True)
    report = _clean_report()
    report["psnr_history_curve_valid"] = False
    report["psnr_history_invalid_ticks"] = [
        {"iteration": 2000, "error": "nonfinite render"}
    ]
    _write_report(run_dir, report)

    summary = build_run_summary(run_dir)

    assert summary.overall["verdict"] == "NOT_QUOTABLE"
    assert "psnr_history_curve_valid" in summary.overall["failing_gates"]
    assert summary.experiment_validity["negative_conclusions_permitted"] is True

    # Markdown renders without raising and mentions the overall verdict.
    md = render_markdown(summary)
    assert "QUOTABLE" in md
    assert "issue #699" in md


# ---------------------------------------------------------------------------
# Scenario 2: run failing several gates -> NOT_QUOTABLE
# ---------------------------------------------------------------------------


def _failing_report() -> dict:
    return {
        "eval_valid": False,  # degenerate split
        "n_test": 0,
        "n_test_metrics_failed": 3,
        "mean_test_psnr": 18.0,
        "eval_metadata": {"holdout_hash": "abc123"},
        "trajectory_coverage": {
            "verdict": "SUSPECTED_TRUNCATED",
            "n_source_frames": 500,
            "frames_admitted": 100,
            "coverage_fraction": 0.2,
            "map_reinit_invalidated": True,
            "map_reinit_reason": "ORB-SLAM3 Atlas re-init detected mid-run",
            "stream_end_reason": "idle_timeout_no_completion_signal",
            "scene_registered": True,
        },
        "seed_fade": {
            "detected": True,
            "max_dip_db": 15.0,  # >= SEVERE_MAX_DIP_DB threshold -> severity="severe"
            "indeterminate": True,
            "indeterminate_reasons": ["resumed across pre-#290 checkpoint"],
            "suppression_confounded": True,
        },
    }


def test_failing_run_is_not_quotable(tmp_path):
    run_dir = str(tmp_path / "failing_run")
    os.makedirs(run_dir, exist_ok=True)
    _write_report(run_dir, _failing_report())

    summary = build_run_summary(run_dir)
    assert summary.overall["verdict"] == "NOT_QUOTABLE"
    failing = set(summary.overall["failing_gates"])
    assert "eval_valid" in failing
    assert "n_test_positive" in failing
    assert "n_test_metrics_failed_zero" in failing
    assert "map_reinit_invalidated" in failing
    assert "stream_end_not_truncated" in failing
    assert "seed_fade_not_confounded" in failing
    assert "seed_fade_not_indeterminate" in failing

    assert summary.seed_fade["confounded"]["status"] == "fail"
    assert summary.seed_fade["indeterminate"]["status"] == "fail"
    assert summary.seed_fade["indeterminate"]["value"] is True
    assert summary.seed_fade["suppression_confounded"]["status"] == "fail"
    # The three seed-fade predicates are independently reported even though
    # they all happen to be "bad" in this fixture -- assert they are NOT
    # collapsed into one flag.
    assert set(summary.seed_fade.keys()) >= {"confounded", "suppression_confounded", "indeterminate"}

    md = render_markdown(summary)
    assert "NOT_QUOTABLE" in md


# ---------------------------------------------------------------------------
# Scenario 3: pre-fix run, several keys absent entirely -> UNKNOWN, never PASS
# ---------------------------------------------------------------------------


def _legacy_minimal_report() -> dict:
    # Deliberately mirrors a real historical minimal report (verified against
    # output/web_viewer_test/comparison/iter_3000/report.json on 2026-08-15):
    # no eval_valid, no trajectory_coverage, no seed_fade, no experiment_validity,
    # no git_provenance/report_provenance, no n_test_metrics_failed.
    return {
        "iteration": 3000,
        "mean_test_psnr": 18.2266,
        "n_test": 25,
        "n_test_metrics": 25,
    }


def test_legacy_run_missing_keys_grades_unknown_not_clean(tmp_path):
    run_dir = str(tmp_path / "legacy_run")
    os.makedirs(run_dir, exist_ok=True)
    _write_report(run_dir, _legacy_minimal_report())

    summary = build_run_summary(run_dir)

    # The single most important invariant this module exists to enforce:
    # a run that never had eval_valid/n_test_metrics_failed/etc. must NOT
    # read as passing those gates.
    assert summary.overall["verdict"] == "UNKNOWN"
    assert summary.overall["failing_gates"] == []
    unknown = set(summary.overall["unknown_gates"])
    assert "eval_valid" in unknown
    assert "n_test_metrics_failed_zero" in unknown
    assert "map_reinit_invalidated" in unknown
    assert "stream_end_not_truncated" in unknown

    assert summary.quotability["eval_valid"] is None
    assert summary.quotability["n_test_metrics_failed"] is None
    assert summary.quotability["n_test_metrics_failed_present"] is False

    # Seed fade: no block at all and no psnr_history.jsonl on disk -> data
    # unavailable, not "no fade".
    assert summary.seed_fade["data_available"] is False
    assert summary.seed_fade["confounded"]["status"] == "unknown"
    assert summary.seed_fade["indeterminate"]["present"] is False

    # experiment_validity / provenance: explicitly flagged as predating the
    # relevant issue, not defaulted to a permissive-looking value.
    assert summary.experiment_validity["present"] is False
    assert summary.experiment_validity["negative_conclusions_permitted"] is None
    assert summary.provenance["git_provenance_present"] is False
    assert summary.provenance["report_provenance_present"] is False

    md = render_markdown(summary)
    assert "UNKNOWN" in md
    assert "predates" in md.lower()


# ---------------------------------------------------------------------------
# Unit-level checks on the smaller composers
# ---------------------------------------------------------------------------


def test_compute_overall_verdict_fail_closed_on_unknown():
    gates = [
        Gate("a", "pass", "ok").to_dict(),
        Gate("b", "unknown", "no data").to_dict(),
    ]
    v = compute_overall_verdict(gates)
    assert v["verdict"] == "UNKNOWN"
    assert v["unknown_gates"] == ["b"]


def test_compute_overall_verdict_fail_wins_over_unknown():
    gates = [
        Gate("a", "fail", "bad").to_dict(),
        Gate("b", "unknown", "no data").to_dict(),
    ]
    v = compute_overall_verdict(gates)
    assert v["verdict"] == "NOT_QUOTABLE"


def test_compute_overall_verdict_all_pass():
    gates = [Gate("a", "pass", "ok").to_dict(), Gate("b", "pass", "ok").to_dict()]
    v = compute_overall_verdict(gates)
    assert v["verdict"] == "QUOTABLE"
    assert v["failing_gates"] == []
    assert v["unknown_gates"] == []


def test_build_holdout_block_ladder_rungs():
    assert build_holdout_block(None)["highest_verified_rung"] == "unknown"
    assert build_holdout_block({"eval_metadata": {"holdout_hash": "h"}})["highest_verified_rung"] == "hash_only"
    assert build_holdout_block(
        {"eval_metadata": {"holdout_hash": "h", "realized_holdout_hash": "r"}}
    )["highest_verified_rung"] == "realized_hash"
    assert build_holdout_block(
        {"eval_metadata": {"holdout_hash": "h", "realized_holdout_hash": "r", "scored_holdout_hash": "s"}}
    )["highest_verified_rung"] == "scored_hash"


def test_holdout_block_labels_legacy_iteration_curve_and_surfaces_primary_axes():
    block = build_holdout_block(
        {
            "eval_metadata": {"holdout_hash": "h"},
            "baseline_comparison": {
                "comparable": True,
                "curve_shape": "diverges",
                "curve_shape_provenance": {
                    "status": "LEGACY_NON_PRIMARY",
                    "primary": False,
                },
                "phase_relative_comparison": {
                    "status": "VALID",
                    "authoritative": True,
                    "axes": {
                        "mapping_active_s": {
                            "status": "VALID",
                            "curve_shape": "superimposable",
                        },
                        "supervision_views": {
                            "status": "VALID",
                            "curve_shape": "superimposable",
                        },
                    },
                },
            },
        }
    )

    assert block["baseline_curve_shape"] == "diverges"
    assert block["baseline_curve_shape_status"] == "LEGACY_NON_PRIMARY"
    assert block["baseline_phase_relative_status"] == "VALID"
    assert block["baseline_phase_relative_curve_shapes"] == {
        "mapping_active_s": "superimposable",
        "supervision_views": "superimposable",
    }


def test_build_provenance_block_absent_vs_present():
    absent = build_provenance_block({})
    assert absent["git_provenance_present"] is False
    assert absent["git_commit"] is None

    present = build_provenance_block({"git_provenance": {"git_commit": "abc", "git_dirty": True}})
    assert present["git_provenance_present"] is True
    assert present["git_commit"] == "abc"
    assert present["git_dirty"] is True


def test_build_experiment_validity_summary_absent_is_not_permissive():
    block = build_experiment_validity_summary(None)
    assert block["present"] is False
    assert block["negative_conclusions_permitted"] is None
    assert block["strong_negative_conclusions_permitted"] is None


def test_build_metrics_block_curve_shape_reused_from_seed_fade():
    seed_fade_block = {
        "curve_shape": "trough_and_recovery",
        "curve_first_db": 10.0,
        "curve_final_db": 20.0,
        "curve_peak_db": 22.0,
        "curve_max_drawdown_db": 8.0,
        "curve_n_ticks": 12,
    }
    m = build_metrics_block({"mean_test_psnr": 20.0}, seed_fade_block)
    assert m["curve_shape"] == "trough_and_recovery"
    assert m["headline"]["mean_test_psnr"] == 20.0


def test_run_dir_with_no_report_at_all(tmp_path):
    run_dir = str(tmp_path / "no_report")
    os.makedirs(run_dir, exist_ok=True)
    summary = build_run_summary(run_dir)
    assert summary.report_found is False
    assert summary.report_path is None
    # Fail-closed: no report at all is UNKNOWN, never QUOTABLE.
    assert summary.overall["verdict"] == "UNKNOWN"
    md = render_markdown(summary)
    assert "UNKNOWN" in md


def test_seed_fade_three_predicates_can_disagree(tmp_path):
    """Confounded (severity) and suppression_confounded and indeterminate are
    independent -- construct a case where they disagree in different
    directions and assert none of them get collapsed."""
    run_dir = str(tmp_path / "mixed_seed_fade")
    os.makedirs(run_dir, exist_ok=True)
    report = _clean_report()
    report["seed_fade"] = {
        "detected": True,
        "severity": "minor",       # NOT confounded (minor stays quotable)
        "confounded": False,
        "suppression_confounded": True,   # controller DID throttle a loss
        "indeterminate": False,
        "indeterminate_reasons": [],
    }
    _write_report(run_dir, report)

    summary = build_run_summary(run_dir)
    sf = summary.seed_fade
    assert sf["confounded"]["status"] == "pass"
    assert sf["suppression_confounded"]["status"] == "fail"
    assert sf["indeterminate"]["status"] == "pass"
    # Overall gate list only escalates on confounded/indeterminate, not on
    # suppression_confounded (it is informational, per #282's own docs) --
    # so this run stays QUOTABLE despite the disagreement.
    assert summary.overall["verdict"] == "QUOTABLE"
