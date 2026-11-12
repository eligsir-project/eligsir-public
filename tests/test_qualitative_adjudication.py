"""Tests for qualitative adjudication protocol (issue #515)."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from eligsir.utils.qualitative_adjudication import (
    QualitativeAdjudicationVerdict,
    TriggerEvidence,
    _evaluate_trigger_1_metric_within_band,
    _evaluate_trigger_2_bootstrap_ci,
    _evaluate_trigger_3_metrics_disagree,
    _evaluate_trigger_4_partition_regression,
    _evaluate_trigger_5_online_quality_differs,
    _evaluate_trigger_6_mechanism_diagnostic_changes,
    _evaluate_trigger_7_visual_artifact,
    _evaluate_trigger_8_targets_poorly_measured_property,
    adjudication_verdict_from_file,
    evaluate_qualitative_adjudication,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def baseline_report():
    """A minimal baseline report."""
    return {
        "mean_test_psnr": 25.0,
        "mean_test_lpips": 0.15,
        "n_test": 50,
        "eval_valid": True,
        "geometry": {
            "needle_frac_lt_0_3": 0.05,
            "floater_volume": 0.01,
        },
        "seed_fade": {
            "severity": "minor",
            "curve_shape": "superimposable",
        },
        "partition_psnr": {
            "p0": 27.0,
            "p1": 25.0,
            "p2": 23.0,
        },
    }


@pytest.fixture
def current_report():
    """A minimal current/candidate report."""
    return {
        "mean_test_psnr": 25.2,
        "mean_test_lpips": 0.14,
        "n_test": 50,
        "eval_valid": True,
        "geometry": {
            "needle_frac_lt_0_3": 0.06,
            "floater_volume": 0.015,
        },
        "seed_fade": {
            "severity": "minor",
            "curve_shape": "superimposable",
        },
        "partition_psnr": {
            "p0": 27.3,
            "p1": 25.1,
            "p2": 23.2,
        },
    }


# ============================================================================
# Trigger #1: Metric within declared band
# ============================================================================

def test_trigger_1_within_band(current_report, baseline_report):
    """Trigger 1 fires when PSNR delta is within declared band."""
    ev = _evaluate_trigger_1_metric_within_band(
        current_report, baseline_report, declared_band_db=0.5
    )
    assert ev.evaluable
    assert ev.fired  # delta is 0.2 dB, within 0.5 dB band
    assert ev.trigger_number == 1


def test_trigger_1_outside_band(current_report, baseline_report):
    """Trigger 1 does not fire when PSNR delta exceeds band."""
    ev = _evaluate_trigger_1_metric_within_band(
        current_report, baseline_report, declared_band_db=0.1
    )
    assert ev.evaluable
    assert not ev.fired  # delta is 0.2 dB, exceeds 0.1 dB band


def test_trigger_1_no_band_declared(current_report, baseline_report):
    """Trigger 1 is non-evaluable when no band is declared."""
    ev = _evaluate_trigger_1_metric_within_band(
        current_report, baseline_report, declared_band_db=None
    )
    assert not ev.evaluable
    assert "declared noise band" in ev.not_evaluable_reason


def test_trigger_1_no_baseline(current_report):
    """Trigger 1 is non-evaluable without baseline."""
    ev = _evaluate_trigger_1_metric_within_band(
        current_report, baseline_report=None, declared_band_db=0.5
    )
    assert not ev.evaluable


def test_trigger_1_missing_psnr(current_report, baseline_report):
    """Trigger 1 is non-evaluable when PSNR is missing."""
    baseline_report["mean_test_psnr"] = None
    ev = _evaluate_trigger_1_metric_within_band(
        current_report, baseline_report, declared_band_db=0.5
    )
    assert not ev.evaluable


# ============================================================================
# Trigger #2: Bootstrap CI
# ============================================================================

def test_trigger_2_always_non_evaluable(current_report, baseline_report):
    """Trigger 2 is always non-evaluable (no bootstrap CI exists)."""
    ev = _evaluate_trigger_2_bootstrap_ci(current_report, baseline_report)
    assert not ev.evaluable
    assert ev.trigger_number == 2


# ============================================================================
# Trigger #3: Metrics disagree
# ============================================================================

def test_trigger_3_psnr_up_lpips_down(current_report, baseline_report):
    """Trigger 3 fires when PSNR improves but LPIPS regresses."""
    current_report["mean_test_psnr"] = 26.0  # +1.0 dB
    current_report["mean_test_lpips"] = 0.20  # +0.05 worse

    ev = _evaluate_trigger_3_metrics_disagree(current_report, baseline_report)
    assert ev.evaluable
    assert ev.fired
    assert "disagreement" in ev.reason.lower()


def test_trigger_3_psnr_down_lpips_up(current_report, baseline_report):
    """Trigger 3 fires when PSNR regresses but LPIPS improves."""
    current_report["mean_test_psnr"] = 23.5  # -1.5 dB
    current_report["mean_test_lpips"] = 0.10  # -0.05 better

    ev = _evaluate_trigger_3_metrics_disagree(current_report, baseline_report)
    assert ev.evaluable
    assert ev.fired


def test_trigger_3_psnr_up_floater_grows(current_report, baseline_report):
    """Trigger 3 fires when PSNR improves but floaters grow."""
    current_report["mean_test_psnr"] = 26.0  # +1.0 dB
    current_report["geometry"]["floater_volume"] = 0.05  # grows

    ev = _evaluate_trigger_3_metrics_disagree(current_report, baseline_report)
    assert ev.evaluable
    assert ev.fired


def test_trigger_3_both_improve(current_report, baseline_report):
    """Trigger 3 does not fire when metrics agree (both improve cleanly)."""
    current_report["mean_test_psnr"] = 26.0  # +1.0 dB
    current_report["mean_test_lpips"] = 0.10  # -0.05 better
    # Keep floater_volume same to avoid triggering on that
    current_report["geometry"]["floater_volume"] = 0.01

    ev = _evaluate_trigger_3_metrics_disagree(current_report, baseline_report)
    assert ev.evaluable
    assert not ev.fired


def test_trigger_3_no_baseline(current_report):
    """Trigger 3 is non-evaluable without baseline."""
    ev = _evaluate_trigger_3_metrics_disagree(current_report, baseline_report=None)
    assert not ev.evaluable


def test_trigger_3_missing_lpips(current_report, baseline_report):
    """Trigger 3 is non-evaluable when LPIPS is missing."""
    baseline_report["mean_test_lpips"] = None
    ev = _evaluate_trigger_3_metrics_disagree(current_report, baseline_report)
    assert not ev.evaluable


# ============================================================================
# Trigger #4: Partition regression
# ============================================================================

def test_trigger_4_partition_regresses(current_report, baseline_report):
    """Trigger 4 fires when global improves but partition regresses."""
    current_report["mean_test_psnr"] = 25.5  # +0.5 global
    current_report["partition_psnr"]["p2"] = 21.0  # -2.0 partition regression

    ev = _evaluate_trigger_4_partition_regression(current_report, baseline_report)
    assert ev.evaluable
    assert ev.fired
    assert len(ev.evidence["partition_regressions"]) > 0


def test_trigger_4_all_improve(current_report, baseline_report):
    """Trigger 4 does not fire when all partitions improve."""
    current_report["mean_test_psnr"] = 25.5  # +0.5 global
    current_report["partition_psnr"]["p0"] = 27.5
    current_report["partition_psnr"]["p1"] = 25.5
    current_report["partition_psnr"]["p2"] = 23.5

    ev = _evaluate_trigger_4_partition_regression(current_report, baseline_report)
    assert ev.evaluable
    assert not ev.fired


def test_trigger_4_global_declines(current_report, baseline_report):
    """Trigger 4 does not fire when global PSNR declines."""
    current_report["mean_test_psnr"] = 24.5  # -0.5 global decline

    ev = _evaluate_trigger_4_partition_regression(current_report, baseline_report)
    assert ev.evaluable
    assert not ev.fired


def test_trigger_4_no_baseline(current_report):
    """Trigger 4 is non-evaluable without baseline."""
    ev = _evaluate_trigger_4_partition_regression(current_report, baseline_report=None)
    assert not ev.evaluable


def test_trigger_4_missing_partitions(current_report, baseline_report):
    """Trigger 4 is non-evaluable without partition data."""
    baseline_report["partition_psnr"] = None
    ev = _evaluate_trigger_4_partition_regression(current_report, baseline_report)
    assert not ev.evaluable


# ============================================================================
# Trigger #5: Online quality differs
# ============================================================================

def test_trigger_5_diverges(current_report):
    """Trigger 5 fires when curve shape is 'diverges'."""
    current_report["baseline_comparison"] = {
        "comparable": True,
        "curve_shape": "diverges",
    }

    ev = _evaluate_trigger_5_online_quality_differs(current_report, baseline_report=None)
    assert ev.evaluable
    assert ev.fired


def test_trigger_5_mixed(current_report):
    """Trigger 5 fires when curve shape is 'mixed'."""
    current_report["baseline_comparison"] = {
        "comparable": True,
        "curve_shape": "mixed",
    }

    ev = _evaluate_trigger_5_online_quality_differs(current_report, baseline_report=None)
    assert ev.evaluable
    assert ev.fired


def test_trigger_5_seed_fade_problematic(current_report):
    """Trigger 5 fires when seed_fade shows problematic shape."""
    current_report["baseline_comparison"] = {
        "comparable": True,
        "curve_shape": "superimposable",
    }
    current_report["seed_fade"]["curve_shape"] = "low_plateau_then_rise"

    ev = _evaluate_trigger_5_online_quality_differs(current_report, baseline_report=None)
    assert ev.evaluable
    assert ev.fired


def test_trigger_5_no_comparable_baseline(current_report):
    """Trigger 5 is non-evaluable without comparable baseline."""
    current_report["baseline_comparison"] = {
        "comparable": False,
        "reason": "no baseline",
    }

    ev = _evaluate_trigger_5_online_quality_differs(current_report, baseline_report=None)
    assert not ev.evaluable


def test_trigger_5_clean(current_report):
    """Trigger 5 does not fire for clean curves."""
    current_report["baseline_comparison"] = {
        "comparable": True,
        "curve_shape": "superimposable",
    }
    current_report["seed_fade"]["curve_shape"] = "superimposable"

    ev = _evaluate_trigger_5_online_quality_differs(current_report, baseline_report=None)
    assert ev.evaluable
    assert not ev.fired


def test_trigger_5_prefers_phase_relative_curves_over_legacy_iteration_shift(
    current_report,
):
    """A mapping-endpoint shift may make the retained absolute-iteration
    diagnostic diverge while both authoritative physical-work curves match.
    Trigger 5 must consume the latter and avoid a false regression.
    """
    current_report["baseline_comparison"] = {
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
    }
    current_report["seed_fade"]["curve_shape"] = "superimposable"

    ev = _evaluate_trigger_5_online_quality_differs(
        current_report, baseline_report=None
    )

    assert ev.evaluable
    assert not ev.fired
    assert ev.evidence["phase_relative_curve_shapes"] == {
        "mapping_active_s": "superimposable",
        "supervision_views": "superimposable",
    }
    assert ev.evidence["legacy_iteration_curve_shape"] == "diverges"


def test_trigger_5_fails_closed_when_authoritative_phase_curve_is_unavailable(
    current_report,
):
    current_report["baseline_comparison"] = {
        "comparable": True,
        "curve_shape": "superimposable",
        "curve_shape_provenance": {
            "status": "LEGACY_NON_PRIMARY",
            "primary": False,
        },
        "phase_relative_comparison": {
            "status": "UNAVAILABLE",
            "authoritative": True,
            "reason": "current row missing mapping_active_s",
            "axes": {},
        },
    }

    ev = _evaluate_trigger_5_online_quality_differs(
        current_report, baseline_report=None
    )

    assert not ev.evaluable
    assert "phase-relative" in ev.not_evaluable_reason
    assert "missing mapping_active_s" in ev.not_evaluable_reason


# ============================================================================
# Trigger #6: Mechanism diagnostic changes
# ============================================================================

def test_trigger_6_seed_fade_severity_change(current_report, baseline_report):
    """Trigger 6 fires when PSNR tied but seed_fade severity changes."""
    current_report["mean_test_psnr"] = 25.0  # same as baseline
    current_report["seed_fade"]["severity"] = "moderate"  # changed from minor

    ev = _evaluate_trigger_6_mechanism_diagnostic_changes(current_report, baseline_report)
    assert ev.evaluable
    assert ev.fired
    assert "seed_fade_severity" in ev.evidence


def test_trigger_6_needle_fraction_change(current_report, baseline_report):
    """Trigger 6 fires when PSNR tied but needle fraction changes."""
    current_report["mean_test_psnr"] = 25.0  # same as baseline
    current_report["geometry"]["needle_frac_lt_0_3"] = 0.20  # +0.15 change

    ev = _evaluate_trigger_6_mechanism_diagnostic_changes(current_report, baseline_report)
    assert ev.evaluable
    assert ev.fired
    assert "needle_fraction" in ev.evidence


def test_trigger_6_psnr_different(current_report, baseline_report):
    """Trigger 6 does not fire when PSNR clearly differs."""
    current_report["mean_test_psnr"] = 26.1  # +1.1 dB different (beyond 1.0 dB noise floor)
    current_report["seed_fade"]["severity"] = "moderate"

    ev = _evaluate_trigger_6_mechanism_diagnostic_changes(current_report, baseline_report)
    assert ev.evaluable
    assert not ev.fired  # only fires when PSNR is tied (within ~1.0 dB)


# ============================================================================
# Trigger #7: Visual artifact
# ============================================================================

def test_trigger_7_always_non_evaluable():
    """Trigger 7 is always non-evaluable (human judgment only)."""
    ev = _evaluate_trigger_7_visual_artifact({}, None)
    assert not ev.evaluable
    assert ev.trigger_number == 7


# ============================================================================
# Trigger #8: Targets poorly-measured property
# ============================================================================

def test_trigger_8_declared_property(current_report):
    """Trigger 8 fires when property is explicitly declared."""
    ev = _evaluate_trigger_8_targets_poorly_measured_property(
        current_report,
        baseline_report=None,
        declared_property="thin-structure recovery",
    )
    assert ev.evaluable
    assert ev.fired
    assert "thin-structure recovery" in ev.reason


def test_trigger_8_no_declared_property(current_report):
    """Trigger 8 is non-evaluable when property is not declared."""
    ev = _evaluate_trigger_8_targets_poorly_measured_property(
        current_report,
        baseline_report=None,
        declared_property=None,
    )
    assert not ev.evaluable


# ============================================================================
# Main Verdict Function
# ============================================================================

def test_main_verdict_qualitative_review_required(current_report, baseline_report):
    """Main verdict returns QUALITATIVE_REVIEW_REQUIRED when triggers fire."""
    current_report["mean_test_psnr"] = 26.0
    current_report["mean_test_lpips"] = 0.20  # disagree
    baseline_report["mean_test_lpips"] = 0.15

    verdict = evaluate_qualitative_adjudication(current_report, baseline_report)
    assert verdict.status == "QUALITATIVE_REVIEW_REQUIRED"
    assert len([t for t in verdict.triggered_reasons if t.fired]) > 0


def test_main_verdict_decided_quantitatively(current_report, baseline_report):
    """Main verdict returns DECIDED_QUANTITATIVELY when no triggers fire."""
    # Both improve, no conflicts — make sure floater_volume matches to avoid trigger 3
    current_report["geometry"]["floater_volume"] = baseline_report["geometry"]["floater_volume"]
    verdict = evaluate_qualitative_adjudication(current_report, baseline_report)
    assert verdict.status == "DECIDED_QUANTITATIVELY"


def test_main_verdict_insufficient_evidence_no_baseline(current_report):
    """Main verdict handles missing baseline gracefully."""
    # Without baseline, most triggers can't fire but are evaluable
    verdict = evaluate_qualitative_adjudication(current_report)
    assert verdict.status in ("DECIDED_QUANTITATIVELY", "INSUFFICIENT_EVIDENCE")


def test_main_verdict_to_dict(current_report, baseline_report):
    """Verdict can be serialized to dict."""
    verdict = evaluate_qualitative_adjudication(current_report, baseline_report)
    verdict_dict = verdict.to_dict()

    assert "status" in verdict_dict
    assert "triggered_reasons" in verdict_dict
    assert "summary" in verdict_dict
    assert "metadata" in verdict_dict

    # Verify structure
    for trigger in verdict_dict["triggered_reasons"]:
        assert "trigger_number" in trigger
        assert "trigger_name" in trigger
        assert "fired" in trigger


# ============================================================================
# File I/O Tests
# ============================================================================

def test_adjudication_from_file(current_report, baseline_report):
    """Can load reports from files and evaluate."""
    # Ensure no trigger conflicts in the fixture
    current_report["geometry"]["floater_volume"] = baseline_report["geometry"]["floater_volume"]

    with tempfile.TemporaryDirectory() as tmpdir:
        current_path = Path(tmpdir) / "current_report.json"
        baseline_path = Path(tmpdir) / "baseline_report.json"

        with open(current_path, "w") as f:
            json.dump(current_report, f)
        with open(baseline_path, "w") as f:
            json.dump(baseline_report, f)

        verdict = adjudication_verdict_from_file(
            str(current_path),
            str(baseline_path),
        )

        assert verdict.status in ("DECIDED_QUANTITATIVELY", "QUALITATIVE_REVIEW_REQUIRED")


def test_adjudication_from_file_missing_report():
    """Raises FileNotFoundError for missing reports."""
    with pytest.raises(FileNotFoundError):
        adjudication_verdict_from_file("/nonexistent/path.json")


def test_adjudication_from_file_no_baseline(current_report):
    """Works with current report only (falls back to INSUFFICIENT_EVIDENCE)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        current_path = Path(tmpdir) / "current_report.json"

        with open(current_path, "w") as f:
            json.dump(current_report, f)

        verdict = adjudication_verdict_from_file(str(current_path))
        # Without a baseline, most triggers are non-evaluable -> INSUFFICIENT_EVIDENCE
        # (which is correct behavior)
        assert verdict.status in ("DECIDED_QUANTITATIVELY", "INSUFFICIENT_EVIDENCE")


# ============================================================================
# Metadata Tests
# ============================================================================

def test_verdict_metadata(current_report, baseline_report):
    """Verdict captures relevant metadata."""
    verdict = evaluate_qualitative_adjudication(current_report, baseline_report)

    assert verdict.metadata["current_mean_test_psnr"] == current_report.get("mean_test_psnr")
    assert verdict.metadata["baseline_mean_test_psnr"] == baseline_report.get("mean_test_psnr")


def test_trigger_evidence_structure():
    """TriggerEvidence has expected structure."""
    ev = TriggerEvidence(
        trigger_number=1,
        trigger_name="Test",
        fired=True,
        reason="test reason",
        evidence={"key": "value"},
        evaluable=True,
    )

    assert ev.trigger_number == 1
    assert ev.fired
    assert ev.evaluable
    assert ev.evidence["key"] == "value"
