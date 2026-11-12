"""Tests for W3-17's throughput-gate fixes: pinned span-version guard and
the absolute-ms budget alongside the relative-percent budget. See
docs/architecture/w3_02_completion.md's 2026-08-09 "round 3" section for
the incident (a begin()/end() span mismatch between arms produced false
12.08%/17.84% "regressions" across two earlier rounds) this directly
guards against, and eligsir/experiments/w3_02_runtime.py's
SPAN_DEFINITION docstring for the pinned-span contract itself."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

# scripts/ has no __init__.py -- import the module directly by path, same
# pattern this repo already uses elsewhere for one-off script modules.
_SPEC = importlib.util.spec_from_file_location(
    "w3_02_level_d_report", Path(__file__).resolve().parents[1] / "scripts" / "w3_02_level_d_report.py"
)
_mod = importlib.util.module_from_spec(_SPEC)
sys.modules["w3_02_level_d_report"] = _mod
_SPEC.loader.exec_module(_mod)  # type: ignore[union-attr]
_throughput_summary = _mod._throughput_summary


def _evidence(events: dict[int, float], span_version: int | None = 2) -> dict:
    return {
        "optimization_gpu_events": [{"iteration": i, "gpu_ms": ms} for i, ms in events.items()],
        "runtime": {"span_definition_version": span_version},
    }


def _flat_events(n: int, ms: float) -> dict[int, float]:
    return {i: ms for i in range(n)}


def test_matching_span_versions_produce_a_comparison():
    baseline = _evidence(_flat_events(9, 5.0), span_version=2)
    treatment = _evidence(_flat_events(9, 5.05), span_version=2)
    result = _throughput_summary(baseline, treatment)
    assert result["available"] is True
    assert result["span_definition_version"] == 2
    assert result["within_budget"] is True  # 1% regression, under the 2% default


def test_mismatched_span_versions_refuse_to_compare():
    baseline = _evidence(_flat_events(9, 5.0), span_version=1)
    treatment = _evidence(_flat_events(9, 5.0), span_version=2)
    result = _throughput_summary(baseline, treatment)
    assert result["available"] is False
    assert "span_definition_version mismatch" in result["reason"]
    assert "1" in result["reason"] and "2" in result["reason"]


def test_missing_span_version_on_one_arm_also_refuses():
    # Pre-dates commit 4b2e22e / this field entirely -- must not be silently
    # treated as compatible with versioned evidence.
    baseline = _evidence(_flat_events(9, 5.0), span_version=None)
    treatment = _evidence(_flat_events(9, 5.0), span_version=2)
    result = _throughput_summary(baseline, treatment)
    assert result["available"] is False
    assert "span_definition_version mismatch" in result["reason"]


def test_both_arms_missing_span_version_still_compares():
    # Both pre-date the field identically -- same (None == None) span, not a
    # mismatch. Historical evidence pairs from before this task remain
    # comparable to each other, just not to versioned evidence.
    baseline = _evidence(_flat_events(9, 5.0), span_version=None)
    treatment = _evidence(_flat_events(9, 5.05), span_version=None)
    result = _throughput_summary(baseline, treatment)
    assert result["available"] is True


def test_absolute_ms_budget_defaults_to_unenforced():
    # A large relative percentage on a tiny absolute delta (fast baseline)
    # must still fail under percent-only gating (preserves prior behavior
    # exactly when regression_budget_ms is left at its default None).
    baseline = _evidence(_flat_events(9, 1.0))
    treatment = _evidence(_flat_events(9, 1.05))  # 5% regression, 0.05ms absolute
    result = _throughput_summary(baseline, treatment)
    assert result["regression_budget_ms"] is None
    assert result["within_ms_budget"] is True  # unenforced -> vacuously true
    assert result["within_percent_budget"] is False  # 5% > 2% default
    assert result["within_budget"] is False


def test_absolute_ms_budget_can_rescue_a_small_delta_from_a_fast_baseline():
    # Same case as above, but with a real absolute floor set (e.g. from a
    # calibration run) -- a 0.05ms delta should not fail the gate just
    # because the baseline happened to be very fast.
    baseline = _evidence(_flat_events(9, 1.0))
    treatment = _evidence(_flat_events(9, 1.05))
    result = _throughput_summary(baseline, treatment, regression_budget_ms=0.5)
    assert result["within_percent_budget"] is False
    assert result["within_ms_budget"] is True
    assert result["within_budget"] is False  # must clear BOTH floors


def test_absolute_ms_budget_can_also_fail_a_comparison_percent_alone_would_pass():
    # A small relative percentage on a large absolute delta (slow baseline)
    # -- the case the relative-only budget could not reliably catch per
    # w3_02_completion.md's noise-floor discussion.
    baseline = _evidence(_flat_events(9, 100.0))
    treatment = _evidence(_flat_events(9, 101.5))  # 1.5% regression, 1.5ms absolute
    result = _throughput_summary(baseline, treatment, regression_budget_ms=1.0)
    assert result["within_percent_budget"] is True  # 1.5% < 2% default
    assert result["within_ms_budget"] is False  # 1.5ms > 1.0ms floor
    assert result["within_budget"] is False


def test_missing_evidence_still_reports_unavailable():
    result = _throughput_summary(None, _evidence(_flat_events(3, 5.0)))
    assert result["available"] is False
    assert "missing" in result["reason"]
