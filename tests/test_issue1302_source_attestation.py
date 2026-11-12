"""Tests for issue #1302: source attestation for historical/unattested bags.

kitchen1 and floor2's historical rosbags predate `eligsir-rgbd-sync`'s
capture-provenance gate entirely (no `RGBDSynced`/`CaptureProvenance` topic
at all), so the production ROS2 ingress path refuses them outright with no
path to a real training run. See
`docs/architecture/2026-08-30-issue1302-legacy-attestation-design.md` for
the full cross-repo design; this module tests the eligsir-core half:

- `utils.source_attestation.source_attestation_from_args()` -- build the
  record from an args namespace (same shape as `git_provenance_from_args`).
- `utils.source_attestation.cap_evidence_class()` -- an unattested/legacy
  source can never push `evidence_class` up to MANAGED.
- `utils.run_summary.write_run_summary()` actually applies that cap when a
  report.json carries `source_attestation`.

Pure stdlib -- no torch dependency, mirrors tests/test_issue997_run_summary.py.
"""

from __future__ import annotations

import json
import pathlib
from types import SimpleNamespace

from eligsir.utils.run_completion_state import RunCompletionState
from eligsir.utils.run_summary import load_run_summary, write_run_summary
from eligsir.utils.source_attestation import (
    VALID_STATUSES,
    cap_evidence_class,
    source_attestation_from_args,
)


# --------------------------------------------------------------------------
# source_attestation_from_args
# --------------------------------------------------------------------------

def test_from_args_none_when_attribute_absent():
    # A hand-rolled namespace that predates this feature (or a call site
    # that forgot to set it) degrades to the pre-existing absent-key
    # behavior, exactly like git_provenance_from_args(args=None).
    assert source_attestation_from_args(None) is None
    assert source_attestation_from_args(SimpleNamespace()) is None


def test_from_args_default_unknown():
    args = SimpleNamespace(
        source_attestation_status="unknown",
        source_attestation_bag_id="",
        source_attestation_reason="",
    )
    result = source_attestation_from_args(args)
    assert result == {"status": "unknown"}


def test_from_args_legacy_unattested_with_bag_id_and_reason():
    args = SimpleNamespace(
        source_attestation_status="legacy_unattested",
        source_attestation_bag_id="sha256:deadbeef",
        source_attestation_reason="kitchen1 predates eligsir-rgbd-sync; see eligsir#1302",
    )
    result = source_attestation_from_args(args)
    assert result["status"] == "legacy_unattested"
    assert result["legacy_source_bag_id"] == "sha256:deadbeef"
    assert "eligsir#1302" in result["reason"]


def test_from_args_invalid_status_coerced_to_unknown():
    args = SimpleNamespace(
        source_attestation_status="totally_made_up",
        source_attestation_bag_id="",
        source_attestation_reason="",
    )
    result = source_attestation_from_args(args)
    assert result["status"] == "unknown"
    assert result["invalid_status_value"] == "totally_made_up"


def test_valid_statuses_contains_expected_values():
    assert set(VALID_STATUSES) == {"unknown", "legacy_unattested", "validated"}


# --------------------------------------------------------------------------
# cap_evidence_class
# --------------------------------------------------------------------------

def test_cap_evidence_class_passthrough_when_no_attestation():
    assert cap_evidence_class("MANAGED", None) == "MANAGED"
    assert cap_evidence_class("UNMANAGED", None) == "UNMANAGED"


def test_cap_evidence_class_forces_unmanaged_for_legacy():
    assert cap_evidence_class("MANAGED", {"status": "legacy_unattested"}) == "UNMANAGED"


def test_cap_evidence_class_forces_unmanaged_for_unknown():
    assert cap_evidence_class("MANAGED", {"status": "unknown"}) == "UNMANAGED"


def test_cap_evidence_class_passthrough_for_validated():
    # Reserved for a future real attestation producer -- not enforced by
    # this module, but must not be spuriously capped either.
    assert cap_evidence_class("MANAGED", {"status": "validated"}) == "MANAGED"


# --------------------------------------------------------------------------
# write_run_summary integration: the cap actually lands in run_summary.json
# --------------------------------------------------------------------------

def _base_report(iteration: int = 5000) -> dict:
    return {
        "eval_valid": True,
        "n_test": 10,
        "n_test_metrics": 10,
        "n_train": 40,
        "iteration": iteration,
        "mean_test_psnr": 24.0,
        "min_test_psnr": 22.0,
        "max_test_psnr": 26.0,
        "median_test_psnr": 24.0,
        "mean_test_ssim": 0.8,
        "mean_test_lpips": 0.2,
        "eval_metadata": {"holdout_hash": "h1"},
        "execution_semantics": {
            "trainer": "streaming",
            "iterations": iteration,
            "resolution": None,
            "device": "cuda",
            "model_layout": "gsplat",
        },
        "git_provenance": {
            "git_commit": "abc123",
            "git_branch": "agent/issue1302-legacy-attestation-design",
            "git_dirty": False,
        },
    }


def _write_report(run_dir: pathlib.Path, report_data: dict, iteration: int = 5000) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    st = RunCompletionState(str(run_dir))
    st.mark_training_complete(final_iteration=iteration)
    comp_dir = run_dir / "comparison" / f"iter_{iteration}"
    comp_dir.mkdir(parents=True, exist_ok=True)
    (comp_dir / "report.json").write_text(json.dumps(report_data, indent=2), encoding="utf-8")


def test_legacy_unattested_run_forced_unmanaged_even_with_researchflow_origin(tmp_path):
    run_dir = tmp_path / "kitchen1_legacy_run"
    report_data = _base_report()
    report_data["source_attestation"] = {
        "status": "legacy_unattested",
        "legacy_source_bag_id": "sha256:kitchen1bag",
        "reason": "predates eligsir-rgbd-sync; eligsir#1302",
    }
    # Even a run that otherwise looks fully ResearchFlow-registered must not
    # come out MANAGED -- this is the load-bearing assertion of #1302's fix:
    # an unattested source can never launder into canonical evidence just
    # because ResearchFlow registration is present.
    report_data["experiment_origin"] = {
        "researchflow_run_id": "rf-run-123",
        "researchflow_plan_sha256": "deadbeef" * 8,
        "status": "complete",
        "task_kind": "github_issue",
    }
    _write_report(run_dir, report_data)

    path = write_run_summary(str(run_dir))
    assert path is not None
    summary = load_run_summary(str(run_dir))
    assert summary["evidence_class"] == "UNMANAGED"
    assert summary["source_attestation"]["status"] == "legacy_unattested"
    assert summary["experiment_origin"]["evidence_class"] == "UNMANAGED"


def test_run_without_source_attestation_field_is_unaffected(tmp_path):
    # A run that predates #1302 (no source_attestation key at all in
    # report.json) must resolve evidence_class exactly as before this
    # feature existed.
    run_dir = tmp_path / "pre_1302_run"
    report_data = _base_report()
    report_data["experiment_origin"] = {
        "researchflow_run_id": "rf-run-456",
        "researchflow_plan_sha256": "cafebabe" * 8,
        "status": "complete",
        "task_kind": "github_issue",
    }
    _write_report(run_dir, report_data)

    path = write_run_summary(str(run_dir))
    assert path is not None
    summary = load_run_summary(str(run_dir))
    assert summary["evidence_class"] == "MANAGED"
    assert summary.get("source_attestation") is None


def test_validated_source_attestation_does_not_block_managed(tmp_path):
    run_dir = tmp_path / "validated_run"
    report_data = _base_report()
    report_data["source_attestation"] = {"status": "validated"}
    report_data["experiment_origin"] = {
        "researchflow_run_id": "rf-run-789",
        "researchflow_plan_sha256": "0badc0de" * 8,
        "status": "complete",
        "task_kind": "github_issue",
    }
    _write_report(run_dir, report_data)

    path = write_run_summary(str(run_dir))
    assert path is not None
    summary = load_run_summary(str(run_dir))
    assert summary["evidence_class"] == "MANAGED"
    assert summary["source_attestation"]["status"] == "validated"
