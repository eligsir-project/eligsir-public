"""Post-hoc coverage-v2 reconstruction is evidence recovery, not inference."""

from __future__ import annotations

import json

from eligsir.utils.trajectory_coverage_regrade import reconstruct_coverage_v2


def _report(*, source: int, admitted: int, holdout: int) -> dict:
    return {
        "trajectory_coverage": {
            "n_source_frames": source,
            "frames_admitted": admitted,
            "final_iteration": 3000,
        },
        "eval_metadata": {
            "n_holdout": holdout, "n_guard": 0,
            "holdout_hash": "bound", "realized_holdout_hash": "bound",
            "scored_holdout_hash": "bound",
        },
        "report_provenance": {"kind": "training_loop"},
    }


def _write_log(path, admitted: int, skipped: int) -> None:
    with path.open("w") as handle:
        for index in range(admitted + skipped):
            handle.write(json.dumps({"frame": index, "iter": index + 1, "admitted": index < admitted}) + "\n")


def _write_audit_bindings(path) -> None:
    (path / "run_completion_state.json").write_text(
        json.dumps({"run_complete": True, "final_iteration": 3000}), encoding="utf-8"
    )
    declaration = {"experiment_declaration_sha256": "declared"}
    (path / "experiment_declaration.json").write_text(json.dumps(declaration), encoding="utf-8")
    (path / "experiment_declaration_receipt.json").write_text(json.dumps(declaration), encoding="utf-8")


def test_494_style_sidecars_are_rerun_required_without_immutable_ledger(tmp_path):
    """Requested bootstrap plus append rows cannot recover source truth."""
    (tmp_path / "cfg_args").write_text(
        "Namespace(streaming_initial_frames=5)\n", encoding="utf-8"
    )
    _write_audit_bindings(tmp_path)
    _write_log(tmp_path / "keyframe_admission_log.jsonl", admitted=585, skipped=0)

    result = reconstruct_coverage_v2(
        tmp_path,
        _report(source=596, admitted=585, holdout=6),
    )

    assert result["status"] == "RERUN_REQUIRED"
    assert "immutable_coverage_ledger_unavailable" in result["reasons"]


def test_499_style_sidecars_are_rerun_required_without_immutable_ledger(tmp_path):
    """Copied declaration strings and monotonic rows are not a ledger."""
    (tmp_path / "cfg_args").write_text(
        "Namespace(streaming_initial_frames=5)\n", encoding="utf-8"
    )
    _write_audit_bindings(tmp_path)
    _write_log(tmp_path / "keyframe_admission_log.jsonl", admitted=2005, skipped=0)

    result = reconstruct_coverage_v2(
        tmp_path,
        _report(source=2297, admitted=2005, holdout=287),
    )

    assert result["status"] == "RERUN_REQUIRED"
    assert "immutable_coverage_ledger_unavailable" in result["reasons"]


def test_refuses_regrade_when_admission_log_cannot_account_for_eligible_frames(tmp_path):
    """A ratio and a final count alone must never fabricate skipped frames."""
    (tmp_path / "cfg_args").write_text(
        "Namespace(streaming_initial_frames=5)\n", encoding="utf-8"
    )
    _write_log(tmp_path / "keyframe_admission_log.jsonl", admitted=584, skipped=0)

    result = reconstruct_coverage_v2(
        tmp_path,
        _report(source=596, admitted=585, holdout=6),
    )

    assert result["status"] == "RERUN_REQUIRED"
    assert "admission_log_admitted_mismatch" in result["reasons"]
