"""Tests for issue #997: Root-level run_summary.json generic application-to-orchestrator handoff.

Verifies:
1. Successful run writes <model_path>/run_summary.json with all 9 canonical fields:
   - schema: "eligsir.run-summary/v1"
   - completion_state: dict from load_run_completion_state (schema 1)
   - reconstruction_metrics: final PSNR, SSIM, LPIPS, holdout counts from canonical report.json
   - eval_metadata: holdout_hash, eval validity, scored view count, seed fade status
   - execution_semantics: trainer type, iterations, resolution, device, model layout
   - git_provenance: commit SHA, branch, dirty status
   - experiment_origin: dict loaded from experiment_origin.json if present
   - report_location: relative path to canonical report.json if present
   - created_at: ISO timestamp
2. Failed report cannot generate a success-looking summary (completion_state reflects failure).
3. Missing required evidence remains invalid.
4. Rerun is idempotent and atomic.
5. Relocated run remains readable with relative paths.
6. External process can recover completion + metrics + provenance from run_summary.json alone.
7. No report-directory globbing required (direct relative pointer).
8. Pure stdlib in utils/run_summary.py (no torch dependency).
9. AST call-site verification on train.py and train_streaming.py.
"""

from __future__ import annotations

import ast
import json
import os
import pathlib
import shutil
import pytest

from eligsir.utils.run_completion_state import (
    FINAL_REPORT_CONTRACT,
    Outcome,
    PhaseStatus,
    RunCompletionState,
    SCHEMA_VERSION as COMPLETION_SCHEMA_VERSION,
    STATE_FILENAME,
)
from eligsir.utils.run_summary import (
    SUMMARY_FILENAME,
    SUMMARY_SCHEMA,
    load_run_summary,
    write_run_summary,
)

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
TRAIN_PY = REPO_ROOT / "train.py"
TRAIN_STREAMING_PY = REPO_ROOT / "train_streaming.py"


# --------------------------------------------------------------------------
# Fixtures & Helpers
# --------------------------------------------------------------------------

def _setup_successful_run(
    run_dir: str | pathlib.Path,
    iteration: int = 30000,
    psnr: float = 27.5,
    ssim: float = 0.88,
    lpips: float = 0.12,
) -> pathlib.Path:
    run_dir = pathlib.Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # 1. run_completion_state.json
    st = RunCompletionState(str(run_dir))
    st.mark_training_complete(final_iteration=iteration)
    comp_dir = run_dir / "comparison" / f"iter_{iteration}"
    comp_dir.mkdir(parents=True, exist_ok=True)
    report_json_path = comp_dir / "report.json"

    # 2. canonical report.json
    report_data = {
        "eval_valid": True,
        "n_test": 50,
        "n_test_metrics": 50,
        "n_train": 200,
        "iteration": iteration,
        "mean_test_psnr": psnr,
        "min_test_psnr": psnr - 2.0,
        "max_test_psnr": psnr + 2.0,
        "median_test_psnr": psnr,
        "mean_test_ssim": ssim,
        "mean_test_lpips": lpips,
        "eval_metadata": {
            "holdout_hash": "abc123holdout",
            "realized_holdout_hash": "def456realized",
            "scored_holdout_hash": "ghi789scored",
            "n_scored": 50,
        },
        "execution_semantics": {
            "trainer": "offline",
            "iterations": iteration,
            "resolution": [800, 800],
            "device": "cuda",
            "model_layout": "gsplat",
        },
        "git_provenance": {
            "git_commit": "deadbeef12345678",
            "git_branch": "main",
            "git_dirty": False,
            "git_dirty_paths": [],
        },
        "experiment_origin": {
            "repository": "eligsir-project/eligsir",
            "issue": 997,
            "task_kind": "github_issue",
            "experiment_id": "exp_summary_v1",
            "status": "complete",
        },
        "seed_fade": {
            "detected": False,
            "confounded": False,
            "severity": "none",
            "status": "clean",
        },
    }
    report_json_path.write_text(json.dumps(report_data, indent=2), encoding="utf-8")

    with st.phase(FINAL_REPORT_CONTRACT) as ph:
        ph.record_output(str(comp_dir))
    st.write()

    # 3. experiment_origin.json
    origin_path = run_dir / "experiment_origin.json"
    origin_path.write_text(json.dumps(report_data["experiment_origin"], indent=2), encoding="utf-8")

    return run_dir


# --------------------------------------------------------------------------
# 1. Successful run writes run_summary.json with all canonical fields
# --------------------------------------------------------------------------

def test_successful_run_writes_run_summary_with_all_fields(tmp_path):
    run_dir = _setup_successful_run(tmp_path / "run_success")

    out_path = write_run_summary(str(run_dir))
    assert out_path is not None
    assert os.path.isfile(out_path)
    assert os.path.basename(out_path) == SUMMARY_FILENAME

    summary = load_run_summary(str(run_dir))
    assert summary is not None

    # Check 1: schema
    assert summary["schema"] == SUMMARY_SCHEMA == "eligsir.run-summary/v1"

    # Check 2: completion_state
    comp = summary["completion_state"]
    assert comp is not None
    assert comp["schema_version"] == COMPLETION_SCHEMA_VERSION == 1
    assert comp["outcome"] == Outcome.RUN_COMPLETE
    assert comp["run_complete"] is True
    assert comp["scientific_result_complete"] is True
    assert comp["training_complete"] is True
    assert comp["report_complete"] is True
    assert comp["required_artifacts_present"] is True

    # Check 3: reconstruction_metrics
    metrics = summary["reconstruction_metrics"]
    assert metrics is not None
    assert metrics["mean_test_psnr"] == 27.5
    assert metrics["mean_test_ssim"] == 0.88
    assert metrics["mean_test_lpips"] == 0.12
    assert metrics["n_test"] == 50
    assert metrics["n_test_metrics"] == 50
    assert metrics["n_train"] == 200
    assert metrics["iteration"] == 30000

    # Check 4: eval_metadata
    eval_meta = summary["eval_metadata"]
    assert eval_meta is not None
    assert eval_meta["eval_valid"] is True
    assert eval_meta["holdout_hash"] == "abc123holdout"
    assert eval_meta["realized_holdout_hash"] == "def456realized"
    assert eval_meta["scored_holdout_hash"] == "ghi789scored"
    assert eval_meta["n_scored"] == 50
    assert eval_meta["seed_fade_detected"] is False
    assert eval_meta["seed_fade_confounded"] is False

    # Check 5: execution_semantics
    exec_sem = summary["execution_semantics"]
    assert exec_sem is not None
    assert exec_sem["trainer"] == "offline"
    assert exec_sem["iterations"] == 30000
    assert exec_sem["resolution"] == [800, 800]
    assert exec_sem["device"] == "cuda"
    assert exec_sem["model_layout"] == "gsplat"

    # Check 6: git_provenance
    git_prov = summary["git_provenance"]
    assert git_prov is not None
    assert git_prov["git_commit"] == "deadbeef12345678"
    assert git_prov["git_branch"] == "main"
    assert git_prov["git_dirty"] is False

    # Check 7: experiment_origin
    origin = summary["experiment_origin"]
    assert origin is not None
    assert origin["repository"] == "eligsir-project/eligsir"
    assert origin["issue"] == 997
    assert origin["task_kind"] == "github_issue"

    # Check 8: report_location (relative path)
    assert summary["report_location"] == "comparison/iter_30000/report.json"

    # Check 9: created_at (ISO timestamp)
    assert isinstance(summary["created_at"], str)
    assert "T" in summary["created_at"]


# --------------------------------------------------------------------------
# 2. Failed report cannot generate success-looking summary
# --------------------------------------------------------------------------

def test_failed_report_cannot_generate_success_summary(tmp_path):
    run_dir = tmp_path / "run_failed_report"
    run_dir.mkdir(parents=True)

    # State records REPORT_FAILED
    st = RunCompletionState(str(run_dir))
    st.mark_training_complete(final_iteration=30000)
    with st.phase(FINAL_REPORT_CONTRACT):
        raise RuntimeError("CUDA out of memory during report rendering")
    st.write()

    write_run_summary(str(run_dir))

    summary = load_run_summary(str(run_dir))
    assert summary is not None
    assert summary["schema"] == SUMMARY_SCHEMA

    # Completion state reflects failure
    comp = summary["completion_state"]
    assert comp["outcome"] == Outcome.REPORT_FAILED
    assert comp["run_complete"] is False
    assert comp["scientific_result_complete"] is False
    assert comp["report_complete"] is False
    assert comp["failure_type"] == "RuntimeError"
    assert "CUDA out of memory" in comp["failure_reason"]

    # Metrics and report location must not imply success
    assert summary["report_location"] is None
    assert summary["reconstruction_metrics"] is None


# --------------------------------------------------------------------------
# 3. Missing required evidence remains invalid
# --------------------------------------------------------------------------

def test_missing_required_evidence_remains_invalid(tmp_path):
    run_dir = tmp_path / "run_missing_evidence"
    run_dir.mkdir(parents=True)

    # State records MISSING_ARTIFACTS
    st = RunCompletionState(str(run_dir))
    st.mark_training_complete(final_iteration=30000)
    comp_dir = run_dir / "comparison" / "iter_30000"
    comp_dir.mkdir(parents=True)
    # Note: no report.json created under comp_dir!

    with st.phase(FINAL_REPORT_CONTRACT) as ph:
        ph.record_output(str(comp_dir))
    st.write()

    write_run_summary(str(run_dir))

    summary = load_run_summary(str(run_dir))
    assert summary is not None
    comp = summary["completion_state"]
    assert comp["outcome"] == Outcome.REPORT_ARTIFACTS_MISSING
    assert comp["required_artifacts_present"] is False
    assert comp["run_complete"] is False
    assert comp["scientific_result_complete"] is False
    assert summary["report_location"] is None


# --------------------------------------------------------------------------
# 4. Rerun is idempotent and atomic
# --------------------------------------------------------------------------

def test_rerun_is_idempotent_and_atomic(tmp_path):
    run_dir = _setup_successful_run(tmp_path / "run_idempotent", psnr=26.0)

    p1 = write_run_summary(str(run_dir))
    s1 = load_run_summary(str(run_dir))
    assert s1["reconstruction_metrics"]["mean_test_psnr"] == 26.0

    # Simulate updated metrics / rerun rewrite
    report_path = run_dir / "comparison" / "iter_30000" / "report.json"
    rep = json.loads(report_path.read_text(encoding="utf-8"))
    rep["mean_test_psnr"] = 28.5
    report_path.write_text(json.dumps(rep, indent=2), encoding="utf-8")

    p2 = write_run_summary(str(run_dir))
    assert p1 == p2
    s2 = load_run_summary(str(run_dir))
    assert s2["reconstruction_metrics"]["mean_test_psnr"] == 28.5

    # Check that no temporary files remain
    assert not os.path.exists(os.path.join(run_dir, f"{SUMMARY_FILENAME}.tmp"))


# --------------------------------------------------------------------------
# 5. Relocated run remains readable with relative paths
# --------------------------------------------------------------------------

def test_relocated_run_remains_readable_with_relative_paths(tmp_path):
    orig_dir = _setup_successful_run(tmp_path / "orig_dir")
    write_run_summary(str(orig_dir))

    s_orig = load_run_summary(str(orig_dir))
    rel_path = s_orig["report_location"]
    assert rel_path == "comparison/iter_30000/report.json"
    assert not os.path.isabs(rel_path)

    # Move entire directory tree to a new location
    moved_dir = tmp_path / "relocated" / "destination"
    shutil.copytree(orig_dir, moved_dir)

    s_moved = load_run_summary(str(moved_dir))
    assert s_moved is not None
    assert s_moved["report_location"] == "comparison/iter_30000/report.json"

    # Resolving relative path against moved directory finds the report
    resolved_report = moved_dir / s_moved["report_location"]
    assert resolved_report.is_file()
    with open(resolved_report, "r", encoding="utf-8") as f:
        rep = json.load(f)
    assert rep["mean_test_psnr"] == 27.5


# --------------------------------------------------------------------------
# 6. External process can recover completion + metrics + provenance from run_summary.json alone
# --------------------------------------------------------------------------

def test_external_process_can_recover_from_run_summary_alone(tmp_path):
    run_dir = _setup_successful_run(tmp_path / "run_handoff")
    write_run_summary(str(run_dir))

    # Read ONLY run_summary.json directly via standard json.load (mirrors ablator/orchestrator)
    summary_file = run_dir / SUMMARY_FILENAME
    assert summary_file.is_file()
    summary = json.loads(summary_file.read_text(encoding="utf-8"))

    assert summary["schema"] == "eligsir.run-summary/v1"
    assert summary["completion_state"]["outcome"] == "RUN_COMPLETE"
    assert summary["reconstruction_metrics"]["mean_test_psnr"] == 27.5
    assert summary["git_provenance"]["git_commit"] == "deadbeef12345678"
    assert summary["experiment_origin"]["issue"] == 997
    assert summary["eval_metadata"]["eval_valid"] is True
    assert summary["execution_semantics"]["trainer"] == "offline"


# --------------------------------------------------------------------------
# 7. No report-directory globbing required (direct relative pointer)
# --------------------------------------------------------------------------

def test_no_report_directory_globbing_required(tmp_path):
    run_dir = tmp_path / "run_multi_reports"
    run_dir.mkdir(parents=True)

    st = RunCompletionState(str(run_dir))
    st.mark_training_complete(final_iteration=50000)

    # Create multiple intermediate/debug reports
    for it in (10000, 20000, 50000):
        c_dir = run_dir / "comparison" / f"iter_{it}"
        c_dir.mkdir(parents=True)
        (c_dir / "report.json").write_text(json.dumps({
            "iteration": it,
            "mean_test_psnr": 20.0 + (it / 10000.0),
            "eval_valid": True,
            "n_test": 10,
        }))

    # Also a debug bootstrap view report
    b_dir = run_dir / "comparison" / "iter_50000_bootstrap_views"
    b_dir.mkdir(parents=True)
    (b_dir / "report.json").write_text(json.dumps({"iteration": 50000, "degenerate_split": True}))

    with st.phase(FINAL_REPORT_CONTRACT) as ph:
        ph.record_output(str(run_dir / "comparison" / "iter_50000"))
    st.write()

    write_run_summary(str(run_dir))

    summary = load_run_summary(str(run_dir))
    # Direct pointer to the canonical final report (iter_50000), not intermediate or bootstrap
    assert summary["report_location"] == "comparison/iter_50000/report.json"
    assert summary["reconstruction_metrics"]["iteration"] == 50000
    assert summary["reconstruction_metrics"]["mean_test_psnr"] == 25.0


# --------------------------------------------------------------------------
# 8. Handling disabled report (--no-streaming_report_final / benchmark)
# --------------------------------------------------------------------------

def test_disabled_report_summary_semantics(tmp_path):
    run_dir = tmp_path / "run_no_report"
    run_dir.mkdir(parents=True)

    st = RunCompletionState(str(run_dir))
    st.mark_training_complete(final_iteration=10000)
    st.mark_not_requested(FINAL_REPORT_CONTRACT, reason="--streaming_benchmark_mode")
    st.write()

    write_run_summary(str(run_dir))

    summary = load_run_summary(str(run_dir))
    assert summary is not None
    assert summary["completion_state"]["outcome"] == Outcome.RUN_COMPLETE_NO_REPORT_REQUESTED
    assert summary["completion_state"]["scientific_result_complete"] is False
    assert summary["report_location"] is None
    assert summary["reconstruction_metrics"] is None


# --------------------------------------------------------------------------
# 9. load_run_summary error and missing file handling
# --------------------------------------------------------------------------

def test_load_run_summary_missing_file_returns_none(tmp_path):
    assert load_run_summary(str(tmp_path / "non_existent")) is None
    assert load_run_summary("") is None


# --------------------------------------------------------------------------
# 10. Pure stdlib check (no torch in utils/run_summary.py)
# --------------------------------------------------------------------------

def test_utils_run_summary_imports_no_torch():
    src = (REPO_ROOT / "utils" / "run_summary.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
        else:
            continue
        for name in names:
            assert not name.split(".")[0] == "torch", f"torch imported via {name}"


# --------------------------------------------------------------------------
# 11. AST call site verification on train.py and train_streaming.py
# --------------------------------------------------------------------------

def test_entrypoints_wire_write_run_summary_before_finalize_staging():
    for file_path in (TRAIN_PY, TRAIN_STREAMING_PY):
        src = file_path.read_text(encoding="utf-8")
        assert "write_run_summary" in src, f"{file_path.name} must call write_run_summary"
        write_state_at = src.rindex("_run_state.write()")
        write_summary_at = src.rindex("write_run_summary(")
        finalize_at = src.rindex("_finalize_output_staging(args)")
        raise_at = src.rindex("_run_state.raise_if_incomplete()")

        assert write_state_at <= write_summary_at < finalize_at < raise_at, (
            f"Ordering in {file_path.name} must be: "
            "_run_state.write() -> write_run_summary() -> _finalize_output_staging() -> _run_state.raise_if_incomplete()"
        )
