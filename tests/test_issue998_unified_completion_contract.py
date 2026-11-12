"""Issue #998: Unified completion contract across offline (train.py) and streaming (train_streaming.py) runs.

Verifies:
1. Offline (train.py) and streaming (train_streaming.py) share the same canonical
   RunCompletionState implementation, schema version 1, and STATE_FILENAME.
2. Complete lifecycle: TRAINING_COMPLETE -> reporting/evidence -> REPORT_COMPLETE -> RUN_COMPLETE.
3. Failures in reporting or missing required report artifacts result in explicit
   states (REPORT_FAILED, REPORT_ARTIFACTS_MISSING) and cause fail-closed non-zero exit.
4. Intentionally disabled reporting (--no-streaming_report_final, --streaming_benchmark_mode)
   results in RUN_COMPLETE_NO_REPORT_REQUESTED and is NOT classified as scientific RUN_COMPLETE.
5. Unqualified 'Training complete.' print is prevented when report phase fails.
6. The state is atomically persisted to <model_path>/run_completion_state.json before exit.
7. Pure stdlib in utils/run_completion_state.py (no torch dependency).
8. Historical runs without run_completion_state.json are classified as UNKNOWN / UNKNOWN_LEGACY.
9. AST call-site verification on train.py and train_streaming.py ensuring proper phase wrapping.
"""

import ast
import json
import pathlib
import pytest

from eligsir.utils.run_completion_state import (
    BOOTSTRAP_VIEWS_CONTRACT,
    FINAL_REPORT_CONTRACT,
    OPTIMIZATION_CURVE_CONTRACT,
    Outcome,
    PhaseContract,
    PhaseResult,
    PhaseStatus,
    RunCompletionState,
    RunIncompleteError,
    SCHEMA_VERSION,
    STATE_FILENAME,
    TRAJECTORY_EVAL_CONTRACT,
    get_run_outcome,
    load_run_completion_state,
)

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
TRAIN_PY = REPO_ROOT / "train.py"
TRAIN_STREAMING_PY = REPO_ROOT / "train_streaming.py"


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _state(tmp_path):
    lines = []
    st = RunCompletionState(str(tmp_path), printer=lines.append)
    return st, lines


def _write_report_json(model_path, subdir="comparison/iter_30000"):
    out_dir = pathlib.Path(model_path) / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text("{}")
    return str(out_dir)


def _persisted(tmp_path):
    return json.loads((pathlib.Path(tmp_path) / STATE_FILENAME).read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# 1. Offline report succeeds + required artifact exists -> RUN_COMPLETE, exit 0
# --------------------------------------------------------------------------

def test_offline_report_succeeds_with_required_artifacts(tmp_path):
    st, lines = _state(tmp_path)
    st.mark_training_complete(final_iteration=30000)

    out_dir = _write_report_json(tmp_path, "comparison/iter_30000")
    with st.phase(FINAL_REPORT_CONTRACT) as ph:
        ph.record_output({"output_dir": out_dir, "n_test": 50})

    phase = st.get("final_report")
    assert phase is not None
    assert phase.status == PhaseStatus.SUCCESS
    assert phase.error is None
    assert any("✓" in line and "done" in line for line in lines)

    assert st.training_complete is True
    assert st.report_complete is True
    assert st.required_artifacts_present is True
    assert st.run_complete is True
    assert st.scientific_result_complete is True
    assert st.outcome == Outcome.RUN_COMPLETE
    st.raise_if_incomplete()  # must not raise (exit 0)

    disk = _persisted(tmp_path)
    assert disk["schema_version"] == SCHEMA_VERSION
    assert disk["outcome"] == Outcome.RUN_COMPLETE
    assert disk["run_complete"] is True
    assert disk["scientific_result_complete"] is True
    assert disk["training_complete"] is True
    assert disk["report_requested"] is True
    assert disk["report_complete"] is True
    assert disk["failure_type"] is None
    assert disk["failure_reason"] is None


# --------------------------------------------------------------------------
# 2. Offline report raises -> REPORT_FAILED, failure reason, non-zero exit
# --------------------------------------------------------------------------

def test_offline_report_raises_marked_report_failed_nonzero_exit(tmp_path):
    st, lines = _state(tmp_path)
    st.mark_training_complete(final_iteration=30000)

    with st.phase(FINAL_REPORT_CONTRACT) as ph:
        raise RuntimeError("GPU OOM during test set rendering")

    phase = st.get("final_report")
    assert phase is not None
    assert phase.status == PhaseStatus.FAILED
    assert phase.error_type == "RuntimeError"
    assert "GPU OOM" in phase.error

    assert not any("✓" in line for line in lines)
    assert any("✗" in line and "FAILED" in line for line in lines)

    assert st.training_complete is True
    assert st.report_complete is False
    assert st.run_complete is False
    assert st.outcome == Outcome.REPORT_FAILED
    assert st.failure_type == "RuntimeError"
    assert "GPU OOM" in st.failure_reason

    disk = _persisted(tmp_path)
    assert disk["outcome"] == Outcome.REPORT_FAILED
    assert disk["run_complete"] is False
    assert disk["failure_type"] == "RuntimeError"
    assert "GPU OOM" in disk["failure_reason"]

    with pytest.raises(RunIncompleteError) as excinfo:
        st.raise_if_incomplete()
    assert "REPORT_FAILED" in str(excinfo.value)


# --------------------------------------------------------------------------
# 3. Offline report returns but required artifact missing -> REPORT_ARTIFACTS_MISSING
# --------------------------------------------------------------------------

def test_offline_report_returns_without_required_artifact_fails_closed(tmp_path):
    st, lines = _state(tmp_path)
    st.mark_training_complete(final_iteration=30000)

    out_dir = tmp_path / "comparison" / "iter_30000"
    out_dir.mkdir(parents=True)
    (out_dir / "test_contact_sheet.png").write_text("dummy")

    with st.phase(FINAL_REPORT_CONTRACT) as ph:
        ph.record_output({"output_dir": str(out_dir)})

    phase = st.get("final_report")
    assert phase is not None
    assert phase.status == PhaseStatus.MISSING_ARTIFACTS
    assert phase.missing_artifacts == ["report.json"]
    assert not any("✓" in line for line in lines)

    assert st.required_artifacts_present is False
    assert st.run_complete is False
    assert st.outcome == Outcome.REPORT_ARTIFACTS_MISSING

    disk = _persisted(tmp_path)
    assert disk["outcome"] == Outcome.REPORT_ARTIFACTS_MISSING
    assert disk["run_complete"] is False
    assert disk["required_artifacts_present"] is False
    assert "MissingRequiredArtifacts" in (disk["failure_type"] or "")

    with pytest.raises(RunIncompleteError):
        st.raise_if_incomplete()


def test_offline_report_returns_without_recording_output_dir_fails_closed(tmp_path):
    st, _ = _state(tmp_path)
    st.mark_training_complete(final_iteration=30000)

    with st.phase(FINAL_REPORT_CONTRACT):
        pass  # forgets to record output dir

    assert st.get("final_report").status == PhaseStatus.MISSING_ARTIFACTS
    assert st.run_complete is False
    assert st.outcome == Outcome.REPORT_ARTIFACTS_MISSING
    with pytest.raises(RunIncompleteError):
        st.raise_if_incomplete()


# --------------------------------------------------------------------------
# 4. Offline report disabled -> RUN_COMPLETE_NO_REPORT_REQUESTED
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "reason", ["--no-streaming_report_final", "--streaming_benchmark_mode"]
)
def test_offline_report_disabled_is_no_report_requested(tmp_path, reason):
    st, lines = _state(tmp_path)
    st.mark_training_complete(final_iteration=30000)
    st.mark_not_requested(FINAL_REPORT_CONTRACT, reason=reason)

    phase = st.get("final_report")
    assert phase is not None
    assert phase.status == PhaseStatus.NOT_REQUESTED
    assert phase.note == reason
    assert phase.error is None

    assert st.blocking_failures == []
    assert st.report_complete is True
    assert st.run_complete is True
    st.raise_if_incomplete()  # exit 0

    assert st.report_requested is False
    assert st.scientific_result_complete is False
    assert st.outcome == Outcome.RUN_COMPLETE_NO_REPORT_REQUESTED

    disk = _persisted(tmp_path)
    assert disk["outcome"] == Outcome.RUN_COMPLETE_NO_REPORT_REQUESTED
    assert disk["run_complete"] is True
    assert disk["scientific_result_complete"] is False
    assert disk["report_requested"] is False


# --------------------------------------------------------------------------
# 5. Parity & Optional Phase Handling across offline and streaming
# --------------------------------------------------------------------------

def test_optional_progress_video_phase_failure_does_not_downgrade_run(tmp_path):
    st, lines = _state(tmp_path)
    st.mark_training_complete(final_iteration=30000)

    # Optional phase: training progress video fails
    video_contract = PhaseContract(
        name="training_progress_video",
        label="training_progress.mp4 (100 frames)",
        required=False,
    )
    with st.phase(video_contract):
        raise RuntimeError("ffmpeg encode error")

    out_dir = _write_report_json(tmp_path, "comparison/iter_30000")
    with st.phase(FINAL_REPORT_CONTRACT) as ph:
        ph.record_output(out_dir)

    assert st.get("training_progress_video").status == PhaseStatus.FAILED
    assert st.get("final_report").status == PhaseStatus.SUCCESS
    assert st.run_complete is True
    assert st.outcome == Outcome.RUN_COMPLETE
    st.raise_if_incomplete()  # does not raise

    disk = _persisted(tmp_path)
    assert disk["outcome"] == Outcome.RUN_COMPLETE
    phases_by_name = {p["name"]: p for p in disk["phases"]}
    assert phases_by_name["training_progress_video"]["status"] == PhaseStatus.FAILED
    assert phases_by_name["training_progress_video"]["required"] is False
    assert phases_by_name["final_report"]["status"] == PhaseStatus.SUCCESS


def test_schema_field_parity_between_offline_and_streaming(tmp_path):
    """Offline and streaming produce exact identical schema keys and version."""
    offline_dir = tmp_path / "offline"
    streaming_dir = tmp_path / "streaming"

    st_off = RunCompletionState(str(offline_dir))
    st_off.mark_training_complete(30000)
    out_off = _write_report_json(offline_dir, "comparison/iter_30000")
    with st_off.phase(FINAL_REPORT_CONTRACT) as ph:
        ph.record_output(out_off)
    st_off.write()

    st_str = RunCompletionState(str(streaming_dir))
    st_str.mark_training_complete(12000)
    out_str = _write_report_json(streaming_dir, "comparison/iter_12000")
    with st_str.phase(FINAL_REPORT_CONTRACT) as ph:
        ph.record_output(out_str)
    with st_str.phase(BOOTSTRAP_VIEWS_CONTRACT) as ph:
        ph.record_output(streaming_dir / "iter_12000_bootstrap_views")
    with st_str.phase(TRAJECTORY_EVAL_CONTRACT):
        pass
    with st_str.phase(OPTIMIZATION_CURVE_CONTRACT):
        pass
    st_str.write()

    disk_off = json.loads((offline_dir / STATE_FILENAME).read_text(encoding="utf-8"))
    disk_str = json.loads((streaming_dir / STATE_FILENAME).read_text(encoding="utf-8"))

    assert disk_off["schema_version"] == SCHEMA_VERSION == 1
    assert disk_str["schema_version"] == SCHEMA_VERSION == 1
    assert set(disk_off.keys()) == set(disk_str.keys())
    expected_keys = {
        "schema_version",
        "issue",
        "updated_at",
        "model_path",
        "final_iteration",
        "outcome",
        "training_complete",
        "report_requested",
        "report_complete",
        "required_artifacts_present",
        "run_complete",
        "scientific_result_complete",
        "failure_type",
        "failure_reason",
        "phases",
    }
    assert set(disk_off.keys()) == expected_keys

    # Phase dictionaries share exact same structure
    off_phase_keys = set(disk_off["phases"][0].keys())
    str_phase_keys = set(disk_str["phases"][0].keys())
    assert off_phase_keys == str_phase_keys
    assert off_phase_keys == {
        "name",
        "label",
        "required",
        "status",
        "duration_s",
        "output_dir",
        "error_type",
        "error",
        "missing_artifacts",
        "required_artifacts",
        "note",
    }


# --------------------------------------------------------------------------
# 6. State written before exit and pending state preserved on crash
# --------------------------------------------------------------------------

def test_state_written_atomically_and_pending_during_phase(tmp_path):
    st, _ = _state(tmp_path)
    st.mark_training_complete(30000)

    with st.phase(FINAL_REPORT_CONTRACT):
        mid = _persisted(tmp_path)
        assert mid["phases"][0]["status"] == PhaseStatus.PENDING
        assert mid["run_complete"] is False
        assert mid["outcome"] == Outcome.REPORT_INCOMPLETE
        raise SystemExit("simulated kill mid-phase")

    disk = _persisted(tmp_path)
    assert disk["outcome"] == Outcome.REPORT_FAILED
    assert disk["run_complete"] is False


# --------------------------------------------------------------------------
# 7. Historical run without state is unknown / legacy
# --------------------------------------------------------------------------

def test_historical_run_without_state_is_unknown_legacy(tmp_path):
    empty_dir = tmp_path / "historical_run"
    empty_dir.mkdir(parents=True)

    loaded = load_run_completion_state(str(empty_dir))
    assert loaded is None

    outcome = get_run_outcome(str(empty_dir))
    assert outcome == Outcome.UNKNOWN_LEGACY

    assert Outcome.UNKNOWN == "UNKNOWN"
    assert Outcome.UNKNOWN_LEGACY == "UNKNOWN_LEGACY"
    assert Outcome.LEGACY == "LEGACY"


def test_load_run_completion_state_reads_existing_state(tmp_path):
    st, _ = _state(tmp_path)
    st.mark_training_complete(30000)
    out_dir = _write_report_json(tmp_path)
    with st.phase(FINAL_REPORT_CONTRACT) as ph:
        ph.record_output(out_dir)

    loaded = load_run_completion_state(str(tmp_path))
    assert loaded is not None
    assert loaded["outcome"] == Outcome.RUN_COMPLETE
    assert get_run_outcome(str(tmp_path)) == Outcome.RUN_COMPLETE


# --------------------------------------------------------------------------
# 8. Pure stdlib check (no torch)
# --------------------------------------------------------------------------

def test_utils_run_completion_state_imports_no_torch():
    src = (REPO_ROOT / "utils" / "run_completion_state.py").read_text(encoding="utf-8")
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
# 9. Call site AST analysis for train.py (parity with train_streaming.py)
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def train_py_tree():
    return ast.parse(TRAIN_PY.read_text(encoding="utf-8"))


def _find_phase_with_blocks(tree, contract_name):
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.With):
            continue
        for item in node.items:
            call = item.context_expr
            if not isinstance(call, ast.Call):
                continue
            fn = call.func
            if (
                isinstance(fn, ast.Attribute)
                and fn.attr == "phase"
                and isinstance(fn.value, ast.Name)
                and fn.value.id == "_run_state"
                and call.args
                and isinstance(call.args[0], ast.Name)
                and call.args[0].id == contract_name
            ):
                found.append(node)
    return found


def test_train_py_wraps_final_report_in_fail_closed_phase(train_py_tree):
    blocks = _find_phase_with_blocks(train_py_tree, "FINAL_REPORT_CONTRACT")
    assert len(blocks) == 1, f"exactly one final-report phase expected in train.py, found {len(blocks)}"
    body_calls = [
        n.func.id
        for n in ast.walk(blocks[0])
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
    ]
    assert "write_post_training_report" in body_calls
    assert "record_output" in [
        n.func.attr
        for n in ast.walk(blocks[0])
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
    ], "train.py final report phase must record its output directory"


def test_train_py_report_imports_are_inside_the_phase_body(train_py_tree):
    blocks = _find_phase_with_blocks(train_py_tree, "FINAL_REPORT_CONTRACT")
    assert blocks, "no FINAL_REPORT_CONTRACT phase found in train.py"
    modules = {
        n.module
        for block in blocks
        for n in ast.walk(block)
        if isinstance(n, ast.ImportFrom)
    }
    assert "utils.comparison_report" in modules


def test_train_py_has_no_bare_except_around_final_report(train_py_tree):
    for node in ast.walk(train_py_tree):
        if not isinstance(node, ast.Try):
            continue
        calls = [
            n.func.id
            for n in ast.walk(node)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
        ]
        if "write_post_training_report" in calls:
            pytest.fail("write_post_training_report in train.py must not be swallowed by a bare try/except")


def test_train_py_fails_closed_before_returning():
    src = TRAIN_PY.read_text(encoding="utf-8")
    assert "_run_state.raise_if_incomplete()" in src
    write_at = src.index("_run_state.write()\n    _run_state.print_summary()")
    # The causal-mapping early return owns a separate, already-tested staging
    # call. This assertion targets the offline completion-state endpoint.
    stage_at = src.rindex("_finalize_output_staging(args)")
    raise_at = src.index("_run_state.raise_if_incomplete()")
    assert write_at < stage_at < raise_at


def test_train_py_unqualified_training_complete_guarded():
    """Verify that print('\\nTraining complete.') in train.py occurs in __main__
    after training() returns, ensuring a RunIncompleteError raised inside training()
    prevents it from executing."""
    src = TRAIN_PY.read_text(encoding="utf-8")
    training_def = src.index("def training(")
    main_block = src.index('if __name__ == "__main__":')
    complete_print = src.index('print("\\nTraining complete.")')
    assert complete_print > main_block > training_def


def test_both_entrypoints_import_canonical_run_completion_state():
    for file_path in (TRAIN_PY, TRAIN_STREAMING_PY):
        tree = ast.parse(file_path.read_text(encoding="utf-8"))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "utils.run_completion_state":
                for alias in node.names:
                    imported.add(alias.name)
        assert "RunCompletionState" in imported, f"{file_path.name} must import RunCompletionState"
        assert "FINAL_REPORT_CONTRACT" in imported, f"{file_path.name} must import FINAL_REPORT_CONTRACT"
