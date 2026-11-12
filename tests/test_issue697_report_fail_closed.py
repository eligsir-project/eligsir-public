"""Issue #697: the report phase must fail closed.

The #646/#647 outage: a streaming run trained to completion, raised inside
``write_post_training_report``, and STILL printed ``[report-phase] ✓ ... done``
and ``Training complete.`` -- because the call was wrapped in a bare
``except Exception: print(...)`` with the success checkmark printed
unconditionally afterwards. No ``report.json``, no ``comparison/``, and a log
that reads exactly like a healthy finished run.

These are the five regression tests the issue asks for, plus a sixth that pins
the CALL SITE. Five tests of a module nobody proves is actually wired in would
be #646 one level up, so ``test_train_streaming_call_site_*`` AST-parse
``train_streaming.py`` the way ``tests/test_issue646_report_kwargs_signature.py``
does -- and for the same reason: ``streaming_training()`` is a ~6000-line
function unreachable without standing up a whole training run.

No torch, direct or transitive. ``tests/test_issue646_report_kwargs_signature.py``
documents that importing torch made the #646 guard *collected away* in CPU CI
(the only required check) -- a guard for a two-day P0 outage that could not run
where it mattered. ``utils.run_completion_state`` is pure stdlib for exactly
this reason.
"""

import ast
import json
import pathlib

import pytest

from eligsir.utils.run_completion_state import (
    FINAL_REPORT_CONTRACT,
    Outcome,
    PhaseContract,
    PhaseStatus,
    RunCompletionState,
    RunIncompleteError,
    STATE_FILENAME,
)

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
TRAIN_STREAMING = REPO_ROOT / "train_streaming.py"


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _state(tmp_path):
    lines = []
    st = RunCompletionState(str(tmp_path), printer=lines.append)
    return st, lines


def _write_report_json(model_path, subdir="comparison/iter_100"):
    out_dir = pathlib.Path(model_path) / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text("{}")
    return str(out_dir)


def _persisted(tmp_path):
    return json.loads((pathlib.Path(tmp_path) / STATE_FILENAME).read_text())


def _has_success_check(lines):
    return any("✓" in line for line in lines)


# --------------------------------------------------------------------------
# 1. report writer raises -> phase FAILED, no checkmark, run result non-success
# --------------------------------------------------------------------------

def test_writer_raises_marks_phase_failed_no_checkmark(tmp_path):
    st, lines = _state(tmp_path)
    st.mark_training_complete(final_iteration=100)

    with st.phase(FINAL_REPORT_CONTRACT) as ph:
        # exactly the #646 shape: a kwarg the writer does not accept
        raise TypeError(
            "write_post_training_report() got an unexpected keyword argument "
            "'spf_resolution'"
        )

    phase = st.get("final_report")
    assert phase.status == PhaseStatus.FAILED
    assert phase.error_type == "TypeError"
    assert "spf_resolution" in phase.error

    # no success marker anywhere in this phase's output
    assert not _has_success_check(lines)
    assert any("✗" in line and "FAILED" in line for line in lines)

    # non-success run result, and it is machine-readable on disk
    assert st.training_complete is True
    assert st.report_complete is False
    assert st.run_complete is False
    assert st.outcome == Outcome.REPORT_FAILED
    disk = _persisted(tmp_path)
    assert disk["run_complete"] is False
    assert disk["outcome"] == Outcome.REPORT_FAILED
    assert disk["failure_type"] == "TypeError"
    assert "spf_resolution" in disk["failure_reason"]

    # fail closed: the run refuses to exit successfully
    with pytest.raises(RunIncompleteError):
        st.raise_if_incomplete()


# --------------------------------------------------------------------------
# 2. report returns but a required artifact is absent -> fail closed
# --------------------------------------------------------------------------

def test_returns_without_required_artifact_fails_closed(tmp_path):
    st, lines = _state(tmp_path)
    st.mark_training_complete(final_iteration=100)

    out_dir = tmp_path / "comparison" / "iter_100"
    out_dir.mkdir(parents=True)
    # directory exists, contact sheets even exist -- but no report.json
    (out_dir / "test_contact_sheet.png").write_text("x")

    with st.phase(FINAL_REPORT_CONTRACT) as ph:
        ph.record_output({"output_dir": str(out_dir)})

    phase = st.get("final_report")
    assert phase.status == PhaseStatus.MISSING_ARTIFACTS
    assert phase.missing_artifacts == ["report.json"]
    assert not _has_success_check(lines)

    assert st.required_artifacts_present is False
    assert st.run_complete is False
    assert st.outcome == Outcome.REPORT_ARTIFACTS_MISSING
    with pytest.raises(RunIncompleteError):
        st.raise_if_incomplete()


def test_returns_without_reporting_output_dir_fails_closed(tmp_path):
    """Unverifiable is not the same as fine. A phase that never says where it
    wrote cannot have its contract checked, so it fails closed."""
    st, _ = _state(tmp_path)
    st.mark_training_complete()

    with st.phase(FINAL_REPORT_CONTRACT):
        pass  # returns, records nothing

    assert st.get("final_report").status == PhaseStatus.MISSING_ARTIFACTS
    assert st.run_complete is False
    with pytest.raises(RunIncompleteError):
        st.raise_if_incomplete()


def test_artifact_check_uses_reported_dir_not_a_recomputed_one(tmp_path):
    """The streaming call site passes ``iteration=opt.iterations``, which for an
    --online_mode run is the 2**31 sentinel and DELIBERATELY names the output
    directory. A validator that recomputed the path from the loop's real final
    iteration would fail-close every healthy online_mode run."""
    st, _ = _state(tmp_path)
    st.mark_training_complete(final_iteration=8123)  # real loop exit iteration

    sentinel_dir = _write_report_json(tmp_path, "comparison/iter_2147483647")

    with st.phase(FINAL_REPORT_CONTRACT) as ph:
        ph.record_output({"output_dir": sentinel_dir})

    assert st.get("final_report").status == PhaseStatus.SUCCESS
    assert st.run_complete is True
    st.raise_if_incomplete()  # must not raise


# --------------------------------------------------------------------------
# 3. report succeeds and required artifacts exist -> success
# --------------------------------------------------------------------------

def test_success_path(tmp_path):
    st, lines = _state(tmp_path)
    st.mark_training_complete(final_iteration=100)

    out_dir = _write_report_json(tmp_path)
    with st.phase(FINAL_REPORT_CONTRACT) as ph:
        ph.record_output({"output_dir": out_dir, "n_test": 10})

    phase = st.get("final_report")
    assert phase.status == PhaseStatus.SUCCESS
    assert phase.error is None
    assert _has_success_check(lines)

    assert st.report_complete is True
    assert st.required_artifacts_present is True
    assert st.run_complete is True
    assert st.scientific_result_complete is True
    assert st.outcome == Outcome.RUN_COMPLETE
    st.raise_if_incomplete()  # must not raise

    disk = _persisted(tmp_path)
    assert disk["outcome"] == Outcome.RUN_COMPLETE
    assert disk["run_complete"] is True
    assert disk["failure_reason"] is None


def test_success_path_accepts_a_bare_path_string(tmp_path):
    st, _ = _state(tmp_path)
    st.mark_training_complete()
    out_dir = _write_report_json(tmp_path)
    with st.phase(FINAL_REPORT_CONTRACT) as ph:
        ph.record_output(out_dir)
    assert st.run_complete is True


# --------------------------------------------------------------------------
# 4. report explicitly disabled -> NOT_REQUESTED, not failed
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "reason", ["--no-streaming_report_final", "--streaming_benchmark_mode"]
)
def test_disabled_report_is_not_requested_not_failed(tmp_path, reason):
    st, lines = _state(tmp_path)
    st.mark_training_complete(final_iteration=100)
    st.mark_not_requested(FINAL_REPORT_CONTRACT, reason=reason)

    phase = st.get("final_report")
    assert phase.status == PhaseStatus.NOT_REQUESTED
    assert phase.note == reason
    assert phase.error is None

    # "not requested" is NOT graded as failure ...
    assert st.blocking_failures == []
    assert st.report_complete is True
    assert st.run_complete is True
    st.raise_if_incomplete()  # must not raise

    # ... but it is also not a scientific result, and says so.
    assert st.report_requested is False
    assert st.scientific_result_complete is False
    assert st.outcome == Outcome.RUN_COMPLETE_NO_REPORT_REQUESTED

    st.print_summary()
    assert any("NOT a scientific result" in line for line in lines)


# --------------------------------------------------------------------------
# 5. optional component fails -> classification follows the declared contract
# --------------------------------------------------------------------------

def test_optional_phase_failure_does_not_downgrade_the_run(tmp_path):
    st, lines = _state(tmp_path)
    st.mark_training_complete(final_iteration=100)

    out_dir = _write_report_json(tmp_path)
    with st.phase(FINAL_REPORT_CONTRACT) as ph:
        ph.record_output({"output_dir": out_dir})

    optional = PhaseContract(
        name="trajectory_eval", label="trajectory evaluation (ATE/RPE)", required=False
    )
    with st.phase(optional):
        raise RuntimeError("no GT trajectory could be associated")

    # recorded honestly ...
    assert st.get("trajectory_eval").status == PhaseStatus.FAILED
    assert any("✗" in line and "trajectory" in line for line in lines)
    disk = _persisted(tmp_path)
    traj = [p for p in disk["phases"] if p["name"] == "trajectory_eval"][0]
    assert traj["status"] == PhaseStatus.FAILED
    assert traj["required"] is False

    # ... but it does not silently downgrade the declared result
    assert st.run_complete is True
    assert st.outcome == Outcome.RUN_COMPLETE
    st.raise_if_incomplete()


def test_required_phase_failure_is_not_upgraded_by_optional_successes(tmp_path):
    """The mirror image: optional phases succeeding must not paper over a
    required-phase failure."""
    st, _ = _state(tmp_path)
    st.mark_training_complete()

    with st.phase(FINAL_REPORT_CONTRACT):
        raise ValueError("boom")
    with st.phase(PhaseContract(name="curve", label="curve", required=False)):
        pass

    assert st.get("curve").status == PhaseStatus.SUCCESS
    assert st.run_complete is False
    assert st.outcome == Outcome.REPORT_FAILED


# --------------------------------------------------------------------------
# extra: state semantics that consumers depend on
# --------------------------------------------------------------------------

def test_state_is_pending_while_the_phase_is_still_running(tmp_path):
    """A process killed inside the report phase must leave a NON-SUCCESS record
    on disk, not nothing (absent) and not a stale success."""
    st, _ = _state(tmp_path)
    st.mark_training_complete()
    with st.phase(FINAL_REPORT_CONTRACT):
        mid = _persisted(tmp_path)
        assert mid["phases"][0]["status"] == PhaseStatus.PENDING
        assert mid["run_complete"] is False
        assert mid["outcome"] == Outcome.REPORT_INCOMPLETE
        raise RuntimeError("killed")


def test_training_incomplete_when_never_marked(tmp_path):
    st, _ = _state(tmp_path)
    assert st.outcome == Outcome.TRAINING_INCOMPLETE
    assert st.run_complete is False
    with pytest.raises(RunIncompleteError):
        st.raise_if_incomplete()


def test_training_crash_is_recorded_and_persisted(tmp_path):
    """issue #1278/#1284: a training crash used to leave no state file at all,
    and train.log's tail could show a masking teardown exception rather than
    the real cause. mark_training_crashed() must persist the real one."""
    st, _ = _state(tmp_path)

    st.mark_training_crashed(RuntimeError("HIP out of memory"))

    assert st.outcome == Outcome.TRAINING_INCOMPLETE  # still never completed
    assert st.failure_type == "RuntimeError"
    assert "HIP out of memory" in st.failure_reason
    disk = _persisted(tmp_path)
    assert disk["outcome"] == Outcome.TRAINING_INCOMPLETE
    assert disk["failure_type"] == "RuntimeError"
    assert "HIP out of memory" in disk["failure_reason"]
    with pytest.raises(RunIncompleteError):
        st.raise_if_incomplete()


def test_state_write_never_raises_on_a_bad_path(tmp_path):
    st = RunCompletionState(str(tmp_path / "file.txt"), printer=lambda _m: None)
    (tmp_path / "file.txt").write_text("not a directory")
    st.mark_training_complete()
    assert st.write() is None
    assert st.write_error is not None  # recorded, not raised


def test_module_imports_no_torch():
    """Guard the #649 CI-reachability property this whole file depends on."""
    src = (REPO_ROOT / "utils" / "run_completion_state.py").read_text()
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
# 6. THE SEAM: the call site in train_streaming.py actually uses all this
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def train_streaming_tree():
    return ast.parse(TRAIN_STREAMING.read_text())


def _phase_with_blocks(tree, contract_name):
    """Every ``with _run_state.phase(<contract_name>)`` block in the module."""
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


def test_call_site_wraps_the_final_report_in_the_fail_closed_phase(train_streaming_tree):
    blocks = _phase_with_blocks(train_streaming_tree, "FINAL_REPORT_CONTRACT")
    assert len(blocks) == 1, (
        "exactly one final-report phase expected in train_streaming.py, "
        f"found {len(blocks)}"
    )
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
    ], "the phase must be told where the report wrote, or it cannot verify it"


def test_report_imports_are_inside_the_phase_body(train_streaming_tree):
    """An ImportError in the report phase must be a recorded phase failure, not
    an exception escaping streaming_training() -- which would skip the whole
    shutdown tail (save_worker.shutdown, _run_state.write,
    _finalize_output_staging), losing the run's outputs AND the state file
    whose entire job is recording this failure."""
    for contract in ("FINAL_REPORT_CONTRACT", "BOOTSTRAP_VIEWS_CONTRACT"):
        blocks = _phase_with_blocks(train_streaming_tree, contract)
        assert blocks, f"no {contract} phase found"
        modules = {
            n.module
            for block in blocks
            for n in ast.walk(block)
            if isinstance(n, ast.ImportFrom)
        }
        assert "utils.comparison_report" in modules, (
            f"{contract}'s phase body must own its report-writer import"
        )


def test_call_site_has_no_bare_except_around_the_final_report(train_streaming_tree):
    """The #646 shape itself: report call inside ``except Exception: print(...)``."""
    for node in ast.walk(train_streaming_tree):
        if not isinstance(node, ast.Try):
            continue
        calls = [
            n.func.id
            for n in ast.walk(node)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
        ]
        if "write_post_training_report" not in calls:
            continue
        # A try/except is allowed around a MID-TRAINING report (owned by #698);
        # it must not swallow the FINAL one. The final call is the only one
        # passing final=True through the shared kwargs builder.
        finals = [
            n
            for n in ast.walk(node)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id == "_streaming_report_kwargs"
            and any(kw.arg == "final" for kw in n.keywords)
        ]
        assert not finals, (
            "the FINAL post-run report is inside a try/except again -- that is "
            "the #646 failure shape (issue #697)"
        )


def test_call_site_emits_no_unconditional_success_checkmark():
    """The success string must exist in exactly one place: the phase wrapper's
    success branch. train_streaming.py must not print it itself."""
    src = TRAIN_STREAMING.read_text()
    assert "[report-phase] ✓" not in src
    assert "_phase_end(" not in src


def test_call_site_fails_closed_before_returning(train_streaming_tree):
    src = TRAIN_STREAMING.read_text()
    assert "_run_state.raise_if_incomplete()" in src
    # ... and the state must be persisted BEFORE output staging moves the
    # directory to its canonical path, or the consumer never sees it.
    write_at = src.index("_run_state.write()\n    _run_state.print_summary()")
    stage_at = src.rindex("_finalize_output_staging(args)")
    raise_at = src.index("_run_state.raise_if_incomplete()")
    assert write_at < stage_at < raise_at
