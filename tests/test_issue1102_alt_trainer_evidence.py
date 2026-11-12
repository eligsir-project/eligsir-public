"""Issue #1102: the `submap_stitch` / `rolling_seed` alternate trainers must
emit the same completion evidence every other training path emits, and must
fail closed when a required evidence phase fails.

Before this fix both modes finalized output staging and `return`ed. They never
wrote `run_completion_state.json`, `run_summary.json` or
`artifact_manifest.json`, and never reached `raise_if_incomplete()` -- so a run
whose report phase blew up exited 0, looked healthy in the log, and could not
be ingested by ResearchFlow at all (its canonical contract is
`run_summary.json` + `run_completion_state.json`).

The tests are deliberately split in two:

* **Behavioural** -- drive the real shared helpers
  (`eligsir.runtime.finalize.report_phase` / `emit_run_evidence`) in the
  exact sequence the call site uses, on a temp directory, and assert the
  artifacts and the fail-closed raise. Pure stdlib; no torch, no GPU, no
  dataset (`streaming_training()` itself is unreachable without all three).
* **Wiring** -- AST/source proof that the two mode modules and the
  `train_streaming.py` call site actually use those helpers in that order,
  which is the half a behavioural test on the helpers cannot cover. Same
  idiom as `tests/test_issue998_unified_completion_contract.py`.
"""

import ast
import json
import os
from argparse import Namespace
from pathlib import Path

import pytest

from eligsir.runtime.finalize import emit_run_evidence, report_phase
from eligsir.utils.run_completion_state import (
    Outcome,
    PhaseStatus,
    RunCompletionState,
    RunIncompleteError,
    load_run_completion_state,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SUBMAP_PY = REPO_ROOT / "eligsir/mapping/integration/submap_stitch_application.py"
ROLLING_PY = REPO_ROOT / "eligsir/mapping/integration/rolling_seed_application.py"
TRAIN_STREAMING_PY = REPO_ROOT / "train_streaming.py"

MODE_MODULES = pytest.mark.parametrize(
    "path", [SUBMAP_PY, ROLLING_PY], ids=["submap_stitch", "rolling_seed"]
)


def _args(model_path):
    return Namespace(model_path=str(model_path), scientific_mode=False)


def _report_dir(model_path, subdir="submap_final"):
    """A stand-in for what `write_post_training_report` produces/returns."""
    out = Path(model_path) / "comparison" / subdir
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps({"mean_test_psnr": 1.0}))
    return {"output_dir": str(out)}


# ---------------------------------------------------------------------------
# (a) a successful mode run produces all three artifacts
# ---------------------------------------------------------------------------


def test_successful_run_emits_all_three_evidence_artifacts(tmp_path):
    state = RunCompletionState(str(tmp_path))
    with report_phase(state, log_prefix="submap") as ph:
        state.mark_training_complete(final_iteration=5000)
        ph.record_output(_report_dir(tmp_path))

    emit_run_evidence(_args(tmp_path), state)

    for name in ("run_completion_state.json", "run_summary.json", "artifact_manifest.json"):
        assert (tmp_path / name).exists(), f"{name} was not written"

    # ...and the gate is satisfied, so the run may exit 0.
    state.raise_if_incomplete()


def test_successful_run_completion_state_schema(tmp_path):
    state = RunCompletionState(str(tmp_path))
    with report_phase(state) as ph:
        state.mark_training_complete(final_iteration=5000)
        ph.record_output(_report_dir(tmp_path))
    emit_run_evidence(_args(tmp_path), state)

    payload = json.loads((tmp_path / "run_completion_state.json").read_text())
    assert payload["outcome"] == Outcome.RUN_COMPLETE
    assert payload["training_complete"] is True
    assert payload["final_iteration"] == 5000
    phases = {p["name"]: p for p in payload["phases"]}
    assert phases["final_report"]["status"] == PhaseStatus.SUCCESS
    assert phases["final_report"]["required"] is True

    # The same verdict must be readable through the canonical loader that
    # ResearchFlow/ablator use, not just off the raw file.
    assert load_run_completion_state(str(tmp_path))["outcome"] == Outcome.RUN_COMPLETE


def test_successful_run_summary_carries_the_completion_state(tmp_path):
    state = RunCompletionState(str(tmp_path))
    with report_phase(state) as ph:
        state.mark_training_complete(final_iteration=5000)
        ph.record_output(_report_dir(tmp_path))
    emit_run_evidence(_args(tmp_path), state)

    summary = json.loads((tmp_path / "run_summary.json").read_text())
    assert summary["evidence_class"] == "UNMANAGED"
    # The completion verdict must round-trip into the orchestrator handoff.
    assert json.dumps(summary).find(Outcome.RUN_COMPLETE) != -1


def test_scientific_mode_marks_the_summary_managed(tmp_path):
    state = RunCompletionState(str(tmp_path))
    with report_phase(state) as ph:
        state.mark_training_complete(final_iteration=1)
        ph.record_output(_report_dir(tmp_path))
    args = _args(tmp_path)
    args.scientific_mode = True
    emit_run_evidence(args, state)
    assert json.loads((tmp_path / "run_summary.json").read_text())["evidence_class"] == "MANAGED"


# ---------------------------------------------------------------------------
# (b) a failed evidence phase now fails closed
# ---------------------------------------------------------------------------


def test_raising_report_phase_is_recorded_and_raises_after_emission(tmp_path):
    """The exact scenario #1102 describes: the report phase blows up, and the
    run must NOT be able to exit 0 -- while still keeping its outputs."""
    state = RunCompletionState(str(tmp_path))
    with report_phase(state, log_prefix="submap") as ph:  # noqa: F841 - phase body raises
        state.mark_training_complete(final_iteration=5000)
        raise TypeError("write_post_training_report() got an unexpected keyword argument")

    # The phase swallowed it (so staging still runs) but recorded the failure.
    emit_run_evidence(_args(tmp_path), state)
    payload = json.loads((tmp_path / "run_completion_state.json").read_text())
    assert payload["outcome"] == Outcome.REPORT_FAILED
    phase = {p["name"]: p for p in payload["phases"]}["final_report"]
    assert phase["status"] == PhaseStatus.FAILED
    assert phase["error_type"] == "TypeError"

    # ...and the gate refuses the run.
    with pytest.raises(RunIncompleteError):
        state.raise_if_incomplete()


def test_report_returning_without_an_output_dir_fails_closed(tmp_path):
    """A mode with no test cameras never calls `record_output`. That is
    unverifiable, not a pass -- fail closed."""
    state = RunCompletionState(str(tmp_path))
    with report_phase(state):
        state.mark_training_complete(final_iteration=5000)
        # no record_output(): nothing declared where report.json would be

    emit_run_evidence(_args(tmp_path), state)
    payload = json.loads((tmp_path / "run_completion_state.json").read_text())
    assert payload["outcome"] == Outcome.REPORT_ARTIFACTS_MISSING
    with pytest.raises(RunIncompleteError):
        state.raise_if_incomplete()


def test_report_dir_without_report_json_fails_closed(tmp_path):
    state = RunCompletionState(str(tmp_path))
    empty = tmp_path / "comparison" / "submap_final"
    empty.mkdir(parents=True)
    with report_phase(state) as ph:
        state.mark_training_complete(final_iteration=5000)
        ph.record_output({"output_dir": str(empty)})

    emit_run_evidence(_args(tmp_path), state)
    payload = json.loads((tmp_path / "run_completion_state.json").read_text())
    assert payload["outcome"] == Outcome.REPORT_ARTIFACTS_MISSING
    assert {p["name"]: p for p in payload["phases"]}["final_report"][
        "missing_artifacts"
    ] == ["report.json"]
    with pytest.raises(RunIncompleteError):
        state.raise_if_incomplete()


def test_evidence_artifacts_still_written_on_a_failed_phase(tmp_path):
    """Regression guard on the ordering: a failed phase must cost the run its
    exit code, never its evidence -- the state file is what records WHY it
    failed."""
    state = RunCompletionState(str(tmp_path))
    with report_phase(state):
        state.mark_training_complete(final_iteration=5000)
        raise RuntimeError("boom")
    emit_run_evidence(_args(tmp_path), state)
    for name in ("run_completion_state.json", "run_summary.json", "artifact_manifest.json"):
        assert (tmp_path / name).exists(), f"{name} lost on the failure path"


def test_no_run_state_preserves_pre_fix_behaviour(tmp_path, capsys):
    """`run_state=None` (report not requested) keeps the old print-and-continue
    behaviour rather than introducing a new failure mode."""
    with report_phase(None, log_prefix="rolling_seed"):
        raise RuntimeError("boom")
    assert "[rolling_seed] Evaluation failed: boom" in capsys.readouterr().out
    assert not os.path.exists(tmp_path / "run_completion_state.json")


# ---------------------------------------------------------------------------
# wiring: the two mode modules
# ---------------------------------------------------------------------------


def _mode_function(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in (
            "_run_submap_stitch",
            "_run_rolling_seed",
        ):
            return node
    raise AssertionError(f"mode entry point not found in {path}")


@MODE_MODULES
def test_mode_accepts_a_run_state(path):
    fn = _mode_function(path)
    names = [a.arg for a in fn.args.args] + [a.arg for a in fn.args.kwonlyargs]
    assert "run_state" in names, f"{path.name} must accept run_state"


@MODE_MODULES
def test_mode_reports_inside_a_recorded_phase(path):
    src = path.read_text(encoding="utf-8")
    assert "_report_phase(" in src, f"{path.name} must wrap its report in report_phase()"
    assert "record_output(" in src, f"{path.name} must record its report output dir"
    assert "mark_training_complete(" in src


@MODE_MODULES
def test_mode_records_the_report_writers_own_return_value(path):
    """The invariant the whole fail-closed chain hangs on.

    `record_output` must receive `write_post_training_report`'s return value
    -- that dict's `output_dir` is the ONLY way the phase can verify
    `report.json` exists (utils/run_completion_state.py's module docstring
    explains why it never reconstructs the path itself). A `record_output`
    call that happens to sit elsewhere in the module, or one wrapping some
    other call, would leave every run of this mode recording
    MISSING_ARTIFACTS and failing closed. A plain substring check cannot tell
    those apart -- this walks the AST.
    """
    fn = _mode_function(path)
    wrapped = []
    for node in ast.walk(fn):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "record_output"
        ):
            assert len(node.args) == 1, "record_output() takes exactly one argument"
            inner = node.args[0]
            assert isinstance(inner, ast.Call) and isinstance(inner.func, ast.Name), (
                f"{path.name}: record_output() must wrap a call, got {ast.dump(inner)[:80]}"
            )
            wrapped.append(inner.func.id)
    assert wrapped == ["write_post_training_report"], (
        f"{path.name}: expected exactly one "
        f"record_output(write_post_training_report(...)), got {wrapped}"
    )


@MODE_MODULES
def test_mode_has_no_bare_except_around_the_report(path):
    """The #646/#647 pattern this fix removes: the report call must not be
    swallowed by a bare try/except, or the real error never reaches the state
    file. Mirrors the identical guard on train.py in
    tests/test_issue998_unified_completion_contract.py."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Try):
            continue
        called = {
            n.func.id
            for n in ast.walk(node)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
        }
        if "write_post_training_report" in called:
            pytest.fail(f"{path.name}: write_post_training_report must not be swallowed")


# ---------------------------------------------------------------------------
# wiring: the train_streaming.py call site
# ---------------------------------------------------------------------------


def _alt_trainer_branch():
    src = TRAIN_STREAMING_PY.read_text(encoding="utf-8")
    _, sep, after = src.partition('if _training_mode in ("submap_stitch", "rolling_seed"):')
    assert sep, "alternate-trainer dispatch branch not found in train_streaming.py"
    return after.split("\n    if _training_mode != ", 1)[0]


def test_call_site_passes_a_run_state_into_both_modes():
    branch = _alt_trainer_branch()
    assert branch.count("run_state=_mode_state,") == 2


def test_call_site_emits_evidence_then_stages_then_gates():
    branch = _alt_trainer_branch()
    emit = branch.index("_emit_run_evidence(args, _mode_state)")
    stage = branch.index("_finalize_output_staging(args)")
    gate = branch.index("_mode_state.raise_if_incomplete()")
    # Evidence first (staging moves the directory to the canonical path),
    # the fail-closed gate last (so a failure costs the exit code, not the
    # outputs). Identical ordering to train.py and the normal streaming path.
    assert emit < stage < gate


def test_call_site_gate_is_the_last_thing_before_the_return():
    branch = _alt_trainer_branch()
    tail = [line.strip() for line in branch.rstrip().splitlines() if line.strip()]
    assert tail[-1] == "return"
    assert tail[-2] == "_mode_state.raise_if_incomplete()"


def test_training_complete_is_unreachable_after_the_gate():
    """`utils/champion_promotion.py` anchors on train.py's "Training complete."
    print. `raise_if_incomplete()` propagates uncaught through
    `train.training()` -> `__main__`, so that string cannot appear in the log
    of an alternate-trainer run whose evidence phase failed."""
    src = (REPO_ROOT / "train.py").read_text(encoding="utf-8")
    assert src.index('print("\\nTraining complete.")') > src.index('if __name__ == "__main__":')
