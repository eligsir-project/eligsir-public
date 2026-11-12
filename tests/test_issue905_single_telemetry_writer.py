"""Regression coverage for issue #905 -- one writer per anchor/bubble scalar.

Background
----------
The per-iteration anchor/bubble TB diagnostics block existed in TWO places:

* ``train_streaming.py`` -- the original, hoisted by PR #884 (``bec78819``) out
  of the inline ``else:`` arm so BOTH optimizer arms reach it (issue #882);
* ``eligsir/optimization/gaussian/objective.py`` -- a verbatim copy (minus
  the ``io/save_queue_depth`` line) carried in by the W3-02 extraction
  (``10e7f8d5``), reached only on the engine path via ``assemble_loss``.

After #884 both fired on an engine run, so seven tags were written twice per
sampled iteration under a single step index. Measured on ``i879engtel_a_engine``:
``streaming/bootstrap_anchor/drift_mean_m`` had n=80 samples over 40 distinct
steps and the two writes disagreed on 39 of them, because that tag is recomputed
from live Gaussian positions at two different points in the iteration.

``assemble_loss`` is additionally called more than once per step on the engine
path (engine.py's extra-view loop), so the copy there can never be a
"once per sampled iteration" writer. The ``train_streaming.py`` copy is the
correct single writer; the ``objective.py`` copy is deleted.

Both writers were TB-only diagnostics -- neither ever fed a loss, so removing
one changes no quality number.

What this file asserts
----------------------
(a) ``objective.py`` emits NONE of the affected scalars (the duplicate writer
    stays gone);
(b) ``train_streaming.py`` still emits all of them (the surviving writer was not
    deleted by mistake) -- complementing
    ``tests/test_issue882_engine_arm_telemetry_block.py``, which proves that
    block is reachable from both arms;
(c) repo-wide, each affected scalar name is written by exactly one production
    module, so a third copy cannot reappear unnoticed.

Purely static/source-level: the duplication's real signature (sample count ==
2x distinct-step count in an engine run's event file) needs a GPU run and is
tracked separately on #905.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
OBJECTIVE_PATH = REPO_ROOT / "eligsir" / "optimization" / "gaussian" / "objective.py"
TRAIN_STREAMING_PATH = REPO_ROOT / "train_streaming.py"

# The seven tags that were double-written on the engine path. `io/save_queue_depth`
# is deliberately NOT here: it only ever existed in the train_streaming.py copy
# (it needs `save_worker`), and it was the one genuinely engine-blind tag #884 fixed.
DUPLICATED_SCALARS = (
    "streaming/bootstrap_anchor/drift_mean_m",
    "streaming/bootstrap_anchor/grad_retain_frac",
    "streaming/insertion_anchor/n_active",
    "streaming/bubble/n_active",
    "streaming/bubble/n_frozen",
    "streaming/bubble/active_frac",
    "streaming/bubble/d_obs_m",
)

# Production modules only -- tests legitimately mention these names as literals.
_SEARCH_ROOTS = ("eligsir", "utils", "scene", "streaming", "geometry", "viz")
_SEARCH_FILES = ("train_streaming.py", "train.py")


def _emitted_scalar_names(path: Path) -> set[str]:
    """Names passed as the first positional arg to any ``*.add_scalar(...)`` call.

    AST-based rather than a substring scan so a scalar name merely mentioned in a
    comment or docstring (this module's own prose, for one) is not miscounted as
    an emission.
    """
    tree = ast.parse(path.read_text(), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr == "add_scalar"):
            continue
        if not node.args:
            continue
        first = node.args[0]
        if isinstance(first, ast.Constant) and isinstance(first.value, str):
            names.add(first.value)
    return names


@pytest.mark.parametrize("scalar_name", DUPLICATED_SCALARS)
def test_objective_does_not_emit_anchor_or_bubble_scalars(scalar_name):
    """The duplicate writer in the loss-assembly module must stay deleted."""
    emitted = _emitted_scalar_names(OBJECTIVE_PATH)
    assert scalar_name not in emitted, (
        f"{OBJECTIVE_PATH.relative_to(REPO_ROOT)} emits {scalar_name!r} again -- "
        "issue #905's duplicate engine-path writer has been reintroduced. This "
        "block belongs in train_streaming.py only (it runs once per sampled "
        "iteration on BOTH arms); assemble_loss runs more than once per step on "
        "the engine path, so a copy here double-writes the tag."
    )


@pytest.mark.parametrize("scalar_name", DUPLICATED_SCALARS)
def test_train_streaming_still_emits_anchor_and_bubble_scalars(scalar_name):
    """The surviving single writer must not have been deleted along with the copy."""
    emitted = _emitted_scalar_names(TRAIN_STREAMING_PATH)
    assert scalar_name in emitted, (
        f"train_streaming.py no longer emits {scalar_name!r} -- issue #905's fix "
        "removed the WRONG copy. The train_streaming.py block is the one that "
        "must survive (see tests/test_issue882_engine_arm_telemetry_block.py, "
        "which proves it is reachable from both optimizer arms)."
    )


def _production_python_files() -> list[Path]:
    files: list[Path] = []
    for name in _SEARCH_FILES:
        p = REPO_ROOT / name
        if p.is_file():
            files.append(p)
    for root in _SEARCH_ROOTS:
        base = REPO_ROOT / root
        if not base.is_dir():
            continue
        for p in base.rglob("*.py"):
            parts = set(p.relative_to(REPO_ROOT).parts)
            if "tests" in parts or "test" in parts:
                continue
            files.append(p)
    return files


@pytest.mark.parametrize("scalar_name", DUPLICATED_SCALARS)
def test_exactly_one_production_writer_per_scalar(scalar_name):
    """No third copy may appear anywhere else in production code."""
    writers = [
        str(p.relative_to(REPO_ROOT))
        for p in _production_python_files()
        if scalar_name in _emitted_scalar_names(p)
    ]
    assert writers == ["train_streaming.py"], (
        f"{scalar_name!r} is written by {writers} -- expected exactly "
        "['train_streaming.py']. Two writers under one step index produce an "
        "interleaved series (issue #905): for recomputed quantities such as "
        "drift_mean_m the two writes disagree, so the stored series does not "
        "mean what its name says."
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
