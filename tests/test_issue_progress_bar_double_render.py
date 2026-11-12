"""``train_streaming.py``'s per-10-iteration logging block calls
``progress_bar.set_postfix(postfix)`` immediately followed by
``progress_bar.update(10)``. ``tqdm.set_postfix()`` refreshes/renders by
default, and ``update()`` renders again right after -- two tqdm refreshes per
checkpoint instead of one.

This is invisible in an interactive terminal (tqdm's carriage-return
overwrites the previous render in place), but once ``train.log`` is piped to
a file, each refresh becomes its own permanent text block -- roughly doubling
the size of every streaming run's log for no informational gain (found
2026-08-15 inspecting ``a recorded streaming run log``, where
every 10-iteration checkpoint appeared as two near-identical progress lines).

Fix: pass ``refresh=False`` to ``set_postfix()`` so only the ``update()``
call renders.

This file is a purely static (AST) test, deliberately torch-free per this
repo's CPU-only CI image (see test_issue646_report_kwargs_signature.py's
docstring for the general rationale: importing train_streaming.py pulls in
torch and gsplat, which are unavailable in CI).
"""

import ast
import pathlib

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
TRAIN_STREAMING = REPO_ROOT / "train_streaming.py"


def _find_set_postfix_calls(tree: ast.AST):
    calls = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "set_postfix"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "progress_bar"
        ):
            calls.append(node)
    return calls


def test_progress_bar_set_postfix_calls_exist():
    """Sanity check the test is looking at the right call site -- if this
    ever fails, the code shape changed enough that the regression check
    below needs to be re-targeted, not silently skipped."""
    tree = ast.parse(TRAIN_STREAMING.read_text())
    calls = _find_set_postfix_calls(tree)
    assert len(calls) >= 1, (
        "expected at least one progress_bar.set_postfix(...) call in "
        "train_streaming.py; none found -- has this been refactored away?"
    )


def test_progress_bar_set_postfix_does_not_force_a_redundant_render():
    """Regression pin: every progress_bar.set_postfix(...) call in
    train_streaming.py must pass refresh=False, since it is immediately
    followed by progress_bar.update(10), which renders on its own. Without
    refresh=False, each 10-iteration checkpoint produces two tqdm renders
    instead of one, roughly doubling every streaming run's train.log."""
    tree = ast.parse(TRAIN_STREAMING.read_text())
    calls = _find_set_postfix_calls(tree)
    assert calls, "no progress_bar.set_postfix(...) calls found"

    for call in calls:
        refresh_kwargs = [kw for kw in call.keywords if kw.arg == "refresh"]
        assert refresh_kwargs, (
            f"progress_bar.set_postfix(...) at train_streaming.py:{call.lineno} "
            "has no refresh= keyword -- tqdm defaults to refresh=True, which "
            "double-renders every checkpoint against the update() call right "
            "after it. Pass refresh=False explicitly."
        )
        value = refresh_kwargs[0].value
        assert isinstance(value, ast.Constant) and value.value is False, (
            f"progress_bar.set_postfix(...) at train_streaming.py:{call.lineno} "
            "must pass refresh=False (literal), not a truthy/non-literal value."
        )
