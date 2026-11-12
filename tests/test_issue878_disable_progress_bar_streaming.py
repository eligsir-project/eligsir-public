"""``--disable_progress_bar`` was declared in ``eligsir/cli/parser.py``
but read by exactly one call site in the whole repo (``train.py:792``). Every
``tqdm`` bar reachable from a *streaming* run ignored it:

* ``train_streaming.py``'s ``"Streaming mapping runtime"`` bar passed only
  ``disable=getattr(args, "quiet", False)``;
* both ``"Global refine"`` bars (``rolling_seed_application.py`` /
  ``submap_stitch_application.py``) passed no ``disable=`` argument at all.

So the one flag that exists to silence the bar did nothing on the streaming
path, and the only flag that worked was ``--quiet`` -- ruled out by standing
project convention (never suppress training stdout; the user monitors
GPU%/it-s live). Beyond log hygiene, the bar's rendering cost sits inside the
interval ``_iter_dt`` measures, and ``streaming/assimilation_optimization_rate``
normalises that against an *absolute* 50 it/s threshold, so the overhead does
not cancel out of the telemetry. See issue #878.

This file is a purely static (AST) test, deliberately torch-free per this
repo's CPU-only CI image (see test_issue646_report_kwargs_signature.py's
docstring for the general rationale: importing train_streaming.py pulls in
torch and gsplat, which are unavailable in CI).
"""

import ast
import pathlib

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]

# Every tqdm construction reachable from a streaming training run. Deliberately
# excludes render.py / metrics.py, which are separate CLI entrypoints that do
# not take the training parser's args.
STREAMING_REACHABLE_TQDM_SITES = (
    "train_streaming.py",
    "eligsir/mapping/integration/rolling_seed_application.py",
    "eligsir/mapping/integration/submap_stitch_application.py",
)

# The legacy (non-streaming) path, which already honoured both flags before
# #878 and is pinned here so a future refactor cannot regress it either.
LEGACY_TQDM_SITE = "train.py"


def _tqdm_calls(path: pathlib.Path):
    tree = ast.parse(path.read_text())
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "tqdm"
    ]


def _disable_kwarg(call: ast.Call):
    for kw in call.keywords:
        if kw.arg == "disable":
            return kw.value
    return None


def _attr_names_read(node: ast.AST) -> set:
    """Every string literal used as a getattr() attribute name anywhere in the
    given expression, plus any bare Name identifiers it references (so a
    pre-computed ``_refine_bar_disabled`` variable can be resolved by the
    caller)."""
    names = set()
    for sub in ast.walk(node):
        if (
            isinstance(sub, ast.Call)
            and isinstance(sub.func, ast.Name)
            and sub.func.id == "getattr"
            and len(sub.args) >= 2
            and isinstance(sub.args[1], ast.Constant)
            and isinstance(sub.args[1].value, str)
        ):
            names.add(sub.args[1].value)
        elif isinstance(sub, ast.Name):
            names.add(sub.id)
    return names


def _resolve_disable_flags(path: pathlib.Path, call: ast.Call) -> set:
    """Attribute names the call's ``disable=`` expression ultimately depends
    on, following at most one level of local-variable indirection (the
    ``_refine_bar_disabled = ...`` shape used by the Global refine sites)."""
    expr = _disable_kwarg(call)
    assert expr is not None, "caller must check for a missing disable= first"

    names = _attr_names_read(expr)
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id in names:
                names |= _attr_names_read(node.value)
    return names


@pytest.mark.parametrize("relpath", STREAMING_REACHABLE_TQDM_SITES)
def test_streaming_tqdm_sites_exist(relpath):
    """Sanity check the regression test is aimed at real call sites -- if this
    fails the code shape changed and the check below must be re-targeted, not
    silently skipped."""
    calls = _tqdm_calls(REPO_ROOT / relpath)
    assert calls, f"expected at least one tqdm(...) construction in {relpath}"


@pytest.mark.parametrize(
    "relpath", STREAMING_REACHABLE_TQDM_SITES + (LEGACY_TQDM_SITE,)
)
def test_tqdm_bars_honour_disable_progress_bar(relpath):
    """Regression pin for #878: every tqdm bar on a training path must consult
    BOTH --disable_progress_bar and --quiet. Passing only --quiet (the
    pre-#878 streaming behaviour) or omitting disable= entirely leaves the
    documented flag inert."""
    path = REPO_ROOT / relpath
    for call in _tqdm_calls(path):
        expr = _disable_kwarg(call)
        assert expr is not None, (
            f"{relpath}: tqdm(...) at line {call.lineno} passes no disable= "
            "argument, so --disable_progress_bar/--quiet cannot turn it off"
        )
        flags = _resolve_disable_flags(path, call)
        missing = {"disable_progress_bar", "quiet"} - flags
        assert not missing, (
            f"{relpath}: tqdm(...) at line {call.lineno} ignores {sorted(missing)}; "
            "its disable= expression must consult both flags"
        )
