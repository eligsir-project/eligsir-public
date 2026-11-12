"""Issue #333 phase 3 ACTIVATION: production call sites must pass `db=`.

PR #780 taught `eligsir/mapping/tier_bind.py`'s `bind_resolution_tier(_ex)`
to call `db.touch_resolution_cache(frame_id)` on a resident cache hit, so
frequently-reused frames stay resident under LRU eviction instead of being
evicted purely by insertion order -- but ONLY when a caller passes `db`.

`train_streaming.py` defines its own thin wrapper, `bind_resolution_tier_ex`,
and ALL THREE production call sites in the training loop went through it
without ever forwarding a `db` -- so PR #780's mechanism was completely inert
in production: `_observation_db.touch_resolution_cache(...)` was never called
by a real training run, even though the underlying implementation and its own
unit tests (`test_issue333_phase3_access_aware_lru.py`, which calls
`eligsir.mapping.tier_bind.bind_resolution_tier` directly) were correct.

This test closes the wiring gap statically (AST), matching the established
pattern for training-loop call-seam guards in this repo (see
`test_issue646_report_kwargs_signature.py`): torch-free, CPU-CI-reachable, and
it inspects the *real* production call sites rather than re-testing
`tier_bind.py`'s already-covered behavior.

Verified to fail against the pre-fix code (a scratch checkout of this
worktree's `train_streaming.py` before this fix's edits): the wrapper had no
`db` parameter, and none of the three call sites passed `db=`.
"""

from __future__ import annotations

import ast
import inspect
import pathlib

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
TRAIN_STREAMING = REPO_ROOT / "train_streaming.py"
WRAPPER_NAME = "bind_resolution_tier_ex"


@pytest.fixture(scope="module")
def tree() -> ast.Module:
    return ast.parse(TRAIN_STREAMING.read_text())


def _find_wrapper_def(tree: ast.Module) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == WRAPPER_NAME:
            return node
    raise AssertionError(
        f"{WRAPPER_NAME}() not found in {TRAIN_STREAMING.name}. If it was "
        "renamed or inlined, retarget this test -- do not delete it; it is "
        "the only guard that PR #780's LRU-refresh mechanism is actually "
        "reachable in production."
    )


def _find_wrapper_calls(tree: ast.Module) -> list:
    """All `bind_resolution_tier_ex(...)` Call nodes anywhere in the module,
    EXCLUDING the internal forwarding call inside the `bind_resolution_tier`
    bool wrapper's own body (that call intentionally omits `db` -- it is not
    a production call site, it is the bool-wrapper's own back-compat
    passthrough, and its non-use in production is checked separately by
    `test_bool_wrapper_not_used_in_production`).

    Deliberately module-wide (minus that one exclusion) rather than scoped to
    `streaming_training()`: the three known production call sites live at
    different nesting depths (inline-loop primary bind, inline-loop
    extra-view batch, optimization-engine extra-view batch) and a scoped walk
    would need to name each enclosing function, which is exactly the kind of
    coupling that breaks silently on a refactor.
    """
    bool_wrapper_calls = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "bind_resolution_tier":
            for inner in ast.walk(node):
                if isinstance(inner, ast.Call):
                    bool_wrapper_calls.add(id(inner))

    calls = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == WRAPPER_NAME
            and id(node) not in bool_wrapper_calls
        ):
            calls.append(node)
    return calls


def test_wrapper_accepts_a_db_parameter(tree):
    """The thin train_streaming.py wrapper must forward a `db` to the impl.

    Without this parameter, no caller in this module can ever activate
    PR #780's LRU-refresh regardless of what the call sites pass.
    """
    wrapper = _find_wrapper_def(tree)
    param_names = {arg.arg for arg in wrapper.args.args + wrapper.args.kwonlyargs}
    assert "db" in param_names, (
        f"{WRAPPER_NAME}() has no 'db' parameter ({sorted(param_names)}). "
        "PR #780's touch_resolution_cache mechanism cannot be reached from "
        "any caller of this wrapper -- it is dead code in production."
    )


def test_wrapper_forwards_db_to_the_real_implementation(tree):
    """The wrapper must actually pass its `db` through, not just accept it."""
    wrapper = _find_wrapper_def(tree)
    src = ast.get_source_segment(TRAIN_STREAMING.read_text(), wrapper) or ""
    # A `db=db` (or equivalent) keyword forward into the impl call.
    found_forward = any(
        isinstance(n, ast.Call)
        and any(
            isinstance(kw, ast.keyword)
            and kw.arg == "db"
            and isinstance(kw.value, ast.Name)
            and kw.value.id == "db"
            for kw in n.keywords
        )
        for n in ast.walk(wrapper)
    )
    assert found_forward, (
        f"{WRAPPER_NAME}() accepts a 'db' parameter but its body never "
        f"forwards it as db=db to the implementation call. Source:\n{src}"
    )


def test_at_least_three_production_call_sites_exist(tree):
    """Sanity floor: if this drops to 0-2, the AST walk itself is broken."""
    calls = _find_wrapper_calls(tree)
    # Subtract 1 for none -- calls found should include all real call sites;
    # the wrapper's own `def` is a FunctionDef, not a Call, so it is not
    # among these.
    assert len(calls) >= 3, (
        f"Expected at least 3 calls to {WRAPPER_NAME}() in "
        f"{TRAIN_STREAMING.name} (the known production call sites as of "
        f"issue #333 phase 3), found {len(calls)}. If call sites were "
        "removed/merged intentionally, lower this floor deliberately -- "
        "don't let it silently drop to 0."
    )


def test_every_production_call_site_passes_db(tree):
    """The actual regression guard: every call site must pass `db=`.

    This is what PR #780 shipped without: an implementation that supports
    LRU-refresh, with zero production callers opting in. A call site passing
    `db=None` explicitly would also fail this (it must resolve to something
    other than the literal `None`), since that is equivalent to not passing
    it at all for touch_resolution_cache's purposes.
    """
    calls = _find_wrapper_calls(tree)
    missing = []
    for call in calls:
        db_kw = next((kw for kw in call.keywords if kw.arg == "db"), None)
        lineno = call.lineno
        if db_kw is None:
            missing.append(f"line {lineno}: no db= keyword at all")
        elif isinstance(db_kw.value, ast.Constant) and db_kw.value.value is None:
            missing.append(f"line {lineno}: db=None (never activates the LRU refresh)")
    assert not missing, (
        f"{WRAPPER_NAME}() call site(s) in {TRAIN_STREAMING.name} do not pass "
        "a real db= argument, so PR #780's touch_resolution_cache LRU-refresh "
        f"is inert for them:\n" + "\n".join(missing)
    )


def test_bool_wrapper_not_used_in_production(tree):
    """`bind_resolution_tier` (the bool wrapper) has no db param of its own.

    It is fine for it to stay a 3-arg, db-less wrapper (existing tests may
    call it), but ONLY as long as nothing in the production training loop
    actually calls it -- otherwise it would be a second, silently-inert path
    exactly like the one this test file exists to close.
    """
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "bind_resolution_tier"
        ):
            pytest.fail(
                f"bind_resolution_tier() (the bool wrapper, no db forwarding) "
                f"is called at line {node.lineno} in {TRAIN_STREAMING.name}. "
                f"If this is a real production call site, either wire a db "
                "through it too or route it through bind_resolution_tier_ex."
            )


def test_runtime_signature_matches_ast():
    """The AST-extracted wrapper signature must not drift from reality.

    Skipped in CPU CI (importing train_streaming needs torch) -- mirrors the
    pattern in test_issue646_report_kwargs_signature.py.
    """
    pytest.importorskip("torch")
    import train_streaming

    sig = inspect.signature(train_streaming.bind_resolution_tier_ex)
    assert "db" in sig.parameters
    assert sig.parameters["db"].default is None
