"""Issue #729: every mid-training report died with a NameError on a default
streaming run because ``_streaming_report_kwargs()`` (``train_streaming.py``,
nested inside ``streaming_training()``) reads ``_git_provenance_from_args`` at
one of its return-dict entries without ever importing that name itself -- it
resolves as a **free variable of the enclosing ``streaming_training()``
scope**.

Before the fix, the only statements in ``streaming_training()`` that bind that
name are:
  * an import gated behind ``if getattr(args, "streaming_report_pre_training",
    False)`` (default ``False``, so normally never executes), and
  * two imports inside the FINAL report block, which only run *after* every
    mid-training report tick has already happened.

So on a default run, the first (and every subsequent) mid-training call to
``_streaming_report_kwargs()`` -- from inside the nested ``_run_mid_report()``
closure -- raised ``NameError: free variable '_git_provenance_from_args'
referenced before assignment``. The exception was swallowed by
``_run_mid_report``'s own ``try/except`` (printed, then dropped), so every
mid-training report silently vanished while the final report kept succeeding
-- the exact asymmetry (final reports present, mid-run reports absent) that
the #696 census used to find this issue's 26 affected run directories.

This file is a purely static (AST) test, deliberately torch-free per this
repo's CPU-only CI image (see test_issue646_report_kwargs_signature.py's
docstring for the general rationale: importing train_streaming.py pulls in
torch and gsplat, which are unavailable in CI).

Two tests:
  * ``test_free_variables_are_all_available_before_any_call`` -- the
    class-level guard the issue asks for: every free variable
    ``_streaming_report_kwargs()`` reads must be bound *unconditionally*
    (never gated behind an ``if``) somewhere in ``streaming_training()``'s own
    body, at a line number strictly before the earliest call to
    ``_streaming_report_kwargs()`` anywhere in the function. This is the
    general shape of the #729 bug class, not just this one name -- the next
    reader who adds a new free variable and only binds it inside a
    conditional/mid-run-only block will trip this test too.
  * ``test_git_provenance_pinned`` -- the specific regression pin: asserts
    ``_git_provenance_from_args`` is one of the names that clears the check
    above (fails before the fix, passes after).
"""

import ast
import builtins
import pathlib

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
TRAIN_STREAMING = REPO_ROOT / "train_streaming.py"
OUTER_FN_NAME = "streaming_training"
BUILDER_NAME = "_streaming_report_kwargs"

_BUILTIN_NAMES = set(dir(builtins))


def _find_function(tree: ast.AST, name: str, *, within=None) -> ast.FunctionDef:
    search_root = within if within is not None else tree
    for node in ast.walk(search_root):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(
        f"{name}() not found in {TRAIN_STREAMING.name}"
        + (f" within {within.name}()" if within is not None else "")
        + ". If it was renamed/moved, retarget this test -- do not delete it; "
        "it is the only guard against the #729 free-variable-NameError class."
    )


class _LocalNameCollector(ast.NodeVisitor):
    """Names bound *within* a function's own body (its locals), including
    names bound by nested ``def``/``class`` statements (the nested function
    itself becomes a local name; its own body is a separate scope and is not
    descended into for this purpose)."""

    def __init__(self):
        self.assigned = set()

    def visit_FunctionDef(self, node):
        self.assigned.add(node.name)
        for a in node.args.args + node.args.posonlyargs + node.args.kwonlyargs:
            self.assigned.add(a.arg)
        # Deliberately do not descend into a nested function's body here --
        # its locals are its own scope, not this function's.

    def visit_ClassDef(self, node):
        self.assigned.add(node.name)

    def visit_Assign(self, node):
        for t in node.targets:
            for n in ast.walk(t):
                if isinstance(n, ast.Name):
                    self.assigned.add(n.id)
        self.generic_visit(node)

    def visit_AugAssign(self, node):
        for n in ast.walk(node.target):
            if isinstance(n, ast.Name):
                self.assigned.add(n.id)
        self.generic_visit(node)

    def visit_NamedExpr(self, node):  # walrus
        if isinstance(node.target, ast.Name):
            self.assigned.add(node.target.id)
        self.generic_visit(node)

    def visit_Import(self, node):
        for a in node.names:
            self.assigned.add((a.asname or a.name).split(".")[0])

    def visit_ImportFrom(self, node):
        for a in node.names:
            self.assigned.add(a.asname or a.name)

    def visit_For(self, node):
        for n in ast.walk(node.target):
            if isinstance(n, ast.Name):
                self.assigned.add(n.id)
        self.generic_visit(node)

    def visit_With(self, node):
        for item in node.items:
            if item.optional_vars is not None:
                for n in ast.walk(item.optional_vars):
                    if isinstance(n, ast.Name):
                        self.assigned.add(n.id)
        self.generic_visit(node)

    def visit_ExceptHandler(self, node):
        if node.name:
            self.assigned.add(node.name)
        self.generic_visit(node)


def _own_locals(fn: ast.FunctionDef) -> set:
    """Names bound anywhere in ``fn``'s own body (its own scope's locals)."""
    lc = _LocalNameCollector()
    for a in fn.args.args + fn.args.posonlyargs + fn.args.kwonlyargs:
        lc.assigned.add(a.arg)
    for stmt in fn.body:
        lc.visit(stmt)
    return lc.assigned


def _loaded_names(fn: ast.FunctionDef) -> set:
    loaded = set()
    for stmt in fn.body:
        for node in ast.walk(stmt):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                loaded.add(node.id)
    return loaded


def _free_variables(fn: ast.FunctionDef) -> set:
    """Names ``fn`` reads that it never binds itself -- i.e. names that must
    resolve from an enclosing (or module/builtin) scope."""
    return {
        n
        for n in _loaded_names(fn)
        if n not in _own_locals(fn) and n not in _BUILTIN_NAMES
    }


def _module_level_names(tree: ast.Module) -> set:
    """Names bound directly at module scope (imports/assigns/def/class),
    regardless of any module-level conditional guard (e.g. an optional-
    dependency ``try/except`` at import time) -- these are always available
    to any function in the module by the time it runs, so they are exempt
    from this test's "unconditional inside the function" requirement."""
    names = set()
    for node in ast.walk(tree):
        # ast.walk includes nested nodes too, but we only want statements
        # that are actually reachable as *module*-level bindings; using
        # ast.walk here (rather than tree.body only) deliberately also
        # catches names bound inside a module-level try/except/if (optional
        # imports), which is correct: those still land in module globals.
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            # Only count as module-level if not nested inside any FunctionDef.
            pass
    # Precise version: walk tree.body recursively but stop descending into
    # any FunctionDef/ClassDef/Lambda boundary.
    def walk(stmts):
        for stmt in stmts:
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.add(stmt.name)
                continue  # do not descend -- its body is a separate scope
            if isinstance(stmt, ast.Lambda):
                continue
            if isinstance(stmt, (ast.Import, ast.ImportFrom)):
                for a in stmt.names:
                    names.add(a.asname or a.name.split(".")[0])
            if isinstance(stmt, ast.Assign):
                for t in stmt.targets:
                    for n in ast.walk(t):
                        if isinstance(n, ast.Name):
                            names.add(n.id)
            if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                names.add(stmt.target.id)
            for field in ("body", "orelse", "finalbody"):
                sub = getattr(stmt, field, None)
                if sub:
                    walk(sub)
            if isinstance(stmt, ast.Try):
                for h in stmt.handlers:
                    walk(h.body)
            if isinstance(stmt, ast.With):
                walk(stmt.body)

    walk(tree.body)
    return names


def _call_sites(fn: ast.FunctionDef, callee_name: str) -> list:
    return [
        node.lineno
        for node in ast.walk(fn)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == callee_name
    ]


def _unconditional_binding_lines(fn: ast.FunctionDef) -> dict:
    """Map name -> earliest line number at which it is bound *unconditionally*
    (i.e. not nested inside any ``if`` block) directly within ``fn``'s own
    body. Does not descend into nested function bodies (those are a separate
    scope and binding a name only there does NOT make it available in ``fn``
    itself, which is exactly the #729 bug: the pre-training-report branch's
    import only bound the name inside its own ``if`` block).
    """
    bindings: dict = {}

    def record(name, lineno):
        if name not in bindings or lineno < bindings[name]:
            bindings[name] = lineno

    def walk(stmts, if_depth):
        for stmt in stmts:
            if isinstance(stmt, ast.If):
                walk(stmt.body, if_depth + 1)
                walk(stmt.orelse, if_depth + 1)
                continue
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if if_depth == 0:
                    record(stmt.name, stmt.lineno)
                continue  # nested function body is a separate scope
            if isinstance(stmt, ast.ClassDef):
                if if_depth == 0:
                    record(stmt.name, stmt.lineno)
                continue
            if if_depth == 0:
                if isinstance(stmt, ast.Assign):
                    for t in stmt.targets:
                        for n in ast.walk(t):
                            if isinstance(n, ast.Name):
                                record(n.id, stmt.lineno)
                elif isinstance(stmt, (ast.Import, ast.ImportFrom)):
                    for a in stmt.names:
                        record(a.asname or a.name.split(".")[0], stmt.lineno)
                elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                    record(stmt.target.id, stmt.lineno)
                elif isinstance(stmt, ast.For):
                    for n in ast.walk(stmt.target):
                        if isinstance(n, ast.Name):
                            record(n.id, stmt.lineno)
            for field in ("body", "orelse", "finalbody"):
                sub = getattr(stmt, field, None)
                if sub:
                    walk(sub, if_depth)
            if isinstance(stmt, ast.Try):
                for h in stmt.handlers:
                    walk(h.body, if_depth)
            if isinstance(stmt, ast.With):
                walk(stmt.body, if_depth)

    walk(fn.body, 0)
    return bindings


@pytest.fixture(scope="module")
def module_tree() -> ast.Module:
    return ast.parse(TRAIN_STREAMING.read_text())


@pytest.fixture(scope="module")
def outer_fn(module_tree) -> ast.FunctionDef:
    return _find_function(module_tree, OUTER_FN_NAME)


@pytest.fixture(scope="module")
def builder_fn(outer_fn) -> ast.FunctionDef:
    return _find_function(outer_fn, BUILDER_NAME, within=outer_fn)


def test_extraction_sanity_floor(outer_fn, builder_fn):
    """If the extraction machinery silently breaks (e.g. the functions get
    renamed/restructured), it must fail loudly rather than pass vacuously."""
    assert outer_fn.args.args, "streaming_training() unexpectedly has no parameters"
    free_vars = _free_variables(builder_fn)
    # Long-standing free variables from unrelated report fields -- if these
    # ever go missing, the extraction itself is broken.
    for anchor in ("args", "streaming_scene"):
        assert anchor in free_vars, (
            f"free-variable extraction looks broken: expected anchor "
            f"{anchor!r} among {sorted(free_vars)}"
        )


def test_free_variables_are_all_available_before_any_call(
    module_tree, outer_fn, builder_fn
):
    """The general #729 guard: every free variable
    ``_streaming_report_kwargs()`` reads must be guaranteed bound by the time
    ANY call to it executes -- not just some call paths.

    A name clears this check if either:
      (a) it is a parameter of ``streaming_training()`` (always bound), or
      (b) it is bound at module scope (always available), or
      (c) it has an UNCONDITIONAL (not nested inside any ``if``) binding
          directly in ``streaming_training()``'s own body, at a line number
          before the earliest textual call to ``_streaming_report_kwargs()``
          anywhere in the function.

    Before the #729 fix, ``_git_provenance_from_args`` fails all three: it is
    not a parameter, not a module-level name, and its only bindings inside
    ``streaming_training()`` are nested inside an ``if`` block (the
    pre-training-report gate) or occur in the FINAL report block, which is
    textually (and, more importantly, causally -- it only runs after every
    mid-training tick) after the earliest real call site.
    """
    free_vars = _free_variables(builder_fn)
    fn_params = {
        a.arg
        for a in outer_fn.args.args + outer_fn.args.posonlyargs + outer_fn.args.kwonlyargs
    }
    module_names = _module_level_names(module_tree)
    unconditional = _unconditional_binding_lines(outer_fn)

    call_lines = _call_sites(outer_fn, BUILDER_NAME)
    assert call_lines, f"found no calls to {BUILDER_NAME}() inside {OUTER_FN_NAME}()"
    min_call_line = min(call_lines)

    unresolved = []
    for name in sorted(free_vars):
        if name in fn_params or name in module_names:
            continue
        bound_line = unconditional.get(name)
        if bound_line is not None and bound_line < min_call_line:
            continue
        unresolved.append((name, bound_line))

    assert not unresolved, (
        f"{BUILDER_NAME}() reads free variables that are not guaranteed bound "
        f"before its earliest call site (line {min_call_line}) in "
        f"{OUTER_FN_NAME}(): {unresolved}. Each of these binds only inside a "
        "conditional block (or not at all) at the point of the earliest "
        "call -- this is exactly the issue #729 NameError class: a mid-run "
        "report call reads a name that only a gated/later code path ever "
        "assigns. Bind it unconditionally, near the top of "
        f"{OUTER_FN_NAME}(), before this test will pass."
    )


def test_git_provenance_pinned(module_tree, outer_fn, builder_fn):
    """The specific #729 regression: ``_git_provenance_from_args`` must be
    one of the names cleared by the general check above. Fails before the
    fix (NameError on every mid-training report), passes after."""
    free_vars = _free_variables(builder_fn)
    assert "_git_provenance_from_args" in free_vars, (
        "extraction regressed: _streaming_report_kwargs() no longer appears "
        "to read _git_provenance_from_args -- if that field was removed "
        "intentionally, retarget/remove this pin; otherwise the extraction "
        "is broken."
    )

    fn_params = {
        a.arg
        for a in outer_fn.args.args + outer_fn.args.posonlyargs + outer_fn.args.kwonlyargs
    }
    module_names = _module_level_names(module_tree)
    unconditional = _unconditional_binding_lines(outer_fn)
    call_lines = _call_sites(outer_fn, BUILDER_NAME)
    min_call_line = min(call_lines)

    name = "_git_provenance_from_args"
    ok = name in fn_params or name in module_names or (
        unconditional.get(name) is not None and unconditional[name] < min_call_line
    )
    assert ok, (
        "_git_provenance_from_args is not guaranteed bound before the "
        f"earliest _streaming_report_kwargs() call (line {min_call_line}) -- "
        "issue #729 is not fixed. It must be bound unconditionally (no "
        "surrounding `if`) directly in streaming_training()'s own body, "
        "before that line."
    )
