"""Static, torch-free AST scan: does *importing* a given ``.py`` file require
``torch`` (directly, or transitively through this repo's own modules)?

Written for issue #478. The problem this exists to solve: ``tests/conftest.py``
used to hand-maintain ``_NEEDS_TORCH``, a list of test files to ``collect_ignore``
in the torch-free CPU image. A new torch-importing test file that nobody
remembers to add to that list is not ignored -- it hard-fails COLLECTION
(``ModuleNotFoundError``), which aborts the entire ``pytest`` run
(``Interrupted: 1 error during collection``). This happened for real: PR #476
added ``tests/test_seed_cohort_offline_replay.py`` without registering it, and
the CPU suite silently dropped from ~1385 tests to 22 until PR #471 added the
entry as a side effect of unrelated work.

This module replaces the hand list with a derivation, so a new torch-importing
file needs zero conftest.py edits to be excluded correctly (option 2 of #478's
three listed options -- see the module docstring addition in conftest.py for
why over option 1's "add a guard test instead").

**What "transitively" covers, concretely (this is the part a naive
top-level-only ``"torch" in source`` or single-file AST scan misses):**

* ``import torch`` / ``from torch import x`` / ``from torch.sub import x`` at
  module scope in the scanned file itself.
* The same, in any repo-local module the file imports at module scope,
  recursively -- e.g. a test file that only does ``from train_streaming import
  X`` is flagged because ``train_streaming.py`` itself does ``import torch``,
  even though the string ``"torch"`` never appears in the test file. Verified
  against a real, already-in-tree case: ``tests/test_replay_buffer_reachability.py``
  imports only ``train_streaming``, with zero literal ``torch`` token anywhere
  in the file, and is correctly flagged by this scanner (see
  ``tests/test_torch_collect_scan.py``).
* Every ancestor package's ``__init__.py`` that Python would execute along the
  way (``import a.b.c`` also runs ``a/__init__.py`` and ``a/b/__init__.py``).
* This repo's ``importlib.util.spec_from_file_location(name, <path>)`` pattern
  for loading a ``scripts/*.py`` module by path (invisible to a plain
  import-statement scan) -- best-effort, see ``_spec_from_file_location_target``.

**What it deliberately does NOT flag (guard-awareness, to avoid re-introducing
issue #462's opposite bug: over-exclusion silently dropping real coverage):**

* An import inside ``try: import torch / except (ImportError,
  ModuleNotFoundError, Exception, BaseException):`` (or a bare ``except:``) --
  the file is importable either way, so it must not be excluded.
* Code inside ``if TYPE_CHECKING:`` -- never executed at runtime.
* Anything textually after a module-scope ``pytest.importorskip("torch")`` --
  pytest treats a raised ``Skipped`` during collection as a skip, not a
  collection error; it does not abort the run the way a raw
  ``ModuleNotFoundError`` does, so these files must not be excluded either
  (``tests/conftest.py``'s own docstring names ``test_scored_subset_hash.py``
  as exactly this pattern).
* Anything inside a function/method/lambda body -- those run when *called*,
  not at import/collection time, so they cannot be what breaks collection.

**Known, honest gaps (not fixed here):**

* Only modules that resolve to a real ``.py``/package under this repo's own
  root are expanded. A third-party package this repo depends on but does not
  vendor (say, some future dependency that conditionally imports torch
  internally) is treated as an opaque leaf -- correct for every dependency
  this repo currently has (none of pydantic/pytest/numpy/etc. import torch),
  but not a guarantee for an arbitrary future one.
* ``importlib.util.spec_from_file_location`` target resolution is a literal-
  string-segment heuristic (join every string constant found in the location
  expression, minus ``".."`` tokens, onto the repo root), not a real constant
  folder -- it resolves every such call actually present in this repo today
  (verified in ``tests/test_torch_collect_scan.py``) but a differently-shaped
  path expression (e.g. built via an f-string, or via a module-level variable
  assigned several statements earlier) would not resolve, and the call is then
  silently skipped -- NOT flagged as needing torch even if its target does.
  This is a real, narrow blind spot: a test file whose *only* torch dependency
  routes through such a call would be silently mis-classified as torch-free.
* Circular imports are treated as "not disqualifying" at the cycle-closing
  edge (returns ``False`` for a module already being visited) rather than
  resolved properly -- correct in every case in this repo today (no import
  cycle involves torch), not a general guarantee.
"""
from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

_BROAD_EXCEPT_NAMES = {"ImportError", "ModuleNotFoundError", "Exception", "BaseException"}
_TORCH_ROOTS = {"torch", "torchvision"}


def _is_type_checking_test(test: ast.expr) -> bool:
    return (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
        isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"
    )


def _handler_catches_import_failure(handler: ast.ExceptHandler) -> bool:
    if handler.type is None:  # bare `except:`
        return True
    elts = handler.type.elts if isinstance(handler.type, ast.Tuple) else [handler.type]
    names = []
    for e in elts:
        if isinstance(e, ast.Name):
            names.append(e.id)
        elif isinstance(e, ast.Attribute):
            names.append(e.attr)
    return any(n in _BROAD_EXCEPT_NAMES for n in names)


def _is_find_spec_torch_call(node: ast.expr) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "find_spec"
        and bool(node.args)
        and isinstance(node.args[0], ast.Constant)
        and isinstance(node.args[0].value, str)
        and node.args[0].value.split(".")[0] in _TORCH_ROOTS
    )


def _is_none_const(node: ast.expr) -> bool:
    return isinstance(node, ast.Constant) and node.value is None


def _torch_presence_polarity(expr: ast.expr, guard_vars: dict[str, bool]) -> bool | None:
    """True if `expr` is truthy exactly when torch IS importable, False if
    truthy exactly when torch is NOT importable, None if unrelated to a
    torch-presence check (`importlib.util.find_spec("torch")`, directly or
    via a variable already known to hold one -- e.g. this repo's own
    ``_HAS_TORCH = importlib.util.find_spec("torch") is not None`` /
    ``_TORCH_AVAILABLE`` pattern, see ``tests/conftest.py`` and
    ``tests/test_accumulate_observed_surface_telemetry.py``)."""
    if isinstance(expr, ast.UnaryOp) and isinstance(expr.op, ast.Not):
        inner = _torch_presence_polarity(expr.operand, guard_vars)
        return None if inner is None else (not inner)
    if isinstance(expr, ast.Name) and expr.id in guard_vars:
        return guard_vars[expr.id]
    if isinstance(expr, ast.Compare) and len(expr.ops) == 1:
        left, op, right = expr.left, expr.ops[0], expr.comparators[0]
        if _is_find_spec_torch_call(left):
            if isinstance(op, ast.IsNot) and _is_none_const(right):
                return True
            if isinstance(op, ast.Is) and _is_none_const(right):
                return False
    if _is_find_spec_torch_call(expr):
        return True
    return None


def _collect_guard_vars(body: list[ast.stmt]) -> dict[str, bool]:
    """Top-level `NAME = <torch-presence-check>` assignments, e.g.
    `_TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None`."""
    guard_vars: dict[str, bool] = {}
    for stmt in body:
        if (
            isinstance(stmt, ast.Assign)
            and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name)
        ):
            polarity = _torch_presence_polarity(stmt.value, guard_vars)
            if polarity is not None:
                guard_vars[stmt.targets[0].id] = polarity
    return guard_vars


def _flatten_executed_stmts(
    body: list[ast.stmt], guard_vars: dict[str, bool] | None = None
) -> list[ast.stmt]:
    """Order-preserving list of statements that run at IMPORT time, given
    torch IS ABSENT (the only environment this scanner's callers care about
    -- see `file_requires_torch`'s docstring).

    Descends into If/Try/With/ClassDef bodies (these execute when the module
    is imported) but never into Function/AsyncFunctionDef/Lambda bodies
    (those execute only when called -- irrelevant to whether collection
    itself fails). A torch import inside a Try guarded by an ImportError-or-
    broader handler is excluded, matching real usage in this repo. An `if`
    branch that only runs when torch IS present (`if _HAS_TORCH:` / `if
    importlib.util.find_spec("torch") is not None:`) is excluded the same
    way -- its imports cannot be what breaks collection in a torch-less
    image, since the branch never executes there.
    """
    if guard_vars is None:
        guard_vars = {}
    out: list[ast.stmt] = []
    for stmt in body:
        out.append(stmt)
        if isinstance(stmt, ast.If):
            polarity = _torch_presence_polarity(stmt.test, guard_vars)
            if polarity is True:
                out.extend(_flatten_executed_stmts(stmt.orelse, guard_vars))
            elif polarity is False:
                out.extend(_flatten_executed_stmts(stmt.body, guard_vars))
            elif not _is_type_checking_test(stmt.test):
                out.extend(_flatten_executed_stmts(stmt.body, guard_vars))
                out.extend(_flatten_executed_stmts(stmt.orelse, guard_vars))
            else:
                out.extend(_flatten_executed_stmts(stmt.orelse, guard_vars))
        elif isinstance(stmt, ast.Try):
            guarded = any(_handler_catches_import_failure(h) for h in stmt.handlers)
            if not guarded:
                out.extend(_flatten_executed_stmts(stmt.body, guard_vars))
            out.extend(_flatten_executed_stmts(stmt.orelse, guard_vars))
            out.extend(_flatten_executed_stmts(stmt.finalbody, guard_vars))
        elif isinstance(stmt, (ast.With, ast.AsyncWith)):
            out.extend(_flatten_executed_stmts(stmt.body, guard_vars))
        elif isinstance(stmt, ast.ClassDef):
            out.extend(_flatten_executed_stmts(stmt.body, guard_vars))
    return out


def _is_importorskip_torch(stmt: ast.stmt) -> bool:
    call = None
    if isinstance(stmt, ast.Expr):
        call = stmt.value
    elif isinstance(stmt, ast.Assign):
        call = stmt.value
    if not isinstance(call, ast.Call):
        return False
    func = call.func
    if not (isinstance(func, ast.Attribute) and func.attr == "importorskip"):
        return False
    if not call.args or not isinstance(call.args[0], ast.Constant):
        return False
    name = call.args[0].value
    return isinstance(name, str) and name.split(".")[0] in _TORCH_ROOTS


def _candidate_targets(stmt: ast.stmt, from_file: Path) -> list[tuple[Path, list[str]]]:
    """[(base_dir, dotted_name_parts), ...] this import statement needs."""
    targets: list[tuple[Path, list[str]]] = []
    if isinstance(stmt, ast.Import):
        for alias in stmt.names:
            targets.append((REPO_ROOT, alias.name.split(".")))
    elif isinstance(stmt, ast.ImportFrom):
        level = stmt.level or 0
        if level > 0:
            base = from_file.parent
            for _ in range(level - 1):
                base = base.parent
        else:
            base = REPO_ROOT
        mod_parts = stmt.module.split(".") if stmt.module else []
        if mod_parts:
            targets.append((base, mod_parts))
        for alias in stmt.names:
            if alias.name == "*":
                continue
            # `from pkg import sub` -- `sub` may itself be a submodule, not
            # just an attribute of `pkg`; try it as one too.
            targets.append((base, mod_parts + [alias.name]))
    return targets


def _module_file_and_ancestors(parts: list[str], base: Path) -> list[Path]:
    """Every file Python executes to import `base/parts...`: each ancestor
    package's __init__.py in order (if it has one -- an ancestor directory
    with NO __init__.py is a valid PEP 420 namespace package, e.g. this
    repo's own `utils/` and `eval/`, and contributes no file to execute),
    then the leaf module/package __init__. Returns [] if any component
    doesn't resolve under `base` at all (i.e. it is not a repo-local module
    -- stdlib/third-party, opaque leaf)."""
    files: list[Path] = []
    cur = base
    for i, part in enumerate(parts):
        cur = cur / part
        if i == len(parts) - 1:
            py = cur.with_suffix(".py")
            if py.is_file():
                files.append(py)
                return files
            init = cur / "__init__.py"
            if init.is_file():
                files.append(init)
                return files
            if cur.is_dir():
                return files  # namespace package leaf: nothing to execute
            return []
        if not cur.is_dir():
            return []
        init = cur / "__init__.py"
        if init.is_file():
            files.append(init)
    return files


def _is_spec_from_file_location_call(node: ast.expr) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "spec_from_file_location"
    )


def _ordered_string_constants(node: ast.AST):
    """Depth-first, field-order (i.e. left-to-right source order) walk
    yielding every string constant under `node`. Deliberately NOT
    `ast.walk` -- that is a breadth-first queue and reorders sibling
    subtrees (e.g. `Path(x).parents[1] / "scripts" / "y.py"` came back as
    `["y.py", "scripts"]`, silently wrong), which matters here because the
    join order below is significant."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        yield node.value
        return
    for _field, value in ast.iter_fields(node):
        if isinstance(value, ast.AST):
            yield from _ordered_string_constants(value)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, ast.AST):
                    yield from _ordered_string_constants(item)


def _spec_from_file_location_target(call: ast.Call) -> Path | None:
    """Best-effort resolve of this repo's `spec_from_file_location(name,
    <path-expr>)` pattern for loading a script by path -- see module
    docstring's "known gaps" section for exactly what this does and does not
    handle."""
    if len(call.args) < 2:
        return None
    segments = [s for s in _ordered_string_constants(call.args[1]) if s != ".."]
    if not segments:
        return None
    candidate = REPO_ROOT.joinpath(*segments)
    return candidate if candidate.is_file() else None


def file_requires_torch(
    path: Path,
    *,
    _memo: dict[Path, bool] | None = None,
    _visiting: set[Path] | None = None,
) -> bool:
    """Does importing `path` require torch, directly or transitively?"""
    path = path.resolve()
    if _memo is None:
        _memo = {}
    if _visiting is None:
        _visiting = set()
    if path in _memo:
        return _memo[path]
    if path in _visiting:
        return False  # import cycle; see module docstring's "known gaps"
    _visiting.add(path)

    try:
        tree = ast.parse(path.read_text(), filename=str(path))
    except (SyntaxError, UnicodeDecodeError, OSError):
        _visiting.discard(path)
        _memo[path] = False
        return False

    result = False
    torch_guarded = False
    guard_vars = _collect_guard_vars(tree.body)
    for stmt in _flatten_executed_stmts(tree.body, guard_vars):
        if _is_importorskip_torch(stmt):
            torch_guarded = True
            continue

        # spec_from_file_location(...) target, wherever it appears in this
        # statement (assignment RHS or bare call expression).
        for node in ast.walk(stmt):
            if _is_spec_from_file_location_call(node):
                target = _spec_from_file_location_target(node)
                if target is not None and not torch_guarded:
                    if file_requires_torch(target, _memo=_memo, _visiting=_visiting):
                        result = True

        if not isinstance(stmt, (ast.Import, ast.ImportFrom)):
            continue
        for base, parts in _candidate_targets(stmt, path):
            if not parts:
                continue
            if parts[0] in _TORCH_ROOTS:
                if not torch_guarded:
                    result = True
                continue
            for f in _module_file_and_ancestors(parts, base):
                if file_requires_torch(f, _memo=_memo, _visiting=_visiting):
                    if not torch_guarded:
                        result = True

    _visiting.discard(path)
    _memo[path] = result
    return result


def discover_torch_needing_test_files(tests_dir: Path | None = None) -> list[str]:
    """Every `tests/test_*.py` file whose import requires torch, by name
    (matching the shape `tests/conftest.py`'s old hand-maintained
    `_NEEDS_TORCH` list used)."""
    tests_dir = tests_dir or (REPO_ROOT / "tests")
    memo: dict[Path, bool] = {}
    out = []
    for path in sorted(tests_dir.glob("test_*.py")):
        if file_requires_torch(path, _memo=memo):
            out.append(path.name)
    return out
