"""AST-based, import-free source inspection for call-site regression guards.

Several regression guards in this suite assert on the *text* of a production
function -- that a call site still passes a given flag, that a trainer still
calls a shared helper, that a module still imports ``os``. The obvious
implementation is ``import train; inspect.getsource(train.training)``, but that
requires **importing** the module, which drags in torch (directly, or
transitively through ``gaussian_renderer`` / ``scene`` / ``tqdm``-adjacent
training code) and therefore raises ``ModuleNotFoundError`` on the deliberately
torch-free CPU smoke image built by ``Dockerfile.ci-cpu``.

That is issue #1112, an instance of the #536 defect class: the guard does not
merely fail to run, it *errors before asserting anything*, so a check that looks
red for an "environment" reason is in fact guarding nothing at all on the
surface CI actually exercises.

Reading the file off disk and locating the function with ``ast`` yields the same
text with **no import at all**, so these guards run -- and actually guard -- on
every surface, torch present or not. It is also a strictly tighter assertion:
``inspect.getsource`` reports whatever object happens to be bound to a name at
runtime (a re-export, a decorator's wrapper, a monkeypatch), whereas this reads
the definition that literally lives in the named file.

The function-range mechanism deliberately mirrors
``tests/test_metric_registry_producers.py::_find_function_in_file``, the
established in-repo precedent for locating a function without importing it.

These helpers are pure-stdlib and import nothing from the repo, so importing
this module can never itself re-introduce the problem it exists to solve.
"""

from __future__ import annotations

import ast
import pathlib
from typing import Union

_FuncDef = Union[ast.FunctionDef, ast.AsyncFunctionDef]


def repo_root() -> pathlib.Path:
    """The repository root, resolved from this file's own location."""
    return pathlib.Path(__file__).resolve().parent.parent


def _resolve(rel_path: str) -> pathlib.Path:
    """Resolve a repo-relative path (the normal case) or an absolute one.

    Absolute paths are accepted so these helpers can be exercised against
    throwaway fixture files under ``tmp_path``; production guards always pass a
    repo-relative path.
    """
    candidate = pathlib.Path(rel_path)
    path = candidate if candidate.is_absolute() else repo_root() / rel_path
    assert path.exists(), f"source file not found: {rel_path}"
    return path


def _parse(rel_path: str) -> tuple[ast.Module, list[str]]:
    path = _resolve(rel_path)
    text = path.read_text()
    return ast.parse(text, filename=str(path)), text.splitlines()


def function_source(rel_path: str, func_name: str) -> str:
    """Return the source text of the module-level ``func_name`` in ``rel_path``.

    Only **module-level** definitions are considered. Matching nested helpers
    too would make the result ambiguous exactly where these guards care most:
    ``train_streaming.streaming_training`` contains dozens of nested closures,
    and silently returning one of those instead would turn a real regression
    into a passing test.

    Raises ``AssertionError`` (not ``LookupError``) so a rename shows up as a
    named test failure rather than an error, matching how the callers report.
    """
    tree, lines = _parse(rel_path)
    matches: list[_FuncDef] = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == func_name
    ]
    assert matches, (
        f"no module-level function named {func_name!r} in {rel_path} "
        f"(renamed, moved, or nested inside another function?)"
    )
    assert len(matches) == 1, (
        f"{len(matches)} module-level definitions of {func_name!r} in {rel_path}; "
        f"the guard cannot tell which one it is meant to inspect"
    )
    node = matches[0]
    end = node.end_lineno or node.lineno
    return "\n".join(lines[node.lineno - 1 : end])


def module_level_imports(rel_path: str) -> set[str]:
    """Names bound by module-scope ``import`` / ``from ... import`` statements.

    Returns the bound names, i.e. the aliases as they end up in the module
    namespace: ``import os.path`` binds ``os``, ``import numpy as np`` binds
    ``np``, ``from x import y as z`` binds ``z``.

    Module-scope only, on purpose: a function-local ``import os`` does not make
    ``os`` available to the *other* functions in the module, which is precisely
    the bug the ``os`` guard below exists to catch.
    """
    tree, _ = _parse(rel_path)
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                # `import os.path` binds the top-level package name `os`.
                names.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                names.add(alias.asname or alias.name)
    return names
