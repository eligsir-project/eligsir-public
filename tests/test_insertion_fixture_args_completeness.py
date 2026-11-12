"""Every partial-args test fixture that reaches ``insert_gaussians_from_frame``
must declare every attribute that ``integration_planner.py`` reads *bare*
(``args.<flag>``, no ``getattr`` fallback).

Why this test exists (issue #408, the tail of issue #368): #368's sweep removes
a ``getattr(args, "<flag>", <literal>)`` fallback whenever the fallback branch
is dead on every production path -- that is the ``_GUARDED`` pattern in
``tests/test_getattr_fallback_freshness.py`` and it is the right call, because
``insert_gaussians_from_frame`` is only ever called from ``train_streaming.py``
and ``eligsir/mapping/coordinator.py``, both with the full parsed args.

But it has a real, already-realised failure mode: the repo's *test* fixtures
are deliberately partial ``_Args`` namespaces, and removing a fallback turns
"this fixture silently got the literal" into ``AttributeError``. PR #394/#404
bare-accessed ``streaming_insert_scale_mode`` and left FOUR such fixtures
undeclared -- two of them (``tests/test_integration_services_delegate.py``)
were red on ``main`` for it, and the other two are GPU-gated so they were not
even visible on a CPU test run.

``test_getattr_fallback_freshness.py``'s ``_GUARDED`` list guards the *source*
side of that change ("this flag must never carry a fallback again"). This test
guards the *fixture* side, and would have caught the #394/#404 breakage on CPU.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_PLANNER = _REPO_ROOT / "eligsir" / "mapping" / "integration" / "integration_planner.py"

# Every test module that builds a partial ``_Args`` and hands it to
# ``insert_gaussians_from_frame`` (directly or via MappingCoordinator/
# RGBDIntegrator). Two of these are ``pytest.mark.skipif(not
# torch.cuda.is_available())`` -- this test reads them as source, so it covers
# them on a CPU-only runner too, which is the whole point.
_FIXTURE_MODULES = [
    "test_insertion_hot_path_backend_parity.py",
    "test_gpu_frame_resident_fused_mask_parity.py",
    "test_insert_from_frame_default_path_birth_iter.py",
    "test_integration_services_delegate.py",
]


def _bare_arg_attrs(path: Path) -> set[str]:
    """Names read as ``args.<name>`` (bare attribute access) in a module."""
    tree = ast.parse(path.read_text())
    found: set[str] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "args"
        ):
            found.add(node.attr)
    return found


def _declared_arg_attrs(path: Path) -> set[str]:
    """Names a module's ``_Args`` fixture declares.

    Covers both shapes used in this repo: a ``class _Args:`` body of plain
    assignments (module-level or nested inside a test function), and
    ``_Args(foo=..., bar=...)`` keyword construction.
    """
    tree = ast.parse(path.read_text())
    declared: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "_Args":
            for stmt in node.body:
                if isinstance(stmt, ast.Assign):
                    for tgt in stmt.targets:
                        if isinstance(tgt, ast.Name):
                            declared.add(tgt.id)
                elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                    declared.add(stmt.target.id)
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_Args"
        ):
            for kw in node.keywords:
                if kw.arg is not None:
                    declared.add(kw.arg)
    return declared


def test_planner_has_at_least_the_flags_issue_408_unwrapped():
    """Sanity check that the AST scan actually sees the #408/#368 bare reads --
    otherwise the real test below could pass vacuously."""
    bare = _bare_arg_attrs(_PLANNER)
    for flag in (
        "streaming_max_new_gaussians_per_frame",
        "streaming_insert_scale_mode",
        "streaming_depth_filter_enabled",
        "streaming_depth_temporal_check",
        "streaming_depth_temporal_require_for_insert",
        "streaming_depth_temporal_require_for_sensor_closer",
    ):
        assert flag in bare, (
            f"{flag} is no longer bare-accessed in integration_planner.py. If a "
            "getattr fallback was deliberately reintroduced, remove it from "
            "_GUARDED in tests/test_getattr_fallback_freshness.py and drop it "
            "from this list; if this is accidental, restore the bare access."
        )


@pytest.mark.parametrize("module_name", _FIXTURE_MODULES)
def test_partial_args_fixtures_declare_every_bare_read_flag(module_name):
    required = _bare_arg_attrs(_PLANNER)
    fixture_path = Path(__file__).parent / module_name
    declared = _declared_arg_attrs(fixture_path)
    assert declared, f"{module_name}: no _Args fixture found -- did it get renamed?"
    missing = sorted(required - declared)
    assert not missing, (
        f"{module_name}'s _Args fixture is missing attribute(s) that "
        f"integration_planner.py reads WITHOUT a getattr fallback: {missing}.\n"
        "This raises AttributeError at runtime (silently invisible on a CPU-only "
        "run for the GPU-gated fixtures). Declare each one at its real parser "
        "default from arguments/__init__.py -- see issue #408."
    )
