"""Regression + demonstration tests for issue #478.

The defect: ``tests/conftest.py`` used to hand-maintain a list of
torch-importing test files (``_NEEDS_TORCH``) to exclude from collection in
the torch-free CPU image. A file that imports torch and isn't on that list
is not excluded -- it hard-fails COLLECTION, which aborts the entire pytest
run (``Interrupted: 1 error during collection``), not just that one file.
This happened for real: PR #476 added ``tests/test_seed_cohort_offline_replay.py``
without registering it, and the CPU suite silently dropped from ~1385 tests
to 22 until PR #471 added the entry as a side effect of unrelated work.

``tests/_torch_collect_scan.py`` replaces the hand list with a derivation
(issue #478's option 2), so a new torch-importing file needs zero
``conftest.py`` edits to be excluded correctly. This file is the proof that
the derivation actually works, including on the transitive-import case a
naive top-level-only scan misses.

Deliberately torch-free itself -- must run (and matter) on the CPU image.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _torch_collect_scan as scan  # noqa: E402

REPO_ROOT = scan.REPO_ROOT


# ---------------------------------------------------------------------------
# Direct case: a file whose own module-scope `import torch` the scanner must
# obviously catch.
# ---------------------------------------------------------------------------
def test_direct_import_is_flagged():
    assert scan.file_requires_torch(REPO_ROOT / "train_streaming.py") is True


# ---------------------------------------------------------------------------
# Transitive case: THE trap this issue calls out by name -- a test file with
# zero literal "torch" token anywhere in it, that only imports a repo-local
# helper which itself imports torch. A naive top-level-only AST/string scan
# of the TEST FILE ALONE cannot see this; it has to follow the import graph.
# ---------------------------------------------------------------------------
def test_transitive_import_is_flagged_train_streaming():
    target = REPO_ROOT / "tests" / "test_mapping_coordinator_backend.py"
    # Prove the premise first, or this assertion would pass vacuously: this
    # file must contain literally zero "torch" token of its own (not even in
    # a comment/docstring, e.g. the word "torch-free" -- checked exactly).
    assert "torch" not in target.read_text(), (
        "test_mapping_coordinator_backend.py now contains a literal 'torch' "
        "token -- it is no longer a pure transitive-import case; find "
        "another zero-literal-token example or this test proves nothing."
    )
    assert scan.file_requires_torch(target) is True
    # And the mechanism: it imports eligsir.mapping.coordinator (a
    # multi-level package, exercising the ancestor-__init__.py walk too),
    # which eventually needs torch.
    assert "eligsir.mapping.coordinator" in target.read_text()


def test_spec_from_file_location_target_is_followed():
    """A second, independently-real transitive shape: a test file that loads
    a scripts/*.py module by path (invisible to any import-STATEMENT scan,
    naive or not) rather than a Python import statement. Confirmed already
    present, and already correctly excluded in `conftest.py`'s hand list --
    but only by luck of a human noticing, not by any mechanism, until now."""
    target = REPO_ROOT / "tests" / "test_w3_02_throughput_gate.py"
    assert "torch" not in target.read_text(), (
        "test_w3_02_throughput_gate.py now contains a literal 'torch' token "
        "-- find another spec_from_file_location-only example."
    )
    assert scan.file_requires_torch(target) is True
    helper = REPO_ROOT / "scripts" / "w3_02_level_d_report.py"
    assert scan.file_requires_torch(helper) is True


# ---------------------------------------------------------------------------
# Guard-awareness: must NOT flag a file whose torch import cannot actually
# break collection -- re-flagging these would re-introduce issue #462
# (silent over-exclusion of real, collectible coverage).
# ---------------------------------------------------------------------------
def test_try_except_import_error_guard_not_flagged(tmp_path):
    (tmp_path / "test_guarded.py").write_text(
        "try:\n"
        "    import torch\n"
        "except ImportError:\n"
        "    torch = None\n"
    )
    assert scan.file_requires_torch(tmp_path / "test_guarded.py") is False


def test_bare_except_guard_not_flagged(tmp_path):
    (tmp_path / "test_guarded.py").write_text(
        "try:\n"
        "    import torch\n"
        "except Exception:\n"
        "    torch = None\n"
    )
    assert scan.file_requires_torch(tmp_path / "test_guarded.py") is False


def test_importorskip_guard_not_flagged(tmp_path):
    (tmp_path / "test_guarded.py").write_text(
        "import pytest\n"
        "pytest.importorskip('torch')\n"
        "import torch\n"
    )
    assert scan.file_requires_torch(tmp_path / "test_guarded.py") is False


def test_type_checking_guard_not_flagged(tmp_path):
    (tmp_path / "test_guarded.py").write_text(
        "from typing import TYPE_CHECKING\n"
        "if TYPE_CHECKING:\n"
        "    import torch\n"
    )
    assert scan.file_requires_torch(tmp_path / "test_guarded.py") is False


def test_find_spec_presence_guard_not_flagged(tmp_path):
    """This repo's own real idiom (tests/test_accumulate_observed_surface_
    telemetry.py) -- a naive version of this scanner false-positived on it
    during development; kept as a regression test."""
    (tmp_path / "test_guarded.py").write_text(
        "import importlib.util\n"
        "_HAS_TORCH = importlib.util.find_spec('torch') is not None\n"
        "if _HAS_TORCH:\n"
        "    import torch\n"
    )
    assert scan.file_requires_torch(tmp_path / "test_guarded.py") is False


def test_function_body_import_not_flagged(tmp_path):
    """An import inside a function/fixture only runs when called -- it
    cannot be what breaks COLLECTION."""
    (tmp_path / "test_guarded.py").write_text(
        "def helper():\n"
        "    import torch\n"
        "    return torch\n"
    )
    assert scan.file_requires_torch(tmp_path / "test_guarded.py") is False


def test_unguarded_module_scope_import_is_flagged(tmp_path):
    """Sanity control for the guard tests above: an UNGUARDED module-scope
    import in the same tmp_path harness must still be flagged."""
    (tmp_path / "test_unguarded.py").write_text("import torch\n")
    assert scan.file_requires_torch(tmp_path / "test_unguarded.py") is True


# ---------------------------------------------------------------------------
# Historical case (issue #478's own reproduction): PR #476 added
# tests/test_seed_cohort_offline_replay.py without registering it in the
# hand list.
# ---------------------------------------------------------------------------
def test_historical_476_file_direct_import_is_flagged():
    """Correction to the issue's own text: #478 describes #476's file as
    importing torch "via a helper chain", implying a top-level scan would
    miss it. Verified against the real merge commit (fdee28f /
    79f45cd, both identical): the file has a literal, module-scope
    `import torch` at line 19 -- a plain top-level scan WOULD have caught
    this specific file. The general transitive-import risk the issue warns
    about is nonetheless real (see test_transitive_import_is_flagged_train_streaming
    and test_spec_from_file_location_target_is_followed above, both
    zero-literal-token real cases already in this tree) -- this test just
    keeps the historical claim about *this specific file* accurate."""
    target = REPO_ROOT / "tests" / "test_seed_cohort_offline_replay.py"
    src = target.read_text()
    tree = ast.parse(src)
    has_direct_module_scope_import = any(
        isinstance(node, ast.Import) and any(a.name == "torch" for a in node.names)
        for node in tree.body
    )
    assert has_direct_module_scope_import, (
        "test_seed_cohort_offline_replay.py no longer has a direct "
        "module-scope `import torch` -- re-check whether #478's "
        "'helper chain' description now applies literally and update this "
        "test's docstring/assertions accordingly."
    )
    assert scan.file_requires_torch(target) is True


def test_synthetic_476_shaped_omission_is_auto_derived(tmp_path, monkeypatch):
    """The actual end-to-end proof: a NEW test file added to a synthetic
    tests/ dir, that nobody registers anywhere, is still correctly derived
    as torch-needing -- reproducing what PR #471 had to do by hand for real,
    with zero list edits. Uses a transitive-only import (not `import torch`
    directly) to also cover the exact shape #478 accuses #476 of using."""
    monkeypatch.setattr(scan, "REPO_ROOT", tmp_path)
    (tmp_path / "helper_needs_torch.py").write_text("import torch\n")
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_new_unregistered.py").write_text(
        "from helper_needs_torch import x\n"
    )
    (tests_dir / "test_already_clean.py").write_text("import os\n")

    discovered = scan.discover_torch_needing_test_files(tests_dir=tests_dir)
    assert discovered == ["test_new_unregistered.py"]


# ---------------------------------------------------------------------------
# Wiring: conftest.py must actually be USING this derivation, not just
# having it sitting unused in the repo.
# ---------------------------------------------------------------------------
def test_conftest_derives_needs_torch_from_the_scanner():
    memo: dict = {}
    expected = [
        p.name
        for p in sorted((REPO_ROOT / "tests").glob("test_*.py"))
        if scan.file_requires_torch(p, _memo=memo)
    ]
    conftest_src = (REPO_ROOT / "tests" / "conftest.py").read_text()
    tree = ast.parse(conftest_src)
    call = None
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "discover_torch_needing_test_files"
        ):
            call = node
            break
    assert call is not None, (
        "tests/conftest.py no longer calls "
        "_torch_collect_scan.discover_torch_needing_test_files() -- the "
        "hand-maintained _NEEDS_TORCH list may have been reintroduced. See "
        "issue #478."
    )
    # And the derivation itself must not be empty/broken.
    assert len(expected) > 100
