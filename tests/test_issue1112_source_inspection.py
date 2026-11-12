"""Unit tests for ``tests/_source_inspection`` (#1112).

The helper exists so that call-site regression guards can read a production
function's text **without importing it**, because importing pulls in torch and
hard-fails on the torch-free CPU smoke image. These tests pin the two
properties the guards rely on:

* it returns the right function's text, and
* it raises loudly rather than returning something plausible-but-wrong when the
  function is missing or ambiguous -- a silent wrong answer here would turn
  every guard built on it into a test that passes while guarding nothing, which
  is the exact defect class (#536) #1112 was filed against.

This file itself must stay importable with torch absent, so it imports only the
helper and the stdlib.
"""

from __future__ import annotations

import pytest

from _source_inspection import function_source, module_level_imports


# --- function_source -------------------------------------------------------


def test_returns_the_named_functions_text():
    src = function_source("tests/_source_inspection.py", "module_level_imports")
    assert src.startswith("def module_level_imports(")
    assert "ast.ImportFrom" in src


def test_slice_stops_at_the_functions_end(tmp_path):
    """The returned text must not bleed into the *following* definition.

    Asserted against a fixture with two adjacent defs rather than against
    ``_source_inspection.py`` itself: a real-file assertion of this shape
    silently depends on definition ORDER, so it can keep passing for the wrong
    reason (the other def simply sits above the slice) after a reorder.
    """
    mod = tmp_path / "two_defs.py"
    mod.write_text("def first():\n    return 1\n\n\ndef second():\n    return 2\n")
    src = function_source(str(mod), "first")
    assert src.startswith("def first(")
    assert "return 1" in src
    assert "def second(" not in src
    assert "return 2" not in src


def test_missing_function_raises_rather_than_returning_empty():
    with pytest.raises(AssertionError, match="no module-level function"):
        function_source("tests/_source_inspection.py", "definitely_not_here")


def test_missing_file_raises():
    with pytest.raises(AssertionError, match="source file not found"):
        function_source("tests/does_not_exist_1112.py", "whatever")


def test_nested_functions_are_not_matched(tmp_path):
    """A helper nested inside another function must not satisfy the lookup.

    ``streaming_training`` is full of nested closures; silently returning one
    of those instead of the module-level target would make a real regression
    look green.
    """
    mod = tmp_path / "nested_sample.py"
    mod.write_text("def outer():\n    def inner():\n        return 1\n    return inner\n")
    with pytest.raises(AssertionError, match="no module-level function"):
        function_source(str(mod), "inner")
    # ...while the enclosing module-level function still resolves.
    assert function_source(str(mod), "outer").startswith("def outer(")


def test_real_targets_resolve():
    """The four call sites #1112's guards actually inspect."""
    for rel_path, fn_name, expect in (
        ("train.py", "training", "def training("),
        ("train_streaming.py", "streaming_training", "def streaming_training("),
        (
            "eligsir/mapping/integration/rolling_seed_application.py",
            "_run_rolling_seed",
            "def _run_rolling_seed(",
        ),
        (
            "eligsir/mapping/integration/submap_stitch_application.py",
            "_run_submap_stitch",
            "def _run_submap_stitch(",
        ),
    ):
        src = function_source(rel_path, fn_name)
        assert src.startswith(expect), f"{rel_path}:{fn_name}"


# --- module_level_imports --------------------------------------------------


def test_collects_plain_and_aliased_and_from_imports():
    names = module_level_imports("tests/_source_inspection.py")
    assert "ast" in names          # import ast
    assert "pathlib" in names      # import pathlib
    assert "Union" in names        # from typing import Union
    assert "annotations" in names  # from __future__ import annotations


def test_dotted_import_binds_the_top_level_package(tmp_path):
    mod = tmp_path / "dotted_sample.py"
    mod.write_text("import os.path\nimport numpy as np\nfrom a.b import c as d\n")
    names = module_level_imports(str(mod))
    assert names == {"os", "np", "d"}


def test_function_local_imports_are_not_counted(tmp_path):
    """A function-local import must NOT satisfy the module-scope `os` guard.

    This is the whole point of that guard: `_run_submap_stitch` calling
    `os.path.join` needs `os` bound at MODULE scope. An `import os` inside some
    other function in the same file leaves the NameError exactly as it was.
    """
    mod = tmp_path / "local_import_sample.py"
    mod.write_text(
        "import sys\n"
        "\n"
        "\n"
        "def helper():\n"
        "    import os\n"
        "    return os.path.sep\n"
    )
    names = module_level_imports(str(mod))
    assert names == {"sys"}, "a function-local import must not be reported"
