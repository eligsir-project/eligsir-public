"""Tests for scripts/report_ci_test_exclusions.py (issue #536).

Deliberately torch-free: this file must actually EXECUTE in the CPU smoke
image, not just be collected there -- it exists specifically to move the
"executed" count for the PR that adds the visibility mechanism itself (see
that PR's description for why an unmoved count would otherwise repeat the
exact silent-exclusion failure #536 is about).

Loads the script the same way this repo's own `importlib.util.
spec_from_file_location` idiom does elsewhere for `scripts/*.py` (recognized
by `tests/_torch_collect_scan.py`'s AST scanner, see that module's
docstring) rather than adding `scripts/` to a package `__init__.py`.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT_PATH = REPO_ROOT / "scripts" / "report_ci_test_exclusions.py"

_spec = importlib.util.spec_from_file_location("report_ci_test_exclusions", _SCRIPT_PATH)
report_ci_test_exclusions = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = report_ci_test_exclusions
_spec.loader.exec_module(report_ci_test_exclusions)


# ---------------------------------------------------------------------------
# parse_executed_summary
# ---------------------------------------------------------------------------


def test_parse_executed_summary_reads_the_last_summary_line(tmp_path):
    out = tmp_path / "pytest_output.txt"
    out.write_text(
        "..........\n"
        "=== warnings summary ===\n"
        "some/file.py::test_x\n"
        "  a warning line\n"
        "-- Docs: https://docs.pytest.org/...\n"
        "1552 passed, 116 skipped, 2 warnings in 28.16s\n"
    )
    assert (
        report_ci_test_exclusions.parse_executed_summary(out)
        == "1552 passed, 116 skipped, 2 warnings in 28.16s"
    )


def test_parse_executed_summary_ignores_rs_skip_reason_lines(tmp_path):
    # `-rs` (added by this PR) prints one "SKIPPED [...] file:line: reason"
    # line per skip, none of which end in "in <N>s" -- the parser must not
    # mistake one of those for the real summary line.
    out = tmp_path / "pytest_output.txt"
    out.write_text(
        "SKIPPED [1] tests/test_a.py:12: could not import 'torch': No module named 'torch'\n"
        "SKIPPED [4] tests/test_b.py: requires torch\n"
        "1552 passed, 116 skipped, 2 warnings in 28.16s\n"
    )
    assert (
        report_ci_test_exclusions.parse_executed_summary(out)
        == "1552 passed, 116 skipped, 2 warnings in 28.16s"
    )


def test_parse_executed_summary_missing_file_returns_none(tmp_path):
    assert report_ci_test_exclusions.parse_executed_summary(tmp_path / "does_not_exist.txt") is None


def test_parse_executed_summary_none_path_returns_none():
    assert report_ci_test_exclusions.parse_executed_summary(None) is None


def test_parse_executed_summary_no_matching_line_returns_none(tmp_path):
    out = tmp_path / "pytest_output.txt"
    out.write_text("nothing here looks like a summary line\njust noise\n")
    assert report_ci_test_exclusions.parse_executed_summary(out) is None


# ---------------------------------------------------------------------------
# _is_added_top_level_test_file (the option-2 PR-file filter)
# ---------------------------------------------------------------------------


def test_added_top_level_test_file_is_accepted():
    entry = {"filename": "tests/test_new_thing.py", "status": "added"}
    assert report_ci_test_exclusions._is_added_top_level_test_file(entry) is True


def test_modified_test_file_is_rejected_even_if_top_level():
    entry = {"filename": "tests/test_existing_thing.py", "status": "modified"}
    assert report_ci_test_exclusions._is_added_top_level_test_file(entry) is False


def test_added_subdirectory_test_file_is_rejected():
    # tests/regression/* and similar are out of scope, matching
    # _torch_collect_scan's own non-recursive tests/test_*.py glob.
    entry = {"filename": "tests/regression/test_new_thing.py", "status": "added"}
    assert report_ci_test_exclusions._is_added_top_level_test_file(entry) is False


def test_added_non_test_python_file_is_rejected():
    entry = {"filename": "tests/conftest.py", "status": "added"}
    assert report_ci_test_exclusions._is_added_top_level_test_file(entry) is False


def test_added_non_python_file_is_rejected():
    entry = {"filename": "tests/test_fixture_data.json", "status": "added"}
    assert report_ci_test_exclusions._is_added_top_level_test_file(entry) is False


def test_added_file_outside_tests_dir_is_rejected():
    entry = {"filename": "scripts/test_helper.py", "status": "added"}
    assert report_ci_test_exclusions._is_added_top_level_test_file(entry) is False


def test_renamed_status_is_rejected():
    # GitHub's file-status enum also includes "removed", "renamed",
    # "copied", "changed", "unchanged" -- only "added" is this PR's concern.
    entry = {"filename": "tests/test_x.py", "status": "renamed"}
    assert report_ci_test_exclusions._is_added_top_level_test_file(entry) is False


# ---------------------------------------------------------------------------
# _all_top_level_test_files: non-recursive scope, matching
# _torch_collect_scan.discover_torch_needing_test_files's own glob shape.
# ---------------------------------------------------------------------------


def test_all_top_level_test_files_excludes_subdirectories():
    files = report_ci_test_exclusions._all_top_level_test_files()
    assert all("/" not in f[len("tests/") :] for f in files)
    # Sanity: this repo actually has a subdirectory with test_*.py files,
    # and none of them should appear in the top-level set.
    assert not any(f.startswith("tests/regression/") for f in files)


def test_all_top_level_test_files_includes_this_file():
    files = report_ci_test_exclusions._all_top_level_test_files()
    assert "tests/test_report_ci_test_exclusions.py" in files


# ---------------------------------------------------------------------------
# reason_for: best-effort attribution, never raises.
# ---------------------------------------------------------------------------


def test_reason_for_a_nonexistent_file_does_not_raise():
    # file_requires_torch() on a missing path returns False (its own
    # documented behavior for unreadable/unparseable files), so this
    # should report the generic non-torch reason, not crash.
    reason = report_ci_test_exclusions.reason_for("tests/test_this_file_does_not_exist_536.py")
    assert isinstance(reason, str)
    assert reason  # non-empty


# ---------------------------------------------------------------------------
# This file itself must not be silently excluded -- verified directly
# against the ground truth (tests/conftest.py's own collect_ignore) and the
# #518 scanner, the same checks this PR's own mechanism performs on a PR's
# new test files.
# ---------------------------------------------------------------------------


def test_this_file_is_not_torch_flagged():
    sys.path.insert(0, str(REPO_ROOT / "tests"))
    import _torch_collect_scan  # noqa: PLC0415

    assert _torch_collect_scan.file_requires_torch(Path(__file__)) is False
