"""Training-surface reports describe actual JUnit outcomes, not import-scan guesses."""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import pytest

_SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts/report_training_surface.py"
_spec = importlib.util.spec_from_file_location("report_training_surface", _SCRIPT_PATH)
report = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(report)

_JUNIT = """<testsuite name="pytest" errors="1" failures="1" skipped="1" tests="4">
<testcase classname="tests.test_alpha" name="test_ok" time="0.1" />
<testcase classname="tests.test_beta" name="test_bad" time="0.1"><failure message="boom">tb</failure></testcase>
<testcase classname="tests.test_beta" name="test_err" time="0.1"><error message="boom">tb</error></testcase>
<testcase classname="tests.test_gamma" name="test_skipped" time="0.0"><skipped type="pytest.skip"/></testcase>
</testsuite>"""


@pytest.mark.parametrize("classname, path", [
    ("tests.test_foo", "tests/test_foo.py"),
    ("tests.regression.test_bar", "tests/regression/test_bar.py"),
    ("tests.test_foo.TestThing", "tests/test_foo.py"),
    ("conftest", ""), ("", ""),
])
def test_classname_to_file(classname, path):
    assert report.classname_to_file(classname) == path


@pytest.mark.parametrize("wrapper", ["{}", "<testsuites>{}</testsuites>"])
def test_junit_outcomes_nodeids_and_report_counts(tmp_path, wrapper):
    path = tmp_path / "junit.xml"
    path.write_text(wrapper.format(_JUNIT), encoding="utf-8")
    cases = report.parse_junit(path)
    assert {c["name"]: c["outcome"] for c in cases} == {
        "test_ok": "passed", "test_bad": "failed", "test_err": "error", "test_skipped": "skipped",
    }
    markdown, stats = report.build_report(cases, "ghcr.io/x/y:tag", "1 passed")
    assert stats == {
        "surface": "training-nogpu", "image": "ghcr.io/x/y:tag", "total": 4,
        "executed": 3, "passed": 1, "failed": 1, "errors": 1, "skipped": 1,
        "summary_line": "1 passed",
    }
    assert "tests/test_beta.py::test_bad" in markdown
    assert "tests/test_beta.py::test_err" in markdown
    assert "training-nogpu" in markdown and "ghcr.io/x/y:tag" in markdown
    assert "invisible" not in markdown
    assert "No failures or errors" not in markdown


@pytest.mark.parametrize("outcome", ["passed", "skipped", None])
def test_green_requires_executed_cases(outcome):
    cases = [] if outcome is None else [{"outcome": outcome}]
    markdown, stats = report.build_report(cases, "image", None)
    assert stats["executed"] == (1 if outcome == "passed" else 0)
    assert ("No failures or errors" in markdown) == (outcome == "passed")
    assert ("not passing coverage" in markdown) == (outcome != "passed")


def test_failure_table_is_bounded_without_truncating_counts():
    cases = [{"nodeid": f"tests/test_many.py::test_{i:03}", "outcome": "failed"} for i in range(105)]
    markdown, stats = report.build_report(cases, "image", None)
    assert stats["failed"] == stats["executed"] == 105
    assert markdown.count("| `tests/test_many.py::") == 100
    assert "5 more" in markdown


@pytest.mark.parametrize("content", [None, "<broken", "<not-junit/>"])
def test_unusable_results_warn_without_fabricating_coverage(monkeypatch, tmp_path, capsys, content):
    path = tmp_path / "junit.xml"
    if content is not None:
        path.write_text(content, encoding="utf-8")
    monkeypatch.setattr(sys, "argv", [str(_SCRIPT_PATH), "--junit-xml", str(path)])
    assert report.main() == 0  # The pytest step owns the job's failure status.
    output = capsys.readouterr().out
    assert "::warning::" in output and "No coverage figures" in output
    assert "::notice::" not in output and "No failures or errors" not in output


def test_cli_reports_without_repository_or_import_scans(tmp_path):
    # The real CLI must work with only the script and recorded results. There is
    # no source tree or conftest to scan/import; inherited CI paths cannot help.
    script = tmp_path / "report.py"
    script.write_bytes(_SCRIPT_PATH.read_bytes())
    junit = tmp_path / "junit.xml"
    junit.write_text(_JUNIT, encoding="utf-8")
    summary = tmp_path / "summary.md"
    summary.write_text("earlier summary\n", encoding="utf-8")
    env = {key: value for key, value in os.environ.items() if key not in {"PYTHONPATH", "GITHUB_STEP_SUMMARY"}}
    result = subprocess.run(
        [sys.executable, "-I", str(script), "--junit-xml", str(junit),
         "--image", "test-image", "--summary-file", str(summary)],
        cwd=tmp_path, env=env, capture_output=True, text=True, check=True,
    )
    assert "3 tests executed (1 passed, 1 failed, 1 errors, 1 skipped)" in result.stdout
    assert "::error::training-nogpu surface is RED" in result.stdout
    assert "invisible" not in result.stdout
    assert summary.read_text(encoding="utf-8").startswith("earlier summary\n## Training-image")
