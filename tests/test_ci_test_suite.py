"""Selection is explicit; a reduced gate must not silently lose its core."""
import json
from types import SimpleNamespace
import subprocess
import sys

import pytest

from scripts import ci_test_suite as ci


@pytest.fixture
def repo(tmp_path):
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests/pr_core.txt").write_text("tests/test_core.py\n")
    (tmp_path / "tests/test_core.py").write_text("def test_core(): pass\n")
    (tmp_path / "tests/test_related.py").write_text("from pkg import worker\n")
    return tmp_path


@pytest.mark.parametrize("changed", [[], ["docs/design.md"], ["tests/README.md"]])
def test_docs_and_empty_diff_keep_core(repo, changed):
    plan = ci.select_tests(repo, changed)
    assert plan["mode"] == "pr"
    assert plan["tests"] == ["tests/test_core.py"]


def test_changed_test_is_automatically_selected(repo):
    assert ci.select_tests(repo, ["tests/test_related.py"])["tests"] == [
        "tests/test_core.py", "tests/test_related.py",
    ]


def test_new_test_is_selected_without_manifest_edit(repo):
    (repo / "tests/test_new.py").write_text("def test_new(): pass\n")
    assert "tests/test_new.py" in ci.select_tests(repo, ["tests/test_new.py"])["tests"]


def test_deleted_and_renamed_tests_do_not_make_missing_arguments(repo):
    plan = ci.select_tests(repo, ["tests/test_removed.py", "tests/test_related.py"])
    assert plan["mode"] == "pr"
    assert "tests/test_related.py" in plan["tests"]
    assert "tests/test_removed.py" not in plan["tests"]


@pytest.mark.parametrize("source", ["from pkg import worker", "from pkg.worker import run",
                                         "Path('pkg/worker.py').read_text()"])
def test_direct_imports_and_source_readers_are_selected(repo, source):
    (repo / "tests/test_related.py").write_text(source)
    assert "tests/test_related.py" in ci.select_tests(repo, ["pkg/worker.py"])["tests"]


@pytest.mark.parametrize("changed", [None, ["pkg/unrecognized.py"], ["tests/conftest.py"],
                                        ["tests/_fixture.py"], ["pyproject.toml"],
                                        ["Dockerfile.ci-sim"], ["uv.lock"]])
def test_unknown_diff_or_shared_changes_select_full_discovery(repo, changed):
    plan = ci.select_tests(repo, changed)
    assert plan["mode"] == "full"
    assert plan["tests"] == ["tests"]


def test_manual_full_stays_full(repo):
    assert ci.select_tests(repo, [], full=True)["mode"] == "full"


@pytest.mark.parametrize("contents", ["", "tests/missing.py\n",
                                         "tests/test_core.py\ntests/test_core.py\n",
                                         "tests/../test_escape.py\n", "tests/pr_core.txt\n"])
def test_invalid_core_manifest_fails_closed(repo, contents):
    (repo / "tests/pr_core.txt").write_text(contents)
    with pytest.raises(ValueError):
        ci.select_tests(repo, [])


@pytest.mark.parametrize("path", ["../escape.py", "/tmp/escape.py"])
def test_unsafe_changed_paths_are_rejected(repo, path):
    with pytest.raises(ValueError):
        ci.select_tests(repo, [path])


def test_native_exclusion_is_reported_not_called_cpu_coverage(repo):
    path = ci.NATIVE_ONLY[0]
    (repo / path).write_text("import native_backend\n")
    plan = ci.select_tests(repo, [path])
    assert path in plan["native_exclusions"]
    assert path not in plan["tests"]


def test_real_git_diff_keeps_docs_spaces_and_both_sides_of_rename(repo):
    def git(*args):
        return subprocess.run(
            ["git", "-c", "user.name=CI test", "-c", "user.email=ci@example.invalid", *args],
            cwd=repo, check=True, capture_output=True, text=True,
        ).stdout.strip()
    git("init", "-q")
    git("add", ".")
    git("commit", "-qm", "initial")
    base = git("rev-parse", "HEAD")
    (repo / "docs").mkdir()
    (repo / "docs/design.md").write_text("documentation\n")
    git("add", ".")
    git("commit", "-qm", "docs")
    assert ci.changed_paths(base, repo) == ["docs/design.md"]
    assert ci.select_tests(repo, ci.changed_paths(base, repo))["tests"] == ["tests/test_core.py"]
    (repo / "tests/test_related.py").rename(repo / "tests/test_new name.py")
    git("add", "-A")
    git("commit", "-qm", "rename")
    changed = ci.changed_paths(base, repo)
    assert changed == ["docs/design.md", "tests/test_new name.py", "tests/test_related.py"]
    assert ci.select_tests(repo, changed)["tests"] == ["tests/test_core.py", "tests/test_new name.py"]


@pytest.mark.parametrize("base", ["", "0" * 40])
def test_absent_base_is_not_empty_diff(repo, base):
    assert ci.changed_paths(base, repo) is None


@pytest.mark.parametrize("error", [
    subprocess.CalledProcessError(128, "git", stderr="fatal: dubious ownership\n" + "x" * 3000),
    FileNotFoundError("git unavailable"),
])
def test_failed_git_diff_is_visible_and_not_an_empty_diff(monkeypatch, repo, capsys, error):
    def fail(*args, **kwargs):
        raise error
    monkeypatch.setattr(subprocess, "run", fail)
    assert ci.changed_paths("missing", repo) is None
    output = capsys.readouterr()
    assert output.out == ""
    assert "Git diff unavailable; selecting full coverage:" in output.err
    assert ("dubious ownership" if isinstance(error, subprocess.CalledProcessError)
            else "git unavailable") in output.err
    assert len(output.err) < 2200


@pytest.mark.parametrize("exit_code, executed, expected", [(0, True, 0), (1, True, 1),
                                                           (2, False, 2), (5, False, 5),
                                                           (0, False, 1)])
def test_runner_preserves_failures_and_rejects_skipped_core(monkeypatch, repo, exit_code, executed, expected):
    monkeypatch.chdir(repo)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False), __version__="test",
        get_num_threads=lambda: 1,
    ))
    def run(args, plugins):
        if executed:
            plugins[0].pytest_runtest_logreport(SimpleNamespace(
                when="call", skipped=False, nodeid="tests/test_core.py::test_core",
            ))
        return exit_code
    monkeypatch.setattr(pytest, "main", run)
    assert ci.run_suite(ci.select_tests(repo, []), repo) == expected


@pytest.mark.parametrize("plan", [{"mode": "pr", "tests": []},
                                    {"mode": "pr", "tests": ["tests/test_related.py"]},
                                    {"mode": "full", "tests": ["tests/test_core.py"]},
                                    {"mode": "pr", "tests": ["tests/test_core.py", "../escape.py"]}])
def test_runner_rejects_incomplete_or_injected_plans(monkeypatch, repo, plan):
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace())
    with pytest.raises(ValueError):
        ci.run_suite(plan, repo)


@pytest.mark.parametrize("flag, mode", [("--core", "pr"), ("--full", "full")])
def test_explicit_plan_needs_no_diff_and_emits_valid_job_output(monkeypatch, repo, capsys, flag, mode):
    monkeypatch.setattr(ci, "ROOT", repo)
    monkeypatch.setattr(sys, "argv", ["ci_test_suite.py", "--plan", flag])
    monkeypatch.setenv("CI_TEST_PLAN", "invalid inherited plan must not be parsed")
    output = repo / "github-output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    def no_diff(*args):
        pytest.fail("Explicit full/core plans do not need Git")
    monkeypatch.setattr(ci, "changed_paths", no_diff)
    assert ci.main() == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["mode"] == mode and plan["changed"] == []
    assert output.read_text() == "plan=" + json.dumps(plan, separators=(",", ":")) + "\n"


@pytest.mark.parametrize("path", [
    "scripts/ci_test_suite.py", "tests/pr_core.txt", ".github/workflows/ci-cpu.yml",
    "pytest.ini", ".pytest.ini", "setup.cfg", "tox.ini", ".dockerignore",
    "pyproject.toml", "uv.lock", "Dockerfile.ci-sim", "conftest.py",
    "tests/subdir/conftest.py", "tests/_fixture.py",
])
def test_gate_surfaces_force_full_even_with_direct_references(repo, path):
    (repo / "tests/test_related.py").write_text(f"# Reads {path}\n")
    plan = ci.select_tests(repo, [path])
    assert plan["mode"] == "full"
    assert plan["tests"] == ["tests"]
    assert plan["reason"] == f"shared test/dependency surface: {path}"


@pytest.mark.parametrize("argv, expected_mode, expected_changed", [
    ([], "pr", ["tests/test_related.py"]),
    (["--full"], "full", []),
    (["--core"], "pr", []),
    (["--base", "abc"], "pr", ["tests/test_related.py"]),
    (["--base", ""], "full", None),
])
def test_explicit_cli_overrides_inherited_plan(monkeypatch, repo, argv, expected_mode, expected_changed):
    inherited = ci.select_tests(repo, ["tests/test_related.py"])
    inherited["reason"] = "upstream plan"
    monkeypatch.setenv("CI_TEST_PLAN", json.dumps(inherited))
    monkeypatch.setattr(ci, "ROOT", repo)
    monkeypatch.setattr(sys, "argv", ["ci_test_suite.py", *argv])
    def diff(base):
        assert argv == ["--base", base]
        return ["tests/test_related.py"] if base else None
    monkeypatch.setattr(ci, "changed_paths", diff)
    def run(plan):
        assert plan["mode"] == expected_mode
        assert plan["changed"] == expected_changed
        assert (plan["reason"] == "upstream plan") == (not argv)
        return 17
    monkeypatch.setattr(ci, "run_suite", run)
    assert ci.main() == 17
