"""Fail-closed contract for ``scripts/compare_baseline.sh`` (issue #1106)."""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
from pathlib import Path


def _run(*args: str, cwd: Path) -> None:
    subprocess.run(args, cwd=cwd, check=True, capture_output=True, text=True)


def test_compare_baseline_refuses_unparsed_pytest_failures(tmp_path: Path) -> None:
    """A failing pytest summary must not be accepted as an empty failure set."""
    repo = tmp_path / "repo"
    scripts = repo / "scripts"
    fake_bin = tmp_path / "bin"
    scripts.mkdir(parents=True)
    fake_bin.mkdir()
    shutil.copy2(
        Path(__file__).resolve().parents[1] / "scripts" / "compare_baseline.sh",
        scripts / "compare_baseline.sh",
    )

    _run("git", "init", "-q", cwd=repo)
    _run("git", "config", "user.name", "Test", cwd=repo)
    _run("git", "config", "user.email", "test@example.invalid", cwd=repo)
    (repo / "payload.txt").write_text("baseline\n", encoding="utf-8")
    _run("git", "add", ".", cwd=repo)
    _run("git", "commit", "-qm", "baseline", cwd=repo)
    (repo / "payload.txt").write_text("current\n", encoding="utf-8")
    _run("git", "commit", "-qam", "current", cwd=repo)

    fake_docker = fake_bin / "docker"
    fake_docker.write_text(
        """#!/usr/bin/env bash
set -eu
case "$*" in
  *--collect-only*)
    echo 'tests/test_sample.py::test_sample'
    echo '1 test collected in 0.01s'
    exit 0
    ;;
  */baseline:/tree*) echo 'tree-identity: baseline' ;;
  *) echo 'tree-identity: current' ;;
esac
# Deliberately omit a `FAILED ...` detail line while retaining pytest's own
# authoritative terminal summary. The comparison must fail closed.
echo '1 failed in 0.01s'
exit 1
""",
        encoding="utf-8",
    )
    fake_docker.chmod(fake_docker.stat().st_mode | stat.S_IXUSR)

    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}:{env['PATH']}"
    result = subprocess.run(
        [
            "bash",
            str(scripts / "compare_baseline.sh"),
            "HEAD~1",
            "HEAD",
            "--",
            "tests/test_sample.py",
        ],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0, result.stdout
    assert "failure summary does not match parsed outcomes" in result.stderr
