"""Tests for the 2026-08-14 tracked-only fix to ``utils.git_metadata.get_git_metadata()``.

Investigation (this session): four real training runs on four different
commits (issue416cohort_sub, issue346q2attrib_S0, issue346q2attrib_S1,
issue285drain_K5) all reported ``report.json["git_provenance"]["git_dirty"]
== true``. The pre-fix implementation computed dirtiness via a plain
``git status --porcelain``, which flags a repo dirty for ANY untracked
file (scratch docs, half-written specs, output/ directories, or -- on
machines lacking the local ``.git/info/exclude`` entry -- accumulated
``.claude/worktrees/`` agent-worktree litter). None of that reflects an
actual code change, which is why the field carried zero bits of
information across those four runs.

The fix restricts the computation to TRACKED files only
(``git status --porcelain --untracked-files=no``) and additionally records
which tracked paths are dirty (``git_dirty_paths``). These tests build a
real, throwaway git repo per test (not this repo -- never touch the
worktree's own git state) and exercise ``get_git_metadata()`` against it
directly by chdir'ing into it, matching how ``train.py``'s ``__main__``
calls it (relies on process cwd, no explicit ``cwd=`` argument).
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from eligsir.utils.git_metadata import get_git_metadata

# Issue #555. The five tests that use the `tiny_repo` fixture below build a
# real throwaway repo with `git init`; without a `git` binary that fixture
# raises FileNotFoundError and they report as ERRORS rather than as anything
# a reader can act on. That is exactly what a full-suite run in
# the project training image produced (5 stable,
# pre-existing setup errors) -- a permanent nonzero error floor on the
# strongest surface, which trains people to subtract a magic number instead
# of reading the count.
#
# Deliberately a PER-TEST marker, not a module-level `pytestmark`:
# `test_outside_a_git_repo_degrades_to_unknown` is pure-Python, needs no git,
# and really does execute without one -- a module-level skip would have
# silently converted a genuinely-executing test into a skip.
#
# Gated on the CAPABILITY (`shutil.which("git")`), deliberately NOT on an
# image tag, container marker or env var: this skips on an image without
# git and genuinely RUNS everywhere git exists (host, `:ros2-jazzy`, the
# CPU CI image, and `:latest-gfx1151` itself once git is added to the
# Dockerfile) with no follow-up edit needed. PR #554 already established
# that tag-based surface detection silently lies. Same shape as #523's
# `requires_git` in tests/test_ablator_runner_version.py.
requires_git = pytest.mark.skipif(
    shutil.which("git") is None,
    reason=(
        "requires a `git` binary (this test `git init`s a throwaway repo); "
        "absent from the eligsir training image -- see issue #555"
    ),
)


def _run(args, cwd):
    subprocess.run(args, cwd=cwd, check=True, capture_output=True, text=True)


@pytest.fixture
def tiny_repo(tmp_path, monkeypatch):
    """A fresh, throwaway git repo with one committed tracked file, cwd'd
    into for the duration of the test."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _run(["git", "init", "-q"], cwd=repo)
    _run(["git", "config", "user.email", "test@example.com"], cwd=repo)
    _run(["git", "config", "user.name", "Test"], cwd=repo)
    tracked = repo / "tracked.py"
    tracked.write_text("x = 1\n")
    _run(["git", "add", "tracked.py"], cwd=repo)
    _run(["git", "commit", "-q", "-m", "initial"], cwd=repo)
    monkeypatch.chdir(repo)
    return repo


@requires_git
def test_clean_repo_is_not_dirty(tiny_repo):
    metadata = get_git_metadata()
    assert metadata["git_dirty"] is False
    assert metadata["git_dirty_paths"] == []
    assert metadata["git_commit"] != "unknown"
    assert metadata["git_branch"] != "unknown"


@requires_git
def test_untracked_litter_alone_does_not_mark_dirty(tiny_repo):
    """The core regression this fix targets: untracked scratch/worktree
    litter (e.g. docs/specs left in the tree, or .claude/worktrees/ agent
    checkouts) must NOT pin git_dirty to True."""
    (tiny_repo / "scratch_notes.md").write_text("some untracked scratch doc\n")
    worktree_litter = tiny_repo / ".claude" / "worktrees" / "agent-xyz" / "file.py"
    worktree_litter.parent.mkdir(parents=True)
    worktree_litter.write_text("stray agent worktree file\n")

    metadata = get_git_metadata()

    assert metadata["git_dirty"] is False, (
        "untracked-only litter must never make git_dirty True -- this is "
        "exactly the bug that made git_dirty=true on every run regardless "
        "of commit"
    )
    assert metadata["git_dirty_paths"] == []


@requires_git
def test_real_tracked_file_modification_marks_dirty(tiny_repo):
    (tiny_repo / "tracked.py").write_text("x = 2\n")

    metadata = get_git_metadata()

    assert metadata["git_dirty"] is True
    assert metadata["git_dirty_paths"] == ["tracked.py"]


@requires_git
def test_staged_tracked_change_marks_dirty(tiny_repo):
    (tiny_repo / "tracked.py").write_text("x = 3\n")
    _run(["git", "add", "tracked.py"], cwd=tiny_repo)

    metadata = get_git_metadata()

    assert metadata["git_dirty"] is True
    assert metadata["git_dirty_paths"] == ["tracked.py"]


@requires_git
def test_tracked_change_plus_untracked_litter_reports_only_tracked_path(tiny_repo):
    """Mixed case: a real tracked change alongside untracked litter must
    report dirty=True with dirty_paths naming only the real change, not
    the untracked noise."""
    (tiny_repo / "tracked.py").write_text("x = 4\n")
    (tiny_repo / "untracked_scratch.json").write_text("{}\n")

    metadata = get_git_metadata()

    assert metadata["git_dirty"] is True
    assert metadata["git_dirty_paths"] == ["tracked.py"]


def test_outside_a_git_repo_degrades_to_unknown(tmp_path, monkeypatch):
    # Issue #670 added a launcher-injected env override as the first fallback,
    # so this test must unset it to stay hermetic: docker-compose.yml's
    # `train`/`train-ros2` services and every training-launching script under
    # human_launch_scripts/*.sh (see e.g. 05_run_offline_rosbag.sh,
    # 04_run_live_training.sh) now put `-e ELIGSIR_GIT_COMMIT=...`
    # / `-e ELIGSIR_GIT_BRANCH=...` into the container command (issue
    # #891/#784), and without these two lines this test would fail for
    # entirely the wrong reason when run inside such a container.
    monkeypatch.delenv("ELIGSIR_GIT_COMMIT", raising=False)
    monkeypatch.delenv("ELIGSIR_GIT_BRANCH", raising=False)
    monkeypatch.chdir(tmp_path)
    metadata = get_git_metadata()
    assert metadata["git_branch"] == "unknown"
    assert metadata["git_commit"] == "unknown"
    assert metadata["git_dirty"] == "unknown"
    assert metadata["git_dirty_paths"] == []
