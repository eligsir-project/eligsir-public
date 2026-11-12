"""Issue #555/#670/#784: git-provenance degradation must be loud, diagnostic,
and -- since #670 -- recovered where it can be.

These tests stay entirely on ``utils.git_metadata`` so the correctness guard
runs on the required torch-free CPU-CI surface. The diagnostic reason is
retained in the warning that #674 re-emits into ``train.log``; the existing
metadata dict shape on a SUCCESSFUL capture remains unchanged so this does not
create an ad-hoc runtime/cfg_args schema field.

Issue #670 changed what a subprocess failure means. It used to imply all four
fields degrade to "unknown"; that was measured to be dead on essentially every
containerised run (the ``type: replay`` image ships no ``git`` binary at all,
on both rtx3090 and r9700). ``get_git_metadata()`` now falls back to a
launcher-injected env override and then to reading ``.git`` off the
filesystem. So the tests below split into two groups:

  * NO fallback available (env unset AND cwd not under any git checkout) --
    the unchanged all-unknown + loud-warning contract, verbatim.
  * a fallback available -- commit/branch recovered, ``git_dirty`` still
    "unknown" (it genuinely cannot be computed without git), and the reason
    string naming both the original error and the recovery mechanism.
"""

from __future__ import annotations

import subprocess

import pytest

from eligsir.utils import git_metadata
from eligsir.utils.git_metadata import get_git_metadata

SHA = "0123456789abcdef0123456789abcdef01234567"


def _force_failure(monkeypatch, exc):
    def _raise(*_args, **_kwargs):
        raise exc

    monkeypatch.setattr(git_metadata.subprocess, "check_output", _raise)


@pytest.fixture
def no_fallbacks(monkeypatch, tmp_path):
    """Disable BOTH #670 fallbacks: no env override, and a cwd with no
    ``.git`` anywhere above it. Without this the tests below would recover
    this repo's own commit off the filesystem, which is exactly the new
    behaviour -- so the total-failure contract has to be tested explicitly."""
    monkeypatch.delenv(git_metadata._ENV_COMMIT_VAR, raising=False)
    monkeypatch.delenv(git_metadata._ENV_BRANCH_VAR, raising=False)
    empty = tmp_path / "not_a_checkout"
    empty.mkdir()
    monkeypatch.chdir(empty)
    return empty


@pytest.fixture
def env_override(monkeypatch, tmp_path):
    """Fallback 1 available: launcher-injected commit/branch."""
    monkeypatch.setenv(git_metadata._ENV_COMMIT_VAR, SHA)
    monkeypatch.setenv(git_metadata._ENV_BRANCH_VAR, "main")
    empty = tmp_path / "not_a_checkout"
    empty.mkdir()
    monkeypatch.chdir(empty)
    return empty


def _make_fake_gitdir(root, *, head, refs=None, packed=None, as_file=False, commondir=None):
    """Build a minimal, hand-written ``.git`` -- no ``git`` binary involved,
    which is the whole point of the fallback under test."""
    work = root / "work"
    work.mkdir(parents=True, exist_ok=True)
    gitdir = (root / "real_gitdir") if as_file else (work / ".git")
    gitdir.mkdir(parents=True, exist_ok=True)
    (gitdir / "HEAD").write_text(head)
    for ref, sha in (refs or {}).items():
        ref_path = gitdir / ref
        ref_path.parent.mkdir(parents=True, exist_ok=True)
        ref_path.write_text(sha + "\n")
    if packed is not None:
        (gitdir / "packed-refs").write_text(packed)
    if commondir is not None:
        (gitdir / "commondir").write_text(str(commondir) + "\n")
    if as_file:
        (work / ".git").write_text(f"gitdir: {gitdir}\n")
    return work


# ---------------------------------------------------------------------------
# Group 1: no fallback available -- today's behaviour, unchanged.
# ---------------------------------------------------------------------------


def test_missing_git_binary_still_returns_unknown_fields(monkeypatch, no_fallbacks):
    _force_failure(monkeypatch, FileNotFoundError(2, "No such file or directory", "git"))

    meta = get_git_metadata()

    # Issue #784 added ONE key on the failure path only: the classified reason.
    # The four pre-#784 fields must still degrade to exactly these values when
    # no #670 fallback can supply anything -- that is what this test has always
    # guarded and neither #784 nor #670 relaxes it.
    assert {k: v for k, v in meta.items() if k != "git_provenance_unavailable_reason"} == {
        "git_branch": "unknown",
        "git_commit": "unknown",
        "git_dirty": "unknown",
        "git_dirty_paths": [],
    }
    # And the reason must actually be populated and classified (#784's point):
    # a present-but-empty reason would be the bare "unknown" #784 exists to kill.
    assert meta["git_provenance_unavailable_reason"].startswith("[git_binary_missing]")
    assert "FileNotFoundError" in meta["git_provenance_unavailable_reason"]
    # No recovery happened, so the reason must NOT claim one.
    assert "recovered_via" not in meta["git_provenance_unavailable_reason"]


def test_missing_git_binary_warns_loudly(monkeypatch, capsys, no_fallbacks):
    _force_failure(monkeypatch, FileNotFoundError(2, "No such file or directory", "git"))

    get_git_metadata()

    out = capsys.readouterr().out
    assert "[git-provenance] WARNING" in out
    assert "FileNotFoundError" in out


def test_git_refusing_also_warns_and_names_its_own_error(monkeypatch, capsys, no_fallbacks):
    _force_failure(
        monkeypatch, subprocess.CalledProcessError(128, ["git", "rev-parse", "HEAD"])
    )

    meta = get_git_metadata()
    out = capsys.readouterr().out

    assert meta["git_commit"] == "unknown"
    assert "[git-provenance] WARNING" in out
    assert "CalledProcessError" in out


def test_warning_says_unknown_is_not_evidence_of_a_clean_checkout(
    monkeypatch, capsys, no_fallbacks
):
    _force_failure(monkeypatch, FileNotFoundError(2, "No such file or directory", "git"))

    get_git_metadata()

    assert "NOT evidence" in capsys.readouterr().out


def test_called_process_error_reports_gits_own_stderr(monkeypatch, capsys, no_fallbacks):
    exc = subprocess.CalledProcessError(128, ["git", "rev-parse", "HEAD"])
    exc.stderr = (
        "fatal: detected dubious ownership in repository at "
        "'/workspace/eligsir'\n"
    )
    _force_failure(monkeypatch, exc)

    get_git_metadata()
    out = capsys.readouterr().out

    assert "CalledProcessError" in out
    assert "dubious ownership" in out
    assert "safe.directory" not in out  # only report what git actually emitted


# ---------------------------------------------------------------------------
# Group 2: issue #670 -- fallback 1, launcher-injected env override.
# ---------------------------------------------------------------------------


def test_env_override_recovers_commit_and_branch(monkeypatch, env_override):
    _force_failure(monkeypatch, FileNotFoundError(2, "No such file or directory", "git"))

    meta = get_git_metadata()

    assert meta["git_commit"] == SHA
    assert meta["git_branch"] == "main"
    # Dirtiness genuinely cannot be computed without git. It must stay unknown,
    # never be guessed to False just because a commit was recovered.
    assert meta["git_dirty"] == "unknown"
    assert meta["git_dirty_paths"] == []


def test_env_recovery_reason_names_both_the_error_and_the_mechanism(
    monkeypatch, env_override
):
    _force_failure(monkeypatch, FileNotFoundError(2, "No such file or directory", "git"))

    reason = get_git_metadata()["git_provenance_unavailable_reason"]

    assert reason.startswith("[git_binary_missing]")  # original error, verbatim
    assert "FileNotFoundError" in reason
    assert "recovered_via=env_override" in reason
    assert "git_dirty unavailable" in reason


def test_env_recovery_still_warns_loudly(monkeypatch, capsys, env_override):
    _force_failure(monkeypatch, FileNotFoundError(2, "No such file or directory", "git"))

    get_git_metadata()
    out = capsys.readouterr().out

    assert "[git-provenance] WARNING" in out
    assert "RECOVERED" in out
    # A reader must never take a recovered commit as a full-fidelity capture.
    assert "NOT evidence" in out
    assert git_metadata.emit_deferred_git_provenance_warning() is True


def test_env_override_without_a_branch_records_unknown_branch(monkeypatch, tmp_path):
    monkeypatch.setenv(git_metadata._ENV_COMMIT_VAR, SHA)
    monkeypatch.delenv(git_metadata._ENV_BRANCH_VAR, raising=False)
    monkeypatch.chdir(tmp_path)
    _force_failure(monkeypatch, FileNotFoundError(2, "No such file or directory", "git"))

    meta = get_git_metadata()

    assert meta["git_commit"] == SHA
    assert meta["git_branch"] == "unknown"


@pytest.mark.parametrize("junk", ["", "   ", "none", "$COMMIT", "not-a-sha", "abc"])
def test_junk_env_override_is_refused_not_recorded(monkeypatch, no_fallbacks, junk):
    """A bogus recorded commit is worse than a loud "unknown" -- it reads as
    real provenance in report.json."""
    monkeypatch.setenv(git_metadata._ENV_COMMIT_VAR, junk)
    _force_failure(monkeypatch, FileNotFoundError(2, "No such file or directory", "git"))

    meta = get_git_metadata()

    assert meta["git_commit"] == "unknown"
    assert "recovered_via" not in meta["git_provenance_unavailable_reason"]


# ---------------------------------------------------------------------------
# Group 2b: issue #670 -- fallback 2, reading .git off the filesystem.
# ---------------------------------------------------------------------------


def _gitdir_case(monkeypatch, work):
    monkeypatch.delenv(git_metadata._ENV_COMMIT_VAR, raising=False)
    monkeypatch.delenv(git_metadata._ENV_BRANCH_VAR, raising=False)
    monkeypatch.chdir(work)
    _force_failure(monkeypatch, FileNotFoundError(2, "No such file or directory", "git"))
    return get_git_metadata()


def test_gitdir_loose_ref_recovers_commit_and_branch(monkeypatch, tmp_path):
    work = _make_fake_gitdir(
        tmp_path,
        head="ref: refs/heads/feature-x\n",
        refs={"refs/heads/feature-x": SHA},
    )

    meta = _gitdir_case(monkeypatch, work)

    assert meta["git_commit"] == SHA
    assert meta["git_branch"] == "feature-x"
    assert meta["git_dirty"] == "unknown"
    assert meta["git_dirty_paths"] == []
    reason = meta["git_provenance_unavailable_reason"]
    assert reason.startswith("[git_binary_missing]")
    assert "recovered_via=gitdir_filesystem" in reason
    assert "git_dirty unavailable" in reason


def test_gitdir_search_walks_up_from_a_subdirectory(monkeypatch, tmp_path):
    work = _make_fake_gitdir(
        tmp_path, head="ref: refs/heads/main\n", refs={"refs/heads/main": SHA}
    )
    nested = work / "a" / "b" / "c"
    nested.mkdir(parents=True)

    meta = _gitdir_case(monkeypatch, nested)

    assert meta["git_commit"] == SHA
    assert meta["git_branch"] == "main"


def test_gitdir_packed_refs_only_recovers_commit(monkeypatch, tmp_path):
    """A freshly-cloned checkout has no loose ref for its branch at all."""
    work = _make_fake_gitdir(
        tmp_path,
        head="ref: refs/heads/main\n",
        packed=(
            "# pack-refs with: peeled fully-peeled sorted \n"
            f"{SHA} refs/heads/main\n"
            "1111111111111111111111111111111111111111 refs/tags/v0.1.0\n"
            "^2222222222222222222222222222222222222222\n"
        ),
    )

    meta = _gitdir_case(monkeypatch, work)

    assert meta["git_commit"] == SHA
    assert meta["git_branch"] == "main"


def test_gitdir_detached_head_records_the_sha_and_detached_branch(monkeypatch, tmp_path):
    work = _make_fake_gitdir(tmp_path, head=SHA + "\n")

    meta = _gitdir_case(monkeypatch, work)

    assert meta["git_commit"] == SHA
    assert meta["git_branch"] == "detached"


def test_gitdir_is_a_file_linked_worktree_is_followed(monkeypatch, tmp_path):
    """``.git`` is a file (``gitdir: ...``) in every linked worktree -- which
    is what this repo's own ``.claude/worktrees/agent-*`` checkouts are."""
    work = _make_fake_gitdir(
        tmp_path,
        head="ref: refs/heads/agent-branch\n",
        refs={"refs/heads/agent-branch": SHA},
        as_file=True,
    )

    meta = _gitdir_case(monkeypatch, work)

    assert meta["git_commit"] == SHA
    assert meta["git_branch"] == "agent-branch"


def test_linked_worktree_resolves_refs_via_commondir(monkeypatch, tmp_path):
    """A real linked worktree keeps its own HEAD but shares ``refs/`` with the
    common git dir, so the ref is NOT resolvable from the worktree gitdir."""
    common = tmp_path / "common_gitdir"
    (common / "refs" / "heads").mkdir(parents=True)
    (common / "refs" / "heads" / "shared-branch").write_text(SHA + "\n")

    work = _make_fake_gitdir(
        tmp_path,
        head="ref: refs/heads/shared-branch\n",
        as_file=True,
        commondir=common,
    )

    meta = _gitdir_case(monkeypatch, work)

    assert meta["git_commit"] == SHA
    assert meta["git_branch"] == "shared-branch"


def test_unresolvable_ref_does_not_fabricate_a_commit(monkeypatch, tmp_path):
    """HEAD names a branch whose ref exists nowhere: a branch name alone is
    not provenance, so this must degrade rather than half-answer."""
    work = _make_fake_gitdir(tmp_path, head="ref: refs/heads/ghost\n")

    meta = _gitdir_case(monkeypatch, work)

    assert meta["git_commit"] == "unknown"
    assert meta["git_branch"] == "unknown"
    assert "recovered_via" not in meta["git_provenance_unavailable_reason"]


def test_env_override_wins_over_the_filesystem(monkeypatch, tmp_path):
    """The launcher knows which commit it dispatched; the bind-mounted
    checkout may have moved on since (issue #259)."""
    other = "fedcba9876543210fedcba9876543210fedcba98"
    work = _make_fake_gitdir(
        tmp_path, head="ref: refs/heads/main\n", refs={"refs/heads/main": other}
    )
    monkeypatch.setenv(git_metadata._ENV_COMMIT_VAR, SHA)
    monkeypatch.setenv(git_metadata._ENV_BRANCH_VAR, "dispatched")
    monkeypatch.chdir(work)
    _force_failure(monkeypatch, FileNotFoundError(2, "No such file or directory", "git"))

    meta = get_git_metadata()

    assert meta["git_commit"] == SHA
    assert meta["git_branch"] == "dispatched"
    assert "recovered_via=env_override" in meta["git_provenance_unavailable_reason"]


# ---------------------------------------------------------------------------
# Success path: unchanged, and must stay unchanged.
# ---------------------------------------------------------------------------


def test_successful_capture_preserves_existing_metadata_shape(monkeypatch):
    outputs = {
        ("git", "rev-parse", "--abbrev-ref", "HEAD"): "main\n",
        ("git", "rev-parse", "HEAD"): "abc123\n",
        ("git", "status", "--porcelain", "--untracked-files=no"): "",
    }

    def fake_check_output(cmd, **_kw):
        # #1180 adds a checkout-scoped `git -c safe.directory=...` prefix.
        # This test owns only the successful metadata shape, not argv details;
        # the dedicated #670 test asserts the new scoped command explicitly.
        if len(cmd) >= 3 and cmd[:2] == ["git", "-c"] and cmd[2].startswith("safe.directory="):
            cmd = ["git", *cmd[3:]]
        return outputs[tuple(cmd)]

    monkeypatch.setattr(
        git_metadata.subprocess,
        "check_output",
        fake_check_output,
    )

    assert get_git_metadata() == {
        "git_branch": "main",
        "git_commit": "abc123",
        "git_dirty": False,
        "git_dirty_paths": [],
    }
