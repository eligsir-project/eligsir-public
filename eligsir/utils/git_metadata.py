"""Process-startup git provenance capture (issue #259, tracked-only fix 2026-08-14).

Split out of ``train.py`` so ``get_git_metadata()`` can be unit-tested
without pulling in ``train.py``'s full heavy import chain (``gsplat``,
``torch``, the renderer stack) -- this module has no such dependency, only
the standard library (``subprocess``/``os``/``pathlib``/``re``).

Issue #670: the subprocess capture is unreliable in containers -- the
``type: replay`` image ships no ``git`` binary at all, and git refuses a
bind-mounted checkout it considers dubiously-owned -- so a failed capture
falls back to a launcher-injected env override and then to reading the
``.git`` directory directly. See ``_recover_provenance()``.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

# Issue #670: env-override names. The launcher (ablator) already knows the
# commit -- it writes `provenance.commit` into its own `queue.jsonl` -- so
# injecting it costs one env var on the container command line and makes the
# recorded value correct-by-construction rather than dependent on whether the
# image happens to ship `git` and whether the bind-mounted checkout passes
# git's ownership check. Nothing in the runtime sets these; they exist for a
# launcher to fill in.
_ENV_COMMIT_VAR = "ELIGSIR_GIT_COMMIT"
_ENV_BRANCH_VAR = "ELIGSIR_GIT_BRANCH"

# Deliberately permissive on length (abbreviated shas are still useful) but
# strict on alphabet, so a junk/placeholder value ("", "none", "$COMMIT"
# from an unexpanded shell variable) is REFUSED rather than recorded as if
# it were provenance.
_SHA_RE = re.compile(r"^[0-9a-fA-F]{7,64}$")


def _git_command(*args: str) -> List[str]:
    """Build a git command that trusts only the checkout being inspected.

    Rootful containers commonly see a bind-mounted checkout owned by the host
    uid and Git refuses it as ``dubious ownership``.  Passing ``safe.directory``
    with ``-c`` fixes that measured #670 failure without mutating global or
    system Git configuration and without trusting any unrelated checkout.
    """
    checkout = Path.cwd().resolve()
    return ["git", "-c", f"safe.directory={checkout}", *args]


def _error_reason(exc: Exception) -> str:
    """Classify and describe a git metadata capture failure (issue #784).

    Issue #784: distinguish between known failure modes so downstream readers
    can tell them apart from ``report.json["git_provenance"]`` without
    consulting ``train.log`` (which may not be readily available).

    Classifies:
      (a) git binary absent: FileNotFoundError/OSError from subprocess.run
      (b) git ran but refused: CalledProcessError (safe.directory, bad repo, etc.)
      (c) linked-worktree gitdir unreachable: CalledProcessError with gitdir error

    Always records verbatim exception + stderr so any git version/wording
    change doesn't lose the evidence, even if classification gets misapplied.
    """
    exc_name = type(exc).__name__
    reason = f"{exc_name}: {exc}"
    git_stderr = (getattr(exc, "stderr", None) or "").strip()

    # Attempt to classify the failure mode.
    classification = "unknown"
    if isinstance(exc, (FileNotFoundError, OSError)):
        if exc_name == "FileNotFoundError":
            classification = "git_binary_missing"
        else:
            classification = "os_error"
    elif git_stderr:
        # CalledProcessError: git ran, exit nonzero, stderr has the explanation.
        if "dubious" in git_stderr.lower() and "safe.directory" in git_stderr:
            classification = "safe_directory_refusal"
        elif "not a git repository" in git_stderr.lower():
            classification = "not_a_git_repo"
        elif "bad revision" in git_stderr.lower() or "bad config value" in git_stderr.lower():
            classification = "bad_git_state"
        else:
            classification = "git_error"

    reason = f"[{classification}] {reason}"
    if git_stderr:
        reason = f"{reason} | stderr: {git_stderr}"
    return reason


def _commit_from_env() -> Optional[Tuple[str, str]]:
    """Fallback 1 (issue #670): a launcher-injected commit/branch.

    Returns ``(commit, branch)`` or ``None`` when the env var is unset or
    does not look like a sha. Rejecting junk matters more than accepting a
    near-miss: a bogus recorded commit is worse than a loud "unknown",
    because it reads as real provenance in ``report.json``.
    """
    commit = (os.environ.get(_ENV_COMMIT_VAR) or "").strip()
    if not _SHA_RE.match(commit):
        return None
    branch = (os.environ.get(_ENV_BRANCH_VAR) or "").strip() or "unknown"
    return commit, branch


def _find_git_dir(start: Path) -> Optional[Path]:
    """Locate the ``.git`` directory by walking up from ``start``.

    Handles the ``.git``-is-a-file form (``gitdir: <path>``) used by linked
    worktrees and submodules -- which is exactly what this repo's own
    ``.claude/worktrees/agent-*`` checkouts are.
    """
    for directory in [start, *start.parents]:
        candidate = directory / ".git"
        if candidate.is_dir():
            return candidate
        if candidate.is_file():
            text = candidate.read_text(errors="replace").strip()
            if not text.startswith("gitdir:"):
                return None
            pointed = Path(text.split(":", 1)[1].strip())
            if not pointed.is_absolute():
                pointed = (directory / pointed).resolve()
            return pointed if pointed.is_dir() else None
    return None


def _ref_search_dirs(gitdir: Path) -> List[Path]:
    """``gitdir`` plus, for a linked worktree, its ``commondir``.

    A linked worktree keeps its own ``HEAD`` but shares ``refs/`` and
    ``packed-refs`` with the common (main) git directory, so resolving a
    branch ref needs both.
    """
    dirs = [gitdir]
    commondir = gitdir / "commondir"
    if commondir.is_file():
        pointed = Path(commondir.read_text(errors="replace").strip())
        if not pointed.is_absolute():
            pointed = (gitdir / pointed).resolve()
        if pointed.is_dir() and pointed != gitdir:
            dirs.append(pointed)
    return dirs


def _resolve_ref(gitdir: Path, ref: str) -> Optional[str]:
    """Resolve ``refs/heads/<name>`` to a sha: loose ref first, then packed-refs."""
    for directory in _ref_search_dirs(gitdir):
        loose = directory / ref
        if loose.is_file():
            value = loose.read_text(errors="replace").strip()
            if _SHA_RE.match(value):
                return value
        packed = directory / "packed-refs"
        if packed.is_file():
            for line in packed.read_text(errors="replace").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or line.startswith("^"):
                    continue
                parts = line.split(None, 1)
                if len(parts) == 2 and parts[1].strip() == ref and _SHA_RE.match(parts[0]):
                    return parts[0]
    return None


def _provenance_from_gitdir() -> Optional[Tuple[str, str]]:
    """Fallback 2 (issue #670): read ``.git`` directly, no ``git`` binary.

    This fixes BOTH observed containerised failure modes at once -- an image
    with no ``git`` binary (rtx3090/r9700 ``type: replay``) and git's
    ``safe.directory``/dubious-ownership refusal on a bind-mounted checkout
    -- because neither a binary nor an ownership check is involved.

    Searches upward from the process cwd ONLY, deliberately: the subprocess
    path it stands in for also resolves against cwd (no ``cwd=`` argument),
    and the container runs with a checkout working directory. Using the
    module's own location instead would silently widen the scope and could
    report a *different* checkout's commit than the one git would have.

    Returns ``(commit, branch)``, with ``branch == "detached"`` for a
    detached HEAD, or ``None`` when nothing conclusive can be read. A
    branch whose ref cannot be resolved to a sha yields ``None`` rather
    than a commit-less half-answer.
    """
    gitdir = _find_git_dir(Path(os.getcwd()))
    if gitdir is None:
        return None
    head = gitdir / "HEAD"
    if not head.is_file():
        return None
    text = head.read_text(errors="replace").strip()
    if text.startswith("ref:"):
        ref = text.split(":", 1)[1].strip()
        commit = _resolve_ref(gitdir, ref)
        if commit is None:
            return None
        branch = ref[len("refs/heads/"):] if ref.startswith("refs/heads/") else ref
        return commit, (branch or "unknown")
    if _SHA_RE.match(text):
        return text, "detached"
    return None


def _recover_provenance() -> Optional[Tuple[str, str, str]]:
    """Try each issue-#670 fallback in order; ``(commit, branch, mechanism)``.

    Never raises: a fallback that itself fails must degrade to today's
    all-unknown behaviour, not turn a provenance miss into a crashed run.
    """
    for mechanism, probe in (
        ("env_override", _commit_from_env),
        ("gitdir_filesystem", _provenance_from_gitdir),
    ):
        try:
            found = probe()
        except Exception:
            found = None
        if found is not None:
            return found[0], found[1], mechanism
    return None


def get_git_metadata() -> Dict[str, Union[str, bool, List[str]]]:
    """Capture commit/branch/dirty state once at process startup.

    ``git_dirty`` (issue: dirty=true on every observed run regardless of
    commit -- see 2026-08-14 investigation) is deliberately computed against
    TRACKED files only (``git status --porcelain --untracked-files=no``).
    A plain ``git status --porcelain`` -- the previous implementation --
    flags the run dirty for ANY untracked file anywhere in the working
    tree: scratch docs, half-written specs, stray output directories, or
    (before it was excluded via this checkout's local
    ``.git/info/exclude``) the ``.claude/worktrees/`` litter from concurrent
    agent sessions. None of that reflects an actual code change to what
    gets vendored into the training container, which is why four real runs
    on four different commits (issue416cohort_sub, issue346q2attrib_S0/S1,
    issue285drain_K5) all reported ``git_dirty: true`` -- the field carried
    zero bits of information. Restricting to tracked files means the flag
    now only trips on a real modification/staged change/deletion to a
    tracked path, and ``git_dirty_paths`` records which ones so a reader
    can tell "one unrelated tracked file touched" apart from "the training
    code itself changed" without re-deriving it from the commit alone.

    This does NOT prevent drift (see issue #259) -- it only makes the
    recorded field meaningful. It also cannot retroactively repair any
    run written before this change; those runs' ``git_dirty: true`` must
    be read as "unknown", not "confirmed dirty", since the old computation
    could not distinguish real drift from untracked scratch litter.
    """
    global _DEFERRED_WARNING
    # Clear any warning retained by an EARLIER call before re-deciding below.
    # emit_deferred_git_provenance_warning() clears on emit, so a single
    # capture+emit pair is already safe -- but a capture that SUCCEEDS after a
    # previous one failed, with no emit in between, would otherwise leave the
    # stale message armed and let the next emit report a degradation that did
    # not happen. Cheap, and keeps "armed" strictly meaning "the most recent
    # capture degraded".
    _DEFERRED_WARNING = None
    metadata: Dict[str, Union[str, bool, List[str]]] = {
        "git_branch": "unknown",
        "git_commit": "unknown",
        "git_dirty": "unknown",
        "git_dirty_paths": [],
    }
    # NOT seeded here. "git_provenance_unavailable_reason" is set ONLY on the
    # failure path below, so a successful capture returns byte-identical shape
    # to pre-#784 (an always-present empty-string key would put a meaningless
    # field into every report.json and silently break every existing consumer
    # that pins this dict's shape -- tests/test_git_metadata_degradation_is_loud.py
    # ::test_successful_capture_preserves_existing_metadata_shape is exactly
    # that contract). Readers must therefore use .get(), never [].
    try:
        metadata["git_branch"] = subprocess.check_output(
            _git_command("rev-parse", "--abbrev-ref", "HEAD"),
            text=True,
            stderr=subprocess.PIPE,
        ).strip()
        metadata["git_commit"] = subprocess.check_output(
            _git_command("rev-parse", "HEAD"),
            text=True,
            stderr=subprocess.PIPE,
        ).strip()
        # --untracked-files=no: only report modified/staged/deleted TRACKED
        # paths. Untracked files (scratch docs, specs, output/, worktree
        # litter) must never be able to pin this flag to True -- see the
        # docstring above.
        status = subprocess.check_output(
            _git_command("status", "--porcelain", "--untracked-files=no"),
            text=True,
            stderr=subprocess.PIPE,
        )
        dirty_paths = [line[3:].strip() for line in status.splitlines() if line.strip()]
        metadata["git_dirty"] = bool(dirty_paths)
        metadata["git_dirty_paths"] = dirty_paths
    except Exception as exc:
        # Issue #555: this used to be a bare `except Exception: pass`, which
        # made a total provenance failure indistinguishable from a successful
        # capture -- the run just quietly recorded git_commit="unknown" and
        # nobody reading its `report.json` afterwards could tell whether that
        # meant "not a git checkout" or "the tool that answers the question
        # was not installed".
        #
        # Container images may omit `git`; launcher-provided overrides keep
        # provenance deterministic in that case.
        #
        # Two distinct failure modes land here and the message distinguishes
        # them, because they need different fixes: FileNotFoundError => the
        # binary is absent (fix the image); CalledProcessError => git ran and
        # refused (e.g. `safe.directory`/dubious-ownership on a bind-mounted
        # repo, or simply not a git checkout). Preserve git's own stderr here:
        # CalledProcessError's generic string only says exit status 128 and
        # otherwise discards the actionable `safe.directory` explanation.
        #
        # Issue #670: a failure here is no longer the end of the road. Two
        # fallbacks recover commit/branch without the git binary and without
        # git's ownership check -- measured causes of provenance being dead
        # on essentially every containerised run. They recover the IDENTITY
        # of the code that ran (which is what #259 drift auditing needs);
        # they cannot recover dirtiness, which stays "unknown" rather than
        # being guessed. The reason string names BOTH the original error and
        # the recovery mechanism, so a recovered commit can never be mistaken
        # for a full-fidelity capture.
        reason = _error_reason(exc)
        recovered = _recover_provenance()
        if recovered is not None:
            commit, branch, mechanism = recovered
            metadata["git_commit"] = commit
            metadata["git_branch"] = branch
            reason = f"{reason} | recovered_via={mechanism} (git_dirty unavailable)"
            metadata["git_provenance_unavailable_reason"] = reason
            _DEFERRED_WARNING = (
                "[git-provenance] WARNING: could not capture git provenance "
                f"({reason}). "
                f"commit/branch were RECOVERED via {mechanism} and ARE "
                "recorded, but report.json['git_provenance'] will record "
                "dirty as \"unknown\" for this run -- that is NOT evidence "
                "the checkout was clean. If this is FileNotFoundError, the "
                "`git` binary is missing from this container image "
                "(see issue #555)."
            )
        else:
            metadata["git_provenance_unavailable_reason"] = reason
            _DEFERRED_WARNING = (
                "[git-provenance] WARNING: could not capture git provenance "
                f"({reason}). "
                "report.json['git_provenance'] will record "
                "commit/branch/dirty as \"unknown\" for this run -- it is NOT "
                "evidence the checkout was clean. If this is FileNotFoundError, "
                "the `git` binary is missing from this container image "
                "(see issue #555)."
            )
        print(_DEFERRED_WARNING)
    return metadata


# Issue #670: get_git_metadata() is called at train.py:__main__ BEFORE
# start_output_log() installs the TeeStream, so the warning printed above goes
# to the container's stdout and never reaches the run's own `train.log`. That
# was measured on three real r9700 runs (issue214shadow_*) whose
# report.json['git_provenance'] recorded "unknown" while
# `grep -c git-provenance train.log` returned 0 -- i.e. the degradation was
# recorded but the diagnostic explaining WHICH failure mode caused it
# (FileNotFoundError => image lacks git; CalledProcessError => git ran and
# refused) was unavailable to anyone reading the run directory afterwards.
#
# Reordering the capture is not an option: ensure_model_path()/
# start_benchmark_log() run between it and the tee, and cfg_args ordering is
# load-bearing. So the message is retained and re-emitted once the tee exists.
_DEFERRED_WARNING = None


def emit_deferred_git_provenance_warning() -> bool:
    """Re-print the provenance warning, if any, after the log tee is installed.

    Call once, immediately after `start_output_log()`. Returns True iff a
    warning was pending (i.e. provenance capture really did degrade). Safe to
    call when nothing failed -- it is then a no-op. Idempotent: the pending
    message is cleared, so a second call prints nothing.
    """
    global _DEFERRED_WARNING
    if _DEFERRED_WARNING is None:
        return False
    print(_DEFERRED_WARNING)
    _DEFERRED_WARNING = None
    return True
