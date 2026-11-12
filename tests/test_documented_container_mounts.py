"""No documented `docker compose run` may bind-mount into `/workspace/` (issue #1106).

`docker-compose.yml`'s `train` service already binds `./:/workspace/eligsir`
and sets that as `working_dir`. Adding `-v <host>:/workspace/<anything>` to a
`docker compose run` invocation therefore has exactly two possible outcomes,
and neither is ever what the author wanted:

* target **is** `/workspace/eligsir` -> podman refuses the container with
  `duplicate mount destination`; docker silently resolves one of the two.
* target is **anything else** under `/workspace/` (e.g. the historical
  `/workspace/3dgs-mcmc`) -> nothing reads that path. `working_dir` is still
  `/workspace/eligsir`, so the container runs whatever `./` resolves to and
  **the flag is a silent no-op**.

The second case is why this test exists. `CLAUDE.md` documented
`-v "$PWD":/workspace/3dgs-mcmc` for a long time, and an agent copying that
pattern to run a *different* tree as a test baseline got a byte-identical,
perfectly green "no regressions" result -- because both arms of the comparison
ran the branch. It hid a real regression and was caught only by accident. The
failure mode is silent and it fails GREEN, so a documentation fix alone is not
a sufficient guard; see `scripts/compare_baseline.sh` for the supported way to
run two trees against each other.

The rule checked here is deliberately narrow, to keep false positives near
zero: it applies only to `docker compose run` invocations (which get the bind
and `working_dir` from the compose file), and only to mount targets under
`/workspace/`. A raw `podman run`/`docker run` of the image is unaffected --
it has no compose bind, so mounting the tree is both necessary and correct
there, as long as it targets the image's real `WORKDIR`.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

# Historical session records: plans, reports and investigation write-ups that
# transcribe commands as they were actually run at the time. They are an
# archive, not a recipe anyone should copy, and rewriting them would falsify
# the record. Everything an author might realistically copy from is checked.
_EXCLUDED_PREFIXES = ("docs/superpowers/",)

_SCANNED_SUFFIXES = (".md", ".sh")

# Only consulted by the `git ls-files`-less fallback below, which is NOT a rare
# edge case -- it is the normal path in a git worktree (`.git` is a file
# pointing outside the container bind), in `git archive` exports, and in
# `Dockerfile.ci-cpu` (which COPYs files, not the repo with `.git`). Run
# artifacts must be excluded there: `output/` holds hundreds of run dirs, and
# ablator writes a generated `reproduce.sh` beside each one's queue provenance.
# Linting a generated artifact would put the guard red on something no one can
# fix by editing docs -- and the likely response to that is to weaken the
# guard, not to fix anything.
_WALK_SKIP_DIRS = frozenset(
    {".git", ".claude", "node_modules", ".venv", "output", "data", "submodules"}
)

# `-v host:/workspace/...` or `--volume host:/workspace/...`, with the target
# optionally carrying mount options (`:Z`, `:ro`, ...).
_WORKSPACE_MOUNT = re.compile(
    r"(?:-v|--volume)[=\s]+\S*?:(?P<target>/workspace/[^\s:\\]*)"
)


def _candidate_relpaths() -> list[str]:
    """Docs/scripts to scan, as repo-relative POSIX paths.

    Prefers `git ls-files` so untracked scratch files are ignored, but MUST
    degrade to a filesystem walk rather than raising: `scripts/compare_baseline.sh`
    runs this suite inside `git archive` exports, which have no `.git` at all.
    Raising here would abort collection of the WHOLE suite (pytest treats a
    collection error as fatal) in exactly the comparison runs this test exists
    to protect.
    """
    rels: list[str] = []
    try:
        out = subprocess.run(
            ["git", "ls-files", "-z"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
            timeout=60,
        ).stdout
        rels = [r for r in out.split("\0") if r]
    except (OSError, subprocess.SubprocessError):
        rels = []

    if not rels:  # git-less export, or git failed for any reason
        for suffix in _SCANNED_SUFFIXES:
            for path in REPO_ROOT.rglob(f"*{suffix}"):
                if any(part in _WALK_SKIP_DIRS for part in path.parts):
                    continue
                rels.append(path.relative_to(REPO_ROOT).as_posix())

    return sorted(
        r
        for r in rels
        if r.endswith(_SCANNED_SUFFIXES) and not r.startswith(_EXCLUDED_PREFIXES)
    )


def _compose_run_commands(text: str) -> list[str]:
    """Every `docker compose run` invocation, with line continuations joined.

    A command is taken to end at the first line that does not end in a
    backslash, which is how all of these are written in this repo.
    """
    commands = []
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if "docker compose run" not in line and "docker-compose run" not in line:
            continue
        parts = [line]
        j = i
        while parts[-1].rstrip().endswith("\\") and j + 1 < len(lines):
            j += 1
            parts.append(lines[j])
        commands.append(" ".join(p.rstrip().rstrip("\\") for p in parts))
    return commands


def test_compose_service_binds_workspace_eligsir():
    """The premise the rest of this module rests on, asserted rather than assumed."""
    compose = (REPO_ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    assert "./:/workspace/eligsir" in compose
    assert "working_dir: /workspace/eligsir" in compose


@pytest.mark.parametrize("rel_path", _candidate_relpaths())
def test_no_documented_compose_run_mounts_into_workspace(rel_path):
    path = REPO_ROOT / rel_path
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        pytest.skip(f"unreadable: {rel_path}")

    offenders = []
    for command in _compose_run_commands(text):
        for match in _WORKSPACE_MOUNT.finditer(command):
            offenders.append((match.group("target"), command.strip()))

    assert not offenders, (
        f"{rel_path} documents a `docker compose run` that bind-mounts into "
        f"/workspace/ (issue #1106).\n\n"
        + "\n\n".join(
            f"  mount target: {target}\n  in: {command[:400]}"
            for target, command in offenders
        )
        + "\n\nThe compose `train` service already binds ./:/workspace/eligsir "
        "and sets it as working_dir, so this mount is either a duplicate "
        "destination or -- worse -- a SILENT NO-OP that leaves the container "
        "running ./ anyway.\n"
        "  * ordinary in-repo command: delete the -v, the tree is already there.\n"
        "  * comparing two git trees:  use scripts/compare_baseline.sh, or mount "
        "outside /workspace (e.g. -v /tmp/tree:/tree) and `cd` into it."
    )
