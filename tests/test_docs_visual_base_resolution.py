"""Regression tests for the docs visual-coverage rule's comparison-base resolution.

PR #721 added `scripts/docs_visuals.py check` to `.github/workflows/docs.yml`.
Its changed-doc rule shells out to
``git diff origin/$GITHUB_BASE_REF...HEAD`` and is deliberately fail-closed
when that cannot be resolved. In CI it could never be resolved: the `build-docs`
job runs in a `python:3.12` container as root while the checked-out workspace is
owned by the runner user, so git aborts with "detected dubious ownership in
repository" -- and the original code sent git's stderr to ``DEVNULL``, so the
only symptom was the generic "could not resolve docs visual comparison base
'main'". The check therefore failed on **every** pull request, including PR
#721's own run 31882358027.

Two guards, both torch-free so they run on the CPU-only CI image:

* ``test_workspace_is_marked_safe_before_the_visuals_check`` -- the workflow
  must mark the workspace a safe git directory, and must do so *before* the
  step that runs ``docs_visuals.py``. This is the actual fix; without it every
  git call from every later step in that job fails the same way.
* ``test_unresolvable_base_reports_gits_own_reason`` -- the fail-closed error
  must carry git's stderr, so the next environment-level breakage is
  distinguishable from a genuinely missing ref instead of presenting
  identically. This test is self-contained (issue #873): it creates a
  throwaway repo in a temp directory with a proper .git directory and
  exercises git ref resolution directly. This approach works regardless of
  whether the real eligsir repo is a git worktree (which would have a .git
  pointer file that doesn't work when bind-mounted into a container) or a
  shared checkout.
"""

import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "docs.yml"
DOCS_VISUALS = REPO_ROOT / "scripts" / "docs_visuals.py"


def _require(path: Path, what: str) -> None:
    """Skip (never fail) when this surface's image does not ship ``path``.

    `Dockerfile.ci-cpu` COPYs a deliberate subset of the tree: `scripts/`,
    `tests/`, the packages, and a few named files -- but NOT `.github/`,
    `mkdocs.yml` or `docs/`. So on the `cpu-ci` surface these inputs are
    genuinely absent, and asserting on them there measures the image's COPY
    list, not the workflow. Skipping with a reason is the honest result;
    the checks still run on `host`, `training-*` and `ros2-*`, which do have
    the whole tree bind-mounted. (Per docs/architecture/
    2026-08-14-issue552-test-surface-contract.md: a green result proves
    something only about the surface it ran on.)
    """
    if not path.exists():
        pytest.skip(f"{what} not present on this test surface ({path} missing)")


def test_workspace_is_marked_safe_before_the_visuals_check() -> None:
    _require(DOCS_WORKFLOW, ".github/workflows/docs.yml")
    text = DOCS_WORKFLOW.read_text(encoding="utf-8")

    safe_dir = re.search(r"safe\.directory", text)
    assert safe_dir is not None, (
        "docs.yml must mark the checked-out workspace as a safe git directory. "
        "The build-docs job runs as root in a container over a runner-owned "
        "workspace, so without it every git invocation in that job fails with "
        "'detected dubious ownership in repository' -- which is what made the "
        "visuals check fail on every PR (run 31882358027)."
    )

    # Match the executable step, not a mention in a comment.
    visuals_step = re.search(r"^\s*run:.*docs_visuals\.py", text, re.MULTILINE)
    assert visuals_step is not None, "docs.yml no longer runs scripts/docs_visuals.py"
    assert safe_dir.start() < visuals_step.start(), (
        "the safe.directory configuration must come BEFORE the step that runs "
        "docs_visuals.py, otherwise the check still cannot resolve its "
        "comparison base."
    )


def test_unresolvable_base_reports_gits_own_reason(tmp_path: Path) -> None:
    """A genuinely bad ref must surface git's stderr, not just 'could not resolve'."""
    # Check that git is available on PATH. Skip if not.
    try:
        subprocess.run(
            ["git", "--version"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("git is not available on PATH")

    # Create a self-contained temporary git repo to exercise git ref resolution
    # directly, independent of whether the real eligsir repo is a worktree
    # (issue #873).
    test_repo = tmp_path / "test_repo"
    test_repo.mkdir()

    # Initialize git repo with minimal config so we can commit.
    # Use `-c` flags to set git config locally without requiring global config.
    subprocess.run(
        [
            "git",
            "-c",
            "user.email=test@example.com",
            "-c",
            "user.name=Test User",
            "init",
        ],
        cwd=str(test_repo),
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Create the minimal directory structure that docs_visuals.py needs to
    # recognize a repo root: mkdocs.yml, docs/ directory, and scripts/ with a
    # copy of docs_visuals.py. We copy the script into the test repo so that
    # when it calls repo_root_from(), it finds the test repo's structure, not
    # the real eligsir repo.
    (test_repo / "mkdocs.yml").touch()
    docs_dir = test_repo / "docs"
    docs_dir.mkdir()
    # Create minimal visual spec directories and files so docs_visuals.py doesn't fail
    # trying to hash spec files that don't exist.
    visuals_dir = docs_dir / "visuals"
    visuals_dir.mkdir()
    (visuals_dir / "diagrams.json").write_text('{"diagrams": []}')
    (visuals_dir / "plots.json").write_text('{"plots": []}')
    (visuals_dir / "assets.json").write_text('{"assets": []}')
    scripts_dir = test_repo / "scripts"
    scripts_dir.mkdir()
    (scripts_dir / "docs_visuals.py").write_text(DOCS_VISUALS.read_text(encoding="utf-8"))

    # Make a commit so there's some history for git diff to work with.
    subprocess.run(
        [
            "git",
            "-c",
            "user.email=test@example.com",
            "-c",
            "user.name=Test User",
            "add",
            "-A",
        ],
        cwd=str(test_repo),
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    subprocess.run(
        [
            "git",
            "-c",
            "user.email=test@example.com",
            "-c",
            "user.name=Test User",
            "commit",
            "-m",
            "initial commit",
        ],
        cwd=str(test_repo),
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Run the copied docs_visuals.py against the test repo with a bad ref.
    test_script = scripts_dir / "docs_visuals.py"
    proc = subprocess.run(
        [sys.executable, str(test_script), "check"],
        cwd=str(test_repo),
        env={
            "PATH": "/usr/bin:/bin:/usr/local/bin",
            "HOME": str(tmp_path),
            "DOCS_VISUAL_BASE": "definitely-not-a-real-ref-for-tests",
        },
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    assert proc.returncode != 0, "an unresolvable comparison base must fail closed"
    assert "could not resolve docs visual comparison base" in proc.stdout
    assert "tried:" in proc.stdout, (
        "the failure must name what was attempted and why; swallowing git's "
        "stderr is what hid the real cause in CI"
    )
    assert "bad revision" in proc.stdout or "unknown revision" in proc.stdout, (
        f"git's own reason must be included; got: {proc.stdout!r}"
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
