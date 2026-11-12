"""Regression tests for scripts/lineage_coverage_audit.py (issue #432).

The pre-fix script scanned exactly one level deep from its output_root, so a
run directory nested in a subdirectory of the root (e.g. output/scratch/<run>)
silently collapsed into a single ``no_checkpoint`` row -- producing a
confident FALSE NEGATIVE (#411's "0 of 174 trustworthy" while 9 trustworthy
runs already existed under output/scratch/, see issue #416 and PR #429).

These tests build synthetic directory trees (no torch, no real checkpoints --
only cfg_args/train.log markers) and assert the fixed script:
  1. discovers run dirs at depth 1
  2. discovers run dirs at depth 2+ (the exact #416/#432 regression case)
  3. discovers run dirs under a container-of-containers root
  4. does not recurse into a run dir's own artifact subdirectories
     (comparison/iter_N/ carries a report.json but is not a run dir)
  5. fails loudly (raises / nonzero exit) on an empty root, rather than
     silently reporting a clean "0 found" bill of health

All classification here uses no chkpnt*.pth, so every discovered run lands in
``no_checkpoint`` and torch is never imported -- these tests are CPU/host
runnable with no container needed.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import pytest

from lineage_coverage_audit import audit, discover_run_dirs


def _mk_run_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "cfg_args").write_text("Namespace()\n")
    (path / "train.log").write_text("[RUN START]\n")


def test_discovers_run_dir_at_depth_1(tmp_path: Path):
    _mk_run_dir(tmp_path / "runA")
    run_dirs, containers, skipped = discover_run_dirs(str(tmp_path))
    assert run_dirs == ["runA"]
    assert skipped == []


def test_discovers_run_dir_at_depth_2_plus(tmp_path: Path):
    """The exact #416/#432 regression: output/scratch/<run> must be found."""
    _mk_run_dir(tmp_path / "scratch" / "run_deep")
    _mk_run_dir(tmp_path / "scratch" / "nested" / "run_deeper")
    run_dirs, containers, skipped = discover_run_dirs(str(tmp_path))
    assert "scratch/run_deep" in run_dirs
    assert "scratch/nested/run_deeper" in run_dirs
    # scratch/ itself and scratch/nested/ are containers, not runs.
    assert "scratch" in containers
    assert "scratch/nested" in containers
    assert "scratch" not in run_dirs
    assert "scratch/nested" not in run_dirs


def test_container_of_containers_root(tmp_path: Path):
    """A root whose immediate children are themselves run-dir containers
    (no run dir directly under root at all) -- the shape issue #432 titled
    the bug after."""
    _mk_run_dir(tmp_path / "site_a" / "run1")
    _mk_run_dir(tmp_path / "site_a" / "run2")
    _mk_run_dir(tmp_path / "site_b" / "deep" / "deeper" / "run3")
    run_dirs, containers, skipped = discover_run_dirs(str(tmp_path))
    assert set(run_dirs) == {"site_a/run1", "site_a/run2", "site_b/deep/deeper/run3"}
    assert set(containers) >= {"site_a", "site_b", "site_b/deep", "site_b/deep/deeper"}


def test_does_not_recurse_into_run_dir_artifacts(tmp_path: Path):
    """A run dir's own comparison/iter_N/report.json must not become a
    second row -- report.json is a plausible-looking marker but only
    cfg_args/train.log/chkpnt*.pth identify a run ROOT."""
    run = tmp_path / "run_with_comparison"
    _mk_run_dir(run)
    iter_dir = run / "comparison" / "iter_500"
    iter_dir.mkdir(parents=True)
    (iter_dir / "report.json").write_text("{}")
    run_dirs, containers, skipped = discover_run_dirs(str(tmp_path))
    assert run_dirs == ["run_with_comparison"]
    assert "run_with_comparison/comparison" not in run_dirs
    assert "run_with_comparison/comparison/iter_500" not in run_dirs


def test_self_referential_symlink_does_not_recurse_unboundedly(tmp_path: Path):
    """issue #443 part 1: a directory symlink pointing back at its own
    parent (e.g. ``output/scratch`` -> a shared mount that itself loops)
    must not send discover_run_dirs into unbounded recursion.

    ``os.listdir``/``os.path.isdir`` follow symlinks blindly, so without a
    visited-(st_dev, st_ino) guard the walk descends through ``loop/self``,
    ``loop/self/self``, ... -- confirmed pre-fix (against #438's unpatched
    _walk) to run for ~40 levels before the OS's own ELOOP symlink-resolution
    cap finally raises OSError and the walk is (accidentally) stopped by that,
    not by any guard in this script. The fix must stop at the FIRST repeat:
    exactly one container (``loop``) and the self-symlink recorded as a
    skipped/dead entry, never re-walked.
    """
    sub = tmp_path / "loop"
    sub.mkdir()
    (sub / "self").symlink_to(sub, target_is_directory=True)

    run_dirs, containers, skipped = discover_run_dirs(str(tmp_path))

    assert run_dirs == []
    assert containers == ["", "loop"]
    assert "loop/self" in skipped
    # The guard must prevent walking BELOW the repeat -- no loop/self/self...
    assert not any(p.startswith("loop/self/") for p in containers)
    assert not any(p.startswith("loop/self/") for p in skipped)


def test_empty_root_found_via_discover(tmp_path: Path):
    """An empty root has no run marker and no subdirectories -- it is a
    dead leaf itself (rel == ''), not a container."""
    run_dirs, containers, skipped = discover_run_dirs(str(tmp_path))
    assert run_dirs == []
    assert containers == []
    assert skipped == [""]


def test_audit_raises_loudly_on_empty_root(tmp_path: Path):
    """The no-runs-found case must be loud, not a silently-clean manifest."""
    with pytest.raises(RuntimeError, match="ZERO run directories"):
        audit(str(tmp_path))


def test_audit_raises_loudly_on_container_only_root(tmp_path: Path):
    """A root with subdirectories but none of them (at any depth) are real
    run dirs must also fail loudly, not report 0 with a clean exit."""
    (tmp_path / "not_a_run" / "also_not_a_run").mkdir(parents=True)
    with pytest.raises(RuntimeError, match="ZERO run directories"):
        audit(str(tmp_path))


def test_audit_finds_scratch_runs_recursively(tmp_path: Path):
    """End-to-end audit() over a root/scratch/run tree: the row for the
    nested run must appear with a path-qualified key, and no placeholder
    'scratch' row (classified no_checkpoint) should exist -- unlike the
    pre-fix script's single-root output/ manifest, which recorded exactly
    such a placeholder for output/scratch itself."""
    _mk_run_dir(tmp_path / "top_run")
    _mk_run_dir(tmp_path / "scratch" / "nested_run")
    manifest = audit(str(tmp_path))
    run_names = {r["run"] for r in manifest["runs"]}
    assert "top_run" in run_names
    assert "scratch/nested_run" in run_names
    assert "scratch" not in run_names
    assert manifest["n_run_dirs"] == 2
    # No chkpnt*.pth anywhere -> both rows are no_checkpoint.
    assert manifest["counts"] == {"no_checkpoint": 2}
    # The root itself ("") and "scratch" are both containers walked into.
    assert manifest["discovery"]["n_containers_traversed"] == 2
    assert set(manifest["discovery"]["containers_traversed"]) == {"", "scratch"}


def test_root_itself_is_a_run_dir(tmp_path: Path):
    """output_root pointed directly at a single run dir (rel == '') must
    still classify correctly and use the root's own basename as the row
    name, not an empty string."""
    _mk_run_dir(tmp_path)
    manifest = audit(str(tmp_path))
    assert manifest["n_run_dirs"] == 1
    assert manifest["runs"][0]["run"] == tmp_path.name


def test_cli_exits_nonzero_and_prints_loud_message_on_empty_root(tmp_path: Path):
    script = str(Path(__file__).resolve().parents[1] / "scripts" / "lineage_coverage_audit.py")
    result = subprocess.run(
        [sys.executable, script, str(tmp_path)],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "ZERO run directories" in result.stderr


def test_cli_reports_discovery_stats_on_success(tmp_path: Path):
    _mk_run_dir(tmp_path / "scratch" / "deep_run")
    script = str(Path(__file__).resolve().parents[1] / "scripts" / "lineage_coverage_audit.py")
    result = subprocess.run(
        [sys.executable, script, str(tmp_path)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "found 1 run dir(s)" in result.stdout
