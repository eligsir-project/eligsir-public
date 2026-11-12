"""Coverage for scripts/ram_dataset_cache.py: argparse contract, LRU eviction,
prune, and the path-traversal guard on --name."""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
import time
from pathlib import Path
from unittest import mock

import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "ram_dataset_cache.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("ram_dataset_cache", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


ram_dataset_cache = _load_module()


def _write_file(path: Path, size: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)


# ---------------------------------------------------------------------------
# (e) regression test for the argparse bug itself: the documented invocations
# from the module docstring must actually parse (subcommand first, then the
# shared --root/--budget-gb options after it).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "extra_args",
    [
        ["evict", "--root", "{root}", "--budget-gb", "200"],
        [
            "ensure",
            "--name",
            "ram-fr3",
            "--source",
            "{source}",
            "--root",
            "{root}",
            "--budget-gb",
            "200",
        ],
        ["prune", "--root", "{root}", "--name", "ram-fr3"],
    ],
)
def test_documented_invocations_parse_and_run(tmp_path: Path, extra_args: list[str]) -> None:
    if extra_args[0] == "ensure" and shutil.which("rsync") is None:
        pytest.skip("requires rsync on PATH (not installed in the CPU smoke image)")

    root = tmp_path / "root"
    root.mkdir()
    source = tmp_path / "source"
    source.mkdir()
    _write_file(source / "a.bin", 128)

    args = [a.format(root=str(root), source=str(source)) for a in extra_args]
    result = subprocess.run(
        [sys.executable, str(MODULE_PATH)] + args,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"documented invocation failed to parse/run: {args}\n"
        f"stdout={result.stdout}\nstderr={result.stderr}"
    )


# ---------------------------------------------------------------------------
# (a) ensure() creates the target dir with a .COMPLETE marker and reuses it
# on a second call without re-copying.
# ---------------------------------------------------------------------------


def test_ensure_creates_and_reuses_without_recopy(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    source = tmp_path / "source"
    source.mkdir()
    _write_file(source / "a.bin", 64)

    call_count = 0
    real_run = subprocess.run

    def fake_run(cmd, *args, **kwargs):
        nonlocal call_count
        if cmd[0] == "rsync":
            call_count += 1
            # Simulate rsync by actually copying, so .COMPLETE reflects reality.
            src_dir = Path(cmd[-2].rstrip("/"))
            dst_dir = Path(cmd[-1].rstrip("/"))
            dst_dir.mkdir(parents=True, exist_ok=True)
            for f in src_dir.iterdir():
                (dst_dir / f.name).write_bytes(f.read_bytes())
            return subprocess.CompletedProcess(cmd, 0)
        return real_run(cmd, *args, **kwargs)

    with mock.patch.object(ram_dataset_cache.subprocess, "run", side_effect=fake_run):
        target1 = ram_dataset_cache.ensure(
            root, "ram-fr3", source, budget_bytes=10**12, reserve_bytes=0
        )
        assert target1.is_dir()
        assert (target1 / ".COMPLETE").exists()
        assert call_count == 1

        target2 = ram_dataset_cache.ensure(
            root, "ram-fr3", source, budget_bytes=10**12, reserve_bytes=0
        )
        assert target2 == target1
        # Second call must reuse the staged copy -- no additional rsync call.
        assert call_count == 1


# ---------------------------------------------------------------------------
# (b) evict() respects the size budget, evicts oldest-by-.last_used-mtime
# first, and never touches an entry passed as --keep.
# ---------------------------------------------------------------------------


def test_evict_respects_budget_oldest_first_and_keep(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()

    entries = ["ram-old", "ram-mid", "ram-new", "ram-kept"]
    now = time.time()
    for i, name in enumerate(entries):
        entry_dir = root / name
        _write_file(entry_dir / "data.bin", 10**6)  # 1 MB each
        marker = entry_dir / ".last_used"
        marker.touch()
        # Stagger mtimes: ram-old is oldest, ram-kept would be oldest of all
        # but is protected via --keep.
        offset = {"ram-kept": -1000, "ram-old": -300, "ram-mid": -200, "ram-new": -100}[name]
        import os

        os.utime(marker, (now + offset, now + offset))

    # Budget only large enough for ~2 entries; force eviction of the rest.
    budget_bytes = 2 * 10**6 + 500_000
    ram_dataset_cache.evict(root, budget_bytes, reserve_bytes=0, keep="ram-kept")

    remaining = {p.name for p in ram_dataset_cache._cache_entries(root)}
    assert "ram-kept" in remaining, "kept entry must never be evicted"
    assert "ram-old" not in remaining, "oldest non-kept entry should be evicted first"
    assert "ram-new" in remaining, "newest entry should survive under budget"


# ---------------------------------------------------------------------------
# (c) prune() removes a named entry.
# ---------------------------------------------------------------------------


def test_prune_removes_named_entry(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    target = root / "ram-scratch"
    _write_file(target / "data.bin", 100)
    assert target.is_dir()

    ram_dataset_cache.prune(root, "ram-scratch")

    assert not target.exists()


def test_prune_missing_entry_is_a_noop(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    # Should not raise even though ram-missing was never created.
    ram_dataset_cache.prune(root, "ram-missing")


# ---------------------------------------------------------------------------
# (d) the path-traversal guard rejects a --name that would resolve outside
# --root.
# ---------------------------------------------------------------------------


def test_safe_target_rejects_traversal_outside_root(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()

    with pytest.raises(SystemExit):
        ram_dataset_cache._safe_target(root, "ram-a/../../evil")


def test_safe_target_rejects_non_ram_prefixed_name(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()

    with pytest.raises(SystemExit):
        ram_dataset_cache._safe_target(root, "not-ram-prefixed")


def test_safe_target_accepts_well_formed_name(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()

    target = ram_dataset_cache._safe_target(root, "ram-fr3")
    assert target == (root / "ram-fr3").resolve()


def test_prune_refuses_traversal_before_rmtree(tmp_path: Path) -> None:
    """End-to-end: prune() must refuse (not rmtree) a traversal --name,
    even though the naive `name.startswith(PREFIX)` check alone would pass."""
    root = tmp_path / "root"
    root.mkdir()
    # A sibling directory outside root that a traversal name could reach.
    escape_target = tmp_path / "evil"
    _write_file(escape_target / "important.txt", 10)
    assert escape_target.exists()

    with pytest.raises(SystemExit):
        ram_dataset_cache.prune(root, "ram-x/../../evil")

    # The out-of-root directory must survive untouched.
    assert escape_target.exists()
    assert (escape_target / "important.txt").exists()


def test_ensure_refuses_traversal_before_mkdir(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    source = tmp_path / "source"
    source.mkdir()
    _write_file(source / "a.bin", 10)

    escape_dir = tmp_path / "evil"
    assert not escape_dir.exists()

    with pytest.raises(SystemExit):
        ram_dataset_cache.ensure(
            root, "ram-x/../../evil", source, budget_bytes=10**9, reserve_bytes=0
        )

    assert not escape_dir.exists()
