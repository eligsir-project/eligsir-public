"""Tests for assert_persistent_mount (issue #293 fix item 2): a mandatory
pre-training preflight that catches a dropped/missing bind mount before a
run silently trains to completion and loses every artifact on container
exit -- see eligsir/runtime/output_staging.py's
UnmountedPersistentPathError docstring for the full incident this guards
against.
"""
from __future__ import annotations

import os

import pytest

from eligsir.runtime.output_staging import (
    UnmountedPersistentPathError,
    assert_persistent_mount,
)


def test_relative_path_is_a_noop(tmp_path, monkeypatch):
    """Relative paths (this project's `output/...` convention) are never
    checked -- they live under the repo's own /workspace bind mount by
    convention, not a named shared-mount prefix."""
    monkeypatch.chdir(tmp_path)
    assert_persistent_mount("output/some_run")  # must not raise


def test_absolute_path_outside_known_prefixes_is_a_noop(tmp_path):
    """An absolute path that doesn't start with a configured shared-mount
    prefix is left alone -- this check is intentionally narrow."""
    path = str(tmp_path / "not_a_shared_mount" / "run1")
    assert_persistent_mount(path, mount_prefixes=("/data/",))  # must not raise


def test_genuinely_distinct_mount_passes(tmp_path):
    """A path whose ancestor's st_dev differs from the (simulated)
    container root passes cleanly -- the normal, correctly-configured case."""
    mount_root = tmp_path / "mnt_fake"
    mount_root.mkdir()
    run_dir = mount_root / "user" / "output" / "run1"
    fake_container_root_dev = os.stat(str(tmp_path)).st_dev + 1  # guaranteed different
    assert_persistent_mount(
        str(run_dir),
        mount_prefixes=(str(mount_root),),
        root_dev=fake_container_root_dev,
    )
    assert run_dir.is_dir()  # the probe's makedirs actually ran


def test_same_device_as_container_root_raises(tmp_path):
    """The core regression: a path under a known shared-mount prefix that
    resolves to the SAME device as the container root ("/") -- i.e. the
    expected bind mount is absent -- must raise loudly, not silently
    succeed."""
    mount_root = tmp_path / "mnt_missing"
    mount_root.mkdir()
    run_dir = mount_root / "user" / "output" / "run1"
    real_root_dev = os.stat(str(tmp_path)).st_dev  # same fs as `tmp_path` itself
    with pytest.raises(UnmountedPersistentPathError):
        assert_persistent_mount(
            str(run_dir),
            mount_prefixes=(str(mount_root),),
            root_dev=real_root_dev,
        )


def test_read_only_mount_raises_on_write_probe(tmp_path, monkeypatch):
    """A path that IS a distinct mount but is read-only (or otherwise
    unwritable) must also be caught -- the write probe is independent of
    the device-identity check."""
    mount_root = tmp_path / "mnt_ro"
    mount_root.mkdir()
    run_dir = mount_root / "run1"

    def _boom_makedirs(*a, **k):
        raise OSError("Read-only file system")

    monkeypatch.setattr(os, "makedirs", _boom_makedirs)
    with pytest.raises(UnmountedPersistentPathError):
        assert_persistent_mount(str(run_dir), mount_prefixes=(str(mount_root),))


def test_env_var_overrides_default_prefixes(tmp_path, monkeypatch):
    """ELIGSIR_PERSISTENT_MOUNT_PREFIXES lets a deployment widen/narrow
    which prefixes are checked without a code change."""
    custom_root = tmp_path / "custom_share"
    custom_root.mkdir()
    run_dir = custom_root / "run1"
    monkeypatch.setenv("ELIGSIR_PERSISTENT_MOUNT_PREFIXES", str(custom_root))
    real_root_dev = os.stat(str(tmp_path)).st_dev
    with pytest.raises(UnmountedPersistentPathError):
        assert_persistent_mount(str(run_dir), root_dev=real_root_dev)


def test_default_prefix_is_mnt():
    """Documented default prefixes do not cover arbitrary workspace paths."""
    assert_persistent_mount("/workspace/output/run1")  # outside default prefixes -> no-op
