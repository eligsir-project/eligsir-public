"""Tests for OutputStagingManager: NFS detection, path mapping, manifest."""
from __future__ import annotations

import json
import multiprocessing
import os
import tempfile
import time

import pytest

from eligsir.runtime.output_staging import (
    ArtifactManifest,
    OutputStagingManager,
    PersistenceWorker,
    is_network_filesystem,
    resolve_staging_root,
)


def _wait_until(predicate, timeout=10.0, interval=0.05):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()

FAKE_MOUNTS = """\
/dev/sda1 / ext4 rw,relatime 0 0
nfs.example.invalid:/data/persistent /data/persistent nfs4 rw,noatime,vers=4.2 0 0
overlay /data/persistent/bindmount overlay rw,relatime,lowerdir=/a,upperdir=/b,workdir=/c 0 0
tmpfs /tmp tmpfs rw,nosuid,nodev 0 0
"""


class TestIsNetworkFilesystem:
    def test_ext4_root_is_not_network(self):
        assert is_network_filesystem("/some/path", mounts_text=FAKE_MOUNTS) is False

    def test_nfs4_mount_is_network(self):
        assert is_network_filesystem(
            "/data/persistent", mounts_text=FAKE_MOUNTS
        ) is True

    def test_nested_path_resolves_longest_prefix(self):
        assert is_network_filesystem(
            "/data/persistent/output/run1", mounts_text=FAKE_MOUNTS
        ) is True

    def test_bind_mount_shaped_row_not_network(self):
        assert is_network_filesystem(
            "/data/persistent/bindmount/deep/path", mounts_text=FAKE_MOUNTS
        ) is False

    def test_tmpfs_not_network(self):
        assert is_network_filesystem("/tmp/foo", mounts_text=FAKE_MOUNTS) is False

    def test_unreadable_mounts_returns_false(self):
        def _boom():
            raise OSError("no /proc/mounts here")

        assert is_network_filesystem("/anything", mounts_text=None, _read_mounts=_boom) is False

    def test_no_matching_mount_returns_false(self):
        assert is_network_filesystem("/anything", mounts_text="") is False


class TestResolveStagingRoot:
    def test_explicit_root_used_when_writable(self, monkeypatch, tmp_path):
        monkeypatch.setattr(os, "access", lambda *a, **k: True)
        monkeypatch.setattr(
            "eligsir.runtime.output_staging.is_network_filesystem",
            lambda *a, **k: False,
        )
        monkeypatch.setattr(
            "eligsir.runtime.output_staging._is_tmpfs_mount", lambda *a, **k: False
        )
        result = resolve_staging_root(str(tmp_path / "explicit"))
        assert result == str(tmp_path / "explicit")

    def test_env_var_used_when_no_explicit_root(self, monkeypatch, tmp_path):
        monkeypatch.delenv("ELIGSIR_OUTPUT_STAGING_ROOT", raising=False)
        monkeypatch.setenv("ELIGSIR_OUTPUT_STAGING_ROOT", str(tmp_path / "envroot"))
        monkeypatch.setattr(os, "access", lambda *a, **k: True)
        monkeypatch.setattr(
            "eligsir.runtime.output_staging.is_network_filesystem",
            lambda *a, **k: False,
        )
        monkeypatch.setattr(
            "eligsir.runtime.output_staging._is_tmpfs_mount", lambda *a, **k: False
        )
        result = resolve_staging_root(None)
        assert result == str(tmp_path / "envroot")

    def test_falls_through_to_tmpdir_when_scratch_unwritable(self, monkeypatch, tmp_path):
        monkeypatch.delenv("ELIGSIR_OUTPUT_STAGING_ROOT", raising=False)
        monkeypatch.setenv("TMPDIR", str(tmp_path / "tmpdir_root"))
        monkeypatch.setattr(
            "eligsir.runtime.output_staging.is_network_filesystem",
            lambda *a, **k: False,
        )
        monkeypatch.setattr(
            "eligsir.runtime.output_staging._is_tmpfs_mount", lambda *a, **k: False
        )

        real_access = os.access

        def fake_access(path, mode):
            if str(path).startswith("/scratch"):
                return False
            return real_access(path, mode) if os.path.exists(path) else True

        monkeypatch.setattr(os, "access", fake_access)
        result = resolve_staging_root(None)
        assert result == os.path.join(str(tmp_path / "tmpdir_root"), "eligsir")

    def test_tmpfs_candidate_skipped(self, monkeypatch, tmp_path):
        monkeypatch.delenv("ELIGSIR_OUTPUT_STAGING_ROOT", raising=False)
        monkeypatch.setenv("TMPDIR", str(tmp_path / "tmpdir_root"))
        monkeypatch.setattr(os, "access", lambda *a, **k: True)

        def fake_is_net(path, **kwargs):
            return False

        # Mark explicit_root candidate as tmpfs via a dedicated fstype check.
        calls = {"count": 0}

        def fake_is_tmpfs(path):
            calls["count"] += 1
            return calls["count"] == 1  # first candidate (explicit_root) is tmpfs

        monkeypatch.setattr(
            "eligsir.runtime.output_staging.is_network_filesystem", fake_is_net
        )
        monkeypatch.setattr(
            "eligsir.runtime.output_staging._is_tmpfs_mount", fake_is_tmpfs
        )
        result = resolve_staging_root(str(tmp_path / "explicit_tmpfs"))
        assert result != str(tmp_path / "explicit_tmpfs")

    def test_raises_when_all_candidates_unwritable_or_tmpfs(self, monkeypatch):
        monkeypatch.delenv("ELIGSIR_OUTPUT_STAGING_ROOT", raising=False)
        monkeypatch.delenv("TMPDIR", raising=False)
        monkeypatch.setattr(os, "access", lambda *a, **k: False)
        monkeypatch.setattr(
            "eligsir.runtime.output_staging.is_network_filesystem",
            lambda *a, **k: False,
        )
        monkeypatch.setattr(
            "eligsir.runtime.output_staging._is_tmpfs_mount", lambda *a, **k: False
        )
        with pytest.raises(RuntimeError):
            resolve_staging_root(None)


class TestOutputStagingManager:
    def test_mode_off_is_inactive_and_paths_match(self, tmp_path):
        persist = str(tmp_path / "persist_out")
        mgr = OutputStagingManager(persist_model_path=persist, mode="off")
        assert mgr.active is False
        assert mgr.local_path == mgr.persist_path == persist

    def test_mode_on_active_regardless_of_fs(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            "eligsir.runtime.output_staging.is_network_filesystem",
            lambda *a, **k: False,
        )
        monkeypatch.setattr(
            "eligsir.runtime.output_staging._is_tmpfs_mount", lambda *a, **k: False
        )
        persist = str(tmp_path / "persist_out")
        mgr = OutputStagingManager(
            persist_model_path=persist, mode="on", staging_root=str(tmp_path / "stage")
        )
        assert mgr.active is True
        assert mgr.local_path != mgr.persist_path

    def test_mode_auto_follows_detection_true(self, monkeypatch, tmp_path):
        persist = str(tmp_path / "persist_out")
        os.makedirs(persist)
        stage = str(tmp_path / "stage")

        def fake_is_net(path, **kwargs):
            return os.path.abspath(path) == os.path.abspath(persist)

        monkeypatch.setattr(
            "eligsir.runtime.output_staging.is_network_filesystem", fake_is_net
        )
        monkeypatch.setattr(
            "eligsir.runtime.output_staging._is_tmpfs_mount", lambda *a, **k: False
        )
        mgr = OutputStagingManager(persist_model_path=persist, mode="auto", staging_root=stage)
        assert mgr.active is True

    def test_mode_auto_follows_detection_false(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            "eligsir.runtime.output_staging.is_network_filesystem",
            lambda *a, **k: False,
        )
        persist = str(tmp_path / "persist_out")
        mgr = OutputStagingManager(persist_model_path=persist, mode="auto")
        assert mgr.active is False
        assert mgr.local_path == persist

    def test_setup_creates_both_dirs_when_active(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            "eligsir.runtime.output_staging.is_network_filesystem",
            lambda *a, **k: False,
        )
        monkeypatch.setattr(
            "eligsir.runtime.output_staging._is_tmpfs_mount", lambda *a, **k: False
        )
        persist = str(tmp_path / "persist_out")
        mgr = OutputStagingManager(
            persist_model_path=persist, mode="on", staging_root=str(tmp_path / "stage")
        )
        mgr.setup()
        assert os.path.isdir(mgr.local_path)
        assert os.path.isdir(mgr.persist_path)

    def test_setup_creates_only_local_dir_when_inactive(self, tmp_path):
        persist = str(tmp_path / "persist_out")
        mgr = OutputStagingManager(persist_model_path=persist, mode="off")
        mgr.setup()
        assert os.path.isdir(mgr.local_path)

    def test_run_id_default_naming_convention(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            "eligsir.runtime.output_staging.is_network_filesystem",
            lambda *a, **k: False,
        )
        monkeypatch.setattr(
            "eligsir.runtime.output_staging._is_tmpfs_mount", lambda *a, **k: False
        )
        persist = str(tmp_path / "myrun")
        mgr = OutputStagingManager(
            persist_model_path=persist, mode="on", staging_root=str(tmp_path / "stage")
        )
        base = os.path.basename(mgr.local_path)
        parts = base.split("-")
        assert parts[0] == "myrun"
        assert len(parts) == 3
        assert len(parts[1]) == 8
        assert parts[2].isdigit()


class TestArtifactManifest:
    def test_load_missing_file_is_empty(self, tmp_path):
        manifest = ArtifactManifest.load(str(tmp_path / "no_such_manifest.json"))
        assert manifest.changed("some/path.ply", 123, 456) is True

    def test_roundtrip_save_load(self, tmp_path):
        manifest_path = str(tmp_path / "manifest.json")
        m = ArtifactManifest()
        m.update("train.log", 100, 111)
        m.update("point_cloud/iter_1000.ply", 5000, 222)
        m.save(manifest_path)

        assert os.path.isfile(manifest_path)
        with open(manifest_path) as f:
            data = json.load(f)
        assert data["train.log"] == [100, 111]

        loaded = ArtifactManifest.load(manifest_path)
        assert loaded.changed("train.log", 100, 111) is False
        assert loaded.changed("point_cloud/iter_1000.ply", 5000, 222) is False

    def test_changed_true_for_new_path(self):
        m = ArtifactManifest()
        assert m.changed("new/path.ply", 1, 1) is True

    def test_changed_true_after_size_change(self):
        m = ArtifactManifest()
        m.update("a.ply", 100, 111)
        assert m.changed("a.ply", 200, 111) is True

    def test_changed_true_after_mtime_change(self):
        m = ArtifactManifest()
        m.update("a.ply", 100, 111)
        assert m.changed("a.ply", 100, 999) is True

    def test_changed_false_when_unchanged(self):
        m = ArtifactManifest()
        m.update("a.ply", 100, 111)
        assert m.changed("a.ply", 100, 111) is False

    def test_update_after_changed_detection(self):
        m = ArtifactManifest()
        m.update("a.ply", 100, 111)
        assert m.changed("a.ply", 200, 222) is True
        m.update("a.ply", 200, 222)
        assert m.changed("a.ply", 200, 222) is False


class TestPersistenceWorker:
    def _active_manager(self, tmp_path):
        local = str(tmp_path / "local")
        persist = str(tmp_path / "persist")
        os.makedirs(local, exist_ok=True)
        os.makedirs(persist, exist_ok=True)
        mgr = OutputStagingManager(persist_model_path=persist, mode="off")
        mgr.local_path = local
        mgr.persist_path = persist
        mgr.active = True
        return mgr

    def test_files_sync_to_persist_path(self, tmp_path):
        mgr = self._active_manager(tmp_path)
        with open(os.path.join(mgr.local_path, "a.txt"), "w") as f:
            f.write("hello")
        nested = os.path.join(mgr.local_path, "point_cloud", "iteration_5000")
        os.makedirs(nested, exist_ok=True)
        with open(os.path.join(nested, "point_cloud.ply"), "w") as f:
            f.write("plydata")

        worker = PersistenceWorker(mgr, log_sync_interval=0.1, artifact_sync_interval=0.5)
        worker.start()
        try:
            dest_a = os.path.join(mgr.persist_path, "a.txt")
            dest_ply = os.path.join(
                mgr.persist_path, "point_cloud", "iteration_5000", "point_cloud.ply"
            )
            assert _wait_until(
                lambda: os.path.isfile(dest_a) and os.path.isfile(dest_ply), timeout=10.0
            )
            with open(dest_a) as f:
                assert f.read() == "hello"
            with open(dest_ply) as f:
                assert f.read() == "plydata"
        finally:
            worker.finalize(timeout=10.0)
            worker.shutdown()

    def test_train_log_appends_not_duplicated(self, tmp_path):
        mgr = self._active_manager(tmp_path)
        log_path = os.path.join(mgr.local_path, "train.log")
        with open(log_path, "w") as f:
            f.write("line1\n")

        worker = PersistenceWorker(mgr, log_sync_interval=0.1, artifact_sync_interval=0.5)
        worker.start()
        try:
            dest_log = os.path.join(mgr.persist_path, "train.log")
            assert _wait_until(lambda: os.path.isfile(dest_log), timeout=10.0)
            assert _wait_until(
                lambda: os.path.isfile(dest_log) and open(dest_log).read() == "line1\n",
                timeout=10.0,
            )

            with open(log_path, "a") as f:
                f.write("line2\n")

            assert _wait_until(
                lambda: os.path.isfile(dest_log)
                and open(dest_log).read() == "line1\nline2\n",
                timeout=10.0,
            )
        finally:
            worker.finalize(timeout=10.0)
            worker.shutdown()

    def test_modified_content_propagates(self, tmp_path):
        mgr = self._active_manager(tmp_path)
        target = os.path.join(mgr.local_path, "cfg_args")
        with open(target, "w") as f:
            f.write("v1")

        worker = PersistenceWorker(mgr, log_sync_interval=0.1, artifact_sync_interval=0.3)
        worker.start()
        try:
            dest = os.path.join(mgr.persist_path, "cfg_args")
            assert _wait_until(
                lambda: os.path.isfile(dest) and open(dest).read() == "v1", timeout=10.0
            )

            time.sleep(0.05)
            with open(target, "w") as f:
                f.write("v2-updated")

            assert _wait_until(
                lambda: os.path.isfile(dest) and open(dest).read() == "v2-updated",
                timeout=10.0,
            )
        finally:
            worker.finalize(timeout=10.0)
            worker.shutdown()

    def test_finalize_flushes_pending_changes_immediately(self, tmp_path):
        mgr = self._active_manager(tmp_path)
        worker = PersistenceWorker(mgr, log_sync_interval=5.0, artifact_sync_interval=300.0)
        worker.start()
        try:
            with open(os.path.join(mgr.local_path, "late.txt"), "w") as f:
                f.write("just written")
            with open(os.path.join(mgr.local_path, "train.log"), "w") as f:
                f.write("final log line\n")

            result = worker.finalize(timeout=15.0)
            assert result is True

            dest_late = os.path.join(mgr.persist_path, "late.txt")
            dest_log = os.path.join(mgr.persist_path, "train.log")
            assert os.path.isfile(dest_late)
            with open(dest_late) as f:
                assert f.read() == "just written"
            assert os.path.isfile(dest_log)
            with open(dest_log) as f:
                assert f.read() == "final log line\n"
        finally:
            worker.shutdown()

    def test_sync_now_flushes_in_child_and_keeps_worker_running(self, tmp_path):
        mgr = self._active_manager(tmp_path)
        worker = PersistenceWorker(
            mgr, log_sync_interval=30.0, artifact_sync_interval=300.0
        )
        worker.start()
        try:
            with open(os.path.join(mgr.local_path, "endpoint.ply"), "w") as handle:
                handle.write("endpoint")

            worker.sync_now(timeout=15.0)
            with open(os.path.join(mgr.persist_path, "endpoint.ply")) as handle:
                assert handle.read() == "endpoint"
            assert worker._process.is_alive()
        finally:
            worker.finalize(timeout=15.0)
            worker.shutdown()

    def test_inactive_manager_start_and_finalize_are_noops(self, tmp_path):
        persist = str(tmp_path / "persist_out")
        mgr = OutputStagingManager(persist_model_path=persist, mode="off")
        worker = PersistenceWorker(mgr, log_sync_interval=0.1, artifact_sync_interval=0.5)
        worker.start()
        assert multiprocessing.active_children() == []
        assert worker.finalize(timeout=1.0) is True
        assert multiprocessing.active_children() == []
