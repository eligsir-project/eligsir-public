"""Tests for train.py's output-staging finalization semantics:
`.COMPLETE`/`.FAILED` markers and exit-code gating on real persistence
success.

CPU-only, no GPU compute exercised -- but importing train.py pulls in torch
and gsplat (this project's only rasterizer backend), so these tests must run
inside the container per CLAUDE.md's Run Contracts, not on bare host Python:

    docker compose run --rm train python -m pytest tests/test_output_staging_finalization.py -q
"""
from __future__ import annotations

import os
from argparse import Namespace

import pytest

import train


class _FakeManager:
    def __init__(self, active=True):
        self.active = active


class _FakeWorker:
    """Stub PersistenceWorker -- no real subprocess, .finalize() returns
    whatever the test wants on demand."""

    def __init__(self, finalize_result):
        self._finalize_result = finalize_result
        self.finalize_calls = []
        self.shutdown_called = False

    def finalize(self, timeout=600.0):
        self.finalize_calls.append(timeout)
        return self._finalize_result

    def shutdown(self):
        self.shutdown_called = True


def _staged_args(tmp_path, finalize_result, output_keep_local=False, active=True):
    local_dir = tmp_path / "local"
    persist_dir = tmp_path / "persist"
    local_dir.mkdir()
    persist_dir.mkdir()
    (local_dir / "point_cloud.ply").write_text("dummy")

    worker = _FakeWorker(finalize_result)
    args = Namespace(
        model_path=str(local_dir),
        persist_model_path=str(persist_dir),
        output_keep_local=output_keep_local,
        _output_staging_manager=_FakeManager(active=active),
        _persistence_worker=worker,
    )
    return args, worker, local_dir, persist_dir


class TestFinalizeOutputStagingSuccess:
    def test_writes_complete_not_failed(self, tmp_path):
        args, worker, local_dir, persist_dir = _staged_args(tmp_path, finalize_result=True)
        train._finalize_output_staging(args)
        assert os.path.isfile(os.path.join(persist_dir, ".COMPLETE"))
        assert not os.path.exists(os.path.join(persist_dir, ".FAILED"))

    def test_worker_shutdown_called(self, tmp_path):
        args, worker, local_dir, persist_dir = _staged_args(tmp_path, finalize_result=True)
        train._finalize_output_staging(args)
        assert worker.shutdown_called

    def test_local_dir_removed_by_default(self, tmp_path):
        args, worker, local_dir, persist_dir = _staged_args(tmp_path, finalize_result=True)
        train._finalize_output_staging(args)
        assert not os.path.exists(local_dir)

    def test_local_dir_preserved_when_keep_local(self, tmp_path):
        args, worker, local_dir, persist_dir = _staged_args(
            tmp_path, finalize_result=True, output_keep_local=True
        )
        train._finalize_output_staging(args)
        assert os.path.isdir(local_dir)
        assert os.path.isfile(local_dir / "point_cloud.ply")

    def test_local_dir_removal_failure_does_not_raise(self, tmp_path, monkeypatch):
        args, worker, local_dir, persist_dir = _staged_args(tmp_path, finalize_result=True)

        def _boom(path):
            raise OSError("simulated rmtree failure")

        monkeypatch.setattr(train.shutil, "rmtree", _boom)
        # Must not raise -- cleanup failure is a nuisance, not a correctness
        # problem, once the canonical result has already landed.
        train._finalize_output_staging(args)
        assert os.path.isfile(os.path.join(persist_dir, ".COMPLETE"))


class TestFinalizeOutputStagingFailure:
    def test_writes_failed_not_complete(self, tmp_path):
        args, worker, local_dir, persist_dir = _staged_args(tmp_path, finalize_result=False)
        with pytest.raises(train.OutputPersistenceError):
            train._finalize_output_staging(args)
        assert os.path.isfile(os.path.join(persist_dir, ".FAILED"))
        assert not os.path.exists(os.path.join(persist_dir, ".COMPLETE"))

    def test_failed_marker_has_nonempty_content(self, tmp_path):
        args, worker, local_dir, persist_dir = _staged_args(tmp_path, finalize_result=False)
        with pytest.raises(train.OutputPersistenceError):
            train._finalize_output_staging(args)
        content = (persist_dir / ".FAILED").read_text()
        assert content.strip()
        assert str(local_dir) in content

    def test_local_dir_never_deleted_on_failure(self, tmp_path):
        args, worker, local_dir, persist_dir = _staged_args(tmp_path, finalize_result=False)
        with pytest.raises(train.OutputPersistenceError):
            train._finalize_output_staging(args)
        assert os.path.isdir(local_dir)
        assert os.path.isfile(local_dir / "point_cloud.ply")

    def test_local_dir_never_deleted_even_with_default_keep_local_false(self, tmp_path):
        # output_keep_local=False is the default -- failure must still never
        # destroy the only surviving copy of the results, regardless of that
        # flag's value.
        args, worker, local_dir, persist_dir = _staged_args(
            tmp_path, finalize_result=False, output_keep_local=False
        )
        with pytest.raises(train.OutputPersistenceError):
            train._finalize_output_staging(args)
        assert os.path.isdir(local_dir)

    def test_worker_shutdown_called_on_failure_too(self, tmp_path):
        args, worker, local_dir, persist_dir = _staged_args(tmp_path, finalize_result=False)
        with pytest.raises(train.OutputPersistenceError):
            train._finalize_output_staging(args)
        assert worker.shutdown_called

    def test_signals_caller_to_exit_nonzero(self, tmp_path):
        """Whatever mechanism the caller (__main__) uses to translate this
        into sys.exit(1), it must be able to observe the failure. train.py
        does this by raising -- the same idiom already used elsewhere in
        this file (ValueError on hard config failures) -- and never catches
        it before __main__, so an uncaught exception propagates and yields
        a non-zero process exit code (Python's default for any uncaught
        exception)."""
        args, worker, local_dir, persist_dir = _staged_args(tmp_path, finalize_result=False)
        with pytest.raises(train.OutputPersistenceError) as excinfo:
            train._finalize_output_staging(args)
        assert isinstance(excinfo.value, RuntimeError)


class TestFinalizeOutputStagingNoOp:
    def test_noop_when_manager_missing(self, tmp_path):
        args = Namespace(model_path=str(tmp_path), persist_model_path=str(tmp_path))
        # Must not raise and must not touch the filesystem.
        train._finalize_output_staging(args)
        assert not os.path.exists(os.path.join(str(tmp_path), ".COMPLETE"))
        assert not os.path.exists(os.path.join(str(tmp_path), ".FAILED"))

    def test_noop_when_staging_inactive(self, tmp_path):
        args, worker, local_dir, persist_dir = _staged_args(
            tmp_path, finalize_result=True, active=False
        )
        train._finalize_output_staging(args)
        assert not worker.finalize_calls
        assert not os.path.exists(os.path.join(persist_dir, ".COMPLETE"))
        # local dir untouched -- unstaged runs are unaffected by this task.
        assert os.path.isdir(local_dir)
