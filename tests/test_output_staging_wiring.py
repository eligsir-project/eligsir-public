"""Tests for wiring OutputStagingManager/PersistenceWorker into
train.py's ensure_model_path and TeeStream.

CPU-only, no GPU compute exercised -- but importing train.py pulls in torch
and gsplat (this project's only rasterizer backend), so these tests must run
inside the container per CLAUDE.md's Run Contracts, not on bare host Python:

    docker compose run --rm train python -m pytest tests/test_output_staging_wiring.py -q
"""
from __future__ import annotations

import multiprocessing
import os
import tempfile
import time
from argparse import Namespace

import pytest

import train


def _wait_until(predicate, timeout=10.0, interval=0.05):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


def _base_args(**overrides):
    ns = Namespace(
        model_path="",
        output_staging="off",
        output_staging_root="",
        output_sync_interval=300.0,
        output_log_sync_interval=2.0,
        output_keep_local=False,
    )
    for k, v in overrides.items():
        setattr(ns, k, v)
    return ns


class TestEnsureModelPathStagingOff:
    def test_persist_equals_model_path(self, tmp_path):
        args = _base_args(model_path=str(tmp_path / "run1"))
        train.ensure_model_path(args)
        assert args.model_path == str(tmp_path / "run1")
        assert args.persist_model_path == args.model_path

    def test_no_persistence_worker_process_spawned(self, tmp_path):
        before = set(p.pid for p in multiprocessing.active_children())
        args = _base_args(model_path=str(tmp_path / "run2"))
        train.ensure_model_path(args)
        after = set(p.pid for p in multiprocessing.active_children())
        assert after == before
        # A PersistenceWorker object is still constructed (uniform interface
        # for Task 4), it just never starts a background process.
        assert args._persistence_worker is not None


class TestEnsureModelPathStagingOn:
    def test_redirects_to_local_staging_dir(self, tmp_path):
        persist_dir = str(tmp_path / "persist_run")
        staging_root = str(tmp_path / "staging_root")
        args = _base_args(
            model_path=persist_dir,
            output_staging="on",
            output_staging_root=staging_root,
        )
        try:
            train.ensure_model_path(args)
            assert args.persist_model_path == persist_dir
            assert args.model_path != persist_dir
            assert os.path.isdir(args.model_path)
            assert args.model_path.startswith(staging_root)

            worker_ok = _wait_until(
                lambda: any(
                    p.pid == args._persistence_worker._process.pid
                    for p in multiprocessing.active_children()
                )
            )
            assert worker_ok
        finally:
            args._persistence_worker.shutdown()


class FakeStream:
    def __init__(self):
        self.chunks = []

    def write(self, text):
        self.chunks.append(text)

    def flush(self):
        pass

    @property
    def content(self):
        return "".join(self.chunks)


class TestTeeStreamBuffering:
    def test_staging_active_batches_console_but_not_local_file(self):
        console = FakeStream()
        local_file = FakeStream()
        tee = train.TeeStream(console, local_file, staging_active=True)
        tee._FLUSH_INTERVAL_S = 10_000  # never time-trigger during this test
        tee._FLUSH_SIZE_BYTES = 1024  # only size-trigger

        tee.write("hello ")
        tee.write("world ")
        # Local file gets every write immediately -- it's what the
        # persistence worker tails.
        assert local_file.content == "hello world "
        # Console has not crossed the size threshold yet.
        assert console.content == ""

        tee.write("x" * 2000)
        # Buffer now exceeds 1024 bytes -- console catches up.
        assert console.content == "hello world " + "x" * 2000
        assert local_file.content == "hello world " + "x" * 2000

    def test_staging_active_time_based_flush(self):
        console = FakeStream()
        local_file = FakeStream()
        tee = train.TeeStream(console, local_file, staging_active=True)
        tee._FLUSH_INTERVAL_S = 0.05
        tee._FLUSH_SIZE_BYTES = 10_000_000  # never size-trigger

        tee.write("first")
        assert console.content == ""
        time.sleep(0.1)
        tee.write("second")
        assert console.content == "firstsecond"

    def test_staging_inactive_every_write_reaches_both_immediately(self):
        console = FakeStream()
        local_file = FakeStream()
        tee = train.TeeStream(console, local_file, staging_active=False)

        tee.write("a")
        tee.write("b")
        tee.write("c")
        assert console.content == "abc"
        assert local_file.content == "abc"


class TestOutputStagingEnvOverridePrecedence:
    """Regression tests for the substring-collision bug: `flag_name=
    "output_staging"` must not falsely match when only the longer
    `--output_staging_root` flag was actually passed on the CLI.
    """

    def test_env_applies_when_only_longer_prefixed_flag_was_passed(self, monkeypatch):
        # --output_staging_root was passed, but --output_staging itself
        # was NOT -- the env var for the shorter flag name must still apply.
        monkeypatch.setattr(train.sys, "argv", ["train.py", "--output_staging_root", "/tmp/x"])
        monkeypatch.setenv("ELIGSIR_OUTPUT_STAGING", "on")
        args = Namespace(output_staging="off")
        train._output_staging_env_override(args, "output_staging", "ELIGSIR_OUTPUT_STAGING")
        assert args.output_staging == "on"

    def test_env_ignored_when_flag_explicitly_passed(self, monkeypatch):
        monkeypatch.setattr(train.sys, "argv", ["train.py", "--output_staging", "off"])
        monkeypatch.setenv("ELIGSIR_OUTPUT_STAGING", "on")
        args = Namespace(output_staging="off")
        train._output_staging_env_override(args, "output_staging", "ELIGSIR_OUTPUT_STAGING")
        assert args.output_staging == "off"

    def test_env_ignored_when_flag_passed_equals_form(self, monkeypatch):
        monkeypatch.setattr(train.sys, "argv", ["train.py", "--output_staging=off"])
        monkeypatch.setenv("ELIGSIR_OUTPUT_STAGING", "on")
        args = Namespace(output_staging="off")
        train._output_staging_env_override(args, "output_staging", "ELIGSIR_OUTPUT_STAGING")
        assert args.output_staging == "off"

    def test_env_applies_when_argv_has_only_program_name(self, monkeypatch):
        monkeypatch.setattr(train.sys, "argv", ["train.py"])
        monkeypatch.setenv("ELIGSIR_OUTPUT_STAGING_ROOT", "/tmp/staging")
        args = Namespace(output_staging_root="")
        train._output_staging_env_override(
            args, "output_staging_root", "ELIGSIR_OUTPUT_STAGING_ROOT"
        )
        assert args.output_staging_root == "/tmp/staging"


class TestEnsureModelPathDoubleCallIdempotent:
    """Regression test for the double-ensure_model_path bug: calling
    ensure_model_path twice on the same args object (e.g. once directly,
    once again via start_output_log) must not spawn a second
    PersistenceWorker/process.
    """

    def test_second_call_reuses_same_manager_and_worker(self, tmp_path):
        persist_dir = str(tmp_path / "persist_run")
        staging_root = str(tmp_path / "staging_root")
        args = _base_args(
            model_path=persist_dir,
            output_staging="on",
            output_staging_root=staging_root,
        )
        try:
            train.ensure_model_path(args)
            manager_first = args._output_staging_manager
            worker_first = args._persistence_worker

            train.ensure_model_path(args)

            assert args._output_staging_manager is manager_first
            assert args._persistence_worker is worker_first
        finally:
            args._persistence_worker.shutdown()

    def test_start_output_log_after_ensure_model_path_spawns_one_worker_process(self, tmp_path):
        import sys as _sys

        persist_dir = str(tmp_path / "persist_run2")
        staging_root = str(tmp_path / "staging_root2")
        args = _base_args(
            model_path=persist_dir,
            output_staging="on",
            output_staging_root=staging_root,
        )
        orig_stdout, orig_stderr = _sys.stdout, _sys.stderr
        try:
            train.ensure_model_path(args)
            worker_ok = _wait_until(
                lambda: any(
                    p.pid == args._persistence_worker._process.pid
                    for p in multiprocessing.active_children()
                )
            )
            assert worker_ok
            worker_pid = args._persistence_worker._process.pid

            # start_output_log calls ensure_model_path again internally --
            # must be a no-op re-entry, not a second worker/process.
            train.start_output_log(args)

            children_matching = [
                p for p in multiprocessing.active_children() if p.pid == worker_pid
            ]
            assert len(children_matching) == 1
        finally:
            _sys.stdout, _sys.stderr = orig_stdout, orig_stderr
            args._persistence_worker.shutdown()
