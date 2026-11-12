"""Tests for the shared asynchronous save worker."""

from __future__ import annotations

import threading

import pytest

from eligsir.runtime.save_worker import AsyncSaveWorker


def test_streaming_save_worker_reexports_canonical_worker():
    from streaming.save_worker import AsyncSaveWorker as CompatAsyncSaveWorker

    assert CompatAsyncSaveWorker is AsyncSaveWorker


def test_async_save_worker_runs_queued_tasks():
    seen: list[int] = []
    worker = AsyncSaveWorker(maxsize=2, join_timeout_s=5)
    try:
        worker.enqueue(lambda: seen.append(1))
    finally:
        worker.shutdown()
    assert seen == [1]


def test_async_save_worker_join_waits_for_queued_tasks():
    seen: list[int] = []
    worker = AsyncSaveWorker(join_timeout_s=5)
    try:
        worker.enqueue(lambda: seen.append(1))
        worker.join()
        assert seen == [1]
        assert worker.queue_depth() == 0
    finally:
        worker.shutdown()


def test_async_save_worker_shutdown_propagates_persistence_failure():
    worker = AsyncSaveWorker(join_timeout_s=5)

    def fail_write():
        raise OSError("disk full")

    worker.enqueue(fail_write)
    with pytest.raises(RuntimeError, match="async save task failed") as error:
        worker.shutdown()

    assert isinstance(error.value.__cause__, OSError)
    assert str(error.value.__cause__) == "disk full"


def test_async_save_worker_shutdown_rejects_live_thread_after_timeout():
    release = threading.Event()
    worker = AsyncSaveWorker(join_timeout_s=0)
    worker.enqueue(release.wait)

    try:
        with pytest.raises(RuntimeError, match="did not stop"):
            worker.shutdown()
    finally:
        release.set()
        worker._thread.join(timeout=1)
