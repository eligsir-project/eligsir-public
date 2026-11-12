"""Regression test: AsyncSaveWorker.enqueue() must never block the calling
thread, even past the old maxsize=4 bound, and queue_depth() must reflect
backlog before the background thread drains it.

Root cause this guards against: a bounded queue(maxsize=4) blocked the main
training thread's enqueue() call once 4 writes were already pending -- a
real, confirmed contributor to a mid-training stall (see
docs/superpowers/reports/2026-07-16-sh-native-backward-parity-findings.md's
Phase B3 addendum).
"""
import threading
import time

from eligsir.runtime.save_worker import AsyncSaveWorker


def test_enqueue_never_blocks_past_old_maxsize_of_four():
    release = threading.Event()

    def _slow_task():
        release.wait(timeout=5.0)

    worker = AsyncSaveWorker()
    try:
        # Fill the background thread with one long-running task so nothing
        # drains while we enqueue more than the old maxsize=4 bound.
        worker.enqueue(_slow_task)

        enqueue_done = threading.Event()

        def _enqueue_many():
            for _ in range(20):
                worker.enqueue(lambda: None)
            enqueue_done.set()

        t = threading.Thread(target=_enqueue_many, daemon=True)
        t.start()
        t.join(timeout=2.0)

        assert enqueue_done.is_set(), (
            "enqueue() blocked past the old maxsize=4 bound -- queue is not unbounded"
        )
    finally:
        release.set()
        worker.shutdown()


def test_queue_depth_reflects_backlog_before_drain():
    release = threading.Event()

    def _slow_task():
        release.wait(timeout=5.0)

    worker = AsyncSaveWorker()
    try:
        worker.enqueue(_slow_task)  # occupies the background thread
        for _ in range(5):
            worker.enqueue(lambda: None)
        time.sleep(0.05)  # let the background thread pick up _slow_task
        assert worker.queue_depth() >= 5
    finally:
        release.set()
        worker.shutdown()
