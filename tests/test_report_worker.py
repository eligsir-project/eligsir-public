"""Tests for AsyncReportWorker: runs at most one report job at a time on a
background thread, skips (does not queue) a new submission while one is
already running, and join() waits for any in-flight job to finish.
"""
import threading
import time

from eligsir.runtime.report_worker import AsyncReportWorker


def test_try_submit_runs_on_background_thread_and_returns_true():
    ran = threading.Event()
    worker = AsyncReportWorker()

    submitted = worker.try_submit(lambda: ran.set())
    assert submitted is True
    assert ran.wait(timeout=2.0)


def test_try_submit_skips_while_previous_job_still_running():
    release = threading.Event()
    started = threading.Event()

    def _slow_job():
        started.set()
        release.wait(timeout=5.0)

    worker = AsyncReportWorker()
    assert worker.try_submit(_slow_job) is True
    assert started.wait(timeout=2.0)

    skip_called = threading.Event()
    submitted_again = worker.try_submit(
        lambda: None, on_skip=lambda: skip_called.set()
    )

    assert submitted_again is False
    assert skip_called.is_set()

    release.set()


def test_join_waits_for_in_flight_job():
    release = threading.Event()
    finished = threading.Event()

    def _slow_job():
        release.wait(timeout=5.0)
        finished.set()

    worker = AsyncReportWorker()
    worker.try_submit(_slow_job)
    release.set()
    worker.join(timeout_s=2.0)
    assert finished.is_set()


def test_exception_in_job_does_not_propagate_to_caller():
    worker = AsyncReportWorker()

    def _bad_job():
        raise RuntimeError("boom")

    submitted = worker.try_submit(_bad_job)
    assert submitted is True
    worker.join(timeout_s=2.0)  # must not raise
