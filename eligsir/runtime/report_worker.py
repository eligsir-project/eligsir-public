"""Async report worker: background thread for mid-training benchmark reports."""
from __future__ import annotations

import threading


class AsyncReportWorker:
    """Runs one report-generation callable at a time on a background daemon
    thread. If a new report is submitted while one is still running, it is
    skipped (not queued) -- mid-training reports are periodic snapshots of
    training progress, not events that must all eventually run, so it's
    correct to drop a tick rather than build an ever-growing backlog of
    stale reports.
    """

    def __init__(self) -> None:
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def try_submit(self, fn, on_skip=None) -> bool:
        """Attempt to run fn() on a background thread.

        Returns True if submitted, False if a previous report is still
        running (in which case on_skip(), if given, is called synchronously
        on the caller's thread before returning False).
        """
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                if on_skip is not None:
                    on_skip()
                return False
            self._thread = threading.Thread(target=self._run, args=(fn,), daemon=True)
            self._thread.start()
            return True

    def _run(self, fn) -> None:
        try:
            fn()
        except Exception:
            import traceback

            traceback.print_exc()

    def join(self, timeout_s: float = 120.0) -> None:
        """Block until any in-flight report finishes.

        Used at end-of-training so the process doesn't exit while a report
        from the last mid-training tick is still writing files.
        """
        t = self._thread
        if t is not None:
            t.join(timeout=timeout_s)
