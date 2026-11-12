"""Async save worker: background thread for PLY/checkpoint writes."""
from __future__ import annotations

import queue
import threading


class AsyncSaveWorker:
    """Enqueue callables to run on a background daemon thread.

    Used to write PLY snapshots and optimizer checkpoints without stalling
    the training loop. The queue is unbounded by default so enqueue() never
    blocks the calling (training) thread -- a bounded queue previously
    blocked once 4 writes were pending, a confirmed contributor to a
    mid-training stall. Callers wanting backlog visibility should poll
    queue_depth() (e.g. into a TensorBoard scalar) rather than relying on
    enqueue() blocking as an implicit backpressure signal.
    """

    def __init__(self, maxsize: int = 0, join_timeout_s: float = 120.0) -> None:
        self._queue: queue.Queue = queue.Queue(maxsize=maxsize)
        self._join_timeout_s = float(join_timeout_s)
        self._failure: Exception | None = None
        self._failure_lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while True:
            task = self._queue.get()
            if task is None:
                break
            try:
                task()
            except Exception as exc:
                import traceback

                traceback.print_exc()
                with self._failure_lock:
                    if self._failure is None:
                        self._failure = exc
            finally:
                self._queue.task_done()

    def _raise_if_failed(self) -> None:
        with self._failure_lock:
            failure = self._failure
        if failure is not None:
            raise RuntimeError("async save task failed") from failure

    def enqueue(self, fn) -> None:
        self._queue.put(fn)

    def queue_depth(self) -> int:
        """Approximate number of tasks currently queued (not yet started).

        Uses Queue.qsize(), which CPython documents as approximate under
        concurrent access -- fine here since this is telemetry, not a
        correctness-critical count.
        """
        return self._queue.qsize()

    def join(self) -> None:
        """Wait until every already-enqueued save has completed."""

        self._queue.join()
        self._raise_if_failed()

    def shutdown(self) -> None:
        self._queue.put(None)
        self._thread.join(timeout=self._join_timeout_s)
        if self._thread.is_alive():
            raise RuntimeError(
                f"async save worker did not stop within {self._join_timeout_s:g}s"
            )
        self._raise_if_failed()
