"""Off-thread preview consumer loop (issue #571, scoped).

Canonical design: ``docs/architecture/2026-08-14-canonical-live-viewer-sidecar.md``
§5 / §11. Parent program: #569.

``CanonicalViewer.drain_to_sinks`` and ``PreviewStagingRing.poll_completed``
are both explicitly documented as consumer-side, off-training-thread
operations (see their own docstrings). This module is that thread: it polls
the staging ring for completed GPU->CPU copies, turns them into
``TrainingPairPreview`` events, submits them to the bus, and drains every
configured sink -- all off the training thread, at a bounded poll rate.

The training thread's only interaction with this module is
``PreviewDrainThread.notify()`` after ``PreviewStagingRing.try_submit_pair``
returns (a plain, uncontended ``threading.Event.set()`` -- not a wait).
"""
from __future__ import annotations

import threading
import time
from typing import Callable, Optional

__all__ = ["PreviewDrainThread"]


class PreviewDrainThread:
    """Background loop: poll staging ring -> submit -> drain_to_sinks.

    ``poll_interval_s`` bounds worst-case preview latency; it is not a
    training deadline -- nothing on the training thread ever waits on this
    thread. Sink work (Viser SDK calls, IPC, encoding) happens entirely here,
    never on the training thread, satisfying the "no backend SDK work in the
    trainer" invariant for the primary 2D path.
    """

    def __init__(
        self,
        *,
        on_tick: Callable[[], None],
        poll_interval_s: float = 0.03,
    ) -> None:
        self._on_tick = on_tick
        self._poll_interval_s = max(0.001, float(poll_interval_s))
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._tick_errors = 0

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run, name="preview-drain", daemon=True
        )
        self._thread.start()

    def notify(self) -> None:
        """Called from the training thread; a cheap, non-blocking wake hint."""
        self._wake.set()

    def stop(self, *, join_timeout_s: float = 2.0) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout=join_timeout_s)
            self._thread = None

    def _run(self) -> None:
        while not self._stop.is_set():
            self._wake.wait(timeout=self._poll_interval_s)
            self._wake.clear()
            try:
                self._on_tick()
            except Exception:  # noqa: BLE001 - this thread must never crash training
                self._tick_errors += 1

    @property
    def tick_errors(self) -> int:
        return self._tick_errors
