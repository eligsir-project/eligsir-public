"""Async ingestion worker — parallelize per-frame CPU prep off the optimizer thread.

Measured motivation (2026-06-30, kitchen1 live): during frame ingestion the iGPU
sits at ~60% (≈40% idle) while ``train.py`` runs ~146% CPU — the GPU is starved
because per-frame ingestion work (``frame_to_camera``, candidate collection, host
syncs) runs *on the optimizer thread between GPU steps*. Once frames stop, the GPU
is fed continuously at ~86%. Moving the CPU prep to a background thread lets the
optimizer keep the GPU busy while the next frame is prepared in parallel.

Design — single-writer handoff, NO lock on the model:
  * The worker thread runs a user-supplied ``prep_fn(raw)`` (CPU-only: frame→camera,
    candidate point collection via ``insert_gaussians_from_frame(_collect_only=True)``)
    and pushes the *read-only* prepared item onto a bounded queue.
  * The optimizer thread is the SOLE mutator of params / optimizer-state / occupancy:
    it pops prepared items and commits them (admission render + add_points + scheduler
    tick). It never blocks — ``poll()`` is non-blocking; if nothing is ready it trains
    on buffered data (the non-blocking invariant).
  * The bounded queue applies back-pressure (the worker waits when full) so frames are
    DEFERRED, never dropped → full-coverage preserved.

This module is intentionally model-agnostic: it knows nothing about Gaussians. The
caller supplies ``source_fn`` (pull next raw frame, non-blocking → None when none
ready) and ``prep_fn`` (CPU prep of a raw frame → prepared item). Default-off; when
unused this code never runs.
"""

from __future__ import annotations

import queue
import threading
from typing import Any, Callable, Optional


class IngestionWorker:
    """Background thread that pre-runs CPU prep for upcoming frames.

    Parameters
    ----------
    source_fn:
        Called on the worker thread to pull the next raw frame. MUST be
        non-blocking and return ``None`` when no frame is currently ready
        (the worker then sleeps briefly and retries). This is typically a
        thin wrapper over the live pump's ``has_ready_frame``/``get`` pair.
    prep_fn:
        Called on the worker thread with a raw frame; returns the prepared,
        commit-ready item (e.g. ``(camera, candidate_dict, stats)``). MUST be
        CPU/GPU-read-only — it must NOT mutate the Gaussian model, optimizer
        state, or occupancy (those are committed by the optimizer thread).
    depth:
        Bounded ready-queue size. Back-pressure defers prep when full.
    poll_interval_s:
        Idle sleep when ``source_fn`` returns None (no frame ready).
    """

    def __init__(
        self,
        source_fn: Callable[[], Optional[Any]],
        prep_fn: Callable[[Any], Optional[Any]],
        *,
        depth: int = 4,
        poll_interval_s: float = 0.002,
        on_error: Optional[Callable[[BaseException], None]] = None,
    ) -> None:
        self._source_fn = source_fn
        self._prep_fn = prep_fn
        self._q: "queue.Queue[Any]" = queue.Queue(maxsize=max(1, int(depth)))
        self._poll = float(poll_interval_s)
        self._on_error = on_error
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        # Diagnostics (read without lock; ints are atomic enough for telemetry).
        self.n_prepared = 0
        self.n_consumed = 0

    # -- lifecycle ---------------------------------------------------------
    def start(self) -> "IngestionWorker":
        """Start the background ingestion worker thread.

        Returns:
            self, for method chaining.
        """
        if self._thread is not None:
            return self
        self._thread = threading.Thread(
            target=self._run, name="ingestion-worker", daemon=True
        )
        self._thread.start()
        return self

    def stop(self, *, join_timeout_s: float = 2.0) -> None:
        """Stop the background worker thread.

        Args:
            join_timeout_s: Maximum time (seconds) to wait for the worker
                           thread to finish. If timeout expires, the thread
                           may still be running (daemon mode).
        """
        self._stop.set()
        t = self._thread
        if t is not None:
            t.join(timeout=join_timeout_s)
        self._thread = None

    # -- optimizer-thread API (non-blocking) -------------------------------
    def poll(self) -> Optional[Any]:
        """Return one prepared item if ready, else None. Never blocks."""
        try:
            item = self._q.get_nowait()
        except queue.Empty:
            return None
        self.n_consumed += 1
        return item

    @property
    def ready_count(self) -> int:
        """Number of prepared items currently waiting in the ready queue."""
        return self._q.qsize()

    # -- worker thread -----------------------------------------------------
    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                raw = self._source_fn()
            except Exception as exc:  # source hiccup must not kill the worker
                if self._on_error is not None:
                    self._on_error(exc)
                self._stop.wait(self._poll)
                continue
            if raw is None:
                # No frame ready — back off briefly and retry (non-blocking source).
                self._stop.wait(self._poll)
                continue
            try:
                item = self._prep_fn(raw)
            except Exception as exc:
                if self._on_error is not None:
                    self._on_error(exc)
                continue
            if item is None:
                continue
            # Bounded put with back-pressure: block (with periodic stop checks) so
            # frames are DEFERRED, never dropped. Coverage preserved.
            while not self._stop.is_set():
                try:
                    self._q.put(item, timeout=self._poll)
                    self.n_prepared += 1
                    break
                except queue.Full:
                    continue
