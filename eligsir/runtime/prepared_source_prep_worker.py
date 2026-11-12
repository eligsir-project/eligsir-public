"""Asynchronous device-preparation owner for prepared-source ingress."""

from __future__ import annotations

import threading

from eligsir.runtime.prepared_runtime_bridge import (
    PreparedRuntimeBridge,
    PreparedRuntimeBridgeResult,
)
from eligsir.runtime.prepared_cold_promotion import PreparedColdPromotionOwner


# TransferEvent completion is poll-only; until it has a completion notification,
# consumers must recheck retryable prepared work at this bounded interval.
READY_WAIT_FALLBACK_S = 0.001

_EMPTY_RESULT = PreparedRuntimeBridgeResult(0, 0, 0, 0)


def _combine(
    left: PreparedRuntimeBridgeResult, right: PreparedRuntimeBridgeResult
) -> PreparedRuntimeBridgeResult:
    return PreparedRuntimeBridgeResult(
        left.submitted + right.submitted,
        left.enqueued + right.enqueued,
        left.deferred + right.deferred,
        left.dropped + right.dropped,
        left.retained_not_resident + right.retained_not_resident,
    )


class PreparedSourcePrepWorker:
    """Own only H2D/preparation/READY work outside the mapping thread."""

    def __init__(
        self,
        bridge: PreparedRuntimeBridge,
        *,
        poll_interval_s: float = 0.05,
        promotion_owner: PreparedColdPromotionOwner | None = None,
    ) -> None:
        if not isinstance(bridge, PreparedRuntimeBridge) and not callable(
            getattr(bridge, "drain", None)
        ):
            raise TypeError("bridge must provide a callable drain()")
        if not isinstance(poll_interval_s, (int, float)) or isinstance(
            poll_interval_s, bool
        ) or poll_interval_s < 0.0:
            raise ValueError("poll_interval_s must be a non-negative number")
        if promotion_owner is not None and not callable(
            getattr(promotion_owner, "pump_once", None)
        ):
            raise TypeError("promotion_owner must provide a callable pump_once() or be None")
        self._bridge = bridge
        self._poll_interval_s = float(poll_interval_s)
        self._promotion_owner = promotion_owner
        self._lock = threading.Lock()
        self._progress = threading.Condition(self._lock)
        self._progress_generation = 0
        self._stop = threading.Event()
        self._pending_result = _EMPTY_RESULT
        self._error: BaseException | None = None
        self._thread: threading.Thread | None = None
        # Register progress callback with handoff (ingress), promotion owner (COLD work),
        # and ready_owner (READY publication). The callback increments generation and
        # notifies under the lock, enabling EDGE-triggered wakeup on commit/enqueue/
        # READY-transition events.
        if hasattr(self._bridge, "handoff") and hasattr(
            self._bridge.handoff, "set_progress_callback"
        ):
            self._bridge.handoff.set_progress_callback(self._signal_progress)
        if (
            self._promotion_owner is not None
            and hasattr(self._promotion_owner, "set_progress_callback")
        ):
            self._promotion_owner.set_progress_callback(self._signal_progress)
        if hasattr(self._bridge, "ready_owner") and hasattr(
            self._bridge.ready_owner, "set_progress_callback"
        ):
            self._bridge.ready_owner.set_progress_callback(self._signal_progress)

    def _signal_progress(self) -> None:
        """Called by handoff when ingress is committed. Increments generation under lock."""
        with self._progress:
            self._progress_generation += 1
            self._progress.notify_all()

    @property
    def running(self) -> bool:
        thread = self._thread
        return bool(thread is not None and thread.is_alive())

    def check_health(self) -> None:
        with self._lock:
            error = self._error
        if error is not None:
            raise error

    def pump_once(self) -> bool:
        """Return whether ingress or one COLD promotion made progress."""

        result = self._bridge.drain()
        if not isinstance(result, PreparedRuntimeBridgeResult):
            raise TypeError("bridge.drain() must return PreparedRuntimeBridgeResult")
        promotion_progress = bool(
            self._promotion_owner is not None and self._promotion_owner.pump_once()
        )
        if result.submitted:
            with self._lock:
                self._pending_result = _combine(self._pending_result, result)
        spill_progress = bool(getattr(self._bridge, "last_background_progress", False))
        progressed = bool(result.submitted > result.deferred or promotion_progress or spill_progress)
        if progressed:
            with self._progress:
                self._progress_generation += 1
                self._progress.notify_all()
        return progressed

    def wait_for_progress(self, *, timeout_s: float) -> bool:
        """Yield until preparation/page-in advances, without waiting on a device event."""

        if not isinstance(timeout_s, (int, float)) or isinstance(timeout_s, bool):
            raise TypeError("timeout_s must be a non-negative number")
        if timeout_s < 0.0:
            raise ValueError("timeout_s must be a non-negative number")
        self.check_health()
        with self._progress:
            generation = self._progress_generation
            self._progress.wait_for(
                lambda: self._progress_generation != generation or self._error is not None,
                timeout=float(timeout_s),
            )
        self.check_health()
        return self._progress_generation != generation

    def take_result(self) -> PreparedRuntimeBridgeResult:
        self.check_health()
        with self._lock:
            result, self._pending_result = self._pending_result, _EMPTY_RESULT
        return result

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                # Snapshot generation before pump so we can detect concurrent ingress commits.
                with self._progress:
                    before_pump = self._progress_generation
                progressed = self.pump_once()
            except BaseException as error:  # noqa: BLE001 - surface asynchronously
                with self._lock:
                    self._error = error
                self._stop.set()
                return
            if not progressed:
                # Wait for progress: generation increments on ingress commits or COLD promotion
                # enqueues (EDGE-triggered via callback). The snapshot before pump ensures a
                # commit landing between pump returning False and wait acquiring the lock is
                # detected by the predicate. Blocked-but-queued work falls back to the bounded
                # 50ms timeout (recovery-only, not steady-state).
                with self._progress:
                    self._progress.wait_for(
                        lambda: (
                            self._progress_generation != before_pump
                            or self._stop.is_set()
                        ),
                        timeout=self._poll_interval_s,
                    )

    def start(self) -> "PreparedSourcePrepWorker":
        if self.running:
            raise RuntimeError("prepared source prep worker is already running")
        if self._stop.is_set():
            raise RuntimeError("prepared source prep worker cannot restart")
        self._thread = threading.Thread(
            target=self._run,
            name="prepared-source-prep",
            daemon=True,
        )
        self._thread.start()
        return self

    def stop(self, *, timeout_s: float = 5.0) -> None:
        if not isinstance(timeout_s, (int, float)) or isinstance(timeout_s, bool):
            raise TypeError("timeout_s must be a non-negative number")
        if timeout_s < 0.0:
            raise ValueError("timeout_s must be a non-negative number")
        self._stop.set()
        with self._progress:
            self._progress.notify_all()
        thread = self._thread
        if thread is not None:
            thread.join(float(timeout_s))
            if thread.is_alive():
                raise RuntimeError("prepared source prep worker did not stop")

    close = stop


__all__ = ["PreparedSourcePrepWorker"]
