"""Background CPU preparation worker for the prepared-candidate pipeline
(plan section 8.4, "Create a CPU/pinned-memory preparation worker").

Consumes :class:`~eligsir.mapping.candidates.contracts.TrackedFramePacket`
from a bounded input queue on its own thread, applies the pre-decode
prefilter gate (``prefilter.should_skip_decode``) so an obviously-stale
pose never pays a decode cost, otherwise runs full CPU preparation
(``preparation.prepare_candidate``: decode, resize, blur/exposure/
valid-depth-fraction, cheap pose novelty), and offers the resulting
:class:`~eligsir.mapping.candidates.contracts.PreparedFrameCandidate`
onto a bounded output queue using the latest-wins policy (see
``eligsir.runtime.queues.policies``).

Named/shaped after the existing ``eligsir.core.ingestion_worker.
IngestionWorker`` (same single-writer-handoff, non-blocking-poll,
own-background-thread design, proven safe under ``--streaming_async_ingest``)
but built on the typed queue contracts from this PR instead of a bare
``queue.Queue``, and doing admission-relevant quality/novelty scoring
inline rather than only warming the decode cache -- so a consumer never
needs to touch a raw frame's buffers to find out whether it is worth
building a training ``Camera`` for.

Everything on this thread is CPU/numpy-only; nothing here touches torch or
the GPU (deliverable #3 -- no full-resolution GPU upload for a rejected
candidate, since this worker never uploads anything at all).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Optional

import numpy as np

from eligsir.mapping.candidates.contracts import (
    PreparedFrameCandidate,
    TrackedFramePacket,
)
from eligsir.mapping.candidates.prefilter import PrefilterConfig, should_skip_decode
from eligsir.mapping.candidates.preparation import prepare_candidate
from eligsir.mapping.scheduler.frame_quality import FrameQualityConfig
from eligsir.runtime.queues.bounded_queue import BoundedQueue, QueuePollState
from eligsir.runtime.queues.policies import offer_latest_wins


@dataclass(frozen=True)
class FramePreparationWorkerConfig:
    """Tunables threaded straight through to ``prefilter``/``preparation``."""

    prefilter: PrefilterConfig = PrefilterConfig()
    quality: Optional[FrameQualityConfig] = None
    poll_interval_s: float = 0.002


class FramePreparationWorker:
    """Owns exactly one background thread. The producer side (whatever
    feeds ``input_queue``) and the coordinator's consumer side (whatever
    polls ``output_queue``) both stay on their own threads -- this class
    only runs the middle CPU-preparation stage.
    """

    def __init__(
        self,
        input_queue: BoundedQueue[TrackedFramePacket],
        output_queue: BoundedQueue[PreparedFrameCandidate],
        *,
        config: Optional[FramePreparationWorkerConfig] = None,
        on_error: Optional[callable] = None,
    ) -> None:
        self._input_queue = input_queue
        self._output_queue = output_queue
        self._config = config or FramePreparationWorkerConfig()
        self._on_error = on_error
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._previous_c2w: Optional[np.ndarray] = None
        # Diagnostics only (single-writer: this thread), read without a
        # lock like IngestionWorker's n_prepared/n_consumed -- plain int
        # increments are atomic enough for telemetry under CPython's GIL.
        self.n_prepared = 0
        self.n_skipped_decode = 0
        self.n_evicted_stale = 0

    def start(self) -> "FramePreparationWorker":
        if self._thread is not None:
            return self
        self._thread = threading.Thread(
            target=self._run, name="frame-preparation-worker", daemon=True
        )
        self._thread.start()
        return self

    def stop(self, *, join_timeout_s: float = 2.0) -> None:
        self._stop.set()
        t = self._thread
        if t is not None:
            t.join(timeout=join_timeout_s)
        self._thread = None

    def _run(self) -> None:
        while not self._stop.is_set():
            polled = self._input_queue.try_poll()
            if polled.state is QueuePollState.EMPTY:
                self._stop.wait(self._config.poll_interval_s)
                continue
            if polled.state in (QueuePollState.CLOSED, QueuePollState.FAILED):
                # Producer is done or has failed: propagate the same
                # terminal state to the output queue so a downstream
                # non-blocking poller learns about it too, then exit --
                # this thread has nothing left to do.
                if polled.state is QueuePollState.FAILED and polled.error is not None:
                    self._output_queue.fail(polled.error)
                else:
                    self._output_queue.close()
                return

            tracked = polled.item
            assert tracked is not None
            try:
                self._prepare_one(tracked)
            except Exception as exc:  # a bad frame must not kill the worker
                if self._on_error is not None:
                    self._on_error(exc)
                continue

    def _prepare_one(self, tracked: TrackedFramePacket) -> None:
        skip = should_skip_decode(tracked, self._previous_c2w, self._config.prefilter)
        if not skip.accept:
            self.n_skipped_decode += 1
            return

        candidate = prepare_candidate(
            tracked,
            previous_c2w=self._previous_c2w,
            quality_config=self._config.quality,
            novelty_translation_norm_m=self._config.prefilter.novelty_translation_norm_m,
            novelty_rotation_norm_deg=self._config.prefilter.novelty_rotation_norm_deg,
        )
        self._previous_c2w = np.asarray(tracked.world_T_camera, dtype=np.float64)
        self.n_prepared += 1

        offered = offer_latest_wins(self._output_queue, candidate)
        if offered.evicted is not None:
            self.n_evicted_stale += 1
