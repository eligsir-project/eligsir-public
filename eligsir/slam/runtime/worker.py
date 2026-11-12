"""``SLAMWorker`` protocol (plan section 9.3/9.4).

One asynchronous interface for both kinds of SLAM backend this project
has:

- **In-process backends** (e.g. ``open3d_odometry_live`` --
  :class:`eligsir.slam.adapters.open3d_odometry.Open3DOdometryBackend``)
  whose ``track()`` call runs real, potentially slow computation on
  whatever thread calls it. Wrapped by
  :class:`eligsir.slam.runtime.process_worker.ProcessSLAMWorker`,
  which runs the backend in a real OS subprocess so its latency cannot
  block the optimization thread (Python's GIL means a background
  *thread* alone does not buy this for CPU-bound work -- see CLAUDE.md's
  "High-throughput, low-complexity data processing belongs in a fast
  language" note, which documents the exact same GIL-contention failure
  mode for rvizsplat's publish thread).
- **Already-external backends** (e.g. ``ros_slam`` /
  :class:`eligsir.slam.adapters.ros_slam.ROSSLAMBackend`, and
  ORB-SLAM3's dedicated ROS2 container) whose real SLAM computation
  already happens outside the calling process; ``track()`` there is a
  cheap, non-blocking read of the latest received pose. Wrapped by
  :class:`eligsir.slam.runtime.external_worker.ExternalSLAMWorker`,
  which still runs that cheap read on its own background thread (never
  the caller's thread) purely for interface uniformity and so a future
  regression in an "already cheap" backend's ``track()`` cannot silently
  reintroduce synchronous blocking.

Both implementations use the same non-blocking submit/poll shape as this
project's other runtime queues (see
``eligsir.runtime.queues.bounded_queue.BoundedQueue`` and
``eligsir.runtime.workers.frame_preparation_worker
.FramePreparationWorker``, from the already-merged W3-03 bounded-queue
work this PR builds on) rather than inventing a new one.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Protocol, runtime_checkable

from eligsir.mapping.candidates.contracts import RawFramePacket, TrackedFramePacket
from .health import WorkerHealth


class SubmitOutcome(str, Enum):
    """Outcome of a non-blocking :meth:`SLAMWorker.submit` call."""

    ACCEPTED = "accepted"
    REJECTED_QUEUE_FULL = "rejected_queue_full"
    REJECTED_WORKER_FAILED = "rejected_worker_failed"
    REJECTED_WORKER_CLOSED = "rejected_worker_closed"


@dataclass(frozen=True)
class SubmitResult:
    """Result of a :meth:`SLAMWorker.submit` call.

    Args:
        outcome: What happened to the submitted frame.
        submitted_at_ns: ``time.monotonic_ns()`` at submit time, so a
            caller can compute round-trip latency once the corresponding
            :class:`~eligsir.mapping.candidates.contracts
            .TrackedFramePacket` comes back through :meth:`try_get`
            (matched by ``frame.source_index``).
    """

    outcome: SubmitOutcome
    submitted_at_ns: int


@runtime_checkable
class SLAMWorker(Protocol):
    """One asynchronous interface for every production SLAM backend.

    Every method is non-blocking. The caller (the mapping/optimization
    loop) is expected to call :meth:`submit` once per arriving frame and
    poll :meth:`try_get` on its own schedule -- never block waiting for a
    result, mirroring ``BoundedQueue.try_offer``/``try_poll``.

    Frames may complete out of submission order is NOT guaranteed by this
    protocol in general (a worker MAY reorder), but the two
    implementations in this package (:mod:`process_worker`,
    :mod:`external_worker`) both preserve submission order, since neither
    SLAM backend they wrap benefits from reordering and callers (pose
    prediction, tracking-confidence smoothing) assume monotonic frame
    order.
    """

    def submit(self, frame: RawFramePacket) -> SubmitResult:
        """Hand ``frame`` to the worker without waiting for a result.

        Must never block on backend computation. May reject (queue full,
        worker failed/closed) -- a caller that gets a non-``ACCEPTED``
        outcome should treat this exactly like "no pose available yet",
        never raise past it (see deliverable 3, "SLAM failure cannot
        deadlock the mapper").
        """
        ...

    def try_get(self) -> Optional[TrackedFramePacket]:
        """Return the next completed tracking result, if one is ready.

        Non-blocking: returns ``None`` immediately when nothing is ready
        rather than waiting. A tracking *failure* on the worker side is
        not surfaced here as an exception -- it is silently dropped (the
        frame simply never appears) and instead counted in
        :meth:`health`'s ``consecutive_failures``/``last_error``, so a
        caller's poll loop can stay a plain non-blocking drain with no
        try/except needed.
        """
        ...

    def health(self) -> WorkerHealth:
        """Return a point-in-time :class:`~.health.WorkerHealth` snapshot."""
        ...

    def close(self) -> None:
        """Shut the worker down.

        Must be safe to call multiple times, and must never hang
        indefinitely on a wedged backend -- implementations enforce a
        bounded join/terminate timeout internally (see deliverable 3).
        """
        ...
