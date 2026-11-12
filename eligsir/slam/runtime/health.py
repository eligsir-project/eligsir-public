"""``WorkerHealth``/``WorkerState`` -- explicit health and shutdown states
for a :class:`~eligsir.slam.runtime.worker.SLAMWorker` (plan section
9.6, deliverable 6).

**Deliverable 6, spelled out explicitly (do not remove this note):**
process separation bounds *latency propagation* (a slow or hung SLAM
backend cannot stall the optimization thread's own progress), it does
NOT buy absolute resource isolation. This box runs a single shared AMD
iGPU with shared VRAM (see CLAUDE.md's ``amd-shared-vram-not-oom-
constraint`` and the R9700 GPU-starvation findings under co-residency
referenced there) -- a worker process still competes for the same GPU,
the same PCIe bus, and the same CPU cores as the optimization process.
``WorkerHealth.isolation_note`` below is the single, always-attached
string carrying this caveat so nothing downstream (a report, a log line,
a dashboard) can quote a "worker is isolated" claim without the caveat
riding along with it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

# Deliberately not "isolated" or "sandboxed" -- see module docstring.
GPU_CONTENTION_DISCLAIMER = (
    "Process separation bounds latency propagation only: a worker process "
    "cannot stall the optimization thread's forward progress. It does NOT "
    "guarantee resource isolation -- on this project's shared-iGPU hardware "
    "(single AMD GPU, shared VRAM, shared PCIe/CPU), a busy worker process "
    "still contends for real GPU/CPU throughput with the optimization "
    "process. Never read a healthy WorkerHealth as 'this backend cannot "
    "slow down training' -- only as 'this backend cannot deadlock or block "
    "training.'"
)


class WorkerState(str, Enum):
    """Coarse lifecycle state of a :class:`SLAMWorker` implementation."""

    STARTING = "starting"
    RUNNING = "running"
    DEGRADED = "degraded"  # alive, but recent submit/track failures observed
    FAILED = "failed"      # worker process/thread died or entered a terminal error
    CLOSED = "closed"      # close() was called and shutdown completed


@dataclass(frozen=True)
class WorkerHealth:
    """Point-in-time health snapshot returned by :meth:`SLAMWorker.health`.

    Args:
        state: Coarse lifecycle state.
        backend_name: The wrapped ``SLAMBackend.name`` (or equivalent),
            so a caller inspecting a bare ``WorkerHealth`` (e.g. in a log
            line or a report) can attribute it to a concrete backend.
        pending_submitted: Frames handed to :meth:`SLAMWorker.submit` that
            have not yet been retrieved via :meth:`SLAMWorker.try_get`
            (either still in flight, or a completed result waiting to be
            polled).
        completed_count: Cumulative count of results successfully
            retrieved via :meth:`try_get`.
        consecutive_failures: Consecutive track/initialize failures since
            the last success (0 while healthy). Distinct from a hard
            :attr:`WorkerState.FAILED` -- a backend can degrade (a few
            failed frames) without being unusable.
        last_error: ``repr()`` of the most recent failure, if any.
        last_result_age_ms: Wall-clock milliseconds since the last result
            was retrieved via :meth:`try_get`. ``None`` if none yet.
        isolation_note: Always :data:`GPU_CONTENTION_DISCLAIMER` -- see
            module docstring. Kept as an instance field (not just a
            module constant) so it survives serialization of a
            ``WorkerHealth`` snapshot into a report/log without a reader
            needing to know to go look up the module constant separately.
    """

    state: WorkerState
    backend_name: str
    pending_submitted: int = 0
    completed_count: int = 0
    consecutive_failures: int = 0
    last_error: Optional[str] = None
    last_result_age_ms: Optional[float] = None
    isolation_note: str = field(default=GPU_CONTENTION_DISCLAIMER)

    @property
    def is_usable(self) -> bool:
        """``True`` unless the worker is FAILED or CLOSED.

        DEGRADED is still usable -- a caller (the mapper) should keep
        calling ``try_get``/``submit`` and simply treat missing results as
        dropped tracking, per deliverable 3 ("SLAM failure cannot deadlock
        the mapper").
        """
        return self.state not in (WorkerState.FAILED, WorkerState.CLOSED)
