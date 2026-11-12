"""Tests for the SLAMWorker protocol shape and WorkerHealth (PR W3-04,
plan section 9)."""

from __future__ import annotations

from eligsir.slam.runtime.health import (
    GPU_CONTENTION_DISCLAIMER,
    WorkerHealth,
    WorkerState,
)
from eligsir.slam.runtime.worker import SLAMWorker, SubmitOutcome, SubmitResult


def test_submit_result_fields():
    r = SubmitResult(outcome=SubmitOutcome.ACCEPTED, submitted_at_ns=123)
    assert r.outcome is SubmitOutcome.ACCEPTED
    assert r.submitted_at_ns == 123


def test_worker_health_carries_isolation_disclaimer_by_default():
    # Deliverable 6: every WorkerHealth snapshot carries the shared-GPU
    # contention caveat, whether or not the caller thinks to ask for it.
    h = WorkerHealth(state=WorkerState.RUNNING, backend_name="dummy")
    assert h.isolation_note == GPU_CONTENTION_DISCLAIMER
    assert "shared" in h.isolation_note.lower()


def test_worker_health_is_usable_property():
    assert WorkerHealth(state=WorkerState.RUNNING, backend_name="d").is_usable
    assert WorkerHealth(state=WorkerState.DEGRADED, backend_name="d").is_usable
    assert not WorkerHealth(state=WorkerState.FAILED, backend_name="d").is_usable
    assert not WorkerHealth(state=WorkerState.CLOSED, backend_name="d").is_usable


def test_process_worker_and_external_worker_satisfy_protocol():
    # Structural (runtime_checkable) protocol conformance -- both
    # implementations must expose submit/try_get/health/close.
    from eligsir.slam.runtime.external_worker import ExternalSLAMWorker
    from eligsir.slam.runtime.process_worker import ProcessSLAMWorker

    for cls in (ProcessSLAMWorker, ExternalSLAMWorker):
        for method in ("submit", "try_get", "health", "close"):
            assert hasattr(cls, method), f"{cls.__name__} missing {method}"
