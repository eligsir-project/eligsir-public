"""Tests for ExternalSLAMWorker (PR W3-04, plan section 9)."""

from __future__ import annotations

import time
from typing import Optional

import numpy as np
import pytest

from eligsir.contracts.types import Pose, RGBDFrame, TrackingResult
from eligsir.mapping.candidates.contracts import RawFramePacket, UndecodedBufferRef
from eligsir.slam.runtime.external_worker import ExternalSLAMWorker
from eligsir.slam.runtime.health import WorkerState
from eligsir.slam.runtime.worker import SubmitOutcome
from eligsir.utils.rgbd_frames import CameraIntrinsics


def _frame(index: int) -> RawFramePacket:
    rng = np.random.default_rng(index)
    return RawFramePacket(
        source_index=index,
        timestamp_ns=index * 33_000_000,
        rgb_ref=UndecodedBufferRef(array=rng.integers(0, 255, size=(8, 8, 3), dtype=np.uint8)),
        depth_ref=UndecodedBufferRef(array=np.ones((8, 8), dtype=np.float32)),
        intrinsics=CameraIntrinsics(width=8, height=8, fx=10.0, fy=10.0, cx=4.0, cy=4.0),
    )


class _FakeExternalBackend:
    """Stands in for ROSSLAMBackend: track() is a cheap read, optionally
    delayed to simulate a regression, or made to raise to test failure
    accounting."""

    name = "fake_external"

    def __init__(self, delay_s: float = 0.0, raise_on_track: bool = False):
        self._delay_s = delay_s
        self._raise = raise_on_track

    def initialize(self, frame: RGBDFrame) -> TrackingResult:
        return TrackingResult(pose=Pose.identity(), confidence=1.0, valid=True, method=self.name)

    def track(self, frame: RGBDFrame, T_pred: Optional[Pose] = None, gaussian_ctx=None) -> TrackingResult:
        if self._delay_s:
            time.sleep(self._delay_s)
        if self._raise:
            raise RuntimeError("simulated external backend failure")
        return TrackingResult(pose=Pose.identity(), confidence=1.0, valid=True, method=self.name)


def _wait_until(predicate, timeout_s: float = 5.0, interval_s: float = 0.01) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval_s)
    return predicate()


def test_submit_and_try_get_roundtrip():
    w = ExternalSLAMWorker(_FakeExternalBackend())
    try:
        result = w.submit(_frame(0))
        assert result.outcome is SubmitOutcome.ACCEPTED

        tracked = None
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and tracked is None:
            tracked = w.try_get()
            if tracked is None:
                time.sleep(0.01)
        assert tracked is not None
        assert tracked.frame.source_index == 0
        assert tracked.tracking_confidence == 1.0
    finally:
        w.close()


def test_slow_backend_track_does_not_block_caller_thread():
    # Even a regression that makes the wrapped "already cheap" backend
    # slow must stay on ExternalSLAMWorker's own thread -- submit()/
    # try_get() on the caller's thread must return immediately regardless.
    w = ExternalSLAMWorker(_FakeExternalBackend(delay_s=0.5))
    try:
        w.submit(_frame(0))
        time.sleep(0.05)

        t0 = time.perf_counter()
        w.submit(_frame(1))
        submit_elapsed = time.perf_counter() - t0
        t0 = time.perf_counter()
        w.try_get()
        tryget_elapsed = time.perf_counter() - t0

        assert submit_elapsed < 0.1
        assert tryget_elapsed < 0.1
    finally:
        w.close()


def test_backend_failure_is_counted_not_raised():
    w = ExternalSLAMWorker(_FakeExternalBackend(raise_on_track=True))
    try:
        # First call goes through initialize(), which succeeds; second
        # goes through track(), which raises inside the worker thread.
        w.submit(_frame(0))
        _wait_until(lambda: w.try_get() is not None)
        w.submit(_frame(1))
        assert _wait_until(lambda: w.health().consecutive_failures > 0)
        health = w.health()
        assert health.state is WorkerState.DEGRADED
        assert health.last_error is not None
        # try_get() must never raise even though track() did.
        assert w.try_get() is None
    finally:
        w.close()


def test_close_is_idempotent_and_bounded():
    w = ExternalSLAMWorker(_FakeExternalBackend())
    t0 = time.perf_counter()
    w.close()
    w.close()  # must not raise / hang the second time
    assert time.perf_counter() - t0 < 5.0
    assert w.health().state is WorkerState.CLOSED
