"""Deliverable #7 / acceptance-gate evidence (plan section 8.7/8.8):
"Synthetic 30 FPS all-rejected arm is within 3% of no-source baseline."

Measures the OPTIMIZER-THREAD-SIDE cost only (the thing the deliverable
actually cares about: does a 30 FPS rejected-everything source measurably
slow down the optimizer loop), by running a tight optimize_step() loop
that only ever does a non-blocking poll, while a real producer thread (30
FPS) + a real FramePreparationWorker thread (doing full CPU
decode/quality/novelty prep, then having every candidate rejected by
prefilter.decide()) run concurrently on their own threads.

This deliberately does NOT reuse
eligsir.runtime.profiling.benchmark_scenarios.run_scenario's
`workload.admit` hook, because that hook runs synchronously on the
harness's own single consumption thread -- a fundamentally different
(and, for this deliverable, wrong) shape than this PR's actual
producer/worker/optimizer three-thread design. It reuses the same
synthetic source (eligsir.io.sources.synthetic) as W3-01's harness for
a reproducible, camera-free workload.

Run inside the container:
  docker compose run --rm train python /workspace/eligsir/w3_03_gate7_benchmark.py
"""
from __future__ import annotations

import json
import sys
import threading
import time

import numpy as np

sys.path.insert(0, "/workspace/eligsir")

from eligsir.io.sources.synthetic import SyntheticRGBDFrameSource, SyntheticSourceConfig
from eligsir.mapping.candidates.contracts import (
    RawFramePacket,
    TrackedFramePacket,
    TrackingState,
    UndecodedBufferRef,
)
from eligsir.mapping.candidates.prefilter import PrefilterConfig, decide
from eligsir.runtime.queues.bounded_queue import BoundedQueue, QueuePollState
from eligsir.runtime.workers.frame_preparation_worker import (
    FramePreparationWorker,
    FramePreparationWorkerConfig,
)

TARGET_DURATION_S = 4.0
FPS = 30.0
NUM_FRAMES = 4 * int(FPS) + 60  # comfortably covers TARGET_DURATION_S at 30 FPS

_A = np.random.default_rng(0).standard_normal((192, 192)).astype(np.float32)
_B = np.random.default_rng(1).standard_normal((192, 192)).astype(np.float32)


def _optimize_step(out_q: BoundedQueue | None) -> None:
    """A ~sub-millisecond CPU-only stand-in for one optimizer iteration's
    non-GPU-bound work (real GPU torch steps are hard to make
    representative in a CPU-only container harness; a modest matmul gives
    each iteration enough real cost that GIL-contention noise from the
    background threads doesn't dominate the measurement, per this
    deliverable's actual concern: does the OPTIMIZER LOOP slow down)."""
    if out_q is not None:
        polled = out_q.try_poll()
        if polled.state is QueuePollState.ITEM:
            decide(polled.item, PrefilterConfig(min_quality_weight=2.0))  # impossible -> always rejected
    _ = float((_A @ _B).sum())


def _optimizer_loop_iters_per_second(out_q: BoundedQueue | None, duration_s: float) -> float:
    n = 0
    t_end = time.perf_counter() + duration_s
    t0 = time.perf_counter()
    while time.perf_counter() < t_end:
        _optimize_step(out_q)
        n += 1
    elapsed = time.perf_counter() - t0
    return n / elapsed


def _run_no_source_baseline() -> float:
    return _optimizer_loop_iters_per_second(None, TARGET_DURATION_S)


def _run_all_rejected_arm() -> float:
    in_q: BoundedQueue = BoundedQueue(capacity=8)
    out_q: BoundedQueue = BoundedQueue(capacity=8)
    worker = FramePreparationWorker(
        in_q, out_q,
        config=FramePreparationWorkerConfig(prefilter=PrefilterConfig(min_pose_novelty=0.0)),
    ).start()

    stop = threading.Event()

    def produce() -> None:
        source = SyntheticRGBDFrameSource(
            SyntheticSourceConfig(seed=0, fps=FPS, num_frames=NUM_FRAMES, realtime=True)
        )
        for frame in source:
            if stop.is_set():
                return
            raw = RawFramePacket(
                source_index=frame.index, timestamp_ns=int(frame.timestamp * 1e9),
                rgb_ref=UndecodedBufferRef(array=frame.rgb), depth_ref=UndecodedBufferRef(array=frame.depth),
                intrinsics=frame.intrinsics,
            )
            tracked = TrackedFramePacket(
                frame=raw, world_T_camera=frame.c2w, tracking_state=TrackingState.OK,
                tracking_confidence=1.0, slam_map_id=0,
            )
            in_q.try_offer(tracked)

    producer = threading.Thread(target=produce, daemon=True)
    producer.start()
    try:
        ips = _optimizer_loop_iters_per_second(out_q, TARGET_DURATION_S)
    finally:
        stop.set()
        worker.stop(join_timeout_s=2.0)
        producer.join(timeout=1.0)
    return ips


def main() -> None:
    # Warm up (JIT-ish numpy/BLAS thread-pool warmup, page faults) so the
    # first timed arm isn't penalized relative to the second.
    _optimizer_loop_iters_per_second(None, 0.5)

    baseline = _run_no_source_baseline()
    arm = _run_all_rejected_arm()
    delta_pct = 100.0 * abs(arm - baseline) / max(baseline, 1e-9)
    result = {
        "duration_s": TARGET_DURATION_S,
        "no_source_activity_iters_per_s": baseline,
        "all_rejected_30fps_iters_per_s": arm,
        "delta_pct": delta_pct,
        "within_3pct": delta_pct <= 3.0,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
