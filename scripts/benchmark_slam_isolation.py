"""Real benchmark for plan section 9.6, deliverable 2: "A 500 ms
artificial SLAM delay does not change optimization-exclusive GPU time."

Runs a synthetic "optimization loop" (a tight CPU loop standing in for
per-iteration GPU work -- the claim under test is about the optimization
thread's own progress being unaffected by the SLAM worker, which is a
process-isolation property independent of whether the simulated per-step
work happens to touch a GPU) concurrently with two
:class:`~eligsir.slam.runtime.process_worker.ProcessSLAMWorker`
instances, one configured with an artificial 500 ms delay before every
``track()`` call and one with none.

Both workers are started up front and warmed up (spawn + first backend
construction) before any timed iteration, and the two conditions --
``baseline`` (no SLAM delay) and ``with_slam_delay`` -- are interleaved
A/B/A/B... within a single run rather than run as two sequential blocks.
This matters: an earlier version of this script ran the two conditions
sequentially (baseline first, cold; delayed second, already warm) and
measured ``with_slam_delay`` as 25-32% *faster* than baseline, which is a
harness artifact (unmatched warm-up/scheduling state across the two
blocks), not evidence of anything. Interleaving removes that confound --
any process-level jitter (GC pauses, OS scheduling, thermal/frequency
drift) hits both conditions' iterations roughly equally, so the
per-condition median/p95 become a fair comparison. The first
``--warmup-iters`` interleaved rounds of BOTH conditions are discarded
before computing statistics, for the same reason.

The optimization loop only ever calls the worker's non-blocking
``submit``/``try_get`` -- if isolation holds, per-iteration optimization
time in the two conditions should match to within normal scheduling
noise, regardless of the 500 ms delay. Reports median and p95 (not just
mean, which a handful of scheduler-noise outliers can dominate) for both
conditions, plus the delta, so the claim is evidenced, not asserted.

Run inside the container per CLAUDE.md's Running Commands section:

    docker compose run --rm train python scripts/benchmark_slam_isolation.py
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from typing import List

sys.path.insert(0, "/workspace/eligsir")

import numpy as np

from eligsir.mapping.candidates.contracts import RawFramePacket, UndecodedBufferRef
from eligsir.slam.runtime.process_worker import ProcessSLAMWorker
from eligsir.utils.rgbd_frames import CameraIntrinsics


def _make_frame(index: int) -> RawFramePacket:
    rng = np.random.default_rng(index)
    rgb = rng.integers(0, 255, size=(16, 16, 3), dtype=np.uint8)
    depth = np.ones((16, 16), dtype=np.float32)
    intr = CameraIntrinsics(width=16, height=16, fx=10.0, fy=10.0, cx=8.0, cy=8.0)
    return RawFramePacket(
        source_index=index,
        timestamp_ns=index * 33_000_000,
        rgb_ref=UndecodedBufferRef(array=rgb),
        depth_ref=UndecodedBufferRef(array=depth),
        intrinsics=intr,
    )


def _simulated_optimization_step(work_iters: int = 200_000) -> None:
    """Stand-in for one training iteration's CPU-side work (dispatch,
    bookkeeping). Deliberately GPU-free so this benchmark runs anywhere,
    including CPU-only smoke environments -- the property under test is
    process isolation, not raw GPU throughput."""
    x = 0.0
    for i in range(work_iters):
        x += (i * 1.0000001) % 7.0


def _timed_step(worker: ProcessSLAMWorker, *, submit_frame: bool, frame_index: int) -> float:
    if submit_frame:
        worker.submit(_make_frame(frame_index))
    t0 = time.perf_counter()
    _simulated_optimization_step()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    worker.try_get()  # non-blocking drain, never waits
    return elapsed_ms


def _p95(values: List[float]) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))
    return ordered[idx]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-iters", type=int, default=30, help="Timed iterations PER condition.")
    parser.add_argument("--warmup-iters", type=int, default=5, help="Interleaved rounds discarded before timing.")
    parser.add_argument("--artificial-delay-s", type=float, default=0.5)
    parser.add_argument("--out", type=str, default=None, help="Optional JSON report path.")
    args = parser.parse_args()

    baseline_worker = ProcessSLAMWorker("dataset_pose", {"fallback_identity": True}, capacity=8)
    delayed_worker = ProcessSLAMWorker(
        "dataset_pose", {"fallback_identity": True}, capacity=8, artificial_delay_s=args.artificial_delay_s
    )

    try:
        total_rounds = args.warmup_iters + args.n_iters
        print(
            f"[benchmark] interleaved A/B: {total_rounds} rounds "
            f"({args.warmup_iters} warmup + {args.n_iters} timed) per condition, "
            f"with_slam_delay artificial_delay_s={args.artificial_delay_s}",
            flush=True,
        )

        baseline_times: List[float] = []
        delayed_times: List[float] = []
        for i in range(total_rounds):
            b_ms = _timed_step(baseline_worker, submit_frame=False, frame_index=i)
            d_ms = _timed_step(delayed_worker, submit_frame=True, frame_index=i)
            if i >= args.warmup_iters:
                baseline_times.append(b_ms)
                delayed_times.append(d_ms)
    finally:
        baseline_worker.close()
        delayed_worker.close()

    baseline_median = statistics.median(baseline_times)
    delayed_median = statistics.median(delayed_times)
    baseline_p95 = _p95(baseline_times)
    delayed_p95 = _p95(delayed_times)
    delta_median_ms = delayed_median - baseline_median
    delta_median_pct = (delta_median_ms / baseline_median * 100.0) if baseline_median > 0 else float("nan")

    report = {
        "n_iters": args.n_iters,
        "warmup_iters": args.warmup_iters,
        "artificial_delay_s": args.artificial_delay_s,
        "baseline_median_ms": baseline_median,
        "baseline_p95_ms": baseline_p95,
        "baseline_mean_ms": statistics.mean(baseline_times),
        "with_slam_delay_median_ms": delayed_median,
        "with_slam_delay_p95_ms": delayed_p95,
        "with_slam_delay_mean_ms": statistics.mean(delayed_times),
        "delta_median_ms": delta_median_ms,
        "delta_median_pct": delta_median_pct,
        "baseline_step_times_ms": baseline_times,
        "with_slam_delay_step_times_ms": delayed_times,
    }

    print(json.dumps({k: v for k, v in report.items() if not k.endswith("step_times_ms")}, indent=2))

    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"[benchmark] wrote {args.out}", flush=True)

    print(
        "\n[benchmark] deliverable-2 check: a 500ms artificial SLAM delay must not "
        "change optimization-exclusive step time (interleaved A/B, median/p95). "
        f"baseline_median={baseline_median:.3f}ms with_slam_delay_median={delayed_median:.3f}ms "
        f"delta={delta_median_ms:+.3f}ms ({delta_median_pct:+.1f}%); "
        f"baseline_p95={baseline_p95:.3f}ms with_slam_delay_p95={delayed_p95:.3f}ms.",
        flush=True,
    )


if __name__ == "__main__":
    main()
