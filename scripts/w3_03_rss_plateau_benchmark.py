"""Acceptance-gate evidence (plan 8.8): "long-duration RSS reaches a
plateau." Runs the producer + FramePreparationWorker + non-blocking
optimizer poll loop for a real (short, container-budget-friendly) duration
and samples RSS periodically, reporting the growth rate in the first vs.
second half of the run -- a plateau shows up as the second-half growth
rate being near zero relative to the first half's.
"""
from __future__ import annotations

import json
import resource
import sys
import threading
import time

sys.path.insert(0, "/workspace/eligsir")

from eligsir.io.sources.synthetic import SyntheticRGBDFrameSource, SyntheticSourceConfig
from eligsir.mapping.candidates.contracts import (
    RawFramePacket,
    TrackedFramePacket,
    TrackingState,
    UndecodedBufferRef,
)
from eligsir.mapping.candidates.prefilter import PrefilterConfig
from eligsir.runtime.queues.bounded_queue import BoundedQueue, QueuePollState
from eligsir.runtime.workers.frame_preparation_worker import (
    FramePreparationWorker,
    FramePreparationWorkerConfig,
)

DURATION_S = 25.0
SAMPLE_INTERVAL_S = 1.0
FPS = 30.0


def _rss_kb() -> int:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


def main() -> None:
    in_q: BoundedQueue = BoundedQueue(capacity=8)
    out_q: BoundedQueue = BoundedQueue(capacity=8)
    worker = FramePreparationWorker(
        in_q, out_q,
        config=FramePreparationWorkerConfig(prefilter=PrefilterConfig(min_pose_novelty=0.0)),
    ).start()

    stop = threading.Event()

    def produce() -> None:
        # num_frames sized generously for DURATION_S at FPS with headroom.
        source = SyntheticRGBDFrameSource(
            SyntheticSourceConfig(seed=0, fps=FPS, num_frames=int(DURATION_S * FPS) + 60, realtime=True)
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

    samples = []
    t_end = time.monotonic() + DURATION_S
    try:
        while time.monotonic() < t_end:
            out_q.try_poll()  # non-blocking optimizer-side drain, same as the real integration
            samples.append((time.monotonic(), _rss_kb()))
            time.sleep(SAMPLE_INTERVAL_S)
    finally:
        stop.set()
        worker.stop(join_timeout_s=2.0)
        producer.join(timeout=1.0)

    n = len(samples)
    half = n // 2
    first_half = samples[:half]
    second_half = samples[half:]

    def _growth_kb_per_s(seg):
        if len(seg) < 2:
            return 0.0
        (t0, r0), (t1, r1) = seg[0], seg[-1]
        return (r1 - r0) / max(t1 - t0, 1e-6)

    result = {
        "n_samples": n,
        "rss_kb_start": samples[0][1] if samples else None,
        "rss_kb_end": samples[-1][1] if samples else None,
        "first_half_growth_kb_per_s": _growth_kb_per_s(first_half),
        "second_half_growth_kb_per_s": _growth_kb_per_s(second_half),
        "n_prepared": worker.n_prepared,
        "n_evicted_stale": worker.n_evicted_stale,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
