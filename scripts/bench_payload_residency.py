#!/usr/bin/env python3
"""CPU-only benchmark for clean payload-residency sweep elimination.

This benchmark exercises the residency owner with lightweight camera doubles;
it does not construct a renderer, allocate CUDA tensors, or launch GPU work.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
from types import SimpleNamespace

# Make ``python scripts/bench_payload_residency.py`` work from the repository
# checkout as well as from the project container.
_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from eligsir.runtime.payload_residency import (
    mark_payload_residency_dirty,
    retire_live_admitted_payloads,
)


class _Frame:
    def __init__(self) -> None:
        self._rgb_np = object()
        self._depth_np = object()
        self._rgb_bytes = None
        self._depth_bytes = None
        self.mapping_frame = object()


class _Camera:
    def __init__(self) -> None:
        self._streaming_frame = _Frame()
        self._mapping_frame = self._streaming_frame.mapping_frame
        self._sensor_depth_cache = object()
        self.original_image = object()
        self._native_image_loader = lambda: None
        self._pending_image_loader = None
        self._canonical_native_resolution_state = object()
        self._live_payload_retired = False


def _scene(n_cameras: int):
    cameras = [_Camera() for _ in range(n_cameras)]
    return SimpleNamespace(
        train_cameras=cameras,
        _replay_buffer=cameras[-8:],
        _global_reservoir=cameras[-32:],
        _hard_frames=[],
        _covisible_cache=[],
        _residual_camera_snapshot=[],
        current_camera=cameras[-1],
        args=SimpleNamespace(
            streaming_keyframe_window=8,
            streaming_live_admitted_payload_max_frames=32,
        ),
        _payload_residency_dirty=True,
    )


def _run(n_cameras: int, iterations: int, *, force_dirty: bool) -> float:
    scene = _scene(n_cameras)
    start = time.perf_counter()
    for _ in range(iterations):
        if force_dirty:
            mark_payload_residency_dirty(scene)
        retire_live_admitted_payloads(scene)
    return time.perf_counter() - start


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cameras", type=int, default=10_000)
    parser.add_argument("--iterations", type=int, default=1_000)
    args = parser.parse_args()
    if args.cameras < 1 or args.iterations < 1:
        parser.error("--cameras and --iterations must be positive")

    clean_s = _run(args.cameras, args.iterations, force_dirty=False)
    dirty_s = _run(args.cameras, args.iterations, force_dirty=True)
    print(f"cameras={args.cameras} iterations={args.iterations}")
    print(f"clean_elapsed_s={clean_s:.6f}")
    print(f"forced_dirty_elapsed_s={dirty_s:.6f}")
    print(f"speedup={dirty_s / max(clean_s, 1e-12):.2f}x")


if __name__ == "__main__":
    main()
