#!/usr/bin/env python3
"""CPU-only profiling/benchmark harness for
``StreamingScene.accumulate_observed_surface`` -- the ingest-path method
GitHub issue #270 ("[Kornia gate] Optimize and re-profile
accumulate_observed_surface before decode promotion") and #276 ("[Ingest
perf] Optimize observed-surface accumulation from profiling evidence") are
about.

Calls the REAL bound method (not a reimplementation) through a minimal
StreamingScene-like harness, on synthetic-but-structured depth frames (a
front wall + receding floor + an invalid-depth doorway + sensor noise --
deliberately not uniform random noise, which would make every voxel key
unique and bias any dedup-sensitive measurement). See the "Workload realism"
note in ``docs/architecture/2026-08-13-issue270-accumulate-observed-surface-cpu-profile.md``
for why this matters.

**This is a CPU-only harness on synthetic data, not a substitute for issue
#275's GPU profiling matrix against real Orbbec/rosbag data.** In particular
this environment has no CUDA, so `_accumulate_sensor_noise`'s GPU-resident
PCA-normal path (``geometry.normals.estimate_depth_normals_pca``) runs on
CPU here -- on this host it dominates every ``--iq-sensor-stride``-th frame
by roughly an order of magnitude relative to a normal frame (measured
~750ms/call CPU-side `torch.linalg.eigh` vs ~25-30ms for an ordinary frame
at the default synthetic settings below). On a real GPU-equipped training
container this same call would run on CUDA and could be far faster or
still dominant -- that split is exactly what #275's GPU profiling matrix
needs to establish; this script cannot.

Usage:
    python scripts/bench_accumulate_observed_surface.py
    python scripts/bench_accumulate_observed_surface.py --n-frames 60 --profile
    python scripts/bench_accumulate_observed_surface.py --width 1280 --height 720 \\
        --output ingest_bench_report.json
"""
from __future__ import annotations

import argparse
import cProfile
import io as _io
import json
import pstats
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from geometry.surface_cells import SurfaceCellIndex
from scene.streaming_scene import StreamingScene
from eligsir.io.sources.decode_cache import StreamingRGBDFrame
from eligsir.mapping.stores.observation_store import ObservationStore


class _BenchArgs:
    streaming_track_scene_complexity = True
    streaming_min_depth = 0.1
    streaming_max_depth = 8.0
    streaming_depth_stride = 8
    streaming_track_input_quality = True
    streaming_depth_filter_enabled = True
    streaming_depth_filter_erode_px = 2
    streaming_depth_filter_window = 5
    streaming_depth_filter_min_valid_ratio = 0.75
    streaming_depth_filter_max_range = 0.08
    streaming_depth_filter_median_thresh = 0.04
    streaming_depth_filter_bottom_margin = 0.0
    streaming_insert_voxel_size = 0.02


class _BenchStreamingScene:
    """Minimal StreamingScene stand-in exposing exactly the real bound
    dependencies accumulate_observed_surface needs -- not a
    reimplementation. Mirrors the harness pattern already used in
    tests/test_streaming_scene_dense_trajectory.py and
    tests/test_observation_db_ingestion_wiring.py."""

    def __init__(self, args: _BenchArgs):
        self.args = args
        self._observation_store = ObservationStore(obs_voxel=float(args.streaming_insert_voxel_size))
        self._surface_cell_index = SurfaceCellIndex(voxel_size=float(args.streaming_insert_voxel_size))

    accumulate_observed_surface = StreamingScene.accumulate_observed_surface
    _stable_depth_samples = staticmethod(StreamingScene._stable_depth_samples)
    _pos_voxel_keys = staticmethod(StreamingScene._pos_voxel_keys)
    _accumulate_thickness = StreamingScene._accumulate_thickness
    _accumulate_sensor_noise = StreamingScene._accumulate_sensor_noise

    _obs_scales = StreamingScene._obs_scales
    _obs_voxel = StreamingScene._obs_voxel
    _obs_keys = StreamingScene._obs_keys
    _obs_vox = StreamingScene._obs_vox
    _obs_n_points = StreamingScene._obs_n_points
    _scene_complexity_cache = StreamingScene._scene_complexity_cache
    _iq_sensor_sum = StreamingScene._iq_sensor_sum
    _iq_sensor_sumsq = StreamingScene._iq_sensor_sumsq
    _iq_sensor_n = StreamingScene._iq_sensor_n
    _iq_sensor_stride = StreamingScene._iq_sensor_stride
    _iq_frame_counter = StreamingScene._iq_frame_counter
    _iq_vox = StreamingScene._iq_vox
    _iq_vox_cap = StreamingScene._iq_vox_cap
    _input_quality_cache = StreamingScene._input_quality_cache


def make_synthetic_frame(idx: int, w: int, h: int, seed: int = 0) -> StreamingRGBDFrame:
    rng = np.random.default_rng(seed + idx)
    fx = fy = w * 0.71  # ~ Orbbec Femto Bolt-shaped FoV
    depth = np.full((h, w), 3.0, dtype=np.float32)
    horizon = int(h * 0.55)
    if horizon < h:
        rows = np.arange(horizon, h)
        depth[horizon:h, :] = np.clip(3.0 - (rows - horizon)[:, None] * 0.02, 0.3, 3.0)
    depth[int(h * 0.2):int(h * 0.8), int(w * 0.45):int(w * 0.55)] = 0.0  # doorway: invalid depth
    depth += rng.normal(0.0, 0.003, size=depth.shape).astype(np.float32)
    depth_mm = np.clip(depth * 1000.0, 0, 65535).astype(np.uint16)
    depth_mm[depth <= 0] = 0

    c2w = np.eye(4, dtype=np.float32)
    c2w[0, 3] = idx * 0.05  # lateral dolly
    c2w[2, 3] = idx * 0.02

    return StreamingRGBDFrame(
        index=idx, timestamp=float(idx), rgb_path="", depth_path=None,
        c2w=c2w, fx=fx, fy=fy, cx=w / 2.0, cy=h / 2.0, width=w, height=h,
        depth_scale=1000.0, _depth_np=depth_mm,
    )


def run(n_frames: int, width: int, height: int, profile: bool) -> dict[str, Any]:
    scene = _BenchStreamingScene(_BenchArgs())
    frames = [make_synthetic_frame(i, width, height) for i in range(n_frames)]

    pr = cProfile.Profile() if profile else None
    if pr:
        pr.enable()

    times = []
    for f in frames:
        t0 = time.perf_counter()
        scene.accumulate_observed_surface(f)
        times.append(time.perf_counter() - t0)

    profile_text = None
    if pr:
        pr.disable()
        s = _io.StringIO()
        pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(25)
        profile_text = s.getvalue()
        print(profile_text)

    times_arr = np.array(times)
    result = {
        "n_frames": n_frames,
        "width": width,
        "height": height,
        "mean_ms": float(times_arr.mean() * 1e3),
        "p50_ms": float(np.percentile(times_arr, 50) * 1e3),
        "p95_ms": float(np.percentile(times_arr, 95) * 1e3),
        "p99_ms": float(np.percentile(times_arr, 99) * 1e3),
        "max_ms": float(times_arr.max() * 1e3),
        "per_frame_ms": [float(t * 1e3) for t in times_arr],
        "obs_n_points": int(scene._obs_n_points),
        "sv_multiscale": {
            f"x{m}": {
                "unique_surface_keys": len(scene._obs_keys[m]),
                "unique_voxel_keys": len(scene._obs_vox[m]),
            }
            for m in scene._obs_scales
        },
        "note": (
            "CPU-only synthetic-data benchmark (this environment has no CUDA -- "
            "the GPU-resident PCA sensor-noise sub-path runs on CPU here). "
            "NOT a substitute for issue #275's GPU profiling matrix on real "
            "Orbbec/rosbag data."
        ),
    }
    print(
        f"n_frames={n_frames} ({width}x{height}) mean={result['mean_ms']:.3f}ms "
        f"p50={result['p50_ms']:.3f}ms p95={result['p95_ms']:.3f}ms "
        f"p99={result['p99_ms']:.3f}ms max={result['max_ms']:.3f}ms"
    )
    print(f"obs_n_points={result['obs_n_points']}")
    for m, stats in result["sv_multiscale"].items():
        print(f"  scale {m}: {stats}")
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-frames", type=int, default=30)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--profile", action="store_true", help="Print a cProfile cumulative-time breakdown")
    ap.add_argument("--output", type=str, default=None, help="Optional path to write a JSON report")
    args = ap.parse_args()

    result = run(args.n_frames, args.width, args.height, args.profile)
    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
