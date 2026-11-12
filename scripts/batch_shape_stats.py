#!/usr/bin/env python3
"""Compute Tier A (model-space, GT-free) shape stats for one or more saved
``point_cloud.ply`` files, optionally across a CPU process pool.

WS6: this is deliberately scoped to Tier A only (utils.scene_geometry_metrics
.shape_stats_from_ply) -- pure numpy, no GPU/torch involvement -- so it is
safe to parallelize on the single shared-VRAM iGPU: workers never touch the
GPU, so they never compete with a concurrent training/eval run for it.
GPU-bound stages (LPIPS/SSIM renders in metrics.py, Tier B/C which read
rendered/sensor depth against live cameras) are explicitly NOT handled here
and must stay sequential.

Usage:
    python scripts/batch_shape_stats.py PLY [PLY ...] [--workers N] [-o out.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

from eligsir.utils.scene_geometry_metrics import shape_stats_from_ply


def _compute_one(path: str) -> dict:
    try:
        return {"path": path, "ok": True, "stats": shape_stats_from_ply(path)}
    except Exception as exc:  # noqa: BLE001 -- surfaced per-path, not fatal for the batch
        return {"path": path, "ok": False, "error": str(exc)}


def compute_batch(paths: list, workers: int = 0) -> list:
    """Compute shape stats for every path in ``paths``.

    ``workers <= 1`` runs serially in-process (the default). ``workers > 1``
    parallelizes across a ``spawn``-context ``ProcessPoolExecutor`` -- spawn
    (not fork) is used deliberately so a worker never inherits any ROCm/CUDA
    context the parent process may have initialized.
    """
    if workers is None or workers <= 1:
        return [_compute_one(p) for p in paths]
    ctx = get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
        return list(pool.map(_compute_one, paths))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("plys", nargs="+", help="One or more point_cloud.ply paths")
    ap.add_argument("--workers", type=int, default=0,
                     help="Process-pool worker count (0/1 = serial, default)")
    ap.add_argument("-o", "--output", default=None, help="Write JSON results here")
    args = ap.parse_args(argv)

    results = compute_batch(args.plys, workers=args.workers)
    text = json.dumps(results, indent=2)
    if args.output:
        with open(args.output, "w") as f:
            f.write(text)
    else:
        print(text)
    n_failed = sum(1 for r in results if not r["ok"])
    return 1 if n_failed else 0


if __name__ == "__main__":
    sys.exit(main())
