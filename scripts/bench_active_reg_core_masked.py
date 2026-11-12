"""Microbenchmark: active_reg_core (dynamic tensor[mask] compaction) vs
active_reg_core_masked (fixed-shape dense masked math), across population
size and active-fraction, to characterize issue #1670's regression shape
(observed: 38->99 supervision_views_per_s ramp on the real fr3 probe,
consistent with fixed per-call launch overhead dominating at small N).

Standalone diagnostic (dev-only, UNMANAGED). Real GPU required.

Usage (inside the container):
    python scripts/bench_active_reg_core_masked.py --repeats 50 --out bench.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from eligsir.utils.compiled_kernels import active_reg_core, active_reg_core_masked


def _bench_one(n: int, active_fraction: float, repeats: int, device: str) -> dict:
    torch.manual_seed(0)
    opacity = torch.rand(n, 1, device=device)
    scaling = torch.rand(n, 3, device=device)
    mask = torch.rand(n, device=device) < active_fraction

    def old_call():
        active_reg_core(opacity[mask], scaling[mask], w_opacity=0.01, w_scale=0.01)

    def new_call():
        active_reg_core_masked(opacity, scaling, mask, w_opacity=0.01, w_scale=0.01)

    def timed(fn) -> float:
        if device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(repeats):
            fn()
        if device == "cuda":
            torch.cuda.synchronize()
        return (time.perf_counter() - start) / repeats * 1000.0

    # Warm up both paths once before timing (first-call kernel-launch/cache
    # warmup is a real, separate cost this benchmark deliberately excludes
    # from the steady-state number, since the fr3 probe's 38->99 ramp
    # spans many iterations, not just the first).
    old_call()
    new_call()

    old_ms = timed(old_call)
    new_ms = timed(new_call)
    return {
        "population_size": n,
        "active_fraction": active_fraction,
        "old_ms": old_ms,
        "new_ms": new_ms,
        "speedup": old_ms / new_ms if new_ms > 0 else float("inf"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    populations = [932, 5000, 20000, 100000]
    fractions = [0.1, 0.5, 0.9, 1.0]

    results = [
        _bench_one(n, frac, args.repeats, device)
        for n in populations
        for frac in fractions
    ]
    for row in results:
        print(
            f"N={row['population_size']:>7} active={row['active_fraction']:.1f} "
            f"old={row['old_ms']:.4f}ms new={row['new_ms']:.4f}ms "
            f"speedup={row['speedup']:.2f}x"
        )
    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
