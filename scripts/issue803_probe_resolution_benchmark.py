#!/usr/bin/env python3
"""Reduced-resolution candidate-probe benchmark (issue #803).

Answers one question: **does a candidate probe need native-resolution
rendering to rank candidates the way a native-resolution probe would?**

For each resolution tier (native / half / quarter) it probes the same camera
cohort against the same fixed Gaussian snapshot and reports, per evidence
signal:

* Spearman rank correlation against the native-resolution probe;
* top-K agreement (overlap of the K highest-residual candidates);
* probe wall and device time, rasterizer calls, pixels;
* peak VRAM attributable to the probe.

**Scope and honesty caveats, stated up front because
``tier_cost_model.py``'s own docstring insists on them:**

* This is a **forward-only** measurement. Forward-only and forward+backward
  cost profiles must never be pooled -- a saving measured here says nothing
  about a supervision transaction's saving.
* Cost calibration is **hardware-local and workload-local**. The device,
  Gaussian-count regime and camera count are printed with every result and
  must be quoted with any number taken from it.
* It is a direct development invocation, therefore **UNMANAGED** evidence per
  CLAUDE.md's ResearchFlow hard constraint. It informs design; it is not a
  scientific result and must not be cited as one.
* The default scene is synthetic. Rank agreement between tiers on a synthetic
  scene is real evidence about the *measurement operator* (does downsampling
  preserve the ordering of these signals) and is deliberately not presented as
  evidence about any particular real dataset.

Run inside the training container with GPU passthrough::

    docker compose run --rm train python scripts/issue803_probe_resolution_benchmark.py \\
        --n-gaussians 50000 --n-cameras 16 --image-size 640 --repeats 3
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from types import SimpleNamespace

import torch


def _spearman(a: list[float], b: list[float]) -> float | None:
    """Spearman rank correlation, ties averaged. ``None`` if degenerate."""

    def ranks(values: list[float]) -> list[float]:
        order = sorted(range(len(values)), key=lambda i: values[i])
        out = [0.0] * len(values)
        index = 0
        while index < len(order):
            end = index
            while end + 1 < len(order) and values[order[end + 1]] == values[order[index]]:
                end += 1
            average = (index + end) / 2.0 + 1.0
            for position in range(index, end + 1):
                out[order[position]] = average
            index = end + 1
        return out

    if len(a) != len(b) or len(a) < 2:
        return None
    ra, rb = ranks(a), ranks(b)
    mean_a = sum(ra) / len(ra)
    mean_b = sum(rb) / len(rb)
    num = sum((x - mean_a) * (y - mean_b) for x, y in zip(ra, rb))
    den_a = sum((x - mean_a) ** 2 for x in ra) ** 0.5
    den_b = sum((y - mean_b) ** 2 for y in rb) ** 0.5
    if den_a == 0.0 or den_b == 0.0:
        return None
    return num / (den_a * den_b)


def _top_k_agreement(a: list[float], b: list[float], k: int) -> float:
    """Fraction of the top-K set (by descending value) shared by both rankings."""

    k = max(1, min(k, len(a)))
    top_a = set(sorted(range(len(a)), key=lambda i: -a[i])[:k])
    top_b = set(sorted(range(len(b)), key=lambda i: -b[i])[:k])
    return len(top_a & top_b) / float(k)


def _build_scene(n_gaussians: int, n_cameras: int, image_size: int, seed: int):
    import numpy as np

    from scene.cameras import Camera
    from scene.gsplat_model import GsplatGaussianModel

    rng = np.random.default_rng(seed)
    pcd = SimpleNamespace(
        points=rng.uniform(-2.0, 2.0, size=(n_gaussians, 3)).astype(np.float64),
        colors=rng.uniform(0.1, 0.9, size=(n_gaussians, 3)).astype(np.float64),
    )
    model = GsplatGaussianModel(sh_degree=0)
    model.create_from_pcd(
        pcd, spatial_lr_scale=1.0, init_scale_mode="fixed", init_scale=0.03
    )

    cameras = []
    for index in range(n_cameras):
        angle = 2.0 * np.pi * index / max(1, n_cameras)
        rotation = np.array(
            [
                [np.cos(angle), 0.0, np.sin(angle)],
                [0.0, 1.0, 0.0],
                [-np.sin(angle), 0.0, np.cos(angle)],
            ]
        )
        translation = np.array([0.0, 0.0, 5.0 + 0.4 * (index % 5)])
        image = torch.from_numpy(
            rng.uniform(0.0, 1.0, size=(3, image_size, image_size)).astype("float32")
        )
        # Explicit intrinsics are mandatory here: materialize_resolution_tier
        # refuses a camera without fx/fy/cx/cy, which would turn every
        # reduced-tier probe into a skipped candidate and quietly produce a
        # "benchmark" comparing native against nothing.
        focal = 0.5 * image_size / np.tan(0.5)
        cameras.append(
            Camera(
                colmap_id=index,
                R=rotation,
                T=translation,
                FoVx=1.0,
                FoVy=1.0,
                image=image,
                gt_alpha_mask=None,
                image_name=f"bench_{index:04d}",
                uid=index,
                fx=float(focal),
                fy=float(focal),
                cx=image_size / 2.0,
                cy=image_size / 2.0,
            )
        )
    return model, cameras


class _Session:
    def __init__(self, model):
        self.model = model
        self.parameter_version = 0
        self.optimizers = getattr(model, "optimizers", None)


class _Backend:
    def __init__(self, model):
        self.map_content_version = 0
        self.topology_version = 0
        self.coordinate_frame_version = 0
        self._session = _Session(model)

    def begin_optimization(self):
        return self._session


#: Signals whose cross-tier rank agreement is the actual question. Every one is
#: a raw evidence signal, never a fused score.
_SIGNALS = (
    "rgb_l1_mean",
    "rgb_mse_mean",
    "rgb_objective",
    "rgb_residual_normalized",
    "residual_entropy",
    "residual_top_decile_mass_fraction",
    "well_explained_pixel_fraction",
    "alpha_mean",
    "supported_pixel_fraction",
    "visible_gaussian_fraction",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-gaussians", type=int, default=50_000)
    parser.add_argument("--n-cameras", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json-out", type=str, default=None)
    args = parser.parse_args(argv)

    if not torch.cuda.is_available():
        print(
            "REFUSING: no GPU visible. A resolution-cost benchmark measured on "
            "CPU would be meaningless and, worse, quotable. Run inside the "
            "training container with GPU passthrough.",
            file=sys.stderr,
        )
        return 2

    from eligsir.cli.parser import build_training_parser
    from eligsir.contracts.control import ResolutionTier
    from eligsir.optimization.gaussian.candidate_probe import (
        CandidateProbeRequest,
        CandidateProbeView,
    )
    from eligsir.optimization.gaussian.contracts import (
        OptimizationRuntimeContext,
    )
    from eligsir.optimization.gaussian.engine import GaussianOptimizationEngine

    bundle = build_training_parser()
    namespace = bundle.parser.parse_args([])
    namespace.optimizer_type = "adam"
    opt = bundle.optimization_params.extract(namespace)
    pipe = bundle.pipeline_params.extract(namespace)

    model, cameras = _build_scene(
        args.n_gaussians, args.n_cameras, args.image_size, args.seed
    )
    backend = _Backend(model)
    engine = GaussianOptimizationEngine(backend)
    runtime = OptimizationRuntimeContext(
        session=backend.begin_optimization(), gaussians=model, opt=opt, pipe=pipe
    )
    background = torch.zeros(3, device="cuda")
    views = tuple(
        CandidateProbeView(
            camera=camera,
            background=background,
            source_id=f"bench-{index:04d}",
            source_ordinal=index,
            render_depth=False,
        )
        for index, camera in enumerate(cameras)
    )

    tiers = (ResolutionTier.NATIVE, ResolutionTier.HALF, ResolutionTier.QUARTER)
    measurements: dict[str, dict] = {}
    for tier in tiers:
        request = CandidateProbeRequest(
            views=views,
            probe_batch_id=f"bench-{tier.value}",
            resolution_tier=tier,
            compute_pairwise=False,
            synchronize_device=True,
        )
        probe = engine.probe_candidates(request, runtime)
        if probe.accounting.candidate_probe_skipped_view_count:
            reasons = {r.skip_reason for r in probe.per_view if r.skipped}
            print(
                "REFUSING: candidates were skipped at tier "
                f"{tier.value}; a benchmark over a partly-empty cohort is not a "
                f"measurement. Reasons: {sorted(reasons)}",
                file=sys.stderr,
            )
            return 3
        # Warm-up: the first probe at a tier pays a one-time materialization
        # (resize + stage + H2D). Pooling that into the steady-state number is
        # exactly the hot/cold confusion tier_cost_model.py warns about.
        engine.probe_candidates(request, runtime)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        baseline_vram = torch.cuda.memory_allocated()

        timings: list[float] = []
        batch = None
        for _ in range(max(1, args.repeats)):
            batch = engine.probe_candidates(request, runtime)
            timings.append(batch.accounting.candidate_probe_wall_ms)
        peak_vram = torch.cuda.max_memory_allocated()

        signals = {
            name: [
                getattr(result.evidence.probe, name).value
                for result in batch.per_view
            ]
            for name in _SIGNALS
        }
        measurements[tier.value] = {
            "signals": signals,
            "wall_ms_mean": sum(timings) / len(timings),
            "wall_ms_min": min(timings),
            "rasterizer_calls": batch.accounting.candidate_probe_rasterizer_call_count,
            "pixels": batch.accounting.candidate_probe_pixel_count,
            "probe_vram_bytes": int(peak_vram - baseline_vram),
            "skipped": batch.accounting.candidate_probe_skipped_view_count,
        }

    native = measurements[ResolutionTier.NATIVE.value]
    report = {
        "issue": 803,
        "evidence_class": "UNMANAGED",
        "pass_mode": "forward_only",
        "scene": "synthetic",
        "device": torch.cuda.get_device_name(0),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "n_gaussians": int(model.get_xyz.shape[0]),
        "n_cameras": len(views),
        "image_size": args.image_size,
        "repeats": args.repeats,
        "tiers": {},
    }
    for tier in tiers:
        entry = measurements[tier.value]
        agreement = {}
        for name in _SIGNALS:
            reference = native["signals"][name]
            values = entry["signals"][name]
            if any(v is None for v in reference) or any(v is None for v in values):
                agreement[name] = {"spearman": None, "top_k": None}
                continue
            agreement[name] = {
                "spearman": _spearman(reference, values),
                "top_k": _top_k_agreement(reference, values, args.top_k),
            }
        report["tiers"][tier.value] = {
            "wall_ms_mean": entry["wall_ms_mean"],
            "wall_ms_min": entry["wall_ms_min"],
            "speedup_vs_native": (
                native["wall_ms_mean"] / entry["wall_ms_mean"]
                if entry["wall_ms_mean"] > 0
                else None
            ),
            "rasterizer_calls": entry["rasterizer_calls"],
            "pixels": entry["pixels"],
            "probe_vram_bytes": entry["probe_vram_bytes"],
            "skipped_views": entry["skipped"],
            "rank_agreement_vs_native": agreement,
        }

    print(json.dumps(report, indent=2))
    print(
        "\nNOTE: forward-only, hardware-local, synthetic-scene, UNMANAGED "
        "development measurement. Do not pool with forward+backward "
        "supervision costs, and do not quote without the device line above.",
        file=sys.stderr,
    )
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
