"""Measures evidence-collection overhead (plan section 10.7, deliverable 4).

Compares cost of ``EvidenceBuilder.build_gaussian_evidence`` +
``build_render_evidence`` -- the exact same builder/reducer code path wired
into ``train_streaming.py``'s real ``_log_terms`` observation call site --
at a realistic Gaussian count under ``EvidenceRequirements.none()``
(production default; nothing consumes evidence yet) vs.
``EvidenceRequirements.everything()`` (worst case, so later PRs know the
cost ceiling of turning evidence collection fully on).

Uses GPU tensors + :mod:`eligsir.runtime.profiling` 's ``StageTimer``
(the same GPU-event-based, non-synchronizing-per-span primitive documented
in ``eligsir/runtime/profiling/events.py``) when CUDA/ROCm is
available, since the real wired call site runs on GPU tensors, not CPU
ones -- a CPU-only number would understate the real cost (GPU kernel
launch overhead per elementwise op, e.g. every ``.detach().clone()``, does
not show up on CPU tensors). Falls back to CPU + ``time.perf_counter``
when no GPU is present (still useful as a relative none-vs-everything
ratio, just not the absolute in-container number).

Run inside the container:

    docker compose run --rm -e PYTHONPATH=/workspace/eligsir train \\
      python scripts/benchmark_evidence_overhead.py
    docker compose run --rm -e PYTHONPATH=/workspace/eligsir train \\
      python scripts/benchmark_evidence_overhead.py --n 300000
"""

from __future__ import annotations

import argparse
import time

import torch

from eligsir.mapping.evidence.builder import EvidenceBuilder
from eligsir.mapping.evidence.gaussian_history import GaussianEvidenceHistory
from eligsir.mapping.evidence.requirements import EvidenceRequirements
from eligsir.runtime.profiling.events import set_profiling_enabled
from eligsir.runtime.profiling.gpu_timing import StageTimer


class _EnergyTerms:
    def __init__(self, n: int, device: str):
        self.alpha = torch.rand(n, device=device)
        self.support = torch.rand(n, device=device)
        self.xyz_grad_norm = torch.rand(n, device=device)
        self.opacity_grad_abs = torch.rand(n, device=device)
        self.visibility = torch.rand(n, device=device) > 0.5
        self.age_frac = torch.rand(n, device=device)
        self.geometry_confidence = torch.rand(n, device=device)
        self.free_space_conflict = torch.rand(n, device=device)
        self.render_contribution_ema = torch.rand(n, device=device)


def _run_gpu(n: int, reps: int, requirements: EvidenceRequirements, device: str) -> dict:
    set_profiling_enabled(True)
    timer = StageTimer("evidence_build")
    history = GaussianEvidenceHistory()
    builder = EvidenceBuilder(requirements, history=history)
    terms = _EnergyTerms(n, device)
    render_pkg = {
        "render": torch.rand(3, 480, 640, device=device),
        "alpha": torch.rand(1, 480, 640, device=device),
        "rendered_depth": torch.rand(1, 480, 640, device=device),
    }
    gt_image = torch.rand(3, 480, 640, device=device)
    gt_depth = torch.rand(1, 480, 640, device=device)

    # Warmup (first call is a cold-start EMA path; also warms up CUDA kernels).
    builder.build_gaussian_evidence(terms, topology_version=0, map_version=0)
    builder.build_render_evidence(render_pkg, gt_image=gt_image, gt_depth=gt_depth)

    for i in range(reps):
        with timer.span():
            builder.build_gaussian_evidence(terms, topology_version=0, map_version=i)
            builder.build_render_evidence(render_pkg, gt_image=gt_image, gt_depth=gt_depth)
    timer.resolve_pending()
    return timer.stats()


def _run_cpu(n: int, reps: int, requirements: EvidenceRequirements) -> float:
    history = GaussianEvidenceHistory()
    builder = EvidenceBuilder(requirements, history=history)
    terms = _EnergyTerms(n, "cpu")
    render_pkg = {
        "render": torch.rand(3, 480, 640),
        "alpha": torch.rand(1, 480, 640),
        "rendered_depth": torch.rand(1, 480, 640),
    }
    gt_image = torch.rand(3, 480, 640)
    gt_depth = torch.rand(1, 480, 640)

    builder.build_gaussian_evidence(terms, topology_version=0, map_version=0)
    builder.build_render_evidence(render_pkg, gt_image=gt_image, gt_depth=gt_depth)

    start = time.perf_counter()
    for i in range(reps):
        builder.build_gaussian_evidence(terms, topology_version=0, map_version=i)
        builder.build_render_evidence(render_pkg, gt_image=gt_image, gt_depth=gt_depth)
    elapsed = time.perf_counter() - start
    return elapsed / reps * 1e3


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=100_000, help="Gaussian count")
    parser.add_argument("--reps", type=int, default=200)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"N={args.n} reps={args.reps} device={device}")

    if device == "cuda":
        none_stats = _run_gpu(args.n, args.reps, EvidenceRequirements.none(), device)
        everything_stats = _run_gpu(args.n, args.reps, EvidenceRequirements.everything(), device)
        print(f"  requirements=none():       mean={none_stats['mean_ms']:.4f}ms p90={none_stats['p90_ms']:.4f}ms")
        print(
            f"  requirements=everything(): mean={everything_stats['mean_ms']:.4f}ms "
            f"p90={everything_stats['p90_ms']:.4f}ms"
        )
        if none_stats["mean_ms"] > 0:
            print(f"  everything/none ratio (mean): {everything_stats['mean_ms'] / none_stats['mean_ms']:.1f}x")
    else:
        none_ms = _run_cpu(args.n, args.reps, EvidenceRequirements.none())
        everything_ms = _run_cpu(args.n, args.reps, EvidenceRequirements.everything())
        print(f"  requirements=none():       {none_ms:.4f} ms/call")
        print(f"  requirements=everything(): {everything_ms:.4f} ms/call")
        if none_ms > 0:
            print(f"  everything/none ratio:     {everything_ms / none_ms:.1f}x")


if __name__ == "__main__":
    main()
