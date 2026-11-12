"""Wave 3, PR W3-12, deliverable 7: launch count, GPU duty cycle, wall
optimization steps/s, capture cost, and amortization point for
``TorchAcceleratorGraphRuntime`` vs. plain eager execution of the same
post-backward tensor-only step (gradient-policy masking placeholder +
SelectiveAdam step + in-place grad zeroing) -- see
``eligsir/optimization/gaussian/graph_runtime/runtime.py`` for exactly
what this PR's capture scope covers and why (render/backward are
excluded, confirmed unsafe on this ROCm hardware).

Run inside the container:

    docker compose run --rm train python scripts/bench_w3_12_graph_runtime.py

This is a microbenchmark against synthetic tensors sized like a real
Gaussian-count capacity bucket -- not a champion-scene training run (this
PR's own scope, per the workstream brief, does not require one). Results
are printed as JSON and also written to
``docs/architecture/evidence/w3_12_graph_runtime_bench.json``.
"""

from __future__ import annotations

import json
import pathlib
import time

import torch
from gsplat.optimizers import SelectiveAdam

from eligsir.optimization.gaussian.graph_runtime.runtime import (
    EagerRuntime,
    TorchAcceleratorGraphRuntime,
    zero_grad_in_place,
)
from eligsir.optimization.gaussian.graph_runtime.signature import GraphSignature

_DEVICE = "cuda"


def _make_step(n: int, dim: int = 59):
    """``dim=59`` approximates one Gaussian's total optimized scalar
    footprint across means(3)+sh0(3)+shN(45)+opacities(1)+scales(3)+
    quats(4) at SH degree 3, flattened -- a representative per-Gaussian
    parameter width, not a claim about the real model's exact packing.
    """
    g = torch.Generator(device=_DEVICE).manual_seed(0)
    param = torch.nn.Parameter(torch.randn(n, dim, device=_DEVICE, generator=g))
    optimizer = SelectiveAdam([param], eps=1e-15, betas=(0.9, 0.999))
    visibility = torch.ones(n, dtype=torch.bool, device=_DEVICE)
    param.grad = torch.zeros_like(param)
    grad_gen = torch.Generator(device=_DEVICE).manual_seed(1)

    def refresh_grad():
        param.grad.copy_(torch.randn(n, dim, device=_DEVICE, generator=grad_gen) * 0.01)

    def step_fn():
        # Gradient-policy masking placeholder: engine.py's
        # GRADIENT_POLICY_ORDER runs FIVE separate policy objects
        # (active_mask, stable_geometry, bootstrap_anchor,
        # insertion_anchor, depth_lock) each doing its own small
        # elementwise masking op -- simulate that call-count shape (five
        # small ops, not one) since the "many small kernel launches"
        # thesis this PR's launch-count deliverable is about applies to
        # call COUNT, not just total FLOPs.
        mask = visibility.to(param.dtype).unsqueeze(-1)
        for _ in range(5):
            param.grad.mul_(mask)
        optimizer.step(visibility)
        zero_grad_in_place([param])

    return param, refresh_grad, step_fn


def _sig(n: int) -> GraphSignature:
    return GraphSignature(
        device_backend="bench",
        capacity_bucket=n,
        active_shape=(n,),
        resolution_bucket=(0, 0),
        sh_degree=3,
        renderer_mode="none",
        packed_mode=False,
        sparse_grad_mode=True,
        optimizer_mode="selective_adam",
        loss_policy_id="bench",
        gradient_policy_id="bench",
    )


def _bench_eager(n: int, n_steps: int) -> dict:
    _, refresh_grad, step_fn = _make_step(n)
    # warmup
    for _ in range(5):
        refresh_grad()
        step_fn()
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(n_steps):
        refresh_grad()
        step_fn()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return {"elapsed_s": elapsed, "steps_per_s": n_steps / elapsed, "n_steps": n_steps}


def _bench_graph(n: int, n_steps: int) -> dict:
    _, refresh_grad, step_fn = _make_step(n)
    runtime = TorchAcceleratorGraphRuntime()
    sig = _sig(n)

    refresh_grad()
    t_capture_start = time.perf_counter()
    outcome = runtime.run_step(sig, step_fn)
    torch.cuda.synchronize()
    capture_cost_s = time.perf_counter() - t_capture_start
    assert outcome.first_capture

    start = time.perf_counter()
    for _ in range(n_steps):
        refresh_grad()
        runtime.run_step(sig, step_fn)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return {
        "elapsed_s": elapsed,
        "steps_per_s": n_steps / elapsed,
        "n_steps": n_steps,
        "capture_cost_s": capture_cost_s,
        "counters": runtime.counters,
    }


def main():
    n_steps = 500
    results = {}
    for n in (4096, 65536, 262144):
        eager = _bench_eager(n, n_steps)
        graph = _bench_graph(n, n_steps)
        speedup = graph["steps_per_s"] / eager["steps_per_s"]
        per_step_saving_s = (1.0 / eager["steps_per_s"]) - (1.0 / graph["steps_per_s"])
        amortization_steps = (
            graph["capture_cost_s"] / per_step_saving_s if per_step_saving_s > 0 else None
        )
        results[str(n)] = {
            "eager": eager,
            "graph": graph,
            "speedup_steps_per_s": speedup,
            "per_step_saving_s": per_step_saving_s,
            "amortization_point_steps": amortization_steps,
        }
        print(
            f"N={n}: eager={eager['steps_per_s']:.1f} steps/s, "
            f"graph={graph['steps_per_s']:.1f} steps/s, "
            f"speedup={speedup:.3f}x, capture_cost={graph['capture_cost_s']*1000:.2f}ms, "
            f"amortization={amortization_steps}"
        )

    out_path = pathlib.Path("docs/architecture/evidence/w3_12_graph_runtime_bench.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
