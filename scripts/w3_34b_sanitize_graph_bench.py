"""W3-34B: the three(-plus-two)-arm sanitizer/graph-capture ROI experiment.

Context: 34A (docs/architecture/2026-08-10-w3-34a-graph-capture-feasibility.md)
found render is not graph-capturable on the current stack, so the only
reachable capture island is the post-backward tensor tail (grad-masking
cascade -> sanitize -> SelectiveAdam step -> zero_grad), and that
production sanitize's host-sync pattern is itself the capture blocker in
that tail. It confirmed a branch-free sanitize rewrite IS capturable, but
explicitly did NOT measure whether it's cheaper (or even not-more-
expensive) in eager mode, nor run the graph-replay arm at all -- that is
this task's job, per the review that redefined 34B (see the findings doc
this script's output feeds, docs/architecture/<date>-w3-34b-sanitize-graph-roi.md).

Arms (all real GsplatGaussianModel + real SelectiveAdam + real
grad-masking cascade + real normalize_rotation_params, on a real render's
visibility mask and real backward's gradient tensors):

  A  : production sanitize (with its diagnostic float(get_scaling.max())
       line), set_to_none=True, eager.
  A' : same as A but with the unconditional float(get_scaling.max())
       diagnostic line removed (isolates the diagnostic's own cost from
       the branch-free rewrite's -- advisor-flagged: this line is an
       unconditional N-sized exp()+max()+host-sync EVERY call, separate
       from the int(tensor.sum()) gates the 34A doc named).
  B  : branch-free sanitize (_sanitize_gaussian_params_branch_free), no
       diagnostics at all, set_to_none=True, eager.
  B' : same as B but set_to_none=False (in-place zero_ instead of None-
       reassignment) -- isolates the zeroing-strategy change C is forced
       into from graph replay's own effect (B-vs-B' = cost of that
       change alone; B'-vs-C = pure replay benefit).
  C  : same as B', captured via torch.cuda.graph and replayed.

A vs B: value of eliminating sanitize's branching/host-sync (bundles the
diagnostic elimination with the branch-free rewrite -- see A' to split
that further). B' vs C: incremental value of graph replay alone, same
sanitize/policies/zeroing on both sides. A vs C: total practical delta,
though C's zeroing strategy differs from A's real set_to_none=True by
necessity (see the findings doc for why that's a required, disclosed
adaptation, not an apples-to-apples free lunch).

Timing methodology (advisor-directed, correcting the "just time the loop"
plan): production sanitize self-synchronizes (int()/float() host syncs
every call); branch-free arms don't. A naive "time N iterations" would
let B/C's launched-but-not-yet-finished GPU work leak past the timing
window while A/A' can't. Every arm is timed as repeated
sync-bracketed BLOCKS: sync(); t0=now(); for _ in range(ITERS_PER_BLOCK):
body(); sync(); t1=now(). NUM_BLOCKS blocks per arm per N, first block
discarded as warmup, mean+-std reported over the rest.

N sweep (10k / 60k / 150k) because a toy N (e.g. the 24-point unit-test
fixture) is launch-bound and would make the branch-free arms look
artificially worse (more unconditional per-param kernels) -- this
directly answers 34A's own flagged open question about whether the
branch-free rewrite's unconditional work ever dominates at realistic N.

Run: docker compose run --rm -e PYTHONPATH=/workspace/eligsir train \
    python scripts/w3_34b_sanitize_graph_bench.py
"""
from __future__ import annotations

import json
import statistics
import sys
import time
from types import SimpleNamespace

import torch

ITERS_PER_BLOCK = 150
NUM_BLOCKS = 11  # first discarded as warmup -> 10 measured blocks
N_SWEEP = (10_000, 60_000, 150_000)


def _pipe():
    return SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)


def _build_args_selective_adam():
    from eligsir.cli.parser import build_training_parser

    bundle = build_training_parser()
    ns = bundle.parser.parse_args([])
    ns.optimizer_type = "selective_adam"
    ns.cap_max = 500_000
    args = bundle.model_params.extract(ns)
    for key, value in vars(ns).items():
        if not hasattr(args, key):
            setattr(args, key, value)
    opt = bundle.optimization_params.extract(ns)
    pipe = bundle.pipeline_params.extract(ns)
    return args, opt, pipe, ns


def _build_gaussians_n(n: int, seed: int = 0):
    import numpy as np

    from scene.gsplat_model import GsplatGaussianModel

    rng = np.random.default_rng(seed)
    # Wide spread (not a tight unit cube) relative to the camera's frustum
    # so only a MINORITY of points are visible from any one camera --
    # matches the real sparse_active_set/SelectiveAdam masked-update
    # pattern (typically well under 100% of the population visible per
    # view in a real streaming scene). A tight cube where every point is
    # visible (measured frac_visible=1.0 during this task's first attempt)
    # would make step_mask all-True and silently fail to exercise the
    # sparse masking path at all -- corrected after that measurement.
    points = rng.uniform(-15.0, 15.0, size=(n, 3)).astype(np.float64)
    colors = rng.uniform(0.2, 0.8, size=(n, 3)).astype(np.float64)
    pcd = SimpleNamespace(points=points, colors=colors)
    g = GsplatGaussianModel(sh_degree=0)
    # fixed init_scale_mode: avoids the KNN-based distCUDA2 cost (irrelevant
    # to this benchmark's scope, and would dominate setup time at N=150k)
    g.create_from_pcd(pcd, spatial_lr_scale=1.0, init_scale_mode="fixed", init_scale=0.05)
    return g


def _build_camera_n(n: int, image_size: int = 64, seed: int = 1):
    import numpy as np

    from scene.cameras import Camera

    rng = np.random.default_rng(seed)
    R = np.eye(3)
    T = np.array([0.0, 0.0, 6.0])  # pulled back further so a wide point spread stays in frustum
    image = torch.from_numpy(
        rng.uniform(0.0, 1.0, size=(3, image_size, image_size)).astype("float32")
    )
    cam = Camera(colmap_id=0, R=R, T=T, FoVx=1.2, FoVy=1.2, image=image,
                 gt_alpha_mask=None, image_name="bench_0000", uid=0)
    return cam


def _setup_state(n: int):
    """Real GsplatGaussianModel + real SelectiveAdam + one real render/
    backward to get a real visibility mask and real gradient tensors
    (magnitudes/sparsity pattern actually produced by the real photometric
    loss + gsplat backward, not hand-crafted)."""
    from gaussian_renderer import render

    args, opt, pipe, _ns = _build_args_selective_adam()
    g = _build_gaussians_n(n)
    g.training_setup(args)

    cam = _build_camera_n(n)
    bg = torch.tensor([0.0, 0.0, 0.0], device="cuda")
    render_pkg = render(cam, g, _pipe(), bg, update_sh_rest=True, render_depth=False)
    image = render_pkg["render"]
    from eligsir.utils.loss_utils import l1_loss, ssim
    gt = cam.original_image
    loss = 0.8 * l1_loss(image, gt) + 0.2 * (1.0 - ssim(image, gt))
    loss.backward()

    visible = render_pkg["visibility_filter"].detach().clone()
    provisional = torch.zeros(n, dtype=torch.bool, device="cuda")
    step_mask = visible | provisional
    frac_visible = float(step_mask.float().mean())

    grad_snapshot = {name: (p.grad.detach().clone() if p.grad is not None
                             else torch.zeros_like(p))
                      for name, p in g.params.items()}
    for p in g.params.values():
        p.grad = None
    return g, args, step_mask, grad_snapshot, frac_visible


def _grad_cascade(g, step_mask):
    with torch.no_grad():
        for p in g.params.values():
            if p.grad is not None and getattr(p.grad, "layout", torch.strided) == torch.strided:
                p.grad[~step_mask] = 0.0


def _normalize_rotation_eager(g, step_mask):
    g.normalize_rotation_params(mask=step_mask)


def _normalize_rotation_capture_safe(g, step_mask):
    """Capture-safe rewrite of GsplatGaussianModel.normalize_rotation_params:
    the real ``quats[mask] = normalize(quats[mask])`` form reads a
    boolean-indexed (data-dependent-size) gather as an rvalue, which this
    task found ALSO breaks torch.cuda.graph capture (a second, previously
    undocumented capture blocker in the post-backward tail beyond
    sanitize -- see the findings doc). Shape-static equivalent: normalize
    over the full N (more unconditional work, exactly the same trade the
    branch-free sanitize rewrite makes) and blend with torch.where."""
    with torch.no_grad():
        quats = g.params["quats"]
        normalized = torch.nn.functional.normalize(quats, dim=1, eps=1e-8)
        quats.copy_(torch.where(step_mask.unsqueeze(-1), normalized, quats))


# ---------------------------------------------------------------------------
# Per-arm iteration bodies
# ---------------------------------------------------------------------------


def _make_body_eager(g, step_mask, grad_snapshot, sanitize_fn, set_to_none):
    params = g.params

    def body():
        for name, p in params.items():
            p.grad = grad_snapshot[name].clone()
        _grad_cascade(g, step_mask)
        sanitize_fn(g)
        g.prepare_selective_adam_step()
        g.optimizer.step(visibility=step_mask)
        _normalize_rotation_eager(g, step_mask)
        g.optimizer.zero_grad(set_to_none=set_to_none)

    return body


def _make_body_capture_safe(g, step_mask, grad_snapshot):
    """B'/C body: branch-free sanitize, capture-safe rotation normalize,
    set_to_none=False (in-place zero, grad tensor identity never
    disappears -- required for capture, see findings doc)."""
    from eligsir.mapping.integration.loss_diagnostics import (
        _sanitize_gaussian_params_branch_free,
    )

    params = g.params
    # Pre-allocate persistent grad buffers ONCE -- capture requires a
    # static tensor address for every op's inputs across replays.
    with torch.no_grad():
        for name, p in params.items():
            p.grad = torch.zeros_like(p)

    def body():
        with torch.no_grad():
            for name, p in params.items():
                p.grad.copy_(grad_snapshot[name])
        _grad_cascade(g, step_mask)
        _sanitize_gaussian_params_branch_free(g)
        g.prepare_selective_adam_step()
        g.optimizer.step(visibility=step_mask)
        _normalize_rotation_capture_safe(g, step_mask)
        with torch.no_grad():
            for p in params.values():
                p.grad.zero_()

    return body


def _sanitize_production(g):
    from eligsir.mapping.integration.loss_diagnostics import _sanitize_gaussian_params
    _sanitize_gaussian_params(g, "bench", iteration=0)


def _sanitize_production_minus_diagnostic(g):
    """A': verbatim production logic minus the unconditional
    float(gaussians.get_scaling.max()) diagnostic line -- isolates that
    line's own cost (an unconditional N x 3 exp() + max() + host sync
    EVERY call, separate from the int(tensor.sum()) gates 34A's doc
    named) from the rest of the branch-free rewrite's changes."""
    with torch.no_grad():
        if hasattr(g, "_scaling") and g._scaling is not None:
            _sc = g._scaling
            _over = int((_sc > 15.0).sum()) if _sc.numel() else 0
            if _over:
                _sc.data.clamp_(max=15.0)
        else:
            _over = 0
        _bad = {}
        for _nm in ("_xyz", "_opacity", "_scaling", "_rotation", "_features_dc", "_features_rest"):
            _t = getattr(g, _nm, None)
            if _t is None:
                continue
            _nf = int((~torch.isfinite(_t)).sum())
            if _nf:
                _bad[_nm] = _nf
                _t.data.copy_(torch.nan_to_num(_t.data, nan=0.0, posinf=1e6, neginf=-1e6))
        # (diagnostic print block intentionally omitted -- that's the whole point of A')


def _sanitize_branch_free(g):
    from eligsir.mapping.integration.loss_diagnostics import (
        _sanitize_gaussian_params_branch_free,
    )
    _sanitize_gaussian_params_branch_free(g)


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------


def _time_blocks(body, num_blocks=NUM_BLOCKS, iters_per_block=ITERS_PER_BLOCK):
    block_times = []
    for _ in range(num_blocks):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters_per_block):
            body()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        block_times.append((t1 - t0) / iters_per_block * 1e3)  # ms/iter
    measured = block_times[1:]  # discard first block as warmup
    return {
        "mean_ms": statistics.mean(measured),
        "std_ms": statistics.pstdev(measured),
        "min_ms": min(measured),
        "max_ms": max(measured),
        "n_blocks_measured": len(measured),
        "all_block_means_ms": block_times,
    }


def _capture_and_time(body, num_blocks=NUM_BLOCKS, iters_per_block=ITERS_PER_BLOCK):
    # Warm up on a side stream (required before capture per torch.cuda.graph docs).
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(5):
            body()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        body()

    def replay_body():
        graph.replay()

    return _time_blocks(replay_body, num_blocks=num_blocks, iters_per_block=iters_per_block), graph


def run_for_n(n: int):
    results = {}

    print(f"\n===== N={n} =====", flush=True)

    # ---- Arm A: production sanitize, set_to_none=True ----
    g, args, step_mask, grad_snapshot, frac_visible = _setup_state(n)
    results["frac_visible"] = frac_visible
    print(f"frac_visible (sparse step_mask fraction True): {frac_visible:.4f}", flush=True)
    body_a = _make_body_eager(g, step_mask, grad_snapshot, _sanitize_production, set_to_none=True)
    results["A_production_eager"] = _time_blocks(body_a)
    print(f"A  (production, set_to_none=True):      {results['A_production_eager']['mean_ms']:.4f} "
          f"+- {results['A_production_eager']['std_ms']:.4f} ms/iter", flush=True)
    del g

    # ---- Arm A': production sanitize minus the diagnostic max() line ----
    g, args, step_mask, grad_snapshot, _ = _setup_state(n)
    body_ap = _make_body_eager(g, step_mask, grad_snapshot, _sanitize_production_minus_diagnostic, set_to_none=True)
    results["Aprime_no_diagnostic_eager"] = _time_blocks(body_ap)
    print(f"A' (production minus diagnostic line):  {results['Aprime_no_diagnostic_eager']['mean_ms']:.4f} "
          f"+- {results['Aprime_no_diagnostic_eager']['std_ms']:.4f} ms/iter", flush=True)
    del g

    # ---- Arm B: branch-free sanitize, set_to_none=True ----
    g, args, step_mask, grad_snapshot, _ = _setup_state(n)
    body_b = _make_body_eager(g, step_mask, grad_snapshot, _sanitize_branch_free, set_to_none=True)
    results["B_branch_free_eager"] = _time_blocks(body_b)
    print(f"B  (branch-free, set_to_none=True):     {results['B_branch_free_eager']['mean_ms']:.4f} "
          f"+- {results['B_branch_free_eager']['std_ms']:.4f} ms/iter", flush=True)
    del g

    # ---- Arm B': branch-free sanitize, set_to_none=False, capture-safe normalize ----
    g, args, step_mask, grad_snapshot, _ = _setup_state(n)
    body_bp = _make_body_capture_safe(g, step_mask, grad_snapshot)
    results["Bprime_capture_safe_eager"] = _time_blocks(body_bp)
    print(f"B' (branch-free, set_to_none=False, eager, capture-safe normalize): "
          f"{results['Bprime_capture_safe_eager']['mean_ms']:.4f} "
          f"+- {results['Bprime_capture_safe_eager']['std_ms']:.4f} ms/iter", flush=True)
    del g

    # ---- Arm C: same as B', captured + replayed ----
    g, args, step_mask, grad_snapshot, _ = _setup_state(n)
    body_c = _make_body_capture_safe(g, step_mask, grad_snapshot)
    try:
        results["C_graph_replay"], _graph = _capture_and_time(body_c)
        print(f"C  (branch-free, captured+replayed):    {results['C_graph_replay']['mean_ms']:.4f} "
              f"+- {results['C_graph_replay']['std_ms']:.4f} ms/iter", flush=True)
        results["C_capture_succeeded"] = True
    except Exception as exc:  # noqa: BLE001
        results["C_capture_succeeded"] = False
        results["C_capture_error"] = f"{type(exc).__name__}: {exc}"
        print(f"C  capture FAILED: {type(exc).__name__}: {exc}", flush=True)
    del g

    return results


def main():
    if not torch.cuda.is_available():
        print("SKIP: no CUDA/ROCm device available")
        return 0

    all_results = {}
    for n in N_SWEEP:
        all_results[str(n)] = run_for_n(n)

    out_path = "/tmp/w3_34b_bench_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
