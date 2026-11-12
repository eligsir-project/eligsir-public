"""W3-62: strict-math vs. fast-math ROCm gsplat -- synthetic adversarial probe.

Runs gsplat.rendering.rasterization forward+backward on a battery of
deliberately adversarial, synthetically-constructed Gaussian parameter
states (very small/large scales, near-camera, off-axis, near-singular
projected covariance, very low/high opacity) plus one "normal" control
state, and dumps forward outputs + parameter gradients to a .pt file.

Each state is run in its OWN subprocess (re-invoking this same script with
--single-state) because a HIP/ROCm illegal-memory-access fault kills the
whole process with no Python-catchable exception -- without isolation, one
adversarial state crashing would silently prevent every other state (and
the whole comparison) from being measured. A crash is recorded as a result
in its own right (crash=True, returncode=...), not silently dropped.

Run ONCE per gsplat build variant (inside a container with that variant's
wheel installed), then compare the two .pt files with
docker/w362_compare_variants.py in any environment that has torch (does not
need gsplat itself).

Usage (inside the container that has the wheel under test installed):
    python3 docker/w362_synthetic_probe.py --variant strict --out /tmp/probe_strict.pt
    python3 docker/w362_synthetic_probe.py --variant fastmath --out /tmp/probe_fastmath.pt

All states are built with a fixed seed and CPU-side RNG only, so both
invocations construct byte-identical input tensors before any GPU/gsplat
code runs (verified by hashing the CPU-built inputs before .to(device), see
input_hashes in the saved artifact -- identical across variants is a
precondition for the comparison to mean anything).
"""
import argparse
import hashlib
import json
import subprocess
import sys

import torch


def _hash_tensor(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().cpu().numpy().tobytes()).hexdigest()[:16]


def _make_camera(width=64, height=48, focal=60.0, cam_z=3.0, device="cpu"):
    # world-to-camera extrinsic: camera sits at world z=-cam_z looking toward +z,
    # so a world point at z=+d (d>0, in front) maps to camera-space z=+d+cam_z>0.
    viewmat = torch.eye(4, dtype=torch.float32)
    viewmat[2, 3] = cam_z
    K = torch.tensor(
        [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )
    return viewmat.unsqueeze(0).to(device), K.unsqueeze(0).to(device), width, height


def _rand_quats(n, generator):
    q = torch.randn(n, 4, generator=generator)
    return q / q.norm(dim=-1, keepdim=True)


def build_states(seed=1234):
    """Returns a dict of {name: dict(means, quats, scales, opacities, colors)}.

    NOTE on SH: this project's real renderer (eligsir/mapping/renderer/
    gsplat_backend.py, `render()`) deliberately evaluates spherical harmonics
    in plain Python/PyTorch and passes the RESULT to gsplat as flat RGB
    `colors` with `sh_degree=None` -- its own comment says why: "Python-side
    SH evaluation to bypass gsplat ROCm SH backward kernel." Production never
    calls gsplat's native CUDA/HIP SH backward kernel at all. This probe
    mirrors that: states carry flat `colors` (not sh0/shN split), and
    `sh_degree=None` is passed to rasterization() below, exactly like
    production. This means the fast-math flag's effect on SH gradients is
    NOT independently measured here (SH gradients in production flow through
    ordinary PyTorch autograd, unaffected by the ROCm hipcc build) -- the
    `colors` gradient captured per-state is what actually matters, since
    that's the real interface between gsplat's compiled kernels and the
    (unaffected) Python-side SH gradient that consumes it downstream.

    All tensors are CPU float32, built from a single fixed-seed generator
    consumed in a fixed order (states inserted in a fixed dict-insertion
    order below), so the two variant runs see byte-identical inputs.
    """
    g = torch.Generator().manual_seed(seed)
    states = {}

    def base(n, means, scales, quats=None, opac=None):
        if quats is None:
            quats = _rand_quats(n, g)
        if opac is None:
            opac = torch.sigmoid(torch.randn(n, generator=g) * 0.5)
        colors = torch.rand(n, 3, generator=g)
        return dict(
            means=means.float(),
            quats=quats.float(),
            scales=scales.float(),
            opacities=opac.float(),
            colors=colors.float(),
        )

    n = 256

    # 1. normal control state: modest scales, spread in front of camera
    means = torch.randn(n, 3, generator=g) * 0.5
    means[:, 2] = torch.rand(n, generator=g) * 3.0 + 1.0
    scales = torch.exp(torch.randn(n, 3, generator=g) * 0.3 - 2.0)  # ~0.02-0.3
    states["normal"] = base(n, means, scales)

    # 2. very small scales (near-degenerate covariance)
    means = torch.randn(n, 3, generator=g) * 0.5
    means[:, 2] = torch.rand(n, generator=g) * 3.0 + 1.0
    scales = torch.exp(torch.randn(n, 3, generator=g) * 0.2 - 12.0)  # ~1e-6 - 1e-5
    states["tiny_scales"] = base(n, means, scales)

    # 3. very large scales
    means = torch.randn(n, 3, generator=g) * 0.5
    means[:, 2] = torch.rand(n, generator=g) * 3.0 + 1.0
    scales = torch.exp(torch.randn(n, 3, generator=g) * 0.3 + 4.0)  # ~20-100
    states["huge_scales"] = base(n, means, scales)

    # 4. Gaussians very near the camera (z close to near_plane)
    means = torch.randn(n, 3, generator=g) * 0.3
    means[:, 2] = torch.rand(n, generator=g) * 0.02 + 0.011  # just past near_plane=0.01
    scales = torch.exp(torch.randn(n, 3, generator=g) * 0.3 - 2.0)
    states["near_camera"] = base(n, means, scales)

    # 5. off-axis but still moderate incidence (large lateral offset relative
    # to depth, but kept within an angle gsplat's tile-intersection code can
    # still index -- an EXTREME version of this (see off_axis_extreme below)
    # is used specifically to probe robustness, kept separate so a crash
    # there doesn't remove this milder, still-adversarial case from the report.
    means = torch.randn(n, 3, generator=g) * 0.2
    means[:, 0] = (torch.rand(n, generator=g) * 2 - 1) * 3.0
    means[:, 2] = torch.rand(n, generator=g) * 1.0 + 0.5
    scales = torch.exp(torch.randn(n, 3, generator=g) * 0.3 - 2.0)
    states["off_axis"] = base(n, means, scales)

    # 6. near-singular projected covariance: one scale axis tiny, others large
    means = torch.randn(n, 3, generator=g) * 0.5
    means[:, 2] = torch.rand(n, generator=g) * 3.0 + 1.0
    scales = torch.stack(
        [
            torch.exp(torch.randn(n, generator=g) * 0.1 - 10.0),  # tiny
            torch.exp(torch.randn(n, generator=g) * 0.2 + 1.0),   # large
            torch.exp(torch.randn(n, generator=g) * 0.2 + 1.0),   # large
        ],
        dim=-1,
    )
    states["near_singular_cov"] = base(n, means, scales)

    # 7. very low opacity (near-zero, gradient-starved regime)
    means = torch.randn(n, 3, generator=g) * 0.5
    means[:, 2] = torch.rand(n, generator=g) * 3.0 + 1.0
    scales = torch.exp(torch.randn(n, 3, generator=g) * 0.3 - 2.0)
    opac = torch.sigmoid(torch.randn(n, generator=g) * 0.3 - 8.0)  # ~1e-4 - 1e-3
    states["low_opacity"] = base(n, means, scales, opac=opac)

    # 8. very high opacity (near-saturated)
    means = torch.randn(n, 3, generator=g) * 0.5
    means[:, 2] = torch.rand(n, generator=g) * 3.0 + 1.0
    scales = torch.exp(torch.randn(n, 3, generator=g) * 0.3 - 2.0)
    opac = torch.sigmoid(torch.randn(n, generator=g) * 0.3 + 8.0)  # ~0.9997-1
    states["high_opacity"] = base(n, means, scales, opac=opac)

    return states


def run_state(name, state, viewmat, K, width, height, device):
    from gsplat.rendering import rasterization

    means = state["means"].to(device).clone().requires_grad_(True)
    quats = state["quats"].to(device).clone().requires_grad_(True)
    scales = state["scales"].to(device).clone().requires_grad_(True)
    opacities = state["opacities"].to(device).clone().requires_grad_(True)
    colors = state["colors"].to(device).clone().requires_grad_(True)

    # tile_size=16 (gsplat's own default) and sh_degree=None (flat RGB
    # colors, no gsplat-native SH backward kernel) deliberately match this
    # project's real renderer -- see the build_states() docstring above and
    # eligsir/mapping/renderer/gsplat_backend.py's own comment: "tile_size=8
    # causes NaN gradients on ROCm/gfx1151 with wave32-patched gsplat" and
    # "Python-side SH evaluation to bypass gsplat ROCm SH backward kernel."
    # An earlier version of this probe used tile_size=8 (gsplat's library
    # default) + sh_degree=1 (gsplat's native SH kernel) and hit a HIP
    # "Memory access fault" / SIGABRT on EVERY state including the
    # non-adversarial control -- i.e. it was exercising a ROCm kernel path
    # this project's own code already found broken and routes around in
    # production, not a fast-math-specific issue. See the findings doc's
    # methodology section for the full trace of that dead end.
    render_colors, render_alphas, meta = rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=viewmat,
        Ks=K,
        width=width,
        height=height,
        sh_degree=None,
        render_mode="RGB+ED",
        packed=True,
        tile_size=16,
    )

    rgb = render_colors[..., :3]
    depth = render_colors[..., 3:4]
    alpha = render_alphas

    torch.manual_seed(0)
    target_rgb = torch.rand_like(rgb)
    loss = (rgb - target_rgb).abs().mean() + 0.1 * depth.mean() + 0.05 * alpha.mean()

    loss.backward()

    grads = {}
    nonfinite = {}
    for pname, p in [
        ("means", means),
        ("quats", quats),
        ("scales", scales),
        ("opacities", opacities),
        ("colors", colors),
    ]:
        g = p.grad
        if g is None:
            grads[pname] = None
            nonfinite[pname] = None
        else:
            grads[pname] = g.detach().cpu().clone()
            nonfinite[pname] = int((~torch.isfinite(g)).sum().item())

    return {
        "crash": False,
        "loss": float(loss.detach().cpu().item()),
        "rgb": rgb.detach().cpu().clone(),
        "depth": depth.detach().cpu().clone(),
        "alpha": alpha.detach().cpu().clone(),
        "grads": grads,
        "nonfinite_grad_counts": nonfinite,
        "rgb_nonfinite": int((~torch.isfinite(rgb)).sum().item()),
        "depth_nonfinite": int((~torch.isfinite(depth)).sum().item()),
        "alpha_nonfinite": int((~torch.isfinite(alpha)).sum().item()),
    }


def _run_single_state(args):
    _states = build_states(args.seed)
    state = _states[args.state_name]
    device = torch.device(args.device)
    viewmat, K, width, height = _make_camera(device=args.device)
    torch.manual_seed(0)
    result = run_state(args.state_name, state, viewmat, K, width, height, device)
    torch.save(result, args.out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, help="label, e.g. strict or fastmath")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--single-state", dest="state_name", default=None,
                     help=argparse.SUPPRESS)  # internal: subprocess worker mode
    args = ap.parse_args()

    if args.state_name is not None:
        _run_single_state(args)
        return

    states = build_states(args.seed)
    input_hashes = {
        name: {k: _hash_tensor(v) for k, v in st.items()} for name, st in states.items()
    }

    results = {}
    for name in states:
        tmp_out = f"{args.out}.{name}.tmp.pt"
        cmd = [
            sys.executable, __file__,
            "--variant", args.variant,
            "--out", tmp_out,
            "--seed", str(args.seed),
            "--device", args.device,
            "--single-state", name,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            results[name] = {
                "crash": True,
                "returncode": proc.returncode,
                "stderr_tail": proc.stderr[-2000:],
                "stdout_tail": proc.stdout[-2000:],
            }
            print(f"[{name}] CRASHED (returncode={proc.returncode})")
        else:
            results[name] = torch.load(tmp_out, weights_only=False)
            print(f"[{name}] OK loss={results[name]['loss']:.6f}")
            import os
            os.remove(tmp_out)

    import gsplat

    out = {
        "variant": args.variant,
        "seed": args.seed,
        "gsplat_version": getattr(gsplat, "__version__", "unknown"),
        "torch_version": torch.__version__,
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "input_hashes": input_hashes,
        "results": results,
    }
    torch.save(out, args.out)
    print(f"Saved probe results for variant={args.variant} to {args.out}")
    print(json.dumps({name: r.get("crash", False) for name, r in results.items()}))


if __name__ == "__main__":
    main()
