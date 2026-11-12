#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

_SSIM_WINDOW_CACHE = {}

# SSIM backend dispatch. The conv-based reference SSIM (`_ssim`) costs ~3.6 ms
# fwd+bwd at 640x480 on gfx1201 and is a large slice of the per-iteration
# launch-bound critical path. Two opt-in alternatives (benchmarked separately):
#   "fused"    — the fused_ssim HIP kernel: ~0.22 ms fwd+bwd (16x), bit-identical
#                value, grad cos-sim ~1.0. Requires the `fused_ssim` package.
#   "compiled" — torch.compile(dynamic=True) of the conv SSIM: fuses the 5 convs +
#                elementwise into fewer kernel launches. dynamic=True avoids
#                recompiles as the MCMC Gaussian count N changes.
# Default "ref" keeps the exact previous behavior.
_SSIM_MODE = "ref"
_FUSED_SSIM_FN = None
_FUSED_SSIM_MAP_FN = None
_COMPILED_SSIM = None


def _reference_ssim_window(img, window_size, channel):
    key = (window_size, channel, img.device, img.dtype)
    window = _SSIM_WINDOW_CACHE.get(key)
    if window is None:
        window = create_window(window_size, channel).to(
            device=img.device, dtype=img.dtype
        )
        _SSIM_WINDOW_CACHE[key] = window
    return window


def evaluation_ssim(img1, img2, window_size=11, size_average=True):
    """Fixed reference SSIM for scientific evaluation.

    Unlike :func:`ssim`, this never consults the mutable training-loss backend.
    """
    channel = img1.size(-3)
    window = _reference_ssim_window(img1, window_size, channel)
    return _ssim(img1, img2, window, window_size, channel, size_average)


def set_ssim_mode(mode, dynamic=True):
    """Select the SSIM backend. Returns the mode actually activated (falls back
    to "ref" if the requested backend is unavailable)."""
    global _SSIM_MODE, _FUSED_SSIM_FN, _FUSED_SSIM_MAP_FN, _COMPILED_SSIM
    if mode == "fused":
        try:
            from fused_ssim import fused_ssim as _f
            from fused_ssim import FusedSSIMMap
            _FUSED_SSIM_FN = _f
            _FUSED_SSIM_MAP_FN = FusedSSIMMap.apply
            _SSIM_MODE = "fused"
        except Exception as exc:  # noqa: BLE001 - missing/broken extension -> fall back
            print(f"[ssim] fused_ssim unavailable ({exc}); falling back to ref SSIM", flush=True)
            _SSIM_MODE = "ref"
            _FUSED_SSIM_MAP_FN = None
    elif mode == "compiled":
        try:
            _COMPILED_SSIM = torch.compile(_ssim, dynamic=dynamic)
            _SSIM_MODE = "compiled"
            _FUSED_SSIM_MAP_FN = None
        except Exception as exc:  # noqa: BLE001
            print(f"[ssim] torch.compile unavailable ({exc}); falling back to ref SSIM", flush=True)
            _SSIM_MODE = "ref"
            _FUSED_SSIM_MAP_FN = None
    else:
        _SSIM_MODE = "ref"
        _FUSED_SSIM_MAP_FN = None
    return _SSIM_MODE


def configure_training_ssim(args):
    """Activate and report the SSIM backend requested by a training policy.

    Every production trainer, including the canonical causal runtime, must use
    this single selector.  Recording ``fused_ssim=True`` without crossing this
    boundary leaves :func:`ssim` on the reference convolution path while the
    run metadata incorrectly claims a fused kernel.
    """

    if args.fused_ssim:
        mode = set_ssim_mode("fused")
    elif args.compile_loss:
        mode = set_ssim_mode(
            "compiled", dynamic=args.compile_dynamic
        )
    else:
        mode = set_ssim_mode("ref")
    print(f"[ssim] backend={mode}", flush=True)
    return mode


def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)

    # Fused HIP kernel path: only valid for the standard 11-window, mean-reduced
    # call (the training loss). fused_ssim wants a 4D (B,C,H,W) batch; the loop
    # passes a 3D (C,H,W) image, so add a batch dim when needed.
    if (_SSIM_MODE == "fused" and _FUSED_SSIM_FN is not None
            and window_size == 11 and size_average):
        a = img1 if img1.dim() == 4 else img1.unsqueeze(0)
        b = img2 if img2.dim() == 4 else img2.unsqueeze(0)
        return _FUSED_SSIM_FN(a, b, padding="same", train=torch.is_grad_enabled())

    # Cache the Gaussian window per (size, channel, device, dtype). ssim() runs every
    # training iteration; rebuilding the window on the CPU + copying it H2D each call
    # was pure waste on the single-threaded critical path (the loop is launch/CPU-bound).
    window = _reference_ssim_window(img1, window_size, channel)

    if _SSIM_MODE == "compiled" and _COMPILED_SSIM is not None and size_average:
        return _COMPILED_SSIM(img1, img2, window, window_size, channel, size_average)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def _batched_ssim_map(img1, img2, window_size):
    """Return SSIM map [B,C,H,W] for a batch.

    In fused mode with window_size==11, uses the fused HIP kernel.
    Otherwise uses the reference convolution path.
    """
    if _SSIM_MODE == "fused" and _FUSED_SSIM_MAP_FN is not None and window_size == 11:
        # fused_ssim.FusedSSIMMap.apply signature:
        # (C1, C2, img1, img2, padding, train) -> ssim_map [B,C,H,W]
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        return _FUSED_SSIM_MAP_FN(
            C1, C2,
            img1.contiguous(), img2.contiguous(),
            "same",
            torch.is_grad_enabled()
        )
    else:
        # Reference path: returns [B,C,H,W] map
        channel = img1.size(1)
        window = _reference_ssim_window(img1, window_size, channel)
        return _ssim_map(img1, img2, window, window_size, channel)


def ssim_per_view(img1, img2, window_size=11):
    """Return one SSIM scalar per batch item, with non-finite views excluded up front.

    Up-front exclusion prevents NaN propagation: in a batched call, backward through
    a non-finite view can multiply its NaN derivative by a zero upstream gradient,
    which reaches Gaussian parameters as NaN through the rasterizer backward.
    Per-view filtering ensures only finite inputs share an autograd node; non-finite
    views are processed individually and kept out of the graph.

    Returns:
        tuple[torch.Tensor, ...]: B 0-d scalars in view order (one per input view).
    """
    if img1.dim() != 4 or img1.shape != img2.shape:
        raise ValueError("ssim_per_view requires equal [B,C,H,W] tensors")

    # Detect non-finite views up front (one host readback, replaces B-1 kernel calls)
    ok = torch.isfinite(img1).flatten(1).all(dim=1) & torch.isfinite(img2).flatten(1).all(dim=1)
    ok_flags = ok.tolist()

    # Process finite views in a single batched call
    results = [None] * img1.shape[0]

    if all(ok_flags):
        # All views are finite: one batched call
        ssim_map = _batched_ssim_map(img1, img2, window_size)
        ssim_vec = ssim_map.mean(dim=(1, 2, 3))  # [B]
        for i in range(img1.shape[0]):
            results[i] = ssim_vec[i]
    else:
        # Mixed: batch the finite subset, process non-finite individually
        idx_finite = torch.tensor(
            [i for i, f in enumerate(ok_flags) if f],
            device=img1.device,
            dtype=torch.long
        )

        if len(idx_finite) > 0:
            # Batched call on finite subset
            img1_finite = img1.index_select(0, idx_finite)
            img2_finite = img2.index_select(0, idx_finite)
            ssim_map = _batched_ssim_map(img1_finite, img2_finite, window_size)
            ssim_vec = ssim_map.mean(dim=(1, 2, 3))  # [K]
            for k, i in enumerate(idx_finite.tolist()):
                results[i] = ssim_vec[k]

        # Non-finite views: individual calls (isolated from graph)
        for i in range(img1.shape[0]):
            if not ok_flags[i]:
                results[i] = ssim(img1[i : i + 1], img2[i : i + 1], window_size)

    return tuple(results)

def _ssim_map(img1, img2, window, window_size, channel):
    """Compute SSIM map [B,C,H,W] without reduction."""
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    ssim_map = _ssim_map(img1, img2, window, window_size, channel)

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)
