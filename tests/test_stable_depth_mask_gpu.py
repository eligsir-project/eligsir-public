"""Numerical equivalence: GPU stable-depth mask vs the numpy reference.

The GPU port (_sample_stable_depth_mask_gpu) must match the numpy
_sample_stable_depth_mask within a tiny mismatch budget (the only allowed
divergence is even-window nanmedian tie-breaking — torch returns the lower
middle, numpy averages the two middles).
"""
import numpy as np
import torch
import pytest

from train_streaming import _sample_stable_depth_mask, _sample_stable_depth_mask_gpu


def _make_depth(H=240, W=320, seed=0):
    rng = np.random.default_rng(seed)
    z = rng.uniform(0.3, 5.0, size=(H, W)).astype(np.float32)
    # holes (0/neg), NaNs, and out-of-range to exercise validity masking
    z[rng.random((H, W)) < 0.1] = 0.0
    z[rng.random((H, W)) < 0.05] = np.nan
    z[rng.random((H, W)) < 0.03] = 7.0
    return z


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU")
@pytest.mark.parametrize("seed", [0, 1, 2, 3])
@pytest.mark.parametrize("window,erode,mvr,mr,med", [
    (5, 2, 0.75, 0.08, 0.04),   # production defaults
    (3, 1, 0.5, 0.05, 0.02),
    (5, 0, 0.0, 0.08, 0.0),     # only range filter
    (1, 0, 0.0, 0.0, 0.04),     # only median (window=1)
])
def test_gpu_matches_numpy(seed, window, erode, mvr, mr, med):
    z = _make_depth(seed=seed)
    H, W = z.shape
    stride = 4
    ys, xs = np.mgrid[0:H:stride, 0:W:stride]
    kw = dict(erode_px=erode, window=window, min_valid_ratio=mvr,
              max_range=mr, median_thresh=med)

    cpu = _sample_stable_depth_mask(z, xs, ys, 0.1, 6.0, **kw)

    dev = torch.device("cuda")
    z_t = torch.from_numpy(z).to(dev)
    ys_t = torch.from_numpy(ys.astype(np.float32)).to(dev)
    xs_t = torch.from_numpy(xs.astype(np.float32)).to(dev)
    gpu = _sample_stable_depth_mask_gpu(z_t, xs_t, ys_t, 0.1, 6.0, **kw).cpu().numpy()

    assert gpu.shape == cpu.shape
    mismatch = int((gpu != cpu).sum())
    frac = mismatch / cpu.size
    # Allow a tiny budget for nanmedian tie-breaking on even valid-counts.
    assert frac < 0.005, f"mismatch frac {frac:.4f} ({mismatch}/{cpu.size}) too high"
