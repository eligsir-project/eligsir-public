"""Fast, bit-exact numpy replacements for hot-path depth-filtering statistics.

Extracted for issue #270 / #276 ("[Kornia gate] Optimize and re-profile
accumulate_observed_surface before decode promotion"). Pure numpy, no
torch/torchvision/gsplat import chain -- safe to unit-test on any host,
independent of the ROCm/gsplat import chain used elsewhere in this repo.
"""
from __future__ import annotations

import numpy as np


def fast_nanmedian_axis0(x: np.ndarray) -> np.ndarray:
    """Bit-exact-equivalent, faster replacement for ``np.nanmedian(x, axis=0)``.

    ``np.nanmedian`` on an array containing NaNs routes through
    ``numpy.ma``'s masked-array machinery, which is measurably slow even on
    small arrays (profiled at ~5x this function's cost on a realistic
    ``StreamingScene._stable_depth_samples`` window-stack shape, see
    ``docs/architecture/2026-08-13-issue270-accumulate-observed-surface-cpu-profile.md``).
    This implementation sorts along axis 0 (NaNs sort to the end in numpy's
    ascending sort) and picks the middle element(s) by the per-column count
    of finite values, which is exactly what ``np.nanmedian`` computes for
    the even/odd-count and all-NaN cases -- verified equal on randomized
    inputs (including all-NaN and single-valid columns) in
    ``tests/test_fast_nanmedian.py``.

    Only supports ``axis=0`` reduction (the only call site needs this), and
    only float arrays (NaN comparisons on integer dtypes are undefined).
    """
    x = np.asarray(x)
    n = x.shape[0]
    if n == 0:
        return np.full(x.shape[1:], np.nan, dtype=x.dtype)
    finite = ~np.isnan(x)
    counts = finite.sum(axis=0)
    xs = np.sort(x, axis=0)  # NaN sorts to the end of ascending order in numpy
    lo = np.clip((counts - 1) // 2, 0, n - 1)
    hi = np.clip(counts // 2, 0, n - 1)
    lo_val = np.take_along_axis(xs, lo[None, ...], axis=0)[0]
    hi_val = np.take_along_axis(xs, hi[None, ...], axis=0)[0]
    med = (lo_val + hi_val) * np.array(0.5, dtype=x.dtype)
    med = np.where(counts == 0, np.array(np.nan, dtype=x.dtype), med)
    return med.astype(x.dtype, copy=False)
