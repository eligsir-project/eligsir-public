"""Regression test: _to_uint8_hwc must not emit NaN-cast warnings.

Covers a real failure seen on a live run: a NaN pixel in a rendered RGB
tensor (e.g. from a numerically unstable Gaussian) survived
``.clamp(0, 1)`` (NaN compares false to both bounds) and triggered
"invalid value encountered in cast" when cast to uint8.
"""

import warnings

import numpy as np
import torch

from eligsir.utils.comparison_report import _to_uint8_hwc


def test_to_uint8_hwc_sanitizes_nan_and_inf():
    img = torch.zeros(3, 4, 4)
    img[0, 0, 0] = float("nan")
    img[1, 0, 1] = float("inf")
    img[2, 0, 2] = float("-inf")

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = _to_uint8_hwc(img)

    assert out.dtype == np.uint8
    assert not np.isnan(out).any()
    assert out[0, 0, 0] == 0
    assert out[0, 1, 1] == 255
    assert out[0, 2, 2] == 0


def test_to_uint8_hwc_normal_values_unaffected():
    img = torch.full((3, 2, 2), 0.5)
    out = _to_uint8_hwc(img)
    assert out.dtype == np.uint8
    assert np.all(out == 128)
