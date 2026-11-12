"""Bit-exactness tests for geometry.depth_filtering.fast_nanmedian_axis0 vs.
np.nanmedian(..., axis=0) -- the CPU optimization in issue #270/#276
(accumulate_observed_surface's _stable_depth_samples median filter)."""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from geometry.depth_filtering import fast_nanmedian_axis0


def _reference(x: np.ndarray) -> np.ndarray:
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmedian(x, axis=0)


def _assert_equal(a: np.ndarray, b: np.ndarray) -> None:
    assert a.shape == b.shape
    both_nan = np.isnan(a) & np.isnan(b)
    assert np.array_equal(np.isnan(a), np.isnan(b)), "NaN pattern mismatch"
    np.testing.assert_allclose(a[~both_nan], b[~both_nan], rtol=0, atol=1e-6)


@pytest.mark.parametrize("seed", range(10))
def test_matches_nanmedian_random_sparse_nans(seed):
    rng = np.random.default_rng(seed)
    x = rng.normal(3.0, 0.05, size=(25, 90, 160)).astype(np.float32)
    x[rng.random(x.shape) < 0.15] = np.nan
    _assert_equal(fast_nanmedian_axis0(x), _reference(x))


def test_all_nan_column_becomes_nan():
    x = np.full((9, 4, 4), np.nan, dtype=np.float32)
    out = fast_nanmedian_axis0(x)
    assert np.all(np.isnan(out))
    _assert_equal(out, _reference(x))


def test_single_valid_value_column():
    x = np.full((9, 1, 1), np.nan, dtype=np.float32)
    x[3, 0, 0] = 5.0
    out = fast_nanmedian_axis0(x)
    assert out[0, 0] == pytest.approx(5.0)
    _assert_equal(out, _reference(x))


def test_no_nans_matches_plain_median():
    rng = np.random.default_rng(42)
    x = rng.normal(0.0, 1.0, size=(25, 30, 30)).astype(np.float32)
    _assert_equal(fast_nanmedian_axis0(x), _reference(x))


def test_even_and_odd_valid_counts():
    # Column 0: even count of valid values (2), column 1: odd count (3).
    x = np.array(
        [
            [[1.0, 10.0]],
            [[3.0, 20.0]],
            [[np.nan, 30.0]],
            [[np.nan, np.nan]],
        ],
        dtype=np.float32,
    )
    out = fast_nanmedian_axis0(x)
    _assert_equal(out, _reference(x))


def test_empty_axis_returns_all_nan():
    x = np.empty((0, 3, 3), dtype=np.float32)
    out = fast_nanmedian_axis0(x)
    assert out.shape == (3, 3)
    assert np.all(np.isnan(out))
