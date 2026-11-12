"""Regression test for issue #1259/#1284.

A real ScanNet++ campaign run on r9700 hit "overflow encountered in
multiply" + "invalid value encountered in cast" at the exact uint8-cast
return line in ``_to_uint8_hwc``, THEN the process went silent (no
traceback, no [TERMINATION] marker, dead not hung) -- despite the
torch-side ``nan_to_num``/``clamp`` guard already being present at the
pinned commit that ran. A mathematically clamped [0, 1] array cannot
overflow on ``* 255.0``, so whatever reached the final cast on that run
was not actually clamped -- the leading hypothesis is a ROCm/HIP kernel
gap where ``torch.nan_to_num``/``clamp`` silently no-ops for some
dtype/shape on that hardware, rather than a bug in the guard logic itself
(reproduced clean on this host's torch/numpy for every synthetic
NaN/Inf/huge-finite/int64/float16-max case -- see the module comment
above ``_NAN_SANITIZE_STATS``).

This test cannot reproduce a ROCm-specific kernel gap on CPU/CUDA
hardware, so it instead verifies the second, independent numpy-level
safety net added for exactly this scenario: even if the torch-side
sanitize+clamp is bypassed entirely (monkeypatched away here to simulate
that), ``_to_uint8_hwc`` must still produce a finite, valid uint8 array
--  sanitizing and logging loudly -- rather than letting a NaN/Inf-tainted
array silently reach the final cast.
"""

import numpy as np
import pytest
import torch

import eligsir.utils.comparison_report as comparison_report
from eligsir.utils.comparison_report import (
    _to_uint8_hwc,
    get_nan_sanitize_report,
    reset_nan_sanitize_tracking,
)


@pytest.fixture(autouse=True)
def _reset_tracking():
    reset_nan_sanitize_tracking()
    yield
    reset_nan_sanitize_tracking()


def test_numpy_level_guard_catches_bypassed_torch_guard(monkeypatch, capsys):
    """Simulate the torch-side guard being a no-op (the suspected ROCm gap)."""

    def _fake_nan_to_num(t, nan=0.0, posinf=1.0, neginf=0.0):
        # Deliberately do NOT sanitize -- simulates a kernel that silently
        # leaves NaN/Inf untouched instead of replacing them.
        return t

    monkeypatch.setattr(comparison_report.torch, "nan_to_num", _fake_nan_to_num)

    img = torch.zeros(3, 4, 4)
    img[0, 0, 0] = float("nan")
    img[1, 0, 1] = float("inf")
    img[2, 0, 2] = float("-inf")

    out = _to_uint8_hwc(img, label="test_view_0042")

    assert out.dtype == np.uint8
    assert np.isfinite(out).all()
    assert out[0, 0, 0] == 0
    # torch.clamp (unlike nan_to_num) DOES bound +/-inf correctly even with
    # the nan_to_num guard bypassed -- only NaN (which compares false to
    # both clamp bounds) survives to reach the numpy-level second guard.
    assert out[0, 1, 1] == 255
    assert out[0, 2, 2] == 0

    captured = capsys.readouterr()
    assert "test_view_0042" in captured.out
    assert "nan=" in captured.out

    report = get_nan_sanitize_report()
    assert report is not None
    assert report["n_sanitized"] == 1
    assert report["examples"][0]["label"] == "test_view_0042"
    assert report["examples"][0]["nan_count"] == 1


def test_numpy_level_guard_catches_fully_bypassed_torch_guard(monkeypatch, capsys):
    """Simulate BOTH nan_to_num and clamp being no-ops (the exact overflow
    shape reported in issue #1284: a genuinely huge/inf value reaching the
    ``* 255.0`` multiply causes numpy's "overflow encountered in multiply",
    immediately followed by "invalid value encountered in cast")."""

    def _identity(t, *a, **kw):
        return t

    monkeypatch.setattr(comparison_report.torch, "nan_to_num", _identity)
    monkeypatch.setattr(torch.Tensor, "clamp", lambda self, *a, **kw: self)

    img = torch.zeros(3, 4, 4)
    img[0, 0, 0] = float("nan")
    img[1, 0, 1] = float("inf")
    img[2, 0, 2] = 1e30  # huge finite, would overflow float32 * 255.0

    out = _to_uint8_hwc(img, label="fully_bypassed_view")

    assert out.dtype == np.uint8
    assert np.isfinite(out).all()

    report = get_nan_sanitize_report()
    assert report is not None
    assert report["examples"][0]["nan_count"] == 1
    assert report["examples"][0]["posinf_count"] >= 1

    captured = capsys.readouterr()
    assert "fully_bypassed_view" in captured.out


def test_clean_input_leaves_sanitize_report_none():
    img = torch.full((3, 2, 2), 0.5)
    out = _to_uint8_hwc(img, label="clean_view")
    assert np.all(out == 128)
    assert get_nan_sanitize_report() is None


def test_reset_clears_accumulated_examples(monkeypatch):
    def _fake_nan_to_num(t, nan=0.0, posinf=1.0, neginf=0.0):
        return t

    monkeypatch.setattr(comparison_report.torch, "nan_to_num", _fake_nan_to_num)

    img = torch.zeros(3, 2, 2)
    img[0, 0, 0] = float("nan")
    _to_uint8_hwc(img, label="leaky_view")
    assert get_nan_sanitize_report() is not None

    reset_nan_sanitize_tracking()
    assert get_nan_sanitize_report() is None
