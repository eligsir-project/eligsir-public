"""Issue #1259: a non-finite per-Gaussian lifecycle utility must not crash the
run.

Root cause: ``ray_preserving`` resolution resampling co-occurred (in one real
GPU comparison run) with the same ``ValueError: lifecycle utility must be a
finite per-Gaussian tensor`` previously seen with ``--lambda_opacity_entropy
0``. The two triggers are unrelated at the surface (a resampling mode vs. a
loss weight) but both can leave a small number of degenerate Gaussians (e.g.
an exploded scale) whose ``compute_gaussian_utility`` output overflows to
``inf``/``nan`` for those rows specifically -- ``_utility_core``'s squared
scale penalty and the aniso-ratio penalty in
``utils/energy_mcmc.py::compute_gaussian_utility`` both square an unbounded
quantity with no upstream NaN required.

The old behavior (`_advance_lifecycle` in
``eligsir/runtime/production_application.py``) treated ANY non-finite
entry, however small a fraction, as whole-population corruption and raised --
crashing the entire run over what is typically one or a handful of degenerate
Gaussians. This test asserts the new, narrower behavior: non-finite entries
are excluded from that scan's EMA update (their prior state is left alone,
finite, and untouched), the scan otherwise proceeds normally, and only a
utility tensor that is non-finite EVERYWHERE still raises (real, whole-model
corruption that a per-Gaussian mask cannot repair).
"""

from __future__ import annotations

import math

import torch

from eligsir.runtime.production_application import CanonicalMappingRuntime


class _FakePolicy:
    streaming_lifecycle_enabled = True
    streaming_mature_min_utility = 0.1
    streaming_utility_ema_beta = 0.9
    streaming_min_support_views = 2
    streaming_mature_age_frames = 15
    streaming_freeze_age_frames = -1


class _FakeGaussians:
    def __init__(self, n: int):
        self.lifecycle_state = torch.zeros(n, dtype=torch.int8)
        self.birth_iter = torch.zeros(n, dtype=torch.long)
        self.provisional = torch.zeros(n, dtype=torch.bool)
        self.utility_ema = torch.full((n,), 0.05)


class _FakeRuntime:
    """Minimal stand-in exposing only what ``_advance_lifecycle`` reads."""

    def __init__(self, gaussians: "_FakeGaussians"):
        self._optimization_policy = _FakePolicy()
        self.gaussians = gaussians
        self._lifecycle_transition_calls = 0

    def _versions(self):
        return (0, 0)


def _run(utility: torch.Tensor, n: int = 5):
    gaussians = _FakeGaussians(n)
    rt = _FakeRuntime(gaussians)
    CanonicalMappingRuntime._advance_lifecycle(
        rt,
        utility=utility,
        optimizer_updates=100,
        expected_topology_version=0,
    )
    return gaussians


def test_all_finite_utility_updates_ema_normally():
    utility = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    gaussians = _run(utility)
    beta = _FakePolicy.streaming_utility_ema_beta
    expected = 0.05 * beta + utility.clamp(min=0.0) * (1 - beta)
    assert torch.allclose(gaussians.utility_ema, expected)


def test_partial_nonfinite_utility_excludes_bad_rows_not_the_whole_scan():
    utility = torch.tensor([1.0, float("nan"), 3.0, float("inf"), 5.0])
    gaussians = _run(utility)

    beta = _FakePolicy.streaming_utility_ema_beta
    # Finite rows (0, 2, 4) update normally.
    for idx in (0, 2, 4):
        expected = 0.05 * beta + float(utility[idx]) * (1 - beta)
        assert math.isclose(float(gaussians.utility_ema[idx]), expected, rel_tol=1e-6)
    # Non-finite rows (1, 3) are excluded -- prior state (0.05) untouched.
    for idx in (1, 3):
        assert math.isclose(float(gaussians.utility_ema[idx]), 0.05, rel_tol=1e-5)
    # The whole buffer must stay finite -- no NaN/inf ever written.
    assert bool(torch.isfinite(gaussians.utility_ema).all())


def test_all_nonfinite_utility_still_raises():
    utility = torch.full((4,), float("nan"))
    try:
        _run(utility, n=4)
    except ValueError as exc:
        assert "finite per-Gaussian tensor" in str(exc)
    else:
        raise AssertionError("expected ValueError for whole-population non-finite utility")


def test_wrong_shape_utility_still_raises():
    utility = torch.tensor([1.0, 2.0])
    try:
        _run(utility, n=5)
    except ValueError as exc:
        assert "finite per-Gaussian tensor" in str(exc)
    else:
        raise AssertionError("expected ValueError for shape mismatch")
