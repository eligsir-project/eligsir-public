"""W3-61: production wiring tests for ``_sanitize_gaussian_params_sampled``.

W3-34B (docs/architecture/2026-08-10-w3-34b-sanitize-graph-roi.md) proved
the branch-free numeric core (``_sanitize_gaussian_params_branch_free``) is
bit-identical to production ``_sanitize_gaussian_params`` but left it
unwired. W3-61 wires it into ``train_streaming.py``'s 3 real call sites via
a new wrapper, ``_sanitize_gaussian_params_sampled``
(``eligsir/mapping/integration/loss_diagnostics.py``), which:

  1. Always runs the branch-free numeric core -- every call, every
     iteration, regardless of sampling. This is the behavior-preserving
     part: production's actual neutralization of non-finite/exp-overflowing
     state never becomes sampled.
  2. Runs the diagnostics-only (counting/rate-limited-print) half only
     during an early burn-in window or at a sampled interval afterward.
     This is the real detection-vs-speed trade -- see the wrapper's own
     docstring and the W3-61 findings doc for the reasoning.

These tests run entirely on CPU (no GPU needed) and check the WIRING
(cadence gating, numeric correction always firing) rather than re-proving
bit-identical parity, which W3-34B's ``test_w3_34b_sanitize_parity.py``
already covers for the branch-free core itself.
"""

from __future__ import annotations

import torch

from eligsir.mapping.integration.loss_diagnostics import (
    _SANITIZE_DIAG_BURN_IN_ITERS,
    _SANITIZE_DIAG_DEFAULT_INTERVAL,
    _sanitize_gaussian_params_diagnostics_only,
    _sanitize_gaussian_params_sampled,
)

_TENSOR_NAMES = ("_xyz", "_opacity", "_scaling", "_rotation", "_features_dc", "_features_rest")


class _FakeGaussians:
    """Same minimal stand-in shape as test_w3_34b_sanitize_parity.py's
    fixture -- exposes exactly the 6 raw tensors sanitize touches plus the
    derived ``get_scaling`` property the diagnostics half reads."""

    def __init__(self, values_by_name):
        for name in _TENSOR_NAMES:
            setattr(self, name, values_by_name.get(name))

    @property
    def get_scaling(self):
        return torch.exp(self._scaling)


def _healthy_gaussians(n=8):
    return _FakeGaussians({name: torch.randn(n, 3) for name in _TENSOR_NAMES})


def _corrupted_gaussians(n=8):
    values = {name: torch.randn(n, 3) for name in _TENSOR_NAMES}
    values["_xyz"][0, 0] = float("nan")
    values["_scaling"][1, 1] = 20.0  # over the 15.0 clamp threshold
    return _FakeGaussians(values)


def _reset_diagnostics_counts():
    # Diagnostics-only's rate limiter is a module-level function attribute
    # (mirrors production _sanitize_gaussian_params's own _counts pattern) --
    # reset it between tests so one test's occurrence count doesn't leak
    # into another's assertions.
    if hasattr(_sanitize_gaussian_params_diagnostics_only, "_counts"):
        _sanitize_gaussian_params_diagnostics_only._counts.clear()


def test_default_constants_match_documented_reasoning():
    # Burn-in covers the floor2 init-seed explosion incident class (bbox
    # explosion within the first ~2000 iterations, see CLAUDE.md); the
    # sampled interval afterward reuses --scalar_log_interval's own default
    # (100) rather than inventing a new number.
    assert _SANITIZE_DIAG_BURN_IN_ITERS == 2000
    assert _SANITIZE_DIAG_DEFAULT_INTERVAL == 100


def test_numeric_correction_always_fires_even_when_diagnostics_sampled_out():
    """The behavior-preserving guarantee: even at an iteration where the
    diagnostics-only half is skipped (outside burn-in, not on the sampled
    interval), the branch-free numeric core must still have run and fixed
    the corrupted tensors -- sampling must never weaken the actual
    mitigation, only visibility into how often it fired."""
    g = _corrupted_gaussians()
    iteration = _SANITIZE_DIAG_BURN_IN_ITERS + 1  # past burn-in
    assert iteration % _SANITIZE_DIAG_DEFAULT_INTERVAL != 0  # not on the sampled tick either

    _sanitize_gaussian_params_sampled(g, "test", iteration)

    assert bool(torch.isfinite(g._xyz).all())
    assert float(g._scaling[1, 1]) <= 15.0


def test_diagnostics_run_every_iteration_during_burn_in():
    _reset_diagnostics_counts()
    for it in range(1, 6):
        g = _corrupted_gaussians()
        _sanitize_gaussian_params_sampled(g, "burn_in_probe", it, _limit=100)
    counts = _sanitize_gaussian_params_diagnostics_only._counts
    assert counts.get("burn_in_probe") == 5


def test_diagnostics_sampled_out_between_ticks_after_burn_in():
    _reset_diagnostics_counts()
    base = _SANITIZE_DIAG_BURN_IN_ITERS
    interval = _SANITIZE_DIAG_DEFAULT_INTERVAL
    # First tick right after burn-in ends, then a handful of off-cadence
    # iterations that should NOT increment the diagnostic counter.
    on_tick_iter = base + interval  # e.g. 2100, divisible by 100
    off_tick_iters = [on_tick_iter + 1, on_tick_iter + 2, on_tick_iter + 3]

    g_on = _corrupted_gaussians()
    _sanitize_gaussian_params_sampled(g_on, "post_burn_in_probe", on_tick_iter, _limit=100)
    counts = _sanitize_gaussian_params_diagnostics_only._counts
    assert counts.get("post_burn_in_probe") == 1

    for it in off_tick_iters:
        g_off = _corrupted_gaussians()
        _sanitize_gaussian_params_sampled(g_off, "post_burn_in_probe", it, _limit=100)
        # Numeric correction still ran even though diagnostics did not.
        assert bool(torch.isfinite(g_off._xyz).all())
    counts = _sanitize_gaussian_params_diagnostics_only._counts
    assert counts.get("post_burn_in_probe") == 1  # unchanged by the 3 off-tick calls


def test_healthy_input_never_triggers_diagnostic_print(capsys):
    _reset_diagnostics_counts()
    g = _healthy_gaussians()
    for it in range(1, 4):
        _sanitize_gaussian_params_sampled(g, "healthy_probe", it, _limit=100)
    out = capsys.readouterr().out
    assert "healthy_probe" not in out


def test_custom_diag_interval_is_respected():
    _reset_diagnostics_counts()
    custom_interval = 7
    hits = 0
    for it in range(1, 22):
        g = _corrupted_gaussians()
        _sanitize_gaussian_params_sampled(
            g, "custom_interval_probe", it,
            diag_interval=custom_interval, burn_in_iters=0, _limit=100,
        )
    counts = _sanitize_gaussian_params_diagnostics_only._counts
    # iterations 7, 14, 21 within range(1, 22) -> 3 hits, with burn-in disabled.
    assert counts.get("custom_interval_probe") == 3
