"""Predicted-GPU-cost proxy for the utility formula's ``w_cost`` term (plan
section 11.5).

Deliberately a simple, honest proxy -- not a real profiled cost model. This
PR (W3-06) is shadow-only observation/logging; a real calibrated cost model
(e.g. regressed against ``realized_gain.py``'s recorded
``compute_consumed_ms``) is future work once enough realized-gain data
exists. The model here captures the two facts that dominate a training
step's wall-clock cost in this codebase (see CLAUDE.md's Parallelism
profile / resolution-cache sections): it scales with the live Gaussian
count ``n_gaussians`` (more splats to rasterize/backprop through) and with
the active resolution tier (``"high"`` renders/backprops far more pixels
than ``"low"``).

Returned value is normalized to roughly ``[0, 1]`` via a fixed reference
scale (``reference_n_gaussians``) so it composes cleanly with the other
``[0, 1]``-ish utility components in the formula -- it is not itself a
millisecond estimate (see ``predicted_cost_ms_estimate`` for that, used
only for human-readable logging, never fed into the formula).
"""

from __future__ import annotations

# Empirically-informed (not measured-in-this-PR) relative per-Gaussian cost
# multipliers by resolution tier -- "high" renders/backprops substantially
# more pixels per Gaussian footprint than "low". Kept as named constants
# rather than inline magic numbers so a future calibration pass has one
# place to update.
_TIER_COST_MULTIPLIER = {
    "low": 0.35,
    "default": 1.0,
    "high": 1.8,
}

# Reference Gaussian count at which the normalized cost proxy crosses 1.0
# for the "default" tier -- chosen to sit near the middle of this project's
# common --cap_max range (200k-600k, see CLAUDE.md), not derived from a
# real profiling run.
_REFERENCE_N_GAUSSIANS = 300_000.0

# Rough per-Gaussian millisecond cost at the reference tier, for the
# human-readable millisecond estimate only (never used in the normalized
# score). Order-of-magnitude only.
_MS_PER_GAUSSIAN_AT_REFERENCE = 0.00025


def predict_gpu_cost(n_gaussians: int, resolution_bucket: str = "default") -> float:
    """Normalized ``[0, ~1.8]`` proxy cost, ``n_gaussians`` linear, scaled by
    tier multiplier and the fixed reference count. Never negative; ``0.0``
    for an empty/degenerate map.
    """
    if n_gaussians <= 0:
        return 0.0
    multiplier = _TIER_COST_MULTIPLIER.get(resolution_bucket, _TIER_COST_MULTIPLIER["default"])
    return float(n_gaussians) / _REFERENCE_N_GAUSSIANS * multiplier


def predicted_cost_ms_estimate(n_gaussians: int, resolution_bucket: str = "default") -> float:
    """Human-readable millisecond estimate for logging only -- not part of
    the utility formula (which uses :func:`predict_gpu_cost`'s normalized
    value via ``w_cost``).
    """
    if n_gaussians <= 0:
        return 0.0
    multiplier = _TIER_COST_MULTIPLIER.get(resolution_bucket, _TIER_COST_MULTIPLIER["default"])
    return float(n_gaussians) * _MS_PER_GAUSSIAN_AT_REFERENCE * multiplier
