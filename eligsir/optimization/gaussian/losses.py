"""Streaming loss/regularizer terms for Gaussian optimization.

Wave 1 consolidation plan step 7 (see
the Gaussian optimization ownership boundary.

Extracted verbatim from ``train_streaming.py`` (formerly lines 249-384) —
no math change. These were already pure functions (take ``gaussians``/
``args``/scalars, return a scalar loss or a grad-snapshot dict, no module-
level state), so moving them here is a pure relocation, not a rewrite.
``train_streaming.py`` now imports them from this module.
"""

from __future__ import annotations

import torch

from eligsir.utils.compiled_kernels import young_isotropy_ramp  # noqa: E402  (re-exported for callers/tests)


def _streaming_young_isotropy_reg(gaussians, args, n_frames_ingested):
    """Decaying isotropy regularizer on provisional+young Gaussians.

    Mirrors ``_streaming_young_opacity_reg``'s guard shape: falls back to a
    no-op (zero-valued, no tensor ops beyond the guard checks) when the
    weight is 0 or the required lifecycle tensors are unavailable, so the
    off-path (default) is byte-identical to before this feature existed.
    """
    weight = float(getattr(args, "streaming_young_isotropy_weight", 0.0))
    if weight == 0.0:
        return None
    lifecycle_state = getattr(gaussians, "lifecycle_state", None)
    birth_frame = getattr(gaussians, "birth_frame", None)
    if lifecycle_state is None or birth_frame is None:
        return None
    n = gaussians.get_xyz.shape[0]
    if lifecycle_state.shape[0] != n or birth_frame.shape[0] != n:
        return None

    immature_mask = lifecycle_state < 2
    if not immature_mask.any():
        # No immature rows right now (e.g. every Gaussian has already
        # matured): an exact finite zero, not None -- the specific contract
        # tests/test_loss_regularizer_zero_safe.py pins. The other early-outs
        # below (ramp not yet active, scaling has <2 dims) are the same
        # "applicable but nothing to act on" shape and could reasonably
        # return zero too, but only this one is under test; left as None
        # rather than widening the fix beyond what's verified.
        return torch.zeros((), dtype=gaussians.get_xyz.dtype, device=gaussians.get_xyz.device)

    ramp_frames = float(getattr(args, "streaming_young_isotropy_ramp_frames", 15))
    age = (n_frames_ingested - birth_frame[immature_mask].long()).clamp(min=0)
    ramp = young_isotropy_ramp(age, ramp_frames)
    # Device-side gating: compute unconditionally; ramp-all-zero gives exactly 0.0 (no host sync).

    s = gaussians.get_scaling[immature_mask]
    if s.shape[-1] < 2:
        return None
    s_max = s.max(dim=-1).values
    s_min = s.min(dim=-1).values.clamp(min=1e-6)
    aniso_term = (s_max / s_min - 1.0).pow(2)
    return weight * (ramp * aniso_term).mean()


def _streaming_young_opacity_reg(
    gaussians, args, optimizer_update_id, active_mask, young_exempt_iters: int, maturity_ramp: bool,
):
    """Opacity-only regularization term with young-exemption + maturity ramp
    + recency-of-supervision ramp.

    Mirrors ``active_reg_core``'s opacity term (``w_opacity * |opacity|.mean()``)
    but multiplies each Gaussian's contribution by a per-Gaussian weight in
    [0, 1] (see ``young_opacity_reg_weights``). ``active_mask`` selects the
    active subset (sparse path) or is None (global path). Returns a scalar loss.

    Falls back to the standard full-strength opacity reg when the per-Gaussian
    state needed for the requested knob is unavailable (e.g. birth_iter missing),
    so it can never crash the live loop.
    """
    opacity = gaussians.get_opacity
    if active_mask is not None:
        opacity = opacity[active_mask]
    opacity_abs = torch.abs(opacity.squeeze(-1))
    if opacity_abs.numel() == 0:
        # Empty active subset (e.g. every candidate filtered out this step):
        # an exact finite zero. `.mean()` over an empty tensor is NaN, not 0.
        return torch.zeros((), dtype=opacity_abs.dtype, device=opacity_abs.device)

    _birth = getattr(gaussians, "birth_iter", None)
    _support = getattr(gaussians, "support_count", None) if maturity_ramp else None
    _n = gaussians.get_opacity.shape[0]

    _recency_horizon = float(getattr(args, "streaming_recency_reg_horizon", 0))
    _last_supervised = getattr(gaussians, "last_supervised_iter", None) if _recency_horizon > 0 else None
    if _last_supervised is not None and _last_supervised.shape[0] != _n:
        _last_supervised = None
        _recency_horizon = 0.0

    age = None
    if young_exempt_iters > 0 and _birth is not None and _birth.shape[0] == _n:
        age = (optimizer_update_id - _birth.to(dtype=torch.int64))
        if active_mask is not None:
            age = age[active_mask]
    if maturity_ramp and (_support is None or _support.shape[0] != _n):
        # Ramp requested but no support state: fall back to full opacity reg.
        maturity_ramp = False
    support = None
    if maturity_ramp and _support is not None:
        support = _support
        if active_mask is not None:
            support = support[active_mask]

    recency_gap = None
    if _recency_horizon > 0 and _last_supervised is not None:
        recency_gap = (
            optimizer_update_id - _last_supervised.to(dtype=torch.int64)
        )
        if active_mask is not None:
            recency_gap = recency_gap[active_mask]

    if age is None and support is None and recency_gap is None:
        # Nothing applicable — emit standard full-strength opacity reg.
        return float(args.opacity_reg) * opacity_abs.mean()

    # If young exemption is off, age tensor is absent but the ramp still needs a
    # per-Gaussian shape; synthesize a zero-age placeholder (>= any threshold,
    # so the disabled exemption is a no-op).
    if age is None:
        age = torch.zeros(opacity_abs.shape[0], dtype=torch.int64, device=opacity_abs.device)

    from eligsir.utils.compiled_kernels import young_opacity_reg_weights
    w = young_opacity_reg_weights(
        age=age,
        support_count=support,
        young_exempt_iters=young_exempt_iters,
        maturity_ramp=maturity_ramp,
        target_support=float(getattr(args, "streaming_debt_replay_target_support", 10)),
        ramp_min=float(getattr(args, "streaming_maturity_reg_ramp_min", 0.2)),
        recency_gap=recency_gap,
        recency_horizon=_recency_horizon,
        recency_ramp_min=float(getattr(args, "streaming_recency_reg_ramp_min", 0.2)),
    )
    return float(args.opacity_reg) * (opacity_abs * w).mean()


def _streaming_young_opacity_reg_weighted(
    gaussians, args, optimizer_update_id, weights,
    young_exempt_iters: int, maturity_ramp: bool, *, batch_size, opacity=None,
):
    """Return the weighted young-opacity regularizer with global divisor B."""
    opacity_full = gaussians.get_opacity if opacity is None else opacity
    opacity_abs = torch.abs(opacity_full.squeeze(-1))

    _birth = getattr(gaussians, "birth_iter", None)
    _support = getattr(gaussians, "support_count", None) if maturity_ramp else None
    _n = opacity_full.shape[0]

    _recency_horizon = float(getattr(args, "streaming_recency_reg_horizon", 0))
    _last_supervised = getattr(gaussians, "last_supervised_iter", None) if _recency_horizon > 0 else None
    if _last_supervised is not None and _last_supervised.shape[0] != _n:
        _last_supervised = None
        _recency_horizon = 0.0

    age = None
    if young_exempt_iters > 0 and _birth is not None and _birth.shape[0] == _n:
        age = (optimizer_update_id - _birth.to(dtype=torch.int64))
    if maturity_ramp and (_support is None or _support.shape[0] != _n):
        # Ramp requested but no support state: fall back to full opacity reg.
        maturity_ramp = False
    support = _support if maturity_ramp else None

    recency_gap = None
    if _recency_horizon > 0 and _last_supervised is not None:
        recency_gap = optimizer_update_id - _last_supervised.to(dtype=torch.int64)

    # Keep the global requested batch as the divisor: a group can omit views.
    denom = float(batch_size)

    if age is None and support is None and recency_gap is None:
        # Nothing applicable -- weighted generalisation of
        # `_streaming_young_opacity_reg`'s own "nothing applicable" return
        # (`opacity_reg * opacity_abs.mean()`, i.e. w == 1 everywhere).
        return float(args.opacity_reg) * (opacity_abs * weights).sum() / denom

    if age is None:
        age = torch.zeros(opacity_abs.shape[0], dtype=torch.int64, device=opacity_abs.device)

    from eligsir.utils.compiled_kernels import young_opacity_reg_weights
    w = young_opacity_reg_weights(
        age=age,
        support_count=support,
        young_exempt_iters=young_exempt_iters,
        maturity_ramp=maturity_ramp,
        target_support=float(getattr(args, "streaming_debt_replay_target_support", 10)),
        ramp_min=float(getattr(args, "streaming_maturity_reg_ramp_min", 0.2)),
        recency_gap=recency_gap,
        recency_horizon=_recency_horizon,
        recency_ramp_min=float(getattr(args, "streaming_recency_reg_ramp_min", 0.2)),
    )
    return float(args.opacity_reg) * (opacity_abs * w * weights).sum() / denom


def assemble_step_regularizers(gaussians, args, optimizer_update_id, weights, *, batch_size):
    """Evaluate view-independent regularizers once using raw view weights.

    ``weights[i] = sum_v mask_v[i] / n_v`` and ``batch_size`` is the global
    requested view count, including skipped or empty views.
    """
    from eligsir.utils.compiled_kernels import active_reg_core_masked

    opacity = gaussians.get_opacity
    scaling = gaussians.get_scaling
    divisor = float(batch_size)

    count = weights.sum().clamp(min=1)
    reg = active_reg_core_masked(
        opacity,
        scaling,
        weights,
        w_opacity=0.0,
        w_scale=float(args.scale_reg),
        w_aniso=float(args.anisotropy_reg),
    )
    # Cancel `active_reg_core_masked`'s internal `count` divisor (correct
    # for a single view's own mask, wrong for our multi-view `weights`) and
    # apply the caller's true global `batch_size` instead.
    reg = reg * (count / divisor)
    reg = reg + _streaming_young_opacity_reg_weighted(
        gaussians,
        args,
        optimizer_update_id,
        weights,
        int(args.streaming_young_reg_exempt_iters),
        bool(args.streaming_maturity_reg_ramp),
        batch_size=batch_size,
        opacity=opacity,
    )
    return reg


def _snapshot_energy_grads(gaussians):
    """Snapshot per-Gaussian leaf-param grads before optimizer.zero_grad().

    The streaming loops call ``optimizer.zero_grad(set_to_none=True)`` before
    ``compute_gaussian_utility``; without this snapshot every ``.grad`` read in
    the utility is ``None`` and the energy gradient term silently vanishes. We
    hold references to the existing grad tensors (no copy) keyed by gsplat
    param name so they survive the ``set_to_none`` wipe.
    """
    params = getattr(gaussians, "params", None)
    snap = {}
    if params is not None:
        for key in ("means", "scales", "opacities"):
            p = params.get(key) if hasattr(params, "get") else None
            if p is not None and p.grad is not None:
                snap[key] = p.grad
        return snap or None
    # Legacy layout: map gsplat keys to legacy attribute grads.
    for key, attr in (("means", "_xyz"), ("scales", "_scaling"), ("opacities", "_opacity")):
        t = getattr(gaussians, attr, None)
        if t is not None and getattr(t, "grad", None) is not None:
            snap[key] = t.grad
    return snap or None
