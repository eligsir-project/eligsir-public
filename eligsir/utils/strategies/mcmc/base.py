"""Shared helper functions used by every MCMC strategy variant.

Moved verbatim out of ``utils/strategies/mcmc_strategy.py`` (Wave 2 Task 4,
2026-08-05) — no behavior change, only module location.
"""

import torch

from eligsir.utils.streaming_depth_lock import depth_lock_enabled, depth_locked_mask


def _effective_cap_max(args):
    """Growth ceiling for MCMC add-Gaussian steps. Normally ``args.cap_max`` (which
    ALSO drives the schedule thresholds — noise_lr, growth-stop, quantile prune).
    When the adaptive-capacity controller is on it publishes a dynamic
    ``_adaptive_growth_cap`` that ONLY throttles growth (decoupled from the schedule
    cap_max, per the cap_max-coupling constraint) — the effective cap is the min of
    the two. Default-off ⇒ returns raw cap_max (byte-identical)."""
    _cap = int(args.cap_max)
    if getattr(args, "streaming_adaptive_capacity", False):
        _dyn = getattr(args, "_adaptive_growth_cap", None)
        if _dyn is not None:
            return int(min(_cap, int(_dyn)))
    return _cap


def _locked_depth_mask(gaussians, args):
    if not depth_lock_enabled(args):
        return None
    mask = depth_locked_mask(gaussians, args)
    if mask.shape[0] != gaussians.get_xyz.shape[0] or not mask.any():
        return None
    return mask


def _exclude_locked_from_mask(mask, gaussians, args):
    locked = _locked_depth_mask(gaussians, args)
    if locked is None or mask is None or mask.shape[0] != locked.shape[0]:
        return mask
    return mask & ~locked


#: Minimum number of in-bubble Gaussians for the mask to be usable. A near-empty
#: bubble (e.g. stale center after a fast turn) must never zero the map-building
#: machinery — below this count the mask degrades to None (global behavior).
#: Only enforced when the total population itself exceeds the threshold, so tiny
#: bootstrap maps (and unit tests) keep exact mask semantics.
_BUBBLE_MIN_ACTIVE = 500
_bubble_guard_warned = False


def _bubble_feature_enabled(args):
    """The bubble MASK is needed by two independent features: (a) the spatial
    freeze (--streaming_spatial_bubble_enabled) and (b) local statistics /
    local-bounds MCMC (--streaming_local_stats). Either flag alone must produce
    a mask; the FREEZE machinery (grad zeroing, snapshot-restore, relocation
    dead gate, noise restore) additionally checks streaming_spatial_bubble_enabled
    explicitly so stats-only runs never freeze anything."""
    return bool(
        getattr(args, "streaming_spatial_bubble_enabled", False)
        or getattr(args, "streaming_local_stats", False)
    )


def _bubble_active_mask(means, args):
    """Component 1 (spatial active bubble). Returns a bool mask [N] where True =
    the Gaussian is INSIDE the active bubble (within ``streaming_bubble_radius`` of
    the robot's current pose ``args._bubble_center``). Out-of-bubble geometry is
    frozen this temporal window (not relocated / not noised / no gradient) so it
    cannot be forgotten while the robot works elsewhere; it reactivates when the
    robot returns because the mask is recomputed from CURRENT positions every call
    (no stored per-Gaussian state → no grow/prune reindexing needed).

    Returns ``None`` (no gating) when the bubble is disabled, the center is unset,
    or the radius is non-positive. ``means`` is any [N,3] position tensor.
    """
    if not _bubble_feature_enabled(args):
        return None
    center = getattr(args, "_bubble_center", None)
    radius = float(args.streaming_bubble_radius or 0.0)
    # Guard band: geometry within radius is the active core; radius..radius+guard is a
    # trainable seam that softens the frozen boundary (avoids alpha-composite seam
    # artifacts). Everything beyond radius+guard is frozen. For Step 1 both core and
    # seam are fully trainable, so the freeze boundary is (radius + guard).
    guard = float(getattr(args, "streaming_bubble_guard_m", 0.0) or 0.0)
    if center is None or radius <= 0.0 or means is None or means.shape[0] == 0:
        return None
    with torch.no_grad():
        d = torch.linalg.norm(means.detach() - center.to(means.device), dim=1)
        mask = d < (radius + max(0.0, guard))
        # Empty-bubble guard: a near-empty bubble must never zero the
        # map-building machinery — fall back to global behavior.
        if means.shape[0] > _BUBBLE_MIN_ACTIVE and int(mask.sum()) < _BUBBLE_MIN_ACTIVE:
            global _bubble_guard_warned
            if not _bubble_guard_warned:
                _bubble_guard_warned = True
                print(
                    f"[bubble] guard tripped: only {int(mask.sum())} of {means.shape[0]} "
                    f"Gaussians in bubble (<{_BUBBLE_MIN_ACTIVE}) — mask disabled (global behavior)"
                )
            return None
        return mask


def _gate_dead_mask_to_visible(mask, render_pkg, args):
    """A1 (--streaming_reloc_local_only): restrict relocation to the VISIBLE set.

    MCMC relocation teleports 'dead' (low-opacity) Gaussians toward high-utility
    parents. On a TRAVELING trajectory the out-of-view bootstrap seed gets zero
    supervision, its opacity drifts down, it is flagged dead, and gets relocated
    AWAY — the mid-trajectory 'scene implosion'. Intersecting dead_mask with the
    current frame's visibility_filter means out-of-view Gaussians are never
    relocated (their geometry is preserved until revisit / consolidation). No-op
    when the flag is off, when render_pkg has no visibility_filter, or on shape
    mismatch (defensive).
    """
    if not getattr(args, "streaming_reloc_local_only", False):
        return mask
    if render_pkg is None:
        return mask
    vis = render_pkg.get("visibility_filter") if hasattr(render_pkg, "get") else None
    if vis is None or vis.shape != mask.shape:
        return mask
    return mask & vis.to(mask.device)


def _augment_dead_mask_knn_isolation(mask, gaussians, args, current_iter=None):
    """Additive KNN-density floater kill (Phase A, flag-gated OFF by default).

    A large Gaussian that is spatially ISOLATED — its self-KNN nearest-neighbour
    distance far exceeds the scene-typical spacing — is almost always a floater
    (white-streak / blue-dot) that the visibility-EMA clauses miss until the EMA
    decays. Uses the fast simple_knn distCUDA2 (self-KNN) over live means; gated by
    a large-scale + low-support test so well-supported sparse detail survives, and
    by birth-iter young protection so freshly-inserted points aren't culled.

    Enabled only when --streaming_knn_isolation_kill_mult > 0; returns `mask`
    unchanged otherwise. Applied OUTSIDE compute_dead_mask, so it never affects the
    parity-tested legacy/energy dead-mask paths.
    """
    mult = float(getattr(args, "streaming_knn_isolation_kill_mult", 0.0))
    if mult <= 0 or mask is None:
        return mask
    try:
        import torch
        from eligsir.utils.rocm_knn_fallback import distCUDA2
        means = gaussians.get_xyz.detach()
        if means.shape[0] < 16 or means.shape[0] != mask.shape[0]:
            return mask
        knn_d = distCUDA2(means.to(torch.float32)).clamp_min(0).sqrt()
        med = torch.median(knn_d)
        isolated = knn_d > (mult * med)
        max_scale = gaussians.get_scaling.max(dim=-1).values
        scale_gate = float(getattr(args, "scale_moderate_threshold", 0.05))
        if hasattr(gaussians, "visibility_ema") and gaussians.visibility_ema.shape[0] == max_scale.shape[0]:
            support = gaussians.visibility_ema.squeeze(-1)
        else:
            support = torch.ones_like(max_scale)
        # Note the deliberate asymmetry: the scale gate above reuses the shared
        # dead-mask knob `scale_moderate_threshold`, while the support gate is
        # KNN-isolation-private. Until 2026-08-13 this line read a non-existent
        # `args.support_threshold` (name drift from GaussianEnergyConfig's field
        # of that name), so it was permanently pinned to the 0.01 literal. The
        # dedicated flag defaults to the same 0.01 — see issue #382.
        supp_thresh = float(getattr(args, "streaming_knn_isolation_support_threshold", 0.01))
        kill = isolated & (max_scale > scale_gate) & (support < supp_thresh)
        # Young protection: don't cull points still building opacity.
        young_prot = int(args.streaming_young_prot_iters)
        if young_prot > 0 and current_iter is not None:
            birth_iter = getattr(gaussians, "birth_iter", None)
            if birth_iter is not None and birth_iter.shape[0] == kill.shape[0]:
                young = (int(current_iter) - birth_iter.to(dtype=torch.int64)) < young_prot
                from eligsir.utils.energy_mcmc import _or_exposure_young
                young = _or_exposure_young(gaussians, young)
                kill = kill & ~young
            elif birth_iter is not None:
                from eligsir.utils.gaussian_lineage import warn_young_prot_mismatch_once
                warn_young_prot_mismatch_once(
                    gaussians,
                    site="base._augment_dead_mask_knn_isolation",
                    warn_attr="_young_prot_mismatch_warned_knn_isolation",
                    birth_len=int(birth_iter.shape[0]),
                    mask_len=int(kill.shape[0]),
                )
        return mask | kill
    except Exception:
        return mask
