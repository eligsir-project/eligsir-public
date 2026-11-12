#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

"""
Energy-Guided Adaptive MCMC for 3DGS.

This module provides differentiable energy losses and per-Gaussian utility
scores to guide MCMC relocation and growth decisions.

Backwards compatibility: all functions here are called conditionally from
train.py when --energy_mcmc is enabled (default). The old fixed-interval
path remains available via --no-energy-mcmc.
"""
import os
import torch
import math
from dataclasses import dataclass, field

_PRUNING_STAT_KEYS = (
    "dead_opacity",
    "low_support",
    "scale_moderate",
    "anisotropy_kill",
    "utility_quantile",
    "young_protected_skip",
    "total_pruned",
)


@dataclass
class MCMCRunState:
    """Explicit, instantiable pruning-stats accumulator for one MCMC run.

    Replaces the historical module-level ``_pruning_stats``/
    ``_pruning_stats_cumulative`` globals (Wave 1 side-track, see
    ``/path/to/design-notes.md`` §7):
    those globals made MCMC pruning-stat scoring non-reentrant across
    multiple runs/models in the same process (e.g. two concurrent streaming
    scenes, or a test suite creating several models). An ``MCMCRunState``
    instance is owned per run/model instead.

    ``windowed`` mirrors the old ``_pruning_stats`` (reset on every
    ``pop()``); ``cumulative`` mirrors ``_pruning_stats_cumulative`` (never
    reset by ``pop()``/``reset()``).
    """

    windowed: dict = field(default_factory=lambda: {k: 0 for k in _PRUNING_STAT_KEYS})
    cumulative: dict = field(default_factory=lambda: {k: 0 for k in _PRUNING_STAT_KEYS})

    def bump(self, key: str, n: int) -> None:
        self.windowed[key] += n
        self.cumulative[key] += n

    def reset(self) -> None:
        for k in self.windowed:
            self.windowed[k] = 0

    def pop(self) -> dict:
        out = dict(self.windowed)
        self.reset()
        return out

    def get_cumulative(self) -> dict:
        return dict(self.cumulative)


# Process-wide default instance, kept for backward compatibility with the
# module-level function API below (existing callers in
# utils/gaussian_energy_policy.py, utils/comparison_report.py, and
# tests/test_pruning_stats_cumulative.py). New call sites that own a
# distinct run/model should construct their own MCMCRunState and pass it to
# compute_dead_mask(run_state=...) instead of relying on this shared default.
_default_run_state = MCMCRunState()

# Back-compat aliases: some callers (utils/gaussian_energy_policy.py) mutate
# this dict directly rather than going through bump()/the function API.
_pruning_stats = _default_run_state.windowed
_pruning_stats_cumulative = _default_run_state.cumulative


def reset_pruning_stats() -> None:
    _default_run_state.reset()


def pop_pruning_stats() -> dict:
    """Return a copy of the since-last-call pruning stats and reset counters.

    Windowed, not cumulative: callers wanting a whole-run total should use
    get_cumulative_pruning_stats() instead.
    """
    return _default_run_state.pop()


def get_cumulative_pruning_stats() -> dict:
    """Return a copy of the whole-run pruning stats (never reset)."""
    return _default_run_state.get_cumulative()


def _bump_pruning_stat(key: str, n: int, run_state: "MCMCRunState | None" = None) -> None:
    """Increment both the windowed (resettable) and cumulative pruning counters."""
    (run_state or _default_run_state).bump(key, n)


def _dense_grad(grad):
    if grad is None:
        return None
    if getattr(grad, "layout", torch.strided) != torch.strided:
        return grad.to_dense()
    return grad


def _named_grad(gaussians, legacy_attr, gsplat_key=None, overrides=None):
    # Captured-grad override: the streaming loop snapshots leaf-param grads
    # before optimizer.zero_grad(set_to_none=True) wipes them, so utility can be
    # computed after the optimizer step without losing the gradient term.
    if overrides is not None and gsplat_key is not None and gsplat_key in overrides:
        grad = _dense_grad(overrides[gsplat_key])
        if grad is not None:
            return grad
    params = getattr(gaussians, "params", None)
    if params is not None and gsplat_key is not None and gsplat_key in params:
        grad = _dense_grad(params[gsplat_key].grad)
        if grad is not None:
            return grad
    # Legacy-layout fallback: only leaf tensors carry a populated .grad. Accessing
    # .grad on a non-leaf (e.g. an activation-transformed view) just emits a noisy
    # UserWarning and returns None anyway, so guard on is_leaf.
    _legacy = getattr(gaussians, legacy_attr, None)
    if _legacy is None or not getattr(_legacy, "is_leaf", True):
        return None
    return _dense_grad(_legacy.grad)


def compute_effective_count(opacities, dead_threshold=0.005, softness=0.002):
    """
    Compute soft effective Gaussian count.

    N_eff = sum_j sigmoid((alpha_j - alpha_dead) / s)

    Args:
        opacities: [N, 1] raw opacity parameters (gaussians._opacity, before sigmoid)
                   because this function participates in the loss graph and
                   must be differentiable w.r.t. raw parameters.
        dead_threshold: opacity threshold
        softness: softness of the sigmoid
    Returns:
        N_eff scalar tensor
    """
    from eligsir.utils.compiled_kernels import effective_count_core
    return effective_count_core(opacities, dead_threshold, softness)


def compute_effective_count_loss(
    opacities, iteration, cap_max,
    dead_threshold=0.005, softness=0.002,
    q_start=0.05, q_end=0.85, tau_N=0.45,
    max_iterations=30000, target_splat_end=None,
    target_capacity=0,
):
    """
    Loss that steers the effective Gaussian count toward a target curve.

    L_eff = ((N_eff - N_target) / cap_max)^2

    Args:
        opacities: [N, 1] raw opacity parameters (gaussians._opacity, before sigmoid)
                   because this function participates in the loss graph.
        iteration: current iteration
        cap_max: maximum allowed Gaussians and fallback target scale
        q_start: initial target cap fraction
        q_end: final target cap fraction
        tau_N: exponential interpolation tau for target curve
        max_iterations: training horizon used to normalize the target curve
        target_splat_end: optional soft target count independent of cap_max
        target_capacity: Merge-A soft target-N (CapacityPolicy). When > 0 it is the
                   highest-priority target scale for the count-loss N_target curve,
                   overriding both target_splat_end and cap_max. Default 0 is
                   behaviour-neutral (falls through to the prior selection).
    Returns:
        scalar loss tensor
    """
    from eligsir.utils.compiled_kernels import effective_count_loss_core

    if cap_max <= 0:
        zero = torch.tensor(0.0, device=opacities.device)
        return zero, zero.detach(), 0.0

    # Merge A (CapacityPolicy): target_capacity (soft TARGET-N) takes precedence
    # over target_splat_end, which in turn falls back to cap_max (the hard ceiling).
    if target_capacity is not None and target_capacity > 0:
        target_scale = target_capacity
    else:
        target_scale = target_splat_end if target_splat_end is not None and target_splat_end > 0 else cap_max
    if target_scale <= 0:
        zero = torch.tensor(0.0, device=opacities.device)
        return zero, zero.detach(), 0.0

    u = min(iteration / float(max(1, max_iterations)), 1.0)

    # Exponential interpolation for q
    tau = max(tau_N, 1e-6)
    denom = 1.0 - math.exp(-1.0 / tau)
    alpha_q = (1.0 - math.exp(-u / tau)) / denom
    q = q_start + (q_end - q_start) * alpha_q

    N_target = target_scale * q
    loss, N_eff = effective_count_loss_core(opacities, N_target, target_scale, dead_threshold, softness)
    return loss, N_eff.detach(), N_target


def compute_opacity_entropy_loss(opacities):
    """
    Opacity entropy loss: pushes opacities toward 0 or 1 (decisive).

    L_entropy = mean(-a log(a) - (1-a) log(1-a))

    Minimizing entropy makes MCMC dead/alive decisions cleaner.

    Args:
        opacities: [N, 1] raw opacity parameters (gaussians._opacity, before sigmoid)
                   because this function participates in the loss graph.
    """
    from eligsir.utils.compiled_kernels import opacity_entropy_core
    return opacity_entropy_core(opacities)


def compute_gaussian_utility(
    gaussians, render_pkg, iteration,
    w_alpha=1.0, w_vis=2.0, w_grad=3.0,
    w_scale=0.5, w_dead=1.0,
    w_support=2.0,
    beta_opacity=1.0, beta_scale=0.5,
    alpha_dead=0.005,
    w_aniso_utility=0.1,
    aniso_soft_limit=10.0,
    args=None,
    grad_overrides=None,
    w_depth_residual=0.0,
    depth_residual_scalar=0.0,
):
    """
    Compute per-Gaussian utility score for MCMC birth/death decisions.

    U_j = w_alpha * alpha_j
        + w_vis * v_j (single-frame visibility)
        + w_support * support_j (multi-view EMA visibility)
        + w_grad * norm_grad_j * (1 + w_depth_residual * depth_residual_scalar)
        - w_scale * scale_penalty_j
        - w_dead * dead_penalty_j

    Args:
        gaussians: GaussianModel instance
        render_pkg: dict from render() containing visibility info
        iteration: current iteration
        various weight coefficients
        args: optional args namespace; when args.energy_mode == 'population_energy',
              routes through GaussianEnergyPolicy.compute_population_energy and returns
              -energy.total (negated so high utility = low energy, matching legacy polarity).
        w_depth_residual: weight for the depth-error-gated temperature multiplier
              on the gradient utility term (see ``depth_residual_scalar``). 0.0 = off
              (no-op; matches prior behaviour exactly).
        depth_residual_scalar: a SIMPLIFIED per-frame scalar approximation of a true
              per-Gaussian depth-residual signal — the mean |sensor_depth -
              rendered_depth| residual over valid pixels for the CURRENT frame,
              reused as-is for every growth-candidate Gaussian in that frame
              (avoids introducing a new per-Gaussian EMA buffer; keeps this diff
              surgical). Higher residual => temporarily boosts the weight of the
              position-gradient term, biasing growth-parent selection toward
              regions this frame's Gaussians whose gradient utility is already
              elevated, in frames where depth supervision disagrees with the
              render the most.
    Returns:
        [N] utility scores tensor
    """
    # Phase 3: population energy routing (gated by energy_mode flag)
    if getattr(args, "energy_mode", "legacy_utility") == "population_energy":
        from eligsir.utils.gaussian_energy_policy import GaussianEnergyPolicy, energy_config_from_args
        # issue #385: the args->config mapping now lives in exactly one place
        # (energy_config_from_args). energy_mode is forced because this branch is
        # already gated on args.energy_mode == "population_energy". alpha_dead
        # comes from args here (not a schedule) -- and with no getattr fallback,
        # since energy_alpha_dead is an unconditional OptimizationParams default
        # always present on the real parsed args (issue #368: the old
        # getattr(..., 0.005) fallback had drifted from the real default of 0.05,
        # the same stale value that caused a confirmed live bug once before the
        # parser default was raised).
        _pop_cfg = energy_config_from_args(
            args,
            alpha_dead=float(args.energy_alpha_dead),
            energy_mode="population_energy",
        )
        _pop_policy = GaussianEnergyPolicy(_pop_cfg)
        _pop_terms = _pop_policy.compute_terms(gaussians, render_pkg, iteration)
        _energy_result = _pop_policy.compute_population_energy(_pop_terms)
        # Negate so polarity matches legacy: high utility = good Gaussian = low energy
        return -_energy_result.total

    from eligsir.utils.compiled_kernels import utility_core

    device = gaussians.get_xyz.device
    N = gaussians.get_xyz.shape[0]

    # Opacity term: get_opacity is already activated (sigmoid applied)
    alpha = gaussians.get_opacity.squeeze(-1)

    # Visibility term (current frame)
    visibility = render_pkg.get("visibility_filter", torch.ones(N, device=device, dtype=torch.bool))
    v = visibility.float()

    # Support EMA term (multi-view visibility, not just current frame).
    # Guard against a stale buffer left over from before an external growth
    # event (e.g. GsplatDefaultStrategy's clone/split, which doesn't extend
    # this MCMC-strategy-specific EMA) -- see the identical guard added to
    # utils/strategies/mcmc/energy_policy.py and base.py for the same bug.
    support = (
        gaussians.visibility_ema.squeeze(-1)
        if hasattr(gaussians, "visibility_ema") and gaussians.visibility_ema.shape[0] == N
        else v
    )

    # Gradient term: norm of xyz gradient
    # NOTE: requires that loss.backward() has been called
    xyz_grad = torch.zeros(N, device=device)
    grad = _named_grad(gaussians, "_xyz", "means", overrides=grad_overrides)
    if grad is not None:
        xyz_grad = grad.norm(dim=1)

    opacity_grad = torch.zeros(N, device=device)
    grad = _named_grad(gaussians, "_opacity", "opacities", overrides=grad_overrides)
    if grad is not None:
        opacity_grad = grad.abs().flatten()

    scale_grad = torch.zeros(N, device=device)
    grad = _named_grad(gaussians, "_scaling", "scales", overrides=grad_overrides)
    if grad is not None:
        scale_grad = grad.norm(dim=1)

    # Diagnostic: confirm the energy gradient term is actually populated (not all
    # zero from grads being wiped before this call). Enable with
    # ELIGSIR_DEBUG_ENERGY_GRAD=1; logs on the energy-mcmc 500-iter cadence.
    if os.environ.get("ELIGSIR_DEBUG_ENERGY_GRAD") == "1" and int(iteration) % 500 == 0:
        print(
            f"[energy-grad] iter={iteration} overrides={'yes' if grad_overrides else 'no'} "
            f"xyz_grad.sum={float(xyz_grad.sum()):.4g} "
            f"opacity_grad.sum={float(opacity_grad.sum()):.4g} "
            f"scale_grad.sum={float(scale_grad.sum()):.4g} "
            f"nonzero_xyz={int((xyz_grad > 0).sum())}/{N}",
            flush=True,
        )

    # Depth-error-gated temperature multiplier on the gradient term (simplified,
    # per-frame-scalar approximation — see docstring). 0.0 weight is a true no-op:
    # xyz_grad is left untouched, byte-identical to before this feature existed.
    if w_depth_residual != 0.0:
        xyz_grad = xyz_grad * (1.0 + w_depth_residual * depth_residual_scalar)

    # Scale penalty: penalize oversized Gaussians
    s = gaussians.get_scaling
    max_scale = s.max(dim=1).values

    scores = utility_core(
        alpha, v, support, xyz_grad, opacity_grad, scale_grad, max_scale,
        w_alpha=w_alpha, w_vis=w_vis, w_support=w_support, w_grad=w_grad,
        w_scale=w_scale, w_dead=w_dead, beta_opacity=beta_opacity,
        beta_scale=beta_scale, alpha_dead=alpha_dead,
    )

    # Anisotropy penalty: discourage elongated splats from winning parent-selection lottery
    min_scale = None
    if w_aniso_utility > 0 and aniso_soft_limit > 0:
        min_scale = s.min(dim=1).values.clamp_min(1e-6)
        aniso_ratio = max_scale / min_scale
        aniso_penalty = torch.clamp(aniso_ratio - aniso_soft_limit, min=0.0) ** 2
        scores = scores - w_aniso_utility * aniso_penalty

    # issue #1259: root-cause diagnostics for the "lifecycle utility must be a
    # finite per-Gaussian tensor" crash (production_application.py's
    # _advance_lifecycle). This block is read-only (never changes `scores`) --
    # it exists purely so the NEXT occurrence self-diagnoses which upstream
    # input actually went non-finite (a squared scale-penalty overflow, an
    # aniso-ratio overflow with a near-zero min_scale, a NaN gradient, etc.)
    # instead of the caller only learning "utility has a NaN somewhere".
    _nonfinite = ~torch.isfinite(scores)
    if bool(_nonfinite.any()):
        _n_bad = int(_nonfinite.sum())
        _report = getattr(compute_gaussian_utility, "_nonfinite_report_count", 0) + 1
        compute_gaussian_utility._nonfinite_report_count = _report
        if _report <= 12:
            with torch.no_grad():
                _diag = {
                    "alpha": alpha,
                    "v": v,
                    "support": support,
                    "xyz_grad": xyz_grad,
                    "opacity_grad": opacity_grad,
                    "scale_grad": scale_grad,
                    "max_scale": max_scale,
                }
                if min_scale is not None:
                    _diag["min_scale"] = min_scale
                _culprits = {
                    _name: int((~torch.isfinite(_t)).sum())
                    for _name, _t in _diag.items()
                    if not bool(torch.isfinite(_t).all())
                }
                _max_scale_bad = float(max_scale[_nonfinite].max()) if _n_bad else float("nan")
            _culprits_desc = _culprits if _culprits else (
                "NONE (overflow happened inside utility_core/aniso arithmetic "
                "itself, e.g. squared scale_penalty or aniso_ratio blowing up to inf)"
            )
            print(
                f"[energy-mcmc] iter={iteration} WARNING: {_n_bad}/{scores.numel()} "
                f"utility scores non-finite (occurrence {_report}). Non-finite upstream "
                f"inputs: {_culprits_desc}. max_scale among bad rows={_max_scale_bad:.4g}",
                flush=True,
            )

    return scores


def compute_dead_mask(
    gaussians, utility=None,
    opacity_threshold=0.005,
    support_threshold=0.01,
    utility_quantile=0.05,
    use_utility_quantile=True,
    min_visibility_count=3,
    scale_kill_threshold=0.20,
    scale_moderate_threshold=0.05,
    scale_moderate_opacity_mult=4.0,
    scale_dead_support_mult=1.0,
    anisotropy_kill_ratio=20.0,
    current_iter: int | None = None,
    young_prot_iters: int = 0,
    run_state: "MCMCRunState | None" = None,
):
    """
    Compute dead mask combining opacity, support, scale, anisotropy, and optionally utility.

    Composite rule:
        1. Unconditional kill: alpha < opacity_threshold * 0.1  (near-zero, no recovery)
        2. Conservative kill:  alpha < opacity_threshold AND support < support_threshold
        3. Scale kill:         max_scale > scale_kill_threshold AND support low
           (large Gaussians must be well-supported or they are runaway floaters)
        4. Medium-scale kill:  max_scale > scale_moderate_threshold
                               AND alpha < opacity_threshold * scale_moderate_opacity_mult
                               AND support low
           (semi-transparent medium-size splats that are unsupported = blue-dot floaters)
        5. Anisotropy kill:    s_max/s_min > anisotropy_kill_ratio
                               AND alpha < opacity_threshold * scale_moderate_opacity_mult
                               AND support low
           (needle Gaussians with high axis ratio = white-streak floaters below 0.2m threshold)
        6. Optional utility bottom-quantile kill (opacity-gated)

    Args:
        gaussians: GaussianModel instance
        utility: [N] optional utility scores
        opacity_threshold: opacity dead threshold
        support_threshold: support EMA dead threshold
        utility_quantile: bottom quantile for utility-based death
        use_utility_quantile: whether to append utility quantile death
        min_visibility_count: unused (kept for API compatibility)
        scale_kill_threshold: max_scale above which low support = dead (default 0.2 m)
        scale_moderate_threshold: max_scale above which relaxed opacity gate applies (default 0.05 m)
        scale_moderate_opacity_mult: opacity multiplier for clauses 4+5 (default 4.0)
        scale_dead_support_mult: multiplier on support_threshold for scale/aniso clauses (default 1.0)
        anisotropy_kill_ratio: s_max/s_min ratio above which needles are dead (0 = disabled)
    Returns:
        [N] bool mask
    """
    # get_opacity is already activated (sigmoid applied)
    alpha = gaussians.get_opacity.squeeze(-1)
    support = (
        gaussians.visibility_ema.squeeze(-1)
        if hasattr(gaussians, "visibility_ema") and gaussians.visibility_ema.shape[0] == alpha.shape[0]
        else torch.ones_like(alpha)
    )

    very_low_opacity = opacity_threshold * 0.1  # ~0.0005 — unambiguously dead
    eff_support = support_threshold * scale_dead_support_mult

    # Clause 1: unconditionally kill near-zero opacity
    dead = alpha < very_low_opacity
    _bump_pruning_stat("dead_opacity", int(dead.sum().item()), run_state=run_state)

    # Clause 2: both opacity AND support below threshold
    clause2 = (alpha < opacity_threshold) & (support < support_threshold)
    _bump_pruning_stat("low_support", int((clause2 & ~dead).sum().item()), run_state=run_state)
    dead = dead | clause2

    # Clauses 3–5: scale/anisotropy-aware kills
    if hasattr(gaussians, "get_scaling"):
        s = gaussians.get_scaling
        max_scale = s.max(dim=-1).values
        # Clause 3: large Gaussian with any low support → floater (white-streak case)
        if scale_kill_threshold > 0:
            clause3 = (max_scale > scale_kill_threshold) & (support < eff_support)
            _bump_pruning_stat("scale_moderate", int((clause3 & ~dead).sum().item()), run_state=run_state)
            dead = dead | clause3
        # Clause 4: medium-scale semi-transparent with low support → blue-dot floater
        if scale_moderate_threshold > 0:
            clause4 = (
                (max_scale > scale_moderate_threshold)
                & (alpha < opacity_threshold * scale_moderate_opacity_mult)
                & (support < eff_support)
            )
            _bump_pruning_stat("scale_moderate", int((clause4 & ~dead).sum().item()), run_state=run_state)
            dead = dead | clause4
        # Clause 5: highly anisotropic needle with low support → white-streak floater
        if anisotropy_kill_ratio > 0:
            min_scale = s.min(dim=-1).values.clamp_min(1e-6)
            anisotropy = max_scale / min_scale
            clause5 = (
                (anisotropy > anisotropy_kill_ratio)
                & (alpha < opacity_threshold * scale_moderate_opacity_mult)
                & (support < eff_support)
            )
            _bump_pruning_stat("anisotropy_kill", int((clause5 & ~dead).sum().item()), run_state=run_state)
            dead = dead | clause5

    if use_utility_quantile and utility is not None and utility.numel() > 0:
        q_val = torch.quantile(utility, utility_quantile)
        clause6 = (alpha < opacity_threshold) & (utility < q_val)
        _bump_pruning_stat("utility_quantile", int((clause6 & ~dead).sum().item()), run_state=run_state)
        dead = dead | clause6

    # Birth-iter protection: Gaussians younger than young_prot_iters are excluded
    # from the dead mask so they have time to build opacity before being pruned.
    if young_prot_iters > 0 and current_iter is not None:
        birth_iter = getattr(gaussians, "birth_iter", None)
        if birth_iter is not None and birth_iter.shape[0] == dead.shape[0]:
            young = (current_iter - birth_iter.to(dtype=torch.int64)) < young_prot_iters
            young = _or_exposure_young(gaussians, young)
            protected = dead & young
            _bump_pruning_stat("young_protected_skip", int(protected.sum().item()), run_state=run_state)
            dead = dead & ~young
        elif birth_iter is not None:
            from eligsir.utils.gaussian_lineage import warn_young_prot_mismatch_once
            warn_young_prot_mismatch_once(
                gaussians,
                site="energy_mcmc.compute_dead_mask",
                warn_attr="_young_prot_mismatch_warned_energy_mcmc",
                birth_len=int(birth_iter.shape[0]),
                mask_len=int(dead.shape[0]),
            )

    _bump_pruning_stat("total_pruned", int(dead.sum().item()), run_state=run_state)

    return dead


def _or_exposure_young(gaussians, young):
    """Compose the iteration-currency young gate with the exposure-currency one.

    Issue #1262 §3.4: the age gate is a good mechanism denominated in the wrong
    currency -- it counts iterations, which stop proxying supervision opportunity
    the moment the tier varies. ``young = age_young | exposure_young`` keeps the
    age gate as a FLOOR (it protects a Gaussian that IS being measured but has
    not converged) and adds protection for one that is not being measured at all.
    No-op unless the resolution-exposure feature is on.
    """
    from eligsir.utils.resolution_exposure import exposure_young_mask
    extra = exposure_young_mask(gaussians, young.shape[0])
    if extra is None:
        return young
    return young | extra.to(young.device)
