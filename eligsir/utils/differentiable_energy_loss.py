"""Differentiable energy losses for Gaussian regularization (Phase 5).

Centralises the regularizer calls currently scattered at the call sites in
``train_streaming.py`` / ``train.py``. The pixel-space RGB/depth losses
remain in the training loop; this module covers the Gaussian-space terms:

- ``active_reg_core`` — penalises opacity of inactive (invisible) Gaussians
- ``opacity_entropy_core`` — entropy regularizer pushing opacity to 0 or 1
- ``compute_effective_count_loss`` — steers effective Gaussian count toward target

State tensors (``support``, ``birth_frame``) used as *detached weights* only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class DifferentiableLossResult:
    """Output of :func:`compute_differentiable_loss`.

    All loss scalars are differentiable w.r.t. Gaussian parameters unless
    otherwise noted.

    Attributes:
        active_reg: Opacity penalty for invisible Gaussians.
        opacity_entropy: Entropy regularizer.
        effective_count: Effective-count steering loss.
        scale_reg: L2 scale prior (zero unless w_scale_reg > 0).
        anisotropy_reg: Anisotropy penalty (zero unless w_anisotropy_reg > 0).
        total: Weighted sum of all active components.
    """

    active_reg: torch.Tensor
    opacity_entropy: torch.Tensor
    effective_count: torch.Tensor
    scale_reg: torch.Tensor
    anisotropy_reg: torch.Tensor
    needle_reg: torch.Tensor
    total: torch.Tensor

    def to_tensorboard_dict(self) -> dict:
        """Return TensorBoard scalars.

        Returns:
            Dict of metric name → float.
        """
        return {
            "energy_loss/active_reg": self.active_reg.item(),
            "energy_loss/opacity_entropy": self.opacity_entropy.item(),
            "energy_loss/effective_count": self.effective_count.item(),
            "energy_loss/scale_reg": self.scale_reg.item(),
            "energy_loss/anisotropy_reg": self.anisotropy_reg.item(),
            "energy_loss/needle_reg": self.needle_reg.item(),
            "energy_loss/total": self.total.item(),
        }


def compute_differentiable_loss(
    gaussians,
    iteration: int,
    cap_max: int,
    active_mask: Optional[torch.Tensor] = None,
    include_effective_count: bool = False,
    w_opacity_reg: float = 0.01,
    w_scale_reg: float = 0.01,
    w_opacity_entropy: float = 1.0,
    w_effective_count: float = 0.1,
    w_scale_prior: float = 0.0,
    w_anisotropy_reg: float = 0.0,
    w_needle_reg: float = 0.0,
    needle_threshold: float = 0.3,
    scale_reg_mode: str = "l1",
    scale_prior_max: float = 0.10,
    anisotropy_soft_limit: float = 10.0,
    max_iterations: int = 30000,
) -> DifferentiableLossResult:
    """Compute all Gaussian-space differentiable regularization losses.

    Phase 5 implementation: delegates to the existing compiled kernel calls
    (``active_reg_core``, ``opacity_entropy_core``, ``compute_effective_count_loss``)
    matching the legacy training-loop calling convention exactly.

    ``w_scale_prior`` and ``w_anisotropy_reg`` default to 0.0 so they are
    off unless the caller opts in — keeping total parity with legacy paths.

    Args:
        gaussians: Gaussian model.
        iteration: Current training iteration.
        cap_max: Maximum Gaussian count (for effective-count target).
        active_mask: Bool visibility mask; inactive Gaussians penalised for
            high opacity. If None, all Gaussians are treated as active.
        include_effective_count: Whether to include the effective-count loss.
        w_opacity_reg: Weight forwarded to active_reg_core w_opacity. Default 0.01.
        w_scale_reg: Weight forwarded to active_reg_core w_scale. Default 0.01.
        w_opacity_entropy: Weight for opacity entropy. Default 1.0.
        w_effective_count: Weight for effective-count loss. Default 0.1.
        w_scale_prior: Weight for extra L2 scale prior. Default 0.0 (off).
        w_anisotropy_reg: Weight for extra anisotropy penalty. Default 0.0 (off).
        scale_prior_max: Max scale above which L2 prior fires. Default 0.10 m.
        anisotropy_soft_limit: Axis ratio above which penalty starts. Default 10.
        max_iterations: Training horizon for effective-count target curve.

    Returns:
        :class:`DifferentiableLossResult` with all loss components.
    """
    from eligsir.utils.compiled_kernels import active_reg_core, opacity_entropy_core

    device = gaussians.get_xyz.device

    # Raw opacities (pre-sigmoid, differentiable)
    raw_op = (
        gaussians.params["opacities"]
        if hasattr(gaussians, "params") and "opacities" in gaussians.params
        else gaussians._opacity
    )

    # --- Active regularizer ---
    # Match the legacy training-loop calling convention exactly:
    # active_reg_core(get_opacity[mask], get_scaling[mask], w_opacity=opacity_reg, w_scale=scale_reg)
    # Campaign C4: scale_reg_mode="l2" replaces the L1 scale term (which is sparsity-inducing
    # and drives minor axes toward zero -> anisotropy at the source) with a Frobenius L2 term.
    if active_mask is not None and active_mask.shape[0] == raw_op.shape[0]:
        _op_active = gaussians.get_opacity[active_mask]
        _sc_active = gaussians.get_scaling[active_mask]
    else:
        _op_active = gaussians.get_opacity
        _sc_active = gaussians.get_scaling
    if scale_reg_mode == "l2":
        active_loss = active_reg_core(_op_active, _sc_active, w_opacity=w_opacity_reg, w_scale=0.0)
        active_loss = active_loss + w_scale_reg * (_sc_active ** 2).mean()
    else:
        active_loss = active_reg_core(_op_active, _sc_active, w_opacity=w_opacity_reg, w_scale=w_scale_reg)

    # --- Opacity entropy ---
    entropy_loss = opacity_entropy_core(raw_op)

    # --- Effective count ---
    eff_count_loss = torch.tensor(0.0, device=device)
    if include_effective_count:
        from eligsir.utils.energy_mcmc import compute_effective_count_loss
        _l, _n_eff, _n_target = compute_effective_count_loss(
            raw_op,
            iteration=iteration,
            cap_max=cap_max,
            max_iterations=max_iterations,
        )
        eff_count_loss = _l

    # --- Optional extra priors (off by default for legacy parity) ---
    scale_loss = torch.tensor(0.0, device=device)
    aniso_loss = torch.tensor(0.0, device=device)
    if w_scale_prior > 0.0:
        s = gaussians.get_scaling
        max_scale = s.max(dim=-1).values
        scale_loss = (max_scale.clamp(min=scale_prior_max) - scale_prior_max).pow(2).mean()
    if w_anisotropy_reg > 0.0:
        s = gaussians.get_scaling
        max_scale = s.max(dim=-1).values
        min_scale = s.min(dim=-1).values.clamp_min(1e-6)
        anisotropy = max_scale / min_scale
        aniso_loss = (anisotropy.clamp(min=anisotropy_soft_limit) - anisotropy_soft_limit).pow(2).mean()

    # --- Needle penalty (targets second-axis collapse: needle <=> s2/s1 small) ---
    # Unlike the s_max/s_min anisotropy term above (which also penalises healthy flat
    # disks where s3<<s2), this acts ONLY on the two largest axes, so it suppresses
    # needle/spike Gaussians without damaging surfels. Hinge: only fires below threshold.
    needle_loss = torch.tensor(0.0, device=device)
    if w_needle_reg > 0.0:
        s = gaussians.get_scaling
        top2 = torch.topk(s, k=min(2, s.shape[-1]), dim=-1).values  # [N,2] descending
        s1 = top2[:, 0].clamp_min(1e-6)
        s2 = top2[:, 1] if top2.shape[-1] > 1 else s1
        ratio = (s2 / s1).clamp(0.0, 1.0)                           # small => needle
        needle_loss = (needle_threshold - ratio).clamp_min(0.0).pow(2).mean()

    # --- Non-finite term guard -------------------------------------------------
    # These regularizers read opacity/scale GLOBALLY (or over the active set), so a
    # single transient extreme Gaussian — e.g. a raw scale that exp()-overflows to
    # inf — can drive one sub-term to NaN/Inf even while the rendered image is fine.
    # Left unguarded that NaN propagates through backward and corrupts params. Drop
    # ONLY the offending term for this iteration, as a DETACHED zero so no NaN
    # gradient flows (nan_to_num on the reduced scalar would not block the backward:
    # 0 * nan = nan). The photometric loss and the finite terms keep training, and
    # the offending Gaussian is removed by the normal pruning machinery → self-heals.
    _dropped = []

    def _finite(term, name):
        if term is None or not bool(torch.isfinite(term)):
            _dropped.append(name)
            return torch.zeros((), device=device)
        return term

    active_loss = _finite(active_loss, "active_reg")
    entropy_loss = _finite(entropy_loss, "opacity_entropy")
    eff_count_loss = _finite(eff_count_loss, "effective_count")
    scale_loss = _finite(scale_loss, "scale_reg")
    aniso_loss = _finite(aniso_loss, "anisotropy_reg")
    needle_loss = _finite(needle_loss, "needle_reg")
    if _dropped:
        _c = getattr(compute_differentiable_loss, "_nan_term_warns", 0)
        if _c < 8:
            compute_differentiable_loss._nan_term_warns = _c + 1
            print(f"[energy-loss] iter={iteration} dropped non-finite term(s): "
                  f"{','.join(_dropped)} (N={raw_op.shape[0]})", flush=True)

    total = (
        active_loss
        + w_opacity_entropy * entropy_loss
        + w_effective_count * eff_count_loss
        + w_scale_prior * scale_loss
        + w_anisotropy_reg * aniso_loss
        + w_needle_reg * needle_loss
    )

    return DifferentiableLossResult(
        active_reg=active_loss,
        opacity_entropy=entropy_loss,
        effective_count=eff_count_loss,
        scale_reg=scale_loss,
        anisotropy_reg=aniso_loss,
        needle_reg=needle_loss,
        total=total,
    )
