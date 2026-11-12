"""EMAs and reducers turning transient per-pixel/per-Gaussian signals into
stable online evidence (plan section 10.4, "Add reducers and EMAs").

These are pure, side-effect-free-on-input helpers: every function here
takes a previous state tensor (or ``None``) and a new observation and
returns a *new* tensor, never mutating either input in place. Callers own
where the returned state is stored (see ``gaussian_history.py``). This
mirrors the existing detach convention in
``utils/strategies/mcmc/energy_policy.py``'s ``GaussianEnergyTerms``
docstring ("All history/state tensors... are detached... so they
contribute no gradient") — every reducer here detaches its output, since
evidence is an observer, never a second source of gradient into the model.
"""

from __future__ import annotations

from typing import Optional

import torch


def ema_update(
    prev: Optional[torch.Tensor],
    observation: torch.Tensor,
    decay: float = 0.9,
) -> torch.Tensor:
    """Exponential moving average: ``decay * prev + (1 - decay) * observation``.

    ``prev is None`` (cold start) returns ``observation`` detached, matching
    the usual EMA-warmup convention elsewhere in this codebase (e.g.
    ``utils/seed_fade_monitor.py``'s running-max bookkeeping) rather than
    seeding from zero, which would bias the first several updates low.
    """
    observation = observation.detach()
    if prev is None:
        return observation.clone()
    if prev.shape != observation.shape:
        # Topology changed shape (Gaussian count changed) without an
        # explicit re-key through gaussian_history.py's index remap —
        # treat as cold start for the mismatched region rather than
        # silently broadcasting/erroring. Real index-aware remapping on
        # insert/prune belongs to gaussian_history.py, not this reducer.
        return observation.clone()
    return decay * prev.detach() + (1.0 - decay) * observation


def masked_ema_update(
    prev: Optional[torch.Tensor],
    observation: torch.Tensor,
    mask: torch.Tensor,
    decay: float = 0.9,
) -> torch.Tensor:
    """EMA update restricted to ``mask`` (e.g. only visible Gaussians this
    frame). Entries outside ``mask`` keep their previous value unchanged
    (or ``0`` at cold start for that entry).
    """
    observation = observation.detach()
    if prev is None:
        prev = torch.zeros_like(observation)
    else:
        prev = prev.detach().clone()
    if prev.shape != observation.shape:
        return torch.where(mask, observation, torch.zeros_like(observation))
    updated = decay * prev + (1.0 - decay) * observation
    return torch.where(mask, updated, prev)


def scalar_mean(observation: torch.Tensor, mask: Optional[torch.Tensor] = None) -> float:
    """Reduce a per-pixel/per-Gaussian tensor to a single float, honoring an
    optional boolean mask. Returns ``0.0`` for an empty/all-False selection
    rather than NaN, so callers building ``EvidenceSummary`` never have to
    special-case an empty scene.
    """
    with torch.no_grad():
        if mask is not None:
            selected = observation[mask]
        else:
            selected = observation
        if selected.numel() == 0:
            return 0.0
        return float(selected.float().mean().item())


def scalar_fraction(mask: torch.Tensor) -> float:
    """Fraction of ``True`` entries in a boolean tensor. ``0.0`` for empty."""
    with torch.no_grad():
        if mask.numel() == 0:
            return 0.0
        return float(mask.float().mean().item())


def under_supported_mass(support: torch.Tensor, opacity: torch.Tensor, threshold: float) -> float:
    """Sum of opacity mass belonging to Gaussians whose ``support`` is below
    ``threshold`` — the raw material for ``EvidenceSummary.under_supported_mass``.
    Mirrors the low-support-opacity-mass (LSOM) concept already referenced
    in ``utils/mcmc_schedule.py``'s docstring, computed here purely as an
    observation (no feedback into any schedule from this module).
    """
    with torch.no_grad():
        if support.numel() == 0:
            return 0.0
        mask = support < threshold
        return float(opacity[mask].sum().item())
