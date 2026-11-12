"""Lifecycle transition rules (Wave 3, PR W3-09, plan section 14.6).

```text
PROVISIONAL
  -> ACTIVE after sufficient multi-view support
  -> SUSPECT after persistent conflict or non-contribution
ACTIVE
  -> STABLE after geometry convergence
  -> SUSPECT after persistent conflict
STABLE
  -> SUSPECT only under strong evidence
SUSPECT
  -> ACTIVE if evidence recovers
  -> RETIRED if confirmed unhelpful
RETIRED
  -> RESEEDED when a high-utility surface request exists
```

Geometry and appearance maturity are tracked independently (two calls to
:func:`transition_lifecycle_track`, one per track, driven by different
evidence signals) — see ``strategy.py``/``mutation_plan.py`` for how the two
tracks are combined into a single proposed
:class:`~eligsir.optimization.gaussian.strategies.contracts.LifecycleUpdateBatch`.

``RETIRED -> RESEEDED`` is not evidence-driven in the same per-Gaussian
sense as the other rows (it depends on whether a high-utility surface
request exists for that Gaussian's cell) — that transition is applied by
``mutation_plan.py`` after reseed requests are scored, not by this module.

This module never touches gaussian model tensors; it only reads/writes
:class:`~eligsir.optimization.gaussian.strategies.live.state.LivePopulationState`
buffers (or returns a proposed update without writing anything, when called
in dry-run mode from ``strategy.py``).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from eligsir.optimization.gaussian.strategies.contracts import GaussianLifecycleState
from eligsir.optimization.gaussian.strategies.live.state import state_to_code


@dataclass(frozen=True)
class LifecycleThresholds:
    """Tunable thresholds. Every field has a documented, conservative
    default; none of these are learned or auto-tuned in this PR."""

    min_support_for_active: float = 3.0
    min_geometry_confidence_for_active: float = 0.2

    min_geometry_confidence_for_stable: float = 0.8
    max_residual_for_stable: float = 0.05
    stable_convergence_streak: int = 10
    """Consecutive qualifying steps required before ACTIVE -> STABLE."""

    suspect_conflict_threshold: float = 0.5
    suspect_noncontribution_threshold: float = 0.02
    suspect_persistence_steps: int = 5
    """Consecutive qualifying (conflicted/non-contributing) steps required
    before PROVISIONAL/ACTIVE -> SUSPECT."""

    stable_suspect_conflict_threshold: float = 0.85
    """Stricter than ``suspect_conflict_threshold`` -- STABLE -> SUSPECT
    requires "strong evidence" per plan section 14.6."""
    stable_suspect_persistence_steps: int = 15

    recover_persistence_steps: int = 5
    """Consecutive qualifying (recovered) steps required before
    SUSPECT -> ACTIVE."""

    confirmed_unhelpful_steps: int = 30
    """Consecutive steps spent in SUSPECT without recovering before
    SUSPECT -> RETIRED."""


@dataclass
class TransitionResult:
    """One track's (geometry or appearance) proposed transitions this step."""

    indices: torch.Tensor
    old_state: torch.Tensor
    new_state: torch.Tensor
    reasons: tuple[str, ...]


_EMPTY_LONG = lambda device: torch.empty(0, dtype=torch.int64, device=device)  # noqa: E731


def transition_lifecycle_track(
    *,
    current_state_codes: torch.Tensor,
    support_count: torch.Tensor,
    geometry_confidence: torch.Tensor,
    conflict_ema: torch.Tensor,
    contribution_ema: torch.Tensor,
    residual_ema: torch.Tensor,
    suspect_streak: torch.Tensor,
    recover_streak: torch.Tensor,
    stable_streak: torch.Tensor,
    thresholds: LifecycleThresholds,
) -> tuple[TransitionResult, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Vectorized transition pass for one lifecycle track (geometry or
    appearance). Returns ``(TransitionResult, new_suspect_streak,
    new_recover_streak, new_stable_streak)`` -- the streak tensors are
    **not** written back by this function; the caller decides whether/when
    to persist them (this module has no side effects).

    ``RETIRED``/``RESEEDED`` Gaussians are left untouched by this function
    (RESEEDED is a terminal state here; RETIRED only ever leaves via the
    external reseed-request path in ``mutation_plan.py``).
    """
    device = current_state_codes.device

    provisional = state_to_code(GaussianLifecycleState.PROVISIONAL)
    active = state_to_code(GaussianLifecycleState.ACTIVE)
    suspect = state_to_code(GaussianLifecycleState.SUSPECT)
    stable = state_to_code(GaussianLifecycleState.STABLE)

    is_provisional = current_state_codes == provisional
    is_active = current_state_codes == active
    is_suspect = current_state_codes == suspect
    is_stable = current_state_codes == stable

    conflicted_or_noncontributing = (conflict_ema > thresholds.suspect_conflict_threshold) | (
        contribution_ema < thresholds.suspect_noncontribution_threshold
    )
    new_suspect_streak = torch.where(
        conflicted_or_noncontributing, suspect_streak + 1, torch.zeros_like(suspect_streak)
    )

    strong_conflict = conflict_ema > thresholds.stable_suspect_conflict_threshold
    new_stable_suspect_streak = torch.where(strong_conflict, suspect_streak + 1, torch.zeros_like(suspect_streak))

    recovered = (conflict_ema <= thresholds.suspect_conflict_threshold) & (
        contribution_ema >= thresholds.suspect_noncontribution_threshold
    )
    new_recover_streak = torch.where(recovered, recover_streak + 1, torch.zeros_like(recover_streak))

    converging = (geometry_confidence >= thresholds.min_geometry_confidence_for_stable) & (
        residual_ema <= thresholds.max_residual_for_stable
    )
    new_stable_streak = torch.where(converging, stable_streak + 1, torch.zeros_like(stable_streak))

    # --- PROVISIONAL -> ACTIVE ---
    provisional_to_active = is_provisional & (support_count >= thresholds.min_support_for_active) & (
        geometry_confidence >= thresholds.min_geometry_confidence_for_active
    )

    # --- PROVISIONAL/ACTIVE -> SUSPECT (persistent conflict/non-contribution) ---
    persistent_suspect = new_suspect_streak >= thresholds.suspect_persistence_steps
    provisional_to_suspect = is_provisional & persistent_suspect & ~provisional_to_active
    active_to_suspect = is_active & persistent_suspect

    # --- ACTIVE -> STABLE (geometry convergence) ---
    active_to_stable = is_active & (new_stable_streak >= thresholds.stable_convergence_streak) & ~active_to_suspect

    # --- STABLE -> SUSPECT (only under strong, sustained evidence) ---
    stable_to_suspect = is_stable & (new_stable_suspect_streak >= thresholds.stable_suspect_persistence_steps)

    # --- SUSPECT -> ACTIVE (recovered) ---
    suspect_to_active = is_suspect & (new_recover_streak >= thresholds.recover_persistence_steps)

    # --- SUSPECT -> RETIRED (confirmed unhelpful) ---
    suspect_to_retired = is_suspect & (suspect_streak >= thresholds.confirmed_unhelpful_steps) & ~suspect_to_active

    indices_list = []
    old_list = []
    new_list = []
    reasons_list: list[str] = []

    def _append(mask: torch.Tensor, new_code: int, reason: str) -> None:
        idx = torch.nonzero(mask, as_tuple=False).flatten()
        if idx.numel() == 0:
            return
        indices_list.append(idx)
        old_list.append(current_state_codes[idx])
        new_list.append(torch.full_like(idx, new_code, dtype=torch.int8))
        reasons_list.extend([reason] * int(idx.numel()))

    _append(provisional_to_active, active, "sufficient_multiview_support")
    _append(provisional_to_suspect, suspect, "persistent_conflict_or_noncontribution")
    _append(active_to_stable, stable, "geometry_convergence")
    _append(active_to_suspect, suspect, "persistent_conflict_or_noncontribution")
    _append(stable_to_suspect, suspect, "strong_sustained_conflict")
    _append(suspect_to_active, active, "evidence_recovered")
    _append(suspect_to_retired, state_to_code(GaussianLifecycleState.RETIRED), "confirmed_unhelpful")

    if indices_list:
        indices = torch.cat(indices_list)
        old_state = torch.cat(old_list)
        new_state = torch.cat(new_list)
    else:
        indices = _EMPTY_LONG(device)
        old_state = torch.empty(0, dtype=torch.int8, device=device)
        new_state = torch.empty(0, dtype=torch.int8, device=device)

    result = TransitionResult(
        indices=indices, old_state=old_state, new_state=new_state, reasons=tuple(reasons_list)
    )

    # Streak feeding into STABLE -> SUSPECT uses the strong-conflict streak,
    # not the ordinary suspect streak -- return that one for STABLE rows,
    # ordinary suspect streak otherwise, so the caller persists the right
    # value per current state.
    persisted_suspect_streak = torch.where(is_stable, new_stable_suspect_streak, new_suspect_streak)

    return result, persisted_suspect_streak, new_recover_streak, new_stable_streak
