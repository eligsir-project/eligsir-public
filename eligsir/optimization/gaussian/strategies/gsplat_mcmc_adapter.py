"""Shadow adapter over the currently-active legacy gsplat MCMC strategy
(Wave 3, PR W3-09, plan section 14.8 Subagent E: "dry-run comparison and
reports").

This module answers one question, read-only: *which indices would the
existing opacity-threshold dead-mask selection (the same rule
``GsplatMCMCBaselineStrategy``/``GsplatEnergyMCMCStrategy`` use before
calling ``gaussians.relocate_gs``) currently select?* — so
``strategy.py``/``mutation_plan.py`` can report an overlap count between
the live strategy's own proposal and what the real, currently-running
strategy is (or would be) doing, per plan section 14.7's "overlap with
current MCMC actions" observe-only telemetry item.

It **replicates the selection rule** (reused, not reinvented: the same
``min_opacity``/``args.mcmc_dead_opacity_end`` threshold and
``_exclude_locked_from_mask``/``_augment_dead_mask_knn_isolation`` helpers
that ``utils/strategies/mcmc/baseline.py`` already applies) but never calls
``gaussians.relocate_gs``/``add_new_gs`` — this module is read-only by
construction, matching every other module in this package.

Why relocation always begins from a dead/retired index (context for
``mutation_plan.py``'s deliverable-5 note): gsplat's MCMC relocation
mechanism (``compute_relocation``/``relocate_gs``) works by teleporting a
*dead* Gaussian (opacity below threshold) into a *live* Gaussian's
neighborhood, splitting mass between the two. A Gaussian never gets
relocated unless it is first selected into the dead mask — this is exactly
the selection this adapter mirrors, and exactly why keeping STABLE
Gaussians out of the live strategy's own retire/prune candidacy (see
``mutation_plan.py``) is the correct place to enforce "no global
teleportation of stable Gaussians."
"""

from __future__ import annotations

from typing import Any

import torch


def shadow_dead_mask(gaussians: Any, args: Any) -> torch.Tensor:
    """Read-only replica of the legacy MCMC baseline's dead-mask selection.

    Never calls any mutating method on ``gaussians``. Falls back to an
    all-False mask if ``gaussians``/``args`` don't expose the expected
    attributes (e.g. a stub in a unit test), rather than raising -- this is
    a best-effort comparison signal, not a hard dependency.
    """
    try:
        # No getattr fallback: mcmc_dead_opacity_end is an unconditional
        # OptimizationParams default, always present on the real parsed args
        # (issue #368 -- old getattr(..., 0.01) fallback drifted from the
        # real default of 0.05). Bare access here still degrades gracefully
        # via this function's own try/except for a genuinely attribute-less
        # stub (see docstring); removed rather than re-stated so a real run
        # can't silently apply the stale threshold.
        min_opacity = float(args.mcmc_dead_opacity_end)
        opacity = gaussians.get_opacity
        dead_mask = (opacity <= min_opacity).squeeze(-1)
    except Exception:
        return torch.zeros(0, dtype=torch.bool)

    try:
        from eligsir.utils.strategies.mcmc.base import _exclude_locked_from_mask

        dead_mask = _exclude_locked_from_mask(dead_mask, gaussians, args)
    except Exception:
        pass
    return dead_mask.detach()


def overlap_report(proposed_indices: torch.Tensor, shadow_mask: torch.Tensor) -> dict:
    """Compare the live strategy's ``proposed_indices`` (retire+prune) to
    the legacy strategy's ``shadow_mask``. Pure telemetry -- returns counts
    only, no side effects."""
    n_shadow = int(shadow_mask.shape[0])
    if n_shadow == 0:
        return {
            "proposed_count": int(proposed_indices.numel()),
            "shadow_count": 0,
            "overlap_count": 0,
            "proposed_only_count": int(proposed_indices.numel()),
            "shadow_only_count": 0,
        }
    in_range = proposed_indices[(proposed_indices >= 0) & (proposed_indices < n_shadow)]
    overlap = shadow_mask[in_range].sum().item() if in_range.numel() else 0
    shadow_count = int(shadow_mask.sum().item())
    proposed_count = int(proposed_indices.numel())
    return {
        "proposed_count": proposed_count,
        "shadow_count": shadow_count,
        "overlap_count": int(overlap),
        "proposed_only_count": proposed_count - int(overlap),
        "shadow_only_count": shadow_count - int(overlap),
    }
