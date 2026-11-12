"""Utility scoring for retire/prune/reseed candidates (Wave 3, PR W3-09,
plan section 14.7's "expected contribution change" / cell-budget ranking
inputs).

Deliberately excludes safety-fuse logic (deliverable 6: fuses live in
``mutation_plan.py``'s ``SafetyFuses``, not here) — every function in this
module is a pure score, never a hard veto/cap.

``free_space_conflict_from_consensus`` wires directly to W3-08's
:class:`~eligsir.mapping.geometry.free_space_consensus.FreeSpaceConsensusIndex`
(per this workstream's brief: "wire to it rather than inventing a parallel
free-space signal") — it queries the same cross-frame voxel-vote store the
tri-state ray-classifier's ``CERTIFIED_FREE_SPACE`` label is built from, so
a Gaussian sitting inside a voxel multiple independent views have certified
as free space scores as being in conflict with that evidence.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from eligsir.mapping.geometry.free_space_consensus import FreeSpaceConsensusIndex


def free_space_conflict_from_consensus(
    positions: torch.Tensor,
    index: "FreeSpaceConsensusIndex",
    *,
    min_views: int = 3,
) -> torch.Tensor:
    """Per-Gaussian ``[N]`` float conflict score in ``[0, 1]``: ``1.0`` where
    ``positions[i]`` falls in a voxel certified free space by ``index``
    (``>= min_views`` corroborating views, see
    ``FreeSpaceConsensusIndex.certified_mask``), ``0.0`` otherwise.

    Read-only against ``index`` — never calls ``vote_free``; that is the
    caller's (mapper's) job when it processes an admitted keyframe.
    """
    if positions.numel() == 0:
        return torch.zeros(0, dtype=torch.float32, device=positions.device)
    mask = index.certified_mask(positions, min_views=min_views)
    return mask.to(torch.float32)


def retire_score(
    *,
    conflict_ema: torch.Tensor,
    contribution_ema: torch.Tensor,
    support_count: torch.Tensor,
) -> torch.Tensor:
    """Higher = more worth retiring. Pure function of evidence, no fuses."""
    support_penalty = 1.0 / (1.0 + support_count.clamp(min=0.0))
    return (conflict_ema.clamp(0.0, 1.0) + support_penalty) - contribution_ema.clamp(0.0, 1.0)


def prune_score(
    *,
    geometry_confidence: torch.Tensor,
    conflict_ema: torch.Tensor,
) -> torch.Tensor:
    """Higher = more worth pruning outright (stronger than retire: prune is
    reserved for Gaussians already RETIRED with no recovering evidence and
    high, sustained free-space conflict -- see ``mutation_plan.py``)."""
    return conflict_ema.clamp(0.0, 1.0) * (1.0 - geometry_confidence.clamp(0.0, 1.0))


def reseed_utility(
    *,
    under_supported_mass: float,
    persistent_residual: float,
    cell_new_surface_ratio: float,
) -> float:
    """Scalar utility for a single candidate reseed cell. Higher = more
    valuable to reseed. All three inputs already exist as observation
    concepts in ``eligsir.mapping.evidence.contracts.EvidenceSummary``
    (``under_supported_mass``, ``persistent_residual``, ``new_surface_ratio``)
    -- this function only combines them, it does not recompute them."""
    return float(under_supported_mass + persistent_residual + cell_new_surface_ratio)
