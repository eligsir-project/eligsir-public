"""Commit pending surface candidates through the sole mutation boundary.

The steady-state path is device native. Legacy NumPy candidate producers are
accepted only at this compatibility edge and are packed into one heterogeneous
pinned slab before one physical H2D; there are no per-field transfers and no
direct ``add_points_as_gaussians`` call from this module.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from geometry.insertion import SurfaceCandidatesT
from eligsir.map.arbiter import (
    GaussianMutationArbiter,
    ProposalKey,
    SurfaceInsertionProposal,
)
from eligsir.mapping.integration.device_candidates import (
    deterministic_subsample_indices,
    deterministic_voxel_first_indices,
    normalize_pending_candidates,
)

if TYPE_CHECKING:
    from argparse import Namespace


def _ephemeral_backend_for_tests(gaussians):
    """Transaction-only fallback for tests that do not build a full runtime."""
    if hasattr(gaussians, "params"):
        from eligsir.map.adapters.gsplat import GsplatMapBackend

        return GsplatMapBackend(gaussians)
    from eligsir.map.adapters.legacy_graphdeco import LegacyGraphdecoMapBackend

    return LegacyGraphdecoMapBackend(gaussians)


def _pending_count(pending) -> int:
    total = 0
    for p in pending:
        total += len(p) if isinstance(p, SurfaceCandidatesT) else int(p["pts"].shape[0])
    return total


def _cohort_frame_indices(pending, selected: torch.Tensor, device: torch.device) -> np.ndarray:
    pieces = []
    for p in pending:
        n = len(p) if isinstance(p, SurfaceCandidatesT) else int(p["pts"].shape[0])
        birth = int(p.birth_frame if isinstance(p, SurfaceCandidatesT) else p["birth_frame"])
        pieces.append(torch.full((n,), birth, dtype=torch.int64, device=device))
    values = torch.cat(pieces, dim=0)[selected]
    return values.detach().cpu().numpy()


def flush_insertion_batch(
    pending: list,
    gaussians,
    streaming_scene,
    iteration: int,
    args: "Namespace",
    map_backend=None,
    cohort_debug_out: list | None = None,
    insertion_plan=None,
    insertion_policy=None,
) -> int:
    """Commit one deterministic device-native insertion batch.

    ``SurfaceCandidatesT`` inputs never leave their device. Legacy NumPy
    inputs cross through ``normalize_pending_candidates`` as one packed H2D.
    Voxel dedup and deterministic subsampling remain torch-native.
    """
    if not pending:
        return 0

    device = gaussians.get_xyz.device
    init_opacity = float(
        getattr(
            pending[0],
            "init_opacity",
            pending[0].get("init_opacity", insertion_policy.initial_opacity)
            if isinstance(pending[0], dict) and insertion_policy is not None
            else (
                insertion_policy.initial_opacity
                if insertion_policy is not None
                else getattr(args, "streaming_insert_opacity", 0.05)
            ),
        )
    )
    candidates = normalize_pending_candidates(
        pending, device=device, init_opacity=init_opacity
    )
    n_before_dedup = len(candidates)
    selected = torch.arange(n_before_dedup, device=device, dtype=torch.long)

    if n_before_dedup > 1:
        voxel_size = float(
            insertion_plan.voxel_size_m
            if insertion_plan is not None
            else getattr(args, "streaming_insert_voxel_size", 0.02)
        )
        keep = deterministic_voxel_first_indices(candidates.pts, voxel_size)
        candidates = candidates.select(keep)
        selected = selected[keep]

    max_new = int(
        insertion_plan.candidate_budget
        if insertion_plan is not None
        else args.streaming_max_new_gaussians_per_frame
    )
    if max_new > 0 and len(candidates) > max_new:
        keep = deterministic_subsample_indices(
            len(candidates), max_new, iteration=iteration, device=device
        )
        candidates = candidates.select(keep)
        selected = selected[keep]

    if cohort_debug_out is not None:
        cohort_debug_out.append(
            {
                "frame_index": _cohort_frame_indices(pending, selected, device),
                "xyz": candidates.pts.detach().cpu().numpy().copy(),
                "n_before_dedup": n_before_dedup,
                "n_deduped": n_before_dedup - len(candidates),
            }
        )

    backend = map_backend if map_backend is not None else _ephemeral_backend_for_tests(gaussians)
    cap = args.cap_max
    if cap is not None:
        try:
            cap = int(cap)
        except (TypeError, ValueError):
            cap = None
    arbiter = GaussianMutationArbiter(backend, max_population=cap)
    proposal = SurfaceInsertionProposal(
        key=ProposalKey(
            stream_epoch=0,
            source_ordinal=int(candidates.birth_frame),
            timestamp_ns=int(iteration),
            source_id="rgbd-surface",
        ),
        batch=candidates.as_insertion_batch(iteration=iteration),
    )
    result = arbiter.commit((proposal,))
    added = int(result.inserted)

    if added > 0:
        new_xyz = gaussians.get_xyz[-added:]
        streaming_scene.add_to_occupancy_hash(new_xyz)
        if bool(getattr(args, "streaming_surface_cell_dedup", False)):
            streaming_scene.add_to_surface_cells(new_xyz, gaussians.anchor_normal[-added:])
        if hasattr(gaussians, "anchor_iter"):
            gaussians.anchor_iter[-added:] = int(iteration)
    return added


class InsertionCommitter:
    """Service wrapper used by the mapping coordinator."""

    def commit(
        self,
        pending: list,
        gaussians,
        streaming_scene,
        iteration: int,
        args: "Namespace",
        map_backend=None,
        insertion_plan=None,
        insertion_policy=None,
    ) -> int:
        return flush_insertion_batch(
            pending,
            gaussians,
            streaming_scene,
            iteration,
            args,
            map_backend=map_backend,
            insertion_plan=insertion_plan,
            insertion_policy=insertion_policy,
        )
