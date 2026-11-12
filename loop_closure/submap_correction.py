"""Loop-closure Gaussian coordinate corrections through the mutation owner."""

from __future__ import annotations

from typing import List

import numpy as np
import torch

from eligsir.map.mutation import SubmapCorrection, mutation_transaction_for_model


def _populated(model, name: str) -> bool:
    value = getattr(model, name, None)
    return value is not None and value.shape[0] == model.get_xyz.shape[0] and value.shape[0] > 0


def _rigid_anchor_payloads(gaussians, mask, R, t):
    anchor_xyz = None
    anchor_normal = None
    if _populated(gaussians, "anchor_xyz"):
        anchor_xyz = (R @ gaussians.anchor_xyz[mask].T).T + t
    if _populated(gaussians, "anchor_normal"):
        rotated = (R @ gaussians.anchor_normal[mask].T).T
        anchor_normal = torch.nn.functional.normalize(rotated, dim=-1, eps=1e-8)
    return anchor_xyz, anchor_normal


def _translated_anchor_payloads(gaussians, indices, delta):
    anchor_xyz = None
    anchor_normal = None
    if _populated(gaussians, "anchor_xyz"):
        anchor_xyz = gaussians.anchor_xyz[indices] + delta
    if _populated(gaussians, "anchor_normal"):
        # Non-rigid pair pull is translational only; normals remain in the same frame.
        anchor_normal = gaussians.anchor_normal[indices].clone()
    return anchor_xyz, anchor_normal


def apply_submap_correction(
    gaussians,
    submap_manager,
    affected_submap_ids: List[int],
    optimised_transforms: List[np.ndarray],
    map_backend=None,
) -> int:
    """Apply rigid submap corrections atomically to means + world-frame anchors."""
    if not hasattr(gaussians, "submap_id") or gaussians.submap_id.shape[0] == 0:
        return 0

    total_corrected = 0
    with torch.no_grad():
        txn = (
            map_backend.begin_mutation()
            if map_backend is not None
            else mutation_transaction_for_model(gaussians)
        )
        for sid in affected_submap_ids:
            if sid >= len(optimised_transforms):
                continue
            T_new = optimised_transforms[sid]
            T_old = submap_manager.get_submap_transform(sid)
            delta = (T_new @ np.linalg.inv(T_old)).astype(np.float32)
            delta_t = torch.as_tensor(delta, device=gaussians.get_xyz.device)
            R = delta_t[:3, :3]
            t = delta_t[:3, 3]

            mask = gaussians.submap_id == sid
            if not mask.any():
                continue
            xyz = gaussians.get_xyz[mask]
            new_xyz = (R @ xyz.T).T + t
            new_anchor_xyz, new_anchor_normal = _rigid_anchor_payloads(
                gaussians, mask, R, t
            )
            txn.apply_pose_correction(
                SubmapCorrection(
                    mask=mask,
                    new_positions=new_xyz,
                    new_anchor_xyz=new_anchor_xyz,
                    new_anchor_normal=new_anchor_normal,
                )
            )
            total_corrected += int(mask.sum())

        if map_backend is not None:
            map_backend.commit_mutation(txn)
        else:
            txn.commit()

    # Non-Gaussian submap transforms/spatial ownership remain with SubmapManager.
    submap_manager.apply_pose_graph_corrections(optimised_transforms, gaussians=None)
    return total_corrected


def apply_gaussian_merge_correction(
    gaussians,
    match_radius: float = 0.1,
    absorb_distance: float = 0.01,
    map_backend=None,
) -> int:
    """Per-pair duplicate correction without bypassing coordinate invariants."""
    from loop_closure.gaussian_force_field import (
        match_gaussians,
        select_mature_targets,
        select_provisional_sources,
    )

    tgt_idx = select_mature_targets(gaussians)
    src_idx = select_provisional_sources(gaussians)
    if tgt_idx.numel() == 0 or src_idx.numel() == 0:
        return 0

    src_xyz = gaussians.get_xyz[src_idx].detach()
    tgt_xyz = gaussians.get_xyz[tgt_idx].detach()
    match_src, match_tgt = match_gaussians(src_xyz, tgt_xyz, radius=match_radius)
    if match_src.numel() == 0:
        return 0

    src_global = src_idx[match_src]
    tgt_global = tgt_idx[match_tgt]
    dists = (gaussians.get_xyz[src_global] - gaussians.get_xyz[tgt_global]).norm(dim=1)

    n_corrected = 0
    with torch.no_grad():
        txn = (
            map_backend.begin_mutation()
            if map_backend is not None
            else mutation_transaction_for_model(gaussians)
        )
        absorb_mask = dists < absorb_distance
        if absorb_mask.any():
            absorbed_src = src_global[absorb_mask]
            absorbed_tgt = tgt_global[absorb_mask]
            gaussians.support_count.index_add_(
                0, absorbed_tgt, gaussians.support_count[absorbed_src]
            )
            if hasattr(gaussians, "provisional"):
                gaussians.provisional[absorbed_src] = False
            n_corrected += int(absorbed_src.numel())

        pull_mask = ~absorb_mask
        if pull_mask.any():
            pull_src = src_global[pull_mask]
            pull_tgt = tgt_global[pull_mask]
            mature_support = gaussians.support_count[pull_tgt].float()
            prov_support = gaussians.support_count[pull_src].float().clamp(min=1.0)
            weight = (mature_support / (mature_support + prov_support)).clamp(0.0, 0.9)
            delta = gaussians.get_xyz[pull_tgt] - gaussians.get_xyz[pull_src]
            applied_delta = weight.unsqueeze(-1) * delta
            new_positions = gaussians.get_xyz[pull_src].detach() + applied_delta
            new_anchor_xyz, new_anchor_normal = _translated_anchor_payloads(
                gaussians, pull_src, applied_delta
            )
            txn.apply_pose_correction(
                SubmapCorrection(
                    mask=pull_src,
                    new_positions=new_positions,
                    new_anchor_xyz=new_anchor_xyz,
                    new_anchor_normal=new_anchor_normal,
                )
            )
            n_corrected += int(pull_src.numel())

        if map_backend is not None:
            map_backend.commit_mutation(txn)
        else:
            txn.commit()

    return n_corrected
