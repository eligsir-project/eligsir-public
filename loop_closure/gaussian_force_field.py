"""Gaussian force-field loop closure diagnostic (Phase 11).

Computes force vectors between provisional Gaussians (sources) and mature
Gaussians at the same scene location (targets) to detect loop-closure
candidates. Phase 11 is **observe-only**: no pose or map correction.

Force field coherence is logged to TensorBoard as a loop-closure signal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
import torch


@dataclass
class ForceFieldResult:
    """Result of a force-field diagnostic run.

    Attributes:
        inliers: Number of matched Gaussian pairs.
        rmse: Alignment RMSE (metres).
        translation_norm: L2 norm of the fitted translation (metres).
        rotation_deg: Rotation magnitude of the fitted SE3 (degrees).
        shear_norm: Frobenius norm of the off-diagonal affine residual.
        coherence: Scalar [0, 1] coherence score.
        T_old_current: Fitted 4×4 world-space SE3 mapping CURRENT
            (provisional) Gaussian world positions onto OLD (mature) Gaussian
            world positions -- i.e. despite the field name, it maps
            current->old, not old->current (fit_weighted_se3(src=current
            provisional, tgt=old mature) fits src->tgt). This is a raw
            world-space delta, not a submap-local pose-graph edge; see
            loop_closure.pose_graph.world_delta_to_local_edge for the
            required conversion before handing it to
            PoseGraph.add_loop_edge.
        old_submap_id: Submap id the matched TARGET (mature) Gaussians
            actually belong to -- the mode over matched targets'
            gaussians.submap_id. This is the correct "old" node for the
            loop edge; the call site must NOT hardcode submap 0, which
            mislabels every revisit that isn't against the very first
            submap (e.g. a long down-and-back corridor revisits mid-
            trajectory submaps continuously, never submap 0 specifically).
    """

    inliers: int = 0
    rmse: float = float("nan")
    translation_norm: float = float("nan")
    rotation_deg: float = float("nan")
    shear_norm: float = float("nan")
    coherence: float = 0.0
    T_old_current: Optional[np.ndarray] = None
    old_submap_id: int = 0

    def to_tensorboard_dict(self) -> dict:
        """Return TensorBoard scalars.

        Returns:
            Dict mapping metric names to float values.
        """
        return {
            "loop_force/inliers": float(self.inliers),
            "loop_force/rmse": self.rmse,
            "loop_force/translation_norm": self.translation_norm,
            "loop_force/rotation_deg": self.rotation_deg,
            "loop_force/shear_norm": self.shear_norm,
            "loop_force/coherence": self.coherence,
        }


def select_mature_targets(gaussians, min_support: float = 0.5) -> torch.Tensor:
    """Select well-supported mature Gaussians as loop-closure targets.

    Args:
        gaussians: Gaussian model with ``visibility_ema`` and optional
            ``provisional`` buffers.
        min_support: Minimum visibility EMA for a target Gaussian.

    Returns:
        ``[M]`` index tensor of mature target Gaussians.
    """
    support = getattr(gaussians, "visibility_ema", None)
    if support is None:
        return torch.arange(gaussians.get_xyz.shape[0], device=gaussians.get_xyz.device)
    support = support.squeeze(-1)
    mask = support >= min_support
    if hasattr(gaussians, "provisional"):
        mask = mask & ~gaussians.provisional
    return mask.nonzero(as_tuple=True)[0]


def select_provisional_sources(gaussians) -> torch.Tensor:
    """Select provisional Gaussians as force-field source candidates.

    Args:
        gaussians: Gaussian model with optional ``provisional`` buffer.

    Returns:
        ``[S]`` index tensor of provisional Gaussians.
    """
    if hasattr(gaussians, "provisional"):
        return gaussians.provisional.nonzero(as_tuple=True)[0]
    return torch.arange(gaussians.get_xyz.shape[0], device=gaussians.get_xyz.device)


def match_gaussians(
    src_xyz: torch.Tensor,
    tgt_xyz: torch.Tensor,
    radius: float = 0.1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Nearest-neighbour matching between source and target Gaussians.

    Args:
        src_xyz: ``[S, 3]`` source positions.
        tgt_xyz: ``[T, 3]`` target positions.
        radius: Maximum match radius (metres).

    Returns:
        Tuple of ``(src_idx, tgt_idx)`` matched index tensors.
    """
    if src_xyz.shape[0] == 0 or tgt_xyz.shape[0] == 0:
        empty = torch.zeros(0, dtype=torch.long, device=src_xyz.device)
        return empty, empty

    dists = torch.cdist(src_xyz, tgt_xyz)
    min_dists, min_idx = dists.min(dim=1)
    valid = min_dists < radius
    src_idx = valid.nonzero(as_tuple=True)[0]
    tgt_idx = min_idx[src_idx]
    return src_idx, tgt_idx


def compute_force_vectors(
    src_xyz: torch.Tensor,
    tgt_xyz: torch.Tensor,
) -> torch.Tensor:
    """Compute per-pair force vectors f_i = tgt - src.

    Args:
        src_xyz: ``[M, 3]`` source positions.
        tgt_xyz: ``[M, 3]`` matched target positions.

    Returns:
        ``[M, 3]`` force vectors pointing from source to target.
    """
    return tgt_xyz - src_xyz


def fit_weighted_se3(
    src_xyz: np.ndarray,
    tgt_xyz: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, float]:
    """Fit a weighted rigid SE3 transform from src to tgt using SVD.

    Args:
        src_xyz: ``[M, 3]`` source positions.
        tgt_xyz: ``[M, 3]`` matched target positions.
        weights: Optional ``[M]`` per-pair weights. Uniform if None.

    Returns:
        Tuple of (4×4 SE3 matrix, RMSE in metres).
    """
    if len(src_xyz) < 3:
        return np.eye(4, dtype=np.float32), float("nan")

    if weights is None:
        weights = np.ones(len(src_xyz))
    weights = weights / weights.sum()

    src_mean = (weights[:, None] * src_xyz).sum(axis=0)
    tgt_mean = (weights[:, None] * tgt_xyz).sum(axis=0)

    src_c = src_xyz - src_mean
    tgt_c = tgt_xyz - tgt_mean

    H = (weights[:, None] * src_c).T @ tgt_c
    U, S, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    D = np.diag([1, 1, d])
    R = Vt.T @ D @ U.T
    t = tgt_mean - R @ src_mean

    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R.astype(np.float32)
    T[:3, 3] = t.astype(np.float32)

    residuals = tgt_xyz - (R @ src_xyz.T).T - t
    rmse = float(np.sqrt((weights * (residuals ** 2).sum(axis=1)).sum()))
    return T, rmse


def score_force_field(
    src_xyz: torch.Tensor,
    tgt_xyz: torch.Tensor,
    T: np.ndarray,
    rmse: float,
    max_rmse: float = 0.05,
) -> float:
    """Compute a coherence score in [0, 1] for the force field.

    Args:
        src_xyz: Matched source positions.
        tgt_xyz: Matched target positions.
        T: Fitted SE3 transform.
        rmse: Alignment RMSE.
        max_rmse: RMSE above this → coherence = 0. Default 0.05 m.

    Returns:
        Coherence score in [0, 1].
    """
    if len(src_xyz) == 0 or np.isnan(rmse):
        return 0.0
    coherence = max(0.0, 1.0 - rmse / max_rmse)
    return coherence


def run_force_field_diagnostic(
    gaussians,
    match_radius: float = 0.1,
    max_src: int = 5000,
    max_rmse: float = 0.05,
) -> ForceFieldResult:
    """Run one complete force-field diagnostic pass (Phase 11 — observe only).

    No map or pose correction is applied.

    Args:
        gaussians: Gaussian model.
        match_radius: Nearest-neighbour matching radius (metres).
        max_src: Maximum provisional sources to consider.
        max_rmse: RMSE above which coherence collapses to 0 (metres). Must be
            widened alongside match_radius -- see score_force_field.

    Returns:
        :class:`ForceFieldResult` with diagnostic metrics.
    """
    tgt_idx = select_mature_targets(gaussians)
    src_idx = select_provisional_sources(gaussians)

    if tgt_idx.numel() == 0 or src_idx.numel() == 0:
        return ForceFieldResult()

    if src_idx.numel() > max_src:
        perm = torch.randperm(src_idx.numel())[:max_src]
        src_idx = src_idx[perm]

    src_xyz = gaussians.get_xyz[src_idx].detach()
    tgt_xyz = gaussians.get_xyz[tgt_idx].detach()

    match_src, match_tgt = match_gaussians(src_xyz, tgt_xyz, radius=match_radius)
    n_inliers = int(match_src.numel())

    if n_inliers < 3:
        return ForceFieldResult(inliers=n_inliers)

    src_np = src_xyz[match_src].cpu().numpy()
    tgt_np = tgt_xyz[match_tgt].cpu().numpy()

    T, rmse = fit_weighted_se3(src_np, tgt_np)
    coherence = score_force_field(src_xyz[match_src], tgt_xyz[match_tgt], T, rmse, max_rmse=max_rmse)

    t_norm = float(np.linalg.norm(T[:3, 3]))
    trace = np.clip((np.trace(T[:3, :3]) - 1.0) * 0.5, -1.0, 1.0)
    rot_deg = float(np.degrees(np.arccos(trace)))

    # Which submap did the matched TARGET (mature) Gaussians actually come
    # from? Mode over matched targets, NOT a hardcoded submap 0 -- on a long
    # down-and-back corridor, revisits are against mid/end-of-trajectory
    # submaps continuously, and mislabeling them all as "vs submap 0" feeds
    # the pose graph wrong constraints for almost every real revisit.
    old_submap_id = 0
    if hasattr(gaussians, "submap_id") and gaussians.submap_id.shape[0] == gaussians.get_xyz.shape[0]:
        matched_tgt_global_idx = tgt_idx[match_tgt]
        matched_submap_ids = gaussians.submap_id[matched_tgt_global_idx]
        if matched_submap_ids.numel() > 0:
            old_submap_id = int(torch.mode(matched_submap_ids).values.item())

    return ForceFieldResult(
        inliers=n_inliers,
        rmse=rmse,
        translation_norm=t_norm,
        rotation_deg=rot_deg,
        old_submap_id=old_submap_id,
        shear_norm=0.0,
        coherence=coherence,
        T_old_current=T,
    )
