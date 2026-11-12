from __future__ import annotations

from typing import Any

import torch


def depth_lock_enabled(args: Any) -> bool:
    return bool(getattr(args, "streaming_depth_lock_centers", False))


def depth_conflict_enabled(args: Any) -> bool:
    return bool(getattr(args, "streaming_depth_multiview_conflict_enabled", False))


def depth_lock_conflict_threshold(args: Any) -> float:
    value = float(args.streaming_depth_lock_conflict_thresh)
    if value > 0:
        return value
    return float(getattr(args, "streaming_depth_consistency_thresh", 0.05))


def _point_count(gaussians) -> int:
    return int(gaussians.get_xyz.shape[0])


def _resize_1d_buffer(gaussians, name: str, *, fill_value: int, dtype: torch.dtype) -> None:
    count = _point_count(gaussians)
    xyz = gaussians.get_xyz
    tensor = getattr(gaussians, name, None)
    if tensor is None:
        setattr(
            gaussians,
            name,
            torch.full((count,), fill_value, dtype=dtype, device=xyz.device),
        )
        return
    if tensor.shape[0] == count:
        return
    if tensor.shape[0] > count:
        setattr(gaussians, name, tensor[:count].contiguous())
        return
    pad = torch.full(
        (count - tensor.shape[0],),
        fill_value,
        dtype=tensor.dtype,
        device=tensor.device,
    )
    setattr(gaussians, name, torch.cat([tensor, pad], dim=0))


def ensure_depth_lock_buffers(gaussians) -> None:
    _resize_1d_buffer(gaussians, "depth_conflict_count", fill_value=0, dtype=torch.int16)
    _resize_1d_buffer(gaussians, "depth_last_support_uid", fill_value=-1, dtype=torch.int32)
    _resize_1d_buffer(gaussians, "depth_last_conflict_uid", fill_value=-1, dtype=torch.int32)


def depth_lock_candidate_mask(gaussians, args: Any) -> torch.Tensor:
    """Return the per-Gaussian candidate mask for depth-lock center-pinning.

    Also serves as the conflict-tracking population when depth_conflict_enabled
    is True without the full depth-lock flag: in that case all depth-inserted
    Gaussians (birth_frame > 0) are candidates so conflict counts accumulate.
    """
    xyz = gaussians.get_xyz
    mask = torch.zeros((xyz.shape[0],), dtype=torch.bool, device=xyz.device)
    birth_frame = getattr(gaussians, "birth_frame", None)
    if birth_frame is None or birth_frame.shape[0] != xyz.shape[0]:
        return mask

    if depth_lock_enabled(args):
        anchor_xyz = getattr(gaussians, "anchor_xyz", None)
        if anchor_xyz is None or anchor_xyz.shape[0] != xyz.shape[0]:
            return mask
        if bool(getattr(args, "streaming_depth_lock_bootstrap", False)):
            mask |= birth_frame == 0
        if bool(getattr(args, "streaming_depth_lock_insertions", False)):
            mask |= birth_frame > 0
    elif depth_conflict_enabled(args):
        mask |= birth_frame > 0

    return mask


def depth_locked_mask(gaussians, args: Any) -> torch.Tensor:
    """Build the device-resident lock mask without reducing it on the host."""
    mask = depth_lock_candidate_mask(gaussians, args)
    if (
        depth_conflict_enabled(args)
        and not bool(getattr(args, "streaming_depth_lock_prune_conflicts", False))
    ):
        ensure_depth_lock_buffers(gaussians)
        min_views = max(1, int(getattr(args, "streaming_depth_unlock_conflict_views", 2)))
        conflicts = getattr(gaussians, "depth_conflict_count")
        if conflicts.shape[0] == mask.shape[0]:
            mask &= conflicts < min_views
    return mask


def zero_depth_locked_mean_grads(
    gaussians, args: Any, mask: torch.Tensor | None = None
) -> None:
    """Mask mean gradients without any device-to-host scalar dependency."""
    if mask is None or mask.shape[0] != gaussians.get_xyz.shape[0]:
        mask = depth_locked_mask(gaussians, args)
    means = gaussians.means_param
    if means.grad is None:
        return
    if getattr(means.grad, "layout", torch.strided) != torch.strided:
        means.grad = means.grad.to_dense().contiguous()
    # Fixed-shape device operation.  An all-false mask is a natural no-op.
    means.grad.mul_((~mask).to(dtype=means.grad.dtype).unsqueeze(-1))


def restore_depth_locked_centers(
    gaussians, args: Any, mask: torch.Tensor | None = None
) -> None:
    """Restore locked means after Adam without synchronizing for diagnostics."""
    if mask is None or mask.shape[0] != gaussians.get_xyz.shape[0]:
        mask = depth_locked_mask(gaussians, args)
    if not hasattr(gaussians, "anchor_xyz"):
        return
    means = gaussians.means_param
    anchors = gaussians.anchor_xyz
    if anchors.shape[0] != means.shape[0]:
        return
    with torch.no_grad():
        anchor_values = anchors.to(device=means.device, dtype=means.dtype)
        means.data.copy_(torch.where(mask.unsqueeze(-1), anchor_values, means.data))


def record_depth_conflicts(
    gaussians, indices: torch.Tensor, view_uid: int
) -> None:
    """Record one conflict per unique view using fixed-shape device updates."""
    if indices.numel() == 0:
        return
    ensure_depth_lock_buffers(gaussians)
    last = gaussians.depth_last_conflict_uid
    counts = gaussians.depth_conflict_count
    prior_last = last[indices]
    fresh = prior_last != int(view_uid)
    prior_counts = counts[indices]
    max_count = torch.iinfo(counts.dtype).max
    incremented = torch.clamp(
        prior_counts.to(torch.int32) + 1,
        max=max_count,
    ).to(dtype=counts.dtype)
    counts[indices] = torch.where(fresh, incremented, prior_counts)
    uid_values = torch.full_like(prior_last, int(view_uid))
    last[indices] = torch.where(fresh, uid_values, prior_last)


def record_depth_support_views(
    gaussians, indices: torch.Tensor, view_uid: int
) -> None:
    """Record one support vote per unique view without boolean compaction."""
    if indices.numel() == 0:
        return
    ensure_depth_lock_buffers(gaussians)
    last = gaussians.depth_last_support_uid
    prior_last = last[indices]
    fresh = prior_last != int(view_uid)
    uid_values = torch.full_like(prior_last, int(view_uid))
    last[indices] = torch.where(fresh, uid_values, prior_last)


def depth_conflict_prune_mask(gaussians, args: Any) -> torch.Tensor:
    mask = depth_lock_candidate_mask(gaussians, args)
    if (
        not depth_conflict_enabled(args)
        or not bool(getattr(args, "streaming_depth_lock_prune_conflicts", False))
    ):
        return torch.zeros_like(mask)
    ensure_depth_lock_buffers(gaussians)
    min_views = max(1, int(getattr(args, "streaming_depth_unlock_conflict_views", 2)))
    conflicts = gaussians.depth_conflict_count
    if conflicts.shape[0] != mask.shape[0]:
        return torch.zeros_like(mask)
    return mask & (conflicts >= min_views)


# ---------------------------------------------------------------------------
# Part C: frustum-expected-visibility multi-view floater consensus
# ---------------------------------------------------------------------------

_CONSENSUS_BUFFER_NAMES = (
    "seen_count",
    "consensus_support_count",
    "consensus_conflict_count",
    "consensus_last_seen_uid",
)


def floater_consensus_enabled(args: Any) -> bool:
    return bool(getattr(args, "streaming_floater_consensus_enabled", False))


def ensure_consensus_buffers(gaussians) -> None:
    """Allocate / resize the per-Gaussian consensus buffers."""
    _resize_1d_buffer(gaussians, "seen_count", fill_value=0, dtype=torch.int32)
    _resize_1d_buffer(gaussians, "consensus_support_count", fill_value=0, dtype=torch.int32)
    _resize_1d_buffer(gaussians, "consensus_conflict_count", fill_value=0, dtype=torch.int32)
    _resize_1d_buffer(gaussians, "consensus_last_seen_uid", fill_value=-1, dtype=torch.int32)


def record_consensus_views(
    gaussians,
    indices: torch.Tensor,
    support: torch.Tensor,
    conflict: torch.Tensor,
    view_uid: int,
) -> int:
    """Accumulate per-unique-view seen / support / conflict counts."""
    if indices.numel() == 0:
        return 0
    ensure_consensus_buffers(gaussians)
    last = gaussians.consensus_last_seen_uid
    fresh = last[indices] != int(view_uid)
    if not fresh.any():
        return 0
    fresh_idx = indices[fresh]
    seen = gaussians.seen_count
    sup = gaussians.consensus_support_count
    con = gaussians.consensus_conflict_count
    sup_max = torch.iinfo(seen.dtype).max
    seen[fresh_idx] = torch.clamp(seen[fresh_idx] + 1, max=sup_max)
    sup_fresh = support[fresh]
    con_fresh = conflict[fresh]
    if sup_fresh.any():
        si = fresh_idx[sup_fresh]
        sup[si] = torch.clamp(sup[si] + 1, max=sup_max)
    if con_fresh.any():
        ci = fresh_idx[con_fresh]
        con[ci] = torch.clamp(con[ci] + 1, max=sup_max)
    last[fresh_idx] = int(view_uid)
    return int(fresh_idx.numel())


def consensus_floater_mask_from_counts(
    support: torch.Tensor,
    conflict: torch.Tensor,
    seen: torch.Tensor,
    *,
    min_views: int,
    conflict_ratio: float,
) -> torch.Tensor:
    """Pure-tensor floater consensus mask (no model / args dependency)."""
    seen_f = seen.to(torch.float32)
    conflict_f = conflict.to(torch.float32)
    denom = torch.clamp(seen_f, min=1.0)
    ratio = conflict_f / denom
    enough_views = seen.to(torch.int64) >= int(min_views)
    return enough_views & (ratio >= float(conflict_ratio)) & (conflict > support)


def consensus_floater_mask(gaussians, args: Any) -> torch.Tensor:
    """Floater consensus prune mask over the live model (reads model buffers)."""
    n = gaussians.get_xyz.shape[0]
    zeros = torch.zeros((n,), dtype=torch.bool, device=gaussians.get_xyz.device)
    if not floater_consensus_enabled(args):
        return zeros
    if not all(hasattr(gaussians, nm) for nm in _CONSENSUS_BUFFER_NAMES):
        return zeros
    seen = gaussians.seen_count
    support = gaussians.consensus_support_count
    conflict = gaussians.consensus_conflict_count
    if seen.shape[0] != n or support.shape[0] != n or conflict.shape[0] != n:
        return zeros
    min_views = max(1, int(getattr(args, "streaming_floater_consensus_min_views", 3)))
    conflict_ratio = float(getattr(args, "streaming_floater_consensus_conflict_ratio", 0.5))
    return consensus_floater_mask_from_counts(
        support,
        conflict,
        seen,
        min_views=min_views,
        conflict_ratio=conflict_ratio,
    )


def consensus_clean_mask(gaussians, args: Any) -> torch.Tensor:
    """True for Gaussians with no net multi-view conflict."""
    n = gaussians.get_xyz.shape[0]
    ones = torch.ones((n,), dtype=torch.bool, device=gaussians.get_xyz.device)
    if not floater_consensus_enabled(args):
        return ones
    if not all(
        hasattr(gaussians, nm)
        for nm in ("consensus_support_count", "consensus_conflict_count")
    ):
        return ones
    support = gaussians.consensus_support_count
    conflict = gaussians.consensus_conflict_count
    if support.shape[0] != n or conflict.shape[0] != n:
        return ones
    return conflict <= support


def reset_consensus_counts(gaussians, indices: torch.Tensor) -> None:
    """Zero the consensus counts for relocated (revived) Gaussian slots."""
    if indices is None or indices.numel() == 0:
        return
    voter = getattr(gaussians, "consensus_last_voter_center", None)
    if voter is not None and voter.shape[0] == gaussians.get_xyz.shape[0]:
        voter[indices] = float("nan")
    if not all(hasattr(gaussians, nm) for nm in _CONSENSUS_BUFFER_NAMES):
        return
    n = gaussians.get_xyz.shape[0]
    if gaussians.seen_count.shape[0] != n:
        return
    gaussians.seen_count[indices] = 0
    gaussians.consensus_support_count[indices] = 0
    gaussians.consensus_conflict_count[indices] = 0
    gaussians.consensus_last_seen_uid[indices] = -1


# ---------------------------------------------------------------------------
# Part D1: parallax / baseline-gated consensus voting
# ---------------------------------------------------------------------------


def floater_consensus_baseline_gate_enabled(args: Any) -> bool:
    return bool(getattr(args, "streaming_floater_consensus_baseline_gate", False))


def ensure_consensus_voter_buffer(gaussians) -> None:
    """Allocate / resize the per-Gaussian last-voter-center buffer ([N,3] NaN)."""
    count = _point_count(gaussians)
    xyz = gaussians.get_xyz
    voter = getattr(gaussians, "consensus_last_voter_center", None)
    if voter is None:
        gaussians.consensus_last_voter_center = torch.full(
            (count, 3), float("nan"), dtype=torch.float32, device=xyz.device
        )
        return
    if voter.shape[0] == count:
        return
    if voter.shape[0] > count:
        gaussians.consensus_last_voter_center = voter[:count].contiguous()
        return
    pad = torch.full(
        (count - voter.shape[0], 3),
        float("nan"),
        dtype=voter.dtype,
        device=voter.device,
    )
    gaussians.consensus_last_voter_center = torch.cat([voter, pad], dim=0)


def baseline_gate_mask(
    last_voter_centers: torch.Tensor,
    cam_center: torch.Tensor,
    min_baseline: float,
) -> torch.Tensor:
    """Boolean accept-mask for parallax-gated consensus voting."""
    if last_voter_centers.numel() == 0:
        return torch.zeros(
            (last_voter_centers.shape[0],),
            dtype=torch.bool,
            device=last_voter_centers.device,
        )
    is_first = torch.isnan(last_voter_centers).any(dim=1)
    cam = cam_center.to(
        device=last_voter_centers.device, dtype=last_voter_centers.dtype
    ).reshape(1, 3)
    dist = (last_voter_centers - cam).norm(dim=1)
    return is_first | (dist >= float(min_baseline))


def depth_lock_geometry_stats(gaussians, args: Any) -> dict:
    """Cold diagnostic materialization; never call from the optimizer fast path."""
    xyz = gaussians.get_xyz.detach()
    if xyz.numel() == 0:
        return {}
    anchor_xyz = getattr(gaussians, "anchor_xyz", None)
    center_source = (
        anchor_xyz.detach()
        if anchor_xyz is not None and anchor_xyz.shape == xyz.shape
        else xyz
    )
    center = center_source.median(dim=0).values
    radius = (xyz - center).norm(dim=1)
    candidate = depth_lock_candidate_mask(gaussians, args)
    locked = depth_locked_mask(gaussians, args)
    stats = {
        "radius_p95_m": float(torch.quantile(radius.float(), 0.95).item()),
        "radius_p99_m": float(torch.quantile(radius.float(), 0.99).item()),
        "radius_max_m": float(radius.max().item()),
        "count_gt_5m": int((radius > 5.0).sum().item()),
        "count_gt_10m": int((radius > 10.0).sum().item()),
        "count_gt_50m": int((radius > 50.0).sum().item()),
        "count_gt_100m": int((radius > 100.0).sum().item()),
        "candidate_count": int(candidate.sum().item()),
        "locked_count": int(locked.sum().item()),
    }
    if anchor_xyz is not None and anchor_xyz.shape == xyz.shape and candidate.any():
        drift = (xyz[candidate] - anchor_xyz.detach()[candidate]).norm(dim=1)
        stats["candidate_drift_max_m"] = float(drift.max().item())
        stats["candidate_drift_mean_m"] = float(drift.mean().item())
        stats["candidate_drift_gt_5cm"] = int((drift > 0.05).sum().item())
        stats["candidate_drift_gt_1m"] = int((drift > 1.0).sum().item())
    if (
        hasattr(gaussians, "depth_conflict_count")
        and gaussians.depth_conflict_count.shape[0] == xyz.shape[0]
    ):
        conflicts = gaussians.depth_conflict_count
        stats["conflict_count_nonzero"] = int((conflicts > 0).sum().item())
        stats["conflict_count_max"] = int(conflicts.max().item()) if conflicts.numel() else 0
    return stats
