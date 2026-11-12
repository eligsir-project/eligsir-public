"""Device-native batched geometry operations for insertion.

This module replaces the old ``GPU KNN -> CPU indices -> Python/NumPy PCA``
shape with one tensor pipeline.  It is backend-portable across CUDA/ROCm.
"""

from __future__ import annotations

import torch

from geometry.pca_degeneracy import EIGENVALUE_RATIO_EPS


def fuse_points_with_knn_torch(
    queries: torch.Tensor,
    pool: torch.Tensor,
    nn_idx: torch.Tensor,
    normal_hints: torch.Tensor,
    *,
    max_radius: float | None = None,
    return_diagnostics: bool = False,
) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Batched PCA surface fusion without CPU/index round-trips.

    Args mirror ``geometry.insertion.fuse_points_with_knn`` but all inputs
    remain tensors on one device.  ``nn_idx`` is consumed directly where the
    KNN kernel produced it.  Eigen decomposition is batched over ``[N,3,3]``.

    A rank-deficient neighbourhood (too few points, or points collapsed onto
    ~2 distinct locations / near-collinear — issue #1149) falls back to
    ``normal_hints[i]``, using the SAME guard as
    ``geometry.insertion.fuse_points_with_knn`` (see
    ``geometry.pca_degeneracy``) so the two implementations cannot
    independently drift on this threshold again.

    ``return_diagnostics=True`` additionally returns a ``[N]`` bool tensor
    (``well_conditioned``) — ``False`` where the query fell back to its hint.
    Default ``False`` keeps the historical 2-tuple return for existing
    callers.
    """
    if queries.ndim != 2 or queries.shape[-1] != 3:
        raise ValueError("queries must be [N,3]")
    if pool.ndim != 2 or pool.shape[-1] != 3:
        raise ValueError("pool must be [M,3]")
    if nn_idx.ndim != 2 or nn_idx.shape[0] != queries.shape[0]:
        raise ValueError("nn_idx must be [N,K]")
    if normal_hints.shape != queries.shape:
        raise ValueError("normal_hints must be [N,3]")
    device = queries.device
    if pool.device != device or nn_idx.device != device or normal_hints.device != device:
        raise ValueError("all fusion inputs must share one device")

    neighbors = pool[nn_idx.long()]  # [N,K,3], no D2H of indices
    n, k, _ = neighbors.shape
    query = queries[:, None, :]
    cloud = torch.cat((query, neighbors), dim=1)  # [N,K+1,3]

    if max_radius is None:
        valid = torch.ones((n, k + 1), dtype=torch.bool, device=device)
    else:
        neighbor_valid = torch.linalg.vector_norm(neighbors - query, dim=-1) <= float(max_radius)
        valid = torch.cat(
            (torch.ones((n, 1), dtype=torch.bool, device=device), neighbor_valid), dim=1
        )

    weights = valid.to(dtype=queries.dtype)
    count = weights.sum(dim=1, keepdim=True).clamp_min(1.0)
    fused = (cloud * weights[..., None]).sum(dim=1) / count

    centered = cloud - fused[:, None, :]
    centered = centered * weights[..., None]
    cov = centered.transpose(1, 2) @ centered
    denom = (count - 1.0).clamp_min(1.0).view(n, 1, 1)
    cov = cov / denom
    eigvals, eigvecs = torch.linalg.eigh(cov)  # eigvals ascending: [...,e0<=e1<=e2]
    normals = eigvecs[..., 0]

    # Degeneracy guard (issue #1149): mirrors
    # geometry.pca_degeneracy.eigvals_well_conditioned exactly (same formula,
    # same threshold) so this batched torch path and the numpy per-point path
    # agree on which neighbourhoods are trustworthy, even though — by design —
    # they may still each independently resolve a *genuinely* degenerate case
    # via the same hint-fallback rather than needing to agree bit-for-bit on
    # an eigensolver tie-break. e0 is expected to be small for a healthy flat
    # patch (that's the point of PCA normal estimation), so the guard checks
    # e1/e2 (NOT e0), which collapses toward zero only when the neighbourhood
    # is rank-deficient (< 2 genuinely independent directions).
    e0 = eigvals[..., 0]
    e1 = eigvals[..., 1]
    e2 = eigvals[..., 2].clamp_min(0.0)
    ratio = e1 / e2.clamp_min(torch.finfo(eigvals.dtype).tiny)
    well_conditioned = (e2 > 0) & (ratio >= EIGENVALUE_RATIO_EPS)
    del e0  # not used directly; kept named for readability against the docstring

    hints = torch.nn.functional.normalize(normal_hints, dim=-1, eps=1e-9)
    enough = count[:, 0] >= 3
    keep_pca = enough & well_conditioned
    normals = torch.nn.functional.normalize(normals, dim=-1, eps=1e-9)
    normals = torch.where(keep_pca[:, None], normals, hints)
    flip = (normals * hints).sum(dim=-1) < 0
    normals = torch.where(flip[:, None], -normals, normals)
    fused_out = fused.to(torch.float32)
    normals_out = normals.to(torch.float32)
    if return_diagnostics:
        return fused_out, normals_out, keep_pca
    return fused_out, normals_out


def knn_isotropic_log_scale_torch(
    points: torch.Tensor,
    squared_distance: torch.Tensor,
    *,
    min_scale: float = 1e-4,
    max_scale: float = 0.05,
) -> torch.Tensor:
    """Convert device-native KNN squared distance to [N,3] log scales."""
    if squared_distance.device != points.device:
        raise ValueError("KNN distance and points must share one device")
    scale = squared_distance.clamp_min(0).sqrt().clamp(float(min_scale), float(max_scale))
    if scale.ndim > 1:
        scale = scale.reshape(scale.shape[0], -1).mean(dim=1)
    return torch.log(scale)[:, None].expand(-1, 3).contiguous()


__all__ = ["fuse_points_with_knn_torch", "knn_isotropic_log_scale_torch"]
