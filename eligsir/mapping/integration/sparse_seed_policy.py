"""Sparse reliable source-depth seeding policy.

This policy filters already-device-native surface candidates and leaves later
mature density allocation to MCMC.  It never mutates the map directly.
"""
from __future__ import annotations

from dataclasses import dataclass
import torch

from geometry.insertion import SurfaceCandidatesT
from eligsir.mapping.integration.device_candidates import deterministic_voxel_first_indices


@dataclass(frozen=True, slots=True)
class SparseSeedConfig:
    min_confidence: float = 0.5
    max_abs_normal_z: float = 1.0
    min_abs_normal_z: float = 0.0
    voxel_size_m: float = 0.02
    max_seeds_per_frame: int = 1000
    reject_zero_normals: bool = True


@dataclass(frozen=True, slots=True)
class SparseSeedStats:
    input_count: int
    reliable_count: int
    voxel_count: int
    output_count: int


def _select(c: SurfaceCandidatesT, idx: torch.Tensor) -> SurfaceCandidatesT:
    return SurfaceCandidatesT(
        pts=c.pts[idx],
        colors=c.colors[idx],
        normals=c.normals[idx],
        log_scales=c.log_scales[idx],
        quats=c.quats[idx],
        confidence=c.confidence[idx],
        source=c.source[idx],
        score=c.score[idx],
        birth_frame=int(c.birth_frame),
    )


def select_sparse_reliable_seeds(
    candidates: SurfaceCandidatesT,
    config: SparseSeedConfig | None = None,
) -> tuple[SurfaceCandidatesT, SparseSeedStats]:
    """Filter and deterministically cap source-born geometry on-device."""
    cfg = config or SparseSeedConfig()
    n = len(candidates)
    if n == 0:
        return candidates, SparseSeedStats(0, 0, 0, 0)
    device = candidates.pts.device
    finite = (
        torch.isfinite(candidates.pts).all(dim=1)
        & torch.isfinite(candidates.colors).all(dim=1)
        & torch.isfinite(candidates.log_scales).all(dim=1)
        & torch.isfinite(candidates.quats).all(dim=1)
        & torch.isfinite(candidates.normals).all(dim=1)
        & torch.isfinite(candidates.confidence)
    )
    normal_norm = torch.linalg.vector_norm(candidates.normals, dim=1)
    reliable = finite & (candidates.confidence >= float(cfg.min_confidence))
    if cfg.reject_zero_normals:
        reliable &= normal_norm > 1e-6
    nz = candidates.normals[:, 2].abs() / normal_norm.clamp_min(1e-6)
    reliable &= nz >= float(cfg.min_abs_normal_z)
    reliable &= nz <= float(cfg.max_abs_normal_z)
    reliable_idx = torch.nonzero(reliable, as_tuple=False).flatten()
    filtered = _select(candidates, reliable_idx)
    reliable_count = len(filtered)

    if reliable_count and cfg.voxel_size_m > 0:
        voxel_idx = deterministic_voxel_first_indices(filtered.pts, float(cfg.voxel_size_m))
        filtered = _select(filtered, voxel_idx)
    voxel_count = len(filtered)

    cap = max(0, int(cfg.max_seeds_per_frame))
    if cap and len(filtered) > cap:
        # deterministic quality-first cap; stable source order breaks ties.
        order = torch.argsort(filtered.score, descending=True, stable=True)[:cap]
        order = torch.sort(order).values
        filtered = _select(filtered, order)
    elif cap == 0:
        empty = torch.empty((0,), dtype=torch.long, device=device)
        filtered = _select(filtered, empty)
    return filtered, SparseSeedStats(n, reliable_count, voxel_count, len(filtered))
