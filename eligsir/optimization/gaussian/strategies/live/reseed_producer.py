"""Real Phase C reseed candidate producer (W3-47, 2026-08-10).

Closes the gap root-caused in
docs/architecture/2026-08-09-w3-45-phase-c-reseed-root-cause.md: Phase C
reseeding was inert because no production code path anywhere in this
codebase ever constructed a :class:`SurfaceSeedRequest`. This module is
that producer.

**Signal**: a pixel with valid sensor depth (real, observed surface) but
low rendered alpha (few/no live Gaussians are covering it there) is an
under-covered-surface candidate. Such pixels are backprojected to
world-space points and aggregated per voxel cell into
:class:`SurfaceSeedRequest` objects, ranked by total under-coverage mass
in that cell.

**Scope, stated explicitly** (full design rationale in
docs/architecture/2026-08-10-w3-47-phase-c-reseed-producer.md): this closes
two of W3-45's three named gap layers -- per-cell evidence decomposition,
and wiring that evidence through to the ``propose()`` call. The third layer
named there, ``LivePopulationState.source_surface_cell`` assignment, is
intentionally NOT implemented here: confirmed via a repo-wide grep that
this field is never *read* by any decision logic anywhere in the codebase
today (only written/reshaped as bookkeeping in ``live/state.py``), so it is
not required for Phase C to functionally reseed -- ``AlignedBufferRegistry
.apply_reseed`` (see ``live/backend.py``) reuses *any* currently-RETIRED
row, not specifically one that originated from the same cell. Assigning it
a real value remains a legitimate future refinement (routing a reseed
toward the cell a RETIRED Gaussian's capacity originally came from) rather
than a blocking gap for this task.
"""

from __future__ import annotations

from typing import Optional

import torch

from eligsir.optimization.gaussian.strategies.contracts import SurfaceSeedRequest

# Mirrors scene/streaming_scene.py's StreamingScene._xyz_to_keys exactly
# (same offset/formula) so cell ids computed here use the same convention,
# in case a future unification wants to cross-reference them against that
# occupancy hash -- not a hard dependency (this module does not import
# StreamingScene at all), just intentionally the same scheme.
_CELL_HASH_OFFSET = 1000.0


def _xyz_to_cell_ids(xyz: torch.Tensor, voxel_size: float) -> torch.Tensor:
    """World-space ``[N, 3]`` positions -> one int64 cell id per voxel."""
    coords = torch.floor((xyz + _CELL_HASH_OFFSET) / max(voxel_size, 1e-6)).to(torch.int64)
    return (coords[:, 0] << 42) | (coords[:, 1] << 21) | coords[:, 2]


def derive_reseed_candidates(
    *,
    gt_depth: Optional[torch.Tensor],
    alpha: Optional[torch.Tensor],
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    c2w: torch.Tensor,
    voxel_size: float = 0.05,
    alpha_threshold: float = 0.3,
    min_depth: float = 0.1,
    max_depth: float = 10.0,
    max_candidates: int = 64,
    reason: str = "under_covered_surface",
) -> tuple[SurfaceSeedRequest, ...]:
    """Real reseed candidates from one view's depth/alpha, or ``()``.

    ``gt_depth``/``alpha`` are ``[1, H, W]`` or ``[H, W]`` tensors (any
    device). ``c2w`` is a ``[4, 4]`` camera-to-world transform in the same
    (position-in-rows-3, standard column-vector) convention as
    :func:`utils.rgbd_frames.backproject_depth_pixels_torch` expects --
    callers deriving this from a :class:`scene.cameras.Camera` should use
    ``camera.world_view_transform.inverse().transpose(0, 1)`` (matches how
    that class derives ``camera_center`` from the same inverse).

    Returns ``()`` immediately (no tensor work beyond a couple of None/
    shape checks) whenever depth or alpha is unavailable, or no pixel
    clears the under-coverage threshold -- callers should only invoke this
    when Phase C reseeding is actually enabled, matching this codebase's
    demand-driven "near-zero-cost when unused" evidence convention (see
    ``eligsir/mapping/evidence/builder.py``'s module docstring).
    """
    if gt_depth is None or alpha is None:
        return ()
    if gt_depth.dim() > 2:
        gt_depth = gt_depth.reshape(gt_depth.shape[-2], gt_depth.shape[-1])
    if alpha.dim() > 2:
        alpha = alpha.reshape(alpha.shape[-2], alpha.shape[-1])
    if gt_depth.shape != alpha.shape:
        return ()

    device = gt_depth.device
    alpha = alpha.to(device=device, dtype=torch.float32)
    gt_depth = gt_depth.to(device=device, dtype=torch.float32)
    valid = (gt_depth > min_depth) & (gt_depth < max_depth) & (alpha < alpha_threshold)
    if not bool(valid.any()):
        return ()

    ys, xs = torch.nonzero(valid, as_tuple=True)
    if ys.numel() == 0:
        return ()
    z_v = gt_depth[ys, xs]
    xs_f = xs.to(torch.float32)
    ys_f = ys.to(torch.float32)

    from eligsir.utils.rgbd_frames import backproject_depth_pixels_torch

    # rgb is an output this producer doesn't need (no color for a seed
    # request) -- a minimal dummy tensor satisfies the shared helper's
    # signature without duplicating its backprojection algebra here.
    dummy_rgb = torch.zeros((1, 1, 3), device=device, dtype=torch.float32)
    pts_world, _ = backproject_depth_pixels_torch(
        z_v, xs_f, ys_f, fx, fy, cx, cy, c2w,
        dummy_rgb, (int(gt_depth.shape[0]), int(gt_depth.shape[1])),
    )

    cell_ids = _xyz_to_cell_ids(pts_world, voxel_size)
    under_coverage = (alpha_threshold - alpha[ys, xs]).clamp(min=0.0)

    unique_cells, inverse = torch.unique(cell_ids, return_inverse=True)
    n_cells = int(unique_cells.numel())
    sums = torch.zeros(n_cells, 3, device=device, dtype=torch.float32)
    counts = torch.zeros(n_cells, device=device, dtype=torch.float32)
    utility_sums = torch.zeros(n_cells, device=device, dtype=torch.float32)
    sums.index_add_(0, inverse, pts_world)
    counts.index_add_(0, inverse, torch.ones_like(inverse, dtype=torch.float32))
    utility_sums.index_add_(0, inverse, under_coverage)
    centroids = sums / counts.clamp(min=1.0).unsqueeze(1)

    k = min(max_candidates, n_cells)
    if k <= 0:
        return ()
    top_utility, top_idx = torch.topk(utility_sums, k)

    requests = []
    for i in range(k):
        idx = int(top_idx[i].item())
        requests.append(
            SurfaceSeedRequest(
                cell_id=int(unique_cells[idx].item()),
                utility=float(top_utility[i].item()),
                reason=reason,
                position=centroids[idx].detach().cpu(),
                normal=None,
                source_gaussian_index=None,
            )
        )
    return tuple(requests)
