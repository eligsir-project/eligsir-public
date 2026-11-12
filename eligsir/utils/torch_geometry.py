"""GPU/torch ports of the streaming voxel-downsample and occupancy-hash helpers.

These reproduce the numpy reference implementations bit-exactly so they can be
swapped in behind the ``--gpu_occupancy`` flag without changing results:

  * ``voxel_downsample_indices_torch`` mirrors ``_voxel_downsample_indices`` in
    ``train_streaming.py`` (floor-bucketed cell keys, np.unique(return_index=True)
    FIRST-occurrence semantics).
  * ``pack_voxel_keys_torch`` mirrors ``StreamingScene._xyz_to_keys`` (offset
    1000.0, packed ``(c0<<42)|(c1<<21)|c2`` int64 keys).

Everything stays on the input tensor's device (CPU or CUDA) and avoids the
per-call ``.cpu().numpy()`` round-trips of the numpy versions.
"""
from __future__ import annotations

import torch


def voxel_downsample_indices_torch(points: torch.Tensor, voxel_size: float):
    """GPU/CPU port of ``_voxel_downsample_indices``.

    Keeps one representative point per voxel cell. Returns
    ``(points_ds, first_idx)`` where ``first_idx`` is a LongTensor of the FIRST
    occurrence index of each surviving cell, ordered by ascending cell key
    (exactly like ``np.unique(keys, return_index=True)``).

    Parity contract with numpy reference:
      * Surviving point SET is identical (one point per occupied cell).
      * ``first_idx`` is the index of the first row (lowest original index)
        whose cell key equals a given unique key, and the returned order is by
        ascending key value -- matching np.unique's sorted-unique ordering.

    Tie-break note: when several input rows share a cell key, both numpy and
    this implementation select the LOWEST original row index. numpy guarantees
    this because ``np.unique(return_index=True)`` uses a stable sort; here we
    take the min original index per unique group via ``scatter_reduce_('amin')``.
    """
    n = points.shape[0]
    if n == 0:
        first = torch.empty(0, dtype=torch.long, device=points.device)
        return points, first

    vs = max(voxel_size, 1e-6)
    mins = points.min(dim=0).values
    cell = torch.floor((points - mins) / vs).to(torch.int64)
    maxc = cell.max(dim=0).values + 1
    stride = torch.tensor(
        [maxc[1] * maxc[2], maxc[2], 1],
        dtype=torch.int64,
        device=points.device,
    )
    # ROCm does not implement int64 addmv/matmul. Elementwise integer
    # arithmetic has the same mixed-radix key semantics on every backend.
    keys = cell[:, 0] * stride[0] + cell[:, 1] * stride[1] + cell[:, 2]

    # uniq is the ascending-sorted unique key set (matches np.unique ordering).
    # inverse[i] gives the index into uniq for row i.
    uniq, inverse = torch.unique(keys, sorted=True, return_inverse=True)

    arange = torch.arange(n, device=points.device, dtype=torch.int64)
    first = torch.full(
        (uniq.shape[0],), n, dtype=torch.int64, device=points.device
    )
    # First-occurrence == minimum original index per unique group.
    first.scatter_reduce_(0, inverse, arange, reduce="amin", include_self=True)

    return points[first], first


def pack_voxel_keys_torch(
    xyz: torch.Tensor, voxel_size: float, offset: float = 1000.0
) -> torch.Tensor:
    """GPU/CPU port of ``StreamingScene._xyz_to_keys`` (bit-identical).

    coords = floor((xyz + offset) / max(voxel_size, 1e-6)).int64
    key    = (c0 << 42) | (c1 << 21) | c2
    """
    vs = max(voxel_size, 1e-6)
    coords = torch.floor((xyz + offset) / vs).to(torch.int64)
    c0 = coords[:, 0]
    c1 = coords[:, 1]
    c2 = coords[:, 2]
    return (c0 << 42) | (c1 << 21) | c2
