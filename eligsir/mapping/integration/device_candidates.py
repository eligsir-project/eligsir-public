"""GPU-native insertion candidate normalization and filtering.

Legacy NumPy producers cross the boundary through one aligned heterogeneous
pinned byte slab. ``SurfaceCandidatesT`` producers do not cross CPU at all.
Pinned staging slots remain leased until their async H2D event completes, so a
later frame cannot overwrite memory still owned by DMA.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import torch

from geometry.insertion import SurfaceCandidatesT
from eligsir.map.mutation import GaussianInsertionBatch
from eligsir.runtime.transfers import DeviceTransferOwner, get_device_transfer_owner


@dataclass(frozen=True)
class DeviceCandidates:
    """Aligned insertion candidates resident on one target device."""

    pts: torch.Tensor
    colors: torch.Tensor
    log_scales: torch.Tensor
    quats: torch.Tensor
    normals: torch.Tensor
    confidence: torch.Tensor
    source: torch.Tensor
    birth_frame: int
    init_opacity: float
    opacities_raw: Optional[torch.Tensor] = None

    def __len__(self) -> int:
        """Return the number of aligned candidate rows."""
        return int(self.pts.shape[0])

    def select(self, index: torch.Tensor) -> "DeviceCandidates":
        """Return an aligned device-native subset without leaving the device."""
        return DeviceCandidates(
            pts=self.pts[index],
            colors=self.colors[index],
            log_scales=self.log_scales[index],
            quats=self.quats[index],
            normals=self.normals[index],
            confidence=self.confidence[index],
            source=self.source[index],
            birth_frame=self.birth_frame,
            init_opacity=self.init_opacity,
            opacities_raw=(None if self.opacities_raw is None else self.opacities_raw[index]),
        )

    def as_insertion_batch(self, *, iteration: int) -> GaussianInsertionBatch:
        """Convert this bundle to the typed mutation insertion contract."""
        return GaussianInsertionBatch(
            points=self.pts,
            colors=self.colors,
            scales=self.log_scales,
            rotations=self.quats,
            normals=self.normals,
            confidence=self.confidence,
            source=self.source,
            opacities_raw=self.opacities_raw,
            init_opacity=self.init_opacity,
            is_provisional=True,
            birth_frame=self.birth_frame,
            birth_iter=int(iteration),
        )


def transfer_for_device(device: torch.device | str):
    """Return the reusable insertion transfer stream for ``device``."""
    return get_device_transfer_owner(device).transfer_stream


def insertion_transfer_stats(device: torch.device | str) -> dict[str, int]:
    """Return cumulative insertion-boundary transfer counters for ``device``."""
    return get_device_transfer_owner(device).stats.snapshot()


def _packed_legacy_pending(
    pending: Sequence[dict],
    device: torch.device,
    transfer_owner: Optional[DeviceTransferOwner] = None,
) -> DeviceCandidates:
    """Pack legacy NumPy candidate fields into one heterogeneous H2D slab."""
    all_pts = np.concatenate([np.asarray(p["pts"], dtype=np.float32) for p in pending], axis=0)
    n = int(all_pts.shape[0])
    all_cols = np.concatenate([np.asarray(p["cols"], dtype=np.float32) for p in pending], axis=0)
    all_scales = np.concatenate([np.asarray(p["log_scales"], dtype=np.float32) for p in pending], axis=0)
    all_quats = np.concatenate([np.asarray(p["q_world"], dtype=np.float32) for p in pending], axis=0)
    all_normals = np.concatenate(
        [
            np.asarray(
                p.get("normals", np.zeros((p["pts"].shape[0], 3), np.float32)),
                dtype=np.float32,
            )
            for p in pending
        ],
        axis=0,
    )
    all_confidence = np.concatenate(
        [
            np.asarray(
                p.get("confidence", np.zeros((p["pts"].shape[0],), np.float32)),
                dtype=np.float32,
            )
            for p in pending
        ],
        axis=0,
    ).reshape(n, 1)
    has_opacity = all(p.get("opacities_raw") is not None for p in pending)
    float_fields = [all_pts, all_cols, all_scales, all_quats, all_normals, all_confidence]
    if has_opacity:
        float_fields.append(
            np.concatenate(
                [
                    np.asarray(p["opacities_raw"], dtype=np.float32).reshape(-1, 1)
                    for p in pending
                ],
                axis=0,
            )
        )
    float_slab = np.concatenate(float_fields, axis=1).astype(np.float32, copy=False)
    int_slab = np.concatenate(
        [
            np.asarray(
                p.get("source", np.zeros((p["pts"].shape[0],), np.int16)),
                dtype=np.int16,
            )
            for p in pending
        ],
        axis=0,
    ).reshape(n, 1)

    owner = transfer_owner or get_device_transfer_owner(device)
    (float_dev, int_dev), _ = owner.upload_packed(
        (float_slab, int_slab),
        purpose="cpu_origin_insertion",
    )
    assert float_dev is not None and int_dev is not None

    cursor = 0

    def take(width: int):
        """Take the next aligned float field from the packed device slab."""
        nonlocal cursor
        out = float_dev[:, cursor : cursor + width]
        cursor += width
        return out

    pts = take(3)
    colors = take(3)
    scales = take(3)
    quats = take(4)
    normals = take(3)
    confidence = take(1).squeeze(1)
    opacity = take(1).squeeze(1) if has_opacity else None
    return DeviceCandidates(
        pts=pts,
        colors=colors,
        log_scales=scales,
        quats=quats,
        normals=normals,
        confidence=confidence,
        source=int_dev[:, 0],
        birth_frame=int(pending[len(pending) // 2]["birth_frame"]),
        init_opacity=float(pending[0]["init_opacity"]),
        opacities_raw=opacity,
    )


def _cat_torch_pending(
    pending: Sequence[SurfaceCandidatesT], device: torch.device, *, init_opacity: float
) -> DeviceCandidates:
    """Concatenate already-device-native candidates without hidden migration."""
    fields = ("pts", "colors", "log_scales", "quats", "normals", "confidence", "source")
    for candidate in pending:
        for name in fields:
            tensor = getattr(candidate, name)
            if tensor.device != device:
                raise ValueError(
                    f"SurfaceCandidatesT.{name} is on {tensor.device}, expected {device}; "
                    "GPU-native candidates may not silently migrate here"
                )

    if len(pending) == 1:
        p = pending[0]
        values = {name: getattr(p, name) for name in fields}
    else:
        values = {
            name: torch.cat([getattr(p, name) for p in pending], dim=0)
            for name in fields
        }
    return DeviceCandidates(
        pts=values["pts"],
        colors=values["colors"],
        log_scales=values["log_scales"],
        quats=values["quats"],
        normals=values["normals"],
        confidence=values["confidence"],
        source=values["source"],
        birth_frame=int(pending[len(pending) // 2].birth_frame),
        init_opacity=float(init_opacity),
        opacities_raw=None,
    )


def normalize_pending_candidates(
    pending: Sequence,
    *,
    device: torch.device | str,
    init_opacity: float,
    transfer_owner: Optional[DeviceTransferOwner] = None,
) -> DeviceCandidates:
    """Normalize one homogeneous pending sequence to a device-native bundle."""
    if not pending:
        raise ValueError("pending candidates must be non-empty")
    target = torch.device(device)
    if all(isinstance(p, SurfaceCandidatesT) for p in pending):
        return _cat_torch_pending(pending, target, init_opacity=init_opacity)
    if any(isinstance(p, SurfaceCandidatesT) for p in pending):
        raise TypeError("do not mix SurfaceCandidatesT and legacy NumPy candidates in one commit")
    return _packed_legacy_pending(pending, target, transfer_owner)


def collected_candidate_points(collected):
    """Return the point tensor of one ``_collect_only`` result, or ``None``.

    ``insert_gaussians_from_frame(_collect_only=True)`` returns a legacy NumPy
    candidate ``dict`` on the compatibility path, a :class:`SurfaceCandidatesT`
    on the device-native path, and an empty ``dict`` when every point was
    filtered out. Callers that accumulate candidates across frames must accept
    all three; reading ``["pts"]`` (or gating on ``isinstance(..., dict)``)
    silently drops device-native candidate sets instead of committing them.
    """
    if isinstance(collected, SurfaceCandidatesT):
        return collected.pts
    if isinstance(collected, dict):
        return collected.get("pts")
    return None


def deterministic_voxel_first_indices(points: torch.Tensor, voxel_size: float) -> torch.Tensor:
    """Return stable first-occurrence indices for voxel deduplication on-device."""
    n = int(points.shape[0])
    if n <= 1 or voxel_size <= 0:
        return torch.arange(n, device=points.device, dtype=torch.long)
    keys = torch.floor(points / float(voxel_size)).to(torch.int64)
    order = torch.arange(n, device=points.device, dtype=torch.long)
    for dim in (2, 1, 0):
        perm = torch.argsort(keys[order, dim], stable=True)
        order = order[perm]
    sorted_keys = keys[order]
    first = torch.ones(n, dtype=torch.bool, device=points.device)
    first[1:] = torch.any(sorted_keys[1:] != sorted_keys[:-1], dim=1)
    return torch.sort(order[first]).values


def deterministic_subsample_indices(
    n: int, count: int, *, iteration: int, device: torch.device
) -> torch.Tensor:
    """Return a reproducible on-device random subset keyed by iteration."""
    if count >= n:
        return torch.arange(n, device=device, dtype=torch.long)
    generator = torch.Generator(device=device)
    generator.manual_seed(int(iteration))
    return torch.randperm(n, generator=generator, device=device)[:count]
