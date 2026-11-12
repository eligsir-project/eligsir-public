"""Surface-cell index for sparse surface-aware deduplication."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SurfaceCellStats:
    gaussian_count: int = 0
    opacity_mass: float = 0.0
    mean_normal: np.ndarray | None = None
    mean_color: np.ndarray | None = None
    residual_ema: float = 0.0
    coverage_ema: float = 0.0


def normal_bin(normals: np.ndarray, bins: int = 8) -> np.ndarray:
    n = np.asarray(normals, dtype=np.float32)
    n = n / (np.linalg.norm(n, axis=-1, keepdims=True) + 1e-8)
    theta = np.arctan2(n[:, 1], n[:, 0])
    z = np.clip(n[:, 2], -1.0, 1.0)
    az = np.floor((theta + np.pi) / (2 * np.pi) * bins).astype(np.int64)
    el = np.floor((z + 1.0) * 0.5 * bins).astype(np.int64)
    return np.clip(az, 0, bins - 1) * bins + np.clip(el, 0, bins - 1)


def surface_cell_keys(
    xyz: np.ndarray,
    normals: np.ndarray,
    scales: np.ndarray | None = None,
    *,
    voxel_size: float = 0.02,
    normal_bins: int = 8,
) -> np.ndarray:
    pts = np.asarray(xyz, dtype=np.float32)
    coords = np.floor((pts + 1000.0) / max(voxel_size, 1e-6)).astype(np.int64)
    nb = normal_bin(normals, normal_bins)
    if scales is None:
        scale_level = np.zeros(pts.shape[0], dtype=np.int64)
    else:
        s = np.asarray(scales, dtype=np.float32)
        smax = np.max(s, axis=-1) if s.ndim == 2 else s
        scale_level = np.clip(np.floor(np.log2(np.maximum(smax, 1e-6) / max(voxel_size, 1e-6)) + 8), 0, 15).astype(np.int64)
    return (coords[:, 0] << 44) ^ (coords[:, 1] << 24) ^ (coords[:, 2] << 8) ^ (nb << 4) ^ scale_level


class SurfaceCellIndex:
    def __init__(self, *, voxel_size: float = 0.02, normal_bins: int = 8) -> None:
        self.voxel_size = float(voxel_size)
        self.normal_bins = int(normal_bins)
        self._keys = np.empty(0, dtype=np.int64)

    def rebuild(self, xyz: np.ndarray, normals: np.ndarray, scales: np.ndarray | None = None) -> None:
        if len(xyz) == 0:
            self._keys = np.empty(0, dtype=np.int64)
        else:
            self._keys = np.unique(surface_cell_keys(xyz, normals, scales, voxel_size=self.voxel_size, normal_bins=self.normal_bins))

    def occupied(self, xyz: np.ndarray, normals: np.ndarray, scales: np.ndarray | None = None) -> np.ndarray:
        if self._keys.size == 0 or len(xyz) == 0:
            return np.zeros(len(xyz), dtype=bool)
        keys = surface_cell_keys(xyz, normals, scales, voxel_size=self.voxel_size, normal_bins=self.normal_bins)
        return np.isin(keys, self._keys)

    def add(self, xyz: np.ndarray, normals: np.ndarray, scales: np.ndarray | None = None) -> None:
        keys = surface_cell_keys(xyz, normals, scales, voxel_size=self.voxel_size, normal_bins=self.normal_bins)
        self._keys = np.union1d(self._keys, keys)

    @property
    def count(self) -> int:
        return int(self._keys.size)
