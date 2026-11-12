"""Surface-prior contracts shared by insertion, losses, and fusion."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Protocol

import torch


class GeometrySource(IntEnum):
    UNKNOWN = 0
    SENSOR = 1
    VGGT = 2
    FUSED = 3
    RGB = 4


class SurfaceState(IntEnum):
    CANDIDATE = 0
    PROVISIONAL = 1
    TRUSTED = 2
    REJECTED = 3


@dataclass
class SurfacePrior:
    depth: torch.Tensor
    confidence: torch.Tensor
    valid: torch.Tensor
    normals: torch.Tensor | None
    uncertainty: torch.Tensor | None
    source: torch.Tensor | None
    frame_uid: int
    normal_frame: str = "camera"

    def __post_init__(self) -> None:
        if self.depth.ndim != 3 or self.depth.shape[0] != 1:
            raise ValueError(f"depth must be [1,H,W], got {tuple(self.depth.shape)}")
        if self.confidence.shape != self.depth.shape:
            raise ValueError("confidence must match depth shape")
        if self.valid.shape != self.depth.shape:
            raise ValueError("valid must match depth shape")
        if self.normals is not None and (self.normals.ndim != 3 or self.normals.shape[0] != 3):
            raise ValueError(f"normals must be [3,H,W], got {tuple(self.normals.shape)}")

    @property
    def device(self) -> torch.device:
        return self.depth.device

    @property
    def height(self) -> int:
        return int(self.depth.shape[1])

    @property
    def width(self) -> int:
        return int(self.depth.shape[2])

    def to(self, *args, **kwargs) -> "SurfacePrior":
        return SurfacePrior(
            depth=self.depth.to(*args, **kwargs),
            confidence=self.confidence.to(*args, **kwargs),
            valid=self.valid.to(*args, **kwargs),
            normals=None if self.normals is None else self.normals.to(*args, **kwargs),
            uncertainty=None if self.uncertainty is None else self.uncertainty.to(*args, **kwargs),
            source=None if self.source is None else self.source.to(*args, **kwargs),
            frame_uid=self.frame_uid,
            normal_frame=self.normal_frame,
        )


class GeometryPriorProvider(Protocol):
    def available(self) -> bool:
        ...

    def predict(self, frame, cam=None) -> SurfacePrior:
        ...
