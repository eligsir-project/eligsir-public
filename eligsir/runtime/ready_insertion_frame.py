"""Minimal frame shim for READY-path device-native depth insertion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class _ReadyInsertionMappingFrame:
    source: Any
    intrinsics: Any


@dataclass(frozen=True, slots=True)
class ReadyInsertionFrame:
    """Expose lease-host metadata for device-native depth insertion."""

    mapping_frame: _ReadyInsertionMappingFrame
    source: Any
    lease: Any
    c2w: Any
    height: int
    width: int
    fx: float
    fy: float
    cx: float
    cy: float
    depth_fx: float
    depth_fy: float
    depth_cx: float
    depth_cy: float
    depth_height: int
    depth_width: int
    _depth_np: Any = None

    @classmethod
    def from_binding(cls, binding: Any) -> "ReadyInsertionFrame":
        """Create frame from a binding with metadata and pose information."""
        intrinsics = binding.metadata.intrinsics
        source = binding.source_identity
        height, width = binding.metadata.depth_shape
        return cls(
            mapping_frame=_ReadyInsertionMappingFrame(source, intrinsics),
            source=source,
            lease=binding.lease,
            c2w=binding.metadata.pose_c2w,
            height=int(height),
            width=int(width),
            **{
                name: float(getattr(intrinsics, name))
                for name in (
                    "fx", "fy", "cx", "cy", "depth_fx", "depth_fy", "depth_cx", "depth_cy"
                )
            },
            depth_height=int(intrinsics.depth_height),
            depth_width=int(intrinsics.depth_width),
        )


__all__ = ["ReadyInsertionFrame"]
