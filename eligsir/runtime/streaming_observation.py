"""Mapper-local mutable view over the canonical :class:`MappingFrame`.

This is not a source/transport contract. ROS, datasets, replay and simulation all
end at ``SensorPacket -> MappingFrame`` first. The streaming mapper still has
bookkeeping code that annotates the current observation while deciding holdout,
admission and insertion. ``StreamingObservation`` provides that mutable local
surface without reviving ``StreamingRGBDFrame`` as a production ingress type.

Pixel payloads are views of the already-materialized MappingFrame buffers; this
projection performs no image/depth copy or unit conversion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from eligsir.contracts.runtime import MappingFrame


@dataclass
class StreamingObservation:
    """Application-local observation consumed by the existing streaming mapper."""

    mapping_frame: MappingFrame = field(repr=False)
    index: int
    timestamp: float
    rgb_path: str = ""
    depth_path: Optional[str] = None
    c2w: Optional[np.ndarray] = field(default=None, repr=False)
    fx: float = 0.0
    fy: float = 0.0
    cx: float = 0.0
    cy: float = 0.0
    width: int = 0
    height: int = 0
    depth_scale: float = 1.0
    depth_fx: Optional[float] = None
    depth_fy: Optional[float] = None
    depth_cx: Optional[float] = None
    depth_cy: Optional[float] = None
    depth_width: Optional[int] = None
    depth_height: Optional[int] = None
    _rgb_np: Any = field(default=None, repr=False)
    _depth_np: Any = field(default=None, repr=False)
    _rgb_bytes: Optional[bytes] = field(default=None, repr=False)
    _depth_bytes: Optional[bytes] = field(default=None, repr=False)
    _sens_header: Optional[dict] = field(default=None, repr=False)
    source_ordinal: Optional[int] = None
    color_ts_ns: Optional[int] = None
    depth_ts_ns: Optional[int] = None
    pose_ts_ns: Optional[int] = None
    rgb_depth_dt_ns: Optional[int] = None
    rgb_pose_dt_ns: Optional[int] = None
    color_bag_ns: Optional[int] = None
    depth_bag_ns: Optional[int] = None

    @classmethod
    def from_mapping_frame(cls, frame: MappingFrame) -> "StreamingObservation":
        """Project canonical buffers/identity/calibration without copying them.

        The currently promoted streaming mapper consumes native HWC ``uint8``
        RGB and metric HxW ``float32`` depth. Fail closed if an application
        constructs its MappingSession with a different materialization policy;
        silently reinterpreting or rescaling here would create a second compute
        boundary and defeat exactly-once materialization.
        """
        rgb = frame.rgb.data
        if not isinstance(rgb, np.ndarray):
            raise TypeError("streaming mapper currently requires CPU NumPy RGB materialization")
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[-1] != 3:
            raise ValueError(
                "streaming mapper requires HxWx3 uint8 RGB; construct MappingSession "
                "with ColorPolicy.PRESERVE"
            )

        depth = None if frame.depth is None else frame.depth.data
        if depth is not None:
            if not isinstance(depth, np.ndarray):
                raise TypeError("streaming mapper currently requires CPU NumPy depth materialization")
            units = (frame.depth.spec.units or "").lower()
            if depth.dtype != np.float32 or depth.ndim != 2 or units not in {
                "m",
                "meter",
                "meters",
                "metre",
                "metres",
            }:
                raise ValueError(
                    "streaming mapper requires HxW metric float32 depth; construct "
                    "MappingSession with DepthPolicy.METERS_FLOAT32"
                )

        source = frame.source
        intr = frame.intrinsics
        pose = None if frame.pose is None else np.asarray(frame.pose.c2w, dtype=np.float32)
        ordinal = int(source.source_ordinal)
        return cls(
            mapping_frame=frame,
            index=ordinal,
            timestamp=source.timestamp_seconds,
            c2w=pose,
            fx=float(intr.fx),
            fy=float(intr.fy),
            cx=float(intr.cx),
            cy=float(intr.cy),
            width=int(intr.width),
            height=int(intr.height),
            depth_scale=1.0,
            depth_fx=float(intr.depth_fx),
            depth_fy=float(intr.depth_fy),
            depth_cx=float(intr.depth_cx),
            depth_cy=float(intr.depth_cy),
            depth_width=int(intr.depth_width),
            depth_height=int(intr.depth_height),
            _rgb_np=rgb,
            _depth_np=depth,
            source_ordinal=ordinal,
            color_ts_ns=int(source.timestamp_ns),
            depth_ts_ns=int(source.timestamp_ns) if depth is not None else None,
            pose_ts_ns=int(source.timestamp_ns) if pose is not None else None,
            rgb_depth_dt_ns=0 if depth is not None else None,
            rgb_pose_dt_ns=0 if pose is not None else None,
        )


__all__ = ["StreamingObservation"]
