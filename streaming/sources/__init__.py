"""Streaming RGB-D frame source registry and lazy compatibility exports."""

from __future__ import annotations

from eligsir.io.sources import (
    FrameSource,
    SOURCES,
    SourceInfo,
    detect_source_kind,
    list_sources,
    make_frame_source,
)

_STREAMING_FRAME_EXPORTS = {
    "StreamingRGBDFrame",
    "load_frame_rgb",
    "load_frame_depth_np",
    "RGBDSequenceFrameSource",
    "TUMFrameSource",
    "ScanNetFrameSource",
    "ScanNetSensFrameSource",
    "ReplicaFrameSource",
    "HyperSimFrameSource",
    "OrbbecExportFrameSource",
    "OrbbecRosBagFrameSource",
    "RealsenseRosBagFrameSource",
}


def __getattr__(name: str):
    if name in _STREAMING_FRAME_EXPORTS:
        from eligsir.utils import streaming_frames

        return getattr(streaming_frames, name)
    raise AttributeError(name)


__all__ = [
    "SOURCES",
    "FrameSource",
    "SourceInfo",
    "detect_source_kind",
    "list_sources",
    "make_frame_source",
    *_STREAMING_FRAME_EXPORTS,
]
