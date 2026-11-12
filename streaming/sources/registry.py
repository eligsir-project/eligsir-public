"""Compatibility wrapper for the canonical package source registry."""

from eligsir.io.sources import (
    FrameSource,
    SOURCES,
    SourceInfo,
    detect_source_kind,
    list_sources,
    make_frame_source,
)

__all__ = [
    "FrameSource",
    "SOURCES",
    "SourceInfo",
    "detect_source_kind",
    "list_sources",
    "make_frame_source",
]
