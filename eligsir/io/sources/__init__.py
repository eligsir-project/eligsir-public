"""Frame source registry."""

from .base import FrameSource, SourceFactory
from .registry import SOURCES, SourceInfo, detect_source_kind, list_sources, make_frame_source

__all__ = [
    "FrameSource",
    "SOURCES",
    "SourceInfo",
    "SourceFactory",
    "detect_source_kind",
    "list_sources",
    "make_frame_source",
]
