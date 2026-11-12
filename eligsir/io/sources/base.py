"""Lightweight frame-source interfaces.

Concrete dataset, bag, and ROS2 live sources remain optional integrations. This
module intentionally avoids importing image, ROS, or bag libraries.
"""

from __future__ import annotations

from typing import Any, Iterator, Protocol, runtime_checkable


@runtime_checkable
class FrameSource(Protocol):
    """Iterable source of RGB-D-like frame objects."""

    def __iter__(self) -> Iterator[Any]:
        ...


class SourceFactory(Protocol):
    """Callable that constructs a frame source from a source path and args."""

    def __call__(self, source_path: str, args: Any) -> FrameSource:
        ...
