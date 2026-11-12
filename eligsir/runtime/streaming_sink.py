"""Minimal embedded-trainer adapter onto the canonical MappingFrame seam.

Production/ROS composition does not use this module. It exists only because the
legacy monolithic dataset trainer still composes its already-initialized
``StreamingScene`` through ``MappingApplicationOptions``. It owns no source,
queue, admission policy, transport, mapper state, or optimizer cadence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

from eligsir.runtime.streaming_observation import StreamingObservation


@dataclass(frozen=True, slots=True)
class StreamingIngestCallbacks:
    """Callbacks already owned by the dataset trainer's ingestion policy."""

    admission_fn: Optional[Callable[..., Any]] = None
    visibility_fn: Optional[Callable[..., Any]] = None
    record_fn: Optional[Callable[..., Any]] = None
    pregate_fn: Optional[Callable[..., Any]] = None
    integrity_observer_fn: Optional[Callable[..., Any]] = None
    defer_extent_update: bool = False


class StreamingSceneMappingSink:
    """Project one canonical MappingFrame into the existing scene ingest owner."""

    __slots__ = ("_scene", "_callbacks")

    def __init__(self, scene: Any, callbacks: StreamingIngestCallbacks) -> None:
        self._scene = scene
        self._callbacks = callbacks

    def __call__(self, frame: Any, _tracking: Any) -> Any:
        observation = StreamingObservation.from_mapping_frame(frame)
        callbacks = self._callbacks
        return self._scene.ingest_frame(
            observation,
            admission_fn=callbacks.admission_fn,
            defer_extent_update=callbacks.defer_extent_update,
            visibility_fn=callbacks.visibility_fn,
            record_fn=callbacks.record_fn,
            pregate_fn=callbacks.pregate_fn,
            integrity_observer_fn=callbacks.integrity_observer_fn,
        )


__all__ = ["StreamingIngestCallbacks", "StreamingSceneMappingSink"]
