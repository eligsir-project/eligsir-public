"""Bounded, versioned, latest-wins Gaussian-map output lane (#1200)."""

from __future__ import annotations

from dataclasses import dataclass, fields
from threading import Lock
from typing import Any

import numpy as np

from eligsir.map.views import GaussianMapOutputSample
from eligsir.viewers.events import MapPreviewSnapshot


@dataclass(frozen=True, slots=True)
class MapOutputVersion:
    map_content_version: int
    topology_version: int
    parameter_version: int

    def __post_init__(self) -> None:
        for field in fields(self):
            value = int(getattr(self, field.name))
            if value < 0:
                raise ValueError(f"{field.name} must be non-negative")
            object.__setattr__(self, field.name, value)


@dataclass(frozen=True, slots=True)
class MapOutputSnapshot:
    version: MapOutputVersion
    iteration: int
    positions: np.ndarray
    colors: np.ndarray
    scales: np.ndarray
    quats: np.ndarray
    opacities: np.ndarray
    n_total_gaussians: int

    def __post_init__(self) -> None:
        for name in ("positions", "colors", "scales", "quats", "opacities"):
            getattr(self, name).flags.writeable = False

    @property
    def n_sampled(self) -> int:
        return int(self.positions.shape[0])

    def to_preview_event(self, *, wall_time_s: float) -> MapPreviewSnapshot:
        return MapPreviewSnapshot(
            iteration=int(self.iteration), wall_time_s=float(wall_time_s),
            map_version=self.version.map_content_version,
            positions=self.positions, colors=self.colors, scales=self.scales,
            quats=self.quats, opacities=self.opacities,
            n_total_gaussians=int(self.n_total_gaussians),
        )


@dataclass(frozen=True, slots=True)
class MapOutputLaneStats:
    submitted: int
    duplicates: int
    coalesced: int
    dropped: int
    completed: int
    resident_outputs: int
    max_resident_outputs: int = 2


@dataclass(frozen=True, slots=True)
class _VersionedDeviceSample:
    tensors: GaussianMapOutputSample
    version: MapOutputVersion
    iteration: int


@dataclass(frozen=True, slots=True)
class _InflightOutput:
    sample: _VersionedDeviceSample
    readback: Any


@dataclass(frozen=True, slots=True)
class _OutputRequest:
    backend: Any
    version: MapOutputVersion
    iteration: int


def _version(sample: GaussianMapOutputSample) -> MapOutputVersion:
    return MapOutputVersion(
        sample.map_content_version, sample.topology_version, sample.parameter_version
    )


class VersionedMapOutputLane:
    """One in-flight plus one latest-wins pending output.

    Producer lock acquisition and pooled readback admission are non-blocking:
    preview work is discarded whenever it could delay canonical mapping.
    """

    _FLOAT_FIELDS_PER_GAUSSIAN = 14

    def __init__(self, transfer_owner: Any, *, max_gaussians: int = 100_000) -> None:
        if int(max_gaussians) <= 0:
            raise ValueError("max_gaussians must be positive")
        self._owner = transfer_owner
        self._max_gaussians = int(max_gaussians)
        self._producer_lock = Lock()
        self._lock = Lock()
        self._inflight: _InflightOutput | None = None
        self._pending: _OutputRequest | None = None
        self._last_requested: MapOutputVersion | None = None
        self._submitted = self._duplicates = self._coalesced = 0
        self._dropped = self._completed = 0
        prepare = getattr(self._owner, "prepare_output_readback", None)
        if prepare is not None:
            prepare(self._max_gaussians * self._FLOAT_FIELDS_PER_GAUSSIAN * 4)

    @staticmethod
    def _regressed(new: MapOutputVersion, previous: MapOutputVersion) -> bool:
        return any(
            getattr(new, field.name) < getattr(previous, field.name)
            for field in fields(MapOutputVersion)
        )

    def _start(self, sample: _VersionedDeviceSample) -> _InflightOutput | None:
        # Keep this CPU-side preview data contract importable in the lean smoke
        # image. The transfer implementation is GPU/Torch-owned and is only
        # needed when a real preview readback is actually started.
        from eligsir.runtime.transfers import ColdTransferBarrier

        tensors = sample.tensors
        try:
            readback = self._owner.readback_packed(
                (tensors.positions, tensors.colors, tensors.scales,
                 tensors.quats, tensors.opacities),
                barrier=ColdTransferBarrier.OUTPUT,
            )
        except RuntimeError:
            self._dropped += 1
            return None
        return _InflightOutput(sample=sample, readback=readback)

    def _start_request(self, request: _OutputRequest) -> _InflightOutput | None:
        """Capture/sample/read back entirely on the preview consumer thread."""
        try:
            tensors = request.backend.sample_output(
                max_gaussians=self._max_gaussians,
                parameter_version=request.version.parameter_version,
            )
            if tensors.positions.device != self._owner.device:
                raise ValueError(
                    f"map sample is on {tensors.positions.device}, transfer owner owns {self._owner.device}"
                )
            version = _version(tensors)
            if version != request.version:
                raise RuntimeError(
                    f"backend output sample version changed: {request.version!r} -> {version!r}"
                )
            return self._start(
                _VersionedDeviceSample(tensors, version, request.iteration)
            )
        except Exception:
            self._dropped += 1
            return None

    def try_submit(self, backend: Any, *, parameter_version: int, iteration: int) -> bool:
        """Enqueue only cheap CPU version facts; all GPU work is off-thread."""
        if not self._producer_lock.acquire(blocking=False):
            with self._lock:
                self._dropped += 1
            return False
        try:
            prospective = MapOutputVersion(
                int(getattr(backend, "map_content_version", backend.map_version)),
                int(backend.topology_version),
                int(parameter_version),
            )
            with self._lock:
                previous = self._last_requested
                if previous == prospective:
                    self._duplicates += 1
                    return False
                if previous is not None and self._regressed(prospective, previous):
                    raise ValueError(
                        f"map output version regressed: {previous!r} -> {prospective!r}"
                    )
            request = _OutputRequest(backend, prospective, int(iteration))
            with self._lock:
                self._last_requested = prospective
                self._submitted += 1
                if self._pending is not None:
                    self._coalesced += 1
                self._pending = request
                return True
        finally:
            self._producer_lock.release()

    def poll_completed(self) -> MapOutputSnapshot | None:
        """Return one completed CPU snapshot; never wait for device completion."""
        if not self._producer_lock.acquire(blocking=False):
            return None
        try:
            with self._lock:
                inflight = self._inflight
                if inflight is None:
                    pending = self._pending
                    self._pending = None
                else:
                    pending = None
                if inflight is not None and not inflight.readback.event.query():
                    return None
            if inflight is None:
                if pending is not None:
                    started = self._start_request(pending)
                    with self._lock:
                        self._inflight = started
                return None
            with self._lock:
                self._inflight = None
                pending = self._pending
                self._pending = None
            flat = np.asarray(inflight.readback.numpy())
            if pending is not None:
                with self._lock:
                    self._inflight = self._start_request(pending)
            with self._lock:
                self._completed += 1
        finally:
            self._producer_lock.release()

        sampled = int(inflight.sample.tensors.positions.shape[0])
        shapes = ((sampled, 3), (sampled, 3), (sampled, 3), (sampled, 4))
        arrays: list[np.ndarray] = []
        offset = 0
        for shape in shapes:
            count = int(np.prod(shape))
            arrays.append(np.array(flat[offset:offset + count].reshape(shape), copy=True))
            offset += count
        opacity_count = int(flat.size) - offset
        if sampled <= 0 or opacity_count % sampled:
            raise RuntimeError("packed opacity output does not align with sampled rows")
        opacities = np.array(flat[offset:].reshape(sampled, opacity_count // sampled), copy=True)
        return MapOutputSnapshot(
            version=inflight.sample.version, iteration=inflight.sample.iteration,
            positions=arrays[0], colors=arrays[1], scales=arrays[2],
            quats=arrays[3], opacities=opacities,
            n_total_gaussians=inflight.sample.tensors.n_total_gaussians,
        )

    def stats(self) -> MapOutputLaneStats:
        with self._lock:
            return MapOutputLaneStats(
                submitted=self._submitted, duplicates=self._duplicates,
                coalesced=self._coalesced, dropped=self._dropped,
                completed=self._completed,
                resident_outputs=int(self._inflight is not None) + int(self._pending is not None),
            )


__all__ = ["MapOutputLaneStats", "MapOutputSnapshot", "MapOutputVersion", "VersionedMapOutputLane"]
