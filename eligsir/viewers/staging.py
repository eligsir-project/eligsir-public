"""Bounded async GPU->CPU preview staging ring.

Pinned slots are reused, copies run on an explicit per-device side stream, and
completion is discovered by event polling from the drain thread.  The copy
stream waits on the producer/current stream before reading a rendered tensor;
the training thread itself never synchronizes on D2H completion.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional

import numpy as np

__all__ = [
    "SlotState",
    "PreviewStagingRing",
    "StagedPair",
    "MapSnapshotStaging",
    "StagedMap",
]


class SlotState(Enum):
    FREE = "FREE"
    COPYING = "COPYING"
    READY = "READY"


@dataclass
class _Half:
    host_buffer: Optional[Any] = None
    event: Optional[Any] = None
    _source_shape: Optional[tuple] = None
    ready_direct: Optional[np.ndarray] = None


@dataclass
class _PairSlot:
    index: int
    state: SlotState = SlotState.FREE
    target: _Half = field(default_factory=_Half)
    render: _Half = field(default_factory=_Half)
    meta: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class StagedPair:
    target: np.ndarray
    render: np.ndarray
    meta: Dict[str, Any]


_COPY_STREAMS: Dict[str, Any] = {}
_COPY_STREAMS_LOCK = threading.Lock()


def _copy_stream_for(tensor: Any, torch):
    """One reusable D2H side stream per device (CUDA API also covers HIP)."""
    key = str(tensor.device)
    with _COPY_STREAMS_LOCK:
        stream = _COPY_STREAMS.get(key)
        if stream is None:
            stream = torch.cuda.Stream(device=tensor.device)
            _COPY_STREAMS[key] = stream
        return stream


def _schedule_half_copy(half: _Half, tensor: Any, torch) -> None:
    if not tensor.is_cuda:
        half.ready_direct = tensor.detach().contiguous().numpy().copy()
        half.event = None
        return
    shape = tuple(tensor.shape)
    if (
        half.host_buffer is None
        or half._source_shape != shape
        or half.host_buffer.dtype != tensor.dtype
    ):
        half.host_buffer = torch.empty(shape, dtype=tensor.dtype, pin_memory=True)
        half._source_shape = shape
    half.ready_direct = None

    copy_stream = _copy_stream_for(tensor, torch)
    producer_stream = torch.cuda.current_stream(tensor.device)
    # Correct cross-stream dependency: copy must not observe a render tensor
    # before its producing kernels finish. This is device-side only.
    copy_stream.wait_stream(producer_stream)
    with torch.cuda.stream(copy_stream):
        half.host_buffer.copy_(tensor.detach(), non_blocking=True)
        event = torch.cuda.Event(enable_timing=False, blocking=False)
        event.record(copy_stream)
    half.event = event


def _half_ready(half: _Half) -> bool:
    if half.ready_direct is not None:
        return True
    if half.event is not None:
        return bool(half.event.query())
    return False


def _materialize_half(half: _Half) -> np.ndarray:
    if half.ready_direct is not None:
        return half.ready_direct
    return np.array(half.host_buffer.numpy(), copy=True)


class PreviewStagingRing:
    """Fixed pool of reusable target+render staging slots."""

    def __init__(self, n_slots: int = 3) -> None:
        if n_slots < 1:
            raise ValueError("n_slots must be >= 1")
        self._lock = threading.Lock()
        self._slots = [_PairSlot(index=i) for i in range(n_slots)]
        self._dropped = 0
        self._staged = 0
        self._completed = 0

    def try_submit_pair(self, target: Any, render: Any, meta: Dict[str, Any]) -> bool:
        import torch

        with self._lock:
            slot = next((s for s in self._slots if s.state is SlotState.FREE), None)
            if slot is None:
                self._dropped += 1
                return False
            slot.state = SlotState.COPYING
            slot.meta = dict(meta)
            self._staged += 1

        self._schedule_copy(slot.target, target, torch)
        self._schedule_copy(slot.render, render, torch)
        with self._lock:
            if self._half_ready(slot.target) and self._half_ready(slot.render):
                slot.state = SlotState.READY
        return True

    @staticmethod
    def _schedule_copy(half: _Half, tensor: Any, torch) -> None:
        _schedule_half_copy(half, tensor, torch)

    @staticmethod
    def _half_ready(half: _Half) -> bool:
        return _half_ready(half)

    def poll_completed(self) -> Optional[StagedPair]:
        with self._lock:
            for slot in self._slots:
                if slot.state is SlotState.COPYING and self._half_ready(slot.target) and self._half_ready(slot.render):
                    slot.state = SlotState.READY
            ready_slot = next((s for s in self._slots if s.state is SlotState.READY), None)
            if ready_slot is None:
                return None
            target_arr = self._materialize(ready_slot.target)
            render_arr = self._materialize(ready_slot.render)
            meta = ready_slot.meta or {}
            self._free(ready_slot)
            self._completed += 1
            return StagedPair(target=target_arr, render=render_arr, meta=meta)

    @staticmethod
    def _materialize(half: _Half) -> np.ndarray:
        return _materialize_half(half)

    @staticmethod
    def _free(slot: _PairSlot) -> None:
        slot.state = SlotState.FREE
        slot.target.event = None
        slot.target.ready_direct = None
        slot.render.event = None
        slot.render.ready_direct = None
        slot.meta = None

    def stats(self) -> Dict[str, int]:
        with self._lock:
            return {
                "staging_staged": self._staged,
                "staging_dropped": self._dropped,
                "staging_completed": self._completed,
            }


@dataclass(frozen=True)
class StagedMap:
    positions: np.ndarray
    colors: Optional[np.ndarray]
    meta: Dict[str, Any]
    scales: Optional[np.ndarray] = None
    quats: Optional[np.ndarray] = None
    opacities: Optional[np.ndarray] = None


_OPTIONAL_MAP_ATTRS = ("colors", "scales", "quats", "opacities")


class MapSnapshotStaging:
    """Single-slot async D2H staging for bounded map snapshots.

    It shares exactly the same pinned-buffer/event/side-stream primitive as
    the preview ring. Submissions while busy are dropped rather than blocking
    the optimizer.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state = SlotState.FREE
        self._positions = _Half()
        self._halves: Dict[str, _Half] = {name: _Half() for name in _OPTIONAL_MAP_ATTRS}
        self._present: Dict[str, bool] = {name: False for name in _OPTIONAL_MAP_ATTRS}
        self._meta: Optional[Dict[str, Any]] = None
        self._dropped = 0
        self._staged = 0
        self._completed = 0

    def try_submit(
        self,
        positions: Any,
        colors: Optional[Any],
        meta: Dict[str, Any],
        *,
        scales: Optional[Any] = None,
        quats: Optional[Any] = None,
        opacities: Optional[Any] = None,
    ) -> bool:
        import torch

        optional = {
            "colors": colors,
            "scales": scales,
            "quats": quats,
            "opacities": opacities,
        }
        with self._lock:
            if self._state is not SlotState.FREE:
                self._dropped += 1
                return False
            self._state = SlotState.COPYING
            self._meta = dict(meta)
            for name, value in optional.items():
                self._present[name] = value is not None
            self._staged += 1

        _schedule_half_copy(self._positions, positions, torch)
        for name, value in optional.items():
            if value is not None:
                _schedule_half_copy(self._halves[name], value, torch)
        with self._lock:
            if self._ready_locked():
                self._state = SlotState.READY
        return True

    def _ready_locked(self) -> bool:
        if not _half_ready(self._positions):
            return False
        for name in _OPTIONAL_MAP_ATTRS:
            if self._present[name] and not _half_ready(self._halves[name]):
                return False
        return True

    def poll_completed(self) -> Optional[StagedMap]:
        with self._lock:
            if self._state is SlotState.COPYING and self._ready_locked():
                self._state = SlotState.READY
            if self._state is not SlotState.READY:
                return None
            positions_arr = _materialize_half(self._positions)
            materialized = {
                name: (_materialize_half(self._halves[name]) if self._present[name] else None)
                for name in _OPTIONAL_MAP_ATTRS
            }
            meta = self._meta or {}
            self._state = SlotState.FREE
            self._positions.event = None
            self._positions.ready_direct = None
            for name in _OPTIONAL_MAP_ATTRS:
                self._halves[name].event = None
                self._halves[name].ready_direct = None
            self._meta = None
            self._completed += 1
            return StagedMap(positions=positions_arr, meta=meta, **materialized)

    def stats(self) -> Dict[str, int]:
        with self._lock:
            return {
                "map_staging_staged": self._staged,
                "map_staging_dropped": self._dropped,
                "map_staging_completed": self._completed,
            }
