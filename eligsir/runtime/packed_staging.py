"""Aligned heterogeneous host->device packing over reusable pinned byte slabs."""

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Callable, Iterable, Optional

import numpy as np
import torch

from eligsir.runtime.transfers import DeviceTransferStream, TransferEvent


@dataclass(frozen=True)
class PackedField:
    offset: int
    nbytes: int
    shape: tuple[int, ...]
    torch_dtype: torch.dtype


# eq=False is load-bearing, not style. A pool slot is an IDENTITY, and
# `release_after` checks membership with `slot not in self._slots`. Python's
# `in` falls back to `==` for every element that is not the same object, so a
# generated dataclass __eq__ would compare the `buffer` field -- a tensor --
# and raise (`Boolean value of Tensor with more than one value is ambiguous`,
# or a shape-mismatch RuntimeError) as soon as a SECOND slot has been given a
# buffer. That made every legacy-NumPy staging transfer after the first one
# crash on this persistent per-device pool. eq=False restores identity
# semantics, which is what membership here means.
@dataclass(eq=False)
class _Slot:
    """One reusable pinned byte-slab allocation.

    ``eq=False`` keeps equality/hash as identity (``is``/``id()``): pool
    membership checks (``slot not in self._slots``) need object identity, not
    structural equality -- comparing ``buffer`` (a ``torch.Tensor``) via
    ``==`` returns an elementwise tensor and crashes membership testing for
    any non-empty, multi-element buffer. Same root cause as issue #1153's
    ``_PinnedTensorSlot`` in ``eligsir/runtime/transfers.py``.
    """

    buffer: Optional[torch.Tensor] = None
    pending: Optional[TransferEvent] = None
    claimed: bool = False

    def reusable(self) -> bool:
        if self.claimed:
            return False
        if self.pending is None:
            return True
        if self.pending.query():
            self.pending = None
            return True
        return False


class PinnedByteSlabPool:
    """Fixed slot pool; slots cannot be overwritten while DMA still reads them."""

    def __init__(
        self,
        slots: int = 4,
        *,
        stats=None,
        capacity_reserver: Optional[Callable[[int], None]] = None,
    ):
        if slots <= 0:
            raise ValueError("slots must be positive")
        self._slots = [_Slot() for _ in range(int(slots))]
        self._next = 0
        self._lock = Lock()
        self.exhaustions = 0
        self._stats = stats
        self._capacity_reserver = capacity_reserver

    def acquire(self, nbytes: int, *, blocking: bool = True, opportunistic: bool = False) -> _Slot:
        if not self._lock.acquire(blocking=blocking):
            self.exhaustions += 1
            if self._stats is not None:
                self._stats.pool_drops += 1
            raise RuntimeError("PinnedByteSlabPool is busy")
        try:
            if opportunistic:
                # Optional feedback may borrow only existing spare storage. Keep
                # the largest idle upload slab and make it the next mandatory
                # choice, so neither a slot nor pinned-byte growth blocks ingress.
                idle = sorted(
                    ((int(slot.buffer.numel()), idx) for idx, slot in enumerate(self._slots)
                     if slot.reusable() and slot.buffer is not None),
                )
                if len(idle) >= 2:
                    for size, idx in idle[:-1]:
                        if size >= nbytes:
                            slot = self._slots[idx]
                            slot.claimed = True
                            self._next = idle[-1][1]
                            return slot
                self.exhaustions += 1
                if self._stats is not None:
                    self._stats.pool_drops += 1
                raise RuntimeError("PinnedByteSlabPool has no spare upload-safe capacity")
            for off in range(len(self._slots)):
                idx = (self._next + off) % len(self._slots)
                slot = self._slots[idx]
                if not slot.reusable():
                    continue
                if slot.buffer is None or int(slot.buffer.numel()) < int(nbytes):
                    previous = 0 if slot.buffer is None else int(slot.buffer.numel())
                    if self._capacity_reserver is not None:
                        self._capacity_reserver(int(nbytes) - previous)
                    slot.buffer = torch.empty(
                        int(nbytes),
                        dtype=torch.uint8,
                        pin_memory=torch.cuda.is_available(),
                    )
                slot.claimed = True
                self._next = (idx + 1) % len(self._slots)
                return slot
            self.exhaustions += 1
            if self._stats is not None:
                self._stats.pool_waits += 1
            raise RuntimeError(
                "PinnedByteSlabPool exhausted: in-flight transfer depth exceeds configured slots"
            )
        finally:
            self._lock.release()

    def release_after(self, slot: _Slot, event: TransferEvent) -> None:
        with self._lock:
            if slot not in self._slots or not slot.claimed:
                raise ValueError("slot is not an acquired member of this pool")
            slot.pending = event
            slot.claimed = False

    def release_now(self, slot: _Slot) -> None:
        """Release a claimed slot whose contents are no longer borrowed."""

        with self._lock:
            if slot not in self._slots or not slot.claimed:
                raise ValueError("slot is not an acquired member of this pool")
            slot.pending = None
            slot.claimed = False

    def preallocate(self, nbytes: int) -> None:
        """Reserve bounded capacity outside a producer hot path."""

        slot = self.acquire(nbytes)
        self.release_now(slot)
        # Make the prepared slab the next candidate.  Otherwise acquire()'s
        # round-robin cursor would grow a second slot on the first output.
        with self._lock:
            self._next = self._slots.index(slot)

    @property
    def pending_count(self) -> int:
        """Return currently claimed or DMA-leased slabs."""
        with self._lock:
            return sum(not slot.reusable() for slot in self._slots)


def _torch_dtype_for_numpy(dtype: np.dtype) -> torch.dtype:
    # Constructing a scalar is cheap and keeps NumPy/PyTorch dtype mapping in
    # one authoritative implementation rather than maintaining a manual map.
    return torch.from_numpy(np.empty((), dtype=dtype)).dtype


def _align(offset: int, alignment: int) -> int:
    return (int(offset) + alignment - 1) // alignment * alignment


def layout_numpy_fields(arrays: Iterable[Optional[np.ndarray]], *, alignment: int = 8):
    fields: list[Optional[PackedField]] = []
    contiguous: list[Optional[np.ndarray]] = []
    offset = 0
    for array in arrays:
        if array is None:
            fields.append(None)
            contiguous.append(None)
            continue
        arr = np.asarray(array)
        if not (arr.flags["C_CONTIGUOUS"] and arr.flags["WRITEABLE"]):
            arr = np.array(arr, copy=True, order="C")
        offset = _align(offset, max(alignment, int(arr.dtype.itemsize)))
        nbytes = int(arr.nbytes)
        fields.append(
            PackedField(
                offset=offset,
                nbytes=nbytes,
                shape=tuple(int(v) for v in arr.shape),
                torch_dtype=_torch_dtype_for_numpy(arr.dtype),
            )
        )
        contiguous.append(arr)
        offset += nbytes
    return tuple(contiguous), tuple(fields), _align(offset, alignment)


def pack_numpy_fields_to_device(
    arrays: Iterable[Optional[np.ndarray]],
    *,
    transfer: DeviceTransferStream,
    pool: PinnedByteSlabPool,
    handoff_to_compute: bool = True,
) -> tuple[tuple[Optional[torch.Tensor], ...], TransferEvent]:
    """Pack heterogeneous arrays into one pinned byte slab and one H2D.

    ``handoff_to_compute`` remains enabled for the established camera and
    insertion callers.  Prepared-source production disables it because the
    source producer only records preparation; the eventual consumer/binder
    owns the compute-stream dependency.
    """
    contiguous, fields, total_bytes = layout_numpy_fields(arrays)
    slot = pool.acquire(total_bytes)
    event = None
    try:
        host = slot.buffer[:total_bytes]
        for arr, field in zip(contiguous, fields):
            if arr is None or field is None:
                continue
            source = torch.from_numpy(arr).view(torch.uint8).reshape(-1)
            host[field.offset : field.offset + field.nbytes].copy_(source)

        device_bytes, event = transfer.h2d(host)
        if handoff_to_compute:
            transfer.handoff_to_compute(event)
        pool.release_after(slot, event)
    except Exception:
        # Before h2d returns there is no DMA reader and immediate release is
        # safe.  Once it returns, retain the slab behind its event even if a
        # later handoff or view operation fails.
        if event is None:
            pool.release_now(slot)
        else:
            pool.release_after(slot, event)
        raise

    views: list[Optional[torch.Tensor]] = []
    for field in fields:
        if field is None:
            views.append(None)
            continue
        raw = device_bytes[field.offset : field.offset + field.nbytes]
        typed = raw.view(field.torch_dtype).reshape(field.shape)
        views.append(typed)
    return tuple(views), event


__all__ = [
    "PackedField",
    "PinnedByteSlabPool",
    "layout_numpy_fields",
    "pack_numpy_fields_to_device",
]
