"""Bounded transfer primitives for frame ingress, camera paging and outputs.

Hot-path invariants:

* pinned host capacity is allocated once and leased until DMA completion;
* H2D/D2H copies run on an explicit reusable side stream with events;
* D2H copy streams wait on the producing compute stream before reading data;
* logical commit order never depends on transfer completion timing;
* camera paging is grouped by shape/dtype, not B individual transfers;
* transfer counts/bytes are observable for regression gates.

PyTorch exposes HIP/ROCm streams/events through ``torch.cuda`` as well, so no
vendor-specific branch is needed here.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from enum import Enum
from threading import Lock
from time import perf_counter
from typing import Callable, Iterable, Optional, Sequence

import numpy as np
import torch


@dataclass
class TransferStats:
    """Cumulative physical transfer and bounded-pool counters."""

    h2d_ops: int = 0
    h2d_bytes: int = 0
    d2h_ops: int = 0
    d2h_bytes: int = 0
    pool_waits: int = 0
    pool_drops: int = 0
    explicit_sync_waits: int = 0
    explicit_sync_wait_ms: float = 0.0

    def snapshot(self) -> dict[str, int | float]:
        """Return an immutable-by-copy dictionary of current counters."""
        return {
            "h2d_ops": int(self.h2d_ops),
            "h2d_bytes": int(self.h2d_bytes),
            "d2h_ops": int(self.d2h_ops),
            "d2h_bytes": int(self.d2h_bytes),
            "pool_waits": int(self.pool_waits),
            "pool_drops": int(self.pool_drops),
            "explicit_sync_waits": int(self.explicit_sync_waits),
            "explicit_sync_wait_ms": float(self.explicit_sync_wait_ms),
        }


@dataclass(frozen=True, slots=True)
class TransferAccounting:
    """Immutable aggregate view published by the sole device transfer owner."""

    h2d_ops: int
    h2d_bytes: int
    d2h_ops: int
    d2h_bytes: int
    pool_waits: int
    pool_drops: int
    pinned_bytes: int
    peak_pinned_bytes: int
    inflight_limit: int
    shape_buckets: int
    shape_bucket_limit: int
    explicit_sync_waits: int = 0
    explicit_sync_wait_ms: float = 0.0


class ColdTransferBarrier(str, Enum):
    """Explicit bulk readback boundaries outside the live mapping hot path."""

    CHECKPOINT = "checkpoint"
    REPORT = "report"
    OUTPUT = "output"


class _PinnedCapacity:
    """One monotonic pinned-allocation budget shared by all owner pools."""

    # Issue #1259: warn once, loudly, the first time observed usage crosses
    # this fraction of the configured ceiling -- visibility into how close a
    # run is running to the limit, without changing the fail-closed behavior
    # when it is actually exceeded.
    _WARN_UTILIZATION_FRACTION = 0.8

    def __init__(self, max_bytes: int, stats: TransferStats) -> None:
        if max_bytes <= 0:
            raise ValueError("max_pinned_bytes must be positive")
        self.max_bytes = int(max_bytes)
        self.current_bytes = 0
        self.peak_bytes = 0
        self._stats = stats
        self._lock = Lock()
        self._warned_high_watermark = False

    def reserve(self, delta: int) -> None:
        delta = int(delta)
        if delta <= 0:
            return
        with self._lock:
            requested = self.current_bytes + delta
            if requested > self.max_bytes:
                self._stats.pool_drops += 1
                raise RuntimeError(
                    "DeviceTransferOwner pinned byte limit exceeded: "
                    f"requested={requested} limit={self.max_bytes}"
                )
            self.current_bytes = requested
            self.peak_bytes = max(self.peak_bytes, requested)
            if (
                not self._warned_high_watermark
                and self.peak_bytes >= self._WARN_UTILIZATION_FRACTION * self.max_bytes
            ):
                self._warned_high_watermark = True
                print(
                    "[DeviceTransferOwner] WARNING: pinned transfer pool usage "
                    f"crossed {self._WARN_UTILIZATION_FRACTION:.0%} of its configured "
                    f"ceiling (peak_bytes={self.peak_bytes} limit={self.max_bytes}, "
                    f"{100.0 * self.peak_bytes / self.max_bytes:.1f}% used). Consider "
                    "raising --streaming_max_transfer_bytes for this scene if this "
                    "run later fails with 'pinned byte limit exceeded'.",
                    flush=True,
                )


class TransferEvent:
    """Backend-neutral completion handle wrapping a torch CUDA/HIP event."""

    def __init__(self, event=None, *, stats: Optional[TransferStats] = None) -> None:
        """Wrap an optional device event; ``None`` denotes immediate completion."""
        self._event = event
        self._stats = stats

    def query(self) -> bool:
        """Poll completion without synchronizing the host."""
        return True if self._event is None else bool(self._event.query())

    def wait(self) -> None:
        """Synchronize the host with this event for cold/output consumers."""
        if self._event is not None:
            started = perf_counter()
            self._event.synchronize()
            if self._stats is not None:
                self._stats.explicit_sync_waits += 1
                self._stats.explicit_sync_wait_ms += (perf_counter() - started) * 1000.0

    def wait_on_current_stream(self, device: torch.device | str) -> None:
        """Insert a device-side dependency into the current compute stream."""
        if self._event is not None:
            torch.cuda.current_stream(torch.device(device)).wait_event(self._event)


class DeviceTransferStream:
    """Explicit reusable copy stream for CUDA and ROCm torch builds."""

    def __init__(self, device: torch.device | str, stats: Optional[TransferStats] = None):
        """Create one transfer owner for a target device and shared counters."""
        self.device = torch.device(device)
        self.stats = stats if stats is not None else TransferStats()
        self._stream = (
            torch.cuda.Stream(device=self.device) if self.device.type == "cuda" else None
        )

    def _context(self):
        """Return the side-stream context or a CPU no-op context."""
        return torch.cuda.stream(self._stream) if self._stream is not None else nullcontext()

    def h2d(self, host: torch.Tensor, *, dtype=None) -> tuple[torch.Tensor, TransferEvent]:
        """Submit one host-to-device copy and return payload plus completion event."""
        if host.device.type != "cpu":
            raise ValueError("h2d source must be a CPU tensor")
        with self._context():
            out = host.to(
                device=self.device,
                dtype=dtype,
                non_blocking=self._stream is not None and host.is_pinned(),
            )
            event = None
            if self._stream is not None:
                event = torch.cuda.Event(blocking=False)
                event.record(self._stream)
        self.stats.h2d_ops += 1
        self.stats.h2d_bytes += int(host.numel()) * int(host.element_size())
        return out, TransferEvent(event, stats=self.stats)

    def prepare(self, operation: Callable[[], object]) -> tuple[object, TransferEvent]:
        """Run destination-device preparation on this stream and fence it.

        The callable is deliberately tiny and synchronous from Python's point
        of view: CUDA/HIP kernels it launches remain ordered on this stream.
        CPU callers use the same contract with an immediate event, which keeps
        ordering tests deterministic without requiring an accelerator.
        """
        if not callable(operation):
            raise TypeError("operation must be callable")
        with self._context():
            value = operation()
            event = None
            if self._stream is not None:
                event = torch.cuda.Event(blocking=False)
                event.record(self._stream)
        return value, TransferEvent(event, stats=self.stats)

    def prepare_ready_calibration(
        self,
        *,
        pose: tuple[tuple[float, ...], ...],
        camera_intrinsics: tuple[tuple[float, ...], ...],
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], TransferEvent]:
        """Construct attested ready-camera payloads without a mutable callback.

        This deliberately does not route through :meth:`prepare`: a callback
        may replace its return values after the producer's allocation.  The
        producer supplies immutable CPU attestation values, and this sole
        device owner turns precisely those values into stream-ordered tensors.

        Calibration values are assembled in a packed CPU buffer and submitted
        in one copy_ per tensor to minimize launch overhead.
        """

        if len(pose) != 4 or any(len(row) != 4 for row in pose):
            raise ValueError("ready calibration pose must be a 4x4 tuple")
        if len(camera_intrinsics) != 3 or any(
            len(row) != 3 for row in camera_intrinsics
        ):
            raise ValueError("ready calibration intrinsics must be a 3x3 tuple")

        # Build packed CPU host buffers before creating device tensors.
        # Float32 narrowing happens once during construction, not per-element.
        c2w_host = torch.tensor(pose, dtype=torch.float32, device="cpu")
        intrinsics_host = torch.tensor(
            camera_intrinsics, dtype=torch.float32, device="cpu"
        )

        with self._context():
            c2w = torch.empty((4, 4), dtype=torch.float32, device=self.device)
            intrinsics = torch.empty((3, 3), dtype=torch.float32, device=self.device)
            # Single copy_ per tensor replaces 16 + 9 scalar device assignments.
            c2w.copy_(c2w_host, non_blocking=False)
            intrinsics.copy_(intrinsics_host, non_blocking=False)
            event = None
            if self._stream is not None:
                event = torch.cuda.Event(blocking=False)
                event.record(self._stream)
        return (c2w, intrinsics), TransferEvent(event, stats=self.stats)

    def d2h_into(self, device_tensor: torch.Tensor, host: torch.Tensor) -> TransferEvent:
        """Submit one device-to-host copy after waiting on its producer stream."""
        if host.device.type != "cpu":
            raise ValueError("D2H destination must be CPU")
        if host.shape != device_tensor.shape or host.dtype != device_tensor.dtype:
            raise ValueError("D2H destination shape/dtype mismatch")
        if device_tensor.device != self.device:
            raise ValueError(
                f"D2H source is on {device_tensor.device}, transfer stream owns {self.device}"
            )
        if self._stream is not None:
            self._stream.wait_stream(torch.cuda.current_stream(self.device))
        with self._context():
            host.copy_(
                device_tensor,
                non_blocking=self._stream is not None and host.is_pinned(),
            )
            event = None
            if self._stream is not None:
                device_tensor.record_stream(self._stream)
                event = torch.cuda.Event(blocking=False)
                event.record(self._stream)
        self.stats.d2h_ops += 1
        self.stats.d2h_bytes += int(device_tensor.numel()) * int(device_tensor.element_size())
        return TransferEvent(event, stats=self.stats)

    def handoff_to_compute(self, *events: TransferEvent) -> None:
        """Make the current compute stream depend on completed copy submissions."""
        for event in events:
            event.wait_on_current_stream(self.device)


@dataclass(eq=False)
class _PinnedTensorSlot:
    """One reusable pinned allocation plus the DMA event currently leasing it.

    ``eq=False`` keeps the dataclass-generated equality/hash as plain identity
    (``is``/``id()``). Pool membership checks (``slot not in self._slots``) and
    prior key lookups need to ask "is this the exact same slot object", not
    "do these slots hold structurally equal contents" -- the latter reading
    would compare ``buffer`` (a ``torch.Tensor``) via ``==``, which returns an
    elementwise tensor rather than a bool and crashes Python's membership-test
    machinery for any non-empty, multi-element buffer (issue #1153).
    """

    buffer: Optional[torch.Tensor] = None
    pending: Optional[TransferEvent] = None
    claimed: bool = False

    def reusable(self) -> bool:
        """Return whether no caller/DMA operation currently owns this slot."""
        if self.claimed:
            return False
        if self.pending is None:
            return True
        if self.pending.query():
            self.pending = None
            return True
        return False


class PinnedTensorPool:
    """Shape-agnostic fixed pool of pinned homogeneous tensor capacity."""

    def __init__(
        self,
        slots: int = 4,
        *,
        stats: Optional[TransferStats] = None,
        capacity_reserver=None,
    ):
        """Allocate pool metadata; pinned buffers grow lazily per slot."""
        if slots <= 0:
            raise ValueError("slots must be positive")
        self.stats = stats if stats is not None else TransferStats()
        self._slots = [_PinnedTensorSlot() for _ in range(int(slots))]
        self._next = 0
        self._lock = Lock()
        self._capacity_reserver = capacity_reserver

    def acquire(self, shape, dtype: torch.dtype) -> tuple[_PinnedTensorSlot, torch.Tensor]:
        """Claim a reusable slot and return a typed view of requested shape."""
        needed = int(np.prod(shape))
        with self._lock:
            for offset in range(len(self._slots)):
                idx = (self._next + offset) % len(self._slots)
                slot = self._slots[idx]
                if not slot.reusable():
                    continue
                if slot.buffer is None or slot.buffer.dtype != dtype or slot.buffer.numel() < needed:
                    previous = (
                        0
                        if slot.buffer is None or slot.buffer.dtype != dtype
                        else int(slot.buffer.numel()) * int(slot.buffer.element_size())
                    )
                    requested = needed * int(torch.empty((), dtype=dtype).element_size())
                    if self._capacity_reserver is not None:
                        self._capacity_reserver(requested - previous)
                    slot.buffer = torch.empty(
                        needed,
                        dtype=dtype,
                        pin_memory=torch.cuda.is_available(),
                    )
                slot.claimed = True
                self._next = (idx + 1) % len(self._slots)
                return slot, slot.buffer[:needed].view(*shape)
            self.stats.pool_waits += 1
            raise RuntimeError("PinnedTensorPool exhausted; in-flight transfer bound exceeded")

    def release_after(self, slot: _PinnedTensorSlot, event: TransferEvent) -> None:
        """Lease a claimed slot to DMA until ``event`` reports completion."""
        with self._lock:
            if slot not in self._slots or not slot.claimed:
                raise ValueError("slot is not an acquired member of this pool")
            slot.pending = event
            slot.claimed = False

    def release_now(self, slot: _PinnedTensorSlot) -> None:
        """Release a slot immediately after its contents have been copied out."""
        with self._lock:
            if slot not in self._slots:
                raise ValueError("slot is not a member of this pool")
            slot.pending = None
            slot.claimed = False

    @property
    def pending_count(self) -> int:
        """Return currently claimed or DMA-leased tensor slots."""
        with self._lock:
            return sum(not slot.reusable() for slot in self._slots)


class PinnedFrameSlot:
    """Reusable host slot for one RGB/depth frame."""

    def __init__(self) -> None:
        """Create an empty slot whose allocations grow monotonically."""
        self.rgb: Optional[torch.Tensor] = None
        self.depth: Optional[torch.Tensor] = None
        self.in_use = False
        self.pending: Optional[TransferEvent] = None
        self.generation = 0

    def reusable(self) -> bool:
        """Poll whether host storage is safe to reuse after its H2D copy."""
        if self.in_use:
            return False
        if self.pending is None:
            return True
        if self.pending.query():
            self.pending = None
            return True
        return False

    @staticmethod
    def _ensure(existing: Optional[torch.Tensor], shape, dtype) -> torch.Tensor:
        """Return pinned capacity large enough for ``shape`` and ``dtype``."""
        needed = int(np.prod(shape))
        if existing is None or existing.dtype != dtype or existing.numel() < needed:
            return torch.empty(needed, dtype=dtype, pin_memory=True)
        return existing

    def stage(self, rgb: np.ndarray, depth: Optional[np.ndarray]):
        """Copy RGB and optional depth into this slot's reusable pinned storage."""
        rgb_np = np.asarray(rgb)
        rgb_dtype = torch.from_numpy(np.empty((), dtype=rgb_np.dtype)).dtype
        self.rgb = self._ensure(self.rgb, rgb_np.shape, rgb_dtype)
        rgb_view = self.rgb[: rgb_np.size].view(*rgb_np.shape)
        rgb_view.copy_(torch.from_numpy(np.ascontiguousarray(rgb_np)))
        depth_view = None
        if depth is not None:
            depth_np = np.asarray(depth)
            depth_dtype = torch.from_numpy(np.empty((), dtype=depth_np.dtype)).dtype
            self.depth = self._ensure(self.depth, depth_np.shape, depth_dtype)
            depth_view = self.depth[: depth_np.size].view(*depth_np.shape)
            depth_view.copy_(torch.from_numpy(np.ascontiguousarray(depth_np)))
        self.generation += 1
        return rgb_view, depth_view


class PinnedFramePool:
    """Fixed-size, non-allocating steady-state pinned frame pool."""

    def __init__(self, slots: int = 4, *, stats: Optional[TransferStats] = None):
        """Create a bounded round-robin set of pinned frame slots."""
        if slots <= 0:
            raise ValueError("slots must be positive")
        self.stats = stats if stats is not None else TransferStats()
        self._slots = [PinnedFrameSlot() for _ in range(int(slots))]
        self._next = 0
        self._lock = Lock()

    def acquire(self) -> PinnedFrameSlot:
        """Claim the next free frame slot or fail when in-flight work is unbounded."""
        with self._lock:
            for offset in range(len(self._slots)):
                idx = (self._next + offset) % len(self._slots)
                slot = self._slots[idx]
                if slot.reusable():
                    slot.in_use = True
                    self._next = (idx + 1) % len(self._slots)
                    return slot
            self.stats.pool_waits += 1
            raise RuntimeError("PinnedFramePool exhausted; caller must bound in-flight frames")

    def release(self, slot: PinnedFrameSlot) -> None:
        """Return a previously acquired frame slot to the pool."""
        with self._lock:
            if slot not in self._slots or not slot.in_use:
                raise ValueError("slot is not an acquired member of this pool")
            slot.in_use = False

    def release_after(self, slot: PinnedFrameSlot, event: TransferEvent) -> None:
        """Keep a staged slot leased until its H2D completion event is ready."""
        if not callable(getattr(event, "query", None)):
            raise TypeError("event must provide query()")
        with self._lock:
            if slot not in self._slots or not slot.in_use:
                raise ValueError("slot is not an acquired member of this pool")
            slot.pending = event
            slot.in_use = False

    def release_now(self, slot: PinnedFrameSlot) -> None:
        """Release a slot when the caller has already established DMA safety."""
        with self._lock:
            if slot not in self._slots:
                raise ValueError("slot is not a member of this pool")
            slot.pending = None
            slot.in_use = False

    @property
    def pending_count(self) -> int:
        """Return slots still claimed by a caller or pending H2D completion."""
        with self._lock:
            return sum(not slot.reusable() for slot in self._slots)


@dataclass
class CameraBatchTransfer:
    """One shape-bucketed camera H2D submission awaiting compute handoff."""

    cameras: tuple
    images: torch.Tensor
    event: TransferEvent
    shape: tuple[int, ...]
    attribute: str = "original_image"

    def bind_after_handoff(self, transfer: DeviceTransferStream) -> None:
        """Insert a compute-stream dependency and bind device image views."""
        transfer.handoff_to_compute(self.event)
        from eligsir.utils.score_contract import reattach_camera_score_provenance

        for index, cam in enumerate(self.cameras):
            setattr(cam, self.attribute, self.images[index])
            if self.attribute == "original_image":
                reattach_camera_score_provenance(cam)


class _CameraBatchPrefetcher:
    """Deterministic shape-bucketed camera H2D prefetch."""

    def __init__(
        self,
        transfer: DeviceTransferStream,
        *,
        slots_per_bucket: int = 3,
        max_shape_buckets: Optional[int] = None,
        capacity_reserver=None,
        bucket_claimer=None,
    ):
        """Bind one device transfer stream and bounded per-shape pools."""
        self.transfer = transfer
        self._pools: dict[tuple, PinnedTensorPool] = {}
        self._slots_per_bucket = int(slots_per_bucket)
        self._max_shape_buckets = (
            None if max_shape_buckets is None else int(max_shape_buckets)
        )
        self._capacity_reserver = capacity_reserver
        self._bucket_claimer = bucket_claimer

    def prefetch(self, cameras: Iterable) -> list[CameraBatchTransfer]:
        """Submit one H2D per CPU-resident camera shape/dtype bucket."""
        buckets: dict[tuple, list] = {}
        for cam in cameras:
            image = getattr(cam, "original_image", None)
            if image is None:
                if getattr(cam, "has_pending_image", False):
                    cam.materialize_pending_image()
                    image = cam.original_image
                else:
                    raise ValueError("camera has no materialized image")
            for attribute, payload in (
                ("original_image", image),
                ("_sensor_depth_cache", getattr(cam, "_sensor_depth_cache", None)),
            ):
                if payload is None or payload is False:
                    continue
                if not isinstance(payload, torch.Tensor):
                    raise TypeError(f"camera {attribute} must be a tensor when present")
                if payload.device == self.transfer.device:
                    continue
                if payload.device.type != "cpu":
                    raise ValueError(
                        f"camera {attribute} is on unexpected device {payload.device}"
                    )
                # Preserve the established RGB bucket identity so its later D2H
                # counterpart shares the reservation; depth gets its own tier.
                key = (
                    (tuple(payload.shape), payload.dtype)
                    if attribute == "original_image"
                    else ("depth", tuple(payload.shape), payload.dtype)
                )
                buckets.setdefault(key, []).append(cam)

        out: list[CameraBatchTransfer] = []
        for key in sorted(buckets, key=str):
            cams = tuple(buckets[key])
            if key[0] == "depth":
                _, payload_shape, payload_dtype = key
                attribute = "_sensor_depth_cache"
            else:
                attribute = "original_image"
                payload_shape, payload_dtype = key
            shape = (len(cams),) + payload_shape
            pool = self._pools.get(key)
            if pool is None:
                if self._bucket_claimer is not None:
                    self._bucket_claimer(key)
                elif (
                    self._max_shape_buckets is not None
                    and len(self._pools) >= self._max_shape_buckets
                ):
                    self.transfer.stats.pool_drops += 1
                    raise RuntimeError(
                        "DeviceTransferOwner shape bucket limit exceeded: "
                        f"limit={self._max_shape_buckets}"
                    )
                pool = PinnedTensorPool(
                    slots=self._slots_per_bucket,
                    stats=self.transfer.stats,
                    capacity_reserver=self._capacity_reserver,
                )
                self._pools[key] = pool
            slot, host = pool.acquire(shape, payload_dtype)
            for index, cam in enumerate(cams):
                host[index].copy_(getattr(cam, attribute))
            device_batch, event = self.transfer.h2d(host)
            pool.release_after(slot, event)
            out.append(
                CameraBatchTransfer(
                    cameras=cams,
                    images=device_batch,
                    event=event,
                    shape=tuple(device_batch.shape),
                    attribute=attribute,
                )
            )
        return out

    def prefetch_and_bind(self, cameras: Iterable) -> int:
        """Submit all buckets, hand them to compute, and return bucket count."""
        transfers = self.prefetch(cameras)
        for pending in transfers:
            pending.bind_after_handoff(self.transfer)
        return len(transfers)

    @property
    def shape_bucket_count(self) -> int:
        """Return the bounded number of allocated camera shape/dtype pools."""

        return len(self._pools)


@dataclass
class PendingCameraOffload:
    """One grouped D2H camera copy retaining its pool lease until completion."""

    cameras: tuple
    host: torch.Tensor
    event: TransferEvent
    pool: PinnedTensorPool
    slot: _PinnedTensorSlot

    def complete(self) -> None:
        """Wait for D2H, give cameras owned CPU storage, then release the slot."""
        self.event.wait()
        from eligsir.utils.score_contract import reattach_camera_score_provenance

        for index, cam in enumerate(self.cameras):
            cam.original_image = self.host[index].clone()
            reattach_camera_score_provenance(cam)
        self.pool.release_now(self.slot)


class CameraBatchOffloader:
    """Group camera D2H by device/shape/dtype and submit all before waiting."""

    def __init__(
        self,
        *,
        slots_per_bucket: int = 3,
        transfer: Optional[DeviceTransferStream] = None,
        capacity_reserver=None,
        bucket_claimer=None,
    ):
        """Create bounded D2H pools, optionally on an injected owner stream."""
        self._transfers: dict[str, DeviceTransferStream] = {}
        self._pools: dict[tuple, PinnedTensorPool] = {}
        self._slots_per_bucket = int(slots_per_bucket)
        self._injected_transfer = transfer
        self._capacity_reserver = capacity_reserver
        self._bucket_claimer = bucket_claimer

    def _transfer(self, device: torch.device) -> DeviceTransferStream:
        """Return the reusable D2H transfer owner for one device."""
        key = str(device)
        if self._injected_transfer is not None:
            if torch.device(self._injected_transfer.device) != device:
                raise ValueError(
                    "camera D2H source device does not match DeviceTransferOwner"
                )
            return self._injected_transfer
        transfer = self._transfers.get(key)
        if transfer is None:
            transfer = DeviceTransferStream(device, TransferStats())
            self._transfers[key] = transfer
        return transfer

    def offload(self, cameras: Iterable) -> int:
        """Offload all accelerator-resident cameras and return camera count."""
        buckets: dict[tuple, list] = {}
        for cam in cameras:
            img = getattr(cam, "original_image", None)
            if img is None or img.device.type == "cpu":
                continue
            key = (str(img.device), tuple(img.shape), img.dtype)
            buckets.setdefault(key, []).append(cam)

        pending: list[PendingCameraOffload] = []
        for key in sorted(buckets, key=lambda k: (k[0], k[1], str(k[2]))):
            cams = tuple(buckets[key])
            device = cams[0].original_image.device
            transfer = self._transfer(device)
            batch = torch.stack([cam.original_image.detach() for cam in cams], dim=0).contiguous()
            pool = self._pools.get(key)
            if pool is None:
                if self._bucket_claimer is not None:
                    self._bucket_claimer((tuple(batch.shape[1:]), batch.dtype))
                pool = PinnedTensorPool(
                    slots=self._slots_per_bucket,
                    stats=transfer.stats,
                    capacity_reserver=self._capacity_reserver,
                )
                self._pools[key] = pool
            slot, host = pool.acquire(tuple(batch.shape), batch.dtype)
            event = transfer.d2h_into(batch, host)
            pending.append(PendingCameraOffload(cams, host, event, pool, slot))

        for item in pending:
            item.complete()
        return sum(len(item.cameras) for item in pending)

    def stats(self) -> dict[str, int]:
        """Aggregate transfer counters across all lazily-created device streams."""
        if self._injected_transfer is not None:
            return self._injected_transfer.stats.snapshot()
        total = TransferStats()
        for transfer in self._transfers.values():
            snap = transfer.stats.snapshot()
            total.h2d_ops += snap["h2d_ops"]
            total.h2d_bytes += snap["h2d_bytes"]
            total.d2h_ops += snap["d2h_ops"]
            total.d2h_bytes += snap["d2h_bytes"]
            total.pool_waits += snap["pool_waits"]
            total.pool_drops += snap["pool_drops"]
            total.explicit_sync_waits += snap.get("explicit_sync_waits", 0)
            total.explicit_sync_wait_ms += snap.get("explicit_sync_wait_ms", 0.0)
        return total.snapshot()


@dataclass
class PackedReadback:
    """Packed host readback plus the event protecting its contents."""

    host: torch.Tensor
    event: TransferEvent
    pool: object | None = None
    slot: object | None = None
    _consumed: bool = False
    _waited: bool = False

    def wait(self) -> None:
        """Wait once for the producer event before consuming the leased slab."""
        if self._consumed:
            raise RuntimeError("packed readback has already been consumed")
        if not self._waited:
            self.event.wait()
            self._waited = True

    def release_after_wait(self) -> None:
        """Return a slab after the caller's explicit completion wait.

        This deliberately does not wait again: a mutation cardinality fence
        has already paid its single attributed host wait before validation.
        """
        if self._consumed:
            raise RuntimeError("packed readback has already been consumed")
        if not self._waited:
            raise RuntimeError("packed readback must be waited before release")
        if self.pool is not None:
            self.pool.release_now(self.slot)
        self._consumed = True

    def numpy(self):
        """Wait for completion and expose the packed host tensor as NumPy."""
        if self._consumed:
            raise RuntimeError("packed readback has already been consumed")
        self.wait()
        array = self.host.numpy().copy()
        self.release_after_wait()
        return array

    def release(self) -> None:
        """Release this completed bounded slab without creating a NumPy copy."""
        if self._consumed:
            raise RuntimeError("packed readback has already been consumed")
        self.wait()
        self.release_after_wait()

    def try_take_cpu_tensor(self) -> torch.Tensor | None:
        """Poll and take a completed slab without introducing a host wait."""

        if self._consumed:
            raise RuntimeError("packed readback has already been consumed")
        if not self.event.query():
            return None
        # The event is complete, so cloning before releasing the slab preserves
        # the caller's ownership while keeping this path poll-only.
        value = self.host.clone()
        self._waited = True
        self.release_after_wait()
        return value

    def try_release_if_ready(self) -> bool:
        """Poll and release a completed slab without copying or waiting."""

        if self._consumed:
            return True
        if not self.event.query():
            return False
        self._waited = True
        self.release_after_wait()
        return True

    def discard(self) -> None:
        """Abandon feedback without waiting or reusing its DMA destination early."""
        if not self._consumed:
            if self.pool is not None:
                self.pool.release_after(self.slot, self.event)
            self._consumed = True


def packed_readback(
    tensors: Sequence[torch.Tensor], transfer: DeviceTransferStream, *, pool=None,
    opportunistic: bool = False,
) -> PackedReadback:
    """Concatenate same-device tensors and issue one bounded asynchronous D2H."""
    if not tensors:
        raise ValueError("packed_readback requires at least one tensor")
    flat = torch.cat([x.detach().reshape(-1) for x in tensors], dim=0).contiguous()
    slot = None
    if pool is None:
        host = torch.empty_like(flat, device="cpu", pin_memory=flat.device.type == "cuda")
    else:
        nbytes = int(flat.numel()) * int(flat.element_size())
        slot = (
            pool.acquire(nbytes, blocking=False, opportunistic=True)
            if opportunistic else pool.acquire(nbytes, blocking=False)
        )
        host = slot.buffer[:nbytes].view(flat.dtype)[: flat.numel()]
    try:
        event = transfer.d2h_into(flat, host)
    except Exception:
        if pool is not None:
            pool.release_now(slot)
        raise
    return PackedReadback(host=host, event=event, pool=pool, slot=slot)


class DeviceTransferOwner:
    """Sole live bulk-transfer owner for one accelerator device.

    The owner promotes the existing copy stream, event handoff, packed slab and
    camera-bucket primitives behind one bounded interface and one accounting
    record.  Callers may share this object, but may not create parallel streams
    or pools for the same live composition.
    """

    def __init__(
        self,
        device: torch.device | str,
        *,
        max_inflight: int = 4,
        max_shape_buckets: int = 8,
        # Raised 2026-08-29 (issue #1259) to match the CLI/config default --
        # see arguments/__init__.py's streaming_max_transfer_bytes comment.
        max_pinned_bytes: int = 2 * 1024 * 1024 * 1024,
        transfer_stream=None,
    ) -> None:
        if max_inflight <= 0:
            raise ValueError("max_inflight must be positive")
        if max_shape_buckets <= 0:
            raise ValueError("max_shape_buckets must be positive")
        self.device = torch.device(device)
        self._max_inflight = int(max_inflight)
        self._max_shape_buckets = int(max_shape_buckets)
        self._transfer = transfer_stream or DeviceTransferStream(self.device)
        if torch.device(self._transfer.device) != self.device:
            raise ValueError("transfer_stream device must match DeviceTransferOwner device")
        self.stats = self._transfer.stats
        self._capacity = _PinnedCapacity(max_pinned_bytes, self.stats)
        self._shape_bucket_keys: set[tuple] = set()

        from eligsir.runtime.packed_staging import PinnedByteSlabPool

        self._packed_pool = PinnedByteSlabPool(
            slots=self._max_inflight,
            stats=self.stats,
            capacity_reserver=self._capacity.reserve,
        )
        self._camera_prefetcher = _CameraBatchPrefetcher(
            self._transfer,
            slots_per_bucket=self._max_inflight,
            max_shape_buckets=self._max_shape_buckets,
            capacity_reserver=self._capacity.reserve,
            bucket_claimer=self._claim_shape_bucket,
        )
        self._camera_offloader = CameraBatchOffloader(
            slots_per_bucket=self._max_inflight,
            transfer=self._transfer,
            capacity_reserver=self._capacity.reserve,
            bucket_claimer=self._claim_shape_bucket,
        )

    @property
    def max_pinned_bytes(self) -> int:
        """Return the immutable shared pinned-memory ceiling."""

        return int(self._capacity.max_bytes)

    def _claim_shape_bucket(self, key: tuple) -> None:
        """Reserve one shared camera shape/dtype bucket across H2D and D2H."""

        if key in self._shape_bucket_keys:
            return
        if len(self._shape_bucket_keys) >= self._max_shape_buckets:
            self.stats.pool_drops += 1
            raise RuntimeError(
                "DeviceTransferOwner shape bucket limit exceeded: "
                f"limit={self._max_shape_buckets}"
            )
        self._shape_bucket_keys.add(key)

    @property
    def transfer_stream(self):
        """Return the owned stream for low-level device-native executors."""

        return self._transfer

    def upload_packed(
        self, arrays, *, purpose: str, handoff_to_compute: bool = True
    ):
        """Pack heterogeneous host fields into one bounded H2D submission."""

        if not purpose:
            raise ValueError("transfer purpose must be non-empty")
        from eligsir.runtime.packed_staging import pack_numpy_fields_to_device

        return pack_numpy_fields_to_device(
            arrays,
            transfer=self._transfer,
            pool=self._packed_pool,
            handoff_to_compute=handoff_to_compute,
        )

    def prepare_packed_sources(self, source_arrays, *, purpose: str):
        """Submit one bounded packed H2D for ordered prepared-source fields.

        Returned tensors are transient views into the batch upload allocation.
        Prepared-source publication must materialize source-local tensors before
        retaining a READY lease, so partial lease eviction cannot retain this
        whole packed allocation.
        """
        source_arrays = tuple(tuple(fields) for fields in source_arrays)
        if not source_arrays or any(not fields for fields in source_arrays):
            raise ValueError("prepared source batch must contain non-empty source fields")
        from eligsir.runtime.packed_staging import layout_numpy_fields

        flat = tuple(field for fields in source_arrays for field in fields)
        _arrays, _fields, packed_bytes = layout_numpy_fields(flat)
        if packed_bytes > self.max_pinned_bytes or self._packed_pool.pending_count >= self._max_inflight:
            raise RuntimeError(
                "PinnedByteSlabPool exhausted: in-flight transfer depth exceeds configured slots"
            )
        uploaded, event = self.upload_packed(flat, purpose=purpose, handoff_to_compute=False)
        offset = 0
        grouped = []
        for fields in source_arrays:
            grouped.append(tuple(uploaded[offset : offset + len(fields)]))
            offset += len(fields)
        return tuple(grouped), event

    def prepare_source(
        self,
        arrays,
        *,
        purpose: str = "prepared_source",
        depth_units: str | None = None,
        depth_scale: float | None = None,
    ):
        """Upload and prepare one RGB/depth source on the owner copy stream.

        The packed slab is released behind the H2D event.  Returned device
        views retain their own device allocation while the preparation event
        is used as the consumer-side compute-stream dependency.
        """
        uploaded, transfer_event = self.upload_packed(
            arrays,
            purpose=purpose,
            handoff_to_compute=False,
        )
        rgb_raw, depth_raw = uploaded
        units = str(depth_units or "").lower()
        if depth_raw is not None and units not in {"m", "metres", "meters"}:
            if (
                depth_scale is None
                or not np.isfinite(float(depth_scale))
                or float(depth_scale) <= 0.0
            ):
                raise ValueError(
                    "raw depth requires a positive finite intrinsics depth_scale"
                )

        def _convert():
            rgb = rgb_raw.to(dtype=torch.float32).div(255.0)
            if depth_raw is None:
                depth = None
            else:
                depth = depth_raw.to(dtype=torch.float32)
                if units not in {"m", "metres", "meters"}:
                    depth = depth / float(depth_scale)
            return rgb, depth

        prepare = getattr(self._transfer, "prepare", None)
        if callable(prepare):
            prepared, prep_event = prepare(_convert)
        else:
            # Compatibility with the small CPU transfer doubles used by the
            # older owner tests.  Real DeviceTransferStream always takes the
            # stream-ordered branch above.
            prepared = _convert()
            prep_event = TransferEvent()
        return prepared, transfer_event, prep_event

    def prepare_ready_calibration(
        self,
        *,
        pose: tuple[tuple[float, ...], ...],
        camera_intrinsics: tuple[tuple[float, ...], ...],
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], TransferEvent]:
        """Make ready calibration only from the producer's frozen attestation."""

        return self._transfer.prepare_ready_calibration(
            pose=pose,
            camera_intrinsics=camera_intrinsics,
        )

    def prepare_source_tiers(
        self,
        rgb_hwc: torch.Tensor,
        depth_meters: torch.Tensor | None,
        camera_intrinsics: torch.Tensor,
        *,
        resampling_mode: str = "legacy",
        depth_downsample: str = "nearest",
        declared_tiers: frozenset | None = None,
    ) -> tuple[dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]], TransferEvent]:
        """Derive declared source tiers on the prep stream.

        The tiers materialized are determined by the declared_tiers set, which
        must not be empty. No lazy tier construction after READY.
        """

        from eligsir.contracts.control import ResolutionTier
        from eligsir.mapping.resolution_materialization import materialize_resolution_tensors

        # Default to every legal adaptive tier when no policy is supplied.
        # This import is local to avoid circular dependency since prepared_source_producer imports this module.
        if declared_tiers is None:
            from eligsir.runtime.prepared_source_producer import DEFAULT_PREPARED_TIERS
            declared_tiers = frozenset(DEFAULT_PREPARED_TIERS)
        if not isinstance(declared_tiers, frozenset):
            declared_tiers = frozenset(declared_tiers)
        if not declared_tiers:
            raise ValueError("declared_tiers must not be empty")

        def _derive():
            depth = depth_meters
            if depth is None:
                depth = torch.zeros(
                    rgb_hwc.shape[:2], dtype=torch.float32, device=rgb_hwc.device
                )
            return {
                tier.value: materialize_resolution_tensors(
                    rgb_hwc,
                    depth,
                    camera_intrinsics,
                    tier,
                    resampling_mode=resampling_mode,
                    depth_downsample=depth_downsample,
                )
                for tier in declared_tiers
            }

        return self._transfer.prepare(_derive)

    def prepare_source_camera_geometry(
        self,
        c2w: torch.Tensor,
        tiers: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]],
        *,
        w2c: torch.Tensor,
    ) -> tuple[
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        dict[str, tuple[torch.Tensor, torch.Tensor]],
        TransferEvent,
    ]:
        """Prepare map-independent camera geometry after tier derivation.

        Camera geometry is computed for all declared tiers (policy-aware as of #1575 C.1).
        """

        if not tiers:
            raise ValueError("prepared camera geometry requires at least one tier")
        if not isinstance(w2c, torch.Tensor):
            raise ValueError("prepared camera geometry requires producer-precomputed w2c")

        def _derive():
            world_view = w2c.transpose(0, 1).contiguous()
            camera_center = c2w[:3, 3].contiguous()
            gsplat_viewmat = w2c
            gsplat_center = camera_center.view(1, 3).contiguous()
            geometry = {}
            for name, (rgb, _depth, intrinsics, _confidence) in tiers.items():
                height, width = (int(dim) for dim in rgb.shape[:2])
                projection = torch.zeros((4, 4), dtype=torch.float32, device=self.device)
                projection[0, 0] = 2.0 * intrinsics[0, 0] / width
                projection[1, 1] = 2.0 * intrinsics[1, 1] / height
                projection[0, 2] = 2.0 * intrinsics[0, 2] / width - 1.0
                projection[1, 2] = 2.0 * intrinsics[1, 2] / height - 1.0
                projection[3, 2] = 1.0
                projection[2, 2] = 100.0 / (100.0 - 0.01)
                projection[2, 3] = -(100.0 * 0.01) / (100.0 - 0.01)
                projection = projection.transpose(0, 1).contiguous()
                geometry[name] = (
                    projection,
                    world_view.unsqueeze(0).bmm(projection.unsqueeze(0)).squeeze(0),
                )
            return (world_view, camera_center, gsplat_viewmat, gsplat_center), geometry

        (common, per_tier), event = self._transfer.prepare(_derive)
        return common, per_tier, event

    # Descriptive alias for callers that name the packed transport boundary.
    prepare_packed_source = prepare_source

    def handoff_to_compute(self, *events: TransferEvent) -> None:
        """Insert explicit prepared-source dependencies at consumption time."""

        self._transfer.handoff_to_compute(*events)

    def prefetch_cameras(self, cameras: Iterable) -> int:
        """Bucket, upload and hand admitted/replay camera pixels to compute."""

        return self._camera_prefetcher.prefetch_and_bind(cameras)

    def offload_cameras(self, cameras: Iterable) -> int:
        """Bucket and page camera pixels out through the owned D2H stream."""

        return self._camera_offloader.offload(cameras)

    def readback_packed(
        self,
        tensors: Sequence[torch.Tensor],
        *,
        barrier: ColdTransferBarrier,
    ) -> PackedReadback:
        """Submit bulk D2H only at a named cold checkpoint/report/output barrier."""

        if not isinstance(barrier, ColdTransferBarrier):
            raise TypeError("bulk D2H requires an explicit ColdTransferBarrier")
        return packed_readback(tensors, self._transfer, pool=self._packed_pool)

    def readback_mutation_cardinality(self, packed: torch.Tensor) -> PackedReadback:
        """Submit the sole bounded ``MUTATION_CARDINALITY`` D2H envelope."""
        if (
            not isinstance(packed, torch.Tensor)
            or packed.device != self.device
            or packed.dtype is not torch.uint8
            or packed.ndim != 1
        ):
            raise ValueError("mutation-cardinality readback requires a 1-D uint8 owner tensor")
        return packed_readback((packed,), self._transfer, pool=self._packed_pool)

    def readback_supervision_evidence(self, packed: torch.Tensor) -> PackedReadback:
        """Stage at most 16 KiB of E0 RGB/depth residuals and support counts.

        Columns are RGB residual, depth residual, RGB pixels, depth samples.
        Borrow only existing spare capacity, preserving the largest idle ingress
        slab. Unavailable capacity defers feedback without growing pinned memory.
        """
        if (
            not isinstance(packed, torch.Tensor)
            or packed.device != self.device
            or packed.dtype is not torch.float32
            or packed.ndim != 2
            or packed.shape[1] != 4
            or not 0 < packed.shape[0] <= 1024
        ):
            raise ValueError("supervision evidence requires a bounded [B, 4] float32 owner tensor")
        return packed_readback(
            (packed,), self._transfer, pool=self._packed_pool, opportunistic=True,
        )

    def prepare_output_readback(self, max_bytes: int) -> None:
        """Preallocate bounded packed output capacity during composition."""

        self._packed_pool.preallocate(int(max_bytes))

    @property
    def pending_count(self) -> int:
        """Publish real in-flight ownership depth across all bounded pools."""
        camera_pools = tuple(self._camera_prefetcher._pools.values()) + tuple(
            self._camera_offloader._pools.values()
        )
        return self._packed_pool.pending_count + sum(
            pool.pending_count for pool in camera_pools
        )

    def accounting(self) -> TransferAccounting:
        """Publish latest aggregate counters without exposing mutable state."""

        snap = self.stats.snapshot()
        return TransferAccounting(
            h2d_ops=snap["h2d_ops"],
            h2d_bytes=snap["h2d_bytes"],
            d2h_ops=snap["d2h_ops"],
            d2h_bytes=snap["d2h_bytes"],
            pool_waits=snap["pool_waits"],
            pool_drops=snap["pool_drops"],
            pinned_bytes=int(self._capacity.current_bytes),
            peak_pinned_bytes=int(self._capacity.peak_bytes),
            inflight_limit=self._max_inflight,
            shape_buckets=len(self._shape_bucket_keys),
            shape_bucket_limit=self._max_shape_buckets,
            explicit_sync_waits=snap.get("explicit_sync_waits", 0),
            explicit_sync_wait_ms=snap.get("explicit_sync_wait_ms", 0.0),
        )


_DEVICE_TRANSFER_OWNERS: dict[str, DeviceTransferOwner] = {}
_DEVICE_TRANSFER_OWNERS_LOCK = Lock()


def get_device_transfer_owner(
    device: torch.device | str,
    *,
    max_pinned_bytes: int | None = None,
) -> DeviceTransferOwner:
    """Return the compatibility owner shared by all lower-level live consumers.

    Production composition should inject its owner explicitly.  This registry
    keeps legacy callers on one owner during the cutover instead of recreating
    the former module-local streams and pools.
    """

    resolved = torch.device(device)
    key = str(resolved)
    with _DEVICE_TRANSFER_OWNERS_LOCK:
        owner = _DEVICE_TRANSFER_OWNERS.get(key)
        if owner is None:
            owner = DeviceTransferOwner(
                resolved,
                max_pinned_bytes=(
                    2 * 1024 * 1024 * 1024
                    if max_pinned_bytes is None
                    else int(max_pinned_bytes)
                ),
            )
            _DEVICE_TRANSFER_OWNERS[key] = owner
        elif (
            max_pinned_bytes is not None
            and owner.max_pinned_bytes != int(max_pinned_bytes)
        ):
            raise RuntimeError(
                "DeviceTransferOwner already configured with a different pinned "
                f"byte limit: existing={owner.max_pinned_bytes} "
                f"requested={int(max_pinned_bytes)}"
            )
        return owner


def peek_device_transfer_owner(device: torch.device | str) -> DeviceTransferOwner | None:
    """Return the already-registered owner for ``device``, or ``None``.

    Unlike :func:`get_device_transfer_owner`, this never constructs a new
    owner -- it exists purely for read-only reporting call sites (issue
    #1259's ``transfer_pool`` report block) that must not create pinned-memory
    state as a side effect of writing a report.
    """

    with _DEVICE_TRANSFER_OWNERS_LOCK:
        return _DEVICE_TRANSFER_OWNERS.get(str(torch.device(device)))


def transfer_pool_report_block(owner: "DeviceTransferOwner | None") -> Optional[dict]:
    """Build the ``report.json["transfer_pool"]`` block from a transfer owner.

    Returns ``None`` when no owner is available (e.g. a run that never
    touched device transfers), matching this codebase's convention of an
    absent optional report key rather than a zeroed-out placeholder.
    """

    if owner is None:
        return None
    limit = int(owner.max_pinned_bytes)
    peak = int(owner.accounting().peak_pinned_bytes)
    return {
        "configured_limit_bytes": limit,
        "peak_used_bytes": peak,
        "peak_utilization_frac": (float(peak) / float(limit) if limit > 0 else 0.0),
    }
