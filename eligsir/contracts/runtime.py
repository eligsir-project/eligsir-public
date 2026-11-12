"""Owner-safe runtime contracts shared by every EliGSiR ingress/runtime path."""

from __future__ import annotations

import hashlib
import numbers
from dataclasses import dataclass, field, fields as dataclass_fields
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping, Optional

import numpy as np

from .types import CameraIntrinsics, Pose


def _canonical_int(value: Any, name: str) -> int:
    """Coerce a real integer (including NumPy integers), rejecting bool/float."""
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, not a bool")
    if not isinstance(value, numbers.Integral):
        raise TypeError(f"{name} must be an integer number, got {type(value).__name__}")
    return int(value)


def _digest(*parts: Any) -> str:
    """Stable content digest over an explicitly ordered tuple of scalars."""
    hasher = hashlib.blake2b(digest_size=16)
    for part in parts:
        hasher.update(repr(part).encode("utf-8"))
        hasher.update(b"\x1f")
    return hasher.hexdigest()


class MemoryDomain(str, Enum):
    """Memory domain containing a sensor/compute buffer."""

    HOST = "host"
    PINNED_HOST = "pinned_host"
    CUDA = "cuda"
    ROCM = "rocm"
    OTHER_DEVICE = "other_device"


class BufferOwnership(str, Enum):
    """Lifetime/ownership relationship of a buffer crossing a boundary."""

    BORROWED = "borrowed"
    OWNED = "owned"
    SHARED_IMMUTABLE = "shared_immutable"
    POOLED = "pooled"


class TrackingStatus(str, Enum):
    """Canonical frontend state independent of transport/backend."""

    INITIALIZING = "initializing"
    VALID = "valid"
    DEGRADED = "degraded"
    LOST = "lost"
    INVALID = "invalid"


class IngressAction(str, Enum):
    """Canonical action taken at one ingress ownership boundary."""

    WAIT = "wait"
    DELIVER = "deliver"
    REJECT = "reject"
    DROP = "drop"
    SPILL = "spill"
    FAIL = "fail"


@dataclass(frozen=True, slots=True)
class SourceIdentity:
    """Drop-invariant physical source identity.

    ``source_ordinal`` is assigned before source-side drop/rejection/decimation.
    The tuple returned by :meth:`ordering_key` is the canonical commit order;
    decode, transfer, ROS callback, and worker completion times must never
    become ordering inputs.
    """

    source_id: str
    stream_epoch: int
    source_ordinal: int
    timestamp_ns: int
    pair_id: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.source_id:
            raise ValueError("source_id must be non-empty")
        object.__setattr__(self, "stream_epoch", _canonical_int(self.stream_epoch, "stream_epoch"))
        object.__setattr__(
            self, "source_ordinal", _canonical_int(self.source_ordinal, "source_ordinal")
        )
        object.__setattr__(self, "timestamp_ns", _canonical_int(self.timestamp_ns, "timestamp_ns"))
        if self.stream_epoch < 0:
            raise ValueError("stream_epoch must be >= 0")
        if self.source_ordinal < 0:
            raise ValueError("source_ordinal must be >= 0")
        if self.pair_id == "":
            raise ValueError("pair_id must be None or non-empty")

    @property
    def timestamp_seconds(self) -> float:
        """Lossy presentation timestamp; integer nanoseconds remain canonical."""
        return self.timestamp_ns * 1e-9

    def ordering_key(self) -> tuple[int, int, int, str]:
        """Canonical deterministic handover/commit key."""
        return (self.stream_epoch, self.source_ordinal, self.timestamp_ns, self.source_id)

    def fingerprint(self) -> str:
        """Stable cross-process/cross-machine digest of this identity."""
        return _digest(
            self.source_id,
            self.stream_epoch,
            self.source_ordinal,
            self.timestamp_ns,
            self.pair_id,
        )


@dataclass(frozen=True, slots=True)
class BufferSpec:
    """Explicit layout/device semantics for a buffer."""

    shape: tuple[int, ...]
    dtype: str
    memory_domain: MemoryDomain
    encoding: str
    units: Optional[str] = None
    device: Optional[str] = None
    contiguous: bool = True
    strides_bytes: Optional[tuple[int, ...]] = None

    def __post_init__(self) -> None:
        if not self.shape or any(int(dim) <= 0 for dim in self.shape):
            raise ValueError(f"shape must contain positive dimensions, got {self.shape!r}")
        if not self.dtype:
            raise ValueError("dtype must be non-empty")
        if not self.encoding:
            raise ValueError("encoding must be non-empty")
        if self.strides_bytes is not None and len(self.strides_bytes) != len(self.shape):
            raise ValueError("strides_bytes rank must match shape rank")
        if self.memory_domain in {
            MemoryDomain.CUDA,
            MemoryDomain.ROCM,
            MemoryDomain.OTHER_DEVICE,
        } and not self.device:
            raise ValueError(f"{self.memory_domain.value} buffers require an explicit device")


@dataclass(frozen=True, slots=True)
class BufferRef:
    """A payload plus explicit memory/lifetime metadata.

    ``owner_token`` may retain a transport message, pooled frame slot, capsule,
    or other object whose lifetime keeps borrowed storage valid.
    """

    data: Any = field(compare=False, repr=False)
    spec: BufferSpec
    ownership: BufferOwnership
    owner_token: Any = field(default=None, compare=False, repr=False)

    @classmethod
    def from_numpy(
        cls,
        array: "np.ndarray",
        *,
        encoding: str,
        ownership: BufferOwnership = BufferOwnership.OWNED,
        units: Optional[str] = None,
        memory_domain: MemoryDomain = MemoryDomain.HOST,
        device: Optional[str] = None,
        owner_token: Any = None,
    ) -> "BufferRef":
        """Wrap an array and record its real layout without copying it."""
        array = np.asarray(array)
        return cls(
            data=array,
            spec=BufferSpec(
                shape=tuple(int(dim) for dim in array.shape),
                dtype=str(array.dtype),
                memory_domain=memory_domain,
                encoding=encoding,
                units=units,
                device=device,
                contiguous=bool(array.flags["C_CONTIGUOUS"]),
                strides_bytes=tuple(int(stride) for stride in array.strides),
            ),
            ownership=ownership,
            owner_token=owner_token,
        )

    def spec_fingerprint(self) -> str:
        """Stable digest of layout/ownership metadata, never the payload."""
        return _digest(
            self.spec.shape,
            self.spec.dtype,
            self.spec.memory_domain.value,
            self.spec.encoding,
            self.spec.units,
            self.spec.device,
            self.spec.contiguous,
            self.spec.strides_bytes,
            self.ownership.value,
        )


@dataclass(frozen=True, slots=True)
class PixelReloadToken:
    """Deterministic, pixel-free locator for a reloadable observation."""

    provider: str
    locators: tuple[str, ...]
    parameters: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if not self.provider:
            raise ValueError("reload-token provider must be non-empty")
        if not self.locators or not self.locators[0]:
            raise ValueError("reload token requires a primary locator")
        object.__setattr__(self, "locators", tuple(str(item) for item in self.locators))
        object.__setattr__(
            self,
            "parameters",
            tuple((str(name), str(value)) for name, value in self.parameters),
        )

    def fingerprint(self) -> str:
        return _digest(self.provider, self.locators, self.parameters)


@dataclass(frozen=True, slots=True)
class TrackingObservation:
    """Transport-independent tracking evidence for one source observation."""

    source: SourceIdentity
    pose: Optional[Pose]
    status: TrackingStatus
    provider: str
    confidence: Optional[float] = None
    covariance: Optional[np.ndarray] = field(default=None, compare=False, repr=False)
    diagnostics: Mapping[str, Any] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        if not self.provider:
            raise ValueError("provider must be non-empty")
        if self.pose is None and self.status in {TrackingStatus.VALID, TrackingStatus.DEGRADED}:
            raise ValueError(f"{self.status.value} tracking requires a pose")
        if self.confidence is not None and not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        if self.covariance is not None:
            covariance = np.array(self.covariance, dtype=np.float64, copy=True)
            if covariance.shape != (6, 6):
                raise ValueError(f"covariance must be (6, 6), got {covariance.shape}")
            covariance.setflags(write=False)
            object.__setattr__(self, "covariance", covariance)
        object.__setattr__(self, "diagnostics", MappingProxyType(dict(self.diagnostics)))

    @property
    def usable(self) -> bool:
        return self.status in {TrackingStatus.VALID, TrackingStatus.DEGRADED}


@dataclass(frozen=True, slots=True)
class SensorPacket:
    """Transport-neutral sensor observation before compute materialization."""

    source: SourceIdentity
    rgb: BufferRef
    depth: Optional[BufferRef]
    intrinsics: CameraIntrinsics
    tracking: Optional[TrackingObservation] = None
    reload_token: Optional[PixelReloadToken] = None

    def __post_init__(self) -> None:
        if self.tracking is not None and self.tracking.source != self.source:
            raise ValueError("tracking.source must match packet source identity")
        if self.reload_token is not None and not isinstance(
            self.reload_token, PixelReloadToken
        ):
            raise TypeError("reload_token must be PixelReloadToken when provided")

    def metadata_fingerprint(self) -> str:
        """Stable digest of identity + layout + calibration, never pixels."""
        intrinsics_parts = tuple(
            (f.name, getattr(self.intrinsics, f.name))
            for f in sorted(dataclass_fields(self.intrinsics), key=lambda f: f.name)
        )
        return _digest(
            self.source.fingerprint(),
            self.rgb.spec_fingerprint(),
            self.depth.spec_fingerprint() if self.depth is not None else None,
            intrinsics_parts,
            None if self.reload_token is None else self.reload_token.fingerprint(),
        )


def _buffer_device(ref: BufferRef) -> str:
    return ref.spec.device or "cpu"


@dataclass(frozen=True, slots=True)
class MappingFrame:
    """Compute-facing frame with ownership attached to each payload.

    A frame-level ownership flag is intentionally absent: RGB and depth may
    have different producers/lifetimes after materialization. Every buffer
    remains a :class:`BufferRef` all the way through the compute boundary, so
    ``spec``, ``ownership`` and ``owner_token`` cannot become detached from the
    payload they describe.

    This module never imports torch/ROCm/CUDA: ``BufferRef.data`` is typed
    ``Any`` and this contract never inspects it, only the ``BufferRef``/
    ``BufferSpec`` wrapper (plain dataclasses) around it -- so a producer
    holding a raw device-resident tensor stays torch-free at this boundary by
    wrapping it in a ``BufferRef`` that declares the real ``device``, not by
    bypassing ``BufferRef`` entirely.
    """

    source: SourceIdentity
    rgb: BufferRef
    depth: Optional[BufferRef]
    intrinsics: CameraIntrinsics
    device: str
    pose: Optional[Pose] = None
    reload_token: Optional[PixelReloadToken] = None

    def __post_init__(self) -> None:
        if not self.device:
            raise ValueError("device must be non-empty")
        if self.reload_token is not None and not isinstance(
            self.reload_token, PixelReloadToken
        ):
            raise TypeError("reload_token must be PixelReloadToken when provided")
        for name, ref in (("rgb", self.rgb), ("depth", self.depth)):
            if ref is None:
                continue
            spec_device = _buffer_device(ref)
            if spec_device != self.device:
                raise ValueError(
                    f"{name}.spec.device ({spec_device!r}) must match frame device "
                    f"({self.device!r}) -- a mismatch is an unnoticed CPU<->GPU bounce"
                )

    def retainable(self) -> bool:
        """Whether every payload can safely outlive its immediate producer."""
        refs = (self.rgb,) if self.depth is None else (self.rgb, self.depth)
        return all(
            ref.ownership in {BufferOwnership.OWNED, BufferOwnership.SHARED_IMMUTABLE}
            for ref in refs
        )


@dataclass(frozen=True, slots=True)
class IngressEvent:
    """Single-owner attribution for one ingress decision."""

    source: SourceIdentity
    stage: str
    owner: str
    action: IngressAction
    reason: str
    queue_depth: Optional[int] = None
    details: Mapping[str, Any] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        if not self.stage:
            raise ValueError("stage must be non-empty")
        if not self.owner:
            raise ValueError("owner must be non-empty")
        if not self.reason:
            raise ValueError("reason must be non-empty")
        if self.queue_depth is not None and self.queue_depth < 0:
            raise ValueError("queue_depth must be >= 0")
        object.__setattr__(self, "details", MappingProxyType(dict(self.details)))
