"""Pass-10 source preparation contracts.

This module describes work which is ready to be handed to a later JIT binding
stage.  It intentionally does not know anything about map, topology, or
parameter versions, and never constructs a :class:`GpuWorkEnvelope`.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import math
import numbers
from types import MappingProxyType
from typing import Any, Deque, Mapping

from eligsir.contracts import BufferOwnership, BufferRef, ModalityEligibility, SourceIdentity
from eligsir.runtime.prepared_ingress import PreparedIngressControlToken


def _immutable_metadata(value: Any, name: str = "source_metadata") -> Any:
    """Validate compact metadata and return an immutable representation."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, tuple):
        return tuple(_immutable_metadata(item, name) for item in value)
    raise TypeError(f"{name} must contain only scalar or tuple values")


def _payload_device(payload: Any) -> str | None:
    if not isinstance(payload, BufferRef):
        raise TypeError("prepared payloads must be explicit BufferRef values")
    if payload.ownership is not BufferOwnership.BORROWED:
        raise ValueError("prepared source payloads must be borrowed BufferRef values")
    token = payload.owner_token
    if token is None or (isinstance(token, (str, bytes, bytearray)) and len(token) == 0):
        raise ValueError("borrowed payloads require a non-empty owner/lifetime token")
    declared = payload.spec.device
    actual = getattr(payload.data, "device", None)
    if declared is None and actual is None:
        raise ValueError("device payload must declare a device")
    if declared is not None and actual is not None and str(actual) != str(declared):
        raise ValueError("payload BufferRef device disagrees with its data device")
    return str(declared if declared is not None else actual)


@dataclass(frozen=True, slots=True)
class PreparedTierPayload:
    """One immutable, device-ready RGB-D supervision tier."""

    name: str
    rgb: BufferRef
    depth: BufferRef
    camera_intrinsics: BufferRef
    width: int
    height: int
    depth_confidence: BufferRef | None = None
    intrinsics: tuple[float, float, float, float] | None = None
    projection_matrix: BufferRef | None = None
    full_proj_transform: BufferRef | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("tier name must be non-empty")
        for field_name, payload, encoding, shape in (
            ("rgb", self.rgb, "rgb_float32_hwc", (self.height, self.width, 3)),
            ("depth", self.depth, "depth_meters", (self.height, self.width)),
            ("camera_intrinsics", self.camera_intrinsics, "camera_intrinsics", (3, 3)),
        ):
            if _payload_device(payload) is None:
                raise ValueError(f"tier {field_name} must declare a device")
            if payload.spec.encoding != encoding or tuple(payload.spec.shape) != shape:
                raise ValueError(f"tier {field_name} does not match its payload contract")
        if self.depth.spec.units != "m":
            raise ValueError("tier depth payload units must be metres")
        if self.depth_confidence is not None:
            if _payload_device(self.depth_confidence) is None:
                raise ValueError("tier depth confidence must declare a device")
            if (
                self.depth_confidence.spec.encoding != "depth_confidence_weight"
                or tuple(self.depth_confidence.spec.shape) != (self.height, self.width)
            ):
                raise ValueError("tier depth confidence does not match its payload contract")
        for name in ("width", "height"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < 1:
                raise ValueError(f"tier {name} must be a positive integer")
        if self.intrinsics is not None:
            if (
                not isinstance(self.intrinsics, tuple)
                or len(self.intrinsics) != 4
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    for value in self.intrinsics
                )
                or float(self.intrinsics[0]) <= 0.0
                or float(self.intrinsics[1]) <= 0.0
            ):
                raise ValueError("tier intrinsics must be four finite values with positive focal lengths")
        for field_name, payload, encoding in (
            ("projection_matrix", self.projection_matrix, "projection_matrix"),
            ("full_proj_transform", self.full_proj_transform, "full_proj_transform"),
        ):
            if payload is not None and (
                _payload_device(payload) is None
                or payload.spec.encoding != encoding
                or tuple(payload.spec.shape) != (4, 4)
            ):
                raise ValueError(f"tier {field_name} does not match its payload contract")


@dataclass(frozen=True, slots=True)
class PreparedSourceLease:
    """Immutable source preparation with transparent borrowed payload refs.

    ``BufferRef.data`` is exposed unchanged for zero-copy interoperability.
    Consumers must not mutate it while the lease is live; ``owner_token``
    keeps producer-owned storage alive for that interval.
    """

    source_identity: SourceIdentity
    modality_eligibility: ModalityEligibility
    source_metadata: Mapping[str, Any]
    tier_preparation_available: tuple[str, ...]
    borrowed_device_payloads: Mapping[str, Any]
    device: str
    ready_event: Any = field(default=None, compare=False, repr=False)
    generation: int = field(default=0, compare=True)
    lifetime_token: Any = field(default=None, compare=False, repr=False)
    control_token: PreparedIngressControlToken | None = field(default=None, compare=True)
    tier_payloads: Mapping[str, PreparedTierPayload] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.source_identity, SourceIdentity):
            raise TypeError("source_identity must be SourceIdentity")
        if not self.source_identity.source_id.strip():
            raise ValueError("source identity must be non-empty")
        if self.control_token is not None:
            if not isinstance(self.control_token, PreparedIngressControlToken):
                raise TypeError("control_token must be PreparedIngressControlToken or None")
            if self.control_token.source != self.source_identity:
                raise ValueError("control_token source must match source_identity")
        if not isinstance(self.modality_eligibility, ModalityEligibility):
            raise TypeError("modality_eligibility must be ModalityEligibility")
        device = str(self.device)
        if not device:
            raise ValueError("device must be non-empty")
        if isinstance(self.generation, bool) or not isinstance(self.generation, numbers.Integral):
            raise TypeError("generation must be a non-negative integer")
        generation = int(self.generation)
        if generation < 0:
            raise ValueError("generation must be a non-negative integer")
        metadata = dict(self.source_metadata)
        for key, value in metadata.items():
            if not isinstance(key, str) or not key:
                raise TypeError("source metadata keys must be non-empty strings")
            metadata[key] = _immutable_metadata(value)
        tiers = tuple(self.tier_preparation_available)
        if any(not isinstance(tier, str) or not tier for tier in tiers):
            raise ValueError("tier preparation names must be non-empty strings")
        if len(set(tiers)) != len(tiers):
            raise ValueError("tier preparation names must be unique")
        payloads = dict(self.borrowed_device_payloads)
        if not payloads:
            raise ValueError("at least one borrowed device payload is required")
        for name, payload in payloads.items():
            if not isinstance(name, str) or not name:
                raise TypeError("payload names must be non-empty strings")
            if _payload_device(payload) != device:
                raise ValueError("all prepared payloads must be on the lease device")
        tier_payloads = dict(self.tier_payloads)
        if tier_payloads and set(tier_payloads) != set(tiers):
            raise ValueError("tier payloads must match advertised tier preparation")
        for name, payload in tier_payloads.items():
            if not isinstance(payload, PreparedTierPayload) or payload.name != name:
                raise TypeError("tier payload mapping must contain matching PreparedTierPayload values")
            if any(
                _payload_device(value) != device
                for value in (
                    payload.rgb,
                    payload.depth,
                    payload.camera_intrinsics,
                    *(() if payload.depth_confidence is None else (payload.depth_confidence,)),
                    *(() if payload.projection_matrix is None else (payload.projection_matrix,)),
                    *(() if payload.full_proj_transform is None else (payload.full_proj_transform,)),
                )
            ):
                raise ValueError("all prepared tier payloads must be on the lease device")
        event = self.ready_event
        if event is not None and not callable(getattr(event, "query", None)):
            raise TypeError("ready_event must provide a callable query() method")
        lifetime_token = self.lifetime_token
        if lifetime_token is None:
            lifetime_token = next(
                payload.owner_token for payload in payloads.values()
            )
        object.__setattr__(self, "device", device)
        object.__setattr__(self, "source_metadata", MappingProxyType(metadata))
        object.__setattr__(self, "tier_preparation_available", tiers)
        object.__setattr__(self, "borrowed_device_payloads", MappingProxyType(payloads))
        object.__setattr__(self, "tier_payloads", MappingProxyType(tier_payloads))
        object.__setattr__(self, "generation", generation)
        object.__setattr__(self, "lifetime_token", lifetime_token)

    @property
    def available_tiers(self) -> tuple[str, ...]:
        return self.tier_preparation_available

    @property
    def payloads(self) -> Mapping[str, Any]:
        return self.borrowed_device_payloads

    def tier_payload(self, name: str) -> PreparedTierPayload:
        return self.tier_payloads[name]

    def is_ready(self) -> bool:
        """Poll preparation completion without waiting on the host."""

        event = self.ready_event
        if event is None:
            return True
        return bool(event.query())

    def handoff_to_compute(self, transfer_owner: Any) -> None:
        """Insert the preparation dependency when a consumer binds this lease.

        Producers must remain independent of the eventual compute stream.  A
        binder/consumer calls this exactly when it is about to consume the
        borrowed device payloads.
        """
        event = self.ready_event
        handoff = getattr(event, "handoff_to_compute", None)
        if callable(handoff):
            handoff(transfer_owner)
            return
        owner_handoff = getattr(transfer_owner, "handoff_to_compute", None)
        if not callable(owner_handoff):
            raise TypeError("transfer_owner must provide handoff_to_compute")
        if event is not None:
            owner_handoff(event)


class PreparedSourceQueue:
    """A bounded FIFO with deterministic, non-blocking offer and poll."""

    def __init__(self, *, max_size: int, max_payload_refs_per_lease: int) -> None:
        if isinstance(max_size, bool) or not isinstance(max_size, numbers.Integral):
            raise TypeError("max_size must be an integer")
        if isinstance(max_payload_refs_per_lease, bool) or not isinstance(
            max_payload_refs_per_lease, numbers.Integral
        ):
            raise TypeError("max_payload_refs_per_lease must be an integer")
        max_size = int(max_size)
        if max_size < 1:
            raise ValueError("max_size must be at least 1")
        max_payload_refs_per_lease = int(max_payload_refs_per_lease)
        if max_payload_refs_per_lease < 1:
            raise ValueError("max_payload_refs_per_lease must be at least 1")
        self.max_size = max_size
        self.max_payload_refs_per_lease = max_payload_refs_per_lease
        self._items: Deque[PreparedSourceLease] = deque()

    def offer(self, lease: PreparedSourceLease) -> bool:
        """Offer once without waiting; this is an SPSC producer operation."""
        if not isinstance(lease, PreparedSourceLease):
            raise TypeError("lease must be PreparedSourceLease")
        if len(lease.borrowed_device_payloads) > self.max_payload_refs_per_lease:
            return False
        if len(self._items) >= self.max_size:
            return False
        self._items.append(lease)
        return True

    try_offer = offer

    def poll(self) -> PreparedSourceLease | None:
        """Poll once without waiting; this is the SPSC consumer operation."""
        try:
            return self._items.popleft()
        except IndexError:
            return None

    try_poll = poll

    def __len__(self) -> int:
        return len(self._items)


__all__ = ["PreparedSourceLease", "PreparedSourceQueue", "PreparedTierPayload"]
