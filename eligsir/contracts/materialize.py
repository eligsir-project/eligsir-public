"""Canonical transport-packet -> compute-frame materialization boundary.

Every materialization keeps ownership metadata attached to the payload via
``BufferRef``. Device moves happen before dtype/unit conversion, so a payload
crosses the host/device boundary at most once and never bounces back through
NumPy/Python.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional, Protocol

import numpy as np

from .runtime import (
    BufferOwnership,
    BufferRef,
    BufferSpec,
    MappingFrame,
    MemoryDomain,
    SensorPacket,
)

__all__ = [
    "ColorPolicy",
    "DepthPolicy",
    "MaterializationPolicy",
    "MaterializationPlan",
    "TensorBackend",
    "NumpyBackend",
    "numpy_backend",
    "torch_backend",
    "torch_gpu_depth_backend",
    "plan_materialization",
    "materialize_mapping_frame",
]

_HOST_DOMAINS = frozenset({MemoryDomain.HOST, MemoryDomain.PINNED_HOST})


class ColorPolicy(str, Enum):
    """Requested color representation at the compute boundary."""

    PRESERVE = "preserve"
    NORMALIZED_FLOAT32 = "normalized_float32"


class DepthPolicy(str, Enum):
    """Requested depth representation at the compute boundary."""

    PRESERVE = "preserve"
    METERS_FLOAT32 = "meters_float32"


@dataclass(frozen=True, slots=True)
class MaterializationPolicy:
    """Destination device and representation policy for one packet."""

    device: str = "cpu"
    color: ColorPolicy = ColorPolicy.PRESERVE
    depth: DepthPolicy = DepthPolicy.PRESERVE

    def __post_init__(self) -> None:
        """Reject an empty destination device name."""
        if not self.device:
            raise ValueError("device must be non-empty")


@dataclass(frozen=True, slots=True)
class MaterializationPlan:
    """Allocation-free description of copies/conversions a packet requires."""

    device: str
    color_copies: int
    depth_copies: int
    color_dtype_change: bool
    depth_dtype_change: bool
    host_device_transfers: int
    would_bounce: bool
    notes: tuple[str, ...] = ()

    @property
    def is_zero_copy(self) -> bool:
        """Return whether materialization needs no copy or host/device move."""
        return (
            self.color_copies == 0
            and self.depth_copies == 0
            and self.host_device_transfers == 0
        )


class TensorBackend(Protocol):
    """Minimal backend used to move and convert opaque tensor payloads."""

    def device_of(self, payload: Any, spec: BufferSpec) -> str:
        """Return the logical device currently holding ``payload``."""
        ...

    def to_device(self, payload: Any, device: str) -> Any:
        """Return ``payload`` resident on ``device``."""
        ...

    def to_float32(self, payload: Any, scale: Optional[float], *, kind: str = "unknown") -> Any:
        """Return float32 payload, optionally divided by ``scale``.

        ``kind`` is ``"color"``, ``"depth"``, or ``"unknown"`` (default, for
        any caller predating this parameter) -- a hint a backend MAY use to
        route the conversion differently (e.g. depth-only GPU-assisted
        conversion, see ``torch_gpu_depth_backend``). Every implementation
        must still produce numerically-identical output regardless of
        ``kind`` when it does not special-case it.
        """
        ...


class NumpyBackend:
    """CPU-only materialization backend for NumPy-compatible payloads."""

    def device_of(self, payload: Any, spec: BufferSpec) -> str:
        """Resolve a NumPy payload to CPU or its explicitly declared device."""
        if spec.device:
            return spec.device
        return "cpu" if spec.memory_domain in _HOST_DOMAINS else "unknown"

    def to_device(self, payload: Any, device: str) -> Any:
        """Keep a payload on CPU and reject unsupported device placement."""
        if device != "cpu":
            raise ValueError(
                f"NumpyBackend cannot place a payload on {device!r}; pass a "
                "device-capable TensorBackend (see torch_backend())"
            )
        return payload

    def to_float32(self, payload: Any, scale: Optional[float], *, kind: str = "unknown") -> Any:
        """Create an owned float32 NumPy representation and apply scaling."""
        out = np.asarray(payload).astype(np.float32, copy=True)
        if scale:
            out /= np.float32(scale)
        return out


def numpy_backend() -> NumpyBackend:
    """Return a fresh CPU-only NumPy materialization backend."""
    return NumpyBackend()


def torch_backend(torch_module: Any = None) -> TensorBackend:
    """Build a torch-backed backend without importing torch at module import."""
    if torch_module is None:  # pragma: no cover
        import torch as torch_module  # type: ignore[no-redef]

    class _TorchBackend:
        """Late-bound torch implementation of :class:`TensorBackend`."""

        def device_of(self, payload: Any, spec: BufferSpec) -> str:
            """Resolve tensor device first, then explicit buffer metadata."""
            device = getattr(payload, "device", None)
            if device is not None:
                return str(device)
            if spec.device:
                return spec.device
            return "cpu" if spec.memory_domain in _HOST_DOMAINS else "unknown"

        def to_device(self, payload: Any, device: str) -> Any:
            """Convert non-tensors once and move tensors only when necessary."""
            tensor = payload
            if not hasattr(tensor, "to"):
                tensor = torch_module.as_tensor(np.asarray(payload))
            if str(getattr(tensor, "device", "")) == device:
                return tensor
            return tensor.to(device)

        def to_float32(self, payload: Any, scale: Optional[float], *, kind: str = "unknown") -> Any:
            """Convert to torch float32 and apply an optional normalization scale."""
            tensor = payload
            if not hasattr(tensor, "to"):
                tensor = torch_module.as_tensor(np.asarray(payload))
            out = tensor.to(torch_module.float32)
            if scale:
                out = out / float(scale)
            return out

    return _TorchBackend()


def torch_gpu_depth_backend(torch_module: Any = None, *, device: str = "cuda") -> TensorBackend:
    """CPU-resident backend whose DEPTH float32/depth_scale conversion runs on ``device``.

    Priority 1 of docs/architecture/2026-08-31-gpu-native-decode-and-compressed-
    ingress.md (issue #1348): "raw depth: direct pinned H2D + HIP/CUDA-side
    depth_scale conversion". This backend is a drop-in ``NumpyBackend``
    replacement -- ``device_of``/``to_device`` behave identically (the
    MappingFrame stays CPU-owned, preserving the "canonical insertion
    MappingFrame must be CPU-owned" contract in
    ``eligsir.contracts.streaming_bridge.mapping_frame_source_arrays`` and
    the single-H2D-owner discipline documented in this module's own docstring
    -- ``DeviceTransferOwner`` remains the sole owner of the *grouped* camera
    H2D that later moves ``cam._sensor_depth_cache`` to the render device).

    Only ``to_float32(..., kind="depth")`` differs: instead of NumPy's
    ``astype(float32)`` + CPU divide, the raw native-dtype array (uint16 for
    TUM/most RGB-D sensors) is pinned, transferred to ``device`` once, divided
    by ``scale`` there, and copied back to a plain (unpinned) CPU float32
    array -- byte-for-byte the same values NumpyBackend would have produced
    (both are a single float32 divide; torch and NumPy agree on this to
    float32 ULP for the raw integer ranges depth sensors produce). RGB
    (``kind="color"``) and any caller that omits ``kind`` are delegated to the
    unmodified NumPy path unchanged -- JPEG/RGB decode is priority 2 of the
    same design doc and is out of scope here.
    """
    if torch_module is None:  # pragma: no cover
        import torch as torch_module  # type: ignore[no-redef]

    numpy_delegate = NumpyBackend()

    class _TorchGpuDepthBackend:
        """NumpyBackend-compatible backend with GPU-assisted depth conversion."""

        def device_of(self, payload: Any, spec: BufferSpec) -> str:
            return numpy_delegate.device_of(payload, spec)

        def to_device(self, payload: Any, device_str: str) -> Any:
            return numpy_delegate.to_device(payload, device_str)

        def to_float32(self, payload: Any, scale: Optional[float], *, kind: str = "unknown") -> Any:
            if kind != "depth":
                return numpy_delegate.to_float32(payload, scale, kind=kind)
            raw = np.asarray(payload)
            tensor = torch_module.from_numpy(np.ascontiguousarray(raw))
            try:
                tensor = tensor.pin_memory()
            except RuntimeError:
                # Pinning can fail under memory pressure; an unpinned H2D is
                # still correct, only slower -- never fail the conversion.
                pass
            on_device = tensor.to(device, non_blocking=True).to(torch_module.float32)
            if scale:
                on_device = on_device / float(scale)
            return on_device.to("cpu").numpy()

    return _TorchGpuDepthBackend()


def _typed_host_payload(ref: BufferRef) -> Any:
    """Expose a host buffer according to its declared ``BufferSpec`` without copying.

    ROS image ingress intentionally hands core a borrowed ``memoryview`` over the
    message storage. ``memoryview`` alone only describes bytes; shape, dtype and
    row stride live in :class:`BufferSpec`. Passing the raw byte view to NumPy or
    torch would therefore reinterpret ``16UC1`` depth byte-wise and can silently
    corrupt metric depth. Reconstruct the declared typed/strided view exactly at
    the materialization boundary.

    Existing ndarray/tensor payloads are already typed and shaped and therefore
    pass through unchanged. Device buffers are never exposed through NumPy here.
    """
    payload = ref.data
    if isinstance(payload, np.ndarray) or ref.spec.memory_domain not in _HOST_DOMAINS:
        return payload
    if hasattr(payload, "device") or hasattr(payload, "to"):
        return payload
    try:
        buffer = memoryview(payload)
    except TypeError:
        return payload
    try:
        return np.ndarray(
            shape=tuple(int(dim) for dim in ref.spec.shape),
            dtype=np.dtype(ref.spec.dtype),
            buffer=buffer,
            strides=(
                tuple(int(stride) for stride in ref.spec.strides_bytes)
                if ref.spec.strides_bytes is not None
                else None
            ),
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "host BufferRef payload does not satisfy its BufferSpec: "
            f"shape={ref.spec.shape!r} dtype={ref.spec.dtype!r} "
            f"strides={ref.spec.strides_bytes!r}"
        ) from exc


def _color_scale(spec: BufferSpec) -> Optional[float]:
    """Return integer color normalization scale, if one is implied by dtype."""
    dtype = str(spec.dtype).lower()
    if dtype in {"uint8", "u1"}:
        return 255.0
    if dtype in {"uint16", "u2"}:
        return 65535.0
    return None


def _depth_scale(spec: BufferSpec, intrinsics: Any) -> Optional[float]:
    """Return source depth scale unless the buffer is already metric."""
    units = (spec.units or "").lower()
    if units in {"m", "metres", "meters"}:
        return None
    scale = getattr(intrinsics, "depth_scale", None)
    if scale in (None, 0):
        return None
    return float(scale)


def _needs_color_conversion(spec: BufferSpec, policy: MaterializationPolicy) -> bool:
    """Return whether color policy requires dtype/value conversion."""
    if policy.color is ColorPolicy.PRESERVE:
        return False
    return str(spec.dtype).lower() != "float32" or _color_scale(spec) is not None


def _needs_depth_conversion(
    spec: BufferSpec, policy: MaterializationPolicy, intrinsics: Any
) -> bool:
    """Return whether depth policy requires dtype/unit conversion."""
    if policy.depth is DepthPolicy.PRESERVE:
        return False
    return str(spec.dtype).lower() != "float32" or _depth_scale(spec, intrinsics) is not None


def plan_materialization(
    packet: SensorPacket,
    policy: Optional[MaterializationPolicy] = None,
    *,
    backend: Optional[TensorBackend] = None,
) -> MaterializationPlan:
    """Describe materialization without allocating anything."""
    policy = policy or MaterializationPolicy()
    backend = backend or numpy_backend()
    notes: list[str] = []
    transfers = 0

    color_on = backend.device_of(packet.rgb.data, packet.rgb.spec)
    color_moves = color_on != policy.device
    if color_moves:
        transfers += 1
        notes.append(f"rgb {color_on}->{policy.device}")

    depth_moves = False
    if packet.depth is not None:
        depth_on = backend.device_of(packet.depth.data, packet.depth.spec)
        depth_moves = depth_on != policy.device
        if depth_moves:
            transfers += 1
            notes.append(f"depth {depth_on}->{policy.device}")
    else:
        notes.append("depth absent")

    color_converts = _needs_color_conversion(packet.rgb.spec, policy)
    depth_converts = bool(
        packet.depth is not None
        and _needs_depth_conversion(packet.depth.spec, policy, packet.intrinsics)
    )
    return MaterializationPlan(
        device=policy.device,
        color_copies=int(color_moves) + int(color_converts),
        depth_copies=int(depth_moves) + int(depth_converts),
        color_dtype_change=color_converts,
        depth_dtype_change=depth_converts,
        host_device_transfers=transfers,
        would_bounce=False,
        notes=tuple(notes),
    )


def materialize_mapping_frame(
    packet: SensorPacket,
    policy: Optional[MaterializationPolicy] = None,
    *,
    backend: Optional[TensorBackend] = None,
) -> MappingFrame:
    """Materialize exactly once, returning per-buffer ownership metadata."""
    policy = policy or MaterializationPolicy()
    backend = backend or numpy_backend()

    rgb_source = _typed_host_payload(packet.rgb)
    rgb_on = backend.device_of(rgb_source, packet.rgb.spec)
    rgb_moved = rgb_on != policy.device
    rgb = backend.to_device(rgb_source, policy.device)
    rgb_converted = _needs_color_conversion(packet.rgb.spec, policy)
    if rgb_converted:
        rgb = backend.to_float32(rgb, _color_scale(packet.rgb.spec), kind="color")
    rgb_ref = _materialized_ref(
        packet.rgb,
        rgb,
        device=policy.device,
        moved=rgb_moved,
        converted=rgb_converted,
        converted_encoding="rgb32f",
        converted_units=None,
    )

    depth_ref: Optional[BufferRef] = None
    if packet.depth is not None:
        depth_source = _typed_host_payload(packet.depth)
        depth_on = backend.device_of(depth_source, packet.depth.spec)
        depth_moved = depth_on != policy.device
        depth = backend.to_device(depth_source, policy.device)
        depth_converted = _needs_depth_conversion(packet.depth.spec, policy, packet.intrinsics)
        if depth_converted:
            depth = backend.to_float32(
                depth, _depth_scale(packet.depth.spec, packet.intrinsics), kind="depth"
            )
        depth_ref = _materialized_ref(
            packet.depth,
            depth,
            device=policy.device,
            moved=depth_moved,
            converted=depth_converted,
            converted_encoding="32FC1",
            converted_units="meters",
        )

    return MappingFrame(
        source=packet.source,
        rgb=rgb_ref,
        depth=depth_ref,
        intrinsics=packet.intrinsics,
        device=policy.device,
        pose=packet.tracking.pose if packet.tracking is not None else None,
        reload_token=packet.reload_token,
    )


def _materialized_ref(
    source: BufferRef,
    payload: Any,
    *,
    device: str,
    moved: bool,
    converted: bool,
    converted_encoding: str,
    converted_units: Optional[str],
) -> BufferRef:
    """Keep lifetime metadata attached to exactly the payload it describes."""
    fresh_allocation = moved or converted
    return BufferRef(
        data=payload,
        spec=_retarget_spec(
            source.spec,
            device,
            converted,
            converted_encoding,
            converted_units,
        ),
        ownership=BufferOwnership.OWNED if fresh_allocation else source.ownership,
        owner_token=None if fresh_allocation else source.owner_token,
    )


def _retarget_spec(
    spec: BufferSpec,
    device: str,
    converted: bool,
    encoding: str,
    units: Optional[str],
) -> BufferSpec:
    """Return metadata describing the destination payload after conversion."""
    domain = spec.memory_domain
    if device == "cpu":
        if domain not in _HOST_DOMAINS:
            domain = MemoryDomain.HOST
        target_device = None
    else:
        if domain in _HOST_DOMAINS:
            domain = MemoryDomain.OTHER_DEVICE
        target_device = device

    return BufferSpec(
        shape=spec.shape,
        dtype="float32" if converted else spec.dtype,
        memory_domain=domain,
        encoding=encoding if converted else spec.encoding,
        units=units if converted else spec.units,
        device=target_device,
        contiguous=True if converted else spec.contiguous,
        strides_bytes=None if converted else spec.strides_bytes,
    )
