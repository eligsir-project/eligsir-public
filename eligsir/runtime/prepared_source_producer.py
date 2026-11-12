"""Standalone raw-frame to ready-source producer.

The producer owns only source preparation.  With the shared
``DeviceTransferOwner`` it submits one packed H2D from the raw RGB/depth pair,
performs metric conversion on the owner's preparation stream, and publishes
an event-gated :class:`PreparedSourceLease`.  It never reads a device tensor
back to the host and has no map or optimizer-version state.  Legacy injected
transfer doubles may still use the former frame-pool staging fallback.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from enum import Enum
import math
import numbers
from typing import Any

import numpy as np
import torch

from eligsir.contracts import (
    BufferOwnership,
    BufferRef,
    BufferSpec,
    MemoryDomain,
    ModalityEligibility,
    SensorPacket,
)
from eligsir.contracts.control import ResolutionTier
from eligsir.mapping.resolution_materialization import resolution_tier_geometry
from eligsir.mapping.scheduler.frame_quality import evaluate_frame_quality
from eligsir.runtime.prepared_source import PreparedSourceLease
from eligsir.runtime.prepared_source import PreparedTierPayload
from eligsir.runtime.prepared_ingress import PreparedIngressControlToken
from eligsir.runtime.ready_source import (
    ReadySourceLane,
    ReadySourceOwner,
    ReplayPriority,
)
from eligsir.runtime.transfers import TransferEvent


# Default prepared tier set when no explicit policy is supplied.
DEFAULT_PREPARED_TIERS = tuple(ResolutionTier)


class PreparedSourceProduceResult(str, Enum):
    """Bounded outcome of one source-preparation offer."""

    ENQUEUED = "enqueued"
    DEFERRED_POOL_EXHAUSTED = "pool_exhausted"
    DROPPED_READY_OWNER = "ready_owner_drop"
    RETAINED_NOT_RESIDENT = "retained_not_resident"
    CAPABILITY_UNAVAILABLE = "capability_unavailable"


@dataclass(slots=True)
class _PreparedSourceLifetime:
    slot: Any | None
    generation: int
    slot_generation: int
    transfer_event: Any = None
    prep_event: Any = None


@dataclass(slots=True)
class _PreparedSubmission:
    """Validated READY reservation and its immutable source facts."""

    packet: SensorPacket
    modality_eligibility: ModalityEligibility
    generation: int
    lifetime: Any
    reservation: Any
    rgb: np.ndarray
    depth: np.ndarray | None
    strict_camera_contract: bool
    neutral_depth: bool
    calibration_attestation: tuple[Any, ...] | None
    native_shape: tuple[int, int]
    native_intrinsics: tuple[float, float, float, float]
    motion_transform: np.ndarray | None
    control_token: PreparedIngressControlToken | None


class _ReadyEvent:
    """Poll-only completion of H2D and destination-device preparation."""

    def __init__(self, prep_event: TransferEvent) -> None:
        self._prep_event = prep_event

    def query(self) -> bool:
        # The final preparation event is recorded after H2D, calibration and
        # every tier kernel on one owner stream, so it transitively fences the
        # copy without a second readiness query.
        return bool(self._prep_event.query())

    def handoff_to_compute(self, transfer_owner: Any) -> None:
        """Insert the prep-stream dependency on the consumer compute stream."""
        handoff = getattr(transfer_owner, "handoff_to_compute", None)
        if not callable(handoff):
            raise TypeError("transfer_owner must provide handoff_to_compute")
        handoff(self._prep_event)


def _exact_generation(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise TypeError("generation must be a non-negative integer")
    value = int(value)
    if value < 0:
        raise ValueError("generation must be a non-negative integer")
    return value


def _same_device(expected: torch.device, actual: torch.device) -> bool:
    expected, actual = map(torch.device, (expected, actual))
    return expected.type == actual.type and (
        expected.index is None or expected.index == actual.index
    )


class PreparedSourceProducer:
    """Prepare raw frames and publish complete leases to ``ReadySourceOwner``."""

    def __init__(
        self,
        ready_owner: ReadySourceOwner,
        *,
        transfer_owner: Any,
        frame_pool: Any | None = None,
        producer_owner: Any | None = None,
        resampling_mode: str = "legacy",
        depth_downsample: str = "nearest",
        prepared_tier_policy: Any | None = None,
        measure_replay_quality: bool = False,
        motion_depth_confidence_mode: str = "off",
        motion_depth_confidence_native_tolerance_px: float = 1.0,
        motion_depth_confidence_min_weight: float = 0.05,
    ) -> None:
        if not isinstance(ready_owner, ReadySourceOwner):
            raise TypeError("ready_owner must be ReadySourceOwner")
        owner_prepare = getattr(transfer_owner, "prepare_source", None)
        if not callable(owner_prepare) and not callable(
            getattr(transfer_owner, "upload_packed", None)
        ):
            raise TypeError("transfer_owner must provide prepare_source or upload_packed")
        if not callable(owner_prepare) and frame_pool is None:
            raise TypeError("legacy transfer owners require frame_pool")
        owner_prepare_available = callable(owner_prepare)
        if not owner_prepare_available and frame_pool is not None and not callable(
            getattr(frame_pool, "acquire", None)
        ):
            raise TypeError("frame_pool must provide acquire")
        if (
            not owner_prepare_available
            and frame_pool is not None
            and not callable(getattr(frame_pool, "release_after", None))
        ):
            raise TypeError("frame_pool must provide release_after")
        from eligsir.contracts.control import PreparedTierPolicy
        from eligsir.mapping.resolution_materialization import (
            DEPTH_DOWNSAMPLE_MODES,
            RESAMPLING_MODES,
        )

        if resampling_mode not in RESAMPLING_MODES:
            raise ValueError(f"resampling_mode must be one of {RESAMPLING_MODES!r}")
        if depth_downsample not in DEPTH_DOWNSAMPLE_MODES:
            raise ValueError(f"depth_downsample must be one of {DEPTH_DOWNSAMPLE_MODES!r}")
        # Default prepared tier policy materializes every legal adaptive tier.
        # to preserve backward compatibility with existing tests and behaviors.
        if prepared_tier_policy is None:
            prepared_tier_policy = PreparedTierPolicy.from_supported_tiers(
                DEFAULT_PREPARED_TIERS
            )
        if not isinstance(prepared_tier_policy, PreparedTierPolicy):
            raise TypeError("prepared_tier_policy must be PreparedTierPolicy or None")
        if not isinstance(measure_replay_quality, bool):
            raise TypeError("measure_replay_quality must be bool")
        if motion_depth_confidence_mode not in {"off", "identity", "motion"}:
            raise ValueError("motion_depth_confidence_mode must be off, identity, or motion")
        if (
            isinstance(motion_depth_confidence_native_tolerance_px, bool)
            or not isinstance(motion_depth_confidence_native_tolerance_px, (int, float))
            or not math.isfinite(motion_depth_confidence_native_tolerance_px)
            or motion_depth_confidence_native_tolerance_px <= 0
        ):
            raise ValueError("motion_depth_confidence_native_tolerance_px must be positive")
        if (
            isinstance(motion_depth_confidence_min_weight, bool)
            or not isinstance(motion_depth_confidence_min_weight, (int, float))
            or not math.isfinite(motion_depth_confidence_min_weight)
            or not 0 <= motion_depth_confidence_min_weight <= 1
        ):
            raise ValueError("motion_depth_confidence_min_weight must be in [0, 1]")
        self._ready_producer = ready_owner.claim_producer(producer_owner)
        self._transfer_owner = transfer_owner
        self._frame_pool = frame_pool
        self._owner_prepare = owner_prepare if callable(owner_prepare) else None
        self._resampling_mode = resampling_mode
        self._depth_downsample = depth_downsample
        self._prepared_tier_policy = prepared_tier_policy
        self._measure_replay_quality = measure_replay_quality
        self._motion_depth_confidence_mode = motion_depth_confidence_mode
        self._motion_depth_confidence_native_tolerance_px = float(
            motion_depth_confidence_native_tolerance_px
        )
        self._motion_depth_confidence_min_weight = float(motion_depth_confidence_min_weight)

    @property
    def ready_owner(self) -> ReadySourceOwner:
        """Return the single reservoir that owns this producer's leases."""

        return self._ready_producer._owner

    def abandon_pending(self, *, reason: str) -> None:
        self._ready_producer.abandon_pending(reason=reason)

    @property
    def device(self) -> torch.device:
        return torch.device(getattr(self._transfer_owner, "device", "cpu"))

    @staticmethod
    def _raw_array(ref: BufferRef, name: str, *, ndim: int) -> np.ndarray:
        if not isinstance(ref, BufferRef):
            raise TypeError(f"{name} must be a BufferRef")
        if not isinstance(ref.data, np.ndarray):
            raise TypeError(f"{name} must contain a NumPy array")
        array = np.asarray(ref.data)
        if array.ndim != ndim or any(int(dim) <= 0 for dim in array.shape):
            raise ValueError(f"{name} must have a non-empty rank-{ndim} array")
        return np.ascontiguousarray(array)

    @staticmethod
    def _raw_depth_scale(packet: SensorPacket) -> float | None:
        units = str(getattr(packet.depth.spec, "units", "") or "").lower()
        if units in {"m", "metres", "meters"}:
            return None
        scale = getattr(packet.intrinsics, "depth_scale", None)
        if scale is None or not np.isfinite(float(scale)) or float(scale) <= 0.0:
            raise ValueError("raw depth requires a positive finite intrinsics depth_scale")
        return float(scale)

    @classmethod
    def _metric_depth(cls, depth: torch.Tensor, packet: SensorPacket) -> torch.Tensor:
        metric = depth.to(dtype=torch.float32)
        scale = cls._raw_depth_scale(packet)
        return metric if scale is None else metric / scale

    @classmethod
    def _replay_quality(
        cls,
        rgb: np.ndarray,
        depth: np.ndarray | None,
        packet: SensorPacket,
    ) -> tuple[float | None, float | None]:
        """Return independent source-quality proxies from decoded CPU pixels."""

        metric_depth = None
        if depth is not None:
            scale = cls._raw_depth_scale(packet)
            metric_depth = np.asarray(depth, dtype=np.float32)
            if scale is not None:
                metric_depth = metric_depth / scale
        quality = evaluate_frame_quality(rgb, metric_depth)

        def bounded(value: float) -> float | None:
            return float(value) if np.isfinite(value) and 0.0 <= value <= 1.0 else None

        rgb_quality = bounded(
            quality.blur_quality_weight * quality.exposure_quality_weight
        )
        depth_quality = (
            None
            if depth is None
            else bounded(quality.depth_valid_ratio * (1.0 - quality.depth_edge_invalid_ratio))
        )
        return rgb_quality, depth_quality

    @staticmethod
    def _validate_camera_contract(
        packet: SensorPacket,
        rgb: np.ndarray,
        depth: np.ndarray | None,
        *,
        allow_neutral_depth: bool = False,
    ) -> None:
        """Reject packets that cannot satisfy the ready-camera contract.

        Validation is deliberately before the transfer submission.  A ready
        lease must contain native RGB-D geometry, calibrated units, and a
        tracked camera pose; silently publishing a partial lease would force
        the binder to invent one of those facts later.
        """

        # Strict prepared RGB is the canonical decoded uint8 source domain.
        # This CPU-side type contract is what makes the later /255 conversion
        # finite and bounded without re-reducing every prepared GPU tensor at
        # each READY replay bind.
        if rgb.dtype != np.uint8:
            raise ValueError("strict prepared RGB must be canonical uint8")
        intrinsics = packet.intrinsics
        if (
            rgb.shape[:2] != (int(intrinsics.height), int(intrinsics.width))
            or int(intrinsics.width) <= 0
            or int(intrinsics.height) <= 0
            or not np.isfinite(float(intrinsics.fx))
            or not np.isfinite(float(intrinsics.fy))
            or not np.isfinite(float(intrinsics.cx))
            or not np.isfinite(float(intrinsics.cy))
            or float(intrinsics.fx) <= 0.0
            or float(intrinsics.fy) <= 0.0
        ):
            raise ValueError("camera intrinsics do not match native RGB geometry")
        if depth is None and not allow_neutral_depth:
            raise ValueError("prepared source requires aligned depth")
        if depth is not None and depth.shape != rgb.shape[:2]:
            raise ValueError("prepared depth must be aligned to RGB")
        if depth is not None and (
            int(intrinsics.depth_width) != int(intrinsics.width)
            or int(intrinsics.depth_height) != int(intrinsics.height)
        ):
            raise ValueError("prepared depth must be aligned to RGB intrinsics")
        if depth is not None and (
            not np.isfinite(float(intrinsics.depth_scale))
            or float(intrinsics.depth_scale) <= 0.0
        ):
            raise ValueError("raw depth requires a positive finite intrinsics depth_scale")
        tracking = packet.tracking
        if tracking is None or not tracking.usable or tracking.pose is None:
            raise ValueError("prepared source requires a usable tracked pose")
        pose = np.asarray(tracking.pose.matrix)
        if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
            raise ValueError("tracked pose must be a finite 4x4 c2w matrix")

    @staticmethod
    def _device_event(device: torch.device) -> TransferEvent:
        if device.type != "cuda":
            return TransferEvent()
        event = torch.cuda.Event(blocking=False)
        event.record(torch.cuda.current_stream(device))
        return TransferEvent(event)

    @staticmethod
    def _payload_ref(
        value: torch.Tensor,
        *,
        encoding: str,
        owner_token: Any,
        units: str | None = None,
    ) -> BufferRef:
        return BufferRef(
            data=value,
            spec=BufferSpec(
                shape=tuple(int(dim) for dim in value.shape),
                dtype=str(value.dtype).removeprefix("torch."),
                memory_domain=MemoryDomain.OTHER_DEVICE,
                encoding=encoding,
                units=units,
                device=str(value.device),
                contiguous=bool(value.is_contiguous()),
            ),
            ownership=BufferOwnership.BORROWED,
            owner_token=owner_token,
        )

    @staticmethod
    def _source_metadata(
        packet: SensorPacket,
        rgb: np.ndarray,
        depth: np.ndarray | None,
        *,
        strict: bool,
        neutral_depth: bool = False,
        calibration_attestation: tuple[Any, ...] | None = None,
    ) -> dict[str, Any]:
        source = packet.source
        metadata: dict[str, Any] = {
            "source_id": source.source_id,
            "stream_epoch": source.stream_epoch,
            "source_ordinal": source.source_ordinal,
            "timestamp_ns": source.timestamp_ns,
            "rgb_shape": tuple(int(dim) for dim in rgb.shape),
            "depth_shape": (
                tuple(int(dim) for dim in rgb.shape[:2])
                if strict and neutral_depth
                else (None if depth is None else tuple(int(dim) for dim in depth.shape))
            ),
            "depth_units": "m" if strict else (None if packet.depth is None else str(packet.depth.spec.units or "")),
        }
        if strict:
            if calibration_attestation is None:
                raise ValueError("strict prepared source requires calibration attestation")
            metadata["depth_aligned_to_rgb"] = True
            if neutral_depth:
                metadata["prepared_neutral_depth"] = True
            metadata["camera_intrinsics"] = tuple(
                getattr(packet.intrinsics, item.name)
                for item in fields(packet.intrinsics)
            )
        else:
            metadata["intrinsics"] = (
                float(packet.intrinsics.fx),
                float(packet.intrinsics.fy),
                float(packet.intrinsics.cx),
                float(packet.intrinsics.cy),
            )
        tracking = packet.tracking
        if strict:
            metadata["pose_c2w"] = calibration_attestation[1]
        elif tracking is not None and tracking.pose is not None:
            metadata["pose_c2w"] = tuple(
                tuple(float(value) for value in row)
                for row in np.asarray(tracking.pose.matrix)
            )
        if strict:
            metadata["calibration_attestation"] = calibration_attestation
        return metadata

    @staticmethod
    def _calibration_attestation(packet: SensorPacket) -> tuple[Any, ...]:
        """Freeze the CPU calibration facts that create the ready payloads."""

        pose = tuple(
            tuple(float(value) for value in row)
            for row in np.asarray(packet.tracking.pose.matrix, dtype=np.float32)
        )
        intrinsics = packet.intrinsics
        camera_intrinsics = (
            (float(intrinsics.fx), 0.0, float(intrinsics.cx)),
            (0.0, float(intrinsics.fy), float(intrinsics.cy)),
            (0.0, 0.0, 1.0),
        )
        return ("prepared-calibration-v1", pose, camera_intrinsics)

    @staticmethod
    def _inverse_pose_host(pose: Any) -> np.ndarray:
        pose_host = np.asarray(pose, dtype=np.float32)
        try:
            w2c_host = np.linalg.inv(pose_host)
        except (np.linalg.LinAlgError, ValueError) as exc:
            raise ValueError("prepared source pose must be invertible") from exc
        if not np.all(np.isfinite(w2c_host)):
            raise ValueError("prepared source pose inverse must be finite")
        return w2c_host

    def _motion_transform_host(self, packet: SensorPacket) -> np.ndarray | None:
        if self._motion_depth_confidence_mode != "motion" or packet.tracking is None:
            return None
        diagnostics = packet.tracking.diagnostics
        if (
            diagnostics.get("motion_depth_confidence_source") != "offline_tum"
            or diagnostics.get("motion_depth_confidence_reason") != "interpolated"
        ):
            return None
        raw_transform = diagnostics.get("motion_depth_confidence_transform")
        if raw_transform is None:
            return None
        transform = np.asarray(raw_transform, dtype=np.float32)
        if transform.shape != (4, 4) or not np.isfinite(transform).all():
            return None
        return np.ascontiguousarray(transform)

    def _motion_tolerance(
        self,
        *,
        native_shape: tuple[int, int],
        native_intrinsics: tuple[float, float, float, float],
        tier: ResolutionTier,
    ) -> tuple[float, float]:
        width, height, *_ = resolution_tier_geometry(
            native_width=native_shape[1], native_height=native_shape[0],
            fx=native_intrinsics[0], fy=native_intrinsics[1],
            cx=native_intrinsics[2], cy=native_intrinsics[3],
            tier=tier,
            resampling_mode=self._resampling_mode,
        )
        if self._resampling_mode == "ray_preserving":
            scale_x = scale_y = 1.0 / tier.downsample_factor
        else:
            scale_x, scale_y = width / native_shape[1], height / native_shape[0]
        return (
            self._motion_depth_confidence_native_tolerance_px * scale_x,
            self._motion_depth_confidence_native_tolerance_px * scale_y,
        )

    def _validate_motion_tolerance(
        self,
        *,
        native_shape: tuple[int, int],
        native_intrinsics: tuple[float, float, float, float],
    ) -> None:
        limits = torch.finfo(torch.float32)
        for tier in self._prepared_tier_policy.declared_tiers:
            if not all(
                limits.tiny <= tolerance <= limits.max
                for tolerance in self._motion_tolerance(
                    native_shape=native_shape, native_intrinsics=native_intrinsics, tier=tier
                )
            ):
                raise ValueError("motion depth confidence tolerance is outside prepared float32 tier range")

    def _prepare_source_bundle(
        self,
        rgb: np.ndarray,
        depth: np.ndarray | None,
        packet: SensorPacket,
        calibration_attestation: tuple[Any, ...],
        motion_transform: np.ndarray | None,
    ) -> tuple[tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor, torch.Tensor], Any, Any] | None:
        """Co-pack pixels and immutable calibration into one owner H2D.

        Production ``DeviceTransferOwner`` exposes both a packed upload and its
        preparation stream. Small injected test/legacy owners may not; those
        retain the established two-stage compatibility path below.
        """

        upload = getattr(self._transfer_owner, "upload_packed", None)
        stream = getattr(self._transfer_owner, "transfer_stream", None)
        prepare = getattr(stream, "prepare", None)
        if not callable(upload) or not callable(prepare):
            return None
        _schema, pose, rows = calibration_attestation
        pose_host = np.asarray(pose, dtype=np.float32)
        w2c_host = self._inverse_pose_host(pose_host)
        intrinsics_host = np.asarray(rows, dtype=np.float32)
        arrays = (rgb, depth, pose_host, intrinsics_host, w2c_host)
        if motion_transform is not None:
            arrays += (motion_transform,)
        uploaded, transfer_event = upload(
            arrays,
            purpose="prepared_source",
            handoff_to_compute=False,
        )
        if not isinstance(uploaded, tuple) or len(uploaded) not in (5, 6):
            raise TypeError("packed prepared source must return RGB/depth/c2w/K/w2c tensors")
        rgb_raw, depth_raw, c2w, camera_intrinsics, w2c, *motion = uploaded

        def _convert_pixels():
            rgb_device = rgb_raw.to(dtype=torch.float32).div(255.0)
            depth_device = (
                None if depth_raw is None else self._metric_depth(depth_raw, packet)
            )
            return rgb_device, depth_device

        (rgb_device, depth_device), prep_event = prepare(_convert_pixels)
        return (
            (
                rgb_device,
                depth_device,
                c2w,
                camera_intrinsics,
                w2c,
                *(motion or (None,)),
            ),
            transfer_event,
            prep_event,
        )

    def _prepare_calibration_payloads(
        self, calibration_attestation: tuple[Any, ...]
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], Any | None]:
        """Compatibility path for transfer doubles without packed calibration."""

        _schema, pose, rows = calibration_attestation
        prepare = getattr(self._transfer_owner, "prepare_ready_calibration", None)
        if not callable(prepare):
            raise TypeError("strict prepared source requires sealed calibration preparation")
        payloads, event = prepare(pose=pose, camera_intrinsics=rows)
        if (
            not isinstance(payloads, tuple)
            or len(payloads) != 2
            or not all(isinstance(item, torch.Tensor) for item in payloads)
        ):
            raise TypeError("prepared calibration must return c2w and camera intrinsics tensors")
        if event is not None and not callable(getattr(event, "query", None)):
            raise TypeError("prepared calibration must return a preparation completion event")
        c2w, camera_intrinsics = payloads
        if not _same_device(self.device, c2w.device) or not _same_device(
            self.device, camera_intrinsics.device
        ):
            raise ValueError("prepared calibration payloads must remain on the producer device")
        return payloads, event

    def _prepare_tier_payloads(
        self,
        rgb: torch.Tensor,
        depth: torch.Tensor | None,
        camera_intrinsics: torch.Tensor,
        *,
        fallback_event: Any,
    ) -> tuple[dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]], Any]:
        """Prepare declared tiers before a lease can become READY.

        The set of tiers materialized is determined by the prepared_tier_policy
        resolved at construction time. No lazy tier creation after READY.
        """

        prepare = getattr(self._transfer_owner, "prepare_source_tiers", None)
        if callable(prepare):
            payloads, event = prepare(
                rgb,
                depth,
                camera_intrinsics,
                resampling_mode=self._resampling_mode,
                depth_downsample=self._depth_downsample,
                declared_tiers=self._prepared_tier_policy.declared_tiers,
            )
        else:
            # Older CPU doubles do not own a stream.  Production transfer owners
            # always take the stream-ordered branch above.
            from eligsir.mapping.resolution_materialization import materialize_resolution_tensors

            payloads = {
                tier.value: materialize_resolution_tensors(
                    rgb,
                    depth,
                    camera_intrinsics,
                    tier,
                    resampling_mode=self._resampling_mode,
                    depth_downsample=self._depth_downsample,
                )
                for tier in self._prepared_tier_policy.declared_tiers
            }
            event = fallback_event
        declared_tier_names = {tier.value for tier in self._prepared_tier_policy.declared_tiers}
        if set(payloads) != declared_tier_names:
            raise TypeError(
                f"prepared tier derivation must return exactly {declared_tier_names!r}, "
                f"got {set(payloads)!r}"
            )
        if not callable(getattr(event, "query", None)):
            raise TypeError("prepared tier derivation must return a completion event")
        return payloads, event

    def _motion_depth_confidence(
        self,
        tiers: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]],
        transform: torch.Tensor | None,
        *,
        fallback_event: Any,
        original_depth_present: bool,
        native_shape: tuple[int, int],
        native_intrinsics: tuple[float, float, float, float],
    ) -> tuple[dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]], Any]:
        if self._motion_depth_confidence_mode == "off" or not original_depth_present:
            return tiers, fallback_event
        from eligsir.mapping.motion_depth_confidence import motion_depth_confidence

        def prepare():
            updated = {}
            for name, (rgb, depth, intrinsics, existing) in tiers.items():
                if depth is None:
                    updated[name] = (rgb, depth, intrinsics, existing)
                    continue
                if self._motion_depth_confidence_mode == "identity" or transform is None:
                    updated[name] = (
                        rgb,
                        depth,
                        intrinsics,
                        torch.ones_like(depth) if existing is None else existing,
                    )
                    continue
                confidence = motion_depth_confidence(
                    depth,
                    intrinsics.to(dtype=depth.dtype),
                    transform.to(dtype=depth.dtype, device=depth.device),
                    tau_px=self._motion_tolerance(
                        native_shape=native_shape,
                        native_intrinsics=native_intrinsics,
                        tier=ResolutionTier(name),
                    ),
                    w_min=self._motion_depth_confidence_min_weight,
                )
                updated[name] = (
                    rgb,
                    depth,
                    intrinsics,
                    confidence if existing is None else existing * confidence,
                )
            return updated

        stream_prepare = getattr(
            getattr(self._transfer_owner, "transfer_stream", None), "prepare", None
        )
        if callable(stream_prepare):
            return stream_prepare(prepare)
        return prepare(), fallback_event

    def _prepare_camera_geometry(
        self,
        c2w: torch.Tensor,
        tiers: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]],
        *,
        fallback_event: Any,
        w2c: torch.Tensor | None = None,
    ) -> tuple[
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None,
        dict[str, tuple[torch.Tensor, torch.Tensor]],
        Any,
    ]:
        """Ask the transfer owner to seal reusable camera geometry before READY."""

        prepare = getattr(self._transfer_owner, "prepare_source_camera_geometry", None)
        if not callable(prepare):
            return None, {}, fallback_event
        common, per_tier, event = prepare(c2w, tiers, w2c=w2c)
        if (
            not isinstance(common, tuple)
            or len(common) != 4
            or not all(isinstance(value, torch.Tensor) for value in common)
            or set(per_tier) != set(tiers)
            or not callable(getattr(event, "query", None))
        ):
            raise TypeError("prepared camera geometry must return complete device payloads and an event")
        return common, per_tier, event

    def _release_now(self, slot: Any) -> None:
        release = getattr(self._frame_pool, "release_now", None)
        if callable(release):
            release(slot)
        else:
            self._frame_pool.release(slot)

    def _preflight(
        self,
        packet: SensorPacket,
        *,
        modality_eligibility: ModalityEligibility,
        lane: str,
        arrival_ordinal: int | None,
        priority: int | str | ReplayPriority | None,
        generation: int,
        lifetime_token: Any | None,
        control_token: PreparedIngressControlToken | None,
        durable_retained: bool,
    ) -> _PreparedSubmission | PreparedSourceProduceResult:
        """Validate and reserve one source before any source payload is uploaded."""

        if not isinstance(packet, SensorPacket):
            raise TypeError("packet must be SensorPacket")
        if not isinstance(modality_eligibility, ModalityEligibility):
            raise TypeError("modality_eligibility must be ModalityEligibility")
        if control_token is not None:
            if not isinstance(control_token, PreparedIngressControlToken):
                raise TypeError("control_token must be PreparedIngressControlToken or None")
            if control_token.source != packet.source:
                raise ValueError("control_token source must match packet source")
        if not isinstance(durable_retained, bool):
            raise TypeError("durable_retained must be bool")
        generation = _exact_generation(generation)
        rgb = self._raw_array(packet.rgb, "RGB", ndim=3)
        depth = None if packet.depth is None else self._raw_array(packet.depth, "depth", ndim=2)
        strict_camera_contract = self._owner_prepare is not None
        neutral_depth = depth is None
        if neutral_depth and (
            not strict_camera_contract
            or not modality_eligibility.rgb_supervision
            or modality_eligibility.depth_supervision
            or modality_eligibility.depth_insertion
            or modality_eligibility.geometry_population_target
        ):
            return PreparedSourceProduceResult.CAPABILITY_UNAVAILABLE
        calibration_attestation = None
        if strict_camera_contract:
            self._validate_camera_contract(
                packet, rgb, depth, allow_neutral_depth=neutral_depth
            )
            calibration_attestation = self._calibration_attestation(packet)
        native_shape = (int(rgb.shape[0]), int(rgb.shape[1]))
        native_intrinsics = (
            float(packet.intrinsics.fx),
            float(packet.intrinsics.fy),
            float(packet.intrinsics.cx),
            float(packet.intrinsics.cy),
        )
        if self._motion_depth_confidence_mode != "off" and not neutral_depth:
            self._validate_motion_tolerance(
                native_shape=native_shape, native_intrinsics=native_intrinsics
            )
        lifetime = (
            lifetime_token
            if lifetime_token is not None
            else _PreparedSourceLifetime(
                slot=None,
                generation=generation,
                slot_generation=0,
                transfer_event=None,
            )
        )
        reservation = self._ready_producer.reserve(
            packet.source,
            lane=lane,
            arrival_ordinal=arrival_ordinal,
            priority=priority,
            generation=generation,
            lifetime_token=lifetime,
            full_is_nonterminal=durable_retained,
            control_token=control_token,
        )
        if reservation is None:
            return (
                PreparedSourceProduceResult.RETAINED_NOT_RESIDENT
                if durable_retained
                else PreparedSourceProduceResult.DROPPED_READY_OWNER
            )
        try:
            motion_transform = (
                None if neutral_depth else self._motion_transform_host(packet)
            )
        except Exception:
            reservation.abort()
            raise
        return _PreparedSubmission(
            packet=packet,
            modality_eligibility=modality_eligibility,
            generation=generation,
            lifetime=lifetime,
            reservation=reservation,
            rgb=rgb,
            depth=depth,
            strict_camera_contract=strict_camera_contract,
            neutral_depth=neutral_depth,
            calibration_attestation=calibration_attestation,
            native_shape=native_shape,
            native_intrinsics=native_intrinsics,
            motion_transform=motion_transform,
            control_token=control_token,
        )

    def submit_many(
        self,
        packets,
        *,
        modality_eligibilities,
        lanes,
        arrival_ordinals,
        priorities,
        generations,
        control_tokens,
        durable_retained,
    ) -> tuple[PreparedSourceProduceResult, ...]:
        """Prepare compatible sources through one transient packed H2D."""
        packets = tuple(packets)
        arguments = tuple(
            zip(
                packets,
                modality_eligibilities,
                lanes,
                arrival_ordinals,
                priorities,
                generations,
                control_tokens,
                durable_retained,
            )
        )
        if not packets or len(arguments) != len(packets):
            raise ValueError("prepared source batch arguments must be non-empty and equal length")
        submissions: list[_PreparedSubmission | None] = [None] * len(packets)
        results: list[PreparedSourceProduceResult | None] = [None] * len(packets)
        try:
            for index, (
                packet, eligibility, lane, ordinal, priority, generation, token, retained
            ) in enumerate(arguments):
                preflight = self._preflight(
                    packet,
                    modality_eligibility=eligibility,
                    lane=lane,
                    arrival_ordinal=ordinal,
                    priority=priority,
                    generation=generation,
                    lifetime_token=None,
                    control_token=token,
                    durable_retained=retained,
                )
                if isinstance(preflight, PreparedSourceProduceResult):
                    results[index] = preflight
                else:
                    submissions[index] = preflight
        except Exception:
            for submission in submissions:
                if submission is not None:
                    submission.reservation.abort()
            raise
        batch_upload = getattr(self._transfer_owner, "prepare_packed_sources", None)
        stream = getattr(self._transfer_owner, "transfer_stream", None)
        prepare = getattr(stream, "prepare", None)
        batchable = tuple(
            (index, submission)
            for index, submission in enumerate(submissions)
            if submission is not None
            and submission.strict_camera_contract
            and not submission.neutral_depth
            and callable(batch_upload)
            and callable(prepare)
        )
        remaining = tuple(
            (index, submission)
            for index, submission in enumerate(submissions)
            if submission is not None and (index, submission) not in batchable
        )
        if batchable:
            source_fields = tuple(
                (
                    submission.rgb,
                    submission.depth,
                    np.asarray(submission.calibration_attestation[1], dtype=np.float32),
                    np.asarray(submission.calibration_attestation[2], dtype=np.float32),
                    self._inverse_pose_host(submission.calibration_attestation[1]),
                    *((submission.motion_transform,) if submission.motion_transform is not None else ()),
                )
                for _index, submission in batchable
            )
            max_pinned_bytes = getattr(self._transfer_owner, "max_pinned_bytes", None)
            if isinstance(max_pinned_bytes, numbers.Integral) and not isinstance(
                max_pinned_bytes, bool
            ):
                from eligsir.runtime.packed_staging import layout_numpy_fields

                selected: list[tuple[int, _PreparedSubmission]] = []
                selected_fields: list[tuple[np.ndarray, ...]] = []
                for offset, ((index, submission), fields) in enumerate(
                    zip(batchable, source_fields)
                ):
                    _arrays, _layout, source_bytes = layout_numpy_fields(fields)
                    if source_bytes > max_pinned_bytes:
                        submission.reservation.abort()
                        results[index] = PreparedSourceProduceResult.DEFERRED_POOL_EXHAUSTED
                        continue
                    flat = tuple(field for group in (*selected_fields, fields) for field in group)
                    _arrays, _layout, packed_bytes = layout_numpy_fields(flat)
                    if packed_bytes > max_pinned_bytes:
                        for tail_index, tail_submission in batchable[offset:]:
                            tail_submission.reservation.abort()
                            results[tail_index] = PreparedSourceProduceResult.DEFERRED_POOL_EXHAUSTED
                        break
                    selected.append((index, submission))
                    selected_fields.append(fields)
                batchable, source_fields = tuple(selected), tuple(selected_fields)
            if batchable:
                try:
                    uploaded, transfer_event = batch_upload(source_fields, purpose="prepared_source")
                except RuntimeError as error:
                    if str(error) in {
                        "PinnedByteSlabPool is busy",
                        "PinnedByteSlabPool exhausted: in-flight transfer depth exceeds configured slots",
                    }:
                        for index, submission in batchable:
                            submission.reservation.abort()
                            results[index] = PreparedSourceProduceResult.DEFERRED_POOL_EXHAUSTED
                        batchable = ()
                    else:
                        for _index, submission in batchable:
                            submission.reservation.abort()
                        raise
            if batchable:
                def convert():
                    def source_local_depth(raw, packet):
                        metric = self._metric_depth(raw, packet)
                        return metric.clone() if metric is raw else metric

                    return tuple(
                        (
                            fields[0].to(dtype=torch.float32).div(255.0),
                            source_local_depth(fields[1], submission.packet),
                            *(value.clone() for value in fields[2:]),
                            *((None,) if len(fields) == 5 else ()),
                        )
                        for fields, (_index, submission) in zip(uploaded, batchable)
                    )

                try:
                    prepared, prep_event = prepare(convert)
                except Exception:
                    for _index, submission in batchable:
                        submission.reservation.abort()
                    raise
                for bundle, (index, submission) in zip(prepared, batchable):
                    results[index] = self._publish(
                        submission, prepared_bundle=(bundle, transfer_event, prep_event)
                    )
        for index, submission in remaining:
            results[index] = self._publish(submission)
        if any(result is None for result in results):
            raise RuntimeError("prepared source batch did not resolve every source")
        return tuple(results)

    def _publish(
        self,
        submission: _PreparedSubmission,
        *,
        prepared_bundle: tuple[tuple[Any, ...], Any, Any] | None = None,
    ) -> PreparedSourceProduceResult:
        """Publish one preflighted source, aborting only its own reservation on failure."""

        packet = submission.packet
        modality_eligibility = submission.modality_eligibility
        generation = submission.generation
        lifetime = submission.lifetime
        reservation = submission.reservation
        rgb = submission.rgb
        depth = submission.depth
        strict_camera_contract = submission.strict_camera_contract
        neutral_depth = submission.neutral_depth
        calibration_attestation = submission.calibration_attestation
        native_shape = submission.native_shape
        native_intrinsics = submission.native_intrinsics
        motion_transform = submission.motion_transform
        control_token = submission.control_token
        slot = None
        if self._owner_prepare is None:
            try:
                slot = self._frame_pool.acquire()
            except (RuntimeError, OverflowError):
                reservation.abort()
                return PreparedSourceProduceResult.DEFERRED_POOL_EXHAUSTED

        transfer_event = None
        transfer_submitted = False
        slot_released_after = False
        try:
            source_metadata = self._source_metadata(
                packet,
                rgb,
                depth,
                strict=strict_camera_contract,
                neutral_depth=neutral_depth,
                calibration_attestation=calibration_attestation,
            )
            if self._measure_replay_quality:
                source_metadata["replay_quality"] = self._replay_quality(rgb, depth, packet)
            prep_event = None
            c2w_device = None
            intrinsics_device = None
            w2c_device = None
            motion_transform_device = None
            bundled = prepared_bundle if prepared_bundle is not None else (
                None
                if calibration_attestation is None
                else self._prepare_source_bundle(
                    rgb,
                    depth,
                    packet,
                    calibration_attestation,
                    motion_transform,
                )
            )
            if bundled is not None:
                uploaded, transfer_event, prep_event = bundled
                (
                    rgb_raw,
                    depth_raw,
                    c2w_device,
                    intrinsics_device,
                    w2c_device,
                    motion_transform_device,
                ) = uploaded
                transfer_submitted = True
            elif self._owner_prepare is not None:
                prepared = self._owner_prepare(
                    (rgb, depth),
                    purpose="prepared_source",
                    depth_units=None if packet.depth is None else packet.depth.spec.units,
                    depth_scale=getattr(packet.intrinsics, "depth_scale", None),
                )
                if not isinstance(prepared, tuple) or len(prepared) != 3:
                    raise TypeError(
                        "prepared source transfer must return payloads, H2D event, prep event"
                    )
                uploaded, transfer_event, prep_event = prepared
                transfer_submitted = True
            else:
                rgb_host, depth_host = slot.stage(rgb, depth)
                host_arrays = (
                    np.asarray(rgb_host),
                    None if depth_host is None else np.asarray(depth_host),
                )
                uploaded, transfer_event = self._transfer_owner.upload_packed(
                    host_arrays,
                    purpose="prepared_source",
                )
                transfer_submitted = True
            if self._motion_depth_confidence_mode != "off":
                source_metadata["motion_depth_confidence_preparation"] = (
                    "no_depth"
                    if neutral_depth
                    else (
                        "identity"
                        if self._motion_depth_confidence_mode == "identity"
                        else (
                            "neutral_source_fallback"
                            if motion_transform is None
                            else (
                                "qualified_transform"
                                if bundled is not None
                                else "neutral_no_packed_calibration"
                            )
                        )
                    )
                )
            if isinstance(lifetime, _PreparedSourceLifetime):
                lifetime.slot = slot
                lifetime.slot_generation = (
                    int(getattr(slot, "generation", 0)) if slot is not None else 0
                )
                lifetime.transfer_event = transfer_event
                lifetime.prep_event = prep_event
            if not isinstance(uploaded, tuple) or len(uploaded) not in (2, 5, 6):
                raise TypeError("prepared source transfer returned an invalid payload tuple")
            if len(uploaded) == 2:
                rgb_raw, depth_raw = uploaded
            if not isinstance(rgb_raw, torch.Tensor) or (
                depth_raw is not None and not isinstance(depth_raw, torch.Tensor)
            ):
                raise TypeError("prepared source transfer returned invalid tensors")
            if not callable(getattr(transfer_event, "query", None)):
                raise TypeError("prepared source transfer must return a completion event")
            if self._owner_prepare is not None:
                if not callable(getattr(prep_event, "query", None)):
                    raise TypeError("prepared source must return a preparation completion event")
                rgb_device, depth_device = rgb_raw, depth_raw
            else:
                rgb_device = rgb_raw.to(dtype=torch.float32).div(255.0)
                depth_device = (
                    None if depth_raw is None else self._metric_depth(depth_raw, packet)
                )
                prep_event = self._device_event(self.device)
            calibration_event = None
            if strict_camera_contract:
                if c2w_device is None or intrinsics_device is None:
                    calibration_payloads, calibration_event = self._prepare_calibration_payloads(
                        calibration_attestation
                    )
                    c2w_device, intrinsics_device = calibration_payloads
                    if self.device.type == "cpu":
                        w2c_device = torch.from_numpy(
                            self._inverse_pose_host(calibration_attestation[1])
                        )
                else:
                    calibration_event = prep_event
                tiers, tier_event = self._prepare_tier_payloads(
                    rgb_device,
                    depth_device,
                    intrinsics_device,
                    fallback_event=calibration_event,
                )
                tiers, tier_event = self._motion_depth_confidence(
                    tiers,
                    motion_transform_device,
                    fallback_event=tier_event,
                    original_depth_present=not neutral_depth,
                    native_shape=native_shape,
                    native_intrinsics=native_intrinsics,
                )
                common_geometry, tier_geometry, geometry_event = self._prepare_camera_geometry(
                    c2w_device,
                    tiers,
                    fallback_event=tier_event,
                    w2c=w2c_device,
                )
            else:
                tiers, tier_event, common_geometry, tier_geometry, geometry_event = (
                    {},
                    prep_event,
                    None,
                    {},
                    prep_event,
                )
            ready_event = _ReadyEvent(geometry_event if strict_camera_contract else transfer_event)
            geometry_payloads = (
                {}
                if common_geometry is None
                else {
                    name: self._payload_ref(value, encoding=name, owner_token=lifetime)
                    for name, value in zip(
                        (
                            "world_view_transform",
                            "camera_center",
                            "gsplat_viewmat",
                            "gsplat_camera_center",
                        ),
                        common_geometry,
                    )
                }
            )
            tier_payloads = {
                name: PreparedTierPayload(
                    name=name,
                    rgb=self._payload_ref(
                        rgb_tier,
                        encoding="rgb_float32_hwc",
                        owner_token=lifetime,
                    ),
                    depth=self._payload_ref(
                        depth_tier,
                        encoding="depth_meters",
                        units="m",
                        owner_token=lifetime,
                    ),
                    camera_intrinsics=self._payload_ref(
                        intrinsics_tier,
                        encoding="camera_intrinsics",
                        owner_token=lifetime,
                    ),
                    width=int(rgb_tier.shape[1]),
                    height=int(rgb_tier.shape[0]),
                    depth_confidence=(
                        None
                        if confidence is None
                        else self._payload_ref(
                            confidence,
                            encoding="depth_confidence_weight",
                            owner_token=lifetime,
                        )
                    ),
                    intrinsics=(
                        resolution_tier_geometry(
                            native_width=int(rgb.shape[1]),
                            native_height=int(rgb.shape[0]),
                            fx=float(packet.intrinsics.fx),
                            fy=float(packet.intrinsics.fy),
                            cx=float(packet.intrinsics.cx),
                            cy=float(packet.intrinsics.cy),
                            tier=ResolutionTier(name),
                            resampling_mode=self._resampling_mode,
                        )[2:]
                    ),
                    projection_matrix=(
                        None
                        if name not in tier_geometry
                        else self._payload_ref(
                            tier_geometry[name][0],
                            encoding="projection_matrix",
                            owner_token=lifetime,
                        )
                    ),
                    full_proj_transform=(
                        None
                        if name not in tier_geometry
                        else self._payload_ref(
                            tier_geometry[name][1],
                            encoding="full_proj_transform",
                            owner_token=lifetime,
                        )
                    ),
                )
                for name, (rgb_tier, depth_tier, intrinsics_tier, confidence) in tiers.items()
            }
            native_tier = tier_payloads.get("native") if tier_payloads else None
            payloads = {
                "rgb": (
                    native_tier.rgb
                    if native_tier is not None
                    else self._payload_ref(
                        rgb_device,
                        encoding="rgb_float32_hwc",
                        owner_token=lifetime,
                    )
                ),
            }
            if depth_device is not None:
                payloads["depth_meters"] = (
                    native_tier.depth
                    if native_tier is not None
                    else self._payload_ref(
                        depth_device,
                        encoding="depth_meters",
                        units="m",
                        owner_token=lifetime,
                    )
                )
            if strict_camera_contract:
                payloads["c2w"] = self._payload_ref(
                    c2w_device,
                    encoding="c2w",
                    owner_token=lifetime,
                )
                payloads["camera_intrinsics"] = (
                    native_tier.camera_intrinsics
                    if native_tier is not None
                    else self._payload_ref(
                        intrinsics_device,
                        encoding="camera_intrinsics",
                        owner_token=lifetime,
                    )
                )
                payloads.update(geometry_payloads)
            lease = PreparedSourceLease(
                source_identity=packet.source,
                modality_eligibility=modality_eligibility,
                source_metadata=source_metadata,
                tier_preparation_available=tuple(tier_payloads) or ("native",),
                borrowed_device_payloads=payloads,
                device=str(rgb_device.device),
                ready_event=ready_event,
                generation=generation,
                lifetime_token=lifetime,
                control_token=control_token,
                tier_payloads=tier_payloads,
            )
            if slot is not None:
                self._frame_pool.release_after(slot, transfer_event)
                slot_released_after = True
            reservation.commit(lease)
        except Exception as error:
            reservation.abort()
            if transfer_submitted and slot is not None:
                if not slot_released_after:
                    try:
                        self._frame_pool.release_after(slot, transfer_event)
                    except Exception:
                        pass
            elif slot is not None:
                try:
                    self._release_now(slot)
                except Exception:
                    pass
            # Packed-slab admission failed before any H2D. The bridge/promotion
            # owner retains the same item and generation for a bounded retry.
            if not transfer_submitted and isinstance(error, RuntimeError) and str(error) in (
                "PinnedByteSlabPool is busy",
                "PinnedByteSlabPool exhausted: in-flight transfer depth exceeds configured slots",
            ):
                return PreparedSourceProduceResult.DEFERRED_POOL_EXHAUSTED
            raise

        return PreparedSourceProduceResult.ENQUEUED

    def submit(
        self,
        packet: SensorPacket,
        *,
        modality_eligibility: ModalityEligibility,
        lane: str = ReadySourceLane.FRESH,
        arrival_ordinal: int | None = None,
        priority: int | str | ReplayPriority | None = None,
        generation: int = 0,
        lifetime_token: Any | None = None,
        control_token: PreparedIngressControlToken | None = None,
        durable_retained: bool = False,
    ) -> PreparedSourceProduceResult:
        """Prepare one raw packet without waiting for pool or ready capacity."""

        preflight = self._preflight(
            packet,
            modality_eligibility=modality_eligibility,
            lane=lane,
            arrival_ordinal=arrival_ordinal,
            priority=priority,
            generation=generation,
            lifetime_token=lifetime_token,
            control_token=control_token,
            durable_retained=durable_retained,
        )
        return preflight if isinstance(preflight, PreparedSourceProduceResult) else self._publish(preflight)


__all__ = ["PreparedSourceProduceResult", "PreparedSourceProducer"]
