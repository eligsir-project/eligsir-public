"""JIT-only binding from a ready device lease to a canonical camera.

The binder is intentionally opt-in and side-effect-free.  It consumes only an
already-selected READY lease, reuses its borrowed device payloads, and returns a
camera plus the versions sampled by ``ReadyOnlyTickSelector``.  A later runtime
owner may attach that result to scene/replay/observation state after its own
admission fence; this module never mutates those owners itself.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
import math
from typing import Any

import torch

from eligsir.contracts import CameraIntrinsics, SourceIdentity
from eligsir.runtime.prepared_ingress import PreparedIngressControlToken
from eligsir.runtime.prepared_pose_correction import PreparedPoseCorrection
from eligsir.runtime.prepared_source import PreparedSourceLease
from eligsir.runtime.ready_only_tick import ReadySourceBinding, ReadySourceVersions


@dataclass(frozen=True, slots=True)
class PreparedCameraMetadata:
    """Authoritative, transport-neutral facts needed to construct one camera."""

    source_identity: SourceIdentity
    intrinsics: CameraIntrinsics
    pose_c2w: tuple[tuple[float, ...], ...]
    rgb_shape: tuple[int, int, int]
    depth_shape: tuple[int, int]
    depth_units: str
    depth_aligned_to_rgb: bool
    modality_eligibility: Any
    tier: str
    control_token: PreparedIngressControlToken


@dataclass(frozen=True, slots=True)
class ReadyCameraBinding:
    """A camera made from a ready lease with the JIT-sampled version fence."""

    camera: Any
    source_identity: SourceIdentity
    modality_eligibility: Any
    control_token: PreparedIngressControlToken
    versions: ReadySourceVersions
    prepared_tier: str
    metadata: PreparedCameraMetadata
    lease: PreparedSourceLease
    pose_revision: int = 0


def _int_tuple(value: Any, *, name: str, length: int) -> tuple[int, ...]:
    if not isinstance(value, tuple) or len(value) != length:
        raise ValueError(f"{name} must be a {length}-element tuple")
    if any(type(item) is not int or item <= 0 for item in value):
        raise ValueError(f"{name} must contain positive exact integers")
    return value


def _float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be finite")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _metadata(lease: PreparedSourceLease, *, tier: str) -> PreparedCameraMetadata:
    metadata = lease.source_metadata
    source = lease.source_identity
    source_fields = ("source_id", "stream_epoch", "source_ordinal", "timestamp_ns")
    if tuple(metadata.get(name) for name in source_fields) != tuple(
        getattr(source, name) for name in source_fields
    ):
        raise ValueError("source metadata does not match lease SourceIdentity")
    rgb_shape = _int_tuple(metadata.get("rgb_shape"), name="rgb shape", length=3)
    if rgb_shape[2] != 3:
        raise ValueError("RGB shape must have three channels")
    neutral_depth = metadata.get("prepared_neutral_depth", False)
    if type(neutral_depth) is not bool:
        raise ValueError("prepared_neutral_depth must be bool when present")
    if neutral_depth and (
        lease.modality_eligibility.depth_supervision
        or lease.modality_eligibility.depth_insertion
        or lease.modality_eligibility.geometry_population_target
    ):
        raise ValueError("neutral prepared depth cannot enable depth or geometry modalities")
    depth_shape = _int_tuple(metadata.get("depth_shape"), name="depth shape", length=2)
    if depth_shape != rgb_shape[:2]:
        raise ValueError("depth shape must match RGB height and width")
    if metadata.get("depth_units") != "m":
        raise ValueError("prepared depth units must be metres ('m')")
    if metadata.get("depth_aligned_to_rgb") is not True:
        raise ValueError("prepared depth must be aligned to RGB")
    raw_intrinsics = metadata.get("camera_intrinsics")
    names = fields(CameraIntrinsics)
    if not isinstance(raw_intrinsics, tuple) or len(raw_intrinsics) != len(names):
        raise ValueError("camera intrinsics metadata has the wrong field count")
    intrinsics = CameraIntrinsics(
        **dict(zip((item.name for item in names), raw_intrinsics))
    )
    fx, fy, cx, cy = (
        _float(getattr(intrinsics, name), "camera_intrinsics")
        for name in ("fx", "fy", "cx", "cy")
    )
    if not neutral_depth and _float(
        intrinsics.depth_scale, "camera_intrinsics depth_scale"
    ) <= 0.0:
        raise ValueError("camera_intrinsics depth_scale must be positive")
    width, height = _int_tuple(
        (intrinsics.width, intrinsics.height), name="camera_intrinsics size", length=2
    )
    if (
        fx <= 0.0
        or fy <= 0.0
        or (width, height) != (rgb_shape[1], rgb_shape[0])
    ):
        raise ValueError("camera intrinsics do not match native RGB geometry")
    pose = metadata.get("pose_c2w")
    if not isinstance(pose, tuple) or len(pose) != 4:
        raise ValueError("pose_c2w must be a 4x4 tuple")
    pose_c2w = tuple(tuple(_float(item, "pose_c2w") for item in row) for row in pose)
    if any(len(row) != 4 for row in pose_c2w):
        raise ValueError("pose_c2w must be a 4x4 tuple")
    expected_attestation = (
        "prepared-calibration-v1",
        pose_c2w,
        ((fx, 0.0, cx), (0.0, fy, cy), (0.0, 0.0, 1.0)),
    )
    if metadata.get("calibration_attestation") != expected_attestation:
        raise ValueError("calibration attestation does not match authoritative metadata")
    if tier not in lease.available_tiers:
        raise ValueError(f"ready camera binding requires prepared {tier!r} tier")
    token = lease.control_token
    if not isinstance(token, PreparedIngressControlToken):
        raise ValueError("ready camera binding requires a prepared control token")
    return PreparedCameraMetadata(
        source_identity=source,
        intrinsics=intrinsics,
        pose_c2w=pose_c2w,
        rgb_shape=rgb_shape,
        depth_shape=depth_shape,
        depth_units="m",
        depth_aligned_to_rgb=True,
        modality_eligibility=lease.modality_eligibility,
        tier=tier,
        control_token=token,
    )


def _payload(lease: PreparedSourceLease, name: str, *, shape: tuple[int, ...], encoding: str) -> torch.Tensor:
    ref = lease.payloads.get(name)
    if ref is None or ref.spec.encoding != encoding or tuple(ref.spec.shape) != shape:
        raise ValueError(f"prepared {name} payload contract does not match metadata")
    value = ref.data
    if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
        raise ValueError(f"prepared {name} payload data does not match metadata")
    if str(value.device) != lease.device or value.dtype != torch.float32:
        raise ValueError(f"prepared {name} payload device/dtype is not authoritative")
    return value


def _full_identity_camera_uid(source: SourceIdentity) -> int:
    """Stable signed-int UID for optimizer attribution of one full source identity."""

    return int(source.fingerprint()[:15], 16)


def _corrected_geometry(
    lease: PreparedSourceLease,
    correction: PreparedPoseCorrection,
    *,
    tier: str,
    prepared: Any,
    metadata: PreparedCameraMetadata,
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], tuple[torch.Tensor, torch.Tensor]]:
    if correction.source_identity != lease.source_identity:
        raise ValueError("prepared pose correction source does not match READY lease")
    if correction.device != lease.device:
        raise ValueError("prepared pose correction device does not match READY lease")
    calibration = correction.tier_calibrations.get(tier)
    geometry = correction.tier_geometry.get(tier)
    if calibration is None or geometry is None:
        raise ValueError("prepared pose correction does not cover requested tier")
    expected = (
        (prepared.width, prepared.height, *prepared.intrinsics)
        if prepared is not None and prepared.intrinsics is not None
        else (
            metadata.intrinsics.width,
            metadata.intrinsics.height,
            metadata.intrinsics.fx,
            metadata.intrinsics.fy,
            metadata.intrinsics.cx,
            metadata.intrinsics.cy,
        )
    )
    if tuple(calibration) != tuple(expected):
        raise ValueError("prepared pose correction tier calibration does not match READY lease")
    return correction.c2w, correction.common_geometry, geometry


def bind_ready_camera(binding: ReadySourceBinding, *, tier: str = "native") -> ReadyCameraBinding:
    """Create one canonical camera from a selected READY binding without H2D.

    The supplied ``ReadySourceBinding`` is the sole source of versions.  READY
    selection is the preparation-completion proof, so this function never polls
    the preparation event again and never constructs a ``GpuWorkEnvelope``.
    """

    if not isinstance(binding, ReadySourceBinding):
        raise TypeError("binding must be ReadySourceBinding")
    lease = binding.lease
    if not isinstance(tier, str) or not tier:
        raise ValueError("prepared tier must be non-empty")
    metadata = _metadata(lease, tier=tier)
    prepared = lease.tier_payloads.get(tier)
    if prepared is None:
        # Tier is not materialized. Check if it was supposed to be prepared.
        # tier_preparation_available contains the declared set of tiers that
        # the policy committed to materialize before READY (issue #1575).
        if tier not in lease.tier_preparation_available:
            raise ValueError(
                f"prepared tier {tier!r} was not declared in the prepared tier policy "
                f"(declared tiers: {sorted(lease.tier_preparation_available)}); "
                f"no lazy tier construction after READY"
            )
        # Tier was supposed to be prepared but isn't materialized yet.
        # Fall back to native (the only tier we can reconstruct from borrowed payloads).
        if tier != "native":
            raise ValueError(
                f"prepared {tier!r} tier payload is unavailable and cannot be reconstructed"
            )
        rgb = _payload(lease, "rgb", shape=metadata.rgb_shape, encoding="rgb_float32_hwc")
        depth = _payload(lease, "depth_meters", shape=metadata.depth_shape, encoding="depth_meters")
        K = _payload(lease, "camera_intrinsics", shape=(3, 3), encoding="camera_intrinsics")
        fx, fy, cx, cy = (
            metadata.intrinsics.fx,
            metadata.intrinsics.fy,
            metadata.intrinsics.cx,
            metadata.intrinsics.cy,
        )
        confidence = None
    else:
        rgb = prepared.rgb.data
        depth = prepared.depth.data
        K = prepared.camera_intrinsics.data
        if not all(isinstance(value, torch.Tensor) for value in (rgb, depth, K)):
            raise ValueError("prepared tier payload tensors are not authoritative")
        if prepared.intrinsics is None:
            raise ValueError("prepared tier payload must carry sealed intrinsics")
        fx, fy, cx, cy = prepared.intrinsics
        confidence = None if prepared.depth_confidence is None else prepared.depth_confidence.data
    correction = binding.pose_correction
    if correction is None:
        c2w = _payload(lease, "c2w", shape=(4, 4), encoding="c2w")
        geometry_names = (
            ("world_view_transform", (4, 4)),
            ("camera_center", (3,)),
            ("gsplat_viewmat", (4, 4)),
            ("gsplat_camera_center", (1, 3)),
        )
        common_geometry = tuple(
            _payload(lease, name, shape=shape, encoding=name)
            for name, shape in geometry_names
        ) if all(name in lease.payloads for name, _shape in geometry_names) else None
        tier_geometry = None if (
            prepared is None
            or prepared.projection_matrix is None
            or prepared.full_proj_transform is None
        ) else (prepared.projection_matrix.data, prepared.full_proj_transform.data)
        pose_revision = 0
    else:
        c2w, common_geometry, tier_geometry = _corrected_geometry(
            lease, correction, tier=tier, prepared=prepared, metadata=metadata
        )
        metadata = replace(metadata, pose_c2w=correction.pose_c2w)
        pose_revision = correction.revision
    if prepared is not None and (common_geometry is None or tier_geometry is None):
        raise ValueError("prepared tier camera geometry must be sealed before READY")
    from scene.cameras import Camera

    camera = Camera.from_prepared_device_payloads(
        rgb_hwc=rgb,
        depth_meters=depth,
        c2w=c2w,
        camera_intrinsics=K,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        uid=_full_identity_camera_uid(metadata.source_identity),
        **(
            {}
            if common_geometry is None or tier_geometry is None
            else {
                "world_view_transform": common_geometry[0],
                "camera_center": common_geometry[1],
                "projection_matrix": tier_geometry[0],
                "full_proj_transform": tier_geometry[1],
                "gsplat_viewmat": common_geometry[2],
                "gsplat_camera_center": common_geometry[3],
            }
        ),
    )
    camera._source_identity = metadata.source_identity
    camera.modality_eligibility = metadata.modality_eligibility
    camera._canonical_resolution_tier = tier
    camera._resolution_depth_confidence_weight = confidence
    return ReadyCameraBinding(
        camera=camera,
        source_identity=metadata.source_identity,
        modality_eligibility=metadata.modality_eligibility,
        control_token=metadata.control_token,
        versions=binding.versions,
        prepared_tier=metadata.tier,
        metadata=metadata,
        lease=lease,
        pose_revision=pose_revision,
    )


__all__ = ["PreparedCameraMetadata", "ReadyCameraBinding", "bind_ready_camera"]
