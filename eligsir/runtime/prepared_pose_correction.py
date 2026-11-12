"""Explicit preparation of one immutable, absolute READY pose correction."""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np
import torch

from eligsir.contracts import SourceIdentity
from eligsir.runtime.prepared_source import PreparedSourceLease


def _exact_non_negative_int(value: Any, name: str) -> int:
    if type(value) is not int or value < 0:
        raise TypeError(f"{name} must be a non-negative exact integer")
    return value


def _rigid_pose(value: object) -> tuple[tuple[float, ...], ...]:
    pose = np.asarray(value, dtype=np.float64)
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise ValueError("pose_c2w must be a finite 4x4 matrix")
    if not np.allclose(pose[3], (0.0, 0.0, 0.0, 1.0), rtol=0.0, atol=1e-6):
        raise ValueError("pose_c2w must be a rigid homogeneous transform")
    rotation = pose[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), rtol=0.0, atol=1e-5) or not np.isclose(
        np.linalg.det(rotation), 1.0, rtol=0.0, atol=1e-5
    ):
        raise ValueError("pose_c2w must have a finite proper rigid rotation")
    return tuple(tuple(float(component) for component in row) for row in pose.astype(np.float32))


def _device_tensor(value: object, *, shape: tuple[int, ...], device: str, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
        raise TypeError(f"{name} must be a {shape} torch tensor")
    if value.dtype != torch.float32 or str(value.device) != device:
        raise ValueError(f"{name} must be float32 on the correction device")
    return value


@dataclass(frozen=True, slots=True)
class PreparedPoseCorrection:
    """One prepared absolute pose and the small camera geometry it needs."""

    source_identity: SourceIdentity
    source_generation: int
    device: str
    pose_c2w: tuple[tuple[float, ...], ...]
    revision: int
    c2w: torch.Tensor
    common_geometry: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    tier_geometry: Mapping[str, tuple[torch.Tensor, torch.Tensor]]
    tier_calibrations: Mapping[str, tuple[int, int, float, float, float, float]]
    completion_event: Any

    def __post_init__(self) -> None:
        if not isinstance(self.source_identity, SourceIdentity):
            raise TypeError("source_identity must be SourceIdentity")
        object.__setattr__(self, "source_generation", _exact_non_negative_int(self.source_generation, "source_generation"))
        object.__setattr__(self, "revision", _exact_non_negative_int(self.revision, "revision"))
        if self.revision == 0:
            raise ValueError("prepared pose correction revision must be positive")
        device = str(self.device)
        if not device:
            raise ValueError("correction device must be non-empty")
        object.__setattr__(self, "device", device)
        object.__setattr__(self, "pose_c2w", _rigid_pose(self.pose_c2w))
        object.__setattr__(self, "c2w", _device_tensor(self.c2w, shape=(4, 4), device=device, name="c2w"))
        common = tuple(self.common_geometry)
        if len(common) != 4:
            raise ValueError("common camera geometry must contain four tensors")
        object.__setattr__(
            self,
            "common_geometry",
            tuple(
                _device_tensor(value, shape=shape, device=device, name="common camera geometry")
                for value, shape in zip(common, ((4, 4), (3,), (4, 4), (1, 3)), strict=True)
            ),
        )
        geometry = dict(self.tier_geometry)
        calibrations = dict(self.tier_calibrations)
        if not geometry or set(geometry) != set(calibrations):
            raise ValueError("prepared pose correction must cover matching non-empty geometry tiers")
        for tier, values in geometry.items():
            if not isinstance(tier, str) or not tier or not isinstance(values, tuple) or len(values) != 2:
                raise ValueError("prepared pose correction tier geometry is malformed")
            geometry[tier] = tuple(
                _device_tensor(value, shape=(4, 4), device=device, name="tier camera geometry")
                for value in values
            )
            calibration = calibrations[tier]
            if (
                not isinstance(calibration, tuple)
                or len(calibration) != 6
                or type(calibration[0]) is not int
                or type(calibration[1]) is not int
                or calibration[0] < 1
                or calibration[1] < 1
                or any(not isinstance(value, (int, float)) or not math.isfinite(float(value)) for value in calibration[2:])
            ):
                raise ValueError("prepared pose correction tier calibration is malformed")
        if not callable(getattr(self.completion_event, "query", None)):
            raise TypeError("prepared pose correction requires a completion event")
        object.__setattr__(self, "tier_geometry", MappingProxyType(geometry))
        object.__setattr__(self, "tier_calibrations", MappingProxyType(calibrations))

    def handoff_to_compute(self, transfer_owner: Any) -> None:
        handoff = getattr(self.completion_event, "handoff_to_compute", None)
        if callable(handoff):
            handoff(transfer_owner)
            return
        owner_handoff = getattr(transfer_owner, "handoff_to_compute", None)
        if not callable(owner_handoff):
            raise TypeError("transfer_owner must provide handoff_to_compute")
        owner_handoff(self.completion_event)


def _tier_inputs(
    lease: PreparedSourceLease,
) -> tuple[
    dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]],
    dict[str, tuple[int, int, float, float, float, float]],
]:
    tiers = {}
    calibrations = {}
    for name, payload in lease.tier_payloads.items():
        if payload.intrinsics is None:
            raise ValueError("prepared pose correction requires sealed tier intrinsics")
        tiers[name] = (payload.rgb.data, payload.depth.data, payload.camera_intrinsics.data, None if payload.depth_confidence is None else payload.depth_confidence.data)
        calibrations[name] = (int(payload.width), int(payload.height), *(float(value) for value in payload.intrinsics))
    if tiers:
        return tiers, calibrations
    if lease.available_tiers != ("native",):
        raise ValueError("prepared pose correction requires every declared tier to be materialized")
    metadata = lease.source_metadata
    intrinsics = metadata.get("camera_intrinsics")
    if not isinstance(intrinsics, tuple) or len(intrinsics) < 6:
        raise ValueError("prepared pose correction requires authoritative native intrinsics")
    rgb, depth, camera_intrinsics = (
        lease.payloads.get("rgb"),
        lease.payloads.get("depth_meters"),
        lease.payloads.get("camera_intrinsics"),
    )
    if any(value is None or not isinstance(value.data, torch.Tensor) for value in (rgb, depth, camera_intrinsics)):
        raise ValueError("prepared pose correction requires native RGB-D calibration payloads")
    tiers["native"] = (rgb.data, depth.data, camera_intrinsics.data, None)
    calibrations["native"] = (int(intrinsics[4]), int(intrinsics[5]), *(float(intrinsics[index]) for index in range(4)))
    return tiers, calibrations


def prepare_pose_correction(
    lease: PreparedSourceLease,
    pose_c2w: object,
    *,
    expected_revision: int,
    transfer_owner: Any,
) -> PreparedPoseCorrection:
    """Prepare all corrected camera geometry before an accepted-state commit."""

    if not isinstance(lease, PreparedSourceLease):
        raise TypeError("lease must be PreparedSourceLease")
    expected_revision = _exact_non_negative_int(expected_revision, "expected_revision")
    pose = _rigid_pose(pose_c2w)
    upload = getattr(transfer_owner, "upload_packed", None)
    geometry_prepare = getattr(transfer_owner, "prepare_source_camera_geometry", None)
    if not callable(upload) or not callable(geometry_prepare):
        raise TypeError("transfer_owner must prepare packed calibration and camera geometry")
    pose_array = np.asarray(pose, dtype=np.float32)
    inverse_array = np.ascontiguousarray(np.linalg.inv(pose_array), dtype=np.float32)
    uploaded, _upload_event = upload(
        (np.ascontiguousarray(pose_array), inverse_array),
        purpose="prepared_pose_correction",
        handoff_to_compute=False,
    )
    if not isinstance(uploaded, tuple) or len(uploaded) != 2:
        raise TypeError("packed pose correction upload must return c2w and w2c")
    c2w, w2c = (
        _device_tensor(value, shape=(4, 4), device=lease.device, name=name)
        for value, name in zip(uploaded, ("c2w", "w2c"), strict=True)
    )
    tiers, calibrations = _tier_inputs(lease)
    common, per_tier, event = geometry_prepare(c2w, tiers, w2c=w2c)
    return PreparedPoseCorrection(
        source_identity=lease.source_identity,
        source_generation=lease.generation,
        device=lease.device,
        pose_c2w=pose,
        revision=expected_revision + 1,
        c2w=c2w,
        common_geometry=common,
        tier_geometry=per_tier,
        tier_calibrations=calibrations,
        completion_event=event,
    )


__all__ = ["PreparedPoseCorrection", "prepare_pose_correction"]
