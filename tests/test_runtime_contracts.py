from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from eligsir.contracts import (
    BufferOwnership,
    BufferRef,
    BufferSpec,
    CameraIntrinsics,
    IngressAction,
    IngressEvent,
    MappingFrame,
    MemoryDomain,
    Pose,
    SensorPacket,
    SourceIdentity,
    TrackingObservation,
    TrackingStatus,
)


def _intrinsics() -> CameraIntrinsics:
    return CameraIntrinsics(fx=500, fy=500, cx=320, cy=240, width=640, height=480)


def _host_spec(shape: tuple[int, ...], dtype: str, encoding: str, units=None) -> BufferSpec:
    return BufferSpec(
        shape=shape,
        dtype=dtype,
        memory_domain=MemoryDomain.HOST,
        encoding=encoding,
        units=units,
    )


def test_source_identity_is_frozen_and_uses_integer_nanoseconds():
    source = SourceIdentity("camera", 0, 7, 1_234_567_890)
    assert source.timestamp_ns == 1_234_567_890
    assert source.timestamp_seconds == pytest.approx(1.23456789)
    with pytest.raises(FrozenInstanceError):
        source.source_ordinal = 8  # type: ignore[misc]


def test_drop_does_not_renumber_later_source_identity():
    source_before = SourceIdentity("camera", 0, 10, 100)
    dropped = SourceIdentity("camera", 0, 11, 200)
    source_after = SourceIdentity("camera", 0, 12, 300)
    event = IngressEvent(
        source=dropped,
        stage="source_queue",
        owner="live_policy",
        action=IngressAction.DROP,
        reason="overflow_drop_oldest",
        queue_depth=4,
    )
    assert source_before.source_ordinal == 10
    assert event.source.source_ordinal == 11
    assert source_after.source_ordinal == 12


def test_device_buffer_requires_explicit_device():
    with pytest.raises(ValueError, match="explicit device"):
        BufferSpec(
            shape=(480, 640, 3),
            dtype="float32",
            memory_domain=MemoryDomain.ROCM,
            encoding="rgb32f",
        )


def test_tracking_observation_freezes_covariance_and_diagnostics():
    source = SourceIdentity("camera", 0, 0, 0)
    covariance = np.eye(6)
    diagnostics = {"fitness": 0.9}
    observation = TrackingObservation(
        source=source,
        pose=Pose.identity(),
        status=TrackingStatus.VALID,
        provider="open3d",
        confidence=0.9,
        covariance=covariance,
        diagnostics=diagnostics,
    )
    covariance[0, 0] = 5.0
    diagnostics["fitness"] = 0.1
    assert observation.covariance is not None
    assert observation.covariance[0, 0] == 1.0
    assert observation.covariance.flags.writeable is False
    assert observation.diagnostics["fitness"] == 0.9
    with pytest.raises(TypeError):
        observation.diagnostics["fitness"] = 0.5  # type: ignore[index]


def test_sensor_packet_requires_tracking_identity_match():
    source = SourceIdentity("camera", 0, 4, 400)
    other = SourceIdentity("camera", 0, 5, 500)
    rgb = BufferRef(
        data=np.zeros((2, 3, 3), dtype=np.uint8),
        spec=_host_spec((2, 3, 3), "uint8", "rgb8"),
        ownership=BufferOwnership.OWNED,
    )
    depth = BufferRef(
        data=np.zeros((2, 3), dtype=np.uint16),
        spec=_host_spec((2, 3), "uint16", "16UC1", "raw_counts"),
        ownership=BufferOwnership.OWNED,
    )
    tracking = TrackingObservation(
        source=other,
        pose=Pose.identity(),
        status=TrackingStatus.VALID,
        provider="gt",
        confidence=1.0,
    )
    with pytest.raises(ValueError, match="tracking.source"):
        SensorPacket(source, rgb, depth, _intrinsics(), tracking)


def test_mapping_frame_keeps_device_payload_metadata_attached_without_torch_dependency():
    # The contract this guards: an opaque, device-resident payload (a raw
    # torch/ROCm/CUDA tensor, or anything else this module never imports to
    # inspect) can still be carried by MappingFrame -- but only wrapped in a
    # BufferRef that declares its real device, never as a bare object bypassing
    # BufferRef's spec/ownership/owner_token entirely. See MappingFrame's own
    # docstring: this module never imports a tensor framework because it only
    # ever touches BufferRef/BufferSpec (plain dataclasses), never `.data`.
    source = SourceIdentity("camera", 0, 1, 100)
    opaque_rgb = object()
    opaque_depth = object()
    rgb = BufferRef(
        data=opaque_rgb,
        spec=BufferSpec(
            shape=(3, 2, 3),
            dtype="float32",
            memory_domain=MemoryDomain.CUDA,
            encoding="rgb32f_chw",
            device="cuda:0",
        ),
        ownership=BufferOwnership.OWNED,
    )
    depth = BufferRef(
        data=opaque_depth,
        spec=BufferSpec(
            shape=(1, 2, 3),
            dtype="float32",
            memory_domain=MemoryDomain.CUDA,
            encoding="depth32f_chw",
            units="m",
            device="cuda:0",
        ),
        ownership=BufferOwnership.OWNED,
    )
    frame = MappingFrame(
        source=source,
        rgb=rgb,
        depth=depth,
        intrinsics=_intrinsics(),
        device="cuda:0",
    )
    assert frame.rgb.data is opaque_rgb
    assert frame.depth is not None and frame.depth.data is opaque_depth
    assert frame.rgb.spec.device == "cuda:0"
    assert frame.depth.spec.device == "cuda:0"
    assert frame.device == "cuda:0"
