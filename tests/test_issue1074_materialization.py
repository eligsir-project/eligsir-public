"""Perf/ownership tests for the canonical packet->frame conversion point."""

from __future__ import annotations

import numpy as np
import pytest

from eligsir.contracts import (
    BufferOwnership,
    BufferRef,
    BufferSpec,
    CameraIntrinsics,
    ColorPolicy,
    DepthPolicy,
    MappingFrame,
    MaterializationPolicy,
    MemoryDomain,
    Pose,
    SensorPacket,
    SourceIdentity,
    TrackingObservation,
    TrackingStatus,
    materialize_mapping_frame,
    numpy_backend,
    plan_materialization,
    torch_backend,
)


def _intrinsics(depth_scale: float = 1000.0) -> CameraIntrinsics:
    return CameraIntrinsics(
        fx=500.0, fy=500.0, cx=320.0, cy=240.0, width=8, height=8, depth_scale=depth_scale
    )


def _packet(
    *,
    rgb: np.ndarray | None = None,
    depth: np.ndarray | None = None,
    rgb_ownership: BufferOwnership = BufferOwnership.OWNED,
    depth_ownership: BufferOwnership | None = None,
    depth_scale: float = 1000.0,
) -> SensorPacket:
    rgb = np.zeros((8, 8, 3), dtype=np.uint8) if rgb is None else rgb
    depth = np.full((8, 8), 1500, dtype=np.uint16) if depth is None else depth
    if depth_ownership is None:
        depth_ownership = rgb_ownership
    return SensorPacket(
        SourceIdentity("camera", 0, 1, 1_699_999_999_123_456_789),
        BufferRef.from_numpy(rgb, encoding="rgb8", ownership=rgb_ownership),
        BufferRef.from_numpy(
            depth, encoding="16UC1", units="raw_counts", ownership=depth_ownership
        ),
        _intrinsics(depth_scale),
    )


def test_default_policy_is_zero_copy_and_preserves_native_dtype():
    packet = _packet()
    frame = materialize_mapping_frame(packet)
    assert frame.rgb.data is packet.rgb.data
    assert frame.depth is not None and frame.depth.data is packet.depth.data
    assert frame.rgb.data.dtype == np.uint8
    assert frame.depth.data.dtype == np.uint16
    assert frame.device == "cpu"


def test_plan_reports_zero_copy_for_default_policy():
    plan = plan_materialization(_packet())
    assert plan.is_zero_copy
    assert plan.host_device_transfers == 0
    assert plan.would_bounce is False


def test_plan_does_not_allocate():
    packet = _packet()
    before = packet.rgb.data.copy()
    plan_materialization(packet, MaterializationPolicy(color=ColorPolicy.NORMALIZED_FLOAT32))
    assert np.array_equal(packet.rgb.data, before)


def test_normalized_color_converts_and_owns_only_rgb():
    packet = _packet(
        rgb=np.full((8, 8, 3), 255, dtype=np.uint8),
        rgb_ownership=BufferOwnership.BORROWED,
        depth_ownership=BufferOwnership.POOLED,
    )
    frame = materialize_mapping_frame(
        packet, MaterializationPolicy(color=ColorPolicy.NORMALIZED_FLOAT32)
    )
    assert frame.rgb.data.dtype == np.float32
    assert frame.rgb.data.max() == pytest.approx(1.0)
    assert frame.rgb.ownership is BufferOwnership.OWNED
    assert frame.depth is not None
    assert frame.depth.ownership is BufferOwnership.POOLED
    assert frame.depth.data is packet.depth.data


def test_depth_conversion_owns_only_depth_and_preserves_borrowed_rgb():
    packet = _packet(
        rgb_ownership=BufferOwnership.BORROWED,
        depth_ownership=BufferOwnership.BORROWED,
        depth_scale=1000.0,
    )
    frame = materialize_mapping_frame(
        packet, MaterializationPolicy(depth=DepthPolicy.METERS_FLOAT32)
    )
    assert frame.rgb.ownership is BufferOwnership.BORROWED
    assert frame.rgb.data is packet.rgb.data
    assert frame.depth is not None
    assert frame.depth.ownership is BufferOwnership.OWNED
    assert frame.depth.data[0, 0] == pytest.approx(1.5)
    assert frame.depth.spec.units == "meters"
    assert frame.depth.spec.dtype == "float32"


def test_depth_already_in_metres_is_not_rescaled_or_reowned():
    metres = np.full((8, 8), 1.5, dtype=np.float32)
    packet = SensorPacket(
        SourceIdentity("camera", 0, 1, 10),
        BufferRef.from_numpy(np.zeros((8, 8, 3), np.uint8), encoding="rgb8"),
        BufferRef.from_numpy(
            metres,
            encoding="32FC1",
            units="meters",
            ownership=BufferOwnership.SHARED_IMMUTABLE,
        ),
        _intrinsics(),
    )
    frame = materialize_mapping_frame(
        packet, MaterializationPolicy(depth=DepthPolicy.METERS_FLOAT32)
    )
    assert frame.depth is not None
    assert frame.depth.data is metres
    assert frame.depth.ownership is BufferOwnership.SHARED_IMMUTABLE


def test_non_contiguous_payload_survives_preserve_without_densification():
    dense = np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3)
    view = dense[::2]
    packet = SensorPacket(
        SourceIdentity("camera", 0, 1, 10),
        BufferRef.from_numpy(view, encoding="rgb8"),
        None,
        _intrinsics(),
    )
    frame = materialize_mapping_frame(packet)
    assert frame.rgb.data is view
    assert frame.rgb.spec.contiguous is False
    assert frame.rgb.spec.strides_bytes == tuple(view.strides)


def test_conversion_produces_a_contiguous_owned_buffer():
    dense = np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3)
    packet = SensorPacket(
        SourceIdentity("camera", 0, 1, 10),
        BufferRef.from_numpy(dense[::2], encoding="rgb8", ownership=BufferOwnership.BORROWED),
        None,
        _intrinsics(),
    )
    frame = materialize_mapping_frame(
        packet, MaterializationPolicy(color=ColorPolicy.NORMALIZED_FLOAT32)
    )
    assert frame.rgb.spec.contiguous is True
    assert frame.rgb.spec.strides_bytes is None
    assert frame.rgb.ownership is BufferOwnership.OWNED


def test_mixed_preserved_ownership_is_never_collapsed_to_rgb_value():
    packet = _packet(
        rgb_ownership=BufferOwnership.BORROWED,
        depth_ownership=BufferOwnership.POOLED,
    )
    frame = materialize_mapping_frame(packet)
    assert frame.rgb.ownership is BufferOwnership.BORROWED
    assert frame.depth is not None
    assert frame.depth.ownership is BufferOwnership.POOLED


def test_owner_tokens_follow_only_preserved_payloads():
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    rgb_token = object()
    depth_token = object()
    packet = SensorPacket(
        SourceIdentity("camera", 0, 1, 10),
        BufferRef.from_numpy(
            rgb, encoding="rgb8", ownership=BufferOwnership.BORROWED, owner_token=rgb_token
        ),
        BufferRef.from_numpy(
            np.ones((8, 8), dtype=np.uint16),
            encoding="16UC1",
            units="raw_counts",
            ownership=BufferOwnership.BORROWED,
            owner_token=depth_token,
        ),
        _intrinsics(),
    )
    frame = materialize_mapping_frame(
        packet, MaterializationPolicy(depth=DepthPolicy.METERS_FLOAT32)
    )
    assert frame.rgb.owner_token is rgb_token
    assert frame.depth is not None and frame.depth.owner_token is None


def test_borrowed_payload_recycled_by_producer_is_visible_through_frame():
    buffer = np.zeros((8, 8, 3), dtype=np.uint8)
    packet = _packet(rgb=buffer, rgb_ownership=BufferOwnership.BORROWED)
    frame = materialize_mapping_frame(packet)
    buffer[0, 0, 0] = 99
    assert frame.rgb.data[0, 0, 0] == 99
    assert frame.rgb.ownership is BufferOwnership.BORROWED
    assert frame.retainable() is False


_DTYPES = {"float32"}


class _StubTensor:
    def __init__(self, array: np.ndarray, device: str, transfers: int = 0) -> None:
        self._array = array
        self.device = device
        self.transfers = transfers

    def to(self, target):
        if target in _DTYPES:
            return _StubTensor(self._array.astype(np.float32), self.device, self.transfers)
        if target != self.device:
            return _StubTensor(self._array, target, self.transfers + 1)
        return self

    def __truediv__(self, scalar):
        return _StubTensor(self._array / scalar, self.device, self.transfers)


class _StubTorch:
    float32 = "float32"

    @staticmethod
    def as_tensor(array):
        return _StubTensor(np.asarray(array), "cpu")


def _device_packet(device: str = "cuda:0") -> SensorPacket:
    rgb = _StubTensor(np.zeros((8, 8, 3), np.float32), device)
    return SensorPacket(
        SourceIdentity("camera", 0, 1, 10),
        BufferRef(
            data=rgb,
            spec=BufferSpec(
                shape=(8, 8, 3),
                dtype="float32",
                memory_domain=MemoryDomain.ROCM,
                encoding="rgb32f",
                device=device,
            ),
            ownership=BufferOwnership.POOLED,
        ),
        None,
        _intrinsics(),
    )


def test_device_resident_packet_stays_on_device_with_no_transfer_or_reownership():
    packet = _device_packet("cuda:0")
    backend = torch_backend(_StubTorch())
    frame = materialize_mapping_frame(
        packet, MaterializationPolicy(device="cuda:0"), backend=backend
    )
    assert frame.rgb.data is packet.rgb.data
    assert frame.rgb.data.transfers == 0
    assert frame.rgb.spec.device == "cuda:0"
    assert frame.rgb.ownership is BufferOwnership.POOLED


def test_host_to_device_move_is_exactly_one_transfer_and_makes_new_owner():
    packet = _packet(rgb_ownership=BufferOwnership.BORROWED)
    backend = torch_backend(_StubTorch())
    policy = MaterializationPolicy(device="cuda:0", color=ColorPolicy.NORMALIZED_FLOAT32)
    plan = plan_materialization(packet, policy, backend=backend)
    assert plan.host_device_transfers == 2
    assert plan.would_bounce is False
    frame = materialize_mapping_frame(packet, policy, backend=backend)
    assert frame.rgb.data.transfers == 1
    assert frame.rgb.ownership is BufferOwnership.OWNED


def test_numpy_backend_refuses_a_device_rather_than_faking_one():
    with pytest.raises(ValueError, match="device-capable TensorBackend"):
        materialize_mapping_frame(
            _packet(), MaterializationPolicy(device="cuda:0"), backend=numpy_backend()
        )


def test_frame_buffer_device_must_match_frame_device():
    with pytest.raises(ValueError, match="unnoticed CPU<->GPU bounce"):
        MappingFrame(
            source=SourceIdentity("camera", 0, 1, 10),
            rgb=BufferRef(
                data=object(),
                spec=BufferSpec(
                    shape=(8, 8, 3),
                    dtype="float32",
                    memory_domain=MemoryDomain.HOST,
                    encoding="rgb32f",
                ),
                ownership=BufferOwnership.OWNED,
            ),
            depth=None,
            intrinsics=_intrinsics(),
            device="cuda:0",
        )


def test_source_identity_ordering_key_is_completion_time_independent():
    src = SourceIdentity("cam", 3, 11, 12345)
    assert src.ordering_key() == (3, 11, 12345, "cam")


def test_tracking_pose_is_carried_onto_mapping_frame():
    identity = SourceIdentity("camera", 0, 1, 10)
    pose = Pose.identity()
    packet = SensorPacket(
        identity,
        BufferRef.from_numpy(np.zeros((8, 8, 3), np.uint8), encoding="rgb8"),
        None,
        _intrinsics(),
        TrackingObservation(
            source=identity, pose=pose, status=TrackingStatus.VALID, provider="gt"
        ),
    )
    frame = materialize_mapping_frame(packet)
    assert frame.pose is pose
    assert frame.source is identity
