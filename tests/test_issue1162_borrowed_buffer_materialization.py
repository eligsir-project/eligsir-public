"""Regression coverage for borrowed ROS image buffers at the core boundary."""

from __future__ import annotations

import numpy as np

from eligsir.contracts import (
    BufferOwnership,
    BufferRef,
    BufferSpec,
    CameraIntrinsics,
    DepthPolicy,
    MaterializationPolicy,
    MemoryDomain,
    SensorPacket,
    SourceIdentity,
    materialize_mapping_frame,
)


def _intrinsics() -> CameraIntrinsics:
    return CameraIntrinsics(
        fx=500.0,
        fy=500.0,
        cx=1.0,
        cy=1.0,
        width=2,
        height=2,
        depth_scale=1000.0,
    )


def _rgb_ref() -> BufferRef:
    raw = bytearray(range(16))  # two RGB rows with two bytes of row padding
    return BufferRef(
        data=memoryview(raw),
        spec=BufferSpec(
            shape=(2, 2, 3),
            dtype="uint8",
            memory_domain=MemoryDomain.HOST,
            encoding="rgb8",
            contiguous=False,
            strides_bytes=(8, 3, 1),
        ),
        ownership=BufferOwnership.BORROWED,
        owner_token=raw,
    )


def test_preserved_borrowed_rgb_uses_declared_shape_and_row_stride_without_copy():
    rgb = _rgb_ref()
    packet = SensorPacket(
        SourceIdentity("ros", 0, 1, 10), rgb, None, _intrinsics()
    )

    frame = materialize_mapping_frame(packet)

    assert frame.rgb.data.shape == (2, 2, 3)
    assert frame.rgb.data.dtype == np.uint8
    assert frame.rgb.data.strides == (8, 3, 1)
    assert frame.rgb.ownership is BufferOwnership.BORROWED
    assert frame.rgb.owner_token is rgb.owner_token
    # Second row starts after the two padding bytes.
    assert frame.rgb.data[1, 0].tolist() == [8, 9, 10]


def test_borrowed_16uc1_depth_is_typed_before_metric_conversion():
    depth_values = np.array([[1000, 2000], [3000, 4000]], dtype=np.uint16)
    depth_storage = bytearray(depth_values.tobytes())
    depth = BufferRef(
        data=memoryview(depth_storage),
        spec=BufferSpec(
            shape=(2, 2),
            dtype="uint16",
            memory_domain=MemoryDomain.HOST,
            encoding="16UC1",
            units="raw_depth_units",
            contiguous=True,
            strides_bytes=(4, 2),
        ),
        ownership=BufferOwnership.BORROWED,
        owner_token=depth_storage,
    )
    packet = SensorPacket(
        SourceIdentity("ros", 0, 1, 10), _rgb_ref(), depth, _intrinsics()
    )

    frame = materialize_mapping_frame(
        packet, MaterializationPolicy(depth=DepthPolicy.METERS_FLOAT32)
    )

    assert frame.depth is not None
    np.testing.assert_allclose(
        frame.depth.data,
        np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
    )
    assert frame.depth.data.dtype == np.float32
    assert frame.depth.ownership is BufferOwnership.OWNED
    assert frame.depth.owner_token is None


def test_inconsistent_borrowed_buffer_spec_fails_closed():
    bad = BufferRef(
        data=memoryview(bytearray(3)),
        spec=BufferSpec(
            shape=(2, 2),
            dtype="uint16",
            memory_domain=MemoryDomain.HOST,
            encoding="16UC1",
            units="raw_depth_units",
        ),
        ownership=BufferOwnership.BORROWED,
    )
    packet = SensorPacket(
        SourceIdentity("ros", 0, 1, 10), _rgb_ref(), bad, _intrinsics()
    )

    try:
        materialize_mapping_frame(packet)
    except ValueError as exc:
        assert "does not satisfy its BufferSpec" in str(exc)
    else:  # pragma: no cover - must fail closed
        raise AssertionError("invalid borrowed buffer unexpectedly materialized")
