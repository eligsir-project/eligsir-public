"""Contract tests for the streaming ``SensorPacket -> MappingFrame`` bridge.

The bridge is consumed by both insertion planners. These tests pin the shared
CPU-pixel and metric-depth contract at that production seam.

The assertions below pin the two properties the consumers depend on:
CPU ownership (so ``DeviceTransferOwner`` can own the grouped H2D) and the
depth UNIT contract -- the materializer divides raw counts by
``intrinsics.depth_scale`` exactly once, which is what makes it safe to feed
the result into an insertion path that also calls ``depth_to_meters``.
"""

from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from eligsir.contracts.runtime import BufferRef, MappingFrame, SourceIdentity
from eligsir.contracts.streaming_bridge import (
    mapping_frame_from_streaming_frame,
    mapping_frame_source_arrays,
)
from eligsir.contracts.types import CameraIntrinsics, Pose


def _frame(depth_scale=1000.0, h=8, w=12):
    return SimpleNamespace(
        index=3,
        timestamp=1.5,
        rgb_path="/tmp/frame.png",
        source_ordinal=3,
        stream_epoch=0,
        fx=50.0, fy=50.0, cx=6.0, cy=4.0,
        depth_fx=None, depth_fy=None, depth_cx=None, depth_cy=None,
        depth_width=None, depth_height=None,
        depth_scale=depth_scale,
        height=h, width=w,
        c2w=np.eye(4, dtype=np.float32),
    )


def test_rgb_stays_cpu_owned_so_the_prefetcher_can_own_the_h2d():
    rgb = torch.rand(3, 8, 12)
    frame_out = mapping_frame_from_streaming_frame(_frame(), rgb, None)
    assert frame_out.rgb.data.device.type == "cpu"
    assert tuple(frame_out.rgb.data.shape) == (3, 8, 12)


def test_a_device_resident_rgb_is_refused():
    """The bridge must not silently accept an already-uploaded tensor: that is
    the exact shape of the pessimization issue #1151 documents."""
    rgb = torch.rand(3, 8, 12)
    if not torch.cuda.is_available():
        pytest.skip("needs GPU to build a device-resident tensor")
    with pytest.raises(ValueError):
        mapping_frame_from_streaming_frame(_frame(), rgb.to("cuda"), None)


def test_raw_depth_counts_are_converted_to_metres_exactly_once():
    rgb = torch.rand(3, 8, 12)
    raw = np.full((8, 12), 2000, dtype=np.uint16)
    out = mapping_frame_from_streaming_frame(_frame(depth_scale=1000.0), rgb, raw)
    depth = out.depth.data
    depth_np = depth.detach().cpu().numpy() if hasattr(depth, "detach") else np.asarray(depth)
    # 2000 counts at 1000 counts/m is 2.0 m -- not 2000.0 and not 0.002.
    assert np.allclose(depth_np, 2.0)
    assert str(depth_np.dtype) == "float32"


def test_metric_depth_is_not_scaled_a_second_time():
    """float32 input is already metres; applying depth_scale again would put
    every point at the wrong range with no error raised."""
    rgb = torch.rand(3, 8, 12)
    metric = np.full((8, 12), 2.0, dtype=np.float32)
    out = mapping_frame_from_streaming_frame(_frame(depth_scale=1000.0), rgb, metric)
    depth = out.depth.data
    depth_np = depth.detach().cpu().numpy() if hasattr(depth, "detach") else np.asarray(depth)
    assert np.allclose(depth_np, 2.0)


def test_depth_is_optional():
    rgb = torch.rand(3, 8, 12)
    out = mapping_frame_from_streaming_frame(_frame(), rgb, None)
    assert out.depth is None


def test_a_non_3d_rgb_is_refused():
    with pytest.raises(ValueError):
        mapping_frame_from_streaming_frame(_frame(), torch.rand(8, 12), None)


def test_a_too_few_channel_rgb_is_refused():
    with pytest.raises(ValueError):
        mapping_frame_from_streaming_frame(_frame(), torch.rand(2, 8, 12), None)


def test_production_mapping_frame_exposes_metric_depth_and_normalized_hwc_rgb():
    rgb = np.full((8, 12, 3), 128, dtype=np.uint8)
    depth = np.full((8, 12), 2.0, dtype=np.float32)
    mapping = MappingFrame(
        source=SourceIdentity("live", 0, 3, 1_500_000_000),
        rgb=BufferRef.from_numpy(rgb, encoding="rgb8"),
        depth=BufferRef.from_numpy(depth, encoding="32FC1", units="meters"),
        intrinsics=CameraIntrinsics(
            fx=50.0, fy=50.0, cx=6.0, cy=4.0, width=12, height=8
        ),
        device="cpu",
        pose=Pose.identity(),
    )
    frame = _frame()
    frame.mapping_frame = mapping

    z, color = mapping_frame_source_arrays(frame)

    assert z is depth
    assert z.dtype == np.float32
    assert color.shape == (8, 12, 3)
    assert color.dtype == np.float32
    assert np.allclose(color, 128.0 / 255.0)


def test_mapping_frame_source_arrays_declines_mismatched_resolved_geometry():
    rgb = np.zeros((8, 12, 3), dtype=np.uint8)
    mapping = MappingFrame(
        source=SourceIdentity("live", 0, 3, 1_500_000_000),
        rgb=BufferRef.from_numpy(rgb, encoding="rgb8"),
        depth=None,
        intrinsics=CameraIntrinsics(
            fx=50.0, fy=50.0, cx=6.0, cy=4.0, width=12, height=8
        ),
        device="cpu",
        pose=Pose.identity(),
    )
    frame = _frame(h=4, w=6)
    frame.mapping_frame = mapping

    with pytest.raises(ValueError, match="resolution disagrees"):
        mapping_frame_source_arrays(frame)


def test_mapping_frame_source_arrays_rejects_divergent_rgb_depth_calibration():
    rgb = np.zeros((8, 12, 3), dtype=np.uint8)
    depth = np.ones((8, 12), dtype=np.float32)
    mapping = MappingFrame(
        source=SourceIdentity("live", 0, 3, 1_500_000_000),
        rgb=BufferRef.from_numpy(rgb, encoding="rgb8"),
        depth=BufferRef.from_numpy(depth, encoding="32FC1", units="meters"),
        intrinsics=CameraIntrinsics(
            fx=50.0,
            fy=50.0,
            cx=6.0,
            cy=4.0,
            width=12,
            height=8,
            depth_fx=48.0,
            depth_fy=49.0,
            depth_cx=5.5,
            depth_cy=3.5,
            depth_width=12,
            depth_height=8,
            intrinsics_source="unregistered_dual_sensor",
        ),
        device="cpu",
        pose=Pose.identity(),
    )
    frame = _frame()
    frame.mapping_frame = mapping

    with pytest.raises(ValueError, match="registered RGB/depth calibration"):
        mapping_frame_source_arrays(frame)
