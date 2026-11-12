from __future__ import annotations

import numpy as np

from eligsir.contracts import (
    BufferOwnership,
    BufferRef,
    CameraIntrinsics,
    MappingFrame,
    MemoryDomain,
    Pose,
    SourceIdentity,
)
from eligsir.runtime.streaming_observation import StreamingObservation


def test_projection_preserves_pixels_depth_identity_and_pose_without_copy():
    rgb = np.zeros((3, 4, 3), dtype=np.uint8)
    depth = np.full((3, 4), 1.25, dtype=np.float32)
    intr = CameraIntrinsics(
        fx=100.0,
        fy=101.0,
        cx=2.0,
        cy=1.5,
        width=4,
        height=3,
        depth_fx=90.0,
        depth_fy=91.0,
        depth_cx=2.0,
        depth_cy=1.5,
        depth_width=4,
        depth_height=3,
    )
    frame = MappingFrame(
        source=SourceIdentity("live", 2, 17, 1_234_567_890),
        rgb=BufferRef.from_numpy(rgb, encoding="rgb8"),
        depth=BufferRef.from_numpy(depth, encoding="32FC1", units="meters"),
        intrinsics=intr,
        device="cpu",
        pose=Pose.identity(),
    )

    obs = StreamingObservation.from_mapping_frame(frame)

    assert obs.mapping_frame is frame
    assert obs._rgb_np is rgb
    assert obs._depth_np is depth
    assert obs._rgb_bytes is None
    assert obs._depth_bytes is None
    assert obs._sens_header is None
    assert obs.index == 17
    assert obs.source_ordinal == 17
    assert obs.color_ts_ns == 1_234_567_890
    assert np.isclose(obs.timestamp, 1_234_567_890e-9)
    assert obs.depth_scale == 1.0
    assert obs.fx == 100.0 and obs.depth_fx == 90.0
    np.testing.assert_array_equal(obs.c2w, np.eye(4, dtype=np.float32))


def test_projection_remains_mutable_for_mapper_local_annotations():
    rgb = np.zeros((2, 2, 3), dtype=np.uint8)
    frame = MappingFrame(
        source=SourceIdentity("live", 0, 0, 1),
        rgb=BufferRef.from_numpy(rgb, encoding="rgb8"),
        depth=None,
        intrinsics=CameraIntrinsics(fx=1.0, fy=1.0, cx=1.0, cy=1.0, width=2, height=2),
        device="cpu",
        pose=Pose.identity(),
    )
    obs = StreamingObservation.from_mapping_frame(frame)
    obs._streaming_admission_stats = {"reason": "test"}
    assert obs._streaming_admission_stats["reason"] == "test"
