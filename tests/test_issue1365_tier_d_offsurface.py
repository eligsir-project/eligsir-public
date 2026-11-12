from types import SimpleNamespace

import numpy as np

from eligsir.contracts.runtime import BufferRef, MappingFrame, SourceIdentity
from eligsir.contracts.types import CameraIntrinsics, Pose
from eligsir.mapping.canonical_scene import CanonicalStreamingScene
from eligsir.runtime.config import CanonicalBootstrapPolicy, CanonicalRuntimePolicy


def _observation(*, ordinal=0, x=0.0):
    depth = np.full((4, 6), 2.0, dtype=np.float32)
    rgb = np.full((4, 6, 3), 128, dtype=np.uint8)
    c2w = np.eye(4, dtype=np.float32)
    c2w[0, 3] = x
    frame = MappingFrame(
        source=SourceIdentity("canonical", 0, ordinal, 1_000_000_000 + ordinal),
        rgb=BufferRef.from_numpy(rgb, encoding="rgb8"),
        depth=BufferRef.from_numpy(depth, encoding="32FC1", units="m"),
        intrinsics=CameraIntrinsics(
            fx=8.0, fy=8.0, cx=3.0, cy=2.0, width=6, height=4,
            intrinsics_source="shared_rgb_depth_pixel_geometry",
        ),
        device="cpu",
        pose=Pose(c2w),
    )
    from eligsir.runtime.streaming_observation import StreamingObservation
    return StreamingObservation.from_mapping_frame(frame)


def test_canonical_scene_exposes_accumulated_sensor_points_from_mapping_frames(
    tmp_path,
):
    scene = CanonicalStreamingScene(
        CanonicalRuntimePolicy(model_path=str(tmp_path), keyframe_window=1, live_payload_max_frames=1),
        CanonicalBootstrapPolicy(depth_stride=1, min_depth_m=0.1, max_depth_m=5.0),
        SimpleNamespace(),
        (_observation(),),
    )

    scene._accumulate_sensor_points(_observation(x=2.0))
    points = scene.get_accumulated_sensor_points()

    assert points.shape == (24, 3)
    assert points.dtype == np.float32
    assert np.allclose(points[:, 2], 2.0)
    assert np.allclose(points[:, 0], np.array([-0.75, -0.5, -0.25, 0.0, 0.25, 0.5] * 4) + 2.0)
