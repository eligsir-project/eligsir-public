import glob
import os

import numpy as np
import pytest

from scripts.bag_image_reader import read_camera_info, read_rgb_frames_near_timestamps

FLOOR3_BAG = "/data/bags/floor3"


def _rosbag2_py_available() -> bool:
    try:
        import rosbag2_py  # noqa: F401
        return True
    except ImportError:
        return False


# scripts.bag_image_reader._iter_bag_messages only has a sqlite3-based
# reader for .db3 bags; anything else (this shared floor3 mount is .mcap)
# falls back to importing rosbag2_py, which is a ROS2-stack package not
# present in the `train` container (CPU/ROCm gsplat only, no ROS2). This
# used to only skip on "bag not mounted", so on any machine where the
# shared dataset mount happens to be present but ROS2 isn't installed (the
# normal state of the `train` container), these tests failed at collection/
# runtime instead of skipping -- a real environment-detection gap, not a
# genuine regression in scripts/bag_image_reader.py itself.
_HAS_DB3 = bool(os.path.isdir(FLOOR3_BAG) and glob.glob(os.path.join(FLOOR3_BAG, "*.db3")))
pytestmark = pytest.mark.skipif(
    not os.path.isdir(FLOOR3_BAG),
    reason=f"needs the real floor3 bag mounted at {FLOOR3_BAG} (shared network dataset, "
           "not present on every machine)",
)
pytestmark = [
    pytestmark,
    pytest.mark.skipif(
        not _HAS_DB3 and not _rosbag2_py_available(),
        reason="floor3 bag is .mcap-only here and rosbag2_py (ROS2 stack) is not "
               "installed in this environment -- needs a ROS2-enabled container "
               "(e.g. eligsir-orbslam3) or a .db3-converted copy of the bag.",
    ),
]


def test_read_camera_info_returns_plausible_floor3_intrinsics():
    info = read_camera_info(FLOOR3_BAG)

    assert info["width"] > 0 and info["height"] > 0
    assert info["fx"] > 0 and info["fy"] > 0
    # cx/cy should sit roughly at image center for a standard RGB-D sensor.
    assert abs(info["cx"] - info["width"] / 2) < info["width"] * 0.25
    assert abs(info["cy"] - info["height"] / 2) < info["height"] * 0.25


def test_read_rgb_frames_near_timestamps_decodes_real_frames():
    # Bag starts at 1753172527.468061166 per metadata.yaml's starting_time -- pick
    # a timestamp a few seconds in so we're not right at the very first message.
    target_ts = 1753172530.0
    frames = read_rgb_frames_near_timestamps(FLOOR3_BAG, [target_ts], max_dt=0.5)

    assert target_ts in frames
    frame = frames[target_ts]
    assert isinstance(frame, np.ndarray)
    assert frame.ndim == 3 and frame.shape[2] == 3
    assert frame.dtype == np.uint8


def test_read_rgb_frames_near_timestamps_raises_on_unmatched_timestamp():
    # Timestamp far outside the bag's ~175.5s duration (metadata.yaml duration) --
    # no message can be within max_dt of it.
    bogus_ts = 1753172527.468061166 + 10_000.0

    with pytest.raises(ValueError, match="no match"):
        read_rgb_frames_near_timestamps(FLOOR3_BAG, [bogus_ts], max_dt=0.02)


def test_read_rgb_frames_near_timestamps_raises_on_missing_topic():
    with pytest.raises(ValueError, match="not found in bag"):
        read_rgb_frames_near_timestamps(
            FLOOR3_BAG, [1753172530.0], topic="/this/topic/does/not/exist", max_dt=0.5
        )


def test_read_camera_info_raises_on_missing_topic():
    with pytest.raises(ValueError, match="not found in bag"):
        read_camera_info(FLOOR3_BAG, topic="/this/topic/does/not/exist")
