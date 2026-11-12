from __future__ import annotations

import pytest

from eligsir.io.sources.datasets.ros2_live import (
    ROS2LiveStreamingFrameSource,
    SourceFailedError,
)
from eligsir.io.sources.registry import detect_source_kind, list_sources
from eligsir.io.sources.ros2_live import LegacyROSPathRemoved, ROS2LiveRGBDSource
from eligsir.slam.pose_sources import normalize_pose_source


def test_ros_live_is_not_a_registered_core_source() -> None:
    assert "ros2_live" not in list_sources()
    assert detect_source_kind("ros2://live") is None


def test_legacy_direct_ros_source_fails_closed() -> None:
    with pytest.raises(LegacyROSPathRemoved):
        ROS2LiveRGBDSource()


def test_legacy_streaming_adapter_uses_the_same_fail_closed_error() -> None:
    assert SourceFailedError is LegacyROSPathRemoved
    with pytest.raises(LegacyROSPathRemoved):
        ROS2LiveStreamingFrameSource()


def test_pose_resolution_has_no_ros_transport_special_case() -> None:
    with pytest.raises(ValueError, match="ROS tracking belongs to eligsir-ros2"):
        normalize_pose_source("ros_slam")


def test_removed_topic_name_heuristic_cannot_enable_core_behavior() -> None:
    import eligsir.slam.pose_sources as pose_sources

    assert not hasattr(pose_sources, "ros2_live_pose_is_external_slam")
