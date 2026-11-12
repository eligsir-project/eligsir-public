"""Compatibility import for the removed core ROS live source."""

from eligsir.io.sources.ros2_live import LegacyROSPathRemoved

SourceFailedError = LegacyROSPathRemoved


class ROS2LiveStreamingFrameSource:
    """Fail-closed migration tombstone; not an executable source."""

    lazy_frames = True

    def __init__(self, *args, **kwargs) -> None:
        raise LegacyROSPathRemoved(
            "The core ROS live source was removed. Use eligsir-project/eligsir-ros2 "
            "and submit SensorPacket values to MappingApplication."
        )


__all__ = ["ROS2LiveStreamingFrameSource", "SourceFailedError"]
