"""Removed legacy ROS live-source path.

ROS transport, synchronization, QoS, TF/pose association and bounded ingress
belong to ``eligsir-project/eligsir-ros2``. Core no longer contains an
executable ROS source. This module exists only so a stale import fails with an
actionable migration error instead of silently selecting another path.
"""

from __future__ import annotations


class LegacyROSPathRemoved(RuntimeError):
    pass


class ROS2LiveRGBDSource:
    """Non-executable tombstone for the removed direct-ROS core source."""

    def __init__(self, *args, **kwargs) -> None:
        raise LegacyROSPathRemoved(
            "ROS2LiveRGBDSource was removed by the canonical data-path hard cut. "
            "Run ROS ingress with eligsir-project/eligsir-ros2 and submit "
            "canonical SensorPacket values to MappingApplication."
        )


__all__ = ["LegacyROSPathRemoved", "ROS2LiveRGBDSource"]
