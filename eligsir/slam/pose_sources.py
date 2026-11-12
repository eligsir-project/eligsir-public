"""Canonical core pose-source names.

Transport ownership is intentionally absent here. ROS topic/TF semantics live in
``eligsir-ros2``; core only resolves abstract tracking-provider names.
"""

from __future__ import annotations

DATASET_POSE_SOURCE = "dataset_pose"
CANONICAL_POSE_SOURCES = {
    DATASET_POSE_SOURCE,
    "open3d_odometry_live",
    "open3d_icp_live",
}

_REMOVED_POSE_SOURCES = {
    "precomputed": "use 'dataset_pose' for frame-attached poses",
    "ros_slam": "ROS tracking belongs to eligsir-ros2",
}


def normalize_pose_source(name: str | None, *, warn: bool = False) -> str:
    """Return a canonical core pose-source name or fail closed."""
    del warn
    value = str(name or DATASET_POSE_SOURCE)
    if value not in CANONICAL_POSE_SOURCES:
        detail = _REMOVED_POSE_SOURCES.get(value)
        if detail is not None:
            raise ValueError(f"pose_source {value!r} was removed: {detail}")
        raise ValueError(
            f"unknown pose_source {value!r}; available: {sorted(CANONICAL_POSE_SOURCES)}"
        )
    return value


def is_dataset_pose_source(name: str | None) -> bool:
    """Return True when *name* selects frame-embedded poses."""
    return normalize_pose_source(name) == DATASET_POSE_SOURCE


__all__ = [
    "CANONICAL_POSE_SOURCES",
    "DATASET_POSE_SOURCE",
    "is_dataset_pose_source",
    "normalize_pose_source",
]
