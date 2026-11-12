"""Coordinate-convention helpers.

Internal EliGSiR poses are OpenCV/COLMAP camera-to-world:

``x = right, y = down, z = forward``.

ROS topic data must use ROS map/link convention at the boundary:

``x = forward, y = left, z = up``.

The helpers here are the only basis transforms that ROS ingestion/export code
should use. They keep the data path explicit:

ROS2 topics -> internal canonical mapper -> ROS2 topics.
"""

from __future__ import annotations

import numpy as np


CANONICAL_TO_ROS_MAP = np.array(
    [
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
    ],
    dtype=np.float32,
)


def canonical_pose_to_ros_link_matrix(matrix: np.ndarray) -> np.ndarray:
    """Convert internal canonical c2w pose to ROS map -> camera_link matrix."""
    pose = np.asarray(matrix, dtype=np.float32)
    basis = CANONICAL_TO_ROS_MAP
    out = np.eye(4, dtype=np.float32)
    out[:3, :3] = basis @ pose[:3, :3] @ basis.T
    out[:3, 3] = basis @ pose[:3, 3]
    return out


def ros_link_pose_to_canonical_matrix(matrix: np.ndarray) -> np.ndarray:
    """Convert ROS map -> camera_link pose to internal canonical c2w matrix."""
    pose = np.asarray(matrix, dtype=np.float32)
    basis = CANONICAL_TO_ROS_MAP
    out = np.eye(4, dtype=np.float32)
    out[:3, :3] = basis.T @ pose[:3, :3] @ basis
    out[:3, 3] = basis.T @ pose[:3, 3]
    return out


def ros_optical_pose_to_canonical_matrix(matrix: np.ndarray) -> np.ndarray:
    """Convert ROS map -> optical-frame pose to internal canonical c2w matrix."""
    pose = np.asarray(matrix, dtype=np.float32)
    basis = CANONICAL_TO_ROS_MAP
    out = np.eye(4, dtype=np.float32)
    out[:3, :3] = basis.T @ pose[:3, :3]
    out[:3, 3] = basis.T @ pose[:3, 3]
    return out


def ros_pose_to_canonical_matrix(matrix: np.ndarray, convention) -> np.ndarray:
    """Dispatch a raw ROS pose matrix to the correct canonical conversion.

    ``convention`` selects which axis convention the *pose message itself*
    was published in:

    - ``"link"``: REP-103 body/link frame (x-forward, y-left, z-up -- e.g.
      ORB-SLAM3's ``/slam/pose``, or a TUM ``/openni_camera`` TF frame).
      Routed to :func:`ros_link_pose_to_canonical_matrix`.
    - anything else (``"optical"``, ``None``, unrecognized): REP-103 optical
      frame (x-right, y-down, z-forward -- the standard ROS camera-driver
      convention). Routed to :func:`ros_optical_pose_to_canonical_matrix`.
      This mirrors the pre-existing fallback semantics at the call sites in
      ``eligsir/io/sources/ros2_live.py`` (issue #313): optical is the
      default/fallback, not link.

    This is the single place that decides which of the two conversions a
    *pose-message-is-the-camera* call site should use -- both the pose/odom
    branch and the tf-pose branch in ``ros2_live.py`` route through this
    function so the selection logic exists in exactly one importable,
    ROS2-free, unit-testable place.
    """
    if convention == "link":
        return ros_link_pose_to_canonical_matrix(matrix)
    return ros_optical_pose_to_canonical_matrix(matrix)


def canonical_points_to_ros(points: np.ndarray) -> np.ndarray:
    """Convert internal canonical xyz points to ROS map coordinates."""
    pts = np.asarray(points, dtype=np.float32)
    return (CANONICAL_TO_ROS_MAP @ pts.T).T.astype(np.float32)


def ros_points_to_canonical(points: np.ndarray) -> np.ndarray:
    """Convert ROS map xyz points to internal canonical coordinates."""
    pts = np.asarray(points, dtype=np.float32)
    return (CANONICAL_TO_ROS_MAP.T @ pts.T).T.astype(np.float32)
