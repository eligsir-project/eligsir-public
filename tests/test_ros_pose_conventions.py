"""Pure convention tests for the ROS-link/optical <-> canonical basis math.

ROS message construction/parsing is owned and tested by ``eligsir-ros2``.
Core retains only the numerical convention transforms because they are plain
matrix math and part of the canonical contract.
"""

import numpy as np

from eligsir.contracts.conventions import (
    CANONICAL_TO_ROS_MAP,
    canonical_pose_to_ros_link_matrix,
    ros_link_pose_to_canonical_matrix,
    ros_optical_pose_to_canonical_matrix,
    ros_pose_to_canonical_matrix,
)


def _nontrivial_ros_pose_matrix():
    angle = np.deg2rad(20.0)
    matrix = np.eye(4, dtype=np.float32)
    matrix[:3, :3] = np.array(
        [
            [np.cos(angle), 0.0, np.sin(angle)],
            [0.0, 1.0, 0.0],
            [-np.sin(angle), 0.0, np.cos(angle)],
        ],
        dtype=np.float32,
    )
    matrix[:3, 3] = np.array([0.2, -0.3, 1.4], dtype=np.float32)
    return matrix


def test_ros_link_pose_round_trip_to_canonical():
    canonical = np.eye(4, dtype=np.float32)
    canonical[:3, 3] = np.array([0.2, -0.3, 1.4], dtype=np.float32)
    ros_pose = canonical_pose_to_ros_link_matrix(canonical)
    np.testing.assert_allclose(
        ros_link_pose_to_canonical_matrix(ros_pose), canonical, atol=1e-6
    )


def test_ros_link_plus_static_optical_matches_canonical_pose():
    canonical = np.eye(4, dtype=np.float32)
    angle = np.deg2rad(20.0)
    canonical[:3, :3] = np.array(
        [
            [np.cos(angle), 0.0, np.sin(angle)],
            [0.0, 1.0, 0.0],
            [-np.sin(angle), 0.0, np.cos(angle)],
        ],
        dtype=np.float32,
    )
    canonical[:3, 3] = np.array([0.2, -0.3, 1.4], dtype=np.float32)

    ros_link = canonical_pose_to_ros_link_matrix(canonical)
    link_to_optical = np.eye(4, dtype=np.float32)
    link_to_optical[:3, :3] = CANONICAL_TO_ROS_MAP
    ros_optical = ros_link @ link_to_optical

    np.testing.assert_allclose(
        ros_optical_pose_to_canonical_matrix(ros_optical), canonical, atol=1e-6
    )


def test_link_convention_dispatches_to_link_conversion():
    matrix = _nontrivial_ros_pose_matrix()
    np.testing.assert_allclose(
        ros_pose_to_canonical_matrix(matrix, "link"),
        ros_link_pose_to_canonical_matrix(matrix),
        atol=1e-6,
    )


def test_optical_convention_dispatches_to_optical_conversion():
    matrix = _nontrivial_ros_pose_matrix()
    np.testing.assert_allclose(
        ros_pose_to_canonical_matrix(matrix, "optical"),
        ros_optical_pose_to_canonical_matrix(matrix),
        atol=1e-6,
    )


def test_link_and_optical_results_are_distinct():
    matrix = _nontrivial_ros_pose_matrix()
    link_result = ros_pose_to_canonical_matrix(matrix, "link")
    optical_result = ros_pose_to_canonical_matrix(matrix, "optical")
    assert not np.allclose(link_result, optical_result, atol=1e-3)


def test_link_vs_optical_relationship_matches_basis_rotation():
    matrix = _nontrivial_ros_pose_matrix()
    link_result = ros_pose_to_canonical_matrix(matrix, "link")
    optical_result = ros_pose_to_canonical_matrix(matrix, "optical")

    expected_link_rotation = optical_result[:3, :3] @ CANONICAL_TO_ROS_MAP
    np.testing.assert_allclose(
        link_result[:3, :3], expected_link_rotation, atol=1e-5
    )
    np.testing.assert_allclose(
        link_result[:3, 3], optical_result[:3, 3], atol=1e-6
    )


def test_unknown_or_none_convention_falls_back_to_optical():
    matrix = _nontrivial_ros_pose_matrix()
    expected = ros_optical_pose_to_canonical_matrix(matrix)
    np.testing.assert_allclose(
        ros_pose_to_canonical_matrix(matrix, None), expected, atol=1e-6
    )
    np.testing.assert_allclose(
        ros_pose_to_canonical_matrix(matrix, "bogus_convention"), expected, atol=1e-6
    )
