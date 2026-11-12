import importlib
import json
from pathlib import Path

import pytest

from eligsir.slam.pose_sources import (
    CANONICAL_POSE_SOURCES,
    DATASET_POSE_SOURCE,
    is_dataset_pose_source,
    normalize_pose_source,
)
from eligsir.tracking.factory import (
    EXTERNAL_POSE_SOURCES,
    build_tracking_frontend,
    build_tracking_provider,
    validate_frontend_name,
)


def test_canonical_pose_source_set_has_one_external_pose_name():
    assert DATASET_POSE_SOURCE == "dataset_pose"
    assert EXTERNAL_POSE_SOURCES == ("dataset_pose",)
    assert CANONICAL_POSE_SOURCES == {"dataset_pose", "open3d_odometry_live", "open3d_icp_live"}


def test_dataset_pose_is_the_only_already_attached_pose_token():
    assert normalize_pose_source(None) == "dataset_pose"
    assert normalize_pose_source("dataset_pose") == "dataset_pose"
    assert is_dataset_pose_source("dataset_pose") is True


@pytest.mark.parametrize("removed", ["precomputed", "ros_slam"])
def test_removed_pose_source_names_fail_closed(removed):
    with pytest.raises(ValueError):
        normalize_pose_source(removed)
    with pytest.raises(KeyError):
        validate_frontend_name(removed)
    with pytest.raises(KeyError):
        build_tracking_frontend(removed)
    with pytest.raises(KeyError):
        build_tracking_provider(removed)


def test_ros_slam_failure_points_at_ros_repository():
    with pytest.raises(KeyError, match="eligsir-ros2"):
        build_tracking_provider("ros_slam")


def test_precomputed_failure_points_at_dataset_pose():
    with pytest.raises(KeyError, match="dataset_pose"):
        build_tracking_provider("precomputed")


def test_dataset_pose_passthrough_remains_constructible():
    frontend = build_tracking_frontend("dataset_pose")
    provider = build_tracking_provider("dataset_pose")
    assert frontend is not None
    assert provider.provider_name == "dataset_pose"


def test_retired_adapter_and_compatibility_packages_are_gone():
    with pytest.raises(ImportError):
        importlib.import_module("eligsir.slam.adapters.precomputed")
    with pytest.raises(ImportError):
        importlib.import_module("slam.adapters.precomputed")
    with pytest.raises(ImportError):
        importlib.import_module("utils.pose_sources")


def test_removed_ros_live_helpers_are_not_exported():
    pose_sources = importlib.import_module("eligsir.slam.pose_sources")
    for name in (
        "LEGACY_PRECOMPUTED_POSE_SOURCE",
        "ROS_SLAM_POSE_SOURCE",
        "backend_name_for_pose_source",
        "resolve_pose_source_for_frame_source",
        "ros2_live_pose_is_external_slam",
    ):
        assert not hasattr(pose_sources, name)


def test_shipped_streaming_configs_use_only_canonical_pose_sources():
    config_root = Path(__file__).resolve().parents[1] / "configs" / "streaming"
    for config_path in config_root.glob("*.json"):
        config = json.loads(config_path.read_text())
        if "pose_source" in config:
            assert config["pose_source"] in CANONICAL_POSE_SOURCES, config_path


@pytest.mark.parametrize(
    "relative_path",
    (
        "scripts/run_benchmark.sh",
        "scripts/needle_opt_sweep.py",
        "scripts/benchmark_slam_isolation.py",
    ),
)
def test_executable_pose_source_inputs_use_dataset_pose(relative_path):
    source = (Path(__file__).resolve().parents[1] / relative_path).read_text(
        encoding="utf-8"
    )
    assert "precomputed" not in source
    assert "dataset_pose" in source
