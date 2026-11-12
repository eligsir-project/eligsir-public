"""Negative ratchet for the destructive core ROS live-source hard cut."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "human_launch_scripts"
FORBIDDEN_EXECUTABLE_PATHS = (
    "eligsir_ros2.live_launcher",
    "ros2://live",
    "rgbd_splitter_node",
)
TOMBSTONE_SCRIPTS = (
    "04_run_live_training.sh",
    "05_run_offline_rosbag.sh",
    "05c_run_offline_rosbag_new_syncer.sh",
    "07_playback_recorded_bags.sh",
    "08_tum_live_compare.sh",
    "20_tum_fr3_orbslam_rviz_webvnc.sh",
    "22_run_offline_realsense_bag.sh",
)


def _shell_scripts() -> list[Path]:
    scripts = sorted(SCRIPTS.glob("*.sh"))
    assert scripts, f"no launch scripts found under {SCRIPTS}"
    return scripts


def _significant_lines(path: Path) -> list[tuple[int, str]]:
    return [
        (number, line)
        for number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), 1
        )
        if not line.lstrip().startswith("#")
    ]


@pytest.mark.parametrize("script", _shell_scripts(), ids=lambda path: path.name)
def test_core_scripts_cannot_execute_deleted_ros_live_paths(script: Path) -> None:
    offenders = [
        f"{number}: {line.strip()}"
        for number, line in _significant_lines(script)
        if any(forbidden in line for forbidden in FORBIDDEN_EXECUTABLE_PATHS)
    ]
    assert not offenders, (
        "core no longer owns ROS live or bag orchestration; use "
        "eligsir-project/eligsir-ros2. Offending lines: "
        f"{offenders}"
    )


@pytest.mark.parametrize("script_name", TOMBSTONE_SCRIPTS)
def test_removed_ros_launchers_fail_closed(script_name: str) -> None:
    result = subprocess.run(
        ["bash", str(SCRIPTS / script_name)], capture_output=True, text=True
    )
    assert result.returncode == 64
    assert "eligsir-ros2" in result.stderr
