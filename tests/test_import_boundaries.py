"""Import-boundary tests for the ROS-free core package surface."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import textwrap


def test_core_contract_imports_do_not_load_optional_integrations():
    """Ordinary core imports must not import ROS or viewer integrations."""

    code = textwrap.dedent(
        """
        import sys

        import eligsir.cli.config
        import eligsir.contracts.runtime
        import eligsir.contracts.types
        import eligsir.core.online_state
        import eligsir.core.stop_policy
        import eligsir.io.snapshots
        import eligsir.io.sources
        import eligsir.legacy
        import eligsir.mapping.scheduler
        import eligsir.mapping.strategies.registry
        import eligsir.runtime.application
        import eligsir.runtime.save_worker
        import eligsir.slam
        import eligsir.viewers

        forbidden = {
            "rclpy",
            "cv_bridge",
            "sensor_msgs",
            "geometry_msgs",
            "nav_msgs",
            "tf2_ros",
            "open3d",
            "rerun",
            "fastapi",
            "uvicorn",
            "websockets",
            "rosbags",
            "cv2",
        }
        loaded = forbidden.intersection(sys.modules)
        if loaded:
            raise SystemExit(f"optional integrations imported by core: {sorted(loaded)}")
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr or result.stdout


def test_removed_ros_entrypoints_and_viewer_are_physically_absent():
    """Hard-cut modules must not remain as aliases or compatibility shims."""

    root = Path(__file__).resolve().parents[1]
    removed = [
        root / "train_ros_node.py",
        root / "eligsir" / "cli" / "ros_node.py",
        root / "eligsir" / "viewers" / "ros2_rviz.py",
        root / "viz" / "ros2_rviz.py",
    ]
    assert not [str(path.relative_to(root)) for path in removed if path.exists()]
