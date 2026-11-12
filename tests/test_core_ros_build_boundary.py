"""Hard-cut guard for the EliGSiR core build/deployment boundary.

ROS2 packaging, ROS launch/config, RVizSplat and ROS-enabled images are owned by
``eligsir-project/eligsir-ros2``. Core may contain transport-neutral bag and
camera tooling, but it must not regain a second ROS deployment path.
"""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


REMOVED_PATHS = (
    "Dockerfile.cuda-ros2",
    "Dockerfile.gsplat-msgs-patch",
    "Dockerfile.rvizsplat",
    "docker/ros2.Dockerfile",
    "docker/ros_entrypoint.sh",
    "docker/rvizsplat_live.rviz",
    "docker/rvizsplat_novnc_start.sh",
    "ros2_profiles",
    "ros2_ws",
)


def _assert_absent(path: str, tokens: tuple[str, ...], *, owner: str) -> None:
    text = (ROOT / path).read_text(encoding="utf-8")
    found = [token for token in tokens if token in text]
    assert found == [], f"{owner} regained ROS ownership: {found}"


def test_ros_deployment_assets_are_not_owned_by_core() -> None:
    present = [path for path in REMOVED_PATHS if (ROOT / path).exists()]
    assert present == [], (
        "ROS deployment assets must live in eligsir-ros2, not core: "
        + ", ".join(present)
    )


def test_core_dockerfile_has_no_ros_or_rviz_targets() -> None:
    _assert_absent(
        "Dockerfile",
        (
            "ros2-jazzy-runtime",
            "eligsir-ros2-jazzy",
            "rvizsplat-jazzy",
            "rvizsplat-novnc",
            "/opt/ros/",
            "ros-jazzy-",
        ),
        owner="core Dockerfile",
    )


def test_core_compose_has_no_ros_services() -> None:
    _assert_absent(
        "docker-compose.yml",
        (
            "train-ros2:",
            "zenoh-router:",
            "rvizsplat:",
            "rvizsplat-webvnc:",
            "ELIGSIR_ROS2_IMAGE_TAG",
        ),
        owner="core Compose",
    )


def test_core_image_workflow_does_not_build_ros() -> None:
    _assert_absent(
        ".github/workflows/ci-public.yml",
        (
            "build-ros2:",
            "eligsir-ros2-jazzy",
            "ros2-jazzy-",
            '"ros2_ws/**"',
            '"Dockerfile.rvizsplat"',
        ),
        owner="core image workflow",
    )


def test_core_rocm_qualification_is_runtime_only() -> None:
    _assert_absent(
        ".github/workflows/ci-public.yml",
        (
            "variant: ros2-jazzy",
            "target: eligsir-ros2-jazzy",
            "ros2-jazzy-rocm-",
            '"ros2_ws/**"',
        ),
        owner="core ROCm qualification workflow",
    )


def test_core_profile_registry_has_no_ros_deployment_profile() -> None:
    # Comment lines are stripped first, unlike the other checks in this module:
    # #1091 left a comment in config.py naming the removed alias to explain why
    # it is gone. A prose mention is not ownership -- what must not come back is
    # the alias itself, in code.
    text = (ROOT / "eligsir" / "cli" / "config.py").read_text(encoding="utf-8")
    code = "\n".join(
        line for line in text.splitlines() if not line.lstrip().startswith("#")
    )
    assert "live/orbbec_ros2_rviz" not in code, (
        "core run-profile registry regained ROS ownership: live/orbbec_ros2_rviz"
    )
