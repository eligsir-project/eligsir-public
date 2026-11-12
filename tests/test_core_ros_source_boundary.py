"""Ratchet the core/ROS2 ownership boundary after the source hard cut."""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RETIRED_PATHS = (
    "contracts/ros_interop.py",
    "rate_publisher.py",
    "rate_publisher_all.py",
    "eligsir/contracts/ros_interop.py",
)
FORBIDDEN_IMPORT_ROOTS = {
    "rclpy",
    "rcl_interfaces",
    "sensor_msgs",
    "geometry_msgs",
    "nav_msgs",
    "tf2_ros",
    "tf2_msgs",
    "builtin_interfaces",
    "std_msgs",
    "cv_bridge",
    "gsplat_msgs",
    "eligsir_ros2",
}


def _root(name: str | None) -> str:
    return (name or "").split(".", 1)[0]


def _imports(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield node.lineno, _root(alias.name)
        elif isinstance(node, ast.ImportFrom):
            yield node.lineno, _root(node.module)


def test_retired_core_ros_paths_are_physically_absent():
    present = [path for path in RETIRED_PATHS if (ROOT / path).exists()]
    assert present == [], "retired ROS paths still exist: " + ", ".join(present)


def test_production_core_has_no_ros_imports():
    candidates = list((ROOT / "eligsir").rglob("*.py"))
    candidates += [path for path in ROOT.glob("*.py") if path.is_file()]
    violations = []
    for path in sorted(candidates):
        for line, imported_root in _imports(path):
            if imported_root in FORBIDDEN_IMPORT_ROOTS:
                violations.append(f"{path.relative_to(ROOT)}:{line}: {imported_root}")
    assert violations == [], "ROS dependency leaked into core:\n" + "\n".join(violations)


def test_ci_cpu_image_does_not_copy_retired_ros_publishers():
    text = (ROOT / "Dockerfile.ci-cpu").read_text(encoding="utf-8")
    assert "rate_publisher.py" not in text
    assert "rate_publisher_all.py" not in text
