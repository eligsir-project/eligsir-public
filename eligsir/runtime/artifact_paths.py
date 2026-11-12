"""Work-clock-qualified paths for canonical mapping artifacts."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


def _non_negative_int(value: object, name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an exact integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


@dataclass(frozen=True, slots=True)
class PointCloudArtifactPaths:
    """Paths to current and legacy point cloud artifacts with work-clock names."""
    directory: Path
    point_cloud: Path
    legacy_directory: Path
    legacy_point_cloud: Path
    optimizer_updates: int
    supervised_views: int


def point_cloud_artifact_paths(
    model_path: str | Path,
    *,
    optimizer_updates: int,
    supervised_views: int,
) -> PointCloudArtifactPaths:
    """Name one snapshot by both optimizer and supervision work clocks."""

    updates = _non_negative_int(optimizer_updates, "optimizer_updates")
    views = _non_negative_int(supervised_views, "supervised_views")
    root = Path(model_path) / "point_cloud"
    directory = root / f"optimizer_updates_{updates}_supervised_views_{views}"
    legacy_directory = root / f"iteration_{updates}"
    return PointCloudArtifactPaths(
        directory=directory,
        point_cloud=directory / "point_cloud.ply",
        legacy_directory=legacy_directory,
        legacy_point_cloud=legacy_directory / "point_cloud.ply",
        optimizer_updates=updates,
        supervised_views=views,
    )


def publish_legacy_iteration_alias(paths: PointCloudArtifactPaths) -> None:
    """Publish the old ``iteration_N`` directory as a relative symlink only."""

    if not paths.directory.is_dir() or not paths.point_cloud.is_file():
        raise RuntimeError("canonical point-cloud artifact is missing or empty")
    if paths.point_cloud.stat().st_size <= 0:
        raise RuntimeError("canonical point-cloud artifact is missing or empty")
    paths.legacy_directory.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(paths.legacy_directory):
        if (
            paths.legacy_directory.is_symlink()
            and paths.legacy_directory.resolve() == paths.directory.resolve()
        ):
            return
        raise FileExistsError(
            f"legacy point-cloud path already exists: {paths.legacy_directory}"
        )
    paths.legacy_directory.symlink_to(paths.directory.name, target_is_directory=True)


__all__ = [
    "PointCloudArtifactPaths",
    "point_cloud_artifact_paths",
    "publish_legacy_iteration_alias",
]
