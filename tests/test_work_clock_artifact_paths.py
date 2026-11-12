from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from eligsir.runtime.artifact_paths import (
    point_cloud_artifact_paths,
    publish_legacy_iteration_alias,
)
from eligsir.runtime.config import CanonicalRuntimePolicy
from eligsir.runtime.production_application import CanonicalMappingRuntime


def test_point_cloud_path_names_both_semantic_work_clocks(tmp_path):
    paths = point_cloud_artifact_paths(
        tmp_path, optimizer_updates=630, supervised_views=2520
    )

    assert paths.directory.name == "optimizer_updates_630_supervised_views_2520"
    assert paths.point_cloud == paths.directory / "point_cloud.ply"
    assert paths.legacy_directory.name == "iteration_630"


def test_legacy_iteration_directory_is_only_a_compatibility_alias(tmp_path):
    paths = point_cloud_artifact_paths(
        tmp_path, optimizer_updates=630, supervised_views=2520
    )
    paths.directory.mkdir(parents=True)
    paths.point_cloud.write_bytes(b"ply\n")

    publish_legacy_iteration_alias(paths)

    assert paths.legacy_directory.is_symlink()
    assert paths.legacy_directory.resolve() == paths.directory.resolve()
    assert paths.legacy_point_cloud.read_bytes() == b"ply\n"


def test_canonical_runtime_flush_saves_to_work_clock_qualified_directory(tmp_path):
    runtime = object.__new__(CanonicalMappingRuntime)
    runtime._runtime_policy = CanonicalRuntimePolicy(model_path=str(tmp_path))
    runtime.initialized = True
    runtime.iteration = 630
    runtime._work_clock_owner = SimpleNamespace(
        snapshot=lambda: SimpleNamespace(
            optimizer_updates=630,
            supervised_rgb_views=2520,
        )
    )

    def save_ply(path):
        output = Path(path)
        output.parent.mkdir(parents=True)
        output.write_bytes(b"ply\n")

    runtime.gaussians = SimpleNamespace(save_ply=save_ply)

    runtime.flush()

    canonical = (
        tmp_path
        / "point_cloud"
        / "optimizer_updates_630_supervised_views_2520"
        / "point_cloud.ply"
    )
    assert canonical.is_file()
    assert (tmp_path / "point_cloud" / "iteration_630").is_symlink()
