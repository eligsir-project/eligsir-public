from __future__ import annotations

from types import SimpleNamespace

from eligsir.runtime.production_application import CanonicalMappingRuntime
from eligsir.runtime.config import CanonicalRuntimePolicy


def test_flush_saves_live_map_in_standard_model_output(tmp_path):
    def save_ply(path):
        from pathlib import Path

        output = Path(path)
        output.parent.mkdir(parents=True)
        output.write_bytes(b"ply\n")

    runtime = object.__new__(CanonicalMappingRuntime)
    runtime._runtime_policy = CanonicalRuntimePolicy(model_path=str(tmp_path))
    runtime.initialized = True
    runtime.iteration = 42
    runtime._work_clock_owner = SimpleNamespace(
        snapshot=lambda: SimpleNamespace(
            optimizer_updates=42,
            supervised_rgb_views=168,
        )
    )
    runtime.gaussians = SimpleNamespace(save_ply=save_ply)

    runtime.flush()

    canonical = (
        tmp_path
        / "point_cloud"
        / "optimizer_updates_42_supervised_views_168"
        / "point_cloud.ply"
    )
    assert canonical.is_file()
    assert (tmp_path / "point_cloud" / "iteration_42").is_symlink()


def test_flush_before_bootstrap_has_no_map_to_save(tmp_path):
    runtime = object.__new__(CanonicalMappingRuntime)
    runtime._runtime_policy = CanonicalRuntimePolicy(model_path=str(tmp_path))
    runtime.initialized = False

    assert runtime.flush() is None
