"""#494: fr1_desk must have a deterministic full-trajectory denominator."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "eval" / "scenes.json"
SCENE = Path(
    "/data/datasets/TUM/"
    "rgbd_dataset_freiburg1_desk"
)
N_SOURCE_FRAMES = 596


def _entry() -> dict:
    return json.loads(REGISTRY.read_text())["fr1_desk"]


def test_fr1desk_registry_resolves_only_the_canonical_scene() -> None:
    from eligsir.utils.partition_eval import registry_n_source_frames

    entry = _entry()
    assert entry["type"] == "replay"
    assert entry["path"] == str(SCENE)
    assert entry["n_source_frames"] == N_SOURCE_FRAMES
    assert entry["basename_fallback"] is False
    assert registry_n_source_frames("fr1_desk") == N_SOURCE_FRAMES
    assert registry_n_source_frames(str(SCENE)) == N_SOURCE_FRAMES
    assert (
        registry_n_source_frames(f"/unregistered/{SCENE.name}") is None
    )


def test_fr1desk_registered_count_matches_project_tum_association() -> None:
    if not SCENE.is_dir():
        pytest.skip(f"dataset not mounted: {SCENE}")

    from scene.readers.tum import _associate_tum, _read_tum_list, _read_tum_poses

    associations = _associate_tum(
        _read_tum_list(str(SCENE / "rgb.txt")),
        _read_tum_list(str(SCENE / "depth.txt")),
        _read_tum_poses(str(SCENE / "groundtruth.txt")),
        max_dt=0.03,
    )
    assert len(associations) == N_SOURCE_FRAMES == _entry()["n_source_frames"]
    assert all(
        (SCENE / row["rgb_rel"]).is_file()
        and (SCENE / row["depth_rel"]).is_file()
        for row in associations
    )
