"""#943: fr2_desk's registry entry must be internally consistent, not merely typed.

The point of these tests is that `eval/scenes.json`'s `n_source_frames` for a TUM
scene is a *derived* quantity — the rgb x depth x groundtruth triple-association
count at the project's own `tum_association_max_dt` tolerance — and NOT
`len(rgb.txt)` or `min(len(rgb), len(depth))`. Confusing those is the exact trap
#943 was filed about: the naive counts for fr2_desk are 2965 / 2964, while the
real answer is 2297.

Two layers, deliberately:

* the registry-lookup tests run everywhere (no dataset needed) and pin the number
  plus the path that #934 Stage 2 will pass as ``SCENE_SOURCE``;
* the re-derivation test is skipped when the NFS share is not mounted, and covers
  **fr3 as well as fr2_desk** so that it tests the derivation *rule* rather than a
  number someone typed. fr3's registered 2525 is the control: if the rule ever
  stops reproducing it, the rule — not the new entry — is what changed.
"""

import json
import os

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REGISTRY_PATH = os.path.join(REPO_ROOT, "eval", "scenes.json")

FR2_DESK_PATH = (
    "/data/datasets/TUM/tum_rgbd/"
    "freiburg2_desk/rgbd_dataset_freiburg2_desk"
)
FR2_DESK_N = 2297

# Naive derivations that #943 explicitly warns against, recorded so a future
# "fix" that re-bases the entry onto one of them fails loudly here.
FR2_DESK_NAIVE_N_RGB = 2965
FR2_DESK_NAIVE_N_MIN_RGB_DEPTH = 2964


def _registry():
    with open(REGISTRY_PATH) as f:
        return json.load(f)


def test_fr2_desk_entry_exists_and_is_well_formed():
    entry = _registry()["fr2_desk"]
    assert entry["type"] == "replay"
    assert entry["path"] == FR2_DESK_PATH
    assert entry["n_source_frames"] == FR2_DESK_N
    # The derivation must be recorded on the entry itself, per #943's "document
    # HOW, not just WHAT" requirement.
    assert "_associate_tum" in entry["note"]
    assert "max_dt=0.03" in entry["note"]


def test_fr2_desk_count_is_not_a_naive_derivation():
    n = _registry()["fr2_desk"]["n_source_frames"]
    assert n != FR2_DESK_NAIVE_N_RGB, "n_source_frames re-based onto len(rgb.txt)"
    assert n != FR2_DESK_NAIVE_N_MIN_RGB_DEPTH, (
        "n_source_frames re-based onto min(len(rgb), len(depth))"
    )


def test_registry_lookup_resolves_the_scene_source_stage2_will_pass():
    """The exact-path branch of registry_n_source_frames must hit.

    Relying on the basename fallback would be fragile here: three directories on
    the share carry the leaf name ``rgbd_dataset_freiburg2_desk`` (this one plus
    the ros2_raw/ and ros2_compressed/ bags).
    """
    from eligsir.utils.partition_eval import registry_n_source_frames

    assert registry_n_source_frames(FR2_DESK_PATH) == FR2_DESK_N
    assert registry_n_source_frames(FR2_DESK_PATH + "/") == FR2_DESK_N
    assert registry_n_source_frames("fr2_desk") == FR2_DESK_N


def test_registry_basename_fallback_is_not_ambiguous():
    """Only one registry entry may carry this basename, or the fallback refuses."""
    reg = _registry()
    basename = os.path.basename(FR2_DESK_PATH)
    matches = [
        name
        for name, entry in reg.items()
        if not name.startswith("_")
        and isinstance(entry, dict)
        and entry.get("path")
        and os.path.basename(os.path.normpath(entry["path"].rstrip("/"))) == basename
    ]
    assert matches == ["fr2_desk"]


@pytest.mark.parametrize(
    "scene_key, expected",
    [
        ("fr2_desk", FR2_DESK_N),
        ("fr3", 2525),
    ],
)
def test_registered_count_reproduces_from_the_dataset(scene_key, expected):
    """Re-derive n_source_frames from the dataset with the repo's own utility.

    fr3 is the control that makes this a test of the derivation rule. The
    tolerance is read from the project default rather than hardcoded, so a change
    to ``tum_association_max_dt`` fails here instead of silently re-basing every
    registered TUM count.
    """
    entry = _registry()[scene_key]
    path = entry["path"]
    if not os.path.isdir(path):
        pytest.skip(f"dataset not mounted: {path}")

    from scene.readers.tum import _associate_tum, _read_tum_list, _read_tum_poses

    max_dt = _tum_association_max_dt_default()

    assocs = _associate_tum(
        _read_tum_list(os.path.join(path, "rgb.txt")),
        _read_tum_list(os.path.join(path, "depth.txt")),
        _read_tum_poses(os.path.join(path, "groundtruth.txt")),
        max_dt=max_dt,
    )
    assert len(assocs) == expected == entry["n_source_frames"]

    # readTUMCameras drops associations whose image file is missing on disk, so
    # the registered count is only honest if nothing is skipped.
    missing = [
        a
        for a in assocs
        if not os.path.exists(os.path.join(path, a["rgb_rel"]))
        or not os.path.exists(os.path.join(path, a["depth_rel"]))
    ]
    assert not missing, f"{len(missing)} associated frames have missing files on disk"


def _tum_association_max_dt_default() -> float:
    """Read the project's TUM association tolerance from its declaration site."""
    import re

    args_path = os.path.join(REPO_ROOT, "arguments", "__init__.py")
    with open(args_path) as f:
        m = re.search(r"self\.tum_association_max_dt\s*=\s*([0-9.eE+-]+)", f.read())
    assert m, "could not locate tum_association_max_dt default"
    return float(m.group(1))
