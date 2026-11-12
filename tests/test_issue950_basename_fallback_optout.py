"""Issue #950 follow-up: a registry entry can opt out of the basename fallback.

Background (see #943, #950): `registry_n_source_frames`'s basename-fallback
branch matches on leaf directory name and only refuses when more than one
*registry entry* shares that basename -- but real directories on disk with the
same leaf name that were never registered (e.g. the fr2_desk ROS2 bag
directories) are invisible to that guard. A run whose ``SCENE_SOURCE`` points
at one of those unregistered bag directories previously silently resolved to
the TUM entry's ``n_source_frames`` (2297), which is wrong for a bag (bags
don't drop unassociated frames the way TUM's rgb/depth/groundtruth
association step does).

This test covers the opt-out: an entry with ``"basename_fallback": false``
is never considered by the basename-fallback branch, while entries without
the field (the default -- and every pre-existing registry entry) keep
resolving via basename fallback exactly as before.
"""

import json
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REGISTRY_PATH = os.path.join(REPO_ROOT, "eval", "scenes.json")

FR2_DESK_PATH = (
    "/data/datasets/TUM/tum_rgbd/"
    "freiburg2_desk/rgbd_dataset_freiburg2_desk"
)
FR2_DESK_N = 2297


def _write_registry(tmp_path, entries):
    reg_dir = tmp_path / "eval"
    reg_dir.mkdir(exist_ok=True)
    reg_path = reg_dir / "scenes.json"
    reg_path.write_text(json.dumps(entries))
    return reg_path


def _lookup_with_registry(monkeypatch, tmp_path, entries, scene_source):
    """Call registry_n_source_frames against a synthetic eval/scenes.json.

    registry_n_source_frames resolves the registry path relative to
    utils/partition_eval.py's own __file__ (``.../utils/../eval/scenes.json``),
    so we monkeypatch its module-level __file__ to point at a fake
    ``<tmp_path>/utils/partition_eval.py`` location whose sibling ``eval/``
    holds our synthetic registry.
    """
    import eligsir.utils.partition_eval as pe

    _write_registry(tmp_path, entries)
    fake_utils_dir = tmp_path / "utils"
    fake_utils_dir.mkdir(exist_ok=True)
    monkeypatch.setattr(pe, "__file__", str(fake_utils_dir / "partition_eval.py"))
    return pe.registry_n_source_frames(scene_source)


def test_entry_without_field_still_resolves_via_basename_fallback(monkeypatch, tmp_path):
    """(a) Regression guard: omitting the field preserves today's behavior."""
    entries = {
        "scene_a": {
            "type": "replay",
            "path": "/data/alias/mount/leafname",
            "n_source_frames": 1234,
        }
    }
    # Query with a DIFFERENT parent path but the SAME basename -- only the
    # basename-fallback branch can resolve this.
    n = _lookup_with_registry(
        monkeypatch, tmp_path, entries, "/data/container_alias/leafname"
    )
    assert n == 1234


def test_entry_with_field_false_does_not_resolve_via_basename_fallback(monkeypatch, tmp_path):
    """(b) basename_fallback: false must refuse the basename-fallback branch."""
    entries = {
        "scene_b": {
            "type": "replay",
            "path": "/data/alias/mount/leafname",
            "n_source_frames": 1234,
            "basename_fallback": False,
        }
    }
    n = _lookup_with_registry(
        monkeypatch, tmp_path, entries, "/data/container_alias/leafname"
    )
    assert n is None


def test_entry_with_field_false_still_resolves_exact_path_and_scene_key(monkeypatch, tmp_path):
    """(c) The opt-out only disables the basename branch, not exact-path/key matching."""
    exact_path = "/data/alias/mount/leafname"
    entries = {
        "scene_b": {
            "type": "replay",
            "path": exact_path,
            "n_source_frames": 1234,
            "basename_fallback": False,
        }
    }
    assert _lookup_with_registry(monkeypatch, tmp_path, entries, exact_path) == 1234
    # trailing slash normalization must still work for exact-path matching
    assert _lookup_with_registry(monkeypatch, tmp_path, entries, exact_path + "/") == 1234
    # scene-key matching
    assert _lookup_with_registry(monkeypatch, tmp_path, entries, "scene_b") == 1234


def test_real_fr2_desk_entry_exact_path_hits_but_bag_style_leaf_now_misses():
    """(d) The real fr2_desk entry: exact path resolves, a same-leaf bag path does not."""
    from eligsir.utils.partition_eval import registry_n_source_frames

    # Exact path (unaffected by the opt-out).
    assert registry_n_source_frames(FR2_DESK_PATH) == FR2_DESK_N

    # A bag-style path sharing the same leaf directory name, previously
    # silently mis-resolving to FR2_DESK_N via the basename fallback.
    bag_style_path = "/data/datasets/TUM/ros2_raw/rgbd_dataset_freiburg2_desk"
    assert registry_n_source_frames(bag_style_path) is None

    compressed_bag_style_path = (
        "/data/datasets/TUM/ros2_compressed/rgbd_dataset_freiburg2_desk"
    )
    assert registry_n_source_frames(compressed_bag_style_path) is None
