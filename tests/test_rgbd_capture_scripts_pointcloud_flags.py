"""Issue #395 regression coverage.

An earlier triage pass (issue #368 follow-up, 2026-08-13) claimed neither
``scripts/capture_ros2_rgbd_sequence.py`` nor ``scripts/extract_rosbag_rgbd.py``
declared ``--pointcloud-preprocess``/``--pcd-voxel-size`` in their own argparse
parsers, so ``utils/rgbd_sequence_writer.py``'s ``getattr(args, ..., <literal>)``
fallback was "always taken". That claim was wrong (a dash/underscore grep
mismatch): both scripts have declared these flags -- with defaults matching
today's effective behavior ("none" / 0.0) -- since the original baseline
import (commit 324e95f, 2026-06-02). This test locks in:

1. Both scripts' parsers genuinely declare the flags (reachable, not dead
   argparse strings that never populate ``args``).
2. Parsing with no extra CLI args reproduces exactly today's effective
   values -- i.e. the change made for issue #395 (removing the now-redundant
   ``getattr`` fallback in ``utils/rgbd_sequence_writer.py`` in favor of bare
   attribute access) is behavior-neutral for the default invocation.
3. ``utils/rgbd_sequence_writer.py`` no longer carries a ``getattr(args, ...)``
   fallback for any of the eight ``pointcloud_preprocess_config`` fields
   (mirrors the dedicated guard in ``tests/test_getattr_fallback_freshness.py``,
   kept here too so this test file is self-contained evidence for #395).
"""
from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_SCRIPTS = _REPO_ROOT / "scripts"


def _load_parse_args(script_name: str):
    """Import a scripts/*.py module by path and return its parse_args()."""
    spec = importlib.util.spec_from_file_location(
        f"_issue395_{script_name.replace('.', '_')}", _SCRIPTS / script_name
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.parse_args


_CAPTURE_MIN_ARGV = ["--out", "/tmp/issue395_out"]
_EXTRACT_MIN_ARGV = ["--bag", "/tmp/issue395.bag", "--out", "/tmp/issue395_out"]

_EXPECTED_DEFAULTS = {
    "pointcloud_preprocess": "none",
    "pcd_voxel_size": 0.0,
    "pcd_outlier_filter": "none",
    "pcd_stat_nb_neighbors": 20,
    "pcd_stat_std_ratio": 2.0,
    "pcd_radius": 0.05,
    "pcd_min_neighbors": 4,
    "pcd_estimate_normals": False,
}


def _parse_with_argv(parse_args_fn, argv):
    old_argv = sys.argv
    sys.argv = ["prog"] + argv
    try:
        return parse_args_fn()
    finally:
        sys.argv = old_argv


def test_capture_ros2_script_declares_flags_with_current_effective_defaults():
    parse_args = _load_parse_args("capture_ros2_rgbd_sequence.py")
    args = _parse_with_argv(parse_args, _CAPTURE_MIN_ARGV)
    for flag, expected in _EXPECTED_DEFAULTS.items():
        assert hasattr(args, flag), f"--{flag.replace('_', '-')} not reachable on parsed args"
        assert getattr(args, flag) == expected, (
            f"{flag}: parsed default {getattr(args, flag)!r} != current effective value {expected!r}"
        )


def test_extract_rosbag_script_declares_flags_with_current_effective_defaults():
    parse_args = _load_parse_args("extract_rosbag_rgbd.py")
    args = _parse_with_argv(parse_args, _EXTRACT_MIN_ARGV)
    for flag, expected in _EXPECTED_DEFAULTS.items():
        assert hasattr(args, flag), f"--{flag.replace('_', '-')} not reachable on parsed args"
        assert getattr(args, flag) == expected, (
            f"{flag}: parsed default {getattr(args, flag)!r} != current effective value {expected!r}"
        )


def test_capture_ros2_script_flag_overrides_reach_args():
    parse_args = _load_parse_args("capture_ros2_rgbd_sequence.py")
    args = _parse_with_argv(
        parse_args,
        _CAPTURE_MIN_ARGV + ["--pointcloud-preprocess", "open3d", "--pcd-voxel-size", "0.02"],
    )
    assert args.pointcloud_preprocess == "open3d"
    assert args.pcd_voxel_size == 0.02


def test_extract_rosbag_script_flag_overrides_reach_args():
    parse_args = _load_parse_args("extract_rosbag_rgbd.py")
    args = _parse_with_argv(
        parse_args,
        _EXTRACT_MIN_ARGV + ["--pointcloud-preprocess", "open3d", "--pcd-voxel-size", "0.02"],
    )
    assert args.pointcloud_preprocess == "open3d"
    assert args.pcd_voxel_size == 0.02


def test_rgbd_sequence_writer_has_no_getattr_fallback_for_preprocess_fields():
    text = (_REPO_ROOT / "utils" / "rgbd_sequence_writer.py").read_text()
    for flag in _EXPECTED_DEFAULTS:
        pattern = re.compile(r'getattr\(\s*args\s*,\s*["\']' + re.escape(flag) + r'["\']\s*,')
        assert not pattern.search(text), (
            f"utils/rgbd_sequence_writer.py still has a getattr(args, {flag!r}, ...) fallback -- "
            "issue #395 removed these in favor of bare attribute access since both real callers "
            "always declare the flag."
        )
        assert f"args.{flag}" in text, f"utils/rgbd_sequence_writer.py no longer reads args.{flag} at all"
