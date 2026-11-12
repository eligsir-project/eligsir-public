"""Tests for lightweight frame-source detection."""

from __future__ import annotations

import subprocess
import sys
import textwrap

from eligsir.io.sources import FrameSource
from eligsir.io.sources.registry import detect_source_kind, list_sources


def test_source_registry_import_is_lightweight():
    code = textwrap.dedent(
        """
        import sys
        import eligsir.io.sources
        forbidden = {"cv2", "open3d", "rosbags", "rclpy"}
        loaded = forbidden.intersection(sys.modules)
        if loaded:
            raise SystemExit(f"source registry imported optional deps: {sorted(loaded)}")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr or result.stdout


def test_frame_source_protocol_accepts_iterable_source():
    class DummySource:
        def __iter__(self):
            return iter(())

    assert isinstance(DummySource(), FrameSource)


def test_rosbag_source_compatibility_wrapper():
    from eligsir.io.sources.rosbag import RosbagFileRGBDSource
    from eligsir.utils.rosbag_rgbd_source import RosbagFileRGBDSource as CompatRosbagFileRGBDSource

    assert CompatRosbagFileRGBDSource is RosbagFileRGBDSource


def test_detect_generic_rgbd_sequence(tmp_path):
    (tmp_path / "frames.jsonl").write_text("", encoding="utf-8")
    (tmp_path / "intrinsics.json").write_text("{}", encoding="utf-8")
    assert detect_source_kind(str(tmp_path)) == "rgbd_sequence"


def test_detect_tum_rgbd_sequence(tmp_path):
    (tmp_path / "rgb.txt").write_text("", encoding="utf-8")
    (tmp_path / "depth.txt").write_text("", encoding="utf-8")
    (tmp_path / "groundtruth.txt").write_text("", encoding="utf-8")
    assert detect_source_kind(str(tmp_path)) == "tum_rgbd"


def test_list_sources_contains_canonical_sources():
    names = list_sources()
    assert "rgbd_sequence" in names
    assert "tum_rgbd" in names
    assert "ros2_bag" in names
