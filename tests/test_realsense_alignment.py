"""Checks for RealSense pyrealsense2 integration and debug visualization."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eligsir.utils.streaming_frames import _write_depth_color_alignment_debug  # noqa: E402


def test_pyrealsense2_importable():
    """Verify pyrealsense2 is installed and importable."""
    import pyrealsense2 as rs  # noqa: F401


def test_alignment_debug_writes_contact_sheet_and_metadata(tmp_path):
    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    rgb[..., 1] = 100
    raw_depth = np.ones((2, 2), dtype=np.float32)
    aligned_depth = np.ones((4, 4), dtype=np.float32)
    aligned_depth[:, 2:] = 1.5

    record = _write_depth_color_alignment_debug(
        str(tmp_path),
        0,
        rgb,
        raw_depth,
        aligned_depth,
        {"depth_alignment_enabled": True, "extrinsic_source": "test"},
    )

    files = record["files"]
    assert Path(files["contact_sheet"]).exists()
    assert Path(files["aligned_depth_edges_on_color"]).exists()
    payload = json.loads(Path(record["metadata_file"]).read_text())
    assert payload["depth_alignment_enabled"] is True
    assert payload["raw_depth_shape"] == [2, 2]
    assert payload["saved_depth_shape"] == [4, 4]


REALSENSE_BAG = os.environ.get("REALSENSE_TEST_BAG", "")


@pytest.mark.skipif(not REALSENSE_BAG, reason="REALSENSE_TEST_BAG not set")
def test_pyrealsense2_reads_bag():
    """Smoke test: open a RealSense .bag file with pyrealsense2 playback."""
    import pyrealsense2 as rs

    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_device_from_file(REALSENSE_BAG, repeat_playback=False)
    cfg.enable_all_streams()
    profile = pipeline.start(cfg)

    playback = profile.get_device().as_playback()
    playback.set_real_time(False)

    # Read one frameset
    frames = pipeline.wait_for_frames(timeout_ms=5000)

    color = frames.get_color_frame()
    depth = frames.get_depth_frame()
    assert color is not None, "No color frame"
    assert depth is not None, "No depth frame"

    # Verify alignment works
    align = rs.align(rs.stream.color)
    aligned = align.process(frames)
    ad = aligned.get_depth_frame()
    assert ad is not None, "Alignment produced no depth frame"

    pipeline.stop()

    # Verify intrinsics are sensible
    color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
    ci = color_stream.get_intrinsics()
    assert ci.fx > 0
    assert ci.fy > 0
    assert ci.width > 0
    assert ci.height > 0

    depth_stream = profile.get_stream(rs.stream.depth).as_video_stream_profile()
    di = depth_stream.get_intrinsics()
    assert di.fx > 0

    # Verify extrinsics
    ext = depth_stream.get_extrinsics_to(color_stream)
    assert len(ext.translation) == 3
    assert len(ext.rotation) == 9
