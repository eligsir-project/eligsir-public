"""Real, CPU-only ffmpeg-encode regression test for issue #397's resolution
policy in ``utils.comparison_report._write_mp4``.

Deliberately NOT part of the CPU smoke image allowlist (this file imports
torch transitively via ``utils.comparison_report``) -- it is a real-encode
check to be run in the dev/ablator container alongside the rest of the
torch-carrying suite, not a pure-function unit test. See
``tests/test_trajectory_video.py`` for the torch-free, CPU-smoke-eligible
pure-function coverage of the same resolution math
(``scale_dims_for_video``); this file instead locks in the actual
``_write_mp4``/ffmpeg wiring, which caught a real width/height-swap bug
during development (``scale_dims_for_video`` returns ``(out_w, out_h)``,
and the first-cut wiring assigned it to ``h2, w2 = ...`` -- silently
writing every upscaled trajectory video in portrait orientation). Requires
ffmpeg + ffprobe on PATH; skips cleanly if either is missing.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile

import numpy as np
import pytest
import torch

from eligsir.utils.comparison_report import _write_mp4, render_camera_path

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="requires ffmpeg + ffprobe on PATH",
)


def _probe_dims(path: str):
    out = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height",
            "-of", "csv=p=0:s=x",
            path,
        ],
        capture_output=True, text=True, check=True,
    )
    w, h = out.stdout.strip().split("x")
    return int(w), int(h)


def _random_frames(h: int, w: int, n: int):
    return [np.random.randint(0, 255, (h, w, 3), dtype=np.uint8) for _ in range(n)]


def test_upscale_below_min_width_preserves_aspect_not_swapped():
    """The regression this file exists for: a 4:3 640x480 source with
    min_width=1920 must land as landscape 1920x1440 -- NOT a swapped
    1440x1920 portrait frame."""
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "out.mp4")
        err = _write_mp4(out, iter(_random_frames(480, 640, 6)), fps=10,
                          min_width=1920, max_width=3840)
        assert err is None
        w, h = _probe_dims(out)
        assert (w, h) == (1920, 1440)


def test_max_width_cap_downscales_and_preserves_aspect():
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "out.mp4")
        err = _write_mp4(out, iter(_random_frames(2160, 3840, 3)), fps=10,
                          min_width=1920, max_width=2000)
        assert err is None
        w, h = _probe_dims(out)
        assert w == 2000
        # aspect preserved within even-rounding tolerance
        assert abs((w / h) - (3840 / 2160)) < 0.01


def test_disabled_min_max_width_reproduces_pre_existing_even_crop_behavior():
    """min_width=max_width=None (the default for every pre-existing caller
    other than the trajectory/depth-timeseries videos) must reproduce the
    exact pre-#397 behavior: raw frame size, only cropped to the nearest
    even dimension."""
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "out.mp4")
        err = _write_mp4(out, iter(_random_frames(481, 641, 4)), fps=10)
        assert err is None
        w, h = _probe_dims(out)
        assert (w, h) == (640, 480)


def test_already_fhd_or_higher_source_unchanged_by_min_width():
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "out.mp4")
        err = _write_mp4(out, iter(_random_frames(1440, 2560, 3)), fps=10,
                          min_width=1920, max_width=3840)
        assert err is None
        w, h = _probe_dims(out)
        assert (w, h) == (2560, 1440)


def _fake_render_fn(cam, gaussians, pipe, background, **kwargs):
    """CPU-only stand-in for the real Gaussian rasterizer -- returns a
    small random CHW tensor. render_camera_path/_render_image only need a
    callable with this shape; no CUDA/ROCm/gsplat involved."""
    return torch.rand(3, 64, 64)


def test_render_camera_path_frame_repeats_produces_correct_encoded_count():
    """Issue #397: render_camera_path's frame_repeats param must render
    exactly len(cams) frames (the bounded render budget) but ENCODE
    sum(frame_repeats) frames -- the mechanism that reproduces the real
    capture duration without multiplying render cost. Exercises the real
    render_camera_path -> _write_mp4 -> ffmpeg path end to end, CPU-only."""
    cams = [
        __import__("types").SimpleNamespace(image_name=f"c{i}")
        for i in range(5)
    ]
    repeats = [3, 1, 5, 2, 4]  # sums to 15
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "out.mp4")
        written = render_camera_path(
            cams, gaussians=None, render_fn=_fake_render_fn, pipe=None, background=None,
            out_path=out, fps=10, frame_repeats=repeats,
        )
        assert written == out
        w, h = _probe_dims(out)
        assert (w, h) == (64, 64)
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-count_frames", "-show_entries", "stream=nb_read_frames",
             "-of", "csv=p=0", out],
            capture_output=True, text=True, check=True,
        )
        assert int(probe.stdout.strip()) == sum(repeats)


def test_render_camera_path_default_repeats_once_per_cam_unchanged():
    """frame_repeats=None (every pre-existing caller, e.g. render_report.py)
    must still encode exactly len(cams) frames -- no behavior change."""
    cams = [__import__("types").SimpleNamespace(image_name=f"c{i}") for i in range(4)]
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "out.mp4")
        render_camera_path(
            cams, gaussians=None, render_fn=_fake_render_fn, pipe=None, background=None,
            out_path=out, fps=10,
        )
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-count_frames", "-show_entries", "stream=nb_read_frames",
             "-of", "csv=p=0", out],
            capture_output=True, text=True, check=True,
        )
        assert int(probe.stdout.strip()) == len(cams)
