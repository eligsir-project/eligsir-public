"""CPU contract for sharing report trajectory/depth renders."""
from __future__ import annotations

import tempfile

import pytest
import torch

from eligsir.utils.comparison_report import render_camera_path


class _Cam:
    def __init__(self, name):
        self.image_name = name
        self.camera_center = torch.zeros(3)


@pytest.fixture(autouse=True)
def _drain_video_frames(monkeypatch):
    def fake_write_mp4(_path, frames, **_kwargs):
        return None if list(frames) else "no frames written"

    monkeypatch.setattr("utils.comparison_report._write_mp4", fake_write_mp4)


def test_shared_depth_camera_renders_once_and_retains_depth_output():
    cams = [_Cam("a"), _Cam("b"), _Cam("c")]
    depth_ids = {id(cams[0]), id(cams[2])}
    calls = []

    def render_fn(cam, *_args, render_depth=False, **_kwargs):
        calls.append((cam.image_name, render_depth))
        output = {"render": torch.zeros(3, 8, 8)}
        if render_depth:
            output["rendered_depth"] = torch.zeros(1, 8, 8)
        return output

    cache = {}
    with tempfile.TemporaryDirectory() as output_dir:
        written = render_camera_path(
            cams, None, render_fn, None, None,
            out_path=f"{output_dir}/trajectory.mp4",
            depth_cam_ids=depth_ids,
            render_cache_out=cache,
        )

    assert written is not None
    assert calls == [("a", True), ("b", False), ("c", True)]
    assert set(cache) == depth_ids
    assert all(cache[cam_id]["rendered_depth"] is not None for cam_id in depth_ids)


def test_without_shared_depth_identity_uses_existing_plain_render_path():
    cam = _Cam("plain")
    calls = []

    def render_fn(_cam, *_args, render_depth=False, **_kwargs):
        calls.append(render_depth)
        return {"render": torch.zeros(3, 8, 8)}

    with tempfile.TemporaryDirectory() as output_dir:
        render_camera_path(
            [cam], None, render_fn, None, None,
            out_path=f"{output_dir}/trajectory.mp4",
        )

    assert calls == [False]


def test_missing_depth_output_keeps_the_trajectory_plain_render_path():
    cam = _Cam("no-depth")
    calls = []

    def render_fn(_cam, *_args, render_depth=False, **_kwargs):
        calls.append(render_depth)
        return {"render": torch.zeros(3, 8, 8)}

    with tempfile.TemporaryDirectory() as output_dir:
        render_camera_path(
            [cam], None, render_fn, None, None,
            out_path=f"{output_dir}/trajectory.mp4",
            depth_cam_ids={id(cam)},
            render_cache_out={},
        )

    assert calls == [True, False]
