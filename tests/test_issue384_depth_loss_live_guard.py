"""Regression tests for the LOSS-path copy of the live sensor-depth guard (issue #384).

PR #379 fixed `utils.comparison_report._get_sensor_depth` (reporting only) to
accept a frame carrying its depth as `_depth_np`, which is the only form a
`ros2://live` frame has (`eligsir/io/sources/datasets/ros2_live.py:556`
builds `StreamingRGBDFrame(_depth_np=depth)` with `depth_path=None` and no
`_depth_bytes`). It deliberately left the identically-named
`eligsir.mapping.integration.depth_helpers._get_sensor_depth` -- the copy
that feeds the DEPTH LOSS (`eligsir/optimization/gaussian/objective.py`,
`eligsir/mapping/integration/frame_preparation.py`, several
`train_streaming.py` sites) -- on the old bytes-only guard, pending the
reachability audit issue #384 tracks.

That audit found the loss path IS reachable with an unpopulated
`cam._sensor_depth_cache` on live sources (see the comment block in
`depth_helpers._get_sensor_depth` for the two traced call sites), so the guard
is now widened to match. These tests pin BOTH branches: the live `_depth_np`
frame now yields real depth, and the offline on-disk frame keeps loading from
disk exactly as before.
"""
import io as _io
import os

import numpy as np
import pytest

# This module needs the real torch (it builds tensors and asserts on their
# values), and `depth_helpers` imports torch transitively. The CPU smoke
# image (`Dockerfile.ci-cpu`) has no torch, where a bare `import torch` at
# module scope is a *collection error* that aborts the whole run with exit
# code 2 -- not a skip. Guard it here rather than by adding this file to
# tests/conftest.py's `_NEEDS_TORCH` list: a self-contained guard cannot go
# stale the way a central list does, and conftest.py already documents this
# per-file form as an accepted alternative.
torch = pytest.importorskip("torch")

from PIL import Image  # noqa: E402

from eligsir.mapping.integration.depth_helpers import _get_sensor_depth  # noqa: E402


class _Frame:
    """Minimal StreamingRGBDFrame stand-in for the fields this path reads."""

    def __init__(self, depth_np=None, depth_bytes=None, depth_path=None, depth_scale=1000.0):
        self._depth_np = depth_np
        self._depth_bytes = depth_bytes
        self._sens_header = None
        self.depth_path = depth_path
        self.depth_scale = depth_scale
        self._cache_pos = None


class _Cam:
    """Streaming camera stand-in. `_streaming_depth_path` must EXIST (the
    function's first guard is `hasattr`), and is None for a live frame."""

    def __init__(self, frame, depth_path=None, depth_scale=1000.0):
        self._streaming_frame = frame
        self._streaming_depth_path = depth_path
        self._streaming_depth_scale = depth_scale
        self._sensor_depth_cache = None
        self.image_height, self.image_width = 4, 6


def _depth_mm(h=4, w=6):
    # 1.0 m .. 2.4 m as uint16 millimetres -- the live driver's own encoding
    return (np.arange(h * w, dtype=np.uint16).reshape(h, w) * 50 + 1000)


def test_live_depth_np_frame_reaches_the_loss_path():
    """The ros2://live shape: depth ONLY as `_depth_np`, no path, no bytes.

    Pre-#384 this returned None and latched `_sensor_depth_cache = False`,
    i.e. permanently no depth supervision for that camera.
    """
    raw = _depth_mm()
    cam = _Cam(_Frame(depth_np=raw))
    out = _get_sensor_depth(cam, cam.image_height, cam.image_width)
    assert out is not None, "live _depth_np frame must yield sensor depth for the loss"
    assert out.shape == (1, cam.image_height, cam.image_width)
    assert torch.allclose(out[0], torch.from_numpy(raw.astype(np.float32) / 1000.0))
    assert cam._sensor_depth_cache is not False


def test_depth_bytes_branch_unchanged():
    """The pre-existing ScanNet-.sens/bytes branch must be untouched."""
    raw = _depth_mm()
    buf = _io.BytesIO()
    Image.fromarray(raw).save(buf, format="PNG")
    cam = _Cam(_Frame(depth_bytes=buf.getvalue()))
    out = _get_sensor_depth(cam, cam.image_height, cam.image_width)
    assert out is not None
    assert torch.allclose(out[0], torch.from_numpy(raw.astype(np.float32) / 1000.0))


def test_offline_on_disk_branch_unchanged(tmp_path):
    """An offline (TUM/ScanNet-path) frame still loads from `depth_path`.

    This is the behaviour-neutrality half of the fix: the widened guard must
    not divert a disk-backed frame into the in-memory branch when it carries
    no in-memory payload.
    """
    raw = _depth_mm()
    p = os.path.join(str(tmp_path), "depth.png")
    Image.fromarray(raw).save(p)
    cam = _Cam(_Frame(depth_path=p), depth_path=p)
    out = _get_sensor_depth(cam, cam.image_height, cam.image_width)
    assert out is not None
    assert torch.allclose(out[0], torch.from_numpy(raw.astype(np.float32) / 1000.0))


def test_no_depth_anywhere_returns_none_without_latching():
    """A frame carrying no depth payload at all returns None -- but must NOT
    latch the permanent `False` sentinel (issue #481(a)).

    This test previously asserted the latch. It was changed deliberately: the
    latch cannot distinguish a genuinely depth-less source (RealityScan Mobile,
    correctly None forever) from a frame whose in-memory depth is simply not
    populated yet, and re-checking costs two getattrs and a None test -- no I/O.
    The RETURN value for a depth-less source is unchanged (None, every call).
    """
    cam = _Cam(_Frame())
    assert _get_sensor_depth(cam, cam.image_height, cam.image_width) is None
    assert cam._sensor_depth_cache is not False, "cheap-to-recheck absence must not latch"
    # ... and stays None on every subsequent call, for a truly depth-less source.
    assert _get_sensor_depth(cam, cam.image_height, cam.image_width) is None


def test_depth_appearing_later_is_picked_up_after_an_earlier_miss():
    """The defect #481(a) exists to fix: a camera asked for depth BEFORE its
    frame's `_depth_np` was populated (live decode, or a `promote_from_cache`
    rebuild ahead of `_predecode_frames_parallel`) used to be denied depth
    supervision permanently -- silently, no log, no raise."""
    frame = _Frame()
    cam = _Cam(frame)
    assert _get_sensor_depth(cam, cam.image_height, cam.image_width) is None

    raw = _depth_mm()
    frame._depth_np = raw  # decoded later by some other path

    out = _get_sensor_depth(cam, cam.image_height, cam.image_width)
    assert out is not None, "a later-populated _depth_np must be picked up"
    assert torch.allclose(out[0], torch.from_numpy(raw.astype(np.float32) / 1000.0))


def test_missing_depth_path_still_latches():
    """A depth_path that does not exist keeps the sentinel: re-checking costs a
    stat on every access, which is exactly what the sentinel exists to avoid."""
    cam = _Cam(_Frame(depth_path="/nonexistent/depth.png"), depth_path="/nonexistent/depth.png")
    assert _get_sensor_depth(cam, cam.image_height, cam.image_width) is None
    assert cam._sensor_depth_cache is False


def test_decode_failure_still_latches(tmp_path):
    """A decode that raises keeps the sentinel: retrying costs a full decode."""
    p = os.path.join(str(tmp_path), "not_an_image.png")
    with open(p, "wb") as fh:
        fh.write(b"definitely not a PNG")
    cam = _Cam(_Frame(depth_path=p), depth_path=p)
    assert _get_sensor_depth(cam, cam.image_height, cam.image_width) is None
    assert cam._sensor_depth_cache is False


def test_depth_scale_is_read_from_the_camera_not_the_frame():
    """This copy scales with `cam._streaming_depth_scale` (unlike the
    comparison_report copy's `getattr` fallback) -- pin that it stays so."""
    raw = _depth_mm()
    cam = _Cam(_Frame(depth_np=raw), depth_scale=5000.0)
    out = _get_sensor_depth(cam, cam.image_height, cam.image_width)
    assert torch.allclose(out[0], torch.from_numpy(raw.astype(np.float32) / 5000.0))


@pytest.mark.parametrize("th,tw", [(8, 12), (2, 3)])
def test_live_depth_is_resized_to_target(th, tw):
    cam = _Cam(_Frame(depth_np=_depth_mm()))
    out = _get_sensor_depth(cam, th, tw)
    assert out.shape == (1, th, tw)


def test_bind_resolution_tier_fails_closed_on_an_empty_resolution_cache():
    """Reachability link for the promoted-camera case.

    `record_frame_observation` only builds a `resolution_cache` for is_train
    frames (train_streaming.py:988), so a frame first seen as a REJECTED
    candidate has an empty one. `promote_from_cache` later rebuilds a Camera
    from that same frame (scene/streaming_scene.py:2634) with
    `_sensor_depth_cache = None` -- and `bind_resolution_tier` cannot fill it,
    which is what routes such a camera into the lazy path this module's guard
    controls.

    Skipped outside the container: `train_streaming` needs the full GPU stack
    to import.
    """
    try:
        import train_streaming
    except Exception as exc:  # noqa: BLE001 - torchvision/GPU stack absent on host
        pytest.skip(f"train_streaming needs the container GPU stack: {exc!r}")

    class _Record:
        frame_id = 7
        resolution_cache: dict = {}

    class _MinimalCam:
        _sensor_depth_cache = None

    cam = _MinimalCam()
    assert train_streaming.bind_resolution_tier(cam, _Record(), "low") is False
    assert cam._sensor_depth_cache is None, "a failed bind must not touch the cache"
