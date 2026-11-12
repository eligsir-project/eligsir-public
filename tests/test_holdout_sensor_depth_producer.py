"""Regression tests for the holdout sensor-depth producer gap (issue #358).

`utils.comparison_report._get_sensor_depth` is the single producer behind
`holdout_gt/<id>_depth.png`, which is the FIXED sensor depth Level A's
per-microcell quality metric backprojects through. Its frame branch used to
require `_depth_bytes`, but a ros2://live frame carries its depth ONLY as
`_depth_np` (`depth_path=None`, no `_depth_bytes`) -- so every live/bag-replay
run silently produced holdout RGB with no holdout depth, and no corridor or
multi-room scene on disk could be measured by Level A at all.

Silently: the caller's only diagnostic is an `except` log, and this path
returned None rather than raising. These tests pin both the fixed behaviour
and the still-correct negative case.
"""
import numpy as np
import pytest
import torch

from eligsir.utils.comparison_report import _get_sensor_depth


class _Frame:
    """Minimal StreamingRGBDFrame stand-in for the fields this path reads."""

    def __init__(self, depth_np=None, depth_bytes=None):
        self._depth_np = depth_np
        self._depth_bytes = depth_bytes
        self._sens_header = None
        self.depth_path = None
        self._cache_pos = None


class _Cam:
    def __init__(self, frame, depth_scale=1000.0):
        self._streaming_frame = frame
        self._streaming_depth_path = None      # live frames have no file
        self._streaming_depth_scale = depth_scale
        self._sensor_depth_cache = None
        self.image_height, self.image_width = 4, 6


def _depth_mm(h=4, w=6):
    # 1.0 m .. 2.4 m as uint16 millimetres, the live driver's own encoding
    return (np.arange(h * w, dtype=np.uint16).reshape(h, w) * 50 + 1000)


def test_live_frame_depth_np_is_recovered():
    """The ros2://live case that produced zero holdout depth PNGs."""
    raw = _depth_mm()
    cam = _Cam(_Frame(depth_np=raw))
    out = _get_sensor_depth(cam, cam.image_height, cam.image_width)
    assert out is not None, "live _depth_np frame must yield sensor depth"
    assert out.shape == (1, cam.image_height, cam.image_width)
    assert torch.allclose(out[0], torch.from_numpy(raw.astype(np.float32) / 1000.0))


def test_depth_bytes_path_still_works():
    """The pre-existing .sens/bytes branch must be untouched."""
    import io as _io
    from PIL import Image

    raw = _depth_mm()
    buf = _io.BytesIO()
    Image.fromarray(raw).save(buf, format="PNG")
    cam = _Cam(_Frame(depth_bytes=buf.getvalue()))
    out = _get_sensor_depth(cam, cam.image_height, cam.image_width)
    assert out is not None
    assert torch.allclose(out[0], torch.from_numpy(raw.astype(np.float32) / 1000.0))


def test_frame_with_no_depth_at_all_still_returns_none():
    """No depth anywhere must stay a clean None, not a fabricated array."""
    cam = _Cam(_Frame())
    assert _get_sensor_depth(cam, cam.image_height, cam.image_width) is None
    # and it must latch, so the miss is not re-attempted per view
    assert cam._sensor_depth_cache is False


def test_depth_scale_is_honoured():
    raw = _depth_mm()
    cam = _Cam(_Frame(depth_np=raw), depth_scale=5000.0)
    out = _get_sensor_depth(cam, cam.image_height, cam.image_width)
    assert torch.allclose(out[0], torch.from_numpy(raw.astype(np.float32) / 5000.0))


@pytest.mark.parametrize("th,tw", [(8, 12), (2, 3)])
def test_resize_to_target_shape(th, tw):
    cam = _Cam(_Frame(depth_np=_depth_mm()))
    out = _get_sensor_depth(cam, th, tw)
    assert out.shape == (1, th, tw)
