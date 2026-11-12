"""Regression tests for issue #617: ``utils/camera_utils.py``'s
``_decode_image_tensor`` streaming fast path checked only
``C_CONTIGUOUS`` before ``torch.from_numpy``, never ``WRITEABLE`` -- the
same defect shape as issue #593 (fixed for
``eligsir/mapping/resolution_cache.py`` in PR #616).

Root cause: an already-contiguous read-only ``uint8`` array (e.g. a ROS2
``sensor_msgs/Image`` decoded via ``cv_bridge.imgmsg_to_cv2``, which wraps
the message's raw ``.data`` buffer in place without copying, or a PIL
``np.asarray(...)``-backed frame) passes a bare ``C_CONTIGUOUS`` check
unchanged. ``torch.from_numpy`` on such an array immediately emits a
``UserWarning`` ("The given NumPy array is not writable") and the
resulting tensor aliases non-writable memory -- undefined behavior on any
later in-place write. PyTorch only warns once per process, so a clean log
does not prove this only happened once.

These tests exercise the real, exported entry points
(``loadCam`` / ``_decode_image_tensor``), not a hand-rolled reimplementation
of the guard, so a regression in the guard's placement is caught the same
way a real caller would hit it.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest
import torch

# Import scene.cameras (which pulls in the `scene` package) BEFORE
# utils.camera_utils: utils/camera_utils.py itself does
# `from scene.cameras import Camera`, and scene/__init__.py does
# `from eligsir.utils.camera_utils import cameraList_from_camInfos, camera_to_JSON`
# -- a genuine circular import between the two modules that only resolves
# cleanly when `scene` (the package) finishes initializing first. This
# import ordering is pre-existing repo behavior, not introduced by this
# test; other tests (e.g. test_camera_image_offload.py) import
# `scene.cameras` first for the same reason.
from scene.readers.common import CameraInfo
from eligsir.utils.camera_utils import _decode_image_tensor


class _Args:
    resolution = -1
    data_device = "cpu"


def _make_cam_info(width: int, height: int) -> CameraInfo:
    return CameraInfo(
        uid=0, R=np.eye(3), T=np.zeros(3),
        FovY=0.8, FovX=1.0,
        image=None, image_path="", image_name="frame_000000",
        width=width, height=height,
    )


def _rgb_array(writable: bool) -> np.ndarray:
    # 6x5 (WxH) RGB uint8, allocated fresh so it starts C-contiguous and
    # OWNDATA. Deliberately non-square with height != 4: _decode_image_
    # tensor's post-permute CHW shape[1] is H, and this repo's alpha-mask
    # detector USED to check `resized_image_rgb.shape[1] == 4` -- a 4x4
    # test image would coincidentally satisfy that and make loaded_mask
    # non-None for reasons unrelated to writability. (That defect is fixed
    # under issue #643 -- the check now reads shape[0], the channel dim;
    # see tests/test_issue643_alpha_channel_dim.py. The shape is kept
    # non-square anyway so these tests stay independent of that fix.)
    arr = np.arange(5 * 6 * 3, dtype=np.uint8).reshape(5, 6, 3)
    assert arr.flags["C_CONTIGUOUS"]
    if not writable:
        arr.setflags(write=False)
    return arr


def test_readonly_contiguous_array_does_not_warn():
    """The confirmed defect: a C-contiguous but non-writable array must not
    reach torch.from_numpy unchanged. Pre-fix this raises UserWarning (here
    escalated to an exception via simplefilter), post-fix it does not."""
    rgb = _rgb_array(writable=False)
    assert rgb.flags["C_CONTIGUOUS"] and not rgb.flags["WRITEABLE"]

    args = _Args()
    cam_info = _make_cam_info(6, 5)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        gt_image, loaded_mask = _decode_image_tensor(
            args, cam_info, orig_w=6, orig_h=5, resolution=(6, 5), rgb_uint8=rgb,
        )

    assert gt_image.shape == (3, 5, 6)
    assert loaded_mask is None


def test_readonly_contiguous_array_result_is_writable_and_correct():
    """The fix must copy data faithfully, not just silence the warning."""
    rgb = _rgb_array(writable=False)
    rgb_writable_reference = np.array(rgb, copy=True)  # same values, safe to mutate

    args = _Args()
    cam_info = _make_cam_info(6, 5)

    gt_image, _ = _decode_image_tensor(
        args, cam_info, orig_w=6, orig_h=5, resolution=(6, 5), rgb_uint8=rgb,
    )

    expected = (
        torch.from_numpy(rgb_writable_reference).permute(2, 0, 1).to(torch.float32) / 255.0
    )[:3, ...]
    assert torch.allclose(gt_image, expected)

    # The copy the guard produces must itself be writable -- otherwise a
    # future in-place op downstream would hit the exact same trap.
    result_np = gt_image.cpu().numpy()
    result_np[0, 0, 0] = 0.0  # must not raise


def test_writable_contiguous_array_still_takes_zero_copy_fast_path_and_does_not_warn():
    """Non-regression: the common (writable, contiguous) case must not be
    penalized by the new check -- and must still not warn."""
    rgb = _rgb_array(writable=True)
    assert rgb.flags["C_CONTIGUOUS"] and rgb.flags["WRITEABLE"]

    args = _Args()
    cam_info = _make_cam_info(6, 5)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        gt_image, _ = _decode_image_tensor(
            args, cam_info, orig_w=6, orig_h=5, resolution=(6, 5), rgb_uint8=rgb,
        )

    assert gt_image.shape == (3, 5, 6)


def test_non_contiguous_writable_array_still_handled():
    """Non-contiguous but writable input (the pre-existing guard's original
    target) must still be handled correctly -- this is a slice view, not the
    read-only case, so it exercises the other half of the flags check."""
    base = np.arange(5 * 12 * 3, dtype=np.uint8).reshape(5, 12, 3)
    rgb = base[:, ::2, :]  # non-contiguous view (shape (5, 6, 3)), still writable
    assert not rgb.flags["C_CONTIGUOUS"]
    assert rgb.flags["WRITEABLE"]

    args = _Args()
    cam_info = _make_cam_info(6, 5)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        gt_image, _ = _decode_image_tensor(
            args, cam_info, orig_w=6, orig_h=5, resolution=(6, 5), rgb_uint8=rgb,
        )

    assert gt_image.shape == (3, 5, 6)
