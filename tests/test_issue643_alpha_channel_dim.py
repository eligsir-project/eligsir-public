"""Regression tests for issue #643 (parent #638): ``utils/camera_utils.py``'s
``_decode_image_tensor`` detected the alpha channel with
``resized_image_rgb.shape[1] == 4``.

The tensor at that point is 3-D **CHW** on both of the function's paths (the
uint8 fast path does an explicit ``.permute(2, 0, 1)``; ``PILtoTorch`` does the
same). So ``shape[0]`` is the channel dimension and ``shape[1]`` is the image
HEIGHT -- the check was wrong in both directions:

* a genuine RGBA input (C == 4) never produced a ``loaded_mask``, and
* an image exactly 4 pixels tall produced a bogus mask, because ``[3:4, ...]``
  on a 3-channel tensor is an EMPTY ``(0, H, W)`` tensor, which then fails in
  ``Camera.__init__``'s ``original_image *= gt_alpha_mask``.

These tests pin the RANK and the channel dimension explicitly, so the
verdict survives any future change to the tensor layout (e.g. a batched
``(B, C, H, W)`` representation would have to update these tests deliberately
rather than silently re-alias the dimensions).
"""
from __future__ import annotations

import numpy as np
import pytest

# This behavior test genuinely needs torch. Keep the file visible on the
# torch-free CPU-CI surface as an explicit skip instead of allowing the
# repository's dynamic collector to exclude it silently (#518/#536).
torch = pytest.importorskip("torch", reason="issue #643 alpha behavior requires torch")
from PIL import Image

# Import `scene.*` before `utils.camera_utils` -- see the same note in
# tests/test_issue617_camera_utils_writability.py (genuine circular import
# between `scene/__init__.py` and `utils/camera_utils.py`).
from scene.readers.common import CameraInfo
from eligsir.utils.camera_utils import _decode_image_tensor
from eligsir.utils.general_utils import PILtoTorch


class _Args:
    resolution = -1
    data_device = "cpu"


def _cam_info(width: int, height: int, image=None) -> CameraInfo:
    return CameraInfo(
        uid=0, R=np.eye(3), T=np.zeros(3),
        FovY=0.8, FovX=1.0,
        image=image, image_path="", image_name="frame_000000",
        width=width, height=height,
    )


# Deliberately non-square and H != 4 so neither dimension can coincidentally
# satisfy the other's check.
W, H = 6, 5


def _rgba_array() -> np.ndarray:
    arr = np.zeros((H, W, 4), dtype=np.uint8)
    arr[..., 0] = 255           # R
    arr[..., 3] = 128           # A -- a distinctive, non-degenerate alpha
    return arr


def test_piltotorch_returns_3d_chw_for_rgba():
    """The load-bearing empirical fact: rank is 3 and channels lead."""
    pil = Image.fromarray(_rgba_array(), "RGBA")
    t = PILtoTorch(pil, (W, H))
    assert t.ndim == 3, "PILtoTorch must return CHW, not a batched (B, C, H, W)"
    assert t.shape == (4, H, W)
    assert t.shape[0] == 4      # channel dim
    assert t.shape[1] == H      # NOT the channel dim


def test_rgba_pil_populates_loaded_mask():
    """Slow (PIL) path: an RGBA source must yield a (1, H, W) alpha mask."""
    pil = Image.fromarray(_rgba_array(), "RGBA")
    gt_image, loaded_mask = _decode_image_tensor(
        _Args(), _cam_info(W, H, image=pil), orig_w=W, orig_h=H, resolution=(W, H),
    )
    assert gt_image.shape == (3, H, W)
    assert loaded_mask is not None, "issue #643: RGBA input must produce loaded_mask"
    assert loaded_mask.shape == (1, H, W)
    assert torch.allclose(loaded_mask, torch.full((1, H, W), 128.0 / 255.0))


def test_rgba_uint8_fast_path_populates_loaded_mask():
    """Fast (uint8, no-rescale) path must agree with the PIL path."""
    rgba = _rgba_array()
    gt_image, loaded_mask = _decode_image_tensor(
        _Args(), _cam_info(W, H), orig_w=W, orig_h=H, resolution=(W, H), rgb_uint8=rgba,
    )
    assert gt_image.shape == (3, H, W)
    assert loaded_mask is not None
    assert loaded_mask.shape == (1, H, W)
    assert torch.allclose(loaded_mask, torch.full((1, H, W), 128.0 / 255.0))


def test_rgba_uint8_slow_path_preserves_alpha():
    """The PIL-rebuild fallback (cam_info.image is None + an actual rescale)
    must not drop alpha via ``rgb_uint8[..., :3]`` -- otherwise the same array
    yields a mask on the fast path and none here."""
    rgba = _rgba_array()
    # resolution != (orig_w, orig_h) forces the fallback branch. Target (5, 3)
    # -> CHW (4, 3, 5): no dimension other than the channel dim equals 4, so a
    # pre-fix `shape[1] == 4` cannot coincidentally satisfy this assertion.
    # (The obvious (W-2, H-1) = (4, 4) would give (4, 4, 4) and pass under the
    # bug via an empty (0, 4, 4) mask -- exactly the coincidence this module's
    # fixture sizes exist to avoid.)
    gt_image, loaded_mask = _decode_image_tensor(
        _Args(), _cam_info(W, H), orig_w=W, orig_h=H, resolution=(5, 3),
        rgb_uint8=rgba,
    )
    assert gt_image.shape == (3, 3, 5)
    assert loaded_mask is not None, "issue #643: fallback path must preserve alpha"
    assert loaded_mask.shape == (1, 3, 5)


@pytest.mark.parametrize("height", [3, 4, 5])
def test_rgb_never_produces_a_mask_regardless_of_height(height):
    """The other half of the bug: a 3-channel image exactly 4 pixels TALL used
    to satisfy ``shape[1] == 4`` and return an empty ``(0, H, W)`` mask."""
    rgb = np.zeros((height, W, 3), dtype=np.uint8)
    gt_image, loaded_mask = _decode_image_tensor(
        _Args(), _cam_info(W, height), orig_w=W, orig_h=height,
        resolution=(W, height), rgb_uint8=rgb,
    )
    assert gt_image.shape == (3, height, W)
    assert loaded_mask is None


def test_rgb_only_samples_are_bit_identical_with_and_without_alpha_channel():
    """`Existing RGB-only datasets retain identical RGB samples` (#643 AC).
    The RGB planes of an RGBA input must match the same image loaded as RGB."""
    rgba = _rgba_array()
    rgb = np.ascontiguousarray(rgba[..., :3])

    gt_rgba, _ = _decode_image_tensor(
        _Args(), _cam_info(W, H), orig_w=W, orig_h=H, resolution=(W, H), rgb_uint8=rgba,
    )
    gt_rgb, mask_rgb = _decode_image_tensor(
        _Args(), _cam_info(W, H), orig_w=W, orig_h=H, resolution=(W, H), rgb_uint8=rgb,
    )
    assert mask_rgb is None
    assert torch.equal(gt_rgba, gt_rgb)
