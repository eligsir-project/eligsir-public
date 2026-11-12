"""Regression tests for issue #593: non-writable NumPy -> PyTorch tensor
conversion in ``eligsir/mapping/resolution_cache.py``.

Root cause: ``np.ascontiguousarray(arr)`` guarantees C-contiguity but NOT
writability. When ``arr`` is already contiguous and backed by read-only
storage (the confirmed real-world case: ``rgb_np = np.asarray(pil_image)``
in ``train_streaming.py``'s ``_build_frame_resolution_cache`` -- PIL images
back their numpy view with an immutable ``.tobytes()`` buffer, so the array
comes back C-contiguous *and* non-writable), ``ascontiguousarray`` is a
no-op passthrough and the resulting ``torch.from_numpy`` tensor aliases
that non-writable memory. Any later in-place write to the tensor is
formally undefined behavior, and PyTorch only warns about it once per
process, so a clean-looking log does not prove it only happened once.

The "high" tier (``low_res_divisor``'s divisor=1 passthrough, in both
``resize_rgbd_frame`` and ``_resize_rgb_only``) returns the caller's array
completely unchanged, so this matrix is exercised end-to-end through
``build_resolution_cache`` with ``low_res_divisor=1`` -- that pins BOTH
tiers to the exact zero-transform path a real read-only ``rgb_np`` would
hit in production, rather than only unit-testing the (unexported) internal
``_to_tensor`` closure.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest
import torch

from eligsir.mapping.resolution_cache import build_resolution_cache
from eligsir.utils.rgbd_frames import CameraIntrinsics


def _make_intrinsics():
    return CameraIntrinsics(
        width=8, height=8, fx=50.0, fy=50.0, cx=4.0, cy=4.0,
        depth_scale=1000.0, camera_frame="camera_color_optical_frame",
    )


def _rgb(contiguous: bool, writable: bool) -> np.ndarray:
    base = np.arange(8 * 16 * 3, dtype=np.uint8).reshape(8, 16, 3)
    # Slicing every other column yields a genuinely non-contiguous VIEW;
    # setting that view's own WRITEABLE flag (not a copy's) keeps both
    # properties independently controllable.
    arr = base if contiguous else base[:, ::2, :]
    if not writable:
        arr.setflags(write=False)
    assert arr.flags["WRITEABLE"] is writable
    assert arr.flags["C_CONTIGUOUS"] is contiguous
    return arr


def _depth(contiguous: bool, writable: bool) -> np.ndarray:
    base = np.arange(8 * 16, dtype=np.float32).reshape(8, 16)
    arr = base if contiguous else base[:, ::2]
    if not writable:
        arr.setflags(write=False)
    assert arr.flags["WRITEABLE"] is writable
    assert arr.flags["C_CONTIGUOUS"] is contiguous
    return arr


def _assert_not_aliasing(t: torch.Tensor, arr: np.ndarray, *, writable: bool) -> None:
    """The only mutation-sensitive assertion available for this bug.

    Two otherwise-obvious checks are VACUOUS here and must not be relied on
    (both confirmed by reverting the fix and re-running this file):

    * ``t.numpy().flags["WRITEABLE"]`` reads ``True`` even when ``t`` was
      built by ``torch.from_numpy`` on a read-only array -- torch does not
      carry numpy's WRITEABLE flag across the conversion.
    * PyTorch's "given NumPy array is not writable" ``UserWarning`` fires at
      most ONCE PER PROCESS. Any test that detects the bug only via
      ``warnings.simplefilter("error")`` therefore passes vacuously as soon as
      some earlier test in the same process has already consumed the warning
      -- making detection depend on test ORDER rather than on behaviour.

    A pointer comparison has neither problem: pre-fix, ``ascontiguousarray``
    is a no-op passthrough for a contiguous read-only array and the resulting
    CPU tensor shares its buffer, so ``data_ptr()`` matches exactly.
    """
    if writable:
        return
    src_ptr = arr.__array_interface__["data"][0]
    assert t.data_ptr() != src_ptr, (
        "issue #593: the cached tensor ALIASES the caller's read-only buffer "
        "(data_ptr matches) -- torch.from_numpy was handed a non-writable "
        "array, which is undefined behavior on any later in-place write."
    )


# ---------------------------------------------------------------------------
# Full writable/non-writable x contiguous/non-contiguous matrix, for both
# rgb and depth, through the real "high" tier (divisor=1) passthrough path.
# ---------------------------------------------------------------------------

_MATRIX = [
    (True, True),
    (True, False),
    (False, True),
    (False, False),
]


@pytest.mark.parametrize("contiguous,writable", _MATRIX)
def test_high_tier_rgb_tensor_is_always_writable(contiguous, writable):
    rgb = _rgb(contiguous, writable)
    intr = CameraIntrinsics(
        width=rgb.shape[1], height=rgb.shape[0], fx=50.0, fy=50.0,
        cx=rgb.shape[1] / 2.0, cy=rgb.shape[0] / 2.0,
        depth_scale=1000.0, camera_frame="camera_color_optical_frame",
    )
    cache = build_resolution_cache(rgb, None, intr, low_res_divisor=1, device="cpu")
    t = cache["high"].rgb
    assert isinstance(t, torch.Tensor)
    # `t.numpy().flags["WRITEABLE"]` is NOT a real check: torch does not
    # propagate numpy's WRITEABLE flag through `from_numpy`, so it reads True
    # even for a tensor aliasing read-only memory (confirmed by mutation --
    # see _assert_not_aliasing's docstring). The load-bearing assertion is the
    # pointer comparison below.
    assert t.numpy().flags["WRITEABLE"] is True
    _assert_not_aliasing(t, rgb, writable=writable)
    t.add_(1)  # would be undefined behavior pre-fix on the non-writable cases
    if not writable:
        # The original caller-owned array must be untouched -- proves the
        # tensor was NOT aliasing the read-only buffer.
        assert rgb.flags["WRITEABLE"] is False
        np.testing.assert_array_equal(rgb, _rgb(contiguous, writable))


@pytest.mark.parametrize("contiguous,writable", _MATRIX)
def test_high_tier_depth_tensor_is_always_writable(contiguous, writable):
    depth = _depth(contiguous, writable)
    rgb = np.zeros((depth.shape[0], depth.shape[1], 3), dtype=np.uint8)
    intr = CameraIntrinsics(
        width=depth.shape[1], height=depth.shape[0], fx=50.0, fy=50.0,
        cx=depth.shape[1] / 2.0, cy=depth.shape[0] / 2.0,
        depth_scale=1000.0, camera_frame="camera_color_optical_frame",
    )
    cache = build_resolution_cache(rgb, depth, intr, low_res_divisor=1, device="cpu")
    t = cache["high"].depth
    assert isinstance(t, torch.Tensor)
    assert t.numpy().flags["WRITEABLE"] is True
    _assert_not_aliasing(t, depth, writable=writable)
    t.add_(1.0)
    if not writable:
        assert depth.flags["WRITEABLE"] is False
        np.testing.assert_array_equal(depth, _depth(contiguous, writable))


def test_readonly_contiguous_rgb_values_are_preserved():
    """The fix must copy the DATA faithfully, not just make it writable."""
    rgb = _rgb(contiguous=True, writable=False)
    intr = CameraIntrinsics(
        width=rgb.shape[1], height=rgb.shape[0], fx=50.0, fy=50.0,
        cx=rgb.shape[1] / 2.0, cy=rgb.shape[0] / 2.0,
        depth_scale=1000.0, camera_frame="camera_color_optical_frame",
    )
    cache = build_resolution_cache(rgb, None, intr, low_res_divisor=1, device="cpu")
    np.testing.assert_array_equal(cache["high"].rgb.numpy(), rgb)


# ---------------------------------------------------------------------------
# Regression test: run the real production trigger for this bug (a
# np.asarray(PIL.Image)-backed array -- confirmed read-only + contiguous,
# see the module docstring) through build_resolution_cache with warnings
# escalated to errors. Pre-fix this raises UserWarning as an error; post-fix
# it must not warn at all.
# ---------------------------------------------------------------------------


def test_pil_backed_readonly_array_does_not_warn():
    from PIL import Image

    pil_image = Image.new("RGB", (8, 8), color=(1, 2, 3))
    rgb_np = np.asarray(pil_image)
    # Confirm this really is the exact hazardous shape this bug needs:
    # contiguous (ascontiguousarray would be a no-op) and non-writable.
    assert rgb_np.flags["C_CONTIGUOUS"] is True
    assert rgb_np.flags["WRITEABLE"] is False

    intr = CameraIntrinsics(
        width=8, height=8, fx=50.0, fy=50.0, cx=4.0, cy=4.0,
        depth_scale=1000.0, camera_frame="camera_color_optical_frame",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cache = build_resolution_cache(rgb_np, None, intr, low_res_divisor=4, device="cpu")
    assert cache["high"].rgb.numpy().flags["WRITEABLE"] is True
    assert cache["low"].rgb.numpy().flags["WRITEABLE"] is True
    # The warnings-as-errors block above is defence in depth ONLY: PyTorch
    # emits that UserWarning at most once per process, so by the time this
    # test runs inside a full-file/full-suite session an earlier test has
    # usually already consumed it and the block catches nothing. The pointer
    # check is what actually detects the regression, in any test order.
    _assert_not_aliasing(cache["high"].rgb, rgb_np, writable=False)


def test_ordinary_writable_contiguous_inputs_still_do_not_warn():
    """Non-regression: the common (already-safe) case stays warning-free."""
    rgb = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    depth = np.random.randint(100, 5000, (64, 64), dtype=np.uint16).astype(np.float32)
    intr = _make_intrinsics_64()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        build_resolution_cache(rgb, depth, intr, low_res_divisor=4, device="cpu")


def _make_intrinsics_64():
    return CameraIntrinsics(
        width=64, height=64, fx=500.0, fy=500.0, cx=32.0, cy=32.0,
        depth_scale=1000.0, camera_frame="camera_color_optical_frame",
    )
