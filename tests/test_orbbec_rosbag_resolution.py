from __future__ import annotations

import io

import numpy as np
from PIL import Image

from eligsir.io.sources.datasets.orbbec_rosbag import (
    _resize_png_payload,
    _scale_intrinsics_for_resize,
)


def _png_bytes(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return buffer.getvalue()


def test_registered_depth_downscale_preserves_uint16_depth_samples():
    depth = np.zeros((8, 8), dtype=np.uint16)
    depth[2:6, 2:6] = np.uint16(2345)

    resized = _resize_png_payload(
        _png_bytes(depth),
        width=4,
        height=4,
        resample=Image.Resampling.NEAREST,
    )

    with Image.open(io.BytesIO(resized)) as image:
        decoded = np.asarray(image)

    assert decoded.shape == (4, 4)
    assert decoded.dtype == np.uint16
    assert np.count_nonzero(decoded) > 0
    assert decoded.max() == np.uint16(2345)


def test_nondivisible_resize_scales_intrinsics_by_actual_output_shape():
    scaled = _scale_intrinsics_for_resize(
        1000.0,
        900.0,
        640.0,
        360.0,
        source_width=1281,
        source_height=721,
        target_width=160,
        target_height=90,
    )

    assert scaled == (
        1000.0 * 160 / 1281,
        900.0 * 90 / 721,
        640.0 * 160 / 1281,
        360.0 * 90 / 721,
    )
