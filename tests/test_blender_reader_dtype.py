"""Regression tests for issue #682: the Blender reader must load at all.

Two distinct defects in the same unexercised code path, both covered here:

1. The composite used ``dtype=np.byte`` (signed int8), which Pillow >= 12
   rejects outright (see below).
2. Three names were used but never imported (``Path``, ``fov2focal``,
   ``BasicPointCloud``), so ``readCamerasFromTransforms`` raised ``NameError``
   on line 39 -- *before* ever reaching the dtype line.

The reader used ``dtype=np.byte`` (signed int8). Pillow >= 12 rejects that
array outright (``TypeError: Cannot handle this data type: (1, 1, 3), |i1``),
a hard crash on any ``transforms_train.json`` scene load.

The dtype assertion below is the version-independent one: under Pillow < 12
``fromarray`` accepted the int8 array and the wrapped bytes happened to read
back with the right values, so a value-only test would not have failed on the
old code.
"""

import json
import math

import numpy as np
import pytest
from PIL import Image

from scene.readers.blender import (
    composite_rgba_to_rgb_uint8,
    readCamerasFromTransforms,
    readNerfSyntheticInfo,
)


@pytest.mark.parametrize("white_background", [False, True])
def test_composite_is_unsigned_and_roundtrips_above_127(white_background):
    # Opaque pixels spanning the signed-int8 wraparound boundary.
    im_data = np.zeros((2, 3, 4), dtype=np.uint8)
    im_data[..., 3] = 255
    im_data[0, 0, :3] = (0, 127, 128)
    im_data[0, 1, :3] = (200, 204, 255)
    im_data[0, 2, :3] = (255, 255, 255)
    im_data[1, :, :3] = 130

    arr = composite_rgba_to_rgb_uint8(im_data, white_background)

    # Primary assertion: fails on the old signed dtype on every Pillow version.
    assert arr.dtype == np.uint8
    assert arr.min() >= 0

    # Secondary: reproduces the actual Pillow >= 12 crash, and checks that
    # values above 127 survive instead of wrapping negative.
    image = Image.fromarray(arr, "RGB")
    back = np.array(image)
    assert back.dtype == np.uint8
    np.testing.assert_array_equal(back, im_data[..., :3])


def test_transparent_pixels_take_the_background():
    im_data = np.zeros((1, 1, 4), dtype=np.uint8)  # alpha = 0

    np.testing.assert_array_equal(
        composite_rgba_to_rgb_uint8(im_data, white_background=True), [[[255, 255, 255]]]
    )
    np.testing.assert_array_equal(
        composite_rgba_to_rgb_uint8(im_data, white_background=False), [[[0, 0, 0]]]
    )


def _write_minimal_blender_scene(root, transforms=("transforms_train.json",)):
    """A synthetic Blender scene: transforms JSON(s) + opaque RGBA PNGs."""
    frames = []
    for idx, translation in enumerate(((0.0, 0.0, 4.0), (0.0, 4.0, 0.0), (4.0, 0.0, 0.0))):
        rgba = np.zeros((4, 6, 4), dtype=np.uint8)
        rgba[..., 3] = 255
        rgba[..., :3] = 200 + idx  # opaque, above the signed-int8 boundary
        Image.fromarray(rgba, "RGBA").save(root / f"r_{idx}.png")

        c2w = np.eye(4)
        c2w[:3, 3] = translation
        frames.append({"file_path": f"./r_{idx}", "transform_matrix": c2w.tolist()})

    payload = json.dumps({"camera_angle_x": 0.6911112070083618, "frames": frames})
    for name in transforms:
        (root / name).write_text(payload)


def test_reader_loads_a_minimal_blender_scene_end_to_end(tmp_path):
    """The whole point of the import fix: the reader entry point is reachable.

    On the pre-fix code this raises NameError('Path') at the image_name line,
    which sits *before* the dtype line -- so the dtype fix alone is unreachable.
    """
    _write_minimal_blender_scene(tmp_path)

    cam_infos = readCamerasFromTransforms(
        str(tmp_path), "transforms_train.json", white_background=False
    )

    assert len(cam_infos) == 3
    for idx, cam in enumerate(cam_infos):
        assert cam.uid == idx
        assert cam.image_name == f"r_{idx}"
        assert (cam.width, cam.height) == (6, 4)
        assert cam.image.mode == "RGB"

        # Opaque pixels above 127 survive the composite un-wrapped.
        px = np.array(cam.image)
        assert px.dtype == np.uint8
        np.testing.assert_array_equal(px, np.full((4, 6, 3), 200 + idx, dtype=np.uint8))

        # FovY is derived from FovX via the focal length (the fov2focal import).
        assert cam.FovX == pytest.approx(0.6911112070083618)
        expected_focal = 6 / (2 * math.tan(cam.FovX / 2))
        assert cam.FovY == pytest.approx(2 * math.atan(4 / (2 * expected_focal)))
        assert cam.FovY < cam.FovX  # height < width


def test_registered_blender_entry_point_loads_a_scene(tmp_path):
    """`readNerfSyntheticInfo` is the callable `dataset_readers` maps to "Blender".

    Also the only path that touches the third missing import, `BasicPointCloud`
    (the random-point-cloud branch taken when no points3d.ply exists).
    """
    _write_minimal_blender_scene(
        tmp_path, transforms=("transforms_train.json", "transforms_test.json")
    )

    scene_info = readNerfSyntheticInfo(str(tmp_path), white_background=False, eval=True)

    assert len(scene_info.train_cameras) == 3
    assert len(scene_info.test_cameras) == 3
    assert (tmp_path / "points3d.ply").exists()
    assert scene_info.point_cloud is not None
    assert len(scene_info.point_cloud.points) == 100_000
    assert scene_info.nerf_normalization["radius"] > 0
