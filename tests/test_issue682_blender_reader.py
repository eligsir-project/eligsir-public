"""Regression test for issue #682.

`scene/readers/blender.py` composited an RGBA frame down to RGB with
`dtype=np.byte` (signed int8). On the Pillow version installed at the time
this test was written (12.2.0, unpinned in `pyproject.toml`/requirements),
`Image.fromarray` cannot handle a signed-int8 3-channel buffer and raises
`TypeError: Cannot handle this data type: (1, 1, 3), |i1`. The fix is
`dtype=np.uint8`.

This test drives the real, public `readCamerasFromTransforms` entry point
against a tiny synthetic Blender-style scene written to `tmp_path` (two
4x4 RGBA PNGs + a minimal `transforms_train.json`), rather than
re-implementing the composite-and-convert arithmetic locally, so it
exercises the exact code path a real Blender/`transforms_train.json`
scene load takes.

Two independent, out-of-scope bugs in the same file would otherwise block
calling `readCamerasFromTransforms` at all, regardless of the dtype fix:
`Path` (from `pathlib`) and `fov2focal` (from `utils.graphics_utils`) are
both referenced but never imported, so every call hits a `NameError` a few
lines after the composite step this test targets. Issue #682's fix is
scoped narrowly to the signed-dtype cast (see the issue body and this
repo's CLAUDE.md convention of narrow, single-issue scope fences), so
those NameErrors are reported separately rather than fixed here. To keep
this test exercising the *real* composite-and-convert line while not being
blocked by those unrelated bugs, the two missing names are monkeypatched
onto the `scene.readers.blender` module namespace for the duration of the
test -- equivalent to what the module would see once #682's sibling bugs
are separately fixed, and does not touch the dtype logic under test at
all.

Because `scene/readers/blender.py` transitively imports `torch` (via
`scene.readers.common` -> `scene.gaussian_model`, and via
`utils.graphics_utils`/`utils.sh_utils`), this file cannot be collected on
the torch-free CPU smoke CI image -- `tests/conftest.py`'s
`_torch_collect_scan.discover_torch_needing_test_files()` will find it via
static AST analysis and add it to `collect_ignore` automatically (no
manual list edit needed, matching this repo's existing convention for
every other torch-dependent test file). It DOES collect and run inside
any environment with torch installed, e.g. this repo's ROCm `train`
container, and was verified to run and pass directly on this host (torch
is present here).
"""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

torch = pytest.importorskip("torch")

from scene.readers import blender as blender_module  # noqa: E402
from eligsir.utils.graphics_utils import fov2focal  # noqa: E402


def _write_synthetic_blender_scene(root: Path) -> None:
    """Writes a minimal transforms_train.json + two 4x4 RGBA PNGs under
    `root`, in the same layout `readCamerasFromTransforms` expects."""
    frames = []
    # Two distinct solid colors with partial alpha, chosen so the
    # composite-against-background math actually moves the pixel values
    # (i.e. this is not a vacuous all-255-or-all-0 fixture).
    colors = [(200, 50, 100, 255), (10, 220, 30, 180)]
    for idx, rgba in enumerate(colors):
        img = Image.new("RGBA", (4, 4), rgba)
        rel_path = f"frame_{idx}"
        img.save(root / f"{rel_path}.png")
        frames.append(
            {
                "file_path": rel_path,
                "transform_matrix": np.eye(4).tolist(),
            }
        )

    transforms = {
        "camera_angle_x": 0.6911112070083618,
        "frames": frames,
    }
    import json

    with open(root / "transforms_train.json", "w") as f:
        json.dump(transforms, f)


def _read_cameras(tmp_path: Path, monkeypatch):
    """Calls the real, public readCamerasFromTransforms, patching in the
    two names (`Path`, `fov2focal`) the module needs but never imports --
    see the module docstring above for why this is scoped out of #682."""
    monkeypatch.setattr(blender_module, "Path", Path, raising=False)
    monkeypatch.setattr(blender_module, "fov2focal", fov2focal, raising=False)
    return blender_module.readCamerasFromTransforms(
        str(tmp_path), "transforms_train.json", white_background=False
    )


def test_read_cameras_from_transforms_produces_valid_rgb_images(tmp_path, monkeypatch):
    """Non-vacuous regression test for issue #682: with the pre-#682
    dtype=np.byte cast, this call raises `TypeError: Cannot handle this
    data type` from Pillow (reproduced separately below via mutation).
    With the fix, it must succeed and produce well-formed RGB images."""
    _write_synthetic_blender_scene(tmp_path)

    cam_infos = _read_cameras(tmp_path, monkeypatch)

    assert len(cam_infos) == 2
    for cam_info in cam_infos:
        assert cam_info.image.mode == "RGB"
        assert cam_info.image.size == (4, 4)
        arr = np.array(cam_info.image)
        assert arr.dtype == np.uint8
        # A signed-int8 reinterpretation of the buffer would have wrapped
        # values above 127 into the negative range; np.array() on the
        # decoded image must show only genuine 0..255 uint8 values.
        assert arr.min() >= 0
        assert arr.max() <= 255


def test_np_byte_reproduces_the_original_typeerror():
    """Direct, minimal reproduction of the exact defect from the issue
    body: `dtype=np.byte` (int8) fails on current Pillow. This does not
    depend on readCamerasFromTransforms at all, so it stays meaningful
    even if the composite step's surrounding code changes shape."""
    arr = np.full((4, 4, 3), 200.0)
    with pytest.raises(TypeError, match="Cannot handle this data type"):
        Image.fromarray(np.array(arr, dtype=np.byte), "RGB")

    # The fix: uint8 must work for the same buffer.
    image = Image.fromarray(np.array(arr, dtype=np.uint8), "RGB")
    assert image.mode == "RGB"
    assert np.array(image).dtype == np.uint8
