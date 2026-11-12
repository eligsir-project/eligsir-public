"""Issue #642 criterion 2: local-quality scripts must prefer exact float32
holdout depth (.npy) over the quantized uint16-mm legacy PNG, and must
record which one was actually used -- never silently.

Covers ``scripts/local_quality_from_run.py::RunCam`` and
``scripts/local_quality_offline_demo.py::RunCam``.

NEEDS TORCH: both scripts import torch unconditionally at module level (this
predates issue #642 and is out of this PR's scope to change), so this whole
module is skipped rather than failed on a torch-free CI image via
``pytest.importorskip("torch")``.
"""
from __future__ import annotations

import importlib.util
import os
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from eligsir.io.holdout_gt_exact import save_holdout_depth  # noqa: E402
from eligsir.io.holdout_gt_depth import encode_holdout_depth_mm  # noqa: E402


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def from_run_mod():
    return _load_module(
        "_issue642_local_quality_from_run",
        os.path.join(REPO_ROOT, "scripts", "local_quality_from_run.py"),
    )


@pytest.fixture(scope="module")
def offline_demo_mod():
    return _load_module(
        "_issue642_local_quality_offline_demo",
        os.path.join(REPO_ROOT, "scripts", "local_quality_offline_demo.py"),
    )


def _entry(cam_id=0):
    return {
        "id": cam_id, "img_name": f"cam{cam_id}",
        "rotation": np.eye(3).tolist(), "position": [0.0, 0.0, 0.0],
        "width": 4, "height": 2, "fx": 1.0, "fy": 1.0, "cx": 2.0, "cy": 1.0,
    }


def test_from_run_runcam_prefers_exact_npy_over_legacy_png(tmp_path, from_run_mod):
    gt_dir = tmp_path / "holdout_gt"
    depth_m = np.array([[1.2345, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], dtype=np.float32)
    save_holdout_depth(str(gt_dir), 0, depth_m)
    # Also write the legacy PNG, as _save_holdout_gt does -- both present.
    import cv2
    cv2.imwrite(str(gt_dir / "0_depth.png"), encode_holdout_depth_mm(depth_m))

    cam = from_run_mod.RunCam(_entry(0), str(tmp_path), depth_scale=1000.0)
    assert cam.has_depth()
    got = cam.sensor_depth()
    np.testing.assert_array_equal(got.numpy(), depth_m)
    assert cam.depth_provenance["format"] == "npy_float32"
    assert cam.depth_provenance["bit_exact"] is True


def test_from_run_runcam_falls_back_to_legacy_png_and_marks_it(tmp_path, from_run_mod):
    gt_dir = tmp_path / "holdout_gt"
    os.makedirs(gt_dir, exist_ok=True)
    depth_m = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], dtype=np.float32)
    import cv2
    cv2.imwrite(str(gt_dir / "0_depth.png"), encode_holdout_depth_mm(depth_m))
    # No .npy written -- pre-#639 run.

    cam = from_run_mod.RunCam(_entry(0), str(tmp_path), depth_scale=1000.0)
    assert cam.has_depth()
    got = cam.sensor_depth()
    np.testing.assert_allclose(got.numpy(), depth_m, atol=1e-3)
    assert cam.depth_provenance["format"] == "png_uint16_mm_legacy"
    assert cam.depth_provenance["lossy"] is True
    assert cam.depth_provenance["bit_exact"] is False


def test_offline_demo_runcam_prefers_exact_npy(tmp_path, offline_demo_mod):
    offline_demo_mod.RUN = str(tmp_path)
    gt_dir = tmp_path / "holdout_gt"
    depth_m = np.array([[1.5, 2.5, 3.5, 4.5], [5.5, 6.5, 7.5, 8.5]], dtype=np.float32)
    save_holdout_depth(str(gt_dir), 0, depth_m)

    cam = offline_demo_mod.RunCam(_entry(0))
    got = cam.sensor_depth()
    np.testing.assert_array_equal(got.numpy(), depth_m)
    assert cam.depth_provenance["format"] == "npy_float32"


def test_offline_demo_runcam_falls_back_to_legacy_and_marks_it(tmp_path, offline_demo_mod):
    offline_demo_mod.RUN = str(tmp_path)
    gt_dir = tmp_path / "holdout_gt"
    os.makedirs(gt_dir, exist_ok=True)
    depth_m = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], dtype=np.float32)
    import cv2
    cv2.imwrite(str(gt_dir / "0_depth.png"), encode_holdout_depth_mm(depth_m))

    cam = offline_demo_mod.RunCam(_entry(0))
    got = cam.sensor_depth()
    np.testing.assert_allclose(got.numpy(), depth_m, atol=1e-3)
    assert cam.depth_provenance["format"] == "png_uint16_mm_legacy"
    assert cam.depth_provenance["lossy"] is True
