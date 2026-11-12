"""Issue #639: holdout GT persistence must be bit-exact in the score path.

``holdout_gt/`` feeds ``render_report.py --trajectory test --eval``, which
assigns what it finds there to ``cam.original_image`` -- i.e. straight into
PSNR/SSIM/LPIPS. It used to be JPEG q92, which (measured on
``output/wave1_verify_fr3``, 8/8 views) INFLATES post-hoc test PSNR by
+0.175 dB.

These are CPU-only and do not import torch, cv2 or the training stack at
module level.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

from eligsir.io.holdout_gt_exact import (
    EXACT_RGB_FORMATS,
    canonical_hash,
    load_holdout_depth,
    load_holdout_rgb,
    save_holdout_depth,
    save_holdout_rgb,
    summarize_provenance,
)


def _uint8_derived_rgb(h=7, w=5, seed=0) -> np.ndarray:
    """GT as it comes off an ordinary uint8 dataset image: float32 CHW i/255."""
    rng = np.random.default_rng(seed)
    u8 = rng.integers(0, 256, size=(3, h, w), dtype=np.uint8)
    return np.ascontiguousarray(u8.astype(np.float32) / np.float32(255.0))


def _genuinely_float_rgb(h=7, w=5, seed=1) -> np.ndarray:
    """GT that is NOT on the 1/255 lattice (resized/synthesized/interpolated)."""
    rng = np.random.default_rng(seed)
    return np.ascontiguousarray(rng.random((3, h, w), dtype=np.float32))


# --- acceptance: reloaded GT is np.array_equal to the canonical source -------

def test_uint8_derived_gt_roundtrips_bit_exactly(tmp_path):
    pytest.importorskip("cv2", reason="lossless PNG branch needs an image codec")
    gt = _uint8_derived_rgb()
    meta = save_holdout_rgb(str(tmp_path), 0, gt)
    assert meta["format"] in EXACT_RGB_FORMATS
    back, prov = load_holdout_rgb(str(tmp_path), 0)
    assert np.array_equal(back, gt), "reloaded GT differs from the canonical samples"
    assert back.dtype == np.float32
    assert prov["bit_exact"] is True
    assert prov.get("lossy") is not True


def test_genuinely_float_gt_takes_the_npy_branch_and_is_exact(tmp_path):
    gt = _genuinely_float_rgb()
    meta = save_holdout_rgb(str(tmp_path), 3, gt)
    assert meta["format"] == "npy_float32", "float GT must not be reduced to 8 bits"
    back, prov = load_holdout_rgb(str(tmp_path), 3)
    assert np.array_equal(back, gt)
    assert prov["bit_exact"] is True


def test_sidecar_records_the_full_representation_contract(tmp_path):
    gt = _genuinely_float_rgb(h=4, w=6)
    save_holdout_rgb(str(tmp_path), 11, gt)
    meta = json.load(open(os.path.join(str(tmp_path), "11_rgb.json")))
    assert meta["dtype"] == "float32"
    assert meta["shape"] == [3, 4, 6]
    assert meta["layout"] == "CHW"
    assert meta["channel_order"] == "RGB"
    assert meta["domain"] == "unit_interval_0_1"
    assert meta["sha256"] == canonical_hash(gt)


# --- no JPEG anywhere in the new score path ---------------------------------

def test_writer_emits_no_jpeg(tmp_path):
    save_holdout_rgb(str(tmp_path), 0, _uint8_derived_rgb())
    save_holdout_rgb(str(tmp_path), 1, _genuinely_float_rgb())
    written = os.listdir(str(tmp_path))
    assert not [f for f in written if f.lower().endswith((".jpg", ".jpeg"))], written


def test_a_visualization_file_cannot_be_selected_as_a_metric_input(tmp_path):
    """A preview artifact next to the GT must never be picked by the loader."""
    gt = _genuinely_float_rgb()
    save_holdout_rgb(str(tmp_path), 0, gt)
    prev_dir = os.path.join(str(tmp_path), "preview")
    os.makedirs(prev_dir)
    open(os.path.join(prev_dir, "0_rgb.jpg"), "wb").write(b"not an image")
    # A rogue jpg beside the exact file must lose to the sidecar-declared one.
    open(os.path.join(str(tmp_path), "0_rgb.jpg"), "wb").write(b"not an image")
    back, prov = load_holdout_rgb(str(tmp_path), 0)
    assert prov["format"] == "npy_float32"
    assert np.array_equal(back, gt)


def test_sidecar_declaring_a_non_allowlisted_format_is_refused(tmp_path):
    gt = _genuinely_float_rgb()
    save_holdout_rgb(str(tmp_path), 0, gt)
    p = os.path.join(str(tmp_path), "0_rgb.json")
    meta = json.load(open(p))
    meta["format"] = "jpeg_q92"
    json.dump(meta, open(p, "w"))
    back, prov = load_holdout_rgb(str(tmp_path), 0)
    assert back is None
    assert prov["bit_exact"] is False


# --- tamper detection -------------------------------------------------------

def test_tampered_payload_is_caught_by_the_canonical_hash(tmp_path):
    gt = _genuinely_float_rgb()
    save_holdout_rgb(str(tmp_path), 0, gt)
    tampered = gt.copy()
    tampered[0, 0, 0] = np.float32(0.5) if tampered[0, 0, 0] != 0.5 else np.float32(0.25)
    np.save(os.path.join(str(tmp_path), "0_rgb.npy"), tampered)
    back, prov = load_holdout_rgb(str(tmp_path), 0)
    assert prov["bit_exact"] is False
    assert prov.get("hash_mismatch") is True
    assert back is not None  # still returned, but explicitly untrusted
    back2, prov2 = load_holdout_rgb(str(tmp_path), 0, require_exact=True)
    assert back2 is None and prov2.get("refused") is True


# --- backward compatibility: pre-#639 JPEG-only runs -------------------------

def _write_legacy_jpeg(gt_dir, cam_id, rgb_chw):
    cv2 = pytest.importorskip("cv2", reason="legacy fixture needs a JPEG encoder")
    os.makedirs(gt_dir, exist_ok=True)
    hwc = np.clip(np.rint(np.transpose(rgb_chw, (1, 2, 0)) * 255.0), 0, 255).astype(np.uint8)
    assert cv2.imwrite(os.path.join(gt_dir, f"{cam_id}_rgb.jpg"),
                       cv2.cvtColor(hwc, cv2.COLOR_RGB2BGR),
                       [int(cv2.IMWRITE_JPEG_QUALITY), 92])


def test_legacy_jpeg_run_still_loads_but_is_marked_lossy(tmp_path):
    gt = _uint8_derived_rgb(h=16, w=16)
    _write_legacy_jpeg(str(tmp_path), 0, gt)
    back, prov = load_holdout_rgb(str(tmp_path), 0)
    assert back is not None, "a pre-#639 run must not crash the reader"
    assert back.shape == gt.shape
    assert prov["format"] == "jpeg_legacy"
    assert prov["bit_exact"] is False
    assert prov["lossy"] is True
    assert "630" in prov["warning"]


def test_require_exact_refuses_a_legacy_jpeg_run(tmp_path):
    _write_legacy_jpeg(str(tmp_path), 0, _uint8_derived_rgb(h=16, w=16))
    back, prov = load_holdout_rgb(str(tmp_path), 0, require_exact=True)
    assert back is None
    assert prov["refused"] is True


def test_resume_across_the_fix_boundary_yields_an_honest_mixed_aggregate(tmp_path):
    """A run resumed under post-#639 code re-writes only the cameras it
    re-materializes, leaving the rest as bare pre-#639 JPEGs. The per-camera
    resolution must be right AND the aggregate must not read as clean."""
    exact_gt = _uint8_derived_rgb(h=16, w=16, seed=7)
    _write_legacy_jpeg(str(tmp_path), 0, exact_gt)   # survives from the old segment
    _write_legacy_jpeg(str(tmp_path), 1, exact_gt)   # cam 1's stale jpg...
    save_holdout_rgb(str(tmp_path), 1, exact_gt)     # ...overwritten exactly on resume

    p0 = load_holdout_rgb(str(tmp_path), 0)[1]
    back1, p1 = load_holdout_rgb(str(tmp_path), 1)
    assert p0["format"] == "jpeg_legacy" and p0["lossy"] is True
    assert p1["bit_exact"] is True, "the sidecar must win over a surviving .jpg"
    assert np.array_equal(back1, exact_gt)

    s = summarize_provenance([p0, p1])
    assert s["bit_exact"] is False, "a partially-exact run must not read as clean"
    assert s["lossy_gt_present"] is True
    assert s["n_bit_exact"] == 1 and s["n_cameras"] == 2


def test_missing_gt_is_reported_not_raised(tmp_path):
    back, prov = load_holdout_rgb(str(tmp_path), 42)
    assert back is None and prov["format"] == "missing"


# --- depth ------------------------------------------------------------------

def test_depth_npy_is_exact_and_preferred_over_the_legacy_uint16_png(tmp_path):
    rng = np.random.default_rng(5)
    depth = np.ascontiguousarray((rng.random((6, 8)) * 4.0).astype(np.float32))
    save_holdout_depth(str(tmp_path), 0, depth)
    back, prov = load_holdout_depth(str(tmp_path), 0)
    assert np.array_equal(back, depth), "1 mm quantization must not survive into the score path"
    assert prov["format"] == "npy_float32" and prov["bit_exact"] is True


def test_legacy_depth_png_alone_is_marked_lossy(tmp_path):
    cv2 = pytest.importorskip("cv2")
    from eligsir.io.holdout_gt_depth import encode_holdout_depth_mm
    depth = np.array([[0.5, 1.25], [2.0, 0.0]], dtype=np.float32)
    os.makedirs(str(tmp_path), exist_ok=True)
    cv2.imwrite(os.path.join(str(tmp_path), "0_depth.png"), encode_holdout_depth_mm(depth))
    back, prov = load_holdout_depth(str(tmp_path), 0)
    assert back is not None
    assert prov["format"] == "png_uint16_mm_legacy"
    assert prov["bit_exact"] is False and prov["lossy"] is True


# --- provenance aggregation -------------------------------------------------

def test_summarize_provenance_is_exact_only_when_every_camera_is(tmp_path):
    ok = {"format": "npy_float32", "bit_exact": True}
    bad = {"format": "jpeg_legacy", "bit_exact": False, "lossy": True, "warning": "w"}
    assert summarize_provenance([ok, ok])["bit_exact"] is True
    s = summarize_provenance([ok, bad])
    assert s["bit_exact"] is False
    assert s["lossy_gt_present"] is True
    assert s["n_bit_exact"] == 1 and s["n_lossy"] == 1
    assert summarize_provenance([])["bit_exact"] is False


# --- the producer itself ----------------------------------------------------

def test_save_holdout_gt_writes_no_jpeg_and_is_reloadable_bit_exactly(tmp_path):
    """End-to-end through the real producer, with a stand-in camera."""
    torch = pytest.importorskip("torch")
    from eligsir.mapping.integration import checkpoint_io

    gt = _uint8_derived_rgb(h=6, w=9)

    class _Cam:
        image_height, image_width = 6, 9
        original_image = torch.from_numpy(gt.copy())

    # _gt_image moves to cuda; on a CPU surface patch it to the identity clamp.
    import eligsir.utils.comparison_report as cr
    orig_gt_image, orig_depth = cr._gt_image, cr._get_sensor_depth
    cr._gt_image = lambda cam: cam.original_image[:3].clamp(0.0, 1.0)
    cr._get_sensor_depth = lambda cam, h, w: None
    checkpoint_io._HOLDOUT_GT_SAVED_IDS.clear()
    try:
        checkpoint_io._save_holdout_gt(None, str(tmp_path), [_Cam()])
    finally:
        cr._gt_image, cr._get_sensor_depth = orig_gt_image, orig_depth
        checkpoint_io._HOLDOUT_GT_SAVED_IDS.clear()

    gt_dir = os.path.join(str(tmp_path), "holdout_gt")
    written = os.listdir(gt_dir)
    assert not [f for f in written if f.lower().endswith((".jpg", ".jpeg"))], written
    back, prov = load_holdout_rgb(gt_dir, 0)
    assert prov["bit_exact"] is True
    assert np.array_equal(back, gt)
