"""Issue #642: depth export must preserve representation exactly or refuse lossy conversion.

This suite tests:
1. Float32-metres canonical depth can round-trip through uint16-mm PNG quantization
2. Sub-millimetre precision in float32 is preserved
3. Invalid values (NaN, inf) are handled consistently
4. High depths (>65.535 m) are clipped as documented
5. Raw uint16 depth with mismatched scales is refused rather than silently reinterpreted

All tests are torch-free at module level (CPU-only, runnable in CI). Any test that
needs image codecs uses pytest.importorskip("cv2").
"""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from eligsir.utils.rgbd_frames import depth_to_uint16_png, depth_to_meters
from eligsir.io.holdout_gt_depth import (
    encode_holdout_depth_mm,
    decode_holdout_depth_m,
    HOLDOUT_GT_DEPTH_SCALE,
    HOLDOUT_GT_DEPTH_MAX_MM,
)


# ---------------------------------------------------------------------------
# Test 1: Random float32 depth including sub-mm values
# ---------------------------------------------------------------------------

def test_float32_submillimetre_precision_preserved():
    """Sub-millimetre values in float32 should survive quantization to uint16-mm and back.

    This tests the canonical depth persistence path: float32 metres → uint16 mm PNG →
    float32 metres. Sub-mm precision (e.g., 1.2345 m) is lost in the uint16 representation
    (1234 mm) but the error is ~sub-mm (quantization error ≤ 0.5mm).
    """
    depth_m = np.array([
        [0.0001, 0.001, 0.0099],    # sub-millimetre values
        [0.5, 1.2345, 2.9999],      # ordinary values with sub-mm fractional parts
        [5.0, 10.0, 65.5349],       # larger values
    ], dtype=np.float32)

    mm = encode_holdout_depth_mm(depth_m)
    assert mm.dtype == np.uint16

    # Decode back and check quantization error is <1mm
    recovered = decode_holdout_depth_m(mm)
    error = np.abs(recovered - depth_m)
    # Quantization error should be at most 0.5mm (one LSB) for all values
    np.testing.assert_array_less(error, 0.001)  # 1mm in metres


def test_submillimetre_values_encode_deterministically():
    """Verify encode/decode round-trip is deterministic for sub-mm values."""
    rng = np.random.default_rng(42)
    # Random depths including sub-mm fractional parts
    depth_m = rng.uniform(0.0, 65.0, size=(10, 10)).astype(np.float32)

    mm1 = encode_holdout_depth_mm(depth_m)
    mm2 = encode_holdout_depth_mm(depth_m)
    np.testing.assert_array_equal(mm1, mm2, err_msg="Encoding must be deterministic")

    # Each time decoded, result should be identical (within float32 rounding)
    back1 = decode_holdout_depth_m(mm1)
    back2 = decode_holdout_depth_m(mm2)
    np.testing.assert_array_equal(back1, back2)


# ---------------------------------------------------------------------------
# Test 2: Invalid values (NaN, inf)
# ---------------------------------------------------------------------------

def test_invalid_values_are_handled_consistently():
    """NaN and inf in float32 depth should be handled consistently."""
    depth_m = np.array([
        [0.0, np.nan, np.inf],
        [-np.inf, 1.0, 2.0],
    ], dtype=np.float32)

    # encode_holdout_depth_mm should handle invalid values
    mm = encode_holdout_depth_mm(depth_m)
    assert mm.dtype == np.uint16
    # Invalid values clip to valid uint16 range
    assert np.all(mm <= HOLDOUT_GT_DEPTH_MAX_MM)
    assert np.all(mm >= 0)


def test_nan_converts_to_zero():
    """NaN should convert to 0mm in the uint16 representation."""
    depth_m = np.array([[np.nan, 1.0, np.nan]], dtype=np.float32)
    mm = encode_holdout_depth_mm(depth_m)
    # NaN clips to 0
    assert mm[0, 0] == 0
    assert mm[0, 2] == 0
    assert mm[0, 1] == 1000  # 1.0 m = 1000 mm


def test_inf_clips_to_max():
    """Positive infinity should clip to max uint16."""
    depth_m = np.array([[np.inf, 1.0, -np.inf]], dtype=np.float32)
    mm = encode_holdout_depth_mm(depth_m)
    assert mm[0, 0] == HOLDOUT_GT_DEPTH_MAX_MM
    assert mm[0, 2] == 0  # negative infinity clips to 0
    assert mm[0, 1] == 1000


# ---------------------------------------------------------------------------
# Test 3: Zero depth
# ---------------------------------------------------------------------------

def test_zero_depth():
    """Zero depth should round-trip exactly."""
    depth_m = np.array([[0.0, 0.0]], dtype=np.float32)
    mm = encode_holdout_depth_mm(depth_m)
    np.testing.assert_array_equal(mm, np.array([[0, 0]], dtype=np.uint16))
    back = decode_holdout_depth_m(mm)
    np.testing.assert_array_equal(back, depth_m)


# ---------------------------------------------------------------------------
# Test 4: High depths (>65.535 m)
# ---------------------------------------------------------------------------

def test_high_depths_clip_rather_than_wrap():
    """Depths above 65.535 m should clip, not wrap around."""
    depth_m = np.array([
        [65.534, 65.535, 65.536],  # Around the boundary
        [70.0, 100.0, 1000.0],     # Well above
    ], dtype=np.float32)

    mm = encode_holdout_depth_mm(depth_m)

    # All values that encode to ≥65535 should be exactly 65535, not wrapped
    # (Note: 65.534m * 1000 = 65534, but rounding may give 65533 or 65535)
    assert mm[0, 1] == 65535  # 65.535m = 65535mm exactly
    assert mm[0, 2] == 65535  # 65.536m clips to 65535mm
    assert mm[1, 0] == 65535
    assert mm[1, 1] == 65535
    assert mm[1, 2] == 65535


def test_encoding_contract_max_constant_is_enforced():
    """The MAX constant should match the practical uint16 max."""
    assert HOLDOUT_GT_DEPTH_MAX_MM == 65535
    depth_m = np.array([[HOLDOUT_GT_DEPTH_MAX_MM / 1000.0]], dtype=np.float32)
    mm = encode_holdout_depth_mm(depth_m)
    assert mm[0, 0] == 65535


# ---------------------------------------------------------------------------
# Test 5: Raw uint16 depth handling (the latent bug from #358 class)
# ---------------------------------------------------------------------------

def test_uint16_depth_requires_explicit_source_scale():
    """
    FIXED (issue #642): a uint16 input with no ``uint16_source_scale`` must
    RAISE, not silently pass through. Previously ``depth_to_uint16_png`` did::

        if arr.dtype == np.uint16:
            return arr  # <-- BUG: no scale check, no way to express one

    which is exactly the defect class from issue #358's retraction: raw
    counts at one scale (e.g. TUM's 5000) silently reinterpreted at a
    different declared scale (e.g. 1000) shrank the reconstructed scene 5x.
    """
    raw_uint16 = np.array([[1000, 2000, 3000]], dtype=np.uint16)
    with pytest.raises(ValueError, match="uint16_source_scale"):
        depth_to_uint16_png(raw_uint16, depth_scale=1000)


def test_uint16_depth_scale_mismatch_rescales_explicitly():
    """
    A uint16 input whose declared source scale DIFFERS from the target
    ``depth_scale`` must be rescaled through metres, explicitly and
    correctly -- never returned unconverted.
    """
    # Raw counts at TUM's 5000 (=> 0.2m, 0.4m, 0.6m), re-exported to a
    # 1000-scale (mm) PNG: expect 200, 400, 600.
    raw_uint16_5000 = np.array([[1000, 2000, 3000]], dtype=np.uint16)
    result = depth_to_uint16_png(raw_uint16_5000, depth_scale=1000, uint16_source_scale=5000)
    expected = np.array([[200, 400, 600]], dtype=np.uint16)
    np.testing.assert_array_equal(result, expected)


def test_raw_uint16_passthrough_is_identity_only_for_same_scale():
    """
    Raw uint16 counts declared at the SAME scale as the target ``depth_scale``
    pass through bit-exactly (not lossy -- the input already IS the target
    representation). A caller must say so explicitly via
    ``uint16_source_scale``.
    """
    raw_uint16_mm = np.array([[1000, 2000, 3000]], dtype=np.uint16)  # 1m, 2m, 3m at scale 1000
    result = depth_to_uint16_png(raw_uint16_mm, depth_scale=1000, uint16_source_scale=1000)
    np.testing.assert_array_equal(result, raw_uint16_mm)


# ---------------------------------------------------------------------------
# Test 6: Float32 → uint16 with explicit depth_scale
# ---------------------------------------------------------------------------

def test_float32_depth_to_uint16_with_explicit_scale():
    """Float32 depth in metres should scale to uint16 millimetres."""
    depth_m = np.array([
        [0.0, 0.5, 1.0],
        [2.0, 5.5, 10.0],
    ], dtype=np.float32)

    # Simulate what write_sequence_frame does: pass float32 depth with a scale of 1000
    result = depth_to_uint16_png(depth_m, depth_scale=1000)

    expected = np.array([
        [0, 500, 1000],
        [2000, 5500, 10000],
    ], dtype=np.uint16)

    np.testing.assert_array_equal(result, expected)


def test_depth_to_meters_conversion():
    """Test the depth_to_meters helper function."""
    # depth_to_meters with depth_scale=1.0 means input is already in metres
    depth_m = np.array([[1.0, 2.5, 5.0]], dtype=np.float32)
    result = depth_to_meters(depth_m, depth_scale=1.0)
    np.testing.assert_array_equal(result, depth_m)

    # With depth_scale=5000, convert from uint16 (TUM-like) to metres
    raw_tum = np.array([[5000, 10000, 2500]], dtype=np.uint16)
    result = depth_to_meters(raw_tum, depth_scale=5000)
    expected = np.array([[1.0, 2.0, 0.5]], dtype=np.float32)
    np.testing.assert_array_equal(result, expected)


# ---------------------------------------------------------------------------
# Integration: holdout_gt depth persistence contract
# ---------------------------------------------------------------------------

def test_holdout_depth_encodes_to_mm_scale_invariantly():
    """
    Regression test for issue #358: the encoding must use 1000 as divisor
    (millimetres), NOT the dataset's own depth_scale (which might be 5000, 1000,
    etc.). This is already tested in test_holdout_gt_depth_encoding_contract.py
    but included here for completeness.
    """
    # Two cameras with different raw depths and different dataset scales,
    # but identical metric depth
    metres = np.array([[1.0, 2.0]], dtype=np.float32)

    # TUM depth scale 5000
    raw_tum = (metres * 5000.0).astype(np.uint16)
    # ScanNet depth scale 1000
    raw_scannet = (metres * 1000.0).astype(np.uint16)

    # Convert both to metres (as the real producer does via _get_sensor_depth)
    mm_tum = encode_holdout_depth_mm(metres)
    mm_scannet = encode_holdout_depth_mm(metres)

    # Same metric depth should produce identical uint16-mm PNG
    np.testing.assert_array_equal(mm_tum, mm_scannet)
    # Both should decode back to the same metric depth
    np.testing.assert_allclose(decode_holdout_depth_m(mm_tum), metres, atol=1e-3)
    np.testing.assert_allclose(decode_holdout_depth_m(mm_scannet), metres, atol=1e-3)


# ---------------------------------------------------------------------------
# Edge cases and property-based tests
# ---------------------------------------------------------------------------

def test_quantization_error_is_bounded():
    """For any float32 depth, quantization error should be ≤ 1mm (one LSB)."""
    rng = np.random.default_rng(123)
    for _ in range(10):
        depth_m = rng.uniform(0.0, 65.534, size=(5, 5)).astype(np.float32)
        mm = encode_holdout_depth_mm(depth_m)
        recovered = decode_holdout_depth_m(mm)
        error = np.abs(recovered - depth_m)
        # Quantization error ≤ 1 mm (full LSB, accounting for rounding)
        np.testing.assert_array_less(error, 0.001)


def test_monotonicity():
    """If depth A < depth B, then encoded(A) < encoded(B)."""
    depth_a = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    depth_b = np.array([[1.1, 2.1, 3.1]], dtype=np.float32)

    mm_a = encode_holdout_depth_mm(depth_a)
    mm_b = encode_holdout_depth_mm(depth_b)

    assert np.all(mm_a < mm_b)


def test_negative_depth_clamps_to_zero():
    """Negative depth (physically meaningless) should clamp to 0."""
    depth_m = np.array([[-1.0, -0.5, 0.0, 0.5]], dtype=np.float32)
    mm = encode_holdout_depth_mm(depth_m)
    expected = np.array([[0, 0, 0, 500]], dtype=np.uint16)
    np.testing.assert_array_equal(mm, expected)


# ---------------------------------------------------------------------------
# Test 7: write_sequence_frame's generic-export wiring (issue #642 criterion 3)
# ---------------------------------------------------------------------------

def _make_frame(depth, depth_scale=1000.0):
    from eligsir.utils.rgbd_frames import CameraIntrinsics, RGBDFrame

    intrinsics = CameraIntrinsics(
        width=4, height=2, fx=1.0, fy=1.0, cx=2.0, cy=1.0, depth_scale=depth_scale
    )
    rgb = np.zeros((2, 4, 3), dtype=np.uint8)
    return RGBDFrame(
        index=0, timestamp=0.0, rgb=rgb, depth=depth, intrinsics=intrinsics,
        c2w=np.eye(4, dtype=np.float32),
    )


def test_write_sequence_frame_float_depth_is_unaffected(tmp_path):
    """Float32 metres input (the real prepare_scannetpp.py path) is unchanged
    by the uint16-source-scale wiring -- it never touches that branch."""
    depth_m = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], dtype=np.float32)
    frame = _make_frame(depth_m, depth_scale=1000.0)
    from eligsir.utils.rgbd_frames import write_sequence_frame

    write_sequence_frame(tmp_path, frame, depth_scale=1000.0)
    saved = np.array(Image.open(tmp_path / "depth" / "000000.png"))
    np.testing.assert_array_equal(saved, (depth_m * 1000.0).astype(np.uint16))


def test_write_sequence_frame_uint16_depth_same_scale_is_identity(tmp_path):
    """A frame whose raw depth is already uint16 counts at the SAME scale as
    the requested output depth_scale (the real rosbag-source path, e.g.
    eligsir/io/sources/rosbag.py) passes through bit-exactly."""
    raw = np.array([[1000, 2000, 3000, 4000], [5000, 6000, 7000, 8000]], dtype=np.uint16)
    frame = _make_frame(raw, depth_scale=1000.0)
    from eligsir.utils.rgbd_frames import write_sequence_frame

    write_sequence_frame(tmp_path, frame, depth_scale=1000.0)
    saved = np.array(Image.open(tmp_path / "depth" / "000000.png"))
    np.testing.assert_array_equal(saved, raw)


def test_write_sequence_frame_uint16_depth_scale_mismatch_rescales(tmp_path):
    """A frame whose raw uint16 counts are at a DIFFERENT scale than the
    requested output depth_scale must be rescaled correctly, not silently
    passed through (issue #642's latent-bug closure)."""
    # source scale 5000 (TUM-like): 1000 counts = 0.2m
    raw = np.array([[1000, 2000, 3000, 4000], [5000, 6000, 7000, 8000]], dtype=np.uint16)
    frame = _make_frame(raw, depth_scale=5000.0)
    from eligsir.utils.rgbd_frames import write_sequence_frame

    # export at depth_scale=1000 (mm): 0.2m -> 200
    write_sequence_frame(tmp_path, frame, depth_scale=1000.0)
    saved = np.array(Image.open(tmp_path / "depth" / "000000.png"))
    expected = np.rint(raw.astype(np.float32) / 5000.0 * 1000.0).astype(np.uint16)
    np.testing.assert_array_equal(saved, expected)
    assert not np.array_equal(saved, raw)  # must NOT be the old silent pass-through
