"""Issue #645: Bit-exact/reversible representation tests for image/depth conversion.

Tests that lossless conversion paths (channel swaps, HWC↔CHW, uint8↔float32,
CPU↔GPU, int16 PNG persistence) preserve samples exactly or within documented
quantization error bounds.

Torch-optional: tests guard torch imports with pytest.importorskip() so the
file is visible on CPU-CI without GPU requirements.
"""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from eligsir.utils.rgbd_frames import depth_to_uint16_png, depth_to_meters, decode_color_image, decode_depth_image


# ============================================================================
# RGB8 codec round-trips (uint8 <-> float32, different channel orders)
# ============================================================================

class TestRGB8Codec:
    """RGB8 quantization and channel conversions are reversible."""

    def test_rgb8_to_float32_normalized_preserves_samples(self):
        """uint8 RGB -> float32 [0,1] normalization is exact when inverted."""
        # Create a random uint8 RGB image
        rng = np.random.default_rng(42)
        rgb8 = rng.integers(0, 256, size=(10, 10, 3), dtype=np.uint8)

        # Normalize to float32
        rgb_normalized = rgb8.astype(np.float32) / 255.0
        assert rgb_normalized.dtype == np.float32
        assert rgb_normalized.min() >= 0.0
        assert rgb_normalized.max() <= 1.0

        # Invert: float32 [0,1] -> uint8
        rgb8_recovered = np.clip(rgb_normalized * 255.0, 0, 255).astype(np.uint8)

        # Must be bit-exact
        np.testing.assert_array_equal(rgb8_recovered, rgb8)

    def test_bgr8_rgb8_roundtrip(self):
        """BGR <-> RGB channel swap is bit-exact."""
        bgr8 = np.array([
            [[255, 0, 0], [0, 255, 0]],    # BGR: pure blue, pure green
            [[0, 0, 255], [255, 255, 0]],  # BGR: pure red, cyan
        ], dtype=np.uint8)

        # BGR -> RGB
        rgb8 = decode_color_image(bgr8, encoding="bgr8")
        assert rgb8.dtype == np.uint8
        assert rgb8.shape == bgr8.shape

        # RGB channels should be in reverse order
        expected_rgb = np.array([
            [[0, 0, 255], [0, 255, 0]],    # RGB: pure red, pure green
            [[255, 0, 0], [0, 255, 255]],  # RGB: pure blue, yellow
        ], dtype=np.uint8)
        np.testing.assert_array_equal(rgb8, expected_rgb)

        # The BGR -> RGB conversion done by decode_color_image is one-way,
        # but the inverse (RGB -> BGR via cv2.cvtColor) is also exact.
        import cv2
        bgr_recovered = cv2.cvtColor(rgb8, cv2.COLOR_RGB2BGR)
        np.testing.assert_array_equal(bgr_recovered, bgr8)


# ============================================================================
# RGBA alpha channel preservation and extraction
# ============================================================================

class TestRGBAAlpha:
    """RGBA alpha channel must survive conversion and extraction."""

    def test_rgba_alpha_channel_preserved_through_uint8_path(self):
        """Alpha channel values must persist through uint8 RGB/alpha split."""
        rng = np.random.default_rng(123)
        rgba = rng.integers(0, 256, size=(5, 5, 4), dtype=np.uint8)

        # Extract RGB and alpha
        rgb = rgba[..., :3]
        alpha = rgba[..., 3:4]  # Keep 3-D (H, W, 1)

        # Verify exact reconstruction
        reconstructed = np.concatenate([rgb, alpha], axis=-1)
        np.testing.assert_array_equal(reconstructed, rgba)

    def test_rgba_alpha_normalized_to_float32(self):
        """Alpha channel normalization (uint8 -> float32 [0,1]) is reversible."""
        alpha_uint8 = np.array([[0, 64, 128, 192, 255]], dtype=np.uint8)

        # Normalize
        alpha_f32 = alpha_uint8.astype(np.float32) / 255.0

        # Expected normalized values
        expected = np.array([[0.0, 64/255, 128/255, 192/255, 1.0]], dtype=np.float32)
        np.testing.assert_allclose(alpha_f32, expected, rtol=1e-6)

        # Invert
        alpha_recovered = np.clip(alpha_f32 * 255.0, 0, 255).astype(np.uint8)
        np.testing.assert_array_equal(alpha_recovered, alpha_uint8)


# ============================================================================
# Depth uint16 bit-exact persistence (PNG round-trip)
# ============================================================================

class TestUint16DepthPersistence:
    """uint16 depth PNG persistence is bit-exact for same-scale data."""

    def test_uint16_depth_png_roundtrip_via_pil(self):
        """uint16 depth values persist exactly through PNG save/load cycle."""
        # Create a test depth array (uint16, representing mm at scale 1000)
        depth_mm = np.array([
            [0, 100, 500, 1000, 5000],
            [10000, 20000, 30000, 40000, 50000],
            [60000, 65000, 65535, 1, 2],  # Include boundary values
        ], dtype=np.uint16)

        # PIL PNG round-trip
        img = Image.fromarray(depth_mm, mode='I;16')
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            temp_path = f.name
            img.save(temp_path)
            loaded = np.array(Image.open(temp_path))

        # Must be bit-exact
        np.testing.assert_array_equal(loaded, depth_mm)

    def test_float32_depth_to_uint16_quantization_bounded(self):
        """Float32 metres -> uint16 mm quantization error <= 1mm."""
        # Create float32 depth values (metres)
        depth_m = np.array([
            [0.0, 0.001, 0.999, 1.0, 1.5],
            [2.0, 5.5, 10.0, 32.767, 65.534],
        ], dtype=np.float32)

        # Convert to uint16 mm
        depth_mm = depth_to_uint16_png(depth_m, depth_scale=1000)
        assert depth_mm.dtype == np.uint16

        # Invert to get back to metres
        recovered_m = depth_mm.astype(np.float32) / 1000.0

        # Error should be <= 1mm (0.001 m)
        error = np.abs(recovered_m - depth_m)
        np.testing.assert_array_less(error, 0.001)


# ============================================================================
# Float32 depth persistence (exact dtype/shape/values)
# ============================================================================

class TestFloat32DepthPersistence:
    """Float32 depth persistence preserves dtype, shape, and values."""

    def test_float32_depth_numpy_persistence(self):
        """NumPy .npy format preserves float32 depth exactly."""
        depth_m = np.array([
            [0.0, 0.001, 0.5, 1.0, 2.5],
            [5.0, 10.5, 32.767, 65.534, np.inf],
        ], dtype=np.float32)

        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/depth.npy"
            np.save(path, depth_m)
            loaded = np.load(path)

            # Must be identical
            assert loaded.dtype == np.float32
            assert loaded.shape == depth_m.shape
            # NaN/inf need special handling
            finite_mask = np.isfinite(depth_m)
            np.testing.assert_array_equal(loaded[finite_mask], depth_m[finite_mask])
            # Check inf survives
            np.testing.assert_array_equal(
                np.isinf(loaded), np.isinf(depth_m)
            )


# ============================================================================
# Tensor permutations (HWC <-> CHW) preserve samples
# ============================================================================

class TestTensorPermutations:
    """Tensor layout conversions (HWC <-> CHW) preserve all samples."""

    def test_hwc_to_chw_permutation_preserves_samples(self):
        """HWC -> CHW permutation is a pure reshape; no data changes."""
        # Create an HWC image (height=4, width=5, channels=3)
        hwc = np.arange(60, dtype=np.float32).reshape(4, 5, 3)

        # Permute to CHW
        chw = hwc.transpose(2, 0, 1)
        assert chw.shape == (3, 4, 5)

        # Inverse permutation
        hwc_recovered = chw.transpose(1, 2, 0)
        np.testing.assert_array_equal(hwc_recovered, hwc)

    def test_chw_to_cpu_gpu_preserves_values_when_torch_available(self):
        """CHW tensor CPU -> GPU -> CPU preserves float32 samples (torch-only)."""
        torch = pytest.importorskip("torch")

        # Create a CHW tensor on CPU
        chw_np = np.arange(60, dtype=np.float32).reshape(3, 4, 5)
        chw_cpu = torch.from_numpy(chw_np)

        # GPU transfer if available
        if torch.cuda.is_available():
            chw_gpu = chw_cpu.to("cuda")
            chw_back = chw_gpu.to("cpu")
        else:
            # Fallback on CPU-only system
            chw_back = chw_cpu

        # Must match exactly
        np.testing.assert_array_equal(chw_back.numpy(), chw_np)


# ============================================================================
# RGB8 PIL codec paths (Image.open/save)
# ============================================================================

class TestPILImageCodec:
    """PIL Image codec preserves uint8 RGB exactly for lossless formats."""

    def test_pil_image_save_load_uint8_rgb(self):
        """PIL PNG codec preserves uint8 RGB exactly."""
        rgb8 = np.array([
            [[255, 0, 0], [0, 255, 0], [0, 0, 255]],
            [[128, 128, 128], [64, 32, 16], [1, 2, 3]],
        ], dtype=np.uint8)

        img = Image.fromarray(rgb8, mode="RGB")
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            temp_path = f.name
            img.save(temp_path)
            loaded = np.array(Image.open(temp_path))

        # PNG is lossless for uint8 RGB
        np.testing.assert_array_equal(loaded, rgb8)


# ============================================================================
# Integration: color and depth decoders (rosbag compat)
# ============================================================================

class TestDecoderIntegration:
    """Decoder functions (used in rosbag ingestion) handle various encodings."""

    def test_decode_color_image_rgb8(self):
        """decode_color_image handles RGB8 encoding."""
        rgb8 = np.array([
            [[255, 0, 0], [0, 255, 0]],
            [[0, 0, 255], [128, 128, 128]],
        ], dtype=np.uint8)

        result = decode_color_image(rgb8, encoding="rgb8")
        np.testing.assert_array_equal(result, rgb8)
        assert result.dtype == np.uint8

    def test_decode_color_image_bgr8(self):
        """decode_color_image handles BGR8 and converts to RGB."""
        bgr8 = np.array([
            [[0, 0, 255], [0, 255, 0]],  # BGR: red and green
            [[255, 0, 0], [128, 128, 128]],  # BGR: blue and gray
        ], dtype=np.uint8)

        result = decode_color_image(bgr8, encoding="bgr8")
        assert result.dtype == np.uint8
        # BGR -> RGB inverts the channel order
        # BGR [0,0,255] = pure red -> RGB [255,0,0]
        assert result[0, 0, 0] == 255 and result[0, 0, 2] == 0  # R channel moved to position 0

    def test_decode_depth_image_uint16(self):
        """decode_depth_image preserves uint16 encoding."""
        depth16 = np.array([
            [0, 100, 1000, 5000],
            [10000, 30000, 50000, 65535],
        ], dtype=np.uint16)

        result = decode_depth_image(depth16, encoding="mono16")
        np.testing.assert_array_equal(result, depth16)
        assert result.dtype == np.uint16

    def test_decode_depth_image_float32(self):
        """decode_depth_image preserves float32 encoding."""
        depth32 = np.array([
            [0.0, 0.1, 1.0, 5.5],
            [10.0, 32.5, 65.5, 100.0],
        ], dtype=np.float32)

        result = decode_depth_image(depth32, encoding="32fc1")
        np.testing.assert_array_equal(result, depth32)
        assert result.dtype == np.float32


# ============================================================================
# Contiguous array guarantees
# ============================================================================

class TestContiguousArrays:
    """Conversion results must be C-contiguous for downstream compatibility."""

    def test_rgb_decode_is_contiguous(self):
        """decode_color_image result is C-contiguous (required by some codecs)."""
        # Create non-contiguous input
        large = np.zeros((100, 100, 6), dtype=np.uint8)
        bgr = large[:, :, :3]  # Sliced, not contiguous
        assert not bgr.flags['C_CONTIGUOUS']

        result = decode_color_image(bgr, encoding="bgr8")
        assert result.flags['C_CONTIGUOUS'], "Result must be C-contiguous"


# ============================================================================
# Edge cases and special values
# ============================================================================

class TestSpecialValues:
    """Special values (NaN, inf, zero, boundaries) are handled correctly."""

    def test_depth_to_uint16_nan_handling(self):
        """NaN in float32 depth clips to 0mm in uint16."""
        depth_m = np.array([[np.nan, 1.0, 2.0]], dtype=np.float32)
        result = depth_to_uint16_png(depth_m, depth_scale=1000)

        assert result[0, 0] == 0  # NaN -> 0mm
        assert result[0, 1] == 1000  # 1.0 m -> 1000 mm
        assert result[0, 2] == 2000  # 2.0 m -> 2000 mm

    def test_depth_to_uint16_inf_handling(self):
        """Inf in float32 depth is converted to 0mm (via nan_to_num, posinf=0.0)."""
        depth_m = np.array([[np.inf, 1.0, -np.inf]], dtype=np.float32)
        result = depth_to_uint16_png(depth_m, depth_scale=1000)

        # depth_to_uint16_png uses np.nan_to_num with posinf=0.0, so both +inf and -inf -> 0
        assert result[0, 0] == 0  # +inf -> 0 (via nan_to_num)
        assert result[0, 1] == 1000
        assert result[0, 2] == 0  # -inf -> 0

    def test_depth_to_uint16_negative_handling(self):
        """Negative depth (physically invalid) clips to 0mm."""
        depth_m = np.array([[-1.0, 0.0, 0.5]], dtype=np.float32)
        result = depth_to_uint16_png(depth_m, depth_scale=1000)

        assert result[0, 0] == 0  # -1.0 -> 0mm
        assert result[0, 1] == 0  # 0.0 -> 0mm
        assert result[0, 2] == 500  # 0.5 m -> 500 mm


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
