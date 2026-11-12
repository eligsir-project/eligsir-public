from __future__ import annotations

import numpy as np

from eligsir.mapping.scheduler import (
    FrameQualityConfig,
    RunningBlurExposureStats,
    evaluate_frame_quality,
    normalized_blur_score,
)


def test_utils_frame_quality_reexports_package_quality():
    from eligsir.utils.frame_quality import evaluate_frame_quality as compat_evaluate_frame_quality

    assert compat_evaluate_frame_quality is evaluate_frame_quality


def test_frame_quality_rejects_clipped_image():
    rgb = np.ones((16, 16, 3), dtype=np.float32)
    depth = np.ones((16, 16), dtype=np.float32)
    q = evaluate_frame_quality(rgb, depth, config=FrameQualityConfig(min_blur_laplacian=0.0, max_clip_ratio=0.5))
    assert q.reject_reason == "exposure_clip"
    assert not q.accepted


def test_frame_quality_accepts_textured_depth_frame():
    yy, xx = np.mgrid[0:32, 0:32]
    rgb = (64 + ((xx // 2 + yy // 2) % 2) * 128).astype(np.uint8)
    rgb = np.stack([rgb, rgb, rgb], axis=-1)
    depth = np.ones((32, 32), dtype=np.float32)
    q = evaluate_frame_quality(rgb, depth, config=FrameQualityConfig(min_blur_laplacian=1.0))
    assert q.accepted
    assert q.depth_valid_ratio == 1.0
    assert 0.0 <= q.quality_weight <= 1.0


def _checkerboard(size=64, block=2, low=64, high=192):
    yy, xx = np.mgrid[0:size, 0:size]
    rgb = (low + ((xx // block + yy // block) % 2) * (high - low)).astype(np.uint8)
    return np.stack([rgb, rgb, rgb], axis=-1)


def _flat(size=64, value=128):
    return np.full((size, size, 3), value, dtype=np.uint8)


class TestRunningBlurExposureStats:
    def test_none_before_warmup(self):
        stats = RunningBlurExposureStats(warmup=5)
        for _ in range(4):
            stats.update(blur=200.0, luma=0.5)
        assert stats.blur_typical is None
        assert stats.luma_mean is None

    def test_available_after_warmup(self):
        stats = RunningBlurExposureStats(warmup=5)
        for _ in range(5):
            stats.update(blur=200.0, luma=0.5)
        assert stats.blur_typical is not None
        assert stats.blur_typical == 200.0  # constant input -> EMA converges exactly
        assert stats.luma_mean == 0.5


class TestAdaptiveQualityColdStartAndCleanScene:
    def test_cold_start_no_penalty_regardless_of_blur(self):
        """No running stats yet -> blur_quality_weight must be 1.0 even for a
        blurry (flat) frame -- never judge the first frames against nothing."""
        rgb = _flat()
        depth = np.ones((64, 64), dtype=np.float32)
        q = evaluate_frame_quality(
            rgb, depth,
            config=FrameQualityConfig(adaptive_quality=True, blur_floor=20.0),
            running_blur_typical=None,
        )
        assert q.blur_quality_weight == 1.0
        assert q.reject_reason != "blur"

    def test_clean_scene_self_disables(self):
        """Every frame clears the absolute floor by a wide margin (like the
        fr3 hand-check: min=87 >> floor=20) -> blur_quality_weight stays 1.0,
        the mechanism does not reject the bottom X% of an already-sharp
        trajectory."""
        sharp = _checkerboard()
        depth = np.ones((64, 64), dtype=np.float32)
        q_probe = evaluate_frame_quality(sharp, depth, config=FrameQualityConfig(adaptive_quality=True))
        running_typical = q_probe.blur_laplacian  # stand-in for a converged running EMA
        assert running_typical > 20.0 * 2  # sanity: this checkerboard is genuinely sharp
        q = evaluate_frame_quality(
            sharp, depth,
            config=FrameQualityConfig(adaptive_quality=True, blur_floor=20.0),
            running_blur_typical=running_typical,
        )
        assert q.blur_quality_weight == 1.0


class TestAdaptiveQualitySoftNotHard:
    def test_genuinely_blurry_frame_is_downweighted_not_rejected(self):
        """A flat (near-zero-Laplacian) frame scored against a sharp running
        typical must get a low blur_quality_weight but MUST NOT hard-reject
        -- soft downweight only, per the coverage-hole reasoning."""
        blurry = _flat()
        depth = np.ones((64, 64), dtype=np.float32)
        q = evaluate_frame_quality(
            blurry, depth,
            config=FrameQualityConfig(adaptive_quality=True, blur_floor=20.0, min_depth_valid_ratio=0.0),
            running_blur_typical=200.0,
        )
        assert q.blur_quality_weight < 0.2
        assert q.reject_reason != "blur"
        assert q.accepted  # never a hard veto for blur under adaptive_quality

    def test_exposure_far_from_running_mean_is_downweighted_not_rejected(self):
        rgb = _checkerboard(low=200, high=220)  # bright but not clipped
        depth = np.ones((64, 64), dtype=np.float32)
        q = evaluate_frame_quality(
            rgb, depth,
            config=FrameQualityConfig(adaptive_quality=True, exposure_tolerance=0.1, min_depth_valid_ratio=0.0),
            running_blur_typical=1.0,  # trivially satisfied, isolate exposure
            running_luma_mean=0.3,  # far darker than this frame's real luma
        )
        assert q.exposure_quality_weight < 1.0
        assert q.reject_reason != "exposure_clip"  # not clipped, just off-average
        assert q.accepted

    def test_legacy_path_unaffected_still_hard_rejects_blur(self):
        """adaptive_quality=False (the default) must keep the old hard
        min_blur_laplacian reject byte-for-byte -- this is what every
        existing config with quality_enabled=True (none currently) relies
        on if it ever gets turned on without adaptive_quality."""
        blurry = _flat()
        depth = np.ones((64, 64), dtype=np.float32)
        q = evaluate_frame_quality(
            blurry, depth,
            config=FrameQualityConfig(adaptive_quality=False, min_blur_laplacian=8.0, min_depth_valid_ratio=0.0),
        )
        assert q.reject_reason == "blur"
        assert not q.accepted


def test_normalized_blur_score_resolution_invariant_ordering():
    """A checkerboard resized 2x must still score as clearly sharper than a
    flat image at either scale -- the metric must not be so resolution-
    sensitive that a fixed absolute floor becomes meaningless across scenes
    shot at different native resolutions."""
    sharp_small = normalized_blur_score(_checkerboard(size=64), target_width=64)
    sharp_large = normalized_blur_score(_checkerboard(size=256, block=8), target_width=64)
    flat_small = normalized_blur_score(_flat(size=64), target_width=64)
    flat_large = normalized_blur_score(_flat(size=256), target_width=64)
    assert sharp_small > flat_small * 5
    assert sharp_large > flat_large * 5
    # Same content at 2 different native resolutions, normalized to the same
    # target width, should land in the same order of magnitude.
    assert 0.2 < (sharp_small / max(sharp_large, 1e-6)) < 5.0
