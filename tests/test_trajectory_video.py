"""Tests for utils/trajectory_video.py (issue #397, torch-free pure
functions driving comparison-report trajectory-video resolution/pacing).

Covers the design doc's (docs/architecture/2026-08-13-comparison-
trajectory-video-quality.md) acceptance criteria at the pure-function
level, since the actual video render+encode needs a GPU (for
render_camera_path) or a real ffmpeg binary (for the write path) -- see
tests/test_trajectory_video_encode.py for the ffmpeg-dependent, still
CPU-only, real-encode cost check.
"""
from __future__ import annotations

import pytest

from eligsir.utils.trajectory_video import (
    DEFAULT_MAX_INTERPOLATION_PER_SLICE,
    DEFAULT_MAX_INTERPOLATION_RENDERS,
    DEFAULT_MAX_VIDEO_WIDTH,
    DEFAULT_MIN_VIDEO_WIDTH,
    cams_have_usable_timestamps,
    compute_interpolation_render_counts,
    compute_playback_speed,
    compute_trajectory_timing,
    distribute_frame_repeats,
    is_upscaled,
    presentation_trajectory_interpolate_kwargs_from_args,
    scale_dims_for_video,
    split_repeats_evenly,
)
from eligsir.utils.trajectory_video import _scale_extra_renders_for_ssaa  # noqa: E402 (private, kept separate)


# --- scale_dims_for_video ---------------------------------------------------

def test_scale_dims_upscales_below_min_width_preserving_aspect():
    # TUM 4:3 source well below FHD.
    w, h = scale_dims_for_video(640, 480)
    assert w >= DEFAULT_MIN_VIDEO_WIDTH
    # exact 4:3 aspect preserved (not stretched to 16:9)
    assert abs((w / h) - (640 / 480)) < 1e-6
    assert w == 1920 and h == 1440


def test_scale_dims_leaves_already_fhd_or_higher_unchanged():
    w, h = scale_dims_for_video(2560, 1440)
    assert (w, h) == (2560, 1440)


def test_scale_dims_caps_very_high_resolution():
    w, h = scale_dims_for_video(7680, 4320, max_width=3840)
    assert w == 3840
    assert abs((w / h) - (7680 / 4320)) < 1e-6


def test_scale_dims_always_even():
    w, h = scale_dims_for_video(641, 481)
    assert w % 2 == 0 and h % 2 == 0


def test_scale_dims_disabled_when_both_none():
    w, h = scale_dims_for_video(640, 480, min_width=None, max_width=None)
    assert (w, h) == (640, 480)


def test_scale_dims_zero_size_passthrough():
    assert scale_dims_for_video(0, 0) == (0, 0)


def test_is_upscaled():
    assert is_upscaled(640, 480, 1920, 1440) is True
    assert is_upscaled(1920, 1440, 1920, 1440) is False
    assert is_upscaled(3840, 2160, 1920, 1080) is False  # downscale, not "upscaled"


# --- cams_have_usable_timestamps --------------------------------------------

def test_usable_timestamps_true_for_clean_sequence():
    assert cams_have_usable_timestamps([0.0, 0.1, 0.2, 0.3]) is True


def test_usable_timestamps_false_on_any_none():
    assert cams_have_usable_timestamps([0.0, None, 0.2]) is False


def test_usable_timestamps_false_on_single_camera():
    assert cams_have_usable_timestamps([0.0]) is False


def test_usable_timestamps_false_on_nan():
    assert cams_have_usable_timestamps([0.0, float("nan"), 0.2]) is False


def test_usable_timestamps_false_on_unparseable():
    assert cams_have_usable_timestamps([0.0, "not-a-number", 0.2]) is False


# --- compute_trajectory_timing ----------------------------------------------

def test_timing_recovers_real_span_and_median_cadence():
    # 300 evenly-spaced samples across an 84s capture -> ~3.56fps median cadence.
    n = 300
    span = 84.0
    ts = [i * span / (n - 1) for i in range(n)]
    result = compute_trajectory_timing(ts)
    assert result["timing_source"] == "camera_timestamp"
    assert abs(result["duration_s"] - span) < 1e-6
    assert result["fps"] == round((n - 1) / span)


def test_timing_regression_2525_to_300_subsample_does_not_speed_up_video():
    """Direct regression for the motivating issue #397 artifact: an 84s
    fr3-like capture subsampled from 2525 to 300 frames must still report
    an ~84s duration, not a duration implied by 300 frames at a flat 30fps
    default (10s -- the "2-3x too fast" bug)."""
    n_full = 2525
    span = 84.0
    full_ts = [i * span / (n_full - 1) for i in range(n_full)]
    # Evenly-spaced subsample of 300, preserving order (mirrors
    # comparison_report._representative_cams' np.linspace selection).
    idxs = sorted({round(i * (n_full - 1) / 299) for i in range(300)})
    sub_ts = [full_ts[i] for i in idxs]
    result = compute_trajectory_timing(sub_ts)
    assert result["duration_s"] is not None
    assert result["duration_s"] > 60.0  # nowhere near the old flat-30fps ~10s artifact
    naive_flat_30fps_duration = len(sub_ts) / 30.0
    assert result["duration_s"] > 2.0 * naive_flat_30fps_duration


def test_timing_falls_back_to_default_fps_on_degenerate_span():
    result = compute_trajectory_timing([1.0, 1.0, 1.0])
    assert result["timing_source"] == "default_fps"
    assert result["fps"] == 30
    assert result["duration_s"] is None


def test_timing_clamps_fps_to_max():
    # 1ms spacing -> 1000fps naive, clamped to max_fps.
    ts = [0.0, 0.001, 0.002, 0.003]
    result = compute_trajectory_timing(ts, max_fps=60)
    assert result["fps"] == 60


def test_timing_custom_default_fps_used_when_unusable():
    result = compute_trajectory_timing([5.0, 5.0], default_fps=24)
    assert result["fps"] == 24


# --- distribute_frame_repeats ------------------------------------------------

def test_distribute_frame_repeats_never_drops_a_frame():
    ts = [0.0, 1.0, 3.0, 3.5, 10.0]
    repeats = distribute_frame_repeats(ts, fps=10)
    assert len(repeats) == len(ts)
    assert all(r >= 1 for r in repeats)


def test_distribute_frame_repeats_uniform_spacing_gives_uniform_repeats():
    ts = [0.0, 1.0, 2.0, 3.0, 4.0]  # 1s gaps
    repeats = distribute_frame_repeats(ts, fps=10)
    assert repeats == [10, 10, 10, 10, 10]  # last frame reuses previous gap


def test_distribute_frame_repeats_larger_gap_gets_more_repeats():
    ts = [0.0, 1.0, 11.0]  # first gap 1s, second gap 10s
    repeats = distribute_frame_repeats(ts, fps=10)
    assert repeats[1] > repeats[0]


def test_distribute_frame_repeats_single_camera():
    repeats = distribute_frame_repeats([5.0], fps=30)
    assert repeats == [30]


def test_distribute_frame_repeats_empty():
    assert distribute_frame_repeats([], fps=30) == []


def test_distribute_frame_repeats_scales_down_when_capped():
    ts = [i * 1.0 for i in range(100)]  # 100 cams, 1s apart, 99s span
    repeats = distribute_frame_repeats(ts, fps=30, max_encoded_frames=300)
    assert len(repeats) == 100
    assert all(r >= 1 for r in repeats)
    assert sum(repeats) <= 300


def test_distribute_frame_repeats_uncapped_when_under_ceiling():
    ts = [0.0, 1.0, 2.0]
    repeats = distribute_frame_repeats(ts, fps=10, max_encoded_frames=10_000)
    assert sum(repeats) == 30  # not scaled down -- well under the ceiling


# --- compute_playback_speed --------------------------------------------------

def test_playback_speed_real_time_is_one():
    assert compute_playback_speed(84.0, 84.0) == 1.0


def test_playback_speed_sped_up_when_render_cap_binds():
    assert compute_playback_speed(84.0, 4.0) == pytest.approx(21.0, rel=1e-3)


def test_playback_speed_none_when_either_duration_missing():
    assert compute_playback_speed(None, 4.0) is None
    assert compute_playback_speed(84.0, None) is None
    assert compute_playback_speed(84.0, 0.0) is None


# --- Regression: the render-budget-capped resample must not become a
# slideshow, and must stay within the pre-existing render-cost budget -----

def test_render_budget_repeat_scheme_reproduces_real_duration_at_conventional_fps():
    """Direct regression for the fr3-like motivating scenario (120 real
    cams evenly spread over an 84s span). NOTE: deriving the video fps
    from the MEDIAN CADENCE OF THE SUBSAMPLED cams (an earlier version of
    this fix) degenerates to fps=1 here -- a slideshow, not a fix; the
    shipped design instead (a) keeps fps at a fixed conventional rate and
    (b) renders exactly the existing budget of unique frames (no render
    -cost regression) while (c) REPEATING each rendered frame on ENCODE
    for as many output frames as its real time slice covers, via
    distribute_frame_repeats -- reproducing the real 84s duration exactly,
    without rendering more than 120 frames."""
    n_cams = 120
    span = 84.0
    ts = [i * span / (n_cams - 1) for i in range(n_cams)]
    conventional_fps = 30

    repeats = distribute_frame_repeats(ts, conventional_fps)
    assert len(repeats) == n_cams  # every real cam still rendered exactly once
    assert all(r >= 1 for r in repeats)  # no real cam ever dropped

    video_duration_s = sum(repeats) / conventional_fps
    assert video_duration_s == pytest.approx(span, abs=1.0 / conventional_fps * n_cams)
    speed = compute_playback_speed(span, video_duration_s)
    assert speed == pytest.approx(1.0, abs=0.05)  # real-time, not sped up -- the cap didn't bind


def test_distribute_frame_repeats_respects_max_encoded_frames_and_reports_speed():
    """When a real span would demand more encoded frames than the ceiling
    allows (e.g. a very long capture), repeats scale down proportionally
    (never below 1) and the resulting video is honestly faster than real
    time, not silently truncated."""
    n_cams = 50
    span = 1200.0  # 20 minutes
    ts = [i * span / (n_cams - 1) for i in range(n_cams)]
    fps = 30
    max_encoded = 900  # 30s ceiling

    repeats = distribute_frame_repeats(ts, fps, max_encoded_frames=max_encoded)
    assert len(repeats) == n_cams
    assert all(r >= 1 for r in repeats)
    assert sum(repeats) <= max_encoded

    video_duration_s = sum(repeats) / fps
    speed = compute_playback_speed(span, video_duration_s)
    assert speed is not None and speed > 1.0


# --- split_repeats_evenly ----------------------------------------------------

def test_split_repeats_evenly_sums_to_total():
    for total, parts in [(10, 3), (7, 7), (1, 1), (100, 8), (0, 4)]:
        out = split_repeats_evenly(total, parts)
        assert len(out) == parts
        assert sum(out) == total


def test_split_repeats_evenly_as_even_as_possible():
    out = split_repeats_evenly(10, 3)
    assert sorted(out) == [3, 3, 4]


def test_split_repeats_evenly_single_part_returns_whole_total():
    assert split_repeats_evenly(42, 1) == [42]


def test_split_repeats_evenly_zero_parts_returns_empty():
    assert split_repeats_evenly(10, 0) == []


# --- compute_interpolation_render_counts ------------------------------------

def test_interpolation_counts_last_slice_always_zero():
    """No 'next' camera to interpolate toward for the final slice -- it must
    keep the pre-existing repeated-hold behavior."""
    counts = compute_interpolation_render_counts([5, 5, 5], max_extra_renders=None, max_per_slice=None)
    assert counts[-1] == 0


def test_interpolation_counts_natural_is_repeats_minus_one():
    counts = compute_interpolation_render_counts([1, 4, 7, 3], max_extra_renders=None, max_per_slice=None)
    assert counts == [0, 3, 6, 0]  # last forced to 0 regardless of its own repeat count


def test_interpolation_counts_never_exceeds_repeats_minus_one():
    """Invariant _expand_cams_with_interpolation's split_repeats_evenly call
    depends on: count + 1 <= frame_repeats[i] for every i."""
    for max_per_slice in (None, 0, 1, 5, 100):
        for max_extra in (None, 0, 1, 10, 10_000):
            repeats = [1, 2, 3, 50, 1, 8]
            counts = compute_interpolation_render_counts(
                repeats, max_extra_renders=max_extra, max_per_slice=max_per_slice)
            assert len(counts) == len(repeats)
            for c, r in zip(counts, repeats):
                assert c + 1 <= r
                assert c >= 0


def test_interpolation_counts_per_slice_cap():
    counts = compute_interpolation_render_counts([1, 100, 1], max_extra_renders=None, max_per_slice=8)
    assert counts[1] == 8  # would naturally be 99, capped to 8


def test_interpolation_counts_global_cap_scales_down_proportionally_never_over_budget():
    repeats = [1, 50, 50, 1]
    counts = compute_interpolation_render_counts(repeats, max_extra_renders=20, max_per_slice=None)
    assert sum(counts) <= 20


def test_interpolation_counts_empty_input():
    assert compute_interpolation_render_counts([]) == []


def test_interpolation_counts_disabled_caps_pass_none():
    # No caps at all -- every non-final slice gets its full repeats-1.
    counts = compute_interpolation_render_counts([10, 10, 10], max_extra_renders=None, max_per_slice=None)
    assert counts == [9, 9, 0]


# --- _scale_extra_renders_for_ssaa -------------------------------------------

def test_scale_extra_renders_ssaa_off_is_unscaled():
    assert _scale_extra_renders_for_ssaa(300, ssaa_factor=1) == 300


def test_scale_extra_renders_divides_by_factor():
    assert _scale_extra_renders_for_ssaa(300, ssaa_factor=8) == max(20, 300 // 8)


def test_scale_extra_renders_never_below_floor():
    assert _scale_extra_renders_for_ssaa(30, ssaa_factor=8, min_extra_renders=20) == 20


def test_scale_extra_renders_none_passthrough():
    assert _scale_extra_renders_for_ssaa(None, ssaa_factor=8) is None


def test_scale_extra_renders_uncapped_stays_uncapped():
    assert _scale_extra_renders_for_ssaa(-1, ssaa_factor=8) == -1


# --- presentation_trajectory_interpolate_kwargs_from_args -------------------

class _Args:
    pass


def test_interpolate_kwargs_from_args_defaults_when_missing():
    """A bare namespace (no presentation_supersample_* attrs either) falls back
    to this module's own bare-namespace default (factor=8, matching
    arguments/__init__.py's real current default) -- so the resolved cap is
    DEFAULT_MAX_INTERPOLATION_RENDERS divided by that factor, per
    _scale_extra_renders_for_ssaa."""
    kwargs = presentation_trajectory_interpolate_kwargs_from_args(_Args())
    assert kwargs["presentation_trajectory_interpolate"] is True
    assert kwargs["presentation_trajectory_interpolate_max_extra_renders"] == max(
        20, DEFAULT_MAX_INTERPOLATION_RENDERS // 8)
    assert kwargs["presentation_trajectory_interpolate_max_per_slice"] == DEFAULT_MAX_INTERPOLATION_PER_SLICE


def test_interpolate_kwargs_from_args_honors_explicit_values_when_ssaa_off():
    args = _Args()
    args.presentation_trajectory_interpolate = False
    args.presentation_trajectory_interpolate_max_extra_renders = 42
    args.presentation_trajectory_interpolate_max_per_slice = 3
    args.presentation_supersample_enabled = False
    kwargs = presentation_trajectory_interpolate_kwargs_from_args(args)
    assert kwargs == {
        "presentation_trajectory_interpolate": False,
        "presentation_trajectory_interpolate_max_extra_renders": 42,
        "presentation_trajectory_interpolate_max_per_slice": 3,
    }


def test_interpolate_kwargs_from_args_scales_cap_by_real_ssaa_factor():
    """The real arguments/__init__.py default is presentation_supersample_factor=8
    -- the resolved cap must be divided by 8, not left at the raw configured value,
    since trajectory.mp4 is also the SSAA video (see _scale_extra_renders_for_ssaa)."""
    args = _Args()
    args.presentation_trajectory_interpolate_max_extra_renders = 300
    args.presentation_supersample_enabled = True
    args.presentation_supersample_factor = 8
    kwargs = presentation_trajectory_interpolate_kwargs_from_args(args)
    assert kwargs["presentation_trajectory_interpolate_max_extra_renders"] == 37  # max(20, 300 // 8)
