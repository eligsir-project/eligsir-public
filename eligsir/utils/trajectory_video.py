"""Pure, torch-free helpers for comparison-report trajectory-video
resolution and pacing (issue #397, "FHD+ naturally paced train/test
trajectory videos"; design doc:
docs/architecture/2026-08-13-comparison-trajectory-video-quality.md).

Split out of ``utils/comparison_report.py`` (which imports torch at module
scope) for the same reason ``utils/baseline_comparison.py`` was split out
(issue #423): keeps this logic importable/testable in the CPU-only smoke
image. Nothing here touches a tensor -- it is plain arithmetic over
width/height ints and float timestamps. ``comparison_report`` imports and
uses these functions directly (not merely re-exporting them).

Two independent problems, two independent fixes:

1. **Resolution** -- ``trajectory.mp4``/``train_depth_timeseries.mp4`` were
   encoded at whatever resolution the first rendered frame happened to be
   (frequently well below FHD for low-res streaming camera profiles).
   ``scale_dims_for_video`` computes an aspect-preserving target canvas
   that is >=``min_width`` (upscale a too-small source) and <=``max_width``
   (safety cap so a very high-resolution source doesn't runaway the report
   step's CPU encode time or output file size). This is a **post-render,
   presentation-only upscale** -- it does not make the Gaussian rasterizer
   render more detail, only resizes the already-rendered frame. See the
   design doc's distinction between "true higher-resolution render" and
   "post-render upscale" -- the former would need a GPU-side change to the
   report-camera intrinsics and is out of scope here.

2. **Pacing** -- ``comparison_report.py`` previously ran EVERY selected
   camera list through ``_order_cams_for_trajectory`` (spatial
   nearest-neighbor reorder) + ``_pace_cams_for_trajectory`` (spatial-gap
   interpolation) unconditionally, even for datasets with real, trustworthy
   per-frame timestamps (TUM, Orbbec, rosbag replay). Both of those
   synthesize interpolated in-between camera objects that carry no
   ``_streaming_timestamp`` attribute, which made the (also pre-existing)
   ``_infer_realtime_fps`` immediately fall back to its ``default_fps``
   (30) the moment ANY interpolated frame was present -- i.e. almost
   always, since real capture spacing is irregular. The result: a
   timestamped sequence with a real, e.g., 84s capture span got encoded at
   a flat 30fps with however many frames survived subsampling, with no
   relationship to the real 84s duration -- exactly the "video is
   perceived as 2-3x too fast" symptom the motivating issue reports.

   The fix is to decide ordering/pacing strategy BEFORE reordering:
   - real timestamps on every selected camera -> keep source/timestamp
     order (never spatially reorder a sequential capture -- reordering is
     only correct for spatially-unordered sources like a RealityScan UUID
     dump), render that same bounded set of cameras (no more, no fewer --
     every real camera is rendered exactly once, matching
     ``_pace_cams_for_trajectory``'s "never dropped" contract), and PACE
     via ``distribute_frame_repeats``: repeat each rendered frame on
     ENCODE for as many output frames as its real time slice covers, at a
     fixed conventional presentation fps (e.g. 30). This reproduces the
     real capture duration exactly -- without rendering more frames than
     the existing render-count budget, since repeated identical frames are
     near-free for libx264 to encode (unlike unique renders). An earlier
     version of this fix instead inferred fps from the median cadence of
     the (already sparse, subsampled) selected cameras -- rejected because
     it degenerates to an unusable ~1fps slideshow on exactly this
     scenario (120 cams over an 84s span); see
     ``compute_trajectory_timing``'s docstring for why its ``fps`` output
     must never be used to drive playback rate.
   - no (or partial/unparseable) timestamps -> fall back to the existing
     spatial reorder + spatial-gap pacing, unchanged.

3. **Motion smoothness within a repeated slice** -- see
   ``compute_interpolation_render_counts``/``split_repeats_evenly`` further
   down for the follow-up that turns each slice's repeated-hold frames into
   an interpolated sweep instead of a static copy, without changing total
   duration/frame count.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

# "FHD-class" per the design doc's resolution policy.
DEFAULT_MIN_VIDEO_WIDTH = 1920
# Safety cap: a very high-resolution capture (e.g. a 4K RealityScan photo
# set) should not silently balloon report-step encode time/output size.
DEFAULT_MAX_VIDEO_WIDTH = 3840
DEFAULT_FPS = 30
DEFAULT_MAX_FPS = 60
# Ceiling on ENCODED (rendered + repeated) frames -- see
# distribute_frame_repeats. Repeated frames are cheap to encode (near-zero
# residual for libx264 between identical consecutive frames), so this can
# be generous without materially slowing the report step; it exists so a
# very long capture (tens of minutes) doesn't emit an unboundedly long
# output file. 9000 frames = 5 minutes at 30fps.
DEFAULT_MAX_ENCODED_FRAMES = 9000


def scale_dims_for_video(
    width: int,
    height: int,
    min_width: Optional[int] = DEFAULT_MIN_VIDEO_WIDTH,
    max_width: Optional[int] = DEFAULT_MAX_VIDEO_WIDTH,
) -> Tuple[int, int]:
    """Return ``(out_w, out_h)`` for encoding ``width x height`` at
    >=``min_width`` while preserving aspect ratio exactly -- never
    distorting a 4:3 source into 16:9.

    ``min_width``/``max_width`` of ``None`` or ``<= 0`` disables that side
    of the policy (pass both ``None`` to fully disable -- output equals
    input, even-cropped). Both output dimensions are always forced even
    (required by yuv420p's 2x2 chroma subsampling).
    """
    width = int(width)
    height = int(height)
    if width <= 0 or height <= 0:
        return width, height

    scale = 1.0
    if max_width and max_width > 0 and width > max_width:
        scale = max_width / width
    elif min_width and min_width > 0 and width < min_width:
        scale = min_width / width

    out_w = int(round(width * scale))
    out_h = int(round(height * scale))
    out_w -= out_w % 2
    out_h -= out_h % 2
    return max(2, out_w), max(2, out_h)


def is_upscaled(width: int, height: int, out_w: int, out_h: int) -> bool:
    """True iff ``scale_dims_for_video`` grew the frame (presentation-only
    upscale, not added reconstruction detail -- see module docstring)."""
    return out_w > int(width) or out_h > int(height)


def cams_have_usable_timestamps(timestamps: Sequence[Optional[float]]) -> bool:
    """True iff every entry is present and parses as a finite float, and
    there are at least 2 of them -- the minimum needed to infer a cadence
    or a duration. A single missing/unparseable timestamp anywhere in the
    selected camera list disqualifies the whole set from timestamp-based
    ordering/timing (falls back to the spatial-pacing path), matching the
    existing conservative behavior of the pre-existing
    ``_infer_realtime_fps``.
    """
    if len(timestamps) < 2:
        return False
    for t in timestamps:
        if t is None:
            return False
        try:
            tf = float(t)
        except (TypeError, ValueError):
            return False
        if tf != tf or tf in (float("inf"), float("-inf")):  # NaN/inf
            return False
    return True


def compute_trajectory_timing(
    timestamps: Sequence[float],
    default_fps: int = DEFAULT_FPS,
    max_fps: int = DEFAULT_MAX_FPS,
) -> Dict[str, object]:
    """Infer ``{"fps", "duration_s", "timing_source"}`` from a real camera
    timestamp sequence (assumed already validated by
    ``cams_have_usable_timestamps``).

    ``duration_s`` is the real span ``max(timestamps) - min(timestamps)``,
    independent of how many cameras/timestamps were sampled -- this is the
    field that matters, and is what ``distribute_frame_repeats`` paces the
    encoded video's duration against.

    ``fps`` is the median inter-frame cadence among the ORIGINAL
    (pre-subsample) timestamps, clamped to ``[1, max_fps]``. **This is
    informational metadata only (the source capture's own cadence) -- it
    must NEVER be used to pick the video's encode/playback fps.** Doing so
    was tried and rejected: on a sparse, already-subsampled camera list
    (e.g. 120 cams over an 84s span, this project's real streaming-report
    trajectory-video budget) the median gap between SELECTED cameras is
    far larger than the source's true per-frame cadence, so this field
    degenerates to ~1fps -- an unusable slideshow, not a fix. The
    conventional presentation fps (``mp4_fps``, e.g. 30) is used instead;
    see the module docstring's "Pacing" section.
    """
    ts = sorted(float(t) for t in timestamps)
    span = ts[-1] - ts[0]
    if span <= 1e-6:
        return {"fps": int(default_fps), "duration_s": None, "timing_source": "default_fps"}
    deltas = sorted(b - a for a, b in zip(ts, ts[1:]) if (b - a) > 1e-6)
    if not deltas:
        return {"fps": int(default_fps), "duration_s": None, "timing_source": "default_fps"}
    n = len(deltas)
    median = deltas[n // 2] if n % 2 == 1 else 0.5 * (deltas[n // 2 - 1] + deltas[n // 2])
    fps = max(1, min(int(max_fps), int(round(1.0 / median))))
    return {"fps": fps, "duration_s": span, "timing_source": "camera_timestamp"}


def distribute_frame_repeats(
    sample_times: Sequence[float],
    fps: float,
    max_encoded_frames: Optional[int] = None,
) -> List[int]:
    """How many times to repeat each of ``len(sample_times)`` RENDERED
    frames on encode so the resulting video reproduces the real duration
    (``sample_times[-1] - sample_times[0]``) at ``fps``, without rendering
    any more frames than were already sampled.

    This is the mechanism that actually resolves the duration/fps/render
    -cost three-way tension (see the module docstring): render a bounded
    number of unique frames (the existing ``mp4_max_frames``/
    ``trajectory_max_frames`` budget -- no change to render cost), then
    stretch each one over its real time slice by REPEATING it on encode.
    Repeated frames are near-free for libx264 (near-zero residual between
    identical consecutive frames), unlike unique renders, which is what
    makes this cheap where naively rendering ``duration_s * fps`` unique
    frames would not be.

    Every sampled frame gets at least 1 repeat -- real renders are never
    dropped, mirroring ``_pace_cams_for_trajectory``'s "real captured
    cameras are never dropped" contract. Each frame's natural repeat count
    is its gap to the NEXT sample (last frame reuses the previous gap, so
    it is not collapsed to a single tick), rounded to whole output frames
    at ``fps``. If the natural total would exceed ``max_encoded_frames``
    (a separate ceiling from the render-count budget -- guards a very long
    capture from emitting an excessively long/huge output file), every
    count is scaled down proportionally, preserving each frame's relative
    share of screen time; the resulting video is then correspondingly
    faster than real time (see ``compute_playback_speed``).
    """
    n = len(sample_times)
    if n == 0:
        return []
    if n == 1:
        return [max(1, int(round(fps)))]
    gaps = [sample_times[i + 1] - sample_times[i] for i in range(n - 1)]
    gaps.append(gaps[-1] if gaps[-1] > 0 else (1.0 / fps if fps > 0 else 1.0))
    raw = [max(1, int(round(g * fps))) for g in gaps]
    total = sum(raw)
    if max_encoded_frames and max_encoded_frames > 0 and total > max_encoded_frames:
        scale = max_encoded_frames / total
        raw = [max(1, int(round(r * scale))) for r in raw]
    return raw


def compute_playback_speed(source_duration_s: Optional[float], video_duration_s: Optional[float]) -> Optional[float]:
    """How much faster than real time the encoded video plays, i.e.
    ``source_duration_s / video_duration_s``.

    Returns ``None`` when either duration is missing/non-positive (nothing
    meaningful to report). A value of ``1.0`` means real-time playback; a
    value of e.g. ``21.2`` means ``distribute_frame_repeats``'s
    ``max_encoded_frames`` ceiling bound the encoded video to a fraction
    of the real capture span, and it plays back 21.2x faster than real
    time -- see the module docstring's discussion of the duration/fps/cost
    three-way tradeoff this exists to make honest instead of silently
    wrong.
    """
    if not source_duration_s or not video_duration_s or video_duration_s <= 0:
        return None
    return round(source_duration_s / video_duration_s, 3)


# ---------------------------------------------------------------------------
# 3. Motion smoothness -- interpolating the REPEAT slots (issue #397 follow-up,
#    "trajectory.mp4 looks chonky/stuttery at 30fps")
#
# ``distribute_frame_repeats`` above fixes DURATION (the video reproduces the
# real capture span) by repeating each unique render for as many output
# frames as its time slice covers. Repeats are pixel-identical, so within a
# slice the camera visibly *holds still* for up to dozens of frames, then
# jump-cuts to the next real render -- correct duration, but stuttery motion.
#
# The fix here is presentation-only and does not touch duration math at all:
# instead of repeating slice ``i``'s rendered frame verbatim, render a few
# EXTRA poses interpolated between camera ``i`` and camera ``i + 1``
# (SLERP rotation + lerp translation -- see ``_interpolate_camera`` in
# ``utils/comparison_report.py``, already used by the spatial-pacing path's
# ``_pace_cams_for_trajectory``) and spread the slice's existing repeat
# budget across them instead of one pose. The TOTAL number of encoded frames
# for the slice -- and therefore the video's total duration -- is completely
# unchanged; only which pose backs each of those frames changes, from "the
# same pixels N times" to "N poses smoothly swept between the two real
# cameras".
#
# This deliberately reuses the ALREADY-REJECTED old attempt's mechanism
# (interpolated poses) in a new, narrower place. The old attempt was wrong
# because it interpolated BEFORE deciding pacing strategy, so the
# interpolated cameras leaked into ``_infer_realtime_fps``'s real-timestamp
# detection (no ``_streaming_timestamp`` attribute -> false "not a real
# capture" -> wrong fps). Here, interpolation happens strictly AFTER
# ``distribute_frame_repeats`` has already computed real duration/fps from
# the ORIGINAL, un-interpolated timestamps -- the synthetic poses exist only
# as extra items handed to the renderer for ``trajectory.mp4`` and are never
# seen by any timing/metrics code path.
#
# Cost is bounded by ``max_extra_renders`` (a hard ceiling on how many
# ADDITIONAL unique renders this feature may add across the whole video,
# mirroring ``DEFAULT_MAX_ENCODED_FRAMES``'s role for encoded-frame count)
# and ``max_per_slice`` (a per-slice ceiling, so one huge repeat count from a
# long real gap can't alone consume the whole budget on a single slice).
# ---------------------------------------------------------------------------

# Default cap on the number of EXTRA (beyond the already-rendered real
# cameras) unique renders this feature may add to one trajectory.mp4.
# ``streaming_report_trajectory_max_frames`` (real renders) defaults to 120;
# capping extra renders at 300 means a worst case of roughly 3.5x the
# baseline render count for this one video -- a real but bounded and
# one-time cost at report-generation time (not training time), well short
# of rendering ``DEFAULT_MAX_ENCODED_FRAMES`` (9000) unique frames, which
# would be the fully-naive "just render every output frame" cost this
# exists to avoid.
#
# IMPORTANT: an interpolated render costs exactly as much as a real one --
# in particular, ``trajectory.mp4`` is ALSO the video Full SSAA
# (``--presentation_supersample_factor``, ``utils/presentation_ssaa.py``)
# applies to, and that factor defaults to 8 (64x the pixels per frame). This
# constant is deliberately sized as if SSAA were off; see
# ``presentation_trajectory_interpolate_kwargs_from_args`` below, which
# scales the EFFECTIVE cap down by the active SSAA factor so this feature's
# ADDED wall-clock cost stays roughly constant regardless of the SSAA
# setting, instead of multiplying an already-64x-inflated per-frame cost by
# up to this many extra renders.
DEFAULT_MAX_INTERPOLATION_RENDERS = 300
# Default cap on interpolated renders added to any single time slice (i.e.
# between one consecutive pair of real cameras), independent of the global
# cap above. Matches the order of magnitude of the existing spatial-pacing
# path's own per-gap cap (``_pace_cams_for_trajectory``'s ``max_steps=12``).
DEFAULT_MAX_INTERPOLATION_PER_SLICE = 8


def split_repeats_evenly(total: int, parts: int) -> List[int]:
    """Split ``total`` into ``parts`` non-negative integers that sum to
    exactly ``total``, as evenly as possible (each part is ``total //
    parts`` or one more, with the remainder handed to the first parts).

    Deterministic and order-stable, used to divide one time slice's
    existing encoded-frame budget (``distribute_frame_repeats``'s per-slice
    repeat count) across the slice's real camera plus however many
    interpolated poses were inserted, so the total is provably unchanged by
    ``compute_interpolation_render_counts``/interpolation being on or off.
    """
    parts = int(parts)
    total = int(total)
    if parts <= 0:
        return []
    if parts == 1:
        return [total]
    base, rem = divmod(total, parts)
    return [base + 1 if i < rem else base for i in range(parts)]


def compute_interpolation_render_counts(
    frame_repeats: Sequence[int],
    max_extra_renders: Optional[int] = DEFAULT_MAX_INTERPOLATION_RENDERS,
    max_per_slice: Optional[int] = DEFAULT_MAX_INTERPOLATION_PER_SLICE,
) -> List[int]:
    """How many EXTRA interpolated poses to render within each of
    ``len(frame_repeats)`` time slices, so the repeated-hold slots in that
    slice become a smooth sweep instead of a static copy.

    ``frame_repeats[i]`` is the number of output frames slice ``i`` already
    occupies (from ``distribute_frame_repeats``). Slice ``i``'s natural
    interpolation count is ``frame_repeats[i] - 1`` (every extra copy beyond
    the first becomes its own unique interpolated render towards camera
    ``i + 1``); the LAST slice always gets ``0`` since there is no "next"
    camera to interpolate towards, so a trailing hold still repeats
    verbatim, matching pre-existing behavior for that one slice.

    ``max_per_slice`` caps a single very-long real time gap from alone
    consuming the whole render budget; the global ``max_extra_renders``
    then scales every slice down proportionally (floor, so the result never
    exceeds the cap) if the per-slice-capped total still exceeds it --
    mirroring ``distribute_frame_repeats``'s own proportional-scaling
    fallback. Passing ``None`` (or a value ``<= 0``) to either disables that
    side of the cap.

    Returns one entry per input ``frame_repeats`` entry;
    ``compute_interpolation_render_counts(r)[i] + 1`` is always
    ``<= frame_repeats[i]`` for every ``i`` -- the invariant
    ``split_repeats_evenly`` (called by the caller) needs each interpolated
    pose to get at least one real encoded frame.
    """
    n = len(frame_repeats)
    if n == 0:
        return []
    raw = [max(0, int(r) - 1) for r in frame_repeats]
    raw[-1] = 0  # no "next" camera for the final slice to interpolate toward
    if max_per_slice is not None and int(max_per_slice) >= 0:
        cap = int(max_per_slice)
        raw = [min(r, cap) for r in raw]
    total = sum(raw)
    if max_extra_renders is not None and int(max_extra_renders) >= 0 and total > int(max_extra_renders):
        budget = int(max_extra_renders)
        if total > 0:
            scale = budget / total
            raw = [int(r * scale) for r in raw]  # floor: never exceeds budget
    return raw


def _scale_extra_renders_for_ssaa(max_extra_renders: Optional[int], ssaa_factor: int,
                                  min_extra_renders: int = 20) -> Optional[int]:
    """Shrink ``max_extra_renders`` by the active SSAA linear factor, floored at
    ``min_extra_renders``.

    ``trajectory.mp4`` is also the video Full SSAA (``utils/presentation_ssaa.py``)
    applies to -- every unique render in it, real or interpolated, costs
    ``ssaa_factor ** 2`` more pixels when SSAA is on. Without this scaling,
    ``max_extra_renders`` would multiply an ALREADY ``ssaa_factor**2``-inflated
    per-frame cost by up to its full value; dividing by ``ssaa_factor`` instead
    keeps this feature's ADDED wall-clock cost roughly constant whether SSAA is
    on or off, at the price of allowing fewer unique interpolated poses when SSAA
    is aggressive (default factor 8). ``min_extra_renders`` keeps that reduction
    from collapsing the feature to a no-op at high factors. ``None``/``<= 0``
    passes through unchanged (uncapped stays uncapped).
    """
    if max_extra_renders is None or int(max_extra_renders) < 0:
        return max_extra_renders
    factor = max(1, int(ssaa_factor))
    if factor <= 1:
        return int(max_extra_renders)
    return max(int(min_extra_renders), int(max_extra_renders) // factor)


def presentation_trajectory_interpolate_kwargs_from_args(args) -> Dict[str, object]:
    """``**``-splat this at every ``write_post_training_report(...)`` call site to wire
    the ``--presentation_trajectory_interpolate``/``--presentation_trajectory_interpolate_max_extra_renders``/
    ``--presentation_trajectory_interpolate_max_per_slice`` CLI flags (``arguments/__init__.py``)
    through to ``write_post_training_report``'s identically-named kwargs. Missing attributes
    (an ``args`` namespace built before this feature existed) fall back to this module's own
    defaults, matching ``presentation_ssaa_kwargs_from_args``/``git_provenance_from_args`` and
    every other ``*_from_args`` helper in this codebase. Deliberately torch-free/pure (unlike
    ``presentation_ssaa``'s helper) so it stays importable from the CPU-only smoke image.

    Reads ``--presentation_supersample_enabled``/``--presentation_supersample_factor`` (issue
    #1323) and runs the resolved extra-render cap through ``_scale_extra_renders_for_ssaa`` --
    see that function's docstring for why. This is the only cross-feature coupling; a
    caller that never enables SSAA gets the cap unchanged.

    The ``presentation_supersample_factor`` fallback literal (``8``) matches
    ``arguments/__init__.py``'s real current default -- see
    ``tests/test_getattr_fallback_freshness.py``, which fails a PR that lets a
    fallback literal drift from the real parser default.
    """
    base_cap = getattr(
        args, "presentation_trajectory_interpolate_max_extra_renders",
        DEFAULT_MAX_INTERPOLATION_RENDERS)
    ssaa_enabled = bool(getattr(args, "presentation_supersample_enabled", True))
    ssaa_factor = int(getattr(args, "presentation_supersample_factor", 8)) if ssaa_enabled else 1
    return {
        "presentation_trajectory_interpolate": bool(
            getattr(args, "presentation_trajectory_interpolate", True)),
        "presentation_trajectory_interpolate_max_extra_renders": _scale_extra_renders_for_ssaa(
            base_cap, ssaa_factor),
        "presentation_trajectory_interpolate_max_per_slice": getattr(
            args, "presentation_trajectory_interpolate_max_per_slice",
            DEFAULT_MAX_INTERPOLATION_PER_SLICE),
    }
