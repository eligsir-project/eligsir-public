"""Tests for the deterministic synthetic RGB-D source + fake SLAM sidecar
(``eligsir/io/sources/synthetic.py``), used by the Wave 3 live-pipeline
benchmark suite. Pure-Python/CPU, no GPU/torch/ROS2 dependency.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from eligsir.io.sources.synthetic import (
    FakeSlamSidecar,
    RejectionPolicy,
    SlamFailure,
    SyntheticRGBDFrameSource,
    SyntheticSourceConfig,
    frozen_artifact_hashes,
)


def _make_source(**overrides) -> SyntheticRGBDFrameSource:
    kwargs = dict(seed=42, fps=30.0, num_frames=20, realtime=False)
    kwargs.update(overrides)
    cfg = SyntheticSourceConfig(**kwargs)
    return SyntheticRGBDFrameSource(cfg)


# ---------------------------------------------------------------------------
# Determinism: same seed -> byte-identical content and timing across runs.
# ---------------------------------------------------------------------------

def test_deterministic_content_across_two_runs():
    src_a = _make_source()
    src_b = _make_source()

    frames_a = list(src_a)
    frames_b = list(src_b)

    assert len(frames_a) == len(frames_b) == 20
    for fa, fb in zip(frames_a, frames_b):
        assert fa.index == fb.index
        assert fa.timestamp == fb.timestamp
        assert np.array_equal(fa.rgb, fb.rgb)
        assert np.array_equal(fa.depth, fb.depth)
        assert np.array_equal(fa.c2w, fb.c2w)


def test_different_seeds_produce_different_content():
    src_a = _make_source()
    src_b = SyntheticRGBDFrameSource(SyntheticSourceConfig(seed=7, fps=30.0, num_frames=20, realtime=False))

    frame_a = next(iter(src_a))
    frame_b = next(iter(src_b))
    assert not np.array_equal(frame_a.rgb, frame_b.rgb)


def test_frozen_hashes_cover_frame_bytes_and_pose_stream_not_scenario_policy():
    first = frozen_artifact_hashes(seed=42, num_frames=4)
    same = frozen_artifact_hashes(seed=42, num_frames=4)
    changed = frozen_artifact_hashes(seed=43, num_frames=4)
    assert first == same
    assert first != changed
    assert set(first) == {"source_artifact_hash", "pose_hash"}


def test_successive_frames_within_a_run_are_distinct():
    src = _make_source()
    frames = list(src)
    assert not np.array_equal(frames[0].rgb, frames[1].rgb)
    assert not np.array_equal(frames[0].depth, frames[1].depth)


def test_timestamps_advance_at_configured_fps():
    src = _make_source(fps=30.0)
    frames = list(src)
    deltas = [b.timestamp - a.timestamp for a, b in zip(frames, frames[1:])]
    for d in deltas:
        assert d == pytest.approx(1.0 / 30.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Finite vs. long-running mode.
# ---------------------------------------------------------------------------

def test_finite_mode_stops_at_exactly_n_frames():
    src = _make_source(num_frames=17)
    frames = list(src)
    assert len(frames) == 17
    assert len(src) == 17


def test_long_running_mode_has_no_fixed_length():
    src = SyntheticRGBDFrameSource(SyntheticSourceConfig(seed=1, fps=30.0, num_frames=None, realtime=False))
    with pytest.raises(TypeError):
        len(src)
    # Confirm it actually keeps going past any "small" finite bound.
    it = iter(src)
    collected = [next(it) for _ in range(500)]
    assert len(collected) == 500
    assert collected[-1].index == 499


# ---------------------------------------------------------------------------
# Realtime pacing (kept short: a handful of frames at a fast fps).
# ---------------------------------------------------------------------------

def test_realtime_mode_paces_emission():
    cfg = SyntheticSourceConfig(seed=0, fps=50.0, num_frames=5, realtime=True)
    src = SyntheticRGBDFrameSource(cfg)
    t0 = time.monotonic()
    list(src)
    elapsed = time.monotonic() - t0
    # 5 frames at 50fps spans ~0.08s (frame 0 emitted immediately); allow slack.
    assert elapsed >= 0.06


# ---------------------------------------------------------------------------
# Burst mode.
# ---------------------------------------------------------------------------

def test_burst_mode_emits_burst_without_pacing_delay():
    cfg = SyntheticSourceConfig(seed=0, fps=5.0, num_frames=10, burst_size=8, realtime=True)
    src = SyntheticRGBDFrameSource(cfg)
    t0 = time.monotonic()
    frames = list(src)
    elapsed = time.monotonic() - t0
    assert len(frames) == 10
    # All 10 frames at 5fps steady-state would take ~1.8s; the first 8 are a
    # burst (no pacing), only frame index 8 waits for steady-state pacing, so
    # total elapsed should be well under the no-burst steady-state duration.
    assert elapsed < 1.0


def test_burst_mode_post_burst_pacing_matches_configured_period():
    """Regression guard: after the burst ends, real wall-clock inter-arrival
    intervals must resume at ~1/fps immediately, not run in a "catch-up"
    free-running mode for the first several post-burst frames (review Fix 5)."""
    fps = 20.0
    period = 1.0 / fps
    cfg = SyntheticSourceConfig(seed=0, fps=fps, num_frames=14, burst_size=8, realtime=True)
    src = SyntheticRGBDFrameSource(cfg)

    arrival_times = []
    for _ in src:
        arrival_times.append(time.monotonic())

    # Frame index 8 (the first post-burst frame) fires immediately with no
    # pacing delay of its own -- it only re-anchors the pacer's schedule.
    # Frames 9..13's inter-arrival deltas are the real steady-state pacing
    # to check. The regression this guards is a sustained catch-up/free-running
    # phase with several near-zero intervals, not one jittery wall-clock sample.
    post_burst_deltas = [
        b - a for a, b in zip(arrival_times[8:], arrival_times[9:])
    ]
    assert len(post_burst_deltas) == 5

    # Use the aggregate shape rather than the minimum. A single interval can
    # undershoot the nominal period on a loaded shared runner around a sleep
    # boundary (issue #963), while a real catch-up regression emits several
    # near-zero intervals and therefore still drives the median below this
    # lower bound. There is deliberately no upper bound: overshoot is scheduler
    # load, not a pacing defect.
    median_delta = float(np.median(post_burst_deltas))
    assert median_delta >= period * 0.5


def test_is_burst_index_matches_config():
    cfg = SyntheticSourceConfig(seed=0, fps=30.0, num_frames=None, burst_size=3, burst_period_frames=5)
    src = SyntheticRGBDFrameSource(cfg)
    # First burst: indices 0,1,2.
    assert [src._is_burst_index(i) for i in range(3)] == [True, True, True]
    # Steady state: indices 3..7 (5 frames) are not burst.
    assert [src._is_burst_index(i) for i in range(3, 8)] == [False] * 5
    # Second burst starts at index 8.
    assert [src._is_burst_index(i) for i in range(8, 11)] == [True, True, True]


# ---------------------------------------------------------------------------
# RejectionPolicy: configurable acceptance ratio pacing.
# ---------------------------------------------------------------------------

def test_rejection_policy_all_rejected():
    policy = RejectionPolicy(accept_ratio=0.0)
    src = _make_source(num_frames=30)
    accepted = [policy.accepts(f) for f in src]
    assert not any(accepted)


def test_rejection_policy_all_accepted():
    policy = RejectionPolicy(accept_ratio=1.0)
    src = _make_source(num_frames=30)
    accepted = [policy.accepts(f) for f in src]
    assert all(accepted)


def test_rejection_policy_target_rate_is_approximately_correct():
    policy = RejectionPolicy.for_target_rate(source_fps=30.0, target_accepted_fps=2.0)
    src = _make_source(fps=30.0, num_frames=300)
    accepted_count = sum(policy.accepts(f) for f in src)
    # 300 frames @ 30fps = 10s of stream; target 2 accepted/s => ~20 accepted.
    assert 18 <= accepted_count <= 22


def test_rejection_policy_rejects_are_evenly_spread_not_batched():
    policy = RejectionPolicy(accept_ratio=0.5)
    src = _make_source(num_frames=20)
    accepted = [policy.accepts(f) for f in src]
    # Alternating acceptance for exactly 0.5 ratio.
    assert accepted == [i % 2 == 1 for i in range(20)]


def test_rejection_policy_rejects_invalid_ratio():
    with pytest.raises(ValueError):
        RejectionPolicy(accept_ratio=1.5)
    with pytest.raises(ValueError):
        RejectionPolicy(accept_ratio=-0.1)


# ---------------------------------------------------------------------------
# FakeSlamSidecar: delay + failure injection.
# ---------------------------------------------------------------------------

def test_slam_sidecar_delay_injection_records_configured_delay():
    sidecar = FakeSlamSidecar(delay_seconds=0.5, realtime=False)
    src = _make_source(num_frames=1)
    frame = next(iter(src))
    pose = sidecar.track(frame)
    assert pose.shape == (4, 4)
    assert sidecar.last_delay_applied == 0.5


def test_slam_sidecar_delay_injection_actually_sleeps_when_realtime():
    sidecar = FakeSlamSidecar(delay_seconds=0.05, realtime=True)
    src = _make_source(num_frames=1)
    frame = next(iter(src))
    t0 = time.monotonic()
    sidecar.track(frame)
    elapsed = time.monotonic() - t0
    assert elapsed >= 0.045


def test_slam_sidecar_failure_fires_at_configured_point():
    sidecar = FakeSlamSidecar(fail_after=3, realtime=False)
    src = _make_source(num_frames=10)
    frames = list(src)
    successes = 0
    failed_at = None
    for i, frame in enumerate(frames):
        try:
            sidecar.track(frame)
            successes += 1
        except SlamFailure:
            failed_at = i
            break
    assert successes == 3
    assert failed_at == 3


def test_slam_sidecar_reset_clears_failure_state():
    sidecar = FakeSlamSidecar(fail_after=2, realtime=False)
    src = _make_source(num_frames=5)
    frames = list(src)
    with pytest.raises(SlamFailure):
        for frame in frames:
            sidecar.track(frame)
    sidecar.reset()
    # After reset, the counter restarts, so the first two calls succeed again.
    sidecar.track(frames[0])
    sidecar.track(frames[1])
    with pytest.raises(SlamFailure):
        sidecar.track(frames[2])


# ---------------------------------------------------------------------------
# Protocol conformance: drop-in RGBDFrameSource shape.
# ---------------------------------------------------------------------------

def test_source_is_iterable_and_yields_rgbdframe_shape():
    from eligsir.utils.rgbd_frames import RGBDFrame

    src = _make_source(num_frames=3)
    frames = list(src)
    assert len(frames) == 3
    for f in frames:
        assert isinstance(f, RGBDFrame)
        assert f.rgb.dtype == np.uint8
        assert f.depth.dtype == np.float32
        assert f.c2w.shape == (4, 4)
        assert f.intrinsics.width > 0 and f.intrinsics.height > 0


def test_lazy_frames_alias_matches_iter():
    src = _make_source(num_frames=3)
    via_iter = [f.index for f in src]
    src2 = _make_source(num_frames=3)
    via_lazy = [f.index for f in src2.lazy_frames()]
    assert via_iter == via_lazy
