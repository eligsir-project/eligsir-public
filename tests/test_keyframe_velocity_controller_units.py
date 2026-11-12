"""Discriminating unit tests for the velocity-adaptive keyframe controller (#793).

These tests are written against a SYNTHETIC CONSTANT-VELOCITY TRAJECTORY with a
known ground-truth speed in m/s. They fail against pre-#793 ``main``, where the
"speed EMA" was fed ``t_m`` -- the CUMULATIVE displacement since the last
ADMITTED keyframe -- so the estimator

* carried units of metres-per-admission-interval, not m/s;
* grew monotonically during a rejection streak and reset on admit, i.e. it was
  a function of the admission pattern rather than of the camera's motion;
* was then multiplied by ``frames_between = input_fps / target_admit_rate_fps``,
  which is not a length.

The controller now samples instantaneous velocity between consecutive CANDIDATE
frames and derives ``effective_min_trans = v_ema [m/s] * (1/target_fps) [s]``.

CPU-only: no ``render_fn`` is supplied, so ``admit()`` takes the
``render_failed`` fast-path and admission is driven purely by the
(velocity-adapted) motion gate. ``max_gap`` is pushed far out and the rotation
criterion disabled in the rate tests so no force path can interfere -- issue
#793's test contract requires clamps and force paths to be separated from
target-rate behaviour.
"""
from __future__ import annotations

import math
from typing import List, Optional

import numpy as np
import pytest

from eligsir.mapping.scheduler.keyframes import (
    FrameInfo,
    KeyframeAdmitter,
    KeyframeAdmitterConfig,
)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _frame(index: int, x: float, timestamp: Optional[float] = None) -> FrameInfo:
    """A frame translated ``x`` metres along +X, optionally timestamped."""
    c2w = np.eye(4, dtype=np.float32)
    c2w[0, 3] = x
    return FrameInfo(c2w=c2w, index=index, timestamp=timestamp)


def _rot_frame(index: int, deg: float, timestamp: Optional[float] = None) -> FrameInfo:
    """A frame rotated ``deg`` degrees about +Z, at the origin."""
    a = math.radians(deg)
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = np.array(
        [[math.cos(a), -math.sin(a), 0.0],
         [math.sin(a), math.cos(a), 0.0],
         [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    return FrameInfo(c2w=c2w, index=index, timestamp=timestamp)


def _adaptive_cfg(**kw) -> KeyframeAdmitterConfig:
    base = dict(
        min_translation_m=0.05,
        min_rotation_deg=1e9,   # disable the (static) rotation criterion
        max_gap=10_000,         # disable max_gap forcing
        velocity_adaptive=True,
        target_admit_rate_fps=2.0,
        input_fps=30.0,
        velocity_ema_alpha=1.0,  # no smoothing: EMA == last instantaneous sample
    )
    base.update(kw)
    return KeyframeAdmitterConfig(**base)


def _drive(cfg, frames: List[FrameInfo]):
    """Feed ``frames`` as candidates. Returns (admitter, decisions)."""
    adm = KeyframeAdmitter(config=cfg)
    adm.note_train_frame(frames[0])
    decisions = []
    for f in frames[1:]:
        d, _ = adm.admit(cam=None, frame=f)
        decisions.append(d)
    return adm, decisions


# ----------------------------------------------------------------------
# 1. Units: the EMA reads the true velocity in m/s
# ----------------------------------------------------------------------
@pytest.mark.parametrize("input_fps,step_m", [(30.0, 0.005), (60.0, 0.005), (30.0, 0.02)])
def test_ema_reads_true_velocity_m_per_s_index_fallback(input_fps, step_m):
    """No timestamps -> documented ``frame_delta / input_fps`` fallback clock.
    The EMA must equal the ground-truth speed in m/s, not m/frame and not a
    cumulative displacement."""
    true_v = step_m * input_fps
    cfg = _adaptive_cfg(input_fps=input_fps)
    frames = [_frame(i, i * step_m) for i in range(0, 60)]
    adm, _ = _drive(cfg, frames)
    assert adm._translation_velocity_ema_m_s == pytest.approx(true_v, rel=1e-4)
    assert adm._timebase_source.startswith("frame_index_nominal_fps")


def test_ema_reads_true_velocity_m_per_s_from_real_timestamps():
    """Real source timestamps in seconds are trusted and used directly."""
    true_v = 0.30  # m/s
    fps = 30.0
    cfg = _adaptive_cfg(input_fps=fps)
    frames = [_frame(i, true_v * (i / fps), timestamp=i / fps) for i in range(0, 60)]
    adm, _ = _drive(cfg, frames)
    assert adm._timebase_source == "source_timestamp"
    assert adm._translation_velocity_ema_m_s == pytest.approx(true_v, rel=1e-4)


# ----------------------------------------------------------------------
# 2. THE discriminating property: the estimate is independent of admission
# ----------------------------------------------------------------------
def test_velocity_estimate_independent_of_admit_or_reject():
    """Same physical trajectory, two admission regimes:

    * ``forced``: max_gap=1, so EVERY candidate is force-admitted;
    * ``starved``: the motion criterion is unreachable, so NO candidate is
      admitted for the whole run.

    The velocity estimate must be identical. Pre-#793 it was not: in the
    starved regime ``t_m`` accumulated with the rejection streak, so the
    "speed" grew without bound.
    """
    step_m, fps, n = 0.005, 30.0, 60
    true_v = step_m * fps
    frames = [_frame(i, i * step_m) for i in range(0, n)]

    forced_cfg = _adaptive_cfg(input_fps=fps, max_gap=1)
    starved_cfg = _adaptive_cfg(
        input_fps=fps,
        max_gap=10_000,
        # Floor/ceil pinned far above anything this trajectory can reach, so
        # the motion criterion can never fire and nothing is ever admitted.
        velocity_adaptive_min_trans_floor=1e6,
        velocity_adaptive_min_trans_ceil=1e9,
    )

    adm_forced, dec_forced = _drive(forced_cfg, frames)
    adm_starved, dec_starved = _drive(starved_cfg, frames)

    assert adm_forced.keyframes_admitted == n - 1        # every candidate forced in
    assert adm_starved.keyframes_admitted == 0           # none admitted at all

    assert adm_forced._translation_velocity_ema_m_s == pytest.approx(true_v, rel=1e-4)
    assert adm_starved._translation_velocity_ema_m_s == pytest.approx(true_v, rel=1e-4)
    assert adm_forced._translation_velocity_ema_m_s == pytest.approx(
        adm_starved._translation_velocity_ema_m_s, rel=1e-6
    )

    # ...and it does not drift over the rejection streak (issue item 6).
    samples = [d.translation_velocity_m_s for d in dec_starved[5:]]
    assert all(s == pytest.approx(true_v, rel=1e-4) for s in samples)


# ----------------------------------------------------------------------
# 3. Realized admit rate tracks target_admit_rate_fps
# ----------------------------------------------------------------------
def _realized_interval_frames(cfg, step_m: float, n: int, fps: float) -> float:
    """Mean source-frame interval between motion admissions over the last
    two thirds of the run (after the EMA has settled)."""
    frames = [_frame(i, i * step_m) for i in range(0, n)]
    adm = KeyframeAdmitter(config=cfg)
    adm.note_train_frame(frames[0])
    admit_idx = []
    for f in frames[1:]:
        d, _ = adm.admit(cam=None, frame=f)
        if d.admitted:
            admit_idx.append(f.index)
    tail = [i for i in admit_idx if i >= n // 3]
    assert len(tail) >= 3, f"too few admissions to measure: {admit_idx}"
    gaps = np.diff(np.asarray(tail, dtype=np.float64))
    return float(np.mean(gaps))


@pytest.mark.parametrize(
    "fps,target_fps,step_m",
    [
        (30.0, 2.0, 0.005),   # 0.15 m/s  -> expect ~15 source frames
        (30.0, 2.0, 0.025),   # 0.75 m/s (5x faster) -> STILL ~15 source frames
        (60.0, 2.0, 0.0025),  # 60 Hz input -> expect ~30 source frames
        (30.0, 3.0, 0.005),   # target 3 Hz -> expect ~10 source frames
    ],
)
def test_realized_admit_interval_tracks_target(fps, target_fps, step_m):
    """Issue #793 test-contract items 1/2/3: the realized motion-admission
    interval must track ``input_fps / target_admit_rate_fps`` source frames,
    across materially different speeds and input rates.

    Speeds are chosen so the adaptive threshold ``v * (1/target_fps)`` lands
    strictly INSIDE [floor=0.01 m, ceil=0.50 m] -- clamp activation is tested
    separately below, so a clamped arm cannot masquerade as a controller
    result.
    """
    expected = fps / target_fps
    v_m_s = step_m * fps
    threshold = v_m_s / target_fps
    assert 0.01 < threshold < 0.50, "test speed must not be clamp-governed"

    cfg = _adaptive_cfg(input_fps=fps, target_admit_rate_fps=target_fps)
    realized = _realized_interval_frames(cfg, step_m, n=600, fps=fps)
    assert realized == pytest.approx(expected, rel=0.15), (
        f"realized {realized:.2f} source frames/admit vs target {expected:.2f}"
    )


def test_fast_and_slow_converge_to_same_interval():
    """5x speed difference -> same realized interval (not 5x different)."""
    cfg = _adaptive_cfg(input_fps=30.0, target_admit_rate_fps=2.0)
    slow = _realized_interval_frames(cfg, 0.005, n=600, fps=30.0)
    fast = _realized_interval_frames(cfg, 0.025, n=600, fps=30.0)
    assert slow == pytest.approx(fast, rel=0.15)
    assert slow == pytest.approx(15.0, rel=0.15)


# ----------------------------------------------------------------------
# 4. Irregular timestamps: the rate follows TIME, not frame index
# ----------------------------------------------------------------------
def test_irregular_timestamps_follow_physical_time():
    """Constant 0.5 m/s expressed with alternating frame periods. The velocity
    estimate must be constant, because the motion per frame is not."""
    v = 0.5
    fps = 30.0
    periods = [1.0 / 30.0, 1.0 / 20.0]
    ts, xs = [0.0], [0.0]
    for i in range(80):
        dt = periods[i % 2]
        ts.append(ts[-1] + dt)
        xs.append(xs[-1] + v * dt)
    frames = [_frame(i, xs[i], timestamp=ts[i]) for i in range(len(ts))]
    cfg = _adaptive_cfg(input_fps=fps)
    adm, dec = _drive(cfg, frames)
    assert adm._timebase_source == "source_timestamp"
    for d in dec[8:]:
        assert d.translation_velocity_m_s == pytest.approx(v, rel=1e-4)
    assert adm._translation_velocity_ema_m_s == pytest.approx(v, rel=1e-4)


# ----------------------------------------------------------------------
# 5. Source stride: units stay correct (issue item 5)
# ----------------------------------------------------------------------
@pytest.mark.parametrize("with_timestamps", [True, False])
def test_source_stride_preserves_velocity_units(with_timestamps):
    """``StridedFrameSource`` yields the underlying frames UNCHANGED, so a
    stride-3 view carries frame-index deltas of 3 (and timestamp deltas of
    3/fps). The reported m/s must be identical to the full-rate view --
    without dividing ``input_fps`` by the stride."""
    fps, step_m, stride = 30.0, 0.005, 3
    true_v = step_m * fps
    cfg = _adaptive_cfg(input_fps=fps)

    def mk(idxs):
        return [
            _frame(i, i * step_m, timestamp=(i / fps) if with_timestamps else None)
            for i in idxs
        ]

    adm_full, _ = _drive(cfg, mk(range(0, 90)))
    adm_strided, _ = _drive(cfg, mk(range(0, 90, stride)))
    assert adm_full._translation_velocity_ema_m_s == pytest.approx(true_v, rel=1e-4)
    assert adm_strided._translation_velocity_ema_m_s == pytest.approx(true_v, rel=1e-4)


# ----------------------------------------------------------------------
# 6. Timebase trust: index-valued and wrong-scale "timestamps" are rejected
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    "ts_fn,label",
    [
        (lambda i: float(i), "index_valued"),        # hypersim/orbbec/realityscan
        (lambda i: float(i) * 33333.0, "microseconds"),  # ScanNet .sens ts_color
        (lambda i: float(i) * 3.33e7, "nanoseconds"),
    ],
)
def test_untrustworthy_timebase_falls_back_to_nominal_fps(ts_fn, label):
    """A timestamp stream whose implied source rate is outside
    ``[input_fps/tol, input_fps*tol]`` must NOT be used as a clock; the
    controller falls back to ``frame_delta / input_fps`` and records why."""
    fps, step_m = 30.0, 0.005
    cfg = _adaptive_cfg(input_fps=fps)
    frames = [_frame(i, i * step_m, timestamp=ts_fn(i)) for i in range(0, 60)]
    adm, dec = _drive(cfg, frames)
    assert adm._timebase_source == "frame_index_nominal_fps", label
    assert adm._implied_input_fps is not None
    # Fallback clock -> the true m/s still comes out.
    assert adm._translation_velocity_ema_m_s == pytest.approx(step_m * fps, rel=1e-4)
    assert dec[-1].timebase_source == "frame_index_nominal_fps"


def test_missing_timestamps_reported_as_fallback():
    cfg = _adaptive_cfg()
    adm, dec = _drive(cfg, [_frame(i, i * 0.005) for i in range(0, 30)])
    assert adm._timebase_source.startswith("frame_index_nominal_fps")
    assert adm._implied_input_fps is None
    assert dec[-1].source_frame_delta == 1
    assert dec[-1].source_dt_s == pytest.approx(1.0 / 30.0)


# ----------------------------------------------------------------------
# 7. Rotation policy (issue item 7): explicitly STATIC, telemetry-only speed
# ----------------------------------------------------------------------
def test_rotation_policy_is_static_with_telemetry_only_velocity():
    """The adaptive law governs TRANSLATION only. On a rotation-only
    trajectory the translation velocity is 0 (threshold pins to the floor),
    the rotation criterion uses the static ``min_rotation_deg``, and the
    rotation speed is reported in deg/s as telemetry."""
    fps, deg_per_frame = 30.0, 0.5
    cfg = _adaptive_cfg(input_fps=fps, min_rotation_deg=5.0, max_gap=10_000)
    frames = [_rot_frame(i, i * deg_per_frame) for i in range(0, 40)]
    adm, dec = _drive(cfg, frames)

    assert adm._translation_velocity_ema_m_s == pytest.approx(0.0, abs=1e-9)
    assert adm._rotation_velocity_ema_deg_s == pytest.approx(deg_per_frame * fps, rel=1e-3)
    # Threshold pinned to the floor by zero translation velocity.
    assert dec[-1].effective_min_translation_m == pytest.approx(
        cfg.velocity_adaptive_min_trans_floor
    )
    # Admissions happen on the STATIC rotation criterion: every 10 frames
    # (0.5 deg/frame vs a 5 deg threshold), unaffected by target_admit_rate_fps.
    admits = [i for i, d in enumerate(dec) if d.admitted]
    assert len(admits) >= 3
    assert float(np.mean(np.diff(np.asarray(admits, dtype=np.float64)))) == pytest.approx(
        10.0, rel=0.15
    )


# ----------------------------------------------------------------------
# 8. Clamp activation is observable and tested separately (issue item 8)
# ----------------------------------------------------------------------
def test_threshold_ceiling_clamp_is_observable():
    """5 m/s at target 2 Hz -> raw threshold 2.5 m, clipped to the 0.50 m
    ceiling. The clamp must be reported, not silently applied."""
    fps = 30.0
    cfg = _adaptive_cfg(input_fps=fps)
    frames = [_frame(i, i * (5.0 / fps)) for i in range(0, 30)]
    adm, dec = _drive(cfg, frames)
    assert dec[-1].threshold_clamp == "ceil"
    assert dec[-1].effective_min_translation_m == pytest.approx(0.50)
    assert adm._threshold_ceil_hits > 0
    assert adm._threshold_floor_hits == 0


def test_threshold_floor_clamp_is_observable():
    """0.003 m/s at target 2 Hz -> raw threshold 0.0015 m, clipped up to the
    0.01 m floor."""
    fps = 30.0
    cfg = _adaptive_cfg(input_fps=fps)
    frames = [_frame(i, i * (0.003 / fps)) for i in range(0, 30)]
    adm, dec = _drive(cfg, frames)
    assert dec[-1].threshold_clamp == "floor"
    assert dec[-1].effective_min_translation_m == pytest.approx(0.01)
    assert adm._threshold_floor_hits > 0
    assert adm._threshold_ceil_hits == 0


def test_unclamped_threshold_reports_none():
    cfg = _adaptive_cfg(input_fps=30.0)
    frames = [_frame(i, i * 0.005) for i in range(0, 30)]
    _, dec = _drive(cfg, frames)
    assert dec[-1].threshold_clamp == "none"
    assert dec[-1].effective_min_translation_m == pytest.approx(0.15 / 2.0, rel=1e-4)


# ----------------------------------------------------------------------
# 9. velocity_adaptive=False is untouched
# ----------------------------------------------------------------------
def test_controller_off_is_static_threshold_and_no_estimation():
    cfg = KeyframeAdmitterConfig(
        min_translation_m=0.05, min_rotation_deg=1e9, max_gap=10_000,
        velocity_adaptive=False,
    )
    adm, dec = _drive(cfg, [_frame(i, i * 0.005) for i in range(0, 30)])
    assert adm._translation_velocity_ema_m_s is None
    assert adm._rotation_velocity_ema_deg_s is None
    assert all(d.threshold_clamp == "n/a" for d in dec)
    assert all(d.effective_min_translation_m == pytest.approx(0.05) for d in dec)
    assert adm._threshold_evals == 0


def test_reset_clears_controller_state():
    cfg = _adaptive_cfg()
    adm, _ = _drive(cfg, [_frame(i, i * 0.005) for i in range(0, 20)])
    assert adm._translation_velocity_ema_m_s is not None
    adm.reset()
    assert adm._translation_velocity_ema_m_s is None
    assert adm._rotation_velocity_ema_deg_s is None
    assert adm._prev_candidate_c2w is None
    assert adm._threshold_evals == 0
    assert adm._source_elapsed_s == 0.0


# ----------------------------------------------------------------------
# 10. #795 controller-realization fingerprint fields
# ----------------------------------------------------------------------
def test_admission_fingerprint_records_controller_realization():
    from eligsir.utils.admission_fingerprint import AdmissionFingerprint

    fps, target_fps, step_m, n = 30.0, 2.0, 0.005, 600
    cfg = _adaptive_cfg(input_fps=fps, target_admit_rate_fps=target_fps)
    frames = [_frame(i, i * step_m, timestamp=i / fps) for i in range(0, n)]

    fp = AdmissionFingerprint()
    fp.nominal_policy = "hybrid_keyframe"
    fp.max_gap = cfg.max_gap
    fp.target_admit_fps = target_fps
    fp.input_fps = fps

    adm = KeyframeAdmitter(config=cfg)
    adm.note_train_frame(frames[0])
    for f in frames[1:]:
        fp.record_candidate()
        d, _ = adm.admit(cam=None, frame=f)
        fp.record_controller_state(d)
        if d.admitted:
            fp.record_admit(reason=d.reason, gap=d.gap)
        else:
            fp.record_reject(d.reason)
    fp.finalize(total_source_frames=n)

    blob = fp.to_dict()["controller"]
    assert blob["velocity_adaptive"] is True
    assert blob["timebase_source"] == "source_timestamp"
    assert blob["configured_input_fps"] == pytest.approx(fps)
    assert blob["implied_input_fps"] == pytest.approx(fps, rel=0.05)
    # Units are m/s: 0.005 m/frame * 30 fps = 0.15 m/s.
    assert blob["translation_velocity_ema_m_s"]["p50"] == pytest.approx(0.15, rel=0.05)
    assert blob["translation_velocity_inst_m_s"]["p50"] == pytest.approx(0.15, rel=0.05)
    # Adaptive threshold distribution: 0.15 m/s / 2 Hz = 0.075 m.
    assert blob["adaptive_threshold_m"]["p50"] == pytest.approx(0.075, rel=0.05)
    # Clamp fractions have a real denominator, so 0.0 != "never ran".
    assert blob["threshold_evals"] > 100
    assert blob["threshold_floor_frac"] == pytest.approx(0.0)
    assert blob["threshold_ceil_frac"] == pytest.approx(0.0)
    # Target-vs-realized rate error, against REAL elapsed source seconds.
    assert blob["source_elapsed_s"] == pytest.approx((n - 2) / fps, rel=0.02)
    assert blob["realized_admit_fps_source_time"] == pytest.approx(target_fps, rel=0.15)
    assert abs(blob["target_vs_realized_rate_rel_error"]) < 0.15
