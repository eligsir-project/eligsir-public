"""Unit tests for the pure input-quality / depth-quality gate predicates.

Covers the byte-of-the-mechanism helpers used by:
  - the depth-insertion CONFIDENCE->opacity gate (Task 1),
  - the BLUR and EXPOSURE keyframe input gates (Task 2).

These exercise only the pure mappings — no GPU, ROS, or training required.
"""

import numpy as np
import pytest

from eligsir.mapping.scheduler.frame_quality import (
    depth_confidence_opacity,
    blur_laplacian_score,
    blur_reject,
    mean_luma,
    exposure_reject,
)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# --------------------------------------------------------------------------
# Task 1: depth confidence -> seed opacity
# --------------------------------------------------------------------------
def test_confidence_opacity_full_confidence_keeps_init():
    # confidence == 1.0 everywhere -> opacity == init_opacity (within clamp).
    conf = np.ones(5, dtype=np.float32)
    raw = depth_confidence_opacity(conf, init_opacity=0.05, floor=0.3)
    np.testing.assert_allclose(_sigmoid(raw), 0.05, atol=1e-4)


def test_confidence_opacity_low_confidence_reduces_opacity():
    # Low confidence -> faint Gaussian (opacity < init_opacity).
    raw = depth_confidence_opacity(np.array([0.3]), init_opacity=0.05, floor=0.3)
    assert _sigmoid(raw)[0] < 0.05


def test_confidence_opacity_monotonic_in_confidence():
    raw = depth_confidence_opacity(
        np.array([0.3, 0.5, 0.8, 1.0]), init_opacity=0.5, floor=0.0
    )
    op = _sigmoid(raw)
    assert np.all(np.diff(op) > 0)


def test_confidence_opacity_floor_bounds_reduction():
    # floor clamps how far confidence can pull opacity down.
    raw_floored = depth_confidence_opacity(np.array([0.01]), init_opacity=0.5, floor=0.3)
    raw_unfloored = depth_confidence_opacity(np.array([0.01]), init_opacity=0.5, floor=0.0)
    # floor=0.3 -> opacity ~ 0.5*0.3 = 0.15; floor=0.0 -> ~0.5*0.01 = 0.005
    np.testing.assert_allclose(_sigmoid(raw_floored)[0], 0.15, atol=1e-3)
    assert _sigmoid(raw_unfloored)[0] < _sigmoid(raw_floored)[0]


def test_confidence_opacity_floor_one_disables():
    raw = depth_confidence_opacity(np.array([0.1, 0.5, 0.9]), init_opacity=0.2, floor=1.0)
    np.testing.assert_allclose(_sigmoid(raw), 0.2, atol=1e-4)


def test_confidence_opacity_probability_clamped_finite():
    raw = depth_confidence_opacity(np.array([0.0]), init_opacity=0.0, floor=0.0)
    assert np.isfinite(raw).all()


# --------------------------------------------------------------------------
# Task 2: blur predicate
# --------------------------------------------------------------------------
def _sharp_image(h=64, w=64):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, ::4] = 255  # high-frequency vertical stripes
    return img


def _flat_image(h=64, w=64):
    return np.full((h, w, 3), 128, dtype=np.uint8)


def test_blur_score_sharp_greater_than_flat():
    assert blur_laplacian_score(_sharp_image()) > blur_laplacian_score(_flat_image())


def test_blur_reject_rejects_flat_image():
    assert blur_reject(_flat_image(), min_laplacian=20.0) is True


def test_blur_reject_passes_sharp_image():
    assert blur_reject(_sharp_image(), min_laplacian=20.0) is False


def test_blur_downsample_runs():
    s = blur_laplacian_score(_sharp_image(), downsample=4)
    assert s >= 0.0


# --------------------------------------------------------------------------
# Task 2: exposure predicate
# --------------------------------------------------------------------------
def test_mean_luma_normalised_0_255():
    assert mean_luma(np.full((4, 4, 3), 128, dtype=np.uint8)) == pytest.approx(128 / 255.0)


def test_mean_luma_normalised_0_1_passthrough():
    assert mean_luma(np.full((4, 4, 3), 0.5, dtype=np.float32)) == pytest.approx(0.5)


def test_exposure_reject_small_delta_passes():
    assert exposure_reject(current_luma=0.51, reference_luma=0.50, max_delta=0.15) is False


def test_exposure_reject_large_delta_rejects():
    # 0.50 -> 0.70 is a 40% jump, well over 15%.
    assert exposure_reject(current_luma=0.70, reference_luma=0.50, max_delta=0.15) is True


def test_exposure_reject_handles_zero_reference():
    # No divide-by-zero; tiny reference -> large relative delta -> reject.
    assert exposure_reject(current_luma=0.5, reference_luma=0.0, max_delta=0.15) is True


# --------------------------------------------------------------------------
# Task 2: KeyframeAdmitter integration (gates wired, default-off neutrality)
# --------------------------------------------------------------------------
from eligsir.mapping.scheduler.keyframes import (  # noqa: E402
    FrameInfo,
    KeyframeAdmitter,
    KeyframeAdmitterConfig,
)


class _FakeFrame:
    def __init__(self, index, c2w):
        self.index = index
        self.c2w = c2w


def _identity():
    return np.eye(4, dtype=np.float32)


def _translated(tx):
    m = np.eye(4, dtype=np.float32)
    m[0, 3] = tx
    return m


def _render_admit(cam, gaussians, pipe, background, render_depth=False):
    # Full overlap -> render gate admits a moved frame.
    return {"alpha": np.ones((4, 4), dtype=np.float32)}


def test_gates_off_do_not_reject():
    # Default-off config: a moved frame is admitted exactly as before.
    a = KeyframeAdmitter(KeyframeAdmitterConfig(), render_fn=_render_admit)
    a.note_train_frame(FrameInfo(c2w=_identity(), index=0))
    d, _ = a.admit(cam=None, frame=_FakeFrame(1, _translated(0.5)))
    assert d.admitted is True
    assert d.reason not in ("blur_reject", "exposure_reject")


def test_blur_gate_rejects_blurry_frame():
    flat = np.full((32, 32, 3), 128, dtype=np.uint8)  # zero Laplacian variance
    cfg = KeyframeAdmitterConfig(
        blur_gate_enabled=True, blur_gate_min_laplacian=20.0,
        blur_gate_force_admit_iters=1000,
    )
    a = KeyframeAdmitter(cfg, render_fn=_render_admit, load_rgb=lambda f: flat)
    a.note_train_frame(FrameInfo(c2w=_identity(), index=0))
    d, _ = a.admit(cam=None, frame=_FakeFrame(1, _translated(0.5)))
    assert d.admitted is False
    assert d.reason == "blur_reject"


def test_blur_gate_force_admit_after_starvation():
    flat = np.full((32, 32, 3), 128, dtype=np.uint8)
    cfg = KeyframeAdmitterConfig(
        blur_gate_enabled=True, blur_gate_min_laplacian=20.0,
        blur_gate_force_admit_iters=1,  # starvation window of 1 -> stop rejecting fast
    )
    a = KeyframeAdmitter(cfg, render_fn=_render_admit, load_rgb=lambda f: flat)
    a.note_train_frame(FrameInfo(c2w=_identity(), index=0))
    a.admit(cam=None, frame=_FakeFrame(1, _translated(0.5)))  # blur_iter advances
    d, _ = a.admit(cam=None, frame=_FakeFrame(2, _translated(1.0)))
    # Past the 1-iter starvation window the blur gate no longer hard-rejects.
    assert d.reason != "blur_reject"


def test_exposure_gate_rejects_brightness_jump():
    imgs = {1: np.full((8, 8, 3), 50, np.uint8), 2: np.full((8, 8, 3), 200, np.uint8)}
    cfg = KeyframeAdmitterConfig(
        exposure_gate_enabled=True, exposure_max_delta=0.15,
        exposure_force_admit_iters=1000,
    )
    a = KeyframeAdmitter(cfg, render_fn=_render_admit, load_rgb=lambda f: imgs[f.index])
    a.note_train_frame(FrameInfo(c2w=_identity(), index=0))
    # Frame 1 sets the luminance reference (admitted: no prior ref to compare).
    d1, _ = a.admit(cam=None, frame=_FakeFrame(1, _translated(0.5)))
    assert d1.admitted is True
    # Frame 2 is a large brightness jump vs the admitted reference -> rejected.
    d2, _ = a.admit(cam=None, frame=_FakeFrame(2, _translated(1.0)))
    assert d2.admitted is False
    assert d2.reason == "exposure_reject"
