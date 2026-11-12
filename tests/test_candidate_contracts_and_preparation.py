"""Tests for the prepared-candidate data contracts + CPU preparation logic
(plan section 8.3/8.4, PR W3-03)."""

from __future__ import annotations

import numpy as np
import pytest

from eligsir.mapping.candidates.contracts import (
    RawFramePacket,
    TrackedFramePacket,
    TrackingState,
    UndecodedBufferRef,
)
from eligsir.mapping.candidates.preparation import (
    coarse_pose_novelty,
    decode_depth_m,
    decode_rgb,
    prepare_candidate,
)
from eligsir.utils.rgbd_frames import CameraIntrinsics


def _intrinsics():
    return CameraIntrinsics(width=8, height=8, fx=10.0, fy=10.0, cx=4.0, cy=4.0, depth_scale=1000.0)


def _c2w(tx: float = 0.0) -> np.ndarray:
    m = np.eye(4, dtype=np.float32)
    m[0, 3] = tx
    return m


def _sharp_rgb() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, size=(8, 8, 3), dtype=np.uint8)


def test_buffer_ref_is_empty_true_only_when_all_fields_unset():
    assert UndecodedBufferRef().is_empty() is True
    assert UndecodedBufferRef(path="x.png").is_empty() is False
    assert UndecodedBufferRef(array=np.zeros((2, 2))).is_empty() is False


def test_decode_rgb_from_array_ref_round_trips():
    rgb = _sharp_rgb()
    out = decode_rgb(UndecodedBufferRef(array=rgb))
    assert out is not None
    assert out.shape == rgb.shape


def test_decode_rgb_of_empty_ref_returns_none():
    assert decode_rgb(UndecodedBufferRef()) is None


def test_decode_depth_scales_uint16_by_depth_scale():
    depth_mm = np.full((4, 4), 2000, dtype=np.uint16)
    out = decode_depth_m(UndecodedBufferRef(array=depth_mm), depth_scale=1000.0)
    assert out is not None
    assert np.allclose(out, 2.0)


def test_coarse_pose_novelty_is_maximal_with_no_previous_pose():
    assert coarse_pose_novelty(None, _c2w()) == 1.0


def test_coarse_pose_novelty_zero_for_identical_pose():
    pose = _c2w(1.0)
    assert coarse_pose_novelty(pose, pose) == pytest.approx(0.0, abs=1e-6)


def test_coarse_pose_novelty_scales_with_translation():
    prev = _c2w(0.0)
    near = _c2w(0.05)
    far = _c2w(5.0)
    n_near = coarse_pose_novelty(prev, near, translation_norm_m=1.0)
    n_far = coarse_pose_novelty(prev, far, translation_norm_m=1.0)
    assert 0.0 < n_near < n_far
    assert n_far == 1.0  # clipped


def _tracked(rgb=None, depth=None, tx: float = 0.0) -> TrackedFramePacket:
    raw = RawFramePacket(
        source_index=0,
        timestamp_ns=0,
        rgb_ref=UndecodedBufferRef(array=rgb if rgb is not None else _sharp_rgb()),
        depth_ref=UndecodedBufferRef(array=depth) if depth is not None else UndecodedBufferRef(),
        intrinsics=_intrinsics(),
    )
    return TrackedFramePacket(
        frame=raw,
        world_T_camera=_c2w(tx),
        tracking_state=TrackingState.OK,
        tracking_confidence=1.0,
        slam_map_id=0,
    )


def test_prepare_candidate_produces_cpu_resident_buffers_not_none_for_valid_frame():
    candidate = prepare_candidate(_tracked(), previous_c2w=None)
    assert candidate.rgb_ref.array is not None
    assert candidate.rgb_ref.width == 8 and candidate.rgb_ref.height == 8
    assert candidate.input_evidence.reject_reason is None or isinstance(
        candidate.input_evidence.reject_reason, str
    )
    assert candidate.coarse_novelty == 1.0  # no previous pose


def test_prepare_candidate_reports_reject_reason_on_decode_failure():
    raw = RawFramePacket(
        source_index=1, timestamp_ns=1,
        rgb_ref=UndecodedBufferRef(), depth_ref=UndecodedBufferRef(), intrinsics=_intrinsics(),
    )
    tracked = TrackedFramePacket(
        frame=raw, world_T_camera=_c2w(), tracking_state=TrackingState.OK,
        tracking_confidence=1.0, slam_map_id=0,
    )
    candidate = prepare_candidate(tracked)
    assert candidate.input_evidence.reject_reason == "rgb_decode_failed"
    assert candidate.rgb_ref.array is None


def test_prepare_candidate_never_returns_a_gpu_tensor():
    """Deliverable #3's guarantee at the unit level: whatever
    prepare_candidate returns is either None or a plain numpy array, never
    a torch.Tensor (which would imply it was already moved to a device)."""
    candidate = prepare_candidate(_tracked())
    assert candidate.rgb_ref.array is None or isinstance(candidate.rgb_ref.array, np.ndarray)
    assert candidate.depth_ref.array is None or isinstance(candidate.depth_ref.array, np.ndarray)
