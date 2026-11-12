"""WS2 M1: torch dense port of the temporal depth-consistency check.

Runs on CPU tensors only (no GPU device required), matching the style of
tests/test_gpu_frame_prep_parity.py. Compares
train_streaming._sample_temporal_depth_mask_gpu (dense, full grid) against
train_streaming._sample_temporal_depth_mask (flat, valid-masked) restricted
to the same valid points -- per the mask-late design, evaluating the dense
version everywhere and then ANDing with `valid` must agree with evaluating
the numpy version only on the valid subset, since the check is purely
per-pixel.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from train_streaming import _sample_temporal_depth_mask, _sample_temporal_depth_mask_gpu


def _make_frame(c2w, height, width, *, depth_fx=None, depth_fy=None, depth_cx=None,
                 depth_cy=None, depth_width=None, depth_height=None,
                 fx=500.0, fy=500.0, cx=None, cy=None):
    cx = cx if cx is not None else width / 2.0
    cy = cy if cy is not None else height / 2.0
    return SimpleNamespace(
        c2w=c2w, height=height, width=width,
        fx=fx, fy=fy, cx=cx, cy=cy,
        depth_fx=depth_fx, depth_fy=depth_fy, depth_cx=depth_cx, depth_cy=depth_cy,
        depth_width=depth_width, depth_height=depth_height,
    )


def _rot_z(theta):
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = np.array(
        [[np.cos(theta), -np.sin(theta), 0.0],
         [np.sin(theta), np.cos(theta), 0.0],
         [0.0, 0.0, 1.0]], dtype=np.float32,
    )
    return c2w


def _build_case(seed=0, h=48, w=64, stride=4, thresh=0.07):
    rng = np.random.default_rng(seed)
    d_fx = d_fy = 500.0
    d_cx, d_cy = w / 2.0, h / 2.0

    c2w = _rot_z(0.05)
    c2w[:3, 3] = [0.02, -0.01, 0.05]
    prev_c2w = _rot_z(0.0)
    prev_c2w[:3, 3] = [0.0, 0.0, 0.0]

    frame = _make_frame(c2w, h, w, fx=d_fx, fy=d_fy, cx=d_cx, cy=d_cy)
    prev_frame = _make_frame(prev_c2w, h, w, fx=d_fx, fy=d_fy, cx=d_cx, cy=d_cy)

    prev_depth = (rng.random((h, w)).astype(np.float32) * 2.0 + 0.5)

    ys, xs = np.mgrid[0:h:stride, 0:w:stride]
    z = (rng.random((h, w)).astype(np.float32) * 2.0 + 0.5)
    z_v_grid = z[ys, xs]
    # A handful of invalid/edge points to exercise the ok-mask bounds check.
    z_v_grid_flat = z_v_grid.copy()
    valid = np.ones_like(z_v_grid, dtype=bool)
    valid[0, 0] = False  # excluded from the numpy reference call, must not matter

    return dict(
        z=z_v_grid, xs=xs, ys=ys, frame=frame, prev_frame=prev_frame,
        prev_depth=prev_depth, d_fx=d_fx, d_fy=d_fy, d_cx=d_cx, d_cy=d_cy,
        thresh=thresh, valid=valid,
    )


def _reference_and_dense(case):
    numpy_ok = _sample_temporal_depth_mask(
        case["z"][case["valid"]].astype(np.float32),
        case["xs"][case["valid"]].astype(np.int32),
        case["ys"][case["valid"]].astype(np.int32),
        case["frame"], case["prev_frame"], case["prev_depth"],
        case["d_fx"], case["d_fy"], case["d_cx"], case["d_cy"],
        thresh=case["thresh"],
    )
    dense = _sample_temporal_depth_mask_gpu(
        torch.from_numpy(case["z"]),
        torch.from_numpy(case["xs"].astype(np.float32)),
        torch.from_numpy(case["ys"].astype(np.float32)),
        case["frame"], case["prev_frame"], case["prev_depth"],
        case["d_fx"], case["d_fy"], case["d_cx"], case["d_cy"],
        thresh=case["thresh"],
    )
    return numpy_ok, dense


def test_dense_matches_numpy_on_valid_subset():
    case = _build_case()
    numpy_ok, dense = _reference_and_dense(case)
    assert numpy_ok is not None and dense is not None
    dense_np = dense.numpy()
    dense_on_valid = dense_np[case["valid"]]
    agreement = (dense_on_valid == numpy_ok).mean()
    assert agreement >= 0.999, f"agreement={agreement}"


def test_returns_none_when_no_prev_frame():
    case = _build_case()
    dense = _sample_temporal_depth_mask_gpu(
        torch.from_numpy(case["z"]),
        torch.from_numpy(case["xs"].astype(np.float32)),
        torch.from_numpy(case["ys"].astype(np.float32)),
        case["frame"], None, case["prev_depth"],
        case["d_fx"], case["d_fy"], case["d_cx"], case["d_cy"],
        thresh=case["thresh"],
    )
    assert dense is None


def test_returns_none_when_thresh_non_positive():
    case = _build_case()
    dense = _sample_temporal_depth_mask_gpu(
        torch.from_numpy(case["z"]),
        torch.from_numpy(case["xs"].astype(np.float32)),
        torch.from_numpy(case["ys"].astype(np.float32)),
        case["frame"], case["prev_frame"], case["prev_depth"],
        case["d_fx"], case["d_fy"], case["d_cx"], case["d_cy"],
        thresh=0.0,
    )
    assert dense is None


def test_returns_none_on_non_finite_c2w():
    case = _build_case()
    bad_frame = _make_frame(np.full((4, 4), np.nan, dtype=np.float32), case["frame"].height, case["frame"].width)
    dense = _sample_temporal_depth_mask_gpu(
        torch.from_numpy(case["z"]),
        torch.from_numpy(case["xs"].astype(np.float32)),
        torch.from_numpy(case["ys"].astype(np.float32)),
        bad_frame, case["prev_frame"], case["prev_depth"],
        case["d_fx"], case["d_fy"], case["d_cx"], case["d_cy"],
        thresh=case["thresh"],
    )
    assert dense is None


def test_dense_returns_correct_shape_and_dtype():
    case = _build_case()
    _, dense = _reference_and_dense(case)
    assert dense.shape == case["z"].shape
    assert dense.dtype == torch.bool


def test_with_depth_intrinsics_and_resolution_rescale():
    """Exercise the depth-intrinsics-differs-from-color-resolution branch
    (depth_fx/depth_width/depth_height set and mismatched from height/width)."""
    case = _build_case(h=32, w=40, stride=4)
    prev_frame = case["prev_frame"]
    prev_frame.depth_fx = case["d_fx"] * 0.5
    prev_frame.depth_fy = case["d_fy"] * 0.5
    prev_frame.depth_cx = case["d_cx"] * 0.5
    prev_frame.depth_cy = case["d_cy"] * 0.5
    prev_frame.depth_width = prev_frame.width // 2
    prev_frame.depth_height = prev_frame.height // 2

    numpy_ok, dense = _reference_and_dense(case)
    assert numpy_ok is not None and dense is not None
    dense_on_valid = dense.numpy()[case["valid"]]
    agreement = (dense_on_valid == numpy_ok).mean()
    assert agreement >= 0.999, f"agreement={agreement}"
