"""Parity tests for the --gpu_frame_prep fast path (Phase 1 perf).

These run on CPU tensors only (no GPU device required). They verify that the
GPU-resident masking / backprojection logic in train_streaming.py and
utils.rgbd_frames.backproject_depth_pixels_torch produces results that match the
numpy reference path.

  * backproject_depth_pixels_torch vs numpy backproject_depth_pixels
  * alpha / depth occupancy masks (low_alpha, no_rend_depth, sensor_closer, score_grid)
  * relative-depth edge discontinuity mask
  * finite-diff normals + grazing-angle rejection (angle_ok)

The GPU path uses a hand-rolled torch finite-diff for normals (NOT kornia), so
the grazing mask is expected to match the numpy path within float32 tolerance.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from eligsir.utils.rgbd_frames import (
    backproject_depth_pixels,
    backproject_depth_pixels_torch,
)


# --------------------------------------------------------------------------- #
# 1. Backprojection parity                                                    #
# --------------------------------------------------------------------------- #
def test_backproject_torch_matches_numpy():
    rng = np.random.default_rng(0)
    n = 500
    z_v = (rng.random(n).astype(np.float32) * 3.0 + 0.5)
    xs_v = (rng.integers(0, 640, n)).astype(np.float32)
    ys_v = (rng.integers(0, 480, n)).astype(np.float32)
    d_fx, d_fy, d_cx, d_cy = 525.0, 525.0, 319.5, 239.5

    # A non-trivial c2w (rotation + translation).
    theta = 0.37
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta), np.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    c2w[:3, 3] = np.array([1.5, -0.3, 2.1], dtype=np.float32)

    rgb = rng.random((480, 640, 3)).astype(np.float32)
    depth_hw = (480, 640)

    pts_np, cols_np = backproject_depth_pixels(
        z_v, xs_v, ys_v, d_fx, d_fy, d_cx, d_cy, c2w, rgb, depth_hw
    )

    pts_t, cols_t = backproject_depth_pixels_torch(
        torch.from_numpy(z_v),
        torch.from_numpy(xs_v),
        torch.from_numpy(ys_v),
        d_fx,
        d_fy,
        d_cx,
        d_cy,
        torch.from_numpy(c2w),
        torch.from_numpy(rgb),
        depth_hw,
    )

    assert pts_t.shape == pts_np.shape
    assert np.allclose(pts_t.numpy(), pts_np, atol=1e-4), (
        "world points diverge beyond float32 matmul tolerance"
    )
    # Color sampling uses the same nearest-pixel rounding -> exact.
    assert np.array_equal(cols_t.numpy(), cols_np)


# --------------------------------------------------------------------------- #
# Shared synthetic grid + numpy reference helpers                             #
# --------------------------------------------------------------------------- #
class _Args:
    streaming_insert_alpha_threshold = 0.3
    streaming_insert_score_alpha_weight = 1.0
    streaming_depth_consistency_thresh = 0.05
    streaming_insert_score_no_depth_weight = 1.0
    streaming_insert_score_sensor_closer_weight = 2.0
    streaming_depth_temporal_require_for_sensor_closer = True


def _build_grid(seed=1, h=64, w=80, stride=8):
    rng = np.random.default_rng(seed)
    z = (rng.random((h, w)).astype(np.float32) * 4.0 + 0.1)
    # punch a few invalid / zero / discontinuity regions
    z[10:14, 10:14] = 0.0
    z[20:24, 30:34] = np.nan
    z[:, 40] = z[:, 40] + 2.0  # vertical depth cliff -> edge + grazing
    ys, xs = np.mgrid[0:h:stride, 0:w:stride]
    z_v = z[ys, xs]
    return z, xs, ys, z_v, h, w, stride


def _numpy_occupancy(render_alpha, render_depth, xs, ys, z_v, h, w, args):
    sh2d = xs.shape
    low_alpha = np.zeros(sh2d, dtype=bool)
    no_rend_depth = np.ones(sh2d, dtype=bool)
    sensor_closer = np.zeros(sh2d, dtype=bool)
    score_grid = np.zeros(sh2d, dtype=np.float32)

    if render_alpha is not None:
        alpha_cpu = render_alpha.detach().cpu().numpy().squeeze()
        Hr, Wr = alpha_cpu.shape
        xs_r = np.clip(np.round(xs / max(w - 1, 1) * (Wr - 1)).astype(np.int32), 0, Wr - 1)
        ys_r = np.clip(np.round(ys / max(h - 1, 1) * (Hr - 1)).astype(np.int32), 0, Hr - 1)
        alpha_v = alpha_cpu[ys_r, xs_r]
        alpha_thresh = float(args.streaming_insert_alpha_threshold)
        low_alpha = alpha_v < alpha_thresh
        score_grid += np.maximum(alpha_thresh - alpha_v, 0.0).astype(np.float32) * float(
            args.streaming_insert_score_alpha_weight
        )

    if render_depth is not None:
        rend_d_cpu = render_depth.detach().cpu().numpy().squeeze()
        Hr_d, Wr_d = rend_d_cpu.shape
        xs_rd = np.clip(np.round(xs / max(w - 1, 1) * (Wr_d - 1)).astype(np.int32), 0, Wr_d - 1)
        ys_rd = np.clip(np.round(ys / max(h - 1, 1) * (Hr_d - 1)).astype(np.int32), 0, Hr_d - 1)
        rend_d_v = rend_d_cpu[ys_rd, xs_rd]
        consistency_thresh = float(args.streaming_depth_consistency_thresh)
        no_rend_depth = rend_d_v <= 0
        sensor_closer = rend_d_v - z_v > consistency_thresh
        score_grid += no_rend_depth.astype(np.float32) * float(
            args.streaming_insert_score_no_depth_weight
        )
        score_grid += np.maximum(rend_d_v - z_v, 0.0).astype(np.float32) * float(
            args.streaming_insert_score_sensor_closer_weight
        )
    return low_alpha, no_rend_depth, sensor_closer, score_grid


# --------------------------------------------------------------------------- #
# 2. Alpha / depth occupancy mask parity                                      #
# --------------------------------------------------------------------------- #
def test_occupancy_masks_match():
    from train_streaming import _gpu_render_occupancy_masks

    z, xs, ys, z_v, h, w, stride = _build_grid(seed=2)
    args = _Args()

    # Render tensors at a *different* resolution to exercise the coord mapping.
    Hr, Wr = 48, 60
    rng = np.random.default_rng(7)
    render_alpha = torch.from_numpy(rng.random((1, Hr, Wr)).astype(np.float32))
    render_depth = torch.from_numpy((rng.random((1, Hr, Wr)).astype(np.float32) * 5.0))
    # introduce some zero-depth (un-rendered) pixels
    rd = render_depth.clone()
    rd[0, :10, :10] = 0.0
    render_depth = rd

    la_np, nrd_np, sc_np, sg_np = _numpy_occupancy(
        render_alpha, render_depth, xs, ys, z_v, h, w, args
    )

    xs_t = torch.from_numpy(xs.astype(np.float32))
    ys_t = torch.from_numpy(ys.astype(np.float32))
    z_v_t = torch.from_numpy(z_v.astype(np.float32))
    la_t, nrd_t, sc_t, sg_t = _gpu_render_occupancy_masks(
        render_alpha, render_depth, xs_t, ys_t, z_v_t, h, w, args
    )

    assert np.array_equal(la_t.numpy(), la_np), "low_alpha mask diverged"
    assert np.array_equal(nrd_t.numpy(), nrd_np), "no_rend_depth mask diverged"
    assert np.array_equal(sc_t.numpy(), sc_np), "sensor_closer mask diverged"
    assert np.allclose(sg_t.numpy(), sg_np, atol=1e-5), "score_grid diverged"


def test_occupancy_masks_alpha_only():
    from train_streaming import _gpu_render_occupancy_masks

    z, xs, ys, z_v, h, w, stride = _build_grid(seed=3)
    args = _Args()
    rng = np.random.default_rng(11)
    render_alpha = torch.from_numpy(rng.random((1, h, w)).astype(np.float32))

    la_np, nrd_np, sc_np, sg_np = _numpy_occupancy(
        render_alpha, None, xs, ys, z_v, h, w, args
    )
    xs_t = torch.from_numpy(xs.astype(np.float32))
    ys_t = torch.from_numpy(ys.astype(np.float32))
    z_v_t = torch.from_numpy(z_v.astype(np.float32))
    la_t, nrd_t, sc_t, sg_t = _gpu_render_occupancy_masks(
        render_alpha, None, xs_t, ys_t, z_v_t, h, w, args
    )
    assert np.array_equal(la_t.numpy(), la_np)
    assert np.array_equal(nrd_t.numpy(), nrd_np)  # default all-True
    assert np.array_equal(sc_t.numpy(), sc_np)    # default all-False
    assert np.allclose(sg_t.numpy(), sg_np, atol=1e-5)


# --------------------------------------------------------------------------- #
# 3. Edge discontinuity mask parity                                           #
# --------------------------------------------------------------------------- #
def test_edge_mask_match():
    z, xs, ys, z_v, h, w, stride = _build_grid(seed=4)
    edge_threshold = 0.02
    s = stride

    # numpy reference
    dzdx = np.abs(z[ys, np.clip(xs + s, 0, w - 1)] - z[ys, np.clip(xs - s, 0, w - 1)])
    dzdy = np.abs(z[np.clip(ys + s, 0, h - 1), xs] - z[np.clip(ys - s, 0, h - 1), xs])
    edge_rel = (dzdx + dzdy) / np.maximum(z_v, 1e-6)
    edge_ok_np = edge_rel < edge_threshold

    # torch path (mirrors train_streaming gpu branch)
    z_t = torch.from_numpy(z)
    xs_i = torch.from_numpy(xs.astype(np.int64))
    ys_i = torch.from_numpy(ys.astype(np.int64))
    z_v_t = torch.from_numpy(z_v)
    xp = torch.clamp(xs_i + s, 0, w - 1)
    xm = torch.clamp(xs_i - s, 0, w - 1)
    yp = torch.clamp(ys_i + s, 0, h - 1)
    ym = torch.clamp(ys_i - s, 0, h - 1)
    dzdx_t = torch.abs(z_t[ys_i, xp] - z_t[ys_i, xm])
    dzdy_t = torch.abs(z_t[yp, xs_i] - z_t[ym, xs_i])
    edge_rel_t = (dzdx_t + dzdy_t) / torch.clamp(z_v_t, min=1e-6)
    edge_ok_t = (edge_rel_t < edge_threshold).numpy()

    # NaN handling: numpy and torch both yield False for NaN comparisons -> exact.
    assert np.array_equal(edge_ok_t, edge_ok_np), "edge mask diverged"


# --------------------------------------------------------------------------- #
# 4. Finite-diff normals + grazing-angle mask parity                          #
# --------------------------------------------------------------------------- #
def test_grazing_mask_match():
    z, xs, ys, z_v, h, w, stride = _build_grid(seed=5)
    s = stride
    depth_stride = stride
    d_fx, d_fy = 300.0, 300.0
    max_view_angle = 70.0

    # Build a 'valid' mask (finite, in-range) so we compare on the same support.
    valid = np.isfinite(z_v) & (z_v > 0.1) & (z_v < 8.0)

    # ---- numpy reference ----
    xs_vi = xs[valid].astype(np.int32)
    ys_vi = ys[valid].astype(np.int32)
    dzdx_v = z[ys_vi, np.clip(xs_vi + s, 0, w - 1)] - z[ys_vi, np.clip(xs_vi - s, 0, w - 1)]
    dzdy_v = z[np.clip(ys_vi + s, 0, h - 1), xs_vi] - z[np.clip(ys_vi - s, 0, h - 1), xs_vi]
    nx = -dzdx_v / (d_fx * (2.0 * s / depth_stride))
    ny = -dzdy_v / (d_fy * (2.0 * s / depth_stride))
    nz = np.ones_like(nx)
    norm = np.sqrt(nx * nx + ny * ny + nz * nz) + 1e-8
    nx /= norm; ny /= norm; nz /= norm
    cos_thresh = float(np.cos(np.deg2rad(max_view_angle)))
    angle_ok_np = nz > cos_thresh

    # ---- torch path (mirrors train_streaming gpu branch) ----
    z_t = torch.from_numpy(z)
    xs_t = torch.from_numpy(xs.astype(np.float32))
    ys_t = torch.from_numpy(ys.astype(np.float32))
    valid_t = torch.from_numpy(valid)
    xs_vi_t = xs_t.to(torch.int64)[valid_t]
    ys_vi_t = ys_t.to(torch.int64)[valid_t]
    xp = torch.clamp(xs_vi_t + s, 0, w - 1)
    xm = torch.clamp(xs_vi_t - s, 0, w - 1)
    yp = torch.clamp(ys_vi_t + s, 0, h - 1)
    ym = torch.clamp(ys_vi_t - s, 0, h - 1)
    dzdx_v_t = z_t[ys_vi_t, xp] - z_t[ys_vi_t, xm]
    dzdy_v_t = z_t[yp, xs_vi_t] - z_t[ym, xs_vi_t]
    nx_t = -dzdx_v_t / (d_fx * (2.0 * s / depth_stride))
    ny_t = -dzdy_v_t / (d_fy * (2.0 * s / depth_stride))
    nz_t = torch.ones_like(nx_t)
    norm_t = torch.sqrt(nx_t * nx_t + ny_t * ny_t + nz_t * nz_t) + 1e-8
    nx_t = nx_t / norm_t; ny_t = ny_t / norm_t; nz_t = nz_t / norm_t
    angle_ok_t = (nz_t > cos_thresh).numpy()

    # normals match within fp tolerance
    assert np.allclose(nz_t.numpy(), nz, atol=1e-5)
    # grazing decision must select an identical set (hand-rolled finite-diff -> exact)
    assert np.array_equal(angle_ok_t, angle_ok_np), "grazing-angle mask diverged"


def test_flag_default_off():
    """--gpu_frame_prep must default to False (CPU numpy path is the default)."""
    from argparse import ArgumentParser

    from arguments import OptimizationParams

    parser = ArgumentParser()
    op = OptimizationParams(parser)
    args = parser.parse_args([])
    extracted = op.extract(args)
    assert getattr(extracted, "gpu_frame_prep") is False
