"""WS2 M1: end-to-end parity for --gpu_frame_resident against
--gpu_frame_prep alone, through the REAL insert_gaussians_from_frame.

Requires a CUDA/ROCm device (skipped otherwise) since gpu_frame_prep's
render_alpha/render_depth tensors must already be on the render device --
this exercises the actual fused-mask code path, not a CPU stand-in.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from train_streaming import insert_gaussians_from_frame
from eligsir.utils.streaming_frames import StreamingRGBDFrame

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU")


class _FakeStreamingScene:
    """check_occupancy reports nothing occupied -- every masked candidate
    survives to _stats, which is all this parity test inspects."""

    def check_occupancy(self, xyz_np, voxel_size, check_neighbors=True):
        return np.zeros(xyz_np.shape[0], dtype=bool)

    def check_surface_cells(self, xyz_np, normals_np):
        return np.zeros(xyz_np.shape[0], dtype=bool)


class _FakeGaussians:
    @property
    def get_xyz(self):
        return torch.zeros((0, 3), device="cuda")


def _make_frame(index, c2w, h, w, rgb_np, depth_np, *, fx=500.0, fy=500.0):
    return StreamingRGBDFrame(
        index=index, timestamp=float(index), rgb_path="", depth_path="dummy.png",
        c2w=c2w, fx=fx, fy=fy, cx=w / 2.0, cy=h / 2.0, width=w, height=h,
        depth_scale=1000.0, _rgb_np=rgb_np, _depth_np=depth_np,
    )


def _rot_c2w(theta, t):
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = np.array(
        [[np.cos(theta), -np.sin(theta), 0.0],
         [np.sin(theta), np.cos(theta), 0.0],
         [0.0, 0.0, 1.0]], dtype=np.float32,
    )
    c2w[:3, 3] = t
    return c2w


class _Args(SimpleNamespace):
    def __getattr__(self, name):
        # getattr(args, name, default) pattern used throughout the function --
        # returning None for anything unset would break `or default` logic, so
        # raise AttributeError as a real namespace would (getattr's default
        # kicks in normally).
        raise AttributeError(name)


def _build_args(gpu_frame_resident: bool) -> _Args:
    return _Args(
        streaming_depth_stride=4,
        streaming_min_depth=0.1,
        streaming_max_depth=8.0,
        streaming_insert_voxel_size=0.02,
        streaming_max_new_gaussians_per_frame=2000,
        streaming_insert_opacity=0.05,
        streaming_depth_edge_threshold=0.02,
        streaming_max_view_angle=70.0,
        streaming_insert_knn_scale=False,
        streaming_insertion_debug=True,
        gpu_frame_prep=True,
        gpu_frame_resident=gpu_frame_resident,
        gpu_frame_prep_normals="finite_diff",
        streaming_depth_filter_enabled=True,
        streaming_depth_filter_erode_px=1,
        streaming_depth_filter_window=3,
        streaming_depth_filter_min_valid_ratio=0.5,
        streaming_depth_filter_max_range=0.5,
        streaming_depth_filter_median_thresh=0.2,
        streaming_depth_filter_bottom_margin=0.0,
        streaming_depth_temporal_check=True,
        streaming_depth_temporal_thresh=0.2,
        streaming_depth_temporal_require_for_insert=True,
        streaming_depth_temporal_require_for_sensor_closer=True,
        streaming_insert_alpha_threshold=0.3,
        streaming_insert_score_alpha_weight=1.0,
        streaming_depth_consistency_thresh=0.05,
        streaming_insert_score_no_depth_weight=1.0,
        streaming_insert_score_sensor_closer_weight=2.0,
        streaming_insert_score_mode="boolean_or",
        # issue #408 / #368: bare-accessed by integration_planner since PR
        # #394/#404 (_GUARDED) -- this GPU-only fixture would raise
        # AttributeError without it. Pinned to "footprint", the exact literal
        # that PR removed from the getattr (commit ab9834f), NOT the parser
        # default "knn" -- "knn" would change this fixture's scale math.
        streaming_insert_scale_mode="footprint",
        streaming_surface_cell_dedup=False,
        streaming_insert_surface_fusion=False,
        streaming_insert_knn_dedup=False,
    )


def _build_case(seed=0, h=64, w=80):
    rng = np.random.default_rng(seed)
    depth_mm = (rng.random((h, w)).astype(np.float32) * 3000 + 300).astype(np.uint16)
    depth_mm[5:10, 5:10] = 0  # a hole
    rgb = (rng.random((h, w, 3)) * 255).astype(np.uint8)

    frame = _make_frame(1, _rot_c2w(0.05, [0.02, -0.01, 0.05]), h, w, rgb, depth_mm)
    prev_depth_mm = (rng.random((h, w)).astype(np.float32) * 3000 + 300).astype(np.uint16)
    prev_frame = _make_frame(0, _rot_c2w(0.0, [0.0, 0.0, 0.0]), h, w, rgb, prev_depth_mm)
    prev_depth_meters = prev_depth_mm.astype(np.float32) / 1000.0

    device = torch.device("cuda")
    # Seeded generator: _build_case(seed=k) is called independently for the
    # "resident off" and "resident on" runs being compared, and must produce
    # IDENTICAL render_alpha/render_depth both times -- torch's global RNG
    # advances between calls, so an unseeded torch.rand() here would silently
    # compare two different frames instead of the same frame through two code
    # paths (caught in development: manual reconstruction with shared tensors
    # matched exactly, while the two independent _build_case() calls did not).
    gen = torch.Generator(device=device).manual_seed(seed + 10_000)
    render_alpha = torch.rand(1, h // 2, w // 2, device=device, generator=gen)
    render_depth = torch.rand(1, h // 2, w // 2, device=device, generator=gen) * 3.0

    return frame, prev_frame, prev_depth_meters, render_alpha, render_depth


def _run(gpu_frame_resident: bool, *, arg_overrides=None, seed=0, use_prev_frame=True):
    frame, prev_frame, prev_depth_meters, render_alpha, render_depth = _build_case(seed=seed)
    args = _build_args(gpu_frame_resident)
    if arg_overrides:
        for k, v in arg_overrides.items():
            setattr(args, k, v)
    scene = _FakeStreamingScene()
    gaussians = _FakeGaussians()
    result, stats = insert_gaussians_from_frame(
        gaussians, frame, args, scene,
        prev_frame=(prev_frame if use_prev_frame else None),
        prev_depth_meters=(prev_depth_meters if use_prev_frame else None),
        render_alpha=render_alpha, render_depth=render_depth,
        current_frame_idx=1, _collect_only=True,
    )
    return result, stats


STAGE_KEYS = [
    "raw_candidates",
    "after_stable_depth_filter",
    "rejected_unstable_depth",
    "after_alpha_depth_mask",
    "insert_path_low_alpha",
    "insert_path_no_depth",
    "insert_path_sensor_closer",
    "after_temporal_depth_filter",
    "rejected_temporal_depth",
    "after_edge_filter",
    "after_grazing_filter",
]


def test_fused_mask_stats_match_unfused_gpu_path():
    _, stats_off = _run(gpu_frame_resident=False)
    _, stats_on = _run(gpu_frame_resident=True)
    mismatches = {
        k: (stats_off.get(k), stats_on.get(k))
        for k in STAGE_KEYS
        if stats_off.get(k) != stats_on.get(k)
    }
    assert not mismatches, f"stage-count mismatch(es): {mismatches}"


def test_fused_mask_survives_full_candidate_count():
    result_off, _ = _run(gpu_frame_resident=False)
    result_on, _ = _run(gpu_frame_resident=True)
    n_off = len(result_off.get("pts", [])) if result_off else 0
    n_on = len(result_on.get("pts", [])) if result_on else 0
    assert n_off == n_on, f"candidate count differs: off={n_off} on={n_on}"
    if n_off > 0:
        np.testing.assert_allclose(result_off["pts"], result_on["pts"], atol=1e-4)
        np.testing.assert_allclose(result_off["normals"], result_on["normals"], atol=1e-4)


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_fused_mask_matches_across_seeds(seed):
    _, stats_off = _run(gpu_frame_resident=False, seed=seed)
    _, stats_on = _run(gpu_frame_resident=True, seed=seed)
    mismatches = {k: (stats_off.get(k), stats_on.get(k)) for k in STAGE_KEYS
                  if stats_off.get(k) != stats_on.get(k)}
    assert not mismatches, f"seed={seed} stage-count mismatch(es): {mismatches}"


def test_fused_mask_matches_without_prev_frame():
    """No temporal check possible (no prev_frame) -- exercises the
    temporal_ok(_t) is None path on both sides."""
    _, stats_off = _run(gpu_frame_resident=False, use_prev_frame=False)
    _, stats_on = _run(gpu_frame_resident=True, use_prev_frame=False)
    mismatches = {k: (stats_off.get(k), stats_on.get(k)) for k in STAGE_KEYS
                  if stats_off.get(k) != stats_on.get(k)}
    assert not mismatches, f"stage-count mismatch(es): {mismatches}"


def test_fused_mask_matches_with_temporal_check_disabled():
    overrides = {"streaming_depth_temporal_check": False}
    _, stats_off = _run(gpu_frame_resident=False, arg_overrides=overrides)
    _, stats_on = _run(gpu_frame_resident=True, arg_overrides=overrides)
    mismatches = {k: (stats_off.get(k), stats_on.get(k)) for k in STAGE_KEYS
                  if stats_off.get(k) != stats_on.get(k)}
    assert not mismatches, f"stage-count mismatch(es): {mismatches}"


def test_fused_mask_matches_with_depth_filter_disabled():
    overrides = {"streaming_depth_filter_enabled": False}
    _, stats_off = _run(gpu_frame_resident=False, arg_overrides=overrides)
    _, stats_on = _run(gpu_frame_resident=True, arg_overrides=overrides)
    mismatches = {k: (stats_off.get(k), stats_on.get(k)) for k in STAGE_KEYS
                  if stats_off.get(k) != stats_on.get(k)}
    assert not mismatches, f"stage-count mismatch(es): {mismatches}"


@pytest.mark.parametrize("mode", ["alpha_only", "depth_gap_only", "sensor_closer_only", "weighted_topk", "boolean_or"])
def test_fused_mask_matches_across_score_modes(mode):
    overrides = {"streaming_insert_score_mode": mode}
    _, stats_off = _run(gpu_frame_resident=False, arg_overrides=overrides)
    _, stats_on = _run(gpu_frame_resident=True, arg_overrides=overrides)
    mismatches = {k: (stats_off.get(k), stats_on.get(k)) for k in STAGE_KEYS
                  if stats_off.get(k) != stats_on.get(k)}
    assert not mismatches, f"score_mode={mode} stage-count mismatch(es): {mismatches}"


def test_fused_mask_matches_with_bottom_margin():
    overrides = {"streaming_depth_filter_bottom_margin": 0.2}
    _, stats_off = _run(gpu_frame_resident=False, arg_overrides=overrides)
    _, stats_on = _run(gpu_frame_resident=True, arg_overrides=overrides)
    mismatches = {k: (stats_off.get(k), stats_on.get(k)) for k in STAGE_KEYS
                  if stats_off.get(k) != stats_on.get(k)}
    assert not mismatches, f"stage-count mismatch(es): {mismatches}"


def test_fused_mask_matches_with_edge_threshold_zero():
    overrides = {"streaming_depth_edge_threshold": 0.0}
    _, stats_off = _run(gpu_frame_resident=False, arg_overrides=overrides)
    _, stats_on = _run(gpu_frame_resident=True, arg_overrides=overrides)
    mismatches = {k: (stats_off.get(k), stats_on.get(k)) for k in STAGE_KEYS
                  if stats_off.get(k) != stats_on.get(k)}
    assert not mismatches, f"stage-count mismatch(es): {mismatches}"


def test_fused_mask_matches_with_kornia_normals_backend():
    """WS2 M2: the kornia-normals branch also needed the dense-grid rewrite
    (sampling _kornia_normal_map at the full grid instead of the pre-
    compacted valid subset) -- verify it still matches the non-resident
    path exactly."""
    pytest.importorskip("kornia")
    overrides = {"gpu_frame_prep_normals": "kornia"}
    result_off, stats_off = _run(gpu_frame_resident=False, arg_overrides=overrides)
    result_on, stats_on = _run(gpu_frame_resident=True, arg_overrides=overrides)
    mismatches = {k: (stats_off.get(k), stats_on.get(k)) for k in STAGE_KEYS
                  if stats_off.get(k) != stats_on.get(k)}
    assert not mismatches, f"kornia backend stage-count mismatch(es): {mismatches}"
    n_off = len(result_off.get("pts", [])) if result_off else 0
    n_on = len(result_on.get("pts", [])) if result_on else 0
    assert n_off == n_on
    if n_off > 0:
        np.testing.assert_allclose(result_off["pts"], result_on["pts"], atol=1e-4)
        np.testing.assert_allclose(result_off["normals"], result_on["normals"], atol=1e-4)
