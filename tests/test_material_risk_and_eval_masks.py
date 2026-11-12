"""Unit tests for geometry-reliability / material-risk heuristics
(eligsir/mapping/geometry/material_risk.py) and pixel-level glass/
specular evaluation masks (eligsir/mapping/geometry/eval_masks.py) —
Wave 3, PR W3-08, deliverables 5 ("pixel masks, not only trajectory index
ranges") and part of the geometry-reliability model (plan section 13.4).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eligsir.mapping.geometry.eval_masks import (  # noqa: E402
    glass_specular_pixel_mask,
    masked_psnr,
    specular_saturation_mask,
    split_render_gt_diff_panel,
)
from eligsir.mapping.geometry.material_risk import (  # noqa: E402
    geometry_reliability_map,
    material_risk_mask,
)


def test_reliability_surface_hit_full_confidence():
    kind_map = torch.zeros(2, 2, dtype=torch.int8)  # all SURFACE_HIT
    depth = torch.full((2, 2), 1.5)
    rgb = torch.full((3, 2, 2), 0.5)
    rel = geometry_reliability_map(kind_map, depth, rgb)
    assert torch.allclose(rel, torch.ones(2, 2))


def test_reliability_unknown_pixels_are_zero_by_default():
    kind_map = torch.full((2, 2), 2, dtype=torch.int8)  # all UNKNOWN
    depth = torch.zeros(2, 2)
    rgb = torch.full((3, 2, 2), 0.5)
    rel = geometry_reliability_map(kind_map, depth, rgb)
    assert torch.allclose(rel, torch.zeros(2, 2))


def test_reliability_saturation_penalized():
    kind_map = torch.zeros(1, 2, dtype=torch.int8)
    depth = torch.full((1, 2), 1.0)
    rgb = torch.zeros(3, 1, 2)
    rgb[:, 0, 0] = 0.5   # normal
    rgb[:, 0, 1] = 0.99  # saturated
    rel = geometry_reliability_map(kind_map, depth, rgb)
    assert rel[0, 1] < rel[0, 0]


def test_reliability_pose_confidence_scales_everything():
    kind_map = torch.zeros(2, 2, dtype=torch.int8)
    depth = torch.full((2, 2), 1.5)
    rgb = torch.full((3, 2, 2), 0.5)
    rel_full = geometry_reliability_map(kind_map, depth, rgb, pose_confidence=1.0)
    rel_half = geometry_reliability_map(kind_map, depth, rgb, pose_confidence=0.5)
    assert torch.allclose(rel_half, rel_full * 0.5)


def test_material_risk_mask_requires_both_unknown_and_low_reliability():
    kind_map = torch.tensor([[2, 0], [2, 2]], dtype=torch.int8)  # UNKNOWN,SURFACE_HIT,UNKNOWN,UNKNOWN
    reliability = torch.tensor([[0.1, 0.9], [0.9, 0.1]])
    mask = material_risk_mask(kind_map, reliability, reliability_thresh=0.35)
    assert mask.tolist() == [[True, False], [False, True]]


def test_specular_saturation_mask_flags_bright_and_dark():
    gt = np.zeros((4, 4, 3), dtype=np.uint8)
    gt[0, 0] = 255  # bright
    gt[1, 1] = 0    # dark
    gt[2, 2] = 128  # normal
    mask = specular_saturation_mask(gt)
    assert mask[0, 0] and mask[1, 1]
    assert not mask[2, 2]


def test_glass_specular_pixel_mask_unions_depth_signal():
    gt = np.full((4, 4, 3), 128, dtype=np.uint8)  # no saturation anywhere
    depth = np.ones((4, 4), dtype=np.float32)
    depth[0, 0] = 0.0  # invalid
    mask = glass_specular_pixel_mask(gt, sensor_depth=depth)
    assert mask[0, 0]
    assert not mask[1, 1]


def test_split_render_gt_diff_panel_roundtrip():
    render = np.full((4, 8, 3), 10, dtype=np.uint8)
    gt = np.full((4, 8, 3), 200, dtype=np.uint8)
    diff = np.full((4, 8, 3), 50, dtype=np.uint8)
    gap = np.full((4, 4, 3), 255, dtype=np.uint8)
    panel = np.concatenate([render, gap, gt, gap, diff], axis=1)
    r, g, d = split_render_gt_diff_panel(panel, gap_px=4)
    assert np.array_equal(r, render)
    assert np.array_equal(g, gt)
    assert np.array_equal(d, diff)


def test_masked_psnr_perfect_match_and_empty_mask():
    render = np.full((2, 2, 3), 100, dtype=np.uint8)
    gt = render.copy()
    mask_all = np.ones((2, 2), dtype=bool)
    mask_none = np.zeros((2, 2), dtype=bool)
    assert masked_psnr(render, gt, mask_all) == 99.0
    assert masked_psnr(render, gt, mask_none) is None
