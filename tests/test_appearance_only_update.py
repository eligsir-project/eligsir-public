"""Unit tests for the appearance-only auxiliary update (Wave 3, PR W3-08,
plan section 13.5) — eligsir/mapping/geometry/appearance_only.py.

Uses a fake ``render_fn``/``gaussians`` duck-type pair (no real renderer, no
GPU) to verify: (1) ``DetachedGeometryView`` correctly detaches only the
geometry properties while delegating everything else straight through, and
(2) ``run_appearance_only_update`` skips work when the mask is empty and
otherwise computes a plain masked L1 loss scaled by ``weight`` (deliverable
6: this weight is entirely independent of the main depth-loss weight — it's
just a function parameter here, not read from any shared config knob).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eligsir.mapping.geometry.appearance_only import (  # noqa: E402
    DetachedGeometryView,
    run_appearance_only_update,
)


class _FakeGaussians:
    def __init__(self):
        self._xyz = torch.zeros(3, 3, requires_grad=True)
        self._scale = torch.ones(3, 3, requires_grad=True)
        self._rot = torch.zeros(3, 4, requires_grad=True)
        self._opac = torch.ones(3, 1, requires_grad=True)
        self.sh0 = torch.zeros(3, 1, 3, requires_grad=True)  # "appearance" params
        self.active_sh_degree = 0

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_scaling(self):
        return self._scale

    @property
    def get_rotation(self):
        return self._rot

    @property
    def _rotation(self):
        return self._rot

    @property
    def get_opacity(self):
        return self._opac


def test_detached_geometry_view_detaches_only_geometry():
    g = _FakeGaussians()
    proxy = DetachedGeometryView(g)
    assert proxy.get_xyz.requires_grad is False
    assert proxy.get_scaling.requires_grad is False
    assert proxy.get_rotation.requires_grad is False
    assert proxy.get_opacity.requires_grad is False
    # Delegated (non-overridden) attribute still live / unchanged.
    assert proxy.sh0 is g.sh0
    assert proxy.sh0.requires_grad is True
    assert proxy.active_sh_degree == 0


def test_run_appearance_only_update_empty_mask_is_noop():
    calls = []

    def fake_render(cam, pc, pipe, bg):
        calls.append(1)
        return {"render": torch.zeros(3, 4, 4)}

    g = _FakeGaussians()
    mask = torch.zeros(4, 4, dtype=torch.bool)
    result = run_appearance_only_update(
        fake_render, g, viewpoint_cam=None, pipe=None, bg_color=None,
        gt_image=torch.zeros(3, 4, 4), mask=mask, weight=1.0,
    )
    assert result is None
    assert calls == []  # never renders when there's nothing to supervise


def test_run_appearance_only_update_computes_masked_weighted_l1():
    def fake_render(cam, pc, pipe, bg):
        rendered = torch.zeros(3, 2, 2)
        rendered[:, 0, 0] = 1.0  # only this pixel differs from GT
        return {"render": rendered}

    g = _FakeGaussians()
    gt = torch.zeros(3, 2, 2)
    mask = torch.zeros(2, 2, dtype=torch.bool)
    mask[0, 0] = True  # restrict loss to the one differing pixel
    loss = run_appearance_only_update(
        fake_render, g, viewpoint_cam=None, pipe=None, bg_color=None,
        gt_image=gt, mask=mask, weight=2.0,
    )
    assert loss is not None
    # |1-0| averaged over the 3 color channels of the single masked pixel = 1.0, * weight 2.0
    assert abs(float(loss) - 2.0) < 1e-6


def test_run_appearance_only_update_uses_detached_geometry_view():
    seen_pc = {}

    def fake_render(cam, pc, pipe, bg):
        seen_pc["pc"] = pc
        return {"render": torch.zeros(3, 2, 2)}

    g = _FakeGaussians()
    mask = torch.ones(2, 2, dtype=torch.bool)
    run_appearance_only_update(
        fake_render, g, viewpoint_cam=None, pipe=None, bg_color=None,
        gt_image=torch.zeros(3, 2, 2), mask=mask, weight=1.0,
    )
    assert isinstance(seen_pc["pc"], DetachedGeometryView)
    assert seen_pc["pc"].get_xyz.requires_grad is False
