"""Verify write_post_training_report actually wires scene_vs_trajectory into
geometry_metrics.json/report.json -- a live-code-path check, not just a
direct call to compare_scene_vs_trajectory (see utils/scene_geometry_metrics).
"""

from __future__ import annotations

import json
import sys
import tempfile
from unittest.mock import MagicMock

import pytest
import torch


@pytest.fixture(autouse=True, scope="module")
def _mock_lpips_import():
    had_original = "lpipsPyTorch" in sys.modules
    original = sys.modules.get("lpipsPyTorch")
    sys.modules["lpipsPyTorch"] = MagicMock()
    try:
        yield
    finally:
        if had_original:
            sys.modules["lpipsPyTorch"] = original
        else:
            sys.modules.pop("lpipsPyTorch", None)


from eligsir.utils.comparison_report import write_post_training_report


class _RealishGaussian:
    """Fake gaussian model exposing real tensors on the same properties
    shape_stats_from_model/compare_scene_vs_trajectory read (get_xyz,
    get_opacity, get_scaling, get_features) -- unlike test_comparison_report_
    seed_fade.py's _FakeGaussian (get_xyz as a bound method, not a tensor),
    which never exercises the geometry block for real."""

    def __init__(self, n=200):
        self._xyz = torch.randn(n, 3)
        self._opacity = torch.full((n,), 0.8)
        self._scaling = torch.full((n, 3), 0.05)
        self._features = torch.zeros(n, 1, 3)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_opacity(self):
        return self._opacity

    @property
    def get_scaling(self):
        return self._scaling

    @property
    def get_features(self):
        return self._features


class _FakeCam:
    def __init__(self, center, name):
        self.camera_center = torch.tensor(center, dtype=torch.float32)
        self.image_name = name
        self.original_image = torch.zeros(3, 64, 64, device="cuda")


class _FakePipe:
    pass


def _minimal_render_fn(cam, gaussians, pipe, background):
    return {"render": torch.zeros(3, 64, 64, device="cuda")}


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason=(
        "utils/comparison_report.py's _gt_image() hardcodes device='cuda' "
        "(correct in the project's ROCm/HIP container, where 'cuda' aliases "
        "the AMD GPU -- see CLAUDE.md) and this fixture's _FakeCam constructs "
        "its GT tensor with device='cuda' directly; there is no accelerator "
        "on this host to exercise either. Host-environment-only failure, "
        "not a product regression -- see issue #437."
    ),
)
def test_scene_vs_trajectory_wired_into_geometry_metrics():
    train_cams = [_FakeCam([0.0, 0.0, 0.0], "a"), _FakeCam([1.0, 0.0, 0.0], "b"), _FakeCam([0.0, 1.0, 0.0], "c")]
    with tempfile.TemporaryDirectory() as tmpdir:
        summary = write_post_training_report(
            model_path=tmpdir,
            iteration=5000,
            gaussians=_RealishGaussian(),
            train_cams=train_cams,
            test_cams=[],
            render_fn=_minimal_render_fn,
            pipe=_FakePipe(),
            background=None,
        )
        geom = summary["geometry"]
        assert "scene_vs_trajectory" in geom
        assert geom["scene_vs_trajectory"]["valid"] is True

        with open(f"{tmpdir}/comparison/iter_5000/geometry_metrics.json") as f:
            on_disk = json.load(f)
        assert "scene_vs_trajectory" in on_disk
        assert on_disk["scene_vs_trajectory"]["valid"] is True
