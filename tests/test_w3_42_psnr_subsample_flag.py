"""Regression test for the test_metrics_subsampled flag (W3-42).

Bug: report.json has always carried n_test (full holdout size) and
n_test_metrics (how many views mean_test_psnr was actually averaged over,
capped by --streaming_report_test_metrics_max, default 128) as two separate
fields, but nothing ever compared them or flagged the divergence -- a real
campaign (kitchen1 G0-G6, n_test=253) quoted mean_test_psnr as if it were a
full-holdout mean without anyone noticing only 128 of 253 views were scored.
Fixed by adding an explicit test_metrics_subsampled boolean plus a loud
one-time print when it fires, mirroring W3-45's "loud, not silent"
retirement-cap warning. See docs/architecture/2026-08-09-w3-42-psnr-subsample-audit.md.
"""

from __future__ import annotations

import sys
import tempfile
from unittest.mock import MagicMock

import pytest
import torch

# utils/comparison_report.py's _gt_image() hardcodes device="cuda" (correct
# in the project's ROCm/HIP container, where "cuda" aliases the AMD GPU --
# see CLAUDE.md). Every test in this module drives write_post_training_report
# through that helper, so all of them need a real accelerator to exercise;
# on a CUDA/ROCm-less host they fail with "Found no NVIDIA driver..." before
# ever reaching the assertions under test. Host-environment-only failure,
# not a product regression -- see issue #437.
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="_gt_image() hardcodes device='cuda'; no accelerator on this host (issue #437)",
)


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


class _FakeGaussian:
    def get_xyz(self):
        import torch
        return torch.zeros(0, 3)


class _FakePipe:
    pass


class _FakeCam:
    def __init__(self, name):
        import torch
        self.image_name = name
        self.original_image = torch.zeros(3, 64, 64)
        from eligsir.utils.score_contract import declare_camera_score_provenance

        declare_camera_score_provenance(self, context="test_subsample_gt")


def _minimal_render_fn(cam, gaussians, pipe, background):
    import torch
    from eligsir.utils.score_contract import declare_canonical_score_tensor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    image, _ = declare_canonical_score_tensor(
        torch.zeros(3, 64, 64, device=device), context="test_subsample_render"
    )
    return {"render": image}


def test_test_metrics_subsampled_true_when_cap_below_n_test():
    with tempfile.TemporaryDirectory() as tmpdir:
        summary = write_post_training_report(
            model_path=tmpdir,
            iteration=5000,
            gaussians=_FakeGaussian(),
            train_cams=[_FakeCam("a")],
            test_cams=[_FakeCam(f"t{i}") for i in range(5)],
            render_fn=_minimal_render_fn,
            pipe=_FakePipe(),
            background=None,
            test_metrics_max=2,
        )
        assert summary["n_test"] == 5
        assert summary["n_test_metrics"] == 2
        assert summary["test_metrics_max"] == 2
        assert summary["test_metrics_subsampled"] is True


def test_test_metrics_subsampled_false_when_cap_covers_all():
    with tempfile.TemporaryDirectory() as tmpdir:
        summary = write_post_training_report(
            model_path=tmpdir,
            iteration=5000,
            gaussians=_FakeGaussian(),
            train_cams=[_FakeCam("a")],
            test_cams=[_FakeCam(f"t{i}") for i in range(3)],
            render_fn=_minimal_render_fn,
            pipe=_FakePipe(),
            background=None,
            test_metrics_max=128,
        )
        assert summary["n_test"] == 3
        assert summary["n_test_metrics"] == 3
        assert summary["test_metrics_max"] == 128
        assert summary["test_metrics_subsampled"] is False


def test_test_metrics_subsampled_false_when_no_cap_given():
    """Default (no test_metrics_max) reproduces existing pre-W3-42 behavior:
    every test camera is scored, subsampled is always False."""
    with tempfile.TemporaryDirectory() as tmpdir:
        summary = write_post_training_report(
            model_path=tmpdir,
            iteration=5000,
            gaussians=_FakeGaussian(),
            train_cams=[_FakeCam("a")],
            test_cams=[_FakeCam(f"t{i}") for i in range(4)],
            render_fn=_minimal_render_fn,
            pipe=_FakePipe(),
            background=None,
        )
        assert summary["n_test"] == 4
        assert summary["n_test_metrics"] == 4
        assert summary["test_metrics_max"] is None
        assert summary["test_metrics_subsampled"] is False
