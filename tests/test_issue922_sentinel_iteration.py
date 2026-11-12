"""Tests for issue #922: final comparison/report folder naming with real iteration.

Verify that final comparison/report directories use the real terminal training
iteration (e.g., iter_6371) instead of sentinel values like iter_2147483647.
"""
from __future__ import annotations

import os
import sys
import tempfile
from unittest.mock import MagicMock

import pytest


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


torch = pytest.importorskip("torch")

from eligsir.utils.comparison_report import write_post_training_report


class _FakeGaussian:
    def get_xyz(self):
        return torch.zeros(0, 3)


class _FakePipe:
    pass


class _FakeCam:
    def __init__(self, name):
        self.image_name = name
        self.original_image = torch.zeros(3, 64, 64)


def _minimal_render_fn(cam, gaussians, pipe, background):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return {"render": torch.zeros(3, 64, 64, device=device)}


def test_final_report_dir_uses_true_final_iteration_not_sentinel():
    """Issue #922: online_mode's bogus iteration=2^31 sentinel must not leak
    into the output directory name when the caller supplies the real
    terminal iteration via true_final_iteration."""
    with tempfile.TemporaryDirectory() as tmpdir:
        write_post_training_report(
            model_path=tmpdir,
            iteration=2_147_483_647,
            gaussians=_FakeGaussian(),
            train_cams=[_FakeCam("a")],
            test_cams=[_FakeCam("t")],
            render_fn=_minimal_render_fn,
            pipe=_FakePipe(),
            background=None,
            true_final_iteration=6371,
        )
        assert os.path.isdir(os.path.join(tmpdir, "comparison", "iter_6371"))
        assert not os.path.isdir(
            os.path.join(tmpdir, "comparison", "iter_2147483647")
        )


def test_final_report_dir_falls_back_to_iteration_without_true_final_iteration():
    """Callers that never pass true_final_iteration keep the pre-existing
    directory-naming behavior unchanged."""
    with tempfile.TemporaryDirectory() as tmpdir:
        write_post_training_report(
            model_path=tmpdir,
            iteration=5000,
            gaussians=_FakeGaussian(),
            train_cams=[_FakeCam("a")],
            test_cams=[_FakeCam("t")],
            render_fn=_minimal_render_fn,
            pipe=_FakePipe(),
            background=None,
        )
        assert os.path.isdir(os.path.join(tmpdir, "comparison", "iter_5000"))


def test_explicit_subdir_still_overrides_iteration_based_naming():
    """An explicit subdir must still win over both iteration and
    true_final_iteration -- issue #922 only changes the default."""
    with tempfile.TemporaryDirectory() as tmpdir:
        write_post_training_report(
            model_path=tmpdir,
            iteration=2_147_483_647,
            gaussians=_FakeGaussian(),
            train_cams=[_FakeCam("a")],
            test_cams=[_FakeCam("t")],
            render_fn=_minimal_render_fn,
            pipe=_FakePipe(),
            background=None,
            true_final_iteration=6371,
            subdir="iter_6371_bootstrap_views",
        )
        assert os.path.isdir(
            os.path.join(tmpdir, "comparison", "iter_6371_bootstrap_views")
        )
