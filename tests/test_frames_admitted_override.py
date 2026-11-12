"""Regression test for the frames_admitted_override kwarg in
write_post_training_report().

Bug: utils/comparison_report.py's trajectory_coverage computation used
len(train_cams) + len(test_cams) as "frames_admitted" -- the CURRENT
active-camera-buffer size at report time, not a cumulative "ever admitted"
count. Partition-Focus v2's demote_partitions() evicts cameras out of
train_cameras, so any run using --streaming_partition_focus_weight
reported a wildly deflated frames_admitted/coverage_fraction (observed:
22/7040 despite a real cumulative admission rate of ~14%, matching a
sibling non-focus run almost exactly). Fixed by threading through the
monotonic streaming_scene._keyframes_admitted counter as an explicit
override.
"""

from __future__ import annotations

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


def _minimal_render_fn(cam, gaussians, pipe, background):
    import torch
    # Real render_fn implementations always produce a GPU tensor (the
    # Gaussian model lives on GPU) -- _gt_image() unconditionally moves the
    # ground-truth image to "cuda" (utils/comparison_report.py:581), so a
    # CPU-only render output was a fixture bug, not a real device-agnostic
    # scenario: it silently only worked in a CPU-only/no-GPU-available test
    # environment, and errors on any machine with a GPU (which is what this
    # container has).
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return {"render": torch.zeros(3, 64, 64, device=device)}


def test_frames_admitted_override_used_when_provided():
    """A demote-shrunk train_cams list must not deflate frames_admitted
    when the caller supplies the real cumulative admission count."""
    with tempfile.TemporaryDirectory() as tmpdir:
        summary = write_post_training_report(
            model_path=tmpdir,
            iteration=5000,
            gaussians=_FakeGaussian(),
            train_cams=[_FakeCam("a"), _FakeCam("b")],  # only 2 left after demotion
            test_cams=[_FakeCam("t")],
            render_fn=_minimal_render_fn,
            pipe=_FakePipe(),
            background=None,
            frames_admitted_override=985,  # real cumulative _keyframes_admitted
        )
        assert summary["trajectory_coverage"]["frames_admitted"] == 985


def test_frames_admitted_falls_back_to_camera_count_when_not_provided():
    """Non-streaming callers (no override) keep the old len()-based behavior."""
    with tempfile.TemporaryDirectory() as tmpdir:
        summary = write_post_training_report(
            model_path=tmpdir,
            iteration=5000,
            gaussians=_FakeGaussian(),
            train_cams=[_FakeCam("a"), _FakeCam("b"), _FakeCam("c")],
            test_cams=[_FakeCam("d")],
            render_fn=_minimal_render_fn,
            pipe=_FakePipe(),
            background=None,
        )
        assert summary["trajectory_coverage"]["frames_admitted"] == 4


def test_frames_admitted_override_none_falls_back():
    """Explicit None override behaves identically to omitting the kwarg."""
    with tempfile.TemporaryDirectory() as tmpdir:
        summary = write_post_training_report(
            model_path=tmpdir,
            iteration=5000,
            gaussians=_FakeGaussian(),
            train_cams=[_FakeCam("a"), _FakeCam("b")],
            test_cams=[_FakeCam("t")],
            render_fn=_minimal_render_fn,
            pipe=_FakePipe(),
            background=None,
            frames_admitted_override=None,
        )
        assert summary["trajectory_coverage"]["frames_admitted"] == 3


def test_budgeted_admission_report_is_persisted_verbatim():
    """The fixed-K manipulation gate needs the cohort owner's real counts."""
    cohort_report = {
        "enabled": True,
        "counts": {
            "candidate": 12,
            "commit_current": 1,
            "commit_cached": 2,
            "drop": 9,
        },
        "pending": 0,
        "commit_budget": 1,
        "rank_policy": "coverage",
        "rank_seed": 7,
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        summary = write_post_training_report(
            model_path=tmpdir,
            iteration=5000,
            gaussians=_FakeGaussian(),
            train_cams=[_FakeCam("a")],
            test_cams=[_FakeCam("t")],
            render_fn=_minimal_render_fn,
            pipe=_FakePipe(),
            background=None,
            budgeted_admission=cohort_report,
        )

    assert summary["budgeted_admission"] == cohort_report
