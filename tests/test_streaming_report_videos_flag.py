"""Tests for the `report_videos` kwarg on write_post_training_report().

Task 8 (2026-09-08): ablator does not reliably pass SPLAT_REPORT_VIDEOS
through to the container for this job type (confirmed live via `docker exec
<container> env` on two separate ablator-launched runs -- the var was absent
despite being set in the ablator spec's per-arm "env" field). CLI flags via
extra_args have worked reliably all session, so `--streaming_report_videos`
(arguments/__init__.py's `self.streaming_report_videos`) is added as a robust
alternative for skipping the two post-mapping trajectory/depth-timeseries
videos, threaded into write_post_training_report()'s `report_videos` kwarg.

The env var SPLAT_REPORT_VIDEOS is kept as a secondary override on top of the
flag (for whichever mechanism does end up reaching the container), so all
three combinations below must hold.
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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return {"render": torch.zeros(3, 64, 64, device=device)}


def _write_report(tmpdir, *, report_videos=None, env=None, monkeypatch):
    if env is None:
        monkeypatch.delenv("SPLAT_REPORT_VIDEOS", raising=False)
    else:
        monkeypatch.setenv("SPLAT_REPORT_VIDEOS", env)
    kwargs = dict(
        model_path=tmpdir,
        iteration=5000,
        gaussians=_FakeGaussian(),
        train_cams=[_FakeCam("a"), _FakeCam("b")],
        test_cams=[_FakeCam("t")],
        render_fn=_minimal_render_fn,
        pipe=_FakePipe(),
        background=None,
    )
    if report_videos is not None:
        kwargs["report_videos"] = report_videos
    return write_post_training_report(**kwargs)


def _has_trajectory_video(summary):
    """Whether trajectory.mp4 was actually rendered+encoded. The fake camera
    fixture has no image_width/height, so summary["trajectory_video"]'s
    width/height block (only populated when those attrs are present) can't
    be used as the signal -- the artifacts list (populated whenever
    render_camera_path successfully writes the file, regardless of that
    optional metadata) is the reliable one. See comparison_report.py's
    render_camera_path call site in write_post_training_report."""
    return any(path.endswith("trajectory.mp4") for path in summary["artifacts"])


def test_flag_off_and_env_unset_skips_video(monkeypatch):
    """--no-streaming_report_videos (report_videos=False) disables video
    rendering on its own, with no need for the env var."""
    with tempfile.TemporaryDirectory() as tmpdir:
        summary = _write_report(tmpdir, report_videos=False, env=None, monkeypatch=monkeypatch)
        assert not _has_trajectory_video(summary)


def test_flag_default_and_env_unset_keeps_video(monkeypatch):
    """Today's unchanged behavior: omitting report_videos (default True)
    with no env var set still renders the trajectory video."""
    with tempfile.TemporaryDirectory() as tmpdir:
        summary = _write_report(tmpdir, report_videos=None, env=None, monkeypatch=monkeypatch)
        assert _has_trajectory_video(summary)


def test_flag_on_but_env_zero_still_skips_video(monkeypatch):
    """The env var remains a secondary override on top of the CLI flag, for
    whichever mechanism does reach the container."""
    with tempfile.TemporaryDirectory() as tmpdir:
        summary = _write_report(tmpdir, report_videos=True, env="0", monkeypatch=monkeypatch)
        assert not _has_trajectory_video(summary)


def test_streaming_report_videos_registered_as_cli_flag():
    """--streaming_report_videos/--no-streaming_report_videos must be real
    parser options (arguments/__init__.py's plain `self.streaming_report_videos
    = True`, auto-registered by ParamGroup's BooleanOptionalAction pattern --
    same convention as the other streaming_report_* booleans).

    `python3 train.py --help` itself hits an unrelated pre-existing argparse
    formatter crash (ValueError: unsupported format character 'm', from some
    other action's help string containing a bare '%m') on this worktree even
    with this task's changes fully reverted, so this asserts against the
    parser's own action table instead of shelling out to --help.
    """
    from eligsir.cli.parser import parse_training_args

    args, bundle = parse_training_args(["-s", "/tmp/does-not-need-to-exist"], description="x")
    assert args.streaming_report_videos is True

    option_strings = [
        opt for action in bundle.parser._actions for opt in action.option_strings
    ]
    matches = [opt for opt in option_strings if "streaming_report_videos" in opt]
    assert matches == ["--streaming_report_videos", "--no-streaming_report_videos"]
