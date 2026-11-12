"""Regression test for GitHub issue (2026-08-13): mid-training and other
non-final write_post_training_report() call sites in train_streaming.py pass
`streaming_scene=` (for partition metrics / four-clocks telemetry) but not
`eval_metadata=`, so their report.json historically had NO `eval_metadata`
key at all -- silently breaking the mandatory holdout_hash comparability
check documented in CLAUDE.md's Run Contracts ("Compared runs must share
the same test cameras: verify report.json["eval_metadata"]["holdout_hash"]
matches").

Confirmed on two real completed runs:
  - issue116_partfocus_ctrl/comparison/iter_7000/report.json (a mid-training
    tick) had no "eval_metadata" key.
  - issue109exp_E1/comparison/iter_104000/report.json (the final report,
    which DOES pass eval_metadata explicitly) had it.

Fix: write_post_training_report() now derives eval_metadata from
`streaming_scene.eval_metadata()` -- the SAME method the final-report call
site already used -- whenever a caller passes `streaming_scene=` but not
`eval_metadata=` explicitly, rather than requiring every call site to
remember to pass it.
"""

from __future__ import annotations

import json
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


from eligsir.utils.comparison_report import write_post_training_report


class _FakeGaussian:
    def get_xyz(self):
        import torch
        return torch.zeros(0, 3)


class _FakePipe:
    pass


class _FakeFrame:
    def __init__(self, index):
        self.index = index


class _FakeCam:
    def __init__(self, index, image_name=None):
        if index is not None:
            self._streaming_frame = _FakeFrame(index)
        self.image_name = image_name
        self.uid = index
        self.original_image = None


class _FakeStreamingScene:
    """Stub mimicking StreamingScene.eval_metadata()'s real return shape
    (see scene/streaming_scene.py)."""

    def __init__(self, holdout_hash="deadbeef16"):
        self._holdout_hash = holdout_hash

    def eval_metadata(self) -> dict:
        return {
            "eval_hold": 8,
            "eval_min_baseline_m": 0.03,
            "eval_min_angle_deg": 2.0,
            "eval_guard": 0,
            "n_holdout": 3,
            "n_guard": 0,
            "holdout_hash": self._holdout_hash,
            "holdout_selection": "precomputed",
        }


def _minimal_render_fn(cam, gaussians, pipe, background):
    import torch
    return {"render": torch.zeros(3, 64, 64)}


def test_mid_training_style_call_gets_eval_metadata_from_streaming_scene():
    """Simulates train_streaming.py's mid-training report call site: passes
    streaming_scene= but NOT eval_metadata= explicitly (the exact bug)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_cams = [_FakeCam(i) for i in (0, 8, 16)]
        summary = write_post_training_report(
            model_path=tmpdir,
            iteration=7000,
            gaussians=_FakeGaussian(),
            train_cams=[],
            test_cams=test_cams,
            render_fn=_minimal_render_fn,
            pipe=_FakePipe(),
            background=None,
            streaming_scene=_FakeStreamingScene(),
        )
        assert "eval_metadata" in summary
        assert summary["eval_metadata"]["holdout_hash"] == "deadbeef16"

        report_path = os.path.join(tmpdir, "comparison", "iter_7000", "report.json")
        with open(report_path) as f:
            report_json = json.load(f)
        assert "eval_metadata" in report_json
        assert report_json["eval_metadata"]["holdout_hash"] == "deadbeef16"


def test_explicit_eval_metadata_still_wins_over_streaming_scene():
    """An explicitly-passed eval_metadata= (e.g. the final-report call site)
    must not be overridden by streaming_scene.eval_metadata()."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_cams = [_FakeCam(i) for i in (0, 8, 16)]
        summary = write_post_training_report(
            model_path=tmpdir,
            iteration=104000,
            gaussians=_FakeGaussian(),
            train_cams=[],
            test_cams=test_cams,
            render_fn=_minimal_render_fn,
            pipe=_FakePipe(),
            background=None,
            eval_metadata={"holdout_hash": "explicit_hash"},
            streaming_scene=_FakeStreamingScene(holdout_hash="should_not_be_used"),
        )
        assert summary["eval_metadata"]["holdout_hash"] == "explicit_hash"


def test_degenerate_split_does_not_synthesize_eval_metadata_from_streaming_scene():
    """Bootstrap-views debug reports (degenerate_split=True) must never emit
    anything that looks like a real holdout fingerprint -- same precedent as
    realized_holdout_hash's existing `not _degenerate` gating."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cams = [_FakeCam(i) for i in (0, 1, 2)]
        summary = write_post_training_report(
            model_path=tmpdir,
            iteration=0,
            gaussians=_FakeGaussian(),
            train_cams=cams,
            test_cams=cams,
            render_fn=_minimal_render_fn,
            pipe=_FakePipe(),
            background=None,
            degenerate_split=True,
            streaming_scene=_FakeStreamingScene(),
        )
        assert "eval_metadata" not in summary


def test_no_streaming_scene_no_holdout_hash_unchanged():
    """Non-streaming callers (e.g. train.py's legacy path) that pass neither
    eval_metadata= nor streaming_scene= see no behavior change: no
    holdout_hash appears (only realized_holdout_hash, from the pre-existing,
    unrelated issue #199 mechanism, may still be present)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_cams = [_FakeCam(i) for i in (0, 1)]
        summary = write_post_training_report(
            model_path=tmpdir,
            iteration=1000,
            gaussians=_FakeGaussian(),
            train_cams=[],
            test_cams=test_cams,
            render_fn=_minimal_render_fn,
            pipe=_FakePipe(),
            background=None,
        )
        assert "holdout_hash" not in (summary.get("eval_metadata") or {})
