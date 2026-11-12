"""Tests for realized_holdout_hash wiring into write_post_training_report()
(GitHub issue #199: holdout_hash matches the holdout SELECTION RULE, not the
test set actually realized/evaluated -- two runs of the same scene at
different iteration budgets can share holdout_hash while realizing
different-sized, differently-composed test sets).
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


from eligsir.utils.comparison_report import write_post_training_report, _realized_holdout_hash


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
    """Minimal camera stub carrying a _streaming_frame like a real
    streaming-scene test camera does at report time."""
    def __init__(self, index, image_name=None):
        if index is not None:
            self._streaming_frame = _FakeFrame(index)
        self.image_name = image_name
        self.uid = index
        self.original_image = None


def _minimal_render_fn(cam, gaussians, pipe, background):
    import torch
    return {"render": torch.zeros(3, 64, 64)}


def test_realized_holdout_hash_written_into_eval_metadata():
    with tempfile.TemporaryDirectory() as tmpdir:
        test_cams = [_FakeCam(i) for i in (0, 4, 8, 12)]
        summary = write_post_training_report(
            model_path=tmpdir,
            iteration=5000,
            gaussians=_FakeGaussian(),
            train_cams=[],
            test_cams=test_cams,
            render_fn=_minimal_render_fn,
            pipe=_FakePipe(),
            background=None,
            eval_metadata={"holdout_hash": "abc123", "n_holdout": 4},
        )
        assert "realized_holdout_hash" in summary["eval_metadata"]
        assert summary["eval_metadata"]["realized_holdout_hash"] == _realized_holdout_hash(test_cams)
        assert summary["eval_metadata"]["holdout_hash"] == "abc123"  # untouched


def test_realized_holdout_hash_differs_when_realized_set_differs_at_equal_hash():
    """The exact issue #199 mechanism: same selection-rule holdout_hash
    (same eval_metadata passed in), but the run only realized a SUBSET of
    the reserved holdout (e.g. a shorter run) -- realized_holdout_hash must
    differ even though holdout_hash is identical."""
    with tempfile.TemporaryDirectory() as tmpdir_a, tempfile.TemporaryDirectory() as tmpdir_b:
        short_run_cams = [_FakeCam(i) for i in (0, 4, 8)]     # reached fewer frames
        long_run_cams = [_FakeCam(i) for i in (0, 4, 8, 12, 16)]  # reached more frames
        shared_eval_metadata = {"holdout_hash": "same_selection_rule", "n_holdout": 5}

        summary_short = write_post_training_report(
            model_path=tmpdir_a, iteration=4000, gaussians=_FakeGaussian(),
            train_cams=[], test_cams=short_run_cams, render_fn=_minimal_render_fn,
            pipe=_FakePipe(), background=None, eval_metadata=dict(shared_eval_metadata),
        )
        summary_long = write_post_training_report(
            model_path=tmpdir_b, iteration=8000, gaussians=_FakeGaussian(),
            train_cams=[], test_cams=long_run_cams, render_fn=_minimal_render_fn,
            pipe=_FakePipe(), background=None, eval_metadata=dict(shared_eval_metadata),
        )
        assert summary_short["eval_metadata"]["holdout_hash"] == summary_long["eval_metadata"]["holdout_hash"]
        assert summary_short["n_test"] != summary_long["n_test"]
        assert (
            summary_short["eval_metadata"]["realized_holdout_hash"]
            != summary_long["eval_metadata"]["realized_holdout_hash"]
        )


def test_realized_holdout_hash_absent_when_no_test_cams():
    with tempfile.TemporaryDirectory() as tmpdir:
        summary = write_post_training_report(
            model_path=tmpdir,
            iteration=5000,
            gaussians=_FakeGaussian(),
            train_cams=[],
            test_cams=[],
            render_fn=_minimal_render_fn,
            pipe=_FakePipe(),
            background=None,
        )
        assert "realized_holdout_hash" not in (summary.get("eval_metadata") or {})


def test_realized_holdout_hash_persisted_in_report_json():
    with tempfile.TemporaryDirectory() as tmpdir:
        test_cams = [_FakeCam(i) for i in (1, 2, 3)]
        write_post_training_report(
            model_path=tmpdir,
            iteration=5000,
            gaussians=_FakeGaussian(),
            train_cams=[],
            test_cams=test_cams,
            render_fn=_minimal_render_fn,
            pipe=_FakePipe(),
            background=None,
            eval_metadata={"holdout_hash": "xyz"},
        )
        report_path = os.path.join(tmpdir, "comparison", "iter_5000", "report.json")
        with open(report_path) as f:
            report_json = json.load(f)
        assert "realized_holdout_hash" in report_json["eval_metadata"]
