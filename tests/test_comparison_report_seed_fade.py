"""Tests for seed_fade kwarg in write_post_training_report()."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True, scope="module")
def _mock_lpips_import():
    """Mock the (heavy, GPU-model-loading) lpipsPyTorch import so this
    module's tests don't need to load the real LPIPS network just to test
    JSON report writing.

    MUST restore sys.modules afterward: a bare `sys.modules['lpipsPyTorch']
    = MagicMock()` at module level (the previous version of this fixture)
    permanently replaced the real module in the process-global cache for
    the rest of the pytest session -- any LATER test that imports the real
    lpipsPyTorch (e.g. tests/regression/test_golden_render.py) silently got
    the MagicMock instead, producing a nonsensical LPIPS distance and a
    spurious failure with no connection to the actual code under test.
    Found via bisection across the full suite; test order/collection
    matters here specifically because `sys.modules` is never restored.
    """
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
    """Minimal gaussian mock for testing."""
    def get_xyz(self):
        import torch
        return torch.zeros(0, 3)


class _FakePipe:
    """Minimal pipe mock."""
    pass


def _minimal_render_fn(cam, gaussians, pipe, background):
    """Minimal render function that returns a valid render dict."""
    import torch
    return {"render": torch.zeros(3, 64, 64)}


def test_seed_fade_kwarg_included_when_provided():
    """Verify seed_fade dict is included in summary when provided."""
    with tempfile.TemporaryDirectory() as tmpdir:
        seed_fade_block = {
            "detected": True,
            "max_dip_db": 15.0,
            "fade_start_iter": 300,
            "recovered": True,
            "controller_engaged": True,
        }
        summary = write_post_training_report(
            model_path=tmpdir,
            iteration=5000,
            gaussians=_FakeGaussian(),
            train_cams=[],
            test_cams=[],
            render_fn=_minimal_render_fn,
            pipe=_FakePipe(),
            background=None,
            seed_fade=seed_fade_block,
        )
        assert "seed_fade" in summary
        # Issue #346: the block is now GRADED on the way out -- every original
        # detector key survives verbatim, with severity/confounded/curve-shape
        # fields added alongside. Asserting a superset (not equality) is the
        # deliberate semantics change; nothing the detector emitted is lost.
        for k, v in seed_fade_block.items():
            assert summary["seed_fade"][k] == v
        assert summary["seed_fade"]["severity"] == "severe"  # max_dip_db=15.0 >= 12.0
        assert summary["seed_fade"]["confounded"] is True
        assert summary["seed_fade_confounded"] is True


def test_seed_fade_key_absent_when_not_provided():
    """Verify seed_fade key is absent from summary when not provided."""
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
        assert "seed_fade" not in summary


def test_seed_fade_none_treated_as_absent():
    """Verify seed_fade=None results in key not being present."""
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
            seed_fade=None,
        )
        assert "seed_fade" not in summary
        # Issue #346: the top-level confounded mirror is gated on the same
        # condition -- a run that passes no seed_fade block gets a report.json
        # byte-identical to the pre-#346 one.
        assert "seed_fade_confounded" not in summary


def test_disabled_seed_lineage_freeze_does_not_attach_or_emit_report_block():
    from types import SimpleNamespace
    from eligsir.utils.seed_lineage_freeze import SeedLineageFreeze

    gaussians = _FakeGaussian()
    SeedLineageFreeze(
        gaussians,
        SimpleNamespace(streaming_seed_lineage_freeze_from_iter=-1),
    )
    assert not hasattr(gaussians, "_seed_lineage_freeze")
    with tempfile.TemporaryDirectory() as tmpdir:
        summary = write_post_training_report(
            model_path=tmpdir,
            iteration=5000,
            gaussians=gaussians,
            train_cams=[],
            test_cams=[],
            render_fn=_minimal_render_fn,
            pipe=_FakePipe(),
            background=None,
        )
    assert "seed_lineage_freeze" not in summary


def test_seed_fade_preserved_in_report_json():
    """Verify seed_fade is written to report.json file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        seed_fade_block = {
            "detected": True,
            "max_dip_db": 15.0,
            "fade_start_iter": 300,
            "recovered": True,
            "controller_engaged": True,
        }
        summary = write_post_training_report(
            model_path=tmpdir,
            iteration=5000,
            gaussians=_FakeGaussian(),
            train_cams=[],
            test_cams=[],
            render_fn=_minimal_render_fn,
            pipe=_FakePipe(),
            background=None,
            seed_fade=seed_fade_block,
        )
        # Read the report.json file
        report_path = os.path.join(tmpdir, "comparison", "iter_5000", "report.json")
        assert os.path.exists(report_path)
        with open(report_path, "r") as f:
            report_json = json.load(f)
        assert "seed_fade" in report_json
        for k, v in seed_fade_block.items():
            assert report_json["seed_fade"][k] == v
        # Issue #346: machine-readable verdict, in the file, top-level too.
        assert report_json["seed_fade"]["severity"] == "severe"
        assert report_json["seed_fade_confounded"] is True


def test_baseline_run_absent_when_not_provided():
    """--baseline_run is strictly opt-in; behavior-neutral when unset."""
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
        assert "baseline_comparison" not in summary


def test_baseline_run_wired_through_to_summary_and_report_json():
    with tempfile.TemporaryDirectory() as tmpdir:
        baseline_dir = os.path.join(tmpdir, "baseline")
        os.makedirs(baseline_dir)
        with open(os.path.join(baseline_dir, "report.json"), "w") as f:
            json.dump({"mean_test_psnr": 22.44, "eval_valid": True, "n_test": 10}, f)

        current_dir = os.path.join(tmpdir, "current")
        os.makedirs(current_dir)

        summary = write_post_training_report(
            model_path=current_dir,
            iteration=5000,
            gaussians=_FakeGaussian(),
            train_cams=[],
            test_cams=[],
            render_fn=_minimal_render_fn,
            pipe=_FakePipe(),
            background=None,
            baseline_run=baseline_dir,
        )
        assert "baseline_comparison" in summary
        assert summary["baseline_comparison"]["baseline_run"] == baseline_dir

        report_path = os.path.join(current_dir, "comparison", "iter_5000", "report.json")
        with open(report_path) as f:
            report_json = json.load(f)
        assert "baseline_comparison" in report_json
