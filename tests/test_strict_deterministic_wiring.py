"""Tests for audit finding §14: determinism mode exists but is unselectable/unauditable.

`utils/general_utils.py`'s `safe_state(..., strict_deterministic=False)` already
implements a real two-mode split -- `strict_deterministic=True` forces
`torch.use_deterministic_algorithms(True)` plus cudnn benchmark-off/
deterministic-on -- but before this change there was no CLI flag anywhere
that let a caller actually select `strict_deterministic=True` when launching
`train.py`, and the run's `report.json` never recorded which mode it
actually ran under, so even editing the source default wouldn't have been
auditable after the fact from a run's own artifacts.

This module tests the three parts of the fix:
  1. `arguments.ModelParams` gained a `--strict_deterministic` /
     `--no-strict_deterministic` CLI flag (default `False`, preserving the
     pre-existing behavior exactly).
  2. `train.py`'s sole `safe_state(...)` call site threads
     `args.strict_deterministic` through to `safe_state`'s
     `strict_deterministic` parameter (source-inspection, mirroring this
     repo's other CLI-to-call-site wiring tests, e.g.
     `test_gpu_frame_prep_parity.py::test_flag_default_off`).
  3. `utils.comparison_report.determinism_from_args()` builds the
     `report.json["determinism"]` block, and `write_post_training_report()`
     records it -- same pattern as `git_provenance_from_args()` /
     `report.json["git_provenance"]` (issue #259).
"""

from __future__ import annotations

import json
import sys
import tempfile
from argparse import ArgumentParser
from types import SimpleNamespace
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


from eligsir.utils.comparison_report import determinism_from_args, write_post_training_report


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


# --- 1. CLI flag: arguments.ModelParams.strict_deterministic ---------------


def test_strict_deterministic_flag_default_off():
    """--strict_deterministic must default to False (regression-safe: this
    preserves the exact pre-existing safe_state() behavior)."""
    from arguments import ModelParams

    parser = ArgumentParser()
    lp = ModelParams(parser)
    args = parser.parse_args([])
    extracted = lp.extract(args)
    assert getattr(extracted, "strict_deterministic") is False


def test_strict_deterministic_flag_can_be_enabled():
    """--strict_deterministic sets the resolved value to True."""
    from arguments import ModelParams

    parser = ArgumentParser()
    lp = ModelParams(parser)
    args = parser.parse_args(["--strict_deterministic"])
    extracted = lp.extract(args)
    assert getattr(extracted, "strict_deterministic") is True


def test_strict_deterministic_flag_has_negation():
    """Follows this repo's --no-<flag> BooleanOptionalAction convention
    (CLAUDE.md's --no-gsplat_sparse_grad precedent) -- explicit negation
    must parse cleanly even though False is already the default."""
    from arguments import ModelParams

    parser = ArgumentParser()
    lp = ModelParams(parser)
    args = parser.parse_args(["--no-strict_deterministic"])
    extracted = lp.extract(args)
    assert getattr(extracted, "strict_deterministic") is False


# --- 2. train.py's safe_state(...) call site is wired ----------------------


def test_train_py_threads_strict_deterministic_into_safe_state():
    """Source-inspection wiring check (this repo's existing convention for
    CLI-to-call-site threading, e.g. test_gpu_frame_prep_parity.py). Import-
    executing train.py needs a full CUDA/gsplat environment this suite does
    not assume, so this asserts the call site text itself passes
    strict_deterministic through rather than importing and running it."""
    import pathlib

    train_py = pathlib.Path(__file__).resolve().parents[1] / "train.py"
    text = train_py.read_text()
    assert "safe_state(" in text, "train.py no longer calls safe_state -- update this test"
    # Find the safe_state( call and confirm strict_deterministic is threaded
    # through from args within a small window after the call opens.
    idx = text.index("safe_state(")
    window = text[idx: idx + 400]
    assert "strict_deterministic" in window, (
        "train.py's safe_state(...) call site does not thread "
        "strict_deterministic through -- audit §14 regression"
    )
    assert "args.strict_deterministic" in window or "getattr(args, \"strict_deterministic\"" in window


# --- 3. determinism_from_args() ---------------------------------------------


def test_determinism_from_args_extracts_true():
    args = SimpleNamespace(strict_deterministic=True)
    assert determinism_from_args(args) == {"strict_deterministic": True}


def test_determinism_from_args_extracts_false():
    args = SimpleNamespace(strict_deterministic=False)
    assert determinism_from_args(args) == {"strict_deterministic": False}


def test_determinism_from_args_none_input_returns_none():
    assert determinism_from_args(None) is None


def test_determinism_from_args_missing_attr_returns_none():
    """A hand-built args namespace that never went through arguments.py
    (e.g. an older call path) must not produce a misleading default --
    degrade to None so the caller's pre-existing behavior is preserved."""
    args = SimpleNamespace(model_path="foo")
    assert determinism_from_args(args) is None


# --- write_post_training_report(determinism=...) ----------------------------


def test_write_post_training_report_records_determinism_true():
    with tempfile.TemporaryDirectory() as tmpdir:
        summary = write_post_training_report(
            model_path=tmpdir,
            iteration=100,
            gaussians=_FakeGaussian(),
            train_cams=[_FakeCam("a")],
            test_cams=[_FakeCam("t")],
            render_fn=_minimal_render_fn,
            pipe=_FakePipe(),
            background=None,
            determinism={"strict_deterministic": True},
        )
        assert summary["determinism"] == {"strict_deterministic": True}

        report_path = f"{tmpdir}/comparison/iter_100/report.json"
        with open(report_path) as f:
            on_disk = json.load(f)
        assert on_disk["determinism"] == {"strict_deterministic": True}


def test_write_post_training_report_records_determinism_false():
    with tempfile.TemporaryDirectory() as tmpdir:
        summary = write_post_training_report(
            model_path=tmpdir,
            iteration=100,
            gaussians=_FakeGaussian(),
            train_cams=[_FakeCam("a")],
            test_cams=[_FakeCam("t")],
            render_fn=_minimal_render_fn,
            pipe=_FakePipe(),
            background=None,
            determinism=determinism_from_args(SimpleNamespace(strict_deterministic=False)),
        )
        assert summary["determinism"] == {"strict_deterministic": False}

        report_path = f"{tmpdir}/comparison/iter_100/report.json"
        with open(report_path) as f:
            on_disk = json.load(f)
        assert on_disk["determinism"] == {"strict_deterministic": False}


def test_write_post_training_report_omits_determinism_when_not_provided():
    """Omitting the kwarg (every call site before this feature existed)
    reproduces the exact report.json from before this parameter existed --
    no 'determinism' key at all, not a null placeholder."""
    with tempfile.TemporaryDirectory() as tmpdir:
        summary = write_post_training_report(
            model_path=tmpdir,
            iteration=100,
            gaussians=_FakeGaussian(),
            train_cams=[_FakeCam("a")],
            test_cams=[_FakeCam("t")],
            render_fn=_minimal_render_fn,
            pipe=_FakePipe(),
            background=None,
        )
        assert "determinism" not in summary


# --- 4. safe_state() itself: sanity check the two-mode split still exists --


def test_safe_state_strict_deterministic_true_calls_use_deterministic_algorithms(monkeypatch):
    import torch

    from eligsir.utils.general_utils import safe_state

    calls = []
    monkeypatch.setattr(torch, "use_deterministic_algorithms", lambda v: calls.append(v))
    safe_state(True, seed=1, strict_deterministic=True)
    assert calls == [True]


def test_safe_state_strict_deterministic_false_does_not_call_use_deterministic_algorithms(monkeypatch):
    import torch

    from eligsir.utils.general_utils import safe_state

    calls = []
    monkeypatch.setattr(torch, "use_deterministic_algorithms", lambda v: calls.append(v))
    safe_state(True, seed=1, strict_deterministic=False)
    assert calls == []
