"""Issue #1302: write_live_eval_report's kwarg-building contract.

Uses an injected ``report_writer`` stub so this exercises the real call
shape (train_cams=[], eval_metadata/source_attestation passthrough, the
empty-test_cams guard) without importing torch or
utils.comparison_report -- keeps this file collectible in the lean CPU CI
image per tests/conftest.py's ``_NEEDS_TORCH`` contract.
"""

from __future__ import annotations

import pytest

from eligsir.runtime.live_eval_holdout import write_live_eval_report


class _FakeCam:
    def __init__(self, name: str) -> None:
        self.image_name = name


def test_raises_on_empty_test_cams():
    with pytest.raises(ValueError):
        write_live_eval_report(
            model_path="/tmp/does-not-matter",
            iteration=100,
            gaussians=object(),
            pipe=object(),
            background=object(),
            render_fn=lambda *a, **k: None,
            test_cams=[],
            eval_metadata={"eval_hold": 8},
            report_writer=lambda *a, **k: {"should": "never run"},
        )


def test_calls_report_writer_with_expected_positional_and_kwargs():
    captured = {}

    def _fake_writer(model_path, iteration, gaussians, train_cams, test_cams,
                      render_fn, pipe, background, **kwargs):
        captured["model_path"] = model_path
        captured["iteration"] = iteration
        captured["gaussians"] = gaussians
        captured["train_cams"] = train_cams
        captured["test_cams"] = test_cams
        captured["render_fn"] = render_fn
        captured["pipe"] = pipe
        captured["background"] = background
        captured["kwargs"] = kwargs
        return {"eval_valid": True, "mean_test_psnr": 27.5}

    gaussians = object()
    pipe = object()
    background = object()
    render_fn = lambda *a, **k: None
    test_cams = [_FakeCam("holdout_0"), _FakeCam("holdout_8")]
    eval_metadata = {"eval_hold": 8, "holdout_selection": "online_every_nth_ordinal_rule"}
    source_attestation = {"status": "unvalidated_source_override"}

    result = write_live_eval_report(
        model_path="/tmp/floor2_run",
        iteration=252,
        gaussians=gaussians,
        pipe=pipe,
        background=background,
        render_fn=render_fn,
        test_cams=test_cams,
        eval_metadata=eval_metadata,
        source_attestation=source_attestation,
        report_writer=_fake_writer,
        subdir="live_eval_iter_252",
    )

    assert result == {"eval_valid": True, "mean_test_psnr": 27.5}
    assert captured["model_path"] == "/tmp/floor2_run"
    assert captured["iteration"] == 252
    assert captured["gaussians"] is gaussians
    assert captured["train_cams"] == []  # default: no persistent train-camera list
    assert captured["test_cams"] == test_cams
    assert captured["render_fn"] is render_fn
    assert captured["pipe"] is pipe
    assert captured["background"] is background
    assert captured["kwargs"]["eval_metadata"] == eval_metadata
    assert captured["kwargs"]["source_attestation"] == source_attestation
    assert captured["kwargs"]["subdir"] == "live_eval_iter_252"


def test_train_cams_passthrough_when_caller_has_some():
    captured = {}

    def _fake_writer(model_path, iteration, gaussians, train_cams, test_cams,
                      render_fn, pipe, background, **kwargs):
        captured["train_cams"] = train_cams
        return {}

    train_cams = [_FakeCam("train_1")]
    write_live_eval_report(
        model_path="/tmp/x",
        iteration=1,
        gaussians=object(),
        pipe=object(),
        background=object(),
        render_fn=lambda *a, **k: None,
        test_cams=[_FakeCam("holdout_0")],
        eval_metadata={"eval_hold": 8},
        train_cams=train_cams,
        report_writer=_fake_writer,
    )
    assert captured["train_cams"] == train_cams
