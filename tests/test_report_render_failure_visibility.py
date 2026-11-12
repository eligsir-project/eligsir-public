"""Issue #437: a swallowed per-camera render failure must be VISIBLE in report.json.

`write_post_training_report`'s metrics loop catches per-camera render failures
and skips the camera (deliberately -- one bad camera must not cost the whole
report). But nothing recorded that it happened, so `n_test_metrics` shrank
silently: "scored 128 of 315 because --streaming_report_test_metrics_max
subsampled the pool" and "scored 128 of 315 because 187 renders raised" were
indistinguishable to every downstream consumer. Same swallow-and-continue
mechanism issue #433/#420 documented from the comparability side.

The fix records `n_<split>_metrics_attempted` / `n_<split>_metrics_failed`
alongside `n_<split>_metrics`, plus a `<split>_metrics_render_failures` detail
block naming the cameras.

ENVIRONMENT NOTE (this is why the assertions below are relational, not
absolute): on a host without a working CUDA/HIP device, `_gt_image()` raises
for EVERY camera (that is #437's own root-cause finding), so `n_test_metrics`
is 0 here and equal to the attempted count inside the container. Asserting
absolute counts would therefore pass in exactly one of the two environments.
The invariant
    n_test_metrics + n_test_metrics_failed == n_test_metrics_attempted
holds in both, and IS the deliverable -- it is precisely what lets a consumer
attribute a shrunken scored count to failures rather than to subsampling.
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


class _FakeCam:
    def __init__(self, name, report_identity=None):
        import torch
        self.image_name = name
        self.original_image = torch.zeros(3, 64, 64)
        if report_identity is not None:
            self._report_source_identity = report_identity
        from eligsir.utils.score_contract import declare_camera_score_provenance

        declare_camera_score_provenance(self, context="test_report_gt")


class _NamelessCam:
    """A camera with no `image_name`.

    Before this fix the except-branch's own `print(f"... {cam.image_name}")`
    raised AttributeError *inside the handler*, escaping the metrics loop and
    killing the entire report.
    """

    def __init__(self):
        import torch
        self.original_image = torch.zeros(3, 64, 64)


def _ok_render_fn(cam, gaussians, pipe, background):
    import torch
    from eligsir.utils.score_contract import declare_canonical_score_tensor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    image, _ = declare_canonical_score_tensor(
        torch.zeros(3, 64, 64, device=device), context="test_report_render"
    )
    return {"render": image}


def _render_fn_failing_on(names):
    def _fn(cam, gaussians, pipe, background):
        if str(getattr(cam, "image_name", "")) in names:
            raise RuntimeError(f"synthetic render failure for {getattr(cam, 'image_name', '?')}")
        return _ok_render_fn(cam, gaussians, pipe, background)
    return _fn


def _report(**kwargs):
    with tempfile.TemporaryDirectory() as tmpdir:
        return write_post_training_report(
            model_path=tmpdir,
            iteration=5000,
            gaussians=_FakeGaussian(),
            pipe=_FakePipe(),
            background=None,
            **kwargs,
        )


def _report_and_file(**kwargs):
    """Return (returned summary, parsed on-disk report.json).

    Read INSIDE the TemporaryDirectory context -- `_report()` above tears the
    directory down before returning, so the file is gone by then.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        summary = write_post_training_report(
            model_path=tmpdir,
            iteration=5000,
            gaussians=_FakeGaussian(),
            pipe=_FakePipe(),
            background=None,
            **kwargs,
        )
        with open(os.path.join(summary["output_dir"], "report.json")) as f:
            return summary, json.load(f)


def test_fields_reach_report_json_on_disk(monkeypatch):
    """The deliverable is visibility in report.json, not in the returned dict.

    write_post_training_report currently `json.dump(summary)`s verbatim, but
    that is an implementation detail -- this asserts the contract directly, so a
    future serialization step that filters or rebuilds the dict cannot silently
    drop these fields while the other tests stay green.
    """
    # This assertion needs at least one successfully scored view.  The
    # production helper deliberately materializes GT on CUDA, so make this
    # fixture's score path explicit and CPU-capable instead of depending on
    # another test's leaked device monkeypatch.
    from eligsir.utils.score_contract import score_tensor_from_camera

    monkeypatch.setattr(
        "utils.comparison_report._gt_image",
        lambda cam: score_tensor_from_camera(cam, device="cpu"),
    )

    summary, on_disk = _report_and_file(
        train_cams=[_FakeCam("a")],
        test_cams=[_FakeCam(f"t{i}") for i in range(4)],
        render_fn=_render_fn_failing_on({"t1"}),
    )
    for key in (
        "n_test_metrics",
        "n_test_metrics_attempted",
        "n_test_metrics_failed",
        "n_train_metrics_attempted",
        "n_train_metrics_failed",
    ):
        assert key in on_disk, f"{key} missing from report.json"
        assert on_disk[key] == summary[key]
    assert (
        on_disk["n_test_metrics"] + on_disk["n_test_metrics_failed"]
        == on_disk["n_test_metrics_attempted"]
    )
    assert on_disk["n_test_metrics_failed"] >= 1
    assert "t1" in on_disk["test_metrics_render_failures"]["failed_cameras"]
    score_provenance = on_disk["score_provenance"]["test"]
    assert score_provenance["contract_version"] == 1
    assert score_provenance["domain"] == "float32_0_1"
    assert score_provenance["resolution"] == [64, 64]
    assert score_provenance["render"]["steps"][0]["type"] == "declared_score_transform"
    assert score_provenance["gt"]["steps"] == []


def test_train_per_view_persists_identity_and_keeps_failed_render_out(monkeypatch):
    from eligsir.contracts.runtime import SourceIdentity

    monkeypatch.setattr(
        "utils.comparison_report._gt_image",
        lambda cam: __import__("utils.score_contract", fromlist=["score_tensor_from_camera"])
        .score_tensor_from_camera(cam, device="cpu"),
    )
    known = SourceIdentity("packet-source", 7, 3, 3_000_000)
    summary, on_disk = _report_and_file(
        train_cams=[
            _FakeCam("train-known", known),
            _FakeCam("train-failed", known),
            _FakeCam("train-unknown"),
        ],
        test_cams=[],
        render_fn=_render_fn_failing_on({"train-failed"}),
        train_metrics_max=3,
        mapping_phase_source="causal_source_completion_boundary",
    )
    rows = on_disk["train_per_view"]
    assert len(rows) == on_disk["n_train_metrics"] == 2
    assert rows[0]["iteration"] == 5000
    assert rows[0]["mapping_phase_source"] == "causal_source_completion_boundary"
    assert rows[0]["identity"]["status"] == "known"
    assert rows[0]["identity"]["fingerprint"] == known.fingerprint()
    assert rows[0]["identity"]["source_ordinal"] == 3
    assert rows[0]["identity"]["timestamp_ns"] == 3_000_000
    assert rows[1]["identity"]["status"] == "unknown"
    assert rows[1]["identity"]["reason"] == "report_source_identity_missing"
    assert all(row["view_name"] != "train-failed" for row in rows)
    assert summary["train_metric_binding"] == {
        "iteration": 5000,
        "mapping_phase_source": "causal_source_completion_boundary",
    }


def test_train_metric_binding_preserves_nonendpoint_phase(monkeypatch):
    monkeypatch.setattr(
        "utils.comparison_report._gt_image",
        lambda cam: __import__("utils.score_contract", fromlist=["score_tensor_from_camera"])
        .score_tensor_from_camera(cam, device="cpu"),
    )
    summary, on_disk = _report_and_file(
        train_cams=[_FakeCam("train-live")],
        test_cams=[],
        render_fn=_ok_render_fn,
        mapping_phase_source="live",
    )
    assert summary["train_metric_binding"]["mapping_phase_source"] == "live"
    assert on_disk["train_per_view"][0]["mapping_phase_source"] == "live"


def test_accounting_invariant_holds_for_test_split():
    """scored + failed == attempted, on any host."""
    summary = _report(
        train_cams=[_FakeCam("a")],
        test_cams=[_FakeCam(f"t{i}") for i in range(5)],
        render_fn=_ok_render_fn,
    )
    assert (
        summary["n_test_metrics"] + summary["n_test_metrics_failed"]
        == summary["n_test_metrics_attempted"]
    )
    assert (
        summary["n_train_metrics"] + summary["n_train_metrics_failed"]
        == summary["n_train_metrics_attempted"]
    )


def test_attempted_reflects_the_cap_not_the_pool():
    """Subsampling shows up as a REDUCED `attempted`, never as a failure.

    This is the distinction the fix exists to make: with a cap of 2 over a pool
    of 5, only 2 cameras are attempted, so a low scored count is attributable to
    subsampling (`test_metrics_subsampled` True, attempted == 2) and not to
    render failures.
    """
    summary = _report(
        train_cams=[_FakeCam("a")],
        test_cams=[_FakeCam(f"t{i}") for i in range(5)],
        render_fn=_ok_render_fn,
        test_metrics_max=2,
    )
    assert summary["n_test"] == 5
    assert summary["test_metrics_subsampled"] is True
    assert summary["n_test_metrics_attempted"] == 2
    assert (
        summary["n_test_metrics"] + summary["n_test_metrics_failed"]
        == summary["n_test_metrics_attempted"]
    )


def test_injected_render_failure_is_named_in_the_report():
    """An injected failure on a specific camera is recorded by name.

    Environment-independent: on a CUDA-less host every camera fails, so this
    asserts t1 is AMONG the failures (and that the detail block is
    self-consistent), not that it is the only one.
    """
    summary = _report(
        train_cams=[_FakeCam("a")],
        test_cams=[_FakeCam(f"t{i}") for i in range(4)],
        render_fn=_render_fn_failing_on({"t1"}),
    )
    assert summary["n_test_metrics_failed"] >= 1
    block = summary["test_metrics_render_failures"]
    assert "t1" in block["failed_cameras"]
    assert block["n_failed"] == summary["n_test_metrics_failed"]
    assert block["n_scored"] == summary["n_test_metrics"]
    assert block["n_attempted"] == summary["n_test_metrics_attempted"]
    # error_types is a {type: count} histogram summing to the failure count,
    # not one stored string per failed camera.
    assert sum(block["error_types"].values()) == block["n_failed"]
    assert isinstance(block["example_error"], str) and block["example_error"]


def test_no_failure_block_when_nothing_failed():
    """The detail block is written only when there is something to report.

    Skipped where CUDA is unavailable, because #437's root cause makes every
    camera fail there and a zero-failure report is unreachable.
    """
    import torch
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/HIP device: every camera fails at _gt_image (issue #437)")
    summary = _report(
        train_cams=[_FakeCam("a")],
        test_cams=[_FakeCam(f"t{i}") for i in range(3)],
        render_fn=_ok_render_fn,
    )
    assert summary["n_test_metrics_failed"] == 0
    assert "test_metrics_render_failures" not in summary
    assert summary["n_test_metrics_attempted"] == 3
    assert summary["n_test_metrics"] == 3


def test_camera_without_image_name_does_not_kill_the_report():
    """Regression: the except-branch used to raise AttributeError itself."""
    summary = _report(
        train_cams=[_FakeCam("a")],
        test_cams=[_NamelessCam(), _FakeCam("t1")],
        render_fn=_render_fn_failing_on({"", "t1"}),
    )
    # Both cameras fail; the report is still produced and both are accounted for.
    assert summary["n_test_metrics_failed"] == 2
    assert summary["n_test_metrics_attempted"] == 2
    block = summary["test_metrics_render_failures"]
    assert any(name.startswith("idx_") for name in block["failed_cameras"])
