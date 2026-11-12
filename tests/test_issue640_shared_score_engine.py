"""Shared #640 score-engine contract.

The post-hoc artifact evaluator and inline report writer must score the same
already-validated tensor pair through one implementation.  The engine owns
only metric sequencing; callers retain their existing renderer, device and
LPIPS-model ownership.
"""

from __future__ import annotations

import inspect
import json
import sys
import types

import pytest

torch = pytest.importorskip("torch")


def _canonical(tensor: torch.Tensor, context: str) -> torch.Tensor:
    from eligsir.utils.score_contract import declare_canonical_score_tensor

    return declare_canonical_score_tensor(tensor, context=context)[0]


def test_shared_engine_matches_the_canonical_psnr_formula_and_uses_injected_metrics():
    """One pair validation feeds every metric without a second score loop."""
    from eligsir.utils.score_contract import scientific_psnr, score_canonical_pair

    render = _canonical(torch.full((1, 3, 4, 5), 0.25), "shared-engine-render")
    gt = _canonical(torch.full((1, 3, 4, 5), 0.50), "shared-engine-gt")
    seen = []

    def ssim_fn(actual_render, actual_gt):
        seen.append(("ssim", actual_render, actual_gt))
        return torch.tensor(0.75)

    def lpips_fn(actual_render, actual_gt):
        seen.append(("lpips", actual_render, actual_gt))
        return torch.tensor(0.125)

    scores = score_canonical_pair(
        render,
        gt,
        ssim_fn=ssim_fn,
        lpips_fn=lpips_fn,
        context="shared-engine-test",
    )

    assert torch.equal(scores.psnr, scientific_psnr(render, gt, context="shared-engine-baseline"))
    assert scores.ssim.item() == pytest.approx(0.75)
    assert scores.lpips.item() == pytest.approx(0.125)
    assert [name for name, *_ in seen] == ["ssim", "lpips"]
    assert all(actual_render is render and actual_gt is gt for _, actual_render, actual_gt in seen)


def test_shared_engine_refuses_bad_pair_before_invoking_ssim_or_lpips():
    from eligsir.utils.score_contract import score_canonical_pair

    render = _canonical(torch.zeros((1, 3, 4, 5)), "shared-engine-bad-render")
    gt = _canonical(torch.zeros((1, 3, 4, 5)), "shared-engine-bad-gt")
    gt[0, 0, 0, 0] = float("nan")
    called = []

    with pytest.raises(ValueError, match=r"finite values in \[0, 1\]"):
        score_canonical_pair(
            render,
            gt,
            ssim_fn=lambda *_: called.append("ssim"),
            lpips_fn=lambda *_: called.append("lpips"),
            context="shared-engine-nonfinite",
        )

    assert called == []


def test_posthoc_and_inline_paths_route_metrics_through_the_shared_engine():
    import metrics
    import eligsir.utils.comparison_report as comparison_report

    for module in (metrics, comparison_report):
        source = inspect.getsource(module)
        assert "score_canonical_pair(" in source


def test_inline_and_posthoc_score_the_same_canonical_pair(tmp_path, monkeypatch):
    """The two public paths agree without a PNG or a second metric formula."""
    import metrics
    import eligsir.utils.comparison_report as comparison_report
    from eligsir.utils.loss_utils import ssim
    from eligsir.utils.score_contract import apply_declared_score_clamp_pair, score_canonical_pair

    render = torch.full((3, 64, 64), 0.25)
    gt = _canonical(torch.full((3, 64, 64), 0.50), "shared-engine-parity-gt")
    pair_render, pair_gt, _ = apply_declared_score_clamp_pair(
        render, gt, context="shared-engine-parity-direct"
    )
    direct = score_canonical_pair(
        pair_render,
        pair_gt,
        ssim_fn=ssim,
        lpips_fn=lambda *_: torch.tensor(0.125),
        context="shared-engine-parity-direct",
    )

    lpips_module = types.ModuleType("lpipsPyTorch")
    lpips_module.lpips = lambda *_args, **_kwargs: torch.tensor(0.125)
    monkeypatch.setitem(sys.modules, "lpipsPyTorch", lpips_module)

    method_dir = tmp_path / "posthoc" / "test" / "ours_1"
    stem = f"00000{metrics.SCORE_ARTIFACT_SUFFIX}"
    metrics.save_score_tensor(
        render,
        method_dir / metrics.SCORE_ARTIFACT_DIRNAME / "renders" / stem,
        kind="render",
        view_name=stem,
    )
    metrics.save_score_tensor(
        gt,
        method_dir / metrics.SCORE_ARTIFACT_DIRNAME / "gt" / stem,
        kind="gt",
        view_name=stem,
    )
    assert metrics.evaluate([str(tmp_path / "posthoc")], device="cpu") == []
    posthoc = json.loads((tmp_path / "posthoc" / "results.json").read_text())["ours_1"]

    class Camera:
        image_name = "parity"
        uid = 1
        original_image = gt

    class Gaussian:
        get_xyz = torch.zeros((0, 3))

    monkeypatch.setenv("SPLAT_REPORT_OVERVIEW", "0")
    monkeypatch.setenv("SPLAT_REPORT_VIDEOS", "0")
    monkeypatch.setattr(comparison_report, "_gt_image", lambda cam: cam.original_image)
    inline = comparison_report.write_post_training_report(
        model_path=str(tmp_path / "inline"),
        iteration=1,
        gaussians=Gaussian(),
        train_cams=[],
        test_cams=[Camera()],
        render_fn=lambda *_args: {"render": render},
        pipe=object(),
        background=None,
    )

    assert posthoc["PSNR"] == pytest.approx(float(direct.psnr.mean()))
    assert posthoc["SSIM"] == pytest.approx(float(direct.ssim))
    assert posthoc["LPIPS"] == pytest.approx(float(direct.lpips))
    assert inline["mean_test_psnr"] == pytest.approx(float(direct.psnr.mean()))
    assert inline["mean_test_lpips"] == pytest.approx(float(direct.lpips))
