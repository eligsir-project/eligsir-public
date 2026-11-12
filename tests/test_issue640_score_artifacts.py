"""Issue #640: the render -> 8-bit PNG -> metrics round trip is retired.

These tests pin the exact-score-artifact contract that replaced it:

* an exact tensor round-trips bit-for-bit through the persistence layer,
  where an 8-bit image round trip does not (and measurably changes PSNR);
* corrupt/mismatched artifacts are refused rather than silently scored;
* the score-domain clamp is an explicit, recorded transform;
* ``metrics.py`` refuses a legacy PNG-only run directory instead of
  falling back to scoring 8-bit data.

CPU-only: no CUDA, no LPIPS, no torchvision.
"""

from __future__ import annotations

import json
import sys
import types

import pytest

torch = pytest.importorskip("torch")

import metrics  # noqa: E402


def _quantize_like_png(tensor: torch.Tensor) -> torch.Tensor:
    """What ``torchvision.utils.save_image`` + PIL reload does to samples.

    save_image clamps to [0, 1], scales by 255 and rounds to uint8; the
    reader divides by 255 again. PNG compression itself is lossless -- the
    loss happens before it.
    """
    return (tensor.clamp(0.0, 1.0) * 255).round().to(torch.uint8).to(torch.float32) / 255.0


def _sample_image(seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.rand(3, 16, 24, generator=g, dtype=torch.float32)


def _declared(tensor: torch.Tensor) -> torch.Tensor:
    from eligsir.utils.score_contract import declare_canonical_score_tensor

    return declare_canonical_score_tensor(tensor, context="issue640:test")[0]


def test_score_tensor_round_trip_is_bit_exact(tmp_path):
    image = _sample_image()
    path = tmp_path / "00000.pt"

    metrics.save_score_tensor(image, path, kind="render", view_name="00000")
    loaded, meta = metrics.load_score_tensor(path)

    assert torch.equal(loaded, image)
    assert meta["dtype"] == "torch.float32"
    assert meta["shape"] == list(image.shape)
    assert meta["clamp_applied_at_save"] is False


def test_png_round_trip_is_not_bit_exact_and_changes_psnr():
    """The defect this issue retires, demonstrated numerically."""
    from eligsir.utils.image_utils import psnr

    render = _sample_image(seed=1)
    gt = _sample_image(seed=2)

    exact_psnr = psnr(_declared(render.unsqueeze(0)), _declared(gt.unsqueeze(0))).mean().item()
    png_psnr = psnr(
        _declared(_quantize_like_png(render).unsqueeze(0)),
        _declared(_quantize_like_png(gt).unsqueeze(0)),
    ).mean().item()

    assert not torch.equal(_quantize_like_png(render), render)
    assert exact_psnr != pytest.approx(png_psnr, abs=1e-9)


def test_out_of_range_samples_survive_persistence_but_png_would_clamp(tmp_path):
    """Renderer output outside [0, 1] is preserved exactly on disk."""
    image = _sample_image(seed=3)
    image[0, 0, 0] = -0.25
    image[0, 0, 1] = 1.75
    path = tmp_path / "00000.pt"

    metrics.save_score_tensor(image, path, kind="render", view_name="00000")
    loaded, meta = metrics.load_score_tensor(path)

    assert torch.equal(loaded, image)
    assert meta["value_min"] == pytest.approx(-0.25)
    assert meta["value_max"] == pytest.approx(1.75)
    # The old PNG path would have destroyed both samples silently.
    quantized = _quantize_like_png(image)
    assert quantized[0, 0, 0].item() == 0.0
    assert quantized[0, 0, 1].item() == 1.0


def test_corrupt_artifact_is_refused(tmp_path):
    image = _sample_image(seed=4)
    path = tmp_path / "00000.pt"
    metrics.save_score_tensor(image, path, kind="render", view_name="00000")

    payload = torch.load(path, map_location="cpu", weights_only=True)
    payload["tensor"] = payload["tensor"] + 1e-3  # same shape/dtype, new samples
    torch.save(payload, path)

    with pytest.raises(metrics.ScoreArtifactError, match="hash mismatch"):
        metrics.load_score_tensor(path)


def test_missing_artifact_is_refused(tmp_path):
    with pytest.raises(metrics.ScoreArtifactError, match="missing score artifact"):
        metrics.load_score_tensor(tmp_path / "nope.pt")


def test_score_clamp_is_explicit_and_recorded():
    image = _sample_image(seed=5)
    image[0, 0, 0] = -0.5
    image[0, 0, 1] = 2.0

    clamped, record = metrics.apply_score_clamp(image)

    assert clamped.min().item() >= 0.0
    assert clamped.max().item() <= 1.0
    assert record["applied"] is True
    assert record["range"] == [0.0, 1.0]
    assert record["samples_changed"] == 2
    assert record["samples_total"] == image.numel()
    assert record["value_min"] == pytest.approx(-0.5)
    assert record["value_max"] == pytest.approx(2.0)


def test_score_clamp_records_zero_change_for_in_range_input():
    _, record = metrics.apply_score_clamp(_sample_image(seed=6))
    assert record["samples_changed"] == 0
    assert record["fraction_changed"] == 0.0
    assert record["samples_nonfinite"] == 0


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")])
def test_score_clamp_refuses_nonfinite_before_scoring(bad_value):
    image = _sample_image(seed=7)
    image[0, 0, 0] = bad_value

    with pytest.raises(ValueError, match=r"finite values in \[0, 1\]"):
        metrics.apply_score_clamp(image)


def test_read_score_pairs_round_trips_a_method_directory(tmp_path):
    method_dir = tmp_path / "test" / "ours_30000"
    renders_dir = method_dir / metrics.SCORE_ARTIFACT_DIRNAME / "renders"
    gts_dir = method_dir / metrics.SCORE_ARTIFACT_DIRNAME / "gt"

    originals = {}
    for idx in range(3):
        render = _sample_image(seed=10 + idx)
        gt = _sample_image(seed=20 + idx)
        stem = f"{idx:05d}{metrics.SCORE_ARTIFACT_SUFFIX}"
        metrics.save_score_tensor(render, renders_dir / stem, kind="render", view_name=stem)
        metrics.save_score_tensor(gt, gts_dir / stem, kind="gt", view_name=stem)
        originals[stem] = (render, gt)

    loaded_renders, loaded_gts, names, metas = metrics.read_score_pairs(method_dir)

    from eligsir.utils.image_utils import psnr

    assert names == sorted(originals)
    assert len(metas) == 3
    for i, name in enumerate(names):
        render, gt = originals[name]
        assert loaded_renders[i].shape == (1, 3, 16, 24)
        assert torch.equal(loaded_renders[i][0], render)
        assert torch.equal(loaded_gts[i][0], gt)
        # Acceptance criterion: direct tensor evaluation and post-hoc
        # evaluation from the persisted artifacts are identical, not merely
        # close.
        direct = psnr(_declared(render.unsqueeze(0)), _declared(gt.unsqueeze(0))).mean().item()
        post_hoc = psnr(_declared(loaded_renders[i]), _declared(loaded_gts[i])).mean().item()
        assert direct == post_hoc


def test_read_score_pairs_refuses_a_legacy_png_only_directory(tmp_path):
    """A pre-#640 run directory has only PNGs -- refuse, never fall back."""
    method_dir = tmp_path / "test" / "ours_30000"
    (method_dir / "renders").mkdir(parents=True)
    (method_dir / "gt").mkdir(parents=True)
    (method_dir / "renders" / "00000.png").write_bytes(b"not-a-real-png")
    (method_dir / "gt" / "00000.png").write_bytes(b"not-a-real-png")

    with pytest.raises(metrics.ScoreArtifactError, match="rendered before issue #640"):
        metrics.read_score_pairs(method_dir)


def test_metrics_module_does_not_import_image_readers():
    """Structural guard: the metric path must not be able to read images."""
    import ast
    import pathlib

    tree = ast.parse(pathlib.Path(metrics.__file__).read_text())
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)

    # The docstring names PIL/torchvision in prose; only real imports matter.
    roots = {name.split(".")[0] for name in imported}
    assert "PIL" not in roots
    assert "torchvision" not in roots


def test_resolution_mismatch_between_render_and_gt_is_refused(tmp_path):
    method_dir = tmp_path / "test" / "ours_30000"
    renders_dir = method_dir / metrics.SCORE_ARTIFACT_DIRNAME / "renders"
    gts_dir = method_dir / metrics.SCORE_ARTIFACT_DIRNAME / "gt"
    stem = f"00000{metrics.SCORE_ARTIFACT_SUFFIX}"

    metrics.save_score_tensor(
        torch.rand(3, 16, 24), renders_dir / stem, kind="render", view_name=stem
    )
    metrics.save_score_tensor(
        torch.rand(3, 8, 12), gts_dir / stem, kind="gt", view_name=stem
    )

    with pytest.raises(metrics.ScoreArtifactError, match="resolution mismatch"):
        metrics.read_score_pairs(method_dir)


def test_score_artifact_era_marker_in_tensor_metadata(tmp_path):
    """Era marker is present in per-tensor metadata for post-#640 scores."""
    image = _sample_image(seed=8)
    path = tmp_path / "00000.pt"

    metrics.save_score_tensor(image, path, kind="render", view_name="00000")
    loaded, meta = metrics.load_score_tensor(path)

    assert "era" in meta
    assert meta["era"] == metrics.SCORE_ARTIFACT_ERA
    assert meta["era"] == "issue_640_exact_float32_tensor"


def test_score_artifact_era_marker_in_results_provenance(tmp_path):
    """Era marker is present in results.json provenance for post-#640 scores."""
    method_dir = tmp_path / "test" / "ours_30000"
    renders_dir = method_dir / metrics.SCORE_ARTIFACT_DIRNAME / "renders"
    gts_dir = method_dir / metrics.SCORE_ARTIFACT_DIRNAME / "gt"

    # Create minimal score artifacts
    render = _sample_image(seed=9)
    gt = _sample_image(seed=10)
    stem = f"00000{metrics.SCORE_ARTIFACT_SUFFIX}"
    metrics.save_score_tensor(render, renders_dir / stem, kind="render", view_name=stem)
    metrics.save_score_tensor(gt, gts_dir / stem, kind="gt", view_name=stem)

    # Simulate what evaluate() does: read artifacts and build provenance
    renders, gts, names, metas = metrics.read_score_pairs(method_dir)

    # Verify metadata has era marker
    for meta in metas:
        assert "era" in meta["render"]
        assert meta["render"]["era"] == metrics.SCORE_ARTIFACT_ERA
        assert "era" in meta["gt"]
        assert meta["gt"]["era"] == metrics.SCORE_ARTIFACT_ERA


def test_posthoc_results_persist_the_canonical_score_provenance(tmp_path, monkeypatch):
    """The exact-artifact CLI emits the same score contract as inline reports."""
    method_dir = tmp_path / "test" / "ours_30000"
    renders_dir = method_dir / metrics.SCORE_ARTIFACT_DIRNAME / "renders"
    gts_dir = method_dir / metrics.SCORE_ARTIFACT_DIRNAME / "gt"
    stem = f"00000{metrics.SCORE_ARTIFACT_SUFFIX}"
    metrics.save_score_tensor(
        _sample_image(seed=30), renders_dir / stem, kind="render", view_name=stem
    )
    metrics.save_score_tensor(
        _sample_image(seed=31), gts_dir / stem, kind="gt", view_name=stem
    )
    lpips_module = types.ModuleType("lpipsPyTorch")
    lpips_module.lpips = lambda *_args, **_kwargs: torch.tensor(0.25)
    monkeypatch.setitem(sys.modules, "lpipsPyTorch", lpips_module)

    assert metrics.evaluate([str(tmp_path)], device="cpu") == []

    results = json.loads((tmp_path / "results.json").read_text())
    contract = results["ours_30000"]["provenance"]["score_contract"]
    assert contract["domain"] == "float32_0_1"
    assert contract["resolution"] == [16, 24]
    assert contract["render"]["steps"][0]["type"] == "declared_score_transform"
    assert contract["gt"]["steps"] == []


def test_tiny_inline_and_posthoc_scores_use_the_same_finished_pair(tmp_path, monkeypatch):
    """All three reported metrics see the same pair after the shared boundary."""
    from eligsir.utils.loss_utils import ssim
    from eligsir.utils.score_contract import apply_declared_score_clamp_pair, score_canonical_pair

    method_dir = tmp_path / "test" / "ours_30000"
    renders_dir = method_dir / metrics.SCORE_ARTIFACT_DIRNAME / "renders"
    gts_dir = method_dir / metrics.SCORE_ARTIFACT_DIRNAME / "gt"
    stem = f"00000{metrics.SCORE_ARTIFACT_SUFFIX}"
    render = _sample_image(seed=41)
    render[0, 0, 0] = 1.1
    gt = _sample_image(seed=42)
    metrics.save_score_tensor(render, renders_dir / stem, kind="render", view_name=stem)
    metrics.save_score_tensor(gt, gts_dir / stem, kind="gt", view_name=stem)

    inline_gt = _declared(gt.unsqueeze(0))
    inline_render, inline_gt, inline_audit = apply_declared_score_clamp_pair(
        render.unsqueeze(0), inline_gt, context="test_inline_pair"
    )
    fake_lpips = lambda a, b, **_kwargs: (a - b).abs().mean()
    inline_scores = score_canonical_pair(
        inline_render,
        inline_gt,
        ssim_fn=ssim,
        lpips_fn=fake_lpips,
        context="test_inline_pair",
    )
    inline = {
        "PSNR": inline_scores.psnr.mean().item(),
        "SSIM": inline_scores.ssim.item(),
        "LPIPS": inline_scores.lpips.item(),
    }

    lpips_module = types.ModuleType("lpipsPyTorch")
    lpips_module.lpips = fake_lpips
    monkeypatch.setitem(sys.modules, "lpipsPyTorch", lpips_module)
    assert metrics.evaluate([str(tmp_path)], device="cpu") == []

    posthoc = json.loads((tmp_path / "results.json").read_text())["ours_30000"]
    assert posthoc["PSNR"] == inline["PSNR"]
    assert posthoc["SSIM"] == inline["SSIM"]
    assert posthoc["LPIPS"] == inline["LPIPS"]
    assert posthoc["provenance"]["score_contract"] == json.loads(
        json.dumps(inline_audit["score_provenance"])
    )
