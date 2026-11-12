"""Regression tests for the final real-renderer parity gate (#645 / #638)."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import types

import numpy as np
import pytest

torch = pytest.importorskip("torch")


def _canonical(tensor: torch.Tensor, context: str) -> torch.Tensor:
    from eligsir.utils.score_contract import declare_canonical_score_tensor

    return declare_canonical_score_tensor(tensor, context=context)[0]


def _fake_lpips(render, gt, **_kwargs):
    return (render - gt).abs().mean()


def _write_complete_score_set(metrics, root, *, role, render=None, gt=None):
    render = torch.zeros(3, 16, 16) if render is None else render
    gt = torch.ones(3, 16, 16) if gt is None else gt
    metrics.save_score_pair(
        render, gt, root, artifact_name="00000.pt", view_name="view-a"
    )
    entries, pairs = metrics._score_pair_ledger(root)
    provenance, score_metrics, _ = metrics.score_exact_pairs(
        entries, pairs, device="cpu", lpips_fn=_fake_lpips
    )
    return metrics.write_score_set_manifest(
        root,
        entries=entries,
        split="test",
        iteration=1,
        role=role,
        score_provenance=provenance,
        metrics=score_metrics,
        tolerances=metrics.SCORE_PARITY_TOLERANCES,
    )


def _rewrite_manifest(metrics, root, mutate):
    path = root / "manifest.json"
    manifest = json.loads(path.read_text())
    mutate(manifest)
    manifest.pop("manifest_payload_sha256")
    manifest["manifest_payload_sha256"] = metrics._canonical_json_sha256(manifest)
    path.write_text(json.dumps(manifest))


def _complete_score_roots(metrics, tmp_path):
    roots = [tmp_path / "left", tmp_path / "right"]
    for root, role in zip(roots, sorted(metrics.SCORE_ARTIFACT_ROLES)):
        _write_complete_score_set(metrics, root, role=role)
    return roots


def _emit_offline_cfg(tmp_path, monkeypatch):
    from argparse import ArgumentParser

    import train as train_module
    from arguments import ModelParams, OptimizationParams, PipelineParams

    local_path = tmp_path / "local"
    persist_path = tmp_path / "persist"
    parser = ArgumentParser()
    model = ModelParams(parser)
    PipelineParams(parser)
    OptimizationParams(parser)
    run_args = parser.parse_args(
        [
            "-s",
            str(tmp_path),
            "-m",
            str(local_path),
            "--no-gsplat_sparse_grad",
            "--no-sparse_mode_detach_sh_dir",
            "--compile_sh",
            "--near_plane",
            "0.125",
        ]
    )
    run_args.persist_model_path = str(persist_path)
    dataset = model.extract(run_args)
    monkeypatch.setattr(train_module, "TENSORBOARD_FOUND", False)
    monkeypatch.setattr(sys, "argv", ["train.py"])
    train_module.prepare_output_and_logger(dataset, run_args=run_args)
    return persist_path


def test_posthoc_render_resolves_saved_renderer_optimization_settings(
    tmp_path, monkeypatch
):
    from argparse import ArgumentParser

    import render as render_module
    from arguments import (
        ModelParams,
        PipelineParams,
        add_renderer_optimization_args,
        get_combined_args,
    )

    model_path = _emit_offline_cfg(tmp_path, monkeypatch)
    parser = ArgumentParser()
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser, fill_none=True)
    add_renderer_optimization_args(parser)
    monkeypatch.setattr(sys, "argv", ["render.py", "-m", str(model_path)])
    args = get_combined_args(parser)

    _, resolved_pipeline = render_module.resolve_render_configuration(
        model, pipeline, args
    )

    assert resolved_pipeline.gsplat_sparse_grad is False
    assert resolved_pipeline.sparse_mode_detach_sh_dir is False
    assert resolved_pipeline.compile_sh is True
    assert resolved_pipeline.near_plane == 0.125


def test_posthoc_renderer_cli_trio_overrides_saved_configuration(
    tmp_path, monkeypatch
):
    from argparse import ArgumentParser

    import render as render_module
    from arguments import (
        ModelParams,
        PipelineParams,
        add_renderer_optimization_args,
        get_combined_args,
    )

    model_path = _emit_offline_cfg(tmp_path, monkeypatch)
    parser = ArgumentParser()
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser, fill_none=True)
    add_renderer_optimization_args(parser)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "render.py",
            "-m",
            str(model_path),
            "--gsplat_sparse_grad",
            "--sparse_mode_detach_sh_dir",
            "--no-compile_sh",
        ],
    )
    args = get_combined_args(parser)

    _, resolved_pipeline = render_module.resolve_render_configuration(
        model, pipeline, args
    )

    assert resolved_pipeline.gsplat_sparse_grad is True
    assert resolved_pipeline.sparse_mode_detach_sh_dir is True
    assert resolved_pipeline.compile_sh is False


def test_offline_writer_persists_full_renderer_configuration(tmp_path, monkeypatch):
    from argparse import Namespace

    persist_path = _emit_offline_cfg(tmp_path, monkeypatch)
    saved = eval(
        (persist_path / "cfg_args").read_text(), {"Namespace": Namespace}
    )
    assert saved.model_path == str(persist_path)
    assert saved.gsplat_sparse_grad is False
    assert saved.sparse_mode_detach_sh_dir is False
    assert saved.compile_sh is True
    assert saved.near_plane == 0.125


def test_legacy_renderer_settings_use_historical_offline_defaults():
    from argparse import Namespace

    from arguments import apply_renderer_optimization_params

    pipeline = apply_renderer_optimization_params(Namespace(), Namespace())

    assert pipeline.gsplat_sparse_grad is True
    assert pipeline.sparse_mode_detach_sh_dir is True
    assert pipeline.compile_sh is False


def test_posthoc_render_warns_when_legacy_cfg_omits_renderer_settings(tmp_path):
    from argparse import ArgumentParser, Namespace

    import render as render_module
    from arguments import ModelParams, PipelineParams

    parser = ArgumentParser()
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser, fill_none=True)
    legacy_args = Namespace(
        source_path=str(tmp_path), model_path=str(tmp_path / "model")
    )

    with pytest.warns(RuntimeWarning, match="historical offline defaults"):
        _, resolved = render_module.resolve_render_configuration(
            model, pipeline, legacy_args
        )

    assert resolved.gsplat_sparse_grad is True
    assert resolved.sparse_mode_detach_sh_dir is True
    assert resolved.compile_sh is False
    assert resolved.sh_impl == "gsplat"


def test_streaming_renderer_configuration_uses_shared_trio_resolver():
    from argparse import Namespace

    import train_streaming

    pipeline = Namespace()
    optimization = Namespace(
        gsplat_sparse_grad=False,
        sparse_mode_detach_sh_dir=False,
        compile_sh=True,
    )

    resolved = train_streaming.resolve_streaming_renderer_configuration(
        pipeline, optimization
    )

    assert resolved is pipeline
    assert pipeline.gsplat_sparse_grad is False
    assert pipeline.sparse_mode_detach_sh_dir is False
    assert pipeline.compile_sh is True


class _SyntheticGaussianState:
    def __init__(self):
        self.active_sh_degree = 0
        self.params = {
            "means": torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]),
            "sh0": torch.zeros((2, 1, 3)),
            "shN": torch.empty((2, 0, 3)),
            "opacities": torch.zeros(2),
            "scales": torch.zeros((2, 3)),
            "quats": torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(2, 1),
        }

    def construct_list_of_attributes(self):
        return [
            "x", "y", "z", "nx", "ny", "nz",
            "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
            "scale_0", "scale_1", "scale_2",
            "rot_0", "rot_1", "rot_2", "rot_3",
        ]

    def capture(self):
        return {"params": self.params}


def test_production_mutation_capture_seam_captures_after_mutation(tmp_path):
    from plyfile import PlyData
    from eligsir.io.snapshots import (
        mutate_and_capture_iteration_outputs,
        write_ply_from_snapshot,
    )

    gaussians = _SyntheticGaussianState()

    def mutate():
        gaussians.params["means"].add_(torch.tensor([2.0, 3.0, 4.0]))

    captured = mutate_and_capture_iteration_outputs(
        gaussians,
        mutate=mutate,
        capture_ply=True,
        capture_checkpoint_state=True,
    )
    path = tmp_path / "after-mutation.ply"
    write_ply_from_snapshot(captured["ply"], str(path))
    vertices = PlyData.read(path)["vertex"]
    saved_means = torch.from_numpy(
        np.stack([vertices["x"], vertices["y"], vertices["z"]], axis=1).copy()
    )

    expected = torch.tensor([[2.0, 4.0, 6.0], [5.0, 7.0, 9.0]])
    assert torch.equal(saved_means, expected)
    assert torch.equal(captured["checkpoint"]["params"]["means"], expected)


def test_offline_completion_rejects_save_failure_before_training_complete():
    from eligsir.runtime.finalize import begin_offline_completion
    from eligsir.runtime.save_worker import AsyncSaveWorker

    class RunState:
        completed = False

        def mark_training_complete(self, *, final_iteration):
            self.completed = True

    run_state = RunState()
    worker = AsyncSaveWorker(join_timeout_s=1)
    worker.enqueue(lambda: (_ for _ in ()).throw(OSError("checkpoint failed")))

    with pytest.raises(RuntimeError, match="async save task failed"):
        begin_offline_completion(worker, run_state, final_iteration=9)

    assert run_state.completed is False


def test_offline_completion_drains_without_restoring_emergency_handler(monkeypatch):
    from eligsir.runtime import finalize as finalize_mod
    from eligsir.runtime.finalize import begin_offline_completion

    events = []

    class Worker:
        def shutdown(self):
            events.append(("shutdown",))

    class RunState:
        def mark_training_complete(self, *, final_iteration):
            events.append(("training_complete", final_iteration))

    monkeypatch.setattr(
        finalize_mod._signal,
        "signal",
        lambda signum, handler: events.append(("signal", signum, handler)),
    )

    begin_offline_completion(
        Worker(),
        RunState(),
        final_iteration=9,
        orig_sigterm=object(),
    )

    assert events == [("shutdown",), ("training_complete", 9)]


@pytest.mark.parametrize(
    ("training_mode", "backend_attr"),
    [("fused", "_FUSED_SSIM_FN"), ("compiled", "_COMPILED_SSIM")],
)
def test_scientific_scores_ignore_mutable_training_ssim_backend(
    tmp_path, monkeypatch, training_mode, backend_attr
):
    import metrics
    import eligsir.utils.comparison_report as comparison_report
    import eligsir.utils.loss_utils as loss_utils
    from eligsir.utils.score_contract import apply_declared_score_clamp_pair

    render = torch.stack([
        torch.full((16, 16), 0.9, dtype=torch.float64),
        torch.full((16, 16), 0.4, dtype=torch.float64),
        torch.full((16, 16), 0.11, dtype=torch.float64),
    ])
    gt = _canonical(torch.stack([
        torch.full((16, 16), 0.1, dtype=torch.float64),
        torch.full((16, 16), 0.35, dtype=torch.float64),
        torch.full((16, 16), 0.1, dtype=torch.float64),
    ]), "issue645-inline-gt")

    class Camera:
        image_name = "view-a"
        uid = 7
        original_image = gt

    class Gaussian:
        get_xyz = torch.zeros((0, 3))

    fake_lpips = lambda a, b, **_kwargs: (a - b).abs().mean()
    lpips_module = types.ModuleType("lpipsPyTorch")
    lpips_module.lpips = fake_lpips
    monkeypatch.setitem(sys.modules, "lpipsPyTorch", lpips_module)
    monkeypatch.setenv("SPLAT_REPORT_OVERVIEW", "0")
    monkeypatch.setenv("SPLAT_REPORT_VIDEOS", "0")
    monkeypatch.setattr(comparison_report, "_gt_image", lambda cam: cam.original_image)

    pair_render, pair_gt, _ = apply_declared_score_clamp_pair(
        render, gt, context="issue645-expected"
    )
    monkeypatch.setattr(loss_utils, "_SSIM_MODE", "ref")
    expected_ssim = float(
        loss_utils.ssim(pair_render.float(), pair_gt.float()).item()
    )
    monkeypatch.setattr(loss_utils, "_SSIM_MODE", training_mode)
    monkeypatch.setattr(
        loss_utils,
        backend_attr,
        lambda image, *_args, **_kwargs: image.new_tensor(-0.25),
    )
    assert loss_utils.ssim(pair_render.float(), pair_gt.float()).item() == -0.25

    report = comparison_report.write_post_training_report(
        model_path=str(tmp_path),
        iteration=9,
        gaussians=Gaussian(),
        train_cams=[],
        test_cams=[Camera()],
        render_fn=lambda *_args: {"render": render},
        pipe=object(),
        background=None,
    )

    assert report["mean_test_ssim"] == pytest.approx(expected_ssim)
    assert len(report["test_per_view"]) == 1
    per_view = report["test_per_view"][0]
    assert per_view["name"] == "view-a"
    assert per_view["source_index"] == 7
    assert per_view["psnr"] == pytest.approx(report["mean_test_psnr"])
    assert per_view["ssim"] == pytest.approx(expected_ssim)
    assert per_view["lpips"] == pytest.approx(report["mean_test_lpips"])

    artifact_record = report["score_artifacts"]["test"]
    root = Path(artifact_record["path"])
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["format_version"] == 2
    assert manifest["n_views"] == 1
    assert manifest["views"][0]["view_name"] == "view-a"
    assert len(manifest["views"][0]["render_sha256"]) == 64
    assert len(manifest["views"][0]["gt_sha256"]) == 64
    assert len(manifest["artifact_set_sha256"]) == 64
    assert len(manifest["manifest_payload_sha256"]) == 64
    assert manifest["score_provenance"] == json.loads(
        json.dumps(report["score_provenance"]["test"])
    )
    assert manifest["metrics"]["per_view"][0]["ssim"] == pytest.approx(expected_ssim)
    assert manifest["metrics"]["aggregate"]["ssim"] == pytest.approx(expected_ssim)
    assert manifest["tolerances"] == {"psnr": 1e-6, "ssim": 1e-7, "lpips": 1e-7}
    _, persisted_provenance, persisted_metrics, _ = metrics.score_persisted_pairs(
        root, device="cpu", lpips_fn=fake_lpips
    )
    assert manifest["score_provenance"] == persisted_provenance
    assert manifest["metrics"] == persisted_metrics
    assert report["mean_test_ssim"] == persisted_metrics["aggregate"]["ssim"]
    saved_render, _ = metrics.load_score_tensor(root / "renders" / "00000.pt")
    saved_gt, _ = metrics.load_score_tensor(root / "gt" / "00000.pt")
    assert torch.equal(saved_render, render.float())
    assert torch.equal(saved_gt, gt.float())

    posthoc_model = tmp_path / "posthoc"
    posthoc_root = posthoc_model / "test" / "ours_9" / metrics.SCORE_ARTIFACT_DIRNAME
    metrics.save_score_pair(
        render,
        gt,
        posthoc_root,
        artifact_name="00000.pt",
        view_name="view-a",
    )
    assert metrics.evaluate([str(posthoc_model)], device="cpu") == []
    assert metrics.compare_score_artifact_sets(
        root, posthoc_root, device="cpu", lpips_fn=fake_lpips
    )["match"] is True


def test_multi_view_comparator_aligns_real_inline_and_render_set_by_view_name(
    tmp_path, monkeypatch
):
    import metrics
    import render as render_module
    import eligsir.utils.comparison_report as comparison_report

    class Camera:
        def __init__(self, name, gt):
            self.image_name = name
            self.uid = 1 if name == "a" else 2
            self.original_image = _canonical(gt, f"issue645-{name}-gt")

    class Gaussian:
        get_xyz = torch.zeros((0, 3))

    gts = {
        "a": torch.zeros(3, 16, 16),
        "b": torch.full((3, 16, 16), 0.2),
    }
    renders = {
        "a": torch.stack([
            torch.full((16, 16), 0.7),
            torch.full((16, 16), 0.1),
            torch.full((16, 16), 0.05),
        ]),
        "b": torch.stack([
            torch.full((16, 16), 0.25),
            torch.full((16, 16), 0.8),
            torch.full((16, 16), 0.3),
        ]),
    }
    cameras = {name: Camera(name, gt) for name, gt in gts.items()}
    fake_lpips = lambda a, b, **_kwargs: (a - b).abs().mean()
    lpips_module = types.ModuleType("lpipsPyTorch")
    lpips_module.lpips = fake_lpips
    monkeypatch.setitem(sys.modules, "lpipsPyTorch", lpips_module)
    monkeypatch.setenv("SPLAT_REPORT_OVERVIEW", "0")
    monkeypatch.setenv("SPLAT_REPORT_VIDEOS", "0")
    monkeypatch.setattr(comparison_report, "_gt_image", lambda cam: cam.original_image)

    inline = comparison_report.write_post_training_report(
        model_path=str(tmp_path / "inline"),
        iteration=11,
        gaussians=Gaussian(),
        train_cams=[],
        test_cams=[cameras["b"], cameras["a"]],
        render_fn=lambda cam, *_args: {"render": renders[cam.image_name]},
        pipe=object(),
        background=None,
    )

    posthoc_model = tmp_path / "posthoc"
    monkeypatch.setattr(
        render_module,
        "render",
        lambda cam, *_args: {"render": renders[cam.image_name]},
    )
    render_module.render_set(
        str(posthoc_model),
        "test",
        11,
        [cameras["a"], cameras["b"]],
        Gaussian(),
        object(),
        None,
    )
    assert metrics.evaluate([str(posthoc_model)], device="cpu") == []

    inline_root = Path(inline["score_artifacts"]["test"]["path"])
    posthoc_root = posthoc_model / "test" / "ours_11" / metrics.SCORE_ARTIFACT_DIRNAME
    result = metrics.compare_score_artifact_sets(
        inline_root,
        posthoc_root,
        device="cpu",
        lpips_fn=lambda a, b: (a - b).abs().mean(),
    )
    assert result["match"] is True
    assert result["n_views"] == 2
    assert (
        result["left_artifact_set_sha256"]
        == result["right_artifact_set_sha256"]
    )

    inline_manifest = json.loads((inline_root / "manifest.json").read_text())
    posthoc_manifest = json.loads((posthoc_root / "manifest.json").read_text())
    assert (
        inline_manifest["artifact_set_sha256"]
        == posthoc_manifest["artifact_set_sha256"]
    )


def test_repeated_inline_report_scores_only_views_written_by_current_invocation(
    tmp_path, monkeypatch
):
    import metrics
    import eligsir.utils.comparison_report as comparison_report

    class Camera:
        def __init__(self, name, value, uid):
            self.image_name = name
            self.uid = uid
            self.original_image = _canonical(
                torch.full((3, 16, 16), value), f"issue645-stale-{name}-gt"
            )

    class Gaussian:
        get_xyz = torch.zeros((0, 3))

    cameras = {
        "a": Camera("a", 0.1, 1),
        "b": Camera("b", 0.2, 2),
        "c": Camera("c", 0.3, 3),
    }
    renders = {
        name: torch.full((3, 16, 16), value)
        for name, value in (("a", 0.4), ("b", 0.5), ("c", 0.6))
    }
    lpips_module = types.ModuleType("lpipsPyTorch")
    lpips_module.lpips = _fake_lpips
    monkeypatch.setitem(sys.modules, "lpipsPyTorch", lpips_module)
    monkeypatch.setenv("SPLAT_REPORT_OVERVIEW", "0")
    monkeypatch.setenv("SPLAT_REPORT_VIDEOS", "0")
    monkeypatch.setattr(comparison_report, "_gt_image", lambda cam: cam.original_image)

    def write(cams):
        return comparison_report.write_post_training_report(
            model_path=str(tmp_path),
            iteration=13,
            gaussians=Gaussian(),
            train_cams=[],
            test_cams=cams,
            render_fn=lambda cam, *_args: {"render": renders[cam.image_name]},
            pipe=object(),
            background=None,
        )

    first = write([cameras["a"], cameras["b"]])
    assert first["n_test_metrics"] == 2

    second = write([cameras["c"]])
    root = Path(second["score_artifacts"]["test"]["path"])
    manifest = json.loads((root / "manifest.json").read_text())

    assert not (root / "renders" / "00001.pt").exists()
    assert second["n_test_metrics"] == 1
    assert [row["name"] for row in second["test_per_view"]] == ["c"]
    assert manifest["n_views"] == 1
    assert [row["view_name"] for row in manifest["views"]] == ["c"]

    posthoc_model = tmp_path / "posthoc-stale-tail"
    posthoc_root = posthoc_model / "test" / "ours_13" / metrics.SCORE_ARTIFACT_DIRNAME
    metrics.save_score_pair(
        renders["c"],
        cameras["c"].original_image,
        posthoc_root,
        artifact_name="00000.pt",
        view_name="c",
    )
    assert metrics.evaluate([str(posthoc_model)], device="cpu") == []
    assert metrics.compare_score_artifact_sets(
        root, posthoc_root, device="cpu", lpips_fn=_fake_lpips
    )["match"] is True


def test_comparator_recomputes_provenance_from_exact_tensors(tmp_path):
    import metrics

    roots = _complete_score_roots(metrics, tmp_path)
    _rewrite_manifest(
        metrics,
        roots[1],
        lambda manifest: manifest.__setitem__(
            "score_provenance", {"contract_version": 999, "fabricated": True}
        ),
    )

    with pytest.raises(metrics.ScoreArtifactError, match="provenance claim mismatch"):
        metrics.compare_score_artifact_sets(
            *roots,
            device="cpu",
            lpips_fn=_fake_lpips,
        )


def test_comparator_recomputes_per_view_metric_from_exact_tensors(tmp_path):
    import metrics

    roots = _complete_score_roots(metrics, tmp_path)
    _rewrite_manifest(
        metrics,
        roots[1],
        lambda manifest: manifest["metrics"]["per_view"][0].__setitem__(
            "psnr", manifest["metrics"]["per_view"][0]["psnr"] + 1.0
        ),
    )

    with pytest.raises(metrics.ScoreArtifactError, match="per-view psnr claim mismatch"):
        metrics.compare_score_artifact_sets(
            *roots,
            device="cpu",
            lpips_fn=_fake_lpips,
        )


def test_comparator_recomputes_aggregate_metric_from_exact_tensors(tmp_path):
    import metrics

    roots = _complete_score_roots(metrics, tmp_path)
    _rewrite_manifest(
        metrics,
        roots[1],
        lambda manifest: manifest["metrics"]["aggregate"].__setitem__(
            "ssim", manifest["metrics"]["aggregate"]["ssim"] + 0.1
        ),
    )

    with pytest.raises(metrics.ScoreArtifactError, match="aggregate ssim claim mismatch"):
        metrics.compare_score_artifact_sets(
            *roots,
            device="cpu",
            lpips_fn=_fake_lpips,
        )


def test_inline_report_requires_lpips_for_exact_evidence(tmp_path, monkeypatch):
    import builtins
    import eligsir.utils.comparison_report as comparison_report

    class Camera:
        image_name = "view-a"
        uid = 1
        original_image = _canonical(
            torch.zeros(3, 16, 16), "issue645-lpips-required-gt"
        )

    class Gaussian:
        get_xyz = torch.zeros((0, 3))

    real_import = builtins.__import__

    def import_without_lpips(name, *args, **kwargs):
        if name == "lpipsPyTorch":
            raise ImportError("lpips unavailable")
        return real_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "lpipsPyTorch", raising=False)
    monkeypatch.setattr(builtins, "__import__", import_without_lpips)
    monkeypatch.setenv("SPLAT_REPORT_OVERVIEW", "0")
    monkeypatch.setenv("SPLAT_REPORT_VIDEOS", "0")
    monkeypatch.setattr(comparison_report, "_gt_image", lambda cam: cam.original_image)

    with pytest.raises(RuntimeError, match="LPIPS is required"):
        comparison_report.write_post_training_report(
            model_path=str(tmp_path),
            iteration=1,
            gaussians=Gaussian(),
            train_cams=[],
            test_cams=[Camera()],
            render_fn=lambda *_args: {"render": torch.zeros(3, 16, 16)},
            pipe=object(),
            background=None,
        )


def test_inline_score_artifact_persistence_failure_propagates(tmp_path, monkeypatch):
    import metrics
    import eligsir.utils.comparison_report as comparison_report

    render = torch.zeros(3, 16, 16)
    gt = _canonical(torch.ones(3, 16, 16), "issue645-persistence-failure-gt")

    class Camera:
        image_name = "view-a"
        uid = 1
        original_image = gt

    class Gaussian:
        get_xyz = torch.zeros((0, 3))

    monkeypatch.setenv("SPLAT_REPORT_OVERVIEW", "0")
    monkeypatch.setenv("SPLAT_REPORT_VIDEOS", "0")
    monkeypatch.setattr(comparison_report, "_gt_image", lambda cam: cam.original_image)
    monkeypatch.setattr(
        metrics,
        "save_score_pair",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("disk full")),
    )

    with pytest.raises(OSError, match="disk full"):
        comparison_report.write_post_training_report(
            model_path=str(tmp_path),
            iteration=1,
            gaussians=Gaussian(),
            train_cams=[],
            test_cams=[Camera()],
            render_fn=lambda *_args: {"render": render},
            pipe=object(),
            background=None,
        )


def test_exact_score_set_comparator_fails_closed_on_one_changed_sample(tmp_path):
    import metrics

    left = tmp_path / "left"
    right = tmp_path / "right"
    render = torch.linspace(0.0, 1.0, 3 * 4 * 4).reshape(3, 4, 4)
    gt = torch.flip(render, dims=(-1,))
    roles = sorted(metrics.SCORE_ARTIFACT_ROLES)
    _write_complete_score_set(metrics, left, role=roles[0], render=render, gt=gt)
    _write_complete_score_set(metrics, right, role=roles[1], render=render, gt=gt)

    matched = metrics.compare_score_artifact_sets(
        left, right, device="cpu", lpips_fn=_fake_lpips
    )
    assert matched["match"] is True
    assert matched["n_views"] == 1
    assert matched["left_artifact_set_sha256"] == matched["right_artifact_set_sha256"]

    metrics.save_score_pair(
        render.clone().index_put_((torch.tensor([0]), torch.tensor([0]), torch.tensor([0])), torch.tensor([0.5])),
        gt,
        right,
        artifact_name="00000.pt",
        view_name="view-a",
    )
    with pytest.raises(metrics.ScoreArtifactError, match="render tensor mismatch.*view-a"):
        metrics.compare_score_artifact_sets(
            left, right, device="cpu", lpips_fn=_fake_lpips
        )


@pytest.mark.parametrize(
    ("field", "bad_value", "message"),
    [
        ("format_version", -1, "invalid format_version"),
        ("era", "legacy", "invalid era"),
        ("score_domain", "png", "invalid score_domain"),
        ("split", "train", "invalid split"),
        ("iteration", -1, "invalid iteration"),
        ("role", "unknown", "invalid role"),
        ("holdout_hash", "wrong", "holdout hash mismatch"),
        ("score_provenance", None, "missing provenance"),
        ("metrics", None, "missing metrics"),
        ("tolerances", None, "missing tolerances"),
    ],
)
def test_exact_score_set_comparator_rejects_incomplete_evidence(
    tmp_path, field, bad_value, message
):
    import metrics

    roots = [tmp_path / "left", tmp_path / "right"]
    for root, role in zip(roots, sorted(metrics.SCORE_ARTIFACT_ROLES)):
        _write_complete_score_set(metrics, root, role=role)
    manifest_path = roots[1] / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest[field] = bad_value
    manifest.pop("manifest_payload_sha256")
    manifest["manifest_payload_sha256"] = metrics._canonical_json_sha256(manifest)
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(metrics.ScoreArtifactError, match=message):
        metrics.compare_score_artifact_sets(
            *roots, device="cpu", lpips_fn=_fake_lpips
        )
