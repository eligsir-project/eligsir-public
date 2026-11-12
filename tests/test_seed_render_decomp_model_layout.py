from __future__ import annotations

import csv
import copy
import gc
import json
import subprocess
import sys
import weakref
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

import scripts.seed_render_decomp as seed_render_decomp
from eligsir.utils.gaussian_lineage import SeedCohortUnavailable
from eligsir.utils.score_contract import declare_canonical_score_tensor


RENDER_KEYS = ("means", "scales", "quats", "opacities", "sh0", "shN")


def test_script_help_runs_through_the_documented_direct_entrypoint():
    completed = subprocess.run(
        [sys.executable, "scripts/seed_render_decomp.py", "--help"],
        cwd=Path(__file__).resolve().parents[1],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--model-path" in completed.stdout


def _checkpoint_state() -> dict:
    n = 5
    return {
        "layout": "gsplat",
        "active_sh_degree": 2,
        "seed_uid_range": (10, 13),
        "params": {
            "means": torch.arange(n * 3, dtype=torch.float32).reshape(n, 3),
            "scales": torch.zeros(n, 3),
            "quats": torch.tensor([[1.0, 0.0, 0.0, 0.0]] * n),
            "opacities": torch.linspace(-2.0, 2.0, n),
            "sh0": torch.zeros(n, 1, 3),
            "shN": torch.zeros(n, 8, 3),
        },
        "streaming_state": {
            # Row 1 deliberately reuses a seed-range UID after relocation,
            # but its donor root is non-seed and its birth method is RESEED.
            "gaussian_uid": torch.tensor([10, 11, 12, 20, 21]),
            "parent_uid": torch.tensor([-1, 20, -1, 10, -1]),
            "root_uid": torch.tensor([10, 20, 12, 10, 21]),
            "generation": torch.tensor([0, 1, 0, 1, 0], dtype=torch.int32),
            "birth_method": torch.tensor([0, 3, 0, 2, 1], dtype=torch.int8),
            "last_reseed_iter": torch.tensor([-1, 500, -1, -1, -1], dtype=torch.int32),
            "birth_iter": torch.tensor([0, 500, 0, 250, 300], dtype=torch.int32),
        },
    }


def _write_checkpoint(tmp_path, iteration=250, *, payload_arity=3, state=None):
    state = _checkpoint_state() if state is None else state
    path = tmp_path / f"chkpnt{iteration}.pth"
    payload = (state, iteration, {"n_frames_ingested": 5})
    torch.save(payload[:payload_arity], path)
    return path


@pytest.mark.parametrize("payload_arity", [2, 3])
def test_load_checkpoint_accepts_exact_two_and_three_tuples(tmp_path, payload_arity):
    path = _write_checkpoint(tmp_path, payload_arity=payload_arity)

    loaded = seed_render_decomp.load_checkpoint_state(path)

    assert loaded.saved_iteration == 250
    assert loaded.tuple_arity == payload_arity
    assert loaded.streaming_meta == (
        {} if payload_arity == 2 else {"n_frames_ingested": 5}
    )
    assert loaded.lineage_view.get_xyz.shape == (5, 3)
    assert loaded.checkpoint_sha256


def test_load_checkpoint_rejects_non_tuple_and_wrong_tuple_length(tmp_path):
    path = tmp_path / "chkpnt250.pth"
    torch.save({"not": "a tuple"}, path)
    with pytest.raises(ValueError, match="2- or 3-tuple"):
        seed_render_decomp.load_checkpoint_state(path)

    torch.save((_checkpoint_state(),), path)
    with pytest.raises(ValueError, match="2- or 3-tuple"):
        seed_render_decomp.load_checkpoint_state(path)


def test_load_checkpoint_rejects_filename_iteration_mismatch(tmp_path):
    path = tmp_path / "chkpnt251.pth"
    torch.save((_checkpoint_state(), 250, {}), path)

    with pytest.raises(ValueError, match="filename iteration 251.*saved iteration 250"):
        seed_render_decomp.load_checkpoint_state(path)


@pytest.mark.parametrize("saved_iteration", [250.0, "250", np.float64(250.0), True])
def test_load_checkpoint_rejects_non_integral_saved_iteration(tmp_path, saved_iteration):
    path = tmp_path / "chkpnt250.pth"
    torch.save((_checkpoint_state(), saved_iteration, {}), path)

    with pytest.raises(ValueError, match="saved iteration must be an integer"):
        seed_render_decomp.load_checkpoint_state(path)


@pytest.mark.parametrize("layout", [None, "legacy"])
def test_load_checkpoint_rejects_missing_or_non_gsplat_layout(tmp_path, layout):
    state = _checkpoint_state()
    if layout is None:
        state.pop("layout")
    else:
        state["layout"] = layout
    path = _write_checkpoint(tmp_path, state=state)

    with pytest.raises(ValueError, match="layout.*gsplat"):
        seed_render_decomp.load_checkpoint_state(path)


@pytest.mark.parametrize("missing_key", RENDER_KEYS)
def test_load_checkpoint_rejects_each_missing_render_parameter(tmp_path, missing_key):
    state = _checkpoint_state()
    del state["params"][missing_key]
    path = _write_checkpoint(tmp_path, state=state)

    with pytest.raises(ValueError, match=missing_key):
        seed_render_decomp.load_checkpoint_state(path)


def test_load_checkpoint_rejects_parameter_row_count_mismatch(tmp_path):
    state = _checkpoint_state()
    state["params"]["shN"] = torch.zeros(4, 8, 3)
    path = _write_checkpoint(tmp_path, state=state)

    with pytest.raises(ValueError, match="shN.*rows"):
        seed_render_decomp.load_checkpoint_state(path)


def test_exact_partitions_exclude_relocated_seed_slot_and_cover_population(tmp_path):
    loaded = seed_render_decomp.load_checkpoint_state(_write_checkpoint(tmp_path))

    masks = seed_render_decomp.exact_partition_masks(loaded.lineage_view)

    assert torch.equal(
        masks["seed_root"], torch.tensor([True, False, True, False, False])
    )
    assert torch.equal(
        masks["seed_lineage"], torch.tensor([True, False, True, True, False])
    )
    assert torch.equal(
        masks["nonseed"], torch.tensor([False, True, False, False, True])
    )
    assert not bool((masks["seed_root"] & ~masks["seed_lineage"]).any())
    assert not bool((masks["seed_lineage"] & masks["nonseed"]).any())
    assert bool((masks["seed_lineage"] | masks["nonseed"]).all())


def test_exact_partitions_fail_closed_without_seed_uid_range(tmp_path):
    state = _checkpoint_state()
    state.pop("seed_uid_range")
    loaded = seed_render_decomp.load_checkpoint_state(
        _write_checkpoint(tmp_path, state=state)
    )

    with pytest.raises(SeedCohortUnavailable, match="predates the SEED uid range"):
        seed_render_decomp.exact_partition_masks(loaded.lineage_view)


def test_build_render_view_masks_all_parameters_without_aliasing_source_tensors():
    state = _checkpoint_state()
    before = {key: value.clone() for key, value in state["params"].items()}
    mask = torch.tensor([True, False, True, False, False])

    view = seed_render_decomp.build_render_view(state, mask=mask, device="cpu")

    assert view.active_sh_degree == 2
    assert set(view.params) == set(RENDER_KEYS)
    assert all(value.shape[0] == 2 for value in view.params.values())
    with torch.no_grad():
        view.params["means"][0, 0] = -999.0
    for key in RENDER_KEYS:
        assert torch.equal(state["params"][key], before[key])


def _write_rgbd_scene(scene_path, colors):
    scene_path.mkdir()
    (scene_path / "intrinsics.json").write_text(
        json.dumps(
            {
                "width": 8,
                "height": 6,
                "fx": 10.0,
                "fy": 11.0,
                "cx": 4.0,
                "cy": 3.0,
                "depth_scale": 1000.0,
            }
        )
    )
    records = []
    for index, color in enumerate(colors):
        rgb_name = f"rgb_{index:06d}.png"
        depth_name = f"depth_{index:06d}.png"
        Image.fromarray(np.full((6, 8, 3), color, dtype=np.uint8)).save(
            scene_path / rgb_name
        )
        Image.fromarray(np.full((6, 8), 1000, dtype=np.uint16)).save(
            scene_path / depth_name
        )
        records.append(
            {
                "id": index,
                "timestamp": float(index),
                "rgb": rgb_name,
                "depth": depth_name,
                "c2w": np.eye(4, dtype=np.float32).tolist(),
            }
        )
    (scene_path / "frames.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records)
    )


def _camera_entry(index):
    return {
        "id": index,
        "img_name": f"{index:06d}",
        "width": 8,
        "height": 6,
        "position": [float(index), 0.0, 0.0],
        "rotation": np.eye(3, dtype=np.float32).tolist(),
        "fx": 10.0,
        "fy": 11.0,
        "cx": 4.0,
        "cy": 3.0,
    }


def test_load_fixed_seed_view_always_uses_saved_camera_zero_and_its_source_gt(tmp_path):
    model_path = tmp_path / "run"
    scene_path = tmp_path / "scene"
    model_path.mkdir()
    _write_rgbd_scene(scene_path, colors=[10, 80])
    entries = [_camera_entry(1), _camera_entry(0)]
    (model_path / "cameras.json").write_text(json.dumps(entries))
    (model_path / "cfg_args").write_text(repr(Namespace(source_path="stale")))

    fixed = seed_render_decomp.load_fixed_seed_view(
        model_path, scene_path, device="cpu"
    )

    assert fixed.camera_index == 0
    assert fixed.entry == entries[0]
    assert fixed.camera.image_name == "000001"
    assert fixed.gt.shape == (3, 6, 8)
    torch.testing.assert_close(fixed.gt, torch.full((3, 6, 8), 80 / 255.0))
    assert len(fixed.camera_entry_sha256) == 64
    assert len(fixed.camera_file_sha256) == 64
    assert len(fixed.gt_sha256) == 64


def test_synthetic_pass_metrics_record_raw_range_then_score_explicit_clamp():
    rgb = torch.tensor([[[-0.25, 1.25]]] * 3)
    gt, _ = declare_canonical_score_tensor(
        torch.zeros((3, 1, 2)), context="synthetic gt"
    )
    alpha = torch.tensor([[[0.6], [1.0]]])  # HWC renderer shape
    depth_ed = torch.tensor([[[3.0, 5.0]]])  # CHW renderer shape

    metrics = seed_render_decomp.compute_pass_metrics(
        rgb, gt, alpha, depth_ed, alpha_threshold=0.5
    )

    assert metrics["raw_rgb_range"] == {
        "dtype": "torch.float32",
        "shape": [3, 1, 2],
        "element_count": 6,
        "finite_count": 6,
        "nonfinite_count": 0,
        "below_zero_count": 3,
        "above_one_count": 3,
        "finite_min": pytest.approx(-0.25),
        "finite_max": pytest.approx(1.25),
    }
    assert metrics["score_domain"] == {
        "status": "available",
        "input": "raw_renderer_rgb",
        "transform": "clamp_to_[0,1]",
        "provenance_boundary": "declare_canonical_score_tensor",
        "canonical_domain": "float_RGB_CHW_[0,1]",
    }
    assert metrics["psnr_clamped_db"] == pytest.approx(3.0102999)
    assert "psnr_db" not in metrics
    assert metrics["mean_abs_error_clamped"] == pytest.approx(0.5)
    assert metrics["mean_alpha"] == pytest.approx(0.8)
    assert metrics["alpha_coverage_fraction"] == pytest.approx(1.0)
    assert metrics["expected_depth_mean_m"] == pytest.approx(4.0)
    assert metrics["expected_depth_valid_fraction"] == pytest.approx(1.0)


def test_nonfinite_raw_rgb_is_diagnosed_without_becoming_a_canonical_score():
    rgb = torch.tensor([[[float("nan")]]] * 3)
    gt, _ = declare_canonical_score_tensor(
        torch.zeros((3, 1, 1)), context="synthetic gt"
    )

    metrics = seed_render_decomp.compute_pass_metrics(
        rgb,
        gt,
        torch.ones(1, 1, 1),
        torch.ones(1, 1, 1),
        alpha_threshold=0.5,
    )

    assert metrics["raw_rgb_range"]["nonfinite_count"] == 3
    assert metrics["score_domain"]["status"] == "unavailable_nonfinite_raw_rgb"
    assert metrics["psnr_clamped_db"] is None


def test_synthetic_overlap_metrics_are_explicitly_approximate_and_alpha_gated():
    seed_alpha = torch.tensor([[[0.6], [0.7]], [[0.1], [0.8]]])
    nonseed_alpha = torch.tensor([[[0.9], [0.2]], [[0.9], [0.9]]])
    seed_depth = torch.tensor([[[1.0, 2.0], [3.0, 8.0]]])
    nonseed_depth = torch.tensor([[[2.0, 1.0], [4.0, 4.0]]])

    metrics = seed_render_decomp.compute_overlap_metrics(
        seed_alpha,
        seed_depth,
        nonseed_alpha,
        nonseed_depth,
        alpha_threshold=0.5,
    )

    assert metrics["semantics"] == "approximate_isolated_pass_overlap"
    assert metrics["alpha_overlap_fraction"] == pytest.approx(0.5)
    assert metrics["depth_comparable_fraction_of_image"] == pytest.approx(0.5)
    assert metrics["depth_comparable_fraction_within_alpha_overlap"] == pytest.approx(1.0)
    assert metrics["seed_in_front_fraction_within_comparable_overlap"] == pytest.approx(0.5)


def test_pipeline_reconstructs_dynamic_sparse_gradient_field_exactly():
    enabled = seed_render_decomp._pipeline_from_cfg(Namespace(gsplat_sparse_grad=True))
    disabled = seed_render_decomp._pipeline_from_cfg(Namespace(gsplat_sparse_grad=False))
    state = _checkpoint_state()
    state["params"]["quats"][0, 0] = 2.0
    view = seed_render_decomp.build_render_view(state, device="cpu")

    assert enabled.gsplat_sparse_grad is True
    assert disabled.gsplat_sparse_grad is False
    assert seed_render_decomp._plain_pipeline_dict(enabled)["gsplat_sparse_grad"] is True
    sparse_branch = view._rotation if enabled.gsplat_sparse_grad else view.get_rotation
    dense_branch = view._rotation if disabled.gsplat_sparse_grad else view.get_rotation
    assert torch.linalg.vector_norm(sparse_branch[0]).item() == pytest.approx(2.0)
    assert torch.linalg.vector_norm(dense_branch[0]).item() == pytest.approx(1.0)


def _fake_camera(gt):
    gt, provenance = declare_canonical_score_tensor(gt, context="fake camera GT")
    return Namespace(original_image=gt, score_provenance=provenance)


def test_render_checkpoint_uses_production_no_depth_then_ed_passes_and_preserves_source():
    state = _checkpoint_state()
    source_before = {key: value.clone() for key, value in state["params"].items()}
    loaded_view = seed_render_decomp.CheckpointLineageView(state)
    masks = seed_render_decomp.exact_partition_masks(loaded_view)
    gt = torch.zeros(3, 2, 2)
    camera = _fake_camera(gt)
    pipe = seed_render_decomp._pipeline_from_cfg(Namespace(gsplat_sparse_grad=True))
    calls = []

    def fake_render(_camera, model, resolved_pipe, _background, *, render_depth):
        calls.append((render_depth, len(model.params["means"]), resolved_pipe.gsplat_sparse_grad))
        package = {"render": torch.full((3, 2, 2), 1.25)}
        if render_depth:
            package.update(
                alpha=torch.ones(2, 2, 1),
                rendered_depth_ed=torch.full((1, 2, 2), 2.0),
            )
        return package

    result = seed_render_decomp.render_checkpoint(
        state,
        masks,
        camera,
        gt,
        pipe,
        torch.zeros(3),
        alpha_threshold=0.5,
        full_rgb_tolerance=0.0,
        device="cpu",
        render_fn=fake_render,
    )

    assert calls == [
        (False, 5, True),
        (True, 5, True),
        (True, 2, True),
        (True, 3, True),
        (True, 2, True),
    ]
    assert result["full_rgb_mode_parity"] == {
        "status": "passed",
        "raw_rgb_max_abs_delta": 0.0,
        "tolerance": 0.0,
        "production_call_render_depth": False,
        "decomposition_call_render_depth": True,
    }
    assert result["passes"]["full"]["psnr_clamped_db"] == pytest.approx(0.0)
    for key in RENDER_KEYS:
        assert torch.equal(state["params"][key], source_before[key])


def test_render_checkpoint_releases_each_model_before_constructing_the_next(monkeypatch):
    state = _checkpoint_state()
    masks = seed_render_decomp.exact_partition_masks(
        seed_render_decomp.CheckpointLineageView(state)
    )
    camera = _fake_camera(torch.zeros(3, 2, 2))
    events = []
    active_views = 0
    maximum_active_views = 0
    production_refs = {}

    class TrackedView:
        def __init__(self):
            nonlocal active_views, maximum_active_views
            active_views += 1
            maximum_active_views = max(maximum_active_views, active_views)
            events.append("build")

        def __del__(self):
            nonlocal active_views
            active_views -= 1
            events.append("release")

    def tracked_build(*_args, **_kwargs):
        gc.collect()
        if production_refs:
            assert production_refs["view"]() is None
            assert production_refs["rgb"]() is None
        assert active_views == 0
        return TrackedView()

    def fake_render(_camera, model, _pipe, _background, *, render_depth):
        rgb = torch.full((3, 2, 2), 0.25)
        package = {"render": rgb, "owner": model}
        if not render_depth:
            production_refs.update(view=weakref.ref(model), rgb=weakref.ref(rgb))
        else:
            package.update(
                alpha=torch.ones(2, 2, 1),
                rendered_depth_ed=torch.ones(1, 2, 2),
            )
        return package

    monkeypatch.setattr(seed_render_decomp, "build_render_view", tracked_build)
    monkeypatch.setattr(
        seed_render_decomp,
        "_empty_cuda_cache",
        lambda _device: events.append("empty_cache"),
        raising=False,
    )

    seed_render_decomp.render_checkpoint(
        state,
        masks,
        camera,
        camera.original_image,
        Namespace(gsplat_sparse_grad=True),
        torch.zeros(3),
        alpha_threshold=0.5,
        full_rgb_tolerance=0.0,
        device="cpu",
        render_fn=fake_render,
    )

    gc.collect()
    assert maximum_active_views == 1
    assert active_views == 0
    assert events[:4] == ["build", "release", "empty_cache", "build"]


def test_bundle_writes_required_artifacts_series_manifest_and_refuses_overwrite(tmp_path):
    out_dir = tmp_path / "analysis"
    seed_render_decomp.create_output_dir(out_dir)
    gt = torch.zeros(3, 2, 2)
    outputs = {}
    for index, name in enumerate(("full", "seed_root", "seed_lineage", "nonseed")):
        outputs[name] = {
            "rgb": torch.full((3, 2, 2), index / 4.0),
            "alpha": torch.full((1, 2, 2), index / 4.0),
            "expected_depth": torch.full((1, 2, 2), float(index + 1)),
        }

    artifact_paths = seed_render_decomp.write_checkpoint_artifacts(
        out_dir,
        250,
        outputs,
        gt,
        production_full_rgb=torch.full((3, 2, 2), 1.25),
    )
    record = {
        "iteration": 250,
        "counts": {"full": 5, "seed_root": 2, "seed_lineage": 3, "nonseed": 2},
        "mean_gaussian_opacity": {
            "full": 0.5,
            "seed_root": 0.75,
            "seed_lineage": 0.6,
            "nonseed": 0.35,
        },
        "production_full_no_depth": {"psnr_clamped_db": 0.5},
        "passes": {
            name: {"psnr_clamped_db": float(index + 1), "mean_alpha": 0.25}
            for index, name in enumerate(outputs)
        },
        "overlap": {"alpha_overlap_fraction": 0.5},
        "artifacts": artifact_paths,
    }
    seed_render_decomp.write_series(out_dir, [record])
    seed_render_decomp.write_manifest(out_dir, {"schema_version": 1})

    for pass_paths in artifact_paths.values():
        if pass_paths is artifact_paths["production_full_no_depth"]:
            assert set(pass_paths) == {"raw_rgb_npy", "rgb_clamped_png"}
            continue
        assert set(pass_paths) == {
            "raw_rgb_npy",
            "rgb_clamped_png",
            "raw_alpha_npy",
            "alpha_png",
            "expected_depth_npy",
            "expected_depth_png",
            "absolute_clamped_rgb_error_png",
        }
        for relative_path in pass_paths.values():
            assert (out_dir / relative_path).is_file()
    saved_raw = np.load(
        out_dir / artifact_paths["production_full_no_depth"]["raw_rgb_npy"]
    )
    assert saved_raw.max() == pytest.approx(1.25)
    jsonl_record = json.loads((out_dir / "series.jsonl").read_text().strip())
    assert jsonl_record["passes"]["full"]["psnr_clamped_db"] == 1.0
    with (out_dir / "series.csv").open(newline="") as handle:
        csv_record = next(csv.DictReader(handle))
    assert csv_record["full_psnr_clamped_db"] == "1.0"
    assert csv_record["production_full_no_depth_psnr_clamped_db"] == "0.5"
    assert csv_record["seed_root_mean_gaussian_opacity"] == "0.75"
    assert json.loads((out_dir / "manifest.json").read_text()) == {
        "schema_version": 1
    }
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        seed_render_decomp.create_output_dir(out_dir)


def _main_argv(model_path, scene_path, out_path, *, extra=()):
    return [
        "--model-path",
        str(model_path),
        "--scene",
        str(scene_path),
        "--iterations",
        "250",
        "--out",
        str(out_path),
        "--acknowledge-checkpoint-iteration-indeterminate",
        "--tool-commit",
        "a" * 40,
        *extra,
    ]


def _write_main_fixture(tmp_path, *, git_commit="c" * 40, git_dirty=True):
    model_path = tmp_path / "run"
    scene_path = tmp_path / "scene"
    model_path.mkdir()
    _write_rgbd_scene(scene_path, colors=[0])
    (model_path / "cameras.json").write_text(json.dumps([_camera_entry(0)]))
    (model_path / "cfg_args").write_text(
        repr(
            Namespace(
                source_path=str(scene_path),
                white_background=False,
                gsplat_sparse_grad=True,
                git_commit=git_commit,
                git_dirty=git_dirty,
            )
        )
    )
    _write_checkpoint(model_path)
    return model_path, scene_path


def _constant_renderer(*, no_depth_value=0.25, depth_value=0.25, fail_depth=False):
    calls = []

    def render(_camera, _model, pipe, _background, *, render_depth):
        calls.append((render_depth, pipe.gsplat_sparse_grad))
        if render_depth and fail_depth:
            raise RuntimeError("synthetic kernel failure")
        value = depth_value if render_depth else no_depth_value
        package = {"render": torch.full((3, 6, 8), value)}
        if render_depth:
            package.update(
                alpha=torch.ones(6, 8, 1),
                rendered_depth_ed=torch.ones(1, 6, 8),
            )
        return package

    return calls, render


def test_main_writes_complete_manifest_with_iteration_ack_and_artifact_integrity(tmp_path):
    model_path, scene_path = _write_main_fixture(tmp_path)
    out_path = tmp_path / "out"
    calls, renderer = _constant_renderer()

    assert seed_render_decomp.main(
        _main_argv(model_path, scene_path, out_path),
        device="cpu",
        render_fn=renderer,
    ) == 0

    manifest = json.loads((out_path / "manifest.json").read_text())
    assert manifest["completion_state"] == "COMPLETE"
    assert manifest["iteration_identity"]["point_in_time"] == "indeterminate"
    assert manifest["iteration_identity"]["acknowledged"] is True
    assert manifest["render"]["pipeline"]["gsplat_sparse_grad"] is True
    assert manifest["realized_iterations"] == [250]
    assert manifest["expected_record_count"] == 1
    assert manifest["realized_record_count"] == 1
    assert manifest["tool"]["commit_identity_source"] == "cli"
    assert (
        manifest["tool"]["commit_verification"]["status"]
        == "declared_unverified_git_unavailable"
    )
    assert set(manifest["output_integrity"]) >= {
        "series.jsonl",
        "series.csv",
        "iteration_000250/full/raw_rgb.npy",
    }
    for relative, identity in manifest["output_integrity"].items():
        assert identity["size_bytes"] == (out_path / relative).stat().st_size
        assert len(identity["sha256"]) == 64
    assert calls == [(False, True)] + [(True, True)] * 4


def test_main_preserves_failed_bundle_when_renderer_kernel_raises(tmp_path):
    model_path, scene_path = _write_main_fixture(tmp_path)
    out_path = tmp_path / "failed"
    _, renderer = _constant_renderer(fail_depth=True)

    with pytest.raises(RuntimeError, match="synthetic kernel failure"):
        seed_render_decomp.main(
            _main_argv(model_path, scene_path, out_path),
            device="cpu",
            render_fn=renderer,
        )

    manifest = json.loads((out_path / "manifest.json").read_text())
    assert manifest["completion_state"] == "FAILED"
    assert manifest["realized_iterations"] == []
    assert manifest["realized_record_count"] == 0
    assert manifest["exception"] == {
        "type": "RuntimeError",
        "message": "synthetic kernel failure",
    }


def test_main_preserves_raw_artifacts_and_fails_mode_parity_after_writing_record(tmp_path):
    model_path, scene_path = _write_main_fixture(tmp_path)
    out_path = tmp_path / "parity_failed"
    calls, renderer = _constant_renderer(no_depth_value=0.1, depth_value=1.25)

    with pytest.raises(RuntimeError, match="production/decomposition RGB parity failed"):
        seed_render_decomp.main(
            _main_argv(
                model_path,
                scene_path,
                out_path,
                extra=("--full-rgb-mode-tolerance", "0.01"),
            ),
            device="cpu",
            render_fn=renderer,
        )

    manifest = json.loads((out_path / "manifest.json").read_text())
    record = json.loads((out_path / "series.jsonl").read_text().strip())
    assert manifest["completion_state"] == "FAILED"
    assert manifest["realized_iterations"] == [250]
    assert manifest["realized_record_count"] == 1
    assert record["full_rgb_mode_parity"]["raw_rgb_max_abs_delta"] == pytest.approx(1.15)
    assert np.load(out_path / record["artifacts"]["full"]["raw_rgb_npy"]).max() == pytest.approx(1.25)
    assert calls == [(False, True), (True, True)]
    assert set(record["artifacts"]) == {"production_full_no_depth", "full"}


def _compatible_manifest():
    return {
        "schema_version": 2,
        "analysis": "exact_seed_lineage_fixed_view_render_decomposition",
        "tool": {
            "script": "scripts/seed_render_decomp.py",
            "script_sha256": "a" * 64,
        },
        "run": {
            "source_git_commit": "c" * 40,
            "source_git_dirty": True,
        },
        "camera": {
            "index": 0,
            "entry": {"id": 0},
            "entry_sha256": "camera",
            "gt_tensor_sha256": "gt",
            "gt_dtype": "torch.float32",
            "gt_shape": [3, 6, 8],
            "resolution_hw": [6, 8],
        },
        "render": {
            "production_full_pass": {"render_depth": False},
            "decomposition_passes": ["full", "seed_root", "seed_lineage", "nonseed"],
            "decomposition_render_depth": True,
            "expected_depth_key": "rendered_depth_ed",
            "full_rgb_mode_tolerance": 1e-5,
            "background_rgb": [0.0, 0.0, 0.0],
            "alpha_threshold": 0.05,
            "pipeline": {"gsplat_sparse_grad": True},
            "canonical_rgb_metric_contract": {
                "input": "raw_renderer_rgb",
                "transform": "clamp_to_[0,1]",
                "provenance_boundary": "declare_canonical_score_tensor",
                "metric_name": "psnr_clamped_db",
            },
        },
        "checkpoint_iteration_grid": [100, 200],
        "expected_record_count": 2,
        "realized_record_count": 2,
        "realized_iterations": [100, 200],
        "checkpoints": [
            {"iteration": 100, "filename": "chkpnt100.pth", "sha256": "1" * 64},
            {"iteration": 200, "filename": "chkpnt200.pth", "sha256": "2" * 64},
        ],
        "series": {"jsonl": "series.jsonl", "csv": "series.csv"},
    }


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("schema_version",), 3),
        (("analysis",), "different_analysis"),
        (("tool", "script_sha256"), "b" * 64),
        (("run", "source_git_dirty"), False),
        (("camera", "entry"), {"id": 1}),
        (("camera", "gt_tensor_sha256"), "different"),
        (("camera", "resolution_hw"), [8, 6]),
        (("render", "background_rgb"), [1.0, 1.0, 1.0]),
        (("render", "pipeline"), {"gsplat_sparse_grad": False}),
        (("render", "alpha_threshold"), 0.1),
        (("render", "canonical_rgb_metric_contract", "transform"), "different"),
        (("render", "production_full_pass", "render_depth"), True),
        (("render", "decomposition_passes"), ["full"]),
        (("render", "decomposition_render_depth"), False),
        (("render", "expected_depth_key"), "different"),
        (("render", "full_rgb_mode_tolerance"), 0.1),
        (("checkpoint_iteration_grid",), [100]),
        (("expected_record_count",), 1),
        (("realized_record_count",), 1),
        (("realized_iterations",), [100]),
    ],
)
def test_reference_manifest_comparison_fails_closed_on_each_comparison_axis(
    path, replacement
):
    current = _compatible_manifest()
    reference = copy.deepcopy(current)
    target = reference
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = replacement

    with pytest.raises(ValueError, match="reference manifest incompatible"):
        seed_render_decomp.validate_reference_compatibility(current, reference)


def test_reference_source_commit_mismatch_requires_narrow_reasoned_acknowledgement():
    current = _compatible_manifest()
    reference = copy.deepcopy(current)
    reference["run"]["source_git_commit"] = "d" * 40

    with pytest.raises(ValueError, match="source git commit mismatch"):
        seed_render_decomp.validate_reference_compatibility(current, reference)

    policy = seed_render_decomp.validate_reference_compatibility(
        current,
        reference,
        source_commit_mismatch_reason="audited empty training-path diff",
    )

    assert policy == {
        "status": "acknowledged_diagnostic_source_commit_mismatch",
        "current_source_git_commit": "c" * 40,
        "reference_source_git_commit": "d" * 40,
        "reason": "audited empty training-path diff",
    }


def test_reference_metric_deltas_are_iteration_aligned_and_named():
    current = [
        {"iteration": 100, "passes": {"full": {"psnr_clamped_db": 20.5}}},
        {"iteration": 200, "passes": {"full": {"psnr_clamped_db": 22.0}}},
    ]
    reference = [
        {"iteration": 100, "passes": {"full": {"psnr_clamped_db": 20.0}}},
        {"iteration": 200, "passes": {"full": {"psnr_clamped_db": 23.0}}},
    ]

    assert seed_render_decomp.compute_reference_metric_deltas(current, reference) == [
        {
            "iteration": 100,
            "metric_deltas_current_minus_reference": {"passes.full.psnr_clamped_db": 0.5},
        },
        {
            "iteration": 200,
            "metric_deltas_current_minus_reference": {"passes.full.psnr_clamped_db": -1.0},
        },
    ]


def test_main_reference_manifest_emits_aligned_metric_delta_series(tmp_path):
    model_path, scene_path = _write_main_fixture(tmp_path)
    reference_out = tmp_path / "reference"
    _, reference_renderer = _constant_renderer(
        no_depth_value=0.25, depth_value=0.25
    )
    seed_render_decomp.main(
        _main_argv(model_path, scene_path, reference_out),
        device="cpu",
        render_fn=reference_renderer,
    )
    current_out = tmp_path / "current"
    _, current_renderer = _constant_renderer(
        no_depth_value=0.5, depth_value=0.5
    )

    seed_render_decomp.main(
        _main_argv(
            model_path,
            scene_path,
            current_out,
            extra=("--reference-manifest", str(reference_out / "manifest.json")),
        ),
        device="cpu",
        render_fn=current_renderer,
    )

    manifest = json.loads((current_out / "manifest.json").read_text())
    comparison = json.loads(
        (current_out / "reference_comparison.jsonl").read_text().strip()
    )
    assert manifest["reference_comparison"]["status"] == "complete"
    assert manifest["reference_comparison"]["aligned_iterations"] == [250]
    assert (
        comparison["metric_deltas_current_minus_reference"][
            "passes.full.psnr_clamped_db"
        ]
        == pytest.approx(-6.0205999)
    )
    assert "reference_comparison.jsonl" in manifest["output_integrity"]


@pytest.mark.parametrize("mutation", ["same_size", "truncated"])
def test_main_rejects_mutated_reference_series_before_render(tmp_path, mutation):
    model_path, scene_path = _write_main_fixture(tmp_path)
    reference_out = tmp_path / "reference"
    _, reference_renderer = _constant_renderer()
    seed_render_decomp.main(
        _main_argv(model_path, scene_path, reference_out),
        device="cpu",
        render_fn=reference_renderer,
    )
    series_path = reference_out / "series.jsonl"
    payload = series_path.read_bytes()
    if mutation == "same_size":
        payload = payload.replace(b'"iteration": 250', b'"iteration": 251', 1)
    else:
        payload = payload[:-1]
    series_path.write_bytes(payload)
    calls, renderer = _constant_renderer()

    with pytest.raises(ValueError, match="reference series integrity mismatch"):
        seed_render_decomp.main(
            _main_argv(
                model_path,
                scene_path,
                tmp_path / "current",
                extra=("--reference-manifest", str(reference_out / "manifest.json")),
            ),
            device="cpu",
            render_fn=renderer,
        )

    assert calls == []


def test_main_rejects_unsafe_reference_series_path_before_render(tmp_path):
    model_path, scene_path = _write_main_fixture(tmp_path)
    reference_out = tmp_path / "reference"
    _, reference_renderer = _constant_renderer()
    seed_render_decomp.main(
        _main_argv(model_path, scene_path, reference_out),
        device="cpu",
        render_fn=reference_renderer,
    )
    manifest_path = reference_out / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["series"]["jsonl"] = "../series.jsonl"
    manifest_path.write_text(json.dumps(manifest))
    calls, renderer = _constant_renderer()

    with pytest.raises(ValueError, match="safe relative path"):
        seed_render_decomp.main(
            _main_argv(
                model_path,
                scene_path,
                tmp_path / "current",
                extra=("--reference-manifest", str(manifest_path)),
            ),
            device="cpu",
            render_fn=renderer,
        )

    assert calls == []


def test_main_rejects_reference_record_checkpoint_identity_before_render(tmp_path):
    model_path, scene_path = _write_main_fixture(tmp_path)
    reference_out = tmp_path / "reference"
    _, reference_renderer = _constant_renderer()
    seed_render_decomp.main(
        _main_argv(model_path, scene_path, reference_out),
        device="cpu",
        render_fn=reference_renderer,
    )
    series_path = reference_out / "series.jsonl"
    record = json.loads(series_path.read_text())
    record["checkpoint_sha256"] = "f" * 64
    series_path.write_text(json.dumps(record, sort_keys=True) + "\n")
    manifest_path = reference_out / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["output_integrity"]["series.jsonl"] = {
        "size_bytes": series_path.stat().st_size,
        "sha256": seed_render_decomp.file_sha256(series_path),
    }
    manifest_path.write_text(json.dumps(manifest))
    calls, renderer = _constant_renderer()

    with pytest.raises(ValueError, match="reference record checkpoint identity"):
        seed_render_decomp.main(
            _main_argv(
                model_path,
                scene_path,
                tmp_path / "current",
                extra=("--reference-manifest", str(manifest_path)),
            ),
            device="cpu",
            render_fn=renderer,
        )

    assert calls == []


def test_main_source_commit_mismatch_requires_and_records_narrow_acknowledgement(tmp_path):
    model_path, scene_path = _write_main_fixture(tmp_path, git_commit="c" * 40)
    reference_out = tmp_path / "reference"
    _, reference_renderer = _constant_renderer()
    seed_render_decomp.main(
        _main_argv(model_path, scene_path, reference_out),
        device="cpu",
        render_fn=reference_renderer,
    )
    (model_path / "cfg_args").write_text(
        repr(
            Namespace(
                source_path=str(scene_path),
                white_background=False,
                gsplat_sparse_grad=True,
                git_commit="d" * 40,
                git_dirty=True,
            )
        )
    )
    calls, renderer = _constant_renderer()

    with pytest.raises(ValueError, match="source git commit mismatch"):
        seed_render_decomp.main(
            _main_argv(
                model_path,
                scene_path,
                tmp_path / "rejected",
                extra=("--reference-manifest", str(reference_out / "manifest.json")),
            ),
            device="cpu",
            render_fn=renderer,
        )
    assert calls == []

    reason = "audited empty training-path diff"
    seed_render_decomp.main(
        _main_argv(
            model_path,
            scene_path,
            tmp_path / "acknowledged",
            extra=(
                "--reference-manifest",
                str(reference_out / "manifest.json"),
                "--acknowledge-reference-source-commit-mismatch",
                reason,
            ),
        ),
        device="cpu",
        render_fn=renderer,
    )
    manifest = json.loads((tmp_path / "acknowledged" / "manifest.json").read_text())
    assert manifest["reference_comparison"]["source_commit_policy"] == {
        "status": "acknowledged_diagnostic_source_commit_mismatch",
        "current_source_git_commit": "d" * 40,
        "reference_source_git_commit": "c" * 40,
        "reason": reason,
    }


def test_tool_identity_requires_full_sha_and_records_source_and_verification(monkeypatch):
    commit = "a" * 40
    script_blob = "b" * 40

    def fake_git_output(*args):
        if args[0] == "rev-parse":
            if args[1].endswith(":scripts/seed_render_decomp.py"):
                return script_blob
            return commit
        if args[0] == "hash-object":
            return script_blob
        if args[0] == "status":
            return ""
        if args[0] == "branch":
            return "fix/test"
        raise AssertionError(args)

    monkeypatch.setattr(seed_render_decomp, "_git_output", fake_git_output)

    identity = seed_render_decomp._tool_identity(commit)

    assert identity["commit"] == commit
    assert identity["commit_identity_source"] == "cli"
    assert identity["commit_verification"] == {
        "status": "verified_commit_and_script_blob",
        "resolved_commit": commit,
        "script_blob": script_blob,
    }
    with pytest.raises(RuntimeError, match="full 40- or 64-hex"):
        seed_render_decomp._tool_identity("abc1234")
