"""CPU-only tests for render_report.py's frozen-checkpoint regeneration
entry point (issue #401).

Scope: this file verifies everything that does NOT require CUDA/a real
rasterizer -- pose reconstruction (device="cpu" override), GT/depth
reattachment against a synthetic on-disk dataset, the regeneration manifest
schema, and the --full-report dispatch/idempotency plumbing (with the heavy
`load_gaussians`/`run_full_report` calls monkeypatched to stubs). It does
NOT exercise actual rendering (write_post_training_report's sheet/video
generation) -- that path is GPU-gated and is not verified by this file; see
the PR description for the honest scope of what remains unverified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import types

import numpy as np
import pytest
from PIL import Image

# `smoke` (.github/workflows/ci-cpu.yml, Dockerfile.ci-cpu) runs
# `python -m pytest -q tests/` in a deliberately torch-free image, where a
# module-level `import torch` is a COLLECTION error -- which aborts the whole
# run, not just this file (`Interrupted: 1 error during collection`), taking
# every other test down with it. That is the exact bug class #449/#450 cleaned
# up. importorskip turns it into a clean per-module skip instead.
#
# `render_report` is guarded for the same reason and must stay BELOW this line:
# it pulls torch transitively, so importing it first would reintroduce the
# collection abort even with torch itself guarded.
#
# Deliberately NOT fixed by adding this file to tests/conftest.py's
# collect_ignore: that hides it from the full training image too, where these
# tests are meant to run (see #462 on conftest's 149 unconditional ignores).
# numpy/pillow need no guard -- both are installed in the CPU image.
torch = pytest.importorskip("torch")

rr = pytest.importorskip("render_report")


@pytest.fixture(autouse=True)
def _fake_gaussian_renderer(monkeypatch):
    """`_process_full_report` does `from gaussian_renderer import render as
    render_fn` -- the real module imports `gsplat`, a ROCm/CUDA extension
    only installed inside the training container, not on this CPU-only test
    host. Every real render call in these tests is itself monkeypatched
    (via `run_full_report`), so the actual `render_fn` object is never
    invoked -- stub the module import out so the plumbing under test can run
    here at all. Real end-to-end rendering is exercised by nothing in this
    file; see the PR description for that gap."""
    fake_module = types.ModuleType("gaussian_renderer")
    fake_module.render = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "gaussian_renderer", fake_module)
    yield


# ---------------------------------------------------------------------------
# _camera_json_to_minicam: CPU pose reconstruction
# ---------------------------------------------------------------------------

def test_camera_json_to_minicam_cpu_device_no_cuda_touch():
    """device="cpu" must not touch CUDA at all -- this is the whole point of
    the override (testability on a CPU-only host)."""
    entry = {
        "id": 0, "img_name": "000000",
        "width": 64, "height": 48,
        "position": [1.0, 2.0, 3.0],
        "rotation": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        "fx": 50.0, "fy": 50.0, "cx": 32.0, "cy": 24.0,
    }
    cam = rr._camera_json_to_minicam(entry, device="cpu")
    assert cam.world_view_transform.device.type == "cpu"
    assert cam.image_name == "000000"
    # camera_center recovers the original position (identity rotation case).
    np.testing.assert_allclose(cam.camera_center.numpy(), [1.0, 2.0, 3.0], atol=1e-4)
    assert cam.fx == 50.0 and cam.fy == 50.0
    assert cam.cx == 32.0 and cam.cy == 24.0


def test_camera_json_to_minicam_roundtrip_c2w_convention():
    """Regression guard for the C2W/W2C confusion fixed in 5de4cb7: build a
    camera_to_JSON-style entry from a KNOWN c2w pose (a 90-degree rotation
    about Y plus a translation), decode it back, and confirm the recovered
    world_view_transform inverts to the same c2w -- not its transpose or a
    mirrored variant."""
    c2w = np.array([
        [0.0, 0.0, 1.0, 5.0],
        [0.0, 1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0, -2.0],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=np.float32)
    entry = {
        "id": 1, "img_name": "000001", "width": 32, "height": 32,
        "position": c2w[:3, 3].tolist(),
        "rotation": [row.tolist() for row in c2w[:3, :3]],
        "fx": 20.0, "fy": 20.0, "cx": 16.0, "cy": 16.0,
    }
    cam = rr._camera_json_to_minicam(entry, device="cpu")
    # world_view_transform is stored TRANSPOSED (row-major -> column-major
    # convention used throughout scene/cameras.py); undo that to recover W2C.
    w2c = cam.world_view_transform.numpy().T
    recovered_c2w = np.linalg.inv(w2c)
    np.testing.assert_allclose(recovered_c2w, c2w, atol=1e-4)


# ---------------------------------------------------------------------------
# attach_source_gt
# ---------------------------------------------------------------------------

def _write_rgbd_sequence_dataset(root: str, n_frames: int = 3):
    """Minimal RGBDSequence layout: frames.jsonl + intrinsics.json + tiny
    per-frame RGB/depth PNGs, one distinct solid color per frame index so a
    test can assert GT pixel content, not just presence."""
    os.makedirs(root, exist_ok=True)
    intr = {
        "width": 8, "height": 8, "fx": 10.0, "fy": 10.0, "cx": 4.0, "cy": 4.0,
        "depth_scale": 1000.0,
    }
    with open(os.path.join(root, "intrinsics.json"), "w") as f:
        json.dump(intr, f)

    c2w = np.eye(4, dtype=np.float32).tolist()
    records = []
    for i in range(n_frames):
        rgb_name = f"rgb_{i:06d}.png"
        depth_name = f"depth_{i:06d}.png"
        color_val = 10 * (i + 1)
        rgb = np.full((8, 8, 3), color_val, dtype=np.uint8)
        Image.fromarray(rgb).save(os.path.join(root, rgb_name))
        depth = np.full((8, 8), 500 * (i + 1), dtype=np.uint16)
        Image.fromarray(depth).save(os.path.join(root, depth_name))
        records.append({
            "id": i, "timestamp": float(i), "rgb": rgb_name, "depth": depth_name,
            "c2w": c2w,
        })
    with open(os.path.join(root, "frames.jsonl"), "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    return root, records


def _fake_pose_only_cam(image_name: str, width: int = 8, height: int = 8):
    cam = argparse.Namespace()
    cam.image_name = image_name
    cam.image_width = width
    cam.image_height = height
    return cam


def test_attach_source_gt_matches_by_frame_index(tmp_path):
    root, records = _write_rgbd_sequence_dataset(str(tmp_path / "dataset"), n_frames=3)
    cams = [_fake_pose_only_cam(f"{i:06d}") for i in range(3)]
    cfg_args = argparse.Namespace(source_path=root)

    stats = rr.attach_source_gt(cams, cfg_args)

    assert stats["n_total"] == 3
    assert stats["n_matched"] == 3
    assert stats["reason"] is None
    for i, cam in enumerate(cams):
        assert isinstance(cam.original_image, torch.Tensor)
        assert cam.original_image.shape == (3, 8, 8)
        expected = 10 * (i + 1) / 255.0
        np.testing.assert_allclose(cam.original_image.numpy(), expected, atol=1e-6)
        assert cam._streaming_depth_path.endswith(f"depth_{i:06d}.png")
        assert cam._streaming_depth_scale == 1000.0
        assert cam._streaming_frame is not None


def test_attach_source_gt_partial_match_reports_unresolved(tmp_path):
    root, _ = _write_rgbd_sequence_dataset(str(tmp_path / "dataset"), n_frames=2)
    cams = [_fake_pose_only_cam("000000"), _fake_pose_only_cam("000099")]
    cfg_args = argparse.Namespace(source_path=root)

    stats = rr.attach_source_gt(cams, cfg_args)

    assert stats["n_matched"] == 1
    assert stats["reason"] is not None and "1/2" in stats["reason"]
    assert hasattr(cams[0], "original_image")
    assert not hasattr(cams[1], "original_image")


def test_attach_source_gt_missing_source_path_degrades_gracefully(tmp_path):
    cams = [_fake_pose_only_cam("000000")]
    cfg_args = argparse.Namespace(source_path=str(tmp_path / "does_not_exist"))

    stats = rr.attach_source_gt(cams, cfg_args)

    assert stats["n_matched"] == 0
    assert "not reachable" in stats["reason"]
    assert not hasattr(cams[0], "original_image")


def test_attach_source_gt_no_parseable_image_name(tmp_path):
    root, _ = _write_rgbd_sequence_dataset(str(tmp_path / "dataset"), n_frames=1)
    cam = _fake_pose_only_cam("frame_not_an_int")
    cfg_args = argparse.Namespace(source_path=root)

    stats = rr.attach_source_gt([cam], cfg_args)

    assert stats["n_matched"] == 0
    assert "parseable" in stats["reason"]


def test_attach_source_gt_empty_cams_list_is_a_noop():
    stats = rr.attach_source_gt([], argparse.Namespace(source_path="/nonexistent"))
    assert stats == {"n_total": 0, "n_matched": 0, "reason": None}


def test_attach_source_gt_source_path_override_takes_precedence(tmp_path):
    root, _ = _write_rgbd_sequence_dataset(str(tmp_path / "real_dataset"), n_frames=1)
    cams = [_fake_pose_only_cam("000000")]
    # cfg_args points at a WRONG/stale path; the explicit override wins.
    cfg_args = argparse.Namespace(source_path=str(tmp_path / "stale_path"))

    stats = rr.attach_source_gt(cams, cfg_args, source_path=root)

    assert stats["n_matched"] == 1


# ---------------------------------------------------------------------------
# Source-derived causal holdout recovery
# ---------------------------------------------------------------------------

def test_load_source_holdout_cameras_recovers_exact_finite_split(tmp_path):
    root, _ = _write_rgbd_sequence_dataset(str(tmp_path / "dataset"), n_frames=10)

    cams, metadata = rr.load_source_holdout_cameras(
        root,
        eval_hold=2,
        initial_frames=1,
        eval_guard=0,
        min_baseline_m=0.03,
        min_angle_deg=2.0,
        expected_n_test=4,
        expected_realized_hash="d7aaa3fdc32b4c88",
        device="cpu",
    )

    assert [cam.image_name for cam in cams] == [
        "000002", "000004", "000006", "000008",
    ]
    assert metadata["n_holdout"] == 4
    assert metadata["realized_holdout_hash"] == "d7aaa3fdc32b4c88"
    assert metadata["scored_holdout_hash"] == "d7aaa3fdc32b4c88"


def test_load_source_holdout_cameras_fails_closed_on_expected_hash_mismatch(tmp_path):
    root, _ = _write_rgbd_sequence_dataset(str(tmp_path / "dataset"), n_frames=10)

    with pytest.raises(ValueError, match="realized holdout hash mismatch"):
        rr.load_source_holdout_cameras(
            root,
            eval_hold=2,
            initial_frames=1,
            eval_guard=0,
            min_baseline_m=0.03,
            min_angle_deg=2.0,
            expected_n_test=4,
            expected_realized_hash="0000000000000000",
            device="cpu",
        )


def test_validate_source_rescore_destination_refuses_source_run_and_reuse(tmp_path):
    model_path = tmp_path / "source-run"
    model_path.mkdir()

    with pytest.raises(ValueError, match="outside the source run"):
        rr.validate_source_rescore_destination(
            str(model_path), str(model_path / "comparison" / "rescore")
        )

    occupied = tmp_path / "occupied"
    occupied.mkdir()
    with pytest.raises(FileExistsError, match="already exists"):
        rr.validate_source_rescore_destination(str(model_path), str(occupied))

    fresh = tmp_path / "fresh-rescore"
    assert rr.validate_source_rescore_destination(
        str(model_path), str(fresh)
    ) == str(fresh.resolve())


def test_resolve_checkpoint_cfg_requires_explicit_renderer_when_cfg_args_missing(tmp_path):
    model_path = tmp_path / "legacy-run"
    model_path.mkdir()
    missing = argparse.Namespace(
        source_holdout_path=str(tmp_path / "dataset"),
        checkpoint_sh_degree=None,
        checkpoint_model_layout=None,
        checkpoint_rasterize_mode=None,
        checkpoint_white_background=None,
    )
    with pytest.raises(ValueError, match="explicit frozen-checkpoint renderer"):
        rr.resolve_checkpoint_cfg(str(model_path), missing)

    explicit = argparse.Namespace(
        source_holdout_path=str(tmp_path / "dataset"),
        checkpoint_sh_degree=3,
        checkpoint_model_layout="gsplat",
        checkpoint_rasterize_mode="classic",
        checkpoint_white_background=False,
    )
    cfg = rr.resolve_checkpoint_cfg(str(model_path), explicit)
    assert cfg.sh_degree == 3
    assert cfg.model_layout == "gsplat"
    assert cfg.rasterize_mode == "classic"
    assert cfg.white_background is False
    assert cfg.source_path == explicit.source_holdout_path


def test_process_source_holdout_rescore_uses_external_output_and_exact_metadata(
    tmp_path, monkeypatch
):
    model_path, ply_path = _make_full_report_run_dir(tmp_path)
    out_dir = str(tmp_path / "external-rescore")
    cfg_args = argparse.Namespace(
        sh_degree=3,
        model_layout="gsplat",
        white_background=False,
        rasterize_mode="classic",
    )
    args = argparse.Namespace(
        source_holdout_path=str(tmp_path / "dataset"),
        streaming_eval_hold=8,
        streaming_initial_frames=5,
        streaming_eval_guard=0,
        streaming_eval_min_baseline_m=0.03,
        streaming_eval_min_angle_deg=2.0,
        expected_n_test=315,
        expected_realized_holdout_hash="7f11555c18457368",
    )
    os.makedirs(args.source_holdout_path)
    cameras = [_fake_pose_only_cam(f"{8 * (i + 1):06d}") for i in range(315)]
    metadata = {
        "n_holdout": 315,
        "realized_holdout_hash": "7f11555c18457368",
        "scored_holdout_hash": "7f11555c18457368",
    }
    monkeypatch.setattr(
        rr, "load_source_holdout_cameras", lambda *a, **k: (cameras, metadata)
    )
    monkeypatch.setattr(rr, "load_gaussians", lambda *a, **k: _FakeGaussians())
    calls = []

    def _fake_run_eval(*a, **k):
        calls.append(k)
        report_path = os.path.join(out_dir, "report.json")
        os.makedirs(out_dir)
        with open(report_path, "w") as f:
            json.dump({
                "eval_valid": True,
                "n_test": 315,
                "mean_test_psnr": 23.0,
                "eval_metadata": metadata,
            }, f)
        return {
            "eval_valid": True,
            "n_test": 315,
            "mean_test_psnr": 23.0,
            "eval_metadata": metadata,
            "artifacts": [report_path],
        }

    monkeypatch.setattr(rr, "run_eval", _fake_run_eval)

    rr._process_source_holdout_rescore(
        args,
        model_path,
        cfg_args,
        1000,
        ply_path,
        pipe=None,
        background=None,
        out_dir=out_dir,
        device="cpu",
    )

    assert len(calls) == 1
    assert calls[0]["eval_metadata"] == metadata
    assert calls[0]["test_metrics_max"] == 315
    assert os.path.exists(os.path.join(out_dir, "report.json"))
    with open(os.path.join(out_dir, "regeneration_manifest.json")) as f:
        manifest = json.load(f)
    assert manifest["camera_set"]["n_test"] == 315
    assert (
        manifest["camera_set"]["train_trajectory_source"]
        == "source_derived_causal_holdout"
    )


# ---------------------------------------------------------------------------
# regeneration manifest
# ---------------------------------------------------------------------------

def test_build_regeneration_manifest_schema(tmp_path):
    ply_path = tmp_path / "point_cloud.ply"
    ply_path.write_bytes(b"fake ply bytes")
    cfg_args = argparse.Namespace(git_commit="abc123", git_branch="main", git_dirty=False)

    manifest = rr.build_regeneration_manifest(
        model_path=str(tmp_path), iteration=5000, ply_path=str(ply_path), cfg_args=cfg_args,
        trajectory_source="recorded",
        gt_stats={"n_total": 4, "n_matched": 3, "reason": "1/4 frame indices not found"},
        n_train=4, n_test=2, test_has_gt=True,
        summary={"trajectory_fps": 24, "artifacts": ["a.png", "b.mp4"]},
        render_width=640, render_height=480,
    )

    assert manifest["schema"] == "eligsir.render_report.regeneration_manifest.v1"
    assert manifest["checkpoint"]["iteration"] == 5000
    assert manifest["checkpoint"]["ply_path"] == str(ply_path)
    assert manifest["checkpoint"]["ply_stat"]["size_bytes"] == len(b"fake ply bytes")
    assert manifest["render_config"] == {"width": 640, "height": 480, "trajectory_fps": 24}
    assert manifest["camera_set"]["n_train"] == 4
    assert manifest["camera_set"]["n_train_with_gt"] == 3
    assert manifest["camera_set"]["train_trajectory_source"] == "recorded"
    assert manifest["camera_set"]["n_test"] == 2
    assert manifest["camera_set"]["test_has_gt"] is True
    assert manifest["code_revision"]["training_run"] == {
        "git_commit": "abc123", "git_branch": "main", "git_dirty": False,
    }
    assert "git_commit" in manifest["code_revision"]["regeneration"]
    assert manifest["artifacts"] == ["a.png", "b.mp4"]


def test_build_regeneration_manifest_missing_ply_does_not_crash(tmp_path):
    cfg_args = argparse.Namespace()
    manifest = rr.build_regeneration_manifest(
        model_path=str(tmp_path), iteration=1, ply_path=str(tmp_path / "missing.ply"),
        cfg_args=cfg_args, trajectory_source="dense",
        gt_stats={"n_total": 0, "n_matched": 0, "reason": None},
        n_train=0, n_test=0, test_has_gt=False, summary={},
    )
    assert "ply_stat" not in manifest["checkpoint"]
    assert manifest["code_revision"]["training_run"] is None


def test_write_regeneration_manifest_round_trips_through_json(tmp_path):
    cfg_args = argparse.Namespace()
    path = rr.write_regeneration_manifest(
        str(tmp_path), model_path=str(tmp_path), iteration=1, ply_path="x.ply",
        cfg_args=cfg_args, trajectory_source="recorded",
        gt_stats={"n_total": 0, "n_matched": 0, "reason": None},
        n_train=0, n_test=0, test_has_gt=False, summary={},
    )
    assert os.path.basename(path) == "regeneration_manifest.json"
    with open(path) as f:
        loaded = json.load(f)
    assert loaded["schema"] == "eligsir.render_report.regeneration_manifest.v1"


# ---------------------------------------------------------------------------
# --full-report dispatch, idempotency, and the "checkpoint state unchanged"
# acceptance requirement (issue #401's own acceptance bullet)
# ---------------------------------------------------------------------------

class _FakeGaussians:
    pass


def _make_full_report_run_dir(tmp_path, n_train=2, n_test=2):
    model_path = tmp_path / "run"
    model_path.mkdir()
    ply_dir = model_path / "point_cloud" / "iteration_1000"
    ply_dir.mkdir(parents=True)
    ply_path = ply_dir / "point_cloud.ply"
    ply_path.write_bytes(b"totally real gaussian data, do not touch")

    def _entry(i):
        return {
            "id": i, "img_name": f"{i:06d}", "width": 8, "height": 8,
            "position": [0.0, 0.0, 0.0],
            "rotation": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "fx": 10.0, "fy": 10.0, "cx": 4.0, "cy": 4.0,
        }

    with open(model_path / "cameras.json", "w") as f:
        json.dump([_entry(i) for i in range(n_train)], f)
    with open(model_path / "test_cameras.json", "w") as f:
        json.dump([_entry(100 + i) for i in range(n_test)], f)
    return str(model_path), str(ply_path)


def _full_report_args(**overrides):
    ns = argparse.Namespace(
        trajectory="recorded", reload_gt=False, source_path_override=None, force=False,
    )
    for k, v in overrides.items():
        setattr(ns, k, v)
    return ns


def test_full_report_dispatch_calls_run_full_report_with_expected_cams(tmp_path, monkeypatch):
    model_path, ply_path = _make_full_report_run_dir(tmp_path, n_train=3, n_test=2)
    cfg_args = argparse.Namespace(sh_degree=3, model_layout="gsplat")

    monkeypatch.setattr(rr, "load_gaussians", lambda *a, **k: _FakeGaussians())
    calls = []

    def _fake_run_full_report(model_path, iteration, gaussians, train_cams, test_cams,
                               render_fn, pipe, background, subdir):
        calls.append({"n_train": len(train_cams), "n_test": len(test_cams), "subdir": subdir})
        return {"eval_valid": True, "n_test": len(test_cams), "mean_test_psnr": 30.0,
                "trajectory_fps": 24, "artifacts": []}

    monkeypatch.setattr(rr, "run_full_report", _fake_run_full_report)

    out_dir = os.path.join(model_path, "comparison", "iter_1000")
    args = _full_report_args()
    rr._process_full_report(args, model_path, cfg_args, 1000, ply_path, pipe=None,
                             background=None, out_dir=out_dir, device="cpu")

    assert len(calls) == 1
    assert calls[0]["n_train"] == 3
    assert calls[0]["n_test"] == 2
    manifest_path = os.path.join(out_dir, "regeneration_manifest.json")
    assert os.path.exists(manifest_path)
    with open(manifest_path) as f:
        manifest = json.load(f)
    assert manifest["camera_set"]["n_train"] == 3
    assert manifest["camera_set"]["n_test"] == 2


def _ply_hash(ply_path):
    with open(ply_path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def test_full_report_is_idempotent_and_never_touches_the_ply(tmp_path, monkeypatch):
    """Covers issue #401's acceptance bullet: 'report rendering does not
    change the saved reconstruction state', plus 'regeneration is
    restartable/idempotent'."""
    model_path, ply_path = _make_full_report_run_dir(tmp_path)
    cfg_args = argparse.Namespace(sh_degree=3, model_layout="gsplat")
    monkeypatch.setattr(rr, "load_gaussians", lambda *a, **k: _FakeGaussians())

    call_count = {"n": 0}

    def _fake_run_full_report(*a, **k):
        call_count["n"] += 1
        return {"eval_valid": True, "n_test": 2, "mean_test_psnr": 30.0,
                "trajectory_fps": 24, "artifacts": []}

    monkeypatch.setattr(rr, "run_full_report", _fake_run_full_report)

    out_dir = os.path.join(model_path, "comparison", "iter_1000")
    ply_hash_before = _ply_hash(ply_path)

    args = _full_report_args()
    rr._process_full_report(args, model_path, cfg_args, 1000, ply_path, pipe=None,
                             background=None, out_dir=out_dir, device="cpu")
    assert call_count["n"] == 1
    assert _ply_hash(ply_path) == ply_hash_before

    # Second call, no --force: must skip (idempotent/restartable).
    rr._process_full_report(args, model_path, cfg_args, 1000, ply_path, pipe=None,
                             background=None, out_dir=out_dir, device="cpu")
    assert call_count["n"] == 1
    assert _ply_hash(ply_path) == ply_hash_before

    # --force: must regenerate.
    args_forced = _full_report_args(force=True)
    rr._process_full_report(args_forced, model_path, cfg_args, 1000, ply_path, pipe=None,
                             background=None, out_dir=out_dir, device="cpu")
    assert call_count["n"] == 2
    assert _ply_hash(ply_path) == ply_hash_before


def test_full_report_warns_when_no_holdout_gt(tmp_path, monkeypatch, capsys):
    model_path, ply_path = _make_full_report_run_dir(tmp_path)
    cfg_args = argparse.Namespace(sh_degree=3, model_layout="gsplat")
    monkeypatch.setattr(rr, "load_gaussians", lambda *a, **k: _FakeGaussians())
    monkeypatch.setattr(rr, "run_full_report", lambda *a, **k: {
        "eval_valid": False, "n_test": 2, "mean_test_psnr": None,
        "trajectory_fps": None, "artifacts": [],
    })

    out_dir = os.path.join(model_path, "comparison", "iter_1000")
    args = _full_report_args()
    rr._process_full_report(args, model_path, cfg_args, 1000, ply_path, pipe=None,
                             background=None, out_dir=out_dir, device="cpu")

    assert "no holdout_gt/" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Non-destructive regeneration (defect found in #452's GPU verification pass)
#
# `write_post_training_report` builds report.json only from what it is HANDED,
# so a standalone regeneration drops seed_fade / pose_provenance /
# eval_metadata.holdout_hash -- all Run-Contract gates (CLAUDE.md). Since
# --full-report writes into the run's own comparison/iter_N/, that was silent
# permanent data destruction. These tests pin that it no longer is.
# ---------------------------------------------------------------------------

def _prior_report_fixture():
    """A realistic finished-run report.json, carrying every contract field a
    regeneration structurally cannot recompute."""
    return {
        "iteration": 1000,
        "n_test": 2,
        "mean_test_psnr": 27.5,
        "mean_train_psnr": 31.0,          # render-derived: must NOT be carried
        "n_train_metrics": 3,             # render-derived: must NOT be carried
        "resumed_from_iteration": 500,
        "seed_fade": {"detected": True, "severity": "severe", "confounded": True,
                      "indeterminate": False, "max_dip_db": 15.56},
        "seed_fade_confounded": True,
        "seed_fade_indeterminate": False,
        "pose_provenance": {"tier": "dataset_gt", "source": "groundtruth.txt"},
        "git_provenance": {"git_commit": "deadbeef", "git_branch": "main", "git_dirty": False},
        "ingestion": {"mode": "iter_based", "reason": "explicit flag"},
        "eval_metadata": {"holdout_hash": "HOLDOUT-RULE-HASH",
                          "realized_holdout_hash": "REALIZED-HASH",
                          "scored_holdout_hash": "SCORED-HASH"},
        "geometry": {"needle_frac_lt_0.3": 0.11,
                     "trajectory_coverage": {"verdict": "COMPLETE", "n_source_frames": 2525}},
    }


def _regenerated_report_fixture():
    """What write_post_training_report emits with no training process to hand
    it the run-property kwargs: metrics present, contract fields absent."""
    return {
        "iteration": 1000,
        "n_test": 2,
        "mean_test_psnr": 27.4,
        "resumed_from_iteration": None,
        "eval_metadata": {"realized_holdout_hash": "REALIZED-HASH",
                          "scored_holdout_hash": "SCORED-HASH"},
        "geometry": {"needle_frac_lt_0.3": 0.10},
        "artifacts": ["test_contact_sheet.png"],
    }


def test_merge_prior_report_preserves_contract_gating_fields():
    merged, prov = rr.merge_prior_report(_prior_report_fixture(), _regenerated_report_fixture())

    # The three fields the GPU verification pass found silently destroyed.
    assert merged["seed_fade"]["severity"] == "severe"
    assert merged["pose_provenance"]["tier"] == "dataset_gt"
    assert merged["eval_metadata"]["holdout_hash"] == "HOLDOUT-RULE-HASH"
    # ...plus the rest of the run-property set.
    assert merged["git_provenance"]["git_commit"] == "deadbeef"
    assert merged["ingestion"]["mode"] == "iter_based"
    # A carried field whose regenerated value is an uninformative None.
    assert merged["resumed_from_iteration"] == 500

    for f in ("seed_fade", "pose_provenance", "git_provenance", "ingestion",
              "resumed_from_iteration", "eval_metadata.holdout_hash"):
        assert f in prov["carried_over"], f


def test_merge_prior_report_regenerated_values_win_over_stale_ones():
    merged, _ = rr.merge_prior_report(_prior_report_fixture(), _regenerated_report_fixture())
    assert merged["mean_test_psnr"] == 27.4          # freshly computed
    assert merged["geometry"]["needle_frac_lt_0.3"] == 0.10
    # ...while a training-only geometry sub-block still survives.
    assert merged["geometry"]["trajectory_coverage"]["verdict"] == "COMPLETE"


def test_merge_prior_report_never_carries_stale_render_metrics_but_reports_them():
    """A regeneration that could not reattach train GT emits no train PSNR.
    Carrying the prior run's would silently misattribute it to this render --
    so it is dropped, but LOUDLY (listed for the caller to warn about and for
    the manifest to record)."""
    merged, prov = rr.merge_prior_report(_prior_report_fixture(), _regenerated_report_fixture())
    assert "mean_train_psnr" not in merged
    assert "n_train_metrics" not in merged
    assert prov["dropped_fields"] == ["mean_train_psnr", "n_train_metrics"]


def test_merge_prior_report_derives_seed_fade_mirrors_from_the_block():
    prior = _prior_report_fixture()
    prior["seed_fade"]["confounded"] = False
    prior["seed_fade"]["indeterminate"] = True
    prior["seed_fade_confounded"] = True      # deliberately inconsistent input
    merged, prov = rr.merge_prior_report(prior, _regenerated_report_fixture())
    # Mirrors are DERIVED from the carried block, never carried independently,
    # so the merged report cannot be internally inconsistent.
    assert merged["seed_fade_confounded"] is False
    assert merged["seed_fade_indeterminate"] is True
    assert "seed_fade_confounded" in prov["derived"]


def test_merge_prior_report_flags_recomputed_holdout_hash_mismatch():
    regenerated = _regenerated_report_fixture()
    regenerated["eval_metadata"]["scored_holdout_hash"] = "DIFFERENT-SCORED-HASH"
    merged, prov = rr.merge_prior_report(_prior_report_fixture(), regenerated)
    # Recomputed value still wins -- it describes what was actually scored --
    # but the disagreement is surfaced, not swallowed.
    assert merged["eval_metadata"]["scored_holdout_hash"] == "DIFFERENT-SCORED-HASH"
    assert prov["recomputed_hash_mismatch"]["scored_holdout_hash"] == {
        "prior": "SCORED-HASH", "regenerated": "DIFFERENT-SCORED-HASH"}


def test_merge_prior_report_with_no_prior_is_a_passthrough():
    new = _regenerated_report_fixture()
    merged, prov = rr.merge_prior_report({}, new)
    assert merged == new
    assert prov["carried_over"] == [] and prov["dropped_fields"] == []


def test_merge_prior_report_stamps_the_regeneration_marker_when_given():
    """A merged report is otherwise indistinguishable from a training-run
    report -- it carries the original holdout_hash/seed_fade/pose_provenance
    beside metrics computed later, from a PLY-loaded model. Every consumer in
    this repo reads report.json alone, so the caveat has to live IN it."""
    marker = {"tool": "render_report.py --full-report", "iteration": 1000}
    merged, _ = rr.merge_prior_report(_prior_report_fixture(),
                                      _regenerated_report_fixture(),
                                      regeneration_marker=marker)
    assert merged["regenerated"]["iteration"] == 1000
    # ...and a marker from a PREVIOUS regeneration is superseded, not reported
    # as a dropped field.
    prior = dict(_prior_report_fixture(), regenerated={"tool": "an older regeneration"})
    merged2, prov2 = rr.merge_prior_report(prior, _regenerated_report_fixture(),
                                           regeneration_marker=marker)
    assert merged2["regenerated"] == marker
    assert "regenerated" not in prov2["dropped_fields"]


def _write_prior_report(out_dir, payload=None):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "report.json")
    with open(path, "w") as f:
        json.dump(payload if payload is not None else _prior_report_fixture(), f, indent=2)
    return path


def _stub_writing_regenerated_report(payload=None):
    """A run_full_report stub that behaves like the real one: it WRITES
    report.json into the directory model_path/comparison/<subdir> resolves to,
    which is also how these tests assert that an out-of-tree --out_dir really
    lands where it claims."""
    def _stub(model_path, iteration, gaussians, train_cams, test_cams,
              render_fn, pipe, background, subdir):
        resolved = os.path.normpath(os.path.join(model_path, "comparison", subdir))
        os.makedirs(resolved, exist_ok=True)
        body = dict(payload if payload is not None else _regenerated_report_fixture())
        with open(os.path.join(resolved, "report.json"), "w") as f:
            json.dump(body, f, indent=2)
        return dict(body, trajectory_fps=24, artifacts=["test_contact_sheet.png"])
    return _stub


def test_full_report_in_place_preserves_prior_contract_fields(tmp_path, monkeypatch):
    """The blocking defect, end to end: regenerating INTO a finished run's own
    comparison/iter_N/ must not destroy seed_fade / pose_provenance /
    eval_metadata.holdout_hash."""
    model_path, ply_path = _make_full_report_run_dir(tmp_path)
    cfg_args = argparse.Namespace(sh_degree=3, model_layout="gsplat")
    monkeypatch.setattr(rr, "load_gaussians", lambda *a, **k: _FakeGaussians())
    monkeypatch.setattr(rr, "run_full_report", _stub_writing_regenerated_report())

    out_dir = os.path.join(model_path, "comparison", "iter_1000")
    _write_prior_report(out_dir)

    rr._process_full_report(_full_report_args(), model_path, cfg_args, 1000, ply_path,
                             pipe=None, background=None, out_dir=out_dir, device="cpu")

    with open(os.path.join(out_dir, "report.json")) as f:
        merged = json.load(f)
    assert merged["seed_fade"]["severity"] == "severe"
    assert merged["seed_fade_confounded"] is True
    assert merged["pose_provenance"]["tier"] == "dataset_gt"
    assert merged["eval_metadata"]["holdout_hash"] == "HOLDOUT-RULE-HASH"
    assert merged["mean_test_psnr"] == 27.4  # regenerated value still wins

    # The verbatim original is preserved on disk too.
    with open(os.path.join(out_dir, "report.pre_regeneration.json")) as f:
        assert json.load(f) == _prior_report_fixture()

    with open(os.path.join(out_dir, "regeneration_manifest.json")) as f:
        manifest = json.load(f)
    rm = manifest["report_merge"]
    assert rm["merged"] is True
    assert rm["prior_report_path"] == os.path.join(out_dir, "report.json")
    assert "seed_fade" in rm["carried_over"]
    assert "eval_metadata.holdout_hash" in rm["carried_over"]
    assert rm["dropped_fields"] == ["mean_train_psnr", "n_train_metrics"]

    # The merged report must SAY it is a regeneration -- otherwise it is
    # indistinguishable from the training run's own report.
    marker = merged["regenerated"]
    assert marker["iteration"] == 1000
    assert marker["from_checkpoint"] == ply_path
    assert marker["prior_report_backup"].endswith("report.pre_regeneration.json")
    assert "seed_fade" in marker["carried_over"]
    assert "code_revision" in marker
    assert "NOT produced by the training run" in marker["note"]


def test_full_report_backs_up_prior_report_before_rendering(tmp_path, monkeypatch):
    """The backup must exist even if regeneration dies mid-render -- otherwise
    'non-destructive' only holds on the happy path."""
    model_path, ply_path = _make_full_report_run_dir(tmp_path)
    cfg_args = argparse.Namespace(sh_degree=3, model_layout="gsplat")
    monkeypatch.setattr(rr, "load_gaussians", lambda *a, **k: _FakeGaussians())

    def _boom(*a, **k):
        raise RuntimeError("GPU exploded mid-render")

    monkeypatch.setattr(rr, "run_full_report", _boom)
    out_dir = os.path.join(model_path, "comparison", "iter_1000")
    _write_prior_report(out_dir)

    with pytest.raises(RuntimeError):
        rr._process_full_report(_full_report_args(), model_path, cfg_args, 1000, ply_path,
                                 pipe=None, background=None, out_dir=out_dir, device="cpu")

    with open(os.path.join(out_dir, "report.pre_regeneration.json")) as f:
        assert json.load(f) == _prior_report_fixture()


def test_full_report_backup_is_never_clobbered_by_a_second_regeneration(tmp_path, monkeypatch):
    model_path, ply_path = _make_full_report_run_dir(tmp_path)
    cfg_args = argparse.Namespace(sh_degree=3, model_layout="gsplat")
    monkeypatch.setattr(rr, "load_gaussians", lambda *a, **k: _FakeGaussians())
    monkeypatch.setattr(rr, "run_full_report", _stub_writing_regenerated_report())
    out_dir = os.path.join(model_path, "comparison", "iter_1000")
    _write_prior_report(out_dir)

    for _ in range(2):
        rr._process_full_report(_full_report_args(force=True), model_path, cfg_args, 1000,
                                 ply_path, pipe=None, background=None, out_dir=out_dir,
                                 device="cpu")

    # Still the TRUE original, not the first regeneration's output.
    with open(os.path.join(out_dir, "report.pre_regeneration.json")) as f:
        assert json.load(f) == _prior_report_fixture()


def test_full_report_out_dir_leaves_the_source_run_byte_identical(tmp_path, monkeypatch):
    """--out_dir: the run directory must be read-only from the tool's point of
    view -- byte-identical before and after, not merely 'the key is still
    there'."""
    model_path, ply_path = _make_full_report_run_dir(tmp_path)
    cfg_args = argparse.Namespace(sh_degree=3, model_layout="gsplat")
    monkeypatch.setattr(rr, "load_gaussians", lambda *a, **k: _FakeGaussians())
    monkeypatch.setattr(rr, "run_full_report", _stub_writing_regenerated_report())

    canonical = os.path.join(model_path, "comparison", "iter_1000")
    canonical_report = _write_prior_report(canonical)
    with open(canonical_report, "rb") as f:
        before = hashlib.sha256(f.read()).hexdigest()
    listing_before = sorted(os.listdir(canonical))
    ply_before = _ply_hash(ply_path)

    out_dir = str(tmp_path / "scratch_regen")
    rr._process_full_report(_full_report_args(), model_path, cfg_args, 1000, ply_path,
                             pipe=None, background=None, out_dir=out_dir, device="cpu")

    with open(canonical_report, "rb") as f:
        assert hashlib.sha256(f.read()).hexdigest() == before
    assert sorted(os.listdir(canonical)) == listing_before
    assert _ply_hash(ply_path) == ply_before

    # The regeneration landed in --out_dir, carrying the contract fields
    # forward from the run's canonical report even though it wrote elsewhere.
    with open(os.path.join(out_dir, "report.json")) as f:
        merged = json.load(f)
    assert merged["eval_metadata"]["holdout_hash"] == "HOLDOUT-RULE-HASH"
    assert merged["seed_fade"]["severity"] == "severe"
    assert os.path.exists(os.path.join(out_dir, "regeneration_manifest.json"))
    # The scratch copy is marked as a regeneration; the run's own untouched
    # report is NOT (it really was produced by the training run).
    assert merged["regenerated"]["from_checkpoint"] == ply_path
    with open(canonical_report) as f:
        assert "regenerated" not in json.load(f)


def test_find_prior_report_falls_back_to_the_benchmark_dir_model_location(tmp_path):
    model_path = str(tmp_path / "run")
    os.makedirs(model_path)
    root_report = os.path.join(model_path, "report.json")
    with open(root_report, "w") as f:
        json.dump({"iteration": 7}, f)
    path, data, reason = rr.find_prior_report(model_path, os.path.join(model_path, "nowhere"), 7)
    assert path == root_report and data["iteration"] == 7 and reason is None


def test_find_prior_report_reports_an_explicit_reason_when_absent(tmp_path):
    model_path = str(tmp_path / "run")
    os.makedirs(model_path)
    path, data, reason = rr.find_prior_report(model_path, model_path, 7)
    assert path is None and data is None
    assert "no prior report.json found" in reason


def test_full_report_warns_loudly_when_no_prior_report_exists(tmp_path, monkeypatch, capsys):
    model_path, ply_path = _make_full_report_run_dir(tmp_path)
    cfg_args = argparse.Namespace(sh_degree=3, model_layout="gsplat")
    monkeypatch.setattr(rr, "load_gaussians", lambda *a, **k: _FakeGaussians())
    monkeypatch.setattr(rr, "run_full_report", _stub_writing_regenerated_report())

    out_dir = os.path.join(model_path, "comparison", "iter_1000")
    rr._process_full_report(_full_report_args(), model_path, cfg_args, 1000, ply_path,
                             pipe=None, background=None, out_dir=out_dir, device="cpu")
    out = capsys.readouterr().out
    assert "no prior report.json found" in out
    with open(os.path.join(out_dir, "regeneration_manifest.json")) as f:
        assert json.load(f)["report_merge"]["prior_report_path"] is None


# ---------------------------------------------------------------------------
# code_revision.regeneration without a `git` binary (defect 2)
# ---------------------------------------------------------------------------

def test_git_metadata_from_dotgit_resolves_branch_and_commit(tmp_path):
    """The training image ships no `git`, so the manifest recorded a bare
    "unknown" on the PRIMARY containerized path. Parse .git/ directly."""
    git_dir = tmp_path / "repo" / ".git"
    (git_dir / "refs" / "heads" / "feature").mkdir(parents=True)
    (git_dir / "HEAD").write_text("ref: refs/heads/feature/x\n")
    (git_dir / "refs" / "heads" / "feature" / "x").write_text("a" * 40 + "\n")

    meta = rr._git_metadata_from_dotgit(str(tmp_path / "repo"))
    assert meta["git_commit"] == "a" * 40
    assert meta["git_branch"] == "feature/x"
    assert meta["source"] == "dotgit"
    # git_dirty is genuinely not derivable this way -- but it must SAY so.
    assert meta["git_dirty"] == "unknown"
    assert "not determinable" in meta["git_dirty_reason"]


def test_git_metadata_from_dotgit_uses_packed_refs(tmp_path):
    git_dir = tmp_path / "repo" / ".git"
    git_dir.mkdir(parents=True)
    (git_dir / "HEAD").write_text("ref: refs/heads/main\n")
    (git_dir / "packed-refs").write_text(
        "# pack-refs with: peeled fully-peeled sorted\n" + "b" * 40 + " refs/heads/main\n")
    meta = rr._git_metadata_from_dotgit(str(tmp_path / "repo"))
    assert meta["git_commit"] == "b" * 40 and meta["git_branch"] == "main"


def test_git_metadata_from_dotgit_follows_a_worktree_gitdir_pointer(tmp_path):
    """Agents in this repo run from `.claude/worktrees/agent-*`, where `.git`
    is a FILE pointing at the real gitdir."""
    real = tmp_path / "realgit"
    real.mkdir()
    (real / "HEAD").write_text("c" * 40 + "\n")  # detached
    wt = tmp_path / "wt"
    wt.mkdir()
    (wt / ".git").write_text(f"gitdir: {real}\n")
    meta = rr._git_metadata_from_dotgit(str(wt))
    assert meta["git_commit"] == "c" * 40
    assert meta["git_branch"] == "HEAD (detached)"


def test_git_metadata_from_dotgit_explains_itself_when_there_is_no_repo(tmp_path):
    meta = rr._git_metadata_from_dotgit(str(tmp_path))
    assert meta["source"] == "unavailable"
    assert meta["git_commit"] == "unknown"
    assert meta["reason"] and "no .git found" in meta["reason"]
