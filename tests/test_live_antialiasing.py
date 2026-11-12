from pathlib import Path
from types import SimpleNamespace
import inspect

import numpy as np
import pytest
import torch

from eligsir.cli.config import load_cli_config
from eligsir.cli.parser import parse_training_args
from eligsir.runtime.config_boundary import mapping_runtime_config_from_args
from eligsir.mapping.renderer.gsplat_backend import validate_rasterize_mode
from scripts.issue106_camera_batch_probe import build_parser
from eligsir.utils.live_antialiasing_metrics import (
    camera_pair_temporal_flicker,
    edge_psnr,
    motion_compensated_temporal_flicker,
    reprojection_grid_a_for_b,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("mode", ("classic", "antialiased"))
def test_live_rasterizer_profiles_are_pure_ab_overlays(mode):
    profile = REPO_ROOT / "configs/profiles/rendering" / f"live_{mode}.yml"
    raw = profile.read_text(encoding="utf-8")
    assert "extends:" not in raw
    assert load_cli_config(config_files=[str(profile)])["rasterize_mode"] == mode
    canonical = load_cli_config(run_profile="live/canonical_mapping")
    composed = load_cli_config(
        run_profile="live/canonical_mapping", config_files=[str(profile)]
    )
    for key, value in canonical.items():
        if key != "rasterize_mode":
            assert composed[key] == value, key
    assert composed["rasterize_mode"] == mode


@pytest.mark.parametrize("mode", ("classic", "antialiased"))
def test_8060s_live_aa_experiment_profiles_are_directly_executable(mode):
    profile = (
        REPO_ROOT
        / "configs/profiles/experiments/live_aa/8060s"
        / f"{mode}.yml"
    )

    resolved = load_cli_config(config_files=[str(profile)])

    assert resolved["execution_trainer"] == "causal_mapping"
    assert resolved["streaming_ingestion_mode"] == "dataset_fps"
    assert resolved["streaming_input_fps_cap"] == 30.0
    assert resolved["streaming_eval_hold"] == 8
    assert resolved["streaming_report_final"] is True
    assert resolved["streaming_resolution"] == 1
    assert resolved["streaming_optimization_batch_size"] == 4
    assert resolved["streaming_optimization_camera_count_max"] == 4
    assert resolved["streaming_optimization_constellation_policy"] == "fixed"
    assert resolved["streaming_optimization_transaction_budget_ms"] == 25
    assert resolved["streaming_report_test_metrics_max"] >= 1_000_000
    assert resolved["streaming_keyframe_max_gap_require_novelty"] is False
    assert resolved["rasterize_mode"] == mode


@pytest.mark.parametrize("invalid", ("mip", "ANTIALIASED", "", 1, True, None))
def test_canonical_runtime_rejects_unknown_or_untyped_rasterize_mode(invalid):
    args, _ = parse_training_args(["-s", "/dev/null"])
    args.model_path = "output/live-aa-validation"
    args.rasterize_mode = invalid
    with pytest.raises((TypeError, ValueError), match="rasterize_mode"):
        mapping_runtime_config_from_args(args)


def test_canonical_runtime_freezes_typed_antialiased_mode():
    args, _ = parse_training_args(["-s", "/dev/null"])
    args.model_path = "output/live-aa-antialiased"
    args.rasterize_mode = "antialiased"
    config = mapping_runtime_config_from_args(args)
    assert config.pipeline.rasterize_mode == "antialiased"


def test_direct_renderer_boundary_and_probe_share_the_two_mode_vocabulary():
    assert validate_rasterize_mode("classic") == "classic"
    assert validate_rasterize_mode("antialiased") == "antialiased"
    with pytest.raises(ValueError, match="rasterize_mode"):
        validate_rasterize_mode("mip")
    assert (
        build_parser().parse_args(["--rasterize-mode", "antialiased"]).rasterize_mode
        == "antialiased"
    )


def test_every_gsplat_backend_rasterization_uses_the_shared_mode_validator():
    from eligsir.mapping.renderer import gsplat_backend

    source = inspect.getsource(gsplat_backend)
    assert 'rasterize_mode=getattr(pipe, "rasterize_mode", "classic")' not in source


def test_production_holdout_cameras_expose_reloadable_depth_without_training_marker():
    from eligsir.runtime.causal_evaluation import (
        CausalHoldoutPlan,
        build_causal_holdout_cameras,
    )

    records = []
    for index in (4, 8):
        depth = np.full((4, 4), 1000, dtype=np.uint16)
        if index == 4:
            # Source A sees an occluder at this pixel; B's one-metre point
            # must be excluded by the production depth-consistency mask.
            depth[0, 0] = 500
        records.append(
            SimpleNamespace(
                index=index,
                source_ordinal=index,
                timestamp=float(index),
                c2w=np.eye(4, dtype=np.float32),
                fx=2.0,
                fy=2.0,
                cx=1.5,
                cy=1.5,
                width=4,
                height=4,
                depth_scale=1000.0,
                depth_path=None,
                rgb_path=None,
                _rgb_np=np.full((4, 4, 3), index, dtype=np.uint8),
                _rgb_bytes=None,
                _depth_np=depth,
                _depth_bytes=None,
            )
        )
    plan = CausalHoldoutPlan(
        holdout_ordinals=frozenset((4, 8)),
        guard_ordinals=frozenset(),
        holdout_records=tuple(records),
        eval_metadata={},
    )

    cameras = build_causal_holdout_cameras(plan, data_device="cpu")

    assert len(cameras) == 2
    assert all(not hasattr(camera, "_streaming_frame") for camera in cameras)
    assert all(callable(camera._sensor_depth_loader) for camera in cameras)
    gt_a = torch.zeros(3, 4, 4)
    gt_b = torch.ones(3, 4, 4) * 0.2
    result = camera_pair_temporal_flicker(
        cameras[0], gt_a + 0.1, gt_a,
        cameras[1], gt_b + 0.1, gt_b,
    )
    assert result.valid_pixels == 15
    assert result.rms < 1e-6


def test_edge_psnr_is_identical_at_each_evaluation_resolution():
    gt = torch.zeros(3, 8, 8)
    gt[:, :, 4:] = 1.0
    native = edge_psnr(gt, gt)
    downsampled = torch.nn.functional.interpolate(
        gt[None], scale_factor=0.5, mode="area"
    )[0]
    scaled = edge_psnr(downsampled, downsampled)
    assert torch.isinf(native)
    assert torch.isinf(scaled)


def test_edge_psnr_penalizes_shifted_edges():
    gt = torch.zeros(3, 8, 8)
    gt[:, :, 4:] = 1.0
    shifted = torch.zeros_like(gt)
    shifted[:, :, 5:] = 1.0
    assert torch.isfinite(edge_psnr(shifted, gt))


def test_motion_compensated_flicker_measures_residual_change_not_camera_motion():
    gt_a = torch.rand(3, 4, 5)
    gt_b = torch.roll(gt_a, shifts=1, dims=-1)
    render_a = gt_a + 0.1
    render_b = gt_b + 0.1
    # B pixels sample A one column to the left, matching the synthetic motion.
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, 4),
        torch.linspace(-1.0, 1.0, 5),
        indexing="ij",
    )
    step = 2.0 / 4.0
    grid = torch.stack((xx - step, yy), dim=-1)
    valid = torch.ones(1, 4, 5, dtype=torch.bool)
    valid[..., 0] = False
    stable = motion_compensated_temporal_flicker(
        render_a, gt_a, render_b, gt_b, grid, valid
    )
    changed = motion_compensated_temporal_flicker(
        render_a, gt_a, render_b + 0.2, gt_b, grid, valid
    )
    assert stable.rms < 1e-6
    assert changed.rms > 0.19
    assert stable.valid_pixels == 16


def test_identity_pose_depth_builds_identity_motion_compensation_grid():
    depth = torch.ones(1, 3, 4)
    intrinsics = torch.eye(3)
    transform = torch.eye(4)
    grid, valid, projected_depth = reprojection_grid_a_for_b(
        depth,
        intrinsics,
        transform,
        intrinsics,
        transform,
        source_height=3,
        source_width=4,
    )
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, 3),
        torch.linspace(-1.0, 1.0, 4),
        indexing="ij",
    )
    assert torch.allclose(grid, torch.stack((xx, yy), dim=-1))
    assert bool(valid.all())
    assert torch.equal(projected_depth, torch.ones(3, 4))


def test_temporal_flicker_fails_closed_without_overlap():
    image = torch.zeros(3, 2, 2)
    grid = torch.zeros(2, 2, 2)
    with pytest.raises(ValueError, match="valid overlap"):
        motion_compensated_temporal_flicker(
            image, image, image, image, grid, torch.zeros(1, 2, 2, dtype=torch.bool)
        )
