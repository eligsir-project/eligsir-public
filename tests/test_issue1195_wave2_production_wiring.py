"""Production reachability gates for #1193/#1198/#1199 integration."""

from dataclasses import replace
import inspect
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest

from eligsir.contracts.runtime import SourceIdentity
from eligsir.mapping.observation_db import FrameObservationDB

from eligsir.runtime.production_application import (
    CanonicalMappingRuntime,
    _resolve_parameters,
)
from eligsir.runtime.config_boundary import mapping_runtime_config_from_args


def test_canonical_mapping_profile_selects_device_native_insertion(tmp_path):
    args, _dataset, _opt, _pipe = _resolve_parameters(
        "live/canonical_mapping", str(tmp_path / "model")
    )

    assert args.gpu_frame_prep is True
    assert args.gpu_frame_resident is True
    assert args.streaming_surface_cell_dedup is False
    assert args.streaming_insert_confidence_opacity is False


def test_production_overrides_win_before_canonical_policy_freeze(tmp_path):
    args, _dataset, _opt, _pipe = _resolve_parameters(
        "live/canonical_mapping",
        str(tmp_path / "model"),
        overrides={
            "streaming_lifecycle_enabled": True,
            "streaming_mature_age_frames": 1,
            "streaming_mature_min_utility": 0.0,
        },
    )

    config = mapping_runtime_config_from_args(args)

    assert config.optimization.streaming_lifecycle_enabled is True
    assert config.optimization.streaming_mature_age_frames == 1
    assert config.optimization.streaming_mature_min_utility == 0.0


def test_transfer_budget_override_crosses_canonical_policy_boundary(tmp_path):
    args, _dataset, _opt, _pipe = _resolve_parameters(
        "live/canonical_mapping",
        str(tmp_path / "model"),
        overrides={"streaming_max_transfer_bytes": 1_073_741_824},
    )

    config = mapping_runtime_config_from_args(args)

    assert config.runtime.max_transfer_bytes == 1_073_741_824


def test_prepared_causal_profile_enables_required_admission_prefetch(tmp_path):
    args, _dataset, _opt, _pipe = _resolve_parameters(
        "live/canonical_mapping",
        str(tmp_path / "model"),
        overrides={"causal_prepared_ready_b4": True},
    )

    assert args.streaming_admission_prefetch is False
    assert mapping_runtime_config_from_args(args).runtime.admission_prefetch_enabled is True


def test_legacy_causal_profile_keeps_admission_prefetch_disabled_by_default(tmp_path):
    args, _dataset, _opt, _pipe = _resolve_parameters(
        "live/canonical_mapping", str(tmp_path / "model")
    )

    assert mapping_runtime_config_from_args(args).runtime.admission_prefetch_enabled is False


def test_mapping_resolution_tier_override_crosses_canonical_policy_boundary(tmp_path):
    args, _dataset, _opt, _pipe = _resolve_parameters(
        "live/canonical_mapping",
        str(tmp_path / "model"),
        overrides={"streaming_mapping_resolution_tier": "native"},
    )

    config = mapping_runtime_config_from_args(args)

    assert config.static_options["mapping_resolution_tier"] == "native"


def test_mapping_resolution_min_tier_limits_adaptive_runtime_ladder(tmp_path):
    args, _dataset, _opt, _pipe = _resolve_parameters(
        "live/canonical_mapping",
        str(tmp_path / "model"),
        overrides={"streaming_mapping_resolution_min_tier": "quarter"},
    )

    config = mapping_runtime_config_from_args(args)

    assert config.static_options["mapping_resolution_min_tier"] == "quarter"


def test_mapping_resolution_tier_rejects_unknown_policy(tmp_path):
    args, _dataset, _opt, _pipe = _resolve_parameters(
        "live/canonical_mapping",
        str(tmp_path / "model"),
        overrides={"streaming_mapping_resolution_tier": "fullish"},
    )

    with pytest.raises(ValueError, match="streaming_mapping_resolution_tier"):
        mapping_runtime_config_from_args(args)


def test_canonical_runtime_uses_typed_pixel_exposure_cadence_defaults(tmp_path):
    args, _dataset, _opt, _pipe = _resolve_parameters(
        "live/canonical_mapping",
        str(tmp_path / "model"),
        overrides={
            "online_mode": True,
            "mcmc_stop_growth_iter": 12_000,
            "streaming_growth_stop_horizon": 0,
        },
    )

    config = mapping_runtime_config_from_args(args)

    assert config.runtime.mcmc_grow_interval_pixels == 30_720_000
    assert config.runtime.mcmc_post_ingestion_grow_interval_pixels == 0
    assert config.runtime.mcmc_relocate_interval_pixels == 15_360_000
    assert config.runtime.mcmc_post_ingestion_relocate_interval_pixels == 0
    assert config.runtime.mcmc_min_optimizer_updates_after_structural_mutation == 2
    assert config.runtime.mcmc_noise_stop_optimizer_updates == 30_000
    assert config.runtime.mcmc_temperature_horizon_optimizer_updates == 30_000
    assert config.runtime.sh_degree_schedule_rgb_pixels == (
        307_200_000,
        614_400_000,
        921_600_000,
    )


def test_legacy_sh_iteration_schedule_cannot_actuate_canonical_runtime(tmp_path):
    args, _dataset, _opt, _pipe = _resolve_parameters(
        "live/canonical_mapping",
        str(tmp_path / "model"),
    )
    args.sh_degree_schedule = [1, 2, 3]

    config = mapping_runtime_config_from_args(args)

    assert config.runtime.sh_degree_schedule_rgb_pixels == (
        307_200_000,
        614_400_000,
        921_600_000,
    )


def test_canonical_runtime_explicitly_selects_semantic_strategy_mode():
    source = inspect.getsource(CanonicalMappingRuntime._initialize)

    assert "semantic_cadence=True" in source


def test_canonical_runtime_honors_explicit_pixel_exposure_cadence(tmp_path):
    args, _dataset, _opt, _pipe = _resolve_parameters(
        "live/canonical_mapping",
        str(tmp_path / "model"),
        overrides={
            "online_mode": True,
            "mcmc_grow_interval_pixels": 61_440_000,
            "streaming_mcmc_post_ingestion_grow_interval_pixels": 122_880_000,
            "mcmc_relocate_interval_pixels": 30_720_000,
            "streaming_mcmc_post_ingestion_relocate_interval_pixels": 61_440_000,
            "mcmc_min_optimizer_updates_after_structural_mutation": 4,
            "mcmc_noise_stop_optimizer_updates": 40_000,
            "mcmc_temperature_horizon_optimizer_updates": 50_000,
        },
    )

    config = mapping_runtime_config_from_args(args)

    assert config.runtime.mcmc_grow_interval_pixels == 61_440_000
    assert config.runtime.mcmc_post_ingestion_grow_interval_pixels == 122_880_000
    assert config.runtime.mcmc_relocate_interval_pixels == 30_720_000
    assert config.runtime.mcmc_post_ingestion_relocate_interval_pixels == 61_440_000
    assert config.runtime.mcmc_min_optimizer_updates_after_structural_mutation == 4
    assert config.runtime.mcmc_noise_stop_optimizer_updates == 40_000
    assert config.runtime.mcmc_temperature_horizon_optimizer_updates == 50_000


def test_canonical_mcmc_clocks_require_exact_integer_types(tmp_path):
    args, _dataset, _opt, _pipe = _resolve_parameters(
        "live/canonical_mapping", str(tmp_path / "model")
    )
    runtime = mapping_runtime_config_from_args(args).runtime

    with pytest.raises(TypeError, match="mcmc_grow_interval_pixels"):
        replace(runtime, mcmc_grow_interval_pixels=30_720_000.5)


def test_legacy_iteration_cadence_is_inert_at_canonical_strategy_boundary(tmp_path):
    args, _dataset, _opt, _pipe = _resolve_parameters(
        "live/canonical_mapping",
        str(tmp_path / "model"),
        overrides={
            "online_mode": False,
            "mcmc_stop_growth_iter": 12_000,
            "mcmc_noise_stop_iter": 1,
            "streaming_growth_stop_horizon": 0,
        },
    )

    config = mapping_runtime_config_from_args(args)

    legacy_iteration_cadence = {
        "densify_from_iter",
        "densify_until_iter",
        "densification_interval",
        "mcmc_stop_growth_iter",
        "mcmc_noise_stop_iter",
        "mcmc_grow_interval_min",
        "mcmc_grow_interval_max",
        "mcmc_relocate_interval_min",
        "mcmc_relocate_interval_max",
        "mcmc_grow_tau",
        "mcmc_relocate_tau",
        "mcmc_growth_factor_tau",
        "mcmc_dead_opacity_power",
        "mcmc_cap_interval_strength",
        "mcmc_cap_interval_power",
        "mcmc_target_q_start",
        "mcmc_target_q_end",
        "mcmc_target_tau",
        "streaming_growth_stop_horizon",
        "streaming_noise_stop_horizon",
    }
    assert legacy_iteration_cadence.isdisjoint(config.optimization.values)
    assert "mcmc_semantic_due_flags" not in config.optimization.values
    assert config.optimization.mcmc_noise_stop_optimizer_updates == 30_000


def test_spatial_focus_and_replay_tail_voxel_cross_the_config_boundary(tmp_path):
    args, _dataset, _opt, _pipe = _resolve_parameters(
        "live/canonical_mapping",
        str(tmp_path / "model"),
        overrides={
            "streaming_replay_depth_voxel_size": 0.05,
            "streaming_partition_replay_focus": True,
            "streaming_spatial_replay_focus": True,
            "streaming_spatial_replay_focus_cell_size": 0.5,
            "streaming_spatial_replay_focus_interval": 25,
            "streaming_adaptive_capacity": True,
            "streaming_adcap_hysteresis": 2,
        },
    )

    config = mapping_runtime_config_from_args(args)

    assert config.static_options["replay_depth_voxel_size_m"] == 0.05
    assert config.static_options["partition_replay_focus"] is True
    assert config.static_options["spatial_replay_focus"] is True
    assert config.static_options["spatial_replay_focus_cell_size_m"] == 0.5
    assert config.static_options["spatial_replay_focus_interval"] == 25
    assert config.static_options["adaptive_growth"] is True
    assert config.static_options["adaptive_growth_hysteresis"] == 2
    assert config.static_options["adaptive_growth_u_brake"] == -1.0e-5
    assert config.static_options["adaptive_growth_u_release"] == 1.0e-5


def test_canonical_adaptive_growth_rejects_holdout_only_authority(tmp_path):
    args, _dataset, _opt, _pipe = _resolve_parameters(
        "live/canonical_mapping",
        str(tmp_path / "model"),
        overrides={
            "streaming_adaptive_capacity": True,
            "streaming_spatial_replay_focus": False,
        },
    )

    with pytest.raises(
        ValueError, match="requires training-only spatial replay focus"
    ):
        mapping_runtime_config_from_args(args)


def test_canonical_live_profile_uses_verified_async_c4_sparse_first_policy(tmp_path):
    args, _dataset, _opt, _pipe = _resolve_parameters(
        "live/canonical_mapping", str(tmp_path / "model")
    )

    assert args.streaming_optimization_batch_size == 4
    assert args.streaming_depth_stride == 12
    assert args.streaming_seed_voxel_size == 0.12
    assert args.streaming_insert_voxel_size == 0.12
    assert args.streaming_max_new_gaussians_per_frame == 16
    assert args.streaming_replay_target_views_per_s == 100.0
    assert args.streaming_replay_depth_max_per_event == 64
    assert args.streaming_replay_depth_interval == 50
    assert args.streaming_replay_depth_voxel_size == 0.03
    assert args.streaming_spatial_replay_focus is True
    assert args.streaming_spatial_replay_focus_cell_size == 1.5
    assert args.streaming_spatial_replay_focus_interval == 25
    assert args.streaming_mcmc_post_ingestion_growth_factor_start == 1.5
    assert args.streaming_lifecycle_enabled is True
    assert args.streaming_lifecycle_interval_updates == 8
    assert args.streaming_mature_age_frames == 15
    assert args.streaming_mature_min_utility == 0.0


def test_production_record_callback_uses_the_shared_observation_database():
    runtime = object.__new__(CanonicalMappingRuntime)
    runtime._observation_db = FrameObservationDB(
        device="cpu", max_active_supervision_frames=2
    )
    source = SourceIdentity("camera", 0, 4, 40_000_000)
    mapping_frame = SimpleNamespace(
        source=source,
        device="cpu",
        rgb=SimpleNamespace(data=np.zeros((2, 3, 3), dtype=np.uint8)),
        depth=SimpleNamespace(data=np.ones((2, 3), dtype=np.float32)),
        intrinsics="calibration-v1",
    )
    frame = SimpleNamespace(
        index=4,
        timestamp=0.04,
        mapping_frame=mapping_frame,
    )
    camera = SimpleNamespace(c2w=np.eye(4, dtype=np.float32))

    runtime._record_scene_candidate(camera, frame, is_train=True, is_holdout=False)

    assert runtime._observation_db.get_metadata(4).timestamp == 0.04
    assert runtime._observation_db.get_cold_pixels(4).source_identity == source.fingerprint()


def test_production_runtime_has_no_post_commit_anchor_writer():
    source = (
        Path(__file__).resolve().parents[1]
        / "eligsir/runtime/production_application.py"
    ).read_text(encoding="utf-8")

    assert "self.gaussians.anchor_iter[" not in source


def test_production_optimization_reuses_the_canonical_transfer_owner():
    source = (
        Path(__file__).resolve().parents[1]
        / "eligsir/runtime/production_application.py"
    ).read_text(encoding="utf-8")

    construction = source.split("self._optimization_runtime = StreamingOptimizationRuntime(", 1)[1]
    construction = construction.split("self._mcmc_cfg =", 1)[0]
    assert "camera_transfer_owner=self._transfer_owner" in construction
