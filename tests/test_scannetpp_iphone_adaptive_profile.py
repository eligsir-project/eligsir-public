"""Contract for the managed ScanNet++ iPhone-LiDAR mapping overlay."""

from __future__ import annotations

from pathlib import Path


_ROOT = Path(__file__).resolve().parents[1]
_PROFILE = _ROOT / "configs/profiles/evaluation/scannetpp_iphone_adaptive.yml"


def test_scannetpp_overlay_keeps_canonical_ready_policy_and_uses_adaptive_tiers(tmp_path):
    """Only dataset/evaluation deltas may differ from canonical live mapping."""

    from eligsir.cli.config import load_cli_config
    from eligsir.cli.parser import parse_training_args
    from eligsir.cli.run_config import validate_config
    from eligsir.runtime.config_boundary import mapping_runtime_config_from_args

    baseline = load_cli_config(run_profile="live/canonical_mapping")
    composed = load_cli_config(
        run_profile="live/canonical_mapping", config_files=[str(_PROFILE)]
    )

    expected_deltas = {
        "online_mode": False,
        "viewer": "none",
        "streaming_camera_profile": "iphone_lidar",
        "streaming_ingestion_mode": "dataset_fps",
        "streaming_input_fps_cap": 30.0,
        "streaming_frame_stride": 1,
        "streaming_resolution": 1,
        "streaming_min_depth": 0.2,
        "streaming_max_depth": 5.0,
        "streaming_depth_consistency_thresh": 0.03,
        "streaming_depth_edge_threshold": 0.015,
        "streaming_depth_stride": 4,
        "streaming_depth_loss_weight": 0.1,
        "streaming_free_space_loss_weight": 0.01,
        "streaming_max_new_gaussians_per_frame": 2000,
        "streaming_keyframe_input_fps": 60.0,
        "streaming_keyframe_max_gap": 20,
        "streaming_keyframe_max_gap_require_novelty": True,
        "streaming_eval_hold": 8,
        "streaming_eval_guard": 0,
        "streaming_eval_min_baseline_m": 0.03,
        "streaming_eval_min_angle_deg": 2.0,
        "streaming_report_final": True,
        # The fixed, evenly spaced scored cohort is deliberately smaller than
        # the 993 reserved holdouts.  The managed receipt pins its identity.
        "streaming_report_test_metrics_max": 128,
        "streaming_track_primary_psnr": True,
        "streaming_mapping_holdout_curve": True,
        "streaming_mapping_holdout_curve_async": True,
        "streaming_psnr_chart_interval": 100,
        "streaming_memory_telemetry": True,
        "streaming_duty_cycle_telemetry": True,
        "causal_prepared_ready_b4": True,
        "causal_prepared_ready_source_insertion": True,
        "streaming_optimization_constellation_policy": "fixed",
        "streaming_replay_depth_max_per_event": 0,
        "streaming_admission_prematerialize": False,
        "streaming_mapping_resolution_tier": "adaptive",
        "streaming_resolution_resampling_mode": "masked",
    }
    assert {key: composed[key] for key in expected_deltas} == expected_deltas
    changed = {
        key: value
        for key, value in composed.items()
        if baseline.get(key) != value
    }
    assert changed == {
        key: value
        for key, value in expected_deltas.items()
        if baseline.get(key) != value
    }

    args, _bundle = parse_training_args([])
    for key, value in validate_config(composed).items():
        setattr(args, key, value)
    args.model_path = str(tmp_path / "model")
    runtime = mapping_runtime_config_from_args(args)

    assert runtime.static_options["prepared_ready_source_insertion"] is True
    assert runtime.static_options["replay_depth_max_per_event"] == 0
    assert runtime.runtime.admission_prematerialize_enabled is False
    assert runtime.static_options["mapping_resolution_tier"] == "adaptive"
    assert runtime.static_options["mapping_resolution_min_tier"] == "eighth"
    assert runtime.static_options["resolution_resampling_mode"] == "masked"
    assert baseline["streaming_mapping_resolution_min_tier"] == "eighth"

    from eligsir.runtime.causal_evaluation import bounded_evaluation_cameras

    # The score cap is load-bearing: mapping curves and the endpoint both use
    # this deterministic native-resolution subset, not merely constructed cams.
    scored = bounded_evaluation_cameras(list(range(993)), max_items=128)
    assert len(scored) == 128
    assert scored[0] == 0
    assert scored[-1] == 992


def test_scannetpp_eighth_profile_prepares_a_240x180_ready_payload(monkeypatch):
    """The profile floor must reach the pre-READY payload producer."""

    from types import SimpleNamespace

    import torch

    from eligsir.cli.config import load_cli_config
    from eligsir.contracts.control import ResolutionTier
    from eligsir.mapping.resolution_materialization import materialize_resolution_tensors
    from eligsir.runtime import causal_prepared_ready as prepared

    composed = load_cli_config(
        run_profile="live/canonical_mapping", config_files=[str(_PROFILE)]
    )
    captured: dict[str, frozenset[ResolutionTier]] = {}

    class Event:
        def query(self):
            return True

    class Transfer:
        device = torch.device("cpu")

        def prepare_source(self, *_args, **_kwargs):
            return None

        def prepare_source_tiers(
            self, rgb, depth, intrinsics, *, resampling_mode, declared_tiers,
            depth_downsample="nearest",
        ):
            captured["tiers"] = declared_tiers
            return (
                {
                    tier.value: materialize_resolution_tensors(
                        rgb, depth, intrinsics, tier, resampling_mode=resampling_mode,
                        depth_downsample=depth_downsample,
                    )
                    for tier in declared_tiers
                },
                Event(),
            )

    class Bridge:
        def __init__(self, handoff, producer, **_kwargs):
            self.handoff = handoff
            self.producer = producer

        def close(self):
            pass

    class Worker:
        def __init__(self, *_args, **_kwargs):
            pass

        def start(self):
            return self

        def close(self):
            pass

    class Runtime:
        initialized = False
        bootstrap_frames = 4

        def enable_ready_source_owner(self, _owner, *, warmup_complete):
            assert warmup_complete is False

        def retain_prepared_source(self, _item, *, lane):
            return True

        def record_retained_prepared_commit(self, _item):
            pass

        def clear_retained_prepared_commit(self, _item):
            pass

        def mark_source_complete(self):
            pass

    monkeypatch.setattr(prepared, "PreparedRuntimeBridge", Bridge)
    monkeypatch.setattr(prepared, "PreparedSourcePrepWorker", Worker)
    monkeypatch.setattr(prepared, "_transfer_owner_for_runtime", lambda _runtime: Transfer())
    path = prepared.build_causal_prepared_ready_path(
        runtime=Runtime(),
        config=SimpleNamespace(
            hard_limits={"optimization_batch_size": 4},
            static_options={
                "optimization_constellation_policy": "fixed",
                "mapping_resolution_tier": composed["streaming_mapping_resolution_tier"],
                "mapping_resolution_min_tier": composed[
                    "streaming_mapping_resolution_min_tier"
                ],
                "resolution_resampling_mode": composed[
                    "streaming_resolution_resampling_mode"
                ],
                "insert_from_depth": False,
                "prepared_ready_source_insertion": False,
            },
        ),
        args=object(),
    )
    try:
        expected_tiers = frozenset(ResolutionTier)
        assert path._runtime._prepared_tier_policy.declared_tiers == expected_tiers

        native_rgb = torch.zeros((1440, 1920, 3), dtype=torch.float32)
        native_depth = torch.ones((1440, 1920), dtype=torch.float32)
        native_K = torch.tensor(
            ((960.0, 0.0, 959.5), (0.0, 720.0, 719.5), (0.0, 0.0, 1.0))
        )
        payloads, _event = path.bridge.producer._prepare_tier_payloads(
            native_rgb, native_depth, native_K, fallback_event=Event()
        )
        assert captured["tiers"] == expected_tiers
        eighth_rgb, eighth_depth, eighth_K, _confidence = payloads["eighth"]
        assert tuple(eighth_rgb.shape) == (180, 240, 3)
        assert tuple(eighth_depth.shape) == (180, 240)
        assert torch.allclose(
            eighth_K,
            torch.tensor(((120.0, 0.0, 119.5), (0.0, 90.0, 89.5), (0.0, 0.0, 1.0))),
        )
    finally:
        path.close()


def test_scannetpp_eighth_materialization_restores_the_native_scored_grid():
    """EIGHTH mapping supervision must not replace native scoring/insertion identity."""

    from types import SimpleNamespace

    import torch

    from eligsir.contracts.control import ResolutionTier
    from eligsir.mapping.resolution_materialization import materialize_resolution_tier
    from eligsir.utils.score_contract import declare_canonical_score_tensor

    native_rgb, native_provenance = declare_canonical_score_tensor(
        torch.zeros((3, 1440, 1920), dtype=torch.float32), context="scannetpp-native"
    )
    camera = SimpleNamespace(
        original_image=native_rgb,
        _sensor_depth_cache=torch.ones((1440, 1920), dtype=torch.float32),
        image_width=1920,
        image_height=1440,
        width=1920,
        height=1440,
        fx=960.0,
        fy=720.0,
        cx=959.5,
        cy=719.5,
        score_provenance=native_provenance,
        gsplat=None,
    )

    materialize_resolution_tier(camera, ResolutionTier.EIGHTH, resampling_mode="masked")
    assert (camera.image_width, camera.image_height) == (240, 180)
    assert camera._canonical_native_resolution_state.rgb is native_rgb
    assert camera._canonical_native_resolution_state.score_provenance is native_provenance

    materialize_resolution_tier(camera, ResolutionTier.NATIVE, resampling_mode="masked")
    assert (camera.image_width, camera.image_height) == (1920, 1440)
    assert camera.original_image is native_rgb
    assert camera.score_provenance is native_provenance
