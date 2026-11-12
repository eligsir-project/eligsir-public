"""Profile observation switches must cross the canonical policy boundary."""


def test_tum_mapping_curve_primary_tracking_reaches_optimization_policy(tmp_path):
    from eligsir.runtime.config_boundary import mapping_runtime_config_from_args
    from eligsir.runtime.production_application import _resolve_parameters

    args, *_ = _resolve_parameters(
        "causal_replay/tum_rgbd", str(tmp_path / "tum-mapping-curves")
    )

    assert args.streaming_track_primary_psnr is True
    assert (
        mapping_runtime_config_from_args(args).optimization.streaming_track_primary_psnr
        is True
    )


def test_live_primary_tracking_stays_off_at_optimization_policy_boundary(tmp_path):
    from eligsir.runtime.config_boundary import mapping_runtime_config_from_args
    from eligsir.runtime.production_application import _resolve_parameters

    args, *_ = _resolve_parameters(
        "live/canonical_mapping", str(tmp_path / "live-no-mapping-curves")
    )

    assert args.streaming_track_primary_psnr is False
    assert (
        mapping_runtime_config_from_args(args).optimization.streaming_track_primary_psnr
        is False
    )
