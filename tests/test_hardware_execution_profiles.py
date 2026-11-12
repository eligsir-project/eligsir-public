from pathlib import Path
import pytest

from eligsir.cli.config import load_cli_config
from eligsir.cli.parser import parse_training_args
from eligsir.mapping.scheduler.work_budget import _AdaptiveConstellationPolicy
from eligsir.runtime.config_boundary import mapping_runtime_config_from_args
from eligsir.runtime.production_application import CanonicalMappingRuntime


REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("gpu", "mode", "camera_cap", "budget_ms"),
    (
        ("8060s", "live_throughput", 4, 25),
        ("r9700", "live_throughput", 16, 25),
        ("rtx3090", "live_throughput", 32, 25),
        ("8060s", "high_frequency", 4, 10),
        ("r9700", "high_frequency", 4, 10),
        ("rtx3090", "high_frequency", 8, 10),
        ("8060s", "refinement", 4, 0),
        ("r9700", "refinement", 32, 0),
        ("rtx3090", "refinement", 32, 0),
    ),
)
def test_kernel_derived_hardware_profiles_do_not_select_method_batch(
    gpu, mode, camera_cap, budget_ms
):
    path = REPO_ROOT / "configs" / "profiles" / "hardware" / gpu / f"{mode}.yml"
    raw = path.read_text(encoding="utf-8")
    assert "streaming_optimization_batch_size:" not in raw

    resolved = load_cli_config(config_files=[str(path)])
    assert resolved["streaming_optimization_camera_count_max"] == camera_cap
    assert resolved["streaming_optimization_transaction_budget_ms"] == budget_ms


def test_hardware_camera_cap_is_independent_of_b_and_never_exceeds_it():
    b64_capped = _AdaptiveConstellationPolicy(
        requested_batch_size=64,
        maximum_camera_count=16,
    )
    assert b64_capped._maximum_camera_count == 16
    assert b64_capped._camera_count == 16

    b4_capped = _AdaptiveConstellationPolicy(
        requested_batch_size=4,
        maximum_camera_count=32,
    )
    assert b4_capped._maximum_camera_count == 4
    assert b4_capped._camera_count == 4


@pytest.mark.parametrize("invalid", (3, 6, 256, True, 4.5, "4"))
def test_hardware_camera_cap_rejects_unqualified_values(invalid):
    with pytest.raises((TypeError, ValueError), match="camera.count|camera count"):
        _AdaptiveConstellationPolicy(
            requested_batch_size=64,
            maximum_camera_count=invalid,
        )


def test_config_boundary_resolves_camera_cap_and_explicit_freshness_budget():
    args, _ = parse_training_args(["-s", "/dev/null"])
    args.model_path = "output/hardware-profile-config-test"
    args.online_mode = True
    args.streaming_optimization_batch_size = 64
    args.streaming_optimization_camera_count_max = 16
    args.streaming_optimization_transaction_budget_ms = 25
    args.streaming_optimization_constellation_policy = "fixed"

    config = mapping_runtime_config_from_args(args)

    assert config.hard_limits["optimization_camera_count_max"] == 16
    assert config.runtime.max_optimization_transaction_ns == 25_000_000
    assert config.static_options["optimization_constellation_policy"] == "fixed"


@pytest.mark.parametrize("invalid", ("", "static", "FIXED", 1, True, None))
def test_config_boundary_rejects_invalid_constellation_policy(invalid):
    args, _ = parse_training_args(["-s", "/dev/null"])
    args.model_path = "output/hardware-profile-invalid-constellation-test"
    args.streaming_optimization_constellation_policy = invalid

    with pytest.raises((TypeError, ValueError), match="constellation_policy"):
        mapping_runtime_config_from_args(args)


@pytest.mark.parametrize("invalid", (6, True, -1))
def test_config_boundary_rejects_invalid_hardware_camera_caps(invalid):
    args, _ = parse_training_args(["-s", "/dev/null"])
    args.model_path = "output/hardware-profile-invalid-test"
    args.streaming_optimization_camera_count_max = invalid

    with pytest.raises((TypeError, ValueError), match="camera_count_max|camera count"):
        mapping_runtime_config_from_args(args)


@pytest.mark.parametrize(
    ("requested", "forced", "expected_enabled", "expected_reason"),
    (
        (False, False, False, "disabled"),
        (True, False, True, "explicit"),
        (False, True, True, "prepared_b4_required"),
        (True, True, True, "explicit"),
    ),
)
def test_admission_prefetch_provenance_distinguishes_requested_from_effective(
    requested, forced, expected_enabled, expected_reason
):
    """Issue #1543 finding A: `causal_prepared_ready_b4` can force admission
    prefetch on independently of `--streaming_admission_prefetch`; config
    provenance must preserve both instead of collapsing them into one flag.
    """

    args, _ = parse_training_args(["-s", "/dev/null"])
    args.model_path = "output/admission-prefetch-provenance-test"
    args.streaming_admission_prefetch = requested
    args.causal_prepared_ready_b4 = forced

    runtime = mapping_runtime_config_from_args(args).runtime

    assert runtime.admission_prefetch_requested is requested
    assert runtime.admission_prefetch_enabled is expected_enabled
    assert runtime.admission_prefetch_reason == expected_reason


def test_zero_budget_keeps_existing_mode_derived_default():
    args, _ = parse_training_args(["-s", "/dev/null"])
    args.model_path = "output/hardware-profile-default-test"
    args.streaming_optimization_transaction_budget_ms = 0

    assert mapping_runtime_config_from_args(args).runtime.max_optimization_transaction_ns is None
    args.online_mode = True
    assert (
        mapping_runtime_config_from_args(args).runtime.max_optimization_transaction_ns
        == 33_333_333
    )


def test_hardware_profile_is_an_execution_only_overlay_on_canonical_live_policy():
    path = REPO_ROOT / "configs/profiles/hardware/r9700/live_throughput.yml"
    canonical = load_cli_config(run_profile="live/canonical_mapping")
    composed = load_cli_config(
        run_profile="live/canonical_mapping", config_files=[str(path)]
    )

    execution_keys = {
        "streaming_optimization_camera_count_max",
        "streaming_optimization_transaction_budget_ms",
    }
    assert set(composed).difference(canonical) <= execution_keys
    for key, value in canonical.items():
        if key not in execution_keys:
            assert composed[key] == value, key
    assert composed["streaming_optimization_camera_count_max"] == 16
    assert composed["streaming_optimization_transaction_budget_ms"] == 25


def test_canonical_runtime_passes_resolved_camera_cap_to_work_budget_owner(monkeypatch):
    args, _ = parse_training_args(["-s", "/dev/null"])
    args.model_path = "output/hardware-profile-production-wiring-test"
    args.streaming_optimization_batch_size = 64
    args.streaming_optimization_camera_count_max = 16
    config = mapping_runtime_config_from_args(args)
    captured = {}

    class CapturingOwner:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    import eligsir.runtime.production_application as production

    monkeypatch.setattr(production, "WorkBudgetOwner", CapturingOwner, raising=False)
    monkeypatch.setattr(
        production,
        "WeightedReplayCandidateProvider",
        lambda **kwargs: ("weighted-replay", kwargs),
        raising=False,
    )
    owner = CanonicalMappingRuntime._make_work_budget_owner(
        config,
        batch_size=64,
        depth_insertion_policy=config.depth_insertion,
    )

    assert isinstance(owner, CapturingOwner)
    assert captured["requested_batch_size"] == 64
    assert captured["maximum_camera_count"] == 16
    assert captured["constellation_policy"] == "adaptive"
