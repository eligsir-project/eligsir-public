"""Configuration boundary for the opt-in unified population owner."""

from __future__ import annotations

import subprocess
import sys

import pytest

from eligsir.cli.parser import build_training_parser, parse_training_args
from eligsir.runtime.config_boundary import mapping_runtime_config_from_args


def test_cli_help_marks_m9_controls_as_reproduction_only():
    help_text = build_training_parser().parser.format_help()

    assert "M9 legacy population controls" in help_text
    assert "reproduction" in help_text
    assert "--streaming_depth_lock_centers" in help_text
    assert "--densification_strategy=gsplat_mcmc" in help_text


def test_direct_parser_rejects_raw_delegated_mcmc_for_live_population(capsys):
    with pytest.raises(SystemExit) as error:
        parse_training_args(
            [
                "-s",
                "/dev/null",
                "--streaming_population_live_enabled",
                "--streaming_population_shadow_enabled",
                "--no-streaming_lifecycle_enabled",
                "--no-streaming_insert_from_depth",
                "--mcmc_noise_stop_iter",
                "5",
            ]
        )

    assert error.value.code == 2
    assert "--mcmc_noise_stop_iter" in capsys.readouterr().err


def test_trainer_rejects_raw_delegated_mcmc_after_live_profile_merge(tmp_path):
    # ``train.py`` deliberately imports the full Torch/renderer stack before
    # reaching its post-profile boundary.  The torch-free smoke image cannot
    # execute that entrypoint; the direct-parser test above still exercises
    # the CPU-safe raw-argv boundary there.  The CPU-Torch simulation lane
    # runs this subprocess test and verifies the actual trainer ordering.
    pytest.importorskip(
        "torch",
        reason="trainer entrypoint requires torch; exercised in the CPU-Torch simulation lane",
    )
    profile = tmp_path / "live_population.yml"
    profile.write_text(
        "\n".join(
            [
                "streaming_population_live_enabled: true",
                "streaming_population_shadow_enabled: true",
                "streaming_lifecycle_enabled: false",
                "streaming_insert_from_depth: false",
            ]
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "train.py",
            "-s",
            "/dev/null",
            "-m",
            str(tmp_path / "output"),
            "--config",
            str(profile),
            "--mcmc_noise_stop_iter",
            "5",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "--mcmc_noise_stop_iter" in result.stderr + result.stdout


def _unified_population_args():
    args, _ = parse_training_args(["-s", "/dev/null", "-m", "output/test"])
    args.streaming_population_live_enabled = True
    args.streaming_population_shadow_enabled = True
    args.densification_strategy = "gsplat_energy_mcmc"
    args.streaming_lifecycle_enabled = False
    args.streaming_post_mapping_mature_mobility = False
    args.streaming_depth_lock_centers = False
    args.streaming_insert_from_depth = False
    return args


@pytest.mark.parametrize(
    ("field", "value", "option"),
    [
        ("streaming_lifecycle_enabled", True, "--streaming-lifecycle-enabled"),
        (
            "streaming_post_mapping_mature_mobility",
            True,
            "--streaming-post-mapping-mature-mobility",
        ),
        ("streaming_depth_lock_centers", True, "--streaming-depth-lock-centers"),
        (
            "streaming_depth_lock_bootstrap",
            True,
            "--streaming-depth-lock-bootstrap",
        ),
        (
            "streaming_depth_lock_insertions",
            True,
            "--streaming-depth-lock-insertions",
        ),
        (
            "streaming_depth_lock_prune_conflicts",
            True,
            "--streaming-depth-lock-prune-conflicts",
        ),
        ("densification_strategy", "gsplat_mcmc", "--densification-strategy"),
        ("streaming_insert_from_depth", True, "--streaming-insert-from-depth"),
    ],
)
def test_live_population_rejects_competing_legacy_population_controls(
    field, value, option
):
    args = _unified_population_args()
    setattr(args, field, value)

    with pytest.raises(ValueError, match=option):
        mapping_runtime_config_from_args(args)


def test_live_population_accepts_the_unified_ownership_configuration():
    config = mapping_runtime_config_from_args(_unified_population_args())

    assert config.static_options["population_live_enabled"] is True
    assert config.static_options["population_ownership_strategy"] == "gsplat_energy_mcmc"
    assert config.static_options["population_control_mode"] == "live_probabilistic"
    assert (
        config.static_options["legacy_population_controls_status"]
        == "reproduction_only_pending_researchflow_horizon"
    )


def test_live_population_disabled_preserves_legacy_defaults():
    args, _ = parse_training_args(["-s", "/dev/null", "-m", "output/test"])

    config = mapping_runtime_config_from_args(args)

    assert config.static_options["population_live_enabled"] is False
    assert config.static_options["population_ownership_strategy"] == "gsplat_energy_mcmc"
    assert config.static_options["legacy_lifecycle_enabled"] is True
    assert config.static_options["legacy_insert_from_depth"] is True
    assert config.static_options["population_control_mode"] == "legacy_compatibility"
    assert (
        config.static_options["legacy_population_controls_status"]
        == "reproduction_only_pending_researchflow_horizon"
    )


def test_legacy_compatibility_preserves_m9_control_values_for_reproduction():
    args, _ = parse_training_args(["-s", "/dev/null", "-m", "output/test"])
    args.streaming_lifecycle_enabled = True
    args.streaming_post_mapping_mature_mobility = True
    args.streaming_depth_lock_centers = True
    args.streaming_depth_lock_bootstrap = True
    args.streaming_depth_lock_insertions = True
    args.streaming_depth_lock_prune_conflicts = True
    args.streaming_insert_from_depth = True
    args.densification_strategy = "gsplat_mcmc"

    config = mapping_runtime_config_from_args(args)

    assert config.static_options["population_control_mode"] == "legacy_compatibility"
    assert config.static_options["legacy_lifecycle_enabled"] is True
    assert config.static_options["legacy_post_mapping_mature_mobility"] is True
    assert config.static_options["legacy_depth_lock_centers"] is True
    assert config.static_options["legacy_depth_lock_bootstrap"] is True
    assert config.static_options["legacy_depth_lock_insertions"] is True
    assert config.static_options["legacy_depth_lock_prune_conflicts"] is True
    assert config.static_options["legacy_insert_from_depth"] is True
    assert config.static_options["legacy_delegated_mcmc"] is True
