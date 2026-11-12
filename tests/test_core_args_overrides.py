"""Core CLI tokens -> mapping-runtime override dict (eligsir#1259).

The canonical ROS2 lifecycle node had no way to express a tuned scene recipe;
this translation is the passthrough it uses. The parity tests below are the
point of the module: an override dict must mean exactly what the same tokens
mean on a ``train.py`` command line, because that is the only form in which
this project's tuned recipes exist.
"""

from __future__ import annotations

import pytest

from eligsir.cli.parser import parse_training_args
from eligsir.runtime.core_args import CoreArgsError, core_args_to_overrides

# The floor2 champion recipe that surfaced the gap during the #1259 GPU
# campaign, including the two mandatory multires floor flags.
FLOOR2_RECIPE = [
    "--online_mode",
    "--streaming_max_iterations", "60000",
    "--energy_dead_mode", "energy_quantile",
    "--streaming_debt_replay",
    "--streaming_consolidation_sampler", "debt",
    "--sh_impl", "gsplat",
    "--streaming_anneal_horizon", "15000",
    "--cap_max", "600000",
    "--streaming_target_capacity", "200000",
    "--loop_closure_enabled",
    "--loop_closure_apply_correction",
    "--loop_force_field_interval", "500",
    "--streaming_multires_it_v_floor", "0",
    "--streaming_multires_map_fps_floor", "0",
]


def test_floor2_recipe_matches_trainer_parser_exactly():
    overrides = core_args_to_overrides(FLOOR2_RECIPE)
    args, _ = parse_training_args(FLOOR2_RECIPE)
    assert overrides == {key: getattr(args, key) for key in overrides}
    # Every named flag is present; nothing else is.
    assert set(overrides) == {
        "online_mode",
        "streaming_max_iterations",
        "energy_dead_mode",
        "streaming_debt_replay",
        "streaming_consolidation_sampler",
        "sh_impl",
        "streaming_anneal_horizon",
        "cap_max",
        "streaming_target_capacity",
        "loop_closure_enabled",
        "loop_closure_apply_correction",
        "loop_force_field_interval",
        "streaming_multires_it_v_floor",
        "streaming_multires_map_fps_floor",
    }
    assert overrides["cap_max"] == 600000
    assert overrides["loop_closure_enabled"] is True


def test_flag_set_to_its_argparse_default_is_still_an_override():
    """A defaults-diff would drop this; the profile value must still be beaten."""
    args, _ = parse_training_args([])
    default_cap = args.cap_max
    overrides = core_args_to_overrides(["--cap_max", str(default_cap)])
    assert overrides == {"cap_max": default_cap}


def test_equals_form_and_value_parity():
    assert core_args_to_overrides(["--cap_max=123456"]) == {"cap_max": 123456}


def test_empty_tokens_produce_no_overrides():
    assert core_args_to_overrides([]) == {}


def test_unknown_flag_is_rejected_loudly():
    with pytest.raises(CoreArgsError, match="unknown core argument"):
        core_args_to_overrides(["--cap_maxx", "1"])


def test_abbreviation_is_rejected():
    with pytest.raises(CoreArgsError, match="unknown core argument"):
        core_args_to_overrides(["--cap_m", "1"])


def test_help_is_rejected_instead_of_exiting_the_process():
    with pytest.raises(CoreArgsError, match="--help"):
        core_args_to_overrides(["--help"])


@pytest.mark.parametrize(
    "tokens",
    [
        ["--model_path", "/tmp/x"],
        ["-m", "/tmp/x"],
        ["--source_path", "/data/x"],
        ["--config", "configs/x.json"],
        ["--ros2_live_pose_topic", "/slam/odom"],
        ["--orbbec_pose_topic", "/camera_pose"],
    ],
)
def test_transport_owned_flags_are_refused(tokens):
    with pytest.raises(CoreArgsError, match="transport-owned configuration"):
        core_args_to_overrides(tokens)


def test_trainer_process_only_flag_is_refused_with_a_specific_message():
    with pytest.raises(CoreArgsError, match="not mapping run-config keys"):
        core_args_to_overrides(["--test_iterations", "7000"])


def test_non_string_tokens_are_refused():
    with pytest.raises(CoreArgsError, match="must be strings"):
        core_args_to_overrides(["--cap_max", 600000])  # type: ignore[list-item]


def test_negative_numeric_values_are_values_not_flags():
    assert core_args_to_overrides(["--noise_lr", "-1.5"]) == {"noise_lr": -1.5}


def test_overrides_actually_reach_the_resolved_mapping_parameters(tmp_path):
    from eligsir.runtime.production_application import (
        DEFAULT_MAPPING_PROFILE,
        _resolve_parameters,
    )

    overrides = core_args_to_overrides(["--cap_max", "600000"])
    args, _dataset, _opt, _pipe = _resolve_parameters(
        DEFAULT_MAPPING_PROFILE, str(tmp_path / "model"), overrides=overrides
    )
    assert args.cap_max == 600000
    # The node-owned model path is not clobbered by the recipe.
    assert args.model_path == str(tmp_path / "model")
