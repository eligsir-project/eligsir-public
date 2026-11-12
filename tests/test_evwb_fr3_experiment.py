"""Contract for the prospective, train-only FR3 EV/WB pair."""

from __future__ import annotations

import copy
import hashlib
import json
import tomllib
from pathlib import Path

import pytest
import yaml

from eligsir.utils.experiment_declaration import canonical_declaration_json, validate_declaration


ROOT = Path(__file__).resolve().parents[1] / "experiments/researchflow"
PROFILE_ROOT = Path(__file__).resolve().parents[1] / "configs/profiles"
CONTROL_PATH = ROOT / "fr3_evwb_control_20260907.yaml"
TREATMENT_PATH = ROOT / "fr3_evwb_treatment_20260907.yaml"
PROFILE_PATH = ROOT / "profiles.evwb_fr3_20260907.toml"
PROFILE_NAME = "fr3-evwb"
REFERENCE = "ed4a1cade5ec2062082d9afa4b0c6ebd"
IDENTITY_PARAMETERS = (
    "arm",
    "model_path",
    "declaration_json",
    "declaration_sha256",
    "experiment_id",
    "experiment_purpose",
)


def _load(path: Path) -> dict:
    assert path.is_file(), f"prospective EV/WB arm is missing: {path.name}"
    return yaml.safe_load(path.read_text())


def _profile() -> dict:
    return tomllib.loads(PROFILE_PATH.read_text())["profiles"][PROFILE_NAME]


def _render(parameters: dict) -> list[str]:
    profile = _profile()
    assert set(parameters) == set(profile["allowed_parameters"])
    return [word.format_map(parameters) for word in profile["argv"]]


def _declaration(parameters: dict) -> dict:
    declaration = json.loads(parameters["declaration_json"])
    validate_declaration(declaration)
    assert canonical_declaration_json(declaration) == parameters["declaration_json"]
    assert hashlib.sha256(parameters["declaration_json"].encode()).hexdigest() == parameters["declaration_sha256"]
    assert declaration["experiment_id"] == parameters["experiment_id"]
    assert declaration["arm"]["id"] == parameters["arm"]
    return declaration


def _normalized_argv(parameters: dict) -> list[str]:
    rendered = _render(parameters)
    for key in IDENTITY_PARAMETERS:
        rendered = [f"{{{key}}}" if word == str(parameters[key]) else word for word in rendered]
    return rendered


def _training_argv(parameters: dict) -> list[str]:
    rendered = _normalized_argv(parameters)
    return rendered[rendered.index("-m") :]


def _option_value(argv: list[str], option: str) -> str:
    return argv[argv.index(option) + 1]


def _assert_single_axis(control: dict, treatment: dict) -> None:
    control_parameters = control["run"]["parameters"]
    treatment_parameters = treatment["run"]["parameters"]
    changed = {key for key in control_parameters if control_parameters[key] != treatment_parameters[key]}
    assert changed == set(IDENTITY_PARAMETERS) | {"evwb_enabled_flag"}
    assert control_parameters["evwb_enabled_flag"] == "--no-streaming_train_evwb_enabled"
    assert treatment_parameters["evwb_enabled_flag"] == "--streaming_train_evwb_enabled"
    differences = [
        pair
        for pair in zip(_training_argv(control_parameters), _training_argv(treatment_parameters), strict=True)
        if pair[0] != pair[1]
    ]
    assert differences == [("--no-streaming_train_evwb_enabled", "--streaming_train_evwb_enabled")]


def test_evwb_pair_preserves_raw_native_fr3_contract() -> None:
    control = _load(CONTROL_PATH)
    treatment = _load(TREATMENT_PATH)
    control_parameters = control["run"]["parameters"]
    treatment_parameters = treatment["run"]["parameters"]

    assert control["run"]["profile"] == treatment["run"]["profile"] == PROFILE_NAME
    assert control["metadata"]["comparison_group_id"] == treatment["metadata"]["comparison_group_id"]
    assert control["metadata"]["expected_scored_holdout_hash"] == "df3a58440e8b7d76"
    assert control["metadata"]["expected_holdout_rule_hash"] == "6de07e64ff7c58df"
    assert control["metadata"]["expected_guard_views"] == 188
    assert control["metadata"]["source_content_digest"] == "b33eed4248cb4db8a6895c00decd61a7875ddbfa1ef70da22d546d7ecb25bda4"
    assert control_parameters["source_commit"] == "e4073f3eac04e4191aa2449de42e06d56c994f07"
    assert "raw/native mapping endpoint and curve artifacts" in control["metadata"]["mapping_curves"]
    assert set(control["evidence"]) == {"metrics", "telemetry"}
    assert "Trusted-mask and per-state-update observations are unavailable" in control["metadata"]["claim_boundary"]
    assert "execution validity only" in control["metadata"]["positive_verdict_scope"]
    assert "raw/native scored hashes, mapping curves, and whole-run telemetry" in control["metadata"]["positive_verdict_scope"]
    assert "not observed" in control["metadata"]["positive_verdict_scope"]
    assert control["acceptance"]["positive"]["all"] == [
        {"metric": "source_offered", "op": "==", "value": 2525},
        {"metric": "holdout_reserved", "op": "==", "value": 47},
        {"metric": "guard_withheld", "op": "==", "value": 188},
        {"metric": "transport_dropped", "op": "==", "value": 0},
        {"metric": "prepared_insertions_executed", "op": ">", "value": 0},
        {"metric": "supervision_views", "op": ">", "value": 0},
    ]
    for key in control_parameters:
        if key not in IDENTITY_PARAMETERS and key != "evwb_enabled_flag":
            assert treatment_parameters[key] == control_parameters[key]
    assert control_parameters["evwb_reference_source"] == REFERENCE

    for parameters, enabled in ((control_parameters, False), (treatment_parameters, True)):
        declaration = _declaration(parameters)
        assert declaration["arm"]["comparison_role"] == ("control" if not enabled else "treatment")
        assert declaration["arm"]["manipulation"].startswith("train-only display-referred EV/WB")
        trainer = _render(parameters)
        assert "--streaming_train_evwb_reference_source" in trainer
        assert trainer[trainer.index("--streaming_train_evwb_reference_source") + 1] == REFERENCE
        assert "--streaming_post_mapping_refinement_steps" in trainer
        assert trainer[trainer.index("--streaming_post_mapping_refinement_steps") + 1] == "0"
        assert "--streaming_post_mapping_refinement_seconds" in trainer
        assert trainer[trainer.index("--streaming_post_mapping_refinement_seconds") + 1] == "0"
        assert "--streaming_replay_challenger_fraction" not in trainer
        assert "--streaming_pose_correction_enabled" not in trainer
        assert "--causal_decision_trace" in trainer
        assert "--streaming_report_final" in trainer
    _assert_single_axis(control, treatment)


def test_evwb_pair_rejects_second_axis_or_declaration_drift() -> None:
    control = _load(CONTROL_PATH)
    treatment = _load(TREATMENT_PATH)

    second_axis = copy.deepcopy(treatment)
    second_axis["run"]["parameters"]["seed"] = 2
    with pytest.raises(AssertionError):
        _assert_single_axis(control, second_axis)

    invalid = copy.deepcopy(treatment)
    invalid["run"]["parameters"]["arm"] = "wrong-arm"
    with pytest.raises(AssertionError):
        _declaration(invalid["run"]["parameters"])


def test_evwb_profile_preflights_attested_source_overlay() -> None:
    profile = _profile()
    preflight = next(word for word in profile["argv"] if "checked_query git -C" in word)
    assert 'checked_query git -C "$1" rev-parse HEAD' in preflight
    assert 'checked_query git -C "$1" status --porcelain --untracked-files=all' in preflight
    assert '-v "$checkout:/workspace/eligsir:ro"' in preflight
    assert '-e ELIGSIR_GIT_COMMIT="$commit"' in preflight


def test_evwb_pair_locks_shared_workload_and_effective_curve_observer() -> None:
    control = _training_argv(_load(CONTROL_PATH)["run"]["parameters"])
    treatment = _training_argv(_load(TREATMENT_PATH)["run"]["parameters"])
    for argv in (control, treatment):
        assert _option_value(argv, "--streaming_input_fps_cap") == "30"
        assert _option_value(argv, "--streaming_optimization_constellation_policy") == "fixed"
        assert _option_value(argv, "--streaming_optimization_batch_size") == "16"
        assert _option_value(argv, "--streaming_optimization_camera_count_max") == "16"
        assert _option_value(argv, "--streaming_mapping_resolution_tier") == "adaptive"
        assert _option_value(argv, "--streaming_replay_target_views_per_s") == "200"
        assert _option_value(argv, "--streaming_resolution_budget_relax_max") == "1"
        assert "--streaming_replay_challenger_fraction" not in argv
        assert "--streaming_pose_correction_enabled" not in argv
    tum_profile = yaml.safe_load((PROFILE_ROOT / "causal_replay/tum_rgbd.yml").read_text())
    curves = yaml.safe_load((PROFILE_ROOT / "evaluation/mapping_curves.yml").read_text())
    assert "../evaluation/mapping_curves.yml" in tum_profile["extends"]
    assert curves == {
        "streaming_track_primary_psnr": True,
        "streaming_mapping_holdout_curve": True,
        "streaming_mapping_holdout_curve_async": True,
        "streaming_psnr_chart_interval": 100,
    }
