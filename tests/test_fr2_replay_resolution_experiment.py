"""Guard the single changed execution axis of the FR2 replay-resolution pair."""

import hashlib
import json
import tomllib
from pathlib import Path

from eligsir.utils.experiment_declaration import (
    canonical_declaration_json,
    validate_declaration,
)


ROOT = Path(__file__).resolve().parents[1] / "experiments/researchflow"


def test_resolution_pair_changes_only_tier_and_run_identity():
    paths = [
        ROOT / f"fr2-replay-resolution-{tier}-20260906.json"
        for tier in ("adaptive", "native")
    ]
    assert all(path.is_file() for path in paths), "both prospective arms must exist"
    specs = [json.loads(path.read_text()) for path in paths]
    profile = tomllib.loads((ROOT / "profiles.fr2_replay_resolution.toml").read_text())[
        "profiles"
    ]["tum-local-replay-resolution"]
    trainer_argv = []
    identity_parameters = {
        "arm",
        "model_path",
        "declaration_json",
        "declaration_sha256",
    }
    parameters = []
    for spec, tier in zip(specs, ("adaptive", "native"), strict=True):
        params = spec["run"]["parameters"]
        declaration = json.loads(params["declaration_json"])
        validate_declaration(declaration)
        assert canonical_declaration_json(declaration) == params["declaration_json"]
        assert (
            hashlib.sha256(params["declaration_json"].encode()).hexdigest()
            == params["declaration_sha256"]
        )
        assert declaration["arm"]["id"] == params["arm"]
        assert params["mapping_resolution_tier"] == tier
        assert params["source_commit"] == "645c3f6d0e23beb93054f7afdc902a3b15d3d67b"
        assert params["source_checkout"].endswith("/fr2-resolution-execution-645c3f6d")
        assert (
            spec["metadata"]["expected_scored_holdouts"]["hash"] == "499bcf9ee206dbca"
        )
        assert (
            len(spec["metadata"]["expected_scored_holdouts"]["source_ordinals"]) == 59
        )
        assert set(params) == set(profile["allowed_parameters"])
        argv = [word.format_map(params) for word in profile["argv"]]
        trainer = argv[argv.index("train.py") :]
        for flag, expected in (
            ("--streaming_optimization_batch_size", "32"),
            ("--streaming_optimization_camera_count_max", "32"),
            ("--streaming_input_fps_cap", "30"),
            ("--streaming_post_mapping_refinement_steps", "0"),
            ("--streaming_post_mapping_refinement_seconds", "0"),
            ("--streaming_mapping_resolution_tier", tier),
        ):
            assert trainer[trainer.index(flag) + 1] == expected
        for flag in (
            "--causal_prepared_ready_source_insertion",
            "--streaming_objective_step_regularizers",
            "--streaming_objective_batched_pixel_terms",
        ):
            assert flag in trainer
        for flag in ("-m", "--experiment_arm"):
            trainer[trainer.index(flag) + 1] = "RUN_IDENTITY"
        trainer_argv.append(trainer)
        parameters.append(
            {
                key: value
                for key, value in params.items()
                if key not in identity_parameters
            }
        )
    assert {
        key for key in parameters[0] if parameters[0][key] != parameters[1][key]
    } == {"mapping_resolution_tier"}
    differences = [
        (left, right)
        for left, right in zip(*trainer_argv, strict=True)
        if left != right
    ]
    assert differences == [("adaptive", "native")]


def test_resolution_profile_preserves_existing_validated_launch_contract():
    path = ROOT / "profiles.fr2_replay_resolution.toml"
    assert path.is_file(), "isolated resolution profile must exist"
    profile = tomllib.loads(path.read_text())["profiles"]["tum-local-replay-resolution"]
    baseline = tomllib.loads(
        (ROOT / "profiles.tum_local_source_b4_objective.toml").read_text()
    )["profiles"]["tum-local-source-b4-objective"]
    profile["allowed_parameters"].remove("mapping_resolution_tier")
    argv = profile["argv"]
    argv[argv.index("--streaming_mapping_resolution_tier") + 1] = "adaptive"
    assert profile == baseline
