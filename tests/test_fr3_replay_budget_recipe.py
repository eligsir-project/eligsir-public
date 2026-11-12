"""Contract for the prospective FR3 replay-target 200/100 views-per-second pair."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest
import yaml

from eligsir.utils.experiment_declaration import (
    DECLARATION_ENV,
    DECLARATION_SHA_ENV,
    ExperimentDeclarationError,
    load_experiment_context,
)


ROOT = Path(__file__).resolve().parents[1] / "experiments/researchflow"
PROFILE_PATH = ROOT / "profiles.fr3_replay_budget_20260907.toml"
PROFILE_NAME = "fr3-replay-budget-r9700"
SPECS = {
    "control": ROOT / "fr3_replay_budget_control_200_20260907.yaml",
    "treatment": ROOT / "fr3_replay_budget_treatment_100_20260907.yaml",
}
SOURCE_COMMIT = "12357445253bf3e9e1e887644681a57d5542ffe9"
IMAGE = "sha256:a0e85ff72e46b304a016609140811526e4336efdec6ba158d5e4237ac71472a2"
SCORING_HASH = "df3a58440e8b7d76"
RULE_HASH = "6de07e64ff7c58df"
IDENTITY_PARAMETERS = {
    "arm",
    "model_path",
    "experiment_id",
    "experiment_purpose",
    "declaration_json",
    "declaration_sha256",
    "replay_target_views_per_s",
}


def _profile() -> dict:
    return tomllib.loads(PROFILE_PATH.read_text(encoding="utf-8"))["profiles"][PROFILE_NAME]


def _specs() -> dict[str, dict]:
    return {name: yaml.safe_load(path.read_text(encoding="utf-8")) for name, path in SPECS.items()}


def _render(profile: dict, parameters: dict) -> list[str]:
    assert set(parameters) == set(profile["allowed_parameters"])
    return [part.format_map(parameters) for part in profile["argv"]]


def _launcher(argv: list[str]) -> str:
    return argv[argv.index("-euc") + 1]


def _mapper_argv(argv: list[str], parameters: dict) -> list[str]:
    mapper = argv[argv.index("-m") :]
    identity_values = {
        str(parameters[key])
        for key in IDENTITY_PARAMETERS - {"replay_target_views_per_s"}
    }
    return ["<identity>" if part in identity_values else part for part in mapper]


def _option(argv: list[str], name: str) -> str:
    return argv[argv.index(name) + 1]


def _declaration(parameters: dict) -> dict:
    context = load_experiment_context(
        {
            DECLARATION_ENV: parameters["declaration_json"],
            DECLARATION_SHA_ENV: parameters["declaration_sha256"],
            "ABLATOR_JOB_ID": "fr3-replay-budget-test",
        }
    )
    return context["experiment_declaration"]


def _assert_recipe_contract(profile: dict, specs: dict[str, dict]) -> None:
    control, treatment = specs.values()
    assert profile["ablator_type"] == "researchflow"
    assert profile["machine"] == "r9700"
    assert profile["ablator_lane"] == 3
    assert profile["metrics_path"] == "{model_path}/causal_replay_summary.json"
    assert profile["resources"] == {"gpus": 1, "cpus": 4, "memory_mb": 24576, "runtime_minutes": 180}
    compile(profile["argv"][2], "telemetry_wrapper", "exec")
    assert "checked_query /usr/bin/python3 -c" in _launcher(profile["argv"])

    control_parameters = control["run"]["parameters"]
    treatment_parameters = treatment["run"]["parameters"]
    for spec, parameters, role, target in (
        (control, control_parameters, "control", "200"),
        (treatment, treatment_parameters, "treatment", "100"),
    ):
        rendered = _render(profile, parameters)
        declaration = _declaration(parameters)
        assert subprocess.run(["bash", "-n"], input=_launcher(rendered), text=True, capture_output=True).returncode == 0
        assert spec["run"]["profile"] == PROFILE_NAME
        assert "required_artifacts" not in spec["metadata"]
        assert "paired_analysis_artifacts" in spec["metadata"]
        assert "Execution validity only" in spec["metadata"]["positive_verdict_scope"]
        assert "no-finer-replay" in spec["metadata"]["positive_verdict_scope"]
        assert parameters["source_commit"] == SOURCE_COMMIT
        assert any(IMAGE in part for part in rendered)
        assert parameters["expected_source_count"] == 2525
        assert parameters["expected_train_count"] == 2290
        assert parameters["expected_scored_count"] == 47
        assert parameters["expected_guard_count"] == 188
        assert parameters["cohort_hash"] == SCORING_HASH
        assert parameters["cohort_rule_hash"] == RULE_HASH
        assert declaration["arm"]["comparison_role"] == role
        assert declaration["fixed"]["source_commit"] == SOURCE_COMMIT
        assert declaration["fixed"]["container_digest"] == IMAGE
        assert declaration["fixed"]["replay_target_views_per_s"] == int(target)
        assert declaration["evaluation"] == {
            "guard": 188,
            "holdout_rule_hash": RULE_HASH,
            "native_holdout": True,
            "scored": 47,
            "scored_hash": SCORING_HASH,
            "source_count": 2525,
            "train": 2290,
        }
        assert declaration["source"]["descriptor_sha256"] == parameters["input_storage_manifest_sha256"]
        assert declaration["source"]["content_sha256"] == parameters["input_storage_content_digest"]
        assert "no default promotion" in declaration["claim_boundary"]
        assert "HSA_XNACK" not in " ".join(rendered)
        assert "HSA_ENABLE_SDMA" not in " ".join(rendered)
        assert _option(rendered, "--streaming_replay_target_views_per_s") == target
        assert _option(rendered, "--streaming_resolution_budget_relax_max") == "1"
        assert _option(rendered, "--streaming_optimization_constellation_policy") == "fixed"
        assert _option(rendered, "--streaming_optimization_batch_size") == "16"
        assert _option(rendered, "--streaming_optimization_camera_count_max") == "16"
        assert _option(rendered, "--streaming_mapping_resolution_tier") == "adaptive"
        assert _option(rendered, "--streaming_input_fps_cap") == "30"
        assert _option(rendered, "--streaming_post_mapping_refinement_steps") == "0"
        assert _option(rendered, "--streaming_post_mapping_refinement_seconds") == "0"
        assert _option(rendered, "--streaming_eval_hold") == "16"
        assert _option(rendered, "--streaming_eval_guard") == "2"
        assert "--no-streaming_train_evwb_enabled" in rendered
        assert "--streaming_pose_correction_enabled" not in rendered
        assert "--streaming_replay_challenger_fraction" not in rendered

    changed = {
        key
        for key in control_parameters
        if control_parameters[key] != treatment_parameters[key]
    }
    assert changed == IDENTITY_PARAMETERS
    control_argv = _mapper_argv(_render(profile, control_parameters), control_parameters)
    treatment_argv = _mapper_argv(_render(profile, treatment_parameters), treatment_parameters)
    differences = [pair for pair in zip(control_argv, treatment_argv, strict=True) if pair[0] != pair[1]]
    assert differences == [("200", "100")]


def test_fr3_replay_budget_pair_renders_one_scientific_axis() -> None:
    _assert_recipe_contract(_profile(), _specs())


@pytest.mark.parametrize(
    ("parameter", "replacement"),
    [("seed", 2), ("source_commit", "0" * 40), ("input_storage_content_digest", "0" * 64)],
)
def test_fr3_replay_budget_pair_rejects_shared_axis_drift(parameter: str, replacement: str) -> None:
    profile = _profile()
    specs = _specs()
    changed = copy.deepcopy(specs)
    changed["treatment"]["run"]["parameters"][parameter] = replacement
    with pytest.raises(AssertionError):
        _assert_recipe_contract(profile, changed)


def test_fr3_replay_budget_pair_rejects_declaration_drift() -> None:
    bad_declaration = copy.deepcopy(_specs()["control"])
    bad_declaration["run"]["parameters"]["declaration_sha256"] = "0" * 64
    with pytest.raises(ExperimentDeclarationError, match="SHA-256 mismatch"):
        _declaration(bad_declaration["run"]["parameters"])



def test_preflight_failure_never_reaches_scientific_docker_launch(tmp_path: Path) -> None:
    """Changing the exact source identity must stop before the mapper command."""
    profile = _profile()
    params = copy.deepcopy(_specs()["control"]["run"]["parameters"])
    wrong_source = tmp_path / "wrong-source"
    wrong_source.mkdir()
    subprocess.run(["git", "init", "-q", str(wrong_source)], check=True)
    subprocess.run(["git", "-C", str(wrong_source), "config", "user.email", "test@example.invalid"], check=True)
    subprocess.run(["git", "-C", str(wrong_source), "config", "user.name", "Test"], check=True)
    (wrong_source / "README").write_text("test\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(wrong_source), "add", "README"], check=True)
    subprocess.run(["git", "-C", str(wrong_source), "commit", "-qm", "test"], check=True)
    params["source_checkout"] = str(wrong_source)
    argv = _render(profile, params)
    docker = tmp_path / "bin/docker"
    docker.parent.mkdir()
    docker.write_text("#!/bin/sh\nprintf docker-should-not-run > \"$DOCKER_SENTINEL\"\n", encoding="utf-8")
    docker.chmod(0o755)
    hostile = tmp_path / "hostile-bash-env.sh"
    hostile.write_text("printf hostile-bash-env-executed >&2\nexit 79\n", encoding="utf-8")
    sentinel = tmp_path / "docker-sentinel"
    result = subprocess.run(
        argv[argv.index("env") :],
        env={
            **os.environ,
            "BASH_ENV": str(hostile),
            "DOCKER_SENTINEL": str(sentinel),
            "PATH": f"{docker.parent}:{os.environ['PATH']}",
        },
        text=True,
        capture_output=True,
    )
    assert result.returncode != 0
    assert "source checkout commit mismatch" in result.stderr
    assert "hostile-bash-env-executed" not in result.stderr
    assert not sentinel.exists()


@pytest.mark.parametrize("docker_ps_fails", [False, True])
def test_rendered_preflight_blocks_foreign_or_unqueryable_gpu_before_any_docker_run(
    tmp_path: Path, docker_ps_fails: bool
) -> None:
    """A complete preflight must reject a live foreign mapper before launching ours."""
    profile = copy.deepcopy(_profile())
    launcher_index = next(
        index for index, value in enumerate(profile["argv"])
        if "descriptor content mismatch" in value
    )
    profile["argv"][launcher_index] = profile["argv"][launcher_index].replace(
        "/usr/bin/python3", sys.executable
    )
    params = copy.deepcopy(_specs()["control"]["run"]["parameters"])
    checkout = tmp_path / "source"
    checkout.mkdir()
    subprocess.run(["git", "init", "-q", str(checkout)], check=True)
    subprocess.run(["git", "-C", str(checkout), "config", "user.email", "test@example.invalid"], check=True)
    subprocess.run(["git", "-C", str(checkout), "config", "user.name", "Test"], check=True)
    (checkout / "README").write_text("test\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(checkout), "add", "README"], check=True)
    subprocess.run(["git", "-C", str(checkout), "commit", "-qm", "test"], check=True)
    commit = subprocess.run(["git", "-C", str(checkout), "rev-parse", "HEAD"], check=True, text=True, capture_output=True).stdout.strip()
    scene = tmp_path / "scene"
    scene.mkdir()
    for name in ("rgb.txt", "depth.txt", "groundtruth.txt"):
        (scene / name).write_text("ok\n", encoding="utf-8")
    digest = subprocess.run(
        ["bash", "-o", "pipefail", "-c", 'cd "$1"; LC_ALL=C find . -type f -printf "%P\\0" | LC_ALL=C sort -z | xargs -0 -r sha256sum | sha256sum', "bash", str(scene)],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.split()[0]
    descriptor = tmp_path / "descriptor.json"
    descriptor.write_text(json.dumps({
        "canonical_source": {"content_sha256": digest, "file_count": 3, "bytes": 9},
        "staged_source": {"path": str(scene), "content_sha256": digest, "file_count": 3, "bytes": 9},
    }), encoding="utf-8")
    params.update({
        "source_checkout": str(checkout),
        "source_commit": commit,
        "physical_source_path": str(scene),
        "canonical_source_path": str(scene),
        "input_storage_manifest_path": str(descriptor),
        "input_storage_manifest_sha256": hashlib.sha256(descriptor.read_bytes()).hexdigest(),
        "input_storage_content_digest": digest,
        "input_storage_file_count": 3,
        "input_storage_bytes": 9,
        "model_path": str(tmp_path / "new-model"),
        "host_name": subprocess.run(["/usr/bin/hostname"], check=True, text=True, capture_output=True).stdout.strip(),
    })
    command_dir = tmp_path / "bin"
    command_dir.mkdir()
    (command_dir / "rocminfo").write_text("#!/bin/sh\nprintf gfx1201\n", encoding="utf-8")
    sentinel = tmp_path / "docker-run"
    docker_ps = "exit 75" if docker_ps_fails else "printf splat_train_foreign\\n"
    (command_dir / "docker").write_text(
        f"#!/bin/sh\nif [ \"$1\" = ps ]; then {docker_ps}; else touch \"$DOCKER_SENTINEL\"; fi\n",
        encoding="utf-8",
    )
    for path in command_dir.iterdir():
        path.chmod(0o755)
    result = subprocess.run(
        _render(profile, params)[_render(profile, params).index("env") :],
        env={**os.environ, "PATH": f"{command_dir}:{os.environ['PATH']}", "DOCKER_SENTINEL": str(sentinel)},
        text=True,
        capture_output=True,
    )
    assert result.returncode != 0
    if docker_ps_fails:
        assert "foreign GPU mapper container is live" not in result.stderr
    else:
        assert "foreign GPU mapper container is live" in result.stderr
    assert not sentinel.exists()
