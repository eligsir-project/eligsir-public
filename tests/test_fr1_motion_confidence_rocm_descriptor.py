"""Pre-Docker contract for the immutable FR1 R9700 staging descriptor."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tomllib
from pathlib import Path

import pytest
import yaml

from eligsir.utils.experiment_declaration import (
    DECLARATION_ENV,
    DECLARATION_SHA_ENV,
    ExperimentDeclarationError,
    canonical_declaration_json,
    declaration_sha256,
    load_experiment_context,
)


ROOT = Path(__file__).resolve().parents[1] / "experiments/researchflow"
PROFILE_PATH = ROOT / "profiles.fr1_motion_confidence_rocm_20260908.toml"
PROFILE_NAME = "fr1-motion-confidence-rocm"
SPEC_PATH = ROOT / "fr1_motion_confidence_control_rocm_20260908.yaml"
V8_SPECS = {
    "control": ROOT / "fr1_motion_confidence_control_rocm_20260908_v8.yaml",
    "treatment": ROOT / "fr1_motion_confidence_treatment_rocm_20260908_v8.yaml",
}


def _profile() -> dict:
    return tomllib.loads(PROFILE_PATH.read_text(encoding="utf-8"))["profiles"][PROFILE_NAME]


def _parameters() -> dict:
    return yaml.safe_load(SPEC_PATH.read_text(encoding="utf-8"))["run"]["parameters"]


def _v8_specs() -> dict[str, dict]:
    return {
        arm: yaml.safe_load(path.read_text(encoding="utf-8"))
        for arm, path in V8_SPECS.items()
    }


def _render(parameters: dict) -> list[str]:
    profile = _profile()
    assert set(parameters) == set(profile["allowed_parameters"])
    return [part.format(**parameters) for part in profile["argv"]]


def _launcher(parameters: dict) -> str:
    argv = _render(parameters)
    return argv[argv.index("-euc") + 1]


def _init_checkout(path: Path) -> str:
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    subprocess.run(["git", "-C", str(path), "config", "user.email", "test@example.invalid"], check=True)
    subprocess.run(["git", "-C", str(path), "config", "user.name", "Test"], check=True)
    (path / "README").write_text("test\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(path), "add", "README"], check=True)
    subprocess.run(["git", "-C", str(path), "commit", "-qm", "test"], check=True)
    return subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()


def _manifest_digest(scene: Path) -> str:
    """Run the immutable descriptor's declared manifest pipeline verbatim."""
    return subprocess.run(
        [
            "bash",
            "-o",
            "pipefail",
            "-c",
            'cd "$1"; LC_ALL=C find . -type f -print0 | LC_ALL=C sort -z | xargs -0 -r sha256sum | sha256sum',
            "bash",
            str(scene),
        ],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.split()[0]


def _flat_descriptor(*, scene: Path, canonical: str, digest: str, count: int, bytes_: int) -> dict:
    return {
        "schema": "eligsir.input-staging-descriptor/v1",
        "canonical_source": canonical,
        "canonical_manifest_sha256": digest,
        "canonical_file_count": count,
        "canonical_file_bytes": bytes_,
        "staged_path": str(scene),
        "staged_manifest_sha256": digest,
        "staged_file_count": count,
        "staged_file_bytes": bytes_,
        "storage_class": "tmpfs_ram",
    }


def _pre_docker_result(
    tmp_path: Path,
    *,
    descriptor_change: tuple[str, object] | None = None,
    mutate_staged_file: bool = False,
) -> tuple[subprocess.CompletedProcess[str], Path]:
    parameters = _parameters()
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    commit = _init_checkout(checkout)
    scene = tmp_path / "scene"
    scene.mkdir()
    for name in ("rgb.txt", "depth.txt", "groundtruth.txt"):
        (scene / name).write_text("ok\n", encoding="utf-8")
    (scene / "nested").mkdir()
    (scene / "nested" / "payload.bin").write_bytes(b"original\x00payload")
    (scene / ".COMPLETE").write_text("complete\n", encoding="utf-8")
    (scene / ".last_used").write_text("touched\n", encoding="utf-8")
    staged_files = [path for path in scene.rglob("*") if path.is_file()]
    digest = _manifest_digest(scene)
    descriptor_values = _flat_descriptor(
        scene=scene,
        canonical="canonical://fr1-test",
        digest=digest,
        count=len(staged_files),
        bytes_=sum(path.stat().st_size for path in staged_files),
    )
    if descriptor_change is not None:
        field, value = descriptor_change
        descriptor_values[field] = value
    if mutate_staged_file:
        (scene / "nested" / "payload.bin").write_bytes(b"changed\x00payload")
    descriptor = tmp_path / "descriptor.json"
    descriptor.write_text(json.dumps(descriptor_values), encoding="utf-8")
    probe = tmp_path / "probe.py"
    probe.write_text("# preflight placeholder\n", encoding="utf-8")
    commands = tmp_path / "commands"
    commands.mkdir()
    mapper = tmp_path / "mapper-reached"
    (commands / "rocminfo").write_text("#!/bin/sh\nprintf gfx1201\n", encoding="utf-8")
    (commands / "amd-smi").write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    (commands / "docker").write_text(
        "#!/bin/sh\n"
        "if [ \"$1\" = ps ]; then exit 0; fi\n"
        "if [ \"$1\" = image ]; then printf '%s\\n' \"$3\"; exit 0; fi\n"
        "case \" $* \" in *' --device /dev/kfd '*) touch \"$MAPPER_SENTINEL\";; esac\n"
        "exit 0\n",
        encoding="utf-8",
    )
    for command in commands.iterdir():
        command.chmod(0o755)
    kfd = tmp_path / "kfd"
    kfd.mkdir()
    parameters.update(
        {
            "source_checkout": str(checkout),
            "source_commit": commit,
            "physical_source_path": str(scene),
            "canonical_source_path": "canonical://fr1-test",
            "input_storage_manifest_path": str(descriptor),
            "input_storage_manifest_sha256": hashlib.sha256(descriptor.read_bytes()).hexdigest(),
            "input_storage_content_digest": digest,
            "input_storage_file_count": descriptor_values["staged_file_count"],
            "input_storage_bytes": descriptor_values["staged_file_bytes"],
            "cohort_probe_path": str(probe),
            "cohort_probe_sha256": hashlib.sha256(probe.read_bytes()).hexdigest(),
            "host_name": subprocess.run(
                ["/usr/bin/hostname"], check=True, text=True, capture_output=True
            ).stdout.strip(),
            "model_path": str(tmp_path / "model"),
        }
    )
    argv = _render(parameters)
    launcher_index = argv.index("-euc") + 1
    argv[launcher_index] = argv[launcher_index].replace("/sys/class/kfd/kfd/proc", str(kfd))
    result = subprocess.run(
        argv[argv.index("env") :],
        text=True,
        capture_output=True,
        env={
            **os.environ,
            "PATH": f"{commands}:{os.environ['PATH']}",
            "MAPPER_SENTINEL": str(mapper),
        },
        check=False,
    )
    return result, mapper


def test_rocm_profile_reaches_the_final_mapper_boundary_for_a_matching_flat_descriptor(tmp_path: Path) -> None:
    result, mapper = _pre_docker_result(tmp_path)
    assert result.returncode == 0, result.stderr
    assert mapper.is_file()


def test_rocm_v8_declarations_bind_sidecars_and_gradeable_consumer() -> None:
    specs = _v8_specs()
    declarations = {}
    for arm, spec in specs.items():
        path = V8_SPECS[arm]
        params = spec["run"]["parameters"]
        declaration = json.loads(path.with_suffix(".declaration.json").read_text())
        raw = canonical_declaration_json(declaration)
        declarations[arm] = declaration

        assert params["declaration_json"] == raw
        assert params["declaration_sha256"] == declaration_sha256(declaration)
        assert declaration["experiment_id"] == params["experiment_id"]
        assert declaration["arm"]["id"] == params["arm"]
        assert declaration["expected_evidence"]
        assert declaration["standalone_reason"].strip()
        assert "synthetic" not in declaration["standalone_reason"].lower()
        assert params["source_commit"] == "9cfb1da4b26ba160f6a119c5afed95ff17e35af6"
        assert params["input_storage_content_digest"] == (
            "88d0fa2702fe86fce2b9be94fedcf2e5fe4c614aff58aaaee2b89677c88372a6"
        )
        assert params["model_path"].endswith(f"_{arm}_9cfb1da4_r9700_v8_20260908")
        assert load_experiment_context(
            {
                DECLARATION_ENV: raw,
                DECLARATION_SHA_ENV: params["declaration_sha256"],
                "ABLATOR_JOB_ID": f"test-fr1-v8-{arm}",
            }
        )["gradeability"] == "GRADEABLE_DECLARED"
        rendered = _render(params)
        assert raw in rendered
        assert params["declaration_sha256"] in rendered
        rendered_text = "\n".join(rendered)
        for flag, value in (
            ("--streaming_optimization_batch_size", "16"),
            ("--streaming_optimization_camera_count_max", "16"),
            ("--causal_prepared_ready_b4", ""),
            ("--streaming_post_mapping_refinement_steps", "0"),
        ):
            assert flag in rendered_text
            if value:
                assert f"{flag}\n{value}" in rendered_text

        for field, error in (
            ("experiment_id", "experiment_id"),
            ("expected_evidence", "expected_evidence"),
            ("standalone_reason", "owner_refs or standalone_reason"),
        ):
            missing = dict(declaration)
            missing.pop(field)
            with pytest.raises(ExperimentDeclarationError, match=error):
                load_experiment_context(
                    {
                        DECLARATION_ENV: canonical_declaration_json(missing),
                        DECLARATION_SHA_ENV: declaration_sha256(missing),
                        "ABLATOR_JOB_ID": f"test-fr1-v8-{arm}-missing-{field}",
                    }
                )

    control = declarations["control"]
    treatment = declarations["treatment"]
    assert control["experiment_id"] == treatment["experiment_id"]
    assert control["fixed"] == treatment["fixed"]
    assert control["evaluation"] == treatment["evaluation"]
    assert control["motion_depth_confidence_mode"] == "identity"
    assert treatment["motion_depth_confidence_mode"] == "motion"
    assert control["arm"]["comparison_role"] == "control"
    assert treatment["arm"]["comparison_role"] == "treatment"


def test_rocm_profile_scopes_git_safe_directory_at_both_a0_container_boundaries() -> None:
    launcher = _launcher(_parameters())
    assert launcher.count('-e GIT_CONFIG_COUNT=1') == 2
    assert launcher.count('-e GIT_CONFIG_KEY_0=safe.directory') == 2
    assert launcher.count('-e GIT_CONFIG_VALUE_0=/workspace/eligsir') == 2
    assert 'git config --global' not in launcher


def test_rocm_profile_rejects_a_changed_staged_regular_file_before_mapper(tmp_path: Path) -> None:
    result, mapper = _pre_docker_result(tmp_path, mutate_staged_file=True)
    assert result.returncode == 65
    assert "staged source content digest mismatch" in result.stderr
    assert not mapper.exists()


@pytest.mark.parametrize(
    "field",
    ["staged_path", "staged_manifest_sha256", "staged_file_count", "staged_file_bytes"],
)
def test_rocm_profile_rejects_a_mismatched_flat_staging_field_before_mapper(
    tmp_path: Path, field: str
) -> None:
    result, mapper = _pre_docker_result(tmp_path, descriptor_change=(field, {
        "staged_path": "/wrong/staging/path",
        "staged_manifest_sha256": "0" * 64,
        "staged_file_count": 4,
        "staged_file_bytes": 10,
    }[field]))
    assert result.returncode != 0
    assert "AssertionError" in result.stderr
    assert not mapper.exists()
