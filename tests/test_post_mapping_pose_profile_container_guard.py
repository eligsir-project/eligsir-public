"""Rendered container-occupancy guard for the diagnostic pose profile."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "experiments/researchflow/profiles.post_mapping_pose_profile_fr3_diagnostic_20260909.toml"


def _researchflow_source() -> Path:
    source = Path(os.environ.get("RESEARCHFLOW_SOURCE", ""))
    if not (source / "src/researchflow/profiles.py").is_file():
        pytest.skip("set RESEARCHFLOW_SOURCE to the ResearchFlow runtime source")
    return source


def _rendered_launcher(parameters: dict[str, str | int]) -> tuple[str, tuple[str, ...]]:
    source = _researchflow_source()
    sys.path.insert(0, str(source / "src"))
    try:
        from researchflow.profiles import load_profiles, render_profile

        profile = load_profiles(PROFILE)["post-mapping-pose-profile-fr3-diagnostic"]
        argv, _, _ = render_profile(
            profile,
            parameters,
        )
    finally:
        sys.path.pop(0)
    launcher_index = next(index for index, value in enumerate(argv) if value.startswith("source=$1"))
    return argv[launcher_index], argv[launcher_index + 1 :]


def _git_checkout(path: Path) -> str:
    path.mkdir()
    subprocess.run(["git", "init", "--quiet"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=path, check=True)
    (path / "pin").write_text("pinned\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=path, check=True)
    subprocess.run(["git", "commit", "--quiet", "-m", "pin"], cwd=path, check=True)
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=path, check=True, text=True, capture_output=True,
    ).stdout.strip()


@pytest.mark.parametrize(
    ("running_names", "allowed"),
    (("other_job", True), ("splat_train", False), ("splat_train_foreign", False)),
)
def test_rendered_container_guard_blocks_foreign_training_before_docker_run(
    tmp_path: Path, running_names: str, allowed: bool,
) -> None:
    source = tmp_path / "source"
    source_commit = _git_checkout(source)
    harness = tmp_path / "harness"
    harness_commit = _git_checkout(harness)
    scene = tmp_path / "scene"
    scene.mkdir()
    for name in ("rgb.txt", "depth.txt", "groundtruth.txt"):
        (scene / name).write_text("input\n", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    manifest_sha256 = subprocess.run(
        ["sha256sum", str(manifest)], check=True, text=True, capture_output=True,
    ).stdout.split()[0]
    launcher, argv = _rendered_launcher(
        {
            "arm": "guard",
            "model_path": str(tmp_path / "output"),
            "source_checkout": str(source),
            "source_commit": source_commit,
            "harness_checkout": str(harness),
            "harness_commit": harness_commit,
            "physical_source_path": str(scene),
            "canonical_source_path": "/canonical/fr3",
            "input_manifest_path": str(manifest),
            "input_manifest_sha256": manifest_sha256,
            "container_image": "image@sha256:test",
            "profile_mode": "trace",
            "prefix_frames": 3,
            "host_name": "test-host",
            "researchflow_telemetry_checkout": "/tmp/telemetry",
            "researchflow_telemetry_commit": "d" * 40,
            "declaration_json": "{}",
            "declaration_sha256": "0" * 64,
        },
    )
    tools = tmp_path / "tools"
    tools.mkdir()
    (tools / "hostname").write_text("#!/bin/sh\nprintf 'test-host\\n'\n", encoding="utf-8")
    (tools / "rocminfo").write_text("#!/bin/sh\nprintf 'gfx1201\\n'\n", encoding="utf-8")
    (tools / "docker").write_text(
        "#!/bin/sh\n"
        "case \"$1\" in\n"
        "  ps) if [ \"$3\" = '{{.Names}}' ]; then printf '%s\\n' \"$DOCKER_PS_NAMES\"; else printf '%s\\n' \"$3\"; fi;;\n"
        "  image) exit 0;;\n"
        "  run) touch \"$DOCKER_RUN_SENTINEL\";;\n"
        "  *) exit 64;;\n"
        "esac\n",
        encoding="utf-8",
    )
    for tool in tools.iterdir():
        tool.chmod(0o755)
    sentinel = tmp_path / "docker-run"
    result = subprocess.run(
        [
            "bash", "-euc", launcher, *argv,
        ],
        env={
            **os.environ,
            "PATH": f"{tools}:{os.environ['PATH']}",
            "DOCKER_PS_NAMES": running_names,
            "DOCKER_RUN_SENTINEL": str(sentinel),
        },
        text=True,
        capture_output=True,
        check=False,
    )

    assert (result.returncode == 0) is allowed, result.stderr
    assert sentinel.exists() is allowed


@pytest.mark.parametrize("arm", ("trace", "untraced"))
def test_declaration_is_rendered_and_reaches_the_pinned_consumer(arm: str) -> None:
    """Exercise the real RF renderer and immutable declaration reader."""
    source = _researchflow_source()
    spec = yaml.safe_load(
        (ROOT / "experiments" / "researchflow" / f"fr3_post_mapping_pose_profile_{arm}_20260909.yaml").read_text()
    )
    parameters = spec["run"]["parameters"]
    sys.path.insert(0, str(source / "src"))
    try:
        from researchflow.profiles import load_profiles, render_profile

        profile = load_profiles(PROFILE)["post-mapping-pose-profile-fr3-diagnostic"]
        argv, env, _ = render_profile(profile, parameters)
    finally:
        sys.path.pop(0)
    assert env["ABLATOR_EXPERIMENT_DECLARATION_JSON"] == parameters["declaration_json"]
    assert env["ABLATOR_EXPERIMENT_DECLARATION_SHA256"] == parameters["declaration_sha256"]
    launcher = next(item for item in argv if "exec docker run" in item)
    assert "-e ABLATOR_JOB_ID -e ABLATOR_EXPERIMENT_DECLARATION_JSON -e ABLATOR_EXPERIMENT_DECLARATION_SHA256" in launcher
    pinned = Path(os.environ.get("ELIGSIR_TASK3_PINNED_SOURCE_ROOT", "/tmp/task3-8064-cpu-fit"))
    if not (pinned / "utils" / "experiment_declaration.py").is_file():
        pytest.skip("Task3 immutable source stage is not mounted")
    result = subprocess.run(
        [sys.executable, "-c", "from eligsir.utils.experiment_declaration import load_experiment_context; import json; print(json.dumps(load_experiment_context()['experiment_declaration'], sort_keys=True))"],
        env={
            **os.environ,
            "PYTHONPATH": str(pinned),
            "ABLATOR_JOB_ID": "rf-declaration-contract",
            "ABLATOR_EXPERIMENT_DECLARATION_JSON": parameters["declaration_json"],
            "ABLATOR_EXPERIMENT_DECLARATION_SHA256": parameters["declaration_sha256"],
        },
        text=True, capture_output=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    declaration = yaml.safe_load(result.stdout)
    assert declaration["deployment"]["mapper_source_commit"] == "8064aca93886dd07ad6d890bccbdcda960a9a2af"
    assert declaration["deployment"]["harness_commit"] == "a5b57947e0d6c7f62e44567fbb82c29d4753f80d"
