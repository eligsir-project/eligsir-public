"""Regression test for the producer side of issue #891/#784's git-provenance fix.

``utils.git_metadata`` reads ``ELIGSIR_GIT_COMMIT`` /
``ELIGSIR_GIT_BRANCH`` from the environment when a bind-mounted linked
worktree cannot resolve its own git metadata from inside a container.

After the ROS2 hard cut this repository owns exactly one Compose training
service: ``train``. ROS2 deployment services and their provenance forwarding
are owned and tested by ``eligsir-project/eligsir-ros2``.

This module still checks legacy human launch scripts while they remain in the
core tree during the staged migration. Once their ROS-owned replacements land,
those scripts and the corresponding assertions move together rather than
leaving a compatibility launcher in core.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

REPO_ROOT = Path(__file__).resolve().parent.parent
COMPOSE_PATH = REPO_ROOT / "docker-compose.yml"
LAUNCH_SCRIPTS_DIR = REPO_ROOT / "human_launch_scripts"

ENV_COMMIT_VAR = "ELIGSIR_GIT_COMMIT"
ENV_BRANCH_VAR = "ELIGSIR_GIT_BRANCH"

# ROS-owned launchers are fail-closed boundary stubs and therefore do not
# launch containers or forward core provenance. Only active core training
# launchers belong in this list.
TRAINING_LAUNCHER_SCRIPTS = [
    "08_part_a_offline_parity_ablation.sh",
    "09_part_bcd_floater_spatial_ablation.sh",
    "10_bcd_ablation_param.sh",
    "11_noise_schedule_sweep.sh",
    "12_dereg_vs_default_fr3.sh",
    "13_spf_vs_coverage_fr3.sh",
    "14_lever1_lever3_fr3.sh",
    "15_lever_ablation_fr3.sh",
    "16_stride_kf_fr3.sh",
    "17_kf_coverage_proof_fr3.sh",
    "18_floor_search_fr3.sh",
    "19_tum_fr3_streaming_rviz_webvnc.sh",
]


def _load_compose() -> dict:
    with COMPOSE_PATH.open() as fh:
        return yaml.safe_load(fh)


def test_core_train_service_declares_git_provenance_env():
    """The core service must explicitly forward both provenance variables."""
    compose = _load_compose()
    services = compose.get("services", {})
    assert set(services) == {"train"}, (
        "core docker-compose.yml must own only the ROS-free train service; "
        f"found {sorted(services)}"
    )
    env = services["train"].get("environment") or {}
    assert ENV_COMMIT_VAR in env, (
        f"train environment is missing {ENV_COMMIT_VAR} -- "
        "the git-provenance env passthrough (issue #891/#784) regressed."
    )
    assert ENV_BRANCH_VAR in env, (
        f"train environment is missing {ENV_BRANCH_VAR} -- "
        "the git-provenance env passthrough (issue #891/#784) regressed."
    )


def _forwarding_pattern(var: str) -> re.Pattern:
    """Match a self-referential forwarding assignment such as ``VAR=$VAR``."""
    return re.compile(rf'{var}=\\?"?\$\{{?{var}\b')


@pytest.mark.parametrize("script_name", TRAINING_LAUNCHER_SCRIPTS)
def test_launch_script_forwards_git_provenance_env(script_name):
    """Each retained launcher must resolve and forward real provenance values."""
    script_path = LAUNCH_SCRIPTS_DIR / script_name
    assert script_path.exists(), f"expected launch script not found: {script_path}"
    text = script_path.read_text()

    assert "resolve_git_provenance" in text, (
        f"{script_name} launches a training container but never calls "
        "resolve_git_provenance (issue #891/#784 regressed)."
    )
    assert _forwarding_pattern(ENV_COMMIT_VAR).search(text), (
        f"{script_name} does not forward {ENV_COMMIT_VAR} into its container launch."
    )
    assert _forwarding_pattern(ENV_BRANCH_VAR).search(text), (
        f"{script_name} does not forward {ENV_BRANCH_VAR} into its container launch."
    )


def test_git_provenance_helper_exists_and_is_shaped_correctly():
    """The shared resolver must exist while retained launchers depend on it."""
    helper_path = LAUNCH_SCRIPTS_DIR / "_git_provenance.sh"
    assert helper_path.exists()
    text = helper_path.read_text()
    assert "resolve_git_provenance" in text
    assert f"export {ENV_COMMIT_VAR}" in text
    assert f"export {ENV_BRANCH_VAR}" in text
