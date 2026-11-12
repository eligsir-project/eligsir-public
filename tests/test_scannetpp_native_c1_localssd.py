"""Rendered-contract checks for the reviewed ScanNet++ B16/C1 refresh pair."""

from __future__ import annotations

import hashlib
import shlex
import subprocess
import tomllib
from pathlib import Path

import yaml

from eligsir.utils.experiment_declaration import DECLARATION_ENV, DECLARATION_SHA_ENV, load_experiment_context


ROOT = Path(__file__).resolve().parents[1] / "experiments/researchflow"
PROJECT = ROOT.parents[1]
PROFILE_PATH = ROOT / "profiles.scannetpp_native_c1_adaptive_refresh_20260908.toml"
SPECS = {
    "quarter": ROOT / "scannetpp_native_c1_quarter_refresh_20260908.yaml",
    "eighth": ROOT / "scannetpp_native_c1_eighth_refresh_20260908.yaml",
}
SOURCE_CHECKOUT = "/var/tmp/eligsir-a2c2ed6411f88b4219ae3397ecf0be640b353945-scan-stage-20260908"
SOURCE_COMMIT = "a2c2ed6411f88b4219ae3397ecf0be640b353945"
# Evidence tests may be run only with an authorized image supplied by the
# operator. Do not embed a private registry reference in the export.
IMAGE = "eligsir:scannetpp-evidence@sha256:cb4ac82904086e28f3b7534127d3e37e630314cada7ebe2287a454568d486973"
PINNED_OVERLAY = "configs/profiles/evaluation/scannetpp_iphone_adaptive.yml"
PINNED_OVERLAY_SHA256 = "a5782f43fcd51bb71de98d3a0f182f072efb609587f90797875b7f744bb9faec"
LEASE_QUALIFICATION_SPEC = ROOT / "scannetpp_native_c1_quarter_lease_qualification_20260908_v2.yaml"
LEASE_QUALIFICATION_STAGE = "/var/tmp/eligsir-50f515b3114b81f0171e5131af8468e5c90cabaa-scan-c1-lease-stage-20260908"
LEASE_QUALIFICATION_COMMIT = "50f515b3114b81f0171e5131af8468e5c90cabaa"


def _resolved_training_args(launcher: str):
    from eligsir.cli.parser import parse_training_args

    command = launcher.split('"$image" train.py ', 1)[1]
    return parse_training_args(shlex.split(command))[0]


def test_scan_pair_renders_local_ssd_stage_with_native_b16_c1_contract() -> None:
    profile = tomllib.loads(PROFILE_PATH.read_text(encoding="utf-8"))["profiles"]["scannetpp-native-c1-adaptive"]
    for floor, path in SPECS.items():
        spec = yaml.safe_load(path.read_text(encoding="utf-8"))
        parameters = spec["run"]["parameters"]
        assert set(parameters) == set(profile["allowed_parameters"])
        assert parameters["source_checkout"] == SOURCE_CHECKOUT
        assert parameters["source_commit"] == SOURCE_COMMIT
        assert parameters["container_image"] == IMAGE
        assert parameters["resolution_floor"] == floor
        declaration = load_experiment_context(
            {
                DECLARATION_ENV: parameters["declaration_json"],
                DECLARATION_SHA_ENV: parameters["declaration_sha256"],
                "ABLATOR_JOB_ID": "scannetpp-native-c1-test",
            }
        )["experiment_declaration"]
        assert declaration["arm"]["id"] == f"scannetpp_native_c1_{floor}_refresh_a2_v5_localssd"
        assert declaration["experiment_id"] == spec["id"]
        assert declaration["deployment"] == {"source_checkout": SOURCE_CHECKOUT, "source_checkout_storage": "local_ssd"}
        assert declaration["fixed"]["native_source_service"] == "B4/C1"
        assert declaration["fixed"]["batch_size"] == 16
        assert declaration["fixed"]["camera_count"] == 1
        argv = [part.format_map(parameters) for part in profile["argv"]]
        launcher = argv[argv.index("-euc") + 1]
        assert subprocess.run(["bash", "-n"], input=launcher, text=True, capture_output=True).returncode == 0
        assert "--security-opt label=disable" in launcher
        assert "-e GIT_CONFIG_COUNT=1 -e GIT_CONFIG_KEY_0=safe.directory -e GIT_CONFIG_VALUE_0=/workspace/eligsir" in launcher
        assert "--causal_prepared_ready_b4" in launcher
        assert "--causal_prepared_ready_source_insertion" in launcher
        assert "--streaming_post_mapping_refinement_steps 0" in launcher
        resolved = _resolved_training_args(launcher)
        assert resolved.streaming_optimization_batch_size == 16
        assert resolved.streaming_optimization_camera_count_max == 1
        assert resolved.streaming_replay_target_views_per_s == 200


def test_scan_pair_final_config_exists_in_the_pinned_runtime() -> None:
    profile = tomllib.loads(PROFILE_PATH.read_text(encoding="utf-8"))["profiles"]["scannetpp-native-c1-adaptive"]
    for path in SPECS.values():
        parameters = yaml.safe_load(path.read_text(encoding="utf-8"))["run"]["parameters"]
        assert parameters["overlay"] == PINNED_OVERLAY
        assert hashlib.sha256((PROJECT / parameters["overlay"]).read_bytes()).hexdigest() == PINNED_OVERLAY_SHA256
        launcher = [part.format_map(parameters) for part in profile["argv"]]
        script = launcher[launcher.index("-euc") + 1]
        assert '--config "$overlay"' in script
        assert '--streaming_mapping_resolution_min_tier "$floor"' in script


def test_quarter_lease_qualification_fails_closed_on_missing_render_lease() -> None:
    profile = tomllib.loads(PROFILE_PATH.read_text(encoding="utf-8"))["profiles"]["scannetpp-native-c1-adaptive"]
    parameters = yaml.safe_load(LEASE_QUALIFICATION_SPEC.read_text(encoding="utf-8"))["run"]["parameters"]
    declaration = load_experiment_context(
        {
            DECLARATION_ENV: parameters["declaration_json"],
            DECLARATION_SHA_ENV: parameters["declaration_sha256"],
            "ABLATOR_JOB_ID": "scannetpp-native-c1-lease-test",
        }
    )["experiment_declaration"]

    assert parameters["source_checkout"] == LEASE_QUALIFICATION_STAGE
    assert parameters["source_commit"] == LEASE_QUALIFICATION_COMMIT
    assert parameters["resolution_floor"] == "quarter"
    assert declaration["arm"]["id"] == "scannetpp_native_c1_quarter_lease_50f515b3_localssd_v2"
    assert declaration["fixed"]["source_commit"] == LEASE_QUALIFICATION_COMMIT
    assert declaration["evaluation"]["scene_registration_gate"] == "reported_not_promoted"
    assert profile["cwd"] == "/tmp/rf725-fr3-region-frame-replay-20260908"
    assert profile["argv"][0] == "/opt/researchflow/.venv/bin/python3"

    launcher = [part.format_map(parameters) for part in profile["argv"]]
    script = launcher[launcher.index("-euc") + 1]
    terminal_contract = "\n".join(launcher)
    assert "render_reuse_unavailable:missing_render_lease" in terminal_contract
    assert "prepared_insertion_seeds_accepted" in terminal_contract
    assert "prepared_insertions_executed" in terminal_contract


def test_quarter_lease_qualification_declares_actual_source_and_first_service_receipts() -> None:
    profile = tomllib.loads(PROFILE_PATH.read_text(encoding="utf-8"))["profiles"]["scannetpp-native-c1-adaptive"]
    evidence = profile["evidence"]
    spec = yaml.safe_load(LEASE_QUALIFICATION_SPEC.read_text(encoding="utf-8"))

    assert "metric:source_service" not in evidence
    assert {
        "metric:source_ingress_offered",
        "metric:source_ingress_admitted",
        "metric:source_ingress_dropped",
        "metric:source_ingress_shutdown",
        "metric:first_service_commits",
        "metric:first_service_completions",
        "metric:first_service_debt",
        "metric:first_service_pending",
        "metric:first_service_in_flight",
    } <= set(evidence)
    expected = {
        ("source_ingress_offered", "source_ingress.ingress_frames_offered", "==", 7946),
        ("source_ingress_admitted", "source_ingress.ingress_frames_admitted", "==", 7946),
        ("source_ingress_dropped", "source_ingress.ingress_frames_dropped", "==", 0),
        ("source_ingress_shutdown", "source_ingress.ingress_shutdown_incomplete", "==", False),
        ("first_service_commits", "ready_drain.at_drain_end.admitted_commit_count", ">", 0),
        ("first_service_completions", "ready_drain.at_drain_end.first_supervision_completions", ">", 0),
        ("first_service_debt", "ready_drain.at_drain_end.first_supervision_debt", "==", 0),
        ("first_service_pending", "ready_drain.at_drain_end.fresh_pending", "==", 0),
        ("first_service_in_flight", "ready_drain.at_drain_end.in_flight", "==", 0),
    }
    actual = {
        (item["name"], item["path"], item["operator"], item["expected_value"])
        for item in spec["evidence"]["metrics"]
        if isinstance(item, dict) and item.get("name", "").startswith(("source_ingress_", "first_service_"))
    }
    assert actual == expected
