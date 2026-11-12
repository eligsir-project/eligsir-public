"""Contract checks for the prospective current-runtime FR3 decomposition."""

import hashlib
import json
import os
from pathlib import Path
import tomllib

import yaml

from eligsir.cli.parser import parse_training_args


ROOT = Path(__file__).resolve().parents[1] / "experiments/researchflow"
PROFILE = ROOT / "profiles.fr3_region_frame_replay_20260908.toml"
SOURCE = "a79ca08787006b4267d38eb40ed936aa4a22c351"
STAGE = "/var/tmp/eligsir-a79ca08787006b4267d38eb40ed936aa4a22c351-hotcold-current-decomposition-20260908"
ARMS = (
    "hotcold_current_baseline",
    "hotcold_current_candidate_only",
    "hotcold_current_prior_aggressive",
    "hotcold_current_stronger_cadence",
)


def test_all_current_runtime_decomposition_specs_and_declarations_exist() -> None:
    for arm in ARMS:
        assert (ROOT / f"fr3_current_hotcold_decomposition_{arm}_20260908.yaml").is_file()
        assert (ROOT / f"fr3_current_hotcold_decomposition_{arm}_20260908.declaration.json").is_file()


def test_current_runtime_decomposition_has_only_declared_turnover_axes() -> None:
    profile = tomllib.loads(PROFILE.read_text())["profiles"]["fr3-region-frame-replay-r9700"]
    expected = {
        "hotcold_current_baseline": (0.0, 0, 0, 8),
        "hotcold_current_candidate_only": (0.75, 4, 0, 8),
        "hotcold_current_prior_aggressive": (0.75, 4, 1, 8),
        "hotcold_current_stronger_cadence": (0.75, 1, 1, 8),
    }
    fixed = None
    for arm in ARMS:
        spec = yaml.safe_load((ROOT / f"fr3_current_hotcold_decomposition_{arm}_20260908.yaml").read_text())
        params = spec["run"]["parameters"]
        declaration = json.loads((ROOT / f"fr3_current_hotcold_decomposition_{arm}_20260908.declaration.json").read_text())
        canonical = json.dumps(declaration, sort_keys=True, separators=(",", ":"))
        rendered = [part.format_map(params) for part in profile["argv"]]
        mapper = rendered[rendered.index("--tum_sequence") :]
        args = parse_training_args([*mapper, *profile["arm_argv"][arm]])[0]
        assert params["source_commit"] == declaration["fixed"]["source_commit"] == SOURCE
        assert params["source_checkout"] == declaration["deployment"]["source_checkout"] == STAGE
        assert params["declaration_json"] == canonical
        assert params["declaration_sha256"] == hashlib.sha256(canonical.encode()).hexdigest()
        assert (args.streaming_replay_challenger_fraction, args.streaming_replay_challenger_demotion_cadence, args.streaming_replay_challenger_demotion_limit, args.streaming_replay_challenger_lease_ticks) == expected[arm]
        assert args.streaming_spatial_replay_focus is False
        assert args.streaming_historical_usefulness_tracking is True
        assert args.streaming_optimization_batch_size == args.streaming_optimization_camera_count_max == 16
        assert args.streaming_mapping_resolution_tier == "adaptive"
        assert args.streaming_post_mapping_refinement_steps == 0
        acceptance = {item["telemetry"]: (item["op"], item["value"]) for item in spec["acceptance"]["positive"]["all"] if "telemetry" in item}
        if arm == "hotcold_current_candidate_only":
            assert acceptance == {
                "replay_challenger.replay_challenger_trials_authorized": (">", 0),
                "replay_challenger.replay_challenger_trials_completed": (">", 0),
                "replay_challenger.replay_challenger_demotions": ("==", 0),
            }
        fixed_axes = (params["physical_source_path"], params["canonical_source_path"], params["input_storage_content_digest"], params["cohort_hash"], params["seed"], params["researchflow_git_sha"], params["researchflow_telemetry_commit"])
        if fixed is None:
            fixed = fixed_axes
        assert fixed_axes == fixed


def test_rf725_compiles_each_current_decomposition_arm() -> None:
    root = os.environ.get("FR3_CURRENT_RF725_PLAN_ROOT")
    if not root:
        return
    for arm in ARMS:
        plan = json.loads((Path(root) / f"{arm}.json").read_text())
        job = plan["jobs"]["run"]
        assert plan["automation"] == "manual"
        assert plan["blocked"] is False and plan["blocked_reasons"] == []
        assert job["parameters"]["source_commit"] == SOURCE
        assert job["parameters"]["source_checkout"] == STAGE
        assert job["parameters"]["arm"] == arm
        assert job["experiment_declaration"]["arm"]["id"] == arm
        assert "--streaming_optimization_batch_size" in job["argv"]
