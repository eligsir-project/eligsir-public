"""Frozen queue-priority contracts for replay-challenger ResearchFlow recipes."""

from __future__ import annotations

import tomllib
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1] / "experiments/researchflow"
PROFILE_PATH = ROOT / "profiles.replay_challengers_fr3_20260907.toml"
HIGH = "replay-challenger-fr3"
LOW = "replay-challenger-fr3-low"
ORIGINAL_SPECS = (
    "replay_challengers_fr3_20260907.yaml",
    "replay_challengers_fr3_20260907_scoring_only.yaml",
    "replay_challengers_fr3_20260907_trials_only.yaml",
    "replay_challengers_fr3_20260907_trials_demotion.yaml",
)
SEED2_SPECS = (
    "replay_challengers_fr3_baseline_seed2_20260907.yaml",
    "replay_challengers_fr3_scoring_only_seed2_20260907.yaml",
)


def test_low_replay_challenger_profile_changes_only_trusted_queue_lane() -> None:
    profiles = tomllib.loads(PROFILE_PATH.read_text())["profiles"]
    high = dict(profiles[HIGH])
    low = dict(profiles[LOW])

    assert high.pop("ablator_lane") == 3
    assert low.pop("ablator_lane") == 1
    assert high == low


def test_replay_challenger_specs_select_their_declared_trusted_priority() -> None:
    for name in ORIGINAL_SPECS:
        spec = yaml.safe_load((ROOT / name).read_text())
        assert spec["run"]["profile"] == HIGH
        assert spec["metadata"]["ablator_lane"] == 3
    for name in SEED2_SPECS:
        spec = yaml.safe_load((ROOT / name).read_text())
        assert spec["run"]["profile"] == LOW
        assert spec["metadata"]["ablator_lane"] == 1
