"""Coverage registration for the canonical recorded Replica room0 source."""

from __future__ import annotations

import json
from pathlib import Path

from eligsir.utils.partition_eval import registry_n_source_frames


REPO_ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = REPO_ROOT / "eval" / "scenes.json"
SOURCE_PATH = Path(
    "/tmp/eligsir-qualification/"
    "replica_room0_full2000_source_v1"
)


def test_recorded_replica_room0_has_exact_coverage_denominator():
    registry = json.loads(REGISTRY_PATH.read_text())
    entry = registry["replica_room0_recorded"]

    assert Path(entry["path"]) == SOURCE_PATH
    assert entry["n_source_frames"] == 2_000
    assert entry["basename_fallback"] is False
    assert registry_n_source_frames(str(SOURCE_PATH)) == 2_000
    assert registry_n_source_frames("replica_room0_recorded") == 2_000


def test_recorded_replica_room0_does_not_claim_unregistered_same_leaf_source():
    assert registry_n_source_frames(f"/unregistered/{SOURCE_PATH.name}") is None
