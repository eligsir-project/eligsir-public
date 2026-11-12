"""Task 5: per-dataset causal_replay run profiles (scannet, replica, scannetpp).

Mirrors tests/test_cli_config.py's test_profile_roundtrip_causal_replay pattern
for the tum_rgbd profile.
"""
from __future__ import annotations

import pytest

from eligsir.cli import config as cfg
from eligsir.cli.config import known_run_profiles


@pytest.mark.parametrize(
    ("profile", "resolution_tier", "thread_pinning_policy"),
    [
        ("causal_replay/scannet", "half", "pinned"),
        ("causal_replay/replica", "native", None),
        ("causal_replay/scannetpp", "half", None),
    ],
)
def test_dataset_profile_resolves_execution_trainer_and_dataset_settings(
    profile, resolution_tier, thread_pinning_policy
):
    merged = cfg.load_cli_config(run_profile=profile)
    assert merged.get("execution_trainer") == "causal_mapping"
    assert merged.get("streaming_ingestion_mode") == "dataset_fps"
    assert merged.get("streaming_frame_admission") == "hybrid_keyframe"

    # All three datasets now pin streaming_mapping_resolution_tier=half: the
    # `adaptive` tier inherited from live/canonical_mapping.yml is
    # unconditionally incompatible with the VECTORIZED_CAMERA_BATCH renderer
    # this trainer selects (mixed-resolution batches fail closed, see
    # replica.yml/scannetpp.yml's comments) -- a correctness requirement,
    # not a performance-tuned choice, for the two datasets without their own
    # measured A/B. Only scannet.yml's thread-pinning override is backed by
    # a real measurement.
    assert merged.get("streaming_mapping_resolution_tier") == resolution_tier

    if thread_pinning_policy is None:
        assert "causal_thread_pinning_policy" not in merged
    else:
        assert merged.get("causal_thread_pinning_policy") == thread_pinning_policy


@pytest.mark.parametrize(
    "profile",
    ["causal_replay/scannet", "causal_replay/replica", "causal_replay/scannetpp"],
)
def test_dataset_profile_does_not_set_decode_workers_or_payload_budget_policy(
    profile,
):
    """Both resolvers were measured null/unhelpful for throughput on the one
    dataset actually tested (ScanNet, PR #1933) -- none of the three profiles
    may set them away from their code defaults ("fixed" / "resolution_scaled").
    The key must be ABSENT from the merged config, not merely equal to the
    default, so a future change to the code default cannot silently drift.
    """
    merged = cfg.load_cli_config(run_profile=profile)
    assert "causal_decode_workers_policy" not in merged
    assert "streaming_payload_budget_policy" not in merged


def test_known_run_profiles_includes_new_dataset_profiles():
    profiles = known_run_profiles()
    assert "causal_replay/scannet" in profiles
    assert "causal_replay/replica" in profiles
    assert "causal_replay/scannetpp" in profiles
