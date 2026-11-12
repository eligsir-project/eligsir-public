"""Tests for the canonical scene replay-capsule registry (issue #634)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from eligsir.experiments.capsule_registry import (
    CapsuleRegistry,
    RegistryEntry,
    RegistryError,
    validate_registry_entry,
)
from eligsir.experiments.replay_capsule import (
    ArtifactRef,
    CapsuleManifest,
    CapsuleValidationError,
    EvaluationIdentity,
    ProducerInfo,
    SourceRef,
)


def test_registry_entry_to_dict():
    """RegistryEntry serializes to dict correctly."""
    entry = RegistryEntry(
        capsule_id="test_v1",
        capsule_hash="abc123" * 11 + "abc",
        status="DRAFT",
        pending_artifacts=("admission/decisions", "schedules/release_schedule"),
        blocked_on=("633",),
        notes="test entry",
    )
    data = entry.to_dict()
    assert data["capsule_id"] == "test_v1"
    assert data["status"] == "DRAFT"
    assert data["pending_artifacts"] == ["admission/decisions", "schedules/release_schedule"]
    assert data["blocked_on"] == ["633"]


def test_registry_entry_from_dict():
    """RegistryEntry deserializes from dict correctly."""
    data = {
        "capsule_id": "test_v1",
        "capsule_hash": "abc123" * 11 + "abc",
        "status": "DRAFT",
        "pending_artifacts": ["admission/decisions"],
        "blocked_on": ["633"],
    }
    entry = RegistryEntry.from_dict(data)
    assert entry.capsule_id == "test_v1"
    assert entry.status == "DRAFT"
    assert entry.pending_artifacts == ("admission/decisions",)
    assert entry.blocked_on == ("633",)


def _make_test_capsule(
    capsule_id: str = "test_v1",
    with_minimal_artifacts: bool = False,
) -> CapsuleManifest:
    """Create a minimal test CapsuleManifest."""
    artifacts = {}
    if with_minimal_artifacts:
        artifacts["source_frames"] = ArtifactRef(
            uri="artifact://test/source_frames.jsonl",
            content_hash="a" * 64,
            kind="id_list",
            row_count=100,
        )

    return CapsuleManifest(
        capsule_id=capsule_id,
        scene_id="test_scene",
        source=SourceRef(
            uri="datasets://test/data",
            content_hash="b" * 64,
            frame_count=100,
        ),
        lane_compatibility=("MAPPER_CORE",),
        frontend_trace_mode="FROZEN_FINAL_POSES",
        admission_trace_mode="FROZEN",
        evaluation=EvaluationIdentity(
            holdout_pool_hash="c" * 64,
            scored_view_hash="d" * 64,
        ),
        producer=ProducerInfo(
            git_commit="abc1234567890def1234567890def1234567890def1234567890def1234567890",
            tool_version="0.1.0",
        ),
        artifacts=artifacts,
    )


def test_validate_registry_entry_success():
    """validate_registry_entry accepts a valid entry."""
    capsule = _make_test_capsule()
    entry = RegistryEntry(
        capsule_id="test_v1",
        capsule_hash=capsule.capsule_hash(),
        status="DRAFT",
    )
    # Should not raise.
    validate_registry_entry(entry, capsule)


def test_validate_registry_entry_hash_mismatch():
    """validate_registry_entry refuses hash mismatch."""
    capsule = _make_test_capsule()
    entry = RegistryEntry(
        capsule_id="test_v1",
        capsule_hash="wrong" * 16,
        status="DRAFT",
    )
    with pytest.raises(RegistryError, match="capsule_hash mismatch"):
        validate_registry_entry(entry, capsule)


def test_validate_registry_entry_id_mismatch():
    """validate_registry_entry refuses ID mismatch."""
    capsule = _make_test_capsule(capsule_id="test_v1")
    entry = RegistryEntry(
        capsule_id="different_id",
        capsule_hash=capsule.capsule_hash(),
        status="DRAFT",
    )
    with pytest.raises(RegistryError, match="does not match"):
        validate_registry_entry(entry, capsule)


def test_validate_registry_entry_invalid_status():
    """validate_registry_entry refuses invalid status."""
    capsule = _make_test_capsule()
    entry = RegistryEntry(
        capsule_id="test_v1",
        capsule_hash=capsule.capsule_hash(),
        status="INVALID_STATUS",
    )
    with pytest.raises(RegistryError, match="status must be one of"):
        validate_registry_entry(entry, capsule)


def test_validate_registry_entry_retired_missing_fields():
    """validate_registry_entry refuses RETIRED without replaced_by/reason."""
    capsule = _make_test_capsule()
    entry = RegistryEntry(
        capsule_id="test_v1",
        capsule_hash=capsule.capsule_hash(),
        status="RETIRED",
        replaced_by=None,  # missing
        retirement_reason="",  # missing
    )
    with pytest.raises(RegistryError, match="must record both replaced_by"):
        validate_registry_entry(entry, capsule)


def test_validate_registry_entry_validated_with_pending():
    """validate_registry_entry refuses VALIDATED with pending artifacts."""
    capsule = _make_test_capsule(with_minimal_artifacts=True)
    entry = RegistryEntry(
        capsule_id="test_v1",
        capsule_hash=capsule.capsule_hash(),
        status="VALIDATED",
        pending_artifacts=("some_artifact",),
    )
    with pytest.raises(RegistryError, match="only allowed for DRAFT"):
        validate_registry_entry(entry, capsule)


def test_capsule_registry_load_empty(tmp_path):
    """CapsuleRegistry loads an empty directory without error."""
    registry = CapsuleRegistry()
    registry.load_directory(tmp_path)
    assert registry.list_capsules() == ()


def test_capsule_registry_load_one(tmp_path):
    """CapsuleRegistry loads a single DRAFT capsule."""
    # Create a capsule directory.
    capsule_dir = tmp_path / "test_v1"
    capsule_dir.mkdir()

    capsule = _make_test_capsule("test_v1")
    hash_val = capsule.capsule_hash()

    # Write manifest and entry.
    with open(capsule_dir / "manifest.json", "w") as f:
        json.dump(capsule.to_dict(), f)
    with open(capsule_dir / "registry_entry.json", "w") as f:
        json.dump(
            {
                "capsule_id": "test_v1",
                "capsule_hash": hash_val,
                "status": "DRAFT",
            },
            f,
        )

    # Load and verify.
    registry = CapsuleRegistry()
    registry.load_directory(tmp_path)
    assert registry.list_capsules() == ("test_v1",)
    assert registry.get_capsule("test_v1") == capsule
    assert registry.get_entry("test_v1").status == "DRAFT"


def test_capsule_registry_immutability(tmp_path):
    """CapsuleRegistry refuses a re-registration with different hash."""
    capsule_dir = tmp_path / "test_v1"
    capsule_dir.mkdir()

    capsule = _make_test_capsule("test_v1")
    hash_val = capsule.capsule_hash()

    # Write original.
    with open(capsule_dir / "manifest.json", "w") as f:
        json.dump(capsule.to_dict(), f)
    with open(capsule_dir / "registry_entry.json", "w") as f:
        json.dump(
            {
                "capsule_id": "test_v1",
                "capsule_hash": hash_val,
                "status": "DRAFT",
            },
            f,
        )

    registry = CapsuleRegistry()
    registry.load_directory(tmp_path)

    # Try to re-register with a different hash.
    capsule_dir_2 = tmp_path / "test_v1_dup"
    capsule_dir_2.mkdir()
    with open(capsule_dir_2 / "manifest.json", "w") as f:
        json.dump(capsule.to_dict(), f)
    with open(capsule_dir_2 / "registry_entry.json", "w") as f:
        json.dump(
            {
                "capsule_id": "test_v1",  # same ID
                "capsule_hash": "wrong" * 16,  # different hash
                "status": "DRAFT",
            },
            f,
        )

    with pytest.raises(RegistryError, match="capsule_hash mismatch"):
        registry.load_directory(tmp_path)


def test_capsule_registry_missing_files(tmp_path):
    """CapsuleRegistry refuses a capsule missing manifest or entry."""
    capsule_dir = tmp_path / "test_v1"
    capsule_dir.mkdir()

    # Write only manifest, missing entry.
    capsule = _make_test_capsule("test_v1")
    with open(capsule_dir / "manifest.json", "w") as f:
        json.dump(capsule.to_dict(), f)

    registry = CapsuleRegistry()
    with pytest.raises(RegistryError, match="missing registry_entry.json"):
        registry.load_directory(tmp_path)


def test_capsule_registry_list_by_status(tmp_path):
    """CapsuleRegistry.list_capsules() filters by status."""
    # Create two DRAFT capsules.
    for i in range(1, 3):
        capsule_dir = tmp_path / f"test_v{i}"
        capsule_dir.mkdir()
        capsule = _make_test_capsule(f"test_v{i}")
        hash_val = capsule.capsule_hash()
        with open(capsule_dir / "manifest.json", "w") as f:
            json.dump(capsule.to_dict(), f)
        with open(capsule_dir / "registry_entry.json", "w") as f:
            json.dump(
                {
                    "capsule_id": f"test_v{i}",
                    "capsule_hash": hash_val,
                    "status": "DRAFT",
                },
                f,
            )

    registry = CapsuleRegistry()
    registry.load_directory(tmp_path)
    assert registry.list_capsules(status="DRAFT") == ("test_v1", "test_v2")
    assert registry.list_capsules(status="VALIDATED") == ()
    assert len(registry.list_capsules(status=None)) == 2


def test_capsule_registry_get_nonexistent():
    """CapsuleRegistry returns None for unregistered capsules."""
    registry = CapsuleRegistry()
    assert registry.get_capsule("nonexistent") is None
    assert registry.get_entry("nonexistent") is None


def test_registry_entry_immutability_notes_dont_count(tmp_path):
    """Changing non-identity fields (notes) does NOT change immutability."""
    capsule_dir = tmp_path / "test_v1"
    capsule_dir.mkdir()

    capsule = _make_test_capsule("test_v1")
    hash_val = capsule.capsule_hash()

    # Write first registration.
    with open(capsule_dir / "manifest.json", "w") as f:
        json.dump(capsule.to_dict(), f)
    with open(capsule_dir / "registry_entry.json", "w") as f:
        json.dump(
            {
                "capsule_id": "test_v1",
                "capsule_hash": hash_val,
                "status": "DRAFT",
                "notes": "original",
            },
            f,
        )

    registry = CapsuleRegistry()
    registry.load_directory(tmp_path)
    entry1 = registry.get_entry("test_v1")
    assert entry1.notes == "original"

    # Changing notes should NOT trigger immutability violation
    # (identity_payload excludes notes).
    capsule_dir_2 = tmp_path / "test_v1_dup"
    capsule_dir_2.mkdir()
    with open(capsule_dir_2 / "manifest.json", "w") as f:
        json.dump(capsule.to_dict(), f)
    with open(capsule_dir_2 / "registry_entry.json", "w") as f:
        json.dump(
            {
                "capsule_id": "test_v1",
                "capsule_hash": hash_val,  # same hash
                "status": "DRAFT",
                "notes": "updated notes",  # different, but doesn't affect identity
            },
            f,
        )

    # Should load without immutability error.
    registry.load_directory(tmp_path)


def test_registry_entry_lifecycle_draft_to_validated(tmp_path):
    """Registry entry can be updated from DRAFT to VALIDATED."""
    # Create a minimal-artifact capsule that can validate.
    capsule_dir = tmp_path / "test_v1"
    capsule_dir.mkdir()

    capsule = _make_test_capsule("test_v1", with_minimal_artifacts=True)
    hash_val = capsule.capsule_hash()

    # Write as VALIDATED (this should pass validate_registry_entry).
    with open(capsule_dir / "manifest.json", "w") as f:
        json.dump(capsule.to_dict(), f)
    with open(capsule_dir / "registry_entry.json", "w") as f:
        json.dump(
            {
                "capsule_id": "test_v1",
                "capsule_hash": hash_val,
                "status": "VALIDATED",
                "pending_artifacts": [],
            },
            f,
        )

    registry = CapsuleRegistry()
    registry.load_directory(tmp_path)
    assert registry.get_entry("test_v1").status == "VALIDATED"


def test_draft_capsule_integration(tmp_path):
    """Integration test: a DRAFT capsule loads and shows blocked status."""
    # This mimics what tum_fr3_mapper_core_v1 looks like.
    capsule_dir = tmp_path / "test_draft_v1"
    capsule_dir.mkdir()

    # Minimal manifest with placeholder hashes.
    manifest_data = {
        "schema_version": 1,
        "capsule_id": "test_draft_v1",
        "scene_id": "test",
        "source": {
            "uri": "datasets://test/data",
            "content_hash": "a" * 64,
            "frame_count": 100,
        },
        "lane_compatibility": ["MAPPER_CORE"],
        "frontend_trace_mode": "FROZEN_FINAL_POSES",
        "admission_trace_mode": "FROZEN",
        "evaluation": {
            "holdout_pool_hash": "b" * 64,
            "scored_view_hash": "c" * 64,
        },
        "producer": {
            "git_commit": "deadbeef" * 8,
            "tool_version": "0.1.0",
        },
        "artifacts": {
            "source_frames": {
                "uri": "artifact://test/frames.jsonl",
                "content_hash": "d" * 64,
                "kind": "id_list",
                "row_count": 100,
            },
        },
        "status": "ACTIVE",
    }

    with open(capsule_dir / "manifest.json", "w") as f:
        json.dump(manifest_data, f)

    capsule = CapsuleManifest.from_dict(manifest_data)
    hash_val = capsule.capsule_hash()

    entry_data = {
        "capsule_id": "test_draft_v1",
        "capsule_hash": hash_val,
        "status": "DRAFT",
        "pending_artifacts": ["admission/decisions", "schedules/release_schedule"],
        "blocked_on": ["633"],
    }

    with open(capsule_dir / "registry_entry.json", "w") as f:
        json.dump(entry_data, f)

    registry = CapsuleRegistry()
    registry.load_directory(tmp_path)

    entry = registry.get_entry("test_draft_v1")
    assert entry.status == "DRAFT"
    assert entry.pending_artifacts == ("admission/decisions", "schedules/release_schedule")
    assert entry.blocked_on == ("633",)
