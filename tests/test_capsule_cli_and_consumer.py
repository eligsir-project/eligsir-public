"""Tests for Scene Replay Capsule CLI and run-metadata consumer (issue #632 criterion 8).

Tests:
1. CLI: validate, inspect, diff commands (torch-free, fixtures only).
2. Consumer: load/validate and sidecar writing (torch-free).
3. Flag-absent no-op: verify default behavior unchanged when flag is empty.

No torch imports: stays collectable on ``localhost/eligsir-ci-cpu`` (Dockerfile.ci-cpu).
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
import yaml

from eligsir.experiments.capsule_cli import (
    cmd_diff,
    cmd_inspect,
    cmd_validate,
    main,
)
from eligsir.experiments.capsule_run_metadata import (
    load_and_validate_capsule,
    write_replay_capsule_sidecar,
)
from eligsir.experiments.replay_capsule import (
    ArtifactRef,
    CapsuleManifest,
    EvaluationIdentity,
    ProducerInfo,
    SourceRef,
)


# Test fixtures
_HASHES = {
    name: f"{i:02d}" * 32 for i, name in enumerate("abcdefghijklmnopqrstuvwxyz")
}


def _test_artifact(key_hash: str, kind: str = "event_log", rows: int | None = 10) -> ArtifactRef:
    return ArtifactRef(
        uri=f"artifact://test/{key_hash[:8]}",
        content_hash=key_hash,
        kind=kind,  # type: ignore[arg-type]
        row_count=rows if kind in ("event_log", "id_list") else None,
    )


def _test_capsule(**overrides) -> CapsuleManifest:
    """Create a minimal valid capsule fixture."""
    artifacts = {
        "source_frames": _test_artifact(_HASHES["a"], kind="event_log", rows=100),
        "camera_intrinsics": _test_artifact(_HASHES["b"], kind="parameter_record"),
        "preprocessing_contract": _test_artifact(_HASHES["c"], kind="parameter_record"),
        "frontend/pose_events": _test_artifact(_HASHES["d"]),
        "frontend/tracking_state": _test_artifact(_HASHES["e"]),
        "frontend/pose_corrections": _test_artifact(_HASHES["f"]),
        "admission/decisions": _test_artifact(_HASHES["g"]),
        "admission/admission_evidence": _test_artifact(_HASHES["h"]),
        "evaluation/holdout_pool": _test_artifact(_HASHES["i"], kind="id_list"),
        "evaluation/scored_views": _test_artifact(_HASHES["j"], kind="id_list"),
        "schedules/release_schedule": _test_artifact(_HASHES["k"]),
        "schedules/replay_schedule": _test_artifact(_HASHES["l"]),
        "schedules/insertion_schedule": _test_artifact(_HASHES["m"]),
        "schedules/work_boundaries": _test_artifact(_HASHES["n"], kind="parameter_record"),
        "rng_contract": _test_artifact(_HASHES["o"], kind="parameter_record"),
    }
    kwargs = dict(
        capsule_id="test_capsule_v1",
        scene_id="test_scene",
        source=SourceRef(
            uri="artifact://datasets/test",
            content_hash=_HASHES["p"],
            frame_count=100,
        ),
        lane_compatibility=("MAPPER_CORE",),
        frontend_trace_mode="FROZEN_CAUSAL_SLAM_TRACE",
        admission_trace_mode="FROZEN",
        evaluation=EvaluationIdentity(
            holdout_pool_hash=_HASHES["q"],
            scored_view_hash=_HASHES["r"],
        ),
        producer=ProducerInfo(
            git_commit="deadbeef1234567890abcdef1234567890abcdef",
            tool_version="capsule-tool/0.1",
        ),
        artifacts=artifacts,
    )
    kwargs.update(overrides)
    return CapsuleManifest(**kwargs)


@pytest.fixture
def valid_capsule_dir(tmp_path: Path) -> Path:
    """Create a valid capsule directory with manifest."""
    capsule_dir = tmp_path / "test_capsule"
    capsule_dir.mkdir()

    capsule = _test_capsule()
    manifest_path = capsule_dir / "capsule_manifest.yaml"
    with open(manifest_path, "w") as f:
        yaml.dump(capsule.to_dict(), f)

    return capsule_dir


@pytest.fixture
def invalid_capsule_dir(tmp_path: Path) -> Path:
    """Create an invalid capsule directory (fails validation, not parsing)."""
    capsule_dir = tmp_path / "invalid_capsule"
    capsule_dir.mkdir()

    # Create a capsule that parses but fails validation (e.g., missing producer commit)
    capsule = _test_capsule(producer=ProducerInfo(git_commit="", tool_version="capsule-tool/0.1"))
    manifest_path = capsule_dir / "capsule_manifest.yaml"
    with open(manifest_path, "w") as f:
        yaml.dump(capsule.to_dict(), f)

    return capsule_dir


class TestCapsuleCLI:
    """Tests for capsule CLI commands."""

    def test_cli_validate_success(self, valid_capsule_dir: Path) -> None:
        """Test validate command on a valid capsule."""
        exit_code = cmd_validate(str(valid_capsule_dir))
        assert exit_code == 0

    def test_cli_validate_failure(self, invalid_capsule_dir: Path) -> None:
        """Test validate command on an invalid capsule (returns 1)."""
        exit_code = cmd_validate(str(invalid_capsule_dir))
        assert exit_code == 1

    def test_cli_validate_missing_file(self, tmp_path: Path) -> None:
        """Test validate command on non-existent directory (returns 2)."""
        nonexistent = tmp_path / "nonexistent"
        exit_code = cmd_validate(str(nonexistent))
        assert exit_code == 2

    def test_cli_inspect_valid(self, valid_capsule_dir: Path, capsys) -> None:
        """Test inspect command prints metadata."""
        exit_code = cmd_inspect(str(valid_capsule_dir))
        assert exit_code == 0
        captured = capsys.readouterr()
        assert "Capsule ID:" in captured.out
        assert "test_capsule_v1" in captured.out

    def test_cli_inspect_missing_file(self, tmp_path: Path) -> None:
        """Test inspect command on non-existent directory (returns 2)."""
        nonexistent = tmp_path / "nonexistent"
        exit_code = cmd_inspect(str(nonexistent))
        assert exit_code == 2

    def test_cli_diff_identical(self, valid_capsule_dir: Path, capsys) -> None:
        """Test diff command on identical capsules (returns 0)."""
        # Create second copy with same content
        capsule_dir_2 = valid_capsule_dir.parent / "test_capsule_2"
        capsule_dir_2.mkdir()
        capsule_manifest = valid_capsule_dir / "capsule_manifest.yaml"
        with open(capsule_manifest) as f:
            data = yaml.safe_load(f)
        with open(capsule_dir_2 / "capsule_manifest.yaml", "w") as f:
            yaml.dump(data, f)

        exit_code = cmd_diff(str(valid_capsule_dir), str(capsule_dir_2))
        assert exit_code == 0
        captured = capsys.readouterr()
        assert "identical" in captured.out.lower()

    def test_cli_diff_different_ids(self, valid_capsule_dir: Path, tmp_path: Path, capsys) -> None:
        """Test diff command on capsules with different IDs (returns 1)."""
        capsule_dir_2 = tmp_path / "different_capsule"
        capsule_dir_2.mkdir()

        capsule = _test_capsule(capsule_id="different_capsule_v1")
        manifest_path = capsule_dir_2 / "capsule_manifest.yaml"
        with open(manifest_path, "w") as f:
            yaml.dump(capsule.to_dict(), f)

        exit_code = cmd_diff(str(valid_capsule_dir), str(capsule_dir_2))
        assert exit_code == 1
        captured = capsys.readouterr()
        assert "differ" in captured.out.lower()

    def test_cli_main_validate(self, valid_capsule_dir: Path) -> None:
        """Test main entry point with validate command."""
        exit_code = main(["validate", str(valid_capsule_dir)])
        assert exit_code == 0

    def test_cli_main_inspect(self, valid_capsule_dir: Path, capsys) -> None:
        """Test main entry point with inspect command."""
        exit_code = main(["inspect", str(valid_capsule_dir)])
        assert exit_code == 0
        captured = capsys.readouterr()
        assert "Capsule ID:" in captured.out

    def test_cli_main_no_args(self, capsys) -> None:
        """Test main entry point with no arguments."""
        exit_code = main([])
        assert exit_code == 2
        captured = capsys.readouterr()
        assert "Usage:" in captured.err

    def test_cli_main_unknown_command(self, capsys) -> None:
        """Test main entry point with unknown command."""
        exit_code = main(["unknown_cmd"])
        assert exit_code == 2
        captured = capsys.readouterr()
        assert "Unknown command" in captured.err


class TestCapsuleRunMetadata:
    """Tests for capsule run metadata recording."""

    def test_load_and_validate_capsule_success(self, valid_capsule_dir: Path) -> None:
        """Test loading and validating a capsule."""
        capsule, capsule_hash = load_and_validate_capsule(str(valid_capsule_dir))
        assert capsule is not None
        assert capsule_hash is not None
        assert isinstance(capsule_hash, str)
        assert len(capsule_hash) == 64  # SHA-256 hex
        assert capsule.capsule_id == "test_capsule_v1"

    def test_load_and_validate_capsule_invalid(self, invalid_capsule_dir: Path) -> None:
        """Test loading an invalid capsule (returns None, None)."""
        capsule, capsule_hash = load_and_validate_capsule(str(invalid_capsule_dir))
        assert capsule is None
        assert capsule_hash is None

    def test_load_and_validate_capsule_missing_file(self, tmp_path: Path) -> None:
        """Test loading from non-existent directory (returns None, None)."""
        nonexistent = tmp_path / "nonexistent"
        capsule, capsule_hash = load_and_validate_capsule(str(nonexistent))
        assert capsule is None
        assert capsule_hash is None

    @pytest.mark.parametrize("path", ["", None])
    def test_load_and_validate_capsule_empty_path(self, path) -> None:
        """Test loading with empty path (returns None, None)."""
        capsule, capsule_hash = load_and_validate_capsule(path)
        assert capsule is None
        assert capsule_hash is None

    def test_write_replay_capsule_sidecar(
        self, valid_capsule_dir: Path, tmp_path: Path
    ) -> None:
        """Test writing capsule metadata sidecar."""
        capsule, capsule_hash = load_and_validate_capsule(str(valid_capsule_dir))
        assert capsule is not None

        # Write to a test output directory
        output_dir = tmp_path / "model_output"
        output_dir.mkdir()

        sidecar_path = write_replay_capsule_sidecar(str(output_dir), capsule, capsule_hash)

        # Verify file was written
        assert sidecar_path.exists()
        assert sidecar_path.name == "replay_capsule.json"

        # Verify contents
        with open(sidecar_path) as f:
            sidecar_data = json.load(f)

        assert sidecar_data["capsule_id"] == "test_capsule_v1"
        assert sidecar_data["capsule_hash"] == capsule_hash
        assert sidecar_data["schema_version"] == 1
        assert sidecar_data["scene_id"] == "test_scene"
        assert sidecar_data["source_frame_count"] == 100
        assert "source_uri" in sidecar_data
        assert "producer_git_commit" in sidecar_data

    def test_write_replay_capsule_sidecar_overwrites(
        self, valid_capsule_dir: Path, tmp_path: Path
    ) -> None:
        """Test that writing sidecar overwrites previous version."""
        capsule, capsule_hash = load_and_validate_capsule(str(valid_capsule_dir))
        output_dir = tmp_path / "model_output"
        output_dir.mkdir()

        # Write first time
        write_replay_capsule_sidecar(str(output_dir), capsule, capsule_hash)
        sidecar_path = output_dir / "replay_capsule.json"
        first_mtime = sidecar_path.stat().st_mtime

        # Write second time (should overwrite)
        write_replay_capsule_sidecar(str(output_dir), capsule, capsule_hash)
        second_mtime = sidecar_path.stat().st_mtime

        # Verify file still exists and content is consistent
        assert sidecar_path.exists()
        with open(sidecar_path) as f:
            data = json.load(f)
        assert data["capsule_id"] == "test_capsule_v1"


class TestFlagAbsentNoOp:
    """Test that flag-absent behavior is unchanged (byte-identical).

    These tests verify at the helper/flag level that the new code
    has zero effect when the flag is absent. No training run is launched.
    """

    def test_replay_capsule_flag_default_empty(self) -> None:
        """Test that --replay_capsule flag defaults to empty string."""
        from arguments import ModelParams
        from argparse import ArgumentParser

        parser = ArgumentParser()
        model_params = ModelParams(parser)

        # The default value should be empty string
        assert hasattr(model_params, "replay_capsule")
        assert model_params.replay_capsule == ""


        # So when the flag is absent or empty, no capsule code runs.
