"""Sidecar metadata recording for runs using a Scene Replay Capsule.

Issue #632 criterion 8: A real consumer that records the capsule ID + content hash
into a run's own metadata, so #203's experiment-validity class and #224/#186's
baselines can key off it.

Torch-free: stays collectable under CI-CPU (Dockerfile.ci-cpu).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from eligsir.experiments.replay_capsule import CapsuleManifest, CapsuleValidationError


def load_and_validate_capsule(
    capsule_dir: str | Path,
) -> tuple[CapsuleManifest, str] | tuple[None, None]:
    """Load and validate a capsule, returning (manifest, hash) or (None, None) on failure.

    Validation failure is non-fatal here (returns None): the caller decides whether
    to raise or warn. This keeps the function's contract simple: succeed or return None.

    Args:
        capsule_dir: Path to capsule directory containing capsule_manifest.yaml.

    Returns:
        (capsule_manifest, capsule_hash_str) on success.
        (None, None) if capsule_dir is None/empty, file not found, or validation fails.

    Raises:
        ValueError: if YAML parsing fails (a parse error is distinct from validation).
    """
    import yaml

    if not capsule_dir:
        return None, None

    capsule_path = Path(capsule_dir)
    manifest_file = capsule_path / "capsule_manifest.yaml"

    if not manifest_file.exists():
        return None, None

    with open(manifest_file) as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"capsule_manifest.yaml must be a YAML dict, got {type(data)}")

    # Parse to object (may raise KeyError if required fields missing)
    try:
        capsule = CapsuleManifest.from_dict(data)
    except (KeyError, TypeError) as e:
        # Parsing failed, not a validation error
        return None, None

    # Load checksums sidecar if present
    checksums_path = capsule_path / "checksums.sha256"
    checksums: dict[str, str] | None = None
    if checksums_path.exists():
        checksums = {}
        with open(checksums_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split(None, 1)
                if len(parts) == 2:
                    sha256, key = parts
                    checksums[key] = sha256

    # Validate -- on failure, return None (not fatal, handled by caller)
    try:
        from eligsir.experiments.replay_capsule import validate_capsule

        validate_capsule(capsule, checksums=checksums)
    except CapsuleValidationError:
        return None, None

    capsule_hash = capsule.capsule_hash()
    return capsule, capsule_hash


def write_replay_capsule_sidecar(
    model_path: str | Path,
    capsule_manifest: CapsuleManifest,
    capsule_hash: str,
) -> Path:
    """Write replay capsule metadata sidecar into model_path.

    The sidecar file contains capsule_id, capsule_hash, schema_version, scene_id,
    source.frame_count, and other metadata so #203's validity grader and #224/#186's
    baseline checks can consume the capsule without re-loading it from disk.

    Args:
        model_path: Output directory (must exist).
        capsule_manifest: Loaded CapsuleManifest.
        capsule_hash: Content hash string.

    Returns:
        Path to the written sidecar file (model_path/replay_capsule.json).
    """
    sidecar_data: dict[str, Any] = {
        "capsule_id": capsule_manifest.capsule_id,
        "capsule_hash": capsule_hash,
        "schema_version": capsule_manifest.schema_version,
        "scene_id": capsule_manifest.scene_id,
        "source_frame_count": capsule_manifest.source.frame_count,
        "source_uri": capsule_manifest.source.uri,
        "source_content_hash": capsule_manifest.source.content_hash,
        "lane_compatibility": list(capsule_manifest.lane_compatibility),
        "frontend_trace_mode": capsule_manifest.frontend_trace_mode,
        "admission_trace_mode": capsule_manifest.admission_trace_mode,
        "producer_git_commit": capsule_manifest.producer.git_commit,
        "producer_tool_version": capsule_manifest.producer.tool_version,
    }

    model_path_obj = Path(model_path)
    sidecar_path = model_path_obj / "replay_capsule.json"

    with open(sidecar_path, "w") as f:
        json.dump(sidecar_data, f, indent=2)

    return sidecar_path
