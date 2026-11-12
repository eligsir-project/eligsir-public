"""CLI for Scene Replay Capsule operations: validate, inspect, diff.

Issue #632 criterion 8: An entry point supporting `validate`, `inspect`,
and `diff` over a capsule directory/manifest.

Pure-python: no torch, no imports outside stdlib + eligsir.experiments.
Stays collectable on ``localhost/eligsir-ci-cpu`` (Dockerfile.ci-cpu).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import yaml

from eligsir.experiments.replay_capsule import (
    CapsuleManifest,
    CapsuleValidationError,
    FreezeAxisManifest,
    validate_capsule,
    validate_freeze_axes,
)


def _load_capsule_manifest(capsule_dir: Path) -> CapsuleManifest:
    """Load capsule manifest from directory, fail fast if parsing fails.

    Args:
        capsule_dir: Directory containing capsule_manifest.yaml.

    Returns:
        Validated CapsuleManifest.

    Raises:
        CapsuleValidationError: on schema or validation failure.
        FileNotFoundError: if manifest file not found.
        ValueError: if YAML/JSON parsing fails.
    """
    manifest_path = capsule_dir / "capsule_manifest.yaml"
    if not manifest_path.exists():
        raise FileNotFoundError(f"capsule_manifest.yaml not found in {capsule_dir}")

    with open(manifest_path) as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"capsule_manifest.yaml must be a YAML dict, got {type(data)}")

    # Parse to object (may raise KeyError/TypeError if fields missing)
    try:
        capsule = CapsuleManifest.from_dict(data)
    except (KeyError, TypeError) as e:
        raise ValueError(f"capsule_manifest.yaml missing required field: {e}")

    # Try to load checksums sidecar if it exists
    checksums_path = capsule_dir / "checksums.sha256"
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

    # Validate
    validate_capsule(capsule, checksums=checksums)
    return capsule


def _load_freeze_axes(capsule_dir: Path, manifest: CapsuleManifest) -> FreezeAxisManifest | None:
    """Load freeze-axis manifest if present in the capsule directory.

    Args:
        capsule_dir: Directory containing optional freeze_axes_manifest.yaml.
        manifest: The loaded capsule manifest (used for reference).

    Returns:
        FreezeAxisManifest if file exists, None otherwise.

    Raises:
        CapsuleValidationError: on schema or validation failure.
        ValueError: if YAML parsing fails.
    """
    freeze_path = capsule_dir / "freeze_axes_manifest.yaml"
    if not freeze_path.exists():
        return None

    with open(freeze_path) as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"freeze_axes_manifest.yaml must be a YAML dict, got {type(data)}")

    freeze = FreezeAxisManifest.from_dict(data)
    validate_freeze_axes(freeze)
    return freeze


def cmd_validate(capsule_dir: str) -> int:
    """Validate a capsule directory.

    Exits with:
        0 if valid.
        1 if CapsuleValidationError (prints the refusal reason).
        2 if other errors (file not found, parse failure).
    """
    try:
        _load_capsule_manifest(Path(capsule_dir))
        print(f"✓ {capsule_dir} is valid")
        return 0
    except CapsuleValidationError as e:
        print(f"✗ Capsule validation failed: {e}", file=sys.stderr)
        return 1
    except (FileNotFoundError, ValueError, KeyError, yaml.YAMLError, TypeError) as e:
        print(f"✗ Error loading capsule: {e}", file=sys.stderr)
        return 2


def cmd_inspect(capsule_dir: str) -> int:
    """Inspect a capsule: print manifest metadata in human-readable form.

    Exits with:
        0 on success.
        2 on load/parse errors.
    """
    try:
        capsule = _load_capsule_manifest(Path(capsule_dir))
        freeze = _load_freeze_axes(Path(capsule_dir), capsule)

        print(f"Capsule ID:         {capsule.capsule_id}")
        print(f"Capsule Hash:       {capsule.capsule_hash()}")
        print(f"Scene ID:           {capsule.scene_id}")
        print(f"Schema Version:     {capsule.schema_version}")
        print(f"Status:             {capsule.status}")
        print(f"Source URI:         {capsule.source.uri}")
        print(f"Source Hash:        {capsule.source.content_hash}")
        print(f"Source Frames:      {capsule.source.frame_count}")
        print(f"Lane Compatibility: {', '.join(capsule.lane_compatibility)}")
        print(f"Frontend Trace:     {capsule.frontend_trace_mode}")
        print(f"Admission Trace:    {capsule.admission_trace_mode}")
        print(f"Artifacts:          {len(capsule.artifacts)} files")
        print(f"Producer Commit:    {capsule.producer.git_commit}")
        print(f"Producer Version:   {capsule.producer.tool_version}")

        if freeze:
            print(f"\nFreeze Axes Manifest:")
            print(f"  Experiment ID:  {freeze.experiment_id}")
            print(f"  Capsule ID:     {freeze.capsule_id}")
            print(f"  Capsule Hash:   {freeze.capsule_hash}")
            print(f"  Lane:           {freeze.lane}")
            print(f"  Axes Declared:  {len(freeze.axes)}")

        return 0
    except (FileNotFoundError, ValueError, KeyError, yaml.YAMLError) as e:
        print(f"✗ Error inspecting capsule: {e}", file=sys.stderr)
        return 2


def cmd_diff(capsule_dir_a: str, capsule_dir_b: str) -> int:
    """Diff two capsules: report differences in hashes and freeze axes.

    Exits with:
        0 if identical.
        1 if differences found.
        2 on load errors.
    """
    try:
        cap_a = _load_capsule_manifest(Path(capsule_dir_a))
        cap_b = _load_capsule_manifest(Path(capsule_dir_b))
        freeze_a = _load_freeze_axes(Path(capsule_dir_a), cap_a)
        freeze_b = _load_freeze_axes(Path(capsule_dir_b), cap_b)

        hash_a = cap_a.capsule_hash()
        hash_b = cap_b.capsule_hash()

        differences = []

        if cap_a.capsule_id != cap_b.capsule_id:
            differences.append(f"  Capsule IDs differ: {cap_a.capsule_id} vs {cap_b.capsule_id}")

        if hash_a != hash_b:
            differences.append(f"  Content hashes differ:\n    {capsule_dir_a}: {hash_a}\n    {capsule_dir_b}: {hash_b}")

        if cap_a.source.content_hash != cap_b.source.content_hash:
            differences.append(f"  Source hashes differ: {cap_a.source.content_hash} vs {cap_b.source.content_hash}")

        if cap_a.source.frame_count != cap_b.source.frame_count:
            differences.append(f"  Frame counts differ: {cap_a.source.frame_count} vs {cap_b.source.frame_count}")

        # Compare freeze axes if both have them
        if freeze_a and freeze_b:
            axes_a = set(freeze_a.axes.keys())
            axes_b = set(freeze_b.axes.keys())

            # Check for differences in axis classification
            for axis in axes_a | axes_b:
                if axis not in axes_a:
                    differences.append(f"  Axis {axis} only in {capsule_dir_b}")
                elif axis not in axes_b:
                    differences.append(f"  Axis {axis} only in {capsule_dir_a}")
                elif freeze_a.axes[axis].classification != freeze_b.axes[axis].classification:
                    differences.append(
                        f"  Axis {axis} classification differs: "
                        f"{freeze_a.axes[axis].classification} vs {freeze_b.axes[axis].classification}"
                    )
        elif freeze_a or freeze_b:
            differences.append(
                f"  Freeze axes manifests not comparable: "
                f"{capsule_dir_a} has freeze manifest: {bool(freeze_a)}, "
                f"{capsule_dir_b} has freeze manifest: {bool(freeze_b)}"
            )

        if not differences:
            print(f"✓ Capsules are identical")
            return 0
        else:
            print(f"✗ Capsules differ:")
            for diff in differences:
                print(diff)
            return 1
    except (FileNotFoundError, ValueError, KeyError, yaml.YAMLError) as e:
        print(f"✗ Error comparing capsules: {e}", file=sys.stderr)
        return 2


def main(argv: list[str] | None = None) -> int:
    """Main entry point for capsule CLI.

    Args:
        argv: Command-line arguments (defaults to sys.argv[1:]).

    Returns:
        Exit code (0 on success, 1-2 on failure).
    """
    if argv is None:
        argv = sys.argv[1:]

    if len(argv) < 1:
        print(
            "Usage:\n"
            "  python -m eligsir.experiments.capsule_cli validate <capsule_dir>\n"
            "  python -m eligsir.experiments.capsule_cli inspect <capsule_dir>\n"
            "  python -m eligsir.experiments.capsule_cli diff <capsule_dir_a> <capsule_dir_b>",
            file=sys.stderr,
        )
        return 2

    cmd = argv[0]

    if cmd == "validate":
        if len(argv) < 2:
            print("Usage: validate <capsule_dir>", file=sys.stderr)
            return 2
        return cmd_validate(argv[1])
    elif cmd == "inspect":
        if len(argv) < 2:
            print("Usage: inspect <capsule_dir>", file=sys.stderr)
            return 2
        return cmd_inspect(argv[1])
    elif cmd == "diff":
        if len(argv) < 3:
            print("Usage: diff <capsule_dir_a> <capsule_dir_b>", file=sys.stderr)
            return 2
        return cmd_diff(argv[1], argv[2])
    else:
        print(f"Unknown command: {cmd}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
