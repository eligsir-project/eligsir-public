#!/usr/bin/env python3
"""Build a standard sequential ablator spec for a frozen W3-02 engine A/B."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eligsir.experiments.manifest import ExperimentManifest
from eligsir.experiments.w3_02_ablation import EvidenceValidationError, build_ablator_spec


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="Frozen Level-D experiment manifest JSON.")
    parser.add_argument("--base", type=Path, required=True, help="JSON object with ablator base fields.")
    parser.add_argument("--arm", choices=("baseline", "treatment"), required=True)
    parser.add_argument(
        "--runtime-manifest",
        required=True,
        help="Repository-relative manifest path available inside the train container.",
    )
    parser.add_argument("--mutation-trace", required=True, help="Repository-relative frozen mutation trace path.")
    parser.add_argument(
        "--frozen-artifacts",
        type=Path,
        required=True,
        help="JSON object mapping config_hash/source_artifact_hash/pose_hash/keyframe_hash/holdout_hash to runtime-visible paths.",
    )
    parser.add_argument("--output", type=Path, required=True, help="Path for the generated ablator spec.")
    args = parser.parse_args(argv)
    try:
        manifest = ExperimentManifest.from_dict(json.loads(args.manifest.read_text()))
        base = json.loads(args.base.read_text())
        frozen_artifact_paths = json.loads(args.frozen_artifacts.read_text())
        if not isinstance(base, dict):
            raise EvidenceValidationError("--base must contain a JSON object")
        if not isinstance(frozen_artifact_paths, dict):
            raise EvidenceValidationError("--frozen-artifacts must contain a JSON object")
        spec = build_ablator_spec(
            manifest, base, arm_name=args.arm, runtime_manifest_path=args.runtime_manifest,
            mutation_trace_path=args.mutation_trace,
            frozen_artifact_paths=frozen_artifact_paths,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n")
    except (OSError, json.JSONDecodeError, EvidenceValidationError) as error:
        parser.error(str(error))
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
