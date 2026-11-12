#!/usr/bin/env python3
"""Generate an auditable frozen W3-02 engine A/B evidence report.

The input run directories must be real, sequential ablator outputs. This
command does not launch training or synthesize measurements.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eligsir.experiments.manifest import ExperimentManifest
from eligsir.experiments.w3_02_ablation import (
    EvidenceValidationError,
    write_w3_02_evidence_report,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="Frozen Level-D experiment manifest JSON.")
    parser.add_argument("--baseline-run", type=Path, required=True, help="Baseline ablator model directory.")
    parser.add_argument("--treatment-run", type=Path, required=True, help="Treatment ablator model directory.")
    parser.add_argument("--output", type=Path, required=True, help="Evidence report JSON to write.")
    parser.add_argument("--claim-improvement", action="store_true")
    parser.add_argument("--negative-control-run", type=Path)
    args = parser.parse_args(argv)
    try:
        manifest = ExperimentManifest.from_dict(json.loads(args.manifest.read_text()))
        output = write_w3_02_evidence_report(
            manifest, args.baseline_run, args.treatment_run, args.output,
            claim_improvement=args.claim_improvement, negative_control_run_dir=args.negative_control_run,
        )
    except (OSError, json.JSONDecodeError, EvidenceValidationError) as error:
        parser.error(str(error))
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
