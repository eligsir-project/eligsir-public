#!/usr/bin/env python3
"""Fail closed unless causal mapping and refinement meet view-rate targets."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping


def _rate(values: Mapping[str, Any], name: str, *, label: str) -> float:
    value = values.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    rate = float(value)
    if not math.isfinite(rate) or rate < 0.0:
        raise ValueError(f"{label} must be finite and non-negative")
    return rate


def validate_phase_throughput(
    mapping_summary: Mapping[str, Any],
    refinement_metadata: Mapping[str, Any],
    *,
    minimum_views_per_s: float,
) -> dict[str, dict[str, float]]:
    """Validate active and wall-clock view rates for both causal phases."""
    target = float(minimum_views_per_s)
    if not math.isfinite(target) or target <= 0.0:
        raise ValueError("minimum views per second must be finite and positive")
    report = {
        "mapping": {
            "active_views_per_s": _rate(
                mapping_summary,
                "supervision_views_per_active_s",
                label="mapping active views per second",
            ),
            "wall_views_per_s": _rate(
                mapping_summary,
                "supervision_views_per_elapsed_s",
                label="mapping wall views per second",
            ),
        },
        "refinement": {
            "active_views_per_s": _rate(
                refinement_metadata,
                "refinement_supervision_views_per_active_s",
                label="refinement active views per second",
            ),
            "wall_views_per_s": _rate(
                refinement_metadata,
                "refinement_supervision_views_per_wall_s",
                label="refinement wall views per second",
            ),
        },
    }
    for phase, rates in report.items():
        for clock, rate in rates.items():
            if rate < target:
                raise ValueError(
                    f"{phase} {clock.replace('_', ' ')} {rate:.3f} is below "
                    f"the required {target:.3f} views/s"
                )
    return report


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"required artifact is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"required JSON object is invalid: {path}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_path", type=Path)
    parser.add_argument("--minimum-views-per-s", type=float, default=100.0)
    args = parser.parse_args()
    root = args.model_path.resolve()
    try:
        report = validate_phase_throughput(
            _read_json(root / "causal_replay_summary.json"),
            _read_json(root / "streaming_phase_metadata.json"),
            minimum_views_per_s=args.minimum_views_per_s,
        )
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
