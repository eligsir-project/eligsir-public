#!/usr/bin/env python3
"""Offline replay of #411's ``streaming/seed_cohort_exact/*`` scalars from checkpoints.

Why this exists (issue #416)
----------------------------
PR #411 wired the EXACT seed-cohort mask (``utils.gaussian_lineage.seed_cohort_mask``,
from #283/#373) into ``train_streaming.py``'s TensorBoard logging and into
``report.json["geometry"]["seed_cohort"]`` -- but no GPU was available when it was
written, so neither had ever executed. Runs that finished BEFORE #411 merged
(``debcd68``, 2026-08-13T17:45:09+02:00) therefore carry no such scalars in their
event files, even when their checkpoints DO carry everything the mask needs
(i.e. they are ``trustworthy`` per ``scripts/lineage_coverage_audit.py``).

This script recovers exactly those four quantities from such a run's checkpoint
series after the fact. It evaluates the SAME primitives the logging site calls
(``seed_uid_range`` / ``seed_cohort_mask`` / ``seed_lineage_mask``) through a
duck-typed shim, rather than reimplementing the mask, so it cannot drift from the
real definition.

What it is NOT
--------------
It is NOT a substitute for validating the live logging path. It proves the mask
and its inputs are sound on real training output; it says nothing about whether
``tb_writer.add_scalar`` at the logging site fires, nor about the report block.
Those need a run that started after ``debcd68``.

Usage
-----
    python scripts/seed_cohort_offline_replay.py <run_dir> [out.json]

Checkpoints are enumerated NUMERICALLY (``chkpnt10000`` > ``chkpnt2000``), never
lexically.
"""
import argparse
import glob
import json
import os
import re
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eligsir.utils.gaussian_lineage import (  # noqa: E402
    CheckpointLineageView,
    SeedCohortUnavailable,
    seed_cohort_mask,
    seed_lineage_mask,
    seed_uid_range,
)

LINEAGE_BUFFERS = (
    "gaussian_uid", "parent_uid", "root_uid", "generation",
    "birth_method", "last_reseed_iter", "birth_iter",
)

#: The shim moved into ``utils.gaussian_lineage`` (issue #416 step 4) so that
#: ``utils/seed_cohort_drift.py`` could route its retargeted cohort mask
#: through the SAME view instead of growing a second one. Kept as an alias:
#: this module's name for it is what the existing tests and prior write-ups
#: refer to, and the class's behaviour is unchanged by the move.
_CheckpointShim = CheckpointLineageView


def numeric_checkpoints(run_dir: str):
    found = []
    for path in glob.glob(os.path.join(run_dir, "chkpnt*.pth")):
        match = re.search(r"chkpnt(\d+)\.pth$", os.path.basename(path))
        if match:
            found.append((int(match.group(1)), path))
    return sorted(found)


def replay(run_dir: str):
    rows = []
    for iteration, path in numeric_checkpoints(run_dir):
        blob = torch.load(path, map_location="cpu", weights_only=False)[0]
        shim = _CheckpointShim(blob)
        try:
            lo, hi = seed_uid_range(shim)
            exact = seed_cohort_mask(shim)
            alive = int(exact.sum().item())
            row = {
                "iteration": iteration,
                "checkpoint": os.path.basename(path),
                "population": int(shim.get_xyz.shape[0]),
                "seed_uid_lo": lo,
                "seed_uid_hi": hi,
                "alive_count": alive,
                "lineage_alive_count": int(seed_lineage_mask(shim).sum().item()),
                "survival_fraction": (alive / (hi - lo)) if hi > lo else 0.0,
                "mean_opacity": (
                    float(shim.get_opacity[exact].mean()) if alive else None
                ),
            }
        except SeedCohortUnavailable as exc:
            row = {
                "iteration": iteration,
                "checkpoint": os.path.basename(path),
                "error": f"SeedCohortUnavailable: {exc}",
            }
        rows.append(row)
        print(json.dumps(row), flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", help="a run directory containing chkpnt*.pth")
    parser.add_argument("out", nargs="?", help="write the rows here as JSON")
    args = parser.parse_args()
    rows = replay(args.run_dir)
    if args.out:
        with open(args.out, "w") as handle:
            json.dump(rows, handle, indent=2)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
