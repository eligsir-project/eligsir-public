#!/usr/bin/env python
"""Grade issue #823's admission-comparison gates from finished run artifacts.

#823 is the canonical matched-budget scientific comparison for frame/keyframe
selection. Before any quality number from it may be read, two families of
pre-registered gates must pass, and until now neither could be evaluated
without hand-parsing `keyframe_admission_log.jsonl`:

**Per-arm (the #947 manipulation check, folded into #823 by the 2026-08-26
consolidation).** The classical arm must be *materially owned by actual
motion/coverage/reliability evidence rather than a periodic backstop*, and
the fixed-stride control must be owned 100% by its own stride. This is
`AdmissionFingerprint.admission_gate` -- computed live into
`report.json["admission_fingerprint"]["admission_gate"]` for runs from this
change onward, and replayed here from the per-decision sidecar for every
run (including historical ones) via
`scripts/offline_admission_fingerprint.replay_log`. No gate logic is
duplicated in this file.

**Cross-arm (#934's Stage-1 gates, carried forward into #823).** Two arms
must select materially DIFFERENT frames (Jaccard of committed source-frame
IDs below 0.60 -- otherwise the policies are choosing substantially the same
supervision set, which is itself the answer) at a MATCHED committed budget
(|K_A - K_B| / K_B <= 0.05 -- otherwise a quality delta is a work-budget
artifact, the exact confound that invalidated #934's own
`i934stage1a3_novelty_force` arm at 8.65x admission inflation).

Read-only. Never touches training, ablator, or any run directory.

Usage
-----
    # grade one or more arms individually
    python scripts/grade_admission_gate.py <run_dir> [<run_dir> ...]

    # additionally check the cross-arm gates between exactly two arms
    python scripts/grade_admission_gate.py --pair <control_dir> <treatment_dir>

    # machine-readable
    python scripts/grade_admission_gate.py <run_dir> ... --json out.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Set, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.offline_admission_fingerprint import (  # noqa: E402
    LOG_NAME,
    replay_log,
)

#: #934 Stage-1 gate 1. At or above this Jaccard the two policies are
#: choosing substantially the same supervision set and the selection
#: experiment is cancelled -- a real result, not a failure.
SELECTION_JACCARD_MAX = 0.60
#: #934 Stage-1 gate 2. Committed-K must match within this relative
#: tolerance or a quality delta is a work-budget artifact.
COMMITTED_K_REL_TOL = 0.05


def admitted_frame_ids(run_dir: str) -> Tuple[Optional[Set[int]], str]:
    """Committed frame identities from a run's per-decision sidecar.

    **What ``frame`` actually is, and the caveat that follows.** The sidecar
    row's ``frame`` key is ``train_streaming.py``'s ``n_frames_ingested`` --
    the run's frame-ARRIVAL ordinal, not a source-trajectory index. It is a
    valid cross-arm identity **only when both arms release the same source
    frames in the same order**, i.e. identical scene, ingestion mode,
    steps-per-frame / release cadence, source stride and
    ``--streaming_max_frames``. #823 already requires all of those held
    identical between arms, so under a conforming spec the ordinal and the
    source frame are in bijection -- but this function cannot verify it, so
    a Jaccard computed from these sets is only as trustworthy as the
    spec-level matching. Check the two arms' ``cfg_args`` before quoting it.

    Returns ``(ids, note)``. ``ids`` is None when the identity cannot be
    established -- deliberately, rather than falling back to row position,
    which would silently compare positions instead of frames and make two
    arms with different candidate populations look artificially similar.
    """
    path = os.path.join(run_dir, LOG_NAME)
    if not os.path.isfile(path):
        return None, f"{LOG_NAME} missing"
    ids: Set[int] = set()
    rows = 0
    without_id = 0
    try:
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not row.get("admitted"):
                    continue
                rows += 1
                fid = row.get("frame")
                if fid is None:
                    without_id += 1
                    continue
                ids.add(int(fid))
    except Exception as exc:  # noqa: BLE001
        return None, f"read_error: {exc}"
    if rows == 0:
        return None, "no admitted rows"
    if without_id:
        return None, f"{without_id}/{rows} admitted rows carry no `frame` id"
    return ids, f"{len(ids)} unique committed frame ids over {rows} admitted rows"


def grade_arm(run_dir: str) -> Dict:
    """Per-arm #947/#823 gate verdict, replayed from the run's own sidecar."""
    log_path = os.path.join(run_dir, LOG_NAME)
    if not os.path.isfile(log_path):
        return {
            "run_dir": run_dir,
            "status": "NO_DECISION_LOG",
            "note": (
                f"{LOG_NAME} absent -- the gate cannot be graded. This is NOT "
                "a pass; it means the arm produced no per-decision evidence."
            ),
        }
    replay = replay_log(log_path)
    if replay.get("status") == "FAILED":
        return {"run_dir": run_dir, "status": "FAILED", "note": replay.get("failure_reason")}
    fingerprint = replay.get("fingerprint", {})
    gate = fingerprint.get("admission_gate", {})
    ids, id_note = admitted_frame_ids(run_dir)
    return {
        "run_dir": run_dir,
        "status": replay.get("status"),
        "scene": replay.get("scene"),
        "nominal_policy": fingerprint.get("nominal_policy"),
        "admitted_total": fingerprint.get("admitted_total"),
        "committed_k": len(ids) if ids is not None else None,
        "committed_k_note": id_note,
        "admission_gate": gate,
        # #795's own, separate verdict. Reported alongside on purpose: it is a
        # different predicate (70% of admits `forced`) from #823's gate (50%
        # of admits owned by a precedence backstop trigger) and the two can
        # legitimately disagree.
        "admission_realized_mismatch": fingerprint.get("admission_realized_mismatch"),
        "mismatch_reason": fingerprint.get("mismatch_reason"),
    }


def grade_pair(arm_a: Dict, arm_b: Dict, ids_a: Optional[Set[int]], ids_b: Optional[Set[int]]) -> Dict:
    """#934 Stage-1 cross-arm gates, carried forward into #823."""
    out: Dict[str, object] = {
        "arm_a": arm_a["run_dir"],
        "arm_b": arm_b["run_dir"],
    }
    if ids_a is None or ids_b is None:
        out["status"] = "UNGRADEABLE"
        out["note"] = (
            "committed frame identity unavailable on at least one arm "
            f"(a: {arm_a.get('committed_k_note')}; b: {arm_b.get('committed_k_note')})"
        )
        return out

    union = ids_a | ids_b
    jaccard = len(ids_a & ids_b) / len(union) if union else 1.0
    k_a, k_b = len(ids_a), len(ids_b)
    k_rel_err = abs(k_a - k_b) / k_b if k_b else float("inf")

    selection_differs = jaccard < SELECTION_JACCARD_MAX
    k_matched = k_rel_err <= COMMITTED_K_REL_TOL
    out.update(
        {
            "committed_k_a": k_a,
            "committed_k_b": k_b,
            "committed_k_rel_error": k_rel_err,
            "committed_k_matched": k_matched,
            "committed_k_rel_tol": COMMITTED_K_REL_TOL,
            "selection_jaccard": jaccard,
            "selection_jaccard_max": SELECTION_JACCARD_MAX,
            "selection_differs_materially": selection_differs,
        }
    )
    if not k_matched:
        out["status"] = "K_MISMATCH"
        out["note"] = (
            "committed budgets are not matched; any quality delta between "
            "these arms is confounded with a different amount of supervision "
            "work and must not be read as a selection result."
        )
    elif not selection_differs:
        out["status"] = "SELECTION_EQUIVALENT"
        out["note"] = (
            "the two policies commit substantially the same supervision set; "
            "per #823 this is itself the answer -- retain the simpler policy."
        )
    else:
        out["status"] = "COMPARABLE"
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dirs", nargs="*", help="run directories to grade individually")
    ap.add_argument("--pair", nargs=2, metavar=("CONTROL", "TREATMENT"),
                    help="additionally grade the cross-arm gates between two arms")
    ap.add_argument("--json", dest="json_out", help="write the full result to this path")
    args = ap.parse_args()

    targets: List[str] = list(args.run_dirs)
    if args.pair:
        for d in args.pair:
            if d not in targets:
                targets.append(d)
    if not targets:
        ap.error("no run directories given")

    arms = {d: grade_arm(d) for d in targets}
    result: Dict[str, object] = {"arms": list(arms.values())}

    for arm in arms.values():
        gate = arm.get("admission_gate") or {}
        print(f"\n=== {arm['run_dir']}")
        print(f"    status            : {arm['status']}")
        print(f"    nominal_policy    : {arm.get('nominal_policy')}")
        print(f"    admitted_total    : {arm.get('admitted_total')}")
        print(f"    committed_k       : {arm.get('committed_k')}  ({arm.get('committed_k_note')})")
        print(f"    #823 gate verdict : {gate.get('verdict')}")
        if "backstop_owner_share" in gate:
            print(f"      backstop share  : {gate['backstop_owner_share']:.3f} "
                  f"(bar < {gate.get('threshold_backstop_owner_share_max')})")
            print(f"      evidence share  : {gate.get('evidence_owner_share'):.3f}")
        if gate.get("motion_owner_decomposition"):
            m = gate["motion_owner_decomposition"]
            print(f"      motion owners   : translation_only={m['translation_only']:.3f} "
                  f"rotation_only={m['rotation_only']:.3f} both={m['both']:.3f} "
                  f"(n={m['n_motion_admits']})")
        if "control_purity" in gate:
            print(f"      control purity  : {gate['control_purity']:.3f}")
        print(f"    #795 mismatch     : {arm.get('mismatch_reason')}")

    if args.pair:
        ids_a, _ = admitted_frame_ids(args.pair[0])
        ids_b, _ = admitted_frame_ids(args.pair[1])
        pair = grade_pair(arms[args.pair[0]], arms[args.pair[1]], ids_a, ids_b)
        result["pair"] = pair
        print("\n=== cross-arm (#934 Stage-1 gates, carried into #823)")
        for k, v in pair.items():
            print(f"    {k}: {v}")

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
