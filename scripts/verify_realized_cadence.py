#!/usr/bin/env python3
"""Reconcile a streaming run's REQUESTED cadence against its REALIZED cadence.

Issue #289 ("[Seed Fade 7] Verify realized frame-release, admission, insertion, and
optimization cadence before interpreting streaming failures").

Motivation
----------
Seed fade / streaming quality failures are rate-and-capacity phenomena.  Before any
causal story is told about a PSNR curve, the run's *actual* clocks must be shown to be
the ones the experiment asked for.  Two documented, still-unfixed defects can silently
break that assumption (see ``docs/architecture/2026-08-09-w3-38-admission-gap.md``):

  (a) ``MappingScheduler.effective_spf`` silently overriding an explicit
      ``--streaming_steps_per_frame`` via the ``min_steps_per_frame=50`` floor on
      registry-registered scenes.
  (b) ``_coverage_n_frames`` resolving against the FULL registered scene length rather
      than the ``--streaming_max_frames`` subset, making (a) fire more readily.

A third, distinct mechanism was found while writing this script (issue #278/#302):
under ``ingestion_mode=dataset_fps`` the frame source is *not* iteration-paced at all,
so the whole trajectory can be released hundreds of times faster than the configured
``steps_per_frame`` — with no warning in the per-run report beyond the mode name.

What it does
------------
For each run directory it reads only that run's own artifacts and reports:

  * requested vs. realized steps-per-frame (and which of the two candidate values —
    the requested ``--streaming_steps_per_frame`` or the scheduler's ``effective_spf``
    — the run actually tracked, or neither);
  * frames released vs. admitted vs. depth-inserted;
  * optimizer updates per released frame and per admitted keyframe;
  * whether the scene hit or missed the ``eval/scenes.json`` registry;
  * the realized *supervision resolution* schedule (multi-resolution tier switches),
    because "an optimizer update" at 160x120 is not the same clock as one at 640x480;
  * the test-PSNR curve knee, so a curve feature can be lined up against the clocks.

Per CLAUDE.md, when ``train.log`` contains multiple ``[RUN START]`` banners only the
content after the LAST one is trustworthy; this script slices the log accordingly and
says how many banners it saw.

Usage
-----
    python scripts/verify_realized_cadence.py RUN_DIR [RUN_DIR ...]
    python scripts/verify_realized_cadence.py --json RUN_DIR > cadence.json

Nothing here launches training or touches the GPU; it is pure artifact analysis.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

# ``--iterations`` sentinel used for --online_mode runs; also the directory name of the
# final ``comparison/iter_*`` report for such runs.
ONLINE_UNBOUNDED_ITERATIONS = 2147483647

RUN_START_RE = re.compile(r"\[RUN START\]\s+(\S+)\s+host=(\S+)\s+pid=(\S+)\s+model_path=(\S+)")
EFFECTIVE_SPF_RE = re.compile(r"\[MappingScheduler\][^\n]*effective_spf=(\d+)")
MIN_SPF_FLOOR_RE = re.compile(r"\[MappingScheduler\] min_spf floor:[^\n]*")
STARTING_SPF_RE = re.compile(r"\[streaming\] Starting:[^\n]*steps_per_frame=(\d+)")
INGESTION_RE = re.compile(r"\[streaming\] ingestion_mode=([\w]+)[^\n]*")
DATASET_FRAMES_RE = re.compile(r"\[streaming\] Dataset: (\d+) frames\.")
FINAL_STEP_RE = re.compile(r"\[FINAL (?:STEP|ITER) (\d+)\]")
MULTIRES_TIER_RE = re.compile(r"\[multires\] step=(\d+) tier=(\w+) cam_h=(\d+) cam_w=(\d+)")
MULTIRES_BIND_RE = re.compile(r"\[multires\] bind stats: (\{[^}]*\})")
PARTITION_REGISTRY_RE = re.compile(r"\[report\] partition PSNR \((\w+)\)")

# cfg_args is a repr() of an argparse Namespace: Namespace(a=1, b='x', ...)
CFG_TOKEN_RE = re.compile(r"([A-Za-z_][A-Za-z_0-9]*)\s*=\s*('[^']*'|\"[^\"]*\"|[^,()\[\]]+)")

CFG_KEYS_OF_INTEREST = (
    "online_mode",
    "streaming_replay",
    "streaming_steps_per_frame",
    "streaming_min_steps_per_frame",
    "streaming_ingestion_mode",
    "streaming_max_frames",
    "streaming_max_iterations",
    "streaming_refinement_iterations",
    "streaming_frame_admission",
    "streaming_insert_from_depth",
    "streaming_multires_low_divisor",
    "streaming_multires_it_v_floor",
    "streaming_multires_map_fps_floor",
    "streaming_eval_hold",
    "iterations",
)


def _num(value: str) -> Any:
    value = value.strip().strip("'\"")
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            continue
    if value in ("True", "False"):
        return value == "True"
    if value == "None":
        return None
    return value


@dataclass
class CadenceReport:
    run_dir: str
    ok: bool = True
    problems: list[str] = field(default_factory=list)

    # provenance
    n_run_start_banners: int = 0
    last_run_start: str | None = None
    host: str | None = None

    # requested
    requested_spf: int | None = None
    requested_min_spf: int | None = None
    requested_ingestion_mode: str | None = None
    requested_max_frames: int | None = None
    online_mode: bool | None = None
    multires_it_v_floor: float | None = None
    multires_map_fps_floor: float | None = None

    # scheduler / registry
    effective_spf: int | None = None
    min_spf_floor_fired: bool = False
    scene_registered: bool | None = None
    registry_n_source_frames: int | None = None
    log_dataset_frames: int | None = None
    starting_spf_logged: int | None = None
    resolved_ingestion_mode: str | None = None
    ingestion_reason: str | None = None

    # realized
    final_iteration: int | None = None
    n_release_events: int = 0
    first_release_iter: int | None = None
    last_release_iter: int | None = None
    first_release_frame: int | None = None
    last_release_frame: int | None = None
    realized_spf_released: float | None = None
    # Issue #289: the in-run realized-cadence block, when the run carries one
    # (report.json["realized_cadence"], written by train_streaming.py since
    # #289). Authoritative when present -- it is measured from the training
    # loop's own clocks rather than reconstructed from the progress bar -- and
    # deliberately NOT merged into the log-derived fields above, so a
    # disagreement between the two stays visible instead of being papered over.
    in_run_cadence: dict[str, Any] | None = None
    in_run_realized_spf: float | None = None
    in_run_cadence_verdict: str | None = None
    n_admitted: int = 0
    first_admit_iter: int | None = None
    last_admit_iter: int | None = None
    realized_iters_per_admitted: float | None = None
    admission_reasons: dict[str, int] = field(default_factory=dict)
    n_holdout: int | None = None
    accounting_identity_ok: bool | None = None
    accounting_identity_detail: str | None = None

    # insertion
    depth_insertion_enabled: bool | None = None
    total_inserted: int | None = None
    inserted_per_admitted: float | None = None

    # supervision resolution (a realized clock in its own right)
    multires_tier_switches: list[dict[str, Any]] = field(default_factory=list)
    multires_bind_stats: str | None = None
    low_tier_iterations: int | None = None
    high_tier_start_iter: int | None = None
    iters_per_second: float | None = None
    implied_map_fps: float | None = None
    multires_floor_feasible: bool | None = None

    # quality curve, for lining features up against the clocks
    psnr_first: tuple[int, float] | None = None
    psnr_last: tuple[int, float] | None = None
    psnr_knee_iter: int | None = None
    psnr_knee_delta_db: float | None = None
    mean_test_psnr: float | None = None
    n_test: int | None = None
    eval_valid: bool | None = None
    seed_fade: dict[str, Any] | None = None

    # verdicts
    spf_tracked: str | None = None
    verdicts: list[str] = field(default_factory=list)


def _read_log_after_last_run_start(path: Path, report: CadenceReport) -> str:
    text = path.read_text(errors="replace")
    matches = list(RUN_START_RE.finditer(text))
    report.n_run_start_banners = len(matches)
    if not matches:
        report.problems.append("train.log has no [RUN START] banner (pre-2026-07-06 run?)")
        return text
    last = matches[-1]
    report.last_run_start = last.group(1)
    report.host = last.group(2)
    if len(matches) > 1:
        report.problems.append(
            f"train.log contains {len(matches)} [RUN START] banners "
            "(reused model_path); only content after the LAST one was read"
        )
    return text[last.start():]


def _parse_cfg_args(run: Path, report: CadenceReport) -> dict[str, Any]:
    cfg_path = run / "cfg_args"
    if not cfg_path.exists():
        report.problems.append("cfg_args missing")
        return {}
    raw = cfg_path.read_text(errors="replace")
    cfg = {k: _num(v) for k, v in CFG_TOKEN_RE.findall(raw)}
    report.requested_spf = cfg.get("streaming_steps_per_frame")
    report.requested_min_spf = cfg.get("streaming_min_steps_per_frame")
    report.requested_ingestion_mode = cfg.get("streaming_ingestion_mode")
    report.requested_max_frames = cfg.get("streaming_max_frames")
    report.online_mode = cfg.get("online_mode")
    report.multires_it_v_floor = cfg.get("streaming_multires_it_v_floor")
    report.multires_map_fps_floor = cfg.get("streaming_multires_map_fps_floor")
    return {k: cfg.get(k) for k in CFG_KEYS_OF_INTEREST if k in cfg}


def _parse_log(text: str, report: CadenceReport) -> None:
    if m := EFFECTIVE_SPF_RE.search(text):
        report.effective_spf = int(m.group(1))
    report.min_spf_floor_fired = bool(MIN_SPF_FLOOR_RE.search(text))
    if m := STARTING_SPF_RE.search(text):
        report.starting_spf_logged = int(m.group(1))
    if m := INGESTION_RE.search(text):
        report.resolved_ingestion_mode = m.group(1)
        line = m.group(0)
        if "resolved to" in line:
            # "[streaming] ingestion_mode=auto resolved to dataset_fps"
            tail = line.split("resolved to", 1)[1].strip()
            report.resolved_ingestion_mode = tail.split()[0]
            report.ingestion_reason = "auto_resolved"
        if rm := re.search(r"reason=([\w]+)", line):
            report.ingestion_reason = rm.group(1)
    if m := DATASET_FRAMES_RE.search(text):
        report.log_dataset_frames = int(m.group(1))
    finals = FINAL_STEP_RE.findall(text)
    if finals:
        report.final_iteration = int(finals[-1])
    if PARTITION_REGISTRY_RE.search(text):
        # "(registry)" vs "(observed)" tells us the denominator source
        report.scene_registered = PARTITION_REGISTRY_RE.search(text).group(1) == "registry"

    switches = [
        {"iter": int(i), "tier": t, "h": int(h), "w": int(w)}
        for i, t, h, w in MULTIRES_TIER_RE.findall(text)
    ]
    report.multires_tier_switches = switches
    if m := MULTIRES_BIND_RE.search(text):
        report.multires_bind_stats = m.group(1)
        if lm := re.search(r"'low_ok':\s*(\d+)", m.group(1)):
            report.low_tier_iterations = int(lm.group(1))
    highs = [s for s in switches if s["tier"] == "high"]
    if len(highs) > 1:
        # first "high" is the bootstrap/step-1 bind; the interesting one is the
        # promotion back to high after the low-tier stretch.
        report.high_tier_start_iter = highs[-1]["iter"]


def _parse_admission_log(run: Path, report: CadenceReport) -> None:
    path = run / "keyframe_admission_log.jsonl"
    if not path.exists():
        report.problems.append(
            "keyframe_admission_log.jsonl missing "
            "(--streaming_admission_log_enabled off?) — realized cadence not measurable"
        )
        return
    rows: list[dict[str, Any]] = []
    for line in path.read_text(errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    if len(rows) < 2:
        report.problems.append("keyframe_admission_log.jsonl has <2 rows")
        return

    report.n_release_events = len(rows)
    report.first_release_iter = rows[0]["iter"]
    report.last_release_iter = rows[-1]["iter"]
    report.first_release_frame = rows[0].get("frame")
    report.last_release_frame = rows[-1].get("frame")
    span_iters = rows[-1]["iter"] - rows[0]["iter"]
    # Divide by the SOURCE-FRAME index span, not the row count: eval holdouts are
    # withheld before admission and never appear in this log, so row count
    # under-counts the frames the scheduler actually released.
    span_frames = None
    if report.first_release_frame is not None and report.last_release_frame is not None:
        span_frames = report.last_release_frame - report.first_release_frame
    if not span_frames:
        span_frames = len(rows) - 1
    report.realized_spf_released = round(span_iters / span_frames, 3)

    reasons: dict[str, int] = {}
    for r in rows:
        reasons[r.get("reason", "?")] = reasons.get(r.get("reason", "?"), 0) + 1
    report.admission_reasons = dict(sorted(reasons.items(), key=lambda kv: -kv[1]))

    adm = [r for r in rows if r.get("admitted")]
    report.n_admitted = len(adm)
    if len(adm) >= 2:
        report.first_admit_iter = adm[0]["iter"]
        report.last_admit_iter = adm[-1]["iter"]
        report.realized_iters_per_admitted = round(
            (adm[-1]["iter"] - adm[0]["iter"]) / (len(adm) - 1), 2
        )


def _parse_trajectory_full(run: Path, report: CadenceReport) -> None:
    """Accounting identity check: released candidates + holdouts == source frames."""
    path = run / "trajectory_full.jsonl"
    if not path.exists():
        report.accounting_identity_detail = "trajectory_full.jsonl missing"
        return
    n_total = n_holdout = 0
    for line in path.read_text(errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        n_total += 1
        if row.get("is_holdout"):
            n_holdout += 1
    report.n_holdout = n_holdout

    # Identity: every source frame past bootstrap is either a withheld holdout or a
    # candidate that got an admission decision.  If this fails, the admission log is
    # incomplete/truncated and none of the realized rates above should be trusted.
    if (
        report.first_release_frame is not None
        and report.last_release_frame is not None
        and report.n_release_events
    ):
        candidates = report.last_release_frame - report.first_release_frame + 1
        expected = report.n_release_events + n_holdout
        report.accounting_identity_ok = abs(candidates - expected) <= 1
        report.accounting_identity_detail = (
            f"frames {report.first_release_frame}..{report.last_release_frame} "
            f"= {candidates} candidates; admission rows {report.n_release_events} "
            f"+ holdouts {n_holdout} = {expected}"
        )


def _pick_final_report(run: Path, report: CadenceReport) -> dict[str, Any] | None:
    comp = run / "comparison"
    if not comp.is_dir():
        report.problems.append("no comparison/ directory")
        return None
    # Sort NUMERICALLY, never as strings: iter_9000 sorts after iter_60000 lexically.
    candidates: list[tuple[int, Path]] = []
    for d in comp.iterdir():
        m = re.fullmatch(r"iter_(\d+)", d.name)
        if m and (d / "report.json").exists():
            candidates.append((int(m.group(1)), d))
    if not candidates:
        report.problems.append("no comparison/iter_*/report.json found")
        return None
    candidates.sort(key=lambda t: t[0])
    chosen = candidates[-1][1]
    try:
        return json.loads((chosen / "report.json").read_text())
    except json.JSONDecodeError:
        report.problems.append(f"{chosen}/report.json is not valid JSON")
        return None


def _parse_report(rep: dict[str, Any], report: CadenceReport) -> None:
    report.mean_test_psnr = rep.get("mean_test_psnr")
    report.n_test = rep.get("n_test")
    report.eval_valid = rep.get("eval_valid")
    report.seed_fade = rep.get("seed_fade")
    ing = rep.get("ingestion") or {}
    if ing:
        report.resolved_ingestion_mode = ing.get("mode", report.resolved_ingestion_mode)
        report.ingestion_reason = ing.get("reason", report.ingestion_reason)
    # Issue #289's in-run block, when present. Prefer it for the headline
    # realized-cadence numbers; fall back to the log parsing above for every run
    # that predates it (which is every run in the corpus as of 2026-08-14).
    rc = rep.get("realized_cadence") or {}
    if rc:
        report.in_run_cadence = rc
        report.in_run_realized_spf = rc.get("realized_steps_per_release")
        report.in_run_cadence_verdict = (rc.get("reconciliation") or {}).get("verdict")
    cov = rep.get("trajectory_coverage") or {}
    if "scene_registered" in cov:
        report.scene_registered = cov["scene_registered"]
    report.registry_n_source_frames = cov.get("n_source_frames")
    geo = rep.get("geometry") or {}
    di = geo.get("depth_insertion") or {}
    if di:
        report.depth_insertion_enabled = di.get("enabled")
        report.total_inserted = di.get("total_inserted")
        if report.n_admitted:
            report.inserted_per_admitted = round(
                (di.get("total_inserted") or 0) / report.n_admitted, 1
            )


def _parse_psnr_history(run: Path, report: CadenceReport) -> None:
    path = run / "psnr_history.jsonl"
    if not path.exists():
        return
    pts: list[tuple[int, float]] = []
    for line in path.read_text(errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "iteration" in row and row.get("mean_test_psnr") is not None:
            pts.append((int(row["iteration"]), float(row["mean_test_psnr"])))
    if not pts:
        return
    pts.sort()
    report.psnr_first = pts[0]
    report.psnr_last = pts[-1]
    if len(pts) > 1:
        deltas = [(pts[i][1] - pts[i - 1][1], pts[i][0]) for i in range(1, len(pts))]
        best = max(deltas)
        report.psnr_knee_delta_db = round(best[0], 3)
        report.psnr_knee_iter = best[1]


def _parse_timing(run: Path, report: CadenceReport) -> None:
    """Derive optimizer throughput, and whether the multires dual floor is satisfiable.

    ``ResolutionController.decide`` demotes supervision to the low tier whenever
    *either* rolling ``it/V`` < ``it_v_floor`` *or* rolling ``map_fps`` <
    ``map_fps_floor``.  Since ``map_fps = (it/s) / (it/V)``, holding both at once
    requires a sustained ``it/s >= it_v_floor * map_fps_floor`` — 90 it/s at the
    argparse defaults (30 x 3).  Below that the two floors are mutually
    unsatisfiable and the run is pinned to the low tier no matter how it is paced.
    """
    path = run / "timing_summary.json"
    if not path.exists():
        return
    try:
        t = json.loads(path.read_text())
    except json.JSONDecodeError:
        return
    train_s = ((t.get("phases") or {}).get("training")) or t.get("total_s")
    if train_s and report.final_iteration:
        report.iters_per_second = round(report.final_iteration / train_s, 1)
    if report.iters_per_second and report.realized_iters_per_admitted:
        report.implied_map_fps = round(
            report.iters_per_second / report.realized_iters_per_admitted, 3
        )
    itv = report.multires_it_v_floor or 0.0
    mfps = report.multires_map_fps_floor or 0.0
    if itv and mfps and report.iters_per_second:
        report.multires_floor_feasible = report.iters_per_second >= itv * mfps


def _verdicts(report: CadenceReport) -> None:
    req = report.requested_spf
    eff = report.effective_spf
    realized = report.realized_spf_released

    if realized is None or req is None:
        report.spf_tracked = "unknown"
    else:
        def close(a: float, b: float) -> bool:
            return b > 0 and abs(a - b) / b <= 0.15

        if close(realized, req):
            report.spf_tracked = "requested"
        elif eff is not None and close(realized, eff):
            report.spf_tracked = "effective_spf"
        else:
            report.spf_tracked = "neither"

    # Defect (a): min_spf floor overriding an explicit lower --streaming_steps_per_frame.
    if (
        eff is not None
        and req is not None
        and eff != req
        and report.min_spf_floor_fired
    ):
        if report.spf_tracked == "effective_spf":
            report.verdicts.append(
                f"DEFECT-A FIRED: requested spf={req} was overridden by "
                f"effective_spf={eff} (min_spf floor); realized={realized}"
            )
        elif report.spf_tracked == "requested":
            report.verdicts.append(
                f"defect-A printed but NOT applied: effective_spf={eff} logged, "
                f"realized={realized} tracks requested spf={req}"
            )
        else:
            report.verdicts.append(
                f"defect-A: effective_spf={eff} != requested={req}; realized={realized} "
                "matches NEITHER — a third mechanism governs release"
            )

    if report.spf_tracked == "neither":
        speedup = None
        if realized and req:
            speedup = round(req / realized, 1) if realized > 0 else None
        report.verdicts.append(
            "CADENCE ANOMALY: realized release rate matches neither the requested nor "
            f"the scheduler spf (realized={realized} it/frame, requested={req}"
            + (f", {speedup}x faster than configured" if speedup and speedup > 1.2 else "")
            + ")"
        )
        if report.resolved_ingestion_mode and report.resolved_ingestion_mode != "iter_based":
            report.verdicts.append(
                f"  likely cause: ingestion_mode={report.resolved_ingestion_mode} "
                f"(reason={report.ingestion_reason}) — release is NOT iteration-paced; "
                "see issues #278/#302"
            )

    # Defect (b): coverage arithmetic resolving against the FULL registered scene length
    # rather than the --streaming_max_frames subset.  This must be judged from the
    # denominator the run's own "[streaming] Dataset: N frames." line actually printed,
    # NOT from report.json's trajectory_coverage.n_source_frames — the latter reports the
    # registered length regardless of what the arithmetic divided by, so comparing against
    # it produces false positives.
    if not report.requested_max_frames:
        report.verdicts.append("defect-B not applicable (--streaming_max_frames unset)")
    elif report.log_dataset_frames is None:
        report.verdicts.append(
            f"defect-B UNKNOWN: --streaming_max_frames={report.requested_max_frames} set but the "
            "run's '[streaming] Dataset: N frames.' line is missing — denominator not observable"
        )
    elif report.log_dataset_frames > report.requested_max_frames:
        report.verdicts.append(
            f"DEFECT-B FIRED: --streaming_max_frames={report.requested_max_frames} but the "
            f"coverage arithmetic divided by n_frames={report.log_dataset_frames} "
            f"(registry n_source_frames={report.registry_n_source_frames})"
        )
    else:
        report.verdicts.append(
            f"defect-B did not fire: coverage arithmetic used n_frames="
            f"{report.log_dataset_frames}, matching --streaming_max_frames="
            f"{report.requested_max_frames} (registry says "
            f"{report.registry_n_source_frames})"
        )

    # Realized supervision resolution — a clock in its own right.
    if report.multires_it_v_floor or report.multires_map_fps_floor:
        report.verdicts.append(
            "MULTIRES FLOORS NON-ZERO: "
            f"it_v_floor={report.multires_it_v_floor}, "
            f"map_fps_floor={report.multires_map_fps_floor} — this run can silently "
            "demote supervision to 1/N resolution when throughput dips; every "
            "evaluation run should pass "
            "--streaming_multires_it_v_floor 0 --streaming_multires_map_fps_floor 0"
        )
    if report.multires_floor_feasible is False:
        need = (report.multires_it_v_floor or 0) * (report.multires_map_fps_floor or 0)
        report.verdicts.append(
            f"  MULTIRES DUAL FLOOR UNSATISFIABLE: holding it/V>={report.multires_it_v_floor} "
            f"AND map_fps>={report.multires_map_fps_floor} needs >={need:.0f} optimizer it/s; "
            f"this run sustained {report.iters_per_second} it/s "
            f"(realized it/V={report.realized_iters_per_admitted}, "
            f"implied map_fps={report.implied_map_fps}) — low tier was unavoidable"
        )
    if report.high_tier_start_iter and report.low_tier_iterations:
        report.verdicts.append(
            f"SUPERVISION RESOLUTION: {report.low_tier_iterations} optimizer updates ran at "
            f"the LOW multires tier; full resolution only from iter "
            f"{report.high_tier_start_iter}"
        )
        if report.psnr_knee_iter is not None:
            gap = report.psnr_knee_iter - report.high_tier_start_iter
            if abs(gap) <= 2000:
                report.verdicts.append(
                    f"  PSNR knee (+{report.psnr_knee_delta_db} dB at iter "
                    f"{report.psnr_knee_iter}) coincides with the low->high tier switch "
                    f"(gap {gap} iters)"
                )

    if report.accounting_identity_ok is False:
        report.problems.append(
            "admission-log accounting identity FAILED — the log is incomplete, "
            f"realized rates above are untrustworthy ({report.accounting_identity_detail})"
        )

    if report.problems:
        report.ok = False


def analyze(run_dir: Path) -> CadenceReport:
    report = CadenceReport(run_dir=str(run_dir))
    if not run_dir.is_dir():
        report.ok = False
        report.problems.append("not a directory")
        return report

    _parse_cfg_args(run_dir, report)

    log = run_dir / "train.log"
    if log.exists():
        _parse_log(_read_log_after_last_run_start(log, report), report)
    else:
        report.problems.append("train.log missing")

    _parse_admission_log(run_dir, report)
    _parse_trajectory_full(run_dir, report)
    _parse_timing(run_dir, report)
    if (rep := _pick_final_report(run_dir, report)) is not None:
        _parse_report(rep, report)
    _parse_psnr_history(run_dir, report)
    _verdicts(report)
    return report


def _fmt(report: CadenceReport) -> str:
    L: list[str] = []
    a = L.append
    a("=" * 78)
    a(f"RUN  {report.run_dir}")
    a("=" * 78)
    a(f"  banners={report.n_run_start_banners} last={report.last_run_start} host={report.host}")
    a("")
    a("  -- requested ------------------------------------------------------------")
    a(f"  steps_per_frame           : {report.requested_spf}")
    a(f"  min_steps_per_frame       : {report.requested_min_spf}")
    a(f"  ingestion_mode (requested): {report.requested_ingestion_mode}")
    a(f"  online_mode               : {report.online_mode}")
    a(f"  streaming_max_frames      : {report.requested_max_frames}")
    a(f"  multires floors (it_v /   : {report.multires_it_v_floor} / {report.multires_map_fps_floor}")
    a("    map_fps; 0/0 = pinned)")
    a("")
    a("  -- scheduler / registry -------------------------------------------------")
    a(f"  effective_spf             : {report.effective_spf}")
    a(f"  min_spf floor line printed: {report.min_spf_floor_fired}")
    a(f"  'Starting: ...spf=' logged: {report.starting_spf_logged}")
    a(f"  ingestion_mode (resolved) : {report.resolved_ingestion_mode} (reason={report.ingestion_reason})")
    a(f"  scene registered          : {report.scene_registered} "
      f"(n_source_frames={report.registry_n_source_frames}, log said {report.log_dataset_frames})")
    a("")
    a("  -- realized -------------------------------------------------------------")
    a(f"  final iteration           : {report.final_iteration}")
    a(f"  release events            : {report.n_release_events} "
      f"(iters {report.first_release_iter} -> {report.last_release_iter})")
    a(f"  realized it / released fr : {report.realized_spf_released}")
    if report.in_run_cadence is not None:
        a(f"  in-run realized steps/rel : {report.in_run_realized_spf} "
          f"(verdict={report.in_run_cadence_verdict}) [#289 block]")
        a("    (counts differ from the log-derived line above by design: the "
          "in-run clock excludes bootstrap frames and INCLUDES BurstBudget-"
          "dropped and holdout frames; compare the RATIOS, not the raw counts)")
    else:
        a("  in-run realized cadence   : absent (run predates #289); "
          "numbers above are log-derived")
    a(f"  admitted keyframes        : {report.n_admitted} "
      f"(iters {report.first_admit_iter} -> {report.last_admit_iter})")
    a(f"  realized it / admitted kf : {report.realized_iters_per_admitted}")
    a(f"  holdouts withheld         : {report.n_holdout}")
    a(f"  log accounting identity   : {report.accounting_identity_ok} "
      f"({report.accounting_identity_detail})")
    a(f"  admission reasons         : {report.admission_reasons}")
    a("")
    a("  -- insertion ------------------------------------------------------------")
    a(f"  depth insertion enabled   : {report.depth_insertion_enabled}")
    a(f"  total gaussians inserted  : {report.total_inserted} "
      f"({report.inserted_per_admitted} per admitted keyframe)")
    a("")
    a("  -- supervision resolution ------------------------------------------------")
    for s in report.multires_tier_switches:
        a(f"    step={s['iter']:<8} tier={s['tier']:<5} {s['w']}x{s['h']}")
    a(f"  bind stats                : {report.multires_bind_stats}")
    a(f"  optimizer throughput      : {report.iters_per_second} it/s "
      f"(implied map_fps {report.implied_map_fps}); dual floor satisfiable="
      f"{report.multires_floor_feasible}")
    a("")
    a("  -- quality curve ---------------------------------------------------------")
    a(f"  test psnr first/last      : {report.psnr_first} -> {report.psnr_last}")
    a(f"  largest 1-checkpoint jump : +{report.psnr_knee_delta_db} dB at iter {report.psnr_knee_iter}")
    a(f"  final mean_test_psnr      : {report.mean_test_psnr} (n_test={report.n_test}, eval_valid={report.eval_valid})")
    a(f"  seed_fade                 : {report.seed_fade}")
    a("")
    a("  -- verdicts --------------------------------------------------------------")
    a(f"  realized spf tracks       : {report.spf_tracked}")
    for v in report.verdicts:
        a(f"  * {v}")
    for p in report.problems:
        a(f"  ! {p}")
    return "\n".join(L)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dirs", nargs="+", type=Path)
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = ap.parse_args(argv)

    reports = [analyze(d) for d in args.run_dirs]
    if args.json:
        print(json.dumps([asdict(r) for r in reports], indent=2, default=str))
    else:
        for r in reports:
            print(_fmt(r))
            print()
    return 0 if all(r.ok for r in reports) else 1


if __name__ == "__main__":
    sys.exit(main())
