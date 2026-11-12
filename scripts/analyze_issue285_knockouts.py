"""Provenance-gated, curve-level comparison of the ``issue285knock_*`` arms.

Issue #285 asks whether the *global* opacity/effective-count regularizers
(``--lambda_opacity_entropy`` / ``--lambda_eff_count``) are causal drivers of
seed fade, via matched knockouts. This script is the read-only analysis half:
it takes a set of finished run directories and emits (a) a provenance report,
(b) an operating-point comparability gate, and (c) full *curve* comparisons --
never an endpoint-only delta.

Why every step here is defensive: this issue's thread has already carried two
public retractions (a depth-scale defect, and a run-root glob error). So:

* ``comparison/iter_*`` is enumerated **numerically**, never lexically, and the
  ``2147483647`` online-mode sentinel is resolved from ``train.log``'s
  ``[FINAL STEP N]`` banner (same rule as ``scripts/local_quality_from_run.py``).
* ``report.json`` lives at ``<run>/comparison/iter_N/report.json``. A run-root
  glob finding nothing is a glob error, not a missing report.
* Every arm is counted for ``[RUN START]`` banners; artifacts are attributed to
  the LAST banner's ``model_path`` by comparing ``report.json["output_dir"]``.
  ``issue285knock_A0`` really does have two banners (SIGTERM at step 17130,
  relaunch into the same host dir), so its directory genuinely mixes two
  container run roots.
* Every arm is defined from its own ``reproduce.sh`` / ``cfg_args``, never from
  the queue spec -- per #285's own last comment, the queue spec lies about what
  ``--streaming_seed_fade_disabled`` does (it disables the suppression
  CONTROLLER, not the MONITOR).
* ``holdout_hash`` (selection rule) AND per-checkpoint ``realized_holdout_hash``
  and ``n_test`` are all reported. ``psnr_history``'s ``mean_test_psnr`` is a
  MOVING-BASKET curve for a streaming run -- ``n_test`` grows as frames are
  admitted -- so curves are only compared on iterations where ``n_test`` agrees
  across the arms being compared.

Usage::

    python scripts/analyze_issue285_knockouts.py \
        output/scratch/issue285knock_A0 \
        output/scratch/issue285knock_A0R \
        output/scratch/issue285knock_M0 \
        output/scratch/issue285knock_K3 \
        --baseline issue285knock_A0 \
        -o /tmp/issue285_analysis

Read-only: it never launches, stops or requeues anything, and never touches a
GPU. Safe to run against a directory whose run is still in flight (an
unfinished arm simply reports fewer checkpoints and is flagged INCOMPLETE).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Dict, List, Optional, Tuple

from eligsir.plot_style import svg_output_path

# online_mode writes this in place of a real iteration.
# See the [[online-mode-report-iteration-sentinel-bug]] memory.
ONLINE_SENTINEL = 2147483647

# Flags whose value defines a knockout arm for this issue. Anything else that
# differs between arms is reported as an UNEXPECTED difference (i.e. a confound).
KNOCKOUT_FLAGS = (
    "--lambda_eff_count",          # K1, K3
    "--lambda_opacity_entropy",    # K2, K3
    "--sparse_energy_global_interval",  # Dplus (positive control, 500x dose)
    "--streaming_count_loss_local",     # K4 (local-vs-global framing)
    "--opacity_reg",
    "--scale_reg",
    "--seed",                      # A0R, A0R2 (replicates)
    # M0 is the multires bridge arm: it deliberately leaves the mandatory
    # multires floors at their defaults instead of zeroing them (see the
    # analysis doc's "M0" section). Without these entries the confound
    # detector flags M0's own defining difference as UNEXPECTED, which is
    # the same false positive fa94ab5 already fixed once for Dplus/K4.
    "--streaming_multires_it_v_floor",   # M0
    "--streaming_multires_map_fps_floor",  # M0
)


# ---------------------------------------------------------------- provenance


def run_start_banners(run_dir: str) -> List[Tuple[int, str]]:
    """[(line_no, model_path)] for every ``[RUN START]`` banner in train.log.

    More than one means the host directory mixes attempts: only content after
    the LAST banner counts (Run Contract, "Unique output folder per run").
    """
    log = os.path.join(run_dir, "train.log")
    out: List[Tuple[int, str]] = []
    if not os.path.exists(log):
        return out
    for i, line in enumerate(open(log, errors="ignore"), start=1):
        if "[RUN START]" in line:
            m = re.search(r"model_path=(\S+)", line)
            out.append((i, m.group(1) if m else ""))
    return out


def resolve_final_iteration(run_dir: str) -> Optional[int]:
    """Real final optimisation step from train.log's own banner.

    Never trust ``report.json["iteration"]`` for an online_mode run -- it is
    ONLINE_SENTINEL, not an iteration.
    """
    log = os.path.join(run_dir, "train.log")
    if not os.path.exists(log):
        return None
    best = None
    for line in open(log, errors="ignore"):
        m = re.search(r"\[FINAL (?:STEP|ITER) (\d+)\]", line)
        if m:
            best = int(m.group(1))
    return best


def discover_checkpoints(run_dir: str) -> Tuple[List[Tuple[int, str]], List[str]]:
    """Numerically-sorted [(iteration, report_path)], plus skipped dir names.

    ``*_bootstrap_views`` dirs are DEBUG artifacts (degenerate_train_views,
    mean_view_psnr over TRAIN cameras) and are never a test result -- dropped.
    """
    comp = os.path.join(run_dir, "comparison")
    ticks: List[Tuple[int, str]] = []
    skipped: List[str] = []
    if not os.path.isdir(comp):
        return ticks, ["comparison/ missing"]
    final = resolve_final_iteration(run_dir)
    for name in os.listdir(comp):
        m = re.fullmatch(r"iter_(\d+)", name)
        if not m:
            skipped.append(name)
            continue
        it = int(m.group(1))
        if it == ONLINE_SENTINEL:
            if final is None:
                skipped.append(f"{name} (sentinel, no [FINAL STEP] banner to resolve it)")
                continue
            it = final
        rep = os.path.join(comp, name, "report.json")
        if not os.path.exists(rep):
            skipped.append(f"{name} (no report.json)")
            continue
        ticks.append((it, rep))
    ticks.sort(key=lambda t: t[0])  # NUMERIC, never lexical
    return ticks, skipped


def _output_dir_matches(report_output_dir: str, last_model_path: str) -> Optional[bool]:
    """Does this report belong to the LAST attempt?

    The ablator stages each attempt under a per-attempt container root
    (``…/issue285knock_A0-<hash>-1``); a requeue into the same host dir leaves
    both attempts' files side by side. ``None`` = cannot tell.
    """
    if not report_output_dir or not last_model_path:
        return None
    return os.path.normpath(report_output_dir).startswith(os.path.normpath(last_model_path))


# --------------------------------------------------------------- arm defs


def arm_command(run_dir: str) -> Optional[str]:
    """The run's own ``train.py`` command line, from its reproduce.sh."""
    path = os.path.join(run_dir, "reproduce.sh")
    if not os.path.exists(path):
        return None
    for line in open(path, errors="ignore"):
        s = line.strip()
        if s.startswith("python train.py"):
            return s
    return None


def parse_flags(cmd: str) -> Dict[str, str]:
    """--flag value / --flag -> {flag: value or ''}. Order-insensitive."""
    toks = cmd.split()
    flags: Dict[str, str] = {}
    i = 0
    while i < len(toks):
        t = toks[i]
        if t.startswith("--"):
            if i + 1 < len(toks) and not toks[i + 1].startswith("--"):
                flags[t] = toks[i + 1]
                i += 2
            else:
                flags[t] = ""
                i += 1
        else:
            i += 1
    return flags


def cfg_value(run_dir: str, key: str) -> Optional[str]:
    """One value out of the run's own cfg_args Namespace snapshot."""
    path = os.path.join(run_dir, "cfg_args")
    if not os.path.exists(path):
        return None
    txt = open(path, errors="ignore").read()
    m = re.search(rf"\b{re.escape(key)}=([^,)\s]+)", txt)
    return m.group(1).strip().strip("'\"") if m else None


def mapping_scheduler_line(run_dir: str, after_line: int = 0) -> Optional[str]:
    """The run's own ``[MappingScheduler]`` startup line, after the LAST banner.

    Part of the operating-point gate: W3-38 showed ``effective_spf`` can
    silently override an explicit ``--streaming_steps_per_frame``, so two arms
    launched from identical recipes can differ in real frame cadence.
    """
    log = os.path.join(run_dir, "train.log")
    if not os.path.exists(log):
        return None
    for i, line in enumerate(open(log, errors="ignore"), start=1):
        if i <= after_line:
            continue
        if "[MappingScheduler]" in line:
            # Strip the trailing "[dd/mm HH:MM:SS]" stamp: it differs between
            # arms by construction and would fake a gate mismatch.
            return re.sub(r"\s*\[\d\d/\d\d \d\d:\d\d:\d\d\]\s*$", "", line.strip())
    return None


# ------------------------------------------------------------------ curves


def psnr_history_from_log(run_dir: str, after_line: int = 0) -> List[dict]:
    """[(iteration, mean_test_psnr, n_test)] from train.log, after a line.

    Preferred over ``psnr_history.jsonl`` on a multi-attempt run: the jsonl
    carries no attempt stamp, whereas a log line's position does. ``n_test`` is
    the basket size at that tick -- the field that makes this a MOVING-basket
    curve on a streaming run.
    """
    log = os.path.join(run_dir, "train.log")
    rows: List[dict] = []
    if not os.path.exists(log):
        return rows
    pat = re.compile(
        r"\[psnr_history\] iter=(\d+) mean_test_psnr=([\d.]+)dB \(n_test=(\d+)\)"
    )
    for i, line in enumerate(open(log, errors="ignore"), start=1):
        if i <= after_line:
            continue
        for m in pat.finditer(line):
            rows.append(
                {
                    "iteration": int(m.group(1)),
                    "mean_test_psnr": float(m.group(2)),
                    "n_test": int(m.group(3)),
                }
            )
    rows.sort(key=lambda r: r["iteration"])
    return rows


def psnr_history_jsonl(run_dir: str) -> List[dict]:
    path = os.path.join(run_dir, "psnr_history.jsonl")
    rows: List[dict] = []
    if not os.path.exists(path):
        return rows
    for line in open(path, errors="ignore"):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    rows.sort(key=lambda r: r.get("iteration", 0))
    return rows


def last_event_file(run_dir: str) -> Optional[str]:
    """The most recent ``events.out.tfevents.*`` -- i.e. the LAST attempt's.

    A requeued arm leaves one event file per attempt in the same host dir; the
    filename's embedded unix timestamp orders them.
    """
    cands = []
    for name in os.listdir(run_dir):
        m = re.match(r"events\.out\.tfevents\.(\d+)\.", name)
        if m:
            cands.append((int(m.group(1)), os.path.join(run_dir, name)))
    if not cands:
        return None
    cands.sort()
    return cands[-1][1]


def tb_scalars(event_path: str, tags: List[str]) -> Dict[str, List[Tuple[int, float]]]:
    """{tag: [(step, value)]} from one events file. Missing tags -> []."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    ea = EventAccumulator(event_path, size_guidance={"scalars": 0})
    ea.Reload()
    have = set(ea.Tags().get("scalars", []))
    out: Dict[str, List[Tuple[int, float]]] = {}
    for t in tags:
        out[t] = [(e.step, float(e.value)) for e in ea.Scalars(t)] if t in have else []
    return out


# --------------------------------------------------------- curve comparison


def align(a: List[dict], b: List[dict], key: str = "mean_test_psnr") -> List[dict]:
    """Inner-join two curves on iteration, carrying both n_test values."""
    bi = {r["iteration"]: r for r in b}
    out = []
    for r in a:
        o = bi.get(r["iteration"])
        if o is None:
            continue
        out.append(
            {
                "iteration": r["iteration"],
                "a": r[key],
                "b": o[key],
                "delta": r[key] - o[key],
                "n_test_a": r.get("n_test"),
                "n_test_b": o.get("n_test"),
                "basket_match": r.get("n_test") == o.get("n_test"),
            }
        )
    return out


def classify_curve(joined: List[dict], noise_db: float) -> dict:
    """superimposable / converges / diverges / mixed, over the aligned grid.

    ``noise_db`` is the empirical replicate floor (|A0 - A0R|); an arm is only
    said to differ where its deviation exceeds it. Mirrors the semantics of
    ``report.json["baseline_comparison"]["curve_shape"]``.
    """
    if len(joined) < 4:
        return {"curve_shape": "insufficient_overlap", "n_points": len(joined)}
    matched = [j for j in joined if j["basket_match"]]
    deltas = [j["delta"] for j in joined]
    n = len(deltas)
    first = deltas[: max(1, n // 3)]
    last = deltas[-max(1, n // 3):]
    mean_first = sum(abs(d) for d in first) / len(first)
    mean_last = sum(abs(d) for d in last) / len(last)
    max_abs = max(abs(d) for d in deltas)
    if max_abs <= noise_db:
        shape = "superimposable"
    elif mean_last > mean_first + noise_db:
        shape = "diverges"
    elif mean_first > mean_last + noise_db:
        shape = "converges"
    else:
        shape = "mixed"
    return {
        "curve_shape": shape,
        "n_points": n,
        "n_points_basket_matched": len(matched),
        "basket_matched_fraction": round(len(matched) / n, 3),
        "noise_floor_db": round(noise_db, 3),
        "max_abs_delta_db": round(max_abs, 3),
        "mean_abs_delta_first_third_db": round(mean_first, 3),
        "mean_abs_delta_last_third_db": round(mean_last, 3),
        "final_delta_db": round(deltas[-1], 3),
    }


# ------------------------------------------------------------------- per-arm

SEED_TAGS = [
    "optimization/seed_psnr",
    "streaming/seed/mean_opacity",
    "streaming/seed/alive_count",
    "streaming/seed_cohort/mean_opacity",
    "streaming/seed_cohort/alive_count",
    "streaming/seed_cohort/population_fraction",
    "streaming/seed_fade/running_max_db",
    "streaming/seed_fade/gap_db",
]

# DOWNSTREAM-EFFECT CHECK. A null knockout is more informative if the knob is
# seen to move SOMETHING downstream of itself. But note the limit, and do not
# oversell this: none of these tags is the energy loss's own magnitude (no such
# TB scalar exists), and they are cumulative/derived counters. Movement here
# shows the population evolved differently -- it does NOT establish that the
# term's gradient contribution was large enough to matter. "The terms are inert
# at this operating point" is ruled out only by the pre-registered `Dplus`
# positive control (500x dose), never by this check.
MANIPULATION_TAGS = [
    "mcmc/dead_total",
    "streaming/dead_fraction",
    "streaming/gaussians",
    "streaming/seed/mean_opacity",
    "streaming/seed_cohort/mean_opacity",
]

# Iterations at which every cross-arm table is sampled. Fixed, not derived from
# any one arm, so adding an arm cannot silently move the comparison grid.
SAMPLE_GRID = (10000, 25000, 50000, 75000, 100000)


def multires_full_res_from(run_dir: str, after_line: int = 0) -> Tuple[int, List[str]]:
    """(iteration from which the run renders at FULL resolution, transitions).

    ``optimization/seed_psnr`` is computed on the CURRENT multires tier, so an
    arm that trains most of its run at ``tier=low cam_h=120 cam_w=160`` is not
    measuring the same quantity as one pinned to ``tier=high``. Comparing their
    seed_psnr peaks/minima across the switch mixes two resolutions -- the
    ``--streaming_multires_*_floor 0`` flags (which pin an arm to high) are
    exactly what ``M0`` drops, so this is load-bearing, not hypothetical.

    Returns 0 when the run never leaves the high tier (nothing to exclude).
    """
    log = os.path.join(run_dir, "train.log")
    transitions: List[str] = []
    if not os.path.exists(log):
        return 0, transitions
    pat = re.compile(r"\[multires\] step=(\d+) tier=(\w+) cam_h=(\d+) cam_w=(\d+)")
    last_low_end = 0
    saw_low = False
    for i, line in enumerate(open(log, errors="ignore"), start=1):
        if i <= after_line:
            continue
        for m in pat.finditer(line):
            step, tier = int(m.group(1)), m.group(2)
            transitions.append(f"step={step} tier={tier} {m.group(4)}x{m.group(3)}")
            if tier != "high":
                saw_low = True
            elif saw_low:
                last_low_end = step
    return last_low_end, transitions


def summarize_series(series: List[Tuple[int, float]], warmup: int = 2000,
                     start: int = 0) -> dict:
    """Absolute post-warmup descriptors -- #285's amended PRIMARY metric.

    ``max_dip_db`` is deliberately NOT used as a primary: it is running-max
    relative, so an arm that never gets good early scores a small dip while
    being catastrophically worse, and an arm that IMPROVES early quality raises
    its own running max and can score a LARGER dip while being strictly better
    (#285's own amendment, from the #346 champ_asis vs issue278mode inversion).
    """
    lo = max(warmup, start)
    post = [(s, v) for s, v in series if s >= lo]
    if not post:
        return {"n": 0, "window_from": lo}
    vals = [v for _, v in post]
    smax = max(post, key=lambda t: t[1])
    smin = min(post, key=lambda t: t[1])
    return {
        "n": len(post),
        "window_from": lo,
        "first": round(vals[0], 4),
        "final": round(vals[-1], 4),
        "peak": round(smax[1], 4),
        "peak_iter": smax[0],
        "min": round(smin[1], 4),
        "min_iter": smin[0],
        "final_minus_peak": round(vals[-1] - smax[1], 4),
    }


def analyze_arm(run_dir: str) -> dict:
    name = os.path.basename(os.path.normpath(run_dir))
    banners = run_start_banners(run_dir)
    last_banner_line = banners[-1][0] if banners else 0
    last_model_path = banners[-1][1] if banners else ""
    final_iter = resolve_final_iteration(run_dir)
    ticks, skipped = discover_checkpoints(run_dir)

    checkpoints = []
    for it, rep_path in ticks:
        d = json.load(open(rep_path))
        em = d.get("eval_metadata") or {}
        sf = d.get("seed_fade") or {}
        tc = d.get("trajectory_coverage") or {}
        checkpoints.append(
            {
                "iteration": it,
                "report_path": rep_path,
                "output_dir": d.get("output_dir"),
                "belongs_to_last_attempt": _output_dir_matches(
                    d.get("output_dir", ""), last_model_path
                ),
                "eval_valid": d.get("eval_valid"),
                "split_kind": d.get("split_kind"),
                "n_test": d.get("n_test"),
                "n_train": d.get("n_train"),
                "mean_test_psnr": d.get("mean_test_psnr"),
                "min_test_psnr": d.get("min_test_psnr"),
                "holdout_hash": em.get("holdout_hash"),
                "realized_holdout_hash": em.get("realized_holdout_hash"),
                "scene_fingerprint": d.get("scene_fingerprint"),
                "resumed_from_iteration": d.get("resumed_from_iteration"),
                "seed_fade": {
                    k: sf.get(k)
                    for k in (
                        "detected",
                        "status",
                        "severity",
                        "severity_reasons",
                        "confounded",
                        "indeterminate",
                        "indeterminate_reasons",
                        "suppression_confounded",
                        "suppression_applied",
                        "observe_only",
                        "max_dip_db",
                        "running_max_db",
                        "first_fade_start_iter",
                        "fade_start_iter",
                        "final_psnr",
                        "min_psnr",
                        "curve_shape",
                        "episode_count",
                        "n_samples",
                    )
                },
                "seed_fade_confounded": d.get("seed_fade_confounded"),
                "seed_fade_indeterminate": d.get("seed_fade_indeterminate"),
                "frames_admitted": tc.get("frames_admitted"),
                "coverage_fraction": tc.get("coverage_fraction"),
                "frame_admission": tc.get("frame_admission"),
                "verdict": tc.get("verdict"),
                "map_reinit_invalidated": tc.get("map_reinit_invalidated"),
            }
        )

    cmd = arm_command(run_dir)
    ev = last_event_file(run_dir)
    all_tags = list(dict.fromkeys(list(SEED_TAGS) + list(MANIPULATION_TAGS)))
    tb = tb_scalars(ev, all_tags) if ev else {}

    return {
        "arm": name,
        "run_dir": run_dir,
        "complete": os.path.exists(os.path.join(run_dir, ".COMPLETE")),
        "n_run_start_banners": len(banners),
        "run_start_banners": [{"line": l, "model_path": p} for l, p in banners],
        "last_banner_line": last_banner_line,
        "final_iteration": final_iter,
        "command": cmd,
        "flags": parse_flags(cmd) if cmd else {},
        "cfg_seed_fade_disabled": cfg_value(run_dir, "streaming_seed_fade_disabled"),
        "cfg_seed_fade_observe_only": cfg_value(run_dir, "streaming_seed_fade_observe_only"),
        "cfg_lambda_eff_count": cfg_value(run_dir, "lambda_eff_count"),
        "cfg_lambda_opacity_entropy": cfg_value(run_dir, "lambda_opacity_entropy"),
        "mapping_scheduler_line": mapping_scheduler_line(run_dir, last_banner_line),
        "multires_full_res_from": multires_full_res_from(run_dir, last_banner_line)[0],
        "multires_transitions": multires_full_res_from(run_dir, last_banner_line)[1],
        "skipped_comparison_dirs": skipped,
        "checkpoints": checkpoints,
        "psnr_curve_from_log": psnr_history_from_log(run_dir, last_banner_line),
        "psnr_curve_from_jsonl": psnr_history_jsonl(run_dir),
        "event_file": ev,
        # A requeue into an occupied model_path is NOT automatically a resume.
        # If the ablator relaunched from scratch (new container run root, fresh
        # event file, fresh psnr_history), the last attempt is a complete,
        # self-contained run and its seed-fade verdict spans that whole run --
        # NOT `indeterminate`. The distinguishing evidence is the first step in
        # the LAST attempt's own event file: near-zero means a fresh start,
        # mid-run means a genuine resume (which WOULD cold-start
        # SeedFadeMonitor._running_max pre-#405).
        "last_attempt_first_tb_step": (
            min((s for s, _ in (tb.get("optimization/seed_psnr") or [])), default=None)
        ),
        "tb_series": {k: v for k, v in tb.items()},
        "tb_summary": {k: summarize_series(v) for k, v in tb.items()},
        # Sampled on a FIXED grid so the cross-arm table cannot silently move
        # when an arm is added or its logging cadence differs.
        "tb_sampled": {
            k: {str(g): dict(v).get(g) for g in SAMPLE_GRID} for k, v in tb.items()
        },
    }


# --------------------------------------------------------------------- main


def build_report(run_dirs: List[str], baseline: Optional[str],
                 resolution_matched_from: Optional[int] = None) -> dict:
    arms = {}
    for d in run_dirs:
        arms[os.path.basename(os.path.normpath(d))] = analyze_arm(d)

    base_name = baseline or sorted(arms)[0]
    if base_name not in arms:
        raise SystemExit(f"baseline {base_name!r} not among {sorted(arms)}")
    base = arms[base_name]

    # --- arm definition diff (from each run's OWN reproduce.sh) -------------
    base_flags = base["flags"]
    defs = {}
    for name, a in arms.items():
        if name == base_name:
            defs[name] = {"differs_from_baseline": {}, "unexpected_differences": {}}
            continue
        diff = {}
        for k in set(base_flags) | set(a["flags"]):
            bv, av = base_flags.get(k), a["flags"].get(k)
            if bv != av:
                diff[k] = {"baseline": bv, "arm": av}
        # model_path/baseline_run always differ by construction; not a confound.
        for k in ("-m", "--model_path", "--baseline_run"):
            diff.pop(k, None)
        expected = {k: v for k, v in diff.items() if k in KNOCKOUT_FLAGS}
        unexpected = {k: v for k, v in diff.items() if k not in KNOCKOUT_FLAGS}
        defs[name] = {
            "differs_from_baseline": expected,
            "unexpected_differences": unexpected,
        }

    # --- operating-point gate, checked BEFORE any PSNR is read -------------
    def final_ck(a):
        cks = [c for c in a["checkpoints"] if c["belongs_to_last_attempt"] is not False]
        return cks[-1] if cks else None

    gate = {}
    bfin = final_ck(base)
    for name, a in arms.items():
        fin = final_ck(a)
        if fin is None or bfin is None:
            gate[name] = {"comparable": None, "reason": "no attributable final report"}
            continue
        checks = {
            "holdout_hash": (fin["holdout_hash"], bfin["holdout_hash"]),
            "scene_fingerprint": (fin["scene_fingerprint"], bfin["scene_fingerprint"]),
            "realized_holdout_hash": (
                fin["realized_holdout_hash"],
                bfin["realized_holdout_hash"],
            ),
            "n_test": (fin["n_test"], bfin["n_test"]),
            "frames_admitted": (fin["frames_admitted"], bfin["frames_admitted"]),
            "mapping_scheduler_line": (
                a["mapping_scheduler_line"],
                base["mapping_scheduler_line"],
            ),
        }
        mismatches = {k: {"arm": v[0], "baseline": v[1]} for k, v in checks.items() if v[0] != v[1]}
        # holdout_hash/scene_fingerprint are HARD gates; the rest are reported
        # as caveats (they legitimately drift with an arm's own dynamics).
        hard = [k for k in ("holdout_hash", "scene_fingerprint") if k in mismatches]
        gate[name] = {
            "comparable": not hard,
            "hard_mismatches": hard,
            "mismatches": mismatches,
            "eval_valid": fin["eval_valid"],
            "n_test": fin["n_test"],
            "final_iteration": a["final_iteration"],
        }

    # --- replicate noise floor from the identical-config replicates --------
    # #285's amended rule: the effect threshold is the RANGE spanned by the
    # identical-config replicates (A0/A0R/A0R2), not a 3-sample sigma.
    replicates = [
        n for n, d in defs.items()
        if n != base_name
        and set(d["differs_from_baseline"]) <= {"--seed"}
        and not d["unexpected_differences"]
    ]
    noise = {"replicate_arms": replicates}
    noise_db = 0.0
    for r in replicates:
        j = align(arms[r]["psnr_curve_from_log"], base["psnr_curve_from_log"])
        if j:
            m = max(abs(x["delta"]) for x in j)
            noise[r] = {
                "n_aligned": len(j),
                "max_abs_delta_db": round(m, 3),
                "n_basket_matched": sum(1 for x in j if x["basket_match"]),
            }
            noise_db = max(noise_db, m)
    noise["noise_floor_db"] = round(noise_db, 3)
    noise["n_replicates"] = len(replicates)
    noise["sufficient"] = len(replicates) >= 2  # #285 pre-registered THREE total

    # --- curve comparisons (never endpoint-only) ---------------------------
    curves = {}
    for name, a in arms.items():
        if name == base_name:
            continue
        j = align(a["psnr_curve_from_log"], base["psnr_curve_from_log"])
        c = classify_curve(j, noise_db if noise_db > 0 else 0.5)
        c["aligned"] = j
        # Same treatment for the amended PRIMARY metric, absolute seed_psnr.
        sa = a["tb_series"].get("optimization/seed_psnr") or []
        sb = base["tb_series"].get("optimization/seed_psnr") or []
        c["seed_psnr_summary_arm"] = summarize_series(sa)
        c["seed_psnr_summary_baseline"] = summarize_series(sb)
        curves[name] = c

    # --- resolution-matched re-summary ------------------------------------
    # seed_psnr is measured on the CURRENT multires tier. If any arm spent part
    # of its run at the low tier, only the window in which EVERY arm is at full
    # resolution is comparable; the naive post-warmup summary above spans the
    # switch and must not be quoted across arms.
    # REPRODUCIBILITY TRAP: this max is over the arms SUPPLIED on the command
    # line. Drop the one low-tier arm and the window silently collapses to 0,
    # turning the resolution-matched table back into the naive one. Pass
    # --resolution-matched-from to pin it when comparing a subset.
    common_from = (
        resolution_matched_from
        if resolution_matched_from is not None
        else max(a["multires_full_res_from"] for a in arms.values())
    )
    for a in arms.values():
        a["resolution_matched_from"] = common_from
        a["tb_summary_resolution_matched"] = {
            k: summarize_series(v, start=common_from) for k, v in a["tb_series"].items()
        }

    # --- downstream-effect check ------------------------------------------------
    # For each non-replicate arm and each tag the knob is supposed to drive:
    # does the arm fall OUTSIDE the band spanned by the identical-config
    # replicates? An arm that is outside on the manipulation tags but inside on
    # the outcome tags is a null WITH a passing manipulation check -- a much
    # stronger result than a bare null, because "the term was inert" is ruled
    # out by the arm's own telemetry rather than only by the Dplus control.
    manip = {}
    band_arms = [base_name] + replicates
    for name, a in arms.items():
        if name in band_arms:
            continue
        per_tag = {}
        for tag in MANIPULATION_TAGS:
            pts = {}
            for g in SAMPLE_GRID:
                band = [
                    arms[b]["tb_sampled"].get(tag, {}).get(str(g)) for b in band_arms
                ]
                band = [v for v in band if v is not None]
                val = a["tb_sampled"].get(tag, {}).get(str(g))
                if val is None or not band:
                    continue
                lo, hi = min(band), max(band)
                pts[str(g)] = {
                    "arm": round(val, 5),
                    "replicate_band": [round(lo, 5), round(hi, 5)],
                    "outside_band": val < lo or val > hi,
                    # How many band-widths away -- a band of width ~0 makes
                    # this large, which is the point: the replicates agree to
                    # 4 decimals on these tags.
                    "excursion_abs": round(min(abs(val - lo), abs(val - hi)), 5)
                    if (val < lo or val > hi)
                    else 0.0,
                }
            per_tag[tag] = pts
        moved = [
            t for t, pts in per_tag.items()
            if any(p["outside_band"] for p in pts.values())
        ]
        manip[name] = {
            "tags_moved_outside_replicate_band": moved,
            # Deliberately NOT "PASS": see MANIPULATION_TAGS' docstring. The
            # strongest honest reading is "the knob moved something downstream",
            # which is compatible with the term still being too weak to matter.
            "downstream_effect": "MOVED" if moved else "NOT_DETECTED",
            "detail": per_tag,
        }

    # --- pre-registered verdict, graded by code not by eye ------------------
    # #285's rule: an arm counts as a real effect ONLY IF its co-primary values
    # fall outside the range spanned by the identical-config replicates AND its
    # curve_shape vs the baseline is not 'superimposable'. Both halves are
    # required; this grades them together so a future arm cannot be judged by
    # reading a table (which is how a 0.18 dB excursion was missed by hand).
    verdicts = {}
    for name, a in arms.items():
        if name in band_arms:
            continue
        excursions = {}
        for desc in ("peak", "min", "final"):
            band = []
            for b in band_arms:
                s = (arms[b].get("tb_summary_resolution_matched") or {}).get(
                    "optimization/seed_psnr"
                ) or {}
                if s.get("n"):
                    band.append(s[desc])
            s = (a.get("tb_summary_resolution_matched") or {}).get(
                "optimization/seed_psnr"
            ) or {}
            if not band or not s.get("n"):
                continue
            lo, hi = min(band), max(band)
            v = s[desc]
            out = v < lo or v > hi
            excursions[f"seed_psnr_resmatched_{desc}"] = {
                "arm": v,
                "replicate_band": [lo, hi],
                "outside_band": out,
                "excursion_db": round(min(abs(v - lo), abs(v - hi)), 4) if out else 0.0,
                "direction": ("below" if v < lo else "above") if out else "inside",
            }
        # max_dip_db: SECONDARY descriptor only (running-max relative, can
        # invert sign vs absolute quality) -- graded, but never decisive.
        def _dip(arm_dict):
            cks = [c for c in arm_dict["checkpoints"]
                   if c["belongs_to_last_attempt"] is not False]
            return cks[-1]["seed_fade"].get("max_dip_db") if cks else None

        band = [_dip(arms[b]) for b in band_arms]
        band = [v for v in band if v is not None]
        v = _dip(a)
        if band and v is not None:
            lo, hi = min(band), max(band)
            out = v < lo or v > hi
            excursions["max_dip_db_SECONDARY"] = {
                "arm": v,
                "replicate_band": [lo, hi],
                "outside_band": out,
                "excursion_db": round(min(abs(v - lo), abs(v - hi)), 4) if out else 0.0,
                "direction": ("below" if v < lo else "above") if out else "inside",
            }

        shape = (curves.get(name) or {}).get("curve_shape")
        n_out = sum(
            1 for k, e in excursions.items()
            if e["outside_band"] and not k.endswith("_SECONDARY")
        )
        verdicts[name] = {
            "curve_shape_vs_baseline": shape,
            "n_coprimary_descriptors_outside_band": n_out,
            "max_excursion_db": max(
                [e["excursion_db"] for e in excursions.values()] or [0.0]
            ),
            # BOTH halves required. A superimposable curve fails the rule no
            # matter how many descriptors sit outside a narrow replicate band.
            "pre_registered_effect": bool(n_out) and shape != "superimposable",
            "replicate_band_arms": band_arms,
            "band_provisional": len(replicates) < 2,
            "excursions": excursions,
        }

    return {
        # Underscore-prefixed: in-memory only (raw TB series, for plotting).
        # Stripped before analysis.json is written -- it would dominate the file.
        "_arms_raw": arms,
        "baseline": base_name,
        "replicate_band_arms": band_arms,
        "manipulation_check": manip,
        "arms": {
            n: {k: v for k, v in a.items() if k not in ("tb_series",)}
            for n, a in arms.items()
        },
        "arm_definitions": defs,
        "operating_point_gate": gate,
        "replicate_noise_floor": noise,
        "pre_registered_verdict": verdicts,
        "curve_comparisons": {
            n: {k: v for k, v in c.items() if k != "aligned"} for n, c in curves.items()
        },
        "aligned_curves": {n: c["aligned"] for n, c in curves.items()},
    }


def print_text(rep: dict) -> None:
    w = sys.stdout.write
    w("\n=== PROVENANCE ===\n")
    w(f"{'arm':<26}{'done':<6}{'banners':<9}{'final_it':<10}{'ckpts':<7}attributable\n")
    for n, a in rep["arms"].items():
        att = sum(1 for c in a["checkpoints"] if c["belongs_to_last_attempt"] is not False)
        w(f"{n:<26}{str(a['complete']):<6}{a['n_run_start_banners']:<9}"
          f"{str(a['final_iteration']):<10}{len(a['checkpoints']):<7}{att}\n")

    w("\n=== ARM DEFINITION (from each run's own reproduce.sh) ===\n")
    for n, d in rep["arm_definitions"].items():
        if n == rep["baseline"]:
            w(f"{n:<26}BASELINE\n")
            continue
        w(f"{n:<26}{d['differs_from_baseline'] or '{}'}\n")
        if d["unexpected_differences"]:
            w(f"{'':<26}  !! UNEXPECTED (confound): {d['unexpected_differences']}\n")

    w("\n=== OPERATING-POINT GATE (checked before reading any PSNR) ===\n")
    for n, g in rep["operating_point_gate"].items():
        w(f"{n:<26}comparable={g.get('comparable')} eval_valid={g.get('eval_valid')} "
          f"n_test={g.get('n_test')}\n")
        for k, v in (g.get("mismatches") or {}).items():
            w(f"{'':<26}  differs: {k}: arm={str(v['arm'])[:60]} base={str(v['baseline'])[:60]}\n")

    nf = rep["replicate_noise_floor"]
    w(f"\n=== REPLICATE NOISE FLOOR ===\n")
    w(f"replicates={nf['replicate_arms']} n={nf['n_replicates']} "
      f"sufficient={nf['sufficient']} floor={nf['noise_floor_db']} dB\n")

    w("\n=== CURVE COMPARISON vs baseline (moving-basket mean_test_psnr) ===\n")
    for n, c in rep["curve_comparisons"].items():
        w(f"{n:<26}{c.get('curve_shape'):<18}max|d|={c.get('max_abs_delta_db')}dB "
          f"final_d={c.get('final_delta_db')}dB pts={c.get('n_points')} "
          f"basket_matched={c.get('basket_matched_fraction')}\n")

    resfrom = next(iter(rep["arms"].values())).get("resolution_matched_from", 0)
    w("\n=== PRIMARY METRIC: absolute optimization/seed_psnr ===\n")
    w("(A) post-warmup, NOT resolution-matched -- do NOT compare across arms\n"
      "    whose multires transitions differ (see the transitions column):\n")
    w(f"{'arm':<26}{'peak':<9}{'peak@':<9}{'min':<9}{'min@':<9}{'final':<9}"
      f"{'final-peak':<12}full_res_from\n")
    for n, a in rep["arms"].items():
        s = (a.get("tb_summary") or {}).get("optimization/seed_psnr") or {}
        if not s.get("n"):
            w(f"{n:<26}(no series)\n")
            continue
        w(f"{n:<26}{s['peak']:<9}{s['peak_iter']:<9}{s['min']:<9}{s['min_iter']:<9}"
          f"{s['final']:<9}{str(s['final_minus_peak']):<12}{a.get('multires_full_res_from')}\n")

    w(f"\n(B) RESOLUTION-MATCHED window (iteration >= {resfrom}) -- the only\n"
      "    cross-arm-comparable form of this metric:\n")
    w(f"{'arm':<26}{'peak':<9}{'min':<9}{'final':<9}{'mean-ish(first)':<16}\n")
    for n, a in rep["arms"].items():
        s = (a.get("tb_summary_resolution_matched") or {}).get("optimization/seed_psnr") or {}
        if not s.get("n"):
            w(f"{n:<26}(no series)\n")
            continue
        w(f"{n:<26}{s['peak']:<9}{s['min']:<9}{s['final']:<9}{s['first']:<16}\n")

    w("\n=== SEED PSNR CURVE (absolute, on the fixed sample grid) ===\n")
    w(f"{'iter':<10}" + "".join(f"{n.replace('issue285knock_',''):>10}" for n in rep["arms"]) + "\n")
    for g in SAMPLE_GRID:
        row = f"{g:<10}"
        for n, a in rep["arms"].items():
            v = (a.get("tb_sampled") or {}).get("optimization/seed_psnr", {}).get(str(g))
            row += f"{v:>10.2f}" if v is not None else f"{'--':>10}"
        w(row + "\n")

    w("\n=== PRE-REGISTERED VERDICT (outside-band AND not superimposable) ===\n")
    for n, v in (rep.get("pre_registered_verdict") or {}).items():
        w(f"{n:<26}effect={v['pre_registered_effect']}  shape={v['curve_shape_vs_baseline']}  "
          f"n_outside={v['n_coprimary_descriptors_outside_band']}  "
          f"max_excursion={v['max_excursion_db']}dB"
          f"{'  (BAND PROVISIONAL: <2 replicates)' if v['band_provisional'] else ''}\n")
        for k, e in v["excursions"].items():
            if e["outside_band"]:
                w(f"{'':<26}  outside: {k} = {e['arm']} vs band {e['replicate_band']} "
                  f"({e['direction']} by {e['excursion_db']}dB)\n")

    w("\n=== DOWNSTREAM-EFFECT CHECK (not a dose check -- see Dplus) ===\n")
    w(f"replicate band spanned by: {rep['replicate_band_arms']}\n")
    for n, m in (rep.get("manipulation_check") or {}).items():
        w(f"{n:<26}{m['downstream_effect']:<14}moved: {m['tags_moved_outside_replicate_band']}\n")

    w("\n=== SEED COHORT OPACITY MASS (#287 prediction) ===\n")
    w(f"{'arm':<26}{'tag':<38}{'first':<9}{'final':<9}{'min':<9}\n")
    for n, a in rep["arms"].items():
        for tag in ("streaming/seed_cohort/mean_opacity", "streaming/seed_cohort/alive_count"):
            s = (a.get("tb_summary") or {}).get(tag) or {}
            if not s.get("n"):
                continue
            w(f"{n:<26}{tag:<38}{s['first']:<9}{s['final']:<9}{s['min']:<9}\n")

    w("\n=== SEED-FADE VERDICT (final attributable checkpoint) ===\n")
    for n, a in rep["arms"].items():
        cks = [c for c in a["checkpoints"] if c["belongs_to_last_attempt"] is not False]
        if not cks:
            w(f"{n:<26}(no attributable checkpoint)\n")
            continue
        sf = cks[-1]["seed_fade"]
        indet = cks[-1]["seed_fade_indeterminate"]
        extra = ""
        if a["n_run_start_banners"] > 1:
            first = a.get("last_attempt_first_tb_step")
            # A fresh relaunch logs from ~step 0 in its own event file; a real
            # resume logs from the checkpoint's iteration onward.
            if first is not None and first <= 1000:
                extra = (
                    f"  (multi-attempt dir, but LAST attempt is a FRESH run:"
                    f" first TB step={first}, n_samples={sf.get('n_samples')}"
                    f" -> NOT a mid-run resume, verdict spans the whole run)"
                )
            else:
                extra = (
                    f"  <-- RESUMED at step>={first} and UNFLAGGED:"
                    " treat as INDETERMINATE (#405 postdates run)"
                )
        w(f"{n:<26}status={sf.get('status')} severity={sf.get('severity')} "
          f"confounded={sf.get('confounded')} indeterminate={indet} "
          f"max_dip={sf.get('max_dip_db')}{extra}\n")


def plot_curves(arms_raw: Dict[str, dict], out_path: str) -> Optional[str]:
    """Full curves, never endpoints (mandatory Run Contract).

    Two panels, because the two are measuring different things: the seed view's
    own absolute PSNR (#285's amended co-primary) and the moving-basket
    holdout mean. Returns the written path, or None if matplotlib is absent.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    for name, a in arms_raw.items():
        label = name.replace("issue285knock_", "")
        s = a["tb_series"].get("optimization/seed_psnr") or []
        if s:
            axes[0].plot([x for x, _ in s], [y for _, y in s], lw=1.0, label=label)
        c = a["psnr_curve_from_log"]
        if c:
            axes[1].plot(
                [r["iteration"] for r in c],
                [r["mean_test_psnr"] for r in c],
                lw=1.2,
                label=label,
            )
    axes[0].set_title("optimization/seed_psnr (absolute, co-primary)")
    axes[1].set_title("mean_test_psnr (MOVING basket: n_test grows with admission)")
    for ax in axes:
        ax.set_xlabel("iteration")
        ax.set_ylabel("PSNR (dB)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    output = svg_output_path(out_path)
    fig.savefig(output, dpi=130)
    plt.close(fig)
    return str(output)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--baseline", default=None,
                    help="arm name (basename of its run dir) to compare against")
    ap.add_argument("-o", "--out-dir", default=None,
                    help="write analysis.json (and the aligned curves) here")
    ap.add_argument("--resolution-matched-from", type=int, default=None,
                    help="pin the resolution-matched window start (default: the "
                         "max over the SUPPLIED arms, which changes if you omit "
                         "the low-tier arm)")
    args = ap.parse_args()

    rep = build_report([os.path.abspath(d) for d in args.run_dirs], args.baseline,
                       args.resolution_matched_from)
    print_text(rep)

    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        svg = plot_curves(rep["_arms_raw"], os.path.join(args.out_dir, "curves.svg"))
        serializable = {k: v for k, v in rep.items() if not k.startswith("_")}
        path = os.path.join(args.out_dir, "analysis.json")
        with open(path, "w") as fh:
            json.dump(serializable, fh, indent=2)
        print(f"\nwrote {path}")
        if svg:
            print(f"wrote {svg}")
        else:
            print("matplotlib unavailable -- no curves.svg written")


if __name__ == "__main__":
    main()
