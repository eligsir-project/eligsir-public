"""Consolidated, human-readable run summary (issue #699, partial slice).

Issue #699 asks for a full automatic post-run evidence pipeline (metrics,
local-quality plots, heavy report, consolidated run summary, fail-closed
validation). **This module delivers only the consolidated ``run_summary``
slice** -- composing verdicts that already live in canonical, merged
modules into one readable object, so a reader does not have to hand-assemble
them from ``report.json`` and CLAUDE.md's Run Contracts every time. It does
NOT implement the orchestrator, the local-quality/causal-metric generation,
or the automatic post-run wiring -- those remain #699's open scope (see the
module docstring in ``scripts/run_summary.py`` and the companion
architecture doc for the exact boundary).

Design rules (non-negotiable, per the task this module was built for)
-----------------------------------------------------------------------
1. **A missing key means "unknown / predates this fix" -- NEVER "clean /
   passed".** This project has been burned by the opposite assumption
   repeatedly (an absent ``n_test_metrics_failed`` does not mean nothing
   failed; a zero-sample seed-fade block does not mean no fade; an absent
   ``experiment_validity`` block does not mean the run is fully validated).
   Every gate below is a three-state ``pass`` / ``fail`` / ``unknown``, and
   ``unknown`` is fail-closed for the purposes of the overall verdict --
   see :func:`compute_overall_verdict`.
2. **No grader is reimplemented here.** Coverage comes only from
   ``utils.coverage_access.get_run_coverage`` (the single authoritative
   accessor -- naive lexical ``iter_*`` sorting picks the wrong report for
   the majority of runs). Seed-fade severity/curve-shape comes only from
   ``utils.seed_fade_severity.grade_seed_fade_for_run`` (the canonical
   grader). ``experiment_validity`` is read verbatim from ``report.json``
   (it is already the output of
   ``utils.experiment_validity.build_experiment_validity_block``, computed
   and persisted at report-write time) -- this module never recomputes it.
   If a canonical module refuses to answer (e.g. a future
   ``SeedCohortUnavailable``-style refusal), that refusal is surfaced
   verbatim, never silently substituted with an approximation.
3. **Three independent seed-fade predicates stay independent.**
   ``severity``/``confounded`` (issue #346's curve-shape-aware severity
   grade), ``suppression_confounded`` (issue #282's "the controller really
   throttled a loss" flag), and ``indeterminate`` (issue #290's "this run
   resumed across a state-losing checkpoint boundary" flag) can disagree in
   both directions and must never be collapsed into a single boolean.
4. **Read-only.** Nothing in this module writes to, mutates, or reruns
   anything under ``run_dir`` -- the CLI wrapper (``scripts/run_summary.py``)
   is the only thing that ever writes ``RUN_SUMMARY.md``, and only that one
   file.
5. **Torch-free.** Every import that could pull in torch/numpy at module
   scope is deferred into function bodies, mirroring
   ``utils/coverage_access.py``'s own precedent -- this module (and its
   tests) must be importable in the lean CPU CI image.

What this composes (all already-merged, canonical accessors)
--------------------------------------------------------------
- ``utils.coverage_access.get_run_coverage`` -- coverage, final iteration,
  last-ingest iteration, stream-end validity, map-reinit invalidation.
- ``utils.seed_fade_severity.grade_seed_fade_for_run`` -- seed-fade
  severity/curve-shape, always re-derived from the run's own
  ``psnr_history.jsonl`` (robust even when the report predates the severity
  fields being baked in at write time).
- ``report.json``'s own ``eval_metadata`` (holdout-hash ladder),
  ``baseline_comparison`` (curve-level comparison, when the run declared
  ``--baseline_run``), ``git_provenance`` (issue #259), ``report_provenance``
  (issue #630, native vs regenerated), and ``experiment_validity`` (issue
  #203) blocks -- all read verbatim, never recomputed.
- ``utils.metric_registry`` -- used only to resolve any legacy metric-name
  aliases encountered in the runtime/throughput fields of a report into
  their canonical names for display, so the summary does not repeat this
  project's own history of ambiguous metric labels (issue #700/#704).
"""
from __future__ import annotations

import dataclasses
import datetime
import json
import os
from typing import Any, Dict, List, Optional, Tuple, Union

SUMMARY_FILENAME = "run_summary.json"
SUMMARY_SCHEMA = "eligsir.run-summary/v1"

__all__ = [
    "SUMMARY_FILENAME",
    "SUMMARY_SCHEMA",
    "load_run_summary",
    "write_run_summary",
    "build_run_summary",
    "render_markdown",
    "RunSummary",
    "Gate",
    "build_quotability_block",
    "build_seed_fade_block",
    "build_holdout_block",
    "build_provenance_block",
    "build_experiment_validity_summary",
    "build_metrics_block",
    "compute_overall_verdict",
]


# ---------------------------------------------------------------------------
# Small, shared plumbing
# ---------------------------------------------------------------------------


def _field(container: Any, key: str) -> tuple:
    """Return ``(value, present)`` for ``container[key]``.

    ``present`` distinguishes "the key is genuinely absent" (predates the
    fix that would have written it) from "the key is present with value
    ``None``" (an explicit, deliberate unknown -- e.g.
    ``coverage_access.CoverageSummary.frames_released``). Callers that only
    care about "do we have a usable value" can just check ``value is not
    None``, but the gate logic below cares about the distinction.
    """
    if not isinstance(container, dict) or key not in container:
        return None, False
    return container[key], True


@dataclasses.dataclass
class Gate:
    """One pass/fail/unknown verdict feeding the overall quotability call.

    ``status`` is one of ``"pass"``, ``"fail"``, ``"unknown"`` --
    deliberately a closed three-state enum, not a boolean, so "we don't
    know" can never be silently treated as "it passed".
    """

    name: str
    status: str
    detail: str

    def to_dict(self) -> dict:
        return {"name": self.name, "status": self.status, "detail": self.detail}


def _load_report(run_dir: str):
    """Thin wrapper over ``coverage_access.load_latest_report`` so this
    module has exactly one place that resolves "the" report.json for a run
    (numeric ``iter_*`` sort, never lexical -- see that module's docstring).
    """
    from eligsir.utils.coverage_access import load_latest_report

    return load_latest_report(run_dir)


# ---------------------------------------------------------------------------
# Quotability gates
# ---------------------------------------------------------------------------


def build_quotability_block(report: Optional[dict], coverage) -> dict:
    """Gate: is this run safe to quote a PSNR/coverage number from at all?

    Composes, per CLAUDE.md's Run Contracts:
    - ``eval_valid`` / ``n_test`` (real-holdout contract)
    - ``n_test_metrics_failed`` (issue #437 -- a low ``n_test_metrics`` can
      mean cameras failed to render, not that the report-cap subsampled
      them)
    - ``map_reinit_invalidated`` / stream-end truncation (both surfaced by
      ``coverage`` -- a ``CoverageSummary`` from ``utils.coverage_access``)
    """
    gates: list = []

    eval_valid, eval_valid_present = _field(report, "eval_valid")
    if not eval_valid_present:
        gates.append(Gate(
            "eval_valid", "unknown",
            "report.json has no 'eval_valid' key -- absent or unreadable report; "
            "never assume a valid holdout.",
        ))
    elif eval_valid is False:
        gates.append(Gate(
            "eval_valid", "fail",
            "report.json['eval_valid'] is False -- not a real evaluation split "
            "(e.g. a *_bootstrap_views/ degenerate split); never quote PSNR from this run.",
        ))
    else:
        gates.append(Gate("eval_valid", "pass", "eval_valid is True."))

    n_test, n_test_present = _field(report, "n_test")
    if not n_test_present or n_test is None:
        gates.append(Gate(
            "n_test_positive", "unknown",
            "report.json has no 'n_test' key.",
        ))
    elif not isinstance(n_test, (int, float)) or n_test <= 0:
        gates.append(Gate(
            "n_test_positive", "fail",
            f"n_test={n_test!r} -- a run with n_test == 0 is not a valid eval.",
        ))
    else:
        gates.append(Gate("n_test_positive", "pass", f"n_test={n_test}."))

    n_failed, n_failed_present = _field(report, "n_test_metrics_failed")
    if not n_failed_present:
        gates.append(Gate(
            "n_test_metrics_failed_zero", "unknown",
            "report.json has no 'n_test_metrics_failed' key -- this report predates "
            "issue #437's render-failure accounting. An absent key does NOT mean "
            "nothing failed.",
        ))
    elif n_failed:
        gates.append(Gate(
            "n_test_metrics_failed_zero", "fail",
            f"n_test_metrics_failed={n_failed} -- mean_test_psnr is over a partial, "
            "failure-selected subset of the holdout; not comparable to a run with a "
            "different failure count.",
        ))
    else:
        gates.append(Gate("n_test_metrics_failed_zero", "pass", "n_test_metrics_failed=0."))

    curve_valid, curve_valid_present = _field(report, "psnr_history_curve_valid")
    if not curve_valid_present or curve_valid is None:
        gates.append(Gate(
            "psnr_history_curve_valid",
            "unknown",
            "report.json has no complete PSNR-history validity verdict; the full "
            "curve contract is unproven.",
        ))
    elif curve_valid is False:
        gates.append(Gate(
            "psnr_history_curve_valid",
            "fail",
            "one or more held-out PSNR-history ticks failed, so the curve is "
            "incomplete and the run is not quotable.",
        ))
    else:
        gates.append(Gate(
            "psnr_history_curve_valid",
            "pass",
            "all recorded held-out PSNR-history ticks are valid.",
        ))

    map_reinit = coverage.map_reinit_invalidated
    if map_reinit is None:
        gates.append(Gate(
            "map_reinit_invalidated", "unknown",
            "trajectory_coverage.map_reinit_invalidated is absent (no coverage block, "
            "or predates issue #506's detector).",
        ))
    elif map_reinit is True:
        gates.append(Gate(
            "map_reinit_invalidated", "fail",
            f"map_reinit_invalidated=True ({coverage.coverage_block.get('map_reinit_reason') if coverage.coverage_block else None}) "
            "-- an ORB-SLAM3 mid-run map re-initialization fused two unrelated "
            "coordinate frames into one trajectory; never quote this run's coverage or PSNR.",
        ))
    else:
        gates.append(Gate("map_reinit_invalidated", "pass", "map_reinit_invalidated=False."))

    if coverage.verdict.startswith("UNKNOWN"):
        gates.append(Gate(
            "stream_end_not_truncated", "unknown",
            f"trajectory_coverage verdict is '{coverage.verdict}' -- no coverage block "
            "resolvable at all.",
        ))
    elif coverage.suspected_truncated:
        gates.append(Gate(
            "stream_end_not_truncated", "fail",
            f"verdict={coverage.verdict!r} stream_end_reason={coverage.stream_end_reason!r} "
            "-- treat exactly like eval_valid==False; never quote this run's coverage or "
            "PSNR as valid.",
        ))
    else:
        gates.append(Gate(
            "stream_end_not_truncated", "pass",
            f"verdict={coverage.verdict!r}, stream_end_reason={coverage.stream_end_reason!r}.",
        ))

    return {
        "gates": [g.to_dict() for g in gates],
        "eval_valid": eval_valid,
        "n_test": n_test,
        "n_test_metrics_failed": n_failed,
        "n_test_metrics_failed_present": n_failed_present,
        "psnr_history_curve_valid": curve_valid,
        "map_reinit_invalidated": map_reinit,
        "stream_end_reason": coverage.stream_end_reason,
        "suspected_truncated": coverage.suspected_truncated,
        "coverage_verdict": coverage.verdict,
    }


# ---------------------------------------------------------------------------
# Seed-fade: three independent predicates, never collapsed
# ---------------------------------------------------------------------------


def build_seed_fade_block(report: Optional[dict], run_dir: str) -> dict:
    """Compose the seed-fade facet via the one canonical grader
    (``utils.seed_fade_severity.grade_seed_fade_for_run``), which always
    re-derives ``severity``/``curve_shape`` from the run's own
    ``psnr_history.jsonl`` on disk -- so this is correct even for a report
    written before severity grading existed. Raw detector fields
    (``indeterminate``, ``suppression_confounded``, ``status``) are passed
    through only if the run's own ``report.json['seed_fade']`` carried them;
    this function never invents them.
    """
    from eligsir.utils.seed_fade_severity import grade_seed_fade_for_run

    raw_seed_fade, raw_present = _field(report, "seed_fade")
    graded = grade_seed_fade_for_run(raw_seed_fade if raw_present else None, run_dir)

    curve_available = graded.get("curve_shape") not in (None, "insufficient_history")
    data_available = raw_present or curve_available

    if not data_available:
        confounded_status = "unknown"
        confounded_detail = (
            "No 'seed_fade' block in report.json and no psnr_history.jsonl curve on "
            "disk -- the seed-fade detector never ran (or predates it) AND there is no "
            "curve to grade retroactively. This is NOT the same as 'no fade'."
        )
    elif graded.get("confounded"):
        confounded_status = "fail"
        confounded_detail = (
            f"severity={graded.get('severity')!r} (moderate/severe) -- treat exactly "
            "like eval_valid==False: never quote this run's PSNR without this caveat."
        )
    else:
        confounded_status = "pass"
        confounded_detail = f"severity={graded.get('severity')!r}."

    indeterminate_present = isinstance(raw_seed_fade, dict) and "indeterminate" in raw_seed_fade
    indeterminate_val = raw_seed_fade.get("indeterminate") if indeterminate_present else None
    if not indeterminate_present:
        indeterminate_status = "unknown"
        indeterminate_detail = (
            "report.json['seed_fade'] has no 'indeterminate' key -- predates issue "
            "#290's resume-state tracking."
        )
    elif indeterminate_val:
        indeterminate_status = "fail"
        indeterminate_detail = (
            "seed_fade['indeterminate']=True -- this run resumed across a checkpoint "
            "that did not carry faithful seed-fade detector state (or a detector-config "
            "change happened across the resume). Never quote this run's seed-fade "
            "verdict as evidence."
        )
    else:
        indeterminate_status = "pass"
        indeterminate_detail = "seed_fade['indeterminate']=False."

    suppression_present = isinstance(raw_seed_fade, dict) and "suppression_confounded" in raw_seed_fade
    suppression_val = raw_seed_fade.get("suppression_confounded") if suppression_present else None

    return {
        "data_available": data_available,
        "detected": graded.get("detected"),
        "severity": graded.get("severity"),
        "severity_reasons": graded.get("severity_reasons"),
        "confounded": {
            "status": confounded_status,
            "value": graded.get("confounded") if data_available else None,
            "detail": confounded_detail,
        },
        "suppression_confounded": {
            "status": "unknown" if not suppression_present else ("fail" if suppression_val else "pass"),
            "value": suppression_val,
            "present": suppression_present,
            "detail": (
                "report.json['seed_fade'] has no 'suppression_confounded' key -- "
                "predates issue #282."
                if not suppression_present else
                "The controller really throttled a loss during this run (issue #282) -- "
                "independent of, and can disagree with, the severity-based 'confounded' "
                "flag above."
                if suppression_val else
                "suppression_confounded=False."
            ),
        },
        "indeterminate": {
            "status": indeterminate_status,
            "value": indeterminate_val,
            "present": indeterminate_present,
            "detail": indeterminate_detail,
        },
        "curve_shape": graded.get("curve_shape"),
        "curve_first_db": graded.get("curve_first_db"),
        "curve_final_db": graded.get("curve_final_db"),
        "curve_peak_db": graded.get("curve_peak_db"),
        "curve_max_drawdown_db": graded.get("curve_max_drawdown_db"),
        "curve_n_ticks": graded.get("curve_n_ticks"),
        "mapping_phase_known": graded.get("mapping_phase_known"),
        "mapping_end_iter": graded.get("mapping_end_iter"),
        "_graded_raw": graded,
    }


# ---------------------------------------------------------------------------
# Holdout comparability ladder
# ---------------------------------------------------------------------------


def build_holdout_block(report: Optional[dict]) -> dict:
    """The ``holdout_hash`` / ``realized_holdout_hash`` / ``scored_holdout_hash``
    ladder (issue #199 / #420), plus ``baseline_comparison``'s
    ``verification_level`` when the run declared ``--baseline_run``. Reads
    both verbatim; computes nothing.
    """
    eval_metadata, eval_metadata_present = _field(report, "eval_metadata")
    holdout_hash, holdout_present = _field(eval_metadata, "holdout_hash")
    realized_hash, realized_present = _field(eval_metadata, "realized_holdout_hash")
    scored_hash, scored_present = _field(eval_metadata, "scored_holdout_hash")

    if scored_present:
        rung = "scored_hash"
    elif realized_present:
        rung = "realized_hash"
    elif holdout_present:
        rung = "hash_only"
    else:
        rung = "unknown"

    baseline_comparison, baseline_present = _field(report, "baseline_comparison")
    verification_level = None
    legacy_curve_status = None
    phase_relative_status = None
    phase_relative_shapes = None
    if baseline_present and isinstance(baseline_comparison, dict):
        verification_level = baseline_comparison.get("verification_level")
        legacy_curve_status = (
            baseline_comparison.get("curve_shape_provenance") or {}
        ).get("status")
        phase_relative = baseline_comparison.get("phase_relative_comparison")
        if isinstance(phase_relative, dict):
            phase_relative_status = phase_relative.get("status")
            axes = phase_relative.get("axes") or {}
            phase_relative_shapes = {
                axis_name: (axes.get(axis_name) or {}).get("curve_shape")
                for axis_name in ("mapping_active_s", "supervision_views")
            }

    return {
        "eval_metadata_present": eval_metadata_present,
        "holdout_hash": holdout_hash,
        "realized_holdout_hash": realized_hash,
        "scored_holdout_hash": scored_hash,
        "highest_verified_rung": rung,
        "baseline_comparison_present": baseline_present,
        "baseline_run": (baseline_comparison or {}).get("baseline_run") if baseline_present else None,
        "baseline_comparable": (baseline_comparison or {}).get("comparable") if baseline_present else None,
        "baseline_curve_shape": (baseline_comparison or {}).get("curve_shape") if baseline_present else None,
        "baseline_curve_shape_status": legacy_curve_status,
        "baseline_phase_relative_status": phase_relative_status,
        "baseline_phase_relative_curve_shapes": phase_relative_shapes,
        "baseline_verification_level": verification_level,
    }


# ---------------------------------------------------------------------------
# Provenance (git + report-generation)
# ---------------------------------------------------------------------------


def build_provenance_block(report: Optional[dict]) -> dict:
    """``git_provenance`` (issue #259) and ``report_provenance`` (issue #630:
    native vs regenerated). Read verbatim; an absent block means "predates
    that fix" and is reported as such, never as "clean"."""
    git_prov, git_present = _field(report, "git_provenance")
    report_prov, report_prov_present = _field(report, "report_provenance")

    return {
        "git_provenance_present": git_present,
        "git_commit": (git_prov or {}).get("git_commit") if git_present else None,
        "git_branch": (git_prov or {}).get("git_branch") if git_present else None,
        "git_dirty": (git_prov or {}).get("git_dirty") if git_present else None,
        "report_provenance_present": report_prov_present,
        "report_kind": (report_prov or {}).get("kind") if report_prov_present else None,
        "report_provenance_raw": report_prov if report_prov_present else None,
    }


# ---------------------------------------------------------------------------
# Experiment validity (issue #203) -- read verbatim, never recomputed
# ---------------------------------------------------------------------------


def build_experiment_validity_summary(report: Optional[dict]) -> dict:
    ev, present = _field(report, "experiment_validity")
    if not present or not isinstance(ev, dict):
        return {
            "present": False,
            "declared": None,
            "achieved": None,
            "effective": None,
            "negative_conclusions_permitted": None,
            "strong_negative_conclusions_permitted": None,
            "mismatch": None,
            "note": (
                "report.json has no 'experiment_validity' block -- this report predates "
                "issue #203. UNKNOWN, not UNDECLARED-permissive: no negative-conclusion "
                "claim from this run can be defended by this field."
            ),
        }
    return {
        "present": True,
        "declared": ev.get("declared"),
        "achieved": ev.get("achieved"),
        "effective": ev.get("effective"),
        "negative_conclusions_permitted": ev.get("negative_conclusions_permitted"),
        "strong_negative_conclusions_permitted": ev.get("strong_negative_conclusions_permitted"),
        "negative_result_status_ceiling": ev.get("negative_result_status_ceiling"),
        "mismatch": ev.get("mismatch"),
        "is_final_report": ev.get("is_final_report"),
        "warnings": ev.get("warnings") or [],
    }


# ---------------------------------------------------------------------------
# Headline metrics
# ---------------------------------------------------------------------------

_HEADLINE_KEYS = (
    "mean_test_psnr", "min_test_psnr", "max_test_psnr",
    "mean_test_lpips", "min_test_lpips", "max_test_lpips",
    "mean_train_psnr",
    "n_train", "n_test", "n_test_metrics", "n_train_metrics",
    "iteration",
)


def build_metrics_block(report: Optional[dict], seed_fade_block: dict) -> dict:
    """Headline scalar metrics plus the PSNR **curve shape**, not just the
    final number (CLAUDE.md's "Compare PSNR curves, not just the final
    number" Run Contract) -- reuses the curve-shape fields already computed
    by ``build_seed_fade_block`` (which calls the same canonical
    ``classify_curve_shape`` regardless of whether a fade was ever detected)
    rather than re-deriving them a second time.
    """
    headline = {}
    for k in _HEADLINE_KEYS:
        v, present = _field(report, k)
        headline[k] = v if present else None

    # Legacy metric-name resolution (issue #700/#704): translate any
    # throughput/rate field names this report might carry under a legacy
    # alias into their canonical registry name, purely for display.
    from eligsir.utils.metric_registry import get_metric_by_alias

    legacy_hits = {}
    if isinstance(report, dict):
        for key in report.keys():
            entry = get_metric_by_alias(key)
            if entry is not None:
                legacy_hits[key] = entry.name

    return {
        "headline": headline,
        "curve_shape": seed_fade_block.get("curve_shape"),
        "curve_first_db": seed_fade_block.get("curve_first_db"),
        "curve_final_db": seed_fade_block.get("curve_final_db"),
        "curve_peak_db": seed_fade_block.get("curve_peak_db"),
        "curve_max_drawdown_db": seed_fade_block.get("curve_max_drawdown_db"),
        "curve_n_ticks": seed_fade_block.get("curve_n_ticks"),
        "legacy_metric_aliases_present": legacy_hits,
    }


# ---------------------------------------------------------------------------
# Overall verdict
# ---------------------------------------------------------------------------


def compute_overall_verdict(gates: list) -> dict:
    """Fail-closed aggregation: any ``fail`` wins; else any ``unknown``
    wins; only if every gate is ``pass`` is the run overall quotable.
    """
    failing = [g for g in gates if g.get("status") == "fail"]
    unknown = [g for g in gates if g.get("status") == "unknown"]
    if failing:
        verdict = "NOT_QUOTABLE"
    elif unknown:
        verdict = "UNKNOWN"
    else:
        verdict = "QUOTABLE"
    return {
        "verdict": verdict,
        "failing_gates": [g["name"] for g in failing],
        "unknown_gates": [g["name"] for g in unknown],
    }


# ---------------------------------------------------------------------------
# Top-level assembly
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class RunSummary:
    run_dir: str
    report_path: Optional[str]
    report_found: bool
    load_error: Optional[str]
    quotability: dict
    seed_fade: dict
    holdout: dict
    coverage: dict
    provenance: dict
    experiment_validity: dict
    metrics: dict
    overall: dict

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def build_run_summary(run_dir: str) -> RunSummary:
    """The single entry point: build a :class:`RunSummary` for ``run_dir``.

    Read-only; never raises (individual composers are defensive, and
    ``coverage_access``/``seed_fade_severity`` are themselves documented as
    never-raising); never reruns training or mutates anything on disk.
    """
    from eligsir.utils.coverage_access import get_run_coverage

    report, report_path, load_error = _load_report(run_dir)
    coverage = get_run_coverage(run_dir)

    quotability = build_quotability_block(report, coverage)
    seed_fade = build_seed_fade_block(report, run_dir)
    holdout = build_holdout_block(report)
    provenance = build_provenance_block(report)
    experiment_validity = build_experiment_validity_summary(report)
    metrics = build_metrics_block(report, seed_fade)

    overall_gates = list(quotability["gates"])
    overall_gates.append({
        "name": "seed_fade_not_confounded",
        "status": seed_fade["confounded"]["status"],
        "detail": seed_fade["confounded"]["detail"],
    })
    overall_gates.append({
        "name": "seed_fade_not_indeterminate",
        "status": seed_fade["indeterminate"]["status"],
        "detail": seed_fade["indeterminate"]["detail"],
    })
    overall = compute_overall_verdict(overall_gates)

    return RunSummary(
        run_dir=run_dir,
        report_path=report_path,
        report_found=report is not None,
        load_error=load_error,
        quotability=quotability,
        seed_fade=seed_fade,
        holdout=holdout,
        coverage=coverage.to_dict(),
        provenance=provenance,
        experiment_validity=experiment_validity,
        metrics=metrics,
        overall=overall,
    )


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------


def _fmt(v: Any) -> str:
    if v is None:
        return "`UNKNOWN`"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def _status_emoji(status: str) -> str:
    return {"pass": "PASS", "fail": "FAIL", "unknown": "UNKNOWN"}.get(status, status.upper())


def render_markdown(summary: RunSummary) -> str:
    d = summary.to_dict()
    lines: list = []
    lines.append(f"# Run summary: `{d['run_dir']}`")
    lines.append("")
    lines.append(
        "_Partial delivery of issue #699 (consolidated run summary only) -- "
        "see `docs/architecture/2026-08-15-issue699-consolidated-run-summary.md` "
        "for the full #699 scope and what remains undelivered._"
    )
    lines.append("")
    lines.append(f"**Report path:** `{d['report_path']}`  ")
    lines.append(f"**Report found:** {d['report_found']}  ")
    if d["load_error"]:
        lines.append(f"**Load error:** {d['load_error']}  ")
    lines.append("")

    o = d["overall"]
    lines.append(f"## Overall verdict: **{o['verdict']}**")
    lines.append("")
    if o["failing_gates"]:
        lines.append(f"- Failing gates: {', '.join(o['failing_gates'])}")
    if o["unknown_gates"]:
        lines.append(f"- Unknown gates (report predates the relevant fix -- NOT clean): "
                      f"{', '.join(o['unknown_gates'])}")
    if not o["failing_gates"] and not o["unknown_gates"]:
        lines.append("- All quotability gates pass.")
    lines.append("")

    lines.append("## Quotability gates")
    lines.append("")
    lines.append("| Gate | Status | Detail |")
    lines.append("|---|---|---|")
    for g in d["quotability"]["gates"]:
        lines.append(f"| {g['name']} | {_status_emoji(g['status'])} | {g['detail']} |")
    lines.append("")

    sf = d["seed_fade"]
    lines.append("## Seed-fade (three independent predicates)")
    lines.append("")
    lines.append(f"- data_available: {sf['data_available']}")
    lines.append(f"- severity: {_fmt(sf['severity'])}  (confounded: "
                  f"{_status_emoji(sf['confounded']['status'])} -- {sf['confounded']['detail']})")
    lines.append(f"- suppression_confounded (issue #282, independent of severity): "
                  f"{_status_emoji(sf['suppression_confounded']['status'])} -- "
                  f"{sf['suppression_confounded']['detail']}")
    lines.append(f"- indeterminate (issue #290, resume-safety, independent of both above): "
                  f"{_status_emoji(sf['indeterminate']['status'])} -- {sf['indeterminate']['detail']}")
    lines.append(f"- curve_shape: {_fmt(sf['curve_shape'])} "
                  f"(first={_fmt(sf['curve_first_db'])}dB, peak={_fmt(sf['curve_peak_db'])}dB, "
                  f"final={_fmt(sf['curve_final_db'])}dB, max_drawdown={_fmt(sf['curve_max_drawdown_db'])}dB, "
                  f"n_ticks={_fmt(sf['curve_n_ticks'])})")
    lines.append("")

    h = d["holdout"]
    lines.append("## Holdout comparability ladder")
    lines.append("")
    lines.append(f"- holdout_hash: {_fmt(h['holdout_hash'])}")
    lines.append(f"- realized_holdout_hash: {_fmt(h['realized_holdout_hash'])}")
    lines.append(f"- scored_holdout_hash: {_fmt(h['scored_holdout_hash'])}")
    lines.append(f"- highest verified rung: {h['highest_verified_rung']}")
    if h["baseline_comparison_present"]:
        lines.append(f"- baseline_run: {_fmt(h['baseline_run'])} "
                      f"(comparable={h['baseline_comparable']}, "
                      f"legacy_iteration_curve_shape={_fmt(h['baseline_curve_shape'])} "
                      f"[{_fmt(h['baseline_curve_shape_status'])}], "
                      f"phase_relative_status={_fmt(h['baseline_phase_relative_status'])}, "
                      f"phase_relative_curve_shapes={_fmt(h['baseline_phase_relative_curve_shapes'])}, "
                      f"verification_level={_fmt(h['baseline_verification_level'])})")
    else:
        lines.append("- baseline_comparison: absent (no `--baseline_run` declared for this run)")
    lines.append("")

    c = d["coverage"]
    lines.append("## Coverage (via `utils.coverage_access.get_run_coverage`, the single authoritative accessor)")
    lines.append("")
    lines.append(f"- verdict: {_fmt(c['verdict'])}")
    lines.append(f"- n_source_frames: {_fmt(c['n_source_frames'])}  frames_admitted: {_fmt(c['frames_admitted'])}  "
                  f"coverage_fraction: {_fmt(c['coverage_fraction'])}")
    lines.append(f"- final_iteration: {_fmt(c['final_iteration'])} "
                  f"(source={c['final_iteration_source']}, confidence={c['final_iteration_confidence']})")
    lines.append(f"- last_ingest_iter: {_fmt(c['last_ingest_iter'])}  "
                  f"refinement_tail_iterations: {_fmt(c['refinement_tail_iterations'])}")
    lines.append(f"- frames_released: `UNKNOWN` ({c['frames_released_unavailable_reason']})")
    if c["warnings"]:
        lines.append(f"- warnings: {'; '.join(c['warnings'])}")
    lines.append("")

    ev = d["experiment_validity"]
    lines.append("## Experiment validity class (issue #203)")
    lines.append("")
    if not ev["present"]:
        lines.append(f"- {ev['note']}")
    else:
        lines.append(f"- declared: {_fmt(ev['declared'])}  achieved: {_fmt(ev['achieved'])}  "
                      f"effective: {_fmt(ev['effective'])}")
        lines.append(f"- negative_conclusions_permitted: {_fmt(ev['negative_conclusions_permitted'])}  "
                      f"strong_negative_conclusions_permitted: {_fmt(ev['strong_negative_conclusions_permitted'])}")
        lines.append(f"- negative_result_status_ceiling: {_fmt(ev['negative_result_status_ceiling'])}  "
                      f"mismatch: {_fmt(ev['mismatch'])}  is_final_report: {_fmt(ev['is_final_report'])}")
        for w in ev["warnings"]:
            lines.append(f"  - warning: {w}")
    lines.append("")

    m = d["metrics"]
    lines.append("## Headline metrics")
    lines.append("")
    for k, v in m["headline"].items():
        lines.append(f"- {k}: {_fmt(v)}")
    lines.append(f"- psnr curve_shape: {_fmt(m['curve_shape'])}")
    if m["legacy_metric_aliases_present"]:
        lines.append(f"- legacy metric aliases present in this report: {m['legacy_metric_aliases_present']} "
                      "(see `utils.metric_registry` for canonical names)")
    lines.append("")

    p = d["provenance"]
    lines.append("## Provenance")
    lines.append("")
    if p["git_provenance_present"]:
        lines.append(f"- git: commit={_fmt(p['git_commit'])} branch={_fmt(p['git_branch'])} "
                      f"dirty={_fmt(p['git_dirty'])}")
    else:
        lines.append("- git_provenance: `UNKNOWN` (report predates issue #259)")
    if p["report_provenance_present"]:
        lines.append(f"- report_provenance: kind={_fmt(p['report_kind'])} "
                      f"(raw: {p['report_provenance_raw']})")
    else:
        lines.append("- report_provenance: `UNKNOWN` (report predates issue #630 -- "
                      "cannot tell native vs regenerated)")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Canonical generic handoff (issue #997)
# ---------------------------------------------------------------------------


def load_run_summary(
    model_path: str | os.PathLike[str],
) -> Optional[Dict[str, Any]]:
    """Load persisted run summary from <model_path>/run_summary.json.

    Returns the parsed summary dictionary, or None if no summary file exists.
    """
    if not model_path:
        return None
    path = os.path.join(os.fspath(model_path), SUMMARY_FILENAME)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def write_run_summary(
    model_path: str | os.PathLike[str],
    *,
    completion_state: Optional[Dict[str, Any]] = None,
    report: Optional[Dict[str, Any]] = None,
    report_path: Optional[str] = None,
    experiment_origin: Optional[Dict[str, Any]] = None,
    git_provenance: Optional[Dict[str, Any]] = None,
    execution_semantics: Optional[Dict[str, Any]] = None,
    evidence_class: Optional[str] = None,
    args: Any = None,
) -> Optional[str]:
    """Atomically write canonical run_summary.json to <model_path>/run_summary.json.

    Pure stdlib, deterministic key ordering, atomic temp file + rename.
    Never raises. Returns the written path or None on failure.
    """
    if not model_path:
        return None
    model_path = os.fspath(model_path)

    # 1. Completion state
    if completion_state is None:
        from eligsir.utils.run_completion_state import load_run_completion_state
        completion_state = load_run_completion_state(model_path)

    # 2. Canonical report & report location
    if report is None or report_path is None:
        from eligsir.utils.coverage_access import load_latest_report
        loaded_report, loaded_path, _ = load_latest_report(model_path)
        if report is None:
            report = loaded_report
        if report_path is None:
            report_path = loaded_path

    report_location: Optional[str] = None
    if report_path and os.path.exists(report_path):
        try:
            report_location = os.path.relpath(report_path, model_path)
        except Exception:
            report_location = os.path.basename(report_path)

    # 3. Reconstruction metrics
    reconstruction_metrics: Optional[Dict[str, Any]] = None
    if isinstance(report, dict):
        reconstruction_metrics = {
            "iteration": report.get("iteration"),
            "max_test_psnr": report.get("max_test_psnr"),
            "mean_test_lpips": report.get("mean_test_lpips"),
            "mean_test_psnr": report.get("mean_test_psnr"),
            "mean_test_ssim": report.get("mean_test_ssim"),
            "median_test_psnr": report.get("median_test_psnr"),
            "min_test_psnr": report.get("min_test_psnr"),
            "n_test": report.get("n_test"),
            "n_test_metrics": report.get("n_test_metrics"),
            "n_train": report.get("n_train"),
        }

    # 4. Evaluation metadata
    eval_metadata: Optional[Dict[str, Any]] = None
    if isinstance(report, dict):
        eval_meta_raw = report.get("eval_metadata")
        eval_meta_dict = eval_meta_raw if isinstance(eval_meta_raw, dict) else {}
        seed_fade_raw = report.get("seed_fade")
        seed_fade_dict = seed_fade_raw if isinstance(seed_fade_raw, dict) else {}
        eval_metadata = {
            "eval_valid": report.get("eval_valid"),
            "holdout_hash": eval_meta_dict.get("holdout_hash"),
            "n_scored": (
                report.get("n_test_metrics")
                if report.get("n_test_metrics") is not None
                else eval_meta_dict.get("n_scored")
            ),
            "n_test": report.get("n_test"),
            "realized_holdout_hash": eval_meta_dict.get("realized_holdout_hash"),
            "scored_holdout_hash": eval_meta_dict.get("scored_holdout_hash"),
            "seed_fade": seed_fade_dict if seed_fade_raw is not None else None,
            "seed_fade_confounded": seed_fade_dict.get("confounded") if seed_fade_raw is not None else None,
            "seed_fade_detected": seed_fade_dict.get("detected") if seed_fade_raw is not None else None,
            "seed_fade_severity": seed_fade_dict.get("severity") if seed_fade_raw is not None else None,
            "seed_fade_status": seed_fade_dict.get("status") if seed_fade_raw is not None else None,
        }

    # 5. Execution semantics
    if execution_semantics is None:
        exec_dict: Dict[str, Any] = {}
        if isinstance(report, dict) and isinstance(report.get("execution_semantics"), dict):
            exec_dict.update(report["execution_semantics"])

        trainer = (
            exec_dict.get("trainer")
            or (
                ("streaming" if bool(getattr(args, "streaming_replay", False)) else "offline")
                if args is not None
                else "unknown"
            )
        )

        iterations = (
            exec_dict.get("iterations")
            or (report.get("iteration") if isinstance(report, dict) and report.get("iteration") is not None else None)
            or (report.get("final_iteration") if isinstance(report, dict) and report.get("final_iteration") is not None else None)
            or (getattr(args, "iterations", None) if args is not None else None)
            or (completion_state.get("final_iteration") if isinstance(completion_state, dict) else None)
        )

        resolution = (
            exec_dict.get("resolution")
            or (report.get("resolution") if isinstance(report, dict) and report.get("resolution") is not None else None)
            or (report.get("spf_resolution") if isinstance(report, dict) and report.get("spf_resolution") is not None else None)
            or (getattr(args, "resolution", None) if args is not None else None)
        )

        device = (
            exec_dict.get("device")
            or (getattr(args, "device", None) if args is not None else None)
            or (report.get("device") if isinstance(report, dict) else None)
            or "cuda"
        )

        model_layout = (
            exec_dict.get("model_layout")
            or (report.get("model_layout") if isinstance(report, dict) and report.get("model_layout") is not None else None)
            or (getattr(args, "model_layout", None) if args is not None else None)
            or "gsplat"
        )

        exec_dict["trainer"] = trainer
        exec_dict["iterations"] = iterations
        exec_dict["resolution"] = resolution
        exec_dict["device"] = device
        exec_dict["model_layout"] = model_layout
        execution_semantics = exec_dict

    # 6. Git provenance
    if git_provenance is None:
        if isinstance(report, dict) and isinstance(report.get("git_provenance"), dict):
            git_provenance = dict(report["git_provenance"])
        elif args is not None and hasattr(args, "git_commit"):
            git_provenance = {
                "git_branch": getattr(args, "git_branch", "unknown"),
                "git_commit": getattr(args, "git_commit", "unknown"),
                "git_dirty": getattr(args, "git_dirty", "unknown"),
                "git_dirty_paths": getattr(args, "git_dirty_paths", []),
            }
        else:
            try:
                from eligsir.utils.git_metadata import get_git_metadata
                git_provenance = get_git_metadata()
            except Exception:
                git_provenance = {
                    "git_branch": "unknown",
                    "git_commit": "unknown",
                    "git_dirty": "unknown",
                    "git_dirty_paths": [],
                }

    # 7. Experiment origin & Evidence Class
    if experiment_origin is None:
        if isinstance(report, dict) and isinstance(report.get("experiment_origin"), dict):
            experiment_origin = dict(report["experiment_origin"])
        else:
            try:
                from eligsir.utils.experiment_origin import read_experiment_origin_file
                experiment_origin = read_experiment_origin_file(model_path)
            except Exception:
                experiment_origin = None

    if evidence_class is None:
        if args is not None and getattr(args, "scientific_mode", False):
            evidence_class = "MANAGED"
        elif isinstance(experiment_origin, dict) and experiment_origin.get("evidence_class"):
            evidence_class = experiment_origin["evidence_class"]
        elif (
            isinstance(experiment_origin, dict)
            and experiment_origin.get("researchflow_run_id")
            and experiment_origin.get("researchflow_plan_sha256")
        ):
            evidence_class = "MANAGED"
        elif (
            isinstance(experiment_origin, dict)
            and experiment_origin.get("status") == "complete"
            and experiment_origin.get("task_kind") in ("github_issue", "github_pull_request")
            and args is not None
            and getattr(args, "scientific_mode", False)
        ):
            evidence_class = "MANAGED"
        else:
            evidence_class = "UNMANAGED"

    # Issue #1302: an operator-declared, unattested/legacy RGB-D source (e.g.
    # the kitchen1/floor2 historical bags) can never be laundered into a
    # MANAGED result no matter what ResearchFlow registration says -- this
    # can only push evidence_class DOWN to UNMANAGED, never up. Absent
    # source_attestation (every pre-#1302 run) leaves evidence_class exactly
    # as already resolved above.
    _source_attestation = report.get("source_attestation") if isinstance(report, dict) else None
    if _source_attestation is None and isinstance(experiment_origin, dict):
        _source_attestation = experiment_origin.get("source_attestation")
    if _source_attestation is not None:
        from eligsir.utils.source_attestation import cap_evidence_class as _cap_evidence_class
        evidence_class = _cap_evidence_class(evidence_class, _source_attestation)

    if isinstance(experiment_origin, dict):
        experiment_origin["evidence_class"] = evidence_class

    # 8. Created at (ISO timestamp)
    created_at = datetime.datetime.now(datetime.timezone.utc).isoformat()

    summary_dict = {
        "schema": SUMMARY_SCHEMA,
        "created_at": created_at,
        "evidence_class": evidence_class,
        "report_location": report_location,
        "completion_state": completion_state,
        "reconstruction_metrics": reconstruction_metrics,
        "eval_metadata": eval_metadata,
        "execution_semantics": execution_semantics,
        "git_provenance": git_provenance,
        "source_attestation": _source_attestation,
        "experiment_origin": experiment_origin,
    }

    target = os.path.join(model_path, SUMMARY_FILENAME)
    tmp = target + ".tmp"
    try:
        os.makedirs(model_path, exist_ok=True)
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(summary_dict, f, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(tmp, target)
        return target
    except Exception:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass
        return None
