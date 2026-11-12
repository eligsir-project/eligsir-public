"""Recover the mapping/refinement phase boundary for a streaming run,
read-only, without re-running anything.

Context (issue #346 follow-up, 2026-08-13): PR #353 ("severity-graded
seed-fade verdict") documented that its severity grader cannot distinguish
a legitimately-low mapping phase from a pathological post-mapping plateau,
because ``report.json`` recorded no machine-readable ``_last_ingest_iter``
(``train_streaming.py``'s own rolling-idle-stop clock -- see that
variable's definition comment above the main training loop). This module
is the fix's read-only half: it lets a future consumer answer "what was
the mapping/refinement boundary for THIS ALREADY-FINISHED run" even for a
run whose report.json predates the live-wiring half of this fix
(``compute_trajectory_coverage``'s new ``last_ingest_iter``/
``final_iteration``/``mapping_phase_source`` fields in
``utils/partition_eval.py``, populated going forward by
``write_post_training_report``'s new ``last_ingest_iter`` /
``true_final_iteration`` / ``mapping_phase_source`` parameters).

Two independent train.log banner phrasings are supported (the message text
changed once during this project's history -- both are real, both are seen
on real runs on disk):

    [streaming] no new frame ingested for 50000 iterations (last at iter=3112)
        — treating source as exhausted/idle; stopping.
    [streaming] no new frame ingested for 50000 optimization steps
        (last at step=3112) — treating source as exhausted/idle; stopping.

This is the EXACT idle-stop banner (the run's real, final rolling-idle-stop
event) -- recovered value is exact. A second, WEAKER signal is the earlier
one-shot stall-WARNING banner (same two phrasings, "-- this may be a
genuine mapping-end, or a silently stalled source"): it only proves a
frame had been ingested by that point, not that no LATER frame ever
arrived (``_stall_warned`` re-arms on the next real ingest, per
``train_streaming.py``), so a value recovered from it is reported as an
approximate LOWER BOUND only, never as exact.

The real final iteration is recovered from the ``[FINAL STEP N]`` (current
banner text) / ``[FINAL ITER N]`` (the same banner's older text, before a
rename) end-of-run save line -- unaffected by the ``report.json["iteration"]``
online_mode sentinel bug (see CLAUDE.md project memory), since it's read
straight out of train.log rather than the JSON.

Nothing in this module writes to any file on disk.
"""
from __future__ import annotations

import os
import re
from typing import Optional

# Idle-stop (real, final) banner -- both historical phrasings.
_IDLE_STOP_RE = re.compile(
    r"no new frame ingested for \d+ (?:iterations|optimization steps) "
    r"\(last at (?:iter|step)=(\d+)\) [—-] treating source as exhausted/idle; stopping\."
)

# Stall-WARNING banner -- both historical phrasings. Weaker: only a lower bound.
_STALL_WARN_RE = re.compile(
    r"no new frame ingested for \d+ (?:iterations|optimization steps) "
    r"\(last at (?:iter|step)=(\d+)\) [—-]{1,2} this may be a genuine mapping-end"
)

# End-of-run exact-final-iteration save banner -- both historical phrasings
# ("FINAL ITER" was renamed to "FINAL STEP"; both are real on runs on disk).
_FINAL_ITER_RE = re.compile(r"\[FINAL (?:STEP|ITER) (\d+)\]")

# CLAUDE.md's "Unique output folder per run" contract: ablator's rerun/requeue
# mechanism reuses the same model_path (does not generate a fresh folder), so
# train.log for a rerun job silently concatenates old and new attempts with
# no separator by default. The loud `[RUN START] <timestamp> host=... pid=...
# model_path=...` banner exists exactly so a reader "always search[es] for
# the LAST such banner and only trust[s] content after it." Applied here
# before any of the regexes above run, so a stale attempt-1 banner never
# outranks a real (possibly banner-less) attempt-2 tail.
_RUN_START_RE = re.compile(r"^\[RUN START\]", re.MULTILINE)


def _truncate_to_last_run_start(log_text: str) -> str:
    matches = list(_RUN_START_RE.finditer(log_text))
    if not matches:
        return log_text
    return log_text[matches[-1].start():]


def extract_last_ingest_iter_from_log(log_text: str) -> tuple:
    """Return ``(last_ingest_iter, confidence)`` recovered from train.log text.

    ``confidence`` is ``"exact"`` (the real idle-stop banner fired) or
    ``"approximate_lower_bound"`` (only a stall-warning banner was found --
    the true value could be later, never earlier). Returns ``(None, None)``
    if neither banner appears (e.g. the run ended by a fixed --iterations
    budget or --streaming_max_iterations without ever going idle -- not
    recoverable from these specific banners; see this module's docstring).

    ``log_text`` is truncated to content after the LAST ``[RUN START]``
    banner first (CLAUDE.md's "Unique output folder per run" contract: a
    rerun-into-the-same-folder job concatenates attempts with no
    separator), so a stale earlier attempt's banner can never outrank the
    real final attempt's. Within that truncated text, if the exact
    idle-stop banner still appears multiple times, the LAST occurrence is
    used.
    """
    log_text = _truncate_to_last_run_start(log_text)
    exact_matches = _IDLE_STOP_RE.findall(log_text)
    if exact_matches:
        return int(exact_matches[-1]), "exact"
    warn_matches = _STALL_WARN_RE.findall(log_text)
    if warn_matches:
        return int(warn_matches[-1]), "approximate_lower_bound"
    return None, None


def extract_final_iteration_from_log(log_text: str) -> Optional[int]:
    """Return the real final optimization step from a ``[FINAL STEP N]`` /
    ``[FINAL ITER N]`` banner, or ``None`` if absent (e.g. a pre-2026-07-24
    log with no such banner at all -- see ``utils/champion_promotion.py``'s
    docstring on a sibling branch for the same gap in a different context).

    Same ``[RUN START]``-truncation as ``extract_last_ingest_iter_from_log``
    -- see its docstring.
    """
    log_text = _truncate_to_last_run_start(log_text)
    matches = _FINAL_ITER_RE.findall(log_text)
    if not matches:
        return None
    return int(matches[-1])


def resolve_mapping_phase(run_dir: str) -> dict:
    """Resolve the mapping-phase boundary for a finished run directory,
    read-only. Never raises; never writes.

    Resolved PER FIELD, not as an all-or-nothing choice between
    report.json and train.log -- a report.json written by a call site that
    doesn't supply ``last_ingest_iter`` (e.g. ``render_report.py``'s
    post-hoc re-render, or the not-yet-wired
    ``rolling_seed_application.py``/``submap_stitch_application.py``
    integration call sites -- see ``write_post_training_report``'s
    docstring) can still have a populated, and even a genuinely correct,
    ``final_iteration`` (it defaults to that call's own ``iteration``
    parameter) while having NO ``last_ingest_iter`` at all. An
    all-or-nothing "does the report already have SOMETHING" gate would
    silently skip train.log parsing in exactly that case and report
    ``last_ingest_iter=None`` even when the idle-stop banner is sitting
    right there in the log -- found in review before this shipped.

    For each of ``last_ingest_iter`` / ``final_iteration``: prefer
    the (numerically-final) ``report.json``'s top-level
    ``trajectory_coverage`` value (via
    ``utils.partition_eval.get_trajectory_coverage`` -- corrected from the
    never-written ``geometry.trajectory_coverage`` path in issue #504) when it
    is not ``None``, else
    fall back to the value recovered from ``<run_dir>/train.log`` (per
    ``extract_last_ingest_iter_from_log`` / ``extract_final_iteration_from_log``).
    ``mapping_phase_source`` is the report's own value when
    ``last_ingest_iter`` came from the report, else the backfill
    confidence label (``"backfilled_exact"`` / ``"backfilled_approximate"``)
    when it came from the log, else ``None``.
    ``refinement_tail_iterations`` is always recomputed from whichever
    ``last_ingest_iter``/``final_iteration`` this function ultimately
    returns -- never taken from the report as-is -- so a report value
    computed against a DIFFERENT final_iteration (e.g. report has
    ``last_ingest_iter`` but no ``final_iteration``, which this function
    then fills in from the log) can't leave a stale, inconsistent tail.

    Returns a dict with keys: ``last_ingest_iter``, ``final_iteration``,
    ``refinement_tail_iterations``, ``mapping_phase_source``, ``detail``.
    """
    report_last_ingest_iter = None
    report_final_iteration = None
    report_mapping_phase_source = None
    report_detail = None
    try:
        # Torch-free source (issue #423/#434). utils.comparison_report re-exports
        # the same function but imports torch at module scope, and the `except`
        # below would swallow that ImportError into a silent "no report found" --
        # i.e. the whole report.json branch would vanish, and this function would
        # fall back to log-scraping, in any environment without torch.
        from eligsir.utils.baseline_comparison import _find_latest_report_json
        from eligsir.utils.partition_eval import get_trajectory_coverage
        import json

        report_path = _find_latest_report_json(run_dir)
        if report_path:
            with open(report_path) as f:
                report = json.load(f)
            # Issue #504: this read `report["geometry"]["trajectory_coverage"]`,
            # a path NO writer in this repo has ever produced (measured: 0 of
            # 400 real reports) -- so this whole report.json branch was dead
            # and every backfill silently fell through to log-scraping, i.e.
            # reported `backfilled_*` provenance when the report's own "live"
            # value was sitting right there. get_trajectory_coverage() reads
            # the real (top-level) location.
            tc = get_trajectory_coverage(report) or {}
            report_last_ingest_iter = tc.get("last_ingest_iter")
            report_final_iteration = tc.get("final_iteration")
            report_mapping_phase_source = tc.get("mapping_phase_source")
            report_detail = f"report.json's own trajectory_coverage ({report_path})"
    except Exception:
        pass

    log_last_ingest_iter, log_confidence = None, None
    log_final_iteration = None
    log_path = os.path.join(run_dir, "train.log")
    log_detail = f"no train.log found at {log_path}"
    if os.path.isfile(log_path):
        try:
            with open(log_path, "r", errors="replace") as f:
                log_text = f.read()
            log_last_ingest_iter, log_confidence = extract_last_ingest_iter_from_log(log_text)
            log_final_iteration = extract_final_iteration_from_log(log_text)
            if log_last_ingest_iter is None:
                log_detail = (
                    f"{log_path} has no idle-stop or stall-warning banner -- this run "
                    "likely ended by a fixed --iterations budget or "
                    "--streaming_max_iterations without ever going idle; not "
                    "recoverable from these banners"
                )
            else:
                log_detail = f"parsed from {log_path} ({log_confidence} last_ingest_iter banner)"
                if log_final_iteration is None:
                    log_detail += "; no [FINAL STEP N]/[FINAL ITER N] banner for final_iteration"
        except Exception as e:
            log_detail = f"train.log unreadable: {e!r}"

    last_ingest_iter = (
        report_last_ingest_iter if report_last_ingest_iter is not None else log_last_ingest_iter
    )
    final_iteration = (
        report_final_iteration if report_final_iteration is not None else log_final_iteration
    )
    if report_last_ingest_iter is not None:
        mapping_phase_source = report_mapping_phase_source
        detail = report_detail
    elif log_last_ingest_iter is not None:
        mapping_phase_source = (
            "backfilled_exact" if log_confidence == "exact" else "backfilled_approximate"
        )
        detail = log_detail
    else:
        mapping_phase_source = None
        detail = report_detail or log_detail

    refinement_tail = None
    if (last_ingest_iter is not None and final_iteration is not None
            and final_iteration >= last_ingest_iter):
        refinement_tail = final_iteration - last_ingest_iter

    return {
        "last_ingest_iter": last_ingest_iter,
        "final_iteration": final_iteration,
        "refinement_tail_iterations": refinement_tail,
        "mapping_phase_source": mapping_phase_source,
        "detail": detail,
    }


def _main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(
        description="Recover the mapping/refinement phase boundary for finished "
                    "streaming run directories (read-only; writes nothing).")
    ap.add_argument("run_dirs", nargs="+")
    args = ap.parse_args(argv)

    print(f"{'run':<44} {'last_ingest':>12} {'final_iter':>12} {'tail':>10} {'source':<22}")
    for d in args.run_dirs:
        r = resolve_mapping_phase(d)
        name = os.path.basename(os.path.normpath(d))
        print(
            f"{name[:44]:<44} "
            f"{(r['last_ingest_iter'] if r['last_ingest_iter'] is not None else '--'):>12} "
            f"{(r['final_iteration'] if r['final_iteration'] is not None else '--'):>12} "
            f"{(r['refinement_tail_iterations'] if r['refinement_tail_iterations'] is not None else '--'):>10} "
            f"{str(r['mapping_phase_source']):<22}"
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
