"""One authoritative, read-only accessor for a streaming run's trajectory
coverage, final iteration, last-ingest iteration, and stream-end validity
(issue #685, parent ledger #614, audit PR #666's finding R7).

Why this exists
----------------
By 2026-08-15 this repo already had FOUR separately-correct pieces that
each solve one slice of "where does the real coverage/iteration data for
a finished run live":

- ``utils.baseline_comparison._find_latest_report_json`` -- resolves the
  real ``<model_path>/comparison/iter_{N}/report.json`` path, NUMERIC
  ``iter_*`` sort (never lexical -- ``iter_2147483647``, the
  ``--online_mode`` sentinel, sorts lexically BEFORE ``iter_30000``
  because ``'2' < '3'``).
- ``utils.partition_eval.get_trajectory_coverage`` -- reads the
  ``trajectory_coverage`` block from its real, TOP-LEVEL location in
  ``report.json`` (issue #504: a prior generation of consumer code read
  ``report["geometry"]["trajectory_coverage"]``, a path no writer in this
  repo has ever produced).
- ``utils.mapping_phase_backfill.resolve_mapping_phase`` -- resolves
  ``last_ingest_iter``/``final_iteration`` per-field, preferring the
  report's own (corrected, top-level) ``trajectory_coverage`` value and
  falling back to parsing ``train.log`` (idle-stop / stall-warning
  banners), truncated to content after the LAST ``[RUN START]`` banner.
- ``utils.champion_promotion.parse_final_iteration_from_log`` -- a
  stronger final-iteration log parser than the one inside
  ``mapping_phase_backfill`` (four-tier confidence: ``[FINAL STEP N]`` /
  ``[FINAL ITER N]`` banner > nearby checkpoint-save line > nearby
  Gaussians-save line > unresolved), also RUN-START-truncated.

Nothing here was individually wrong by 2026-08-15. What was missing was a
SINGLE call a script or doc author could make to get every field at once,
with the report.json's own bogus ``iteration`` sentinel (2**31-1 for
``--online_mode`` runs) never treated as the real final iteration, and a
uniform ``UNKNOWN`` (never ``0`` and never a value implying full coverage)
for any field that cannot be resolved. ``get_run_coverage`` is that call.
It composes the four modules above; it does not recompute
``verdict``/``coverage_fraction`` itself -- those stay owned by
``compute_trajectory_coverage`` in ``utils/partition_eval.py``.

Four distinct quantities (per this project's own documented past
conflation error -- see CLAUDE.md's Run Contracts and issue #685's own
guardrails):

1. **source available** -- ``n_source_frames`` (the scene's real frame
   count from the ``eval/scenes.json`` registry, when the scene resolves
   in it; ``None``/unregistered otherwise).
2. **frame admitted** -- ``frames_admitted`` (post-admission-gate count;
   this is what ``coverage_fraction`` divides into (1)). This is NOT the
   same as "released" -- see (4).
3. **depth inserted** -- ``total_inserted`` (Phase-2 depth-insertion
   count, ``report.json["geometry"]["depth_insertion"]["total_inserted"]``
   -- a DIFFERENT top-level path than ``trajectory_coverage``; do not
   confuse the two "lives under geometry or not" conventions).
4. **source released** -- ``FrameScheduler.should_release()``'s pacing
   decision. **Not persisted anywhere in report.json.** Exposed here as an
   explicit ``frames_released`` field pinned to ``None`` with a
   ``frames_released_unavailable_reason`` string, so a caller cannot
   silently substitute (2) for it -- exactly the conflation CLAUDE.md /
   issue #685 warn about.

Torch-free (issue #423/#434/#449 precedent): every import of a module
that pulls in ``torch``/``numpy`` at module scope (``utils.partition_eval``
does) is deferred into function bodies, mirroring
``utils.mapping_phase_backfill``'s own existing pattern -- so this module
stays importable, and its tests runnable, without a GPU container.

Nothing in this module writes to any file, reruns training, or launches a
container. Read-only, by construction.
"""
from __future__ import annotations

import dataclasses
import glob
import json
import os
import re
from typing import Optional

ONLINE_UNBOUNDED_ITERATIONS = 2147483647

# Trajectory-coverage verdicts that carry the same standing as
# ``eval_valid == False`` per CLAUDE.md's Run Contracts -- never quote a
# run's coverage or PSNR as valid when its resolved verdict is one of these.
_INVALIDATING_VERDICTS = frozenset({"SUSPECTED_TRUNCATED"})

_RUN_START_RE = re.compile(r"^\[RUN START\]", re.MULTILINE)
_ITER_DIR_RE = re.compile(r"^iter_(\d+)$")


def _truncate_to_last_run_start(log_text: str) -> str:
    """Return ``log_text`` truncated to content after the LAST
    ``[RUN START]`` banner (CLAUDE.md's "Unique output folder per run"
    contract: ablator's rerun/requeue mechanism reuses ``model_path``, so a
    reused ``train.log`` silently concatenates attempts with no
    separator). Returns the text unchanged if no banner is present.
    """
    matches = list(_RUN_START_RE.finditer(log_text))
    if not matches:
        return log_text
    return log_text[matches[-1].start():]


def find_latest_report_path(run_dir: str) -> Optional[str]:
    """Resolve the real, final ``report.json`` for ``run_dir``.

    Thin re-export of ``utils.baseline_comparison._find_latest_report_json``
    (numeric ``iter_*`` sort, skips ``_bootstrap_views`` debug variants, no
    bogus ``<run_dir>/report.json`` root fallback). Kept as a separate
    top-level function here (rather than requiring every caller to import
    the private helper directly) so this module is the one place a script
    needs to import from.
    """
    from eligsir.utils.baseline_comparison import _find_latest_report_json

    return _find_latest_report_json(run_dir)


def load_latest_report(run_dir: str) -> tuple:
    """Return ``(report_dict_or_None, report_path_or_None, error_or_None)``
    for ``run_dir``'s latest real report.json. Never raises."""
    try:
        report_path = find_latest_report_path(run_dir)
    except Exception as e:  # pragma: no cover - defensive
        return None, None, f"error resolving report path: {e!r}"
    if not report_path:
        if not os.path.isdir(os.path.join(run_dir, "comparison")):
            return None, None, "no comparison/ dir"
        return None, None, "no iter_N/report.json under comparison/"
    try:
        with open(report_path) as f:
            report = json.load(f)
    except Exception as e:
        return None, report_path, f"report.json unreadable: {e!r}"
    return report, report_path, None


def legacy_naive_report_path(run_dir: str) -> Optional[str]:
    """Reproduce the PRE-fix report-selection bug for measurement purposes
    only: a lexical (string) sort of ``comparison/iter_*`` directory names,
    exactly the bug ``scripts/w3_13_tabulate.py``'s own docstring and
    ``utils.baseline_comparison._find_latest_report_json``'s docstring
    describe (``key=lambda d: d.name`` -- ``'iter_2147483647'`` sorts
    BEFORE ``'iter_30000'`` because ``'2' < '3'``).

    Used only by ``diff_legacy_vs_corrected`` below to measure how many
    real run directories this bug actually picked the wrong report for.
    Never used by any other function in this module for a real read.
    """
    pattern = os.path.join(run_dir, "comparison", "iter_*")
    candidates = []
    for path in glob.glob(pattern):
        name = os.path.basename(path)
        if not os.path.isfile(os.path.join(path, "report.json")):
            continue
        candidates.append(name)
    if not candidates:
        return None
    candidates.sort()  # LEXICAL, deliberately -- this is the bug being measured.
    return os.path.join(run_dir, "comparison", candidates[-1], "report.json")


def legacy_naive_trajectory_coverage(report: Optional[dict]) -> Optional[dict]:
    """Reproduce the PRE-#504 consumer bug for measurement purposes only:
    read ``report["geometry"]["trajectory_coverage"]``, the path no writer
    in this repo has ever produced. Used only by ``diff_legacy_vs_corrected``.
    """
    if not isinstance(report, dict):
        return None
    geom = report.get("geometry")
    if isinstance(geom, dict):
        tc = geom.get("trajectory_coverage")
        if isinstance(tc, dict):
            return tc
    return None


def legacy_naive_final_iteration(report: Optional[dict]) -> Optional[int]:
    """Reproduce the PRE-fix "just read report.json['iteration']" bug: bogus
    ``2**31-1`` sentinel for ``--online_mode`` runs. Measurement-only."""
    if not isinstance(report, dict):
        return None
    it = report.get("iteration")
    return int(it) if isinstance(it, (int, float)) else None


def get_coverage_block(report: Optional[dict]) -> Optional[dict]:
    """Corrected accessor for ``report.json``'s ``trajectory_coverage``
    block. Thin re-export of
    ``utils.partition_eval.get_trajectory_coverage`` (deferred import to
    stay torch-free -- ``utils.partition_eval`` imports ``torch`` at
    module scope). Returns the block verbatim, or ``None`` if genuinely
    absent -- never synthesizes a fabricated "clean" block.
    """
    from eligsir.utils.partition_eval import get_trajectory_coverage

    return get_trajectory_coverage(report)


def get_depth_insertion_block(report: Optional[dict]) -> Optional[dict]:
    """Accessor for ``report.json["geometry"]["depth_insertion"]`` -- a
    DIFFERENT top-level location than ``trajectory_coverage`` (this one
    genuinely does live under ``geometry``; see
    ``utils.comparison_report.depth_insertion_report_block``). Carries
    ``enabled``, ``total_inserted``, ``frames_admitted`` (its OWN copy,
    independent of ``trajectory_coverage["frames_admitted"]`` -- see
    ``get_run_coverage``'s ``frames_admitted_mismatch`` field), and
    ``inert``. Returns ``None`` if absent.
    """
    if not isinstance(report, dict):
        return None
    geom = report.get("geometry")
    if not isinstance(geom, dict):
        return None
    di = geom.get("depth_insertion")
    return di if isinstance(di, dict) else None


def resolve_mapping_phase(run_dir: str) -> dict:
    """Thin re-export of
    ``utils.mapping_phase_backfill.resolve_mapping_phase`` (per-field
    report-then-log resolution of ``last_ingest_iter`` /
    ``final_iteration`` / ``refinement_tail_iterations`` /
    ``mapping_phase_source``)."""
    from eligsir.utils.mapping_phase_backfill import resolve_mapping_phase as _resolve

    return _resolve(run_dir)


def resolve_final_iteration_from_log(run_dir: str):
    """Stronger, confidence-graded final-iteration resolution straight from
    ``train.log`` (thin re-export of
    ``utils.champion_promotion.parse_final_iteration_from_log``): four
    tiers (``final_step_banner``/``final_iter_banner`` high-confidence >
    ``checkpoint_save_near_complete``/``gaussians_save_near_complete``
    medium-confidence > none), always truncated to the LAST ``[RUN START]``
    segment. Returns ``None`` if ``train.log`` is absent, else a
    ``FinalIterationResult`` dataclass instance (see that module).
    """
    from eligsir.utils.champion_promotion import parse_final_iteration_from_log

    log_path = os.path.join(run_dir, "train.log")
    if not os.path.isfile(log_path):
        return None
    with open(log_path, "r", errors="replace") as f:
        log_text = f.read()
    return parse_final_iteration_from_log(log_text)


@dataclasses.dataclass
class CoverageSummary:
    """Everything a script or doc needs to safely quote a run's coverage,
    final iteration, and stream-end validity -- resolved once, from the
    right places, with explicit ``UNKNOWN`` semantics.

    A missing/unresolvable field is ``None`` (read as "UNKNOWN") -- never
    ``0`` and never a value implying full coverage. See each field's
    inline comment for its exact source.
    """

    run_dir: str
    report_path: Optional[str]
    report_found: bool
    load_error: Optional[str]

    # --- trajectory_coverage block (top-level per #504) ---
    coverage_block: Optional[dict]
    # "UNKNOWN_NO_REPORT" when no report was found at all (distinct from
    # partition_eval's own "UNKNOWN_*" verdicts, which require a report).
    verdict: str
    scene_registered: Optional[bool]
    n_source_frames: Optional[int]          # (1) source available
    frames_admitted: Optional[int]          # (2) frame admitted
    coverage_fraction: Optional[float]
    frame_admission: Optional[str]
    stream_end_reason: Optional[str]
    suspected_truncated: bool
    map_reinit_invalidated: Optional[bool]
    mapping_result_valid: Optional[bool]

    # --- depth insertion (geometry.depth_insertion -- separate path) ---
    depth_insertion_enabled: Optional[bool]
    total_inserted: Optional[int]           # (3) depth inserted
    depth_insertion_inert: Optional[bool]
    # True iff trajectory_coverage["frames_admitted"] and
    # geometry.depth_insertion["frames_admitted"] both exist and disagree
    # -- a real conflation/consistency signal, not expected to ever fire
    # on a healthy report (both should be written from the same counter).
    frames_admitted_mismatch: bool

    # --- (4) source released: never persisted, always UNKNOWN ---
    frames_released: None
    frames_released_unavailable_reason: str

    # --- final iteration / last ingest ---
    final_iteration: Optional[int]
    final_iteration_source: str
    final_iteration_confidence: str
    is_online_sentinel: bool                # final_iteration == 2**31-1
    report_iteration_is_bogus_sentinel: bool  # report["iteration"] == 2**31-1
    last_ingest_iter: Optional[int]
    mapping_phase_source: Optional[str]
    refinement_tail_iterations: Optional[int]

    # --- eval validity ---
    eval_valid: Optional[bool]
    n_test: Optional[int]
    holdout_hash: Optional[str]

    warnings: list = dataclasses.field(default_factory=list)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def get_run_coverage(run_dir: str) -> CoverageSummary:
    """The single authoritative entry point: resolve everything about
    ``run_dir``'s trajectory coverage, final iteration, last-ingest
    iteration, and stream-end validity from artifacts already on disk.
    Read-only; never raises; never reruns anything.
    """
    warnings: list = []
    report, report_path, load_error = load_latest_report(run_dir)
    report_found = report is not None

    coverage_block = get_coverage_block(report) if report_found else None
    if coverage_block is None:
        verdict = "UNKNOWN_NO_REPORT" if not report_found else "UNKNOWN_NO_COVERAGE_BLOCK"
    else:
        verdict = coverage_block.get("verdict") or "UNKNOWN_NO_VERDICT_FIELD"

    di_block = get_depth_insertion_block(report) if report_found else None

    tc_frames_admitted = (coverage_block or {}).get("frames_admitted")
    di_frames_admitted = (di_block or {}).get("frames_admitted")
    frames_admitted_mismatch = (
        tc_frames_admitted is not None
        and di_frames_admitted is not None
        and tc_frames_admitted != di_frames_admitted
    )
    if frames_admitted_mismatch:
        warnings.append(
            f"frames_admitted disagreement: trajectory_coverage={tc_frames_admitted} "
            f"vs geometry.depth_insertion={di_frames_admitted}"
        )

    # Final iteration / last-ingest: prefer the report's own (corrected)
    # trajectory_coverage value; fall back to champion_promotion's
    # confidence-graded train.log parser (stronger than
    # mapping_phase_backfill's own internal one) for final_iteration, and
    # to mapping_phase_backfill's idle-stop/stall-warn parser for
    # last_ingest_iter. mapping_phase_backfill.resolve_mapping_phase
    # already implements exactly this per-field preference for both
    # fields using ITS OWN (weaker) log parser, so call it first for
    # last_ingest_iter and mapping_phase_source, then separately try the
    # stronger champion_promotion parser for final_iteration specifically
    # when the report didn't supply one.
    mp = resolve_mapping_phase(run_dir)
    last_ingest_iter = mp.get("last_ingest_iter")
    mapping_phase_source = mp.get("mapping_phase_source")

    final_iteration = (coverage_block or {}).get("final_iteration")
    final_iteration_source = "report.trajectory_coverage" if final_iteration is not None else ""
    final_iteration_confidence = "high" if final_iteration is not None else ""

    report_iteration = report.get("iteration") if report_found else None
    report_iteration_is_bogus_sentinel = report_iteration == ONLINE_UNBOUNDED_ITERATIONS

    if final_iteration is None:
        log_result = resolve_final_iteration_from_log(run_dir)
        if log_result is not None and log_result.iteration is not None:
            final_iteration = log_result.iteration
            final_iteration_source = log_result.source
            final_iteration_confidence = log_result.confidence
            warnings.extend(log_result.warnings)
        elif mp.get("final_iteration") is not None:
            # mapping_phase_backfill's own (weaker) FINAL STEP/ITER regex,
            # in case champion_promotion's parser found nothing (e.g. no
            # "Training complete." anchor at all) but the simpler regex did.
            final_iteration = mp.get("final_iteration")
            final_iteration_source = "mapping_phase_backfill_log"
            final_iteration_confidence = "high"
        elif (
            report_iteration is not None
            and not report_iteration_is_bogus_sentinel
            and isinstance(report_iteration, (int, float))
        ):
            # Only fall back to report["iteration"] when it is NOT the
            # known-bogus online_mode sentinel -- a legacy train.py (--eval,
            # non-streaming) run's "iteration" field is real.
            final_iteration = int(report_iteration)
            final_iteration_source = "report.iteration (non-sentinel)"
            final_iteration_confidence = "medium"
        else:
            final_iteration_source = "unresolved"
            final_iteration_confidence = "none"
            if report_iteration_is_bogus_sentinel:
                warnings.append(
                    "report['iteration'] is the online_mode bogus sentinel "
                    f"({ONLINE_UNBOUNDED_ITERATIONS}) and no real final "
                    "iteration could be recovered from train.log or "
                    "trajectory_coverage.final_iteration -- final_iteration "
                    "is UNKNOWN, not the sentinel."
                )

    is_online_sentinel = final_iteration == ONLINE_UNBOUNDED_ITERATIONS

    cb = coverage_block or {}
    stream_end_reason = cb.get("stream_end_reason")
    suspected_truncated = verdict in _INVALIDATING_VERDICTS

    eval_valid = report.get("eval_valid") if report_found else None
    n_test = report.get("n_test") if report_found else None
    holdout_hash = (
        (report.get("eval_metadata") or {}).get("holdout_hash") if report_found else None
    )

    if load_error:
        warnings.append(load_error)

    return CoverageSummary(
        run_dir=run_dir,
        report_path=report_path,
        report_found=report_found,
        load_error=load_error,
        coverage_block=coverage_block,
        verdict=verdict,
        scene_registered=cb.get("scene_registered"),
        n_source_frames=cb.get("n_source_frames"),
        frames_admitted=tc_frames_admitted,
        coverage_fraction=cb.get("coverage_fraction"),
        frame_admission=cb.get("frame_admission"),
        stream_end_reason=stream_end_reason,
        suspected_truncated=suspected_truncated,
        map_reinit_invalidated=cb.get("map_reinit_invalidated"),
        mapping_result_valid=cb.get("mapping_result_valid"),
        depth_insertion_enabled=(di_block or {}).get("enabled"),
        total_inserted=(di_block or {}).get("total_inserted"),
        depth_insertion_inert=(di_block or {}).get("inert"),
        frames_admitted_mismatch=frames_admitted_mismatch,
        frames_released=None,
        frames_released_unavailable_reason=(
            "FrameScheduler.should_release()'s pacing decision is not "
            "persisted anywhere in report.json -- this is a distinct "
            "quantity from frames_admitted (post-admission-gate count) "
            "and must never be inferred from it."
        ),
        final_iteration=final_iteration,
        final_iteration_source=final_iteration_source,
        final_iteration_confidence=final_iteration_confidence,
        is_online_sentinel=is_online_sentinel,
        report_iteration_is_bogus_sentinel=bool(report_iteration_is_bogus_sentinel),
        last_ingest_iter=last_ingest_iter,
        mapping_phase_source=mapping_phase_source,
        refinement_tail_iterations=mp.get("refinement_tail_iterations"),
        eval_valid=eval_valid,
        n_test=n_test,
        holdout_hash=holdout_hash,
        warnings=warnings,
    )


def diff_legacy_vs_corrected(run_dir: str) -> dict:
    """Measure whether the PRE-fix report-selection/coverage-read bugs
    (lexical ``iter_*`` sort; ``report["geometry"]["trajectory_coverage"]``
    read path; bare ``report["iteration"]`` as final iteration) would have
    produced a DIFFERENT outcome for this specific run directory than the
    corrected accessor does.

    Returns a dict with:
      - ``report_path_differs``: legacy (lexical-sort) vs corrected
        (numeric-sort) picked a different report.json path.
      - ``legacy_verdict`` / ``corrected_verdict``: the trajectory_coverage
        verdict each path would report (``UNKNOWN_NO_REPORT``-equivalent
        strings when no block is found).
      - ``gate_flip``: whether the two verdicts differ in whether they
        would trigger CLAUDE.md's "treat like eval_valid=False" exclusion
        (SUSPECTED_TRUNCATED specifically -- the concrete exclusion
        ``scripts/w3_13_tabulate.py`` documented as silently never firing).
      - ``legacy_final_iteration`` / ``corrected_final_iteration``.
      - ``sentinel_read_as_real``: legacy path returned the bogus
        2**31-1 sentinel as if it were a real final iteration.

    Read-only; never raises.
    """
    corrected = get_run_coverage(run_dir)

    legacy_path = legacy_naive_report_path(run_dir)
    legacy_report = None
    if legacy_path and os.path.isfile(legacy_path):
        try:
            with open(legacy_path) as f:
                legacy_report = json.load(f)
        except Exception:
            legacy_report = None

    legacy_tc = legacy_naive_trajectory_coverage(legacy_report)
    legacy_verdict = (
        (legacy_tc.get("verdict") if legacy_tc else None) or "UNKNOWN_NO_REPORT_OR_BLOCK"
    )
    legacy_final_iteration = legacy_naive_final_iteration(legacy_report)

    report_path_differs = bool(
        legacy_path and corrected.report_path and legacy_path != corrected.report_path
    )
    gate_flip = (legacy_verdict in _INVALIDATING_VERDICTS) != (
        corrected.verdict in _INVALIDATING_VERDICTS
    )
    sentinel_read_as_real = legacy_final_iteration == ONLINE_UNBOUNDED_ITERATIONS

    return {
        "run_dir": run_dir,
        "legacy_report_path": legacy_path,
        "corrected_report_path": corrected.report_path,
        "report_path_differs": report_path_differs,
        "legacy_verdict": legacy_verdict,
        "corrected_verdict": corrected.verdict,
        "gate_flip": gate_flip,
        "legacy_final_iteration": legacy_final_iteration,
        "corrected_final_iteration": corrected.final_iteration,
        "sentinel_read_as_real": sentinel_read_as_real,
    }
