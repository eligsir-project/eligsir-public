"""Champion-run artifact retention: validation, selection, and manifest
building for the champion-promotion workflow (issue #332).

This module is deliberately pure/offline: every function here operates on
already-read strings/dicts/file paths and does no network I/O (no `gh`
calls). That's what lets `tests/test_champion_promotion.py` exercise the
full validation/selection logic against small fixture `report.json`/
`train.log` text without needing real (multi-hundred-MB) run artifacts or
GPU access. `scripts/promote_champion.py` is the thin CLI that wires this
module to the filesystem and to the GitHub Release upload backend.

Design context (see docs/architecture/2026-08-13-champion-artifact-retention-decision.md
for the full writeup and measured evidence):

- Real champion runs already show two of the exact failure modes this
  project's Run Contracts warn about: `fr3par_hybrid_gate`'s `train.log`
  has SIX concatenated "Training complete." markers from a reused output
  folder (see CLAUDE.md's "Bash-tool timeout + naive relaunch" hazard),
  and `floor2loopclosure_loop_closure_fix` predates the 2026-07-24
  final-iteration-PLY-save fix, so its retained PLY (iteration_52000) is
  NOT provably the exact state `report.json` scored (training continued
  to ~53900 per the tqdm progress line). Both are real, not hypothetical,
  which is why the selection logic below defaults to failing loudly
  rather than guessing.
- `[FINAL STEP N]` (current) and `[FINAL ITER N]` (an intermediate name
  used 2026-07-24 before a later rename) are BOTH real banner strings
  seen in this project's history -- match both.
- Older `train_streaming.py` runs (pre-2026-07-24) have neither banner at
  all; the best available signal is the last `[ITER N] Saving Checkpoint`
  (or, failing that, `[ITER N] Saving Gaussians`) line before the final
  "Training complete." marker. This is a genuinely weaker signal
  (periodic-save cadence, not an exact final-step save) and is surfaced
  as `confidence="medium"`, never silently treated as equal to a real
  `[FINAL STEP N]` hit.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import re
from typing import Optional


# ---------------------------------------------------------------------------
# Log parsing: real final iteration
# ---------------------------------------------------------------------------

_RUN_START_RE = re.compile(r"\[RUN START\]")
_FINAL_STEP_RE = re.compile(r"\[FINAL STEP (\d+)\]")
_FINAL_ITER_RE = re.compile(r"\[FINAL ITER (\d+)\]")
# "Training complete." appears both bare (train.py legacy path) and with a
# "[streaming] " prefix (train_streaming.py) -- match either, and dedupe
# same-line double-logging (both prefixed and bare lines are usually
# printed back to back) by tracking line index, not just the number of
# regex hits.
_TRAINING_COMPLETE_RE = re.compile(r"Training complete\.")
_ITER_SAVING_CHECKPOINT_RE = re.compile(r"\[ITER (\d+)\] Saving Checkpoint")
_ITER_SAVING_GAUSSIANS_RE = re.compile(r"\[ITER (\d+)\] Saving Gaussians")

# How many lines to scan backward from the "Training complete." anchor
# line, looking for the nearest preceding checkpoint/PLY save banner, on
# a legacy log with no [FINAL STEP]/[FINAL ITER] marker at all.
_BACKWARD_SCAN_WINDOW = 4000


@dataclasses.dataclass
class FinalIterationResult:
    iteration: Optional[int]
    source: str  # "final_step_banner" | "final_iter_banner" |
    # "checkpoint_save_near_complete" | "gaussians_save_near_complete" |
    # "none"
    confidence: str  # "high" | "medium" | "none"
    n_run_start_banners: int
    n_training_complete_markers: int
    warnings: list = dataclasses.field(default_factory=list)


def parse_final_iteration_from_log(log_text: str) -> FinalIterationResult:
    """Best-effort extraction of the REAL final training iteration from a
    train.log, per CLAUDE.md's "always trust the LAST banner" rule for
    reused output folders.

    Never trusts a lexically-sorted comparison/iter_* directory name (the
    documented cause of two real mistakes this session) and never trusts
    report.json's own "iteration" field alone (bogus sentinel for
    online_mode runs -- see the online-mode-report-iteration-sentinel-bug
    memory).
    """
    lines = log_text.splitlines()
    warnings: list = []

    run_start_idxs = [i for i, line in enumerate(lines) if _RUN_START_RE.search(line)]
    n_run_start = len(run_start_idxs)

    anchors: list = []
    for i, line in enumerate(lines):
        m = _FINAL_STEP_RE.search(line)
        if m:
            anchors.append((i, "final_step_banner", int(m.group(1))))
            continue
        m = _FINAL_ITER_RE.search(line)
        if m:
            anchors.append((i, "final_iter_banner", int(m.group(1))))
            continue
        if _TRAINING_COMPLETE_RE.search(line):
            anchors.append((i, "training_complete", None))

    n_training_complete = sum(1 for _, kind, _ in anchors if kind == "training_complete")

    if n_run_start > 1:
        warnings.append(
            f"train.log contains {n_run_start} [RUN START] banners -- this "
            "output folder was reused across multiple launch attempts "
            "(see CLAUDE.md's 'Bash-tool timeout + naive relaunch' "
            "hazard). Only content after the LAST banner is trusted."
        )
    if n_training_complete > 1:
        warnings.append(
            f"train.log contains {n_training_complete} 'Training complete.' "
            "markers -- multiple training attempts are concatenated in "
            "this log. Only the LAST one is used to locate the real "
            "final iteration."
        )

    if not anchors:
        return FinalIterationResult(
            iteration=None,
            source="none",
            confidence="none",
            n_run_start_banners=n_run_start,
            n_training_complete_markers=n_training_complete,
            warnings=warnings + [
                "No [FINAL STEP N]/[FINAL ITER N]/'Training complete.' "
                "marker found in train.log -- real final iteration could "
                "not be determined automatically. Pass --final-iteration "
                "explicitly (recorded in the manifest as human-asserted, "
                "not log-derived)."
            ],
        )

    if run_start_idxs:
        last_start = run_start_idxs[-1]
        anchors = [a for a in anchors if a[0] >= last_start] or anchors

    last_idx, last_kind, last_val = anchors[-1]

    if last_kind == "final_step_banner":
        return FinalIterationResult(
            iteration=last_val,
            source="final_step_banner",
            confidence="high",
            n_run_start_banners=n_run_start,
            n_training_complete_markers=n_training_complete,
            warnings=warnings,
        )
    if last_kind == "final_iter_banner":
        return FinalIterationResult(
            iteration=last_val,
            source="final_iter_banner",
            confidence="high",
            n_run_start_banners=n_run_start,
            n_training_complete_markers=n_training_complete,
            warnings=warnings,
        )

    window_start = max(0, last_idx - _BACKWARD_SCAN_WINDOW)
    final_step_hit = None
    final_iter_hit = None
    checkpoint_hit = None
    gaussians_hit = None
    for i in range(last_idx, window_start - 1, -1):
        line = lines[i]
        if final_step_hit is None:
            m = _FINAL_STEP_RE.search(line)
            if m:
                final_step_hit = int(m.group(1))
                break
        if final_iter_hit is None:
            m = _FINAL_ITER_RE.search(line)
            if m:
                final_iter_hit = int(m.group(1))
        if checkpoint_hit is None:
            m = _ITER_SAVING_CHECKPOINT_RE.search(line)
            if m:
                checkpoint_hit = int(m.group(1))
        if gaussians_hit is None:
            m = _ITER_SAVING_GAUSSIANS_RE.search(line)
            if m:
                gaussians_hit = int(m.group(1))

    if final_step_hit is not None:
        return FinalIterationResult(
            iteration=final_step_hit,
            source="final_step_banner",
            confidence="high",
            n_run_start_banners=n_run_start,
            n_training_complete_markers=n_training_complete,
            warnings=warnings,
        )
    if final_iter_hit is not None:
        return FinalIterationResult(
            iteration=final_iter_hit,
            source="final_iter_banner",
            confidence="high",
            n_run_start_banners=n_run_start,
            n_training_complete_markers=n_training_complete,
            warnings=warnings,
        )
    if checkpoint_hit is not None:
        return FinalIterationResult(
            iteration=checkpoint_hit,
            source="checkpoint_save_near_complete",
            confidence="medium",
            n_run_start_banners=n_run_start,
            n_training_complete_markers=n_training_complete,
            warnings=warnings + [
                "No [FINAL STEP N]/[FINAL ITER N] banner (pre-2026-07-24 "
                "log format) -- real final iteration inferred from the "
                "last periodic '[ITER N] Saving Checkpoint' line before "
                "'Training complete.'. This is a PERIODIC save point, not "
                "necessarily the exact final optimization step -- treat "
                "with medium confidence and verify the PLY/checkpoint at "
                "this iteration is the one report.json actually scored."
            ],
        )
    if gaussians_hit is not None:
        return FinalIterationResult(
            iteration=gaussians_hit,
            source="gaussians_save_near_complete",
            confidence="medium",
            n_run_start_banners=n_run_start,
            n_training_complete_markers=n_training_complete,
            warnings=warnings + [
                "No [FINAL STEP N]/[FINAL ITER N]/checkpoint-save banner "
                "found -- real final iteration inferred from the last "
                "periodic '[ITER N] Saving Gaussians' line before "
                "'Training complete.'. No checkpoint was ever saved near "
                "the end of this run (no resumable state will exist for "
                "this champion unless --allow-missing-checkpoint is used)."
            ],
        )

    return FinalIterationResult(
        iteration=None,
        source="none",
        confidence="none",
        n_run_start_banners=n_run_start,
        n_training_complete_markers=n_training_complete,
        warnings=warnings + [
            "'Training complete.' found but no nearby checkpoint/Gaussians "
            "save banner within the backward scan window -- real final "
            "iteration could not be determined. Pass --final-iteration "
            "explicitly."
        ],
    )


# ---------------------------------------------------------------------------
# report.json discovery
# ---------------------------------------------------------------------------

_BOOTSTRAP_SUFFIX = "_bootstrap_views"


@dataclasses.dataclass
class ReportDirResult:
    report_dir: Optional[str]
    candidates: list
    warnings: list = dataclasses.field(default_factory=list)


def select_report_dir(run_dir: str) -> ReportDirResult:
    """Locate the single real (holdout-split) report.json under
    ``<run_dir>/comparison/iter_*/``.

    Deliberately does NOT pick "the lexically largest iter_* directory" --
    that convention silently picks the wrong report for online_mode runs,
    whose directory name is the ONLINE_UNBOUNDED_ITERATIONS sentinel
    (2147483647) rather than the real final iteration, and has caused real
    mistakes in this project (see CLAUDE.md). Instead this enumerates
    every real (non-bootstrap-debug) report.json, requires split_kind to
    be "holdout" (or absent, for very old reports predating that field),
    and refuses to guess if more than one candidate remains.
    """
    comparison_dir = os.path.join(run_dir, "comparison")
    candidates = []
    warnings: list = []
    if not os.path.isdir(comparison_dir):
        return ReportDirResult(report_dir=None, candidates=[], warnings=[
            f"no comparison/ directory found under {run_dir}"
        ])

    for name in sorted(os.listdir(comparison_dir)):
        subdir = os.path.join(comparison_dir, name)
        report_path = os.path.join(subdir, "report.json")
        if not os.path.isfile(report_path):
            continue
        if name.endswith(_BOOTSTRAP_SUFFIX):
            continue
        try:
            with open(report_path) as f:
                report = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            warnings.append(f"{report_path}: failed to parse ({exc}), skipped")
            continue
        split_kind = report.get("split_kind")
        if split_kind not in (None, "holdout"):
            warnings.append(
                f"{report_path}: split_kind={split_kind!r}, not a holdout "
                "report, skipped"
            )
            continue
        candidates.append(subdir)

    if not candidates:
        return ReportDirResult(report_dir=None, candidates=[], warnings=warnings + [
            "no eligible (holdout, non-bootstrap) report.json found"
        ])
    if len(candidates) > 1:
        return ReportDirResult(report_dir=None, candidates=candidates, warnings=warnings + [
            "ambiguous: multiple eligible report.json candidates found: "
            + ", ".join(candidates)
            + " -- pass --report-dir explicitly to disambiguate"
        ])
    return ReportDirResult(report_dir=candidates[0], candidates=candidates, warnings=warnings)


# ---------------------------------------------------------------------------
# Run Contract validation
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class ContractValidation:
    blockers: list
    warnings: list

    @property
    def ok(self) -> bool:
        return not self.blockers


# Issue #433 / #420 / PR #427. `utils/baseline_comparison.compare_against_baseline`
# grades holdout comparability on this ladder:
#   scored_hash > realized_hash > n_test_metrics_only > n_test_only >
#   hash_only / unverified
# with a family of `*_mismatch` levels that refuse the comparison outright,
# and `holdout_verified is True` reachable ONLY at `scored_hash`.
#
# The promotion gate below CONSUMES those levels rather than re-implementing a
# weaker check of its own. It deliberately does NOT import that module: the
# verdict is already persisted into the candidate's own
# `report.json["baseline_comparison"]` whenever the run was launched with
# `--baseline_run`, so reading it keeps this module pure/offline/torch-free
# (see this file's docstring) and keeps `utils/baseline_comparison.py`
# untouched.
_REFUSAL_VERIFICATION_LEVELS = frozenset({
    "hash_mismatch",
    "realized_hash_mismatch",
    "scored_hash_mismatch",
    "n_test_mismatch",
    "n_test_metrics_mismatch",
})
_STRONGEST_VERIFICATION_LEVEL = "scored_hash"


def validate_report_contract(
    report: dict,
    *,
    force_seed_fade_ack: bool = False,
    allow_unverifiable_scored_subset: bool = False,
    allow_unmanaged: bool = False,
    run_dir: Optional[str] = None,
) -> ContractValidation:
    """Validate a report.json against this project's Run Contracts
    (CLAUDE.md: real test holdouts, seed-fade disclosure, map-reinit /
    truncation invalidation).

    Returns hard blockers (never promotable without an explicit override
    at a higher layer -- this function does not accept per-blocker
    overrides, callers decide) and warnings (disclose, don't block).
    """
    blockers: list = []
    warnings: list = []

    # Issue #1013: canonical champion promotion requires managed ResearchFlow evidence.
    evidence_class = None
    if run_dir:
        from eligsir.utils.run_summary import load_run_summary
        run_sum = load_run_summary(run_dir)
        if run_sum:
            evidence_class = run_sum.get("evidence_class")
    if not evidence_class and isinstance(report, dict):
        evidence_class = report.get("evidence_class") or (
            report.get("experiment_origin", {}).get("evidence_class")
            if isinstance(report.get("experiment_origin"), dict) else None
        )

    if evidence_class == "UNMANAGED":
        if allow_unmanaged:
            warnings.append(
                "run evidence_class is 'UNMANAGED' (direct developer run) -- "
                "promoted as non-canonical champion via --allow-unmanaged."
            )
        else:
            blockers.append(
                "run evidence_class is 'UNMANAGED' (direct/unmanaged developer run) -- "
                "canonical champion promotion requires managed ResearchFlow evidence. "
                "Pass --allow-unmanaged to override for non-canonical promotion."
            )

    eval_valid = report.get("eval_valid")
    if eval_valid is not True:
        blockers.append(f"eval_valid is {eval_valid!r}, must be True (real holdout contract)")

    n_test = report.get("n_test")
    if not isinstance(n_test, int) or n_test <= 0:
        blockers.append(f"n_test is {n_test!r}, must be > 0 (real holdout contract)")

    split_kind = report.get("split_kind")
    if split_kind not in (None, "holdout"):
        blockers.append(f"split_kind is {split_kind!r}, not 'holdout'")

    eval_metadata = report.get("eval_metadata") or {}
    holdout_hash = eval_metadata.get("holdout_hash")
    if not holdout_hash:
        warnings.append(
            "eval_metadata.holdout_hash missing -- this report predates "
            "the holdout-hash comparability contract (issue-referenced in "
            "CLAUDE.md); future comparisons against this champion cannot "
            "be verified as using the same test split."
        )
    # Issue #433: champion self-provenance. A champion is the reference every
    # future run is compared against, so its OWN report must carry the fields
    # `compare_against_baseline` needs to reach `verification_level=
    # "scored_hash"` -- the only level that earns `holdout_verified is True`
    # since PR #427. Both hashes are required, not just the scored one:
    # `_scored_subset_check()` is reached only under a MATCHING
    # `realized_holdout_hash`, so a report carrying a scored hash but no pool
    # hash is structurally incapable of ever being verified at `scored_hash`
    # (see that function's docstring: "a report carrying a scored hash but no
    # pool hash can never be promoted").
    #
    # Missing either field is a BLOCKER, not a warning: issue #420 proved
    # `realized_holdout_hash` presence alone does not establish which cameras
    # a run's PSNR was averaged over (`fr3postmerge_champ_asis`: n_test=315,
    # n_test_metrics=128 -- 187 holdout views never scored). Every report
    # written before 2026-08-13 predates `scored_holdout_hash` and therefore
    # refuses here EXPLICITLY -- the point of this gate is that a legacy
    # report degrades to a loud refusal (overridable, see
    # `allow_unverifiable_scored_subset`), never to a silent pass.
    realized_hash = eval_metadata.get("realized_holdout_hash")
    scored_hash = eval_metadata.get("scored_holdout_hash")
    _missing_provenance = []
    if not realized_hash:
        _missing_provenance.append("eval_metadata.realized_holdout_hash (issue #199)")
    if not scored_hash:
        _missing_provenance.append("eval_metadata.scored_holdout_hash (issue #420)")
    if _missing_provenance:
        _msg = (
            "holdout provenance incomplete: missing "
            + ", ".join(_missing_provenance)
            + ". A comparison against this champion could never reach "
            "verification_level='scored_hash' (the only level that earns "
            "holdout_verified is True, see PR #427), so its mean_test_psnr "
            "cannot be established as measured over a known camera set "
            f"(n_test={report.get('n_test')!r}, "
            f"n_test_metrics={report.get('n_test_metrics')!r})."
        )
        if allow_unverifiable_scored_subset:
            warnings.append(
                _msg + " Promoted anyway via allow_unverifiable_scored_subset."
            )
        else:
            blockers.append(
                _msg + " Reports written before 2026-08-13 all predate "
                "scored_holdout_hash; re-render the report from the retained "
                "checkpoint, or promote explicitly with "
                "allow_unverifiable_scored_subset."
            )

    # Issue #433: when the candidate run was launched with `--baseline_run`,
    # its report already carries `compare_against_baseline`'s verdict. Consume
    # it instead of second-guessing it.
    baseline_comparison = report.get("baseline_comparison")
    if isinstance(baseline_comparison, dict):
        _level = baseline_comparison.get("verification_level")
        _bc_reason = baseline_comparison.get("reason")
        if _level in _REFUSAL_VERIFICATION_LEVELS:
            blockers.append(
                f"baseline_comparison.verification_level is {_level!r} -- "
                "compare_against_baseline REFUSED this run's comparison "
                "against its own declared baseline "
                f"({baseline_comparison.get('baseline_run')!r}). Reason: "
                f"{_bc_reason!r}. A run that is not comparable to its own "
                "baseline must not become the reference other runs are "
                "compared against."
            )
        elif baseline_comparison.get("comparable") is not True:
            # Infrastructure failure (baseline dir deleted, no report.json,
            # unreadable psnr_history) -- not a comparability refusal.
            # Disclose, don't block: a cleaned-up baseline directory must not
            # make an otherwise-valid champion unpromotable.
            warnings.append(
                "baseline_comparison.comparable is "
                f"{baseline_comparison.get('comparable')!r} at "
                f"verification_level={_level!r} (reason: {_bc_reason!r}) -- "
                "the baseline could not be loaded/compared. Not treated as a "
                "comparability refusal; verify manually if this champion's "
                "claim rests on that comparison."
            )
        elif _level != _STRONGEST_VERIFICATION_LEVEL:
            warnings.append(
                f"baseline_comparison.verification_level is {_level!r} "
                f"(holdout_verified={baseline_comparison.get('holdout_verified')!r}), "
                f"below {_STRONGEST_VERIFICATION_LEVEL!r} -- usually because "
                "the BASELINE side is a legacy report predating "
                "scored_holdout_hash. Disclosed, not blocking: this "
                "candidate's own provenance is gated separately above."
            )

    trajectory_coverage = report.get("trajectory_coverage")
    if trajectory_coverage is None:
        warnings.append(
            "trajectory_coverage block missing entirely -- this report "
            "predates map-reinit-invalidation / truncation-verdict "
            "tracking. Manually verify orbslam_live.log has exactly one "
            "'Creation of new map with id:' line (if this is a "
            "ros2://live source) and that the run was not truncated "
            "before promoting."
        )
    else:
        map_reinit = trajectory_coverage.get("map_reinit_invalidated")
        if map_reinit is True:
            blockers.append(
                "trajectory_coverage.map_reinit_invalidated is True -- "
                "ORB-SLAM3 re-initialized mid-run, poses from before/after "
                "are not in a shared coordinate frame (same standing as "
                "eval_valid=False, never promotable)."
            )
        elif map_reinit is None:
            warnings.append(
                "trajectory_coverage.map_reinit_invalidated absent -- "
                "predates the detector (or /slam/tracking_status_stamped "
                "was silently missing on the deployed ORB-SLAM3 image, a "
                "known gap). Manually grep orbslam_live.log / check for "
                "orbslam_map_reinit_detected.json before promoting a "
                "ros2://live champion."
            )

        verdict = trajectory_coverage.get("verdict")
        if verdict == "SUSPECTED_TRUNCATED":
            blockers.append(
                "trajectory_coverage.verdict is SUSPECTED_TRUNCATED -- "
                "same standing as eval_valid=False, never promotable."
            )

    seed_fade = report.get("seed_fade")
    if seed_fade is None:
        warnings.append(
            "seed_fade block missing entirely -- this report predates "
            "seed-fade detection; manually inspect training_curves.svg "
            "before promoting."
        )
    else:
        detected = seed_fade.get("detected")
        recovered = seed_fade.get("recovered")
        from eligsir.utils.seed_fade_severity import ended_healthy

        healthy = ended_healthy(seed_fade)
        if detected is True and healthy is False:
            if force_seed_fade_ack:
                warnings.append(
                    "seed_fade.detected=True and the run ENDED INSIDE AN ACTIVE FADE "
                    f"(recovered_final={seed_fade.get('recovered_final')!r}, "
                    f"fade_active_at_end={seed_fade.get('fade_active_at_end')!r}) -- "
                    "promoted anyway via --force-seed-fade-ack. max_dip_db="
                    f"{seed_fade.get('max_dip_db')!r}."
                )
            else:
                blockers.append(
                    "seed_fade.detected=True and the run ENDED INSIDE AN ACTIVE FADE "
                    f"(recovered_final={seed_fade.get('recovered_final')!r}, "
                    f"fade_active_at_end={seed_fade.get('fade_active_at_end')!r}) -- an "
                    "UNRECOVERED regression; pass --force-seed-fade-ack only after "
                    "reviewing the full fade evidence."
                )
        elif detected is True and healthy is None:
            warnings.append(
                "seed_fade.detected=True and END-OF-RUN FADE STATE IS UNVERIFIABLE: "
                "this report predates issue #282's recovered_final/fade_active_at_end "
                f"fields, and recovered={recovered!r} is a sticky latch that does NOT "
                "mean the run ended healthy. Manually inspect the seed-PSNR curve "
                "before promoting."
            )
        elif detected is True:
            warnings.append(
                "seed_fade.detected=True but the run ended outside the fade "
                "(recovered_final=True) -- disclosed per Run Contract "
                f"(max_dip_db={seed_fade.get('max_dip_db')!r}, "
                f"fade_start_iter={seed_fade.get('fade_start_iter')!r}, "
                f"controller_engaged={seed_fade.get('controller_engaged')!r})."
            )

    # Issue #290 -- resume-invalidated seed-fade verdict. Deliberately a
    # SEPARATE block appended after the detected/recovered chain above rather
    # than a new branch inside it: the two ask different questions (that chain
    # asks "was a fade detected and did it clear"; this asks "is the answer to
    # that question even computable for this run"), and keeping them apart
    # keeps this change off the lines PR #370 is editing.
    _indeterminate = report.get("seed_fade_indeterminate")
    if _indeterminate is None and isinstance(seed_fade, dict):
        _indeterminate = seed_fade.get("indeterminate")
    if _indeterminate is True:
        _reasons = []
        if isinstance(seed_fade, dict):
            _reasons = list(seed_fade.get("indeterminate_reasons") or [])
        blockers.append(
            "seed_fade verdict is INDETERMINATE (issue #290): training resumed "
            "from a checkpoint without the detector's decision state, so the "
            "seed-fade numbers cover only part of the run and are unreliable in "
            "both directions. Same standing as eval_valid=False -- re-run from "
            "scratch rather than promoting."
            + (" Reasons: " + "; ".join(_reasons) if _reasons else "")
        )

    return ContractValidation(blockers=blockers, warnings=warnings)


# ---------------------------------------------------------------------------
# Artifact selection (PLY / checkpoint)
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class ArtifactSelection:
    path: Optional[str]
    iteration_found: Optional[int]
    exact: bool
    warnings: list = dataclasses.field(default_factory=list)
    blockers: list = dataclasses.field(default_factory=list)


def select_ply(
    run_dir: str,
    real_final_iteration: Optional[int],
    *,
    allow_nearest_tolerance: int = 0,
) -> ArtifactSelection:
    """Locate ``point_cloud/iteration_<N>/point_cloud.ply``.

    Defaults to requiring an EXACT match against the real final
    iteration -- the Tier 1 contract is "exact post-flush final PLY
    corresponding to the state scored by the report." A real champion in
    this project's own archive (floor2loopclosure_loop_closure_fix)
    demonstrates why: it predates the 2026-07-24 final-iteration-PLY-save
    fix, so its only retained PLY (iteration_52000) is a periodic save,
    NOT provably the state report.json scored (training continued to
    ~53900 per the run's own tqdm progress line). That kind of gap must
    be a loud failure, not a silent nearest-match.
    """
    pc_dir = os.path.join(run_dir, "point_cloud")
    if not os.path.isdir(pc_dir):
        return ArtifactSelection(None, None, False, blockers=[
            f"no point_cloud/ directory under {run_dir}"
        ])

    available = {}
    for name in os.listdir(pc_dir):
        m = re.match(r"iteration_(\d+)$", name)
        if not m:
            continue
        ply_path = os.path.join(pc_dir, name, "point_cloud.ply")
        if os.path.isfile(ply_path):
            available[int(m.group(1))] = ply_path

    if not available:
        return ArtifactSelection(None, None, False, blockers=[
            f"no point_cloud.ply found under any point_cloud/iteration_*/ in {run_dir}"
        ])

    if real_final_iteration is None:
        return ArtifactSelection(None, None, False, blockers=[
            "real final iteration unknown -- cannot verify PLY exactness "
            "(pass --final-iteration)"
        ])

    if real_final_iteration in available:
        return ArtifactSelection(
            available[real_final_iteration], real_final_iteration, True,
        )

    nearest_iter = min(available, key=lambda i: abs(i - real_final_iteration))
    gap = abs(nearest_iter - real_final_iteration)
    if allow_nearest_tolerance > 0 and gap <= allow_nearest_tolerance:
        return ArtifactSelection(
            available[nearest_iter], nearest_iter, False,
            warnings=[
                f"no exact PLY at iteration {real_final_iteration}; using "
                f"nearest available (iteration_{nearest_iter}, gap={gap}) "
                "under --allow-nearest-ply -- this is NOT provably the "
                "exact state report.json scored."
            ],
        )
    return ArtifactSelection(None, None, False, blockers=[
        f"no exact PLY at iteration {real_final_iteration} (real final "
        f"iteration); nearest available is iteration_{nearest_iter} "
        f"(gap={gap}). Pass --allow-nearest-ply {gap} to accept this "
        "explicitly, or investigate why the exact-final-iteration save "
        "is missing (this run may predate the 2026-07-24 final-iteration "
        "PLY-save fix, commit bc13d3f/77dd20c)."
    ])


def select_checkpoint(
    run_dir: str,
    real_final_iteration: Optional[int],
    *,
    allow_missing: bool = False,
    allow_nearest_tolerance: int = 0,
) -> ArtifactSelection:
    """Locate ``chkpnt<N>.pth`` at the model-path root.

    Same exactness policy as ``select_ply``: a champion missing an exact
    final checkpoint has no resumable state, which is exactly the gap
    this issue exists to close (confirmed for real on the fr3 champion:
    its train.log shows "[ITER 52000] Saving Checkpoint" but no chkpnt
    file survives on the shared mount today).
    """
    available = {}
    if os.path.isdir(run_dir):
        for name in os.listdir(run_dir):
            m = re.match(r"chkpnt(\d+)\.pth$", name)
            if m:
                available[int(m.group(1))] = os.path.join(run_dir, name)

    if not available:
        if allow_missing:
            return ArtifactSelection(None, None, False, warnings=[
                "no chkpnt*.pth found for this run -- promoted WITHOUT a "
                "resumable checkpoint via --allow-missing-checkpoint. "
                "This champion cannot be resumed/continued, only "
                "re-rendered from its PLY."
            ])
        return ArtifactSelection(None, None, False, blockers=[
            "no chkpnt*.pth found for this run -- pass "
            "--allow-missing-checkpoint to promote without a resumable "
            "checkpoint (not recommended; the Tier 1 contract requires "
            "one)."
        ])

    if real_final_iteration is None:
        return ArtifactSelection(None, None, False, blockers=[
            "real final iteration unknown -- cannot verify checkpoint "
            "exactness (pass --final-iteration)"
        ])

    if real_final_iteration in available:
        return ArtifactSelection(
            available[real_final_iteration], real_final_iteration, True,
        )

    nearest_iter = min(available, key=lambda i: abs(i - real_final_iteration))
    gap = abs(nearest_iter - real_final_iteration)
    if allow_nearest_tolerance > 0 and gap <= allow_nearest_tolerance:
        return ArtifactSelection(
            available[nearest_iter], nearest_iter, False,
            warnings=[
                f"no exact checkpoint at iteration {real_final_iteration}; "
                f"using nearest available (chkpnt{nearest_iter}.pth, "
                f"gap={gap}) under --allow-nearest-checkpoint."
            ],
        )
    if allow_missing:
        return ArtifactSelection(None, None, False, warnings=[
            f"no exact checkpoint at iteration {real_final_iteration} "
            f"(nearest is chkpnt{nearest_iter}.pth, gap={gap}); promoted "
            "WITHOUT a checkpoint via --allow-missing-checkpoint rather "
            "than use a mismatched one."
        ])
    return ArtifactSelection(None, None, False, blockers=[
        f"no exact checkpoint at iteration {real_final_iteration}; "
        f"nearest available is chkpnt{nearest_iter}.pth (gap={gap}). Pass "
        f"--allow-nearest-checkpoint {gap} to accept this explicitly, or "
        "--allow-missing-checkpoint to promote without one."
    ])


# ---------------------------------------------------------------------------
# Hashing / sizing / manifest
# ---------------------------------------------------------------------------

def sha256_file(path: str, chunk_size: int = 1 << 20) -> str:
    """Chunked SHA256 -- safe for the multi-hundred-MB checkpoints this
    workflow handles (a real measured example: 674MB) without loading the
    whole file into memory.
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def file_record(path: str, *, compute_hash: bool = True) -> dict:
    size = os.path.getsize(path)
    record = {"filename": os.path.basename(path), "size_bytes": size}
    if compute_hash:
        record["sha256"] = sha256_file(path)
    return record


def generate_champion_id(scene: str, date_str: str, git_sha: str) -> str:
    short_sha = git_sha[:7] if git_sha else "unknown"
    return f"{scene}-{date_str}-{short_sha}"


GITHUB_RELEASE_ASSET_LIMIT_BYTES = 2 * 1024 * 1024 * 1024


def build_manifest(
    *,
    champion_id: str,
    dataset: str,
    scene: str,
    run_dir: str,
    reason_issue: int,
    real_final_iteration: int,
    final_iteration_source: str,
    final_iteration_confidence: str,
    ply: ArtifactSelection,
    checkpoint: ArtifactSelection,
    report: dict,
    report_dir: str,
    git_sha: str,
    hostname: str,
    date_promoted: str,
    storage_backend: str,
    storage_locator: Optional[str] = None,
    superseded_champion_id: Optional[str] = None,
) -> dict:
    eval_metadata = report.get("eval_metadata") or {}
    seed_fade = report.get("seed_fade") or {}
    trajectory_coverage = report.get("trajectory_coverage") or {}

    manifest = {
        "schema_version": 1,
        "champion_id": champion_id,
        "status": "current",
        "dataset": dataset,
        "scene": scene,
        "run_name": os.path.basename(os.path.normpath(run_dir)),
        "run_dir_origin": run_dir,
        "git_commit_sha": git_sha,
        "hostname": hostname,
        "date_promoted": date_promoted,
        "reason_issue": reason_issue,
        "superseded_champion_id": superseded_champion_id,
        "real_final_iteration": real_final_iteration,
        "final_iteration_source": final_iteration_source,
        "final_iteration_confidence": final_iteration_confidence,
        "final_checkpoint": (
            {**file_record(checkpoint.path), "iteration": checkpoint.iteration_found,
             "exact_final_iteration": checkpoint.exact}
            if checkpoint.path else None
        ),
        "final_ply": (
            {**file_record(ply.path), "iteration": ply.iteration_found,
             "exact_final_iteration": ply.exact}
            if ply.path else None
        ),
        "mandatory_metrics": {
            "report_json": os.path.join(report_dir, "report.json"),
            "mean_test_psnr": report.get("mean_test_psnr"),
            "mean_test_lpips": report.get("mean_test_lpips"),
            "n_test": report.get("n_test"),
            # Issue #420/#433: n_test alone is not the set metrics were
            # averaged over -- the report subsamples the realized pool down to
            # --streaming_report_test_metrics_max first. Persist both the
            # scored-subset count and its hash, or the gate would require a
            # field at promotion time and then throw it away, leaving the
            # champion's manifest unable to support a future `scored_hash`
            # verification.
            "n_test_metrics": report.get("n_test_metrics"),
            "eval_valid": report.get("eval_valid"),
            "holdout_hash": eval_metadata.get("holdout_hash"),
            "realized_holdout_hash": eval_metadata.get("realized_holdout_hash"),
            "scored_holdout_hash": eval_metadata.get("scored_holdout_hash"),
            "seed_fade": seed_fade or None,
            "trajectory_coverage": trajectory_coverage or None,
        },
        "storage_backend": storage_backend,
        "storage_locator": storage_locator,
    }
    return manifest
