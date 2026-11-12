"""Severity grading for seed fade -- the VERDICT layer, downstream of detection.

Issue #346. The detector (``utils.seed_fade_monitor.SeedFadeMonitor``) emits
``detected`` / ``recovered`` / ``controller_engaged`` booleans plus a
``max_dip_db`` float. Two real fr3 runs, both valid comparable evals on the
same scene, produced **identical booleans**::

    issue278mode_iterbased_spf20   detected=T recovered=T engaged=T  max_dip_db=15.56
    fr3postmerge_champ_asis        detected=T recovered=T engaged=T  max_dip_db= 2.93

The first sat pinned at ~9 dB test PSNR for 75,000 of its 100,000 iterations
before spiking to 21.5 dB; the second is a 2.9 dB blip. Both land within
0.27 dB of each other at the end, so an endpoint-only read calls them
equivalent and every boolean agrees. The only thing separating a routine blip
from a catastrophe was a single float that nothing gated on.

This module turns those numbers -- plus the run's FULL ``psnr_history.jsonl``
curve, per CLAUDE.md's "Compare PSNR curves, not just the final number" run
contract -- into an explicit, documented, machine-readable severity grade and
a ``confounded`` boolean that downstream tooling can refuse to quote PSNR
against.

**Reporting-only. Nothing here is read by any training, loss, lambda, or
controller path.** ``SeedFadeMonitor.suppression_scale()`` (the one thing that
does feed training) is untouched by this module and by every call site added
for it.

Grading model
-------------

Two independent axes, deliberately kept separate because the two calibration
anchors above differ on one and agree on the other:

1. **Fade magnitude** (``max_dip_db``, from the detector) -> base severity.
2. **Curve shape** (from ``psnr_history``) -> a one-level escalation.

Base severity from ``max_dip_db``:

===================  =========
``max_dip_db``       severity
===================  =========
detector not fired   ``none``
< 6.0 dB             ``minor``
6.0 - 12.0 dB        ``moderate``
>= 12.0 dB           ``severe``
===================  =========

*Why 6.0*: the geometric midpoint of the two real anchors is
``sqrt(2.93 * 15.56) = 6.75 dB``; 6.0 sits just below it with margin, and is
3x the monitor's own ``tolerance_db`` default (2.0 dB) -- i.e. a dip has to be
three times worse than "worth detecting at all" before it stops being a blip.
*Why 12.0*: a >15x intensity ratio. The 15.56 dB anchor clears it with margin;
nothing resembling a routine blip approaches it.

Escalation (one level, capped at ``severe``) if ANY of:

- the run ends inside an active fade (using ``recovered_final`` /
  ``fade_active_at_end`` when available; the legacy ``recovered`` field is a
  sticky latch and is not interpreted as final health);
- ``curve_shape`` is ``ends_below_peak`` or ``low_plateau_then_rise``;
- ``low_plateau_iter_frac >= 0.35``.

``confounded`` is True iff severity is ``moderate`` or ``severe``: the run's
PSNR must not be quoted without the caveat. ``minor`` stays quotable (with the
number visible), matching how the existing Run Contract already treats a small
detected dip as "note it, check it, then decide".

Curve shape
-----------

``classify_curve_shape`` looks for the specific hazard CLAUDE.md names: a run
that was pinned low for a long stretch and only rose late, versus one that
climbed steadily to the same endpoint. It reports, in priority order:

``insufficient_history``
    Fewer than ``MIN_TICKS`` psnr_history entries -- no shape claim made.
``ends_below_peak``
    Final value more than ``DEPRESSION_DB`` below the run's own best. The run
    ended degraded; the peak was real and was lost.
``low_plateau_then_rise``
    There exists a contiguous iteration span covering at least
    ``PLATEAU_FRAC`` of the run over which the curve is flat (range <=
    ``FLAT_DB``) at a level at least ``DEPRESSION_DB`` below the final value.
    This is the flat-then-late-spike shape.
``low_plateau_in_mapping``
    The same long low plateau, but the mapping/refinement boundary is known
    for this run and the plateau does NOT persist into the post-ingest tail.
    This is the legitimately-low mapping phase, not a pathology -- it does
    not escalate severity. Only reachable when ``mapping_end_iter`` is
    supplied; see the mapping-boundary section below.
``trough_and_recovery``
    Max drawdown below the running maximum reached ``DEPRESSION_DB`` at some
    point, but the curve neither plateaued long nor ended down.
``steady_climb``
    None of the above.

Mapping-boundary awareness (issue #419)
---------------------------------------

A streaming run's mapping phase is legitimately low-PSNR, and quality
legitimately rises once insertion stops (the W3-66 mechanism in CLAUDE.md).
The sharp discriminator is therefore "does the plateau extend past
mapping-end", and this module now makes that distinction **when, and only
when, the boundary is actually resolvable for the run in question**.

``classify_curve_shape``/``grade_seed_fade`` take an optional
``mapping_end_iter`` (the run's ``_last_ingest_iter``).
``grade_seed_fade_for_run`` resolves it automatically via
``utils.mapping_phase_backfill.resolve_mapping_phase``. When it is known and
the curve actually extends past it, the low-plateau search is repeated over
the post-ingest tail alone, and both the shape label and the plateau-duration
escalation are decided on ``post_mapping_plateau_frac`` rather than the
whole-run ``low_plateau_iter_frac``.

**The legacy (no-boundary) path is the PRIMARY path on today's corpus, not
an exotic fallback.** Measured directly with ``grade_run_dir``/
``resolve_mapping_phase`` over this repo's ``output/`` plus
``/path/to/evidence/output/`` on 2026-08-14 (439 run
directories total, 299 with a final ``report.json``):

* ``report.json``'s ``geometry.trajectory_coverage.last_ingest_iter`` --
  the "machine-readable" source -- is present on **0 of 299** runs with a
  final report. The field's wiring (PR #359) landed 2026-08-13; every run
  on disk predates it.
* The ``train.log`` idle-stop backfill recovers a boundary for **58 of 439**
  run directories (27 ``exact``, 31 ``approximate_lower_bound``); of those,
  **38** also clear this module's own ``MIN_POST_MAPPING_TICKS`` /
  ``MIN_POST_MAPPING_SPAN_FRAC`` guard (below) and so grade with
  ``mapping_phase_known=True``. The remaining ~86% of run directories ended
  on a fixed ``--iterations``/``--streaming_max_iterations`` budget without
  ever going idle, and no banner records their boundary.
* Across all 299 gradeable reports, before/after comparison found **zero**
  ``severity``/``confounded``/``curve_shape`` changes from this feature --
  see the PR body for #419 for the full corpus-scan methodology.

**Superseded by issue #504 (2026-08-14, landed after this scan was run):**
the ``report.json`` branch above was dead for a *different* reason than "the
field is new" -- ``resolve_mapping_phase`` was reading
``report["geometry"]["trajectory_coverage"]``, a path no writer in this repo
has ever produced (measured elsewhere at 0 of 400 real reports), instead of
the TOP-LEVEL ``trajectory_coverage`` key every report actually carries.
``utils.mapping_phase_backfill.resolve_mapping_phase`` now reads the correct
location via ``utils.partition_eval.get_trajectory_coverage`` -- but
**re-measured directly (2026-08-14, 834 real
``output/`` + ``comparison/`` + ``report.json`` files on disk): 0 have a
non-null top-level ``trajectory_coverage.last_ingest_iter``, and 0 of the 834
postdate 2026-08-13 (when PR #359 wired the field)**. Every report on disk
predates the wiring entirely, so this scan cannot distinguish "the corrected
path recovers nothing because no run has ever populated the field" from "the
corrected path recovers nothing because no exercising run exists yet" --
those are the same fact stated two ways, not two different claims, and
neither implies a separate defect. The corrected path is simply unexercised
on today's corpus. The 38-run ``mapping_phase_known=True`` figure (reached
purely via ``train.log`` backfill, the only working source measured) is the
real current total; whether a future run with a populated top-level field
recovers more is unmeasured, not "very likely higher" or "a known gap" --
both would be unverified claims this docstring should not make.

So ~13% of the existing corpus gets the new discrimination on this scan, and every other
run -- plus every run whose grading is done without a ``model_path`` --
grades **bit-for-bit as it did before**. That equivalence is asserted by
test, not assumed.

Three deliberate conservatism choices, all to keep this from ever
*inventing* a pathology:

1. The boundary can only **downgrade**: ``low_plateau_then_rise`` becomes
   ``low_plateau_in_mapping``, never the reverse, and the plateau-duration
   escalation consults ``post_mapping_plateau_frac`` **only when the
   whole-run ``low_plateau_iter_frac`` already cleared the same
   threshold**. Without that clamp the boundary could escalate a curve
   ABOVE its legacy grade -- a short tail containing a flat depressed
   stretch can score a high post-mapping fraction while the whole-run
   fraction is far below ``PLATEAU_FRAC`` (a real, constructed case, pinned
   in the tests).
2. A boundary that leaves too little observed tail -- fewer than
   ``MIN_POST_MAPPING_TICKS`` ticks, or less than
   ``MIN_POST_MAPPING_SPAN_FRAC`` of the curve's span -- leaves
   ``mapping_phase_known`` ``False``. This is the mid-run-report case:
   ``write_post_training_report`` runs at every ``--test_iterations``
   checkpoint and passes the CURRENT ``_last_ingest_iter``, which mid-run
   means "most recent ingest so far", not "mapping ended" -- frames are
   still arriving.
3. If ``mapping_end_iter`` lands at or past the last ``psnr_history`` tick
   there is no observed refinement tail to test against, so
   ``mapping_phase_known`` stays ``False`` and legacy grading applies
   (``mapping_boundary_note`` says so). Note also that an
   ``approximate_lower_bound`` boundary errs the safe way for this use: the
   true mapping end can only be *later* than the recovered value, so the
   tail considered is too long, which can only over-escalate, never
   under-escalate.
"""

from __future__ import annotations

import json
import os
from typing import List, Optional, Sequence, Tuple

# --- thresholds (all explicit, all justified in the module docstring) -------

#: Base-severity boundaries on the detector's max_dip_db, in dB.
MINOR_MAX_DIP_DB = 6.0
SEVERE_MAX_DIP_DB = 12.0

#: A curve point is "depressed" when it sits this far below the reference
#: level (the run's own peak, or its final value, depending on the test).
#: Chosen an order of magnitude above the ~0.2 dB tick-to-tick measurement
#: noise measured on real 200k-iteration runs (see
#: utils.psnr_history.check_psnr_plateau's docstring) so it can never fire on
#: noise, and equal to 1.5x the detector's own 2.0 dB tolerance.
DEPRESSION_DB = 3.0

#: Max peak-to-trough range for a span to count as "flat", in dB.
FLAT_DB = 1.5

#: Minimum fraction of the run's iteration span a low plateau must cover
#: before it counts as the flat-then-late-rise shape.
PLATEAU_FRAC = 0.35

#: Fewer psnr_history entries than this and no shape claim is made.
MIN_TICKS = 6

#: Minimum observed post-mapping tail before the mapping boundary is trusted
#: to grade a plateau against (issue #419). Both must hold. A short tail is
#: the mid-run-report case -- write_post_training_report runs at every
#: --test_iterations checkpoint and passes the CURRENT _last_ingest_iter,
#: which mid-run means "most recent ingest so far", not "mapping ended".
MIN_POST_MAPPING_TICKS = 6
MIN_POST_MAPPING_SPAN_FRAC = 0.10

SEVERITY_ORDER = ("none", "minor", "moderate", "severe")
CONFOUNDING_SEVERITIES = ("moderate", "severe")
BAD_SHAPES = ("ends_below_peak", "low_plateau_then_rise")


def _escalate(severity: str) -> str:
    i = SEVERITY_ORDER.index(severity)
    return SEVERITY_ORDER[min(i + 1, len(SEVERITY_ORDER) - 1)]


def ended_healthy(seed_fade: Optional[dict]) -> Optional[bool]:
    """Return whether a detected fade run ended healthy, or ``None`` if unknown.

    Issue #282/#346: ``seed_fade['recovered']`` is a sticky historical latch:
    it becomes true when *any* fade episode closes and is never cleared by a
    later episode. It therefore cannot answer whether the run ended outside a
    fade. New reports carry an explicit end-state signal; legacy reports only
    have one conclusive direction (``detected=True, recovered=False`` means no
    episode ever closed, so the run necessarily ended in fade).
    """
    block = seed_fade or {}

    recovered_final = block.get("recovered_final")
    if isinstance(recovered_final, bool):
        return recovered_final

    fade_active_at_end = block.get("fade_active_at_end")
    if isinstance(fade_active_at_end, bool):
        return not fade_active_at_end

    if block.get("detected") is True and block.get("recovered") is False:
        return False

    return None


def longest_low_plateau(
    iters: Sequence[float],
    vals: Sequence[float],
    *,
    flat_db: float = FLAT_DB,
    depression_db: float = DEPRESSION_DB,
) -> Tuple[float, Optional[float], Optional[float]]:
    """Longest contiguous span that is flat AND far below the final value.

    Returns ``(span_iters, plateau_level_db, deficit_db)`` -- the iteration
    width of the longest qualifying span, the mean value over it, and how far
    below the curve's final value that level sits. ``(0.0, None, None)`` when
    no span qualifies.

    "Qualifying" means: the span's max-min range is <= ``flat_db`` (it really
    is a plateau, not a climb) and its mean is at least ``depression_db``
    below the curve's LAST value (the run really did rise out of it). The
    comparison is against the final value rather than the global peak so that
    an ordinary rising curve -- whose early points are far below its peak
    purely because it is rising -- cannot qualify: a rising span's own range
    exceeds ``flat_db`` long before it gets wide enough to matter.

    Thin wrapper over :func:`longest_low_plateau_interval` -- see that
    function for the ``(start_iter, end_iter)`` bounds this one discards.
    """
    span, level, deficit, _s, _e = longest_low_plateau_interval(
        iters, vals, flat_db=flat_db, depression_db=depression_db)
    return span, level, deficit


def longest_low_plateau_interval(
    iters: Sequence[float],
    vals: Sequence[float],
    *,
    flat_db: float = FLAT_DB,
    depression_db: float = DEPRESSION_DB,
    final_ref: Optional[float] = None,
) -> Tuple[float, Optional[float], Optional[float], Optional[float], Optional[float]]:
    """:func:`longest_low_plateau` plus the plateau's own iteration bounds.

    Returns ``(span_iters, plateau_level_db, deficit_db, start_iter,
    end_iter)``; the last three are ``None`` when no span qualifies. The
    bounds are what lets a caller ask *where* the plateau sat relative to
    some other landmark -- specifically, relative to the mapping/refinement
    boundary (issue #419).

    ``final_ref`` pins the reference value the "depressed" test is made
    against (default: ``vals[-1]``, this curve's own last value). The
    post-mapping pass passes the FULL run's final value explicitly so the
    sub-curve is graded against the same anchor as the whole-curve pass --
    today the two coincide (a tail with a non-empty span always contains the
    final point), so this is defensive, not load-bearing.
    """
    n = len(vals)
    if n < 2:
        return 0.0, None, None, None, None
    final = vals[-1] if final_ref is None else float(final_ref)
    best_span = 0.0
    best_level: Optional[float] = None
    best_bounds: Optional[Tuple[float, float]] = None
    start = 0
    # Two-pointer: extend [start, end] while it stays flat.
    for end in range(1, n):
        while start < end and (max(vals[start:end + 1]) - min(vals[start:end + 1])) > flat_db:
            start += 1
        if end == start:
            continue
        window = vals[start:end + 1]
        level = sum(window) / len(window)
        if final - level < depression_db:
            continue
        span = float(iters[end] - iters[start])
        if span > best_span:
            best_span = span
            best_level = level
            best_bounds = (float(iters[start]), float(iters[end]))
    if best_level is None or best_bounds is None:
        return 0.0, None, None, None, None
    return best_span, best_level, float(final - best_level), best_bounds[0], best_bounds[1]


def max_drawdown_db(vals: Sequence[float]) -> float:
    """Largest drop below the running maximum anywhere in the curve, in dB."""
    running = float("-inf")
    worst = 0.0
    for v in vals:
        if v > running:
            running = v
        worst = max(worst, running - v)
    return worst if worst > 0 else 0.0


def classify_curve_shape(
    iters: Sequence[float],
    vals: Sequence[float],
    *,
    mapping_end_iter: Optional[float] = None,
) -> dict:
    """Shape descriptor for a psnr_history curve. See module docstring.

    Pure function over two parallel sequences; never raises on short or empty
    input (returns ``insufficient_history``).

    ``mapping_end_iter`` is the run's ``_last_ingest_iter`` -- the iteration
    at which the final source frame was ingested, i.e. the mapping →
    refinement boundary (issue #419). When supplied AND the curve actually
    extends past it, the low-plateau finding is repeated over the
    post-mapping tail alone, and a long plateau that does NOT survive into
    that tail is reported as ``low_plateau_in_mapping`` instead of
    ``low_plateau_then_rise``. When it is ``None`` (the common case on the
    existing run corpus -- see the module docstring's availability note) the
    classification is bit-for-bit the pre-#419 legacy behaviour.
    """
    out = {
        "curve_shape": "insufficient_history",
        "curve_n_ticks": len(vals),
        "curve_first_db": None,
        "curve_final_db": None,
        "curve_peak_db": None,
        "curve_max_drawdown_db": None,
        "low_plateau_iter_frac": None,
        "low_plateau_iters": None,
        "low_plateau_level_db": None,
        "low_plateau_deficit_db": None,
        # --- mapping-boundary fields (issue #419) ---
        "mapping_end_iter": None,
        "mapping_phase_known": False,
        "post_mapping_span_iters": None,
        "post_mapping_plateau_iters": None,
        "post_mapping_plateau_frac": None,
        "post_mapping_plateau_level_db": None,
        "mapping_boundary_note": "no mapping_end_iter supplied (legacy grading)",
    }
    if len(vals) < MIN_TICKS or len(iters) != len(vals):
        return out
    span = float(iters[-1] - iters[0])
    if span <= 0:
        return out
    peak = max(vals)
    final = vals[-1]
    out["curve_first_db"] = round(float(vals[0]), 3)
    out["curve_final_db"] = round(float(final), 3)
    out["curve_peak_db"] = round(float(peak), 3)
    drawdown = max_drawdown_db(vals)
    out["curve_max_drawdown_db"] = round(float(drawdown), 3)

    plateau_span, plateau_level, plateau_deficit = longest_low_plateau(iters, vals)
    out["low_plateau_iters"] = round(plateau_span, 1)
    out["low_plateau_iter_frac"] = round(plateau_span / span, 4)
    if plateau_level is not None:
        out["low_plateau_level_db"] = round(float(plateau_level), 3)
        out["low_plateau_deficit_db"] = round(float(plateau_deficit), 3)

    # --- mapping-boundary refinement (issue #419) ---------------------------
    # Only ever DOWNGRADES a low_plateau_then_rise to low_plateau_in_mapping;
    # it can never invent a bad shape that legacy grading would not have
    # reported. See the module docstring for why that direction was chosen.
    post_ok = False
    if mapping_end_iter is not None:
        me = float(mapping_end_iter)
        out["mapping_end_iter"] = (
            int(me) if float(me).is_integer() else me
        )
        post_span = float(iters[-1]) - me
        out["post_mapping_span_iters"] = round(post_span, 1)
        tail = [(i, v) for i, v in zip(iters, vals) if float(i) >= me]
        if post_span <= 0 or len(tail) < 2:
            out["mapping_phase_known"] = False
            out["mapping_boundary_note"] = (
                f"mapping_end_iter={me:g} is at or past the end of the psnr_history curve "
                f"(last tick {float(iters[-1]):g}) -- there is no observed refinement tail to "
                "test the plateau against, so legacy (conservative) grading is used"
            )
        elif (len(tail) < MIN_POST_MAPPING_TICKS
                or post_span < MIN_POST_MAPPING_SPAN_FRAC * span):
            # A too-short tail is the mid-run-report case: every
            # `--test_iterations` checkpoint calls write_post_training_report
            # with the CURRENT `_last_ingest_iter`, which at that moment means
            # "most recent ingest so far", not "mapping ended" -- frames are
            # still arriving. Judging a plateau against a few-hundred-iteration
            # tail there would manufacture a verdict out of noise.
            out["mapping_phase_known"] = False
            out["mapping_boundary_note"] = (
                f"mapping_end_iter={me:g} leaves only {post_span:g} iters / {len(tail)} ticks "
                f"of observed tail (need >= {MIN_POST_MAPPING_TICKS} ticks and "
                f">= {MIN_POST_MAPPING_SPAN_FRAC:.0%} of the curve span) -- too short to judge "
                "the plateau against; likely a mid-run report whose boundary is provisional, "
                "so legacy (conservative) grading is used"
            )
        else:
            post_ok = True
            out["mapping_phase_known"] = True
            t_iters = [float(i) for i, _ in tail]
            t_vals = [float(v) for _, v in tail]
            p_span, p_level, _p_def, _p_s, _p_e = longest_low_plateau_interval(
                t_iters, t_vals, final_ref=final)
            out["post_mapping_plateau_iters"] = round(p_span, 1)
            out["post_mapping_plateau_frac"] = round(p_span / post_span, 4)
            if p_level is not None:
                out["post_mapping_plateau_level_db"] = round(float(p_level), 3)
            out["mapping_boundary_note"] = (
                f"mapping ended at iter {me:g}; refinement tail spans {post_span:g} iters "
                f"({len(tail)} psnr_history ticks)"
            )

    if peak - final >= DEPRESSION_DB:
        out["curve_shape"] = "ends_below_peak"
    elif out["low_plateau_iter_frac"] >= PLATEAU_FRAC:
        if post_ok and out["post_mapping_plateau_frac"] < PLATEAU_FRAC:
            out["curve_shape"] = "low_plateau_in_mapping"
        else:
            out["curve_shape"] = "low_plateau_then_rise"
    elif drawdown >= DEPRESSION_DB:
        out["curve_shape"] = "trough_and_recovery"
    else:
        out["curve_shape"] = "steady_climb"
    return out


def grade_seed_fade(
    seed_fade: Optional[dict],
    psnr_iters: Optional[Sequence[float]] = None,
    psnr_vals: Optional[Sequence[float]] = None,
    *,
    mapping_end_iter: Optional[float] = None,
    mapping_phase_source: Optional[str] = None,
) -> dict:
    """Grade a ``seed_fade`` report block into a severity verdict.

    Returns a NEW dict: the original block's keys, plus ``severity``,
    ``severity_reasons``, ``confounded``, and the curve-shape fields from
    ``classify_curve_shape``. The input dict is never mutated.

    ``psnr_iters``/``psnr_vals`` are the run's own ``psnr_history.jsonl``
    curve. Omitting them (or passing a too-short curve) degrades gracefully:
    grading falls back to the detector scalars alone and ``curve_shape``
    reads ``insufficient_history``, so a run with no curve on disk still gets
    a magnitude-based grade rather than an exception or a silent ``none``.

    ``mapping_end_iter`` (issue #419) is the run's ``_last_ingest_iter``.
    Supplying it lets a long low plateau that is confined to the mapping
    phase grade as ``low_plateau_in_mapping`` and NOT escalate; omitting it
    reproduces the pre-#419 grade exactly. ``mapping_phase_source`` is
    recorded verbatim (``"live"`` / ``"backfilled_exact"`` /
    ``"backfilled_approximate"`` / ``None``) so a reader can tell how the
    boundary was established.
    """
    block = dict(seed_fade or {})
    shape = classify_curve_shape(
        list(psnr_iters or []), list(psnr_vals or []),
        mapping_end_iter=mapping_end_iter,
    )
    block.update(shape)
    block["mapping_phase_source"] = mapping_phase_source

    detected = bool(block.get("detected"))
    if not detected:
        # Severity is a SEED-FADE severity: with no fade detected there is
        # nothing to grade. The curve-shape fields are still emitted above so
        # downstream tooling can act on a bad shape independently.
        block["severity"] = "none"
        block["severity_reasons"] = []
        block["confounded"] = False
        block["end_state_verifiable"] = None
        return block

    try:
        max_dip = float(block.get("max_dip_db") or 0.0)
    except (TypeError, ValueError):
        max_dip = 0.0

    reasons: List[str] = []
    if max_dip >= SEVERE_MAX_DIP_DB:
        severity = "severe"
        reasons.append(f"max_dip_db={max_dip:.2f} >= {SEVERE_MAX_DIP_DB}")
    elif max_dip >= MINOR_MAX_DIP_DB:
        severity = "moderate"
        reasons.append(f"max_dip_db={max_dip:.2f} >= {MINOR_MAX_DIP_DB}")
    else:
        severity = "minor"
        reasons.append(f"max_dip_db={max_dip:.2f} < {MINOR_MAX_DIP_DB}")

    if shape.get("mapping_phase_known"):
        reasons.append(
            f"mapping boundary known (last_ingest_iter={shape['mapping_end_iter']:g}, "
            f"source={mapping_phase_source}); plateau duration judged over the "
            "post-ingest tail"
        )
    elif shape.get("curve_shape") != "insufficient_history":
        reasons.append(
            "mapping boundary unavailable -- legacy whole-run plateau grading "
            f"({shape.get('mapping_boundary_note')})"
        )

    escalations: List[str] = []
    healthy = ended_healthy(block)
    block["end_state_verifiable"] = healthy is not None
    if healthy is False:
        escalations.append("fade active at end of run")
    elif healthy is None:
        reasons.append(
            "end-of-run fade state unverifiable: legacy recovered is a sticky latch"
        )

    if shape["curve_shape"] in BAD_SHAPES:
        escalations.append(f"curve_shape={shape['curve_shape']}")
    # Plateau-duration escalation. When the mapping boundary IS known and the
    # curve really has an observed refinement tail, the duration that matters
    # is the plateau's persistence INTO that tail -- a plateau that ended with
    # ingestion is the legitimately-low mapping phase, not a pathology
    # (issue #419). With no boundary, this falls back to the whole-run
    # fraction, i.e. the pre-#419 behaviour verbatim.
    #
    # CLAMPED: the post-mapping fraction is only ever consulted when the
    # WHOLE-RUN fraction already cleared the same threshold. Without that
    # clamp the boundary could ESCALATE above legacy -- a short tail
    # containing a flat depressed stretch can score post_frac >= PLATEAU_FRAC
    # while the whole-run frac is far below it (real, constructed and pinned
    # in tests/test_seed_fade_severity.py). The whole point of #419 is to stop
    # a legitimate mapping phase from being called pathological, never to
    # invent a pathology legacy grading would not have reported.
    legacy_frac = shape.get("low_plateau_iter_frac")
    if (shape.get("mapping_phase_known") and legacy_frac is not None
            and legacy_frac >= PLATEAU_FRAC):
        frac = shape.get("post_mapping_plateau_frac")
        frac_label = "post_mapping_plateau_frac"
    else:
        frac = legacy_frac
        frac_label = "low_plateau_iter_frac"
    if frac is not None and frac >= PLATEAU_FRAC:
        escalations.append(f"{frac_label}={frac:.2f} >= {PLATEAU_FRAC}")
    if escalations:
        severity = _escalate(severity)
        reasons.append("escalated: " + "; ".join(escalations))

    block["severity"] = severity
    block["severity_reasons"] = reasons
    block["confounded"] = severity in CONFOUNDING_SEVERITIES
    return block


def grade_seed_fade_for_run(
    seed_fade: Optional[dict],
    model_path: Optional[str],
    *,
    mapping_end_iter: Optional[float] = None,
    mapping_phase_source: Optional[str] = None,
) -> dict:
    """``grade_seed_fade`` with the curve AND the mapping/refinement boundary
    read from ``model_path`` itself. Never raises -- an unreadable/absent
    history degrades to scalar-only grading, and an unresolvable boundary
    degrades to the legacy whole-run plateau grading (issue #419).

    The boundary comes from :func:`utils.mapping_phase_backfill.resolve_mapping_phase`,
    which prefers ``report.json``'s own
    ``geometry.trajectory_coverage.last_ingest_iter`` (populated by runs from
    2026-08-13 / PR #359 onward) and otherwise recovers it from the run's
    ``train.log`` idle-stop banner. **On the run corpus that exists today the
    report.json path yields nothing at all** (0 of 299 reports scanned carry
    the field; every one predates the wiring), so the log backfill is the
    real, primary source -- and it too only fires for runs that ended by
    going idle (58 of 439 run directories scanned; 27 of them exact) with
    enough observed post-ingest tail to grade (38 of those 58). Every
    other run grades exactly as it did before this change.

    An explicit ``mapping_end_iter`` (with its ``mapping_phase_source``)
    overrides the on-disk resolution entirely. ``write_post_training_report``
    passes its own live ``_last_ingest_iter`` this way, because at report-write
    time the run's FINAL report.json does not exist yet -- resolving from disk
    there would read an EARLIER checkpoint's (possibly stale, always
    ``None``-or-smaller) value.
    """
    iters: List[float] = []
    vals: List[float] = []
    if model_path:
        try:
            from eligsir.utils.psnr_history_io import read_psnr_history

            iters, vals = read_psnr_history(model_path)
        except Exception:
            iters, vals = [], []
    if model_path and mapping_end_iter is None:
        try:
            from eligsir.utils.mapping_phase_backfill import resolve_mapping_phase

            mp = resolve_mapping_phase(model_path)
            mapping_end_iter = mp.get("last_ingest_iter")
            mapping_phase_source = mp.get("mapping_phase_source")
        except Exception:
            mapping_end_iter, mapping_phase_source = None, None
    return grade_seed_fade(
        seed_fade, iters, vals,
        mapping_end_iter=mapping_end_iter,
        mapping_phase_source=mapping_phase_source,
    )


# --- offline backfill / corpus scan ----------------------------------------


def _final_report_path(run_dir: str) -> Optional[str]:
    """Path to a run's FINAL report.json.

    Thin alias over ``utils.comparison_report._find_latest_report_json`` --
    reused rather than reimplemented so there is exactly one place in the
    repo that knows how to pick a run's final report. That helper already
    encodes both hazards this module cares about: checkpoint directories
    (``comparison/iter_<N>``) MUST be ordered NUMERICALLY, because the
    online-mode sentinel ``iter_2147483647`` sorts BEFORE ``iter_30000`` and
    ``iter_7000`` lexically (a mistake made repeatedly against exactly these
    run directories), and ``iter_<N>_bootstrap_views`` must be skipped as a
    degenerate debug split. Note a real run writes NO top-level
    ``<run_dir>/report.json`` at all -- reports only ever live under
    ``comparison/`` -- so there is deliberately no root fallback here.

    Sourced from ``utils.baseline_comparison`` (torch-free) rather than from
    ``utils.comparison_report``, which re-exports the same function but imports
    torch at module scope. That distinction is load-bearing: the ``except``
    below swallows an ImportError into a silent ``None``, so under the previous
    import this whole module degraded to "every run is ungradeable" in any
    environment without torch -- a wrong answer, not a loud failure (issue #423).
    The lazy import is kept so the module stays importable (and ``python -m
    utils.seed_fade_severity`` usable) if that import ever fails for some other
    reason.
    """
    try:
        from eligsir.utils.baseline_comparison import _find_latest_report_json

        return _find_latest_report_json(run_dir)
    except Exception:
        return None


def grade_run_dir(run_dir: str) -> Optional[dict]:
    """Grade an already-finished run on disk, without re-running anything.

    Reads the run's FINAL ``report.json`` (numerically-selected, see
    ``_final_report_path``) and its ``psnr_history.jsonl``. Returns the graded
    block, or ``None`` if the run has no readable report.
    """
    p = _final_report_path(run_dir)
    if not p:
        return None
    try:
        with open(p) as f:
            report = json.load(f)
    except Exception:
        return None
    graded = grade_seed_fade_for_run(report.get("seed_fade"), run_dir)
    graded["_report_path"] = p
    graded["_mean_test_psnr"] = report.get("mean_test_psnr")
    graded["_n_test"] = report.get("n_test")
    graded["_eval_valid"] = report.get("eval_valid")
    return graded


def _main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(
        description="Backfill seed-fade severity over finished run directories "
                    "(read-only; writes nothing unless --write).")
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--json", action="store_true", help="dump full graded blocks as JSON")
    args = ap.parse_args(argv)

    rows = []
    for d in args.run_dirs:
        g = grade_run_dir(d)
        if g is None:
            continue
        rows.append((os.path.basename(os.path.normpath(d)), g))
    if args.json:
        print(json.dumps({n: g for n, g in rows}, indent=2))
        return 0
    print(f"{'run':<44} {'sev':<9} {'conf':<5} {'dip_db':>7} {'shape':<24} {'plat':>6} {'psnr':>7}")
    for name, g in rows:
        print(f"{name[:44]:<44} {str(g['severity']):<9} {str(g['confounded']):<5} "
              f"{(g.get('max_dip_db') or 0):>7.2f} {str(g.get('curve_shape')):<24} "
              f"{(g.get('low_plateau_iter_frac') if g.get('low_plateau_iter_frac') is not None else -1):>6.2f} "
              f"{(g.get('_mean_test_psnr') or float('nan')):>7.2f}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())