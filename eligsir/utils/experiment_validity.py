"""Experiment validity classification (issue #203, DAG lane P0-E of #202).

This module is deliberately **torch-free and side-effect-free** so it can be
unit-tested in the lean CPU CI image (``Dockerfile.ci-cpu``) without any
``tests/conftest.py`` registration -- see issues #423/#462/#478.

Why this exists
---------------
This project has repeatedly drawn *negative architectural conclusions*
("mechanism X has no effect") from short prefix runs that never exercised
the mechanism in question -- the canonical case being an fr3 run that
covered ``126 / 2525 ~= 5%`` of the source trajectory and was nonetheless
read as evidence about replay/forgetting/assimilation. Issue #203 defines
three mutually-exclusive experiment classes; this module makes them
machine-readable, and -- crucially -- **derives what a run actually
realized** instead of trusting what its spec declared.

The declared/achieved split mirrors the pattern this repo already uses for
``holdout_hash`` (the selection *rule*) vs ``realized_holdout_hash`` (the
pool actually realized) vs ``scored_holdout_hash`` (the cameras actually
scored) -- see ``utils/comparison_report.py``. A spec claiming
``FULL_CONTINUAL_VALIDATION`` that realized a 5% prefix must say so loudly
in its own ``report.json``.

The three classes (verbatim intent from #203)
---------------------------------------------
``SMOKE_ONLY``
    Tiny prefix / wiring check. May conclude ONLY: the code path works or
    does not, a knob bound or did not bind, a catastrophic regression.
    FORBIDDEN from concluding "no effect" for replay, forgetting,
    assimilation, capacity, mutation cadence, scene growth, or long-term
    stability.
``DISTRIBUTED_TRAJECTORY_PROXY``
    Observations sampled across the WHOLE source trajectory (not a leading
    prefix), chronological/causal ordering preserved, multiple spatial
    regions. The preferred cheap proxy for research discrimination.
``FULL_CONTINUAL_VALIDATION``
    Complete representative trajectory + full mapping phase + specified
    refinement. Required for any promotion or strong negative result.

Two extra values exist and are NOT permissive (requirement D of the #203
work order, consistent with PR #497's ``measurement_incomplete``
precedent):

``UNDECLARED``
    The run declared no class. Never silently upgraded to anything.
``UNKNOWN``
    The achieved class could not be derived from this run's own artifacts
    (no registered source-frame total, no source-index semantics, a
    non-streaming run, ...). ``UNKNOWN`` sits at the BOTTOM of the lattice,
    so a declared-``FULL_CONTINUAL_VALIDATION`` run whose achieved class is
    ``UNKNOWN`` does **not** inherit permission from its own declaration.

Ordering / the effective class
------------------------------
``UNKNOWN == UNDECLARED < SMOKE_ONLY < DISTRIBUTED_TRAJECTORY_PROXY <
FULL_CONTINUAL_VALIDATION``. The load-bearing value is
``effective = min(declared, achieved)`` -- the *more conservative* of what
was claimed and what was realized. Every permission field is derived from
``effective``, never from ``declared``.

Mid-run reports
---------------
A mid-run checkpoint report (``comparison/iter_7000/report.json``) will
legitimately grade ``achieved = SMOKE_ONLY`` even for a run that ends as
``FULL_CONTINUAL_VALIDATION`` -- at iteration 7000 the run genuinely had
only covered a prefix. That is honest, not a bug: read the FINAL report's
block for the run's verdict, never a mid-run one.

Scope fence (what this module does NOT do)
------------------------------------------
It does not select frames, does not define #203's canonical fast proxies,
and does not gate, refuse, or alter any run. It is pure reporting, exactly
like ``git_provenance`` / ``map_reinit_invalidated``.
"""
from __future__ import annotations

from typing import Iterable, List, Optional, Sequence

from eligsir.utils.partition_eval import partition_count, partition_id

# --- The class vocabulary --------------------------------------------------

CLASS_SMOKE_ONLY = "SMOKE_ONLY"
CLASS_DISTRIBUTED_TRAJECTORY_PROXY = "DISTRIBUTED_TRAJECTORY_PROXY"
CLASS_FULL_CONTINUAL_VALIDATION = "FULL_CONTINUAL_VALIDATION"
CLASS_UNDECLARED = "UNDECLARED"
CLASS_UNKNOWN = "UNKNOWN"

#: The values a spec/CLI may legally *declare*. ``UNDECLARED`` is what an
#: absent declaration resolves to; it is never something a spec sets.
DECLARABLE_CLASSES = (
    CLASS_SMOKE_ONLY,
    CLASS_DISTRIBUTED_TRAJECTORY_PROXY,
    CLASS_FULL_CONTINUAL_VALIDATION,
)

#: Conservatism lattice. ``UNKNOWN``/``UNDECLARED`` are the BOTTOM element:
#: absent evidence must never read as verified evidence.
_CLASS_RANK = {
    CLASS_UNKNOWN: 0,
    CLASS_UNDECLARED: 0,
    CLASS_SMOKE_ONLY: 1,
    CLASS_DISTRIBUTED_TRAJECTORY_PROXY: 2,
    CLASS_FULL_CONTINUAL_VALIDATION: 3,
}

# --- Grading thresholds ----------------------------------------------------
# Echoed verbatim into every block as ``thresholds`` so a future reader can
# tell WHICH grading produced a given verdict -- #282 and #302 are both
# "a field's meaning was undocumented and a session misread it".

#: Fraction of the registered source trajectory whose frames must actually
#: have been selected/trained for FULL_CONTINUAL_VALIDATION. Deliberately
#: well below 1.0: a run legitimately reserves every Nth frame as an eval
#: holdout (``--streaming_eval_hold``, default 8) and those are NEVER
#: trained, so a genuinely complete run realizes at most ~87.5% of the
#: source frames at the default, or exactly 80% at ``--streaming_eval_hold
#: 5``. Known limitation: a run with ``--streaming_eval_hold < 5`` reserves
#: more than 20% as holdout and can therefore under-grade to
#: DISTRIBUTED_TRAJECTORY_PROXY even when complete -- under-grading is the
#: safe direction and no such spec exists in ``specs/`` today, but a
#: holdout-aware denominator is the correct eventual fix.
FULL_COVERAGE_FRACTION_MIN = 0.80

#: Fraction of the trajectory the selected frames must SPAN (max-min) for
#: FULL_CONTINUAL_VALIDATION.
FULL_SPAN_FRACTION_MIN = 0.90

#: Span requirement for DISTRIBUTED_TRAJECTORY_PROXY: the selection must
#: reach the far end of the route, not stop in a prefix.
PROXY_SPAN_FRACTION_MIN = 0.80

#: Fraction of trajectory partitions (``utils.partition_eval.partition_count``
#: -- the SAME per-scene-constant binning the report's ``partition_psnr``
#: already uses; a second, divergent definition of "trajectory region" is
#: exactly the drift this repo keeps getting burned by) that must contain at
#: least one selected frame for DISTRIBUTED_TRAJECTORY_PROXY. A run that
#: reaches the end of the route but skipped the whole middle is not a
#: distributed proxy.
PROXY_OCCUPIED_REGION_FRACTION_MIN = 0.60

#: A selection whose lowest source index is within this fraction of the
#: trajectory start is considered to begin at the beginning -- used only to
#: label a low-span selection as a *leading prefix* in the report text.
PREFIX_START_FRACTION_MAX = 0.05

THRESHOLDS = {
    "full_coverage_fraction_min": FULL_COVERAGE_FRACTION_MIN,
    "full_span_fraction_min": FULL_SPAN_FRACTION_MIN,
    "proxy_span_fraction_min": PROXY_SPAN_FRACTION_MIN,
    "proxy_occupied_region_fraction_min": PROXY_OCCUPIED_REGION_FRACTION_MIN,
    "prefix_start_fraction_max": PREFIX_START_FRACTION_MAX,
}

# --- #203 promotion semantics ---------------------------------------------
# "A negative from SMOKE_ONLY that lacks the hypothesized regime must become
# `needs_better_proxy`, NOT `full_negative`." This is the machine-readable
# form of that rule. It is a CEILING on how strong a negative result may be
# labelled; nothing in this repo consumes it as a pipeline yet.
_NEGATIVE_CEILING = {
    CLASS_UNKNOWN: "needs_better_proxy",
    CLASS_UNDECLARED: "needs_better_proxy",
    CLASS_SMOKE_ONLY: "needs_better_proxy",
    CLASS_DISTRIBUTED_TRAJECTORY_PROXY: "provisional_negative",
    CLASS_FULL_CONTINUAL_VALIDATION: "full_negative",
}

#: The mechanisms #203 forbids a SMOKE_ONLY run from concluding "no effect"
#: about. Echoed into the block so the prohibition is readable in place.
FORBIDDEN_SMOKE_CONCLUSION_TOPICS = (
    "replay",
    "forgetting",
    "assimilation",
    "capacity",
    "mutation_cadence",
    "scene_growth",
    "long_term_stability",
)


def normalize_declared_class(value: Optional[str]) -> str:
    """Resolve a spec/CLI declaration to a canonical class name.

    ``None``/empty -> ``UNDECLARED``. An unrecognized string ALSO resolves to
    ``UNDECLARED`` (never to a permissive value); the caller records the
    verbatim rejected string in ``declared_raw`` so a typo is visible rather
    than silently upgraded. Case- and separator-insensitive, so
    ``"smoke_only"``, ``"smoke-only"`` and ``"SMOKE_ONLY"`` all work.
    """
    if value is None:
        return CLASS_UNDECLARED
    key = str(value).strip().upper().replace("-", "_").replace(" ", "_")
    if key in _CLASS_RANK and key in DECLARABLE_CLASSES:
        return key
    return CLASS_UNDECLARED


def class_rank(name: Optional[str]) -> int:
    """Conservatism rank; anything unrecognized ranks at the bottom (0)."""
    return _CLASS_RANK.get(str(name or ""), 0)


def more_conservative(a: str, b: str) -> str:
    """The lower-ranked (more conservative) of two class names.

    Ties prefer ``a`` unless ``a`` is a placeholder and ``b`` is not -- so
    ``more_conservative(UNDECLARED, UNKNOWN)`` is stable and either answer is
    equally non-permissive.
    """
    if class_rank(a) <= class_rank(b):
        return a
    return b


def _distinct_sorted_indices(indices: Optional[Iterable]) -> List[int]:
    out = set()
    for v in indices or ():
        if v is None:
            continue
        try:
            iv = int(v)
        except (TypeError, ValueError):
            continue
        if iv >= 0:
            out.add(iv)
    return sorted(out)


def derive_achieved_class(
    source_indices: Optional[Sequence],
    source_frame_count_total: Optional[int],
    *,
    source_index_provenance: Optional[str] = None,
    is_streaming_run: Optional[bool] = None,
) -> dict:
    """Derive what a run ACTUALLY realized, from its own selected frames.

    ``source_indices``: the source-trajectory frame indices this run actually
    selected/trained (see ``utils.partition_eval.camera_source_index``).
    Holdout/test cameras must NOT be mixed in: holdouts are reserved every
    Nth frame across the whole trajectory *before* admission, so including
    them makes a pure prefix run look distributed.

    ``source_frame_count_total``: the per-scene constant from
    ``eval/scenes.json`` (``registry_n_source_frames``). Without it there is
    no denominator and therefore no derivable class -- the result is
    ``UNKNOWN`` with a verbatim reason, never a guess.

    ``is_streaming_run``: the SECOND half of the gate, and it must be
    explicit. Camera ``uid``/``colmap_id`` only carries source-trajectory
    position for a streaming run (``StreamingScene`` sets
    ``CameraInfo(uid=frame.index, ...)``); on the legacy ``train.py`` path
    every camera is loaded at once with an arbitrary index, there is no
    causal frame ordering and no mapping phase at all. Do NOT rely on the
    registry lookup to exclude those runs: ``scene_source`` comes from the
    ``SCENE_SOURCE`` environment variable, which the launch scripts/ablator
    set for streaming and non-streaming jobs alike, so a legacy
    ``train.py --eval -s /data/tum/rgbd_desk`` run WOULD hit the registry
    and would otherwise grade ``FULL_CONTINUAL_VALIDATION`` with
    ``strong_negative_conclusions_permitted`` -- exactly the
    permissive-by-accident outcome #203 exists to prevent. ``None`` (the
    default) means "not established" and is treated exactly like ``False``.

    Returns a dict of derived evidence -- see ``build_experiment_validity_block``
    for how it is embedded.
    """
    idx = _distinct_sorted_indices(source_indices)
    total = None
    try:
        if source_frame_count_total is not None:
            total = int(source_frame_count_total)
    except (TypeError, ValueError):
        total = None

    out = {
        "achieved": CLASS_UNKNOWN,
        "derivation_reason": None,
        "source_index_provenance": source_index_provenance,
        "is_streaming_run": is_streaming_run,
        "source_frame_count_total": total,
        "source_frames_selected": len(idx) if idx else 0,
        "source_progress_min": None,
        "source_progress_max": None,
        "source_progress_fraction": None,
        "source_progress_span_fraction": None,
        "last_source_frame_reached": (idx[-1] if idx else None),
        "occupied_spatial_cells_or_regions": None,
        "total_spatial_cells_or_regions": None,
        "occupied_region_fraction": None,
        "leading_prefix_only": None,
    }

    if not is_streaming_run:
        out["derivation_reason"] = (
            "not established as a streaming run -- camera uid/colmap_id only "
            "carries source-trajectory position under StreamingScene; a legacy "
            "(non-streaming) run has no causal frame ordering and no mapping "
            "phase, so no validity class is derivable from it"
        )
        return out
    if not idx:
        out["derivation_reason"] = (
            "no source frame indices available -- achieved class not derivable"
        )
        return out
    if not total or total <= 1:
        out["derivation_reason"] = (
            "no registered source_frame_count_total (eval/scenes.json miss, live "
            "or unregistered source, or a non-streaming scene with no "
            "trajectory-position semantics) -- achieved class not derivable"
        )
        return out

    denom = float(total - 1)
    lo, hi = idx[0], idx[-1]
    coverage_fraction = len(idx) / float(total)
    progress_min = lo / denom
    progress_max = min(1.0, hi / denom)
    span_fraction = max(0.0, progress_max - progress_min)

    k = partition_count(total)
    max_source_index = total - 1
    occupied = {partition_id(i, max_source_index, k) for i in idx if i <= max_source_index}
    # Indices past the registered length (a longer local copy of the scene,
    # or a mis-registered n_source_frames) are clamped into the last bin
    # rather than dropped, so they cannot silently vanish from the count.
    if any(i > max_source_index for i in idx):
        occupied.add(k - 1)
    occupied_fraction = len(occupied) / float(k)

    out.update({
        "source_progress_min": round(progress_min, 6),
        "source_progress_max": round(progress_max, 6),
        "source_progress_fraction": round(coverage_fraction, 6),
        "source_progress_span_fraction": round(span_fraction, 6),
        "occupied_spatial_cells_or_regions": len(occupied),
        "total_spatial_cells_or_regions": int(k),
        "occupied_region_fraction": round(occupied_fraction, 6),
        "leading_prefix_only": bool(
            progress_min <= PREFIX_START_FRACTION_MAX
            and span_fraction < PROXY_SPAN_FRACTION_MIN
        ),
    })

    if (coverage_fraction >= FULL_COVERAGE_FRACTION_MIN
            and span_fraction >= FULL_SPAN_FRACTION_MIN):
        out["achieved"] = CLASS_FULL_CONTINUAL_VALIDATION
        out["derivation_reason"] = (
            f"selected {len(idx)}/{total} source frames "
            f"({coverage_fraction:.1%} >= {FULL_COVERAGE_FRACTION_MIN:.0%}) spanning "
            f"{span_fraction:.1%} of the trajectory "
            f"(>= {FULL_SPAN_FRACTION_MIN:.0%})"
        )
    elif (span_fraction >= PROXY_SPAN_FRACTION_MIN
            and occupied_fraction >= PROXY_OCCUPIED_REGION_FRACTION_MIN):
        out["achieved"] = CLASS_DISTRIBUTED_TRAJECTORY_PROXY
        out["derivation_reason"] = (
            f"selected {len(idx)}/{total} source frames ({coverage_fraction:.1%}) "
            f"spanning {span_fraction:.1%} of the trajectory and occupying "
            f"{len(occupied)}/{k} trajectory partitions "
            f"({occupied_fraction:.1%}) -- distributed, but below the "
            f"{FULL_COVERAGE_FRACTION_MIN:.0%} coverage required for full validation"
        )
    else:
        out["achieved"] = CLASS_SMOKE_ONLY
        out["derivation_reason"] = (
            f"selected {len(idx)}/{total} source frames ({coverage_fraction:.1%}) "
            f"spanning only {span_fraction:.1%} of the trajectory and occupying "
            f"{len(occupied)}/{k} trajectory partitions ({occupied_fraction:.1%}) "
            f"-- below the distributed-proxy floor "
            f"(span >= {PROXY_SPAN_FRACTION_MIN:.0%} AND regions >= "
            f"{PROXY_OCCUPIED_REGION_FRACTION_MIN:.0%})"
        )
    return out


def build_experiment_validity_block(
    *,
    declared: Optional[str],
    source_indices: Optional[Sequence] = None,
    source_frame_count_total: Optional[int] = None,
    source_index_provenance: Optional[str] = None,
    is_streaming_run: Optional[bool] = None,
    realized_release_count: Optional[int] = None,
    realized_admission_count: Optional[int] = None,
    source_insert_count: Optional[int] = None,
    optimizer_updates: Optional[int] = None,
    supervision_views: Optional[int] = None,
    supervision_views_unavailable_reason: Optional[str] = None,
    mapping_end_iter: Optional[int] = None,
    final_iteration: Optional[int] = None,
    last_source_frame_reached: Optional[int] = None,
    trajectory_distance_or_proxy: Optional[float] = None,
    world_spatial_extent: Optional[dict] = None,
    is_final_report: Optional[bool] = None,
    eval_valid: Optional[bool] = None,
    n_test: Optional[int] = None,
) -> dict:
    """Assemble the ``report.json["experiment_validity"]`` block.

    Never raises and never mutates its inputs. Every one of #203's
    "Mandatory report fields" is present as a key: a value that could not be
    computed is written as ``None`` **with a reason string** in
    ``field_provenance``, never omitted -- absent must stay distinguishable
    from verified (requirement D; cf. PR #497's ``measurement_incomplete``).

    ``eval_valid``/``n_test`` (issue #1045, optional): the caller's own
    holdout-validity signals (``report.json["eval_valid"]``/``["n_test"]``),
    threaded through purely so this function can raise a loud,
    ``unregistered_scene_low_coverage_risk`` warning for the combination this
    module cannot otherwise see -- an unregistered/undeclared-source-total
    scene (``achieved == UNKNOWN``, e.g. Replica, which is not in
    ``eval/scenes.json``) silently disables coverage derivation entirely, so
    a badly-undercovered run on such a scene got no coverage-validity signal
    of ANY kind (see the issue for the originating case: an 8000-iteration
    Replica run that covered 54/2000 source frames, reported
    ``eval_valid=False``/``n_test=0``, and still silently completed with no
    warning). This is additive and report-only, exactly like every other
    field in this block -- it gates nothing and changes no run's behavior.
    Omitting both (the default) reproduces this function's exact prior
    output.
    """
    declared_norm = normalize_declared_class(declared)
    declared_raw = None if declared is None else str(declared)
    declaration_rejected = bool(
        declared_raw is not None
        and declared_raw.strip() != ""
        and declared_norm == CLASS_UNDECLARED
    )

    derived = derive_achieved_class(
        source_indices,
        source_frame_count_total,
        source_index_provenance=source_index_provenance,
        is_streaming_run=is_streaming_run,
    )
    achieved = derived["achieved"]
    effective = more_conservative(declared_norm, achieved)

    mismatch = class_rank(achieved) < class_rank(declared_norm)
    mismatch_reason = None
    if mismatch:
        mismatch_reason = (
            f"declared {declared_norm} but this run realized {achieved}: "
            f"{derived['derivation_reason']}"
        )

    negative_permitted = class_rank(effective) >= _CLASS_RANK[
        CLASS_DISTRIBUTED_TRAJECTORY_PROXY]
    strong_negative_permitted = (
        effective == CLASS_FULL_CONTINUAL_VALIDATION
    )

    # #203 "Specific regression guard": a run that covers only a small
    # chronological prefix while claiming to be more than a smoke test must
    # be impossible to miss. The historical 126/2525 ~= 5% fr3 case is the
    # named acceptance test for this.
    prefix_guard = bool(
        derived.get("leading_prefix_only")
        and class_rank(declared_norm) > _CLASS_RANK[CLASS_SMOKE_ONLY]
    )

    # #1045: an unregistered/undeclared-total scene (achieved == UNKNOWN)
    # gets NO coverage-validity signal from this module at all -- there is
    # no denominator to derive coverage against. That silence is fine on its
    # own (it is documented, intentional behavior for a registry miss), but
    # combined with a run that ALSO failed the real-holdout contract
    # (`eval_valid is False` or `n_test == 0`, per CLAUDE.md's "Real test
    # holdouts" Run Contract) and was not merely declared a smoke test, the
    # combination is exactly the silent gap the issue names: a short,
    # badly-undercovered run whose top-level completion status looks
    # identical to a genuinely complete, well-covered one. `eval_valid`/
    # `n_test` being omitted (``None``, the default) never triggers this --
    # it only fires when the caller actually supplies the failing signal.
    n_test_is_zero = n_test is not None and int(n_test) == 0
    unregistered_low_coverage_risk = bool(
        achieved == CLASS_UNKNOWN
        and declared_norm != CLASS_SMOKE_ONLY
        and (eval_valid is False or n_test_is_zero)
    )
    unregistered_low_coverage_risk_reason = None
    if unregistered_low_coverage_risk:
        _eval_bits = []
        if eval_valid is False:
            _eval_bits.append("eval_valid=False")
        if n_test_is_zero:
            _eval_bits.append("n_test=0")
        unregistered_low_coverage_risk_reason = (
            f"scene's achieved experiment_validity class is UNKNOWN ({derived['derivation_reason']}), "
            f"so no coverage-validity signal is derivable, AND this run also failed the real-holdout "
            f"contract ({' and '.join(_eval_bits)}), while declared as {declared_norm} (not SMOKE_ONLY) "
            "-- this run's PSNR/coverage must NOT be quoted; see CLAUDE.md's 'Real test holdouts' Run "
            "Contract and issue #1045"
        )

    warnings: List[str] = []
    if unregistered_low_coverage_risk:
        warnings.append(
            f"UNREGISTERED SCENE LOW COVERAGE RISK: {unregistered_low_coverage_risk_reason}"
        )
    if declaration_rejected:
        warnings.append(
            f"unrecognized experiment_validity_class {declared_raw!r} -- treated as "
            f"UNDECLARED (legal values: {', '.join(DECLARABLE_CLASSES)})"
        )
    if declared_norm == CLASS_UNDECLARED and not declaration_rejected:
        warnings.append(
            "no experiment_validity_class declared -- recorded as UNDECLARED "
            "(#203 requires every spec/report to declare exactly one class)"
        )
    if achieved == CLASS_UNKNOWN:
        warnings.append(
            f"achieved class NOT derivable: {derived['derivation_reason']} -- "
            "recorded as UNKNOWN, which is the least permissive value, not a pass"
        )
    if mismatch:
        warnings.append(f"VALIDITY CLASS MISMATCH: {mismatch_reason}")
    if prefix_guard:
        warnings.append(
            "PREFIX GUARD: this run covers only a leading chronological prefix "
            f"(source progress {derived.get('source_progress_min')}.."
            f"{derived.get('source_progress_max')}) while declaring "
            f"{declared_norm}. Per #203 it may NOT be cited as evidence about "
            "replay, forgetting, assimilation, capacity, mutation cadence, "
            "scene growth, or long-term stability."
        )
    if not negative_permitted:
        warnings.append(
            f"negative_conclusions_permitted=False (effective class {effective}): "
            "a null/no-effect result from this run must be recorded as "
            f"'{_NEGATIVE_CEILING[effective]}', never as a full negative."
        )

    post_mapping_span = None
    if mapping_end_iter is not None and final_iteration is not None:
        try:
            post_mapping_span = max(0, int(final_iteration) - int(mapping_end_iter))
        except (TypeError, ValueError):
            post_mapping_span = None

    last_reached = (
        last_source_frame_reached
        if last_source_frame_reached is not None
        else derived.get("last_source_frame_reached")
    )

    field_provenance = {}
    _NO_PROGRESS_REASON = (
        "source-progress not derivable: needs BOTH source frame indices and a "
        "registered source_frame_count_total"
    )

    def _unknown(name, value, reason):
        if value is None:
            field_provenance[name] = reason
        return value

    block = {
        # --- declared vs achieved vs effective ---------------------------
        "declared": declared_norm,
        "declared_raw": declared_raw,
        "declaration_rejected": declaration_rejected,
        "achieved": achieved,
        "effective": effective,
        "mismatch": mismatch,
        "mismatch_reason": mismatch_reason,
        "derivation_reason": derived["derivation_reason"],
        "source_index_provenance": derived["source_index_provenance"],
        "is_streaming_run": derived["is_streaming_run"],
        "is_final_report": is_final_report,

        # --- machine-readable permissions (#203 promotion semantics) -----
        "negative_conclusions_permitted": negative_permitted,
        "strong_negative_conclusions_permitted": strong_negative_permitted,
        "negative_result_status_ceiling": _NEGATIVE_CEILING[effective],
        "forbidden_smoke_conclusion_topics": list(FORBIDDEN_SMOKE_CONCLUSION_TOPICS),
        "prefix_guard_triggered": prefix_guard,
        # --- #1045: unregistered-scene silent coverage-guard gap ---------
        "unregistered_scene_low_coverage_risk": unregistered_low_coverage_risk,
        "unregistered_scene_low_coverage_risk_reason": unregistered_low_coverage_risk_reason,

        # --- #203 "Mandatory report fields" ------------------------------
        "source_frame_count_total": _unknown(
            "source_frame_count_total", derived["source_frame_count_total"],
            "no eval/scenes.json registry hit for this scene_source"),
        "source_frames_selected": derived["source_frames_selected"],
        "source_progress_min": _unknown(
            "source_progress_min", derived["source_progress_min"],
            _NO_PROGRESS_REASON),
        "source_progress_max": _unknown(
            "source_progress_max", derived["source_progress_max"],
            _NO_PROGRESS_REASON),
        "source_progress_fraction": _unknown(
            "source_progress_fraction", derived["source_progress_fraction"],
            _NO_PROGRESS_REASON),
        "source_progress_span_fraction": _unknown(
            "source_progress_span_fraction",
            derived["source_progress_span_fraction"], _NO_PROGRESS_REASON),
        "leading_prefix_only": derived["leading_prefix_only"],
        "last_source_frame_reached": _unknown(
            "last_source_frame_reached", last_reached,
            "no admitted source frame index available"),
        "trajectory_distance_or_proxy": _unknown(
            "trajectory_distance_or_proxy", trajectory_distance_or_proxy,
            "not computed at report time (no cumulative camera-translation "
            "accumulator exists yet) -- remaining #203 scope"),
        "world_spatial_extent": _unknown(
            "world_spatial_extent", world_spatial_extent,
            "not threaded into this block (scene bbox lives in the geometry "
            "block) -- remaining #203 scope"),
        "occupied_spatial_cells_or_regions": _unknown(
            "occupied_spatial_cells_or_regions",
            derived["occupied_spatial_cells_or_regions"],
            "not derivable without a registered source-frame total"),
        "total_spatial_cells_or_regions": derived["total_spatial_cells_or_regions"],
        "occupied_region_fraction": derived["occupied_region_fraction"],
        "mapping_end_iter": _unknown(
            "mapping_end_iter", mapping_end_iter,
            "no _last_ingest_iter available (non-streaming run, or a caller "
            "that does not pass it)"),
        "post_mapping_span": _unknown(
            "post_mapping_span", post_mapping_span,
            "requires both mapping_end_iter and final_iteration"),
        "realized_release_count": _unknown(
            "realized_release_count", realized_release_count,
            "no frame-release counter available for this run type"),
        "realized_admission_count": _unknown(
            "realized_admission_count", realized_admission_count,
            "no keyframe-admission counter available for this run type"),
        "source_insert_count": _unknown(
            "source_insert_count", source_insert_count,
            "depth-insertion accounting only present on the FINAL streaming "
            "report"),
        "optimizer_updates": _unknown(
            "optimizer_updates", optimizer_updates,
            "no final-iteration value passed"),
        # The supervision-view clock as defined in ``utils/realized_cadence.py``
        # (issue #289). Wiring this existing counter into this pre-existing
        # field (issue #203/#289) does NOT answer #426's open Q6 (whether the
        # canonical work counter should be views/pixels/optimizer-updates) --
        # it is exactly one of the candidates listed there, unchanged.
        "supervision_views": _unknown(
            "supervision_views", supervision_views,
            supervision_views_unavailable_reason or (
                "no realized-cadence tracker available for this run (the "
                "offline train.py path, and any streaming run whose caller "
                "predates issue #289, have no supervision-view clock)")),

        # --- grading provenance ------------------------------------------
        "thresholds": dict(THRESHOLDS),
        "field_provenance": field_provenance,
        "warnings": warnings,
        "issue_refs": ["#203", "#202"],
    }
    return block
