"""Admission/cadence invariants, queue telemetry, and the run-level summary
artifact (issue #602).

Scope: **validation and telemetry only.** This module deliberately implements
no admission policy and no backpressure policy. Those are owned by issues #600
(observation -> admission -> insertion decisions, soft-all vs force-all) and
#601 (the single bounded backpressure owner with representative coalescing).
This module records what those owners *did* and judges whether the record is
self-consistent.

Why a separate module
---------------------
``train_streaming.py`` is ~12k lines and is the file #600/#601 are actively
rewriting. Everything here is pure Python -- no torch, no scene, no GPU, no
imports from the training loop -- so the whole contract is unit-testable on
CPU and adds a minimal surface to that file.

Relationship to the clocks that already exist
---------------------------------------------
This module does **not** re-count anything already counted:

``utils.realized_cadence.RealizedCadenceTracker`` (issue #289)
    Owns the per-optimizer-step cadence clocks: ``source_released``,
    ``source_ingested``, ``candidate_evaluated``, ``keyframe_admitted``,
    ``insertion_commit``, ``inserted_gaussians``, ``supervision_view``,
    ``mcmc_mutation``, plus backlog/queue-age means and the
    requested-vs-realized ``reconcile()`` verdict. ``summarize()`` embeds that
    block verbatim under ``realized_cadence`` rather than duplicating it.
``four_clocks`` (Wave 2)
    Owns **wall-clock Hz** rates and ``map_age``. Complementary, not
    subsumed. Embedded verbatim under ``mapping_clocks`` when supplied.

What this module adds on top is the part neither of those has: the
**disposition** of each mapping candidate (admitted / metadata-only /
coalesced / skipped-for-capacity / skipped-as-redundant), the **pending-queue
percentiles**, the **release -> first-supervision / release -> insertion
latencies**, the **resolved-vs-requested semantics** block, and the
**invariants** that make a #600/#601 regression loud instead of silent.

The ``absent != zero`` rule
---------------------------
Every field whose *producing mechanism does not exist in this run* is emitted
as ``None`` with a reason string -- never as ``0``. ``mapping_coalesced: 0``
on a run predating #601 is indistinguishable from "coalescing shipped and
never fired", and this repo has shipped that exact ambiguity three times
(#282's ``n_samples == 0`` vs ``status == NO_SAMPLES``; #437's absent
``n_test_metrics_failed`` meaning "predates the fix" rather than "nothing
failed"; #727's ``CLOCK_COVERAGE``). The convention followed here is #727's:
a ``{"complete": bool, "note": str}`` entry per unavailable field, surfaced in
``summarize()["field_coverage"]``.

Concretely this means a **declaration** is what turns a field on:

* ``declare_release_policy(...)`` -- the release/cadence semantics.
* ``declare_admission_target(...)`` -- #600's resolved decision contract.
  Until called, ``admission_target_resolved`` / ``admission_budgeted`` /
  ``source_coverage_target`` are ``None`` + reason.
* ``declare_backpressure_owner(...)`` -- #601's bounded owner. Until called,
  the coalesce/skip/pending-bound fields are ``None`` + reason, and every
  invariant that depends on them is **skipped**, not failed. This is what
  keeps landing #602 first from firing spurious violations on every
  pre-#600/#601 run.

Cost
----
Every ``record_*`` is integer arithmetic plus, at most, one bounded-container
append. The pending-percentile reservoir is capped
(``_RESERVOIR_CAP``) and the disposition ledger is a bounded ring
(``_LEDGER_CAP``), so memory is O(1) in run length -- the same standing
invariant ``realized_cadence`` documents: observability must not measurably
slow or grow with training.

Failure policy
--------------
``check_invariants()`` is a **pure function returning violations**. It never
raises. Report generation runs after hours of GPU work, and aborting there
would destroy the whole report to complain about telemetry. The caller prints
loudly and sets ``invariants_violated``; only an explicit ``strict=True``
(used by the tests) turns violations into an exception.
"""

from __future__ import annotations

import random
from typing import Any, Iterable, Optional

# Version of the semantics contract this block encodes. Bump when the MEANING
# of an existing field changes, so a reader can tell the schemas apart without
# guessing from the run's date. Mirrors utils.execution_semantics' versioned
# axis convention and realized_cadence's own ``schema_version``.
ADMISSION_SEMANTICS_VERSION = 1

# Bounded-container caps. Both are O(1)-in-run-length guarantees, not tuning
# knobs -- see the module docstring's Cost section.
_RESERVOIR_CAP = 4096
_LEDGER_CAP = 8192

# The three resolved admission targets #600 defines. Kept as a module constant
# so a typo in a declaration is caught at declare time rather than silently
# producing an unrecognized target that every invariant then skips.
ADMISSION_TARGETS = ("soft_all", "selective", "force_all_stress")

# Terminal dispositions: exactly one may apply to a given mapping candidate.
# ``admitted`` and ``metadata_only`` consume the frame; the ``skipped_*`` and
# ``coalesced`` dispositions decline it. A frame carrying two terminal
# dispositions is the "simultaneously admitted and skipped" hard error.
_ADMITTING = frozenset({"admitted", "metadata_only"})
_DECLINING = frozenset({"coalesced", "skipped_capacity", "skipped_redundant"})
TERMINAL_DISPOSITIONS = _ADMITTING | _DECLINING

# Reason strings for the absent != zero rule. Written once here so the same
# text appears in field_coverage and in any violation that had to be skipped.
_NO_ADMISSION_DECL = (
    "no admission-target declaration was made by this run "
    "(declare_admission_target() never called). This is expected for a run "
    "predating issue #600's decision contract. It is NOT evidence that the "
    "run admitted everything, nor that it admitted a budgeted subset -- the "
    "resolved target is simply unknown."
)
_NO_BACKPRESSURE_DECL = (
    "no backpressure owner was declared by this run "
    "(declare_backpressure_owner() never called). Expected for a run "
    "predating issue #601's bounded owner. Coalescing and capacity/redundancy "
    "skipping did not merely fail to fire -- they were not instrumented, and "
    "may not have existed. A zero here would be a false measurement, so these "
    "fields are null."
)
_NO_RELEASE_DECL = (
    "no release-policy declaration was made by this run "
    "(declare_release_policy() never called)."
)


def _pct(sorted_vals: list, q: float):
    """Nearest-rank percentile over a pre-sorted list. None when empty."""
    if not sorted_vals:
        return None
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    idx = int(round(q * (len(sorted_vals) - 1)))
    return sorted_vals[max(0, min(len(sorted_vals) - 1, idx))]


def _ratio(num, den):
    """num/den as a float, or None when the denominator carries no evidence."""
    if den is None or num is None:
        return None
    try:
        den = float(den)
    except (TypeError, ValueError):
        return None
    if den <= 0:
        return None
    return float(num) / den


class _Reservoir:
    """Bounded, unbiased sample of a stream, for percentile estimation.

    Classic reservoir sampling: the first ``cap`` observations are kept
    outright, after which observation ``n`` replaces a uniformly-chosen
    resident with probability ``cap/n``. Memory is O(cap) regardless of run
    length, and the retained sample is an unbiased draw from the whole stream
    -- which a "keep the first N" or "keep the last N" buffer is not. Exact
    ``count``/``max`` are tracked separately and are never estimates.
    """

    __slots__ = ("cap", "_vals", "count", "_max", "_rng")

    def __init__(self, cap: int = _RESERVOIR_CAP, seed: int = 0):
        self.cap = int(cap)
        self._vals: list = []
        self.count = 0
        self._max = None
        # A module-local RNG instance: sampling must never perturb the global
        # random stream, or enabling telemetry would change training results.
        self._rng = random.Random(seed)

    def add(self, v) -> None:
        if v is None:
            return
        v = float(v)
        self.count += 1
        if self._max is None or v > self._max:
            self._max = v
        if len(self._vals) < self.cap:
            self._vals.append(v)
            return
        j = self._rng.randrange(self.count)
        if j < self.cap:
            self._vals[j] = v

    def summary(self) -> dict:
        vals = sorted(self._vals)
        return {
            "samples": self.count,
            "sampled": len(vals),
            "p50": _pct(vals, 0.50),
            "p95": _pct(vals, 0.95),
            # max is exact over the WHOLE stream, not over the reservoir --
            # a sampled max would understate the peak, which is the single
            # number a bound check depends on.
            "max": self._max,
            "mean": (sum(vals) / len(vals)) if vals else None,
        }


class _DispositionLedger:
    """Bounded frame_id -> disposition ring, for cross-disposition conflicts.

    Only the most recent ``cap`` frames are retained. A conflict between two
    dispositions of the same frame that are more than ``cap`` frames apart
    would be missed -- but such a conflict does not describe any real code
    path (a frame is dispositioned once, within one mapping step), and an
    unbounded dict would violate the O(1) memory guarantee. The bound is
    recorded in the summary so the limitation is visible rather than implied.
    """

    __slots__ = ("cap", "_map", "_order", "conflicts", "examples",
                 "inserted_after_decline")

    def __init__(self, cap: int = _LEDGER_CAP):
        self.cap = int(cap)
        self._map: dict = {}
        self._order: list = []
        self.conflicts = 0
        self.inserted_after_decline = 0
        self.examples: list = []

    def note(self, frame_id, disposition: str) -> Optional[str]:
        """Record a disposition. Returns a conflict description, or None."""
        if frame_id is None:
            return None
        prior = self._map.get(frame_id)
        conflict = None
        if prior is not None and prior != disposition:
            if (prior in _ADMITTING and disposition in _DECLINING) or (
                    prior in _DECLINING and disposition in _ADMITTING):
                conflict = (
                    f"frame {frame_id!r} was classified {prior!r} and then "
                    f"{disposition!r}; these are mutually exclusive")
                self.conflicts += 1
                if len(self.examples) < 10:
                    self.examples.append(conflict)
        self._map[frame_id] = disposition
        self._order.append(frame_id)
        if len(self._order) > self.cap:
            evicted = self._order.pop(0)
            # Only drop the mapping if this id has no newer entry queued.
            if evicted not in self._order:
                self._map.pop(evicted, None)
        return conflict

    def disposition_of(self, frame_id):
        return self._map.get(frame_id)

    def note_insertion(self, frame_id) -> Optional[str]:
        """Flag a source-depth insertion attributed to a declined frame."""
        if frame_id is None:
            return None
        prior = self._map.get(frame_id)
        if prior in _DECLINING:
            self.inserted_after_decline += 1
            msg = (
                f"frame {frame_id!r} was classified {prior!r} but still "
                f"produced a source-depth insertion")
            if len(self.examples) < 10:
                self.examples.append(msg)
            return msg
        return None


class AdmissionCadenceRecorder:
    """Records admission dispositions, queue pressure, and resolved semantics.

    All ``record_*`` methods are safe to call unconditionally; a call for a
    mechanism that was never declared still counts (the declaration governs
    whether a *zero* is reported as a real zero or as ``None`` + reason, not
    whether real events are counted).
    """

    def __init__(self, *, reservoir_cap: int = _RESERVOIR_CAP,
                 ledger_cap: int = _LEDGER_CAP):
        # --- declarations (absent != zero) --------------------------------
        self._release_declared = False
        self.release_policy_requested = None
        self.release_policy_resolved = None
        self.nominal_steps_per_frame = None
        self.effective_spf = None

        self._admission_declared = False
        self.admission_policy_requested = None
        self.admission_target_resolved = None
        self.admission_budgeted = None
        self.source_coverage_target = None
        self.admission_declaration_notes: list = []

        self._backpressure_declared = False
        self.backpressure_owner = None
        self.pending_max_count = None
        self.pending_max_age_ms = None
        self.emergency_transitions: list = []

        self.stress_run_declared = False

        # --- dispositions --------------------------------------------------
        self.mapping_candidates = 0
        self.mapping_admitted = 0
        self.mapping_metadata_only = 0
        self.mapping_coalesced = 0
        self.mapping_skipped_capacity = 0
        self.mapping_skipped_redundant = 0
        self.geometry_inserted_frames = 0
        self.source_inserted_gaussians = 0

        # --- queue pressure -------------------------------------------------
        self._pending_depth = _Reservoir(reservoir_cap)
        self._pending_age = _Reservoir(reservoir_cap)
        # Per-window max age, for the monotonic-growth warning. Bounded by
        # closing windows rather than by per-sample retention.
        self._age_windows: list = []
        self._age_window_max = None
        self._age_window_n = 0
        self.age_window_size = 256

        # --- latencies (in optimizer steps) ----------------------------------
        self._release_step: dict = {}
        self._release_order: list = []
        self._first_supervision_latency = _Reservoir(reservoir_cap)
        self._insertion_latency = _Reservoir(reservoir_cap)

        self._ledger = _DispositionLedger(ledger_cap)

        # True once any real disposition was recorded. Until then, summarize()
        # falls back to deriving candidates/admitted from the realized-cadence
        # clocks, so this artifact answers the acceptance question on a run
        # that predates #600/#601's per-frame disposition call sites.
        self._dispositions_recorded = False

    # ------------------------------------------------------------------ #
    # Declarations                                                        #
    # ------------------------------------------------------------------ #

    def declare_release_policy(self, *, requested=None, resolved=None,
                               nominal_steps_per_frame=None,
                               effective_spf=None) -> None:
        """Record the raw request and the resolved release semantics.

        Both are persisted separately and deliberately: W3-38 established that
        an explicit ``--streaming_steps_per_frame 20`` can be silently clamped
        to 50 by ``MappingScheduler.min_steps_per_frame`` on a registry-hit
        scene. Collapsing request and resolution into one field is what made
        that override invisible in the first place.
        """
        self._release_declared = True
        self.release_policy_requested = requested
        self.release_policy_resolved = resolved
        self.nominal_steps_per_frame = nominal_steps_per_frame
        self.effective_spf = effective_spf

    def declare_admission_target(self, *, policy_requested=None,
                                 target_resolved=None, budgeted=None,
                                 source_coverage_target=None,
                                 stress_run: bool = False) -> None:
        """Record #600's resolved admission decision contract.

        ``policy_requested`` is what the launch surface asked for (the
        ``cfg_args`` value); ``target_resolved`` is what the live path
        actually resolved to. They are kept separate on purpose -- a
        divergence between them is a reported finding, not something to
        arbitrate here.
        """
        self._admission_declared = True
        self.admission_policy_requested = policy_requested
        self.admission_target_resolved = target_resolved
        self.admission_budgeted = (
            None if budgeted is None else bool(budgeted))
        self.source_coverage_target = source_coverage_target
        self.stress_run_declared = bool(stress_run)
        if (target_resolved is not None
                and target_resolved not in ADMISSION_TARGETS):
            self.admission_declaration_notes.append(
                f"unrecognized admission_target_resolved "
                f"{target_resolved!r}; expected one of {ADMISSION_TARGETS}")

    def declare_backpressure_owner(self, *, owner=None, pending_max_count=None,
                                   pending_max_age_ms=None) -> None:
        """Record #601's single bounded backpressure owner and its bounds.

        Calling this is what flips the coalesce/skip counters from
        ``None`` + reason to real, zero-initialized measurements.
        """
        self._backpressure_declared = True
        self.backpressure_owner = owner
        self.pending_max_count = pending_max_count
        self.pending_max_age_ms = pending_max_age_ms

    def record_emergency_transition(self, reason: str, *, at_step=None) -> None:
        """An explicit, deliberate transition out of the configured bound.

        Exceeding the pending bound is a hard error *unless* the run declared
        an emergency transition -- this is the escape hatch that makes the
        error meaningful rather than something operators learn to ignore.
        """
        self.emergency_transitions.append(
            {"reason": str(reason), "at_optimizer_step": at_step})

    # ------------------------------------------------------------------ #
    # Disposition recording                                               #
    # ------------------------------------------------------------------ #

    def record_mapping_candidate(self, frame_id=None) -> None:
        self._dispositions_recorded = True
        self.mapping_candidates += 1
        if frame_id is not None:
            self._ledger.note(frame_id, "candidate")

    def record_mapping_admitted(self, frame_id=None) -> None:
        self._dispositions_recorded = True
        self.mapping_admitted += 1
        self._ledger.note(frame_id, "admitted")

    def record_mapping_metadata_only(self, frame_id=None) -> None:
        """Frame observed and its metadata retained, but not materialized.

        Counts as *observed* for coverage purposes and as *not admitted* for
        exposure purposes -- the distinction the acceptance criterion turns on.
        """
        self._dispositions_recorded = True
        self.mapping_metadata_only += 1
        self._ledger.note(frame_id, "metadata_only")

    def record_mapping_coalesced(self, frame_id=None, *, into=None) -> None:
        self._dispositions_recorded = True
        self.mapping_coalesced += 1
        self._ledger.note(frame_id, "coalesced")

    def record_mapping_skipped_capacity(self, frame_id=None) -> None:
        self._dispositions_recorded = True
        self.mapping_skipped_capacity += 1
        self._ledger.note(frame_id, "skipped_capacity")

    def record_mapping_skipped_redundant(self, frame_id=None) -> None:
        self._dispositions_recorded = True
        self.mapping_skipped_redundant += 1
        self._ledger.note(frame_id, "skipped_redundant")

    def record_geometry_insertion(self, frame_id=None, *,
                                  n_gaussians: int = 0) -> Optional[str]:
        """A source-depth insertion committed for a frame.

        Returns a conflict description when the frame had already been
        declined -- the "skipped/coalesced frame caused source-depth
        insertion" hard error, caught at the moment it happens rather than
        inferred later from aggregate counts.
        """
        self.geometry_inserted_frames += 1
        self.source_inserted_gaussians += int(n_gaussians or 0)
        return self._ledger.note_insertion(frame_id)

    # ------------------------------------------------------------------ #
    # Queue pressure and latency                                          #
    # ------------------------------------------------------------------ #

    def record_pending(self, *, depth=None, age_ms=None) -> None:
        """Sample the pending queue's depth and oldest-entry age."""
        if depth is not None:
            self._pending_depth.add(depth)
        if age_ms is not None:
            self._pending_age.add(age_ms)
            a = float(age_ms)
            if self._age_window_max is None or a > self._age_window_max:
                self._age_window_max = a
            self._age_window_n += 1
            if self._age_window_n >= self.age_window_size:
                self._age_windows.append(self._age_window_max)
                self._age_window_max = None
                self._age_window_n = 0

    def record_source_release(self, frame_id=None, *, at_step=None) -> None:
        """Mark when a source frame was released, to time what follows it."""
        if frame_id is None or at_step is None:
            return
        self._release_step[frame_id] = int(at_step)
        self._release_order.append(frame_id)
        if len(self._release_order) > _LEDGER_CAP:
            ev = self._release_order.pop(0)
            if ev not in self._release_order:
                self._release_step.pop(ev, None)

    def record_first_supervision(self, frame_id=None, *, at_step=None) -> None:
        rel = self._release_step.get(frame_id)
        if rel is not None and at_step is not None:
            self._first_supervision_latency.add(max(0, int(at_step) - rel))

    def record_insertion_latency(self, frame_id=None, *, at_step=None) -> None:
        rel = self._release_step.get(frame_id)
        if rel is not None and at_step is not None:
            self._insertion_latency.add(max(0, int(at_step) - rel))

    # ------------------------------------------------------------------ #
    # Derivation                                                          #
    # ------------------------------------------------------------------ #

    def _field_coverage(self) -> dict:
        """Per-field {complete, note} for anything not really measured.

        Follows #727's CLOCK_COVERAGE convention rather than inventing a
        parallel one.
        """
        cov: dict = {}
        if not self._admission_declared:
            for f in ("admission_target_resolved", "admission_budgeted",
                      "source_coverage_target"):
                cov[f] = {"complete": False, "note": _NO_ADMISSION_DECL}
        if not self._backpressure_declared:
            for f in ("mapping_coalesced", "mapping_skipped_capacity",
                      "mapping_skipped_redundant", "pending_max_count",
                      "pending_max_age_ms"):
                cov[f] = {"complete": False, "note": _NO_BACKPRESSURE_DECL}
        if not self._release_declared:
            for f in ("release_policy_requested", "release_policy_resolved",
                      "nominal_steps_per_frame", "effective_spf"):
                cov[f] = {"complete": False, "note": _NO_RELEASE_DECL}
        return cov

    def _dispositions(self, realized_cadence: Optional[dict] = None) -> dict:
        """Disposition counts, with the absent != zero rule applied.

        The backpressure-owned counters report a real integer only when #601's
        owner declared itself. Otherwise they are None -- a zero would assert
        "coalescing ran and never fired", which is a claim this run cannot make.

        When no per-frame disposition was recorded at all (a run predating
        #600/#601's call sites), ``mapping_candidates``/``mapping_admitted``/
        ``geometry_inserted_frames`` are DERIVED from the realized-cadence
        clocks that already count the same events run-wide, and the derivation
        is declared in ``provenance``. This keeps the acceptance question
        answerable today without inventing a second, drift-prone count of
        something ``RealizedCadenceTracker`` already counts correctly.
        """
        bp = self._backpressure_declared

        def _bp(v):
            return int(v) if bp else None

        cand = int(self.mapping_candidates)
        adm = int(self.mapping_admitted)
        ins = int(self.geometry_inserted_frames)
        ins_g = int(self.source_inserted_gaussians)
        provenance = "recorded"

        if not self._dispositions_recorded:
            clocks = (realized_cadence or {}).get("clocks") or {}
            derived = False
            if clocks.get("source_ingested") is not None:
                cand = int(clocks["source_ingested"])
                derived = True
            if clocks.get("keyframe_admitted") is not None:
                adm = int(clocks["keyframe_admitted"])
                derived = True
            if clocks.get("insertion_commit") is not None:
                ins = int(clocks["insertion_commit"])
                derived = True
            if clocks.get("inserted_gaussians") is not None:
                ins_g = int(clocks["inserted_gaussians"])
                derived = True
            provenance = "derived_from_realized_cadence" if derived else "absent"

        return {
            "mapping_candidates": cand,
            "mapping_admitted": adm,
            "mapping_metadata_only": (
                int(self.mapping_metadata_only)
                if self._dispositions_recorded else None),
            "mapping_coalesced": _bp(self.mapping_coalesced),
            "mapping_skipped_capacity": _bp(self.mapping_skipped_capacity),
            "mapping_skipped_redundant": _bp(self.mapping_skipped_redundant),
            "geometry_inserted_frames": ins,
            "source_inserted_gaussians": ins_g,
        }, provenance

    def _age_trend(self) -> dict:
        """Is the pending age growing monotonically across windows?"""
        wins = list(self._age_windows)
        if self._age_window_max is not None:
            wins = wins + [self._age_window_max]
        if len(wins) < 3:
            return {"windows": len(wins), "monotonic_increasing": None,
                    "note": "fewer than 3 closed age windows; no trend claim"}
        rising = sum(1 for a, b in zip(wins, wins[1:]) if b > a)
        return {
            "windows": len(wins),
            "first": wins[0],
            "last": wins[-1],
            "rising_steps": rising,
            "monotonic_increasing": rising == len(wins) - 1,
            "series": wins[-32:],
        }

    def summarize(self, *, realized_cadence: Optional[dict] = None,
                  mapping_clocks: Optional[dict] = None,
                  duty_cycle: Optional[dict] = None) -> dict:
        """Build the JSON-serializable summary. Pure derivation, never raises.

        ``realized_cadence`` / ``mapping_clocks`` / ``duty_cycle`` are embedded
        verbatim when supplied, so this artifact is a single stop for a
        reviewer without this module re-deriving (and potentially
        contradicting) numbers those owners already publish.
        """
        d, disposition_provenance = self._dispositions(realized_cadence)
        rc = realized_cadence or {}
        clocks = rc.get("clocks") or {}
        opt_steps = clocks.get("optimizer_step")
        released = clocks.get("source_released")

        # "Observed" = every source frame the mapper saw, whatever it then did
        # with it. That is the numerator the trajectory-coverage question asks
        # about, and it is deliberately NOT the admitted count: under soft-all
        # the whole point is that observation and admission diverge.
        #
        # COUNTING CONVENTION (#600 must honor this): record_mapping_candidate()
        # is called for EVERY observed frame, including ones that then go
        # metadata-only. metadata_only is therefore a SUBSET of candidates, not
        # a disjoint category, and is not added in here -- doing so would
        # double-count those frames. The `or 0` below only covers the
        # pre-declaration case where metadata_only is null and candidates was
        # derived from source_ingested.
        observed = d["mapping_candidates"] or (d["mapping_metadata_only"] or 0)
        if not observed and clocks.get("source_ingested") is not None:
            observed = int(clocks["source_ingested"])

        realized_updates_per_admitted = _ratio(opt_steps, d["mapping_admitted"])
        realized_updates_per_release = _ratio(opt_steps, released)

        out: dict = {
            "admission_semantics_version": ADMISSION_SEMANTICS_VERSION,
            "resolved_semantics": {
                "release_policy_requested": self.release_policy_requested,
                "release_policy_resolved": self.release_policy_resolved,
                "nominal_steps_per_frame": self.nominal_steps_per_frame,
                "effective_spf": self.effective_spf,
                "admission_policy_requested": self.admission_policy_requested,
                "admission_target_resolved": self.admission_target_resolved,
                "admission_budgeted": self.admission_budgeted,
                "source_coverage_target": self.source_coverage_target,
                "pending_max_count": self.pending_max_count,
                "pending_max_age_ms": self.pending_max_age_ms,
                "backpressure_owner": self.backpressure_owner,
                "stress_run_declared": bool(self.stress_run_declared),
                # The divergence this schema exists to expose. Deliberately
                # NOT arbitrated: both values are reported from their real
                # origin and the disagreement is named.
                "policy_divergence": self._policy_divergence(),
                "declaration_notes": list(self.admission_declaration_notes),
            },
            "dispositions": d,
            # Sibling key, deliberately NOT inside "dispositions": that dict is
            # meant to be iterated as counters by #289/#290/#563 consumers, and
            # a string value mixed in among the ints is a trap once those
            # consumers exist.
            "disposition_provenance": disposition_provenance,
            "source_observed": observed,
            "pending_queue": {
                "depth": self._pending_depth.summary(),
                "age_ms": self._pending_age.summary(),
                "age_trend": self._age_trend(),
                "configured_max_count": self.pending_max_count,
                "configured_max_age_ms": self.pending_max_age_ms,
                "emergency_transitions": list(self.emergency_transitions),
                "reservoir_cap": self._pending_depth.cap,
            },
            "latency_optimizer_steps": {
                "release_to_first_supervision":
                    self._first_supervision_latency.summary(),
                "release_to_insertion": self._insertion_latency.summary(),
            },
            "realized_exposure": {
                # The measured answer to "did the run actually deliver the
                # optimizer exposure the experiment claimed". effective_spf is
                # a CONFIGURED release cadence and is not evidence of
                # per-admitted-frame update count -- see the
                # spf_misread_as_exposure warning.
                "optimizer_updates_per_admitted_frame":
                    realized_updates_per_admitted,
                "optimizer_updates_per_source_release":
                    realized_updates_per_release,
                "configured_effective_spf": self.effective_spf,
                "measured": realized_updates_per_admitted is not None,
            },
            "consistency": {
                "conflicting_dispositions": self._ledger.conflicts,
                "insertions_after_decline": self._ledger.inserted_after_decline,
                "examples": list(self._ledger.examples),
                "ledger_cap": self._ledger.cap,
            },
            "field_coverage": self._field_coverage(),
            "notes": (
                "Validation/telemetry only (issue #602): this block records "
                "what the admission (#600) and backpressure (#601) owners did "
                "and judges self-consistency. It implements neither policy. "
                "Fields that are null carry a reason in field_coverage -- a "
                "null is 'not measured', never 'measured as zero'."
            ),
        }
        if realized_cadence is not None:
            out["realized_cadence"] = realized_cadence
        if mapping_clocks is not None:
            out["mapping_clocks"] = mapping_clocks
        if duty_cycle is not None:
            out["duty_cycle"] = duty_cycle

        out["verdict"] = self._verdict(out)
        violations = check_invariants(out)
        out["invariants"] = {
            "violations": violations,
            "n_errors": sum(1 for v in violations if v["severity"] == "error"),
            "n_warnings": sum(
                1 for v in violations if v["severity"] == "warning"),
            "violated": any(v["severity"] == "error" for v in violations),
        }
        return out

    def _policy_divergence(self) -> dict:
        req = self.admission_policy_requested
        res = self.admission_target_resolved
        if req is None or res is None:
            return {
                "diverged": None,
                "note": (
                    "cannot compare: "
                    + ("requested" if req is None else "resolved")
                    + " admission policy was not declared"),
            }
        # 'all' is the historical request spelling; #600 splits it into
        # soft_all (budgeted) and force_all_stress (unbudgeted). Requesting
        # 'all' and resolving to either is a REFINEMENT, not a divergence --
        # flagging it would make the warning fire on every default run.
        refinements = {"all": {"soft_all", "force_all_stress"}}
        if res in refinements.get(str(req), set()):
            return {"diverged": False, "requested": req, "resolved": res,
                    "note": "resolved target refines the requested policy"}
        return {
            "diverged": req != res,
            "requested": req,
            "resolved": res,
            "note": (
                "the launch surface requested one admission policy and the "
                "live path resolved to another; both are reported from their "
                "real origin rather than reconciled here"
            ) if req != res else "requested and resolved agree",
        }

    def _verdict(self, out: dict) -> dict:
        """Four top-line fields answering #602's acceptance question.

        Each is ``full``/``partial``/``unknown`` (or a named mode) rather than
        a boolean, because "we could not determine this" is a distinct and
        common answer that a boolean silently collapses into "no".
        """
        d = out["dispositions"]
        observed = out["source_observed"]
        target = self.source_coverage_target

        # 1. Did the run observe the full trajectory?
        if target is None or not observed:
            trajectory = "unknown"
            traj_note = (
                "no source_coverage_target declared, or nothing observed; "
                "cannot judge trajectory coverage from this block alone")
        else:
            frac = _ratio(observed, target) or 0.0
            trajectory = "full" if frac >= 0.999 else "partial"
            traj_note = f"observed {observed} of {target} source frames"

        # 2. Admitted all, a budgeted subset, or an explicit stress-all stream?
        res = self.admission_target_resolved
        if res is None:
            mode = "unknown"
            mode_note = _NO_ADMISSION_DECL
        elif res == "force_all_stress":
            mode = "force_all_stress"
            mode_note = (
                "explicit stress/non-production stream: admission is NOT "
                "budgeted and backpressure is bypassed by design")
        elif res == "soft_all":
            mode = "soft_all"
            mode_note = (
                "all source frames observed; mapping admission is budgeted "
                "and may decline frames under pressure")
        else:
            mode = res
            mode_note = "selective admission: a gate decided per frame"

        # 3. Did backlog accumulate?
        age_trend = out["pending_queue"]["age_trend"]
        depth = out["pending_queue"]["depth"]
        if not depth["samples"]:
            backlog = "unknown"
            backlog_note = "pending queue was never sampled"
        elif age_trend.get("monotonic_increasing"):
            backlog = "growing"
            backlog_note = (
                "pending age rose in every observed window -- the mapper did "
                "not keep up and the backlog did not plateau")
        else:
            bound = self.pending_max_count
            mx = depth.get("max")
            if bound is not None and mx is not None and mx > bound:
                backlog = "exceeded_bound"
                backlog_note = (
                    f"peak pending depth {mx} exceeded the configured bound "
                    f"{bound}")
            else:
                backlog = "bounded"
                backlog_note = (
                    f"peak pending depth {mx}; no monotonic age growth")

        # 4. Did it actually deliver the claimed optimizer exposure?
        exposure = out["realized_exposure"]
        if not exposure["measured"]:
            exp = "unknown"
            exp_note = (
                "no admitted-frame count and/or optimizer_step clock; "
                "per-frame optimizer exposure was not measured. Do NOT read "
                "effective_spf as a substitute -- it is a release cadence")
        else:
            exp = "measured"
            exp_note = (
                f"{exposure['optimizer_updates_per_admitted_frame']:.2f} "
                f"optimizer updates per admitted frame (realized, not "
                f"inferred from SPF)")

        return {
            "trajectory_observed": trajectory,
            "trajectory_observed_note": traj_note,
            "admission_mode": mode,
            "admission_mode_note": mode_note,
            "admission_budgeted": self.admission_budgeted,
            "backlog_accumulated": backlog,
            "backlog_accumulated_note": backlog_note,
            "optimizer_exposure_verified": exp,
            "optimizer_exposure_note": exp_note,
        }


# ---------------------------------------------------------------------- #
# Invariants                                                              #
# ---------------------------------------------------------------------- #

# How far realized per-admitted-frame updates may sit from the configured
# effective_spf before the "report text reads effective_spf as exposure"
# warning fires. Deliberately loose, matching realized_cadence's own reasoning:
# the failure modes worth catching are multiples, not percentages, and a tight
# threshold on a jittery quantity trains readers to ignore the warning.
SPF_EXPOSURE_TOLERANCE = 2.0

# Realized admission fraction at or above which soft-all is "admitting ~all".
SOFT_ALL_NEAR_TOTAL = 0.99


def _v(code, severity, message, **extra) -> dict:
    out = {"code": code, "severity": severity, "message": message}
    out.update(extra)
    return out


def check_invariants(summary: dict) -> list:
    """Judge a summary for self-consistency. Pure; never raises.

    Every check is **gated on its own measurement being available**. A run
    predating #600/#601 declares nothing, so the checks that depend on those
    semantics are skipped rather than firing against a null -- which is what
    makes it safe to land this module before its dependencies.
    """
    v: list = []
    sem = summary.get("resolved_semantics") or {}
    d = summary.get("dispositions") or {}
    cons = summary.get("consistency") or {}
    pending = summary.get("pending_queue") or {}
    exposure = summary.get("realized_exposure") or {}
    target = sem.get("admission_target_resolved")

    # ---------------- hard errors ---------------------------------------
    # 1. A declined frame must not have caused source-depth insertion.
    if cons.get("insertions_after_decline"):
        v.append(_v(
            "declined_frame_inserted", "error",
            f"{cons['insertions_after_decline']} source-depth insertion(s) "
            f"were attributed to frames classified skipped/coalesced. A "
            f"declined frame must not reach geometry insertion.",
            examples=cons.get("examples", [])[:5]))

    # 2. A frame must not be both admitted and skipped.
    if cons.get("conflicting_dispositions"):
        v.append(_v(
            "conflicting_disposition", "error",
            f"{cons['conflicting_dispositions']} frame(s) carried both an "
            f"admitting and a declining disposition.",
            examples=cons.get("examples", [])[:5]))

    # 3. Production soft-all must not exceed its pending bound without an
    #    explicit emergency transition. Gated on a declared bound.
    bound = pending.get("configured_max_count")
    peak = (pending.get("depth") or {}).get("max")
    if (target == "soft_all" and bound is not None and peak is not None
            and peak > bound and not pending.get("emergency_transitions")):
        v.append(_v(
            "pending_bound_exceeded", "error",
            f"production soft-all reached pending depth {peak}, above the "
            f"configured bound {bound}, with no emergency transition "
            f"recorded. Either the bound is not being enforced or the "
            f"transition was not declared.",
            peak=peak, bound=bound))
    age_bound = pending.get("configured_max_age_ms")
    age_peak = (pending.get("age_ms") or {}).get("max")
    if (target == "soft_all" and age_bound is not None and age_peak is not None
            and age_peak > age_bound
            and not pending.get("emergency_transitions")):
        v.append(_v(
            "pending_age_bound_exceeded", "error",
            f"production soft-all reached pending age {age_peak}ms, above "
            f"the configured bound {age_bound}ms, with no emergency "
            f"transition recorded.",
            peak_ms=age_peak, bound_ms=age_bound))

    # 4. force_all_stress must be marked stress/non-production in the report.
    if target == "force_all_stress":
        marked = bool(sem.get("stress_run_declared"))
        budgeted = sem.get("admission_budgeted")
        if not marked:
            v.append(_v(
                "stress_run_not_marked", "error",
                "admission_target_resolved is 'force_all_stress' but the "
                "report does not mark this run as stress/non-production. Its "
                "numbers must not be read as a production result."))
        if budgeted:
            v.append(_v(
                "stress_run_marked_budgeted", "error",
                "admission_target_resolved is 'force_all_stress' but "
                "admission_budgeted is true; force-all is by definition "
                "unbudgeted."))

    # ---------------- loud warnings --------------------------------------
    # 5. Realized source cadence diverges from the configured iter-based one.
    rc = summary.get("realized_cadence") or {}
    rec = rc.get("reconciliation") or rc.get("reconcile") or {}
    verdict = rec.get("verdict")
    if verdict and verdict not in ("ok", "insufficient_evidence"):
        v.append(_v(
            "release_cadence_divergence", "warning",
            f"realized source-release cadence diverges from the configured "
            f"iter-based cadence (realized_cadence verdict: {verdict!r}).",
            verdict=verdict))

    # 6. Soft-all admitting ~everything while backlog/age stay above pressure
    #    thresholds: the budget is nominally on but is not actually binding.
    if target == "soft_all":
        observed = summary.get("source_observed") or 0
        admitted = d.get("mapping_admitted")
        frac = _ratio(admitted, observed)
        depth_p95 = (pending.get("depth") or {}).get("p95")
        age_p95 = (pending.get("age_ms") or {}).get("p95")
        over_depth = (bound is not None and depth_p95 is not None
                      and depth_p95 > bound)
        over_age = (age_bound is not None and age_p95 is not None
                    and age_p95 > age_bound)
        if frac is not None and frac >= SOFT_ALL_NEAR_TOTAL and (
                over_depth or over_age):
            v.append(_v(
                "soft_all_backpressure_ineffective", "warning",
                f"admission target is soft-all and realized admission is "
                f"{frac:.1%} of observed frames, yet pending pressure stayed "
                f"persistently above the configured threshold (depth p95="
                f"{depth_p95}, age p95={age_p95}ms). Backpressure is "
                f"configured but does not appear to bind.",
                realized_admission_fraction=frac))

    # 7. effective_spf must not be read as per-admitted-frame optimizer
    #    exposure without measured evidence.
    spf = sem.get("effective_spf")
    realized = exposure.get("optimizer_updates_per_admitted_frame")
    if spf is not None and realized is None:
        v.append(_v(
            "spf_exposure_unmeasured", "warning",
            f"effective_spf={spf} is configured but per-admitted-frame "
            f"optimizer exposure was NOT measured for this run. "
            f"effective_spf is a release cadence; do not report it as N "
            f"optimizer updates per admitted frame."))
    elif spf is not None and realized is not None:
        try:
            r = _ratio(realized, float(spf))
        except (TypeError, ValueError):
            r = None
        if r is not None and (r > SPF_EXPOSURE_TOLERANCE
                              or r < 1.0 / SPF_EXPOSURE_TOLERANCE):
            v.append(_v(
                "spf_exposure_divergence", "warning",
                f"configured effective_spf={spf} but realized optimizer "
                f"updates per admitted frame is {realized:.2f} "
                f"({r:.2f}x). Reporting effective_spf as the run's "
                f"per-frame optimizer exposure would misstate it.",
                configured=spf, realized=realized, ratio=r))

    # 8. Pending age increasing monotonically over the observation window.
    trend = pending.get("age_trend") or {}
    if trend.get("monotonic_increasing"):
        v.append(_v(
            "pending_age_monotonic_growth", "warning",
            f"pending queue age rose in every one of the "
            f"{trend.get('windows')} observed windows "
            f"({trend.get('first')}ms -> {trend.get('last')}ms). The backlog "
            f"is not plateauing; the mapper is not keeping up.",
            first_ms=trend.get("first"), last_ms=trend.get("last")))

    return v


def raise_on_errors(summary: dict) -> None:
    """Strict mode: turn hard-error violations into an exception.

    Deliberately NOT the default. ``summarize()`` runs at report time, after
    hours of GPU work; raising there would destroy the entire report to
    complain about telemetry. Production records violations and prints; tests
    call this.
    """
    errs = [v for v in (summary.get("invariants") or {}).get("violations", [])
            if v["severity"] == "error"]
    if errs:
        raise AdmissionInvariantError(
            "admission/cadence invariant violation(s): "
            + "; ".join(e["message"] for e in errs))


class AdmissionInvariantError(RuntimeError):
    """Raised only by :func:`raise_on_errors` (strict mode)."""


def format_violations(violations: Iterable[dict]) -> list:
    """Human-readable lines for stdout at report time."""
    out = []
    for v in violations:
        tag = "ERROR" if v["severity"] == "error" else "WARNING"
        out.append(f"[admission_cadence] {tag}: {v['message']}")
    return out


def admission_cadence_report_block(
    recorder: Optional[AdmissionCadenceRecorder],
    *,
    realized_cadence: Optional[dict] = None,
    mapping_clocks: Optional[dict] = None,
    duty_cycle: Optional[dict] = None,
) -> Optional[dict]:
    """Build the report block, or None when the run carried no recorder.

    Returning None (rather than an empty block) keeps the absent != zero rule
    at the top level too: a report with no ``admission_cadence`` key predates
    this instrumentation and is not a run that measured nothing.
    """
    if recorder is None:
        return None
    try:
        return recorder.summarize(
            realized_cadence=realized_cadence,
            mapping_clocks=mapping_clocks,
            duty_cycle=duty_cycle,
        )
    except Exception as exc:  # pragma: no cover - defensive at report time
        return {
            "admission_semantics_version": ADMISSION_SEMANTICS_VERSION,
            "error": f"summary derivation failed: {exc!r}",
        }
