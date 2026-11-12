"""Issue #326 -- derive the conflict metrics from a recorded operation trace.

Torch-free by construction: it consumes :class:`GaussianOpEvent` rows (or the
JSONL a run wrote) and produces #326's derived metrics plus the **ranked
conflict table** the issue's acceptance criterion asks for.

#326's bar: *"A pair is 'confirmed fighting' only when both mechanisms touch the
same state/UID in an opposing or overriding way at runtime."* That is encoded
literally in :func:`ranked_conflict_table` -- a pair is only emitted when two
mechanisms produced APPLIED deltas to the SAME (uid, state key) inside the same
iteration, and the second either reversed the first (negative projection, for
vector state) or overwrote it. Co-occurrence alone never qualifies.

EVIDENCE ASYMMETRY (issue #203). A short run's table is legitimate POSITIVE
evidence that a pair fights. It is NOT evidence that an unlisted pair does not
-- absence here is absence of observation.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import Any, Iterable, Sequence

from eligsir.diagnostics.gaussian_op_trace import GaussianOpEvent

#: Operations that change the row count of the population.
POPULATION_CHANGING = frozenset({"grow", "retire", "create_bootstrap"})

#: Vector state keys whose deltas can be projected against each other.
VECTOR_KEYS = ("xyz",)
#: Scalar state keys where "opposing" means opposite sign.
SCALAR_KEYS = ("opacity", "scale", "rotation")


def _norm(v: Sequence[float]) -> float:
    return math.sqrt(sum(float(x) * float(x) for x in v))


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(float(x) * float(y) for x, y in zip(a, b))


# --------------------------------------------------------------------------
# Windowing
# --------------------------------------------------------------------------

def _applied_uid_deltas(events: Iterable[GaussianOpEvent]):
    """Yield ``(iteration, uid, key, value, event)`` for every applied uid delta."""
    for ev in events:
        if ev.scope != "uid" or ev.uid is None:
            continue
        if ev.phase not in ("applied", "cancelled", "bracket"):
            continue
        for key, value in (ev.applied_delta or {}).items():
            yield ev.iteration, int(ev.uid), key, value, ev


# --------------------------------------------------------------------------
# Metric 1 -- movement cancellation ratio
# --------------------------------------------------------------------------

def movement_cancellation_ratio(events: Sequence[GaussianOpEvent]) -> dict[str, Any]:
    """How much of the xyz motion applied in a step is undone within that step.

    ``ratio = sum(cancelled magnitude) / sum(applied magnitude)`` over watched
    UIDs, where "cancelled" is the component of an earlier mechanism's delta
    that a later mechanism in the SAME iteration reversed.

    Per-pair breakdown is included because the aggregate ratio alone cannot say
    *who* cancels *whom* -- which is the whole point of #326.
    """
    by_cell: dict[tuple[int, int], list[tuple[str, list[float]]]] = defaultdict(list)
    for it, uid, key, value, ev in _applied_uid_deltas(events):
        if key != "xyz":
            continue
        by_cell[(it, uid)].append((ev.mechanism, list(value)))

    total_applied = 0.0
    total_cancelled = 0.0
    pair_applied: dict[tuple[str, str], float] = defaultdict(float)
    pair_cancelled: dict[tuple[str, str], float] = defaultdict(float)
    for (_it, _uid), seq in by_cell.items():
        for i, (mech_a, delta_a) in enumerate(seq):
            mag_a = _norm(delta_a)
            total_applied += mag_a
            if mag_a == 0.0:
                continue
            for mech_b, delta_b in seq[i + 1:]:
                proj = -_dot(delta_a, delta_b) / mag_a  # >0 => b reverses a
                pair_applied[(mech_a, mech_b)] += mag_a
                if proj > 0.0:
                    cancelled = min(proj, mag_a)
                    total_cancelled += cancelled
                    pair_cancelled[(mech_a, mech_b)] += cancelled

    pairs = [
        {
            "mover": a, "canceller": b,
            "applied_magnitude": pair_applied[(a, b)],
            "cancelled_magnitude": pair_cancelled[(a, b)],
            "ratio": (pair_cancelled[(a, b)] / pair_applied[(a, b)])
            if pair_applied[(a, b)] else 0.0,
        }
        for (a, b) in sorted(pair_applied)
    ]
    pairs.sort(key=lambda p: -p["cancelled_magnitude"])
    return {
        "total_applied_magnitude": total_applied,
        "total_cancelled_magnitude": total_cancelled,
        "ratio": (total_cancelled / total_applied) if total_applied else 0.0,
        "n_uid_iteration_cells": len(by_cell),
        "by_pair": pairs,
    }


# --------------------------------------------------------------------------
# Metric 2 -- same-UID opposing action rate
# --------------------------------------------------------------------------

def same_uid_opposing_action_rate(events: Sequence[GaussianOpEvent]) -> dict[str, Any]:
    """Fraction of (uid, iteration, state-key) cells where two mechanisms pushed
    in opposing directions (vector: negative dot; scalar: opposite sign)."""
    cells: dict[tuple[int, int, str], list[tuple[str, Any]]] = defaultdict(list)
    for it, uid, key, value, ev in _applied_uid_deltas(events):
        cells[(it, uid, key)].append((ev.mechanism, value))

    multi = 0
    opposing = 0
    pair_counts: dict[tuple[str, str, str], int] = defaultdict(int)
    for (_it, _uid, key), seq in cells.items():
        mechs = {m for m, _ in seq}
        if len(mechs) < 2:
            continue
        multi += 1
        found = False
        for i, (ma, va) in enumerate(seq):
            for mb, vb in seq[i + 1:]:
                if ma == mb:
                    continue
                if key in VECTOR_KEYS:
                    opposed = _dot(va, vb) < 0.0
                else:
                    opposed = (float(va) * float(vb)) < 0.0
                if opposed:
                    found = True
                    pair_counts[(min(ma, mb), max(ma, mb), key)] += 1
        opposing += int(found)
    table = [
        {"mechanism_a": a, "mechanism_b": b, "state": k, "n_opposing_cells": n}
        for (a, b, k), n in sorted(pair_counts.items(), key=lambda kv: -kv[1])
    ]
    return {
        "n_multi_mechanism_cells": multi,
        "n_opposing_cells": opposing,
        "rate": (opposing / multi) if multi else 0.0,
        "by_pair": table,
    }


# --------------------------------------------------------------------------
# Metric 3 -- grow-then-prune / relocate-then-restore within a window
# --------------------------------------------------------------------------

def grow_then_prune_rate(
    events: Sequence[GaussianOpEvent], window_iters: int = 500
) -> dict[str, Any]:
    """UIDs created by one mechanism and removed by another within ``window_iters``."""
    births: dict[int, tuple[int, str]] = {}
    deaths: list[tuple[int, int, str]] = []
    for ev in events:
        if ev.scope != "uid" or ev.uid is None or ev.phase != "applied":
            continue
        if ev.op in ("grow", "create_bootstrap") and int(ev.uid) not in births:
            births[int(ev.uid)] = (ev.iteration, ev.mechanism)
        elif ev.op == "retire":
            deaths.append((int(ev.uid), ev.iteration, ev.mechanism))

    hits: list[dict[str, Any]] = []
    for uid, death_it, killer in deaths:
        born = births.get(uid)
        if born is None:
            continue
        birth_it, creator = born
        if 0 <= death_it - birth_it <= window_iters:
            hits.append({
                "uid": uid, "born_iter": birth_it, "died_iter": death_it,
                "lifetime_iters": death_it - birth_it,
                "creator": creator, "killer": killer,
                "cross_mechanism": creator != killer,
            })
    pair: dict[tuple[str, str], int] = defaultdict(int)
    for h in hits:
        pair[(h["creator"], h["killer"])] += 1
    return {
        "window_iters": window_iters,
        "n_births_observed": len(births),
        "n_short_lived": len(hits),
        "rate": (len(hits) / len(births)) if births else 0.0,
        "n_cross_mechanism": sum(1 for h in hits if h["cross_mechanism"]),
        "by_pair": [
            {"creator": a, "killer": b, "n": n}
            for (a, b), n in sorted(pair.items(), key=lambda kv: -kv[1])
        ],
        "examples": hits[:50],
    }


def relocate_then_restore_rate(
    events: Sequence[GaussianOpEvent], window_iters: int = 1
) -> dict[str, Any]:
    """UIDs relocated/moved by one mechanism then restored by another, within
    ``window_iters`` iterations. A restore is any applied xyz delta whose
    projection reverses >= 50% of the prior move."""
    moves: dict[int, list[tuple[int, str, list[float]]]] = defaultdict(list)
    for it, uid, key, value, ev in _applied_uid_deltas(events):
        if key != "xyz":
            continue
        moves[uid].append((it, ev.mechanism, list(value)))

    n_moves = 0
    n_restored = 0
    pair: dict[tuple[str, str], int] = defaultdict(int)
    for uid, seq in moves.items():
        seq.sort(key=lambda t: t[0])
        for i, (it_a, mech_a, da) in enumerate(seq):
            mag = _norm(da)
            if mag == 0.0:
                continue
            n_moves += 1
            for it_b, mech_b, db in seq[i + 1:]:
                if it_b - it_a > window_iters:
                    break
                if mech_b == mech_a:
                    continue
                if -_dot(da, db) / mag >= 0.5 * mag:
                    n_restored += 1
                    pair[(mech_a, mech_b)] += 1
                    break
    return {
        "window_iters": window_iters,
        "n_moves": n_moves,
        "n_restored": n_restored,
        "rate": (n_restored / n_moves) if n_moves else 0.0,
        "by_pair": [
            {"mover": a, "restorer": b, "n": n}
            for (a, b), n in sorted(pair.items(), key=lambda kv: -kv[1])
        ],
    }


# --------------------------------------------------------------------------
# Metric 4 -- death-candidate overlap across mechanisms
# --------------------------------------------------------------------------

def death_candidate_overlap(events: Sequence[GaussianOpEvent]) -> dict[str, Any]:
    """Jaccard overlap of retire-CANDIDATE uid sets nominated in the same iteration.

    This is why the tracer has a ``candidate`` phase at all: an *applied* retire
    only ever has one owner (whoever got there first), so overlap between
    MCMC/lifecycle/safety death sets is invisible unless nominations are
    recorded separately from executions.
    """
    per_iter: dict[int, dict[str, set[int]]] = defaultdict(lambda: defaultdict(set))
    for ev in events:
        if ev.phase != "candidate" or ev.op not in ("retire", "relocate"):
            continue
        if ev.candidate_uids:
            per_iter[ev.iteration][ev.mechanism].update(int(u) for u in ev.candidate_uids)
        elif ev.uid is not None:
            per_iter[ev.iteration][ev.mechanism].add(int(ev.uid))

    pair_inter: dict[tuple[str, str], int] = defaultdict(int)
    pair_union: dict[tuple[str, str], int] = defaultdict(int)
    pair_iters: dict[tuple[str, str], int] = defaultdict(int)
    for _it, by_mech in per_iter.items():
        mechs = sorted(by_mech)
        for i, a in enumerate(mechs):
            for b in mechs[i + 1:]:
                inter = len(by_mech[a] & by_mech[b])
                union = len(by_mech[a] | by_mech[b])
                pair_inter[(a, b)] += inter
                pair_union[(a, b)] += union
                pair_iters[(a, b)] += 1
    table = [
        {
            "mechanism_a": a, "mechanism_b": b,
            "n_iterations_both_nominated": pair_iters[(a, b)],
            "n_shared_uids": pair_inter[(a, b)],
            "jaccard": (pair_inter[(a, b)] / pair_union[(a, b)]) if pair_union[(a, b)] else 0.0,
        }
        for (a, b) in sorted(pair_union)
    ]
    table.sort(key=lambda r: -r["n_shared_uids"])
    return {"n_iterations_with_candidates": len(per_iter), "by_pair": table}


# --------------------------------------------------------------------------
# Metric 5 -- topology mutation while the scheduler gate is off
# --------------------------------------------------------------------------

def topology_mutation_while_gate_off(events: Sequence[GaussianOpEvent]) -> dict[str, Any]:
    by_mech: dict[tuple[str, str], int] = defaultdict(int)
    total = 0
    gated = 0
    examples: list[dict[str, Any]] = []
    for ev in events:
        if ev.op not in POPULATION_CHANGING or ev.phase != "applied":
            continue
        total += 1
        gate = ev.gate or {}
        key = "allow_growth" if ev.op in ("grow", "create_bootstrap") else "allow_prune"
        allowed = gate.get(key)
        if allowed is None and ev.op == "grow":
            allowed = gate.get("allow_insertion")
        if allowed is False:
            gated += 1
            by_mech[(ev.mechanism, ev.op)] += 1
            if len(examples) < 50:
                examples.append({
                    "iteration": ev.iteration, "mechanism": ev.mechanism,
                    "op": ev.op, "n_rows": ev.n_rows, "gate": gate,
                })
    return {
        "n_population_changing_events": total,
        "n_while_gate_off": gated,
        "rate": (gated / total) if total else 0.0,
        "by_mechanism": [
            {"mechanism": m, "op": o, "n": n}
            for (m, o), n in sorted(by_mech.items(), key=lambda kv: -kv[1])
        ],
        "examples": examples,
    }


# --------------------------------------------------------------------------
# Metric 6 -- population change with NO topology_version advance
#             (runtime check of #543's secondary finding #1)
# --------------------------------------------------------------------------

def population_change_without_topology_advance(
    events: Sequence[GaussianOpEvent],
) -> dict[str, Any]:
    """#543 found *statically* that no MCMC growth/relocation path advances
    ``map_backend.topology_version``, so ``train_streaming.py``'s
    ``if map_backend.topology_version != _w3_02_mcmc_topo_before`` guard can
    never fire for an ordinary MCMC mutation -- and flagged it "unverified at
    runtime; candidate for #326". This turns it into a runtime assertion.

    An aggregate event that reports ``n_rows > 0`` while
    ``evidence['topology_version_before'] == evidence['topology_version_after']``
    is a population mutation invisible to every topology_version consumer.
    """
    silent: list[dict[str, Any]] = []
    by_mech: dict[str, int] = defaultdict(int)
    total = 0
    for ev in events:
        if ev.op not in POPULATION_CHANGING or ev.phase != "applied":
            continue
        if not ev.n_rows:
            continue
        ev_before = (ev.evidence or {}).get("topology_version_before")
        ev_after = (ev.evidence or {}).get("topology_version_after")
        if ev_before is None or ev_after is None:
            continue
        total += 1
        if int(ev_before) == int(ev_after):
            by_mech[ev.mechanism] += 1
            if len(silent) < 50:
                silent.append({
                    "iteration": ev.iteration, "mechanism": ev.mechanism,
                    "op": ev.op, "n_rows": ev.n_rows,
                    "topology_version": int(ev_before),
                })
    return {
        "n_population_events_with_topology_evidence": total,
        "n_silent_population_changes": sum(by_mech.values()),
        "rate": (sum(by_mech.values()) / total) if total else 0.0,
        "by_mechanism": [
            {"mechanism": m, "n": n}
            for m, n in sorted(by_mech.items(), key=lambda kv: -kv[1])
        ],
        "examples": silent,
    }


# --------------------------------------------------------------------------
# Metric 7 -- mutations following support starvation
# --------------------------------------------------------------------------

def mutations_after_support_starvation(
    events: Sequence[GaussianOpEvent], support_threshold: float = 0.0
) -> dict[str, Any]:
    """Applied mutations on UIDs whose recorded support evidence was at/below
    ``support_threshold`` at the time of the mutation."""
    total = 0
    starved = 0
    by_mech: dict[tuple[str, str], int] = defaultdict(int)
    for ev in events:
        if ev.scope != "uid" or ev.phase != "applied":
            continue
        support = (ev.evidence or {}).get("support")
        if support is None:
            continue
        total += 1
        if float(support) <= support_threshold:
            starved += 1
            by_mech[(ev.mechanism, ev.op)] += 1
    return {
        "support_threshold": support_threshold,
        "n_mutations_with_support_evidence": total,
        "n_on_starved": starved,
        "rate": (starved / total) if total else 0.0,
        "by_mechanism": [
            {"mechanism": m, "op": o, "n": n}
            for (m, o), n in sorted(by_mech.items(), key=lambda kv: -kv[1])
        ],
    }


# --------------------------------------------------------------------------
# Metric 8 -- growth near shape-constrained (bound-hit) Gaussians (issue #329,
# class 2: "shape constraints -> compensatory growth")
# --------------------------------------------------------------------------

def growth_near_shape_bound_hit_rate(
    events: Sequence[GaussianOpEvent],
) -> dict[str, Any]:
    """Applied ``grow``/``create_bootstrap`` events whose recorded evidence
    reports a nearby scale/anisotropy bound-hit (``evidence["shape_bound_hit"]``
    truthy, e.g. from a scale-prior or anisotropy-soft-limit clamp).

    This is #326 Sec.4's explicitly-unimplemented metric
    ("growth near Gaussians hitting shape constraints ... needs a per-Gaussian
    clamp-hit signal the model does not currently expose") and directly serves
    issue #329's class 2 hypothesis: a scale/isotropy/surface-thickness clamp
    that prevents an existing Gaussian from reshaping enough can leave a
    persistent photometric/geometric residual that MCMC then "solves" by
    growing a *new* Gaussian nearby instead.

    Mirrors :func:`mutations_after_support_starvation`'s honesty contract: no
    production hook populates ``evidence["shape_bound_hit"]`` yet (see the
    repo's #329 synthesis doc), so on today's traces this reports
    ``n_growth_events_with_shape_evidence: 0`` rather than a fabricated rate.
    A future hook that threads a per-Gaussian bound-hit flag (from the same
    ``scale_prior_max``/``anisotropy_soft_limit`` thresholds
    ``utils/differentiable_energy_loss.py`` already uses) into the `grow`
    event's evidence makes this metric real without any change here.
    """
    total = 0
    near_bound = 0
    by_mech: dict[str, int] = defaultdict(int)
    for ev in events:
        if ev.phase != "applied" or ev.op not in ("grow", "create_bootstrap"):
            continue
        hit = (ev.evidence or {}).get("shape_bound_hit")
        if hit is None:
            continue
        total += 1
        if bool(hit):
            near_bound += 1
            by_mech[ev.mechanism] += 1
    return {
        "n_growth_events_with_shape_evidence": total,
        "n_near_shape_bound": near_bound,
        "rate": (near_bound / total) if total else 0.0,
        "by_mechanism": [
            {"mechanism": m, "n": n}
            for m, n in sorted(by_mech.items(), key=lambda kv: -kv[1])
        ],
    }


# --------------------------------------------------------------------------
# modified-again annotation (#326 schema field)
# --------------------------------------------------------------------------

def annotate_modified_again(
    events: Sequence[GaussianOpEvent],
) -> list[GaussianOpEvent]:
    """Fill each uid event's ``modified_again_by`` with the next DIFFERENT
    mechanism to apply a delta to the same (uid, state key) in the same
    iteration. Returns new events; inputs are frozen and untouched."""
    order: dict[tuple[int, int, str], list[int]] = defaultdict(list)
    idx_of: dict[int, int] = {}
    for i, ev in enumerate(events):
        idx_of[ev.seq] = i
        if ev.scope != "uid" or ev.uid is None:
            continue
        for key in (ev.applied_delta or {}):
            order[(ev.iteration, int(ev.uid), key)].append(i)

    later: dict[int, str] = {}
    for _cell, idxs in order.items():
        idxs.sort()
        for a, i in enumerate(idxs):
            for j in idxs[a + 1:]:
                if events[j].mechanism != events[i].mechanism:
                    later[i] = events[j].mechanism
                    break
    out = list(events)
    for i, mech in later.items():
        ev = out[i]
        out[i] = GaussianOpEvent(**{**asdict(ev), "modified_again_by": mech})
    return out


# --------------------------------------------------------------------------
# Ranked conflict table (the #326 acceptance artifact)
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class ConflictRow:
    mechanism_a: str
    mechanism_b: str
    state: str
    n_events: int
    magnitude: float
    kind: str
    confirmed_fighting: bool
    note: str = ""


def ranked_conflict_table(events: Sequence[GaussianOpEvent]) -> list[dict[str, Any]]:
    """#326 acceptance: a table ranked by frequency and magnitude, where a pair
    is ``confirmed_fighting`` ONLY when both mechanisms touched the same
    state/UID in an opposing or overriding way at runtime."""
    rows: list[ConflictRow] = []

    cancel = movement_cancellation_ratio(events)
    for p in cancel["by_pair"]:
        if p["cancelled_magnitude"] <= 0.0:
            continue
        rows.append(ConflictRow(
            mechanism_a=p["mover"], mechanism_b=p["canceller"], state="xyz",
            n_events=0, magnitude=p["cancelled_magnitude"],
            kind="movement_cancellation", confirmed_fighting=True,
            note=f"{p['ratio']:.1%} of {p['mover']}'s applied motion reversed by {p['canceller']}",
        ))

    opposing = same_uid_opposing_action_rate(events)
    for p in opposing["by_pair"]:
        rows.append(ConflictRow(
            mechanism_a=p["mechanism_a"], mechanism_b=p["mechanism_b"],
            state=p["state"], n_events=p["n_opposing_cells"], magnitude=0.0,
            kind="opposing_same_uid", confirmed_fighting=True,
            note="opposing-sign applied deltas on the same uid+state in one iteration",
        ))

    gtp = grow_then_prune_rate(events)
    for p in gtp["by_pair"]:
        if p["creator"] == p["killer"]:
            continue
        rows.append(ConflictRow(
            mechanism_a=p["creator"], mechanism_b=p["killer"], state="population",
            n_events=p["n"], magnitude=float(p["n"]),
            kind="grow_then_prune", confirmed_fighting=True,
            note=f"{p['n']} uids created by {p['creator']} removed by {p['killer']} within window",
        ))

    overlap = death_candidate_overlap(events)
    for p in overlap["by_pair"]:
        if p["n_shared_uids"] <= 0:
            continue
        rows.append(ConflictRow(
            mechanism_a=p["mechanism_a"], mechanism_b=p["mechanism_b"], state="death_candidates",
            n_events=p["n_shared_uids"], magnitude=p["jaccard"],
            kind="death_candidate_overlap", confirmed_fighting=False,
            note="both nominated the same uids for removal; NOT itself an applied "
                 "conflict -- overlap alone means redundant authority, not a fight",
        ))

    rows.sort(key=lambda r: (-r.n_events, -r.magnitude))
    return [asdict(r) for r in rows]


def derive_all(events: Sequence[GaussianOpEvent], window_iters: int = 500) -> dict[str, Any]:
    """Every #326 derived metric, plus the ranked table, in one dict."""
    annotated = annotate_modified_again(events)
    n_uid = sum(1 for e in events if e.scope == "uid")
    return {
        "issue": 326,
        "n_events": len(events),
        "n_uid_scoped_events": n_uid,
        "n_aggregate_events": len(events) - n_uid,
        "iterations_covered": sorted({e.iteration for e in events})[:0] or (
            [min(e.iteration for e in events), max(e.iteration for e in events)]
            if events else []
        ),
        "mechanisms_seen": sorted({e.mechanism for e in events}),
        "operations_seen": sorted({e.op for e in events}),
        "movement_cancellation": movement_cancellation_ratio(events),
        "same_uid_opposing_actions": same_uid_opposing_action_rate(events),
        "grow_then_prune": grow_then_prune_rate(events, window_iters=window_iters),
        "relocate_then_restore": relocate_then_restore_rate(events),
        "death_candidate_overlap": death_candidate_overlap(events),
        "topology_mutation_while_gate_off": topology_mutation_while_gate_off(events),
        "population_change_without_topology_advance":
            population_change_without_topology_advance(events),
        "mutations_after_support_starvation": mutations_after_support_starvation(events),
        "growth_near_shape_bound_hit": growth_near_shape_bound_hit_rate(events),
        "n_events_modified_again_in_window":
            sum(1 for e in annotated if e.modified_again_by is not None),
        "ranked_conflict_table": ranked_conflict_table(events),
        "evidence_class_caveat":
            "POSITIVE evidence only. A pair absent from ranked_conflict_table was "
            "not observed to fight in THIS trace; per issue #203 that is not a "
            "negative result unless the run's experiment_validity_class permits one.",
    }
