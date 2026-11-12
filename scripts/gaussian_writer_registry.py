#!/usr/bin/env python3
"""Policy-aware Gaussian writer/owner registry.

The original #325 registry used a 1,000+ line hand-maintained annotation table.
That table was useful while ownership was being discovered, but became the
largest source of false evidence after the #331/#1130 hard cut: the same raw
MCMC executor sites were still reported as independent policy owners even
though production dispatch had moved above them into ``GaussianMutationArbiter``.

This generator keeps the mechanical AST discovery from
``gaussian_writer_discovery.py`` and derives ownership from the *current
production boundary* instead:

* ``eligsir.map`` adapters/transactions/arbiter are mechanism owners;
* production MCMC strategy dispatch is proposal/executor code underneath the
  arbiter, not an independent topology-commit authority;
* production source-depth insertion is transaction/arbiter mediated;
* retired live-strategy mutation phases are source-visible but non-production;
* coordinate corrections are distinguished from generic xyz writes and the
  atomic anchor-update primitives are recognized explicitly.

The result remains deliberately conservative about raw tensor writes: an
arbiter-owned delegated executor can still be a check-(b) raw-access finding
without becoming a second check-(a)/(c) topology policy owner.

The public API and JSON/Markdown schema are retained for the #325 guard tests
and the #326 runtime vocabulary join.  Stdlib only; no torch/numpy imports.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from gaussian_writer_discovery import (  # noqa: E402
    KNOWN_BLIND_SPOTS,
    REPO_ROOT,
    WriteSite,
    discover,
    group_units,
)

# Keep actual row values joinable to eligsir.diagnostics.gaussian_op_trace.
SEMANTIC_OPERATIONS = frozenset(
    {
        "grow",
        "retire",
        "relocate",
        "move_xyz",
        "opacity",
        "scale",
        "rotation",
        "support_update",
        "lifecycle_transition",
        "anchor_reference_update",
        "coordinate_correction",
        "identity_provenance_update",
        "create_bootstrap",
        "optimizer_state_update",
        "parameter_storage",
        "buffer_alloc",
        "checkpoint_restore",
        "strategy_dispatch",
        "not_a_gaussian_writer",
    }
)

OWNERS = frozenset(
    {
        "mcmc",
        "source_depth",
        "lifecycle",
        "anchors_depth_locks",
        "freespace_contradiction",
        "mutation_transaction",
        "map_frame_correction",
        "loop_closure_merge",
        "optimizer",
        "model_class",
        "checkpoint_io",
        "none",
    }
)

MECHANISM_OWNERS = frozenset(
    {"mutation_transaction", "model_class", "checkpoint_io", "none"}
)

SANCTIONED_RAW_LAYERS = (
    "scene/gaussian_model.py",
    "scene/gsplat_model.py",
    "eligsir/map/mutation.py",
    "eligsir/map/adapters/",
)

# Kept as an empty compatibility symbol for callers that imported ANNOTATIONS.
# There is intentionally no longer a second hand-maintained truth source.
ANNOTATIONS: dict = {}

@dataclass
class RegistryRow:
    path: str
    qualname: str
    state_written: list[str]
    semantic_operation: str
    declared_owner: str
    trigger: str
    gate: str
    evidence_consumed: str
    hard_safety: bool
    production: bool
    is_mechanism: bool
    sanctioned_raw_access: bool
    coordinate_correction: bool
    updates_anchor_reference: bool
    linked_to_transaction: bool
    notes: str
    lines: list[int]
    kinds: list[str]
    examples: list[str]

    @property
    def key(self) -> str:
        return f"{self.path}::{self.qualname}"


@dataclass
class RegistryResult:
    rows: list[RegistryRow]
    unclassified: list[str]
    stale_annotations: list[str]
    vocabulary_errors: list[str]
    n_sites: int
    n_units: int


# ---------------------------------------------------------------------------
# Policy-aware classification
# ---------------------------------------------------------------------------


def _is_nonproduction_path(path: str, qualname: str) -> bool:
    if path.startswith("scripts/") or path.startswith("viz/"):
        return True
    # The live-strategy mutating phases are rejected by production MCMC dispatch
    # after #331. Keep them visible as source writers, but do not let unreachable
    # code create a false competing production owner.
    if path == "eligsir/optimization/gaussian/strategies/live/backend.py" and (
        "apply_retire" in qualname or "apply_reseed" in qualname
    ):
        return True
    # This legacy second correction implementation has no production caller with
    # a Gaussian model; real callers explicitly pass gaussians=None.
    if path == "utils/submap_manager.py" and "apply_pose_graph_corrections" in qualname:
        return True
    return False


def _is_coordinate_correction(path: str, qualname: str) -> bool:
    q = qualname.lower()
    return (
        "pose_correction" in q
        or "pose_graph_correction" in q
        or "submap_correction" in q
        or "gaussian_merge_correction" in q
        or (path.startswith("loop_closure/") and "correction" in q)
    )


def _updates_anchor_reference(path: str, qualname: str) -> bool:
    if path == "eligsir/map/mutation.py" and "apply_pose_correction" in qualname:
        return True
    if path == "eligsir/map/adapters/legacy_graphdeco.py" and "apply_pose_correction" in qualname:
        return True
    if path == "loop_closure/submap_correction.py" and (
        "apply_submap_correction" in qualname or "apply_gaussian_merge_correction" in qualname
    ):
        return True
    return False


def _operation(site: WriteSite) -> str:
    path = site.path
    q = site.qualname.lower()
    g = site.state_group
    t = site.target.lower()

    if _is_coordinate_correction(path, site.qualname) and g in {
        "means_xyz", "population", "topology_transaction", "anchor_reference"
    }:
        return "coordinate_correction"
    if (
        "restore" in q
        or "load_ply" in q
        or (
            path == "eligsir/checkpointing/exact_refinement_resume.py"
            and q == "exactcausalforkscenestate.load_resume_state_dict"
        )
    ):
        return "checkpoint_restore"
    if g == "population":
        if any(x in q or x in t for x in ("relocate", "recycle")):
            return "relocate"
        if any(x in q or x in t for x in ("prune", "retire", "remove")):
            return "retire"
        if any(x in q for x in ("initialize", "create_from_pcd", "bootstrap")):
            return "create_bootstrap"
        return "grow"
    if g == "means_xyz":
        return "move_xyz"
    if g == "scales":
        return "scale"
    if g == "rotation":
        return "rotation"
    if g == "opacity":
        return "opacity"
    if g == "lifecycle_state":
        return "lifecycle_transition"
    if g == "support_state":
        return "support_update"
    if g == "anchor_reference":
        return "anchor_reference_update"
    if g == "identity_provenance":
        return "identity_provenance_update"
    if g == "optimizer_state":
        return "optimizer_state_update"
    if g == "strategy_dispatch" or g == "topology_transaction":
        return "strategy_dispatch"
    if g in {"params_bulk", "sh_features"}:
        return "parameter_storage"
    return "not_a_gaussian_writer"


def _owner(site: WriteSite, operation: str) -> tuple[str, bool, str]:
    """Return (declared_owner, linked_to_transaction, explanatory note)."""
    p = site.path
    q = site.qualname

    if (
        p == "eligsir/checkpointing/exact_refinement_resume.py"
        and q == "ExactCausalForkSceneState.load_resume_state_dict"
    ):
        return "checkpoint_io", True, "Exact causal-fork checkpoint restore."

    if p.startswith("eligsir/map/"):
        return (
            "mutation_transaction",
            True,
            "Canonical map adapter/transaction/arbiter mechanism; sole commit authority.",
        )
    if p in {"scene/gaussian_model.py", "scene/gsplat_model.py"}:
        return "model_class", True, "Storage primitive called by an owner layer."
    if p == "scene/__init__.py" and q == "Scene.__init__":
        return (
            "source_depth",
            True,
            "Scene construction owns bootstrap population creation or checkpoint restore.",
        )
    if p == "eligsir/eval/_render.py" and q == "render_checkpoint_view":
        return (
            "checkpoint_io",
            True,
            "Read-only evaluation constructs its model from a checkpoint.",
        )
    if p.startswith("utils/strategies/mcmc/"):
        # Every production strategy returned by registry.make_mcmc_strategy is
        # wrapped by ArbiterOwnedMCMCStrategy. Raw writes below that wrapper are
        # delegated numerical execution, not an independent commit authority.
        return (
            "mcmc",
            True,
            "Delegated MCMC numerical executor; production step_post_backward is serialized and version-reconciled by GaussianMutationArbiter.",
        )
    if p.startswith("eligsir/optimization/gaussian/strategies/live/"):
        return "lifecycle", False, "Retired/inert live-strategy writer retained for source visibility."
    if p.startswith("eligsir/mapping/integration/"):
        if operation in {"grow", "retire", "relocate", "coordinate_correction"}:
            return (
                "mutation_transaction",
                True,
                "Production integration mutation is committed through GaussianMutationArbiter/map_backend; source_depth is a proposal producer, not the writer authority.",
            )
        if operation == "anchor_reference_update":
            return "anchors_depth_locks", True, "Insertion/anchor metadata owned by the mapping contract."
        return "source_depth", True, "Source-depth integration evidence/bookkeeping."
    if p == "scene/streaming_scene.py":
        if operation in {"grow", "create_bootstrap"}:
            return "source_depth", True, "Bootstrap/source-depth population creation."
        return "source_depth", True, "Streaming scene source evidence/bookkeeping."
    if p == "eligsir/mapping/canonical_scene.py":
        return (
            "source_depth",
            True,
            "Canonical bootstrap population creation from validated metric RGB-D.",
        )
    if p.startswith("loop_closure/"):
        owner = "loop_closure_merge" if "merge" in q.lower() else "map_frame_correction"
        return owner, True, "Coordinate correction committed through the canonical mutation transaction."
    if p == "utils/submap_manager.py" and "apply_pose_graph_corrections" in q:
        return "map_frame_correction", True, "Legacy unreachable correction branch; retained as a static source finding."
    if p == "utils/submap_manager.py" and operation == "identity_provenance_update":
        return (
            "source_depth",
            True,
            "Submap boundary owns the provenance tag applied to subsequent insertions.",
        )
    if p.startswith("utils/streaming_depth_lock.py"):
        return "freespace_contradiction", True, "Depth-lock/freespace support evidence."
    if p == "utils/seed_lineage_freeze.py":
        return "optimizer", True, "Seed-lineage optimizer guard restores protected parameter rows."
    if p.startswith("utils/geometry_metrics.py"):
        return "mcmc", True, "MCMC support/visibility evidence bookkeeping."
    if p.startswith("eligsir/optimization/gaussian/"):
        return "optimizer", True, "Fixed-topology optimizer/strategy mechanism."
    if (
        p == "eligsir/runtime/production_application.py"
        and q == "ContinualMappingRuntime._mutate_after_step"
        and site.kind == "strategy"
        and site.target == "self._mcmc_strategy.step_post_backward()"
    ):
        return (
            "mutation_transaction",
            True,
            "Canonical runtime dispatches the arbiter-owned MCMC mutation boundary.",
        )
    if p in {"train.py", "train_streaming.py"}:
        if operation == "checkpoint_restore":
            return "checkpoint_io", True, "Checkpoint restore."
        if operation in {"retire", "opacity"}:
            return "freespace_contradiction", True, "Trainer safety/capacity mutation is reconciled to the map backend."
        if operation in {"grow", "relocate"}:
            return "mutation_transaction", True, "Trainer mutation routes through canonical owner or delegated strategy."
        if operation in {"anchor_reference_update", "lifecycle_transition"}:
            return "anchors_depth_locks", True, "Trainer-side lifecycle/anchor bookkeeping."
        return "optimizer", True, "Trainer optimization bookkeeping."
    if p.startswith("scripts/") or p.startswith("viz/"):
        return "none", False, "Offline/manual diagnostic writer; not a production training owner."
    if operation in {"anchor_reference_update", "lifecycle_transition", "support_update"}:
        return "lifecycle", False, "Auxiliary lifecycle/support writer outside canonical map transaction."
    return "none", False, "Conservatively classified source-visible writer with no production mutation authority."


def _hard_safety(site: WriteSite, operation: str) -> bool:
    q = site.qualname.lower()
    if "sanitize" in q or "safety" in q:
        return True
    if site.path == "train_streaming.py" and operation in {"retire", "opacity"}:
        return True
    return False


def _sanctioned(path: str) -> bool:
    return any(path == prefix or path.startswith(prefix) for prefix in SANCTIONED_RAW_LAYERS)


def _rows_for_sites(sites: Iterable[WriteSite]) -> list[RegistryRow]:
    buckets: dict[tuple, dict] = {}
    for site in sites:
        op = _operation(site)
        owner, linked, note = _owner(site, op)
        prod = not _is_nonproduction_path(site.path, site.qualname)
        coordinate = _is_coordinate_correction(site.path, site.qualname)
        anchor_safe = _updates_anchor_reference(site.path, site.qualname)
        mechanism = owner in MECHANISM_OWNERS
        key = (
            site.path,
            site.qualname,
            op,
            owner,
            linked,
            prod,
            coordinate,
            anchor_safe,
            mechanism,
            _hard_safety(site, op),
        )
        bucket = buckets.setdefault(
            key,
            {
                "groups": set(),
                "lines": [],
                "kinds": set(),
                "examples": [],
                "note": note,
            },
        )
        bucket["groups"].add(site.state_group)
        bucket["lines"].append(int(site.lineno))
        bucket["kinds"].add(site.kind)
        if site.target not in bucket["examples"] and len(bucket["examples"]) < 4:
            bucket["examples"].append(site.target)

    rows: list[RegistryRow] = []
    for key in sorted(buckets):
        (
            path,
            qualname,
            op,
            owner,
            linked,
            prod,
            coordinate,
            anchor_safe,
            mechanism,
            hard_safety,
        ) = key
        b = buckets[key]
        rows.append(
            RegistryRow(
                path=path,
                qualname=qualname,
                state_written=sorted(b["groups"]),
                semantic_operation=op,
                declared_owner=owner,
                trigger="source-visible writer; runtime cadence determined by caller",
                gate="production call-site/config gate",
                evidence_consumed="see owning subsystem",
                hard_safety=bool(hard_safety),
                production=bool(prod),
                is_mechanism=bool(mechanism),
                sanctioned_raw_access=_sanctioned(path),
                coordinate_correction=bool(coordinate),
                updates_anchor_reference=bool(anchor_safe),
                linked_to_transaction=bool(linked),
                notes=b["note"],
                lines=sorted(set(b["lines"])),
                kinds=sorted(b["kinds"]),
                examples=list(b["examples"]),
            )
        )
    return rows


def _unclassified_rows(rows: Iterable[RegistryRow]) -> list[str]:
    """Return production mutation writers that lack an explicit owner."""

    return [
        row.key
        for row in rows
        if (
            row.production
            and row.declared_owner == "none"
            and row.semantic_operation != "not_a_gaussian_writer"
        )
    ]


def build_registry() -> RegistryResult:
    sites = discover()
    units = group_units(sites)
    rows = _rows_for_sites(sites)
    vocabulary_errors = []
    for row in rows:
        if row.semantic_operation not in SEMANTIC_OPERATIONS:
            vocabulary_errors.append(f"{row.key}: unknown operation {row.semantic_operation}")
        if row.declared_owner not in OWNERS:
            vocabulary_errors.append(f"{row.key}: unknown owner {row.declared_owner}")
    # There is no independent annotation table any more, so mechanical
    # discovery is the source list. Ownership still fails closed: a production
    # mutation that reaches the conservative ``none`` fallback is unclassified.
    return RegistryResult(
        rows=rows,
        unclassified=_unclassified_rows(rows),
        stale_annotations=[],
        vocabulary_errors=vocabulary_errors,
        n_sites=len(sites),
        n_units=len(units),
    )


# ---------------------------------------------------------------------------
# Conflict checks
# ---------------------------------------------------------------------------


def _row_dict(row: RegistryRow) -> dict:
    return {
        "unit": row.key,
        "operation": row.semantic_operation,
        "owner": row.declared_owner,
        "state_written": row.state_written,
        "lines": row.lines,
        "notes": row.notes,
    }


def _duplicate_owners(rows: Iterable[RegistryRow], operations: set[str] | None = None) -> dict:
    by_op: dict[str, dict[str, list[str]]] = {}
    for row in rows:
        if not row.production or row.hard_safety or row.is_mechanism:
            continue
        if operations is not None and row.semantic_operation not in operations:
            continue
        by_op.setdefault(row.semantic_operation, {}).setdefault(row.declared_owner, []).append(row.key)
    out = {}
    for op, owners in sorted(by_op.items()):
        if len(owners) > 1:
            out[op] = {
                "n_owners": len(owners),
                "owners": {k: sorted(set(v)) for k, v in sorted(owners.items())},
            }
    return out


def _family_duplicate_owners(rows: Iterable[RegistryRow]) -> dict:
    families = {
        "population_increase": {"grow", "create_bootstrap"},
        "removal_or_recycle": {"retire", "relocate"},
    }
    out = {}
    for family, ops in families.items():
        owners: dict[str, list[str]] = {}
        for row in rows:
            if (
                row.production
                and not row.hard_safety
                and not row.is_mechanism
                and row.semantic_operation in ops
            ):
                owners.setdefault(row.declared_owner, []).append(row.key)
        if len(owners) > 1:
            out[family] = {
                "operations": sorted(ops),
                "n_owners": len(owners),
                "owners": {k: sorted(set(v)) for k, v in sorted(owners.items())},
            }
    return out


def conflict_report(result: RegistryResult) -> dict:
    rows = result.rows
    check_a = _duplicate_owners(rows, {"grow", "relocate", "retire"})
    check_a_family = _family_duplicate_owners(rows)

    raw_kinds = {"assign", "augassign", "inplace", "alias_assign", "setattr"}
    check_b = [
        _row_dict(row)
        for row in rows
        if row.production
        and not row.sanctioned_raw_access
        and any(kind in raw_kinds for kind in row.kinds)
        and row.semantic_operation not in {"not_a_gaussian_writer", "strategy_dispatch"}
    ]
    check_c = [
        _row_dict(row)
        for row in rows
        if row.production
        and "population" in row.state_written
        and not row.linked_to_transaction
    ]
    check_c_capacity = [
        _row_dict(row)
        for row in rows
        if row.production
        and row.semantic_operation in {"retire", "relocate"}
        and "population" not in row.state_written
        and not row.linked_to_transaction
    ]
    check_d_rows = [
        row
        for row in rows
        if row.production and row.coordinate_correction and not row.updates_anchor_reference
    ]
    check_d = [_row_dict(row) for row in check_d_rows]

    sensitivity = []
    for row in rows:
        if not (row.production and row.hard_safety and not row.is_mechanism):
            continue
        flipped = []
        for op in ("grow", "relocate", "retire"):
            if row.semantic_operation != op:
                continue
            owners = {
                candidate.declared_owner
                for candidate in rows
                if candidate.production
                and not candidate.is_mechanism
                and candidate.semantic_operation == op
                and (not candidate.hard_safety or candidate is row)
            }
            if len(owners) > 1:
                flipped.append(op)
        sensitivity.append(
            {
                "unit": row.key,
                "check_a_operations_if_not_hard_safety": sorted(flipped),
            }
        )

    d_call_sites = sorted(
        row.key for row in check_d_rows if not row.is_mechanism
    )
    d_primitives = sorted(
        row.key for row in check_d_rows if row.is_mechanism
    )
    return {
        "check_a_duplicate_non_safety_owners": check_a,
        "check_a_family_duplicate_non_safety_owners": check_a_family,
        "check_b_raw_writes_outside_sanctioned_layers": check_b,
        "check_a_hard_safety_sensitivity": sensitivity,
        "check_c_topology_mutation_not_transaction_linked": check_c,
        "check_c_capacity_mutation_not_transaction_linked": check_c_capacity,
        "check_d_coordinate_correction_without_anchor_update": check_d,
        "check_d_breakdown": {
            "production_correction_call_sites": d_call_sites,
            "shared_write_primitives": d_primitives,
        },
    }


def summary(result: RegistryResult, conflicts: dict) -> dict:
    a = conflicts["check_a_duplicate_non_safety_owners"]
    af = conflicts["check_a_family_duplicate_non_safety_owners"]
    b = conflicts["check_b_raw_writes_outside_sanctioned_layers"]
    c = conflicts["check_c_topology_mutation_not_transaction_linked"]
    cc = conflicts["check_c_capacity_mutation_not_transaction_linked"]
    d = conflicts["check_d_coordinate_correction_without_anchor_update"]
    db = conflicts["check_d_breakdown"]
    return {
        "n_write_sites": result.n_sites,
        "n_writer_units": result.n_units,
        "n_classified_rows": len(result.rows),
        "n_production_rows": sum(int(r.production) for r in result.rows),
        "n_unclassified_sites": len(result.unclassified),
        "n_stale_annotations": len(result.stale_annotations),
        "no_unclassified_mutation_paths": not result.unclassified,
        "n_semantic_operations_with_competing_non_safety_owners": len(a),
        "semantic_operations_with_competing_non_safety_owners": sorted(a),
        "n_operation_families_with_competing_non_safety_owners": len(af),
        "operation_families_with_competing_non_safety_owners": sorted(af),
        "n_check_b_raw_writes": len(b),
        "n_check_c_unlinked_topology": len(c),
        "n_check_c_unlinked_capacity": len(cc),
        "n_check_d_coordinate_gaps": len(d),
        "n_check_d_production_call_sites": len(db["production_correction_call_sites"]),
        "n_check_d_shared_primitives": len(db["shared_write_primitives"]),
    }


_PRODUCTION_TEST_HINTS = {
    "eligsir/map/arbiter.py": "tests/test_owner_transfer_sequence.py",
    "eligsir/map/mutation.py": "tests/test_mutation_sites_backend_versions.py",
    "loop_closure/submap_correction.py": "tests/test_gaussian_merge_correction.py",
    "utils/strategies/mcmc/": "tests/test_owner_transfer_sequence.py",
    "eligsir/mapping/integration/": "tests/test_transfer_budget_contract.py",
}


def _test_hint(path: str) -> str:
    for prefix, hint in _PRODUCTION_TEST_HINTS.items():
        if path == prefix or path.startswith(prefix):
            return hint
    return "static writer-registry coverage"


def to_payload(result: RegistryResult, conflicts: dict) -> dict:
    registry = []
    for row in result.rows:
        registry.append(
            {
                "file/function": row.key,
                "state_written": row.state_written,
                "semantic_operation": row.semantic_operation,
                "declared_owner": row.declared_owner,
                "trigger/cadence": row.trigger,
                "gate": row.gate,
                "evidence_consumed": row.evidence_consumed,
                "hard_safety?": row.hard_safety,
                "production": row.production,
                "is_mechanism": row.is_mechanism,
                "sanctioned_raw_access": row.sanctioned_raw_access,
                "linked_to_transaction": row.linked_to_transaction,
                "updates_anchor_reference": row.updates_anchor_reference,
                "potential_competitors": [],
                "production_reader/writer test": _test_hint(row.path),
                "lines": row.lines,
                "kinds": row.kinds,
                "examples": row.examples,
                "notes": row.notes,
            }
        )
    return {
        "schema": "eligsir.gaussian-writer-registry/v2",
        "summary": summary(result, conflicts),
        "conflicts": conflicts,
        "known_blind_spots": list(KNOWN_BLIND_SPOTS),
        "registry": registry,
    }


def _fill_competitors(payload: dict) -> None:
    by_op: dict[str, set[str]] = {}
    for row in payload.get("registry", []):
        if not row.get("production") or row.get("hard_safety?") or row.get("is_mechanism"):
            continue
        by_op.setdefault(row["semantic_operation"], set()).add(row["declared_owner"])
    for row in payload.get("registry", []):
        owners = by_op.get(row["semantic_operation"], set())
        row["potential_competitors"] = sorted(
            owner for owner in owners if owner != row["declared_owner"]
        )


def _md_cell(value) -> str:
    if isinstance(value, (list, tuple)):
        value = ", ".join(str(x) for x in value)
    return str(value).replace("|", "\\|").replace("\n", " ")


def render_markdown(payload: dict) -> str:
    s = payload["summary"]
    lines = [
        "---",
        "visual_exempt: \"Generated ownership reference tables; the canonical "
        "control-flow diagram lives on the adjacent architecture pages.\"",
        "---",
        "",
        "# Gaussian writer/owner registry",
        "",
        "GENERATED by `python3 scripts/gaussian_writer_registry.py --write`.",
        "The v2 generator derives ownership from canonical production boundaries; raw delegated MCMC executor writes no longer masquerade as independent topology policy owners.",
        "",
        "## Summary",
        "",
    ]
    for key, value in s.items():
        lines.append(f"- `{key}`: {value}")
    lines += [
        "",
        "## Ownership interpretation",
        "",
        "- `GaussianMutationArbiter` is the sole production mutation commit policy owner.",
        "- MCMC raw tensor/state surgery is a delegated numerical executor under that arbiter and remains visible to raw-access check (b).",
        "- Source-depth insertion produces candidates; production population writes commit through the canonical transaction owner.",
        "- Fixed-topology optimizer steps advance map-content version but not topology/coordinate versions.",
        "- Retired live-strategy phases are retained as source-visible non-production rows rather than counted as competing active owners.",
        "",
        "## Conflict checks",
        "",
        "```json",
        json.dumps(payload["conflicts"], indent=2, sort_keys=True),
        "```",
        "",
        "## Registry",
        "",
        "| file/function | state_written | semantic_operation | declared_owner | trigger/cadence | gate | evidence_consumed | hard_safety? | potential_competitors | production_reader/writer test |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in payload["registry"]:
        cols = [
            row["file/function"],
            row["state_written"],
            row["semantic_operation"],
            row["declared_owner"],
            row["trigger/cadence"],
            row["gate"],
            row["evidence_consumed"],
            row["hard_safety?"],
            row["potential_competitors"],
            row["production_reader/writer test"],
        ]
        lines.append("| " + " | ".join(_md_cell(c) for c in cols) + " |")
    lines += ["", "## Known blind spots", ""]
    lines.extend(f"- {x}" for x in payload.get("known_blind_spots", []))
    return "\n".join(lines) + "\n"


def _build_payload() -> dict:
    result = build_registry()
    conflicts = conflict_report(result)
    payload = to_payload(result, conflicts)
    _fill_competitors(payload)
    return payload


def _write(payload: dict) -> None:
    out_dir = REPO_ROOT / "docs" / "architecture"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "gaussian_writer_registry.scan.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (out_dir / "gaussian_writer_registry.scan.md").write_text(
        render_markdown(payload), encoding="utf-8"
    )


def _main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--markdown", action="store_true")
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    payload = _build_payload()
    if args.write:
        _write(payload)
    elif args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    elif args.markdown:
        print(render_markdown(payload), end="")
    else:
        print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    result = build_registry()
    return 1 if (result.unclassified or result.stale_annotations or result.vocabulary_errors) else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
