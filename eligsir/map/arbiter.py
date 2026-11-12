"""Deterministic sole-policy owner for live Gaussian-map mutations.

Normal producers submit typed proposals. Legacy strategy executors whose
optimizer-state surgery cannot yet be expressed as a pure tensor plan are
forced through this owner via :class:`DelegatedMutationProposal`. The bridge
cannot be mixed with typed transaction proposals, backend version state is
reconciled immediately after the executor returns, and every committed
control-plane mutation is appended to the deterministic replay digest.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from enum import IntEnum
from typing import Callable, Optional, Sequence, Union

import torch

from eligsir.map.mutation import (
    GaussianInsertionBatch,
    RelocationPlan,
    SubmapCorrection,
    TopologyChange,
)
from eligsir.map.population_mutation_plan import PopulationMutationPlan
from eligsir.map.population_target_payload import PopulationTargetPayloadBatch
from eligsir.map.row_state import RowStateDescriptor
from eligsir.optimization.gaussian.supervision_evidence import EvidenceSnapshotToken
from eligsir.runtime.deterministic_digest import record_global


class MutationPriority(IntEnum):
    """Stable execution order for competing mutation proposal classes."""

    SAFETY_PRUNE = 0
    POSE_CORRECTION = 10
    LIFECYCLE_PRUNE = 20
    RELOCATION = 30
    SURFACE_INSERTION = 40
    APPEARANCE_STATE = 50
    DELEGATED_STRATEGY = 60
    POPULATION_PLAN = 65


@dataclass(frozen=True)
class ProposalKey:
    """Deterministic source identity used to order equal-priority proposals."""

    stream_epoch: int
    source_ordinal: int
    timestamp_ns: int
    source_id: str
    local_ordinal: int = 0

    def as_tuple(self) -> tuple:
        """Return the canonical sortable representation of this key."""

        return (
            int(self.stream_epoch), int(self.source_ordinal), int(self.timestamp_ns),
            str(self.source_id), int(self.local_ordinal),
        )


@dataclass(frozen=True)
class SurfaceInsertionProposal:
    """Request insertion of one prepared Gaussian batch."""

    key: ProposalKey
    batch: GaussianInsertionBatch
    priority: MutationPriority = MutationPriority.SURFACE_INSERTION


@dataclass(frozen=True)
class PruneProposal:
    """Request removal of the Gaussian rows selected by ``mask``."""

    key: ProposalKey
    mask: torch.Tensor
    priority: MutationPriority = MutationPriority.LIFECYCLE_PRUNE


@dataclass(frozen=True)
class RelocationProposal:
    """Request a backend-owned Gaussian relocation plan."""

    key: ProposalKey
    plan: RelocationPlan
    priority: MutationPriority = MutationPriority.RELOCATION


@dataclass(frozen=True)
class PoseCorrectionProposal:
    """Request application of a submap pose correction."""

    key: ProposalKey
    correction: SubmapCorrection
    priority: MutationPriority = MutationPriority.POSE_CORRECTION


@dataclass(frozen=True)
class OpacityUpdateProposal:
    """Request a deterministic opacity-logit update on selected rows."""

    key: ProposalKey
    mask: torch.Tensor
    value: object
    priority: MutationPriority = MutationPriority.APPEARANCE_STATE


@dataclass(frozen=True)
class DelegatedMutationEffect:
    """Exact effect returned by a delegated numerical mutation executor."""

    content_changed: bool
    topology_changed: bool
    relocated: int = 0
    corrected: int = 0

    def __post_init__(self) -> None:
        if type(self.content_changed) is not bool or type(self.topology_changed) is not bool:
            raise TypeError("delegated mutation change flags must be bool")
        if isinstance(self.relocated, bool) or int(self.relocated) < 0:
            raise ValueError("relocated must be a non-negative integer")
        if isinstance(self.corrected, bool) or int(self.corrected) < 0:
            raise ValueError("corrected must be a non-negative integer")


@dataclass(frozen=True)
class DelegatedMutationProposal:
    """Arbiter-owned bridge for a legacy strategy's coupled state mutation."""

    key: ProposalKey
    executor: Callable[[], object]
    content_may_change: bool = True
    topology_may_change: bool = True
    priority: MutationPriority = MutationPriority.DELEGATED_STRATEGY


@dataclass(frozen=True)
class PopulationMutationProposal:
    """One complete atomic population-plan transaction.

    The proposal is intentionally homogeneous and cannot be paired with an
    ordinary INSERT/PRUNE/RELOCATE proposal.  That prohibition prevents a
    future policy from silently decomposing a constant-capacity RELOCATE into
    old sidecar surgery on either side of the row-state transform.
    """

    key: ProposalKey
    plan: PopulationMutationPlan
    target_payload: PopulationTargetPayloadBatch
    row_state_registry_snapshot: tuple[RowStateDescriptor, ...]
    priority: MutationPriority = MutationPriority.POPULATION_PLAN

    def __post_init__(self) -> None:
        if not isinstance(self.key, ProposalKey):
            raise TypeError("population mutation proposal key must be a ProposalKey")
        if not isinstance(self.plan, PopulationMutationPlan):
            raise TypeError("population mutation proposal requires PopulationMutationPlan")
        if not isinstance(self.target_payload, PopulationTargetPayloadBatch):
            raise TypeError("population mutation proposal requires PopulationTargetPayloadBatch")
        if (
            not isinstance(self.row_state_registry_snapshot, tuple)
            or not all(isinstance(descriptor, RowStateDescriptor) for descriptor in self.row_state_registry_snapshot)
        ):
            raise TypeError("population mutation proposal requires an immutable registry snapshot")
        if self.row_state_registry_snapshot != self.plan.row_state_registry_snapshot:
            raise ValueError("proposal registry snapshot must exactly match its mutation plan")
        expected_targets = int(self.plan.relocate_target_rows.numel() + self.plan.create_rows.numel())
        if self.target_payload.count != expected_targets:
            raise ValueError("target payload count must exactly match plan target rows")
        # ``PopulationMutationPlan.target_payload`` remains the compatibility
        # position matrix while callers migrate to the complete typed payload.
        # Never allow two position truths to cross the same fence.
        if not torch.equal(self.plan.target_payload, self.target_payload.positions):
            raise ValueError("target payload positions must exactly match mutation plan positions")
        self.plan.validate(current_snapshot=self.plan.snapshot_token)


MutationProposal = Union[
    SurfaceInsertionProposal,
    PruneProposal,
    RelocationProposal,
    PoseCorrectionProposal,
    OpacityUpdateProposal,
    DelegatedMutationProposal,
    PopulationMutationProposal,
]


@dataclass(frozen=True)
class MutationCommitResult:
    """Aggregate result and capacity accounting for one arbiter commit."""

    change: TopologyChange
    submitted: int
    applied: int
    inserted: int
    capacity_dropped: int


def _proposal_sort_key(proposal: MutationProposal) -> tuple:
    return (int(proposal.priority),) + proposal.key.as_tuple()


def _slice_optional(value, count: int):
    return None if value is None else value[:count]


def _slice_insertion(batch: GaussianInsertionBatch, count: int) -> GaussianInsertionBatch:
    return replace(
        batch,
        points=batch.points[:count], colors=batch.colors[:count],
        normals=_slice_optional(batch.normals, count),
        scales=_slice_optional(batch.scales, count),
        rotations=_slice_optional(batch.rotations, count),
        opacities_raw=_slice_optional(batch.opacities_raw, count),
        sh_rest=_slice_optional(batch.sh_rest, count),
        confidence=_slice_optional(batch.confidence, count),
        source=_slice_optional(batch.source, count),
    )


_INSERT_TENSOR_FIELDS = (
    "points", "colors", "normals", "scales", "rotations", "opacities_raw",
    "sh_rest", "confidence", "source",
)
_INSERT_POLICY_FIELDS = (
    "init_scale", "init_opacity", "use_knn_scale", "is_provisional",
    "birth_frame", "birth_iter",
)


def _coalesce_insertions(batches: Sequence[GaussianInsertionBatch]) -> GaussianInsertionBatch:
    """Form one device-native insertion batch without touching map storage."""
    first = batches[0]
    for field in _INSERT_POLICY_FIELDS:
        if any(getattr(batch, field) != getattr(first, field) for batch in batches[1:]):
            raise ValueError(f"insertion proposals disagree on {field}")
    combined = {}
    for field in _INSERT_TENSOR_FIELDS:
        values = tuple(getattr(batch, field) for batch in batches)
        if all(value is None for value in values):
            combined[field] = None
            continue
        if any(value is None for value in values):
            raise ValueError(f"insertion proposals disagree on optional field {field}")
        reference = values[0]
        if any(
            value.device != reference.device
            or value.dtype != reference.dtype
            or value.shape[1:] != reference.shape[1:]
            for value in values[1:]
        ):
            raise ValueError(f"insertion proposals have incompatible {field} tensors")
        combined[field] = torch.cat(values, dim=0)
    return replace(first, **combined)


def _coalesce_prunes(proposals: Sequence[PruneProposal]) -> torch.Tensor:
    masks = tuple(proposal.mask for proposal in proposals)
    first = masks[0]
    if any(mask.device != first.device or mask.dtype != first.dtype for mask in masks[1:]):
        raise ValueError("prune proposals must use matching mask device and dtype")
    if first.dtype == torch.bool:
        if any(mask.shape != first.shape for mask in masks[1:]):
            raise ValueError("boolean prune masks must have matching shapes")
        return torch.stack(masks).any(dim=0)
    return torch.cat(masks).unique()


def _change_payload(change: TopologyChange) -> dict:
    return {
        "map_version": int(getattr(change, "map_version", 0)),
        "topology_version": int(getattr(change, "topology_version", 0)),
        "coordinate_frame_version": int(getattr(change, "coordinate_frame_version", 0)),
        "schema_version": int(getattr(change, "schema_version", 0)),
        "inserted": int(getattr(change, "inserted", 0)),
        "pruned": int(getattr(change, "pruned", 0)),
        "content_changed": bool(getattr(change, "content_changed", False)),
    }


def _proposal_payload(proposal: MutationProposal) -> dict:
    return {
        "type": type(proposal).__name__,
        "priority": int(proposal.priority),
        "key": proposal.key.as_tuple(),
    }


def _prevalidate_proposal(txn, proposal: MutationProposal) -> None:
    """Validate a proposal without writes; called for the whole batch first."""
    if isinstance(proposal, SurfaceInsertionProposal):
        validator = getattr(txn, "validate_insert", None)
        if validator is not None:
            validator(proposal.batch)
        else:
            count = proposal.batch.points.shape[0]
            if proposal.batch.points.ndim != 2 or proposal.batch.points.shape[1] != 3:
                raise ValueError("points must be a (N, 3) tensor")
            if proposal.batch.colors.shape != (count, 3):
                raise ValueError(f"colors must have shape ({count}, 3)")
    elif isinstance(proposal, PruneProposal):
        validator = getattr(txn, "validate_prune", None)
        if validator is not None:
            validator(proposal.mask)
    elif isinstance(proposal, RelocationProposal):
        validator = getattr(txn, "validate_relocate", None)
        if validator is not None:
            validator(proposal.plan)
    elif isinstance(proposal, PoseCorrectionProposal):
        validator = getattr(txn, "validate_pose_correction", None)
        if validator is not None:
            validator(proposal.correction)
    elif isinstance(proposal, OpacityUpdateProposal):
        validator = getattr(txn, "validate_opacity_update", None)
        if validator is not None:
            validator(proposal.mask, proposal.value)
    elif isinstance(proposal, PopulationMutationProposal):
        validator = getattr(txn, "validate_population_plan", None)
        if not callable(validator):
            raise TypeError(
                "mutation transaction must implement validate_population_plan; "
                "PopulationMutationProposal fails closed"
            )
        validator(
            proposal.plan,
            proposal.target_payload,
            proposal.row_state_registry_snapshot,
        )
    else:
        raise TypeError(f"unsupported mutation proposal {type(proposal).__name__}")


class GaussianMutationArbiter:
    """The only policy object allowed to turn mutation proposals into writes."""

    def __init__(self, backend, *, max_population: Optional[int] = None):
        self._backend = backend
        self._max_population = None if max_population is None else int(max_population)
        if self._max_population is not None and self._max_population < 0:
            raise ValueError("max_population must be >= 0")

    def _population(self) -> int:
        """Read the row count through the backend's O(1) ownership contract.

        Capacity arbitration needs one integer. Reading it off
        ``GaussianMapBackend.population_count`` avoids materializing a whole
        read view (and its parameter tensor references) per commit.
        """
        return int(self._backend.population_count)

    def _record_delegated_change(
        self, *, inserted: int, pruned: int, content_changed: bool,
        topology_changed: bool, relocated: int = 0, corrected: int = 0,
    ) -> TopologyChange:
        recorder = getattr(self._backend, "record_delegated_mutation", None)
        if callable(recorder):
            return recorder(
                inserted=int(inserted),
                pruned=int(pruned),
                content_changed=bool(content_changed),
                topology_changed=bool(topology_changed),
                relocated=int(relocated),
                corrected=int(corrected),
            )

        external = getattr(self._backend, "record_external_mutation", None)
        if not callable(external):
            raise TypeError(
                "backend must implement record_delegated_mutation or record_external_mutation"
            )
        before_content = int(getattr(self._backend, "map_content_version", 0))
        count_changed = bool(inserted or pruned)
        external(inserted=int(inserted), pruned=int(pruned))
        if topology_changed and not count_changed:
            if not hasattr(self._backend, "_topology_version"):
                raise TypeError(
                    "equal-count delegated topology mutation requires "
                    "record_delegated_mutation"
                )
            self._backend._topology_version += 1
        if content_changed and not count_changed:
            if not hasattr(self._backend, "_map_content_version"):
                raise TypeError(
                    "content-only delegated mutation requires record_delegated_mutation"
                )
            self._backend._map_content_version = before_content + 1

        return TopologyChange(
            map_version=int(getattr(self._backend, "map_content_version", before_content)),
            topology_version=int(getattr(self._backend, "topology_version", 0)),
            coordinate_frame_version=int(
                getattr(self._backend, "coordinate_frame_version", 0)
            ),
            schema_version=int(getattr(self._backend, "schema_version", 0)),
            inserted=int(inserted),
            pruned=int(pruned),
            relocated=int(relocated),
            corrected=int(corrected),
            content_changed=bool(content_changed or topology_changed or count_changed),
        )

    def _commit_delegated(self, proposal: DelegatedMutationProposal) -> MutationCommitResult:
        before = self._population()
        execution_result = proposal.executor()
        after = self._population()
        inserted = max(after - before, 0)
        pruned = max(before - after, 0)
        effect = (
            execution_result
            if isinstance(execution_result, DelegatedMutationEffect)
            else DelegatedMutationEffect(
                content_changed=bool(proposal.content_may_change),
                topology_changed=bool(proposal.topology_may_change),
            )
        )
        change = self._record_delegated_change(
            inserted=inserted,
            pruned=pruned,
            content_changed=bool(effect.content_changed),
            topology_changed=bool(effect.topology_changed or inserted or pruned),
            relocated=int(effect.relocated),
            corrected=int(effect.corrected),
        )
        result = MutationCommitResult(
            change=change,
            submitted=1,
            applied=1,
            inserted=inserted,
            capacity_dropped=0,
        )
        record_global(
            "mutation_commit",
            delegated=True,
            proposals=[_proposal_payload(proposal)],
            population_before=before,
            population_after=after,
            submitted=1,
            applied=1,
            capacity_dropped=0,
            **_change_payload(change),
        )
        return result

    def _current_population_plan_snapshot(self) -> EvidenceSnapshotToken:
        """Read the authoritative all-version snapshot or fail closed.

        A population plan is version-bound evidence.  Reconstructing a token
        from a subset of backend counters would make parameter/optimizer or
        policy staleness invisible, so an opt-in backend must expose the exact
        immutable token itself.
        """

        provider = getattr(self._backend, "current_population_plan_snapshot", None)
        if not callable(provider):
            raise TypeError(
                "backend does not support current_population_plan_snapshot; "
                "PopulationMutationProposal fails closed"
            )
        snapshot = provider()
        if not isinstance(snapshot, EvidenceSnapshotToken):
            raise TypeError("backend population-plan snapshot must be an EvidenceSnapshotToken")
        return snapshot

    def _authoritative_population_plan_registry_snapshot(
        self, txn
    ) -> tuple[RowStateDescriptor, ...]:
        """Read a concrete row-state attestation before any raw plan apply.

        The proposal's tuple is a frozen description, not proof that the live
        backend has the same concrete storage inventory.  A transaction may
        attest it directly; a backend-level capability is also allowed for
        layouts whose transaction object is intentionally thin.  No provider
        means no mutation.
        """

        snapshot = None
        for owner in (txn, self._backend):
            provider = getattr(owner, "current_population_plan_registry_snapshot", None)
            if callable(provider):
                snapshot = provider()
                break
        if snapshot is None:
            raise TypeError(
                "mutation transaction/backend must provide a concrete population-plan "
                "registry snapshot; PopulationMutationProposal fails closed"
            )
        if (
            not isinstance(snapshot, tuple)
            or not all(isinstance(descriptor, RowStateDescriptor) for descriptor in snapshot)
            or len({descriptor.name for descriptor in snapshot}) != len(snapshot)
        ):
            raise TypeError("authoritative population-plan registry snapshot is invalid")
        return snapshot

    def _commit_population_plan(
        self, proposal: PopulationMutationProposal
    ) -> MutationCommitResult:
        """Apply one non-decomposed plan through one backend transaction call."""

        current_snapshot = self._current_population_plan_snapshot()
        proposal.plan.validate(current_snapshot=current_snapshot)
        population_before = self._population()
        if proposal.plan.old_population != population_before:
            raise ValueError("population mutation plan old population is stale")

        txn = self._backend.begin_mutation()
        authoritative_registry = self._authoritative_population_plan_registry_snapshot(txn)
        if authoritative_registry != proposal.row_state_registry_snapshot:
            raise ValueError(
                "authoritative population-plan registry snapshot does not match proposal"
            )
        _prevalidate_proposal(txn, proposal)
        apply_plan = getattr(txn, "apply_population_plan", None)
        if not callable(apply_plan):
            raise TypeError(
                "mutation transaction does not implement apply_population_plan; "
                "PopulationMutationProposal fails closed"
            )
        # This is intentionally the only raw population-mutating operation in
        # this branch.  In particular, never call insert/prune/relocate here.
        commit = getattr(self._backend, "commit_mutation", None)
        if not callable(commit):
            raise TypeError("mutation backend must own commit_mutation(transaction)")
        apply_plan(
            proposal.plan,
            proposal.target_payload,
            proposal.row_state_registry_snapshot,
        )
        change = commit(txn)
        committed_snapshot = self._current_population_plan_snapshot()
        if committed_snapshot != proposal.plan.next_snapshot_token:
            raise RuntimeError(
                "backend post-mutation snapshot does not exactly match the population plan"
            )
        population_after = self._population()
        if population_after != proposal.plan.new_population:
            raise RuntimeError("backend population does not match committed population plan")
        result = MutationCommitResult(
            change=change,
            submitted=1,
            applied=1,
            inserted=int(getattr(change, "inserted", 0)),
            capacity_dropped=0,
        )
        record_global(
            "mutation_commit",
            delegated=False,
            population_plan=True,
            proposals=[_proposal_payload(proposal)],
            population_before=population_before,
            population_after=population_after,
            submitted=1,
            applied=1,
            capacity_dropped=0,
            **_change_payload(change),
        )
        return result

    def commit(self, proposals: Sequence[MutationProposal]) -> MutationCommitResult:
        """Fence the complete mutation epoch against asynchronous readers."""
        lock = getattr(self._backend, "state_lock", None)
        if lock is None:
            return self._commit_fenced(proposals)
        with lock:
            return self._commit_fenced(proposals)

    def _commit_fenced(self, proposals: Sequence[MutationProposal]) -> MutationCommitResult:
        """Order and atomically apply one compatible set of proposals."""

        ordered = sorted(tuple(proposals), key=_proposal_sort_key)
        delegated = [p for p in ordered if isinstance(p, DelegatedMutationProposal)]
        if delegated:
            if len(ordered) != 1:
                raise ValueError(
                    "DelegatedMutationProposal must be committed alone; mixed legacy and typed "
                    "writes would make transaction ordering/version semantics ambiguous"
                )
            return self._commit_delegated(delegated[0])

        population_plans = [p for p in ordered if isinstance(p, PopulationMutationProposal)]
        if population_plans:
            if len(ordered) != 1:
                raise ValueError(
                    "PopulationMutationProposal must be committed alone; mixed typed "
                    "writes would decompose the atomic population plan"
                )
            return self._commit_population_plan(population_plans[0])

        proposal_types = {type(proposal) for proposal in ordered}
        if len(proposal_types) > 1:
            raise ValueError(
                "typed mutation commits must be homogeneous so exactly one backend "
                "mutation operation crosses the transaction boundary"
            )
        if len(ordered) > 1 and proposal_types and next(iter(proposal_types)) not in {
            SurfaceInsertionProposal, PruneProposal,
        }:
            raise ValueError(
                f"multiple {next(iter(proposal_types)).__name__} values cannot be "
                "coalesced into one failure-atomic backend operation"
            )

        txn = self._backend.begin_mutation()
        for proposal in ordered:
            _prevalidate_proposal(txn, proposal)
        population_before = self._population()
        population = population_before
        applied = inserted = capacity_dropped = 0
        if ordered and isinstance(ordered[0], SurfaceInsertionProposal):
            accepted_batches = []
            for proposal in ordered:
                batch = proposal.batch
                requested = int(batch.points.shape[0])
                if requested == 0:
                    continue
                allowed = requested
                if self._max_population is not None:
                    allowed = max(0, min(requested, self._max_population - population))
                if allowed == 0:
                    capacity_dropped += requested
                    continue
                if allowed != requested:
                    batch = _slice_insertion(batch, allowed)
                    capacity_dropped += requested - allowed
                accepted_batches.append(batch)
                population += allowed
                applied += 1
            if accepted_batches:
                combined = _coalesce_insertions(accepted_batches)
                inserted = int(txn.insert(combined))
                population = population_before + inserted
        elif ordered and isinstance(ordered[0], PruneProposal):
            proposal = ordered[0]
            combined_mask = _coalesce_prunes(ordered)
            before_prune = population
            reported = txn.prune(combined_mask)
            if reported is None:
                count = max(0, before_prune - self._population())
            else:
                count = int(reported)
            if count:
                population = max(0, population - count)
                applied = len(ordered)
        elif ordered:
            proposal = ordered[0]
            if isinstance(proposal, RelocationProposal):
                txn.relocate(proposal.plan)
                applied = 1
            elif isinstance(proposal, PoseCorrectionProposal):
                txn.apply_pose_correction(proposal.correction)
                applied = 1
            elif isinstance(proposal, OpacityUpdateProposal):
                txn.update_opacity_logits(proposal.mask, proposal.value)
                applied = 1
            else:  # prevalidation already rejects this; defensive fail closed
                raise TypeError(f"unsupported mutation proposal {type(proposal).__name__}")

        commit = getattr(self._backend, "commit_mutation", None)
        if commit is None:
            raise TypeError("mutation backend must own commit_mutation(transaction)")
        change = commit(txn)
        result = MutationCommitResult(change, len(ordered), applied, inserted, capacity_dropped)
        record_global(
            "mutation_commit",
            delegated=False,
            proposals=[_proposal_payload(p) for p in ordered],
            population_before=population_before,
            population_after=population,
            submitted=len(ordered),
            applied=applied,
            capacity_dropped=capacity_dropped,
            **_change_payload(change),
        )
        return result


__all__ = [
    "GaussianMutationArbiter", "MutationCommitResult", "MutationPriority",
    "OpacityUpdateProposal", "PoseCorrectionProposal", "ProposalKey",
    "PruneProposal", "RelocationProposal", "SurfaceInsertionProposal",
    "DelegatedMutationProposal", "PopulationMutationProposal",
]
