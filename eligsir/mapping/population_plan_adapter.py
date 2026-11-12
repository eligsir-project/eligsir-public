"""Fail-closed M5 bridge from one policy decision to one atomic plan.

This boundary is deliberately a *fence* adapter.  Policy and surface-target
selection happen on device before it is called; the adapter is allowed to
materialize only the bounded, already-authorized target rows on CPU at the
attributed ``MUTATION_CARDINALITY`` fence.  It neither changes a Gaussian row
nor submits to the arbiter.  Callers must commit the returned homogeneous
``PopulationMutationProposal`` alone.

The policy action vector has no implicit target-row mapping.  A caller must
therefore preselect an exactly-sized ``SurfaceTargetBatch``.  Rejecting an
undersized *or* oversized batch is intentional: guessing a subset would make
the action owner ambiguous again.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math

import torch

from eligsir.map.arbiter import PopulationMutationProposal, ProposalKey
from eligsir.map.population_mutation_plan import PopulationMutationPlan
from eligsir.map.population_target_payload import (
    PopulationTargetPayloadBatch,
    PopulationTargetPayloadSchema,
)
from eligsir.map.row_state import (
    GaussianRowStateRegistry,
    PopulationRowCreation,
    PopulationRowRelocation,
    PopulationRowTransform,
    RowStateDescriptor,
)
from eligsir.mapping.evidence.surface_targets import (
    SurfaceTargetBatch,
    SurfaceTargetSnapshot,
)
from eligsir.mapping.evidence.surface_target_attestation import (
    SurfaceTargetQualificationAttestation,
)
from eligsir.mapping.population_policy import (
    PopulationAction,
    PopulationDecision,
    PopulationStructuralSelection,
)
from eligsir.runtime.transfers import DeviceTransferOwner, PackedReadback
from eligsir.optimization.gaussian.supervision_evidence import EvidenceSnapshotToken


class PopulationPlanDropReason(Enum):
    """Why a decision was not allowed to cross the mutation fence."""

    READY = "ready"
    NO_STRUCTURAL_ACTION = "no_structural_action"
    STALE_DECISION = "stale_decision"
    STALE_TARGETS = "stale_targets"
    MISSING_TARGETS = "missing_targets"
    AMBIGUOUS_TARGET_PAYLOAD = "ambiguous_target_payload"
    MIXED_STRUCTURAL_ACTIONS = "mixed_structural_actions"
    MIXED_NONSTRUCTURAL_ACTIONS = "mixed_nonstructural_actions"
    CAPACITY_CONFLICT = "capacity_conflict"
    INVALID_UID_INVENTORY = "invalid_uid_inventory"
    INVALID_REGISTRY = "invalid_registry"
    INVALID_DECISION = "invalid_decision"
    INVALID_TARGET_PAYLOAD = "invalid_target_payload"
    TARGET_PAIR_MISMATCH = "target_pair_mismatch"
    ACTION_BUDGET_EXCEEDED = "action_budget_exceeded"


@dataclass(frozen=True)
class PopulationMutationFenceTelemetry:
    """Attribution for the sole bounded staging transfer at a mutation fence."""

    reason: str
    snapshot: EvidenceSnapshotToken
    source_device: torch.device
    packed_bytes: int
    transfer_calls: int
    transfer_owner_d2h_ops_before: int | None = None
    transfer_owner_d2h_ops_after: int | None = None
    event_name: str = "MUTATION_CARDINALITY_STAGING_COMPLETE"

    def __post_init__(self) -> None:
        if self.reason != "MUTATION_CARDINALITY":
            raise ValueError("population fence telemetry requires MUTATION_CARDINALITY")
        if type(self.packed_bytes) is not int or self.packed_bytes < 0:
            raise ValueError("packed_bytes must be non-negative")
        if self.transfer_calls != 1:
            raise ValueError("population fence must issue exactly one packed staging transfer")
        if (self.transfer_owner_d2h_ops_before is None) != (self.transfer_owner_d2h_ops_after is None):
            raise ValueError("transfer-owner D2H accounting must be complete or absent")
        if self.transfer_owner_d2h_ops_before is not None and (
            self.transfer_owner_d2h_ops_after != self.transfer_owner_d2h_ops_before + 1
        ):
            raise ValueError("population fence must account for exactly one owner D2H operation")


@dataclass(frozen=True)
class _PackedField:
    name: str
    offset: int
    shape: tuple[int, ...]
    dtype: torch.dtype


@dataclass(frozen=True)
class _FenceStaging:
    backing: torch.Tensor
    fields: tuple[_PackedField, ...]
    telemetry: PopulationMutationFenceTelemetry
    readback: PackedReadback

    def value(self, name: str) -> torch.Tensor:
        for field in self.fields:
            if field.name == name:
                count = math.prod(field.shape)
                byte_count = count * torch.empty((), dtype=field.dtype).element_size()
                return self.backing.narrow(0, field.offset, byte_count).view(field.dtype).reshape(field.shape)
        raise KeyError(name)

    def has(self, name: str) -> bool:
        return any(field.name == name for field in self.fields)

    def release(self) -> None:
        self.readback.release_after_wait()


@dataclass(frozen=True)
class PopulationPlanAdapterContext:
    """Authoritative facts captured at one prospective mutation fence.

    ``next_hypothesis_uid`` is a reservation watermark, not an allocator
    mutation.  The returned proposal is valid only for ``current_snapshot``;
    a rejected/stale proposal cannot reserve identities in the live map.
    """

    current_snapshot: EvidenceSnapshotToken
    decision_snapshot: EvidenceSnapshotToken
    capacity: int
    hypothesis_uids: tuple[str, ...]
    next_hypothesis_uid: int
    row_state_registry: GaussianRowStateRegistry
    proposal_key: ProposalKey
    max_actions: int
    transfer_owner: DeviceTransferOwner

    def __post_init__(self) -> None:
        if not isinstance(self.current_snapshot, EvidenceSnapshotToken):
            raise TypeError("current_snapshot must be an EvidenceSnapshotToken")
        if not isinstance(self.decision_snapshot, EvidenceSnapshotToken):
            raise TypeError("decision_snapshot must be an EvidenceSnapshotToken")
        if type(self.capacity) is not int or self.capacity < 0:
            raise ValueError("capacity must be a non-negative integer")
        if not isinstance(self.hypothesis_uids, tuple):
            raise TypeError("hypothesis_uids must be an immutable tuple")
        if type(self.next_hypothesis_uid) is not int or self.next_hypothesis_uid < 0:
            raise ValueError("next_hypothesis_uid must be a non-negative integer")
        if not isinstance(self.row_state_registry, GaussianRowStateRegistry):
            raise TypeError("row_state_registry must be GaussianRowStateRegistry")
        if not isinstance(self.proposal_key, ProposalKey):
            raise TypeError("proposal_key must be ProposalKey")
        if type(self.max_actions) is not int or self.max_actions < 0:
            raise ValueError("max_actions must be a non-negative integer")
        if not isinstance(self.transfer_owner, DeviceTransferOwner):
            raise TypeError("transfer_owner must be DeviceTransferOwner")

    @property
    def population(self) -> int:
        return len(self.hypothesis_uids)


@dataclass(frozen=True)
class PopulationPlanBuildResult:
    """The only output of the adapter; no mutation is performed here."""

    proposal: PopulationMutationProposal | None
    reason: PopulationPlanDropReason
    fence: PopulationMutationFenceTelemetry | None = None
    target_attestation: SurfaceTargetQualificationAttestation | None = None


@dataclass(frozen=True)
class _PayloadLayout:
    means: RowStateDescriptor
    scales: RowStateDescriptor
    quats: RowStateDescriptor
    opacities: RowStateDescriptor
    sh0: RowStateDescriptor
    shn: RowStateDescriptor


class PopulationPolicyPlanAdapter:
    """Materialize the sole permitted population proposal at the host fence."""

    _STRUCTURAL = frozenset({
        PopulationAction.CREATE,
        PopulationAction.RELOCATE,
        PopulationAction.RETIRE,
    })
    _NONSTRUCTURAL = frozenset({PopulationAction.CORRECT, PopulationAction.EXPLORE})

    def __init__(
        self,
        *,
        neutral_rgb: tuple[float, float, float] = (0.5, 0.5, 0.5),
        initial_opacity_logit: float = 0.0,
        minimum_scale: float = 1e-3,
    ) -> None:
        if (
            not isinstance(neutral_rgb, tuple)
            or len(neutral_rgb) != 3
            or any(type(value) not in (int, float) or not math.isfinite(value) for value in neutral_rgb)
        ):
            raise ValueError("neutral_rgb must be three finite scalar values")
        if type(initial_opacity_logit) not in (int, float) or not math.isfinite(initial_opacity_logit):
            raise ValueError("initial_opacity_logit must be finite")
        if type(minimum_scale) not in (int, float) or not math.isfinite(minimum_scale) or minimum_scale <= 0:
            raise ValueError("minimum_scale must be positive and finite")
        self._neutral_rgb = tuple(float(value) for value in neutral_rgb)
        self._initial_opacity_logit = float(initial_opacity_logit)
        self._minimum_scale = float(minimum_scale)

    def build(
        self,
        *,
        decision: PopulationDecision,
        targets: SurfaceTargetBatch | None,
        context: PopulationPlanAdapterContext,
    ) -> PopulationPlanBuildResult:
        """Build one immutable proposal only when every fence fact matches."""

        if not isinstance(decision, PopulationDecision):
            raise TypeError("decision must be a PopulationDecision")
        if not isinstance(context, PopulationPlanAdapterContext):
            raise TypeError("context must be PopulationPlanAdapterContext")
        if context.decision_snapshot != context.current_snapshot:
            return self._drop(PopulationPlanDropReason.STALE_DECISION)
        if not decision.has_authentic_structural_selection():
            return self._drop(PopulationPlanDropReason.INVALID_DECISION)
        if context.population > context.capacity:
            return self._drop(PopulationPlanDropReason.CAPACITY_CONFLICT)
        if not self._valid_uid_inventory(context):
            return self._drop(PopulationPlanDropReason.INVALID_UID_INVENTORY)
        layout = self._payload_layout(context.row_state_registry)
        if layout is None:
            return self._drop(PopulationPlanDropReason.INVALID_REGISTRY)
        if targets is not None:
            if not isinstance(targets, SurfaceTargetBatch):
                return self._drop(PopulationPlanDropReason.MISSING_TARGETS)
            if not targets.is_current(self._expected_surface_snapshot(context.current_snapshot)):
                return self._drop(PopulationPlanDropReason.STALE_TARGETS)
            if targets.device != decision.structural_selection.source_rows.device:
                return self._drop(PopulationPlanDropReason.INVALID_TARGET_PAYLOAD)
            if targets.count > context.max_actions:
                return self._drop(PopulationPlanDropReason.AMBIGUOUS_TARGET_PAYLOAD)
            if decision.structural_selection._bound_target_snapshot is None:
                return self._drop(PopulationPlanDropReason.TARGET_PAIR_MISMATCH)
            if decision.structural_selection._bound_target_snapshot != targets.snapshot:
                return self._drop(PopulationPlanDropReason.TARGET_PAIR_MISMATCH)
            if decision.structural_selection._target_binding_action is PopulationAction.RETIRE:
                return self._drop(PopulationPlanDropReason.TARGET_PAIR_MISMATCH)
        staging = self._stage_at_fence(
            selection=decision.structural_selection,
            actions=decision.actions,
            target_indices=decision.target_indices,
            targets=targets,
            snapshot=context.current_snapshot,
            transfer_owner=context.transfer_owner,
        )
        if staging is None:
            return self._drop(PopulationPlanDropReason.INVALID_DECISION)
        try:
            selected = self._selected_actions_at_fence(staging, context.population, context.max_actions)
            if selected is None:
                return self._drop(PopulationPlanDropReason.INVALID_DECISION, staging.telemetry)
            if not selected:
                return self._drop(PopulationPlanDropReason.NO_STRUCTURAL_ACTION, staging.telemetry)
            structural = [action for _, action, _ in selected]
            kinds = set(structural)
            if len(kinds) != 1:
                return self._drop(PopulationPlanDropReason.MIXED_STRUCTURAL_ACTIONS, staging.telemetry)
            if len(structural) > context.max_actions:
                return self._drop(PopulationPlanDropReason.ACTION_BUDGET_EXCEEDED, staging.telemetry)

            action = structural[0]
            if action is PopulationAction.CREATE:
                if context.population + len(structural) > context.capacity:
                    return self._drop(PopulationPlanDropReason.CAPACITY_CONFLICT, staging.telemetry)
                return self._build_create(selected, staging, context, layout)
            if action is PopulationAction.RELOCATE:
                if context.population != context.capacity:
                    return self._drop(PopulationPlanDropReason.CAPACITY_CONFLICT, staging.telemetry)
                return self._build_relocate(selected, staging, context, layout)
            return self._build_retire(selected, staging, context, layout)
        finally:
            staging.release()

    @staticmethod
    def _drop(
        reason: PopulationPlanDropReason,
        fence: PopulationMutationFenceTelemetry | None = None,
    ) -> PopulationPlanBuildResult:
        return PopulationPlanBuildResult(proposal=None, reason=reason, fence=fence)

    @staticmethod
    def _expected_surface_snapshot(token: EvidenceSnapshotToken) -> SurfaceTargetSnapshot:
        return SurfaceTargetSnapshot(
            map_content=token.map_content,
            topology=token.topology,
            coordinate_frame=token.coordinate_frame,
            schema=token.schema,
            surface_evidence=token.surface_evidence,
        )

    def _target_payload(
        self,
        staging: _FenceStaging,
        *,
        expected_count: int,
        expected_target_ids: tuple[int, ...],
        context: PopulationPlanAdapterContext,
        layout: _PayloadLayout,
    ) -> tuple[PopulationTargetPayloadBatch | None, PopulationPlanDropReason | None]:
        if expected_count == 0:
            if staging.has("target_ids") and staging.value("target_ids").numel() != 0:
                return None, PopulationPlanDropReason.AMBIGUOUS_TARGET_PAYLOAD
            return self._empty_payload(layout), None
        if not staging.has("target_ids"):
            return None, PopulationPlanDropReason.MISSING_TARGETS
        if staging.value("target_ids").numel() != expected_count:
            return None, PopulationPlanDropReason.AMBIGUOUS_TARGET_PAYLOAD
        try:
            # The input contract is already shape/device checked on GPU.  At
            # this one attributed fence, materialize the bounded selected
            # geometry exactly once, then validate the concrete CPU payload.
            # Calling SurfaceTargetBatch.validate_debug() here would instead
            # introduce several separate device-scalar synchronizations.
            # These copies sever the returned proposal from the leased pooled
            # byte slab before its finally-path release.  Every payload tensor
            # below is therefore stable across later mutation-fence readbacks.
            positions = staging.value("positions").to(dtype=layout.means.dtype).clone()
            uncertainty = staging.value("uncertainty").to(dtype=layout.scales.dtype).clone()
            normals = staging.value("normals").to(dtype=layout.quats.dtype).clone()
            target_ids = tuple(staging.value("target_ids").to(dtype=torch.int64).tolist())
            if target_ids != expected_target_ids or len(set(target_ids)) != len(target_ids):
                return None, PopulationPlanDropReason.TARGET_PAIR_MISMATCH
            if not (
                staging.has("support_evidence")
                and staging.has("conflict_evidence")
                and staging.has("independent_view_count")
            ):
                return None, PopulationPlanDropReason.INVALID_TARGET_PAYLOAD
            support = staging.value("support_evidence")
            conflict = staging.value("conflict_evidence")
            independent = staging.value("independent_view_count")
            if (
                support.numel() != expected_count
                or conflict.numel() != expected_count
                or independent.numel() != expected_count
                or not bool(torch.isfinite(support).all())
                or not bool(torch.isfinite(conflict).all())
                or bool((support <= 0).any())
                or bool((conflict < 0).any())
                or bool((independent < 2).any())
            ):
                return None, PopulationPlanDropReason.INVALID_TARGET_PAYLOAD
            if not bool(torch.isfinite(positions).all()) or not bool(torch.isfinite(uncertainty).all()):
                return None, PopulationPlanDropReason.INVALID_TARGET_PAYLOAD
            if not bool(torch.isfinite(normals).all()):
                return None, PopulationPlanDropReason.INVALID_TARGET_PAYLOAD
            if bool((uncertainty < 0).any()):
                return None, PopulationPlanDropReason.INVALID_TARGET_PAYLOAD
            colors = torch.full(
                (expected_count, 3), self._neutral_rgb[0], dtype=layout.sh0.dtype,
            )
            colors[:, 1] = self._neutral_rgb[1]
            colors[:, 2] = self._neutral_rgb[2]
            raw_scales = torch.log(uncertainty.clamp_min(self._minimum_scale))
            quaternions = self._surface_aligned_quaternions(normals)
            opacity_logits = torch.full(
                (expected_count, 1), self._initial_opacity_logit, dtype=layout.opacities.dtype,
            )
            sh_rest = torch.zeros((expected_count,) + layout.shn.shape_suffix, dtype=layout.shn.dtype)
            payload = PopulationTargetPayloadBatch(
                schema=PopulationTargetPayloadSchema(), positions=positions,
                colors=colors, raw_scales=raw_scales, quaternions=quaternions,
                opacity_logits=opacity_logits, sh_rest=sh_rest, opaque_values=(),
                max_targets=expected_count,
            )
            payload.validate_debug()
            return payload, None
        except (TypeError, ValueError, RuntimeError):
            return None, PopulationPlanDropReason.INVALID_TARGET_PAYLOAD

    @staticmethod
    def _empty_payload(layout: _PayloadLayout) -> PopulationTargetPayloadBatch:
        return PopulationTargetPayloadBatch(
            schema=PopulationTargetPayloadSchema(),
            positions=torch.empty((0, 3), dtype=layout.means.dtype),
            colors=torch.empty((0, 3), dtype=layout.sh0.dtype),
            raw_scales=torch.empty((0, 3), dtype=layout.scales.dtype),
            quaternions=torch.empty((0, 4), dtype=layout.quats.dtype),
            opacity_logits=torch.empty((0, 1), dtype=layout.opacities.dtype),
            sh_rest=torch.empty((0,) + layout.shn.shape_suffix, dtype=layout.shn.dtype),
            opaque_values=(), max_targets=0,
        )

    def _build_create(
        self, selected: list[tuple[int, PopulationAction, int]], staging: _FenceStaging,
        context: PopulationPlanAdapterContext, layout: _PayloadLayout,
    ) -> PopulationPlanBuildResult:
        create_old_rows = [row for row, _, _ in selected]
        target_ids = tuple(target_id for _, _, target_id in selected)
        payload, reason = self._target_payload(
            staging, expected_count=len(create_old_rows), expected_target_ids=target_ids,
            context=context, layout=layout,
        )
        if reason is not None:
            return self._drop(reason, staging.telemetry)
        assert payload is not None
        old = context.population
        new = old + len(create_old_rows)
        created_uids = self._fresh_uids(context, len(create_old_rows))
        create_rows = torch.arange(old, new, dtype=torch.int64)
        transform = PopulationRowTransform(
            old_population=old, new_population=new,
            survivor_old_indices=torch.arange(old, dtype=torch.int64),
            retired_old_indices=torch.empty(0, dtype=torch.int64),
            created_rows=tuple(PopulationRowCreation(int(row), uid) for row, uid in zip(create_rows.tolist(), created_uids)),
            relocated_rows=(), covered_fields=context.row_state_registry.names,
        )
        return self._proposal(
            context=context, row_transform=transform, new_population=new,
            retire_rows=torch.empty(0, dtype=torch.int64),
            relocate_donor_rows=torch.empty(0, dtype=torch.int64),
            relocate_target_rows=torch.empty(0, dtype=torch.int64), create_rows=create_rows,
            donor_uids=(), relocated_uids=(), created_uids=created_uids, payload=payload,
            fence=staging.telemetry,
            target_validated=True,
        )

    def _build_relocate(
        self, selected: list[tuple[int, PopulationAction, int]], staging: _FenceStaging,
        context: PopulationPlanAdapterContext, layout: _PayloadLayout,
    ) -> PopulationPlanBuildResult:
        donor_rows = [row for row, _, _ in selected]
        target_ids = tuple(target_id for _, _, target_id in selected)
        payload, reason = self._target_payload(
            staging, expected_count=len(donor_rows), expected_target_ids=target_ids,
            context=context, layout=layout,
        )
        if reason is not None:
            return self._drop(reason, staging.telemetry)
        assert payload is not None
        donor_tensor = torch.tensor(donor_rows, dtype=torch.int64)
        survivors = torch.tensor(
            [index for index in range(context.population) if index not in set(donor_rows)], dtype=torch.int64,
        )
        target_rows = torch.arange(survivors.numel(), context.population, dtype=torch.int64)
        donor_uids = tuple(context.hypothesis_uids[row] for row in donor_rows)
        relocated_uids = self._fresh_uids(context, len(donor_rows))
        transform = PopulationRowTransform(
            old_population=context.population, new_population=context.population,
            survivor_old_indices=survivors, retired_old_indices=donor_tensor,
            created_rows=(),
            relocated_rows=tuple(
                PopulationRowRelocation(
                    row_index=int(target), donor_row_index=int(donor),
                    old_hypothesis_uid=old_uid, new_hypothesis_uid=new_uid,
                )
                for target, donor, old_uid, new_uid in zip(
                    target_rows.tolist(), donor_rows, donor_uids, relocated_uids,
                )
            ),
            covered_fields=context.row_state_registry.names,
        )
        return self._proposal(
            context=context, row_transform=transform, new_population=context.population,
            retire_rows=torch.empty(0, dtype=torch.int64), relocate_donor_rows=donor_tensor,
            relocate_target_rows=target_rows, create_rows=torch.empty(0, dtype=torch.int64),
            donor_uids=donor_uids, relocated_uids=relocated_uids, created_uids=(), payload=payload,
            fence=staging.telemetry,
            target_validated=True,
        )

    def _build_retire(
        self, selected: list[tuple[int, PopulationAction, int]], staging: _FenceStaging,
        context: PopulationPlanAdapterContext, layout: _PayloadLayout,
    ) -> PopulationPlanBuildResult:
        retire_rows_list = [row for row, _, _ in selected]
        # A generic production binding may carry the provider's bounded batch
        # through the one attributed fence even when the authenticated policy
        # action is RETIRE.  The decision kept this row's target ID at -1, so
        # those staged geometry tensors were never bound to the retirement and
        # must not turn into an ambiguous payload or a second action decision.
        # An explicitly RETIRE-bound target remains rejected before staging.
        payload = self._empty_payload(layout)
        retire_rows = torch.tensor(retire_rows_list, dtype=torch.int64)
        survivors = torch.tensor(
            [index for index in range(context.population) if index not in set(retire_rows_list)], dtype=torch.int64,
        )
        survivor_uids = tuple(context.hypothesis_uids[index] for index in survivors.tolist())
        transform = PopulationRowTransform(
            old_population=context.population, new_population=len(survivor_uids),
            survivor_old_indices=survivors, retired_old_indices=retire_rows,
            created_rows=(), relocated_rows=(), covered_fields=context.row_state_registry.names,
        )
        return self._proposal(
            context=context, row_transform=transform, new_population=len(survivor_uids),
            retire_rows=retire_rows, relocate_donor_rows=torch.empty(0, dtype=torch.int64),
            relocate_target_rows=torch.empty(0, dtype=torch.int64), create_rows=torch.empty(0, dtype=torch.int64),
            donor_uids=(), relocated_uids=(), created_uids=(), payload=payload,
            fence=staging.telemetry,
        )

    def _proposal(
        self,
        *,
        context: PopulationPlanAdapterContext,
        row_transform: PopulationRowTransform,
        new_population: int,
        retire_rows: torch.Tensor,
        relocate_donor_rows: torch.Tensor,
        relocate_target_rows: torch.Tensor,
        create_rows: torch.Tensor,
        donor_uids: tuple[str, ...],
        relocated_uids: tuple[str, ...],
        created_uids: tuple[str, ...],
        payload: PopulationTargetPayloadBatch,
        fence: PopulationMutationFenceTelemetry,
        target_validated: bool = False,
    ) -> PopulationPlanBuildResult:
        survivor_uids = tuple(context.hypothesis_uids[index] for index in row_transform.survivor_old_indices.tolist())
        next_snapshot = EvidenceSnapshotToken(
            map_content=context.current_snapshot.map_content + 1,
            topology=context.current_snapshot.topology + 1,
            coordinate_frame=context.current_snapshot.coordinate_frame,
            schema=context.current_snapshot.schema,
            optimizer_update=context.current_snapshot.optimizer_update + 1,
            surface_evidence=context.current_snapshot.surface_evidence,
            policy=context.current_snapshot.policy,
            parameter=context.current_snapshot.parameter + 1,
        )
        try:
            plan = PopulationMutationPlan(
                snapshot_token=context.current_snapshot, next_snapshot_token=next_snapshot,
                row_transform=row_transform, survivor_hypothesis_uids=survivor_uids,
                mutation_fence_reason="MUTATION_CARDINALITY",
                mutation_fence_snapshot=context.current_snapshot, capacity=context.capacity,
                old_population=context.population, new_population=new_population,
                retire_rows=retire_rows, relocate_donor_rows=relocate_donor_rows,
                relocate_target_rows=relocate_target_rows, create_rows=create_rows,
                target_payload=payload.positions, donor_hypothesis_uids=donor_uids,
                relocated_hypothesis_uids=relocated_uids, created_hypothesis_uids=created_uids,
                reset_fields=frozenset(PopulationMutationPlan.REQUIRED_RESET_FIELDS),
                max_actions=context.max_actions,
                row_state_registry_snapshot=PopulationMutationPlan.snapshot_registry(context.row_state_registry),
            )
            return PopulationPlanBuildResult(
                proposal=PopulationMutationProposal(
                    key=context.proposal_key, plan=plan, target_payload=payload,
                    row_state_registry_snapshot=plan.row_state_registry_snapshot,
                ),
                reason=PopulationPlanDropReason.READY,
                fence=fence,
                target_attestation=(
                    SurfaceTargetQualificationAttestation(
                        snapshot=context.current_snapshot,
                        parameter_version=context.current_snapshot.parameter,
                        qualified=True,
                    )
                    if target_validated
                    else None
                ),
            )
        except (TypeError, ValueError):
            return self._drop(PopulationPlanDropReason.INVALID_REGISTRY, fence)

    @staticmethod
    def _stage_at_fence(
        *,
        selection: PopulationStructuralSelection | None,
        actions: torch.Tensor,
        target_indices: torch.Tensor,
        targets: SurfaceTargetBatch | None,
        snapshot: EvidenceSnapshotToken,
        transfer_owner: DeviceTransferOwner,
    ) -> _FenceStaging | None:
        """Pack all bounded control/target fields and issue exactly one H2D/D2H call."""

        if not isinstance(selection, PopulationStructuralSelection):
            return None
        source = selection.source_rows
        if (
            not isinstance(actions, torch.Tensor)
            or not isinstance(target_indices, torch.Tensor)
            or actions.ndim != 1
            or target_indices.shape != actions.shape
            or actions.device != source.device
            or target_indices.device != source.device
        ):
            return None
        if actions.numel() == 0 and selection.max_actions:
            return None
        if selection._binding_action_consistency is None:
            return None
        if selection.max_actions:
            safe_rows = selection.source_rows.clamp(min=0, max=actions.shape[0] - 1)
            source_in_range = (selection.source_rows >= 0) & (selection.source_rows < actions.shape[0])
            expected_actions = actions.index_select(0, safe_rows)
            expected_targets = target_indices.index_select(0, safe_rows)
            structural = (
                (selection.action_codes == PopulationAction.CREATE.code)
                | (selection.action_codes == PopulationAction.RELOCATE.code)
                | (selection.action_codes == PopulationAction.RETIRE.code)
            )
            target_matches = torch.where(
                selection.action_codes == PopulationAction.RETIRE.code,
                selection.surface_target_ids == -1,
                expected_targets == selection.surface_target_ids,
            )
            selection_consistency = (
                (~selection.valid_mask)
                | (
                    source_in_range
                    & structural
                    & (expected_actions == selection.action_codes)
                    & target_matches
                    & (selection.source_rows == selection._attested_source_rows)
                    & (selection.action_codes == selection._attested_action_codes)
                    & (selection.surface_target_ids == selection._attested_target_ids)
                )
            )
        else:
            selection_consistency = torch.empty((0,), dtype=torch.bool, device=source.device)
        fields: list[tuple[str, torch.Tensor]] = [
            ("source_rows", selection.source_rows),
            ("action_codes", selection.action_codes),
            ("selection_target_ids", selection.surface_target_ids),
            ("valid_mask", selection.valid_mask),
            ("selection_consistency", selection_consistency),
            ("binding_action_consistency", selection._binding_action_consistency),
        ]
        if targets is not None:
            fields.extend((
                ("positions", targets.positions),
                ("normals", targets.normals),
                ("uncertainty", targets.uncertainty),
                ("support_evidence", targets.support_evidence),
                ("conflict_evidence", targets.conflict_evidence),
                ("independent_view_count", targets.independent_view_count),
                ("target_ids", targets.target_ids),
            ))
        if any(value.device != source.device for _, value in fields):
            return None
        packed_parts: list[torch.Tensor] = []
        layouts: list[_PackedField] = []
        offset = 0
        for name, value in fields:
            contiguous = value.detach().contiguous()
            raw = contiguous.view(torch.uint8).reshape(-1)
            padding = (-offset) % 8
            if padding:
                packed_parts.append(torch.zeros((padding,), dtype=torch.uint8, device=source.device))
                offset += padding
            layouts.append(_PackedField(name, offset, tuple(contiguous.shape), contiguous.dtype))
            packed_parts.append(raw)
            offset += raw.numel()
        packed_device = torch.cat(packed_parts) if packed_parts else torch.empty(
            (0,), dtype=torch.uint8, device=source.device,
        )
        # The only normal-path device-to-host call in this adapter.  All
        # fields below are typed views of this one bounded transfer slab.  A
        # live composition must route it through its existing transfer owner
        # so the event and byte/op accounting remain globally attributable.
        if transfer_owner.device != source.device:
            return None
        owner_before = transfer_owner.stats.snapshot()["d2h_ops"]
        readback = transfer_owner.readback_mutation_cardinality(packed_device)
        # This is the explicitly attributed cardinality fence, not a routine
        # synchronization.  The owner supplies the bounded byte slab, event,
        # and accounting; direct CPU staging is forbidden.
        readback.wait()
        owner_after = transfer_owner.stats.snapshot()["d2h_ops"]
        return _FenceStaging(
            backing=readback.host,
            fields=tuple(layouts),
            telemetry=PopulationMutationFenceTelemetry(
                reason="MUTATION_CARDINALITY", snapshot=snapshot,
                source_device=source.device, packed_bytes=int(readback.host.numel()), transfer_calls=1,
                transfer_owner_d2h_ops_before=owner_before,
                transfer_owner_d2h_ops_after=owner_after,
            ),
            readback=readback,
        )

    @staticmethod
    def _selected_actions_at_fence(
        staging: _FenceStaging,
        population: int,
        max_actions: int,
    ) -> list[tuple[int, PopulationAction, int]] | None:
        try:
            if not bool(torch.all(staging.value("selection_consistency"))):
                return None
            if not bool(torch.all(staging.value("binding_action_consistency"))):
                return None
            rows = staging.value("source_rows").to(dtype=torch.int64).tolist()
            actions = staging.value("action_codes").to(dtype=torch.int64).tolist()
            surface_ids = staging.value("selection_target_ids").to(dtype=torch.int64).tolist()
            valid = staging.value("valid_mask").to(dtype=torch.bool).tolist()
            if len(rows) > max_actions:
                return None
            selected = [
                (row, PopulationAction(action), target_id)
                for row, action, target_id, keep in zip(rows, actions, surface_ids, valid)
                if keep
            ]
            if any(row < 0 or row >= population for row, _, _ in selected):
                return None
            if len({row for row, _, _ in selected}) != len(selected):
                return None
            if any(action not in PopulationPolicyPlanAdapter._STRUCTURAL for _, action, _ in selected):
                return None
            if any(target_id < 0 for _, action, target_id in selected if action is not PopulationAction.RETIRE):
                return None
            if any(target_id != -1 for _, action, target_id in selected if action is PopulationAction.RETIRE):
                return None
            return selected
        except (ValueError, RuntimeError):
            return None

    @staticmethod
    def _surface_aligned_quaternions(normals: torch.Tensor) -> torch.Tensor:
        """Return unit quaternions rotating +Z onto validated surface normals."""

        lengths = torch.linalg.vector_norm(normals, dim=1)
        if bool((torch.abs(lengths - 1.0) > 1e-3).any()):
            raise ValueError("surface normals must be unit length at the mutation fence")
        unit = normals / lengths.unsqueeze(1)
        quaternions = torch.zeros((unit.shape[0], 4), dtype=unit.dtype)
        # q = [1 + z.n, z x n], normalized.  The antiparallel case uses an
        # explicit 180-degree rotation about +X.
        quaternions[:, 0] = 1.0 + unit[:, 2]
        quaternions[:, 1] = -unit[:, 1]
        quaternions[:, 2] = unit[:, 0]
        antiparallel = unit[:, 2] < (-1.0 + 1e-6)
        quaternions[antiparallel] = torch.tensor(
            [0.0, 1.0, 0.0, 0.0], dtype=unit.dtype,
        )
        quaternions = quaternions / torch.linalg.vector_norm(quaternions, dim=1).unsqueeze(1)
        return quaternions

    @staticmethod
    def _valid_uid_inventory(context: PopulationPlanAdapterContext) -> bool:
        uids = context.hypothesis_uids
        if any(not isinstance(uid, str) or not uid or not uid.isdecimal() or str(int(uid)) != uid for uid in uids):
            return False
        parsed = tuple(int(uid) for uid in uids)
        return len(set(parsed)) == len(parsed) and (not parsed or context.next_hypothesis_uid > max(parsed))

    @staticmethod
    def _fresh_uids(context: PopulationPlanAdapterContext, count: int) -> tuple[str, ...]:
        return tuple(str(context.next_hypothesis_uid + offset) for offset in range(count))

    @staticmethod
    def _payload_layout(registry: GaussianRowStateRegistry) -> _PayloadLayout | None:
        descriptors = {descriptor.name: descriptor for descriptor in registry.descriptors}
        required = {
            "params.means": ((3,),),
            "params.scales": ((3,),),
            "params.quats": ((4,),),
            "params.opacities": ((),),
            "params.sh0": ((1, 3),),
            "params.shN": None,
        }
        if not all(name in descriptors for name in required):
            return None
        for name, suffixes in required.items():
            descriptor = descriptors[name]
            if descriptor.family != "params" or not isinstance(descriptor.dtype, torch.dtype):
                return None
            if suffixes is not None and descriptor.shape_suffix not in suffixes:
                return None
        shn = descriptors["params.shN"]
        if len(shn.shape_suffix) < 2:
            return None
        return _PayloadLayout(
            means=descriptors["params.means"], scales=descriptors["params.scales"],
            quats=descriptors["params.quats"], opacities=descriptors["params.opacities"],
            sh0=descriptors["params.sh0"], shn=shn,
        )


__all__ = [
    "PopulationPlanAdapterContext", "PopulationPlanBuildResult",
    "PopulationPlanDropReason", "PopulationPolicyPlanAdapter",
]
