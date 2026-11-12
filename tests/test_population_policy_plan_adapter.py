"""CPU-Torch contract tests for the M5 decision-to-plan boundary."""

from __future__ import annotations

from dataclasses import replace
import inspect

import pytest
import torch

from eligsir.map.arbiter import ProposalKey
from eligsir.map.row_state import GaussianRowStateRegistry, RowStateDescriptor
from eligsir.mapping.evidence.evidence_demand import EvidenceDemandBatch
from eligsir.mapping.evidence.surface_targets import (
    SurfaceTargetBatch,
    SurfaceTargetSnapshot,
)
from eligsir.mapping.population_plan_adapter import (
    PopulationPlanAdapterContext,
    PopulationPlanDropReason,
    PopulationPolicyPlanAdapter,
)
from eligsir.mapping.population_policy import (
    GaussianPopulationPolicy,
    PopulationAction,
    PopulationDecision,
)
from eligsir.runtime.transfers import DeviceTransferOwner, PackedReadback
from eligsir.optimization.gaussian.supervision_evidence import EvidenceSnapshotToken


def _token(**overrides) -> EvidenceSnapshotToken:
    values = dict(
        map_content=4, topology=5, coordinate_frame=6, schema=7,
        optimizer_update=8, surface_evidence=9, policy=10, parameter=11,
    )
    values.update(overrides)
    return EvidenceSnapshotToken(**values)


def _registry() -> GaussianRowStateRegistry:
    fields = (
        ("params.means", "params", torch.float32, (3,)),
        ("params.scales", "params", torch.float32, (3,)),
        ("params.quats", "params", torch.float32, (4,)),
        ("params.opacities", "params", torch.float32, ()),
        ("params.sh0", "params", torch.float32, (1, 3)),
        ("params.shN", "params", torch.float32, (0, 3)),
        ("optimizer.means.exp_avg", "optimizer", torch.float32, (3,)),
        ("belief.p_exist", "belief", torch.float32, ()),
        ("provenance.gaussian_uid", "provenance", torch.int64, ()),
        ("surface.anchor_xyz", "surface", torch.float32, (3,)),
    )
    return GaussianRowStateRegistry(
        RowStateDescriptor(
            name=name, family=family, owner="test", shape_suffix=suffix, dtype=dtype,
            checkpoint=True, default="zero", create="zero", grow="zero",
            relocate="zero", retire="select_survivors",
        )
        for name, family, dtype, suffix in fields
    )


def _context(
    *,
    token: EvidenceSnapshotToken | None = None,
    decision_snapshot: EvidenceSnapshotToken | None = None,
    capacity: int = 4,
    uids: tuple[str, ...] = ("10", "11"),
    next_uid: int = 100,
    transfer_owner: DeviceTransferOwner | None = None,
) -> PopulationPlanAdapterContext:
    token = _token() if token is None else token
    return PopulationPlanAdapterContext(
        current_snapshot=token,
        decision_snapshot=token if decision_snapshot is None else decision_snapshot,
        capacity=capacity,
        hypothesis_uids=uids,
        next_hypothesis_uid=next_uid,
        row_state_registry=_registry(),
        proposal_key=ProposalKey(3, 4, 5, "population-policy"),
        max_actions=2,
        transfer_owner=DeviceTransferOwner("cpu") if transfer_owner is None else transfer_owner,
    )


def _targets(
    *,
    count: int = 2,
    snapshot: SurfaceTargetSnapshot | None = None,
    target_ids: torch.Tensor | None = None,
    normals: torch.Tensor | None = None,
    independent_views: int = 2,
) -> SurfaceTargetBatch:
    token = _token()
    snapshot = snapshot or SurfaceTargetSnapshot(
        map_content=token.map_content, topology=token.topology,
        coordinate_frame=token.coordinate_frame, schema=token.schema,
        surface_evidence=token.surface_evidence,
    )
    positions = torch.arange(count * 3, dtype=torch.float32).reshape(count, 3)
    return SurfaceTargetBatch(
        snapshot=snapshot,
        positions=positions,
        normals=torch.tensor([[0.0, 0.0, 1.0]]).repeat(count, 1) if normals is None else normals,
        uncertainty=torch.full((count, 3), 0.02),
        support_evidence=torch.ones(count), conflict_evidence=torch.zeros(count),
        independent_view_count=torch.full((count,), independent_views, dtype=torch.int32),
        repeated_exposure_count=torch.ones(count, dtype=torch.int32),
        target_ids=torch.arange(200, 200 + count, dtype=torch.int64) if target_ids is None else target_ids,
        source_modality=torch.full((count,), 4, dtype=torch.int8), max_targets=max(1, count),
    )


def _decision(
    *actions: PopulationAction,
    selected_target_ids: tuple[int, ...] = (),
    selected_rows: tuple[int, ...] | None = None,
) -> PopulationDecision:
    structural_rows = tuple(
        index
        for index, action in enumerate(actions)
        if action in {PopulationAction.CREATE, PopulationAction.RELOCATE, PopulationAction.RETIRE}
    )
    if selected_rows is None:
        selected_rows = structural_rows
    selected_actions = tuple(actions[row] for row in selected_rows)
    if not selected_target_ids:
        selected_target_ids = tuple(-1 if action is PopulationAction.RETIRE else 200 + index for index, action in enumerate(selected_actions))
    action_tensor = torch.tensor([action.code for action in actions], dtype=torch.int8)
    target_indices = torch.full((len(actions),), -1, dtype=torch.int64)
    scores = torch.zeros(len(actions), dtype=torch.float32)
    for ordinal, row in enumerate(selected_rows):
        scores[row] = float(len(selected_rows) - ordinal)
    selection = GaussianPopulationPolicy._structural_selection(
        action_tensor, scores, target_indices,
        max_actions=len(selected_rows),
    )
    decision = PopulationDecision.policy_output(
        actions=action_tensor,
        target_indices=target_indices,
        evidence_demands=EvidenceDemandBatch(()), demand_masks=(), structural_selection=selection,
    )
    if any(target_id >= 0 for target_id in selected_target_ids) and len(set(selected_actions)) == 1:
        decision = decision.bind_surface_target_ids(
            torch.tensor(selected_target_ids, dtype=torch.int64),
            target_snapshot=_targets(count=max(1, len(selected_target_ids))).snapshot,
            structural_action=selected_actions[0],
        )
    return decision


def test_create_decision_builds_one_complete_atomic_proposal_from_exact_targets():
    result = PopulationPolicyPlanAdapter().build(
        decision=_decision(PopulationAction.CREATE, PopulationAction.CREATE),
        targets=_targets(count=2), context=_context(),
    )

    assert result.reason is PopulationPlanDropReason.READY
    assert result.proposal is not None
    plan = result.proposal.plan
    assert plan.create_rows.tolist() == [2, 3]
    assert plan.relocate_donor_rows.numel() == 0
    assert plan.created_hypothesis_uids == ("100", "101")
    assert plan.next_snapshot_token == _token(
        map_content=5, topology=6, optimizer_update=9, parameter=12,
    )
    assert plan.row_transform.covered_fields == _registry().names
    assert result.proposal.target_payload.count == 2
    assert result.proposal.target_payload.positions.device.type == "cpu"
    assert torch.equal(result.proposal.target_payload.positions, _targets(count=2).positions)
    assert result.fence is not None
    assert result.fence.reason == "MUTATION_CARDINALITY"
    assert result.fence.transfer_calls == 1
    assert result.fence.packed_bytes > 0
    assert result.target_attestation is not None
    assert result.target_attestation.qualified is True
    assert result.target_attestation.snapshot == _token()


def test_capacity_relocation_pairs_each_donor_with_one_current_target_and_fresh_identity():
    result = PopulationPolicyPlanAdapter().build(
        decision=_decision(PopulationAction.RELOCATE, PopulationAction.NONE),
        targets=_targets(count=1), context=_context(capacity=2),
    )

    assert result.reason is PopulationPlanDropReason.READY
    assert result.proposal is not None
    plan = result.proposal.plan
    assert plan.old_population == plan.new_population == 2
    assert plan.relocate_donor_rows.tolist() == [0]
    assert plan.relocate_target_rows.tolist() == [1]
    assert plan.donor_hypothesis_uids == ("10",)
    assert plan.relocated_hypothesis_uids == ("100",)
    assert plan.retire_rows.numel() == plan.create_rows.numel() == 0
    assert result.target_attestation is not None
    assert result.target_attestation.qualified is True


def test_relocation_n4_k1_places_the_target_in_the_single_final_tail_row():
    result = PopulationPolicyPlanAdapter().build(
        decision=_decision(
            PopulationAction.NONE, PopulationAction.RELOCATE,
            PopulationAction.NONE, PopulationAction.NONE,
        ),
        targets=_targets(count=1),
        context=_context(capacity=4, uids=("10", "11", "12", "13")),
    )

    assert result.proposal is not None
    plan = result.proposal.plan
    assert plan.row_transform.survivor_old_indices.tolist() == [0, 2, 3]
    assert plan.relocate_target_rows.tolist() == [3]


def test_relocation_n5_k2_places_both_targets_in_the_final_two_tail_rows():
    result = PopulationPolicyPlanAdapter().build(
        decision=_decision(
            PopulationAction.NONE, PopulationAction.RELOCATE,
            PopulationAction.NONE, PopulationAction.NONE, PopulationAction.RELOCATE,
        ),
        targets=_targets(count=2),
        context=_context(capacity=5, uids=("10", "11", "12", "13", "14")),
    )

    assert result.proposal is not None
    plan = result.proposal.plan
    assert plan.row_transform.survivor_old_indices.tolist() == [0, 2, 3]
    assert plan.relocate_target_rows.tolist() == [3, 4]


def test_none_and_abstain_do_not_construct_a_population_proposal_or_require_targets():
    result = PopulationPolicyPlanAdapter().build(
        decision=_decision(PopulationAction.NONE, PopulationAction.ABSTAIN),
        targets=None, context=_context(),
    )

    assert result.proposal is None
    assert result.reason is PopulationPlanDropReason.NO_STRUCTURAL_ACTION


def test_stale_surface_targets_are_rejected_before_any_plan_is_constructed():
    stale = _targets(snapshot=SurfaceTargetSnapshot(3, 5, 6, 7, 9))
    result = PopulationPolicyPlanAdapter().build(
        decision=_decision(PopulationAction.CREATE, PopulationAction.NONE),
        targets=stale, context=_context(),
    )

    assert result.proposal is None
    assert result.reason is PopulationPlanDropReason.STALE_TARGETS


def test_one_view_surface_target_is_rejected_at_the_single_mutation_cardinality_fence():
    owner = DeviceTransferOwner("cpu")
    before = owner.accounting()
    result = PopulationPolicyPlanAdapter().build(
        decision=_decision(PopulationAction.CREATE, PopulationAction.NONE),
        targets=_targets(count=1, independent_views=1),
        context=_context(transfer_owner=owner),
    )

    after = owner.accounting()
    assert result.proposal is None
    assert result.reason is PopulationPlanDropReason.INVALID_TARGET_PAYLOAD
    assert result.fence is not None
    assert result.fence.transfer_calls == 1
    assert after.d2h_ops == before.d2h_ops + 1
    assert after.h2d_ops == before.h2d_ops


def test_insufficient_or_ambiguous_target_payload_is_rejected_instead_of_guessed():
    adapter = PopulationPolicyPlanAdapter()
    insufficient = adapter.build(
        decision=_decision(PopulationAction.CREATE, PopulationAction.CREATE),
        targets=_targets(count=1), context=_context(),
    )
    ambiguous = adapter.build(
        decision=_decision(PopulationAction.CREATE, PopulationAction.NONE),
        targets=_targets(count=2), context=_context(),
    )

    assert insufficient.reason is PopulationPlanDropReason.AMBIGUOUS_TARGET_PAYLOAD
    assert ambiguous.reason is PopulationPlanDropReason.AMBIGUOUS_TARGET_PAYLOAD
    assert insufficient.proposal is ambiguous.proposal is None


def test_target_pairing_requires_exact_surface_ids_in_exact_selected_order():
    decision = _decision(
        PopulationAction.CREATE, PopulationAction.CREATE,
        selected_target_ids=(200, 201),
    )
    reordered = _targets(count=2, target_ids=torch.tensor([201, 200], dtype=torch.int64))

    result = PopulationPolicyPlanAdapter().build(
        decision=decision, targets=reordered, context=_context(),
    )

    assert result.proposal is None
    assert result.reason is PopulationPlanDropReason.TARGET_PAIR_MISMATCH


def test_payload_quaternion_is_surface_aligned_and_rejects_nonunit_normals():
    normal = torch.tensor([[0.0, 1.0, 0.0]])
    aligned = PopulationPolicyPlanAdapter().build(
        decision=_decision(PopulationAction.CREATE, PopulationAction.NONE),
        targets=_targets(count=1, normals=normal), context=_context(),
    )
    invalid = PopulationPolicyPlanAdapter().build(
        decision=_decision(PopulationAction.CREATE, PopulationAction.NONE),
        targets=_targets(count=1, normals=torch.tensor([[0.0, 3.0, 0.0]])),
        context=_context(),
    )

    assert aligned.proposal is not None
    assert torch.allclose(
        aligned.proposal.target_payload.quaternions,
        torch.tensor([[2**-0.5, -(2**-0.5), 0.0, 0.0]]),
    )
    assert invalid.proposal is None
    assert invalid.reason is PopulationPlanDropReason.INVALID_TARGET_PAYLOAD


def test_mixed_unpaired_structural_actions_without_bound_create_target_are_rejected():
    result = PopulationPolicyPlanAdapter().build(
        decision=_decision(PopulationAction.CREATE, PopulationAction.RETIRE),
        targets=None, context=_context(),
    )

    assert result.proposal is None
    assert result.reason is PopulationPlanDropReason.INVALID_DECISION


def test_create_at_capacity_and_stale_decision_each_fail_closed():
    adapter = PopulationPolicyPlanAdapter()
    at_capacity = adapter.build(
        decision=_decision(PopulationAction.CREATE, PopulationAction.NONE),
        targets=_targets(count=1), context=_context(capacity=2),
    )
    stale_decision = adapter.build(
        decision=_decision(PopulationAction.CREATE, PopulationAction.NONE),
        targets=_targets(count=1),
        context=_context(decision_snapshot=_token(parameter=10)),
    )

    assert at_capacity.reason is PopulationPlanDropReason.CAPACITY_CONFLICT
    assert stale_decision.reason is PopulationPlanDropReason.STALE_DECISION
    assert at_capacity.proposal is stale_decision.proposal is None


def test_invalid_or_non_monotonic_uid_inventory_fails_closed():
    result = PopulationPolicyPlanAdapter().build(
        decision=_decision(PopulationAction.CREATE, PopulationAction.NONE),
        targets=_targets(count=1), context=_context(uids=("10", "10")),
    )
    non_monotonic = PopulationPolicyPlanAdapter().build(
        decision=_decision(PopulationAction.CREATE, PopulationAction.NONE),
        targets=_targets(count=1), context=_context(next_uid=10),
    )

    assert result.reason is PopulationPlanDropReason.INVALID_UID_INVENTORY
    assert non_monotonic.reason is PopulationPlanDropReason.INVALID_UID_INVENTORY


def test_missing_bounded_structural_selection_cannot_cross_the_fence():
    decision = PopulationDecision(
        actions=torch.tensor([PopulationAction.CREATE.code, PopulationAction.NONE.code], dtype=torch.int8),
        target_indices=torch.full((2,), -1, dtype=torch.int64),
        evidence_demands=EvidenceDemandBatch(()), demand_masks=(),
    )

    result = PopulationPolicyPlanAdapter().build(
        decision=decision, targets=_targets(count=1), context=_context(),
    )

    assert result.proposal is None
    assert result.reason is PopulationPlanDropReason.INVALID_DECISION


def test_replaced_selection_breaks_private_policy_attestation_and_is_rejected():
    decision = _decision(PopulationAction.CREATE, PopulationAction.NONE)
    assert decision.structural_selection is not None
    forged = replace(
        decision,
        structural_selection=decision.structural_selection.with_surface_target_ids(
            torch.tensor([999], dtype=torch.int64),
            target_snapshot=_targets(count=1).snapshot,
            target_binding_action=PopulationAction.CREATE,
        ),
    )

    result = PopulationPolicyPlanAdapter().build(
        decision=forged, targets=_targets(count=1), context=_context(),
    )

    assert result.proposal is None
    assert result.reason is PopulationPlanDropReason.INVALID_DECISION


def test_policy_output_rejects_a_selection_attested_to_a_different_action_tensor():
    action_tensor = torch.tensor([PopulationAction.CREATE.code], dtype=torch.int8)
    selection = GaussianPopulationPolicy._structural_selection(
        action_tensor, torch.ones(1), torch.full((1,), -1, dtype=torch.int64), max_actions=1,
    )

    with pytest.raises(ValueError, match="not produced"):
        PopulationDecision.policy_output(
            actions=action_tensor.clone(),
            target_indices=torch.full((1,), -1, dtype=torch.int64),
            evidence_demands=EvidenceDemandBatch(()), demand_masks=(),
            structural_selection=selection,
        )


def test_nonfinite_selected_target_payload_is_rejected_after_the_bounded_fence_copy():
    targets = _targets(count=1)
    targets.positions[0, 0] = float("nan")

    result = PopulationPolicyPlanAdapter().build(
        decision=_decision(PopulationAction.CREATE, PopulationAction.NONE),
        targets=targets, context=_context(),
    )

    assert result.proposal is None
    assert result.reason is PopulationPlanDropReason.INVALID_TARGET_PAYLOAD
    assert result.fence is not None


def test_retire_only_builds_an_empty_target_payload_without_claiming_a_relocation():
    result = PopulationPolicyPlanAdapter().build(
        decision=_decision(PopulationAction.RETIRE, PopulationAction.NONE),
        targets=None, context=_context(),
    )

    assert result.reason is PopulationPlanDropReason.READY
    assert result.proposal is not None
    assert result.proposal.plan.retire_rows.tolist() == [0]
    assert result.proposal.plan.relocate_donor_rows.numel() == 0
    assert result.proposal.target_payload.count == 0


def test_arbitrary_surface_binding_keeps_retire_targetless_after_the_fence():
    targets = _targets(count=1)
    decision = _decision(PopulationAction.RETIRE, PopulationAction.NONE)
    decision = decision.bind_surface_target_ids(
        targets.target_ids,
        target_snapshot=targets.snapshot,
        structural_action=None,
    )

    result = PopulationPolicyPlanAdapter().build(
        decision=decision, targets=targets, context=_context(),
    )

    assert result.reason is PopulationPlanDropReason.READY
    assert result.proposal is not None
    assert result.proposal.plan.retire_rows.tolist() == [0]
    assert result.proposal.target_payload.count == 0


def test_bound_target_snapshot_mismatch_is_rejected_before_target_staging():
    decision = _decision(PopulationAction.CREATE, PopulationAction.NONE)
    stale_binding = SurfaceTargetSnapshot(3, 5, 6, 7, 9)
    decision = decision.bind_surface_target_ids(
        torch.tensor([200], dtype=torch.int64), target_snapshot=stale_binding,
        structural_action=PopulationAction.CREATE,
    )
    result = PopulationPolicyPlanAdapter().build(
        decision=decision, targets=_targets(count=1), context=_context(),
    )
    assert result.reason is PopulationPlanDropReason.TARGET_PAIR_MISMATCH
    assert result.fence is None


@pytest.mark.parametrize("mutation", ("action", "source_row", "action_code", "target_id"))
def test_in_place_selection_or_action_mutation_fails_device_consistency_at_the_fence(mutation: str):
    decision = _decision(PopulationAction.CREATE, PopulationAction.NONE)
    assert decision.structural_selection is not None
    if mutation == "action":
        decision.actions[0] = PopulationAction.NONE.code
    elif mutation == "source_row":
        decision.structural_selection.source_rows[0] = 1
    elif mutation == "action_code":
        decision.structural_selection.action_codes[0] = PopulationAction.RETIRE.code
    else:
        decision.structural_selection.surface_target_ids[0] = 201
    result = PopulationPolicyPlanAdapter().build(
        decision=decision, targets=_targets(count=1), context=_context(),
    )
    assert result.reason is PopulationPlanDropReason.INVALID_DECISION
    assert result.fence is not None
    assert result.fence.transfer_calls == 1


def test_oversized_target_batch_and_retire_target_payload_reject_before_staging():
    oversized = PopulationPolicyPlanAdapter().build(
        decision=_decision(PopulationAction.CREATE, PopulationAction.NONE),
        targets=_targets(count=3), context=_context(),
    )
    retire_with_target = PopulationPolicyPlanAdapter().build(
        decision=_decision(PopulationAction.RETIRE, PopulationAction.NONE),
        targets=_targets(count=1), context=_context(),
    )
    assert oversized.reason is PopulationPlanDropReason.AMBIGUOUS_TARGET_PAYLOAD
    assert retire_with_target.reason is PopulationPlanDropReason.TARGET_PAIR_MISMATCH
    assert oversized.fence is retire_with_target.fence is None


def test_authenticated_retire_target_binding_rejects_any_target_batch_before_staging():
    decision = _decision(PopulationAction.RETIRE, PopulationAction.NONE)
    decision = decision.bind_surface_target_ids(
        torch.tensor([200], dtype=torch.int64),
        target_snapshot=_targets(count=1).snapshot,
        structural_action=PopulationAction.RETIRE,
    )
    result = PopulationPolicyPlanAdapter().build(
        decision=decision, targets=_targets(count=1), context=_context(),
    )
    assert result.reason is PopulationPlanDropReason.TARGET_PAIR_MISMATCH
    assert result.fence is None


def test_retire_mislabeled_as_create_reaches_one_fence_then_fails_device_consistency():
    decision = _decision(PopulationAction.RETIRE, PopulationAction.NONE)
    decision = decision.bind_surface_target_ids(
        torch.tensor([200], dtype=torch.int64),
        target_snapshot=_targets(count=1).snapshot,
        structural_action=PopulationAction.CREATE,
    )
    result = PopulationPolicyPlanAdapter().build(
        decision=decision, targets=_targets(count=1), context=_context(),
    )
    assert result.reason is PopulationPlanDropReason.INVALID_DECISION
    assert result.fence is not None


def test_binding_path_has_no_python_gpu_scalar_reduction_before_mutation_fence():
    source = (
        inspect.getsource(PopulationDecision.bind_surface_target_ids)
        + inspect.getsource(GaussianPopulationPolicy._structural_selection)
    )
    assert "bool(torch.any" not in source
    assert "bool(torch.all" not in source
    assert ".item()" not in source


def test_fence_uses_the_existing_transfer_owner_for_one_packed_staging_event():
    owner = DeviceTransferOwner("cpu", max_pinned_bytes=1024 * 1024)
    result = PopulationPolicyPlanAdapter().build(
        decision=_decision(PopulationAction.CREATE, PopulationAction.NONE),
        targets=_targets(count=1), context=_context(transfer_owner=owner),
    )
    assert result.reason is PopulationPlanDropReason.READY
    assert result.fence is not None
    assert owner.stats.d2h_ops == 1
    assert result.fence.transfer_owner_d2h_ops_before == 0
    assert result.fence.transfer_owner_d2h_ops_after == 1
    assert owner.pending_count == 0


def test_coordinated_public_action_and_selection_mutation_cannot_rewrite_attested_selection():
    decision = _decision(PopulationAction.CREATE, PopulationAction.NONE)
    assert decision.structural_selection is not None
    decision.actions[0] = PopulationAction.RETIRE.code
    decision.structural_selection.action_codes[0] = PopulationAction.RETIRE.code
    decision.structural_selection.surface_target_ids[0] = -1

    result = PopulationPolicyPlanAdapter().build(
        decision=decision, targets=_targets(count=1), context=_context(),
    )

    assert result.proposal is None
    assert result.reason is PopulationPlanDropReason.INVALID_DECISION
    assert result.fence is not None


def test_prior_payload_is_independent_of_the_reused_pooled_cardinality_slab():
    owner = DeviceTransferOwner("cpu", max_inflight=1, max_pinned_bytes=1024 * 1024)
    first_targets = _targets(count=1)
    first_targets.positions.fill_(3.0)
    first = PopulationPolicyPlanAdapter().build(
        decision=_decision(PopulationAction.CREATE, PopulationAction.NONE),
        targets=first_targets, context=_context(transfer_owner=owner),
    )
    assert first.proposal is not None
    second_targets = _targets(count=1)
    second_targets.positions.fill_(9.0)
    second = PopulationPolicyPlanAdapter().build(
        decision=_decision(PopulationAction.CREATE, PopulationAction.NONE),
        targets=second_targets, context=_context(transfer_owner=owner),
    )
    assert second.proposal is not None
    assert first.proposal.target_payload.positions.tolist() == [[3.0, 3.0, 3.0]]
    assert second.proposal.target_payload.positions.tolist() == [[9.0, 9.0, 9.0]]
    assert owner.pending_count == 0


def test_packed_readback_releases_after_an_explicit_single_wait_without_rewaiting():
    class _Event:
        def __init__(self):
            self.wait_count = 0

        def wait(self):
            self.wait_count += 1

    class _Pool:
        def __init__(self):
            self.releases = 0

        def release_now(self, slot):
            self.releases += 1

    event, pool = _Event(), _Pool()
    readback = PackedReadback(torch.empty(1), event, pool, object())
    readback.wait()
    readback.release_after_wait()
    assert event.wait_count == 1
    assert pool.releases == 1
