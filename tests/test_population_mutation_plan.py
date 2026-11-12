import pytest
import torch

from eligsir.map.population_mutation_plan import (
    MutationAction,
    PopulationMutationPlan,
)
from eligsir.optimization.gaussian.supervision_evidence import EvidenceSnapshotToken
from eligsir.map.row_state import PopulationRowCreation, PopulationRowRelocation, PopulationRowTransform, RowStateDescriptor


def descriptors():
    return tuple(RowStateDescriptor(name=name, family="canonical", owner="arbiter", shape_suffix=(), dtype=torch.float32,
                                    checkpoint=True, default=0, create="fresh", grow="fresh", relocate="fresh", retire="gather")
                 for name in PopulationMutationPlan.REQUIRED_RESET_FIELDS)


def token(**overrides):
    values = dict(map_content=4, topology=7, coordinate_frame=2, schema=1,
                  optimizer_update=11, surface_evidence=3, policy=5, parameter=9)
    values.update(overrides)
    return EvidenceSnapshotToken(**values)


def transform(old, new, survivors, retired, created=(), relocated=()):
    return PopulationRowTransform(
        old_population=old, new_population=new,
        survivor_old_indices=torch.tensor(survivors, dtype=torch.int64),
        retired_old_indices=torch.tensor(retired, dtype=torch.int64),
        created_rows=tuple(PopulationRowCreation(**x) for x in created),
        relocated_rows=tuple(PopulationRowRelocation(**x) for x in relocated),
        covered_fields=frozenset(PopulationMutationPlan.REQUIRED_RESET_FIELDS),
    )


def test_plan_validates_paired_relocation_and_fresh_identity():
    plan = PopulationMutationPlan(
        snapshot_token=token(),
        next_snapshot_token=token(map_content=5, topology=8, optimizer_update=12, parameter=10),
        row_transform=transform(4, 4, [0, 1], [2, 3], created=({"row_index": 3, "hypothesis_uid": "fresh-create"},), relocated=({"row_index": 2, "old_hypothesis_uid": "old-donor", "new_hypothesis_uid": "fresh-relocation", "donor_row_index": 2},)),
        survivor_hypothesis_uids=("survivor-0", "survivor-1"),
        mutation_fence_reason="MUTATION_CARDINALITY", mutation_fence_snapshot=token(),
        row_state_registry_snapshot=descriptors(),
        capacity=8,
        old_population=4,
        new_population=4,
        retire_rows=torch.tensor([3]),
        relocate_donor_rows=torch.tensor([2]),
        relocate_target_rows=torch.tensor([2]),
        create_rows=torch.tensor([3]),
        target_payload=torch.tensor([[0.0, 1.0, 2.0], [1.0, 2.0, 3.0]]),
        donor_hypothesis_uids=("old-donor",),
        relocated_hypothesis_uids=("fresh-relocation",),
        created_hypothesis_uids=("fresh-create",),
        reset_fields=frozenset(PopulationMutationPlan.REQUIRED_RESET_FIELDS),
        max_actions=4,
    )
    plan.validate(current_snapshot=token())
    assert plan.action_count == 3
    assert plan.has_action(MutationAction.RELOCATE)


def test_noop_plan_is_bounded_and_does_not_require_reset():
    plan = PopulationMutationPlan.noop(snapshot_token=token(), population=4, capacity=8)
    plan.validate(current_snapshot=token())
    assert plan.action_count == 0
    assert plan.actions == (MutationAction.NONE,)


def test_rejects_stale_snapshot_and_bad_pairing():
    kwargs = dict(
        snapshot_token=token(),
        next_snapshot_token=token(map_content=5, topology=8, optimizer_update=12, parameter=10),
        row_transform=transform(2, 2, [1], [0], relocated=({"row_index": 1, "old_hypothesis_uid": "same", "new_hypothesis_uid": "fresh", "donor_row_index": 0},)), survivor_hypothesis_uids=("s1",),
        row_state_registry_snapshot=descriptors(),
        mutation_fence_reason="MUTATION_CARDINALITY", mutation_fence_snapshot=token(),
        capacity=4, old_population=2, new_population=2,
        retire_rows=torch.tensor([], dtype=torch.int64),
        relocate_donor_rows=torch.tensor([0]), relocate_target_rows=torch.tensor([1]),
        create_rows=torch.tensor([], dtype=torch.int64), target_payload=torch.empty((1, 3)),
        donor_hypothesis_uids=("same",),
        relocated_hypothesis_uids=("fresh",), created_hypothesis_uids=(),
        reset_fields=frozenset(PopulationMutationPlan.REQUIRED_RESET_FIELDS), max_actions=2,
    )
    with pytest.raises(ValueError, match="stale|snapshot"):
        PopulationMutationPlan(**kwargs).validate(current_snapshot=token(parameter=8))
    with pytest.raises(ValueError, match="fresh"):
        PopulationMutationPlan(**{**kwargs, "relocated_hypothesis_uids": ("same",), "donor_hypothesis_uids": ("same",)})


def test_rejects_duplicate_rows_capacity_and_missing_atomic_reset_fields():
    kwargs = dict(
        snapshot_token=token(), next_snapshot_token=token(map_content=5, topology=8, optimizer_update=12, parameter=10),
        row_transform=transform(2, 3, [], [0, 1], created=({"row_index": 2, "hypothesis_uid": "x"},)), survivor_hypothesis_uids=(),
        row_state_registry_snapshot=descriptors(),
        mutation_fence_reason="MUTATION_CARDINALITY", mutation_fence_snapshot=token(),
        capacity=2, old_population=2, new_population=3,
        retire_rows=torch.tensor([0, 0]), relocate_donor_rows=torch.empty(0, dtype=torch.int64),
        relocate_target_rows=torch.empty(0, dtype=torch.int64), create_rows=torch.tensor([2]),
        target_payload=torch.ones((1, 3)), donor_hypothesis_uids=(), relocated_hypothesis_uids=(), created_hypothesis_uids=("x",),
        reset_fields=frozenset({"belief"}), max_actions=4,
    )
    with pytest.raises(ValueError, match="capacity"):
        PopulationMutationPlan(**kwargs)
    with pytest.raises(ValueError, match="unique"):
        PopulationMutationPlan(**{**kwargs, "new_population": 1, "capacity": 3,
                                  "row_transform": transform(2, 1, [1], [0]), "survivor_hypothesis_uids": ("s1",), "retire_rows": torch.tensor([0, 0]), "create_rows": torch.empty(0, dtype=torch.int64),
                                  "target_payload": torch.empty((0, 3)), "created_hypothesis_uids": (),
                                  "reset_fields": frozenset(PopulationMutationPlan.REQUIRED_RESET_FIELDS)})
    with pytest.raises(ValueError, match="reset"):
        PopulationMutationPlan(**{**kwargs, "capacity": 3, "retire_rows": torch.tensor([0, 1]), "new_population": 2,
                                  "row_transform": transform(2, 2, [0, 1], []),
                                  "create_rows": torch.empty(0, dtype=torch.int64), "target_payload": torch.empty((0, 3)),
                                  "created_hypothesis_uids": ()}).validate(current_snapshot=token())


def test_rejects_duplicate_survivor_identity_and_missing_relocation_donor_slot():
    with pytest.raises(ValueError, match="survivor hypothesis UIDs"):
        PopulationMutationPlan(
            snapshot_token=token(), next_snapshot_token=token(),
            row_transform=transform(2, 2, [0, 1], []), survivor_hypothesis_uids=("same", "same"),
            mutation_fence_reason="MUTATION_CARDINALITY", mutation_fence_snapshot=token(),
            capacity=2, old_population=2, new_population=2,
            retire_rows=torch.empty(0, dtype=torch.int64), relocate_donor_rows=torch.empty(0, dtype=torch.int64),
            relocate_target_rows=torch.empty(0, dtype=torch.int64), create_rows=torch.empty(0, dtype=torch.int64),
            target_payload=torch.empty((0, 0)), donor_hypothesis_uids=(), relocated_hypothesis_uids=(),
            created_hypothesis_uids=(), reset_fields=frozenset(), max_actions=0,
        )


@pytest.mark.parametrize("invalid_uid", ["", 7, None])
def test_rejects_invalid_survivor_identity(invalid_uid):
    with pytest.raises(ValueError, match="non-empty strings"):
        PopulationMutationPlan(
            snapshot_token=token(), next_snapshot_token=token(),
            row_transform=transform(1, 1, [0], []), survivor_hypothesis_uids=(invalid_uid,),
            mutation_fence_reason="MUTATION_CARDINALITY", mutation_fence_snapshot=token(),
            capacity=1, old_population=1, new_population=1,
            retire_rows=torch.empty(0, dtype=torch.int64), relocate_donor_rows=torch.empty(0, dtype=torch.int64),
            relocate_target_rows=torch.empty(0, dtype=torch.int64), create_rows=torch.empty(0, dtype=torch.int64),
            target_payload=torch.empty((0, 0)), donor_hypothesis_uids=(), relocated_hypothesis_uids=(),
            created_hypothesis_uids=(), reset_fields=frozenset(), max_actions=0,
        )


def test_rejects_incomplete_concrete_registry_snapshot():
    with pytest.raises(ValueError, match="concrete registry snapshot"):
        PopulationMutationPlan(
            snapshot_token=token(), next_snapshot_token=token(map_content=5, topology=8, optimizer_update=12, parameter=10),
            row_transform=transform(1, 2, [0], [], created=({"row_index": 1, "hypothesis_uid": "new"},)), survivor_hypothesis_uids=("s",),
            row_state_registry_snapshot=descriptors()[:-1], capacity=2, old_population=1, new_population=2,
            mutation_fence_reason="MUTATION_CARDINALITY", mutation_fence_snapshot=token(),
            retire_rows=torch.tensor([], dtype=torch.int64), relocate_donor_rows=torch.empty(0, dtype=torch.int64),
            relocate_target_rows=torch.empty(0, dtype=torch.int64), create_rows=torch.tensor([1]),
            target_payload=torch.ones((1, 3)), donor_hypothesis_uids=(), relocated_hypothesis_uids=(),
            created_hypothesis_uids=("new",), reset_fields=frozenset(PopulationMutationPlan.REQUIRED_RESET_FIELDS), max_actions=1,
        )

    with pytest.raises(ValueError, match="donor_row_index"):
        PopulationMutationPlan(
            snapshot_token=token(), next_snapshot_token=token(map_content=5, topology=8, optimizer_update=12, parameter=10),
            row_transform=transform(2, 2, [1], [0], relocated=({"row_index": 1, "old_hypothesis_uid": "old", "new_hypothesis_uid": "new"},)),
            survivor_hypothesis_uids=("survivor",), row_state_registry_snapshot=descriptors(), mutation_fence_reason="MUTATION_CARDINALITY", mutation_fence_snapshot=token(), capacity=2, old_population=2, new_population=2,
            retire_rows=torch.empty(0, dtype=torch.int64), relocate_donor_rows=torch.tensor([0]),
            relocate_target_rows=torch.tensor([1]), create_rows=torch.empty(0, dtype=torch.int64),
            target_payload=torch.ones((1, 3)), donor_hypothesis_uids=("old",), relocated_hypothesis_uids=("new",),
            created_hypothesis_uids=(), reset_fields=frozenset(PopulationMutationPlan.REQUIRED_RESET_FIELDS), max_actions=1,
        )
