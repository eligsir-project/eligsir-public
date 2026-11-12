import pytest
import torch

from eligsir.utils.gaussian_belief import (
    ExistenceBelief,
    GaussianBelief,
    GeometryBelief,
    ObservabilityBelief,
    ProvenanceBelief,
    RepresentationBelief,
    TemporalBelief,
)
from eligsir.mapping.population_policy import (
    GaussianPopulationPolicy,
    MutationWorkToken,
    PopulationAction,
    PopulationEvidence,
)
from eligsir.mapping.scheduler.mcmc_cadence import MCMCCadenceOwner
from eligsir.contracts.work_clocks import RuntimeWorkClockSnapshot


def _belief(n: int) -> GaussianBelief:
    x = torch.zeros(n, 3)
    return GaussianBelief(
        provenance=ProvenanceBelief(
            gaussian_uid=torch.arange(n), birth_update=torch.zeros(n, dtype=torch.int32),
            birth_method=torch.zeros(n, dtype=torch.int8), parent_uid=torch.full((n,), -1),
            root_uid=torch.arange(n), generation=torch.zeros(n, dtype=torch.int32),
            birth_anchor_xyz=x,
        ),
        observability=ObservabilityBelief(
            meaningful_exposure_count=torch.full((n,), 3, dtype=torch.int32),
            independent_support_views=torch.full((n,), 2, dtype=torch.int16),
            baseline_diversity=torch.ones(n), highest_supervised_tier=None,
            last_exposed_update=torch.zeros(n, dtype=torch.int32), last_supported_update=None,
        ),
        existence=ExistenceBelief(
            support_evidence=torch.ones(n), contradiction_evidence=torch.zeros(n),
            p_exist=torch.ones(n),
        ),
        geometry=GeometryBelief(
            surface_anchor_id=torch.full((n,), -1), p_surface_consistent=torch.ones(n),
            surface_position_uncertainty=torch.zeros(n), surface_normal_uncertainty=torch.zeros(n),
        ),
        representation=RepresentationBelief(
            rgb_residual_ema=torch.zeros(n), depth_residual_ema=torch.zeros(n),
            render_contribution=torch.ones(n), gradient_utility_evidence=torch.zeros(n),
            redundancy=torch.zeros(n), expected_marginal_gain=torch.zeros(n),
        ),
        temporal=TemporalBelief(
            change_probability=torch.zeros(n), evidence_recency=torch.zeros(n, dtype=torch.int32),
            confidence_decay=torch.zeros(n),
        ),
        n=n,
    )


def test_policy_abstains_without_scheduler_mutation_quota():
    result = GaussianPopulationPolicy().decide(
        _belief(3), MutationWorkToken(epoch=1, units=0), PopulationEvidence.empty(3)
    )
    assert torch.equal(result.actions, torch.full((3,), PopulationAction.ABSTAIN.code, dtype=torch.int8))


def test_underexposed_rows_abstain_even_when_render_contribution_is_low():
    belief = _belief(2)
    belief.observability.meaningful_exposure_count[0] = 0
    belief.representation.redundancy[:] = 1.0
    result = GaussianPopulationPolicy(min_meaningful_exposures=1).decide(
        belief, MutationWorkToken(epoch=1, units=2), PopulationEvidence.empty(2)
    )
    assert result.actions[0].item() == PopulationAction.ABSTAIN.code


def test_capacity_at_cap_emits_paired_relocation_not_retire_and_create():
    belief = _belief(3)
    belief.representation.expected_marginal_gain[:] = torch.tensor([0.1, 0.2, 0.9])
    belief.representation.redundancy[:] = torch.tensor([0.9, 0.2, 0.1])
    evidence = PopulationEvidence.empty(3)
    evidence.target_gain[:] = torch.tensor([0.1, 0.2, 0.95])
    result = GaussianPopulationPolicy().decide(
        belief, MutationWorkToken(epoch=2, units=1), evidence, capacity=3
    )
    assert result.actions[0].item() == PopulationAction.RELOCATE.code
    assert result.target_indices[0].item() == 2
    assert not bool(torch.any(result.actions == PopulationAction.CREATE.code))
    assert not bool(torch.any(result.actions == PopulationAction.RETIRE.code))


def test_neutral_scheduler_token_leaves_create_or_relocate_to_population_policy():
    """The same scheduler budget permits either policy-selected action class."""

    token = MCMCCadenceOwner(
        grow_interval_pixels=100,
        relocate_interval_pixels=100,
        min_optimizer_updates_after_structural_mutation=0,
    ).authorize_population_mutation(
        RuntimeWorkClockSnapshot(optimizer_updates=1, depth_valid_samples=1),
        topology_version=0,
        max_actions=1,
        geometry_opportunity=True,
    )
    create_belief = _belief(3)
    create_evidence = PopulationEvidence.empty(3)
    create_evidence.target_gain[:] = torch.tensor([0.1, 0.2, 1.0])
    relocate_belief = _belief(3)
    relocate_belief.representation.redundancy[:] = torch.tensor([1.0, 0.2, 0.1])
    relocate_evidence = PopulationEvidence.empty(3)
    relocate_evidence.target_gain[:] = torch.tensor([0.1, 0.2, 1.0])

    create = GaussianPopulationPolicy().decide(
        create_belief, token, create_evidence, capacity=4
    )
    relocate = GaussianPopulationPolicy().decide(
        relocate_belief, token, relocate_evidence, capacity=3
    )

    assert torch.any(create.actions == PopulationAction.CREATE.code)
    assert torch.any(relocate.actions == PopulationAction.RELOCATE.code)


def test_depth_conflict_corrects_trusted_rows_and_uncertainty_explores():
    belief = _belief(2)
    belief.geometry.p_surface_consistent[:] = torch.tensor([0.95, 0.2])
    belief.geometry.surface_position_uncertainty[:] = torch.tensor([0.1, 1.0])
    evidence = PopulationEvidence.empty(2)
    evidence.depth_residual[:] = torch.tensor([1.0, 0.0])
    result = GaussianPopulationPolicy().decide(
        belief, MutationWorkToken(epoch=1, units=2), evidence
    )
    assert result.actions[0].item() == PopulationAction.CORRECT.code
    assert result.actions[1].item() == PopulationAction.EXPLORE.code


def test_token_rejects_invalid_budget():
    with pytest.raises(ValueError):
        MutationWorkToken(epoch=1, units=-1)


def test_empty_population_is_safe_even_at_zero_capacity():
    result = GaussianPopulationPolicy().decide(
        _belief(0), MutationWorkToken(epoch=1, units=1), PopulationEvidence.empty(0), capacity=0
    )
    assert result.actions.shape == (0,)
    assert result.target_indices.shape == (0,)


def test_finite_validation_is_explicit_not_part_of_each_policy_epoch():
    evidence = PopulationEvidence(
        target_gain=torch.tensor([float("nan")]),
        depth_residual=torch.zeros(1),
        free_space_conflict=torch.zeros(1),
    )
    with pytest.raises(AssertionError):
        evidence.validate_finite()


def test_create_is_bounded_by_remaining_capacity():
    belief = _belief(3)
    evidence = PopulationEvidence.empty(3)
    evidence.target_gain[:] = 1.0
    result = GaussianPopulationPolicy().decide(
        belief, MutationWorkToken(epoch=1, units=3), evidence, capacity=4
    )
    assert int(torch.count_nonzero(result.actions == PopulationAction.CREATE.code)) == 1


def test_missing_evidence_emits_versioned_bounded_scheduler_demands():
    belief = _belief(2)
    belief.observability.meaningful_exposure_count[:] = 0
    result = GaussianPopulationPolicy().decide(
        belief,
        MutationWorkToken(epoch=4, units=1),
        PopulationEvidence.empty(2),
        snapshot_versions={
            "map": 3, "topology": 4, "coordinate": 5, "schema": 6,
            "optimizer": 7, "surface": 8, "policy": 9,
        },
    )
    assert len(result.evidence_demands) == 2
    assert result.evidence_demands.demands[0].map_version == 3
    assert result.evidence_demands.demands[0].topology_version == 4
    assert result.demand_masks[0].dtype is torch.bool
    assert bool(torch.all(result.demand_masks[0]))


def test_missing_snapshot_versions_never_fabricates_demand_versions():
    belief = _belief(1)
    belief.observability.meaningful_exposure_count[:] = 0
    result = GaussianPopulationPolicy().decide(
        belief, MutationWorkToken(epoch=1, units=1), PopulationEvidence.empty(1)
    )
    assert len(result.evidence_demands) == 0


def test_zero_mutation_quota_still_returns_evidence_demands():
    belief = _belief(1)
    belief.observability.meaningful_exposure_count[:] = 0
    versions = {name: 1 for name in ("map", "topology", "coordinate", "schema", "optimizer", "surface", "policy")}
    result = GaussianPopulationPolicy().decide(
        belief, MutationWorkToken(epoch=1, units=0), PopulationEvidence.empty(1),
        snapshot_versions=versions,
    )
    assert len(result.evidence_demands) == 2
    assert torch.all(result.actions == PopulationAction.ABSTAIN.code)


def test_units_cap_total_emitted_action_work():
    belief = _belief(4)
    belief.geometry.p_surface_consistent[:] = 1.0
    evidence = PopulationEvidence.empty(4)
    evidence.depth_residual[:] = 1.0
    result = GaussianPopulationPolicy().decide(
        belief, MutationWorkToken(epoch=1, units=2), evidence
    )
    assert int(torch.count_nonzero(result.actions >= PopulationAction.CORRECT.code)) == 2


def test_capacity_mode_does_not_mix_unpaired_retirement_with_relocation():
    belief = _belief(3)
    belief.existence.p_exist[:] = torch.tensor([0.01, 1.0, 1.0])
    belief.representation.redundancy[:] = torch.tensor([1.0, 0.1, 0.1])
    evidence = PopulationEvidence.empty(3)
    evidence.target_gain[2] = 1.0
    result = GaussianPopulationPolicy().decide(
        belief, MutationWorkToken(epoch=1, units=2), evidence, capacity=3
    )
    assert int(torch.count_nonzero(result.actions == PopulationAction.RELOCATE.code)) == 1
    assert int(torch.count_nonzero(result.actions == PopulationAction.RETIRE.code)) == 0


def test_relocation_donor_and_target_are_distinct():
    belief = _belief(2)
    belief.representation.redundancy[:] = torch.tensor([1.0, 0.1])
    evidence = PopulationEvidence.empty(2)
    evidence.target_gain[:] = 1.0
    result = GaussianPopulationPolicy().decide(
        belief, MutationWorkToken(epoch=1, units=1), evidence, capacity=2
    )
    donor = torch.nonzero(result.actions == PopulationAction.RELOCATE.code).flatten()[0]
    assert donor.item() != result.target_indices[donor].item()


def test_rejects_nonfinite_threshold_and_wrong_evidence_dtype():
    with pytest.raises(ValueError):
        GaussianPopulationPolicy(target_gain_threshold=float("nan"))
    belief = _belief(1)
    evidence = PopulationEvidence(
        target_gain=torch.zeros(1, dtype=torch.int32),
        depth_residual=torch.zeros(1),
        free_space_conflict=torch.zeros(1),
    )
    with pytest.raises(TypeError):
        GaussianPopulationPolicy().decide(belief, MutationWorkToken(1, 1), evidence)


def test_policy_exports_only_a_bounded_device_structural_selection_envelope():
    belief = _belief(8)
    evidence = PopulationEvidence.empty(8)
    evidence.target_gain[:] = 1.0
    result = GaussianPopulationPolicy().decide(
        belief, MutationWorkToken(epoch=2, units=2), evidence, capacity=10,
    )

    selection = result.structural_selection
    assert selection is not None
    assert selection.max_actions == 2
    assert selection.source_rows.shape == (2,)
    assert selection.action_codes.shape == (2,)
    assert selection.valid_mask.shape == (2,)
    assert torch.all(selection.surface_target_ids == -1)


def test_policy_never_exports_mixed_create_and_retire_structural_selection():
    belief = _belief(3)
    belief.existence.p_exist[0] = 0.0
    evidence = PopulationEvidence.empty(3)
    evidence.target_gain[:] = 1.0
    result = GaussianPopulationPolicy().decide(
        belief, MutationWorkToken(epoch=3, units=2), evidence, capacity=5,
    )

    selection = result.structural_selection
    assert selection is not None
    selected = selection.action_codes[selection.valid_mask]
    assert torch.unique(selected).numel() <= 1
    assert result.has_authentic_structural_selection()
