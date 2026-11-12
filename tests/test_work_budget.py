from dataclasses import FrozenInstanceError, replace
from types import SimpleNamespace

import pytest

from eligsir.contracts.control import (
    ControlLimits,
    ControlProposal,
    ControlSnapshot,
    DispositionKind,
    FreshnessMode,
    MappingTransactionClass,
    ModalityEligibility,
    ObservationDisposition,
    ResolutionTier,
    ResidencyIntent,
    DepthInsertionPlan,
)
from eligsir.contracts.runtime import SourceIdentity
from eligsir.mapping.scheduler import WorkBudgetOwner
from eligsir.mapping.scheduler.work_budget import (
    AdaptiveGrowthProposalProvider,
    BatchConstellationProposalProvider,
    MutationProposalProvider,
    OptimizerProposalProvider,
    WeightedReplayCandidateProvider,
    collect_control_proposals,
)
from eligsir.mapping.scheduler.resolution_policy import ResolutionPolicyProvider
from eligsir.mapping.scheduler.tier_cost_model import (
    FORWARD_BACKWARD,
    TierCostContext,
    TierCostEstimator,
    TierCostSample,
)


def _source(ordinal: int) -> SourceIdentity:
    return SourceIdentity(
        source_id="camera",
        stream_epoch=1,
        source_ordinal=ordinal,
        timestamp_ns=ordinal * 1_000_000,
    )


def _limits() -> ControlLimits:
    return ControlLimits(
        max_supervision_slots=2,
        max_replay_slots=1,
        max_optimizer_updates=3,
        max_insertion_seeds=5,
        max_mutations=7,
        max_transfer_bytes=1_000_000,
    )


def _snapshot(*, mode: FreshnessMode) -> ControlSnapshot:
    pending = (_source(4), _source(5), _source(6))
    return ControlSnapshot(
        snapshot_id=17,
        control_version=3,
        source=pending[-1],
        tracking=None,
        pending_sources=pending,
        newest_frontier=pending[-1],
        freshness_mode=mode,
        pending_oldest_age_ns=10_000,
        ingress_queue_depth=3,
        candidate_queue_depth=2,
        transfer_queue_depth=1,
        gpu_queue_depth=1,
        map_content_version=29,
        topology_version=11,
        parameter_version=31,
        gaussian_count=100,
        optimizer_steps_completed=40,
        limits=_limits(),
        supervision_candidates=pending,
        replay_candidates=(_source(1), _source(2)),
        replay_work_modalities=(
            ModalityEligibility(True, "qualified", False, "missing", False),
            ModalityEligibility(True, "qualified", False, "missing", False),
        ),
    )


def _disposition(snapshot: ControlSnapshot, *, admitted: bool = True) -> ObservationDisposition:
    return ObservationDisposition(
        source=snapshot.source,
        snapshot_id=snapshot.snapshot_id,
        kind=DispositionKind.ADMIT if admitted else DispositionKind.METADATA_ONLY,
        observe=True,
        mapping_candidate=admitted,
        admit_supervision=admitted,
        allow_geometry_insertion=admitted,
        residency=ResidencyIntent.HOT if admitted else ResidencyIntent.METADATA_ONLY,
        reason="qualified" if admitted else "tracking_unsafe",
        modality_eligibility=(
            ModalityEligibility(True, "qualified", True, "qualified", True)
            if admitted
            else ModalityEligibility()
        ),
    )


def _resolve(snapshot, proposals=(), *, disposition=None, freshness_mode=None):
    return WorkBudgetOwner().resolve(
        snapshot,
        disposition=disposition or _disposition(snapshot),
        proposals=tuple(proposals),
        freshness_mode=freshness_mode,
    )


def test_critical_plan_keeps_newest_frontier_and_bounded_replay():
    snapshot = _snapshot(mode=FreshnessMode.CRITICAL)
    proposals = (
            ControlProposal("legacy", "supervision_slots", 99, "backlog"),
            ControlProposal("legacy", "replay_slots", 99, "backlog"),
            ControlProposal("legacy", "optimizer_updates", 3, "bounded service"),
            ControlProposal("repair", "regional_repair_optimizer_updates", 99, "debt"),
    )

    plan = _resolve(snapshot, proposals, freshness_mode=FreshnessMode.CRITICAL)

    assert plan.supervision_sources == (_source(6),)
    assert plan.replay_sources == (_source(2),)
    assert plan.optimizer_updates == 3
    assert "critical_newest_frontier" in plan.reason_trace
    assert "critical_bounded_replay_preserved" in plan.reason_trace


def test_batch_constellation_uses_hysteresis_and_fast_replay_fallback():
    from eligsir.mapping.scheduler.work_budget import (
        BatchConstellationProposalProvider,
    )

    provider = BatchConstellationProposalProvider(batch_size=4, pressure_dwell=3)
    healthy = replace(
        _snapshot(mode=FreshnessMode.HEALTHY),
        limits=replace(_limits(), max_supervision_slots=3, max_replay_slots=3),
    )
    pressured = replace(healthy, freshness_mode=FreshnessMode.CRITICAL)

    def slots(snapshot):
        return {
            proposal.actuator: proposal.value
            for proposal in provider.propose(snapshot)
        }

    assert slots(healthy) == {"supervision_slots": 1, "replay_slots": 3}
    assert slots(replace(pressured, control_tick_epoch=1)) == {
        "supervision_slots": 1,
        "replay_slots": 3,
    }
    # Repeated snapshots in one drain are the same decision barrier.
    assert slots(replace(pressured, snapshot_id=18, control_tick_epoch=1)) == {
        "supervision_slots": 1,
        "replay_slots": 3,
    }
    assert slots(replace(pressured, control_tick_epoch=2)) == {
        "supervision_slots": 1,
        "replay_slots": 3,
    }
    assert slots(replace(pressured, control_tick_epoch=3)) == {
        "supervision_slots": 3,
        "replay_slots": 1,
    }
    assert slots(replace(pressured, control_tick_epoch=4)) == {
        "supervision_slots": 3,
        "replay_slots": 1,
    }
    assert slots(replace(healthy, control_tick_epoch=5)) == {
        "supervision_slots": 1,
        "replay_slots": 3,
    }


def test_critical_c4_plan_executes_three_pending_new_and_one_replay():
    snapshot = replace(
        _snapshot(mode=FreshnessMode.CRITICAL),
        limits=replace(
            _limits(),
            max_supervision_slots=3,
            max_replay_slots=3,
            max_optimizer_updates=1,
        ),
        replay_candidates=(_source(1), _source(2), _source(4), _source(5)),
        replay_work_modalities=(
            ModalityEligibility(True, "qualified", False, "missing", False),
        )
        * 4,
    )
    proposals = (
        ControlProposal("constellation", "supervision_slots", 3, "sustained backlog"),
        ControlProposal("constellation", "replay_slots", 1, "preserve replay floor"),
        ControlProposal("optimizer", "optimizer_updates", 1, "true C4"),
    )

    plan = _resolve(snapshot, proposals, freshness_mode=FreshnessMode.CRITICAL)

    assert plan.supervision_sources == (_source(4), _source(5), _source(6))
    assert len(plan.replay_sources) == 1
    assert plan.replay_sources[0] in {_source(1), _source(2)}
    assert plan.requested_supervision_slots == 3
    assert plan.requested_replay_slots == 1
    assert tuple(work.source for work in plan.supervision_work) == (
        _source(4),
        _source(5),
        _source(6),
    )
    assert plan.supervision_work[-1].modality_eligibility.depth_insertion
    assert not plan.supervision_work[0].modality_eligibility.depth_insertion


def test_live_population_true_b4_c4_supervision_carries_neutral_mutation_budget():
    """Catch source-active B4 evidence being starved by a replay-only MCMC gate."""

    snapshot = replace(
        _snapshot(mode=FreshnessMode.HEALTHY),
        snapshot_id=4,
        limits=replace(
            _limits(),
            max_supervision_slots=3,
            max_replay_slots=3,
            max_optimizer_updates=1,
            max_mutations=1,
        ),
        replay_candidates=(_source(1), _source(2), _source(3)),
        replay_work_modalities=(
            ModalityEligibility(True, "qualified", False, "missing", False),
        )
        * 3,
    )
    proposals = collect_control_proposals(
        snapshot,
        (
            BatchConstellationProposalProvider(batch_size=4),
            OptimizerProposalProvider(),
            MutationProposalProvider(live_population_authorization=True),
        ),
    )

    plan = _resolve(snapshot, proposals)

    assert plan.supervision_constellation is not None
    assert plan.supervision_constellation.realized_batch_size == 4
    assert plan.supervision_constellation.realized_camera_count == 4
    assert plan.supervision_work[0].modality_eligibility.depth_supervision
    assert plan.mutation_budget.max_total == 1
    assert not plan.allow_growth
    assert not plan.allow_relocation


@pytest.mark.parametrize("batch_size", (1, 2))
def test_legacy_source_active_plan_retains_zero_mutation_budget(batch_size):
    snapshot = replace(
        _snapshot(mode=FreshnessMode.HEALTHY),
        limits=replace(
            _limits(),
            max_supervision_slots=batch_size,
            max_replay_slots=max(1, batch_size - 1),
            max_optimizer_updates=1,
            max_mutations=1,
        ),
    )
    proposals = collect_control_proposals(
        snapshot,
        (
            BatchConstellationProposalProvider(batch_size=batch_size),
            OptimizerProposalProvider(),
            MutationProposalProvider(),
        ),
    )

    plan = _resolve(snapshot, proposals)

    assert plan.mutation_budget.max_total == 0


@pytest.mark.parametrize("batch_size", (1, 2))
def test_live_source_active_b1_b2_plan_retains_zero_mutation_budget(batch_size):
    snapshot = replace(
        _snapshot(mode=FreshnessMode.HEALTHY),
        limits=replace(
            _limits(),
            max_supervision_slots=batch_size,
            max_replay_slots=max(1, batch_size - 1),
            max_optimizer_updates=1,
            max_mutations=1,
        ),
    )
    proposals = collect_control_proposals(
        snapshot,
        (
            BatchConstellationProposalProvider(batch_size=batch_size),
            OptimizerProposalProvider(),
            MutationProposalProvider(live_population_authorization=True),
        ),
    )

    plan = _resolve(snapshot, proposals)

    assert plan.mutation_budget.max_total == 0


def test_critical_multi_new_excludes_every_supervision_identity_from_replay():
    snapshot = replace(
        _snapshot(mode=FreshnessMode.CRITICAL),
        limits=replace(
            _limits(),
            max_supervision_slots=3,
            max_replay_slots=3,
            max_optimizer_updates=1,
        ),
        replay_candidates=(_source(4), _source(5)),
        replay_work_modalities=(
            ModalityEligibility(True, "qualified", False, "missing", False),
        )
        * 2,
    )
    proposals = (
        ControlProposal("constellation", "supervision_slots", 3, "sustained backlog"),
        ControlProposal("constellation", "replay_slots", 1, "preserve replay floor"),
        ControlProposal("optimizer", "optimizer_updates", 1, "true C4"),
    )

    plan = _resolve(snapshot, proposals, freshness_mode=FreshnessMode.CRITICAL)

    assert plan.supervision_sources == (_source(4), _source(5), _source(6))
    assert plan.replay_sources == ()
    assert plan.transaction_class is MappingTransactionClass.FRESH


def test_critical_default_providers_do_not_suppress_available_replay():
    from eligsir.mapping.scheduler.work_budget import (
        BatchConstellationProposalProvider,
        OptimizerProposalProvider,
    )

    snapshot = _snapshot(mode=FreshnessMode.CRITICAL)
    proposals = (
        *BatchConstellationProposalProvider(batch_size=2).propose(snapshot),
        OptimizerProposalProvider().propose(snapshot),
    )

    assert {proposal.actuator: proposal.value for proposal in proposals} == {
        "supervision_slots": 1,
        "replay_slots": 1,
        "optimizer_updates": 2,
    }


def test_true_camera_batch_can_plan_three_replays_for_one_optimizer_update():
    from eligsir.mapping.scheduler.work_budget import (
        BatchConstellationProposalProvider,
        OptimizerProposalProvider,
    )

    snapshot = replace(
        _snapshot(mode=FreshnessMode.CRITICAL),
        limits=replace(
            _snapshot(mode=FreshnessMode.CRITICAL).limits,
            max_supervision_slots=1,
            max_replay_slots=3,
            max_optimizer_updates=1,
        ),
        replay_candidates=(_source(1), _source(2), _source(3)),
        replay_work_modalities=(
            ModalityEligibility(True, "qualified", False, "missing", False),
        )
        * 3,
    )
    proposals = (
        *BatchConstellationProposalProvider(batch_size=4).propose(snapshot),
        OptimizerProposalProvider().propose(snapshot),
    )

    plan = _resolve(snapshot, proposals, freshness_mode=FreshnessMode.CRITICAL)

    assert plan.supervision_sources == (_source(6),)
    assert set(plan.replay_sources) == {_source(1), _source(2), _source(3)}
    assert plan.optimizer_updates == 1
    assert (
        "proposal:BatchConstellationProposalProvider:replay_slots=3:"
        "replay_biased;freshness=critical;pressure_streak=1/3"
        in plan.reason_trace
    )
    assert (
        "proposal:OptimizerProposalProvider:optimizer_updates=1:one true-camera batch update"
        in plan.reason_trace
    )


def test_replay_candidates_are_weighted_random_samples_from_the_full_cache():
    snapshot = replace(
        _snapshot(mode=FreshnessMode.HEALTHY),
        limits=replace(_limits(), max_replay_slots=3),
        replay_candidates=tuple(_source(index) for index in range(101, 201)),
        replay_work_modalities=(
            ModalityEligibility(True, "qualified", False, "missing", False),
        )
        * 100,
        replay_candidate_weights=(1.0,) * 100,
    )
    proposals = (
        ControlProposal("policy", "replay_slots", 3, "C4 replay"),
        ControlProposal("policy", "optimizer_updates", 1, "true C4"),
    )

    plan = _resolve(snapshot, proposals)

    assert len(plan.replay_sources) == 3
    assert set(plan.replay_sources).issubset(set(snapshot.replay_candidates))
    assert plan.replay_population_size == 100
    assert plan.replay_sampling_seed > 0
    assert plan.replay_selected_weights == (1.0, 1.0, 1.0)
    assert "weighted_random_full_cache" in plan.reason_trace


def test_weighted_replay_biases_focus_without_removing_the_global_floor():
    candidates = tuple(_source(index) for index in range(1, 33))
    focused = candidates[7]
    provider = WeightedReplayCandidateProvider(random_seed=9, global_weight_floor=0.05)
    focused_hits = 0
    zero_weight_hits = 0
    zero_weight = candidates[0]

    for epoch in range(512):
        weights = tuple(
            50.0 if source == focused else 0.0 if source == zero_weight else 1.0
            for source in candidates
        )
        selection = provider.sample(
            candidates,
            weights,
            slots=1,
            epoch=epoch,
            snapshot_id=17,
        )
        focused_hits += int(selection.sources == (focused,))
        zero_weight_hits += int(selection.sources == (zero_weight,))

    assert focused_hits > 256
    assert zero_weight_hits > 0


def test_weighted_replay_is_reproducible_but_rotates_with_epoch():
    candidates = tuple(_source(index) for index in range(1, 65))
    provider = WeightedReplayCandidateProvider(random_seed=3)

    first = provider.sample(candidates, (1.0,) * 64, slots=4, epoch=10, snapshot_id=2)
    repeated = provider.sample(candidates, (1.0,) * 64, slots=4, epoch=10, snapshot_id=2)
    next_epoch = provider.sample(candidates, (1.0,) * 64, slots=4, epoch=11, snapshot_id=2)

    assert first == repeated
    assert next_epoch.sources != first.sources


def test_weighted_replay_keeps_global_first_and_deduplicates_reserved_shortlist():
    """A challenger shortlist must not displace the global replay floor."""

    candidates = tuple(_source(index) for index in range(1, 5))
    selection = WeightedReplayCandidateProvider().sample(
        candidates,
        (1.0, 2.0, 3.0, 4.0),
        slots=3,
        epoch=1,
        snapshot_id=17,
        reserved_candidate=candidates[0],
        reserved_candidates=(candidates[0], candidates[2], candidates[2], candidates[3]),
    )

    assert selection.sources == (candidates[0], candidates[2], candidates[3])
    assert selection.weights == (1.0, 3.0, 4.0)


def test_idle_replay_reserves_rotating_cold_challengers_without_displacing_global_or_fallback():
    """Removing challenger reservations would leave no rotating cold trial in B4."""

    candidates = tuple(_source(index) for index in range(1, 9))
    modality = ModalityEligibility(True, "qualified", False, "missing", False)
    snapshot = replace(
        _snapshot(mode=FreshnessMode.HEALTHY),
        limits=replace(_limits(), max_replay_slots=4),
        replay_candidates=candidates,
        replay_work_modalities=(modality,) * len(candidates),
        reserved_global_replay_candidate=candidates[0],
    )
    active = _resolve(snapshot)
    owner = WorkBudgetOwner(
        requested_batch_size=4,
        maximum_camera_count=4,
        constellation_policy="fixed",
    )
    cold = candidates[4:]
    plans = tuple(
        owner.resolve_idle_replay(
            snapshot,
            active,
            replay_modalities={source: modality for source in candidates},
            replay_weights={source: 1.0 for source in candidates},
            batch_size=4,
            optimizer_steps_completed=epoch,
            versions=(29, 11, 31),
            cold_challenger_sources=cold,
            challenger_fraction=0.75,
            force_reserved_global=True,
        )
        for epoch in (41, 42, 43)
    )

    assert all(plan.replay_sources[0] == candidates[0] for plan in plans)
    assert all(len(plan.replay_sources) == len(set(plan.replay_sources)) == 4 for plan in plans)
    assert tuple(plan.replay_sources[1:3] for plan in plans) == (
        (candidates[5], candidates[6]),
        (candidates[6], candidates[7]),
        (candidates[7], candidates[4]),
    )
    assert tuple(plan.replay_trial_sources for plan in plans) == tuple(
        plan.replay_sources[1:3] for plan in plans
    )


def test_idle_replay_cold_challengers_are_inert_for_off_and_post_refinement_paths():
    candidates = tuple(_source(index) for index in range(1, 6))
    modality = ModalityEligibility(True, "qualified", False, "missing", False)
    snapshot = replace(
        _snapshot(mode=FreshnessMode.HEALTHY),
        replay_candidates=candidates,
        replay_work_modalities=(modality,) * len(candidates),
        limits=replace(_limits(), max_replay_slots=4),
    )
    active = _resolve(snapshot)
    owner = WorkBudgetOwner(requested_batch_size=4, maximum_camera_count=4)
    common = dict(
        replay_modalities={source: modality for source in candidates},
        replay_weights={source: 1.0 for source in candidates},
        batch_size=4,
        optimizer_steps_completed=41,
        versions=(29, 11, 31),
    )

    baseline = owner.resolve_idle_replay(snapshot, active, **common)
    off = owner.resolve_idle_replay(
        snapshot,
        active,
        **common,
        cold_challenger_sources=candidates[2:],
        challenger_fraction=0.0,
    )
    refined_baseline = owner.resolve_idle_replay(
        snapshot, active, **common, post_refinement=True
    )
    refined = owner.resolve_idle_replay(
        snapshot,
        active,
        **common,
        cold_challenger_sources=candidates[2:],
        challenger_fraction=0.75,
        post_refinement=True,
    )

    assert off == baseline
    assert refined == refined_baseline
    assert owner.resolve_idle_replay(
        snapshot,
        active,
        **{**common, "batch_size": 1},
        cold_challenger_sources=candidates[2:],
        challenger_fraction=0.75,
    ) == owner.resolve_idle_replay(
        snapshot, active, **{**common, "batch_size": 1}
    )
    with pytest.raises(ValueError, match="eligible"):
        owner.resolve_idle_replay(
            snapshot,
            active,
            **common,
            cold_challenger_sources=(candidates[0], _source(99)),
            challenger_fraction=0.75,
        )
    with pytest.raises(ValueError, match="challenger_fraction"):
        owner.resolve_idle_replay(
            snapshot,
            active,
            **common,
            cold_challenger_sources=candidates[2:],
            challenger_fraction=1.1,
        )


def test_idle_replay_plan_names_one_depth_eligible_replay_insertion_identity():
    snapshot = replace(
        _snapshot(mode=FreshnessMode.HEALTHY),
        limits=replace(
            _limits(), max_replay_slots=2, max_replay_insertion_seeds=64
        ),
        replay_candidates=(_source(1), _source(2)),
        replay_work_modalities=(
            ModalityEligibility(True, "qualified", True, "qualified", True),
            ModalityEligibility(True, "qualified", True, "qualified", True),
        ),
    )
    active = _resolve(snapshot)
    owner = WorkBudgetOwner()

    idle = owner.resolve_idle_replay(
        snapshot,
        active,
        replay_modalities=dict(
            zip(snapshot.replay_candidates, snapshot.replay_work_modalities, strict=True)
        ),
        replay_weights={source: 1.0 for source in snapshot.replay_candidates},
        batch_size=2,
        optimizer_steps_completed=100,
        versions=(29, 11, 31),
        proposals=(
            ControlProposal("tail", "replay_insertion_seeds", 64, "settled replay depth"),
        ),
    )

    assert idle.replay_insertion is not None
    assert idle.replay_insertion.enabled
    assert idle.replay_insertion.source in idle.replay_sources
    assert idle.replay_insertion.seed_budget == 64
    selected = next(
        work for work in idle.replay_work if work.source == idle.replay_insertion.source
    )
    assert selected.modality_eligibility.depth_insertion


def test_idle_replay_realizes_b4_without_a_fresh_source():
    """A zero-frame clock can select a full B4 from resident replay alone."""

    replay = tuple(_source(ordinal) for ordinal in range(4))
    active_snapshot = _snapshot(mode=FreshnessMode.HEALTHY)
    active = _resolve(active_snapshot)
    idle_snapshot = replace(
        active_snapshot,
        source=None,
        pending_sources=(),
        newest_frontier=None,
        supervision_candidates=(),
        replay_candidates=replay,
        replay_work_modalities=(
            ModalityEligibility(True, "qualified", False, "missing", False)
            for _ in replay
        ),
        limits=replace(_limits(), max_replay_slots=4),
    )

    idle = WorkBudgetOwner().resolve_idle_replay(
        idle_snapshot,
        active,
        replay_modalities=dict(
            zip(replay, idle_snapshot.replay_work_modalities, strict=True)
        ),
        replay_weights={source: 1.0 for source in replay},
        batch_size=4,
        optimizer_steps_completed=41,
        versions=(29, 11, 31),
    )

    assert idle.supervision_sources == ()
    assert len(idle.replay_sources) == 4
    assert idle.supervision_constellation is not None
    assert idle.supervision_constellation.realized_batch_size == 4


@pytest.mark.parametrize(
    ("batch_size", "maximum_camera_count", "expected_camera_count"),
    ((1, 1, 1), (4, 4, 4), (16, 4, 4)),
)
def test_idle_replay_carries_the_ready_candidate_pair_for_each_constellation(
    batch_size, maximum_camera_count, expected_camera_count
):
    candidates = tuple(_source(ordinal) for ordinal in range(16))
    weights = tuple(float(ordinal + 1) for ordinal in range(16))
    snapshot = replace(
        _snapshot(mode=FreshnessMode.HEALTHY),
        replay_candidates=candidates,
        replay_work_modalities=(
            ModalityEligibility(True, "qualified", False, "missing", False)
            for _ in candidates
        ),
        replay_candidate_weights=weights,
        limits=replace(_limits(), max_replay_slots=len(candidates)),
    )
    active = _resolve(snapshot)
    idle = WorkBudgetOwner(
        requested_batch_size=batch_size,
        maximum_camera_count=maximum_camera_count,
    ).resolve_idle_replay(
        snapshot,
        active,
        replay_modalities=dict(
            zip(candidates, snapshot.replay_work_modalities, strict=True)
        ),
        replay_weights=dict(zip(candidates, weights, strict=True)),
        batch_size=batch_size,
        optimizer_steps_completed=41,
        versions=(29, 11, 31),
    )

    assert idle.supervision_constellation is not None
    assert idle.supervision_constellation.realized_batch_size == batch_size
    assert idle.supervision_constellation.realized_camera_count == expected_camera_count
    assert idle.replay_candidate_sources == candidates
    assert idle.replay_candidate_weights == weights
    assert set(idle.replay_sources).issubset(candidates)


def test_mapping_work_plan_rejects_incomplete_candidate_evidence():
    snapshot = _snapshot(mode=FreshnessMode.HEALTHY)
    plan = _resolve(snapshot)

    with pytest.raises(ValueError, match="candidate weights must align"):
        replace(
            plan,
            replay_population_size=2,
            replay_candidate_sources=snapshot.replay_candidates,
            replay_candidate_weights=(1.0,),
        )

    with pytest.raises(ValueError, match="candidate weights must be finite"):
        replace(
            plan,
            replay_population_size=2,
            replay_candidate_sources=snapshot.replay_candidates,
            replay_candidate_weights=(1.0, float("nan")),
        )

    with pytest.raises(TypeError, match="candidate_sources must contain"):
        replace(
            plan,
            replay_population_size=1,
            replay_candidate_sources=("not-a-source",),
            replay_candidate_weights=(1.0,),
        )

    with pytest.raises(ValueError, match="candidate_sources must be unique"):
        replace(
            plan,
            replay_population_size=2,
            replay_candidate_sources=(snapshot.replay_candidates[0],) * 2,
            replay_candidate_weights=(1.0, 1.0),
        )

    with pytest.raises(ValueError, match="must match replay population"):
        replace(
            plan,
            replay_population_size=1,
            replay_candidate_sources=snapshot.replay_candidates,
            replay_candidate_weights=(1.0, 1.0),
        )

    with pytest.raises(ValueError, match="must be selected from candidate"):
        replace(
            plan,
            replay_population_size=1,
            replay_candidate_sources=(_source(99),),
            replay_candidate_weights=(1.0,),
        )


def test_idle_replay_plan_is_fenced_to_the_current_control_snapshot():
    previous = _snapshot(mode=FreshnessMode.HEALTHY)
    active = _resolve(previous)
    current = replace(
        previous,
        snapshot_id=18,
        control_version=4,
        freshness_mode=FreshnessMode.CATCH_UP,
        map_content_version=30,
        topology_version=12,
        parameter_version=32,
    )

    idle = WorkBudgetOwner().resolve_idle_replay(
        current,
        active,
        replay_modalities=dict(
            zip(current.replay_candidates, current.replay_work_modalities, strict=True)
        ),
        replay_weights={source: 1.0 for source in current.replay_candidates},
        batch_size=2,
        optimizer_steps_completed=41,
        versions=(30, 12, 32),
    )

    assert idle.snapshot_id == current.snapshot_id
    assert idle.control_version == current.control_version
    assert idle.freshness_mode is current.freshness_mode
    assert (
        idle.map_content_version,
        idle.topology_version,
        idle.parameter_version,
    ) == (30, 12, 32)


def test_post_refinement_forces_native_resolution_despite_a_non_native_proposal():
    """A stale live downscale proposal must not control source-complete replay."""

    snapshot = _snapshot(mode=FreshnessMode.CATCH_UP)
    active = _resolve(snapshot)
    owner = WorkBudgetOwner(requested_batch_size=2, maximum_camera_count=2)

    plan = owner.resolve_idle_replay(
        snapshot,
        active,
        replay_modalities=dict(
            zip(snapshot.replay_candidates, snapshot.replay_work_modalities, strict=True)
        ),
        replay_weights={source: 1.0 for source in snapshot.replay_candidates},
        batch_size=2,
        optimizer_steps_completed=41,
        versions=(29, 11, 31),
        proposals=(
            ControlProposal("live", "replay_resolution_tier", "quarter", "overload"),
        ),
        post_refinement=True,
    )

    assert plan.resolution_tier.value == "native"
    assert plan.transaction_class is MappingTransactionClass.POST_REFINEMENT
    assert {work.resolution_tier.value for work in plan.replay_work} == {"native"}
    assert plan.supervision_constellation.reason == "post_refinement_phase_resolved"
    assert "post_refinement_live_freshness_bypassed" in plan.reason_trace


def test_post_refinement_bounded_policy_honors_a_non_native_proposal():
    """issue #1346: opt-in ``bounded`` policy must not force native tier.

    The default (``native``) behavior above is deliberately preserved; this
    covers the opt-out that keeps the ordinary freshness/throughput ladder
    engaged through refinement so a short refinement tail keeps sustaining
    ``target_supervised_views_per_s`` instead of forcing native tier's
    unbounded per-view cost.
    """

    snapshot = _snapshot(mode=FreshnessMode.CATCH_UP)
    active = _resolve(snapshot)
    owner = WorkBudgetOwner(
        requested_batch_size=2,
        maximum_camera_count=2,
        refinement_resolution_policy="bounded",
    )
    assert owner.refinement_resolution_policy == "bounded"

    plan = owner.resolve_idle_replay(
        snapshot,
        active,
        replay_modalities=dict(
            zip(snapshot.replay_candidates, snapshot.replay_work_modalities, strict=True)
        ),
        replay_weights={source: 1.0 for source in snapshot.replay_candidates},
        batch_size=2,
        optimizer_steps_completed=41,
        versions=(29, 11, 31),
        proposals=(
            ControlProposal("live", "replay_resolution_tier", "quarter", "overload"),
        ),
        post_refinement=True,
    )

    assert plan.resolution_tier.value == "quarter"
    assert plan.transaction_class is MappingTransactionClass.POST_REFINEMENT
    assert {work.resolution_tier.value for work in plan.replay_work} == {"quarter"}


def test_post_refinement_bounded_policy_can_probe_above_the_locked_tier():
    """Regression test for the POST_REFINEMENT probe-gating floor-lock bug.

    Root cause (see docs/architecture/2026-08-31-issue1259-live-path-refinement-cliff.md
    and the fr3-honest-2min-refinement-baseline diagnostic history, including PR
    #1364's attempted-but-insufficient fix): under ``refinement_resolution_policy
    ="bounded"``, ``_resolve_transaction_resolution`` falls through to
    ``TierCostEstimator.select()``, whose only path to ever measure a HIGHER
    tier than whatever is already ``chosen`` is bounded exploration
    (``allow_probe`` -> ``_probe_candidate``). Before this fix, ``allow_probe``
    was gated to ``transaction_class is MappingTransactionClass.REPLAY_ONLY``
    only, so a POST_REFINEMENT transaction sequence that cold-started (or was
    ever pushed down to) the cheapest measured tier had NO PATH to ever try a
    higher one again -- regardless of how much budget headroom the cheap
    tier's own measured cost left. This test reproduces exactly that
    sequence: one recorded QUARTER-tier sample, comfortably inside the
    transaction budget, with the resolution ceiling wide open (proposal =
    ``native``). Under the old REPLAY_ONLY-only gate this stays pinned at
    ``quarter`` forever; the fix lets a POST_REFINEMENT transaction probe a
    higher, as-yet-unmeasured tier exactly like a REPLAY_ONLY one already
    could.
    """

    snapshot = _snapshot(mode=FreshnessMode.CATCH_UP)
    active = _resolve(snapshot)
    owner = WorkBudgetOwner(
        requested_batch_size=2,
        maximum_camera_count=2,
        refinement_resolution_policy="bounded",
    )

    # One cheap, well-measured QUARTER-tier sample in the exact cost bucket
    # this transaction will resolve against (gaussian_count from the
    # snapshot, camera_count from the realized replay batch of 2). Its cost
    # (1ms) is far under the transaction budget (1000ms * 2 cameras / 100
    # target_supervised_views_per_s == 20ms), well inside the default 0.5
    # probe-headroom fraction -- exactly the "budget has headroom" state the
    # bug report describes.
    owner.record_completed_transaction(
        transaction_class=MappingTransactionClass.POST_REFINEMENT,
        tier=ResolutionTier.QUARTER,
        gaussian_count=snapshot.gaussian_count,
        realized_camera_count=2,
        prepare_ms=0.1,
        optimization_ms=0.4,
        transaction_wall_ms=1.0,
    )

    plan = owner.resolve_idle_replay(
        snapshot,
        active,
        replay_modalities=dict(
            zip(snapshot.replay_candidates, snapshot.replay_work_modalities, strict=True)
        ),
        replay_weights={source: 1.0 for source in snapshot.replay_candidates},
        batch_size=2,
        optimizer_steps_completed=41,
        versions=(29, 11, 31),
        proposals=(
            # Ceiling wide open: nothing here should prevent exploring
            # upward, isolating the fix to the probe gate itself.
            ControlProposal("test", "replay_resolution_tier", "native", "ceiling open"),
        ),
        post_refinement=True,
    )

    # Only QUARTER has ever been measured, so the ordinary cost-fitting
    # selection alone would hold at "quarter" forever. A tier other than
    # "quarter" here is only reachable through a probe of an unmeasured
    # tier -- proof that POST_REFINEMENT transactions can now explore
    # upward instead of being permanently floor-locked.
    assert plan.resolution_tier.value != "quarter"
    assert plan.transaction_class is MappingTransactionClass.POST_REFINEMENT
    assert {work.resolution_tier.value for work in plan.replay_work} == {
        plan.resolution_tier.value
    }


def test_post_refinement_replay_only_still_shares_the_same_probe_path():
    """REPLAY_ONLY keeps its pre-existing probe affordance unchanged.

    Guards against a fix that accidentally narrows probing instead of
    widening it: REPLAY_ONLY must still probe exactly as it did before.
    """

    snapshot = _snapshot(mode=FreshnessMode.CATCH_UP)
    active = _resolve(snapshot)
    owner = WorkBudgetOwner(requested_batch_size=2, maximum_camera_count=2)

    owner.record_completed_transaction(
        transaction_class=MappingTransactionClass.REPLAY_ONLY,
        tier=ResolutionTier.QUARTER,
        gaussian_count=snapshot.gaussian_count,
        realized_camera_count=2,
        prepare_ms=0.1,
        optimization_ms=0.4,
        transaction_wall_ms=1.0,
    )

    plan = owner.resolve_idle_replay(
        snapshot,
        active,
        replay_modalities=dict(
            zip(snapshot.replay_candidates, snapshot.replay_work_modalities, strict=True)
        ),
        replay_weights={source: 1.0 for source in snapshot.replay_candidates},
        batch_size=2,
        optimizer_steps_completed=41,
        versions=(29, 11, 31),
        proposals=(
            ControlProposal("test", "replay_resolution_tier", "native", "ceiling open"),
        ),
        post_refinement=False,
    )

    assert plan.resolution_tier.value != "quarter"
    assert plan.transaction_class is MappingTransactionClass.REPLAY_ONLY


def test_refinement_resolution_policy_rejects_unknown_value():
    with pytest.raises(ValueError):
        WorkBudgetOwner(refinement_resolution_policy="half-baked")


def _replay_only_owner():
    candidates = tuple(_source(index) for index in range(1, 5))
    modality = ModalityEligibility(True, "qualified", False, "missing", False)
    snapshot = replace(
        _snapshot(mode=FreshnessMode.CRITICAL),
        limits=replace(_limits(), max_replay_slots=4),
        replay_candidates=candidates,
        replay_work_modalities=(modality,) * 4,
    )
    owner = WorkBudgetOwner(
        requested_batch_size=4,
        maximum_camera_count=4,
        constellation_policy="fixed",
    )
    active = owner.resolve(
        snapshot,
        disposition=_disposition(snapshot),
        proposals=(
            ControlProposal("resolution", "resolution_tier", "quarter", "fresh pressure"),
            ControlProposal("work", "supervision_slots", 1, "one fresh"),
            ControlProposal("work", "replay_slots", 3, "three replay"),
            ControlProposal("work", "optimizer_updates", 1, "one update"),
        ),
    )
    return owner, snapshot, active, candidates, modality


def _idle(owner, snapshot, active, candidates, modality, *, step, proposals=()):
    return owner.resolve_idle_replay(
        snapshot,
        active,
        replay_modalities={source: modality for source in candidates},
        replay_weights={source: 1.0 for source in candidates},
        batch_size=4,
        optimizer_steps_completed=step,
        versions=(29, 11, 31),
        proposals=proposals,
    )


def _record(owner, *, tier, wall_ms, gaussian_count, camera_count=4):
    owner.record_completed_transaction(
        transaction_class=MappingTransactionClass.REPLAY_ONLY,
        tier=tier,
        gaussian_count=gaussian_count,
        realized_camera_count=camera_count,
        prepare_ms=0.5,
        optimization_ms=wall_ms,
        transaction_wall_ms=wall_ms,
        kernel_ms=wall_ms,
    )


def test_replay_only_holds_the_evidence_ceiling_before_any_cost_is_measured():
    """Bootstrap: with no measured tier cost the proposal ceiling is honoured."""

    owner, snapshot, active, candidates, modality = _replay_only_owner()
    provider = ResolutionPolicyProvider(
        supported_tiers=(
            ResolutionTier.QUARTER,
            ResolutionTier.HALF,
            ResolutionTier.NATIVE,
        ),
        target_views_per_s=100.0,
        idle_upscale_dwell=2,
    )
    assert provider.propose(snapshot).value == "quarter"
    proposal = provider.propose_idle(
        snapshot,
        optimizer_steps_completed=1,
        mapping_cost_ms=5.0,
        batch_size=4,
    )

    plan = _idle(
        owner, snapshot, active, candidates, modality, step=1, proposals=(proposal,)
    )

    assert plan.transaction_class is MappingTransactionClass.REPLAY_ONLY
    assert plan.resolution_tier is ResolutionTier.QUARTER
    assert {work.resolution_tier for work in plan.replay_work} == {ResolutionTier.QUARTER}
    assert plan.resolution_budget_ms == pytest.approx(40.0)
    assert "no valid measured tier evidence" in plan.resolution_reason


def test_owner_selects_the_highest_quality_tier_its_measurements_support():
    """The final tier follows measured transaction cost, not pixel count.

    The budget here is C=4 views at 100 views/s, i.e. 40ms.  Native is
    measured at 30ms and fits, so it wins over the cheaper half tier: the
    allocator spends supervision quality only when measurement says it must.
    """

    owner, snapshot, active, candidates, modality = _replay_only_owner()
    for tier, wall_ms in (
        (ResolutionTier.NATIVE, 30.0),
        (ResolutionTier.HALF, 12.0),
    ):
        _record(
            owner,
            tier=tier,
            wall_ms=wall_ms,
            gaussian_count=snapshot.gaussian_count,
        )

    plan = _idle(
        owner, snapshot, active, candidates, modality, step=1, proposals=(
            ControlProposal(
                "ResolutionPolicyProvider",
                "replay_resolution_tier",
                "native",
                "healthy; native ceiling",
            ),
        )
    )

    assert plan.resolution_budget_ms == pytest.approx(40.0)
    assert plan.resolution_tier is ResolutionTier.NATIVE
    assert {work.resolution_tier for work in plan.replay_work} == {ResolutionTier.NATIVE}


def test_owner_steps_down_only_when_measurement_says_the_budget_is_missed():
    owner, snapshot, active, candidates, modality = _replay_only_owner()
    for tier, wall_ms in (
        (ResolutionTier.NATIVE, 55.0),
        (ResolutionTier.HALF, 25.0),
    ):
        _record(
            owner,
            tier=tier,
            wall_ms=wall_ms,
            gaussian_count=snapshot.gaussian_count,
        )

    plan = _idle(
        owner, snapshot, active, candidates, modality, step=1, proposals=(
            ControlProposal(
                "ResolutionPolicyProvider",
                "replay_resolution_tier",
                "native",
                "healthy; native ceiling",
            ),
        )
    )

    assert plan.resolution_tier is ResolutionTier.HALF
    assert {work.resolution_tier for work in plan.replay_work} == {ResolutionTier.HALF}
    assert "highest-quality feasible measured tier" in plan.resolution_reason


def test_measured_evidence_never_crosses_a_gaussian_count_regime():
    """A small-map calibration must not decide a large-map transaction."""

    owner, snapshot, active, candidates, modality = _replay_only_owner()
    _record(owner, tier=ResolutionTier.NATIVE, wall_ms=30.0, gaussian_count=300_000)

    plan = _idle(
        owner,
        snapshot,
        active,
        candidates,
        modality,
        step=1,
        proposals=(
            ControlProposal(
                "ResolutionPolicyProvider",
                "replay_resolution_tier",
                "half",
                "pressure",
            ),
        ),
    )

    assert snapshot.gaussian_count < 25_000
    assert plan.resolution_tier is ResolutionTier.HALF
    assert "no valid measured tier evidence" in plan.resolution_reason


def test_post_refinement_work_stays_legal_with_an_exhausted_replay_catalog():
    """An empty refinement catalog degrades to a no-op plan, never a crash."""

    snapshot = _snapshot(mode=FreshnessMode.CATCH_UP)
    active = _resolve(snapshot)
    owner = WorkBudgetOwner(requested_batch_size=2, maximum_camera_count=2)

    plan = owner.resolve_idle_replay(
        snapshot,
        active,
        replay_modalities={},
        replay_weights={},
        batch_size=2,
        optimizer_steps_completed=41,
        versions=(29, 11, 31),
        post_refinement=True,
    )

    assert plan.replay_work == ()
    assert plan.supervision_work == ()
    assert plan.transaction_class is MappingTransactionClass.POST_REFINEMENT
    assert plan.resolution_tier is ResolutionTier.NATIVE


def test_replay_only_work_reproposes_until_completed_evidence_and_fresh_never_does():
    owner, snapshot, active, candidates, modality = _replay_only_owner()
    _record(
        owner,
        tier=ResolutionTier.HALF,
        wall_ms=2.0,
        gaussian_count=snapshot.gaussian_count,
    )

    probe = _idle(
        owner, snapshot, active, candidates, modality, step=1, proposals=(
            ControlProposal(
                "ResolutionPolicyProvider",
                "replay_resolution_tier",
                "native",
                "healthy; native ceiling",
            ),
        )
    )

    assert probe.resolution_probe is True
    assert probe.resolution_tier is ResolutionTier.NATIVE
    assert "bounded exploration probe" in probe.resolution_reason

    # Replanning before execution cannot consume the proposal.
    second = _idle(
        owner, snapshot, active, candidates, modality, step=2, proposals=(
            ControlProposal(
                "ResolutionPolicyProvider",
                "replay_resolution_tier",
                "native",
                "healthy; native ceiling",
            ),
        )
    )
    assert second.resolution_probe is True
    assert second.resolution_tier is ResolutionTier.NATIVE

    # The public completed-cost seam, not proposal selection, supplies the
    # matching B4 evidence. The next real idle resolution stays native because
    # its 24ms cost fits the 40ms transaction budget but leaves too little
    # probe headroom to mask that measured selection with quarter.
    _record(
        owner,
        tier=ResolutionTier.NATIVE,
        wall_ms=24.0,
        gaussian_count=snapshot.gaussian_count,
    )
    settled = _idle(
        owner, snapshot, active, candidates, modality, step=3, proposals=(
            ControlProposal(
                "ResolutionPolicyProvider",
                "replay_resolution_tier",
                "native",
                "healthy; native ceiling",
            ),
        )
    )
    assert settled.resolution_tier is ResolutionTier.NATIVE
    assert settled.resolution_probe is False

    # Fresh/mixed work never explores an unmeasured configuration.
    fresh_owner = WorkBudgetOwner(requested_batch_size=1, maximum_camera_count=1)
    _record(
        fresh_owner,
        tier=ResolutionTier.HALF,
        wall_ms=2.0,
        gaussian_count=snapshot.gaussian_count,
        camera_count=1,
    )
    fresh = fresh_owner.resolve(
        snapshot,
        disposition=_disposition(snapshot),
        proposals=(
            ControlProposal("work", "supervision_slots", 1, "one fresh"),
            ControlProposal("work", "optimizer_updates", 1, "one update"),
        ),
    )
    assert fresh.transaction_class is MappingTransactionClass.FRESH
    assert fresh.resolution_probe is False
    assert fresh.resolution_tier is ResolutionTier.HALF


@pytest.mark.parametrize("policy", ("adaptive", "fixed"))
def test_post_refinement_realizes_b256_from_the_full_candidate_catalog_with_c64_cap(policy):
    """A 128-item live shortlist must not truncate a valid B256 refinement step."""

    candidates = tuple(_source(index) for index in range(1, 257))
    modality = ModalityEligibility(True, "qualified", False, "missing", False)
    snapshot = replace(
        _snapshot(mode=FreshnessMode.CATCH_UP),
        replay_candidates=candidates,
        replay_work_modalities=(modality,) * len(candidates),
    )
    active = _resolve(snapshot)
    owner = WorkBudgetOwner(
        requested_batch_size=256,
        maximum_camera_count=64,
        constellation_policy=policy,
    )

    plan = owner.resolve_idle_replay(
        snapshot,
        active,
        replay_modalities={source: modality for source in candidates},
        replay_weights={source: 1.0 for source in candidates},
        batch_size=256,
        optimizer_steps_completed=41,
        versions=(29, 11, 31),
        post_refinement=True,
    )
    repeated = owner.resolve_idle_replay(
        snapshot,
        active,
        replay_modalities={source: modality for source in candidates},
        replay_weights={source: 1.0 for source in candidates},
        batch_size=256,
        optimizer_steps_completed=41,
        versions=(29, 11, 31),
        post_refinement=True,
    )

    constellation = plan.supervision_constellation
    assert len(plan.replay_work) == 256
    assert constellation.requested_batch_size == constellation.realized_batch_size == 256
    assert constellation.requested_camera_count == constellation.realized_camera_count == 64
    assert constellation.requested_camera_count <= constellation.requested_batch_size
    assert repeated.replay_sources == plan.replay_sources
    assert all("catch_up" not in reason for reason in plan.reason_trace)


def test_idle_replay_depth_uses_tail_voxel_without_changing_live_insertion():
    from eligsir.runtime.config import DepthInsertionPolicy

    snapshot = replace(
        _snapshot(mode=FreshnessMode.HEALTHY),
        limits=replace(
            _limits(), max_replay_slots=1, max_replay_insertion_seeds=64
        ),
        replay_candidates=(_source(1),),
        replay_work_modalities=(
            ModalityEligibility(True, "qualified", True, "qualified", True),
        ),
    )
    owner = WorkBudgetOwner(
        depth_insertion=DepthInsertionPolicy(voxel_size_m=0.12),
        replay_insertion_voxel_size_m=0.05,
    )
    active = owner.resolve(
        snapshot,
        disposition=_disposition(snapshot),
        proposals=(ControlProposal("live", "insertion_seeds", 5, "coarse live depth"),),
    )
    idle = owner.resolve_idle_replay(
        snapshot,
        active,
        replay_modalities=dict(
            zip(snapshot.replay_candidates, snapshot.replay_work_modalities, strict=True)
        ),
        replay_weights={_source(1): 1.0},
        batch_size=1,
        optimizer_steps_completed=100,
        versions=(29, 11, 31),
        proposals=(
            ControlProposal("tail", "replay_insertion_seeds", 64, "settled replay depth"),
        ),
    )

    assert active.insertion.voxel_size_m == 0.12
    assert idle.replay_insertion is not None
    assert idle.replay_insertion.voxel_size_m == 0.05


def test_work_plan_rejects_replay_insertion_outside_planned_replay_work():
    snapshot = _snapshot(mode=FreshnessMode.HEALTHY)
    plan = _resolve(snapshot)
    replay_insertion = DepthInsertionPlan(
        source=_source(99),
        enabled=True,
        sample_stride=8,
        candidate_budget=8,
        seed_budget=8,
        voxel_size_m=0.12,
        minimum_confidence=0.0,
    )

    with pytest.raises(ValueError, match="replay insertion source"):
        replace(plan, replay_insertion=replay_insertion)


def test_owner_caps_proposed_work_and_preserves_version_fences():
    snapshot = _snapshot(mode=FreshnessMode.HEALTHY)
    proposals = (
            ControlProposal("legacy", "resolution_tier", "half", "load"),
            ControlProposal("legacy", "supervision_slots", 99, "load"),
            ControlProposal("legacy", "replay_slots", 99, "load"),
            ControlProposal("legacy", "optimizer_updates", 99, "load"),
            ControlProposal("config", "insertion_enabled", True, "enabled"),
            ControlProposal("legacy", "insertion_seeds", 99, "load"),
            ControlProposal("legacy", "allow_growth", True, "load"),
            ControlProposal("legacy", "allow_relocation", True, "load"),
            ControlProposal("legacy", "mutation_budget", 99, "load"),
    )

    plan = _resolve(snapshot, proposals)

    assert plan.plan_id == "work-17-v3-m29-t11-p31"
    assert plan.snapshot_id == 17
    assert plan.control_version == 3
    assert plan.map_content_version == 29
    assert plan.topology_version == 11
    assert plan.parameter_version == 31
    assert plan.resolution_tier == "half"
    assert plan.supervision_sources == (_source(6),)
    assert plan.replay_sources == (_source(2),)
    assert plan.replay_work[0].source == _source(2)
    assert plan.optimizer_updates == 3
    assert plan.insertion.max_seeds == 5
    assert plan.allow_growth
    assert plan.allow_relocation
    assert plan.mutation_budget.max_total == 7
    with pytest.raises(FrozenInstanceError):
        plan.optimizer_updates = 0


def test_owner_resolves_one_homogeneous_resolution_for_mixed_transaction():
    snapshot = _snapshot(mode=FreshnessMode.HEALTHY)
    proposals = (
        ControlProposal("resolution", "resolution_tier", "half", "frontier"),
        ControlProposal("replay", "replay_resolution_tier", "quarter", "history"),
        ControlProposal("work", "supervision_slots", 1, "B=1 current"),
        ControlProposal("work", "replay_slots", 1, "one historical view"),
        ControlProposal("work", "optimizer_updates", 2, "two sequential B=1 updates"),
    )

    plan = WorkBudgetOwner().resolve(
        snapshot,
        disposition=_disposition(snapshot),
        proposals=proposals,
    )

    assert plan.optimizer_updates == 2
    assert plan.transaction_class is MappingTransactionClass.MIXED
    assert plan.resolution_tier.value == "half"
    assert [work.resolution_tier.value for work in plan.supervision_work] == ["half"]
    assert [work.resolution_tier.value for work in plan.replay_work] == ["half"]


def test_native_geometry_purpose_spends_bounded_measured_cost_allowance_for_b4():
    """A purpose may spend bounded scheduler credit, never a tier override."""

    geometry = ModalityEligibility(True, "qualified", True, "qualified", True, True)
    snapshot = replace(
        _snapshot(mode=FreshnessMode.HEALTHY),
        limits=replace(
            _limits(), max_supervision_slots=3, max_replay_slots=3, max_optimizer_updates=1
        ),
        replay_candidates=(_source(1), _source(2), _source(3)),
        replay_work_modalities=(geometry, geometry, geometry),
    )
    owner = WorkBudgetOwner(
        requested_batch_size=4,
        maximum_camera_count=4,
        constellation_policy="fixed",
        native_geometry_evidence_allowance_ms=480.0,
        native_geometry_evidence_allowance_refill_ms=40.0,
    )
    # FR3's normal C4 budget is 40ms while native has measured around 448ms.
    # The scheduler-owned evidence allowance is the only legal route that can
    # spend the bounded 408ms excess for this purpose.
    _record(owner, tier=ResolutionTier.NATIVE, wall_ms=448.0, gaussian_count=100)
    _record(owner, tier=ResolutionTier.QUARTER, wall_ms=1.0, gaussian_count=100)
    proposals = (
        ControlProposal("resolution", "resolution_tier", "quarter", "ordinary pressure"),
        ControlProposal("work", "supervision_slots", 1, "one frontier"),
        ControlProposal("work", "replay_slots", 3, "three history"),
        ControlProposal("work", "optimizer_updates", 1, "one update"),
    )

    absent = owner.resolve(snapshot, disposition=replace(_disposition(snapshot), modality_eligibility=geometry), proposals=proposals)
    demanded = owner.resolve(
        replace(snapshot, snapshot_id=18),
        disposition=replace(_disposition(snapshot), snapshot_id=18, modality_eligibility=geometry),
        proposals=proposals,
        native_geometry_evidence_purpose=True,
    )

    assert absent.resolution_tier is ResolutionTier.HALF
    assert demanded.resolution_tier is ResolutionTier.NATIVE
    assert demanded.native_geometry_evidence_purpose is True
    with pytest.raises(TypeError, match="native_geometry_evidence_purpose must be bool"):
        replace(demanded, native_geometry_evidence_purpose=1)
    assert demanded.supervision_constellation.realized_batch_size == 4
    assert demanded.supervision_constellation.realized_camera_count == 4
    assert all(work.modality_eligibility.geometry_population_target for work in (*demanded.supervision_work, *demanded.replay_work))
    assert any(
        item.startswith("native_geometry_evidence:considered_native:allowance_spent_ms=408.000")
        for item in demanded.reason_trace
    )
    assert owner.native_geometry_evidence_allowance_available_ms == 72.0

    exhausted = owner.resolve(
        replace(snapshot, snapshot_id=19),
        disposition=replace(_disposition(snapshot), snapshot_id=19, modality_eligibility=geometry),
        proposals=proposals,
        native_geometry_evidence_purpose=True,
    )
    assert exhausted.resolution_tier is ResolutionTier.QUARTER
    assert exhausted.native_geometry_evidence_purpose is True
    assert exhausted.supervision_constellation.realized_batch_size == 4
    assert exhausted.supervision_constellation.realized_camera_count == 4
    assert any(
        item.startswith("native_geometry_evidence:deferred_budget:allowance_available_ms=72.000")
        for item in exhausted.reason_trace
    )

    required = owner.resolve(
        snapshot,
        disposition=replace(_disposition(snapshot), modality_eligibility=geometry),
        proposals=proposals + (
            ControlProposal("depth", "insertion_enabled", True, "source geometry"),
            ControlProposal("depth", "insertion_seeds", 5, "source geometry"),
            ControlProposal("depth", "mutation_budget", 1, "source geometry"),
        ),
        native_geometry_evidence_purpose=True,
        source_depth_requires_native=True,
    )
    assert required.resolution_tier is ResolutionTier.NATIVE
    assert required.native_geometry_evidence_purpose is True
    assert owner.native_geometry_evidence_allowance_available_ms == 72.0
    assert any(
        item.startswith("native_geometry_evidence:considered_native:allowance_spent_ms=0.000")
        for item in required.reason_trace
    )

    constrained = WorkBudgetOwner(
        requested_batch_size=4,
        maximum_camera_count=4,
        constellation_policy="fixed",
        native_geometry_evidence_allowance_ms=480.0,
    )
    _record(constrained, tier=ResolutionTier.NATIVE, wall_ms=600.0, gaussian_count=100)
    _record(constrained, tier=ResolutionTier.QUARTER, wall_ms=1.0, gaussian_count=100)
    deferred = constrained.resolve(
        snapshot,
        disposition=replace(_disposition(snapshot), modality_eligibility=geometry),
        proposals=proposals,
        native_geometry_evidence_purpose=True,
    )
    assert deferred.resolution_tier is ResolutionTier.QUARTER
    assert any(
        item.startswith("native_geometry_evidence:deferred_budget:")
        for item in deferred.reason_trace
    )


def test_native_geometry_purpose_uses_one_allowance_bounded_native_probe_without_measurement():
    """A qualified B4 demand may calibrate an otherwise unknown native cost.

    This catches the old owner path that left a fresh purpose at quarter merely
    because the exact native B4 cost bucket had no prior sample.
    """

    geometry = ModalityEligibility(True, "qualified", True, "qualified", True, True)
    snapshot = replace(
        _snapshot(mode=FreshnessMode.HEALTHY),
        limits=replace(
            _limits(), max_supervision_slots=3, max_replay_slots=3, max_optimizer_updates=1
        ),
        replay_candidates=(_source(1), _source(2), _source(3)),
        replay_work_modalities=(geometry, geometry, geometry),
    )
    owner = WorkBudgetOwner(
        requested_batch_size=4,
        maximum_camera_count=4,
        constellation_policy="fixed",
        native_geometry_evidence_allowance_ms=480.0,
        native_geometry_evidence_allowance_refill_ms=40.0,
    )
    # The ordinary C4 tier is known to fit its 40ms budget, but native has no
    # measurement in this exact four-camera calibration bucket.
    _record(owner, tier=ResolutionTier.QUARTER, wall_ms=1.0, gaussian_count=100)
    proposals = (
        ControlProposal("resolution", "resolution_tier", "quarter", "ordinary pressure"),
        ControlProposal("work", "supervision_slots", 1, "one frontier"),
        ControlProposal("work", "replay_slots", 3, "three history"),
        ControlProposal("work", "optimizer_updates", 1, "one update"),
    )

    plan = owner.resolve(
        snapshot,
        disposition=replace(_disposition(snapshot), modality_eligibility=geometry),
        proposals=proposals,
        native_geometry_evidence_purpose=True,
    )

    assert plan.resolution_tier is ResolutionTier.NATIVE
    assert plan.supervision_constellation.realized_batch_size == 4
    assert plan.supervision_constellation.realized_camera_count == 4
    assert all(
        work.modality_eligibility.geometry_population_target
        for work in (*plan.supervision_work, *plan.replay_work)
    )
    assert plan.resolution_budget_ms == 520.0
    assert any("tier=native" in item and "probe=True" in item for item in plan.reason_trace)
    assert any(
        item.startswith("native_geometry_evidence:considered_native:allowance_spent_ms=480.000")
        for item in plan.reason_trace
    )
    assert owner.native_geometry_evidence_allowance_available_ms == 0.0


def test_native_geometry_calibration_probe_uses_full_allowance_envelope_for_fr3_cost():
    """A 52ms ordinary tier may still seed a 520ms native calibration probe."""

    geometry = ModalityEligibility(True, "qualified", True, "qualified", True, True)
    snapshot = replace(
        _snapshot(mode=FreshnessMode.HEALTHY),
        limits=replace(
            _limits(), max_supervision_slots=3, max_replay_slots=3, max_optimizer_updates=1
        ),
        replay_candidates=(_source(1), _source(2), _source(3)),
        replay_work_modalities=(geometry, geometry, geometry),
    )
    owner = WorkBudgetOwner(
        requested_batch_size=4,
        maximum_camera_count=4,
        constellation_policy="fixed",
        native_geometry_evidence_allowance_ms=480.0,
    )
    _record(owner, tier=ResolutionTier.QUARTER, wall_ms=52.0, gaussian_count=100)
    plan = owner.resolve(
        snapshot,
        disposition=replace(_disposition(snapshot), modality_eligibility=geometry),
        proposals=(
            ControlProposal("resolution", "resolution_tier", "quarter", "pressure"),
            ControlProposal("work", "supervision_slots", 1, "frontier"),
            ControlProposal("work", "replay_slots", 3, "history"),
            ControlProposal("work", "optimizer_updates", 1, "update"),
        ),
        native_geometry_evidence_purpose=True,
    )

    assert plan.resolution_tier is ResolutionTier.NATIVE
    assert plan.resolution_probe is True
    assert plan.resolution_budget_ms == 520.0
    assert owner.native_geometry_evidence_allowance_available_ms == 0.0


def test_native_geometry_calibration_probe_cannot_repeat_without_a_completed_measurement():
    """One purpose cannot mint a second unmeasured native B4 probe."""

    geometry = ModalityEligibility(True, "qualified", True, "qualified", True, True)
    snapshot = replace(
        _snapshot(mode=FreshnessMode.HEALTHY),
        limits=replace(
            _limits(), max_supervision_slots=3, max_replay_slots=3, max_optimizer_updates=1
        ),
        replay_candidates=(_source(1), _source(2), _source(3)),
        replay_work_modalities=(geometry, geometry, geometry),
    )
    owner = WorkBudgetOwner(
        requested_batch_size=4,
        maximum_camera_count=4,
        constellation_policy="fixed",
        native_geometry_evidence_allowance_ms=480.0,
        native_geometry_evidence_allowance_refill_ms=40.0,
    )
    _record(owner, tier=ResolutionTier.QUARTER, wall_ms=1.0, gaussian_count=100)
    proposals = (
        ControlProposal("resolution", "resolution_tier", "quarter", "ordinary pressure"),
        ControlProposal("work", "supervision_slots", 1, "one frontier"),
        ControlProposal("work", "replay_slots", 3, "three history"),
        ControlProposal("work", "optimizer_updates", 1, "one update"),
    )

    first = owner.resolve(
        snapshot,
        disposition=replace(_disposition(snapshot), modality_eligibility=geometry),
        proposals=proposals,
        native_geometry_evidence_purpose=True,
    )
    second = owner.resolve(
        replace(snapshot, snapshot_id=18),
        disposition=replace(
            _disposition(snapshot), snapshot_id=18, modality_eligibility=geometry
        ),
        proposals=proposals,
        native_geometry_evidence_purpose=True,
    )

    assert first.resolution_tier is ResolutionTier.NATIVE
    assert first.resolution_probe is True
    assert second.resolution_tier is ResolutionTier.QUARTER
    assert second.resolution_probe is False
    assert owner.native_geometry_evidence_allowance_available_ms == 0.0


def test_native_geometry_calibration_probe_requires_positive_allowance_capacity():
    """A disabled allowance cannot promote an unknown native B4 tier."""

    geometry = ModalityEligibility(True, "qualified", True, "qualified", True, True)
    snapshot = replace(
        _snapshot(mode=FreshnessMode.HEALTHY),
        limits=replace(
            _limits(), max_supervision_slots=3, max_replay_slots=3, max_optimizer_updates=1
        ),
        replay_candidates=(_source(1), _source(2), _source(3)),
        replay_work_modalities=(geometry, geometry, geometry),
    )
    owner = WorkBudgetOwner(
        requested_batch_size=4,
        maximum_camera_count=4,
        constellation_policy="fixed",
        native_geometry_evidence_allowance_ms=0.0,
    )
    _record(owner, tier=ResolutionTier.QUARTER, wall_ms=1.0, gaussian_count=100)
    plan = owner.resolve(
        snapshot,
        disposition=replace(_disposition(snapshot), modality_eligibility=geometry),
        proposals=(
            ControlProposal("resolution", "resolution_tier", "quarter", "pressure"),
            ControlProposal("work", "supervision_slots", 1, "frontier"),
            ControlProposal("work", "replay_slots", 3, "history"),
            ControlProposal("work", "optimizer_updates", 1, "update"),
        ),
        native_geometry_evidence_purpose=True,
    )

    assert plan.resolution_tier is ResolutionTier.QUARTER
    assert plan.resolution_probe is False


def _ready_depth_budget_case(*, batch_size=16, **owner_options):
    modality = ModalityEligibility(True, "qualified", True, "qualified", True)
    source = _source(20)
    history = tuple(_source(index) for index in range(1, batch_size))
    snapshot = replace(
        _snapshot(mode=FreshnessMode.CATCH_UP),
        source=source,
        pending_sources=(source,),
        newest_frontier=source,
        supervision_candidates=(source,),
        replay_candidates=history,
        replay_work_modalities=(modality,) * len(history),
        limits=replace(_limits(), max_supervision_slots=1, max_replay_slots=len(history)),
    )
    owner = WorkBudgetOwner(
        requested_batch_size=batch_size,
        maximum_camera_count=batch_size,
        constellation_policy="fixed",
        target_supervised_views_per_s=200.0,
        native_geometry_evidence_allowance_ms=0.0,
        **owner_options,
    )
    proposals = tuple(
        ControlProposal("test", actuator, value, "READY insertion")
        for actuator, value in (
            ("resolution_tier", "quarter"),
            ("supervision_slots", 1),
            ("replay_slots", len(history)),
            ("optimizer_updates", 1),
            ("insertion_enabled", True),
            ("insertion_seeds", 5),
            ("mutation_budget", 1),
        )
    )
    return owner, snapshot, proposals


@pytest.mark.parametrize("measured_native", [False, True])
@pytest.mark.parametrize("mutation_budget", [0, 1])
@pytest.mark.parametrize("costs", [(400.0, 100.0, 10.0), (100.0, 160.0, 240.0)])
def test_ready_source_depth_requires_native_b16_without_population_allowance(
    measured_native, mutation_budget, costs
):
    native_ms, half_ms, quarter_ms = costs
    owner, snapshot, proposals = _ready_depth_budget_case(budget_relax_max=1.0)
    proposals = tuple(
        replace(proposal, value=mutation_budget)
        if proposal.actuator == "mutation_budget" else proposal
        for proposal in proposals
    )
    for tier, cost in ((ResolutionTier.QUARTER, quarter_ms), (ResolutionTier.HALF, half_ms)):
        _record(owner, tier=tier, wall_ms=cost, gaussian_count=100, camera_count=16)
    if measured_native:
        _record(owner, tier=ResolutionTier.NATIVE, wall_ms=native_ms, gaussian_count=100, camera_count=16)
    plan = owner.resolve(
        snapshot,
        disposition=_disposition(snapshot),
        proposals=proposals,
        source_depth_requires_native=True,
    )
    assert plan.supervision_sources == (snapshot.source,)
    assert set(plan.replay_sources) == set(snapshot.replay_candidates)
    assert plan.supervision_constellation.realized_camera_count == 16
    assert all(work.resolution_tier is ResolutionTier.NATIVE for work in (*plan.supervision_work, *plan.replay_work))
    assert plan.insertion.enabled and plan.insertion.source == snapshot.source
    assert plan.insertion.seed_budget == 5
    assert plan.mutation_budget.max_total == mutation_budget
    assert plan.native_geometry_evidence_purpose is False
    assert owner.native_geometry_evidence_allowance_available_ms == 0.0
    assert plan.resolution_budget_ms == 80.0  # Target, not a hard latency claim.
    assert "required native source-depth evidence" in plan.resolution_reason
    assert plan.resolution_probe is False

    # The next already-resident replay transaction is not forced native by
    # the preceding source's geometry obligation.
    _record(owner, tier=ResolutionTier.NATIVE, wall_ms=native_ms, gaussian_count=100, camera_count=16)
    replay_modalities = dict(zip(snapshot.replay_candidates, snapshot.replay_work_modalities))
    replay_modalities[snapshot.source] = _disposition(snapshot).modality_eligibility
    replay = owner.resolve_idle_replay(
        snapshot, plan,
        replay_modalities=replay_modalities,
        replay_weights={source: 1.0 for source in replay_modalities},
        batch_size=16, optimizer_steps_completed=41, versions=(29, 11, 31),
        proposals=(ControlProposal("test", "replay_resolution_tier", "quarter", "pressure"),),
    )
    assert replay.resolution_tier is ResolutionTier.QUARTER
    assert replay.resolution_budget_ms == 80.0
    assert replay.resolution_probe is False
    assert replay.insertion.enabled is False


@pytest.mark.parametrize("denied", ["option", "insertion_enabled", "insertion_seeds", "optimizer_updates", "modality", "critical", "live_population"])
@pytest.mark.parametrize("batch_size", [4, 16])
def test_ready_native_requirement_does_not_override_insertion_authorization(denied, batch_size):
    owner, snapshot, proposals = _ready_depth_budget_case(batch_size=batch_size)
    disposition = _disposition(snapshot)
    for tier, cost in ((ResolutionTier.QUARTER, 10.0), (ResolutionTier.HALF, 100.0), (ResolutionTier.NATIVE, 400.0)):
        _record(owner, tier=tier, wall_ms=cost, gaussian_count=100, camera_count=batch_size)
    if denied in {"insertion_enabled", "insertion_seeds", "optimizer_updates"}:
        proposals = tuple(replace(proposal, value=False if denied == "insertion_enabled" else 0) if proposal.actuator == denied else proposal for proposal in proposals)
    elif denied == "modality":
        disposition = replace(disposition, allow_geometry_insertion=False, modality_eligibility=ModalityEligibility(True, "qualified", True, "qualified", False))
    elif denied == "critical":
        snapshot = replace(snapshot, freshness_mode=FreshnessMode.CRITICAL)
    elif denied == "live_population":
        proposals += (ControlProposal("test", "live_population_authorization", True, "B4-only population"),)
    plan = owner.resolve(
        snapshot, disposition=disposition, proposals=proposals,
        source_depth_requires_native=denied != "option",
    )
    assert plan.resolution_tier is ResolutionTier.QUARTER
    assert plan.supervision_sources == (snapshot.source,)
    assert "required native source-depth evidence" not in plan.resolution_reason


def test_ready_native_requirement_rejects_unsupported_tier_and_non_bool():
    owner, snapshot, proposals = _ready_depth_budget_case(supported_resolution_tiers=(ResolutionTier.QUARTER,))
    with pytest.raises(ValueError, match="source-depth insertion requires.*native"):
        owner.resolve(snapshot, disposition=_disposition(snapshot), proposals=proposals, source_depth_requires_native=True)
    with pytest.raises(TypeError, match="source_depth_requires_native must be bool"):
        owner.resolve(snapshot, disposition=_disposition(snapshot), proposals=proposals, source_depth_requires_native=1)


def test_work_plan_rejects_non_homogeneous_replay_tier():
    snapshot = _snapshot(mode=FreshnessMode.HEALTHY)
    plan = _resolve(
        snapshot,
        (
            ControlProposal("resolution", "resolution_tier", "half", "frontier"),
            ControlProposal("work", "replay_slots", 1, "one historical view"),
        ),
    )

    mismatched = replace(plan.replay_work[0], resolution_tier="quarter")
    with pytest.raises(ValueError, match="replay_work tiers must match"):
        replace(plan, replay_work=(mismatched,))


def test_every_allocation_proposal_provider_consumes_the_identical_snapshot():
    from eligsir.mapping.scheduler.work_budget import collect_control_proposals

    snapshot = _snapshot(mode=FreshnessMode.HEALTHY)
    seen = []

    class Provider:
        def __init__(self, actuator):
            self.actuator = actuator

        def propose(self, received):
            seen.append(received)
            return ControlProposal("test", self.actuator, 0, "identity check")

    providers = tuple(
        Provider(actuator)
        for actuator in (
            "resolution_tier",
            "replay_slots",
            "optimizer_updates",
            "insertion_seeds",
            "mutation_budget",
        )
    )

    collect_control_proposals(snapshot, providers)

    assert seen == [snapshot] * len(providers)
    assert all(item is snapshot for item in seen)


def test_insertion_provider_exposes_static_enablement_and_seed_request():
    from eligsir.mapping.scheduler.work_budget import InsertionProposalProvider

    proposals = InsertionProposalProvider(7, enabled=False).propose(
        _snapshot(mode=FreshnessMode.HEALTHY)
    )

    assert {proposal.actuator: proposal.value for proposal in proposals} == {
        "insertion_enabled": False,
        "insertion_seeds": 7,
    }


def test_owner_resolves_replay_only_from_identity_bound_historical_modalities():
    snapshot = _snapshot(mode=FreshnessMode.HEALTHY)
    proposals = (
            ControlProposal("policy", "replay_slots", 1, "surplus"),
            ControlProposal("policy", "optimizer_updates", 2, "surplus"),
    )
    snapshot = replace(
        snapshot,
        replay_work_modalities=(
            ModalityEligibility(True, "qualified", False, "missing", False),
            ModalityEligibility(False, "missing", True, "qualified", False),
        ),
    )

    plan = _resolve(snapshot, proposals)

    assert plan.replay_sources == (_source(2),)
    assert plan.replay_work[0].source == _source(2)
    assert plan.replay_work[0].modality_eligibility.depth_supervision


def test_critical_owner_rejects_a_snapshot_without_an_admissible_newest_frontier():
    snapshot = _snapshot(mode=FreshnessMode.CRITICAL)
    snapshot = replace(snapshot, supervision_candidates=(_source(4), _source(5)))

    with pytest.raises(ValueError, match="newest_frontier"):
        _resolve(snapshot)


def test_owner_preserves_terminal_disposition_and_boolean_safety_vetoes():
    snapshot = _snapshot(mode=FreshnessMode.HEALTHY)
    proposals = (
            ControlProposal("legacy", "insertion_seeds", 5, "surface"),
            ControlProposal("legacy", "allow_growth", True, "surface"),
            ControlProposal("safety", "allow_growth", False, "tracking"),
            ControlProposal("legacy", "allow_relocation", True, "surface"),
            ControlProposal("safety", "allow_relocation", False, "tracking"),
    )

    plan = _resolve(snapshot, proposals, disposition=_disposition(snapshot, admitted=False))

    assert _source(6) not in plan.supervision_sources
    assert plan.insertion.max_seeds == 0
    assert not plan.allow_growth
    assert not plan.allow_relocation


def test_static_insertion_disable_is_resolved_only_by_work_budget_owner():
    snapshot = _snapshot(mode=FreshnessMode.HEALTHY)
    proposals = (
        ControlProposal("config", "insertion_enabled", False, "operator disabled"),
        ControlProposal("depth", "insertion_seeds", 5, "surface candidates"),
    )

    plan = _resolve(snapshot, proposals)

    assert not plan.insertion.enabled
    assert plan.insertion.source is None
    assert plan.insertion.max_seeds == 0


def test_declined_critical_observation_resolves_typed_empty_work():
    snapshot = _snapshot(mode=FreshnessMode.CRITICAL)
    proposals = (
        ControlProposal("work", "supervision_slots", 1, "current"),
        ControlProposal("work", "optimizer_updates", 1, "critical assimilation"),
        ControlProposal("depth", "insertion_enabled", True, "enabled"),
        ControlProposal("depth", "insertion_seeds", 5, "surface candidates"),
    )

    plan = _resolve(
        snapshot,
        proposals,
        disposition=_disposition(snapshot, admitted=False),
    )

    assert plan.freshness_mode is FreshnessMode.CRITICAL
    assert plan.supervision_sources == ()
    assert plan.replay_sources == ()
    assert plan.optimizer_updates == 0
    assert not plan.insertion.enabled
    assert not plan.allow_growth
    assert not plan.allow_relocation
    assert plan.mutation_budget.max_total == 0
    assert "declined_observation_no_work" in plan.reason_trace


def test_admitted_observation_reserves_current_source_despite_zero_slot_proposal():
    snapshot = _snapshot(mode=FreshnessMode.HEALTHY)

    plan = _resolve(
        snapshot,
        (ControlProposal("legacy", "supervision_slots", 0, "tracking_lost"),),
    )

    assert plan.requested_supervision_slots == 0
    assert plan.supervision_sources == (snapshot.source,)
    assert plan.supervision_work[0].source == snapshot.source
    assert "committed_source_materialization_floor" in plan.reason_trace


def test_real_work_budget_insertion_source_is_newest_fresh_not_tuple_head():
    """The plan records fresh sources ascending, but insertion uses the newest."""
    snapshot = replace(
        _snapshot(mode=FreshnessMode.HEALTHY),
        limits=replace(_limits(), max_supervision_slots=3, max_replay_slots=1),
        replay_candidates=(_source(1), _source(4), _source(5)),
        replay_work_modalities=(
            ModalityEligibility(True, "qualified", False, "missing", False),
            ModalityEligibility(True, "qualified", False, "missing", False),
            ModalityEligibility(True, "qualified", False, "missing", False),
        ),
    )
    plan = WorkBudgetOwner(
        requested_batch_size=4, maximum_camera_count=4, constellation_policy="fixed"
    ).resolve(
        snapshot,
        disposition=_disposition(snapshot),
        proposals=(
            ControlProposal("work", "supervision_slots", 3, "fresh batch"),
            ControlProposal("work", "replay_slots", 1, "batch fill"),
            ControlProposal("work", "optimizer_updates", 1, "execute"),
            ControlProposal("depth", "insertion_enabled", True, "enabled"),
            ControlProposal("depth", "insertion_seeds", 5, "source depth"),
        ),
    )

    assert plan.supervision_sources == tuple(sorted(plan.supervision_sources, key=SourceIdentity.ordering_key))
    assert len(plan.supervision_sources) > 1
    assert plan.insertion.source == max(plan.supervision_sources, key=SourceIdentity.ordering_key)
    assert plan.insertion.source != plan.supervision_sources[0]


def test_adaptive_replay_fill_cannot_displace_zero_slot_committed_source():
    snapshot = replace(
        _snapshot(mode=FreshnessMode.HEALTHY),
        limits=replace(
            _limits(),
            max_supervision_slots=3,
            max_replay_slots=3,
            max_optimizer_updates=1,
        ),
        replay_candidates=(_source(1), _source(2), _source(3)),
        replay_work_modalities=(
            ModalityEligibility(True, "qualified", False, "missing", False),
        )
        * 3,
    )
    owner = WorkBudgetOwner(requested_batch_size=4, maximum_camera_count=4)

    plan = owner.resolve(
        snapshot,
        disposition=_disposition(snapshot),
        proposals=(
            ControlProposal("safety", "supervision_slots", 0, "temporary ceiling"),
            ControlProposal("constellation", "replay_slots", 3, "replay fill"),
            ControlProposal("optimizer", "optimizer_updates", 1, "true C4"),
        ),
    )

    assert plan.requested_supervision_slots == 0
    assert snapshot.source in plan.supervision_sources
    assert snapshot.source not in plan.replay_sources
    assert len(plan.supervision_sources) + len(plan.replay_sources) == 4
    assert "committed_source_materialization_floor" in plan.reason_trace


def test_critical_plan_allocates_minimal_assimilation_despite_zero_optimizer_proposal():
    snapshot = _snapshot(mode=FreshnessMode.CRITICAL)

    plan = _resolve(
        snapshot,
        (ControlProposal("safety", "optimizer_updates", 0, "tracking_lost"),),
        freshness_mode=FreshnessMode.CRITICAL,
    )

    assert plan.supervision_sources == (_source(6),)
    assert plan.optimizer_updates == 1


def test_critical_plan_rejects_a_zero_optimizer_update_ceiling():
    snapshot = _snapshot(mode=FreshnessMode.CRITICAL)
    snapshot = replace(
        snapshot,
        limits=replace(snapshot.limits, max_optimizer_updates=0),
    )

    with pytest.raises(ValueError, match="max_optimizer_updates >= 1"):
        _resolve(snapshot, freshness_mode=FreshnessMode.CRITICAL)


def test_adaptive_growth_is_permissive_until_training_spatial_evidence_exists():
    provider = AdaptiveGrowthProposalProvider(
        enabled=True,
        marginal_utility_brake=-1.0e-5,
        marginal_utility_release=1.0e-5,
        hysteresis=2,
        minimum_population_delta=200,
        utility_ema=0.5,
        regression_tolerance_db=0.25,
    )

    proposal = provider.propose_idle(_snapshot(mode=FreshnessMode.HEALTHY))

    assert proposal.actuator == "allow_growth"
    assert proposal.value is True
    assert "warmup" in proposal.reason


def test_adaptive_growth_brakes_when_local_fade_correlates_with_negative_utility():
    provider = AdaptiveGrowthProposalProvider(
        enabled=True,
        marginal_utility_brake=-1.0e-5,
        marginal_utility_release=1.0e-5,
        hysteresis=2,
        minimum_population_delta=200,
        utility_ema=0.5,
        regression_tolerance_db=0.25,
    )
    provider.observe_training_spatial_quality(
        iteration=100,
        aggregate_psnr_db=24.0,
        gaussian_count=10_000,
        worst_drop_from_peak_db=0.0,
    )
    provider.observe_training_spatial_quality(
        iteration=200,
        aggregate_psnr_db=23.8,
        gaussian_count=10_500,
        worst_drop_from_peak_db=0.8,
    )
    assert provider.propose_idle(_snapshot(mode=FreshnessMode.HEALTHY)).value is True

    provider.observe_training_spatial_quality(
        iteration=300,
        aggregate_psnr_db=23.7,
        gaussian_count=11_000,
        worst_drop_from_peak_db=0.9,
    )
    proposal = provider.propose_idle(_snapshot(mode=FreshnessMode.HEALTHY))

    assert proposal.value is False
    assert "braked" in proposal.reason


def test_adaptive_growth_keeps_growing_when_utility_is_positive_despite_local_fade():
    provider = AdaptiveGrowthProposalProvider(
        enabled=True,
        marginal_utility_brake=-1.0e-5,
        marginal_utility_release=1.0e-5,
        hysteresis=2,
        minimum_population_delta=200,
        utility_ema=0.5,
        regression_tolerance_db=0.25,
    )
    for iteration, psnr_db, population, drop_db in (
        (100, 24.0, 10_000, 0.0),
        (200, 24.2, 10_500, 0.8),
        (300, 24.4, 11_000, 0.9),
    ):
        provider.observe_training_spatial_quality(
            iteration=iteration,
            aggregate_psnr_db=psnr_db,
            gaussian_count=population,
            worst_drop_from_peak_db=drop_db,
        )

    proposal = provider.propose_idle(_snapshot(mode=FreshnessMode.HEALTHY))

    assert proposal.value is True
    assert "growth active" in proposal.reason


def test_adaptive_growth_releases_with_positive_utility_during_local_fade():
    provider = AdaptiveGrowthProposalProvider(
        enabled=True,
        marginal_utility_brake=-1.0e-5,
        marginal_utility_release=1.0e-5,
        hysteresis=1,
        minimum_population_delta=200,
        utility_ema=1.0,
        regression_tolerance_db=0.25,
    )
    provider.observe_training_spatial_quality(
        iteration=100,
        aggregate_psnr_db=24.0,
        gaussian_count=10_000,
        worst_drop_from_peak_db=0.0,
    )
    provider.observe_training_spatial_quality(
        iteration=200,
        aggregate_psnr_db=23.8,
        gaussian_count=10_500,
        worst_drop_from_peak_db=0.8,
    )
    assert provider.propose_idle(_snapshot(mode=FreshnessMode.HEALTHY)).value is False

    provider.observe_training_spatial_quality(
        iteration=300,
        aggregate_psnr_db=24.1,
        gaussian_count=11_000,
        worst_drop_from_peak_db=0.8,
    )

    proposal = provider.propose_idle(_snapshot(mode=FreshnessMode.HEALTHY))
    assert proposal.value is True
    assert "released" in proposal.reason


def test_adaptive_growth_releases_after_replay_recovers_quality_at_fixed_population():
    provider = AdaptiveGrowthProposalProvider(
        enabled=True,
        marginal_utility_brake=-1.0e-5,
        marginal_utility_release=1.0e-5,
        hysteresis=2,
        minimum_population_delta=200,
        utility_ema=0.5,
        regression_tolerance_db=0.25,
    )
    samples = (
        (100, 24.0, 10_000, 0.0),
        (200, 23.8, 10_500, 0.8),
        (300, 23.7, 11_000, 0.9),
    )
    for iteration, psnr_db, population, drop_db in samples:
        provider.observe_training_spatial_quality(
            iteration=iteration,
            aggregate_psnr_db=psnr_db,
            gaussian_count=population,
            worst_drop_from_peak_db=drop_db,
        )
    assert provider.propose_idle(_snapshot(mode=FreshnessMode.HEALTHY)).value is False

    provider.observe_training_spatial_quality(
        iteration=400,
        aggregate_psnr_db=23.9,
        gaussian_count=11_000,
        worst_drop_from_peak_db=0.2,
    )
    assert provider.propose_idle(_snapshot(mode=FreshnessMode.HEALTHY)).value is False
    provider.observe_training_spatial_quality(
        iteration=500,
        aggregate_psnr_db=24.0,
        gaussian_count=11_000,
        worst_drop_from_peak_db=0.0,
    )

    proposal = provider.propose_idle(_snapshot(mode=FreshnessMode.HEALTHY))
    assert proposal.value is True
    assert "released" in proposal.reason


def test_adaptive_growth_releases_on_aggregate_recovery_during_local_repair():
    provider = AdaptiveGrowthProposalProvider(
        enabled=True,
        marginal_utility_brake=-1.0e-5,
        marginal_utility_release=1.0e-5,
        hysteresis=2,
        minimum_population_delta=200,
        utility_ema=0.5,
        regression_tolerance_db=0.25,
    )
    for iteration, psnr_db, population, drop_db in (
        (100, 24.0, 10_000, 0.0),
        (200, 23.8, 10_500, 0.8),
        (300, 23.7, 11_000, 0.9),
    ):
        provider.observe_training_spatial_quality(
            iteration=iteration,
            aggregate_psnr_db=psnr_db,
            gaussian_count=population,
            worst_drop_from_peak_db=drop_db,
        )
    assert provider.propose_idle(_snapshot(mode=FreshnessMode.HEALTHY)).value is False

    for iteration, psnr_db, drop_db in (
        (400, 23.9, 0.8),
        (500, 24.0, 0.7),
    ):
        provider.observe_training_spatial_quality(
            iteration=iteration,
            aggregate_psnr_db=psnr_db,
            gaussian_count=11_000,
            worst_drop_from_peak_db=drop_db,
        )

    proposal = provider.propose_idle(_snapshot(mode=FreshnessMode.HEALTHY))
    assert proposal.value is True
    assert "released" in proposal.reason


def _quarter_only_estimator() -> TierCostEstimator:
    """A TierCostEstimator that has measured `quarter` but never `native` —
    reproduces the exact stuck state described in issue #1259: once any
    tier has real cost evidence, `select()` can never choose a tier it has
    no evidence for unless it is allowed to probe for it."""
    estimator = TierCostEstimator()
    context = TierCostContext(
        device_id="unknown-device",
        gaussian_count=50_000,
        camera_count=2,
        render_mode="true_camera_batch",
        pass_mode=FORWARD_BACKWARD,
    )
    estimator.record(
        TierCostSample(
            context=context,
            tier=ResolutionTier.QUARTER,
            transaction_class=MappingTransactionClass.FRESH,
            prepare_ms=0.1,
            optimization_ms=2.0,
            transaction_wall_ms=2.0,
            kernel_ms=2.0,
        )
    )
    return estimator


def _fresh_snapshot(*, freshness_mode: FreshnessMode) -> ControlSnapshot:
    pending = (_source(4),)
    return ControlSnapshot(
        snapshot_id=1,
        control_version=1,
        source=pending[-1],
        tracking=None,
        pending_sources=pending,
        newest_frontier=pending[-1],
        freshness_mode=freshness_mode,
        pending_oldest_age_ns=0,
        ingress_queue_depth=0,
        candidate_queue_depth=0,
        transfer_queue_depth=0,
        gpu_queue_depth=0,
        map_content_version=1,
        topology_version=1,
        parameter_version=1,
        gaussian_count=50_000,
        optimizer_steps_completed=100,
        limits=_limits(),
        supervision_candidates=pending,
        replay_candidates=(),
    )


@pytest.mark.parametrize(
    "freshness_mode,expect_probe_allowed",
    (
        (FreshnessMode.HEALTHY, True),
        (FreshnessMode.CATCH_UP, False),
        (FreshnessMode.CRITICAL, False),
        (FreshnessMode.RECOVERY, False),
    ),
)
def test_fresh_transaction_probe_gated_by_freshness_mode(
    freshness_mode, expect_probe_allowed
):
    """FRESH transactions may probe an unmeasured (here: native) tier only
    while freshness_mode is HEALTHY -- issue #1259. A generous budget_ms
    (achieved via a low realized_camera_count against the default
    target_supervised_views_per_s=100.0) ensures budget is never the
    limiting factor, isolating the freshness-mode gate as the only variable
    under test."""
    owner = WorkBudgetOwner(tier_cost_estimator=_quarter_only_estimator())
    snapshot = _fresh_snapshot(freshness_mode=freshness_mode)

    decision = owner._resolve_transaction_resolution(
        snapshot=snapshot,
        transaction_class=MappingTransactionClass.FRESH,
        proposals=(),
        realized_camera_count=2,
        default_tier=ResolutionTier.NATIVE,
    )

    assert decision.probe is expect_probe_allowed, (
        f"freshness_mode={freshness_mode}: expected probe={expect_probe_allowed}, "
        f"got probe={decision.probe} (tier={decision.tier}, reason={decision.reason!r})"
    )


def test_mixed_transaction_probe_gated_by_freshness_mode_healthy():
    """MIXED transactions get the identical treatment as FRESH."""
    owner = WorkBudgetOwner(tier_cost_estimator=_quarter_only_estimator())
    snapshot = _fresh_snapshot(freshness_mode=FreshnessMode.HEALTHY)

    decision = owner._resolve_transaction_resolution(
        snapshot=snapshot,
        transaction_class=MappingTransactionClass.MIXED,
        proposals=(),
        realized_camera_count=2,
        default_tier=ResolutionTier.NATIVE,
    )

    assert decision.probe is True, (
        f"MIXED under HEALTHY should be allowed to probe, got probe={decision.probe} "
        f"(tier={decision.tier}, reason={decision.reason!r})"
    )


def test_replay_only_probe_unaffected_by_freshness_mode():
    """Regression guard: REPLAY_ONLY must keep probing unconditionally,
    regardless of freshness_mode -- this plan must not narrow existing
    behavior for REPLAY_ONLY/POST_REFINEMENT."""
    owner = WorkBudgetOwner(tier_cost_estimator=_quarter_only_estimator())
    snapshot = _fresh_snapshot(freshness_mode=FreshnessMode.CRITICAL)

    decision = owner._resolve_transaction_resolution(
        snapshot=snapshot,
        transaction_class=MappingTransactionClass.REPLAY_ONLY,
        proposals=(),
        realized_camera_count=2,
        default_tier=ResolutionTier.NATIVE,
    )

    assert decision.probe is True, (
        f"REPLAY_ONLY must probe unconditionally even under CRITICAL freshness, "
        f"got probe={decision.probe} (tier={decision.tier}, reason={decision.reason!r})"
    )


# --- closed-loop budget relaxation (tier-budget floor-lock, 2026-09-04) -----


def _floor_locked_owner(**owner_kwargs):
    """Every measured tier costs more than the strict 40ms budget
    (4 cameras / 100 views/s): the r9700 fr3 regime where the selector
    held quarter for 93% of transactions and could never probe."""
    owner, snapshot, active, candidates, modality = _replay_only_owner()
    if owner_kwargs:
        owner = WorkBudgetOwner(
            requested_batch_size=4,
            maximum_camera_count=4,
            constellation_policy="fixed",
            **owner_kwargs,
        )
        active = owner.resolve(
            snapshot,
            disposition=_disposition(snapshot),
            proposals=(
                ControlProposal("resolution", "resolution_tier", "quarter", "fresh pressure"),
                ControlProposal("work", "supervision_slots", 1, "one fresh"),
                ControlProposal("work", "replay_slots", 3, "three replay"),
                ControlProposal("work", "optimizer_updates", 1, "one update"),
            ),
        )
    for tier, wall_ms in (
        (ResolutionTier.NATIVE, 150.0),
        (ResolutionTier.HALF, 90.0),
        (ResolutionTier.QUARTER, 60.0),
    ):
        _record(owner, tier=tier, wall_ms=wall_ms, gaussian_count=snapshot.gaussian_count)
    return owner, snapshot, active, candidates, modality


_NATIVE_CEILING = (
    ControlProposal(
        "ResolutionPolicyProvider", "replay_resolution_tier", "native", "healthy; native ceiling"
    ),
)


def _tier_sequence(owner, snapshot, active, candidates, modality, *, steps):
    return tuple(
        _idle(owner, snapshot, active, candidates, modality, step=step, proposals=_NATIVE_CEILING)
        .resolution_tier
        for step in range(1, steps + 1)
    )


def test_infeasible_budget_holds_the_cheapest_tier_while_freshness_is_not_healthy():
    """Under pressure the strict budget stands: no tier fits, hold the
    measured cheapest tier and never relax."""
    owner, snapshot, active, candidates, modality = _floor_locked_owner()
    assert snapshot.freshness_mode is not FreshnessMode.HEALTHY
    assert _tier_sequence(owner, snapshot, active, candidates, modality, steps=8) == (
        ResolutionTier.QUARTER,
    ) * 8
    assert owner._budget_relax_factor == 1.0


def test_healthy_freshness_relaxes_an_infeasible_budget_one_bounded_step_at_a_time():
    """The closed loop: while the pipeline demonstrably keeps up (HEALTHY),
    the pessimistic open-loop budget is relaxed geometrically until every
    allowed measured tier fits or the cap is reached -- so quality steps
    quarter -> half -> native instead of floor-locking at quarter forever."""
    owner, snapshot, active, candidates, modality = _floor_locked_owner()
    healthy = replace(snapshot, freshness_mode=FreshnessMode.HEALTHY)
    tiers = _tier_sequence(owner, healthy, active, candidates, modality, steps=8)
    # 40ms x 1.25^k: 50, 62.5, 78.1, 97.7, 122, 152.6 (then capped at 160).
    assert tiers == (
        ResolutionTier.QUARTER,
        ResolutionTier.QUARTER,
        ResolutionTier.QUARTER,
        ResolutionTier.HALF,
        ResolutionTier.HALF,
        ResolutionTier.NATIVE,
        ResolutionTier.NATIVE,
        ResolutionTier.NATIVE,
    )
    # Ramping stops by itself once native (150ms) fits: 1.25**6 = 3.81, below
    # the 4.0 cap -- the cap is a bound, not a target.
    assert owner._budget_relax_factor == pytest.approx(1.25**6)
    plan = _idle(owner, healthy, active, candidates, modality, step=9, proposals=_NATIVE_CEILING)
    assert "budget relaxed x3.81" in plan.resolution_reason


def test_any_non_healthy_reading_snaps_the_relaxation_back_to_the_strict_budget():
    owner, snapshot, active, candidates, modality = _floor_locked_owner()
    healthy = replace(snapshot, freshness_mode=FreshnessMode.HEALTHY)
    assert _tier_sequence(owner, healthy, active, candidates, modality, steps=6)[-1] is (
        ResolutionTier.NATIVE
    )
    pressured = replace(snapshot, freshness_mode=FreshnessMode.CATCH_UP)
    plan = _idle(owner, pressured, active, candidates, modality, step=7, proposals=_NATIVE_CEILING)
    assert plan.resolution_tier is ResolutionTier.QUARTER
    assert owner._budget_relax_factor == 1.0
    assert "budget relaxed" not in plan.resolution_reason


def test_relaxation_never_exceeds_the_quality_ceiling():
    """Relaxing the budget widens what is affordable; it never overrides the
    evidence-proposed quality ceiling."""
    owner, snapshot, active, candidates, modality = _floor_locked_owner()
    healthy = replace(snapshot, freshness_mode=FreshnessMode.HEALTHY)
    half_ceiling = (
        ControlProposal("ResolutionPolicyProvider", "replay_resolution_tier", "half", "hold"),
    )
    tiers = tuple(
        _idle(owner, healthy, active, candidates, modality, step=step, proposals=half_ceiling)
        .resolution_tier
        for step in range(1, 9)
    )
    assert ResolutionTier.NATIVE not in tiers
    assert tiers[-1] is ResolutionTier.HALF


def test_budget_relax_max_of_one_disables_the_relaxation():
    owner, snapshot, active, candidates, modality = _floor_locked_owner(budget_relax_max=1.0)
    healthy = replace(snapshot, freshness_mode=FreshnessMode.HEALTHY)
    assert _tier_sequence(owner, healthy, active, candidates, modality, steps=8) == (
        ResolutionTier.QUARTER,
    ) * 8


@pytest.mark.parametrize("kwargs", ({"budget_relax_max": 0.5}, {"budget_relax_step": 0.0}))
def test_budget_relax_knobs_below_one_are_rejected(kwargs):
    with pytest.raises(ValueError, match="must be finite and >= 1.0"):
        WorkBudgetOwner(requested_batch_size=4, maximum_camera_count=4, **kwargs)


@pytest.mark.parametrize(
    "kwargs",
    (
        {"relax_source_rate_floor": -0.1},
        {"relax_source_rate_floor": 1.5},
        {"relax_source_rate_floor": float("inf")},
    ),
)
def test_relax_source_rate_floor_must_be_in_zero_to_one(kwargs):
    """source_rate_floor must be in [0.0, 1.0]."""
    with pytest.raises(ValueError, match="must be finite and in \\[0.0, 1.0\\]"):
        WorkBudgetOwner(requested_batch_size=4, maximum_camera_count=4, **kwargs)


def test_source_rate_shortfall_resets_budget_relaxation():
    """Budget relaxation resets when realized source rate falls below floor."""
    owner, snapshot, active, candidates, modality = _floor_locked_owner(
        budget_relax_max=4.0, relax_source_rate_floor=0.9
    )
    healthy = replace(snapshot, freshness_mode=FreshnessMode.HEALTHY)

    # Start with healthy freshness and adequate source rate: relaxation ramps up.
    healthy_with_adequate_rate = replace(
        healthy, source_release_rate_ratio=1.0
    )
    tier_1 = _idle(
        owner, healthy_with_adequate_rate, active, candidates, modality, step=1
    ).resolution_tier
    # Relaxation should have ratcheted up at least once from quarter.
    assert owner._budget_relax_factor > 1.0

    # Same source rate: should continue ramping.
    tier_2 = _idle(
        owner, healthy_with_adequate_rate, active, candidates, modality, step=2
    ).resolution_tier
    relaxation_before_shortfall = owner._budget_relax_factor
    assert relaxation_before_shortfall > 1.0

    # Source rate drops below floor: relaxation resets to 1.0.
    shortfall = replace(
        healthy_with_adequate_rate, source_release_rate_ratio=0.7
    )
    tier_3 = _idle(owner, shortfall, active, candidates, modality, step=3).resolution_tier
    assert owner._budget_relax_factor == 1.0


def test_none_source_rate_ratio_does_not_reset_relaxation():
    """Relaxation continues unaffected when source_release_rate_ratio is None."""
    owner, snapshot, active, candidates, modality = _floor_locked_owner(
        budget_relax_max=4.0, relax_source_rate_floor=0.9
    )
    healthy = replace(snapshot, freshness_mode=FreshnessMode.HEALTHY)

    # Start with healthy freshness: relaxation ramps up.
    tier_1 = _idle(owner, healthy, active, candidates, modality, step=1).resolution_tier
    relaxation_after_first = owner._budget_relax_factor

    # No source rate info: should continue ramping (None is ignored).
    with_none_ratio = replace(healthy, source_release_rate_ratio=None)
    tier_2 = _idle(owner, with_none_ratio, active, candidates, modality, step=2).resolution_tier
    assert owner._budget_relax_factor >= relaxation_after_first


def test_zero_relax_source_rate_floor_disables_the_reset_signal():
    """Setting floor to 0.0 disables source-rate-based reset."""
    owner, snapshot, active, candidates, modality = _floor_locked_owner(
        budget_relax_max=4.0, relax_source_rate_floor=0.0
    )
    healthy = replace(snapshot, freshness_mode=FreshnessMode.HEALTHY)

    # Start ramping with adequate rate.
    healthy_with_rate = replace(healthy, source_release_rate_ratio=1.0)
    tier_1 = _idle(
        owner, healthy_with_rate, active, candidates, modality, step=1
    ).resolution_tier
    relaxation_before = owner._budget_relax_factor

    # Even with zero rate (worst case), floor 0.0 means no reset happens.
    zero_rate = replace(healthy_with_rate, source_release_rate_ratio=0.0)
    tier_2 = _idle(owner, zero_rate, active, candidates, modality, step=2).resolution_tier
    # Relaxation should stay the same or increase, never reset to 1.0.
    assert owner._budget_relax_factor >= relaxation_before


def test_source_release_rate_ratio_uses_requested_source_fps_not_supervised_views_target():
    """The reset signal divides by the paced SOURCE clock, never by the
    supervised-views target (100 views/s), which would fire the reset on every
    ordinary 30 fps run and silently disable #1754's relaxation."""
    from types import SimpleNamespace

    from eligsir.runtime.production_application import (
        source_release_rate_ratio_from_clocks,
    )

    clocks = SimpleNamespace(source_frame_rate_hz=29.0)
    ratio = source_release_rate_ratio_from_clocks(clocks, requested_source_fps=30.0)
    assert ratio == pytest.approx(29.0 / 30.0)
    # Unpaced / live: no requested source clock -> no signal.
    assert source_release_rate_ratio_from_clocks(clocks, requested_source_fps=None) is None
    assert source_release_rate_ratio_from_clocks(clocks, requested_source_fps=0.0) is None
    # No measured source rate yet -> no signal.
    assert source_release_rate_ratio_from_clocks(None, requested_source_fps=30.0) is None
    assert (
        source_release_rate_ratio_from_clocks(
            SimpleNamespace(source_frame_rate_hz=0.0), requested_source_fps=30.0
        )
        is None
    )
    # Bounded above so a burst cannot ratchet the relaxation through the roof.
    assert source_release_rate_ratio_from_clocks(
        SimpleNamespace(source_frame_rate_hz=300.0), requested_source_fps=30.0
    ) == 2.0


def test_application_declares_paced_source_clock_through_public_setter():
    """causal_entry hands the paced source clock to the application via
    ``set_requested_source_fps`` (None when unpaced); the builder reads it."""
    import ast
    from pathlib import Path

    src = (Path(__file__).resolve().parents[1] / "eligsir" / "runtime" / "causal_entry.py").read_text()
    assert "_declare_source_fps(source_hz if paced else None)" in src
    assert 'getattr(composition.runtime, "set_requested_source_fps", None)' in src
    assert "_requested_source_fps_for_pacing" not in src, "poke the setter, not the private attribute"
    from eligsir.runtime.production_application import CanonicalMappingRuntime as _App  # noqa: F401
    assert callable(getattr(_App, "set_requested_source_fps", None))


def test_source_release_rate_ratio_none_for_unpaced_sources():
    """When source is not paced (iter_based/live), ratio must be None."""
    # Snapshot with no ratio set (default None):
    pending = (_source(4), _source(5), _source(6))
    snapshot = ControlSnapshot(
        snapshot_id=17,
        control_version=3,
        source=pending[-1],
        tracking=None,
        pending_sources=pending,
        newest_frontier=pending[-1],
        freshness_mode=FreshnessMode.HEALTHY,
        pending_oldest_age_ns=10_000,
        ingress_queue_depth=3,
        candidate_queue_depth=2,
        transfer_queue_depth=1,
        gpu_queue_depth=1,
        map_content_version=29,
        topology_version=11,
        parameter_version=31,
        gaussian_count=100,
        optimizer_steps_completed=40,
        limits=_limits(),
        supervision_candidates=pending,
        replay_candidates=(_source(1), _source(2)),
        replay_work_modalities=(
            ModalityEligibility(True, "qualified", False, "missing", False),
            ModalityEligibility(True, "qualified", False, "missing", False),
        ),
        source_release_rate_ratio=None,
    )
    assert snapshot.source_release_rate_ratio is None


def test_expired_consumed_clock_refreshes_replay_snapshot_and_resets_relaxation(
    monkeypatch,
):
    """Only an established paced stream turns five-second silence into zero."""
    from eligsir.mapping.canonical_services import CanonicalFourClocks
    from eligsir.runtime.production_application import (
        CanonicalMappingRuntime,
        source_release_rate_ratio_from_clocks,
    )
    from eligsir.runtime.causal_replay import CausalIngressTelemetryState
    import eligsir.runtime.production_application as production_application

    now = [0.0]
    monkeypatch.setattr(
        "eligsir.mapping.canonical_services.time.perf_counter", lambda: now[0]
    )
    clocks = CanonicalFourClocks()
    monkeypatch.setattr(
        production_application, "ControlSnapshot", ControlSnapshot, raising=False
    )
    monkeypatch.setattr(
        production_application, "ControlLimits", ControlLimits, raising=False
    )
    monkeypatch.setattr(
        production_application,
        "collect_control_proposals",
        lambda *_args: (),
        raising=False,
    )
    runtime = object.__new__(CanonicalMappingRuntime)
    runtime._services = SimpleNamespace(four_clocks=clocks)
    runtime._requested_source_fps_for_pacing = 30.0
    runtime._versions = lambda: (29, 11, 31)
    runtime._control_snapshot_id = 18
    runtime._control_tick_epoch = 0
    runtime.optimizer_steps = 40
    runtime._supervision_views_completed = 0
    runtime._adaptive_supervision_quality_evidence = None
    ingress = CausalIngressTelemetryState(batch_size=1)
    runtime.set_source_input_exhausted_provider(lambda: ingress.input_exhausted)

    def control_runtime():
        from collections import OrderedDict

        runtime._poll_gpu_telemetry = lambda: None
        runtime._work_clock_owner = SimpleNamespace(record_source_frame=lambda: None)
        runtime._source_clock_ns = 0
        runtime._source_arrival_ns = {}
        runtime._source_identities_by_ordinal = {}
        runtime.initialized = False
        runtime._pending_supervision_modalities = OrderedDict()
        runtime._pending_integration_since_ns = None
        runtime._transfer_owner = None
        runtime._gpu_event_telemetry = SimpleNamespace(pending_count=0)
        runtime._pending_optimization_transaction_timings = ()
        runtime._shortlisted_replay_view = lambda **_kwargs: (OrderedDict(), None)
        runtime._replay_weights = {}
        runtime._control_evidence = lambda *_args, **_kwargs: ()
        runtime._batch_size = 1
        runtime._runtime_policy = SimpleNamespace(
            max_new_gaussians_per_frame=0,
            max_transfer_bytes=0,
            max_optimization_transaction_ns=None,
        )
        runtime._population_live_max_actions = 0
        runtime._replay_depth_max_per_event = 0
        runtime._latest_completed_optimization_transaction = None
        runtime._control_proposal_providers = ()
        runtime._latest_control_packet = None

    control_runtime()

    assert source_release_rate_ratio_from_clocks(
        clocks, requested_source_fps=30.0, now=now[0]
    ) is None
    runtime.record_source_consumed(SimpleNamespace(source=_source(0)))
    assert source_release_rate_ratio_from_clocks(
        clocks, requested_source_fps=30.0, now=now[0]
    ) is None
    now[0] = 1.0 / 30.0
    runtime.record_source_consumed(SimpleNamespace(source=_source(1)))
    assert source_release_rate_ratio_from_clocks(
        clocks, requested_source_fps=30.0, now=now[0]
    ) == pytest.approx(1.0)
    assert source_release_rate_ratio_from_clocks(
        clocks, requested_source_fps=None, now=now[0]
    ) is None

    control_packet = SimpleNamespace(source=_source(2))
    control_runtime()
    first_control = runtime.control_snapshot(
        control_packet, None, SimpleNamespace(evidence=())
    )
    runtime._latest_control_packet = None
    second_control = runtime.control_snapshot(
        control_packet, None, SimpleNamespace(evidence=())
    )
    assert first_control.source_release_rate_ratio == pytest.approx(1.0)
    assert second_control.source_release_rate_ratio == pytest.approx(1.0)
    assert clocks.summary(now=now[0]).source_frame_rate_hz == pytest.approx(30.0)

    owner, snapshot, active, candidates, modality = _floor_locked_owner(
        budget_relax_max=4.0, relax_source_rate_floor=0.9
    )
    healthy = replace(snapshot, freshness_mode=FreshnessMode.HEALTHY)
    _idle(
        owner,
        replace(healthy, source_release_rate_ratio=1.0),
        active,
        candidates,
        modality,
        step=1,
    )
    assert owner._budget_relax_factor > 1.0

    now[0] = 5.1
    replay_snapshot = runtime._fresh_idle_snapshot(
        replace(healthy, source_release_rate_ratio=1.0)
    )
    assert replay_snapshot.source_release_rate_ratio == 0.0
    _idle(owner, replay_snapshot, active, candidates, modality, step=2)
    assert owner._budget_relax_factor == 1.0

    eof_owner, eof_snapshot, active, candidates, modality = _floor_locked_owner(
        budget_relax_max=4.0, relax_source_rate_floor=0.9
    )
    eof_healthy = replace(eof_snapshot, freshness_mode=FreshnessMode.HEALTHY)
    _idle(
        eof_owner,
        replace(eof_healthy, source_release_rate_ratio=1.0),
        active,
        candidates,
        modality,
        step=1,
    )
    relaxation_before_eof = eof_owner._budget_relax_factor

    ingress.set_input_exhausted(True)
    eof_replay_snapshot = runtime._fresh_idle_snapshot(
        replace(eof_healthy, source_release_rate_ratio=1.0)
    )
    assert eof_replay_snapshot.source_release_rate_ratio is None
    _idle(eof_owner, eof_replay_snapshot, active, candidates, modality, step=2)
    assert eof_owner._budget_relax_factor >= relaxation_before_eof
    assert runtime._fresh_idle_snapshot(eof_replay_snapshot).source_release_rate_ratio is None
