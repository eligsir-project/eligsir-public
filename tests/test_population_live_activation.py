"""M5/M6 production live-policy activation ownership tests.

These tests exercise the composition boundary, not a policy approximation:
the only structural write must be one ``PopulationMutationProposal`` at the
population fence.  Incomplete/stale activation intentionally has no legacy
fallback because the feature request has already transferred ordinary
decision ownership to the population policy.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from eligsir.contracts.work_clocks import RuntimeWorkClockOwner
from eligsir.map.arbiter import PopulationMutationProposal, ProposalKey
from eligsir.map.row_state import GaussianRowStateRegistry, RowStateDescriptor
from eligsir.mapping.evidence.surface_targets import (
    SurfaceTargetBatch,
    SurfaceTargetSnapshot,
)
from eligsir.mapping.evidence.evidence_demand import EvidenceDemandBatch
from eligsir.mapping.population_plan_adapter import (
    PopulationPlanAdapterContext,
    PopulationPlanDropReason,
)
from eligsir.mapping.population_policy import (
    GaussianPopulationPolicy,
    MutationWorkToken,
    PopulationAction,
    PopulationDecision,
)
from eligsir.mapping.population_shadow_runtime import (
    PopulationActivationObservation,
    PopulationShadowVersions,
)
from eligsir.optimization.gaussian.supervision_evidence import EvidenceSnapshotToken
from eligsir.runtime.production_application import CanonicalMappingRuntime
from eligsir.runtime.transfers import DeviceTransferOwner


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


def _versions() -> PopulationShadowVersions:
    return PopulationShadowVersions(
        map_content=4, topology=5, coordinate_frame=6, schema=7,
        optimizer_update=8, surface_evidence=9, policy=10, parameter=12,
    )


def _origin_snapshot() -> EvidenceSnapshotToken:
    versions = _versions()
    return EvidenceSnapshotToken(
        map_content=versions.map_content, topology=versions.topology,
        coordinate_frame=versions.coordinate_frame, schema=versions.schema,
        optimizer_update=versions.optimizer_update,
        surface_evidence=versions.surface_evidence, policy=versions.policy,
        parameter=11,
    )


def _targets() -> SurfaceTargetBatch:
    versions = _versions()
    return SurfaceTargetBatch(
        snapshot=SurfaceTargetSnapshot(
            map_content=versions.map_content, topology=versions.topology,
            coordinate_frame=versions.coordinate_frame, schema=versions.schema,
            surface_evidence=versions.surface_evidence,
        ),
        positions=torch.tensor([[1.0, 2.0, 3.0]]),
        normals=torch.tensor([[0.0, 0.0, 1.0]]),
        uncertainty=torch.full((1, 3), 0.02),
        support_evidence=torch.ones(1), conflict_evidence=torch.zeros(1),
        independent_view_count=torch.full((1,), 2, dtype=torch.int32),
        repeated_exposure_count=torch.full((1,), 2, dtype=torch.int32),
        target_ids=torch.tensor([100], dtype=torch.int64),
        source_modality=torch.full((1,), 4, dtype=torch.int8), max_targets=1,
    )


def _decision(snapshot: SurfaceTargetSnapshot) -> PopulationDecision:
    actions = torch.tensor([PopulationAction.CREATE.code, PopulationAction.NONE.code], dtype=torch.int8)
    target_indices = torch.full((2,), -1, dtype=torch.int64)
    selection = GaussianPopulationPolicy._structural_selection(
        actions, torch.tensor([1.0, 0.0]), target_indices, max_actions=1,
    )
    decision = PopulationDecision.policy_output(
        actions=actions, target_indices=target_indices,
        evidence_demands=EvidenceDemandBatch(()),
        demand_masks=(), structural_selection=selection,
    )
    return decision.bind_surface_target_ids(
        torch.tensor([100], dtype=torch.int64),
        target_snapshot=snapshot, structural_action=PopulationAction.CREATE,
    )


class _Bridge:
    def __init__(self, observation):
        self._observation = observation
        self.calls = 0
        self.rejection = None

    def take_latest_activation_observation(self):
        self.calls += 1
        value, self._observation = self._observation, None
        return value

    def record_adapter_rejection(self, observation, *, plan_drop_reason=None):
        self.rejection = (observation, plan_drop_reason)


def _runtime(*, live: bool = True, shadow: bool = True, units: int = 1):
    runtime = object.__new__(CanonicalMappingRuntime)
    runtime.config = SimpleNamespace(
        static_options={
            "population_live_enabled": live,
            "population_shadow_enabled": shadow,
        }
    )
    runtime.backend = SimpleNamespace(
        map_content_version=4, topology_version=5,
    )
    runtime._engine = SimpleNamespace(session=SimpleNamespace(parameter_version=12))
    runtime._runtime_policy = SimpleNamespace(cap_max=4)
    runtime._transfer_owner = DeviceTransferOwner("cpu")
    clocks = RuntimeWorkClockOwner()
    clocks.record_optimization(optimizer_updates=8, supervised_views=1, supervised_pixels=1, mapping_active_ns=0)
    runtime._work_clock_owner = clocks
    targets = _targets()
    observation = PopulationActivationObservation(
        transaction_id="b4-8", token=MutationWorkToken(epoch=8, units=units),
        origin_snapshot=_origin_snapshot(), versions=_versions(),
        decision=_decision(targets.snapshot),
    )
    bridge = _Bridge(observation)
    runtime._optimization_runtime = SimpleNamespace(
        population_shadow_enabled=shadow, population_shadow_runtime=bridge,
    )
    runtime._population_live_target_binding_provider = (
        lambda decision, _origin, _fence, _work: (decision, targets)
    )
    runtime._population_live_context_provider = lambda token, work, _plan: PopulationPlanAdapterContext(
        current_snapshot=token, decision_snapshot=token, capacity=4,
        hypothesis_uids=("10", "11"), next_hypothesis_uid=100,
        row_state_registry=_registry(), proposal_key=ProposalKey(1, 2, 3, "population"),
        max_actions=work.units, transfer_owner=runtime._transfer_owner,
    )
    return runtime, bridge


def _plan(units: int = 1):
    return SimpleNamespace(mutation_budget=SimpleNamespace(max_total=units))


class _RejectingLegacyMCMC:
    """Every legacy MCMC hook is an error in an enabled live-policy tick."""

    def step_pre_backward(self, *args, **kwargs):
        pytest.fail("legacy MCMC pre-backward hook called")

    def inject_noise(self, *args, **kwargs):
        pytest.fail("legacy MCMC noise writer called")

    def delegated_post_backward_proposal(self, *args, **kwargs):
        pytest.fail("delegated MCMC proposal builder called")

    def step_post_backward(self, *args, **kwargs):
        pytest.fail("direct legacy MCMC mutation writer called")


def test_live_activation_submits_exactly_one_population_proposal_and_never_legacy(monkeypatch):
    runtime, _bridge = _runtime()
    submitted = []
    runtime._mcmc_strategy = _RejectingLegacyMCMC()
    runtime._commit_mutation_barrier = lambda proposals, **kwargs: (
        submitted.append((proposals, kwargs))
        or SimpleNamespace(
            applied=1, change=SimpleNamespace(topology_version=5)
        )
    )
    runtime._run_due_lifecycle = lambda *args, **kwargs: pytest.fail("legacy lifecycle called")
    runtime._mutate_after_step = lambda *args, **kwargs: pytest.fail("delegated MCMC called")
    runtime._prepare_insertion_proposal = lambda *args, **kwargs: pytest.fail("depth insertion called")
    from eligsir.utils import streaming_state_sync
    from eligsir.utils.strategies.mcmc import arbiter_owned

    monkeypatch.setattr(
        streaming_state_sync, "_sync_streaming_state_lengths",
        lambda *_args, **_kwargs: pytest.fail("legacy lifecycle padding called"),
    )
    monkeypatch.setattr(
        arbiter_owned, "_resync_gaussian_lifecycle_buffers",
        lambda *_args, **_kwargs: pytest.fail("legacy lifecycle resync called"),
    )

    used_live, commit = runtime._post_step_structural_mutation(
        SimpleNamespace(), _plan(), post_step_utility=None
    )

    assert used_live and commit.applied == 1
    assert len(submitted) == 1
    proposals, kwargs = submitted[0]
    assert len(proposals) == 1
    assert isinstance(proposals[0], PopulationMutationProposal)
    assert kwargs == {"expected_map_content_version": 4, "expected_topology_version": 5}


def test_live_plan_rejection_sets_coarse_disposition_and_retains_fine_reason(monkeypatch):
    runtime, bridge = _runtime()
    runtime._commit_mutation_barrier = lambda *args, **kwargs: pytest.fail("must not commit")

    from eligsir.mapping.population_plan_adapter import PopulationPolicyPlanAdapter

    monkeypatch.setattr(
        PopulationPolicyPlanAdapter,
        "build",
        lambda *args, **kwargs: SimpleNamespace(
            proposal=None,
            reason=PopulationPlanDropReason.CAPACITY_CONFLICT,
            target_attestation=None,
        ),
    )

    used_live, commit = runtime._post_step_structural_mutation(
        SimpleNamespace(), _plan(), post_step_utility=None
    )

    assert used_live and commit is None
    assert runtime._population_live_activation_disposition == "population_plan_rejected"
    assert bridge.rejection[1] is PopulationPlanDropReason.CAPACITY_CONFLICT


def test_live_policy_abstention_never_falls_back_to_legacy_mcmc_or_lifecycle(monkeypatch):
    runtime, _bridge = _runtime()
    runtime._mcmc_strategy = _RejectingLegacyMCMC()
    runtime._population_live_target_binding_provider = (
        lambda *_args: (_ for _ in ()).throw(RuntimeError("target binding unavailable"))
    )
    runtime._run_due_lifecycle = lambda *args, **kwargs: pytest.fail("legacy lifecycle called")
    runtime._mutate_after_step = lambda *args, **kwargs: pytest.fail("delegated MCMC called")
    runtime._prepare_insertion_proposal = lambda *args, **kwargs: pytest.fail("depth insertion called")
    runtime._commit_mutation_barrier = lambda *args, **kwargs: pytest.fail("must not commit")
    from eligsir.utils import streaming_state_sync
    from eligsir.utils.strategies.mcmc import arbiter_owned

    monkeypatch.setattr(
        streaming_state_sync, "_sync_streaming_state_lengths",
        lambda *_args, **_kwargs: pytest.fail("legacy lifecycle padding called"),
    )
    monkeypatch.setattr(
        arbiter_owned, "_resync_gaussian_lifecycle_buffers",
        lambda *_args, **_kwargs: pytest.fail("legacy lifecycle resync called"),
    )

    used_live, commit = runtime._post_step_structural_mutation(
        SimpleNamespace(), _plan(), post_step_utility=None
    )

    assert used_live and commit is None


def test_live_policy_excludes_legacy_mcmc_from_fixed_topology_executor():
    runtime, _bridge = _runtime(live=True)
    legacy = _RejectingLegacyMCMC()
    runtime._mcmc_strategy = legacy

    strategy, fixed_shape_mcmc = runtime._fixed_topology_mcmc_runtime_options()

    assert strategy is None
    assert fixed_shape_mcmc is False

    compatible, _bridge = _runtime(live=False)
    compatible._mcmc_strategy = legacy
    strategy, fixed_shape_mcmc = compatible._fixed_topology_mcmc_runtime_options()

    assert strategy is legacy
    assert fixed_shape_mcmc is True


def test_fixed_topology_backward_allows_live_runtime_without_legacy_mcmc_hooks():
    from eligsir.optimization.gaussian.engine import GaussianOptimizationEngine

    loss = torch.tensor(3.0, requires_grad=True)

    GaussianOptimizationEngine.render_and_backward(
        object(), object(), object(), 1, None, {}, loss
    )

    assert loss.grad is not None
    assert float(loss.grad) == 1.0


def test_zero_budget_does_not_invoke_target_or_context_adapters_or_submit():
    runtime, bridge = _runtime(units=0)
    calls = []
    runtime._population_live_target_binding_provider = lambda *args: calls.append("target")
    runtime._population_live_context_provider = lambda *args: calls.append("context")
    runtime._commit_mutation_barrier = lambda *args, **kwargs: pytest.fail("must not commit")

    requested, commit = runtime._maybe_actuate_population_policy_after_step(
        SimpleNamespace(), _plan(0)
    )

    assert requested and commit is None
    assert bridge.calls == 1
    assert calls == []


def test_stale_current_versions_drop_without_legacy_fallback_or_submission():
    runtime, _bridge = _runtime()
    runtime.backend.map_content_version = 99
    calls = []
    runtime._population_live_target_binding_provider = lambda *args: calls.append("target")
    runtime._population_live_context_provider = lambda *args: calls.append("context")
    runtime._commit_mutation_barrier = lambda *args, **kwargs: pytest.fail("must not commit")

    requested, commit = runtime._maybe_actuate_population_policy_after_step(
        SimpleNamespace(), _plan()
    )

    assert requested and commit is None
    assert calls == []


def test_later_post_adam_parameter_version_drops_before_target_binding():
    """Catch a delayed action fence rebinding E0 targets after another Adam step."""

    runtime, _bridge = _runtime()
    runtime._engine.session.parameter_version = 13
    calls = []
    runtime._population_live_target_binding_provider = lambda *args: calls.append(args)
    runtime._population_live_context_provider = lambda *args: pytest.fail("context called")
    runtime._commit_mutation_barrier = lambda *args, **kwargs: pytest.fail("must not commit")

    requested, commit = runtime._maybe_actuate_population_policy_after_step(
        SimpleNamespace(), _plan()
    )

    assert requested and commit is None
    assert calls == []


def test_missing_target_binding_fails_closed_without_a_mutation_fence():
    runtime, _bridge = _runtime()
    runtime._population_live_target_binding_provider = lambda *args: None
    runtime._commit_mutation_barrier = lambda *args, **kwargs: pytest.fail("must not commit")

    requested, commit = runtime._maybe_actuate_population_policy_after_step(
        SimpleNamespace(), _plan()
    )

    assert requested and commit is None
    assert runtime._population_live_activation_disposition == "target_binding_rejected"


def test_unavailable_target_binding_provider_is_not_policy_abstention():
    runtime, bridge = _runtime()
    runtime._population_live_target_binding_provider = None
    runtime._commit_mutation_barrier = lambda *args, **kwargs: pytest.fail("must not commit")

    requested, commit = runtime._maybe_actuate_population_policy_after_step(
        SimpleNamespace(), _plan()
    )

    assert requested and commit is None
    assert bridge.calls == 0
    assert runtime._population_live_activation_disposition == "target_binding_rejected"


def test_target_provider_cannot_substitute_another_authenticated_policy_decision():
    runtime, _bridge = _runtime()
    replacement = _decision(_targets().snapshot)
    runtime._population_live_target_binding_provider = (
        lambda _decision_value, _origin, _fence, _work: (replacement, _targets())
    )
    runtime._commit_mutation_barrier = lambda *args, **kwargs: pytest.fail("must not commit")

    requested, commit = runtime._maybe_actuate_population_policy_after_step(
        SimpleNamespace(), _plan()
    )

    assert requested and commit is None


def test_disabled_live_policy_preserves_compatibility_and_does_not_consume_shadow_state():
    runtime, bridge = _runtime(live=False)
    runtime._commit_mutation_barrier = lambda *args, **kwargs: pytest.fail("must not commit")

    requested, commit = runtime._maybe_actuate_population_policy_after_step(
        SimpleNamespace(), _plan()
    )

    assert not requested and commit is None
    assert bridge.calls == 0


def test_live_policy_requires_the_shadow_feature_gate():
    runtime, _bridge = _runtime(live=True, shadow=False)

    with pytest.raises(RuntimeError, match="requires the enabled shadow"):
        runtime._maybe_actuate_population_policy_after_step(SimpleNamespace(), _plan())


def test_live_mutation_dispatch_bypasses_delegated_mcmc_only_when_requested():
    runtime, _bridge = _runtime(live=True)
    population_commit = SimpleNamespace(
        applied=1, change=SimpleNamespace(topology_version=6)
    )
    runtime._maybe_actuate_population_policy_after_step = lambda *_args: (
        True, population_commit
    )
    runtime._mutate_after_step = lambda *_args, **_kwargs: pytest.fail("legacy MCMC called")

    used_live, mutation = runtime._post_step_structural_mutation(
        SimpleNamespace(), _plan(), post_step_utility=None
    )
    assert used_live and mutation is population_commit

    compatible, _bridge = _runtime(live=False)
    compatible._maybe_actuate_population_policy_after_step = lambda *_args: pytest.fail("live path called")
    compatible._mutate_after_step = lambda *_args, **_kwargs: "legacy"
    used_live, mutation = compatible._post_step_structural_mutation(
        SimpleNamespace(), _plan(), post_step_utility=None
    )
    assert not used_live and mutation == "legacy"


@pytest.mark.parametrize("branch", ["b1", "b4"])
def test_live_population_commit_recorder_is_shared_by_b1_and_b4_dispatch(branch):
    runtime, _bridge = _runtime(live=True)
    calls = []
    runtime._maybe_actuate_population_policy_after_step = lambda *_args: (
        True,
        SimpleNamespace(applied=1, change=SimpleNamespace(topology_version=6)),
    )
    runtime._record_completed_population_policy_commit = lambda commit: calls.append(
        (branch, commit.change.topology_version)
    )
    runtime._mutate_after_step = lambda *_args, **_kwargs: pytest.fail("legacy MCMC called")

    used_live, _commit = runtime._post_step_structural_mutation(
        SimpleNamespace(), _plan(), post_step_utility=None
    )

    assert used_live
    assert calls == [(branch, 6)]


def test_population_commit_recorder_consumes_neutral_cadence_without_mcmc_clock():
    runtime, _bridge = _runtime(live=True)

    class _Cadence:
        def __init__(self):
            self.calls = []

        def record_population_mutation(self, snapshot, **kwargs):
            self.calls.append((snapshot, kwargs))

    cadence = _Cadence()
    runtime._mcmc_cadence_owner = cadence
    before = runtime._work_clock_owner.snapshot()
    returned = runtime._record_completed_population_policy_commit(
        SimpleNamespace(applied=1, change=SimpleNamespace(topology_version=6))
    )

    assert returned == before
    assert returned.mcmc_mutation_epochs == 0
    assert len(cadence.calls) == 1
    snapshot, kwargs = cadence.calls[0]
    assert snapshot == before
    assert kwargs == {"topology_version_after": 6}
