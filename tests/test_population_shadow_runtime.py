"""Deterministic contracts for the observational M5 population-policy bridge."""

from types import SimpleNamespace

import torch

from eligsir.mapping.population_policy import MutationWorkToken, PopulationAction
from eligsir.mapping.population_plan_adapter import PopulationPlanDropReason
from eligsir.mapping.optimization_runtime import StreamingOptimizationRuntime
from eligsir.mapping.population_shadow_runtime import (
    _compact_async_cpu,
    PopulationShadowDropReason,
    PopulationShadowRuntime,
    PopulationShadowVersions,
)
from eligsir.mapping.evidence.surface_targets import (
    SurfaceTargetBatch,
    SurfaceTargetSnapshot,
)
from eligsir.runtime.transfers import PackedReadback, TransferEvent
from eligsir.optimization.gaussian.supervision_evidence import (
    EvidenceSnapshotToken,
    SupervisionEvidenceBatch,
    SupervisionModality,
)
from eligsir.utils.gaussian_belief import (
    ExistenceBelief,
    GaussianBelief,
    GeometryBelief,
    ObservabilityBelief,
    ProvenanceBelief,
    RepresentationBelief,
    TemporalBelief,
)


def _belief(n: int = 2) -> GaussianBelief:
    device = torch.device("cpu")
    return GaussianBelief(
        provenance=ProvenanceBelief(
            gaussian_uid=torch.arange(n, device=device),
            birth_update=torch.zeros(n, dtype=torch.int32, device=device),
            birth_method=torch.zeros(n, dtype=torch.int8, device=device),
            parent_uid=torch.full((n,), -1, device=device),
            root_uid=torch.arange(n, device=device),
            generation=torch.zeros(n, dtype=torch.int32, device=device),
            birth_anchor_xyz=torch.zeros(n, 3, device=device),
        ),
        observability=ObservabilityBelief(
            meaningful_exposure_count=torch.zeros(n, dtype=torch.int32, device=device),
            independent_support_views=torch.zeros(n, dtype=torch.int16, device=device),
            baseline_diversity=torch.zeros(n, device=device),
            highest_supervised_tier=None,
            last_exposed_update=torch.zeros(n, dtype=torch.int32, device=device),
            last_supported_update=None,
        ),
        existence=ExistenceBelief(
            support_evidence=torch.zeros(n, device=device),
            contradiction_evidence=torch.zeros(n, device=device),
            p_exist=torch.ones(n, device=device),
        ),
        geometry=GeometryBelief(
            surface_anchor_id=torch.full((n,), -1, device=device),
            p_surface_consistent=torch.ones(n, device=device),
            surface_position_uncertainty=torch.zeros(n, device=device),
            surface_normal_uncertainty=torch.zeros(n, device=device),
        ),
        representation=RepresentationBelief(
            render_contribution=torch.ones(n, device=device),
            rgb_residual_ema=torch.zeros(n, device=device),
            depth_residual_ema=torch.zeros(n, device=device),
            gradient_utility_evidence=torch.zeros(n, device=device),
            redundancy=torch.zeros(n, device=device),
            expected_marginal_gain=torch.zeros(n, device=device),
        ),
        temporal=TemporalBelief(
            evidence_recency=torch.zeros(n, dtype=torch.int32, device=device),
            change_probability=torch.zeros(n, device=device),
            confidence_decay=torch.zeros(n, device=device),
        ),
        n=n,
    )


def _versions() -> PopulationShadowVersions:
    return PopulationShadowVersions(
        map_content=4,
        topology=5,
        coordinate_frame=6,
        schema=7,
        optimizer_update=8,
        surface_evidence=9,
        policy=10,
        parameter=12,
    )


def _evidence(*, parameter: int = 11, gaussian_count: int = 0) -> SupervisionEvidenceBatch:
    values = _versions()
    token = EvidenceSnapshotToken(
        map_content=4,
        topology=5,
        coordinate_frame=6,
        schema=7,
        optimizer_update=8,
        surface_evidence=9,
        policy=10,
        parameter=parameter,
    )
    kwargs = {}
    if gaussian_count:
        kwargs["gradient_utility_evidence"] = torch.zeros(gaussian_count)
    return SupervisionEvidenceBatch(
        snapshot_token=token,
        transaction_id="transaction-1",
        modalities=(SupervisionModality(),),
        rgb_residual_summary=torch.zeros(1),
        **kwargs,
    )


def _result(evidence: SupervisionEvidenceBatch) -> SimpleNamespace:
    return SimpleNamespace(
        evidence=evidence,
        map_version_before=4,
        map_version_after=4,
        topology_version=5,
        parameter_version_before=11,
        parameter_version_after=12,
    )


def test_disabled_bridge_is_a_true_observational_noop():
    bridge = PopulationShadowRuntime(enabled=False)
    belief = _belief()
    before = belief.representation.expected_marginal_gain.clone()

    drop_reason = bridge.observe(
        result=_result(_evidence()), belief=belief,
        token=MutationWorkToken(epoch=8, units=1), current_versions=_versions(), capacity=2,
    )

    assert drop_reason is PopulationShadowDropReason.DISABLED
    assert bridge.pending_event_count == 0
    assert torch.equal(belief.representation.expected_marginal_gain, before)


class _CudaSentinel:
    device = torch.device("cuda")

    def detach(self):
        raise AssertionError("unowned CUDA telemetry must not materialize")


def test_cuda_compact_telemetry_drops_without_exact_owner():
    events = []
    assert _compact_async_cpu(
        _CudaSentinel(), transfer_owner=None, transfer_events=events
    ) is None
    assert events == []


def test_cuda_compact_telemetry_drops_for_mismatched_owner():
    events = []
    owner = SimpleNamespace(device=torch.device("cpu"), transfer_stream=object())
    assert _compact_async_cpu(
        _CudaSentinel(), transfer_owner=owner, transfer_events=events
    ) is None
    assert events == []


def test_packed_compact_readback_is_poll_only_and_releases_after_clone():
    class _Event:
        def __init__(self):
            self.ready = False
            self.waits = 0

        def query(self):
            return self.ready

        def wait(self):
            self.waits += 1
            raise AssertionError("compact trace polling must not wait")

    event = _Event()
    readback = PackedReadback(
        host=torch.tensor([1, 2], dtype=torch.uint8),
        event=TransferEvent(event),
    )
    assert readback.try_take_cpu_tensor() is None
    event.ready = True
    assert torch.equal(readback.try_take_cpu_tensor(), torch.tensor([1, 2], dtype=torch.uint8))
    assert event.waits == 0


def test_matching_pre_adam_e0_evidence_publishes_only_policy_and_demands():
    bridge = PopulationShadowRuntime(enabled=True)
    drop_reason = bridge.observe(
        result=_result(_evidence()), belief=_belief(),
        token=MutationWorkToken(epoch=8, units=0), current_versions=_versions(), capacity=2,
    )

    assert drop_reason is None
    assert bridge.pending_event_count == 1
    summaries = bridge.drain_async_summaries()
    assert len(summaries) == 1
    assert summaries[0].demand_count == 2
    assert summaries[0].action_counts_cpu.device.type == "cpu"
    assert summaries[0].action_counts_cpu.sum() == 0
    assert bridge.pending_event_count == 0


def test_accepted_shadow_decision_exports_bounded_demands_without_tensor_readback():
    bridge = PopulationShadowRuntime(enabled=True)

    assert bridge.observe(
        result=_result(_evidence()), belief=_belief(),
        token=MutationWorkToken(epoch=8, units=0), current_versions=_versions(), capacity=2,
    ) is None

    demands = bridge.take_latest_evidence_demands()

    assert len(demands) == 2
    assert bridge.take_latest_evidence_demands() == ()


def test_accepted_policy_can_be_consumed_once_for_the_live_fence_without_losing_telemetry():
    bridge = PopulationShadowRuntime(enabled=True)
    evidence = _evidence()
    assert bridge.observe(
        result=_result(evidence), belief=_belief(),
        token=MutationWorkToken(epoch=8, units=1), current_versions=_versions(), capacity=2,
    ) is None

    activation = bridge.take_latest_activation_observation()

    assert activation is not None
    assert activation.token == MutationWorkToken(epoch=8, units=1)
    assert activation.origin_snapshot is evidence.snapshot_token
    assert activation.versions == _versions()
    assert bridge.take_latest_activation_observation() is None
    # The activation handoff is independent of async observability.
    assert len(bridge.drain_async_summaries()) == 1


def test_bridge_stale_drops_when_evidence_is_not_bound_to_result_pre_adam_snapshot():
    bridge = PopulationShadowRuntime(enabled=True)
    drop_reason = bridge.observe(
        result=_result(_evidence(parameter=10)), belief=_belief(),
        token=MutationWorkToken(epoch=8, units=1), current_versions=_versions(), capacity=2,
    )

    assert drop_reason is PopulationShadowDropReason.STALE_EVIDENCE_SNAPSHOT


def test_bridge_stale_drops_when_current_post_adam_versions_advanced():
    bridge = PopulationShadowRuntime(enabled=True)
    versions = _versions()
    stale_current = PopulationShadowVersions(**{**versions.__dict__, "parameter": 13})
    drop_reason = bridge.observe(
        result=_result(_evidence()), belief=_belief(),
        token=MutationWorkToken(epoch=8, units=1), current_versions=stale_current, capacity=2,
    )

    assert drop_reason is PopulationShadowDropReason.STALE_CURRENT_SNAPSHOT


def test_stale_current_trace_attributes_parameter_axis_with_expected_and_current_values():
    bridge = PopulationShadowRuntime(enabled=True)
    versions = _versions()
    stale_current = PopulationShadowVersions(**{**versions.__dict__, "parameter": 13})

    assert bridge.observe(
        result=_result(_evidence()), belief=_belief(),
        token=MutationWorkToken(epoch=8, units=1), current_versions=stale_current, capacity=2,
    ) is PopulationShadowDropReason.STALE_CURRENT_SNAPSHOT

    summary = bridge.drain_async_summaries()[0]
    assert summary.stale_current_mismatches == (("parameter", 12, 13),)


def test_stale_current_trace_attributes_optimizer_update_separately_from_parameter():
    bridge = PopulationShadowRuntime(enabled=True)
    versions = _versions()
    stale_current = PopulationShadowVersions(**{**versions.__dict__, "optimizer_update": 14})

    assert bridge.observe(
        result=_result(_evidence()), belief=_belief(),
        token=MutationWorkToken(epoch=8, units=1), current_versions=stale_current, capacity=2,
    ) is PopulationShadowDropReason.STALE_CURRENT_SNAPSHOT

    summary = bridge.drain_async_summaries()[0]
    assert summary.stale_current_mismatches == (("optimizer_update", 8, 14),)


def test_accepted_summary_has_no_stale_current_trace():
    bridge = PopulationShadowRuntime(enabled=True)
    assert bridge.observe(
        result=_result(_evidence()), belief=_belief(),
        token=MutationWorkToken(epoch=8, units=0), current_versions=_versions(), capacity=2,
    ) is None

    assert bridge.drain_async_summaries()[0].stale_current_mismatches == ()


def test_native_target_evidence_is_scored_by_policy_without_mutation():
    """The depth reducer may inform E0 scoring but cannot write topology."""

    bridge = PopulationShadowRuntime(enabled=True)
    versions = _versions()
    targets = SurfaceTargetBatch(
        snapshot=SurfaceTargetSnapshot(
            map_content=versions.map_content,
            topology=versions.topology,
            coordinate_frame=versions.coordinate_frame,
            schema=versions.schema,
            surface_evidence=versions.surface_evidence,
        ),
        positions=torch.tensor([[1.0, 2.0, 3.0]]),
        normals=torch.tensor([[0.0, 0.0, 1.0]]),
        uncertainty=torch.full((1, 3), 0.02),
        support_evidence=torch.ones(1),
        conflict_evidence=torch.zeros(1),
        independent_view_count=torch.full((1,), 2, dtype=torch.int32),
        repeated_exposure_count=torch.full((1,), 2, dtype=torch.int32),
        target_ids=torch.tensor([11], dtype=torch.int64),
        source_modality=torch.full((1,), 4, dtype=torch.int8),
        max_targets=1,
    )

    belief = _belief()
    belief.observability.meaningful_exposure_count.fill_(1)
    belief.observability.independent_support_views.fill_(2)

    assert bridge.observe(
        result=_result(_evidence()), belief=belief, token=MutationWorkToken(8, 1),
        current_versions=versions, capacity=4, native_targets=targets,
    ) is None

    activation = bridge.take_latest_activation_observation()
    assert activation is not None
    assert torch.any(activation.decision.actions == PopulationAction.CREATE.code)
    summary = bridge.drain_async_summaries()[0]
    assert summary.native_target_status_cpu.item() == 2  # qualified native target


def test_native_target_conflict_disqualifies_only_target_not_existing_rows():
    """A target-local depth conflict must never manufacture row RETIRE proof."""

    bridge = PopulationShadowRuntime(enabled=True)
    versions = _versions()
    targets = SurfaceTargetBatch(
        snapshot=SurfaceTargetSnapshot(
            map_content=versions.map_content, topology=versions.topology,
            coordinate_frame=versions.coordinate_frame, schema=versions.schema,
            surface_evidence=versions.surface_evidence,
        ),
        positions=torch.tensor([[1.0, 2.0, 3.0]]),
        normals=torch.tensor([[0.0, 0.0, 1.0]]),
        uncertainty=torch.full((1, 3), 0.02),
        support_evidence=torch.ones(1),
        conflict_evidence=torch.full((1,), 0.6),
        independent_view_count=torch.full((1,), 2, dtype=torch.int32),
        repeated_exposure_count=torch.full((1,), 2, dtype=torch.int32),
        target_ids=torch.tensor([13], dtype=torch.int64),
        source_modality=torch.full((1,), 4, dtype=torch.int8), max_targets=1,
    )
    belief = _belief()
    belief.observability.meaningful_exposure_count.fill_(1)
    belief.observability.independent_support_views.fill_(2)

    assert bridge.observe(
        result=_result(_evidence()), belief=belief, token=MutationWorkToken(8, 1),
        current_versions=versions, capacity=4, native_targets=targets,
    ) is None
    activation = bridge.take_latest_activation_observation()
    assert activation is not None
    assert not torch.any(activation.decision.actions == PopulationAction.RETIRE.code)
    assert not torch.any(activation.decision.actions == PopulationAction.CREATE.code)


def test_epoch_summary_reports_e0_counts_maxima_and_selection_rejection():
    """Input telemetry keeps native E0 evidence compact and epoch-local."""

    bridge = PopulationShadowRuntime(enabled=True, full_diagnostics=True)
    belief = _belief()
    belief.observability.meaningful_exposure_count[:] = torch.tensor([1, 0])
    belief.observability.independent_support_views[:] = torch.tensor([2, 0])
    versions = _versions()
    targets = SurfaceTargetBatch(
        snapshot=SurfaceTargetSnapshot(
            map_content=versions.map_content,
            topology=versions.topology,
            coordinate_frame=versions.coordinate_frame,
            schema=versions.schema,
            surface_evidence=versions.surface_evidence,
        ),
        positions=torch.tensor([[1.0, 2.0, 3.0]]),
        normals=torch.tensor([[0.0, 0.0, 1.0]]),
        uncertainty=torch.full((1, 3), 0.02),
        support_evidence=torch.full((1,), 0.9),
        conflict_evidence=torch.zeros(1),
        independent_view_count=torch.full((1,), 2, dtype=torch.int32),
        repeated_exposure_count=torch.full((1,), 2, dtype=torch.int32),
        target_ids=torch.tensor([12], dtype=torch.int64),
        source_modality=torch.full((1,), 4, dtype=torch.int8),
        max_targets=1,
    )
    result = _result(_evidence())
    result.population_policy_e0_accumulator_status = "updated"

    assert bridge.observe(
        result=result, belief=belief, token=MutationWorkToken(8, 1),
        current_versions=versions, capacity=4, native_targets=targets,
    ) is None
    bridge.record_adapter_rejection(bridge.take_latest_activation_observation())

    summary = bridge.drain_async_summaries()[0]
    assert summary.accumulator_status == "updated"
    torch.testing.assert_close(
        summary.e0_input_counts_maxima_cpu,
        torch.tensor([1.0, 1.0, 1.0, 2.0, 2.0, 0.45]),
    )
    assert bool(summary.selection_rejected_cpu)
    assert summary.adapter_rejected is True


def test_epoch_summary_classifies_missing_native_target_without_reading_device_arrays():
    """A missing target is distinct from a policy-selected action envelope."""

    bridge = PopulationShadowRuntime(enabled=True)
    belief = _belief()
    belief.observability.meaningful_exposure_count.fill_(1)
    belief.observability.independent_support_views.fill_(2)

    assert bridge.observe(
        result=_result(_evidence()), belief=belief,
        token=MutationWorkToken(8, 1), current_versions=_versions(), capacity=4,
    ) is None

    summary = bridge.drain_async_summaries()[0]
    assert summary.native_target_count == 0
    assert summary.native_target_status_cpu.item() == 0  # no qualified native target
    assert summary.selected_action_histogram_cpu is summary.action_counts_cpu


def test_epoch_summary_classifies_missing_independent_native_support():
    """A present native target with no independent support is explicit telemetry."""

    bridge = PopulationShadowRuntime(enabled=True)
    versions = _versions()
    targets = SurfaceTargetBatch(
        snapshot=SurfaceTargetSnapshot(
            map_content=versions.map_content, topology=versions.topology,
            coordinate_frame=versions.coordinate_frame, schema=versions.schema,
            surface_evidence=versions.surface_evidence,
        ),
        positions=torch.tensor([[1.0, 2.0, 3.0]]),
        normals=torch.tensor([[0.0, 0.0, 1.0]]),
        uncertainty=torch.full((1, 3), 0.02),
        support_evidence=torch.ones(1), conflict_evidence=torch.zeros(1),
        independent_view_count=torch.zeros(1, dtype=torch.int32),
        repeated_exposure_count=torch.zeros(1, dtype=torch.int32),
        target_ids=torch.tensor([11], dtype=torch.int64),
        source_modality=torch.full((1,), 4, dtype=torch.int8), max_targets=1,
    )
    belief = _belief()
    belief.observability.meaningful_exposure_count.fill_(1)
    belief.observability.independent_support_views.fill_(2)

    assert bridge.observe(
        result=_result(_evidence()), belief=belief,
        token=MutationWorkToken(8, 1), current_versions=versions,
        capacity=4, native_targets=targets,
    ) is None

    summary = bridge.drain_async_summaries()[0]
    assert summary.native_target_count == 1
    assert summary.native_target_status_cpu.item() == 1  # target unqualified


def test_adapter_plan_drop_reason_is_correlated_with_selection_summary():
    """The bounded late rejection record retains the concrete adapter reason."""

    bridge = PopulationShadowRuntime(enabled=True)
    assert bridge.observe(
        result=_result(_evidence()), belief=_belief(),
        token=MutationWorkToken(8, 1), current_versions=_versions(), capacity=2,
    ) is None
    observation = bridge.take_latest_activation_observation()
    assert observation is not None

    bridge.drain_async_summaries()
    bridge.record_adapter_rejection(
        observation, plan_drop_reason=PopulationPlanDropReason.CAPACITY_CONFLICT
    )
    summary = bridge.drain_async_summaries()[0]

    assert summary.adapter_rejected is True
    assert summary.adapter_plan_drop_reason == "capacity_conflict"
    assert summary.action_counts_cpu is None
    assert summary.native_target_count is None
    assert summary.native_target_status_cpu is None


def test_adapter_rejection_is_bounded_and_survives_drain_before_actuation():
    """Overflow cannot leak decision ids or lose a late fence rejection."""

    bridge = PopulationShadowRuntime(enabled=True, max_pending_telemetry=1)
    for _ in range(100):
        assert bridge.observe(
            result=_result(_evidence()), belief=_belief(),
            token=MutationWorkToken(8, 1), current_versions=_versions(), capacity=2,
        ) is None
    observation = bridge.take_latest_activation_observation()
    assert observation is not None
    assert bridge.pending_event_count == 1
    # The original telemetry is already gone from the queue when the fence
    # rejects.  A stable decision token must publish one bounded late record.
    pending = bridge.drain_async_summaries()[0]
    assert pending.adapter_rejected is None
    assert pending.decision_token == observation.decision_token
    bridge.record_adapter_rejection(observation)
    summaries = bridge.drain_async_summaries()
    assert len(summaries) == 1
    assert summaries[0].adapter_rejected is True
    assert summaries[0].decision_token == observation.decision_token
    assert summaries[0].action_counts_cpu is None
    assert summaries[0].native_target_count is None
    assert summaries[0].native_target_status_cpu is None
    assert bridge.pending_event_count == 0
    assert not hasattr(bridge, "_adapter_rejected_decisions")


def test_live_e0_accumulator_rejection_blocks_stale_belief_reuse():
    """A rejected true-B4 accumulator cannot be followed by an old-row plan."""

    bridge = PopulationShadowRuntime(enabled=True)
    belief = _belief()
    belief.observability.meaningful_exposure_count.fill_(1)
    belief.observability.independent_support_views.fill_(2)
    result = _result(_evidence())
    result.population_policy_e0_accumulator_status = "invalid_evidence"

    assert bridge.observe(
        result=result, belief=belief, token=MutationWorkToken(8, 1),
        current_versions=_versions(), capacity=4,
    ) is PopulationShadowDropReason.INVALID_E0_ACCUMULATOR
    assert bridge.take_latest_activation_observation() is None


def test_bridge_drops_compact_rows_until_atomic_row_id_mapping_is_live():
    bridge = PopulationShadowRuntime(enabled=True)
    drop_reason = bridge.observe(
        result=_result(_evidence(gaussian_count=1)), belief=_belief(2),
        token=MutationWorkToken(epoch=8, units=1), current_versions=_versions(), capacity=2,
    )

    assert drop_reason is PopulationShadowDropReason.INVALID_EVIDENCE_SHAPE


def test_bridge_drops_evidence_on_a_different_device_without_copying_it():
    token = _evidence().snapshot_token
    evidence = SupervisionEvidenceBatch(
        snapshot_token=token,
        transaction_id="transaction-meta",
        modalities=(SupervisionModality(),),
        rgb_residual_summary=torch.zeros(1, device="meta"),
    )
    bridge = PopulationShadowRuntime(enabled=True)

    drop_reason = bridge.observe(
        result=_result(evidence), belief=_belief(),
        token=MutationWorkToken(epoch=8, units=1), current_versions=_versions(), capacity=2,
    )

    assert drop_reason is PopulationShadowDropReason.INVALID_EVIDENCE_SHAPE


def test_observe_never_invokes_an_arbitrary_hostile_sink():
    calls = []

    def hostile_sink(_event):
        calls.append("called")
        raise AssertionError("transaction-time callback is forbidden")

    bridge = PopulationShadowRuntime(enabled=True)
    # A legacy/injected attribute cannot become an execution hook: observation
    # only owns its internal bounded device-event queue.
    bridge.telemetry_sink = hostile_sink
    drop_reason = bridge.observe(
        result=_result(_evidence()), belief=_belief(),
        token=MutationWorkToken(epoch=8, units=0), current_versions=_versions(), capacity=2,
    )

    assert drop_reason is None
    assert calls == []


def test_streaming_runtime_owns_opt_in_bridge_but_never_calls_legacy_actuators():
    legacy_mcmc = SimpleNamespace(calls=0)
    belief = _belief()
    runtime = StreamingOptimizationRuntime(
        session=SimpleNamespace(model=object()), gaussians=object(), args=SimpleNamespace(),
        opt=object(), pipe=object(), mcmc_strategy=legacy_mcmc, tb_writer=None,
        optimizer_type="adam", sparse_active_set=False, use_energy_mcmc=False,
        fixed_shape_mcmc=False, mcmc_local=False, loss_assembly_kwargs={}, iter_since_new_frame=0,
        population_shadow_enabled=True,
        population_shadow_work_token_provider=lambda _result, _context: MutationWorkToken(8, 0),
        population_shadow_versions_provider=lambda _result, _context: _versions(),
        population_shadow_belief_provider=lambda _gaussians: belief,
        population_shadow_capacity=2,
    )
    _request, context = runtime.build_step(
        camera="camera", background="background", iteration=1, map_version=4,
        topology_version=5, resolution_bucket="native", update_sh_rest=False,
        render_depth=False, n_frames_ingested=1, visible_mask=None,
        random_state_token=SimpleNamespace(), optimizer_update_id=8,
        optimizer_update_permitted=True,
    )

    runtime.commit_step(context, _result(_evidence()))

    assert runtime.last_population_shadow_drop_reason is None
    assert runtime.population_shadow_runtime is not None
    assert runtime.population_shadow_runtime.pending_event_count == 1
    assert legacy_mcmc.calls == 0
