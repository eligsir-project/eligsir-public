"""Production-only M5 shadow bridge wiring contracts.

These tests deliberately exercise the canonical composition adapters rather
than the bridge in isolation.  The bridge is telemetry-only: no test here
permits a mutation proposal, MCMC call, or scheduler side effect.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

from eligsir.contracts.work_clocks import RuntimeWorkClockOwner
from eligsir.map.row_state import GaussianRowStateRegistry, RowStateDescriptor
from eligsir.mapping.optimization_runtime import StreamingOptimizationRuntime
from eligsir.mapping.population_policy import PopulationAction
from eligsir.mapping.population_shadow_runtime import PopulationShadowDropReason
from eligsir.mapping.scheduler.mcmc_cadence import MCMCCadenceOwner
from eligsir.optimization.gaussian.supervision_evidence import (
    EvidenceSnapshotToken,
    SupervisionEvidenceBatch,
    SupervisionModality,
)
from eligsir.runtime.production_application import CanonicalMappingRuntime


def _rows(n: int = 2, *, with_optional_exposure: bool = False) -> SimpleNamespace:
    """Small, complete pre-existing row state for the read-only adapter."""

    rows = SimpleNamespace(
        get_xyz=torch.zeros(n, 3),
        gaussian_uid=torch.arange(n, dtype=torch.int64),
        birth_iter=torch.zeros(n, dtype=torch.int32),
        birth_method=torch.zeros(n, dtype=torch.int8),
        parent_uid=torch.full((n,), -1, dtype=torch.int64),
        root_uid=torch.arange(n, dtype=torch.int64),
        generation=torch.zeros(n, dtype=torch.int32),
        support_count=torch.zeros(n, dtype=torch.int16),
        population_policy_exposure_count=torch.zeros(n, dtype=torch.int32),
        population_policy_support_count=torch.zeros(n, dtype=torch.int16),
        baseline_diversity=torch.zeros(n),
        anchor_confidence=torch.ones(n),
        visibility_ema=torch.ones(n),
        utility_ema=torch.zeros(n),
        birth_anchor_xyz=torch.zeros(n, 3),
        last_exposed_update=torch.zeros(n, dtype=torch.int32),
        support_evidence=torch.zeros(n),
        contradiction_evidence=torch.zeros(n),
        p_exist=torch.ones(n),
        surface_anchor_id=torch.full((n,), -1, dtype=torch.int64),
        surface_position_uncertainty=torch.zeros(n),
        surface_normal_uncertainty=torch.zeros(n),
        rgb_residual_ema=torch.zeros(n),
        depth_residual_ema=torch.zeros(n),
        redundancy=torch.zeros(n),
        expected_marginal_gain=torch.zeros(n),
        evidence_recency=torch.zeros(n, dtype=torch.int32),
        change_probability=torch.zeros(n),
        confidence_decay=torch.zeros(n),
    )
    registry_names = (
        "provenance.gaussian_uid", "provenance.birth_iter",
        "provenance.birth_method", "provenance.parent_uid",
        "provenance.root_uid", "provenance.generation",
        "belief.birth_anchor_xyz", "provenance.support_count",
        "belief.population_policy_exposure_count",
        "belief.population_policy_support_count",
        "belief.baseline_diversity", "belief.last_exposed_update",
        "belief.support_evidence", "belief.contradiction_evidence",
        "belief.p_exist", "belief.surface_anchor_id",
        "surface.anchor_confidence", "belief.surface_position_uncertainty",
        "belief.surface_normal_uncertainty", "render.visibility_ema",
        "belief.rgb_residual_ema", "belief.depth_residual_ema",
        "provenance.utility_ema", "belief.redundancy",
        "belief.expected_marginal_gain", "belief.evidence_recency",
        "belief.change_probability", "belief.confidence_decay",
    )
    if with_optional_exposure:
        rows.exposure_count = torch.ones(n, dtype=torch.int32)
        rows.supervised_tier_high_water = torch.ones(n, dtype=torch.int8)
        rows.last_supervised_iter = torch.full((n,), 8, dtype=torch.int32)
        registry_names += (
            "compatibility.exposure_count",
            "compatibility.supervised_tier_high_water",
            "compatibility.last_supervised_iter",
        )
    rows.build_row_state_registry = lambda: GaussianRowStateRegistry(
        RowStateDescriptor(
            name=name, family="test", owner="test", shape_suffix=(),
            dtype=torch.float32, checkpoint=True, default="zero", create="zero",
            grow="zero", relocate="zero", retire="survive",
        )
        for name in registry_names
    )
    return rows


def _production_runtime(
    *, enabled: bool, live: bool = False, parameter: int = 12, units: int = 0
):
    runtime = object.__new__(CanonicalMappingRuntime)
    runtime.config = SimpleNamespace(
        static_options={
            "population_shadow_enabled": enabled,
            "population_live_enabled": live,
        },
    )
    runtime.backend = SimpleNamespace(
        map_content_version=4,
        topology_version=5,
        coordinate_frame_version=6,
        schema_version=7,
    )
    runtime._engine = SimpleNamespace(
        session=SimpleNamespace(parameter_version=parameter),
    )
    runtime._work_clock_owner = RuntimeWorkClockOwner()
    runtime._work_clock_owner.record_optimization(
        optimizer_updates=8,
        supervised_views=1,
        supervised_pixels=16,
        mapping_active_ns=0,
    )
    runtime._population_shadow_surface_evidence_version = 9
    runtime._population_shadow_policy_version = 10
    runtime._population_shadow_active_plan = SimpleNamespace(
        mutation_budget=SimpleNamespace(max_total=units),
    )
    runtime._runtime_policy = SimpleNamespace(cap_max=2)
    return runtime


def _e0() -> SupervisionEvidenceBatch:
    return SupervisionEvidenceBatch(
        snapshot_token=EvidenceSnapshotToken(
            map_content=4,
            topology=5,
            coordinate_frame=6,
            schema=7,
            optimizer_update=8,
            surface_evidence=9,
            policy=10,
            parameter=11,
        ),
        transaction_id="production-b4-8",
        modalities=(SupervisionModality(),),
        rgb_residual_summary=torch.zeros(1),
    )


def _result() -> SimpleNamespace:
    return SimpleNamespace(
        evidence=_e0(),
        map_version_before=4,
        map_version_after=4,
        topology_version=5,
        parameter_version_before=11,
        parameter_version_after=12,
        optimizer_step_count=1,
    )


def _streaming_runtime(production: CanonicalMappingRuntime) -> StreamingOptimizationRuntime:
    return StreamingOptimizationRuntime(
        session=production._engine.session,
        gaussians=_rows(),
        args=SimpleNamespace(),
        opt=object(),
        pipe=object(),
        mcmc_strategy=SimpleNamespace(),
        tb_writer=None,
        optimizer_type="adam",
        sparse_active_set=False,
        use_energy_mcmc=False,
        fixed_shape_mcmc=False,
        mcmc_local=False,
        loss_assembly_kwargs={},
        iter_since_new_frame=0,
        **production._population_shadow_runtime_kwargs(),
    )


def test_production_shadow_flag_is_disabled_by_default_and_preserves_evidence_compatibility():
    production = _production_runtime(enabled=False)
    runtime = _streaming_runtime(production)

    request, _context = runtime.build_step(
        camera="camera",
        background="background",
        iteration=8,
        map_version=4,
        topology_version=5,
        resolution_bucket="native",
        update_sh_rest=False,
        render_depth=False,
        n_frames_ingested=1,
        visible_mask=None,
        random_state_token=SimpleNamespace(),
        optimizer_update_id=8,
        optimizer_update_permitted=True,
    )

    assert not runtime.population_shadow_enabled
    assert runtime.population_shadow_runtime is None
    assert not request.evidence_requirements.want_supervision_evidence


def test_historical_frame_value_request_enables_e0_without_population_shadow():
    """Frame-value E0 demand must not turn on the population-shadow policy."""

    production = _production_runtime(enabled=False)
    runtime = _streaming_runtime(production)
    runtime.args.streaming_historical_usefulness_tracking = True

    request, _context = runtime.build_step(
        camera="camera",
        background="background",
        iteration=8,
        map_version=4,
        topology_version=5,
        resolution_bucket="native",
        update_sh_rest=False,
        render_depth=False,
        n_frames_ingested=1,
        visible_mask=None,
        random_state_token=SimpleNamespace(),
        optimizer_update_id=8,
        optimizer_update_permitted=True,
    )

    assert not runtime.population_shadow_enabled
    assert runtime.population_shadow_runtime is None
    assert request.evidence_requirements.want_supervision_evidence


def test_enabled_production_shadow_uses_current_rows_and_emits_telemetry_only():
    production = _production_runtime(enabled=True)
    runtime = _streaming_runtime(production)
    request, context = runtime.build_step(
        camera="camera",
        background="background",
        iteration=8,
        map_version=4,
        topology_version=5,
        resolution_bucket="native",
        update_sh_rest=False,
        render_depth=False,
        n_frames_ingested=1,
        visible_mask=None,
        random_state_token=SimpleNamespace(),
        optimizer_update_id=8,
        optimizer_update_permitted=True,
    )

    assert request.evidence_requirements.want_supervision_evidence
    result = _result()
    runtime.commit_step(context, result)
    drop = production._observe_population_shadow_after_step(runtime, result)

    assert drop is None
    assert runtime.population_shadow_runtime is not None
    assert runtime.population_shadow_runtime.pending_event_count == 1
    summary = runtime.population_shadow_runtime.drain_async_summaries()[0]
    assert summary.units == 0
    assert summary.demand_count == 2


def test_post_adam_map_commit_rebinds_completed_result_for_the_shadow_fence():
    """A completed Adam result must carry the committed map-content version."""

    production = _production_runtime(enabled=True)
    runtime = _streaming_runtime(production)
    _request, context = runtime.build_step(
        camera="camera", background="background", iteration=8,
        map_version=4, topology_version=5, resolution_bucket="native",
        update_sh_rest=False, render_depth=False, n_frames_ingested=1,
        visible_mask=None, random_state_token=SimpleNamespace(),
        optimizer_update_id=8, optimizer_update_permitted=True,
    )
    result = _result()

    # E0 and the engine result are both stamped before Adam commits map content.
    assert result.evidence.snapshot_token.map_content == result.map_version_after == 4
    production.backend.map_content_version = 5
    production._stamp_completed_result_map_version_after(result)
    runtime.commit_step(context, result)

    assert result.map_version_before == 4
    assert result.map_version_after == 5
    assert production._observe_population_shadow_after_step(runtime, result) is None


def test_later_map_advance_remains_stale_after_post_adam_result_rebind():
    """A result rebound to its own Adam commit cannot authorize a later commit."""

    production = _production_runtime(enabled=True)
    runtime = _streaming_runtime(production)
    _request, context = runtime.build_step(
        camera="camera", background="background", iteration=8,
        map_version=4, topology_version=5, resolution_bucket="native",
        update_sh_rest=False, render_depth=False, n_frames_ingested=1,
        visible_mask=None, random_state_token=SimpleNamespace(),
        optimizer_update_id=8, optimizer_update_permitted=True,
    )
    result = _result()
    production.backend.map_content_version = 5
    production._stamp_completed_result_map_version_after(result)
    production.backend.map_content_version = 6
    runtime.commit_step(context, result)

    drop = production._observe_population_shadow_after_step(runtime, result)

    assert result.map_version_after == 5
    assert drop is PopulationShadowDropReason.STALE_CURRENT_SNAPSHOT
    assert runtime.last_population_shadow_stale_current_mismatches == (("map", 5, 6),)


def test_production_shadow_drops_a_stale_current_parameter_snapshot():
    production = _production_runtime(enabled=True, parameter=13)
    runtime = _streaming_runtime(production)
    _request, context = runtime.build_step(
        camera="camera", background="background", iteration=8,
        map_version=4, topology_version=5, resolution_bucket="native",
        update_sh_rest=False, render_depth=False, n_frames_ingested=1,
        visible_mask=None, random_state_token=SimpleNamespace(),
        optimizer_update_id=8, optimizer_update_permitted=True,
    )

    result = _result()
    runtime.commit_step(context, result)
    drop = production._observe_population_shadow_after_step(runtime, result)

    assert drop is PopulationShadowDropReason.STALE_CURRENT_SNAPSHOT
    assert runtime.last_population_shadow_stale_current_mismatches == (
        ("parameter", 12, 13),
    )


def test_production_shadow_zero_scheduler_budget_still_mints_only_abstention_and_demands():
    production = _production_runtime(enabled=True)
    runtime = _streaming_runtime(production)
    _request, context = runtime.build_step(
        camera="camera", background="background", iteration=8,
        map_version=4, topology_version=5, resolution_bucket="native",
        update_sh_rest=False, render_depth=False, n_frames_ingested=1,
        visible_mask=None, random_state_token=SimpleNamespace(),
        optimizer_update_id=8, optimizer_update_permitted=True,
    )

    result = _result()
    assert production._population_shadow_work_token(result, context).units == 0
    runtime.commit_step(context, result)
    assert production._observe_population_shadow_after_step(runtime, result) is None
    summary = runtime.population_shadow_runtime.drain_async_summaries()[0]
    assert summary.units == 0
    assert summary.demand_count == 2
    assert summary.action_counts_cpu.sum() == 0


def test_live_shadow_token_uses_depth_opportunity_without_waiting_for_rgb_mcmc_due():
    """Live M5 work is authorized by the scheduler without choosing an action."""

    production = _production_runtime(enabled=True, live=True, units=1)
    production._mcmc_cadence_owner = MCMCCadenceOwner(
        grow_interval_pixels=100,
        relocate_interval_pixels=100,
        min_optimizer_updates_after_structural_mutation=0,
    )
    production._work_clock_owner.record_depth_supervision(16)
    result = SimpleNamespace(
        depth_valid_sample_count=16,
        realized_batch_size=4,
        realized_camera_count=4,
        supervision_view_count=4,
        optimizer_step_count=1,
        evidence=_e0(),
    )
    context = SimpleNamespace(optimizer_update_id=8)

    token = production._population_shadow_work_token(result, context)

    assert token is not None
    assert token.epoch == 8
    assert token.units == 1
    assert token.geometry_opportunity is True


def test_live_shadow_no_scheduler_opportunity_mints_an_abstention_token():
    """A non-due live epoch cannot fall through to a structural action."""

    production = _production_runtime(enabled=True, live=True, units=1)
    production._mcmc_cadence_owner = MCMCCadenceOwner(
        grow_interval_pixels=100,
        relocate_interval_pixels=100,
        min_optimizer_updates_after_structural_mutation=0,
    )
    result = SimpleNamespace(
        depth_valid_sample_count=0,
        realized_batch_size=4,
        realized_camera_count=4,
        supervision_view_count=4,
        optimizer_step_count=1,
        evidence=_e0(),
    )
    context = SimpleNamespace(optimizer_update_id=8)

    token = production._population_shadow_work_token(result, context)

    assert token is not None
    assert token.units == 0
    assert token.geometry_opportunity is False


def test_live_shadow_without_a_cadence_owner_mints_explicit_abstention_telemetry():
    """Missing live scheduler ownership cannot reuse a plan-budget token."""

    production = _production_runtime(enabled=True, live=True, units=1)
    runtime = _streaming_runtime(production)
    _request, context = runtime.build_step(
        camera="camera", background="background", iteration=8,
        map_version=4, topology_version=5, resolution_bucket="native",
        update_sh_rest=False, render_depth=False, n_frames_ingested=1,
        visible_mask=None, random_state_token=SimpleNamespace(),
        optimizer_update_id=8, optimizer_update_permitted=True,
    )
    result = _result()
    runtime.commit_step(context, result)

    drop = production._observe_population_shadow_after_step(runtime, result)

    assert drop is None
    activation = runtime.population_shadow_runtime.take_latest_activation_observation()
    assert activation is not None
    assert activation.token.units == 0
    summary = runtime.population_shadow_runtime.drain_async_summaries()[0]
    assert summary.units == 0
    assert summary.action_counts_cpu.sum() == 0


def test_population_belief_adapter_drops_missing_row_without_allocating_or_repairing_it():
    rows = _rows()
    del rows.p_exist
    before_names = frozenset(vars(rows))
    before_ids = {name: id(value) for name, value in vars(rows).items()}

    belief = CanonicalMappingRuntime._population_shadow_belief_from_rows(rows)

    assert belief is None
    assert frozenset(vars(rows)) == before_names
    assert {name: id(value) for name, value in vars(rows).items()} == before_ids


def test_population_belief_adapter_drops_mismatched_row_without_replacing_it():
    rows = _rows()
    rows.depth_residual_ema = torch.zeros(1)
    before = rows.depth_residual_ema

    belief = CanonicalMappingRuntime._population_shadow_belief_from_rows(rows)

    assert belief is None
    assert rows.depth_residual_ema is before


def test_population_belief_adapter_views_registered_column_visibility_without_copy():
    rows = _rows()
    rows.visibility_ema = torch.ones(2, 1)
    registry = rows.build_row_state_registry()
    rows.build_row_state_registry = lambda: GaussianRowStateRegistry(
        RowStateDescriptor(
            name=descriptor.name,
            family=descriptor.family,
            owner=descriptor.owner,
            shape_suffix=(1,) if descriptor.name == "render.visibility_ema" else descriptor.shape_suffix,
            dtype=rows.visibility_ema.dtype if descriptor.name == "render.visibility_ema" else descriptor.dtype,
            checkpoint=descriptor.checkpoint,
            default=descriptor.default,
            create=descriptor.create,
            grow=descriptor.grow,
            relocate=descriptor.relocate,
            retire=descriptor.retire,
        )
        for descriptor in registry.descriptors
    )

    belief = CanonicalMappingRuntime._population_shadow_belief_from_rows(rows)

    assert belief is not None
    contribution = belief.representation.render_contribution
    assert contribution.shape == (2,)
    assert contribution.untyped_storage().data_ptr() == rows.visibility_ema.untyped_storage().data_ptr()


def test_population_belief_adapter_drops_noncanonical_visibility_suffix():
    rows = _rows()
    rows.visibility_ema = torch.ones(2, 2)
    registry = rows.build_row_state_registry()
    rows.build_row_state_registry = lambda: GaussianRowStateRegistry(
        RowStateDescriptor(
            name=descriptor.name,
            family=descriptor.family,
            owner=descriptor.owner,
            shape_suffix=(2,) if descriptor.name == "render.visibility_ema" else descriptor.shape_suffix,
            dtype=rows.visibility_ema.dtype if descriptor.name == "render.visibility_ema" else descriptor.dtype,
            checkpoint=descriptor.checkpoint,
            default=descriptor.default,
            create=descriptor.create,
            grow=descriptor.grow,
            relocate=descriptor.relocate,
            retire=descriptor.retire,
        )
        for descriptor in registry.descriptors
    )

    belief = CanonicalMappingRuntime._population_shadow_belief_from_rows(rows)

    assert belief is None


def test_population_belief_adapter_exposes_dedicated_policy_e0_counts():
    belief = CanonicalMappingRuntime._population_shadow_belief_from_rows(_rows())

    assert belief is not None
    assert torch.equal(
        belief.observability.meaningful_exposure_count,
        torch.zeros(2, dtype=torch.int32),
    )
    assert belief.observability.highest_supervised_tier is None
    assert belief.observability.last_supported_update is None


def test_registered_policy_e0_counts_allow_shadow_policy_rows_to_avoid_abstention():
    production = _production_runtime(enabled=True, units=1)
    runtime = _streaming_runtime(production)
    runtime.gaussians = _rows(with_optional_exposure=True)
    runtime.gaussians.population_policy_exposure_count.fill_(1)
    runtime.gaussians.population_policy_support_count.fill_(1)
    _request, context = runtime.build_step(
        camera="camera", background="background", iteration=8,
        map_version=4, topology_version=5, resolution_bucket="native",
        update_sh_rest=False, render_depth=False, n_frames_ingested=1,
        visible_mask=None, random_state_token=SimpleNamespace(),
        optimizer_update_id=8, optimizer_update_permitted=True,
    )
    result = _result()
    runtime.commit_step(context, result)

    assert production._observe_population_shadow_after_step(runtime, result) is None
    summary = runtime.population_shadow_runtime.drain_async_summaries()[0]
    assert summary.action_counts_cpu.sum() == 0


def test_population_belief_adapter_drops_stale_registered_policy_e0_without_repair():
    rows = _rows(with_optional_exposure=True)
    rows.population_policy_exposure_count = torch.ones(1, dtype=torch.int32)
    before = rows.population_policy_exposure_count

    belief = CanonicalMappingRuntime._population_shadow_belief_from_rows(rows)

    assert belief is None
    assert rows.population_policy_exposure_count is before
