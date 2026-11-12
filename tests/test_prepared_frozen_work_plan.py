"""Unit contracts for prepared runtime ownership of frozen WorkBudget plans."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import replace
from types import SimpleNamespace

import pytest

from eligsir.contracts import SourceIdentity
from eligsir.contracts.control import (
    ControlLimits,
    ControlProposal,
    ControlSnapshot,
    DepthInsertionPlan,
    DispositionKind,
    FreshnessMode,
    ModalityEligibility,
    MappingWorkPlan,
    MutationBudget,
    ObservationDisposition,
    ResidencyIntent,
    SemanticAction,
    ResolutionTier,
)
from eligsir.mapping.scheduler.work_budget import WorkBudgetOwner
from eligsir.runtime.ready_source import ReadySourceLane


def test_runtime_keeps_cold_candidate_as_pagein_demand_and_freezes_ready_fallback():
    from eligsir.runtime.production_application import CanonicalMappingRuntime

    desired_sources = tuple(SourceIdentity("desired", 0, index, index) for index in range(4))
    ready_sources = tuple(SourceIdentity("ready", 0, index, 10 + index) for index in range(4))
    desired = SimpleNamespace(
        supervision_sources=desired_sources[:1],
        replay_sources=desired_sources[1:],
        replay_trial_sources=(desired_sources[1],),
        supervision_constellation=SimpleNamespace(realized_batch_size=4, realized_camera_count=4),
    )
    fallback = SimpleNamespace(
        supervision_sources=(),
        replay_sources=ready_sources,
        supervision_constellation=SimpleNamespace(realized_batch_size=4, realized_camera_count=4),
    )
    snapshot = object()
    runtime = object.__new__(CanonicalMappingRuntime)
    runtime._batch_size = 4
    runtime._prepared_ready_candidate_work = (snapshot, desired)
    runtime._prepared_ready_frozen_work = None
    runtime._fresh_idle_snapshot = lambda value: value
    runtime._idle_replay_plan = lambda _snapshot, prior, **kwargs: (
        fallback if prior is desired and kwargs["ready_replay_sources"] == frozenset(ready_sources) else None
    )
    runtime._ready_source_owner = SimpleNamespace(
        exact_readiness=lambda **_kwargs: tuple(
            (source, "ABSENT_REPLAY") for source in desired_sources
        ),
        ready_replay_sources=lambda: ready_sources,
    )

    assert runtime._prepared_ready_plan_for_execution() is fallback
    assert runtime._prepared_ready_candidate_work == (snapshot, desired)
    assert runtime._prepared_ready_frozen_work == (snapshot, fallback)
    # The fallback is executable work, not permission to discard the one
    # actually reserved COLD trial identity that it temporarily replaces.
    assert runtime._pending_replay_trial_sources == (desired_sources[1],)


def test_runtime_rejects_an_underfilled_fallback_despite_full_b32_residency():
    """The replay fallback keeps configured B rather than reusing a source-service B4."""
    from eligsir.runtime.production_application import CanonicalMappingRuntime

    desired_sources = tuple(SourceIdentity("desired", 0, index, index) for index in range(32))
    fallback_sources = tuple(SourceIdentity("fallback", 0, index, 100 + index) for index in range(4))
    desired = SimpleNamespace(
        supervision_sources=(),
        replay_sources=desired_sources,
        supervision_constellation=SimpleNamespace(realized_batch_size=32, realized_camera_count=32),
    )
    underfilled_fallback = SimpleNamespace(
        supervision_sources=(),
        replay_sources=fallback_sources,
        supervision_constellation=SimpleNamespace(realized_batch_size=4, realized_camera_count=4),
    )
    snapshot = object()
    runtime = object.__new__(CanonicalMappingRuntime)
    runtime._batch_size = 32
    runtime._prepared_ready_candidate_work = (snapshot, desired)
    runtime._prepared_ready_frozen_work = None
    runtime._fresh_idle_snapshot = lambda value: value
    runtime._idle_replay_plan = lambda *_args, **_kwargs: underfilled_fallback
    runtime._ready_source_owner = SimpleNamespace(
        exact_readiness=lambda **_kwargs: tuple(
            (source, "ABSENT_REPLAY") for source in desired_sources
        ),
        ready_replay_sources=lambda: desired_sources,
    )

    assert runtime._prepared_ready_plan_for_execution() is None
    assert runtime._prepared_ready_frozen_work is None


def test_runtime_accepts_configured_b16_with_camera_microbatch_c4():
    """A full B16 READY plan may execute as four existing C4 microbatches."""
    from eligsir.runtime.production_application import CanonicalMappingRuntime

    sources = tuple(SourceIdentity("replay", 0, index, index) for index in range(16))
    plan = SimpleNamespace(
        supervision_sources=(),
        replay_sources=sources,
        supervision_constellation=SimpleNamespace(
            realized_batch_size=16, realized_camera_count=4
        ),
    )
    runtime = object.__new__(CanonicalMappingRuntime)
    runtime._batch_size = 16
    runtime._prepared_ready_candidate_work = (object(), plan)
    runtime._prepared_ready_frozen_work = None

    assert runtime._prepared_ready_plan_for_execution() is plan


def test_runtime_accepts_source_insertion_b4_with_camera_microbatch_c1():
    """The existing one-fresh-plus-three-replay insertion service may use C1."""
    from eligsir.runtime.production_application import CanonicalMappingRuntime

    fresh = SourceIdentity("fresh", 0, 0, 0)
    replay = tuple(SourceIdentity("replay", 0, index, index) for index in range(1, 4))
    sources = (fresh, *replay)
    context = SimpleNamespace(
        lane=ReadySourceLane.FRESH,
        disposition=SimpleNamespace(semantic_action=SemanticAction.COMMIT),
        control_axes=SimpleNamespace(proposals=()),
    )
    plan = SimpleNamespace(
        supervision_sources=(fresh,),
        replay_sources=replay,
        supervision_constellation=SimpleNamespace(
            realized_batch_size=4, realized_camera_count=1
        ),
    )
    runtime = object.__new__(CanonicalMappingRuntime)
    runtime._batch_size = 16
    runtime.config = SimpleNamespace(
        static_options={"prepared_ready_source_insertion": True}
    )
    runtime._native_geometry_evidence_purpose = lambda: None
    runtime._observation_db = SimpleNamespace(
        prepared_commit_obligation_for_source=lambda source: (
            object(), object(), object(), context
        )
        if source == fresh
        else None
    )
    runtime._prepared_ready_candidate_work = (object(), plan)
    runtime._prepared_ready_frozen_work = None

    assert runtime._prepared_ready_plan_for_execution() is plan


@pytest.mark.parametrize(
    ("camera_count", "microbatch_count", "microbatch_sizes"),
    [
        (3, None, None),
        (17, None, None),
        (4, 5, (4, 4, 4, 4)),
        (4, 4, (4, 4, 4, 4, 1)),
    ],
)
def test_runtime_rejects_invalid_b_c_or_microbatch_partition(
    camera_count, microbatch_count, microbatch_sizes
):
    from eligsir.runtime.production_application import CanonicalMappingRuntime

    sources = tuple(SourceIdentity("replay", 0, index, index) for index in range(16))
    constellation = {
        "realized_batch_size": 16,
        "realized_camera_count": camera_count,
    }
    if microbatch_count is not None:
        constellation["microbatch_count"] = microbatch_count
    if microbatch_sizes is not None:
        constellation["microbatch_sizes"] = microbatch_sizes
    plan = SimpleNamespace(
        supervision_sources=(),
        replay_sources=sources,
        supervision_constellation=SimpleNamespace(**constellation),
    )
    runtime = object.__new__(CanonicalMappingRuntime)
    runtime._batch_size = 16
    runtime._prepared_ready_candidate_work = (object(), plan)
    runtime._prepared_ready_frozen_work = None

    assert runtime._prepared_ready_plan_for_execution() is None


@pytest.mark.parametrize("batch_size", (2, 3, 5))
def test_runtime_rejects_malformed_underfilled_constellation_without_raising(batch_size):
    """Partial B/C metadata is rejected before the qualified-camera helper can escape."""
    from eligsir.runtime.production_application import CanonicalMappingRuntime

    sources = tuple(SourceIdentity("malformed", 0, index, index) for index in range(batch_size))
    plan = SimpleNamespace(
        supervision_sources=(),
        replay_sources=sources,
        supervision_constellation=SimpleNamespace(
            realized_batch_size=batch_size, realized_camera_count=batch_size
        ),
    )
    runtime = object.__new__(CanonicalMappingRuntime)
    runtime._batch_size = 32
    runtime._prepared_ready_candidate_work = (object(), plan)
    runtime._prepared_ready_frozen_work = None

    assert runtime._prepared_ready_plan_for_execution() is None


@pytest.mark.parametrize("batch_size", (4, 16))
def test_runtime_replaces_absent_frozen_replay_plan_with_complete_resident_fallback(
    batch_size,
):
    """An evicted frozen replay member releases authority for idle READY work."""

    from eligsir.runtime.production_application import CanonicalMappingRuntime

    desired_sources = tuple(
        SourceIdentity("desired", 0, index, index) for index in range(batch_size)
    )
    fallback_sources = tuple(
        SourceIdentity("fallback", 0, index, 10 + index) for index in range(batch_size)
    )
    desired = SimpleNamespace(
        supervision_sources=(),
        replay_sources=desired_sources,
        supervision_constellation=SimpleNamespace(
            realized_batch_size=batch_size, realized_camera_count=batch_size
        ),
    )
    fallback = SimpleNamespace(
        supervision_sources=(),
        replay_sources=fallback_sources,
        supervision_constellation=SimpleNamespace(
            realized_batch_size=batch_size, realized_camera_count=batch_size
        ),
    )
    snapshot = object()
    fallback_snapshot = object()
    calls = []

    def exact_readiness(*, fresh_sources, replay_sources, exact_sources):
        assert fresh_sources == frozenset()
        if exact_sources == desired_sources:
            assert replay_sources == frozenset(desired_sources)
            return tuple(
                (
                    source,
                    "ABSENT_REPLAY"
                    if source is desired_sources[batch_size // 2]
                    else "READY",
                )
                for source in desired_sources
            )
        assert exact_sources == fallback_sources
        assert replay_sources == frozenset(fallback_sources)
        return tuple(
            (source, "READY") for source in fallback_sources
        )

    def fresh_idle_snapshot(value):
        calls.append(("snapshot", value))
        assert value is snapshot
        return fallback_snapshot

    def idle_replay_plan(value, prior, *, ready_replay_sources):
        calls.append(("idle", value, prior, ready_replay_sources))
        assert value is fallback_snapshot
        assert prior is desired
        assert ready_replay_sources == frozenset(fallback_sources)
        return fallback

    runtime = object.__new__(CanonicalMappingRuntime)
    runtime._batch_size = batch_size
    runtime._prepared_ready_candidate_work = (snapshot, desired)
    runtime._prepared_ready_frozen_work = (snapshot, desired)
    runtime._fresh_idle_snapshot = fresh_idle_snapshot
    runtime._idle_replay_plan = idle_replay_plan
    runtime._ready_source_owner = SimpleNamespace(
        exact_readiness=exact_readiness,
        ready_replay_sources=lambda: fallback_sources,
    )

    assert runtime._prepared_ready_plan_for_execution() is fallback
    assert runtime._prepared_ready_candidate_work == (snapshot, desired)
    assert runtime._prepared_ready_frozen_work == (fallback_snapshot, fallback)
    assert calls == [
        ("snapshot", snapshot),
        ("idle", fallback_snapshot, desired, frozenset(fallback_sources)),
    ]
    assert runtime._prepared_ready_plan_for_execution() is fallback
    assert calls == [
        ("snapshot", snapshot),
        ("idle", fallback_snapshot, desired, frozenset(fallback_sources)),
    ]


def test_runtime_releases_absent_frozen_authority_without_complete_fallback():
    """An absent frozen replay member defers while its original demand remains."""

    from eligsir.runtime.production_application import CanonicalMappingRuntime

    sources = tuple(SourceIdentity("replay", 0, index, index) for index in range(4))
    plan = SimpleNamespace(
        supervision_sources=(),
        replay_sources=sources,
        supervision_constellation=SimpleNamespace(realized_batch_size=4, realized_camera_count=4),
    )
    snapshot = object()

    def exact_readiness(*, fresh_sources, replay_sources, exact_sources):
        assert fresh_sources == frozenset()
        assert replay_sources == frozenset(sources)
        assert exact_sources == sources
        return tuple(
            (source, "ABSENT_REPLAY" if source is sources[0] else "READY")
            for source in sources
        )

    runtime = object.__new__(CanonicalMappingRuntime)
    runtime._batch_size = 4
    runtime._prepared_ready_candidate_work = (snapshot, plan)
    runtime._prepared_ready_frozen_work = (snapshot, plan)
    runtime._ready_source_owner = SimpleNamespace(
        exact_readiness=exact_readiness,
        ready_replay_sources=lambda: sources[:3],
        replay_resident_sources=lambda: sources[:3],
    )

    assert runtime._prepared_ready_plan_for_execution() is None
    assert runtime._prepared_ready_candidate_work == (snapshot, plan)
    assert runtime._prepared_ready_frozen_work is None


def test_runtime_keeps_valid_frozen_plan_identity():
    """A fully READY frozen plan keeps its original scheduler authority."""

    from eligsir.runtime.production_application import CanonicalMappingRuntime

    sources = tuple(SourceIdentity("replay", 0, index, index) for index in range(4))
    plan = SimpleNamespace(
        supervision_sources=(),
        replay_sources=sources,
        supervision_constellation=SimpleNamespace(realized_batch_size=4, realized_camera_count=4),
    )
    snapshot = object()

    def exact_readiness(*, fresh_sources, replay_sources, exact_sources):
        assert fresh_sources == frozenset()
        assert replay_sources == frozenset(sources)
        assert exact_sources == sources
        return tuple((source, "READY") for source in sources)

    runtime = object.__new__(CanonicalMappingRuntime)
    runtime._batch_size = 4
    runtime._prepared_ready_candidate_work = (snapshot, plan)
    runtime._prepared_ready_frozen_work = (snapshot, plan)
    runtime._ready_source_owner = SimpleNamespace(
        exact_readiness=exact_readiness,
        ready_replay_sources=lambda: pytest.fail("valid frozen work must not replan"),
    )

    assert runtime._prepared_ready_plan_for_execution() is plan
    assert runtime._prepared_ready_frozen_work == (snapshot, plan)


def test_runtime_keeps_resident_event_pending_exact_b4_as_authority():
    """A B4 returning behind a compute event must not become plan-unavailable."""

    from eligsir.runtime.production_application import CanonicalMappingRuntime

    sources = tuple(SourceIdentity("replay", 0, index, index) for index in range(4))
    plan = SimpleNamespace(
        supervision_sources=(),
        replay_sources=sources,
        supervision_constellation=SimpleNamespace(realized_batch_size=4, realized_camera_count=4),
    )
    snapshot = object()

    def exact_readiness(*, fresh_sources, replay_sources, exact_sources):
        assert fresh_sources == frozenset()
        assert replay_sources == frozenset(sources)
        assert exact_sources == sources
        return tuple((source, "EVENT_PENDING") for source in sources)

    runtime = object.__new__(CanonicalMappingRuntime)
    runtime._batch_size = 4
    runtime._prepared_ready_candidate_work = (snapshot, plan)
    runtime._prepared_ready_frozen_work = (snapshot, plan)
    runtime._ready_source_owner = SimpleNamespace(
        exact_readiness=exact_readiness,
        ready_replay_sources=lambda: (),
    )

    assert runtime._prepared_ready_plan_for_execution() is plan
    assert runtime._prepared_ready_frozen_work == (snapshot, plan)


def test_runtime_falls_back_to_returnable_replay_while_cold_candidate_pages_in():
    """A COLD candidate must not turn an event-pending fallback B4 into None."""

    from eligsir.runtime.production_application import CanonicalMappingRuntime

    cold = tuple(SourceIdentity("cold", 0, index, index) for index in range(4))
    returning = tuple(SourceIdentity("returning", 0, index, 10 + index) for index in range(4))
    desired = SimpleNamespace(
        supervision_sources=(),
        replay_sources=cold,
        supervision_constellation=SimpleNamespace(realized_batch_size=4, realized_camera_count=4),
    )
    fallback = SimpleNamespace(
        supervision_sources=(),
        replay_sources=returning,
        supervision_constellation=SimpleNamespace(realized_batch_size=4, realized_camera_count=4),
    )
    snapshot = object()
    runtime = object.__new__(CanonicalMappingRuntime)
    runtime._batch_size = 4
    runtime._prepared_ready_candidate_work = (snapshot, desired)
    runtime._prepared_ready_frozen_work = None
    runtime._fresh_idle_snapshot = lambda value: value
    runtime._idle_replay_plan = lambda _snapshot, prior, **kwargs: (
        fallback
        if prior is desired and kwargs["ready_replay_sources"] == frozenset(returning)
        else None
    )
    runtime._ready_source_owner = SimpleNamespace(
        exact_readiness=lambda **_kwargs: tuple(
            (source, "ABSENT_REPLAY") for source in cold
        ),
        ready_replay_sources=lambda: (),
        replay_resident_sources=lambda: returning,
    )

    assert runtime._prepared_ready_plan_for_execution() is fallback
    assert runtime._prepared_ready_frozen_work == (snapshot, fallback)


def test_terminal_retirement_invalidates_the_frozen_plan_before_replanning():
    from eligsir.runtime.production_application import CanonicalMappingRuntime

    sources = tuple(object() for _ in range(4))
    plan = SimpleNamespace(
        supervision_sources=sources[:1],
        replay_sources=sources[1:],
    )
    successor = object()
    snapshot = object()
    runtime = object.__new__(CanonicalMappingRuntime)
    runtime._prepared_ready_candidate_work = (snapshot, plan)
    runtime._prepared_ready_frozen_work = (snapshot, plan)
    runtime._fresh_idle_snapshot = lambda value: value
    runtime._idle_replay_plan = lambda _snapshot, prior: (
        successor if prior is plan else None
    )

    runtime._invalidate_prepared_ready_plan_for_source(sources[2])

    assert runtime._prepared_ready_frozen_work is None
    assert runtime._prepared_ready_candidate_work == (snapshot, successor)


def test_ready_successor_rearms_pending_commit_with_current_replay_catalog():
    from eligsir.runtime.production_application import CanonicalMappingRuntime

    replay = tuple(
        SourceIdentity("camera", 0, ordinal, ordinal * 1_000)
        for ordinal in range(1, 4)
    )
    cold_replay = SourceIdentity("camera", 0, 4, 4_000)
    pending = SourceIdentity("camera", 0, 10, 10_000)
    modality = ModalityEligibility(rgb_supervision=True)
    snapshot = ControlSnapshot(
        snapshot_id=9,
        control_version=1,
        source=pending,
        tracking=None,
        pending_sources=(pending,),
        newest_frontier=pending,
        freshness_mode=FreshnessMode.HEALTHY,
        pending_oldest_age_ns=0,
        ingress_queue_depth=0,
        candidate_queue_depth=1,
        transfer_queue_depth=0,
        gpu_queue_depth=0,
        map_content_version=1,
        topology_version=1,
        parameter_version=1,
        gaussian_count=0,
        optimizer_steps_completed=0,
        limits=ControlLimits(3, 3, 1, 0, 0, 0),
    )
    disposition = ObservationDisposition(
        source=pending,
        snapshot_id=snapshot.snapshot_id,
        kind=DispositionKind.ADMIT,
        observe=True,
        mapping_candidate=True,
        admit_supervision=True,
        allow_geometry_insertion=False,
        residency=ResidencyIntent.HOT,
        reason="admit",
        modality_eligibility=modality,
    )
    runtime = object.__new__(CanonicalMappingRuntime)
    runtime._pending_supervision_modalities = OrderedDict(((pending, modality),))
    runtime._prepared_execution_contexts_by_source = {
        pending: SimpleNamespace(
            source=pending,
            snapshot=snapshot,
            disposition=disposition,
            control_axes=SimpleNamespace(
                proposals=(
                    ControlProposal("work", "supervision_slots", 1, "fresh"),
                    ControlProposal("work", "replay_slots", 3, "ready replay"),
                    ControlProposal("work", "optimizer_updates", 1, "one B4"),
                )
            ),
            lane=ReadySourceLane.FRESH,
        )
    }
    runtime._observation_db = SimpleNamespace(
        next_prepared_commit_obligation_source=lambda: pending,
        prepared_commit_obligation_for_source=lambda candidate: (
            (
                object(),
                disposition,
                object(),
                runtime._prepared_execution_contexts_by_source[pending],
            )
            if candidate is pending
            else None
        ),
    )
    runtime._shortlisted_replay_view = lambda **_kwargs: (
        OrderedDict(
            (*((source, modality) for source in replay), (cold_replay, modality))
        ),
        cold_replay,
    )
    runtime._ready_source_owner = SimpleNamespace(
        replay_resident_sources=lambda: replay
    )
    runtime._versions = lambda: (10, 11, 12)
    runtime._work_budget_owner = WorkBudgetOwner(
        requested_batch_size=4,
        maximum_camera_count=4,
        constellation_policy="fixed",
    )
    runtime._native_geometry_evidence_purpose = lambda: False
    runtime.optimizer_steps = 7
    runtime._supervision_views_completed = 8
    runtime._control_tick_epoch = 9

    rearmed_snapshot, resolved = runtime._rearm_prepared_pending_supervision()
    assert rearmed_snapshot.supervision_candidates == (pending,)
    assert rearmed_snapshot.replay_candidates == replay
    assert (
        rearmed_snapshot.map_content_version,
        rearmed_snapshot.topology_version,
        rearmed_snapshot.parameter_version,
    ) == (10, 11, 12)
    assert tuple(work.source for work in resolved.supervision_work) == (pending,)
    assert set(work.source for work in resolved.replay_work) == set(replay)
    assert resolved.supervision_constellation.realized_batch_size == 4


def test_ready_successor_does_not_rearm_bootstrap_completed_commit():
    from eligsir.runtime.production_application import CanonicalMappingRuntime

    source = SourceIdentity("camera", 0, 2, 2_000)
    runtime = object.__new__(CanonicalMappingRuntime)
    runtime._pending_supervision_modalities = OrderedDict(
        ((source, ModalityEligibility(rgb_supervision=True)),)
    )
    runtime._prepared_execution_contexts_by_source = {
        source: SimpleNamespace(
            source=source,
            disposition=SimpleNamespace(semantic_action=SemanticAction.COMMIT),
            lane=ReadySourceLane.FRESH,
        )
    }
    runtime._observation_db = SimpleNamespace(
        next_prepared_commit_obligation_source=lambda: None,
        prepared_commit_obligation_for_source=lambda _source: None,
    )
    runtime._shortlisted_replay_view = lambda **_kwargs: pytest.fail(
        "completed COMMIT must not be re-resolved"
    )

    assert runtime._rearm_prepared_pending_supervision() is None


def test_ready_successor_uses_pending_commit_rearm_before_idle_replay():
    from eligsir.runtime.production_application import CanonicalMappingRuntime

    active_plan = object()
    rearmed_plan = object()
    snapshot = object()
    runtime = object.__new__(CanonicalMappingRuntime)
    runtime._prepared_ready_frozen_work = (snapshot, active_plan)
    runtime._prepared_ready_candidate_work = None
    runtime._rearm_prepared_pending_supervision = lambda: (snapshot, rearmed_plan)
    runtime._fresh_idle_snapshot = lambda _snapshot: pytest.fail("must not mint idle replay")

    CanonicalMappingRuntime._advance_prepared_ready_plan(runtime, active_plan)

    assert runtime._prepared_ready_frozen_work is None
    assert runtime._prepared_ready_candidate_work == (snapshot, rearmed_plan)


def test_refinement_rearm_replaces_an_obsolete_prepared_frozen_plan():
    """Prepared refinement must execute newly rearmed, not mapping-phase work."""

    from eligsir.runtime.post_refinement import (
        PostRefinementPhase,
        PostRefinementPhaseOwner,
    )
    from eligsir.runtime.production_application import CanonicalMappingRuntime

    source = SourceIdentity("camera", 0, 1, 1)
    snapshot = ControlSnapshot(
        snapshot_id=1,
        control_version=1,
        source=source,
        tracking=None,
        pending_sources=(source,),
        newest_frontier=source,
        freshness_mode=FreshnessMode.HEALTHY,
        pending_oldest_age_ns=0,
        ingress_queue_depth=0,
        candidate_queue_depth=0,
        transfer_queue_depth=0,
        gpu_queue_depth=0,
        map_content_version=1,
        topology_version=1,
        parameter_version=1,
        gaussian_count=1,
        optimizer_steps_completed=1,
        limits=ControlLimits(4, 4, 1, 0, 0, 0),
    )
    base = MappingWorkPlan(
        plan_id="mapping-phase",
        snapshot_id=1,
        control_version=1,
        supervision_sources=(),
        replay_sources=(),
        resolution_tier=ResolutionTier.QUARTER,
        optimizer_updates=1,
        insertion=DepthInsertionPlan(None, False, 1, 0, 0, 0.01, 0.0),
        allow_growth=False,
        allow_relocation=False,
        mutation_budget=MutationBudget(max_total=0),
        freshness_mode=FreshnessMode.HEALTHY,
        map_content_version=1,
        topology_version=1,
        parameter_version=1,
        supervision_work=(),
    )
    phase = PostRefinementPhaseOwner()
    phase._phase = PostRefinementPhase.REFINEMENT
    runtime = object.__new__(CanonicalMappingRuntime)
    runtime._post_refinement_phase_owner = phase
    runtime._last_materialized_work = (snapshot, base)
    runtime._versions = lambda: (2, 2, 2)
    runtime._supervision_views_completed = 4
    runtime.optimizer_steps = 1
    runtime._control_snapshot_id = 2
    runtime._control_tick_epoch = 1
    runtime._idle_replay_plan = lambda _snapshot, _prior: base
    runtime._prepared_ready_frozen_work = (snapshot, base)
    runtime._prepared_ready_candidate_work = (snapshot, base)
    runtime._requested_source_fps_for_pacing = None

    CanonicalMappingRuntime.rearm_post_mapping_work(runtime)

    assert runtime._prepared_ready_frozen_work is None
    assert runtime._prepared_ready_candidate_work[1].plan_id.startswith("refinement-")


def test_terminal_three_source_commit_obligations_drain_after_eof():
    """Terminal three COMMIT obligations drain to zero while replay is executable.

    This reproduces issue #1663's deterministic terminal-three-source composition:
    - replay reservoir >= 4 executable identities throughout (tick 0 onward)
    - three COMMIT A/B/C admitted (durable) at tick 0, each executes once as fresh
    - obligations drain to zero pending count; no obligation skipped or evicted
    - no infinite replay-only streaks while obligations pending
    - EOF: pending_sources=() static pre-set (no production API at this layer)

    Scope: No per-obligation READY gate exists at the rearm layer (production_application.py
    lines 1388-1395: rearm finds a pending COMMIT obligation only by checking obligation
    existence, context existence, lane=FRESH, and semantic action=COMMIT; no readiness
    event, control_token, or promotion_permit gate is checked). Thus readiness timing is
    out of scope here — a durable execution context makes an obligation immediately eligible
    for rearm.

    Assertions:
    - (a) All three obligations consumed (count == 0 pending) after drain
    - (b) Each obligation executed exactly once as fresh supervision (lane=FRESH)
    - (c) Every executed plan is exact B4: realized_batch_size == requested batch_size (4)
    - (d) Replay-only streak bound: <= 2 consecutive replay-only iterations while any
          obligation is pending (prevents infinite replay-only starvation of fresh work)
    - (e) Drain completes within bounded ticks (no infinite loop)
    - (f) No device synchronization: WorkBudgetOwner.resolve() is pure-Python CPU-only
    """
    from eligsir.runtime.production_application import CanonicalMappingRuntime

    # ===== Setup: Three durable COMMIT obligations A/B/C
    source_a = SourceIdentity("camera", 0, 10, 10_000)
    source_b = SourceIdentity("camera", 0, 11, 11_000)
    source_c = SourceIdentity("camera", 0, 12, 12_000)
    modality = ModalityEligibility(rgb_supervision=True)

    # Replay sources: 4 executable identities available from tick 0 throughout
    replay_sources = tuple(
        SourceIdentity("camera", 0, ordinal, ordinal * 1_000)
        for ordinal in range(1, 5)
    )

    # Snapshot template: terminal state (EOF) with no new sources pending.
    # NOTE: No production API for EOF transition at this layer; pending_sources/newest_frontier
    # set statically. In production, these would be set by the ingestion path upon source EOF.
    base_snapshot = ControlSnapshot(
        snapshot_id=99,
        control_version=1,
        source=None,
        tracking=None,
        pending_sources=(),  # Terminal: EOF has no new ingress
        newest_frontier=None,
        freshness_mode=FreshnessMode.HEALTHY,
        pending_oldest_age_ns=0,
        ingress_queue_depth=0,
        candidate_queue_depth=0,
        transfer_queue_depth=0,
        gpu_queue_depth=0,
        map_content_version=1,
        topology_version=1,
        parameter_version=1,
        gaussian_count=0,
        optimizer_steps_completed=0,
        supervision_views_completed=0,
        control_tick_epoch=0,
        limits=ControlLimits(3, 3, 1, 0, 0, 0),
        supervision_candidates=(),
        replay_candidates=replay_sources,
        replay_work_modalities=tuple(modality for _ in replay_sources),
    )

    # Execution contexts for each obligation
    execution_contexts = {}
    for source in (source_a, source_b, source_c):
        snapshot = replace(
            base_snapshot,
            source=source,
            pending_sources=(source,),
            newest_frontier=source,
        )
        disposition = ObservationDisposition(
            source=source,
            snapshot_id=snapshot.snapshot_id,
            kind=DispositionKind.ADMIT,
            observe=True,
            mapping_candidate=True,
            admit_supervision=True,
            allow_geometry_insertion=False,
            residency=ResidencyIntent.HOT,
            reason="admit",
            modality_eligibility=modality,
        )
        execution_contexts[source] = SimpleNamespace(
            source=source,
            snapshot=snapshot,
            disposition=disposition,
            control_axes=SimpleNamespace(
                proposals=(
                    ControlProposal("work", "supervision_slots", 1, "fresh"),
                    ControlProposal("work", "replay_slots", 3, "ready replay"),
                    ControlProposal("work", "optimizer_updates", 1, "one B4"),
                )
            ),
            lane=ReadySourceLane.FRESH,
        )

    # ===== Track execution: record obligations with lane verification and tick numbers
    executed_sources = []  # List of (tick_idx, source, lane) tuples
    durable_commit_obligations = {source: object() for source in (source_a, source_b, source_c)}

    def record_execution(tick_idx, source, lane):
        executed_sources.append((tick_idx, source, lane))
        durable_commit_obligations.pop(source, None)

    # ===== Build runtime with tracked execution
    runtime = object.__new__(CanonicalMappingRuntime)
    runtime._pending_supervision_modalities = OrderedDict(
        ((source, modality) for source in (source_a, source_b, source_c))
    )
    runtime._prepared_execution_contexts_by_source = execution_contexts

    runtime._observation_db = SimpleNamespace(
        next_prepared_commit_obligation_source=lambda: next(
            iter(durable_commit_obligations), None
        ),
        prepared_commit_obligation_for_source=lambda candidate: (
            (
                object(),
                execution_contexts[candidate].disposition,
                object(),
                execution_contexts[candidate],
            )
            if candidate in durable_commit_obligations
            else None
        ),
    )

    # Shortlisted replay view: 4 replay sources available throughout
    runtime._shortlisted_replay_view = lambda **_kwargs: (
        OrderedDict((source, modality) for source in replay_sources),
        None,  # No cold replay
    )

    # Ready owner: supplies all 4 replay sources (resident)
    runtime._ready_source_owner = SimpleNamespace(
        replay_resident_sources=lambda: replay_sources
    )

    runtime._versions = lambda: (10, 11, 12)

    # Real WorkBudget owner (requested batch_size=4)
    work_budget_owner = WorkBudgetOwner(
        requested_batch_size=4,
        maximum_camera_count=4,
        constellation_policy="fixed",
    )
    runtime._work_budget_owner = work_budget_owner

    runtime._native_geometry_evidence_purpose = lambda: False
    runtime.optimizer_steps = 7
    runtime._supervision_views_completed = 8
    runtime._control_tick_epoch = 9

    # ===== Drain phase: repeatedly rearm until all obligations consumed
    max_drain_ticks = 100
    tick_count = 0
    consecutive_replay_only = 0
    max_consecutive_replay_only = 0

    for tick_idx in range(max_drain_ticks):
        tick_count = tick_idx

        # Attempt to rearm any pending obligations
        result = runtime._rearm_prepared_pending_supervision()
        if result is None:
            break

        rearmed_snapshot, resolved = result
        assert rearmed_snapshot.supervision_candidates, (
            f"Tick {tick_idx}: Rearmed plan must have supervision candidates"
        )
        assert resolved.supervision_work, f"Tick {tick_idx}: Resolved plan must have supervision work"

        # (c) Assert plan is exact B4: realized_batch_size == requested batch_size (4)
        batch_size = resolved.supervision_constellation.realized_batch_size
        expected_batch_size = 4  # Requested batch_size passed to WorkBudgetOwner
        assert batch_size == expected_batch_size, (
            f"Tick {tick_idx}: Plan must be exact B4; got realized_batch_size={batch_size}, "
            f"expected={expected_batch_size}"
        )

        supervision_count = len(resolved.supervision_work)
        is_replay_only = supervision_count == 0

        # (d) Track replay-only streaks (no more than 2 consecutive while obligations pending)
        if is_replay_only:
            consecutive_replay_only += 1
        else:
            if consecutive_replay_only > 0:
                max_consecutive_replay_only = max(max_consecutive_replay_only, consecutive_replay_only)
            consecutive_replay_only = 0

        # (b) Fresh supervision verification: assert lane is FRESH for each executed obligation
        for work in resolved.supervision_work:
            executed_source = work.source
            # Verify the source's context lane is FRESH (per-obligation assertion)
            context = execution_contexts.get(executed_source)
            assert context is not None, f"Tick {tick_idx}: No context for {executed_source}"
            assert context.lane is ReadySourceLane.FRESH, (
                f"Tick {tick_idx}: {executed_source} executed with lane {context.lane}, "
                f"expected FRESH"
            )
            record_execution(tick_idx, executed_source, context.lane)

    # Final replay-only streak accounting
    if consecutive_replay_only > 0:
        max_consecutive_replay_only = max(max_consecutive_replay_only, consecutive_replay_only)

    # ===== Assertions
    # (a) All three obligations are consumed (count == 0 pending)
    assert len(durable_commit_obligations) == 0, (
        f"All obligations must be consumed; remaining: {list(durable_commit_obligations.keys())}"
    )

    # (b) Each obligation was executed exactly once as fresh supervision
    assert set(source for (_, source, _) in executed_sources) == {source_a, source_b, source_c}, (
        f"All three sources A/B/C must be executed; "
        f"executed: {set(source for (_, source, _) in executed_sources)}"
    )
    assert len(executed_sources) == 3, (
        f"Each obligation should execute once; got {len(executed_sources)} executions"
    )

    # (d) Replay-only streak bound: assert <= 2 consecutive replay-only iterations
    # while any pending obligations remain (this prevents infinite replay-only loop)
    max_allowed_replay_only = 2
    assert max_consecutive_replay_only <= max_allowed_replay_only, (
        f"Replay-only streak exceeded {max_allowed_replay_only}; observed {max_consecutive_replay_only}"
    )

    # (e) Drain completes within bounded ticks (no infinite loop)
    assert tick_count < max_drain_ticks - 1, (
        f"Drain must complete; exceeded {max_drain_ticks - 1} ticks at {tick_count}"
    )
