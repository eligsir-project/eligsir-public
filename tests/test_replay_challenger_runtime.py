"""Runtime actuation for opt-in COLD replay trials and READY turnover."""

import json
import math
from collections import OrderedDict
from dataclasses import replace
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from eligsir.contracts import (
    BufferOwnership,
    BufferRef,
    BufferSpec,
    MemoryDomain,
    ModalityEligibility,
    SourceIdentity,
)
from eligsir.contracts.control import (
    ControlLimits,
    ControlSnapshot,
    DispositionKind,
    FreshnessMode,
    ObservationDisposition,
    ResidencyIntent,
)
from eligsir.mapping.scheduler.work_budget import WorkBudgetOwner
from eligsir.mapping.scheduler.historical_candidate_index import (
    HistoricalCandidateIndex,
    HistoricalFrameMetadata,
)
from eligsir.mapping.scheduler.hot_cold_challenger import HotColdChallengerArena
from eligsir.runtime.prepared_source import PreparedSourceLease
from eligsir.runtime.production_application import CanonicalMappingRuntime
from eligsir.runtime.ready_source import ReadySourceOwner


def _runtime_config_args(*options: str):
    from eligsir.cli.parser import parse_training_args

    args, _ = parse_training_args(["-s", "/dev/null", *options])
    args.model_path = "output/replay-challenger-config"
    return args


def test_replay_challenger_options_reach_canonical_runtime_config():
    from eligsir.runtime.config_boundary import mapping_runtime_config_from_args

    defaults = mapping_runtime_config_from_args(_runtime_config_args()).optimization
    assert (
        defaults.streaming_replay_challenger_fraction,
        defaults.streaming_replay_challenger_demotion_cadence,
        defaults.streaming_replay_challenger_demotion_limit,
        defaults.streaming_replay_challenger_lease_ticks,
    ) == (0.0, 0, 0, 100)

    args = _runtime_config_args(
        "--streaming_replay_challenger_fraction",
        "0.75",
        "--streaming_replay_challenger_demotion_cadence",
        "4",
        "--streaming_replay_challenger_demotion_limit",
        "1",
        "--streaming_replay_challenger_lease_ticks",
        "8",
    )
    aggressive = mapping_runtime_config_from_args(args).optimization
    assert (
        aggressive.streaming_replay_challenger_fraction,
        aggressive.streaming_replay_challenger_demotion_cadence,
        aggressive.streaming_replay_challenger_demotion_limit,
        aggressive.streaming_replay_challenger_lease_ticks,
    ) == (0.75, 4, 1, 8)


@pytest.mark.parametrize(
    ("name", "value"),
    (
        ("streaming_replay_challenger_fraction", math.nan),
        ("streaming_replay_challenger_fraction", math.inf),
        ("streaming_replay_challenger_fraction", -math.inf),
        ("streaming_replay_challenger_fraction", -0.01),
        ("streaming_replay_challenger_fraction", 1.01),
        ("streaming_replay_challenger_demotion_cadence", True),
        ("streaming_replay_challenger_demotion_cadence", 1.5),
        ("streaming_replay_challenger_demotion_cadence", -1),
        ("streaming_replay_challenger_demotion_limit", True),
        ("streaming_replay_challenger_demotion_limit", 1.5),
        ("streaming_replay_challenger_demotion_limit", -1),
        ("streaming_replay_challenger_lease_ticks", True),
        ("streaming_replay_challenger_lease_ticks", 1.5),
        ("streaming_replay_challenger_lease_ticks", -1),
    ),
)
def test_replay_challenger_options_reject_invalid_constructor_values(name, value):
    from eligsir.runtime.config_boundary import mapping_runtime_config_from_args

    args = _runtime_config_args()
    setattr(args, name, value)
    with pytest.raises((TypeError, ValueError), match="replay_challenger"):
        mapping_runtime_config_from_args(args)


def _source(ordinal: int) -> SourceIdentity:
    return SourceIdentity("train", 0, ordinal, ordinal)


def _lease(ordinal: int, generation: int = 0) -> PreparedSourceLease:
    payload = BufferRef(
        data=torch.zeros(2),
        spec=BufferSpec((2,), "float32", MemoryDomain.OTHER_DEVICE, "tensor", device="cpu"),
        ownership=BufferOwnership.BORROWED,
        owner_token=object(),
    )
    return PreparedSourceLease(
        source_identity=_source(ordinal),
        modality_eligibility=ModalityEligibility(rgb_supervision=True),
        source_metadata={"replay_quality": (1.0, None)},
        tier_preparation_available=("native",),
        borrowed_device_payloads={"rgb": payload},
        device="cpu",
        generation=generation,
    )


def _runtime(*, fraction: float = 0.75, cadence: int = 4, limit: int = 1):
    runtime = object.__new__(CanonicalMappingRuntime)
    runtime._optimization_policy = SimpleNamespace(
        streaming_historical_usefulness_tracking=True,
        streaming_replay_challenger_fraction=fraction,
        streaming_replay_challenger_demotion_cadence=cadence,
        streaming_replay_challenger_demotion_limit=limit,
    )
    runtime.optimizer_steps = cadence
    runtime.iteration = cadence
    runtime._prepared_ready_frozen_work = None
    runtime._prepared_ready_candidate_work = None
    runtime._latest_tick_work = (None, SimpleNamespace(replay_trial_sources=(_source(3),)))
    runtime._post_refinement_phase_owner = None
    runtime._observation_db = SimpleNamespace(
        get_frame=lambda _frame_id: SimpleNamespace(
            semantic_action="commit", prepared_commit_context=object(),
        ),
        prepared_commit_obligation_for_source=lambda _source: None,
    )
    runtime._historical_candidate_index = HistoricalCandidateIndex()
    runtime._historical_frame_ids_by_source = {_source(i): i for i in range(8)}
    for i in range(8):
        runtime._historical_candidate_index.register(HistoricalFrameMetadata(i, 0))
    arena = runtime._hot_cold_challenger_arena = HotColdChallengerArena(lease_ticks=0)
    for frame_id, weight in ((0, 0.1), (1, 0.2), (2, 0.3), (3, 2.0)):
        for _ in range(3):
            arena.record_modality_observation(frame_id, "rgb", weight)
    owner = runtime._ready_source_owner = ReadySourceOwner(max_fresh=1, max_replay=8, replay_reserve=1)
    producer = owner.claim_producer("decode")
    assert producer.offer(_lease(0), lane="replay", arrival_ordinal=0)
    assert producer.offer(_lease(1), lane="replay", arrival_ordinal=1)
    assert producer.offer(_lease(3), lane="replay", arrival_ordinal=3)
    return runtime, owner, producer


def _saved_challenger_trace(tmp_path, runtime):
    from eligsir.runtime.causal_replay import JsonlDecisionTraceObserver

    path = tmp_path / "decision-trace.jsonl"
    observer = JsonlDecisionTraceObserver(path)
    observer(
        SimpleNamespace(
            replay_challenger_trace=tuple(runtime._replay_challenger_trace_events),
            replay_challenger_trace_truncated=bool(
                getattr(runtime, "_replay_challenger_trace_truncated", False)
            ),
        )
    )
    observer.flush()
    return json.loads(path.read_text())


def test_aggressive_turnover_evicts_a_real_idle_incumbent_but_keeps_global_reserve():
    runtime, owner, producer = _runtime()
    before = owner.ready_bytes_current

    assert runtime._actuate_replay_challenger_turnover((_source(3),)) == (_source(0),)
    assert owner.ready_bytes_current < before
    assert owner.ready_replay_sources() == (_source(1), _source(3))
    assert not producer.offer(_lease(0), lane="replay", arrival_ordinal=0)
    assert producer.offer(_lease(0, generation=1), lane="replay", arrival_ordinal=0)


@pytest.mark.parametrize(
    ("batch_size", "batch_shape"), ((1, "B1"), (4, "B=C"), (16, "B>C"))
)
def test_challenger_trace_is_opt_in_bounded_and_marks_truncation(batch_size, batch_shape):
    runtime, _owner, _producer = _runtime()
    runtime._population_selection_trace_enabled = True
    runtime._batch_size = batch_size
    for index in range(batch_size * 4 + 1):
        runtime._record_replay_challenger_trace({"index": index})

    assert [event["index"] for event in runtime._replay_challenger_trace_events] == list(
        range(1, batch_size * 4 + 1)
    )
    assert runtime._replay_challenger_trace_truncated is True
    assert batch_shape in {"B1", "B=C", "B>C"}


def test_challenger_trace_keeps_reordered_and_repeated_identity_events():
    runtime, _owner, _producer = _runtime()
    runtime._population_selection_trace_enabled = True
    first = _source(3).fingerprint()
    second = _source(4).fingerprint()
    runtime._record_replay_challenger_trace({"source": second, "outcome": "retain"})
    runtime._record_replay_challenger_trace({"source": first, "outcome": "promote"})
    runtime._record_replay_challenger_trace({"source": first, "outcome": "demote"})

    assert [event["source"] for event in runtime._replay_challenger_trace_events] == [
        second, first, first
    ]


def test_challenger_trace_records_fallback_without_claiming_turnover():
    runtime, _owner, _producer = _runtime(fraction=0.0)
    runtime._population_selection_trace_enabled = True

    assert runtime._actuate_replay_challenger_turnover((_source(3),)) == ()
    assert runtime._replay_challenger_trace_events == [
        {
            "stage": "turnover",
            "outcome": "not_attempted",
            "reason": "feature_disabled",
            "sources": [_source(3).fingerprint()],
        }
    ]


def test_challenger_trace_records_arena_and_ready_turnover_outcomes():
    runtime, _owner, _producer = _runtime()
    runtime._population_selection_trace_enabled = True
    runtime._versions = lambda: (4, 5, 6)

    assert runtime._actuate_replay_challenger_turnover((_source(3),)) == (_source(0),)

    arena_events = [
        event for event in runtime._replay_challenger_trace_events
        if event.get("stage") == "arena"
    ]
    assert len(arena_events) == 1
    assert arena_events[0]["source"] == _source(3).fingerprint()
    assert arena_events[0]["incumbent"] == _source(0).fingerprint()
    assert arena_events[0]["arena_outcome"] == "promote"
    assert arena_events[0]["residency_outcome"] == "demote"
    assert arena_events[0]["versions"] == {
        "map_content": 4,
        "topology": 5,
        "parameter": 6,
    }


def test_completed_trial_trace_keeps_measured_scores_and_ready_generation():
    import numpy as np
    from eligsir.optimization.gaussian.supervision_evidence import (
        EvidenceSnapshotToken, SupervisionEvidenceBatch, SupervisionModality,
    )
    from eligsir.runtime import production_application
    from eligsir.runtime.transfers import DeviceTransferOwner

    runtime, _owner, _producer = _runtime()
    runtime._population_selection_trace_enabled = True
    runtime._versions = lambda: (4, 5, 6)
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(production_application, "torch", torch)
    runtime._transfer_owner = DeviceTransferOwner("cpu")
    for _ in range(2):
        runtime._transfer_owner.upload_packed((np.zeros(64, dtype=np.uint8),), purpose="test_ingress")
    packet = SupervisionEvidenceBatch(
        snapshot_token=EvidenceSnapshotToken(0, 0, 0, 0, 4, 0, 0, 0),
        transaction_id=4,
        source_identities=(_source(3),),
        modalities=(SupervisionModality(),),
        rgb_residual_summary=torch.tensor([2.0]),
        depth_residual_summary=torch.zeros(1),
        rgb_valid_pixels=torch.tensor([12]),
        depth_valid_pixels=torch.zeros(1, dtype=torch.int64),
    )
    transaction = SimpleNamespace(
        plan=SimpleNamespace(replay_trial_sources=(_source(3),), resolution_tier="native"),
        bindings=(SimpleNamespace(source_identity=_source(3), lease=_lease(3, generation=2)),),
    )
    runtime._observe_replay_evidence(
        SimpleNamespace(evidence=packet, optimizer_step_count=1, supervision_view_count=1),
        transaction,
    )
    runtime._poll_replay_evidence()
    monkeypatch.undo()

    completion = [
        event for event in runtime._replay_challenger_trace_events
        if event.get("stage") == "completion"
    ]
    assert len(completion) == 1
    assert completion[0]["completion"] == "completed"
    assert completion[0]["scores"]["rgb"]["q"] == 1.0
    assert completion[0]["scores"]["rgb"]["residual"] == 2.0
    assert completion[0]["scores"]["rgb"]["valid_pixels"] == 12
    assert completion[0]["lease_generation"] == 2
    assert completion[0]["snapshot"] is not None


def test_partial_trial_trace_is_explicitly_unavailable_and_cannot_promote(tmp_path):
    import numpy as np
    from eligsir.optimization.gaussian.supervision_evidence import (
        EvidenceSnapshotToken, SupervisionEvidenceBatch, SupervisionModality,
    )
    from eligsir.runtime import production_application
    from eligsir.runtime.transfers import DeviceTransferOwner

    runtime, _owner, _producer = _runtime()
    runtime._population_selection_trace_enabled = True
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(production_application, "torch", torch)
    runtime._transfer_owner = DeviceTransferOwner("cpu")
    for _ in range(2):
        runtime._transfer_owner.upload_packed((np.zeros(64, dtype=np.uint8),), purpose="test_ingress")
    packet = SupervisionEvidenceBatch(
        snapshot_token=EvidenceSnapshotToken(0, 0, 0, 0, 4, 0, 0, 0),
        transaction_id=4,
        source_identities=(_source(3),),
        modalities=(SupervisionModality(),),
        rgb_residual_summary=torch.tensor([2.0]),
        depth_residual_summary=torch.zeros(1),
        rgb_valid_pixels=torch.tensor([12]),
        depth_valid_pixels=torch.zeros(1, dtype=torch.int64),
    )
    runtime._observe_replay_evidence(
        SimpleNamespace(evidence=packet, optimizer_step_count=1, supervision_view_count=0),
        SimpleNamespace(
            plan=SimpleNamespace(
                replay_trial_sources=(_source(3),), resolution_tier="native"
            ),
            bindings=(
                SimpleNamespace(source_identity=_source(3), lease=_lease(3, generation=2)),
            ),
        ),
    )
    monkeypatch.undo()

    assert any(
        event.get("completion") == "partial"
        and event.get("reason") == "completion_unattributed"
        for event in runtime._replay_challenger_trace_events
    )
    assert not any(
        event.get("stage") == "arena"
        for event in runtime._replay_challenger_trace_events
    )
    saved = _saved_challenger_trace(tmp_path, runtime)
    event = saved["replay_challenger_trace"][0]
    assert event["source"] == _source(3).fingerprint()
    assert event["completion"] == "partial"
    assert event["resolution_tier"] == "native"
    assert event["lease_generation"] == 2
    assert event["snapshot"] is not None
    assert saved["replay_challenger_trace_complete"] is True


def test_not_attempted_fallback_reaches_saved_jsonl_with_identity_and_tier(tmp_path):
    runtime, _owner, _producer = _runtime()
    runtime._population_selection_trace_enabled = True
    source = _source(3)
    runtime._observe_replay_evidence(
        SimpleNamespace(optimizer_step_count=0),
        SimpleNamespace(
            plan=SimpleNamespace(
                replay_trial_sources=(source,), resolution_tier="native"
            ),
            bindings=(
                SimpleNamespace(source_identity=source, lease=_lease(3, generation=4)),
            ),
        ),
    )

    saved = _saved_challenger_trace(tmp_path, runtime)
    event = saved["replay_challenger_trace"][0]
    assert event == {
        "stage": "completion",
        "source": source.fingerprint(),
        "source_ordinal": source.source_ordinal,
        "timestamp_ns": source.timestamp_ns,
        "completion": "not_attempted",
        "reason": "no_optimizer_step",
        "lease_generation": 4,
        "resolution_tier": "native",
        "snapshot": None,
    }
    assert saved["replay_challenger_trace_complete"] is True


def test_turnover_is_off_cadence_bounded_and_does_not_promote_untrusted_challengers():
    runtime, owner, _producer = _runtime(fraction=0.0)
    assert runtime._actuate_replay_challenger_turnover((_source(3),)) == ()
    assert len(owner.ready_replay_sources()) == 3

    runtime, owner, _producer = _runtime(cadence=4)
    runtime.optimizer_steps = 3
    assert runtime._actuate_replay_challenger_turnover((_source(3),)) == ()
    assert len(owner.ready_replay_sources()) == 3
    assert runtime._actuate_replay_challenger_turnover((_source(99),)) == ()


@pytest.mark.parametrize("intervene", (False, True))
def test_completed_trial_feedback_drives_one_real_demotion_without_active_plan_rebinding(
    monkeypatch, intervene,
):
    import numpy as np
    from eligsir.optimization.gaussian.supervision_evidence import (
        EvidenceSnapshotToken, SupervisionEvidenceBatch, SupervisionModality,
    )
    from eligsir.runtime import production_application
    from eligsir.runtime.transfers import DeviceTransferOwner

    runtime, owner, _ = _runtime()
    monkeypatch.setattr(production_application, "torch", torch)
    runtime._transfer_owner = DeviceTransferOwner("cpu")
    for _ in range(2):
        runtime._transfer_owner.upload_packed((np.zeros(64, dtype=np.uint8),), purpose="test_ingress")
    if not intervene:
        monkeypatch.setattr(runtime, "_actuate_replay_challenger_turnover", lambda _: ())
    packet = SupervisionEvidenceBatch(
        snapshot_token=EvidenceSnapshotToken(0, 0, 0, 0, 4, 0, 0, 0),
        transaction_id=4, source_identities=(_source(3),),
        modalities=(SupervisionModality(),),
        rgb_residual_summary=torch.tensor([2.0]), depth_residual_summary=torch.zeros(1),
        rgb_valid_pixels=torch.tensor([12]), depth_valid_pixels=torch.zeros(1, dtype=torch.int64),
    )
    transaction = SimpleNamespace(
        plan=SimpleNamespace(replay_trial_sources=(_source(3),)),
        bindings=(SimpleNamespace(source_identity=_source(3), lease=_lease(3)),),
    )
    runtime._pending_replay_trial_sources = (_source(3),)
    runtime._observe_replay_evidence(
        SimpleNamespace(evidence=packet, optimizer_step_count=1, supervision_view_count=1),
        transaction,
    )
    # The scheduler has moved on when DMA completes. Trial authority belongs to
    # the completed packet, not this different next plan.
    runtime._latest_tick_work = (None, SimpleNamespace(replay_trial_sources=()))
    before = owner.ready_bytes_current
    runtime._poll_replay_evidence()
    runtime._poll_replay_evidence()
    assert owner.ready_bytes_current == before - (8 if intervene else 0)
    assert _source(3) in owner.replay_resident_sources()
    telemetry = runtime.historical_usefulness_telemetry()
    assert telemetry["replay_challenger_trials_completed"] == 1
    assert telemetry["replay_challenger_demotions"] == int(intervene)
    assert runtime._pending_replay_trial_sources == ()


def test_pending_trial_demand_prunes_retired_ineligible_unreloadable_and_first_service_sources():
    """Only a durable, RGB-authorized source may survive a fallback boundary."""

    viable, retired, ineligible, unreloadable, first_service = tuple(
        _source(index) for index in range(20, 25)
    )
    records = {
        source: SimpleNamespace(
            semantic_action="commit", prepared_commit_context=object()
        )
        for source in (viable, retired, ineligible, first_service)
    }
    runtime = object.__new__(CanonicalMappingRuntime)
    runtime._pending_replay_trial_sources = (
        viable,
        retired,
        ineligible,
        unreloadable,
        first_service,
    )
    runtime._replay_modalities = {
        source: ModalityEligibility(rgb_supervision=source is not ineligible)
        for source in runtime._pending_replay_trial_sources
    }
    runtime._retired_replay_sources = {retired: True}
    runtime._historical_frame_ids_by_source = {
        source: index for index, source in enumerate(runtime._pending_replay_trial_sources)
    }
    runtime._observation_db = SimpleNamespace(
        get_frame=lambda frame_id: records.get(runtime._pending_replay_trial_sources[frame_id]),
        prepared_commit_obligation_for_source=lambda source: (
            object() if source is first_service else None
        ),
    )

    assert runtime._prune_pending_replay_trial_sources() == (viable,)


def test_cold_trial_survives_fallback_map_update_ready_publish_and_requeue():
    """A COLD trial remains an attributed current-version trial after READY fallback work."""

    from eligsir.optimization.gaussian.supervision_evidence import (
        EvidenceSnapshotToken,
        SupervisionEvidenceBatch,
        SupervisionModality,
    )

    fallback_sources = tuple(_source(index) for index in range(30, 34))
    trial = _source(34)
    sources = (*fallback_sources, trial)
    modality = ModalityEligibility(rgb_supervision=True)
    snapshot = ControlSnapshot(
        snapshot_id=1,
        control_version=1,
        source=fallback_sources[0],
        tracking=None,
        pending_sources=(fallback_sources[0],),
        newest_frontier=fallback_sources[0],
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
        optimizer_steps_completed=1,
        limits=ControlLimits(4, 4, 1, 0, 0, 0),
        supervision_candidates=(fallback_sources[0],),
        replay_candidates=sources,
        replay_work_modalities=(modality,) * len(sources),
    )
    disposition = ObservationDisposition(
        source=fallback_sources[0],
        snapshot_id=1,
        kind=DispositionKind.ADMIT,
        observe=True,
        mapping_candidate=True,
        admit_supervision=True,
        allow_geometry_insertion=False,
        residency=ResidencyIntent.HOT,
        reason="test",
        modality_eligibility=modality,
    )
    budget = WorkBudgetOwner(
        requested_batch_size=4, maximum_camera_count=4, constellation_policy="fixed"
    )
    active = budget.resolve(snapshot, disposition=disposition, proposals=())
    runtime = object.__new__(CanonicalMappingRuntime)
    runtime._batch_size = 4
    runtime._work_budget_owner = budget
    runtime._optimization_policy = SimpleNamespace(
        streaming_replay_challenger_fraction=0.75,
        streaming_historical_usefulness_tracking=True,
    )
    runtime._post_refinement_phase_owner = None
    runtime._replay_depth_due = lambda: False
    runtime._resolution_policy_provider = None
    runtime._mutation_policy_provider = None
    runtime._adaptive_growth_provider = None
    runtime._replay_modalities = OrderedDict((source, modality) for source in sources)
    runtime._replay_weights = {source: 1.0 for source in sources}
    runtime._historical_frame_ids_by_source = {source: index for index, source in enumerate(sources)}
    runtime._retired_replay_sources = OrderedDict()
    runtime._observation_db = SimpleNamespace(
        get_frame=lambda _frame_id: SimpleNamespace(
            semantic_action="commit", prepared_commit_context=object()
        ),
        prepared_commit_obligation_for_source=lambda _source: None,
    )
    runtime._shortlisted_replay_view = lambda **_kwargs: (
        OrderedDict((source, modality) for source in sources), None
    )
    versions = [1, 1, 1]
    runtime._versions = lambda: tuple(versions)
    runtime.optimizer_steps = 1
    runtime._fresh_idle_snapshot = lambda prior: replace(
        prior,
        snapshot_id=prior.snapshot_id + 1,
        control_version=prior.control_version + 1,
        map_content_version=versions[0],
        topology_version=versions[1],
        parameter_version=versions[2],
    )
    owner = runtime._ready_source_owner = ReadySourceOwner(max_fresh=1, max_replay=8)
    producer = owner.claim_producer("background")
    for source in fallback_sources:
        assert producer.offer(_lease(source.source_ordinal), lane="replay")
    runtime._prepared_ready_frozen_work = None
    runtime._prepared_ready_candidate_work = (snapshot, runtime._idle_replay_plan(snapshot, active))

    fallback = runtime._prepared_ready_plan_for_execution()
    assert fallback is not None and trial not in fallback.replay_sources
    assert runtime._pending_replay_trial_sources == (trial,)

    # Background COLD promotion publishes the same identity; no compute path
    # loads or waits for it.
    assert producer.offer(_lease(trial.source_ordinal), lane="replay")
    versions[:] = (2, 3, 4)
    selected = owner.select(4, replay_sources=frozenset(fallback.replay_sources))
    owner.requeue_batch(selected, ready_event=None)
    runtime._advance_prepared_ready_plan(fallback)

    current = runtime._prepared_ready_candidate_work[1]
    assert (current.map_content_version, current.topology_version, current.parameter_version) == (2, 3, 4)
    assert current.replay_trial_sources == (trial,)
    current_selection = owner.select(4, replay_sources=frozenset(current.replay_sources))
    trial_lease = next(lease for lease in current_selection if lease.source_identity == trial)
    runtime._poll_replay_evidence = lambda: None
    runtime._pending_replay_evidence = object()
    runtime._observe_replay_evidence(
        SimpleNamespace(
            optimizer_step_count=1,
            supervision_view_count=1,
            evidence=SupervisionEvidenceBatch(
                snapshot_token=EvidenceSnapshotToken(0, 0, 0, 0, 1, 0, 0, 0),
                transaction_id=1,
                source_identities=(trial,),
                modalities=(SupervisionModality(),),
                rgb_residual_summary=torch.tensor([1.0]),
                depth_residual_summary=torch.zeros(1),
                rgb_valid_pixels=torch.tensor([1]),
                depth_valid_pixels=torch.zeros(1, dtype=torch.int64),
            ),
        ),
        SimpleNamespace(plan=current, bindings=(SimpleNamespace(source_identity=trial, lease=trial_lease),)),
    )
    owner.requeue_batch(current_selection, ready_event=None)
    assert runtime._pending_replay_trial_sources == ()
    assert runtime._prepared_ready_plan_for_execution() is current
    runtime._advance_prepared_ready_plan(current)
    assert runtime._prepared_ready_candidate_work[1].replay_trial_sources == ()


def test_skipped_trial_execution_keeps_pending_demand():
    """A trial source is consumed by attested service, never by a skipped optimizer tick."""

    runtime, _owner, _producer = _runtime()
    runtime._pending_replay_trial_sources = (_source(3),)

    runtime._observe_replay_evidence(
        SimpleNamespace(optimizer_step_count=0),
        SimpleNamespace(plan=SimpleNamespace(replay_trial_sources=(_source(3),))),
    )

    assert runtime._pending_replay_trial_sources == (_source(3),)


@pytest.mark.parametrize("guard", ("fresh", "unreloadable", "closed", "refinement", "absent_winner", "other_modality"))
def test_turnover_protections_prevent_irrecoverable_or_unqualified_replacement(guard):
    from eligsir.runtime.post_refinement import PostRefinementPhase

    runtime, owner, _ = _runtime()
    if guard == "fresh":
        runtime._observation_db.prepared_commit_obligation_for_source = lambda _: object()
    elif guard == "unreloadable":
        runtime._observation_db.get_frame = lambda _: None
    elif guard == "closed":
        runtime._closed = True
    elif guard == "refinement":
        runtime._post_refinement_phase_owner = SimpleNamespace(phase=PostRefinementPhase.REFINEMENT)
    elif guard == "absent_winner":
        owner.evict_replay(target_source=_source(3))
    else:
        for frame_id in (0, 1):
            for _ in range(3):
                runtime._hot_cold_challenger_arena.record_modality_observation(frame_id, "depth", 2.0)
    before = owner.ready_bytes_current
    assert runtime._actuate_replay_challenger_turnover((_source(3),)) == ()
    assert owner.ready_bytes_current == before


def test_turnover_protects_both_frozen_and_candidate_plans():
    runtime, owner, _ = _runtime()
    runtime._prepared_ready_frozen_work = (None, SimpleNamespace(supervision_sources=(_source(0),), replay_sources=()))
    runtime._prepared_ready_candidate_work = (None, SimpleNamespace(supervision_sources=(), replay_sources=(_source(1),)))
    assert runtime._actuate_replay_challenger_turnover((_source(3),)) == ()
    assert len(owner.ready_replay_sources()) == 3


def test_multiple_winners_obey_total_demotion_limit_and_keep_the_winners():
    runtime, owner, producer = _runtime(limit=1)
    for frame_id in (4, 5):
        assert producer.offer(_lease(frame_id), lane="replay", arrival_ordinal=frame_id)
        for _ in range(3):
            runtime._hot_cold_challenger_arena.record_modality_observation(frame_id, "rgb", 3.0)
    assert runtime._actuate_replay_challenger_turnover((_source(3), _source(4), _source(5))) == (_source(0),)
    assert {_source(3), _source(4), _source(5)} <= set(owner.replay_resident_sources())
