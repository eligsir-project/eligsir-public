from types import SimpleNamespace
from collections import OrderedDict


def _source(ordinal):
    from eligsir.contracts.runtime import SourceIdentity
    return SourceIdentity("camera", 0, ordinal, ordinal * 1_000_000)


def _runtime(active, cached, *, flag_enabled):
    from eligsir.runtime.production_application import CanonicalMappingRuntime

    modalities = OrderedDict((s, object()) for s in (*active, *cached))

    class _FakeCandidateIndex:
        history_size = len(cached)

        def shortlist(self, **kwargs):
            return [
                SimpleNamespace(frame_id=str(s.source_ordinal), evidence=SimpleNamespace(lanes=()))
                for s in cached
            ]

    class _FakeScene:
        train_cameras = [SimpleNamespace(_source_identity=s) for s in active]

    runtime = object.__new__(CanonicalMappingRuntime)
    runtime._replay_modalities = modalities
    runtime._observation_db = None
    runtime._historical_candidate_index = _FakeCandidateIndex()
    runtime._historical_sources_by_frame_id = {str(s.source_ordinal): s for s in cached}
    runtime._work_budget_owner = None
    runtime.optimizer_steps = 0
    runtime.scene = _FakeScene()
    runtime.initialized = True
    runtime._partition_focus_actuation_reaches_active_sources = flag_enabled
    return runtime


def test_shortlisted_replay_view_includes_a_sample_of_active_sources():
    active = [_source(i) for i in range(10)]
    cached = [_source(100 + i) for i in range(5)]
    runtime = _runtime(active, cached, flag_enabled=True)

    shortlisted, _reserved = runtime._shortlisted_replay_view(snapshot_id=0, control_tick_epoch=0)

    included_active = set(shortlisted) & set(active)
    assert included_active, "no active sources were included in the shortlist"
    assert set(shortlisted) & set(cached) == set(cached), "cached shortlist must be preserved"


def test_shortlisted_replay_view_defaults_off_and_is_historical_only_when_disabled():
    active = [_source(i) for i in range(10)]
    cached = [_source(100 + i) for i in range(5)]
    runtime = _runtime(active, cached, flag_enabled=False)

    shortlisted, _reserved = runtime._shortlisted_replay_view(snapshot_id=0, control_tick_epoch=0)

    assert set(shortlisted) == set(cached), (
        "with the flag off, the shortlist must be byte-identical to pre-#1723 "
        "behavior: historical entries only, no active sources unioned in"
    )
    assert not (set(shortlisted) & set(active))


def test_shortlisted_replay_view_active_union_respects_cap_and_is_evenly_sampled():
    from eligsir.utils.partition_control import _deterministic_probe_indices

    active = [_source(i) for i in range(200)]
    cached = [_source(1000 + i) for i in range(5)]
    runtime = _runtime(active, cached, flag_enabled=True)
    pending = active[18]
    runtime._observation_db = SimpleNamespace(
        prepared_commit_obligation_for_source=lambda source: (
            object() if source is pending else None
        )
    )

    shortlisted, _reserved = runtime._shortlisted_replay_view(snapshot_id=0, control_tick_epoch=0)

    MAX_SHORTLIST = 128
    assert len(shortlisted) <= MAX_SHORTLIST

    # Historical entries are always preserved in full.
    assert set(shortlisted) & set(cached) == set(cached)

    eligible_active = tuple(source for source in active if source is not pending)
    active_budget = max(0, min(MAX_SHORTLIST // 2, MAX_SHORTLIST - len(cached)))
    assert active_budget < len(eligible_active), "test must exercise the truncating branch"

    ordered_active = tuple(sorted(eligible_active, key=lambda s: s.source_ordinal))
    expected_positions = _deterministic_probe_indices(
        tuple(range(len(ordered_active))), active_budget
    )
    expected_active = {ordered_active[position] for position in expected_positions}

    included_active = set(shortlisted) & set(active)
    assert included_active == expected_active, (
        "active sources included must match _deterministic_probe_indices' own "
        "even-distribution selection, not an arbitrary prefix/suffix"
    )
    assert pending not in shortlisted
    # Sanity: an even sample across 200 ordinals with a 64-slot budget must
    # span close to the full range, not cluster at one end.
    assert min(s.source_ordinal for s in included_active) < 10
    assert max(s.source_ordinal for s in included_active) > 190


def test_shortlist_uses_real_index_before_pending_obligation_lookup():
    """A pending reserved identity cannot cause a catalog-wide obligation scan."""
    from eligsir.mapping.scheduler.historical_candidate_index import (
        HistoricalCandidateIndex,
        HistoricalFrameMetadata,
    )
    from eligsir.runtime.production_application import CanonicalMappingRuntime

    sources = tuple(_source(index) for index in range(700))
    candidate_index = HistoricalCandidateIndex(seed=17)
    for source in sources:
        candidate_index.register(
            HistoricalFrameMetadata(
                frame_id=source.source_ordinal, region_id=source.source_ordinal % 7
            )
        )
    candidates = candidate_index.shortlist(
        snapshot_id=31, epoch=5, reserved_global_epoch=0
    )
    pending = next(
        source
        for candidate in candidates
        if "reserved_global" in candidate.evidence.lanes
        for source in (sources[candidate.frame_id],)
    )
    calls = []
    runtime = object.__new__(CanonicalMappingRuntime)
    runtime.initialized = True
    runtime.optimizer_steps = 0
    runtime._work_budget_owner = None
    runtime._partition_focus_actuation_reaches_active_sources = False
    runtime._historical_candidate_index = candidate_index
    runtime._historical_sources_by_frame_id = {
        source.source_ordinal: source for source in sources
    }
    runtime._replay_modalities = OrderedDict((source, object()) for source in sources)
    runtime._observation_db = SimpleNamespace(
        prepared_commit_obligation_for_source=lambda source: (
            calls.append(source) or (object() if source is pending else None)
        )
    )

    shortlisted, reserved = runtime._shortlisted_replay_view(
        snapshot_id=31, control_tick_epoch=5
    )

    expected = tuple(
        sources[candidate.frame_id]
        for candidate in candidates
        if sources[candidate.frame_id] is not pending
    )
    assert tuple(shortlisted.items()) == tuple(
        (source, runtime._replay_modalities[source]) for source in expected
    )
    assert reserved is None
    assert pending not in shortlisted
    assert len(calls) == len(candidates)
    assert len(calls) < len(sources)


def test_shortlist_no_index_fallback_filters_exact_pending_identity():
    """The no-index fallback retains the durable first-supervision exclusion."""
    from eligsir.contracts.runtime import SourceIdentity
    from eligsir.runtime.production_application import CanonicalMappingRuntime

    pending = SourceIdentity("camera", 0, 7, 7_000_000, pair_id="pending")
    eligible_other_pair = SourceIdentity(
        "camera", 0, 7, 7_000_000, pair_id="eligible-other-pair"
    )
    calls = []
    runtime = object.__new__(CanonicalMappingRuntime)
    runtime.initialized = True
    runtime._replay_modalities = OrderedDict(
        ((pending, object()), (eligible_other_pair, object()))
    )
    runtime._observation_db = SimpleNamespace(
        prepared_commit_obligation_for_source=lambda source: (
            calls.append(source) or (object() if source is pending else None)
        )
    )
    runtime._historical_candidate_index = None

    shortlisted, reserved = runtime._shortlisted_replay_view(
        snapshot_id=0, control_tick_epoch=0
    )

    assert tuple(shortlisted) == (eligible_other_pair,)
    assert reserved is None
    assert calls == [pending, eligible_other_pair]
