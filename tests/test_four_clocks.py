"""Unit tests for ``FourClocksAggregator`` (Wave 2 Task 8 telemetry,
F1/F2/F3 fixes, 2026-08-06). Pure CPU/logic tests -- no GPU, no torch, no
real Gaussian model construction. All timing is driven via the aggregator's
own explicit ``wall_ts``/``now`` parameters, never real ``time.sleep()``, so
these are deterministic and fast.
"""

from __future__ import annotations

import copy

import pytest

import eligsir.mapping.coordinator as coordinator_module
from eligsir.mapping.coordinator import FourClocksAggregator, MappingCoordinator


# ---------------------------------------------------------------------------
# F1 -- rates are wall-time windows, not event-count windows.
# ---------------------------------------------------------------------------

def test_stalled_source_rate_decays_to_zero_not_frozen():
    agg = FourClocksAggregator(rate_window_s=5.0)
    # Five source-frame events, 0.1s apart, ending at t=0.4 -- a healthy
    # ~10Hz burst.
    for i in range(5):
        agg.record_source_frame(wall_ts=float(i) * 0.1)

    healthy = agg.summary(now=0.4)
    assert healthy.source_frame_rate_hz > 0.0

    # Now the source stalls for far longer than the rate window (no new
    # events at all). The OLD event-count-window implementation would keep
    # reporting the same frozen nonzero rate indefinitely here, since it
    # only looked at "the last N events" with no wall-time cutoff.
    stalled = agg.summary(now=0.4 + 600.0)  # 10 minutes later
    assert stalled.source_frame_rate_hz == 0.0


def test_rate_recovers_once_stall_ends():
    agg = FourClocksAggregator(rate_window_s=5.0)
    agg.record_source_frame(wall_ts=0.0)
    agg.record_source_frame(wall_ts=0.1)
    assert agg.summary(now=600.0).source_frame_rate_hz == 0.0

    # New events arrive after the stall -- rate should reflect them, not
    # remain stuck at zero or blend in the ancient pre-stall events.
    agg.record_source_frame(wall_ts=600.5)
    agg.record_source_frame(wall_ts=600.6)
    resumed = agg.summary(now=600.6)
    assert resumed.source_frame_rate_hz > 0.0


# ---------------------------------------------------------------------------
# F2 -- backlog fields are split into genuinely distinct, non-conflated
# quantities.
# ---------------------------------------------------------------------------

def test_backlog_fields_are_distinct_from_conflated_legacy_field():
    agg = FourClocksAggregator()
    for _ in range(10):
        agg.record_source_frame()
    for _ in range(6):
        agg.record_admitted_keyframe()
    for _ in range(2):
        agg.record_frame_holdout()
    for _ in range(1):
        agg.record_frame_rejected()
    agg.record_source_queue_depth(3)

    s = agg.summary()
    # Legacy conflated field: 10 - 6 = 4 (holdouts + rejections + anything
    # else not yet admitted, all mixed together).
    assert s.input_backlog_frames == 4
    # New fields report the real, distinct components.
    assert s.holdout_frames_total == 2
    assert s.frames_rejected_total == 1
    # 10 source - 6 admitted - 2 holdout - 1 rejected = 1 genuinely
    # unaccounted-for frame.
    assert s.frames_unprocessed == 1
    assert s.source_queue_depth == 3
    # The new fields must not just mirror the legacy one.
    assert s.holdout_frames_total != s.input_backlog_frames
    assert s.frames_unprocessed != s.input_backlog_frames


def test_source_queue_depth_none_when_source_has_no_buffered_count():
    agg = FourClocksAggregator()
    agg.record_source_frame()
    # Never called record_source_queue_depth -- source doesn't expose one.
    assert agg.summary().source_queue_depth is None


# ---------------------------------------------------------------------------
# F3 -- map-freshness clocks also fire on non-insertion mutations, not just
# insertion.
# ---------------------------------------------------------------------------

def test_insertion_only_clock_ignores_non_insertion_mutation():
    agg = FourClocksAggregator()
    agg.record_map_commit(wall_ts=0.0, n_inserted=5)

    s = agg.summary(now=10.0)
    assert s.last_observation_integration_age_ms == 10.0 * 1000.0
    assert s.map_age_ms == s.last_observation_integration_age_ms

    # A non-insertion mutation (map_version advances, e.g. a prune or a
    # loop-closure correction) must NOT move the insertion-only clock.
    agg.record_map_version(map_version=1, topology_version=0, wall_ts=0.0)  # baseline
    agg.record_map_version(map_version=2, topology_version=0, wall_ts=9.0)  # pure param update, no topology change
    s2 = agg.summary(now=10.0)
    assert s2.last_observation_integration_age_ms == 10.0 * 1000.0  # unchanged


def test_topology_and_parameter_clocks_update_on_non_insertion_events():
    agg = FourClocksAggregator()
    # Baseline call establishes starting versions without recording an age.
    agg.record_map_version(map_version=0, topology_version=0, wall_ts=0.0)
    assert agg.summary(now=0.0).last_topology_change_age_ms is None
    assert agg.summary(now=0.0).last_parameter_update_age_ms is None

    # A same-shape parameter update (e.g. an optimizer/loop-closure pose
    # correction): map_version bumps, topology_version does not.
    agg.record_map_version(map_version=1, topology_version=0, wall_ts=3.0)
    s = agg.summary(now=5.0)
    assert s.last_parameter_update_age_ms == 2000.0
    assert s.last_topology_change_age_ms is None  # topology untouched yet

    # A structural change (e.g. a prune or relocation): both bump.
    agg.record_map_version(map_version=2, topology_version=1, wall_ts=4.0)
    s2 = agg.summary(now=5.0)
    assert s2.last_topology_change_age_ms == 1000.0
    assert s2.last_parameter_update_age_ms == 1000.0


def test_map_version_first_call_establishes_baseline_without_false_age():
    """A restored checkpoint can start with nonzero map_version/
    topology_version; the very first record_map_version() call must not be
    treated as an instantaneous change at process start."""
    agg = FourClocksAggregator()
    agg.record_map_version(map_version=42, topology_version=7, wall_ts=100.0)
    s = agg.summary(now=100.0)
    assert s.last_topology_change_age_ms is None
    assert s.last_parameter_update_age_ms is None


def test_four_clocks_resume_round_trip_preserves_counters_rates_and_ages(monkeypatch):
    source = FourClocksAggregator(rate_window_s=5.0)
    source.record_source_frame(wall_ts=90.0, frame_timestamp=10.0)
    source.record_source_frame(wall_ts=91.0, frame_timestamp=12.0)
    source.record_admitted_keyframe(wall_ts=90.5)
    source.record_admitted_keyframe(wall_ts=91.5)
    source.record_frame_rejected()
    source.record_frame_holdout()
    source.record_source_queue_depth(3)
    source.record_map_commit(wall_ts=91.0, n_inserted=4)
    source.record_map_version(7, 2, wall_ts=90.0)
    source.record_map_version(8, 3, wall_ts=91.25)
    monkeypatch.setattr(coordinator_module.time, "perf_counter", lambda: 92.0)

    payload = source.resume_state_dict()
    monkeypatch.setattr(coordinator_module.time, "perf_counter", lambda: 100.0)
    restored = FourClocksAggregator()
    restored.load_resume_state_dict(payload)
    summary = restored.summary(now=100.0)

    assert summary.source_frame_rate_hz == 1.0
    assert summary.admitted_keyframe_rate_hz == 1.0
    assert summary.map_age_ms == 1000.0
    assert summary.last_parameter_update_age_ms == 750.0
    assert summary.last_topology_change_age_ms == 750.0
    assert summary.frames_rejected_total == 1
    assert summary.holdout_frames_total == 1
    assert summary.input_backlog_frames == 0
    assert summary.source_queue_depth == 3
    assert summary.realtime_factor == 2.0


def test_four_clocks_resume_rejects_malformed_state_atomically(monkeypatch):
    target = FourClocksAggregator()
    target.record_source_frame(wall_ts=1.0, frame_timestamp=1.0)
    monkeypatch.setattr(coordinator_module.time, "perf_counter", lambda: 2.0)
    before = copy.deepcopy(target.resume_state_dict())
    broken = copy.deepcopy(before)
    broken["n_source_total"] = -1

    with pytest.raises(ValueError, match="n_source_total"):
        target.load_resume_state_dict(broken)

    assert target.resume_state_dict() == before


def test_coordinator_poll_map_activity_updates_clocks_without_step():
    """MappingCoordinator.poll_map_activity() must pick up version changes
    on the backend even when they didn't happen via MappingCoordinator.step()
    -- e.g. a prune/relocation/loop-closure commit made directly through
    begin_mutation()/commit_mutation() elsewhere in the training loop."""

    class _FakeLegacyModel:
        pass

    coordinator = MappingCoordinator()
    model = _FakeLegacyModel()
    backend = coordinator.backend_for(model)

    coordinator.poll_map_activity()  # baseline at version 0, real wall clock
    assert coordinator.four_clocks.summary().last_parameter_update_age_ms is None

    # Simulate a committed mutation elsewhere bumping the backend's version
    # counters directly (what begin_mutation()/commit_mutation() does).
    backend._map_version = 1
    backend._topology_version = 1
    coordinator.poll_map_activity()

    s = coordinator.four_clocks.summary()
    assert s.last_parameter_update_age_ms is not None
    assert s.last_parameter_update_age_ms < 1000.0  # just happened
    assert s.last_topology_change_age_ms is not None
    assert s.last_topology_change_age_ms < 1000.0
