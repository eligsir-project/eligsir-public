"""Behavioral tests for the bounded historical replay candidate index."""

from __future__ import annotations

import pytest

from eligsir.mapping.scheduler.historical_candidate_index import (
    HistoricalCandidateConfig,
    HistoricalCandidateIndex,
    HistoricalFrameMetadata,
)


def _index(**overrides: int) -> HistoricalCandidateIndex:
    config_values = dict(
        shortlist_size=64,
        recent_lane_size=16,
        focus_lane_size=16,
        protected_lane_size=8,
        global_reservoir_size=32,
        global_floor=12,
    )
    config_values.update(overrides)
    config = HistoricalCandidateConfig(**config_values)
    return HistoricalCandidateIndex(config=config, seed=17)


def test_frame_zero_remains_reachable_after_more_than_512_registrations():
    """Dropping the first identity from the protected-global anchor is a bug."""
    index = _index()
    for frame_id in range(700):
        index.register(HistoricalFrameMetadata(frame_id=frame_id, region_id=frame_id % 7))

    candidates = index.shortlist(snapshot_id="snapshot-a", epoch=3)

    assert 0 in {candidate.frame_id for candidate in candidates}


def test_focus_upweights_its_region_without_excluding_global_history():
    """Removing the global floor or ignoring focus weights must fail here."""
    index = _index(
        recent_lane_size=64,
        focus_lane_size=32,
        protected_lane_size=1,
        global_reservoir_size=64,
    )
    for frame_id in range(160):
        region = "weak" if frame_id == 32 else "global"
        index.register(HistoricalFrameMetadata(frame_id=frame_id, region_id=region))

    baseline = {
        candidate.frame_id: candidate
        for candidate in index.shortlist(
            snapshot_id="snapshot-b", epoch=5, local_ids=range(64)
        )
    }
    focused = index.shortlist(
        snapshot_id="snapshot-b",
        epoch=5,
        local_ids=range(64),
        focus_region_weights={"weak": 3.0},
    )
    focused_by_id = {candidate.frame_id: candidate for candidate in focused}

    assert focused_by_id[32].weight > baseline[32].weight
    assert any(
        "global_reservoir" in candidate.evidence.lanes
        and candidate.evidence.region_id != "weak"
        for candidate in focused
    )


def test_shortlist_is_bounded_and_deterministic_for_snapshot_and_epoch():
    """A history-wide random scan would violate this shortlist contract."""
    index = _index()
    for frame_id in range(1000):
        index.register(HistoricalFrameMetadata(frame_id=frame_id, region_id=frame_id % 11))

    first = index.shortlist(
        snapshot_id="fixed-snapshot",
        epoch=9,
        local_ids=(990, 991, 992),
        focus_region_weights=((3, 2.0),),
    )
    second = index.shortlist(
        snapshot_id="fixed-snapshot",
        epoch=9,
        local_ids=(990, 991, 992),
        focus_region_weights=((3, 2.0),),
    )

    assert len(first) <= 64
    assert first == second


def test_registration_preserves_identity_metadata_without_eviction():
    """A bounded candidate view must never imply bounded identity history."""
    index = _index()
    for frame_id in range(700):
        index.register(HistoricalFrameMetadata(frame_id=frame_id, region_id=frame_id % 5))

    assert index.history_size == 700
    assert index.metadata_for(0).frame_id == 0
    assert index.metadata_for(699).frame_id == 699


def test_global_floor_rotates_across_the_complete_retained_history():
    """A fixed reservoir must not make most deferred identities unreachable."""

    index = _index()
    for frame_id in range(700):
        index.register(
            HistoricalFrameMetadata(frame_id=frame_id, region_id=frame_id % 7)
        )

    globally_exposed = set()
    for epoch in range(60):
        globally_exposed.update(
            candidate.frame_id
            for candidate in index.shortlist(snapshot_id="rotation", epoch=epoch)
            if "global_rotation" in candidate.evidence.lanes
        )

    assert globally_exposed == set(range(700))


def test_old_identity_focus_enters_shortlist_outside_every_rolling_lane():
    """Dropping an explicit weak cold identity before weighting is a bug."""

    index = _index()
    for frame_id in range(700):
        index.register(
            HistoricalFrameMetadata(frame_id=frame_id, region_id=frame_id % 7)
        )

    baseline = index.shortlist(snapshot_id="identity-focus", epoch=4)
    baseline_ids = {candidate.frame_id for candidate in baseline}
    old_cold_id = next(
        frame_id for frame_id in range(1, 500) if frame_id not in baseline_ids
    )

    focused = index.shortlist(
        snapshot_id="identity-focus",
        epoch=4,
        focus_identity_weights={old_cold_id: 7.0},
    )
    focused_by_id = {candidate.frame_id: candidate for candidate in focused}

    assert old_cold_id in focused_by_id
    assert "identity_focus" in focused_by_id[old_cold_id].evidence.lanes
    assert focused_by_id[old_cold_id].weight == 8.0


def test_identity_focus_preserves_the_complete_configured_global_floor():
    """A full focus lane must not displace rotating global-history identities."""

    index = _index()
    for frame_id in range(700):
        index.register(
            HistoricalFrameMetadata(frame_id=frame_id, region_id=frame_id % 7)
        )
    baseline = index.shortlist(snapshot_id="floor", epoch=11)
    baseline_ids = {candidate.frame_id for candidate in baseline}
    focused_ids = tuple(
        frame_id for frame_id in range(700) if frame_id not in baseline_ids
    )[: index.config.focus_lane_size]

    focused = index.shortlist(
        snapshot_id="floor",
        epoch=11,
        focus_identity_weights={frame_id: 4.0 for frame_id in focused_ids},
    )

    assert len(focused) <= index.config.shortlist_size
    assert sum(
        "global_rotation" in candidate.evidence.lanes for candidate in focused
    ) == index.config.global_floor
    assert set(focused_ids).issubset(
        {candidate.frame_id for candidate in focused}
    )


def test_identity_focus_input_is_bounded_before_shortlist_work():
    """Accepting an unbounded explicit focus input would restore O(history) ticks."""

    index = _index()
    for frame_id in range(100):
        index.register(HistoricalFrameMetadata(frame_id=frame_id, region_id=0))

    with pytest.raises(ValueError, match="focus_identity_weights"):
        index.shortlist(
            snapshot_id="bounded-focus",
            epoch=0,
            focus_identity_weights={
                frame_id: 2.0
                for frame_id in range(index.config.focus_lane_size + 1)
            },
        )


def test_reserved_global_identity_rotates_by_reservation_epoch():
    """A deterministic reservation must eventually address all retained history."""

    index = _index()
    for frame_id in range(700):
        index.register(
            HistoricalFrameMetadata(frame_id=frame_id, region_id=frame_id % 7)
        )

    reserved = []
    for reservation_epoch in range(700):
        shortlist = index.shortlist(
            snapshot_id="reserved-global",
            epoch=0,
            reserved_global_epoch=reservation_epoch,
        )
        exact = tuple(
            candidate.frame_id
            for candidate in shortlist
            if "reserved_global" in candidate.evidence.lanes
        )
        assert len(exact) == 1
        reserved.extend(exact)

    assert set(reserved) == set(range(700))


@pytest.mark.parametrize(
    ("overrides", "message"),
    (
        ({"shortlist_size": 129}, "shortlist_size"),
        (
            {
                "shortlist_size": 64,
                "focus_lane_size": 32,
                "global_reservoir_size": 64,
                "global_floor": 32,
            },
            "global floor, reservation, and focus lane",
        ),
        (
            {
                "shortlist_size": 64,
                "focus_lane_size": 1,
                "global_reservoir_size": 64,
                "global_floor": 63,
            },
            "global floor, reservation, and focus lane",
        ),
    ),
)
def test_config_rejects_unbounded_or_impossible_lane_budgets(overrides, message):
    with pytest.raises(ValueError, match=message):
        _index(**overrides)


def test_config_accepts_exact_global_reservation_focus_budget():
    index = _index(
        shortlist_size=64,
        focus_lane_size=32,
        global_reservoir_size=31,
        global_floor=31,
    )

    assert index.config.global_floor + 1 + index.config.focus_lane_size == 64
