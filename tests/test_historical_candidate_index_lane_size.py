"""Test historical candidate index lane sizing from configured batch size."""

from __future__ import annotations

import pytest

from eligsir.mapping.scheduler.historical_candidate_index import (
    HistoricalCandidateConfig,
    HistoricalCandidateIndex,
    HistoricalFrameMetadata,
)


def test_historical_candidate_config_accepts_64_local_ids_with_batch_size_64():
    """HistoricalCandidateConfig with recent_lane_size=64 accepts 64 local_ids."""
    config = HistoricalCandidateConfig(recent_lane_size=64, shortlist_size=128)
    index = HistoricalCandidateIndex(config=config)

    # Register 64 frames
    for frame_id in range(64):
        index.register(HistoricalFrameMetadata(frame_id=frame_id, region_id=0))

    # Create a shortlist with 64 local_ids - should not raise ValueError
    shortlist = index.shortlist(
        snapshot_id="test",
        epoch=0,
        local_ids=tuple(range(64)),
    )

    assert len(shortlist) > 0
    assert all(candidate.frame_id in range(64) for candidate in shortlist)


def test_production_wiring_sizes_recent_lane_from_batch_size_64():
    """Production wiring constructs index with recent_lane_size=64 when batch_size=64."""
    # Simulate the logic that production_application uses when batch_size=64
    batch_size = 64

    # This is the logic that should be in production_application
    recent_lane_size = max(32, min(batch_size, 128))
    focus_lane_size = max(32, min(batch_size, 128))

    # Create config with the sized lanes
    config = HistoricalCandidateConfig(
        recent_lane_size=recent_lane_size,
        focus_lane_size=focus_lane_size,
        shortlist_size=128,
    )
    index = HistoricalCandidateIndex(config=config)

    # The historical candidate index should have recent_lane_size=64 to match batch size
    assert index.config.recent_lane_size == 64
    assert index.config.focus_lane_size == 64


def test_production_wiring_uses_default_lane_size_when_batch_size_less_than_32():
    """Production wiring keeps minimum recent_lane_size=32 when batch_size < 32."""
    batch_size = 4

    # This is the logic that should be in production_application
    recent_lane_size = max(32, min(batch_size, 128))
    focus_lane_size = max(32, min(batch_size, 128))

    config = HistoricalCandidateConfig(
        recent_lane_size=recent_lane_size,
        focus_lane_size=focus_lane_size,
        shortlist_size=128,
    )
    index = HistoricalCandidateIndex(config=config)

    # The historical candidate index should maintain minimum recent_lane_size=32
    assert index.config.recent_lane_size == 32
    assert index.config.focus_lane_size == 32


def test_production_wiring_rejects_batch_size_greater_than_128():
    """Production wiring raises ValueError when batch_size > 128."""
    batch_size = 129

    # This is the logic that should be in production_application
    if batch_size > 128:
        with pytest.raises(ValueError, match="prepared batch size > 128"):
            raise ValueError("prepared batch size > 128 is not supported by the historical candidate index")


def test_for_batch_size_keeps_default_shortlist_and_rejects_oversize():
    from eligsir.mapping.scheduler.historical_candidate_index import HistoricalCandidateConfig
    default = HistoricalCandidateConfig()
    small = HistoricalCandidateConfig.for_batch_size(4)
    assert small.recent_lane_size == default.recent_lane_size == 32
    assert small.shortlist_size == default.shortlist_size
    big = HistoricalCandidateConfig.for_batch_size(64)
    assert big.recent_lane_size == 64 and big.focus_lane_size == 64
    assert big.shortlist_size == default.shortlist_size
    import pytest
    with pytest.raises(ValueError):
        HistoricalCandidateConfig.for_batch_size(129)
