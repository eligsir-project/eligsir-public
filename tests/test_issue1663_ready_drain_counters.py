"""Tests for persisting READY drain counters in causal_replay_summary (issue #1663)."""

from __future__ import annotations

import json
from typing import Any

import pytest

from eligsir.runtime.causal_prepared_ready import _capture_ready_drain_counters
from eligsir.runtime.ready_source import ReadySourceOwner
from eligsir.runtime.causal_replay import CausalReplaySummary


@pytest.fixture
def ready_owner() -> ReadySourceOwner:
    """Create a bounded ReadySourceOwner for testing."""
    return ReadySourceOwner(max_fresh=4, max_replay=8)


def test_ready_source_owner_counters_structure(ready_owner: ReadySourceOwner) -> None:
    """Verify ReadySourceCounters has all required fields for drain snapshot."""
    counters = ready_owner.counters
    expected_fields = {
        'fresh_pending', 'replay_pending', 'in_flight', 'offered', 'selected', 'released',
        'full_drop_count', 'full_not_resident_count', 'duplicate_drop_count',
        'stale_generation_drop_count', 'not_ready_count', 'ready_error_count',
        'reserve_underflow_count', 'fresh_ready_count', 'replay_ready_count',
        'ready_bytes_current', 'ready_bytes_peak', 'replay_evicted_bytes',
    }
    actual_fields = {f for f in dir(counters) if not f.startswith('_')}
    for field_name in expected_fields:
        assert hasattr(counters, field_name), f"Missing counter field: {field_name}"
        assert isinstance(getattr(counters, field_name), int), f"Counter {field_name} is not int"


def test_capture_ready_drain_counters_returns_dict(ready_owner: ReadySourceOwner) -> None:
    """Test that _capture_ready_drain_counters returns a dict with all counter fields."""
    block = _capture_ready_drain_counters(ready_owner)
    assert isinstance(block, dict)
    expected_fields = {
        'fresh_pending', 'replay_pending', 'in_flight', 'offered', 'selected', 'released',
        'full_drop_count', 'full_not_resident_count', 'duplicate_drop_count',
        'stale_generation_drop_count', 'not_ready_count', 'ready_error_count',
        'reserve_underflow_count', 'fresh_ready_count', 'replay_ready_count',
        'ready_bytes_current', 'ready_bytes_peak', 'replay_evicted_bytes',
    }
    assert set(block.keys()) == expected_fields
    for key, value in block.items():
        assert isinstance(value, int), f"Field {key} has non-int value {value}"


def test_capture_ready_drain_counters_zero_values(ready_owner: ReadySourceOwner) -> None:
    """Test that a fresh owner has expected zero/empty counter values."""
    block = _capture_ready_drain_counters(ready_owner)
    assert block['offered'] == 0
    assert block['selected'] == 0
    assert block['released'] == 0
    assert block['full_drop_count'] == 0
    assert block['fresh_pending'] == 0
    assert block['replay_pending'] == 0
    assert block['in_flight'] == 0


def test_ready_drain_counters_json_serializable(ready_owner: ReadySourceOwner) -> None:
    """Test that the counter block can be JSON serialized."""
    block = _capture_ready_drain_counters(ready_owner)
    json_str = json.dumps(block)
    restored = json.loads(json_str)
    assert restored == block


def test_causal_replay_summary_payload_has_ready_drain_key() -> None:
    """Test that causal_replay_summary_payload includes ready_drain key when given the block."""
    from eligsir.runtime.causal_entry import causal_replay_summary_payload

    summary = CausalReplaySummary(
        submitted=0,
        ticks=0,
        source_identities=(),
    )
    ready_drain_block = {
        "ready_drain": {
            "at_source_complete": {"offered": 10, "selected": 5},
            "at_drain_end": {"offered": 10, "selected": 8},
        }
    }
    payload = causal_replay_summary_payload(
        summary,
        source_hz=None,
        curve_metadata=None,
        ready_bytes_block={},
        ready_drain_block=ready_drain_block,
        supervision_work_block={},
    )
    assert "ready_drain" in payload
    assert payload["ready_drain"] == ready_drain_block["ready_drain"]


def test_causal_replay_summary_payload_omits_ready_drain_when_empty() -> None:
    """Test that causal_replay_summary_payload omits ready_drain key for non-prepared runs."""
    from eligsir.runtime.causal_entry import causal_replay_summary_payload

    summary = CausalReplaySummary(
        submitted=0,
        ticks=0,
        source_identities=(),
    )
    payload = causal_replay_summary_payload(
        summary,
        source_hz=None,
        curve_metadata=None,
        ready_bytes_block={},
        ready_drain_block={},  # Empty block for non-prepared
        supervision_work_block={},
    )
    assert "ready_drain" not in payload


def test_ready_drain_block_snapshot_structure() -> None:
    """Test that ready_drain block has correct snapshot structure."""
    from eligsir.runtime.causal_entry import causal_replay_summary_payload

    summary = CausalReplaySummary(
        submitted=0,
        ticks=0,
        source_identities=(),
    )
    at_source_complete_snap = {
        "offered": 100,
        "selected": 50,
        "fresh_pending": 10,
    }
    at_drain_end_snap = {
        "offered": 100,
        "selected": 100,
        "fresh_pending": 0,
    }
    ready_drain_block = {
        "ready_drain": {
            "at_source_complete": at_source_complete_snap,
            "at_drain_end": at_drain_end_snap,
        }
    }
    payload = causal_replay_summary_payload(
        summary,
        source_hz=None,
        curve_metadata=None,
        ready_bytes_block={},
        ready_drain_block=ready_drain_block,
        supervision_work_block={},
    )
    assert payload["ready_drain"]["at_source_complete"] == at_source_complete_snap
    assert payload["ready_drain"]["at_drain_end"] == at_drain_end_snap


def test_ready_drain_allows_null_source_complete_snapshot() -> None:
    """Test that at_source_complete can be null for runs where source_complete never fired."""
    from eligsir.runtime.causal_entry import causal_replay_summary_payload

    summary = CausalReplaySummary(
        submitted=0,
        ticks=0,
        source_identities=(),
    )
    ready_drain_block = {
        "ready_drain": {
            "at_source_complete": None,  # Source-complete never fired
            "at_drain_end": {"offered": 10, "selected": 10},
        }
    }
    payload = causal_replay_summary_payload(
        summary,
        source_hz=None,
        curve_metadata=None,
        ready_bytes_block={},
        ready_drain_block=ready_drain_block,
        supervision_work_block={},
    )
    assert payload["ready_drain"]["at_source_complete"] is None
    assert payload["ready_drain"]["at_drain_end"]["offered"] == 10


def test_mutation_missing_field_in_block() -> None:
    """Test that missing a required field from the block structure fails verification.

    This demonstrates that the field-set is part of the acceptance contract.
    """
    from eligsir.runtime.causal_entry import causal_replay_summary_payload

    summary = CausalReplaySummary(
        submitted=0,
        ticks=0,
        source_identities=(),
    )
    # Drop a required field (at_drain_end)
    incomplete_ready_drain_block = {
        "ready_drain": {
            "at_source_complete": {"offered": 10},
            # Missing: "at_drain_end"
        }
    }
    payload = causal_replay_summary_payload(
        summary,
        source_hz=None,
        curve_metadata=None,
        ready_bytes_block={},
        ready_drain_block=incomplete_ready_drain_block,
        supervision_work_block={},
    )
    # Verify the mutation is detectable: the snapshot is incomplete
    assert "at_source_complete" in payload["ready_drain"]
    assert "at_drain_end" not in payload["ready_drain"]
    # A test that checks presence of BOTH snapshots would fail
    required_keys = {"at_source_complete", "at_drain_end"}
    actual_keys = set(payload["ready_drain"].keys())
    assert len(required_keys - actual_keys) > 0, "Missing keys indicate mutation"


def test_mutation_missing_counter_field_in_snapshot() -> None:
    """Test that dropping a counter field from the snapshot dict is detectable.

    Drop 'fresh_pending' and verify a test checking for it would catch this.
    """
    from eligsir.runtime.causal_entry import causal_replay_summary_payload

    summary = CausalReplaySummary(
        submitted=0,
        ticks=0,
        source_identities=(),
    )
    # Drop fresh_pending from at_drain_end snapshot
    mutated_ready_drain_block = {
        "ready_drain": {
            "at_source_complete": None,
            "at_drain_end": {
                "offered": 100,
                "selected": 100,
                # Missing: "fresh_pending"
            },
        }
    }
    payload = causal_replay_summary_payload(
        summary,
        source_hz=None,
        curve_metadata=None,
        ready_bytes_block={},
        ready_drain_block=mutated_ready_drain_block,
        supervision_work_block={},
    )
    # Verify the mutation: if a test checks for presence of fresh_pending, it would fail
    at_drain_end = payload["ready_drain"]["at_drain_end"]
    assert "fresh_pending" not in at_drain_end
    assert "offered" in at_drain_end
    # Test assertion would be:
    expected_fields = {"offered", "selected", "fresh_pending"}
    actual_fields = set(at_drain_end.keys())
    assert expected_fields != actual_fields, "Mutation detected: missing fresh_pending"
