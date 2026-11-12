"""Tests for the canonical control-plane replay digest."""

from eligsir.runtime.deterministic_digest import (
    record_global,
    report_with_global_digest,
    reset_global,
)


def test_report_block_preserves_mode_and_appends_current_digest() -> None:
    reset_global()
    record_global("ingress", source_id="frame-7")

    block = report_with_global_digest({"strict_deterministic": True})

    assert block["strict_deterministic"] is True
    assert block["replay_digest"]["event_count"] == 1
    assert block["replay_digest"]["event_counts"] == {"ingress": 1}
    assert block["replay_digest"]["tensor_payload_bytes_hashed"] == 0
    assert block["replay_digest"]["coverage_complete"] is False
    assert block["replay_digest"]["missing_event_kinds"] == [
        "mutation_commit",
        "optimizer_batch",
    ]


def test_report_block_exists_without_legacy_determinism_mode() -> None:
    reset_global()

    block = report_with_global_digest(None)

    assert set(block) == {"replay_digest"}
    assert block["replay_digest"]["event_count"] == 0
