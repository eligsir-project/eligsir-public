"""Issue #1302: the transport-neutral holdout rule extracted for a
StreamingScene-free live caller (eligsir-ros2's lifecycle node).

Deliberately torch-free (no ``scene.streaming_scene`` import) so this file
collects and runs in the lean CPU CI image, per ``tests/conftest.py``'s
``_NEEDS_TORCH`` contract.
"""

from __future__ import annotations

import pytest

from eligsir.utils.holdout_rule import (
    LiveHoldoutCollector,
    online_every_nth_holdout,
    online_holdout_eval_metadata,
    online_holdout_rule_hash,
    online_holdout_rule_payload,
)


class TestOnlineEveryNthHoldout:
    def test_disabled_when_eval_hold_zero_or_negative(self):
        for ordinal in range(0, 20):
            assert online_every_nth_holdout(ordinal, 0) is False
            assert online_every_nth_holdout(ordinal, -1) is False

    def test_every_nth_ordinal_is_holdout(self):
        eval_hold = 5
        holdouts = [o for o in range(0, 30) if online_every_nth_holdout(o, eval_hold)]
        assert holdouts == [0, 5, 10, 15, 20, 25]

    def test_matches_streaming_scene_inline_arithmetic(self):
        # Pins the exact arithmetic StreamingScene.holdout_status's live
        # branch used before the #1302 extraction: `eval_hold > 0 and
        # ordinal % eval_hold == 0`.
        for eval_hold in (0, 1, 3, 7):
            for ordinal in range(0, 40):
                expected = eval_hold > 0 and ordinal % eval_hold == 0
                assert online_every_nth_holdout(ordinal, eval_hold) is expected


class TestRuleHash:
    def test_hash_none_when_disabled(self):
        assert (
            online_holdout_rule_hash(eval_hold=0, ordinal_definition_version="v1")
            is None
        )

    def test_hash_deterministic_and_16_hex_chars(self):
        h1 = online_holdout_rule_hash(
            eval_hold=8,
            ordinal_definition_version="received_seq_v1",
            eval_guard=1,
            eval_min_baseline_m=0.03,
            eval_min_angle_deg=2.0,
            source_id="/data/kitchen1",
        )
        h2 = online_holdout_rule_hash(
            eval_hold=8,
            ordinal_definition_version="received_seq_v1",
            eval_guard=1,
            eval_min_baseline_m=0.03,
            eval_min_angle_deg=2.0,
            source_id="/data/kitchen1",
        )
        assert h1 == h2
        assert h1 is not None
        assert len(h1) == 16
        int(h1, 16)  # hex

    def test_hash_changes_with_source_id(self):
        common = dict(
            eval_hold=8,
            ordinal_definition_version="received_seq_v1",
            eval_guard=0,
            eval_min_baseline_m=0.03,
            eval_min_angle_deg=2.0,
        )
        h_a = online_holdout_rule_hash(source_id="/data/kitchen1", **common)
        h_b = online_holdout_rule_hash(source_id="/data/floor2", **common)
        assert h_a != h_b

    def test_payload_matches_expected_pipe_format(self):
        payload = online_holdout_rule_payload(
            eval_hold=8,
            ordinal_definition_version="received_seq_v1",
            eval_guard=1,
            eval_min_baseline_m=0.03,
            eval_min_angle_deg=2.0,
            source_id="/data/kitchen1",
        )
        assert payload == (
            "eval_hold=8|ordinal=received_seq_v1|guard=1|"
            "min_t=0.030000|min_a=2.000000|source=/data/kitchen1"
        )


class TestEvalMetadataShape:
    def test_disabled_shape(self):
        meta = online_holdout_eval_metadata(eval_hold=0, n_holdout=0)
        assert meta["holdout_hash"] is None
        assert meta["holdout_selection"] == "online_every_nth"
        assert meta["n_holdout"] == 0

    def test_enabled_shape_has_streaming_scene_compatible_keys(self):
        meta = online_holdout_eval_metadata(
            eval_hold=8,
            n_holdout=12,
            ordinal_definition_version="received_seq_v1",
            eval_guard=1,
            eval_min_baseline_m=0.03,
            eval_min_angle_deg=2.0,
            source_id="/data/kitchen1",
            n_guard=3,
        )
        assert set(meta.keys()) == {
            "eval_hold",
            "eval_min_baseline_m",
            "eval_min_angle_deg",
            "eval_guard",
            "n_holdout",
            "n_guard",
            "holdout_hash",
            "holdout_selection",
        }
        assert meta["holdout_selection"] == "online_every_nth_ordinal_rule"
        assert meta["n_holdout"] == 12
        assert meta["n_guard"] == 3
        assert meta["holdout_hash"] is not None


class TestLiveHoldoutCollector:
    def test_disabled_collector_never_designates(self):
        c = LiveHoldoutCollector(eval_hold=0)
        assert c.enabled is False
        for ordinal in range(20):
            assert c.classify(ordinal) is False
        assert c.counts() == {
            "n_holdout_designated": 0,
            "n_holdout_captured": 0,
            "n_holdout_dropped_over_cap": 0,
        }

    def test_classify_matches_rule(self):
        c = LiveHoldoutCollector(eval_hold=4)
        assert c.enabled is True
        results = [c.classify(o) for o in range(12)]
        assert results == [
            o % 4 == 0 for o in range(12)
        ]
        assert c.counts()["n_holdout_designated"] == 3  # 0, 4, 8

    def test_no_cap_always_allows_capture(self):
        c = LiveHoldoutCollector(eval_hold=2, max_holdout_frames=None)
        for ordinal in range(0, 20, 2):
            assert c.classify(ordinal) is True
            assert c.should_capture() is True
            c.record_captured()
        assert c.counts()["n_holdout_captured"] == 10
        assert c.counts()["n_holdout_dropped_over_cap"] == 0

    def test_cap_stops_capture_but_never_stops_designation(self):
        c = LiveHoldoutCollector(eval_hold=1, max_holdout_frames=3)
        captured = 0
        dropped = 0
        for ordinal in range(10):
            assert c.classify(ordinal) is True  # eval_hold=1 -> every frame
            if c.should_capture():
                c.record_captured()
                captured += 1
            else:
                c.record_dropped_over_cap()
                dropped += 1
        assert captured == 3
        assert dropped == 7
        counts = c.counts()
        assert counts["n_holdout_designated"] == 10
        assert counts["n_holdout_captured"] == 3
        assert counts["n_holdout_dropped_over_cap"] == 7

    def test_negative_cap_rejected(self):
        with pytest.raises(ValueError):
            LiveHoldoutCollector(eval_hold=1, max_holdout_frames=-1)

    def test_eval_metadata_uses_captured_not_designated_count(self):
        c = LiveHoldoutCollector(eval_hold=1, max_holdout_frames=2)
        for ordinal in range(5):
            c.classify(ordinal)
            if c.should_capture():
                c.record_captured()
            else:
                c.record_dropped_over_cap()
        meta = c.eval_metadata(source_id="/data/floor2")
        assert meta["n_holdout"] == 2  # captured, not the 5 designated
