"""Tests for issue #1114: the bag-source completeness guard must tell an
ATTRIBUTED matcher drop apart from a silently lost frame.

Background. ``utils.partition_eval.compute_bag_source_gap`` (issue #865,
shipped in #1057) flags ``mapping_result_valid = False`` whenever the bag's
own ``min(color, depth)`` message count exceeds ``ingest_counters.
rgb_received``. On a genuinely CLEAN kitchen1 replay that difference is
exactly ``1`` -- a real matcher drop attributed to a 34.6 ms stream gap --
so every healthy run on the promoted path was being stamped invalid, which
CLAUDE.md gives the standing of ``eval_valid == False``.

Both anchor cases below use the REAL measured numbers from the two runs in
``docs/architecture/2026-08-23-issue865-false-clean-guard-verification.md``:

* clean arm  (``i865_clean_a``):      bag 2056/2055, ``rgb_received`` 2054
* loss arm   (``i865_lossinject_a``): bag 2056/2055, ``rgb_received`` 1988
  (``BAG_PLAY_EXTRA_ARGS='--playback-duration 66'``)

The ledger halves of those two cases are reconstructed from the sync node's
published accounting for that bag (2054 published, 1 attributed matcher
drop, 1 pending at teardown); the *parser* that reads such a ledger off disk
is separately tested against verbatim log lines captured from a real running
``sync_node`` (``tests/fixtures/issue1114/`` -- see its README).

One number in the clean row -- ``color.arrived == 2056`` -- is synthesized
rather than measured: neither #1113 run's image carried the ledger, so no
arrival count was ever recorded for that bag. It should hold (canonical mode
runs ``decompress_rgbd`` with ``sync:=false``, a per-message decode-and-
republish), but it is unverified in production and is the first thing to
check on the next real replay. See the honest-gaps section of
``docs/architecture/2026-08-23-issue1114-bag-source-attribution.md``.

Scoping note, same as ``tests/test_issue865_bag_source_gap.py``: only pure
functions reachable without a GPU/rclpy runtime are exercised here.
"""

from __future__ import annotations

import pathlib

import pytest

from eligsir.utils.partition_eval import compute_bag_source_gap, compute_trajectory_coverage
from eligsir.utils.sync_pair_ledger import (
    attribution_trust_refusal,
    load_sync_pair_ledger,
    parse_pair_accounting_line,
    parse_sync_pair_ledger_log,
)

FIXTURES = pathlib.Path(__file__).resolve().parent / "fixtures" / "issue1114"

# kitchen1's real bag, per the #1113 verification doc.
KITCHEN1_MANIFEST = {
    "bag_path": "/bags/kitchen1",
    "color_topic": "/camera/color/image_raw/compressed",
    "depth_topic": "/camera/depth/image_raw/compressed",
    "topics": {
        "/camera/color/image_raw/compressed": {"message_count": 2056},
        "/camera/depth/image_raw/compressed": {"message_count": 2055},
    },
}


def _ledger(
    color_arrived=2056,
    color_published=2054,
    color_pending=1,
    color_dropped=1,
    depth_arrived=2055,
    depth_published=2054,
    depth_pending=0,
    depth_dropped=1,
    balanced=True,
    reconstructed=0,
):
    return {
        "line_kind": "periodic",
        "pairs_matched": color_published,
        "pairs_published": color_published,
        "paired_without_arrival_record": reconstructed,
        "color": {
            "arrived": color_arrived,
            "published": color_published,
            "pending": color_pending,
            "dropped": color_dropped,
            "balanced": balanced,
            "drop_reasons": {"unpaired_retired": color_dropped},
        },
        "depth": {
            "arrived": depth_arrived,
            "published": depth_published,
            "pending": depth_pending,
            "dropped": depth_dropped,
            "balanced": balanced,
            "drop_reasons": {"unpaired_retired": depth_dropped},
        },
    }


# --- 1. The parser, against verbatim output of a real sync_node --------------


class TestParserAgainstRealSyncNodeOutput:
    def test_final_line_from_a_real_run_parses(self):
        parsed = parse_sync_pair_ledger_log(str(FIXTURES / "rgbd_sync_node_gap.log"))
        assert parsed is not None
        # The FINAL line is preferred over the periodic ones in the same file.
        assert parsed["line_kind"] == "final"
        assert parsed["n_accounting_lines"] == 5
        assert parsed["pairs_matched"] == 59
        assert parsed["pairs_published"] == 59
        assert parsed["paired_without_arrival_record"] == 0
        assert parsed["color"] == {
            "arrived": 60,
            "published": 59,
            "pending": 0,
            "dropped": 1,
            "balanced": True,
            "drop_reasons": {"unpaired_retired": 1},
        }
        assert parsed["depth"]["arrived"] == 59
        assert parsed["depth"]["published"] == 59
        assert parsed["depth"]["drop_reasons"] == {}

    def test_periodic_snapshot_with_pending_frames_parses(self):
        parsed = parse_sync_pair_ledger_log(str(FIXTURES / "rgbd_sync_node_pending.log"))
        assert parsed is not None
        # No FINAL line in this capture -> the LAST periodic line is used.
        assert parsed["line_kind"] == "periodic"
        assert parsed["color"]["arrived"] == 60
        assert parsed["color"]["published"] == 56
        assert parsed["color"]["pending"] == 4
        assert parsed["depth"]["arrived"] == 56

    def test_non_accounting_lines_are_ignored(self):
        assert parse_pair_accounting_line("[INFO] [x] [node]: received first color CameraInfo") is None
        assert parse_pair_accounting_line("") is None
        assert parse_pair_accounting_line(None) is None

    def test_truncated_line_is_refused_not_partially_parsed(self):
        """A half-written line must degrade to None, never to a dict a caller
        could mistake for a complete ledger."""
        line = "[INFO] [x] [node]: pair accounting: pairs_matched=59 pairs_published=59 | color: arriv"
        assert parse_pair_accounting_line(line) is None

    def test_missing_log_is_none_not_an_exception(self, tmp_path):
        assert load_sync_pair_ledger(str(tmp_path)) is None
        assert load_sync_pair_ledger(None) is None
        assert parse_sync_pair_ledger_log(str(tmp_path / "nope.log")) is None

    def test_load_reads_the_conventional_basename(self, tmp_path):
        (tmp_path / "rgbd_sync_node.log").write_text(
            (FIXTURES / "rgbd_sync_node_gap.log").read_text()
        )
        assert load_sync_pair_ledger(str(tmp_path))["pairs_published"] == 59


# --- 2. TRUE NEGATIVE: the clean kitchen1 arm is no longer refused -----------


class TestCleanArmNoLongerFalseFlagged:
    """The exact numbers from ``output/i865_clean_a`` (issue #1114's report)."""

    def test_without_a_ledger_the_pre_1114_false_positive_is_reproduced(self):
        gap = compute_bag_source_gap(KITCHEN1_MANIFEST, {"rgb_received": 2054}, 2054)
        assert gap["bag_source_frames_unaccounted"] == 1
        assert gap["bag_source_tail_loss_detected"] is True
        assert gap["bag_source_completeness_verified"] is False
        # ... and it self-identifies as having had no attribution channel.
        assert gap["bag_source_attribution_basis"] is None
        assert gap["bag_source_attributed_drops"] is None

    def test_with_the_ledger_the_clean_arm_reports_zero_unaccounted(self):
        gap = compute_bag_source_gap(
            KITCHEN1_MANIFEST, {"rgb_received": 2054}, 2054, sync_pair_ledger=_ledger()
        )
        assert gap["bag_source_raw_gap"] == 1
        assert gap["bag_source_attributed_drops"] == 1
        assert gap["bag_source_frames_unaccounted"] == 0
        assert gap["bag_source_tail_loss_detected"] is False
        assert gap["bag_source_completeness_verified"] is True
        assert gap["bag_source_attribution_trusted"] is True
        assert gap["bag_source_attribution_refused_reason"] is None
        assert gap["bag_source_upstream_shortfall"] == {"color": 0, "depth": 0}
        assert gap["bag_source_unpaired_beyond_source_asymmetry"] == 0

    def test_clean_arm_mapping_result_valid_flips_back_to_true(self):
        cov = compute_trajectory_coverage(
            scene_source="/bags/kitchen1",
            n_reg=2055,
            frames_admitted=2054,
            partition_psnr={},
            stream_end_reason="bag_complete",
            eval_valid=True,
            coverage_counters={
                "coverage_ledger_version": 2,
                "resolved_source_frames": 2055,
                "raw_registry_source_frames": 2055,
                "source_denominator_provenance": "post_stride_frame_source",
                "source_frames_ingested_total": 2055,
                "bootstrap_frames_consumed": 0,
                "reserved_holdout_frames": 0,
                "reserved_guard_frames": 0,
                "rejected_invalid_frames": 1,
                "eligible_mapping_frames": 2054,
                "frames_admitted": 2054,
                "mapping_skipped_frames": 0,
                "unprocessed_frames": 0,
            },
            ingest_counters={"rgb_received": 2054},
            bag_source_manifest=KITCHEN1_MANIFEST,
            sync_pair_ledger=_ledger(),
        )
        assert cov["bag_source_frames_unaccounted"] == 0
        assert cov["bag_source_completeness_verified"] is True
        assert cov["mapping_result_valid"] is True


# --- 3. TRUE POSITIVE: injected loss is still caught -------------------------


class TestInjectedLossStillCaught:
    """``output/i865_lossinject_a``: playback truncated at 66 s, 67 frames
    genuinely never reach the trainer. The sync node sees them go missing
    upstream of itself, so its `arrived` counters fall short of the bag's
    message counts and attribution is refused outright."""

    LOSS_LEDGER = _ledger(
        color_arrived=1989,
        color_published=1988,
        color_pending=1,
        color_dropped=0,
        depth_arrived=1988,
        depth_published=1988,
        depth_pending=0,
        depth_dropped=0,
    )

    def test_attribution_is_refused_when_the_node_never_saw_the_frames(self):
        refusal = attribution_trust_refusal(self.LOSS_LEDGER, 2056, 2055)
        assert refusal is not None and "color_arrived=1989" in refusal

    def test_loss_arm_still_reports_67_unaccounted(self):
        gap = compute_bag_source_gap(
            KITCHEN1_MANIFEST,
            {"rgb_received": 1988},
            1988,
            sync_pair_ledger=self.LOSS_LEDGER,
        )
        assert gap["bag_source_raw_gap"] == 67
        assert gap["bag_source_frames_unaccounted"] == 67
        assert gap["bag_source_tail_loss_detected"] is True
        assert gap["bag_source_completeness_verified"] is False
        assert gap["bag_source_attribution_trusted"] is False
        assert gap["bag_source_attributed_drops"] is None
        # The upstream shortfall is named explicitly, so a reader can see
        # WHERE the frames went missing rather than only that they did.
        assert gap["bag_source_upstream_shortfall"] == {"color": 67, "depth": 67}

    def test_loss_arm_mapping_result_valid_stays_false(self):
        cov = compute_trajectory_coverage(
            scene_source="/bags/kitchen1",
            n_reg=2055,
            frames_admitted=1988,
            partition_psnr={},
            stream_end_reason="bag_complete",
            eval_valid=True,
            ingest_counters={"rgb_received": 1988},
            bag_source_manifest=KITCHEN1_MANIFEST,
            sync_pair_ledger=self.LOSS_LEDGER,
        )
        assert cov["bag_source_frames_unaccounted"] == 67
        assert cov["mapping_result_valid"] is False


# --- 4. The subtraction is a MEASURED BOUND, not a tolerance -----------------


class TestAttributionCannotLaunderRealLoss:
    def test_matcher_losing_pairable_frames_is_not_absorbed(self):
        """Both streams arrive in full, but the matcher publishes 500 fewer
        pairs than it could have. Those frames really are lost to the
        trainer: the source's own asymmetry justifies exactly 1 of them, and
        the other 499 must stay flagged."""
        ledger = _ledger(
            color_arrived=2056,
            color_published=1554,
            color_pending=0,
            color_dropped=502,
            depth_arrived=2055,
            depth_published=1554,
            depth_pending=0,
            depth_dropped=501,
        )
        gap = compute_bag_source_gap(
            KITCHEN1_MANIFEST, {"rgb_received": 1554}, 1554, sync_pair_ledger=ledger
        )
        assert gap["bag_source_attribution_trusted"] is True
        assert gap["bag_source_raw_gap"] == 501
        # Only the 1 frame the bag's own color/depth asymmetry proves.
        assert gap["bag_source_attributed_drops"] == 1
        assert gap["bag_source_frames_unaccounted"] == 500
        assert gap["bag_source_tail_loss_detected"] is True
        assert gap["bag_source_unpaired_beyond_source_asymmetry"] == 500

    def test_symmetric_bag_justifies_no_subtraction_at_all(self):
        manifest = {
            "color_topic": "/c",
            "depth_topic": "/d",
            "topics": {"/c": {"message_count": 100}, "/d": {"message_count": 100}},
        }
        ledger = _ledger(
            color_arrived=100,
            color_published=95,
            color_pending=0,
            color_dropped=5,
            depth_arrived=100,
            depth_published=95,
            depth_pending=0,
            depth_dropped=5,
        )
        gap = compute_bag_source_gap(manifest, {"rgb_received": 95}, 95, sync_pair_ledger=ledger)
        assert gap["bag_source_attributed_drops"] == 0
        assert gap["bag_source_frames_unaccounted"] == 5

    @pytest.mark.parametrize(
        "kwargs, needle",
        [
            ({"reconstructed": 3}, "arrival_counts_reconstructed"),
            ({"balanced": False}, "ledger_unbalanced"),
            ({"depth_arrived": 2050}, "depth_arrived=2050"),
        ],
    )
    def test_untrustworthy_ledgers_are_refused(self, kwargs, needle):
        assert needle in attribution_trust_refusal(_ledger(**kwargs), 2056, 2055)

    def test_a_stale_snapshot_fails_the_same_gate(self):
        """A periodic snapshot taken before the last frames were processed
        reports arrived < message_count, so freshness needs no separate
        heuristic -- it fails the arrival gate and falls back to pre-#1114
        behaviour."""
        stale = _ledger(color_arrived=1000, color_published=999, depth_arrived=1000,
                        depth_published=999)
        gap = compute_bag_source_gap(
            KITCHEN1_MANIFEST, {"rgb_received": 2054}, 2054, sync_pair_ledger=stale
        )
        assert gap["bag_source_attribution_trusted"] is False
        assert gap["bag_source_frames_unaccounted"] == 1

    def test_no_ledger_is_cannot_verify_not_attributed_zero(self):
        assert attribution_trust_refusal(None, 2056, 2055) == "no_sync_pair_ledger"
        gap = compute_bag_source_gap(None, None, None)
        for key in (
            "bag_source_attributed_drops",
            "bag_source_attribution_basis",
            "bag_source_attribution_trusted",
            "bag_source_raw_gap",
            "bag_source_upstream_shortfall",
        ):
            assert gap[key] is None, key


# --- 5. Behaviour-neutrality for every run that has no sync node -------------


def test_pre_1114_call_signature_still_works_and_is_unchanged():
    """Every existing caller passes three positional args. Their results must
    be bit-identical to before #1114 (no ledger -> no subtraction)."""
    gap = compute_bag_source_gap(KITCHEN1_MANIFEST, {"rgb_received": 2000}, 2000)
    assert gap["bag_source_frames_estimate"] == 2055
    assert gap["bag_source_frames_arrived"] == 2000
    assert gap["bag_source_frames_unaccounted"] == 55
    assert gap["bag_source_arrival_basis"] == "ingest_counters.rgb_received"
    assert gap["bag_source_tail_loss_detected"] is True
