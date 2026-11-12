"""Tests for issue #865, acceptance items 1-2: cross-check a bag-replay run's
real ingest count against the bag's own message index, and stop
``stream_end_reason == "bag_complete"`` from asserting a clean end when
frames are silently unaccounted for.

Covers only the pure detection/plumbing logic added for these items:

1. ``utils.partition_eval.compute_bag_source_gap`` -- the standalone gap
   calculation, given a synthetic ``bag_source_manifest`` (mirroring what
   ``human_launch_scripts/05_run_offline_rosbag.sh`` writes from a real
   ``metadata.yaml``) and synthetic ``ingest_counters``/``frames_admitted``.
2. ``compute_trajectory_coverage`` surfaces the same fields and folds a
   positively detected gap into ``mapping_result_valid`` -- reproducing the
   exact real-run numbers from issue #865 (kitchen1,
   ``i858_bagsmoke_20260816_004950``: bag has 2056 color / 2055 depth
   messages, trainer accepted 2023 frames, ``stream_end_reason ==
   "bag_complete"``) and confirming ``mapping_result_valid`` now flips to
   ``False`` where it previously read ``True``.
3. The "cannot verify" (``None``) case is distinct from "verified clean"
   (``True``) and from "verified a real gap" (``False``) at every layer --
   ``bag_source_manifest`` absent must never read as evidence of anything.
4. ``train_streaming._load_bag_source_manifest`` -- the file-reading glue
   that turns ``<model_path>/bag_source_manifest.json`` (or its absence)
   into the dict (or ``None``) fed into the above. A live GPU/container run
   was not exercised in this environment (no rclpy runtime available here);
   this file only exercises pure functions reachable without one -- same
   scoping note as ``tests/test_issue865_ingest_counters.py``.
"""

from __future__ import annotations

import ast
import json
import os
import pathlib

import pytest

from eligsir.utils.partition_eval import (
    PARTITIONS,
    compute_bag_source_gap,
    compute_trajectory_coverage,
)

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
TRAIN_STREAMING = REPO_ROOT / "train_streaming.py"


def _extract_load_bag_source_manifest():
    """Pull ``_load_bag_source_manifest`` out of ``train_streaming.py`` via
    AST + exec, rather than ``import train_streaming`` -- that module
    imports torch/torchvision at module level (it's the main training
    loop), which is unavailable/broken outside the training container on
    at least this environment. Same rationale as
    ``tests/test_issue646_report_kwargs_signature.py``'s own note on why it
    reads that module statically instead of importing it: the function
    under test has no dependency on torch, so there is no reason a torch
    import failure elsewhere in the file should make it untestable.
    """
    tree = ast.parse(TRAIN_STREAMING.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_load_bag_source_manifest":
            src = ast.get_source_segment(TRAIN_STREAMING.read_text(), node)
            ns: dict = {"os": os, "json": json, "Optional": __import__("typing").Optional}
            exec(src, ns)  # noqa: S102 -- test-only, extracting one pure function
            return ns["_load_bag_source_manifest"]
    raise AssertionError(
        "_load_bag_source_manifest() not found in train_streaming.py -- if it "
        "was renamed or moved, retarget this test."
    )

# The real per-topic message counts from issue #865's own evidence table
# (kitchen1_0.mcap, read via mcap.reader) -- used throughout so these tests
# are anchored to the actual incident, not an invented example.
REAL_ISSUE_MANIFEST = {
    "bag_path": "/tmp/kitchen1",
    "color_topic": "/camera/color/image_raw/compressed",
    "depth_topic": "/camera/depth/image_raw/compressed",
    "topics": {
        "/camera/color/image_raw/compressed": {"message_count": 2056},
        "/camera/depth/image_raw/compressed": {"message_count": 2055},
    },
}
# The trainer's real ingest_counters from that same run, per PR #867's
# mechanism -- rgb_received is what ros2_live.py's receive callback saw
# BEFORE any admission/pose-staleness rejection.
REAL_ISSUE_INGEST_COUNTERS = {"accepted_frames": 2018, "rgb_received": 2023}


# --- 1. compute_bag_source_gap -----------------------------------------------

class TestComputeBagSourceGap:
    def test_none_manifest_is_all_none(self):
        """No manifest (every non-bag-replay run, or a bag-replay run
        launched by a script predating this mechanism) must read as
        'cannot verify' everywhere -- never a fabricated clean/gap verdict."""
        gap = compute_bag_source_gap(None, None, None)
        assert gap["bag_source_frames_estimate"] is None
        assert gap["bag_source_frames_arrived"] is None
        assert gap["bag_source_frames_unaccounted"] is None
        assert gap["bag_source_tail_loss_detected"] is None
        assert gap["bag_source_completeness_verified"] is None

    def test_detects_the_real_issue_865_tail_loss(self):
        """Reproduces the exact incident: bag has 2056/2055 real messages,
        the trainer's rgb_received counter shows only 2023 -- a real,
        positive gap must be detected and quantified."""
        gap = compute_bag_source_gap(
            REAL_ISSUE_MANIFEST, REAL_ISSUE_INGEST_COUNTERS, frames_admitted=2018
        )
        assert gap["bag_source_n_messages"] == {"color": 2056, "depth": 2055}
        # Conservative (min-of-both-topics) estimate, per the docstring.
        assert gap["bag_source_frames_estimate"] == 2055
        assert gap["bag_source_frames_arrived"] == 2023
        assert gap["bag_source_arrival_basis"] == "ingest_counters.rgb_received"
        assert gap["bag_source_frames_unaccounted"] == 32
        assert gap["bag_source_tail_loss_detected"] is True
        assert gap["bag_source_completeness_verified"] is False
        assert "32" in gap["bag_source_tail_loss_reason"]

    def test_no_gap_when_counts_match(self):
        manifest = {
            "color_topic": "/c",
            "depth_topic": "/d",
            "topics": {"/c": {"message_count": 100}, "/d": {"message_count": 100}},
        }
        gap = compute_bag_source_gap(manifest, {"rgb_received": 100}, frames_admitted=100)
        assert gap["bag_source_frames_unaccounted"] == 0
        assert gap["bag_source_tail_loss_detected"] is False
        assert gap["bag_source_completeness_verified"] is True
        assert gap["bag_source_tail_loss_reason"] is None

    def test_prefers_ingest_counters_rgb_received_over_frames_admitted(self):
        """The arrival-count basis must prefer the more precise counter
        (rgb_received, which counts frames BEFORE admission rejection) and
        record which one it actually used."""
        manifest = {
            "color_topic": "/c",
            "depth_topic": "/d",
            "topics": {"/c": {"message_count": 100}, "/d": {"message_count": 100}},
        }
        gap = compute_bag_source_gap(
            manifest, {"rgb_received": 95}, frames_admitted=40
        )
        assert gap["bag_source_frames_arrived"] == 95
        assert gap["bag_source_arrival_basis"] == "ingest_counters.rgb_received"
        assert gap["bag_source_frames_unaccounted"] == 5

    def test_falls_back_to_frames_admitted_when_ingest_counters_absent(self):
        """An older run (or a source without ingest_counters) still gets a
        (weaker, documented-as-such) comparison rather than no signal at
        all."""
        manifest = {
            "color_topic": "/c",
            "depth_topic": "/d",
            "topics": {"/c": {"message_count": 100}, "/d": {"message_count": 100}},
        }
        gap = compute_bag_source_gap(manifest, None, frames_admitted=70)
        assert gap["bag_source_frames_arrived"] == 70
        assert gap["bag_source_arrival_basis"] == "frames_admitted_fallback"
        assert gap["bag_source_frames_unaccounted"] == 30
        assert gap["bag_source_tail_loss_detected"] is True

    def test_missing_arrival_data_is_unverifiable_not_clean(self):
        manifest = {
            "color_topic": "/c",
            "depth_topic": "/d",
            "topics": {"/c": {"message_count": 100}, "/d": {"message_count": 100}},
        }
        gap = compute_bag_source_gap(manifest, None, frames_admitted=None)
        assert gap["bag_source_frames_arrived"] is None
        assert gap["bag_source_frames_unaccounted"] is None
        assert gap["bag_source_tail_loss_detected"] is False
        assert gap["bag_source_completeness_verified"] is None

    def test_malformed_manifest_topics_missing(self):
        """A manifest present but missing expected keys degrades to
        unverifiable rather than raising."""
        gap = compute_bag_source_gap({"color_topic": "/c"}, {"rgb_received": 5}, 5)
        assert gap["bag_source_frames_estimate"] is None
        assert gap["bag_source_completeness_verified"] is None


# --- 2/3. compute_trajectory_coverage plumbing -------------------------------

class TestComputeTrajectoryCoverageBagSourceManifest:
    BASE = dict(scene_source="ros2://live", n_reg=None, frames_admitted=2018)

    def test_absent_manifest_leaves_every_bag_source_field_none(self):
        tc = compute_trajectory_coverage(partition_psnr=None, **self.BASE)
        for key in (
            "bag_source_n_messages",
            "bag_source_frames_estimate",
            "bag_source_frames_arrived",
            "bag_source_frames_unaccounted",
            "bag_source_tail_loss_detected",
            "bag_source_completeness_verified",
        ):
            assert tc[key] is None, key

    def test_absent_manifest_does_not_change_mapping_result_valid(self):
        """A run that never supplies a manifest (every run before this
        mechanism, or a non-bag-replay source) must be bit-for-bit
        unaffected -- this is purely additive."""
        partition_psnr = {f"p{i}": 25.0 for i in range(PARTITIONS)}
        without = compute_trajectory_coverage(
            partition_psnr=partition_psnr, eval_valid=True,
            stream_end_reason="bag_complete", **self.BASE,
        )
        with_ingest_only = compute_trajectory_coverage(
            partition_psnr=partition_psnr, eval_valid=True,
            stream_end_reason="bag_complete",
            ingest_counters=REAL_ISSUE_INGEST_COUNTERS,
            **self.BASE,
        )
        assert without["mapping_result_valid"] is True
        assert with_ingest_only["mapping_result_valid"] is True

    def test_reproduces_issue_865_mapping_result_valid_flips_false(self):
        """The decisive regression test: with the manifest AND
        ingest_counters both present, a real tail-loss run that used to
        report mapping_result_valid=True (stream_end_reason=='bag_complete'
        actively asserting a clean end) must now report False."""
        partition_psnr = {f"p{i}": 25.0 for i in range(PARTITIONS)}
        tc = compute_trajectory_coverage(
            partition_psnr=partition_psnr,
            eval_valid=True,
            stream_end_reason="bag_complete",
            ingest_counters=REAL_ISSUE_INGEST_COUNTERS,
            bag_source_manifest=REAL_ISSUE_MANIFEST,
            **self.BASE,
        )
        # stream_end_reason itself is untouched -- issue #865's own
        # suggested acceptance explicitly allows a companion field rather
        # than mutating this one's semantics.
        assert tc["stream_end_reason"] == "bag_complete"
        assert tc["bag_source_tail_loss_detected"] is True
        assert tc["bag_source_completeness_verified"] is False
        assert tc["mapping_result_valid"] is False

    def test_clean_bag_replay_with_manifest_present_stays_valid(self):
        """A genuinely complete bag replay (manifest present, counts match)
        must not be penalized merely for having a manifest at all."""
        manifest = {
            "color_topic": "/c", "depth_topic": "/d",
            "topics": {"/c": {"message_count": 50}, "/d": {"message_count": 50}},
        }
        partition_psnr = {f"p{i}": 25.0 for i in range(PARTITIONS)}
        tc = compute_trajectory_coverage(
            partition_psnr=partition_psnr,
            eval_valid=True,
            stream_end_reason="bag_complete",
            ingest_counters={"rgb_received": 50},
            bag_source_manifest=manifest,
            **self.BASE,
        )
        assert tc["bag_source_tail_loss_detected"] is False
        assert tc["bag_source_completeness_verified"] is True
        assert tc["mapping_result_valid"] is True

    def test_does_not_override_other_invalidating_signals(self):
        """A manifest confirming completeness must not paper over a
        SEPARATE, already-invalidating condition (e.g. a tracking reinit)."""
        manifest = {
            "color_topic": "/c", "depth_topic": "/d",
            "topics": {"/c": {"message_count": 50}, "/d": {"message_count": 50}},
        }
        partition_psnr = {f"p{i}": 25.0 for i in range(PARTITIONS)}
        tc = compute_trajectory_coverage(
            partition_psnr=partition_psnr,
            eval_valid=True,
            stream_end_reason="bag_complete",
            ingest_counters={"rgb_received": 50},
            bag_source_manifest=manifest,
            tracking_reinit_events=[{"frame": 123}],
            **self.BASE,
        )
        assert tc["bag_source_completeness_verified"] is True
        assert tc["map_reinit_invalidated"] is True
        assert tc["mapping_result_valid"] is False


# --- 4. train_streaming._load_bag_source_manifest ----------------------------

class TestLoadBagSourceManifest:
    def _load(self):
        return _extract_load_bag_source_manifest()

    def test_none_model_path(self):
        _load_bag_source_manifest = self._load()
        assert _load_bag_source_manifest(None) is None
        assert _load_bag_source_manifest("") is None

    def test_missing_file(self, tmp_path):
        _load_bag_source_manifest = self._load()
        assert _load_bag_source_manifest(str(tmp_path)) is None

    def test_reads_real_manifest(self, tmp_path):
        _load_bag_source_manifest = self._load()
        (tmp_path / "bag_source_manifest.json").write_text(
            json.dumps(REAL_ISSUE_MANIFEST)
        )
        loaded = _load_bag_source_manifest(str(tmp_path))
        assert loaded == REAL_ISSUE_MANIFEST

    def test_malformed_json_returns_none_not_raise(self, tmp_path):
        _load_bag_source_manifest = self._load()
        (tmp_path / "bag_source_manifest.json").write_text("{not valid json")
        assert _load_bag_source_manifest(str(tmp_path)) is None

    def test_non_dict_json_returns_none(self, tmp_path):
        _load_bag_source_manifest = self._load()
        (tmp_path / "bag_source_manifest.json").write_text("[1, 2, 3]")
        assert _load_bag_source_manifest(str(tmp_path)) is None
