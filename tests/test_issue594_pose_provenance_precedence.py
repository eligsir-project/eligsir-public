"""Issue #594: pose provenance must not contradict the resolved pose source.

The observed run logged, from ONE namespace:

    [execution] pose_source          = dataset_pose
    [streaming] pose source: dataset_pose
    [pose-provenance] --- tier=slam_estimated method=auto_detected_topic (topic=/camera_pose)

``--orbbec_pose_topic`` defaults to ``/camera_pose`` and is read at exactly one
call site (``eligsir/io/sources/registry.py``), reached only for a
rosbag2/MCAP source. For a TUM directory it is inert -- so the topic name was
never evidence about that run's poses, yet it outranked the dataset layout.

These tests are table-driven per the issue's own test list, and are torch-free
so they collect in the CPU-CI image.
"""
from __future__ import annotations

import argparse

import pytest

from eligsir.slam.pose_provenance import (
    PoseProvenance,
    ProvenanceMethod,
    pose_provenance_contradictions,
    pose_topic_is_live_channel,
    resolve_pose_provenance,
)
from eligsir.utils.execution_semantics import (
    build_execution_semantics_block,
    resolve_execution_semantics,
)


def _tum(tmp_path, name="fr3"):
    d = tmp_path / name
    d.mkdir()
    (d / "groundtruth.txt").write_text("# stub\n")
    (d / "rgb.txt").write_text("# stub\n")
    (d / "depth.txt").write_text("# stub\n")
    return str(d)


def _rosbag2(tmp_path, name="bag"):
    d = tmp_path / name
    d.mkdir()
    (d / "metadata.yaml").write_text("rosbag2_bagfile_information:\n")
    return str(d)


# --------------------------------------------------------------------------
# 1-5: the issue's own table
# --------------------------------------------------------------------------

def test_case1_dataset_pose_run_with_default_camera_pose_topic(tmp_path):
    """The reported bug: TUM dataset + inert ``/camera_pose`` default."""
    result = resolve_pose_provenance(
        cli_override="auto",
        source_path=_tum(tmp_path),
        pose_topic="/camera_pose",
        resolved_pose_source="dataset_pose",
    )
    assert result.tier is PoseProvenance.DATASET_GT
    assert result.method is ProvenanceMethod.AUTO_DETECTED_DATASET_FORMAT
    assert result.topic_evidence_admissible is False
    assert result.provider == "dataset_files"
    assert result.live is False
    assert pose_provenance_contradictions(
        result=result, resolved_pose_source="dataset_pose",
        source_path=_tum(tmp_path, "fr3b"),
    ) == []


def test_case2_live_ros_camera_pose_is_slam_estimated_and_live():
    result = resolve_pose_provenance(
        cli_override="auto",
        source_path="ros2://live",
        pose_topic="/slam/pose",
        resolved_pose_source="dataset_pose",
    )
    assert result.tier is PoseProvenance.SLAM_ESTIMATED
    assert result.live is True
    assert result.provider == "live_ros_topic:/slam/pose"
    # A ros2:// source is EXEMPT from the contradiction check: the live
    # resolver normalizes ros_slam -> dataset_pose by design.
    assert pose_provenance_contradictions(
        result=result, resolved_pose_source="dataset_pose",
        source_path="ros2://live",
    ) == []


def test_case3_prerecorded_slam_estimates_from_a_bag(tmp_path):
    bag = _rosbag2(tmp_path)
    result = resolve_pose_provenance(
        cli_override="auto",
        source_path=bag,
        pose_topic="/camera_pose",
        resolved_pose_source="dataset_pose",
    )
    assert result.tier is PoseProvenance.SLAM_ESTIMATED
    assert result.method is ProvenanceMethod.AUTO_DETECTED_TOPIC
    assert result.topic_evidence_admissible is True
    # Prerecorded, not estimated online during THIS run -- recorded as a
    # provider/live field, deliberately not as a new tier.
    assert result.live is False
    assert result.provider == "bag_topic:/camera_pose"


def test_case4_explicit_user_override_still_wins(tmp_path):
    result = resolve_pose_provenance(
        cli_override="colmap_pseudo_gt",
        source_path=_tum(tmp_path),
        pose_topic="/camera_pose",
        resolved_pose_source="dataset_pose",
    )
    assert result.tier is PoseProvenance.COLMAP_PSEUDO_GT
    assert result.method is ProvenanceMethod.EXPLICIT_CLI_OVERRIDE
    assert result.provider == "declared"


def test_case5_no_trustworthy_inference_is_unknown_and_says_topic_was_inert(tmp_path):
    unknown_dir = tmp_path / "mystery"
    unknown_dir.mkdir()
    result = resolve_pose_provenance(
        cli_override="auto",
        source_path=str(unknown_dir),
        pose_topic="/camera_pose",
        resolved_pose_source="dataset_pose",
    )
    assert result.tier is PoseProvenance.UNKNOWN
    assert "INERT" in result.detail


# --------------------------------------------------------------------------
# admissibility gate
# --------------------------------------------------------------------------

@pytest.mark.parametrize("explicit", [True, False])
def test_explicit_admissibility_overrides_the_source_heuristic(tmp_path, explicit):
    result = resolve_pose_provenance(
        cli_override="auto",
        source_path=_tum(tmp_path),
        pose_topic="/camera_pose",
        pose_topic_active=explicit,
    )
    assert result.topic_evidence_admissible is explicit
    assert result.tier is (
        PoseProvenance.SLAM_ESTIMATED if explicit else PoseProvenance.DATASET_GT
    )


def test_empty_source_path_keeps_topic_evidence_admissible():
    """Back-compat: with no source to inspect, a caller-supplied topic still
    counts. The gate only demotes topics for a source known not to read one."""
    assert pose_topic_is_live_channel("") is True
    result = resolve_pose_provenance(cli_override="auto", pose_topic="/camera_pose")
    assert result.tier is PoseProvenance.SLAM_ESTIMATED


def test_mcap_file_source_is_a_live_topic_channel(tmp_path):
    mcap = tmp_path / "session.mcap"
    mcap.write_bytes(b"\x89MCAP")
    assert pose_topic_is_live_channel(str(mcap)) is True


# --------------------------------------------------------------------------
# contradiction detection
# --------------------------------------------------------------------------

def test_contradiction_is_reported_when_forced(tmp_path):
    forced = resolve_pose_provenance(
        cli_override="slam_estimated", source_path=_tum(tmp_path)
    )
    msgs = pose_provenance_contradictions(
        result=forced, resolved_pose_source="dataset_pose",
        source_path=_tum(tmp_path, "fr3c"),
    )
    assert len(msgs) == 1
    assert "different benchmark conditions" in msgs[0]


def test_contradiction_reported_for_estimator_run_claiming_dataset_gt(tmp_path):
    declared = resolve_pose_provenance(
        cli_override="dataset_gt", source_path=_tum(tmp_path)
    )
    msgs = pose_provenance_contradictions(
        result=declared, resolved_pose_source="open3d_odometry_live",
        source_path=_tum(tmp_path, "fr3d"),
    )
    assert len(msgs) == 1
    assert "in-process pose estimator" in msgs[0]


# --------------------------------------------------------------------------
# execution-semantics wiring (startup banner + report.json)
# --------------------------------------------------------------------------

def _args(**kw):
    base = dict(
        source_path="",
        streaming_replay=True,
        online_mode=False,
        pose_source="dataset_pose",
        pose_provenance="auto",
        orbbec_pose_topic="/camera_pose",
        orbbec_pose_source="camera_pose",
        streaming_ingestion_mode="iter_based",
    )
    base.update(kw)
    return argparse.Namespace(**base)


def test_execution_semantics_records_the_resolved_tier(tmp_path):
    sem = resolve_execution_semantics(_args(source_path=_tum(tmp_path)))
    prov = sem["pose_provenance_resolved"]
    assert sem["pose_source"] == "dataset_pose"
    assert prov["tier"] == "dataset_gt"
    assert prov["error"] == ""
    assert prov["contradictions"] == []
    # the raw request is still recorded separately
    assert sem["pose_provenance"] == "auto"


def test_execution_semantics_surfaces_the_contradiction_as_a_warning(tmp_path):
    block = build_execution_semantics_block(
        _args(source_path=_tum(tmp_path), pose_provenance="slam_estimated")
    )
    assert any("different benchmark conditions" in w for w in block["warnings"])
    # a warning, never an error: the combination is reachable.
    assert block["valid"] is True


def test_open3d_odometry_live_is_mirrored_into_the_resolved_pose_source(tmp_path):
    sem = resolve_execution_semantics(
        _args(source_path=_tum(tmp_path), orbbec_pose_source="open3d_odometry_live")
    )
    assert sem["pose_source"] == "open3d_odometry_live"
    assert sem["pose_provenance_resolved"]["tier"] == "slam_estimated"
    # and therefore NOT flagged as a contradiction
    assert sem["pose_provenance_resolved"]["contradictions"] == []


def test_live_ros2_run_emits_no_pose_contradiction_warning():
    block = build_execution_semantics_block(
        _args(
            source_path="ros2://live",
            ros2_live_pose_topic="/slam/pose",
            ros2_live_pose_source="odom",
        )
    )
    assert not any("benchmark conditions" in w for w in block["warnings"])
    assert block["pose_provenance_resolved"]["tier"] == "slam_estimated"
    assert block["pose_provenance_resolved"]["live"] is True
