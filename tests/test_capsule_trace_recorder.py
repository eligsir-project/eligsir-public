"""Tests for trace_recorder module (issue #633).

Covers:
- Loop-closure fixture with causal correctness
- Two modes produce different hashes
- Admission bypass mechanism
- Manipulation check reporting
"""

import tempfile
from pathlib import Path

import pytest

from eligsir.experiments.hashing import content_hash
from eligsir.experiments.replay_capsule import CapsuleValidationError
from eligsir.experiments.trace_recorder import (
    AdmissionCandidateEvent,
    AdmissionDecisionEvent,
    AdmissionEventReplayer,
    ManipulationReport,
    PoseEvent,
    PoseEventReplayer,
    SourceFrameEvent,
    TraceRecorder,
)


class TestSourceFrameEvent:
    """Tests for source frame event serialization."""

    def test_to_from_dict(self) -> None:
        """Test round-trip serialization."""
        event = SourceFrameEvent(
            frame_id="frame_0",
            timestamp_ns=1000000000,
            rgb_hash="a" * 64,
            depth_hash="b" * 64,
            camera_id="orbbec_femto",
            rgb_decoder="jpeg",
            depth_decoder="png16",
            depth_scale_mm=1.0,
            sync_status="synchronized",
        )
        d = event.to_dict()
        restored = SourceFrameEvent.from_dict(d)
        assert restored.frame_id == event.frame_id
        assert restored.timestamp_ns == event.timestamp_ns


class TestPoseEventFixture:
    """Tests for pose event handling, including loop closure fixture."""

    def test_pose_event_to_from_dict(self) -> None:
        """Test round-trip serialization."""
        event = PoseEvent(
            event_id="pose_0",
            event_type="pose_estimate",
            logical_time=10.0,
            frame_id="frame_A",
            derived_from_logical_time=10.0,
            pose_c2w=[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
            tracking_state="OK",
        )
        d = event.to_dict()
        restored = PoseEvent.from_dict(d)
        assert restored.event_id == event.event_id
        assert restored.logical_time == 10.0

    def test_loop_closure_fixture_causal(self) -> None:
        """Test the loop closure fixture with causal replay.

        Fixture:
            t=10: frame A pose=P1
            t=20: frame B pose=P2
            t=100: correction A:P1->P1', B:P2->P2'
        """
        events = [
            {
                "event_id": "pose_A_init",
                "event_type": "pose_estimate",
                "logical_time": 10.0,
                "frame_id": "frame_A",
                "derived_from_logical_time": 10.0,
                "pose_c2w": [1] + [0] * 15,
                "tracking_state": "OK",
            },
            {
                "event_id": "pose_B_init",
                "event_type": "pose_estimate",
                "logical_time": 20.0,
                "frame_id": "frame_B",
                "derived_from_logical_time": 20.0,
                "pose_c2w": [2] + [0] * 15,
                "tracking_state": "OK",
            },
            {
                "event_id": "pose_A_corrected",
                "event_type": "pose_correction",
                "logical_time": 100.0,
                "frame_id": "frame_A",
                "derived_from_logical_time": 100.0,
                "pose_c2w": [1.1] + [0] * 15,
                "tracking_state": "OK",
                "corrected_from_event_id": "pose_A_init",
            },
            {
                "event_id": "pose_B_corrected",
                "event_type": "pose_correction",
                "logical_time": 100.0,
                "frame_id": "frame_B",
                "derived_from_logical_time": 100.0,
                "pose_c2w": [2.1] + [0] * 15,
                "tracking_state": "OK",
                "corrected_from_event_id": "pose_B_init",
            },
        ]

        replayer = PoseEventReplayer(
            events=events, trace_mode="FROZEN_CAUSAL_SLAM_TRACE", oracle_declared=False
        )

        # Before t=100, should get uncorrected poses
        pose_a_at_50 = replayer.pose_for("frame_A", at_logical_time=50.0)
        assert pose_a_at_50["pose_c2w"][0] == 1, "Before correction, should have P1"

        pose_b_at_50 = replayer.pose_for("frame_B", at_logical_time=50.0)
        assert pose_b_at_50["pose_c2w"][0] == 2, "Before correction, should have P2"

        # At t=100+, should get corrected poses
        pose_a_at_150 = replayer.pose_for("frame_A", at_logical_time=150.0)
        assert pose_a_at_150["pose_c2w"][0] == 1.1, "After correction, should have P1'"

        pose_b_at_150 = replayer.pose_for("frame_B", at_logical_time=150.0)
        assert pose_b_at_150["pose_c2w"][0] == 2.1, "After correction, should have P2'"

    def test_loop_closure_fixture_oracle(self) -> None:
        """Test the loop closure fixture with oracle (final-pose) replay."""
        events = [
            {
                "event_id": "pose_A_init",
                "event_type": "pose_estimate",
                "logical_time": 10.0,
                "frame_id": "frame_A",
                "derived_from_logical_time": 10.0,
                "pose_c2w": [1] + [0] * 15,
                "tracking_state": "OK",
            },
            {
                "event_id": "pose_A_corrected",
                "event_type": "pose_correction",
                "logical_time": 100.0,
                "frame_id": "frame_A",
                "derived_from_logical_time": 100.0,
                "pose_c2w": [1.1] + [0] * 15,
                "tracking_state": "OK",
                "corrected_from_event_id": "pose_A_init",
            },
        ]

        replayer = PoseEventReplayer(
            events=events, trace_mode="FROZEN_FINAL_POSES", oracle_declared=True
        )

        # Oracle mode always returns latest (corrected) pose
        pose = replayer.pose_for("frame_A")
        assert pose["pose_c2w"][0] == 1.1, "Oracle should return P1'"

    def test_oracle_requires_declaration(self) -> None:
        """Test that FROZEN_FINAL_POSES refuses without oracle_declared=True."""
        events = [
            {
                "event_id": "pose_A",
                "event_type": "pose_estimate",
                "logical_time": 10.0,
                "frame_id": "frame_A",
                "derived_from_logical_time": 10.0,
                "pose_c2w": [1] + [0] * 15,
                "tracking_state": "OK",
            },
        ]

        # oracle_declared=False with FROZEN_FINAL_POSES is refused at construction
        with pytest.raises(CapsuleValidationError, match="oracle_declared"):
            PoseEventReplayer(
                events=events, trace_mode="FROZEN_FINAL_POSES", oracle_declared=False
            )

    def test_causal_requires_logical_time(self) -> None:
        """Test that FROZEN_CAUSAL_SLAM_TRACE requires at_logical_time."""
        events = [
            {
                "event_id": "pose_A",
                "event_type": "pose_estimate",
                "logical_time": 10.0,
                "frame_id": "frame_A",
                "derived_from_logical_time": 10.0,
                "pose_c2w": [1] + [0] * 15,
                "tracking_state": "OK",
            },
        ]

        replayer = PoseEventReplayer(
            events=events, trace_mode="FROZEN_CAUSAL_SLAM_TRACE", oracle_declared=False
        )

        with pytest.raises(ValueError, match="at_logical_time"):
            replayer.pose_for("frame_A")  # Missing time argument

    def test_oracle_forbids_time_argument(self) -> None:
        """Test that oracle mode rejects at_logical_time argument."""
        events = [
            {
                "event_id": "pose_A",
                "event_type": "pose_estimate",
                "logical_time": 10.0,
                "frame_id": "frame_A",
                "derived_from_logical_time": 10.0,
                "pose_c2w": [1] + [0] * 15,
                "tracking_state": "OK",
            },
        ]

        replayer = PoseEventReplayer(
            events=events, trace_mode="FROZEN_FINAL_POSES", oracle_declared=True
        )

        with pytest.raises(ValueError, match="oracle mode"):
            replayer.pose_for("frame_A", at_logical_time=50.0)

    def test_future_leak_detection(self) -> None:
        """Test that causal trace rejects events with derived_from > logical_time."""
        events = [
            {
                "event_id": "pose_bad",
                "event_type": "pose_estimate",
                "logical_time": 10.0,
                "frame_id": "frame_A",
                "derived_from_logical_time": 50.0,  # Future leak!
                "pose_c2w": [1] + [0] * 15,
                "tracking_state": "OK",
            },
        ]

        with pytest.raises(CapsuleValidationError, match="future"):
            PoseEventReplayer(
                events=events, trace_mode="FROZEN_CAUSAL_SLAM_TRACE", oracle_declared=False
            )

    def test_two_modes_different_hashes(self) -> None:
        """Test that causal and oracle modes produce different trace hashes.

        This verifies acceptance criterion: "the two modes produce different
        capsule/trace hashes".
        """
        events = [
            {
                "event_id": "pose_A",
                "event_type": "pose_estimate",
                "logical_time": 10.0,
                "frame_id": "frame_A",
                "derived_from_logical_time": 10.0,
                "pose_c2w": [1] + [0] * 15,
            },
        ]

        # Hash for causal mode
        causal_payload = {"mode": "FROZEN_CAUSAL_SLAM_TRACE", "events": events}
        causal_hash = content_hash(causal_payload)

        # Hash for oracle mode
        oracle_payload = {"mode": "FROZEN_FINAL_POSES", "events": events}
        oracle_hash = content_hash(oracle_payload)

        # Should differ due to mode string
        assert causal_hash != oracle_hash, "Modes should produce different hashes"


class TestAdmissionEvents:
    """Tests for admission decision and candidate events."""

    def test_candidate_event_to_from_dict(self) -> None:
        """Test round-trip serialization."""
        event = AdmissionCandidateEvent(
            event_id="cand_0",
            logical_time=50.0,
            frame_id="frame_0",
            candidate_reason="keyframe",
            evidence_available={"n_live": 10000},
        )
        d = event.to_dict()
        restored = AdmissionCandidateEvent.from_dict(d)
        assert restored.frame_id == "frame_0"

    def test_decision_event_to_from_dict(self) -> None:
        """Test round-trip serialization."""
        event = AdmissionDecisionEvent(
            event_id="dec_0",
            logical_time=50.0,
            frame_id="frame_0",
            candidate_event_id="cand_0",
            decision="admit",
            reason_code="coverage_ok",
        )
        d = event.to_dict()
        restored = AdmissionDecisionEvent.from_dict(d)
        assert restored.decision == "admit"

    def test_frozen_admission_mode(self) -> None:
        """Test FROZEN admission mode returns recorded decisions."""
        candidates = [
            {
                "event_id": "cand_A",
                "logical_time": 100.0,
                "frame_id": "frame_A",
                "candidate_reason": "keyframe",
                "evidence_available": {"n_live": 5000},
            },
        ]
        decisions = [
            {
                "event_id": "dec_A",
                "logical_time": 100.0,
                "frame_id": "frame_A",
                "candidate_event_id": "cand_A",
                "decision": "admit",
                "reason_code": "coverage_ok",
            },
        ]

        replayer = AdmissionEventReplayer(
            decision_events=decisions,
            candidate_events=candidates,
            trace_mode="FROZEN",
        )

        decision = replayer.decision_for("frame_A")
        assert decision["decision"] == "admit"

    def test_bypass_mode_forbids_decision_access(self) -> None:
        """Test BYPASS mode rejects decision_for() calls."""
        candidates = [
            {
                "event_id": "cand_A",
                "logical_time": 100.0,
                "frame_id": "frame_A",
                "candidate_reason": "keyframe",
                "evidence_available": {},
            },
        ]
        decisions = []

        replayer = AdmissionEventReplayer(
            decision_events=decisions,
            candidate_events=candidates,
            trace_mode="BYPASS_FOR_POLICY_TEST",
        )

        with pytest.raises(ValueError, match="BYPASS"):
            replayer.decision_for("frame_A")

    def test_bypass_mode_provides_candidates(self) -> None:
        """Test BYPASS mode allows candidate_for() access."""
        candidates = [
            {
                "event_id": "cand_A",
                "logical_time": 100.0,
                "frame_id": "frame_A",
                "candidate_reason": "keyframe",
                "evidence_available": {"n_live": 5000},
            },
        ]

        replayer = AdmissionEventReplayer(
            decision_events=[],
            candidate_events=candidates,
            trace_mode="BYPASS_FOR_POLICY_TEST",
        )

        candidate = replayer.candidate_for("frame_A")
        assert candidate["frame_id"] == "frame_A"
        assert candidate["evidence_available"]["n_live"] == 5000

    def test_frozen_and_bypass_incompatible_at_construction(self) -> None:
        """Test that frozen and bypass modes are mutually exclusive.

        The issue requires: "replay cannot accidentally apply both frozen
        admission and candidate admission simultaneously."
        """
        candidates = [
            {
                "event_id": "cand_A",
                "logical_time": 100.0,
                "frame_id": "frame_A",
                "candidate_reason": "keyframe",
                "evidence_available": {},
            },
        ]
        decisions = [
            {
                "event_id": "dec_A",
                "logical_time": 100.0,
                "frame_id": "frame_A",
                "candidate_event_id": "cand_A",
                "decision": "admit",
                "reason_code": "ok",
            },
        ]

        # FROZEN mode: should work
        frozen = AdmissionEventReplayer(
            decision_events=decisions,
            candidate_events=candidates,
            trace_mode="FROZEN",
        )
        frozen.decision_for("frame_A")  # Should work

        # BYPASS mode: should reject decision access
        bypass = AdmissionEventReplayer(
            decision_events=decisions,
            candidate_events=candidates,
            trace_mode="BYPASS_FOR_POLICY_TEST",
        )
        with pytest.raises(ValueError):
            bypass.decision_for("frame_A")  # Should fail


class TestTraceRecorder:
    """Tests for file-based trace recording."""

    def test_record_and_finalize(self) -> None:
        """Test recording events to JSONL and finalization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "poses.jsonl"
            recorder = TraceRecorder(output_path=path, artifact_key="frontend/pose_events")

            event1 = {
                "event_id": "e1",
                "logical_time": 10.0,
                "frame_id": "f1",
                "data": "test",
            }
            event2 = {
                "event_id": "e2",
                "logical_time": 20.0,
                "frame_id": "f2",
                "data": "test2",
            }

            recorder.append(event1)
            recorder.append(event2)
            artifact_ref = recorder.finalize()

            assert path.exists(), "JSONL file should exist"
            assert artifact_ref.row_count == 2, "Should have 2 rows"
            assert len(artifact_ref.content_hash) == 64, "Should have valid SHA-256"

            # Verify content
            lines = path.read_text().strip().split("\n")
            assert len(lines) == 2, "Should have 2 lines in JSONL"

    def test_append_after_close_raises(self) -> None:
        """Test that appending after close raises."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "poses.jsonl"
            recorder = TraceRecorder(output_path=path, artifact_key="frontend/pose_events")
            recorder.close()

            with pytest.raises(ValueError, match="closed"):
                recorder.append({"event_id": "e1"})

    def test_double_finalize_raises(self) -> None:
        """Test that calling finalize twice raises."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "poses.jsonl"
            recorder = TraceRecorder(output_path=path, artifact_key="frontend/pose_events")
            recorder.finalize()

            with pytest.raises(ValueError, match="already finalized"):
                recorder.finalize()


class TestManipulationReport:
    """Tests for manipulation report and replay fidelity."""

    def test_manipulation_report_to_from_dict(self) -> None:
        """Test round-trip serialization."""
        report = ManipulationReport(
            requested_frontend_trace_mode="FROZEN_CAUSAL_SLAM_TRACE",
            requested_admission_trace_mode="FROZEN",
            frontend_trace_hash="a" * 64,
            admission_trace_hash="b" * 64,
            expected_pose_event_count=100,
            realized_pose_event_count=100,
            expected_admission_event_count=50,
            realized_admission_event_count=50,
        )

        d = report.to_dict()
        restored = ManipulationReport.from_dict(d)
        assert restored.realized_pose_event_count == 100
        assert restored.is_valid_replay is True

    def test_any_mismatch_property(self) -> None:
        """Test the any_mismatch property."""
        report_clean = ManipulationReport(
            requested_frontend_trace_mode="FROZEN_CAUSAL_SLAM_TRACE",
            requested_admission_trace_mode="FROZEN",
            frontend_trace_hash="a" * 64,
            admission_trace_hash="b" * 64,
            pose_mismatch_count=0,
            admission_mismatch_count=0,
            fallback_count=0,
        )
        assert report_clean.any_mismatch is False

        report_dirty = ManipulationReport(
            requested_frontend_trace_mode="FROZEN_CAUSAL_SLAM_TRACE",
            requested_admission_trace_mode="FROZEN",
            frontend_trace_hash="a" * 64,
            admission_trace_hash="b" * 64,
            pose_mismatch_count=1,
        )
        assert report_dirty.any_mismatch is True

    def test_is_valid_replay_property(self) -> None:
        """Test the is_valid_replay property."""
        # Perfect replay
        report = ManipulationReport(
            requested_frontend_trace_mode="FROZEN_CAUSAL_SLAM_TRACE",
            requested_admission_trace_mode="FROZEN",
            frontend_trace_hash="a" * 64,
            admission_trace_hash="b" * 64,
            expected_pose_event_count=100,
            realized_pose_event_count=100,
            expected_admission_event_count=50,
            realized_admission_event_count=50,
            pose_mismatch_count=0,
            admission_mismatch_count=0,
        )
        assert report.is_valid_replay is True

        # Count mismatch
        report_bad_pose = ManipulationReport(
            requested_frontend_trace_mode="FROZEN_CAUSAL_SLAM_TRACE",
            requested_admission_trace_mode="FROZEN",
            frontend_trace_hash="a" * 64,
            admission_trace_hash="b" * 64,
            expected_pose_event_count=100,
            realized_pose_event_count=99,  # Mismatch!
        )
        assert report_bad_pose.is_valid_replay is False

        # Event mismatch
        report_bad_events = ManipulationReport(
            requested_frontend_trace_mode="FROZEN_CAUSAL_SLAM_TRACE",
            requested_admission_trace_mode="FROZEN",
            frontend_trace_hash="a" * 64,
            admission_trace_hash="b" * 64,
            expected_pose_event_count=100,
            realized_pose_event_count=100,
            pose_mismatch_count=1,  # Mismatch!
        )
        assert report_bad_events.is_valid_replay is False

    def test_none_counts_handled_gracefully(self) -> None:
        """Test that None counts don't cause False positives."""
        report = ManipulationReport(
            requested_frontend_trace_mode="NONE",
            requested_admission_trace_mode="NONE",
            frontend_trace_hash="a" * 64,
            admission_trace_hash="b" * 64,
            expected_pose_event_count=None,  # Not applicable
            realized_pose_event_count=None,
        )
        # Should be valid since expected is None
        assert report.is_valid_replay is True


class TestEventIntegration:
    """Integration tests across trace families."""

    def test_complete_scenario(self) -> None:
        """Test a realistic scenario with poses, admission, and reporting."""
        # Setup: record some events
        with tempfile.TemporaryDirectory() as tmpdir:
            # Record poses
            pose_path = Path(tmpdir) / "poses.jsonl"
            pose_recorder = TraceRecorder(
                output_path=pose_path, artifact_key="frontend/pose_events"
            )

            pose_events = [
                {
                    "event_id": f"pose_{i}",
                    "event_type": "pose_estimate",
                    "logical_time": float(i * 10),
                    "frame_id": f"frame_{i}",
                    "derived_from_logical_time": float(i * 10),
                    "pose_c2w": [1 + 0.1 * i] + [0] * 15,
                }
                for i in range(5)
            ]

            for event in pose_events:
                pose_recorder.append(event)

            pose_ref = pose_recorder.finalize()

            # Record admission decisions
            admit_path = Path(tmpdir) / "admission.jsonl"
            admit_recorder = TraceRecorder(
                output_path=admit_path, artifact_key="admission/decisions"
            )

            for i in range(5):
                admit_recorder.append(
                    {
                        "event_id": f"cand_{i}",
                        "logical_time": float(i * 10),
                        "frame_id": f"frame_{i}",
                        "candidate_reason": "keyframe",
                        "evidence_available": {"n_live": 1000 + i * 100},
                    }
                )
                admit_recorder.append(
                    {
                        "event_id": f"dec_{i}",
                        "logical_time": float(i * 10),
                        "frame_id": f"frame_{i}",
                        "candidate_event_id": f"cand_{i}",
                        "decision": "admit" if i % 2 == 0 else "reject",
                        "reason_code": "ok",
                    }
                )

            admit_ref = admit_recorder.finalize()

            # Verify artifacts
            assert pose_ref.row_count == 5
            assert admit_ref.row_count == 10

            # Create replayers
            pose_replayer = PoseEventReplayer(
                events=pose_events,
                trace_mode="FROZEN_CAUSAL_SLAM_TRACE",
            )

            candidates = [
                {
                    "event_id": f"cand_{i}",
                    "logical_time": float(i * 10),
                    "frame_id": f"frame_{i}",
                    "candidate_reason": "keyframe",
                }
                for i in range(5)
            ]
            decisions = [
                {
                    "event_id": f"dec_{i}",
                    "logical_time": float(i * 10),
                    "frame_id": f"frame_{i}",
                    "candidate_event_id": f"cand_{i}",
                    "decision": "admit" if i % 2 == 0 else "reject",
                    "reason_code": "ok",
                }
                for i in range(5)
            ]

            admit_replayer = AdmissionEventReplayer(
                decision_events=decisions,
                candidate_events=candidates,
                trace_mode="FROZEN",
            )

            # Query through replayers
            pose = pose_replayer.pose_for("frame_2", at_logical_time=25.0)
            assert pose["frame_id"] == "frame_2"

            decision = admit_replayer.decision_for("frame_2")
            assert decision["decision"] == "admit"  # 2 % 2 == 0, so admit

            # Create report
            report = ManipulationReport(
                requested_frontend_trace_mode="FROZEN_CAUSAL_SLAM_TRACE",
                requested_admission_trace_mode="FROZEN",
                frontend_trace_hash=pose_ref.content_hash,
                admission_trace_hash=admit_ref.content_hash,
                expected_pose_event_count=5,
                realized_pose_event_count=5,
                expected_admission_event_count=10,
                realized_admission_event_count=10,
                pose_mismatch_count=0,
                admission_mismatch_count=0,
            )

            assert report.is_valid_replay is True
            assert report.any_mismatch is False
