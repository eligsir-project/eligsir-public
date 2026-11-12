"""Replay qualification and canonical MappingApplication cadence tests (#1075/#1162)."""

from __future__ import annotations

import numpy as np

from eligsir.io.sources.decode_cache import StreamingRGBDFrame
from eligsir.runtime.replay import (
    packet_from_streaming_frame,
    packets_from_frame_source,
    run_replay_session,
)
from eligsir.tracking.providers import ExternalTrackingProvider


def _frame(index: int, *, pose: bool = True, source_ordinal=None) -> StreamingRGBDFrame:
    return StreamingRGBDFrame(
        index=index,
        timestamp=1.0 + index,
        rgb_path="",
        c2w=np.eye(4, dtype=np.float32) if pose else None,
        fx=500.0,
        fy=500.0,
        cx=32.0,
        cy=24.0,
        width=64,
        height=48,
        depth_scale=1000.0,
        _rgb_np=np.zeros((48, 64, 3), dtype=np.uint8),
        _depth_np=np.full((48, 64), 1000, dtype=np.uint16),
        source_ordinal=source_ordinal,
    )


def test_packet_carries_intrinsics_and_pose_from_the_frame():
    frame = _frame(2)
    packet = packet_from_streaming_frame(
        frame,
        source_id="cam",
        rgb=frame._rgb_np,
        depth=np.ones((48, 64), dtype=np.float32),
    )
    assert packet.intrinsics.width == 64
    assert packet.source.source_ordinal == 2
    assert packet.tracking is not None
    assert packet.tracking.source == packet.source


def test_packet_without_a_pose_carries_no_tracking_observation():
    frame = _frame(0, pose=False)
    packet = packet_from_streaming_frame(
        frame,
        source_id="cam",
        rgb=frame._rgb_np,
        depth=np.ones((48, 64), dtype=np.float32),
    )
    assert packet.tracking is None


def test_source_ordinal_is_preferred_over_drop_sensitive_index():
    frame = _frame(3, source_ordinal=17)
    packet = packet_from_streaming_frame(
        frame,
        source_id="cam",
        rgb=frame._rgb_np,
        depth=np.ones((48, 64), dtype=np.float32),
    )
    assert packet.source.source_ordinal == 17


def test_frame_source_decodes_real_depth_to_metric_float32():
    packet = next(packets_from_frame_source([_frame(0)], source_id="cam"))
    assert packet.depth is not None
    assert packet.depth.spec.units == "m"
    assert packet.depth.data.dtype == np.float32
    assert np.all(packet.depth.data == np.float32(1.0))


def test_ingress_only_replay_accepts_without_claiming_mapping():
    frames = [_frame(i) for i in range(5)]
    summary = run_replay_session(
        packets_from_frame_source(frames, source_id="cam"),
        ExternalTrackingProvider(),
        source_id="cam",
    )
    assert summary.submitted == 5
    assert summary.accepted == 5
    assert summary.mapped == 0
    assert summary.rejected == 0
    assert summary.action_counts["deliver"] == 5


def test_ingress_only_replay_rejects_untracked_frames_and_attributes_them():
    frames = [_frame(0), _frame(1, pose=False), _frame(2)]
    summary = run_replay_session(
        packets_from_frame_source(frames, source_id="cam"),
        ExternalTrackingProvider(),
    )
    assert summary.accepted == 2
    assert summary.mapped == 0
    assert summary.rejected == 1
    assert summary.action_counts["reject"] == 1


def test_max_frames_truncates_the_qualification_pass():
    frames = [_frame(i) for i in range(10)]
    summary = run_replay_session(
        packets_from_frame_source(frames, source_id="cam", max_frames=4),
        ExternalTrackingProvider(),
    )
    assert summary.submitted == 4
    assert summary.accepted == 4
    assert summary.mapped == 0


def test_ingress_qualification_is_deterministic_across_passes():
    frames = [_frame(i, pose=(i % 3 != 0)) for i in range(12)]

    def run():
        return run_replay_session(
            packets_from_frame_source(frames, source_id="cam"),
            ExternalTrackingProvider(),
        ).as_dict()

    assert run() == run()


def test_cli_reports_an_unknown_pose_source_without_traceback(capsys):
    from eligsir.cli.session import main

    assert main(["--source", "/nonexistent", "--pose-source", "ros_slam"]) == 2
    assert "eligsir-ros2" in capsys.readouterr().err
