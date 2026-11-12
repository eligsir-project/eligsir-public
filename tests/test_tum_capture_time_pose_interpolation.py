"""Offline TUM capture-time pose interpolation stays outside holdout scoring."""

from copy import copy
from types import SimpleNamespace

import numpy as np
from PIL import Image

from eligsir.io.sources.datasets.tum import TUMFrameSource
from eligsir.runtime.causal_evaluation import (
    build_causal_holdout_cameras,
    causal_holdout_plan,
    enable_training_capture_pose_candidates,
)
from eligsir.runtime.causal_replay import StreamingFramePacketAdapter
from eligsir.runtime.replay import streaming_frame_source_identity


def _write_tum(path, *, poses, depth_timestamp=0.010):
    (path / "rgb").mkdir(exist_ok=True)
    (path / "depth").mkdir(exist_ok=True)
    Image.fromarray(np.zeros((2, 2, 3), dtype=np.uint8)).save(path / "rgb" / "0.png")
    Image.fromarray(np.ones((2, 2), dtype=np.uint16)).save(path / "depth" / "0.png")
    (path / "rgb.txt").write_text("0.010 rgb/0.png\n")
    (path / "depth.txt").write_text(f"{depth_timestamp:.3f} depth/0.png\n")
    (path / "groundtruth.txt").write_text(
        "\n".join(
            f"{timestamp:.3f} {tx} 0 0 {qx} {qy} {qz} {qw}"
            for timestamp, tx, qx, qy, qz, qw in poses
        )
        + "\n"
    )


def _eval_args():
    return SimpleNamespace(
        streaming_eval_hold=1,
        streaming_initial_frames=0,
        streaming_eval_guard=0,
        streaming_eval_min_baseline_m=0.0,
        streaming_eval_min_angle_deg=0.0,
        source_path="/offline/tum",
    )


def test_capture_time_interpolation_changes_training_packet_not_source_or_holdout_pose(tmp_path):
    _write_tum(
        tmp_path,
        poses=((0.0, 0.0, 0.0, 0.0, 0.0, 1.0), (0.02, 2.0, 0.0, 0.0, 1.0, 0.0)),
    )

    control = TUMFrameSource(str(tmp_path))
    treatment = TUMFrameSource(str(tmp_path), capture_time_pose_interpolation=True)
    control_frame = control.get_all()[0]
    treatment_frame = treatment.get_all()[0]

    assert not hasattr(control_frame, "_mapping_c2w")
    assert control_frame.color_ts_ns == treatment_frame.color_ts_ns == 10_000_000
    assert control_frame.depth_ts_ns == treatment_frame.depth_ts_ns == 10_000_000
    assert control_frame.pose_ts_ns == treatment_frame.pose_ts_ns == 0
    assert control_frame.rgb_depth_dt_ns == treatment_frame.rgb_depth_dt_ns == 0
    assert control_frame.rgb_pose_dt_ns == treatment_frame.rgb_pose_dt_ns == 10_000_000
    assert np.allclose(control_frame.c2w, treatment_frame.c2w)
    assert streaming_frame_source_identity(control_frame, source_id="tum") == streaming_frame_source_identity(
        treatment_frame, source_id="tum"
    )
    control_plan = causal_holdout_plan(control.get_all(), _eval_args())
    treatment_plan = causal_holdout_plan(treatment.get_all(), _eval_args())
    assert control_plan.holdout_ordinals == treatment_plan.holdout_ordinals
    control_camera = build_causal_holdout_cameras(control_plan, data_device="cpu")[0]
    treatment_camera = build_causal_holdout_cameras(treatment_plan, data_device="cpu")[0]
    assert np.allclose(control_camera.R, treatment_camera.R)
    assert np.allclose(control_camera.T, treatment_camera.T)


def test_offline_tum_records_immutable_capture_to_capture_motion_facts(tmp_path):
    _write_tum(
        tmp_path,
        poses=((0.0, 0.0, 0.0, 0.0, 0.0, 1.0), (0.02, 2.0, 0.0, 0.0, 0.0, 1.0)),
        depth_timestamp=0.005,
    )

    frame = TUMFrameSource(
        str(tmp_path),
        motion_depth_confidence_mode="motion",
    ).get_all()[0]
    packet = StreamingFramePacketAdapter("tum").packet_from_record(frame)

    assert packet.tracking.diagnostics["motion_depth_confidence_source"] == "offline_tum"
    assert packet.tracking.diagnostics["motion_depth_confidence_reason"] == "interpolated"
    transform = np.asarray(packet.tracking.diagnostics["motion_depth_confidence_transform"])
    assert transform.shape == (4, 4)
    assert not transform.flags.writeable
    assert np.allclose(transform[:3, 3], (-0.5, 0.0, 0.0))
    assert np.allclose(packet.tracking.pose.matrix, frame.c2w)


def test_capture_time_pose_reaches_only_training_packets_after_causal_role_selection(tmp_path):
    _write_tum(
        tmp_path,
        poses=((0.0, 0.0, 0.0, 0.0, 0.0, 1.0), (0.02, 2.0, 0.0, 0.0, 0.0, 1.0)),
    )
    source_frame = TUMFrameSource(
        str(tmp_path), capture_time_pose_interpolation=True
    ).get_all()[0]
    frames = tuple(copy(source_frame) for _ in range(3))
    for index, frame in enumerate(frames, start=1):
        frame.index = index
    args = SimpleNamespace(
        streaming_eval_hold=3,
        streaming_initial_frames=0,
        streaming_eval_guard=1,
        streaming_eval_min_baseline_m=0.0,
        streaming_eval_min_angle_deg=0.0,
        source_path="/offline/tum",
    )
    plan = causal_holdout_plan(frames, args)

    enable_training_capture_pose_candidates(frames, plan)
    packets = tuple(
        StreamingFramePacketAdapter("tum").packet_from_record(frame) for frame in frames
    )

    assert tuple(plan.role_for_packet(packet) for packet in packets) == (
        "train", "guard", "holdout"
    )
    assert np.allclose(packets[0].tracking.pose.matrix[:3, 3], (1.0, 0.0, 0.0))
    assert packets[0].tracking.diagnostics["tum_pose_time_method"] == "slerp"
    for frame, packet in zip(frames[1:], packets[1:], strict=True):
        assert np.allclose(packet.tracking.pose.matrix, frame.c2w)
        assert packet.tracking.diagnostics == {}
        assert np.allclose(frame.c2w[:3, 3], (0.0, 0.0, 0.0))


def test_capture_time_interpolation_keeps_nearest_pose_on_duplicate_or_too_wide_bracket(tmp_path):
    _write_tum(
        tmp_path,
        poses=(
            (0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
            (0.0, 4.0, 0.0, 0.0, 0.0, 1.0),
            (0.02, 2.0, 0.0, 0.0, 0.0, 1.0),
        ),
    )

    frame = TUMFrameSource(str(tmp_path), capture_time_pose_interpolation=True).get_all()[0]
    frame._mapping_pose_for_training = True
    packet = StreamingFramePacketAdapter("tum").packet_from_record(frame)

    assert np.allclose(packet.tracking.pose.matrix, frame.c2w)
    assert packet.tracking.diagnostics["tum_pose_time_method"] == "nearest"
    assert packet.tracking.diagnostics["tum_pose_time_reason"] == "duplicate_bracket"


def test_capture_time_interpolation_keeps_nearest_pose_for_wide_or_nonfinite_bracket(tmp_path):
    _write_tum(
        tmp_path,
        poses=((0.0, 0.0, 0.0, 0.0, 0.0, 1.0), (0.08, 8.0, 0.0, 0.0, 0.0, 1.0)),
    )
    wide_frame = TUMFrameSource(str(tmp_path), capture_time_pose_interpolation=True).get_all()[0]
    wide_frame._mapping_pose_for_training = True
    wide_packet = StreamingFramePacketAdapter("tum").packet_from_record(wide_frame)

    assert np.allclose(wide_packet.tracking.pose.matrix, wide_frame.c2w)
    assert wide_packet.tracking.diagnostics["tum_pose_time_reason"] == "bracket_gap_exceeded"

    _write_tum(
        tmp_path,
        poses=((0.0, 0.0, 0.0, 0.0, 0.0, 1.0), (0.02, 2.0, 0.0, float("nan"), 0.0, 1.0)),
    )
    nonfinite_frame = TUMFrameSource(str(tmp_path), capture_time_pose_interpolation=True).get_all()[0]
    nonfinite_frame._mapping_pose_for_training = True
    nonfinite_packet = StreamingFramePacketAdapter("tum").packet_from_record(nonfinite_frame)

    assert np.allclose(nonfinite_packet.tracking.pose.matrix, nonfinite_frame.c2w)
    assert nonfinite_packet.tracking.diagnostics["tum_pose_time_reason"] == "nonfinite_pose"


def test_capture_time_interpolation_uses_shortest_quaternion_arc(tmp_path):
    _write_tum(
        tmp_path,
        poses=((0.0, 0.0, 0.0, 0.0, 0.0, 1.0), (0.02, 0.0, 0.0, 0.0, -0.70710678, -0.70710678)),
    )

    frame = TUMFrameSource(str(tmp_path), capture_time_pose_interpolation=True).get_all()[0]
    frame._mapping_pose_for_training = True
    rotation = StreamingFramePacketAdapter("tum").packet_from_record(frame).tracking.pose.matrix[:3, :3]

    assert np.allclose(
        rotation @ np.array((1.0, 0.0, 0.0)),
        (np.sqrt(0.5), np.sqrt(0.5), 0.0),
        atol=1e-6,
    )


def test_capture_time_interpolation_is_opt_in_through_tum_source_registry(tmp_path):
    _write_tum(
        tmp_path,
        poses=((0.0, 0.0, 0.0, 0.0, 0.0, 1.0), (0.02, 2.0, 0.0, 0.0, 0.0, 1.0)),
    )
    from eligsir.io.sources.registry import make_frame_source

    args = SimpleNamespace(
        streaming_max_frames=0,
        tum_frame_stride=1,
        tum_association_max_dt=0.03,
        tum_sequence="",
        tum_capture_time_pose_interpolation=True,
    )
    frame = make_frame_source(str(tmp_path), args).get_all()[0]

    assert hasattr(frame, "_mapping_c2w")
    assert np.allclose(frame.c2w[:3, 3], (0.0, 0.0, 0.0))


def test_ready_capture_and_cold_replay_metadata_keep_corrected_training_pose(tmp_path, monkeypatch):
    _write_tum(
        tmp_path,
        poses=((0.0, 0.0, 0.0, 0.0, 0.0, 1.0), (0.02, 2.0, 0.0, 0.0, 0.0, 1.0)),
    )
    (tmp_path / "config.yaml").write_text(
        "H: 2\nW: 2\nfx: 1.0\nfy: 1.0\ncx: 0.5\ncy: 0.5\ndepth_scale: 1000.0\n"
    )
    from eligsir.contracts import ModalityEligibility
    from eligsir.runtime.prepared_source_producer import PreparedSourceProducer
    from eligsir.runtime.ready_source import ReadySourceOwner
    from eligsir.runtime.transfers import DeviceTransferOwner

    frame = TUMFrameSource(str(tmp_path), capture_time_pose_interpolation=True).get_all()[0]
    frame._mapping_pose_for_training = True
    packet = StreamingFramePacketAdapter("tum").packet_from_record(frame)
    owner = DeviceTransferOwner("cpu", max_inflight=1, max_pinned_bytes=4096)
    captured = []
    upload = owner.upload_packed

    def capture(arrays, **kwargs):
        captured.append(np.asarray(arrays[2]))
        return upload(arrays, **kwargs)

    monkeypatch.setattr(owner, "upload_packed", capture)
    ready = ReadySourceOwner(max_fresh=1, max_replay=1)
    producer = PreparedSourceProducer(ready, transfer_owner=owner, producer_owner="test")
    producer.submit(
        packet,
        modality_eligibility=ModalityEligibility(rgb_supervision=True, depth_supervision=True),
    )
    lease = ready.claim_consumer("test").poll()

    expected = np.eye(4, dtype=np.float32)
    expected[0, 3] = 1.0
    assert lease is not None and lease.source_identity == packet.source
    assert len(captured) == 1 and np.allclose(captured[0], expected)
    assert lease.source_metadata["pose_c2w"][0][3] == 1.0

    from eligsir.runtime.production_application import CanonicalMappingRuntime

    cold = SimpleNamespace(
        rgb=np.zeros((2, 2, 3), dtype=np.uint8),
        depth=np.ones((2, 2), dtype=np.float32),
        intrinsics=packet.intrinsics,
    )
    runtime = object.__new__(CanonicalMappingRuntime)
    runtime._observation_db = SimpleNamespace(
        frame_id_for_source=lambda source: 3 if source == packet.source else None,
        get_cold_pixels=lambda frame_id: cold if frame_id == 3 else None,
        get_metadata=lambda frame_id: SimpleNamespace(
            pose_c2w=lease.source_metadata["pose_c2w"]
        ) if frame_id == 3 else None,
    )
    replayed = runtime._replay_mapping_frame(packet.source)

    assert replayed is not None
    assert np.allclose(replayed.pose.matrix, expected)
