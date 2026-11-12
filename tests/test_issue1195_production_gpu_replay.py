"""Real Orbbec replay qualification for the production MappingApplication.

This is an UNMANAGED development smoke, not scientific evidence.  It is opt-in
because it requires both the project GPU container and a real recorded bag.
"""

from __future__ import annotations

import json
import os
import shutil
from hashlib import sha256
from pathlib import Path

import numpy as np
import pytest


def _synthetic_packet(ordinal: int, *, translation_m: float | None = None):
    from eligsir.contracts.runtime import BufferRef, SensorPacket, SourceIdentity
    from eligsir.contracts.types import CameraIntrinsics
    from eligsir.tracking.providers import observation_from_pose_matrix

    source = SourceIdentity(
        "synthetic_gpu", 0, ordinal, 1_000_000_000 + ordinal * 33_333_333
    )
    pose = np.eye(4, dtype=np.float64)
    pose[0, 3] = ordinal * 0.03 if translation_m is None else translation_m
    h, w = 48, 64
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[..., 0] = np.arange(w, dtype=np.uint8)[None, :]
    rgb[..., 1] = ordinal * 10
    depth = np.full((h, w), 1.0 + ordinal * 0.01, dtype=np.float32)
    return SensorPacket(
        source=source,
        rgb=BufferRef.from_numpy(rgb, encoding="rgb8"),
        depth=BufferRef.from_numpy(depth, encoding="32FC1", units="m"),
        intrinsics=CameraIntrinsics(
            fx=50.0, fy=50.0, cx=w / 2.0, cy=h / 2.0, width=w, height=h
        ),
        tracking=observation_from_pose_matrix(source, pose, provider="synthetic"),
    )


def test_factory_application_zero_frame_replay_advances_on_available_gpu(tmp_path):
    import torch

    if not torch.cuda.is_available():
        pytest.skip("requires a CUDA or ROCm project GPU container")

    from eligsir.runtime.application import create_mapping_application
    from eligsir.runtime.production_application import CanonicalMappingRuntime

    application = create_mapping_application(
        model_path=str(tmp_path / "factory_zero_frame_replay")
    )
    runtime = application._options.mapper_sink.__self__
    assert isinstance(runtime, CanonicalMappingRuntime)
    try:
        first_mapping_tick = None
        for ordinal in range(runtime.bootstrap_frames):
            outcome = application.session.submit(_synthetic_packet(ordinal))
            assert outcome.frame is not None
            tick = application.tick()
            if tick.executed:
                first_mapping_tick = tick

        assert first_mapping_tick is not None
        parameter_before = first_mapping_tick.parameter_version
        idle = application.tick()

        assert idle.executed
        assert idle.optimizer_steps > 0
        assert idle.parameter_version > parameter_before
        assert idle.map_content_version > first_mapping_tick.map_content_version
        assert idle.work_plan is not None
        assert idle.work_plan.supervision_sources == ()
        assert idle.work_plan.replay_sources
        assert "source_idle_replay_only" in idle.work_plan.reason_trace
        assert str(runtime.gaussians.get_xyz.device).startswith("cuda")
    finally:
        application.close()


@pytest.mark.skipif(
    os.environ.get("ELIGSIR_LONG_GPU_MCMC_SMOKE") != "1",
    reason="set ELIGSIR_LONG_GPU_MCMC_SMOKE=1 for the 1000-step GPU gate",
)
def test_replay_only_mcmc_grows_population_by_1000_gpu_steps(tmp_path):
    import torch

    if not torch.cuda.is_available():
        pytest.skip("requires a CUDA or ROCm project GPU container")

    from eligsir.runtime.application import create_mapping_application

    application = create_mapping_application(
        model_path=str(tmp_path / "factory_replay_mcmc_1000"),
        overrides={
            "streaming_lifecycle_enabled": True,
            "streaming_mature_age_frames": 1,
            "streaming_mature_min_utility": 0.0,
            # The 48x64 synthetic cameras expose only 3,072 RGB pixels per
            # update. Lower the production exposure interval for this gate so
            # 1,000 updates must cross the real semantic mutation barrier.
            "mcmc_grow_interval_pixels": 100_000,
            "mcmc_relocate_interval_pixels": 50_000,
        },
    )
    runtime = application._options.mapper_sink.__self__
    try:
        for ordinal in range(runtime.bootstrap_frames):
            application.session.submit(_synthetic_packet(ordinal))
            application.tick()
        population_before = int(runtime.gaussians.get_xyz.shape[0])
        topology_before = int(runtime.backend.topology_version)
        runtime.mark_source_complete()

        last = None
        while runtime.iteration < 1000:
            last = application.tick()

        assert last is not None and last.executed
        assert last.work_plan is not None
        assert last.work_plan.supervision_sources == ()
        assert {work.resolution_tier.value for work in last.work_plan.replay_work} == {
            "native"
        }
        assert int(runtime.gaussians.get_xyz.shape[0]) > population_before
        assert int(runtime.backend.topology_version) > topology_before
        mature_count = int((runtime.gaussians.lifecycle_state == 2).sum().item())
        assert mature_count > 0, {
            "frames_seen": int(runtime.frames_seen),
            "mature_age": int(runtime._optimization_policy.streaming_mature_age_frames),
            "minimum_utility": float(
                runtime._optimization_policy.streaming_mature_min_utility
            ),
            "birth_min": int(runtime.gaussians.birth_iter.min().item()),
            "birth_max": int(runtime.gaussians.birth_iter.max().item()),
            "provisional": int(runtime.gaussians.provisional.sum().item()),
            "young": int((runtime.gaussians.lifecycle_state == 1).sum().item()),
            "transition_calls": int(runtime._lifecycle_transition_calls),
        }
    finally:
        application.close()


@pytest.mark.skipif(
    os.environ.get("ELIGSIR_LONG_GPU_REPLAY_DEPTH_SMOKE") != "1",
    reason="set ELIGSIR_LONG_GPU_REPLAY_DEPTH_SMOKE=1 for the 1000-step GPU gate",
)
def test_replay_depth_refines_novel_cached_view_through_one_mutation_barrier(tmp_path):
    import torch

    if not torch.cuda.is_available():
        pytest.skip("requires a CUDA or ROCm project GPU container")

    from eligsir.runtime.application import create_mapping_application

    application = create_mapping_application(
        model_path=str(tmp_path / "factory_replay_depth_1000"),
        overrides={
            "streaming_max_new_gaussians_per_frame": 0,
            "streaming_replay_depth_max_per_event": 64,
            "streaming_replay_depth_interval": 25,
            "streaming_replay_depth_voxel_size": 0.01,
            "streaming_depth_stride": 4,
            "streaming_lifecycle_enabled": True,
            "streaming_mature_age_frames": 1,
            "streaming_mature_min_utility": 0.0,
        },
    )
    runtime = application._options.mapper_sink.__self__
    try:
        for ordinal in range(runtime.bootstrap_frames):
            application.session.submit(_synthetic_packet(ordinal))
            application.tick()
        novel = _synthetic_packet(runtime.bootstrap_frames, translation_m=2.0)
        application.session.submit(novel)
        application.tick()
        runtime.publish_replay_focus_weights({novel.source: 1000.0})
        runtime.mark_source_complete()
        population_before = int(runtime.gaussians.get_xyz.shape[0])
        replay_depth_plans = 0
        replay_depth_inserted = 0

        while runtime.iteration < 1000:
            tick = application.tick()
            if tick.work_plan is not None and tick.work_plan.replay_insertion is not None:
                replay_depth_plans += 1
                replay_depth_inserted += sum(
                    int(event.inserted)
                    for event in tick.mutation_events
                    if event.kind == "replay_depth"
                )

        assert replay_depth_plans >= 10
        assert replay_depth_inserted > 0
        assert int(runtime.gaussians.get_xyz.shape[0]) > population_before
    finally:
        application.close()


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bag_identity_artifact(bag_path: str) -> Path:
    path = Path(bag_path)
    return path if path.is_file() else path / "metadata.yaml"


def _publish_durable_artifacts(evidence: dict, output_ply: Path) -> None:
    """Copy qualification artifacts out of an ephemeral test container."""
    durable_evidence_path = os.environ.get("ELIGSIR_GPU_REPLAY_EVIDENCE_PATH")
    if not durable_evidence_path:
        return
    destination = Path(durable_evidence_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    durable_ply = destination.with_name("issue1195_gpu_replay_output.ply")
    shutil.copy2(output_ply, durable_ply)
    evidence["durable_output_ply"] = str(durable_ply)
    destination.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")


def test_bag_identity_artifact_supports_directory_and_single_mcap(tmp_path):
    bag_directory = tmp_path / "bag"
    bag_directory.mkdir()
    metadata = bag_directory / "metadata.yaml"
    metadata.write_text("rosbag2_bagfile_information: {}\n")
    mcap = tmp_path / "capture.mcap"
    mcap.write_bytes(b"mcap")

    assert _bag_identity_artifact(str(bag_directory)) == metadata
    assert _bag_identity_artifact(str(mcap)) == mcap


def test_publish_durable_artifacts_copies_ply_and_receipt(tmp_path, monkeypatch):
    output_ply = tmp_path / "output.ply"
    output_ply.write_bytes(b"ply\n")
    receipt = tmp_path / "receipts" / "qualification.json"
    monkeypatch.setenv("ELIGSIR_GPU_REPLAY_EVIDENCE_PATH", str(receipt))

    evidence = {"output_ply": str(output_ply)}
    _publish_durable_artifacts(evidence, output_ply)

    durable_ply = receipt.with_name("issue1195_gpu_replay_output.ply")
    assert durable_ply.read_bytes() == b"ply\n"
    assert evidence["durable_output_ply"] == str(durable_ply)
    assert json.loads(receipt.read_text()) == evidence


def test_factory_application_advances_optimizer_and_map_on_real_orbbec_replay(tmp_path):
    bag_path = os.environ.get("ELIGSIR_ORBBEC_REPLAY_BAG", "")
    if not bag_path:
        pytest.skip("ELIGSIR_ORBBEC_REPLAY_BAG is not set")

    import torch

    if not torch.cuda.is_available() or torch.version.hip is None:
        pytest.skip("requires the project ROCm GPU container")

    from eligsir.io.sources.datasets.orbbec_rosbag import (
        OrbbecRosBagFrameSource,
    )
    from eligsir.runtime.application import create_mapping_application
    from eligsir.runtime.production_application import CanonicalMappingRuntime
    from eligsir.runtime.replay import packets_from_frame_source

    model_path = tmp_path / "factory_orbbec_replay"
    application = create_mapping_application(model_path=str(model_path))
    runtime = application._options.mapper_sink.__self__
    assert isinstance(runtime, CanonicalMappingRuntime)
    gates_disabled = os.environ.get("ELIGSIR_FORCE_GATES_DISABLED") == "1"
    if gates_disabled:
        from eligsir.mapping.scheduler.backpressure import (
            BackpressureAction,
            BackpressureDecision,
        )

        # Qualification-only proposal: supervision/optimizer work remains live,
        # while source insertion and growth/relocation budgets are forced off.
        # The production ObservationOwner still mints and records the terminal
        # disposition; the test does not bypass the owner boundary.
        runtime.observation_owner._proposal = lambda snapshot: BackpressureDecision(
            frame_id=snapshot.source.source_ordinal,
            action=BackpressureAction.ADMIT,
            reason="soft_all_capacity_available",
            admit_supervision=True,
            allow_geometry_insertion=False,
        )

    evidence = {}
    try:
        zero_frame = application.tick()
        assert not zero_frame.executed
        assert zero_frame.reason == "bootstrap_pending"

        post_bootstrap_attempts = int(
            os.environ.get("ELIGSIR_ORBBEC_REPLAY_POST_BOOTSTRAP_FRAMES", "1")
        )
        frame_count = runtime.bootstrap_frames + post_bootstrap_attempts
        pose_source = os.environ.get(
            "ELIGSIR_ORBBEC_REPLAY_POSE_SOURCE", "camera_pose"
        )
        source = OrbbecRosBagFrameSource(
            bag_path,
            pose_topic=os.environ.get("ELIGSIR_ORBBEC_REPLAY_POSE_TOPIC", "/gt/pose"),
            pose_source=pose_source,
            frame_stride=int(os.environ.get("ELIGSIR_ORBBEC_REPLAY_STRIDE", "15")),
            max_frames=frame_count,
            sync_strict=False,
            sync_report_json=str(model_path / "sync_report.json"),
            open3d_odom_cache_dir=os.environ.get(
                "ELIGSIR_ORBBEC_REPLAY_ODOM_CACHE_DIR",
                str(model_path / "odometry_cache"),
            ),
            streaming_resolution=int(
                os.environ.get("ELIGSIR_ORBBEC_REPLAY_RESOLUTION", "8")
            ),
        )
        packets = iter(
            packets_from_frame_source(
                source,
                source_id="orbbec_rosbag_replay",
                provider=pose_source,
                max_frames=frame_count,
            )
        )

        tick_trace = []
        for index in range(runtime.bootstrap_frames - 1):
            packet = next(packets)
            if index == 0:
                assert packet.rgb.spec.encoding == "rgb8"
                assert packet.rgb.spec.dtype == "uint8"
                assert packet.tracking is not None
                assert "overlap" not in packet.tracking.diagnostics
            assert packet.depth is not None
            if index == 0:
                assert packet.depth.spec.encoding == "32FC1"
                assert packet.depth.spec.units == "m"
            assert np.count_nonzero(packet.depth.data) > 0
            outcome = application.session.submit(packet)
            assert outcome.frame is not None
            pending_tick = application.tick()
            assert not pending_tick.executed
            assert pending_tick.snapshot_id is not None
            assert pending_tick.work_plan is not None
            assert pending_tick.work_plan.supervision_sources == (packet.source,)
            assert tuple(item.source for item in pending_tick.dispositions) == (
                packet.source,
            )
            if index == 0:
                disposition = pending_tick.dispositions[0]
                assert disposition.kind.value == "admit"
                assert disposition.modality_eligibility.rgb_supervision
                assert disposition.modality_eligibility.depth_supervision
                assert disposition.modality_eligibility.depth_insertion
            tick_trace.append(
                {
                    "frame": index,
                    "executed": pending_tick.executed,
                    "reason": pending_tick.reason,
                }
            )

        bootstrap_packet = next(packets)
        assert bootstrap_packet.depth is not None
        assert np.count_nonzero(bootstrap_packet.depth.data) > 0
        bootstrap = application.session.submit(bootstrap_packet)
        assert bootstrap.frame is not None
        content_version_before_tick = int(runtime.backend.map_content_version)
        parameter_version_before_tick = runtime._versions()[2]

        first_tick = application.tick()
        assert first_tick.executed
        assert first_tick.optimizer_steps > 0
        assert first_tick.telemetry is not None
        assert first_tick.telemetry.snapshot_id == first_tick.snapshot_id
        assert (
            first_tick.telemetry.map_state.parameter_version
            == first_tick.parameter_version
        )
        assert first_tick.work_plan is not None
        assert first_tick.work_plan.supervision_sources == (bootstrap_packet.source,)
        assert first_tick.transfers is not None
        assert first_tick.map_content_version > content_version_before_tick
        assert first_tick.parameter_version > parameter_version_before_tick
        assert (
            first_tick.parameter_version - parameter_version_before_tick
            == first_tick.optimizer_steps
        )
        content_version_after_tick = int(runtime.backend.map_content_version)
        tick_trace.append(
            {
                "frame": runtime.bootstrap_frames - 1,
                "executed": first_tick.executed,
                "optimizer_steps": first_tick.optimizer_steps,
                "content_version": content_version_after_tick,
            }
        )

        refinement_tick = application.tick()
        assert refinement_tick.executed
        assert refinement_tick.optimizer_steps > 0
        assert refinement_tick.work_plan is not None
        assert refinement_tick.work_plan.supervision_sources == ()
        assert refinement_tick.work_plan.replay_sources
        assert "source_idle_replay_only" in refinement_tick.work_plan.reason_trace
        # This synchronize belongs to the opt-in qualification test, never the
        # production telemetry path. A later zero-frame tick must publish the
        # now-completed event-timed GPU transaction.
        torch.cuda.synchronize()
        completed_telemetry_tick = application.tick()
        assert completed_telemetry_tick.telemetry is not None
        assert completed_telemetry_tick.telemetry.gpu_transaction is not None
        assert completed_telemetry_tick.telemetry.gpu_transaction.duration_ms > 0.0
        tick_trace.append(
            {
                "frame": None,
                "executed": refinement_tick.executed,
                "optimizer_steps": refinement_tick.optimizer_steps,
                "content_version": int(runtime.backend.map_content_version),
            }
        )

        content_version_before_insert = int(runtime.backend.map_content_version)
        gaussian_count_before_insert = int(runtime.gaussians.get_xyz.shape[0])
        topology_version_before_insert = int(runtime.backend.topology_version)
        mutation_events_before_insert = application.session.deterministic_digest.snapshot()[
            "event_counts"
        ].get("mutation_commit", 0)

        second_tick = None
        mapped_post_bootstrap = 0
        for post_index in range(post_bootstrap_attempts):
            inserted = application.session.submit(next(packets))
            if inserted.frame is None:
                rejected_tick = application.tick()
                assert not rejected_tick.executed
                tick_trace.append(
                    {
                        "frame": runtime.bootstrap_frames + post_index,
                        "executed": False,
                        "reason": rejected_tick.reason,
                    }
                )
                continue
            mapped_post_bootstrap += 1
            second_tick = application.tick()
            assert second_tick.work_plan is not None
            assert second_tick.work_plan.supervision_sources == (inserted.frame.source,)
            assert second_tick.transfers is not None
            assert second_tick.transfers.h2d_ops > 0
            assert second_tick.transfers.h2d_bytes > 0
            assert second_tick.telemetry is not None
            assert second_tick.telemetry.residency.hot_gpu <= min(
                runtime._runtime_policy.max_active_supervision_frames,
                runtime._runtime_policy.keyframe_window,
            )
            tick_trace.append(
                {
                    "frame": runtime.bootstrap_frames + post_index,
                    "executed": second_tick.executed,
                    "optimizer_steps": second_tick.optimizer_steps,
                    "content_version": int(runtime.backend.map_content_version),
                    "topology_version": int(runtime.backend.topology_version),
                }
            )
        assert second_tick is not None, "replay produced no post-bootstrap mapping work"
        torch.cuda.synchronize()
        completed_telemetry_tick = application.tick()
        assert completed_telemetry_tick.telemetry.gpu_transaction is not None
        bag_identity_path = _bag_identity_artifact(bag_path)
        evidence = {
            "evidence_class": "UNMANAGED",
            "source": "orbbec_rosbag_replay",
            "core_commit": os.environ.get("ELIGSIR_GIT_COMMIT", "unknown"),
            "core_branch": os.environ.get("ELIGSIR_GIT_BRANCH", "unknown"),
            "core_dirty": os.environ.get("ELIGSIR_GIT_DIRTY", "unknown"),
            "bag_path": bag_path,
            "bag_identity_artifact": str(bag_identity_path),
            "bag_identity_sha256": _sha256(bag_identity_path),
            "submitted": application.session.submitted_count,
            "accepted": application.session.accepted_count,
            "mapped": application.session.mapped_count,
            "rejected": application.session.rejected_count,
            "zero_frame_reason": zero_frame.reason,
            "optimizer_steps": runtime.optimizer_steps,
            "last_tick_optimizer_steps": second_tick.optimizer_steps,
            "last_tick_snapshot_id": second_tick.snapshot_id,
            "last_tick_plan_id": second_tick.work_plan.plan_id,
            "last_tick_dispositions": [
                item.kind.value for item in second_tick.dispositions
            ],
            "transfer_accounting": {
                "h2d_ops": second_tick.transfers.h2d_ops,
                "h2d_bytes": second_tick.transfers.h2d_bytes,
                "d2h_ops": second_tick.transfers.d2h_ops,
                "d2h_bytes": second_tick.transfers.d2h_bytes,
                "peak_pinned_bytes": second_tick.transfers.peak_pinned_bytes,
            },
            "parameter_version_after_first_tick": first_tick.parameter_version,
            "content_version_before_tick": content_version_before_tick,
            "content_version_after_tick": content_version_after_tick,
            "content_version_before_insert": content_version_before_insert,
            "content_version_after_insert": int(runtime.backend.map_content_version),
            "topology_version_after_insert": int(runtime.backend.topology_version),
            "gaussians_before_insert": gaussian_count_before_insert,
            "gaussians_after_insert": int(runtime.gaussians.get_xyz.shape[0]),
            "gates_disabled": gates_disabled,
            "mutation_events_before_insert": mutation_events_before_insert,
            "tick_trace": tick_trace,
            "deterministic_replay": application.session.deterministic_digest.snapshot(),
            "device": str(runtime.gaussians.get_xyz.device),
            "torch": str(torch.__version__),
            "rocm": str(torch.version.hip),
            "gpu_name": torch.cuda.get_device_name(0),
        }
        assert second_tick.executed
        assert second_tick.optimizer_steps > 0
        assert evidence["submitted"] == evidence["accepted"] == frame_count
        assert evidence["mapped"] == runtime.bootstrap_frames + mapped_post_bootstrap
        assert first_tick.parameter_version > 0
        assert content_version_after_tick > content_version_before_tick
        assert evidence["content_version_after_insert"] > content_version_before_insert
        assert evidence["device"].startswith("cuda")
        if os.environ.get("ELIGSIR_REQUIRE_MUTATION") == "1":
            assert evidence["gaussians_after_insert"] > gaussian_count_before_insert
            assert evidence["topology_version_after_insert"] > 0
            assert (
                evidence["deterministic_replay"]["event_counts"]["mutation_commit"]
                > 0
            )
        if gates_disabled:
            assert evidence["gaussians_after_insert"] == gaussian_count_before_insert
            assert evidence["topology_version_after_insert"] == topology_version_before_insert
            assert (
                evidence["deterministic_replay"]["event_counts"].get(
                    "mutation_commit", 0
                )
                == mutation_events_before_insert
            )
    finally:
        application.close()

    ply_files = sorted(model_path.glob("point_cloud/iteration_*/point_cloud.ply"))
    assert len(ply_files) == 1
    evidence["output_ply"] = str(ply_files[0])
    evidence["output_ply_sha256"] = _sha256(ply_files[0])
    evidence_path = model_path / "issue1195_gpu_replay_evidence.json"
    _publish_durable_artifacts(evidence, ply_files[0])
    evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    print(json.dumps(evidence, sort_keys=True))
