"""Production wiring guards for deferred cold-history observations."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from eligsir.contracts.control import (
    ControlLimits,
    ControlSnapshot,
    DispositionKind,
    HistoricalEligibility,
    ModalityEligibility,
    ObservationDisposition,
    ResidencyIntent,
    SemanticAction,
    FreshnessMode,
)
from eligsir.contracts.runtime import (
    BufferRef,
    MappingFrame,
    PixelReloadToken,
    SensorPacket,
    SourceIdentity,
)
from eligsir.contracts.work_clocks import RuntimeWorkClockOwner
from eligsir.contracts.types import CameraIntrinsics, Pose
from eligsir.mapping.admission.observation_owner import (
    ColdRetentionPermit,
    ObservationOwner,
)
from eligsir.mapping.canonical_scene import CanonicalStreamingScene
from eligsir.mapping.observation_db import FrameObservationDB
from eligsir.runtime.production_application import CanonicalMappingRuntime
from eligsir.runtime.replay import packet_from_streaming_frame
from eligsir.tracking.providers import observation_from_pose_matrix


def _source(ordinal: int = 12) -> SourceIdentity:
    return SourceIdentity("tum", 0, ordinal, 1_000_000_000 + ordinal)


def _intrinsics() -> CameraIntrinsics:
    return CameraIntrinsics(
        fx=2.0,
        fy=2.0,
        cx=1.0,
        cy=1.0,
        width=2,
        height=2,
        depth_fx=2.0,
        depth_fy=2.0,
        depth_cx=1.0,
        depth_cy=1.0,
        depth_width=2,
        depth_height=2,
        depth_scale=1.0,
    )


def _packet(*, reloadable: bool) -> SensorPacket:
    source = _source()
    return SensorPacket(
        source=source,
        rgb=BufferRef.from_numpy(np.zeros((2, 2, 3), dtype=np.uint8), encoding="rgb8"),
        depth=BufferRef.from_numpy(
            np.ones((2, 2), dtype=np.float32), encoding="32FC1", units="m"
        ),
        intrinsics=_intrinsics(),
        tracking=observation_from_pose_matrix(source, np.eye(4), provider="dataset_pose"),
        reload_token=(
            PixelReloadToken(
                provider="rgbd_files_v1",
                locators=("/dataset/rgb/12.png", "/dataset/depth/12.png"),
                parameters=(("depth_scale", "5000.0"),),
            )
            if reloadable
            else None
        ),
    )


def _disposition() -> ObservationDisposition:
    modalities = ModalityEligibility(True, "qualified", True, "qualified", True)
    return ObservationDisposition(
        source=_source(),
        snapshot_id=4,
        kind=DispositionKind.SKIP_REDUNDANT,
        semantic_action=SemanticAction.DEFER,
        historical_eligibility=HistoricalEligibility.ELIGIBLE,
        observe=True,
        mapping_candidate=False,
        admit_supervision=False,
        allow_geometry_insertion=False,
        residency=ResidencyIntent.COLD,
        reason="no_motion",
        modality_eligibility=modalities,
    )


def _control_snapshot(packet: SensorPacket, snapshot_id: int) -> ControlSnapshot:
    return ControlSnapshot(
        snapshot_id=snapshot_id,
        control_version=1,
        source=packet.source,
        tracking=packet.tracking,
        pending_sources=(packet.source,),
        newest_frontier=packet.source,
        freshness_mode=FreshnessMode.HEALTHY,
        pending_oldest_age_ns=0,
        ingress_queue_depth=0,
        candidate_queue_depth=0,
        transfer_queue_depth=0,
        gpu_queue_depth=0,
        map_content_version=0,
        topology_version=7,
        parameter_version=0,
        gaussian_count=1,
        optimizer_steps_completed=0,
        limits=ControlLimits(1, 1, 1, 0, 0, 1024),
    )


def test_reload_token_is_part_of_packet_metadata_identity():
    reloadable = _packet(reloadable=True)
    resident = _packet(reloadable=False)

    assert reloadable.metadata_fingerprint() != resident.metadata_fingerprint()
    assert reloadable.reload_token is not None


def test_deferred_dataset_packet_records_reloadable_pixel_free_history():
    runtime = object.__new__(CanonicalMappingRuntime)
    runtime._observation_db = FrameObservationDB(
        device="cpu", max_cold_payload_frames=1, max_cold_payload_bytes=64
    )
    packet = _packet(reloadable=True)
    observation = packet.tracking
    assert observation is not None

    record = runtime._retain_deferred_observation(
        _disposition(),
        ColdRetentionPermit(packet.source, 4),
        packet,
        observation,
    )

    handle = runtime._observation_db.get_cold_pixels(record.frame_id)
    candidates = runtime._observation_db.historical_candidates(require_rgb=True)
    assert handle is not None and handle.reloadable
    assert not handle.has_resident_payload
    assert len(candidates) == 1 and candidates[0].can_materialize
    assert runtime._observation_db.cold_payload_occupancy == 0
    direct = runtime._observation_db.historical_candidate_for_source(
        packet.source, require_rgb=True
    )
    assert direct is not None
    assert direct.disposition_snapshot_id == 4
    assert direct.semantic_action == "defer"


def test_deferred_live_packet_uses_bounded_cpu_archive_without_gpu_materialization():
    runtime = object.__new__(CanonicalMappingRuntime)
    runtime._observation_db = FrameObservationDB(
        device="cpu", max_cold_payload_frames=1, max_cold_payload_bytes=64
    )
    packet = _packet(reloadable=False)
    observation = packet.tracking
    assert observation is not None

    record = runtime._retain_deferred_observation(
        _disposition(),
        ColdRetentionPermit(packet.source, 4),
        packet,
        observation,
    )

    handle = runtime._observation_db.get_cold_pixels(record.frame_id)
    assert handle is not None and handle.has_resident_payload
    assert not handle.reloadable
    assert runtime._observation_db.cold_payload_occupancy == 1
    assert runtime._observation_db.active_supervision_occupancy == 0


def test_file_backed_dataset_packet_carries_deterministic_reload_token(tmp_path):
    frame = SimpleNamespace(
        source_ordinal=12,
        index=12,
        timestamp=1.0,
        color_ts_ns=None,
        c2w=np.eye(4),
        rgb_path=str(tmp_path / "rgb.png"),
        depth_path=str(tmp_path / "depth.png"),
        depth_scale=5000.0,
        fx=2.0,
        fy=2.0,
        cx=1.0,
        cy=1.0,
        width=2,
        height=2,
        depth_fx=None,
        depth_fy=None,
        depth_cx=None,
        depth_cy=None,
        depth_width=None,
        depth_height=None,
        _rgb_np=None,
        _depth_np=None,
    )

    packet = packet_from_streaming_frame(
        frame,
        source_id="tum",
        rgb=np.zeros((2, 2, 3), dtype=np.uint8),
        depth=np.ones((2, 2), dtype=np.float32),
    )

    assert packet.reload_token == PixelReloadToken(
        provider="rgbd_files_v1",
        locators=(frame.rgb_path, frame.depth_path),
        parameters=(("depth_scale", "5000.0"),),
    )


def test_exactly_once_materialization_preserves_pixel_reload_provenance():
    from eligsir.contracts.materialize import materialize_mapping_frame

    packet = _packet(reloadable=True)
    mapping = materialize_mapping_frame(packet)

    assert mapping.reload_token == packet.reload_token


def test_file_reload_token_decodes_rgb_and_metric_depth_only_when_selected(tmp_path):
    from PIL import Image

    rgb_path = tmp_path / "rgb.png"
    depth_path = tmp_path / "depth.png"
    Image.fromarray(np.full((2, 2, 3), 17, dtype=np.uint8)).save(rgb_path)
    Image.fromarray(np.full((2, 2), 2500, dtype=np.uint16)).save(depth_path)
    token = PixelReloadToken(
        provider="rgbd_files_v1",
        locators=(str(rgb_path), str(depth_path)),
        parameters=(("depth_scale", "5000.0"),),
    )

    rgb, depth = CanonicalMappingRuntime._decode_reload_token(token)

    assert rgb.dtype == np.uint8 and rgb.shape == (2, 2, 3)
    assert depth is not None and depth.dtype == np.float32
    np.testing.assert_allclose(depth, 0.5)


def test_terminal_defer_sink_retains_packet_and_registers_future_replay(monkeypatch):
    packet = _packet(reloadable=True)
    tracking = packet.tracking
    assert tracking is not None
    disposition = _disposition()
    snapshot = _control_snapshot(packet, 4)
    retained = []
    runtime = object.__new__(CanonicalMappingRuntime)
    runtime._latest_control_outputs = (snapshot, ())
    runtime._latest_control_packet = packet
    runtime._work_budget_owner = SimpleNamespace(
        resolve=lambda *_args, **_kwargs: SimpleNamespace()
    )
    runtime._dispositions_since_tick = __import__("collections").deque(maxlen=4)
    runtime._latest_report_work = None
    runtime._replay_modalities = __import__("collections").OrderedDict()
    runtime._replay_weights = {}
    runtime.observation_owner = SimpleNamespace(
        cold_retention_permit=lambda value: ColdRetentionPermit(
            value.source, value.snapshot_id
        )
    )
    monkeypatch.setattr(
        runtime,
        "_retain_deferred_observation",
        lambda *args: retained.append(args),
    )

    runtime.record_disposition(disposition)

    assert len(retained) == 1 and retained[0][2] is packet
    assert runtime._latest_control_packet is None
    assert runtime._replay_modalities[packet.source] == disposition.modality_eligibility
    assert runtime._replay_weights[packet.source] == 1.0


def test_selected_deferred_identity_promotes_to_camera_without_live_ingest(monkeypatch):
    packet = _packet(reloadable=False)
    mapping = MappingFrame(
        source=packet.source,
        rgb=packet.rgb,
        depth=packet.depth,
        intrinsics=packet.intrinsics,
        device="cpu",
        pose=Pose(np.eye(4, dtype=np.float32)),
    )
    promoted = []
    camera = SimpleNamespace(_source_identity=packet.source)
    scene = SimpleNamespace(
        train_cameras=[],
        current_camera=None,
        promote_historical_frame=lambda observation: (
            promoted.append(observation) or camera
        ),
    )
    runtime = object.__new__(CanonicalMappingRuntime)
    runtime.scene = scene
    runtime._source_identities_by_ordinal = {}
    runtime._admitted_clock_ns = 0
    runtime._work_clock_owner = RuntimeWorkClockOwner()
    admitted_optimizer_updates = []
    runtime._optimization_runtime = SimpleNamespace(
        note_admitted_source=lambda *, optimizer_updates_completed: (
            admitted_optimizer_updates.append(optimizer_updates_completed)
        )
    )
    promoted_clocks = []
    runtime._services = SimpleNamespace(
        four_clocks=SimpleNamespace(
            record_admitted_keyframe=lambda: promoted_clocks.append(True)
        )
    )
    runtime._observation_db = SimpleNamespace(
        historical_candidate_for_source=lambda source, **_kwargs: SimpleNamespace(
            source_identity=source.fingerprint(),
            disposition_snapshot_id=4,
            semantic_action="defer",
        ),
        prepared_source_reference_for_source=lambda _source: None,
    )
    current_snapshot = _control_snapshot(packet, 5)
    runtime.observation_owner = SimpleNamespace(
        promotion_permit_from_retained=lambda **kwargs: SimpleNamespace(
            source=kwargs["source"],
            promotion_snapshot_id=kwargs["current_snapshot"].snapshot_id,
            topology_version=kwargs["current_snapshot"].topology_version,
        )
    )
    monkeypatch.setattr(runtime, "_replay_mapping_frame", lambda source: mapping)

    resolved = runtime._camera_for_source(
        packet.source, current_snapshot=current_snapshot
    )

    assert resolved is camera
    assert len(promoted) == 1
    assert promoted[0].mapping_frame.source == packet.source
    assert scene.current_camera is None
    assert runtime._admitted_clock_ns == packet.source.timestamp_ns
    assert promoted_clocks == [True]
    assert runtime._latest_admitted_optimizer_update == 0
    assert admitted_optimizer_updates == [0]


def test_durable_promotion_permit_rejects_non_defer_and_stale_snapshots():
    packet = _packet(reloadable=True)
    owner = object.__new__(ObservationOwner)
    current = _control_snapshot(packet, 5)

    permit = owner.promotion_permit_from_retained(
        source=packet.source,
        disposition_snapshot_id=4,
        semantic_action="defer",
        current_snapshot=current,
    )

    assert permit is not None
    assert permit.topology_version == current.topology_version
    assert (
        owner.promotion_permit_from_retained(
            source=packet.source,
            disposition_snapshot_id=4,
            semantic_action="reject",
            current_snapshot=current,
        )
        is None
    )
    import pytest

    with pytest.raises(ValueError, match="newer control snapshot"):
        owner.promotion_permit_from_retained(
            source=packet.source,
            disposition_snapshot_id=5,
            semantic_action="defer",
            current_snapshot=current,
        )


def test_admitted_camera_lookup_does_not_scan_unbounded_camera_history():
    """Canonical scene lookup must avoid walking its append-only pose history."""

    wanted = _packet(reloadable=True).source
    camera = SimpleNamespace(_source_identity=wanted)

    class _UnscannableHistory:
        def __iter__(self):
            raise AssertionError("admitted camera lookup scanned train_cameras")

    runtime = object.__new__(CanonicalMappingRuntime)
    runtime._source_identities_by_ordinal = {}
    runtime.scene = SimpleNamespace(
        train_cameras=_UnscannableHistory(),
        current_camera=None,
        admitted_camera_for_source=lambda source: camera if source == wanted else None,
    )

    assert runtime._camera_for_source(wanted) is camera


def test_canonical_scene_registers_only_admitted_camera_identities():
    source = _source(99)
    camera = object()
    scene = object.__new__(CanonicalStreamingScene)
    scene._admitted_cameras_by_source = {}

    scene._register_admitted_camera(
        SimpleNamespace(mapping_frame=SimpleNamespace(source=source)), camera
    )

    assert scene.admitted_camera_for_source(source) is camera
