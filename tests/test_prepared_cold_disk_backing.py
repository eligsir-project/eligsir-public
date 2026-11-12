"""CPU contracts for the prepared source-independent COLD spill tier."""

from __future__ import annotations

from collections import OrderedDict, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from eligsir.contracts import (
    BufferOwnership,
    BufferRef,
    CameraIntrinsics,
    MemoryDomain,
    SensorPacket,
    SourceIdentity,
)
from eligsir.contracts.runtime import PixelReloadToken
from eligsir.contracts.control import (
    DispositionKind,
    HistoricalEligibility,
    ModalityEligibility,
    ObservationDisposition,
    ResidencyIntent,
    SemanticAction,
)
from eligsir.mapping.observation_db import ColdPixelHandle, FrameObservationDB
from eligsir.runtime.prepared_ingress import (
    PreparedIngressControlToken,
    PreparedIngressDeferred,
    PreparedIngressItem,
    PreparedSourceIngressHandoff,
)
from eligsir.runtime.prepared_runtime_bridge import PreparedRuntimeBridge
from eligsir.runtime.prepared_source_prep_worker import PreparedSourcePrepWorker
from eligsir.runtime.prepared_source_producer import PreparedSourceProducer
from eligsir.runtime.production_application import CanonicalMappingRuntime
from eligsir.runtime.ready_source import ReadySourceOwner
from eligsir.runtime.ready_source import ReadySourceLane
from eligsir.runtime.transfers import DeviceTransferOwner
from eligsir.tracking.providers import observation_from_pose_matrix


def _source(ordinal: int) -> SourceIdentity:
    return SourceIdentity("live-cache", 0, ordinal, ordinal * 1_000)


def _record(db: FrameObservationDB, ordinal: int, rgb: np.ndarray, depth: np.ndarray):
    source = _source(ordinal)
    intrinsics = CameraIntrinsics(
        fx=2.0, fy=2.0, cx=1.0, cy=1.0, width=3, height=2, depth_scale=1000.0
    )
    return db.record_prepared_historical_candidate(
        source_identity=source,
        pose_c2w=np.eye(4, dtype=np.float32),
        timestamp=source.timestamp_seconds,
        intrinsics=intrinsics,
        cold_pixels=ColdPixelHandle(
            source_identity=source.fingerprint(),
            rgb=rgb,
            depth=depth,
            intrinsics=intrinsics,
            rgb_encoding="rgb8",
            depth_encoding="16UC1" if depth.dtype == np.uint16 else "depth_float32",
            depth_units="mm" if depth.dtype == np.uint16 else "m",
        ),
        modality_eligibility=ModalityEligibility(rgb_supervision=True, depth_supervision=True),
        disposition_snapshot_id=ordinal,
        semantic_action="commit",
    )


def _spill_for_next_packet(db: FrameObservationDB, *, payload_bytes: int) -> int:
    """Drive the existing worker/receipt boundary until one pending packet fits."""
    assert not db.can_accept_prepared_cold_payload(payload_bytes=payload_bytes)
    committed = db.commit_cold_payload_spill_receipts(
        db.write_pending_cold_payload_spills(max_items=2)
    )
    assert db.can_accept_prepared_cold_payload(payload_bytes=payload_bytes)
    return committed


def _prepared_item(
    ordinal: int,
    rgb: np.ndarray,
    depth: np.ndarray,
    *,
    reload_token: PixelReloadToken | None = None,
):
    source = _source(ordinal)
    intrinsics = CameraIntrinsics(
        fx=2.0, fy=2.0, cx=1.0, cy=1.0, width=3, height=2, depth_scale=1000.0
    )
    observation = observation_from_pose_matrix(source, np.eye(4), provider="test")
    disposition = ObservationDisposition(
        source=source,
        snapshot_id=ordinal,
        kind=DispositionKind.ADMIT,
        observe=True,
        mapping_candidate=True,
        admit_supervision=True,
        allow_geometry_insertion=False,
        residency=ResidencyIntent.HOT,
        reason="test",
        modality_eligibility=ModalityEligibility(rgb_supervision=True, depth_supervision=True),
        semantic_action=SemanticAction.COMMIT,
        historical_eligibility=HistoricalEligibility.ELIGIBLE,
    )
    packet = SensorPacket(
        source=source,
        rgb=BufferRef.from_numpy(
            rgb, encoding="rgb8", ownership=BufferOwnership.BORROWED, owner_token=object()
        ),
        depth=BufferRef.from_numpy(
            depth,
            encoding="16UC1",
            units="mm",
            ownership=BufferOwnership.BORROWED,
            owner_token=object(),
        ),
        intrinsics=intrinsics,
        tracking=observation,
        reload_token=reload_token,
    )
    token = PreparedIngressControlToken(ordinal, source, ordinal, disposition)
    return PreparedIngressItem(ordinal, packet, observation, disposition, token)


def test_prepared_cold_spill_is_source_independent_after_input_buffers_are_reused(tmp_path):
    """A spill token names only the local cache and restores exact CPU bytes."""
    db = FrameObservationDB(
        device="cpu",
        max_cold_payload_frames=1,
        cold_payload_spill_root=tmp_path / "prepared_cold_payloads",
    )
    rgb0 = np.full((2, 3, 3), 17, dtype=np.uint8)
    depth0 = np.full((2, 3), 1007, dtype=np.uint16)
    first = _record(db, 17, rgb0, depth0)
    assert _spill_for_next_packet(db, payload_bytes=30) == 1
    _record(
        db,
        18,
        np.full((2, 3, 3), 18, dtype=np.uint8),
        np.full((2, 3), 1008, dtype=np.uint16),
    )

    rgb0.fill(0)
    depth0.fill(0)
    handle = db.get_cold_pixels(first.frame_id)
    assert handle is not None
    assert handle.reload_token is not None
    assert handle.reload_token.provider == "prepared_cold_disk_rgbd_v1"
    assert all(str(path).startswith(str(tmp_path)) for path in handle.reload_token.locators)

    rgb, depth = db.load_cold_payload_from_token(handle.reload_token)
    assert np.array_equal(rgb, np.full((2, 3, 3), 17, dtype=np.uint8))
    assert np.array_equal(depth, np.full((2, 3), 1007, dtype=np.uint16))


def test_causal_holdout_cache_is_not_train_eligible_and_survives_buffer_reuse(tmp_path):
    """The direct holdout bypass shares cache backing, not train history."""
    db = FrameObservationDB(
        device="cpu",
        max_cold_payload_frames=1,
        cold_payload_spill_root=tmp_path / "prepared_cold_payloads",
    )
    runtime = object.__new__(CanonicalMappingRuntime)
    runtime._observation_db = db
    rgb = np.full((2, 3, 3), 41, dtype=np.uint8)
    depth = np.full((2, 3), 1041, dtype=np.uint16)
    first = _prepared_item(41, rgb, depth).packet
    second = _prepared_item(
        42,
        np.full((2, 3, 3), 42, dtype=np.uint8),
        np.full((2, 3), 1042, dtype=np.uint16),
    ).packet

    runtime.cache_causal_bypass_packet(first, role="holdout")
    with pytest.raises(PreparedIngressDeferred, match="capacity is unavailable"):
        runtime.cache_causal_bypass_packet(second, role="guard")
    assert db.commit_cold_payload_spill_receipts(
        db.write_pending_cold_payload_spills(max_items=2)
    ) == 1
    runtime.cache_causal_bypass_packet(second, role="guard")

    rgb.fill(0)
    depth.fill(0)
    frame_id = db.frame_id_for_source(first.source)
    assert frame_id is not None
    assert db.historical_candidate_for_source(first.source) is None
    handle = db.get_cold_pixels(frame_id)
    assert handle is not None and handle.reload_token is not None
    replay_rgb, replay_depth = db.load_cold_payload_from_token(handle.reload_token)
    assert np.array_equal(replay_rgb, np.full((2, 3, 3), 41, dtype=np.uint8))
    assert np.array_equal(replay_depth, np.full((2, 3), 1041, dtype=np.uint16))


def test_terminal_tracking_rejection_is_cache_owned_without_a_pose(tmp_path):
    """A tracking failure retains bytes but never manufactures replay metadata."""
    db = FrameObservationDB(
        device="cpu",
        max_cold_payload_frames=1,
        cold_payload_spill_root=tmp_path / "prepared_cold_payloads",
    )
    runtime = object.__new__(CanonicalMappingRuntime)
    runtime._observation_db = db
    rgb = np.full((2, 3, 3), 57, dtype=np.uint8)
    depth = np.full((2, 3), 1057, dtype=np.uint16)
    raw = replace(_prepared_item(57, rgb, depth).packet, tracking=None)
    captured = runtime.capture_packet_for_history(raw)
    rgb.fill(0)
    depth.fill(0)

    runtime.cache_terminal_packet(
        captured, SimpleNamespace(accepted=False, disposition=None)
    )

    frame_id = db.frame_id_for_source(captured.source)
    assert frame_id is not None
    record = db.get_frame(frame_id)
    assert record is not None
    assert record.pose_c2w is None
    assert record.historical_eligible is False
    assert record.semantic_action == "terminal"
    assert db.historical_candidate_for_source(captured.source) is None
    assert _spill_for_next_packet(db, payload_bytes=30) == 1
    _record(
        db,
        58,
        np.full((2, 3, 3), 58, dtype=np.uint8),
        np.full((2, 3), 1058, dtype=np.uint16),
    )
    handle = db.get_cold_pixels(frame_id)
    assert handle is not None and handle.reload_token is not None
    replay_rgb, replay_depth = db.load_cold_payload_from_token(handle.reload_token)
    assert np.array_equal(replay_rgb, np.full((2, 3, 3), 57, dtype=np.uint8))
    assert np.array_equal(replay_depth, np.full((2, 3), 1057, dtype=np.uint16))


def test_prepared_cold_spill_uses_the_available_bounded_batch_without_wait_to_fill(tmp_path):
    db = FrameObservationDB(
        device="cpu",
        max_cold_payload_bytes=60,
        cold_payload_spill_root=tmp_path / "prepared_cold_payloads",
    )
    for ordinal in (21, 22):
        _record(
            db,
            ordinal,
            np.full((2, 3, 3), ordinal, dtype=np.uint8),
            np.full((2, 3), ordinal, dtype=np.uint16),
        )

    assert not db.can_accept_prepared_cold_payload(payload_bytes=60)
    receipts = db.write_pending_cold_payload_spills(max_items=2)
    assert [receipt.frame_id for receipt in receipts] == [21, 22]
    assert db.commit_cold_payload_spill_receipts(receipts) == 2
    assert db.pending_cold_payload_spill_count == 0


def test_spill_preflight_holds_new_packets_only_while_a_bounded_spill_is_pending(tmp_path):
    db = FrameObservationDB(
        device="cpu",
        max_cold_payload_frames=1,
        cold_payload_spill_root=tmp_path / "prepared_cold_payloads",
    )
    _record(
        db, 24, np.full((2, 3, 3), 24, np.uint8), np.full((2, 3), 24, np.uint16)
    )
    assert not db.can_accept_prepared_cold_payload(payload_bytes=24)
    assert db.commit_cold_payload_spill_receipts(
        db.write_pending_cold_payload_spills(max_items=1)
    ) == 1
    assert db.can_accept_prepared_cold_payload(payload_bytes=24)
    _record(
        db, 25, np.full((2, 3, 3), 25, np.uint8), np.full((2, 3), 25, np.uint16)
    )
    assert not db.can_accept_prepared_cold_payload(payload_bytes=24)
    assert db.commit_cold_payload_spill_receipts(
        db.write_pending_cold_payload_spills(max_items=1)
    ) == 1
    assert db.can_accept_prepared_cold_payload(payload_bytes=24)


def test_disk_capacity_failure_keeps_the_protected_victim_callable(tmp_path):
    db = FrameObservationDB(
        device="cpu",
        max_cold_payload_frames=1,
        cold_payload_spill_root=tmp_path / "prepared_cold_payloads",
        cold_payload_spill_max_bytes=1,
    )
    first = _record(
        db, 26, np.full((2, 3, 3), 26, np.uint8), np.full((2, 3), 26, np.uint16)
    )
    assert not db.can_accept_prepared_cold_payload(payload_bytes=24)
    receipts = db.write_pending_cold_payload_spills(max_items=1)
    assert len(receipts) == 1 and receipts[0].error is not None
    assert db.commit_cold_payload_spill_receipts(receipts) == 0
    handle = db.get_cold_pixels(first.frame_id)
    assert handle is not None and handle.has_resident_payload and handle.reload_token is None
    assert db.take_cold_payload_spill_failures() == receipts


def test_single_oversized_ram_payload_fails_before_resident_cap_is_exceeded(tmp_path):
    db = FrameObservationDB(
        device="cpu",
        max_cold_payload_bytes=1,
        cold_payload_spill_root=tmp_path / "prepared_cold_payloads",
    )
    with pytest.raises(ValueError, match="cannot fit configured resident bounds"):
        _record(
            db, 28, np.full((2, 3, 3), 28, np.uint8), np.full((2, 3), 2.8, np.float32)
        )
    assert db.cold_payload_bytes == 0
    assert db.frame_id_for_source(_source(28)) is None


def test_paused_spill_does_not_block_mapper_metadata_or_capacity_checks(tmp_path):
    """The worker writes outside the shared COLD metadata lock."""
    db = FrameObservationDB(
        device="cpu",
        max_cold_payload_frames=1,
        cold_payload_spill_root=tmp_path / "prepared_cold_payloads",
    )
    first = _record(
        db, 29, np.full((2, 3, 3), 29, np.uint8), np.full((2, 3), 29, np.uint16)
    )
    assert not db.can_accept_prepared_cold_payload(payload_bytes=30)
    started = threading.Event()
    release = threading.Event()
    original_write = db._cold_payload_disk.write

    def paused_write(request):
        started.set()
        assert release.wait(timeout=2)
        return original_write(request)

    db._cold_payload_disk.write = paused_write
    receipts: list[object] = []
    worker = threading.Thread(
        target=lambda: receipts.extend(db.write_pending_cold_payload_spills(max_items=1))
    )
    worker.start()
    assert started.wait(timeout=2)
    assert db.get_cold_pixels(first.frame_id) is not None
    assert not db.can_accept_prepared_cold_payload(payload_bytes=30)
    release.set()
    worker.join(timeout=2)
    assert not worker.is_alive()
    assert db.commit_cold_payload_spill_receipts(tuple(receipts)) == 1


def test_corrupt_local_cache_token_is_retired_instead_of_repeating_pending(tmp_path):
    db = FrameObservationDB(
        device="cpu",
        max_cold_payload_frames=1,
        cold_payload_spill_root=tmp_path / "prepared_cold_payloads",
    )
    record = _record(
        db, 30, np.full((2, 3, 3), 30, np.uint8), np.full((2, 3), 30, np.uint16)
    )
    assert _spill_for_next_packet(db, payload_bytes=30) == 1
    handle = db.get_cold_pixels(record.frame_id)
    assert handle is not None and handle.reload_token is not None
    Path(handle.reload_token.locators[0]).write_text("{}", encoding="utf-8")

    runtime = object.__new__(CanonicalMappingRuntime)
    runtime._observation_db = db
    runtime._admission_prefetch_lock = threading.Lock()
    runtime._admission_prefetch_pending = set()
    runtime._admission_prefetch_terminal_failures = OrderedDict()
    runtime._admission_prefetch_failed_total = 0
    runtime._admission_prefetch_terminal_failed_total = 0
    retired: list[tuple[int, str]] = []
    runtime._retire_historical_identity = (
        lambda frame_id, *, reason: retired.append((frame_id, reason)) or True
    )

    runtime._prefetch_decode_worker(record.frame_id, handle.reload_token)
    runtime._commit_prepared_cold_payload_spill_receipts()

    assert retired == [(record.frame_id, "cache_backing_failure")]
    assert db.historical_candidate_for_source(_source(30)) is None
    assert runtime._admission_prefetch_terminal_failed_total == 1


def test_preparation_worker_spills_without_a_committed_ingress_item(tmp_path):
    """Pressure recovery keeps using the existing prep worker after ingress drains."""
    db = FrameObservationDB(
        device="cpu",
        max_cold_payload_frames=1,
        cold_payload_spill_root=tmp_path / "prepared_cold_payloads",
    )
    _record(
        db,
        31,
        np.full((2, 3, 3), 31, dtype=np.uint8),
        np.full((2, 3), 31, dtype=np.uint16),
    )
    assert not db.can_accept_prepared_cold_payload(payload_bytes=30)
    owner = ReadySourceOwner(max_fresh=1, max_replay=1)
    bridge = PreparedRuntimeBridge(
        PreparedSourceIngressHandoff(capacity=1),
        PreparedSourceProducer(owner, transfer_owner=DeviceTransferOwner("cpu", max_inflight=1)),
        ready_owner=owner,
        cold_payload_spill_writer=db.write_pending_cold_payload_spills,
    )
    worker = PreparedSourcePrepWorker(bridge)

    assert worker.pump_once() is True
    assert db.commit_cold_payload_spill_receipts(
        bridge.take_cold_payload_spill_receipts()
    ) == 1
    assert db.pending_cold_payload_spill_count == 0


def test_common_runtime_spills_without_prepared_b4_worker(tmp_path):
    """Default cache pressure uses the runtime's existing bounded I/O executor."""
    db = FrameObservationDB(
        device="cpu",
        max_cold_payload_frames=1,
        cold_payload_spill_root=tmp_path / "prepared_cold_payloads",
    )
    record = _record(
        db,
        32,
        np.full((2, 3, 3), 32, dtype=np.uint8),
        np.full((2, 3), 32, dtype=np.uint16),
    )
    assert not db.can_accept_prepared_cold_payload(payload_bytes=30)
    runtime = object.__new__(CanonicalMappingRuntime)
    runtime._observation_db = db
    runtime._admission_prefetch_executor = ThreadPoolExecutor(max_workers=1)
    runtime._admission_prefetch_lock = threading.Lock()
    runtime._admission_prefetch_terminal_failures = OrderedDict()
    runtime._admission_prefetch_terminal_failed_total = 0
    started = threading.Event()
    release = threading.Event()
    original_write = db._cold_payload_disk.write

    def paused_write(request):
        started.set()
        assert release.wait(timeout=2)
        return original_write(request)

    db._cold_payload_disk.write = paused_write

    runtime._commit_prepared_cold_payload_spill_receipts()
    future = runtime._cold_payload_spill_future
    assert future is not None
    assert started.wait(timeout=2)
    # Mapper-side metadata and its normal tick fence remain nonblocking while
    # the shared executor owns storage I/O.
    assert db.get_metadata(record.frame_id) is not None
    runtime._commit_prepared_cold_payload_spill_receipts()
    assert runtime._cold_payload_spill_future is future
    release.set()
    future.result(timeout=2)  # Declared test barrier, outside mapper work.
    runtime._commit_prepared_cold_payload_spill_receipts()
    runtime._admission_prefetch_executor.shutdown(wait=True)

    handle = db.get_cold_pixels(record.frame_id)
    assert handle is not None and handle.reload_token is not None
    assert db.pending_cold_payload_spill_count == 0


def test_prepared_retention_copies_borrowed_ingress_before_local_spill(tmp_path):
    """Prepared retention cannot retain an MCAP/camera buffer or its reload token."""
    db = FrameObservationDB(
        device="cpu",
        max_cold_payload_frames=1,
        cold_payload_spill_root=tmp_path / "prepared_cold_payloads",
    )
    runtime = object.__new__(CanonicalMappingRuntime)
    runtime._observation_db = db
    runtime._register_historical_record = lambda *_args, **_kwargs: None
    rgb = np.full((2, 3, 3), 71, dtype=np.uint8)
    depth = np.full((2, 3), 1071, dtype=np.uint16)
    first = _prepared_item(
        71,
        rgb,
        depth,
        reload_token=PixelReloadToken(
            provider="rgbd_files_v1", locators=("closed-rgb", "closed-depth")
        ),
    )
    source_packet = first.packet
    first = replace(first, packet=runtime.capture_packet_for_history(source_packet))
    source_packet.rgb.data.fill(0)
    source_packet.depth.data.fill(0)
    source_packet.tracking.pose.matrix[...] = 0
    assert int(first.packet.rgb.data[0, 0, 0]) == 71
    assert int(first.packet.depth.data[0, 0]) == 1071
    assert np.array_equal(first.packet.tracking.pose.matrix, np.eye(4, dtype=np.float32))
    assert runtime.retain_prepared_source(first, lane=ReadySourceLane.FRESH)
    assert _spill_for_next_packet(db, payload_bytes=30) == 1
    assert runtime.retain_prepared_source(
        _prepared_item(
            72,
            np.full((2, 3, 3), 72, dtype=np.uint8),
            np.full((2, 3), 1072, dtype=np.uint16),
        ),
        lane=ReadySourceLane.FRESH,
    )

    rgb.fill(0)
    depth.fill(0)
    record = db.historical_candidate_for_source(first.packet.source)
    assert record is not None
    handle = db.get_cold_pixels(record.frame_id)
    assert handle is not None and handle.reload_token is not None
    assert handle.reload_token.provider == "prepared_cold_disk_rgbd_v1"
    restored_rgb, restored_depth = db.load_cold_payload_from_token(handle.reload_token)
    assert np.array_equal(restored_rgb, np.full((2, 3, 3), 71, dtype=np.uint8))
    assert np.array_equal(restored_depth, np.full((2, 3), 1071, dtype=np.uint16))


def test_common_capture_rejects_device_only_packet_before_source_acknowledgement(tmp_path):
    """The CPU cache contract never hides a device-to-host synchronization."""
    runtime = object.__new__(CanonicalMappingRuntime)
    runtime._observation_db = FrameObservationDB(
        device="cpu",
        max_cold_payload_frames=1,
        cold_payload_spill_root=tmp_path / "prepared_cold_payloads",
    )
    item = _prepared_item(
        73,
        np.full((2, 3, 3), 73, dtype=np.uint8),
        np.full((2, 3), 1073, dtype=np.uint16),
    )
    packet = replace(
        item.packet,
        rgb=replace(
            item.packet.rgb,
            spec=replace(
                item.packet.rgb.spec,
                memory_domain=MemoryDomain.CUDA,
                device="cuda",
            ),
        ),
    )

    with pytest.raises(ValueError, match="only CPU ndarray packets"):
        runtime.capture_packet_for_history(packet)


def test_rejected_input_is_cache_owned_without_replay_eligibility(tmp_path):
    """Quality rejection retains source-independent bytes but no training lane."""
    runtime = object.__new__(CanonicalMappingRuntime)
    runtime._observation_db = FrameObservationDB(
        device="cpu",
        max_cold_payload_frames=1,
        cold_payload_spill_root=tmp_path / "prepared_cold_payloads",
    )
    incoming = _prepared_item(
        74,
        np.full((2, 3, 3), 74, dtype=np.uint8),
        np.full((2, 3), 1074, dtype=np.uint16),
    ).packet

    captured = runtime.capture_packet_for_history(incoming)
    incoming.rgb.data.fill(0)
    incoming.depth.data.fill(0)
    snapshot = SimpleNamespace(source=captured.source, snapshot_id=74)
    runtime._latest_control_packet = captured
    runtime._latest_control_outputs = (snapshot, ())
    runtime._work_budget_owner = SimpleNamespace(resolve=lambda *_args, **_kwargs: object())
    runtime._native_geometry_evidence_purpose = lambda: None
    runtime._dispositions_since_tick = deque(maxlen=4)
    runtime._latest_report_work = None
    runtime.record_disposition(
        ObservationDisposition(
            source=captured.source,
            snapshot_id=74,
            kind=DispositionKind.SKIP_QUALITY,
            observe=True,
            mapping_candidate=False,
            admit_supervision=False,
            allow_geometry_insertion=False,
            residency=ResidencyIntent.METADATA_ONLY,
            reason="quality rejected",
            semantic_action=SemanticAction.REJECT,
            historical_eligibility=HistoricalEligibility.NONE,
            modality_eligibility=ModalityEligibility(),
        )
    )

    frame_id = runtime._observation_db.frame_id_for_source(captured.source)
    assert frame_id is not None
    record = runtime._observation_db.get_frame(frame_id)
    handle = runtime._observation_db.get_cold_pixels(frame_id)
    assert record is not None and not record.historical_eligible
    assert record.semantic_action == "rejected"
    assert runtime._observation_db.historical_candidate_for_source(captured.source) is None
    assert handle is not None and handle.rgb is captured.rgb.data
    assert np.array_equal(handle.rgb, np.full((2, 3, 3), 74, dtype=np.uint8))
    assert np.array_equal(handle.depth, np.full((2, 3), 1074, dtype=np.uint16))
    assert _spill_for_next_packet(runtime._observation_db, payload_bytes=1) == 1
    handle = runtime._observation_db.get_cold_pixels(frame_id)
    assert handle is not None and handle.reload_token is not None
    replay_rgb, replay_depth = runtime._observation_db.load_cold_payload_from_token(
        handle.reload_token
    )
    assert not runtime._observation_db.get_frame(frame_id).historical_eligible
    assert runtime._observation_db.historical_candidate_for_source(captured.source) is None
    assert np.array_equal(replay_rgb, np.full((2, 3, 3), 74, dtype=np.uint8))
    assert np.array_equal(replay_depth, np.full((2, 3), 1074, dtype=np.uint16))


def test_prepared_handoff_upload_uses_cache_owned_borrowed_buffers(tmp_path):
    """Borrowed ingress may be reused before worker H2D without changing READY."""
    db = FrameObservationDB(
        device="cpu",
        max_cold_payload_frames=1,
        cold_payload_spill_root=tmp_path / "prepared_cold_payloads",
    )
    runtime = object.__new__(CanonicalMappingRuntime)
    runtime._observation_db = db
    runtime._register_historical_record = lambda *_args, **_kwargs: None
    rgb = np.full((2, 3, 3), 91, dtype=np.uint8)
    depth = np.full((2, 3), 1091, dtype=np.uint16)
    item = _prepared_item(91, rgb, depth)
    owner = ReadySourceOwner(max_fresh=1, max_replay=1)

    def commit_sink(incoming):
        cached = replace(
            incoming,
            packet=runtime.capture_packet_for_history(incoming.packet),
        )
        assert runtime.retain_prepared_source(cached, lane=ReadySourceLane.FRESH)
        return cached

    bridge = PreparedRuntimeBridge(
        PreparedSourceIngressHandoff(capacity=1, commit_sink=commit_sink),
        PreparedSourceProducer(owner, transfer_owner=DeviceTransferOwner("cpu", max_inflight=1)),
        ready_owner=owner,
        consumer_owner="test",
    )
    assert bridge.handoff.try_offer(item).value == "enqueued"
    rgb.fill(0)
    depth.fill(0)
    assert bridge.drain().enqueued == 1
    lease = owner.select(1)[0]
    assert np.allclose(
        lease.borrowed_device_payloads["rgb"].data.numpy(),
        np.full((2, 3, 3), 91 / 255.0, dtype=np.float32),
    )
    assert np.allclose(
        lease.borrowed_device_payloads["depth_meters"].data.numpy(),
        np.full((2, 3), 1.091, dtype=np.float32),
    )
