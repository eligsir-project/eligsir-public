"""CPU contracts for accepted, prepared READY pose corrections."""

from dataclasses import fields, replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scene.cameras import Camera
from eligsir.contracts import (
    BufferOwnership, BufferRef, BufferSpec, CameraIntrinsics, MemoryDomain,
    ModalityEligibility, SourceIdentity,
)
from eligsir.mapping.observation_db import FrameObservationDB
from eligsir.runtime.prepared_ingress import PreparedIngressControlToken
from eligsir.runtime.prepared_pose_correction import prepare_pose_correction
from eligsir.runtime.prepared_runtime_bridge import PreparedRuntimeBridge
from eligsir.runtime.prepared_source import PreparedSourceLease, PreparedTierPayload
from eligsir.runtime.ready_b4_streaming_adapter import (
    ReadyB4StreamingOptimizationAdapter, ReadyB4StreamingStepInputs,
)
from eligsir.runtime.ready_camera_binder import bind_ready_camera
from eligsir.runtime.ready_insertion_frame import ReadyInsertionFrame
from eligsir.runtime.ready_only_tick import ReadySourceBinding, ReadySourceVersions
from eligsir.runtime.ready_source import ReadySourceLane, ReadySourceOwner
from eligsir.runtime.transfers import DeviceTransferOwner


TIERS = {
    "native": (8, 8.0, 12.0, 3.5),
    "half": (4, 4.0, 6.0, 1.5),
    "quarter": (2, 2.0, 3.0, 0.5),
}
VERSIONS = ReadySourceVersions(1, 2, 3)


def _source(ordinal=17):
    return SourceIdentity("prepared-pose", 2, ordinal, 1_700_000_000 + ordinal)


def _pose(x=1.0, y=2.0):
    return np.array(
        ((0, -1, 0, x), (1, 0, 0, y), (0, 0, 1, 0.5), (0, 0, 0, 1)),
        dtype=np.float32,
    )


def _intrinsics():
    return CameraIntrinsics(
        fx=8.0, fy=12.0, cx=3.5, cy=3.5, width=8, height=8,
        depth_fx=8.0, depth_fy=12.0, depth_cx=3.5, depth_cy=3.5,
        depth_width=8, depth_height=8, depth_scale=1000.0, intrinsics_source="test",
    )


def _payload(tensor, encoding, units=None):
    return BufferRef(
        data=tensor,
        spec=BufferSpec(
            shape=tuple(tensor.shape), dtype="float32", memory_domain=MemoryDomain.OTHER_DEVICE,
            encoding=encoding, units=units, device="cpu", contiguous=tensor.is_contiguous(),
        ),
        ownership=BufferOwnership.BORROWED, owner_token=object(),
    )


def _lease(transfer, source=None, *, pose=None, generation=0):
    """Valid resident RGB-D lease, including real original renderer geometry."""
    source = _source() if source is None else source
    pose = np.eye(4, dtype=np.float32) if pose is None else pose
    c2w = torch.from_numpy(pose.copy())
    tensors = {
        name: (
            torch.arange(size * size * 3, dtype=torch.float32).reshape(size, size, 3) / 255,
            torch.ones((size, size)),
            torch.tensor(((fx, 0, center), (0, fy, center), (0, 0, 1)), dtype=torch.float32),
            None,
        )
        for name, (size, fx, fy, center) in TIERS.items()
    }
    common, geometry, event = transfer.prepare_source_camera_geometry(
        c2w, tensors, w2c=torch.from_numpy(np.linalg.inv(pose)),
    )
    tiers = {}
    for name, (size, fx, fy, center) in TIERS.items():
        rgb, depth, K, _ = tensors[name]
        projection, full_projection = geometry[name]
        tiers[name] = PreparedTierPayload(
            name=name, rgb=_payload(rgb, "rgb_float32_hwc"),
            depth=_payload(depth, "depth_meters", "m"),
            camera_intrinsics=_payload(K, "camera_intrinsics"),
            width=size, height=size, intrinsics=(fx, fy, center, center),
            projection_matrix=_payload(projection, "projection_matrix"),
            full_proj_transform=_payload(full_projection, "full_proj_transform"),
        )
    native = tiers["native"]
    pose_tuple = tuple(tuple(float(value) for value in row) for row in pose)
    return PreparedSourceLease(
        source_identity=source, generation=generation,
        modality_eligibility=ModalityEligibility(
            rgb_supervision=True, depth_supervision=True, depth_insertion=True,
        ),
        source_metadata={
            **{name: getattr(source, name) for name in ("source_id", "stream_epoch", "source_ordinal", "timestamp_ns")},
            "rgb_shape": (8, 8, 3), "depth_shape": (8, 8),
            "depth_units": "m", "depth_aligned_to_rgb": True,
            "camera_intrinsics": tuple(getattr(_intrinsics(), field.name) for field in fields(CameraIntrinsics)),
            "pose_c2w": pose_tuple,
            "calibration_attestation": (
                "prepared-calibration-v1", pose_tuple,
                ((8.0, 0.0, 3.5), (0.0, 12.0, 3.5), (0.0, 0.0, 1.0)),
            ),
        },
        tier_preparation_available=tuple(tiers), tier_payloads=tiers,
        borrowed_device_payloads={
            "rgb": native.rgb, "depth_meters": native.depth,
            "c2w": _payload(c2w, "c2w"), "camera_intrinsics": native.camera_intrinsics,
            **{name: _payload(value, name) for name, value in zip(
                ("world_view_transform", "camera_center", "gsplat_viewmat", "gsplat_camera_center"), common,
            )},
        },
        device="cpu", ready_event=event, lifetime_token=object(),
        control_token=PreparedIngressControlToken(8, source, None, None),
    )


def _record(db, lease):
    return db.record_prepared_candidate(
        source_identity=lease.source_identity, pose_c2w=lease.source_metadata["pose_c2w"],
        timestamp=lease.source_identity.timestamp_seconds, intrinsics=_intrinsics(),
        ready_source_reference=lease, modality_eligibility=lease.modality_eligibility, is_train=True,
    )


def _assert_geometry(binding, pose, revision):
    """Independent rigid inverse and hand-derived projection reference."""
    camera = binding.camera
    np.testing.assert_allclose(camera.c2w.numpy(), pose, atol=1e-6, err_msg="accepted camera C2W")
    w2c = np.eye(4, dtype=np.float32)
    w2c[:3, :3] = pose[:3, :3].T
    w2c[:3, 3] = -pose[:3, :3].T @ pose[:3, 3]
    offset = {"native": -0.125, "half": -0.25, "quarter": -0.5}[binding.prepared_tier]
    projection = np.array(
        ((2, 0, 0, 0), (0, 3, 0, 0), (offset, offset, 100 / 99.99, 1), (0, 0, -1 / 99.99, 0)),
        dtype=np.float32,
    )
    for actual, expected in (
        (camera.world_view_transform, w2c.T), (camera.gsplat.viewmat, w2c),
        (camera.camera_center, pose[:3, 3]), (camera.gsplat.camera_center, pose[:3, 3][None]),
        (camera.projection_matrix, projection), (camera.full_proj_transform, w2c.T @ projection),
    ):
        np.testing.assert_allclose(actual.numpy(), expected, atol=1e-6)
    np.testing.assert_array_equal(binding.metadata.pose_c2w, pose)
    np.testing.assert_array_equal(ReadyInsertionFrame.from_binding(binding).c2w, pose)
    assert binding.pose_revision == revision
    assert binding.versions == VERSIONS


@pytest.mark.parametrize("rejection", ["pending", "stale", "revision_gap", "foreign", "malformed", "retired"])
def test_rejected_publication_preserves_accepted_snapshot_and_cpu_history(rejection):
    transfer = DeviceTransferOwner("cpu")
    lease = _lease(transfer)
    db = FrameObservationDB(device="cpu")
    record = _record(db, lease)
    first = prepare_pose_correction(lease, _pose(), expected_revision=0, transfer_owner=transfer)
    assert db.commit_prepared_pose_correction(first, expected_revision=0)
    second = prepare_pose_correction(lease, _pose(4, 5), expected_revision=1, transfer_owner=transfer)
    expected_revision = 1
    if rejection == "pending":
        second = replace(second, completion_event=SimpleNamespace(query=lambda: False))
    elif rejection == "stale":
        expected_revision = 0
    elif rejection == "revision_gap":
        second = replace(second, revision=3)
    elif rejection == "foreign":
        second = replace(second, source_identity=replace(lease.source_identity, stream_epoch=3))
    elif rejection == "malformed":
        second = object()
    else:
        db._retire_unmaterializable(record.frame_id)
        assert db.unmaterializable_evictions == 1
    assert db.commit_prepared_pose_correction(second, expected_revision=expected_revision) is False
    assert db.prepared_pose_correction_for_source(lease.source_identity) is first
    np.testing.assert_array_equal(record.pose_c2w, _pose())


def test_incomplete_first_publication_keeps_original_pose():
    transfer = DeviceTransferOwner("cpu")
    lease = _lease(transfer)
    db = FrameObservationDB(device="cpu")
    record = _record(db, lease)
    correction = prepare_pose_correction(lease, _pose(), expected_revision=0, transfer_owner=transfer)
    pending = replace(correction, completion_event=SimpleNamespace(query=lambda: False))
    assert db.commit_prepared_pose_correction(pending, expected_revision=0) is False
    assert db.prepared_pose_correction_for_source(lease.source_identity) is None
    np.testing.assert_array_equal(record.pose_c2w, np.eye(4))


@pytest.mark.parametrize("tier", TIERS)
def test_repeated_corrections_bind_real_geometry_and_preserve_old_camera_and_pixels(tier):
    transfer = DeviceTransferOwner("cpu")
    lease = _lease(transfer)
    db = FrameObservationDB(device="cpu")
    record = _record(db, lease)
    pixels = lease.tier_payloads[tier].rgb.data.clone()
    depth = lease.tier_payloads[tier].depth.data.clone()
    first = prepare_pose_correction(lease, _pose(), expected_revision=0, transfer_owner=transfer)
    assert db.commit_prepared_pose_correction(first, expected_revision=0)
    old = bind_ready_camera(ReadySourceBinding(lease, VERSIONS, first), tier=tier)
    second = prepare_pose_correction(lease, _pose(4, 5), expected_revision=1, transfer_owner=transfer)
    assert db.commit_prepared_pose_correction(second, expected_revision=1)
    new = bind_ready_camera(ReadySourceBinding(lease, VERSIONS, second), tier=tier)
    _assert_geometry(old, _pose(), 1)
    _assert_geometry(new, _pose(4, 5), 2)
    np.testing.assert_array_equal(record.pose_c2w, _pose(4, 5))
    assert new.camera.c2w is second.c2w
    assert old.camera.c2w is first.c2w
    assert new.camera.original_image.data_ptr() == lease.tier_payloads[tier].rgb.data.data_ptr()
    assert new.camera._sensor_depth_cache is lease.tier_payloads[tier].depth.data
    assert torch.equal(lease.tier_payloads[tier].rgb.data, pixels)
    assert torch.equal(lease.tier_payloads[tier].depth.data, depth)
    np.testing.assert_array_equal(lease.source_metadata["pose_c2w"], np.eye(4))
    np.testing.assert_array_equal(lease.payloads["c2w"].data.numpy(), np.eye(4))


@pytest.mark.parametrize("tier", TIERS)
def test_new_generation_already_containing_absolute_correction_is_not_corrected_twice(tier):
    transfer = DeviceTransferOwner("cpu")
    original = _lease(transfer)
    correction = prepare_pose_correction(original, _pose(), expected_revision=0, transfer_owner=transfer)
    reloaded = _lease(transfer, pose=_pose(), generation=1)
    # Baseline binds successfully: payload, metadata and attestation all contain the new pose.
    _assert_geometry(bind_ready_camera(ReadySourceBinding(reloaded, VERSIONS), tier=tier), _pose(), 0)
    bound = bind_ready_camera(ReadySourceBinding(reloaded, VERSIONS, correction), tier=tier)
    _assert_geometry(bound, _pose(), 1)
    assert bound.camera.c2w is correction.c2w
    assert reloaded.generation > correction.source_generation
    assert torch.equal(reloaded.tier_payloads[tier].rgb.data, original.tier_payloads[tier].rgb.data)


def test_repeated_binding_borrows_geometry_without_materialization(monkeypatch):
    import eligsir.mapping.resolution_materialization as materialization

    transfer = DeviceTransferOwner("cpu")
    lease = _lease(transfer)
    correction = prepare_pose_correction(lease, _pose(), expected_revision=0, transfer_owner=transfer)
    binding = ReadySourceBinding(lease, VERSIONS, correction)
    handoffs = []
    original_handoff = transfer.handoff_to_compute

    def handoff(*events):
        handoffs.extend(events)
        original_handoff(*events)

    def forbidden(*_args, **_kwargs):
        pytest.fail("READY binding attempted materialization or transfer")

    with monkeypatch.context() as patch:
        patch.setattr(transfer, "handoff_to_compute", handoff)
        for event in (lease.ready_event, correction.completion_event):
            patch.setattr(event, "query", forbidden)
            patch.setattr(event, "wait", forbidden)
        for owner, names in (
            (DeviceTransferOwner, ("upload_packed", "prepare_source_camera_geometry", "prepare_source_tiers", "prepare_ready_calibration")),
            (np.linalg, ("inv",)), (torch.linalg, ("inv",)),
            (torch, ("inverse", "zeros", "empty", "eye", "tensor", "as_tensor", "from_numpy")),
            (torch.Tensor, ("to", "cpu", "cuda", "numpy", "item", "clone", "contiguous")),
            (torch.nn.functional, ("interpolate",)),
            (materialization, ("materialize_resolution_tensors",)), (Camera, ("__init__",)),
        ):
            for name in names:
                patch.setattr(owner, name, forbidden)
        bound = []
        for _ in range(2):
            for tier in TIERS:
                binding.handoff_to_compute(transfer)
                bound.append(bind_ready_camera(binding, tier=tier))
    assert handoffs == [lease.ready_event, correction.completion_event] * 6
    for item in bound:
        _assert_geometry(item, _pose(), 1)


@pytest.mark.parametrize("mismatch", ["source", "device", "tier", "calibration"])
def test_incompatible_correction_cannot_bind_or_mutate_accepted_state(mismatch):
    transfer = DeviceTransferOwner("cpu")
    lease = _lease(transfer)
    db = FrameObservationDB(device="cpu")
    record = _record(db, lease)
    accepted = prepare_pose_correction(lease, _pose(), expected_revision=0, transfer_owner=transfer)
    assert db.commit_prepared_pose_correction(accepted, expected_revision=0)
    if mismatch == "source":
        correction = replace(accepted, source_identity=replace(lease.source_identity, timestamp_ns=7))
    elif mismatch == "device":
        correction = replace(
            accepted, device="meta", c2w=accepted.c2w.to("meta"),
            common_geometry=tuple(value.to("meta") for value in accepted.common_geometry),
            tier_geometry={name: tuple(value.to("meta") for value in values) for name, values in accepted.tier_geometry.items()},
        )
    elif mismatch == "tier":
        correction = replace(
            accepted, tier_geometry={"native": accepted.tier_geometry["native"]},
            tier_calibrations={"native": accepted.tier_calibrations["native"]},
        )
    else:
        correction = replace(accepted, tier_calibrations={**accepted.tier_calibrations, "half": (4, 4, 9.0, 6.0, 1.5, 1.5)})
    with pytest.raises(ValueError, match=mismatch):
        bind_ready_camera(ReadySourceBinding(lease, VERSIONS, correction), tier="half")
    assert db.prepared_pose_correction_for_source(lease.source_identity) is accepted
    np.testing.assert_array_equal(record.pose_c2w, _pose())


class _ReplayBridge(PreparedRuntimeBridge):
    """Already-resident replay harness; production selection/requeue stays real."""

    def __init__(self, owner):
        self._ready_owner = owner


@pytest.mark.parametrize("state", ["accepted", "malformed", "lookup_bypassed"])
def test_adapter_renders_the_db_pose_with_real_binder_and_detects_lookup_bypass(monkeypatch, state):
    transfer = DeviceTransferOwner("cpu")
    leases = tuple(_lease(transfer, _source(index)) for index in range(4))
    db = FrameObservationDB(device="cpu")
    records = tuple(_record(db, lease) for lease in leases)
    correction = prepare_pose_correction(leases[0], _pose(), expected_revision=0, transfer_owner=transfer)
    assert db.commit_prepared_pose_correction(correction, expected_revision=0)
    owner = ReadySourceOwner(max_fresh=4, max_replay=8)
    for lease in leases:
        assert owner.offer(lease, lane=ReadySourceLane.REPLAY)
    sources = tuple(lease.source_identity for lease in leases)
    work = tuple(SimpleNamespace(source=source, resolution_tier=SimpleNamespace(value="native")) for source in sources)
    plan = SimpleNamespace(
        supervision_work=(), replay_work=work, supervision_sources=(), replay_sources=sources,
        supervision_constellation=SimpleNamespace(realized_batch_size=4, realized_camera_count=4),
        insertion=SimpleNamespace(enabled=False), replay_insertion=None,
        map_content_version=1, topology_version=2, parameter_version=3,
    )
    transactions = []

    def consume(*, ready_transaction):
        transactions.append(ready_transaction)
        assert tuple(item.source_identity for item in ready_transaction.bindings) == sources
        for index, binding in enumerate(ready_transaction.bindings):
            pose = _pose() if index == 0 else np.eye(4, dtype=np.float32)
            _assert_geometry(binding, pose, 1 if index == 0 else 0)
            np.testing.assert_array_equal(records[index].pose_c2w, pose)
            assert ready_transaction.cameras[sources[index]] is binding.camera
        return SimpleNamespace(executed=True, optimizer_steps=1, supervision_views=4)

    runtime = SimpleNamespace(_versions=lambda: (1, 2, 3), _observation_db=db, tick=consume)
    adapter = ReadyB4StreamingOptimizationAdapter(
        _ReplayBridge(owner), runtime=runtime, transfer_owner=transfer, warmup_complete=True,
    )
    inputs = ReadyB4StreamingStepInputs(None, plan)
    if state == "malformed":
        records[0].accepted_pose_correction = object()
        with pytest.raises(ValueError, match="malformed accepted pose correction"):
            adapter.tick(inputs)
        assert transactions == []
        assert db.commit_prepared_pose_correction(correction, expected_revision=0) is False
    else:
        assert adapter.tick(inputs).executed
        assert len(transactions) == 1
        if state == "lookup_bypassed":
            # Mutation proof: the same renderer/history assertions must fail when lookup is removed.
            monkeypatch.setattr(db, "prepared_pose_correction_for_source", lambda _source: None)
            with pytest.raises(AssertionError, match="accepted camera C2W"):
                adapter.tick(inputs)
            assert len(transactions) == 2
        assert records[0].accepted_pose_correction is correction
    np.testing.assert_array_equal(records[0].pose_c2w, _pose())
    assert owner.counters.in_flight == 0
