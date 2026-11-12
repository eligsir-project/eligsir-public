"""CPU contracts for opt-in device-resident ready-camera bootstrap."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from eligsir.contracts import (
    BufferOwnership,
    BufferRef,
    BufferSpec,
    CameraIntrinsics,
    MemoryDomain,
    ModalityEligibility,
    SourceIdentity,
)
from eligsir.mapping.observation_db import FrameObservationDB
from eligsir.runtime.config import CanonicalBootstrapPolicy, CanonicalRuntimePolicy
from eligsir.runtime.ready_camera_binder import PreparedCameraMetadata, ReadyCameraBinding
from eligsir.runtime.ready_only_tick import ReadyOnlyTickSelection, ReadySourceVersions
from eligsir.runtime.ready_source import ReadySourceLane, ReadySourceOwner
from eligsir.runtime.prepared_source import PreparedSourceLease


def _binding(*, ordinal: int, x: float = 0.0) -> ReadyCameraBinding:
    source = SourceIdentity("prepared-bootstrap", 2, ordinal, 1_700_000_000 + ordinal)
    rgb = torch.full((2, 3, 3), 0.5, dtype=torch.float32)
    depth = torch.full((2, 3), 2.0, dtype=torch.float32)
    c2w = torch.eye(4, dtype=torch.float32)
    c2w[0, 3] = x
    K = torch.tensor(((4.0, 0.0, 1.0), (0.0, 5.0, 0.5), (0.0, 0.0, 1.0)))
    camera = SimpleNamespace(
        original_image=rgb.permute(2, 0, 1),
        _sensor_depth_cache=depth,
        c2w=c2w,
        gsplat=SimpleNamespace(K=K),
        _source_identity=source,
    )
    metadata = PreparedCameraMetadata(
        source_identity=source,
        intrinsics=CameraIntrinsics(fx=4.0, fy=5.0, cx=1.0, cy=0.5, width=3, height=2, depth_scale=1000.0),
        pose_c2w=tuple(tuple(float(value) for value in row) for row in c2w.tolist()),
        rgb_shape=(2, 3, 3),
        depth_shape=(2, 3),
        depth_units="m",
        depth_aligned_to_rgb=True,
        modality_eligibility=ModalityEligibility(rgb_supervision=True, depth_supervision=True),
        tier="native",
        control_token=SimpleNamespace(source=source),
    )
    return ReadyCameraBinding(
        camera=camera,
        source_identity=source,
        modality_eligibility=metadata.modality_eligibility,
        control_token=metadata.control_token,
        versions=ReadySourceVersions(7, 11, 13),
        prepared_tier="native",
        metadata=metadata,
        lease=SimpleNamespace(),
    )


def test_prepared_candidate_records_authoritative_metadata_without_cold_pixels() -> None:
    binding = _binding(ordinal=9)
    db = FrameObservationDB(device="cpu")

    record = db.record_prepared_candidate(
        source_identity=binding.source_identity,
        pose_c2w=binding.metadata.pose_c2w,
        timestamp=binding.source_identity.timestamp_seconds,
        intrinsics=binding.metadata.intrinsics,
        ready_source_reference=binding,
        modality_eligibility=binding.modality_eligibility,
        is_train=True,
    )

    assert record.source_identity == binding.source_identity.fingerprint()
    assert record.pose_c2w[0, 3] == pytest.approx(0.0)
    assert db.get_cold_pixels(record.frame_id) is None
    assert db.cold_payload_occupancy == 0
    assert record.intrinsics == binding.metadata.intrinsics
    assert db.prepared_source_reference_for_source(binding.source_identity) is binding
    assert record.historical_eligible is False
    assert record.semantic_action == "prepared_device_transient"
    assert db.historical_candidate_for_source(binding.source_identity) is None


def test_runtime_bootstrap_records_prepared_source_as_transient_provenance() -> None:
    from eligsir.runtime.production_application import CanonicalMappingRuntime

    binding = _binding(ordinal=10)
    runtime = object.__new__(CanonicalMappingRuntime)
    runtime._observation_db = FrameObservationDB(device="cpu")
    runtime._register_historical_record = lambda *_args, **_kwargs: pytest.fail(
        "prepared bootstrap source entered the durable historical catalog"
    )
    camera = SimpleNamespace()

    runtime._record_ready_scene_candidate(camera, binding, is_train=True, is_holdout=False)

    frame_id = runtime._observation_db.frame_id_for_source(binding.source_identity)
    assert frame_id is not None
    record = runtime._observation_db.get_frame(frame_id)
    assert record is not None
    assert record.historical_eligible is False
    assert record.semantic_action == "prepared_device_transient"
    assert runtime._prepared_device_transient_retired_total == 1


def test_ready_train_record_allocates_evwb_before_any_optimizer_step() -> None:
    from eligsir.runtime.production_application import CanonicalMappingRuntime

    binding = _binding(ordinal=11)
    runtime = object.__new__(CanonicalMappingRuntime)
    runtime._observation_db = FrameObservationDB(device="cpu")
    runtime._optimization_policy = SimpleNamespace(
        streaming_train_evwb_enabled=True,
        streaming_train_evwb_reference_source="other-source",
    )
    runtime._register_historical_record = lambda *_args, **_kwargs: None

    runtime._record_ready_scene_candidate(binding.camera, binding, is_train=True, is_holdout=False)

    record = runtime._observation_db.get_frame(
        runtime._observation_db.frame_id_for_source(binding.source_identity)
    )
    assert record is not None and record.photometric_evwb is not None
    assert record.photometric_evwb.requires_grad


def test_scene_initializes_from_ready_bindings_without_legacy_cpu_bootstrap(monkeypatch, tmp_path) -> None:
    from eligsir.mapping import canonical_scene as scene_module
    from eligsir.mapping.canonical_scene import CanonicalStreamingScene

    bindings = (_binding(ordinal=0), _binding(ordinal=1, x=2.0))
    legacy_calls: list[str] = []

    def forbidden(*_args, **_kwargs):
        legacy_calls.append("legacy")
        raise AssertionError("prepared bootstrap entered a legacy CPU path")

    monkeypatch.setattr(scene_module, "canonical_bootstrap_point_cloud", forbidden)
    monkeypatch.setattr(scene_module, "canonical_camera_from_observation", forbidden)
    for binding in bindings:
        binding.camera.materialize_pending_image = forbidden
        binding.camera.rematerialize_native_image = forbidden

    class Gaussians:
        def create_from_pcd(self, *_args, **_kwargs):
            forbidden()

        def initialize_from_prepared_device_seed(self, seed, spatial_lr_scale, **kwargs):
            self.seed = seed
            self.scale = spatial_lr_scale
            self.kwargs = kwargs
            self.get_xyz = seed.points

    gaussians = Gaussians()
    scene = CanonicalStreamingScene(
        CanonicalRuntimePolicy(model_path=str(tmp_path), keyframe_window=4),
        CanonicalBootstrapPolicy(depth_stride=1, seed_voxel_size_m=0.0, init_scale_mode="fixed"),
        gaussians,
        (),
    )
    recorded = []

    scene.initialize_from_ready_bindings(bindings, record_fn=lambda camera, binding, **kwargs: recorded.append((camera, binding, kwargs)))

    assert legacy_calls == []
    assert gaussians.seed.points.device.type == "cpu"
    assert gaussians.seed.colors.device.type == "cpu"
    assert gaussians.seed.points.shape == (12, 3)
    assert scene.cameras_extent == pytest.approx(1.51327455)
    assert scene.train_cameras == [binding.camera for binding in bindings]
    assert [entry[1].source_identity for entry in recorded] == [binding.source_identity for binding in bindings]


def test_ready_bootstrap_rejects_non_native_or_missing_device_payloads(tmp_path) -> None:
    from eligsir.mapping.canonical_scene import CanonicalStreamingScene

    binding = _binding(ordinal=1)
    non_native = replace(binding, prepared_tier="half")
    scene = CanonicalStreamingScene(
        CanonicalRuntimePolicy(model_path=str(tmp_path)),
        CanonicalBootstrapPolicy(),
        SimpleNamespace(),
        (),
    )

    with pytest.raises(ValueError, match="native"):
        scene.initialize_from_ready_bindings((non_native,))


def test_runtime_accumulates_ready_bindings_then_requeues_after_bootstrap_event() -> None:
    from eligsir.runtime.production_application import CanonicalMappingRuntime

    runtime = object.__new__(CanonicalMappingRuntime)
    runtime.bootstrap_frames = 2
    runtime.initialized = False
    runtime._ready_bootstrap_bindings = []
    initialized = []
    requeued = []
    warmup = []
    runtime._initialize = lambda bindings: initialized.append(tuple(bindings))
    runtime.map_frame = lambda *_args, **_kwargs: pytest.fail("ready bootstrap entered map_frame")
    runtime._ready_only_tick_selector = SimpleNamespace(
        requeue_batch=lambda leases, *, ready_event, lane: requeued.append((leases, ready_event, lane))
    )
    runtime.mark_ready_source_warmup_complete = lambda: warmup.append(True)
    first = ReadyOnlyTickSelection((), 0, 0, False, False)
    second = ReadyOnlyTickSelection((), 0, 0, False, False)
    selected_bindings = {
        id(first): (_binding(ordinal=0),),
        id(second): (_binding(ordinal=1),),
    }
    runtime._bind_ready_bootstrap_cameras = lambda selection: selected_bindings[id(selection)]
    event = SimpleNamespace(query=lambda: True)

    assert runtime.bootstrap_ready_only_selection(first, bootstrap_event=event) is False
    assert initialized == []
    assert requeued == []

    assert runtime.bootstrap_ready_only_selection(second, bootstrap_event=event) is True
    assert [binding.source_identity.source_ordinal for binding in initialized[0]] == [0, 1]
    assert requeued == [((), event, "replay")]
    assert warmup == [True]


def test_runtime_ready_bootstrap_never_rounds_initialization_up_to_b4() -> None:
    """A five-source bootstrap initializes five bindings, even if a caller selected 4+4."""

    from eligsir.runtime.production_application import CanonicalMappingRuntime

    runtime = object.__new__(CanonicalMappingRuntime)
    runtime.bootstrap_frames = 5
    runtime.initialized = False
    runtime._ready_bootstrap_bindings = []
    runtime._ready_bootstrap_selections = []
    initialized = []
    runtime._initialize = lambda bindings: initialized.append(tuple(bindings))
    runtime._ready_only_tick_selector = SimpleNamespace(
        requeue_batch=lambda *_args, **_kwargs: None
    )
    runtime.mark_ready_source_warmup_complete = lambda: None
    first = ReadyOnlyTickSelection((), 0, 0, False, False)
    second = ReadyOnlyTickSelection((), 0, 0, False, False)
    selected = {
        id(first): tuple(_binding(ordinal=index) for index in range(4)),
        id(second): tuple(_binding(ordinal=index) for index in range(4, 8)),
    }
    runtime._bind_ready_bootstrap_cameras = lambda selection: selected[id(selection)]
    event = SimpleNamespace(query=lambda: True)

    assert runtime.bootstrap_ready_only_selection(first, bootstrap_event=event) is False
    assert runtime.bootstrap_ready_only_selection(second, bootstrap_event=event) is True
    assert [binding.source_identity.source_ordinal for binding in initialized[0]] == [
        0,
        1,
        2,
        3,
        4,
    ]


def test_ready_bootstrap_defers_replay_successor_until_commit_completion(
    monkeypatch,
) -> None:
    """The causal owner must clear COMMIT debt before minting replay authority."""

    from eligsir.runtime.production_application import CanonicalMappingRuntime

    runtime = object.__new__(CanonicalMappingRuntime)
    runtime.bootstrap_frames = 1
    runtime.initialized = False
    runtime._ready_bootstrap_bindings = []
    runtime._ready_bootstrap_selections = []
    plan = object()
    runtime._prepared_ready_candidate_work = (object(), plan)
    runtime._initialize = lambda _bindings: None
    runtime._ready_only_tick_selector = SimpleNamespace(requeue_batch=lambda *_args, **_kwargs: None)
    runtime.mark_ready_source_warmup_complete = lambda: None
    selected = ReadyOnlyTickSelection((), 0, 0, False, False)
    runtime._bind_ready_bootstrap_cameras = lambda _selection: (_binding(ordinal=0),)
    calls = []
    monkeypatch.setattr(
        CanonicalMappingRuntime,
        "_advance_prepared_ready_plan",
        lambda self, plan: calls.append((self, plan)),
        raising=False,
    )

    assert runtime.bootstrap_ready_only_selection(
        selected, bootstrap_event=SimpleNamespace(query=lambda: True)
    ) is True
    assert calls == []


def test_prepared_ready_plan_advance_replaces_fresh_candidate_with_idle_replay() -> None:
    from eligsir.runtime.production_application import CanonicalMappingRuntime

    runtime = object.__new__(CanonicalMappingRuntime)
    prebootstrap_snapshot = object()
    prebootstrap_plan = object()
    idle_snapshot = object()
    replay_plan = object()
    runtime._prepared_ready_candidate_work = None
    runtime._prepared_ready_frozen_work = (prebootstrap_snapshot, prebootstrap_plan)
    runtime._fresh_idle_snapshot = lambda snapshot: (
        idle_snapshot if snapshot is prebootstrap_snapshot else pytest.fail("wrong snapshot")
    )
    runtime._idle_replay_plan = lambda snapshot, plan: (
        replay_plan
        if snapshot is idle_snapshot and plan is prebootstrap_plan
        else pytest.fail("wrong prebootstrap authority")
    )

    CanonicalMappingRuntime._advance_prepared_ready_plan(runtime, prebootstrap_plan)

    assert runtime._prepared_ready_frozen_work is None
    assert runtime._prepared_ready_candidate_work == (idle_snapshot, replay_plan)


def test_ready_bootstrap_remaining_defaults_to_empty_pending_list() -> None:
    from eligsir.runtime.production_application import CanonicalMappingRuntime

    runtime = object.__new__(CanonicalMappingRuntime)
    runtime.bootstrap_frames = 5

    assert runtime.ready_bootstrap_remaining() == 5


def test_prepared_session_commit_sink_bootstraps_exact_five_sources_in_4_plus_1() -> None:
    """Pre-handoff COMMIT retention survives split prepared bootstrap ownership."""

    from eligsir.contracts.control import ControlLimits, ControlSnapshot, FreshnessMode
    from eligsir.contracts import SensorPacket
    from eligsir.mapping.admission.observation_owner import ObservationOwner
    from eligsir.mapping.scheduler.backpressure import (
        BackpressureAction,
        BackpressureDecision,
    )
    from eligsir.runtime.prepared_ingress import PreparedSourceIngressHandoff
    from eligsir.runtime.prepared_source import PreparedSourceLease
    from eligsir.runtime.production_application import CanonicalMappingRuntime
    from eligsir.runtime.session import MappingSession
    from eligsir.tracking.providers import ExternalTrackingProvider
    from eligsir.tracking.providers import observation_from_pose_matrix

    def source_packet(ordinal: int) -> SensorPacket:
        source = _source(ordinal)
        tracking = observation_from_pose_matrix(source, np.eye(4), provider="test")
        return SensorPacket(
            source=source,
            rgb=BufferRef.from_numpy(
                np.full((2, 3, 3), 128, dtype=np.uint8), encoding="rgb8"
            ),
            depth=BufferRef.from_numpy(
                np.ones((2, 3), dtype=np.float32), encoding="32FC1", units="m"
            ),
            intrinsics=CameraIntrinsics(
                fx=2.0, fy=2.0, cx=1.0, cy=1.0, width=3, height=2, depth_scale=1000.0
            ),
            tracking=tracking,
        )

    def snapshot(packet, observation, _ingress):
        return ControlSnapshot(
            snapshot_id=1,
            control_version=1,
            source=packet.source,
            tracking=observation,
            pending_sources=(packet.source,),
            newest_frontier=packet.source,
            freshness_mode=FreshnessMode.HEALTHY,
            pending_oldest_age_ns=0,
            ingress_queue_depth=0,
            candidate_queue_depth=0,
            transfer_queue_depth=0,
            gpu_queue_depth=0,
            map_content_version=0,
            topology_version=0,
            parameter_version=0,
            gaussian_count=0,
            optimizer_steps_completed=0,
            limits=ControlLimits(1, 0, 1, 1, 1, 1 << 20),
            supervision_candidates=(packet.source,),
        )

    admission = ObservationOwner(
        lambda value: BackpressureDecision(
            frame_id=value.source.source_ordinal,
            action=BackpressureAction.ADMIT,
            reason="soft_all_capacity_available",
            admit_supervision=True,
            allow_geometry_insertion=False,
        )
    )
    retained = []
    recorded = []

    class SinkRuntime:
        def retain_prepared_source(self, item, *, lane):
            retained.append((item, lane))
            return True

        def record_retained_prepared_commit(self, item):
            recorded.append(item)

    sink_runtime = SinkRuntime()
    handoff = PreparedSourceIngressHandoff(
        capacity=5,
        commit_sink=lambda item: (
            sink_runtime.retain_prepared_source(item, lane=ReadySourceLane.FRESH),
            sink_runtime.record_retained_prepared_commit(item),
        ),
    )
    session = MappingSession(
        ExternalTrackingProvider(),
        observation_owner=admission,
        control_snapshot_provider=snapshot,
        prepared_mode=True,
        prepared_ingress_handoff=handoff,
    )

    for ordinal in range(5):
        assert session.submit(source_packet(ordinal)).accepted
    assert len(retained) == 5
    assert len(recorded) == 5

    def lease(source):
        payload = BufferRef(
            data=torch.zeros(1),
            spec=BufferSpec(
                (1,), "float32", MemoryDomain.OTHER_DEVICE, "tensor", device="cpu"
            ),
            ownership=BufferOwnership.BORROWED,
            owner_token=object(),
        )
        return PreparedSourceLease(
            source_identity=source,
            modality_eligibility=ModalityEligibility(
                rgb_supervision=True, depth_supervision=True
            ),
            source_metadata={"width": 1, "height": 1},
            tier_preparation_available=("native",),
            borrowed_device_payloads={"rgb": payload},
            device="cpu",
        )

    selections = [
        ReadyOnlyTickSelection(
            tuple(lease(_source(index)) for index in range(4)), 4, 0, False, False
        ),
        ReadyOnlyTickSelection((lease(_source(4)),), 1, 0, False, False),
    ]
    initialized = []
    runtime = object.__new__(CanonicalMappingRuntime)
    runtime.bootstrap_frames = 5
    runtime.initialized = False
    runtime._ready_only_tick_selector = SimpleNamespace(
        requeue_batch=lambda *_args, **_kwargs: None
    )
    runtime._bind_ready_bootstrap_cameras = lambda selection: tuple(
        SimpleNamespace(source_identity=item.source_identity) for item in selection.leases
    )
    runtime._initialize = lambda bindings: initialized.append(tuple(bindings))
    runtime.mark_ready_source_warmup_complete = lambda: None

    assert runtime.bootstrap_ready_only_selection(
        selections[0], bootstrap_event=SimpleNamespace(query=lambda: True)
    ) is False
    assert runtime.bootstrap_ready_only_selection(
        selections[1], bootstrap_event=SimpleNamespace(query=lambda: True)
    ) is True
    assert len(initialized) == 1
    assert [binding.source_identity.source_ordinal for binding in initialized[0]] == [
        0, 1, 2, 3, 4
    ]


def test_ready_bootstrap_handoffs_lease_before_binding_device_payloads(monkeypatch) -> None:
    from eligsir.runtime.production_application import CanonicalMappingRuntime

    runtime = object.__new__(CanonicalMappingRuntime)
    runtime.bootstrap_frames = 1
    runtime.initialized = False
    runtime._ready_bootstrap_bindings = []
    runtime._ready_bootstrap_selections = []
    runtime._transfer_owner = object()
    runtime._ready_only_tick_selector = SimpleNamespace(
        preflight_requeue_batch=lambda *_args, **_kwargs: None,
        requeue_batch=lambda *_args, **_kwargs: None,
    )
    runtime._initialize = lambda _bindings: None
    runtime.mark_ready_source_warmup_complete = lambda: None
    calls = []
    source_binding = SimpleNamespace(
        handoff_to_compute=lambda owner: calls.append(("handoff", owner)),
        lease=SimpleNamespace(source_identity=_source(0)),
    )
    runtime.bind_ready_only_sources = lambda _selection: (source_binding,)
    runtime.bind_ready_only_cameras = lambda _selection: pytest.fail(
        "bootstrap bound a camera before handoff"
    )
    monkeypatch.setattr(
        "eligsir.runtime.ready_camera_binder.bind_ready_camera",
        lambda binding, **_kwargs: calls.append(("bind", binding)) or _binding(ordinal=0),
    )
    selection = ReadyOnlyTickSelection((), 0, 0, False, False)

    assert runtime.bootstrap_ready_only_selection(
        selection, bootstrap_event=SimpleNamespace(query=lambda: True)
    ) is True

    assert calls[0] == ("handoff", runtime._transfer_owner)
    assert calls[1] == ("bind", source_binding)


def _ready_lease(ordinal: int) -> PreparedSourceLease:
    source = _source(ordinal)
    payload = BufferRef(
        data=torch.zeros(1),
        spec=BufferSpec((1,), "float32", MemoryDomain.OTHER_DEVICE, "tensor", device="cpu"),
        ownership=BufferOwnership.BORROWED,
        owner_token=object(),
    )
    return PreparedSourceLease(
        source_identity=source,
        modality_eligibility=ModalityEligibility(rgb_supervision=True),
        source_metadata={"width": 1, "height": 1},
        tier_preparation_available=("native",),
        borrowed_device_payloads={"rgb": payload},
        device="cpu",
    )


def _source(ordinal: int) -> SourceIdentity:
    return SourceIdentity("ready-bootstrap-owner", 0, ordinal, 1_800_000_000 + ordinal)


def test_ready_bootstrap_replay_capacity_failure_precedes_initialization() -> None:
    from eligsir.runtime.production_application import CanonicalMappingRuntime
    from eligsir.runtime.ready_only_tick import ReadyOnlyTickSelector

    owner = ReadySourceOwner(max_fresh=1, max_replay=1)
    producer = owner.claim_producer("producer")
    fresh = _ready_lease(0)
    replay = _ready_lease(1)
    assert producer.offer(fresh, lane=ReadySourceLane.FRESH)
    assert producer.offer(replay, lane=ReadySourceLane.REPLAY)
    selector = ReadyOnlyTickSelector(owner, batch_size=1)
    selection = selector.select(fresh_sources=(fresh.source_identity,))
    assert selection.leases == (fresh,)

    runtime = object.__new__(CanonicalMappingRuntime)
    runtime.bootstrap_frames = 1
    runtime.initialized = False
    runtime._ready_bootstrap_bindings = []
    runtime._ready_bootstrap_selections = []
    runtime._ready_only_tick_selector = selector
    runtime._bind_ready_bootstrap_cameras = lambda _selection: (_binding(ordinal=0),)
    runtime._initialize = lambda _bindings: pytest.fail("capacity failure initialized the model")

    with pytest.raises(RuntimeError, match="target lane is full"):
        runtime.bootstrap_ready_only_selection(
            selection, bootstrap_event=SimpleNamespace(query=lambda: True)
        )

    assert owner.counters.in_flight == 1
    assert owner.pending_count == 1
    assert runtime._ready_bootstrap_bindings == []
    assert runtime._ready_bootstrap_selections == []


def test_ready_bootstrap_requeue_waits_for_the_bootstrap_event() -> None:
    from eligsir.runtime.production_application import CanonicalMappingRuntime
    from eligsir.runtime.ready_only_tick import ReadyOnlyTickSelector

    owner = ReadySourceOwner(max_fresh=1, max_replay=1)
    producer = owner.claim_producer("producer")
    lease = _ready_lease(0)
    assert producer.offer(lease, lane=ReadySourceLane.FRESH)
    selector = ReadyOnlyTickSelector(owner, batch_size=1)
    selection = selector.select(fresh_sources=(lease.source_identity,))
    event = SimpleNamespace(complete=False, query=lambda: event.complete)

    runtime = object.__new__(CanonicalMappingRuntime)
    runtime.bootstrap_frames = 1
    runtime.initialized = False
    runtime._ready_bootstrap_bindings = []
    runtime._ready_bootstrap_selections = []
    runtime._ready_only_tick_selector = selector
    runtime._bind_ready_bootstrap_cameras = lambda _selection: (_binding(ordinal=0),)
    runtime._initialize = lambda _bindings: None
    runtime.mark_ready_source_warmup_complete = lambda: None

    assert runtime.bootstrap_ready_only_selection(selection, bootstrap_event=event) is True
    assert selector.select(replay_sources=(lease.source_identity,)).leases == ()

    event.complete = True
    assert selector.select(replay_sources=(lease.source_identity,)).leases == (lease,)
