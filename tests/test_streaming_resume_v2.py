import copy
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from eligsir.checkpointing.streaming_resume import (
    build_resume_envelope,
    capture_rng_state,
    restore_rng_state,
    validate_resume_envelope,
)
from eligsir.mapping.scheduler.frame_scheduler import FrameScheduler
from eligsir.mapping.scheduler.keyframes import KeyframeAdmitter, KeyframeAdmitterConfig


class _Owner:
    def __init__(self, state):
        self.state = state

    def resume_state_dict(self):
        return self.state


def _identity(**overrides):
    identity = {
        "source_hash": "source-a",
        "holdout_hash": "holdout-a",
        "resolved_config_hash": "config-a",
        "runtime_hash": "runtime-a",
    }
    identity.update(overrides)
    return identity


def test_rng_state_round_trip_restores_python_numpy_and_torch_cpu():
    random.seed(17)
    np.random.seed(23)
    torch.manual_seed(29)
    state = capture_rng_state()

    expected = (random.random(), np.random.rand(3), torch.rand(3))
    random.seed(101)
    np.random.seed(103)
    torch.manual_seed(107)

    restore_rng_state(state)
    actual = (random.random(), np.random.rand(3), torch.rand(3))

    assert actual[0] == expected[0]
    np.testing.assert_array_equal(actual[1], expected[1])
    assert torch.equal(actual[2], expected[2])


def test_rng_state_round_trip_restores_every_available_gpu_stream():
    if not torch.cuda.is_available():
        pytest.skip("no Torch GPU available")

    torch.cuda.manual_seed_all(31)
    state = capture_rng_state()
    expected = [torch.rand(3, device=i).cpu() for i in range(torch.cuda.device_count())]
    torch.cuda.manual_seed_all(109)

    restore_rng_state(state)
    actual = [torch.rand(3, device=i).cpu() for i in range(torch.cuda.device_count())]

    assert len(state["torch_gpu"]) == torch.cuda.device_count()
    assert all(torch.equal(got, want) for got, want in zip(actual, expected))


def test_build_resume_envelope_captures_owner_states_without_aliasing():
    owner = _Owner({"counter": 7, "tensor": torch.tensor([2.0, 3.0])})

    payload = build_resume_envelope(
        identity=_identity(),
        owners={"mcmc": owner},
    )
    owner.state["counter"] = 99
    owner.state["tensor"].zero_()

    assert payload["schema"] == "streaming_checkpoint_v2"
    assert payload["version"] == 2
    assert payload["identity"] == _identity()
    assert payload["owners"]["mcmc"]["counter"] == 7
    assert torch.equal(payload["owners"]["mcmc"]["tensor"], torch.tensor([2.0, 3.0]))


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda p: p.update(schema="streaming_checkpoint_v1"), "schema"),
        (lambda p: p.update(version=3), "version"),
        (lambda p: p.update(identity={"source_hash": "other"}), "identity"),
        (lambda p: p["owners"].pop("mcmc"), "owner"),
        (lambda p: p["rng"].pop("torch_cpu"), "rng"),
    ],
)
def test_validate_resume_envelope_rejects_inexact_payloads(mutation, match):
    identity = _identity()
    payload = build_resume_envelope(identity, {"mcmc": _Owner({"counter": 7})})
    broken = copy.deepcopy(payload)
    mutation(broken)

    with pytest.raises(ValueError, match=match):
        validate_resume_envelope(broken, identity, {"mcmc"})


def test_restore_rng_state_rejects_a_different_gpu_stream_count(monkeypatch):
    state = capture_rng_state()
    state["torch_gpu"] = [torch.tensor([1], dtype=torch.uint8)]
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)

    with pytest.raises(ValueError, match="GPU RNG stream count"):
        restore_rng_state(state)


def test_restore_rng_validation_failure_does_not_mutate_any_global_stream():
    python_before = copy.deepcopy(random.getstate())
    numpy_before = copy.deepcopy(np.random.get_state())
    torch_before = torch.get_rng_state().clone()
    broken = capture_rng_state()
    broken["torch_cpu"] = torch.tensor([1.0])

    with pytest.raises(ValueError, match="Torch CPU"):
        restore_rng_state(broken)

    assert random.getstate() == python_before
    numpy_after = np.random.get_state()
    assert numpy_after[0] == numpy_before[0]
    np.testing.assert_array_equal(numpy_after[1], numpy_before[1])
    assert numpy_after[2:] == numpy_before[2:]
    assert torch.equal(torch.get_rng_state(), torch_before)


def test_build_resume_envelope_rejects_nonportable_owner_state():
    owner = _Owner({"camera_object": object()})

    with pytest.raises(ValueError, match="unsupported exact-resume state type"):
        build_resume_envelope(_identity(), {"keyframes": owner})


@pytest.mark.parametrize(
    "identity",
    [
        {},
        {"source_hash": "source-a"},
        {**_identity(), "extra_hash": "extra"},
        _identity(runtime_hash=3),
        _identity(holdout_hash="  "),
    ],
)
def test_build_resume_envelope_rejects_malformed_identity(identity):
    with pytest.raises(ValueError, match="identity"):
        build_resume_envelope(identity, {"mcmc": _Owner({"counter": 7})})


def test_iter_based_frame_scheduler_exact_state_round_trip():
    scheduler = FrameScheduler(fps=30.0, steps_per_frame=3, ingestion_mode="iter_based")
    assert scheduler.should_release(1)
    scheduler.mark_released()
    scheduler.mark_mapped()
    scheduler.set_next_interval(1, 7)

    payload = scheduler.resume_state_dict()
    restored = FrameScheduler(fps=30.0, steps_per_frame=3, ingestion_mode="iter_based")
    restored.load_resume_state_dict(payload)

    assert payload["owner_schema"] == "frame_scheduler_v1"
    assert restored.frames_released == 1
    assert restored.frames_mapped == 1
    assert restored.should_release(7) is False
    assert restored.should_release(8) is True


def test_frame_scheduler_resume_rejects_wall_time_and_config_drift():
    scheduler = FrameScheduler(fps=30.0, steps_per_frame=3, ingestion_mode="iter_based")
    payload = scheduler.resume_state_dict()

    with pytest.raises(ValueError, match="configuration"):
        FrameScheduler(fps=30.0, steps_per_frame=4, ingestion_mode="iter_based").load_resume_state_dict(payload)

    wallclock = FrameScheduler(fps=30.0, ingestion_mode="wallclock_strict")
    with pytest.raises(ValueError, match="iter_based"):
        wallclock.resume_state_dict()


def _keyframe(index: int, x: float):
    pose = np.eye(4, dtype=np.float32)
    pose[0, 3] = x
    return SimpleNamespace(index=index, c2w=pose, timestamp=index / 30.0)


def test_hybrid_keyframe_exact_state_round_trip_preserves_next_decision():
    config = KeyframeAdmitterConfig(
        min_translation_m=0.05,
        min_rotation_deg=5.0,
        max_gap=3,
        motion_pregate=True,
    )
    source = KeyframeAdmitter(config=config)
    source.admit(cam=None, frame=_keyframe(0, 0.0))
    source.pregate(_keyframe(1, 0.01))
    payload = source.resume_state_dict()

    restored = KeyframeAdmitter(config=KeyframeAdmitterConfig(
        min_translation_m=0.05,
        min_rotation_deg=5.0,
        max_gap=3,
        motion_pregate=True,
    ))
    restored.load_resume_state_dict(payload)

    source_decision, _ = source.admit(cam=None, frame=_keyframe(4, 0.01))
    restored_decision, _ = restored.admit(cam=None, frame=_keyframe(4, 0.01))
    assert payload["owner_schema"] == "hybrid_keyframe_admitter_v1"
    assert source_decision.admitted == restored_decision.admitted
    assert source_decision.reason == restored_decision.reason
    assert source_decision.gap == restored_decision.gap


def test_hybrid_keyframe_resume_rejects_config_drift_and_pending_scratch():
    source = KeyframeAdmitter(config=KeyframeAdmitterConfig(max_gap=3))
    payload = source.resume_state_dict()

    with pytest.raises(ValueError, match="configuration"):
        KeyframeAdmitter(
            config=KeyframeAdmitterConfig(max_gap=4)
        ).load_resume_state_dict(payload)

    source._pending_luma = 0.5
    with pytest.raises(ValueError, match="pending per-call scratch"):
        source.resume_state_dict()

    payload["keyframes_admitted"] = -1
    with pytest.raises(ValueError, match="keyframes_admitted"):
        KeyframeAdmitter(
            config=KeyframeAdmitterConfig(max_gap=3)
        ).load_resume_state_dict(payload)


def test_real_hybrid_keyframe_owner_is_portable_in_v2_envelope():
    owner = KeyframeAdmitter(config=KeyframeAdmitterConfig(max_gap=3))
    owner.note_train_frame(_keyframe(0, 0.0))

    envelope = build_resume_envelope(_identity(), {"admission": owner})

    payload = envelope["owners"]["admission"]
    assert payload["owner_schema"] == "hybrid_keyframe_admitter_v1"
    assert payload["last_train_c2w"] == np.eye(4, dtype=np.float32).tolist()


def _exact_causal_scene():
    from geometry.surface_cells import SurfaceCellIndex
    from scene.streaming_scene import StreamingScene
    from eligsir.mapping.stores.keyframe_store import KeyframeStore
    from eligsir.mapping.stores.occupancy_index import OccupancyIndex

    scene = object.__new__(StreamingScene)
    scene.args = SimpleNamespace(
        gpu_occupancy=False,
        streaming_extent_growth_ceiling_multiplier=1.2,
        streaming_extent_outlier_corroboration=2,
    )
    scene.gaussians = SimpleNamespace(
        get_xyz=torch.empty((0, 3)), _current_submap_id=0
    )
    scene._keyframe_store = KeyframeStore()
    scene._keyframe_store.train_cameras = [
        SimpleNamespace(camera_center=torch.tensor([0.0, 0.0, 0.0])),
        SimpleNamespace(camera_center=torch.tensor([1.0, 0.0, 0.0])),
        SimpleNamespace(camera_center=torch.tensor([5.0, 0.0, 0.0])),
    ]
    scene._keyframe_store.cameras_extent = 1.1
    scene._occupancy_index = OccupancyIndex(voxel_size=0.5)
    xyz = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
    scene._occupancy_index.keys = np.sort(scene._xyz_to_keys(xyz, 0.5))
    scene._occupancy_index.keys_gpu = torch.as_tensor(
        scene._occupancy_index.keys.copy(), dtype=torch.int64
    )
    scene._surface_cell_index = SurfaceCellIndex(voxel_size=0.5, normal_bins=8)
    scene._surface_cell_index.add(
        xyz, np.asarray([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    )
    scene._extent_update_counter = 7
    scene._extent_pending_growth = 2.5
    scene._extent_pending_corroborations = 1
    scene._extent_pending_driver_idx = 0
    scene._extent_outlier_rejections_total = 3
    scene._pending_extent_update = False
    return scene


def _exact_causal_scene_owner(scene):
    from eligsir.checkpointing.exact_refinement_resume import (
        ExactCausalForkSceneState,
    )
    from eligsir.utils.submap_manager import SubmapManager

    manager = SubmapManager(frames_per_submap=50)
    return ExactCausalForkSceneState(
        scene, submap_manager=manager, gaussians=scene.gaussians
    )


def test_exact_causal_scene_owner_round_trip_preserves_next_queries_and_extent_decision():
    from eligsir.checkpointing.exact_refinement_resume import (
        ExactCausalForkSceneState,
    )

    source = _exact_causal_scene()
    restored = _exact_causal_scene()
    restored._occupancy_index.keys = np.empty(0, dtype=np.int64)
    restored._occupancy_index.keys_gpu = None
    restored._surface_cell_index._keys = np.empty(0, dtype=np.int64)
    restored._extent_update_counter = 0
    restored._extent_pending_growth = None
    restored._extent_pending_corroborations = 0
    restored._extent_pending_driver_idx = None
    restored._extent_outlier_rejections_total = 0

    payload = _exact_causal_scene_owner(source).resume_state_dict()
    _exact_causal_scene_owner(restored).load_resume_state_dict(payload)

    query = np.asarray([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=np.float32)
    normals = np.asarray([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    np.testing.assert_array_equal(
        source.check_occupancy(query, 0.5), restored.check_occupancy(query, 0.5)
    )
    np.testing.assert_array_equal(
        source.check_surface_cells(query, normals),
        restored.check_surface_cells(query, normals),
    )

    source.update_cameras_extent()
    restored.update_cameras_extent()
    for name in (
        "cameras_extent",
        "_extent_update_counter",
        "_extent_pending_growth",
        "_extent_pending_corroborations",
        "_extent_pending_driver_idx",
        "_extent_outlier_rejections_total",
    ):
        assert getattr(restored, name) == getattr(source, name)


def test_exact_causal_scene_owner_rejects_pending_extent_and_malformed_index():
    from eligsir.checkpointing.exact_refinement_resume import (
        ExactCausalForkSceneState,
    )

    source = _exact_causal_scene()
    owner = _exact_causal_scene_owner(source)
    payload = owner.resume_state_dict()
    payload["occupancy_cpu_keys"] = torch.tensor([2, 1], dtype=torch.int64)
    with pytest.raises(ValueError, match="sorted unique"):
        owner.load_resume_state_dict(payload)

    source._pending_extent_update = True
    with pytest.raises(ValueError, match="pending extent"):
        owner.resume_state_dict()


def test_exact_causal_scene_owner_preserves_next_submap_boundary():
    from eligsir.checkpointing.exact_refinement_resume import (
        ExactCausalForkSceneState,
    )
    from eligsir.utils.submap_manager import SubmapManager

    source_scene = _exact_causal_scene()
    source_manager = SubmapManager(frames_per_submap=2)
    source_manager.set_submap0_anchor(np.eye(4, dtype=np.float32))
    source_manager.maybe_start_new_submap(0, gaussians=source_scene.gaussians)
    source_manager.maybe_start_new_submap(
        1, gaussians=source_scene.gaussians, anchor_c2w=np.eye(4, dtype=np.float32)
    )
    source_manager.maybe_start_new_submap(2, gaussians=source_scene.gaussians)

    restored_scene = _exact_causal_scene()
    restored_manager = SubmapManager(frames_per_submap=2)
    payload = ExactCausalForkSceneState(
        source_scene,
        submap_manager=source_manager,
        gaussians=source_scene.gaussians,
    ).resume_state_dict()
    ExactCausalForkSceneState(
        restored_scene,
        submap_manager=restored_manager,
        gaussians=restored_scene.gaussians,
    ).load_resume_state_dict(payload)

    next_anchor = np.eye(4, dtype=np.float32)
    next_anchor[0, 3] = 4.0
    assert source_manager.maybe_start_new_submap(
        3, gaussians=source_scene.gaussians, anchor_c2w=next_anchor
    )
    assert restored_manager.maybe_start_new_submap(
        3, gaussians=restored_scene.gaussians, anchor_c2w=next_anchor
    )
    assert restored_manager._frame_count == source_manager._frame_count
    assert restored_manager.current_submap_id == source_manager.current_submap_id
    assert restored_scene.gaussians._current_submap_id == source_scene.gaussians._current_submap_id
    np.testing.assert_array_equal(
        restored_manager.get_submap_raw_transform(2),
        source_manager.get_submap_raw_transform(2),
    )


def _coverage_admitter(observation_db):
    config = KeyframeAdmitterConfig(
        coverage_gain_gate=True,
        min_new_voxel_frac=0.5,
        max_gap=1000,
        coverage_gain_depth_stride=1,
        motion_pregate=False,
    )
    return KeyframeAdmitter(
        config=config,
        render_fn=lambda cam, render_depth=False: {"alpha": torch.ones(1, 4, 4)},
        gaussians=SimpleNamespace(
            get_xyz=torch.empty((0, 3)), gaussian_uid=torch.empty(0, dtype=torch.int64)
        ),
        occupancy_fn=lambda pts, vs, check_neighbors=True: np.ones(len(pts), dtype=bool),
        load_depth_m=lambda frame: np.ones((4, 4), dtype=np.float32),
        observation_db=observation_db,
    )


def _coverage_candidate(index, pose, uids):
    frame = SimpleNamespace(index=index, c2w=pose.astype(np.float32))
    cam = SimpleNamespace(
        c2w=pose.astype(np.float32),
        R=pose[:3, :3].astype(np.float32),
        T=(-pose[:3, :3].T @ pose[:3, 3]).astype(np.float32),
        world_view_transform=torch.eye(4),
        fx=2.0,
        fy=2.0,
        cx=2.0,
        cy=2.0,
        _observed_gaussian_uids=torch.as_tensor(uids, dtype=torch.int64),
    )
    return cam, frame


def test_exact_causal_observation_owner_preserves_rejected_history_next_decision():
    from eligsir.checkpointing.exact_refinement_resume import (
        ExactCausalForkObservationDBState,
    )
    from eligsir.mapping.observation_db import (
        BufferStage,
        FrameObservationDB,
        FrameRecord,
    )

    source = FrameObservationDB(device="cpu")
    historical_pose = np.diag([-1.0, 1.0, -1.0, 1.0]).astype(np.float32)
    source.add_frame(
        FrameRecord(
            frame_id=7,
            pose_c2w=historical_pose,
            timestamp=0.7,
            submap_id=2,
            buffer_stage=BufferStage.EVICTED_REDUNDANT,
            visible_gaussian_uids=torch.tensor([1, 2, 3, 4], dtype=torch.int64),
            resolution_cache={},
            resolution_cache_absent_reason="not_train_frame",
            source_identity="source:7",
            source_ordinal=7,
        )
    )
    source._source_identity_to_frame_id["source:7"] = 7
    restored = FrameObservationDB(device="cpu")
    ExactCausalForkObservationDBState(restored).load_resume_state_dict(
        ExactCausalForkObservationDBState(source).resume_state_dict()
    )

    candidate_pose = np.eye(4, dtype=np.float32)
    source_cam, source_frame = _coverage_candidate(8, candidate_pose, [1, 2, 3, 4])
    restored_cam, restored_frame = _coverage_candidate(8, candidate_pose, [1, 2, 3, 4])
    empty_cam, empty_frame = _coverage_candidate(8, candidate_pose, [1, 2, 3, 4])
    source_decision, _ = _coverage_admitter(source).admit(source_cam, source_frame)
    restored_admitter = _coverage_admitter(restored)
    restored_decision, _ = restored_admitter.admit(restored_cam, restored_frame)
    empty_decision, _ = _coverage_admitter(FrameObservationDB(device="cpu")).admit(
        empty_cam, empty_frame
    )

    assert source_decision.admitted and source_decision.reason == "db_diverse_revisit"
    assert restored_decision == source_decision
    assert not empty_decision.admitted
    assert restored_admitter._observation_db is restored
    restored_record = restored.get_frame(7)
    assert restored_record.buffer_stage is BufferStage.EVICTED_REDUNDANT
    assert restored_record.source_identity == "source:7"
    assert restored_record.source_ordinal == 7


def test_exact_causal_observation_owner_rejects_unresolved_prepared_commit_obligation():
    """Exact fork capture must not silently discard an unsupervised COMMIT.

    The exact observation-state schema deliberately does not yet serialize the
    prepared COLD route (tracking observation, disposition, control token and
    pixel/reload handle) required to make this obligation actionable after
    resume.  Capturing only the opaque marker would manufacture an apparently
    resumable but permanently unserviceable COMMIT, so capture must fail until
    that broader prepared-resume schema exists.
    """

    from eligsir.checkpointing.exact_refinement_resume import (
        ExactCausalForkObservationDBState,
    )
    from eligsir.mapping.observation_db import (
        BufferStage,
        FrameObservationDB,
        FrameRecord,
    )

    source = FrameObservationDB(device="cpu")
    source.add_frame(
        FrameRecord(
            frame_id=9,
            pose_c2w=np.eye(4, dtype=np.float32),
            timestamp=0.9,
            submap_id=0,
                buffer_stage=BufferStage.COLD_CACHE,
            visible_gaussian_uids=torch.empty(0, dtype=torch.int64),
            resolution_cache={},
            source_identity="source:9",
            source_ordinal=9,
            historical_eligible=True,
            semantic_action="commit",
            prepared_commit_obligation=object(),
        )
    )
    source._source_identity_to_frame_id["source:9"] = 9

    with pytest.raises(RuntimeError, match="unresolved prepared COMMIT"):
        ExactCausalForkObservationDBState(source).resume_state_dict()
