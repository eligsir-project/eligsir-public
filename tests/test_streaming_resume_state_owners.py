from types import SimpleNamespace

import pytest
import torch

from eligsir.mapping.stores.keyframe_store import KeyframeStore
from eligsir.mapping.stores.observation_store import ObservationStore
from eligsir.mapping.stores.replay_view_store import ReplayViewStore
from eligsir.utils.strategies.mcmc.energy import GsplatEnergyMCMCStrategy
from eligsir.utils.strategies.mcmc.legacy import ScheduledMCMCStrategy


def _camera(source_index):
    return SimpleNamespace(uid=source_index, image_name=f"{source_index:06d}")


def test_energy_mcmc_resume_state_round_trip_preserves_upstream_tensors_and_schedule():
    source = GsplatEnergyMCMCStrategy()
    source.strategy = object()
    source.state = {
        "binoms": torch.tensor([1.0, 4.0]),
        "nested": {"count": 9, "mask": torch.tensor([True, False])},
    }
    source.noise_stop_iter = 1200
    source._next_reloc_iter = 700
    source._next_grow_iter = 800

    payload = source.resume_state_dict()
    source.state["binoms"].zero_()
    restored = GsplatEnergyMCMCStrategy()
    restored.strategy = object()
    restored.state = {
        "binoms": torch.zeros(2),
        "nested": {"count": 0, "mask": torch.zeros(2, dtype=torch.bool)},
    }
    restored.load_resume_state_dict(payload, resolver=None)

    assert torch.equal(restored.state["binoms"], torch.tensor([1.0, 4.0]))
    assert torch.equal(restored.state["nested"]["mask"], torch.tensor([True, False]))
    assert restored.state["nested"]["count"] == 9
    assert restored.noise_stop_iter == 1200
    assert restored._next_reloc_iter == 700
    assert restored._next_grow_iter == 800


def test_energy_mcmc_resume_rejects_restore_before_strategy_initialization():
    source = GsplatEnergyMCMCStrategy()
    source.strategy = object()
    source.state = {"binoms": torch.tensor([1.0])}
    source.noise_stop_iter = 100

    with pytest.raises(ValueError, match="initialized before restore"):
        GsplatEnergyMCMCStrategy().load_resume_state_dict(
            source.resume_state_dict(), resolver=None
        )


@pytest.mark.parametrize(
    "saved_state",
    [
        {"other": torch.zeros(2)},
        {"binoms": torch.zeros(3)},
        {"binoms": torch.zeros(2, dtype=torch.float64)},
    ],
)
def test_energy_mcmc_resume_rejects_incompatible_initialized_state(saved_state):
    target = GsplatEnergyMCMCStrategy()
    target.strategy = object()
    target.state = {"binoms": torch.ones(2)}
    target.noise_stop_iter = 10
    payload = {
        "owner_schema": "gsplat_energy_mcmc_v1",
        "state": saved_state,
        "noise_stop_iter": 20,
        "next_reloc_iter": 30,
        "next_grow_iter": 40,
    }

    with pytest.raises(ValueError, match="MCMC state"):
        target.load_resume_state_dict(payload, resolver=None)

    assert torch.equal(target.state["binoms"], torch.ones(2))
    assert target.noise_stop_iter == 10


def test_energy_mcmc_late_counter_failure_is_atomic():
    target = GsplatEnergyMCMCStrategy()
    target.strategy = object()
    target.state = {"binoms": torch.ones(2)}
    target.noise_stop_iter = 10
    target._next_reloc_iter = 11
    target._next_grow_iter = 12
    payload = {
        "owner_schema": "gsplat_energy_mcmc_v1",
        "state": {"binoms": torch.zeros(2)},
        "noise_stop_iter": 20,
        "next_reloc_iter": 21,
        "next_grow_iter": -1,
    }

    with pytest.raises(ValueError, match="next_grow_iter"):
        target.load_resume_state_dict(payload, resolver=None)

    assert torch.equal(target.state["binoms"], torch.ones(2))
    assert (target.noise_stop_iter, target._next_reloc_iter, target._next_grow_iter) == (
        10,
        11,
        12,
    )


def test_legacy_mcmc_resume_state_round_trip_preserves_schedule():
    source = ScheduledMCMCStrategy("mcmc")
    source._next_reloc_iter = 500
    source._next_grow_iter = 600

    restored = ScheduledMCMCStrategy("mcmc")
    restored.load_resume_state_dict(source.resume_state_dict(), resolver=None)

    assert restored._next_reloc_iter == 500
    assert restored._next_grow_iter == 600


def test_mcmc_resume_rejects_a_strategy_identity_mismatch():
    payload = ScheduledMCMCStrategy("mcmc").resume_state_dict()

    with pytest.raises(ValueError, match="strategy identity"):
        ScheduledMCMCStrategy("hybrid").load_resume_state_dict(payload, resolver=None)


def test_scheduled_mcmc_late_counter_failure_is_atomic():
    target = ScheduledMCMCStrategy("mcmc")
    target._next_reloc_iter = 10
    target._next_grow_iter = 20
    payload = {
        "owner_schema": "scheduled_mcmc_v1",
        "strategy_name": "mcmc",
        "next_reloc_iter": 30,
        "next_grow_iter": -1,
    }

    with pytest.raises(ValueError, match="next_grow_iter"):
        target.load_resume_state_dict(payload, resolver=None)

    assert (target._next_reloc_iter, target._next_grow_iter) == (10, 20)


def test_keyframe_store_round_trip_rebinds_source_identities_and_counters():
    cameras = {index: _camera(index) for index in (2, 4, 5, 8)}
    source = KeyframeStore()
    source.train_cameras = [cameras[2], cameras[4]]
    source.test_cameras = [cameras[5]]
    source.current_camera = cameras[4]
    source.keyframes_admitted = 2
    source.keyframes_rejected = 3
    source.last_admitted_source_idx = 4
    source.bootstrap_count = 1
    source.n_train_frames_ingested = 5
    source.cameras_extent = 3.5
    source.holdout_indices = {5, 8}
    source.guard_indices = {8}
    source.camera_centers_np = ["stale-cache"]
    source.camera_centers_gpu = ["stale-cache"]
    source.camera_viewdirs_np = ["stale-cache"]

    restored = KeyframeStore()
    restored.load_resume_state_dict(source.resume_state_dict(), cameras.get)

    assert restored.train_cameras == [cameras[2], cameras[4]]
    assert restored.test_cameras == [cameras[5]]
    assert restored.current_camera is cameras[4]
    assert restored.keyframes_admitted == 2
    assert restored.keyframes_rejected == 3
    assert restored.last_admitted_source_idx == 4
    assert restored.bootstrap_count == 1
    assert restored.n_train_frames_ingested == 5
    assert restored.cameras_extent == 3.5
    assert restored.holdout_indices == {5, 8}
    assert restored.guard_indices == {8}
    assert restored.camera_centers_np == []
    assert restored.camera_centers_gpu == []
    assert restored.camera_viewdirs_np == []


def test_replay_view_store_round_trip_rebinds_identities_and_weights():
    cameras = {index: _camera(index) for index in (1, 3, 7)}
    source = ReplayViewStore()
    source.replay_buffer = [cameras[3], cameras[7]]
    source.global_reservoir = [cameras[1], cameras[7]]
    source.priority_frame_indices = {3, 7}
    source.priority_frame_weights = {3: 1.25, 7: 2.5}
    source.consol_debt_weights = [0.2, 0.8]
    source.consol_debt_pool_len = 2
    source.consol_debt_age = 11

    restored = ReplayViewStore()
    restored.load_resume_state_dict(source.resume_state_dict(), cameras.get)

    assert restored.replay_buffer == [cameras[3], cameras[7]]
    assert restored.global_reservoir == [cameras[1], cameras[7]]
    assert restored.priority_frame_indices == {3, 7}
    assert restored.priority_frame_weights == {3: 1.25, 7: 2.5}
    assert restored.consol_debt_weights == [0.2, 0.8]
    assert restored.consol_debt_pool_len == 2
    assert restored.consol_debt_age == 11


def test_keyframe_store_late_validation_failure_is_atomic():
    cameras = {index: _camera(index) for index in (2, 4, 9)}
    source = KeyframeStore()
    source.train_cameras = [cameras[2]]
    source.current_camera = cameras[2]
    payload = source.resume_state_dict()
    payload["guard_indices"] = [4, "late-invalid"]

    target = KeyframeStore()
    target.train_cameras = [cameras[9]]
    target.current_camera = cameras[9]
    target.keyframes_admitted = 9
    target.camera_centers_np = ["preserve"]

    with pytest.raises(ValueError, match="guard_indices"):
        target.load_resume_state_dict(payload, cameras.get)

    assert target.train_cameras == [cameras[9]]
    assert target.current_camera is cameras[9]
    assert target.keyframes_admitted == 9
    assert target.camera_centers_np == ["preserve"]


def test_replay_view_store_late_validation_failure_is_atomic():
    cameras = {index: _camera(index) for index in (2, 9)}
    source = ReplayViewStore()
    source.replay_buffer = [cameras[2]]
    payload = source.resume_state_dict()
    payload["consol_debt_age"] = "late-invalid"

    target = ReplayViewStore()
    target.replay_buffer = [cameras[9]]
    target.priority_frame_weights = {9: 3.0}

    with pytest.raises(ValueError, match="consol_debt_age"):
        target.load_resume_state_dict(payload, cameras.get)

    assert target.replay_buffer == [cameras[9]]
    assert target.priority_frame_weights == {9: 3.0}


def test_observation_store_resume_round_trip_preserves_report_accumulators():
    source = ObservationStore(obs_voxel=0.02)
    source.obs_keys = {1: {11, 12}, 2: {21}, 4: {41}}
    source.obs_vox = {1: {101}, 2: {201, 202}, 4: {401}}
    source.obs_n_points = 27
    source.scene_complexity_cache = {"sv": 2, "complexity_dim": 2.4}
    source.iq_sensor_sum = 1.5
    source.iq_sensor_sumsq = 0.75
    source.iq_sensor_n = 9
    source.iq_frame_counter = 12
    source.iq_vox = {
        7: [4, 1.0, 2.0, 3.0, 1.1, 2.2, 3.3, 0.2, 0.3, 0.4]
    }
    source.input_quality_cache = {
        "surface_thickness_m": 0.03,
        "complexity_dim": 2.4,
    }

    restored = ObservationStore(obs_voxel=0.02)
    restored.load_resume_state_dict(source.resume_state_dict(), resolver=None)

    assert restored.resume_state_dict() == source.resume_state_dict()


def test_observation_store_resume_rejects_malformed_state_atomically():
    target = ObservationStore(obs_voxel=0.02)
    target.obs_n_points = 5
    before = target.resume_state_dict()
    broken = dict(before)
    broken["iq_vox"] = {7: [4, 1.0]}

    with pytest.raises(ValueError, match="iq_vox"):
        target.load_resume_state_dict(broken, resolver=None)

    assert target.resume_state_dict() == before


@pytest.mark.parametrize("store", [KeyframeStore(), ReplayViewStore()])
def test_camera_store_resume_rejects_an_unresolvable_source_index(store):
    camera = _camera(41)
    if isinstance(store, KeyframeStore):
        store.train_cameras = [camera]
    else:
        store.replay_buffer = [camera]

    with pytest.raises(ValueError, match="source index 41"):
        type(store)().load_resume_state_dict(store.resume_state_dict(), lambda _index: None)
