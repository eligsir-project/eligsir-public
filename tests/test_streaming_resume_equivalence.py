from __future__ import annotations

import copy
import json
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from eligsir.checkpointing.exact_refinement_resume import (
    CachedCameraResolver,
    ExactCausalForkTrainerState,
    ExactTrainerState,
    REQUIRED_CAUSAL_FORK_OWNERS,
    build_causal_fork_contract,
    build_admission_work_signature,
    build_exact_resume_identity,
    qualify_exact_causal_fork_resume,
    persistent_endpoint_paths,
    qualify_exact_refinement_resume,
    rebuild_exact_refinement_observations,
    restore_exact_resume_owners,
    validate_causal_fork_treatment,
    verify_published_endpoint,
)
from eligsir.checkpointing.streaming_resume import build_resume_envelope
from eligsir.mapping.stores.keyframe_store import KeyframeStore
from eligsir.mapping.stores.observation_store import ObservationStore
from eligsir.mapping.stores.replay_view_store import ReplayViewStore
from eligsir.mapping.coordinator import FourClocksAggregator
from eligsir.core.online_state import DirtyRegionQueue, MapWorkItem
from eligsir.runtime.streaming_phase import StreamingPhaseController
from eligsir.utils.realized_cadence import RealizedCadenceTracker
from eligsir.utils.strategies.mcmc.energy import GsplatEnergyMCMCStrategy


def _camera(source_index: int):
    return SimpleNamespace(
        uid=source_index,
        image_name=f"{source_index:06d}",
        source_index=source_index,
    )


def _identity():
    return {
        "source_hash": "source-a",
        "holdout_hash": "holdout-a",
        "resolved_config_hash": "config-a",
        "runtime_hash": "runtime-a",
    }


def _energy_owner():
    owner = GsplatEnergyMCMCStrategy()
    owner.strategy = object()
    owner.state = {"binoms": torch.tensor([1.0, 2.0])}
    owner.noise_stop_iter = 80
    owner._next_reloc_iter = 12
    owner._next_grow_iter = 14
    return owner


class _PortableOwner:
    def __init__(self, payload):
        self.payload = copy.deepcopy(payload)

    def resume_state_dict(self):
        return copy.deepcopy(self.payload)

    def load_resume_state_dict(self, payload, resolver=None):
        del resolver
        self.payload = copy.deepcopy(payload)


def _owners(cameras):
    keyframes = KeyframeStore()
    keyframes.train_cameras = [cameras[0], cameras[2]]
    keyframes.test_cameras = [cameras[1]]
    keyframes.current_camera = cameras[2]
    keyframes.keyframes_admitted = 2
    keyframes.keyframes_rejected = 1
    keyframes.bootstrap_count = 1

    replay = ReplayViewStore()
    replay.replay_buffer = [cameras[0], cameras[2]]
    replay.global_reservoir = [cameras[2]]
    replay.priority_frame_indices = {2}
    replay.priority_frame_weights = {2: 1.5}

    dirty_regions = DirtyRegionQueue()
    dirty_regions.push(MapWorkItem("refine", 1.0, cameras[0], 9))
    dirty_regions.push(MapWorkItem("refine", 1.0, cameras[2], 10))

    phase = StreamingPhaseController(3)
    phase.observe_source(True, True, 10, 20)

    cadence = RealizedCadenceTracker(first_iteration=0, window_steps=2)
    cadence.record_source_released()
    cadence.record_source_ingested()
    cadence.record_optimizer_step(20)

    trainer = ExactTrainerState(
        {
            "source_idx": 3,
            "source_frame_count": 3,
            "n_frames_ingested": 3,
            "last_ingest_iter": 20,
            "optimizer_step_id": 10,
            "supervision_view_count": 12,
            "rasterizer_call_count": 12,
            "supervised_pixel_count": 120,
            "sensor_frame_count": 3,
            "mutation_event_id": 4,
            "depth_mutation_event_id": 2,
            "mcmc_mutation_event_id": 2,
            "total_inserted": 30,
            "last_flush_iteration": 20,
            "iter_since_new_frame": 5,
            "scene_steps_since_new_frame": 5,
            "opt_batch_distinct_total": 8,
            "opt_batch_count": 4,
            "per_source_supervision_counts": {0: 6, 2: 6},
            "bootstrap_source_indices": [0],
            "source_dispositions": {0: "admitted", 1: "holdout", 2: "admitted"},
            "insertion_batch_empty": True,
            "consolidation_queue_empty": True,
        }
    )
    return {
        "mcmc": _energy_owner(),
        "keyframes": keyframes,
        "replay": replay,
        "dirty_regions": dirty_regions,
        "phase": phase,
        "cadence": cadence,
        "trainer": trainer,
        "mapping_clocks": FourClocksAggregator(),
        "observations": ObservationStore(obs_voxel=0.02),
    }


def test_cached_camera_resolver_materializes_each_source_index_once():
    calls = []
    resolver = CachedCameraResolver(lambda index: calls.append(index) or _camera(index))

    assert resolver(2) is resolver(2)
    assert resolver(1) is resolver(1)
    assert calls == [2, 1]


def test_exact_refinement_rebuilds_required_observations_once_in_source_order():
    cameras = [_camera(2), _camera(0), _camera(3), _camera(2)]
    dispositions = {0: "admitted", 1: "rejected", 2: "admitted", 3: "holdout"}
    calls = []

    def record(**kwargs):
        calls.append(kwargs)

    rebuild_exact_refinement_observations(
        db=object(),
        cameras=cameras,
        source_dispositions=dispositions,
        get_frame=lambda index: f"frame-{index}",
        record=record,
        low_res_divisor=4,
        depthless_rgb_cache=True,
    )

    assert [call["frame_id"] for call in calls] == [0, 2, 3]
    assert [(call["is_train"], call["is_holdout"]) for call in calls] == [
        (True, False),
        (True, False),
        (False, True),
    ]
    assert [call["frame"] for call in calls] == ["frame-0", "frame-2", "frame-3"]


def test_exact_resume_restores_owners_then_rng_for_identical_continuation():
    cameras = {index: _camera(index) for index in range(3)}
    source = _owners(cameras)
    random.seed(7)
    np.random.seed(11)
    torch.manual_seed(13)
    envelope = build_resume_envelope(_identity(), source)

    expected_draws = (random.random(), float(np.random.rand()), torch.rand(2))
    expected_view = random.choice([0, 2])
    expected_schedule = (
        source["mcmc"]._next_reloc_iter,
        source["mcmc"]._next_grow_iter,
    )

    restored_cameras = {}
    restored = _owners({index: _camera(99 + index) for index in range(3)})
    resolver = CachedCameraResolver(
        lambda index: restored_cameras.setdefault(index, _camera(index))
    )
    random.seed(101)
    np.random.seed(103)
    torch.manual_seed(107)

    restore_exact_resume_owners(
        envelope=envelope,
        expected_identity=_identity(),
        owners=restored,
        resolver=resolver,
    )

    actual_draws = (random.random(), float(np.random.rand()), torch.rand(2))
    actual_view = random.choice([cam.source_index for cam in restored["replay"].replay_buffer])
    assert actual_draws[0:2] == expected_draws[0:2]
    assert torch.equal(actual_draws[2], expected_draws[2])
    assert actual_view == expected_view
    assert [c.source_index for c in restored["keyframes"].train_cameras] == [0, 2]
    assert restored["trainer"].state["source_dispositions"] == source["trainer"].state["source_dispositions"]
    assert (
        restored["mcmc"]._next_reloc_iter,
        restored["mcmc"]._next_grow_iter,
    ) == expected_schedule
    assert restored["phase"].resume_state_dict() == source["phase"].resume_state_dict()
    assert restored["cadence"].resume_state_dict() == source["cadence"].resume_state_dict()
    assert restored["dirty_regions"].resume_state_dict() == source["dirty_regions"].resume_state_dict()


def test_uninterrupted_and_resumed_refinement_match_topology_optimizer_and_phase():
    cameras = {index: _camera(index) for index in range(3)}
    direct_owners = _owners(cameras)
    direct_owners["mcmc"]._next_grow_iter = 11
    random.seed(17)
    np.random.seed(19)
    torch.manual_seed(23)
    envelope = build_resume_envelope(_identity(), direct_owners)

    direct_param = torch.nn.Parameter(torch.tensor([0.25, -0.5]))
    direct_optimizer = torch.optim.Adam([direct_param], lr=0.01)
    saved_param = direct_param.detach().clone()
    saved_optimizer = copy.deepcopy(direct_optimizer.state_dict())
    saved_topology = [0, 1]

    def _continue(param, optimizer, topology, owners):
        sampled_views = []
        for successful_step in (11, 12, 13):
            sampled_views.append(
                random.choice([c.source_index for c in owners["replay"].replay_buffer])
            )
            target = torch.tensor([random.random(), float(np.random.rand())])
            optimizer.zero_grad(set_to_none=True)
            (param - target).square().sum().backward()
            optimizer.step()
            owners["cadence"].record_optimizer_step(successful_step)
            owners["phase"].observe_source(True, True, successful_step, successful_step + 10)
            if owners["mcmc"]._next_grow_iter == successful_step:
                topology.append(100 + int(torch.randint(0, 50, ()).item()))
                owners["mcmc"]._next_grow_iter += 100
        return sampled_views

    direct_topology = list(saved_topology)
    direct_views = _continue(
        direct_param, direct_optimizer, direct_topology, direct_owners
    )

    resumed_owners = _owners({index: _camera(10 + index) for index in range(3)})
    restore_exact_resume_owners(
        envelope=envelope,
        expected_identity=_identity(),
        owners=resumed_owners,
        resolver=CachedCameraResolver(cameras.get),
    )
    resumed_param = torch.nn.Parameter(saved_param.clone())
    resumed_optimizer = torch.optim.Adam([resumed_param], lr=0.01)
    resumed_optimizer.load_state_dict(saved_optimizer)
    resumed_topology = list(saved_topology)
    resumed_views = _continue(
        resumed_param, resumed_optimizer, resumed_topology, resumed_owners
    )

    assert resumed_views == direct_views
    assert resumed_topology == direct_topology
    assert torch.equal(resumed_param, direct_param)
    direct_state = direct_optimizer.state[direct_param]
    resumed_state = resumed_optimizer.state[resumed_param]
    assert torch.equal(resumed_state["exp_avg"], direct_state["exp_avg"])
    assert torch.equal(resumed_state["exp_avg_sq"], direct_state["exp_avg_sq"])
    assert resumed_owners["phase"].resume_state_dict() == direct_owners["phase"].resume_state_dict()
    assert resumed_owners["cadence"].resume_state_dict() == direct_owners["cadence"].resume_state_dict()


def test_owner_restore_is_atomic_across_owners():
    cameras = {index: _camera(index) for index in range(3)}
    source = _owners(cameras)
    envelope = build_resume_envelope(_identity(), source)
    broken = copy.deepcopy(envelope)
    broken["owners"]["trainer"]["consolidation_queue_empty"] = False
    target = _owners(cameras)
    before = {name: owner.resume_state_dict() for name, owner in target.items()}

    with pytest.raises(ValueError, match="consolidation"):
        restore_exact_resume_owners(
            envelope=broken,
            expected_identity=_identity(),
            owners=target,
            resolver=CachedCameraResolver(cameras.get),
        )

    after = {name: owner.resume_state_dict() for name, owner in target.items()}
    assert torch.equal(after["mcmc"]["state"]["binoms"], before["mcmc"]["state"]["binoms"])
    for name in set(target) - {"mcmc"}:
        assert after[name] == before[name]


@pytest.mark.parametrize(
    ("override", "reason"),
    [
        ({"checkpoint_reason": "periodic"}, "checkpoint_reason_periodic"),
        ({"ingestion_mode": "wallclock_strict"}, "ingestion_mode_wallclock_strict"),
        ({"ingestion_mode": "dataset_fps"}, "ingestion_mode_dataset_fps"),
        ({"live_source": True}, "live_source"),
        ({"async_ingestion": True}, "async_ingestion_enabled"),
        ({"online_wall_time_controller": True}, "online_wall_time_controller_enabled"),
        ({"online_mode": True}, "online_mode_enabled"),
        ({"online_mode": None}, "online_mode_enabled"),
        ({"bounded_queue_pipeline": True}, "bounded_queue_pipeline_enabled"),
        ({"sampling_mode": "stratified"}, "unsupported_sampling_mode_stratified"),
        ({"residual_replay_weight": 0.5}, "residual_replay_enabled"),
        ({"growth_staleness_max_iters": 50}, "growth_staleness_brake_enabled"),
        ({"adaptive_schedule_horizons": True}, "adaptive_schedule_horizons_enabled"),
        ({"adaptive_capacity": True}, "adaptive_capacity_enabled"),
        ({"pose_or_loop_correction": True}, "pose_or_loop_correction_enabled"),
        ({"refinement_resolution_tier": "low"}, "refinement_resolution_tier_low"),
        ({"strategy": "mcmc"}, "unsupported_strategy_mcmc"),
        ({"insertion_batch_empty": False}, "insertion_batch_not_empty"),
        ({"consolidation_queue_empty": False}, "consolidation_queue_not_empty"),
        ({"envelope": None}, "streaming_checkpoint_v2_missing"),
    ],
)
def test_exact_resume_qualification_fails_closed(override, reason):
    owners = _owners({index: _camera(index) for index in range(3)})
    runtime = {
        "checkpoint_reason": "mapping_endpoint",
        "ingestion_mode": "iter_based",
        "live_source": False,
        "async_ingestion": False,
        "online_wall_time_controller": False,
        "online_mode": False,
        "bounded_queue_pipeline": False,
        "sampling_mode": "legacy",
        "residual_replay_weight": 0.0,
        "growth_staleness_max_iters": 0,
        "adaptive_schedule_horizons": False,
        "adaptive_capacity": False,
        "pose_or_loop_correction": False,
        "refinement_resolution_tier": "high",
        "strategy": "gsplat_energy_mcmc",
        "insertion_batch_empty": True,
        "consolidation_queue_empty": True,
        "streaming_phase": {"phase": "refinement"},
        "envelope": build_resume_envelope(_identity(), owners),
    }
    runtime.update(override)

    verdict = qualify_exact_refinement_resume(runtime, _identity(), set(owners))

    assert verdict.eligible is False
    assert reason in verdict.reasons


def test_exact_resume_qualification_rejects_identity_and_owner_mismatch():
    owners = _owners({index: _camera(index) for index in range(3)})
    envelope = build_resume_envelope(_identity(), owners)
    runtime = {
        "checkpoint_reason": "mapping_endpoint",
        "ingestion_mode": "iter_based",
        "live_source": False,
        "async_ingestion": False,
        "online_wall_time_controller": False,
        "online_mode": False,
        "bounded_queue_pipeline": False,
        "sampling_mode": "legacy",
        "residual_replay_weight": 0.0,
        "growth_staleness_max_iters": 0,
        "adaptive_schedule_horizons": False,
        "adaptive_capacity": False,
        "pose_or_loop_correction": False,
        "refinement_resolution_tier": "high",
        "strategy": "gsplat_energy_mcmc",
        "insertion_batch_empty": True,
        "consolidation_queue_empty": True,
        "streaming_phase": {"phase": "refinement"},
        "envelope": envelope,
    }
    assert "resume_identity_missing_or_mismatch" in qualify_exact_refinement_resume(
        runtime, {**_identity(), "source_hash": "other"}, set(owners)
    ).reasons

    envelope["owners"].pop("replay")
    verdict = qualify_exact_refinement_resume(runtime, _identity(), set(owners))
    assert "owner_set_mismatch" in verdict.reasons
    assert "owner_state_missing_or_invalid" in verdict.reasons


def test_exact_resume_qualification_rejects_nonmapping_owners_without_crashing():
    owners = _owners({index: _camera(index) for index in range(3)})
    envelope = build_resume_envelope(_identity(), owners)
    envelope["owners"] = None
    runtime = {
        "checkpoint_reason": "mapping_endpoint",
        "ingestion_mode": "iter_based",
        "live_source": False,
        "async_ingestion": False,
        "online_wall_time_controller": False,
        "online_mode": False,
        "bounded_queue_pipeline": False,
        "sampling_mode": "legacy",
        "residual_replay_weight": 0.0,
        "growth_staleness_max_iters": 0,
        "adaptive_schedule_horizons": False,
        "adaptive_capacity": False,
        "pose_or_loop_correction": False,
        "refinement_resolution_tier": "high",
        "strategy": "gsplat_energy_mcmc",
        "insertion_batch_empty": True,
        "consolidation_queue_empty": True,
        "streaming_phase": {"phase": "refinement"},
        "envelope": envelope,
    }

    verdict = qualify_exact_refinement_resume(runtime, _identity(), set(owners))

    assert verdict.eligible is False
    assert "owner_state_missing_or_invalid" in verdict.reasons


def test_exact_identity_is_stable_and_authoritative_values_win():
    one = build_exact_resume_identity(
        scene_identity={"source_path": "/data/fr3", "frames": 3},
        holdout_identity={"indices": [1]},
        effective_args={
            "cap_max": 4,
            "model_path": "/scratch/a",
            "start_checkpoint": "a",
            "run_log_path": "/scratch/a/train.log",
        },
        runtime_identity={"torch": "2", "strategy": "gsplat_energy_mcmc"},
    )
    two = build_exact_resume_identity(
        scene_identity={"frames": 3, "source_path": "/data/fr3"},
        holdout_identity={"indices": [1]},
        effective_args={
            "start_checkpoint": "b",
            "model_path": "/scratch/b",
            "run_log_path": "/scratch/b/train.log",
            "cap_max": 4,
        },
        runtime_identity={"strategy": "gsplat_energy_mcmc", "torch": "2"},
        authoritative={"source_hash": "rf-source"},
    )

    assert one["resolved_config_hash"] == two["resolved_config_hash"]
    assert two["source_hash"] == "rf-source"


def test_causal_fork_identity_allows_only_declared_observational_labels():
    common = dict(
        scene_identity={"source_path": "/data/fr3", "frames": 3},
        holdout_identity={"indices": [1]},
        runtime_identity={"torch": "2", "strategy": "gsplat_energy_mcmc"},
        causal_fork=True,
    )
    one = build_exact_resume_identity(
        **common,
        effective_args={
            "cap_max": 4,
            "streaming_seed_lineage_freeze_from_iter": -1,
            "baseline_run": "/output/baseline-a",
            "experiment_origin": {
                "researchflow_run_id": "run-a",
                "researchflow_plan_sha256": "plan-a",
                "experiment_id": "science-346",
                "revision": "r1",
                "run_label": "P",
                "arm": "baseline",
                "purpose": "prefix",
                "source_hash": "source-a",
            },
        },
    )
    two = build_exact_resume_identity(
        **common,
        effective_args={
            "cap_max": 4,
            "streaming_seed_lineage_freeze_from_iter": 40,
            "baseline_run": "/output/baseline-b",
            "experiment_origin": {
                "researchflow_run_id": "run-b",
                "researchflow_plan_sha256": "plan-b",
                "experiment_id": "science-346-fork",
                "revision": "r2",
                "run_label": "F",
                "arm": "treatment",
                "purpose": "causal-tail",
                "source_hash": "source-a",
            },
        },
    )
    assert one == two

    changed = build_exact_resume_identity(
        **common,
        effective_args={
            "cap_max": 4,
            "baseline_run": "/output/baseline-b",
            "experiment_origin": {
                "researchflow_run_id": "run-b",
                "researchflow_plan_sha256": "plan-b",
                "experiment_id": "science-346",
                "arm": "treatment",
                "source_hash": "different-source",
            },
        },
    )
    assert changed["resolved_config_hash"] != one["resolved_config_hash"]


def test_freeze_delta_is_normalized_only_for_causal_fork_identity():
    kwargs = dict(
        scene_identity={"source": "fr3"},
        holdout_identity={"indices": [1]},
        runtime_identity={"torch": "2"},
    )
    prefix = {"cap_max": 4, "streaming_seed_lineage_freeze_from_iter": -1}
    treatment = {"cap_max": 4, "streaming_seed_lineage_freeze_from_iter": 40}

    assert build_exact_resume_identity(
        **kwargs, effective_args=prefix, causal_fork=True
    ) == build_exact_resume_identity(
        **kwargs, effective_args=treatment, causal_fork=True
    )
    assert build_exact_resume_identity(
        **kwargs, effective_args=prefix
    )["resolved_config_hash"] != build_exact_resume_identity(
        **kwargs, effective_args=treatment
    )["resolved_config_hash"]


@pytest.mark.parametrize(
    ("checkpoint_value", "resume_value", "checkpoint_iteration", "resume_fork_iteration"),
    [(-1, -1, 40, 40), (-1, 40, 40, 40)],
)
def test_causal_fork_treatment_accepts_only_aa_or_declared_k_delta(
    checkpoint_value, resume_value, checkpoint_iteration, resume_fork_iteration
):
    contract = build_causal_fork_contract(
        checkpoint_iteration=checkpoint_iteration,
        seed_lineage_freeze_from_iter=checkpoint_value,
        fork_boundary_flush={
            "iteration": checkpoint_iteration,
            "pending_items": 2,
            "flushed_items": 2,
            "inserted_count": 7,
            "work_id": 9,
            "optimizer_step_id": 10,
            "mutation_event_id": 11,
        },
    )
    validate_causal_fork_treatment(
        contract,
        resume_fork_iteration=resume_fork_iteration,
        resume_seed_lineage_freeze_from_iter=resume_value,
    )


@pytest.mark.parametrize(
    ("checkpoint_value", "resume_value", "checkpoint_iteration", "resume_fork_iteration"),
    [(5, -1, 40, 40), (-1, 41, 40, 40), (-1, 40, 40, 41), (-1, True, 40, 40), (-1, "40", 40, 40)],
)
def test_causal_fork_treatment_rejects_reverse_mismatch_and_nonliteral_values(
    checkpoint_value, resume_value, checkpoint_iteration, resume_fork_iteration
):
    contract = build_causal_fork_contract(
        checkpoint_iteration=checkpoint_iteration,
        seed_lineage_freeze_from_iter=checkpoint_value,
        fork_boundary_flush={
            "iteration": checkpoint_iteration,
            "pending_items": 0,
            "flushed_items": 0,
            "inserted_count": 0,
            "work_id": 0,
            "optimizer_step_id": 10,
            "mutation_event_id": 11,
        },
    )
    with pytest.raises(ValueError, match="causal fork"):
        validate_causal_fork_treatment(
            contract,
            resume_fork_iteration=resume_fork_iteration,
            resume_seed_lineage_freeze_from_iter=resume_value,
        )


def _causal_trainer_state():
    state = {
            "source_idx": 2,
            "source_frame_count": 3,
            "n_frames_ingested": 2,
            "last_ingest_iter": 40,
            "optimizer_step_id": 10,
            "supervision_view_count": 12,
            "rasterizer_call_count": 12,
            "supervised_pixel_count": 120,
            "sensor_frame_count": 2,
            "mutation_event_id": 4,
            "depth_mutation_event_id": 2,
            "mcmc_mutation_event_id": 2,
            "total_inserted": 30,
            "last_flush_iteration": 40,
            "iter_since_new_frame": 5,
            "scene_steps_since_new_frame": 5,
            "opt_batch_distinct_total": 8,
            "opt_batch_count": 4,
            "per_source_supervision_counts": {0: 6},
            "bootstrap_source_indices": [0],
            "source_dispositions": {0: "admitted", 1: "holdout"},
            "insertion_batch_empty": True,
            "consolidation_queue_source_indices": [0],
            "fork_boundary_flush": {
                "iteration": 40,
                "pending_items": 1,
                "flushed_items": 1,
                "inserted_count": 3,
                "work_id": 7,
                "optimizer_step_id": 10,
                "mutation_event_id": 4,
            },
        }
    state["admission_work_signature"] = build_admission_work_signature(
        state["source_dispositions"],
        optimizer_step_id=state["optimizer_step_id"],
        supervision_view_count=state["supervision_view_count"],
        sensor_frame_count=state["sensor_frame_count"],
    )
    return ExactCausalForkTrainerState(state)


def test_causal_fork_trainer_state_supports_partial_cursor_and_exact_queue():
    state = _causal_trainer_state()
    payload = state.resume_state_dict()
    restored = _causal_trainer_state()
    restored.load_resume_state_dict(payload)
    assert payload["owner_schema"] == "exact_causal_fork_trainer_v1"
    assert restored.state["source_idx"] == 2
    assert restored.state["consolidation_queue_source_indices"] == [0]

    payload["fork_boundary_flush"] = None
    with pytest.raises(ValueError, match="boundary flush"):
        restored.load_resume_state_dict(payload)


def test_causal_fork_qualification_requires_mapping_phase_and_all_owners():
    cameras = {index: _camera(index) for index in range(3)}
    owners = _owners(cameras)
    owners["trainer"] = _causal_trainer_state()
    owners["phase"] = StreamingPhaseController(3)
    from eligsir.mapping.scheduler.frame_scheduler import FrameScheduler
    from eligsir.mapping.scheduler.keyframes import KeyframeAdmitter
    owners["frame_scheduler"] = FrameScheduler(steps_per_frame=3, ingestion_mode="iter_based")
    owners["admission"] = KeyframeAdmitter()
    owners["scene"] = _PortableOwner({"owner_schema": "exact_causal_fork_scene_v1"})
    owners["frame_observations"] = _PortableOwner(
        {"owner_schema": "exact_causal_fork_observation_db_v1"}
    )
    contract = build_causal_fork_contract(
        checkpoint_iteration=40,
        seed_lineage_freeze_from_iter=-1,
        fork_boundary_flush=owners["trainer"].state["fork_boundary_flush"],
    )
    runtime = {
        "checkpoint_reason": "causal_fork",
        "ingestion_mode": "iter_based",
        "live_source": False,
        "async_ingestion": False,
        "online_wall_time_controller": False,
        "online_mode": False,
        "bounded_queue_pipeline": False,
        "sampling_mode": "legacy",
        "residual_replay_weight": 0.0,
        "growth_staleness_max_iters": 0,
        "adaptive_schedule_horizons": False,
        "adaptive_capacity": False,
        "pose_or_loop_correction": False,
        "reactive_multires": False,
        "strategy": "gsplat_energy_mcmc",
        "admission_mode": "hybrid_keyframe",
        "adaptive_keyframe_quality": False,
        "tracking_confidence": False,
        "keyframe_tracking_confidence": False,
        "shadow_utility": False,
        "temporal_depth_modes": False,
        "bootstrap_seed_decay": False,
        "assimilation_control": False,
        "historical_repair_controller": False,
        "backpressure": False,
        "insertion_batching": False,
        "dataset_pose_source": True,
        "admission_rollout_mode": "shadow",
        "insertion_batch_empty": True,
        "computational_queues_serialized": True,
        "state_writers_joined": True,
        "streaming_phase": {"phase": "mapping"},
        "causal_fork": contract,
        "envelope": build_resume_envelope(_identity(), owners),
    }
    assert set(owners) == REQUIRED_CAUSAL_FORK_OWNERS
    assert qualify_exact_causal_fork_resume(runtime, _identity(), set(owners)).eligible

    broken = copy.deepcopy(runtime)
    broken["envelope"]["owners"].pop("admission")
    verdict = qualify_exact_causal_fork_resume(broken, _identity(), set(owners))
    assert not verdict.eligible
    assert "owner_set_mismatch" in verdict.reasons


@pytest.mark.parametrize(
    ("override", "reason"),
    [
        ({"reactive_multires": True}, "reactive_multires_enabled"),
        ({"temporal_depth_modes": True}, "temporal_depth_modes_enabled"),
        ({"bootstrap_seed_decay": True}, "bootstrap_seed_decay_enabled"),
        ({"assimilation_control": True}, "assimilation_control_enabled"),
        ({"historical_repair_controller": True}, "historical_repair_controller_enabled"),
        ({"backpressure": True}, "backpressure_enabled"),
        ({"insertion_batching": True}, "nondefault_insertion_batching"),
        ({"dataset_pose_source": False}, "stateful_or_non_dataset_pose_source"),
        ({"state_writers_joined": False}, "state_writers_not_joined"),
        ({"computational_queues_serialized": False}, "computational_queues_not_serialized"),
        ({"streaming_phase": {"phase": "refinement"}}, "causal_fork_phase_missing_or_invalid"),
        ({"admission_mode": "all"}, "unsupported_admission_mode_all"),
        ({"adaptive_keyframe_quality": True}, "adaptive_keyframe_quality_enabled"),
        ({"tracking_confidence": True}, "tracking_confidence_enabled"),
        ({"keyframe_tracking_confidence": True}, "keyframe_tracking_confidence_enabled"),
        ({"shadow_utility": True}, "shadow_utility_enabled"),
        ({"admission_rollout_mode": "active"}, "unsupported_admission_rollout_mode_active"),
    ],
)
def test_causal_fork_qualification_fails_closed_on_inexact_runtime(override, reason):
    runtime = {
        "checkpoint_reason": "causal_fork",
        "ingestion_mode": "iter_based",
        "live_source": False,
        "async_ingestion": False,
        "online_wall_time_controller": False,
        "online_mode": False,
        "bounded_queue_pipeline": False,
        "sampling_mode": "legacy",
        "residual_replay_weight": 0.0,
        "growth_staleness_max_iters": 0,
        "adaptive_schedule_horizons": False,
        "adaptive_capacity": False,
        "pose_or_loop_correction": False,
        "reactive_multires": False,
        "strategy": "gsplat_energy_mcmc",
        "admission_mode": "hybrid_keyframe",
        "adaptive_keyframe_quality": False,
        "tracking_confidence": False,
        "keyframe_tracking_confidence": False,
        "shadow_utility": False,
        "temporal_depth_modes": False,
        "bootstrap_seed_decay": False,
        "assimilation_control": False,
        "historical_repair_controller": False,
        "backpressure": False,
        "insertion_batching": False,
        "dataset_pose_source": True,
        "admission_rollout_mode": "shadow",
        "insertion_batch_empty": True,
        "computational_queues_serialized": True,
        "state_writers_joined": True,
        "streaming_phase": {"phase": "mapping"},
        "causal_fork": {},
        "envelope": None,
    }
    runtime.update(override)
    verdict = qualify_exact_causal_fork_resume(runtime, _identity(), REQUIRED_CAUSAL_FORK_OWNERS)
    assert not verdict.eligible
    assert reason in verdict.reasons


def test_endpoint_metadata_publishes_persistent_path_not_local_staging():
    paths = persistent_endpoint_paths(
        "/scratch/unknown/run-ephemeral",
        "/tmp/output/paper-run",
        42,
    )

    assert paths["directory"] == "/tmp/output/paper-run/mapping_endpoint"
    assert paths["checkpoint"].endswith("/mapping_endpoint/chkpnt42.pth")
    assert paths["resolved_local_directory"] == "/scratch/unknown/run-ephemeral/mapping_endpoint"


def test_published_endpoint_verification_requires_nonempty_checkpoint_ply_and_report(
    tmp_path,
):
    local_root = tmp_path / "local"
    published_root = tmp_path / "published"
    paths = persistent_endpoint_paths(str(local_root), str(published_root), 42)
    phase = {
        "phase": "refinement",
        "mapping_end_iteration": 42,
        "source_coverage_fraction": 1.0,
        "exact_resume": {"classification": "EXACT", "reasons": []},
    }
    local_files = {
        "checkpoint": local_root / "mapping_endpoint" / "chkpnt42.pth",
        "point_cloud": local_root / "mapping_endpoint" / "point_cloud.ply",
        "phase_metadata": local_root / "mapping_endpoint" / "streaming_phase_metadata.json",
        "report": local_root / "comparison" / "mapping_endpoint" / "report.json",
    }
    for key, local_path in local_files.items():
        published_path = Path(paths[key])
        local_path.parent.mkdir(parents=True, exist_ok=True)
        published_path.parent.mkdir(parents=True, exist_ok=True)
        payload = (
            json.dumps(phase if key == "phase_metadata" else {"streaming_phase": phase})
            if key in {"phase_metadata", "report"}
            else "artifact"
        )
        local_path.write_text(payload)
        published_path.write_text(payload)

    verify_published_endpoint(str(local_root), paths, iteration=42)

    published_report = Path(paths["report"])
    valid_report = published_report.read_bytes()
    published_report.write_bytes(b"{" + b" " * (len(valid_report) - 1))
    with pytest.raises(RuntimeError, match="JSON is invalid"):
        verify_published_endpoint(str(local_root), paths, iteration=42)

    published_report.write_bytes(valid_report)
    Path(paths["report"]).write_bytes(b"")
    with pytest.raises(RuntimeError, match="report.*empty"):
        verify_published_endpoint(str(local_root), paths, iteration=42)
