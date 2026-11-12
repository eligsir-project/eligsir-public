"""Fail-closed composition for mapping-endpoint refinement resume."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import torch

from eligsir.checkpointing.streaming_resume import (
    restore_rng_state,
    validate_resume_envelope,
)


REQUIRED_EXACT_OWNERS = {
    "mcmc", "keyframes", "replay", "dirty_regions", "phase", "cadence", "trainer",
    "mapping_clocks", "observations",
}
REQUIRED_CAUSAL_FORK_OWNERS = REQUIRED_EXACT_OWNERS | {
    "frame_scheduler", "admission", "scene", "frame_observations",
}
_OPERATIONAL_CONFIG_KEYS = {
    "model_path", "persist_model_path", "start_checkpoint", "checkpoint",
    "checkpoint_iterations", "checkpoint_interval", "save_iterations",
    "save_interval", "output_staging", "output_staging_root",
    "output_log_sync_interval", "output_sync_interval",
    "run_log_path",
}
_CAUSAL_FORK_OBSERVATIONAL_ORIGIN_KEYS = {
    "researchflow_run_id", "researchflow_plan_sha256", "experiment_id",
    "revision", "run_label", "arm", "purpose",
}
_CAUSAL_FORK_OBSERVATIONAL_ARG_KEYS = {
    "baseline_run", "experiment_id", "experiment_arm", "experiment_purpose",
    "researchflow_run_id", "researchflow_plan_sha256", "revision", "run_label",
}


def _portable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _portable(v) for k, v in sorted(value.items(), key=lambda p: str(p[0]))}
    if isinstance(value, (list, tuple)):
        return [_portable(v) for v in value]
    if isinstance(value, set):
        return sorted(_portable(v) for v in value)
    if hasattr(value, "tolist") and callable(value.tolist):
        return _portable(value.tolist())
    if hasattr(value, "__dict__"):
        return _portable(vars(value))
    return repr(value)


def _hash_payload(value: Any) -> str:
    encoded = json.dumps(
        _portable(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _causal_fork_computational_args(args_dict: Mapping[str, Any]) -> dict[str, Any]:
    frozen = copy.deepcopy(dict(args_dict))
    for key in _CAUSAL_FORK_OBSERVATIONAL_ARG_KEYS:
        frozen.pop(key, None)
    # The immutable causal-fork contract validates the sole allowed -1 -> K
    # treatment delta. Keeping this field in the general config hash would
    # reject that declared delta before the dedicated validator could inspect it.
    frozen.pop("streaming_seed_lineage_freeze_from_iter", None)
    origin = frozen.get("experiment_origin")
    if isinstance(origin, Mapping):
        cleaned = {
            key: copy.deepcopy(value) for key, value in origin.items()
            if key not in _CAUSAL_FORK_OBSERVATIONAL_ORIGIN_KEYS
        }
        researchflow = cleaned.get("researchflow")
        if isinstance(researchflow, Mapping):
            researchflow = {
                key: copy.deepcopy(value) for key, value in researchflow.items()
                if key not in {"run_id", "plan_sha256", "experiment_id", "revision",
                               "run_label", "arm", "purpose"}
            }
            if researchflow:
                cleaned["researchflow"] = researchflow
            else:
                cleaned.pop("researchflow", None)
        frozen["experiment_origin"] = cleaned
    return frozen


def build_exact_resume_identity(
    *, scene_identity: Mapping[str, Any], holdout_identity: Mapping[str, Any],
    effective_args: Mapping[str, Any] | Any,
    runtime_identity: Mapping[str, Any],
    authoritative: Mapping[str, Any] | None = None,
    causal_fork: bool = False,
) -> dict[str, str]:
    """Hash frozen inputs/config/runtime, preferring authoritative declarations."""
    args_dict = dict(effective_args) if isinstance(effective_args, Mapping) else vars(effective_args)
    frozen_args = {
        k: v for k, v in args_dict.items()
        if k not in _OPERATIONAL_CONFIG_KEYS and not k.startswith("_")
    }
    if causal_fork:
        frozen_args = _causal_fork_computational_args(frozen_args)
    identity = {
        "source_hash": _hash_payload(scene_identity),
        "holdout_hash": _hash_payload(holdout_identity),
        "resolved_config_hash": _hash_payload(frozen_args),
        "runtime_hash": _hash_payload(runtime_identity),
    }
    authoritative_values = dict(authoritative or {})
    if causal_fork:
        # RF declaration hashes may include the deliberately different arm/run
        # labels or treatment flag. Source and holdout declarations remain
        # authoritative; config/runtime are recomputed from the exact filtered
        # computational inputs above.
        authoritative_values = {
            key: value for key, value in authoritative_values.items()
            if key in {"source_hash", "holdout_hash"}
        }
    for key, value in authoritative_values.items():
        if key in identity and isinstance(value, str) and value.strip():
            identity[key] = value.strip()
    return identity


def authoritative_identity_hashes(value: Any) -> dict[str, str]:
    """Extract exact identity hashes (and common manifest aliases) recursively."""
    aliases = {
        "source_hash": ("source_hash", "source_artifact_hash"),
        "holdout_hash": ("holdout_hash",),
        "resolved_config_hash": ("resolved_config_hash", "config_hash"),
        "runtime_hash": ("runtime_hash",),
    }
    found: dict[str, str] = {}

    def _visit(node: Any) -> None:
        if not isinstance(node, Mapping):
            return
        for canonical, names in aliases.items():
            if canonical in found:
                continue
            for name in names:
                candidate = node.get(name)
                if isinstance(candidate, str) and candidate.strip():
                    found[canonical] = candidate.strip()
                    break
        for child in node.values():
            _visit(child)

    _visit(value)
    return found


@dataclass(frozen=True)
class ExactResumeVerdict:
    eligible: bool
    reasons: tuple[str, ...]

    @property
    def evidence_class(self) -> str:
        return "EXACT" if self.eligible else "RECOVERY_ONLY"


_FORK_FLUSH_KEYS = {
    "iteration", "pending_items", "flushed_items", "inserted_count", "work_id",
    "optimizer_step_id", "mutation_event_id",
}


def _validate_fork_flush(value: Any, checkpoint_iteration: int) -> dict[str, int]:
    if not isinstance(value, Mapping) or set(value) != _FORK_FLUSH_KEYS:
        raise ValueError("causal fork boundary flush keys mismatch")
    result = dict(value)
    for key, item in result.items():
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise ValueError(f"causal fork boundary flush {key} must be a non-negative integer")
    if result["iteration"] != checkpoint_iteration:
        raise ValueError("causal fork boundary flush iteration mismatch")
    if result["flushed_items"] != result["pending_items"]:
        raise ValueError("causal fork boundary did not flush every insertion item")
    return result


def build_causal_fork_contract(
    *, checkpoint_iteration: int, seed_lineage_freeze_from_iter: int,
    fork_boundary_flush: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the immutable, narrowly-scoped fork/treatment declaration."""
    if isinstance(checkpoint_iteration, bool) or not isinstance(checkpoint_iteration, int) \
            or checkpoint_iteration < 0:
        raise ValueError("causal fork checkpoint iteration must be non-negative")
    if isinstance(seed_lineage_freeze_from_iter, bool) \
            or not isinstance(seed_lineage_freeze_from_iter, int):
        raise ValueError("causal fork treatment must be a literal integer")
    flush = _validate_fork_flush(fork_boundary_flush, checkpoint_iteration)
    payload = {
        "schema": "causal_fork_v1",
        "checkpoint_iteration": checkpoint_iteration,
        "seed_lineage_freeze_from_iter": seed_lineage_freeze_from_iter,
        "fork_boundary_flush": flush,
    }
    payload["immutable_hash"] = _hash_payload(payload)
    return payload


def _validate_causal_fork_contract(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("causal fork contract is missing")
    keys = {
        "schema", "checkpoint_iteration", "seed_lineage_freeze_from_iter",
        "fork_boundary_flush", "immutable_hash",
    }
    if set(value) != keys or value.get("schema") != "causal_fork_v1":
        raise ValueError("causal fork contract schema mismatch")
    checkpoint_iteration = value["checkpoint_iteration"]
    treatment = value["seed_lineage_freeze_from_iter"]
    if isinstance(checkpoint_iteration, bool) or not isinstance(checkpoint_iteration, int) \
            or checkpoint_iteration < 0:
        raise ValueError("causal fork checkpoint iteration is invalid")
    if isinstance(treatment, bool) or not isinstance(treatment, int):
        raise ValueError("causal fork treatment is not a literal integer")
    flush = _validate_fork_flush(value["fork_boundary_flush"], checkpoint_iteration)
    unhashed = {
        "schema": value["schema"],
        "checkpoint_iteration": checkpoint_iteration,
        "seed_lineage_freeze_from_iter": treatment,
        "fork_boundary_flush": flush,
    }
    if value["immutable_hash"] != _hash_payload(unhashed):
        raise ValueError("causal fork immutable hash mismatch")
    return dict(value)


def validate_causal_fork_treatment(
    contract: Mapping[str, Any], *, resume_fork_iteration: int,
    resume_seed_lineage_freeze_from_iter: int,
) -> None:
    contract = _validate_causal_fork_contract(contract)
    checkpoint_iteration = contract["checkpoint_iteration"]
    checkpoint_treatment = contract["seed_lineage_freeze_from_iter"]
    if any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in (resume_fork_iteration, resume_seed_lineage_freeze_from_iter)
    ):
        raise ValueError("causal fork resume values must be literal integers")
    if resume_fork_iteration != checkpoint_iteration:
        raise ValueError("causal fork resume iteration mismatch")
    if checkpoint_treatment != -1:
        raise ValueError("causal fork prefix must have seed lineage freeze disabled")
    if resume_seed_lineage_freeze_from_iter not in {-1, checkpoint_iteration}:
        raise ValueError("causal fork permits only A/A or -1 to checkpoint-K treatment")


def activate_causal_fork_treatment(
    controller: Any, *, causal_fork_contract: Mapping[str, Any],
    resume_seed_lineage_freeze_from_iter: int,
) -> None:
    """Capture treatment baseline at checkpoint K before the K+1 loop begins."""
    checkpoint_iteration = _validate_causal_fork_contract(
        causal_fork_contract
    )["checkpoint_iteration"]
    if resume_seed_lineage_freeze_from_iter == -1:
        if controller is not None:
            raise ValueError("causal fork A/A resume unexpectedly constructed a freeze controller")
        return
    if resume_seed_lineage_freeze_from_iter != checkpoint_iteration or controller is None:
        raise ValueError("causal fork treatment/controller mismatch")
    controller.assert_integrity(iteration=checkpoint_iteration, context="causal-fork-resume")


def build_admission_work_signature(
    source_dispositions: Mapping[int, str], *, optimizer_step_id: int,
    supervision_view_count: int, sensor_frame_count: int,
) -> dict[str, Any]:
    """Canonical computational work signature, independent of report counters."""
    ordered = {int(key): source_dispositions[key] for key in sorted(source_dispositions)}
    signature = {
        "source_dispositions_hash": _hash_payload(ordered),
        "source_seen": len(ordered),
        "admitted": sum(value == "admitted" for value in ordered.values()),
        "holdout": sum(value == "holdout" for value in ordered.values()),
        "rejected": sum(value == "rejected" for value in ordered.values()),
        "optimizer_step_id": optimizer_step_id,
        "supervision_view_count": supervision_view_count,
        "sensor_frame_count": sensor_frame_count,
    }
    signature["work_hash"] = _hash_payload(signature)
    return signature


def qualify_exact_refinement_resume(
    runtime: Mapping[str, Any], expected_identity: Mapping[str, str],
    required_owners: set[str] = REQUIRED_EXACT_OWNERS,
) -> ExactResumeVerdict:
    """Classify a checkpoint without mutating any live state."""
    reasons: list[str] = []
    reason = runtime.get("checkpoint_reason")
    if reason != "mapping_endpoint":
        reasons.append(f"checkpoint_reason_{reason or 'missing'}")
    ingestion = runtime.get("ingestion_mode")
    if ingestion != "iter_based":
        reasons.append(f"ingestion_mode_{ingestion or 'missing'}")
    if runtime.get("live_source") is not False:
        reasons.append("live_source")
    if runtime.get("async_ingestion") is not False:
        reasons.append("async_ingestion_enabled")
    if runtime.get("online_wall_time_controller") is not False:
        reasons.append("online_wall_time_controller_enabled")
    if runtime.get("online_mode") is not False:
        reasons.append("online_mode_enabled")
    if runtime.get("bounded_queue_pipeline") is not False:
        reasons.append("bounded_queue_pipeline_enabled")
    sampling_mode = runtime.get("sampling_mode")
    if sampling_mode != "legacy":
        reasons.append(f"unsupported_sampling_mode_{sampling_mode or 'missing'}")
    if runtime.get("residual_replay_weight") != 0.0:
        reasons.append("residual_replay_enabled")
    if runtime.get("growth_staleness_max_iters") != 0:
        reasons.append("growth_staleness_brake_enabled")
    if runtime.get("adaptive_schedule_horizons") is not False:
        reasons.append("adaptive_schedule_horizons_enabled")
    if runtime.get("adaptive_capacity") is not False:
        reasons.append("adaptive_capacity_enabled")
    if runtime.get("pose_or_loop_correction") is not False:
        reasons.append("pose_or_loop_correction_enabled")
    refinement_tier = runtime.get("refinement_resolution_tier")
    if refinement_tier != "high":
        reasons.append(f"refinement_resolution_tier_{refinement_tier or 'missing'}")
    strategy = runtime.get("strategy")
    if strategy != "gsplat_energy_mcmc":
        reasons.append(f"unsupported_strategy_{strategy or 'missing'}")
    if runtime.get("insertion_batch_empty") is not True:
        reasons.append("insertion_batch_not_empty")
    if runtime.get("consolidation_queue_empty") is not True:
        reasons.append("consolidation_queue_not_empty")
    phase_metadata = runtime.get("streaming_phase")
    if not isinstance(phase_metadata, Mapping) or phase_metadata.get("phase") != "refinement":
        reasons.append("mapping_endpoint_phase_missing_or_invalid")

    envelope = runtime.get("envelope")
    if not isinstance(envelope, Mapping):
        reasons.append("streaming_checkpoint_v2_missing")
    else:
        owner_payloads = envelope.get("owners")
        if not isinstance(owner_payloads, Mapping):
            reasons.append("owner_state_missing_or_invalid")
        elif set(owner_payloads) != set(required_owners):
            reasons.append("owner_set_mismatch")
        else:
            phase_owner = owner_payloads.get("phase")
            if not isinstance(phase_owner, Mapping) or phase_owner.get("phase") != "refinement":
                reasons.append("phase_owner_not_at_refinement_boundary")
        try:
            validate_resume_envelope(dict(envelope), dict(expected_identity), set(required_owners))
        except ValueError as exc:
            message = str(exc)
            if "identity" in message:
                reasons.append("resume_identity_missing_or_mismatch")
            elif "owner" in message:
                reasons.append("owner_state_missing_or_invalid")
            else:
                reasons.append("streaming_checkpoint_v2_invalid")
    return ExactResumeVerdict(not reasons, tuple(dict.fromkeys(reasons)))


def qualify_exact_causal_fork_resume(
    runtime: Mapping[str, Any], expected_identity: Mapping[str, str],
    required_owners: set[str] = REQUIRED_CAUSAL_FORK_OWNERS,
) -> ExactResumeVerdict:
    """Classify a supported mid-mapping causal fork without mutating state."""
    reasons: list[str] = []
    if runtime.get("checkpoint_reason") != "causal_fork":
        reasons.append(f"checkpoint_reason_{runtime.get('checkpoint_reason') or 'missing'}")
    if runtime.get("ingestion_mode") != "iter_based":
        reasons.append(f"ingestion_mode_{runtime.get('ingestion_mode') or 'missing'}")
    for key, reason in (
        ("live_source", "live_source"),
        ("async_ingestion", "async_ingestion_enabled"),
        ("online_wall_time_controller", "online_wall_time_controller_enabled"),
        ("online_mode", "online_mode_enabled"),
        ("bounded_queue_pipeline", "bounded_queue_pipeline_enabled"),
        ("adaptive_schedule_horizons", "adaptive_schedule_horizons_enabled"),
        ("adaptive_capacity", "adaptive_capacity_enabled"),
        ("pose_or_loop_correction", "pose_or_loop_correction_enabled"),
        ("reactive_multires", "reactive_multires_enabled"),
    ):
        if runtime.get(key) is not False:
            reasons.append(reason)
    if runtime.get("sampling_mode") != "legacy":
        reasons.append(
            f"unsupported_sampling_mode_{runtime.get('sampling_mode') or 'missing'}"
        )
    if runtime.get("residual_replay_weight") != 0.0:
        reasons.append("residual_replay_enabled")
    if runtime.get("growth_staleness_max_iters") != 0:
        reasons.append("growth_staleness_brake_enabled")
    if runtime.get("strategy") != "gsplat_energy_mcmc":
        reasons.append(f"unsupported_strategy_{runtime.get('strategy') or 'missing'}")
    if runtime.get("admission_mode") != "hybrid_keyframe":
        reasons.append(f"unsupported_admission_mode_{runtime.get('admission_mode') or 'missing'}")
    for key, reason in (
        ("adaptive_keyframe_quality", "adaptive_keyframe_quality_enabled"),
        ("tracking_confidence", "tracking_confidence_enabled"),
        ("keyframe_tracking_confidence", "keyframe_tracking_confidence_enabled"),
        ("shadow_utility", "shadow_utility_enabled"),
        ("temporal_depth_modes", "temporal_depth_modes_enabled"),
        ("bootstrap_seed_decay", "bootstrap_seed_decay_enabled"),
        ("assimilation_control", "assimilation_control_enabled"),
        ("historical_repair_controller", "historical_repair_controller_enabled"),
        ("backpressure", "backpressure_enabled"),
        ("insertion_batching", "nondefault_insertion_batching"),
    ):
        if runtime.get(key) is not False:
            reasons.append(reason)
    if runtime.get("dataset_pose_source") is not True:
        reasons.append("stateful_or_non_dataset_pose_source")
    rollout = runtime.get("admission_rollout_mode")
    if rollout != "shadow":
        reasons.append(f"unsupported_admission_rollout_mode_{rollout or 'missing'}")
    if runtime.get("insertion_batch_empty") is not True:
        reasons.append("insertion_batch_not_empty")
    if runtime.get("computational_queues_serialized") is not True:
        reasons.append("computational_queues_not_serialized")
    if runtime.get("state_writers_joined") is not True:
        reasons.append("state_writers_not_joined")
    phase_metadata = runtime.get("streaming_phase")
    if not isinstance(phase_metadata, Mapping) or phase_metadata.get("phase") != "mapping":
        reasons.append("causal_fork_phase_missing_or_invalid")
    try:
        contract = _validate_causal_fork_contract(runtime.get("causal_fork"))
        if contract["seed_lineage_freeze_from_iter"] != -1:
            reasons.append("causal_fork_prefix_treatment_not_disabled")
    except ValueError:
        reasons.append("causal_fork_contract_invalid")

    envelope = runtime.get("envelope")
    if not isinstance(envelope, Mapping):
        reasons.append("streaming_checkpoint_v2_missing")
    else:
        owner_payloads = envelope.get("owners")
        if not isinstance(owner_payloads, Mapping):
            reasons.append("owner_state_missing_or_invalid")
        elif set(owner_payloads) != set(required_owners):
            reasons.append("owner_set_mismatch")
        else:
            phase_owner = owner_payloads.get("phase")
            if not isinstance(phase_owner, Mapping) or phase_owner.get("phase") != "mapping":
                reasons.append("phase_owner_not_at_mapping_boundary")
        try:
            validate_resume_envelope(
                dict(envelope), dict(expected_identity), set(required_owners)
            )
        except ValueError as exc:
            message = str(exc)
            if "identity" in message:
                reasons.append("resume_identity_missing_or_mismatch")
            elif "owner" in message:
                reasons.append("owner_state_missing_or_invalid")
            else:
                reasons.append("streaming_checkpoint_v2_invalid")
    return ExactResumeVerdict(not reasons, tuple(dict.fromkeys(reasons)))


class CachedCameraResolver:
    """Materialize each stable source-index camera at most once."""

    def __init__(self, materialize: Callable[[int], Any]):
        self._materialize = materialize
        self._cache: dict[int, Any] = {}

    def __call__(self, source_index: int) -> Any:
        if isinstance(source_index, bool) or not isinstance(source_index, int) or source_index < 0:
            raise ValueError(f"invalid camera source index {source_index!r}")
        if source_index not in self._cache:
            camera = self._materialize(source_index)
            if camera is None:
                raise ValueError(f"cannot resolve camera source index {source_index}")
            self._cache[source_index] = camera
        return self._cache[source_index]

    @property
    def cache(self) -> Mapping[int, Any]:
        return dict(self._cache)


def rebuild_exact_refinement_observations(
    *, db: Any, cameras: list[Any], source_dispositions: Mapping[int, str],
    get_frame: Callable[[int], Any], record: Callable[..., None],
    low_res_divisor: int, depthless_rgb_cache: bool,
) -> None:
    """Rebuild bounded train/holdout cache records through the normal recorder."""
    from eligsir.utils.partition_eval import camera_source_index

    unique: dict[int, Any] = {}
    for camera in cameras:
        source_index = camera_source_index(camera)
        if source_index is None:
            raise ValueError("restored refinement camera has no stable source index")
        unique.setdefault(int(source_index), camera)
    for source_index in sorted(unique):
        disposition = source_dispositions.get(source_index)
        if disposition not in {"admitted", "holdout"}:
            raise ValueError(
                f"restored camera {source_index} has invalid disposition {disposition!r}"
            )
        record(
            db=db,
            cam=unique[source_index],
            visible_uids=None,
            frame_id=source_index,
            submap_id=0,
            is_train=disposition == "admitted",
            is_holdout=disposition == "holdout",
            frame=get_frame(source_index),
            low_res_divisor=low_res_divisor,
            depthless_rgb_cache=depthless_rgb_cache,
        )


def _validate_sorted_unique_int64_tensor(value: Any, label: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or value.dtype != torch.int64 \
            or value.device.type != "cpu" or value.ndim != 1:
        raise ValueError(f"{label} must be a portable CPU int64 vector")
    out = value.detach().clone()
    if out.numel() > 1 and not bool(torch.all(out[1:] > out[:-1])):
        raise ValueError(f"{label} must be sorted unique")
    return out


class ExactCausalForkSceneState:
    """Exact causal-fork owner for future-affecting StreamingScene indexes."""

    OWNER_SCHEMA = "exact_causal_fork_scene_v1"
    def __init__(self, scene: Any, *, submap_manager: Any, gaussians: Any):
        self._scene = scene
        self._submap_manager = submap_manager
        self._gaussians = gaussians
        self._expected_config = self._config(scene, submap_manager)

    def __copy__(self):
        probe = object.__new__(type(self))
        probe._scene = None
        probe._submap_manager = None
        probe._gaussians = None
        probe._expected_config = copy.deepcopy(self._expected_config)
        return probe

    @staticmethod
    def _config(scene: Any, submap_manager: Any) -> dict[str, Any]:
        occupancy = scene._occupancy_index
        surface = scene._surface_cell_index
        return {
            "gpu_occupancy": bool(
                getattr(getattr(scene, "args", None), "gpu_occupancy", False)
            ),
            "occupancy_voxel_size": float(occupancy.voxel_size),
            "surface_voxel_size": float(surface.voxel_size),
            "surface_normal_bins": int(surface.normal_bins),
            "submap_frames_per_submap": int(
                submap_manager._frames_per_submap
            ),
        }

    @staticmethod
    def _validate(payload: Mapping[str, Any]) -> dict[str, Any]:
        keys = {
            "owner_schema", "gpu_occupancy", "occupancy_voxel_size",
            "occupancy_cpu_keys", "occupancy_gpu_keys", "surface_voxel_size",
            "surface_normal_bins", "surface_keys", "extent_update_counter",
            "extent_pending_growth", "extent_pending_corroborations",
            "extent_pending_driver_idx", "extent_outlier_rejections_total",
            "pending_extent_update", "submap_frames_per_submap",
            "submap_frame_count", "submap_current_id", "submaps",
            "submap0_anchored", "gaussian_current_submap_id",
        }
        if not isinstance(payload, Mapping) or set(payload) != keys \
                or payload.get("owner_schema") != ExactCausalForkSceneState.OWNER_SCHEMA:
            raise ValueError("unsupported exact causal fork scene state")
        out = copy.deepcopy(dict(payload))
        if not isinstance(out["gpu_occupancy"], bool):
            raise ValueError("exact causal scene gpu_occupancy must be boolean")
        for key in ("occupancy_voxel_size", "surface_voxel_size"):
            value = out[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)) \
                    or not math.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(f"exact causal scene {key} is invalid")
            out[key] = float(value)
        normal_bins = out["surface_normal_bins"]
        if isinstance(normal_bins, bool) or not isinstance(normal_bins, int) \
                or normal_bins <= 0:
            raise ValueError("exact causal scene surface_normal_bins is invalid")
        frames_per_submap = out["submap_frames_per_submap"]
        if isinstance(frames_per_submap, bool) or not isinstance(
            frames_per_submap, int
        ):
            raise ValueError("exact causal scene submap cadence is invalid")
        out["occupancy_cpu_keys"] = _validate_sorted_unique_int64_tensor(
            out["occupancy_cpu_keys"], "occupancy CPU keys"
        )
        gpu_keys = out["occupancy_gpu_keys"]
        if gpu_keys is not None:
            out["occupancy_gpu_keys"] = _validate_sorted_unique_int64_tensor(
                gpu_keys, "occupancy GPU keys"
            )
        if out["gpu_occupancy"] and gpu_keys is None:
            raise ValueError("gpu occupancy causal fork is missing GPU keys")
        out["surface_keys"] = _validate_sorted_unique_int64_tensor(
            out["surface_keys"], "surface keys"
        )
        for key in (
            "extent_update_counter", "extent_pending_corroborations",
            "extent_outlier_rejections_total",
        ):
            value = out[key]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"exact causal scene {key} is invalid")
        pending_growth = out["extent_pending_growth"]
        if pending_growth is not None and (
            isinstance(pending_growth, bool)
            or not isinstance(pending_growth, (int, float))
            or not math.isfinite(float(pending_growth))
            or float(pending_growth) <= 0.0
        ):
            raise ValueError("exact causal scene pending extent growth is invalid")
        if pending_growth is not None:
            out["extent_pending_growth"] = float(pending_growth)
        driver = out["extent_pending_driver_idx"]
        if driver is not None and (
            isinstance(driver, bool) or not isinstance(driver, int) or driver < 0
        ):
            raise ValueError("exact causal scene pending extent driver is invalid")
        corroborations = out["extent_pending_corroborations"]
        if (
            corroborations == 0
            and (pending_growth is not None or driver is not None)
        ) or (
            corroborations > 0
            and (pending_growth is None or driver is None)
        ):
            raise ValueError("exact causal scene pending extent fields are inconsistent")
        if out["pending_extent_update"] is not False:
            raise ValueError("exact causal fork cannot capture a pending extent update")
        for key in ("submap_frame_count", "submap_current_id", "gaussian_current_submap_id"):
            value = out[key]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"exact causal scene {key} is invalid")
        if not isinstance(out["submap0_anchored"], bool):
            raise ValueError("exact causal scene submap anchor state is invalid")
        submaps = out["submaps"]
        submap_keys = {
            "submap_id", "start_frame", "end_frame", "T_world_submap",
            "T_world_submap_raw", "gaussian_count",
        }
        if not isinstance(submaps, list) or not submaps:
            raise ValueError("exact causal scene submaps are invalid")
        for expected_id, state in enumerate(submaps):
            if not isinstance(state, Mapping) or set(state) != submap_keys \
                    or state["submap_id"] != expected_id:
                raise ValueError("exact causal scene submap IDs are invalid")
            for key in ("start_frame", "gaussian_count"):
                value = state[key]
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise ValueError(f"exact causal scene submap {key} is invalid")
            end_frame = state["end_frame"]
            if isinstance(end_frame, bool) or not isinstance(end_frame, int) \
                    or end_frame < -1:
                raise ValueError("exact causal scene submap end frame is invalid")
            for key in ("T_world_submap", "T_world_submap_raw"):
                transform = state[key]
                if not isinstance(transform, torch.Tensor) \
                        or transform.device.type != "cpu" \
                        or transform.shape != (4, 4) \
                        or not transform.dtype.is_floating_point \
                        or not bool(torch.isfinite(transform).all()):
                    raise ValueError("exact causal scene submap transform is invalid")
        if out["submap_current_id"] != len(submaps) - 1 \
                or out["gaussian_current_submap_id"] != out["submap_current_id"]:
            raise ValueError("exact causal scene current submap state is inconsistent")
        return out

    def resume_state_dict(self) -> dict[str, Any]:
        scene = self._scene
        if scene is None:
            raise ValueError("validation-only scene owner cannot capture state")
        occupancy_gpu = scene._occupancy_index.keys_gpu
        payload = {
            "owner_schema": self.OWNER_SCHEMA,
            **self._config(scene, self._submap_manager),
            "occupancy_cpu_keys": torch.as_tensor(
                np.asarray(scene._occupancy_index.keys, dtype=np.int64).copy(),
                dtype=torch.int64,
            ),
            "occupancy_gpu_keys": None if occupancy_gpu is None else (
                occupancy_gpu.detach().to(device="cpu", dtype=torch.int64).clone()
            ),
            "surface_keys": torch.as_tensor(
                np.asarray(scene._surface_cell_index._keys, dtype=np.int64).copy(),
                dtype=torch.int64,
            ),
            "extent_update_counter": int(scene._extent_update_counter),
            "extent_pending_growth": scene._extent_pending_growth,
            "extent_pending_corroborations": int(
                scene._extent_pending_corroborations
            ),
            "extent_pending_driver_idx": scene._extent_pending_driver_idx,
            "extent_outlier_rejections_total": int(
                scene._extent_outlier_rejections_total
            ),
            "pending_extent_update": bool(
                getattr(scene, "_pending_extent_update", False)
            ),
            "submap_frame_count": int(self._submap_manager._frame_count),
            "submap_current_id": int(self._submap_manager._current_id),
            "submaps": [
                {
                    "submap_id": int(state.submap_id),
                    "start_frame": int(state.start_frame),
                    "end_frame": int(state.end_frame),
                    "T_world_submap": torch.as_tensor(
                        np.asarray(state.T_world_submap).copy()
                    ).cpu(),
                    "T_world_submap_raw": torch.as_tensor(
                        np.asarray(state.T_world_submap_raw).copy()
                    ).cpu(),
                    "gaussian_count": int(state.gaussian_count),
                }
                for _, state in sorted(self._submap_manager._submaps.items())
            ],
            "submap0_anchored": bool(self._submap_manager._submap0_anchored),
            "gaussian_current_submap_id": int(
                getattr(self._gaussians, "_current_submap_id", -1)
            ),
        }
        return self._validate(payload)

    def load_resume_state_dict(self, payload: Mapping[str, Any], resolver=None) -> None:
        del resolver
        restored = self._validate(payload)
        restored_config = {
            key: restored[key]
            for key in self._expected_config
        }
        if restored_config != self._expected_config:
            raise ValueError("exact causal scene index configuration mismatch")
        if self._scene is None:
            return
        scene = self._scene
        scene._occupancy_index.keys = restored["occupancy_cpu_keys"].numpy().copy()
        gpu_keys = restored["occupancy_gpu_keys"]
        scene._occupancy_index.keys_gpu = None if gpu_keys is None else gpu_keys.to(
            device=scene.gaussians.get_xyz.device, dtype=torch.int64
        ).clone()
        scene._surface_cell_index._keys = restored["surface_keys"].numpy().copy()
        scene._extent_update_counter = restored["extent_update_counter"]
        scene._extent_pending_growth = restored["extent_pending_growth"]
        scene._extent_pending_corroborations = restored[
            "extent_pending_corroborations"
        ]
        scene._extent_pending_driver_idx = restored["extent_pending_driver_idx"]
        scene._extent_outlier_rejections_total = restored[
            "extent_outlier_rejections_total"
        ]
        scene._pending_extent_update = False
        from eligsir.utils.submap_manager import SubmapState
        manager = self._submap_manager
        manager._frame_count = restored["submap_frame_count"]
        manager._current_id = restored["submap_current_id"]
        manager._submaps = {
            item["submap_id"]: SubmapState(
                submap_id=item["submap_id"],
                start_frame=item["start_frame"],
                end_frame=item["end_frame"],
                T_world_submap=item["T_world_submap"].numpy().copy(),
                T_world_submap_raw=item["T_world_submap_raw"].numpy().copy(),
                gaussian_count=item["gaussian_count"],
            )
            for item in restored["submaps"]
        }
        manager._submap0_anchored = restored["submap0_anchored"]
        self._gaussians._current_submap_id = restored[
            "gaussian_current_submap_id"
        ]


class ExactCausalForkObservationDBState:
    """Portable exact owner for all coverage-affecting observation history."""

    OWNER_SCHEMA = "exact_causal_fork_observation_db_v1"

    def __init__(self, observation_db: Any):
        self._db = observation_db
        self._expected_config = {
            "device": str(torch.device(observation_db._device)),
            "max_active_supervision_frames": int(
                observation_db._max_resolution_cache_frames
            ),
        }

    def __copy__(self):
        probe = object.__new__(type(self))
        probe._db = None
        probe._expected_config = copy.deepcopy(self._expected_config)
        return probe

    @classmethod
    def _validate(cls, payload: Mapping[str, Any]) -> dict[str, Any]:
        keys = {
            "owner_schema", "device", "max_active_supervision_frames",
            "records", "source_identity_to_frame_id",
        }
        if not isinstance(payload, Mapping) or set(payload) != keys \
                or payload.get("owner_schema") != cls.OWNER_SCHEMA:
            raise ValueError("unsupported exact causal fork observation DB state")
        out = copy.deepcopy(dict(payload))
        try:
            out["device"] = str(torch.device(out["device"]))
        except (TypeError, RuntimeError) as exc:
            raise ValueError("exact causal observation DB device is invalid") from exc
        bound = out["max_active_supervision_frames"]
        if isinstance(bound, bool) or not isinstance(bound, int) or bound < 0:
            raise ValueError("exact causal observation DB cache bound is invalid")
        records = out["records"]
        if not isinstance(records, list):
            raise ValueError("exact causal observation DB records are invalid")
        expected_record_keys = {
            "frame_id", "pose_c2w", "timestamp", "submap_id", "buffer_stage",
            "visible_gaussian_uids", "resolution_cache_evicted",
            "resolution_cache_absent_reason", "source_identity", "source_ordinal",
        }
        prior_id = -1
        for record in records:
            if not isinstance(record, Mapping) or set(record) != expected_record_keys:
                raise ValueError("exact causal observation record keys mismatch")
            frame_id = record["frame_id"]
            if isinstance(frame_id, bool) or not isinstance(frame_id, int) \
                    or frame_id <= prior_id:
                raise ValueError("exact causal observation frame ids must increase")
            prior_id = frame_id
            pose = record["pose_c2w"]
            if not isinstance(pose, torch.Tensor) or pose.device.type != "cpu" \
                    or pose.shape != (4, 4) or not pose.dtype.is_floating_point \
                    or not bool(torch.isfinite(pose).all()):
                raise ValueError("exact causal observation pose is invalid")
            timestamp = record["timestamp"]
            if isinstance(timestamp, bool) or not isinstance(timestamp, (int, float)) \
                    or not math.isfinite(float(timestamp)):
                raise ValueError("exact causal observation timestamp is invalid")
            submap_id = record["submap_id"]
            if isinstance(submap_id, bool) or not isinstance(submap_id, int) \
                    or submap_id < 0:
                raise ValueError("exact causal observation submap is invalid")
            from eligsir.mapping.observation_db import BufferStage
            try:
                BufferStage(record["buffer_stage"])
            except (TypeError, ValueError) as exc:
                raise ValueError("exact causal observation buffer stage is invalid") from exc
            uids = record["visible_gaussian_uids"]
            if not isinstance(uids, torch.Tensor) or uids.device.type != "cpu" \
                    or uids.dtype != torch.int64 or uids.ndim != 1 \
                    or (uids.numel() and bool(torch.any(uids < 0))):
                raise ValueError("exact causal observation visible UIDs are invalid")
            if not isinstance(record["resolution_cache_evicted"], bool):
                raise ValueError("exact causal observation cache eviction state is invalid")
            absent = record["resolution_cache_absent_reason"]
            if absent is not None and (not isinstance(absent, str) or not absent):
                raise ValueError("exact causal observation cache absence reason is invalid")
            identity = record["source_identity"]
            if identity is not None and (not isinstance(identity, str) or not identity):
                raise ValueError("exact causal observation source identity is invalid")
            ordinal = record["source_ordinal"]
            if ordinal is not None and (
                isinstance(ordinal, bool) or not isinstance(ordinal, int) or ordinal < 0
            ):
                raise ValueError("exact causal observation source ordinal is invalid")
        mapping = out["source_identity_to_frame_id"]
        record_by_id = {record["frame_id"]: record for record in records}
        indexed_records = [
            record for record in records if record["source_identity"] is not None
        ]
        expected_mapping = {
            record["source_identity"]: record["frame_id"]
            for record in indexed_records
        }
        if not isinstance(mapping, Mapping) or any(
            not isinstance(key, str) or not key
            or isinstance(value, bool) or not isinstance(value, int)
            or value not in record_by_id
            or record_by_id[value]["source_identity"] != key
            for key, value in mapping.items()
        ) or len(expected_mapping) != len(indexed_records) \
                or dict(mapping) != expected_mapping:
            raise ValueError("exact causal observation source index is invalid")
        out["source_identity_to_frame_id"] = dict(mapping)
        return out

    def resume_state_dict(self) -> dict[str, Any]:
        if self._db is None:
            raise ValueError("validation-only observation owner cannot capture state")
        # A retained prepared COMMIT is not ordinary coverage history: it is an
        # unresolved first-supervision obligation whose opaque value contains
        # the prepared COLD route (tracking, disposition, control token and
        # pixel/reload ownership).  This v1 exact-fork schema intentionally
        # does not serialize that route.  Omitting the marker would silently
        # turn an admitted-but-unsupervised source into a permanently
        # unserviceable record after resume, so fail closed until a dedicated
        # prepared-resume schema owns the complete state transition.
        if any(
            record.prepared_commit_obligation is not None
            for record in self._db._frames.values()
        ):
            raise RuntimeError(
                "exact causal observation resume cannot capture unresolved "
                "prepared COMMIT obligation"
            )
        records = []
        for frame_id in sorted(self._db._frames):
            record = self._db._frames[frame_id]
            records.append(
                {
                    "frame_id": int(record.frame_id),
                    "pose_c2w": torch.as_tensor(
                        np.asarray(record.pose_c2w).copy()
                    ).cpu(),
                    "timestamp": float(record.timestamp),
                    "submap_id": int(record.submap_id),
                    "buffer_stage": record.buffer_stage.value,
                    "visible_gaussian_uids": record.visible_gaussian_uids.detach().to(
                        device="cpu", dtype=torch.int64
                    ).clone(),
                    "resolution_cache_evicted": bool(record.resolution_cache_evicted),
                    "resolution_cache_absent_reason": record.resolution_cache_absent_reason,
                    "source_identity": record.source_identity,
                    "source_ordinal": record.source_ordinal,
                }
            )
        return self._validate(
            {
                "owner_schema": self.OWNER_SCHEMA,
                **self._expected_config,
                "records": records,
                "source_identity_to_frame_id": dict(
                    self._db._source_identity_to_frame_id
                ),
            }
        )

    def load_resume_state_dict(self, payload: Mapping[str, Any], resolver=None) -> None:
        del resolver
        restored = self._validate(payload)
        restored_config = {
            key: restored[key]
            for key in self._expected_config
        }
        if restored_config != self._expected_config:
            raise ValueError("exact causal observation DB configuration mismatch")
        if self._db is None:
            return
        from eligsir.mapping.observation_db import BufferStage, FrameRecord
        db = self._db
        db._frames.clear()
        db._cold_pixels.clear()
        db._source_identity_to_frame_id.clear()
        db._active_supervision_cache._order.clear()
        db._max_uid_seen = -1
        db._max_frame_id_seen = -1
        db._dirty = True
        db._csr = None
        db._incremental_valid = True
        db._crow_built_upto_row = -1
        db._crow = [0]
        db._buf_col = None
        db._buf_len = 0
        for item in restored["records"]:
            record = FrameRecord(
                frame_id=item["frame_id"],
                pose_c2w=item["pose_c2w"].numpy().copy(),
                timestamp=float(item["timestamp"]),
                submap_id=item["submap_id"],
                buffer_stage=BufferStage(item["buffer_stage"]),
                visible_gaussian_uids=item["visible_gaussian_uids"].to(
                    device=db._device, dtype=torch.int64
                ).clone(),
                resolution_cache={},
                resolution_cache_evicted=item["resolution_cache_evicted"],
                resolution_cache_absent_reason=item[
                    "resolution_cache_absent_reason"
                ],
                source_identity=item["source_identity"],
                source_ordinal=item["source_ordinal"],
            )
            db.add_frame(record)
        db._source_identity_to_frame_id.update(
            restored["source_identity_to_frame_id"]
        )


_TRAINER_KEYS = {
    "source_idx", "source_frame_count", "n_frames_ingested", "last_ingest_iter",
    "optimizer_step_id", "supervision_view_count", "rasterizer_call_count",
    "supervised_pixel_count", "sensor_frame_count", "mutation_event_id",
    "depth_mutation_event_id", "mcmc_mutation_event_id", "total_inserted",
    "last_flush_iteration", "iter_since_new_frame", "scene_steps_since_new_frame",
    "opt_batch_distinct_total", "opt_batch_count", "per_source_supervision_counts",
    "bootstrap_source_indices", "source_dispositions", "insertion_batch_empty",
    "consolidation_queue_empty",
}
_NONNEGATIVE_TRAINER_INTS = _TRAINER_KEYS - {
    "per_source_supervision_counts", "bootstrap_source_indices",
    "source_dispositions", "insertion_batch_empty", "consolidation_queue_empty",
}


class ExactTrainerState:
    """Validated owner for future-affecting loop-local counters."""

    OWNER_SCHEMA = "exact_refinement_trainer_v1"

    def __init__(self, state: Mapping[str, Any]):
        self.state = self._validate(state)

    @classmethod
    def _validate(cls, state: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(state, Mapping) or set(state) != _TRAINER_KEYS:
            raise ValueError("exact trainer state keys mismatch")
        out = copy.deepcopy(dict(state))
        for key in _NONNEGATIVE_TRAINER_INTS:
            value = out[key]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"exact trainer {key} must be a non-negative integer")
        for key in ("insertion_batch_empty", "consolidation_queue_empty"):
            if out[key] is not True:
                label = "insertion batch" if key.startswith("insertion") else "consolidation queue"
                raise ValueError(f"exact trainer {label} must be empty")
        counts = out["per_source_supervision_counts"]
        if not isinstance(counts, Mapping) or any(
            isinstance(k, bool) or not isinstance(k, int) or k < 0
            or isinstance(v, bool) or not isinstance(v, int) or v < 0
            for k, v in counts.items()
        ):
            raise ValueError("invalid per-source supervision counters")
        bootstrap = out["bootstrap_source_indices"]
        if not isinstance(bootstrap, list) or any(
            isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in bootstrap
        ):
            raise ValueError("invalid bootstrap source identities")
        dispositions = out["source_dispositions"]
        if not isinstance(dispositions, Mapping) or any(
            isinstance(k, bool) or not isinstance(k, int) or k < 0
            or not isinstance(v, str) or not v for k, v in dispositions.items()
        ):
            raise ValueError("invalid source dispositions")
        if out["source_idx"] != out["source_frame_count"]:
            raise ValueError("exact trainer state is not at 100% source coverage")
        return out

    def resume_state_dict(self) -> dict[str, Any]:
        return {"owner_schema": self.OWNER_SCHEMA, **copy.deepcopy(self.state)}

    def load_resume_state_dict(self, payload: Mapping[str, Any], resolver=None) -> None:
        if not isinstance(payload, Mapping) or payload.get("owner_schema") != self.OWNER_SCHEMA:
            raise ValueError("unsupported exact trainer checkpoint state")
        restored = self._validate({k: v for k, v in payload.items() if k != "owner_schema"})
        self.state = restored


_CAUSAL_TRAINER_KEYS = (_TRAINER_KEYS - {"consolidation_queue_empty"}) | {
    "consolidation_queue_source_indices", "fork_boundary_flush",
    "admission_work_signature",
}
_CAUSAL_TRAINER_NONNEGATIVE = _NONNEGATIVE_TRAINER_INTS


class ExactCausalForkTrainerState:
    """Validated mid-mapping loop state at the explicit fork boundary."""

    OWNER_SCHEMA = "exact_causal_fork_trainer_v1"

    def __init__(self, state: Mapping[str, Any]):
        self.state = self._validate(state)

    @classmethod
    def _validate(cls, state: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(state, Mapping) or set(state) != _CAUSAL_TRAINER_KEYS:
            raise ValueError("exact causal fork trainer state keys mismatch")
        out = copy.deepcopy(dict(state))
        for key in _CAUSAL_TRAINER_NONNEGATIVE:
            value = out[key]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(
                    f"exact causal fork trainer {key} must be a non-negative integer"
                )
        if out["insertion_batch_empty"] is not True:
            raise ValueError("exact causal fork insertion batch must be empty")
        if out["source_idx"] > out["source_frame_count"]:
            raise ValueError("exact causal fork source cursor exceeds source size")
        if out["n_frames_ingested"] != out["source_idx"]:
            raise ValueError("exact causal fork source and ingest cursors differ")
        counts = out["per_source_supervision_counts"]
        if not isinstance(counts, Mapping) or any(
            isinstance(k, bool) or not isinstance(k, int) or k < 0
            or isinstance(v, bool) or not isinstance(v, int) or v < 0
            for k, v in counts.items()
        ):
            raise ValueError("invalid causal fork per-source supervision counters")
        bootstrap = out["bootstrap_source_indices"]
        if not isinstance(bootstrap, list) or len(set(bootstrap)) != len(bootstrap) or any(
            isinstance(v, bool) or not isinstance(v, int)
            or v < 0 or v >= out["source_idx"] for v in bootstrap
        ):
            raise ValueError("invalid causal fork bootstrap source identities")
        dispositions = out["source_dispositions"]
        if not isinstance(dispositions, Mapping) or set(dispositions) != set(
            range(out["source_idx"])
        ) or any(v not in {"admitted", "holdout", "rejected"} for v in dispositions.values()):
            raise ValueError("invalid causal fork source dispositions")
        queue = out["consolidation_queue_source_indices"]
        if not isinstance(queue, list) or any(
            isinstance(v, bool) or not isinstance(v, int)
            or v < 0 or v >= out["source_idx"] for v in queue
        ):
            raise ValueError("invalid causal fork consolidation queue")
        flush = out["fork_boundary_flush"]
        if not isinstance(flush, Mapping):
            raise ValueError("causal fork boundary flush is invalid")
        out["fork_boundary_flush"] = _validate_fork_flush(
            flush, flush.get("iteration", -1)
        )
        if out["fork_boundary_flush"]["optimizer_step_id"] != out["optimizer_step_id"] \
                or out["fork_boundary_flush"]["mutation_event_id"] != out["mutation_event_id"]:
            raise ValueError("causal fork flush clocks do not match trainer clocks")
        expected_signature = build_admission_work_signature(
            dispositions,
            optimizer_step_id=out["optimizer_step_id"],
            supervision_view_count=out["supervision_view_count"],
            sensor_frame_count=out["sensor_frame_count"],
        )
        if out["admission_work_signature"] != expected_signature:
            raise ValueError("causal fork admission work signature mismatch")
        return out

    def resume_state_dict(self) -> dict[str, Any]:
        return {"owner_schema": self.OWNER_SCHEMA, **copy.deepcopy(self.state)}

    def load_resume_state_dict(self, payload: Mapping[str, Any], resolver=None) -> None:
        del resolver
        if not isinstance(payload, Mapping) or payload.get("owner_schema") != self.OWNER_SCHEMA:
            raise ValueError("unsupported exact causal fork trainer checkpoint state")
        self.state = self._validate(
            {key: value for key, value in payload.items() if key != "owner_schema"}
        )


def restore_exact_resume_owners(
    *, envelope: Mapping[str, Any], expected_identity: Mapping[str, str],
    owners: Mapping[str, Any], resolver: CachedCameraResolver,
    restore_rng: bool = True,
) -> None:
    """Validate all owners first, mutate them second, restore RNG last."""
    required = set(owners)
    if required not in (REQUIRED_EXACT_OWNERS, REQUIRED_CAUSAL_FORK_OWNERS):
        raise ValueError("exact resume owner set mismatch")
    validate_resume_envelope(dict(envelope), dict(expected_identity), required)
    payloads = envelope["owners"]
    if set(payloads) != required:
        raise ValueError("exact resume owner set mismatch")
    for name, owner in owners.items():
        probe = copy.copy(owner)
        loader = getattr(probe, "load_resume_state_dict", None)
        if not callable(loader):
            raise ValueError(f"resume owner {name!r} has no compatible loader")
        loader(copy.deepcopy(payloads[name]), resolver)
    for name, owner in owners.items():
        owner.load_resume_state_dict(copy.deepcopy(payloads[name]), resolver)
    if restore_rng:
        restore_rng_state(envelope["rng"])


def persistent_endpoint_paths(
    local_model_path: str, persist_model_path: str | None, iteration: int,
) -> dict[str, str]:
    """Publish canonical paths while retaining the resolved staging location."""
    root = persist_model_path or local_model_path
    endpoint = str(Path(root) / "mapping_endpoint")
    return {
        "directory": endpoint,
        "checkpoint": str(Path(endpoint) / f"chkpnt{int(iteration)}.pth"),
        "point_cloud": str(Path(endpoint) / "point_cloud.ply"),
        "report": str(Path(root) / "comparison" / "mapping_endpoint" / "report.json"),
        "phase_metadata": str(Path(endpoint) / "streaming_phase_metadata.json"),
        "test_cameras": str(Path(root) / "test_cameras.json"),
        "holdout_gt": str(Path(root) / "holdout_gt"),
        "resolved_local_directory": str(Path(local_model_path) / "mapping_endpoint"),
    }


def verify_published_endpoint(
    local_model_path: str, published_paths: Mapping[str, str], *, iteration: int,
) -> None:
    """Fail closed unless the score-bearing endpoint is durably identical."""
    local_root = Path(local_model_path)
    local_paths = {
        "checkpoint": local_root / "mapping_endpoint" / f"chkpnt{int(iteration)}.pth",
        "point_cloud": local_root / "mapping_endpoint" / "point_cloud.ply",
        "phase_metadata": local_root / "mapping_endpoint" / "streaming_phase_metadata.json",
        "report": local_root / "comparison" / "mapping_endpoint" / "report.json",
    }
    for name, local_path in local_paths.items():
        published = Path(published_paths.get(name, ""))
        for label, path in (("local", local_path), ("published", published)):
            if not path.is_file():
                raise RuntimeError(f"{name} {label} artifact is missing: {path}")
            if path.stat().st_size <= 0:
                raise RuntimeError(f"{name} {label} artifact is empty: {path}")
        if local_path.stat().st_size != published.stat().st_size:
            raise RuntimeError(f"{name} local/published size mismatch")

    json_pairs = (
        ("local", local_paths["phase_metadata"], local_paths["report"]),
        (
            "published",
            Path(published_paths["phase_metadata"]),
            Path(published_paths["report"]),
        ),
    )
    for location, phase_path, report_path in json_pairs:
        try:
            phase = json.loads(phase_path.read_text())
            report = json.loads(report_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"mapping endpoint JSON is invalid: {exc}") from exc
        report_phase = (
            report.get("streaming_phase") if isinstance(report, Mapping) else None
        )
        for label, metadata in (
            (f"{location} phase metadata", phase),
            (f"{location} report", report_phase),
        ):
            if not isinstance(metadata, Mapping):
                raise RuntimeError(f"{label} streaming phase is missing")
            exact = metadata.get("exact_resume")
            valid = (
                metadata.get("phase") == "refinement"
                and metadata.get("mapping_end_iteration") == int(iteration)
                and metadata.get("source_coverage_fraction") == 1.0
                and isinstance(exact, Mapping)
                and exact.get("classification") == "EXACT"
            )
            if not valid:
                raise RuntimeError(
                    f"{label} does not describe an exact 100% mapping endpoint"
                )
