"""
Streaming-replay training loop for 3DGS-MCMC.

Activated by --streaming_replay.  Simulates real-time RGB-D input by
consuming an existing dataset (RGBDSequence / TUM / ScanNet) as an ordered
frame stream instead of loading all cameras at once.

Phase 1  – ordered windowed training
    Frames arrive every streaming_steps_per_frame iterations (deterministic)
    or at streaming_input_fps wall-clock rate.  The optimizer sees only the
    recent keyframe window plus an occasional older replay frame.

Phase 2  – incremental depth insertion
    Each new RGB-D frame backprojects its depth into world space, voxel-
    downsamples the result, filters away already-covered regions, and appends
    new Gaussians via gaussians.add_points_as_gaussians().

The offline training() path in train.py is untouched.
"""

from __future__ import annotations

import dataclasses as _dataclasses
import json
import math
import os
import queue
import sys
import threading
import time

from eligsir.runtime.save_worker import AsyncSaveWorker  # noqa: E402
from eligsir.mapping.batch_contract import validate_supervision_batch_size
from eligsir.runtime.finalize import (  # noqa: E402
    finalize_run,
    emit_run_evidence as _emit_run_evidence,
)
from eligsir.runtime.report_kwargs import (  # noqa: E402
    build_streaming_report_kwargs as _build_streaming_report_kwargs,
)
from eligsir.io.snapshots import (
    snapshot_gaussians_for_ply as _snapshot_for_ply,
    subset_ply_snapshot as _subset_snapshot,
    write_ply_from_snapshot as _write_ply,
)
from argparse import Namespace
from typing import Optional

import numpy as np
import torch
from torchvision.utils import save_image
from tqdm import tqdm

from gaussian_renderer import render
from arguments import apply_renderer_optimization_params

# --- TEMP DIAGNOSTIC (memory-mystery investigation, revert before merge) ---
_MEMPROBE = bool(os.environ.get("SPLATO_MEMPROBE", ""))


def resolve_streaming_renderer_configuration(pipe, optimization):
    return apply_renderer_optimization_params(pipe, optimization)


def _memprobe(tag, iteration, extra=""):
    if not _MEMPROBE or not torch.cuda.is_available():
        return
    torch.cuda.synchronize()
    alloc = torch.cuda.memory_allocated() / 1e9
    reserv = torch.cuda.memory_reserved() / 1e9
    peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"[MEMPROBE] it={iteration} {tag} alloc={alloc:.3f}GB reserved={reserv:.3f}GB peak={peak:.3f}GB {extra}", flush=True)
# --- END TEMP DIAGNOSTIC ---
from scene import GaussianModel
from scene.gsplat_model import GsplatGaussianModel
from scene.streaming_scene import StreamingScene
from eligsir.utils.image_utils import psnr
from eligsir.utils.score_contract import apply_declared_score_clamp_pair, score_tensor_from_camera
from eligsir.utils.loss_utils import l1_loss, ssim
from eligsir.utils.assimilation_pressure import (
    assimilation_control_config,
    classify_assimilation_pressure,
    shadow_decision,
)
from eligsir.utils.depth_alpha_diagnostics import compute_alpha_bin_residuals, log_alpha_bin_residuals_to_tb
from eligsir.utils.realized_cadence import RealizedCadenceTracker, realized_cadence_report_block
from eligsir.utils.admission_cadence import (
    AdmissionCadenceRecorder,
    admission_cadence_report_block,
)
from eligsir.utils.supervision_work import (  # noqa: F401 (production consumer, first wiring)
    rasterizer_calls_per_optimization_step,
    supervised_pixels_per_optimization_step,
    supervision_views_per_optimization_step,
    supervision_work_report_block,
)
from eligsir.mapping.scheduler import (
    INGESTION_MODE_ITER_BASED,
    INGESTION_REASON_ONLINE_MODE_DEFAULT,
    ONLINE_MODE_DATASET_FPS_WARNING,
    FrameScheduler,
    build_spf_resolution,
    resolve_coverage_planning_budget,
    resolve_ingestion_mode_with_reason,
    resolve_source_decimation,
    resolve_source_sampling_application,
)
from eligsir.mapping.resolution_provenance import (
    NATIVE_SOURCE_FRAME_SOURCE,
    NATIVE_SOURCE_INTRINSICS_JSON,
    NATIVE_SOURCE_UNKNOWN,
    ResolutionTierLog,
    build_resolution_provenance,
    read_native_resolution,
    resolution_report_block,
)
from eligsir.mapping.optimization_runtime import (
    StreamingOptimizationRuntime,
    merge_deferred_prune_masks,
    run_streaming_optimization_step,
)
from eligsir.runtime.application import (
    MappingApplicationOptions,
    create_mapping_application,
)
from eligsir.runtime.streaming_sink import (
    StreamingIngestCallbacks,
    StreamingSceneMappingSink,
)
from eligsir.optimization.gaussian.contracts import RandomStateToken
from eligsir.mapping.evidence.builder import EvidenceBuilder
from eligsir.mapping.evidence.requirements import EvidenceRequirements
from eligsir.mapping.evidence.gaussian_history import GaussianEvidenceHistory
from eligsir.io.sources import make_frame_source
from eligsir.utils.streaming_depth_lock import (
    consensus_clean_mask,
    depth_conflict_enabled,
    depth_conflict_prune_mask,
    depth_lock_conflict_threshold,
    depth_lock_enabled,
    depth_lock_geometry_stats,
    depth_lock_candidate_mask,
    depth_locked_mask,
    ensure_depth_lock_buffers,
    record_depth_conflicts,
    record_depth_support_views,
    restore_depth_locked_centers,
    zero_depth_locked_mean_grads,
)
from eligsir.utils.general_utils import inverse_sigmoid
from eligsir.runtime.streaming_phase import Phase, StreamingPhaseController
from eligsir.checkpointing.exact_refinement_resume import (
    activate_causal_fork_treatment,
    build_causal_fork_contract,
    build_admission_work_signature,
    CachedCameraResolver,
    ExactCausalForkObservationDBState,
    ExactCausalForkSceneState,
    ExactCausalForkTrainerState,
    ExactTrainerState,
    REQUIRED_CAUSAL_FORK_OWNERS,
    REQUIRED_EXACT_OWNERS,
    authoritative_identity_hashes,
    build_exact_resume_identity,
    persistent_endpoint_paths,
    qualify_exact_causal_fork_resume,
    qualify_exact_refinement_resume,
    rebuild_exact_refinement_observations,
    restore_exact_resume_owners,
    validate_causal_fork_treatment,
    verify_published_endpoint,
)
from eligsir.checkpointing.streaming_resume import (
    build_resume_envelope,
    restore_rng_state,
)
# Issue #168: resolved run-mode axes for report.json["execution_semantics"].
# Module-level (not a lazy in-function import) so the report closure does not
# depend on a later local import having already bound the name.
# Issue #701: also import the plain (unaliased) name plus
# format_execution_semantics, for the startup banner print in
# streaming_training() -- see that call site's comment.
from eligsir.utils.execution_semantics import (
    ADMISSION_TARGET_FORCE_ALL_STRESS,
    ADMISSION_TARGET_SOFT_ALL,
    build_execution_semantics_block,
    build_execution_semantics_block as _execution_semantics_from_args,
    format_execution_semantics,
    resolve_admission_semantics,
)
from eligsir.core.stop_policy import should_apply_adaptive_schedule_horizons
from geometry.priors import GeometrySource
from eligsir.mapping.scheduler import (
    FrameQualityConfig,
    RunningBlurExposureStats,
    evaluate_frame_quality,
)
from eligsir.mapping.scheduler.frame_quality import mean_luma
from eligsir.mapping.scheduler import KeyframeScoringConfig, score_keyframe, score_to_stats
from eligsir.mapping.scheduler.admission_preflight import check_admission_preflight
from eligsir.utils.surface_losses import compute_surface_anchor_losses
from eligsir.utils.admission_fingerprint import (
    AdmissionFingerprint,
    keyframe_decision_sidecar_fields,
)
from eligsir.utils.coverage_ledger import (
    CoverageLedger,
    MANAGED_STAGED_SOURCE_FRAMES_ENV,
    resolve_finite_source_denominator,
    resolve_managed_staged_source_denominator,
)

try:
    from torch.utils.tensorboard import SummaryWriter
    _TB = True
except ImportError:
    _TB = False


def _source_disposition_state(streaming_scene) -> tuple[bool, bool]:
    """Return full-source completion from the source cursor, including holdouts."""

    source_idx = int(streaming_scene._source_idx)
    frame_count = int(streaming_scene._frame_count)
    complete = source_idx >= frame_count
    return complete, complete


def _streaming_mcmc_schedule(*, iteration, current_n, cap_max, cfg, streaming_scene):
    """Resolve MCMC phase from the canonical full-source completion boundary."""

    from eligsir.utils.mcmc_schedule import get_mcmc_schedule

    source_exhausted, _ = _source_disposition_state(streaming_scene)
    return get_mcmc_schedule(
        iteration=iteration,
        current_n=current_n,
        cap_max=cap_max,
        cfg=cfg,
        ingestion_complete=source_exhausted,
    )


def _write_streaming_phase_metadata(model_path: str, metadata: dict) -> str:
    """Atomically persist the small mapping/refinement phase record."""

    os.makedirs(model_path, exist_ok=True)
    out = os.path.join(model_path, "streaming_phase_metadata.json")
    tmp = out + ".tmp"
    with open(tmp, "w") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
    os.replace(tmp, out)
    return out


def _exact_streaming_identity(
    streaming_scene,
    dataset,
    args,
    opt,
    *,
    ingestion_mode: str,
    densification_strategy: str,
    model_layout: str,
    causal_fork: bool = False,
) -> dict[str, str]:
    """Resolve the immutable identity used by mapping-endpoint continuation."""
    frames = []
    for source_index in range(int(streaming_scene._frame_count)):
        frame = streaming_scene._get_source_frame(source_index)
        frames.append(
            {
                "index": source_index,
                "rgb_path": getattr(frame, "rgb_path", None),
                "depth_path": getattr(frame, "depth_path", None),
                "timestamp": getattr(frame, "timestamp", None),
                "c2w": getattr(frame, "c2w", None),
                "intrinsics": [
                    getattr(frame, name, None)
                    for name in ("width", "height", "fx", "fy", "cx", "cy", "depth_scale")
                ],
            }
        )
    eval_metadata = dict(streaming_scene.eval_metadata() or {})

    def _stable_args(namespace, prefix=""):
        portable_types = (type(None), bool, int, float, str, list, tuple, dict)
        return {
            prefix + key: value
            for key, value in vars(namespace).items()
            if not key.startswith("_") and isinstance(value, portable_types)
        }

    authoritative = authoritative_identity_hashes(
        {"experiment_origin": getattr(args, "experiment_origin", None), "args": vars(args)}
    )
    if isinstance(eval_metadata.get("holdout_hash"), str) and eval_metadata["holdout_hash"]:
        authoritative["holdout_hash"] = eval_metadata["holdout_hash"]
    source_artifact = args.w3_02_source_artifact
    if isinstance(source_artifact, str) and source_artifact:
        authoritative["source_hash"] = source_artifact
    return build_exact_resume_identity(
        scene_identity={
            "source_path": getattr(dataset, "source_path", None),
            "frame_count": int(streaming_scene._frame_count),
            "frames": frames,
        },
        holdout_identity={
            "eval_metadata": eval_metadata,
            "holdout_indices": sorted(streaming_scene._holdout_indices or []),
            "guard_indices": sorted(streaming_scene._guard_indices or []),
        },
        effective_args={**_stable_args(args), **_stable_args(opt, "opt.")},
        runtime_identity={
            "python": sys.version,
            "torch": torch.__version__,
            "torch_hip": getattr(torch.version, "hip", None),
            # Runtime-assigned unconditionally by train.py before dispatch.
            "git_commit": args.git_commit,
            "git_dirty": args.git_dirty,
            "ingestion_mode": ingestion_mode,
            "densification_strategy": densification_strategy,
            "model_layout": model_layout,
        },
        authoritative=authoritative,
        causal_fork=causal_fork,
    )


class PhaseTimer:
    """Records wall-clock durations for named phases and writes a JSON summary."""

    def __init__(self, model_path: str, tb_writer=None):
        self._model_path = model_path
        self._tb = tb_writer
        self._phases: dict[str, float] = {}
        self._t0: float | None = None
        self._current: str | None = None
        self._iter_times: list[float] = []
        self._insert_times: list[float] = []
        self._process_start = time.perf_counter()

    def start(self, phase: str) -> None:
        self._end_current()
        self._current = phase
        self._t0 = time.perf_counter()
        print(f"[timing] {phase} started", flush=True)

    def _end_current(self) -> float:
        if self._current is None or self._t0 is None:
            return 0.0
        elapsed = time.perf_counter() - self._t0
        self._phases[self._current] = self._phases.get(self._current, 0.0) + elapsed
        print(f"[timing] {self._current} done: {elapsed:.2f}s", flush=True)
        if self._tb:
            self._tb.add_scalar(f"timing/{self._current}_s", elapsed, 0)
        self._current = None
        self._t0 = None
        return elapsed

    def end(self, phase: str | None = None) -> float:
        if phase and self._current != phase:
            return 0.0
        return self._end_current()

    def record_iter(self, dt: float) -> None:
        self._iter_times.append(dt)

    def record_insert(self, dt: float) -> None:
        self._insert_times.append(dt)

    def report_window(self) -> None:
        """Print + reset rolling iter/insert timings for the recent window.

        Discriminates the live-SLAM bottleneck: a window with inserts>0
        (ingestion) vs inserts=0 (pure training burst). If iter_ms_mean during
        ingestion >> burst iter_ms_mean but insert_ms_total accounts for the
        gap, it's insertion-compute-bound; if not, it's frame-acquisition/
        pose-association blocking.
        """
        its = self._iter_times
        ins = self._insert_times
        if not its:
            return
        msg = (
            f"[timing-window] steps={len(its)} "
            f"step_ms_mean={1000 * sum(its) / len(its):.1f} "
            f"step_ms_max={1000 * max(its):.1f}"
        )
        if ins:
            msg += (
                f" | inserts={len(ins)} insert_ms_mean={1000 * sum(ins) / len(ins):.1f} "
                f"insert_ms_total={1000 * sum(ins):.1f}"
            )
        else:
            msg += " | inserts=0"
        # Per-phase breakdown (mean ms × count) for whatever _timed regions fired
        # this window, then reset so each window is independent.
        if _PHASE_OPS:
            parts = []
            for name, (s, c, mx) in sorted(_PHASE_OPS.items(), key=lambda kv: -kv[1][0]):
                if c:
                    parts.append(f"{name}={1000 * s / c:.1f}ms×{c}(max{1000 * mx:.0f})")
            if parts:
                msg += " | " + " ".join(parts)
            _PHASE_OPS.clear()
        print(msg, flush=True)
        self._iter_times = []
        self._insert_times = []

    def write_summary(self) -> None:
        self._end_current()
        total = time.perf_counter() - self._process_start
        summary = {
            "total_s": round(total, 2),
            "phases": {k: round(v, 3) for k, v in self._phases.items()},
        }
        if self._iter_times:
            its = self._iter_times
            # Canonical names: the outer-loop interval (includes ingestion, admission,
            # insertion, maintenance, logging — not just the optimizer update).
            # See issue #702.
            outer_loop_mean = round(1000 * sum(its) / len(its), 1)
            outer_loop_p95 = round(1000 * sorted(its)[int(len(its) * 0.95)], 1)
            summary["outer_loop_iteration_wall_ms_mean"] = outer_loop_mean
            summary["outer_loop_iteration_wall_ms_p95"] = outer_loop_p95
            # Deprecated compatibility aliases — kept for downstream tooling reading
            # timing_summary.json; see issue #702 and docs/DEPRECATIONS.md.
            # These measure the same outer-loop bracket as the canonical keys above,
            # not the actual optimizer update (which is narrower).
            summary["optimization_step_ms_mean"] = outer_loop_mean
            summary["optimization_step_ms_p95"] = outer_loop_p95
            summary["iter_ms_mean"] = outer_loop_mean
            summary["iter_ms_p95"] = outer_loop_p95
        if self._insert_times:
            ins = self._insert_times
            # Per-frame depth insertion timing. Measured bracket: from frame arrival
            # to insertion completion. See issue #702.
            summary["depth_insertion_wall_ms_mean"] = round(1000 * sum(ins) / len(ins), 1)
            # Deprecated alias for compatibility.
            summary["insert_ms_mean"] = summary["depth_insertion_wall_ms_mean"]
        out = os.path.join(self._model_path, "timing_summary.json")
        with open(out, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[timing] summary written to {out}", flush=True)
        if self._tb:
            for k, v in summary["phases"].items():
                self._tb.add_scalar(f"timing/{k}_s", v, 1)


# ---------------------------------------------------------------------------
# Lightweight per-phase profiler (env-gated; SPLAT_PROFILE_PHASES=1).
# Synchronizes CUDA around named regions so wall-time is attributed correctly.
# Zero overhead and no behavior change when the env var is unset.
#
# Implementation lives in utils/ingest_phase_profiler.py (issue #274, 2026-08-13)
# -- a leaf module with no torch/torchvision/gsplat import chain -- so
# scene/streaming_scene.py's StreamingScene.accumulate_observed_surface() can
# import _timed/record_count directly without a circular import onto this much
# heavier module, and so the profiler itself is unit-testable on CPU without
# pulling in train_streaming.py at all. Names below are re-exported unchanged
# so every existing `_timed(...)` call site in this file keeps working as-is.
# See that module's docstring for the _PHASE_OPS vs _PHASE_OPS_LIFETIME split.
# ---------------------------------------------------------------------------
from eligsir.utils.ingest_phase_profiler import (  # noqa: E402
    _PHASE_COUNTS,
    _PHASE_OPS,
    _PHASE_OPS_LIFETIME,
    _PHASE_SAMPLE_CAP,
    _percentiles,
    _PROFILE_PHASES,
    _timed,
    dump_phase_ops as _dump_phase_ops_impl,
    record_count,
)


from eligsir.utils.compiled_kernels import young_isotropy_ramp  # noqa: E402  (re-exported for callers/tests)

# Loss/regularizer assembly moved to eligsir/optimization/gaussian/losses.py
# (Wave 1 consolidation plan step 7) -- extracted verbatim, no math change.
# Re-imported here (not re-exported by name change) so every existing call
# site in this file keeps working unchanged.
from eligsir.optimization.gaussian.losses import (  # noqa: E402
    _snapshot_energy_grads,
    _streaming_young_isotropy_reg,
    _streaming_young_opacity_reg,
)


class StridedFrameSource:
    """Sample uniformly across the full trajectory at a fixed stride.

    Wraps a FrameSource-like object (anything with __len__ and either
    __getitem__ or __iter__) and yields every `stride`-th item, e.g.
    stride=2 -> indices [0, 2, 4, ...]. Used to pre-stride the frame
    source independently of enforcement-driven striding, so that
    --streaming_source_stride N runs match the holdout set of runs that
    rely on enforcement-driven striding (e.g. AB1/AB2 vs A1/A2).
    """

    def __init__(self, src, stride: int):
        self.lazy_frames = getattr(src, "lazy_frames", False)
        indices = list(range(0, len(src), stride))
        if hasattr(src, "__getitem__"):
            self._src = src
            self._indices = indices
            self._frames: list = []
        else:
            idx_set = set(indices)
            self._frames = [f for j, f in enumerate(src) if j in idx_set]
            self._src = None
            self._indices = list(range(len(self._frames)))

    def __len__(self):
        return len(self._frames) if self._frames else len(self._indices)

    def __iter__(self):
        if self._frames:
            yield from self._frames
        else:
            for i in self._indices:
                yield self._src[i]

    def __getitem__(self, idx):
        if self._frames:
            return self._frames[idx]
        return self._src[self._indices[idx]]

    def get_frame(self, idx):
        return self[idx]


def _load_bag_source_manifest(model_path: str | None) -> Optional[dict]:
    """Read ``<model_path>/bag_source_manifest.json`` if present (issue #865,
    acceptance items 1-2).

    Written by ``human_launch_scripts/05_run_offline_rosbag.sh`` BEFORE the
    trainer container is even launched -- the bag file is static, so its own
    per-topic message counts (read from rosbag2's ``metadata.yaml``) are
    known up front and don't depend on anything the trainer does. See
    ``utils.partition_eval.compute_bag_source_gap``'s docstring for the
    expected shape and how this cross-checks against ``ingest_counters``.

    Returns ``None`` (not an exception, not a fabricated empty dict) for
    every run that isn't a bag replay launched by that script, predates this
    mechanism, or hits a malformed/unreadable file -- ``None`` means "cannot
    verify" everywhere this flows, never "clean". Deliberately tolerant:
    a missing or corrupt manifest must never abort report generation, the
    same way every other optional report-block accessor in this module
    degrades silently rather than raising mid-report.
    """
    if not model_path:
        return None
    _path = os.path.join(model_path, "bag_source_manifest.json")
    try:
        with open(_path, "r") as _f:
            _data = json.load(_f)
    except (OSError, ValueError):
        return None
    return _data if isinstance(_data, dict) else None


def _dump_phase_ops(model_path: str | None = None) -> None:
    """Final, whole-run per-region timing + workload-counter summary.

    Thin wrapper around utils.ingest_phase_profiler.dump_phase_ops() (kept as
    a train_streaming-local name since this is the existing call site every
    prior version of this profiler used). Reads the LIFETIME store, not the
    windowed one report_window() clears every ~100 iterations -- see that
    module's docstring -- so the numbers printed/written here genuinely cover
    the full run, not just whatever fired since the last windowed report.
    """
    _dump_phase_ops_impl(model_path)


_STREAMING_PARAM_DEFAULTS = {
    "streaming_min_depth": 0.1,
    "streaming_max_depth": 8.0,
    "streaming_depth_consistency_thresh": 0.05,
    "streaming_depth_edge_threshold": 0.02,
    "streaming_depth_stride": 8,
    "streaming_depth_loss_weight": 0.05,
    "streaming_free_space_loss_weight": 0.0,
    "streaming_max_new_gaussians_per_frame": 1000,
    "streaming_keyframe_input_fps": 30.0,
    "streaming_keyframe_max_gap": 10,
    # Real default flipped to True 2026-08-15 (issue #792/#794) -- kept in
    # sync with arguments/__init__.py's factory default so this profile-
    # override-detection dict doesn't drift and silently mis-detect a CLI
    # override (see _apply_camera_profile's current_val == factory_val check).
    "streaming_keyframe_max_gap_require_novelty": True,
}


_CAMERA_PROFILES = {
    "tum_kinect_v1": {
        # TUM RGB-D uses Kinect v1. Keep noisy-depth defaults, but reject
        # depths outside the practical Kinect v1 operating range.
        "streaming_min_depth": 0.4,
        "streaming_max_depth": 3.5,
    },
    "kinect_v1": {
        # Alias for non-TUM Kinect v1 RGB-D streaming runs.
        "streaming_min_depth": 0.4,
        "streaming_max_depth": 3.5,
    },
    "orbbec_femto_bolt": {
        # Orbbec Femto Bolt: indirect ToF, <=17 mm sigma, 0.25-5.46 m range.
        "streaming_min_depth": 0.3,
        "streaming_max_depth": 5.0,
        "streaming_depth_consistency_thresh": 0.03,
        "streaming_depth_edge_threshold": 0.015,
        "streaming_depth_stride": 4,
        "streaming_depth_loss_weight": 0.1,
        "streaming_free_space_loss_weight": 0.01,
        "streaming_max_new_gaussians_per_frame": 2000,
    },
    "iphone_lidar": {
        # ScanNet++ iPhone LiDAR (Pro models, ARKit scene-depth API): direct
        # dToF, documented usable range ~0.2-5m indoors, generally cleaner/
        # lower-noise than Femto Bolt iToF but exact noise sigma isn't
        # published by Apple -- these are defensible range/stride defaults,
        # not measured noise characteristics like the Femto Bolt profile.
        "streaming_min_depth": 0.2,
        "streaming_max_depth": 5.0,
        "streaming_depth_consistency_thresh": 0.03,
        "streaming_depth_edge_threshold": 0.015,
        "streaming_depth_stride": 4,
        "streaming_depth_loss_weight": 0.1,
        "streaming_free_space_loss_weight": 0.01,
        "streaming_max_new_gaussians_per_frame": 2000,
        # ScanNet++ iPhone captures run at ~60 Hz (confirmed empirically from
        # frames.jsonl timestamps: mean dt ~= 0.01668s across all 3 scenes used
        # this session), not the 30fps the velocity-adaptive keyframe admitter
        # assumes by default (KeyframeAdmitterConfig.input_fps=30.0). The
        # admitter derives its effective min-translation threshold as
        # speed_ema * (input_fps / target_admit_rate_fps); a stale 30fps
        # assumption against real 60fps capture halves "frames_between_admits"
        # in wall-clock terms, doubling the realized velocity-gate admission
        # rate relative to the configured --streaming_keyframe_target_admit_fps.
        # Only takes effect when --streaming_keyframe_velocity_adaptive is
        # also enabled.
        "streaming_keyframe_input_fps": 60.0,
        # max_gap (default 10) force-admits every Nth frame UNCONDITIONALLY --
        # zero novelty check -- and empirically DOMINATES admission on the
        # ScanNet++ ablation runs this session: measured adm% (10-20%) sits
        # right on the 1/max_gap=10% floor across scenes of very different
        # raw frame counts (1144-6704 admitted-candidate range), while the
        # velocity-adaptive target (2fps @ 60Hz = 3.3%) is well below it --
        # i.e. max_gap is firing before the motion/overlap gates would, so
        # it is the binding constraint, not the (also real, fixed above)
        # input_fps mismatch. Doubled to 20 to hold the same wall-clock
        # inter-admit spacing as TUM's default (10 frames @ ~30fps = 0.33s)
        # against ScanNet++'s ~60fps capture, and gated behind
        # max_gap_require_novelty so a forced admit over already-mapped
        # space (a near-duplicate frame at video-rate) is deferred instead
        # of unconditionally accepted -- requires the coverage_gain_gate
        # machinery, already enabled in the ScanNet++ ablation specs this
        # session.
        "streaming_keyframe_max_gap": 20,
        "streaming_keyframe_max_gap_require_novelty": True,
    },
    "realsense_d435i": {
        # Intel RealSense D435i: active IR stereo, ±2% within 0.3-3 m,
        # higher temporal noise than Femto Bolt iToF.
        "streaming_min_depth": 0.3,
        "streaming_max_depth": 5.0,
        "streaming_depth_consistency_thresh": 0.04,
        "streaming_depth_edge_threshold": 0.02,
        "streaming_depth_stride": 4,
        "streaming_depth_loss_weight": 0.1,
        "streaming_free_space_loss_weight": 0.01,
        "streaming_max_new_gaussians_per_frame": 2000,
    },
}


def _cli_overrode_arg(key: str) -> bool:
    """Return True when the current process argv includes --<key>."""
    flag = "--" + key
    return any(arg == flag or arg.startswith(flag + "=") for arg in sys.argv[1:])


def _resolve_insertion_stop_iter(opt, args):
    """Combine the online_mode-coupled insertion-stop horizon
    (opt._streaming_insertion_stop_iter, set only by apply_online_schedule_overrides)
    with the standalone --streaming_insert_stop_iter flag (W3-66 2x2 experiment,
    2026-08-10), which works without --online_mode. Returns the more restrictive
    (earlier) of the two when both are set, or whichever one is set, or None."""
    online_stop = getattr(opt, "_streaming_insertion_stop_iter", None)
    manual_stop = int(getattr(args, "streaming_insert_stop_iter", 0) or 0)
    manual_stop = manual_stop if manual_stop > 0 else None
    if online_stop is None:
        return manual_stop
    if manual_stop is None:
        return online_stop
    return min(online_stop, manual_stop)


def _should_flush_insertion_batch(
    iteration: int,
    last_flush_iteration: int,
    batch_steps: int,
    frames_since_flush: int,
    batch_frames: int,
    pending_pts: int,
    batch_max_pts: int,
    disable_point_cap: bool,
) -> bool:
    """Decide whether to flush the pending insertion batch this iteration.

    When ``batch_steps > 0``, step-based cadence takes priority and the
    frame-count trigger is ignored entirely -- fires on step-count threshold
    OR point-count cap (unless disabled). Without this, the frame-count
    trigger (which fires on literally every admitted frame at the default
    ``batch_frames=1``) would make ``batch_steps`` silently inert unless the
    caller also raised ``batch_frames`` far above what it wants -- exactly
    the "inert by construction" failure class this project has hit before
    (W3-38's silent min_steps_per_frame clamp, W3-45's inert Phase C).

    When ``batch_steps == 0`` (the default), unchanged legacy behavior:
    fires on frame-count threshold OR point-count cap (unless disabled).

    Pure function, no side effects -- callers own their own state updates.
    """
    if batch_steps > 0:
        if (iteration - last_flush_iteration) >= batch_steps:
            return True
        if not disable_point_cap and batch_max_pts > 0 and pending_pts >= batch_max_pts:
            return True
        return False
    if frames_since_flush >= batch_frames:
        return True
    if not disable_point_cap and batch_max_pts > 0 and pending_pts >= batch_max_pts:
        return True
    return False


# Mirror of ``StreamingParams.streaming_replay_buffer``'s declared default
# (`arguments/__init__.py`). Kept as a plain constant so the reachability check
# below stays a pure, import-cheap function; `tests/test_replay_buffer_
# reachability.py` asserts it still matches the real declared default, so
# drift is caught by the test suite rather than silently mis-warning.
STREAMING_REPLAY_BUFFER_DEFAULT = 512


def check_replay_buffer_reachability(args) -> "list[str]":
    """Detect the inert ``--streaming_replay_buffer`` configuration (issue #360).

    ``--streaming_replay_buffer`` is documented as "ring buffer size for older
    frames", but ``StreamingScene._sample_legacy`` / ``_sample_stratified``
    both take the global-replay draw from ``_global_reservoir`` FIRST and only
    fall through to ``_replay_buffer`` when the reservoir is empty::

        if reservoir_stride > 0 and self._global_reservoir:
            return self._weighted_choice(self._global_reservoir, kw)
        if self._replay_buffer:
            ...

    The reservoir is appended to on every ``stride``-th ingested train frame
    and is NEVER evicted, so with the default
    ``--streaming_global_reservoir_stride 1`` it becomes non-empty on the very
    first admitted train frame and stays non-empty for the rest of the run.
    Consequence: past the bootstrap window the ring buffer is never sampled,
    and its *size* can never bind — the ring only ever holds
    ``streaming_initial_frames`` (~5) cameras at the point where the reservoir
    takes over, far below any configured size. Eviction from the ring also
    frees nothing (``train_cameras`` holds every camera regardless), so the
    parameter has no memory effect either.

    The real lever for global-replay pool composition is
    ``--streaming_global_reservoir_stride`` (set it to ``0`` to disable the
    reservoir and hand the global-replay draw back to the ring buffer).

    Pure function: takes the merged ``run_args`` namespace, returns a list of
    human-readable warning strings (empty when nothing is wrong). Performs no
    I/O, mutates nothing, and never affects sampling. The caller prints them.
    """
    warnings: "list[str]" = []
    if not bool(getattr(args, "streaming_replay", False)):
        return warnings

    stride = int(args.streaming_global_reservoir_stride)
    if stride <= 0:
        # Reservoir disabled -> the ring buffer IS the global replay pool and
        # --streaming_replay_buffer is a live lever. Nothing to warn about.
        return warnings

    default_size = STREAMING_REPLAY_BUFFER_DEFAULT
    size = int(getattr(args, "streaming_replay_buffer", default_size))
    if size == default_size:
        return warnings

    warnings.append(
        f"[streaming] WARNING: --streaming_replay_buffer={size} (non-default, "
        f"default {default_size}) has NO EFFECT in this configuration. "
        f"--streaming_global_reservoir_stride={stride} (>0) makes the "
        f"never-evicted global reservoir take every global-replay draw once "
        f"the first train frame is ingested, so the replay ring buffer is "
        f"only sampled during the bootstrap window and its size can never "
        f"bind. Pass --streaming_global_reservoir_stride 0 to make "
        f"--streaming_replay_buffer live, or drop the flag. See issue #360."
    )
    return warnings


def validate_streaming_optimization_batch_size(args) -> None:
    """Validate ``--streaming_optimization_batch_size`` (PR3, 2026-08-12).

    Takes the RAW, merged ``run_args`` namespace (the object
    ``streaming_training()`` assigns to ``args``), NOT a per-``ParamGroup``
    ``.extract()`` result: ``optimizer_type``/``gsplat_sparse_grad`` are
    declared on ``OptimizationParams`` while
    ``streaming_optimization_batch_size``/``streaming_spatial_bubble_enabled``
    are declared on ``StreamingParams``, and ``ParamGroup.extract()`` only
    copies attributes declared on that one group -- so no single extracted
    object carries all four.

    Raises ``ValueError`` (returns ``None`` otherwise) when:
      * the batch size is not one of the explicitly bounded candidates
        {1, 2, 4, 6, 8, 16, 32, 64, 128, 256}; or
      * the batch size is > 1 while any of the following per-iteration,
        single-visibility-set mechanisms is active: ``selective_adam``,
        sparse gradients, the spatial active bubble, depth lock, or the
        opt-in streaming optimization engine. Each of these snapshots or
        masks per-parameter state around ONE ``optimizer.step()`` keyed to
        ONE iteration's visibility set; their correctness under multi-view
        gradient accumulation is unverified, so they are gated out with a
        clear error rather than silently producing wrong results.
    """
    try:
        batch_size = validate_supervision_batch_size(
            getattr(args, "streaming_optimization_batch_size", 1)
        )
    except ValueError as error:
        raise ValueError(
            "--streaming_optimization_batch_size " + str(error).removeprefix("supervision batch size ")
        ) from error
    if batch_size == 1:
        return

    # Issue #106: the ENGINE path (--streaming_optimization_engine) resolves
    # the single-visibility-set problem properly -- GaussianOptimizationEngine.
    # run_batched_step() unions every batched view's visibility filter and
    # steps against that union, so selective_adam / sparse gradients see all
    # rows any view in the batch made visible. The inline path has no such
    # union and still has to refuse those two combinations.
    _engine = bool(args.streaming_optimization_engine)

    conflicts = []
    if (
        not _engine
        and str(getattr(args, "optimizer_type", "selective_adam")) == "selective_adam"
    ):
        conflicts.append(
            "optimizer_type=selective_adam (prepare_selective_adam_step assumes a "
            "single per-iteration visibility set; supported on the engine path "
            "via --streaming_optimization_engine, which unions the batch's "
            "visibility filters)"
        )
    if not _engine and bool(getattr(args, "gsplat_sparse_grad", True)):
        conflicts.append(
            "gsplat_sparse_grad=True (sparse gradients assume a single "
            "per-iteration visibility set; supported on the engine path via "
            "--streaming_optimization_engine)"
        )
    if bool(getattr(args, "streaming_spatial_bubble_enabled", False)):
        conflicts.append(
            "streaming_spatial_bubble_enabled=True (spatial_bubble snapshots and "
            "restores per-parameter state around one optimizer.step())"
        )
    if bool(getattr(args, "streaming_depth_lock_centers", False)):
        conflicts.append(
            "streaming_depth_lock_centers=True (depth-lock snapshots and restores "
            "per-parameter centers around one optimizer.step())"
        )
    if conflicts:
        raise ValueError(
            "--streaming_optimization_batch_size > 1 (multi-view gradient "
            "accumulation) is not supported together with: "
            + "; ".join(conflicts)
            + ". Pass --parallelism_profile off (which selects dense Adam and "
            "non-sparse gradients since issue #228 was fixed) or the equivalent "
            "explicit pair --optimizer_type adam --no-gsplat_sparse_grad (note "
            "the hyphen in --no-gsplat_sparse_grad; --gsplat_sparse_grad False "
            "is an argparse error), and disable the listed features -- or leave "
            "the batch size at 1. NOTE: --no-gsplat_sparse_grad on its own is "
            "still not enough, because optimizer_type=selective_adam requires "
            "sparse gradients (that combination now raises a loud error rather "
            "than silently re-enabling them). Runs launched BEFORE the #228 fix "
            "with --parallelism_profile off alone really did train sparse -- "
            "check the run's own [parallelism] log line / cfg_args (see "
            "docs/architecture/2026-08-12-issue228-parallelism-profile-off-audit.md "
            "and docs/architecture/2026-08-13-issue228-fix-impact.md)."
        )


def _apply_camera_profile(args) -> None:
    """Apply known camera-profile defaults without clobbering CLI overrides."""
    profile = getattr(args, "streaming_camera_profile", "default")
    if profile == "default":
        return

    overrides = _CAMERA_PROFILES.get(profile)
    if overrides is None:
        print(f"[streaming] Warning: unknown camera profile '{profile}', ignoring.", flush=True)
        return

    applied = []
    for key, profile_val in overrides.items():
        current_val = getattr(args, key, None)
        factory_val = _STREAMING_PARAM_DEFAULTS.get(key)
        if current_val == factory_val and not _cli_overrode_arg(key):
            setattr(args, key, profile_val)
            applied.append(f"  {key}: {factory_val} -> {profile_val}")

    if applied:
        print(f"[streaming] Camera profile '{profile}':\n" + "\n".join(applied), flush=True)
    else:
        print(f"[streaming] Camera profile '{profile}': no default parameters to override.", flush=True)


# ---------------------------------------------------------------------------
# Point-cloud insertion helpers (Phase 2)
# ---------------------------------------------------------------------------

def _voxel_downsample_indices(points: np.ndarray, voxel_size: float):
    if points.shape[0] == 0:
        return points, np.array([], dtype=np.int64)
    mins = points.min(axis=0)
    cell = np.floor((points - mins) / max(voxel_size, 1e-6)).astype(np.int64)
    maxc = cell.max(axis=0) + 1
    stride = np.array([maxc[1] * maxc[2], maxc[2], 1], dtype=np.int64)
    keys = cell @ stride
    _, first = np.unique(keys, return_index=True)
    return points[first], first


def _voxel_downsample(points: np.ndarray, colors: np.ndarray, voxel_size: float):
    """Keep one representative point per voxel cell (O(N log N))."""
    p, i = _voxel_downsample_indices(points, voxel_size)
    return p, colors[i]


def _save_depth_comparison(
    out_dir: str,
    frame_idx: int,
    rgb_tensor: torch.Tensor,
    rendered_depth: torch.Tensor,
    sensor_depth: torch.Tensor,
) -> None:
    """Save a 3-panel depth comparison PNG: RGB | rendered depth | sensor depth."""
    import numpy as np
    try:
        from PIL import Image as _PILImage
    except ImportError:
        return
    os.makedirs(out_dir, exist_ok=True)
    # Convert RGB [3, H, W] float to uint8
    rgb_np = (rgb_tensor.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    H, W = rgb_np.shape[:2]

    def _depth_to_color(d_tensor: torch.Tensor, vmax: float) -> np.ndarray:
        d = d_tensor.detach().cpu().numpy().squeeze()
        if d.shape != (H, W):
            from PIL import Image as _I
            d = np.array(_I.fromarray(d).resize((W, H), _I.Resampling.NEAREST))
        valid = d > 0
        norm = np.zeros_like(d, dtype=np.float32)
        if valid.any() and vmax > 0:
            norm[valid] = np.clip(d[valid] / vmax, 0, 1)
        # Jet-like colormap via lookup: blue→cyan→green→yellow→red
        r = np.clip(1.5 - abs(norm * 4 - 3), 0, 1)
        g = np.clip(1.5 - abs(norm * 4 - 2), 0, 1)
        b = np.clip(1.5 - abs(norm * 4 - 1), 0, 1)
        out = (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)
        out[~valid] = 0  # black for invalid/zero depth
        return out

    vmax = float(sensor_depth[sensor_depth > 0].max()) if (sensor_depth > 0).any() else 5.0
    rend_col = _depth_to_color(rendered_depth, vmax)
    sens_col = _depth_to_color(sensor_depth, vmax)
    diff = np.abs(
        rendered_depth.detach().cpu().numpy().squeeze().astype(np.float32) -
        sensor_depth.detach().cpu().numpy().squeeze().astype(np.float32)
    )
    diff_col = _depth_to_color(torch.from_numpy(diff.clip(0, vmax * 0.5)[None]), vmax * 0.5)

    panel = np.concatenate([rgb_np, rend_col, sens_col, diff_col], axis=1)
    _PILImage.fromarray(panel).save(os.path.join(out_dir, f"frame_{frame_idx:06d}.png"))


# _get_sensor_depth / _get_surface_prior moved verbatim to
# eligsir/mapping/integration/depth_helpers.py (PR #39 review fix, P0):
# eligsir/optimization/gaussian/objective.py needs them but must not
# depend backwards on this root entry-point module. Re-imported here (not
# re-exported by name change) so every existing call site in this file
# keeps working unchanged.
from eligsir.mapping.integration.depth_helpers import (  # noqa: E402
    _get_sensor_depth,
    _get_surface_prior,
    set_sensor_depth_cache_bound,
)

# Wave 3, PR W3-08 (plan section 13): tri-state ray semantics. This file's
# inline loss-assembly block (below) is the legacy, still-default-active
# path (--streaming_optimization_engine defaults False -- see
# eligsir/optimization/gaussian/engine.py's docstring), so the same
# ray-semantics wiring added to eligsir/optimization/gaussian/objective.py
# (the NEW engine's loss path) must ALSO be applied here for the feature to
# be reachable by any real (engine-off, i.e. today's default) run. Reusing
# the same helper functions rather than duplicating the classification
# adapter logic a second time.
from eligsir.optimization.gaussian.objective import (  # noqa: E402
    _classify_view_rays,
    _get_ray_consensus_index,
)
from eligsir.optimization.gaussian.depth_loss_weighting import (  # noqa: E402
    DEPTH_LOSS_QUANTITIES,
    combine_depth_loss_weights,
    weighted_depth_error_mean,
)
from eligsir.mapping.geometry.appearance_only import run_appearance_only_update  # noqa: E402
from eligsir.mapping.geometry.material_risk import geometry_reliability_map, material_risk_mask  # noqa: E402
from eligsir.mapping.tier_bind import (  # noqa: E402
    ExtraViewBatchCollector as _ExtraViewBatchCollector,
    MultiresBindAccounting as _MultiresBindAccounting,
    bind_resolution_tier as _bind_resolution_tier_impl,
    describe_startup_resolution_hazard as _describe_startup_resolution_hazard,
    extra_view_admitted as _extra_view_admitted,
)

# One-time warning latch for the (inline-loop copy of the) legacy-carve /
# ray-semantics combined-flags case -- see the "Free-space / floater loss"
# block below and its objective.py twin (_RS_WARNED_LEGACY_CARVE_COMBO).
_RS_WARNED_LEGACY_CARVE_COMBO_INLINE = False

# One-time startup/warning latches for the (inline-loop copy of the) depth
# supervision quantity selection -- see the "Depth supervision loss" block
# below and its objective.py twin (_DEPTH_LOSS_STARTUP_LOGGED /
# _DEPTH_LOSS_ED_ALPHA_GATE_WARNED).
_DEPTH_LOSS_STARTUP_LOGGED_INLINE = False
_DEPTH_LOSS_ED_ALPHA_GATE_WARNED_INLINE = False


# issue #333: tier-bind semantics + bounded-cardinality accounting live in
# eligsir/mapping/tier_bind.py (importable without train_streaming's heavy
# dependency graph, so its tests run CPU-only). This module keeps the two
# thin wrappers the training loop and the existing tests call.
_MULTIRES_BIND_ACCOUNTING = _MultiresBindAccounting()
_MULTIRES_MISMATCH_WARNED: dict = {}


def bind_resolution_tier_ex(cam, record, tier: str, db=None):
    """Bind `cam` to `tier` and return the full `TierBindResult`.

    Prefer this over the bool wrapper below in the training loop: the result's
    `tier_after` / `usable` fields are what let a caller avoid training (or
    batching) a camera at a tier nobody requested -- see issue #333 and
    `eligsir/mapping/tier_bind.py`'s module docstring.

    `db` (issue #333 phase 3, PR #780 activation): the `FrameObservationDB`
    instance owning `record`, if the caller has one in scope. When provided
    and this call resolves as a resident cache hit, it is forwarded to
    `_bind_resolution_tier_impl`, which calls `db.touch_resolution_cache(...)`
    to refresh LRU recency. Defaults to `None` (pre-#780 behavior, and what
    the existing 3-arg test call sites keep exercising) so PR #780's
    LRU-refresh mechanism stays inert unless a caller opts in by passing it --
    every production training-loop call site below does.
    """
    return _bind_resolution_tier_impl(
        cam, record, tier, accounting=_MULTIRES_BIND_ACCOUNTING, db=db
    )


def bind_resolution_tier(cam, record, tier: str) -> bool:
    """Back-compatible bool wrapper around `bind_resolution_tier_ex`.

    True means the camera really is at `tier` afterwards (a resident cache
    hit, or -- for a HIGH request -- the camera's own native image, which IS
    the high tier). False means the request was NOT resolved; the camera was
    left untouched and `bind_resolution_tier_ex(...).tier_after` states which
    tier it is actually at.
    """
    return bool(bind_resolution_tier_ex(cam, record, tier).ok)


# Moved to utils/streaming_state_sync.py (issue #1364 audit, 2026-08-31): this
# pure function had a genuine canonical-side caller (utils/strategies/mcmc/
# arbiter_owned.py, part of the live gsplat_mcmc strategy) importing it
# directly from this module, which pulled this entire legacy file (and
# everything it imports) into check_canonical_mapping_hard_cut.py's canonical
# dependency cone -- a real regression in the #1224 routing boundary, not just
# a test artifact. Re-exported here, unchanged, so both existing call sites in
# this file keep working.
from eligsir.utils.streaming_state_sync import _sync_streaming_state_lengths  # noqa: E402,F401


# --------------------------------------------------------------------------
# Multires resolution-cache health accounting (issue #464)
#
# ``record_frame_observation``'s ``build_resolution_cache`` call is wrapped in a
# deliberately BROAD ``except Exception`` so that one bad frame can never kill a
# live SLAM run -- CLAUDE.md's top-priority live-camera hard constraint. That
# property is preserved verbatim here; #464 is not about making the failure
# fatal, it is about making it VISIBLE.
#
# Before this fix the warning was ONE-SHOT (``_warned_exc``), so a run in which
# the cache build failed on EVERY frame printed exactly one line at the first
# occurrence and recorded nothing anywhere else -- i.e. 100%
# ``bind_resolution_tier`` failure for the whole run with essentially no
# diagnostic trail. This mirrors the #437/#445 treatment of swallowed per-camera
# render failures: keep swallowing, but COUNT the occurrences and surface the
# totals in the run's own artifacts (``report.json["multires_cache"]`` -- TOP-LEVEL,
# deliberately NOT under ``geometry``, so a swallowed geometry failure cannot erase
# the block whose entire purpose is to report a swallowed failure).
#
# Process-global mutable state, matching the existing ``_warned_*`` function-
# attribute convention in this module. It is reset at ``streaming_training()``
# entry (and must be reset by any test asserting on counts) so counts never leak
# across runs/tests in one process.
_MULTIRES_CACHE_STATS: dict = {}


def reset_multires_cache_stats() -> None:
    """Zero the issue-#464 resolution-cache accounting.

    Called once at ``streaming_training()`` entry. Tests that assert on counts
    MUST call this first -- these are module-global counters, so without a reset
    a count assertion silently picks up occurrences from earlier tests in the
    same process (test order is not guaranteed; ``-p no:randomly`` only masks it).
    """
    _MULTIRES_CACHE_STATS.clear()
    _MULTIRES_CACHE_STATS.update({
        "frames_attempted": 0,      # is_train frames with a frame object
        "frames_cached": 0,         # ... that ended up with a non-empty cache
        "frames_failed": 0,         # ... that raised (swallowed) inside the try
        "frames_empty_no_depth": 0, # ... that legitimately produced no cache
        "error_types": {},
        "example_error": None,
        "first_failure_frame_id": None,
        "last_failure_frame_id": None,
        "device": None,
    })
    # The old one-shot latches (`record_frame_observation._warned_exc` /
    # `._warned_no_depth`) are gone -- the print cadence is now derived purely
    # from these counters via _should_print_multires_warning, so clearing the
    # counters is by itself a complete reset of both the accounting AND the
    # logging state. Any stale attribute left on the function object by an
    # older process is simply never read.
    _MULTIRES_CACHE_STATS["_n_no_depth"] = 0


reset_multires_cache_stats()


def _should_print_multires_warning(count: int) -> bool:
    """Escalating (not one-shot, not every-frame) print cadence.

    Prints at 1, 10, 100, 1000, ... occurrences. One-shot was the #464 defect;
    every-frame would flood a live run's log at frame rate. The exact total is
    always recoverable from ``multires_cache_report_block()`` regardless of what
    was printed.
    """
    if count <= 0:
        return False
    while count % 10 == 0:
        count //= 10
    return count == 1


def multires_cache_report_block() -> dict:
    """ALWAYS-PRESENT top-level ``report.json["multires_cache"]`` block (issue #464).

    NOTE the location: TOP-LEVEL, not under ``geometry``. A block whose whole
    purpose is to report a *swallowed* failure must not itself be erasable by a
    *different* swallowed failure, so it is written outside the ``geometry``
    try/except (same placement as #437's ``n_test_metrics_failed``).

    Consumed by ``write_post_training_report``'s ``multires_cache`` parameter.
    Invariant, mirroring #437's ``n_*_metrics`` accounting::

        frames_cached + frames_failed + frames_empty_no_depth == frames_attempted

    ``failures`` is present only when something actually failed. An ABSENT
    ``multires_cache`` key in a report.json means the report predates this fix,
    NOT that the cache was healthy.
    """
    s = _MULTIRES_CACHE_STATS
    attempted = int(s.get("frames_attempted", 0))
    cached = int(s.get("frames_cached", 0))
    failed = int(s.get("frames_failed", 0))
    empty = int(s.get("frames_empty_no_depth", 0))
    block = {
        "frames_attempted": attempted,
        "frames_cached": cached,
        "frames_failed": failed,
        "frames_empty_no_depth": empty,
        "device": s.get("device"),
        # True iff EVERY attempted train frame failed -- the total, near-silent
        # failure mode #464 was filed about (100% bind_resolution_tier failure).
        "all_failed": bool(attempted > 0 and failed == attempted),
        "cache_hit_fraction": (cached / attempted) if attempted else None,
    }
    if failed:
        block["failures"] = {
            "n_failed": failed,
            "error_types": dict(s.get("error_types") or {}),
            "example_error": s.get("example_error"),
            "first_failure_frame_id": s.get("first_failure_frame_id"),
            "last_failure_frame_id": s.get("last_failure_frame_id"),
        }
    return block


def _resolution_cache_device(preferred=None) -> str:
    """Resolve the device ``build_resolution_cache`` should upload tiers to.

    Issue #464: this was hardcoded to ``"cuda"``, which raises
    ``RuntimeError: No HIP GPUs are available`` on a GPU-masked/CPU host and was
    then swallowed by the broad ``except`` below -- leaving the cache empty for
    the rest of the run. An explicit ``preferred`` (threaded from the caller)
    always wins; otherwise derive from what torch can actually see. On any real
    GPU training run this resolves to ``"cuda"`` exactly as before, so the
    production path is unchanged.
    """
    if preferred:
        return str(preferred)
    try:
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"



def build_frame_resolution_cache(
    frame,
    frame_id: int,
    *,
    low_res_divisor: int = 4,
    depthless_rgb_cache: bool = True,
    device=None,
    depth_downsample: str = "nearest",
) -> "tuple[dict, str | None]":
    """Build one frame's dual-tier multires ``resolution_cache``.

    Extracted verbatim from ``record_frame_observation`` (issue #481) so a
    SECOND producer can exist without re-adding a ``FrameRecord``:
    ``StreamingScene.promote_from_cache`` needs the cache for a frame that
    was first recorded as a rejected candidate (``is_train=False``, hence
    ``resolution_cache_absent_reason == "not_train_frame"``), and
    ``FrameObservationDB.add_frame`` is NOT re-callable for an existing
    frame_id -- it re-appends the frame's visible-uid row to the incremental
    CSR buffer (double-counting the observation) and would overwrite the real
    visibility recorded at ingest with whatever the caller passes now.
    ``FrameObservationDB.attach_resolution_cache`` + this function are that
    second producer; ``record_frame_observation`` keeps calling it for the
    ordinary ingest path, so there is exactly one implementation.

    Returns ``(resolution_cache, absent_reason)``; ``absent_reason`` is None
    whenever the cache was really built (issue #333's semantics, unchanged).
    """
    resolution_cache: dict = {}
    _cache_absent_reason: "str | None" = None
    _MULTIRES_CACHE_STATS["frames_attempted"] = _MULTIRES_CACHE_STATS.get("frames_attempted", 0) + 1
    _cache_device = _resolution_cache_device(device)
    _MULTIRES_CACHE_STATS["device"] = _cache_device
    try:
        from eligsir.mapping.resolution_cache import build_resolution_cache
        from eligsir.utils.streaming_frames import load_frame_rgb
        from eligsir.utils.rgbd_frames import CameraIntrinsics
        _rgb_img = load_frame_rgb(frame)
        # Reconcile the decoded image's actual pixel size against the
        # frame's DECLARED width/height (what the intrinsics below are
        # built from) -- these can legitimately differ: e.g.
        # RealityScanFrameSource's --streaming_auto_downscale_max_pixels
        # gate rewrites frame.width/height/fx/fy/cx/cy to a divided-down
        # size while frame.rgb_path still points at the native-resolution
        # source photo (eligsir/io/sources/datasets/realityscan.py,
        # divisor block). Mirrors the same reconciliation
        # StreamingScene._frame_to_camera already does for the ordinary
        # (non-cache) camera-image path (`image.resize(_target, ...)`) --
        # without it, build_resolution_cache would derive each tier's
        # pixel dimensions from the wrong (native) base while its
        # CacheHandle.intrinsics claims the divided-down base, and
        # bind_resolution_tier would then set cam.image_height/width
        # from one and cam.fx/fy/cx/cy from the other -- disagreeing.
        _rgb_target = (int(frame.width), int(frame.height))
        if _rgb_img.size != _rgb_target:
            from PIL import Image as _PILImage
            _rgb_img = _rgb_img.resize(_rgb_target, _PILImage.BILINEAR)
        rgb_np = np.asarray(_rgb_img)
        depth_m = _load_depth_meters(frame)
        if depth_m is None:
            # _load_depth_meters returns None for two structurally
            # different reasons -- only the first is safe to treat as
            # "this source has no depth at all":
            #   (a) STRUCTURAL: frame._depth_np/depth_path/_depth_bytes
            #       are all None -- this frame's SOURCE never had depth
            #       (e.g. RealityScan Mobile). Every frame from this
            #       source is depth-less; safe to build an rgb-only tier.
            #   (b) TRANSIENT FAILURE: a depth-having source (TUM/
            #       ScanNet/Orbbec) where this one frame's depth file is
            #       missing/corrupt/failed to decode (the bare
            #       `except Exception: return None` in
            #       _load_depth_meters). Building an rgb-only cache here
            #       would make bind_resolution_tier succeed and rebind
            #       cam.image_height/width/fx/fy/cx/cy to a tier, while
            #       _get_sensor_depth's own independent lazy-load path
            #       still hands back NATIVE-resolution depth for this
            #       same camera on a later access -- a resolution/
            #       intrinsics mismatch that never existed pre-fix (the
            #       old behavior for case (b) was simply an empty cache,
            #       which is what must be preserved here).
            _structurally_depthless = (
                frame._depth_np is None
                and frame.depth_path is None
                and frame._depth_bytes is None
            )
            if depthless_rgb_cache and _structurally_depthless:
                # 2026-08-12 fix: depth-less source (e.g. RealityScan
                # Mobile) -- build rgb-only tiers so bind_resolution_tier
                # can still rebind image_height/width/fx/fy/cx/cy per the
                # throughput-driven tier decision. depth stays absent
                # (CacheHandle.depth=None); bind_resolution_tier leaves
                # cam._sensor_depth_cache untouched for these tiers, so
                # depth-supervision loss code's existing "no depth for
                # this source" handling is unaffected.
                intrinsics = CameraIntrinsics(
                    width=int(frame.width), height=int(frame.height),
                    fx=float(frame.fx), fy=float(frame.fy),
                    cx=float(frame.cx), cy=float(frame.cy),
                    depth_scale=1.0,
                )
                resolution_cache = build_resolution_cache(
                    rgb_np, None, intrinsics, low_res_divisor=low_res_divisor,
                    device=_cache_device, depth_downsample=depth_downsample,
                )
            if not resolution_cache:
                _cache_absent_reason = (
                    "depth_load_failed_transient" if not _structurally_depthless
                    else "depthless_rgb_cache_disabled"
                )
            # Issue #464: counted, escalating cadence (was one-shot).
            _n_no_depth = _MULTIRES_CACHE_STATS.get("_n_no_depth", 0) + 1
            _MULTIRES_CACHE_STATS["_n_no_depth"] = _n_no_depth
            if _should_print_multires_warning(_n_no_depth):
                _reason = (
                    "rgb-only (depthless_rgb_cache=True, structurally depth-less source)"
                    if (depthless_rgb_cache and _structurally_depthless)
                    else "stays empty (transient per-frame depth-load failure on a depth-having source)"
                    if not _structurally_depthless
                    else "stays empty (depthless_rgb_cache=False)"
                )
                print(f"[multires-debug] frame_id={frame_id} depth_m is None "
                      f"(depth_path={frame.depth_path!r} _depth_np={frame._depth_np is not None} "
                      f"_depth_bytes={frame._depth_bytes is not None}) -- resolution_cache {_reason} "
                      f"[occurrence #{_n_no_depth}]",
                      flush=True)
        else:
            intrinsics = CameraIntrinsics(
                width=int(frame.width), height=int(frame.height),
                fx=float(frame.fx), fy=float(frame.fy),
                cx=float(frame.cx), cy=float(frame.cy),
                depth_scale=1.0,  # depth_m is already metres -- no further scaling
            )
            resolution_cache = build_resolution_cache(
                rgb_np, depth_m, intrinsics, low_res_divisor=low_res_divisor,
                device=_cache_device, depth_downsample=depth_downsample,
            )
        if resolution_cache:
            _MULTIRES_CACHE_STATS["frames_cached"] = _MULTIRES_CACHE_STATS.get("frames_cached", 0) + 1
        else:
            _MULTIRES_CACHE_STATS["frames_empty_no_depth"] = (
                _MULTIRES_CACHE_STATS.get("frames_empty_no_depth", 0) + 1
            )
    except Exception as e:
        # DELIBERATELY still a broad `except` that keeps the run alive
        # (CLAUDE.md live-camera hard constraint) -- issue #464 narrows the
        # VISIBILITY gap, not the exception. Every occurrence is counted and
        # surfaced in report.json["geometry"]["multires_cache"]; the print is
        # escalating (1, 10, 100, ...) rather than one-shot, so a run that
        # fails on every frame can no longer look like a single blip.
        _n_exc = _MULTIRES_CACHE_STATS.get("frames_failed", 0) + 1
        _MULTIRES_CACHE_STATS["frames_failed"] = _n_exc
        _etypes = _MULTIRES_CACHE_STATS.setdefault("error_types", {})
        _etypes[type(e).__name__] = _etypes.get(type(e).__name__, 0) + 1
        if _MULTIRES_CACHE_STATS.get("example_error") is None:
            _MULTIRES_CACHE_STATS["example_error"] = f"{type(e).__name__}: {e}"
            _MULTIRES_CACHE_STATS["first_failure_frame_id"] = frame_id
        _MULTIRES_CACHE_STATS["last_failure_frame_id"] = frame_id
        if _should_print_multires_warning(_n_exc):
            import traceback
            print(f"[multires-debug] build_frame_resolution_cache EXCEPTION frame_id={frame_id} "
                  f"[occurrence #{_n_exc}, resolution_cache left EMPTY for this frame]: "
                  f"{type(e).__name__}: {e}\n{traceback.format_exc()}", flush=True)
        resolution_cache = {}
        _cache_absent_reason = "build_exception"
    return resolution_cache, (None if resolution_cache else _cache_absent_reason)

def record_frame_observation(
    db,
    cam,
    visible_uids,
    frame_id: int,
    submap_id: int,
    is_train: bool,
    is_holdout: bool,
    frame=None,
    low_res_divisor: int = 4,
    depthless_rgb_cache: bool = True,
    device=None,
    depth_downsample: str = "nearest",
) -> None:
    """Record one candidate frame's fast-projected visibility into the
    observation DB (Task 2-4 of the keyframe-database design). Called for
    EVERY ingested frame, not just admitted ones -- the DB's whole point is
    comparing candidates against full history, so rejected frames' content
    still needs to be indexed for future revisit matching.

    ``visible_uids`` is computed by the caller (via
    ``eligsir.mapping.visibility.project_visible_gaussian_uids``) once
    per frame and passed in here rather than recomputed, since the caller
    also stashes it on the camera for admission scoring (Task 5) -- running
    the frustum projection twice per frame would be wasted work.

    ``resolution_cache`` (multires-cache plan, Task 4) is only populated for
    ``is_train`` frames -- those are the only ones ever sampled as
    ``viewpoint_cam`` by the training loop's per-iteration camera selection,
    so building+uploading the dual-tier cache for rejected/holdout candidates
    would be pure wasted GPU work with no consumer. This is a deliberate,
    documented narrowing of the literal "every ingested frame" phrasing.

    ``depthless_rgb_cache`` (2026-08-12 fix, default True ==
    ``--streaming_multires_depthless_rgb_cache``): when the frame has no
    depth (``_load_depth_meters`` returns None -- e.g. RealityScan Mobile
    scenes), still build rgb-only tiers instead of leaving the cache
    permanently empty. The low/high tier DECISION (ResolutionController)
    is throughput-driven, not depth-driven, so this makes the resolution
    controller actually functional for depth-less sources instead of
    100%-failing bind_resolution_tier for the whole run. Set False to
    restore the old always-empty-cache behavior for a depth-less source.

    ``device`` (issue #464, default None == auto): where the built tiers are
    uploaded. This used to be hardcoded ``"cuda"`` at both
    ``build_resolution_cache`` call sites, which raised on a GPU-masked/CPU host
    and was swallowed by the broad ``except`` below, silently leaving the cache
    empty for that frame AND every later one. ``None`` derives the device from
    what torch can actually see (see ``_resolution_cache_device``) -- ``"cuda"``
    on any real GPU run, so production behavior is unchanged.
    """
    from eligsir.mapping.observation_db import BufferStage, FrameRecord

    if is_holdout:
        stage = BufferStage.HOLDOUT_TEST
    elif is_train:
        stage = BufferStage.MAPPING_KEYFRAME
    else:
        stage = BufferStage.EVICTED_REDUNDANT

    resolution_cache: dict = {}
    # issue #333: WHY a cache is absent is knowable here and nowhere else.
    # Several absences are by design (a non-train frame is never cached at all;
    # a transient per-frame depth-load failure deliberately leaves it empty --
    # see build_frame_resolution_cache's branches), and lumping them in with
    # "never built" recreates exactly the ambiguity #333 removes from the
    # eviction case. None == the cache was really built for this frame.
    _cache_absent_reason: "str | None" = None if (is_train and frame is not None) else "not_train_frame"
    if is_train and frame is not None:
        resolution_cache, _cache_absent_reason = build_frame_resolution_cache(
            frame, frame_id, low_res_divisor=low_res_divisor,
            depthless_rgb_cache=depthless_rgb_cache,
            depth_downsample=depth_downsample, device=device,
        )

    if visible_uids is None:
        # Bootstrap frames (via initialize_from_frames) call this before any
        # Gaussians exist, so there is nothing to project visibility against
        # yet -- FrameObservationDB.add_frame requires a real tensor to call
        # .numel() on.
        visible_uids = torch.zeros(0, dtype=torch.int64)

    db.add_frame(FrameRecord(
        frame_id=frame_id,
        pose_c2w=np.asarray(getattr(cam, "c2w", np.eye(4)), dtype=np.float32),
        timestamp=float(frame_id),
        submap_id=submap_id,
        buffer_stage=stage,
        visible_gaussian_uids=visible_uids,
        resolution_cache=resolution_cache,
        resolution_cache_absent_reason=(None if resolution_cache else _cache_absent_reason),
    ))


from eligsir.mapping.integration.geometry_helpers import (  # noqa: E402
    _gaussian_min_axis_normals,
    _gpu_render_occupancy_masks,
    _kornia_normal_map,
    _load_depth_meters,
    _quat_wxyz_to_rotmat,
    _quaternion_from_normal,
    _resize_nearest_np,
    _sample_stable_depth_mask,
    _sample_stable_depth_mask_gpu,
    _sample_temporal_depth_mask,
    _sample_temporal_depth_mask_gpu,
    knn_smoothness_reg,
)
from eligsir.mapping.coordinator import MappingCoordinator, MappingCycleResult  # noqa: E402


# _flush_insertion_batch moved verbatim to
# eligsir/mapping/integration/insertion_committer.py (Wave 1 step 8) --
# no math change. Imported directly here (not via rgbd_integrator_impl.py's
# re-export block below) because train_streaming.py's own streaming_training()
# calls it at two call sites of its own (the batch-insertion path, active only
# when --streaming_insertion_batch_frames > 1), independent of the
# insert_gaussians_from_frame/_run_rolling_seed/_run_submap_stitch cluster
# that moved in Wave 2 Task 2 -- it was never part of that cluster's own
# dependency closure (confirmed unused inside rgbd_integrator_impl.py), so it
# stays bound in this module's own namespace rather than round-tripping
# through the re-export block.
from eligsir.mapping.integration.insertion_committer import (  # noqa: E402
    flush_insertion_batch as _flush_insertion_batch,
)
from eligsir.mapping.integration.device_candidates import (  # noqa: E402
    collected_candidate_points as _collected_candidate_points,
)
from eligsir.mapping.integration.insertion_cadence import (  # noqa: E402
    _should_flush_insertion_batch,
    _mcmc_suppressed_by_insertion_offset,
)

# insert_gaussians_from_frame / _run_rolling_seed / _run_submap_stitch and their
# helper cluster physically moved to
# eligsir/mapping/integration/rgbd_integrator_impl.py (Wave 2 Task 2). Re-exported
# here (not renamed) so every existing call site in this file, and every existing
# `from train_streaming import <name>` in tests, keeps working unchanged.
from eligsir.mapping.integration.rgbd_integrator_impl import (  # noqa: E402
    _add_guarded,
    _admission_quality_weight_for_stats,
    _bootstrap_mask,
    _bootstrap_motion_stats,
    _capture_bootstrap_positions,
    _export_ply_snapshot,
    _flush_dense_trajectory,
    _log_bootstrap_motion,
    _maybe_create_pose_graph,
    _nan_term_report,
    _pose_provenance_report_dict,
    _render_streaming_snapshot,
    _run_rolling_seed,
    _run_submap_stitch,
    _sanitize_gaussian_params_sampled,
    _save_cameras_json,
    _save_holdout_gt,
    _save_streaming_checkpoint,
    _save_test_cameras_json,
    _update_provisional_support,
    _write_bootstrap_motion_ply,
    insert_gaussians_from_frame,
)
# NOTE: _HOLDOUT_GT_SAVED_IDS (state private to _save_holdout_gt) now lives in
# rgbd_integrator_impl.py alongside that function; nothing outside this cluster
# referenced it directly.


def _submit_frame_to_bounded_pipeline(coordinator, frame) -> None:
    """Build a ``TrackedFramePacket`` from an already-ingested
    ``StreamingRGBDFrame`` and offer it (non-blocking) to
    ``coordinator.bounded_pipeline``. Only called when
    ``--streaming_bounded_queue_pipeline`` is on (see the call site in
    ``streaming_training()``); a full no-op otherwise.

    Shadow-only integration (plan section 8, PR W3-03): this exercises the
    real bounded-queue/worker/prefilter machinery on real frame data for
    observability and the synthetic/real acceptance gates, but does not
    (yet) change which frame the existing ``KeyframeAdmitter`` admission
    path trains on -- see ``arguments/__init__.py``'s
    ``streaming_bounded_queue_pipeline`` docstring for why that migration
    is deliberately deferred.
    """
    from eligsir.utils.rgbd_frames import CameraIntrinsics
    from eligsir.mapping.candidates.contracts import (
        RawFramePacket,
        TrackedFramePacket,
        TrackingState,
        UndecodedBufferRef,
    )
    from eligsir.mapping.scheduler.tracking_confidence import (
        TRACKING_STATE_LOST,
        TRACKING_STATE_NOT_INITIALIZED,
        TRACKING_STATE_NO_IMAGES_YET,
        TRACKING_STATE_RECENTLY_LOST,
    )

    if getattr(frame, "c2w", None) is None:
        return  # no pose yet -- nothing to track

    rgb_np = getattr(frame, "_rgb_np", None)
    rgb_ref = UndecodedBufferRef(array=rgb_np) if rgb_np is not None else UndecodedBufferRef(path=frame.rgb_path)
    depth_np = getattr(frame, "_depth_np", None)
    depth_ref = (
        UndecodedBufferRef(array=depth_np) if depth_np is not None else UndecodedBufferRef(path=frame.depth_path)
    )
    intrinsics = CameraIntrinsics(
        width=int(frame.width), height=int(frame.height),
        fx=float(frame.fx), fy=float(frame.fy), cx=float(frame.cx), cy=float(frame.cy),
        depth_scale=float(frame.depth_scale),
    )
    raw = RawFramePacket(
        source_index=int(frame.index),
        timestamp_ns=int(float(frame.timestamp) * 1e9),
        rgb_ref=rgb_ref,
        depth_ref=depth_ref,
        intrinsics=intrinsics,
    )

    tstatus = getattr(frame, "_tracking_status", None)
    tracking_state = TrackingState.UNKNOWN
    tracking_confidence = 1.0
    if tstatus is not None:
        if tstatus.state in (TRACKING_STATE_NOT_INITIALIZED, TRACKING_STATE_NO_IMAGES_YET):
            tracking_state = TrackingState.INITIALIZING
        elif tstatus.state in (TRACKING_STATE_LOST, TRACKING_STATE_RECENTLY_LOST):
            tracking_state = TrackingState.LOST
        else:
            tracking_state = TrackingState.OK
        tracking_confidence = 1.0 if tracking_state is TrackingState.OK else 0.0

    tracked = TrackedFramePacket(
        frame=raw,
        world_T_camera=np.asarray(frame.c2w, dtype=np.float32),
        tracking_state=tracking_state,
        tracking_confidence=tracking_confidence,
        slam_map_id=0,
    )
    coordinator.bounded_pipeline.submit_raw_frame(tracked)


def _mirror_commit_to_pending_queue(coordinator, frame) -> None:
    """Shadow-mirror one real, already-applied accepted-frame map mutation
    into ``coordinator.bounded_pipeline.commit_queue`` (plan section 8.4's
    ``PendingCommitQueue``). Only called when
    ``--streaming_bounded_queue_pipeline`` is on, immediately after the
    REAL commit (``_coordinator.step()`` / ``_flush_insertion_batch()``)
    has already run and mutated the live Gaussian model -- see the call
    site in ``streaming_training()``.

    Not 1:1 with real map mutations: the call site only runs this inside
    the ``is_train and _p8_allow_insert and streaming_insert_from_depth``
    depth-insertion block -- an accepted frame skipped by the cap guard
    (``streaming_depth_respects_cap``), warmup
    (``streaming_warmup_min_frames``), or the pure-refinement horizon
    (``_streaming_insertion_stop_iter``) produces no mirrored commit at
    all, since no real depth-insertion cycle ran to mirror. Under batched
    insertion (``_batch_frames > 1``) this fires once per accepted frame
    with whatever ``added`` count that accumulation step reported
    (usually 0, since the real Gaussian insertion happens later at
    ``_flush_insertion_batch()``), not once per real flush -- deliberately
    a per-accepted-frame cadence signal, not a per-real-commit one. The
    exact per-frame insertion count is not carried into the mirrored
    ``CommitRequest`` (see below) since ``PreparedFrameCandidate``/
    ``CommitRequest`` have no field for it and this mirror only needs to
    exercise queue mechanics, not reconstruct the real commit's payload.

    Why observation-only, not the real commit path: this PR (W3-03) is a
    shadow-integration PR -- every other piece of the bounded-queue
    pipeline (``BoundedCandidatePipeline.submit_raw_frame``/
    ``poll_prepared_candidate`` above) exercises the new machinery on real
    data without letting it drive any real decision, and this mirror
    keeps that same contract for the commit queue specifically. Making
    ``PendingCommitQueue`` the actual, authoritative commit path would
    mean routing the existing admission-gated insertion call
    (``_coordinator.step`` / the batch-flush branch a few lines above)
    THROUGH ``submit_commit``/``drain_commits`` instead of calling it
    directly -- i.e. replacing today's synchronous "admit decision ->
    insert immediately" flow with "admit decision -> enqueue commit
    request -> drain at an explicit barrier -> insert". That is a real
    behavior change to the map-mutation call path (batching semantics,
    ordering guarantees under backpressure, what happens on
    ``DROPPED_QUEUE_FULL``) that needs its own dedicated design/testing,
    not a good default to fold silently into this PR's shadow wiring --
    tracked as real future work (plan section 8's PR W3-07, the same PR
    that migrates real admission onto this pipeline). Mirroring after the
    fact, with a lightweight synthetic candidate (no image/depth arrays,
    the real ones were already consumed by the real commit above), is
    the safe way to exercise ``PendingCommitQueue.submit``/``drain`` on
    real per-frame cadence without risking a double-commit or a
    real/shadow map-state divergence.
    """
    from eligsir.contracts.types import Pose
    from eligsir.mapping.candidates.contracts import (
        InputEvidence,
        PreparedDepthRef,
        PreparedFrameCandidate,
        PreparedImageRef,
    )
    from eligsir.mapping.commit_queue import CommitRequest

    c2w = getattr(frame, "c2w", None)
    if c2w is None:
        return
    candidate = PreparedFrameCandidate(
        source_index=int(frame.index),
        timestamp_ns=int(float(frame.timestamp) * 1e9),
        pose=Pose(np.asarray(c2w, dtype=np.float32)),
        # Deliberately empty: the real RGB/depth arrays were already
        # decoded and consumed by the real commit that just ran; this
        # mirror only needs to exercise queue mechanics (offer/drain,
        # depth/backpressure metrics), not re-carry full frame payloads.
        rgb_ref=PreparedImageRef(array=None, width=0, height=0),
        depth_ref=PreparedDepthRef(array=None, width=0, height=0, valid_fraction=0.0),
        input_evidence=InputEvidence(
            blur_laplacian=0.0, exposure_luma=0.0, white_clip_ratio=0.0,
            black_clip_ratio=0.0, valid_depth_fraction=0.0,
            quality_weight=1.0, reject_reason=None,
        ),
        coarse_novelty=0.0,
    )
    request = CommitRequest(candidate=candidate, slam_map_id=0)
    # A DROPPED_QUEUE_FULL here is real backpressure per PendingCommitQueue's
    # own contract (see its module docstring); in shadow mode there is
    # nothing to widen or slow, so it is left as a visible
    # metrics().dropped_count tick rather than raising -- the drain below
    # still runs every call, so steady-state depth should stay low. The
    # QueueOfferResult return value itself is intentionally not branched on.
    coordinator.bounded_pipeline.submit_commit(request)
    # Barrier drain every call (mirrors the existing
    # poll_prepared_candidate() non-blocking-drain idiom a few lines above
    # in the ingest loop) so the queue doesn't silently accumulate
    # unobserved; the drained requests are discarded -- shadow-only, see
    # this function's docstring.
    coordinator.bounded_pipeline.drain_commits()


def bootstrap_frame_passes_confidence_gate(
    tracking_result, min_confidence: float
) -> bool:
    """W3-30: should this SLAM-backend tracking result be accepted as a
    bootstrap (initial-seed) frame?

    A hard, absolute-threshold gate -- distinct from the existing soft,
    trajectory-relative tracking_confidence weighting (which downweights an
    already-admitted frame's loss contribution, applied after bootstrap) and
    from the motion pregate (which rejects stale/interpolated poses). This
    gate exists because bootstrap seeds cameras_extent (monotonic,
    never-shrinks, computed once from the bootstrap set with no outlier
    rejection) and the initial point cloud -- both permanent for the rest of
    the run -- so a single low-confidence early pose (SLAM tracking
    confidence is characteristically lowest right after backend
    initialize()/track(), before pose covariance has converged) can
    permanently corrupt scene scale. See
    docs/architecture/2026-08-10-w3-30-init-seed-bootstrap-confidence-gate.md.

    ``min_confidence <= 0.0`` disables the gate entirely (matches every
    non-ROS adapter's fixed confidence=1.0, so this is a no-op for them
    regardless of value) -- returns True whenever the result is otherwise
    valid, reproducing pre-W3-30 behavior exactly.
    """
    if tracking_result is None or not tracking_result.valid:
        return False
    if min_confidence <= 0.0:
        return True
    return tracking_result.confidence >= min_confidence


def _streaming_emergency_finalize_output_staging(args):
    """Mirrors train.py's `_emergency_finalize_output_staging`: a shorter
    timeout than the normal-completion path (600s) because this runs inside
    a preemption grace window, not an unbounded budget. Never raises --
    the SIGTERM/SIGINT handlers that call this must still reach their
    `os._exit`/`raise` regardless of outcome -- but still writes the
    `.COMPLETE`/`.FAILED` marker so a gracefully-stopped staged run isn't
    left local-only with no explanation. A no-op when staging was never
    active (delegates entirely to `train._finalize_output_staging`'s own
    no-op check).

    Module-level (not a nested closure inside `streaming_training`) so it
    can be unit-tested directly without running the full training loop --
    see tests/test_output_staging_streaming.py.
    """
    from train import _finalize_output_staging, OutputPersistenceError
    try:
        _finalize_output_staging(args, timeout=60.0)
    except OutputPersistenceError:
        pass
    except Exception as e:
        print(f"[streaming] output-staging finalize FAILED during emergency "
              f"shutdown: {e}", flush=True)


def _alt_trainer_run_state(args):
    """Issue #1102: build the `RunCompletionState` for an alternate trainer.

    `submap_stitch` / `rolling_seed` are dispatched before the main loop and
    used to emit no completion evidence at all. They now carry the same state
    object every other training path carries; the mode records its report as a
    real evidence phase into it, and the call site writes it, emits the
    evidence artifacts from it, and gates the exit on it.

    Deliberately NOT wired to `--no-streaming_report_final` /
    `--streaming_benchmark_mode`: neither mode reads those flags, so both write
    their report whenever test cameras exist regardless. Recording a
    NOT_REQUESTED verdict here would state the opposite of what the run then
    puts on disk -- and a state file that contradicts the artifacts is worse
    than no state file. Making those flags actually skip the report inside the
    two modes is a behaviour change to the trainers, tracked separately.
    """
    from eligsir.utils.run_completion_state import RunCompletionState

    return RunCompletionState(args.model_path)


def _compose_mapping_application(
    streaming_scene,
    optimizer_tick,
    callbacks: StreamingIngestCallbacks,
):
    """Join the initialized trainer owners through the canonical factory."""
    return create_mapping_application(
        MappingApplicationOptions(
            mapper_sink=StreamingSceneMappingSink(streaming_scene, callbacks),
            optimizer_tick=optimizer_tick,
            flush_hook=streaming_scene.flush_deferred_extent_update,
            close_hook=streaming_scene.close_decode_cache,
        )
    )


def streaming_training(
    dataset,
    opt,
    pipe,
    testing_iterations,
    saving_iterations,
    checkpoint_iterations,
    checkpoint,
    debug_from,
    sh_degree_schedule,
    run_args=None,
):
    if run_args is None:
        raise ValueError("streaming_training() requires run_args.")
    args = run_args

    # Issue #399 follow-up: the bound was previously parsed into `args` but
    # never applied anywhere -- `_enforce_sensor_depth_cache_bound` had no
    # production caller. This makes the flag load-bearing for the whole run.
    set_sensor_depth_cache_bound(
        int(getattr(args, "streaming_sensor_depth_cache_max_frames", 0))
    )

    from eligsir.utils.experiment_origin import validate_researchflow_origin
    validate_researchflow_origin(args)
    # Issue #951: the scored holdout remains evaluation-only unless the user
    # explicitly requests the historical oracle intervention. The runtime owns
    # role assignment, disjointness validation, and the durable action trace.
    from eligsir.utils.partition_control import (
        ControlSignalSource as _PartitionControlSignalSource,
        EvaluationLeakageError as _EvaluationLeakageError,
        PartitionControlRuntime as _PartitionControlRuntime,
        decide_partition_focus as _decide_partition_focus,
    )
    _partition_focus_enabled = (
        float(getattr(args, "streaming_partition_focus_weight", 0.0)) > 0.0
    )
    _partition_control_runtime = _PartitionControlRuntime(
        source=args.streaming_partition_focus_signal_source,
        control_probe_n=args.streaming_partition_focus_control_probe_n,
        model_path=args.model_path,
        controller_enabled=_partition_focus_enabled,
    )
    _partition_control_roles = None
    _partition_control_last_decision = None
    if _partition_focus_enabled:
        if (
            _partition_control_runtime.source
            is _PartitionControlSignalSource.SCORED_HOLDOUT_ORACLE
        ):
            print(
                "[partition-control] ORACLE_CONTROL enabled: scored-holdout "
                "observations will actuate training; this run is an "
                "ORACLE_INTERVENTION and cannot support an unbiased test verdict.",
                flush=True,
            )
        else:
            print(
                "[partition-control] non-oracle source="
                f"{_partition_control_runtime.source.value}; scored holdout is "
                "evaluation-only.",
                flush=True,
            )
    # Issue #729: `_streaming_report_kwargs()` (defined further down, shared
    # by all four report call sites -- pre-training, mid-training via
    # `_run_mid_report`, and both final-report branches) reads
    # `_git_provenance_from_args` as a free variable resolved from THIS
    # enclosing scope. It used to be bound only by the pre-training import
    # (gated behind `--streaming_report_pre_training`, default off) or the
    # final-report imports (which run only after every mid-training report
    # has already fired) -- so on a default run the mid-training path hit a
    # `NameError` on every tick, silently losing the per-checkpoint PSNR-curve
    # evidence CLAUDE.md's run contracts require. Bind it once, here,
    # unconditionally, before any report call site can possibly run. Guarded
    # (unlike the imports inside the FINAL report's try-protected phase
    # blocks, which are deliberately left where a mid-run import failure
    # cannot skip the shutdown tail -- this one is different: it runs before
    # any of that machinery even exists, so an unguarded ImportError here
    # would abort the run before training starts. Degrade instead: report
    # writes simply omit git_provenance, same as an args namespace missing
    # git_commit/git_branch/git_dirty already does per that function's own
    # contract).
    try:
        from eligsir.utils.comparison_report import (
            git_provenance_from_args as _git_provenance_from_args,
        )
    except ImportError as _e:
        print(
            f"[streaming] WARNING: git_provenance_from_args unavailable "
            f"({_e}); report.json['git_provenance'] will be omitted for this "
            f"run.",
            flush=True,
        )

        def _git_provenance_from_args(_args):
            return None
    # Issue #464: these are module-global counters, so a second
    # streaming_training() in the same process would otherwise inherit the
    # previous run's resolution-cache failure totals.
    reset_multires_cache_stats()
    # PR3 (2026-08-12 cadence/batching plan): reject an unsupported
    # multi-view gradient-accumulation combination at startup, before any
    # scene/model work, rather than mid-run. No-op at the default
    # streaming_optimization_batch_size == 1.
    validate_streaming_optimization_batch_size(args)
    # Issue #360: warn (never alter behavior) when --streaming_replay_buffer was
    # set to a non-default value in a configuration where the global reservoir
    # makes it unreachable. Print-only; the sampler is untouched.
    for _inert_warning in check_replay_buffer_reachability(args):
        print(_inert_warning, flush=True)
    # Per-run MappingCoordinator instance (was a module-level singleton
    # until 2026-08-06; that mixed telemetry from multiple runs in the
    # same process, could keep an old run's model GPU-reachable after it
    # ended, and had a theoretical Python object-id-reuse risk for the
    # backend cache). Constructed here, at the actual composition root,
    # and closed over by the three integration call sites below (the
    # per-frame depth-insertion call in the main loop, and the
    # rolling_seed/submap_stitch mode-dispatch branches) -- same call
    # sites as before, just no longer sharing state across runs.
    _coordinator = MappingCoordinator()
    # Wave 3, PR W3-05: observation-only mapping-evidence collection (see
    # eligsir/mapping/evidence/ and plan section 10). No consumer PR
    # exists yet, so requirements defaults to EvidenceRequirements.none()
    # (every flag off) -- EvidenceBuilder.build_gaussian_evidence() then
    # returns None immediately, before touching any tensor, at the one
    # call site below that constructs a GaussianEnergyTerms instance
    # anyway for TensorBoard logging. This does not change what gets
    # rendered/backpropagated/decided: it only ever reads the already-
    # constructed `_log_terms` object, never `utility`/`temperature` (the
    # values that actually drive MCMC birth/death), and its result is
    # discarded, not stored anywhere the training loop reads back from.
    # Wave 3, PR W3-06: shadow frame-utility scoring (see
    # eligsir/mapping/admission/ and plan section 11). Consumes exactly
    # the evidence fields the utility formula (plan section 11.3) needs --
    # want_input_evidence, want_render_evidence/want_rgb_residual/
    # want_alpha/want_rendered_depth, want_gaussian_evidence/
    # want_visibility/want_xyz_grad_ema/want_opacity_grad_ema/want_support/
    # want_geometry_confidence, want_summary -- rather than
    # EvidenceRequirements.everything(), so the policy-cost measurement
    # (deliverable 5) reflects this scorer's actual cost, not the
    # worst-case "every flag on" cost W3-05 already benchmarked separately.
    # Default OFF (--streaming_shadow_utility_enabled) reproduces
    # EvidenceRequirements.none() exactly, same as before this PR.
    #
    # Wave 3, PR W3-07: any rollout mode other than "shadow" needs a real
    # utility score to act on, so it force-enables the underlying W3-06
    # scorer even if --streaming_shadow_utility_enabled itself was left at
    # its default False. The *rollout* default is "shadow" (see
    # --streaming_admission_rollout_mode's docstring below), so this has
    # zero effect on any run that doesn't explicitly opt into a non-shadow
    # rollout mode -- byte-identical to pre-W3-07 behavior in that case.
    _admission_rollout_mode_str = str(getattr(args, "streaming_admission_rollout_mode", "shadow"))
    _shadow_utility_enabled = bool(getattr(args, "streaming_shadow_utility_enabled", False)) or (
        _admission_rollout_mode_str != "shadow"
    )
    if _shadow_utility_enabled:
        _evidence_requirements = EvidenceRequirements(
            want_input_evidence=True,
            want_render_evidence=True,
            want_rgb_residual=True,
            want_alpha=True,
            want_rendered_depth=True,
            want_gaussian_evidence=True,
            want_visibility=True,
            want_xyz_grad_ema=True,
            want_opacity_grad_ema=True,
            want_support=True,
            want_geometry_confidence=True,
            want_summary=True,
        )
    else:
        _evidence_requirements = EvidenceRequirements.none()
    _gaussian_evidence_history = GaussianEvidenceHistory()
    _evidence_builder = EvidenceBuilder(_evidence_requirements, history=_gaussian_evidence_history)
    from eligsir.mapping.admission.shadow_policy import ShadowUtilityPolicy
    from eligsir.mapping.admission.utility_contracts import UtilityWeights

    _utility_weights = UtilityWeights(
        w_new_surface=float(getattr(args, "streaming_shadow_utility_w_new_surface", 1.0)),
        w_residual=float(getattr(args, "streaming_shadow_utility_w_residual", 1.0)),
        w_debt=float(getattr(args, "streaming_shadow_utility_w_debt", 1.0)),
        w_gradient=float(getattr(args, "streaming_shadow_utility_w_gradient", 1.0)),
        w_parallax=float(getattr(args, "streaming_shadow_utility_w_parallax", 0.5)),
        w_redundancy=float(getattr(args, "streaming_shadow_utility_w_redundancy", 1.0)),
        w_specular=float(getattr(args, "streaming_shadow_utility_w_specular", 1.0)),
        w_cost=float(getattr(args, "streaming_shadow_utility_w_cost", 0.25)),
    )
    _shadow_utility_policy = ShadowUtilityPolicy(
        enabled=_shadow_utility_enabled,
        log_dir=os.path.join(args.model_path, "shadow_utility") if _shadow_utility_enabled else None,
        realized_gain_horizon=int(getattr(args, "streaming_shadow_utility_realized_gain_horizon", 200)),
        gaussian_history=_gaussian_evidence_history,
        weights=_utility_weights,
        # Issue #118: default "raw" reproduces byte-identical pre-existing
        # behavior; see arguments/__init__.py's flag docstring.
        debt_normalization=str(getattr(args, "streaming_shadow_utility_debt_normalization", "raw")),
    )
    # Wave 3, PR W3-07: graduated rollout policy acting on the score above.
    # **The wired-in default is RolloutMode.SHADOW** -- no decision changes
    # -- because there is not yet cross-scene evidence to justify promoting
    # this scorer to advisory/active for any real run (deliverable 6). See
    # eligsir/mapping/admission/rollout_policy.py's module docstring
    # for the hard-safety-gate invariant this policy can never override,
    # in any mode.
    from eligsir.mapping.admission.budget_controller import (
        MappingBudgetController,
        MappingBudgetControllerConfig,
    )
    from eligsir.mapping.admission.cost_model import predict_gpu_cost
    from eligsir.mapping.admission.decision_contracts import RolloutMode
    from eligsir.mapping.admission.rollout_policy import AdmissionRolloutPolicy

    _admission_rollout_mode = RolloutMode(_admission_rollout_mode_str)
    _admission_budget_controller = None
    if _admission_rollout_mode == RolloutMode.ACTIVE_WITH_BUDGET:
        _admission_budget_controller = MappingBudgetController(
            MappingBudgetControllerConfig(
                max_frames_per_window=int(getattr(args, "streaming_admission_budget_max_frames", 0)),
                max_compute_per_window=float(getattr(args, "streaming_admission_budget_max_compute", 0.0)),
                max_burst_steps_per_frame=int(getattr(args, "streaming_admission_budget_max_burst_steps", 200)),
            )
        )
    _admission_rollout_policy = AdmissionRolloutPolicy(
        mode=_admission_rollout_mode,
        utility_admit_threshold=float(getattr(args, "streaming_admission_utility_threshold", 0.0)),
        budget_controller=_admission_budget_controller,
    )
    # Bounded-queue prepared-candidate pipeline (PR W3-03, default off --
    # see --streaming_bounded_queue_pipeline's docstring in
    # arguments/__init__.py). Enabling this only starts the
    # FramePreparationWorker background thread and its three bounded
    # queues; it does not yet change which frames the existing admission
    # path (_keyframe_admission_closure / KeyframeAdmitter) trains on.
    if getattr(args, "streaming_bounded_queue_pipeline", False):
        from eligsir.runtime.workers.frame_preparation_worker import (
            FramePreparationWorkerConfig,
        )

        _coordinator.enable_bounded_queue_pipeline(
            input_capacity=int(getattr(args, "streaming_bounded_queue_input_capacity", 4)),
            output_capacity=int(getattr(args, "streaming_bounded_queue_output_capacity", 4)),
            commit_capacity=int(getattr(args, "streaming_bounded_queue_commit_capacity", 8)),
            worker_config=FramePreparationWorkerConfig(),
        )
        print(
            "[streaming] bounded-queue prepared-candidate pipeline ON "
            "(FramePreparationWorker thread started; shadow/observe-only, "
            "not yet driving admission).",
            flush=True,
        )
    # Loud, unmissable run-start banner. ablator/queue reruns reuse the same
    # model_path across attempts (CLAUDE.md's "unique output folder per run"
    # is violated by the rerun/requeue mechanism itself), so train.log
    # silently concatenates old and new runs' output with no visual
    # separator -- repeatedly caused real confusion this session (stale
    # cfg_args/report.json mistaken for the current attempt, host/machine
    # of an OLD run mistaken for the current one). This banner makes a new
    # attempt starting in an already-populated log impossible to miss.
    import socket as _socket
    import os as _os
    import datetime as _datetime
    print(
        "\n"
        + "=" * 88 + "\n"
        + f"[RUN START] {_datetime.datetime.now().isoformat(timespec='seconds')} "
        + f"host={_socket.gethostname()} pid={_os.getpid()} "
        + f"model_path={args.model_path}\n"
        + "=" * 88,
        flush=True,
    )
    # Issue #705: the entrypoint already validated/persisted this context after
    # installing train.log. Re-emit the identical banner after [RUN START] so
    # the last-attempt segment of a reused legacy/requeue log carries its own
    # declaration bytes and hash rather than borrowing an earlier attempt's.
    _experiment_context = getattr(args, "experiment_context", None)
    if _experiment_context is not None:
        from eligsir.utils.experiment_declaration import render_experiment_banner

        print(render_experiment_banner(args.experiment_context), flush=True)
    # Issue #701: print the resolved execution-semantics banner here too, not
    # only relying on train.py's __main__ printing it before dispatch. The
    # __main__ banner (build_execution_semantics_block()/
    # format_execution_semantics(), also issue #168) already fires on every
    # real launch since train.py is the sole entry point that reaches
    # streaming_training() -- but that print lands BEFORE the [RUN START]
    # banner above and before _apply_camera_profile() below, so on a rerun
    # whose train.log silently concatenates multiple [RUN START] segments
    # (CLAUDE.md's "unique output folder per run" note on the rerun/requeue
    # mechanism), the __main__ banner from an EARLIER attempt in the same log
    # is not reliably attributable to the CURRENT attempt without also
    # searching before the last [RUN START] line -- exactly the ambiguity
    # that banner exists to remove. Printing it again immediately after
    # [RUN START] keeps every streaming attempt self-describing from the
    # same anchor point a reader already searches for.
    print(
        format_execution_semantics(build_execution_semantics_block(args)),
        flush=True,
    )
    _apply_camera_profile(args)

    # Issue #792: detect max_gap < frames_between_admits mismatch after profile application.
    _admission_preflight = check_admission_preflight(
        # Issue #363/#368: these six are unconditional ParamGroup defaults
        # (arguments/__init__.py:1298-1378), always present on the real parsed
        # args -- bare attribute access, never a getattr fallback carrying a
        # hand-copied literal that can silently drift from the real default.
        admission_mode=str(args.streaming_frame_admission),
        velocity_adaptive=bool(args.streaming_keyframe_velocity_adaptive),
        input_fps=float(args.streaming_keyframe_input_fps),
        target_admit_fps=float(args.streaming_keyframe_target_admit_fps),
        max_gap=int(args.streaming_keyframe_max_gap),
        max_gap_require_novelty=bool(args.streaming_keyframe_max_gap_require_novelty),
        # Issue #928: lets the preflight tell a statically-known-inert
        # novelty gate (coverage_gain_gate=False) apart from a merely
        # requested one -- see check_admission_preflight's own docstring.
        coverage_gain_gate=bool(args.streaming_keyframe_coverage_gain_gate),
    )
    if _admission_preflight.warning_text:
        print(_admission_preflight.warning_text, flush=True)

    from eligsir.slam.pose_sources import (
        is_dataset_pose_source,
        normalize_pose_source,
    )

    _pose_source_name = normalize_pose_source(getattr(args, "pose_source", "dataset_pose"))
    if is_dataset_pose_source(_pose_source_name) and args.orbbec_pose_source == "open3d_odometry_live":
        _pose_source_name = "open3d_odometry_live"

    # Make the pose provenance unmistakable in the logs. For ros2://live the frame
    # source owns pose sync and embeds the pose in frame.c2w, so the in-process
    # SLAM backend track() is intentionally NOT called — the poses still come from
    # the external SLAM frontend over ROS (e.g. ORB-SLAM3 /slam/pose). Without this
    # line "dataset_pose" reads as if baked poses were used, which they are not.
    if str(getattr(dataset, "source_path", "")).startswith("ros2://"):
        _live_pose_topic = args.ros2_live_pose_topic
        _live_pose_kind = args.ros2_live_pose_source
        print(
            f"[streaming] pose source: ros2 live — consuming external frontend poses "
            f"from topic '{_live_pose_topic}' (kind={_live_pose_kind}); in-process "
            f"SLAM track() is skipped because the frame source already embeds the pose.",
            flush=True,
        )
    else:
        print(f"[streaming] pose source: {_pose_source_name}", flush=True)

    from eligsir.slam.pose_provenance import resolve_pose_provenance

    _is_ros2_live = str(getattr(dataset, "source_path", "")).startswith("ros2://")
    _provenance_pose_topic = (
        args.ros2_live_pose_topic
        if _is_ros2_live
        else args.orbbec_pose_topic
    )
    pose_provenance_result = resolve_pose_provenance(
        cli_override=getattr(args, "pose_provenance", "auto"),
        source_path=getattr(dataset, "source_path", ""),
        pose_topic=_provenance_pose_topic,
        orbbec_pose_source=args.orbbec_pose_source,
    )
    args.pose_provenance_result = pose_provenance_result
    print(pose_provenance_result.banner_line(), flush=True)
    if pose_provenance_result.tier.value == "unknown":
        print(
            "[pose-provenance] !!! WARNING: pose provenance could not be auto-detected "
            "for this run. Pass --pose_provenance {dataset_gt,colmap_pseudo_gt,"
            "slam_estimated,unknown} explicitly if you know the source, otherwise "
            "every quality claim (PSNR, geometry) from this run must be treated as "
            "provenance-unverified.",
            flush=True,
        )

    # #602: capture the REQUESTED admission policy before the in-process
    # override below can rewrite it. Nothing else persists this value for a
    # streaming run: cfg_args is written later (~line 2155), and
    # _execution_semantics_from_args() is evaluated at report time, so both
    # record the post-override RESOLVED value. Without this line the requested
    # value survives only in a stdout banner, and a reader cannot tell an
    # explicitly-requested hybrid_keyframe run apart from an `all` run that
    # was silently promoted to one.
    #
    # Scoping on a RESUME: this is whatever the resuming invocation passed, so
    # it describes the post-resume segment only -- and the recorder itself is
    # in-memory and deliberately not checkpointed, so the whole
    # admission_cadence block inherits the same scoping. That is the same
    # honest-partial-coverage stance RealizedCadenceTracker already takes (see
    # its summary()["coverage"]["partial_coverage"]), not a new caveat.
    _admission_policy_requested = str(args.streaming_frame_admission)

    if (
        args.orbbec_pose_source == "open3d_odometry_live"
        and args.streaming_frame_admission == "all"
    ):
        args.streaming_frame_admission = "hybrid_keyframe"

    if getattr(args, "online_mode", False):
        from eligsir.core.stop_policy import apply_online_schedule_overrides

        # Three independent online horizons (all 0/default = neutral):
        #  - schedule_horizon (A6): anneals the FULL schedule incl. dead-opacity ramp
        #    (over-prunes short runs; use with care).
        #  - anneal_horizon (Merge B): anneals only the count-loss N_target toward
        #    streaming_target_capacity, leaving dead-opacity unbounded (no over-prune)
        #    — the gentle N-bound lever.
        #  - noise_stop_horizon (2026-07-06, extended 2026-07-06b): pure-refinement
        #    switch — stops MCMC xyz-noise injection, MCMC growth/relocation-driven
        #    insertion, AND the streaming pipeline's own depth-based Gaussian
        #    insertion (gated below via opt._streaming_insertion_stop_iter) all at
        #    the same point, leaving dead-opacity-ramp/densify-until unbounded (no
        #    schedule_horizon over-prune risk). Live testing found noise-stop ALONE
        #    doesn't reduce floaters (new Gaussians kept entering from newly-ingested
        #    frames regardless) — this makes it a real "stop building, start
        #    refining" switch instead.
        _sched_horizon = int(getattr(args, "streaming_online_schedule_horizon", 0) or 0)
        _anneal_h = int(getattr(args, "streaming_anneal_horizon", 0) or 0)
        _noise_stop_h = int(getattr(args, "streaming_noise_stop_horizon", 0) or 0)
        _growth_stop_h = int(getattr(args, "streaming_growth_stop_horizon", 0) or 0)
        apply_online_schedule_overrides(
            opt,
            schedule_horizon=_sched_horizon if _sched_horizon > 0 else None,
            anneal_horizon=_anneal_h if _anneal_h > 0 else None,
            noise_stop_horizon=_noise_stop_h if _noise_stop_h > 0 else None,
            growth_stop_horizon=_growth_stop_h if _growth_stop_h > 0 else None,
        )
        # Bug found 2026-07-06b: apply_online_schedule_overrides() mutates `opt`
        # (an OptimizationParams().extract(args) GroupParams — a SEPARATE object
        # from `args`/run_args), but utils/strategies/mcmc_strategy.py's actual
        # growth-stop/noise-stop reads are `getattr(args, "mcmc_stop_growth_iter"/
        # "mcmc_noise_stop_iter", ...)` — i.e. from `args`, never from `opt`. With
        # no sync step, --streaming_noise_stop_horizon's opt.mcmc_noise_stop_iter
        # write was silently never seen by the strategy that reads args. Mirror
        # the overridden fields onto `args` too so the strategy actually sees them.
        args.mcmc_noise_stop_iter = opt.mcmc_noise_stop_iter
        args.mcmc_stop_growth_iter = opt.mcmc_stop_growth_iter
        args.densify_until_iter = opt.densify_until_iter

        # Scene-complexity (SV) and input-quality accumulators are *diagnostic
        # telemetry* for offline ablation reports; nothing in the live loop reads
        # them. They are also the dominant per-frame ingest cost: measured on
        # kitchen1 ORB-SLAM live, accumulate_observed_surface was ~64ms of a
        # ~80ms m_ingest (×2 frames/iter), so disabling it drops the ingestion
        # iteration from ~150ms to ~22ms (~6-7× live it/s) with byte-identical
        # reconstruction. Default them off in online_mode; an offline
        # streaming_replay eval keeps them on (global default True). Explicit
        # CLI opt-in still wins via --streaming_track_scene_complexity.
        for _diag_attr in ("streaming_track_scene_complexity",
                           "streaming_track_input_quality"):
            if not _cli_overrode_arg(_diag_attr):
                setattr(args, _diag_attr, False)

    # Issue #396 is a fixed-topology residency measurement.  Fail closed
    # before initialization if online normalization or a later insertion path
    # could invalidate that premise.  The dedicated horizons are required even
    # when depth insertion is disabled so the run receipt proves both owners'
    # resolved stop boundaries.
    if str(args.experiment_origin_issue) == "396":
        from eligsir.core.stop_policy import validate_fixed_topology_contract

        _fixed_topology_errors = validate_fixed_topology_contract(
            online_mode=bool(args.online_mode),
            requested_growth_stop=args.mcmc_stop_growth_iter,
            explicit_growth_stop=args.streaming_growth_stop_horizon,
            resolved_growth_stop=args.mcmc_stop_growth_iter,
            insertion_enabled=bool(args.streaming_insert_from_depth),
            explicit_insertion_stop=args.streaming_insert_stop_iter,
        )
        if _fixed_topology_errors:
            raise ValueError(
                "issue #396 fixed-topology preflight failed: "
                + "; ".join(_fixed_topology_errors)
            )

    # --- Platform banner (restored from 8c47d93, lost in branch lineage) ----
    # Identifies which machine/GPU produced a run directly from train.log.
    import platform as _platform
    _gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    _gpu_mem_gb = (torch.cuda.get_device_properties(0).total_memory / 1e9) if torch.cuda.is_available() else 0
    print(
        f"[platform] host={_platform.node()} "
        f"python={_platform.python_version()} "
        f"torch={torch.__version__} "
        f"gpu={_gpu_name!r} "
        f"vram={_gpu_mem_gb:.1f}GB",
        flush=True,
    )

    # --- Output / logger ---------------------------------------------------
    os.makedirs(args.model_path, exist_ok=True)
    from eligsir.utils.cfg_args_io import write_cfg_args
    write_cfg_args(os.path.join(args.model_path, "cfg_args"), args)

    # Issue #511: the streaming path writes its own root-level run files, so
    # write the task-provenance record here too. Idempotent with train.py's
    # startup write (same resolved object, same path) -- belt and braces so the
    # file follows whatever path cfg_args uses, independent of staging.
    try:
        from eligsir.utils.experiment_origin import write_experiment_origin_file
        _origin = getattr(args, "experiment_origin", None)
        if isinstance(_origin, dict):
            write_experiment_origin_file(
                args.model_path, _origin,
                persist_path=getattr(args, "persist_model_path", None))
    except Exception as _e:
        print(f"[origin] failed to write experiment_origin.json: {_e!r}", flush=True)

    # Write reproduce.sh (includes docker compose wrapper when running in container)
    import sys as _sys
    from eligsir.utils.comparison_report import write_reproduce_sh
    write_reproduce_sh(os.path.join(args.model_path, "reproduce.sh"), _sys.argv)

    # W3-02 opt-in Level-D evidence: GPU-event optimization-step timing +
    # frozen-mutation-trace record/verify (eligsir/experiments/w3_02_runtime.py,
    # eligsir/optimization/gaussian/mutation_trace.py). Both are strict no-ops
    # unless explicitly requested via --w3_02_* flags -- every other run pays
    # zero overhead and writes zero extra files.
    _w3_02_evidence = None
    if getattr(args, "w3_02_evidence_manifest", ""):
        if not getattr(args, "w3_02_evidence_arm", None):
            raise ValueError("--w3_02_evidence_manifest requires --w3_02_evidence_arm")
        if not args.start_checkpoint:
            raise ValueError("W3-02 Level-D evidence requires --start_checkpoint")
        if not getattr(args, "w3_02_mutation_trace", ""):
            raise ValueError("W3-02 Level-D evidence requires --w3_02_mutation_trace")
        if not getattr(args, "w3_02_git_commit", "") or not getattr(args, "w3_02_clean_checkout", False):
            raise ValueError("W3-02 Level-D evidence requires host-injected --w3_02_git_commit and --w3_02_clean_checkout")
        for _artifact_arg in (
            "w3_02_config_artifact",
            "w3_02_source_artifact", "w3_02_pose_artifact",
            "w3_02_keyframe_artifact", "w3_02_holdout_artifact",
        ):
            if not getattr(args, _artifact_arg, ""):
                raise ValueError(f"W3-02 Level-D evidence requires --{_artifact_arg}")
        from eligsir.experiments.w3_02_runtime import W3_02OptimizationTimingRecorder
        _w3_02_evidence = W3_02OptimizationTimingRecorder(torch)

    from eligsir.optimization.gaussian.mutation_trace import MutationTraceRecorder
    _w3_02_mutation_trace = MutationTraceRecorder(
        mode=getattr(args, "w3_02_mutation_trace_mode", "off"),
        path=getattr(args, "w3_02_mutation_trace_events", ""),
    )

    from eligsir.optimization.gaussian.cohort_trace import CohortTraceRecorder
    _cohort_trace = CohortTraceRecorder(
        enabled=bool(getattr(args, "streaming_cohort_trace_enabled", False)),
        path=os.path.join(args.model_path, "cohort_trace.jsonl") if getattr(args, "streaming_cohort_trace_enabled", False) else "",
    )

    tb_writer = None
    if _TB:
        # max_queue defaults to 10: add_scalar() blocks the MAIN training
        # thread once the queue fills, waiting for the background writer
        # thread to drain events to disk. model_path is typically NFS-backed
        # (for example, a local scratch directory), so write latency spikes stall training
        # every iteration a scalar is logged — confirmed via py-spy: 4/6
        # samples caught MainThread blocked in add_event->queue.put() during
        # a "slow" ingestion-phase run. A much larger queue absorbs NFS
        # latency bursts without blocking the training loop.
        tb_writer = SummaryWriter(args.model_path, max_queue=1000, flush_secs=30)
        tb_writer.add_text("run/pose_provenance", pose_provenance_result.banner_line(), 0)
    else:
        print("[streaming] TensorBoard not available.")

    save_worker = AsyncSaveWorker()
    from eligsir.runtime.report_worker import AsyncReportWorker
    report_worker = AsyncReportWorker()
    from eligsir.utils.seed_fade_monitor import SeedFadeMonitor
    # issue #282: observe_only folds BOTH flags into one piece of monitor
    # state so report_block() can honestly say whether suppression was ever
    # applied (the old `controller_engaged` was an alias of `detected` and
    # reported True even for --streaming_seed_fade_disabled runs).
    seed_fade_monitor = SeedFadeMonitor(
        warmup_iters=getattr(args, "streaming_seed_fade_warmup_iters", 2000),
        # Bare attribute access, not getattr(..., <literal>): both flags are
        # unconditional store_true parser arguments (eligsir/cli/parser.py),
        # so a fallback literal here could only ever mask a real wiring break
        # -- and would be exactly the drift class issue #363 removed.
        observe_only=bool(
            args.streaming_seed_fade_disabled
            or args.streaming_seed_fade_observe_only
        ),
        # No getattr fallback: scalar_log_interval is an unconditional
        # OptimizationParams default, always present on the real parsed args
        # (issue #368 -- old getattr(..., 10) fallback drifted from the real
        # default of 100; removed rather than re-stated -- exactly the drift
        # class the comment above already warns about).
        sample_interval=max(1, int(args.scalar_log_interval)),
    )
    if seed_fade_monitor.observe_only:
        print(
            "[SEED-FADE] observation-only mode: fade is detected and reported "
            "but lambda_eff_count/lambda_opacity_entropy are never throttled.",
            flush=True,
        )

    import signal as _signal
    # Lazy import (mirrors train.py's lazy `from train_streaming import
    # streaming_training` in the other direction) -- avoids any import-order
    # coupling between the two modules while sharing the single
    # implementation of the block/marker/shutdown contract instead of
    # duplicating it here.
    from train import _finalize_output_staging
    _orig_sigint = _signal.getsignal(_signal.SIGINT)
    _orig_sigterm = _signal.getsignal(_signal.SIGTERM)
    # Populated/refreshed once per training iteration below (loop body) so the
    # signal handler always has a recent, consistent snapshot of "what to save"
    # without needing nonlocal/global rebinding tricks mid-signal.
    _emergency_state: dict = {}

    def _emergency_finalize_output_staging():
        _streaming_emergency_finalize_output_staging(args)

    def _emergency_checkpoint_and_exit(sig, frame):
        # SIGTERM is what a preempted k8s pod actually receives (KAI Scheduler
        # eviction) before terminationGracePeriodSeconds expires and SIGKILL
        # follows — this must be fast and SYNCHRONOUS (not just enqueued to
        # AsyncSaveWorker, which could still be mid-flush or queue-full) so a
        # real checkpoint lands on durable storage inside the grace window.
        sig_name = _signal.Signals(sig).name
        st = _emergency_state
        if not st:
            print(f"\n[streaming] {sig_name} received before first optimization step — "
                  "nothing to checkpoint, exiting.", flush=True)
            _emergency_finalize_output_staging()
            os._exit(1)
        print(f"\n[streaming] {sig_name} received at step={st.get('iteration')} — "
              "writing emergency checkpoint synchronously...", flush=True)
        try:
            blob = st["map_backend"].checkpoint_state()
            state = blob.payload
            meta = {
                "checkpoint_reason": f"signal_{sig_name}",
                "n_frames_ingested": int(st["n_frames_ingested"]),
                "last_ingest_iter": int(st["last_ingest_iter"]),
                "source_idx": int(st["source_idx"]),
                "map_version": int(blob.map_version),
                "topology_version": int(blob.topology_version),
                "per_source_supervision_counts": {
                    int(source_index): max(0, int(count))
                    for source_index, count in st["per_source_supervision_counts"].items()
                },
            }
            if isinstance(st.get("coverage_ledger"), dict):
                meta["coverage_ledger"] = dict(st["coverage_ledger"])
            # Issue #290: the incident this exists for (s8b_ctrl, SIGTERM'd by
            # the GPU-memory-danger guard at iter 7712) went through THIS
            # path, not the periodic one — both must carry the state or the
            # requeued attempt still grades itself against a reset running
            # max. Cheap (plain scalars), and guarded so a failure here can
            # never cost the model checkpoint inside the grace window.
            try:
                meta["seed_fade"] = seed_fade_monitor.state_dict()
            except Exception as _sf_exc:
                print(f"[streaming] WARNING: seed-fade state not checkpointed: "
                      f"{_sf_exc}", flush=True)
            chk = os.path.join(st["model_path"], f"chkpnt{st['iteration']}.pth")
            tmp = chk + ".tmp"
            torch.save((state, st["iteration"], meta), tmp)
            os.replace(tmp, chk)
            print(f"[streaming] emergency checkpoint written: {chk}", flush=True)
        except Exception as e:
            print(f"[streaming] emergency checkpoint FAILED: {e}", flush=True)
        try:
            save_worker.shutdown()
        except Exception:
            pass
        _emergency_finalize_output_staging()
        os._exit(143 if sig == _signal.SIGTERM else 130)

    def _graceful_shutdown(sig, frame):
        print("\n[streaming] Interrupted — flushing save worker...", flush=True)
        save_worker.shutdown()
        _emergency_finalize_output_staging()
        _signal.signal(_signal.SIGINT, _orig_sigint)
        raise KeyboardInterrupt

    _signal.signal(_signal.SIGINT, _graceful_shutdown)
    _signal.signal(_signal.SIGTERM, _emergency_checkpoint_and_exit)

    # --- Build Gaussian model ----------------------------------------------
    model_layout = getattr(args, "model_layout", "gsplat").lower()
    model_cls = GsplatGaussianModel if model_layout == "gsplat" else GaussianModel
    gaussians = model_cls(dataset.sh_degree)

    # `gaussians` identity is fixed for the rest of this run (never
    # reassigned -- .restore()/rolling-seed/submap-stitch all mutate this
    # same object in place). Construct the run's single GaussianMapBackend
    # here so checkpointing/viewer-read/mutation call sites below can share
    # it rather than each deriving their own (see
    # MappingCoordinator.backend_for's docstring for why one shared backend
    # instance, not several independently-cached ones, matters for
    # map_version/topology_version correctness). Wave 2 backend-ownership
    # pass, 2026-08-06.
    map_backend = _coordinator.backend_for(gaussians)

    # --- Hard resolution-scaling gate (2026-07-27) --------------------------
    # ResolutionController (eligsir/mapping/resolution_controller.py) throttles
    # resolution reactively based on measured throughput (it_v/map_fps) -- disabling
    # it (--streaming_multires_it_v_floor/map_fps_floor 0, needed to fix the
    # unbounded-quality-loss bug it otherwise causes) removes that safety valve
    # entirely, which OOMs on genuinely high-resolution sources (confirmed:
    # ScanNet++ iPhone-LiDAR at 1920x1440, ~2.76MP, ~9x a typical 640x480 scene).
    # This is a separate, unconditional PRE-CHECK against the source's native
    # resolution (from intrinsics.json, before any frame is loaded) -- deterministic,
    # not reactive, so it can't be starved out by a lucky-fast start the way the
    # throughput-based controller can. Only auto-applies when the user hasn't
    # already set --streaming_resolution explicitly (default sentinel 1).
    #
    # Issue #595: this block's DECISION is unchanged (same predicate, same
    # divisor arithmetic, same log line, same defaults). What is new is that
    # every input and output it consumed is now captured into locals and
    # recorded in report.json["resolution"] below -- previously the fact that a
    # run trained at half linear resolution survived only in train.log prose,
    # so two runs launched from one spec on machines with different memory
    # limits could differ 4x in pixel budget and look identical in every
    # machine-readable artifact.
    _res_native_w, _res_native_h, _res_native_err = read_native_resolution(dataset.source_path)
    _res_requested_scale = int(getattr(args, "streaming_resolution", 1))
    _res_gate_disabled = bool(getattr(args, "streaming_auto_downscale_disabled", False))
    # Bare attribute access, not a getattr fallback: this is an unconditional
    # StreamingParams default, always present on real parsed args, and a
    # hand-copied literal here would be exactly the #363/#368 stale-default
    # trap (tests/test_getattr_fallback_freshness.py enforces this).
    _res_max_px_arg = int(args.streaming_auto_downscale_max_pixels or 0)
    _res_gate_evaluated = False
    _res_gate_fired = False
    if _res_requested_scale == 1 and not _res_gate_disabled:
        _max_px = int(args.streaming_auto_downscale_max_pixels or 0)
        if _max_px > 0:
            _intr_path = os.path.join(dataset.source_path, "intrinsics.json")
            if os.path.isfile(_intr_path):
                if _res_native_err is not None:
                    print(f"[streaming] auto-downscale gate: failed to read {_intr_path}: {_res_native_err}", flush=True)
                else:
                    _w, _h = int(_res_native_w or 0), int(_res_native_h or 0)
                    _res_gate_evaluated = _w > 0 and _h > 0
                    if _w > 0 and _h > 0 and _w * _h > _max_px:
                        _divisor = 1
                        while (_w // (_divisor + 1)) * (_h // (_divisor + 1)) > _max_px:
                            _divisor += 1
                        _divisor += 1
                        args.streaming_resolution = _divisor
                        _res_gate_fired = True
                        print(
                            f"[streaming] auto-downscale gate: native {_w}x{_h} "
                            f"({_w * _h} px) exceeds --streaming_auto_downscale_max_pixels="
                            f"{_max_px}; forcing --streaming_resolution={_divisor} "
                            f"({_w // _divisor}x{_h // _divisor}) to bound render/backward "
                            f"memory regardless of ResolutionController's throughput-based "
                            f"tiering.",
                            flush=True,
                        )

    # --- Resolve source-sampling ownership (issue #834) ----------------------
    # streaming_frame_stride (applied INSIDE native dataset-reader
    # construction, below) and streaming_source_stride (applied by the
    # generic StridedFrameSource wrapper, also below) are two
    # independently-active source-rate knobs that used to silently MULTIPLY
    # whenever both were set to different non-1 values (each applied in full,
    # composing to frame_stride * source_stride instead of the requested
    # single stride). Source sampling must be requested once, resolved once,
    # applied once: resolve which single owner applies the stride HERE,
    # before the frame source is even constructed, so a conflicting request
    # (--streaming_source_sampling_mode single_owner, the default, with both
    # strides set to different non-1 values) fails LOUDLY at startup -- an
    # actionable error the operator can fix -- rather than training to
    # completion on a silently over-decimated trajectory and only surfacing
    # the problem later when building report.json.
    _source_sampling_mode = str(
        getattr(args, "streaming_source_sampling_mode", "single_owner") or "single_owner"
    )
    _sampling_application = resolve_source_sampling_application(
        getattr(args, "streaming_frame_stride", 1),
        getattr(args, "streaming_source_stride", 1),
        sampling_mode=_source_sampling_mode,
    )
    _sampling_policy = _sampling_application["policy"]
    _sampling_owner = _sampling_policy["owner"]
    _legacy_composition_mode = _sampling_application["legacy_composition_mode"]
    _requested_frame_stride = _sampling_application["requested_frame_stride"]
    _requested_source_stride = _sampling_application["requested_source_stride"]

    # Apply the resolved policy by overriding the args attribute the native
    # readers actually read (registry.make_frame_source's
    # getattr(args, "streaming_frame_stride", 1) call sites, and
    # ROS2LiveStreamingFrameSource's own read of the same attribute) so
    # construction below applies EXACTLY the stride the policy assigned to
    # the native owner -- never both the native reader's own stride AND an
    # independent generic wrap. Restored to the originally-requested value
    # immediately after construction (cfg_args was already written earlier in
    # this function, but restoring keeps `args` itself honest for any later
    # reader, e.g. a diagnostic script re-inspecting this object).
    args.streaming_frame_stride = _sampling_application["native_stride_to_apply"]
    _apply_source_stride = _sampling_application["generic_stride_to_apply"]

    # --- Build frame source and streaming scene ----------------------------
    frame_source = make_frame_source(dataset.source_path, args)
    args.streaming_frame_stride = _requested_frame_stride  # restore requested value
    if len(frame_source) == 0:
        raise RuntimeError("[streaming] Frame source is empty — check dataset path.")

    # --- Issue #641: Replica RGB source fidelity marker -----
    # Track that Replica synthetic RGB source is lossless (no JPEG recompression).
    # The marker is only emitted for runs that actually use the Replica source;
    # it is absent/null for non-Replica runs rather than defaulting incorrectly.
    from eligsir.io.sources.datasets.replica import ReplicaFrameSource
    _replica_rgb_report = None
    if isinstance(frame_source, ReplicaFrameSource):
        _replica_rgb_report = {
            "rgb_source": "replica_synthetic",
            "rgb_lossless": True,
            "reason": "Replica synthetic frames generated with no JPEG encoding (issue #641)",
        }

    # Apply explicit source stride BEFORE the scheduler so it sees the correct
    # frame count. --streaming_source_stride N pre-strides the frame source
    # independently of enforcement — used to match the holdout set of runs that
    # rely on enforcement-driven striding (e.g. AB1/AB2 vs A1/A2).
    #
    # Apply the resolved generic-owner stride BEFORE the scheduler so it sees
    # the correct frame count. --streaming_source_stride N pre-strides the
    # frame source independently of enforcement — used to match the holdout
    # set of runs that rely on enforcement-driven striding (e.g. AB1/AB2 vs
    # A1/A2). Issue #834: _apply_source_stride is > 1 here only when the
    # resolved policy actually assigned the generic wrapper as the owner (or
    # under the explicit legacy_compose escape hatch) -- never merely because
    # --streaming_source_stride was requested, since a native-owned or
    # no-sampling resolution already zeroed it out above.
    #
    # NOTE (2026-08-06): this used to be applied THREE times in sequence here
    # (each a verbatim-duplicated inline class), silently composing to
    # stride**3 instead of stride for any run passing --streaming_source_stride.
    # Consolidated to the single module-level StridedFrameSource, applied once.
    #
    # Issue #805: Record both frame_stride (native reader) and source_stride
    # (generic post-hoc) for report.json, with their composition and frame-count
    # provenance. Hoisted unconditionally so values are available even when
    # source_stride == 1.
    _frames_before_source_stride = len(frame_source)  # Captured before wrapping
    _frames_source_provenance = (
        "placeholder" if len(frame_source) >= 999999 else "pre_stride"
    )

    if _apply_source_stride > 1:
        frame_source = StridedFrameSource(frame_source, _apply_source_stride)
        print(
            f"[streaming] source-stride: striding full trajectory "
            f"({_frames_before_source_stride} frames) at stride={_apply_source_stride} → {len(frame_source)} frames",
            flush=True,
        )

    _frames_after_source_stride = len(frame_source)

    # Issue #805/#834: emit the composed-decimation notice only when the
    # resolved policy actually applied BOTH owners (the explicit
    # legacy_compose escape hatch) — the canonical single_owner path never
    # reaches this branch, since resolve_source_sampling_policy already
    # refused a conflicting both-non-1-and-unequal request above.
    if _sampling_owner == "both":
        from eligsir.mapping.scheduler import COMPOSED_DECIMATION_NOTICE
        print(
            COMPOSED_DECIMATION_NOTICE.format(
                frame_stride=_requested_frame_stride,
                source_stride=_requested_source_stride,
                effective_stride=_sampling_policy["effective_stride"],
            ),
            flush=True,
        )

    # --- Phase 7: MappingScheduler (replaces inline full-coverage enforcement) -
    # Real trajectory-length classification (2026-07-04 fix): the frame source
    # for EVERY ros2://live job (which includes every bag-replay job in this
    # codebase — bag replay is implemented by publishing over ROS2 topics and
    # consuming through this same live path) reports len(frame_source) as
    # whatever --streaming_max_frames was set to (universally 999999 as an
    # "unlimited" convention across every launch script) — never the real
    # trajectory length. Feeding that placeholder into the coverage-enforcement
    # math below made it a silent no-op (confirmed: floor3/fr3 jobs landed at
    # 16-29% real coverage despite enforce_full_coverage=True). Resolve the
    # REAL frame count from the scene registry instead, using the same
    # SCENE_SOURCE-first resolution already used for partition-eval denominators
    # (train_streaming.py's partition telemetry, ~line 6125) — do not look up by
    # source_path alone, since for a ros2://live job that string is literally
    # "ros2://live" and would never match eval/scenes.json's path-keyed entries.
    #
    # Classification, not just a fallback: the registry resolving to a real
    # count means this is a FINITE, KNOWABLE bag-replay — enforce coverage
    # against that real n_frames. The registry NOT resolving (unregistered
    # scene, no SCENE_SOURCE set) means this is a GENUINELY LIVE camera with
    # truly unknown length — force enforce_full_coverage off unconditionally
    # regardless of the CLI flag, since n_frames has no real meaning there and
    # the coverage math would just be confidently wrong again.
    from eligsir.mapping.scheduler import MappingScheduler
    from eligsir.utils.partition_eval import registry_n_source_frames, check_coverage_budget_feasible
    _scene_src_for_coverage = os.environ.get("SCENE_SOURCE", "") or getattr(args, "source_path", "")
    _registry_n_frames = registry_n_source_frames(_scene_src_for_coverage)
    _coverage_source_effective_stride = (
        max(1, int(_sampling_application["native_stride_to_apply"]))
        * max(1, int(_sampling_application["generic_stride_to_apply"]))
    )
    _managed_staged_source_frames, _managed_staged_provenance = (
        resolve_managed_staged_source_denominator(
            scientific_mode=bool(getattr(args, "scientific_mode", False)),
            researchflow_run_id=os.environ.get("RESEARCHFLOW_RUN_ID"),
            researchflow_plan_sha256=os.environ.get("RESEARCHFLOW_PLAN_SHA256"),
            staged_source_frames=os.environ.get(MANAGED_STAGED_SOURCE_FRAMES_ENV),
            source_identity=_scene_src_for_coverage,
        )
    )
    if _managed_staged_source_frames is not None:
        _coverage_n_frames = _managed_staged_source_frames
        _coverage_resolved_source_frames = _managed_staged_source_frames
        _coverage_denominator_provenance = _managed_staged_provenance
        _is_finite_replay = True
    else:
        _coverage_resolved_source_frames, _coverage_denominator_provenance = (
            resolve_finite_source_denominator(
                raw_registry_source_frames=_registry_n_frames,
                frame_source_length=len(frame_source),
                effective_stride=_coverage_source_effective_stride,
            )
        )
    if (
        _managed_staged_source_frames is None
        and _registry_n_frames
        and _coverage_resolved_source_frames is not None
    ):
        _coverage_n_frames = _coverage_resolved_source_frames
        _is_finite_replay = True
    elif _managed_staged_source_frames is None:
        _coverage_n_frames = len(frame_source)
        _coverage_resolved_source_frames = None
        _is_finite_replay = False
        print(
            f"[streaming] scene '{_scene_src_for_coverage or '(unset)'}' not in "
            "eval/scenes.json registry — treating as a genuinely live/unbounded "
            "source; full-coverage enforcement disabled (frame count has no "
            "fixed meaning here).",
            flush=True,
        )
    check_coverage_budget_feasible(
        is_finite_replay=_is_finite_replay,
        enforce_full_coverage=getattr(args, "streaming_enforce_full_coverage", True),
        n_frames=_coverage_n_frames,
        iterations=opt.iterations,
        scene_source=_scene_src_for_coverage,
    )
    # --- Issue #1044: coverage must plan against a budget that can exist -----
    # `opt.iterations` is the ONLINE_UNBOUNDED_ITERATIONS sentinel (2^31-1) for
    # an --online_mode run, so the coverage clamp
    # `min(requested, floor(budget / n_frames))` could never bind and spf stayed
    # at the raw request. Measured consequence on fr3: spf=150 governed release,
    # the run needed 379,500 iterations to see all 2525 frames, and reached
    # frame 130 (5.1%) -- which is the whole reason hybrid_keyframe "plateaued"
    # at ~20 admitted views. Resolve a budget the operator can actually be held
    # to; `None` means genuinely unbounded, in which case no coverage claim is
    # made at all (rather than the old "All frames covered.").
    _coverage_planning_budget, _coverage_budget_source = resolve_coverage_planning_budget(
        loop_iterations=int(opt.iterations),
        streaming_max_iterations=int(getattr(args, "streaming_max_iterations", 0) or 0),
        # `args.iterations` still holds the operator's own value:
        # apply_online_schedule_overrides writes the sentinel to `opt`, a
        # SEPARATE GroupParams object (see its call site above).
        # No getattr fallback (issue #368): `iterations` is an unconditional
        # OptimizationParams default, always present on the real parsed args
        # (the bare `args.streaming_global_replay_ratio` access below relies on
        # the same property). A `0` fallback would also drift from the real
        # default of 30000.
        operator_iterations=int(args.iterations or 0),
        online_mode=bool(getattr(args, "online_mode", False)),
    )
    _mapping_scheduler = MappingScheduler(
        steps_per_frame=max(1, getattr(args, "streaming_steps_per_frame", 150)),
        n_frames=_coverage_n_frames,
        iterations=(
            int(_coverage_planning_budget) if _coverage_planning_budget is not None
            else 0  # 0 disables the arithmetic: no budget => no coverage claim
        ),
        enforce_full_coverage=(_is_finite_replay and getattr(args, "streaming_enforce_full_coverage", True)),
        keyframe_window=max(1, getattr(args, "streaming_keyframe_window", 120)),
        # No getattr fallback: streaming_global_replay_ratio is an
        # unconditional StreamingParams default, always present on the real
        # parsed args (issue #368 -- old getattr(..., 0.1) fallback drifted
        # from the real default of 0.3; removed rather than re-stated).
        global_replay_ratio=float(args.streaming_global_replay_ratio),
        tracking_min_conf_persistent=float(getattr(args, "tracking_min_conf_persistent", 0.7)),
        tracking_min_conf_provisional=float(getattr(args, "tracking_min_conf_provisional", 0.4)),
        min_steps_per_frame=int(getattr(args, "streaming_min_steps_per_frame", 50)),
    )
    # Second, independent consumer of the same tracking-status signal
    # KeyframeAdmitter uses (see tracking_confidence.py) -- feeds
    # TrackingResult.confidence at the dataset_pose call site below instead
    # of the admission quality_weight, so it needs its own causal EMA state
    # (a shared instance would mix two different sampling populations: every
    # arriving frame here vs. only admission-scored candidates there).
    from eligsir.mapping.scheduler.tracking_confidence import (
        RunningTrackingStats as _MSRunningTrackingStats,
        tracking_confidence_weight as _ms_tracking_confidence_weight,
    )
    _ms_tracking_running_stats = _MSRunningTrackingStats(
        alpha=float(getattr(args, "mapping_scheduler_tracking_confidence_ema_alpha", 0.05)),
        warmup=int(getattr(args, "mapping_scheduler_tracking_confidence_warmup", 20)),
    )
    # Milestone 2: wrap MappingScheduler in an OnlineController when enabled.
    # When disabled, behaviour is bit-for-bit identical to the pre-M2 path
    # (the controller's tick() simply uses BALANCED mode with adaptive off).
    _online_controller = None
    if bool(getattr(args, "streaming_online_controller", False)):
        from eligsir.mapping.scheduler import OnlineController, OnlineConfig
        _online_controller = OnlineController(
            scheduler=_mapping_scheduler,
            config=OnlineConfig(
                map_age_target_s=float(getattr(args, "streaming_map_age_target_s", 0.5)),
                map_age_catch_up_enter_s=float(getattr(args, "streaming_map_age_catch_up_enter_s", 0.8)),
                map_age_catch_up_exit_s=float(getattr(args, "streaming_map_age_catch_up_exit_s", 0.3)),
                map_age_track_only_enter_s=float(getattr(args, "streaming_map_age_track_only_enter_s", 1.5)),
                map_age_track_only_exit_s=float(getattr(args, "streaming_map_age_track_only_exit_s", 0.5)),
                min_steps=int(getattr(args, "streaming_online_min_steps", 1)),
                max_steps=int(getattr(args, "streaming_online_max_steps", 50)),
                iter_dt_ema_beta=float(getattr(args, "streaming_online_iter_dt_ema_beta", 0.5)),
                adaptive_steps=True,
                # Milestone 3: wire the new pause threshold CLI flag into the
                # controller config so ``pause=True`` actually fires on the
                # operator-supplied idle window. 0 (legacy default) keeps the
                # twice-target-period heuristic.
                pause_idle_threshold_s=float(args.streaming_pause_idle_threshold_s),
                # D3: velocity-adaptive refinement (off by default).
                velocity_adaptive_steps=bool(getattr(args, "streaming_velocity_adaptive_steps", False)),
                velocity_stall_mps=float(getattr(args, "streaming_velocity_stall_mps", 0.02)),
                velocity_fast_mps=float(getattr(args, "streaming_velocity_fast_mps", 0.30)),
                velocity_step_gain=float(getattr(args, "streaming_velocity_step_gain", 1.5)),
            ),
            target_input_fps=float(getattr(args, "streaming_input_fps", 30.0)),
            budget_per_iter_s=float(getattr(args, "streaming_online_budget_per_iter_s", 0.05)),
            insertion_cap=int(getattr(args, "streaming_online_insertion_cap", 1500)),
            sampling_mode=str(getattr(args, "streaming_sampling_mode", "legacy")),
            rviz_publish_interval_iters=int(getattr(args, "streaming_online_rviz_publish_interval", 10)),
        )
    # Milestone 3: BurstBudget guards the drain loop against bursts and
    # drops redundant non-keyframes when backlog grows.
    _burst_budget = None
    if _online_controller is not None and bool(getattr(args, "streaming_online_controller", False)):
        from eligsir.mapping.scheduler import BurstBudget, BurstConfig
        _burst_budget = BurstBudget(
            controller=_online_controller,
            config=BurstConfig(
                max_per_iter=int(getattr(args, "streaming_ingestion_max_per_iter", 2)),
                drop_nonkf_backlog=int(getattr(args, "streaming_drop_nonkf_backlog", 4)),
                drop_nonkf_in_catch_up=bool(getattr(args, "streaming_drop_nonkf_in_catch_up", True)),
            ),
        )
    # -------------------------------------------------------------------
    # Issue #601: the ONE bounded backpressure owner.
    #
    # When enabled it is the single execution authority for skip/coalesce
    # decisions: it bounds pending mapping work by skipping/coalescing
    # redundant observations BEFORE expensive RGB/depth materialization,
    # rather than deferring an ever-staler queue to preserve raw-frame
    # coverage. Default OFF => never constructed, drain loop unchanged.
    #
    # Single-owner enforcement (#601's "after this lands" requirements):
    #   * BurstBudget's overlapping DROP execution is neutered here (its
    #     max_per_iter cadence cap survives as a pure signal);
    #   * the drain-loop gate below is the terminal authority -- a frame
    #     the owner skipped is never re-offered, and never reaches camera
    #     materialization, replay residency, multires cache construction
    #     or depth insertion.
    # -------------------------------------------------------------------
    _backpressure_owner = None
    _backpressure_cfg = None
    if bool(getattr(args, "streaming_backpressure", False)) or bool(
        getattr(args, "streaming_backpressure_force_all_stress", False)
    ):
        from eligsir.mapping.scheduler import (
            BackpressureOwner,
            backpressure_config_from_args,
            burst_config_as_signal_provider,
        )
        from dataclasses import replace as _bp_replace

        _backpressure_cfg = backpressure_config_from_args(args)
        if _backpressure_cfg.resolution_mode != "causal":
            print(
                "[backpressure] WARNING: resolution_mode="
                f"{_backpressure_cfg.resolution_mode!r} is not wired into the drain "
                "loop (it needs a source that can materialize a frame after its "
                "cursor advanced). Falling back to 'causal' for this run.",
                flush=True,
            )
            _backpressure_cfg = _bp_replace(
                _backpressure_cfg, resolution_mode="causal"
            )
        _backpressure_owner = BackpressureOwner(_backpressure_cfg)
        if _backpressure_cfg.force_all_stress:
            print(
                "[backpressure] *** FORCE-ALL STRESS MODE *** the bounded owner is "
                "bypassed; mapping backlog is intentionally unbounded. This is a "
                "diagnostic assimilation-overload condition, NOT a production run.",
                flush=True,
            )
        else:
            print(
                "[backpressure] enabled: mode=causal "
                f"pending_max_count={_backpressure_cfg.pending_max_count} "
                "pending_max_age_control_ticks="
                f"{_backpressure_cfg.pending_max_age_control_ticks} "
                f"max_admissions_per_window={_backpressure_cfg.max_admissions_per_window}"
                f"/{_backpressure_cfg.admission_window_control_ticks}control_ticks "
                f"frontier_quota={_backpressure_cfg.frontier_quota}",
                flush=True,
            )
            if bool(getattr(args, "streaming_enforce_full_coverage", True)):
                print(
                    "[backpressure] NOTE: --streaming_enforce_full_coverage is on. "
                    "Skipped observations still advance pose/source/history "
                    "metadata (observe=True) and still count as source-released, "
                    "but do NOT become expensive mapping work. Read "
                    "report.json[\"backpressure\"][\"n_admitted\"] as the "
                    "expensive-mapping subset, never as the coverage numerator.",
                    flush=True,
                )
        if _burst_budget is not None and not _backpressure_cfg.force_all_stress:
            # BurstBudget becomes a thin signal provider: no second dropper.
            #
            # Deliberately NOT done under force_all_stress: that mode exists to
            # reproduce the historical hard-all behaviour exactly, and silently
            # disabling BurstBudget's drops alongside it would make the stress
            # arm differ from the baseline it is supposed to reproduce.
            _burst_budget._config = burst_config_as_signal_provider(
                _burst_budget._config
            )
            print(
                "[backpressure] BurstBudget drop policy disabled (signal-provider "
                "mode): the backpressure owner is the single drop/coalesce owner.",
                flush=True,
            )
    # Issue #596: two "full-coverage arithmetic (informational only) ... Not
    # enforced" messages used to be printed HERE, ~495 lines before
    # _use_enforced_spf / _frame_scheduler_spf are computed — i.e. before the
    # fact they asserted even existed. They were true when written (2026-07-04,
    # when the unified rolling-budget termination rule demoted effective_spf to
    # a statistic) and were falsified by the 2026-07-26 re-enablement for
    # non---online_mode runs, which hands effective_spf straight to
    # FrameScheduler. A real 2026-08-14 run (issue290xsv2_sb20_obs) printed
    # "steps_per_frame 150 would need to shrink to 2 ... Not enforced" and then
    # trained with exactly that 2 governing frame release.
    #
    # Both messages are now folded into ONE authoritative summary emitted after
    # every override is resolved — see the [spf] block just below the
    # FrameScheduler construction. Nothing about which value wins changed.
    #
    # Still true, and still the reason no frame-source striding/capping happens
    # here: n_frames_capped is DIAGNOSTIC ONLY, and args.streaming_steps_per_frame
    # is never mutated. "Steps per frame" also remains incoherent as a CONTROLLING
    # parameter for a source whose total length may not be fixed (live camera) —
    # which is precisely why the summary below reports governs_release explicitly
    # instead of leaving the reader to infer it.
    _orig_spf = max(1, getattr(args, "streaming_steps_per_frame", 150))

    from eligsir.core.online_state import OnlineClock, DirtyRegionQueue
    online_clock = OnlineClock()
    dirty_region_queue = DirtyRegionQueue()

    streaming_scene = StreamingScene(args, gaussians, frame_source)
    # Same #729 class as `_git_provenance_from_args` above: `wall_time_s`
    # passed into every `_streaming_report_kwargs()` call site is derived
    # from `_train_t0`, which used to be bound much later in this function
    # (only reached after the pre-training report's call site, itself only
    # reachable behind --streaming_report_pre_training). Bind it here,
    # unconditionally and as close as possible to the historical "training
    # setup is done" point (right after scene construction) rather than at
    # the top of the function, so `wall_time_s` keeps meaning what it always
    # meant instead of also swallowing dataset load and scene construction.
    _train_t0 = time.perf_counter()  # for live mapping-FPS / iters-per-view stats

    # --- Issue #595: resolution provenance (records only, decides nothing) ---
    # Built HERE, not at the gate above, for two reasons: (a) the frame source
    # exists, so a RealityScan source -- which has no intrinsics.json and runs
    # its own equivalent gate internally -- can report its real native size and
    # applied divisor instead of being recorded as "unknown"; (b) StreamingScene
    # .__init__ has by now resolved `args.resolution`, the SECOND, independent
    # camera-level downscale that composes multiplicatively with the frame
    # source's divisor. Recording only the divisor would let report.json claim
    # 960x720 for a run that actually trained at 480x360 under `-r 2`.
    _res_native_source = (
        NATIVE_SOURCE_INTRINSICS_JSON if (_res_native_w and _res_native_h) else NATIVE_SOURCE_UNKNOWN
    )
    _src_native = getattr(frame_source, "native_resolution", None)
    if not (_res_native_w and _res_native_h) and _src_native:
        _res_native_w, _res_native_h = int(_src_native[0]), int(_src_native[1])
        _res_native_source = NATIVE_SOURCE_FRAME_SOURCE
    # A frame source that ran its own auto-downscale gate (RealityScan) applies
    # the divisor internally and never touches args.streaming_resolution.
    _src_divisor = int(getattr(frame_source, "applied_divisor", 0) or 0)
    _res_resolved_scale = int(getattr(args, "streaming_resolution", 1))
    if _src_divisor > 1 and _res_resolved_scale == 1:
        _res_resolved_scale = _src_divisor
        _res_gate_evaluated = True
        _res_gate_fired = True
    _resolution_provenance = build_resolution_provenance(
        native_width=_res_native_w,
        native_height=_res_native_h,
        native_source=_res_native_source,
        native_read_error=_res_native_err,
        requested_scale=_res_requested_scale,
        hard_max_pixels=_res_max_px_arg,
        hard_gate_disabled=_res_gate_disabled,
        hard_gate_evaluated=_res_gate_evaluated,
        hard_gate_fired=_res_gate_fired,
        resolved_scale=_res_resolved_scale,
        # Bare access: an unconditional ModelParams default (-1), already
        # resolved/pinned by StreamingScene.__init__ above. `resolution_scale`
        # is NOT an argparse flag at all -- it is loadCam's per-call argument,
        # 1.0 for every streaming camera today -- so it is recorded as the
        # constant it is rather than read from args.
        camera_resolution_arg=args.resolution,
        camera_resolution_scale=1.0,
    )
    print(_resolution_provenance.format_log(), flush=True)

    # --- Payload budget policy resolution (Issue #TBD) -----------------------
    # Resolve the admitted-payload frame count and hot-tier byte cap based on
    # native resolution and device memory (if not explicitly passed).
    from eligsir.runtime.config_boundary import resolve_admitted_payload_bounds
    _payload_budget_frames, _payload_budget_hot_bytes, _payload_budget_metadata = (
        resolve_admitted_payload_bounds(
            args,
            native_width=_res_native_w,
            native_height=_res_native_h,
        )
    )
    _policy_str = _payload_budget_metadata.get("policy", "unknown")
    _frames_source = _payload_budget_metadata.get("source", "unknown")
    _hot_source = _payload_budget_metadata.get("hot_tier_source", "unknown")
    _device_mem_str = ""
    if _payload_budget_metadata.get("device_total_memory"):
        _dev_mem_gb = _payload_budget_metadata["device_total_memory"] / 1e9
        _hot_bytes_gb = _payload_budget_hot_bytes / 1e9
        _device_mem_str = f" (20% of {_dev_mem_gb:.1f} GB)"
    if _res_native_w and _res_native_h:
        _native_str = f"{_res_native_w}x{_res_native_h}"
    else:
        _native_str = "unknown"
    print(
        f"[payload-budget] policy={_policy_str} native={_native_str} "
        f"admitted_payload_max_frames={_payload_budget_frames} (from 512@640x480, {_frames_source}) "
        f"hot_tier_max_bytes={_payload_budget_hot_bytes} ({_hot_source}){_device_mem_str}",
        flush=True,
    )
    # Store resolved values in args for later use (e.g., summary JSON)
    args._resolved_payload_budget_policy = _policy_str
    args._resolved_payload_budget_native_w = _res_native_w
    args._resolved_payload_budget_native_h = _res_native_h
    args._resolved_payload_budget_frames = _payload_budget_frames
    args._resolved_payload_budget_hot_bytes = _payload_budget_hot_bytes

    _seed_points = getattr(frame_source, "seed_points", None)
    if _seed_points is not None:
        streaming_scene.set_external_seed(_seed_points, getattr(frame_source, "seed_colors", None))

    phase_timer = PhaseTimer(args.model_path, tb_writer)
    phase_timer.start("bootstrap")

    # --- SLAM backend (Phase 1.5) -------------------------------------------
    # Hoisted above the pose-aware bootstrap below: that bootstrap branch
    # references _pose_source_name / _slam_backend, so the backend must be
    # resolved first (regression fix — was previously defined further down).
    # #1075: the SLAMBackend registry and its four adapter-import side effects
    # are gone. Selection is an explicit factory; the frontend is bound to the
    # canonical TrackingProvider contract by the providers below.
    from eligsir.tracking.factory import build_tracking_frontend

    _pose_source_name = normalize_pose_source(getattr(args, "pose_source", "dataset_pose"))
    # Legacy alias: orbbec_pose_source → pose_source
    if is_dataset_pose_source(_pose_source_name) and args.orbbec_pose_source == "open3d_odometry_live":
        _pose_source_name = "open3d_odometry_live"
    _backend_name = _pose_source_name

    # No frontend today takes constructor kwargs from the CLI. The former
    # ros_slam kwarg block (pose_topic/msg_type/staleness/...) is gone with the
    # backend itself: #1075 moved ROS pose/odom/TF subscription and conversion
    # into eligsir-ros2, which stamps an already-converted
    # TrackingObservation onto each SensorPacket. The --slam_pose_* CLI flags
    # remain declared but are inert for the core trainer; a ros2:// source
    # resolves to dataset_pose and consumes the pose its frame source joined.
    _backend_kwargs: dict = {}

    # The in-process tracking frontend for this run, selected explicitly rather
    # than through the retired registry. `dataset_pose` (the default, and what
    # every ros2:// source normalizes to) yields a FramePosePassthroughTracker,
    # the exact behavioural equivalent of the removed PrecomputedBackend.
    #
    # DEFERRED (#1075): this loop still drives the frontend directly via the
    # inline initialize()/track() call sites below, rather than through a
    # TrackingProvider and MappingSession. Constructing a provider here that
    # nothing consumed would be decoration, so it is deliberately not built --
    # the canonical composition path is exercised by
    # `eligsir.runtime.replay` / `eligsir-session` and by the ROS
    # package. Collapsing this 10.9k-line loop onto MappingSession is tracked
    # as the remainder of #1075; see the architecture note.
    _slam_backend = build_tracking_frontend(_backend_name, **_backend_kwargs)
    _mapping_scheduler.log_startup()

    # Deferred-admission SLAM worker integration (W3-15). Default off (see
    # --streaming_slam_worker_deferred_admission's docstring in
    # arguments/__init__.py). Only meaningful for the in-process-backend
    # branch below (main loop's else: not is_dataset_pose_source(...)) --
    # a dataset-baked-pose or ros2://live run never calls _slam_backend
    # .track() in the loop at all, so this stays None (a documented no-op)
    # for those runs regardless of the flag.
    _deferred_slam_tracker = None
    if bool(getattr(args, "streaming_slam_worker_deferred_admission", False)):
        if is_dataset_pose_source(_pose_source_name):
            print(
                "[streaming][deferred_slam] --streaming_slam_worker_deferred_admission "
                f"is set but pose_source='{_pose_source_name}' resolves to a "
                "frame-embedded pose (dataset replay or ros2://live, which joins "
                "pose/odom/TF at the frame-source level) -- there is no "
                "_slam_backend.track() call in the main loop for this run to "
                "defer, so this flag is a no-op here.",
                flush=True,
            )
        else:
            from eligsir.mapping.scheduler.deferred_slam_admission import (
                build_deferred_admission_tracker,
            )
            _deferred_slam_tracker = build_deferred_admission_tracker(
                _pose_source_name,
                _backend_kwargs,
                capacity=int(getattr(args, "streaming_slam_worker_capacity", 4)),
                pending_timeout_s=float(getattr(args, "streaming_slam_worker_pending_timeout_s", 5.0)),
            )
            print(
                "[streaming][deferred_slam] deferred-admission SLAM worker ENABLED "
                f"for backend '{_pose_source_name}' -- "
                "frames are admitted only once their real, asynchronously-computed "
                "pose has arrived; never a stale/predicted pose.",
                flush=True,
            )

    # Phase 1-SLAM: backend initialization state.
    # For the precomputed path the backend is never called; for live backends
    # initialize() is called lazily on the first frame that arrives in the
    # drain loop (see below).
    _slam_backend_initialized: bool = is_dataset_pose_source(_pose_source_name)
    # Per-frame tracking result (None until first frame arrives in the loop)
    _tracking_result = None  # type: ignore[assignment]
    # Bootstrap state machine: tracks whether we are still waiting for the first
    # SLAM pose before bootstrap frames can be admitted.
    _bootstrap_waiting_for_pose: bool = not is_dataset_pose_source(_pose_source_name)

    # Phase 1-SLAM: pose-aware bootstrap.
    # For "precomputed" backends: use the original synchronous path.
    # For live backends (ros_slam, open3d_odometry_live): drain frames until we
    # have enough with valid poses before building the initial point cloud.
    # This handles the SLAM-late-start case where the first RGB-D frames arrive
    # before the first pose is published.
    _pose_source_name = normalize_pose_source(getattr(args, "pose_source", "dataset_pose"))
    if is_dataset_pose_source(_pose_source_name) and args.orbbec_pose_source == "open3d_odometry_live":
        _pose_source_name = "open3d_odometry_live"

    # Multi-resolution frame cache: the observation DB and its bootstrap-time
    # recorder must exist BEFORE bootstrap ingestion (below), not just before
    # the main loop's own _record_fn (defined much later in this function) --
    # bootstrap frames bypass ingest_next_frame entirely, and
    # _effective_keyframe_window deliberately keeps them in the active
    # sampling window early in training, so skipping them here left every
    # bootstrap camera's bind_resolution_tier lookup missing unconditionally
    # (found via real-run validation, 2026-07-26: 100% "record_none" on a
    # short run whose sampling stayed almost entirely within the bootstrap
    # set). `_bootstrap_record_fn` is deliberately minimal (no submap/
    # visibility tracking -- no Gaussians or SubmapManager exist yet at
    # bootstrap time) rather than reusing the full `_record_fn` defined later.
    from eligsir.mapping.observation_db import FrameObservationDB
    _multires_low_divisor = int(getattr(args, "streaming_multires_low_divisor", 8))
    _multires_depthless_rgb_cache = bool(getattr(args, "streaming_multires_depthless_rgb_cache", True))
    _multires_depth_downsample = str(
        getattr(args, "streaming_depth_loss_downsample", "nearest")
    )
    _observation_db = FrameObservationDB(
        device="cuda",
        max_resolution_cache_frames=int(getattr(args, "streaming_resolution_cache_max_frames", 400)),
        resolution_cache_rebuilder=lambda frame, frame_id: build_frame_resolution_cache(
            frame,
            frame_id,
            low_res_divisor=_multires_low_divisor,
            depthless_rgb_cache=_multires_depthless_rgb_cache,
            depth_downsample=_multires_depth_downsample,
        ),
    )
    # Attach so StreamingScene's tiered frame cache (promote_from_cache/
    # demote_partitions/_cache_inactive_frame/_archive_to_icecold) keeps
    # this DB's per-frame buffer_stage in sync with which tier a frame is
    # actually in, instead of it freezing at whatever stage record_fn first
    # recorded (see StreamingScene.set_observation_db).
    streaming_scene.set_observation_db(_observation_db)
    # issue #333 follow-up (2026-08-14): read AFTER StreamingScene.__init__ has
    # applied its `args.resolution = 1` pin, so this reflects what loadCam will
    # really do. A non-identity camera downscale makes EVERY evicted-cache HIGH
    # bind `unresolved_native_res_mismatch` -- say so once, loudly, at startup
    # instead of only per-bind deep in the loop.
    _res_hazard = _describe_startup_resolution_hazard(
        args.resolution,
        getattr(streaming_scene, "resolution_scale", 1.0),
    )
    if _res_hazard:
        print(_res_hazard, flush=True)

    def _bootstrap_record_fn(cam, frame, is_train, is_holdout):
        record_frame_observation(
            _observation_db, cam=cam, visible_uids=None,
            frame_id=frame.index, submap_id=0,
            is_train=is_train, is_holdout=is_holdout,
            frame=frame, low_res_divisor=_multires_low_divisor,
            depthless_rgb_cache=_multires_depthless_rgb_cache,
            depth_downsample=_multires_depth_downsample,
        )

    if is_dataset_pose_source(_pose_source_name):
        # Synchronous bootstrap — all frames have poses embedded.
        n_init = max(1, getattr(args, "streaming_initial_frames", 5))
        streaming_scene.initialize_from_frames(n_init, record_fn=_bootstrap_record_fn)
    else:
        # Live SLAM: drain frames until we have n_init posed frames.
        # While draining we call the backend's track() on each frame so it
        # builds its internal state (keyframe history, covisibility, etc.).
        n_init = max(1, getattr(args, "streaming_initial_frames", 5))
        max_attempts = n_init * int(getattr(args, "bootstrap_max_attempts_multiplier", 10))
        max_attempts = min(max_attempts, len(frame_source))
        allow_random_fallback = bool(getattr(args, "bootstrap_random_fallback", False))

        from eligsir.utils.streaming_frames import frame_has_pose

        posed_frames: list = []
        attempts = 0
        _saved_idx = streaming_scene._source_idx  # remember where bootstrap started

        print(f"[streaming] Waiting for {n_init} posed frames from SLAM backend...", flush=True)

        while attempts < max_attempts and len(posed_frames) < n_init:
            if not streaming_scene.has_next_frame():
                break
            frame = streaming_scene._get_source_frame(streaming_scene._source_idx)
            streaming_scene._source_idx += 1
            attempts += 1

            # Feed frame to backend for tracking / initialisation. SLAMBackend
            # implementations (e.g. Open3DOdometryBackend) are written against
            # the canonical contracts.types.RGBDFrame (.pose/.rgb/.depth/
            # .intrinsics), not the lightweight, lazily-hydrated
            # StreamingRGBDFrame `frame` is here (paths + optional c2w only,
            # no `.pose` field at all -- see eligsir/io/sources/
            # decode_cache.py). Passing `frame` straight through raised
            # AttributeError inside backend.initialize()/.track() (task #20:
            # "open3d_odometry_live bootstrap bug: StreamingRGBDFrame has no
            # .pose attribute"). Build the same canonical-frame conversion
            # the main tracking loop already does correctly further below
            # (see the `_canonical_frame` construction near the
            # "Live backend path" comment), decoding real pixels via
            # load_frame_rgb/load_frame_depth_np rather than reusing that
            # site's `_rgb_np`/`_depth_np`-or-zeros shortcut, since those
            # dataclass fields are normally unset for disk-backed sources
            # (TUM/generic) and zeros would starve Open3D odometry of any
            # real signal to track against during bootstrap.
            try:
                from contracts.types import (
                    CameraIntrinsics as _CIntr_boot,
                    Pose as _CPos_boot,
                    RGBDFrame as _CRF_boot,
                )
                from eligsir.io.sources.decode_cache import (
                    load_frame_depth_np,
                    load_frame_rgb,
                )
                from eligsir.utils.rgbd_frames import depth_to_meters

                _rgb_boot = np.asarray(load_frame_rgb(frame), dtype=np.float32) / 255.0
                _depth_raw_boot = load_frame_depth_np(frame)
                _depth_boot = (
                    depth_to_meters(_depth_raw_boot, frame.depth_scale)
                    if _depth_raw_boot is not None
                    else np.zeros((frame.height, frame.width), dtype=np.float32)
                )
                _intr_boot = _CIntr_boot(
                    fx=frame.fx, fy=frame.fy, cx=frame.cx, cy=frame.cy,
                    width=frame.width, height=frame.height,
                )
                _frame_c2w_boot = getattr(frame, "c2w", None)
                _canonical_boot_frame = _CRF_boot(
                    index=frame.index,
                    timestamp=float(frame.timestamp),
                    rgb=_rgb_boot,
                    depth=_depth_boot,
                    intrinsics=_intr_boot,
                    pose=(
                        _CPos_boot(np.asarray(_frame_c2w_boot, dtype=np.float32))
                        if _frame_c2w_boot is not None else None
                    ),
                )
            except Exception as exc:
                print(f"[streaming] bootstrap frame conversion for SLAM backend failed: {exc}", flush=True)
                _canonical_boot_frame = None

            if _canonical_boot_frame is None:
                _tracking_result = None
                if not _slam_backend_initialized:
                    # Mirrors the original initialize()-branch behavior below:
                    # a failed attempt still counts as "initialization was
                    # attempted", so subsequent frames go through track().
                    _slam_backend_initialized = True
            elif not _slam_backend_initialized:
                try:
                    _tracking_result = _slam_backend.initialize(_canonical_boot_frame)
                except Exception as exc:
                    print(f"[streaming] SLAM backend initialize() failed: {exc}", flush=True)
                    _tracking_result = None
                _slam_backend_initialized = True
            else:
                try:
                    _tracking_result = _slam_backend.track(_canonical_boot_frame)
                except Exception as exc:
                    print(f"[streaming] SLAM backend track() failed: {exc}", flush=True)
                    _tracking_result = None

            # Attach the received pose to the frame so initialize_from_frames can use it.
            _min_conf = float(getattr(args, "streaming_bootstrap_min_confidence", 0.0))
            _passes_gate = bootstrap_frame_passes_confidence_gate(_tracking_result, _min_conf)
            if _passes_gate:
                frame.c2w = _tracking_result.pose.matrix
                posed_frames.append(frame)
                print(
                    f"[streaming] Bootstrap frame {len(posed_frames)}/{n_init}: "
                    f"idx={frame.index} confidence={_tracking_result.confidence:.3f}",
                    flush=True,
                )
            elif _tracking_result is not None and _tracking_result.valid:
                # W3-30: gate rejected this frame on confidence alone (it was
                # otherwise valid) -- distinct from "no pose yet" below. Kept
                # as opt-in defense-in-depth; see
                # docs/architecture/2026-08-10-w3-30-init-seed-bootstrap-confidence-gate.md
                # for why this is NOT believed to be the floor2 scale-explosion
                # fix (that hypothesis was checked against real output and
                # falsified -- the doc names the real next step).
                print(
                    f"[streaming] Bootstrap: rejecting idx={frame.index}, "
                    f"confidence={_tracking_result.confidence:.3f} < "
                    f"streaming_bootstrap_min_confidence={_min_conf:.3f} "
                    f"(attempt {attempts}/{max_attempts})",
                    flush=True,
                )
            else:
                # No valid pose yet — keep draining but don't count this frame.
                print(
                    f"[streaming] Bootstrap waiting... attempt {attempts}/{max_attempts}, "
                    f"tracking valid={_tracking_result.valid if _tracking_result else None}",
                    flush=True,
                )

        if len(posed_frames) < n_init:
            if allow_random_fallback and len(posed_frames) > 0:
                print(
                    f"[streaming] Bootstrap: only {len(posed_frames)}/{n_init} posed frames "
                    f"available; proceeding with {len(posed_frames)} frames (random_fallback=True).",
                    flush=True,
                )
            elif len(posed_frames) > 0:
                print(
                    f"[streaming] Bootstrap: only {len(posed_frames)}/{n_init} posed frames "
                    f"found after {attempts} attempts. Proceeding anyway (identity pose for missing).",
                    flush=True,
                )
            else:
                raise RuntimeError(
                    f"[streaming] Bootstrap failed: no valid poses received from SLAM backend "
                    f"in {attempts} attempts. Is the SLAM node publishing on {_backend_kwargs.get('pose_topic', '/slam/pose')}?"
                )

        # Temporarily replace _frame_source so that initialize_from_frames uses posed_frames.
        # This handles both the lazy (_frame_source.get_frame) and eager (_all_frames) paths.
        _orig_frame_source = streaming_scene._frame_source

        class _BootstrapFrameSource:
            """Wraps posed_frames list to provide the get_frame(index) interface."""
            def __init__(self, frames):
                self._frames = frames
            def __len__(self):
                return len(self._frames)
            def get_frame(self, index):
                return self._frames[min(index, len(self._frames) - 1)]
            def __getitem__(self, key):
                return self._frames[key]

        streaming_scene._frame_source = _BootstrapFrameSource(posed_frames)
        try:
            streaming_scene.initialize_from_frames(len(posed_frames), record_fn=_bootstrap_record_fn)
        finally:
            streaming_scene._frame_source = _orig_frame_source

        # Restore _source_idx so the main drain loop starts AFTER the bootstrap
        # frames (which have already been consumed by the bootstrap loop).
        # _saved_idx is where bootstrap started; posed_frames were consumed from
        # the stream starting there, so advance past them.
        streaming_scene._source_idx = _saved_idx + len(posed_frames)

    phase_timer.end("bootstrap")

    phase_timer.start("training_setup")
    gaussians.training_setup(opt)

    # Ensure SLAM lifecycle attributes exist for bootstrap Gaussians (birth_frame=0).
    # add_points_as_gaussians() creates these lazily on first insertion; initialise
    # them here so that --streaming_anchor_bootstrap works from iteration 1 even when
    # no depth insertion has yet occurred.
    if not hasattr(gaussians, "birth_frame"):
        _n_boot = gaussians.get_xyz.shape[0]
        _dev = gaussians.get_xyz.device
        gaussians.provisional = torch.zeros(_n_boot, dtype=torch.bool, device=_dev)
        gaussians.support_count = torch.zeros(_n_boot, dtype=torch.int32, device=_dev)
        gaussians.birth_frame = torch.zeros(_n_boot, dtype=torch.int32, device=_dev)

    # Bootstrap position anchor: record the initial depth-seeded positions so
    # the anchor loss can penalise center drift during early training.
    # Only initialise if not already set (checkpoint resume preserves existing anchor).
    if not hasattr(gaussians, "anchor_xyz"):
        gaussians.anchor_xyz = gaussians.get_xyz.detach().clone()

    # Restore checkpoint if requested
    first_iter = 0
    _ckpt_streaming_meta: dict = {}
    _exact_resume_envelope = None
    _exact_resume_verdict = None
    _resumed_from_iteration = None
    if checkpoint:
        _loaded = torch.load(checkpoint, weights_only=False)
        # Backward compatible: older checkpoints are a plain (state, iteration)
        # 2-tuple with no streaming-cursor payload. Newer checkpoints (see the
        # save site below) are a 3-tuple (state, iteration, streaming_meta)
        # carrying n_frames_ingested / _last_ingest_iter / the frame-source
        # cursor, so a resumed run does not silently re-ingest / re-admit
        # frames already consumed before the checkpoint (or re-trigger the
        # bootstrap phase) — see docs on resume in CLAUDE.md's Run Contracts.
        if len(_loaded) == 3:
            model_params, first_iter, _ckpt_streaming_meta = _loaded
        else:
            model_params, first_iter = _loaded
            print("[streaming] WARNING: resuming from a checkpoint with no "
                  "streaming-cursor metadata (pre-resume-fix checkpoint) — "
                  "frame ingestion will restart from the bootstrap position; "
                  "this is a KNOWN-STALE resume, not a faithful one.", flush=True)
        from eligsir.map.contracts import CheckpointBlob as _CheckpointBlob
        map_backend.restore_checkpoint(
            _CheckpointBlob(
                layout="gsplat" if model_layout == "gsplat" else "legacy_graphdeco",
                payload=model_params,
                map_version=int(_ckpt_streaming_meta.get("map_version", 0)),
                topology_version=int(_ckpt_streaming_meta.get("topology_version", 0)),
            ),
            opt,
        )
        if _ckpt_streaming_meta and "source_idx" in _ckpt_streaming_meta:
            _exact_resume_envelope = _ckpt_streaming_meta.get("streaming_resume_v2")
            # Reposition the frame source's real cursor (the thing that
            # actually gates which frame gets read next — see
            # StreamingScene.ingest_next_frame/_source_idx) to the checkpoint's
            # saved position. Restoring n_frames_ingested alone is NOT
            # sufficient: it is just a training-loop counter, independent of
            # the frame source's own sequential cursor, which bootstrap left
            # at n_init.
            streaming_scene._source_idx = int(_ckpt_streaming_meta["source_idx"])
            print(f"[streaming] resume: restored frame-source cursor "
                  f"_source_idx={streaming_scene._source_idx} from checkpoint", flush=True)
            if "keyframes_admitted" in _ckpt_streaming_meta:
                streaming_scene._keyframes_admitted = int(_ckpt_streaming_meta["keyframes_admitted"])
                print(f"[streaming] resume: restored keyframes_admitted="
                      f"{streaming_scene._keyframes_admitted} from checkpoint", flush=True)
            # Rebuild test cameras for holdout frames the cursor already
            # passed — otherwise the resumed run's report.json silently gets
            # n_test=0/eval_valid=False (see materialize_missed_holdouts
            # docstring). Must run regardless of resume position (start,
            # mid-stream, or post-coverage) since the holdout set spans the
            # whole trajectory.
            if _exact_resume_envelope is None:
                _n_holdouts_rebuilt = streaming_scene.materialize_missed_holdouts()
                print(f"[streaming] resume: rebuilt {_n_holdouts_rebuilt} holdout test "
                      f"camera(s) below the restored cursor (n_test now "
                      f"{len(streaming_scene.getTestCameras())})", flush=True)
        _resumed_from_iteration = first_iter
        # Issue #290: restore the seed-fade detector's decision state, or say
        # loudly that we could not. `mark_resumed` is called UNCONDITIONALLY
        # on a resume (even with no payload) -- the record of the resume is
        # what makes the run's verdict honestly `indeterminate` instead of
        # silently describing only the post-resume segment.
        if seed_fade_monitor.mark_resumed(
            first_iter, _ckpt_streaming_meta.get("seed_fade")
        ):
            print(f"[streaming] resume: restored seed-fade detector state "
                  f"(running_max={seed_fade_monitor._running_max:.3f}dB, "
                  f"n_samples={seed_fade_monitor.n_samples}, "
                  f"episodes={len(seed_fade_monitor.episodes)}) from checkpoint",
                  flush=True)
        else:
            print("[SEED-FADE] WARNING: resumed from a checkpoint WITHOUT the "
                  "detector's decision state -- this run's seed-fade verdict is "
                  "INDETERMINATE and must not be quoted as evidence "
                  "(report.json['seed_fade_indeterminate'] = true). Reasons: "
                  + "; ".join(seed_fade_monitor.resume_notes), flush=True)

    # --- Frame scheduler ---------------------------------------------------
    _ingestion_mode, _ingestion_reason = resolve_ingestion_mode_with_reason(
        dataset.source_path,
        getattr(args, "streaming_ingestion_mode", "auto"),
        online_mode=bool(getattr(args, "online_mode", False)),
    )
    print(
        f"[streaming] ingestion_mode={_ingestion_mode} "
        f"(requested={getattr(args, 'streaming_ingestion_mode', 'auto')}, "
        f"reason={_ingestion_reason})",
        flush=True,
    )
    if _ingestion_reason == INGESTION_REASON_ONLINE_MODE_DEFAULT:
        # Issue #302: the --online_mode -> dataset_fps default is a reverted
        # fix that was deliberately retained rather than silently re-flipped.
        # It must never again be possible to run under it without knowing.
        print(ONLINE_MODE_DATASET_FPS_WARNING, flush=True)
    # Recorded in report.json["ingestion"] so every run self-identifies which
    # pacing it actually ran under (the thing that would have made the #302
    # corpus question answerable instead of forensic).
    # Issue #834: resolve_source_decimation is fed the same
    # (_requested_frame_stride, _requested_source_stride, _legacy_composition_mode)
    # triple that the frame-source construction above already resolved and
    # enforced via `_sampling_policy` -- so `effective_stride`/`sampling_owner`
    # here are not a resolved *intent* that might disagree with what actually
    # ran, they describe exactly what the args.streaming_frame_stride override
    # + StridedFrameSource wrap decision (both derived from that same
    # `_sampling_policy`) applied. `application_verified` records that this
    # identity is guaranteed by construction, not merely assumed.
    _source_decimation_block = resolve_source_decimation(
        _requested_frame_stride,
        _requested_source_stride,
        frames_before=_frames_before_source_stride,
        frames_after=_frames_after_source_stride,
        frames_source=_frames_source_provenance,
        legacy_composition_mode=_legacy_composition_mode,
    )
    _source_decimation_block["sampling_mode"] = _source_sampling_mode
    _source_decimation_block["application_verified"] = True
    _ingestion_report = {
        "mode": _ingestion_mode,
        "reason": _ingestion_reason,
        "requested": str(getattr(args, "streaming_ingestion_mode", "auto")),
        "online_mode": bool(getattr(args, "online_mode", False)),
        "steps_per_frame_honored": _ingestion_mode == INGESTION_MODE_ITER_BASED,
        "source_decimation": _source_decimation_block,
    }
    # Full-coverage enforcement (2026-07-26 fix): the 2026-07-04 refactor demoted
    # _mapping_scheduler.effective_spf to an informational-only statistic, reasoning
    # that the rolling-idle-stop termination rule (_last_ingest_iter /
    # --streaming_refinement_iterations) now extends the real iteration budget to
    # fit actual ingestion need. But that extension only exists for --online_mode
    # runs (opt.iterations becomes the ONLINE_UNBOUNDED_ITERATIONS sentinel there,
    # see eligsir/core/stop_policy.py) -- a fixed, non-online_mode --iterations
    # budget never extends, so the default streaming_steps_per_frame=150 silently
    # stranded most of the trajectory unseen on any dataset where
    # n_frames * 150 > iterations (confirmed 2026-07-26: fr3's 2525-frame TUM
    # sequence reached only ~7% coverage at 30000 iterations, vs the historical
    # champion's full 2525/2525 coverage under the same nominal recipe -- the
    # champion's run predates the 2026-07-04 demotion and still had real
    # enforcement). Use the coverage-enforced effective_spf whenever the run
    # cannot rely on the rolling-idle-stop extension to compensate.
    #
    # #1044 CORRECTION: the "online_mode can rely on rolling-idle to compensate"
    # reasoning above is only sound when the run is genuinely unbounded. An
    # online run with a finite planning budget (--streaming_max_iterations, or
    # the operator's own --iterations) has exactly the same stranding problem a
    # fixed-budget run has, and measurably suffered it (fr3: 130/2525 frames).
    # Enforce coverage whenever a finite planning budget was resolved,
    # regardless of online_mode; bypass only when there is truly no budget to
    # plan against.
    # Scoped to iter_based deliberately: under dataset_fps/wallclock_strict the
    # release gate never consults SPF, so enforcing it there would change a
    # reported number without changing any behaviour. The online bypass stays
    # byte-identical for those modes.
    _use_enforced_spf = (not bool(getattr(args, "online_mode", False))) or (
        _coverage_planning_budget is not None
        and _ingestion_mode == INGESTION_MODE_ITER_BASED
    )
    _frame_scheduler_spf = (
        max(1, _mapping_scheduler.effective_spf) if _use_enforced_spf
        else max(1, getattr(args, "streaming_steps_per_frame", 50))
    )
    # Honest projection: how much of the trajectory this SPF can actually
    # release inside the planning budget (see SpfResolution.format_log).
    _projected_coverage_frac: "float | None" = None
    if _coverage_planning_budget is not None and _coverage_n_frames > 0:
        _releasable = int(_coverage_planning_budget) // max(1, _frame_scheduler_spf)
        _projected_coverage_frac = min(1.0, _releasable / float(_coverage_n_frames))
    scheduler = FrameScheduler(
        fps=getattr(args, "streaming_input_fps", 30.0),
        steps_per_frame=_frame_scheduler_spf,
        wallclock=getattr(args, "streaming_wallclock", False),
        ingestion_mode=_ingestion_mode,
        fps_cap=getattr(args, "streaming_input_fps_cap", 30.0),
    )
    # --- Issue #596: ONE authoritative steps-per-frame summary --------------
    # Emitted HERE, and only here, because this is the first point at which
    # every override is resolved: the argparse/profile request, the coverage
    # arithmetic, the min_spf floor, the --online_mode bypass, and the
    # ingestion mode that decides whether the release gate reads the value at
    # all. Every field is RECORDED from an already-computed variable — most
    # importantly resolved_spf is _frame_scheduler_spf verbatim, the exact
    # value FrameScheduler was constructed with two lines up — so this block
    # cannot change which SPF wins.
    #
    # governs_release reuses _ingestion_report["steps_per_frame_honored"]
    # rather than re-deriving `_ingestion_mode == iter_based`: one fact, one
    # place (the same lesson issue #302 drew about ingestion provenance).
    #
    # Realized SPF is deliberately NOT measured here — utils/realized_cadence.py
    # already emits it (report.json["realized_cadence"]), sourced from this same
    # _frame_scheduler_spf, so the two blocks cannot drift.
    _spf_resolution = build_spf_resolution(
        requested_spf=_orig_spf,
        resolved_spf=_frame_scheduler_spf,
        scheduler_effective_spf=_mapping_scheduler.effective_spf,
        scheduler_resolved_by=_mapping_scheduler.spf_resolved_by,
        coverage_required_spf=_mapping_scheduler.coverage_required_spf,
        min_spf_floor=_mapping_scheduler.min_steps_per_frame,
        use_enforced_spf=_use_enforced_spf,
        governs_release=bool(_ingestion_report["steps_per_frame_honored"]),
        ingestion_mode=_ingestion_mode,
        coverage_budget=int(_coverage_planning_budget or 0),
        coverage_n_frames=int(_coverage_n_frames),
        registry_hit=bool(_is_finite_replay),
        full_coverage_enforced=bool(_mapping_scheduler.stats.full_coverage),
        n_frames_capped_diagnostic=_mapping_scheduler.n_frames_capped,
        online_mode=bool(getattr(args, "online_mode", False)),
        coverage_budget_source=_coverage_budget_source,
        projected_coverage_frac=_projected_coverage_frac,
    )
    print(_spf_resolution.format_log(), flush=True)
    _iter_wall_start: float = time.perf_counter()  # for per-iter dt in dataset_fps mode

    # --- Realized-cadence accounting (issue #289) ---------------------------
    # Seven authoritative clocks, denominated in OPTIMIZER STEPS (not
    # iterations), persisted to report.json["realized_cadence"] for every
    # streaming run. This exists because requested cadence has now been shown
    # to diverge from realized cadence in both directions -- #278/#302 (an
    # advertised effective_spf=150 released the whole trajectory by step ~1250,
    # because dataset_fps pacing never reads steps_per_frame) and W3-38 (an
    # explicit low steps_per_frame silently clamped UP by min_steps_per_frame).
    # Instantiated here, before the admission closures are defined, so those
    # closures can record the candidate-evaluation clock. All recording is O(1)
    # integer work: no tensor reads, no device syncs (see the module docstring's
    # cost section and CLAUDE.md's observability invariant).
    _cadence = RealizedCadenceTracker(
        first_iteration=first_iter,
        window_steps=max(1, int(getattr(args, "streaming_cadence_window_steps", 500))),
        resumed=_resumed_from_iteration is not None,
    )

    # #602 admission/cadence contract. Records only what is knowable here; the
    # per-frame disposition call sites (admitted / coalesced / skipped) belong
    # to #600 and #601 and are deliberately NOT added by this issue, which is
    # validation/telemetry-only. Until those owners declare themselves, the
    # disposition counters are derived from _cadence's clocks and the
    # backpressure-owned fields report null-with-a-reason rather than a
    # misleading zero -- see utils/admission_cadence's "absent != zero" rule.
    _admission_cadence = AdmissionCadenceRecorder()
    _admission_cadence.declare_release_policy(
        requested=(float(getattr(args, "streaming_steps_per_frame", 0) or 0)
                   or None),
        resolved=float(_frame_scheduler_spf),
        nominal_steps_per_frame=(
            float(getattr(args, "streaming_steps_per_frame", 0) or 0) or None),
        effective_spf=float(_mapping_scheduler.effective_spf),
    )
    # admission_policy_requested is the pre-override value captured near the
    # top of this function. #602 deliberately left admission_target_resolved
    # undeclared, since resolving "all" into soft_all vs force_all_stress is
    # #600's decision; this is #600 making it, through the single authoritative
    # resolver rather than a second string test.
    _admission_cadence.admission_policy_requested = _admission_policy_requested
    _admission_semantics = resolve_admission_semantics(
        args.streaming_frame_admission,
        source_mode=(
            "live" if str(getattr(args, "source_path", "")).startswith("ros2://")
            else None
        ),
    )
    _admission_cadence.declare_admission_target(
        # The pre-override request stays #602's value; we declare only what the
        # LIVE path resolved to, so a divergence between them stays visible as
        # the finding #602's docstring says it is, rather than being arbitrated.
        policy_requested=_admission_policy_requested,
        target_resolved=_admission_semantics["admission_target"],
        budgeted=_admission_semantics["admission_budgeted"],
        source_coverage_target=_admission_semantics["source_coverage_target"],
        # force_all is an intentional assimilation-overload stress condition,
        # not a production configuration (#600/#603).
        stress_run=(
            _admission_semantics["admission_target"]
            == ADMISSION_TARGET_FORCE_ALL_STRESS
        ),
    )
    # #601's bounded backpressure owner has now landed. When it is active it
    # IS the single execution owner, so it -- not soft-all -- is what gets
    # declared here, with its own real bounds.
    #
    # Otherwise fall back to #602's original reading: what soft-all has is a
    # genuine but deliberately minimal bound, declared only when actually in
    # force. Declaring an owner for force_all or a gated mode would flip
    # #602's coalesce/skip counters from "absent, with a reason" to a
    # misleading zero -- exactly the "absent != zero" rule that module states.
    if _backpressure_owner is not None and not _backpressure_cfg.force_all_stress:
        _admission_cadence.declare_backpressure_owner(
            owner="backpressure_owner",
            pending_max_count=int(_backpressure_cfg.pending_max_count),
            # Deliberately None unless the operator explicitly opted into the
            # wall-clock age axis. #601's default age bound is measured in
            # optimizer ITERATIONS, because a millisecond bound cannot satisfy
            # the same issue's requirement that the retained set be
            # deterministic in offline replay. Reporting a converted number
            # here would fabricate a bound this run does not actually use;
            # the real one is in report.json["backpressure"]["config"].
            pending_max_age_ms=(
                float(_backpressure_cfg.pending_max_age_ms)
                if _backpressure_cfg.age_axis == "wallclock"
                and _backpressure_cfg.pending_max_age_ms > 0
                else None
            ),
        )
    elif _admission_semantics["admission_target"] == ADMISSION_TARGET_SOFT_ALL:
        _admission_cadence.declare_backpressure_owner(
            owner="soft_all_admitter",
            pending_max_count=int(
                getattr(args, "streaming_soft_all_max_consecutive_skips", 4)
            ),
            # No age bound in wall-clock terms: soft-all's age bound is the
            # consecutive-skip count above, not a millisecond deadline. Left
            # None rather than converted, so it reads as "this owner has no
            # ms-based bound" instead of a fabricated number.
            pending_max_age_ms=None,
        )

    def _cadence_reconcile_kwargs() -> dict:
        """Arguments for the requested-vs-realized reconciliation.

        The GOVERNING value is ``_frame_scheduler_spf`` -- the value
        ``FrameScheduler`` was actually constructed with -- not
        ``_mapping_scheduler.effective_spf``, because for an ``--online_mode``
        run the latter is never handed to the scheduler at all (see
        ``_use_enforced_spf`` above). Reconciling against a number the release
        gate never saw would report a divergence that isn't the run's.

        ``MappingScheduler.effective_spf`` is passed too, under its own name, so
        the persisted block lines up with ``train.log``'s ``[MappingScheduler]``
        line instead of quietly disagreeing with it on the runs where the two
        differ.
        """
        return {
            "requested_steps_per_frame": float(
                getattr(args, "streaming_steps_per_frame", 0) or 0) or None,
            "frame_scheduler_spf": float(_frame_scheduler_spf),
            "mapping_scheduler_effective_spf": float(
                _mapping_scheduler.effective_spf),
            "ingestion_mode": _ingestion_mode,
            "frame_admission": args.streaming_frame_admission,
        }

    # --- MCMC / strategy setup ---------------------------------------------
    from eligsir.utils.mcmc_schedule import MCMCScheduleConfig, get_mcmc_schedule
    from eligsir.core.stop_policy import enforce_growth_stop
    from eligsir.utils.strategies import make_mcmc_strategy
    from eligsir.utils.compiled_kernels import configure_torch_compile, set_compile_iteration

    densification_strategy = getattr(opt, "densification_strategy", "gsplat_energy_mcmc").lower()
    if int(getattr(args, "streaming_seed_lineage_freeze_from_iter", -1)) >= 0:
        args._seed_lineage_freeze_realized_strategy = densification_strategy
    from eligsir.utils.seed_lineage_freeze import get_seed_lineage_freeze
    _seed_lineage_freeze = get_seed_lineage_freeze(gaussians, args)
    if _seed_lineage_freeze is not None:
        _seed_lineage_freeze.validate_configuration()

    def _sanitize_with_seed_freeze(where, current_iteration, *, diag_interval):
        protected = (
            None
            if _seed_lineage_freeze is None
            else _seed_lineage_freeze.current_mask(iteration=current_iteration)
        )
        _sanitize_gaussian_params_sampled(
            gaussians,
            where,
            current_iteration,
            diag_interval=diag_interval,
            protected_mask=protected,
        )
        if _seed_lineage_freeze is not None:
            _seed_lineage_freeze.assert_integrity(
                iteration=current_iteration, context=f"sanitize:{where}"
            )
    mcmc_cfg = MCMCScheduleConfig(
        start_iter=opt.densify_from_iter,
        # _mcmc_growth_schedule_horizon (set by apply_online_schedule_overrides,
        # 2026-07-18) is the growth-RATE schedule's normalization horizon,
        # deliberately decoupled from mcmc_stop_growth_iter's hard-stop gate --
        # see that function's docstring. Falls back to mcmc_stop_growth_iter
        # for callers that never went through the online-mode overrides.
        stop_growth_iter=getattr(opt, "_mcmc_growth_schedule_horizon", None)
        or getattr(opt, "mcmc_stop_growth_iter", 12_000),
        stop_reloc_iter=opt.densify_until_iter,
        growth_factor_start=getattr(opt, "mcmc_growth_factor_start", 1.05),
        post_ingestion_growth_factor_start=getattr(
            opt, "streaming_mcmc_post_ingestion_growth_factor_start", 0.0
        ),
        growth_factor_min=getattr(opt, "mcmc_growth_factor_min", 1.002),
        growth_factor_tau=getattr(opt, "mcmc_growth_factor_tau", 0.35),
        grow_interval_min=getattr(opt, "mcmc_grow_interval_min", 100),
        grow_interval_max=getattr(opt, "mcmc_grow_interval_max", 2000),
        grow_tau=getattr(opt, "mcmc_grow_tau", 0.35),
        relocate_interval_min=getattr(opt, "mcmc_relocate_interval_min", 50),
        relocate_interval_max=getattr(opt, "mcmc_relocate_interval_max", 500),
        relocate_tau=getattr(opt, "mcmc_relocate_tau", 0.65),
        cap_growth_power=getattr(opt, "mcmc_cap_growth_power", 2.0),
        cap_interval_strength=getattr(opt, "mcmc_cap_interval_strength", 4.0),
        cap_interval_power=getattr(opt, "mcmc_cap_interval_power", 2.0),
        cap_stop_ratio=getattr(opt, "mcmc_cap_stop_ratio", 0.98),
        dead_opacity_start=getattr(opt, "mcmc_dead_opacity_start", 0.003),
        dead_opacity_end=getattr(opt, "mcmc_dead_opacity_end", 0.010),
        dead_opacity_power=getattr(opt, "mcmc_dead_opacity_power", 1.5),
        use_target_deficit=getattr(opt, "mcmc_use_target_deficit", False),
        target_splat_end=getattr(opt, "mcmc_target_splat_end", 150000),
        target_q_start=getattr(opt, "mcmc_target_q_start", 0.05),
        target_q_end=getattr(opt, "mcmc_target_q_end", 0.85),
        target_tau=getattr(opt, "mcmc_target_tau", 0.45),
        target_capacity=int(getattr(args, "streaming_target_capacity", 0)),
    )
    mcmc_strategy = make_mcmc_strategy(densification_strategy, gaussians=gaussians, args=args)
    mcmc_strategy.initialize_state(
        gaussians=gaussians, args=args,
        scene_scale=float(getattr(streaming_scene, "cameras_extent", 1.0)),
    )
    configure_torch_compile(args)

    # --- SSIM backend selection (per-iteration launch-bound critical path) ----
    from eligsir.utils.loss_utils import configure_training_ssim
    configure_training_ssim(opt)

    # --- SLAM backend (Phase 1.5) — setup hoisted above the bootstrap section.

    # --- Live viewer (Phase 1.5) ---------------------------------------------
    from eligsir.viewers import make_viewer
    _viewer_name = getattr(args, "viewer", "none")
    _viewer_kwargs = {}
    if _viewer_name == "rerun":
        _viewer_kwargs = {
            "serve_port": getattr(args, "viewer_rerun_port", 9876),
            "max_gaussians_logged": getattr(args, "viewer_rerun_max_gaussians", 50000),
        }
    elif _viewer_name == "ros2_rviz":
        _viewer_kwargs = {
            "topic": getattr(args, "viewer_rviz_topic", "/eligsir/gaussian_splats"),
            "publish_interval": getattr(args, "viewer_rviz_publish_interval", 100),
            "min_publish_period_s": getattr(args, "viewer_rviz_min_publish_period_s", 2.0),
            "camera_publish_interval": getattr(args, "viewer_rviz_camera_interval", 1),
            "publish_clock": getattr(args, "viewer_rviz_publish_clock", True),
            "max_gaussians": getattr(args, "viewer_rviz_max_gaussians", 0),
            "only_when_map_version_changed": getattr(args, "viewer_rviz_only_when_map_version_changed", False),
            "world_frame": getattr(args, "viewer_rviz_world_frame", "map"),
            "transport": args.viewer_rviz_transport,
            "blob_topic": getattr(args, "viewer_rviz_blob_topic", "/eligsir/splat_map/blob_chunks"),
            "tile_topic": getattr(args, "viewer_rviz_tile_topic", "/eligsir/splat_map/tile_chunks"),
            "session_id": getattr(args, "viewer_rviz_session_id", "eligsir"),
            "chunk_max_bytes": getattr(args, "viewer_rviz_chunk_max_bytes", 1048576),
            "tile_size_m": getattr(args, "viewer_rviz_tile_size_m", 1.0),
            # Spatial active-bubble viz (sphere anchored to camera optical frame).
            "bubble_enabled": bool(getattr(args, "streaming_spatial_bubble_enabled", False)),
            "bubble_radius": float(getattr(args, "streaming_bubble_radius", 5.0)),
            "bubble_guard": float(getattr(args, "streaming_bubble_guard_m", 1.0)),
            "bubble_forward_offset": float(getattr(args, "streaming_bubble_forward_offset", 0.0) or 3.0),
        }
    elif _viewer_name == "viser":
        # Opt-in canonical path (#570/#572). See eligsir/viewers/registry.py's
        # _make_viser_viewer and canonical_adapter.py -- byte-unchanged for
        # every other --viewer value.
        _viewer_kwargs = {
            "port": getattr(args, "viewer_viser_port", 8080),
            "host": getattr(args, "viewer_viser_host", "0.0.0.0"),
            "map_max_gaussians": getattr(args, "viewer_map_max_gaussians", 100_000),
        }
    # #570/#574/#583: --viewer_sinks is an explicit, comma-separated
    # canonical sink list, independent of --viewer/_viewer_name above. When
    # it names "viser" (whether or not --viewer is also "viser"), the viser
    # sink still needs its port/host kwargs -- registry.make_viewer's
    # canonical-sink path reads them via dict.get, so passing them
    # unconditionally here is harmless for any OTHER requested sink, it is
    # simply unused. Left unset (the default), this changes nothing: the
    # elif chain above is untouched and make_viewer's viewer_sinks=None
    # branch dispatches exactly as it always has.
    _viewer_sinks_arg = str(getattr(args, "viewer_sinks", "") or "")
    _viewer_sink_names = [s.strip() for s in _viewer_sinks_arg.split(",") if s.strip()]
    if "viser" in _viewer_sink_names and "port" not in _viewer_kwargs:
        _viewer_kwargs["port"] = getattr(args, "viewer_viser_port", 8080)
        _viewer_kwargs["host"] = getattr(args, "viewer_viser_host", "0.0.0.0")
        _viewer_kwargs["map_max_gaussians"] = getattr(args, "viewer_map_max_gaussians", 100_000)
    _viewer = make_viewer(
        _viewer_name,
        viewer_sinks=(_viewer_sinks_arg or None),
        **_viewer_kwargs,
    )
    # Discriminates "routed through the canonical PreviewBus adapter" from
    # every legacy backend (none/rerun/the removed ros2_rviz), independent of
    # the specific --viewer/--viewer_sinks spelling that got us there -- this
    # is what un-gates the PreviewBus producer call sites below from the old
    # literal `_viewer_name == "viser"` check (#570/#574/#583: those sites
    # were reachable only via --viewer viser; a `--viewer_sinks viser` run,
    # or a future non-viser canonical sink, builds the identical adapter and
    # must fire the same producer calls). The adapter class
    # (canonical_adapter.py) is the only Viewer implementation exposing this
    # property -- checked via hasattr rather than isinstance/import to keep
    # this call site free of any new import.
    _viewer_is_canonical = hasattr(_viewer, "canonical_viewer")
    _map_output_lane = None
    _map_output_last_wall_s = 0.0
    if _viewer_is_canonical and hasattr(_viewer, "bind_map_output_lane"):
        from eligsir.runtime.transfers import get_device_transfer_owner
        from eligsir.viewers.map_output_lane import VersionedMapOutputLane

        _map_cfg = getattr(_viewer.canonical_viewer, "config", None)
        _map_output_lane = VersionedMapOutputLane(
            get_device_transfer_owner(gaussians.get_xyz.device),
            max_gaussians=int(getattr(_map_cfg, "map_max_gaussians", 100_000)),
        )
        _viewer.bind_map_output_lane(_map_output_lane)

    # Trajectory accumulator for viewer
    _trajectory_est: list = []

    # --- Submap manager (Phase 12) ------------------------------------------
    from eligsir.utils.submap_manager import SubmapManager
    _submap_frames = int(getattr(args, "submap_frames_per_submap", 50))
    _submap_manager = SubmapManager(frames_per_submap=_submap_frames)
    if not hasattr(gaussians, "_current_submap_id"):
        gaussians._current_submap_id = 0

    # --- Frame observation DB (keyframe-database design, Task 4) ------------
    # _observation_db / _multires_low_divisor are created earlier (before
    # bootstrap ingestion) -- see _bootstrap_record_fn above.

    # --- Multi-resolution frame cache + controller (multires-cache plan, Task 4) --
    from eligsir.mapping.resolution_controller import ResolutionController
    # Issue #1259: "native" (default) preserves the historical permanent
    # lock-to-high-resolution behavior on mapping_done. "mapping_tier"
    # disables the lock so the reactive floor logic (and, in practice, the
    # frozen rolling window once admission events stop) carries the tier
    # mapping ended at through the refinement tail instead -- see
    # ResolutionController's lock_to_high_on_mapping_done docstring for the
    # full mechanism and docs/architecture/2026-08-31-issue1259-refinement-tier-cliff.md
    # for the measured cliff this addresses.
    _refinement_resolution_policy = str(
        getattr(args, "streaming_refinement_resolution_policy", "native")
    )
    if _refinement_resolution_policy not in ("native", "mapping_tier"):
        raise ValueError(
            "--streaming_refinement_resolution_policy must be 'native' or "
            f"'mapping_tier', got {_refinement_resolution_policy!r}"
        )
    if (
        _refinement_resolution_policy == "mapping_tier"
        and int(getattr(args, "streaming_post_mapping_refinement_steps", 0) or 0) > 0
    ):
        raise ValueError(
            "--streaming_refinement_resolution_policy mapping_tier is "
            "incompatible with --streaming_post_mapping_refinement_steps > 0 "
            "('exact refinement' hard-requires native resolution for "
            "owner-transfer/resume-envelope reproducibility -- see "
            "train_streaming.py's 'exact refinement boundary requires "
            "native HIGH resolution' guard). Use the default 'native' "
            "policy for exact-refinement/paper-comparison runs; "
            "'mapping_tier' is only for the general rolling-idle "
            "refinement tail (--streaming_post_mapping_refinement_steps 0, "
            "the default)."
        )
    _resolution_controller = ResolutionController(
        it_v_floor=float(getattr(args, "streaming_multires_it_v_floor", 30.0)),
        map_fps_floor=float(getattr(args, "streaming_multires_map_fps_floor", 3.0)),
        scheduled_low_until_iteration=int(
            getattr(args, "streaming_multires_low_until_iteration", 0)
        ),
        lock_to_high_on_mapping_done=(_refinement_resolution_policy == "native"),
    )
    _prev_active_tier = [None]
    _multires_bind_stats = {"low_ok": 0, "low_fail": 0, "high_ok": 0, "high_fail": 0}
    # Issue #595: the DYNAMIC half of report.json["resolution"]. Fed from
    # `decide()`'s RETURN value (see ResolutionTierLog's docstring for why not
    # ResolutionController._tier), so it agrees with the
    # streaming_multires/active_tier_is_high TB scalar by construction.
    _resolution_tier_log = ResolutionTierLog(
        it_v_floor=float(getattr(args, "streaming_multires_it_v_floor", 30.0)),
        map_fps_floor=float(getattr(args, "streaming_multires_map_fps_floor", 3.0)),
        low_divisor=int(_multires_low_divisor),
        scheduled_low_until_iteration=int(
            getattr(args, "streaming_multires_low_until_iteration", 0)
        ),
    )

    def _visibility_fn(cam, frame):
        # Task 5 fix (C1): stash observed-uids on `cam` from INSIDE
        # ingest_next_frame, before admission_fn runs -- see
        # scene/streaming_scene.py's ingest_next_frame docstring.
        from eligsir.mapping.visibility import project_visible_gaussian_uids
        cam._observed_gaussian_uids = project_visible_gaussian_uids(gaussians, cam)

    def _record_fn(cam, frame, is_train, is_holdout):
        # Task 4 fix (C2): called for every candidate frame that reaches a
        # real `cam` object -- admitted, holdout, AND rejected -- not just
        # the ones that survive admission.
        if not getattr(_record_fn, "_warned_record", False):
            print(f"[multires-debug] _record_fn called: frame.index={frame.index!r} "
                  f"is_train={is_train} cam._streaming_frame is frame: {getattr(cam, '_streaming_frame', None) is frame}",
                  flush=True)
            _record_fn._warned_record = True
        # Four-clocks telemetry (F2 fix, 2026-08-06): this callback fires
        # for every candidate frame reaching a real `cam` -- admitted,
        # holdout, AND rejected -- so it's the one place that can split
        # `input_backlog_frames`'s conflated count into its real
        # components. Admitted frames are already counted separately via
        # record_admitted_keyframe() at the ingest call site.
        if is_holdout:
            _coordinator.four_clocks.record_frame_holdout()
        elif not is_train:
            _coordinator.four_clocks.record_frame_rejected()
        record_frame_observation(
            _observation_db, cam=cam, visible_uids=getattr(cam, "_observed_gaussian_uids", None),
            frame_id=frame.index, submap_id=_submap_manager.current_submap_id,
            is_train=is_train, is_holdout=is_holdout,
            frame=frame, low_res_divisor=_multires_low_divisor,
            depthless_rgb_cache=_multires_depthless_rgb_cache,
            depth_downsample=_multires_depth_downsample,
        )
        # W3-06: shadow frame-utility scoring for admission modes OTHER than
        # hybrid_keyframe. `_keyframe_admission_closure` (below) is the ONLY
        # call site under hybrid_keyframe -- it has the richer decision-
        # derived signals (render_overlap, pose_delta_deg, quality_weight)
        # that `_record_fn` doesn't. Gating on admission mode here avoids
        # double-scoring the same frame twice under hybrid_keyframe (this
        # callback fires for every candidate frame regardless of mode, per
        # its own docstring above). Under `admission=all` (this project's
        # documented full-coverage convention, see CLAUDE.md) this is the
        # ONLY shadow call site that ever fires, since admission_fn is never
        # invoked at all in that mode -- no render_pkg/decision exists here,
        # so redundancy/parallax/RGB-residual terms stay at their neutral
        # defaults; still exercises pose_quality/image_quality/cost/logging.
        if _shadow_utility_policy.enabled and args.streaming_frame_admission != "hybrid_keyframe":
            try:
                _last_summary = _shadow_utility_policy.last_summary
                _shadow_utility_policy.observe_candidate(
                    frame_id=int(frame.index),
                    camera_key=id(cam),
                    admitted=bool(is_train),
                    iteration=int(iteration),
                    pose_confidence=1.0,
                    image_quality=1.0,
                    depth_valid_fraction=1.0,
                    n_gaussians=int(gaussians.get_xyz.shape[0]),
                    resolution_bucket=str(getattr(_resolution_controller, "_tier", "default")),
                    topology_version=int(getattr(gaussians, "topology_version", 0)),
                    # W3-06/issue #241: this is the ONLY observe_candidate call
                    # site that fires under --streaming_frame_admission all
                    # (this project's documented full-coverage convention --
                    # see CLAUDE.md), so its map_version pattern must match the
                    # two other real call sites (train_streaming.py's
                    # _keyframe_admission_closure and its recompute_utility
                    # sibling): map_version=iteration, NOT the pinned-0
                    # gaussians.topology_version fallback the pre-fix code used
                    # here. scripts/analyze_frame_utility.py's per-component
                    # join (issue #118 PR #234/#240) matches each
                    # realized_gain.jsonl row back to the scores.jsonl row it
                    # came from on (frame_id, admitted_iteration) vs.
                    # (frame_id, meta.map_version) -- register_admission()
                    # below is called with this same `iteration`, so the two
                    # must agree for the join to ever match. Pinning
                    # map_version to a constant (0) broke that join for every
                    # admission=all run (confirmed in PR #240's diagnosis).
                    # This does NOT touch topology_version (still the
                    # documented W3-17/W3-39 pinned-0 gap, out of scope here)
                    # and does not change training/mutation behavior -- this
                    # whole block is shadow-only telemetry (see
                    # ShadowUtilityPolicy's module docstring).
                    map_version=int(iteration),
                    under_supported_mass=float(_last_summary.under_supported_mass) if _last_summary is not None else 0.0,
                )
            except Exception as _shadow_record_exc:  # noqa: BLE001 -- shadow scoring must never break ingestion
                print(f"[shadow-utility] _record_fn observe_candidate failed (non-fatal): {_shadow_record_exc}", flush=True)

    # --- Phase 13: PoseGraph loop closure -----------------------------------
    _pose_graph = _maybe_create_pose_graph(args)
    _last_optimized_iter = 0
    _loop_closure_enabled = getattr(args, "loop_closure_enabled", False)
    _lc_min_coherence = float(getattr(args, "loop_closure_min_coherence", 0.6))
    _lc_min_inliers = int(getattr(args, "loop_closure_min_inliers", 10))
    _lc_optimize_interval = int(getattr(args, "loop_closure_optimize_interval", 500))
    _lc_apply_correction = getattr(args, "loop_closure_apply_correction", False)
    # Milestone 5: when --streaming_correction_default_on_for_live is True
    # and we're using a live SLAM backend, flip on loop-closure corrections
    # automatically. Offline / precomputed runs keep the legacy default.
    # ROS transports own their upstream pose-provider semantics after #1073;
    # core only auto-enables this for an in-process tracking frontend.
    if (
        not _lc_apply_correction
        and bool(getattr(args, "streaming_correction_default_on_for_live", True))
        and (
            _pose_source_name == "open3d_odometry_live"
        )
    ):
        _lc_apply_correction = True
        _lc_reason = _pose_source_name
        print(
            f"[streaming] live SLAM backend ({_lc_reason}): enabling "
            "loop-closure corrections by default (override with "
            "--no-streaming_correction_default_on_for_live).",
            flush=True,
        )
    if _seed_lineage_freeze is not None:
        _seed_lineage_freeze.validate_pose_correction(_lc_apply_correction)
    # Milestone 5: instantiate the PoseCorrector coordinator.
    from eligsir.utils.pose_corrections import PoseCorrector, PoseCorrectorConfig, CorrectionMode
    _correction_mode_str = str(getattr(args, "streaming_correction_mode", "medium"))
    try:
        _correction_mode = CorrectionMode(_correction_mode_str)
    except ValueError:
        _correction_mode = CorrectionMode.MEDIUM
    _pose_corrector = PoseCorrector(
        config=PoseCorrectorConfig(
            enabled=bool(_lc_apply_correction),
            default_mode=_correction_mode,
            max_submaps_per_event=int(getattr(args, "streaming_correction_max_submaps_per_event", 0)),
            min_iter_dt_s=float(getattr(args, "streaming_correction_min_dt_s", 0.0)),
        ),
    )

    # --- Loop force-field diagnostic interval (Phase 11) -------------------
    _loop_ff_interval = int(getattr(args, "loop_force_field_interval", 0))
    _loop_ff_radius = float(getattr(args, "loop_force_field_match_radius", 0.1))
    _loop_ff_max_rmse = float(getattr(args, "loop_force_field_max_rmse", 0.05))
    if _loop_closure_enabled and _loop_ff_interval <= 0:
        print(
            "[loop-closure] WARNING: loop_closure_enabled=True but loop_force_field_interval=0 "
            "(force-field diagnostic disabled). Loop edges will never be discovered. "
            "Set --loop_force_field_interval (e.g. 500) to enable loop detection.",
            flush=True,
        )

    optimizer_type = getattr(opt, "optimizer_type", "adam").lower()
    sh_update_interval = max(1, int(getattr(opt, "sh_update_interval", 1)))
    sparse_active_set = optimizer_type == "selective_adam" and not getattr(
        args, "selective_adam_allow_dense_grads", False
    )
    resolve_streaming_renderer_configuration(pipe, opt)
    print(
        f"[parallelism] profile={getattr(opt, 'parallelism_profile', 'off')} "
        f"optimizer={optimizer_type} sparse_grad={pipe.gsplat_sparse_grad} "
        f"sh_update_interval={sh_update_interval}"
        + (
            " [EXPLICIT: this run's --sh_update_interval overrode the 'safe' "
            "profile default of 16 -- see issue #239]"
            if getattr(args, "_sh_update_interval_override_warned", False)
            else ""
        ),
        flush=True,
    )

    # --- Background --------------------------------------------------------
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    def _pose_delta_from_last_train(frame) -> tuple[float, float, int]:
        if not streaming_scene.train_cameras:
            return 0.0, 0.0, 0
        last_frame = getattr(streaming_scene.train_cameras[-1], "_streaming_frame", None)
        if last_frame is None:
            return 0.0, 0.0, 0
        t_m = float(np.linalg.norm(frame.c2w[:3, 3] - last_frame.c2w[:3, 3]))
        r_rel = last_frame.c2w[:3, :3].T @ frame.c2w[:3, :3]
        cos_angle = np.clip((np.trace(r_rel) - 1.0) * 0.5, -1.0, 1.0)
        r_deg = float(np.degrees(np.arccos(cos_angle)))
        gap = int(frame.index - last_frame.index)
        return t_m, r_deg, gap

    # Milestone 4: centralized keyframe admission (replaces the inline
    # _hybrid_keyframe_admission closure). Keeps the same two-stage
    # gate (cheap pose-delta, expensive render overlap) and emits a
    # typed KeyframeDecision.  The admitter's keyframe_indices list is
    # exposed via ``_keyframe_admitter.keyframe_indices`` for any
    # scheduler / sampler that wants to prioritise keyframes.
    from eligsir.mapping.scheduler import (
        KeyframeAdmitter,
        keyframe_admitter_config_from_args,
    )

    # Causal (frames-seen-so-far only) tracker for the adaptive blur/exposure
    # keyframe-quality path -- see FrameQualityConfig.adaptive_quality's
    # docstring. Lives on the streaming_training closure (one per run), not
    # on streaming_scene, since it is pure admission-time bookkeeping with no
    # bearing on scene state.
    _quality_running_stats = RunningBlurExposureStats(
        alpha=float(getattr(args, "streaming_keyframe_quality_ema_alpha", 0.05)),
        warmup=int(getattr(args, "streaming_keyframe_quality_warmup", 20)),
    )
    # Tracking-confidence signal (2026-07-30 design). Instantiated
    # unconditionally (cheap) but only consulted when
    # --streaming_tracking_confidence_enabled AND a frame actually carries a
    # ``_tracking_status`` attribute (set by eligsir/io/sources/
    # ros2_live.py when the ros2://live source is a patched ORB-SLAM3
    # rgbd_node_cpp / rgbd_inertial_node_cpp build, eligsir-orbslam3
    # commit b3c7153, feat/slam-pose-convention branch).
    from eligsir.mapping.scheduler.tracking_confidence import (
        RunningTrackingStats as _RunningTrackingStats,
        tracking_confidence_weight as _tracking_confidence_weight,
    )
    _tracking_running_stats = _RunningTrackingStats(
        alpha=float(getattr(args, "streaming_tracking_confidence_ema_alpha", 0.05)),
        warmup=int(getattr(args, "streaming_tracking_confidence_warmup", 20)),
    )
    # Mutable holder for the per-call tracking-confidence weight, read back by
    # _keyframe_admission_closure right after admit() returns (synchronous,
    # single-threaded -- admit() calls evaluate_quality exactly once per
    # candidate frame, so this always reflects the frame just scored).
    # Added 2026-08-02 per advisor: the composite quality_weight in the
    # admission log conflates blur/exposure and tracking-confidence
    # contributions -- without this field there is no way to tell whether a
    # low quality_weight came from tracking loss or from blur, which matters
    # for attributing any PSNR delta on the RealSense degraded-tracking scene
    # to the tracking-confidence mechanism specifically.
    _last_tracking_weight = {"value": None}

    # Map-reinitialization hard veto (Phase 2c, 2026-08-06). Distinct from
    # (and independent of) --streaming_tracking_confidence_enabled's soft
    # weighting above: ORB-SLAM3's Atlas creates a brand-new, independently
    # ORIGINATED map whenever tracking is fully lost and later reacquired
    # without relocalizing into the existing map (state cycles back to
    # NOT_INITIALIZED/NO_IMAGES_YET *after* tracking had already reached OK
    # at least once, as opposed to the expected one-time startup lifecycle).
    # A second map's poses have NO DEFINED RELATIONSHIP to the first map's
    # coordinate frame -- concatenating them into the trajectory as if
    # continuous silently corrupts it (root-caused 2026-08-06: a real floor2
    # session showed exactly this, `orbslam_live.log`'s "Creation of new map
    # with id: 1" after "TRACK_REF_KF: Less than 15 matches!!", producing a
    # trajectory whose raw, pre-loop-closure shape was mirrored relative to
    # a clean session). This is a data-integrity issue, not a
    # confidence/quality tuning knob, so it applies unconditionally whenever
    # a frame carries `_tracking_status` -- independent of
    # --streaming_tracking_confidence_enabled.
    from eligsir.mapping.scheduler.tracking_confidence import (
        MapReinitTracker,
        OrbslamLogReinitWatcher,
    )
    _reinit_tracker = MapReinitTracker(
        settle_frames=int(getattr(args, "streaming_tracking_reinit_settle_frames", 30))
    )
    # Fallback log-tail detector (2026-08-06) -- see OrbslamLogReinitWatcher's
    # own docstring for why this is needed independent of the
    # /slam/tracking_status_stamped-based path above: that topic is
    # confirmed silently absent on at least one deployed eligsir-orbslam3
    # image build, and a real floor2 re-init produced zero warning anywhere
    # until this fallback existed. Always constructed (cheap: no-ops until
    # the log file actually appears); only ever matters for
    # --ros2_live_bag_replay / live-camera runs where orbslam_live.log exists.
    _orbslam_log_watcher = OrbslamLogReinitWatcher(
        os.path.join(args.model_path, "orbslam_live.log")
    )

    def _check_tracking_reinit(frame) -> None:
        """Feed `frame._tracking_status` into `_reinit_tracker`, AND poll the
        log-tail fallback detector. Must run on EVERY ingested frame (not
        just ones that reach quality scoring), since a re-init can happen on
        a frame the motion pregate or other early checks would otherwise
        short-circuit past.

        CALL SITE (corrected #806 follow-up, 2026-08-15): this is called from
        `_integrity_observer_closure` ONLY, which `StreamingScene.ingest_next_frame`
        invokes as its `integrity_observer_fn` -- unconditionally, for every
        materialized frame, in BOTH `--streaming_frame_admission all` and
        `hybrid_keyframe`, and strictly before the odom-validity check,
        `pregate_fn`, `admission_fn`, and any Camera construction. This
        docstring previously said "called from `_keyframe_admission_closure`
        before any other admission logic"; that call site was REMOVED by #806
        (PR #817) precisely because it sat *downstream* of the motion pregate
        and so could be short-circuited past -- the exact bug the sentence
        above warns about. There is deliberately no second, defense-in-depth
        call: the ordering invariant is pinned instead by the AST regression
        tests in `tests/test_issue806_integrity_observer_ordering.py`
        (`test_ast_guard_integrity_observer_precedes_pregate_and_admission_calls`
        and `test_integrity_observer_wiring_is_unconditional_on_admission_mode`),
        which fail loudly if either half is reordered or unwired. If you add a
        NEW `ingest_next_frame` call site, it must pass `integrity_observer_fn`
        or it silently has no map-reinit veto.

        The state machines themselves live
        in MapReinitTracker/OrbslamLogReinitWatcher (unit-tested there); this
        wrapper only owns logging, which needs run-local context (frame
        index) those classes deliberately don't know about."""
        tstatus = getattr(frame, "_tracking_status", None)
        _frame_idx = int(getattr(frame, "index", -1))
        if tstatus is not None:
            _event = _reinit_tracker.observe(tstatus, _frame_idx)
            if _event == "reinit_started":
                # PERMANENT veto (2026-08-06 correction): MapReinitTracker no
                # longer auto-resumes admission after a settle window -- see
                # its docstring for why (settle_frames only proves the NEW
                # Atlas map tracks cleanly, not that it shares a coordinate
                # frame with the old one). This run's admission stays
                # vetoed for the rest of the run, full stop.
                print(
                    f"[tracking] SLAM map RE-INITIALIZED at frame {_frame_idx} "
                    f"-- admission is now PERMANENTLY VETOED for the rest of "
                    f"this run. This frontend session's poses from this "
                    f"point forward have no defined relationship to the "
                    f"earlier trajectory's coordinate frame, and there is no "
                    f"verified way to re-align them. This run's results must "
                    f"be treated as INVALID.",
                    flush=True,
                )
        if _orbslam_log_watcher.check():
            _reinit_tracker.vetoing = True
            _reinit_tracker.ever_ok = True
            _reinit_tracker.events.append({
                "frame": _frame_idx, "recovered_at_frame": None,
                "source": "orbslam_log_tail",
            })
            print(
                f"\n{'!' * 88}\n"
                f"[tracking] ⚠⚠⚠ SLAM MAP RE-INITIALIZATION DETECTED (orbslam_live.log, "
                f"{_orbslam_log_watcher.map_creation_count} 'Creation of new map' events, "
                f"expected exactly 1) at frame {_frame_idx}.\n"
                f"This frontend session's poses from this point forward have NO "
                f"defined relationship to the earlier trajectory's coordinate "
                f"frame. Admission is now PERMANENTLY VETOED for the rest of this "
                f"run (detected via log-tail fallback, not the tracking-status "
                f"topic -- no reliable single-line 'recovered' signal exists in "
                f"the log, so this does not auto-resume like the topic-based path "
                f"can). This run's results must be treated as INVALID.\n"
                f"{'!' * 88}\n",
                flush=True,
            )
            try:
                import json as _json_reinit
                with open(
                    os.path.join(args.model_path, "orbslam_map_reinit_detected.json"), "w"
                ) as _f:
                    _json_reinit.dump({
                        "map_creation_count": _orbslam_log_watcher.map_creation_count,
                        "detected_at_frame": _frame_idx,
                        "detected_live": True,
                        "orbslam_live_log": _orbslam_log_watcher.log_path,
                    }, _f, indent=2)
            except OSError:
                pass

    def _evaluate_quality_and_tracking(rgb_np, depth_np, frame=None):
        """Composed keyframe-quality signal: blur/exposure (+ optional causal
        adaptive baseline) folded with tracking-confidence downweighting.

        Passed as KeyframeAdmitter(evaluate_quality=...) below -- this IS on
        the live admission call path (keyframes.py's `admit()`, gated on
        `self._evaluate_quality is not None`).

        FIXED 2026-07-31: this composed scoring used to live in a
        `_hybrid_keyframe_admission` closure that Milestone 4 (commit
        cb5bf83/251e59a, 2026-06-13) replaced with KeyframeAdmitter for the
        pose/overlap gate, but left un-migrated and dead -- so quality/
        tracking scoring silently had ZERO effect on any real run since
        2026-06-13 despite `--streaming_keyframe_quality_enabled` appearing
        to work. See specs/*_kfthreshold_*.json and specs/*adaptive_quality*
        .json notes for the retraction of any prior finding that relied on
        it. Do not reintroduce a second, unwired copy of this logic --
        `evaluate_quality=` below is the only call site that matters.
        """
        _adaptive_q = bool(getattr(args, "streaming_keyframe_quality_adaptive", False))
        quality = evaluate_frame_quality(
            rgb_np,
            depth_np,
            config=FrameQualityConfig(
                min_blur_laplacian=float(getattr(args, "streaming_keyframe_quality_min_blur", 8.0)),
                max_clip_ratio=float(getattr(args, "streaming_keyframe_quality_max_clip_ratio", 0.25)),
                min_depth_valid_ratio=float(getattr(args, "streaming_keyframe_quality_min_depth_valid", 0.10)),
                depth_min=float(getattr(args, "streaming_min_depth", 0.1)),
                depth_max=float(getattr(args, "streaming_max_depth", 8.0)),
                adaptive_quality=_adaptive_q,
                blur_floor=float(getattr(args, "streaming_keyframe_quality_blur_floor", 20.0)),
                exposure_tolerance=float(getattr(args, "streaming_keyframe_quality_exposure_tolerance", 0.35)),
            ),
            running_blur_typical=_quality_running_stats.blur_typical if _adaptive_q else None,
            running_luma_mean=_quality_running_stats.luma_mean if _adaptive_q else None,
        )
        if _adaptive_q:
            # Update AFTER scoring this frame so its own value never
            # influences the comparison it was just judged against --
            # keeps the tracker strictly causal (frames seen before
            # this one only).
            _quality_running_stats.update(quality.blur_laplacian, mean_luma(rgb_np))
        _last_tracking_weight["value"] = None
        if bool(getattr(args, "streaming_tracking_confidence_enabled", False)):
            _tstatus = getattr(frame, "_tracking_status", None)
            if _tstatus is not None:
                _tw = _tracking_confidence_weight(
                    _tstatus,
                    running_typical=_tracking_running_stats.typical,
                    floor_weight=float(getattr(args, "streaming_tracking_confidence_floor", 0.4)),
                )
                _tracking_running_stats.update(_tstatus.num_tracked)
                _last_tracking_weight["value"] = float(_tw)
                if _tw < 1.0:
                    import dataclasses as _dc_tw
                    quality = _dc_tw.replace(quality, quality_weight=quality.quality_weight * _tw)
        return quality

    from eligsir.utils.streaming_frames import load_frame_rgb as _load_frame_rgb_for_admitter

    _keyframe_admitter = KeyframeAdmitter(
        config=keyframe_admitter_config_from_args(args),
        # ``render`` is the module-level gsplat renderer, bound at
        # module scope; the old ``'render' in dir()`` test never sees
        # it. Bind the actual callable and let the admitter pass the
        # gaussians/pipe/background into the call itself.
        render_fn=render if callable(render) else None,
        gaussians=gaussians,
        pipe=pipe,
        background=background,
        # Track A coverage gate: thread the scene occupancy hash + depth loader.
        # check_occupancy(xyz_np, voxel_size, check_neighbors=True) matches the
        # offline occupancy_fn(pts, voxel_size) signature.
        occupancy_fn=streaming_scene.check_occupancy if hasattr(streaming_scene, "check_occupancy") else None,
        load_depth_m=_load_depth_meters,
        observation_db=_observation_db,
        evaluate_quality=_evaluate_quality_and_tracking,
        load_rgb=lambda f: np.asarray(_load_frame_rgb_for_admitter(f)),
    )

    # Issue #795: instantiate realized admission-policy fingerprint (persists to report.json).
    _admission_fingerprint = AdmissionFingerprint()
    _admission_fingerprint.nominal_policy = str(args.streaming_frame_admission)
    _admission_fingerprint.max_gap = int(getattr(args, "streaming_keyframe_max_gap", 10))
    _admission_fingerprint.target_admit_fps = float(getattr(args, "streaming_keyframe_target_admit_fps", 2.0))
    _admission_fingerprint.input_fps = float(getattr(args, "streaming_keyframe_input_fps", 30.0))
    # Issue #958: surface #794's contradictory-config warnings
    # (KeyframeAdmitter.precedence_config_warnings, populated at construction
    # by validate_backstop_config()) so they are checkable post-hoc from
    # report.json, not just from a live admitter object. Report-only: reads
    # an existing property, changes no admission behaviour. `_keyframe_admitter`
    # is constructed unconditionally above, but guarded with getattr() anyway
    # in case a future admission mode skips construction.
    _precedence_config_warnings = list(
        getattr(_keyframe_admitter, "precedence_config_warnings", []) or []
    )
    _admission_fingerprint.precedence_config_warnings = _precedence_config_warnings
    for _pw in _precedence_config_warnings:
        print(f"[admission] WARNING (#794 precedence): {_pw}")

    def _keyframe_admission_closure(cam, frame):
        """Bridge the inline admission contract to KeyframeAdmitter.admit()."""
        # #289 candidate-evaluation clock: counted HERE, at the real invocation,
        # not derived. Under --streaming_frame_admission all (the default) this
        # closure is never wired at all, so candidate_evaluated stays 0 -- which
        # means "no admission gate ran", not "instrumentation broken". The
        # report block records frame_admission alongside it so the zero is
        # readable.
        #
        # #806 correction (2026-08-15): the tracking-reinit hard veto used to
        # be checked HERE (`_check_tracking_reinit(frame)` +
        # `_reinit_tracker.vetoing`), which meant a frame the motion pregate
        # rejected first (`_keyframe_pregate_closure`, called BEFORE this
        # closure) never reached this check at all -- the exact ordering bug
        # #806 fixed. The veto is now applied earlier, in
        # `StreamingScene.ingest_next_frame` itself, via the mandatory
        # `integrity_observer_fn` stage (`_integrity_observer_closure` below)
        # that runs before `pregate_fn` for every candidate. A frame that
        # reaches this closure at all has therefore already been proven NOT
        # vetoed for this call -- `_reinit_tracker.vetoing` is intentionally
        # not re-checked here (see `_integrity_observer_closure`'s docstring
        # for the single-call-site rationale and idempotency argument; kept
        # single rather than duplicated to avoid double-touching
        # `MapReinitTracker`'s informational `ok_streak` bookkeeping and
        # double-reading `OrbslamLogReinitWatcher`'s log tail per frame).
        _cadence.record_candidate_evaluated()
        # Issue #795: record fingerprint candidate evaluation count.
        _admission_fingerprint.record_candidate()
        decision, render_pkg = _keyframe_admitter.admit(cam, frame)
        # Issue #795 controller-realization group (deferred until #793 landed):
        # record the velocity-controller state this candidate was judged under.
        # Recording-only; never influences the decision above.
        _admission_fingerprint.record_controller_state(decision)
        _admission_fingerprint.record_decision_observation(decision)
        # Issue #823 (ex-#947): aggregate the #946 per-decision backstop /
        # motion-ownership state so the run's own report.json can state its
        # classical-treatment calibration verdict, instead of that verdict
        # being derivable only by hand-parsing keyframe_admission_log.jsonl.
        # Deliberately NOT called on the `_keyframe_pregate_closure`
        # rejection path below: a pregate reject never reaches
        # `record_candidate()` either, so counting it here would give the
        # candidate-side motion histogram a different denominator from
        # `candidates_evaluated`.
        _admission_fingerprint.record_backstop_state(decision, bool(decision.admitted))
        # Issue #795: record fingerprint for admission decision.
        if decision.admitted:
            _admission_fingerprint.record_admit(
                reason=decision.reason,
                gap=decision.gap,
                forced_hard=(decision.forced and decision.reason == "max_gap"),
                forced_soft=(decision.forced and decision.reason != "max_gap"),
            )
        else:
            _admission_fingerprint.record_reject(
                decision.reason, population="candidate_evaluated"
            )
        # Match the legacy return contract used by StreamingScene.ingest_next_frame:
        #   (admitted: bool, stats: dict, render_pkg: optional)
        # Kept backwards-compatible with the existing call site in
        # the drain loop — and we propagate the render package so the
        # depth-consistency / training path can reuse it instead of
        # rendering the same camera twice.
        stats = {
            "mode": args.streaming_frame_admission,
            "admitted": int(decision.admitted),
            "reason": decision.reason,
            "score": decision.score,
            "cheap_score": decision.cheap_score,
            "pose_delta_m": float(decision.pose_delta_m),
            "pose_delta_deg": float(decision.pose_delta_deg),
            "gap": int(decision.gap),
            "overlap": decision.render_overlap or 0.0,
            "forced": int(decision.forced),
            # #934: the novelty-DRIVEN trigger. Deliberately a SEPARATE field
            # from "forced" (which drives force_share, itself a #934 Stage 1a
            # pass/fail gate) -- see KeyframeDecision.novelty_forced.
            "novelty_forced": int(decision.novelty_forced),
            "moved": int(decision.moved),
            "novel": int(decision.novel),
            **keyframe_decision_sidecar_fields(decision),
        }
        # Issue #794: per-decision precedence reason chain (candidate_trigger /
        # bypassed_gates / evaluated_gates / final_owner / hard_force). Recorded
        # for EVERY decision, including under the default legacy scope where the
        # bypass itself is unchanged -- the point is that a bypass can no longer
        # happen invisibly. Aggregated counters live on the admitter itself
        # (`precedence_bypass_counts` / `precedence_trigger_counts`).
        if decision.trace is not None:
            stats["precedence"] = decision.trace
            # #795's `gate_bypasses` field existed but nothing ever populated
            # it -- the information simply did not exist before #794. Feed it
            # the REAL bypasses (enabled gates only; permitted-but-disabled
            # gates would inflate the count without a quality check having
            # been skipped), so report.json's admission_fingerprint answers
            # "which live gates did a force actually skip this run?".
            for _bypassed_gate in decision.trace.get("bypassed_gates", ()):
                _admission_fingerprint.record_gate_bypass(_bypassed_gate)
            # Issue #928: tri-state novelty-gate outcome for this decision
            # (no-op for "not_applicable" candidates -- see
            # AdmissionFingerprint.record_novelty_gate's docstring).
            _admission_fingerprint.record_novelty_gate(decision.trace)
            # Issue #934: the novelty-DRIVEN trigger's own tally (separate
            # mechanism, separate denominator -- see record_novelty_force).
            _admission_fingerprint.record_novelty_force(
                decision.trace, bool(decision.admitted)
            )
        # FIXED 2026-07-31: decision.quality (KeyframeDecision.quality, set
        # from quality.quality_weight at keyframes.py ~1002) was never copied
        # into stats, so the persisted keyframe_admission_log.jsonl could
        # never show quality_weight even once the evaluate_quality wiring
        # itself was fixed (the persistence code below only ever reads
        # `quality_weight` out of this dict).
        _qw = _admission_quality_weight_for_stats(
            decision, bool(getattr(args, "streaming_keyframe_quality_enabled", False))
        )
        if bool(getattr(args, "streaming_keyframe_quality_enabled", False)):
            stats["quality_weight"] = _qw
        # Dedicated tracking-confidence weight, separate from the composite
        # quality_weight above (see _last_tracking_weight comment at its
        # definition). None when tracking-confidence is disabled or the
        # frame carried no _tracking_status (e.g. non-ros2 sources).
        if bool(getattr(args, "streaming_tracking_confidence_enabled", False)):
            stats["tracking_weight"] = _last_tracking_weight["value"]
        # W3-06: shadow frame-utility scoring. Runs strictly AFTER
        # `decision`/`stats`/`render_pkg` above are already final -- reads
        # them, never writes back into them, and its return value is
        # discarded (not part of this function's return contract). A true
        # no-op when disabled (ShadowUtilityPolicy.observe_candidate's first
        # line is `if not self._enabled: return None`). See
        # eligsir/mapping/admission/shadow_policy.py.
        _utility_score = None
        if _shadow_utility_policy.enabled:
            try:
                _gt_image = getattr(cam, "original_image", None)
                _last_summary = _shadow_utility_policy.last_summary
                _utility_score = _shadow_utility_policy.observe_candidate(
                    frame_id=int(getattr(frame, "index", -1)),
                    camera_key=id(cam),
                    admitted=bool(decision.admitted),
                    iteration=int(iteration),
                    pose_confidence=float(_last_tracking_weight["value"] or 1.0),
                    image_quality=float(_qw if bool(getattr(args, "streaming_keyframe_quality_enabled", False)) else 1.0),
                    depth_valid_fraction=float(getattr(decision, "depth_valid_fraction", 1.0) or 1.0),
                    n_gaussians=int(gaussians.get_xyz.shape[0]),
                    resolution_bucket=str(getattr(_resolution_controller, "_tier", "default")),
                    render_pkg=render_pkg,
                    gt_image=_gt_image,
                    redundancy=float(decision.render_overlap or 0.0),
                    useful_parallax=float(decision.pose_delta_deg) / 180.0,
                    new_surface=1.0 - float(decision.render_overlap or 0.0),
                    under_supported_mass=float(_last_summary.under_supported_mass) if _last_summary is not None else 0.0,
                    topology_version=int(getattr(gaussians, "topology_version", 0)),
                    map_version=int(iteration),
                )
                _record_deferred = getattr(
                    _keyframe_admitter,
                    "record_deferred_shadow_observation",
                    None,
                )
                if _utility_score is not None and _record_deferred is not None:
                    _record_deferred(
                        decision,
                        score_map_version=_utility_score.meta.map_version,
                        score_topology_version=_utility_score.meta.topology_version,
                        predicted_utility=_utility_score.utility,
                        under_supported_mass_before=(
                            float(_last_summary.under_supported_mass)
                            if _last_summary is not None
                            else 0.0
                        ),
                    )
                _drain_cached = getattr(
                    _keyframe_admitter,
                    "drain_cached_admissions",
                    None,
                )
                if _drain_cached is not None:
                    for _cached_admission in _drain_cached():
                        _shadow_utility_policy.register_cached_admission(
                            camera_key=id(_cached_admission.camera),
                            frame_id=_cached_admission.frame_id,
                            admitted_iteration=int(iteration),
                            score_map_version=_cached_admission.score_map_version,
                            score_topology_version=_cached_admission.score_topology_version,
                            predicted_utility=_cached_admission.predicted_utility,
                            under_supported_mass_before=(
                                _cached_admission.under_supported_mass_before
                            ),
                        )
            except Exception as _shadow_exc:  # noqa: BLE001 -- shadow scoring must never break real admission
                print(f"[shadow-utility] observe_candidate failed (non-fatal, shadow-only): {_shadow_exc}", flush=True)
        # Wave 3, PR W3-07: consult the graduated rollout policy. A true
        # no-op in RolloutMode.SHADOW (the wired-in default) -- decide()'s
        # SHADOW branch always returns admitted == baseline_admitted with
        # overridden=False, so `decision.admitted` is never mutated below
        # for any run that hasn't explicitly opted into a non-shadow mode.
        # The hard-safety-gate veto (`_reinit_tracker.vetoing`, applied in
        # `_integrity_observer_closure`/`StreamingScene.ingest_next_frame`
        # -- see #806) already short-circuited before this closure was even
        # called, so `hard_gate_blocked=False` here is correct -- a real
        # hard-gate rejection never reaches this call in the first place;
        # see rollout_policy.py's module docstring for why this policy ALSO
        # defends against hard_gate_blocked defensively despite that.
        try:
            _rollout_decision = _admission_rollout_policy.decide(
                frame_id=int(getattr(frame, "index", -1)),
                baseline_admitted=bool(decision.admitted),
                hard_gate_blocked=False,
                utility_score=_utility_score,
                predicted_cost=predict_gpu_cost(
                    int(gaussians.get_xyz.shape[0]), resolution_bucket=str(getattr(_resolution_controller, "_tier", "default"))
                ),
            )
            if _rollout_decision.overridden:
                decision.admitted = _rollout_decision.admitted
                stats["admitted"] = int(decision.admitted)
            if _admission_rollout_policy.mode != RolloutMode.SHADOW:
                stats["rollout_mode"] = _admission_rollout_policy.mode.value
                stats["rollout_reason"] = _rollout_decision.reason_code
                stats["rollout_overridden"] = int(_rollout_decision.overridden)
                stats["rollout_utility"] = _rollout_decision.utility
        except Exception as _rollout_exc:  # noqa: BLE001 -- rollout policy must never crash real admission
            print(f"[admission-rollout] decide failed (non-fatal, baseline decision kept): {_rollout_exc}", flush=True)
        return bool(decision.admitted), stats, render_pkg

    def _keyframe_pregate_closure(frame):
        """Cheap no-cam pre-check bridging to KeyframeAdmitter.pregate().

        Only ever wired alongside `_keyframe_admission_closure` (hybrid_keyframe
        admission mode) -- never under admission=all, where every frame must be
        admitted unconditionally to keep the full-coverage guarantee that mode
        relies on (see Run Contracts: "Coverage gap only applies to gated
        admission"). Returns the same stats-dict shape as
        `_keyframe_admission_closure`'s rejection path (admitted=0,
        reason="no_motion") when the frame can already be rejected without a
        camera, else None to mean "build the camera and go through
        `_keyframe_admission_closure` as normal."
        """
        decision = _keyframe_admitter.pregate(frame)
        if decision is None:
            return None
        # Issue #795: record fingerprint for pregate rejection.
        _admission_fingerprint.record_reject(
            decision.reason, population="prefilter_evaluated"
        )
        _admission_fingerprint.record_controller_state(decision)
        _admission_fingerprint.record_decision_observation(decision)
        return {
            # Issue #794 precedence reason chain (see the admit closure above).
            **({"precedence": decision.trace} if decision.trace is not None else {}),
            "mode": args.streaming_frame_admission,
            "admitted": int(decision.admitted),
            "reason": decision.reason,
            "score": decision.score,
            "cheap_score": decision.cheap_score,
            "pose_delta_m": float(decision.pose_delta_m),
            "pose_delta_deg": float(decision.pose_delta_deg),
            "gap": int(decision.gap),
            "overlap": decision.render_overlap or 0.0,
            "forced": int(decision.forced),
            # #934: the novelty-DRIVEN trigger. Deliberately a SEPARATE field
            # from "forced" (which drives force_share, itself a #934 Stage 1a
            # pass/fail gate) -- see KeyframeDecision.novelty_forced.
            "novelty_forced": int(decision.novelty_forced),
            "moved": int(decision.moved),
            "novel": int(decision.novel),
            **keyframe_decision_sidecar_fields(decision),
        }

    # Issue #934: `fixed_stride` admission is the principal experimental
    # CONTROL arm for the hybrid_keyframe smart-selector comparison. The
    # existing `--streaming_source_stride`/`--streaming_frame_stride` knobs
    # are NOT a valid control -- they decimate the frame SOURCE before the
    # candidate population exists, changing the very
    # "candidate population / source-decimation stack" axis #934 requires
    # held constant between arms. This closure instead sits at the exact
    # same admission_fn call site hybrid_keyframe uses (wired below,
    # `StreamingScene.ingest_next_frame`'s `admission_fn`), so both arms
    # share the population of frames reaching `ingest_next_frame` -- every
    # non-holdout, non-guard materialized frame -- and commits every Nth
    # one, deterministically, with zero information about the frame's
    # content, pose, or motion. That shared population is NOT the same as
    # an identical post-filter candidate stream: hybrid_keyframe additionally
    # wires a `pregate_fn` (motion pre-check, on by default; see the
    # dispatch site below) that runs before its admission_fn and drops
    # stale-pose/no-motion frames first, while fixed_stride wires no
    # `pregate_fn` and sees the full pre-pregate stream. The motion pregate
    # is deliberately part of hybrid_keyframe's own POLICY, not part of the
    # shared population -- so `--streaming_fixed_stride_n` (default 10) is a
    # starting point, not a value that can be assumed to match
    # hybrid_keyframe's realized admit rate; tune it empirically against
    # that run's actual committed-K, and always name which population
    # (pre- vs. post-pregate) any per-candidate admission rate is computed
    # over.
    from eligsir.mapping.admission.fixed_stride import FixedStrideAdmitter

    _fixed_stride_admitter = FixedStrideAdmitter(
        stride_n=int(getattr(args, "streaming_fixed_stride_n", 10))
    )

    # Issue #600: production soft-all. The backlog signal is the authoritative
    # optimizer-step clock (#289's denominator), NOT `iteration` -- the two
    # diverge under gradient accumulation and the diagnostic training modes, and
    # measuring the real one is precisely what turns "effective_spf=2 implies 2
    # updates per admitted frame" from an assumption into an observation.
    from eligsir.mapping.admission.observation_decision import (
        REASON_FRONTIER_FORCED as _OBS_REASON_FRONTIER_FORCED,
        SoftAllAdmitter,
    )

    # Issue #601 single-owner reconciliation: SoftAllAdmitter (#600) and the
    # bounded backpressure owner are both capable of refusing a frame for
    # capacity. #601 requires exactly ONE execution owner, so when the
    # backpressure owner is active SoftAllAdmitter's capacity bound is
    # disabled (`min_steps_per_admit=0`) and it degrades to a pass-through
    # admitter -- a signal provider, exactly as BurstBudget does.
    #
    # This is the right direction rather than the reverse because the owner
    # decides BEFORE materialization (the whole point of #601), whereas
    # `admission_fn` runs after the camera is already built: leaving both
    # armed would let a frame the owner already paid to materialize be
    # thrown away by a second, later capacity judgement.
    #
    # force_all_stress is excluded: that mode exists to reproduce historical
    # behaviour, so it must not quietly alter SoftAllAdmitter either.
    _soft_all_min_steps = int(
        getattr(args, "streaming_soft_all_min_steps_per_admit", 1)
    )
    if (
        _backpressure_owner is not None
        and _backpressure_cfg is not None
        and not _backpressure_cfg.force_all_stress
        and _soft_all_min_steps > 0
    ):
        print(
            "[backpressure] SoftAllAdmitter capacity bound disabled "
            f"(min_steps_per_admit {_soft_all_min_steps} -> 0): the bounded "
            "backpressure owner is the single capacity/skip owner and decides "
            "before materialization.",
            flush=True,
        )
        _soft_all_min_steps = 0

    _soft_all_admitter = SoftAllAdmitter(
        min_steps_per_admit=_soft_all_min_steps,
        max_consecutive_skips=int(
            getattr(args, "streaming_soft_all_max_consecutive_skips", 4)
        ),
        steps_provider=lambda: _cadence.optimizer_steps,
    )
    # #600 invariant counter: a frame skipped for mapping supervision must never
    # reach Phase 2 depth insertion. This must read exactly 0 in every run; it is
    # the machine-readable form of the requirement, and what makes the claim
    # provable from artifacts rather than by reading the control flow.
    _soft_all_invariant = {"inserted_from_skipped_frame": 0}

    def _fixed_stride_admission_closure(cam, frame):
        """Bridge the inline admission_fn contract to FixedStrideAdmitter.

        Admits iff this is the Nth candidate (N =
        `--streaming_fixed_stride_n`, default 10) since the last admit,
        counting ordinally over the population of frames reaching
        `ingest_next_frame` -- the same population `_keyframe_admission_closure`
        is invoked over under hybrid_keyframe. No `pregate_fn` is wired for
        this mode (see the dispatch site below), so every candidate that
        reaches here is unfiltered; hybrid_keyframe, by contrast, wires a
        motion `pregate_fn` (on by default) that drops stale-pose/no-motion
        frames BEFORE `_keyframe_admission_closure` sees them. So this is
        the shared population the two arms are meant to hold constant, not
        an identical post-filter candidate stream -- the motion pregate is
        part of hybrid_keyframe's own admission policy, not part of that
        shared population. `stride_n` (default 10) is therefore a starting
        point, not a value guaranteed to reproduce hybrid_keyframe's
        realized admit rate; tune it empirically against that run's actual
        committed-K, and when reporting a per-candidate admission rate
        alongside these set-comparison gates, name which population
        (pre- vs. post-pregate) it was computed over.
        The first candidate (ordinal 0) is always admitted so bootstrap has
        an initial frame.

        Deliberately does NOT: run the admission render/overlap gate
        (`render_pkg` is always None -- no render call is made), consult
        motion thresholds, consult max_gap/starvation, or consult novelty.
        Being information-free about the frame is the entire point of a
        control arm -- any quality delta measured against `hybrid_keyframe`
        at the same realized admission rate is then attributable to
        hybrid_keyframe's actual selectivity, not merely to "admitting
        fewer frames."
        """
        decision = _fixed_stride_admitter.admit()
        # #289 candidate-evaluation clock + #795 fingerprint candidate count,
        # mirrored from `_keyframe_admission_closure` for telemetry parity.
        _cadence.record_candidate_evaluated()
        _admission_fingerprint.record_candidate()
        # Distinct, unambiguous owner/reason strings (never "max_gap"/
        # "forced") so a run's admission-owner shares can't be misread as
        # hybrid_keyframe's own mechanisms firing.
        if decision.admitted:
            _admission_fingerprint.record_admit(
                reason=decision.reason,
                gap=decision.stride_n,
                forced_hard=False,
                forced_soft=False,
            )
        else:
            _admission_fingerprint.record_reject(
                decision.reason, population="candidate_evaluated"
            )
        # Issue #823: the control arm carries none of the #946 backstop
        # fields, so this contributes only its trigger bucket -- which is
        # exactly what the gate's `control_purity` check needs (a
        # `fixed_stride` arm whose admissions are owned by anything else has
        # leaked, and that is only visible if the arm is graded at all).
        _admission_fingerprint.record_backstop_state(decision, bool(decision.admitted))
        stats = {
            "mode": args.streaming_frame_admission,
            "admitted": int(decision.admitted),
            "reason": decision.reason,
            "score": None,
            "cheap_score": None,
            "pose_delta_m": 0.0,
            "pose_delta_deg": 0.0,
            "gap": int(decision.stride_n),
            "overlap": 0.0,
            "forced": 0,
            "moved": 0,
            "novel": 0,
            "candidate_ordinal": int(decision.ordinal),
            "fixed_stride_n": int(decision.stride_n),
        }
        return bool(decision.admitted), stats, None

    def _soft_all_admission_closure(cam, frame):
        """Bridge the admission_fn contract to SoftAllAdmitter (issue #600).

        Production ``all``. Admits every observation the mapper can keep up
        with, and skips SUPERVISION (never observation, never pose/history) for
        a frame arriving while the previously admitted frame has not yet
        received ``--streaming_soft_all_min_steps_per_admit`` real optimizer
        steps.

        A skipped frame takes ``ingest_next_frame``'s existing reject path, so
        it still records pose/trajectory/history and returns ``is_train=False``
        -- which is what makes it ineligible for Phase 2 depth insertion. That
        reuse, rather than a new code path, is what enforces #600's "a frame
        rejected for mapping cannot accidentally be inserted from depth".

        Deliberately information-free about the frame's CONTENT: soft-all is a
        capacity policy, not a selector. Content-aware skipping is
        ``hybrid_keyframe``'s job, and a richer backlog controller is #601's.
        """
        decision = _soft_all_admitter.decide()
        _cadence.record_candidate_evaluated()
        _admission_fingerprint.record_candidate()
        if decision.admitted:
            _admission_fingerprint.record_admit(
                reason=decision.reason,
                gap=0,
                forced_hard=False,
                # A frontier force-admit is a real soft-admit override; record
                # it as such so the fingerprint can't read as an unforced admit.
                forced_soft=(decision.reason == _OBS_REASON_FRONTIER_FORCED),
            )
        else:
            _admission_fingerprint.record_reject(decision.reason)
        stats = decision.as_stats(str(args.streaming_frame_admission))
        stats.update({
            "admitted": int(decision.admitted),
            "score": None,
            "cheap_score": None,
            "pose_delta_m": 0.0,
            "pose_delta_deg": 0.0,
            "gap": 0,
            "overlap": 0.0,
            "forced": int(decision.reason == _OBS_REASON_FRONTIER_FORCED),
            "moved": 0,
            "novel": 0,
        })
        return bool(decision.admitted), stats, None

    def _integrity_observer_closure(frame):
        """Mandatory integrity observation (#806), wired as
        `StreamingScene.ingest_next_frame`'s `integrity_observer_fn`.

        Runs on EVERY materialized/ingested frame -- wired unconditionally
        below, for BOTH `--streaming_frame_admission all` and
        `hybrid_keyframe` -- BEFORE `pregate_fn`/`admission_fn` get a chance
        to short-circuit past it. This closes the real ordering bug #806
        found: `_check_tracking_reinit` used to be called only from
        `_keyframe_admission_closure`, which `_keyframe_pregate_closure`
        (the motion pregate) sits in front of and can reject a frame before
        ever reaching -- so a re-init observable only on a pregate-rejected
        frame could be silently missed. See `_check_tracking_reinit`'s own
        docstring for the full mechanism (feeds `frame._tracking_status`
        into `_reinit_tracker`, polls `_orbslam_log_watcher` as an
        independent fallback).

        Returns the SAME "tracking_reinit" rejection-stats dict shape
        `_keyframe_admission_closure` used to build (reusing the existing
        `"tracking_reinit"` reason constant rather than inventing a new
        one) once `_reinit_tracker.vetoing` has gone permanently True,
        else `None`. `ingest_next_frame` gives this precedence over
        `pregate_fn`'s own rejection reason (e.g. "no_motion") for the
        same frame -- see its docstring's `integrity_observer_fn` section.

        Single call site by design (#806 acceptance criteria ask for a
        deliberate, justified choice here, not a default): previously
        `_check_tracking_reinit` was ALSO called from
        `_keyframe_admission_closure`, reached only for frames that were
        NOT vetoed by this earlier stage (a vetoed frame now never reaches
        `_keyframe_admission_closure` at all, since this closure's rejection
        short-circuits before `pregate_fn`/`admission_fn` run). Keeping a
        second call there would therefore only ever re-observe an
        already-non-vetoed frame -- functionally harmless (verified against
        `MapReinitTracker.observe()`: a same-frame double-call while
        `vetoing` is already True only re-touches the now-purely-
        informational `ok_streak` counter and does not re-append to
        `events`; `OrbslamLogReinitWatcher.check()` is a stateful tail-read
        that returns new bytes exactly once per call regardless, so a
        second same-iteration call just reads zero new bytes) but pure
        overhead (a second log-file read + observe() call per admitted
        frame) with no observational benefit, so it was removed rather than
        kept as a redundant defense-in-depth copy.

        Deliberately does NOT touch `_cadence.record_candidate_evaluated()`
        here: that counter is specifically about hybrid_keyframe's
        admission-GATE throughput (see its own comment at
        `_keyframe_admission_closure`), and this closure is wired for
        admission=all too, which has no admission gate at all -- counting
        "candidates" there would be a real, unrequested behavior change for
        that mode. `_admission_fingerprint.record_reject("tracking_reinit")`
        IS still called here (preserved from the removed
        `_keyframe_admission_closure` veto block) so a run's
        `report.json` admission fingerprint keeps recording this rejection
        reason exactly as it did before -- this one is comparatively benign
        under admission=all (a reject-reason tally, not a throughput
        denominator) and only ever fires at all once a run is already
        permanently vetoed, i.e. already invalid by the map-reinit hard
        constraint regardless of what its fingerprint says.
        """
        _check_tracking_reinit(frame)
        if not _reinit_tracker.vetoing:
            return None
        # Only record rejection in admission fingerprint for true candidates under gated admission
        _eval_hold = args.streaming_eval_hold
        _is_candidate = (args.streaming_frame_admission != "all")
        if _eval_hold > 0 and getattr(frame, "index", 0) % _eval_hold == 0:
            _is_candidate = False
        if _is_candidate:
            _admission_fingerprint.record_candidate()
            _admission_fingerprint.record_reject("tracking_reinit")
        return {
            "mode": args.streaming_frame_admission,
            "admitted": 0,
            "reason": "tracking_reinit",
            "score": 0.0,
            "cheap_score": 0.0,
            "pose_delta_m": 0.0,
            "pose_delta_deg": 0.0,
            "gap": 0,
            "overlap": 0.0,
            "forced": 0,
            "moved": 0,
            "novel": 0,
        }

    def _cheap_keyframe_estimate(frame) -> bool:
        """Cheap pose-delta-only keyframe estimate for the burst budget.

        Used by the Milestone 3 BurstBudget to decide whether to skip
        a non-keyframe *before* calling the expensive
        :meth:`_hybrid_keyframe_admission` (which renders the current
        view to estimate overlap). Returns True for any frame whose
        pose delta, rotation delta, or frame-index gap exceeds the
        configured keyframe thresholds.

        Args:
            frame: Candidate :class:`StreamingRGBDFrame`.

        Returns:
            True if the frame is likely a keyframe.
        """
        t_m, r_deg, gap = _pose_delta_from_last_train(frame)
        min_t = float(getattr(args, "streaming_keyframe_min_translation", 0.05))
        min_r = float(getattr(args, "streaming_keyframe_min_rotation_deg", 5.0))
        max_gap = max(1, int(getattr(args, "streaming_keyframe_max_gap", 10)))
        if gap >= max_gap or t_m >= min_t or r_deg >= min_r:
            return True
        return False

    def _record_backpressure_disposition(decision) -> None:
        """Bridge one #601 decision onto #602's AdmissionCadenceRecorder.

        Every observation the owner sees is a mapping candidate (holdouts and
        guards bypass the owner entirely and are never offered here), so the
        candidate counter is bumped for all of them and then exactly one
        disposition counter follows.
        """
        from eligsir.mapping.scheduler import BackpressureAction

        frame_id = decision.frame_id
        _admission_cadence.record_mapping_candidate(frame_id)
        action = decision.action
        if action is BackpressureAction.ADMIT:
            _admission_cadence.record_mapping_admitted(frame_id)
        elif action is BackpressureAction.METADATA_ONLY:
            _admission_cadence.record_mapping_metadata_only(frame_id)
        elif action is BackpressureAction.COALESCE_INTO:
            _admission_cadence.record_mapping_coalesced(
                frame_id, into=decision.representative_frame_id
            )
        elif action is BackpressureAction.SKIP_REDUNDANT:
            _admission_cadence.record_mapping_skipped_redundant(frame_id)
        elif action is BackpressureAction.SKIP_CAPACITY:
            _admission_cadence.record_mapping_skipped_capacity(frame_id)

    def _backpressure_observation(frame, control_tick: int):
        """Cheap :class:`PendingObservation` for the #601 owner.

        Every signal here is derived from pose + frame index + timestamp
        only -- no decode, no camera, no GPU. ``translation_m`` /
        ``rotation_deg`` are the deltas accumulated since the last
        ADMITTED frame (``_pose_delta_from_last_train``), which is why the
        owner's config sets ``causal_deltas_are_cumulative=True``.
        """
        from eligsir.mapping.scheduler import PendingObservation

        t_m, r_deg, gap = _pose_delta_from_last_train(frame)
        min_t = float(getattr(args, "streaming_keyframe_min_translation", 0.05))
        min_r = float(getattr(args, "streaming_keyframe_min_rotation_deg", 5.0))
        # Explicit, non-learned normalizations against the same thresholds
        # the cheap keyframe estimate already uses.
        novelty = min(
            1.0,
            max(
                t_m / max(1e-6, 2.0 * min_t),
                r_deg / max(1e-6, 2.0 * min_r),
            ),
        )
        motion = min(1.0, t_m / max(1e-6, 4.0 * min_t))
        return PendingObservation(
            frame_id=int(getattr(frame, "index", 0)),
            source_timestamp=float(getattr(frame, "timestamp", 0.0)),
            release_control_tick=int(control_tick),
            cheap_keyframe=_cheap_keyframe_estimate(frame),
            novelty=float(novelty),
            # No trustworthy pre-materialization quality signal exists
            # (blur/exposure need pixels), so quality is deliberately left
            # at 0.0 rather than fabricated -- the quality_protect rule
            # simply never fires until a cheap estimator exists.
            quality=0.0,
            motion=float(motion),
            translation_m=float(t_m),
            rotation_deg=float(r_deg),
        )

    # --- Loop state --------------------------------------------------------
    ema_loss = 0.0
    # Bootstrap may consume fewer frames than requested (short source / pose
    # gate). The source cursor is the sole realized fact, never ``n_init``.
    n_frames_ingested = int(streaming_scene._source_idx)
    _coverage_source_compatibility = {
        "source_identity": str(_scene_src_for_coverage),
        "raw_registry_source_frames": _registry_n_frames,
        "effective_stride": _coverage_source_effective_stride,
        "resolved_source_frames": _coverage_resolved_source_frames,
        "source_denominator_provenance": _coverage_denominator_provenance,
    }
    _coverage_ledger = CoverageLedger.restore(
        _ckpt_streaming_meta.get("coverage_ledger"),
        expected_source=_coverage_source_compatibility,
    ) if _ckpt_streaming_meta else None
    if not _ckpt_streaming_meta and _coverage_resolved_source_frames is not None:
        _coverage_ledger = CoverageLedger.fresh(
            resolved_source_frames=_coverage_resolved_source_frames,
            raw_registry_source_frames=_registry_n_frames,
            source_denominator_provenance=_coverage_denominator_provenance,
            bootstrap_frames_consumed=int(streaming_scene._source_idx),
            source_identity=_coverage_source_compatibility["source_identity"],
            effective_stride=_coverage_source_effective_stride,
        )

    def _coverage_snapshot() -> dict | None:
        return _coverage_ledger.snapshot() if _coverage_ledger is not None else None

    def _record_coverage_preingest_skip(frame) -> None:
        """Account for a source frame deliberately consumed before materialization.

        Backpressure and BurstBudget call ``skip_next_frame`` before creating a
        camera.  They still consume one finite-source record, so treating them
        as unprocessed would make a fully traversed gated source look truncated.
        """
        if _coverage_ledger is None:
            return
        is_holdout, is_guard = streaming_scene.holdout_status(frame)
        _coverage_ledger.record_preingest_skip(
            is_holdout=is_holdout,
            is_guard=is_guard,
            odom_valid=getattr(frame, "_odom_valid", None),
        )

    # PR2 (2026-08-12 cadence/batching plan): explicit optimizer/view/mutation
    # clocks -- six plain counters, telemetry only (gated on
    # streaming_cadence_telemetry_enabled below), zero training behavior
    # change. Not the CohortRecord/ProposalRecord machinery from
    # agent/evidence-cadence-phase0 -- deliberately smaller.
    _clk_optimizer_step_id = 0        # incremented once per gaussians.optimizer.step() call
    _clk_supervision_view_count = 0   # incremented once per rendered/supervised view
    # Issue #703: supervision-work accounting (rasterizer calls, supervised pixels)
    # Counters track BOTH inline and engine-path rendering/backwarding.
    # _clk_rasterizer_call_count: incremented for each render invocation in training
    # _clk_supervised_pixel_count: incremented only for views whose loss was finite and backwarded
    # Asymmetry: supervision_view_count counts attempted views; supervised_pixel_count counts only
    # successful (backwarded) ones, so supervised_pixels_per_step won't equal H*W * views/step.
    _clk_rasterizer_call_count = 0
    _clk_supervised_pixel_count = 0
    # Issue #703: one bounded counter per admitted training source, never a
    # per-step history. The source index is the stable camera identity used by
    # the batch engine as well as the inline path.
    from eligsir.utils.partition_eval import camera_source_index as _supervision_source_index
    _per_source_supervision_counts: dict[int, int] = {}

    def _register_supervision_source(camera) -> None:
        source_index = _supervision_source_index(camera)
        if source_index is not None:
            _per_source_supervision_counts.setdefault(int(source_index), 0)

    def _record_successful_supervision(source_index: int | None) -> None:
        if source_index is not None:
            stable_source_index = int(source_index)
            _per_source_supervision_counts[stable_source_index] = (
                _per_source_supervision_counts.get(stable_source_index, 0) + 1
            )

    for _supervision_camera in streaming_scene.train_cameras:
        _register_supervision_source(_supervision_camera)
    # Issue #106: distinct-camera accounting for engine-path supervision
    # batches (see the collection site for why this is measured, not assumed).
    _opt_batch_distinct_total = 0
    _opt_batch_count = 0
    _clk_sensor_frame_count = 0       # incremented once per admitted source frame
    _clk_mutation_event_id = 0        # incremented on ANY topology mutation (depth insert OR MCMC)
    _clk_depth_mutation_event_id = 0  # incremented on each depth-insertion flush
    _clk_mcmc_mutation_event_id = 0   # incremented on each MCMC growth/relocation event
    if _ckpt_streaming_meta:
        # Faithful resume: override the bootstrap-derived ingestion cursor
        # with the checkpoint's saved position so we don't re-ingest/re-admit
        # already-seen frames or re-run the bootstrap window. The frame
        # source's own internal cursor (_source_idx) — the thing that
        # actually determines which frame gets read next — is overridden
        # right after streaming_scene is constructed above; see there.
        n_frames_ingested = int(_ckpt_streaming_meta.get("n_frames_ingested", n_frames_ingested))
        print(f"[streaming] resume: restored n_frames_ingested={n_frames_ingested} "
              f"from checkpoint (was n_init={n_init})", flush=True)
        _saved_supervision_counts = _ckpt_streaming_meta.get(
            "per_source_supervision_counts", {}
        )
        if isinstance(_saved_supervision_counts, dict):
            for _saved_source_index, _saved_count in _saved_supervision_counts.items():
                try:
                    _per_source_supervision_counts[int(_saved_source_index)] = max(
                        0, int(_saved_count)
                    )
                except (TypeError, ValueError):
                    continue
    _seed_decay_done = False    # one-shot latch for streaming_bootstrap_seed_decay_frames
    _n_frames_trained = 0       # frames that received at least one training iteration
    total_inserted = 0
    # Admission-gate counters (hybrid_keyframe mode)
    _admit_cnt = {
        "forced": 0, "moved": 0, "novel": 0, "rejected_overlap": 0, "rejected_motion": 0, "total": 0,
        # Explicit buckets for the database-driven diversity gate (keyframes.py's
        # _db_diversity_score) -- without these, both reasons silently fell into
        # the generic "novel"/"rejected_motion" buckets below, making it
        # impossible to tell from telemetry whether this branch ever fires on
        # real data. Found live 2026-07-25 validating the keyframe-database plan.
        "db_diverse_revisit": 0, "rejected_redundant_angle": 0,
        # #934: explicit bucket for the novelty-driven admission trigger, for
        # the same reason as db_diverse_revisit above -- it carries forced=0 and
        # moved=0 and would otherwise be indistinguishable from render-overlap
        # novelty ("new_content") in the "novel" bucket.
        "novelty_admit": 0,
    }
    # Task 3: post-admission consolidation burst. After each keyframe admission,
    # enqueue the spatially nearest cameras so the next N training steps give the
    # newborn Gaussians immediate multi-view supervision. Drained before normal
    # sampling — no extra iterations added to the budget. Default off (queue stays
    # empty), so behavior-neutral when the flag is unset.
    from collections import deque as _deque
    _consolidation_burst_enabled = bool(args.streaming_consolidation_burst)
    _consolidation_burst_steps = int(getattr(args, "streaming_consolidation_burst_steps", 4))
    _consolidation_burst_queue: _deque = _deque()
    _prev_insert_frame = None       # previous frame for depth consistency check
    _prev_insert_depth_m = None     # depth map (metres) for previous frame
    # Phase 4.1: pending batch insertion buffer
    _insertion_batch: list = []     # list of candidate dicts from _collect_only calls
    _insertion_batch_frames: int = 0  # frames accumulated since last flush
    _batch_frames = max(1, int(getattr(args, "streaming_insertion_batch_frames", 1)))
    _batch_max_pts = int(getattr(args, "streaming_insertion_batch_max_points", 8000))
    # Evidence-driven-mutation-cadence experiment (Task 1): step-based flush
    # trigger, alternative to the frame-count trigger above. 0 = disabled
    # (default), behavior identical to before this flag existed. See
    # _should_flush_insertion_batch in
    # eligsir/mapping/integration/insertion_cadence.py.
    _batch_steps = int(getattr(args, "streaming_insertion_batch_steps", 0))
    _disable_point_cap = bool(getattr(args, "streaming_insertion_batch_disable_point_cap", False))
    _last_flush_iteration = first_iter  # re-armed on every flush below
    # Task 5 prep (C4 arm): 0 = no suppression, existing MCMC-step behavior unchanged.
    _mcmc_offset_from_insertion = int(getattr(args, "streaming_mcmc_offset_from_insertion", 0))
    # PR3 (2026-08-12 cadence/batching plan, feat/realityscan-streaming): multi-
    # keyframe gradient accumulation. 1 (the default) means the extra-view block
    # in the main loop never runs and every statement below is exactly the
    # pre-PR3 code. Already validated at startup
    # (validate_streaming_optimization_batch_size). Independent of the Task
    # 1/3/5 flags above -- no overlap in what they control.
    _opt_batch_size = max(1, int(getattr(args, "streaming_optimization_batch_size", 1)))
    _report_train_metrics_max = int(getattr(args, "streaming_report_train_metrics_max", 128))
    _report_test_metrics_max = int(getattr(args, "streaming_report_test_metrics_max", 128))
    _report_trajectory_max_frames = int(getattr(args, "streaming_report_trajectory_max_frames", 120))
    depth_loss_weight = args.streaming_depth_loss_weight
    use_depth_loss = depth_loss_weight > 0
    depth_loss_type = args.streaming_depth_loss_type
    streaming_mcmc_local = getattr(args, "streaming_mcmc_local_only", True)
    # No getattr fallback here: streaming_global_maintenance_interval is an
    # unconditional StreamingParams default (arguments/__init__.py), always set
    # on the real parsed args this function is ever called with (issue #363's
    # residue -- the old getattr(..., 500) fallback literal had drifted from
    # the real default of 200 and was reachability-audited dead code, never a
    # live bug; removed rather than re-stated to stop it from drifting again).
    global_maint_interval = max(0, args.streaming_global_maintenance_interval)

    # Diagnostic training modes (H1/H2 ablation + H11 submap-stitch)
    _training_mode = getattr(args, "streaming_training_mode", "normal")
    _placement_only = _training_mode == "placement_only"   # H1: no backward/step/MCMC
    _colors_only    = _training_mode == "colors_only"       # H2: geometry frozen, SH trains
    _skip_mcmc      = _placement_only or _colors_only
    if _colors_only:
        # Freeze geometry params under both gsplat and legacy model layouts.
        _geom_names_gsplat = {"means", "scales", "quats", "opacities"}
        _geom_names_legacy = {"xyz", "scaling", "rotation", "opacity"}
        for _pg in gaussians.optimizer.param_groups:
            if _pg.get("name") in _geom_names_gsplat | _geom_names_legacy:
                _pg["lr"] = 0.0
        gaussians.xyz_scheduler_args = lambda _step: 0.0
    # Fine-grained freeze: zero-LR on specific param groups for isolation ablations.
    # Applied *after* colors_only so it can extend or override that freeze.
    _freeze_params: set[str] = {
        p.strip()
        for p in getattr(args, "streaming_freeze_params", "").split(",")
        if p.strip()
    }
    if _freeze_params:
        _means_frozen = False
        for _pg in gaussians.optimizer.param_groups:
            if _pg.get("name") in _freeze_params:
                _pg["lr"] = 0.0
                if _pg.get("name") in ("means", "xyz"):
                    _means_frozen = True
        if _means_frozen:
            gaussians.xyz_scheduler_args = lambda _step: 0.0
        print(f"[streaming] freeze_params={sorted(_freeze_params)}", flush=True)
    if _training_mode in ("submap_stitch", "rolling_seed"):
        # Issue #1102: these two alternate trainers used to `return` without
        # emitting ANY completion evidence -- no run_completion_state.json, no
        # run_summary.json, no artifact_manifest.json, and no fail-closed
        # `raise_if_incomplete()` gate -- so they could exit 0 with a failed
        # required evidence phase and could not be ingested by ResearchFlow at
        # all. They now run the identical evidence tail as `train.py` and the
        # normal streaming path below: phase-recorded report (inside the mode),
        # then evidence artifacts, then output staging, then the gate.
        # NB `_mode_state`, deliberately NOT a name ending in `_run_state`:
        # the issue #697/#997/#998 ordering tests locate the NORMAL path's
        # tail by searching this file's source text for that variable's
        # method calls, and any such name here (in code OR in a comment
        # quoting one) would shadow the real hit with this much earlier one.
        _mode_state = _alt_trainer_run_state(args)
        if _training_mode == "submap_stitch":
            # H11: dispatch to submap-stitching path before entering the main loop
            _coordinator.run_submap_stitch_mode(
                gaussians, streaming_scene, opt, pipe, args,
                background=None,  # built inside helper
                dataset=dataset, tb_writer=tb_writer, save_worker=save_worker,
                testing_iterations=testing_iterations, saving_iterations=saving_iterations,
                sh_degree_schedule=sh_degree_schedule,
                run_state=_mode_state,
            )
        else:
            _coordinator.run_rolling_seed_mode(
                gaussians, streaming_scene, opt, pipe, args,
                background=None,
                dataset=dataset, tb_writer=tb_writer, save_worker=save_worker,
                testing_iterations=testing_iterations, saving_iterations=saving_iterations,
                sh_degree_schedule=sh_degree_schedule,
                map_backend=map_backend,
                run_state=_mode_state,
            )
        # Issue #1075 P1: shared shutdown tail. `shutdown_first=True` preserves
        # this path's existing (drain-then-restore) order verbatim -- issue
        # #1102's part 2 (reconciling the two shutdown orders) is deliberately
        # NOT decided here. `finalize_staging=False` because the evidence
        # artifacts below must land BEFORE staging moves the directory to the
        # canonical path, exactly as on the two other paths.
        finalize_run(
            save_worker,
            shutdown_first=True,
            finalize_staging=False,
            orig_sigint=_orig_sigint,
            orig_sigterm=_orig_sigterm,
        )
        _emit_run_evidence(args, _mode_state)
        _finalize_output_staging(args)
        _mode_state.raise_if_incomplete()
        return
    if _training_mode != "normal":
        print(f"[streaming] training_mode={_training_mode}: "
              f"placement_only={_placement_only} colors_only={_colors_only}", flush=True)
    # No getattr fallback: scalar_log_interval is an unconditional
    # OptimizationParams default, always present on the real parsed args
    # (issue #368 -- old getattr(..., 10) fallback drifted from the real
    # default of 100; removed rather than re-stated).
    scalar_log_interval = max(1, int(args.scalar_log_interval))
    # ---- Issue #399: per-subsystem GPU memory residency telemetry ----------
    # Constructed only when enabled, so the disabled path costs exactly one
    # `is None` test per logging interval (zero-overhead-when-off).
    _mem_tracker = None
    _mem_interval = scalar_log_interval
    if bool(getattr(args, "streaming_memory_telemetry", True)):
        from eligsir.utils.gpu_memory_telemetry import GPUMemoryResidencyTracker

        _mem_interval = int(getattr(args, "streaming_memory_telemetry_interval", 0) or 0)
        _mem_interval = _mem_interval if _mem_interval > 0 else scalar_log_interval
        _mem_tracker = GPUMemoryResidencyTracker(
            max_history=int(getattr(args, "streaming_memory_telemetry_max_samples", 512)),
        )
        print(
            f"[memory-telemetry] enabled (issue #399): device={_mem_tracker.device_type} "
            f"interval={_mem_interval} max_samples={_mem_tracker.max_history}",
            flush=True,
        )
    # ---- Host resource monitor (SPLAT_RESOURCE_MONITOR=1) ------------------
    # Constructed only when enabled -- zero cost when off beyond the one
    # `is None` check at each of its (cheap) call sites. Polls the LOCAL
    # host's Glances REST API from a background daemon thread; see
    # utils/resource_monitor.py's module docstring for how the container
    # reaches the host (host.docker.internal / host.containers.internal) and
    # for its graceful, never-crash-training failure handling.
    _resource_monitor = None
    if bool(os.environ.get("SPLAT_RESOURCE_MONITOR")):
        from eligsir.utils.resource_monitor import ResourceMonitor

        _resource_monitor = ResourceMonitor(args.model_path)
        _resource_monitor.start()
    # ---- Issue #562: CPU/GPU duty-cycle timeline + stall taxonomy ----------
    # Constructed only when enabled. While it is None, `_timed` pays exactly one
    # module-global load and one `is not None` branch per region enter/exit --
    # no clock read, no dict write, no call.
    # NOTHING here synchronizes -- see utils/duty_cycle.py's module docstring for
    # why a `torch.cuda.synchronize()`-based GPU-busy number is refused, and for
    # exactly what `optimizer_region_frac` does and does not bound.
    _duty = None
    if bool(getattr(args, "streaming_duty_cycle_telemetry", False)):
        from eligsir.utils.duty_cycle import DutyCycleTimeline
        from eligsir.utils.ingest_phase_profiler import set_duty_cycle_recorder

        _duty = DutyCycleTimeline(
            window_iters=int(getattr(args, "streaming_duty_cycle_window", 100)),
            max_windows=int(getattr(args, "streaming_duty_cycle_max_windows", 4096)),
            stall_ms=float(getattr(args, "streaming_duty_cycle_stall_ms", 250.0)),
            max_stalls=int(getattr(args, "streaming_duty_cycle_max_stalls", 64)),
        )
        set_duty_cycle_recorder(_duty)
        print(
            f"[duty-cycle] enabled (issue #562): window={_duty.window_iters} iters "
            f"max_windows={_duty.max_windows} stall_ms={_duty.stall_s * 1000:.0f} "
            "-- no cuda synchronize is added; `optimizer_region_frac` bounds the "
            "OPTIMIZER's GPU time only, not total GPU-busy time",
            flush=True,
        )
    save_interval = max(0, int(getattr(args, "save_interval", 2000)))
    chk_interval = max(0, int(getattr(args, "checkpoint_interval", 2000)))
    save_frame_interval = max(0, int(getattr(args, "streaming_save_frame_interval", 50)))
    _post_mapping_refinement_steps = int(
        getattr(args, "streaming_post_mapping_refinement_steps", 0)
    )
    _phase_controller = StreamingPhaseController(_post_mapping_refinement_steps)
    _mapping_iteration_budget = (
        int(_coverage_planning_budget)
        if _coverage_planning_budget is not None
        else int(opt.iterations)
    )
    if _post_mapping_refinement_steps > 0 and _coverage_planning_budget is None:
        raise ValueError(
            "explicit post-mapping refinement requires a finite mapping budget"
        )
    _streaming_phase = None
    _mapping_phase_wall_start = None
    _refinement_phase_wall_start = None
    # Track which frame milestone we last saved at (avoids repeated saves for same frame)
    _last_frame_save = n_init  # bootstrap frames already "processed"

    # Warmup gate: suppress depth insertion + MCMC growth until this many train
    # cameras have arrived (0 = disabled). Avoids growing/overfitting on too few
    # early views before the scene is sufficiently constrained.
    _warmup_min_frames = max(0, int(getattr(args, "streaming_warmup_min_frames", 0)))

    # H7: old-geometry gradient freeze parameters
    _freeze_old = getattr(args, "streaming_freeze_old_geometry", False)
    _young_age_frames = max(0, int(getattr(args, "streaming_young_age_frames", 5)))
    _freeze_new_frame_steps = max(0, int(getattr(args, "streaming_freeze_new_frame_steps", 50)))
    _iter_since_new_frame = _freeze_new_frame_steps  # start without strict freeze
    if _freeze_old:
        print(
            "[streaming] WARNING: --streaming_freeze_old_geometry is enabled. "
            "This flag is experimentally confirmed to cause a catastrophic, "
            "non-recovering global optimization collapse (Replica room0: "
            "27.95dB -> 14.54dB; kitchen1 Orbbec: 16-20dB -> 5-7dB, with "
            "train PSNR == test PSNR for the whole run), not selective "
            "old-region forgetting. See docs/streaming-anti-forgetting-defaults.md "
            "before trusting results from this run.",
            flush=True,
        )

    # H9: anchor loss parameters (provisional splats only)
    _anchor_loss_weight = float(getattr(args, "streaming_anchor_loss_weight", 0.0))
    _anchor_decay_steps = max(1, int(getattr(args, "streaming_anchor_decay_steps", 500)))

    # Bootstrap anchor: penalise center drift of birth_frame==0 Gaussians.
    _bootstrap_anchor_weight = float(args.streaming_bootstrap_anchor_weight)
    _bootstrap_anchor_decay = max(1, int(getattr(args, "streaming_bootstrap_anchor_decay_steps", 2000)))
    _insertion_anchor_weight = float(getattr(args, "streaming_insertion_anchor_weight", 1.0))
    _insertion_anchor_decay  = max(1, int(getattr(args, "streaming_insertion_anchor_decay_steps", 200)))

    # ---- Issue #326: runtime proposed/applied/cancelled operation trace ------
    # OFF by default. `install_from_env` returns None unless
    # ELIGSIR_GAUSSIAN_OP_TRACE is truthy, and `_optrace` is then held in a
    # local for the whole loop -- so the disabled path costs exactly one
    # `is not None` test per hook site and NOTHING else. Every tensor touch
    # (and therefore every device->host sync) lives lexically inside a guard.
    from eligsir.diagnostics.gaussian_op_trace import install_from_env as _install_optrace
    # `args.model_path` is an unconditional ParamGroup default -- bare attribute
    # access, no getattr fallback literal (issues #363/#368).
    _optrace = _install_optrace(model_path=args.model_path)
    if _optrace is not None:
        print(
            f"[gaussian-op-trace] ENABLED (issue #326) out={_optrace.out_path} "
            f"uid_stride={_optrace.uid_stride} trace_every={_optrace.trace_every} "
            f"max_events={_optrace.max_events} -- this materially slows training; "
            "it is a diagnostic run, not a benchmark run.",
            flush=True,
        )

    _depth_lock_active = depth_lock_enabled(args)
    _depth_conflict_active = depth_conflict_enabled(args)
    _depth_lock_report_every = max(
        0,
        int(getattr(args, "streaming_depth_lock_report_interval", 100)),
    )
    if _depth_lock_active or _depth_conflict_active:
        ensure_depth_lock_buffers(gaussians)
        print(
            "[streaming-depth-lock] "
            f"centers={_depth_lock_active} "
            f"bootstrap={bool(getattr(args, 'streaming_depth_lock_bootstrap', False))} "
            f"insertions={bool(getattr(args, 'streaming_depth_lock_insertions', False))} "
            f"multiview_conflict={_depth_conflict_active} "
            f"prune_conflicts={bool(getattr(args, 'streaming_depth_lock_prune_conflicts', False))} "
            f"conflict_views={int(getattr(args, 'streaming_depth_unlock_conflict_views', 2))} "
            f"conflict_thresh={depth_lock_conflict_threshold(args):.4f}m",
            flush=True,
        )
    # Disable report, trajectory eval, and progress video for throughput profiling.
    _benchmark_mode = getattr(args, "streaming_benchmark_mode", False)

    # Training-progress video setup (disabled in benchmark_mode to save VRAM/CPU overhead)
    _progress_video_interval = 0 if _benchmark_mode else max(0, int(args.progress_video_interval))
    _progress_video_fps = int(getattr(args, "progress_video_fps", 10))
    _progress_cams: list = []     # fixed camera objects once selected
    _progress_frames: list = []   # accumulated side-by-side uint8 HWC numpy frames

    from gaussian_renderer.gsplat_backend import render_batch as _render_batch

    def _depth_insertion_report_block() -> dict:
        """Phase 2 depth-insertion sanity gate (issue #303).

        ``inert`` means depth insertion was ENABLED and frames WERE admitted,
        yet Phase 2 never inserted a single Gaussian -- the signature of a
        silently-dropped depth stream (e.g. the compressedDepth transport bug
        this gate was added with, or the earlier ``live-insertion-noop-bug``).
        Requiring ``frames_admitted > 0`` keeps a run that admitted nothing for
        an unrelated reason from being mislabeled as a depth-decode failure.
        """
        from eligsir.utils.comparison_report import depth_insertion_report_block

        return depth_insertion_report_block(
            enabled=args.streaming_insert_from_depth,
            total_inserted=total_inserted,
            frames_admitted=getattr(streaming_scene, "_keyframes_admitted", 0),
        )

    def _ingress_ownership_report_block() -> dict:
        """Issue #837: machine-readable ingress ownership contract, sibling
        of ``_source_decimation_block`` (issue #834/PR #818) built above.

        Scope note: this repo's #837 slice covers acceptance criteria 1-7,
        plus criterion 8's ``source_sampling`` (issue #834) and
        ``overflow_drop_policy`` (issue #836) halves. Criterion 8's
        remaining two halves -- #832/#833 feeding their REALIZED owner
        state into this contract -- are deliberately not wired here (those
        PRs are still blocked/in flight); see
        ``eligsir.io.ingress_ownership``'s module docstring and
        ``register_ingress_owner``/``registered_declarations`` for the
        declarative API those sibling components will call once they land.

        ``source_sampling`` is populated from real runtime state, reusing
        the SAME resolved policy ``_source_decimation_block`` already
        carries (single source of truth -- this never re-derives sampling
        ownership). ``overflow_drop_policy`` is populated by reading the
        realized ``(inner_policy, outer_policy)`` pair issue #836's
        ``resolve_ingress_policies(args)`` already resolved once for this
        run, off the live ``frame_source`` object -- see
        ``eligsir.io.policies.ingress_overflow_ownership``'s module
        docstring for why that pair collapses into ONE owner declaration
        (declaring both queue boundaries separately would trip
        ``DUPLICATE_INGRESS_AUTHORITY`` on every live run) and for why this
        reads the already-resolved policies off ``frame_source`` rather
        than calling ``resolve_ingress_policies(args)`` a second time. A
        non-``ros2://live`` run (TUM/ScanNet/generic streaming replay/
        offline ``train.py``) has no overflow-drop-policy owner at all --
        that responsibility is left absent/unresolved, exactly like
        ``source_sampling`` when ``sampling_owner`` is ``None``, never
        fabricated.

        Any declarations registered via ``register_ingress_owner`` (none,
        as of this PR) are merged in automatically. ``legacy_topology``
        reuses ``_source_decimation_block["legacy_composition_active"]`` --
        the one existing signal in this run for "two sampling owners
        really did both apply", so this block never disagrees with the
        decimation block it is a sibling of.

        Behavior-neutral: computes and returns a plain dict for
        ``report.json["ingress_ownership"]`` only. Never raises, never
        gates admission/training.
        """
        from eligsir.io.ingress_ownership import (
            OwnerDeclaration,
            OwnershipRole,
            Responsibility,
            registered_declarations,
            resolve_ingress_ownership,
        )
        from eligsir.io.policies.ingress_overflow_ownership import (
            resolve_overflow_drop_policy_owner_declaration,
            unwrap_frame_source_overflow_policies,
        )
        from eligsir.io.policies.rgbd_pairing_ownership import (
            rgbd_pairing_owner_declarations_from_env,
        )

        declarations = list(registered_declarations())

        # Issue #832 half of criterion 8: the REALIZED RGB-D pairing /
        # timestamp-correction / transport-decode topology this bag run was
        # launched with, derived from the three raw launch inputs that
        # 05_run_offline_rosbag.sh exports onto this container. Returns []
        # for every non-bag run (TUM/ScanNet/generic replay/offline
        # train.py), leaving those responsibilities absent rather than
        # fabricating an owner -- the same rule source_sampling and
        # overflow_drop_policy already follow. The historical stacked
        # composition (which used to emit TWO mutating owners for pairing
        # and correction and trip DUPLICATE_INGRESS_AUTHORITY) was deleted by
        # #832's hard-cut amendment -- the launch scripts refuse
        # RGBD_PAIRING_OWNER=legacy_stacked at startup now, so this always
        # resolves to a single owner per responsibility -- see that module's
        # docstring.
        declarations.extend(rgbd_pairing_owner_declarations_from_env())
        _sampling_owner = _source_decimation_block.get("sampling_owner")
        if _sampling_owner is not None:
            declarations.append(
                OwnerDeclaration(
                    Responsibility.SOURCE_SAMPLING,
                    owner=(
                        "native_source_stride" if _sampling_owner == "native"
                        else "generic_strided_frame_source" if _sampling_owner == "generic"
                        else str(_sampling_owner)
                    ),
                    role=OwnershipRole.MUTATOR,
                    attributes={
                        "effective_stride": _source_decimation_block.get("effective_stride"),
                        "requested_frame_stride": _source_decimation_block.get("frame_stride"),
                        "requested_source_stride": _source_decimation_block.get("source_stride"),
                    },
                )
            )

        # Issue #836 half of criterion 8: read the already-resolved
        # overflow-policy pair off frame_source (never re-derive; see the
        # docstring above and unwrap_frame_source_overflow_policies's own
        # docstring). None/None means this run has no ros2://live overflow
        # policy at all -- leave the responsibility unresolved.
        _inner_overflow_policy, _outer_overflow_policy = unwrap_frame_source_overflow_policies(
            frame_source
        )
        if _inner_overflow_policy is not None and _outer_overflow_policy is not None:
            declarations.append(
                resolve_overflow_drop_policy_owner_declaration(
                    _inner_overflow_policy, _outer_overflow_policy
                )
            )

        result = resolve_ingress_ownership(
            declarations,
            legacy_topology=bool(_source_decimation_block.get("legacy_composition_active")),
        )
        return result.to_report_dict()

    def _lineage_preserving_setup_report_block() -> "dict | None":
        """Issue #374 item 3: count + iteration of every mid-run
        ``training_setup(preserve_lineage=True)`` call. See
        ``lineage_preserving_setup_report_block``'s docstring -- not gated on
        ``final`` like ``depth_insertion`` since the event list is only ever
        appended to (there is nothing "not yet real" about a mid-run reading
        of it, unlike depth-insertion's total_inserted==0 pre-run case).
        """
        from eligsir.utils.comparison_report import lineage_preserving_setup_report_block

        return lineage_preserving_setup_report_block(
            getattr(gaussians, "_lineage_preserving_setup_events", [])
        )

    def _seed_cohort_report_block() -> dict:
        """Issue #346: record whether the EXACT #283 seed cohort is
        determinable for this run, in the run's own ``report.json``.

        Always emitted (both mid-run and final): ``available: False`` with the
        reason is the informative case, and until now it was recorded nowhere
        at all -- see ``seed_cohort_report_block``'s docstring for the audit
        that motivated this.
        """
        from eligsir.utils.comparison_report import seed_cohort_report_block

        return seed_cohort_report_block(gaussians)

    def _exposure_opacity_survival_block() -> dict | None:
        """Issue #1262 §6.2.1: per-Gaussian opacity survival by exposure group.

        REQUIRED alongside ``H(t)`` -- topology destruction and zombie opacity
        death produce identical hole-ishness signatures and are told apart only
        by the fraction of the UNEXPOSED (E-) depth-inserted population below the
        rasterizer's own 1/255 cull. ``None`` (key absent) when the
        resolution-exposure feature is off, so a flag-off report is unchanged.
        """
        from eligsir.utils.resolution_exposure import (
            exposure_config_from_args,
            exposure_veto_telemetry,
        )

        cfg = exposure_config_from_args(args)
        if not cfg.enabled:
            return None
        exposed = getattr(gaussians, "_last_exposure_mask", None)
        alpha = gaussians.get_opacity.detach().squeeze(-1)
        if exposed is None or exposed.shape[0] != alpha.shape[0]:
            return {
                "available": False,
                "reason": "no exposure evaluated yet for this population",
                "veto": exposure_veto_telemetry(gaussians),
            }
        from eligsir.utils.exposure_opacity_survival import (
            exposure_opacity_survival_block,
            snapshot,
        )

        _snap = snapshot(
            iteration=int(getattr(gaussians, "_last_exposure_iteration", 0)),
            alpha=alpha,
            exposed=exposed,
            birth_method=getattr(gaussians, "birth_method", None),
            gaussian_uid=getattr(gaussians, "gaussian_uid", None),
            alpha_dead=float(args.energy_alpha_dead) * 0.1,
        )
        block = exposure_opacity_survival_block(
            at_tier_switch=None,
            at_run_end=_snap,
            predicate=cfg.predicate,
            alpha_dead_effective=float(args.energy_alpha_dead) * 0.1,
            proxy_ambiguous_frac=getattr(
                gaussians, "_last_exposure_ambiguous_frac", None
            ),
        )
        block["available"] = True
        block["veto"] = exposure_veto_telemetry(gaussians)
        # §6.2.1's three-row outcome table. With a single run there is no
        # tier-switch anchor and no NATIVE-tier control, so the relocation-rate
        # and H(t) inputs are unavailable and this legitimately grades
        # "indeterminate" -- the cross-run resolution is the decisive experiment's
        # job, not this block's.
        from eligsir.utils.exposure_opacity_survival import classify_outcome, per_checkpoint_row

        block["outcome"] = classify_outcome(
            relocation_rate_e_minus=None,
            relocation_rate_e_plus=None,
            holeishness_series=None,
            frac_below_cull_e_minus_at_switch=None,
            frac_below_cull_e_minus_at_end=_snap.e_minus["frac_below_rasterizer_cull"],
        )
        block["checkpoint_row"] = per_checkpoint_row(
            _snap,
            resolution_tier=getattr(gaussians, "_last_exposure_tier", None),
        )
        return block

    def _streaming_report_kwargs(
        final: bool = False, *, coverage_snapshot: dict | None = None
    ) -> dict:
        """Bind this loop's live state to the shared report-kwargs builder.

        Issue #1075 (task P3): the ~280-line body this used to hold now lives
        in ``eligsir/runtime/report_kwargs.py``, where every one of its
        ~35 formerly-implicit closure captures is a named keyword argument.
        What stays here is only the binding of those names to this loop's
        state -- deliberately still a closure, so the four existing
        ``**_streaming_report_kwargs()`` call sites (and the #729
        free-variable scope guard that watches which names they reach) are
        unchanged.
        """

        def _set_partition_control_roles(roles) -> None:
            # Issue #951's roles refresh was a ``nonlocal`` write from inside
            # the old closure body; it stays a real side effect of building
            # the report, now routed through an explicit callback.
            nonlocal _partition_control_roles
            _partition_control_roles = roles

        # Mid-run reports execute in a worker thread. Snapshot this mutable
        # ledger here so the report's iteration and dispositions agree.
        coverage_counters = (
            dict(coverage_snapshot)
            if isinstance(coverage_snapshot, dict) else _coverage_snapshot()
        )
        _wall_time_s = time.perf_counter() - _train_t0

        return _build_streaming_report_kwargs(
            final=final,
            args=args,
            streaming_scene=streaming_scene,
            frame_source=frame_source,
            first_iter=first_iter,
            seed_fade_monitor=seed_fade_monitor,
            cadence=_cadence,
            admission_cadence=_admission_cadence,
            coordinator=_coordinator,
            spf_resolution=_spf_resolution,
            resolution_provenance=_resolution_provenance,
            resolution_tier_log=_resolution_tier_log,
            ingestion_report=_ingestion_report,
            replica_rgb_report=_replica_rgb_report,
            soft_all_admitter=_soft_all_admitter,
            soft_all_invariant=_soft_all_invariant,
            reinit_tracker=_reinit_tracker,
            partition_control_runtime=_partition_control_runtime,
            mem_tracker=_mem_tracker,
            resource_monitor=_resource_monitor,
            duty=_duty,
            backpressure_owner=_backpressure_owner,
            clk_optimizer_step_id=_clk_optimizer_step_id,
            clk_supervision_view_count=_clk_supervision_view_count,
            clk_rasterizer_call_count=_clk_rasterizer_call_count,
            clk_supervised_pixel_count=_clk_supervised_pixel_count,
            per_source_supervision_counts=_per_source_supervision_counts,
            wall_time_s=_wall_time_s,
            report_train_metrics_max=_report_train_metrics_max,
            report_test_metrics_max=_report_test_metrics_max,
            report_trajectory_max_frames=_report_trajectory_max_frames,
            depth_insertion_report_block=_depth_insertion_report_block,
            lineage_preserving_setup_report_block=_lineage_preserving_setup_report_block,
            seed_cohort_report_block=_seed_cohort_report_block,
            exposure_opacity_survival_block=_exposure_opacity_survival_block,
            ingress_ownership_report_block=_ingress_ownership_report_block,
            register_supervision_source=_register_supervision_source,
            git_provenance_from_args=_git_provenance_from_args,
            multires_cache_report_block=multires_cache_report_block,
            load_bag_source_manifest=_load_bag_source_manifest,
            set_partition_control_roles=_set_partition_control_roles,
            cadence_reconcile_kwargs=_cadence_reconcile_kwargs,
            streaming_phase=_streaming_phase,
            coverage_counters=coverage_counters,
        )

    # Async render: renders go on a side CUDA stream; CPU work (numpy/cv2) goes to a daemon thread.
    # This overlaps GPU renders and CPU image assembly with the next training iteration.
    import threading as _threading
    import queue as _queue_mod
    _pv_stream = torch.cuda.Stream()
    _pv_cpu_queue: _queue_mod.Queue = _queue_mod.Queue(maxsize=2)

    def _pv_cpu_worker():
        while True:
            item = _pv_cpu_queue.get()
            if item is None:
                break
            _gpu_panels, _pv_iter, _pv_total = item
            _panels = []
            for _t in _gpu_panels:
                _arr = _t.cpu().numpy()  # waits for _pv_stream to finish this tensor
                _arr = np.transpose(_arr, (1, 2, 0))
                _arr = (_arr * 255 + 0.5).astype(np.uint8)
                _panels.append(_arr)
            _sep = np.ones((_panels[0].shape[0], 2, 3), dtype=np.uint8) * 80
            _combined = np.concatenate([_panels[0], _sep, _panels[1], _sep, _panels[2]], axis=1)
            try:
                import cv2 as _cv2
                _lbl = f"step {_pv_iter:>6d} / {_pv_total}"
                _cv2.putText(_combined, _lbl, (8, 20), _cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, _cv2.LINE_AA)
                _cv2.putText(_combined, _lbl, (8, 20), _cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, _cv2.LINE_AA)
            except Exception:
                pass
            _progress_frames.append(_combined)

    _pv_thread = _threading.Thread(target=_pv_cpu_worker, daemon=True)
    _pv_thread.start()

    # Four-state lifecycle parameters (Component B/C)
    _lifecycle_enabled = args.streaming_lifecycle_enabled
    _mature_age_frames = max(1, int(getattr(args, "streaming_mature_age_frames", 15)))
    _mature_min_utility = float(getattr(args, "streaming_mature_min_utility", 0.1))
    _freeze_age_frames = int(getattr(args, "streaming_freeze_age_frames", -1))
    _utility_ema_beta = float(getattr(args, "streaming_utility_ema_beta", 0.95))
    # Mature-Gaussian anchor loss weights (Component C)
    _mature_anchor_xyz_w = float(getattr(args, "streaming_mature_anchor_xyz_weight", 0.0))
    _mature_anchor_scale_w = float(getattr(args, "streaming_mature_anchor_scale_weight", 0.0))
    _mature_anchor_opacity_w = float(getattr(args, "streaming_mature_anchor_opacity_weight", 0.0))
    use_energy_mcmc = getattr(args, "energy_mcmc", True) and densification_strategy in {
        "mcmc", "hybrid", "gsplat_energy_mcmc", "gsplat_live_strategy"
    }

    _application_admission_fn = None
    _application_pregate_fn = None
    if args.streaming_frame_admission == "hybrid_keyframe":
        _application_admission_fn = _keyframe_admission_closure
        _application_pregate_fn = _keyframe_pregate_closure
    elif args.streaming_frame_admission == "fixed_stride":
        _application_admission_fn = _fixed_stride_admission_closure
    elif args.streaming_frame_admission != "force_all":
        _application_admission_fn = _soft_all_admission_closure
    _mapping_ingest_callbacks = StreamingIngestCallbacks(
        admission_fn=_application_admission_fn,
        visibility_fn=_visibility_fn,
        record_fn=_record_fn,
        pregate_fn=_application_pregate_fn,
        integrity_observer_fn=_integrity_observer_closure,
        defer_extent_update=True,
    )

    # Keep the extracted fixed-topology engine opt-in until full frozen-run
    # parity is recorded. The existing inline block below remains the control.
    _streaming_optimization_runtime = None
    _streaming_optimization_engine = None
    _mapping_application = None
    if bool(args.streaming_optimization_engine):
        from eligsir.runtime.transfers import get_device_transfer_owner

        _streaming_optimization_engine = _coordinator.optimization_engine_for(gaussians)
        _streaming_optimization_runtime = StreamingOptimizationRuntime(
            session=_streaming_optimization_engine.session,
            gaussians=gaussians,
            args=args,
            opt=opt,
            pipe=pipe,
            mcmc_strategy=mcmc_strategy,
            tb_writer=tb_writer,
            # Issue #279: same wiring as production_application.py -- without
            # this, engine.py's own per-phase timing brackets never fire on
            # this (--streaming_optimization_engine) path either, regardless
            # of SPLAT_PROFILE_PHASES. `_timed` is zero-overhead when unset.
            timing_scope=_timed,
            optimizer_type=optimizer_type,
            sparse_active_set=sparse_active_set,
            use_energy_mcmc=use_energy_mcmc,
            fixed_shape_mcmc=(
                not _skip_mcmc
                and densification_strategy in {
                    "mcmc", "hybrid", "gsplat_mcmc", "gsplat_energy_mcmc", "gsplat_live_strategy",
                }
            ),
            mcmc_local=streaming_mcmc_local,
            placement_only=_placement_only,
            loss_assembly_kwargs={
                "tb_writer": tb_writer,
                "streaming_scene": streaming_scene,
                "seed_fade_monitor": seed_fade_monitor,
                "depth_loss_type": depth_loss_type,
                "depth_loss_weight": depth_loss_weight,
                "scalar_log_interval": scalar_log_interval,
                "_anchor_loss_weight": _anchor_loss_weight,
                "_anchor_decay_steps": _anchor_decay_steps,
                "_bootstrap_anchor_weight": _bootstrap_anchor_weight,
                "_bootstrap_anchor_decay": _bootstrap_anchor_decay,
                "_insertion_anchor_weight": _insertion_anchor_weight,
                "_insertion_anchor_decay": _insertion_anchor_decay,
                "_lifecycle_enabled": _lifecycle_enabled,
                "_mature_anchor_xyz_w": _mature_anchor_xyz_w,
                "_mature_anchor_scale_w": _mature_anchor_scale_w,
                "_mature_anchor_opacity_w": _mature_anchor_opacity_w,
                "nan_skip_count_holder": [0],
            },
            iter_since_new_frame=_iter_since_new_frame,
            optimizer_iteration_limit=(
                None
                if _post_mapping_refinement_steps == 0
                else (2**31 - 1)
            ),
            camera_transfer_owner=get_device_transfer_owner(
                gaussians.get_xyz.device
            ),
        )

        def _optimizer_tick():
            """Run one promoted optimizer iteration through its existing owner."""
            streaming_scene.flush_deferred_extent_update()
            with _timed("m_optimization_engine"):
                return run_streaming_optimization_step(
                    _streaming_optimization_runtime,
                    _streaming_optimization_engine,
                    camera=viewpoint_cam,
                    background=bg,
                    iteration=iteration,
                    map_version=map_backend.map_version,
                    topology_version=map_backend.topology_version,
                    resolution_bucket=_active_tier,
                    update_sh_rest=update_sh_rest,
                    render_depth=use_depth_loss,
                    n_frames_ingested=n_frames_ingested,
                    visible_mask=None,
                    random_state_token=RandomStateToken.capture(),
                    extra_cameras=_engine_extra_cameras,
                    skipped_extra_view_count=_engine_skipped_extra_views,
                )

        _mapping_application = _compose_mapping_application(
            streaming_scene,
            _optimizer_tick,
            _mapping_ingest_callbacks,
        )

    phase_timer.end("training_setup")
    phase_timer.start("training")
    _mapping_phase_wall_start = time.perf_counter()

    _n_total_frames = streaming_scene._frame_count
    _spf = scheduler.steps_per_frame
    _min_iters_all = (getattr(args, "streaming_initial_frames", 5) * _spf) + _n_total_frames * _spf
    print(
        f"[streaming] Starting: strategy={densification_strategy} "
        f"optimizer={optimizer_type} sparse={sparse_active_set} "
        f"steps_per_frame={_spf} "
        f"wallclock={scheduler.wallclock}",
        flush=True,
    )
    # #1044: this line used to divide by `opt.iterations`, which is the 2^31
    # online sentinel on every --online_mode run -- so it announced
    # "covers 100%. All frames covered." for a run that in fact reached 5.1% of
    # the trajectory. Report against the resolved planning budget, and say
    # nothing about coverage when there is no budget to say it against.
    if _coverage_planning_budget is None:
        print(
            f"[streaming] Dataset: {_n_total_frames} frames. "
            f"Optimization steps to cover all frames: {_min_iters_all}. "
            f"Budget is UNBOUNDED (--online_mode, rolling-idle termination): "
            f"coverage is not predictable from the iteration budget and is NOT "
            f"claimed here. See the [spf] block above.",
            flush=True,
        )
    else:
        _budget = int(_coverage_planning_budget)
        print(
            f"[streaming] Dataset: {_n_total_frames} frames. "
            f"Optimization steps to cover all frames: {_min_iters_all} "
            f"(planning budget={_budget} [{_coverage_budget_source}], "
            f"covers {min(_budget, _min_iters_all) * 100 // max(1, _min_iters_all)}%). "
            + ("Remaining frames will be flushed post-loop."
               if _budget < _min_iters_all else "All frames covered."),
            flush=True,
        )

    # Snapshot the bootstrap camera list so the end-of-training reporter can
    # render the SAME views with the trained Gaussians for a direct diff.
    _bootstrap_cams = list(streaming_scene.getTrainCameras())

    # Pre-training snapshot: render the post-bootstrap state (opt-in; adds startup overhead)
    if getattr(args, "streaming_report_pre_training", False):
        try:
            from eligsir.utils.comparison_report import write_post_training_report, git_provenance_from_args as _git_provenance_from_args
            print(
                f"[streaming] Bootstrap (post-init) report at step_0 over "
                f"{len(_bootstrap_cams)} bootstrap views...",
                flush=True,
            )
            write_post_training_report(
                streaming_scene=streaming_scene,
                model_path=args.model_path,
                iteration=0,
                gaussians=gaussians,
                train_cams=_bootstrap_cams,
                test_cams=_bootstrap_cams,
                render_fn=render,
                pipe=pipe,
                background=background,
                tb_writer=tb_writer,
                log_prefix="streaming_report",
                subdir="iter_0_bootstrap_views",
                degenerate_split=True,
                **_streaming_report_kwargs(),
            )
        except Exception as e:
            print(f"[streaming-report] pre-run report failed: {e}", flush=True)

    _debug_bootstrap_motion = bool(getattr(args, "streaming_debug_bootstrap_motion", False))
    _debug_bootstrap_ply_interval = max(0, int(getattr(args, "streaming_debug_bootstrap_ply_interval", 0)))
    _debug_bootstrap_first_iters = max(0, int(getattr(args, "streaming_debug_bootstrap_first_iters", 200)))
    _debug_bootstrap_threshold = float(getattr(args, "streaming_debug_bootstrap_motion_threshold", 0.05))
    if _debug_bootstrap_motion:
        _log_bootstrap_motion(
            tb_writer,
            args,
            0,
            "total",
            _bootstrap_motion_stats(gaussians, threshold=_debug_bootstrap_threshold),
        )
        _write_bootstrap_motion_ply(gaussians, args, 0, "post_init")

    # Honest progress-bar bound: online_mode sets opt.iterations to the 2^31
    # unbounded sentinel (a live source has no known length), which renders the
    # bar total/ETA meaningless. When --streaming_max_iterations caps the run,
    # display that cap instead — it is the real stop condition.
    _bar_total = opt.iterations
    _bar_cap = int(getattr(args, "streaming_max_iterations", 0) or 0)
    if _bar_cap > 0:
        _bar_total = min(_bar_total, _bar_cap)
    if _post_mapping_refinement_steps > 0:
        _bar_total = _mapping_iteration_budget + _post_mapping_refinement_steps
    progress_bar = tqdm(
        range(first_iter, _bar_total),
        desc="Streaming mapping runtime",
        # #878: --disable_progress_bar was declared but never read on this path,
        # so the one flag meant to silence the bar did nothing and only --quiet
        # (ruled out by project convention) worked. Mirror train.py:792.
        disable=(
            getattr(args, "disable_progress_bar", False)
            or getattr(args, "quiet", False)
        ),
    )
    first_iter += 1

    def _terminal_eval_tick_clock_snapshot() -> dict:
        """Exact terminal streaming counters for #698's final report tick."""
        _source_frame_index = getattr(streaming_scene, "_source_idx", None)
        return {
            "source_frame_index": (
                int(_source_frame_index)
                if _source_frame_index is not None else None
            ),
            "frames_ingested": int(n_frames_ingested),
            "frames_trained": int(_n_frames_trained),
            "keyframes_admitted": int(
                getattr(streaming_scene, "_keyframes_admitted", 0) or 0
            ),
            "optimizer_step": int(_clk_optimizer_step_id),
            "supervision_views": int(_clk_supervision_view_count),
            "wall_clock_s": float(time.perf_counter() - _train_t0),
            "accumulated_iteration_wall_s": float(
                getattr(online_clock, "wall_time_s", 0.0)
            ),
            "last_ingest_iter": int(_last_ingest_iter),
            "mapping_idle_threshold_iters": (
                int(args.streaming_refinement_iterations or 0) or None
            ),
        }

    # ---- Async ingestion worker (Variant B; --streaming_async_ingest) --------
    # Default off => byte-identical to today (worker never started). When on, a
    # background thread runs the CPU-only prep seam (pull next raw frame + decode
    # RGB/depth into the per-frame cache) AHEAD of the optimizer, so the GPU stops
    # idling while frames arrive. The optimizer thread stays the SOLE mutator:
    # candidate collection (needs a render of the live model), Camera GPU upload
    # (data_device=cuda), and every scene-list / occupancy mutation remain on the
    # optimizer and are committed via the unchanged ingest_next_frame + drain body.
    # The handoff is the *decoded* frame; coverage is preserved because the
    # optimizer still drains every frame in cursor order and the bounded queue
    # back-pressure DEFERS (never drops) when the worker runs too far ahead.
    _ingest_worker = None
    if getattr(args, "streaming_async_ingest", False):
        from eligsir.core.ingestion_worker import IngestionWorker
        from eligsir.utils.streaming_frames import load_frame_rgb, load_frame_depth_np

        # Worker-owned look-ahead cursor; never advances the optimizer's
        # _source_idx (single cursor owner = optimizer via ingest_next_frame).
        _async_state = {"ahead_idx": streaming_scene._source_idx}
        _async_depth = max(1, int(getattr(args, "streaming_async_ingest_depth", 4)))

        def _async_source_fn():
            # Non-blocking: return the next not-yet-decoded source frame, but stay
            # within _async_depth of the optimizer's consume position so the worker
            # cannot race arbitrarily far ahead of committed state. Returns None
            # when no frame is ready (source not yet buffered, exhausted, or the
            # look-ahead window is full) -> optimizer trains on buffered data.
            idx = _async_state["ahead_idx"]
            if idx >= streaming_scene._frame_count:
                return None
            if idx - streaming_scene._source_idx >= _async_depth:
                return None
            bc = getattr(streaming_scene._frame_source, "buffered_count", None)
            if callable(bc) and idx >= bc():
                return None  # live source has not buffered this frame yet
            _async_state["ahead_idx"] = idx + 1
            return streaming_scene._get_source_frame(idx)

        def _async_prep_fn(frame):
            # CPU-only, read-only: decode RGB + depth so the cache is warm when the
            # optimizer's _frame_to_camera / insertion re-reads them. Does NOT touch
            # the Gaussian model, optimizer state, occupancy, or the GPU.
            if frame is None:
                return None
            try:
                load_frame_rgb(frame)
            except Exception:
                pass
            try:
                load_frame_depth_np(frame)
            except Exception:
                pass
            return frame

        def _async_on_error(exc):
            print(f"[streaming] async-ingest worker prep error: {exc}", flush=True)

        _ingest_worker = IngestionWorker(
            source_fn=_async_source_fn,
            prep_fn=_async_prep_fn,
            depth=_async_depth,
            on_error=_async_on_error,
        ).start()
        print(
            f"[streaming] async ingestion worker ON (depth={_async_depth}); "
            "CPU prep (decode) overlapped with optimizer GPU step.",
            flush=True,
        )

    if _backpressure_owner is not None and (
        _ingest_worker is not None or _deferred_slam_tracker is not None
    ):
        print(
            "[backpressure] WARNING: the owner is INERT this run -- its "
            "drain-loop gate is skipped while the async ingest worker or the "
            "deferred-SLAM admission tracker is active (the peeked frame's pose "
            "is not resolved yet at that point). No skip/coalesce decision will "
            "be taken; the backpressure report block will show n_observed=0.",
            flush=True,
        )

    # Seed-fade tracking (ablation): the FIRST bootstrap camera is captured once
    # (lazily, the first time train_cameras is non-empty) and re-rendered every
    # scalar_log_interval to measure whether the initial seed geometry survives as
    # the camera travels away — the core "scene implodes mid-trajectory" symptom.
    _seed_cam = None
    # Local-capacity telemetry cross-sample state: last (psnr, n_local) per region
    # cell, for the finite-difference dPSNR/dN marginal-utility estimate.
    _local_cap_prev = {}
    # Adaptive-capacity controller v1 (brake-only) cross-sample state.
    # Per-partition sentinel state: cached denominator + one-time fallback warning
    # (same deterministic binning as utils/comparison_report.py — shared helpers
    # in utils/partition_eval.py).
    _psent = {"max_src": None, "denom_src": None, "warned": False}
    _adcap = {"prev_p": None, "prev_n": None, "u_ema": 0.0, "brake_streak": 0,
              "release_streak": 0, "state": "GROW"}
    if getattr(args, "streaming_adaptive_capacity", False):
        # Start permissive (grow up to cap_max); the controller only clamps on brake.
        args._adaptive_growth_cap = int(args.cap_max)

    # Unified rolling-budget termination rule (2026-07-04 redesign): the loop
    # keeps running as long as new frames keep arriving — bag-replay and live
    # camera alike, no separate "coverage phase" vs "refinement phase"
    # bookkeeping. Every time a frame is ingested, _last_ingest_iter re-arms;
    # the run only ends once streaming_refinement_iterations have elapsed with
    # nothing new.
    #
    # This is deliberately an ADMISSION-INDEPENDENT clock — it re-arms on any
    # frame reaching the ingest path (streaming_scene.ingest_next_frame(...)
    # returning non-None), matching F=/n_frames_ingested exactly, NOT on
    # keyframe-admitted training views. Two other "idle" clocks already exist
    # and were considered for unification here, but are semantically distinct
    # on purpose, not an accidental duplication:
    #   - streaming_scene._steps_since_new_frame (consulted by
    #     streaming_consolidation_idle_steps at ~4813/5231/6347) resets ONLY
    #     on an ADMITTED training keyframe (scene/streaming_scene.py:1454) —
    #     it answers "has no new TRAINING VIEW shown up," a training-quality
    #     question for the bubble-center/soft-anchor/sampler policies. Given
    #     the keyframe-admission gate can reject far more candidates than it
    #     admits (a separate, confirmed issue — see Bug 2 in this session's
    #     coverage investigation), merging this with the admission-INDEPENDENT
    #     termination clock above would be wrong: the source can still be
    #     actively producing frames through a long streak of gate rejections,
    #     and termination must not mistake that for the source going quiet.
    #   - OnlineController's _input_window/pause_idle_threshold_s (only when
    #     --streaming_online_controller is on) IS the same conceptual signal
    #     as this clock — record_frame_arrival() fires admission-independently,
    #     "each time a new RGB-D frame is released from the source" — just
    #     wall-clock instead of iteration-count, because that controller
    #     reasons about real elapsed time for backpressure/pacing independent
    #     of GPU speed. Not merged into one shared counter (different units,
    #     different consumers with genuinely different needs), but both
    #     ultimately answer the same "is the source still alive" question.
    # For a finite bag, this naturally produces a clean tail of
    # exactly streaming_refinement_iterations once the last source frame has
    # been seen (no more frames will ever arrive, so nothing re-arms it) — for
    # a genuinely live source, this is exactly "keep going while frames
    # arrive." --streaming_max_iterations remains available as a genuine,
    # OPTIONAL hard safety ceiling for bounded/ablation runs (0 = none, rely
    # purely on the rolling rule) — it is no longer a silent truncator: if it
    # fires before the rolling rule would have stopped naturally, that's
    # logged so it's a visible, deliberate trade-off, not an unnoticed cap.
    _max_iters_cap = int(getattr(args, "streaming_max_iterations", 0) or 0)
    _refinement_iters = max(1, int(args.streaming_refinement_iterations))
    # The rolling idle-stop only applies to online_mode runs (bag-replay-via-ros2
    # or genuinely-live camera). opt.iterations is the artificial 2^31 sentinel
    # there (eligsir/core/stop_policy.py), so real termination MUST be
    # governed by frame arrival. For a non-online_mode run, opt.iterations is
    # already a deliberate, finite, real budget set by the user — a mid-run
    # idle stretch (e.g. a deliberately long post-ingestion refinement window)
    # is normal there and must not be cut short by this rule.
    _use_rolling_idle_stop = (
        bool(getattr(args, "online_mode", False))
        and _post_mapping_refinement_steps == 0
    )
    _last_ingest_iter = int(_ckpt_streaming_meta.get("last_ingest_iter", first_iter)) \
        if _ckpt_streaming_meta else first_iter

    def _capture_exact_trainer_owner() -> ExactTrainerState:
        train_indices = {
            int(index)
            for camera in streaming_scene.train_cameras
            if (index := _supervision_source_index(camera)) is not None
        }
        test_indices = {
            int(index)
            for camera in streaming_scene.getTestCameras()
            if (index := _supervision_source_index(camera)) is not None
        }
        source_dispositions = {
            index: (
                "admitted" if index in train_indices
                else "holdout" if index in test_indices
                else "rejected"
            )
            for index in range(int(streaming_scene._source_idx))
        }
        bootstrap_source_indices = [
            int(index)
            for camera in streaming_scene.train_cameras[: streaming_scene._bootstrap_count]
            if (index := _supervision_source_index(camera)) is not None
        ]
        return ExactTrainerState(
            {
                "source_idx": int(streaming_scene._source_idx),
                "source_frame_count": int(streaming_scene._frame_count),
                "n_frames_ingested": int(n_frames_ingested),
                "last_ingest_iter": int(_last_ingest_iter),
                "optimizer_step_id": int(_clk_optimizer_step_id),
                "supervision_view_count": int(_clk_supervision_view_count),
                "rasterizer_call_count": int(_clk_rasterizer_call_count),
                "supervised_pixel_count": int(_clk_supervised_pixel_count),
                "sensor_frame_count": int(_clk_sensor_frame_count),
                "mutation_event_id": int(_clk_mutation_event_id),
                "depth_mutation_event_id": int(_clk_depth_mutation_event_id),
                "mcmc_mutation_event_id": int(_clk_mcmc_mutation_event_id),
                "total_inserted": int(total_inserted),
                "last_flush_iteration": int(_last_flush_iteration),
                "iter_since_new_frame": int(
                    _streaming_optimization_runtime.iter_since_new_frame
                    if _streaming_optimization_runtime is not None
                    else _iter_since_new_frame
                ),
                "scene_steps_since_new_frame": int(
                    getattr(streaming_scene, "_steps_since_new_frame", 0)
                ),
                "opt_batch_distinct_total": int(_opt_batch_distinct_total),
                "opt_batch_count": int(_opt_batch_count),
                "per_source_supervision_counts": dict(_per_source_supervision_counts),
                "bootstrap_source_indices": bootstrap_source_indices,
                "source_dispositions": source_dispositions,
                "insertion_batch_empty": not _insertion_batch,
                "consolidation_queue_empty": not _consolidation_burst_queue,
            }
        )

    def _capture_exact_causal_fork_trainer_owner(
        fork_boundary_flush,
    ) -> ExactCausalForkTrainerState:
        train_indices = {
            int(index)
            for camera in streaming_scene.train_cameras
            if (index := _supervision_source_index(camera)) is not None
        }
        test_indices = {
            int(index)
            for camera in streaming_scene.getTestCameras()
            if (index := _supervision_source_index(camera)) is not None
        }
        source_dispositions = {
            index: (
                "admitted" if index in train_indices
                else "holdout" if index in test_indices
                else "rejected"
            )
            for index in range(int(streaming_scene._source_idx))
        }
        queue_indices = []
        for camera in _consolidation_burst_queue:
            source_index = _supervision_source_index(camera)
            if source_index is None:
                raise RuntimeError(
                    "exact causal fork cannot serialize a consolidation view "
                    "without a stable source index"
                )
            queue_indices.append(int(source_index))
        bootstrap_source_indices = [
            int(index)
            for camera in streaming_scene.train_cameras[: streaming_scene._bootstrap_count]
            if (index := _supervision_source_index(camera)) is not None
        ]
        return ExactCausalForkTrainerState(
            {
                "source_idx": int(streaming_scene._source_idx),
                "source_frame_count": int(streaming_scene._frame_count),
                "n_frames_ingested": int(n_frames_ingested),
                "last_ingest_iter": int(_last_ingest_iter),
                "optimizer_step_id": int(_clk_optimizer_step_id),
                "supervision_view_count": int(_clk_supervision_view_count),
                "rasterizer_call_count": int(_clk_rasterizer_call_count),
                "supervised_pixel_count": int(_clk_supervised_pixel_count),
                "sensor_frame_count": int(_clk_sensor_frame_count),
                "mutation_event_id": int(_clk_mutation_event_id),
                "depth_mutation_event_id": int(_clk_depth_mutation_event_id),
                "mcmc_mutation_event_id": int(_clk_mcmc_mutation_event_id),
                "total_inserted": int(total_inserted),
                "last_flush_iteration": int(_last_flush_iteration),
                "iter_since_new_frame": int(
                    _streaming_optimization_runtime.iter_since_new_frame
                    if _streaming_optimization_runtime is not None
                    else _iter_since_new_frame
                ),
                "scene_steps_since_new_frame": int(
                    getattr(streaming_scene, "_steps_since_new_frame", 0)
                ),
                "opt_batch_distinct_total": int(_opt_batch_distinct_total),
                "opt_batch_count": int(_opt_batch_count),
                "per_source_supervision_counts": dict(_per_source_supervision_counts),
                "bootstrap_source_indices": bootstrap_source_indices,
                "source_dispositions": source_dispositions,
                "insertion_batch_empty": not _insertion_batch,
                "consolidation_queue_source_indices": queue_indices,
                "fork_boundary_flush": dict(fork_boundary_flush),
                "admission_work_signature": build_admission_work_signature(
                    source_dispositions,
                    optimizer_step_id=int(_clk_optimizer_step_id),
                    supervision_view_count=int(_clk_supervision_view_count),
                    sensor_frame_count=int(_clk_sensor_frame_count),
                ),
            }
        )

    _exact_resume_identity = {
        key: "unavailable"
        for key in (
            "source_hash", "holdout_hash", "resolved_config_hash", "runtime_hash"
        )
    }
    _exact_fork_iteration = int(
        getattr(args, "streaming_exact_fork_iteration", -1)
    )
    _is_causal_fork_resume = (
        _ckpt_streaming_meta.get("checkpoint_reason") == "causal_fork"
    )
    if (
        _post_mapping_refinement_steps > 0
        or _exact_fork_iteration >= 0
        or _exact_resume_envelope is not None
    ):
        _exact_resume_identity = _exact_streaming_identity(
            streaming_scene,
            dataset,
            args,
            opt,
            ingestion_mode=_ingestion_mode,
            densification_strategy=densification_strategy,
            model_layout=model_layout,
            causal_fork=(_exact_fork_iteration >= 0 or _is_causal_fork_resume),
        )

    def _exact_refinement_runtime(
        checkpoint_reason, streaming_phase, envelope, causal_fork=None,
    ):
        _scheduled_multires = int(
            getattr(args, "streaming_multires_low_until_iteration", 0) or 0
        )
        _reactive_multires = (
            _scheduled_multires <= 0
            and (
                float(getattr(args, "streaming_multires_it_v_floor", 30.0)) > 0.0
                or float(getattr(args, "streaming_multires_map_fps_floor", 3.0)) > 0.0
            )
        )
        return {
            "checkpoint_reason": checkpoint_reason,
            "ingestion_mode": _ingestion_mode,
            "live_source": str(getattr(dataset, "source_path", "")).startswith(
                "ros2://"
            ),
            "async_ingestion": bool(
                getattr(args, "streaming_async_ingest", False)
            ),
            "online_wall_time_controller": bool(
                getattr(args, "streaming_online_controller", False)
                or getattr(args, "streaming_wallclock", False)
            ),
            "online_mode": bool(getattr(args, "online_mode", False)),
            "bounded_queue_pipeline": bool(
                getattr(args, "streaming_bounded_queue_pipeline", False)
            ),
            "sampling_mode": str(
                getattr(args, "streaming_sampling_mode", "legacy")
            ),
            "residual_replay_weight": float(
                getattr(args, "streaming_residual_replay_weight", 0.0)
            ),
            "growth_staleness_max_iters": int(
                getattr(args, "streaming_growth_staleness_max_iters", 0) or 0
            ),
            "adaptive_schedule_horizons": bool(
                getattr(args, "streaming_adaptive_schedule_horizons", False)
            ),
            "adaptive_capacity": bool(
                getattr(args, "streaming_adaptive_capacity", False)
            ),
            "pose_or_loop_correction": bool(
                _loop_closure_enabled or _lc_apply_correction
            ),
            "reactive_multires": _reactive_multires,
            # Explicit audit of every asynchronous/concurrent path. The two
            # viewer lanes are observational readers only: map output samples
            # deterministic cloned rows under map_backend.state_lock, and the
            # progress-video worker consumes detached panels without model/RNG
            # mutation. Computational workers remain unsupported below.
            "async_ingest_worker": _ingest_worker is not None,
            "deferred_slam_worker": _deferred_slam_tracker is not None,
            "bounded_pipeline_worker": _coordinator.bounded_pipeline is not None,
            "save_worker": save_worker is not None,
            "report_worker": report_worker is not None,
            "map_output_reader": _map_output_lane is not None,
            "progress_video_reader": _pv_cpu_queue is not None,
            "refinement_resolution_tier": (
                streaming_phase.get("refinement_resolution_tier")
                if isinstance(streaming_phase, dict)
                else None
            ),
            "strategy": densification_strategy,
            "admission_mode": str(args.streaming_frame_admission),
            "adaptive_keyframe_quality": bool(
                getattr(args, "streaming_keyframe_quality_adaptive", False)
            ),
            "tracking_confidence": bool(
                getattr(args, "mapping_scheduler_tracking_confidence_enabled", False)
            ),
            "keyframe_tracking_confidence": bool(
                getattr(args, "streaming_tracking_confidence_enabled", False)
            ),
            "temporal_depth_modes": any(
                bool(getattr(args, name, False))
                for name in (
                    "streaming_depth_temporal_check",
                    "streaming_depth_temporal_require_for_sensor_closer",
                    "streaming_depth_temporal_require_for_insert",
                    "streaming_depth_temporal_require_for_seed",
                )
            ),
            "bootstrap_seed_decay": int(
                getattr(args, "streaming_bootstrap_seed_decay_frames", 0) or 0
            ) > 0,
            "assimilation_control": bool(
                assimilation_control_config()["enabled"]
            ),
            "historical_repair_controller": bool(_partition_focus_enabled),
            "backpressure": _backpressure_owner is not None,
            "insertion_batching": _batch_frames > 1 or _batch_steps > 0,
            "dataset_pose_source": bool(
                is_dataset_pose_source(_pose_source_name)
            ),
            "shadow_utility": bool(
                getattr(args, "streaming_shadow_utility_enabled", False)
            ),
            "admission_rollout_mode": str(
                getattr(args, "streaming_admission_rollout_mode", "shadow")
            ),
            "insertion_batch_empty": not _insertion_batch,
            "consolidation_queue_empty": not _consolidation_burst_queue,
            "computational_queues_serialized": (
                _ingest_worker is None
                and _deferred_slam_tracker is None
                and _coordinator.bounded_pipeline is None
                and not _insertion_batch
            ),
            "state_writers_joined": True,
            "streaming_phase": streaming_phase,
            "causal_fork": causal_fork,
            "envelope": envelope,
        }

    def _normalize_exact_refinement_owners(
        envelope, owners, trainer_owner
    ):
        nonlocal _observation_db

        def _rematerialize_exact_camera(source_index: int):
            frame = streaming_scene._get_source_frame(source_index)
            return streaming_scene._frame_to_camera(frame, load_image=False)

        camera_resolver = CachedCameraResolver(_rematerialize_exact_camera)
        restore_exact_resume_owners(
            envelope=envelope,
            expected_identity=_exact_resume_identity,
            owners=owners,
            resolver=camera_resolver,
            restore_rng=False,
        )
        restored_state = trainer_owner.state
        bootstrap_indices = set(restored_state["bootstrap_source_indices"])
        for source_index, camera in camera_resolver.cache.items():
            camera._is_bootstrap = source_index in bootstrap_indices
        restored_tier = _resolution_controller.decide(mapping_done=True)
        if restored_tier != "high":
            raise RuntimeError(
                "exact refinement continuation requires native HIGH resolution"
            )
        _observation_db = FrameObservationDB(
            device="cuda",
            max_resolution_cache_frames=int(
                getattr(args, "streaming_resolution_cache_max_frames", 400)
            ),
        )
        streaming_scene.set_observation_db(_observation_db)
        rebuild_exact_refinement_observations(
            db=_observation_db,
            cameras=[
                *streaming_scene.train_cameras,
                *streaming_scene.getTestCameras(),
            ],
            source_dispositions=restored_state["source_dispositions"],
            get_frame=streaming_scene._get_source_frame,
            record=record_frame_observation,
            low_res_divisor=_multires_low_divisor,
            depthless_rgb_cache=_multires_depthless_rgb_cache,
        )
        return restored_state

    def _normalize_exact_causal_fork_owners(envelope, owners, trainer_owner):
        nonlocal _observation_db, _consolidation_burst_queue

        def _rematerialize_exact_camera(source_index: int):
            frame = streaming_scene._get_source_frame(source_index)
            return streaming_scene._frame_to_camera(frame, load_image=False)

        camera_resolver = CachedCameraResolver(_rematerialize_exact_camera)
        restore_exact_resume_owners(
            envelope=envelope,
            expected_identity=_exact_resume_identity,
            owners=owners,
            resolver=camera_resolver,
            restore_rng=False,
        )
        restored_state = trainer_owner.state
        bootstrap_indices = set(restored_state["bootstrap_source_indices"])
        for source_index, camera in camera_resolver.cache.items():
            camera._is_bootstrap = source_index in bootstrap_indices
        _consolidation_burst_queue.clear()
        _consolidation_burst_queue.extend(
            camera_resolver(source_index)
            for source_index in restored_state["consolidation_queue_source_indices"]
        )
        streaming_scene.set_observation_db(_observation_db)
        if _keyframe_admitter._observation_db is not _observation_db:
            raise RuntimeError(
                "exact causal fork observation DB identity is not shared"
            )
        return restored_state

    if checkpoint:
        _exact_runtime = _exact_refinement_runtime(
            _ckpt_streaming_meta.get("checkpoint_reason"),
            _ckpt_streaming_meta.get("streaming_phase"),
            _exact_resume_envelope,
            _ckpt_streaming_meta.get("causal_fork"),
        )
        if _is_causal_fork_resume:
            try:
                validate_causal_fork_treatment(
                    _ckpt_streaming_meta.get("causal_fork"),
                    resume_fork_iteration=_exact_fork_iteration,
                    resume_seed_lineage_freeze_from_iter=int(
                        getattr(args, "streaming_seed_lineage_freeze_from_iter", -1)
                    ),
                )
            except ValueError as exc:
                raise RuntimeError(f"invalid exact causal fork resume: {exc}") from exc
            _exact_resume_verdict = qualify_exact_causal_fork_resume(
                _exact_runtime,
                _exact_resume_identity,
                REQUIRED_CAUSAL_FORK_OWNERS,
            )
        else:
            _exact_resume_verdict = qualify_exact_refinement_resume(
                _exact_runtime, _exact_resume_identity, REQUIRED_EXACT_OWNERS
            )
        _exact_status = {
            "classification": _exact_resume_verdict.evidence_class,
            "reasons": list(_exact_resume_verdict.reasons),
            "checkpoint": str(checkpoint),
        }
        _write_streaming_phase_metadata(
            args.model_path, {"exact_resume": _exact_status}
        )
        print(
            "[streaming] checkpoint resume classification: "
            f"{_exact_resume_verdict.evidence_class}; "
            f"reasons={list(_exact_resume_verdict.reasons)}",
            flush=True,
        )
        if "resume_identity_missing_or_mismatch" in _exact_resume_verdict.reasons:
            _saved_identity = (
                _exact_resume_envelope.get("identity", {})
                if isinstance(_exact_resume_envelope, dict)
                else {}
            )
            _identity_mismatch = {
                key: {"checkpoint": _saved_identity.get(key), "resolved": value}
                for key, value in _exact_resume_identity.items()
                if _saved_identity.get(key) != value
            }
            print(
                f"[streaming] exact-resume identity mismatch: {_identity_mismatch}",
                flush=True,
            )
        if _post_mapping_refinement_steps > 0 and not _exact_resume_verdict.eligible:
            raise RuntimeError(
                "post-mapping refinement continuation requires an exact mapping-endpoint "
                "checkpoint: " + "; ".join(_exact_resume_verdict.reasons)
            )
        if _is_causal_fork_resume and not _exact_resume_verdict.eligible:
            raise RuntimeError(
                "causal-fork continuation requires an exact mapping checkpoint: "
                + "; ".join(_exact_resume_verdict.reasons)
            )
        if _exact_resume_verdict.eligible and _is_causal_fork_resume:
            _trainer_payload = _exact_resume_envelope["owners"]["trainer"]
            _trainer_owner = ExactCausalForkTrainerState(
                {
                    key: value for key, value in _trainer_payload.items()
                    if key != "owner_schema"
                }
            )
            _exact_owners = {
                "mcmc": mcmc_strategy,
                "keyframes": streaming_scene._keyframe_store,
                "replay": streaming_scene._replay_view_store,
                "dirty_regions": dirty_region_queue,
                "phase": _phase_controller,
                "cadence": _cadence,
                "trainer": _trainer_owner,
                "mapping_clocks": _coordinator.four_clocks,
                "observations": streaming_scene._observation_store,
                "frame_scheduler": scheduler,
                "admission": _keyframe_admitter,
                "scene": ExactCausalForkSceneState(
                    streaming_scene,
                    submap_manager=_submap_manager,
                    gaussians=gaussians,
                ),
                "frame_observations": ExactCausalForkObservationDBState(
                    _observation_db
                ),
            }
            restored = _normalize_exact_causal_fork_owners(
                _exact_resume_envelope, _exact_owners, _trainer_owner
            )
            streaming_scene._source_idx = restored["source_idx"]
            n_frames_ingested = restored["n_frames_ingested"]
            _last_ingest_iter = restored["last_ingest_iter"]
            _clk_optimizer_step_id = restored["optimizer_step_id"]
            _clk_supervision_view_count = restored["supervision_view_count"]
            _clk_rasterizer_call_count = restored["rasterizer_call_count"]
            _clk_supervised_pixel_count = restored["supervised_pixel_count"]
            _clk_sensor_frame_count = restored["sensor_frame_count"]
            _clk_mutation_event_id = restored["mutation_event_id"]
            _clk_depth_mutation_event_id = restored["depth_mutation_event_id"]
            _clk_mcmc_mutation_event_id = restored["mcmc_mutation_event_id"]
            total_inserted = restored["total_inserted"]
            _last_flush_iteration = restored["last_flush_iteration"]
            _iter_since_new_frame = restored["iter_since_new_frame"]
            if _streaming_optimization_runtime is not None:
                _streaming_optimization_runtime.iter_since_new_frame = int(
                    _iter_since_new_frame
                )
            streaming_scene._steps_since_new_frame = restored[
                "scene_steps_since_new_frame"
            ]
            _opt_batch_distinct_total = restored["opt_batch_distinct_total"]
            _opt_batch_count = restored["opt_batch_count"]
            _per_source_supervision_counts = dict(
                restored["per_source_supervision_counts"]
            )
            activate_causal_fork_treatment(
                _seed_lineage_freeze,
                causal_fork_contract=_ckpt_streaming_meta.get("causal_fork"),
                resume_seed_lineage_freeze_from_iter=int(
                    getattr(args, "streaming_seed_lineage_freeze_from_iter", -1)
                ),
            )
        elif _exact_resume_verdict.eligible:
            _trainer_owner = _capture_exact_trainer_owner()
            _exact_owners = {
                "mcmc": mcmc_strategy,
                "keyframes": streaming_scene._keyframe_store,
                "replay": streaming_scene._replay_view_store,
                "dirty_regions": dirty_region_queue,
                "phase": _phase_controller,
                "cadence": _cadence,
                "trainer": _trainer_owner,
                "mapping_clocks": _coordinator.four_clocks,
                "observations": streaming_scene._observation_store,
            }
            restored = _normalize_exact_refinement_owners(
                _exact_resume_envelope,
                _exact_owners,
                _trainer_owner,
            )
            streaming_scene._source_idx = restored["source_idx"]
            n_frames_ingested = restored["n_frames_ingested"]
            _last_ingest_iter = restored["last_ingest_iter"]
            _clk_optimizer_step_id = restored["optimizer_step_id"]
            _clk_supervision_view_count = restored["supervision_view_count"]
            _clk_rasterizer_call_count = restored["rasterizer_call_count"]
            _clk_supervised_pixel_count = restored["supervised_pixel_count"]
            _clk_sensor_frame_count = restored["sensor_frame_count"]
            _clk_mutation_event_id = restored["mutation_event_id"]
            _clk_depth_mutation_event_id = restored["depth_mutation_event_id"]
            _clk_mcmc_mutation_event_id = restored["mcmc_mutation_event_id"]
            total_inserted = restored["total_inserted"]
            _last_flush_iteration = restored["last_flush_iteration"]
            _iter_since_new_frame = restored["iter_since_new_frame"]
            if _streaming_optimization_runtime is not None:
                _streaming_optimization_runtime.iter_since_new_frame = int(
                    _iter_since_new_frame
                )
            streaming_scene._steps_since_new_frame = restored["scene_steps_since_new_frame"]
            _opt_batch_distinct_total = restored["opt_batch_distinct_total"]
            _opt_batch_count = restored["opt_batch_count"]
            _per_source_supervision_counts = dict(restored["per_source_supervision_counts"])
            _streaming_phase = dict(_ckpt_streaming_meta["streaming_phase"])
            phase_timer.end("training")
            phase_timer.start("post_mapping_refinement")
            _refinement_phase_wall_start = time.perf_counter()
            # Issue #1833: mirror the fresh-transition override below for a
            # resume that restarts directly into the refinement tail.
            _refinement_depth_loss_weight = float(
                getattr(args, "streaming_refinement_depth_loss_weight", -1.0)
            )
            if _refinement_depth_loss_weight >= 0.0:
                depth_loss_weight = _refinement_depth_loss_weight
                use_depth_loss = depth_loss_weight > 0
                if _streaming_optimization_runtime is not None:
                    _streaming_optimization_runtime.loss_assembly_kwargs[
                        "depth_loss_weight"
                    ] = depth_loss_weight
    # Issue #1044 growth-staleness brake: iteration of the last real map commit
    # (a depth-insertion cycle that actually added >0 Gaussians), tracked
    # independently of `_last_ingest_iter` (which re-arms on every INGESTED
    # frame regardless of whether it produced any new geometry). Not restored
    # from a checkpoint (no schema change here) -- a resumed run simply gets
    # the same first-iter grace period as a fresh one, which only means the
    # brake cannot fire in the first `streaming_growth_staleness_max_iters`
    # iterations after a resume; it can never cause a FALSE brake activation.
    _last_map_commit_iter = first_iter
    # Adaptive schedule horizons (2026-07-18): rather than hand-tuning
    # --streaming_noise_stop_horizon / --position_lr_max_steps / anneal
    # horizon per scene size, detect the real mapping-end point (no new
    # frame ingested for streaming_adaptive_mapping_end_threshold iterations
    # -- a small threshold, distinct from streaming_refinement_iterations'
    # much larger idle-stop-the-whole-run threshold) and, once, snap MCMC
    # noise/growth-stop, the count-loss anneal clock, and the position-LR
    # schedule to that point. Default off (behavior-neutral).
    _adaptive_horizons_on = bool(getattr(args, "streaming_adaptive_schedule_horizons", False))
    _adaptive_mapping_end_threshold = int(getattr(args, "streaming_adaptive_mapping_end_threshold", 1000))
    _adaptive_horizons_applied = False
    _stall_warn_iters = int(getattr(args, "streaming_stall_warn_iters", 1000) or 0)
    _stall_warned = False
    # W3-14B: reuse the same --profile mechanism train.py's legacy loop already
    # has (eligsir/cli/parser.py's --profile flag was previously wired only
    # into train.py, not this streaming loop). Off by default (args.profile is
    # None unless explicitly passed); when set, captures a torch.profiler CPU+CUDA
    # trace across a short window of real streaming-loop iterations, itemized by
    # operator/function -- used to root-cause the W3-02 engine throughput
    # regression (see docs/architecture/w3_02_completion.md's 2026-08-09 section).
    _profile_path = getattr(args, "profile", None)
    _profile_wait = max(0, int(getattr(args, "profile_wait", 1)))
    _profile_active = max(1, int(getattr(args, "profile_active", 50)))
    if _profile_path:
        _prof = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=_profile_wait, warmup=1, active=_profile_active, repeat=1),
            record_shapes=True,
            profile_memory=False,
            with_stack=False,
        )
        _prof.__enter__()
        print(
            f"[profile] Profiling enabled (streaming loop) -> {_profile_path} "
            f"(wait={_profile_wait}, warmup=1, active={_profile_active})",
            flush=True,
        )
    else:
        _prof = None
    # Issue #358 Level A: lazily constructed at the first psnr-chart tick when
    # --streaming_local_quality_cells is on; stays None (and costs nothing) for
    # every run that leaves the flag off, which is the default.
    _local_quality_tracker = None
    # Number of frozen cell maps the Level B co-visibility snapshot on disk
    # reflects; -1 = never written. Only used to avoid rewriting an unchanged
    # snapshot every tick.
    _lq_covis_written = -1
    # Issue #396: bounded GPU residency for replay-paged camera images. Shares
    # the --streaming_offload_camera_images gate with every existing page-in/
    # page-out site, so that flag still disables the whole paging policy rather
    # than only half of it.
    from eligsir.utils.camera_residency import CameraResidencyLedger
    _residency_ledger = CameraResidencyLedger(
        enabled=bool(getattr(args, "streaming_offload_camera_images", True))
    )
    _loop_iteration_limit = (
        int(opt.iterations)
        if _post_mapping_refinement_steps == 0
        else 2**31 - 1
    )
    _optimizer_iteration_limit = _loop_iteration_limit
    _skip_periodic_checkpoint_iteration = -1
    _exact_fork_written = _is_causal_fork_resume
    # Exact continuation restores RNG LAST: setup/profiler/residency creation
    # above cannot consume the saved streams and perturb the first refined step.
    if _exact_resume_verdict is not None and _exact_resume_verdict.eligible:
        restore_rng_state(_exact_resume_envelope["rng"])
    _backpressure_control_tick = 0
    for iteration in range(first_iter, _loop_iteration_limit + 1):
        _backpressure_control_tick += 1
        # Issue #396: release the PREVIOUS iteration's replay page-ins. Done at
        # the top of the next iteration (not after each render) because every
        # consumer of iteration N-1 -- the primary view, the B>1 supervision
        # batch on either the inline or engine path, any post-loss telemetry --
        # has provably returned by now, so there is no window in which this can
        # page a tensor out from under a live consumer. HOT cameras (the tail
        # --streaming_keyframe_window of train_cameras, plus current_camera) are
        # retained. See utils/camera_residency.py for why this shape was chosen
        # over per-call offloads or a per-iteration full sweep.
        _residency_ledger.release(streaming_scene.hot_camera_ids())
        # Four-clocks telemetry (F3 fix, 2026-08-06): poll the backend's
        # version counters every iteration, not just inside
        # MappingCoordinator.step() -- pruning/relocation/loop-closure
        # corrections can mutate the map outside that call, and this is
        # the only way last_topology_change_age_ms/
        # last_parameter_update_age_ms see those changes too. Cheap: plain
        # int comparisons, no tensor materialization (see
        # MappingCoordinator.poll_map_activity's docstring).
        _coordinator.poll_map_activity()
        # Refresh the snapshot the SIGTERM handler reads (cheap: dict of refs
        # + a few ints, no copy of tensors). Must happen before any
        # early-continue/break in this iteration so an emergency checkpoint
        # always reflects genuinely-committed progress, never a half-updated
        # frame count from a still-in-flight ingest.
        _emergency_state["gaussians"] = gaussians
        _emergency_state["map_backend"] = map_backend
        _emergency_state["iteration"] = iteration
        _emergency_state["n_frames_ingested"] = n_frames_ingested
        _emergency_state["last_ingest_iter"] = _last_ingest_iter
        _emergency_state["source_idx"] = streaming_scene._source_idx
        _emergency_state["coverage_ledger"] = _coverage_snapshot()
        _emergency_state["model_path"] = args.model_path
        _emergency_state["per_source_supervision_counts"] = _per_source_supervision_counts
        if (
            _post_mapping_refinement_steps > 0
            and not _phase_controller.in_refinement
            and iteration > _mapping_iteration_budget
        ):
            raise RuntimeError(
                "mapping budget exhausted before full source disposition: "
                f"source_idx={streaming_scene._source_idx} "
                f"frame_count={streaming_scene._frame_count} "
                f"budget={_mapping_iteration_budget}"
            )
        if (
            _max_iters_cap > 0
            and not _phase_controller.in_refinement
            and iteration > _max_iters_cap
        ):
            _idle_for = iteration - _last_ingest_iter
            if _use_rolling_idle_stop and _idle_for < _refinement_iters:
                print(
                    f"[streaming] reached streaming_max_iterations={_max_iters_cap}; stopping "
                    f"(note: a frame was ingested only {_idle_for} optimization steps ago — the rolling "
                    f"idle rule would have kept going for {_refinement_iters - _idle_for} more "
                    "optimization steps; this hard ceiling truncated real ingestion work, not just idle time).",
                    flush=True,
                )
            else:
                print(f"[streaming] reached streaming_max_iterations={_max_iters_cap}; stopping.", flush=True)
            break
        if _use_rolling_idle_stop and iteration - _last_ingest_iter >= _refinement_iters:
            print(
                f"[streaming] no new frame ingested for {_refinement_iters} optimization steps "
                f"(last at step={_last_ingest_iter}) — treating source as exhausted/idle; stopping.",
                flush=True,
            )
            break
        # Early stall warning (2026-07-26): loud one-shot notice well before
        # the full _refinement_iters idle-stop -- found via a real floor3 run
        # where ingestion silently stalled (live SLAM frontend stopped
        # publishing poses) and the run burned ~30 minutes idling to the
        # default 50000-iteration threshold before anyone noticed. This does
        # NOT stop the run (a genuine mapping->refinement transition looks
        # identical for the first _stall_warn_iters iterations) -- it only
        # makes a real stall visible immediately instead of retroactively.
        if (
            _use_rolling_idle_stop and not _stall_warned and _stall_warn_iters > 0
            and iteration - _last_ingest_iter >= _stall_warn_iters
        ):
            print(
                f"[streaming] WARNING: no new frame ingested for {_stall_warn_iters} optimization steps "
                f"(last at step={_last_ingest_iter}) -- this may be a genuine mapping-end, or a "
                "silently stalled source (e.g. a live SLAM frontend that stopped publishing "
                f"poses). Will hard-stop at idle={_refinement_iters} if this doesn't recover.",
                flush=True,
            )
            if tb_writer is not None:
                tb_writer.add_scalar("streaming/ingestion_stall_warning", 1.0, iteration)
            _stall_warned = True
        if should_apply_adaptive_schedule_horizons(
            _adaptive_horizons_on, _adaptive_horizons_applied,
            _last_ingest_iter, first_iter, iteration, _adaptive_mapping_end_threshold,
        ):
            _adaptive_horizons_applied = True
            # 2026-07-18 correction: this block used to ALSO stop MCMC growth
            # (mcmc_stop_growth_iter) and finalize the position-LR schedule at
            # the detected mapping-end point. Real-GPU testing found coupling
            # growth-stop to mapping-end starves the model of capacity (MCMC
            # growth is an ongoing densification mechanism independent of new-
            # frame arrival, and mapping typically ends FAR earlier than growth
            # should stop -- see apply_online_schedule_overrides's docstring
            # for the full floor3 finding: N frozen at ~13k vs a healthy ~230k,
            # test PSNR 19dB vs 22dB). Scaled back to only the two mechanisms
            # validated as safe to auto-stop early: position noise (a
            # perturbation on top of gradient-driven updates, not a hard
            # growth/learning stop) and the count-loss anneal clock (relaxes
            # an existing suppression loss, doesn't halt anything). Growth-stop
            # timing and the position-LR schedule remain manual
            # (--streaming_growth_stop_horizon / --position_lr_max_steps)
            # pending a real adaptive signal for "growth has actually
            # saturated" (NOT the same as "mapping ended").
            opt.mcmc_noise_stop_iter = iteration
            # mcmc_strategy.py reads mcmc_noise_stop_iter from `args`, never
            # `opt` (see the 2026-07-06b opt/args mirror-sync note at this
            # function's own apply_online_schedule_overrides call site) --
            # without mirroring onto args too, the write above would be
            # silently invisible to the strategy that actually reads it.
            args.mcmc_noise_stop_iter = iteration
            if getattr(opt, "_anneal_clock", None) is not None:
                # AnnealClock is a frozen dataclass -- replace, don't mutate.
                # Callers re-fetch via getattr(opt, "_anneal_clock", None)
                # every use (never cache the object), so reassigning here is
                # immediately visible.
                from eligsir.core.stop_policy import AnnealClock
                opt._anneal_clock = AnnealClock(iteration)
            print(
                f"[ADAPTIVE-HORIZON] step={iteration}: no new frame ingested for "
                f"{_adaptive_mapping_end_threshold} optimization steps (last at step={_last_ingest_iter}) "
                "— treating mapping as ended; MCMC position noise stopped and count-loss "
                "anneal clock finalized to this optimization step (growth-stop and position-LR "
                "schedule are NOT auto-adjusted -- see 2026-07-18 correction note above).",
                flush=True,
            )
            if tb_writer is not None:
                tb_writer.add_scalar("streaming/adaptive_horizon_applied_iter", iteration, iteration)
        set_compile_iteration(iteration)
        _debug_bootstrap_this_iter = (
            _debug_bootstrap_motion
            and (_debug_bootstrap_first_iters <= 0 or iteration <= _debug_bootstrap_first_iters)
        )

        # Measure per-iteration wall time for dataset_fps simulated clock
        _now = time.perf_counter()
        _iter_dt = _now - _iter_wall_start
        _iter_wall_start = _now

        # Reset this iteration's Phase-2 depth-insertion count so the
        # assimilation-pressure telemetry below (streaming/assimilation_raw_added,
        # streaming/assimilation_source_insert_rate) reflects THIS iteration, not
        # a stale value from whenever insertion last actually ran. Without this,
        # `locals().get("added", 0)` (used below specifically to survive
        # iterations where the gated insertion block never executes at all)
        # instead silently returns an arbitrarily-old value on EVERY iteration
        # between insertion events (2026-08-11 review finding) -- insertion only
        # happens roughly once per streaming_steps_per_frame iterations, so most
        # iterations were reporting a Python-local-variable leftover, not "no
        # insertion this iteration".
        added = 0

        online_clock.session_step += 1
        online_clock.wall_time_s += _iter_dt
        if iteration > 1:  # skip first iter which includes drain-loop startup
            phase_timer.record_iter(_iter_dt)
        if _PROFILE_PHASES and iteration % 100 == 0:
            phase_timer.report_window()
        # Issue #562: close out the previous iteration's wall-clock partition.
        # Also self-heals a leaked region bracket (see DutyCycleTimeline.
        # iteration_boundary). One `is None` test when telemetry is off.
        if _duty is not None:
            _duty_row = _duty.iteration_boundary(iteration)
            if _duty_row is not None and tb_writer is not None:
                for _k, _v in _duty.tb_scalars().items():
                    tb_writer.add_scalar(_k, _v, iteration)
        # Milestone 2: feed measured iteration cost into the OnlineController
        # (only the EMA; mode is updated once per ingestion event below).
        if _online_controller is not None:
            _online_controller.record_iter_dt(_iter_dt if iteration > 1 else None)
        # Milestone 3: reset the per-iter ingestion budget counter.
        if _burst_budget is not None:
            _burst_budget.begin_iter(iteration)

        _decision = None  # Phase 8: reset per iteration; set inside frame ingestion block

        # ---- Frame ingestion -----------------------------------------------
        # Drain loop: ingest every frame whose pose is ready and whose release
        # time has been reached. dt is passed only on the first call to avoid
        # double-advancing the simulated clock within the same training iteration.
        _drain_dt = _iter_dt
        # has_ready_frame() is non-blocking: for live sources it only ingests
        # frames already buffered by the background pump, so the training thread
        # never stalls in get_frame() waiting at sensor rate. When nothing is
        # ready the loop falls through to a training step (concurrent mapping).
        #
        # _drain_cap bounds frames ingested per training iteration so per-frame
        # ingest work (~40 ms: frame->camera + surface accumulation) cannot
        # monopolise the loop — excess buffered frames are DEFERRED to the next
        # iterations (not dropped; coverage preserved), interleaving training.
        # Pre-existing bug, exposed by faithful resume: when the frame source
        # is already exhausted this iteration (all frames ingested — e.g. the
        # normal post-coverage refinement tail, or immediately after a resume
        # whose cursor lands past the last frame), the ingest while-loop body
        # below never executes, so is_train/new_cam/new_frame were previously
        # left unbound from a prior iteration (or entirely undefined on the
        # very first such iteration) -> UnboundLocalError at the
        # streaming_consolidation_burst check further down (default-on per
        # arguments/__init__.py). Defaulting them here every iteration is
        # correct regardless of resume — any run's tail once ingestion
        # completes was always exposed to this, just rarely hit because a
        # continuous replay/live source keeps drip-feeding frames until then.
        is_train = False
        new_cam = None
        new_frame = None
        _deferred_tracked_result = None  # set below when _deferred_slam_tracker admits a frame
        _drained_this_iter = 0
        _drain_cap = max(1, int(getattr(args, "streaming_max_drain_per_iter", 2)))
        # Variant C (--streaming_batch_ingest): widen the per-iter drain and defer
        # the per-frame cameras_extent readback to one flush after the batch.
        _batch_ingest = bool(getattr(args, "streaming_batch_ingest", False))
        if _batch_ingest:
            _drain_cap = max(_drain_cap, int(getattr(args, "streaming_batch_ingest_max", 4)))
        while (
            _drained_this_iter < _drain_cap
            and not _phase_controller.in_refinement
            and streaming_scene.has_ready_frame()
            and scheduler.should_release(iteration, dt=_drain_dt)
        ):
            # Variant B: when the async worker is on, the optimizer only commits a
            # frame whose CPU prep (decode) the worker already finished. poll() is
            # non-blocking: None => keep training on buffered data (the optimizer
            # never waits). The worker pre-decoded this exact frame (cursor order
            # preserved), so ingest_next_frame below is a warm-cache hit. No drop:
            # if the worker is behind, the frame is simply DEFERRED to a later iter.
            if _ingest_worker is not None and _ingest_worker.poll() is None:
                break
            _drained_this_iter += 1
            _drain_dt = 0.0  # subsequent calls in the same iter consume no virtual time
            # Milestone 3: BurstBudget decides whether to skip the next frame
            # (per-iter cap, CATCH_UP drop, or backlog-driven drop). The
            # decision is cheap (pose-delta only) so it sits in front of
            # the expensive ingest / admission path. Disabled under async ingest:
            # Variant B preserves full coverage via queue back-pressure (no drops).
            # Also disabled under the deferred-admission SLAM worker: the
            # peeked frame's pose is not resolved yet at this point (that's
            # the entire premise of deferred admission -- see the gate
            # below), so a pose-delta-based keyframe estimate here would
            # read a not-yet-known frame.c2w.
            # -----------------------------------------------------------
            # Issue #601: bounded backpressure owner. Runs BEFORE the
            # BurstBudget window and before ANY expensive work: at this
            # point only cheap metadata exists (index, timestamp, pose,
            # intrinsics) -- peek_next_frame() does not decode RGB/depth,
            # build a camera, upload to GPU, touch the replay buffer or
            # the multires cache. A skipped frame therefore costs nothing
            # beyond its metadata, which is the whole point.
            #
            # Coverage bookkeeping mirrors the pre-existing BurstBudget
            # drop path exactly (skip_next_frame + mark_released +
            # record_source_released), so a skipped observation still
            # counts as source-released on the #289 release clock.
            # -----------------------------------------------------------
            if (
                _backpressure_owner is not None
                and _ingest_worker is None
                and _deferred_slam_tracker is None
            ):
                _bp_peeked = streaming_scene.peek_next_frame()
                # HARD BYPASS -- holdouts and guards are never subject to
                # backpressure. They are reserved deterministically up-front and
                # the Run Contracts require the selection to be IDENTICAL across
                # admission modes; a holdout skipped here would never be realized
                # as a test camera at all (is_holdout is determined inside
                # ingest_next_frame), silently shrinking n_test, changing
                # realized_holdout_hash, and making any --baseline_run comparison
                # -- including #601's own performance acceptance -- refuse or lie.
                # This is a bypass, NOT a decision: routing it through the owner
                # as mapping_candidate=False would yield METADATA_ONLY, whose
                # materializes is also False, i.e. the same bug.
                _bp_is_holdout, _bp_is_guard = (
                    streaming_scene.holdout_status(_bp_peeked)
                    if _bp_peeked is not None
                    else (False, False)
                )
                if _bp_peeked is not None and not (_bp_is_holdout or _bp_is_guard):
                    _bp_obs = _backpressure_observation(
                        _bp_peeked, _backpressure_control_tick
                    )
                    _bp_materialize = True
                    for _bp_dec in _backpressure_owner.submit(
                        _bp_obs,
                        control_tick=_backpressure_control_tick,
                        source_time_s=float(getattr(_bp_peeked, "timestamp", 0.0)),
                    ):
                        # Feed #602's disposition ledger. This is the first
                        # real producer for those counters: #602 built the
                        # recorder and #600 declared semantics, but nothing
                        # was recording per-frame dispositions yet, so the
                        # coalesce/skip counts read "absent, with a reason".
                        _record_backpressure_disposition(_bp_dec)
                        if _bp_dec.frame_id == _bp_obs.frame_id:
                            _bp_materialize = _bp_dec.materializes
                    if not _bp_materialize:
                        _record_coverage_preingest_skip(_bp_peeked)
                        streaming_scene.skip_next_frame()
                        scheduler.mark_released()
                        _cadence.record_source_released()
                        if _online_controller is not None:
                            _online_controller.record_drop(
                                timestamp=float(
                                    getattr(_bp_peeked, "timestamp", 0.0)
                                ),
                                is_keyframe=bool(_bp_obs.cheap_keyframe),
                            )
                        continue
            if _burst_budget is not None and _ingest_worker is None and _deferred_slam_tracker is None:
                _peeked = streaming_scene.peek_next_frame()
                _burst_is_holdout, _burst_is_guard = (
                    streaming_scene.holdout_status(_peeked)
                    if _peeked is not None else (False, False)
                )
                if _peeked is not None and not (_burst_is_holdout or _burst_is_guard):
                    _peek_kf = _cheap_keyframe_estimate(_peeked)
                    _should_ingest, _ = _burst_budget.acquire(_peeked, _peek_kf)
                    if not _should_ingest:
                        _record_coverage_preingest_skip(_peeked)
                        streaming_scene.skip_next_frame()
                        scheduler.mark_released()
                        # #289 release clock: counted on the SCHEDULER side, so
                        # a frame released and then dropped by BurstBudget still
                        # counts as released. Release and ingest are different
                        # clocks; conflating them is exactly the F-vs-V
                        # confusion #278 had to disentangle.
                        _cadence.record_source_released()
                        continue
            # W3-15 deferred-admission gate: when the async SLAM worker is
            # active for this run's backend, the next source frame must not
            # be admitted (i.e. ingest_next_frame must not run for it) until
            # its REAL, asynchronously-computed pose has actually arrived.
            # poll() is non-blocking and never fabricates/predicts a pose --
            # it only ever returns a TrackedFramePacket once a genuine
            # worker result exists for that exact frame (see
            # DeferredSlamAdmissionTracker's docstring). When nothing is
            # ready yet this call, the frame is simply not admitted this
            # iteration -- training continues on buffered data and the same
            # frame is retried on a later call, never dropped silently.
            if _deferred_slam_tracker is not None:
                _deferred_tracked_result = _deferred_slam_tracker.poll(streaming_scene)
                if _deferred_tracked_result is None:
                    break  # next frame's real pose not ready yet -- go train instead
            admission_fn = None
            pregate_fn = None
            if args.streaming_frame_admission == "hybrid_keyframe":
                # Milestone 4: use the centralised KeyframeAdmitter
                # (the inline _hybrid_keyframe_admission closure is
                # retained for back-compat with non-hybrid modes but
                # the hybrid_keyframe mode goes through the new
                # wrapper).
                admission_fn = _keyframe_admission_closure
                # Only wired alongside hybrid_keyframe: admission=all must
                # admit every frame unconditionally (see _keyframe_pregate_closure
                # docstring) so pregate_fn stays None for every other mode.
                pregate_fn = _keyframe_pregate_closure
            elif args.streaming_frame_admission == "fixed_stride":
                # Issue #934: budget-matched, information-free CONTROL arm.
                # No pregate_fn: fixed_stride sees the pre-pregate
                # population, NOT hybrid_keyframe's post-filter stream
                # (that arm's pregate_fn -- motion pre-check, default on --
                # is its own policy, not the shared population). Tune
                # streaming_fixed_stride_n against the realized committed-K,
                # not a nominal value -- see FixedStrideAdmitter's docstring.
                admission_fn = _fixed_stride_admission_closure
            elif args.streaming_frame_admission == "force_all":
                # Issue #600 STRESS mode: the historical hard-all. No
                # admission_fn at all -- byte-for-byte the pre-#600 `all` path,
                # which is what makes this a valid historical-reproduction
                # control. Deliberately an explicit branch rather than a
                # fall-through, so `force_all` cannot be confused with an
                # unrecognized token that merely happens to behave the same way.
                admission_fn = None
            else:
                # Issue #600: production soft-all (`all`, and any unrecognized
                # token, which has always fallen through to the ungated path).
                admission_fn = _soft_all_admission_closure
            with _timed("m_ingest"):
                result = streaming_scene.ingest_next_frame(
                    admission_fn=admission_fn, defer_extent_update=_batch_ingest,
                    visibility_fn=_visibility_fn, record_fn=_record_fn,
                    pregate_fn=pregate_fn,
                    # #806: mandatory integrity observation -- wired
                    # unconditionally (both admission=all and
                    # hybrid_keyframe), unlike admission_fn/pregate_fn above
                    # which are hybrid_keyframe-only. The invariant this
                    # closes is "every ingested frame", not "every
                    # hybrid_keyframe frame" -- see
                    # _integrity_observer_closure's own docstring, and the
                    # PR body for the admission=all scope decision (this
                    # makes report.json's tracking_reinit_events telemetry
                    # accurate under admission=all too, but does NOT add an
                    # admission-veto enforcement mechanism for admission=all,
                    # since that mode has no admission_fn/pregate_fn to
                    # enforce it -- a separate, out-of-scope follow-up).
                    integrity_observer_fn=_integrity_observer_closure)
            if result is None:
                break  # source exhausted
            new_cam, new_frame, is_train = result
            scheduler.mark_released()
            n_frames_ingested += 1
            if _coverage_ledger is not None:
                _coverage_holdout, _coverage_guard = streaming_scene.holdout_status(new_frame)
                _coverage_ledger.record_materialized(
                    is_holdout=_coverage_holdout,
                    is_guard=_coverage_guard,
                    odom_valid=getattr(new_frame, "_odom_valid", None),
                    admitted=bool(is_train),
                )
            # PR2 cadence clocks: one real, admitted source frame.
            _clk_sensor_frame_count += 1
            # #289: release and ingest clocks. Note _clk_sensor_frame_count is
            # NOT an admission clock (it ticks here regardless of is_train) --
            # the admitted clock is recorded in the is_train branch below.
            _cadence.record_source_released()
            _cadence.record_source_ingested()
            _memprobe("after_ingest", iteration, f"n_frames={n_frames_ingested} N={gaussians.get_xyz.shape[0]}")
            _last_ingest_iter = iteration  # re-arm the rolling idle-stop (see above)
            _stall_warned = False  # re-arm the early stall warning too
            _resolution_controller.record_admission(iteration, time.perf_counter())
            # Four-clocks telemetry (Wave 2 Task 8): every ingested source
            # frame feeds the "source" clock; only frames admitted as a
            # training keyframe (is_train) feed the "admitted" clock. The
            # "commit" clock is recorded inside MappingCoordinator.step()
            # itself (see coordinator.py), and map_age/backlog/realtime
            # are derived from these three streams in .four_clocks.summary().
            _coordinator.four_clocks.record_source_frame(
                frame_timestamp=getattr(new_frame, "timestamp", None)
            )
            # F2 fix: push the source's current real queue depth (None for
            # sources with no buffered_count(), e.g. fully materialized
            # offline datasets -- see StreamingScene.queue_depth()).
            _queue_depth_now = streaming_scene.queue_depth()
            _coordinator.four_clocks.record_source_queue_depth(_queue_depth_now)
            # #289 backlog clock: reuses the depth just read -- no extra query.
            _cadence.record_backlog(_queue_depth_now)
            if _coordinator.bounded_pipeline is not None:
                # Shadow-only (see the enable-call's print banner above):
                # exercises the bounded queues/worker/prefilter on real
                # frame data for observability and acceptance-gate
                # measurement, without yet driving the admission decision
                # made above by admission_fn/_keyframe_admission_closure.
                #
                # Deliberately NOT calling
                # _coordinator.sync_source_backlog_from_pipeline() here:
                # that would overwrite four_clocks.source_queue_depth
                # (just set above from streaming_scene.queue_depth(), the
                # real, pre-existing source-backlog signal) with the
                # bounded pipeline's OWN input-queue depth instead --  a
                # fundamentally different, and in shadow mode misleading,
                # number (this pipeline is fed post-hoc, one submission
                # per already-ingested frame, so its queue depth reflects
                # "is the shadow worker keeping up," not real source
                # backlog). sync_source_backlog_from_pipeline() is real
                # and correct for a future integration where the pipeline
                # IS the source-backlog signal of record; it is not that
                # yet.
                _submit_frame_to_bounded_pipeline(_coordinator, new_frame)
                # Non-blocking drain so the worker's output queue doesn't
                # sit at latest-wins-eviction steady state uninspected;
                # the drained candidate itself is not consumed by anything
                # yet (shadow-only).
                _coordinator.bounded_pipeline.poll_prepared_candidate()
            if is_train:
                _coordinator.four_clocks.record_admitted_keyframe()
                _cadence.record_keyframe_admitted()   # #289 admitted clock
                _n_frames_trained += 1
                _register_supervision_source(new_cam)
                from eligsir.core.online_state import MapWorkItem
                dirty_region_queue.push(MapWorkItem(
                    item_type="refine_new_keyframe",
                    priority=1.0,
                    region_id=new_cam,
                    created_at_step=online_clock.session_step
                ))

            online_clock.frame_idx = n_frames_ingested
            online_clock.keyframe_idx = _n_frames_trained
            gaussians.current_frame_idx = n_frames_ingested

            _iter_since_new_frame = 0  # H7: reset freeze counter on new frame
            if _streaming_optimization_runtime is not None:
                _streaming_optimization_runtime.iter_since_new_frame = 0

            # ---- Phase 1.5-SLAM + Phase 7/8: tracking → decision --------------
            # Build a TrackingResult for every arriving frame so MappingScheduler
            # can apply confidence gates (Phase 8). For precomputed poses the
            # result is synthesized from the pre-assigned c2w; for live backends
            # we call initialize() on the first frame and track() on subsequent ones.
            if _deferred_tracked_result is not None:
                # W3-15: this frame's pose was already resolved by
                # DeferredSlamAdmissionTracker.poll() BEFORE ingest_next_frame
                # ran above (stamped onto new_frame.c2w there) -- do NOT call
                # _slam_backend.track() again here, that would be a second,
                # redundant synchronous call defeating the entire point of
                # deferring. Reuse the worker's own real TrackingResult
                # directly instead.
                from contracts.types import Pose as _CPosDef, TrackingResult as _CTRDef
                _tracking_result = _CTRDef(
                    pose=_CPosDef(np.asarray(_deferred_tracked_result.world_T_camera, dtype=np.float32)),
                    confidence=float(_deferred_tracked_result.tracking_confidence),
                    valid=True,
                    method="deferred_slam_worker",
                )
            elif is_dataset_pose_source(_pose_source_name):
                _frame_c2w = getattr(new_frame, "c2w", None)
                if _frame_c2w is not None:
                    from contracts.types import Pose as _CPos, TrackingResult as _CTR
                    # Real tracking-quality signal, when available: the same
                    # /slam/tracking_status_stamped join KeyframeAdmitter
                    # already consumes (eligsir/io/sources/ros2_live.py),
                    # here feeding MappingScheduler's confidence gate instead
                    # of admission quality_weight. Gated behind its own flag
                    # (default off) -- `dataset_pose` also covers plain
                    # baked-pose dataset replay with no tracking-status
                    # attribute at all, where this must stay confidence=1.0.
                    _ms_conf = 1.0
                    if bool(getattr(args, "mapping_scheduler_tracking_confidence_enabled", False)):
                        _ms_tstatus = getattr(new_frame, "_tracking_status", None)
                        if _ms_tstatus is not None:
                            _ms_conf = _ms_tracking_confidence_weight(
                                _ms_tstatus,
                                running_typical=_ms_tracking_running_stats.typical,
                                floor_weight=float(getattr(args, "mapping_scheduler_tracking_confidence_floor", 0.4)),
                            )
                            _ms_tracking_running_stats.update(_ms_tstatus.num_tracked)
                    _tracking_result = _CTR(
                        pose=_CPos(_frame_c2w.astype(np.float32)),
                        confidence=_ms_conf,
                        valid=True,
                        method="dataset_pose",
                    )
                else:
                    _tracking_result = None
            else:
                # Live backend path: build a canonical RGBDFrame and call backend.
                try:
                    from contracts.types import (
                        CameraIntrinsics as _CIntr2,
                        Pose as _CPos2,
                        RGBDFrame as _CRF2,
                        TrackingResult as _CTR2,
                    )
                    import numpy as _np_tr
                    _rgb_tr = getattr(new_frame, "_rgb_np", None)
                    _depth_tr = getattr(new_frame, "_depth_np", None)
                    _intr_tr = _CIntr2(
                        fx=new_frame.fx, fy=new_frame.fy,
                        cx=new_frame.cx, cy=new_frame.cy,
                        width=new_frame.width, height=new_frame.height,
                    )
                    _frame_c2w_tr = getattr(new_frame, "c2w", None)
                    _frame_pose_tr = _CPos2(_frame_c2w_tr.astype(_np_tr.float32)) if _frame_c2w_tr is not None else None
                    _canonical_frame = _CRF2(
                        index=new_frame.index,
                        timestamp=float(getattr(new_frame, "timestamp", n_frames_ingested)),
                        rgb=(_rgb_tr.astype(_np_tr.float32) / 255.0
                             if _rgb_tr is not None and _rgb_tr.dtype == _np_tr.uint8
                             else (_rgb_tr if _rgb_tr is not None
                                   else _np_tr.zeros((_intr_tr.height, _intr_tr.width, 3), dtype=_np_tr.float32))),
                        depth=(_depth_tr if _depth_tr is not None
                               else _np_tr.zeros((_intr_tr.height, _intr_tr.width), dtype=_np_tr.float32)),
                        intrinsics=_intr_tr,
                        pose=_frame_pose_tr,
                    )
                    with _timed("online_track"):
                        if not _slam_backend_initialized:
                            _tracking_result = _slam_backend.initialize(_canonical_frame)
                            _slam_backend_initialized = True
                        else:
                            _tracking_result = _slam_backend.track(_canonical_frame, T_pred=None)
                    # Propagate the received pose back onto the streaming frame so
                    # subsequent pipeline stages (viewer, trajectory logging) see it.
                    if _tracking_result is not None and _tracking_result.valid:
                        new_frame.c2w = _tracking_result.pose.matrix
                except NotImplementedError:
                    raise  # propagate — user selected an unimplemented backend
                except Exception as _tr_exc:
                    print(f"[streaming] SLAM backend track() failed: {_tr_exc}", flush=True)
                    _tracking_result = None

            # Phase 7: obtain mapping decision (confidence gates applied inside decide())
            with _timed("online_decide"):
                _decision = _mapping_scheduler.decide(
                    tracking_result=_tracking_result,
                    keyframe_indices=[],
                    replay_indices=[],
                )
            # Milestone 2: layer OnlineController on top of the scheduler when
            # enabled. The controller is the single source of truth for
            # do_mapping / steps / allow_insertion / allow_growth / freeze.
            if _online_controller is not None:
                try:
                    _ts_attr = getattr(new_frame, "timestamp", None)
                    _sensor_ts = float(_ts_attr) if _ts_attr is not None else float(n_frames_ingested)
                except (TypeError, ValueError):
                    _sensor_ts = float(n_frames_ingested)
                # Read admission stats straight off the frame: the module-level
                # `_admission_stats` is only assigned later (~3924), so referencing
                # it here raised UnboundLocalError whenever the controller was
                # enabled (it is skipped when the controller is off, so the bug was
                # dormant by default).
                _admit_stats_cur = getattr(new_frame, "_streaming_admission_stats", None)
                _is_keyframe = bool(_admit_stats_cur and _admit_stats_cur.get("admitted"))
                # backlog = current queue depth (frames released by the source
                # but not yet consumed by the drain loop). ``scheduler.frames_dropped``
                # is a cumulative wallclock_strict drop count, not a depth, so we
                # approximate the live backlog from the difference between
                # released and mapped frames.
                _backlog = max(
                    0,
                    int(getattr(scheduler, "frames_released", 0))
                    - int(getattr(scheduler, "frames_mapped", 0)),
                )
                with _timed("online_controller_tick"):
                    _online_decision, _base_decision = _online_controller.tick(
                        tracking_result=_tracking_result,
                        timestamp=_sensor_ts,
                        backlog=_backlog,
                        keyframe_indices=[],
                        replay_indices=[],
                        is_keyframe=_is_keyframe,
                    )
                # Overwrite the scheduler decision with the controller's view.
                _decision.do_mapping = _online_decision.do_mapping
                _decision.steps = _online_decision.steps
                _decision.allow_insertion = _online_decision.allow_insertion
                _decision.allow_growth = _online_decision.allow_growth
                _decision.freeze_persistent = _online_decision.freeze_persistent
                _decision.confidence = _online_decision.confidence
                # Only advance map_clock when the controller actually mapped
                # this frame. TRACK_ONLY / CATCH_UP with steps==0 must leave
                # map_age intact so the hysteresis state machine can recover.
                if _online_decision.do_mapping:
                    _online_controller.record_mapping(_sensor_ts)
                    scheduler.mark_mapped()
                    # D3: feed per-frame camera translation so the controller can
                    # spend more steps when the camera stalls, fewer when it races.
                    if getattr(_online_controller._config, "velocity_adaptive_steps", False):
                        _online_controller.record_camera_speed(streaming_scene.recent_camera_baseline(1))

            # Phase 8: log tracking confidence to TensorBoard.
            # Throttled: this block runs once per INGESTED FRAME (not once
            # per iteration), and each add_scalar call costs real
            # synchronous protobuf-serialization time — at high admission
            # rates that couples training it/s to frame arrival rate.
            _tb_frame_interval = max(1, int(getattr(args, "streaming_tb_frame_log_interval", 5)))
            _log_frame_tb = (n_frames_ingested % _tb_frame_interval) == 0
            if tb_writer and _tracking_result is not None and _log_frame_tb:
                with _timed("online_tb"):
                    for _tb_key, _tb_val in _tracking_result.to_tensorboard_dict().items():
                        tb_writer.add_scalar(_tb_key, _tb_val, iteration)

            # Phase 12: maybe start a new submap
            # Real anchor pose for the (potential) new submap and for
            # submap 0's one-shot bootstrap anchor -- root-caused 2026-08-06:
            # both SubmapManager's T_world_submap and the pose-graph's
            # odometry edges below previously used identity in place of this
            # real value, corrupting camera translations the first time a
            # loop-closure correction actually fired (see
            # SubmapManager.maybe_start_new_submap's anchor_c2w docstring).
            _new_cam_c2w = None
            if new_cam is not None and hasattr(new_cam, "R") and hasattr(new_cam, "T"):
                _R_c2w = np.asarray(new_cam.R, dtype=np.float64)
                _t_w2c = np.asarray(new_cam.T, dtype=np.float64)
                _new_cam_c2w = np.eye(4, dtype=np.float64)
                _new_cam_c2w[:3, :3] = _R_c2w
                _new_cam_c2w[:3, 3] = -_R_c2w @ _t_w2c
                _submap_manager.set_submap0_anchor(_new_cam_c2w)
            _prev_submap_id = _submap_manager.current_submap_id
            # RAW (never-corrected) anchor -- pose-graph odometry edges must
            # be built from real tracked-camera measurements only, never
            # from the current/corrected estimate (see
            # SubmapState.T_world_submap_raw's docstring: root-caused
            # 2026-08-06 -- mixing a corrected reference with a raw one here
            # silently corrupted every subsequent odometry edge, forcing
            # each pose-graph solve to keep re-adjusting the whole chain
            # even with zero loop edges).
            _prev_submap_anchor_c2w = _submap_manager.get_submap_raw_transform(_prev_submap_id)
            with _timed("online_submap"):
                _new_submap_started = _submap_manager.maybe_start_new_submap(
                    n_frames_ingested, gaussians=gaussians, anchor_c2w=_new_cam_c2w
                )
            # Stamp the camera just ingested above (new_cam) with the submap id
            # it belongs to. Without this, PoseCorrector._update_cameras_with_delta
            # (utils/pose_corrections.py) finds every camera's `_submap_id`
            # attribute missing (getattr default None) and skips ALL of them
            # unconditionally -- loop-closure correction was a complete no-op in
            # production regardless of how many loop edges were ever added or how
            # well-tuned coherence/rmse thresholds were. Found live 2026-07-25 on
            # floor3/trolley: loop edges fired (up to 4, coherence up to 0.91) yet
            # loop_closure/cameras_corrected and gaussians_corrected stayed exactly
            # 0 the entire run. new_cam belongs to whichever submap is CURRENT
            # after this frame's maybe_start_new_submap decision.
            if new_cam is not None:
                new_cam._submap_id = _submap_manager.current_submap_id
                # Visibility projection + observation-DB recording for this
                # frame already happened INSIDE ingest_next_frame via the
                # `_visibility_fn`/`_record_fn` hooks passed above (fix for
                # C1/C2 -- see scene/streaming_scene.py's ingest_next_frame).
                # Do not recompute/re-record here: it would double-project
                # and double-index this frame.
            # Phase 13: add odometry edge when a new submap begins
            if _loop_closure_enabled and _new_submap_started and _pose_graph is not None:
                try:
                    _cur_submap_id = _submap_manager.current_submap_id
                    # Raw anchor here too -- at this exact point _cur_submap_id
                    # was JUST created this frame so raw == current, but using
                    # the raw accessor explicitly keeps this call site correct
                    # even if that assumption ever changes.
                    _cur_anchor_c2w = _submap_manager.get_submap_raw_transform(_cur_submap_id)
                    # Seed the new node with its real world anchor (not
                    # identity) so the pose graph's own initial guess matches
                    # SubmapManager's belief -- see maybe_start_new_submap's
                    # anchor_c2w docstring for why an identity seed corrupted
                    # camera translations once a correction fired.
                    while _pose_graph.get_node_count() <= _cur_submap_id:
                        _pose_graph.add_node(_cur_anchor_c2w)
                    # Real relative transform between the two submaps' anchor
                    # poses (2026-08-06 root-cause fix) -- replaces the prior
                    # hardcoded identity ("no external odometry feed"), which
                    # told the solver every submap sits at the same pose by
                    # default. Both submaps' own tracked poses already ARE
                    # the real odometry signal; there is no need for a
                    # separate external feed.
                    # Edge-direction convention (settled 2026-08-06, session
                    # 2): PoseGraphEdge(src, dst, T_src_dst) takes
                    # T_src_dst = inv(world_T_src) @ world_T_dst (the
                    # body-relative, frame-invariant form every caller here
                    # naturally computes). PoseGraph.optimize() hands Open3D
                    # inv(T_src_dst) directly, with node.pose storing
                    # world_T_node uninverted -- no anchor-conjugation step;
                    # see loop_closure.pose_graph.PoseGraph.optimize's
                    # convention comment for the derivation.
                    _T_src_dst = np.linalg.inv(_prev_submap_anchor_c2w.astype(np.float64)) @ _cur_anchor_c2w.astype(np.float64)
                    # Information scale (Phase 2b root-cause fix, 2026-08-06):
                    # these edges come directly from real, already-tracked
                    # camera poses (high confidence), unlike loop-closure
                    # edges (Gaussian-correspondence matches, inherently
                    # noisier -- see PoseGraph.add_loop_edge's
                    # inliers/(rmse^2)-derived information, which for a real
                    # match with rmse~0.02m and ~50 inliers lands in the
                    # 1e5-1e6 range). The previous fixed value of 10 was
                    # arbitrary and low enough that Open3D's
                    # global_optimization visibly distorted a PURE
                    # odometry-only chain (confirmed via --loop_closure_debug:
                    # a node moved >2m between an optimize() call's pre/post
                    # snapshot with zero loop edges present in the graph at
                    # that call) -- the solver had no reason to trust these
                    # edges enough to hold the chain rigid. Also see
                    # PoseGraph.optimize()'s companion fix (persisting solved
                    # poses back into self._nodes) -- that bug alone caused
                    # visible drift regardless of this value; 1e4 keeps the
                    # tracked backbone comparably trusted to a genuine,
                    # well-matched loop edge without the numerical
                    # ill-conditioning risk of an even larger constant.
                    _odom_information = np.eye(6) * 1e4
                    _pose_graph.add_odometry_edge(
                        _prev_submap_id,
                        _cur_submap_id,
                        _T_src_dst,
                        information=_odom_information,
                    )
                    if bool(getattr(args, "loop_closure_debug", False)):
                        try:
                            import json as _json_lc2
                            with open(os.path.join(args.model_path, "loop_closure_debug.jsonl"), "a") as _lc_f2:
                                _lc_f2.write(_json_lc2.dumps({
                                    "iteration": iteration,
                                    "odometry_edge_added": {
                                        "src": _prev_submap_id,
                                        "dst": _cur_submap_id,
                                        "T_src_dst_translation_norm": float(np.linalg.norm(_T_src_dst[:3, 3])),
                                        "information_trace": float(np.trace(_odom_information)),
                                    },
                                }) + "\n")
                        except Exception as _lc_dbg_e2:
                            print(f"[loop_closure_debug] logging failed: {_lc_dbg_e2}", flush=True)
                except Exception:
                    pass  # pose-graph errors must never crash training
            # Log per-frame odometry diagnostics when available
            _odom_stats = getattr(new_frame, "_odom_stats", None)
            if _odom_stats is not None:
                from streaming.odometry.async_worker import log_odom_stats, OdometryStats
                if not isinstance(_odom_stats, OdometryStats):
                    # Convert raw dict emitted by OrbbecRosBag into OdometryStats
                    _odom_stats = OdometryStats(
                        translation_m=float(_odom_stats.get("translation_m", 0.0)),
                        rotation_deg=float(_odom_stats.get("rotation_deg", 0.0)),
                        fitness=_odom_stats.get("fitness"),
                        inlier_rmse=_odom_stats.get("inlier_rmse"),
                        info_trace=_odom_stats.get("info_trace"),
                        method=str(_odom_stats.get("method", "")),
                        valid=bool(_odom_stats.get("valid", True)),
                    )
                # Gate on the same per-frame TB throttle as the tracking-confidence
                # block above (streaming_tb_frame_log_interval): log_odom_stats()
                # writes 6 add_scalar calls (translation/rotation/fitness/rmse/
                # info_trace/valid) unconditionally per INGESTED frame, which is
                # the same synchronous protobuf-serialization cost that coupled
                # training it/s to frame arrival rate in the tracking-confidence
                # block this was missed alongside.
                if _log_frame_tb:
                    log_odom_stats(tb_writer, _odom_stats, iteration)
            _admission_stats = getattr(new_frame, "_streaming_admission_stats", None)
            if _admission_stats and _admission_stats.get("mode") == "hybrid_keyframe":
                _admit_cnt["total"] += 1
                _reason = _admission_stats.get("reason", "")
                if _admission_stats.get("admitted"):
                    if _reason == "db_diverse_revisit":
                        _admit_cnt["db_diverse_revisit"] += 1
                    elif _reason == "novelty_admit":
                        # #934: a novelty-forced admission carries forced=0 and
                        # moved=0, so without this branch it fell into "novel"
                        # -- the SAME bucket as "new_content", which is
                        # render-overlap novelty and a DIFFERENT quantity (see
                        # KeyframeDecision.reason). Same rationale as the
                        # db_diverse_revisit bucket above: an explicit branch is
                        # the only way telemetry can say whether this mechanism
                        # ever fired on real data.
                        _admit_cnt["novelty_admit"] += 1
                    elif _admission_stats.get("forced"):
                        _admit_cnt["forced"] += 1
                    elif _admission_stats.get("moved"):
                        _admit_cnt["moved"] += 1
                    else:
                        _admit_cnt["novel"] += 1
                else:
                    if _reason == "low_overlap":
                        _admit_cnt["rejected_overlap"] += 1
                    elif _reason == "rej_redundant_angle":
                        _admit_cnt["rejected_redundant_angle"] += 1
                    else:
                        _admit_cnt["rejected_motion"] += 1
            # Durable per-frame admission telemetry (2026-07-30): append the
            # same stats dict already computed above to a JSONL sidecar,
            # regardless of admission mode (not gated on hybrid_keyframe like
            # the aggregate _admit_cnt counters above) -- this is what makes a
            # future keyframe-selection-method comparison interpretable
            # per-frame instead of only via aggregate PSNR. Best-effort,
            # append-only, one small JSON line per ingested frame; mirrors the
            # existing local_capacity.jsonl append pattern elsewhere in this
            # file. Never allowed to affect training if it fails.
            if _admission_stats and bool(getattr(args, "streaming_admission_log_enabled", True)):
                try:
                    import json as _json_admit_log
                    _admit_log_row = {
                        "iter": int(iteration),
                        "frame": int(n_frames_ingested),
                        "mode": _admission_stats.get("mode"),
                        "admitted": bool(_admission_stats.get("admitted")),
                        "reason": _admission_stats.get("reason"),
                    }
                    for _k in ("overlap", "pose_delta_m", "pose_delta_deg", "gap",
                               # Issue #937: `score`/`cheap_score`/`forced`/`moved`/
                               # `novel` are computed and placed into
                               # `_streaming_admission_stats` (see the two dict
                               # builders in `_keyframe_admission_closure`) but were
                               # never whitelisted here, so they never reached this
                               # sidecar -- the same lost-data class #928 hit with
                               # `precedence`. This matters concretely: on the
                               # `db_diverse_revisit` / `rej_redundant_angle` exits
                               # `score` IS the DB overlap fraction the gate
                               # thresholds on, and it is the ONLY record of it.
                               # Without it, #937 (that branch compares an overlap
                               # fraction against `min_new_voxel_frac` instead of
                               # `min_overlap`) cannot be settled from artifacts:
                               # an audit of 100 historical admission logs found
                               # 12 runs with 238 `rej_redundant_angle` rows each
                               # and ZERO `db_diverse_revisit` rows anywhere, but
                               # could not tell whether overlap or the 20-deg angle
                               # bar was binding, because only `overlap` (which is
                               # `decision.render_overlap`, a DIFFERENT quantity,
                               # 0.0 on this exit) was persisted.
                               # WARNING for consumers: `score` is exit-dependent --
                               # it is the DB overlap fraction on those two reasons
                               # and a render/heuristic score elsewhere. ALWAYS
                               # filter by `reason` before aggregating it.
                               "score", "cheap_score", "forced",
                               # Issue #934: the novelty-DRIVEN trigger fired for
                               # this candidate. SEPARATE from "forced" on
                               # purpose -- "forced" drives force_share, itself a
                               # #934 Stage 1a pass/fail gate, and novelty force
                               # is the selector working rather than starvation
                               # relief. Read alongside `precedence.
                               # novelty_force_decisive` to tell "the trigger
                               # fired" from "the trigger owned the admission".
                               "novelty_forced",
                               "moved", "novel",
                               "quality_weight", "quality_blur_laplacian",
                               "quality_depth_valid_ratio", "tracking_weight",
                               # W3-07: only present when a non-shadow rollout
                               # mode is active (see _keyframe_admission_closure).
                               "rollout_mode", "rollout_reason", "rollout_overridden",
                               "rollout_utility",
                               # Issue #794/#928: the precedence reason chain
                               # (candidate_trigger/bypassed_gates/evaluated_gates/
                               # final_owner/hard_force plus #928's
                               # novelty_gate_outcome/novelty_gate_unavailable_reason/
                               # novelty_gate_fraction). `stats["precedence"]` has
                               # been set since #927 merged but was never added to
                               # this whitelist, so it never actually reached this
                               # sidecar -- fixed here so a `forced=True,
                               # reason="max_gap"` row can finally say WHICH of the
                               # two byte-identical-looking paths produced it
                               # without cross-referencing TensorBoard/report.json.
                               "precedence",
                               # Issue #934: fixed_stride-mode-only fields (the
                               # ordinal this decision was made at, and the
                               # configured stride N) -- absent from every
                               # other mode's stats dict, so harmless to
                               # whitelist unconditionally here.
                               "candidate_ordinal", "fixed_stride_n",
                               # Issue #946: soft/hard starvation backstop
                               # ownership, per decision. `soft_backstop_*` /
                               # `hard_backstop_*` are the two INDEPENDENTLY
                               # resolved thresholds (identical on every row of
                               # a run -- they are config, restated per row so
                               # the sidecar is self-describing); the
                               # `starvation_age_*` trio is what actually
                               # varies. `*_triggered` records the RAW
                               # threshold crossing, so comparing it against
                               # `precedence.candidate_trigger` reveals a
                               # #826 novelty-veto withdrawal that would
                               # otherwise be invisible. `motion_trigger`
                               # splits `reason="pose_delta"` into
                               # translation/rotation ownership, which no
                               # existing field could do. The five controller
                               # fields after it are #793/#795 state that was
                               # computed but never persisted.
                               "starvation_age_source_s",
                               "starvation_age_source_frames",
                               "starvation_age_basis",
                               "soft_backstop_mode",
                               "soft_backstop_domain",
                               "hard_backstop_domain",
                               "soft_backstop_threshold_s",
                               "soft_backstop_threshold_frames",
                               "hard_backstop_threshold_s",
                               "hard_backstop_threshold_frames",
                               "soft_backstop_preemptable",
                               "soft_backstop_triggered",
                               "hard_backstop_triggered",
                               "motion_trigger",
                               "translation_velocity_m_s",
                               "translation_velocity_ema_m_s",
                               "rotation_velocity_deg_s",
                               "rotation_velocity_ema_deg_s",
                               "source_dt_s",
                               "source_frame_delta",
                               "implied_input_fps",
                               "effective_min_translation_m",
                               "controller_target_interval_s",
                               "controller_target_interval_source_frames",
                               "timebase_source", "threshold_clamp"):
                        if _k in _admission_stats:
                            _admit_log_row[_k] = _admission_stats[_k]
                    with open(os.path.join(args.model_path, "keyframe_admission_log.jsonl"), "a") as _admit_log_f:
                        _admit_log_f.write(_json_admit_log.dumps(_admit_log_row) + "\n")
                except Exception:
                    pass  # telemetry errors must never crash training
            if tb_writer and _admission_stats and _log_frame_tb:
                with _timed("online_tb"):
                    tb_writer.add_scalar(
                        "streaming/keyframes_admitted",
                        int(getattr(streaming_scene, "_keyframes_admitted", 0)),
                        iteration,
                    )
                    tb_writer.add_scalar(
                        "streaming/keyframes_rejected",
                        int(getattr(streaming_scene, "_keyframes_rejected", 0)),
                        iteration,
                    )
                    for _k in ("overlap", "pose_delta_m", "pose_delta_deg", "gap", "admitted"):
                        if _k in _admission_stats:
                            tb_writer.add_scalar(f"streaming/keyframe/{_k}", float(_admission_stats[_k]), iteration)
            if new_cam is None:
                if iteration % 100 == 0 or (
                    _admission_stats and _admission_stats.get("reason") == "max_gap"
                ):
                    print(
                        f"[streaming] step={iteration} frame={n_frames_ingested} "
                        f"pose-only skip reason={_admission_stats.get('reason') if _admission_stats else 'unknown'}",
                        flush=True,
                    )
                is_train = False

            # Viewer: log RGB/depth, camera, keyframe, and update trajectory
            if new_frame is not None and _viewer_name != "none":
                from contracts.types import RGBDFrame as _CRGBDFrame, CameraIntrinsics as _CIntr, Pose as _CPose
                import numpy as _np_v
                try:
                    _c2w = getattr(new_frame, "c2w", None)
                    _cf_pose = None
                    if _c2w is not None:
                        _cf_pose = _CPose(_c2w.astype(_np_v.float32))
                        _trajectory_est.append(_cf_pose)
                    # B1 fix: explicit is None checks — `getattr(a) or getattr(b)`
                    # raises ValueError on numpy arrays.
                    _rgb = getattr(new_frame, "_rgb_np", None)
                    if _rgb is None:
                        _rgb = getattr(new_frame, "rgb", None)
                    _depth = getattr(new_frame, "_depth_np", None)
                    if _depth is None:
                        _depth = getattr(new_frame, "depth", None)
                    # Viser "Raw Input" panel fix: for path-only frame sources
                    # (e.g. TUM RGB-D, which never populate `_rgb_np`/`.rgb` --
                    # only `rgb_path`/`depth_path` on disk), neither getattr
                    # above ever succeeds, so `_viewer.log_rgb` was silently
                    # never called and the panel stayed empty for the entire
                    # run. Lazily decode via the shared decode-cache helpers
                    # (same code path camera prep already uses, so this is a
                    # cache hit whenever prefetch is on, and otherwise a single
                    # bounded PIL decode per ingested frame -- not per
                    # iteration). Errors here must never affect training; the
                    # outer try/except already covers that.
                    if _rgb is None and hasattr(new_frame, "rgb_path"):
                        from eligsir.io.sources.decode_cache import (
                            load_frame_rgb as _load_frame_rgb,
                            load_frame_depth_np as _load_frame_depth_np,
                        )
                        _pil_rgb = _load_frame_rgb(new_frame)
                        _rgb = _np_v.asarray(_pil_rgb)
                        if _depth is None:
                            _depth = _load_frame_depth_np(new_frame)
                    if _rgb is not None:
                        _intr = _CIntr(
                            fx=new_frame.fx, fy=new_frame.fy,
                            cx=new_frame.cx, cy=new_frame.cy,
                            width=new_frame.width, height=new_frame.height,
                        )
                        _cf = _CRGBDFrame(
                            index=new_frame.index,
                            timestamp=float(getattr(new_frame, "timestamp", n_frames_ingested)),
                            rgb=_rgb.astype(_np_v.float32) / 255.0 if _rgb.dtype == _np_v.uint8 else _rgb,
                            depth=_depth if _depth is not None else _np_v.zeros((_intr.height, _intr.width), dtype=_np_v.float32),
                            intrinsics=_intr,
                            pose=_cf_pose,
                        )
                        _viewer.log_rgb(_cf, iteration)
                        if _depth is not None:
                            _viewer.log_depth(_cf, iteration)
                        if _cf_pose is not None and hasattr(_viewer, "log_camera"):
                            _viewer.log_camera(_cf, _cf_pose, iteration)
                    if len(_trajectory_est) > 0:
                        _viewer.log_trajectory(_trajectory_est, None, iteration)
                except Exception:
                    pass  # viewer errors must never crash training

            # Viewer: log keyframe frustum on admission
            if new_cam is not None and _viewer_name != "none" and hasattr(_viewer, "log_keyframe"):
                try:
                    from contracts.types import CameraIntrinsics as _CIntr2
                    _kf_c2w = getattr(new_cam, "c2w", None)
                    _kf_fx = getattr(new_cam, "fx", None)
                    _kf_fy = getattr(new_cam, "fy", None)
                    _kf_cx = getattr(new_cam, "cx", None)
                    _kf_cy = getattr(new_cam, "cy", None)
                    _kf_w = getattr(new_cam, "image_width", None) or getattr(new_cam, "width", None)
                    _kf_h = getattr(new_cam, "image_height", None) or getattr(new_cam, "height", None)
                    if (
                        _kf_c2w is not None
                        and _kf_fx is not None and _kf_fy is not None
                        and _kf_cx is not None and _kf_cy is not None
                        and _kf_w is not None and _kf_h is not None
                    ):
                        from contracts.types import Pose as _CPose2
                        _kf_pose = _CPose2(_kf_c2w.astype(_np_v.float32))
                        _kf_intr = _CIntr2(
                            fx=float(_kf_fx), fy=float(_kf_fy),
                            cx=float(_kf_cx), cy=float(_kf_cy),
                            width=int(_kf_w), height=int(_kf_h),
                        )
                        _viewer.log_keyframe(_kf_pose, _kf_intr, iteration)
                except Exception:
                    pass  # viewer errors must never crash training

            # Phase 2: insert new Gaussians from depth (Phase 8: gate on _decision.allow_insertion)
            _p8_allow_insert = _decision.allow_insertion if _decision is not None else True
            # Pure-refinement horizon (2026-07-06b): past --streaming_noise_stop_horizon,
            # stop streaming depth-based insertion too, alongside MCMC noise/growth
            # (see eligsir/core/stop_policy.py's apply_online_schedule_overrides).
            # Frames still admit/track/pose normally past this point; they just stop
            # contributing new Gaussians, so no NEW floater candidates can appear while
            # pruning catches up on the frozen population.
            _insertion_stop_iter = _resolve_insertion_stop_iter(opt, args)
            if _insertion_stop_iter is not None and iteration > _insertion_stop_iter:
                _p8_allow_insert = False
            # Warmup: hold off insertion until enough train cameras have arrived.
            if _warmup_min_frames > 0 and len(streaming_scene.train_cameras) < _warmup_min_frames:
                _p8_allow_insert = False
            # Issue #600 insertion invariant, enforced rather than assumed.
            # `allow_geometry_insertion` is the ONE field that may permit new
            # source geometry, and ObservationDecision guarantees it can only be
            # True when the frame was admitted for supervision. A skipped frame
            # already returns is_train=False, so the counter below must stay 0;
            # if it ever ticks, the two decisions have silently decoupled and the
            # run's geometry is attributable to frames the mapper rejected.
            _obs_stats = getattr(new_cam, "_streaming_admission_stats", None) or (
                getattr(new_frame, "_streaming_admission_stats", None) or {}
            )
            _obs_allows_insert = _obs_stats.get("allow_geometry_insertion", True)
            if is_train and not _obs_allows_insert:
                _soft_all_invariant["inserted_from_skipped_frame"] += 1
                _p8_allow_insert = False
            if _phase_controller.in_refinement:
                _p8_allow_insert = False
            if is_train and _p8_allow_insert and args.streaming_insert_from_depth:
                cap = args.cap_max
                current_n = gaussians.get_xyz.shape[0]
                depth_respects_cap = getattr(args, "streaming_depth_respects_cap", False)
                if (not depth_respects_cap) or cap <= 0 or current_n < cap:
                    # Step 4: render new_cam to get alpha/depth mask
                    with torch.no_grad():
                        pkg_new = getattr(new_cam, "_streaming_admission_render_pkg", None)
                        if pkg_new is None:
                            pkg_new = render(new_cam, gaussians, pipe, background, render_depth=True)
                        alpha_new = pkg_new["alpha"]
                        depth_new = pkg_new.get("rendered_depth", None)

                    # Depth comparison export
                    if getattr(args, "streaming_export_depth_comparison", False) and depth_new is not None:
                        _sensor_d = _get_sensor_depth(new_cam, new_cam.image_height, new_cam.image_width)
                        if _sensor_d is not None:
                            _dc_dir = os.path.join(args.model_path, "depth_comparison")
                            _save_depth_comparison(
                                _dc_dir,
                                n_frames_ingested,
                                pkg_new["render"],
                                depth_new,
                                _sensor_d,
                            )

                    _t_insert_start = time.perf_counter()
                    # Issue #562: the one taxonomy category with no pre-existing
                    # `_timed()` region. Added through the SAME mechanism rather
                    # than a second one, and as a manual __enter__/__exit__ pair
                    # (the convention already used for `m_post_render` below) so
                    # this ~200-line span's indentation is untouched. A no-op
                    # unless SPLAT_PROFILE_PHASES or the duty-cycle timeline is
                    # on; the timeline additionally self-heals if this bracket
                    # ever leaks (DutyCycleTimeline.iteration_boundary).
                    _duty_insert_bracket = _timed("depth_insertion")
                    _duty_insert_bracket.__enter__()
                    # _batch_steps > 0 also enables batching -- otherwise a run that
                    # only sets --streaming_insertion_batch_steps (leaving the
                    # default streaming_insertion_batch_frames=1) would silently
                    # never batch at all, since _collect_only=_use_batch gates
                    # whether _coordinator.step() even returns a candidates dict
                    # instead of a plain int.
                    _use_batch = _batch_frames > 1 or _batch_steps > 0
                    # Issue #328 follow-up to #326 hook E: the NON-batch branch
                    # below (_use_batch False -- the default streaming
                    # configuration, see the issue #727 comment on that branch)
                    # has _coordinator.step() perform the real insertion commit
                    # SYNCHRONOUSLY, before `cands_or_n`/`added` are even known.
                    # Hook E only instruments the batched-flush branch, so on a
                    # default run every real Phase-2 insertion was invisible to
                    # the op-trace (confirmed empirically: a real fr3 op-trace run
                    # with total_inserted=46 recorded zero source_depth/grow
                    # events). Snapshot before-state here, unconditionally on
                    # both branches, so the non-batch branch below can emit the
                    # same event hook E already emits for the batched branch.
                    _ot_ins_topo_before = int(map_backend.topology_version)
                    _ot_ins_n_before = int(gaussians.get_xyz.shape[0])
                    _cycle_result = _coordinator.step(
                        gaussians, new_frame, args,
                        streaming_scene=streaming_scene,
                        prev_frame=_prev_insert_frame,
                        prev_depth_meters=_prev_insert_depth_m,
                        render_alpha=alpha_new,
                        render_depth=depth_new,
                        current_frame_idx=n_frames_ingested,
                        iteration=iteration,
                        _collect_only=_use_batch,
                        map_backend=map_backend,
                    )
                    cands_or_n = _cycle_result.raw
                    _insert_stats = _cycle_result.stats
                    if tb_writer:
                        tb_writer.add_scalar("mapping/cycle/n_inserted", _cycle_result.n_inserted, iteration)
                        # Four-clocks telemetry (Wave 2 Task 8) -- same cadence as
                        # n_inserted above (every insertion cycle, not every
                        # iteration): source/admitted/commit rates, map-freshness
                        # age, backlog, and realtime factor. See
                        # FourClocksSummary's docstring (coordinator.py) for the
                        # precise definition of each field.
                        _clocks = _coordinator.four_clocks.summary()
                        tb_writer.add_scalar("mapping/clocks/source_frame_rate_hz", _clocks.source_frame_rate_hz, iteration)
                        tb_writer.add_scalar("mapping/clocks/admitted_keyframe_rate_hz", _clocks.admitted_keyframe_rate_hz, iteration)
                        tb_writer.add_scalar("mapping/clocks/map_commit_rate_hz", _clocks.map_commit_rate_hz, iteration)
                        if _clocks.map_age_ms is not None:
                            tb_writer.add_scalar("mapping/clocks/map_age_ms", _clocks.map_age_ms, iteration)
                        tb_writer.add_scalar("mapping/clocks/input_backlog_frames", _clocks.input_backlog_frames, iteration)
                        if _clocks.realtime_factor is not None:
                            tb_writer.add_scalar("mapping/clocks/realtime_factor", _clocks.realtime_factor, iteration)
                        # F2/F3 fixes (2026-08-06): the new, non-conflated
                        # backlog breakdown and the two broader map-freshness
                        # clocks (see FourClocksSummary's docstring).
                        if _clocks.last_topology_change_age_ms is not None:
                            tb_writer.add_scalar(
                                "mapping/clocks/last_topology_change_age_ms",
                                _clocks.last_topology_change_age_ms, iteration)
                        if _clocks.last_parameter_update_age_ms is not None:
                            tb_writer.add_scalar(
                                "mapping/clocks/last_parameter_update_age_ms",
                                _clocks.last_parameter_update_age_ms, iteration)
                        if _clocks.source_queue_depth is not None:
                            tb_writer.add_scalar(
                                "mapping/clocks/source_queue_depth",
                                _clocks.source_queue_depth, iteration)
                        tb_writer.add_scalar(
                            "mapping/clocks/frames_rejected_total", _clocks.frames_rejected_total, iteration)
                        tb_writer.add_scalar(
                            "mapping/clocks/holdout_frames_total", _clocks.holdout_frames_total, iteration)
                        tb_writer.add_scalar(
                            "mapping/clocks/frames_unprocessed", _clocks.frames_unprocessed, iteration)
                    if _use_batch:
                        # Accumulate; flush when batch is full
                        # PR #1142: the GPU-resident producer returns a
                        # SurfaceCandidatesT, not a dict. A bare isinstance-dict
                        # guard here would silently DROP every device-native
                        # candidate set on the batched path (no error, no
                        # telemetry -- just zero insertions), so accept both
                        # shapes and read the point count uniformly.
                        _cand_pts = _collected_candidate_points(cands_or_n)
                        if _cand_pts is not None and _cand_pts.shape[0] > 0:
                            _insertion_batch.append(cands_or_n)
                        _insertion_batch_frames += 1
                        _pending_pts = sum(
                            _collected_candidate_points(c).shape[0] for c in _insertion_batch
                        )
                        _should_flush, _flush_trigger_reason = _should_flush_insertion_batch(
                            iteration=iteration,
                            last_flush_iteration=_last_flush_iteration,
                            batch_steps=_batch_steps,
                            frames_since_flush=_insertion_batch_frames,
                            batch_frames=_batch_frames,
                            pending_pts=_pending_pts,
                            batch_max_pts=_batch_max_pts,
                            disable_point_cap=_disable_point_cap,
                        )
                        if _should_flush:
                            _w3_02_n_before = gaussians.get_xyz.shape[0]
                            _w3_02_topo_before = map_backend.topology_version
                            _cohort_debug_out = [] if _cohort_trace.enabled else None
                            added = _flush_insertion_batch(
                                _insertion_batch, gaussians, streaming_scene, iteration, args,
                                map_backend=map_backend,
                                cohort_debug_out=_cohort_debug_out)
                            # PR2 cadence clocks (feat/realityscan-streaming): a real
                            # depth-insertion commit.
                            _clk_depth_mutation_event_id += 1
                            _clk_mutation_event_id += 1
                            # #289 insertion-commit clock, carrying the real
                            # inserted-Gaussian count for the per-admitted-frame
                            # and per-optimizer-step insertion rates.
                            _cadence.record_insertion_commit(added)
                            # Issue #326 hook E -- source-depth growth, aggregate scope.
                            # topology_version_before/after is what makes
                            # `population_change_without_topology_advance` derivable
                            # (runtime check of #543's secondary finding #1).
                            if _optrace is not None:
                                _optrace.current_frame = int(n_frames_ingested)
                                _optrace.record(
                                    iteration=iteration, op="grow", mechanism="source_depth",
                                    phase="applied", scope="aggregate",
                                    topology_version=int(map_backend.topology_version),
                                    n_rows=int(added),
                                    evidence={
                                        "topology_version_before": int(_w3_02_topo_before),
                                        "topology_version_after": int(map_backend.topology_version),
                                        "n_before": int(_w3_02_n_before),
                                        "n_after": int(gaussians.get_xyz.shape[0]),
                                        "trigger": str(_flush_trigger_reason),
                                    },
                                )
                            if _w3_02_mutation_trace.enabled:
                                _w3_02_mutation_trace.wrap(
                                    iteration=iteration, op="depth_insertion_flush",
                                    topology_version_before=_w3_02_topo_before,
                                    topology_version_after=map_backend.topology_version,
                                    n_before=_w3_02_n_before, n_after=gaussians.get_xyz.shape[0],
                                    content=int(added),
                                )
                            if _cohort_trace.enabled and _cohort_debug_out:
                                _cd = _cohort_debug_out[0]
                                _cd_frame_idx = _cd["frame_index"]
                                _cd_xyz = _cd["xyz"]
                                # add_points_as_gaussians always inserts every point it's given
                                # (verified: scene/gsplat_model.py returns N unconditionally for
                                # N>0) -- assert rather than silently min()-ing over a mismatch
                                # that would otherwise mis-attribute proposal rows to the wrong
                                # points.
                                assert int(added) == _cd_xyz.shape[0], (
                                    f"cohort telemetry invariant violated: added={added} != "
                                    f"post-dedup/subsample xyz count={_cd_xyz.shape[0]}"
                                )
                                _n_committed = _cd_xyz.shape[0]
                                _batch_id = f"batch_i{iteration}"
                                _proposal_ids = []
                                for _pi in range(_n_committed):
                                    _pid = f"{_batch_id}_p{_pi}"
                                    _proposal_ids.append(_pid)
                                    _cohort_trace.record_proposal(
                                        proposal_id=_pid,
                                        source_frame_index=int(_cd_frame_idx[_pi]),
                                        world_xyz=(
                                            float(_cd_xyz[_pi, 0]),
                                            float(_cd_xyz[_pi, 1]),
                                            float(_cd_xyz[_pi, 2]),
                                        ),
                                        birth_iteration=iteration,
                                        birth_topology_version=_w3_02_topo_before,
                                        # NOTE: StreamingScene._xyz_to_keys returns a single packed
                                        # int64 key, not a decomposed (i, j, k) triple -- no existing
                                        # helper decodes it back out. Left as (0, 0, 0) rather than
                                        # fabricating a wrong triple; world_xyz above already carries
                                        # the real position, which is what this experiment's cadence
                                        # question actually needs. Revisit if voxel-cell-level analysis
                                        # becomes load-bearing for a future phase.
                                        voxel_key=(0, 0, 0),
                                    )
                                _cohort_trace.record_cohort_birth(
                                    cohort_id=_batch_id,
                                    birth_batch_id=_batch_id,
                                    birth_iteration=iteration,
                                    member_proposal_ids=_proposal_ids,
                                    maturity_score_history=[],
                                )
                                _cohort_trace.record_mutation_batch(
                                    batch_id=_batch_id,
                                    commit_iteration=iteration,
                                    topology_version_before=_w3_02_topo_before,
                                    topology_version_after=map_backend.topology_version,
                                    n_proposals_committed=_n_committed,
                                    n_proposals_deduped=int(_cd["n_deduped"]),
                                    trigger_reason=_flush_trigger_reason,
                                )
                            # Wave 2 Task 8 four-clocks fix (final-review finding 1): the
                            # batched-insertion path's real commit happens here, not inside
                            # _coordinator.step() (which only sees the collected candidates
                            # dict, not an int, under _collect_only=True -- see
                            # MappingCycleResult.from_step_result). Route the real count back
                            # into the aggregator so map_commit_rate_hz/map_age_ms reflect
                            # batched commits instead of reading permanently 0/None.
                            _coordinator.four_clocks.record_map_commit(n_inserted=added)
                            if added > 0:
                                _last_map_commit_iter = iteration  # issue #1044 growth-staleness brake
                            _insertion_batch = []
                            _insertion_batch_frames = 0
                            _last_flush_iteration = iteration
                        else:
                            added = 0
                    else:
                        added = cands_or_n
                        # Issue #727 fix: the #289 insertion-commit clock was
                        # only ever recorded on the BATCHED insertion path
                        # (_use_batch True, i.e. --streaming_insertion_batch_
                        # frames>1 or --streaming_insertion_batch_steps>0).
                        # This non-batch branch is the path a default streaming
                        # run actually takes (both flags default to their
                        # off/1 values) -- _coordinator.step() above already
                        # performed the real, unbatched depth-insertion commit
                        # for this frame, but nothing recorded it, so
                        # insertion_commit/inserted_gaussians read structurally
                        # zero on every default run regardless of how many real
                        # insertions happened. One call to _coordinator.step()
                        # here is the direct non-batch analogue of one flush in
                        # the batched branch above (which calls
                        # record_insertion_commit exactly once per flush) --
                        # same "one commit event" semantics, so no double-
                        # counting relative to the batch path.
                        _cadence.record_insertion_commit(added)
                        # Issue #1044 fix: the #727 fix above wired `_cadence`
                        # (RealizedCadenceTracker) for this non-batch path but left
                        # `_coordinator.four_clocks` -- a SEPARATE aggregator --
                        # unwired here. four_clocks.record_map_commit() is the ONLY
                        # thing that feeds map_commit_rate_hz/map_age_ms (the
                        # "how stale is the map" clock read by the Viser live-stats
                        # panel, report.json, and any growth-staleness brake). Since
                        # --streaming_insertion_batch_frames/_steps default off, this
                        # non-batch branch is the path every default streaming run
                        # takes -- so map_commit_rate_hz/map_age_ms read structurally
                        # 0.00Hz/ever-growing-stale on every default run regardless of
                        # how much real depth-insertion activity happened, exactly
                        # matching the "map hadn't received a fresh observation in 14
                        # seconds" telemetry in issue #1044 even while insertion was
                        # (likely) proceeding normally. One call per real commit here
                        # mirrors the batch path's single call per flush -- same "one
                        # commit event" semantics, no double-counting.
                        _coordinator.four_clocks.record_map_commit(n_inserted=added)
                        if added > 0:
                            _last_map_commit_iter = iteration  # issue #1044 growth-staleness brake
                        # H9: record insertion iteration for anchor-loss decay (non-batch path)
                        if added > 0 and hasattr(gaussians, "anchor_iter"):
                            gaussians.anchor_iter[-added:] = iteration
                        # Issue #328 follow-up to #326 hook E -- the non-batch
                        # analogue of hook E, see the comment above
                        # `_ot_ins_topo_before`/`_ot_ins_n_before`. Same event
                        # shape (op="grow", mechanism="source_depth",
                        # scope="aggregate") so gaussian_op_conflicts.py's
                        # per-mechanism derivations see source-depth insertion
                        # on a default (non-batched) run for the first time.
                        if _optrace is not None:
                            _optrace.current_frame = int(n_frames_ingested)
                            _optrace.record(
                                iteration=iteration, op="grow", mechanism="source_depth",
                                phase="applied", scope="aggregate",
                                topology_version=int(map_backend.topology_version),
                                n_rows=int(added),
                                evidence={
                                    "topology_version_before": _ot_ins_topo_before,
                                    "topology_version_after": int(map_backend.topology_version),
                                    "n_before": _ot_ins_n_before,
                                    "n_after": int(gaussians.get_xyz.shape[0]),
                                    "trigger": "non_batch_insertion",
                                },
                            )
                    total_inserted += added
                    # Adaptive optimizer-exposure-per-insertion (issue #109 E3,
                    # arguments/__init__.py's streaming_adaptive_steps_per_insertion
                    # docstring has the full design). Sizes the NEXT frame's
                    # release interval from THIS frame's real insertion count —
                    # a genuinely per-frame-responsive cadence, distinct from
                    # E1/E2's fixed steps_per_frame multiplier.
                    if bool(getattr(args, "streaming_adaptive_steps_per_insertion", False)):
                        _adap_min = max(1, int(getattr(args, "streaming_adaptive_steps_min", 20)))
                        _adap_max = max(_adap_min, int(getattr(args, "streaming_adaptive_steps_max", 400)))
                        _adap_gain = float(getattr(args, "streaming_adaptive_steps_gain", 1.0))
                        _adap_next_steps = int(min(max(_adap_min, _adap_gain * added), _adap_max))
                        scheduler.set_next_interval(iteration, _adap_next_steps)
                        if tb_writer:
                            tb_writer.add_scalar(
                                "streaming/adaptive_steps_per_insertion", _adap_next_steps, iteration)
                    if _coordinator.bounded_pipeline is not None:
                        # Shadow-mirror this real, already-applied commit
                        # into PendingCommitQueue (see
                        # _mirror_commit_to_pending_queue's docstring for
                        # why this stays observation-only in this PR).
                        _mirror_commit_to_pending_queue(_coordinator, new_frame)
                        if tb_writer:
                            # Deliverable-#5-adjacent observability: the
                            # commit queue is otherwise invisible in a real
                            # run (queue_depths() is only read once, at
                            # shutdown, in the final summary print below).
                            # Same "mapping/clocks/*" convention as the
                            # existing input_backlog_frames/
                            # source_queue_depth scalars (see
                            # FourClocksSummary in coordinator.py).
                            _commit_metrics = _coordinator.bounded_pipeline.commit_queue.metrics()
                            tb_writer.add_scalar(
                                "mapping/clocks/commit_queue_depth",
                                _commit_metrics.depth, iteration)
                            tb_writer.add_scalar(
                                "mapping/clocks/commit_queue_drained_total",
                                _commit_metrics.dequeued_count, iteration)
                            tb_writer.add_scalar(
                                "mapping/clocks/commit_queue_dropped_total",
                                _commit_metrics.dropped_count, iteration)
                    _duty_insert_bracket.__exit__(None, None, None)
                    phase_timer.record_insert(time.perf_counter() - _t_insert_start)
                    _memprobe("after_insert", iteration, f"added={added} N={gaussians.get_xyz.shape[0]}")
                    if added > 0 and (iteration % 100 == 0 or added > 1000):
                        print(
                            f"[streaming] step={iteration} frame={n_frames_ingested} "
                            f"inserted={added} total_inserted={total_inserted} N={gaussians.get_xyz.shape[0]}",
                            flush=True,
                        )
                    # Log per-filter insertion telemetry
                    if tb_writer and _insert_stats:
                        for _k, _v in _insert_stats.items():
                            tb_writer.add_scalar(f"streaming/insertion/{_k}", _v, iteration)
                elif iteration % 500 == 0:
                    print(
                        f"[streaming] step={iteration} frame={n_frames_ingested} "
                        f"skipped depth insertion because N={current_n} >= cap_max={cap} "
                        "and streaming_depth_respects_cap=True",
                        flush=True,
                    )
                # Cache this frame as the previous frame for next insertion
                _prev_insert_frame = new_frame
                _prev_insert_depth_m = _load_depth_meters(new_frame)

            # Drop the one-shot admission render cache regardless of whether
            # depth insertion ran (both branches above are its only consumer,
            # and only for the frame that was JUST ingested). Without this,
            # streaming_scene.py's `cam._streaming_admission_render_pkg =
            # admission_render_pkg` (set unconditionally on every admitted
            # keyframe, scene/streaming_scene.py ingest_next_frame) permanently
            # attaches a full-resolution render/alpha/depth GPU tensor bundle
            # to the Camera object, which then lives forever in
            # train_cameras/_replay_buffer/_global_reservoir — an unbounded
            # per-admitted-frame GPU leak, independent of --cap_max,
            # --streaming_resolution, and --streaming_keyframe_window (none of
            # those levers touch this attribute). Measured on ScanNet++
            # b20a261fdf/r9700: ~70-130MB/admitted-frame growth with N and
            # cap_max both FLAT (cap already reached), isolating this as the
            # leak (2026-07-07 memory-mystery investigation).
            if hasattr(new_cam, "_streaming_admission_render_pkg"):
                new_cam._streaming_admission_render_pkg = None

        # Task 3: post-admission consolidation burst — when a keyframe was just
        # admitted, queue nearest-neighbour cameras so the next N steps give the
        # newborn Gaussians immediate multi-view supervision before returning to
        # normal sampling. No-op when the flag is off (queue stays empty).
        if _consolidation_burst_enabled and is_train and new_cam is not None:
            _admitted_now = bool(_admission_stats and _admission_stats.get("admitted"))
            if _admitted_now:
                for _bc in streaming_scene.get_nearest_cameras(new_cam, k=_consolidation_burst_steps):
                    _consolidation_burst_queue.append(_bc)

        # Variant C: flush the single deferred cameras_extent readback for the
        # batch just drained (no-op when batch ingest is off or nothing pending).
        if _batch_ingest:
            streaming_scene.flush_deferred_extent_update()

        # Component 1 (spatial active bubble): stash the bubble CENTER so the
        # downstream freeze gates (relocation / noise / snapshot-restore) can recompute
        # the in-bubble mask from live positions. Anchor to the live robot pose (latest
        # keyframe — moves smoothly, unlike the randomly-sampled training view), but
        # OFFSET the center FORWARD onto the observed geometry: the camera center sits
        # in empty space ~d_obs in front of the surfaces it maps, so a sphere on the
        # raw camera center freezes exactly what should train. forward = R[:,2] (world
        # optical axis, same convention as spatial_window_weights). d_obs = median
        # forward-distance of geometry ahead (auto), or a fixed offset if configured.
        # The center must exist for EITHER the freeze (bubble) OR local-stats mask.
        from eligsir.utils.strategies.mcmc_strategy import _bubble_feature_enabled as _bfe
        if _bfe(args) and streaming_scene.train_cameras:
            try:
                _rc = streaming_scene.train_cameras[-1]
                _cc = _rc.camera_center.detach().reshape(3)
                _fwd = torch.as_tensor(
                    np.asarray(_rc.R, dtype=np.float32)[:, 2], device=_cc.device
                )
                _fwd = _fwd / torch.clamp(torch.linalg.norm(_fwd), min=1e-8)
                _off = float(getattr(args, "streaming_bubble_forward_offset", 0.0) or 0.0)
                _radius = float(getattr(args, "streaming_bubble_radius", 5.0) or 5.0)
                if _off > 0.0:
                    _d_obs = _off
                else:
                    # Prefer the MOST RECENTLY BORN Gaussians (this frame's actual sensor
                    # observation) over a stale median-of-the-whole-cloud: the existing
                    # point cloud lags the live camera (esp. after a turn), which pinned
                    # d_obs at either a stray near-field point (~0.45m) or the corridor's
                    # bulk median (~18m) and produced a multi-thousand-iter active_frac=0
                    # dead zone as the robot advanced away from that stale center.
                    # Search window for d_obs candidates.
                    _near_range = 2.0 * _radius
                    _min_range = max(0.3, 0.25 * _radius)
                    # Hard cap: camera must stay inside the core bubble (d_obs < radius).
                    # 0.8*radius → camera is 0.8R from center, 0.2R inside the core edge.
                    _max_d_obs = 0.8 * _radius
                    _prev = float(getattr(args, "_bubble_d_obs", 0.0) or 0.0)
                    _d_obs = None
                    with torch.no_grad():
                        _birth = getattr(gaussians, "birth_frame", None)
                        _xyz = gaussians.get_xyz.detach()
                        if _birth is not None and _birth.shape[0] == _xyz.shape[0] and _birth.numel() > 0:
                            _max_b = int(_birth.max().item())
                            _cohort = _birth >= (_max_b - 2)  # this + last couple admitted frames
                            if bool(_cohort.any()):
                                _rel_c = _xyz[_cohort] - _cc.reshape(1, 3)
                                _proj_c = (_rel_c @ _fwd.reshape(3, 1))[:, 0]
                                _fwd_pts = _proj_c[(_proj_c > _min_range) & (_proj_c < _near_range)]
                                if _fwd_pts.numel() > 5:
                                    _d_obs = float(_fwd_pts.median())
                        if _d_obs is None:
                            # Fallback: nearby-geometry median (older cohort didn't qualify).
                            _rel = _xyz - _cc.reshape(1, 3)
                            _proj = _rel @ _fwd.reshape(3, 1)
                            _lateral = torch.linalg.norm(_rel - _proj * _fwd.reshape(1, 3), dim=1)
                            _ahead = _proj[
                                (_proj[:, 0] > _min_range)
                                & (_proj[:, 0] < _near_range)
                                & (_lateral < _near_range)
                            ]
                            _d_obs = (
                                float(_ahead.median())
                                if _ahead.numel() > 10
                                else (_prev if _prev > 0.0 else _radius)
                            )
                        _d_obs = max(_min_range, min(_d_obs, _max_d_obs))
                args._bubble_center = (_cc + _fwd * _d_obs).detach()
                args._bubble_d_obs = _d_obs
            except Exception:
                args._bubble_center = None

        # Phase 8: skip map update when confidence is below provisional threshold
        if _decision is not None and not _decision.do_mapping:
            continue

        # ---- Learning rate update -----------------------------------------
        if _streaming_optimization_runtime is None:
            xyz_lr = gaussians.update_learning_rate(iteration)

        # SH degree schedule
        if iteration in sh_degree_schedule:
            gaussians.oneupSHdegree()

        # ---- Camera selection ---------------------------------------------
        viewpoint_cam = None
        # Once a finite source is exhausted, the explicit mapping endpoint
        # must drain its finite consolidation backlog before entering the
        # refinement tail.  Dirty-region work is intentionally preserved for
        # that tail (and exact-resume checkpoints now own it); letting it keep
        # priority here can starve consolidation until the mapping budget is
        # exhausted.
        _draining_mapping_endpoint = (
            _post_mapping_refinement_steps > 0
            and streaming_scene._source_idx >= streaming_scene._frame_count
            and bool(_consolidation_burst_queue)
        )
        if _draining_mapping_endpoint:
            viewpoint_cam = _consolidation_burst_queue.popleft()
        elif len(dirty_region_queue) > 0:
            work_item = dirty_region_queue.pop()
            viewpoint_cam = work_item.region_id

            # Phase 4: Simple priority decay mechanism for submap refinement
            # The region stays in the queue until its priority drops below 0.1
            if work_item.priority > 0.1:
                work_item.priority *= 0.9  # Decay priority with each sampling
                dirty_region_queue.push(work_item)

        if viewpoint_cam is None:
            # Task 3: drain the consolidation-burst queue before normal sampling
            # (no-op when the flag is off — queue is always empty).
            if _consolidation_burst_queue:
                viewpoint_cam = _consolidation_burst_queue.popleft()
            else:
                viewpoint_cam = streaming_scene.sample_training_camera(
                    gaussians, render_fn=render, pipe=pipe, background=background
                )
        if viewpoint_cam is None:
            continue
        # PR2 cadence clocks: one supervised/rendered view. PR3 keeps this
        # increment here (it counts THIS iteration's primary view) and adds one
        # increment per EXTRA accumulated view inside the PR3 block below, so
        # the counter still advances exactly once per rendered/supervised view
        # -- i.e. by streaming_optimization_batch_size per iteration.
        _clk_supervision_view_count += 1
        _cadence.record_supervision_view()   # #289 supervision clock

        # Page the sampled camera's image back onto the GPU (no-op if it's already
        # resident, e.g. still inside the recency window). Counterpart to the
        # offload_camera_image() eviction in StreamingScene.ingest_next_frame() --
        # see that call site's comment for why this exists.
        if bool(getattr(args, "streaming_offload_camera_images", True)):
            from scene.cameras import prepare_camera_for_render
            prepare_camera_for_render(viewpoint_cam, device="cuda")
            # Issue #396: register the page-in so it is released at the top of
            # the next iteration unless this camera is HOT.
            _residency_ledger.note_paged_in(viewpoint_cam)

        # Center policy: once ingestion has been idle for
        # streaming_consolidation_idle_steps, the "last admitted keyframe" is a
        # stale anchor pinned to the trajectory end — post-ingest replay would
        # keep the bubble there forever. Recompute the center from the CURRENTLY
        # SAMPLED training camera instead, reusing the last known d_obs.
        if (
            _bfe(args)
            and getattr(args, "_bubble_center", None) is not None
            and getattr(streaming_scene, "_steps_since_new_frame", 0)
            > int(getattr(args, "streaming_consolidation_idle_steps", 500))
            and hasattr(viewpoint_cam, "camera_center")
        ):
            try:
                _ic_cc = viewpoint_cam.camera_center.detach().reshape(3)
                _ic_fwd = torch.as_tensor(
                    np.asarray(viewpoint_cam.R, dtype=np.float32)[:, 2], device=_ic_cc.device
                )
                _ic_fwd = _ic_fwd / torch.clamp(torch.linalg.norm(_ic_fwd), min=1e-8)
                _ic_radius = float(getattr(args, "streaming_bubble_radius", 5.0) or 5.0)
                _ic_d = float(getattr(args, "_bubble_d_obs", 0.0) or 0.0)
                if _ic_d <= 0.0:
                    _ic_d = 0.8 * _ic_radius
                _ic_d = min(_ic_d, 0.8 * _ic_radius)
                args._bubble_center = (_ic_cc + _ic_fwd * _ic_d).detach()
            except Exception:
                pass

        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3,), device="cuda") if opt.random_background else background

        # ---- Forward pass -------------------------------------------------
        update_sh_rest = (
            gaussians.active_sh_degree == 0
            or sh_update_interval <= 1
            or iteration % sh_update_interval == 0
        )
        # Safety net + producer check #3 (relocation/growth, which run after the
        # two per-step checks below): neutralize any non-finite / exp-overflowing
        # parameter BEFORE rendering so the render and loss are always finite.
        # W3-61: branch-free numeric core runs every call (unconditionally
        # neutralizes non-finite/exp-overflowing state, same as production
        # always did); the diagnostic count/print is sampled at
        # scalar_log_interval (with an every-iteration burn-in window) -- see
        # docs/architecture/2026-08-10-w3-61-sanitize-production-wiring.md.
        _sanitize_with_seed_freeze(
            "loop_top", iteration, diag_interval=scalar_log_interval
        )
        # Multi-resolution frame cache (multires-cache plan, Task 4): decide
        # the active tier from admission throughput, then rebind the
        # currently-sampled training camera's rgb/depth to that tier's
        # cached tensors before rendering.
        #
        # BUG FIX (found via real-run validation, Task 5): `mapping_done`
        # cannot reuse the exact hard-stop expression
        # `iteration - _last_ingest_iter >= _refinement_iters` verbatim --
        # the main loop's own termination check (same expression, above)
        # `break`s the instant that becomes true, before this line ever
        # runs. So the literal-reuse version was mathematically
        # unreachable as True in any executed iteration: the resolution
        # controller's one-way lock to high-res never engaged in
        # production despite passing all unit tests (which exercised
        # `decide()` directly, not the main-loop wiring). Use a distinct,
        # earlier fraction of the same idle window instead, so a real
        # high-res refinement span actually executes before the hard
        # stop -- half of `_refinement_iters` of idle time is "mapping is
        # clearly over" without flapping on ordinary inter-frame gaps
        # (which are a handful of iterations, not hundreds).
        _mapping_done = _phase_controller.in_refinement or (
            _use_rolling_idle_stop
            and (iteration - _last_ingest_iter) >= max(1, _refinement_iters // 2)
        )
        _active_tier = _resolution_controller.decide(
            mapping_done=_mapping_done, iteration=iteration
        )
        if tb_writer is not None:
            tb_writer.add_scalar("streaming_multires/active_tier_is_high", 1.0 if _active_tier == "high" else 0.0, iteration)
        _streaming_frame_obj = getattr(viewpoint_cam, "_streaming_frame", None)
        if _streaming_frame_obj is not None:
            _frame_record = _observation_db.get_frame(_streaming_frame_obj.index)
            if _frame_record is None and not getattr(bind_resolution_tier, "_warned_lookup", False):
                print(f"[multires-debug] lookup miss: viewpoint_cam._streaming_frame.index="
                      f"{_streaming_frame_obj.index!r} db_len={len(_observation_db)} "
                      f"db_known_frame_ids_sample={list(_observation_db._frames.keys())[:10]}", flush=True)
                bind_resolution_tier._warned_lookup = True
            _bind_result = bind_resolution_tier_ex(
                viewpoint_cam, _frame_record, _active_tier, db=_observation_db
            )
            _bind_ok = bool(_bind_result.ok)
            _multires_bind_stats[f"{_active_tier}_{'ok' if _bind_ok else 'fail'}"] += 1
            # issue #333: an UNRESOLVED bind means this camera is not at the
            # requested tier. It is self-consistent (nothing was mutated), so
            # training proceeds -- but it proceeds at `tier_after`, and that
            # must be visible rather than silent. Rate-limited by outcome
            # class so a long run cannot spam O(frames) lines.
            if not _bind_ok:
                _mm_n = _MULTIRES_MISMATCH_WARNED.get(_bind_result.outcome, 0)
                if _mm_n < 3:
                    print(
                        f"[multires] WARNING iter={iteration} requested tier="
                        f"{_bind_result.requested_tier} NOT realized: "
                        f"outcome={_bind_result.outcome} cause={_bind_result.cause} "
                        f"tier_after={_bind_result.tier_after} detail={_bind_result.detail}",
                        flush=True,
                    )
                _MULTIRES_MISMATCH_WARNED[_bind_result.outcome] = _mm_n + 1
        # Issue #595: record the tier actually in force plus the REAL camera
        # size this iteration trained on (after both startup downscales and any
        # tier rebind above) -- observed, never re-derived from args.
        _resolution_tier_log.observe(
            iteration, _active_tier,
            width=int(viewpoint_cam.image_width), height=int(viewpoint_cam.image_height),
        )
        _cur_multires_key = (_active_tier, int(viewpoint_cam.image_height), int(viewpoint_cam.image_width))
        if _cur_multires_key != _prev_active_tier[0]:
            print(f"[multires] step={iteration} tier={_active_tier} "
                  f"cam_h={viewpoint_cam.image_height} cam_w={viewpoint_cam.image_width}", flush=True)
            _prev_active_tier[0] = _cur_multires_key

        # ---- PR3: multi-keyframe gradient accumulation (extra views) --------
        # (2026-08-12 cadence/batching plan.) Accumulate the PRIMARY photometric
        # loss (the exact (1-lambda_dssim)*L1 + lambda_dssim*(1-SSIM) formula the
        # inline path computes below) over B-1 ADDITIONAL sampled views before
        # this iteration's own view is rendered. `.grad` is only zeroed at the
        # tail of the iteration (after optimizer.step()), so these backwards
        # accumulate into the same buffers the primary view's backward writes,
        # and the single optimizer.step() below sees the averaged gradient.
        #
        # Every statement in this block is skipped entirely at the default
        # streaming_optimization_batch_size == 1 -- the rest of the loop body is
        # the untouched pre-PR3 code, save for one `if _opt_batch_size > 1`
        # divide guard at the primary-loss assignment.
        #
        # Deliberate narrowing (documented, not silent): extra views contribute
        # ONLY the primary photometric term. They do not run depth/energy/
        # regularizer/anchor/auxiliary loss terms, MCMC's step_pre_backward hook,
        # provisional-support / visibility-EMA / last-supervised-iter /
        # stratified-sampling bookkeeping, or the dirty-region / consolidation-
        # burst queues (those keep their one-item-per-iteration cadence).
        if (
            _opt_batch_size > 1
            and not _placement_only
            and _streaming_optimization_runtime is None
        ):
            # sample_training_camera() advances streaming_scene._steps_since_new_frame
            # as a SIDE EFFECT on every call, and that counter drives per-ITERATION
            # cadences (H8 new-frame warmup, the consolidation-sampler idle switch,
            # the soft-anchor idle multiplier, the bubble re-centering policy).
            # Sampling B-1 extra views would otherwise advance those cadences B x
            # faster at B>1, silently changing scheduling rather than just batching.
            # Snapshot and restore it so it still ticks exactly once per iteration.
            _steps_since_new_frame_snapshot = getattr(
                streaming_scene, "_steps_since_new_frame", None
            )
            for _ in range(_opt_batch_size - 1):
                _extra_cam = streaming_scene.sample_training_camera(
                    gaussians, render_fn=render, pipe=pipe, background=background
                )
                if _extra_cam is None:
                    break
                if bool(getattr(args, "streaming_offload_camera_images", True)):
                    from scene.cameras import prepare_camera_for_render
                    prepare_camera_for_render(_extra_cam, device="cuda")
                    # Issue #396: every view in this B>1 batch is registered, so
                    # the whole batch is released together at the top of the next
                    # iteration -- never mid-batch, while another view of the
                    # same accumulation step could still hold it.
                    _residency_ledger.note_paged_in(_extra_cam)
                # Bind the same resolution tier the primary view uses, so a
                # multires run doesn't silently mix tiers within one batch.
                # A bind FAILURE leaves the extra camera at whatever tier it last
                # carried, so record it in its own bucket rather than silently.
                _extra_frame_obj = getattr(_extra_cam, "_streaming_frame", None)
                if _extra_frame_obj is not None:
                    _extra_bind_res = bind_resolution_tier_ex(
                        _extra_cam, _observation_db.get_frame(_extra_frame_obj.index),
                        _active_tier, db=_observation_db,
                    )
                    _extra_bind_ok = bool(_extra_bind_res.ok)
                    _extra_bind_key = f"{_active_tier}_extra_{'ok' if _extra_bind_ok else 'fail'}"
                    _multires_bind_stats[_extra_bind_key] = (
                        _multires_bind_stats.get(_extra_bind_key, 0) + 1
                    )
                    # issue #333 acceptance criterion 4: a batch must never
                    # silently contain mixed tiers. If this extra camera could
                    # not be put at the batch's tier, drop it from the batch
                    # (same `continue`-style skip the non-finite guard below
                    # uses; `_opt_batch_size` stays the divisor, as it already
                    # does for that skip) rather than supervising a view at a
                    # resolution nobody requested.
                    #
                    # The divisor claim IS true on this path and only on this
                    # path: `_extra_loss` below is divided by `_opt_batch_size`
                    # at construction, so a `continue` here cannot change it.
                    # The engine path's twin below had to be fixed to match
                    # (review of PR #528) -- it derives its divisor from the
                    # camera list it dispatches.
                    if not _extra_view_admitted(_extra_bind_res, _active_tier):
                        _multires_bind_stats["batch_tier_skip"] = (
                            _multires_bind_stats.get("batch_tier_skip", 0) + 1
                        )
                        # This can be SYSTEMATIC, not rare: e.g. a >1600px-wide
                        # source under the `-r -1` auto-downscale gives every
                        # evicted-cache camera `unresolved_native_res_mismatch`,
                        # so the batch shrinks toward 1 every iteration. The
                        # divisor stays `_opt_batch_size` (see above), so this
                        # is not an effective-LR change -- but B is then not
                        # buying the multi-view supervision it was set for.
                        # Say so loudly once instead of only in the JSON.
                        if not _MULTIRES_MISMATCH_WARNED.get("batch_tier_skip"):
                            _MULTIRES_MISMATCH_WARNED["batch_tier_skip"] = 1
                            print(
                                f"[multires] WARNING iter={iteration} dropped an extra batch view: "
                                f"could not put it at tier={_active_tier} "
                                f"(outcome={_extra_bind_res.outcome} "
                                f"tier_after={_extra_bind_res.tier_after}). The batch supervises "
                                f"fewer views than --streaming_optimization_batch_size; the loss "
                                f"divisor stays at the requested batch size, so the effective LR "
                                f"is unchanged and the step is simply supervised by fewer "
                                f"distinct views. If this recurs (see batch_tier_skip in "
                                f"multires_bind_stats.json), B is not buying the multi-view "
                                f"supervision it was set for.",
                                flush=True,
                            )
                        continue
                # PR2 cadence clocks: one more supervised/rendered view.
                _clk_supervision_view_count += 1
                _cadence.record_supervision_view()   # #289 supervision clock
                # Issue #703: rasterizer call for this extra view
                _clk_rasterizer_call_count += 1
                _extra_pkg = render(
                    _extra_cam, gaussians, pipe, bg,
                    update_sh_rest=update_sh_rest, render_depth=False,
                )
                _extra_gt = _extra_cam.original_image
                _extra_loss = (
                    (1.0 - opt.lambda_dssim) * l1_loss(_extra_pkg["render"], _extra_gt)
                    + opt.lambda_dssim * (1.0 - ssim(_extra_pkg["render"], _extra_gt))
                ) / _opt_batch_size
                # The primary view has its own non-finite guard below; extra
                # views need their own, or a NaN here would silently land in
                # .grad and be stepped on by a finite primary loss.
                if not bool(torch.isfinite(_extra_loss)):
                    _nan_term_report("opt_batch_extra_view", iteration)
                    if tb_writer is not None:
                        tb_writer.add_scalar("streaming/opt_batch_extra_nan_skip", 1.0, iteration)
                    del _extra_pkg, _extra_loss
                    continue
                # Issue #703: extra view's supervised pixels only if loss is finite (backward will run)
                _clk_supervised_pixel_count += (
                    int(_extra_cam.image_height) * int(_extra_cam.image_width)
                )
                _extra_loss.backward()
                _record_successful_supervision(
                    _supervision_source_index(_extra_cam)
                )
                del _extra_pkg, _extra_loss
            if _steps_since_new_frame_snapshot is not None:
                streaming_scene._steps_since_new_frame = _steps_since_new_frame_snapshot

        # ---- Issue #106: engine-path multi-keyframe supervision ------------
        # The block above is the INLINE path's gradient accumulation (PR #119,
        # photometric-only extra views, refuses selective_adam/sparse grads).
        # On the engine path we instead only COLLECT the extra cameras here and
        # hand them to GaussianOptimizationEngine.run_batched_step(), which
        # runs the full objective per view and steps against the union of the
        # batch's visibility filters. Same _steps_since_new_frame snapshot/
        # restore discipline as the inline block, and for the same reason: the
        # per-ITERATION cadences that counter drives must still tick once per
        # iteration, not once per supervision view.
        _engine_extra_cameras = []
        _engine_skipped_extra_views = 0
        if (
            _opt_batch_size > 1
            and not _placement_only
            and _streaming_optimization_runtime is not None
        ):
            # The divisor lives INSIDE run_batched_step (batch_size is derived
            # from the requests it receives), so unlike the inline path above a
            # camera dropped here also shrinks the divisor unless the drop is
            # declared. `ExtraViewBatchCollector` owns both halves of that
            # decision; see its docstring and the dispatch call below.
            _engine_batch = _ExtraViewBatchCollector(_active_tier, _opt_batch_size)
            _steps_since_new_frame_snapshot = getattr(
                streaming_scene, "_steps_since_new_frame", None
            )
            for _ in range(_opt_batch_size - 1):
                _extra_cam = streaming_scene.sample_training_camera(
                    gaussians, render_fn=render, pipe=pipe, background=background
                )
                if _extra_cam is None:
                    break
                if bool(getattr(args, "streaming_offload_camera_images", True)):
                    from scene.cameras import prepare_camera_for_render
                    prepare_camera_for_render(_extra_cam, device="cuda")
                    # Issue #396: every view in this B>1 batch is registered, so
                    # the whole batch is released together at the top of the next
                    # iteration -- never mid-batch, while another view of the
                    # same accumulation step could still hold it. Registered
                    # BEFORE the #333 tier-admission check below on purpose: a
                    # camera dropped by that check was still paged in here, and
                    # must still be paged back out at the top of the next
                    # iteration.
                    _residency_ledger.note_paged_in(_extra_cam)
                _extra_bind_res = None
                _extra_frame_obj = getattr(_extra_cam, "_streaming_frame", None)
                if _extra_frame_obj is not None:
                    _extra_bind_res = bind_resolution_tier_ex(
                        _extra_cam, _observation_db.get_frame(_extra_frame_obj.index),
                        _active_tier, db=_observation_db,
                    )
                    _extra_bind_ok = bool(_extra_bind_res.ok)
                    _extra_bind_key = f"{_active_tier}_extra_{'ok' if _extra_bind_ok else 'fail'}"
                    _multires_bind_stats[_extra_bind_key] = (
                        _multires_bind_stats.get(_extra_bind_key, 0) + 1
                    )
                # issue #333 acceptance criterion 4 (engine path twin of the
                # inline block above): never collect a camera that is not at
                # the batch's requested tier.
                if not _engine_batch.admit(_extra_cam, _extra_bind_res):
                    _multires_bind_stats["batch_tier_skip"] = (
                        _multires_bind_stats.get("batch_tier_skip", 0) + 1
                    )
                    # This can be SYSTEMATIC, not rare: e.g. a >1600px-wide
                    # source under the `-r -1` auto-downscale gives every
                    # evicted-cache camera `unresolved_native_res_mismatch`, so
                    # the batch shrinks toward 1 every iteration. The declared
                    # skip below keeps the DIVISOR at the requested batch size
                    # (so this is not an effective-LR change), but fewer
                    # distinct views are still being supervised per step.
                    if not _MULTIRES_MISMATCH_WARNED.get("batch_tier_skip"):
                        _MULTIRES_MISMATCH_WARNED["batch_tier_skip"] = 1
                        print(
                            f"[multires] WARNING iter={iteration} dropped an extra batch view: "
                            f"could not put it at tier={_active_tier} "
                            f"(outcome={_extra_bind_res.outcome} "
                            f"tier_after={_extra_bind_res.tier_after}). The batch supervises "
                            f"fewer views than --streaming_optimization_batch_size; the loss "
                            f"divisor stays at the requested batch size (declared to the engine "
                            f"as skipped_view_count), so the effective LR is unchanged and the "
                            f"step is simply supervised by fewer distinct views. If this recurs "
                            f"(see batch_tier_skip in multires_bind_stats.json), B is not "
                            f"buying the multi-view supervision it was set for.",
                            flush=True,
                        )
                    continue
                # PR2 cadence clocks: one more supervised/rendered view. Counted
                # at COLLECTION time to match the inline path's placement; the
                # engine reports its own actually-backwarded count separately as
                # OptimizationStepResult.supervision_view_count.
                _clk_supervision_view_count += 1
                _cadence.record_supervision_view()   # #289 supervision clock
            _engine_extra_cameras = _engine_batch.cameras
            _engine_skipped_extra_views = _engine_batch.skipped_view_count
            if _steps_since_new_frame_snapshot is not None:
                streaming_scene._steps_since_new_frame = _steps_since_new_frame_snapshot

            # sample_training_camera() draws WITH REPLACEMENT from the recent
            # keyframe window (--streaming_keyframe_window, default 8) plus an
            # occasional replay draw, so a size-B batch can contain the same
            # camera more than once -- at B=4 against a window of 8, only ~41%
            # of batches are all-distinct, and it is worse early in a run when
            # fewer frames have arrived. Two identical views at L/B each are
            # arithmetically one view at 2L/B, so this does not break
            # correctness, but it does mean "B=4" is not the same claim as
            # "4 distinct keyframes supervised this step". Measure it rather
            # than assume it: without this number, a null B>1 result cannot be
            # told apart from "the batch was not actually multi-view".
            # (The same draw-with-replacement pattern exists in the inline
            # path merged by #119 -- this is not engine-specific.)
            _batch_cams = [viewpoint_cam, *_engine_extra_cameras]
            _opt_batch_distinct_total += len({id(_c) for _c in _batch_cams})
            _opt_batch_count += 1
            # Issue #16b (camera-batching wiring audit): every camera in this
            # batch already had its own prepare_camera_for_render() call above
            # (individually, at collection time -- deliberately, since #396/
            # #333 need per-camera residency-ledger registration and tier
            # admission ordering as each view is sampled, before the full
            # batch membership is even known). This call is therefore a no-op
            # in steady state (cam.gsplat is already fresh for every warm
            # camera) and exists only to collapse any still-cold members'
            # tensor construction into one batched op now that the full,
            # already-collected batch is in hand -- it changes no ordering,
            # no residency semantics, and nothing observable when every
            # camera is already warm.
            if len(_batch_cams) > 1:
                from scene.cameras import prepare_camera_batch_for_render
                prepare_camera_batch_for_render(_batch_cams, device="cuda")

        if _w3_02_evidence is not None:
            _w3_02_evidence.begin(iteration)
        if _streaming_optimization_runtime is not None:
            _backward_ran = False
            _deferred_stale_mask = None
            _deferred_stale_count = 0

            def _streaming_post_render_hook(render_outputs):
                nonlocal _deferred_stale_mask, _deferred_stale_count
                render_pkg = render_outputs.payload
                _sync_streaming_state_lengths(gaussians)
                if _depth_lock_active or _depth_conflict_active:
                    ensure_depth_lock_buffers(gaussians)
                if float(getattr(args, "streaming_recency_reg_horizon", 0)) > 0:
                    _n_now = gaussians.get_opacity.shape[0]
                    _lsi = getattr(gaussians, "last_supervised_iter", None)
                    if _lsi is None or _lsi.shape[0] != _n_now:
                        gaussians.last_supervised_iter = torch.zeros(_n_now, dtype=torch.int32, device=gaussians.get_xyz.device)
                    gaussians.last_supervised_iter[render_pkg["visibility_filter"].detach()] = iteration
                _update_provisional_support(gaussians, viewpoint_cam, render_pkg, args)
                visible = render_pkg["visibility_filter"].detach().to(dtype=torch.bool).contiguous()
                if hasattr(gaussians, "visibility_ema") and gaussians.visibility_ema.numel() == visible.shape[0]:
                    from eligsir.utils.geometry_metrics import update_visibility_ema
                    update_visibility_ema(gaussians, visible)
                # issue #1262: accumulate per-Gaussian RESOLUTION EXPOSURE
                # (E^measurement) for this supervision view. Deliberately NOT derived
                # from render_pkg["visibility_filter"]/["radii"] above -- those answer
                # V^renderer, which eps2d floors independent of tier, and reading them
                # here would ship the mechanism silently inert. No-op when off.
                if (getattr(args, "streaming_resolution_exposure_veto", False)
                        or getattr(args, "streaming_resolution_exposure_telemetry", False)):
                    from eligsir.utils.resolution_exposure import (
                        accumulate_exposure,
                        exposure_config_from_args,
                        publish_exposure_young_min,
                    )
                    _exposure_result = accumulate_exposure(
                        gaussians, viewpoint_cam, exposure_config_from_args(args)
                    )
                    publish_exposure_young_min(gaussians, args)
                    if _exposure_result is not None:
                        _last_exposure_mask = _exposure_result.exposed
                        gaussians._last_exposure_mask = _last_exposure_mask
                        gaussians._last_exposure_ambiguous_frac = (
                            _exposure_result.ambiguous_frac
                        )
                        gaussians._last_exposure_iteration = int(iteration)
                        gaussians._last_exposure_tier = getattr(
                            getattr(
                                viewpoint_cam, "_canonical_resolution_tier", None
                            ),
                            "value",
                            None,
                        )
                if not _placement_only and iteration % 100 == 0:
                    with torch.no_grad():
                        promote_mask = gaussians.provisional & (
                            gaussians.support_count >= args.streaming_min_support_views
                        )
                        if promote_mask.any():
                            gaussians.provisional[promote_mask] = False
                            gaussians.raw_opacities.data[promote_mask] = inverse_sigmoid(
                                torch.tensor(args.streaming_promote_opacity, device="cuda")
                            )
                        age = n_frames_ingested - gaussians.birth_frame
                        stale_mask = gaussians.provisional & (
                            age > args.streaming_provisional_max_age
                        )
                        if stale_mask.any():
                            _deferred_stale_mask = stale_mask
                            _deferred_stale_count = int(stale_mask.sum().item())
                        conflict_prune = depth_conflict_prune_mask(gaussians, args)
                        if conflict_prune.any():
                            _deferred_stale_mask = merge_deferred_prune_masks(
                                _deferred_stale_mask, conflict_prune,
                            )
                            _deferred_stale_count = int(_deferred_stale_mask.sum().item())

            _streaming_optimization_runtime.post_render_hook = _streaming_post_render_hook

            # W3-14C round-3 bisection instrumentation (advisor-directed
            # discriminator check): brackets EVERY _sanitize_gaussian_params()
            # call site -- both arms' -- with the same "m_sanitize" region name,
            # to directly measure its per-call cost and confirm call COUNT
            # parity (2x/step each arm) rather than inferring it from reading
            # the hook wiring. See docs/architecture/w3_02_completion.md's
            # 2026-08-09 round-3 section.
            def _sanitize_timed(_where):
                with _timed("m_sanitize"):
                    # W3-61: branch-free core every call, sampled diagnostics
                    # (see the "loop_top" call site above for the cadence
                    # rationale).
                    _sanitize_with_seed_freeze(
                        _where, iteration, diag_interval=scalar_log_interval
                    )

            _streaming_optimization_runtime.post_optimizer_hook = lambda _: _sanitize_timed("after_optstep")
            _streaming_optimization_runtime.post_mcmc_hook = lambda _: _sanitize_timed("after_mcmc_noise")
            _streaming_optimization_runtime.timing_scope = _timed
            _engine_result = _mapping_application.tick()
            if _w3_02_evidence is not None:
                _w3_02_evidence.end()
            render_pkg = _engine_result.render_outputs.payload
            image = render_pkg["render"]
            loss = _engine_result.loss_breakdown.total_tensor
            xyz_lr = _engine_result.xyz_lr
            Ll1 = _engine_result.loss_breakdown.components["Ll1"]
            _ssim = _engine_result.loss_breakdown.components["ssim"]
            gt_image = viewpoint_cam.original_image
            _engine_components = _engine_result.loss_breakdown.components
            _depth_loss_val = _engine_components.get("depth")
            _free_loss_val = _engine_components.get("free_space")
            _anchor_loss_val = _engine_components.get("anchor")
            _mature_anchor_val = _engine_components.get("mature_anchor")
            _soft_anchor_val = _engine_components.get("soft_anchor")
            visible = render_pkg["visibility_filter"].detach().to(dtype=torch.bool).contiguous()
            _backward_ran = _engine_result.backward_ran
            # Issue #703: accumulate engine's supervision-work counters
            # The engine reports rasterizer_call_count (one per view rendered, including
            # primary + extra batched views) and supervised_pixel_count (sum of H*W for
            # views whose loss was finite and backwarded). Both are authoritative for
            # the engine arm; the inline arm maintains its own inline counter.
            _clk_rasterizer_call_count += _engine_result.rasterizer_call_count
            _clk_supervised_pixel_count += _engine_result.supervised_pixel_count
            for _successful_source_index in (
                _engine_result.successful_supervision_source_indices
            ):
                _record_successful_supervision(_successful_source_index)
            # Issue #809: the engine takes its optimizer step inside
            # run_batched_step() and cannot reach the inline arm's clock
            # increments at the `if iteration < opt.iterations:` block below.
            # It reports its own real step count (authoritative for this arm,
            # exactly as with rasterizer_call_count above); the inline arm
            # keeps its own increments, so neither arm double-counts.
            for _ in range(int(_engine_result.optimizer_step_count)):
                _clk_optimizer_step_id += 1
                _cadence.record_optimizer_step(iteration)   # #289 denominator clock
            _iter_since_new_frame = int(
                _streaming_optimization_runtime.iter_since_new_frame
            )
            _energy_grad_snapshot = (_engine_result.evidence.payload if _engine_result.evidence is not None else None)
            if _placement_only or _engine_result.step_allowed:
                with _timed("m_loss_item"):
                    _loss_val = _engine_result.loss_breakdown.total.item()
                streaming_scene.update_stratified_state(viewpoint_cam, _loss_val)
        else:
            _backward_ran = False
            # Issue #703: rasterizer call for primary inline view
            _clk_rasterizer_call_count += 1
            with _timed("m_render"):
                render_pkg = render(viewpoint_cam, gaussians, pipe, bg, update_sh_rest=update_sh_rest,
                                    render_depth=use_depth_loss)
            image = render_pkg["render"]
            # W3-14C round-3 bisection instrumentation (SPLAT_PROFILE_PHASES-gated,
            # zero overhead otherwise): "m_post_render" brackets this path's own
            # inline equivalent of the engine path's _streaming_post_render_hook
            # (state-length sync, depth-lock buffer ensure, recency-of-supervision
            # bookkeeping, provisional-support update, promote/stale/conflict-prune
            # sweep) so it is NOT silently folded into either arm's
            # "m_loss_assembly" number -- see engine.py's matching "m_post_render"
            # bracket and docs/architecture/w3_02_completion.md's 2026-08-09
            # round-3 section (advisor-caught asymmetry, fixed before the real A/B).
            # Manual __enter__/__exit__ (not `with`) so this span's existing
            # indentation is untouched.
            _w3_14c_post_render_bracket = _timed("m_post_render")
            _w3_14c_post_render_bracket.__enter__()

            _sync_streaming_state_lengths(gaussians)
            if _depth_lock_active or _depth_conflict_active:
                ensure_depth_lock_buffers(gaussians)

            # Recency-of-supervision opacity-reg relaxation: record which Gaussians
            # were actually in this iteration's training render, so
            # _streaming_young_opacity_reg can relax pressure on ones that haven't
            # been supervised in a while (not just newborns). Default-off
            # (streaming_recency_reg_horizon == 0) and a no-op cost otherwise: one
            # boolean-mask assignment on an int32 buffer.
            if float(getattr(args, "streaming_recency_reg_horizon", 0)) > 0:
                _n_now = gaussians.get_opacity.shape[0]
                _lsi = getattr(gaussians, "last_supervised_iter", None)
                if _lsi is None or _lsi.shape[0] != _n_now:
                    gaussians.last_supervised_iter = torch.zeros(_n_now, dtype=torch.int32, device=gaussians.get_xyz.device)
                gaussians.last_supervised_iter[render_pkg["visibility_filter"].detach()] = iteration

            # Support update for provisional Gaussians (Step 8)
            _update_provisional_support(gaussians, viewpoint_cam, render_pkg, args)

            # SLAM Lifecycle: Promote non-structurally before forward pass consumption.
            # Promotion only modifies in-place tensor values (opacities, provisional flag)
            # and does NOT change N, so it is safe before backward.
            # PRUNING is deferred to after optimizer.step() + MCMC so the render/loss/
            # backward graph is never built on one parameter tensor set while the
            # optimizer references a different (sliced) one.
            _deferred_stale_mask = None
            _deferred_stale_count = 0
            if not _placement_only and iteration % 100 == 0:
                with torch.no_grad():
                    # Promote (in-place, no structural change)
                    promote_mask = gaussians.provisional & (gaussians.support_count >= args.streaming_min_support_views)
                    if promote_mask.any():
                        gaussians.provisional[promote_mask] = False
                        target_op = inverse_sigmoid(torch.tensor(args.streaming_promote_opacity, device="cuda"))
                        gaussians.raw_opacities.data[promote_mask] = target_op
                        print(f"[streaming] step={iteration} promoted {promote_mask.sum().item()} points to permanent structure.", flush=True)

                    # Compute stale mask but defer the actual prune to after backward.
                    age = n_frames_ingested - gaussians.birth_frame
                    stale_mask = gaussians.provisional & (age > args.streaming_provisional_max_age)
                    if stale_mask.any():
                        _deferred_stale_mask = stale_mask
                        _deferred_stale_count = int(stale_mask.sum().item())

                    conflict_prune = depth_conflict_prune_mask(gaussians, args)
                    if conflict_prune.any():
                        _deferred_stale_mask = merge_deferred_prune_masks(
                            _deferred_stale_mask, conflict_prune,
                        )
                        if _deferred_stale_mask is not None:
                            _deferred_stale_count = int(_deferred_stale_mask.sum().item())

            _w3_14c_post_render_bracket.__exit__(None, None, None)
            _w3_14c_loss_bracket = _timed("m_loss_assembly")
            _w3_14c_loss_bracket.__enter__()
            # ---- Loss ---------------------------------------------------------
            gt_image = viewpoint_cam.original_image
            if not bool(torch.isfinite(gt_image).all()):
                # Poisoned source frame (confirmed live: a corrupted ScanNet++
                # iPhone capture with NaN pixels caused a sustained NaN-loss tail
                # because the per-step nan-guard below only skips the optimizer
                # step, it doesn't stop this camera from being resampled forever).
                # Sanitize the cached tensor IN PLACE (once) so every future
                # sample of this camera — training, keyframe re-checks, replay —
                # sees clean data instead of re-triggering the guard every time.
                _vp_id = getattr(viewpoint_cam, "image_name", getattr(viewpoint_cam, "uid", "?"))
                print(
                    f"[gt-sanitize] non-finite ground-truth pixels in view={_vp_id} "
                    f"— sanitizing cached original_image in place (one-time).",
                    flush=True,
                )
                gt_image = torch.nan_to_num(gt_image, nan=0.0, posinf=1.0, neginf=0.0)
                viewpoint_cam.original_image = gt_image.detach()
                from eligsir.utils.score_contract import declare_camera_score_provenance

                declare_camera_score_provenance(
                    viewpoint_cam,
                    context="streaming_loss_nan_sanitization",
                )
            Ll1 = l1_loss(image, gt_image)
            _ssim = ssim(image, gt_image)
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - _ssim)
            # PR3 (2026-08-12 cadence/batching plan): when accumulating over B
            # views, this view's primary photometric term is one of B summands,
            # so it carries 1/B weight -- matching the extra views above. No-op
            # (dead branch) at the default B == 1, which is why every other
            # statement in this region is untouched pre-PR3 code. Auxiliary /
            # depth / energy / regularizer terms added BELOW this point are
            # deliberately NOT divided: they are computed once per optimizer
            # step against this view (see the task report's Bucket A/B list).
            if _opt_batch_size > 1:
                loss = loss / _opt_batch_size

            # W3-06: realized-gain bookkeeping (plan section 11.4). Piggy-
            # backs on the loss the loop already computed for its own
            # supervision purposes -- reads `loss`, never writes it, never
            # affects backward/optimizer. True no-op (dict lookup that misses)
            # for any camera not currently tracked, and a true no-op overall
            # when shadow-utility is disabled. See realized_gain.py's module
            # docstring for the "no extra render" design.
            if _shadow_utility_policy.enabled:
                try:
                    _shadow_utility_policy.observe_training_loss(
                        id(viewpoint_cam), int(iteration), float(loss.detach())
                    )
                except Exception as _shadow_loss_exc:  # noqa: BLE001 -- must never break real training
                    print(f"[shadow-utility] observe_training_loss failed (non-fatal): {_shadow_loss_exc}", flush=True)
                # Periodically persist the policy-cost summary (plan section
                # 11.7 deliverable 5) so scripts/analyze_frame_utility.py has
                # something to read even if the run is killed before a clean
                # exit -- cheap (json.dump of ~3 floats), overwritten in place.
                if int(iteration) % 500 == 0:
                    try:
                        import json as _json_shadow
                        with open(
                            os.path.join(args.model_path, "shadow_utility", "shadow_utility_cost.json"), "w"
                        ) as _f_shadow_cost:
                            _json_shadow.dump(_shadow_utility_policy.cost_overhead_ms_summary(), _f_shadow_cost, indent=2)
                    except OSError:
                        pass

            # Phase 5: differentiable energy loss flag (replaces reg call sites when enabled)
            _energy_loss_enabled_main = getattr(args, "energy_loss_enabled", False)
            if _energy_loss_enabled_main:
                from eligsir.utils.differentiable_energy_loss import compute_differentiable_loss
                if sparse_active_set:
                    _p5_visible = render_pkg["visibility_filter"].detach()
                    _p5_provisional = gaussians.provisional.detach()
                    _p5_active = _p5_visible | _p5_provisional
                else:
                    _p5_active = None
                _p5_result = compute_differentiable_loss(
                    gaussians=gaussians,
                    iteration=iteration,
                    cap_max=args.cap_max,
                    active_mask=_p5_active,
                    include_effective_count=use_energy_mcmc,
                    w_opacity_reg=args.opacity_reg,
                    w_scale_reg=args.scale_reg,
                    w_opacity_entropy=args.lambda_opacity_entropy,
                    w_effective_count=args.lambda_eff_count,
                    w_anisotropy_reg=float(args.anisotropy_reg),
                    anisotropy_soft_limit=float(getattr(args, "anisotropy_soft_limit", 10.0)),
                    w_needle_reg=float(getattr(args, "needle_reg_weight", 0.0)),
                    needle_threshold=float(getattr(args, "needle_reg_threshold", 0.3)),
                    scale_reg_mode=str(getattr(args, "scale_reg_mode", "l1")),
                    max_iterations=opt.iterations,
                )
                loss = loss + _p5_result.total
                if tb_writer:
                    for _k, _v in _p5_result.to_tensorboard_dict().items():
                        tb_writer.add_scalar(_k, _v, iteration)
            else:
                # Regularisation: local (visible) or global depending on active-set mode
                from eligsir.utils.compiled_kernels import active_reg_core
                _w_aniso = float(args.anisotropy_reg)
                # ---- Birth-local opacity-reg relaxation (live path) --------------
                # Two composable, default-off knobs ported from the offline loop
                # (train.py) and feat/offline-gap; both DEAD on the live streaming
                # path before this. When BOTH are off, the code below collapses to
                # the original single active_reg_core(..., w_opacity=args.opacity_reg)
                # call (behavior-neutral — see _streaming_young_opacity_reg).
                #   Task 1: streaming_young_reg_exempt_iters — exempt Gaussians whose
                #           age (iter - birth_iter) is below the threshold from the
                #           opacity regularizer entirely.
                #   Task 2: streaming_maturity_reg_ramp — scale per-Gaussian opacity
                #           pressure by clamp(support_count/target_support,
                #           ramp_min, 1.0) so newborns get ~ramp_min pressure.
                _young_exempt_iters = int(args.streaming_young_reg_exempt_iters)
                _maturity_ramp = bool(args.streaming_maturity_reg_ramp)
                _recency_reg_on = float(getattr(args, "streaming_recency_reg_horizon", 0)) > 0
                _opacity_reg_custom = _young_exempt_iters > 0 or _maturity_ramp or _recency_reg_on
                if sparse_active_set:
                    # Step 10: Refine active set to include provisional/new points
                    _visible = render_pkg["visibility_filter"].detach()
                    _provisional = gaussians.provisional.detach()
                    _active = _visible | _provisional

                    if _opacity_reg_custom:
                        # Scale reg / anisotropy unchanged; opacity reg handled per-Gaussian.
                        loss = _add_guarded(loss, active_reg_core(
                            gaussians.get_opacity[_active],
                            gaussians.get_scaling[_active],
                            w_opacity=0.0,
                            w_scale=args.scale_reg,
                            w_aniso=_w_aniso,
                        ) + _streaming_young_opacity_reg(
                            gaussians, args, iteration, _active,
                            _young_exempt_iters, _maturity_ramp,
                        ), "active_reg+young", iteration)
                    else:
                        loss = _add_guarded(loss, active_reg_core(
                            gaussians.get_opacity[_active],
                            gaussians.get_scaling[_active],
                            w_opacity=args.opacity_reg,
                            w_scale=args.scale_reg,
                            w_aniso=_w_aniso,
                        ), "active_reg", iteration)
                else:
                    if _opacity_reg_custom:
                        loss = _add_guarded(loss, active_reg_core(
                            gaussians.get_opacity,
                            gaussians.get_scaling,
                            w_opacity=0.0,
                            w_scale=args.scale_reg,
                            w_aniso=_w_aniso,
                        ) + _streaming_young_opacity_reg(
                            gaussians, args, iteration, None,
                            _young_exempt_iters, _maturity_ramp,
                        ), "active_reg+young", iteration)
                    else:
                        loss = _add_guarded(loss, active_reg_core(
                            gaussians.get_opacity,
                            gaussians.get_scaling,
                            w_opacity=args.opacity_reg,
                            w_scale=args.scale_reg,
                            w_aniso=_w_aniso,
                        ), "active_reg", iteration)

                # Energy-guided losses (full path, same as offline when enabled)
                if use_energy_mcmc:
                    from eligsir.utils.energy_mcmc import (
                        compute_effective_count_loss,
                        compute_opacity_entropy_loss,
                        compute_gaussian_utility,
                    )
                    # Advance the fade-suppression ramp every iteration, decoupled from
                    # _run_energy's ~sparse_energy_global_interval (default 500) cadence --
                    # a real fade can bottom out in ~600-700 iterations (measured on floor3),
                    # so gating the ramp to the same coarse interval as the loss itself left
                    # the controller only 1-2 ramp steps by the time the dip peaked. The loss
                    # below still only applies every _run_energy tick, but now reads whatever
                    # _fade_scale has ramped to by that point instead of computing it fresh.
                    # issue #282: the disabled/observe-only ternary now lives
                    # inside the monitor (effective_suppression_scale), which
                    # is also the single place the APPLIED scale is recorded
                    # for report.json -- historically it had to be inferred
                    # from code because nothing logged it.
                    _fade_scale = seed_fade_monitor.effective_suppression_scale()
                    _energy_interval = getattr(args, "sparse_energy_global_interval", 500)
                    _run_energy = not sparse_active_set or _energy_interval <= 0 or iteration % _energy_interval == 0
                    if _run_energy:
                        # Seed-fade fix: the global count/entropy opacity losses take the FULL
                        # _opacity tensor, so they push opacity DOWN on out-of-view geometry that
                        # has no reconstruction loss to resist it -> the seed implodes mid-travel
                        # (worse in online_mode where N_target is pinned ~q_start*cap_max while N
                        # balloons). When enabled, detach the gradient on inactive (out-of-view,
                        # non-provisional) opacities so count-control only pressures supervised
                        # Gaussians. N_eff stays numerically global (detach preserves value).
                        _op_in = gaussians._opacity
                        if getattr(args, "streaming_count_loss_local", False):
                            _vis = render_pkg["visibility_filter"].detach()
                            _prov = gaussians.provisional.detach() if hasattr(gaussians, "provisional") else torch.zeros_like(_vis)
                            _active_op = _vis | _prov
                            if _active_op.shape[0] == _op_in.shape[0]:
                                _op_in = torch.where(
                                    _active_op.unsqueeze(-1), _op_in, _op_in.detach()
                                )
                        # Merge B: anneal the count-loss N_target over the AnnealClock horizon
                        # when stashed (decoupled from the loop bound). This subsumes the A6
                        # _online_schedule_horizon: apply_online_schedule_overrides sets the
                        # clock = sched (schedule_horizon) unless an explicit anneal_horizon is
                        # given. Default => horizon == loop bound => byte-identical.
                        _anneal_clock = getattr(opt, "_anneal_clock", None)
                        _eff_horizon = _anneal_clock.horizon if _anneal_clock is not None else opt.iterations
                        L_eff, N_eff, N_target = compute_effective_count_loss(
                            _op_in,
                            iteration=iteration,
                            cap_max=args.cap_max,
                            max_iterations=_eff_horizon,
                            target_splat_end=args.mcmc_target_splat_end,
                            q_start=getattr(args, "mcmc_target_q_start", 0.05),
                            q_end=getattr(args, "mcmc_target_q_end", 0.85),
                            tau_N=getattr(args, "mcmc_target_tau", 0.45),
                            target_capacity=int(getattr(args, "streaming_target_capacity", 0)),
                        )
                        loss = _add_guarded(loss, _fade_scale * args.lambda_eff_count * L_eff, "eff_count", iteration)
                        loss = _add_guarded(loss, _fade_scale * args.lambda_opacity_entropy * compute_opacity_entropy_loss(_op_in), "opacity_entropy", iteration)

            # ---- Decaying isotropy reg on immature (provisional+young) Gaussians ----
            if float(getattr(args, "streaming_young_isotropy_weight", 0.0)) > 0.0:
                _iso_loss = _streaming_young_isotropy_reg(gaussians, args, n_frames_ingested)
                if _iso_loss is not None:
                    loss = _add_guarded(loss, _iso_loss, "young_isotropy", iteration)

            # ---- Depth supervision loss ----------------------------------------
            _depth_loss_val = None
            if use_depth_loss and "rendered_depth" in render_pkg:
                _surface_prior = _get_surface_prior(viewpoint_cam, args, timed=_timed) if getattr(args, "surface_confidence_depth_loss", False) else None
                sensor_d_cpu = _surface_prior.depth if _surface_prior is not None else _get_sensor_depth(
                    viewpoint_cam, image.shape[1], image.shape[2]
                )
                if sensor_d_cpu is not None:
                    sensor_d = sensor_d_cpu.to(device="cuda", non_blocking=True)
                    prior_conf = None
                    if _surface_prior is not None:
                        prior_conf = _surface_prior.confidence.to(device="cuda", non_blocking=True)
                    _depth_quantity = str(getattr(args, "streaming_depth_loss_quantity", "d")).lower()
                    _alpha_gate = float(getattr(args, "streaming_depth_loss_alpha_gate", 0.0))
                    if _depth_quantity not in DEPTH_LOSS_QUANTITIES:
                        raise ValueError(
                            f"Unrecognized --streaming_depth_loss_quantity={_depth_quantity!r}; "
                            f"expected one of {DEPTH_LOSS_QUANTITIES!r}."
                        )
                    if not math.isfinite(_alpha_gate) or not (0.0 <= _alpha_gate < 1.0):
                        raise ValueError(
                            f"Invalid --streaming_depth_loss_alpha_gate={_alpha_gate!r}; "
                            f"expected a finite value in [0.0, 1.0)."
                        )
                    global _DEPTH_LOSS_STARTUP_LOGGED_INLINE, _DEPTH_LOSS_ED_ALPHA_GATE_WARNED_INLINE
                    if not _DEPTH_LOSS_STARTUP_LOGGED_INLINE:
                        print(f"[DepthLoss] quantity={_depth_quantity} alpha_gate={_alpha_gate}", flush=True)
                        _DEPTH_LOSS_STARTUP_LOGGED_INLINE = True
                    if _depth_quantity == "ed" and _alpha_gate == 0.0 and not _DEPTH_LOSS_ED_ALPHA_GATE_WARNED_INLINE:
                        print(
                            "[DepthLoss] WARNING: quantity='ed' with alpha_gate=0.0 is "
                            "numerically unstable at low alpha -- consider setting "
                            "--streaming_depth_loss_alpha_gate",
                            flush=True,
                        )
                        _DEPTH_LOSS_ED_ALPHA_GATE_WARNED_INLINE = True
                    _depth_quantity_uses_ed = _depth_quantity in ("ed", "ed_alpha_weighted")
                    if _depth_quantity_uses_ed:
                        if "rendered_depth_ed" not in render_pkg:
                            raise RuntimeError(
                                f"streaming_depth_loss_quantity={_depth_quantity!r} was requested "
                                "but render_pkg has no 'rendered_depth_ed' key -- the renderer did "
                                "not produce it (check render_depth=True was passed). Refusing to "
                                "silently fall back to D, since that would silently corrupt an "
                                "experiment comparison."
                            )
                        rend_d = render_pkg["rendered_depth_ed"]  # [1, H, W], alpha-normalised
                    else:
                        rend_d = render_pkg["rendered_depth"]  # [1, H, W]
                    valid = (sensor_d > 0) & (rend_d.detach() > 0)
                    # render_pkg["alpha"] is [H, W, 1], not [1, H, W] -- permute
                    # before comparing against/broadcasting with `valid`/`rend_d`
                    # (see gsplat_backend.py's NOTE on this same pitfall; a bare
                    # [1,H,W] & [H,W,1] broadcast silently misaligns whenever
                    # H == W and errors otherwise). Fetched once and reused for
                    # both the optional hard gate and issue #1833's per-pixel
                    # alpha weighting.
                    _alpha_map = None
                    if _alpha_gate > 0.0 or _depth_quantity == "ed_alpha_weighted":
                        _alpha_map = render_pkg.get("alpha")
                        if _alpha_map is None:
                            raise RuntimeError(
                                (
                                    "streaming_depth_loss_alpha_gate > 0.0"
                                    if _alpha_gate > 0.0
                                    else "streaming_depth_loss_quantity='ed_alpha_weighted'"
                                )
                                + " was requested but render_pkg has no 'alpha' key -- refusing "
                                "to silently apply no gate/weight at all, since that would "
                                "silently corrupt an experiment comparison (same failure mode as "
                                "the ed-without-rendered_depth_ed guard above)."
                            )
                        _alpha_map = _alpha_map.detach()
                        if _alpha_map.dim() == 3 and _alpha_map.shape[-1] == 1:
                            _alpha_map = _alpha_map.permute(2, 0, 1)
                        if _alpha_gate > 0.0:
                            valid = valid & (_alpha_map > _alpha_gate)
                    # NOTE: keep the boolean-indexed (valid-subset) form — masking
                    # BEFORE huber keeps autograd to valid pixels only. A full-image
                    # masked-sum backprops through every pixel and ~2x'd iter time on
                    # sparse depth. The .any() host-sync is cheaper than that.
                    if valid.any():
                        if depth_loss_type == "huber":
                            depth_err = torch.nn.functional.huber_loss(
                                rend_d[valid], sensor_d[valid], reduction="none", delta=0.1
                            )
                        else:
                            depth_err = (rend_d[valid] - sensor_d[valid]).abs()
                        _disc_weight = getattr(viewpoint_cam, "_resolution_depth_confidence_weight", None)
                        if _disc_weight is not None:
                            _disc_weight = _disc_weight.to(device=rend_d.device, non_blocking=True)
                            if _disc_weight.dim() == 2:
                                _disc_weight = _disc_weight.unsqueeze(0)
                        # Issue #1833: multiply the detached per-pixel alpha into
                        # the weight stack for 'ed_alpha_weighted' -- see
                        # objective.py's matching implementation/rationale.
                        _alpha_weight = _alpha_map if _depth_quantity == "ed_alpha_weighted" else None
                        combined_weights = combine_depth_loss_weights(prior_conf, _disc_weight, _alpha_weight)
                        L_depth = weighted_depth_error_mean(depth_err, combined_weights, valid)
                        # Rendered DEPTH is a separate raster channel from the color
                        # image; a degenerate Gaussian can put NaN/Inf there while the
                        # color render stays finite. Drop the term (detached zero, so no
                        # NaN gradient) for this iter rather than let it poison the loss.
                        if not bool(torch.isfinite(L_depth)):
                            _nan_term_report("depth_loss", iteration)
                            L_depth = torch.zeros((), device=L_depth.device)
                        loss = loss + depth_loss_weight * L_depth
                        if tb_writer is not None and iteration % scalar_log_interval == 0:
                            _depth_loss_val = L_depth.item()
                            tb_writer.add_scalar(
                                "streaming/depth_loss_raw_magnitude",
                                float(L_depth.detach().item()),
                                iteration,
                            )
                            if bool(getattr(args, "streaming_depth_alpha_bin_diagnostic", False)):
                                # Training-time D/ED-vs-alpha instrumentation -- kept in sync
                                # with objective.py's identical block (see that file's comment
                                # for the full rationale). rendered_depth_ed is guaranteed
                                # present here regardless of streaming_depth_loss_quantity --
                                # render_depth=use_depth_loss is exactly the condition gating
                                # this whole block.
                                _diag_alpha_map = render_pkg.get("alpha")
                                if _diag_alpha_map is not None:
                                    _diag_alpha_map = _diag_alpha_map.detach()
                                    if _diag_alpha_map.dim() == 3 and _diag_alpha_map.shape[-1] == 1:
                                        _diag_alpha_map = _diag_alpha_map.permute(2, 0, 1)
                                    _diag_stats = compute_alpha_bin_residuals(
                                        render_pkg["rendered_depth"],
                                        render_pkg["rendered_depth_ed"],
                                        sensor_d,
                                        _diag_alpha_map,
                                    )
                                    log_alpha_bin_residuals_to_tb(tb_writer, _diag_stats, iteration)

            # ---- H9: Anchor loss for young provisional splats ------------------
            # Penalise drift from the depth-insertion position while the splat is
            # young. Weight decays linearly to zero over anchor_decay_steps iters.
            _anchor_loss_val = None
            if _anchor_loss_weight > 0 and hasattr(gaussians, "anchor_xyz"):
                with torch.no_grad():
                    age_iters = (iteration - gaussians.anchor_iter.long()).clamp(min=0)
                    young_mask = (age_iters < _anchor_decay_steps) & gaussians.provisional
                if young_mask.any():
                    decay = (1.0 - age_iters[young_mask].float() / _anchor_decay_steps).clamp(0.0, 1.0)
                    dxyz = gaussians.get_xyz[young_mask] - gaussians.anchor_xyz[young_mask]
                    L_anchor = (decay.unsqueeze(1) * dxyz.pow(2)).sum(dim=1).mean()
                    loss = loss + _anchor_loss_weight * L_anchor
                    if tb_writer is not None and iteration % scalar_log_interval == 0:
                        _anchor_loss_val = L_anchor.item()

            # ---- Mature-Gaussian soft-anchor anti-fade losses (Component C) -----
            # Constant-weight (non-decaying) anchor losses on xyz/scale/opacity for
            # MATURE Gaussians. Prevents confirmed geometry from fading when old
            # frames leave the training window.
            # Task 5 (Axis-A quiescence isolation): the LIFECYCLE QUIESCENCE behaviour
            # — MATURE/FROZEN Gaussians are excluded from MCMC relocation/perturbation
            # — is implemented entirely in utils/strategies/mcmc_strategy.py (dead_mask
            # &= lifecycle_state < 2) and in the FROZEN grad-zeroing below. It is gated
            # ONLY by streaming_lifecycle_enabled + the presence of lifecycle_state, and
            # is INDEPENDENT of the mature-anchor losses gated immediately below (which
            # require the anchor weights to be > 0) and of streaming_sampling_mode.
            #
            # Therefore the endorsed quiescence-only flag combo is:
            #     streaming_lifecycle_enabled=True
            #     streaming_mature_anchor_xyz_weight=0.0
            #     streaming_mature_anchor_scale_weight=0.0
            #     streaming_mature_anchor_opacity_weight=0.0   (all default 0.0)
            #     streaming_sampling_mode=legacy               (default)
            # With the anchor weights at 0 the block below is a no-op (L_ma never added),
            # so quiescence runs WITHOUT the harmful mature-anchor losses. See
            # tests/test_lifecycle_quiescence_isolation.py.
            _mature_anchor_val = None
            if _lifecycle_enabled and _mature_anchor_xyz_w + _mature_anchor_scale_w + _mature_anchor_opacity_w > 0 and hasattr(gaussians, "lifecycle_state"):
                with torch.no_grad():
                    mature_mask = (gaussians.lifecycle_state >= 2)  # MATURE or FROZEN
                if mature_mask.any():
                    L_ma = torch.tensor(0.0, device="cuda")
                    if _mature_anchor_xyz_w > 0 and hasattr(gaussians, "anchor_xyz"):
                        dxyz = gaussians.get_xyz[mature_mask] - gaussians.anchor_xyz[mature_mask]
                        L_ma = L_ma + _mature_anchor_xyz_w * dxyz.pow(2).sum(dim=1).mean()
                    if _mature_anchor_scale_w > 0 and hasattr(gaussians, "anchor_scale_log"):
                        d_scale = gaussians.raw_scales[mature_mask] - gaussians.anchor_scale_log[mature_mask]
                        L_ma = L_ma + _mature_anchor_scale_w * d_scale.pow(2).mean()
                    if _mature_anchor_opacity_w > 0 and hasattr(gaussians, "anchor_opacity_logit"):
                        d_op = gaussians.raw_opacities[mature_mask] - gaussians.anchor_opacity_logit[mature_mask]
                        L_ma = L_ma + _mature_anchor_opacity_w * d_op.pow(2).mean()
                    loss = loss + L_ma
                    _mature_anchor_val = L_ma.item()

            # ---- 2.4: distance-weighted soft anchor (bubble-as-local-bounds) ----
            # Loss-native replacement for the hard out-of-bubble freeze: mature
            # geometry is pulled toward its anchor snapshot with a weight that is
            # ~0 inside the bubble (fully plastic) and ->1 well outside (seam band
            # R..1.2R). The outer world stays rendered and optimizable — drift is
            # penalized, not forbidden — so MCMC statistics, Adam state, and alpha
            # compositing all stay clean. lambda:
            #   L = lam * mean_i[ w_i * (||dxyz||^2/R^2 + ||dscale||^2 + ||dop||^2) ]
            #   w_i = sigmoid((d_i - R) / (0.2 R)),  i in MATURE|FROZEN (anchored)
            # In the idle regime (no admission for consolidation_idle_steps) lam is
            # loosened by streaming_soft_anchor_idle_mult so global refinement can
            # actually move things.
            _soft_anchor_val = None
            _sa_lam = float(getattr(args, "streaming_soft_anchor_weight", 0.0))
            if (
                _sa_lam > 0
                and _lifecycle_enabled
                and hasattr(gaussians, "lifecycle_state")
                and hasattr(gaussians, "anchor_xyz")
                and getattr(args, "_bubble_center", None) is not None
            ):
                _sa_R = float(getattr(args, "streaming_bubble_radius", 5.0))
                with torch.no_grad():
                    _sa_anchored = gaussians.lifecycle_state >= 2
                if _sa_anchored.any() and _sa_R > 0:
                    _sa_center = args._bubble_center.to(gaussians.get_xyz.device)
                    _sa_xyz = gaussians.get_xyz[_sa_anchored]
                    with torch.no_grad():
                        _sa_d = torch.linalg.norm(_sa_xyz.detach() - _sa_center, dim=1)
                        # Hard-zero inside the bubble: the raw sigmoid has heavy
                        # tails (w=0.27 at d=0.8R, 0.5 at d=R), which pulled MATURE
                        # Gaussians inside the ACTIVE region toward their maturity
                        # snapshots — the known-harmful mature-anchor pathology
                        # (measured -0.89 dB on fr3guards_bubble_noop). The anchor
                        # must be a pure outside-consolidation force.
                        _sa_w = torch.sigmoid((_sa_d - _sa_R) / (0.2 * _sa_R)) * (_sa_d > _sa_R)
                    _sa_terms = (
                        (_sa_xyz - gaussians.anchor_xyz[_sa_anchored]).pow(2).sum(dim=1) / (_sa_R * _sa_R)
                    )
                    if hasattr(gaussians, "anchor_scale_log"):
                        _sa_terms = _sa_terms + (
                            gaussians.raw_scales[_sa_anchored] - gaussians.anchor_scale_log[_sa_anchored]
                        ).pow(2).sum(dim=1)
                    if hasattr(gaussians, "anchor_opacity_logit"):
                        _sa_terms = _sa_terms + (
                            gaussians.raw_opacities[_sa_anchored] - gaussians.anchor_opacity_logit[_sa_anchored]
                        ).pow(2).squeeze(-1)
                    _sa_eff_lam = _sa_lam
                    _sa_idle_steps = int(getattr(args, "streaming_consolidation_idle_steps", 500))
                    if getattr(streaming_scene, "_steps_since_new_frame", 0) > _sa_idle_steps:
                        _sa_eff_lam = _sa_lam * float(getattr(args, "streaming_soft_anchor_idle_mult", 0.3))
                    L_sa = _sa_eff_lam * (_sa_w * _sa_terms).mean()
                    if torch.isfinite(L_sa):
                        loss = loss + L_sa
                        _soft_anchor_val = L_sa.item()

            # ---- Tri-state ray semantics classification (Wave 3, PR W3-08) --
            # Same logic as eligsir/optimization/gaussian/objective.py's
            # assemble_loss (kept in sync deliberately; see that module for
            # full commentary) -- this inline block is the legacy,
            # still-default-active loss path (--streaming_optimization_engine
            # defaults False), so the wiring must exist here too or the
            # feature is unreachable by any real run today.
            _rs_enabled = bool(getattr(args, "streaming_ray_semantics_enabled", False))
            _ao_enabled = bool(getattr(args, "streaming_appearance_only_enabled", False))
            _rs_classification = None
            _rs_sensor_d_hw = None
            if _rs_enabled or _ao_enabled:
                _rs_sensor_d = _get_sensor_depth(viewpoint_cam, image.shape[1], image.shape[2])
                if _rs_sensor_d is not None:
                    _rs_sensor_d = _rs_sensor_d.to("cuda", non_blocking=True)
                    _rs_sensor_d_hw = _rs_sensor_d.squeeze(0)
                    _rs_classification = _classify_view_rays(viewpoint_cam, _rs_sensor_d_hw, args, streaming_scene, iteration)
                    if _rs_classification is not None and tb_writer is not None and iteration % scalar_log_interval == 0:
                        for _rs_k, _rs_v in _rs_classification.counts.as_dict().items():
                            tb_writer.add_scalar(f"streaming/ray_semantics/{_rs_k}", _rs_v, iteration)

            # ---- Free-space / floater loss -------------------------------------
            # Penalise opacity that is rendered in front of the observed surface.
            # Targets mid-air splats that photometric loss would otherwise keep.
            _free_loss_val = None
            free_space_weight = getattr(args, "streaming_free_space_loss_weight", 0.0)
            if free_space_weight > 0 and use_depth_loss and "alpha" in render_pkg and "rendered_depth" in render_pkg:
                _surface_prior_fs = _get_surface_prior(viewpoint_cam, args, timed=_timed) if getattr(args, "surface_confidence_depth_loss", False) else None
                sensor_d_fs = _surface_prior_fs.depth if _surface_prior_fs is not None else _get_sensor_depth(viewpoint_cam, image.shape[1], image.shape[2])
                if sensor_d_fs is not None:
                    sensor_d_fs = sensor_d_fs.to("cuda", non_blocking=True)
                    rend_d_fs = render_pkg["rendered_depth"]         # [1, H, W]
                    # render_alphas[0] from gsplat is [H, W, 1]; permute to [1, H, W]
                    rend_a_raw = render_pkg["alpha"]
                    if rend_a_raw.dim() == 3 and rend_a_raw.shape[-1] == 1:
                        rend_a_fs = rend_a_raw.permute(2, 0, 1)     # [1, H, W]
                    else:
                        rend_a_fs = rend_a_raw.unsqueeze(0) if rend_a_raw.dim() == 2 else rend_a_raw
                    # Floater: rendered alpha is significant AND rendered depth is
                    # shallower than the sensor surface by at least 5 cm.
                    floater = (
                        (sensor_d_fs > 0)
                        & (rend_d_fs.detach() > 0)
                        & (rend_a_fs.detach() > 0.2)
                        & (rend_d_fs.detach() < sensor_d_fs - 0.05)
                    )
                    # Beyond-range carving (escape-plume fix): pixels with INVALID
                    # sensor depth (beyond the ToF range — doorways, windows, dark
                    # surfaces) were a permanent free-fire zone: RGB-only loss, no
                    # carving. Live opaque mass escaped there and MCMC growth
                    # amplified it (kitchen1: 55% of live mass outside the room by
                    # 60k). Treat invalid depth as "free space at least to
                    # max_depth": carve alpha rendered SHALLOWER than max_depth on
                    # those rays. Genuine far content (rendered beyond range) is
                    # untouched.
                    #
                    # W3-08: when streaming_ray_semantics_enabled, this legacy
                    # single-frame invalid-depth heuristic is REPLACED by the
                    # multi-view-corroborated CERTIFIED_FREE_SPACE mask
                    # (deliverable 1: a single frame's own invalid-depth pixels
                    # can never certify free space alone). streaming_carve_
                    # invalid_depth stays available standalone as an explicit
                    # legacy ablation (deliverable 3); if both are set, ray
                    # semantics takes precedence (logged once).
                    if _rs_enabled and _rs_classification is not None:
                        if getattr(args, "streaming_carve_invalid_depth", False):
                            global _RS_WARNED_LEGACY_CARVE_COMBO_INLINE
                            if not _RS_WARNED_LEGACY_CARVE_COMBO_INLINE:
                                print(
                                    "[W3-08] streaming_ray_semantics_enabled=True AND "
                                    "streaming_carve_invalid_depth=True are both set; ray "
                                    "semantics takes precedence, legacy carving is skipped "
                                    "for this run (they are not meant to run combined).",
                                    flush=True,
                                )
                                _RS_WARNED_LEGACY_CARVE_COMBO_INLINE = True
                        # No getattr fallback: streaming_max_depth is an
                        # unconditional StreamingParams default, always present
                        # on the real parsed args (issue #368 -- old
                        # getattr(..., 5.0) fallback drifted from the real
                        # default of 8.0; removed rather than re-stated).
                        _cv_max_d = float(args.streaming_max_depth)
                        _certified = (_rs_classification.kind_map == 1).unsqueeze(0)  # CERTIFIED_FREE_SPACE
                        _cv = (
                            _certified
                            & (rend_d_fs.detach() > 0)
                            & (rend_a_fs.detach() > 0.2)
                            & (rend_d_fs.detach() < _cv_max_d - 0.05)
                        )
                        if _cv.any():
                            floater = floater | _cv
                            sensor_d_fs = torch.where(_cv, torch.full_like(sensor_d_fs, _cv_max_d), sensor_d_fs)
                    elif getattr(args, "streaming_carve_invalid_depth", False):
                        # No getattr fallback: streaming_max_depth is an
                        # unconditional StreamingParams default, always present
                        # on the real parsed args (issue #368 -- old
                        # getattr(..., 5.0) fallback drifted from the real
                        # default of 8.0; removed rather than re-stated).
                        _cv_max_d = float(args.streaming_max_depth)
                        # Only carve LARGE contiguous invalid regions (doorways,
                        # windows — genuinely beyond range). IR-absorbing/glossy
                        # NEAR objects (black coffee machine, soap bottle) also
                        # return invalid depth but sit inside valid surroundings;
                        # carving them erases real geometry. Erode the invalid
                        # mask: a pixel carves only if its kxk neighborhood is
                        # entirely invalid.
                        _cv_k = int(getattr(args, "streaming_carve_erode_px", 15))
                        _inv = (sensor_d_fs <= 0).float().unsqueeze(0)
                        _inv_core = torch.nn.functional.avg_pool2d(
                            _inv, kernel_size=2 * _cv_k + 1, stride=1, padding=_cv_k
                        ).squeeze(0) > 0.999
                        _cv = (
                            _inv_core
                            & (rend_d_fs.detach() > 0)
                            & (rend_a_fs.detach() > 0.2)
                            & (rend_d_fs.detach() < _cv_max_d - 0.05)
                        )
                        if _cv.any():
                            floater = floater | _cv
                            sensor_d_fs = torch.where(_cv, torch.full_like(sensor_d_fs, _cv_max_d), sensor_d_fs)
                    # Keep valid-subset masking (see depth-loss note above).
                    if floater.any():
                        free_err = rend_a_fs[floater] * (sensor_d_fs[floater] - rend_d_fs[floater]).detach()
                        if _surface_prior_fs is not None:
                            conf = _surface_prior_fs.confidence.to("cuda", non_blocking=True)[floater].clamp_min(1e-4)
                            L_free = (free_err * conf).sum() / conf.sum()
                        else:
                            L_free = free_err.mean()
                        # Same rendered-depth/alpha NaN risk as the depth term above.
                        if not bool(torch.isfinite(L_free)):
                            _nan_term_report("free_space_loss", iteration)
                            L_free = torch.zeros((), device=L_free.device)
                        loss = loss + free_space_weight * L_free
                        if tb_writer is not None and iteration % scalar_log_interval == 0:
                            _free_loss_val = L_free.item()

            # ---- Appearance-only auxiliary update (Wave 3, PR W3-08, 13.5) --
            # Independent of the depth/geometry loss above (deliverable 6).
            _ao_interval = max(1, int(getattr(args, "streaming_appearance_only_interval", 20)))
            if (
                _ao_enabled
                and _rs_classification is not None
                and _rs_sensor_d_hw is not None
                and background is not None
                and iteration % _ao_interval == 0
            ):
                _ao_weight = float(getattr(args, "streaming_appearance_only_weight", 0.0))
                if _ao_weight > 0:
                    _reliability = geometry_reliability_map(
                        _rs_classification.kind_map, _rs_sensor_d_hw, gt_image.detach(),
                        pose_confidence=float(getattr(viewpoint_cam, "pose_confidence", 1.0) or 1.0),
                    )
                    _risk_mask = material_risk_mask(
                        _rs_classification.kind_map, _reliability,
                        reliability_thresh=float(getattr(args, "streaming_appearance_only_reliability_thresh", 0.35)),
                    )
                    if bool(_risk_mask.any()):
                        from eligsir.mapping.renderer.gsplat_backend import render as _gsplat_render
                        L_ao = run_appearance_only_update(
                            _gsplat_render, gaussians, viewpoint_cam, pipe, background,
                            gt_image, _risk_mask, weight=_ao_weight,
                        )
                        if L_ao is not None and bool(torch.isfinite(L_ao)):
                            loss = loss + L_ao
                            if tb_writer is not None and iteration % scalar_log_interval == 0:
                                tb_writer.add_scalar("streaming/appearance_only/risk_pixel_frac", float(_risk_mask.float().mean()), iteration)

            _surface_loss_result = None
            if bool(getattr(args, "surface_loss_enabled", False)):
                _surface_loss_result = compute_surface_anchor_losses(
                    gaussians,
                    min_confidence=float(getattr(args, "surface_loss_min_confidence", 0.1)),
                    normal_weight=float(getattr(args, "surface_anchor_normal_weight", 0.0)),
                    tangent_weight=float(getattr(args, "surface_anchor_tangent_weight", 0.0)),
                    normal_align_weight=float(getattr(args, "surface_normal_align_weight", 0.0)),
                    thinness_weight=float(getattr(args, "surface_thinness_weight", 0.0)),
                )
                loss = _add_guarded(loss, _surface_loss_result.total, "surface", iteration)
                if tb_writer is not None and iteration % scalar_log_interval == 0:
                    for _k, _v in _surface_loss_result.to_tensorboard_dict().items():
                        tb_writer.add_scalar(_k, _v, iteration)

            # ---- Hot-path KNN scale-smoothness regularizer (amortized; flag-gated) --
            # Applies in both energy and non-energy paths; no-op when --knn_reg_weight=0.
            _knn_reg = knn_smoothness_reg(gaussians, args, iteration)
            if not isinstance(_knn_reg, float):
                loss = _add_guarded(loss, _knn_reg, "knn_reg", iteration)
                if tb_writer is not None and iteration % scalar_log_interval == 0:
                    tb_writer.add_scalar("knn_reg/scale_smoothness", float(_knn_reg.detach()), iteration)

            # ---- Stratified sampling state update --------------------------------
            # In placement_only mode there is no backward, so reading loss here costs
            # nothing extra. In the normal path we DEFER the loss.item() host-sync to
            # AFTER loss.backward() (below) so it no longer serializes forward->backward.
            if _placement_only:
                with _timed("m_loss_item"):
                    _loss_val = loss.item()
                streaming_scene.update_stratified_state(viewpoint_cam, _loss_val)

            # ---- Non-finite loss guard ------------------------------------------
            # A single NaN/Inf loss (seen late on long-travel floor3 runs, ~iter
            # 11.8k, data/numerically triggered — not a fixed schedule event) would
            # otherwise propagate through backward + optimizer.step and corrupt the
            # whole model irrecoverably. Skip this iteration's backward + step when
            # the loss is non-finite, so one bad frame can't brick a live run. Log a
            # term-level breakdown the first few times so the root cause stays
            # visible instead of being silently masked.
            _loss_finite = bool(torch.isfinite(loss))
            # W3-14C round-3 bisection instrumentation: closes the bracket opened
            # right after render() above, at the same logical point
            # objective.assemble_loss() finishes computing LossBreakdown.loss_finite.
            _w3_14c_loss_bracket.__exit__(None, None, None)
            if not _loss_finite:
                _nan_skip_count = locals().get("_nan_skip_count", 0) + 1
                if _nan_skip_count <= 5:
                    # Report only terms that actually exist this iter. energy_total is
                    # meaningful ONLY when the energy path is enabled; otherwise the
                    # non-energy reg/depth/free terms are the relevant ones. (An earlier
                    # hardcoded energy_total=nan fallback was a misleading red herring.)
                    _e_txt = ""
                    if _energy_loss_enabled_main:
                        try:
                            _e_txt = f" energy_total={float(_p5_result.total.detach()):.4g}"
                        except Exception:
                            _e_txt = " energy_total=<err>"
                    _vp_id = getattr(viewpoint_cam, "image_name", getattr(viewpoint_cam, "uid", "?"))
                    _dv = locals().get("_depth_loss_val", None)
                    _fv = locals().get("_free_loss_val", None)
                    print(
                        f"[nan-guard] step={iteration} SKIPPING step — non-finite loss."
                        f" Ll1={float(Ll1.detach()):.4g} ssim={float(_ssim.detach()):.4g}{_e_txt}"
                        f" depth_val={_dv} free_val={_fv}"
                        f" color_finite={bool(torch.isfinite(image).all())}"
                        f" gt_finite={bool(torch.isfinite(gt_image).all())}"
                        f" N={gaussians.get_xyz.shape[0]} view={_vp_id}",
                        flush=True,
                    )
                if tb_writer is not None:
                    tb_writer.add_scalar("streaming/nan_skip", 1.0, iteration)
                gaussians.optimizer.zero_grad(set_to_none=True)

            # ---- Backward + optimizer step (skipped in placement_only mode) -----
            visible = render_pkg["visibility_filter"].detach().to(dtype=torch.bool).contiguous()
            # Update visibility EMA so trust-region noise and energy terms have valid support
            if hasattr(gaussians, "visibility_ema") and gaussians.visibility_ema.numel() == visible.shape[0]:
                from eligsir.utils.geometry_metrics import update_visibility_ema
                update_visibility_ema(gaussians, visible)
            _step_mask = visible  # default; overridden below when sparse_active_set
            _energy_grad_snapshot = None  # set just before zero_grad when energy MCMC is on
            if not _placement_only and _loss_finite:
                _before_optimizer_bootstrap = (
                    _capture_bootstrap_positions(gaussians) if _debug_bootstrap_this_iter else None
                )
                with _timed("m_backward"):
                    _coordinator.optimization_engine_for(gaussians).render_and_backward(
                        gaussians, args, iteration, mcmc_strategy, render_pkg, loss,
                    )
                _backward_ran = True
                # Issue #703: supervised pixels only for this view (inline path only).
                # Read H*W AFTER multires rebind (line ~6114) so we get the real size trained on.
                # Incremented only when backward ran (loss was finite).
                _clk_supervised_pixel_count += (
                    int(viewpoint_cam.image_height) * int(viewpoint_cam.image_width)
                )
                _record_successful_supervision(
                    _supervision_source_index(viewpoint_cam)
                )
                _memprobe("after_backward", iteration, f"N={gaussians.get_xyz.shape[0]}")

                # Stratified sampling state update — run AFTER backward so the
                # loss.item() host-sync overlaps the optimizer prep instead of
                # serializing forward->backward. The loss scalar is still valid.
                with _timed("m_loss_item"):
                    _loss_val = loss.item()
                streaming_scene.update_stratified_state(viewpoint_cam, _loss_val)

                # W3-14C round-3 bisection instrumentation: brackets this inline
                # path's post-backward gradient-masking cascade (active-set zero,
                # H7 freeze-old-geometry, lifecycle/bubble freeze mask,
                # bootstrap-anchor and insertion-anchor masking, bubble
                # freeze-snapshot capture) so it is directly comparable to the
                # engine path's "m_grad_policies" bracket around
                # run_gradient_policies() (see engine.py's run_post_backward_step)
                # -- the engine-only dispatch chain this inline block has no
                # counterpart object/method-call structure for, but performs the
                # same masking work inline. See
                # docs/architecture/w3_02_completion.md's 2026-08-09 round-3
                # section. Manual __enter__/__exit__ so this span's existing
                # indentation is untouched.
                _w3_14c_grad_mask_bracket = _timed("m_grad_masking")
                _w3_14c_grad_mask_bracket.__enter__()

                # Zero invisible grad rows in strided grads (active-set safety)
                if sparse_active_set and getattr(args, "selective_adam_zero_invisible_grads", True):
                    # Use the same active mask as the regularizer (visible | provisional)
                    # so provisional Gaussians that aren't visible keep their gradients.
                    _step_mask = visible | gaussians.provisional.detach()
                    for group in gaussians.optimizer.param_groups:
                        p = group["params"][0]
                        if p.grad is not None and getattr(p.grad, "layout", torch.strided) == torch.strided:
                            p.grad[~_step_mask] = 0.0

                # H7: freeze confirmed old geometry — zero geometry gradients for splats
                # that are old enough not to be "young" and are not provisional.
                # Prevents optimizer from dragging confirmed good splats to explain new views.
                # streaming_freeze_new_frame_steps: for this many steps after a new frame,
                # treat ALL non-provisional splats as "old" (stricter than frame-age test).
                _iter_since_new_frame += 1
                if _freeze_old and hasattr(gaussians, "birth_frame") and hasattr(gaussians, "provisional"):
                    with torch.no_grad():
                        _frame_age = n_frames_ingested - gaussians.birth_frame.long()
                        _strict_new_frame_phase = (_freeze_new_frame_steps > 0 and
                                                   _iter_since_new_frame < _freeze_new_frame_steps)
                        if _strict_new_frame_phase:
                            _young = gaussians.provisional
                        else:
                            _young = gaussians.provisional | (_frame_age <= _young_age_frames)
                        _old = ~_young
                        # Unconditional masked zero (no .any() host-sync; no-op when empty).
                        for _param in gaussians.geometry_params:
                            if (_param.grad is not None
                                    and getattr(_param.grad, "layout", torch.strided) == torch.strided):
                                _param.grad[_old] = 0.0

                # Lifecycle: zero all gradients for FROZEN Gaussians (lifecycle_state==3).
                # Spatial active bubble (Component 1): ALSO zero gradients for OUT-OF-BUBBLE
                # Gaussians so travel-away geometry never updates (no forgetting) — the
                # freeze mask is (frozen | out-of-bubble), recomputed from current xyz.
                _freeze_grad_mask = None
                if _lifecycle_enabled and hasattr(gaussians, "lifecycle_state"):
                    _freeze_grad_mask = gaussians.lifecycle_state == 3
                if getattr(args, "streaming_spatial_bubble_enabled", False):
                    from eligsir.utils.strategies.mcmc_strategy import _bubble_active_mask as _bam
                    _bub = _bam(gaussians.get_xyz, args)
                    if _bub is not None:
                        _oob = ~_bub
                        _freeze_grad_mask = _oob if _freeze_grad_mask is None else (_freeze_grad_mask | _oob)
                if _freeze_grad_mask is not None:
                    # Unconditional masked zero (no .any() host-sync; no-op when empty).
                    with torch.no_grad():
                        for _param in gaussians.all_learnable_params:
                            if (_param.grad is not None
                                    and getattr(_param.grad, "layout", torch.strided) == torch.strided):
                                _param.grad[_freeze_grad_mask] = 0.0

                # Bootstrap means gradient masking: suppress center drift of depth-seeded
                # Gaussians (birth_frame==0) during early training. The fraction of gradients
                # zeroed decays linearly from 1 → 0 over bootstrap_anchor_decay_steps so
                # the constraint softens gradually rather than cutting off hard.
                if (_bootstrap_anchor_weight > 0 and iteration < _bootstrap_anchor_decay
                        and hasattr(gaussians, "birth_frame")):
                    with torch.no_grad():
                        _boot_m = (gaussians.birth_frame == 0)
                        # retain_frac ramps 0→1 (python; no sync); at iter=0 zero all bootstrap
                        # gradients, at iter=decay keep all.
                        _retain_frac = iteration / _bootstrap_anchor_decay
                        if _boot_m.shape[0] == gaussians.get_xyz.shape[0] and _retain_frac < 1.0:
                            if getattr(args, "streaming_anchor_fullN_draw", True):
                                # Full-N sync-free draw: each splat kept i.i.d. with prob
                                # retain_frac. Statistically identical to the masked-count
                                # draw, but uses NO device→host sync (no .sum()-as-size
                                # `torch.rand` arg, no .any() guard) — the launch-bound loop
                                # is sync-sensitive. Densify is gated on the python flags
                                # above (not a GPU .any()); an all-False mask is a no-op write.
                                _dev = gaussians.get_xyz.device
                                _keep = torch.rand(_boot_m.shape[0], device=_dev) < _retain_frac
                                _zero_boot = _boot_m & ~_keep
                                _p_means = gaussians.means_param
                                if _p_means.grad is not None:
                                    if getattr(_p_means.grad, "layout", torch.strided) != torch.strided:
                                        _p_means.grad = _p_means.grad.to_dense().contiguous()
                                    _p_means.grad[_zero_boot] = 0.0
                            elif _boot_m.any():
                                # Original masked-count path (kept for --no-streaming_anchor_fullN_draw
                                # A/B; carries the .sum()-as-size + .any() device→host syncs).
                                _rand = torch.rand(_boot_m.sum(), device=gaussians.get_xyz.device)
                                _zero_boot = _boot_m.clone()
                                _zero_boot[_boot_m] = _rand >= _retain_frac
                                _p_means = gaussians.means_param
                                if _p_means.grad is not None:
                                    if getattr(_p_means.grad, "layout", torch.strided) != torch.strided:
                                        _p_means.grad = _p_means.grad.to_dense().contiguous()
                                    _p_means.grad[_zero_boot] = 0.0

                # Insertion means gradient masking: per-Gaussian stochastic suppression
                # for depth-inserted Gaussians (birth_frame>0) during their first
                # insertion_anchor_decay_steps iterations after anchor_iter.
                # Each Gaussian's retain probability ramps 0→1 independently based on
                # its own age, giving newly added geometry time to settle before the
                # optimizer is free to slide it under photometric gradient.
                # The bootstrap block above may have already densified means.grad.
                if (_insertion_anchor_weight > 0
                        and hasattr(gaussians, "birth_frame")
                        and hasattr(gaussians, "anchor_iter")):
                    with torch.no_grad():
                        _ins_m = (gaussians.birth_frame > 0)
                        if _ins_m.shape[0] == gaussians.get_xyz.shape[0]:
                            _age_all = (iteration - gaussians.anchor_iter.long()).clamp(min=0)
                            _within = _ins_m & (_age_all < _insertion_anchor_decay)
                            if getattr(args, "streaming_anchor_fullN_draw", True):
                                # Full-N sync-free: per-Gaussian retain prob ramps with age;
                                # each within-window inserted splat kept i.i.d. with that prob.
                                # No .any()/boolean-indexed-size syncs. Statistically identical
                                # to the original masked-count draw.
                                _dev = gaussians.get_xyz.device
                                _retain = (_age_all.float() / _insertion_anchor_decay).clamp(0.0, 1.0)
                                _keep = torch.rand(_ins_m.shape[0], device=_dev) < _retain
                                _zero_ins = _within & ~_keep
                                _p_means = gaussians.means_param
                                if _p_means.grad is not None:
                                    if getattr(_p_means.grad, "layout", torch.strided) != torch.strided:
                                        _p_means.grad = _p_means.grad.to_dense().contiguous()
                                    _p_means.grad[_zero_ins] = 0.0
                            elif _ins_m.any() and _within.any():
                                # Original boolean-indexed path (--no-streaming_anchor_fullN_draw
                                # A/B; carries the two .any() device→host syncs).
                                _age_young = _age_all[_within].float()
                                _retain_per = (_age_young / _insertion_anchor_decay).clamp(0.0, 1.0)
                                _rand_ins = torch.rand(
                                    _retain_per.shape[0], device=gaussians.get_xyz.device
                                )
                                _zero_ins = _within.clone()
                                _zero_ins[_within] = _rand_ins >= _retain_per
                                _p_means = gaussians.means_param
                                if _p_means.grad is not None:
                                    if getattr(_p_means.grad, "layout", torch.strided) != torch.strided:
                                        _p_means.grad = _p_means.grad.to_dense().contiguous()
                                    _p_means.grad[_zero_ins] = 0.0

                _depth_locked_step_mask = None
                if _depth_lock_active:
                    with torch.no_grad():
                        _depth_locked_step_mask = depth_locked_mask(gaussians, args)
                    if _depth_locked_step_mask.any():
                        zero_depth_locked_mean_grads(
                            gaussians,
                            args,
                            mask=_depth_locked_step_mask,
                        )

                # Spatial active bubble (Component 1): a TRUE freeze of out-of-bubble
                # Gaussians. Zeroing gradients is insufficient — Adam momentum
                # (exp_avg / exp_avg_sq) and weight decay still move a zero-grad param.
                # Snapshot the frozen rows of every learnable param BEFORE the step and
                # restore them AFTER (mirrors restore_depth_locked_centers), so frozen
                # geometry is bit-for-bit unchanged regardless of optimizer state. No-op
                # when the bubble is off. Uses the live-position mask so a returning
                # region reactivates automatically.
                _bubble_freeze_snap = None
                _bubble_freeze_mask = None
                if getattr(args, "streaming_spatial_bubble_enabled", False):
                    from eligsir.utils.strategies.mcmc_strategy import _bubble_active_mask as _bam2
                    _bub2 = _bam2(gaussians.get_xyz, args)
                    if _bub2 is not None and bool((~_bub2).any()):
                        _bubble_freeze_mask = ~_bub2
                        with torch.no_grad():
                            _bubble_freeze_snap = [
                                (p, p.data[_bubble_freeze_mask].clone())
                                for p in gaussians.all_learnable_params
                                if p is not None and p.data.shape[0] == _bubble_freeze_mask.shape[0]
                            ]

                # Issue #326 hook A -- optimizer bracket (see UNATTRIBUTABLE
                # 'optimizer/move_xyz': the fused kernel's proposal is not
                # observable from Python, so only a before/after bracket is real).
                _optrace_snap = None
                if _optrace is not None and _optrace.traces_iteration(iteration):
                    from eligsir.diagnostics import gaussian_op_probe as _gop
                    _optrace.current_frame = int(n_frames_ingested)
                    _optrace.set_gate_state(
                        allow_insertion=(None if _decision is None else bool(_decision.allow_insertion)),
                        allow_growth=(None if _decision is None else bool(_decision.allow_growth)),
                    )
                    _optrace_snap, _optrace_uid_real = _gop.snapshot(gaussians, _optrace.uid_stride)
                    if not _optrace_uid_real:
                        _optrace.set_gate_state(
                            **dict(_optrace.gate_state, uid_keys_are_row_indices=True)
                        )

                _w3_14c_grad_mask_bracket.__exit__(None, None, None)
                if iteration < _optimizer_iteration_limit:
                    from eligsir.utils.seed_lineage_freeze import seed_lineage_optimizer_guard
                    with seed_lineage_optimizer_guard(
                        gaussians,
                        args,
                        backend=map_backend,
                        iteration=iteration,
                    ):
                        if optimizer_type == "selective_adam":
                            gaussians.prepare_selective_adam_step(
                                allow_dense_grads=getattr(args, "selective_adam_allow_dense_grads", False)
                            )
                            with _timed("m_optstep"):
                                gaussians.optimizer.step(visibility=_step_mask if sparse_active_set else visible)
                            # PR2 cadence clocks: one real optimizer step.
                            _clk_optimizer_step_id += 1
                            _cadence.record_optimizer_step(iteration)   # #289 denominator clock
                        else:
                            with _timed("m_optstep"):
                                gaussians.optimizer.step()
                            # PR2 cadence clocks: one real optimizer step.
                            _clk_optimizer_step_id += 1
                            _cadence.record_optimizer_step(iteration)   # #289 denominator clock
                        _memprobe("after_optstep", iteration, f"N={gaussians.get_xyz.shape[0]}")
                        # Restore the frozen (out-of-bubble) rows bit-for-bit.
                        if _bubble_freeze_snap is not None:
                            with torch.no_grad():
                                for p, saved in _bubble_freeze_snap:
                                    if p.data.shape[0] == _bubble_freeze_mask.shape[0]:
                                        p.data[_bubble_freeze_mask] = saved
                        if pipe.gsplat_sparse_grad:
                            gaussians.normalize_rotation_params(
                                mask=_step_mask if sparse_active_set else None
                            )
                        if _optrace is not None and _optrace_snap is not None:
                            from eligsir.diagnostics import gaussian_op_probe as _gop
                            _snap_post_opt, _ = _gop.snapshot(gaussians, _optrace.uid_stride)
                            _gop.emit_uid_deltas(
                                _optrace, before=_optrace_snap, after=_snap_post_opt,
                                iteration=iteration, op="move_xyz", mechanism="optimizer",
                                phase="bracket",
                                topology_version=int(getattr(map_backend, "topology_version", 0)),
                                evidence={"bracket": "optimizer.step()+bubble_restore"},
                            )
                            _optrace_snap = _snap_post_opt

                        if _depth_lock_active:
                            restore_depth_locked_centers(
                                gaussians,
                                args,
                                mask=_depth_locked_step_mask,
                            )
                            # Issue #326 hook B -- the cancellation half: whatever the
                            # optimizer just moved, depth-lock may have put back.
                            if _optrace is not None and _optrace_snap is not None:
                                from eligsir.diagnostics import gaussian_op_probe as _gop
                                _snap_post_lock, _ = _gop.snapshot(gaussians, _optrace.uid_stride)
                                _gop.emit_uid_deltas(
                                    _optrace, before=_optrace_snap, after=_snap_post_lock,
                                    iteration=iteration, op="move_xyz",
                                    mechanism="anchors_depth_locks", phase="cancelled",
                                    topology_version=int(getattr(map_backend, "topology_version", 0)),
                                    evidence={"reason": "restore_depth_locked_centers(post_optimizer)"},
                                )
                                _optrace_snap = _snap_post_lock
                    if _debug_bootstrap_this_iter:
                        _log_bootstrap_motion(
                            tb_writer,
                            args,
                            iteration,
                            "optimizer",
                            _bootstrap_motion_stats(
                                gaussians,
                                before=_before_optimizer_bootstrap,
                                threshold=_debug_bootstrap_threshold,
                            ),
                        )
                    _energy_grad_snapshot = (
                        _snapshot_energy_grads(gaussians) if use_energy_mcmc else None
                    )
                    gaussians.optimizer.zero_grad(set_to_none=True)
                    # Producer check #1: did the optimizer step just write a non-finite
                    # (or exp-overflowing) parameter? Reports + neutralizes at source.
                    # W3-61: branch-free core every call, sampled diagnostics.
                    with _timed("m_sanitize"):
                        _sanitize_with_seed_freeze(
                            "after_optstep", iteration, diag_interval=scalar_log_interval
                        )

                    # MCMC noise injection — skipped in placement_only / colors_only modes
                    if not _skip_mcmc:
                        if densification_strategy in {
                            "mcmc", "hybrid", "gsplat_mcmc", "gsplat_energy_mcmc", "gsplat_live_strategy",
                        }:
                            _before_noise_bootstrap = (
                                _capture_bootstrap_positions(gaussians) if _debug_bootstrap_this_iter else None
                            )
                            if streaming_mcmc_local and sparse_active_set:
                                with _timed("m_noise"):
                                    mcmc_strategy.inject_noise(
                                        gaussians=gaussians, args=args, xyz_lr=xyz_lr,
                                        visible=visible, sparse_active_set=True, iteration=iteration,
                                        tb_writer=tb_writer,
                                    )
                            else:
                                with _timed("m_noise"):
                                    mcmc_strategy.inject_noise(
                                        gaussians=gaussians, args=args, xyz_lr=xyz_lr,
                                        visible=None, sparse_active_set=False, iteration=iteration,
                                        tb_writer=tb_writer,
                                    )
                            if _debug_bootstrap_this_iter:
                                _log_bootstrap_motion(
                                    tb_writer,
                                    args,
                                    iteration,
                                    "mcmc_noise",
                                    _bootstrap_motion_stats(
                                        gaussians,
                                        before=_before_noise_bootstrap,
                                        threshold=_debug_bootstrap_threshold,
                                    ),
                                )
                            # Issue #326 hook C -- MCMC noise APPLIED delta.
                            if _optrace is not None and _optrace_snap is not None:
                                from eligsir.diagnostics import gaussian_op_probe as _gop
                                _snap_post_noise, _ = _gop.snapshot(gaussians, _optrace.uid_stride)
                                _gop.emit_uid_deltas(
                                    _optrace, before=_optrace_snap, after=_snap_post_noise,
                                    iteration=iteration, op="move_xyz", mechanism="mcmc",
                                    phase="applied",
                                    topology_version=int(getattr(map_backend, "topology_version", 0)),
                                    evidence={"source": "mcmc_strategy.inject_noise",
                                              "local_only": bool(streaming_mcmc_local and sparse_active_set)},
                                )
                                _optrace_snap = _snap_post_noise
                            if _depth_lock_active:
                                restore_depth_locked_centers(gaussians, args)
                                # Issue #326 hook D -- depth-lock cancelling MCMC noise.
                                if _optrace is not None and _optrace_snap is not None:
                                    from eligsir.diagnostics import gaussian_op_probe as _gop
                                    _snap_post_lock2, _ = _gop.snapshot(gaussians, _optrace.uid_stride)
                                    _gop.emit_uid_deltas(
                                        _optrace, before=_optrace_snap, after=_snap_post_lock2,
                                        iteration=iteration, op="move_xyz",
                                        mechanism="anchors_depth_locks", phase="cancelled",
                                        topology_version=int(getattr(map_backend, "topology_version", 0)),
                                        evidence={"reason": "restore_depth_locked_centers(post_mcmc_noise)"},
                                    )
                                    _optrace_snap = _snap_post_lock2
                    # Producer check #2: did MCMC noise injection write a bad parameter?
                    # W3-61: branch-free core every call, sampled diagnostics.
                    with _timed("m_sanitize"):
                        _sanitize_with_seed_freeze(
                            "after_mcmc_noise", iteration, diag_interval=scalar_log_interval
                        )
                    # W3-14C round-3 fix (coordinator-directed metric audit,
                    # 2026-08-09): --w3_02_evidence_manifest's `.end()` used to be
                    # called right after zero_grad(), BEFORE this arm's own MCMC
                    # noise injection (m_noise) and both `_sanitize_gaussian_params`
                    # hook calls. The engine arm's `.end()` (train_streaming.py,
                    # engine branch) is called only after
                    # run_streaming_optimization_step() returns, which -- per
                    # engine.py's run_step -- includes MCMC noise injection AND
                    # both equivalent post_optimizer_hook/post_mcmc_hook calls
                    # internally. So the two arms' `optimization_gpu_ms` were
                    # measuring spans of different width: the engine arm's number
                    # included strictly more real work than the inline arm's,
                    # inflating the apparent regression independent of any actual
                    # performance difference. Moving `.end()` here makes both arms
                    # measure the identical logical span (through MCMC noise +
                    # both sanitize hooks). See
                    # docs/architecture/w3_02_completion.md's 2026-08-09 round-3
                    # section.
                    if _w3_02_evidence is not None:
                        _w3_02_evidence.end()

        # ---- Issue #399 / #879: GPU memory residency snapshot ------------------
        # Deliberately NOT gated on tb_writer: report.json["gpu_memory"]
        # must be populated even for a run with TensorBoard disabled.
        # Metadata-only walk; introduces no host/device synchronization
        # (see utils/gpu_memory_telemetry's module docstring).
        # Issue #879: hoisted out of the inline `else:` arm above (where it
        # was originally added for #399) to just after the engine/inline
        # if/else rejoins, so BOTH arms reach this call once per sampled
        # iteration. All of its inputs (iteration, n_frames_ingested,
        # gaussians, streaming_scene, _observation_db, tb_writer,
        # _mem_tracker, _mem_interval) are set once per outer loop iteration
        # before the if/else and are not reassigned differently by either
        # arm above (see the `if _streaming_optimization_runtime is not None:`
        # / `else:` split a few hundred lines up).
        if _mem_tracker is not None and iteration % _mem_interval == 0:
            try:
                _mem_sample = _mem_tracker.snapshot(
                    iteration=iteration,
                    frames_ingested=n_frames_ingested,
                    gaussians=gaussians,
                    streaming_scene=streaming_scene,
                    observation_db=_observation_db,
                )
                if tb_writer is not None:
                    for _mk, _mv in _mem_tracker.tb_scalars(_mem_sample).items():
                        tb_writer.add_scalar(_mk, _mv, iteration)
            except Exception as _mem_exc:  # never let telemetry kill a run
                print(f"[memory-telemetry] snapshot failed: {_mem_exc}", flush=True)
                _mem_tracker = None

        # Host resource monitor: cheap per-iteration tick (a lock + int
        # write on the background thread's shared state) so each sample the
        # poller writes carries the training iteration it landed near, for
        # correlation. No-op when disabled.
        if _resource_monitor is not None:
            _resource_monitor.set_iteration(iteration)

        # ---- Issue #882: per-iteration anchor/bubble telemetry (hoisted from inline arm) ---
        # Hoisted out of the inline `else:` arm (where it lived at lines 7382-7413
        # before this fix) to just after the engine/inline if/else rejoins, so BOTH
        # arms reach this block once per sampled iteration. All inputs (tb_writer,
        # iteration, scalar_log_interval, save_worker, gaussians, _bootstrap_anchor_weight,
        # _bootstrap_anchor_decay, _insertion_anchor_weight, _insertion_anchor_decay, args)
        # are set once per outer loop iteration before the if/else and are not reassigned
        # differently by either arm above. Four scalar groups emit here:
        # io/* (save_queue_depth), streaming/bootstrap_anchor/*, streaming/insertion_anchor/*,
        # streaming/bubble/* (8 scalars total that issue #882 addresses).
        # Anchor diagnostics — TB only; actual constraint via gradient masking after backward.
        if tb_writer is not None and iteration % scalar_log_interval == 0:
            if save_worker is not None:
                tb_writer.add_scalar("io/save_queue_depth", save_worker.queue_depth(), iteration)
            if _bootstrap_anchor_weight > 0 and hasattr(gaussians, "anchor_xyz") and hasattr(gaussians, "birth_frame"):
                with torch.no_grad():
                    _bm = (gaussians.birth_frame == 0) if gaussians.birth_frame.shape[0] == gaussians.get_xyz.shape[0] else None
                    if _bm is not None and _bm.any():
                        _dxyz_tb = gaussians.get_xyz[_bm] - gaussians.anchor_xyz[_bm]
                        tb_writer.add_scalar("streaming/bootstrap_anchor/drift_mean_m",
                                             _dxyz_tb.norm(dim=1).mean().item(), iteration)
                _retain = max(0.0, 1.0 - iteration / _bootstrap_anchor_decay)
                tb_writer.add_scalar("streaming/bootstrap_anchor/grad_retain_frac", 1.0 - _retain, iteration)
            if _insertion_anchor_weight > 0 and hasattr(gaussians, "birth_frame") and hasattr(gaussians, "anchor_iter"):
                with torch.no_grad():
                    _age_tb = (iteration - gaussians.anchor_iter.long()).clamp(min=0)
                    _n_active = int(((gaussians.birth_frame > 0) & (_age_tb < _insertion_anchor_decay)).sum())
                tb_writer.add_scalar("streaming/insertion_anchor/n_active", _n_active, iteration)
            # Spatial active bubble telemetry — so the freeze is VISIBLE (the gap that
            # caused the first blind run). Logs active/frozen counts + geometry so we
            # can confirm active_frac is sane (not ~0 = froze everything, not ~1 = froze
            # nothing) as the robot travels.
            if getattr(args, "streaming_spatial_bubble_enabled", False):
                from eligsir.utils.strategies.mcmc_strategy import _bubble_active_mask as _bam_tb
                _bmask = _bam_tb(gaussians.get_xyz, args)
                if _bmask is not None:
                    with torch.no_grad():
                        _n_act = int(_bmask.sum()); _n_tot = int(_bmask.shape[0])
                        tb_writer.add_scalar("streaming/bubble/n_active", _n_act, iteration)
                        tb_writer.add_scalar("streaming/bubble/n_frozen", _n_tot - _n_act, iteration)
                        tb_writer.add_scalar("streaming/bubble/active_frac", _n_act / max(1, _n_tot), iteration)
                        tb_writer.add_scalar("streaming/bubble/d_obs_m", float(getattr(args, "_bubble_d_obs", 0.0) or 0.0), iteration)

        # ---- Utility for energy MCMC --------------------------------------
        if use_energy_mcmc:
            from eligsir.utils.energy_mcmc import compute_gaussian_utility
            # Depth-error-gated temperature signal (simplified per-frame scalar —
            # see compute_gaussian_utility docstring). Skip computing the residual
            # entirely when the weight is 0.0 so the default path is a true no-op.
            _w_depth_residual = float(getattr(args, "energy_w_depth_residual", 0.0))
            _depth_residual_scalar = 0.0
            if _w_depth_residual != 0.0 and "rendered_depth" in render_pkg:
                _dr_sensor = _get_sensor_depth(viewpoint_cam, image.shape[1], image.shape[2])
                if _dr_sensor is not None:
                    _dr_sensor = _dr_sensor.to(device="cuda", non_blocking=True)
                    _dr_rend = render_pkg["rendered_depth"].detach()
                    _dr_valid = (_dr_sensor > 0) & (_dr_rend > 0)
                    if _dr_valid.any():
                        _depth_residual_scalar = float(
                            (_dr_rend[_dr_valid] - _dr_sensor[_dr_valid]).abs().mean()
                        )
            utility = compute_gaussian_utility(
                gaussians=gaussians, render_pkg=render_pkg, iteration=iteration,
                grad_overrides=_energy_grad_snapshot,
                w_alpha=args.energy_w_alpha, w_vis=args.energy_w_vis,
                w_grad=args.energy_w_grad, w_scale=args.energy_w_scale,
                w_dead=args.energy_w_dead,
                w_support=getattr(args, "energy_w_support", 2.0),
                beta_opacity=getattr(args, "energy_beta_opacity", 1.0),
                beta_scale=getattr(args, "energy_beta_scale", 0.5),
                # No getattr fallback: energy_alpha_dead is an unconditional
                # OptimizationParams default, always present on the real parsed
                # args (issue #368 -- old getattr(..., 0.005) fallback drifted
                # from the real default of 0.05, the same stale value that
                # already caused a confirmed live bug once before the parser
                # default was raised; removed rather than re-stated).
                alpha_dead=args.energy_alpha_dead,
                w_aniso_utility=float(getattr(args, "utility_aniso_weight", 0.1)),
                aniso_soft_limit=float(getattr(args, "utility_aniso_soft_limit", 10.0)),
                args=args,
                w_depth_residual=_w_depth_residual,
                depth_residual_scalar=_depth_residual_scalar,
            )
            # Phase 3: log population energy stats when mode is active
            _want_pop_energy_log = bool(tb_writer) and getattr(args, "energy_mode", "legacy_utility") == "population_energy"
            # W3-06 addition: also build GaussianEnergyTerms (and, from it,
            # evidence) whenever shadow-utility scoring needs it, independent
            # of energy_mode/tb_writer -- otherwise GaussianEvidenceHistory
            # would never warm up under the default gsplat_energy_mcmc mode.
            # any_needed() is False (this branch never taken) unless
            # --streaming_shadow_utility_enabled is set, so this is still a
            # true no-op for every existing run.
            if _want_pop_energy_log or _evidence_requirements.any_needed():
                from eligsir.utils.gaussian_energy_policy import GaussianEnergyPolicy, energy_config_from_args
                # issue #385: see train.py's twin of this site. compute_terms()
                # (the evidence/shadow-utility consumer below) reads only
                # detach_state, so widening this config affects the two
                # energy/population_* TB scalars only -- not the evidence path.
                # No getattr fallback for energy_alpha_dead -- see comment
                # at this same flag's other call site above (issue #368).
                _log_pop_cfg = energy_config_from_args(
                    args,
                    alpha_dead=float(args.energy_alpha_dead),
                    energy_mode="population_energy",
                )
                _log_policy = GaussianEnergyPolicy(_log_pop_cfg)
                _log_terms = _log_policy.compute_terms(gaussians, render_pkg, iteration)
                if _want_pop_energy_log:
                    with torch.no_grad():
                        _log_energy = _log_policy.compute_population_energy(_log_terms)
                    tb_writer.add_scalar("energy/population_mean", _log_energy.population_mean, iteration)
                    tb_writer.add_scalar("energy/population_p95", _log_energy.population_p95, iteration)
                # W3-05 evidence observation: read-only, discarded result.
                # any_needed() is False by default (EvidenceRequirements.none()),
                # so build_gaussian_evidence() returns None before touching any
                # tensor -- this line does not change `_log_energy`/`utility`/
                # `temperature` or anything else the loop reads back from.
                if _evidence_requirements.any_needed():
                    _evidence_builder.build_gaussian_evidence(
                        _log_terms,
                        topology_version=getattr(gaussians, "topology_version", 0),
                        map_version=iteration,
                    )
                    # W3-06: EvidenceSummary.under_supported_mass is the only
                    # correct (real-opacity-backed) source for the utility
                    # formula's under_supported_mass term -- see
                    # ShadowUtilityPolicy.observe_candidate's docstring for why
                    # this isn't recomputed from GaussianEvidence fields alone.
                    _shadow_last_summary = _evidence_builder.build_summary(
                        _log_terms,
                        support_threshold=float(getattr(args, "streaming_shadow_utility_support_threshold", 0.01)),
                    )
                    if _shadow_last_summary is not None:
                        _shadow_utility_policy.last_summary = _shadow_last_summary
            u_temp = min(iteration / 30000.0, 1.0)
            tau_t = max(getattr(args, "energy_temp_tau", 0.4), 1e-6)
            denom_t = 1.0 - math.exp(-1.0 / tau_t)
            temperature = args.energy_temp_min + (args.energy_temp_start - args.energy_temp_min) * (
                1.0 - (1.0 - math.exp(-u_temp / tau_t)) / denom_t
            )
            # Restrict utility to active set in local mode
            if streaming_mcmc_local and sparse_active_set:
                utility[~visible] = (
                    getattr(args, "energy_w_alpha", 1.0) * gaussians.get_opacity.squeeze(-1)[~visible]
                )
        else:
            utility = None
            temperature = 1.0

        # ---- Lifecycle: update utility_ema and advance states ---------------
        if _lifecycle_enabled and iteration % 100 == 0 and hasattr(gaussians, "lifecycle_state"):
            with torch.no_grad():
                # Update per-Gaussian utility EMA using last computed utility score
                if utility is not None and hasattr(gaussians, "utility_ema"):
                    beta = _utility_ema_beta
                    u_score = utility.detach().clamp(min=0.0)
                    gaussians.utility_ema.mul_(beta).add_(u_score, alpha=(1.0 - beta))

                # lifecycle-v2: decay utility for INVISIBLE Gaussians instead of
                # letting the opacity-proxy hold them above the maturity bar. A
                # Gaussian that hasn't been observed recently decays below
                # mature_min_utility and cannot mature on stale/opacity-only
                # evidence — maturity now requires RECENT visual support.
                if getattr(args, "streaming_lifecycle_v2", False) and hasattr(gaussians, "utility_ema"):
                    _v2_vis = None
                    if render_pkg is not None and hasattr(render_pkg, "get"):
                        _v2_vis = render_pkg.get("visibility_filter")
                    if _v2_vis is not None and _v2_vis.shape[0] == gaussians.utility_ema.shape[0]:
                        gaussians.utility_ema[~_v2_vis.bool()] *= float(
                            getattr(args, "streaming_lifecycle_v2_invis_decay", 0.98)
                        )

                lc = gaussians.lifecycle_state  # [N] int8
                frame_age = (n_frames_ingested - gaussians.birth_frame.long()).clamp(min=0)

                # PROVISIONAL (0) → YOUNG (1): existing promote logic already sets provisional=False;
                # keep in sync by upgrading lifecycle_state to YOUNG for newly promoted.
                newly_promoted = (~gaussians.provisional) & (lc == 0)
                if newly_promoted.any():
                    gaussians.lifecycle_state[newly_promoted] = 1  # YOUNG

                # YOUNG (1) → MATURE (2)
                young_mask = lc == 1
                if young_mask.any():
                    age_ok = frame_age >= _mature_age_frames
                    if hasattr(gaussians, "utility_ema") and _mature_min_utility > 0:
                        util_ok = gaussians.utility_ema >= _mature_min_utility
                    else:
                        util_ok = torch.ones_like(young_mask)
                    to_mature = young_mask & age_ok & util_ok
                    # P0.5: multi-view consensus gate — a Gaussian with net depth
                    # conflict (conflict > support) cannot mature, so floaters
                    # from bad depth can't lock in behind the MATURE shield.
                    if getattr(args, "streaming_maturity_consensus", False) and to_mature.any():
                        to_mature = to_mature & consensus_clean_mask(gaussians, args)
                    if to_mature.any():
                        gaussians.lifecycle_state[to_mature] = 2  # MATURE
                        # BUG (found 2026-07-07): re-snapshotting anchor_xyz here at the
                        # CURRENT (possibly already-drifted) pose defeats the purpose of
                        # the restoring-force loss. anchor_xyz is already the correct
                        # birth/depth-insertion position (set once in
                        # add_points_as_gaussians / bootstrap init and never overwritten
                        # elsewhere) -- if a Gaussian drifted during its YOUNG phase
                        # (before it accrues _mature_age_frames), overwriting anchor_xyz
                        # here just locks in that drift as the new "true" position,
                        # so the mature-anchor loss can never pull it back to where it
                        # was actually observed. Default (False) preserves this
                        # historical (buggy) re-snapshot behavior so already-queued
                        # ablation arms using streaming_mature_anchor_*_weight remain
                        # unaffected; --streaming_mature_anchor_use_birth_xyz switches
                        # to anchoring at the true birth/insertion position instead.
                        if not getattr(args, "streaming_mature_anchor_use_birth_xyz", False):
                            if hasattr(gaussians, "anchor_xyz"):
                                gaussians.anchor_xyz[to_mature] = gaussians.get_xyz[to_mature].detach().clone()
                        # Scale/opacity re-snapshot at maturity is left as-is: unlike
                        # position (which should reflect the sensor-observed depth),
                        # scale/opacity legitimately evolve during early optimization
                        # (initial splats are often too small/too transparent), so
                        # locking to the "settled" mature value rather than the
                        # birth value is an intentional, separate design choice.
                        # Reads via the sanctioned raw_scales/raw_opacities
                        # public properties instead of gaussians.params[...]
                        # directly (see docs/architecture/gaussian_ownership_audit.md).
                        if hasattr(gaussians, "anchor_scale_log"):
                            gaussians.anchor_scale_log[to_mature] = gaussians.raw_scales[to_mature].detach().clone()
                        if hasattr(gaussians, "anchor_opacity_logit"):
                            gaussians.anchor_opacity_logit[to_mature] = gaussians.raw_opacities[to_mature].detach().clone()

                # lifecycle-v2: DEMOTION on multi-view conflict. Maturity was
                # monotonic — a mature-but-wrong floater (bad depth) was
                # permanent unless conflict-pruned. When the consensus buffers
                # show net conflict (conflict > support), demote MATURE/FROZEN
                # back to YOUNG so noise/relocation can correct it.
                if getattr(args, "streaming_lifecycle_v2", False):
                    _v2_clean = consensus_clean_mask(gaussians, args)
                    _v2_demote = (lc >= 2) & ~_v2_clean
                    from eligsir.utils.seed_lineage_freeze import exclude_frozen_seed_mask
                    _v2_demote = exclude_frozen_seed_mask(
                        gaussians,
                        args,
                        _v2_demote,
                        iteration=iteration,
                        role="lifecycle_demote",
                    )
                    if _v2_demote.any():
                        gaussians.lifecycle_state[_v2_demote] = 1  # back to YOUNG
                        # Localized opacity reset (MCMC-review recommendation):
                        # demotion alone re-exposes a suspect Gaussian to
                        # noise/relocation but its opacity can still be high
                        # from having been confidently (wrongly) mature — it
                        # keeps rendering as if trustworthy while it re-earns
                        # evidence. Clamp toward a low value so it must
                        # re-accumulate support, mirroring the existing
                        # frame-0 seed clamp pattern (~5838-5847) applied to
                        # this different trigger population.
                        _sor_w = float(getattr(args, "streaming_suspect_opacity_reset", 0.0) or 0.0)
                        if _sor_w > 0:
                            _sor_cap = float(inverse_sigmoid(torch.tensor(_sor_w)))
                            # Was a raw params["opacities"].data read+write;
                            # routed through the mutation transaction instead
                            # (see docs/architecture/gaussian_ownership_audit.md).
                            # Goes through the run's shared map_backend
                            # (Wave 2 backend-ownership pass) so
                            # map_version/topology_version actually advance
                            # rather than the previous throwaway 0/0 transaction.
                            _demote_txn = map_backend.begin_mutation()
                            _clamped = gaussians.raw_opacities[_v2_demote].clamp(max=_sor_cap)
                            _demote_txn.update_opacity_logits(_v2_demote, _clamped)
                            map_backend.commit_mutation(_demote_txn)

                # MATURE (2) → FROZEN (3): only when freeze_age_frames >= 0
                if _freeze_age_frames >= 0:
                    mature_mask = lc == 2
                    if mature_mask.any():
                        to_freeze = mature_mask & (frame_age >= _freeze_age_frames)
                        if to_freeze.any():
                            gaussians.lifecycle_state[to_freeze] = 3  # FROZEN

                if _seed_lineage_freeze is not None:
                    _seed_lineage_freeze.assert_integrity(
                        iteration=iteration, context="lifecycle-v2"
                    )

        # ---- MCMC mutation ------------------------------------------------
        # 2.2 (bubble-as-local-bounds): with streaming_local_stats + bubble on,
        # the schedule's rho = N/cap is computed over the LOCAL population and a
        # LOCAL cap so every bubble runs at the same MCMC operating point
        # regardless of how big the world has grown. cap_local = density x
        # sphere volume, clamped to the remaining global headroom.
        _sched_n = gaussians.get_xyz.shape[0]
        _sched_cap = args.cap_max
        if getattr(args, "streaming_local_stats", False):
            from eligsir.utils.strategies.mcmc_strategy import _bubble_active_mask as _ls_bam
            _ls_bubble = _ls_bam(gaussians.get_xyz, args)
            if _ls_bubble is not None:
                _n_local = int(_ls_bubble.sum())
                _dens = float(getattr(args, "streaming_local_cap_density", 0.0) or 0.0)
                if _dens > 0 and _n_local > 0:
                    _r = float(getattr(args, "streaming_bubble_radius", 5.0))
                    _cap_local = int(_dens * (4.0 / 3.0) * 3.14159265 * _r ** 3)
                    _headroom = int(_sched_cap) - (_sched_n - _n_local) if _sched_cap > 0 else _cap_local
                    _sched_n = _n_local
                    _sched_cap = max(1, min(_cap_local, _headroom))
        sched = _streaming_mcmc_schedule(
            iteration=iteration,
            current_n=_sched_n,
            cap_max=_sched_cap,
            cfg=mcmc_cfg,
            streaming_scene=streaming_scene,
        )

        # In local-only mode, suppress global population growth entirely;
        # new geometry comes from depth insertion instead.
        if streaming_mcmc_local:
            sched["allow_growth"] = False
            # Global maintenance: allow full reloc/grow on configured cadence
            if global_maint_interval > 0 and iteration % global_maint_interval == 0:
                sched["allow_growth"] = True
                sched["allow_relocation"] = True

        # Phase 8: override growth if confidence gate prohibits it
        if _decision is not None and not _decision.allow_growth:
            sched["allow_growth"] = False

        # Warmup: no MCMC growth until enough train cameras have arrived.
        if _warmup_min_frames > 0 and len(streaming_scene.train_cameras) < _warmup_min_frames:
            sched["allow_growth"] = False

        # Observation-staleness growth brake (issue #1044, part 1). OFF BY
        # DEFAULT (`--streaming_growth_staleness_max_iters 0`) -- opt-in only,
        # for genuine-stall investigations. See that flag's default comment in
        # arguments/__init__.py: default-on would suppress growth across the
        # whole refinement tail of every streaming run, re-introducing the
        # 2026-07-18 capacity-starvation regression. Defends
        # against a genuine full stall of new geometry (e.g. depth
        # unavailable / occupancy machinery unwired) by suppressing growth
        # once more than `streaming_growth_staleness_max_iters` iterations
        # have elapsed since the last real depth-insertion commit
        # (`_last_map_commit_iter`, re-armed at both insertion call sites
        # above). Iteration-based, not wall-clock, to stay robust across this
        # project's own large per-hardware speed variance.
        #
        # NOTE (measured directly against the #1044 fr3 repro run,
        # `viser_watch_fr3_full`'s own train.log): this brake does NOT, by
        # itself, explain or prevent that run's pathology. `total_inserted`
        # there climbed steadily and non-trivially throughout (1 -> 47 by
        # iter 21000) -- depth insertion never actually stalled for anywhere
        # near this brake's default threshold. Yet N still ballooned
        # 224,145 -> 253,564 (+29,419) in the FINAL 1,660 iterations alone,
        # while admitted view count V stayed pinned at 20 the entire time.
        # The real mechanism is MCMC's OWN growth schedule (get_mcmc_schedule,
        # purely iteration/cap_max-driven) having no awareness of how many
        # DISTINCT views are actually supervising the scene -- see the
        # admitted-view-density cap immediately below, which is what actually
        # catches this run's pathology. Keeping this brake regardless: it is
        # real, independently-useful defense-in-depth for the genuine-stall
        # case (which this run did not exercise), at zero cost when it never
        # fires.
        # NOTE the getattr fallback is 0 (OFF), matching the argparse default.
        # See that default's own comment in arguments/__init__.py for why this
        # must stay off by default: mapping-end precedes growth-saturation by
        # tens of thousands of iterations, so a default-on staleness brake
        # re-introduces the growth-stop-at-mapping-end capacity starvation that
        # the 2026-07-18 [ADAPTIVE-HORIZON] correction removed (floor3: N ~13k
        # vs ~230k, 19dB vs 22dB).
        _growth_staleness_max_iters = int(getattr(args, "streaming_growth_staleness_max_iters", 0) or 0)
        if _growth_staleness_max_iters > 0:
            _commit_stale_for = iteration - _last_map_commit_iter
            if _commit_stale_for >= _growth_staleness_max_iters:
                if not getattr(streaming_training, "_growth_staleness_warned", False):
                    print(
                        f"[streaming] growth staleness brake: no real map commit for "
                        f"{_commit_stale_for} iterations (>= {_growth_staleness_max_iters}) "
                        f"at step={iteration} N={gaussians.get_xyz.shape[0] if hasattr(gaussians, 'get_xyz') else '?'} "
                        f"-- suppressing MCMC growth until a fresh observation is committed. "
                        f"Set --streaming_growth_staleness_max_iters 0 to disable.",
                        flush=True,
                    )
                    streaming_training._growth_staleness_warned = True
                sched["allow_growth"] = False
            elif getattr(streaming_training, "_growth_staleness_warned", False):
                print(
                    f"[streaming] growth staleness brake: fresh commit landed at step={iteration} "
                    f"-- growth re-enabled.",
                    flush=True,
                )
                streaming_training._growth_staleness_warned = False

        # Admitted-view-density growth brake (issue #1044, part 2 -- the fix
        # that actually catches the observed pathology, see the note above).
        # get_mcmc_schedule()'s growth curve targets cap_max purely as a
        # function of iteration count; it has no notion of how many DISTINCT
        # admitted views are actually supervising the scene. On a real fr3
        # (TUM) --online_mode --streaming_frame_admission hybrid_keyframe run
        # (`viser_watch_fr3_full`), admitted view count (V, ==
        # len(streaming_scene.train_cameras)) plateaued at 20 for the back
        # half of the run while N kept climbing under MCMC's own internal
        # growth/relocation dynamics alone, reaching N=253,564 with
        # frac_mass_outside_bbox=0.66 -- population size had become
        # completely decoupled from the amount of real photometric/geometric
        # grounding available to justify it (a diffuse floater/overgrowth
        # pathology, distinct from the discrete giant-single-Gaussian
        # relocation pathology tracked separately in issue #1043 -- this run's
        # own max Gaussian scale stayed a sane 1.27-1.46m throughout).
        #
        # This brake caps live population against admitted-view count directly:
        # once `current_n >= len(train_cameras) *
        # streaming_growth_max_gaussians_per_view`, growth is suppressed until
        # either the population is pruned back down or another view is
        # admitted (raising the cap). Deliberately does NOT touch relocation/
        # noise (existing budget can still be redistributed for better fit
        # against the current views) and does NOT apply during bootstrap
        # (`len(train_cameras) == 0`, before the warmup gate above even lets
        # growth start) to avoid a spurious divide-by-zero-shaped block on the
        # very first frames. The default (8000/view) is a deliberately
        # generous, defense-in-depth ceiling -- e.g. a 5-view bootstrap alone
        # already implies a 40,000-Gaussian allowance before any additional
        # view is ever admitted -- chosen to catch genuinely unbounded
        # runaway growth like the #1044 case (which crossed roughly 12,000
        # effective Gaussians/view by the time it was killed) without
        # constraining normal, healthy growth on scenes with reasonable
        # admission diversity. Not yet tuned via a real ablation sweep across
        # scenes -- treat the default as a safety ceiling, not a validated
        # optimum, and reduce it for scenes known to admit few, static views.
        # 0 disables (pre-#1044 behavior).
        _growth_per_view_cap = int(getattr(args, "streaming_growth_max_gaussians_per_view", 8000) or 0)
        if _growth_per_view_cap > 0:
            _n_admitted_views = len(streaming_scene.train_cameras)
            if _n_admitted_views > 0:
                _view_density_ceiling = _n_admitted_views * _growth_per_view_cap
                _n_current = int(gaussians.get_xyz.shape[0]) if hasattr(gaussians, "get_xyz") else 0
                if _n_current >= _view_density_ceiling:
                    if not getattr(streaming_training, "_growth_density_warned", False):
                        print(
                            f"[streaming] growth density brake: N={_n_current} >= "
                            f"{_n_admitted_views} admitted views * {_growth_per_view_cap}/view = "
                            f"{_view_density_ceiling} at step={iteration} -- suppressing MCMC growth "
                            f"until more views are admitted or the population shrinks. Set "
                            f"--streaming_growth_max_gaussians_per_view 0 to disable.",
                            flush=True,
                        )
                        streaming_training._growth_density_warned = True
                    sched["allow_growth"] = False
                elif getattr(streaming_training, "_growth_density_warned", False):
                    print(
                        f"[streaming] growth density brake: N={_n_current} back under the "
                        f"{_view_density_ceiling} ceiling at step={iteration} -- growth re-enabled.",
                        flush=True,
                    )
                    streaming_training._growth_density_warned = False

        # VRAM-exhaustion growth brake (2026-07-27): cap_max alone doesn't bound
        # memory when per-frame render/backward cost (native resolution) or GTT
        # spillover dominates instead of Gaussian count -- confirmed on
        # ScanNet++'s 1920x1440 iPhone-LiDAR source, which OOM'd at N=150107
        # (well under its 150000 cap_max having only just been reached) because
        # the render buffers for that resolution, not N, were the actual
        # bottleneck. Unlike cap_max (a static ceiling picked ahead of time,
        # wrong whenever the real per-frame memory cost wasn't known upfront),
        # this reacts to the ACTUAL measured allocation, so growth throttles
        # exactly when memory is genuinely tight regardless of cause (resolution,
        # GTT/shared-VRAM pressure from co-resident processes, driver overhead)
        # rather than guessing a fixed N ceiling in advance. Checked every
        # iteration (torch.cuda.memory_reserved is a cheap driver-side counter
        # read, not a sync) so it reacts immediately, not on a periodic cadence.
        if iteration % max(1, int(getattr(args, "streaming_vram_brake_check_interval", 1))) == 0:
            _vram_frac = float(getattr(args, "streaming_vram_brake_fraction", 0.0) or 0.0)
            if _vram_frac > 0.0 and torch.cuda.is_available():
                _reserved = torch.cuda.memory_reserved()
                _total = torch.cuda.get_device_properties(0).total_memory
                _used_frac = _reserved / max(1, _total)
                if _used_frac >= _vram_frac:
                    if not getattr(streaming_training, "_vram_brake_warned", False):
                        print(
                            f"[streaming] VRAM brake: {_used_frac:.1%} of device memory "
                            f"reserved (threshold {_vram_frac:.1%}) at step={iteration} "
                            f"N={gaussians.get_xyz.shape[0] if hasattr(gaussians, 'get_xyz') else '?'} "
                            f"-- suppressing MCMC growth until usage drops back below "
                            f"threshold. Set --streaming_vram_brake_fraction 0 to disable.",
                            flush=True,
                        )
                        streaming_training._vram_brake_warned = True
                    sched["allow_growth"] = False
                elif getattr(streaming_training, "_vram_brake_warned", False):
                    print(
                        f"[streaming] VRAM brake: usage back below threshold "
                        f"({_used_frac:.1%} < {_vram_frac:.1%}) at step={iteration} -- "
                        f"growth re-enabled.",
                        flush=True,
                    )
                    streaming_training._vram_brake_warned = False
                # Graceful stop tier (2026-07-27): --streaming_vram_brake_fraction
                # only pauses growth -- if memory keeps climbing anyway (existing
                # Gaussians' render/backward cost alone, not new growth, e.g. when
                # testing native resolution deliberately with the auto-downscale
                # gate disabled to see whether downscaling itself was degrading
                # quality), pausing growth forever wastes the rest of the
                # iteration budget spinning with no chance of finishing cleanly.
                # This second, higher threshold breaks out of the main loop
                # instead -- lands on the exact same end-of-training save/report
                # path a normal opt.iterations completion does, not an
                # uncaught-OOM crash or a signal-handler path that skips the
                # final report.
                _vram_stop_frac = float(getattr(args, "streaming_vram_brake_stop_fraction", 0.0) or 0.0)
                if _vram_stop_frac > 0.0 and _used_frac >= _vram_stop_frac:
                    if _post_mapping_refinement_steps > 0:
                        raise RuntimeError(
                            "VRAM safety stop fired before the explicit "
                            "mapping/refinement contract completed"
                        )
                    print(
                        f"[streaming] VRAM brake: STOPPING mapping runtime gracefully -- "
                        f"{_used_frac:.1%} of device memory reserved exceeds the "
                        f"stop threshold ({_vram_stop_frac:.1%}) at step={iteration} "
                        f"N={gaussians.get_xyz.shape[0] if hasattr(gaussians, 'get_xyz') else '?'}. "
                        f"Proceeding to save + final report at this optimization step instead "
                        f"of risking an uncaught OOM crash.",
                        flush=True,
                    )
                    _vram_stop_triggered = True
                    break

        # The explicit growth horizon is authoritative.  Apply it after the
        # local-maintenance/controller/VRAM overlays above because maintenance
        # is allowed to re-enable growth for its cadence, but never beyond a
        # caller-declared fixed-topology boundary (issue #396).
        enforce_growth_stop(
            sched,
            iteration=iteration,
            stop_growth_iter=args.mcmc_stop_growth_iter,
        )

        # Anchor bootstrap Gaussians: save positions before step_post_backward
        # (which includes noise injection) and restore them after, so that MCMC
        # noise never displaces the initial geometry.
        _anchor_bootstrap = getattr(args, "streaming_anchor_bootstrap", False)
        _anchor_mask = None
        _anchor_pos = None
        if _anchor_bootstrap and hasattr(gaussians, "birth_frame"):
            _anchor_mask = (gaussians.birth_frame == 0)
            if _anchor_mask.any():
                _anchor_pos = gaussians.get_xyz[_anchor_mask].detach().clone()

        _before_mcmc_post_bootstrap = (
            _capture_bootstrap_positions(gaussians)
            if (_debug_bootstrap_this_iter and not _skip_mcmc)
            else None
        )
        _mcmc_offset_suppressed = _mcmc_suppressed_by_insertion_offset(
            iteration=iteration,
            last_flush_iteration=_last_flush_iteration,
            offset=_mcmc_offset_from_insertion,
        )
        if _backward_ran and not _skip_mcmc and not _mcmc_offset_suppressed:
            _w3_02_mcmc_n_before = gaussians.get_xyz.shape[0]
            _w3_02_mcmc_topo_before = map_backend.topology_version
            # W3-47: real sensor depth for this view, fetched ONLY when Phase C
            # reseeding is enabled -- feeds LiveWrappedMCMCStrategy's reseed
            # candidate producer (eligsir/optimization/gaussian/strategies/
            # live/reseed_producer.py). A no-op (None, unused kwarg) for every
            # other strategy/phase combination, so this costs nothing when
            # Phase C is off (the default).
            _w3_47_gt_depth = None
            if getattr(args, "streaming_live_strategy_phase_c_reseeding", False):
                _w3_47_gt_depth = _get_sensor_depth(viewpoint_cam, image.shape[1], image.shape[2])
                if _w3_47_gt_depth is not None:
                    _w3_47_gt_depth = _w3_47_gt_depth.to("cuda", non_blocking=True)
            with _timed("step_post_backward"):
                mcmc_strategy.step_post_backward(
                    gaussians=gaussians, args=args, sched=sched, iteration=iteration,
                    utility=utility, temperature=temperature, use_energy_mcmc=use_energy_mcmc,
                    tb_writer=tb_writer,
                    should_log_strategy=lambda i: i % max(1, getattr(args, "strategy_log_interval", 500)) == 0,
                    render_pkg=render_pkg, lr=xyz_lr,
                    # W3-39: gives LiveWrappedMCMCStrategy (gsplat_live_strategy)
                    # the real topology_version plus a real inserted/pruned split
                    # (via map_backend.cumulative_inserted/cumulative_pruned) so
                    # it can tell pure-append Phase-2 growth apart from a
                    # structural change instead of cold-starting its
                    # LivePopulationState on every count change. A no-op for
                    # every other strategy (unused kwarg).
                    map_backend=map_backend,
                    # W3-47: real Phase C reseed candidate producer inputs --
                    # unused kwargs (None) for every strategy/phase combination
                    # other than gsplat_live_strategy with phase_c_reseeding on.
                    viewpoint_camera=viewpoint_cam,
                    gt_depth=_w3_47_gt_depth,
                )
            if _w3_02_mutation_trace.enabled:
                _w3_02_mutation_trace.wrap(
                    iteration=iteration, op="mcmc_step_post_backward",
                    topology_version_before=_w3_02_mcmc_topo_before,
                    topology_version_after=map_backend.topology_version,
                    n_before=_w3_02_mcmc_n_before, n_after=gaussians.get_xyz.shape[0],
                    content=gaussians.get_xyz.shape[0],
                )
            # Issue #326 hook F -- MCMC growth/relocation, aggregate scope.
            # #543 found STATICALLY that no MCMC path advances
            # map_backend.topology_version, so the `!=` guard immediately below
            # can never fire for an ordinary MCMC mutation, and flagged it
            # "unverified at runtime; candidate for #326". Recording n_before/
            # n_after ALONGSIDE topology_version_before/after is exactly what
            # makes that claim runtime-checkable, via
            # gaussian_op_conflicts.population_change_without_topology_advance.
            if _optrace is not None:
                _ot_n_after = int(gaussians.get_xyz.shape[0])
                _ot_delta = _ot_n_after - int(_w3_02_mcmc_n_before)
                # Every real population change is always recorded; the
                # no-population-change (relocation/no-op) case is sampled on the
                # trace cadence so it cannot flood the trace.
                if _ot_delta != 0 or _optrace.traces_iteration(iteration):
                    _optrace.record(
                        iteration=iteration,
                        op=("grow" if _ot_delta > 0 else "retire" if _ot_delta < 0 else "relocate"),
                        mechanism="mcmc", phase="applied", scope="aggregate",
                        topology_version=int(map_backend.topology_version),
                        n_rows=abs(_ot_delta),
                        evidence={
                            "topology_version_before": int(_w3_02_mcmc_topo_before),
                            "topology_version_after": int(map_backend.topology_version),
                            "n_before": int(_w3_02_mcmc_n_before),
                            "n_after": _ot_n_after,
                            "strategy": str(densification_strategy),
                        },
                    )

            # PR2 cadence clocks / issue #727 fix: a real MCMC growth/relocation
            # mutation happened iff EITHER step_post_backward bumped
            # topology_version OR the live population count changed.
            #
            # The topology_version check alone (the original, pre-#727
            # condition) can NEVER fire for ordinary MCMC growth/relocation:
            # #543 found STATICALLY that no MCMC path (add_energy_guided,
            # relocate_energy_guided, and their legacy/baseline equivalents)
            # calls map_backend.commit_mutation()/record_external_mutation() --
            # those tensors are mutated directly on `gaussians`, bypassing the
            # map_backend transaction API entirely. topology_version only
            # advances for depth-insertion commits and the two direct
            # gaussians.prune_points(mask) call sites that explicitly call
            # record_external_mutation(). Left alone, this makes
            # streaming/cadence's mcmc_mutation clock structurally zero on
            # every run regardless of how much real MCMC mutation occurred
            # (issue #727).
            #
            # The population-count check added here (gaussians.get_xyz.shape[0]
            # read is plain tensor metadata, no device sync -- consistent with
            # _w3_02_mcmc_n_before two lines above) correctly catches every
            # growth (add) event, since growth always changes the count. It
            # does NOT catch a pure relocation-only mutation (relocate_energy_
            # guided swaps dead Gaussians for live ones at a fixed population
            # size) -- that remains a real, documented gap tracked as a
            # separate, riskier fix under #326 (making MCMC relocation
            # participate in map_backend's mutation-tracking system), not
            # attempted here. See realized_cadence.py's MCMC_MUTATION_COVERAGE
            # note, mirrored into report.json["realized_cadence"].
            _mcmc_n_after = int(gaussians.get_xyz.shape[0])
            _mcmc_topology_or_n_changed = (
                map_backend.topology_version != _w3_02_mcmc_topo_before
                or _mcmc_n_after != int(_w3_02_mcmc_n_before)
            )
            if _mcmc_topology_or_n_changed:
                _clk_mcmc_mutation_event_id += 1
                _cadence.record_mcmc_mutation()   # #289 mutation clock
                _clk_mutation_event_id += 1
            # Issue #330 hook I -- op-trace visibility for MCMC POPULATION
            # CHANGES (growth/death), closing a gap #543 (gaussian_writer_
            # registry) and #326's own runtime-conflict-trace design flagged
            # but never wired: no MCMC path (add_energy_guided,
            # relocate_energy_guided, or their legacy/baseline equivalents)
            # ever called _optrace.record(), so gaussian_op_conflicts.py's
            # topology_mutation_while_gate_off() (Metric 5 -- exactly the
            # "runtime invariant" issue #330 asks for) could structurally
            # never see an MCMC event and therefore could never catch an MCMC
            # growth committed while the scheduler said allow_growth=False,
            # even though _optrace.set_gate_state() was already being fed the
            # real per-iteration `_decision.allow_growth`/`allow_insertion`
            # values a few hundred lines above (before step_post_backward
            # runs). This hook reuses the population-count/topology_version
            # before/after bracket already computed above for the #289
            # mutation clock, so it costs nothing extra to compute -- it only
            # adds the op-trace emission itself (a no-op unless
            # ELIGSIR_GAUSSIAN_OP_TRACE is enabled).
            #
            # KNOWN REMAINING GAP (documented, not fixed here): a pure
            # relocation-only mutation (relocate_energy_guided swaps a dead
            # Gaussian for a live one's state at a FIXED population size, and
            # -- per the comment a few lines above -- never bumps
            # topology_version either) changes neither signal this hook reads,
            # so it stays invisible to the op trace. Making relocation visible
            # needs a real before/after per-uid state snapshot bracket around
            # step_post_backward (the pattern gaussian_op_probe.emit_uid_deltas
            # already supports elsewhere in this file, e.g. around noise
            # injection) -- deliberately not bundled into this
            # instrumentation-only change because a full-population snapshot
            # every traced iteration is expensive enough to want its own
            # follow-up and its own GPU cost/benefit check.
            if _optrace is not None and _mcmc_topology_or_n_changed:
                _mcmc_delta = _mcmc_n_after - int(_w3_02_mcmc_n_before)
                if _mcmc_delta > 0:
                    _mcmc_op = "grow"
                elif _mcmc_delta < 0:
                    _mcmc_op = "retire"
                else:
                    # topology_version moved with no population-count change --
                    # not currently reachable by any production MCMC path (see
                    # the comment above), kept as an honest fallback label
                    # rather than silently mislabeling it "grow"/"retire".
                    _mcmc_op = "relocate"
                _optrace.record(
                    iteration=iteration, op=_mcmc_op, mechanism="mcmc",
                    phase="applied", scope="aggregate",
                    topology_version=int(map_backend.topology_version),
                    n_rows=int(abs(_mcmc_delta)),
                    evidence={
                        "topology_version_before": int(_w3_02_mcmc_topo_before),
                        "topology_version_after": int(map_backend.topology_version),
                        "n_before": int(_w3_02_mcmc_n_before),
                        "n_after": _mcmc_n_after,
                        "strategy": str(densification_strategy),
                        "coverage_note": (
                            "aggregate population/topology-delta signal; does "
                            "NOT cover pure relocation-only mutations (see "
                            "issue #330 hook I comment above)"
                        ),
                    },
                )
            _memprobe("after_step_post_backward", iteration, f"N={gaussians.get_xyz.shape[0]}")

            # Restore bootstrap positions after noise injection. Was a raw
            # params["means"]/._xyz write dispatched on layout; routed
            # through the mutation transaction instead (see
            # docs/architecture/gaussian_ownership_audit.md).
            if _anchor_mask is not None and _anchor_pos is not None:
                if gaussians.get_xyz.shape[0] == _anchor_mask.shape[0]:
                    from eligsir.map.mutation import SubmapCorrection

                    # Issue #326/#327 hook I -- anchor-bootstrap restore
                    # cancelling MCMC noise/growth/relocation on birth_frame==0
                    # rows. Found during #327's structural movement-conflict
                    # analysis: this hard restore is NOT one of #326's original
                    # hooks A-H, and is likewise absent from #325's static
                    # writer/owner registry (the registry's manually-curated
                    # `apply_pose_correction` caller list only names the three
                    # loop_closure/submap_manager call sites, not this one).
                    # Structurally identical to hooks B/D's depth-lock restore:
                    # applied strictly AFTER step_post_backward (hook F), and
                    # unconditionally overwrites whatever MCMC just wrote for
                    # the affected rows -- just a different row subset
                    # (birth_frame==0) and gated by a different flag
                    # (--streaming_anchor_bootstrap instead of
                    # --streaming_depth_lock_centers).
                    _optrace_snap_anchor = None
                    if _optrace is not None:
                        from eligsir.diagnostics import gaussian_op_probe as _gop
                        _optrace_snap_anchor, _ = _gop.snapshot(gaussians, _optrace.uid_stride)

                    _anchor_txn = map_backend.begin_mutation()
                    _anchor_txn.apply_pose_correction(
                        SubmapCorrection(mask=_anchor_mask, new_positions=_anchor_pos)
                    )
                    map_backend.commit_mutation(_anchor_txn)

                    if _optrace is not None and _optrace_snap_anchor is not None:
                        _snap_post_anchor, _ = _gop.snapshot(gaussians, _optrace.uid_stride)
                        _gop.emit_uid_deltas(
                            _optrace, before=_optrace_snap_anchor, after=_snap_post_anchor,
                            iteration=iteration, op="move_xyz",
                            mechanism="anchors_depth_locks", phase="cancelled",
                            topology_version=int(getattr(map_backend, "topology_version", 0)),
                            evidence={"reason": "streaming_anchor_bootstrap_restore(post_mcmc)"},
                        )
            if _depth_lock_active:
                restore_depth_locked_centers(gaussians, args)
            if _seed_lineage_freeze is not None:
                _seed_lineage_freeze.assert_integrity(
                    iteration=iteration,
                    context="post-mcmc-depth-lock",
                    checkpoint=True,
                )
            if _debug_bootstrap_this_iter:
                _log_bootstrap_motion(
                    tb_writer,
                    args,
                    iteration,
                    "mcmc_post_backward",
                    _bootstrap_motion_stats(
                        gaussians,
                        before=_before_mcmc_post_bootstrap,
                        threshold=_debug_bootstrap_threshold,
                    ),
                )

        if _debug_bootstrap_this_iter:
            _log_bootstrap_motion(
                tb_writer,
                args,
                iteration,
                "total",
                _bootstrap_motion_stats(gaussians, threshold=_debug_bootstrap_threshold),
            )
            if _debug_bootstrap_ply_interval > 0 and (
                iteration <= 5 or iteration % _debug_bootstrap_ply_interval == 0
            ):
                _write_bootstrap_motion_ply(gaussians, args, iteration, "post_mcmc")

        # ---- Deferred lifecycle pruning (safe: optimizer/MCMC already done) --
        if _deferred_stale_mask is not None and _deferred_stale_count > 0:
            with torch.no_grad():
                current_n = gaussians.get_xyz.shape[0]
                mask_n = _deferred_stale_mask.shape[0]
                if mask_n < current_n:
                    # MCMC growth added entries after mask was computed; pad with False
                    pad = torch.zeros(current_n - mask_n, dtype=torch.bool, device=_deferred_stale_mask.device)
                    _deferred_stale_mask = torch.cat([_deferred_stale_mask, pad])
                from eligsir.utils.seed_lineage_freeze import exclude_frozen_seed_mask
                _deferred_stale_mask = exclude_frozen_seed_mask(
                    gaussians,
                    args,
                    _deferred_stale_mask,
                    iteration=iteration,
                    role="prune",
                )
                _deferred_stale_count = int(_deferred_stale_mask.sum().item())
                if (
                    _deferred_stale_mask.shape[0] == current_n
                    and _deferred_stale_count > 0
                ):
                    print(
                        f"[streaming] step={iteration} pruning {_deferred_stale_count} deferred depth/lifecycle points.",
                        flush=True,
                    )
                    # Issue #326 hook G -- lifecycle death CANDIDATES, recorded
                    # BEFORE the prune (the mask indexes pre-prune rows) and
                    # separately from the execution, because an applied retire
                    # only ever has one owner: candidate-set overlap between
                    # lifecycle / mcmc / safety is invisible otherwise.
                    _ot_topo_before = int(getattr(map_backend, "topology_version", 0))
                    if _optrace is not None:
                        from eligsir.diagnostics import gaussian_op_probe as _gop
                        _gop.emit_candidate_set(
                            _optrace, gaussians, _deferred_stale_mask,
                            iteration=iteration, op="retire", mechanism="lifecycle",
                            topology_version=_ot_topo_before,
                            reason="deferred_depth_lifecycle_stale_mask",
                        )
                    gaussians.prune_points(_deferred_stale_mask)
                    # W3-39: this prune bypasses map_backend.begin_mutation()/
                    # commit_mutation() -- record it against the same
                    # cumulative_pruned counter/topology_version bump a
                    # transaction-mediated prune would produce, or
                    # LiveWrappedMCMCStrategy's pure-append-growth signal
                    # would silently miss this structural change (see
                    # GsplatMapBackend.record_external_mutation's docstring).
                    map_backend.record_external_mutation(pruned=_deferred_stale_count)
                    if _optrace is not None:
                        _optrace.record(
                            iteration=iteration, op="retire", mechanism="lifecycle",
                            phase="applied", scope="aggregate",
                            topology_version=int(map_backend.topology_version),
                            n_rows=int(_deferred_stale_count),
                            evidence={
                                "topology_version_before": _ot_topo_before,
                                "topology_version_after": int(map_backend.topology_version),
                                "n_before": int(current_n),
                                "n_after": int(gaussians.get_xyz.shape[0]),
                                "reason": "deferred_depth_lifecycle_stale_mask",
                            },
                        )
                    streaming_scene.maintain_occupancy_hash(getattr(args, "streaming_insert_voxel_size", 0.02))
                elif _deferred_stale_mask.shape[0] != current_n:
                    print(f"[streaming] WARNING: deferred prune mask shape mismatch ({_deferred_stale_mask.shape[0]} vs {current_n}), skipping", flush=True)

        # Periodic occupancy rebuild (independent of pruning)
        with torch.no_grad():
            _occ_rebuild_every = getattr(args, "streaming_occupancy_rebuild_interval", 200)
            if _occ_rebuild_every > 0 and iteration % _occ_rebuild_every == 0:
                with _timed("occupancy_rebuild"):
                    streaming_scene.maintain_occupancy_hash(getattr(args, "streaming_insert_voxel_size", 0.02))

        # ---- Bootstrap seed decay (one-shot) --------------------------------
        # When the trajectory has ingested enough frames, the frame-0 bootstrap
        # seeds should have been superseded by depth-inserted geometry. Clamp
        # their opacity to <=0.1 and demote them to YOUNG so they must re-earn
        # maturity (or get relocated/pruned) instead of lingering as stale mass.
        _seed_decay_frames = int(getattr(args, "streaming_bootstrap_seed_decay_frames", 0) or 0)
        if (
            _seed_decay_frames > 0
            and not _seed_decay_done
            and n_frames_ingested >= _seed_decay_frames
        ):
            _seed_decay_done = True
            with torch.no_grad():
                _seed_mask = _bootstrap_mask(gaussians)
                if _seed_mask is not None:
                    from eligsir.utils.seed_lineage_freeze import exclude_frozen_seed_mask
                    _seed_mask = exclude_frozen_seed_mask(
                        gaussians,
                        args,
                        _seed_mask,
                        iteration=iteration,
                        role="decay",
                    )
                    _op_cap = float(inverse_sigmoid(torch.tensor(0.1)))
                    _op = gaussians.params["opacities"].data
                    _op[_seed_mask] = _op[_seed_mask].clamp(max=_op_cap)
                    if (
                        hasattr(gaussians, "lifecycle_state")
                        and gaussians.lifecycle_state.shape[0] == _seed_mask.shape[0]
                    ):
                        gaussians.lifecycle_state[_seed_mask] = 1  # YOUNG
                    print(
                        f"[streaming] step={iteration} bootstrap seed decay: "
                        f"clamped {int(_seed_mask.sum().item())} frame-0 splats to opacity<=0.1 (YOUNG)",
                        flush=True,
                    )

        # ---- Out-of-bounds bbox prune (at global-maintenance cadence) ---------
        # Gaussians that have drifted outside the camera trajectory bbox + margin
        # can never contribute positively.  Remove them to keep N bounded and stop
        # the runaway-extent feedback loop.
        _bbox_prune_enabled = getattr(args, "streaming_bbox_prune", True)
        if (
            _bbox_prune_enabled
            and global_maint_interval > 0
            and iteration % global_maint_interval == 0
        ):
            with _timed("bbox_prune"), torch.no_grad():
                _all_cams = list(streaming_scene.getTrainCameras())
                if len(_all_cams) >= 2:
                    import numpy as _np
                    _max_d = float(getattr(args, "streaming_max_depth", 8.0))
                    # OOB margin beyond the camera-trajectory bbox. The legacy 3×max_depth
                    # (=24 m at 8 m range) is far too lenient for small scenes — floaters
                    # bleed 10–26 m out and survive. Flag-tunable; lower culls far floaters.
                    _margin = float(getattr(args, "streaming_bbox_prune_margin_mult", 3.0)) * _max_d
                    # camera positions in world space: -R_c2w @ t_w2c
                    _cam_pos = torch.tensor(
                        _np.stack([
                            (-_np.array(c.R) @ _np.array(c.T))
                            for c in _all_cams
                        ]),
                        dtype=torch.float32,
                        device=gaussians.get_xyz.device,
                    )
                    _bbox_min = _cam_pos.min(dim=0).values - _margin
                    _bbox_max = _cam_pos.max(dim=0).values + _margin
                    _xyz = gaussians.get_xyz
                    _oob = ((_xyz < _bbox_min) | (_xyz > _bbox_max)).any(dim=-1)
                    # Reachability prune (escape-plume fix): a splat farther
                    # than max_depth+margin from EVERY camera can never be
                    # depth-verified — it lives in the supervision-free zone
                    # where the plume grows. Distance to nearest camera via
                    # chunked cdist. Flag streaming_reach_prune_margin_m > 0.
                    _reach_margin = float(getattr(args, "streaming_reach_prune_margin_m", 0.0) or 0.0)
                    if _reach_margin > 0:
                        _reach_lim = _max_d + _reach_margin
                        _min_d = torch.full((_xyz.shape[0],), float("inf"), device=_xyz.device)
                        for _ci in range(0, _cam_pos.shape[0], 256):
                            _d = torch.cdist(_xyz, _cam_pos[_ci:_ci + 256]).min(dim=1).values
                            _min_d = torch.minimum(_min_d, _d)
                        _unreach = _min_d > _reach_lim
                        from eligsir.utils.seed_lineage_freeze import exclude_frozen_seed_mask
                        _recyclable = exclude_frozen_seed_mask(
                            gaussians,
                            args,
                            _unreach,
                            iteration=iteration,
                            role="recycle",
                        )
                        if (
                            getattr(args, "streaming_reach_recycle", False)
                            and _recyclable.any()
                            and iteration < opt.densify_until_iter
                        ):
                            # "Move back" instead of delete: crush opacity so the
                            # MCMC dead-relocation teleports these splats onto
                            # in-scene high-utility parents (capacity recycled).
                            # Was a raw params["opacities"].data write; routed
                            # through the mutation transaction instead (see
                            # docs/architecture/gaussian_ownership_audit.md).
                            with torch.no_grad():
                                _reach_txn = map_backend.begin_mutation()
                                _reach_txn.update_opacity_logits(_recyclable, -9.21)  # sigmoid ~1e-4
                                map_backend.commit_mutation(_reach_txn)
                                if (
                                    getattr(args, "streaming_recycle_relocatable", False)
                                    and hasattr(gaussians, "lifecycle_state")
                                    and gaussians.lifecycle_state.shape[0] == _recyclable.shape[0]
                                ):
                                    # Reset to PROVISIONAL so the mature-gate does
                                    # not shield crushed splats from relocation.
                                    gaussians.lifecycle_state[_recyclable] = 0
                        else:
                            # Past densify_until_iter relocation no longer runs —
                            # crushing would just leave dead mass; hard-prune.
                            _oob = _oob | _unreach
                    # Absolute extreme-scale safety net (unconditional on lifecycle
                    # state / support): the relative scale_kill_threshold dead-mask
                    # (utils/strategies/mcmc_strategy.py) is explicitly gated off
                    # for MATURE Gaussians, so a >5m-radius floater that matures
                    # before being killed can never be removed by that mechanism.
                    # This is a hard absolute cap, independent of maturity.
                    _max_radius_m = float(getattr(args, "streaming_max_gaussian_radius_m", 0.0) or 0.0)
                    if _max_radius_m > 0:
                        _max_scale = gaussians.get_scaling.max(dim=-1).values
                        _oob = _oob | (_max_scale > _max_radius_m)
                    from eligsir.utils.seed_lineage_freeze import exclude_frozen_seed_mask
                    _oob = exclude_frozen_seed_mask(
                        gaussians,
                        args,
                        _oob,
                        iteration=iteration,
                        role="prune",
                    )
                    _oob_count = int(_oob.sum().item())
                    if _oob_count > 0:
                        # Issue #326 hook H -- hard-safety death candidates +
                        # execution. Same candidate/applied split as hook G.
                        _ot_oob_topo_before = int(getattr(map_backend, "topology_version", 0))
                        _ot_oob_n_before = int(gaussians.get_xyz.shape[0])
                        if _optrace is not None:
                            from eligsir.diagnostics import gaussian_op_probe as _gop
                            _gop.emit_candidate_set(
                                _optrace, gaussians, _oob,
                                iteration=iteration, op="retire",
                                mechanism="freespace_contradiction",
                                topology_version=_ot_oob_topo_before,
                                reason="oob_bbox_or_max_radius_safety_prune",
                            )
                        gaussians.prune_points(_oob)
                        # W3-39: same rationale as the deferred-stale-mask prune
                        # above -- this bypasses map_backend's transaction API,
                        # so record it explicitly against the cumulative
                        # counters LiveWrappedMCMCStrategy's pure-append-growth
                        # signal relies on.
                        map_backend.record_external_mutation(pruned=_oob_count)
                        if _optrace is not None:
                            _optrace.record(
                                iteration=iteration, op="retire",
                                mechanism="freespace_contradiction",
                                phase="applied", scope="aggregate",
                                topology_version=int(map_backend.topology_version),
                                n_rows=int(_oob_count),
                                evidence={
                                    "topology_version_before": _ot_oob_topo_before,
                                    "topology_version_after": int(map_backend.topology_version),
                                    "n_before": _ot_oob_n_before,
                                    "n_after": int(gaussians.get_xyz.shape[0]),
                                    "reason": "oob_bbox_or_max_radius_safety_prune",
                                },
                            )
                        streaming_scene.maintain_occupancy_hash(
                            getattr(args, "streaming_insert_voxel_size", 0.02)
                        )
                        if tb_writer:
                            tb_writer.add_scalar("streaming/oob_pruned", _oob_count, iteration)

        # ---- Logging ------------------------------------------------------
        with torch.no_grad():
            # Only materialize the loss scalar (a host-sync) when it's actually
            # consumed — the progress-bar EMA (every 10 iters) or tb logging
            # (every scalar_log_interval) — instead of every iteration.
            _loss_val = None
            if iteration % 10 == 0 or (tb_writer is not None and iteration % scalar_log_interval == 0):
                _loss_val = loss.item()
            if iteration % 10 == 0:
                ema_loss = 0.4 * _loss_val + 0.6 * ema_loss
                # Mapping-perf stats (canonical, see docs/architecture/pipeline-nomenclature.md).
                # All from counters already in hand — a few divisions per 10 iters, no GPU work.
                _adm = int(getattr(streaming_scene, "_keyframes_admitted", 0)) or 1
                _views = len(getattr(streaming_scene, "train_cameras", []) or []) or _adm
                _elapsed = max(1e-6, time.perf_counter() - _train_t0)
                # Issue #701: use compute_progress_postfix() for testable, correct denominator.
                from eligsir.utils.execution_semantics import compute_progress_postfix
                postfix = compute_progress_postfix(
                    iteration=iteration,
                    n_gaussians=gaussians.get_xyz.shape[0],
                    n_frames_ingested=n_frames_ingested,
                    n_train_cameras=_views,
                    keyframes_admitted=_adm,
                    elapsed_s=_elapsed,
                    ema_loss=ema_loss,
                )
                # refresh=False: set_postfix() renders by default, and update()
                # immediately below renders again -- two tqdm refreshes per
                # 10-iteration checkpoint. Invisible interactively (carriage
                # return overwrites in place) but roughly doubles every
                # streaming run's train.log, since each refresh becomes a
                # permanent text block once piped to a file (found 2026-08-15
                # inspecting the recorded streaming run log).
                progress_bar.set_postfix(postfix, refresh=False)
                progress_bar.update(10)

            # Phase 5: Online/Offline unification.
            # BUG (found 2026-07-06): this used to unconditionally break as soon
            # as the source ran out of frames and the dirty-region backlog
            # drained -- with zero regard for _refinement_iters/the rolling
            # idle-stop window (top of this loop, ~line 4223). Since the dirty
            # queue typically drains within a handful of iterations of the last
            # ingest, this fired almost immediately every time, completely
            # bypassing the configured streaming_refinement_iterations (50000
            # by default) -- e.g. fr3par_hybrid_ctrl terminated at iter 6111
            # regardless of refinement_iterations being correctly set to 50000
            # in cfg_args, because THIS check broke the loop first, every run.
            # The rolling-idle-stop rule at the top of the loop is the sole,
            # already-correct authority for online_mode termination; this
            # redundant, refinement-window-blind check is removed rather than
            # patched, since "input exhausted + no next frame" is exactly the
            # condition the rolling rule already measures against
            # _refinement_iters via _last_ingest_iter.

            # ---- Training-progress video capture (async) --------------------
            if _progress_video_interval > 0 and iteration % _progress_video_interval == 0:
                # Lock the progress-video cameras once. Recomputing "closest"
                # cameras as more frames arrive makes the exported video cycle
                # through different camera IDs, which is misleading.
                if not _progress_cams:
                    _arrived_trcams = list(streaming_scene.train_cameras)
                    if len(_arrived_trcams) >= 3:
                        _n_arr = len(_arrived_trcams)
                        _progress_cams = [
                            _arrived_trcams[0],
                            _arrived_trcams[(_n_arr - 1) // 2],
                            _arrived_trcams[_n_arr - 1],
                        ]
                if _progress_cams:
                    # Submit batched render to side CUDA stream
                    with torch.cuda.stream(_pv_stream):
                        _pv_stream.wait_stream(torch.cuda.current_stream())
                        with torch.no_grad():
                            _gpu_panels = _render_batch(
                                _progress_cams, gaussians, pipe, background
                            )
                    # Hand off to CPU worker; drop frame if worker is still busy
                    try:
                        _pv_cpu_queue.put_nowait((_gpu_panels, iteration, opt.iterations))
                    except _queue_mod.Full:
                        pass

            should_log = tb_writer and iteration % scalar_log_interval == 0
            # Issue #279 (Phase 1) -- shared scalar readback.
            #
            # The TensorBoard block below and the viewer block further down are
            # gated on the SAME cadence (`iteration % scalar_log_interval == 0`)
            # and were each independently recomputing `psnr(image, gt_image)`
            # and re-reading `Ll1`: with both active that is two full PSNR
            # kernels plus three separate `.item()` host readbacks (three
            # implicit device syncs) for two scalars that are identical by
            # construction -- same `image`, same `gt_image`, same `Ll1`, same
            # iteration, no mutation in between.
            #
            # These are computed here ONLY under `should_log`, and each consumer
            # falls back to its own original inline computation when the value
            # is None. That keeps the change strictly work-reducing:
            #   TB + viewer : 2 psnr + 3 item()  ->  1 psnr + 1 item()
            #   TB only     : 1 psnr + 2 item()  ->  1 psnr + 1 item()
            #   viewer only : 1 psnr + 1 item()  ->  unchanged (computed in the
            #                 viewer's own try/except, exactly as before)
            # so no configuration gains work, and the viewer block's exception
            # handling semantics are untouched.
            _metric_psnr = None
            _metric_l1 = None
            _metric_score_transform = None
            if should_log:
                _score_image, _score_gt, _metric_score_transform = (
                    apply_declared_score_clamp_pair(
                        image,
                        gt_image,
                        context="streaming_progress_psnr",
                    )
                )
                _metric_psnr = psnr(_score_image, _score_gt).mean().item()
                _metric_l1 = Ll1.item()
            if should_log:
                tb_writer.add_scalar("optimization/l1_loss", _metric_l1, iteration)
                tb_writer.add_scalar("optimization/total_loss", _loss_val, iteration)
                tb_writer.add_scalar("optimization/psnr", _metric_psnr, iteration)
                _score_render_record = _metric_score_transform["render"]
                tb_writer.add_scalar(
                    "optimization/score_clamp_fraction",
                    _score_render_record["fraction_changed"],
                    iteration,
                )
                tb_writer.add_scalar(
                    "optimization/score_render_raw_max",
                    _score_render_record["value_max"],
                    iteration,
                )
                tb_writer.add_scalar("streaming/frames_ingested", n_frames_ingested, iteration)
                tb_writer.add_scalar("streaming/gaussians", gaussians.get_xyz.shape[0], iteration)
                tb_writer.add_scalar("streaming/total_inserted", total_inserted, iteration)
                # Issue #396: camera-image page-in/page-out EVENT counters. These
                # are deliberately not resident bytes -- byte-level per-subsystem
                # residency attribution is issue #399's GPUMemoryResidencyTracker
                # (PR #513) and is not reimplemented here. A healthy run has
                # paged_out tracking paged_in with a small constant gap (the HOT
                # retentions); paged_in growing while paged_out stays flat is the
                # #396 leak.
                _res_stats = _residency_ledger.stats()
                tb_writer.add_scalar(
                    "streaming/residency/camera_page_in", _res_stats["paged_in"], iteration)
                tb_writer.add_scalar(
                    "streaming/residency/camera_page_out", _res_stats["paged_out"], iteration)
                tb_writer.add_scalar(
                    "streaming/residency/camera_retained_hot", _res_stats["retained_hot"], iteration)
                tb_writer.add_scalar(
                    "streaming/residency/max_concurrent_paged_in",
                    _res_stats["max_concurrent_paged_in"], iteration)
                # Issue #795: emit admission fingerprint scalars (running counts).
                tb_writer.add_scalar(
                    "streaming/admission/total_admitted",
                    _admission_fingerprint.admitted_total, iteration)
                tb_writer.add_scalar(
                    "streaming/admission/total_rejected",
                    _admission_fingerprint.rejected_total, iteration)
                tb_writer.add_scalar(
                    "streaming/admission/force_hard_count",
                    _admission_fingerprint.force_hard_count, iteration)
                tb_writer.add_scalar(
                    "streaming/admission/force_soft_count",
                    _admission_fingerprint.force_soft_count, iteration)

                # PR2 (2026-08-12 cadence/batching plan): explicit optimizer/
                # view/mutation clocks, telemetry only. Gated on
                # streaming_cadence_telemetry_enabled (default False) --
                # the counters above always increment regardless, but nothing
                # is logged and no control flow branches on them unless this
                # flag is set, so this is a zero-behavior-change addition.
                if getattr(args, "streaming_cadence_telemetry_enabled", False):
                    tb_writer.add_scalar(
                        "streaming/clocks/optimizer_step_id", _clk_optimizer_step_id, iteration)
                    tb_writer.add_scalar(
                        "streaming/clocks/supervision_view_count", _clk_supervision_view_count, iteration)
                    if _opt_batch_count > 0:
                        # Issue #106: mean DISTINCT cameras per supervision
                        # batch. Compare against --streaming_optimization_batch_size:
                        # a value materially below B means the batch is drawing
                        # duplicates and "B views" overstates the real
                        # multi-view supervision the step received.
                        tb_writer.add_scalar(
                            "streaming/opt_batch_distinct_views",
                            _opt_batch_distinct_total / _opt_batch_count, iteration)
                    tb_writer.add_scalar(
                        "streaming/clocks/sensor_frame_count", _clk_sensor_frame_count, iteration)
                    tb_writer.add_scalar(
                        "streaming/clocks/mutation_event_id", _clk_mutation_event_id, iteration)
                    tb_writer.add_scalar(
                        "streaming/clocks/depth_mutation_event_id", _clk_depth_mutation_event_id, iteration)
                    tb_writer.add_scalar(
                        "streaming/clocks/mcmc_mutation_event_id", _clk_mcmc_mutation_event_id, iteration)
                    # Issue #703: supervision-work accounting scalars (rasterizer calls, supervised pixels)
                    tb_writer.add_scalar(
                        "streaming/clocks/rasterizer_call_count", _clk_rasterizer_call_count, iteration)
                    tb_writer.add_scalar(
                        "streaming/clocks/supervised_pixel_count", _clk_supervised_pixel_count, iteration)
                    # Derived ratios from eligsir.utils/supervision_work.py
                    if _clk_optimizer_step_id > 0:
                        tb_writer.add_scalar(
                            "streaming/work/rasterizer_calls_per_step",
                            rasterizer_calls_per_optimization_step(_clk_optimizer_step_id, _clk_rasterizer_call_count),
                            iteration)
                        tb_writer.add_scalar(
                            "streaming/work/supervised_pixels_per_step",
                            supervised_pixels_per_optimization_step(_clk_optimizer_step_id, _clk_supervised_pixel_count),
                            iteration)
                        tb_writer.add_scalar(
                            "streaming/work/supervision_views_per_step",
                            supervision_views_per_optimization_step(_clk_optimizer_step_id, _clk_supervision_view_count),
                            iteration)

                # --- Realized cadence (issue #289) ---------------------------
                # Cumulative realized RATES, denominated in optimizer steps.
                # Logged unconditionally (not behind
                # streaming_cadence_telemetry_enabled) because no seed-fade or
                # streaming-quality experiment may cite nominal steps_per_frame
                # any more -- realized cadence has to be readable from every
                # run, not only from runs that remembered a flag. Cost is six
                # pure-Python divisions on the existing TB cadence.
                _cad_realized_spf = _cadence.realized_steps_per_release()
                if _cad_realized_spf is not None:
                    tb_writer.add_scalar(
                        "streaming/cadence/realized_steps_per_release",
                        _cad_realized_spf, iteration)
                tb_writer.add_scalar(
                    "streaming/cadence/optimizer_steps", _cadence.optimizer_steps, iteration)
                tb_writer.add_scalar(
                    "streaming/cadence/source_released", _cadence.source_released, iteration)
                tb_writer.add_scalar(
                    "streaming/cadence/keyframes_admitted", _cadence.keyframe_admitted, iteration)
                tb_writer.add_scalar(
                    "streaming/cadence/inserted_gaussians", _cadence.inserted_gaussians, iteration)
                # One-shot requested-vs-realized divergence warning. Latched
                # inside the tracker, so this prints at most once per run and
                # never costs more than the reconciliation arithmetic.
                _cad_warn = _cadence.maybe_warn(**_cadence_reconcile_kwargs())
                if _cad_warn:
                    print(_cad_warn, flush=True)

                # --- Assimilation-pressure telemetry (Phase 1, pure observation) ---
                # Uncalibrated first pass -- see utils/assimilation_pressure.py's module
                # docstring and docs/architecture/2026-08-10-debt-controlled-sparse-
                # assimilation-plan.md Phase 1. Every signal below is a best-effort proxy
                # built from whatever is already computed at this point in the loop; none
                # of this feeds back into training decisions. locals().get(...) guards are
                # used where the source variable is only ever assigned inside a gated/
                # conditional branch elsewhere in the loop (so it may be absent on early
                # iterations, or absent entirely for a whole run when that feature is off)
                # -- this keeps the block a pure add with zero risk of a NameError
                # crashing training, per this task's file-scope contract.
                #
                # source_insert_rate <- `added`, the most recent Phase-2 depth-insertion
                #   per-frame insert count (utils/streaming_frames.py insertion path,
                #   set at the "total_inserted += added" line above/earlier in the loop).
                #   Normalized by a fixed cap of 1000 new Gaussians/frame (matches
                #   --streaming_max_new_gaussians_per_frame's default order of magnitude).
                _assim_added_last = locals().get("added", 0) or 0
                _source_insert_rate = min(1.0, float(_assim_added_last) / 1000.0)

                # support_debt <- mean under-support fraction of currently-provisional
                #   (newborn, not-yet-promoted) Gaussians: clamp(1 - support_count /
                #   streaming_min_support_views, 0, 1), averaged over the provisional
                #   population. 0 when there are no provisional Gaussians (nothing is
                #   under-supported) or before the model/attrs exist.
                _support_debt = 0.0
                if (
                    hasattr(gaussians, "provisional")
                    and hasattr(gaussians, "support_count")
                    and gaussians.get_xyz.shape[0] > 0
                ):
                    # issue #902: `or 3` silently rewrote a legitimate
                    # --streaming_min_support_views 0 to 3 (0 is falsy), and
                    # the hand-copied fallback (3) had already drifted from
                    # the real argparse default of 2 (arguments/__init__.py)
                    # -- the exact issue #363/#368 stale-default-trap shape.
                    # This is an unconditional ParamGroup default, always
                    # present on the real parsed args -- bare attribute
                    # access, no fallback literal left to drift.
                    _min_support = float(args.streaming_min_support_views)
                    _prov_mask = gaussians.provisional
                    if bool(_prov_mask.any()):
                        _support_debt = float(
                            (1.0 - gaussians.support_count[_prov_mask].float() / max(_min_support, 1e-6))
                            .clamp(0.0, 1.0)
                            .mean()
                            .item()
                        )

                # residual_pressure <- the current step's L1 photometric loss (`Ll1`,
                #   already logged above as optimization/l1_loss), normalized by a fixed
                #   cap of 0.2 (a high-but-plausible L1 value on normalized [0,1] images;
                #   this run's actual worst-case L1 was not empirically characterized, so
                #   treat this cap as a first-pass approximation pending calibration).
                # raw value for residual_pressure is already logged above as optimization/l1_loss
                # Issue #279: reuse the shared `should_log` readback of Ll1
                # (this block is itself inside `should_log`, so `_metric_l1` is
                # always populated here) instead of a third host sync.
                _residual_pressure = min(
                    1.0,
                    float(_metric_l1 if _metric_l1 is not None else Ll1.item()) / 0.2,
                )

                # partition_pressure <- fraction of trajectory partitions flagged as low
                #   outliers by the previous role-safe partition-control decision. Since
                #   the control block runs AFTER this one, this is deliberately one tick
                #   stale. Defaults to 0.0 when control is off or has not fired yet.
                # Issue #951: only the role-safe controller decision is
                # eligible. In particular, never read the scored PSNR tick or
                # the old holdout-derived `_bad_partitions` locals here.
                _partition_pressure = (
                    _partition_control_last_decision.partition_pressure
                    if _partition_control_last_decision is not None
                    else 0.0
                )

                # newborn_pressure <- fraction of live Gaussians still `provisional`
                #   (not yet promoted past streaming_min_support_views). Already
                #   naturally in [0, 1].
                _newborn_pressure = 0.0
                if hasattr(gaussians, "provisional") and gaussians.get_xyz.shape[0] > 0:
                    _newborn_pressure = float(gaussians.provisional.float().mean().item())

                # optimization_rate <- this iteration's measured WHOLE-LOOP wall-clock
                #   time (`_iter_dt` = time since the previous iteration's own
                #   measurement, set unconditionally near the top of this loop for the
                #   dataset_fps clock). This is NOT optimizer-exclusive throughput --
                #   it includes frame ingestion, depth insertion, and every other
                #   per-iteration cost alongside the actual backward/optimizer-step
                #   work (2026-08-11 review finding: the name "optimization_rate"
                #   overstates what's measured; treat this as a loop-throughput proxy,
                #   not a true optimizer-capacity signal, and do not use it to
                #   distinguish "the optimizer is slow" from "ingestion/insertion is
                #   slow this iteration" -- those are conflated here). Converted to
                #   it/s and normalized against a fixed cap of 50 it/s (a
                #   fast-but-plausible ceiling for this codebase per observed runs --
                #   see e.g. the fr3 fast-profile note, ~160it/s on an outlier
                #   scene/config; treat this cap as approximate pending calibration).
                #   Higher = more throughput available = less pressure, matching
                #   classify_assimilation_pressure's expected (uninverted) input
                #   convention.
                _assim_iter_dt = locals().get("_iter_dt", None)
                if _assim_iter_dt and _assim_iter_dt > 0:
                    _optimization_rate = min(1.0, (1.0 / _assim_iter_dt) / 50.0)
                else:
                    _optimization_rate = 1.0

                tb_writer.add_scalar("streaming/assimilation_source_insert_rate", _source_insert_rate, iteration)
                tb_writer.add_scalar("streaming/assimilation_support_debt", _support_debt, iteration)
                tb_writer.add_scalar("streaming/assimilation_residual_pressure", _residual_pressure, iteration)
                tb_writer.add_scalar("streaming/assimilation_partition_pressure", _partition_pressure, iteration)
                tb_writer.add_scalar("streaming/assimilation_newborn_pressure", _newborn_pressure, iteration)
                tb_writer.add_scalar("streaming/assimilation_optimization_rate", _optimization_rate, iteration)
                tb_writer.add_scalar("streaming/assimilation_raw_added", float(_assim_added_last), iteration)
                tb_writer.add_scalar("streaming/assimilation_raw_iter_dt_s", float(_assim_iter_dt or 0.0), iteration)

                _assim_pressure_tier = classify_assimilation_pressure(
                    source_insert_rate=_source_insert_rate,
                    support_debt=_support_debt,
                    residual_pressure=_residual_pressure,
                    partition_pressure=_partition_pressure,
                    newborn_pressure=_newborn_pressure,
                    optimization_rate=_optimization_rate,
                )
                tb_writer.add_scalar("streaming/assimilation_pressure_tier", int(_assim_pressure_tier), iteration)

                # --- Issue #110 shadow mode (opt-in, log-only, no behavior change) ---
                # Logs what each of the four assimilation-pressure levers WOULD decide
                # at the current tier. Never applied to a real MappingDecision -- see
                # docs/architecture/2026-08-12-w3-issue110-assimilation-shadow-mode-design.md.
                if getattr(args, "streaming_assimilation_shadow_mode", False):
                    _shadow_decision = shadow_decision(
                        _assim_pressure_tier, base_steps=_mapping_scheduler.effective_spf
                    )
                    tb_writer.add_scalar(
                        "streaming/assim_shadow/would_suppress_insertion",
                        int(_shadow_decision.would_suppress_insertion), iteration)
                    tb_writer.add_scalar(
                        "streaming/assim_shadow/would_reduce_steps",
                        int(_shadow_decision.would_reduce_steps), iteration)
                    tb_writer.add_scalar(
                        "streaming/assim_shadow/would_be_steps",
                        _shadow_decision.would_be_steps, iteration)
                    tb_writer.add_scalar(
                        "streaming/assim_shadow/would_bias_replay_weak",
                        int(_shadow_decision.would_bias_replay_weak), iteration)
                    tb_writer.add_scalar(
                        "streaming/assim_shadow/would_suppress_growth",
                        int(_shadow_decision.would_suppress_growth), iteration)
                # --- end assimilation-pressure telemetry ---

                tb_writer.add_scalar("streaming/keyframe_window",
                                     len(streaming_scene.get_local_cameras()), iteration)
                tb_writer.add_scalar("streaming/replay_buffer",
                                     len(streaming_scene._replay_buffer), iteration)
                if scheduler.frames_dropped > 0:
                    tb_writer.add_scalar("streaming/frames_dropped", scheduler.frames_dropped, iteration)
                _vf = render_pkg.get("visibility_filter")
                if _vf is not None and gaussians.get_xyz.shape[0] > 0:
                    tb_writer.add_scalar(
                        "streaming/visible_fraction",
                        _vf.float().mean().item(),
                        iteration,
                    )
                    tb_writer.add_scalar(
                        "streaming/dead_fraction",
                        (gaussians.get_opacity.squeeze(-1) < 0.05).float().mean().item(),
                        iteration,
                    )
                    # Track anisotropy: mean ratio of max/min scale (healthy = near 1–3)
                    with torch.no_grad():
                        _s = gaussians.get_scaling
                        _aniso = (_s.max(dim=-1).values / _s.min(dim=-1).values.clamp(min=1e-6)).mean()
                    tb_writer.add_scalar("streaming/anisotropy_mean", _aniso.item(), iteration)
                # ---- Seed-fade metrics (ablation) -------------------------------
                # Render the fixed bootstrap seed view + track the bootstrap cohort
                # (birth_frame==0) opacity/alive-count. A healthy run keeps these
                # FLAT across the trajectory; the "implosion" shows as a mid-traj dip
                # in seed_psnr / seed mean_opacity that only recovers post-ingestion.
                if _seed_cam is None and getattr(streaming_scene, "train_cameras", None):
                    _seed_cam = streaming_scene.train_cameras[0]
                if _seed_cam is not None:
                    try:
                        with torch.no_grad():
                            _seed_img = render(_seed_cam, gaussians, pipe, background)["render"]
                            _seed_gt = score_tensor_from_camera(_seed_cam, device=_seed_img.device)
                            _seed_score, _seed_gt_score, _seed_transform = (
                                apply_declared_score_clamp_pair(
                                    _seed_img,
                                    _seed_gt,
                                    context="streaming_seed_psnr",
                                )
                            )
                            _seed_psnr_val = psnr(_seed_score, _seed_gt_score).mean().item()
                            tb_writer.add_scalar("optimization/seed_psnr", _seed_psnr_val, iteration)
                            tb_writer.add_scalar(
                                "optimization/seed_score_clamp_fraction",
                                _seed_transform["render"]["fraction_changed"],
                                iteration,
                            )
                            # Record the resolution this sample was measured
                            # at (issue #346 sub-finding). The seed camera is
                            # a TRAINING camera, so under multires its
                            # original_image/image_height/image_width are
                            # rebound to the active tier's cached tensors by
                            # bind_resolution_tier -- but only on iterations
                            # where it happens to be the sampled viewpoint.
                            # Its evaluation resolution is therefore a side
                            # effect of replay sampling, and PSNR is not
                            # comparable across image sizes. Read the rendered
                            # image itself rather than the camera attributes
                            # so this is the resolution actually measured.
                            _seed_res = (int(_seed_img.shape[-2]), int(_seed_img.shape[-1]))
                            tb_writer.add_scalar("optimization/seed_psnr_height", _seed_res[0], iteration)
                            tb_writer.add_scalar("optimization/seed_psnr_width", _seed_res[1], iteration)
                            _fade_was_active = seed_fade_monitor.fade_active
                            seed_fade_monitor.update(iteration, _seed_psnr_val, resolution=_seed_res)
                            if seed_fade_monitor.fade_active and not _fade_was_active:
                                print(
                                    f"[SEED-FADE WARNING] step={iteration} seed_psnr={_seed_psnr_val:.2f}dB "
                                    f"dropped {seed_fade_monitor._running_max - _seed_psnr_val:.2f}dB below "
                                    f"running max ({seed_fade_monitor._running_max:.2f}dB) — "
                                    + ("OBSERVE-ONLY, controller NOT engaged"
                                       if seed_fade_monitor.observe_only
                                       else "controller engaged")
                                    + f" [episode #{len(seed_fade_monitor.episodes)}]",
                                    flush=True,
                                )
                            tb_writer.add_scalar(
                                "streaming/seed_fade_active",
                                int(seed_fade_monitor.fade_active), iteration)
                            # issue #282: the detector's own reference state
                            # and the scale it ACTUALLY applied. Previously
                            # only `seed_fade_active` was logged, so a fade's
                            # depth, its reference, and the applied
                            # suppression all had to be inferred post-hoc
                            # from source code + the one-line stdout warning.
                            _run_max = seed_fade_monitor._running_max
                            if _run_max != float("-inf"):
                                tb_writer.add_scalar(
                                    "streaming/seed_fade/running_max_db",
                                    _run_max, iteration)
                                tb_writer.add_scalar(
                                    "streaming/seed_fade/gap_db",
                                    _run_max - _seed_psnr_val, iteration)
                            tb_writer.add_scalar(
                                "streaming/seed_fade/below_tolerance_streak",
                                seed_fade_monitor._below_tolerance_streak, iteration)
                            tb_writer.add_scalar(
                                "streaming/seed_fade/episode_count",
                                len(seed_fade_monitor.episodes), iteration)
                            if seed_fade_monitor.last_applied_scale is not None:
                                tb_writer.add_scalar(
                                    "streaming/seed_fade/applied_suppression_scale",
                                    seed_fade_monitor.last_applied_scale, iteration)
                            tb_writer.add_scalar(
                                "streaming/seed_fade/observe_only",
                                int(seed_fade_monitor.observe_only), iteration)
                    except Exception as _seed_measurement_error:
                        seed_fade_monitor.measurement_failures += 1
                        seed_fade_monitor.last_measurement_failure = repr(
                            _seed_measurement_error
                        )
                        if tb_writer:
                            tb_writer.add_scalar(
                                "streaming/seed_fade/measurement_failures",
                                seed_fade_monitor.measurement_failures,
                                iteration,
                            )
                        print(
                            "[seed_fade] measurement failed; verdict is indeterminate: "
                            f"{_seed_measurement_error!r}",
                            flush=True,
                        )
                    if hasattr(gaussians, "birth_frame") and gaussians.birth_frame.numel():
                        # KNOWN-CONTAMINATED (issue #282, verified 2026-08-13):
                        # `birth_frame == 0` does NOT select the bootstrap
                        # cohort. utils/strategies/mcmc/proposal.py's
                        # append_running_state() pads `birth_frame` with ZEROS
                        # for every MCMC-grown Gaussian (it pads `birth_iter`
                        # with the real iteration, but birth_frame is in the
                        # generic zero-pad loop), so this mask collects
                        # bootstrap + all MCMC growth. Measured on two real fr3
                        # runs: this count is 97-99.7% of the TOTAL population
                        # (issue278mode_iterbased_spf20: 421920 of 423171 at
                        # iter 75100). These two scalars are therefore
                        # whole-population statistics wearing a seed-cohort
                        # label; kept only for continuity with historical runs.
                        # Use the streaming/seed_cohort/* pair below instead.
                        _bm = gaussians.birth_frame == 0
                        _nb = int(_bm.sum())
                        tb_writer.add_scalar("streaming/seed/alive_count", _nb, iteration)
                        if _nb > 0:
                            tb_writer.add_scalar(
                                "streaming/seed/mean_opacity",
                                float(gaussians.get_opacity[_bm].mean()), iteration)
                    _bi = getattr(gaussians, "birth_iter", None)
                    if _bi is not None and _bi.numel() == gaussians.get_xyz.shape[0]:
                        # Better (not perfect) bootstrap-cohort mask. Unlike
                        # birth_frame, birth_iter is NOT zero-padded on growth:
                        # MCMC growth pads it with the real iteration, depth
                        # insertion passes birth_iter=iteration, and relocation
                        # rewrites the revived slot's birth_iter -- so a
                        # relocated/reseeded slot correctly leaves the cohort.
                        # CAVEAT (issue #282): _reset_running_state() zeroes
                        # birth_iter for the WHOLE population, and load_ply()
                        # calls it (a PLY carries no lineage buffers), as does
                        # restore() -> training_setup() before re-applying
                        # streaming_state. A .pth restore re-applies the saved
                        # birth_iter and is fine; a PLY-loaded model, or a
                        # restore whose buffer is skipped on a size mismatch,
                        # collapses the whole population into this cohort just
                        # like birth_frame does. An exact cohort needs
                        # gaussian_uid < n_bootstrap or a dedicated immutable
                        # buffer -- follow-up, not done here.
                        _cm = _bi == 0
                        _nc = int(_cm.sum())
                        tb_writer.add_scalar("streaming/seed_cohort/alive_count", _nc, iteration)
                        tb_writer.add_scalar(
                            "streaming/seed_cohort/population_fraction",
                            _nc / max(1, gaussians.get_xyz.shape[0]), iteration)
                        if _nc > 0:
                            tb_writer.add_scalar(
                                "streaming/seed_cohort/mean_opacity",
                                float(gaussians.get_opacity[_cm].mean()), iteration)
                    # EXACT cohort (issue #283 -> wired up for #346). The two
                    # blocks above are both approximations kept for continuity
                    # with historical runs; this one is exact by construction
                    # (contiguous bootstrap uid block AND root_uid == own uid
                    # AND birth_method == BOOTSTRAP, so a relocated capacity
                    # slot correctly leaves the cohort) or absent. #283 built
                    # seed_cohort_mask() but wired no consumer, so until now
                    # every run -- including future ones -- logged only the
                    # approximate masks. Unavailability is reported ONCE, loudly,
                    # rather than silently degrading to the approximation.
                    from eligsir.utils.gaussian_lineage import (
                        SeedCohortUnavailable, seed_cohort_mask, seed_lineage_mask,
                        seed_uid_range,
                    )

                    try:
                        # Built from the primitives rather than
                        # seed_cohort_summary() so the cohort mask is evaluated
                        # ONCE and reused for the opacity below -- this runs at
                        # TB-logging cadence inside the training loop, and the
                        # summary helper would re-derive it.
                        _lo, _hi = seed_uid_range(gaussians)
                        _em = seed_cohort_mask(gaussians)
                        _n_alive = int(_em.sum().item())
                        tb_writer.add_scalar(
                            "streaming/seed_cohort_exact/alive_count",
                            _n_alive, iteration)
                        tb_writer.add_scalar(
                            "streaming/seed_cohort_exact/lineage_alive_count",
                            int(seed_lineage_mask(gaussians).sum().item()), iteration)
                        tb_writer.add_scalar(
                            "streaming/seed_cohort_exact/survival_fraction",
                            (_n_alive / (_hi - _lo)) if _hi > _lo else 0.0, iteration)
                        if _n_alive > 0:
                            tb_writer.add_scalar(
                                "streaming/seed_cohort_exact/mean_opacity",
                                float(gaussians.get_opacity[_em].mean()), iteration)
                    except SeedCohortUnavailable as _sc_exc:
                        if not getattr(gaussians, "_seed_cohort_exact_warned", False):
                            gaussians._seed_cohort_exact_warned = True
                            print(
                                "[seed_cohort] WARNING: streaming/seed_cohort_exact/* "
                                "will NOT be logged for this run -- "
                                f"{_sc_exc}. Only the APPROXIMATE "
                                "streaming/seed_cohort/* (birth_iter == 0) and "
                                "streaming/seed/* (birth_frame == 0) masks are "
                                "available; do not read either as a real seed "
                                "cohort (issue #282/#346).",
                                flush=True,
                            )
                    except Exception as _sc_err:
                        # Defence in depth: this block has never executed inside
                        # a real training loop (no GPU was available when it was
                        # written), and it does tensor work -- a dtype/device
                        # mismatch on the mask `&` would raise a bare
                        # RuntimeError, NOT SeedCohortUnavailable, and would kill
                        # a multi-hour run at a logging tick. Telemetry must
                        # never be able to do that; every neighbouring optional
                        # TB block in this function is guarded the same way
                        # (e.g. the seed-fade and partition-PSNR blocks).
                        if not getattr(gaussians, "_seed_cohort_exact_err_warned", False):
                            gaussians._seed_cohort_exact_err_warned = True
                            print(
                                "[seed_cohort] WARNING: streaming/seed_cohort_exact/* "
                                "raised an UNEXPECTED error and was SKIPPED for this "
                                f"logging tick: {_sc_err!r}. Training continues; the "
                                "exact-cohort scalars may be missing or sparse for "
                                "this run. This warning prints once.",
                                flush=True,
                            )

                # GLOBAL multi-view sentinel PSNR. The seed view is ONE (origin) view —
                # it can read "not imploded" (>12 dB) while the scene is globally poor.
                # This renders a FIXED-stride subset of ALL admitted keyframes (spanning
                # the whole visited trajectory) vs their own GT and logs mean/min/p10, so
                # the report reflects WHOLE-scene quality, not just the origin. Coarse
                # cadence (~every 500 iters, capped view count) keeps it cheap.
                _gpsnr_interval = max(500, scalar_log_interval * 20)
                _train_cams = getattr(streaming_scene, "train_cameras", None)
                if (_train_cams and iteration % _gpsnr_interval == 0):
                    streaming_scene._global_psnr_measurement_attempts = (
                        int(getattr(
                            streaming_scene,
                            "_global_psnr_measurement_attempts",
                            0,
                        )) + 1
                    )
                    try:
                        _cap = 24
                        _stride = max(1, len(_train_cams) // _cap)
                        _sent = _train_cams[::_stride][:_cap]
                        with torch.no_grad():
                            _vals = []
                            _global_score_changed = 0
                            _global_score_total = 0
                            for _c in _sent:
                                _im = render(_c, gaussians, pipe, background)["render"]
                                _gt = score_tensor_from_camera(_c, device=_im.device)
                                _score_im, _score_gt, _score_transform = (
                                    apply_declared_score_clamp_pair(
                                        _im,
                                        _gt,
                                        context="streaming_global_psnr",
                                    )
                                )
                                _vals.append(float(psnr(_score_im, _score_gt).mean().item()))
                                _global_score_changed += _score_transform["render"]["samples_changed"]
                                _global_score_total += _score_transform["render"]["samples_total"]
                        if _vals:
                            streaming_scene._global_psnr_measurement_successes = (
                                int(getattr(
                                    streaming_scene,
                                    "_global_psnr_measurement_successes",
                                    0,
                                )) + 1
                            )
                            _va = np.asarray(_vals, dtype=np.float32)
                            tb_writer.add_scalar("optimization/global_psnr", float(_va.mean()), iteration)
                            tb_writer.add_scalar("optimization/global_psnr_min", float(_va.min()), iteration)
                            tb_writer.add_scalar("optimization/global_psnr_p10", float(np.percentile(_va, 10)), iteration)
                            tb_writer.add_scalar("optimization/global_psnr_nviews", len(_vals), iteration)
                            tb_writer.add_scalar(
                                "optimization/global_score_clamp_fraction",
                                _global_score_changed / max(1, _global_score_total),
                                iteration,
                            )

                            # ---- PER-PARTITION sentinel PSNR --------------------
                            # Bin each sentinel view into its deterministic
                            # trajectory partition (SAME helpers/denominator as the
                            # post-training report) and log per-partition means so
                            # per-segment lifecycle (build → fade → oscillate) is
                            # visible over time, not just as a final snapshot.
                            try:
                                from eligsir.utils.partition_eval import (
                                    PARTITIONS as _P_K,
                                    camera_source_index as _p_src,
                                    partition_id as _p_id,
                                    resolve_max_source_index as _p_denom,
                                )
                                if _psent["max_src"] is None:
                                    _scene_src = (os.environ.get("SCENE_SOURCE", "")
                                                  or getattr(args, "source_path", ""))
                                    _ms, _ds = _p_denom(_scene_src, _train_cams)
                                    _psent["max_src"], _psent["denom_src"] = _ms, _ds
                                    if _ds != "registry" and not _psent["warned"]:
                                        _psent["warned"] = True
                                        print("[partition_psnr] WARNING: scene not in "
                                              "eval/scenes.json registry — falling back to "
                                              "max admitted source index as partition "
                                              "denominator (bins are run-dependent)",
                                              flush=True)
                                _bins = {i: [] for i in range(_P_K)}
                                for _c, _v in zip(_sent, _vals):
                                    _si = _p_src(_c)
                                    if _si is not None:
                                        _bins[_p_id(_si, _psent["max_src"])].append(_v)
                                for _pi, _bv in _bins.items():
                                    if _bv:
                                        tb_writer.add_scalar(
                                            f"optimization/partition_psnr/p{_pi}",
                                            float(np.mean(_bv)), iteration)
                            except Exception:
                                pass

                            # ---- ADAPTIVE-CAPACITY controller v1 (brake-only) ----
                            # Uses the RELIABLE whole-scene global PSNR (just computed)
                            # vs N. Growth stays permissive (cap=cap_max) until marginal
                            # utility dPSNR/dN goes sustainedly NEGATIVE (overgrowth — the
                            # empirically-validated brake signal), then the growth cap is
                            # frozen at current N. Releases back to cap_max when utility
                            # recovers. kitchen1 (u>0) never brakes; floor3 (u<0) does.
                            if getattr(args, "streaming_adaptive_capacity", False):
                                _gp = float(_va.mean())
                                _nn = int(gaussians.get_xyz.shape[0])
                                _pp, _pn = _adcap["prev_p"], _adcap["prev_n"]
                                if _pp is not None and abs(_nn - _pn) >= int(getattr(args, "streaming_adcap_min_dN", 200)):
                                    _u = (_gp - _pp) / float(_nn - _pn)
                                    _a = float(getattr(args, "streaming_adcap_ema", 0.5))
                                    _adcap["u_ema"] = _a * _u + (1 - _a) * _adcap["u_ema"]
                                    _ue = _adcap["u_ema"]
                                    _hk = int(args.streaming_adcap_hysteresis)
                                    if _ue < float(getattr(args, "streaming_adcap_u_brake", -1e-5)):
                                        _adcap["brake_streak"] += 1; _adcap["release_streak"] = 0
                                        if _adcap["brake_streak"] >= _hk:
                                            args._adaptive_growth_cap = _nn  # freeze growth here
                                            _adcap["state"] = "BRAKED"
                                    elif _ue > float(getattr(args, "streaming_adcap_u_release", 1e-5)):
                                        _adcap["release_streak"] += 1; _adcap["brake_streak"] = 0
                                        if _adcap["release_streak"] >= _hk:
                                            args._adaptive_growth_cap = int(args.cap_max)
                                            _adcap["state"] = "GROW"
                                    else:
                                        _adcap["brake_streak"] = 0; _adcap["release_streak"] = 0
                                    if tb_writer:
                                        tb_writer.add_scalar("adcap/u_ema", _adcap["u_ema"], iteration)
                                        tb_writer.add_scalar("adcap/growth_cap", int(args._adaptive_growth_cap), iteration)
                                        tb_writer.add_scalar("adcap/state_braked", 1.0 if _adcap["state"] == "BRAKED" else 0.0, iteration)
                                    print(f"[adcap] step={iteration} global_psnr={_gp:.2f} N={_nn} "
                                          f"u_ema={_adcap['u_ema']:.2e} state={_adcap['state']} "
                                          f"growth_cap={int(args._adaptive_growth_cap)}", flush=True)
                                _adcap["prev_p"], _adcap["prev_n"] = _gp, _nn
                    except Exception as _global_psnr_error:
                        streaming_scene._global_psnr_measurement_failures = (
                            int(getattr(
                                streaming_scene,
                                "_global_psnr_measurement_failures",
                                0,
                            )) + 1
                        )
                        streaming_scene._global_psnr_last_failure = repr(
                            _global_psnr_error
                        )
                        if tb_writer:
                            tb_writer.add_scalar(
                                "optimization/global_psnr_measurement_failures",
                                streaming_scene._global_psnr_measurement_failures,
                                iteration,
                            )
                        print(
                            "[global_psnr] measurement failed; adaptive-capacity "
                            f"controller skipped this tick: {_global_psnr_error!r}",
                            flush=True,
                        )

                # ---- LOCAL-CAPACITY telemetry (instrumentation only; flag-gated) ---
                # Marginal-utility signal for a future per-region adaptive capacity
                # controller: does adding Gaussians to the region around the robot's
                # latest pose still buy PSNR (dPSNR/dN > 0), or has it saturated /
                # gone negative (overgrowth)? Records local PSNR, N in-region, #views,
                # region cell, and finite-difference dPSNR/dN. Default OFF => no-op.
                if (getattr(args, "streaming_local_capacity_telemetry", False)
                        and _train_cams and iteration % _gpsnr_interval == 0):
                    try:
                        _lc_R = float(getattr(args, "streaming_local_cap_radius", 5.0))
                        _lc_cell = float(getattr(args, "streaming_local_cap_region_m", 5.0))
                        _robot_c = _train_cams[-1].camera_center.detach().reshape(3)
                        # In-region Gaussians (N_local).
                        _xyz = gaussians.get_xyz.detach()
                        _n_local = int((torch.linalg.norm(_xyz - _robot_c, dim=-1) < _lc_R).sum())
                        # Nearby keyframes = the local eval set; PSNR over them vs GT.
                        _local_cams = [
                            _c for _c in _train_cams
                            if float(torch.linalg.norm(
                                _c.camera_center.detach().reshape(3) - _robot_c)) < _lc_R
                        ]
                        _lp = []
                        _local_score_changed = 0
                        _local_score_total = 0
                        with torch.no_grad():
                            for _c in _local_cams[:16]:
                                _im = render(_c, gaussians, pipe, background)["render"]
                                _gt = score_tensor_from_camera(_c, device=_im.device)
                                _score_im, _score_gt, _score_transform = (
                                    apply_declared_score_clamp_pair(
                                        _im,
                                        _gt,
                                        context="streaming_local_psnr",
                                    )
                                )
                                _lp.append(float(psnr(_score_im, _score_gt).mean().item()))
                                _local_score_changed += _score_transform["render"]["samples_changed"]
                                _local_score_total += _score_transform["render"]["samples_total"]
                        if _lp:
                            _lpm = float(np.mean(_lp))
                            # Region cell id = quantized robot position.
                            _cell = tuple(int(np.floor(float(_robot_c[_k]) / _lc_cell)) for _k in range(3))
                            # Finite-difference marginal utility vs this cell's last sample.
                            _prev = _local_cap_prev.get(_cell)
                            _dpsnr_dn = None
                            if _prev is not None:
                                _dN = _n_local - _prev[1]
                                if abs(_dN) >= 1:
                                    _dpsnr_dn = (_lpm - _prev[0]) / float(_dN)
                            _local_cap_prev[_cell] = (_lpm, _n_local)
                            if tb_writer:
                                tb_writer.add_scalar("local/psnr_mean", _lpm, iteration)
                                tb_writer.add_scalar(
                                    "local/score_clamp_fraction",
                                    _local_score_changed / max(1, _local_score_total),
                                    iteration,
                                )
                                tb_writer.add_scalar("local/n_gaussians", _n_local, iteration)
                                tb_writer.add_scalar("local/n_views", len(_local_cams), iteration)
                                if _dpsnr_dn is not None:
                                    tb_writer.add_scalar("local/dpsnr_dn", _dpsnr_dn, iteration)
                            # Append a per-region row for offline PSNR-vs-N analysis.
                            import json as _json
                            with open(os.path.join(args.model_path, "local_capacity.jsonl"), "a") as _f:
                                _f.write(_json.dumps({
                                    "optimization_step": int(iteration),
                                    "iter": int(iteration),  # deprecated alias, see docs/DEPRECATIONS.md
                                    "region": list(_cell),
                                    "center": [float(_robot_c[_k]) for _k in range(3)],
                                    "local_psnr": _lpm,
                                    "n_local": _n_local,
                                    "n_views": len(_local_cams),
                                    "n_total": int(_xyz.shape[0]),
                                    "dpsnr_dn": _dpsnr_dn,
                                }) + "\n")
                    except Exception as _lce:
                        if iteration % (_gpsnr_interval * 10) == 0:
                            print(f"[local-cap] telemetry failed: {_lce}", flush=True)

                if _depth_loss_val is not None:
                    tb_writer.add_scalar("optimization/depth_loss", _depth_loss_val, iteration)
                if _free_loss_val is not None:
                    tb_writer.add_scalar("optimization/free_space_loss", _free_loss_val, iteration)
                if _anchor_loss_val is not None:
                    tb_writer.add_scalar("optimization/anchor_loss", _anchor_loss_val, iteration)
                if _mature_anchor_val is not None:
                    tb_writer.add_scalar("optimization/mature_anchor_loss", _mature_anchor_val, iteration)
                # 2.4 soft anchor loss (distance-weighted, bubble-as-local-bounds)
                if _soft_anchor_val is not None:
                    tb_writer.add_scalar("optimization/soft_anchor_loss", _soft_anchor_val, iteration)
                # P0.4 two-regime sampler: 1 while the idle/consolidation regime
                # is active (sampler draws globally), 0 during ingest.
                if getattr(args, "streaming_consolidation_sampler", "off") != "off":
                    _consol_on = int(
                        getattr(streaming_scene, "_steps_since_new_frame", 0)
                        > int(getattr(args, "streaming_consolidation_idle_steps", 500))
                    )
                    tb_writer.add_scalar("streaming/consolidation_active", _consol_on, iteration)
                # 5A parallax gate: cumulative rejections
                if getattr(args, "streaming_kf_parallax_gate", False) and _keyframe_admitter is not None:
                    tb_writer.add_scalar(
                        "streaming/kf_rej_parallax",
                        int(getattr(_keyframe_admitter, "_keyframes_rejected_parallax", 0)),
                        iteration,
                    )
                # Lifecycle state counts (when enabled)
                if _lifecycle_enabled and hasattr(gaussians, "lifecycle_state"):
                    lc = gaussians.lifecycle_state
                    # Issue #279: four `int(tensor)` casts were four separate
                    # host readbacks (four implicit device syncs) for one
                    # 4-bucket histogram. Stack the four GPU-side sums and read
                    # them back once. The four comparisons/sums are unchanged --
                    # only the number of device->host crossings drops 4 -> 1 --
                    # so the logged values are identical by construction, with
                    # no bincount-style assumption about the value range.
                    _lc_counts = torch.stack(
                        [(lc == _s).sum() for _s in (0, 1, 2, 3)]
                    ).tolist()
                    tb_writer.add_scalar("streaming/lifecycle/n_provisional", int(_lc_counts[0]), iteration)
                    tb_writer.add_scalar("streaming/lifecycle/n_young", int(_lc_counts[1]), iteration)
                    tb_writer.add_scalar("streaming/lifecycle/n_mature", int(_lc_counts[2]), iteration)
                    tb_writer.add_scalar("streaming/lifecycle/n_frozen", int(_lc_counts[3]), iteration)

            # Viewer metrics + Gaussian map (every scalar_log_interval)
            if _viewer_name != "none" and iteration % scalar_log_interval == 0:
                try:
                    # Issue #279: reuse the shared readback computed under
                    # `should_log` above when it is available. The fallback is
                    # the original inline computation, so a viewer-only run
                    # (no TensorBoard writer) behaves exactly as before --
                    # including keeping the PSNR call inside this try/except.
                    _psnr_val = (
                        _metric_psnr if _metric_psnr is not None
                        else psnr(
                            *apply_declared_score_clamp_pair(
                                image,
                                gt_image,
                                context="streaming_viewer_psnr",
                            )[:2]
                        ).mean().item()
                    )
                    _viewer.log_metrics({
                        "psnr": _psnr_val,
                        "l1_loss": _metric_l1 if _metric_l1 is not None else Ll1.item(),
                        "gaussian_count": float(gaussians.get_xyz.shape[0]),
                        "frames_ingested": float(n_frames_ingested),
                    }, iteration)
                    # Milestone 2: publish OnlineController / scheduler_stats.
                    # The viewer-side gate (scalar_log_interval) throttles the
                    # outer block; the controller-specific flag
                    # ``streaming_online_rviz_publish_interval`` further throttles
                    # the controller diagnostic publish so the operator can run
                    # viewer snapshots at 100 iters but get a denser controller
                    # stream (e.g. every 10 iters) for RViz.
                    if _online_controller is not None and bool(getattr(args, "streaming_online_publish_stats", True)):
                        _rviz_interval = max(
                            1,
                            int(
                                getattr(
                                    _online_controller,
                                    "_rviz_publish_interval",
                                    getattr(args, "streaming_online_rviz_publish_interval", 10),
                                )
                            ),
                        )
                        if iteration % _rviz_interval == 0:
                            try:
                                if hasattr(_viewer, "log_controller_stats"):
                                    _viewer.log_controller_stats(_online_controller.stats, iteration)
                                _ctrl_stats = _online_controller.stats
                                _viewer.log_metrics({
                                    "controller/map_age_s": _ctrl_stats.map_age_s,
                                    "controller/optimization_steps": float(_ctrl_stats.train_steps),
                                    "controller/mode_" + _ctrl_stats.mode: 1.0,
                                    "controller/input_fps": _ctrl_stats.measured_input_fps,
                                    "controller/pose_fps": _ctrl_stats.pose_fps,
                                    "controller/optimization_step_dt_ema_s": _ctrl_stats.iter_dt_ema_s,
                                    "controller/confidence": _ctrl_stats.confidence,
                                    "controller/pause": 1.0 if _ctrl_stats.pause else 0.0,
                                    "controller/burst": 1.0 if _ctrl_stats.burst else 0.0,
                                    "controller/quality_idle_eligible": 1.0 if _ctrl_stats.quality_idle_eligible else 0.0,
                                }, iteration)
                            except Exception:
                                pass  # controller diagnostics are best-effort
                    if tb_writer is not None and _online_controller is not None:
                        try:
                            for _k, _v in _online_controller.stats.to_tensorboard_dict().items():
                                tb_writer.add_scalar(_k, _v, iteration)
                        except Exception:
                            pass
                    if hasattr(_viewer, "log_gaussian_model") and _viewer_name == "ros2_rviz":
                        _viewer.log_gaussian_model(map_backend.read_view(), iteration)
                    elif _viewer_is_canonical and _map_output_lane is not None:
                        # #1200: one bounded, versioned OUTPUT lane replaces
                        # the old O(N) randperm snapshot producer.  This
                        # producer performs only O(cap) device gathers and an
                        # event-backed transfer-owner readback; the viewer
                        # drain thread publishes completed immutable output.
                        _map_cfg = getattr(_viewer.canonical_viewer, "config", None)
                        _map_hz = float(getattr(_map_cfg, "map_max_hz", 0.5) or 0.0)
                        _map_now = time.monotonic()
                        if _map_hz <= 0.0 or (
                            _map_now - _map_output_last_wall_s
                        ) >= (1.0 / _map_hz):
                            _parameter_version = int(
                                _streaming_optimization_engine.session.parameter_version
                                if _streaming_optimization_engine is not None
                                else iteration
                            )
                            if _map_output_lane.try_submit(
                                map_backend,
                                parameter_version=_parameter_version,
                                iteration=int(iteration),
                            ):
                                _map_output_last_wall_s = _map_now
                                _viewer.notify_map_output()
                    elif not _viewer_is_canonical:
                        # #571/#572: the canonical (viser) path's map-snapshot
                        # readback happens above via log_gaussian_map_tensors
                        # (async, bounded); this legacy synchronous
                        # .cpu().numpy() readback is for the other backends
                        # (rerun, none) which never got the async path.
                        with torch.no_grad():
                            _gpos = gaussians.get_xyz.detach().cpu().numpy()
                            _gcol = None
                            if hasattr(gaussians, "get_features"):
                                _gsh0 = gaussians.get_features[:, :1, :].squeeze(1)
                                _gcol = (0.5 + 0.282095 * _gsh0).clamp(0, 1).detach().cpu().numpy()
                            elif hasattr(gaussians, "params") and "sh0" in gaussians.params:
                                _gsh0 = gaussians.params["sh0"][:, :1, :].squeeze(1)
                                _gcol = (0.5 + 0.282095 * _gsh0).clamp(0, 1).detach().cpu().numpy()
                        _viewer.log_gaussian_map(_gpos, _gcol, None, iteration)
                    # Log rendered view
                    if _viewer_is_canonical and hasattr(_viewer, "log_training_pair_tensors"):
                        # Canonical path (#570/#571/#572): hand over the RAW
                        # tensors already computed for THIS iteration's loss
                        # (image=render, gt_image=target) -- no extra
                        # rasterization, and no synchronous .cpu().numpy() on
                        # the training thread. The staging ring inside the
                        # adapter does the D2H asynchronously off-thread.
                        _cam_id = int(
                            getattr(viewpoint_cam, "colmap_id", None)
                            or getattr(viewpoint_cam, "uid", 0)
                            or 0
                        )
                        _is_new_cam = viewpoint_cam is getattr(streaming_scene, "current_camera", None)
                        # Viser training-pair panels are (H, W, C); gt_image/
                        # image are (C, H, W) like every other render tensor
                        # in this file. Permute is a GPU-side metadata op (no
                        # sync) -- the actual D2H happens async inside the
                        # staging ring (see canonical_adapter.py).
                        _src_frame_id = getattr(
                            getattr(viewpoint_cam, "_streaming_frame", None), "index", None
                        )
                        _viewer.log_training_pair_tensors(
                            target=gt_image.permute(1, 2, 0),
                            render=image.permute(1, 2, 0),
                            camera_id=_cam_id,
                            iteration=int(iteration),
                            source_frame_id=_src_frame_id,
                            camera_role=("NEW" if _is_new_cam else "OTHER"),
                            n_gaussians=int(gaussians.get_xyz.shape[0]),
                            psnr_db=float(_psnr_val),
                            frames_ingested=int(n_frames_ingested),
                        )
                    else:
                        _rimg = image.permute(1, 2, 0).detach().cpu().numpy()
                        _viewer.log_rendered(_rimg, None, iteration)
                except Exception:
                    pass  # viewer errors must never crash training

            # Phase 11: Gaussian force-field loop diagnostic
            if (
                _loop_ff_interval > 0
                and iteration % _loop_ff_interval == 0
                and gaussians.get_xyz.shape[0] > 10
            ):
                try:
                    from loop_closure.gaussian_force_field import run_force_field_diagnostic
                    _ff_result = run_force_field_diagnostic(
                        gaussians, match_radius=_loop_ff_radius, max_rmse=_loop_ff_max_rmse
                    )
                    if tb_writer is not None:
                        import math as _math
                        for _k, _v in _ff_result.to_tensorboard_dict().items():
                            if not _math.isnan(_v):
                                tb_writer.add_scalar(_k, _v, iteration)

                    # Phase 13: add loop edge when force-field signals a loop closure
                    if (
                        _loop_closure_enabled
                        and _pose_graph is not None
                        and getattr(_ff_result, "T_old_current", None) is not None
                        and _ff_result.coherence >= _lc_min_coherence
                        and _ff_result.inliers >= _lc_min_inliers
                    ):
                        try:
                            _current_sub = int(
                                getattr(_submap_manager, "current_submap_id", 0)
                            )
                            # old_submap_id is the mode over matched TARGET
                            # Gaussians' actual submap membership (see
                            # ForceFieldResult docstring) -- NOT hardcoded to
                            # submap 0. Guard against a same-submap self-edge
                            # (nothing to correct) and ensure the pose graph
                            # already has this node registered.
                            _old_sub = int(getattr(_ff_result, "old_submap_id", 0))
                            if _old_sub != _current_sub:
                                # Pad with the real SubmapManager anchor, not
                                # identity: world_delta_to_local_edge a few
                                # lines below uses the REAL
                                # world_T_old = _submap_manager.get_submap_transform(_old_sub)
                                # to build that same T_src_dst. Padding this
                                # node with identity here would make those
                                # two anchors disagree for a submap the pose
                                # graph hadn't registered yet via the
                                # odometry path (line ~3326) -- an
                                # identity-conjugated edge built from a
                                # real-anchor-derived T_src_dst, exactly the
                                # anchor-mismatch class flagged in this
                                # session's pose-graph fix. Flagged 2026-08-06
                                # in final review; normally a no-op since
                                # _old_sub is expected to already be
                                # registered by the time a loop-closure
                                # candidate references it, but pad correctly
                                # regardless.
                                while _pose_graph.get_node_count() <= _old_sub:
                                    _pad_idx = _pose_graph.get_node_count()
                                    _pad_anchor = _submap_manager.get_submap_transform(_pad_idx)
                                    _pose_graph.add_node(
                                        _pad_anchor if _pad_anchor is not None else np.eye(4)
                                    )
                                # `_ff_result.T_old_current` is fitted directly
                                # between WORLD-SPACE Gaussian positions
                                # (gaussian_force_field.fit_weighted_se3(src=
                                # current-provisional, tgt=old-mature), both
                                # already in world/global coordinates): it is
                                # a world-space correction delta, NOT a
                                # submap-local pose-graph edge. add_loop_edge
                                # (like add_odometry_edge) expects a LOCAL
                                # T_src_dst edge in the pose-graph's own
                                # convention (see
                                # loop_closure.pose_graph.PoseGraphEdge /
                                # world_delta_to_local_edge docstrings for the
                                # exact, Open3D-verified formula). Convert
                                # explicitly rather than passing the raw
                                # world-space delta straight through.
                                from loop_closure.pose_graph import world_delta_to_local_edge
                                world_T_old = _submap_manager.get_submap_transform(_old_sub)
                                world_T_cur = _submap_manager.get_submap_transform(_current_sub)
                                world_delta_T = _ff_result.T_old_current
                                old_T_cur = world_delta_to_local_edge(
                                    world_T_old, world_T_cur, world_delta_T
                                )
                                _pose_graph.add_loop_edge(
                                    _old_sub,
                                    _current_sub,
                                    old_T_cur,
                                    _ff_result.inliers,
                                    _ff_result.rmse,
                                )
                                if tb_writer is not None:
                                    tb_writer.add_scalar(
                                        "loop_closure/loop_edges_added", 1, iteration
                                    )
                        except Exception:
                            pass  # loop-edge errors must never crash training
                except Exception:
                    pass  # diagnostic must never crash training

            # Phase 13: periodic pose-graph optimization
            if (
                _loop_closure_enabled
                and _pose_graph is not None
                and iteration - _last_optimized_iter >= _lc_optimize_interval
                and len(_pose_graph._edges) > 0
            ):
                try:
                    _lc_debug_enabled = bool(getattr(args, "loop_closure_debug", False))
                    if _lc_debug_enabled:
                        _lc_debug_pre = _pose_graph.debug_snapshot()
                    with _timed("loop_closure_optimize"):
                        _optimised_poses = _pose_graph.optimize()
                    _last_optimized_iter = iteration
                    if _lc_debug_enabled:
                        try:
                            import json as _json_lc
                            with open(os.path.join(args.model_path, "loop_closure_debug.jsonl"), "a") as _lc_f:
                                _lc_f.write(_json_lc.dumps({
                                    "iteration": iteration,
                                    "pre": _lc_debug_pre,
                                    "post_node_translations": [n[:3, 3].tolist() for n in _optimised_poses],
                                }) + "\n")
                        except Exception as _lc_dbg_e:
                            print(f"[loop_closure_debug] logging failed: {_lc_dbg_e}", flush=True)
                    if tb_writer is not None:
                        tb_writer.add_scalar(
                            "loop_closure/nodes_optimized",
                            len(_optimised_poses),
                            iteration,
                        )

                    # Phase 14 + Milestone 5: propagate the optimised
                    # submap poses to cameras and Gaussians via the
                    # PoseCorrector coordinator. The coordinator:
                    #   * counts loop edges
                    #   * updates train_cameras c2w
                    #   * transforms Gaussians (medium / expensive)
                    #   * rate-limits and caps per event
                    if _optimised_poses:
                        try:
                            _all_submap_ids = list(range(len(_optimised_poses)))
                            # Use the public ``get_edge_count`` accessor
                            # rather than poking at ``_pose_graph._edges``
                            # — the private attribute is an implementation
                            # detail and may not be populated in tests.
                            _n_loop_edges = _pose_graph.get_edge_count(loop_only=True)
                            _correction_event = _pose_corrector.propagate(
                                train_cameras=streaming_scene.train_cameras
                                if hasattr(streaming_scene, "train_cameras")
                                else [],
                                submap_manager=_submap_manager,
                                gaussians=gaussians,
                                submap_ids=_all_submap_ids,
                                transforms=_optimised_poses,
                                loop_edges=_n_loop_edges,
                                map_backend=map_backend,
                            )
                            if _correction_event is not None and tb_writer is not None:
                                tb_writer.add_scalar(
                                    "loop_closure/gaussians_corrected",
                                    _correction_event.n_corrected_gaussians,
                                    iteration,
                                )
                                tb_writer.add_scalar(
                                    "loop_closure/cameras_corrected",
                                    _correction_event.n_corrected_cameras,
                                    iteration,
                                )
                                tb_writer.add_scalar(
                                    "loop_closure/loop_edges",
                                    _correction_event.loop_edges,
                                    iteration,
                                )
                            # Non-rigid per-pair merge, ADDITIVE to (not a
                            # replacement for) PoseCorrector's rigid per-submap
                            # apply_submap_correction above, which still runs at
                            # its default MEDIUM mode -- that handles gross
                            # submap-level drift; this cleans up residual near-
                            # field duplicates the rigid transform can't reach
                            # (e.g. duplicates straddling a corrected/uncorrected
                            # submap boundary). See design doc section 5.
                            from loop_closure.submap_correction import apply_gaussian_merge_correction
                            _n_merged = apply_gaussian_merge_correction(
                                gaussians, match_radius=_loop_ff_radius,
                                map_backend=map_backend,
                            )
                            if tb_writer is not None:
                                tb_writer.add_scalar(
                                    "loop_closure/gaussians_merged", _n_merged, iteration
                                )
                        except Exception:
                            pass  # correction must never crash training
                except Exception:
                    pass  # optimization must never crash training

            if (
                (_depth_lock_active or _depth_conflict_active)
                and _depth_lock_report_every > 0
                and iteration % _depth_lock_report_every == 0
            ):
                _depth_stats = depth_lock_geometry_stats(gaussians, args)
                if tb_writer is not None:
                    for _key, _value in _depth_stats.items():
                        tb_writer.add_scalar(f"streaming/depth_lock/{_key}", _value, iteration)
                print(
                    "[streaming-depth-lock] "
                    f"step={iteration} candidates={_depth_stats.get('candidate_count', 0)} "
                    f"locked={_depth_stats.get('locked_count', 0)} "
                    f"conflicts={_depth_stats.get('conflict_count_nonzero', 0)} "
                    f"radius_p99={_depth_stats.get('radius_p99_m', 0.0):.3f}m "
                    f"drift_gt_5cm={_depth_stats.get('candidate_drift_gt_5cm', 0)}",
                    flush=True,
                )

            if iteration == opt.iterations:
                progress_bar.close()

            # Hard constraint (CLAUDE.md Run Contracts): a test-holdout PSNR-
            # over-time chart must ALWAYS be produced, even for online_mode
            # runs -- unlike the heavy full report below (gated behind the
            # opt-in --streaming_refinement_eval_interval), this only renders
            # test cameras and computes PSNR, so it runs synchronously on the
            # training thread at a fixed interval without the online_mode/
            # rolling-idle-stop gating the heavy path needs.
            _psnr_chart_interval = int(getattr(args, "streaming_psnr_chart_interval", 1000) or 0)
            if (
                _psnr_chart_interval > 0
                and iteration > first_iter
                and iteration % _psnr_chart_interval == 0
                and iteration != opt.iterations
            ):
                from eligsir.utils.psnr_history import record_and_plot as _psnr_record_and_plot
                _lq_test_cams = list(streaming_scene.getTestCameras())
                # Issue #358 Level A: per-microcell local quality history.
                # PURE TELEMETRY, opt-in (--streaming_local_quality_cells,
                # default off). Rides the SAME render pass record_and_plot is
                # about to do via its per_view_hook -- no second render, no
                # per-view sync (see utils/local_quality.py). Holdout pixels
                # are assigned to immutable world cells from FIXED sensor/GT
                # depth + the FIXED holdout pose, never the current map's
                # rendered depth, so the partition can't depend on the
                # reconstruction being measured. Everything is inside
                # try/except (matching the partition-focus block below):
                # telemetry must never take down a run.
                _lq_hook = None
                _lq_acc = None
                if bool(getattr(args, "streaming_local_quality_cells", False)):
                    try:
                        if _local_quality_tracker is None:
                            from eligsir.utils.local_quality import LocalQualityTracker as _LQT
                            _local_quality_tracker = _LQT(
                                cell_size=float(getattr(args, "streaming_local_quality_cell_size", 0.5)),
                                # No getattr fallback: streaming_max_depth is an
                                # unconditional StreamingParams default, always
                                # present on the real parsed args (issue #368 --
                                # old getattr(..., 10.0) fallback drifted from
                                # the real default of 8.0; removed rather than
                                # re-stated).
                                max_depth_m=float(args.streaming_max_depth or 8.0),
                            )
                        _local_quality_tracker.prepare(_lq_test_cams)
                        _lq_acc = _local_quality_tracker.new_accumulator()
                        _lq_hook = (
                            lambda _c, _r, _g, _t=_local_quality_tracker, _a=_lq_acc:
                            _t.observe(_c, _r, _g, _a)
                        )
                    except Exception as e:
                        print(f"[local_quality] tick setup failed at iter={iteration}: {e}", flush=True)
                        _lq_hook = _lq_acc = None
                _psnr_tick = _psnr_record_and_plot(
                    args.model_path, iteration, gaussians, _lq_test_cams,
                    render, pipe, background, per_view_hook=_lq_hook,
                )
                # Issue #398: mid-training PSNR ticks must leave holdout image
                # residency where they found it (see _run_test_eval's note --
                # normally a no-op, kept as the enforcement point).
                streaming_scene.offload_test_camera_images()
                # Issue #698: ONE canonical evaluation-tick record. Consumes the
                # tick record_and_plot JUST produced -- no second holdout render,
                # no per-view .item() loop (utils/eval_tick.py reduces the
                # GPU-resident per-view tensor with a scatter_add_ and does
                # exactly ONE .cpu() sync per tick). Persists the causal clocks
                # (#428 axis names) joined at this exact tick alongside the
                # temporal-partition PSNR vector from that same render, into
                # evaluation_ticks.jsonl. Placed BEFORE the partition-focus block
                # below on purpose: that block promotes/demotes train cameras, so
                # recording after it would join these clocks to a scene state the
                # tick's render never saw. Pure telemetry -- nothing here mutates
                # the model, the cameras, the sampling weights, or which cameras
                # were scored, and the whole thing is exception-swallowed.
                _canonical_eval_tick = None
                if _psnr_tick is not None:
                    try:
                        from eligsir.utils.eval_tick import record_eval_tick as _record_eval_tick
                        _canonical_eval_tick = _record_eval_tick(
                            model_path=args.model_path,
                            iteration=iteration,
                            psnr_tick=_psnr_tick,
                            test_cams=_lq_test_cams,
                            scene_source=(os.environ.get("SCENE_SOURCE", "")
                                          or getattr(args, "source_path", "")),
                            source_frame_index=int(getattr(streaming_scene, "_source_idx", 0) or 0),
                            frames_ingested=int(n_frames_ingested),
                            frames_trained=int(_n_frames_trained),
                            keyframes_admitted=int(getattr(streaming_scene, "_keyframes_admitted", 0) or 0),
                            optimizer_step=int(_clk_optimizer_step_id),
                            supervision_views=int(_clk_supervision_view_count),
                            wall_clock_s=float(time.perf_counter() - _train_t0),
                            accumulated_iteration_wall_s=float(getattr(online_clock, "wall_time_s", 0.0)),
                            last_ingest_iter=int(_last_ingest_iter),
                            # Read directly, NOT via getattr(..., 0): this is an
                            # unconditional ParamGroup default whose real value is
                            # 50000, so a 0 fallback here is the stale-default trap
                            # #363/#368 root-caused -- it can never fire, and if it
                            # ever did it would silently claim the run's mapping-idle
                            # threshold was 0. Line 4580 already reads it directly,
                            # which is what makes the fallback provably dead.
                            mapping_idle_threshold_iters=(
                                int(args.streaming_refinement_iterations or 0) or None
                            ),
                        )
                    except Exception as e:
                        print(f"[eval_tick] record failed at iter={iteration}: {e}", flush=True)
                if _lq_acc is not None and _local_quality_tracker is not None:
                    try:
                        from eligsir.utils.local_quality import append_local_quality_history
                        _lq_rec = _local_quality_tracker.record_tick(
                            iteration,
                            _lq_acc,
                            canonical_evaluation_tick=_canonical_eval_tick,
                        )
                        append_local_quality_history(args.model_path, _lq_rec)
                        print(
                            f"[local_quality] iter={iteration} "
                            f"cells_observed={_lq_rec['n_cells_observed']} "
                            f"established={_lq_rec['n_cells_established']} "
                            f"worst_drop_from_peak={_lq_rec['worst_drop_from_peak_db']:.2f}dB"
                            + (
                                f" INCOMPLETE(no_depth={_lq_rec['cams_without_sensor_depth']} "
                                f"unprepared={_lq_rec['views_skipped_unprepared']} "
                                f"res_mismatch={_lq_rec['views_skipped_resolution_mismatch']} "
                                f"errors={_lq_rec.get('views_skipped_error', 0)})"
                                if _lq_rec["measurement_incomplete"] else ""
                            ),
                            flush=True,
                        )
                    except Exception as e:
                        print(f"[local_quality] record failed at iter={iteration}: {e}", flush=True)
                    try:
                        # Issue #358 Level B evidence: which holdout camera's
                        # FIXED sensor depth landed in which microcell. Pure
                        # geometry (no PSNR, no residual, nothing from the
                        # reconstruction) and rewritten only when the set of
                        # frozen cell maps actually grew, so it costs one small
                        # unique() per camera at most once per new camera --
                        # not per tick. Consumed offline by
                        # scripts/local_region_report.py; nothing reads it
                        # during training.
                        _lq_n_maps = len(_local_quality_tracker._cell_maps)
                        if _lq_n_maps != _lq_covis_written:
                            from eligsir.utils.local_quality import write_covisibility_snapshot
                            write_covisibility_snapshot(
                                args.model_path,
                                _local_quality_tracker.covisibility_snapshot(),
                                _local_quality_tracker.grid.cell_size,
                            )
                            _lq_covis_written = _lq_n_maps
                    except Exception as e:
                        print(f"[local_quality] covisibility snapshot failed at "
                              f"iter={iteration}: {e}", flush=True)
                # Partition focus is a controller, not telemetry. Issue #951
                # therefore resolves and validates its observation role before
                # rendering any control input or applying any action. The
                # scored `_psnr_tick` remains evaluation-only in both normal
                # modes; only explicit scored_holdout_oracle reuses it.
                if _psnr_tick is not None and _partition_focus_enabled:
                    try:
                        from eligsir.utils.partition_eval import (
                            camera_source_index, partition_count,
                            registry_n_source_frames,
                        )
                        from eligsir.utils.psnr_history import (
                            PsnrTickResult as _PsnrTickResult,
                            compute_per_view_test_psnr_gpu as _compute_control_psnr_gpu,
                        )
                        from eligsir.utils.psnr_history_io import (
                            PSNR_REDUCTION_LEGACY_CHW_CHANNEL_MEAN_V1,
                        )
                        from eligsir.utils.score_contract import scientific_psnr as _legacy_control_psnr

                        _scene_src = (
                            os.environ.get("SCENE_SOURCE", "")
                            or getattr(args, "source_path", "")
                        )
                        _n_reg = registry_n_source_frames(_scene_src)
                        _all_train_cams = list(streaming_scene.train_cameras)
                        _all_test_cams = list(_lq_test_cams)
                        _all_idx = [
                            camera_source_index(c)
                            for c in _all_train_cams + _all_test_cams
                        ]
                        _all_idx = [index for index in _all_idx if index is not None]
                        _max_src = (
                            (_n_reg - 1)
                            if (_n_reg and _n_reg > 1)
                            else (max(_all_idx) if _all_idx else 0)
                        )
                        _focus_k = partition_count(_n_reg)

                        def _render_partition_control_tick(_control_cams):
                            _cams, _scores = _compute_control_psnr_gpu(
                                gaussians,
                                list(_control_cams),
                                render,
                                pipe,
                                background,
                                score_fn=_legacy_control_psnr,
                            )
                            return _PsnrTickResult(
                                cams=_cams,
                                psnr_tensor=_scores,
                                mean_psnr=(
                                    float(_scores.mean().item())
                                    if _scores is not None else None
                                ),
                                psnr_reduction=PSNR_REDUCTION_LEGACY_CHW_CHANNEL_MEAN_V1,
                            )

                        _partition_control_roles, _role_observations = (
                            _partition_control_runtime.prepare_tick(
                                training_cameras=_all_train_cams,
                                scored_holdout_cameras=_all_test_cams,
                                scored_holdout_tick=_psnr_tick,
                                render_control_tick=_render_partition_control_tick,
                            )
                        )
                        _active_indices = tuple(
                            index for index in (
                                camera_source_index(cam) for cam in _all_train_cams
                            ) if index is not None
                        )
                        _decision_without_cache = _decide_partition_focus(
                            roles=_partition_control_roles,
                            training_observations=_role_observations["training_derived"],
                            control_probe_observations=_role_observations["control_probe"],
                            scored_holdout_observations=_role_observations["scored_holdout"],
                            max_source_index=_max_src,
                            partition_count=_focus_k,
                            outlier_k=args.streaming_partition_focus_outlier_k,
                            active_training_indices=_active_indices,
                            cached_indices=(),
                            max_promote=0,
                        )
                        _cached_indices = streaming_scene.cached_indices_in_partitions(
                            set(_decision_without_cache.bad_partitions),
                            _max_src,
                            _focus_k,
                        )
                        _decision = _decide_partition_focus(
                            roles=_partition_control_roles,
                            training_observations=_role_observations["training_derived"],
                            control_probe_observations=_role_observations["control_probe"],
                            scored_holdout_observations=_role_observations["scored_holdout"],
                            max_source_index=_max_src,
                            partition_count=_focus_k,
                            outlier_k=args.streaming_partition_focus_outlier_k,
                            active_training_indices=_active_indices,
                            cached_indices=_cached_indices,
                            max_promote=args.streaming_partition_focus_max_promote,
                        )

                        # issue #481(b): promoted frames need the same multires
                        # cache builder as ordinary ingestion.
                        _active_before = set(_active_indices)
                        streaming_scene.set_priority_frame_weights(
                            _decision.priority_weights,
                            resolution_cache_fn=lambda _f: build_frame_resolution_cache(
                                _f,
                                int(getattr(_f, "index", -1)),
                                low_res_divisor=_multires_low_divisor,
                                depthless_rgb_cache=_multires_depthless_rgb_cache,
                                depth_downsample=_multires_depth_downsample,
                            )[0],
                        )
                        _active_after_promote = {
                            camera_source_index(cam)
                            for cam in streaming_scene.train_cameras
                            if camera_source_index(cam) is not None
                        }
                        _actual_promoted = tuple(sorted(
                            _active_after_promote.difference(_active_before)
                        ))
                        _requested_demote = tuple(
                            _decision.demote_indices[:len(_actual_promoted)]
                        )
                        streaming_scene.demote_partitions(set(_requested_demote))
                        _active_after_demote = {
                            camera_source_index(cam)
                            for cam in streaming_scene.train_cameras
                            if camera_source_index(cam) is not None
                        }
                        _actual_demoted = tuple(sorted(
                            set(_requested_demote).difference(_active_after_demote)
                        ))
                        _partition_control_last_decision = _dataclasses.replace(
                            _decision,
                            promote_indices=_actual_promoted,
                            demote_indices=_actual_demoted,
                        )
                        _partition_control_runtime.recorder.record(
                            iteration=iteration,
                            roles=_partition_control_roles,
                            decision=_partition_control_last_decision,
                        )
                        print(
                            "[streaming] partition-focus "
                            f"source={_partition_control_roles.source.value}: "
                            f"low={list(_decision.bad_partitions)} "
                            f"high={list(_decision.good_partitions)} "
                            f"priority={len(_decision.priority_weights)} "
                            f"promoted={len(_actual_promoted)} "
                            f"demoted={len(_actual_demoted)}",
                            flush=True,
                        )
                    except _EvaluationLeakageError:
                        # STOP-THE-LINE correctness contract: never swallow an
                        # overlap and continue training.
                        raise
                    except Exception as e:
                        print(
                            f"[streaming] partition-focus control skipped: {e}",
                            flush=True,
                        )

                # Early stopping on a flat test-PSNR curve (opt-in,
                # --streaming_early_stop_patience_ticks > 0, default off --
                # existing runs are unaffected). Checked at the same cadence
                # as the psnr_over_time.svg tick above, so it costs nothing
                # extra: just a read of the jsonl history just appended to.
                _early_stop_patience = int(getattr(args, "streaming_early_stop_patience_ticks", 0) or 0)
                if (
                    _early_stop_patience > 0
                    and _post_mapping_refinement_steps == 0
                ):
                    from eligsir.utils.psnr_history import check_psnr_plateau as _psnr_plateaued
                    _early_stop_min_delta = float(getattr(args, "streaming_early_stop_min_delta_db", 0.1))
                    if _psnr_plateaued(args.model_path, _early_stop_patience, _early_stop_min_delta):
                        print(
                            f"[streaming] early stop: test PSNR flat within "
                            f"{_early_stop_min_delta:.3f}dB over the last {_early_stop_patience} "
                            f"psnr-chart ticks -- stopping at iteration {iteration} "
                            f"(iterations={opt.iterations}).",
                            flush=True,
                        )
                        break

            # Test evaluation: full comparison report (renders + contact sheet + MP4)
            # --test_iterations is a fixed list, unsuited to online_mode's unbounded
            # rolling-idle-stop termination (the final iteration isn't known ahead of
            # time). --streaming_refinement_eval_interval additionally fires this on
            # a regular cadence for online_mode runs, spanning both the mapping phase
            # and the post-coverage refinement tail, so the resulting sequence of
            # comparison/iter_N/report.json files traces a real test-PSNR-vs-iteration
            # curve usable to find the refinement-length sweet spot (2026-07-04).
            _refinement_eval_interval = int(getattr(args, "streaming_refinement_eval_interval", 0) or 0)
            _is_refinement_eval_tick = (
                _use_rolling_idle_stop
                and _refinement_eval_interval > 0
                and iteration > first_iter
                and iteration % _refinement_eval_interval == 0
            )
            if (iteration in testing_iterations or _is_refinement_eval_tick) and iteration != opt.iterations:
                try:
                    from eligsir.io.snapshots import snapshot_gaussians_for_render
                    with _timed("mid_training_report_snapshot"):
                        _report_snapshot = snapshot_gaussians_for_render(gaussians)

                    def _run_mid_report(snap=_report_snapshot, it=iteration,
                                         last_ingest_iter_snap=_last_ingest_iter,
                                         train_cams=list(streaming_scene.getTrainCameras()),
                                         test_cams=list(streaming_scene.getTestCameras()),
                                         coverage_snapshot=_coverage_snapshot()):
                        from eligsir.utils.comparison_report import write_post_training_report, git_provenance_from_args as _git_provenance_from_args
                        from gaussian_renderer.gsplat_backend import render_batch as _rb_mid
                        try:
                            write_post_training_report(
                                streaming_scene=streaming_scene,
                                model_path=args.model_path,
                                iteration=it,
                                gaussians=snap,
                                train_cams=train_cams,
                                test_cams=test_cams,
                                render_fn=render,
                                pipe=pipe,
                                background=background,
                                tb_writer=tb_writer,
                                log_prefix="streaming_report",
                                batch_render_fn=_rb_mid,
                                # Issue #346 follow-up: snapshotted at closure-creation
                                # time (default-arg trick, same as `it=iteration` above)
                                # -- this runs on a background report-worker thread, so
                                # reading the live `_last_ingest_iter` at call time would
                                # race ahead of what `it` actually captured.
                                last_ingest_iter=last_ingest_iter_snap,
                                mapping_phase_source="live",
                                # Issue #290: mid-training ticks historically
                                # omitted this, so a run killed before its FINAL
                                # report left only reports that could not be
                                # identified as resumed at all -- exactly the
                                # artifacts a preempted/requeued run leaves
                                # behind. Passing it here makes every report a
                                # resumed run writes self-identify.
                                resumed_from_iteration=_resumed_from_iteration,
                                **_streaming_report_kwargs(
                                    coverage_snapshot=coverage_snapshot,
                                ),
                            )
                        except Exception as _e:
                            print(f"[streaming-report] mid-mapping-run report failed step={it}: {_e}", flush=True)

                    _submitted = report_worker.try_submit(
                        _run_mid_report,
                        on_skip=lambda it=iteration: print(
                            f"[streaming-report] step={it}: previous mid-mapping-run report still running, skipping this tick",
                            flush=True,
                        ),
                    )
                    if not _submitted:
                        _run_test_eval(tb_writer, iteration, streaming_scene, gaussians, render, pipe, background, args)
                except Exception as _e:
                    print(f"[streaming-report] mid-mapping-run report snapshot failed step={iteration}: {_e}", flush=True)
                    _run_test_eval(tb_writer, iteration, streaming_scene, gaussians, render, pipe, background, args)

        # ---- Exact causal fork boundary -----------------------------------
        if iteration == _exact_fork_iteration and not _is_causal_fork_resume:
            if _phase_controller.phase is not Phase.MAPPING:
                raise RuntimeError("exact causal fork must occur during mapping")

            # Reports can sample the live model and consume RNG; checkpoint
            # saves retain captured tensor references. Quiesce both before the
            # common fork intervention and capture the RNG/owner envelope only
            # after they have returned.
            report_worker.join()
            save_worker.join()

            _fork_pending_items = len(_insertion_batch)
            _fork_inserted_count = 0
            if _insertion_batch:
                _fork_inserted_count = _flush_insertion_batch(
                    _insertion_batch,
                    gaussians,
                    streaming_scene,
                    iteration,
                    args,
                    map_backend=map_backend,
                )
                _clk_depth_mutation_event_id += 1
                _clk_mutation_event_id += 1
                _cadence.record_insertion_commit(_fork_inserted_count)
                _coordinator.four_clocks.record_map_commit(
                    n_inserted=_fork_inserted_count
                )
                _insertion_batch = []
                _insertion_batch_frames = 0
                _last_flush_iteration = iteration
                total_inserted += _fork_inserted_count

            _fork_boundary_flush = {
                "iteration": int(iteration),
                "pending_items": int(_fork_pending_items),
                "flushed_items": int(_fork_pending_items),
                "inserted_count": int(_fork_inserted_count),
                "work_id": int(_clk_depth_mutation_event_id),
                "optimizer_step_id": int(_clk_optimizer_step_id),
                "mutation_event_id": int(_clk_mutation_event_id),
            }
            _causal_fork_contract = build_causal_fork_contract(
                checkpoint_iteration=int(iteration),
                seed_lineage_freeze_from_iter=int(
                    getattr(args, "streaming_seed_lineage_freeze_from_iter", -1)
                ),
                fork_boundary_flush=_fork_boundary_flush,
            )
            _causal_fork_trainer_owner = (
                _capture_exact_causal_fork_trainer_owner(_fork_boundary_flush)
            )
            _causal_fork_owners = {
                "mcmc": mcmc_strategy,
                "keyframes": streaming_scene._keyframe_store,
                "replay": streaming_scene._replay_view_store,
                "dirty_regions": dirty_region_queue,
                "phase": _phase_controller,
                "cadence": _cadence,
                "trainer": _causal_fork_trainer_owner,
                "mapping_clocks": _coordinator.four_clocks,
                "observations": streaming_scene._observation_store,
                "frame_scheduler": scheduler,
                "admission": _keyframe_admitter,
                "scene": ExactCausalForkSceneState(
                    streaming_scene,
                    submap_manager=_submap_manager,
                    gaussians=gaussians,
                ),
                "frame_observations": ExactCausalForkObservationDBState(
                    _observation_db
                ),
            }
            _causal_fork_envelope = build_resume_envelope(
                _exact_resume_identity, _causal_fork_owners
            )
            _causal_fork_phase = {
                "version": 1,
                "phase": "mapping",
                "fork_iteration": int(iteration),
                "source_idx": int(streaming_scene._source_idx),
                "source_frame_count": int(streaming_scene._frame_count),
                "admission_work_signature": dict(
                    _causal_fork_trainer_owner.state["admission_work_signature"]
                ),
            }
            _causal_fork_runtime = _exact_refinement_runtime(
                "causal_fork",
                _causal_fork_phase,
                _causal_fork_envelope,
                _causal_fork_contract,
            )
            _causal_fork_verdict = qualify_exact_causal_fork_resume(
                _causal_fork_runtime,
                _exact_resume_identity,
                REQUIRED_CAUSAL_FORK_OWNERS,
            )
            if not _causal_fork_verdict.eligible:
                raise RuntimeError(
                    "exact causal fork is unsupported in this runtime: "
                    + "; ".join(_causal_fork_verdict.reasons)
                )
            _save_streaming_checkpoint(
                save_worker,
                gaussians,
                args.model_path,
                iteration,
                n_frames_ingested,
                _last_ingest_iter,
                streaming_scene._source_idx,
                keyframes_admitted=streaming_scene._keyframes_admitted,
                reason="causal_fork",
                map_backend=map_backend,
                seed_fade_monitor=seed_fade_monitor,
                per_source_supervision_counts=_per_source_supervision_counts,
                coverage_ledger=_coverage_snapshot(),
                extra_metadata={
                    "streaming_phase": _causal_fork_phase,
                    "causal_fork": _causal_fork_contract,
                    "streaming_resume_v2": _causal_fork_envelope,
                },
            )
            save_worker.join()
            _persistence_worker = getattr(args, "_persistence_worker", None)
            if _persistence_worker is not None:
                _persistence_worker.sync_now(timeout=600.0)
            _skip_periodic_checkpoint_iteration = iteration
            _exact_fork_written = True
            print(
                "[streaming] exact causal fork saved at "
                f"iteration={iteration}, source_idx={streaming_scene._source_idx}, "
                f"flushed_items={_fork_pending_items}, "
                f"inserted={_fork_inserted_count}",
                flush=True,
            )

        # ---- Explicit mapping endpoint boundary ---------------------------
        if _post_mapping_refinement_steps > 0:
            _source_exhausted, _dispositions_complete = _source_disposition_state(
                streaming_scene
            )
            if (
                _phase_controller.phase is Phase.MAPPING
                and _source_exhausted
                and _dispositions_complete
                and _insertion_batch
            ):
                _remaining = _flush_insertion_batch(
                    _insertion_batch,
                    gaussians,
                    streaming_scene,
                    iteration,
                    args,
                    map_backend=map_backend,
                )
                _clk_depth_mutation_event_id += 1
                _clk_mutation_event_id += 1
                _cadence.record_insertion_commit(_remaining)
                _coordinator.four_clocks.record_map_commit(n_inserted=_remaining)
                _insertion_batch = []
                _insertion_batch_frames = 0
                _last_flush_iteration = iteration
                if _remaining > 0:
                    total_inserted += _remaining
                    print(
                        "[streaming] Mapping-boundary batch flush: inserted "
                        f"{_remaining} Gaussians",
                        flush=True,
                    )

            _mapping_transition = _phase_controller.observe_source(
                source_exhausted=_source_exhausted,
                # The endpoint is quiescent only after the normal mapping loop
                # has consumed every queued consolidation view and flushed the
                # insertion batch.  Until then we keep mapping and drain work.
                dispositions_complete=(
                    _dispositions_complete
                    and not _insertion_batch
                    and not _consolidation_burst_queue
                ),
                successful_optimization_steps=_clk_optimizer_step_id,
                iteration=iteration,
            )
            if (
                _mapping_transition is not None
                and _mapping_transition.from_phase is Phase.MAPPING
                and _mapping_transition.to_phase is Phase.REFINEMENT
            ):
                _refinement_resolution_tier = _resolution_controller.decide(
                    mapping_done=True
                )
                if _refinement_resolution_tier != "high":
                    raise RuntimeError(
                        "exact refinement boundary requires native HIGH resolution"
                    )
                _mapping_wall_time_s = time.perf_counter() - _mapping_phase_wall_start
                _mapping_endpoint_dir = os.path.join(
                    args.model_path, "mapping_endpoint"
                )
                os.makedirs(_mapping_endpoint_dir, exist_ok=True)
                _mapping_ply_path = os.path.join(
                    _mapping_endpoint_dir, "point_cloud.ply"
                )
                _published_mapping_endpoint = persistent_endpoint_paths(
                    args.model_path,
                    getattr(args, "persist_model_path", None),
                    iteration,
                )
                _streaming_phase = {
                    "version": 1,
                    "phase": "refinement",
                    "mapping_end_iteration": int(iteration),
                    "mapping_optimizer_steps": int(_clk_optimizer_step_id),
                    "refinement_optimizer_steps_requested": int(
                        _post_mapping_refinement_steps
                    ),
                    "refinement_optimizer_steps": 0,
                    "mapping_wall_time_s": float(_mapping_wall_time_s),
                    "refinement_wall_time_s": 0.0,
                    "source_idx": int(streaming_scene._source_idx),
                    "source_frame_count": int(streaming_scene._frame_count),
                    "source_coverage_fraction": 1.0,
                    "refinement_resolution_tier": _refinement_resolution_tier,
                    "mapping_endpoint": _published_mapping_endpoint,
                }
                _mapping_trainer_owner = _capture_exact_trainer_owner()
                _mapping_exact_owners = {
                    "mcmc": mcmc_strategy,
                    "keyframes": streaming_scene._keyframe_store,
                    "replay": streaming_scene._replay_view_store,
                    "dirty_regions": dirty_region_queue,
                    "phase": _phase_controller,
                    "cadence": _cadence,
                    "trainer": _mapping_trainer_owner,
                    "mapping_clocks": _coordinator.four_clocks,
                    "observations": streaming_scene._observation_store,
                }
                _mapping_resume_envelope = build_resume_envelope(
                    _exact_resume_identity, _mapping_exact_owners
                )
                _mapping_resume_runtime = _exact_refinement_runtime(
                    "mapping_endpoint",
                    _streaming_phase,
                    _mapping_resume_envelope,
                )
                _mapping_resume_verdict = qualify_exact_refinement_resume(
                    _mapping_resume_runtime,
                    _exact_resume_identity,
                    REQUIRED_EXACT_OWNERS,
                )
                _streaming_phase["exact_resume"] = {
                    "classification": _mapping_resume_verdict.evidence_class,
                    "reasons": list(_mapping_resume_verdict.reasons),
                }
                # Existing asynchronous checkpoints retain live tensor
                # references until serialization. Drain them before taking the
                # score-bearing endpoint snapshot, then keep training paused
                # until every endpoint artifact is durable.
                save_worker.join()
                _save_streaming_checkpoint(
                    save_worker,
                    gaussians,
                    _mapping_endpoint_dir,
                    iteration,
                    n_frames_ingested,
                    _last_ingest_iter,
                    streaming_scene._source_idx,
                    keyframes_admitted=streaming_scene._keyframes_admitted,
                    reason="mapping_endpoint",
                    map_backend=map_backend,
                    seed_fade_monitor=seed_fade_monitor,
                    per_source_supervision_counts=_per_source_supervision_counts,
                    coverage_ledger=_coverage_snapshot(),
                    extra_metadata={
                        "streaming_phase": dict(_streaming_phase),
                        "streaming_resume_v2": _mapping_resume_envelope,
                    },
                )
                _export_ply_snapshot(
                    gaussians,
                    streaming_scene,
                    _mapping_ply_path,
                    save_worker,
                    min_opacity=float(
                        getattr(args, "streaming_export_min_opacity", 0.0)
                    ),
                )
                _save_cameras_json(
                    save_worker,
                    args.model_path,
                    streaming_scene.train_cameras,
                )
                _save_test_cameras_json(
                    save_worker,
                    args.model_path,
                    streaming_scene._test_cameras,
                )
                _save_holdout_gt(
                    save_worker,
                    args.model_path,
                    streaming_scene._test_cameras,
                )
                _flush_dense_trajectory(
                    save_worker, args.model_path, streaming_scene
                )
                save_worker.join()
                _write_streaming_phase_metadata(
                    _mapping_endpoint_dir, _streaming_phase
                )
                _write_streaming_phase_metadata(
                    args.model_path, _streaming_phase
                )

                # The mapping endpoint is score-bearing, so it may not be
                # skipped when a periodic report is in flight.
                report_worker.join()
                _admission_fingerprint.finalize(
                    total_source_frames=streaming_scene._frame_count
                )
                from eligsir.utils.comparison_report import write_post_training_report
                from gaussian_renderer.gsplat_backend import render_batch as _rb_mapping

                write_post_training_report(
                    streaming_scene=streaming_scene,
                    model_path=args.model_path,
                    iteration=iteration,
                    gaussians=gaussians,
                    train_cams=list(streaming_scene.getTrainCameras()),
                    test_cams=list(streaming_scene.getTestCameras()),
                    render_fn=render,
                    pipe=pipe,
                    background=background,
                    tb_writer=tb_writer,
                    log_prefix="streaming_mapping_endpoint",
                    subdir="mapping_endpoint",
                    batch_render_fn=_rb_mapping,
                    eval_metadata=streaming_scene.eval_metadata(),
                    expressiveness=streaming_scene.scene_complexity(),
                    input_quality=streaming_scene.input_quality(),
                    last_ingest_iter=_last_ingest_iter,
                    true_final_iteration=iteration,
                    mapping_phase_source="live",
                    admission_fingerprint=_admission_fingerprint,
                    evaluation_tick_clock_snapshot=(
                        _terminal_eval_tick_clock_snapshot()
                    ),
                    **_streaming_report_kwargs(final=True),
                )
                _persistence_worker = getattr(args, "_persistence_worker", None)
                if _persistence_worker is not None:
                    _persistence_worker.sync_now(timeout=600.0)
                verify_published_endpoint(
                    args.model_path,
                    _published_mapping_endpoint,
                    iteration=iteration,
                )
                # Endpoint export/evaluation may consume random streams.  Reset
                # to the state embedded in the durable checkpoint so the live
                # in-process tail and a resumed tail begin identically.
                restore_rng_state(_mapping_resume_envelope["rng"])
                phase_timer.end("training")
                phase_timer.start("post_mapping_refinement")
                _refinement_phase_wall_start = time.perf_counter()
                # Issue #1833: decouple the refinement-phase depth-loss weight
                # from the mapping-phase one. -1.0 (default) leaves
                # `depth_loss_weight` (and therefore
                # `_streaming_optimization_runtime.loss_assembly_kwargs`,
                # which eligsir/mapping/optimization_runtime.py re-reads via
                # `dict(self.loss_assembly_kwargs)` on every call, so this
                # mutation is picked up starting with the very next
                # refinement step) exactly as the mapping phase set it --
                # zero behavior change when unset. A value >= 0.0 overrides
                # it for the refinement tail only (e.g. 0.0 to fully disable
                # depth supervision during refinement).
                _refinement_depth_loss_weight = float(
                    getattr(args, "streaming_refinement_depth_loss_weight", -1.0)
                )
                if _refinement_depth_loss_weight >= 0.0:
                    depth_loss_weight = _refinement_depth_loss_weight
                    use_depth_loss = depth_loss_weight > 0
                    if _streaming_optimization_runtime is not None:
                        _streaming_optimization_runtime.loss_assembly_kwargs[
                            "depth_loss_weight"
                        ] = depth_loss_weight
                print(
                    "[streaming] mapping endpoint saved at "
                    f"iteration={iteration}, optimizer_steps={_clk_optimizer_step_id}; "
                    f"starting {_post_mapping_refinement_steps} in-process "
                    "refinement optimizer steps",
                    flush=True,
                )
            elif (
                _mapping_transition is not None
                and _mapping_transition.to_phase is Phase.COMPLETE
            ):
                _streaming_phase["phase"] = "complete"
                _streaming_phase["refinement_optimizer_steps"] = int(
                    _phase_controller.refinement_steps_completed
                )
                _streaming_phase["refinement_wall_time_s"] = float(
                    time.perf_counter() - _refinement_phase_wall_start
                )
                _streaming_phase["final_iteration"] = int(iteration)
                _streaming_phase["final_optimizer_steps"] = int(
                    _clk_optimizer_step_id
                )
                _published_model_root = (
                    getattr(args, "persist_model_path", None) or args.model_path
                )
                _streaming_phase["refined_endpoint"] = {
                    "point_cloud": os.path.join(
                        _published_model_root,
                        f"point_cloud/iteration_{iteration}/point_cloud.ply",
                    ),
                    "report": os.path.join(
                        _published_model_root,
                        "comparison",
                        f"iter_{iteration}",
                        "report.json",
                    ),
                }
                _write_streaming_phase_metadata(
                    args.model_path, _streaming_phase
                )
                phase_timer.end("post_mapping_refinement")
                progress_bar.close()
                print(
                    "[streaming] post-mapping refinement complete: "
                    f"{_phase_controller.refinement_steps_completed}/"
                    f"{_post_mapping_refinement_steps} successful optimizer steps",
                    flush=True,
                )
                break

        # ---- Saving -------------------------------------------------------
        # Frame-based PLY saves: fire when frame count crosses a new milestone
        if save_frame_interval > 0:
            _milestone = (n_frames_ingested // save_frame_interval) * save_frame_interval
            if _milestone > _last_frame_save and _milestone > 0:
                _last_frame_save = _milestone
                print(f"\n[FRAME {n_frames_ingested}] Saving Gaussians (step={iteration})", flush=True)
                _export_ply_snapshot(
                    gaussians, streaming_scene,
                    os.path.join(args.model_path, f"point_cloud/frame_{n_frames_ingested}/point_cloud.ply"),
                    save_worker,
                    min_opacity=float(getattr(args, "streaming_export_min_opacity", 0.0)),
                )
                if getattr(args, "streaming_render_at_saves", False):
                    _render_streaming_snapshot(
                        gaussians, streaming_scene, render, pipe, background,
                        args, n_frames_ingested, tb_writer, save_worker=save_worker,
                    )

        # Iteration-based PLY saves (explicit list or interval fallback)
        if (iteration in saving_iterations) or (save_interval > 0 and iteration % save_interval == 0):
            print(f"\n[STEP {iteration}] Saving Gaussians", flush=True)
            _export_ply_snapshot(
                gaussians, streaming_scene,
                os.path.join(args.model_path, f"point_cloud/iteration_{iteration}/point_cloud.ply"),
                save_worker,
                min_opacity=float(getattr(args, "streaming_export_min_opacity", 0.0)),
            )
            # Root-level artifacts a post-hoc render/eval module needs: the exact
            # test-holdout split + its GT, and the dense (every-frame) trajectory.
            # Piggybacked on this cadence (not checkpoint_interval, which defaults
            # to 0 / often unset) so they update reliably across a run.
            _save_test_cameras_json(save_worker, args.model_path, streaming_scene._test_cameras)
            _save_holdout_gt(save_worker, args.model_path, streaming_scene._test_cameras)
            _flush_dense_trajectory(save_worker, args.model_path, streaming_scene)

        if iteration != _skip_periodic_checkpoint_iteration and (
            (iteration in checkpoint_iterations)
            or (chk_interval > 0 and iteration % chk_interval == 0)
        ):
            _save_streaming_checkpoint(
                save_worker, gaussians, args.model_path, iteration,
                n_frames_ingested, _last_ingest_iter, streaming_scene._source_idx,
                keyframes_admitted=streaming_scene._keyframes_admitted,
                reason="periodic",
                map_backend=map_backend,
                # Issue #290: carry the seed-fade detector's decision state
                # across a resume; without it a resumed run's verdict is
                # computed against a running max that restarted from -inf.
                seed_fade_monitor=seed_fade_monitor,
                per_source_supervision_counts=_per_source_supervision_counts,
                coverage_ledger=_coverage_snapshot(),
            )
            # Keep a root cameras.json in sync so any saved checkpoint can be rendered.
            _save_cameras_json(save_worker, args.model_path, streaming_scene.train_cameras)
            _save_test_cameras_json(save_worker, args.model_path, streaming_scene._test_cameras)
            _save_holdout_gt(save_worker, args.model_path, streaming_scene._test_cameras)
            _flush_dense_trajectory(save_worker, args.model_path, streaming_scene)

        if _prof is not None:
            _prof.step()
            if iteration - first_iter + 1 >= _profile_wait + 1 + _profile_active:  # wait + warmup + active
                _prof.__exit__(None, None, None)
                _prof.export_chrome_trace(_profile_path)
                _key_avg_path = _profile_path + ".key_averages.txt"
                with open(_key_avg_path, "w") as _f:
                    _f.write(_prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=60))
                    _f.write("\n\n")
                    _f.write(_prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=60))
                print(f"[profile] Trace saved to {_profile_path}; key-averages -> {_key_avg_path}", flush=True)
                _prof = None

    # Safety net: loop ended (e.g. --iterations too small) before the
    # profiler's active window completed -- still export whatever was captured.
    if (
        _post_mapping_refinement_steps > 0
        and _phase_controller.phase is not Phase.COMPLETE
    ):
        raise RuntimeError(
            "explicit mapping/refinement loop exited before the configured "
            "successful optimizer-step budget completed"
        )
    if _exact_fork_iteration >= 0 and not _exact_fork_written:
        raise RuntimeError(
            "streaming loop ended before the requested exact causal fork iteration"
        )
    if _prof is not None:
        _prof.__exit__(None, None, None)
        _prof.export_chrome_trace(_profile_path)
        _key_avg_path = _profile_path + ".key_averages.txt"
        with open(_key_avg_path, "w") as _f:
            _f.write(_prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=60))
            _f.write("\n\n")
            _f.write(_prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=60))
        print(f"[profile] Trace saved to {_profile_path} (loop ended early); key-averages -> {_key_avg_path}", flush=True)
        _prof = None

    # Stop the async ingestion worker (no-op when off). Daemon thread, so this
    # is best-effort clean shutdown; queued-but-uncommitted decoded frames are
    # harmless (cache entries) and any remaining undrained source frames are
    # handled by the existing has_next_frame() flush path below.
    if _ingest_worker is not None:
        _ingest_worker.stop()
        print(
            f"[streaming] async ingestion worker stopped "
            f"(prepared={_ingest_worker.n_prepared}, consumed={_ingest_worker.n_consumed}).",
            flush=True,
        )

    # Stop the deferred-admission SLAM worker subprocess (no-op when off).
    # Same best-effort convention as the ingestion worker above (see its
    # comment) -- not exception-safe against a mid-loop crash, matching the
    # existing risk profile of that worker rather than introducing new
    # try/finally scoping around the whole loop.
    if _deferred_slam_tracker is not None:
        _deferred_telemetry = _deferred_slam_tracker.telemetry()
        _deferred_slam_tracker.close()
        print(
            f"[streaming][deferred_slam] worker stopped "
            f"(submitted={_deferred_telemetry['frames_submitted']}, "
            f"admitted={_deferred_telemetry['frames_admitted']}, "
            f"skipped_no_pose={_deferred_telemetry['frames_skipped_no_pose']}, "
            f"pending_timeouts={_deferred_telemetry['pending_timeouts']}).",
            flush=True,
        )

    # Stop the bounded-queue prepared-candidate pipeline (no-op when off).
    if _coordinator.bounded_pipeline is not None:
        _pipeline_depths = _coordinator.bounded_pipeline.queue_depths()
        _coordinator.shutdown_bounded_queue_pipeline()
        print(
            f"[streaming] bounded-queue prepared-candidate pipeline stopped "
            f"(final queue depths={_pipeline_depths}).",
            flush=True,
        )

    # Admit-rate summary (hybrid_keyframe mode)
    if _admit_cnt["total"] > 0:
        _n_admitted = (
            _admit_cnt["forced"] + _admit_cnt["moved"] + _admit_cnt["novel"]
            + _admit_cnt["db_diverse_revisit"] + _admit_cnt["novelty_admit"]
        )
        _admit_rate = 100.0 * _n_admitted / _admit_cnt["total"]
        print(
            f"[streaming] Keyframe admission: {_n_admitted}/{_admit_cnt['total']} admitted "
            f"({_admit_rate:.1f}%) — forced={_admit_cnt['forced']} moved={_admit_cnt['moved']} "
            f"novel={_admit_cnt['novel']} db_diverse_revisit={_admit_cnt['db_diverse_revisit']} "
            f"novelty_admit={_admit_cnt['novelty_admit']} "
            f"rej_overlap={_admit_cnt['rejected_overlap']} "
            f"rej_motion={_admit_cnt['rejected_motion']} "
            f"rej_redundant_angle={_admit_cnt['rejected_redundant_angle']}",
            flush=True,
        )
        if tb_writer:
            tb_writer.add_scalar("streaming/admit_rate_pct", _admit_rate, opt.iterations)
            for _gate, _cnt in _admit_cnt.items():
                tb_writer.add_scalar(f"streaming/admit_gate/{_gate}", _cnt, opt.iterations)

    if streaming_scene.has_next_frame():
        # Use the real registry frame count (_coverage_n_frames, set near the
        # MappingScheduler construction ~line 3057) rather than
        # _n_total_frames/streaming_scene._frame_count, which is the fake
        # 999999 sentinel for lazy bag/live sources and would make this
        # diagnostic print a meaningless "5270/999999" even when real
        # coverage was achieved correctly (2026-07-04, found via a coverage-
        # matched re-run: termination fired correctly, but this print alone
        # still used the stale denominator).
        _real_n_frames = _coverage_n_frames if _is_finite_replay else _n_total_frames
        _remaining = max(0, _real_n_frames - n_frames_ingested)
        _eff_spf = opt.iterations / max(1, _n_frames_trained)
        print(
            f"[streaming] Optimization-step budget exhausted with {_remaining} dataset frames unseen. "
            f"Coverage: {_n_frames_trained} frames optimized × {_eff_spf:.1f} avg optimization steps per admitted keyframe "
            f"= {n_frames_ingested}/{_real_n_frames} dataset frames.",
            flush=True,
        )

    # Flush any remaining pending batch insertions before final reporting/eval so
    # the exported metrics include all admitted geometry.
    if _insertion_batch:
        _remaining = _flush_insertion_batch(
            _insertion_batch, gaussians, streaming_scene, args.iterations, args,
            map_backend=map_backend)
        # PR2 cadence clocks: end-of-run flush is also a real depth-insertion commit.
        _clk_depth_mutation_event_id += 1
        _clk_mutation_event_id += 1
        _cadence.record_insertion_commit(_remaining)   # #289 insertion clock
        # Same four-clocks fix as the in-loop flush above: this end-of-run flush is
        # also a real map commit and must be recorded too.
        _coordinator.four_clocks.record_map_commit(n_inserted=_remaining)
        _insertion_batch = []
        _insertion_batch_frames = 0
        _last_flush_iteration = args.iterations
        if _remaining > 0:
            total_inserted += _remaining
            print(f"[streaming] End-of-run batch flush: inserted {_remaining} Gaussians", flush=True)

    # Guarantee a PLY exists at the exact final iteration, independent of the
    # periodic save_interval boundary. Online-mode's rolling-idle-stop rule
    # (and any other data-driven termination) almost never lands exactly on a
    # save_interval multiple, so without this the newest PLY can be hundreds
    # of iterations stale relative to what the post-training report below
    # actually scores -- see docs/superpowers/specs/2026-07-24-final-iteration-ply-save-design.md.
    # Placed AFTER the insertion-batch flush above (not right at loop-exit) so
    # the exported PLY includes any geometry flushed there -- otherwise a run
    # that exits mid-batch would export a PLY missing exactly what the report
    # below scores, reintroducing the same PLY-vs-report divergence this fix
    # exists to eliminate (found in final review 2026-07-24).
    # If `iteration` already coincides with a periodic-save boundary, this is
    # an idempotent overwrite of identical content, not a duplicate.
    if _opt_batch_count > 0:
        # Issue #106: printed unconditionally at the end of any engine-path
        # run with --streaming_optimization_batch_size > 1, so the real
        # multi-view content of the batches is on the record in train.log
        # (not only in TensorBoard) for anyone reading a benchmark result.
        print(
            f"[OPT BATCH] requested_B={_opt_batch_size} "
            f"mean_distinct_views_per_batch="
            f"{_opt_batch_distinct_total / _opt_batch_count:.3f} "
            f"batches={_opt_batch_count}",
            flush=True,
        )
    # Issue #326: close the operation trace before the final save/report so the
    # JSONL is complete on disk even if a later report stage fails.
    if _optrace is not None:
        _optrace.flush()
        print(
            f"[gaussian-op-trace] {len(_optrace.events)} events recorded "
            f"({_optrace.dropped} dropped at max_events) -> {_optrace.out_path}\n"
            f"[gaussian-op-trace] derive metrics with: "
            f"python3 scripts/gaussian_op_conflicts.py {_optrace.out_path}",
            flush=True,
        )
        _optrace.close()

    print(f"\n[FINAL STEP {iteration}] Saving Gaussians at exact final optimization step", flush=True)
    _export_ply_snapshot(
        gaussians, streaming_scene,
        os.path.join(args.model_path, f"point_cloud/iteration_{iteration}/point_cloud.ply"),
        save_worker,
        min_opacity=float(getattr(args, "streaming_export_min_opacity", 0.0)),
    )
    # Final flush of the root-level artifacts a post-hoc render/eval module
    # needs (test holdout split + GT, dense trajectory) -- guarantees these are
    # complete and up to date even if the run ended off a save_interval boundary.
    _save_test_cameras_json(save_worker, args.model_path, streaming_scene._test_cameras)
    _save_holdout_gt(save_worker, args.model_path, streaming_scene._test_cameras)
    _flush_dense_trajectory(save_worker, args.model_path, streaming_scene)

    # Post-training report: bounded test/train metrics + side-by-side PNGs +
    # contact sheet + trajectory MP4. Can be disabled for core-training profiling.
    # _benchmark_mode was resolved earlier; report is skipped when it is active.
    # Report/eval artifact generation runs several minute-scale, individually
    # silent stages (multi-view renders, contact sheets, mp4s, ATE eval). Bracket
    # each with a "started … / done (Ns)" line so the process never looks hung.
    #
    # Issue #697 (fail-closed evidence phase): each phase is wrapped by
    # utils.run_completion_state, which prints the "✓ ... done" checkmark ONLY
    # on the success path and records FAILED / MISSING_ARTIFACTS / NOT_REQUESTED
    # into <model_path>/run_completion_state.json instead. Before this, the
    # final report lived inside `except Exception: print(...)` and the "✓" was
    # printed unconditionally afterwards -- the #646/#647 outage, in which
    # every streaming run for two days produced no report.json while logging
    # exactly like a healthy run.
    from eligsir.utils.run_completion_state import (
        RunCompletionState,
        FINAL_REPORT_CONTRACT,
        BOOTSTRAP_VIEWS_CONTRACT,
        TRAJECTORY_EVAL_CONTRACT,
        OPTIMIZATION_CURVE_CONTRACT,
        PhaseContract as _PhaseContract,
    )

    _run_state = RunCompletionState(args.model_path)
    _run_state.mark_training_complete(final_iteration=iteration)

    _report_final = getattr(args, "streaming_report_final", True) and not _benchmark_mode
    # Bootstrap-view comparison renders are always written when the bootstrap-motion
    # diagnostic is active — even if --no-streaming_report_final was passed.
    _force_bootstrap_views = _debug_bootstrap_motion and bool(_bootstrap_cams)
    # Always wait for any in-flight mid-training report before the process can
    # exit -- even when the final report itself is skipped (--no-streaming_report_final
    # or _benchmark_mode), a background report from the last mid-training tick
    # could otherwise still be writing files when training ends.
    report_worker.join()
    # Stop the resource-monitor thread before the final report reads its
    # summary, so the summary genuinely covers "whole run" rather than racing
    # a still-running poller for the last sample or two.
    if _resource_monitor is not None:
        _resource_monitor.stop()
    if _report_final or _force_bootstrap_views:
        # Reports use already materialized train/test cameras. Do not drain
        # live odometry across unseen source frames after a budget-limited run.
        try:
            streaming_scene.stop_pending_odometry()
        except Exception:
            pass

    _final_report_iteration = (
        int(iteration)
        if _post_mapping_refinement_steps > 0
        else int(opt.iterations)
    )
    if _report_final:
        with _run_state.phase(FINAL_REPORT_CONTRACT) as _ph:
            # Imported INSIDE the phase body (and duplicated in the bootstrap
            # phase below -- module import is cached, so the cost is nil): an
            # ImportError here is a report-phase failure and must be recorded
            # as one. Hoisting these above the phases would let an ImportError
            # escape streaming_training() entirely, skipping the whole shutdown
            # tail -- save_worker.shutdown(), _run_state.write(), and
            # _finalize_output_staging() -- i.e. losing the run's outputs AND
            # the very state file that exists to record this failure.
            # `with` creates no scope, so `_git_provenance_from_args` still
            # binds as a streaming_training() local and
            # `_streaming_report_kwargs`'s closure cell resolves as before.
            from eligsir.utils.comparison_report import write_post_training_report, git_provenance_from_args as _git_provenance_from_args
            from gaussian_renderer.gsplat_backend import render_batch as _rb_final
            # Issue #795: finalize admission fingerprint before report generation.
            _admission_fingerprint.finalize(total_source_frames=streaming_scene._frame_count if hasattr(streaming_scene, '_frame_count') else None)
            _ph.record_output(
                write_post_training_report(
                    streaming_scene=streaming_scene,
                    model_path=args.model_path,
                    iteration=_final_report_iteration,
                    gaussians=gaussians,
                    train_cams=list(streaming_scene.getTrainCameras()),
                    test_cams=list(streaming_scene.getTestCameras()),
                    render_fn=render,
                    pipe=pipe,
                    background=background,
                    tb_writer=tb_writer,
                    log_prefix="streaming_report",
                    batch_render_fn=_rb_final,
                    eval_metadata=streaming_scene.eval_metadata(),
                    expressiveness=streaming_scene.scene_complexity(),
                    input_quality=streaming_scene.input_quality(),
                    resumed_from_iteration=_resumed_from_iteration,
                    admission_fingerprint=_admission_fingerprint,
                    budgeted_admission=(
                        _keyframe_admitter.budgeted_admission_report
                        if bool(args.streaming_budgeted_admission)
                        else None
                    ),
                    # Issue #346 follow-up (2026-08-13): makes the mapping/
                    # refinement phase boundary machine-readable in
                    # report.json["trajectory_coverage"] (TOP-LEVEL, not
                    # under "geometry" -- issue #504).
                    # `_last_ingest_iter` is the loop's own rolling-idle-stop
                    # clock (real for every streaming run, online_mode or not
                    # -- see its definition comment above the main loop).
                    # `true_final_iteration=iteration` passes the REAL loop
                    # exit iteration to work around this call's own
                    # `iteration=opt.iterations` argument being the known
                    # online_mode 2^31 sentinel (see write_post_training_report's
                    # docstring) -- deliberately does NOT change the
                    # `iteration=opt.iterations` argument itself, since that
                    # also drives this report's output directory name
                    # (`comparison/iter_2147483647/`), which every existing
                    # consumer (numeric-final-checkpoint selection, PR #353's
                    # corpus scan, etc.) already depends on.
                    last_ingest_iter=_last_ingest_iter,
                    true_final_iteration=iteration,
                    mapping_phase_source="live",
                    evaluation_tick_clock_snapshot=(
                        _terminal_eval_tick_clock_snapshot()
                    ),
                    **_streaming_report_kwargs(final=True),
                )
            )
    else:
        _run_state.mark_not_requested(
            FINAL_REPORT_CONTRACT,
            reason=(
                "--streaming_benchmark_mode" if _benchmark_mode
                else "--no-streaming_report_final"
            ),
        )
        print("[streaming-report] final report disabled by --no-streaming_report_final", flush=True)

    # Re-render the bootstrap views with the trained Gaussians so the
    # iter_0 vs end-of-training comparison is over the same viewpoints.
    # Written unconditionally when --streaming_debug_bootstrap_motion is set.
    # Issue #697: its OWN phase, with its own contract. It used to share the
    # final report's single try/except, so a failure here was indistinguishable
    # from a failure of the required scientific report -- and vice versa. This
    # is a debug diagnostic: it is recorded honestly when it fails but never
    # downgrades the run result.
    if _bootstrap_cams and (_report_final or _force_bootstrap_views):
        with _run_state.phase(BOOTSTRAP_VIEWS_CONTRACT) as _ph:
            # See the note in the final-report phase above. Needed here too:
            # this path also runs when _report_final is False
            # (--streaming_debug_bootstrap_motion alone), and
            # `**_streaming_report_kwargs()` reads `_git_provenance_from_args`.
            from eligsir.utils.comparison_report import write_post_training_report, git_provenance_from_args as _git_provenance_from_args
            from gaussian_renderer.gsplat_backend import render_batch as _rb_final
            _ph.record_output(
                write_post_training_report(
                    streaming_scene=streaming_scene,
                    model_path=args.model_path,
                    iteration=iteration,
                    gaussians=gaussians,
                    train_cams=_bootstrap_cams,
                    test_cams=_bootstrap_cams,
                    render_fn=render,
                    pipe=pipe,
                    background=background,
                    tb_writer=tb_writer,
                    log_prefix="streaming_report",
                    subdir=f"iter_{iteration}_bootstrap_views",
                    batch_render_fn=_rb_final,
                    degenerate_split=True,
                    last_ingest_iter=_last_ingest_iter,
                    mapping_phase_source="live",
                    admission_fingerprint=_admission_fingerprint,
                    budgeted_admission=(
                        _keyframe_admitter.budgeted_admission_report
                        if bool(args.streaming_budgeted_admission)
                        else None
                    ),
                    **_streaming_report_kwargs(),
                )
            )

    # ---- Trajectory evaluation -----------------------------------------------
    if getattr(args, "streaming_trajectory_eval", True) and not _benchmark_mode:
        with _run_state.phase(TRAJECTORY_EVAL_CONTRACT):
            from eligsir.utils.trajectory_eval import run_trajectory_eval, discover_gt_trajectory
            _gt_tum_path = getattr(args, "tum_gt_path", "") or ""
            if _gt_tum_path:
                print(f"[trajectory_eval] using explicit GT override: {_gt_tum_path}", flush=True)
            else:
                _gt_tum_path = discover_gt_trajectory(getattr(args, "source_path", "")) or ""
                if _gt_tum_path:
                    print(f"[trajectory_eval] auto-discovered GT: {_gt_tum_path}", flush=True)
                else:
                    print("[trajectory_eval] no GT trajectory available for this source — skipping ATE/RPE", flush=True)
            # method_label used to default unconditionally to
            # `orbbec_open3d_odom_method` (default "hybrid") for every streaming
            # source -- correct for Orbbec/ROS2 sources where that flag really
            # is how poses were estimated, but a misleading label on any other
            # source (e.g. a RealityScan Mobile import surfaced "Trajectory —
            # hybrid" on its plot even though no Open3D odometry ran at all;
            # found 2026-08-12 investigating a real "distance: 2213 m" report).
            # Only sources that actually run Open3D RGBD odometry get that
            # flag's value as their label; everything else reports its own
            # source class as a label that at least reflects reality.
            _orbbec_odom_sources = (
                "OrbbecRosBagFrameSource", "RealsenseRosBagFrameSource",
                "ROS2LiveStreamingFrameSource", "OrbbecExportFrameSource",
            )
            _src_cls_name = type(frame_source).__name__
            if _src_cls_name in _orbbec_odom_sources:
                _traj_method_label = args.orbbec_open3d_odom_method
            else:
                _traj_method_label = _src_cls_name
            run_trajectory_eval(
                output_dir=os.path.join(args.model_path, "trajectory_eval"),
                train_cams=list(streaming_scene.getTrainCameras()),
                test_cams=list(streaming_scene.getTestCameras()),
                gt_tum_path=_gt_tum_path or None,
                association_max_dt=float(getattr(args, "tum_association_max_dt", 0.03)),
                method_label=_traj_method_label,
            )

    # ---- Training-quality curve (PSNR / N over iterations) --------------------
    # Always written (cheap: reads the run's own TB events). The last-value psnr is
    # unreliable for streaming — this curve + its std/%<12dB exposes oscillation.
    with _run_state.phase(OPTIMIZATION_CURVE_CONTRACT):
        if tb_writer is not None:
            tb_writer.flush()
        from eligsir.utils.training_curves import plot_training_curves
        _curve = plot_training_curves(args.model_path)
        if _curve:
            print(f"[streaming-report] wrote optimization curve: {_curve}", flush=True)
        from eligsir.mapping.scheduler.resolution_tier_usage import (
            flush_resolution_tier_usage,
            plot_resolution_tier_usage,
        )
        # Issue #1664: rows are buffered in memory now; flush before reading
        # the trace back for this report.
        flush_resolution_tier_usage(args.model_path)
        _tier_usage_plot = plot_resolution_tier_usage(args.model_path)
        if _tier_usage_plot:
            print(
                f"[streaming-report] wrote resolution-tier usage heatmap: {_tier_usage_plot}",
                flush=True,
            )

    _w3_02_mutation_trace_summary = (
        _w3_02_mutation_trace.finalize() if _w3_02_mutation_trace.enabled else None
    )
    if _w3_02_mutation_trace_summary is not None:
        print(f"[w3-02-mutation-trace] {_w3_02_mutation_trace_summary}", flush=True)

    if _cohort_trace.enabled:
        _cohort_trace.finalize()
        print(f"[cohort-trace] wrote {_cohort_trace.path}", flush=True)
        if _w3_02_mutation_trace.mode == "verify" and not _w3_02_mutation_trace_summary.get(
            "mutation_trace_applied", False
        ):
            print(
                "[w3-02-mutation-trace] WARNING: this run's mutation trace did NOT "
                "fully match the frozen baseline -- do not report mutation_trace_applied=true "
                "for this evidence pair.",
                flush=True,
            )
    if _w3_02_evidence is not None:
        _w3_02_evidence.finalize(
            model_path=args.model_path,
            manifest_path=args.w3_02_evidence_manifest,
            arm=args.w3_02_evidence_arm,
            git_commit=args.w3_02_git_commit,
            clean_checkout=args.w3_02_clean_checkout,
            seed=int(getattr(args, "seed", 1)),
            checkpoint_path=args.start_checkpoint,
            mutation_trace_path=args.w3_02_mutation_trace,
            frozen_artifact_paths={
                "config_hash": args.w3_02_config_artifact,
                "source_artifact_hash": args.w3_02_source_artifact,
                "pose_hash": args.w3_02_pose_artifact,
                "keyframe_hash": args.w3_02_keyframe_artifact,
                "holdout_hash": args.w3_02_holdout_artifact,
            },
            mutation_trace_summary=_w3_02_mutation_trace_summary,
        )

    # ---- Training-progress video write ---------------------------------------
    # Signal the CPU worker to stop, then wait for it to finish any in-flight frames.
    _pv_cpu_queue.put(None)
    _pv_thread.join(timeout=120)
    if _progress_frames:
        from eligsir.utils.comparison_report import _write_mp4
        _prog_path = os.path.join(args.model_path, "training_progress.mp4")
        # Optional phase (issue #697): a failed progress video is recorded in
        # run_completion_state.json but never downgrades the run result. The
        # raise inside the body is how a non-raising failure (_write_mp4
        # returns an error string) reaches the phase wrapper -- without it the
        # "✓ done" checkmark would print over a failed write, the exact class
        # of confusion #697 exists to remove.
        with _run_state.phase(_PhaseContract(
            name="training_progress_video",
            label=f"training_progress.mp4 ({len(_progress_frames)} frames)",
            required=False,
        )):
            print(f"[streaming] Writing training_progress.mp4 ({len(_progress_frames)} frames @ {_progress_video_fps}fps)…",
                  flush=True)
            _err = _write_mp4(_prog_path, iter(_progress_frames), fps=_progress_video_fps)
            if _err is not None:
                raise RuntimeError(f"training_progress.mp4 write failed: {_err}")
            print(f"[streaming] training_progress.mp4 → {_prog_path}", flush=True)

    # Issue #1075 P1: shared shutdown tail. `finalize_staging=False` -- the
    # run_state / run_summary / artifact_manifest writes below must land BEFORE
    # the staging-finalize call, so that call stays where it is, further
    # down (deliberately not named verbatim here -- several contract tests
    # locate it by string index in this file's source).
    finalize_run(
        save_worker,
        shutdown_first=False,
        finalize_staging=False,
        orig_sigint=_orig_sigint,
        orig_sigterm=_orig_sigterm,
    )
    if _mapping_application is not None:
        _mapping_application.close()
    else:
        streaming_scene.close_decode_cache()
    phase_timer.write_summary()
    # Issue #562: duty-cycle artifacts are written ONCE, here -- never flushed
    # periodically during training (this repo has a documented incident where
    # the TensorBoard writer's small periodic writes stalled the main thread on
    # NFS latency). Detaching the recorder first restores `_timed`'s disabled
    # fast path for anything that runs after the loop.
    if _duty is not None:
        from eligsir.utils.ingest_phase_profiler import set_duty_cycle_recorder as _set_duty

        _set_duty(None)
        print(_duty.format_line(), flush=True)
        _duty.write_artifacts(args.model_path)
    # Bare attribute access on purpose -- do NOT reintroduce a getattr
    # fallback here. model_path is an unconditional ParamGroup default (real
    # default ''), and this line is only ever reached from
    # streaming_training() with the full parsed args. The fallback form this
    # replaces used a None literal that had already drifted from that ''
    # default -- exactly the stale-literal trap issue #363/#368 exists to
    # catch, and tests/test_getattr_fallback_freshness.py fails on it.
    _dump_phase_ops(args.model_path)
    _viewer.close()
    # issue #333 acceptance criterion 6: the shutdown log is O(number of
    # reason CLASSES), not O(number of frame ids). The old second line printed
    # a dict keyed by `cache_empty(frame_id=N)` -- one key per replayed frame,
    # tens of thousands of them on a long run. Detailed per-class stats (with
    # a bounded frame-id example list) go to multires_bind_stats.json instead.
    print(f"[multires] bind stats: {_multires_bind_stats}", flush=True)
    print(_MULTIRES_BIND_ACCOUNTING.summary_line(), flush=True)
    try:
        _bind_stats_payload = _MULTIRES_BIND_ACCOUNTING.to_dict()
        _bind_stats_payload["legacy_tier_counters"] = dict(_multires_bind_stats)
        _bind_stats_payload["resolution_cache"] = {
            # Bare attribute access on purpose (issue #368 / the model_path
            # note above): this line only ever runs from streaming_training()
            # with fully parsed args, so a fallback literal here could only
            # drift away from arguments/__init__.py's real default (400).
            "max_frames": int(args.streaming_resolution_cache_max_frames or 0),
            "evictions": int(getattr(_observation_db, "resolution_cache_evictions", 0)),
            "occupancy": int(getattr(_observation_db, "resolution_cache_occupancy", 0)),
            "reconstruction_attempts": int(
                getattr(_observation_db, "resolution_cache_reconstruction_attempts", 0)
            ),
            "reconstructions": int(
                getattr(_observation_db, "resolution_cache_reconstructions", 0)
            ),
            "reconstruction_failures": int(
                getattr(_observation_db, "resolution_cache_reconstruction_failures", 0)
            ),
        }
        _bind_stats_path = os.path.join(args.model_path, "multires_bind_stats.json")
        os.makedirs(args.model_path, exist_ok=True)
        with open(_bind_stats_path, "w") as _f:
            json.dump(_bind_stats_payload, _f, indent=2, sort_keys=True)
    except Exception as _e:  # never let telemetry writing fail a finished run
        print(f"[multires] WARNING could not write multires_bind_stats.json: {_e}", flush=True)
    print("\n[streaming] Mapping runtime complete.", flush=True)

    # Issue #697: the authoritative, machine-readable verdict for this run.
    # Written BEFORE _finalize_output_staging so it rides along to the
    # canonical path with everything else (args.model_path is the local
    # staging dir until that call moves it), and the fail-closed raise comes
    # AFTER it so a failed evidence phase costs the run its exit code, never
    # its outputs.
    _run_state.write()
    _run_state.print_summary()

    # Issue #997 / #1013: root-level run_summary.json canonical generic handoff
    from eligsir.utils.run_summary import write_run_summary
    write_run_summary(
        args.model_path,
        completion_state=_run_state.to_dict(),
        evidence_class="MANAGED" if getattr(args, "scientific_mode", False) else "UNMANAGED",
        args=args,
    )

    # Issue #1020: root-level artifact_manifest.json fast-ingest contract
    try:
        from eligsir.utils.artifact_manifest import build_artifact_manifest
        build_artifact_manifest(args.model_path)
    except Exception as _e:
        print(f"[artifact-manifest] Warning: failed to build artifact manifest: {_e}")

    # Gate a successful exit on the local staging dir actually having landed
    # on the canonical path -- mirrors train.py's Task-4 placement on the
    # offline path exactly (after signal handlers are restored, after the
    # async save worker is drained). Raises OutputPersistenceError on
    # failure, which propagates uncaught up through train.py's `training()`
    # -> __main__, this project's existing idiom for a hard training
    # failure, producing a non-zero exit code instead of a silent
    # exit-code-0 loss of nearly all output.
    _finalize_output_staging(args)

    # Issue #697 fail-closed gate. Raises RunIncompleteError when a REQUIRED
    # evidence phase failed or its declared artifacts are absent. It
    # propagates uncaught through train.training() -> __main__ exactly as
    # OutputPersistenceError does, so the process exits non-zero and
    # train.py's `print("\nTraining complete.")` is never reached -- that
    # string can no longer appear in the log of a run with no report.json.
    # (utils/champion_promotion.py anchors on that same string, so such a run
    # is now correctly refused for promotion instead of silently accepted.)
    _run_state.raise_if_incomplete()


# ---------------------------------------------------------------------------
# Test evaluation helper
# ---------------------------------------------------------------------------

def _run_test_eval(tb_writer, iteration, streaming_scene, gaussians, render_fn, pipe, background, args):
    test_cams = streaming_scene.getTestCameras()
    if not test_cams:
        return
    torch.cuda.empty_cache()
    l1_total = 0.0
    psnr_total = 0.0
    score_clamp_changed = 0
    score_clamp_total = 0
    for idx, cam in enumerate(test_cams):
        with torch.no_grad():
            img = render_fn(cam, gaussians, pipe, background)["render"]
            gt = score_tensor_from_camera(cam, device="cuda")
            score_img, score_gt, score_transform = apply_declared_score_clamp_pair(
                img,
                gt,
                context="streaming_test_eval",
            )
        l1_total += l1_loss(img, gt).mean().double().item()
        psnr_total += psnr(score_img, score_gt).mean().double().item()
        score_clamp_changed += score_transform["render"]["samples_changed"]
        score_clamp_total += score_transform["render"]["samples_total"]
        if tb_writer and idx < 5:
            tb_writer.add_images(f"test_view_{cam.image_name}/render", img[None], global_step=iteration)
    n = max(len(test_cams), 1)
    print(f"\n[STEP {iteration}] Test L1={l1_total / n:.4f} PSNR={psnr_total / n:.2f}")
    if tb_writer:
        tb_writer.add_scalar("test/l1", l1_total / n, iteration)
        tb_writer.add_scalar("test/psnr", psnr_total / n, iteration)
        tb_writer.add_scalar(
            "test/score_clamp_fraction",
            score_clamp_changed / max(1, score_clamp_total),
            iteration,
        )
    # Issue #398: return holdout image residency to its pre-eval bound. The loop
    # above deliberately uploads a TRANSIENT per-camera copy (`.to("cuda")`
    # above binds a local, not `cam.original_image`), so this is normally a
    # no-op -- it exists so that any future eval-path change that does page a
    # holdout image in permanently (e.g. a `prepare_camera_for_render` call)
    # self-corrects here instead of silently ratcheting VRAM upward over a run.
    streaming_scene.offload_test_camera_images()
    torch.cuda.empty_cache()
