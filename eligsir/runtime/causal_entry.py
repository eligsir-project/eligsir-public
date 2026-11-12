"""Executable dataset causal-mapping entrypoint for the canonical runtime."""
from __future__ import annotations

from collections.abc import Sequence
from contextlib import nullcontext
from dataclasses import asdict, dataclass, replace
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Callable, Mapping
from types import SimpleNamespace

import numpy as np

from eligsir.io.sources import make_frame_source
from eligsir.runtime.application import ComposedMappingApplication, MappingApplicationOptions
from eligsir.runtime.artifact_paths import point_cloud_artifact_paths
from eligsir.runtime.causal_replay import (
    CausalIngressTelemetryState,
    CausalReplayClock,
    CausalReplayDriver,
    CausalReplayTickPolicy,
    DatasetPacketAdapter,
    JsonlDecisionTraceObserver,
    PacedCausalReplayDriver,
    StreamingFramePacketAdapter,
)
from eligsir.runtime.prepared_source_prep_worker import READY_WAIT_FALLBACK_S
from eligsir.runtime.source_ingress import IngestionValueConfig
from eligsir.runtime.runtime_binding import runtime_binding_from_runtime
from eligsir.runtime.config import MappingRuntimeConfig, RuntimeBinding
from eligsir.runtime.config_boundary import mapping_runtime_config_from_args
from eligsir.telemetry import system_sampler
from eligsir.tracking.providers import ExternalTrackingProvider

mark_system_telemetry = system_sampler.mark_system_telemetry


@dataclass(frozen=True, slots=True)
class CausalMappingComposition:
    source: Any
    config: MappingRuntimeConfig
    binding: RuntimeBinding
    runtime: Any
    # The prepared path is an explicit transition profile.  ``None`` keeps
    # the historical causal composition exactly intact.
    prepared_ready_path: Any | None = None


def _control_probe_candidate_is_accepted(
    accepted: Mapping[int, float], candidate: Mapping[int, float]
) -> bool:
    """Require a complete control-probe vector to improve without regression."""

    if set(accepted) != set(candidate) or not accepted:
        return False
    try:
        baseline = {int(key): float(value) for key, value in accepted.items()}
        observed = {int(key): float(value) for key, value in candidate.items()}
    except (TypeError, ValueError):
        return False
    if set(baseline) != set(observed) or not all(
        np.isfinite(value) for value in (*baseline.values(), *observed.values())
    ):
        return False
    if any(observed[key] < baseline[key] for key in baseline):
        return False
    baseline_gap = max(baseline.values()) - min(baseline.values())
    observed_gap = max(observed.values()) - min(observed.values())
    if min(observed.values()) <= min(baseline.values()):
        return False
    return observed_gap == 0.0 if baseline_gap == 0.0 else observed_gap < baseline_gap


def _control_probe_partition_means(branch_runtime: Any) -> dict[int, float]:
    """Read the sealed fixed-view probe without publishing evaluator state."""

    evaluate = getattr(branch_runtime, "evaluate_fixed_views", None)
    if not callable(evaluate):
        raise TypeError("control-probe refinement requires fixed-view evaluation")
    values = evaluate(verify_read_only=True)
    if not isinstance(values, Mapping):
        raise TypeError("control-probe refinement must return partition metrics")
    try:
        result = {int(key): float(value) for key, value in values.items()}
    except (TypeError, ValueError) as error:
        raise ValueError("control-probe refinement returned invalid partition metrics") from error
    if not result or not all(np.isfinite(value) for value in result.values()):
        raise ValueError("control-probe refinement returned non-finite partition metrics")
    return result


def _restore_accepted_map_branch(
    application: Any, branch_runtime: Any, state: Any
) -> None:
    """Restore a rejected candidate or prevent it from being flushed."""

    try:
        branch_runtime.restore_live_state(state)
    except Exception as error:
        abort = getattr(application, "abort_unaccepted_refinement", None)
        if callable(abort):
            try:
                abort()
            except Exception:
                pass
        raise RuntimeError("accepted-map branch restore failed") from error


def _canonical_runtime_types() -> tuple[type[Any], Callable[..., Any]]:
    """Load the accelerator-owning runtime only at the composition root.

    Kept as a narrow helper so the causal composition can be exercised with a
    CPU fake without importing accelerator dependencies in its contract tests.
    """

    from eligsir.runtime.production_application import (
        CanonicalMappingRuntime,
        MappingBackendCompatibility,
    )

    return CanonicalMappingRuntime, MappingBackendCompatibility


def _prepared_ready_b4_enabled(args: Any) -> bool:
    """Return the explicit non-default transition-profile request."""

    return bool(getattr(args, "causal_prepared_ready_b4", False))


def _build_prepared_causal_ready_path(
    *, runtime: Any, config: MappingRuntimeConfig, args: Any
) -> Any:
    """Build the real bounded ingress-to-ready-B4 path for causal mapping.

    The implementation lives in a separate module to keep this entrypoint
    light when the transition profile is not requested.
    """

    from eligsir.runtime.causal_prepared_ready import (
        build_causal_prepared_ready_path,
    )

    return build_causal_prepared_ready_path(runtime=runtime, config=config, args=args)


def causal_mapping_application_options(
    composition: CausalMappingComposition,
    *,
    args: Any,
    ingress_telemetry: Any,
) -> MappingApplicationOptions:
    """Construct one causal application, with prepared mode only by opt-in.

    The legacy mapper remains the only mapper in the default composition.  In
    the transition profile it is deliberately absent: accepted packets cross
    the real prepared ingress handoff and no synchronous MappingFrame mapper
    is available as a fallback.
    """

    binding = composition.binding
    prepared = composition.prepared_ready_path
    return MappingApplicationOptions(
        mapper_sink=None if prepared is not None else binding.mapper_sink,
        optimizer_tick=binding.optimizer_tick,
        tracking_provider=ExternalTrackingProvider(),
        require_tracking=True,
        materialization_policy=binding.materialization_policy,
        materialization_backend=binding.materialization_backend,
        # Prepared ingress owns EOF polling/bootstrapping.  Calling the
        # legacy hook here would initialize from legacy Cameras before four
        # prepared device leases become ready.
        source_complete_hook=(
            binding.source_complete_hook
            if prepared is None
            else prepared.mark_source_complete
        ),
        flush_hook=binding.flush_hook,
        close_hook=binding.close_hook,
        observation_owner=binding.observation_owner,
        control_snapshot_provider=binding.control_snapshot_provider,
        disposition_sink=binding.disposition_sink if prepared is None else composition.runtime.record_prepared_disposition,
        ingress_telemetry_provider=ingress_telemetry,
        packet_capture=getattr(binding, "packet_capture", None),
        source_consumed_observer=binding.source_consumed_observer,
        runtime=composition.runtime,
        prepared_mode=prepared is not None,
        prepared_ingress_handoff=(None if prepared is None else prepared.handoff),
        prepared_runtime_bridge=(None if prepared is None else prepared.bridge),
        prepared_tick=(None if prepared is None else prepared.tick),
        prepared_close=(
            None if prepared is None else getattr(prepared, "close", None)
        ),
        prepared_progress_waiter=(
            None if prepared is None else prepared.wait_for_progress
        ),
    )


def _resolve_viewer_device(args: Any, runtime: Any) -> Any:
    """Resolve a torch device for the viewer's map-output lane at construction
    time, without requiring the Gaussian model to already be bootstrapped.

    Issue #1393: ``build_causal_viewer`` runs immediately after
    ``CanonicalMappingRuntime`` is constructed, before the first frame has
    been admitted/ticked -- ``create_from_pcd``/bootstrap only happens on the
    first ticked frame, so the model's ``nn.ParameterDict`` has no ``means``/
    ``sh0`` keys yet and ``backend.read_view()`` raises ``AttributeError``.
    Prefer the model's real parameter device when it happens to already be
    populated (e.g. resuming from a checkpoint), and otherwise fall back to
    the configured ``--data_device``, which is always available regardless of
    bootstrap state.

    Follow-up regression (found 2026-08-31, real GPU run): the fallback MUST
    return a fully-qualified device (an explicit CUDA/HIP index), never a
    bare ``torch.device("cuda")``. ``VersionedMapOutputLane._start_request``
    later compares this device for exact equality against a REAL sampled
    tensor's ``.device`` -- and PyTorch device equality is index-sensitive:
    ``torch.device("cuda") != torch.device("cuda", 0)`` even though both
    refer to the same physical GPU. A bare fallback device therefore never
    equals any real tensor's device once the model bootstraps, so every
    ``_start_request`` call raises (caught and silently dropped) and the map
    preview lane's ``poll_completed()`` never returns a snapshot -- Gaussians
    silently never reach the viewer even though ``try_submit`` reports
    success on every call. Normalizing to the concrete current device here
    is what makes the equality check in ``_start_request`` ever pass.
    """
    import torch

    backend = getattr(runtime, "backend", None)
    model = getattr(backend, "_model", None)
    if model is None:
        # Some backend duck-types (tests, alternate adapters) expose the
        # underlying model as a public `.model` attribute rather than the
        # canonical `GsplatMapBackend._model`.
        model = getattr(backend, "model", None)
    params = getattr(model, "params", None)
    if params is not None and "means" in params:
        return params["means"].device
    data_device = str(getattr(args, "data_device", "cuda") or "cuda")
    resolved = torch.device(data_device)
    if resolved.index is None and resolved.type == "cuda" and torch.cuda.is_available():
        resolved = torch.device(resolved.type, torch.cuda.current_device())
    return resolved


def build_causal_viewer(args: Any, runtime: Any) -> tuple[Any, Any, float]:
    """Construct the optional live viewer once, at startup (issue #1383).

    Zero-intrusion consumer (#1200/#1196/#1195): mirrors ``train_streaming.py``'s
    viewer construction and per-tick submission pattern exactly -- same
    ``make_viewer``/``CanonicalViewerAdapter``/``VersionedMapOutputLane``
    machinery, same ``population_count``/zero-copy ``read_view()`` reads, no
    new rasterization or synchronous map readback. Returns
    ``(viewer, map_output_lane, map_max_hz)``; ``viewer`` is ``None`` when the
    viewer is disabled (the default) -- callers must no-op in that case so an
    absent ``--viewer`` flag changes nothing about causal_mapping's behavior.
    """
    viewer_name = str(getattr(args, "viewer", "none") or "none")
    viewer_sinks_arg = str(getattr(args, "viewer_sinks", "") or "")
    viewer_sink_names = [s.strip() for s in viewer_sinks_arg.split(",") if s.strip()]
    if viewer_name == "none" and not viewer_sink_names:
        return None, None, 0.0

    from eligsir.viewers import make_viewer

    viewer_kwargs: dict[str, Any] = {}
    if viewer_name == "viser" or "viser" in viewer_sink_names:
        viewer_kwargs["port"] = getattr(args, "viewer_viser_port", 8080)
        viewer_kwargs["host"] = getattr(args, "viewer_viser_host", "0.0.0.0")
        viewer_kwargs["map_max_gaussians"] = getattr(args, "viewer_map_max_gaussians", 100_000)
    viewer = make_viewer(
        viewer_name, viewer_sinks=(viewer_sinks_arg or None), **viewer_kwargs
    )

    map_output_lane = None
    map_max_hz = 0.0
    if hasattr(viewer, "canonical_viewer") and hasattr(viewer, "bind_map_output_lane"):
        from eligsir.runtime.transfers import get_device_transfer_owner
        from eligsir.viewers.map_output_lane import VersionedMapOutputLane

        backend = getattr(runtime, "backend", None)
        # Zero-copy metadata read only (#1200 model 1): a borrowed tensor's
        # .device attribute, never the tensor contents. Resolved without
        # requiring the model to already be bootstrapped (issue #1393) --
        # see _resolve_viewer_device's docstring.
        device = _resolve_viewer_device(args, runtime)
        map_cfg = getattr(viewer.canonical_viewer, "config", None)
        # get_device_transfer_owner's registry is a process-wide singleton
        # keyed by device string (eligsir/runtime/transfers.py); the
        # FIRST caller to reach it for a given device wins and fixes its
        # max_pinned_bytes for every later caller. build_causal_viewer runs
        # before CanonicalMappingRuntime's own lazy _initialize() (which
        # requests the SAME "cuda:0" key with
        # args.streaming_max_transfer_bytes -- production_application.py's
        # _initialize()), so it must request that SAME byte limit here or
        # the later real call raises RuntimeError on a mismatched limit
        # (found 2026-08-31, real GPU run, after fixing #1393's device
        # index mismatch made both callers agree on the same device key for
        # the first time).
        max_transfer_bytes = int(
            getattr(args, "streaming_max_transfer_bytes", 2 * 1024 * 1024 * 1024)
        )
        map_output_lane = VersionedMapOutputLane(
            get_device_transfer_owner(device, max_pinned_bytes=max_transfer_bytes),
            max_gaussians=int(getattr(map_cfg, "map_max_gaussians", 100_000)),
        )
        viewer.bind_map_output_lane(map_output_lane)
        map_max_hz = float(getattr(map_cfg, "map_max_hz", 0.5) or 0.0)
    return viewer, map_output_lane, map_max_hz


def submit_causal_viewer_tick(
    viewer: Any,
    map_output_lane: Any,
    map_max_hz: float,
    backend: Any,
    tick: Any,
    *,
    last_map_wall_s_box: list[float],
    runtime: Any = None,
    trajectory_trail: Any = None,
) -> None:
    """Publish one tick's already-computed preview data. Never raises, never blocks.

    ``backend.population_count`` and ``tick.primary_training_psnr_db``/
    ``tick.parameter_version`` are all already-produced values from the
    canonical mapping tick -- no extra render, no extra snapshot. The map
    preview is submitted through the bounded ``VersionedMapOutputLane``
    (``try_submit`` never blocks; a busy lane just drops the request), gated
    by the viewer's own configured ``map_max_hz`` exactly as
    ``train_streaming.py`` gates it.

    ``runtime``/``trajectory_trail`` (both optional, default ``None`` --
    existing callers that omit them get byte-identical behavior) wire the
    live camera position + trajectory trail (issue #1407): ``runtime`` is the
    ``CanonicalMappingRuntime`` whose ``_last_admitted_pose`` (a copied
    ``(4, 4)`` c2w array, set only on real admission) is the freshest pose
    this application has observed. ``trajectory_trail`` is a caller-owned,
    already-bounded ``collections.deque`` of ``Pose`` -- unlike
    ``train_streaming.py``'s unboundedly-growing ``_trajectory_est``, the
    cap lives at the producer here. A pose identical to the trail's last
    entry is skipped (admission-driven ``_last_admitted_pose`` does not
    change every tick, and rebuilding/re-publishing an unchanged trajectory
    on every mapping tick would be pure waste for zero visual change).

    ``tick.telemetry`` (a ``RuntimeTelemetrySnapshot``) is likewise already
    produced by ``MappingCoordinator.tick()`` on every real tick
    (``production_application.py``'s ``self._telemetry.publish_tick(...)``,
    unconditional) -- forwarding it to ``log_runtime_telemetry_snapshot``
    populates the viewer's Decision/Freshness/GPU-Transfer/Map-Residency
    panels for free. This hook had zero production callers anywhere in the
    repo before this (found 2026-08-31): it exists on ``CanonicalViewerAdapter``
    but neither this trainer nor ``train_streaming.py`` ever called it, so
    those four panels stayed permanently on their "waiting for..." placeholder
    regardless of ``--viewer``.
    """
    if viewer is None:
        return
    iteration = int(getattr(tick, "iteration", 0))
    try:
        metrics: dict[str, float] = {
            "gaussian_count": float(getattr(backend, "population_count", 0))
        }
        psnr_db = getattr(tick, "primary_training_psnr_db", None)
        if psnr_db is not None:
            metrics["psnr"] = float(psnr_db)
        viewer.log_metrics(metrics, iteration)
    except Exception:  # noqa: BLE001 - a viewer error must never affect mapping
        pass
    telemetry = getattr(tick, "telemetry", None)
    if telemetry is not None and hasattr(viewer, "log_runtime_telemetry_snapshot"):
        try:
            viewer.log_runtime_telemetry_snapshot(telemetry, iteration=iteration)
        except Exception:  # noqa: BLE001 - a viewer error must never affect mapping
            pass
    if map_output_lane is not None:
        try:
            now = time.monotonic()
            if map_max_hz <= 0.0 or (now - last_map_wall_s_box[0]) >= (1.0 / map_max_hz):
                parameter_version = int(getattr(tick, "parameter_version", iteration))
                if map_output_lane.try_submit(
                    backend, parameter_version=parameter_version, iteration=iteration
                ):
                    last_map_wall_s_box[0] = now
                    viewer.notify_map_output()
        except Exception:  # noqa: BLE001 - a viewer error must never affect mapping
            pass
    if runtime is not None and trajectory_trail is not None:
        try:
            pose_arr = getattr(runtime, "_last_admitted_pose", None)
            if pose_arr is not None:
                last = trajectory_trail[-1] if trajectory_trail else None
                if last is None or not np.array_equal(last.matrix, pose_arr):
                    from eligsir.contracts.types import Pose

                    trajectory_trail.append(
                        Pose(np.array(pose_arr, dtype=np.float32, copy=True))
                    )
                    viewer.log_trajectory(list(trajectory_trail), None, iteration)
        except Exception:  # noqa: BLE001 - a viewer error must never affect mapping
            pass


def _write_causal_checkpoint_boundary(
    application: Any,
    runtime: Any,
    *,
    source_frame_count: int,
    model_path: str | Path,
    endpoint_dir_name: str,
    checkpoint_reason: str,
    prune_previous: bool = False,
) -> dict[str, Any]:
    """Flush the current model state and persist a durable checkpoint boundary.

    Shared by the pre-refinement ``mapping_endpoint`` boundary and the
    periodic ``refinement_checkpoint`` boundary written during post-mapping
    refinement (issue #1306) -- both need the identical
    flush-then-snapshot-then-persist sequence, just under a different
    directory name/reason tag.

    ``prune_previous=True`` deletes any prior ``chkpnt*.pth`` in
    ``endpoint_dir_name`` (after the new one lands, never before) so a long
    refinement run's periodic checkpoint keeps only its most recent
    ``chkpnt*.pth``/``point_cloud.ply`` under ``endpoint_dir_name`` itself,
    rather than one pair per interval. This does NOT bound disk growth end
    to end: ``application.flush()`` below also leaves its own
    ``point_cloud/optimizer_updates_<iter>_.../`` PLY behind under the run's
    ``point_cloud/`` directory on every call, and those per-interval
    directories are never pruned by this function.
    """

    import torch

    root = Path(model_path).resolve()
    application.flush()
    iteration = int(getattr(runtime, "iteration", 0))
    clock_owner = getattr(runtime, "_work_clock_owner", None)
    snapshot = getattr(clock_owner, "snapshot", None)
    if not callable(snapshot):
        raise RuntimeError("causal checkpoint boundary requires semantic work clocks")
    work = snapshot()
    if int(work.optimizer_updates) != iteration:
        raise RuntimeError(
            "causal checkpoint boundary optimizer clock disagrees with runtime iteration"
        )
    source_paths = point_cloud_artifact_paths(
        root,
        optimizer_updates=iteration,
        supervised_views=int(work.supervised_rgb_views),
    )
    source_ply = source_paths.point_cloud
    if not source_ply.is_file() or source_ply.stat().st_size <= 0:
        raise RuntimeError(
            "causal checkpoint flush did not produce a durable point-cloud endpoint"
        )

    endpoint_dir = root / endpoint_dir_name
    endpoint_dir.mkdir(parents=True, exist_ok=True)
    endpoint_ply = endpoint_dir / "point_cloud.ply"
    ply_temporary = endpoint_ply.with_suffix(endpoint_ply.suffix + ".tmp")
    shutil.copy2(source_ply, ply_temporary)
    os.replace(ply_temporary, endpoint_ply)

    backend = getattr(runtime, "backend", None)
    checkpoint_state = getattr(backend, "checkpoint_state", None)
    if not callable(checkpoint_state):
        raise RuntimeError(
            "causal checkpoint boundary requires backend checkpoint ownership"
        )
    blob = checkpoint_state()
    checkpoint = endpoint_dir / f"chkpnt{iteration}.pth"
    checkpoint_temporary = checkpoint.with_suffix(checkpoint.suffix + ".tmp")
    checkpoint_metadata = {
        "checkpoint_reason": checkpoint_reason,
        "source_frame_count": int(source_frame_count),
        "layout": str(blob.layout),
        "map_version": int(blob.map_version),
        "topology_version": int(blob.topology_version),
        "coordinate_frame_version": int(blob.coordinate_frame_version),
        "schema_version": int(blob.schema_version),
    }
    torch.save((blob.payload, iteration, checkpoint_metadata), checkpoint_temporary)
    os.replace(checkpoint_temporary, checkpoint)
    if prune_previous:
        # Prune AFTER the new checkpoint lands (never before): deleting the
        # old one first would leave a window with zero durable checkpoints
        # on disk if the process died between the delete and the new write.
        for stale in endpoint_dir.glob("chkpnt*.pth"):
            if stale != checkpoint:
                stale.unlink(missing_ok=True)

    endpoint: dict[str, Any] = {
        "iteration": iteration,
        "optimizer_updates": iteration,
        "supervised_views": int(work.supervised_rgb_views),
        "directory": str(endpoint_dir),
        "checkpoint": str(checkpoint),
        "point_cloud": str(endpoint_ply),
        "source_point_cloud": str(source_ply),
        "legacy_source_point_cloud": str(source_paths.legacy_point_cloud),
    }
    return endpoint


def _write_emergency_backend_checkpoint(
    runtime: Any,
    *,
    source_frame_count: int,
    model_path: str | Path,
    endpoint_dir_name: str,
    checkpoint_reason: str,
    prune_previous: bool = False,
) -> dict[str, Any]:
    """Persist a raw backend checkpoint WITHOUT flushing or a point-cloud snapshot.

    Issue #1306: this is deliberately weaker than
    ``_write_causal_checkpoint_boundary`` and exists only for the tick-loop
    exception handler. The real production incident this exists for was a
    ``torch.OutOfMemoryError`` with the GPU already at 90 MiB free --
    ``_write_causal_checkpoint_boundary``'s ``application.flush()`` calls
    ``gaussians.save_ply()``, which allocates fresh GPU tensors to format the
    PLY payload and can itself OOM immediately after the tick that just
    failed for the identical reason, losing the emergency checkpoint too.
    ``backend.checkpoint_state()`` only ``.detach()``es already-resident
    parameter tensors (see ``GsplatGaussianModel.capture``) -- no new GPU
    allocation -- so this path is the one still expected to succeed with
    (close to) zero free VRAM. The point cloud is reconstructible later from
    this checkpoint; skipping it here is the point.
    """

    import torch

    root = Path(model_path).resolve()
    iteration = int(getattr(runtime, "iteration", 0))
    backend = getattr(runtime, "backend", None)
    checkpoint_state = getattr(backend, "checkpoint_state", None)
    if not callable(checkpoint_state):
        raise RuntimeError(
            "causal emergency checkpoint requires backend checkpoint ownership"
        )
    blob = checkpoint_state()
    endpoint_dir = root / endpoint_dir_name
    endpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = endpoint_dir / f"chkpnt{iteration}.pth"
    checkpoint_temporary = checkpoint.with_suffix(checkpoint.suffix + ".tmp")
    checkpoint_metadata = {
        "checkpoint_reason": checkpoint_reason,
        "source_frame_count": int(source_frame_count),
        "layout": str(blob.layout),
        "map_version": int(blob.map_version),
        "topology_version": int(blob.topology_version),
        "coordinate_frame_version": int(blob.coordinate_frame_version),
        "schema_version": int(blob.schema_version),
        "point_cloud_snapshot": False,
    }
    torch.save((blob.payload, iteration, checkpoint_metadata), checkpoint_temporary)
    os.replace(checkpoint_temporary, checkpoint)
    if prune_previous:
        # Prune AFTER the new checkpoint lands -- see the identical comment
        # in _write_causal_checkpoint_boundary for why the order matters.
        for stale in endpoint_dir.glob("chkpnt*.pth"):
            if stale != checkpoint:
                stale.unlink(missing_ok=True)

    return {
        "iteration": iteration,
        "directory": str(endpoint_dir),
        "checkpoint": str(checkpoint),
        "point_cloud": None,
    }


def _publish_causal_mapping_endpoint(
    application: Any,
    runtime: Any,
    *,
    source_frame_count: int,
    model_path: str | Path,
) -> dict[str, Any]:
    """Persist the quiescent pre-refinement model boundary."""

    return _write_causal_checkpoint_boundary(
        application,
        runtime,
        source_frame_count=source_frame_count,
        model_path=model_path,
        endpoint_dir_name="mapping_endpoint",
        checkpoint_reason="mapping_endpoint",
    )


def _compute_refinement_progress_fraction(
    *, completed: int, requested: int, elapsed_s: float, requested_seconds: float
) -> float:
    """Progress fraction for the ``progressive`` refinement resolution policy.

    Issue #1384: the previous implementation hardcoded ``1.0`` whenever
    ``requested`` (the step-count budget) was ``0`` -- which is exactly the
    standard time-cap-only refinement mode this project's Run Contracts
    mandate (``--streaming_post_mapping_refinement_seconds`` without an
    explicit ``--streaming_post_mapping_refinement_steps``). A hardcoded
    ``1.0`` is always ``>= switch_frac``, so ``forces_native_post_refinement``
    returned ``True`` on the very first refinement tick, silently degrading
    ``progressive`` to plain ``native`` for the entire refinement window.

    Three cases:

    * ``requested > 0`` (step-count-governed mode, unchanged): a pure
      step-count fraction, deliberately not wall-clock-derived, so the
      policy's switch point is reproducible for a fixed step budget.
    * ``requested == 0`` and ``requested_seconds > 0`` (time-cap-only mode,
      the #1384 bug): ``elapsed_s / requested_seconds``, clamped to
      ``[0, 1]`` -- the real wall-clock cap the refinement loop is already
      tracking for its own stop condition is now also progressive's genuine
      progress signal.
    * ``requested == 0`` and ``requested_seconds == 0`` (no cap of any kind):
      there is no budget to measure progress against at all. In production
      this branch is unreachable -- the refinement loop's own ``while``
      condition never executes a single iteration when both budgets are
      zero, so this value is never actually consulted. For a caller that
      invokes this helper directly (e.g. a unit test), we return ``0.0``
      ("never switch yet") rather than ``1.0`` ("always switched"): with no
      evidence a budget is even in effect, the safer default is the one
      `_post_refinement_forces_native`'s own ``None``-fraction docstring
      already uses -- falling through to the measured-cost/bounded-like
      path, not silently assuming the run is nearly done.
    """

    if requested > 0:
        return completed / requested
    if requested_seconds > 0.0:
        if elapsed_s <= 0.0:
            return 0.0
        return min(1.0, elapsed_s / requested_seconds)
    return 0.0


def run_causal_post_mapping_refinement(
    application: Any, runtime: Any, *, source_frame_count: int,
    requested_steps: int, model_path: str | Path, mapping_wall_time_s: float,
    tick_observer: Callable[[Any], None] | None = None,
    retry_tick_observer: Callable[[Any], None] | None = None,
    tick_clock_observer: Callable[[Any, CausalReplayClock], None] | None = None,
    initial_clock: CausalReplayClock | None = None,
    mapping_endpoint_reporter: (
        Callable[[dict[str, Any]], Mapping[str, str] | None] | None
    ) = None,
    checkpoint_interval_steps: int = 0,
    requested_seconds: float = 0.0,
    branch_runtime: Any | None = None,
    control_probe_chunk_steps: int = 0,
    control_probe_max_rejections: int = 3,
    on_control_chunk_accepted: Callable[[], None] | None = None,
    allocator_phase_tracker: Any | None = None,
) -> dict[str, Any]:
    """Execute the requested post-source optimizer steps and persist endpoints.

    ``requested_seconds`` (0.0 disables it, the historical/default behavior)
    is a REAL wall-clock cap measured with ``time.perf_counter()`` from the
    moment the mapping endpoint is published to "now" on every loop
    iteration -- never an iteration-count proxy. When both ``requested_steps``
    and ``requested_seconds`` are positive, refinement stops at whichever
    limit is reached first. ``metadata["refinement_stop_reason"]`` records
    which one actually bound the run: ``"requested_steps_zero"`` (no
    refinement was requested at all), ``"step_cap"`` (ran to the full
    requested step count, time cap absent or never reached), or
    ``"time_cap"`` (the wall-clock cap fired before the step count was
    reached).
    """
    requested = int(requested_steps)
    if requested < 0:
        raise ValueError("requested_steps must be non-negative")
    time_cap_s = float(requested_seconds)
    if time_cap_s < 0:
        raise ValueError("requested_seconds must be non-negative")
    source_count = int(source_frame_count)
    if source_count < 0:
        raise ValueError("source_frame_count must be non-negative")
    if (tick_clock_observer is None) != (initial_clock is None):
        raise ValueError(
            "refinement tick_clock_observer and initial_clock must be provided together"
        )
    guard_active = branch_runtime is not None
    if guard_active and (requested <= 0 or time_cap_s > 0.0):
        raise ValueError(
            "control-probe refinement requires a positive step budget without a time cap"
        )
    if guard_active and int(control_probe_chunk_steps) <= 0:
        raise ValueError(
            "control_probe_chunk_steps must be positive when control-probe refinement is enabled"
        )
    if guard_active and int(control_probe_max_rejections) <= 0:
        raise ValueError(
            "control_probe_max_rejections must be positive when control-probe refinement is enabled"
        )
    root = Path(model_path).resolve()
    mapping_endpoint = _publish_causal_mapping_endpoint(
        application,
        runtime,
        source_frame_count=source_count,
        model_path=root,
    )
    mark_system_telemetry("mapping_endpoint", iteration=int(mapping_endpoint["iteration"]))
    mapping_iteration = int(mapping_endpoint["iteration"])
    mapping_steps = int(getattr(runtime, "optimizer_steps", 0))
    mapping_phase_path = (
        Path(mapping_endpoint["directory"]) / "streaming_phase_metadata.json"
    )
    mapping_endpoint["phase_metadata"] = str(mapping_phase_path)
    metadata = {
        "phase": "refinement",
        "source_idx": source_count,
        "source_frame_count": source_count,
        "source_coverage_fraction": 1.0,
        "mapping_endpoint": mapping_endpoint,
        "refined_endpoint": None,
        "mapping_optimizer_steps": mapping_steps,
        "refinement_optimizer_steps_requested": requested,
        "refinement_optimizer_steps": 0,
        "mapping_wall_time_s": float(mapping_wall_time_s),
        "refinement_wall_time_s": 0.0,
        "refinement_mapping_active_s": 0.0,
        "refinement_supervision_views": 0,
        "refinement_supervision_views_per_active_s": 0.0,
        "refinement_supervision_views_per_wall_s": 0.0,
        "refinement_requested_seconds": time_cap_s,
        "refinement_stop_reason": (
            "requested_steps_zero" if (requested == 0 and time_cap_s <= 0.0) else "step_cap"
        ),
        "refinement_rejected_control_chunks": 0,
        "refinement_control_probe_max_rejections": int(control_probe_max_rejections),
        "allocator_phase_samples": {
            "enabled": os.environ.get("SPLAT_REPORT_ALLOCATOR_PHASE_SAMPLES", "0") == "1",
            "available": False,
            "reason": "SPLAT_REPORT_ALLOCATOR_PHASE_SAMPLES=0",
            "samples": {},
        },
    }

    def write_phase_metadata(path: Path) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
        temporary.replace(path)

    write_phase_metadata(mapping_phase_path)
    write_phase_metadata(root / "streaming_phase_metadata.json")
    if mapping_endpoint_reporter is not None:
        # The report consumes the durable phase boundary (for phase-relative
        # PSNR alignment), so metadata must exist before report publication.
        mapping_endpoint.update(
            dict(mapping_endpoint_reporter(dict(mapping_endpoint)) or {})
        )
        required = (
            "report",
            "test_contact_sheet",
            "normals_contact_sheet",
            "trajectory_psnr_by_distance",
        )
        for name in required:
            path = Path(str(mapping_endpoint.get(name, "")))
            if not path.is_file() or path.stat().st_size <= 0:
                raise RuntimeError(
                    f"required mapping endpoint artifact {name!r} is missing or empty"
                )
        # Persist the artifact paths into both endpoint and root metadata only
        # after the complete bundle has passed its fail-closed validation.
        write_phase_metadata(mapping_phase_path)
        write_phase_metadata(root / "streaming_phase_metadata.json")

    allocator_phases = metadata["allocator_phase_samples"]
    if allocator_phases["enabled"] and allocator_phase_tracker is not None:
        try:
            sample = allocator_phase_tracker.phase_snapshot(
                "pre_refinement_handoff",
                iteration=mapping_iteration,
                frames_ingested=source_count,
                gaussians=getattr(runtime, "gaussians", None),
                streaming_scene=getattr(runtime, "scene", None),
                observation_db=None,
            )
            allocator_phases["samples"]["pre_refinement_handoff"] = sample
            if sample.get("status") == "measured":
                allocator_phases["available"] = True
                allocator_phases["reason"] = None
            else:
                allocator_phases["reason"] = str(
                    sample.get("reason") or "allocator phase sample unavailable"
                )
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            allocator_phases["samples"]["pre_refinement_handoff"] = {
                "phase": "pre_refinement_handoff",
                "status": "unavailable",
                "reason": reason,
            }
            allocator_phases["reason"] = reason
    elif allocator_phases["enabled"]:
        allocator_phases["reason"] = "no_existing_gpu_memory_residency_tracker"
    write_phase_metadata(mapping_phase_path)
    write_phase_metadata(root / "streaming_phase_metadata.json")
    enter_refinement = getattr(runtime, "enter_post_mapping_refinement", None)
    if not callable(enter_refinement):
        raise RuntimeError("causal refinement requires typed post-refinement phase ownership")
    enter_refinement(mapping_endpoint)
    mark_system_telemetry(
        "refinement_start", requested_steps=requested, requested_seconds=time_cap_s
    )
    completed = 0
    started = time.perf_counter()
    refinement_mapping_active_s = 0.0
    refinement_supervision_views = 0
    checkpoint_interval = max(0, int(checkpoint_interval_steps))
    last_checkpoint_completed = 0
    retrying_prepared_ready = False
    retry_trace_emitted = False
    accepted_control_probe = (
        _control_probe_partition_means(branch_runtime) if guard_active else None
    )
    candidate_branch_state = None
    candidate_completed = 0
    candidate_mapping_active_s = 0.0
    candidate_supervision_views = 0
    candidate_result = None
    accepted_control_publisher = None
    # Allow refinement if EITHER steps or time cap is positive (issue #1369 follow-up).
    # When requested_steps==0 but requested_seconds>0, refinement runs until the time cap.
    while completed < requested or (requested == 0 and time_cap_s > 0.0):
        if time_cap_s > 0.0 and (time.perf_counter() - started) >= time_cap_s:
            # Real elapsed-time cap, not an iteration-count proxy: measured
            # from the mapping-endpoint boundary (`started`, set just above)
            # to the wall-clock instant of this check, independent of how
            # fast/slow this host's GPU realizes optimizer steps.
            metadata["refinement_stop_reason"] = "time_cap"
            break
        # issue #1346 follow-up ("progressive" refinement resolution policy),
        # issue #1384 fix: step-count fraction when a step budget governs the
        # run, elapsed/requested wall-clock fraction when a time cap governs
        # it instead -- see `_compute_refinement_progress_fraction`'s
        # docstring for the full case breakdown and rationale.
        refinement_progress_fraction = _compute_refinement_progress_fraction(
            completed=completed,
            requested=requested,
            elapsed_s=time.perf_counter() - started,
            requested_seconds=time_cap_s,
        )
        if guard_active and candidate_branch_state is None:
            assert branch_runtime is not None
            candidate_branch_state = branch_runtime.checkpoint_state()
        if not retrying_prepared_ready:
            remaining_updates = requested - completed
            if guard_active:
                remaining_updates = min(
                    remaining_updates,
                    int(control_probe_chunk_steps) - candidate_completed,
                )
            runtime.rearm_post_mapping_work(
                optimizer_updates=remaining_updates,
                refinement_progress_fraction=refinement_progress_fraction,
            )
        before = int(getattr(runtime, "optimizer_steps", mapping_steps + completed))
        tick_started = time.perf_counter()
        loss_context = getattr(runtime, "post_mapping_refinement_loss_context", None)
        try:
            with loss_context() if callable(loss_context) else nullcontext():
                result = application.tick()
        except Exception as tick_error:
            if guard_active and candidate_branch_state is not None:
                assert branch_runtime is not None
                _restore_accepted_map_branch(
                    application, branch_runtime, candidate_branch_state
                )
            # Issue #1306: a real, catchable CUDA OOM during backward has been
            # observed mid-refinement in production (fr3_..._issue1259_v3,
            # 2026-08-29), aborting a 40,000-step refinement run with only the
            # pre-refinement mapping_endpoint/ surviving -- everything since
            # the last periodic checkpoint above was lost. Best-effort attempt
            # a final emergency checkpoint before propagating the original
            # error; a failure here must never mask `tick_error`.
            print(
                f"[causal_refinement] tick FAILED at completed={completed} "
                f"({type(tick_error).__name__}: {tick_error}) -- attempting "
                "emergency checkpoint before re-raising...",
                flush=True,
            )
            try:
                import torch as _torch
                if _torch.cuda.is_available():
                    _torch.cuda.empty_cache()
            except Exception as cache_error:
                print(
                    f"[causal_refinement] emergency cuda.empty_cache() failed: "
                    f"{cache_error}",
                    flush=True,
                )
            try:
                emergency_endpoint = _write_emergency_backend_checkpoint(
                    runtime,
                    source_frame_count=source_count,
                    model_path=root,
                    endpoint_dir_name="refinement_checkpoint",
                    checkpoint_reason=f"emergency_{type(tick_error).__name__}",
                    prune_previous=True,
                )
                print(
                    "[causal_refinement] emergency checkpoint written: "
                    f"{emergency_endpoint['checkpoint']}",
                    flush=True,
                )
            except Exception as checkpoint_error:
                print(
                    "[causal_refinement] emergency checkpoint FAILED (original "
                    f"error still propagating): {checkpoint_error}",
                    flush=True,
                )
            raise
        tick_elapsed_s = max(0.0, time.perf_counter() - tick_started)
        after = int(getattr(runtime, "optimizer_steps", before))
        delta = after - before
        if delta <= 0:
            delta = int(getattr(result, "optimizer_steps", 0))
        if (
            delta <= 0
            and bool(getattr(result, "retryable", False))
        ):
            # The normal tick observer is deliberately reserved for completed
            # refinement work so evaluation never mistakes a READY poll for a
            # mapping transaction.  An explicitly configured decision trace
            # may still need this retry record to diagnose a stalled exact
            # prepared plan.
            if retry_tick_observer is not None and not retry_trace_emitted:
                retry_tick_observer(result)
                retry_trace_emitted = True
            retrying_prepared_ready = True
            wait_for_progress = getattr(application, "wait_for_prepared_progress", None)
            if callable(wait_for_progress):
                wait_for_progress(timeout_s=READY_WAIT_FALLBACK_S)
            continue
        retrying_prepared_ready = False
        retry_trace_emitted = False
        # Allow delta==0 when running on time cap only (requested==0); reject if steps
        # were explicitly requested but not realized (issue #1374 part 2).
        # Part 3: Only check step-count constraints when requested > 0 (not in time-cap-only mode).
        # In time-cap-only mode (requested==0), any delta value (including >0) is acceptable.
        remaining_updates = requested - completed
        if guard_active:
            remaining_updates = min(
                remaining_updates,
                int(control_probe_chunk_steps) - candidate_completed,
            )
        if requested > 0 and (delta <= 0 or delta > remaining_updates):
            state = getattr(result, "state", None)
            state = getattr(state, "value", state)
            raise RuntimeError(
                "causal post-mapping refinement did not realize requested optimizer steps: "
                f"state={state} reason={getattr(result, 'reason', None)} "
                f"plan_id={getattr(getattr(result, 'work_plan', None), 'plan_id', None)}"
            )
        if guard_active:
            candidate_completed += delta
            candidate_mapping_active_s += tick_elapsed_s
            candidate_supervision_views += int(
                getattr(result, "supervision_views", 0)
            )
            candidate_result = result
            if (
                candidate_completed < int(control_probe_chunk_steps)
                and completed + candidate_completed < requested
            ):
                continue
            assert branch_runtime is not None
            assert accepted_control_probe is not None
            try:
                candidate_control_probe = _control_probe_partition_means(branch_runtime)
            except Exception:
                _restore_accepted_map_branch(
                    application, branch_runtime, candidate_branch_state
                )
                raise
            if not _control_probe_candidate_is_accepted(
                accepted_control_probe, candidate_control_probe
            ):
                _restore_accepted_map_branch(
                    application, branch_runtime, candidate_branch_state
                )
                metadata["refinement_rejected_control_chunks"] += 1
                if (
                    metadata["refinement_rejected_control_chunks"]
                    >= int(control_probe_max_rejections)
                ):
                    metadata["refinement_stop_reason"] = "control_probe_rejected"
                    write_phase_metadata(mapping_phase_path)
                    write_phase_metadata(root / "streaming_phase_metadata.json")
                    raise RuntimeError(
                        "causal post-mapping refinement control-probe acceptance "
                        f"rejected {metadata['refinement_rejected_control_chunks']} chunks"
                    )
                candidate_branch_state = None
                candidate_completed = 0
                candidate_mapping_active_s = 0.0
                candidate_supervision_views = 0
                candidate_result = None
                continue
            accepted_control_probe = candidate_control_probe
            accepted_control_publisher = on_control_chunk_accepted
            completed += candidate_completed
            refinement_mapping_active_s += candidate_mapping_active_s
            refinement_supervision_views += candidate_supervision_views
            result = candidate_result
            candidate_branch_state = None
            candidate_completed = 0
            candidate_mapping_active_s = 0.0
            candidate_supervision_views = 0
            candidate_result = None
        else:
            completed += delta
            if delta > 0:
                refinement_mapping_active_s += tick_elapsed_s
            refinement_supervision_views += int(
                getattr(result, "supervision_views", 0)
            )
        if tick_observer is not None:
            tick_observer(result)
        if tick_clock_observer is not None:
            assert initial_clock is not None
            tick_clock_observer(
                result,
                CausalReplayClock(
                    optimizer_steps=initial_clock.optimizer_steps + completed,
                    supervision_views=(
                        initial_clock.supervision_views
                        + refinement_supervision_views
                    ),
                    mapping_active_s=(
                        initial_clock.mapping_active_s
                        + refinement_mapping_active_s
                    ),
                    elapsed_s=(
                        initial_clock.elapsed_s
                        + max(0.0, time.perf_counter() - started)
                    ),
                    source_frame_index=initial_clock.source_frame_index,
                ),
            )
        if accepted_control_publisher is not None:
            publisher = accepted_control_publisher
            accepted_control_publisher = None
            publisher()
        if (
            checkpoint_interval > 0
            and completed - last_checkpoint_completed >= checkpoint_interval
            and completed < requested
        ):
            # Issue #1306: durable mid-refinement boundary so a later crash
            # loses at most `checkpoint_interval` steps of RECOVERABLE STATE
            # (a renderable/reportable model snapshot), not the whole
            # refinement run -- not a resume path: nothing in this trainer
            # loads a refinement_checkpoint/ and continues from it, same
            # caveat as the mapping_endpoint checkpoint. Overwrites the
            # previous periodic checkpoint's chkpnt*.pth/point_cloud.ply
            # (prune_previous=True) -- this is a snapshot boundary, not an
            # archive of every interval. Best-effort: a transient failure
            # here must not abort an otherwise-healthy refinement run.
            try:
                periodic_endpoint = _write_causal_checkpoint_boundary(
                    application,
                    runtime,
                    source_frame_count=source_count,
                    model_path=root,
                    endpoint_dir_name="refinement_checkpoint",
                    checkpoint_reason="periodic_refinement",
                    prune_previous=True,
                )
                metadata["last_refinement_checkpoint"] = periodic_endpoint
                write_phase_metadata(root / "streaming_phase_metadata.json")
                last_checkpoint_completed = completed
            except Exception as periodic_error:
                print(
                    "[causal_refinement] periodic checkpoint at "
                    f"completed={completed} FAILED (continuing refinement): "
                    f"{periodic_error}",
                    flush=True,
                )
    application.flush()
    mark_system_telemetry("refinement_end", refinement_optimizer_steps=completed)
    refined_iteration = int(getattr(runtime, "iteration", mapping_iteration))
    clock_owner = getattr(runtime, "_work_clock_owner", None)
    snapshot = getattr(clock_owner, "snapshot", None)
    if not callable(snapshot):
        raise RuntimeError("causal refinement endpoint requires semantic work clocks")
    refined_work = snapshot()
    if int(refined_work.optimizer_updates) != refined_iteration:
        raise RuntimeError(
            "causal refinement endpoint optimizer clock disagrees with runtime iteration"
        )
    refined_paths = point_cloud_artifact_paths(
        root,
        optimizer_updates=refined_iteration,
        supervised_views=int(refined_work.supervised_rgb_views),
    )
    refined_endpoint = refined_paths.point_cloud
    if not refined_endpoint.is_file():
        raise RuntimeError("causal refinement flush did not produce a durable point-cloud endpoint")
    refinement_wall_time_s = max(0.0, time.perf_counter() - started)
    metadata.update(
        {
            "phase": "complete",
            "refined_endpoint": {
                "point_cloud": str(refined_endpoint),
                "legacy_point_cloud": str(refined_paths.legacy_point_cloud),
                "optimizer_updates": refined_iteration,
                "supervised_views": int(refined_work.supervised_rgb_views),
            },
            "refinement_optimizer_steps": completed,
            "refinement_wall_time_s": refinement_wall_time_s,
            "refinement_mapping_active_s": refinement_mapping_active_s,
            "refinement_supervision_views": refinement_supervision_views,
            "refinement_supervision_views_per_active_s": (
                0.0
                if refinement_mapping_active_s <= 0.0
                else refinement_supervision_views / refinement_mapping_active_s
            ),
            "refinement_supervision_views_per_wall_s": (
                0.0
                if refinement_wall_time_s <= 0.0
                else refinement_supervision_views / refinement_wall_time_s
            ),
        }
    )
    write_phase_metadata(root / "streaming_phase_metadata.json")
    return metadata


def _build_ready_drain_block_from_owner(ready_owner: Any, *, adapter: Any | None = None) -> dict[str, Any]:
    """Build terminal READY drain counters from the owner without error suppression.

    Issue #1663: raise if counters are missing or malformed; never swallow
    exceptions. A missing key must indicate legacy (non-prepared) run, not
    a swallowed error.
    """
    from eligsir.runtime.causal_prepared_ready import _capture_ready_drain_counters

    return _capture_ready_drain_counters(ready_owner, adapter=adapter)


def causal_replay_summary_payload(
    summary: Any,
    *,
    source_hz: float | None,
    curve_metadata: Any,
    ready_bytes_block: dict[str, Any],
    ready_drain_block: dict[str, Any] | None = None,
    supervision_work_block: dict[str, Any],
    runtime: Any | None = None,
    experiment_origin: Mapping[str, Any] | None = None,
    experiment_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the ``causal_replay_summary.json`` payload.

    Read already-published runtime counters without waiting for device feedback;
    the remaining report blocks are pure input data.
    """
    # Issue #805: canonical ingress funnel vocabulary, boundaries, and rate surfaces.
    from eligsir.io.ingress_funnel import ingress_funnel_report_block

    ingress_funnel_block: dict[str, Any] = {}
    try:
        # Wire rate surfaces from summary counters and wall time.
        # source_observed_hz: offered frames / elapsed_s
        # materialized_valid_hz: frames that completed decode / elapsed_s
        source_ingress_block = summary.source_ingress or {}
        block = ingress_funnel_report_block(
            source_observed_count=summary.source_offered,
            materialized_valid_count=int(source_ingress_block.get("ingress_frames_admitted", 0)),
            wall_time_s=summary.elapsed_s if summary.elapsed_s > 0.0 else None,
        )
        if block:
            ingress_funnel_block = {"ingress_funnel": block}
    except Exception:
        pass
    throughput_block: dict[str, Any] = {}
    if curve_metadata is not None and not bool(curve_metadata["throughput_evidence_valid"]):
        throughput_block = {
            "throughput_evidence_valid": False,
            "throughput_evidence_status": str(curve_metadata["throughput_evidence_status"]),
        }
    replay_challenger = None  # No runtime is unavailable, not measured zero.
    if runtime is not None:
        telemetry = runtime.historical_usefulness_telemetry()
        if not isinstance(telemetry, Mapping) or any(type(value) is not int or value < 0 for value in telemetry.values()):
            raise TypeError("runtime historical usefulness telemetry must contain nonnegative integer counters")
        replay_challenger = dict(telemetry)
    # A free-form origin label is not execution provenance.  The startup
    # contracts already validate both the immutable Ablator declaration and
    # ResearchFlow registration; this report-only consumer merely combines
    # those established facts and fails closed for direct or undeclared runs.
    evidence_class = "UNMANAGED"
    if (
        isinstance(experiment_context, Mapping)
        and experiment_context.get("execution_management") == "ABLATOR"
        and experiment_context.get("gradeability") == "GRADEABLE_DECLARED"
    ):
        from eligsir.utils.experiment_origin import is_valid_researchflow_origin

        valid_origin, _ = is_valid_researchflow_origin(experiment_origin, env={})
        if valid_origin:
            evidence_class = "MANAGED"
    return {
        "evidence_class": evidence_class,
        "requested_source_fps": source_hz,
        **throughput_block,
        **asdict(summary),
        "prepared_insertion_skip_total": summary.prepared_insertion_skip_total,
        "replay_challenger": replay_challenger,
        **ready_bytes_block,
        **(ready_drain_block or {}),
        **supervision_work_block,
        **ingress_funnel_block,
    }


def write_causal_trajectory(model_path: str | Path, frames: tuple[Any, ...] | list[Any]) -> Path:
    """Persist the source trajectory consumed by the causal replay driver.

    This is the authoritative per-frame source artifact for the canonical
    runtime path. Timestamps come from the frame's source timestamp fields;
    they are never reconstructed from frame IDs. A missing pose or timestamp
    is an integrity error rather than an omitted/ fabricated trajectory row.
    """
    output = Path(model_path) / "trajectory_full.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as handle:
        for line_number, frame in enumerate(frames, start=1):
            ordinal = getattr(frame, "source_ordinal", None)
            if ordinal is None:
                ordinal = getattr(frame, "index", None)
            timestamp_ns = getattr(frame, "color_ts_ns", None)
            timestamp = (
                float(timestamp_ns) / 1_000_000_000.0
                if timestamp_ns is not None
                else getattr(frame, "timestamp", None)
            )
            pose = getattr(frame, "c2w", None)
            if pose is None:
                pose_obj = getattr(frame, "pose", None)
                pose = getattr(pose_obj, "c2w", None) if pose_obj is not None else None
            if ordinal is None or timestamp is None or pose is None:
                raise ValueError(
                    f"source trajectory frame {line_number} lacks an ordinal, "
                    "source timestamp, or pose"
                )
            timestamp = float(timestamp)
            matrix = np.asarray(pose, dtype=np.float64)
            if not np.isfinite(timestamp) or matrix.shape != (4, 4) or not np.isfinite(matrix).all():
                raise ValueError(f"invalid source trajectory frame {line_number}")
            entry = {
                "id": int(ordinal),
                "timestamp": timestamp,
                "position": matrix[:3, 3].tolist(),
                "rotation": matrix[:3, :3].tolist(),
            }
            handle.write(json.dumps(entry, separators=(",", ":")) + "\n")
    return output


def _stream_jsonl_rows(path: Path, *, max_bytes: int | None = None) -> list[dict[str, Any]]:
    """Parse a JSONL file one line at a time, never materializing it whole.

    Unlike ``path.read_text().splitlines()`` (which allocates the entire
    file as one string, then a second full list of lines), this walks the
    file handle's own buffered iterator so at most one raw line is held
    outside the accumulated ``rows`` list -- bounded working memory for
    arbitrarily long histories.

    ``max_bytes`` bounds the read to a byte-exact prefix of the file (a
    writer that only ever appends complete ``"...\\n"`` lines, such as
    ``JsonlDecisionTraceObserver``, guarantees any byte offset it is caught
    at sits exactly on a line boundary). This is how a caller recovers a
    stable "as of this moment" prefix from a file that keeps growing after
    that moment -- e.g. the decision trace, which refinement keeps
    appending to long after mapping's own snapshot was taken.
    """
    rows: list[dict[str, Any]] = []
    consumed = 0
    with path.open("rb") as handle:
        for raw_line in handle:
            if max_bytes is not None and consumed >= max_bytes:
                break
            consumed += len(raw_line)
            stripped = raw_line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def write_mapped_camera_trajectory_plot(
    model_path: str | Path,
    trajectory_path: str | Path,
    decision_trace_path: str | Path | None,
    *,
    successful_supervision_uses: dict[int, int] | None = None,
    decision_trace_byte_limit: int | None = None,
    out_name: str = "mapped_camera_trajectory",
) -> Path | None:
    """Write a camera-selection plot from trace rows or completed source uses.

    ``out_name`` distinguishes the mapping-phase plot (the default,
    ``mapped_camera_trajectory.svg``) from the post-refinement one
    (``mapped_camera_trajectory_refined``, see ``run_causal_mapping``'s
    second call site) -- refinement keeps drawing supervision from the
    cached/replayed frame pool well after mapping ends, so the
    mapping-phase-only snapshot goes stale the moment refinement starts. A
    single call site cannot serve both moments: by the time refinement
    finishes, mapping's own selection is long over and must stay visible
    for comparison, not be silently overwritten.

    ``successful_supervision_uses`` (issue #1687) is the authoritative
    source of which cameras actually supervised mapping/refinement; the
    decision trace, when present, is consulted only for extra
    ``mapping_candidate`` provenance decoration
    (``mapped_camera_trajectory_selection`` never derives used-source
    identity from it when ``successful_supervision_uses`` is given).
    ``decision_trace_byte_limit`` (issue #1653), when set, bounds the trace
    read to the byte prefix that existed at the moment the caller snapshot
    ``successful_supervision_uses`` -- so a trace that keeps growing after
    that moment (e.g. refinement continuing to append to it) cannot leak
    later decisions into an earlier phase's plot.
    """
    trace = None if decision_trace_path is None else Path(decision_trace_path)
    if trace is None or not trace.is_file():
        decision_rows: list[dict[str, Any]] = []
    else:
        decision_rows = _stream_jsonl_rows(trace, max_bytes=decision_trace_byte_limit)
    if not decision_rows and not successful_supervision_uses:
        return None
    trajectory_rows = _stream_jsonl_rows(Path(trajectory_path))
    from eligsir.utils.trajectory_eval import plot_mapped_camera_trajectory

    root = Path(model_path)
    selection = plot_mapped_camera_trajectory(
        str(root / f"{out_name}.svg"),
        trajectory_rows,
        decision_rows,
        successful_supervision_uses=successful_supervision_uses,
    )
    (root / f"{out_name}.json").write_text(
        json.dumps(selection, indent=2, sort_keys=True) + "\n"
    )
    return root / f"{out_name}.svg"


def ingress_telemetry_from_config(
    config: MappingRuntimeConfig, *, paced: bool
) -> CausalIngressTelemetryState | None:
    """Bind transport-pressure thresholds to the realized supervision batch."""

    if not paced:
        return None
    return CausalIngressTelemetryState(
        batch_size=int(config.hard_limits["optimization_batch_size"])
    )


def decision_trace_observer_from_args(
    *, dataset: Any, args: Any
) -> JsonlDecisionTraceObserver | None:
    """Resolve the explicit debug trace without adding normal-path I/O."""

    configured = str(getattr(args, "causal_decision_trace", "") or "").strip()
    if not configured:
        return None
    path = Path(configured)
    if not path.is_absolute():
        path = Path(dataset.model_path) / path
    path.parent.mkdir(parents=True, exist_ok=True)
    return JsonlDecisionTraceObserver(path)


class _SourceCompleteEvaluationGate:
    """Keep evaluator EOF behind prepared bootstrap completion."""

    def __init__(self, application: Any, evaluation_observer: Any) -> None:
        self._application = application
        self._evaluation_observer = evaluation_observer
        self._pending = False
        self._emitted = False
        self._last_tick: Any | None = None
        self._last_executed_tick: Any | None = None

    @staticmethod
    def _prepared_pending(value: Any) -> bool:
        state = getattr(value, "state", None)
        return getattr(state, "value", state) == "bootstrap_pending"

    def _remember_tick(self, tick: Any) -> None:
        """Retain the real tick that established the endpoint map state.

        ``mark_source_complete()`` returns a phase-transition result in the
        prepared path.  That result deliberately has no optimizer work and
        must never overwrite the last executed supervision transaction used
        by the evaluator.
        """

        self._last_tick = tick
        if (
            bool(getattr(tick, "executed", False))
            and int(getattr(tick, "optimizer_steps", 0)) > 0
        ):
            self._last_executed_tick = tick

    def _emit(self) -> None:
        tick = self._last_executed_tick or self._last_tick
        if (
            tick is not None
            and self._evaluation_observer is not None
            and not self._emitted
        ):
            self._evaluation_observer.source_complete(tick)
            self._emitted = True

    def source_complete(self, tick: Any) -> None:
        self._remember_tick(tick)
        completion = self._application.mark_source_complete()
        if self._prepared_pending(completion) or not self._application.source_complete:
            self._pending = True
            return
        self._emit()

    def observe_tick(self, tick: Any) -> None:
        self._remember_tick(tick)
        if not self._pending or self._prepared_pending(tick):
            return
        completion = self._application.mark_source_complete()
        if self._prepared_pending(completion) or not self._application.source_complete:
            return
        self._pending = False
        self._emit()


def apply_residency_budget(args: Any, budget: Any) -> None:
    """Clamp args window settings when residency budget is starved.

    When the residency budget cannot accommodate the requested keyframe window,
    this function clamps both `streaming_keyframe_window` and
    `streaming_resolution_cache_max_frames` to `budget.window_effective`.

    Args:
        args: Configuration namespace (modified in place).
        budget: ResidencyBudget object from resolve_residency_budget.
    """
    if budget.starved:
        args.streaming_keyframe_window = budget.window_effective
        args.streaming_resolution_cache_max_frames = budget.window_effective


def resolve_decode_workers_for_args(
    args: Any,
    *,
    native_width: int | None,
    native_height: int | None,
    cpu_count_fn: Any = os.cpu_count,
) -> None:
    """Resolve `args.causal_decode_workers` per `--causal_decode_workers_policy`.

    "fixed" (the default) leaves `args.causal_decode_workers` exactly as
    parsed -- byte-identical to pre-existing behavior. "cpu_scaled" derives a
    worker count from the estimated per-frame decode cost at the source's
    native resolution (see `payload_budget.estimate_decode_ms_per_frame`) and
    overwrites `args.causal_decode_workers` in place, so the existing
    `int(getattr(args, "causal_decode_workers", 2))` construction site needs
    no change.

    Args:
        args: Configuration namespace (modified in place).
        native_width: Source native frame width, or None if unresolved.
        native_height: Source native frame height, or None if unresolved.
        cpu_count_fn: Injectable `os.cpu_count`-shaped probe for testing.
    """
    from eligsir.runtime.payload_budget import (
        estimate_decode_ms_per_frame,
        resolve_decode_workers,
    )

    # Some tests exercise this resolver with a minimal/bare args double that
    # predates this flag (e.g. a plain `object()`). Leave those untouched
    # rather than requiring every caller to carry the new attributes.
    if not hasattr(args, "causal_decode_workers_policy") or not hasattr(
        args, "causal_decode_workers"
    ):
        return

    policy = args.causal_decode_workers_policy
    fixed_workers = args.causal_decode_workers
    if policy == "cpu_scaled" and native_width and native_height:
        per_frame_decode_ms_estimate = estimate_decode_ms_per_frame(
            native_width, native_height
        )
    else:
        per_frame_decode_ms_estimate = 0.0

    workers, metadata = resolve_decode_workers(
        policy=policy,
        fixed_workers=fixed_workers,
        cpu_count=cpu_count_fn(),
        source_hz=args.streaming_input_fps_cap,
        per_frame_decode_ms_estimate=per_frame_decode_ms_estimate,
    )
    args.causal_decode_workers = workers
    args._resolved_decode_workers_metadata = metadata


def resolve_causal_mapping_composition(
    *, dataset: Any, opt: Any, pipe: Any, args: Any
) -> CausalMappingComposition:
    """Resolve the source, immutable config and one runtime binding once."""

    source = make_frame_source(dataset.source_path, args)

    # --- Payload budget policy resolution (Resolution- and device-aware defaults) ---
    # Resolve the admitted-payload frame count and hot-tier byte cap based on
    # native resolution and device memory (if not explicitly passed).
    # Must happen BEFORE mapping_runtime_config_from_args below so the resolved
    # values are available when the residency policy is frozen.
    from eligsir.runtime.config_boundary import resolve_admitted_payload_bounds, source_native_resolution
    _native_res, _native_method = source_native_resolution(source)
    _native_w = None
    _native_h = None
    if _native_res is not None:
        _native_w, _native_h = _native_res
    _payload_budget_frames, _payload_budget_hot_bytes, _payload_budget_metadata = (
        resolve_admitted_payload_bounds(
            args,
            native_width=_native_w,
            native_height=_native_h,
        )
    )
    _policy_str = _payload_budget_metadata.get("policy", "unknown")
    _frames_source = _payload_budget_metadata.get("source", "unknown")
    _hot_source = _payload_budget_metadata.get("hot_tier_source", "unknown")

    # Resolve --causal_decode_workers_policy before _PacedPacketProducer/
    # decode_workers= construction reads args.causal_decode_workers (see
    # resolve_decode_workers_for_args). Must happen alongside the payload
    # budget resolution above so both share one [payload-budget] log line.
    resolve_decode_workers_for_args(
        args,
        native_width=_native_w,
        native_height=_native_h,
    )
    _decode_workers_metadata = getattr(args, "_resolved_decode_workers_metadata", None)
    if _decode_workers_metadata is not None:
        _decode_workers_suffix = (
            f" decode_workers={args.causal_decode_workers}"
            f"(policy={_decode_workers_metadata['policy']})"
        )
    else:
        _decode_workers_suffix = ""

    # Handle residency policy: apply budget clamping and use format_budget_line
    if _policy_str == "residency" and "residency" in _payload_budget_metadata:
        from eligsir.runtime.payload_budget import format_budget_line
        _residency_budget = _payload_budget_metadata["residency"]
        apply_residency_budget(args, _residency_budget)
        print(format_budget_line(_residency_budget) + _decode_workers_suffix, flush=True)
        args._resolved_prefetch_entries = _residency_budget.prefetch_entries
    else:
        # Legacy/resolution_scaled path
        _device_mem_str = ""
        if _payload_budget_metadata.get("device_total_memory"):
            _dev_mem_gb = _payload_budget_metadata["device_total_memory"] / 1e9
            _hot_bytes_gb = _payload_budget_hot_bytes / 1e9
            _device_mem_str = f" (20% of {_dev_mem_gb:.1f} GB)"
        if _native_w and _native_h:
            _native_str = f"{_native_w}x{_native_h} via={_native_method}"
        else:
            _native_str = "unknown"
        print(
            f"[payload-budget] policy={_policy_str} native={_native_str} "
            f"admitted_payload_max_frames={_payload_budget_frames} (from 512@640x480, {_frames_source}) "
            f"hot_tier_max_bytes={_payload_budget_hot_bytes} ({_hot_source}){_device_mem_str}"
            f"{_decode_workers_suffix}",
            flush=True,
        )

    # Store resolved values in args for later use (e.g., summary JSON)
    args._resolved_payload_budget_policy = _policy_str
    args._resolved_native_width = _native_w  # For config_boundary resolver
    args._resolved_native_height = _native_h  # For config_boundary resolver
    args._resolved_native_resolution_method = _native_method  # For summary block
    args._resolved_payload_budget_frames = _payload_budget_frames
    args._resolved_payload_budget_hot_bytes = _payload_budget_hot_bytes

    # 2026-09-07 (post-mapping refinement "all"-mode catalog collapse, see
    # `eligsir.runtime.config_boundary.resolve_cold_payload_bounds`'s
    # docstring): `--streaming_observation_db_retain_all` must be resolved
    # BEFORE `mapping_runtime_config_from_args` below bakes the DB's cold-
    # payload frame/byte bounds into `config`/the `FrameObservationDB`
    # construction that follows -- auto-enabling it later (e.g. next to
    # `configure_post_mapping_refinement_sampling`, which only runs once
    # source completion is reached) would be long after eviction has
    # already happened. A user-set `True` is never overridden back to
    # `False`; a `sampling="pool"` run leaves the flag exactly as parsed
    # (byte-identical to before this existed).
    sampling_mode = str(
        getattr(args, "streaming_post_mapping_refinement_sampling", "pool") or "pool"
    )
    if sampling_mode != "pool" and not bool(
        getattr(args, "streaming_observation_db_retain_all", False)
    ):
        args.streaming_observation_db_retain_all = True
    config = mapping_runtime_config_from_args(args)
    canonical_runtime, compatibility_type = _canonical_runtime_types()
    runtime = canonical_runtime(
        config,
        compatibility_type(dataset=dataset, opt=opt, pipe=pipe),
    )
    prepared = (
        _build_prepared_causal_ready_path(runtime=runtime, config=config, args=args)
        if _prepared_ready_b4_enabled(args)
        else None
    )
    return CausalMappingComposition(
        source=source,
        config=config,
        binding=runtime_binding_from_runtime(runtime, args=args),
        runtime=runtime,
        prepared_ready_path=prepared,
    )


def run_causal_mapping(*, dataset: Any, opt: Any, pipe: Any, args: Any):
    """Run an ordered dataset source through MappingSession and application ticks.

    The dataset is used only at this composition root to create the legacy
    model/backend and source.  The application receives SensorPackets only.
    """
    # --causal_thread_pinning_policy (see eligsir/cli/parser.py and
    # docs/superpowers/specs/2026-09-08-residency-budget-design.md,
    # "Addendum 2"). Must run before any frame source / decode-worker
    # construction below -- the whole point is that the FIRST
    # --causal_decode_workers thread already sees a pinned cv2/BLAS thread
    # pool, not just later ones.
    # Some tests exercise this entrypoint with a minimal/bare args double
    # that predates this flag -- leave those untouched rather than requiring
    # every caller to carry the new attribute (same hasattr guard used just
    # below for --causal_decode_workers_policy).
    if hasattr(args, "causal_thread_pinning_policy"):
        if args.causal_thread_pinning_policy == "pinned":
            from eligsir.runtime.thread_pinning import pin_decode_threads

            pin_decode_threads()
            print("[thread-pinning] policy=pinned cv2_threads=1 blas_threads=1", flush=True)
        else:
            print("[thread-pinning] policy=unbound", flush=True)
    source_hz = float(args.streaming_input_fps_cap)
    from eligsir.utils.execution_semantics import (
        RELEASE_DATASET_FPS,
        RELEASE_ITER_BASED,
        RELEASE_WALLCLOCK_STRICT,
        resolve_execution_semantics,
    )

    release_policy = resolve_execution_semantics(args)["release_policy"]
    paced = release_policy in (RELEASE_DATASET_FPS, RELEASE_WALLCLOCK_STRICT)
    paced = paced and source_hz > 0.0
    if _prepared_ready_b4_enabled(args) and not paced:
        raise ValueError(
            "causal_prepared_ready_b4 requires dataset_fps or wallclock_strict "
            "release with a positive streaming_input_fps_cap"
        )
    from eligsir.telemetry.system_sampler import (
        start_system_telemetry,
        stop_system_telemetry,
        write_system_performance_report,
    )

    # PR #1783 follow-up: default-on per-run system-performance trace (see
    # eligsir/telemetry/system_sampler.py's module docstring). Started
    # here, after the fail-fast validation above (so a malformed/incomplete
    # ``dataset``/``args`` -- e.g. a unit test asserting composition never
    # starts -- still fails on the SAME guard it always did, before this
    # telemetry ever touches ``dataset.model_path``), but still ahead of
    # composition/mapping work so the trace covers composition-build cost
    # too. Stopped unconditionally in this function's own ``finally:`` below
    # (alongside ``application.close()``).
    _system_sampler = start_system_telemetry(
        dataset.model_path,
        interval_s=float(getattr(args, "system_telemetry_interval_s", 1.0)),
        enabled=bool(getattr(args, "system_telemetry", True)),
    )
    composition = resolve_causal_mapping_composition(
        dataset=dataset, opt=opt, pipe=pipe, args=args
    )
    # Store requested source fps for budget relaxation signal in control loop.
    # Only valid when paced (dataset_fps or wallclock_strict); None for iter_based/live.
    # Test compositions stub the runtime with a bare namespace; production runtimes
    # always expose the setter (CanonicalMappingRuntime.set_requested_source_fps).
    _declare_source_fps = getattr(composition.runtime, "set_requested_source_fps", None)
    if _declare_source_fps is not None:
        _declare_source_fps(source_hz if paced else None)
    _viewer, _map_output_lane, _map_max_hz = build_causal_viewer(args, composition.runtime)
    _map_output_last_wall_s = [0.0]
    # Bounded at the producer (unlike train_streaming.py's unbounded
    # _trajectory_est) -- see CanonicalViewerAdapter._MAX_TRAJECTORY_POSES
    # for why this exact cap size was chosen.
    from collections import deque as _deque

    from eligsir.viewers.canonical_adapter import _MAX_TRAJECTORY_POSES

    _trajectory_trail: _deque = _deque(maxlen=_MAX_TRAJECTORY_POSES)
    ingress_telemetry = ingress_telemetry_from_config(
        composition.config, paced=paced
    )
    _declare_input_exhausted = getattr(
        composition.runtime, "set_source_input_exhausted_provider", None
    )
    if _declare_input_exhausted is not None:
        _declare_input_exhausted(
            None
            if ingress_telemetry is None
            else lambda: ingress_telemetry.input_exhausted
        )
    adapter: DatasetPacketAdapter = StreamingFramePacketAdapter(str(dataset.source_path))
    ticks_per_submission = (
        int(args.streaming_steps_per_frame)
        if release_policy == RELEASE_ITER_BASED
        else 1
    )
    tick_policy = CausalReplayTickPolicy(ticks_per_submission=ticks_per_submission)
    decision_observer = decision_trace_observer_from_args(dataset=dataset, args=args)
    # Population compact telemetry is strictly opt-in with the normal
    # decision-trace sink; trace-off runs must not reduce or transfer it.
    composition.runtime._population_selection_trace_enabled = (
        decision_observer is not None
    )
    from eligsir.runtime.causal_evaluation import (
        CausalEvaluationObserver,
        causal_holdout_plan,
        validate_train_evwb_reference_source,
        enable_training_capture_pose_candidates,
    )
    source_frames_value = getattr(composition.source, "frames", None)
    if source_frames_value is None:
        get_all = getattr(composition.source, "get_all", None)
        source_frames_value = get_all() if callable(get_all) else None
    elif callable(source_frames_value):
        source_frames_value = source_frames_value()
    if paced and not isinstance(source_frames_value, Sequence):
        raise TypeError("paced causal replay requires a finite source catalogue")
    source_frames = tuple(source_frames_value or ())
    holdout_plan = causal_holdout_plan(source_frames, args)
    from eligsir.utils.coverage_ledger import CoverageLedger

    causal_coverage_ledger = CoverageLedger.fresh(
        resolved_source_frames=len(source_frames),
        raw_registry_source_frames=None,
        source_denominator_provenance="causal_source_catalog",
        bootstrap_frames_consumed=0,
        source_identity=str(dataset.source_path),
        effective_stride=1,
    )
    validate_train_evwb_reference_source(
        source_frames,
        holdout_plan,
        args,
        source_id=adapter.source_id,
        stream_epoch=adapter.stream_epoch,
    )
    enable_training_capture_pose_candidates(source_frames, holdout_plan)
    partition_control_enabled = (
        float(getattr(args, "streaming_partition_focus_weight", 0.0)) > 0.0
    )
    evaluation_observer = (
        CausalEvaluationObserver(
            runtime=composition.runtime,
            plan=holdout_plan,
            dataset=dataset,
            args=args,
            source_total=(len(source_frames) if source_frames else None),
            source_records=source_frames,
            packet_source_id=adapter.source_id,
            packet_stream_epoch=adapter.stream_epoch,
        )
        if holdout_plan.holdout_ordinals or partition_control_enabled
        else None
    )

    application_options = causal_mapping_application_options(
        composition, args=args, ingress_telemetry=ingress_telemetry
    )
    # Async mapping-holdout evaluation (issue #1314) needs the mutation
    # barrier's wait to happen immediately before the NEXT in-place Gaussian
    # mutation on the default stream. `optimizer_tick` (legacy mapper) and
    # `prepared_tick` (prepared-ingress mapper) are the two mutation choke
    # points every such mutation (optimizer step, MCMC relocation, append)
    # can pass through in this execution path, so wrap whichever one is
    # wired rather than reaching into the mapping runtime's mutation
    # internals.
    if evaluation_observer is not None:
        if application_options.prepared_mode:
            _base_prepared_tick = application_options.prepared_tick

            def _guarded_prepared_tick():
                evaluation_observer.wait_for_pending_snapshot_copy()
                return _base_prepared_tick()

            application_options = replace(
                application_options, prepared_tick=_guarded_prepared_tick
            )
        else:
            _base_optimizer_tick = application_options.optimizer_tick

            def _guarded_optimizer_tick():
                evaluation_observer.wait_for_pending_snapshot_copy()
                return _base_optimizer_tick()

            application_options = replace(
                application_options, optimizer_tick=_guarded_optimizer_tick
            )
    application = ComposedMappingApplication(application_options)
    source_complete_gate = _SourceCompleteEvaluationGate(
        application, evaluation_observer
    )

    def source_complete_observer(tick):
        source_complete_gate.source_complete(tick)

    def tick_observer(tick):
        if decision_observer is not None:
            decision_observer(tick)
        if evaluation_observer is not None and not paced:
            evaluation_observer(tick)
        source_complete_gate.observe_tick(tick)
        if _viewer is not None:
            submit_causal_viewer_tick(
                _viewer, _map_output_lane, _map_max_hz,
                getattr(composition.runtime, "backend", None), tick,
                last_map_wall_s_box=_map_output_last_wall_s,
                runtime=composition.runtime, trajectory_trail=_trajectory_trail,
            )

    def tick_clock_observer(tick, clock):
        if evaluation_observer is not None:
            evaluation_observer.observe_tick(tick, clock)

    def refinement_tick_observer(tick):
        if decision_observer is not None:
            decision_observer(tick)
        if _viewer is not None:
            submit_causal_viewer_tick(
                _viewer, _map_output_lane, _map_max_hz,
                getattr(composition.runtime, "backend", None), tick,
                last_map_wall_s_box=_map_output_last_wall_s,
                runtime=composition.runtime, trajectory_trail=_trajectory_trail,
            )

    def refinement_retry_trace_observer(tick):
        if decision_observer is not None:
            decision_observer(tick)

    def refinement_tick_clock_observer(tick, clock):
        if evaluation_observer is not None:
            evaluation_observer.observe_tick(tick, clock)
    try:
        mapping_started = time.perf_counter()
        source_consumed_observer = getattr(
            composition.runtime, "record_source_consumed", None
        )
        bypass_packet_capture = getattr(
            composition.runtime, "cache_causal_bypass_packet", None
        )
        if paced:
            summary = PacedCausalReplayDriver(
                application,
                adapter,
                source_hz=source_hz,
                source_buffer_frames=int(
                    getattr(args, "causal_source_buffer_frames", 256)
                ),
                decode_workers=int(getattr(args, "causal_decode_workers", 2)),
                ingestion_config=IngestionValueConfig.from_args(args),
                max_new_sources_per_tick=max(
                    1,
                    int(composition.config.hard_limits["optimization_batch_size"]) - 1,
                ),
                max_source_packets_per_tick=max(
                    1,
                    16
                    * (
                        int(
                            composition.config.hard_limits[
                                "optimization_batch_size"
                            ]
                        )
                        - 1
                    ),
                ),
                ingress_telemetry=ingress_telemetry,
                tick_observer=(
                    tick_observer
                    if decision_observer is not None or evaluation_observer is not None
                    or _viewer is not None
                    else None
                ),
                tick_clock_observer=(
                    tick_clock_observer
                    if evaluation_observer is not None
                    else None
                ),
                packet_role=(
                    holdout_plan.role_for_packet
                    if holdout_plan.holdout_ordinals or holdout_plan.guard_ordinals
                    else None
                ),
                bypass_packet_capture=(
                    bypass_packet_capture if callable(bypass_packet_capture) else None
                ),
                source_consumed_observer=source_consumed_observer,
                source_complete_observer=source_complete_observer,
                coverage_ledger=causal_coverage_ledger,
                bootstrap_frames=int(getattr(composition.runtime, "bootstrap_frames", 0)),
                progress_interval_s=float(
                    getattr(args, "causal_progress_interval_s", 10.0)
                ),
                progress_sink=lambda message: print(message, flush=True),
            ).run(source_frames)
        else:
            summary = CausalReplayDriver(
                application,
                adapter,
                tick_policy=tick_policy,
                tick_observer=(
                    tick_observer
                    if decision_observer is not None or evaluation_observer is not None
                    or _viewer is not None
                    else None
                ),
                packet_role=(
                    holdout_plan.role_for_packet
                    if holdout_plan.holdout_ordinals or holdout_plan.guard_ordinals
                    else None
                ),
                bypass_packet_capture=(
                    bypass_packet_capture if callable(bypass_packet_capture) else None
                ),
                source_consumed_observer=source_consumed_observer,
                source_complete_observer=source_complete_observer,
                coverage_ledger=causal_coverage_ledger,
                bootstrap_frames=int(getattr(composition.runtime, "bootstrap_frames", 0)),
            ).run(composition.source)
        # Issue #1653: freeze the mapping-phase wall clock and snapshot the
        # compact runtime supervision counts THE MOMENT mapping semantically
        # drains -- before any diagnostic/report work (summary/trajectory
        # writes, and especially the mapped-camera plot) has a chance to run.
        # `successful_supervision_uses_by_ordinal()` is a bounded host-dict
        # read (#1687); it is cheap enough to snapshot synchronously here.
        # The decision-trace file (when tracing is enabled) is append-only
        # and refinement keeps appending to the very same file, so its exact
        # byte length right now is the boundary between "mapping decisions"
        # and "refinement decisions" -- stat() is O(1) and does not read the
        # file, so recording it costs nothing and does not delay refinement.
        # Issue #1664: the observer now batches rows in memory rather than
        # writing them as they are observed, so its on-disk size can lag its
        # true row count -- flush explicitly right before the stat() below
        # (cheap: at most ``flush_every`` already-serialized rows) so this
        # boundary is still exact.
        mapping_wall_time_s = max(0.0, time.perf_counter() - mapping_started)
        # PR #1783 follow-up: this is the exact semantic-drain boundary the
        # surrounding comment already documents -- the moment the source is
        # exhausted and mapping proper ends, before any diagnostic/report
        # work runs. Record it on the system-performance trace too.
        mark_system_telemetry("source_eof", mapping_wall_time_s=mapping_wall_time_s)
        successful_supervision_uses = getattr(
            composition.runtime, "successful_supervision_uses_by_ordinal", lambda: {}
        )()
        if decision_observer is not None:
            decision_observer.flush()
        mapping_decision_trace_size = (
            decision_observer._path.stat().st_size
            if decision_observer is not None and decision_observer._path.is_file()
            else None
        )
        summary_path = Path(dataset.model_path) / "causal_replay_summary.json"
        curve_metadata = (
            evaluation_observer._mapping_holdout_curve_metadata
            if evaluation_observer is not None
            else None
        )
        # Extract ready_bytes from the prepared_ready_path if available
        ready_bytes_block = {}
        if composition.prepared_ready_path is not None:
            try:
                ready_owner = composition.prepared_ready_path.bridge.ready_owner
                ready_bytes_block = {
                    "ready_bytes": {
                        "peak": int(ready_owner.ready_bytes_peak),
                        "current_at_end": int(ready_owner.ready_bytes_current),
                        "by_tier_kind": {
                            f"{tier}/{kind}": int(nbytes)
                            for (tier, kind), nbytes in ready_owner.ready_bytes_by_tier_kind.items()
                        },
                    }
                }
            except (AttributeError, TypeError):
                pass
        # Extract terminal READY drain counters from the prepared_ready_path if available
        # Issue #1663: capture at source-complete and at drain-end
        ready_drain_block = {}
        if composition.prepared_ready_path is not None:
            prepared_ready_path = composition.prepared_ready_path
            at_source_complete = prepared_ready_path.ready_drain_at_source_complete
            # Note: _adapter is private but necessary for drain counters
            adapter = getattr(prepared_ready_path, "_adapter", None)
            at_drain_end = _build_ready_drain_block_from_owner(
                prepared_ready_path.bridge.ready_owner,
                adapter=adapter,
            )
            ready_drain_block = {
                "ready_drain": {
                    "at_source_complete": at_source_complete,
                    "at_drain_end": at_drain_end,
                }
            }
        # Issue #703: supervision-work report block from causal mapping runtime.
        # Issue #701: execution-summary block with realized execution shape.
        supervision_work_block = {}
        if evaluation_observer is not None and hasattr(evaluation_observer, "runtime"):
            runtime = getattr(evaluation_observer, "runtime", None)
            if runtime is not None and hasattr(runtime, "_supervision_work_clocks"):
                clocks = getattr(runtime, "_supervision_work_clocks", None)
                if clocks is not None:
                    elapsed_s = getattr(evaluation_observer, "_elapsed_s", 0.0)
                    block = clocks.report_block(wall_time_s=float(elapsed_s))
                    if block:  # Non-empty dict
                        supervision_work_block = {"supervision_work": block}
                    # Issue #701: print execution-summary block at shutdown
                    from eligsir.utils.supervision_work import execution_summary_lines
                    requested_batch_size = int(
                        getattr(args, "streaming_optimization_batch_size", 1)
                    )
                    summary_lines = execution_summary_lines(
                        clocks=clocks,
                        requested_batch_size=requested_batch_size,
                        wall_time_s=float(elapsed_s),
                    )
                    if summary_lines:
                        print("\n[execution-summary]", flush=True)
                        for line in summary_lines:
                            print(line, flush=True)

        summary_path.write_text(
            json.dumps(
                causal_replay_summary_payload(
                    summary,
                    source_hz=source_hz if paced else None,
                    curve_metadata=curve_metadata,
                    ready_bytes_block=ready_bytes_block,
                    ready_drain_block=ready_drain_block,
                    supervision_work_block=supervision_work_block,
                    runtime=composition.runtime,
                    experiment_origin=getattr(args, "experiment_origin", None),
                    experiment_context=getattr(args, "experiment_context", None),
                ),
                indent=2,
                sort_keys=True,
                default=str,
            )
            + "\n"
        )
        # Issue #428 Phase 1: persist the causal reference if builder was active.
        # _causal_reference_builder's own construction (a CausalReferenceWorker
        # this block would consume) never landed in this merge -- neither the
        # class nor any assignment to this name exists anywhere in the repo --
        # so every read below was an unconditional NameError the instant this
        # line was reached. Reproduced live on causal_replay/replica
        # (2026-09-09): the first run to ever finish training and reach the
        # report-persist phase crashed here. None keeps this guard's evident
        # intent (a permanently-inactive optional feature) instead of a hard
        # crash; wiring up the actual builder is a separate, unstarted task.
        _causal_reference_builder = None
        _causal_ref = None
        _causal_reference_observation = None
        if _causal_reference_builder is not None:
            try:
                from eligsir.utils.causal_reference import write_causal_reference as _write_causal_ref, build_holdout_membership as _build_holdout_membership
                from eligsir.utils.comparison_report import _get_sensor_depth

                # Issue #428 Phase 6: build HoldoutMembership for holdout_resolvable
                _holdout_membership = None
                if evaluation_observer is not None and evaluation_observer.eval_cameras:
                    try:
                        _holdout_entries = []
                        for _cam in evaluation_observer.eval_cameras:
                            _cam_id = str(getattr(_cam, "image_name", None) or id(_cam))
                            _depth = _get_sensor_depth(_cam, int(_cam.image_height), int(_cam.image_width))
                            if _depth is not None:
                                import torch
                                if hasattr(_depth, "dim"):
                                    _depth = _depth.reshape(_depth.shape[-2], _depth.shape[-1])
                                _depth_np = _depth.cpu().numpy() if torch.is_tensor(_depth) else _depth
                                # Backproject depth to world points using camera info
                                from eligsir.utils.local_quality import camera_c2w, camera_intrinsics
                                from eligsir.utils.rgbd_frames import backproject_depth_pixels_torch
                                import torch
                                _depth_t = torch.as_tensor(_depth_np, dtype=torch.float32, device='cpu')
                                _c2w = camera_c2w(_cam)
                                # Move c2w to CPU if it's a torch tensor (from GPU model)
                                if torch.is_tensor(_c2w):
                                    _c2w = _c2w.cpu()
                                _fx, _fy, _cx, _cy = camera_intrinsics(_cam)
                                _h, _w = _depth_np.shape
                                # Compute points in camera frame then transform to world
                                _ys, _xs = torch.meshgrid(torch.arange(_h, dtype=torch.float32), torch.arange(_w, dtype=torch.float32), indexing='ij')
                                _z = _depth_t
                                _x_cam = (_xs - _cx) / _fx * _z
                                _y_cam = (_ys - _cy) / _fy * _z
                                _points_cam = torch.stack([_x_cam, _y_cam, _z], dim=-1).reshape(-1, 3)
                                _points_world = (_points_cam @ _c2w[:3, :3].T + _c2w[:3, 3]).numpy()
                                _holdout_entries.append((_cam_id, _points_world))
                        if _holdout_entries:
                            _holdout_membership = _build_holdout_membership(
                                _holdout_entries,
                                cell_size_m=float(getattr(args, "streaming_local_quality_cell_size", 0.5)),
                            )
                    except Exception as _he:
                        print(f"[causal-reference] Warning: HoldoutMembership build failed: {_he}", flush=True)
                        _holdout_membership = None

                _causal_ref = _causal_reference_builder.finalize(
                    holdout_membership=_holdout_membership,
                    n_source_frames_realized=int(summary.source_offered),
                    # F7: () on an unpaced run (no producer to have recorded
                    # it) -- finalize() treats that identically to None, so
                    # t_visible_wall_clock_s stays unset rather than
                    # fabricated.
                    offer_wall_clock_s=(
                        getattr(summary, "offer_wall_clock_s", None) or None
                    ),
                    depth_stride=int(
                        getattr(
                            args,
                            "streaming_local_quality_cells_depth_stride",
                            _DEFAULT_CAUSAL_REFERENCE_DEPTH_STRIDE,
                        )
                    ),
                    worker_mode=str(getattr(args, "streaming_local_quality_cells_worker", "process")),
                    queue_capacity=int(
                        getattr(
                            args,
                            "streaming_local_quality_cells_queue_capacity",
                            _DEFAULT_REF_QUEUE_CAPACITY,
                        )
                    ),
                )
                _causal_ref_path = _write_causal_ref(_causal_ref, dataset.model_path)
                print(f"[causal-reference] written: {_causal_ref_path}", flush=True)
            except Exception as _e:
                print(
                    f"[causal-reference] Warning: failed to finalize/write causal reference: {_e}",
                    flush=True,
                )
            finally:
                # The worker (mode="process" spawns a real OS subprocess)
                # must be reaped even when finalize()/write above raised --
                # otherwise a failed finalize leaks a child process forever.
                # Idempotent and safe even after a successful finalize()
                # already joined everything (CausalReferenceWorker.close's
                # own contract).
                try:
                    _causal_reference_builder.close()
                except Exception as _ce:
                    print(
                        f"[causal-reference] Warning: worker close failed: {_ce}",
                        flush=True,
                    )

            # Acceptance evidence (mirrors _resolve_async_throughput_evidence's
            # shape in causal_evaluation.py): built from the worker's own
            # final stats(), which stay queryable after close() (returns the
            # last-known numbers rather than raising -- see CausalReference
            # Worker.stats's docstring). A drop is folded into `status`
            # here, not just `frames_dropped`, because ANY drop changes
            # eligibility_hash (a dropped frame is never appended to
            # source_frame_ids -- see causal_reference_worker.py's module
            # docstring): the artifact's stated "bit-identical mask" purpose
            # silently stops holding, which "ok" must never paper over.
            try:
                _ref_stats = _causal_reference_builder.stats()
                _ref_dropped = int(_ref_stats.dropped_total)
                _ref_stall_ms_max = float(_ref_stats.producer_stall_ms_max)
                _ref_stall_exceeded = (
                    _ref_stall_ms_max > _CAUSAL_REF_PRODUCER_STALL_THRESHOLD_MS
                )
                _ref_status_reasons = []
                if _ref_dropped > 0:
                    _ref_status_reasons.append("frames_dropped")
                if _ref_stall_exceeded:
                    _ref_status_reasons.append("stall_exceeded")
                _causal_reference_observation = {
                    "mode": _ref_stats.mode,
                    "frames_submitted": int(_ref_stats.submitted),
                    "frames_processed": int(_ref_stats.processed),
                    "frames_dropped": _ref_dropped,
                    "producer_stall_ms_max": _ref_stall_ms_max,
                    "producer_stall_threshold_ms": _CAUSAL_REF_PRODUCER_STALL_THRESHOLD_MS,
                    "status": "+".join(_ref_status_reasons) if _ref_status_reasons else "ok",
                }
            except Exception as _se:
                print(
                    f"[causal-reference] Warning: failed to assemble worker "
                    f"acceptance stats: {_se}",
                    flush=True,
                )

        # Payload budget block (resolution- and device-aware defaults)
        _payload_budget_block = None
        if hasattr(args, "_resolved_payload_budget_policy"):
            try:
                _payload_budget_block = {
                    "policy": str(getattr(args, "_resolved_payload_budget_policy", "unknown")),
                    "native_resolution": (
                        [
                            int(getattr(args, "_resolved_native_width", 0)),
                            int(getattr(args, "_resolved_native_height", 0)),
                        ]
                        if (
                            getattr(args, "_resolved_native_width", None)
                            and getattr(args, "_resolved_native_height", None)
                        )
                        else None
                    ),
                    "native_resolution_method": str(
                        getattr(args, "_resolved_native_resolution_method", "unknown")
                    ),
                    "admitted_payload_max_frames": int(
                        getattr(args, "_resolved_payload_budget_frames", 512)
                    ),
                    "hot_tier_max_bytes": int(
                        getattr(args, "_resolved_payload_budget_hot_bytes", 0)
                    ),
                }
            except Exception as _pbe:
                print(
                    f"[payload-budget] Warning: failed to assemble budget metadata: {_pbe}",
                    flush=True,
                )
        if _payload_budget_block is not None:
            existing = json.loads(summary_path.read_text())
            existing.update({"payload_budget": _payload_budget_block})
            summary_path.write_text(
                json.dumps(existing, indent=2, sort_keys=True, default=str) + "\n"
            )

        if _causal_reference_observation is not None:
            summary = replace(summary, causal_reference_observation=_causal_reference_observation)
            # `summary_path` was already written once above (line ~2296,
            # `causal_replay_summary_payload`'s `**asdict(summary)`), before
            # this worker-acceptance block was computed -- and
            # `run_causal_mapping`'s return value is discarded by train.py
            # for this trainer ("Causal mapping returns before the legacy
            # offline report tail below"), so the dataclass field alone
            # never reaches disk. Same pattern as post_mapping_pose_align_
            # block/final_prune_block below: merge into the already-durable
            # JSON rather than re-deriving/re-writing the whole payload, so
            # a run with the gate off (builder never constructed, this
            # whole branch skipped) leaves the file byte-identical to
            # before this field existed.
            existing = json.loads(summary_path.read_text())
            existing.update({"causal_reference_observation": _causal_reference_observation})
            summary_path.write_text(
                json.dumps(existing, indent=2, sort_keys=True, default=str) + "\n"
            )

        # F3: shared per-call metrics assembly. Called once right after the
        # mapping driver returns (tag="mapping_endpoint", below) and again
        # after `run_causal_post_mapping_refinement` completes
        # (tag="final", near the end of this function) so causal_metrics.json
        # covers the refinement tail rather than freezing at the mapping
        # endpoint. This is safe to call twice with a GROWING tick list
        # because `evaluation_observer._cvq_ticks` is a single accumulator
        # that is never reset or replaced across mapping/mapping_endpoint/
        # refinement/post_mapping_report (see `CausalEvaluationObserver.
        # __init__`/`_evaluate_sync` in causal_evaluation.py) -- the
        # "final" call simply sees more ticks, not a different list.
        def _assemble_causal_metrics(tag: str) -> dict | None:
            """Build+write causal_metrics_<tag>.json / causal_metrics.json
            (always the latest tag written) and the causal_cell_quality.jsonl
            sidecar (truncated and rewritten from the CURRENT full tick list
            every call, so a second call never duplicates rows), then return
            the compact report.json["causal_metrics"] summary -- or None if
            the causal reference isn't available or assembly raised.
            """
            if _causal_reference_builder is None or _causal_ref is None:
                return None
            try:
                import json as _json_module
                from eligsir.utils.causal_quality import (
                    build_cvq_block as _build_cvq_block,
                    compute_tick_cvq as _compute_tick_cvq,
                    CausalQualityParams as _CQP,
                )
                from eligsir.utils.causal_assimilation import (
                    build_cuc_block as _build_cuc_block,
                    build_assimilation_block as _build_assimilation_block,
                    compute_tick_cuc as _compute_tick_cuc,
                    CucParams as _CucParams,
                )
                from eligsir.utils.causal_retention import (
                    build_retention_block as _build_retention_block,
                    RetentionParams as _RetentionParams,
                )
                from eligsir.utils.local_health_state import (
                    classify_health as _classify_health,
                    QualitySeries as _QualitySeries,
                )
                import numpy as _np

                # Issue #428 Phase 6 / F3: the FULL tick list accumulated so
                # far. At tag="mapping_endpoint" this is mapping+mapping_
                # endpoint-phase ticks only (refinement hasn't run yet); at
                # tag="final" it also includes every refinement-phase tick
                # appended since -- same accumulator, same observer instance.
                _ticks = list(getattr(evaluation_observer, "_cvq_ticks", []))

                # Phase 2: compute CVQ for each tick, staging its sidecar row.
                _tick_cvqs = []
                _sidecar_rows = []
                _cell_size_m = float(
                    getattr(args, "streaming_local_quality_cell_size", 0.5)
                )
                for _obs_tick in _ticks:
                    try:
                        _tick_cvq = _compute_tick_cvq(_causal_ref, _obs_tick, params=_CQP())
                        _tick_cvqs.append(_tick_cvq)
                        if _tick_cvq.measurable_cell_keys:
                            _sidecar_rows.append(
                                {
                                    "schema_version": 1,
                                    "iteration": int(_tick_cvq.iteration or 0),
                                    "wall_clock_s": _tick_cvq.wall_clock_s,
                                    "source_offered": _tick_cvq.source_offered,
                                    "source_progress": _tick_cvq.source_progress,
                                    # F3: the tick's own real phase, not a
                                    # hardcoded "mapping" -- falls back to
                                    # "mapping" only for a tick that predates
                                    # phase tagging (phase is None).
                                    "phase": str(_tick_cvq.phase or "mapping"),
                                    "cell_size_m": _cell_size_m,
                                    "n_eligible": int(_tick_cvq.n_eligible),
                                    "cell_key": [
                                        int(k) for k in _tick_cvq.measurable_cell_keys
                                    ],
                                    "q_db": [
                                        float(v) for v in _tick_cvq.measurable_cell_q_db
                                    ],
                                    "n_samples": [
                                        float(v)
                                        for v in _tick_cvq.measurable_cell_n_samples
                                    ],
                                }
                            )
                    except Exception as _cvq_e:
                        print(f"[causal-metrics] warning: CVQ computation failed for tick: {_cvq_e}", flush=True)

                # F4/F3: (re)write the sidecar from scratch every assembly.
                # Truncating (mode "w"), not appending, is what makes calling
                # this twice safe -- the "final" call's tick list is a
                # superset of "mapping_endpoint"'s, so rewriting the whole
                # file in one shot can never duplicate a row.
                _cell_quality_sidecar_path = Path(dataset.model_path) / "causal_cell_quality.jsonl"
                try:
                    with open(_cell_quality_sidecar_path, "w") as _f:
                        for _row in _sidecar_rows:
                            _json_module.dump(_row, _f)
                            _f.write("\n")
                except Exception as _sidecar_e:
                    print(
                        f"[causal-metrics] warning: cell quality sidecar write failed: {_sidecar_e}",
                        flush=True,
                    )

                # Phase 3: compute CUC for each CVQ (stateful accumulation)
                _cuc_ticks = []
                _cuc_state = {}
                for _tick_cvq in _tick_cvqs:
                    try:
                        _tick_cuc, _cuc_state = _compute_tick_cuc(
                            _causal_ref, _tick_cvq, params=_CucParams(), state=_cuc_state
                        )
                        _cuc_ticks.append(_tick_cuc)
                    except Exception as _cuc_e:
                        print(f"[causal-metrics] warning: CUC computation failed: {_cuc_e}", flush=True)
                _cuc_state_final = _cuc_state

                # Phase 4: retention (accumulate per-cell quality series and
                # classify health) -- over ALL ticks this call was given, so
                # the "final" assembly's retention block spans mapping AND
                # refinement (F3).
                _retention_params = _RetentionParams()
                _cell_quality_series_dict = {}  # Dict[cell_key, Dict[str, list]]

                # Accumulate per-cell quality data across all ticks
                for _tick_idx, _tick_cvq in enumerate(_tick_cvqs):
                    for _cell_key, _cell_q_db in zip(
                        _tick_cvq.measurable_cell_keys,
                        _tick_cvq.measurable_cell_q_db,
                    ):
                        if _cell_key not in _cell_quality_series_dict:
                            _cell_quality_series_dict[_cell_key] = {
                                "iterations": [],
                                "psnr_db": [],
                                "established": [],
                                "measured": [],
                                "active": [],
                                "confounded": [],
                            }
                        _cell_quality_series_dict[_cell_key]["iterations"].append(_tick_idx)
                        _cell_quality_series_dict[_cell_key]["psnr_db"].append(_cell_q_db)
                        _cell_quality_series_dict[_cell_key]["established"].append(True)
                        _cell_quality_series_dict[_cell_key]["measured"].append(True)
                        _cell_quality_series_dict[_cell_key]["active"].append(True)
                        _cell_quality_series_dict[_cell_key]["confounded"].append(False)

                # Build QualitySeries and HealthResult for each cell with sufficient data
                _health_results = []
                _retention_cells_insufficient = 0
                for _cell_key, _series_dict in _cell_quality_series_dict.items():
                    try:
                        n_measured = len(_series_dict["psnr_db"])
                        # Only include cells with at least min_auc_checkpoints measured ticks
                        if n_measured < _retention_params.min_auc_checkpoints:
                            _retention_cells_insufficient += 1
                            continue

                        _series = _QualitySeries(
                            entity_id=f"cell_{_cell_key}",
                            label=f"cell_{_cell_key}",
                            kind="cell",
                            iterations=_np.asarray(_series_dict["iterations"], dtype=_np.int64),
                            psnr_db=_np.asarray(_series_dict["psnr_db"], dtype=_np.float64),
                            established=_np.asarray(_series_dict["established"], dtype=bool),
                            measured=_np.asarray(_series_dict["measured"], dtype=bool),
                            active=_np.asarray(_series_dict["active"], dtype=bool),
                            confounded=_np.asarray(_series_dict["confounded"], dtype=bool),
                        )
                        _health = _classify_health(_series)
                        _health_results.append(_health)
                    except Exception as _health_e:
                        print(
                            f"[causal-metrics] warning: health classification failed for cell {_cell_key}: {_health_e}",
                            flush=True,
                        )

                # Assemble blocks with real data
                _cvq_block = _build_cvq_block(_causal_ref, ticks=_tick_cvqs)
                _cuc_block = _build_cuc_block(_causal_ref, cuc_ticks=_cuc_ticks)
                # F7: tick_cvqs unlocks quality_at_delay / early_assimilation_auc
                # (spec §8.2-§8.3), which need the full per-tick series, not
                # just each cell's single reached-threshold delay.
                _assimilation_block = _build_assimilation_block(
                    _causal_ref, cuc_state_final=_cuc_state_final, tick_cvqs=_tick_cvqs
                )
                _retention_block = _build_retention_block(health_results=_health_results)

                # Convert retention MetricValues to dicts (unique among the four)
                def _metric_to_dict(mv):
                    if hasattr(mv, 'as_dict'):
                        return mv.as_dict()
                    return mv

                _retention_block_serializable = {
                    k: (_metric_to_dict(v) if hasattr(v, 'as_dict') else v)
                    for k, v in _retention_block.items()
                }

                # Assemble full causal_metrics.json payload
                _missing_inputs = []
                if not _ticks:
                    _missing_inputs.append("no_eligible_cells_observed")
                if not _tick_cvqs:
                    _missing_inputs.append("cvq_computation_failed")
                if not _cuc_ticks:
                    _missing_inputs.append("cuc_computation_failed")
                if not _health_results:
                    _missing_inputs.append("retention_health_results_insufficient_support")
                if _retention_cells_insufficient > 0:
                    _missing_inputs.append(
                        f"retention_cells_dropped_insufficient_checkpoints:{_retention_cells_insufficient}"
                    )

                _causal_metrics_payload = {
                    "spec_id": _causal_ref.spec_id,
                    "spec_version": _causal_ref.spec_version,
                    "status": "ok",
                    "reason": None,
                    "assembly_tag": tag,
                    "missing_inputs": _missing_inputs,
                    "cvq": _cvq_block,
                    "cuc": _cuc_block,
                    "assimilation": _assimilation_block,
                    "retention": _retention_block_serializable,
                    "plots": [],
                }

                # Write causal_metrics_<tag>.json, then (re)write the
                # canonical causal_metrics.json to match it -- so
                # causal_metrics.json always reflects whichever call ran
                # most recently (final, if refinement ran; mapping_endpoint
                # otherwise), per F3.
                _tagged_path = str(
                    Path(dataset.model_path) / f"causal_metrics_{tag}.json"
                )
                with open(_tagged_path, "w") as _f:
                    _json_module.dump(_causal_metrics_payload, _f, indent=2, default=str)
                _metrics_path = str(Path(dataset.model_path) / "causal_metrics.json")
                with open(_metrics_path, "w") as _f:
                    _json_module.dump(_causal_metrics_payload, _f, indent=2, default=str)
                print(f"[causal-metrics] written: {_tagged_path} (and {_metrics_path})", flush=True)

                # Build compact summary for report.json's causal_metrics key
                _blocking_reasons = _missing_inputs if _missing_inputs else []
                _compact_summary = {
                    "spec_version": _causal_ref.spec_version,
                    "status": "ok",
                    "quotable": len(_blocking_reasons) == 0,
                    "blocking_reasons": _blocking_reasons,
                    "eligibility_hash": _causal_ref.eligibility_hash,
                    "artifact_path": "causal_metrics.json",
                    "headline": {
                        "cvq_auc_region_balanced_progress_mean_over_span": (
                            _cvq_block.get("auc", {}).get(
                                "region_balanced_progress_mean_over_span", {}
                            )
                        ),
                        "cvq_final_region_balanced": (
                            (_cvq_block.get("final") or {}).get("region_balanced", {})
                        ),
                        "cuc_final": (_cuc_block.get("final") or {}),
                        "ttq_median_wall_clock_s": (
                            (_assimilation_block.get("ttq", {}).get("wall_clock_s") or {}).get(
                                "median", {}
                            )
                        ),
                        "ttq_fraction_censored": (
                            (_assimilation_block.get("ttq", {}).get("wall_clock_s") or {}).get(
                                "fraction_censored", {}
                            )
                        ),
                        "retention_debt_mean": (
                            _retention_block_serializable.get("retention_debt_mean", {})
                        ),
                        "max_established_drop_p95": (
                            _retention_block_serializable.get("max_established_drop_p95", {})
                        ),
                    }
                }
            except Exception as _e:
                print(
                    f"[causal-metrics] Warning: failed to assemble causal metrics (tag={tag}): {_e}",
                    flush=True,
                )
                import traceback
                traceback.print_exc()
                return None

            # Issue #428 Phase 5: plot causal metrics over time (figures A-E via plot_causal_metrics_all)
            try:
                from eligsir.utils.eval_tick import plot_causal_metrics_all
                import os
                _plot_paths = plot_causal_metrics_all(str(dataset.model_path))
                if _plot_paths:
                    _plot_basenames = [os.path.basename(p) for p in _plot_paths]
                    print(f"[causal-metrics] plots written: {', '.join(_plot_basenames)}", flush=True)
                    _compact_summary["plots"] = _plot_basenames
                    # Update the payload dict and re-write the JSON files to include plots
                    _causal_metrics_payload["plots"] = _plot_basenames
                    with open(_tagged_path, "w") as _f:
                        _json_module.dump(_causal_metrics_payload, _f, indent=2, default=str)
                    with open(_metrics_path, "w") as _f:
                        _json_module.dump(_causal_metrics_payload, _f, indent=2, default=str)
                    print(f"[causal-metrics] plots recorded in causal_metrics JSON", flush=True)
                else:
                    print(f"[causal-metrics] no plots generated", flush=True)
            except Exception as _e:
                print(
                    f"[causal-metrics] Warning: failed to plot causal metrics: {_e}",
                    flush=True,
                )
            return _compact_summary

        # Issue #428 Phase 5: wire CVQ/CUC/TTQ/RetentionDebt into
        # causal_metrics.json and report.json's causal_metrics key, AS OF
        # THE MAPPING ENDPOINT (F3) -- called again with tag="final" after
        # refinement, near the end of this function.
        _causal_metrics_compact_summary = _assemble_causal_metrics(tag="mapping_endpoint")

        trajectory_path = write_causal_trajectory(dataset.model_path, source_frames)
        # Issue #1653: the mapped-camera plot itself (full trajectory/trace
        # parsing plus matplotlib rendering) is deferred to the post-run
        # diagnostics section below, AFTER refinement dispatches -- it must
        # not sit in the mapping-to-refinement handover. `successful_
        # supervision_uses` and `mapping_decision_trace_size` were already
        # snapshotted above, at the exact semantic-drain moment, so the plot
        # rendered later still reflects mapping's own selection, not
        # whatever refinement has appended to the trace file by then.

        def mapping_endpoint_reporter(_endpoint: dict[str, Any]):
            if evaluation_observer is None or not holdout_plan.holdout_ordinals:
                return None
            # Issue #1601 slice 1: explicit final flush of assimilation support summary
            # (source-complete report path, zero device readback).
            from eligsir.utils.assimilation_pressure import flush_assimilation_support
            final_support_summary = flush_assimilation_support()
            if final_support_summary is not None:
                summary["assimilation_support_debt_final"] = float(final_support_summary.debt)
                summary["assimilation_support_pool_final"] = int(final_support_summary.support_pool_n)
            evaluation_observer.finalize(
                summary,
                subdir="mapping_endpoint",
                experiment_validity_is_final=False,
                mapping_phase_source="causal_quiescent_mapping_endpoint",
            )
            comparison = Path(dataset.model_path) / "comparison" / "mapping_endpoint"
            return {
                "report": str(comparison / "report.json"),
                "test_contact_sheet": str(comparison / "test_contact_sheet.png"),
                "normals_contact_sheet": str(comparison / "normals_contact_sheet.png"),
                "trajectory_psnr_by_distance": str(
                    comparison / "trajectory_psnr_by_distance.svg"
                ),
            }

        control_probe_chunk_steps = int(
            getattr(args, "streaming_refinement_control_probe_chunk_steps", 0)
        )
        control_probe_max_rejections = int(
            getattr(args, "streaming_refinement_control_probe_max_rejections", 3)
        )
        branch_runtime = None
        on_control_chunk_accepted = None
        if control_probe_chunk_steps > 0:
            if evaluation_observer is None:
                raise RuntimeError("control-probe refinement requires causal evaluation")
            if getattr(evaluation_observer, "_partition_control_runtime", None) is None:
                raise RuntimeError("control-probe refinement requires partition control")
            pending_measurement = [None]

            def evaluate_control_probe() -> Mapping[int, float]:
                tick = SimpleNamespace(
                    iteration=int(getattr(composition.runtime, "iteration", 0)),
                    optimizer_steps=0,
                    supervision_views=0,
                )
                measurement = evaluation_observer._measure_partition_control(tick)
                if measurement is None:
                    raise RuntimeError(
                        "control-probe refinement could not evaluate a partition vector"
                    )
                means = getattr(measurement.decision, "partition_means", None)
                if not isinstance(means, Mapping):
                    raise RuntimeError(
                        "control-probe refinement produced no partition means"
                    )
                pending_measurement[0] = measurement
                return {int(key): float(value) for key, value in means.items()}

            branch_factory = getattr(composition.runtime, "refinement_branch_runtime", None)
            if not callable(branch_factory):
                raise RuntimeError(
                    "control-probe refinement requires canonical branch support"
                )
            branch_runtime = branch_factory(evaluate_fn=evaluate_control_probe)

            def publish_accepted_control_probe() -> None:
                measurement = pending_measurement[0]
                if measurement is None:
                    raise RuntimeError("accepted control-probe measurement is unavailable")
                tick = SimpleNamespace(
                    iteration=int(getattr(composition.runtime, "iteration", 0)),
                    optimizer_steps=0,
                    supervision_views=0,
                )
                evaluation_observer._publish_partition_control(tick, measurement)

            on_control_chunk_accepted = publish_accepted_control_probe

        run_causal_post_mapping_refinement(
            application,
            composition.runtime,
            source_frame_count=len(source_frames),
            requested_steps=int(
                getattr(args, "streaming_post_mapping_refinement_steps", 0)
            ),
            model_path=dataset.model_path,
            mapping_wall_time_s=mapping_wall_time_s,
            tick_observer=(
                refinement_tick_observer
                if decision_observer is not None or _viewer is not None
                else None
            ),
            retry_tick_observer=(
                refinement_retry_trace_observer
                if decision_observer is not None
                else None
            ),
            tick_clock_observer=(
                refinement_tick_clock_observer
                if evaluation_observer is not None
                else None
            ),
            initial_clock=(
                CausalReplayClock(
                    optimizer_steps=int(summary.optimizer_steps),
                    supervision_views=int(summary.supervision_views),
                    mapping_active_s=float(summary.mapping_active_s),
                    elapsed_s=float(summary.elapsed_s),
                    source_frame_index=getattr(
                        evaluation_observer, "_source_frame_index", None
                    ),
                )
                if evaluation_observer is not None
                else None
            ),
            mapping_endpoint_reporter=(
                mapping_endpoint_reporter
                if evaluation_observer is not None and holdout_plan.holdout_ordinals
                else None
            ),
            checkpoint_interval_steps=int(
                getattr(args, "streaming_refinement_checkpoint_interval", 0)
            ),
            requested_seconds=float(
                getattr(args, "streaming_post_mapping_refinement_seconds", 0.0)
            ),
            branch_runtime=branch_runtime,
            control_probe_chunk_steps=control_probe_chunk_steps,
            control_probe_max_rejections=control_probe_max_rejections,
            on_control_chunk_accepted=on_control_chunk_accepted,
            allocator_phase_tracker=(
                None
                if evaluation_observer is None
                else getattr(evaluation_observer, "_gpu_memory_tracker", None)
            ),
        )
        if decision_observer is not None:
            decision_observer.finalize()
        # Issue #1653: both mapped-camera plots render here, in the post-run
        # diagnostics section -- strictly after refinement has already run
        # to completion, never between mapping's drain and refinement's
        # dispatch. The mapping-phase plot uses the counts and decision-trace
        # byte boundary snapshotted at the semantic-drain point above, so it
        # reports exactly what mapping (not refinement) selected, even though
        # it is only rendered now.
        if decision_observer is not None or successful_supervision_uses:
            write_mapped_camera_trajectory_plot(
                dataset.model_path,
                trajectory_path,
                None if decision_observer is None else decision_observer._path,
                successful_supervision_uses=successful_supervision_uses,
                decision_trace_byte_limit=mapping_decision_trace_size,
            )
        refined_successful_supervision_uses = getattr(
            composition.runtime, "successful_supervision_uses_by_ordinal", lambda: {}
        )()
        if decision_observer is not None or refined_successful_supervision_uses:
            write_mapped_camera_trajectory_plot(
                dataset.model_path,
                trajectory_path,
                None if decision_observer is None else decision_observer._path,
                successful_supervision_uses=refined_successful_supervision_uses,
                out_name="mapped_camera_trajectory_refined",
            )
        if evaluation_observer is not None:
            evaluation_observer.finalize(summary)
        return summary
    finally:
        # Issue #279: write the whole-run per-phase timing summary (populated
        # only when SPLAT_PROFILE_PHASES=1, via the timing_scope now threaded
        # through StreamingOptimizationRuntime in production_application.py --
        # no-op / empty dict otherwise) alongside train_streaming.py's own
        # equivalent final dump.
        from eligsir.utils.ingest_phase_profiler import dump_phase_ops as _dump_causal_phase_ops

        _dump_causal_phase_ops(str(dataset.model_path))
        # Issue #1664: exception-safe drain of the decision trace's buffered
        # rows -- the normal ``decision_observer.finalize()`` call above only
        # runs on the success path, so an exception raised during mapping or
        # refinement would otherwise leave not-yet-flushed rows unwritten.
        # ``flush()`` is idempotent (a no-op once already flushed) and must
        # never itself mask the original exception propagating through this
        # ``finally``.
        if decision_observer is not None:
            try:
                decision_observer.flush()
            except Exception:  # noqa: BLE001 - shutdown must not mask the real error
                pass
        application.close()
        if _viewer is not None:
            try:
                _viewer.close()
            except Exception:  # noqa: BLE001 - a viewer error must never affect mapping
                pass
        # Stop the system-performance sampler (a no-op when telemetry was
        # disabled -- ``stop_system_telemetry(None)`` short-circuits) and
        # render its plot/summary last, once the run -- success or crash --
        # is otherwise fully torn down.
        stop_system_telemetry(_system_sampler)
        if _system_sampler is not None:
            write_system_performance_report(dataset.model_path)


__all__ = [
    "CausalMappingComposition",
    "build_causal_viewer",
    "decision_trace_observer_from_args",
    "ingress_telemetry_from_config",
    "resolve_causal_mapping_composition",
    "run_causal_mapping",
    "submit_causal_viewer_tick",
]
