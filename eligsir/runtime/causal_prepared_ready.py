"""Causal prepared ingress with exact READY-B4 canonical execution."""

from __future__ import annotations

import os
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import Any, Callable

from eligsir.contracts.control import SemanticAction
from eligsir.runtime.prepared_runtime_bridge import PreparedRuntimeBridge, PreparedRuntimeBridgeResult
from eligsir.runtime.prepared_source_prep_worker import PreparedSourcePrepWorker
from eligsir.runtime.ready_b4_streaming_adapter import (
    ReadyB4StreamingOptimizationAdapter,
    ReadyB4StreamingStepInputs,
    ReadyB4StreamingStepResult,
)
from eligsir.runtime.ready_only_tick import ReadyOnlyTickSelection
from eligsir.runtime.ready_source import ReadySourceLane


def _capture_ready_drain_counters(ready_owner: Any, *, adapter: Any | None = None) -> dict[str, Any]:
    """Capture terminal READY drain counters from the owner and adapter.

    Issue #1663: build a snapshot of all counters from ReadySourceCounters
    plus adapter acceptance counters, and return as a dict suitable for JSON serialization.
    """
    counters = getattr(ready_owner, "counters", None)
    if counters is None:
        return {}
    block = {
        "fresh_pending": int(counters.fresh_pending),
        "replay_pending": int(counters.replay_pending),
        "in_flight": int(counters.in_flight),
        "offered": int(counters.offered),
        "selected": int(counters.selected),
        "released": int(counters.released),
        "full_drop_count": int(counters.full_drop_count),
        "full_not_resident_count": int(counters.full_not_resident_count),
        "duplicate_drop_count": int(counters.duplicate_drop_count),
        "stale_generation_drop_count": int(counters.stale_generation_drop_count),
        "not_ready_count": int(counters.not_ready_count),
        "ready_error_count": int(counters.ready_error_count),
        "reserve_underflow_count": int(counters.reserve_underflow_count),
        "fresh_ready_count": int(counters.fresh_ready_count),
        "replay_ready_count": int(counters.replay_ready_count),
        "ready_bytes_current": int(counters.ready_bytes_current),
        "ready_bytes_peak": int(counters.ready_bytes_peak),
        "replay_evicted_bytes": int(counters.replay_evicted_bytes),
    }
    # Issue #1663: add adapter-specific counters if available
    if adapter is not None:
        admitted_count = int(getattr(adapter, "admitted_commit_count", 0))
        first_supervision = int(getattr(adapter, "first_supervision_completions", 0))
        block["admitted_commit_count"] = admitted_count
        block["first_supervision_completions"] = first_supervision
        block["first_supervision_debt"] = admitted_count - first_supervision
        block["replay_only_steps_while_fresh_pending"] = int(
            getattr(adapter, "replay_only_steps_while_fresh_pending", 0)
        )
    return block


@dataclass(frozen=True, slots=True)
class PreparedBootstrapOutcome:
    """Result of a prepared bootstrap attempt."""

    state: str
    ingress: PreparedRuntimeBridgeResult
    selection: ReadyOnlyTickSelection
    iteration: int = 0
    reason: str | None = None
    completed_supervision_sources: tuple[Any, ...] = ()

    @property
    def executed(self) -> bool:
        """Return whether the outcome represents an executed step."""
        return False

    @property
    def optimizer_steps(self) -> int:
        """Return the number of optimizer steps taken during bootstrap."""
        return 0

    @property
    def supervision_views(self) -> int:
        """Return the number of supervision views used during bootstrap."""
        return 0

    @property
    def retryable(self) -> bool:
        """Return whether the bootstrap attempt is retryable."""
        return self.state == "bootstrap_pending"


def _state_value(outcome: Any) -> str:
    state = getattr(outcome, "state", "")
    return str(getattr(state, "value", state))


class _BootstrapCompletion:
    def __init__(self) -> None:
        self._event: Any | bool | None = None

    def record_after_bootstrap(self) -> None:
        import torch

        if not torch.cuda.is_available():
            self._event = False
            return
        event = torch.cuda.Event(enable_timing=False)
        event.record(torch.cuda.current_stream())
        self._event = event

    def query(self) -> bool:
        return self._event is False or (self._event is not None and bool(self._event.query()))

    def handoff_to_compute(self, _transfer_owner: Any) -> None:
        if self._event is False:
            return
        if self._event is None:
            raise RuntimeError("prepared bootstrap completion was not recorded")
        import torch

        torch.cuda.current_stream().wait_event(self._event)


class CausalPreparedReadyB4Path:
    """Causal prepared ingress with READY-B4 canonical execution semantics."""

    def __init__(
        self,
        *,
        runtime: Any,
        bridge: PreparedRuntimeBridge,
        adapter_factory: Callable[[], ReadyB4StreamingOptimizationAdapter],
        step_inputs_factory: Callable[[], ReadyB4StreamingStepInputs],
        bootstrap_completion_factory: Callable[[], Any] = _BootstrapCompletion,
        source_complete_phase_hook: Callable[[], Any] | None = None,
        prep_worker: PreparedSourcePrepWorker | None = None,
        promotion_owner: Any | None = None,
    ) -> None:
        if not isinstance(bridge, PreparedRuntimeBridge):
            raise TypeError("bridge must be PreparedRuntimeBridge")
        self._runtime = runtime
        self.bridge = bridge
        self.handoff = bridge.handoff
        self._adapter_factory = adapter_factory
        self._step_inputs_factory = step_inputs_factory
        self._bootstrap_completion_factory = bootstrap_completion_factory
        self._source_complete_phase_hook = source_complete_phase_hook
        self._source_complete_phase_marked = False
        self._source_complete_requested = False
        self._prep_worker = prep_worker
        self._promotion_owner = promotion_owner
        self._adapter: ReadyB4StreamingOptimizationAdapter | None = None
        self._bootstrap_fresh_sources: dict[Any, None] = {}
        # Snapshot of READY drain counters at source-complete phase
        self._ready_drain_at_source_complete: dict[str, Any] | None = None

    def _take_ingress_result(self) -> PreparedRuntimeBridgeResult:
        if self._prep_worker is None:
            return PreparedRuntimeBridgeResult(0, 0, 0, 0)
        self._prep_worker.check_health()
        return self._prep_worker.take_result()

    def close(self) -> None:
        """Close the prepared source preparation worker."""
        if self._prep_worker is not None:
            self._prep_worker.close()

    def wait_for_progress(self, *, timeout_s: float) -> bool:
        """Wait for progress from the preparation worker."""
        return False if self._prep_worker is None else self._prep_worker.wait_for_progress(
            timeout_s=timeout_s
        )

    def _maybe_complete_source_phase(self, outcome: Any) -> None:
        if (
            _state_value(outcome) != "bootstrap_pending"
            and bool(getattr(self._runtime, "initialized", False))
            and self._source_complete_requested
            and not self._source_complete_phase_marked
            and self._source_complete_phase_hook is not None
        ):
            self._source_complete_phase_hook()
            # Capture READY drain counters at source-complete (issue #1663)
            self._ready_drain_at_source_complete = _capture_ready_drain_counters(
                getattr(self.bridge, "ready_owner", None),
                adapter=getattr(self, "_adapter", None),
            )
            self._source_complete_phase_marked = True

    def _bootstrap_tick(self) -> PreparedBootstrapOutcome:
        ingress = self._take_ingress_result()
        requested = min(4, int(getattr(self._runtime, "ready_bootstrap_remaining", lambda: 4)()))
        owner = self.bridge.ready_owner
        ready_count = sum(lease.is_ready() for lease in owner.pending_leases)
        if ready_count < requested:
            return PreparedBootstrapOutcome(
                "bootstrap_pending",
                ingress,
                ReadyOnlyTickSelection((), 0, 0, False, True),
            )
        leases = owner.select(requested)
        lanes = tuple(owner.selected_lane(lease) for lease in leases)
        fresh_sources = tuple(
            lease.source_identity
            for lease, lane in zip(leases, lanes)
            if lane is ReadySourceLane.FRESH
        )
        fresh_count = len(fresh_sources)
        selection = ReadyOnlyTickSelection(
            leases=leases,
            fresh_count=fresh_count,
            replay_count=len(leases) - fresh_count,
            fallback_to_replay=bool(leases) and fresh_count == 0,
            starved=len(leases) < requested,
        )
        if len(leases) < requested:
            if leases:
                self.bridge.requeue_ready_only_sources(selection, ready_event=None)
            return PreparedBootstrapOutcome("bootstrap_pending", ingress, selection)

        completion = self._bootstrap_completion_factory()
        try:
            initialized = self._runtime.bootstrap_ready_only_selection(selection, bootstrap_event=completion)
        finally:
            completion.record_after_bootstrap()

        if initialized and bool(getattr(self._runtime, "initialized", False)):
            completed_sources = tuple(dict.fromkeys((*self._bootstrap_fresh_sources, *fresh_sources)))
            recorder = getattr(self._runtime, "record_successful_supervision_sources", None)
            if callable(recorder):
                recorder(completed_sources)
            for source in completed_sources:
                if not self.bridge.complete_retained_commit_first_supervision(source):
                    raise RuntimeError("bootstrap COMMIT lost its first-supervision obligation")
            self._bootstrap_fresh_sources.clear()
            if (candidate := self._runtime._prepared_ready_candidate_work) is not None:
                self._runtime._prepared_ready_frozen_work = candidate
                self._runtime._advance_prepared_ready_plan(candidate[1])
            state = "bootstrapped"
        else:
            completed_sources = ()
            self._bootstrap_fresh_sources.update(dict.fromkeys(fresh_sources))
            state = "bootstrap_pending"
        return PreparedBootstrapOutcome(
            state,
            ingress,
            selection,
            iteration=int(getattr(self._runtime, "iteration", 0)),
            completed_supervision_sources=completed_sources,
        )

    def tick(self) -> PreparedBootstrapOutcome | ReadyB4StreamingStepResult:
        """Execute one causal prepared ready-B4 iteration."""
        poll_feedback = getattr(self._runtime, "_poll_replay_evidence", None)
        if callable(poll_feedback):
            poll_feedback()
        maintain_future_working_set: Callable[[], None] | None = None
        if self._promotion_owner is not None and self._runtime.initialized:
            freeze = getattr(self._runtime, "_prepared_ready_plan_for_execution", None)
            if callable(freeze):
                freeze()
            frozen_work = getattr(self._runtime, "_prepared_ready_frozen_work", None)
            candidate_work = getattr(self._runtime, "_prepared_ready_candidate_work", None)
            set_decode_protection = getattr(
                self._runtime, "_set_admission_prefetch_protected_sources", None
            )
            if callable(set_decode_protection):
                set_decode_protection(
                    frozenset(
                        source
                        for work in (frozen_work, candidate_work)
                        if work is not None
                        for source in (
                            tuple(getattr(work[1], "supervision_sources", ()))
                            + tuple(getattr(work[1], "replay_sources", ()))
                        )
                    )
                )
            protected = {
                source
                for work in (frozen_work,)
                if work is not None
                for source in (
                    tuple(getattr(work[1], "supervision_sources", ()))
                    + tuple(getattr(work[1], "replay_sources", ()))
                )
            }
            owner = getattr(self._runtime, "_ready_source_owner", None)
            if candidate_work is not None and owner is not None:
                candidate = candidate_work[1]
                fresh = frozenset(getattr(candidate, "supervision_sources", ()))
                replay = frozenset(getattr(candidate, "replay_sources", ()))
                exact = tuple((*fresh, *replay))
                if exact:
                    ready_sources = tuple(
                        source
                        for source, state in owner.exact_readiness(
                            fresh_sources=fresh,
                            replay_sources=replay,
                            exact_sources=exact,
                        )
                        if state == "READY"
                    )
                    protected.update(ready_sources)
            protected_sources = frozenset(protected)

            def maintain_future_working_set() -> None:
                """Schedule cold promotion candidate sources."""
                promotion_snapshot = getattr(
                    self._runtime, "_prepared_cold_promotion_snapshot", None
                )
                if callable(promotion_snapshot):
                    snapshot = promotion_snapshot()
                else:
                    outputs = getattr(self._runtime, "_latest_control_outputs", None)
                    snapshot = None if outputs is None else outputs[0]
                candidates = self._runtime.prepared_cold_promotion_shortlist()
                self._promotion_owner.schedule_shortlist(
                    (
                        candidate
                        for candidate in candidates
                        if (
                            record := self._runtime._observation_db.get_frame(
                                int(candidate.frame_id)
                            )
                        ) is not None
                        and record.modality_eligibility.rgb_supervision
                    ),
                    promotion_fence=None if snapshot is None else (int(snapshot.snapshot_id), int(snapshot.control_version), int(snapshot.topology_version)),
                    protected_sources=protected_sources,
                )

        if not bool(getattr(self._runtime, "initialized", False)):
            outcome = self._bootstrap_tick()
        else:
            if self._adapter is None:
                self._adapter = self._adapter_factory()
            outcome = self._adapter.tick(self._step_inputs_factory())
            if _state_value(outcome) == "stale":
                self._runtime._advance_prepared_ready_plan(outcome.work_plan)
        if maintain_future_working_set is not None:
            maintain_future_working_set()
        self._maybe_complete_source_phase(outcome)
        return outcome

    def mark_source_complete(self) -> PreparedBootstrapOutcome:
        """Mark the source phase as complete."""
        self._source_complete_requested = True
        if bool(getattr(self._runtime, "initialized", False)):
            outcome = PreparedBootstrapOutcome(
                "bootstrapped",
                self._take_ingress_result(),
                ReadyOnlyTickSelection((), 0, 0, False, False),
                iteration=int(getattr(self._runtime, "iteration", 0)),
            )
            self._maybe_complete_source_phase(outcome)
            return outcome
        outcome = self._bootstrap_tick()
        if _state_value(outcome) == "bootstrap_pending":
            return replace(outcome, reason="source_complete_preparation_pending")
        self._maybe_complete_source_phase(outcome)
        return outcome

    @property
    def ready_drain_at_source_complete(self) -> dict[str, Any] | None:
        """Return the READY drain counters snapshot at source-complete, or None if not yet fired."""
        return self._ready_drain_at_source_complete


def _transfer_owner_for_runtime(runtime: Any) -> Any:
    import torch
    from eligsir.runtime.transfers import get_device_transfer_owner

    device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
    policy = getattr(runtime, "_runtime_policy", None)
    owner = get_device_transfer_owner(
        device,
        max_pinned_bytes=int(getattr(policy, "max_transfer_bytes", 2 * 1024 * 1024 * 1024)),
    )
    runtime._transfer_owner = owner
    return owner


def _step_inputs_for_runtime(runtime: Any) -> ReadyB4StreamingStepInputs:
    ready_plan = getattr(runtime, "_prepared_ready_plan_for_execution", None)
    if callable(ready_plan):
        plan = ready_plan()
        if plan is None:
            return ReadyB4StreamingStepInputs(None, None)
        work = getattr(runtime, "_prepared_ready_frozen_work", None)
        if work is None:
            return ReadyB4StreamingStepInputs(None, plan)
        snapshot, _candidate = work
        return ReadyB4StreamingStepInputs(snapshot, plan)
    work = runtime._latest_tick_work or runtime._latest_report_work
    if work is None and runtime.initialized and runtime._last_materialized_work is not None:
        snapshot, prior = runtime._last_materialized_work
        idle = runtime._fresh_idle_snapshot(snapshot)
        work = (idle, runtime._idle_replay_plan(idle, prior))
    if work is None:
        return ReadyB4StreamingStepInputs(None, None)
    snapshot, candidate = work
    return ReadyB4StreamingStepInputs(snapshot, candidate)


@contextmanager
def _stderr_timing_scope(name):
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        print(f"[timing_scope] {name} {elapsed_ms:.3f}ms", file=sys.stderr)


def _measurement_timing_scope():
    if os.environ.get("ELIGSIR_TIME_EVICTION_HOLD"):
        return _stderr_timing_scope
    return None


def build_causal_prepared_ready_path(*, runtime: Any, config: Any, args: Any) -> CausalPreparedReadyB4Path:
    """Build a causal prepared READY-B4 path with validation and setup."""
    from eligsir.mapping.batch_contract import qualified_camera_count_for_batch

    batch_size = int(config.hard_limits["optimization_batch_size"])
    if qualified_camera_count_for_batch(batch_size) != batch_size:
        raise ValueError(
            "causal_prepared_ready_b4 requires a qualified exact batch size; "
            f"got {batch_size}"
        )
    options = config.static_options
    if options["optimization_constellation_policy"] != "fixed":
        raise ValueError(
            "causal_prepared_ready_b4 requires "
            "streaming_optimization_constellation_policy=fixed"
        )
    source_insertion = bool(options.get("prepared_ready_source_insertion", False))
    if options["insert_from_depth"] and not source_insertion:
        raise ValueError(
            "causal_prepared_ready_b4 requires --no-streaming_insert_from_depth "
            "(or opt in with --causal_prepared_ready_source_insertion)"
        )
    if int(options.get("replay_depth_max_per_event", 0)) != 0:
        raise ValueError(
            "causal_prepared_ready_b4 requires --streaming_replay_depth_max_per_event=0"
            + (
                " (replay depth insertion is not supported on the READY path, "
                "even with --causal_prepared_ready_source_insertion)"
                if source_insertion else ""
            )
        )
    depth_insertion = getattr(config, "depth_insertion", None)
    if (
        source_insertion
        and bool(getattr(depth_insertion, "temporal_check_enabled", False))
        and (
            bool(getattr(depth_insertion, "require_temporal_stability", False))
            or bool(
                getattr(depth_insertion, "require_temporal_for_sensor_closer", False)
            )
        )
    ):
        raise ValueError(
            "causal_prepared_ready_b4 source insertion does not support required "
            "temporal depth masks"
        )
    if bool(options.get("admission_prematerialize", False)):
        # Measured 2026-09-04 (rtx4090c, fr3): with the legacy-path admission
        # pre-materialize hooks active, the READY-B4 loop ran 66 steps and then
        # starved (`prepared_authorized_sources_not_ready`, optimizer_steps=0 for
        # 30 s with every source submitted). The prepared path builds every tier
        # off-thread before READY and measured zero materialize misses without
        # the flag, so it has nothing to gain from it; refuse at construction
        # rather than starve at runtime.
        raise ValueError(
            "causal_prepared_ready_b4 requires --no-streaming_admission_prematerialize "
            "(the admission pre-materialize hooks are legacy-path only and starve "
            "READY selection)"
        )
    from eligsir.runtime.prepared_ingress import PreparedSourceIngressHandoff
    from eligsir.runtime.prepared_source_producer import PreparedSourceProducer
    from eligsir.runtime.ready_source import ReadySourceOwner
    from eligsir.runtime.prepared_cold_promotion import FrameObservationPreparedColdResolver, PreparedColdPromotionOwner

    transfer_owner = _transfer_owner_for_runtime(runtime)
    bootstrap_capacity = max(batch_size, int(getattr(runtime, "bootstrap_frames", batch_size)))
    owner = ReadySourceOwner(
        max_fresh=bootstrap_capacity,
        max_replay=max(16, 4 * batch_size, bootstrap_capacity),
        replay_reserve=4,
    )
    runtime.enable_ready_source_owner(owner, warmup_complete=False)
    def commit_item(item: Any) -> Any:
        """Commit a prepared source item to the ready lane history."""
        disposition = item.disposition
        if disposition is None:
            return
        if disposition.semantic_action is not SemanticAction.COMMIT or item.packet.depth is None:
            raise ValueError("prepared handoff requires RGB-D COMMIT work")
        if runtime.retain_prepared_source(item, lane=ReadySourceLane.FRESH) is not True:
            raise RuntimeError("prepared COMMIT did not publish durable history")
        runtime.record_retained_prepared_commit(item)
        return item

    handoff = PreparedSourceIngressHandoff(
        capacity=max(16, 4 * batch_size),
        commit_sink=commit_item,
    )
    # Resolve the prepared tier policy from the configured resolution-tier policy.
    # This determines exactly which tiers are materialized before READY; no lazy
    # tier creation occurs after READY (issue #1575 C.1).
    from eligsir.contracts.control import ResolutionTier, PreparedTierPolicy

    resolution_tier_policy = str(options["mapping_resolution_tier"])
    minimum_resolution_tier = ResolutionTier(
        str(options.get("mapping_resolution_min_tier", "eighth"))
    )
    supported_resolution_tiers = (
        tuple(
            tier
            for tier in ResolutionTier
            if tier.downsample_factor <= minimum_resolution_tier.downsample_factor
        )
        if resolution_tier_policy == "adaptive"
        else (ResolutionTier(resolution_tier_policy),)
    )
    # The native tier is required regardless of the work tier: prepared bootstrap
    # (canonical_scene.initialize_from_ready_bindings), the base source payload and
    # native geometry evidence (#1766) all read it. Work tiers come from the
    # resolution policy; native is always materialised (as before #1822).
    prepared_tier_policy = PreparedTierPolicy.from_supported_tiers(
        frozenset((ResolutionTier.NATIVE, *supported_resolution_tiers))
    )
    # Store the policy on the runtime so bootstrap can access it without kwarg passing
    runtime._prepared_tier_policy = prepared_tier_policy
    producer = PreparedSourceProducer(
        owner,
        transfer_owner=transfer_owner,
        producer_owner="causal-prepared-source",
        resampling_mode=options.get("resolution_resampling_mode", "legacy"),
        depth_downsample=options.get("depth_loss_downsample", "nearest"),
        prepared_tier_policy=prepared_tier_policy,
        measure_replay_quality=bool(
            getattr(
                getattr(config, "optimization", None),
                "streaming_historical_usefulness_tracking",
                False,
            )
        ),
        motion_depth_confidence_mode=str(
            getattr(args, "streaming_motion_depth_confidence_mode", "off")
        ),
        motion_depth_confidence_native_tolerance_px=float(
            getattr(args, "streaming_motion_depth_confidence_native_tolerance_px", 1.0)
        ),
        motion_depth_confidence_min_weight=float(
            getattr(args, "streaming_motion_depth_confidence_min_weight", 0.05)
        ),
    )
    bridge_options = {
        "consumer_owner": "causal-prepared-gpu",
        "max_items_per_tick": 4,
        "retained_commit_clearer": runtime.clear_retained_prepared_commit,
    }
    bridge = PreparedRuntimeBridge(
        handoff,
        producer,
        ready_owner=owner,
        **bridge_options,
    )
    promotion_owner = None
    resolver_context = getattr(runtime, "prepared_cold_promotion_context_for", None)
    if callable(getattr(runtime, "prepared_cold_promotion_shortlist", None)) and callable(resolver_context):
        def reload_pixels(frame_id: int, handle: Any):
            """Reload pixels for a frame from admission prefetch or schedule loading."""
            resident = runtime._pop_admission_prefetch(frame_id)
            if resident is not None:
                return resident
            runtime._maybe_schedule_admission_prefetch(frame_id, handle)
            return None

        scheduler = owner.claim_scheduler("causal-prepared-cold-scheduler")
        resolver = FrameObservationPreparedColdResolver(
            runtime._observation_db,
            context_for_source=resolver_context,
            reload_pixels=reload_pixels,
        )
        promotion_owner = PreparedColdPromotionOwner(
            bridge,
            ready_scheduler=scheduler,
            resolve=resolver,
            retire_frame=runtime._retire_historical_identity,
        )

    prep_worker = PreparedSourcePrepWorker(bridge, promotion_owner=promotion_owner).start()
    return CausalPreparedReadyB4Path(
        runtime=runtime,
        bridge=bridge,
        adapter_factory=lambda: ReadyB4StreamingOptimizationAdapter(
            bridge,
            runtime=runtime,
            transfer_owner=transfer_owner,
            batch_size=batch_size,
            source_insertion=source_insertion,
            ingress_result_provider=prep_worker.take_result,
            warmup_complete=True,
            trace_ready_state=bool(
                str(getattr(args, "causal_decision_trace", "") or "").strip()
                or bool(getattr(args, "causal_ready_diagnostics", False))
            ),
            timing_scope=_measurement_timing_scope(),
        ),
        step_inputs_factory=lambda: _step_inputs_for_runtime(runtime),
        source_complete_phase_hook=runtime.mark_source_complete,
        prep_worker=prep_worker,
        promotion_owner=promotion_owner,
    )


__all__ = ["CausalPreparedReadyB4Path", "PreparedBootstrapOutcome", "build_causal_prepared_ready_path"]
