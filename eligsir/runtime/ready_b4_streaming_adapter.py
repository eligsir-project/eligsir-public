"""Exact READY batch availability adapter for the canonical mapping runtime."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable

from eligsir.mapping.batch_contract import (
    qualified_camera_count_for_batch,
    validate_supervision_camera_count,
    validate_supervision_batch_size,
)
from eligsir.runtime.prepared_runtime_bridge import PreparedRuntimeBridge, PreparedRuntimeBridgeResult
from eligsir.runtime.ready_camera_binder import bind_ready_camera
from eligsir.runtime.ready_only_tick import ReadyOnlyTickSelection, ReadySourceBinding, ReadySourceVersions
from eligsir.runtime.ready_source import ProvisionalReplayEviction, ReadySourceLane
from eligsir.runtime.transfers import TransferEvent


class ReadyB4StreamingStepState(str, Enum):
    EXECUTED = "executed"
    STARVED = "starved"
    STALE = "stale"
    CAPABILITY_UNAVAILABLE = "capability_unavailable"
    RESERVOIR_CAPACITY_UNAVAILABLE = "reservoir_capacity_unavailable"
    DEFERRED = "deferred"


@dataclass(frozen=True, slots=True)
class ReadyB4StreamingStepInputs:
    snapshot: Any | None
    work_plan: Any | None


@dataclass(frozen=True, slots=True)
class ReadyB4StreamingStepResult:
    state: ReadyB4StreamingStepState
    ingress: PreparedRuntimeBridgeResult
    selection: ReadyOnlyTickSelection
    result: Any | None = None
    completed_supervision_sources: tuple[Any, ...] = ()
    reason: str | None = None
    work_plan: Any | None = None
    ready_counters: Any | None = None
    ready_count: int | None = None
    # Cumulative frozen-plan invalidations for an ABSENT authorized source
    # (bounded re-plan); surfaced on the low-rate progress line.
    absent_replans: int | None = None

    @property
    def executed(self) -> bool:
        return self.state is ReadyB4StreamingStepState.EXECUTED

    @property
    def retryable(self) -> bool:
        """Whether READY availability may change without reauthorizing work."""

        return self.state is ReadyB4StreamingStepState.STARVED

    def __getattr__(self, name: str) -> Any:
        result = object.__getattribute__(self, "result")
        if name == "optimizer_result":
            return result
        if result is None:
            if name in {"optimizer_steps", "supervision_views"}:
                return 0
            raise AttributeError(name)
        return getattr(result, name)


@dataclass(frozen=True, slots=True)
class _ReadyTransaction:
    snapshot: Any
    plan: Any
    cameras: dict[Any, Any]
    bindings: tuple[Any, ...]
    source_insertion: bool = False


def _compute_completion() -> TransferEvent:
    import torch

    if not torch.cuda.is_available():
        return TransferEvent()
    event = torch.cuda.Event(blocking=False)
    event.record(torch.cuda.current_stream())
    return TransferEvent(event)


class ReadyB4StreamingOptimizationAdapter:
    """Realize one qualified exact READY batch and enter canonical ``tick``."""

    _ABSENT_REPLAN_AFTER_TICKS = 64

    def __init__(
        self,
        bridge: PreparedRuntimeBridge,
        *,
        runtime: Any,
        transfer_owner: Any,
        batch_size: int = 4,
        source_insertion: bool = False,
        ingress_result_provider: Callable[[], PreparedRuntimeBridgeResult] | None = None,
        warmup_complete: bool = False,
        trace_ready_state: bool = False,
        timing_scope: Any = None,
    ) -> None:
        if not isinstance(bridge, PreparedRuntimeBridge):
            raise TypeError("bridge must be PreparedRuntimeBridge")
        if not callable(getattr(runtime, "tick", None)):
            raise TypeError("runtime must provide canonical tick()")
        batch_size = validate_supervision_batch_size(batch_size)
        if qualified_camera_count_for_batch(batch_size) != batch_size:
            raise ValueError(
                "ReadyB4StreamingOptimizationAdapter requires a qualified exact "
                f"batch size; got {batch_size}"
            )
        self._bridge = bridge
        self._runtime = runtime
        self._transfer_owner = transfer_owner
        self._batch_size = batch_size
        self._source_insertion = bool(source_insertion)
        self._ingress_result_provider = ingress_result_provider
        self._warmup_complete = bool(warmup_complete)
        self._trace_ready_state = bool(trace_ready_state)
        self._timing_scope = timing_scope or (lambda _name: nullcontext())
        # Bounded re-plan tracking for ABSENT authorized sources.
        self._absent_starved_ticks = 0
        self._absent_starved_plan_id: int | None = None
        self.absent_replans = 0
        # Issue #1663: drain counters for acceptance testing
        self._first_supervision_completions = 0
        self._first_supervision_sources: set[Any] = set()  # Fingerprints of sources completed first supervision
        self._replay_only_steps_while_fresh_pending = 0
        self._admitted_commit_sources: set[Any] = set()  # Fingerprints of distinct fresh sources ever selected

    @property
    def first_supervision_completions(self) -> int:
        """Return cumulative count of distinct sources that completed first supervision."""
        return self._first_supervision_completions

    @property
    def replay_only_steps_while_fresh_pending(self) -> int:
        """Return count of replay-only steps executed while fresh sources were pending."""
        return self._replay_only_steps_while_fresh_pending

    @property
    def admitted_commit_count(self) -> int:
        """Return count of distinct fresh sources ever offered to the READY owner."""
        return len(self._admitted_commit_sources)

    def _ingress(self) -> PreparedRuntimeBridgeResult:
        result = (
            PreparedRuntimeBridgeResult(0, 0, 0, 0)
            if self._ingress_result_provider is None
            else self._ingress_result_provider()
        )
        if not isinstance(result, PreparedRuntimeBridgeResult):
            raise TypeError("ingress_result_provider must return PreparedRuntimeBridgeResult")
        return result

    def _selection(
        self, leases: tuple[Any, ...], *, fresh: frozenset[Any], batch_size: int | None = None
    ) -> ReadyOnlyTickSelection:
        batch_size = self._batch_size if batch_size is None else batch_size
        fresh_count = sum(lease.source_identity in fresh for lease in leases)
        return ReadyOnlyTickSelection(
            leases=leases,
            fresh_count=fresh_count,
            replay_count=len(leases) - fresh_count,
            fallback_to_replay=len(leases) > fresh_count and fresh_count < batch_size,
            starved=self._warmup_complete and len(leases) < batch_size,
        )

    def _consume_plan(self, plan: Any) -> None:
        consumed = getattr(self._runtime, "_advance_prepared_ready_plan", None)
        if callable(consumed):
            consumed(plan)
        for name in ("_latest_tick_work", "_latest_report_work"):
            work = getattr(self._runtime, name, None)
            if work is not None and len(work) == 2 and work[1] is plan:
                setattr(self._runtime, name, None)

    def _retired_missing_source(self, missing: set[Any]) -> bool:
        lookup = getattr(getattr(self._runtime, "_observation_db", None), "frame_id_for_source", None)
        return callable(lookup) and any(lookup(source) is None for source in missing)

    def tick(self, inputs: ReadyB4StreamingStepInputs) -> ReadyB4StreamingStepResult:
        if not isinstance(inputs, ReadyB4StreamingStepInputs):
            raise TypeError("inputs must be ReadyB4StreamingStepInputs")
        ingress = self._ingress()
        plan = inputs.work_plan
        empty_selection = self._selection((), fresh=frozenset())

        def outcome(
            state: ReadyB4StreamingStepState,
            *,
            selection: ReadyOnlyTickSelection = empty_selection,
            result: Any | None = None,
            completed: tuple[Any, ...] = (),
            reason: str | None = None,
            ready_count: int | None = None,
        ) -> ReadyB4StreamingStepResult:
            return ReadyB4StreamingStepResult(
                state,
                ingress,
                selection,
                result=result,
                completed_supervision_sources=completed,
                reason=reason,
                work_plan=plan,
                ready_counters=(
                    self._bridge.ready_owner.counters if self._trace_ready_state else None
                ),
                ready_count=ready_count,
                absent_replans=int(self.absent_replans),
            )

        if plan is None:
            return outcome(ReadyB4StreamingStepState.DEFERRED, reason="prepared_work_plan_unavailable")

        work = tuple(getattr(plan, "supervision_work", ())) + tuple(getattr(plan, "replay_work", ()))
        constellation = getattr(plan, "supervision_constellation", None)
        raw_batch_size = (
            self._batch_size
            if constellation is None
            else getattr(constellation, "realized_batch_size", 0)
        )
        raw_camera_count = (
            raw_batch_size
            if constellation is None
            else getattr(constellation, "realized_camera_count", None)
        )
        try:
            if type(raw_batch_size) is not int or type(raw_camera_count) is not int:
                raise TypeError("prepared constellation B/C must be exact integers")
            batch_size = validate_supervision_batch_size(raw_batch_size)
            camera_count = validate_supervision_camera_count(raw_camera_count)
        except (TypeError, ValueError):
            batch_size = 0
            camera_count = 0
        if batch_size and camera_count:
            full, remainder = divmod(batch_size, camera_count)
            expected_microbatch_sizes = (camera_count,) * full + (
                (remainder,) if remainder else ()
            )
            declared_microbatch_sizes = getattr(
                constellation, "microbatch_sizes", expected_microbatch_sizes
            )
            partition_valid = (
                camera_count <= batch_size
                and (
                    getattr(
                        constellation, "microbatch_count", len(expected_microbatch_sizes)
                    )
                    == len(expected_microbatch_sizes)
                )
                and (
                    declared_microbatch_sizes is not None
                    and tuple(declared_microbatch_sizes) == expected_microbatch_sizes
                )
            )
        else:
            partition_valid = False
        if (
            batch_size == 0
            or batch_size > self._batch_size
            or not partition_valid
            or len(work) != batch_size
            or (self._source_insertion and batch_size < 4)
        ):
            return outcome(
                ReadyB4StreamingStepState.DEFERRED,
                reason="prepared_authorized_plan_is_not_exact_qualified_batch",
            )
        expected = tuple(getattr(item, "source", None) for item in work)
        if (
            None in expected
            or len(set(expected)) != batch_size
        ):
            return outcome(
                ReadyB4StreamingStepState.DEFERRED,
                reason=f"prepared_authorized_plan_is_not_exact_b{batch_size}",
            )
        planned_tiers = {getattr(getattr(item, "resolution_tier", None), "value", None) for item in work}
        if len(planned_tiers) != 1 or None in planned_tiers:
            return outcome(
                ReadyB4StreamingStepState.CAPABILITY_UNAVAILABLE,
                reason="ready_b4_requires_one_homogeneous_prepared_tier",
            )
        planned_tier = planned_tiers.pop()
        if getattr(plan, "replay_insertion", None) is not None:
            return outcome(
                ReadyB4StreamingStepState.CAPABILITY_UNAVAILABLE,
                reason=(
                    "prepared_replay_depth_insertion_unsupported"
                    if self._source_insertion
                    else "prepared_depth_insertion_requires_prepared_mapping_frame"
                ),
            )
        if not self._source_insertion and bool(
            getattr(getattr(plan, "insertion", None), "enabled", False)
        ):
            return outcome(
                ReadyB4StreamingStepState.CAPABILITY_UNAVAILABLE,
                reason="prepared_depth_insertion_requires_prepared_mapping_frame",
            )

        supervision = frozenset(getattr(plan, "supervision_sources", ()))
        replay = frozenset(getattr(plan, "replay_sources", ()))
        fresh = frozenset(source for source in expected if source in supervision)
        requested_replay = frozenset(source for source in expected if source in replay)
        if len(fresh) + len(requested_replay) != batch_size:
            return outcome(
                ReadyB4StreamingStepState.DEFERRED,
                reason="prepared_authorized_plan_work_mismatch",
            )

        owner = self._bridge.ready_owner
        selected = tuple(
            owner.select(
                batch_size,
                fresh_sources=fresh,
                replay_sources=requested_replay,
                exact_sources=expected,
            )
        )
        by_source = {lease.source_identity: lease for lease in selected}
        if len(selected) != batch_size or set(by_source) != set(expected):
            selection = self._selection(selected, fresh=fresh, batch_size=batch_size)
            missing = set(expected) - set(by_source)
            if missing and self._retired_missing_source(missing):
                self._consume_plan(plan)
                return outcome(
                    ReadyB4StreamingStepState.DEFERRED,
                    selection=selection,
                    reason="prepared_authorized_source_retired",
                )
            # Bounded re-plan for ABSENT authorized sources (issue #279):
            # track consecutive STARVED ticks with zero in-flight requests;
            # after 64 ticks with no progress, force a fresh plan.
            counters = owner.counters
            readiness = owner.exact_readiness(
                fresh_sources=fresh,
                replay_sources=requested_replay,
                exact_sources=expected,
            )
            absent = {
                source for source, state in readiness if state.startswith("ABSENT_")
            }
            if self._absent_starved_plan_id != id(plan):
                self._absent_starved_plan_id = id(plan)
                self._absent_starved_ticks = 0
            if absent and counters.in_flight == 0:
                self._absent_starved_ticks += 1
            else:
                self._absent_starved_ticks = 0
            if (
                absent
                and counters.in_flight == 0
                and self._absent_starved_ticks >= self._ABSENT_REPLAN_AFTER_TICKS
            ):
                self._absent_starved_ticks = 0
                self.absent_replans += 1
                self._consume_plan(plan)
                return outcome(
                    ReadyB4StreamingStepState.DEFERRED,
                    selection=selection,
                    reason="prepared_authorized_source_absent",
                )
            reason = "prepared_authorized_sources_not_ready"
            ready_count = None
            if self._trace_ready_state:
                ready_count = sum(state == "READY" for _source, state in readiness)
                reason += ":" + ",".join(
                    f"{source.fingerprint()}={state}"
                    for source, state in readiness
                )
                reason += (
                    f";ready_post_select=f{counters.fresh_ready_count},r{counters.replay_ready_count};"
                    f"lane_occupancy=f{counters.fresh_pending},r{counters.replay_pending},"
                    f"i{counters.in_flight},nonresident={counters.full_not_resident_count}"
                )
                if absent:
                    reason += f";absent_ticks={self._absent_starved_ticks}"
            return outcome(
                ReadyB4StreamingStepState.STARVED,
                selection=selection,
                reason=reason,
                ready_count=ready_count,
            )
        selection = self._selection(
            tuple(by_source[source] for source in expected), fresh=fresh, batch_size=batch_size
        )
        if any(planned_tier not in lease.available_tiers for lease in selection.leases):
            self._bridge.requeue_ready_only_sources(selection, ready_event=None)
            return outcome(
                ReadyB4StreamingStepState.CAPABILITY_UNAVAILABLE,
                selection=selection,
                reason="ready_b4_requested_tier_unavailable",
            )

        successful_return_lanes: dict[Any, ReadySourceLane] = {}
        bound = []
        executed = False
        consumption_started = False
        # Exceptions may follow device submission; non-committing evictions are restored.
        replay_evictions: list[ProvisionalReplayEviction] = []
        try:
            current = tuple(self._runtime._versions())
            planned = (
                int(getattr(plan, "map_content_version", -1)),
                int(getattr(plan, "topology_version", -1)),
                int(getattr(plan, "parameter_version", -1)),
            )
            if current != planned:
                return outcome(
                    ReadyB4StreamingStepState.STALE,
                    selection=selection,
                    reason="prepared_work_plan_version_stale",
                )
            return_lanes = {source: ReadySourceLane.REPLAY for source in expected}
            for eviction_attempt in range(selection.fresh_count + 1):
                try:
                    owner.reserve_return_lanes(selection.leases, lane_by_source=return_lanes)
                    break
                except RuntimeError as error:
                    if str(error) != "ready-source return lane capacity is unavailable" or (
                        eviction_attempt == selection.fresh_count
                    ):
                        return outcome(
                            ReadyB4StreamingStepState.RESERVOIR_CAPACITY_UNAVAILABLE,
                            selection=selection,
                            reason=str(error),
                        )
                    with self._timing_scope("ready_b4_eviction_hold"):
                        evicted = owner.evict_replay_provisional(count=1)
                    if not evicted:
                        return outcome(
                            ReadyB4StreamingStepState.RESERVOIR_CAPACITY_UNAVAILABLE,
                            selection=selection,
                            reason=str(error),
                        )
                    replay_evictions.extend(evicted)

            versions = ReadySourceVersions(*current)
            correction_for_source = getattr(
                getattr(self._runtime, "_observation_db", None),
                "prepared_pose_correction_for_source",
                None,
            )
            for lease in selection.leases:
                correction = (
                    correction_for_source(lease.source_identity)
                    if callable(correction_for_source)
                    else None
                )
                binding = ReadySourceBinding(
                    lease=lease, versions=versions, pose_correction=correction
                )
                binding.handoff_to_compute(self._transfer_owner)
                bound.append(bind_ready_camera(binding, tier=planned_tier))
            try:
                result = self._runtime.tick(
                    ready_transaction=_ReadyTransaction(
                        snapshot=inputs.snapshot,
                        plan=plan,
                        cameras={item.source_identity: item.camera for item in bound},
                        bindings=tuple(bound),
                        source_insertion=self._source_insertion,
                    )
                )
            except BaseException:
                consumption_started = True
                raise
            if not bool(getattr(result, "executed", False)):
                return outcome(
                    ReadyB4StreamingStepState.DEFERRED,
                    selection=selection,
                    reason=str(getattr(result, "reason", "canonical_prepared_tick_deferred")),
                )

            executed = True

            # Issue #1663: track admitted fresh commit count (distinct fresh sources ever selected)
            for source in fresh:
                fingerprint = source.fingerprint()
                self._admitted_commit_sources.add(fingerprint)

            # Issue #1663: track replay-only steps while fresh pending
            _owner_counters = getattr(getattr(self._bridge, "ready_owner", None), "counters", None)
            has_supervision_work = bool(getattr(plan, "supervision_work", ()))
            if not has_supervision_work and int(getattr(_owner_counters, "fresh_pending", 0) or 0) > 0:
                self._replay_only_steps_while_fresh_pending += 1

            successful_sources = getattr(result, "successful_supervision_sources", None)
            if successful_sources is None:
                completed = (
                    expected
                    if int(getattr(result, "optimizer_steps", 0)) > 0
                    and int(getattr(result, "supervision_views", 0)) == batch_size
                    else ()
                )
            else:
                successful = frozenset(successful_sources)
                completed = tuple(source for source in expected if source in successful)
            for source in (source for source in completed if source in fresh):
                self._bridge.complete_retained_commit_first_supervision(source)
                successful_return_lanes[source] = ReadySourceLane.REPLAY
                # Issue #1663: track first-supervision completions (count each source once)
                fingerprint = source.fingerprint()
                if fingerprint not in self._first_supervision_sources:
                    self._first_supervision_sources.add(fingerprint)
                    self._first_supervision_completions += 1
            self._consume_plan(plan)
            return outcome(
                ReadyB4StreamingStepState.EXECUTED,
                selection=selection,
                result=result,
                completed=completed,
            )
        finally:
            if replay_evictions:
                owner.resolve_replay_eviction(tuple(replay_evictions), commit=executed)
            self._bridge.requeue_ready_only_sources(
                selection,
                ready_event=_compute_completion() if (executed or consumption_started) else None,
                lane_by_source=successful_return_lanes or None,
            )


__all__ = [
    "ReadyB4StreamingOptimizationAdapter",
    "ReadyB4StreamingStepInputs",
    "ReadyB4StreamingStepResult",
    "ReadyB4StreamingStepState",
]
