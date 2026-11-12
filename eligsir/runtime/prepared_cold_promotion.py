"""Runtime-resolved COLD promotion with worker-only device preparation."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
import numbers
import threading
from typing import Any, Callable, Iterable

import numpy as np

from eligsir.contracts import BufferRef, SensorPacket, SourceIdentity, TrackingObservation
from eligsir.contracts.control import SemanticAction
from eligsir.mapping.observation_db import ColdPixelHandle, FrameObservationDB
from eligsir.mapping.scheduler.historical_candidate_index import HistoricalReplayCandidate
from eligsir.runtime.prepared_ingress import PreparedIngressItem
from eligsir.runtime.prepared_source_producer import PreparedSourceProduceResult
from eligsir.runtime.ready_source import ReadySourceScheduler


class PreparedColdPromotionState(str, Enum):
    READY = "ready"
    PENDING = "pending"
    RETIRE = "retire"


@dataclass(frozen=True, slots=True)
class PreparedColdPromotionResolution:
    state: PreparedColdPromotionState
    item: PreparedIngressItem | Any | None = None
    promotion_permit: Any | None = None
    initial_fresh: bool = False

    @classmethod
    def ready(
        cls,
        item: PreparedIngressItem | Any,
        *,
        promotion_permit: Any | None = None,
        initial_fresh: bool | None = None,
    ):
        if initial_fresh is None:
            initial_fresh = (
                getattr(getattr(item, "disposition", None), "semantic_action", None)
                is SemanticAction.COMMIT
            )
        return cls(
            PreparedColdPromotionState.READY,
            item,
            promotion_permit,
            initial_fresh,
        )

    @classmethod
    def pending(cls):
        return cls(PreparedColdPromotionState.PENDING)

    @classmethod
    def retire(cls):
        return cls(PreparedColdPromotionState.RETIRE)

    def __post_init__(self) -> None:
        if not isinstance(self.state, PreparedColdPromotionState):
            raise TypeError("state must be PreparedColdPromotionState")
        if self.state is not PreparedColdPromotionState.READY:
            if self.item is not None or self.promotion_permit is not None or self.initial_fresh:
                raise ValueError("pending/retire promotion cannot carry work")
            return
        if not isinstance(self.initial_fresh, bool):
            raise TypeError("initial_fresh must be bool")
        if not isinstance(self.item, PreparedIngressItem) or self.item.disposition is None:
            raise ValueError("ready promotion requires a controlled ingress item")
        action = self.item.disposition.semantic_action
        if action is SemanticAction.COMMIT:
            if self.promotion_permit is not None:
                raise ValueError("COMMIT promotion cannot carry a historical permit")
        elif action is SemanticAction.DEFER:
            if self.initial_fresh or self.promotion_permit is None:
                raise ValueError("DEFER promotion requires a historical permit")
        else:
            raise ValueError("ready promotion requires COMMIT or DEFER")


@dataclass(slots=True)
class _PromotionRequest:
    frame_id: int
    resolution: PreparedColdPromotionResolution
    protected_sources: frozenset[SourceIdentity]
    generation: int = 1


@dataclass(frozen=True, slots=True)
class PreparedColdPromotionContext:
    observation: TrackingObservation
    disposition: Any
    control_token: Any
    promotion_permit: Any | None = None
    initial_fresh: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.observation, TrackingObservation) or self.disposition is None or self.control_token is None:
            raise TypeError("promotion context requires tracking, disposition and control token")
        action = self.disposition.semantic_action
        if action is SemanticAction.COMMIT:
            if self.promotion_permit is not None:
                raise ValueError("COMMIT promotion cannot carry a historical permit")
        elif action is SemanticAction.DEFER:
            if self.initial_fresh or self.promotion_permit is None:
                raise ValueError("DEFER promotion requires a historical permit")
        else:
            raise ValueError("promotion context requires COMMIT or DEFER")


class FrameObservationPreparedColdResolver:
    """Resolve one durable historical record without blocking."""

    def __init__(
        self,
        observation_db: FrameObservationDB,
        *,
        context_for_source: Callable[[Any], PreparedColdPromotionContext | None],
        reload_pixels: Callable[[int, ColdPixelHandle], tuple[Any, Any | None] | None] | None = None,
    ) -> None:
        if not isinstance(observation_db, FrameObservationDB):
            raise TypeError("observation_db must be FrameObservationDB")
        if not callable(context_for_source):
            raise TypeError("context_for_source must be callable")
        if reload_pixels is not None and not callable(reload_pixels):
            raise TypeError("reload_pixels must be callable or None")
        self._db = observation_db
        self._context_for_source = context_for_source
        self._reload_pixels = reload_pixels

    @staticmethod
    def _encodings(handle: ColdPixelHandle, rgb: np.ndarray, depth: np.ndarray | None):
        rgb_encoding = handle.rgb_encoding or ("rgb8" if rgb.dtype == np.uint8 else "rgb_float32_hwc")
        if depth is None:
            return rgb_encoding, None, None
        return (
            rgb_encoding,
            handle.depth_encoding or ("16UC1" if depth.dtype == np.uint16 else "depth_float32"),
            handle.depth_units or ("mm" if depth.dtype == np.uint16 else "m"),
        )

    def __call__(self, candidate: HistoricalReplayCandidate) -> PreparedColdPromotionResolution:
        record = self._db.get_frame(int(candidate.frame_id))
        if record is None or not record.historical_eligible:
            return PreparedColdPromotionResolution.retire()
        handle = self._db.get_cold_pixels(record.frame_id)
        if handle is None or not handle.can_materialize:
            return PreparedColdPromotionResolution.retire()
        context = self._context_for_source(record.source_identity)
        if context is None:
            return PreparedColdPromotionResolution.pending()
        source = context.observation.source
        if source.fingerprint() != record.source_identity:
            raise RuntimeError("durable COLD source identity differs from control context")
        rgb, depth = handle.rgb, handle.depth
        if rgb is None and handle.reloadable:
            if self._reload_pixels is None:
                return PreparedColdPromotionResolution.pending()
            reloaded = self._reload_pixels(record.frame_id, handle)
            if reloaded is None:
                return PreparedColdPromotionResolution.pending()
            rgb, depth = reloaded
        if rgb is None:
            return PreparedColdPromotionResolution.retire()

        rgb = np.asarray(rgb)
        depth = None if depth is None else np.asarray(depth)
        if rgb.ndim != 3 or (depth is not None and depth.ndim != 2):
            return PreparedColdPromotionResolution.retire()
        rgb_encoding, depth_encoding, depth_units = self._encodings(handle, rgb, depth)
        packet = SensorPacket(
            source=source,
            rgb=BufferRef.from_numpy(rgb, encoding=rgb_encoding),
            depth=None if depth is None else BufferRef.from_numpy(depth, encoding=depth_encoding, units=depth_units),
            intrinsics=handle.intrinsics,
            tracking=context.observation,
            reload_token=handle.reload_token,
        )
        item = PreparedIngressItem(
            context.control_token.sequence,
            packet,
            context.observation,
            disposition=context.disposition,
            control_token=context.control_token,
        )
        return PreparedColdPromotionResolution.ready(
            item,
            promotion_permit=context.promotion_permit,
            initial_fresh=context.initial_fresh,
        )


class PreparedColdPromotionOwner:
    """Bounded COLD page-in queue whose worker never reads runtime semantics."""

    def __init__(
        self,
        bridge: Any,
        *,
        ready_scheduler: ReadySourceScheduler,
        resolve: Callable[[HistoricalReplayCandidate], PreparedColdPromotionResolution],
        retire_frame: Callable[[int], None] | None = None,
        capacity: int = 16,
        progress_callback: Callable[[], None] | None = None,
    ) -> None:
        if not callable(getattr(bridge, "submit_promoted", None)):
            raise TypeError("bridge must provide submit_promoted")
        if not isinstance(ready_scheduler, ReadySourceScheduler):
            raise TypeError("ready_scheduler must be ReadySourceScheduler")
        if not callable(resolve):
            raise TypeError("resolve must be callable")
        if retire_frame is not None and not callable(retire_frame):
            raise TypeError("retire_frame must be callable or None")
        if isinstance(capacity, bool) or not isinstance(capacity, numbers.Integral) or capacity < 1:
            raise ValueError("capacity must be a positive integer")
        if progress_callback is not None and not callable(progress_callback):
            raise TypeError("progress_callback must be callable or None")
        self._bridge = bridge
        self._ready_scheduler = ready_scheduler
        self._resolve = resolve
        self._retire_frame = retire_frame
        self._capacity = int(capacity)
        source_batch_limit = getattr(bridge, "max_items_per_tick", 1)
        if isinstance(source_batch_limit, bool) or not isinstance(source_batch_limit, numbers.Integral):
            raise TypeError("bridge max_items_per_tick must be a positive integer")
        if int(source_batch_limit) < 1:
            raise ValueError("bridge max_items_per_tick must be a positive integer")
        self._source_batch_limit = int(source_batch_limit)
        self._queue: deque[_PromotionRequest] = deque()
        self._scheduled: dict[int, _PromotionRequest | None] = {}
        self._promotion_fence: tuple[int, int, int] | None = None
        self._lock = threading.Lock()
        self._progress_callback = progress_callback

    def set_progress_callback(self, callback: Callable[[], None] | None) -> None:
        """Register a callback to signal when promotion work is enqueued.

        The callback is called from within schedule_shortlist() to notify waiters
        when COLD promotion requests become available, enabling EDGE-triggered wakeup
        rather than LEVEL-triggered polling.
        """
        if callback is not None and not callable(callback):
            raise TypeError("progress_callback must be callable or None")
        self._progress_callback = callback

    def _discard(self, frame_id: int, request: _PromotionRequest | None = None) -> None:
        with self._lock:
            if request is None or self._scheduled.get(frame_id) is request:
                self._scheduled.pop(frame_id, None)

    def schedule_shortlist(
        self,
        candidates: Iterable[HistoricalReplayCandidate],
        *,
        promotion_fence: tuple[int, int, int] | None = None,
        protected_sources: frozenset[SourceIdentity] = frozenset(),
    ) -> int:
        """Resolve on the runtime thread and queue immutable preparation work."""
        if not isinstance(protected_sources, frozenset) or not all(
            isinstance(source, SourceIdentity) for source in protected_sources
        ):
            raise TypeError("protected_sources must be a frozenset of SourceIdentity")
        candidates = tuple(candidates)
        if not all(isinstance(candidate, HistoricalReplayCandidate) for candidate in candidates):
            raise TypeError("candidates must contain HistoricalReplayCandidate")
        candidate_ids = {int(candidate.frame_id) for candidate in candidates}
        with self._lock:
            self._promotion_fence = promotion_fence
            for frame_id, request in tuple(self._scheduled.items()):
                if frame_id not in candidate_ids:
                    self._scheduled.pop(frame_id)
                elif request is not None:
                    if request.resolution.item.disposition.semantic_action is SemanticAction.DEFER and (
                        getattr(request.resolution.promotion_permit, "promotion_snapshot_id", None),
                        getattr(request.resolution.promotion_permit, "control_version", None),
                        getattr(request.resolution.promotion_permit, "topology_version", None),
                    ) != promotion_fence:
                        self._scheduled.pop(frame_id)
                    else:
                        request.protected_sources = protected_sources
        scheduled = 0
        for candidate in candidates:
            frame_id = int(candidate.frame_id)
            with self._lock:
                if frame_id in self._scheduled:
                    continue
                if len(self._scheduled) >= self._capacity:
                    break
                self._scheduled[frame_id] = None
            try:
                resolution = self._resolve(candidate)
            except BaseException:
                self._discard(frame_id)
                raise
            if not isinstance(resolution, PreparedColdPromotionResolution):
                self._discard(frame_id)
                raise TypeError("resolve must return PreparedColdPromotionResolution")
            if resolution.state is PreparedColdPromotionState.PENDING:
                self._discard(frame_id)
                continue
            if resolution.state is PreparedColdPromotionState.RETIRE:
                self._discard(frame_id)
                if self._retire_frame is not None:
                    self._retire_frame(frame_id)
                continue
            request = _PromotionRequest(
                frame_id,
                resolution,
                protected_sources,
                self._ready_scheduler.next_generation(resolution.item.packet.source),
            )
            with self._lock:
                self._scheduled[frame_id] = request
                self._queue.append(request)
                # Signal progress via callback (EDGE-triggered wakeup like ingress commits)
                if self._progress_callback is not None:
                    self._progress_callback()
            scheduled += 1
        return scheduled

    def _take(self) -> _PromotionRequest | None:
        with self._lock:
            while self._queue:
                request = self._queue.popleft()
                if self._scheduled.get(request.frame_id) is request:
                    return request
            return None

    def _take_compatible(self, request: _PromotionRequest) -> tuple[_PromotionRequest, ...]:
        """Take the immediately available compatible prefix without waiting."""
        item = request.resolution.item.packet
        key = (
            request.resolution.initial_fresh,
            item.rgb.spec.shape,
            item.rgb.spec.dtype,
            None if item.depth is None else item.depth.spec.shape,
            None if item.depth is None else item.depth.spec.dtype,
        )
        batch = [request]
        with self._lock:
            while self._queue and len(batch) < self._source_batch_limit:
                candidate = self._queue[0]
                candidate_item = candidate.resolution.item.packet
                candidate_key = (
                    candidate.resolution.initial_fresh,
                    candidate_item.rgb.spec.shape,
                    candidate_item.rgb.spec.dtype,
                    None if candidate_item.depth is None else candidate_item.depth.spec.shape,
                    None if candidate_item.depth is None else candidate_item.depth.spec.dtype,
                )
                if self._scheduled.get(candidate.frame_id) is not candidate:
                    self._queue.popleft()
                    continue
                if candidate_key != key:
                    break
                batch.append(self._queue.popleft())
        return tuple(batch)

    def _retry(self, request: _PromotionRequest) -> None:
        with self._lock:
            if self._scheduled.get(request.frame_id) is request and not any(
                queued is request for queued in self._queue
            ):
                self._queue.append(request)

    def _finish(self, request: _PromotionRequest) -> None:
        self._discard(request.frame_id, request)

    def _retry_after_nonresident(
        self,
        request: _PromotionRequest,
        *,
        count: int = 1,
        protected_sources: frozenset[SourceIdentity] | None = None,
    ) -> bool:
        """Free one eligible READY lease before retrying a retained request."""
        with self._lock:
            if self._scheduled.get(request.frame_id) is not request:
                return True
            protected_sources = request.protected_sources if protected_sources is None else protected_sources
            if request.resolution.initial_fresh:
                if not protected_sources:
                    self._scheduled.pop(request.frame_id)
                    return True
                return bool(
                    self._ready_scheduler.evict_fresh(
                        protected_sources=protected_sources
                    )
                )
            if self._ready_scheduler.replay_residency_depth < self._ready_scheduler.replay_capacity:
                return False
            return bool(
                self._ready_scheduler.evict_replay(
                    count=count,
                    protected_sources=protected_sources,
                )
            )

    def _preflight_submit(self, request: _PromotionRequest) -> bool | None:
        """Return progress for an already-resolved request, or ``None`` to submit it."""
        resolution = request.resolution
        item = resolution.item
        source = item.packet.source
        if self._ready_scheduler.has_source(source):
            if self._ready_scheduler.has_source(source, ready=True):
                self._finish(request)
                return True
            self._retry(request)
            return False
        if item.disposition.semantic_action is SemanticAction.DEFER:
            permit = resolution.promotion_permit
            with self._lock:
                fence = self._promotion_fence
            if fence is None or (
                getattr(permit, "promotion_snapshot_id", None),
                getattr(permit, "control_version", None),
                getattr(permit, "topology_version", None),
            ) != fence:
                self._finish(request)
                return True
        return None

    def _submit(self, batch: tuple[_PromotionRequest, ...]) -> tuple[PreparedSourceProduceResult, ...]:
        """Submit one already-validated compatible cohort through its existing bridge surface."""
        if len(batch) == 1:
            request = batch[0]
            resolution = request.resolution
            result = (
                self._bridge.submit_retained_commit(resolution.item, generation=request.generation)
                if resolution.initial_fresh
                else self._bridge.submit_promoted(
                    resolution.item,
                    promotion_permit=resolution.promotion_permit,
                    generation=request.generation,
                )
            )
            if not isinstance(result, PreparedSourceProduceResult):
                raise TypeError("prepared bridge returned invalid producer result")
            return (result,)
        if batch[0].resolution.initial_fresh:
            results = self._bridge.submit_retained_commit_batch(
                tuple(item.resolution.item for item in batch),
                generations=tuple(item.generation for item in batch),
            )
        else:
            results = self._bridge.submit_promoted_batch(
                tuple(item.resolution.item for item in batch),
                promotion_permits=tuple(item.resolution.promotion_permit for item in batch),
                generations=tuple(item.generation for item in batch),
            )
        if len(results) != len(batch):
            raise TypeError("prepared bridge batch returned invalid producer results")
        return tuple(results)

    def _resolve_results(
        self,
        batch: tuple[_PromotionRequest, ...],
        results: tuple[PreparedSourceProduceResult, ...],
    ) -> bool:
        """Finish successes and retry only bounded durable outcomes for one cohort."""
        progressed = False
        retries = []
        retained = []
        for item, result in zip(batch, results):
            if result in (
                PreparedSourceProduceResult.ENQUEUED,
                PreparedSourceProduceResult.CAPABILITY_UNAVAILABLE,
                PreparedSourceProduceResult.DROPPED_READY_OWNER,
            ):
                self._finish(item)
                progressed = True
            elif result in (
                PreparedSourceProduceResult.DEFERRED_POOL_EXHAUSTED,
                PreparedSourceProduceResult.RETAINED_NOT_RESIDENT,
            ):
                retries.append(item)
                if result is PreparedSourceProduceResult.RETAINED_NOT_RESIDENT:
                    retained.append(item)
            else:
                raise RuntimeError(f"unknown prepared promotion result: {result!r}")
        # A singleton keeps the long-standing exact plan guard.  A grouped
        # capacity repair, however, must protect every cohort member: one
        # sibling may already have published while another needs eviction.
        protected_sources = (
            batch[0].protected_sources
            if len(batch) == 1
            else frozenset(
                source
                for item in batch
                for source in (*item.protected_sources, item.resolution.item.packet.source)
            )
        )
        evicted = bool(retained) and self._retry_after_nonresident(
            retained[0], count=len(retained), protected_sources=protected_sources
        )
        for item in retries:
            self._retry(item)
        return progressed or evicted

    def pump_once(self) -> bool:
        """Return whether one promotion made progress; blocked retries return false."""
        request = self._take()
        if request is None:
            return False
        batch = self._take_compatible(request)
        submit_batch = []
        preflight_progress = False
        for item in batch:
            outcome = self._preflight_submit(item)
            if outcome is None:
                submit_batch.append(item)
            else:
                preflight_progress = preflight_progress or outcome
        submit_batch = tuple(submit_batch)
        if not submit_batch:
            return preflight_progress
        try:
            submit_progress = self._resolve_results(submit_batch, self._submit(submit_batch))
            return preflight_progress or submit_progress
        except BaseException:
            retry = len(batch) > 1
            on_error = batch if retry else submit_batch
            for item in on_error:
                (self._retry if retry else self._finish)(item)
            raise


__all__ = [
    "PreparedColdPromotionOwner",
    "PreparedColdPromotionContext",
    "FrameObservationPreparedColdResolver",
    "PreparedColdPromotionResolution",
    "PreparedColdPromotionState",
]
