"""Bounded device-preparation bridge from accepted ingress to READY leases."""

from __future__ import annotations

from dataclasses import dataclass
import numbers
import threading
from typing import Any, Callable, Mapping

from eligsir.contracts import ModalityEligibility
from eligsir.contracts.control import SemanticAction
from eligsir.mapping.admission.observation_owner import HistoricalPromotionPermit
from eligsir.runtime.prepared_ingress import PreparedIngressItem, PreparedSourceIngressHandoff
from eligsir.runtime.prepared_source_producer import (
    PreparedSourceProduceResult,
    PreparedSourceProducer,
)
from eligsir.runtime.ready_only_tick import ReadyOnlyTickSelection
from eligsir.runtime.ready_source import ReadySourceLane, ReadySourceOwner, ReplayPriority


@dataclass(frozen=True, slots=True)
class PreparedRuntimeBridgeResult:
    submitted: int
    enqueued: int
    deferred: int
    dropped: int
    retained_not_resident: int = 0

    def __post_init__(self) -> None:
        for name in ("submitted", "enqueued", "deferred", "dropped", "retained_not_resident"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, numbers.Integral):
                raise TypeError(f"{name} must be a non-negative integer")
            value = int(value)
            if value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
            object.__setattr__(self, name, value)
        outcomes = self.enqueued + self.deferred + self.dropped + self.retained_not_resident
        if outcomes != self.submitted:
            raise ValueError("prepared bridge requires exactly one outcome per submission")


class PreparedRuntimeBridge:
    """Prepare already-controlled items; never own runtime execution policy."""

    def __init__(
        self,
        handoff: PreparedSourceIngressHandoff,
        producer: PreparedSourceProducer,
        *,
        ready_owner: ReadySourceOwner,
        consumer_owner: Any | None = None,
        max_items_per_tick: int = 1,
        retained_commit_clearer: Callable[[PreparedIngressItem], None] | None = None,
        cold_payload_spill_writer: Callable[[int], tuple[Any, ...]] | None = None,
    ) -> None:
        if not isinstance(handoff, PreparedSourceIngressHandoff):
            raise TypeError("handoff must be PreparedSourceIngressHandoff")
        if not isinstance(producer, PreparedSourceProducer):
            raise TypeError("producer must be PreparedSourceProducer")
        if not isinstance(ready_owner, ReadySourceOwner):
            raise TypeError("ready_owner must be ReadySourceOwner")
        if producer.ready_owner is not ready_owner:
            raise ValueError("producer and bridge must share the same ReadySourceOwner")
        if isinstance(max_items_per_tick, bool) or not isinstance(
            max_items_per_tick, numbers.Integral
        ) or int(max_items_per_tick) < 1:
            raise ValueError("max_items_per_tick must be a positive integer")
        if retained_commit_clearer is not None and not callable(retained_commit_clearer):
            raise TypeError("retained_commit_clearer must be callable or None")
        if cold_payload_spill_writer is not None and not callable(cold_payload_spill_writer):
            raise TypeError("cold_payload_spill_writer must be callable or None")
        self._handoff = handoff
        self._consumer = handoff.claim_consumer(consumer_owner)
        self._producer = producer
        self._ready_owner = ready_owner
        self._max_items_per_tick = int(max_items_per_tick)
        self._retained_commit_clearer = retained_commit_clearer
        self._cold_payload_spill_writer = cold_payload_spill_writer
        self._cold_payload_spill_receipts: list[Any] = []
        self._cold_payload_spill_lock = threading.Lock()
        self._last_background_progress = False
        self._deferred_item: PreparedIngressItem | None = None
        self._retained_commit_ready_items: dict[Any, PreparedIngressItem] = {}

    @property
    def handoff(self) -> PreparedSourceIngressHandoff:
        return self._handoff

    @property
    def ready_owner(self) -> ReadySourceOwner:
        return self._ready_owner

    @property
    def max_items_per_tick(self) -> int:
        """Return the existing bounded prepared-source work budget."""

        return self._max_items_per_tick

    @property
    def last_background_progress(self) -> bool:
        """Whether this worker drain wrote one available COLD spill record."""
        return self._last_background_progress

    def take_cold_payload_spill_receipts(self) -> tuple[Any, ...]:
        """Transfer worker-produced receipts to the mapper without waiting."""
        with self._cold_payload_spill_lock:
            receipts, self._cold_payload_spill_receipts = self._cold_payload_spill_receipts, []
        return tuple(receipts)

    @staticmethod
    def _eligibility(item: PreparedIngressItem) -> ModalityEligibility:
        if item.disposition is not None:
            return item.disposition.modality_eligibility
        return ModalityEligibility(
            rgb_supervision=True,
            depth_supervision=item.packet.depth is not None,
        )

    @staticmethod
    def _prepared_source_supported(item: PreparedIngressItem) -> bool:
        """Accept RGB-only work only when its immutable policy disables depth."""

        eligibility = PreparedRuntimeBridge._eligibility(item)
        return bool(eligibility.rgb_supervision) and (
            item.packet.depth is not None
            or not (
                eligibility.depth_supervision
                or eligibility.depth_insertion
                or eligibility.geometry_population_target
            )
        )

    @staticmethod
    def _ready_lane(
        item: PreparedIngressItem,
        *,
        promotion_permit: Any | None = None,
    ) -> tuple[ReadySourceLane, ReplayPriority] | None:
        disposition = item.disposition
        if disposition is None:
            return ReadySourceLane.FRESH, ReplayPriority.NORMAL
        if disposition.semantic_action is SemanticAction.COMMIT:
            return (
                (ReadySourceLane.FRESH, ReplayPriority.NORMAL)
                if promotion_permit is None
                else None
            )
        if disposition.semantic_action is not SemanticAction.DEFER or not isinstance(
            promotion_permit, HistoricalPromotionPermit
        ):
            return None
        if (
            promotion_permit.source != item.packet.source
            or promotion_permit.disposition_snapshot_id != disposition.snapshot_id
        ):
            return None
        return ReadySourceLane.REPLAY, ReplayPriority.DEMANDED

    @staticmethod
    def _generation(value: int) -> int:
        if isinstance(value, bool) or not isinstance(value, numbers.Integral):
            raise TypeError("generation must be a non-negative integer")
        value = int(value)
        if value < 0:
            raise ValueError("generation must be a non-negative integer")
        return value

    def _submit(
        self,
        item: PreparedIngressItem,
        *,
        lane: ReadySourceLane,
        priority: ReplayPriority,
        generation: int = 0,
        durable_retained: bool,
    ) -> PreparedSourceProduceResult:
        return self._producer.submit(
            item.packet,
            modality_eligibility=self._eligibility(item),
            lane=lane,
            arrival_ordinal=item.packet.source.source_ordinal,
            priority=priority,
            generation=generation,
            control_token=item.control_token,
            durable_retained=durable_retained,
        )

    def _submit_many(
        self,
        items: tuple[PreparedIngressItem, ...],
        *,
        lanes: tuple[ReadySourceLane, ...],
        priorities: tuple[ReplayPriority, ...],
        generations: tuple[int, ...],
    ) -> tuple[PreparedSourceProduceResult, ...]:
        if len(items) > self._max_items_per_tick:
            raise ValueError("prepared source batch exceeds the configured work budget")
        return self._producer.submit_many(
            tuple(item.packet for item in items),
            modality_eligibilities=tuple(self._eligibility(item) for item in items),
            lanes=lanes,
            arrival_ordinals=tuple(item.packet.source.source_ordinal for item in items),
            priorities=priorities,
            generations=generations,
            control_tokens=tuple(item.control_token for item in items),
            durable_retained=(True,) * len(items),
        )

    def _submit_admitted_batch(
        self,
        accepted: list[tuple[int, PreparedIngressItem, ReadySourceLane, ReplayPriority, int]],
    ) -> tuple[PreparedSourceProduceResult, ...]:
        """Submit the already-authorized subset through the single producer path."""

        return () if not accepted else self._submit_many(
            tuple(item for _index, item, _lane, _priority, _generation in accepted),
            lanes=tuple(lane for _index, _item, lane, _priority, _generation in accepted),
            priorities=tuple(priority for _index, _item, _lane, priority, _generation in accepted),
            generations=tuple(generation for _index, _item, _lane, _priority, generation in accepted),
        )

    def submit_promoted_batch(
        self,
        items: tuple[PreparedIngressItem, ...],
        *,
        promotion_permits: tuple[Any, ...],
        generations: tuple[int, ...],
    ) -> tuple[PreparedSourceProduceResult, ...]:
        items, promotion_permits, generations = tuple(items), tuple(promotion_permits), tuple(generations)
        if not items or len(items) != len(promotion_permits) or len(items) != len(generations):
            raise ValueError("promoted batch requires non-empty matching item, permit, and generation tuples")
        results: list[PreparedSourceProduceResult | None] = [None] * len(items)
        accepted: list[tuple[int, PreparedIngressItem, ReadySourceLane, ReplayPriority, int]] = []
        for index, (item, permit, generation) in enumerate(
            zip(items, promotion_permits, generations)
        ):
            if not isinstance(item, PreparedIngressItem):
                raise TypeError("items must contain PreparedIngressItem")
            generation = self._generation(generation)
            if not self._prepared_source_supported(item):
                results[index] = PreparedSourceProduceResult.CAPABILITY_UNAVAILABLE
            elif item.disposition is not None and item.disposition.semantic_action is SemanticAction.COMMIT:
                accepted.append(
                    (index, item, ReadySourceLane.REPLAY, ReplayPriority.NORMAL, generation)
                )
            else:
                lane = self._ready_lane(item, promotion_permit=permit)
                if lane is None or lane[0] is not ReadySourceLane.REPLAY:
                    results[index] = PreparedSourceProduceResult.DROPPED_READY_OWNER
                else:
                    accepted.append((index, item, lane[0], lane[1], generation))
        for result, (index, _item, _lane, _priority, _generation) in zip(
            self._submit_admitted_batch(accepted), accepted
        ):
            results[index] = result
        return tuple(results)

    def submit_retained_commit_batch(
        self,
        items: tuple[PreparedIngressItem, ...],
        *,
        generations: tuple[int, ...],
    ) -> tuple[PreparedSourceProduceResult, ...]:
        items, generations = tuple(items), tuple(generations)
        if not items or len(items) != len(generations):
            raise ValueError("retained commit batch requires non-empty matching item and generation tuples")
        results: list[PreparedSourceProduceResult | None] = [None] * len(items)
        accepted: list[tuple[int, PreparedIngressItem, ReadySourceLane, ReplayPriority, int]] = []
        for index, (item, generation) in enumerate(zip(items, generations)):
            if not isinstance(item, PreparedIngressItem):
                raise TypeError("items must contain PreparedIngressItem")
            generation = self._generation(generation)
            if not self._prepared_source_supported(item):
                results[index] = PreparedSourceProduceResult.CAPABILITY_UNAVAILABLE
                continue
            lane = self._ready_lane(item)
            if lane is None or lane[0] is not ReadySourceLane.FRESH:
                results[index] = PreparedSourceProduceResult.DROPPED_READY_OWNER
            else:
                accepted.append((index, item, lane[0], lane[1], generation))
        for result, (index, item, _lane, _priority, _generation) in zip(
            self._submit_admitted_batch(accepted), accepted
        ):
            results[index] = result
            if result is PreparedSourceProduceResult.ENQUEUED:
                self._retained_commit_ready_items[item.packet.source] = item
        return tuple(results)

    def submit_promoted(
        self,
        item: PreparedIngressItem,
        *,
        promotion_permit: Any,
        generation: int = 0,
    ) -> PreparedSourceProduceResult:
        if not isinstance(item, PreparedIngressItem):
            raise TypeError("item must be PreparedIngressItem")
        generation = self._generation(generation)
        if not self._prepared_source_supported(item):
            return PreparedSourceProduceResult.CAPABILITY_UNAVAILABLE
        if item.disposition is not None and item.disposition.semantic_action is SemanticAction.COMMIT:
            return self._submit(
                item,
                lane=ReadySourceLane.REPLAY,
                priority=ReplayPriority.NORMAL,
                generation=generation,
                durable_retained=True,
            )
        lane = self._ready_lane(item, promotion_permit=promotion_permit)
        if lane is None or lane[0] is not ReadySourceLane.REPLAY:
            return PreparedSourceProduceResult.DROPPED_READY_OWNER
        return self._submit(
            item,
            lane=ReadySourceLane.REPLAY,
            priority=lane[1],
            generation=generation,
            durable_retained=True,
        )

    def submit_retained_commit(
        self,
        item: PreparedIngressItem,
        *,
        generation: int = 0,
    ) -> PreparedSourceProduceResult:
        if not isinstance(item, PreparedIngressItem):
            raise TypeError("item must be PreparedIngressItem")
        generation = self._generation(generation)
        if not self._prepared_source_supported(item):
            return PreparedSourceProduceResult.CAPABILITY_UNAVAILABLE
        lane = self._ready_lane(item)
        if lane is None or lane[0] is not ReadySourceLane.FRESH:
            raise RuntimeError("retained COMMIT no longer has fresh execution authority")
        result = self._submit(
            item,
            lane=ReadySourceLane.FRESH,
            priority=lane[1],
            generation=generation,
            durable_retained=True,
        )
        if result is PreparedSourceProduceResult.ENQUEUED:
            self._retained_commit_ready_items[item.packet.source] = item
        return result

    def complete_retained_commit_first_supervision(self, source: Any) -> bool:
        item = self._retained_commit_ready_items.get(source)
        if item is None:
            return False
        if not callable(self._retained_commit_clearer):
            raise RuntimeError("retained COMMIT completion requires an explicit clearer")
        self._retained_commit_clearer(item)
        self._retained_commit_ready_items.pop(source, None)
        return True

    def drain(self) -> PreparedRuntimeBridgeResult:
        submitted = enqueued = deferred = dropped = retained_not_resident = 0
        self._last_background_progress = False
        for _ in range(self._max_items_per_tick):
            item = self._deferred_item
            if item is None:
                item = self._consumer.try_poll()
                if item is None:
                    break
            else:
                self._deferred_item = None

            # COMMIT durability and its first-supervision obligation are
            # recorded by the pre-handoff sink.  The producer can construct a
            # neutral prepared depth plane only for RGB-authorized items whose
            # immutable policy disables every depth/geometry consumer.
            if not self._prepared_source_supported(item):
                continue

            lane = self._ready_lane(item)
            if lane is None:
                submitted += 1
                dropped += 1
                continue
            result = self._submit(
                item,
                lane=lane[0],
                priority=lane[1],
                durable_retained=item.disposition is not None,
            )
            submitted += 1
            if result is PreparedSourceProduceResult.ENQUEUED:
                enqueued += 1
                if (
                    lane[0] is ReadySourceLane.FRESH
                    and item.disposition is not None
                    and item.disposition.semantic_action is SemanticAction.COMMIT
                ):
                    self._retained_commit_ready_items[item.packet.source] = item
            elif result is PreparedSourceProduceResult.DEFERRED_POOL_EXHAUSTED:
                self._deferred_item = item
                deferred += 1
                break
            elif result is PreparedSourceProduceResult.RETAINED_NOT_RESIDENT:
                retained_not_resident += 1
            elif result is PreparedSourceProduceResult.DROPPED_READY_OWNER:
                dropped += 1
            else:
                raise RuntimeError(f"unknown prepared-source result: {result!r}")
        if self._cold_payload_spill_writer is not None:
            receipts = tuple(
                self._cold_payload_spill_writer(max_items=self._max_items_per_tick)
            )
            if receipts:
                with self._cold_payload_spill_lock:
                    self._cold_payload_spill_receipts.extend(receipts)
                self._last_background_progress = True
        return PreparedRuntimeBridgeResult(
            submitted, enqueued, deferred, dropped, retained_not_resident
        )

    def requeue_ready_only_sources(
        self,
        selection: ReadyOnlyTickSelection,
        *,
        ready_event: Any,
        lane_by_source: Mapping[Any, str] | None = None,
    ) -> None:
        self._ready_owner.requeue_batch(
            selection.leases,
            ready_event=ready_event,
            lane_by_source=lane_by_source,
        )

    def close(self) -> None:
        self._deferred_item = None
        while self._consumer.try_poll() is not None:
            pass
        self._producer.abandon_pending(reason="shutdown_ready_pending")


__all__ = ["PreparedRuntimeBridge", "PreparedRuntimeBridgeResult"]
