"""Canonical transport-free mapping session.

The session owns deterministic packet ordering, duplicate suppression and
tracking association. The default production mapping path materializes every
accepted packet exactly once into ``MappingFrame`` before the mapping sink
runs. Explicit ``prepared_mode=True`` instead hands the accepted source to a
bounded runtime-owned ingress queue and never invokes a mapper synchronously.
The only other mapperless mode is explicit ``ingress_only=True`` qualification;
it never claims that an accepted packet was mapped.
The session also records the control-plane replay digest for every source
identity/action; payload tensors are never hashed. A real mapper never receives
transport buffers directly: accepted packets are materialized exactly once into
``MappingFrame`` before the mapping sink runs.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, replace
from typing import Any, Callable, List, Optional, Tuple

import numpy as np

from eligsir.contracts.materialize import (
    MaterializationPolicy,
    TensorBackend,
    materialize_mapping_frame,
    numpy_backend,
)
from eligsir.contracts.control import ControlSnapshot, ObservationDisposition
from eligsir.contracts.runtime import (
    IngressAction,
    IngressEvent,
    MappingFrame,
    SensorPacket,
    TrackingObservation,
)
from eligsir.contracts.tracking import TrackingProvider
from eligsir.mapping.admission.observation_owner import ObservationOwner
from eligsir.runtime.deterministic_digest import (
    DeterministicReplayDigest,
    GLOBAL_REPLAY_DIGEST,
)
from eligsir.runtime.config import IngressTelemetrySnapshot
from eligsir.runtime.profiling.execution_trace import ExecutionTraceSink
from eligsir.runtime.prepared_ingress import (
    PreparedIngressItem,
    PreparedIngressControlToken,
    PreparedIngressDeferred,
    PreparedIngressMetrics,
    PreparedIngressOfferResult,
    PreparedIngressReservation,
    PreparedSourceIngressHandoff,
)

STAGE = "mapping_session"
OWNER = "MappingSession"


def _same_tracking_evidence(
    snapshot: TrackingObservation, observation: TrackingObservation
) -> bool:
    """Compare a snapshot-owned tracking copy without ndarray truth coercion."""

    if (
        snapshot.source != observation.source
        or snapshot.status is not observation.status
        or snapshot.provider != observation.provider
        or snapshot.confidence != observation.confidence
    ):
        return False
    if not _same_diagnostic(snapshot.diagnostics, observation.diagnostics):
        return False
    if (snapshot.pose is None) != (observation.pose is None):
        return False
    if snapshot.pose is not None and not np.array_equal(
        snapshot.pose.matrix, observation.pose.matrix
    ):
        return False
    if (snapshot.covariance is None) != (observation.covariance is None):
        return False
    return snapshot.covariance is None or np.array_equal(
        snapshot.covariance, observation.covariance
    )


def _same_diagnostic(left: Any, right: Any) -> bool:
    if isinstance(left, dict) or hasattr(left, "items"):
        if not (isinstance(right, dict) or hasattr(right, "items")):
            return False
        left_items = dict(left)
        right_items = dict(right)
        return left_items.keys() == right_items.keys() and all(
            _same_diagnostic(left_items[key], right_items[key]) for key in left_items
        )
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        return np.array_equal(np.asarray(left), np.asarray(right))
    if isinstance(left, (tuple, list)) or isinstance(right, (tuple, list)):
        return len(left) == len(right) and all(
            _same_diagnostic(a, b) for a, b in zip(left, right)
        )
    if isinstance(left, (set, frozenset)) or isinstance(right, (set, frozenset)):
        return set(left) == set(right)
    return bool(left == right)


class MappingSessionClosed(RuntimeError):
    """Raised when a packet is submitted after session shutdown."""

    pass


@dataclass(frozen=True, slots=True)
class SubmitOutcome:
    """Canonical disposition, evidence, and optional materialized frame."""

    sequence: int
    accepted: bool
    observation: Optional[TrackingObservation]
    event: IngressEvent
    frame: Optional[MappingFrame] = None
    disposition: Optional[ObservationDisposition] = None


ControlSnapshotProvider = Callable[
    [SensorPacket, TrackingObservation, IngressTelemetrySnapshot], ControlSnapshot
]
DispositionSink = Callable[[ObservationDisposition], Any]


class MappingSession:
    """One synchronous mapping runtime with one materialization boundary.

    The compatibility/default path has a mapping sink and one synchronous
    materialization boundary. Explicit ``prepared_mode=True`` is the staged
    asynchronous ingress path: it requires a bounded handoff, may omit the
    mapper, and deliberately does not materialize or increment ``mapped_count``.
    Mapperless operation outside prepared mode is available only through
    explicit ``ingress_only=True`` qualification.
    """

    def __init__(
        self,
        tracking_provider: TrackingProvider,
        *,
        mapper: Optional[Callable[[MappingFrame, TrackingObservation], Any]] = None,
        ingress_only: bool = False,
        require_tracking: bool = True,
        max_events: int = 4096,
        materialization_policy: Optional[MaterializationPolicy] = None,
        materialization_backend: Optional[TensorBackend] = None,
        deterministic_digest: Optional[DeterministicReplayDigest] = None,
        observation_owner: Optional[ObservationOwner] = None,
        control_snapshot_provider: Optional[ControlSnapshotProvider] = None,
        disposition_sink: Optional[DispositionSink] = None,
        ingress_telemetry_provider: Optional[
            Callable[[SensorPacket, TrackingObservation], IngressTelemetrySnapshot]
        ] = None,
        packet_capture: Optional[Callable[[SensorPacket], SensorPacket]] = None,
        terminal_packet_sink: Optional[Callable[[SensorPacket, SubmitOutcome], Any]] = None,
        source_consumed_observer: Optional[Callable[[SensorPacket], Any]] = None,
        execution_trace_sink: Optional[ExecutionTraceSink] = None,
        prepared_mode: bool = False,
        prepared_ingress_handoff: Optional[PreparedSourceIngressHandoff] = None,
    ) -> None:
        if max_events <= 0:
            raise ValueError("max_events must be positive")
        self._ingress_only = bool(ingress_only)
        if mapper is None and not self._ingress_only and not prepared_mode:
            raise ValueError(
                "MappingSession requires a mapping sink; use ingress_only=True "
                "only for explicit ingress qualification"
            )
        if mapper is not None and self._ingress_only:
            raise ValueError("ingress_only=True cannot be combined with a mapping sink")
        if prepared_mode and self._ingress_only:
            raise ValueError("prepared_mode=True cannot be combined with ingress_only=True")
        if prepared_mode and prepared_ingress_handoff is None:
            raise ValueError(
                "prepared_mode=True requires prepared_ingress_handoff"
            )
        if not prepared_mode and prepared_ingress_handoff is not None:
            raise ValueError(
                "prepared_ingress_handoff requires prepared_mode=True"
            )
        if prepared_ingress_handoff is not None and not isinstance(
            prepared_ingress_handoff, PreparedSourceIngressHandoff
        ):
            raise TypeError(
                "prepared_ingress_handoff must be PreparedSourceIngressHandoff"
            )
        if (observation_owner is None) != (control_snapshot_provider is None):
            raise ValueError(
                "observation_owner and control_snapshot_provider must be injected together"
            )
        if disposition_sink is not None and observation_owner is None:
            raise ValueError("disposition_sink requires an observation_owner")
        if ingress_telemetry_provider is not None and not callable(ingress_telemetry_provider):
            raise TypeError("ingress_telemetry_provider must be callable when provided")
        if packet_capture is not None and not callable(packet_capture):
            raise TypeError("packet_capture must be callable when provided")
        if terminal_packet_sink is not None and not callable(terminal_packet_sink):
            raise TypeError("terminal_packet_sink must be callable when provided")
        if source_consumed_observer is not None and not callable(source_consumed_observer):
            raise TypeError("source_consumed_observer must be callable when provided")
        self._tracking_provider = tracking_provider
        self._mapper = mapper
        self._require_tracking = bool(require_tracking)
        self._max_events = int(max_events)
        self._materialization_policy = materialization_policy or MaterializationPolicy()
        self._materialization_backend = materialization_backend or numpy_backend()
        self._digest = deterministic_digest or GLOBAL_REPLAY_DIGEST
        self._observation_owner = observation_owner
        self._control_snapshot_provider = control_snapshot_provider
        self._disposition_sink = disposition_sink
        self._ingress_telemetry_provider = ingress_telemetry_provider
        self._packet_capture = packet_capture
        self._terminal_packet_sink = terminal_packet_sink
        self._source_consumed_observer = source_consumed_observer
        self._execution_trace_sink = execution_trace_sink
        self._prepared_mode = bool(prepared_mode)
        self._prepared_ingress_handoff = prepared_ingress_handoff
        self._events: List[IngressEvent] = []
        self._terminal_fates: OrderedDict[object, tuple[str, SubmitOutcome]] = OrderedDict()
        self._sequence = 0
        self._submitted_count = 0
        self._accepted_count = 0
        self._mapped_count = 0
        self._rejected_count = 0
        self._flush_count = 0
        self._closed = False
        self._last_order_key = None

    @property
    def closed(self) -> bool:
        """Return whether this session has released its tracking provider."""

        return self._closed

    @property
    def submitted_count(self) -> int:
        """Return the number of packets presented to the session."""

        return self._submitted_count

    @property
    def accepted_count(self) -> int:
        return self._accepted_count

    @property
    def mapped_count(self) -> int:
        """Return the number of packets accepted for mapping."""

        return self._mapped_count

    @property
    def ingress_only(self) -> bool:
        return self._ingress_only

    @property
    def has_mapper(self) -> bool:
        return self._mapper is not None

    @property
    def prepared_mode(self) -> bool:
        """Return whether accepted sources use the asynchronous handoff."""

        return self._prepared_mode

    @property
    def prepared_ingress_handoff(self) -> Optional[PreparedSourceIngressHandoff]:
        """Return the bounded runtime-side handoff, when prepared mode is active."""

        return self._prepared_ingress_handoff

    @property
    def rejected_count(self) -> int:
        """Return the number of duplicate, invalid, or unsafe packets."""

        return self._rejected_count

    @property
    def deterministic_digest(self) -> DeterministicReplayDigest:
        """Return the digest that owns this session's ingress events."""

        return self._digest

    def events(self) -> Tuple[IngressEvent, ...]:
        """Return the bounded immutable ingress-event history."""

        return tuple(self._events)

    def submit(self, packet: SensorPacket) -> SubmitOutcome:
        """Validate, track, materialize, and optionally map one packet."""

        if self._closed:
            raise MappingSessionClosed(
                f"submit() after close(); packet {packet.source.source_id!r} "
                f"ordinal {packet.source.source_ordinal}"
            )

        source_packet = packet
        original_source = packet.source
        if self._packet_capture is not None:
            packet = self._packet_capture(packet)
            if not isinstance(packet, SensorPacket):
                raise TypeError("packet_capture must return SensorPacket")
            if packet.source != original_source:
                raise ValueError("packet_capture must preserve source identity")

        defer_consumed_observer = bool(
            self._prepared_mode
            and self._prepared_ingress_handoff is not None
            and self._prepared_ingress_handoff.has_retention_preflight
        )
        if defer_consumed_observer and not self._prepared_ingress_handoff.preflight_retention(packet):
            # Capacity is packet-only: do not advance a tracker, sequence, event
            # history, source-consumed callback, telemetry, or control decision.
            raise PreparedIngressDeferred(
                "prepared retention capacity is unavailable; source remains pending"
            )
        if self._source_consumed_observer is not None and not defer_consumed_observer:
            self._source_consumed_observer(source_packet)

        trace_started_ns = (
            self._execution_trace_sink.begin()
            if self._execution_trace_sink is not None
            else None
        )
        self._submitted_count += 1
        prior_fate = self._terminal_fates.get(packet.source)
        if prior_fate is not None:
            fingerprint, outcome = prior_fate
            if fingerprint != packet.metadata_fingerprint():
                raise ValueError(
                    "duplicate source identity metadata conflicts with its terminal fate"
                )
            self._terminal_fates.move_to_end(packet.source)
            return self._record_terminal(
                packet,
                outcome,
                trace_started_ns=trace_started_ns,
                duplicate=True,
            )
        sequence = self._sequence
        self._sequence += 1
        order_key = packet.source.ordering_key()
        if self._last_order_key is not None:
            if order_key == self._last_order_key:
                self._rejected_count += 1
                event = self._emit(
                    packet,
                    IngressAction.REJECT,
                    "duplicate source identity suppressed idempotently",
                    sequence=sequence,
                    order_key=order_key,
                )
                return self._record_terminal(
                    packet,
                    SubmitOutcome(sequence, False, None, event),
                    trace_started_ns=trace_started_ns,
                )
            if order_key < self._last_order_key:
                self._rejected_count += 1
                event = self._emit(
                    packet,
                    IngressAction.FAIL,
                    "source identity arrived out of canonical order",
                    sequence=sequence,
                    order_key=order_key,
                )
                return self._record_terminal(
                    packet,
                    SubmitOutcome(sequence, False, None, event),
                    trace_started_ns=trace_started_ns,
                )
        self._last_order_key = order_key

        observation = self._tracking_provider.observe(packet)
        if observation.source != packet.source:
            self._rejected_count += 1
            event = self._emit(
                packet,
                IngressAction.FAIL,
                "tracking provider returned a foreign source identity",
                sequence=sequence,
                order_key=order_key,
            )
            return self._record_terminal(
                packet,
                SubmitOutcome(sequence, False, observation, event),
                trace_started_ns=trace_started_ns,
            )

        ingress_telemetry = IngressTelemetrySnapshot()
        if self._ingress_telemetry_provider is not None:
            ingress_telemetry = self._ingress_telemetry_provider(packet, observation)
            if not isinstance(ingress_telemetry, IngressTelemetrySnapshot):
                raise TypeError(
                    "ingress_telemetry_provider must return IngressTelemetrySnapshot"
                )

        if self._require_tracking and not observation.usable:
            self._rejected_count += 1
            event = self._emit(
                packet,
                IngressAction.REJECT,
                f"tracking status {observation.status.value} is not usable",
                sequence=sequence,
                order_key=order_key,
            )
            return self._record_terminal(
                packet,
                SubmitOutcome(sequence, False, observation, event),
                ingress_telemetry=ingress_telemetry,
                trace_started_ns=trace_started_ns,
            )

        tracked_packet = packet if packet.tracking is observation else replace(
            packet, tracking=observation
        )
        prepared_reservation: PreparedIngressReservation | None = None
        if self._prepared_mode:
            assert self._prepared_ingress_handoff is not None
            if defer_consumed_observer and self._source_consumed_observer is not None:
                self._source_consumed_observer(source_packet)
            reservation_or_status = self._prepared_ingress_handoff.reserve(
                PreparedIngressItem(sequence, tracked_packet, observation)
            )
            if isinstance(reservation_or_status, PreparedIngressOfferResult):
                self._rejected_count += 1
                event = self._emit(
                    packet,
                    IngressAction.REJECT,
                    f"prepared ingress handoff {reservation_or_status.value}",
                    sequence=sequence,
                    order_key=order_key,
                )
                return self._record_terminal(
                    packet,
                    SubmitOutcome(sequence, False, observation, event),
                    ingress_telemetry=ingress_telemetry,
                    trace_started_ns=trace_started_ns,
                )
            prepared_reservation = reservation_or_status

        self._accepted_count += 1
        if self._ingress_only:
            event = self._emit(
                packet,
                IngressAction.DELIVER,
                "accepted ingress-only (not mapped)",
                sequence=sequence,
                order_key=order_key,
            )
            return self._record_terminal(
                packet,
                SubmitOutcome(sequence, True, observation, event),
                ingress_telemetry=ingress_telemetry,
                trace_started_ns=trace_started_ns,
            )

        disposition = None
        if self._observation_owner is not None:
            assert self._control_snapshot_provider is not None
            try:
                snapshot = self._control_snapshot_provider(
                    packet, observation, ingress_telemetry
                )
                if not isinstance(snapshot, ControlSnapshot):
                    raise TypeError("control_snapshot_provider must return ControlSnapshot")
                snapshot_evidence = {fact.name: fact for fact in snapshot.evidence}
                missing_or_changed = tuple(
                    fact.name
                    for fact in ingress_telemetry.evidence
                    if snapshot_evidence.get(fact.name) != fact
                )
                if missing_or_changed:
                    raise ValueError(
                        "control snapshot provider must include immutable ingress telemetry "
                        "during construction: " + ", ".join(missing_or_changed)
                    )
                if snapshot.source != packet.source or not _same_tracking_evidence(
                    snapshot.tracking, observation
                ):
                    raise ValueError(
                        "control snapshot must preserve the submitted source and tracking observation"
                    )
                disposition = self._observation_owner.decide(snapshot)
                if self._disposition_sink is not None:
                    self._disposition_sink(disposition)
                permit = self._observation_owner.permit(disposition)
            except BaseException:
                if prepared_reservation is not None:
                    prepared_reservation.cancel()
                    prepared_reservation = None
                raise
            if permit is None:
                if prepared_reservation is not None:
                    prepared_reservation.cancel()
                    prepared_reservation = None
                event = self._emit(
                    packet,
                    IngressAction.DELIVER,
                    f"observed; semantic disposition {disposition.kind.value}: "
                    f"{disposition.reason}",
                    sequence=sequence,
                    order_key=order_key,
                )
                return self._record_terminal(
                    packet,
                    SubmitOutcome(
                        sequence,
                        True,
                        observation,
                        event,
                        disposition=disposition,
                    ),
                    ingress_telemetry=ingress_telemetry,
                    trace_started_ns=trace_started_ns,
                )

        if self._prepared_mode:
            assert self._prepared_ingress_handoff is not None
            assert prepared_reservation is not None
            # Tracking observations can contain ndarray-backed poses; identity
            # is the only safe no-copy check here (dataclass equality would
            # attempt ambiguous ndarray truth evaluation).
            try:
                queued = prepared_reservation.commit(
                    PreparedIngressItem(
                        sequence=sequence,
                        packet=tracked_packet,
                        observation=observation,
                        disposition=disposition,
                        control_token=PreparedIngressControlToken(
                            sequence=sequence,
                            source=packet.source,
                            snapshot_id=(
                                None if disposition is None else disposition.snapshot_id
                            ),
                            disposition=disposition,
                        ),
                    )
                )
            except BaseException:
                prepared_reservation.cancel()
                raise
            prepared_reservation = None
            assert queued is PreparedIngressOfferResult.ENQUEUED
            event = self._emit(
                packet,
                IngressAction.DELIVER,
                "accepted into prepared ingress handoff",
                sequence=sequence,
                order_key=order_key,
            )
            return self._record_terminal(
                packet,
                SubmitOutcome(
                    sequence,
                    True,
                    observation,
                    event,
                    disposition=disposition,
                ),
                ingress_telemetry=ingress_telemetry,
                trace_started_ns=trace_started_ns,
            )

        # Construction rejects a missing mapper outside explicit ingress-only
        # qualification, so this is a production invariant rather than an
        # optional best-effort callback.
        assert self._mapper is not None
        frame = materialize_mapping_frame(
            tracked_packet,
            self._materialization_policy,
            backend=self._materialization_backend,
        )
        self._mapper(frame, observation)
        self._mapped_count += 1
        event = self._emit(
            packet,
            IngressAction.DELIVER,
            "materialized and mapped",
            sequence=sequence,
            order_key=order_key,
        )
        return self._record_terminal(
            packet,
            SubmitOutcome(
                sequence,
                True,
                observation,
                event,
                frame,
                disposition,
            ),
            ingress_telemetry=ingress_telemetry,
            trace_started_ns=trace_started_ns,
        )

    def _record_terminal(
        self,
        packet: SensorPacket,
        outcome: SubmitOutcome,
        *,
        ingress_telemetry: Optional[IngressTelemetrySnapshot] = None,
        trace_started_ns: Optional[int] = None,
        duplicate: bool = False,
    ) -> SubmitOutcome:
        """Persist exactly one terminal fate for each full source identity."""

        if (
            not duplicate
            and not outcome.accepted
            and outcome.disposition is None
            and self._terminal_packet_sink is not None
        ):
            self._terminal_packet_sink(packet, outcome)

        fingerprint = packet.metadata_fingerprint()
        prior = self._terminal_fates.get(packet.source)
        if prior is not None and prior != (fingerprint, outcome):
            raise RuntimeError("source identity received conflicting terminal fates")
        self._terminal_fates[packet.source] = (fingerprint, outcome)
        self._terminal_fates.move_to_end(packet.source)
        if len(self._terminal_fates) > self._max_events:
            self._terminal_fates.popitem(last=False)
        if self._execution_trace_sink is not None:
            assert trace_started_ns is not None
            self._execution_trace_sink.record_terminal(
                source=packet.source,
                action=outcome.event.action.value,
                reason=outcome.event.reason,
                evidence=() if ingress_telemetry is None else ingress_telemetry.evidence,
                started_monotonic_ns=trace_started_ns,
                duplicate=duplicate,
            )
        return outcome

    def flush(self) -> int:
        """Record a synchronization boundary and return the mapped count."""

        self._flush_count += 1
        return self._mapped_count

    def close(self) -> None:
        """Idempotently close the underlying tracking provider."""

        if self._closed:
            return
        self._closed = True
        if self._prepared_ingress_handoff is not None:
            self._prepared_ingress_handoff.close()
        self._tracking_provider.close()

    def __enter__(self) -> "MappingSession":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _emit(
        self,
        packet: SensorPacket,
        action: IngressAction,
        reason: str,
        *,
        sequence: int,
        order_key,
    ) -> IngressEvent:
        event = IngressEvent(
            source=packet.source,
            stage=STAGE,
            owner=OWNER,
            action=action,
            reason=reason,
            details={"sequence": sequence, "order_key": order_key},
        )
        self._events.append(event)
        if len(self._events) > self._max_events:
            del self._events[: len(self._events) - self._max_events]
        self._digest.record(
            "ingress",
            {
                "sequence": int(sequence),
                "source_id": packet.source.source_id,
                "stream_epoch": int(packet.source.stream_epoch),
                "source_ordinal": int(packet.source.source_ordinal),
                "timestamp_ns": int(packet.source.timestamp_ns),
                "order_key": tuple(order_key),
                "action": action.value,
                "reason": reason,
            },
        )
        return event


__all__ = [
    "STAGE",
    "OWNER",
    "ControlSnapshotProvider",
    "DispositionSink",
    "MappingSession",
    "MappingSessionClosed",
    "PreparedIngressItem",
    "PreparedIngressControlToken",
    "PreparedIngressMetrics",
    "PreparedIngressOfferResult",
    "PreparedSourceIngressHandoff",
    "SubmitOutcome",
]
