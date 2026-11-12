"""Joined attribution ledger for ingress overflow events (issue #836).

A single :class:`IngressAttributionLedger` instance is shared across both
queue boundaries of the ``ros2://live`` ingestion path (the inner
``ROS2LiveRGBDSource`` emit queue and the outer
``ROS2LiveStreamingFrameSource`` resident buffer), so a run's frame-loss
history can be read from one place instead of reconciling two independent,
differently-shaped counters by hand.

This does not replace the existing per-boundary counters
(``ROS2LiveRGBDSource.stats["queue_overflow_dropped_oldest"]``,
``ROS2LiveStreamingFrameSource.frames_dropped_by_inner_queue``,
``ROS2LiveStreamingFrameSource.bag_replay_report`` /
``BagReplayIntegrityReport``) -- those stay exactly as they are, for
diagnostics and backward compatibility. The ledger is an additional,
unified view over the same underlying events, keyed by the frame's real
drop-invariant ordinal (``RGBDFrame.received_seq`` /
``StreamingRGBDFrame.source_ordinal``) rather than a per-boundary arrival
index.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from eligsir.io.policies.ingress_overflow import IngressOverflowPolicy, LossSemantics


@dataclass
class OverflowEvent:
    """A single frame-ingestion pressure event at one queue boundary."""

    source_ordinal: Optional[int]  # RGBDFrame.received_seq / StreamingRGBDFrame.source_ordinal
    boundary: str  # e.g. "inner_ros2_live_emit_queue" / "outer_resident_buffer"
    policy_owner: str  # repr() of the IngressOverflowPolicy that decided this event
    action: str  # "spill" / "drop" / "backpressure_enqueued" / "enqueued"
    drop_reason: Optional[str]  # e.g. "spill_exhausted", "real_time_drop_oldest"
    queue_capacity: int
    queue_depth_before: int
    queue_depth_after: int
    cumulative_preserved: int
    cumulative_dropped: int
    cumulative_spilled: int
    mapping_result_valid: bool  # Run-validity state immediately after this event


@dataclass
class IngressAttributionLedger:
    """Cumulative, cross-boundary ledger of ingress overflow events.

    A single ledger instance is shared across boundaries that can be
    governed by DIFFERENT policies (e.g. the outer resident buffer is
    unconditionally NO_SILENT_LOSS while the inner emit queue's policy
    depends on source kind -- see
    ``eligsir.io.policies.ingress_overflow.resolve_ingress_policies``),
    so each recorded event carries its OWN acting policy rather than the
    ledger assuming one policy for everything it observes. :attr:`policy`
    is kept only as the ledger's "primary"/representative policy (the one
    it was constructed with -- conventionally the outer boundary's, since
    that is the boundary every ``ros2://live`` run always has), used as the
    default when a caller does not pass one explicitly.

    ``mapping_result_valid`` is a one-way latch: once a real loss occurs
    under a NO_SILENT_LOSS policy it stays ``False`` for the rest of the
    run (matching ``BagReplayIntegrityReport.mapping_result_valid`` and the
    CLAUDE.md standing of ``eval_valid == False`` /
    ``map_reinit_invalidated``). A legitimate drop under a REAL_TIME_DROP
    policy is recorded but does NOT flip this latch -- shedding frames
    under overload is expected, sanctioned behavior for a live sensor, not
    an integrity failure.
    """

    policy: IngressOverflowPolicy
    events: List[OverflowEvent] = field(default_factory=list)

    frames_preserved: int = 0  # enqueued directly, after backpressure, or spilled
    frames_spilled: int = 0
    frames_dropped: int = 0  # includes both legitimate REAL_TIME_DROP and real losses
    frames_lost: int = 0  # NO_SILENT_LOSS violations only -- subset of frames_dropped

    mapping_result_valid: bool = True

    def record_pressure_event(
        self,
        *,
        source_ordinal: Optional[int],
        boundary: str,
        action: str,
        drop_reason: Optional[str] = None,
        queue_capacity: int = 0,
        queue_depth_before: int = 0,
        queue_depth_after: int = 0,
        policy: Optional[IngressOverflowPolicy] = None,
    ) -> OverflowEvent:
        """Record one overflow-boundary event. ``action`` must be one of
        ``"spill"``, ``"drop"``, ``"backpressure_enqueued"``. ``policy`` is
        the policy that ACTUALLY governed this event -- defaults to
        :attr:`self.policy` when omitted, but callers reporting for a
        boundary whose policy differs from the ledger's primary one (e.g.
        the inner emit queue under a live-sensor ledger constructed around
        the outer, always-NO_SILENT_LOSS policy) must pass their own. Only
        a ``"drop"`` recorded under a NO_SILENT_LOSS policy flips
        ``mapping_result_valid`` -- a REAL_TIME_DROP policy's drops are
        sanctioned, not failures."""
        acting_policy = policy if policy is not None else self.policy
        if action == "spill":
            self.frames_spilled += 1
            self.frames_preserved += 1
        elif action == "backpressure_enqueued":
            self.frames_preserved += 1
        elif action == "drop":
            self.frames_dropped += 1
            if acting_policy.loss_semantics is LossSemantics.NO_SILENT_LOSS:
                self.frames_lost += 1
                self.mapping_result_valid = False
        else:
            raise ValueError(f"unknown ingress overflow action: {action!r}")

        event = OverflowEvent(
            source_ordinal=source_ordinal,
            boundary=boundary,
            policy_owner=repr(acting_policy),
            action=action,
            drop_reason=drop_reason,
            queue_capacity=queue_capacity,
            queue_depth_before=queue_depth_before,
            queue_depth_after=queue_depth_after,
            cumulative_preserved=self.frames_preserved,
            cumulative_dropped=self.frames_dropped,
            cumulative_spilled=self.frames_spilled,
            mapping_result_valid=self.mapping_result_valid,
        )
        self.events.append(event)
        return event

    def as_geometry_fields(self) -> dict:
        """Shaped for ``report.json["trajectory_coverage"]`` (top-level,
        matching ``BagReplayIntegrityReport.as_geometry_fields``'s naming
        convention -- issue #504). Not wired into the real per-run report
        by this change; see this repo's CLAUDE.md HARD CONSTRAINT on real
        GPU/bag integration runs being a separate, out-of-scope acceptance
        item for this leaf."""
        return {
            "ingress_frames_preserved": self.frames_preserved,
            "ingress_frames_spilled": self.frames_spilled,
            "ingress_frames_dropped": self.frames_dropped,
            "ingress_frames_lost": self.frames_lost,
            "ingress_mapping_result_valid": self.mapping_result_valid,
            "ingress_overflow_events": [
                {
                    "source_ordinal": e.source_ordinal,
                    "boundary": e.boundary,
                    "policy_owner": e.policy_owner,
                    "action": e.action,
                    "drop_reason": e.drop_reason,
                    "queue_capacity": e.queue_capacity,
                    "queue_depth_before": e.queue_depth_before,
                    "queue_depth_after": e.queue_depth_after,
                    "mapping_result_valid": e.mapping_result_valid,
                }
                for e in self.events
            ],
        }
