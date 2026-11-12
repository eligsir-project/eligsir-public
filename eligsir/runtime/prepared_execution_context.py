"""Durable control-only context for one prepared source execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from eligsir.contracts import SourceIdentity
from eligsir.contracts.control import (
    ControlLimits,
    ControlProposal,
    FreshnessMode,
    ObservationDisposition,
)
from eligsir.runtime.prepared_ingress import PreparedIngressControlToken
from eligsir.runtime.ready_source import ReadySourceLane, ReplayPriority


@dataclass(frozen=True, slots=True)
class PreparedControlAxes:
    """Resolved control/policy facts intentionally excluding map versions."""

    snapshot_id: int
    control_version: int
    freshness_mode: FreshnessMode
    limits: ControlLimits
    proposals: tuple[ControlProposal, ...]


@dataclass(frozen=True, slots=True)
class PreparedExecutionContext:
    """Source-local execution authority retained from terminal admission."""

    source: SourceIdentity
    control_token: PreparedIngressControlToken
    disposition: ObservationDisposition
    control_axes: PreparedControlAxes
    snapshot: Any
    lane: ReadySourceLane
    replay_priority: ReplayPriority
    promotion_permit: Any | None = None

    def __post_init__(self) -> None:
        if self.control_token.source != self.source:
            raise ValueError("prepared execution token source differs")
        if self.disposition.source != self.source:
            raise ValueError("prepared execution disposition source differs")


__all__ = ["PreparedControlAxes", "PreparedExecutionContext"]
