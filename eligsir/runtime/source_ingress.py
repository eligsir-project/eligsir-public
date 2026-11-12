"""Finite paced-source ingress accounting and historical report compatibility."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class IngestionValueConfig:
    """Parsed legacy ingress configuration.

    Finite paced replay accepts only FIFO configuration; its producer rejects
    ``enabled`` value admission before it can discard a catalogue record.
    Keep the parsed fields so existing command-line configurations fail closed
    at that boundary instead of becoming silently ignored.
    """

    enabled: bool = False
    history_frames: int = 64
    rotation_weight_m_per_rad: float = 0.25
    pressure_exponent: float = 2.0
    threshold_scale: float = 1.0
    lag_budget_s: float = 0.0

    @classmethod
    def from_args(cls, args: Any) -> "IngestionValueConfig":
        mode = str(
            getattr(args, "causal_ingestion_policy", "fifo_drop_newest")
            or "fifo_drop_newest"
        )
        if mode not in ("fifo_drop_newest", "value_ranked"):
            raise ValueError(
                "causal_ingestion_policy must be "
                f"'fifo_drop_newest' or 'value_ranked', got {mode!r}"
            )
        return cls(
            enabled=mode == "value_ranked",
            history_frames=int(getattr(args, "causal_ingestion_history_frames", 64)),
            rotation_weight_m_per_rad=float(
                getattr(args, "causal_ingestion_rotation_weight", 0.25)
            ),
            pressure_exponent=float(
                getattr(args, "causal_ingestion_pressure_exponent", 2.0)
            ),
            threshold_scale=float(
                getattr(args, "causal_ingestion_threshold_scale", 1.0)
            ),
            lag_budget_s=float(getattr(args, "causal_ingestion_lag_budget_s", 0.0)),
        )

class CausalIngressRecorder:
    """Count finite-catalogue frames admitted by paced replay.

    Preservation has no causal-source drop boundary: every offered frame is
    decoded and published exactly once unless cancellation or an explicit
    error terminates the producer.
    """

    def __init__(self) -> None:
        self._admitted = 0

    def record_admitted(self, _record: Any) -> None:
        self._admitted += 1

    def as_report_block(self) -> dict[str, Any]:
        return {
            "ingress_frames_dropped": 0,
            "ingress_dropped_at_arrival": 0,
            "ingress_dropped_at_enqueue": 0,
            "ingress_frames_admitted": self._admitted,
            "ingress_dropped_by_boundary": {},
            "ingress_dropped_by_reason": {},
            "ingress_overflow_events": [],
            "ingress_events_recorded_total": 0,
            "ingress_events_truncated": False,
            "ingress_information_coverage": 1.0,
            "ingress_mapping_result_valid": True,
        }


def source_ingress_warning(block: Optional[dict]) -> Optional[str]:
    """Return the retained warning for a historical lossy report block."""

    if not block:
        return None
    dropped = int(block.get("ingress_frames_dropped", 0) or 0)
    if dropped <= 0:
        return None
    return (
        "[source_ingress] WARNING: the causal source buffer dropped "
        f"{dropped} frame(s) before the mapper ever saw them "
        f"(arrival={block.get('ingress_dropped_at_arrival')}, "
        f"enqueue={block.get('ingress_dropped_at_enqueue')}, "
        f"policy={block.get('ingestion_policy')}, "
        f"information_coverage={block.get('ingress_information_coverage')}, "
        f"max_buffer_lag_s={block.get('ingress_max_observed_buffer_lag_s')}). "
        "This run did NOT see its full source stream -- see "
        'report.json["source_ingress"].'
    )


__all__ = [
    "CausalIngressRecorder",
    "IngestionValueConfig",
    "source_ingress_warning",
]
