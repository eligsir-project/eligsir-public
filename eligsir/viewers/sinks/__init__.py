"""Canonical preview sink contract (issue #570).

No concrete backend sink lives here. Rerun (#573), RVizSplat/ROS (#574),
Viser (#572) and the recorder (#575) each own their own.
"""
from __future__ import annotations

from eligsir.viewers.sinks.base import (  # noqa: F401
    DeliveryStats,
    NullPreviewSink,
    PreviewSink,
    SinkContractError,
    deliver_pending,
)

__all__ = [
    "PreviewSink",
    "NullPreviewSink",
    "SinkContractError",
    "DeliveryStats",
    "deliver_pending",
]
