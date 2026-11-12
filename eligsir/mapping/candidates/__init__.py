"""Prepared-candidate pipeline: raw-frame -> tracked-frame -> prepared
candidate data contracts and CPU-side preparation/prefilter logic.

Part of PR W3-03 (see
``plans/2026-08-06-wave3-live-mapping-runtime-evidence-strategy-implementation-plan.md``
section 8). This package is pure infrastructure: nothing here is wired
into the default live/streaming path until the coordinator's bounded
queue pipeline is explicitly enabled (``--streaming_bounded_queue_pipeline``,
off by default). See ``eligsir/runtime/workers/frame_preparation_worker.py``
for the background thread that turns raw frames into
:class:`~eligsir.mapping.candidates.contracts.PreparedFrameCandidate`
instances, and ``eligsir/mapping/coordinator.py`` for how the
coordinator wires the queues together behind the feature flag.
"""

from __future__ import annotations

from .contracts import (
    InputEvidence,
    PreparedDepthRef,
    PreparedFrameCandidate,
    PreparedImageRef,
    RawFramePacket,
    TrackedFramePacket,
    TrackingState,
    UndecodedBufferRef,
)

__all__ = [
    "InputEvidence",
    "PreparedDepthRef",
    "PreparedFrameCandidate",
    "RawFramePacket",
    "PreparedImageRef",
    "TrackedFramePacket",
    "TrackingState",
    "UndecodedBufferRef",
]
