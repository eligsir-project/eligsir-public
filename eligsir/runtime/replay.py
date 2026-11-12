"""Deterministic non-ROS ingress/tracking replay qualification helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, Optional

import numpy as np

from eligsir.contracts.runtime import (
    BufferOwnership,
    BufferRef,
    IngressAction,
    PixelReloadToken,
    SensorPacket,
    SourceIdentity,
    TrackingObservation,
)
from eligsir.contracts.tracking import TrackingProvider
from eligsir.contracts.types import CameraIntrinsics
from eligsir.runtime.deterministic_digest import DeterministicReplayDigest
from eligsir.runtime.session import MappingSession
from eligsir.tracking.providers import observation_from_pose_matrix

_NS = 1_000_000_000


@dataclass(frozen=True, slots=True)
class ReplaySummary:
    """Terminal replay counts and deterministic control-plane digest."""

    source_id: str
    submitted: int
    accepted: int
    mapped: int
    rejected: int
    action_counts: Dict[str, int]
    deterministic_replay: Dict[str, Any]

    def as_dict(self) -> Dict[str, Any]:
        """Return a detached JSON-compatible summary mapping."""

        return {
            "source_id": self.source_id,
            "submitted": self.submitted,
            "accepted": self.accepted,
            "mapped": self.mapped,
            "rejected": self.rejected,
            "action_counts": dict(self.action_counts),
            "deterministic_replay": dict(self.deterministic_replay),
        }


def streaming_frame_source_identity(
    frame: Any, *, source_id: str, stream_epoch: int = 0
) -> SourceIdentity:
    """Share the packet identity with report-only source selection, without decoding."""
    ordinal = frame.source_ordinal
    if ordinal is None:
        ordinal = frame.index
    return SourceIdentity(
        source_id=source_id,
        stream_epoch=int(stream_epoch),
        source_ordinal=int(ordinal),
        timestamp_ns=(
            int(frame.color_ts_ns)
            if getattr(frame, "color_ts_ns", None) is not None
            else int(round(float(frame.timestamp) * _NS))
        ),
    )


def packet_from_streaming_frame(
    frame: Any,
    *,
    source_id: str,
    stream_epoch: int = 0,
    provider: str = "dataset_pose",
    rgb: Optional[np.ndarray] = None,
    depth: Optional[np.ndarray] = None,
) -> SensorPacket:
    """Adapt one decoded dataset observation to the canonical packet contract.

    ``depth`` is metric float32 when supplied. If either payload is omitted, an
    already-decoded in-frame array is reused; no synthetic image/depth is invented.
    File-backed decoding belongs to :func:`packets_from_frame_source`.
    """
    depth_fx = frame.depth_fx if frame.depth_fx is not None else frame.fx
    depth_fy = frame.depth_fy if frame.depth_fy is not None else frame.fy
    depth_cx = frame.depth_cx if frame.depth_cx is not None else frame.cx
    depth_cy = frame.depth_cy if frame.depth_cy is not None else frame.cy
    depth_width = frame.depth_width if frame.depth_width is not None else frame.width
    depth_height = (
        frame.depth_height if frame.depth_height is not None else frame.height
    )
    intrinsics = CameraIntrinsics(
        fx=float(frame.fx),
        fy=float(frame.fy),
        cx=float(frame.cx),
        cy=float(frame.cy),
        width=int(frame.width),
        height=int(frame.height),
        depth_fx=float(depth_fx),
        depth_fy=float(depth_fy),
        depth_cx=float(depth_cx),
        depth_cy=float(depth_cy),
        depth_width=int(depth_width),
        depth_height=int(depth_height),
        depth_scale=1.0,
    )
    source = streaming_frame_source_identity(
        frame, source_id=source_id, stream_epoch=stream_epoch
    )
    if rgb is None:
        rgb = getattr(frame, "_rgb_np", None)
    if depth is None:
        depth = getattr(frame, "_depth_np", None)

    tracking: Optional[TrackingObservation] = None
    use_mapping_pose = bool(getattr(frame, "_mapping_pose_for_training", False))
    c2w = getattr(frame, "_mapping_c2w", None) if use_mapping_pose else None
    if c2w is None:
        c2w = getattr(frame, "c2w", None)
    if c2w is not None:
        diagnostics = dict(getattr(frame, "_motion_depth_confidence_diagnostics", {}) or {})
        if use_mapping_pose:
            diagnostics = {
                **(getattr(frame, "_pose_time_diagnostics", {}) or {}),
                **diagnostics,
            }
        tracking = observation_from_pose_matrix(
            source,
            c2w,
            provider=provider,
            diagnostics=diagnostics or None,
        )
    rgb_ref = None
    if rgb is not None:
        rgb_arr = np.ascontiguousarray(np.asarray(rgb, dtype=np.uint8))
        rgb_ref = BufferRef.from_numpy(
            rgb_arr, encoding="rgb8", ownership=BufferOwnership.OWNED
        )
    if rgb_ref is None:
        raise ValueError("canonical RGB-D replay packet requires a real RGB payload")

    depth_ref = None
    if depth is not None:
        depth_arr = np.ascontiguousarray(np.asarray(depth, dtype=np.float32))
        depth_ref = BufferRef.from_numpy(
            depth_arr,
            encoding="32FC1",
            units="m",
            ownership=BufferOwnership.OWNED,
        )

    rgb_path = str(getattr(frame, "rgb_path", "") or "")
    depth_path = str(getattr(frame, "depth_path", "") or "")
    reload_token = None
    if rgb_path:
        reload_token = PixelReloadToken(
            provider="rgbd_files_v1",
            locators=(rgb_path, depth_path),
            parameters=(("depth_scale", repr(float(frame.depth_scale))),),
        )

    return SensorPacket(
        source=source,
        rgb=rgb_ref,
        depth=depth_ref,
        intrinsics=intrinsics,
        tracking=tracking,
        reload_token=reload_token,
    )


def packets_from_frame_source(
    frame_source: Iterable[Any],
    *,
    source_id: str,
    stream_epoch: int = 0,
    provider: str = "dataset_pose",
    max_frames: Optional[int] = None,
) -> Iterator[SensorPacket]:
    """Decode an ordered dataset source into real canonical packets.

    This is the non-ROS producer boundary. RGB is decoded once to HWC uint8 and
    raw depth is converted once to metric float32 using the source-declared
    ``depth_scale``. MappingSession therefore receives the same representation
    contract as the ROS producer and never sees placeholder pixels.
    """
    from eligsir.io.sources.decode_cache import load_frame_depth_np, load_frame_rgb
    from eligsir.utils.rgbd_frames import depth_to_meters

    for count, frame in enumerate(frame_source):
        if max_frames is not None and count >= max_frames:
            return
        rgb = np.asarray(load_frame_rgb(frame), dtype=np.uint8)
        raw_depth = load_frame_depth_np(frame)
        depth = None
        if raw_depth is not None:
            depth = depth_to_meters(raw_depth, float(frame.depth_scale)).astype(
                np.float32, copy=False
            )
        yield packet_from_streaming_frame(
            frame,
            source_id=source_id,
            stream_epoch=stream_epoch,
            provider=provider,
            rgb=rgb,
            depth=depth,
        )


def _summary(session: Any, source_id: str, counts: Dict[str, int]) -> ReplaySummary:
    digest = session.deterministic_digest
    digest.record(
        "replay_summary",
        {
            "source_id": source_id,
            "submitted": session.submitted_count,
            "accepted": session.accepted_count,
            "mapped": session.mapped_count,
            "rejected": session.rejected_count,
            "action_counts": counts,
        },
    )
    return ReplaySummary(
        source_id=source_id,
        submitted=int(session.submitted_count),
        accepted=int(session.accepted_count),
        mapped=int(session.mapped_count),
        rejected=int(session.rejected_count),
        action_counts=counts,
        deterministic_replay=digest.snapshot(),
    )


def run_replay_session(
    packets: Iterable[SensorPacket],
    provider: TrackingProvider,
    *,
    source_id: str = "replay",
    require_tracking: bool = True,
    max_events: int = 4096,
) -> ReplaySummary:
    """Ingress/tracking qualification only; deliberately performs no mapping."""
    counts: Dict[str, int] = {action.value: 0 for action in IngressAction}
    digest = DeterministicReplayDigest()
    with MappingSession(
        provider,
        ingress_only=True,
        require_tracking=require_tracking,
        max_events=max_events,
        deterministic_digest=digest,
    ) as session:
        for packet in packets:
            outcome = session.submit(packet)
            counts[outcome.event.action.value] += 1
        return _summary(session, source_id, counts)


__all__ = [
    "ReplaySummary",
    "packet_from_streaming_frame",
    "packets_from_frame_source",
    "run_replay_session",
]
