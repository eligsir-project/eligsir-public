"""Trace recorder and replayer for deterministic experiment capsules (issue #633).

This module records and replays the exact observation/evaluation substrate seen by
EliGSiR. It captures source frames, frontend pose/tracking, admission decisions,
and optional replay schedules as append-only JSONL event logs, enabling deterministic
replay with full causal semantics.

A **trace** is an immutable record of what happened at runtime. A **replayer** is
a provider that serves recorded events back to the mapper, optionally enforcing
causal (as-of) semantics or oracle/final-pose modes.

Key architectural decisions:

- **Append-only JSONL:** one event per line, deterministic to hash.
- **Causal-correct by default:** a `FROZEN_CAUSAL_SLAM_TRACE` replayer refuses to
  leak future corrections into past frames. A `FROZEN_FINAL_POSES` replayer is
  explicitly marked oracle (requires ``oracle_declared=True``).
- **Event-scoped availability:** evidence/poses/corrections are served as-of a
  logical time, not "latest". Missing this makes causal enforcement impossible at
  read time.
- **Admission bypass is explicit:** frozen and bypass modes are incompatible at
  construction time (never both active at once).

## Not implemented (deferred to #633 part 2)

**Derived rosbag/MCAP materialization:** see the issue's "Derived rosbag
materialization" section and acceptance criteria 6/9. Constructing a ROS2-compatible
bag or MCAP file from a validated trace is a separate, optional feature. This module
records and replays traces, not materializes them to bag format.

**Source import:** reading and validating raw RGB-D datasets into `source_frames.jsonl`
is owned by #634 (canonical scene registry) and dataset-specific tools, not by this
module. The recorder assumes source frames already exist as an external list.

**Frame preprocessing validation:** detecting RGB/depth sync issues, verifying
registration/rectification, or auto-detecting invalid-depth masks is owned by
#625 (eligsir-rgbd-sync) and dataset-specific validation, not by this module.

## House style (inherited from #632)

- Pure Python, CPU-only, torch-free, no side effects on import.
- Refuses (raises) rather than warns on undeclared/underivable state -- the bottom of
  the lattice, never a permissive default.
- Reuses contracts from :mod:`eligsir.experiments.replay_capsule`: enums, error
  classes, validation rules, hashing.
- ``dataclass.to_dict()`` / ``from_dict()`` for all serializable objects.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping, MutableMapping, Sequence

from eligsir.experiments.hashing import content_hash, hash_file
from eligsir.experiments.replay_capsule import (
    AXIS_ADMISSION_DECISIONS,
    AXIS_ADMISSION_EVIDENCE,
    AXIS_FRONTEND_POSE,
    AXIS_KEYFRAME_CANDIDATES,
    AXIS_REPLAY_SELECTION,
    AXIS_TRACKING_STATE,
    AdmissionTraceMode,
    ArtifactRef,
    CapsuleValidationError,
    FrontendTraceMode,
    validate_causal_pose_events,
)

# Event type literals
PoseEventType = Literal["pose_estimate", "pose_correction"]
TrackingEventType = Literal["tracking_state", "frame_admitted"]
AdmissionEventType = Literal["candidate_offered", "decision_made"]
ReplayEventType = Literal["camera_selected"]


@dataclass
class SourceFrameEvent:
    """A source frame record (one row in source_frames.jsonl).

    Args:
        frame_id: Unique, stable identifier for this source frame.
        timestamp_ns: Capture timestamp in nanoseconds (sensor time).
        rgb_hash: SHA-256 of RGB image bytes.
        depth_hash: SHA-256 of depth image bytes.
        camera_id: Camera model/instance identifier.
        rgb_decoder: Decoder used (e.g. "jpeg", "png", "raw").
        depth_decoder: Decoder used (e.g. "png16", "zstd_uint16", "raw").
        depth_scale_mm: Depth pixel value -> mm conversion factor.
        rgb_valid_mask_hash: SHA-256 of RGB validity mask, if present.
        depth_valid_mask_hash: SHA-256 of depth validity mask, if present.
        sync_status: "synchronized" / "unsynchronized_approx_time" / etc.
        logical_time: Mapper's internal logical clock when this frame arrived.
        source_release_event_id: Unique event ID in the release schedule.
        description: Free-text notes.
    """

    frame_id: str
    timestamp_ns: int
    rgb_hash: str
    depth_hash: str
    camera_id: str
    rgb_decoder: str
    depth_decoder: str
    depth_scale_mm: float
    rgb_valid_mask_hash: str | None = None
    depth_valid_mask_hash: str | None = None
    sync_status: str = "unknown"
    logical_time: int = 0
    source_release_event_id: str = ""
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return this event as a JSON-serialisable dict."""
        return {
            "frame_id": self.frame_id,
            "timestamp_ns": self.timestamp_ns,
            "rgb_hash": self.rgb_hash,
            "depth_hash": self.depth_hash,
            "camera_id": self.camera_id,
            "rgb_decoder": self.rgb_decoder,
            "depth_decoder": self.depth_decoder,
            "depth_scale_mm": self.depth_scale_mm,
            "rgb_valid_mask_hash": self.rgb_valid_mask_hash,
            "depth_valid_mask_hash": self.depth_valid_mask_hash,
            "sync_status": self.sync_status,
            "logical_time": self.logical_time,
            "source_release_event_id": self.source_release_event_id,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SourceFrameEvent":
        """Rebuild the event from its ``to_dict`` payload."""
        return cls(
            frame_id=data["frame_id"],
            timestamp_ns=int(data["timestamp_ns"]),
            rgb_hash=data["rgb_hash"],
            depth_hash=data["depth_hash"],
            camera_id=data["camera_id"],
            rgb_decoder=data["rgb_decoder"],
            depth_decoder=data["depth_decoder"],
            depth_scale_mm=float(data["depth_scale_mm"]),
            rgb_valid_mask_hash=data.get("rgb_valid_mask_hash"),
            depth_valid_mask_hash=data.get("depth_valid_mask_hash"),
            sync_status=data.get("sync_status", "unknown"),
            logical_time=int(data.get("logical_time", 0)),
            source_release_event_id=data.get("source_release_event_id", ""),
            description=data.get("description", ""),
        )


@dataclass
class PoseEvent:
    """A frontend pose estimate or correction event.

    Args:
        event_id: Unique, stable event identifier.
        event_type: "pose_estimate" (initial) or "pose_correction" (retroactive).
        logical_time: Mapper's internal clock when this event occurred.
        frame_id: Which source frame this pose is for.
        derived_from_logical_time: Latest mapper observation that informed this
            estimate. For causal replay, if this > logical_time, the event is a
            future leak.
        pose_c2w: Camera-to-world 4x4 as a flat list [16 floats].
        tracking_state: "OK" / "lost" / "relocalizing" / "re_initialized" / etc.
        pose_confidence: Optional confidence/covariance metric.
        corrected_from_event_id: If this is a correction, the event_id of the
            original estimate. Required for corrections.
        affected_frame_ids: For a loop-closure/batch correction, which other frames
            are affected (optional).
        source: "orbslam3" / "vins" / "realsense_t265" / etc.
        description: Free-text notes.
    """

    event_id: str
    event_type: PoseEventType
    logical_time: float
    frame_id: str
    derived_from_logical_time: float
    pose_c2w: list[float]  # 16 elements
    tracking_state: str = "OK"
    pose_confidence: float | None = None
    corrected_from_event_id: str | None = None
    affected_frame_ids: list[str] = field(default_factory=list)
    source: str = "unknown"
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return this event as a JSON-serialisable dict."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "logical_time": self.logical_time,
            "frame_id": self.frame_id,
            "derived_from_logical_time": self.derived_from_logical_time,
            "pose_c2w": self.pose_c2w,
            "tracking_state": self.tracking_state,
            "pose_confidence": self.pose_confidence,
            "corrected_from_event_id": self.corrected_from_event_id,
            "affected_frame_ids": self.affected_frame_ids,
            "source": self.source,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PoseEvent":
        """Rebuild the event from its ``to_dict`` payload."""
        return cls(
            event_id=data["event_id"],
            event_type=data["event_type"],
            logical_time=float(data["logical_time"]),
            frame_id=data["frame_id"],
            derived_from_logical_time=float(data["derived_from_logical_time"]),
            pose_c2w=list(data["pose_c2w"]),
            tracking_state=data.get("tracking_state", "OK"),
            pose_confidence=data.get("pose_confidence"),
            corrected_from_event_id=data.get("corrected_from_event_id"),
            affected_frame_ids=list(data.get("affected_frame_ids", [])),
            source=data.get("source", "unknown"),
            description=data.get("description", ""),
        )


@dataclass
class TrackingStateEvent:
    """A tracking state change event.

    Args:
        event_id: Unique event identifier.
        logical_time: When this state change occurred.
        frame_id: Source frame at this time.
        state: "OK" / "lost" / "relocalizing" / "re_initialized" / "not_initialized".
        severity: "nominal" / "warning" / "critical" / etc.
        description: Free-text notes.
    """

    event_id: str
    logical_time: float
    frame_id: str
    state: str
    severity: str = "nominal"
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return this event as a JSON-serialisable dict."""
        return {
            "event_id": self.event_id,
            "logical_time": self.logical_time,
            "frame_id": self.frame_id,
            "state": self.state,
            "severity": self.severity,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TrackingStateEvent":
        """Rebuild the event from its ``to_dict`` payload."""
        return cls(
            event_id=data["event_id"],
            logical_time=float(data["logical_time"]),
            frame_id=data["frame_id"],
            state=data["state"],
            severity=data.get("severity", "nominal"),
            description=data.get("description", ""),
        )


@dataclass
class AdmissionCandidateEvent:
    """A keyframe candidate offered to the admission policy.

    Args:
        event_id: Unique event identifier.
        logical_time: When the candidate became available.
        frame_id: The candidate frame.
        candidate_reason: Why this frame was a candidate ("keyframe" / "depth_insert" / etc.).
        evidence_available: Snapshot of evidence at decision time (dict).
        description: Free-text notes.
    """

    event_id: str
    logical_time: float
    frame_id: str
    candidate_reason: str
    evidence_available: dict[str, Any] = field(default_factory=dict)
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return this event as a JSON-serialisable dict."""
        return {
            "event_id": self.event_id,
            "logical_time": self.logical_time,
            "frame_id": self.frame_id,
            "candidate_reason": self.candidate_reason,
            "evidence_available": self.evidence_available,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AdmissionCandidateEvent":
        """Rebuild the event from its ``to_dict`` payload."""
        return cls(
            event_id=data["event_id"],
            logical_time=float(data["logical_time"]),
            frame_id=data["frame_id"],
            candidate_reason=data["candidate_reason"],
            evidence_available=dict(data.get("evidence_available", {})),
            description=data.get("description", ""),
        )


@dataclass
class AdmissionDecisionEvent:
    """A keyframe admission decision (accept/reject).

    Args:
        event_id: Unique event identifier.
        logical_time: When the decision was made.
        frame_id: Frame being decided on.
        candidate_event_id: The AdmissionCandidateEvent this answers.
        decision: "admit" / "reject" / "defer" / etc.
        reason_code: Short code for why ("coverage_sufficient" / "no_pose" / etc.).
        requested_budget: Target admission rate, if applicable.
        realized_budget: Actual rate/count realized, if applicable.
        description: Free-text notes.
    """

    event_id: str
    logical_time: float
    frame_id: str
    candidate_event_id: str
    decision: str
    reason_code: str
    requested_budget: float | None = None
    realized_budget: float | None = None
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return this event as a JSON-serialisable dict."""
        return {
            "event_id": self.event_id,
            "logical_time": self.logical_time,
            "frame_id": self.frame_id,
            "candidate_event_id": self.candidate_event_id,
            "decision": self.decision,
            "reason_code": self.reason_code,
            "requested_budget": self.requested_budget,
            "realized_budget": self.realized_budget,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AdmissionDecisionEvent":
        """Rebuild the event from its ``to_dict`` payload."""
        return cls(
            event_id=data["event_id"],
            logical_time=float(data["logical_time"]),
            frame_id=data["frame_id"],
            candidate_event_id=data["candidate_event_id"],
            decision=data["decision"],
            reason_code=data["reason_code"],
            requested_budget=data.get("requested_budget"),
            realized_budget=data.get("realized_budget"),
            description=data.get("description", ""),
        )


@dataclass
class ReplaySelectionEvent:
    """A historical camera replay selection event.

    Args:
        event_id: Unique event identifier.
        logical_time: When the selection was made.
        selected_keyframe_ids: Which keyframes were selected for this optimization step.
        selection_reason: "covisibility_based" / "temporal_proximity" / etc.
        batch_index: Batch number in the training loop, if meaningful.
        description: Free-text notes.
    """

    event_id: str
    logical_time: float
    selected_keyframe_ids: list[str]
    selection_reason: str = "unknown"
    batch_index: int | None = None
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return this event as a JSON-serialisable dict."""
        return {
            "event_id": self.event_id,
            "logical_time": self.logical_time,
            "selected_keyframe_ids": self.selected_keyframe_ids,
            "selection_reason": self.selection_reason,
            "batch_index": self.batch_index,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ReplaySelectionEvent":
        """Rebuild the event from its ``to_dict`` payload."""
        return cls(
            event_id=data["event_id"],
            logical_time=float(data["logical_time"]),
            selected_keyframe_ids=list(data["selected_keyframe_ids"]),
            selection_reason=data.get("selection_reason", "unknown"),
            batch_index=data.get("batch_index"),
            description=data.get("description", ""),
        )


@dataclass
class TraceRecorder:
    """Records append-only JSONL event logs for a single trace family.

    One recorder per artifact/trace family (source_frames, frontend/pose_events,
    admission/decisions, etc.). Events are flushed to JSONL on each call to
    :meth:`append` or :meth:`finalize`.

    Args:
        output_path: Path to the JSONL file to write. Parent directory is created
            if needed.
        artifact_key: The artifact key name (e.g. "frontend/pose_events"), used in
            reporting only.
        mode: "w" to truncate, "a" to append. Defaults to "w".
    """

    output_path: Path
    artifact_key: str
    mode: Literal["w", "a"] = "w"
    _fh: Any = field(default=None, init=False, repr=False)
    _row_count: int = field(default=0, init=False, repr=False)
    _closed: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        self.output_path = Path(self.output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.output_path, self.mode)

    def append(self, event: dict[str, Any]) -> None:
        """Append one event (as a dict) to the JSONL file.

        Raises:
            ValueError: if the recorder is closed.
        """
        if self._closed:
            raise ValueError(f"Recorder for {self.artifact_key} is closed; cannot append")
        line = json.dumps(event, separators=(",", ":"), sort_keys=True)
        self._fh.write(line + "\n")
        self._fh.flush()
        self._row_count += 1

    def close(self) -> None:
        """Finalize and close the JSONL file."""
        if not self._closed and self._fh:
            self._fh.close()
            self._closed = True

    def finalize(self) -> ArtifactRef:
        """Close the file and return an :class:`ArtifactRef` for the recorded artifact.

        Returns:
            ArtifactRef with content_hash and row_count populated.

        Raises:
            ValueError: if finalize is called twice.
        """
        if self._closed:
            raise ValueError(f"Recorder for {self.artifact_key} was already finalized")
        self.close()

        file_hash = hash_file(self.output_path)
        return ArtifactRef(
            uri=f"artifact://{self.artifact_key}",
            content_hash=file_hash,
            kind="event_log",
            row_count=self._row_count,
            local_path=str(self.output_path),
            description=f"Recorded {self._row_count} events",
        )

    def __del__(self) -> None:
        """Ensure file is closed on garbage collection."""
        if not self._closed and self._fh:
            try:
                self._fh.close()
            except Exception:
                pass


@dataclass
class PoseEventReplayer:
    """Replays recorded frontend pose events with causal or oracle semantics.

    Args:
        events: List of PoseEvent dicts (from recorded JSONL).
        trace_mode: "FROZEN_CAUSAL_SLAM_TRACE" or "FROZEN_FINAL_POSES".
        oracle_declared: Must be True if trace_mode is "FROZEN_FINAL_POSES".

    A **causal replayer** answers `pose_for(frame_id, at_logical_time=T)` and
    returns only poses that existed at time T (no future corrections applied before
    T). A **final-pose replayer** is an oracle and returns the latest/corrected pose
    regardless of time (use only for final-pose validation).
    """

    events: list[dict[str, Any]]
    trace_mode: FrontendTraceMode
    oracle_declared: bool = False
    _pose_by_frame: MutableMapping[str, dict[str, Any]] = field(default_factory=dict, init=False)
    _corrections: list[dict[str, Any]] = field(default_factory=list, init=False)
    _validated: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        self._validate_and_build_index()

    def _validate_and_build_index(self) -> None:
        """Validate causality (if applicable) and build lookup structures."""
        if self._validated:
            return

        # Convert events to dicts if they're dataclass instances
        events = [e if isinstance(e, dict) else dataclasses.asdict(e) for e in self.events]

        # Use the canonical validation from #632
        validate_causal_pose_events(
            events, mode=self.trace_mode, oracle_declared=self.oracle_declared
        )

        # Build index: latest pose per frame, and track corrections
        for event in events:
            frame_id = event["frame_id"]
            event_type = event.get("event_type", "pose_estimate")

            if event_type == "pose_estimate":
                self._pose_by_frame[frame_id] = event
            elif event_type == "pose_correction":
                self._corrections.append(event)

        self._validated = True

    def pose_for(self, frame_id: str, at_logical_time: float | None = None) -> dict[str, Any]:
        """Get the pose for a frame, optionally at a specific logical time.

        Args:
            frame_id: The frame to query.
            at_logical_time: For causal replay, the time to query at. Required for
                FROZEN_CAUSAL_SLAM_TRACE, forbidden (raises) for FROZEN_FINAL_POSES.

        Returns:
            The pose event (as a dict).

        Raises:
            ValueError: if frame has no pose, or if at_logical_time is missing for
                causal mode, or if oracle mode receives a time argument.
            CapsuleValidationError: if oracle_declared is False for FROZEN_FINAL_POSES.
        """
        if self.trace_mode == "FROZEN_FINAL_POSES":
            if at_logical_time is not None:
                raise ValueError(
                    "oracle mode (FROZEN_FINAL_POSES) does not accept at_logical_time; "
                    "call pose_for(frame_id) without a time argument"
                )
            if not self.oracle_declared:
                raise CapsuleValidationError(
                    f"pose_for() called on FROZEN_FINAL_POSES replayer without "
                    f"oracle_declared=True; refusing to leak future information"
                )
            # Return latest pose (with all corrections applied)
            pose = self._pose_by_frame.get(frame_id)
            if pose is None:
                raise ValueError(f"No pose recorded for frame {frame_id!r}")
            result = dict(pose)
            # Apply all corrections (oracle has all information)
            for correction in self._corrections:
                if correction["frame_id"] == frame_id:
                    result = dict(correction)  # Update with latest correction
            return result

        # FROZEN_CAUSAL_SLAM_TRACE: require at_logical_time
        if at_logical_time is None:
            raise ValueError(
                f"pose_for(frame_id, at_logical_time=...) requires at_logical_time "
                f"for FROZEN_CAUSAL_SLAM_TRACE replay"
            )

        # Find the pose as it existed at that time
        pose = self._pose_by_frame.get(frame_id)
        if pose is None:
            raise ValueError(f"No pose recorded for frame {frame_id!r}")

        # Apply corrections that are visible at at_logical_time
        result = dict(pose)  # Copy
        for correction in self._corrections:
            if correction["logical_time"] <= at_logical_time and correction["frame_id"] == frame_id:
                result = dict(correction)  # Update with correction

        return result


@dataclass
class AdmissionEventReplayer:
    """Replays recorded admission decisions or bypasses them for policy experiments.

    Args:
        decision_events: List of AdmissionDecisionEvent dicts.
        candidate_events: List of AdmissionCandidateEvent dicts.
        trace_mode: "FROZEN" or "BYPASS_FOR_POLICY_TEST".

    In FROZEN mode, the replayer returns the recorded admit/reject decision.
    In BYPASS mode, the replayer provides candidates and their evidence at decision time,
    allowing a live policy to make its own decisions.
    """

    decision_events: list[dict[str, Any]]
    candidate_events: list[dict[str, Any]]
    trace_mode: AdmissionTraceMode
    _decisions_by_candidate: MutableMapping[str, dict[str, Any]] = field(
        default_factory=dict, init=False
    )
    _candidates_by_frame: MutableMapping[str, dict[str, Any]] = field(
        default_factory=dict, init=False
    )
    _validated: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        self._validate_and_build_index()

    def _validate_and_build_index(self) -> None:
        """Build lookup structures."""
        if self._validated:
            return

        # Build candidate index
        for event in self.candidate_events:
            candidate_id = event.get("event_id")
            frame_id = event.get("frame_id")
            if candidate_id and frame_id:
                self._candidates_by_frame[frame_id] = event

        # Build decision index
        for event in self.decision_events:
            candidate_id = event.get("candidate_event_id")
            if candidate_id:
                self._decisions_by_candidate[candidate_id] = event

        self._validated = True

    def decision_for(self, frame_id: str, at_logical_time: float | None = None) -> dict[str, Any]:
        """Get the recorded admission decision for a frame.

        In FROZEN mode, returns the decision. In BYPASS mode, raises ValueError.

        Args:
            frame_id: The frame to query.
            at_logical_time: Ignored; provided for API consistency.

        Returns:
            The decision event (as a dict).

        Raises:
            ValueError: if frame has no decision, or if called in BYPASS mode.
        """
        if self.trace_mode != "FROZEN":
            raise ValueError(
                f"decision_for() is only available in FROZEN admission mode, "
                f"not in {self.trace_mode!r}; use candidate_for() instead for bypass mode"
            )

        candidate = self._candidates_by_frame.get(frame_id)
        if candidate is None:
            raise ValueError(f"No candidate recorded for frame {frame_id!r}")

        candidate_id = candidate["event_id"]
        decision = self._decisions_by_candidate.get(candidate_id)
        if decision is None:
            raise ValueError(f"No decision recorded for candidate {candidate_id!r}")

        return decision

    def candidate_for(
        self, frame_id: str, at_logical_time: float | None = None
    ) -> dict[str, Any]:
        """Get the candidate stream entry for a frame.

        In BYPASS mode, returns the candidate and its evidence at decision time.
        In FROZEN mode, also works but doesn't enforce that decisions are ignored.

        Args:
            frame_id: The frame to query.
            at_logical_time: Optional time filter (future: enforce evidence causality).

        Returns:
            The candidate event (as a dict).

        Raises:
            ValueError: if frame has no candidate.
        """
        candidate = self._candidates_by_frame.get(frame_id)
        if candidate is None:
            raise ValueError(f"No candidate recorded for frame {frame_id!r}")
        return candidate


@dataclass
class ManipulationReport:
    """Report on trace fidelity and replay mismatches (acceptance criterion #8).

    Args:
        requested_frontend_trace_mode: What the run declared.
        requested_admission_trace_mode: What the run declared.
        frontend_trace_hash: SHA-256 of the recorded pose events.
        admission_trace_hash: SHA-256 of the recorded admission events.
        capsule_hash: The capsule this pairing is against (optional).
        expected_pose_event_count: Declared/expected count from capsule.
        realized_pose_event_count: Actual events read back.
        expected_admission_event_count: Declared/expected count from capsule.
        realized_admission_event_count: Actual events read back.
        pose_mismatch_count: Events that differed on replay.
        admission_mismatch_count: Events that differed on replay.
        fallback_count: Times the replayer fell back to a default/None.
        first_mismatch_reason: Human-readable description of the first mismatch, if any.
        notes: Free-text diagnostic notes.
    """

    requested_frontend_trace_mode: str
    requested_admission_trace_mode: str
    frontend_trace_hash: str
    admission_trace_hash: str
    expected_pose_event_count: int | None = None
    realized_pose_event_count: int | None = None
    expected_admission_event_count: int | None = None
    realized_admission_event_count: int | None = None
    pose_mismatch_count: int = 0
    admission_mismatch_count: int = 0
    fallback_count: int = 0
    capsule_hash: str | None = None
    first_mismatch_reason: str = ""
    notes: str = ""

    @property
    def any_mismatch(self) -> bool:
        """True if any mismatch or fallback occurred."""
        return (
            self.pose_mismatch_count > 0
            or self.admission_mismatch_count > 0
            or self.fallback_count > 0
        )

    @property
    def is_valid_replay(self) -> bool:
        """True if trace replay can be considered faithful."""
        if self.requested_frontend_trace_mode == "NONE":
            expected_pose = 0
        else:
            expected_pose = self.expected_pose_event_count

        if self.requested_admission_trace_mode == "NONE":
            expected_admission = 0
        else:
            expected_admission = self.expected_admission_event_count

        pose_match = (
            expected_pose is None
            or self.realized_pose_event_count is None
            or self.realized_pose_event_count == expected_pose
        )
        admission_match = (
            expected_admission is None
            or self.realized_admission_event_count is None
            or self.realized_admission_event_count == expected_admission
        )
        no_mismatches = self.pose_mismatch_count == 0 and self.admission_mismatch_count == 0

        return pose_match and admission_match and no_mismatches

    def to_dict(self) -> dict[str, Any]:
        """Return this report as a JSON-serialisable dict."""
        return {
            "requested_frontend_trace_mode": self.requested_frontend_trace_mode,
            "requested_admission_trace_mode": self.requested_admission_trace_mode,
            "frontend_trace_hash": self.frontend_trace_hash,
            "admission_trace_hash": self.admission_trace_hash,
            "capsule_hash": self.capsule_hash,
            "expected_pose_event_count": self.expected_pose_event_count,
            "realized_pose_event_count": self.realized_pose_event_count,
            "expected_admission_event_count": self.expected_admission_event_count,
            "realized_admission_event_count": self.realized_admission_event_count,
            "pose_mismatch_count": self.pose_mismatch_count,
            "admission_mismatch_count": self.admission_mismatch_count,
            "fallback_count": self.fallback_count,
            "first_mismatch_reason": self.first_mismatch_reason,
            "any_mismatch": self.any_mismatch,
            "is_valid_replay": self.is_valid_replay,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ManipulationReport":
        """Rebuild the report from its ``to_dict`` payload."""
        return cls(
            requested_frontend_trace_mode=data["requested_frontend_trace_mode"],
            requested_admission_trace_mode=data["requested_admission_trace_mode"],
            frontend_trace_hash=data["frontend_trace_hash"],
            admission_trace_hash=data["admission_trace_hash"],
            capsule_hash=data.get("capsule_hash"),
            expected_pose_event_count=data.get("expected_pose_event_count"),
            realized_pose_event_count=data.get("realized_pose_event_count"),
            expected_admission_event_count=data.get("expected_admission_event_count"),
            realized_admission_event_count=data.get("realized_admission_event_count"),
            pose_mismatch_count=int(data.get("pose_mismatch_count", 0)),
            admission_mismatch_count=int(data.get("admission_mismatch_count", 0)),
            fallback_count=int(data.get("fallback_count", 0)),
            first_mismatch_reason=data.get("first_mismatch_reason", ""),
            notes=data.get("notes", ""),
        )
