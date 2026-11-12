"""Versioned Scene Replay Capsule schema + freeze-axis contract (issue #632).

A **Scene Replay Capsule** is the content-addressed description of exactly what
observation/evaluation substrate an experiment was allowed to see. It is not a
dataset container: large RGB-D payloads, bags and tensor caches stay external
and are referenced by a stable locator + content hash.

This module owns *schema and validation only*. It records no traces, reads no
rosbag, resolves no scene registry, and is imported by no training path --
those are issues #633 (trace recorders + derived bags) and #634 (canonical
capsule registry). See
``docs/architecture/2026-08-15-deterministic-experiment-replay-and-adjudication.md``
(PR #637) for the surrounding architecture, and
``docs/architecture/2026-08-15-scene-replay-capsule-schema.md`` for this
module's own notes and open questions.

Two objects, deliberately
-------------------------

``CapsuleManifest``
    What substrate *exists* and is content-hashed. One capsule can serve
    several lanes (``lane_compatibility``).

``FreezeAxisManifest``
    One experiment's pairing with one capsule: a single ``lane`` plus a
    classification of every axis as ``FROZEN`` / ``UNDER_TEST`` /
    ``FREE_BUT_MEASURED`` / ``NOT_APPLICABLE``.

Keeping the axis classification out of the capsule is what makes a single
capsule usable by both a ``MAPPER_CORE`` arm (admission ``FROZEN``) and a
``POLICY`` arm (admission ``UNDER_TEST``). The most valuable validator rule is
therefore cross-object: **an axis declared ``FROZEN`` must be backed by a real,
content-hashed artifact present in the capsule** -- otherwise "frozen" is a
word in a YAML file rather than a property of the experiment.

Relationship to the existing ``ExperimentManifest``
---------------------------------------------------

``eligsir.experiments.manifest`` already carries a coarse, monotone
``frozen_level`` ladder (``A``..``D``, Wave 3). The per-axis map here is
strictly more expressive: A/B/C/D can only freeze *more* as the letter grows
and has no way to say "everything upstream is frozen but admission is the
mechanism under test". This module does not delete, rewrite or import that
ladder; whether the two should be unified (and in which direction) is a
decision for #633/#634 and is recorded as an open question on #632 rather than
made here.

Refusal, never a warning
------------------------

Every check below raises :class:`CapsuleValidationError`. An undeclared or
underivable state is the BOTTOM of the lattice, never a permissive default --
the same principle #203's ``UNDECLARED``/``UNKNOWN`` classes apply. In
particular an *absent* axis is not ``NOT_APPLICABLE``: N/A must be declared,
with a reason.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Literal, Mapping, Sequence

from eligsir.experiments.hashing import content_hash

CAPSULE_SCHEMA_VERSION = 1
FREEZE_MANIFEST_SCHEMA_VERSION = 1

_SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")

FreezeClass = Literal["FROZEN", "UNDER_TEST", "FREE_BUT_MEASURED", "NOT_APPLICABLE"]
Lane = Literal["MAPPER_CORE", "POLICY", "END_TO_END"]
FrontendTraceMode = Literal["FROZEN_FINAL_POSES", "FROZEN_CAUSAL_SLAM_TRACE", "NONE"]
AdmissionTraceMode = Literal["FROZEN", "BYPASS_FOR_POLICY_TEST", "NONE"]
CapsuleStatus = Literal["ACTIVE", "RETIRED"]

VALID_FREEZE_CLASSES: tuple[str, ...] = (
    "FROZEN",
    "UNDER_TEST",
    "FREE_BUT_MEASURED",
    "NOT_APPLICABLE",
)
VALID_LANES: tuple[str, ...] = ("MAPPER_CORE", "POLICY", "END_TO_END")
VALID_FRONTEND_TRACE_MODES: tuple[str, ...] = (
    "FROZEN_FINAL_POSES",
    "FROZEN_CAUSAL_SLAM_TRACE",
    "NONE",
)
VALID_ADMISSION_TRACE_MODES: tuple[str, ...] = ("FROZEN", "BYPASS_FOR_POLICY_TEST", "NONE")

# ---------------------------------------------------------------------------
# Canonical axis list (issue #632's candidate list, one name per axis).
#
# The set is CLOSED: an unknown axis name is refused rather than ignored. A
# forward-compatible "ignore unknown keys" escape hatch is exactly how a
# misspelled axis silently becomes an unfrozen nuisance variable.
# ---------------------------------------------------------------------------
AXIS_RAW_SOURCE = "raw_source"
AXIS_PREPROCESSING = "preprocessing"
AXIS_CALIBRATION = "calibration"
AXIS_SOURCE_TIMESTAMPS = "source_timestamps"
AXIS_FRONTEND_POSE = "frontend_pose"
AXIS_TRACKING_STATE = "tracking_state"
AXIS_POSE_CORRECTIONS = "pose_corrections"
AXIS_KEYFRAME_CANDIDATES = "keyframe_candidates"
AXIS_ADMISSION_DECISIONS = "admission_decisions"
AXIS_ADMISSION_EVIDENCE = "admission_evidence"
AXIS_REPLAY_SELECTION = "replay_selection"
AXIS_INSERTION_SCHEDULE = "insertion_schedule"
AXIS_RNG_STREAMS = "rng_streams"
AXIS_EVALUATION = "evaluation"
AXIS_RELEASE_SCHEDULE = "release_schedule"
AXIS_WORK_BOUNDARIES = "work_boundaries"

CAPSULE_AXES: tuple[str, ...] = (
    AXIS_RAW_SOURCE,
    AXIS_PREPROCESSING,
    AXIS_CALIBRATION,
    AXIS_SOURCE_TIMESTAMPS,
    AXIS_FRONTEND_POSE,
    AXIS_TRACKING_STATE,
    AXIS_POSE_CORRECTIONS,
    AXIS_KEYFRAME_CANDIDATES,
    AXIS_ADMISSION_DECISIONS,
    AXIS_ADMISSION_EVIDENCE,
    AXIS_REPLAY_SELECTION,
    AXIS_INSERTION_SCHEDULE,
    AXIS_RNG_STREAMS,
    AXIS_EVALUATION,
    AXIS_RELEASE_SCHEDULE,
    AXIS_WORK_BOUNDARIES,
)

# Artifact keys a FROZEN axis must be backed by. This is the cross-object rule
# that makes a freeze declaration non-vacuous.
_AXIS_REQUIRED_ARTIFACTS: dict[str, tuple[str, ...]] = {
    AXIS_RAW_SOURCE: (),  # backed by CapsuleManifest.source, checked separately
    AXIS_PREPROCESSING: ("preprocessing_contract",),
    AXIS_CALIBRATION: ("camera_intrinsics",),
    AXIS_SOURCE_TIMESTAMPS: ("source_frames",),
    AXIS_FRONTEND_POSE: ("frontend/pose_events",),
    AXIS_TRACKING_STATE: ("frontend/tracking_state",),
    AXIS_POSE_CORRECTIONS: ("frontend/pose_corrections",),
    AXIS_KEYFRAME_CANDIDATES: ("admission/decisions",),
    AXIS_ADMISSION_DECISIONS: ("admission/decisions",),
    AXIS_ADMISSION_EVIDENCE: ("admission/admission_evidence",),
    AXIS_REPLAY_SELECTION: ("schedules/replay_schedule",),
    AXIS_INSERTION_SCHEDULE: ("schedules/insertion_schedule",),
    AXIS_RNG_STREAMS: ("rng_contract",),
    AXIS_EVALUATION: ("evaluation/holdout_pool", "evaluation/scored_views"),
    AXIS_RELEASE_SCHEDULE: ("schedules/release_schedule",),
    AXIS_WORK_BOUNDARIES: ("schedules/work_boundaries",),
}

# Axes whose backing artifact must be a RECORDED EVENT LOG, not a parameter
# record. This is the direct schema answer to "a scalar knob was silently
# overridden at runtime" (``effective_spf``, ``online_mode`` -> ``dataset_fps``):
# a declared release cadence of "spf=20" is a *request*; only a recorded
# release-event log is a frozen schedule.
_EVENT_BACKED_AXES: frozenset[str] = frozenset(
    {
        AXIS_RELEASE_SCHEDULE,
        AXIS_REPLAY_SELECTION,
        AXIS_INSERTION_SCHEDULE,
        AXIS_FRONTEND_POSE,
        AXIS_TRACKING_STATE,
        AXIS_POSE_CORRECTIONS,
        AXIS_ADMISSION_DECISIONS,
        AXIS_KEYFRAME_CANDIDATES,
        AXIS_ADMISSION_EVIDENCE,
    }
)

ArtifactKind = Literal["event_log", "id_list", "parameter_record", "blob", "tensor_cache"]
VALID_ARTIFACT_KINDS: tuple[str, ...] = (
    "event_log",
    "id_list",
    "parameter_record",
    "blob",
    "tensor_cache",
)

# ---------------------------------------------------------------------------
# Lane rules
#
# A lane says which axes must be frozen upstream of the mechanism under test,
# and which axes the lane is allowed to leave under test. Declaring an axis
# UNDER_TEST that the lane requires frozen is a self-contradiction and is
# refused -- e.g. a MAPPER_CORE arm whose admission is still live is not a
# mapper-core experiment, whatever the manifest calls it.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LaneRule:
    """Per-lane freeze requirements.

    Args:
        must_be_frozen: Axes that must be classified ``FROZEN`` (or
            ``NOT_APPLICABLE`` with a reason, for a substrate that genuinely
            has no such axis).
        may_be_under_test: Axes this lane is permitted to leave ``UNDER_TEST``.
        min_under_test: Minimum number of ``UNDER_TEST`` axes. A ``POLICY``
            lane with nothing under test is self-contradictory.
    """

    must_be_frozen: frozenset[str]
    may_be_under_test: frozenset[str]
    min_under_test: int = 0


_MAPPER_CORE_FROZEN = frozenset(
    {
        AXIS_RAW_SOURCE,
        AXIS_PREPROCESSING,
        AXIS_CALIBRATION,
        AXIS_SOURCE_TIMESTAMPS,
        AXIS_FRONTEND_POSE,
        AXIS_TRACKING_STATE,
        AXIS_POSE_CORRECTIONS,
        AXIS_KEYFRAME_CANDIDATES,
        AXIS_ADMISSION_DECISIONS,
        AXIS_EVALUATION,
        AXIS_RELEASE_SCHEDULE,
    }
)

_POLICY_FROZEN = frozenset(
    {
        AXIS_RAW_SOURCE,
        AXIS_PREPROCESSING,
        AXIS_CALIBRATION,
        AXIS_SOURCE_TIMESTAMPS,
        AXIS_FRONTEND_POSE,
        AXIS_TRACKING_STATE,
        AXIS_EVALUATION,
    }
)

LANE_RULES: dict[str, LaneRule] = {
    "MAPPER_CORE": LaneRule(
        must_be_frozen=_MAPPER_CORE_FROZEN,
        may_be_under_test=frozenset(
            {AXIS_INSERTION_SCHEDULE, AXIS_REPLAY_SELECTION, AXIS_RNG_STREAMS, AXIS_WORK_BOUNDARIES}
        ),
    ),
    "POLICY": LaneRule(
        must_be_frozen=_POLICY_FROZEN,
        may_be_under_test=frozenset(
            {
                AXIS_KEYFRAME_CANDIDATES,
                AXIS_ADMISSION_DECISIONS,
                AXIS_ADMISSION_EVIDENCE,
                AXIS_REPLAY_SELECTION,
                AXIS_INSERTION_SCHEDULE,
                AXIS_RELEASE_SCHEDULE,
                AXIS_RNG_STREAMS,
                AXIS_WORK_BOUNDARIES,
            }
        ),
        min_under_test=1,
    ),
    "END_TO_END": LaneRule(
        must_be_frozen=frozenset({AXIS_RAW_SOURCE, AXIS_SOURCE_TIMESTAMPS, AXIS_EVALUATION}),
        may_be_under_test=frozenset(CAPSULE_AXES)
        - frozenset({AXIS_RAW_SOURCE, AXIS_SOURCE_TIMESTAMPS, AXIS_EVALUATION}),
        min_under_test=1,
    ),
}

# Run-side fields the capsule CANNOT verify itself but that a paired run must
# report for the comparison to be checkable. The capsule pins the declared
# selection rule and the declared scored set; whether the run actually realized
# and actually scored that set is run-side evidence (#199/#420/#433/#437).
_EVALUATION_RUN_FIELDS: tuple[str, ...] = (
    "eval_metadata.realized_holdout_hash",
    "eval_metadata.scored_holdout_hash",
    "n_test_metrics_attempted",
    "n_test_metrics_failed",
)
_RELEASE_RUN_FIELDS: tuple[str, ...] = (
    "ingestion.mode",
    "ingestion.reason",
    "realized_release_events",
)
_ADMISSION_RUN_FIELDS: tuple[str, ...] = ("realized_admitted_frames",)
_COVERAGE_RUN_FIELDS: tuple[str, ...] = ("realized_source_frames_ingested",)


class CapsuleValidationError(ValueError):
    """Raised when a capsule or freeze-axis manifest is refused.

    Every failure mode in this module raises rather than warns: a capsule that
    cannot be trusted must not be usable at all.
    """


# ---------------------------------------------------------------------------
# Schema objects
# ---------------------------------------------------------------------------


@dataclass
class ArtifactRef:
    """A content-addressed reference to one capsule artifact.

    Args:
        uri: Stable, location-independent locator, e.g.
            ``artifact://datasets/tum/fr3/pose_events.jsonl``. Participates in
            capsule identity.
        content_hash: SHA-256 hex digest of the artifact's bytes.
        kind: What sort of artifact this is. ``event_log`` is a recorded event
            stream; ``parameter_record`` is a declared configuration value.
            The distinction is load-bearing (see ``_EVENT_BACKED_AXES``).
        row_count: Number of records, when meaningful. Required for
            ``event_log``/``id_list`` so a row-count cross-check is possible.
        local_path: Optional convenience path on the machine that produced or
            fetched this artifact. **Deliberately excluded from capsule
            identity** -- relocating a file must not change what the capsule
            means.
        description: Free-text note.
    """

    uri: str
    content_hash: str
    kind: ArtifactKind
    row_count: int | None = None
    local_path: str | None = None
    description: str = ""

    def identity_payload(self) -> dict[str, Any]:
        """Identity-participating fields only (no ``local_path``)."""
        return {
            "uri": self.uri,
            "content_hash": self.content_hash,
            "kind": self.kind,
            "row_count": self.row_count,
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dictionary including all fields."""
        return {
            "uri": self.uri,
            "content_hash": self.content_hash,
            "kind": self.kind,
            "row_count": self.row_count,
            "local_path": self.local_path,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ArtifactRef":
        """Deserialize from a dictionary."""
        return cls(
            uri=data["uri"],
            content_hash=data["content_hash"],
            kind=data["kind"],
            row_count=data.get("row_count"),
            local_path=data.get("local_path"),
            description=data.get("description", ""),
        )


@dataclass
class SourceRef:
    """The immutable raw source this capsule is built over.

    Args:
        uri: Stable locator for the raw dataset/bag.
        content_hash: SHA-256 hex digest of the raw source (or of its own
            manifest, when the payload is a directory tree).
        frame_count: Authoritative number of source frames. This is the
            capsule's own denominator for coverage -- it exists precisely so a
            run never has to depend on a runtime ``eval/scenes.json`` lookup
            that may miss and silently disable coverage enforcement.
        raw_payload_embedded: Whether the raw payload lives inside the capsule
            directory (normally ``False``).
    """

    uri: str
    content_hash: str
    frame_count: int
    raw_payload_embedded: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dictionary."""
        return {
            "uri": self.uri,
            "content_hash": self.content_hash,
            "frame_count": self.frame_count,
            "raw_payload_embedded": self.raw_payload_embedded,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SourceRef":
        """Deserialize from a dictionary."""
        return cls(
            uri=data["uri"],
            content_hash=data["content_hash"],
            frame_count=int(data["frame_count"]),
            raw_payload_embedded=bool(data.get("raw_payload_embedded", False)),
        )


@dataclass
class ResolvedValue:
    """A configuration value recorded as *requested* AND *resolved*.

    This project has repeatedly been burned by a knob whose requested value is
    not the value that ran (``--streaming_steps_per_frame 20`` clamped to
    ``effective_spf=50``; ``--online_mode`` resolving to ``dataset_fps``
    pacing). A capsule that records only the request cannot express what
    actually happened, so both sides are mandatory here and ``resolved=None``
    is refused.

    Args:
        requested: What the launcher/spec asked for. ``None`` when nothing was
            explicitly requested (a default was taken).
        resolved: What the system actually resolved to. Must not be ``None``.
        resolver: Free-text identifier of what decided it, e.g.
            ``"MappingScheduler.effective_spf"``.
    """

    requested: Any
    resolved: Any
    resolver: str = ""

    @property
    def overridden(self) -> bool:
        """True when a request existed and the resolved value differs from it."""
        return self.requested is not None and self.requested != self.resolved

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dictionary."""
        return {
            "requested": self.requested,
            "resolved": self.resolved,
            "resolver": self.resolver,
            "overridden": self.overridden,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ResolvedValue":
        """Deserialize from a dictionary."""
        return cls(
            requested=data.get("requested"),
            resolved=data.get("resolved"),
            resolver=data.get("resolver", ""),
        )


@dataclass
class EvaluationIdentity:
    """Evaluation identity, integrating #199/#420/#433 semantics.

    Three distinct hashes, because this project has learned the hard way that
    they are three different things:

    Args:
        holdout_pool_hash: Hash of the holdout SELECTION RULE / declared pool.
        scored_view_hash: Hash of the exact camera IDs the capsule declares
            must be scored.
        probe_view_hash: Hash of fixed probe/local-region views, when used.
        eval_resolution: ``(width, height)`` the evaluation renders at.
        eval_intrinsics_hash: Hash of the intrinsics used for evaluation after
            any resize/crop.
        spatial_regions_hash: Hash of canonical spatial regions, when used.

    Note the *realized* holdout identity is deliberately absent: it can only be
    known from the run, not from the capsule. :func:`required_run_fields`
    returns the run-side field names a paired run must report so a comparison
    can check it -- recording the requirement here without pretending the
    capsule can satisfy it.
    """

    holdout_pool_hash: str
    scored_view_hash: str
    probe_view_hash: str | None = None
    eval_resolution: tuple[int, int] | None = None
    eval_intrinsics_hash: str | None = None
    spatial_regions_hash: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dictionary."""
        return {
            "holdout_pool_hash": self.holdout_pool_hash,
            "scored_view_hash": self.scored_view_hash,
            "probe_view_hash": self.probe_view_hash,
            "eval_resolution": list(self.eval_resolution) if self.eval_resolution else None,
            "eval_intrinsics_hash": self.eval_intrinsics_hash,
            "spatial_regions_hash": self.spatial_regions_hash,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EvaluationIdentity":
        """Deserialize from a dictionary."""
        resolution = data.get("eval_resolution")
        return cls(
            holdout_pool_hash=data["holdout_pool_hash"],
            scored_view_hash=data["scored_view_hash"],
            probe_view_hash=data.get("probe_view_hash"),
            eval_resolution=(int(resolution[0]), int(resolution[1])) if resolution else None,
            eval_intrinsics_hash=data.get("eval_intrinsics_hash"),
            spatial_regions_hash=data.get("spatial_regions_hash"),
        )


@dataclass
class ProducerInfo:
    """Who/what produced this capsule.

    Args:
        git_commit: Commit the producing tool ran at.
        tool_version: Version string of the capsule-producing tool.
        created_utc: ISO-8601 UTC timestamp.
        notes: Free-text.
    """

    git_commit: str
    tool_version: str
    created_utc: str = ""
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "git_commit": self.git_commit,
            "tool_version": self.tool_version,
            "created_utc": self.created_utc,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ProducerInfo":
        return cls(
            git_commit=data["git_commit"],
            tool_version=data["tool_version"],
            created_utc=data.get("created_utc", ""),
            notes=data.get("notes", ""),
        )


@dataclass
class CapsuleManifest:
    """The versioned Scene Replay Capsule manifest.

    Args:
        capsule_id: Immutable scientific input identity, e.g.
            ``tum_fr3_mapper_core_v1``. Never mutate a published ID in place;
            publish a new version and retire the old one.
        scene_id: Scene this capsule is built over.
        source: The immutable raw source (locator, hash, authoritative frame
            count).
        lane_compatibility: Lanes this capsule can legitimately serve.
        frontend_trace_mode: ``FROZEN_FINAL_POSES`` (one authoritative pose per
            frame; an oracle relative to a causal frontend) vs.
            ``FROZEN_CAUSAL_SLAM_TRACE`` (the event history the online mapper
            actually saw) vs. ``NONE``. Participates in capsule identity, so
            the two modes hash differently even over identical final poses.
        admission_trace_mode: ``FROZEN`` / ``BYPASS_FOR_POLICY_TEST`` / ``NONE``.
        evaluation: Evaluation identity (see :class:`EvaluationIdentity`).
        producer: Producing tool/commit.
        artifacts: ``{artifact_key: ArtifactRef}``. Keys are the layout paths
            from #632 (``frontend/pose_events``, ``admission/decisions``, ...).
        resolved_semantics: ``{name: ResolvedValue}`` -- requested-vs-resolved
            records for every run-semantics knob that materially shaped this
            substrate (ingestion mode, effective steps-per-frame, coverage
            denominator source, ...).
        schema_version: Schema version of this manifest.
        status: ``ACTIVE`` or ``RETIRED``.
        superseded_by: Required when ``status == "RETIRED"``.
        retirement_reason: Free-text, required when ``status == "RETIRED"``.
        notes: Free-text.
    """

    capsule_id: str
    scene_id: str
    source: SourceRef
    lane_compatibility: tuple[str, ...]
    frontend_trace_mode: FrontendTraceMode
    admission_trace_mode: AdmissionTraceMode
    evaluation: EvaluationIdentity
    producer: ProducerInfo
    artifacts: dict[str, ArtifactRef] = field(default_factory=dict)
    resolved_semantics: dict[str, ResolvedValue] = field(default_factory=dict)
    schema_version: int = CAPSULE_SCHEMA_VERSION
    status: CapsuleStatus = "ACTIVE"
    superseded_by: str | None = None
    retirement_reason: str = ""
    notes: str = ""

    def identity_payload(self) -> dict[str, Any]:
        """The content-hashed identity of this capsule.

        Built from content hashes and declared URIs only -- never from
        filesystem locations. Relocating the capsule directory therefore leaves
        its identity unchanged by construction.
        """
        return {
            "schema_version": self.schema_version,
            "capsule_id": self.capsule_id,
            "scene_id": self.scene_id,
            "source": self.source.to_dict(),
            "lane_compatibility": sorted(self.lane_compatibility),
            "frontend_trace_mode": self.frontend_trace_mode,
            "admission_trace_mode": self.admission_trace_mode,
            "evaluation": self.evaluation.to_dict(),
            "artifacts": {
                key: ref.identity_payload() for key, ref in sorted(self.artifacts.items())
            },
            "resolved_semantics": {
                key: value.to_dict() for key, value in sorted(self.resolved_semantics.items())
            },
            "status": self.status,
            "superseded_by": self.superseded_by,
        }

    def capsule_hash(self) -> str:
        """SHA-256 of :meth:`identity_payload`, via the repo's canonical hasher."""
        return content_hash(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "capsule_id": self.capsule_id,
            "scene_id": self.scene_id,
            "source": self.source.to_dict(),
            "lane_compatibility": list(self.lane_compatibility),
            "frontend_trace_mode": self.frontend_trace_mode,
            "admission_trace_mode": self.admission_trace_mode,
            "evaluation": self.evaluation.to_dict(),
            "producer": self.producer.to_dict(),
            "artifacts": {key: ref.to_dict() for key, ref in sorted(self.artifacts.items())},
            "resolved_semantics": {
                key: value.to_dict() for key, value in sorted(self.resolved_semantics.items())
            },
            "status": self.status,
            "superseded_by": self.superseded_by,
            "retirement_reason": self.retirement_reason,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CapsuleManifest":
        return cls(
            capsule_id=data["capsule_id"],
            scene_id=data["scene_id"],
            source=SourceRef.from_dict(data["source"]),
            lane_compatibility=tuple(data["lane_compatibility"]),
            frontend_trace_mode=data["frontend_trace_mode"],
            admission_trace_mode=data["admission_trace_mode"],
            evaluation=EvaluationIdentity.from_dict(data["evaluation"]),
            producer=ProducerInfo.from_dict(data["producer"]),
            artifacts={
                key: ArtifactRef.from_dict(ref) for key, ref in data.get("artifacts", {}).items()
            },
            resolved_semantics={
                key: ResolvedValue.from_dict(value)
                for key, value in data.get("resolved_semantics", {}).items()
            },
            schema_version=int(data.get("schema_version", CAPSULE_SCHEMA_VERSION)),
            status=data.get("status", "ACTIVE"),
            superseded_by=data.get("superseded_by"),
            retirement_reason=data.get("retirement_reason", ""),
            notes=data.get("notes", ""),
        )


@dataclass
class AxisDeclaration:
    """One axis's classification for one experiment.

    Args:
        classification: ``FROZEN`` / ``UNDER_TEST`` / ``FREE_BUT_MEASURED`` /
            ``NOT_APPLICABLE``.
        reason: Why. **Required** for ``NOT_APPLICABLE`` (an axis is never
            silently inapplicable) and for ``FREE_BUT_MEASURED`` (which must
            say what will be measured).
        measured_fields: For ``FREE_BUT_MEASURED``, the run-report fields whose
            realized values must be recorded. Required and non-empty for that
            class -- "measured" without a named measurement is just "free".
    """

    classification: FreezeClass
    reason: str = ""
    measured_fields: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "classification": self.classification,
            "reason": self.reason,
            "measured_fields": list(self.measured_fields),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AxisDeclaration":
        return cls(
            classification=data["classification"],
            reason=data.get("reason", ""),
            measured_fields=tuple(data.get("measured_fields", ())),
        )


@dataclass
class FreezeAxisManifest:
    """One experiment's freeze-axis contract against one capsule.

    This object is deliberately standalone and serializable from either
    location -- next to the capsule, or next to the experiment spec. Where it
    is *stored* is a registry/layout decision owned by #634; nothing here
    assumes either.

    Args:
        experiment_id: The experiment this pairing belongs to.
        capsule_id: The capsule being paired with.
        capsule_hash: The capsule's :meth:`CapsuleManifest.capsule_hash` at
            declaration time. Checked against the real capsule on validation.
        lane: Exactly one lane (``MAPPER_CORE`` / ``POLICY`` / ``END_TO_END``).
            Unlike ``CapsuleManifest.lane_compatibility`` this is singular: an
            experiment is run in one lane.
        axes: ``{axis_name: AxisDeclaration}``. Must cover every axis in
            :data:`CAPSULE_AXES` -- an absent axis is refused, not treated as
            ``NOT_APPLICABLE``.
        oracle_declared: Set ``True`` to declare, explicitly, that this
            experiment is a final-pose oracle (it may consume poses that
            incorporate future information). Required to pair with a
            ``FROZEN_FINAL_POSES`` capsule.
        schema_version: Schema version of this manifest.
    """

    experiment_id: str
    capsule_id: str
    capsule_hash: str
    lane: Lane
    axes: dict[str, AxisDeclaration] = field(default_factory=dict)
    oracle_declared: bool = False
    schema_version: int = FREEZE_MANIFEST_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "experiment_id": self.experiment_id,
            "capsule_id": self.capsule_id,
            "capsule_hash": self.capsule_hash,
            "lane": self.lane,
            "oracle_declared": self.oracle_declared,
            "axes": {name: decl.to_dict() for name, decl in sorted(self.axes.items())},
        }

    def freeze_manifest_hash(self) -> str:
        return content_hash(self.to_dict())

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FreezeAxisManifest":
        return cls(
            experiment_id=data["experiment_id"],
            capsule_id=data["capsule_id"],
            capsule_hash=data["capsule_hash"],
            lane=data["lane"],
            axes={
                name: AxisDeclaration.from_dict(decl) for name, decl in data.get("axes", {}).items()
            },
            oracle_declared=bool(data.get("oracle_declared", False)),
            schema_version=int(data.get("schema_version", FREEZE_MANIFEST_SCHEMA_VERSION)),
        )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _refuse(message: str) -> None:
    raise CapsuleValidationError(message)


def _require_sha256(value: Any, what: str) -> None:
    if not isinstance(value, str) or not _SHA256_HEX_RE.match(value):
        _refuse(f"{what} must be a 64-char lowercase sha256 hex digest, got {value!r}")


def validate_capsule(
    capsule: CapsuleManifest,
    *,
    checksums: Mapping[str, str] | None = None,
) -> None:
    """Validate a capsule manifest standalone. Raises :class:`CapsuleValidationError`.

    Args:
        capsule: The manifest to check.
        checksums: Optional ``{artifact_key: sha256}`` read from the capsule's
            own ``checksums.sha256`` sidecar. Any disagreement with the
            manifest's recorded hash, and any artifact missing from either
            side, is refused -- a checksum file that does not match the
            manifest means one of the two is stale and neither can be trusted.
    """
    if capsule.schema_version != CAPSULE_SCHEMA_VERSION:
        _refuse(
            f"capsule schema_version {capsule.schema_version!r} is not supported "
            f"(this build understands {CAPSULE_SCHEMA_VERSION}); refusing rather than "
            "best-effort parsing an unknown schema"
        )

    if not capsule.capsule_id:
        _refuse("capsule_id must be a non-empty immutable identity string")
    if not capsule.scene_id:
        _refuse("scene_id must be non-empty")

    if capsule.frontend_trace_mode not in VALID_FRONTEND_TRACE_MODES:
        _refuse(
            f"frontend_trace_mode must be one of {VALID_FRONTEND_TRACE_MODES}, "
            f"got {capsule.frontend_trace_mode!r}"
        )
    if capsule.admission_trace_mode not in VALID_ADMISSION_TRACE_MODES:
        _refuse(
            f"admission_trace_mode must be one of {VALID_ADMISSION_TRACE_MODES}, "
            f"got {capsule.admission_trace_mode!r}"
        )

    if not capsule.lane_compatibility:
        _refuse("lane_compatibility must declare at least one lane")
    for lane in capsule.lane_compatibility:
        if lane not in VALID_LANES:
            _refuse(f"unknown lane {lane!r} in lane_compatibility; valid lanes are {VALID_LANES}")

    if capsule.status not in ("ACTIVE", "RETIRED"):
        _refuse(f"status must be ACTIVE or RETIRED, got {capsule.status!r}")
    if capsule.status == "RETIRED":
        if not capsule.superseded_by or not capsule.retirement_reason:
            _refuse(
                "a RETIRED capsule must record both superseded_by and retirement_reason "
                "so historical experiment reports stay interpretable"
            )

    # Source: the capsule's own authoritative coverage denominator.
    _require_sha256(capsule.source.content_hash, "source.content_hash")
    if capsule.source.frame_count <= 0:
        _refuse(
            "source.frame_count must be a positive integer -- it is the capsule's own "
            "authoritative coverage denominator and must never fall back to a runtime "
            "scene-registry lookup that may miss"
        )

    _require_sha256(capsule.evaluation.holdout_pool_hash, "evaluation.holdout_pool_hash")
    _require_sha256(capsule.evaluation.scored_view_hash, "evaluation.scored_view_hash")
    if capsule.evaluation.probe_view_hash is not None:
        _require_sha256(capsule.evaluation.probe_view_hash, "evaluation.probe_view_hash")

    if not capsule.producer.git_commit or capsule.producer.git_commit == "unknown":
        _refuse(
            "producer.git_commit must be a real resolved commit, not empty or 'unknown' -- "
            "an unresolved producer is the bottom of the lattice, not a permissive default"
        )

    # Artifacts.
    for key, ref in capsule.artifacts.items():
        _require_sha256(ref.content_hash, f"artifacts[{key!r}].content_hash")
        if ref.kind not in VALID_ARTIFACT_KINDS:
            _refuse(
                f"artifacts[{key!r}].kind must be one of {VALID_ARTIFACT_KINDS}, got {ref.kind!r}"
            )
        if not ref.uri:
            _refuse(
                f"artifacts[{key!r}].uri must be a stable, location-independent locator "
                "(a local_path alone is not a locator)"
            )
        if ref.kind in ("event_log", "id_list") and ref.row_count is None:
            _refuse(
                f"artifacts[{key!r}] is a {ref.kind} and must record row_count so a "
                "row-count cross-check against source.frame_count is possible"
            )

    source_frames = capsule.artifacts.get("source_frames")
    if source_frames is not None and source_frames.row_count is not None:
        if source_frames.row_count != capsule.source.frame_count:
            _refuse(
                f"source.frame_count ({capsule.source.frame_count}) disagrees with the "
                f"source_frames artifact row_count ({source_frames.row_count}); the capsule's "
                "coverage denominator is ambiguous and it must be refused"
            )

    for name, value in capsule.resolved_semantics.items():
        if value.resolved is None:
            _refuse(
                f"resolved_semantics[{name!r}].resolved is None -- a knob whose resolved "
                "value is unknown cannot be part of a frozen substrate (this is exactly "
                "the requested-vs-resolved gap that let effective_spf and the "
                "online_mode->dataset_fps default silently override a spec)"
            )

    if checksums is not None:
        manifest_keys = set(capsule.artifacts)
        checksum_keys = set(checksums)
        if manifest_keys != checksum_keys:
            _refuse(
                "checksums sidecar does not cover the same artifact set as the manifest: "
                f"only-in-manifest={sorted(manifest_keys - checksum_keys)!r}, "
                f"only-in-checksums={sorted(checksum_keys - manifest_keys)!r}"
            )
        for key in sorted(manifest_keys):
            if checksums[key] != capsule.artifacts[key].content_hash:
                _refuse(
                    f"checksum mismatch for artifact {key!r}: manifest records "
                    f"{capsule.artifacts[key].content_hash!r}, checksums sidecar records "
                    f"{checksums[key]!r}"
                )


def validate_freeze_axes(freeze: FreezeAxisManifest) -> None:
    """Validate a freeze-axis manifest standalone (no capsule cross-check)."""
    if freeze.schema_version != FREEZE_MANIFEST_SCHEMA_VERSION:
        _refuse(
            f"freeze-axis manifest schema_version {freeze.schema_version!r} is not supported "
            f"(this build understands {FREEZE_MANIFEST_SCHEMA_VERSION})"
        )
    if freeze.lane not in VALID_LANES:
        _refuse(f"lane must be one of {VALID_LANES}, got {freeze.lane!r}")

    unknown = sorted(set(freeze.axes) - set(CAPSULE_AXES))
    if unknown:
        _refuse(
            f"unknown freeze axis/axes {unknown!r}; the axis set is closed "
            f"({list(CAPSULE_AXES)!r}). Unknown keys are refused, never ignored -- a "
            "misspelled axis would otherwise become a silently unfrozen nuisance variable"
        )

    missing = sorted(set(CAPSULE_AXES) - set(freeze.axes))
    if missing:
        _refuse(
            f"freeze-axis manifest does not classify axis/axes {missing!r}; every axis must "
            "be declared explicitly. An absent axis is NOT NOT_APPLICABLE -- declare "
            "NOT_APPLICABLE with a reason if it genuinely does not apply"
        )

    for axis, decl in sorted(freeze.axes.items()):
        if decl.classification not in VALID_FREEZE_CLASSES:
            _refuse(
                f"axis {axis!r}: classification must be one of {VALID_FREEZE_CLASSES}, "
                f"got {decl.classification!r}"
            )
        if decl.classification == "NOT_APPLICABLE" and not decl.reason.strip():
            _refuse(
                f"axis {axis!r} is declared NOT_APPLICABLE without a reason; N/A must be "
                "argued, not assumed"
            )
        if decl.classification == "FREE_BUT_MEASURED":
            if not decl.reason.strip():
                _refuse(f"axis {axis!r} is declared FREE_BUT_MEASURED without a reason")
            if not decl.measured_fields:
                _refuse(
                    f"axis {axis!r} is declared FREE_BUT_MEASURED but names no "
                    "measured_fields; 'measured' without a named measurement is just 'free'"
                )

    rule = LANE_RULES[freeze.lane]
    under_test = sorted(
        axis for axis, decl in freeze.axes.items() if decl.classification == "UNDER_TEST"
    )
    if len(under_test) < rule.min_under_test:
        _refuse(
            f"lane {freeze.lane!r} requires at least {rule.min_under_test} axis/axes "
            f"UNDER_TEST, got {len(under_test)}; a policy/end-to-end experiment with "
            "nothing under test is self-contradictory"
        )

    illegal_under_test = sorted(set(under_test) - rule.may_be_under_test)
    if illegal_under_test:
        _refuse(
            f"lane {freeze.lane!r} cannot leave axis/axes {illegal_under_test!r} UNDER_TEST -- "
            f"that lane requires them frozen upstream of the mechanism under test. A "
            f"{freeze.lane} arm whose {illegal_under_test[0]} is live is not a "
            f"{freeze.lane} experiment, whatever the manifest calls it"
        )

    not_frozen = sorted(
        axis
        for axis in rule.must_be_frozen
        if freeze.axes[axis].classification not in ("FROZEN", "NOT_APPLICABLE")
    )
    # Anything in must_be_frozen that is UNDER_TEST has already been refused
    # above; this catches FREE_BUT_MEASURED on a must-freeze axis.
    if not_frozen:
        _refuse(
            f"lane {freeze.lane!r} requires axis/axes {not_frozen!r} to be FROZEN (or "
            "explicitly NOT_APPLICABLE with a reason); FREE_BUT_MEASURED is not sufficient "
            "for an axis this lane's attribution depends on"
        )


def validate_pairing(
    freeze: FreezeAxisManifest,
    capsule: CapsuleManifest,
    *,
    checksums: Mapping[str, str] | None = None,
) -> None:
    """Validate a freeze-axis manifest against the capsule it pairs with.

    This is where the contract stops being a description and starts being
    enforceable: a ``FROZEN`` axis must be backed by a real, content-hashed
    artifact in the capsule, and for schedule/event axes that artifact must be
    a recorded ``event_log``, not a ``parameter_record``.

    Raises:
        CapsuleValidationError: on any refusal.
    """
    validate_capsule(capsule, checksums=checksums)
    validate_freeze_axes(freeze)

    if freeze.capsule_id != capsule.capsule_id:
        _refuse(
            f"freeze-axis manifest pairs with capsule_id {freeze.capsule_id!r} but was "
            f"validated against {capsule.capsule_id!r}"
        )

    actual_hash = capsule.capsule_hash()
    if freeze.capsule_hash != actual_hash:
        _refuse(
            f"capsule_hash mismatch: freeze-axis manifest declares {freeze.capsule_hash!r}, "
            f"capsule content hashes to {actual_hash!r}. The frozen substrate changed after "
            "this experiment declared it"
        )

    if capsule.status == "RETIRED":
        _refuse(
            f"capsule {capsule.capsule_id!r} is RETIRED (superseded by "
            f"{capsule.superseded_by!r}); a retired capsule cannot back a new experiment"
        )

    if freeze.lane not in capsule.lane_compatibility:
        _refuse(
            f"lane {freeze.lane!r} is not in the capsule's lane_compatibility "
            f"{list(capsule.lane_compatibility)!r}"
        )

    # Frozen axes must be backed by real, content-hashed artifacts.
    for axis, decl in sorted(freeze.axes.items()):
        if decl.classification != "FROZEN":
            continue
        if axis == AXIS_RAW_SOURCE:
            # Backed by CapsuleManifest.source, already hash-checked.
            continue
        required = _AXIS_REQUIRED_ARTIFACTS[axis]
        for artifact_key in required:
            ref = capsule.artifacts.get(artifact_key)
            if ref is None:
                _refuse(
                    f"axis {axis!r} is declared FROZEN but the capsule carries no "
                    f"{artifact_key!r} artifact to freeze it against; a freeze declaration "
                    "with no content-hashed substrate behind it is vacuous"
                )
            if axis in _EVENT_BACKED_AXES and ref.kind != "event_log":
                _refuse(
                    f"axis {axis!r} is declared FROZEN but its backing artifact "
                    f"{artifact_key!r} has kind {ref.kind!r}; this axis must be frozen "
                    "against a RECORDED event_log, not a declared parameter. A requested "
                    "cadence/mode is not a frozen schedule -- that is precisely how "
                    "effective_spf silently overrode --streaming_steps_per_frame and how "
                    "--online_mode silently resolved to dataset_fps pacing"
                )

    # Trace-mode coherence.
    pose_decl = freeze.axes[AXIS_FRONTEND_POSE]
    if pose_decl.classification == "FROZEN" and capsule.frontend_trace_mode == "NONE":
        _refuse(
            "frontend_pose is declared FROZEN but the capsule's frontend_trace_mode is NONE"
        )
    if capsule.frontend_trace_mode == "FROZEN_FINAL_POSES" and not freeze.oracle_declared:
        _refuse(
            "capsule uses FROZEN_FINAL_POSES (one authoritative pose per frame, which may "
            "incorporate information from future observations) but the experiment does not "
            "set oracle_declared=True; a final-pose oracle must say so explicitly"
        )

    admission_decl = freeze.axes[AXIS_ADMISSION_DECISIONS]
    if admission_decl.classification == "FROZEN" and capsule.admission_trace_mode != "FROZEN":
        _refuse(
            f"admission_decisions is declared FROZEN but the capsule's admission_trace_mode "
            f"is {capsule.admission_trace_mode!r}"
        )
    if admission_decl.classification == "UNDER_TEST" and capsule.admission_trace_mode == "FROZEN":
        _refuse(
            "admission_decisions is UNDER_TEST but the capsule's admission_trace_mode is "
            "FROZEN; a policy arm must consume the candidate stream with the recorded "
            "decision BYPASSED (admission_trace_mode=BYPASS_FOR_POLICY_TEST), never both "
            "apply the frozen decision and run a live policy"
        )


def required_run_fields(freeze: FreezeAxisManifest) -> tuple[str, ...]:
    """Run-report fields a paired run must fill for this pairing to be checkable.

    The capsule pins the *declared* substrate. Whether a run actually realized
    and actually scored that substrate is run-side evidence the capsule cannot
    verify about itself -- notably the realized/scored holdout identity
    (#199/#420/#433), the resolved ingestion pacing (#302), and realized
    coverage against ``source.frame_count``. Recording the requirement here
    keeps the gap explicit instead of implying the capsule closes it.

    The recorder that populates these fields is #633's; this function only
    names them.
    """
    fields: list[str] = list(_COVERAGE_RUN_FIELDS)
    for axis, decl in sorted(freeze.axes.items()):
        if decl.classification == "FREE_BUT_MEASURED":
            fields.extend(decl.measured_fields)
        if axis == AXIS_EVALUATION and decl.classification in ("FROZEN", "FREE_BUT_MEASURED"):
            fields.extend(_EVALUATION_RUN_FIELDS)
        if axis == AXIS_RELEASE_SCHEDULE and decl.classification != "NOT_APPLICABLE":
            fields.extend(_RELEASE_RUN_FIELDS)
        if axis == AXIS_ADMISSION_DECISIONS and decl.classification != "NOT_APPLICABLE":
            fields.extend(_ADMISSION_RUN_FIELDS)
    return tuple(sorted(set(fields)))


def validate_causal_pose_events(
    events: Sequence[Mapping[str, Any]],
    *,
    mode: FrontendTraceMode,
    oracle_declared: bool = False,
) -> None:
    """Refuse a causal pose trace that leaks future information.

    Each event is a mapping with at least:

    ``logical_time``
        When the mapper received this estimate.
    ``frame_id``
        Which source frame the estimate is for.
    ``derived_from_logical_time``
        The latest observation time that informed this estimate. An estimate
        informed by an observation the mapper had not yet seen is a future
        leak.

    In ``FROZEN_FINAL_POSES`` mode the trace is an oracle by construction, so
    the leak check is skipped -- but only when ``oracle_declared`` is True,
    which is the point of making that flag explicit.

    Raises:
        CapsuleValidationError: on non-monotonic ``logical_time``, a missing
            required key, or an illegal future reference.
    """
    if mode == "NONE":
        if events:
            _refuse("frontend_trace_mode is NONE but pose events were supplied")
        return

    last_time: float | None = None
    for index, event in enumerate(events):
        for key in ("logical_time", "frame_id", "derived_from_logical_time"):
            if key not in event:
                _refuse(f"pose event #{index} is missing required key {key!r}")
        logical_time = event["logical_time"]
        if last_time is not None and logical_time < last_time:
            _refuse(
                f"pose event #{index} has logical_time {logical_time!r} < previous "
                f"{last_time!r}; a causal trace must be ordered in logical time"
            )
        last_time = logical_time

    if mode == "FROZEN_FINAL_POSES":
        if not oracle_declared:
            _refuse(
                "FROZEN_FINAL_POSES traces are oracles relative to a causal frontend; "
                "validating one requires oracle_declared=True"
            )
        return

    for index, event in enumerate(events):
        if event["derived_from_logical_time"] > event["logical_time"]:
            _refuse(
                f"pose event #{index} (frame {event['frame_id']!r}) is derived from "
                f"logical_time {event['derived_from_logical_time']!r} but is delivered at "
                f"{event['logical_time']!r}: a causal replay may not hand the mapper a "
                "future-corrected pose early. Reclassify as a final-pose oracle if this is "
                "intended"
            )


def make_axis_map(
    default: FreezeClass,
    *,
    overrides: Mapping[str, AxisDeclaration] | None = None,
    default_reason: str = "",
) -> dict[str, AxisDeclaration]:
    """Build a complete axis map covering every axis in :data:`CAPSULE_AXES`.

    A convenience for constructing manifests and tests without hand-writing 16
    entries. It does not relax any validation rule -- the resulting map is
    still validated normally, and a ``NOT_APPLICABLE``/``FREE_BUT_MEASURED``
    default still needs a reason.

    Args:
        default: Classification for every axis not in ``overrides``.
        overrides: Per-axis declarations that replace the default.
        default_reason: Reason attached to the defaulted axes.
    """
    overrides = dict(overrides or {})
    unknown = sorted(set(overrides) - set(CAPSULE_AXES))
    if unknown:
        _refuse(f"make_axis_map(): unknown axis/axes {unknown!r}")
    return {
        axis: overrides.get(axis, AxisDeclaration(classification=default, reason=default_reason))
        for axis in CAPSULE_AXES
    }


def axes_by_class(freeze: FreezeAxisManifest, classification: FreezeClass) -> tuple[str, ...]:
    """Return the axes in ``freeze`` carrying ``classification``, sorted."""
    return tuple(
        sorted(axis for axis, decl in freeze.axes.items() if decl.classification == classification)
    )


def describe_refusals() -> tuple[str, ...]:
    """Human-readable list of what :func:`validate_pairing` refuses.

    Kept next to the implementation so the documented contract and the enforced
    contract cannot drift apart silently.
    """
    return (
        "unsupported capsule/freeze-manifest schema_version (refused, never best-effort parsed)",
        "unknown freeze axis name (the axis set is closed)",
        "missing axis declaration (absent is not NOT_APPLICABLE)",
        "NOT_APPLICABLE or FREE_BUT_MEASURED without a reason",
        "FREE_BUT_MEASURED with no named measured_fields",
        "an axis UNDER_TEST that the declared lane requires frozen",
        "a POLICY/END_TO_END lane with zero axes UNDER_TEST",
        "a lane not in the capsule's lane_compatibility",
        "a FROZEN axis with no content-hashed backing artifact in the capsule",
        "a FROZEN schedule/event axis backed by a parameter_record instead of an event_log",
        "capsule_hash declared by the experiment != the capsule's real content hash",
        "checksums sidecar disagreeing with the manifest (mismatched hash or artifact set)",
        "source.frame_count absent/non-positive, or disagreeing with the source_frames rows",
        "a resolved_semantics entry whose resolved value is None",
        "producer.git_commit empty or the literal 'unknown'",
        "a RETIRED capsule backing a new experiment, or retired without superseded_by/reason",
        "FROZEN_FINAL_POSES paired with an experiment that does not declare oracle_declared",
        "admission UNDER_TEST against an admission_trace_mode=FROZEN capsule (and vice versa)",
        "a causal pose event derived from a logical time later than its own delivery time",
    )


def iter_axis_documentation() -> Iterable[tuple[str, str]]:
    """Yield ``(axis, one-line description)`` for every canonical axis."""
    descriptions = {
        AXIS_RAW_SOURCE: "raw RGB/depth payload identity (external, content-addressed)",
        AXIS_PREPROCESSING: "decode/resize/colour-space/depth-scale/invalid-mask semantics",
        AXIS_CALIBRATION: "intrinsics, distortion and extrinsics (post-resize/crop)",
        AXIS_SOURCE_TIMESTAMPS: "source timestamps and frame ordering",
        AXIS_FRONTEND_POSE: "frontend pose estimates delivered to the mapper",
        AXIS_TRACKING_STATE: "tracking validity/state (OK, lost, relocalizing, re-init)",
        AXIS_POSE_CORRECTIONS: "loop closures and retroactive pose corrections",
        AXIS_KEYFRAME_CANDIDATES: "the candidate frame stream offered to admission",
        AXIS_ADMISSION_DECISIONS: "admit/reject per candidate, with reason codes",
        AXIS_ADMISSION_EVIDENCE: "the evidence available at each admission decision",
        AXIS_REPLAY_SELECTION: "which historical cameras are replayed, and when",
        AXIS_INSERTION_SCHEDULE: "the geometry/depth insertion schedule",
        AXIS_RNG_STREAMS: "python/numpy/torch/sampler seeds and RNG consumption contract",
        AXIS_EVALUATION: "holdout pool, scored views, fixed probes, eval resolution",
        AXIS_RELEASE_SCHEDULE: "the logical-clock release schedule of source frames",
        AXIS_WORK_BOUNDARIES: "mapping-end and refinement work boundaries",
    }
    for axis in CAPSULE_AXES:
        yield axis, descriptions[axis]
