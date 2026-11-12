"""One bounded backpressure owner (issue #601, design parent #599).

Problem this module owns
------------------------
When source observations arrive faster than the mapper can assimilate
them, EliGSiR must **bound** the pending mapping work by skipping or
coalescing redundant observations *before* expensive RGB/depth
materialization -- not accumulate an ever-staler queue in order to
preserve raw-frame coverage.

Before this module, four mechanisms partly touched that problem:

* :class:`~eligsir.mapping.scheduler.frame_scheduler.FrameScheduler`
  decides *when* a source frame becomes due (release pacing);
* the ``train_streaming.py`` drain loop could *defer* excess buffered
  frames ("back-pressure defers, never drops -> coverage preserved");
* :class:`~eligsir.mapping.scheduler.burst_budget.BurstBudget` could
  *drop* some non-keyframes in ``CATCH_UP`` / high-backlog;
* admission policy separately decided which frames become
  keyframes/supervision.

That is a multi-owner architecture. This module is the **single
execution owner** of the skip/coalesce decision; the mechanisms above
become signal providers (see ``single-owner contract`` below).

Pipeline position
-----------------
::

    source observation metadata          (cheap: id, stamp, pose, flags)
            |
            v
    PendingObservationCohort(max_count, max_age)
            |
            v
    BackpressureDecision
      admit | metadata_only | coalesce_into(frame_id)
            | skip_redundant | skip_capacity
            |
            v
    only ADMIT reaches expensive camera/depth materialization

There is no second frame database here. Callers keep recording metadata
and history through the existing observation/history primitives; this
module only decides.

Determinism (hard requirement)
------------------------------
Issue #601 requires "the exact retained set must be deterministic in
offline replay mode". Therefore **no decision in this module may read a
host wall clock** unless the operator explicitly opts into
:attr:`BackpressureConfig.age_axis` ``= "wallclock"``. The default age
axis is the canonical ``"control_tick"`` epoch. ``"source_time"`` uses
dataset/source timestamps, which are equally reproducible. Optimizer
iteration is an artifact identifier and is never accepted as admission
time. Measured host-side signals (ms/frame, optimizer it/s, GPU duty
cycle) are deliberately *not* pressure inputs for the default
configuration -- they are recorded as telemetry only.

Policy order (issue #599/#601)
------------------------------
Under LOW pressure: **soft-all** -- admit useful observations liberally
(this is what production ``all`` means after #600; it is not the
historical hard-all, which survives as an explicitly named stress mode).

Under rising pressure, in this order:

1. discard/coalesce clearly redundant intermediate observations;
2. protect the newest useful frontier so map age cannot grow without
   bound (:attr:`BackpressureConfig.frontier_quota`);
3. protect high-novelty / high-quality / cheap-keyframe observations
   (:attr:`novelty_protect`, :attr:`quality_protect`,
   :attr:`keyframe_quota`);
4. select ``<= K`` representatives from the local pending segment
   (:attr:`representatives_per_segment`);
5. prevent skipped frames from triggering insertion, replay residency,
   multires cache construction, or GPU camera materialization -- which
   is enforced by the caller honouring
   :attr:`BackpressureDecision.admit_supervision` /
   :attr:`~BackpressureDecision.allow_geometry_insertion`.

No learned or opaque score is used. Every signal that feeds a decision
is a named float carried on the decision itself
(:attr:`BackpressureDecision.signals`) so it can be logged and tested.

Single-owner contract
---------------------
After this lands, when :attr:`BackpressureConfig.enabled` is True:

* ``BurstBudget``'s overlapping *drop execution* must be neutralized by
  the caller (see :func:`burst_config_as_signal_provider`); its
  per-iteration cap becomes an input signal, not a second dropper.
* The drain loop's "defer forever for coverage" behaviour must not
  resurrect a frame this owner already decided to skip: a decision is
  terminal for that ``frame_id``.
* Admission must consume this decision rather than independently
  re-admitting a skipped frame.
* Geometry insertion must require ``allow_geometry_insertion``.

Relationship to #600
--------------------
Issue #600 (semantic prerequisite) introduces an ``ObservationDecision``
contract with the fields ``observe`` / ``mapping_candidate`` /
``admit_supervision`` / ``allow_geometry_insertion`` / ``reason``.
:class:`BackpressureDecision` mirrors **exactly those field names** so
that reconciliation is a single adapter swap;
:func:`as_observation_decision` is that single adapter. It prefers the
real #600 contract when importable and falls back to a structurally
identical local record otherwise -- see that function's docstring.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Dict, List, Mapping, Optional, Sequence, Tuple


def _control_tick(value: object, name: str = "control_tick") -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an exact integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _finite_seconds(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite number")
    resolved = float(value)
    if not math.isfinite(resolved):
        raise ValueError(f"{name} must be finite")
    return resolved

__all__ = [
    "BackpressureAction",
    "BackpressureConfig",
    "BackpressureDecision",
    "BackpressureOwner",
    "BackpressureTelemetry",
    "ObservationDecisionView",
    "PendingObservation",
    "PendingObservationCohort",
    "PressureState",
    "as_observation_decision",
    "backpressure_config_from_args",
    "burst_config_as_signal_provider",
    "pose_delta_signals",
]


# ----------------------------------------------------------------------
# Enums and reason codes
# ----------------------------------------------------------------------
class BackpressureAction(str, Enum):
    """What the owner decided for one source observation.

    ``str`` subclass so JSON/telemetry serialize as plain strings.
    """

    ADMIT = "admit"
    METADATA_ONLY = "metadata_only"
    COALESCE_INTO = "coalesce_into"
    SKIP_REDUNDANT = "skip_redundant"
    SKIP_CAPACITY = "skip_capacity"


class PressureState(str, Enum):
    """Coarse, explicitly-thresholded pressure bucket.

    Deliberately not one opaque scalar (#599: "Avoid one opaque scalar
    initially. Record components and test threshold policies
    independently"). The components that produced the state are always
    carried alongside it on the decision.
    """

    LOW = "low"
    RISING = "rising"
    HIGH = "high"
    CRITICAL = "critical"


REASON_SOFT_ALL = "soft_all_capacity_available"
REASON_FORCE_ALL_STRESS = "force_all_stress"
REASON_DISABLED = "backpressure_disabled"
REASON_NOT_MAPPING_CANDIDATE = "not_mapping_candidate"
REASON_REPRESENTATIVE = "segment_representative"
REASON_PROTECTED_NOVELTY = "protected_high_novelty"
REASON_PROTECTED_QUALITY = "protected_high_quality"
REASON_PROTECTED_KEYFRAME = "protected_cheap_keyframe"
REASON_PROTECTED_FRONTIER = "protected_frontier"
REASON_FORCED_FRONTIER_ADMIT = "forced_frontier_admit_age_bound"
REASON_COALESCED_REDUNDANT = "coalesced_into_representative"
REASON_SKIP_REDUNDANT = "skipped_redundant_intermediate"
REASON_SKIP_CAPACITY = "skipped_capacity_exhausted"

BACKPRESSURE_REASON_CODES = frozenset(
    {
        REASON_SOFT_ALL,
        REASON_FORCE_ALL_STRESS,
        REASON_DISABLED,
        REASON_NOT_MAPPING_CANDIDATE,
        REASON_REPRESENTATIVE,
        REASON_PROTECTED_NOVELTY,
        REASON_PROTECTED_QUALITY,
        REASON_PROTECTED_KEYFRAME,
        REASON_PROTECTED_FRONTIER,
        REASON_FORCED_FRONTIER_ADMIT,
        REASON_COALESCED_REDUNDANT,
        REASON_SKIP_REDUNDANT,
        REASON_SKIP_CAPACITY,
    }
)

#: Reason codes that mean "this observation was kept because a
#: protection rule fired", as opposed to ordinary representative
#: selection. Used by the telemetry's frontier/protection retention
#: accounting and by tests.
PROTECTION_REASON_CODES = frozenset(
    {
        REASON_PROTECTED_NOVELTY,
        REASON_PROTECTED_QUALITY,
        REASON_PROTECTED_KEYFRAME,
        REASON_PROTECTED_FRONTIER,
        REASON_FORCED_FRONTIER_ADMIT,
    }
)


# ----------------------------------------------------------------------
# Cheap observation metadata
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class PendingObservation:
    """Cheap, pre-materialization metadata for one source observation.

    Everything here is available *before* any RGB/depth decode, camera
    construction, GPU upload, multires cache build, or depth insertion.
    Constructing one of these must never be expensive.

    Attributes:
        frame_id: Source-frame index (the stable identity used by every
            downstream consumer and by ``coalesce_into``).
        source_timestamp: Dataset/sensor timestamp in seconds. Used for
            the ``source_time`` age axis and for telemetry. Reproducible
            in offline replay.
        release_control_tick: Canonical control epoch at which the source
            released this frame. This is independent of optimizer progress.
        cheap_keyframe: The existing cheap pose-delta/gap keyframe
            estimate, if the caller has one. Protected under pressure.
        novelty: ``[0, 1]`` cheap novelty estimate (e.g. normalized pose
            baseline against the last admitted pose, or a weak-region
            hint). ``0.0`` when the caller has no estimate.
        quality: ``[0, 1]`` cheap quality estimate (blur/exposure/pose
            confidence). ``0.0`` when the caller has no estimate.
        motion: ``[0, 1]`` cheap motion magnitude since the previous
            observation.
        translation_m: Metres translated since the previous observation
            (or since the last admitted frame -- the caller chooses, it
            is only used for the redundancy threshold).
        rotation_deg: Degrees rotated over the same interval.
        wallclock_s: Host monotonic time of observation, telemetry only.
            **Never read by a decision** unless
            :attr:`BackpressureConfig.age_axis` is ``"wallclock"``.
    """

    frame_id: int
    source_timestamp: float = 0.0
    release_control_tick: int = 0
    cheap_keyframe: bool = False
    novelty: float = 0.0
    quality: float = 0.0
    motion: float = 0.0
    translation_m: float = 0.0
    rotation_deg: float = 0.0
    wallclock_s: float = 0.0

    def __post_init__(self) -> None:
        _control_tick(self.release_control_tick, "release_control_tick")
        _finite_seconds(self.source_timestamp, "source_timestamp")
        _finite_seconds(self.wallclock_s, "wallclock_s")


def pose_delta_signals(
    prev_c2w: Optional[Sequence[Sequence[float]]],
    c2w: Optional[Sequence[Sequence[float]]],
) -> Tuple[float, float]:
    """Cheap ``(translation_m, rotation_deg)`` between two 4x4 c2w poses.

    Pure Python/`math` on 16 floats -- no numpy import, no GPU, safe to
    call once per source frame in the drain loop. Returns ``(0.0, 0.0)``
    when either pose is missing (the caller then simply has no
    redundancy signal and the owner falls back to score-based
    representative selection).
    """
    if prev_c2w is None or c2w is None:
        return 0.0, 0.0
    try:
        dt = math.sqrt(
            sum((float(c2w[i][3]) - float(prev_c2w[i][3])) ** 2 for i in range(3))
        )
        # Relative rotation angle: trace(R_prev^T @ R) = sum_i sum_k
        # R_prev[k][i] * R[k][i]; theta = acos((trace - 1) / 2).
        trace = 0.0
        for i in range(3):
            for k in range(3):
                trace += float(prev_c2w[k][i]) * float(c2w[k][i])
        cos_theta = max(-1.0, min(1.0, (trace - 1.0) * 0.5))
        return dt, math.degrees(math.acos(cos_theta))
    except (TypeError, IndexError, ValueError):
        return 0.0, 0.0


# ----------------------------------------------------------------------
# Decision
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class BackpressureDecision:
    """Terminal decision for one source observation.

    The ``observe`` / ``mapping_candidate`` / ``admit_supervision`` /
    ``allow_geometry_insertion`` / ``reason`` quartet mirrors #600's
    ``ObservationDecision`` field names exactly (see module docstring).

    Invariants enforced in ``__post_init__``:

    * ``admit_supervision`` is True **iff** ``action is ADMIT``;
    * ``allow_geometry_insertion`` implies ``admit_supervision`` -- a
      frame rejected for mapping can never be inserted from depth;
    * ``representative_frame_id`` is set **iff** ``action is
      COALESCE_INTO``, and never points at the frame itself;
    * ``observe`` is always True -- source observation/tracking never
      waits for mapping admission (#600 code invariant).
    """

    frame_id: int
    action: BackpressureAction
    reason: str
    pressure_state: PressureState = PressureState.LOW
    representative_frame_id: Optional[int] = None
    pending_depth: int = 0
    pending_age_control_ticks: int = 0
    pending_age_source_s: float = 0.0
    decided_control_tick: int = 0
    signals: Mapping[str, float] = field(default_factory=dict)
    stress_mode: bool = False

    # ---- #600-mirroring view ----------------------------------------
    observe: bool = True
    mapping_candidate: bool = True
    admit_supervision: bool = False
    allow_geometry_insertion: bool = False

    def __post_init__(self) -> None:
        if self.reason not in BACKPRESSURE_REASON_CODES:
            raise ValueError(
                "BackpressureDecision.reason must be one of "
                f"{sorted(BACKPRESSURE_REASON_CODES)}, got {self.reason!r}"
            )
        if not self.observe:
            raise ValueError(
                "BackpressureDecision invariant violated: observe must always be True "
                "(source observation/tracking never waits for mapping admission)"
            )
        is_admit = self.action is BackpressureAction.ADMIT
        if self.admit_supervision != is_admit:
            raise ValueError(
                "BackpressureDecision invariant violated: admit_supervision must be True "
                f"iff action is ADMIT (action={self.action.value}, "
                f"admit_supervision={self.admit_supervision})"
            )
        if self.allow_geometry_insertion and not self.admit_supervision:
            raise ValueError(
                "BackpressureDecision invariant violated: allow_geometry_insertion=True "
                "requires admit_supervision=True (a frame rejected for mapping must never "
                "be inserted from source depth)"
            )
        is_coalesce = self.action is BackpressureAction.COALESCE_INTO
        if is_coalesce != (self.representative_frame_id is not None):
            raise ValueError(
                "BackpressureDecision invariant violated: representative_frame_id must be "
                "set iff action is COALESCE_INTO"
            )
        if is_coalesce and self.representative_frame_id == self.frame_id:
            raise ValueError(
                "BackpressureDecision invariant violated: a frame cannot be coalesced "
                "into itself"
            )

    @property
    def materializes(self) -> bool:
        """True iff this frame may reach expensive materialization.

        The single predicate a drain loop should branch on before doing
        any RGB/depth decode, camera construction, GPU upload, multires
        cache build, replay-buffer residency, or depth insertion.
        """
        return self.admit_supervision

    @property
    def skipped(self) -> bool:
        """True iff this observation did not become mapping work."""
        return self.action is not BackpressureAction.ADMIT


@dataclass(frozen=True)
class ObservationDecisionView:
    """Local stand-in for #600's ``ObservationDecision``.

    Used only when the real contract from issue #600 is not importable
    (i.e. #600 has not landed yet). Field names are identical, so
    replacing this with the real class is a one-line import change
    inside :func:`as_observation_decision`.
    """

    observe: bool
    mapping_candidate: bool
    admit_supervision: bool
    allow_geometry_insertion: bool
    reason: str


#: Translation from this module's fine-grained reason codes to #600's
#: coarser ``ObservationDecision`` reason vocabulary
#: (``eligsir.mapping.admission.observation_decision``). #601's codes
#: are strictly finer -- e.g. #600 has one ``redundant`` where this module
#: distinguishes "coalesced into a representative" from "skipped as a
#: redundant intermediate" -- so the mapping is many-to-one and lossless
#: only in the #600 direction. The original #601 code is always preserved
#: verbatim under ``extra["backpressure_reason"]``.
_OBSERVATION_DECISION_REASONS = {
    REASON_SOFT_ALL: "admitted",
    REASON_DISABLED: "admitted",
    REASON_FORCE_ALL_STRESS: "force_all_stress",
    REASON_NOT_MAPPING_CANDIDATE: "quality",
    REASON_REPRESENTATIVE: "admitted",
    REASON_PROTECTED_NOVELTY: "admitted",
    REASON_PROTECTED_QUALITY: "admitted",
    REASON_PROTECTED_KEYFRAME: "admitted",
    REASON_PROTECTED_FRONTIER: "admitted",
    REASON_FORCED_FRONTIER_ADMIT: "frontier_forced",
    REASON_COALESCED_REDUNDANT: "redundant",
    REASON_SKIP_REDUNDANT: "redundant",
    REASON_SKIP_CAPACITY: "capacity",
}


def as_observation_decision(decision: BackpressureDecision):
    """Adapt a :class:`BackpressureDecision` to #600's contract.

    **This is the single reconciliation point with issue #600.** It
    prefers the real contract
    (``eligsir.mapping.admission.observation_decision.ObservationDecision``)
    when importable, and falls back to the structurally identical
    :class:`ObservationDecisionView` otherwise -- which is what happens on
    any checkout where #600 has not landed yet. Nothing else in this
    module, and nothing in the training loop, should construct an
    ``ObservationDecision`` directly.

    Reasons are translated to #600's coarser vocabulary (see
    :data:`_OBSERVATION_DECISION_REASONS`); the exact #601 code survives
    in ``extra["backpressure_reason"]`` together with the pressure state
    and the named signals the decision was taken on, so nothing an
    operator needs for diagnosis is lost in the narrowing.
    """
    mapped_reason = _OBSERVATION_DECISION_REASONS.get(decision.reason, decision.reason)
    for module_name in (
        "eligsir.mapping.admission.observation_decision",
        # Alternate landing site, in case #600 lands the class alongside
        # the existing W3-07 AdmissionDecision contracts instead.
        "eligsir.mapping.admission.decision_contracts",
    ):
        try:
            module = __import__(module_name, fromlist=["ObservationDecision"])
        except ImportError:
            continue
        real = getattr(module, "ObservationDecision", None)
        if real is None:
            continue
        return real(
            observe=decision.observe,
            mapping_candidate=decision.mapping_candidate,
            admit_supervision=decision.admit_supervision,
            allow_geometry_insertion=decision.allow_geometry_insertion,
            reason=mapped_reason,
            consecutive_skips=int(decision.pending_depth),
            extra={
                "owner": "BackpressureOwner",
                "backpressure_reason": decision.reason,
                "backpressure_action": decision.action.value,
                "pressure_state": decision.pressure_state.value,
                "representative_frame_id": decision.representative_frame_id,
                "pending_age_control_ticks": int(decision.pending_age_control_ticks),
                "signals": dict(decision.signals),
            },
        )
    return ObservationDecisionView(
        observe=decision.observe,
        mapping_candidate=decision.mapping_candidate,
        admit_supervision=decision.admit_supervision,
        allow_geometry_insertion=decision.allow_geometry_insertion,
        reason=mapped_reason,
    )


# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class BackpressureConfig:
    """Hard bounds and policy thresholds. All of these are echoed into
    ``report.json`` by :meth:`BackpressureTelemetry.report_block` -- #601
    requires production mode to *configure and report* them.

    Attributes:
        enabled: Master switch. Default ``False`` => the owner is never
            constructed and behaviour is byte-identical to today.
        force_all_stress: #600's explicitly-named stress mode. Bypasses
            this owner entirely (every observation ADMITs) and raises a
            loud flag in the report. Never a production setting.
        pending_max_count: Hard cap on the pending cohort size. Memory
            can never scale with source length.
        pending_max_age_control_ticks: Hard cap on the oldest pending
            observation's age in canonical control epochs.
        pending_max_age_source_s: Hard cap on that age measured on the
            source/dataset clock, in seconds. ``0`` disables this axis.
        pending_max_age_ms: Wall-clock variant, milliseconds. Only read
            when :attr:`age_axis` is ``"wallclock"`` -- it is
            **non-deterministic by construction** and must not be used
            for offline replay comparisons.
        age_axis: ``"control_tick"`` (default, deterministic),
            ``"source_time"`` (deterministic) or ``"wallclock"``
            (non-deterministic; opt-in only).
        max_admissions_per_window: Maximum *expensive* admissions inside
            one :attr:`admission_window_control_ticks` control window.
        admission_window_control_ticks: Length of that window, in control
            epochs.
        frontier_quota: Newest N pending observations are never
            coalesced/skipped by a compression pass -- this is what
            bounds source-to-map age.
        keyframe_quota: Max cheap-keyframes protected per compression
            pass (on top of the frontier).
        representatives_per_segment: ``K`` -- representatives chosen from
            one compressible pending segment.
        novelty_protect: Novelty at or above this is protected.
        quality_protect: Quality at or above this is protected.
        redundant_translation_m: A non-representative observation within
            this translation of the representative is ``coalesce_into``
            rather than ``skip_redundant``.
        redundant_rotation_deg: Same, for rotation.
        causal_deltas_are_cumulative: When True, ``translation_m`` /
            ``rotation_deg`` on each observation are read as the delta
            **already accumulated since the last admitted frame** rather
            than as a per-observation increment (the training loop's
            ``_pose_delta_from_last_train`` reports the former).
        decision_ledger_size: How many recent ``frame_id -> action`` entries
            :meth:`BackpressureOwner.decision_for` remembers. Bounded so
            nothing in this module scales with source length.
        rising_fill_frac: Pending fill fraction at which pressure leaves
            LOW.
        high_fill_frac: Pending fill fraction at which pressure reaches
            HIGH.
        novelty_weight/quality_weight/motion_weight: Explicit linear
            weights of the representative score. Not learned, not
            opaque; all three components are logged per decision.
    """

    enabled: bool = False
    force_all_stress: bool = False

    resolution_mode: str = "causal"

    pending_max_count: int = 8
    pending_max_age_control_ticks: int = 600
    pending_max_age_source_s: float = 0.0
    pending_max_age_ms: float = 0.0
    age_axis: str = "control_tick"

    max_admissions_per_window: int = 4
    admission_window_control_ticks: int = 100

    frontier_quota: int = 1
    keyframe_quota: int = 1
    representatives_per_segment: int = 1

    novelty_protect: float = 0.85
    quality_protect: float = 0.95

    redundant_translation_m: float = 0.05
    redundant_rotation_deg: float = 5.0

    causal_admit_translation_m: float = 0.10
    causal_admit_rotation_deg: float = 10.0
    causal_deltas_are_cumulative: bool = False
    decision_ledger_size: int = 4096

    rising_fill_frac: float = 0.5
    high_fill_frac: float = 0.8

    novelty_weight: float = 1.0
    quality_weight: float = 0.5
    motion_weight: float = 0.25

    def __post_init__(self) -> None:
        if self.resolution_mode not in ("causal", "lookahead"):
            raise ValueError(
                "BackpressureConfig.resolution_mode must be 'causal' or 'lookahead', "
                f"got {self.resolution_mode!r}"
            )
        if self.age_axis not in ("control_tick", "source_time", "wallclock"):
            raise ValueError(
                "BackpressureConfig.age_axis must be 'control_tick', 'source_time' or "
                f"'wallclock', got {self.age_axis!r}"
            )
        if self.pending_max_count < 1:
            raise ValueError("pending_max_count must be >= 1")
        if self.admission_window_control_ticks < 1:
            raise ValueError("admission_window_control_ticks must be >= 1")
        if self.max_admissions_per_window < 0:
            raise ValueError("max_admissions_per_window must be >= 0")
        if self.representatives_per_segment < 1:
            raise ValueError("representatives_per_segment must be >= 1")
        if self.frontier_quota < 0:
            raise ValueError("frontier_quota must be >= 0")
        if self.decision_ledger_size < 1:
            raise ValueError("decision_ledger_size must be >= 1")

    @property
    def deterministic(self) -> bool:
        """True iff no decision in this config can read a host clock."""
        return self.age_axis != "wallclock"

    def as_report_dict(self) -> Dict[str, object]:
        """The resolved bounds, for ``report.json``."""
        return {
            "enabled": bool(self.enabled),
            "force_all_stress": bool(self.force_all_stress),
            "resolution_mode": str(self.resolution_mode),
            "pending_max_count": int(self.pending_max_count),
            "pending_max_age_control_ticks": int(self.pending_max_age_control_ticks),
            "pending_max_age_source_s": float(self.pending_max_age_source_s),
            "pending_max_age_ms": float(self.pending_max_age_ms),
            "age_axis": str(self.age_axis),
            "deterministic": bool(self.deterministic),
            "max_admissions_per_window": int(self.max_admissions_per_window),
            "admission_window_control_ticks": int(self.admission_window_control_ticks),
            "frontier_quota": int(self.frontier_quota),
            "keyframe_quota": int(self.keyframe_quota),
            "representatives_per_segment": int(self.representatives_per_segment),
            "novelty_protect": float(self.novelty_protect),
            "quality_protect": float(self.quality_protect),
            "redundant_translation_m": float(self.redundant_translation_m),
            "redundant_rotation_deg": float(self.redundant_rotation_deg),
            "causal_admit_translation_m": float(self.causal_admit_translation_m),
            "causal_admit_rotation_deg": float(self.causal_admit_rotation_deg),
            "causal_deltas_are_cumulative": bool(self.causal_deltas_are_cumulative),
            "decision_ledger_size": int(self.decision_ledger_size),
            "rising_fill_frac": float(self.rising_fill_frac),
            "high_fill_frac": float(self.high_fill_frac),
            "score_weights": {
                "novelty": float(self.novelty_weight),
                "quality": float(self.quality_weight),
                "motion": float(self.motion_weight),
            },
        }


def backpressure_config_from_args(args) -> BackpressureConfig:
    """Resolve a :class:`BackpressureConfig` from parsed training args.

    Every default here must match ``arguments/__init__.py`` exactly --
    ``tests/test_getattr_fallback_freshness.py`` audits that invariant
    repo-wide, and a drifted literal is precisely the class of silent bug
    that audit exists to catch.
    """
    legacy_fields = (
        "streaming_backpressure_pending_max_age_iters",
        "streaming_backpressure_admission_window_iters",
    )
    present_legacy = tuple(name for name in legacy_fields if hasattr(args, name))
    if present_legacy:
        raise ValueError(
            "legacy optimizer-iteration backpressure fields are not accepted by "
            "the canonical owner; migrate to "
            "streaming_backpressure_pending_max_age_control_ticks and "
            "streaming_backpressure_admission_window_control_ticks: "
            + ", ".join(present_legacy)
        )
    return BackpressureConfig(
        enabled=bool(getattr(args, "streaming_backpressure", False)),
        force_all_stress=bool(
            getattr(args, "streaming_backpressure_force_all_stress", False)
        ),
        resolution_mode=str(
            getattr(args, "streaming_backpressure_resolution_mode", "causal")
        ),
        pending_max_count=int(
            getattr(args, "streaming_backpressure_pending_max_count", 8)
        ),
        pending_max_age_control_ticks=int(
            getattr(args, "streaming_backpressure_pending_max_age_control_ticks", 600)
        ),
        max_admissions_per_window=int(
            getattr(args, "streaming_backpressure_max_admissions_per_window", 4)
        ),
        admission_window_control_ticks=int(
            getattr(args, "streaming_backpressure_admission_window_control_ticks", 100)
        ),
        frontier_quota=int(getattr(args, "streaming_backpressure_frontier_quota", 1)),
        keyframe_quota=int(getattr(args, "streaming_backpressure_keyframe_quota", 1)),
        representatives_per_segment=int(
            getattr(args, "streaming_backpressure_representatives", 1)
        ),
        novelty_protect=float(
            getattr(args, "streaming_backpressure_novelty_protect", 0.85)
        ),
        quality_protect=float(
            getattr(args, "streaming_backpressure_quality_protect", 0.95)
        ),
        redundant_translation_m=float(
            getattr(args, "streaming_backpressure_redundant_translation_m", 0.05)
        ),
        redundant_rotation_deg=float(
            getattr(args, "streaming_backpressure_redundant_rotation_deg", 5.0)
        ),
        causal_admit_translation_m=float(
            getattr(args, "streaming_backpressure_admit_translation_m", 0.10)
        ),
        causal_admit_rotation_deg=float(
            getattr(args, "streaming_backpressure_admit_rotation_deg", 10.0)
        ),
        # train_streaming feeds _pose_delta_from_last_train(), which is
        # already cumulative since the last admitted frame.
        causal_deltas_are_cumulative=True,
        decision_ledger_size=int(
            getattr(args, "streaming_backpressure_decision_ledger_size", 4096)
        ),
    )


def burst_config_as_signal_provider(burst_config):
    """Neutralize ``BurstBudget``'s overlapping drop execution.

    #601's single-owner requirement: once :class:`BackpressureOwner` is
    enabled, ``BurstBudget`` must either become a thin signal provider or
    have its overlapping drop execution removed. This helper implements
    the first option -- it returns a copy of *burst_config* whose
    non-keyframe drop policy can never fire, leaving only its
    ``max_per_iter`` cadence cap (a genuine, non-overlapping ingestion
    signal) in force.

    Returns the config unchanged if it is not a ``BurstConfig``-shaped
    object (so a caller can pass ``None`` through safely).
    """
    if burst_config is None:
        return None
    try:
        return replace(
            burst_config,
            drop_nonkf_backlog=2**31 - 1,
            drop_nonkf_in_catch_up=False,
        )
    except TypeError:
        return burst_config


# ----------------------------------------------------------------------
# Bounded pending cohort
# ----------------------------------------------------------------------
class PendingObservationCohort:
    """The one scheduler-owned bounded pending structure.

    Holds :class:`PendingObservation` metadata records only -- never a
    materialized camera, image tensor, or depth map. Its size is hard
    bounded by :attr:`BackpressureConfig.pending_max_count`, so memory
    cannot scale with source length however long the run is.

    This is a container with bookkeeping; all policy lives in
    :class:`BackpressureOwner`.
    """

    def __init__(self, config: BackpressureConfig) -> None:
        self._config = config
        self._items: List[PendingObservation] = []

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self):
        return iter(self._items)

    @property
    def items(self) -> Tuple[PendingObservation, ...]:
        """Return all pending observations as a tuple."""
        return tuple(self._items)

    @property
    def oldest(self) -> Optional[PendingObservation]:
        """Return the oldest pending observation or None if empty."""
        return self._items[0] if self._items else None

    @property
    def newest(self) -> Optional[PendingObservation]:
        """Return the newest pending observation or None if empty."""
        return self._items[-1] if self._items else None

    def append(self, obs: PendingObservation) -> None:
        """Add an observation to the cohort."""
        self._items.append(obs)

    def pop_oldest(self) -> PendingObservation:
        """Remove and return the oldest pending observation."""
        return self._items.pop(0)

    def retain(self, frame_ids) -> None:
        """Keep only the observations whose ``frame_id`` is in *frame_ids*,
        preserving source order."""
        keep = set(frame_ids)
        self._items = [o for o in self._items if o.frame_id in keep]

    def age_control_ticks(self, control_tick: int) -> int:
        """Return the age of the oldest observation in control ticks."""
        oldest = self.oldest
        return 0 if oldest is None else max(0, int(control_tick) - int(oldest.release_control_tick))

    def age_source_s(self, source_time_s: float) -> float:
        """Return the age of the oldest observation in source seconds."""
        oldest = self.oldest
        if oldest is None:
            return 0.0
        return max(0.0, float(source_time_s) - float(oldest.source_timestamp))

    def age_wallclock_ms(self, wallclock_now: float) -> float:
        """Return the age of the oldest observation in wall-clock milliseconds."""
        oldest = self.oldest
        if oldest is None:
            return 0.0
        return max(0.0, (float(wallclock_now) - float(oldest.wallclock_s)) * 1000.0)


# ----------------------------------------------------------------------
# Telemetry
# ----------------------------------------------------------------------
class BackpressureTelemetry:
    """Per-run aggregate accounting for the backpressure owner.

    Everything #599's "Required telemetry" section asks for at the
    aggregate level: queue-depth quantiles, max/p95 pending age, skipped
    fraction by reason, protected/frontier retention, and the resolved
    config bounds.
    """

    def __init__(self, config: BackpressureConfig) -> None:
        self._config = config
        self.n_observed = 0
        self.n_by_action: Dict[str, int] = {a.value: 0 for a in BackpressureAction}
        self.n_by_reason: Dict[str, int] = {}
        self.n_protected = 0
        self.n_forced_frontier_admits = 0
        self._depths: List[int] = []
        self._age_control_ticks: List[int] = []
        self._age_source_s: List[float] = []

    def record(self, decision: BackpressureDecision) -> None:
        """Record a backpressure decision for telemetry aggregation."""
        self.n_observed += 1
        self.n_by_action[decision.action.value] = (
            self.n_by_action.get(decision.action.value, 0) + 1
        )
        self.n_by_reason[decision.reason] = self.n_by_reason.get(decision.reason, 0) + 1
        if decision.reason in PROTECTION_REASON_CODES:
            self.n_protected += 1
        if decision.reason == REASON_FORCED_FRONTIER_ADMIT:
            self.n_forced_frontier_admits += 1
        self._depths.append(int(decision.pending_depth))
        self._age_control_ticks.append(int(decision.pending_age_control_ticks))
        self._age_source_s.append(float(decision.pending_age_source_s))

    @staticmethod
    def _quantile(values: Sequence[float], q: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        idx = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
        return float(ordered[idx])

    def report_block(self) -> Dict[str, object]:
        """The ``report.json["backpressure"]`` payload."""
        admitted = self.n_by_action.get(BackpressureAction.ADMIT.value, 0)
        skipped = self.n_observed - admitted
        denom = float(self.n_observed) if self.n_observed else 1.0
        return {
            "config": self._config.as_report_dict(),
            "owner": "BackpressureOwner",
            "single_owner_enforced": bool(
                self._config.enabled and not self._config.force_all_stress
            ),
            "stress_mode": bool(self._config.force_all_stress),
            "stress_mode_warning": (
                "FORCE-ALL STRESS MODE: the bounded backpressure owner was bypassed; "
                "mapping backlog was intentionally unbounded. Not a production result."
                if self._config.force_all_stress
                else ""
            ),
            "n_observed": int(self.n_observed),
            "n_admitted": int(admitted),
            "n_skipped": int(skipped),
            "skipped_fraction": float(skipped) / denom,
            "decisions_by_action": dict(self.n_by_action),
            "decisions_by_reason": dict(self.n_by_reason),
            "n_protected_retained": int(self.n_protected),
            "n_forced_frontier_admits": int(self.n_forced_frontier_admits),
            "pending_depth_p50": self._quantile(self._depths, 0.5),
            "pending_depth_p95": self._quantile(self._depths, 0.95),
            "pending_depth_max": float(max(self._depths)) if self._depths else 0.0,
            "pending_age_control_ticks_p95": self._quantile(self._age_control_ticks, 0.95),
            "pending_age_control_ticks_max": (
                float(max(self._age_control_ticks)) if self._age_control_ticks else 0.0
            ),
            "pending_age_source_s_p95": self._quantile(self._age_source_s, 0.95),
            "pending_age_source_s_max": (
                float(max(self._age_source_s)) if self._age_source_s else 0.0
            ),
        }


# ----------------------------------------------------------------------
# The owner
# ----------------------------------------------------------------------
class BackpressureOwner:
    """The single execution owner of skip/coalesce backpressure.

    Usage from a drain loop, at the one point where a source frame has
    become due but nothing expensive has happened yet::

        for decision in owner.submit(obs, control_tick=epoch, source_time_s=ts):
            if decision.materializes:
                materialize_and_train(decision.frame_id)
            else:
                record_metadata_only(decision)   # pose/history only

    :meth:`submit` returns *zero or more* decisions because the owner
    holds observations in a bounded cohort: submitting frame ``N`` may
    resolve nothing (still pending), may resolve ``N`` immediately (soft
    all under LOW pressure), or may resolve a whole compressed segment
    of earlier frames at once. Every ``frame_id`` gets **exactly one**
    decision over the life of the run, and that decision is terminal --
    no deferral path may resurrect a skipped frame.
    """

    def __init__(self, config: Optional[BackpressureConfig] = None) -> None:
        self._config = config or BackpressureConfig()
        self._cohort = PendingObservationCohort(self._config)
        self._telemetry = BackpressureTelemetry(self._config)
        # Count admissions per control epoch. Multiple source observations may
        # share an epoch, so storing one list entry per observation would make
        # memory scale with source burst size while the control clock is still.
        self._admission_control_ticks: "OrderedDict[int, int]" = OrderedDict()
        # Terminal-decision ledger. Bounded to the most recent
        # `decision_ledger_size` frame_ids: a defer/retry path only ever
        # re-offers a RECENT frame, so an unbounded ledger would be the one
        # structure in this module that still scales with source length --
        # exactly what #601 forbids. `decision_for()` returns None for an
        # evicted frame; see its docstring.
        self._decided: "OrderedDict[int, BackpressureAction]" = OrderedDict()
        self._last_admitted: Optional[PendingObservation] = None
        # --- causal-mode state (no buffering; see `resolution_mode`) ---
        self._last_admit_control_tick: Optional[int] = None
        self._last_admit_source_time_s: Optional[float] = None
        self._last_admit_wallclock_s: Optional[float] = None
        self._consecutive_skips: int = 0
        self._accum_translation_m: float = 0.0
        self._accum_rotation_deg: float = 0.0
        self._causal_kf_since_admit: int = 0
        self._last_control_tick: Optional[int] = None
        self._last_source_time_s: Optional[float] = None
        self._last_wallclock_s: Optional[float] = None
        # Why a still-pending observation survived its compression pass.
        # Carried forward so the eventual ADMIT records the real reason
        # (``segment_representative`` / ``protected_*``) rather than the
        # generic soft-all one.
        self._keep_reasons: Dict[int, str] = {}

    # -- accessors ----------------------------------------------------
    @property
    def config(self) -> BackpressureConfig:
        """Return the backpressure configuration."""
        return self._config

    @property
    def cohort(self) -> PendingObservationCohort:
        """Return the pending observation cohort."""
        return self._cohort

    @property
    def telemetry(self) -> BackpressureTelemetry:
        """Return the backpressure telemetry accumulator."""
        return self._telemetry

    @property
    def pending_depth(self) -> int:
        """Return the number of pending observations."""
        return len(self._cohort)

    def decision_for(self, frame_id: int) -> Optional[BackpressureAction]:
        """The terminal action already recorded for *frame_id*, if any.

        A drain/defer path must consult this before re-offering a frame:
        a decision is terminal, and re-admitting a skipped frame would
        reintroduce the multi-owner behaviour #601 removes.

        Returns ``None`` both for a frame never seen and for one whose
        entry has aged out of the bounded ledger (see
        :attr:`BackpressureConfig.decision_ledger_size`). The bound is
        deliberate -- the ledger is the only per-frame structure here, and
        letting it grow with source length would defeat the memory bound
        this module exists to provide. A defer path only ever re-offers a
        recent frame, so an evicted entry means "far older than any retry
        window", not "safe to re-admit"; callers that need an absolute
        guarantee for the whole run must raise the bound explicitly.
        """
        return self._decided.get(int(frame_id))

    def _observe_clocks(
        self,
        *,
        control_tick: int,
        source_time_s: Optional[float] = None,
        wallclock_s: Optional[float] = None,
    ) -> tuple[int, Optional[float], Optional[float]]:
        """Validate monotone semantic clocks before mutating policy state."""

        tick = _control_tick(control_tick)
        if self._last_control_tick is not None and tick < self._last_control_tick:
            raise ValueError(
                "control_tick must be monotonic: "
                f"previous={self._last_control_tick}, current={tick}"
            )
        source = None
        if source_time_s is not None:
            source = _finite_seconds(source_time_s, "source_time_s")
            if (
                self._last_source_time_s is not None
                and source < self._last_source_time_s
            ):
                raise ValueError(
                    "source_time_s must be monotonic: "
                    f"previous={self._last_source_time_s}, current={source}"
                )
        wallclock = None
        if wallclock_s is not None:
            wallclock = _finite_seconds(wallclock_s, "wallclock_now")
            if (
                self._last_wallclock_s is not None
                and wallclock < self._last_wallclock_s
            ):
                raise ValueError(
                    "wallclock_now must be monotonic: "
                    f"previous={self._last_wallclock_s}, current={wallclock}"
                )
        self._last_control_tick = tick
        if source is not None:
            self._last_source_time_s = source
        if wallclock is not None:
            self._last_wallclock_s = wallclock
        return tick, source, wallclock

    # -- pressure -----------------------------------------------------
    def _prune_admission_window(self, control_tick: int) -> None:
        cutoff = int(control_tick) - int(self._config.admission_window_control_ticks)
        while self._admission_control_ticks:
            oldest_tick = next(iter(self._admission_control_ticks))
            if oldest_tick > cutoff:
                break
            self._admission_control_ticks.popitem(last=False)

    def _admissions_in_window(self, control_tick: int) -> int:
        cutoff = int(control_tick) - int(self._config.admission_window_control_ticks)
        return sum(
            count
            for tick, count in self._admission_control_ticks.items()
            if tick > cutoff
        )

    def _budget_remaining(self, control_tick: int) -> int:
        return max(
            0,
            int(self._config.max_admissions_per_window)
            - self._admissions_in_window(control_tick),
        )

    def pressure_state(
        self,
        control_tick: int,
        *,
        source_time_s: float = 0.0,
        wallclock_now: float = 0.0,
    ) -> PressureState:
        """Explicit, thresholded pressure bucket from named components."""
        control_tick = _control_tick(control_tick)
        cfg = self._config
        fill = len(self._cohort) / float(cfg.pending_max_count)
        over_age = self._over_age(control_tick, source_time_s, wallclock_now)
        if over_age or fill >= 1.0:
            return PressureState.CRITICAL
        if fill >= cfg.high_fill_frac:
            return PressureState.HIGH
        if fill >= cfg.rising_fill_frac or self._budget_remaining(control_tick) == 0:
            return PressureState.RISING
        return PressureState.LOW

    def _over_age(
        self,
        control_tick: int,
        source_time_s: float = 0.0,
        wallclock_now: float = 0.0,
    ) -> bool:
        cfg = self._config
        if not len(self._cohort):
            return False
        if cfg.age_axis == "wallclock":
            if cfg.pending_max_age_ms > 0.0:
                return self._cohort.age_wallclock_ms(wallclock_now) > cfg.pending_max_age_ms
            return False
        if cfg.age_axis == "source_time":
            if cfg.pending_max_age_source_s > 0.0:
                return self._cohort.age_source_s(source_time_s) > cfg.pending_max_age_source_s
            return False
        return (
            cfg.pending_max_age_control_ticks > 0
            and self._cohort.age_control_ticks(control_tick)
            > cfg.pending_max_age_control_ticks
        )

    # -- scoring ------------------------------------------------------
    def _score(self, obs: PendingObservation) -> float:
        cfg = self._config
        return (
            cfg.novelty_weight * float(obs.novelty)
            + cfg.quality_weight * float(obs.quality)
            + cfg.motion_weight * float(obs.motion)
        )

    def _protection_reason(self, obs: PendingObservation) -> Optional[str]:
        cfg = self._config
        if obs.novelty >= cfg.novelty_protect:
            return REASON_PROTECTED_NOVELTY
        if obs.quality >= cfg.quality_protect:
            return REASON_PROTECTED_QUALITY
        return None

    # -- decision emission --------------------------------------------
    def _emit(
        self,
        obs: PendingObservation,
        action: BackpressureAction,
        reason: str,
        *,
        control_tick: int,
        source_time_s: float,
        pressure: PressureState,
        representative_frame_id: Optional[int] = None,
        pending_depth: Optional[int] = None,
        pending_age_control_ticks: Optional[int] = None,
        pending_age_source_s: Optional[float] = None,
    ) -> BackpressureDecision:
        admit = action is BackpressureAction.ADMIT
        decision = BackpressureDecision(
            frame_id=obs.frame_id,
            action=action,
            reason=reason,
            pressure_state=pressure,
            representative_frame_id=representative_frame_id,
            pending_depth=(
                len(self._cohort) if pending_depth is None else int(pending_depth)
            ),
            pending_age_control_ticks=(
                self._cohort.age_control_ticks(control_tick)
                if pending_age_control_ticks is None
                else int(pending_age_control_ticks)
            ),
            pending_age_source_s=(
                self._cohort.age_source_s(source_time_s)
                if pending_age_source_s is None
                else float(pending_age_source_s)
            ),
            decided_control_tick=int(control_tick),
            signals={
                "novelty": float(obs.novelty),
                "quality": float(obs.quality),
                "motion": float(obs.motion),
                "translation_m": float(obs.translation_m),
                "rotation_deg": float(obs.rotation_deg),
                "cheap_keyframe": 1.0 if obs.cheap_keyframe else 0.0,
                "score": self._score(obs),
                "budget_remaining": float(self._budget_remaining(control_tick)),
            },
            stress_mode=bool(self._config.force_all_stress),
            observe=True,
            mapping_candidate=action is not BackpressureAction.METADATA_ONLY,
            admit_supervision=admit,
            allow_geometry_insertion=admit,
        )
        self._decided[obs.frame_id] = action
        while len(self._decided) > self._config.decision_ledger_size:
            self._decided.popitem(last=False)
        if admit:
            self._prune_admission_window(control_tick)
            self._admission_control_ticks[control_tick] = (
                self._admission_control_ticks.get(control_tick, 0) + 1
            )
            self._last_admitted = obs
            self._last_admit_control_tick = int(control_tick)
            self._last_admit_source_time_s = float(source_time_s)
            self._last_admit_wallclock_s = self._last_wallclock_s
            self._consecutive_skips = 0
            self._accum_translation_m = 0.0
            self._accum_rotation_deg = 0.0
            self._causal_kf_since_admit = 0
        elif action is not BackpressureAction.METADATA_ONLY:
            self._consecutive_skips += 1
        self._telemetry.record(decision)
        return decision

    # -- main entry point ---------------------------------------------
    def submit(
        self,
        obs: PendingObservation,
        *,
        control_tick: int,
        source_time_s: Optional[float] = None,
        wallclock_now: float = 0.0,
        mapping_candidate: bool = True,
    ) -> Tuple[BackpressureDecision, ...]:
        """Observe *obs* cheaply and return every decision it resolved.

        Args:
            obs: Cheap pre-materialization metadata. Never expensive.
            control_tick: Current canonical control epoch. This clock is
                independent of optimizer progress.
            source_time_s: Current source/dataset timestamp. Defaults to
                ``obs.source_timestamp``.
            wallclock_now: Host monotonic seconds. **Only read when
                ``age_axis == "wallclock"``**; ignored otherwise, which
                is what makes offline replay deterministic.
            mapping_candidate: False when an upstream hard gate (tracking
                validity, map-id consistency, holdout reservation) has
                already ruled this frame out of mapping. Such a frame
                resolves immediately as ``metadata_only`` and never
                enters the cohort.

        Returns:
            Zero or more terminal decisions, in source order.
        """
        if source_time_s is None:
            source_time_s = float(obs.source_timestamp)
        control_tick = _control_tick(control_tick)
        source_time_s = _finite_seconds(source_time_s, "source_time_s")
        wallclock_now = _finite_seconds(wallclock_now, "wallclock_now")
        if obs.release_control_tick > control_tick:
            raise ValueError(
                "release_control_tick cannot be newer than control_tick: "
                f"release={obs.release_control_tick}, current={control_tick}"
            )
        if self._config.age_axis == "source_time" and obs.source_timestamp > source_time_s:
            raise ValueError(
                "source_timestamp cannot be newer than source_time_s when source time "
                "owns admission age"
            )
        control_tick, resolved_source_time, resolved_wallclock = self._observe_clocks(
            control_tick=control_tick,
            source_time_s=source_time_s,
            wallclock_s=wallclock_now,
        )
        assert resolved_source_time is not None
        assert resolved_wallclock is not None
        source_time_s = resolved_source_time
        wallclock_now = resolved_wallclock
        cfg = self._config

        if cfg.force_all_stress:
            return (
                self._emit(
                    obs,
                    BackpressureAction.ADMIT,
                    REASON_FORCE_ALL_STRESS,
                    control_tick=control_tick,
                    source_time_s=source_time_s,
                    pressure=PressureState.LOW,
                ),
            )
        if not cfg.enabled:
            return (
                self._emit(
                    obs,
                    BackpressureAction.ADMIT,
                    REASON_DISABLED,
                    control_tick=control_tick,
                    source_time_s=source_time_s,
                    pressure=PressureState.LOW,
                ),
            )
        if not mapping_candidate:
            return (
                self._emit(
                    obs,
                    BackpressureAction.METADATA_ONLY,
                    REASON_NOT_MAPPING_CANDIDATE,
                    control_tick=control_tick,
                    source_time_s=source_time_s,
                    pressure=self.pressure_state(
                        control_tick,
                        source_time_s=source_time_s,
                        wallclock_now=wallclock_now,
                    ),
                ),
            )

        if cfg.resolution_mode == "causal":
            return (
                self._resolve_causal(
                    obs,
                    control_tick=control_tick,
                    source_time_s=source_time_s,
                    wallclock_now=wallclock_now,
                ),
            )

        self._cohort.append(obs)
        return self._resolve(
            control_tick=control_tick,
            source_time_s=source_time_s,
            wallclock_now=wallclock_now,
        )

    # -- causal (live-compatible, non-buffering) resolution -----------
    def _causal_depth(self) -> int:
        """Virtual pending depth: consecutive observations skipped since
        the last expensive admission."""
        return int(self._consecutive_skips)

    def _causal_age_control_ticks(self, control_tick: int) -> int:
        if self._last_admit_control_tick is None:
            return 0
        return max(0, int(control_tick) - int(self._last_admit_control_tick))

    def _causal_age_source_s(self, source_time_s: float) -> float:
        if self._last_admit_source_time_s is None:
            return 0.0
        return max(0.0, source_time_s - self._last_admit_source_time_s)

    def _causal_age_wallclock_ms(self, wallclock_now: float) -> float:
        if self._last_admit_wallclock_s is None:
            return 0.0
        return max(0.0, wallclock_now - self._last_admit_wallclock_s) * 1000.0

    def causal_pressure_state(
        self,
        control_tick: int,
        *,
        source_time_s: float = 0.0,
        wallclock_now: float = 0.0,
    ) -> PressureState:
        """Return the pressure state based on virtual observation depth and age."""
        cfg = self._config
        fill = self._causal_depth() / float(cfg.pending_max_count)
        if cfg.age_axis == "source_time":
            age_over = (
                cfg.pending_max_age_source_s > 0.0
                and self._causal_age_source_s(source_time_s)
                > cfg.pending_max_age_source_s
            )
        elif cfg.age_axis == "wallclock":
            age_over = (
                cfg.pending_max_age_ms > 0.0
                and self._causal_age_wallclock_ms(wallclock_now)
                > cfg.pending_max_age_ms
            )
        else:
            age_over = (
                cfg.pending_max_age_control_ticks > 0
                and self._causal_age_control_ticks(control_tick)
                > cfg.pending_max_age_control_ticks
            )
        if age_over or fill >= 1.0:
            return PressureState.CRITICAL
        if fill >= cfg.high_fill_frac:
            return PressureState.HIGH
        if fill >= cfg.rising_fill_frac or self._budget_remaining(control_tick) == 0:
            return PressureState.RISING
        return PressureState.LOW

    def _resolve_causal(
        self,
        obs: PendingObservation,
        *,
        control_tick: int,
        source_time_s: float,
        wallclock_now: float,
    ) -> BackpressureDecision:
        """Decide *obs* immediately, with no lookahead.

        This is the formulation a live camera can actually run: the
        decision for an observation is made at the moment it is observed,
        using only what is already known. Representatives are therefore
        selected by *threshold crossing* on accumulated cheap signals
        rather than by argmax over a buffered segment -- deterministic,
        and with no added latency between observation and materialization.

        Policy order is the same as the lookahead path's:
        soft-all under capacity -> protect novelty/quality/keyframes ->
        protect the frontier by bounding age -> representative by
        accumulated-motion threshold -> coalesce/skip the rest.
        """
        cfg = self._config
        if cfg.causal_deltas_are_cumulative:
            # The caller already reports the delta accumulated since the
            # last ADMIT (e.g. train_streaming's _pose_delta_from_last_train),
            # so assigning rather than adding avoids double counting.
            self._accum_translation_m = float(obs.translation_m)
            self._accum_rotation_deg = float(obs.rotation_deg)
        else:
            self._accum_translation_m += float(obs.translation_m)
            self._accum_rotation_deg += float(obs.rotation_deg)
        if obs.cheap_keyframe:
            self._causal_kf_since_admit += 1

        pressure = self.causal_pressure_state(
            control_tick,
            source_time_s=source_time_s,
            wallclock_now=wallclock_now,
        )
        depth = self._causal_depth()
        age = self._causal_age_control_ticks(control_tick)
        budget = self._budget_remaining(control_tick)

        def emit(action: BackpressureAction, reason: str, rep=None):
            return self._emit(
                obs,
                action,
                reason,
                control_tick=control_tick,
                source_time_s=source_time_s,
                pressure=pressure,
                representative_frame_id=rep,
                pending_depth=depth,
                pending_age_control_ticks=age,
                pending_age_source_s=(
                    0.0
                    if self._last_admitted is None
                    else max(
                        0.0,
                        float(source_time_s) - float(self._last_admitted.source_timestamp),
                    )
                ),
            )

        # (0) Soft-all: under LOW pressure with capacity available, admit
        #     useful observations liberally. This is production `all`.
        if pressure is PressureState.LOW and budget > 0:
            return emit(BackpressureAction.ADMIT, REASON_SOFT_ALL)

        # (1)/(3) Protection: exceptional novelty/quality survives overload.
        protect = self._protection_reason(obs)
        if protect is not None:
            return emit(BackpressureAction.ADMIT, protect)
        if (
            obs.cheap_keyframe
            and self._causal_kf_since_admit <= max(0, cfg.keyframe_quota)
            and budget > 0
        ):
            return emit(BackpressureAction.ADMIT, REASON_PROTECTED_KEYFRAME)

        # (2) Protect the newest frontier: bound pending depth and map age
        #     even when the admission budget is exhausted, so source->map
        #     age cannot grow without bound.
        if depth >= cfg.pending_max_count or pressure is PressureState.CRITICAL:
            return emit(BackpressureAction.ADMIT, REASON_FORCED_FRONTIER_ADMIT)

        # (4) Representative selection by accumulated-motion threshold.
        if budget > 0 and (
            self._accum_translation_m >= cfg.causal_admit_translation_m
            or self._accum_rotation_deg >= cfg.causal_admit_rotation_deg
        ):
            return emit(BackpressureAction.ADMIT, REASON_REPRESENTATIVE)

        # (5) Coalesce clearly-redundant intermediates into the last
        #     admitted representative; skip the rest.
        redundant = (
            self._accum_translation_m <= cfg.redundant_translation_m
            and self._accum_rotation_deg <= cfg.redundant_rotation_deg
        )
        if redundant and self._last_admitted is not None:
            return emit(
                BackpressureAction.COALESCE_INTO,
                REASON_COALESCED_REDUNDANT,
                rep=self._last_admitted.frame_id,
            )
        if redundant:
            return emit(BackpressureAction.SKIP_REDUNDANT, REASON_SKIP_REDUNDANT)
        return emit(BackpressureAction.SKIP_CAPACITY, REASON_SKIP_CAPACITY)

    def flush(
        self, *, control_tick: int, source_time_s: Optional[float] = None
    ) -> Tuple[BackpressureDecision, ...]:
        """Resolve everything still pending at end of stream.

        Emitted as ADMIT (the source has stopped, so there is no longer
        any backlog pressure to relieve) in source order, ignoring the
        per-window admission budget -- the budget exists to pace a live
        source, not to strand already-observed frames at shutdown.
        """
        if source_time_s is None:
            source_time_s = (
                0.0 if self._last_source_time_s is None else self._last_source_time_s
            )
        control_tick, resolved_source_time, _ = self._observe_clocks(
            control_tick=control_tick,
            source_time_s=source_time_s,
        )
        assert resolved_source_time is not None
        source_time_s = resolved_source_time
        out: List[BackpressureDecision] = []
        while len(self._cohort):
            oldest = self._cohort.pop_oldest()
            out.append(
                self._emit(
                    oldest,
                    BackpressureAction.ADMIT,
                    self._keep_reasons.pop(oldest.frame_id, REASON_SOFT_ALL),
                    control_tick=control_tick,
                    source_time_s=source_time_s,
                    pressure=PressureState.LOW,
                )
            )
        return tuple(out)

    # -- resolution ---------------------------------------------------
    def _resolve(
        self, *, control_tick: int, source_time_s: float, wallclock_now: float
    ) -> Tuple[BackpressureDecision, ...]:
        out: List[BackpressureDecision] = []
        # Bounded loop: every pass either emits at least one decision or
        # breaks, so this can never spin.
        for _ in range(len(self._cohort) + 2):
            cfg = self._config
            over_count = len(self._cohort) > cfg.pending_max_count
            over_age = self._over_age(control_tick, source_time_s, wallclock_now)
            pressure = self.pressure_state(
                control_tick,
                source_time_s=source_time_s,
                wallclock_now=wallclock_now,
            )

            if over_count or over_age:
                emitted = self._compress(
                    control_tick=control_tick,
                    source_time_s=source_time_s,
                    pressure=pressure,
                    over_age=over_age,
                )
                if emitted:
                    out.extend(emitted)
                    continue
                # Compression could not make progress (everything left is
                # protected frontier). Bound map age by force-admitting the
                # oldest, outside the per-window budget.
                if over_age and len(self._cohort):
                    oldest = self._cohort.pop_oldest()
                    self._keep_reasons.pop(oldest.frame_id, None)
                    out.append(
                        self._emit(
                            oldest,
                            BackpressureAction.ADMIT,
                            REASON_FORCED_FRONTIER_ADMIT,
                            control_tick=control_tick,
                            source_time_s=source_time_s,
                            pressure=pressure,
                        )
                    )
                    continue
                break

            # Soft-all release under the per-window admission budget.
            if len(self._cohort) and self._budget_remaining(control_tick) > 0:
                oldest = self._cohort.pop_oldest()
                out.append(
                    self._emit(
                        oldest,
                        BackpressureAction.ADMIT,
                        self._keep_reasons.pop(oldest.frame_id, REASON_SOFT_ALL),
                        control_tick=control_tick,
                        source_time_s=source_time_s,
                        pressure=pressure,
                    )
                )
                continue
            break
        return tuple(out)

    def _compress(
        self,
        *,
        control_tick: int,
        source_time_s: float,
        pressure: PressureState,
        over_age: bool,
    ) -> List[BackpressureDecision]:
        """One deterministic compression pass over the pending segment.

        Policy order (issue #601):
        1. protect the newest ``frontier_quota`` observations -- never
           compressible, which is what bounds source-to-map age;
        2. protect high-novelty / high-quality observations;
        3. protect up to ``keyframe_quota`` cheap keyframes;
        4. keep ``<= representatives_per_segment`` further
           representatives by explicit score;
        5. everything else is coalesced into its nearest kept
           representative when it is genuinely redundant (within the
           translation/rotation thresholds), otherwise skipped for
           capacity.
        """
        cfg = self._config
        items = self._cohort.items
        if not items:
            return []
        frontier_n = min(cfg.frontier_quota, len(items))
        segment = items[: len(items) - frontier_n] if frontier_n else items
        frontier = items[len(items) - frontier_n :] if frontier_n else ()
        if not segment:
            return []

        keep_reasons: Dict[int, str] = {}
        # 2. novelty/quality protection
        for obs in segment:
            reason = self._protection_reason(obs)
            if reason is not None:
                keep_reasons[obs.frame_id] = reason
        # 3. cheap-keyframe protection (deterministic: source order,
        #    highest score first, ties broken by frame_id)
        kf_candidates = [
            o
            for o in segment
            if o.cheap_keyframe and o.frame_id not in keep_reasons
        ]
        kf_candidates.sort(key=lambda o: (-self._score(o), o.frame_id))
        for obs in kf_candidates[: max(0, cfg.keyframe_quota)]:
            keep_reasons[obs.frame_id] = REASON_PROTECTED_KEYFRAME
        # 4. score-ranked representatives
        remaining = [o for o in segment if o.frame_id not in keep_reasons]
        remaining.sort(key=lambda o: (-self._score(o), o.frame_id))
        for obs in remaining[: max(0, cfg.representatives_per_segment)]:
            keep_reasons[obs.frame_id] = REASON_REPRESENTATIVE

        dropped = [o for o in segment if o.frame_id not in keep_reasons]
        if not dropped:
            # Nothing compressible: the whole segment is protected. Fall
            # back to dropping the single lowest-scoring representative
            # so the cohort stays bounded, unless that would empty the
            # protected set entirely.
            if len(keep_reasons) > 1 and (
                len(items) > cfg.pending_max_count or over_age
            ):
                ranked = sorted(segment, key=lambda o: (self._score(o), o.frame_id))
                dropped = ranked[:1]
                keep_reasons.pop(dropped[0].frame_id, None)
            else:
                return []

        kept = [o for o in segment if o.frame_id in keep_reasons]
        # 5. coalesce vs. skip
        out: List[BackpressureDecision] = []
        keep_ids = {o.frame_id for o in kept} | {o.frame_id for o in frontier}
        self._cohort.retain(keep_ids)
        # Carry each survivor's protection/representative reason forward so
        # its eventual ADMIT is attributed to the rule that actually saved
        # it, not to the generic soft-all release.
        for _fid, _reason in keep_reasons.items():
            if _fid in keep_ids:
                self._keep_reasons[_fid] = _reason
        for _obs in frontier:
            self._keep_reasons.setdefault(_obs.frame_id, REASON_PROTECTED_FRONTIER)
        self._keep_reasons = {
            k: v for k, v in self._keep_reasons.items() if k in keep_ids
        }
        for obs in dropped:
            rep = self._nearest_representative(obs, kept)
            if rep is not None and self._is_redundant(obs, rep):
                out.append(
                    self._emit(
                        obs,
                        BackpressureAction.COALESCE_INTO,
                        REASON_COALESCED_REDUNDANT,
                        control_tick=control_tick,
                        source_time_s=source_time_s,
                        pressure=pressure,
                        representative_frame_id=rep.frame_id,
                    )
                )
            elif rep is not None and self._score(obs) <= self._score(rep):
                out.append(
                    self._emit(
                        obs,
                        BackpressureAction.SKIP_REDUNDANT,
                        REASON_SKIP_REDUNDANT,
                        control_tick=control_tick,
                        source_time_s=source_time_s,
                        pressure=pressure,
                    )
                )
            else:
                out.append(
                    self._emit(
                        obs,
                        BackpressureAction.SKIP_CAPACITY,
                        REASON_SKIP_CAPACITY,
                        control_tick=control_tick,
                        source_time_s=source_time_s,
                        pressure=pressure,
                    )
                )
        return out

    @staticmethod
    def _nearest_representative(
        obs: PendingObservation, kept: Sequence[PendingObservation]
    ) -> Optional[PendingObservation]:
        """Nearest kept observation in source order (deterministic)."""
        if not kept:
            return None
        return min(kept, key=lambda o: (abs(o.frame_id - obs.frame_id), o.frame_id))

    def _is_redundant(
        self, obs: PendingObservation, rep: PendingObservation
    ) -> bool:
        cfg = self._config
        # Cheap redundancy proxy: the pose delta accumulated between the
        # two observations, approximated by the per-observation deltas
        # already recorded (no pose math, no materialization).
        return (
            abs(obs.translation_m - rep.translation_m) <= cfg.redundant_translation_m
            and abs(obs.rotation_deg - rep.rotation_deg) <= cfg.redundant_rotation_deg
        )

    # -- reporting ----------------------------------------------------
    def report_block(self) -> Dict[str, object]:
        """Return the backpressure owner's report dictionary."""
        block = self._telemetry.report_block()
        block["pending_depth_at_end"] = (
            int(self._consecutive_skips)
            if self._config.resolution_mode == "causal"
            else int(len(self._cohort))
        )
        block["coverage_note"] = (
            "Skipped observations still advance source/pose/history metadata "
            "(observe=True). Coverage accounting must count OBSERVED source frames; "
            "n_admitted is the expensive-mapping subset, not the coverage numerator."
        )
        return block
