"""Explicit admission precedence contract (issue #794).

Before this module existed, keyframe admission had **no single place** that
said which force/starvation path is allowed to bypass which gate. The
permissions were scattered as ad-hoc booleans across
``KeyframeAdmitter.admit()``: an early ``if forced: admit; return`` at the top
(``max_gap``), a ``not forced`` guard inside the parallax gate, two
independent ``*_force_admit_iters`` starvation counters (exposure, blur), and
a third (``low_overlap_force_admit_iters``) further down. Reading "can a
max-gap force skip the blur gate?" required tracing the whole function.

This module is that single place. It is **declarative and side-effect-free**:
it describes the contract and records what happened. It does not itself admit
or reject anything -- ``KeyframeAdmitter`` consults it.

Precedence, highest authority first
-----------------------------------

===  ============================  ==========================================
 #   Stage                          Who may bypass it
===  ============================  ==========================================
 0   ``INTEGRITY``                  **NOBODY.** Not a single force path.
 1   ``MOTION``                     soft max_gap, hard ceiling, novelty
                                    force (#934)
 2   ``COVERAGE_NOVELTY``           hard ceiling (soft max_gap only when the
                                    coverage machinery is unwired -- see
                                    ``max_gap_require_novelty``, PR #826)
 3   ``PARALLAX``                   soft max_gap (legacy scope), hard ceiling
 4   ``EXPOSURE``                   soft max_gap (legacy scope), hard ceiling,
                                    exposure starvation
 5   ``BLUR``                       soft max_gap (legacy scope), hard ceiling,
                                    blur starvation
 6   ``RENDER_OVERLAP``             soft max_gap (legacy scope), hard ceiling,
                                    low-overlap starvation
 7   ``QUALITY_SCORE``              soft max_gap (legacy scope), hard ceiling
===  ============================  ==========================================

``INTEGRITY`` (stage 0) is the ORB-SLAM3 map-reinit integrity veto established
by issue #806 / PR #817. It is deliberately **not** evaluated inside
``KeyframeAdmitter.admit()``: it is enforced upstream in
``StreamingScene.ingest_next_frame``'s mandatory ``integrity_observer_fn``
stage, which runs before the motion pregate, before camera construction, and
before ``admission_fn`` -- for every materialized frame, under every admission
mode. Re-checking it here would double-touch ``MapReinitTracker.ok_streak``
and double-read ``OrbslamLogReinitWatcher``'s log tail per frame (see the
rationale in ``train_streaming.py``'s ``_keyframe_admission_closure``). Its
presence in this table is therefore a *statement about the pipeline*, enforced
by a test asserting it appears in **no** force path's bypass set, plus the AST
guard in ``tests/test_issue806_integrity_observer_ordering.py``.

Soft escalation vs. hard force
------------------------------

Issue #794's core complaint is that ``max_gap`` conflated three distinct jobs:
prevent starvation, relax the motion criterion, and unconditionally authorize
a frame as good training evidence. The contract splits them:

* ``SOFT_MAX_GAP`` -- "cheap motion filtering may no longer suppress this
  candidate". Under ``MaxGapForceScope.MOTION_ONLY`` it bypasses the motion
  gate **only**; every information/quality gate still binds. Bypassing motion
  means motion is *satisfied*, not skipped -- the candidate is treated as
  ``moved`` for the downstream ``overlap_ok and (moved or novel)`` conjunction,
  otherwise "fall through to the render gate" would silently reinstate the
  starvation the backstop exists to prevent.
* ``HARD_MAX_GAP_CEILING`` -- the one explicitly-named unconditional force
  (``gap >= max_gap * max_gap_hard_mult``). It may bypass every gate except
  ``INTEGRITY``, and it is separately named, counted and reported.
* The three starvation reliefs (``EXPOSURE_STARVATION``, ``BLUR_STARVATION``,
  ``LOW_OVERLAP_STARVATION``) each bypass exactly their own gate. They were
  already conformant before #794; the table pins that.
* ``NOVELTY_FORCE`` (#934) is the odd one out and deliberately so: every other
  member above fires because evidence is *absent* (starvation), while this one
  fires because direct coverage evidence is *present* (the candidate's measured
  new-voxel fraction reached ``novelty_force_frac``). Before it existed, the
  coverage/novelty signal could only ever *veto* -- it withdrew a max_gap force
  (PR #826) or rejected below ``min_new_voxel_frac``, and the motion pregate
  short-circuited before it was even consulted for a non-moving candidate.
  Two fr3 GPU runs measured the consequence: ~96% of admissions owned by the
  motion gate, 4-5 ever owned by novelty. This trigger makes the gate able to
  CAUSE an admission, bypassing ``MOTION`` alone.

Default behaviour is unchanged
------------------------------

``MaxGapForceScope.LEGACY_ALL_GATES`` is the default and reproduces
``main``'s behaviour byte-for-byte: a soft max_gap force still short-circuits
to admission. What changes on a default run is that the bypass is now
**recorded** (``PrecedenceTrace``) instead of being invisible.
``MOTION_ONLY`` is the #794 contract and must be opted into with
``--streaming_keyframe_max_gap_force_scope motion_only``.

Which gates are actually live on a default run
----------------------------------------------

Measured against ``arguments/__init__.py`` (2026-08-16 CLI defaults, not the
library-safe dataclass defaults): ``coverage_gain_gate=True``,
``max_gap_require_novelty=True``, ``motion_pregate=True``,
``velocity_adaptive=True``; ``parallax_gate``, ``exposure_gate``,
``blur_gate``, ``quality_enabled`` all **False** and
``low_overlap_force_admit_iters=0``. So on a stock run the only gates a soft
max_gap force actually skips today are ``RENDER_OVERLAP`` (live) and, when the
coverage machinery is unwired, ``COVERAGE_NOVELTY``. The exposure/blur/
parallax/quality bypasses are **latent** -- real on opt-in specs, inert on a
default run. The trace records enabled-vs-disabled separately for exactly this
reason (``bypassed_gates`` counts only gates that were actually enabled).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, FrozenSet, List, Mapping, Optional


class Gate(str, Enum):
    """One decision stage, in precedence order (lowest ordinal = highest
    authority). String-valued so traces serialise straight to JSON."""

    INTEGRITY = "integrity"
    MOTION = "motion"
    COVERAGE_NOVELTY = "coverage_novelty"
    PARALLAX = "parallax"
    EXPOSURE = "exposure"
    BLUR = "blur"
    RENDER_OVERLAP = "render_overlap"
    QUALITY_SCORE = "quality_score"


#: Gates in precedence order, highest authority first.
GATE_ORDER: List[Gate] = [
    Gate.INTEGRITY,
    Gate.MOTION,
    Gate.COVERAGE_NOVELTY,
    Gate.PARALLAX,
    Gate.EXPOSURE,
    Gate.BLUR,
    Gate.RENDER_OVERLAP,
    Gate.QUALITY_SCORE,
]

#: Gates enforced OUTSIDE ``KeyframeAdmitter.admit()`` (see module docstring).
UPSTREAM_GATES: FrozenSet[Gate] = frozenset({Gate.INTEGRITY})


class ForcePath(str, Enum):
    """A candidate trigger: how this frame entered the decision pipeline."""

    #: Ordinary candidate. Bypasses nothing.
    NORMAL = "normal"
    #: Soft starvation escalation: ``gap >= max_gap``.
    SOFT_MAX_GAP = "soft_max_gap"
    #: The single explicitly-named unconditional force:
    #: ``gap >= max_gap * max_gap_hard_mult``.
    HARD_MAX_GAP_CEILING = "hard_starvation_ceiling"
    #: ``exposure_force_admit_iters`` elapsed since the last admission.
    EXPOSURE_STARVATION = "exposure_starvation"
    #: ``blur_gate_force_admit_iters`` elapsed since the last admission.
    BLUR_STARVATION = "blur_starvation"
    #: ``low_overlap_force_admit_iters`` elapsed since the last admission.
    LOW_OVERLAP_STARVATION = "low_overlap_starvation"
    #: Issue #934: direct coverage evidence -- this candidate's measured
    #: new-voxel fraction reached ``novelty_force_frac``. The ONLY trigger in
    #: this table that fires *because of* evidence rather than because of its
    #: absence (every other member is a starvation/backstop relief), which is
    #: why it gets its own member rather than sharing one with #313's floors
    #: (see #936). Bypasses ``MOTION`` alone; every information/quality gate
    #: still binds.
    NOVELTY_FORCE = "novelty_force"


class MaxGapForceScope(str, Enum):
    """What a *soft* ``max_gap`` force is permitted to bypass."""

    #: Pre-#794 behaviour: a soft force short-circuits straight to admission,
    #: skipping every remaining gate. DEFAULT -- changes nothing on existing runs.
    LEGACY_ALL_GATES = "legacy_all_gates"
    #: The #794 contract: a soft force relaxes the motion criterion only;
    #: information/quality gates still bind.
    MOTION_ONLY = "motion_only"


#: Gates every non-hard force path may bypass, independent of scope.
_STARVATION_BYPASS: Mapping[ForcePath, FrozenSet[Gate]] = {
    ForcePath.NORMAL: frozenset(),
    ForcePath.EXPOSURE_STARVATION: frozenset({Gate.EXPOSURE}),
    ForcePath.BLUR_STARVATION: frozenset({Gate.BLUR}),
    ForcePath.LOW_OVERLAP_STARVATION: frozenset({Gate.RENDER_OVERLAP}),
    # #934: novelty force relaxes the cheap MOTION prior only. It is
    # scope-independent by construction -- its bypass set is already the
    # minimal one MOTION_ONLY grants, so no MaxGapForceScope can widen it.
    # RENDER_OVERLAP is deliberately EXCLUDED: see the "Decision:
    # RENDER_OVERLAP is NOT in the bypass set" section of
    # docs/architecture/2026-08-16-issue934-novelty-driven-admission.md for the
    # measured evidence (168/168 rendered candidates on the fr3 Stage 1a
    # substrate cleared min_overlap with ~4x margin, so the AND-term does not
    # bind there) and for the triggered-vs-decisive counters that make a
    # scene where it DOES bind visible instead of silent.
    ForcePath.NOVELTY_FORCE: frozenset({Gate.MOTION}),
}

#: Every gate a hard force may bypass: all of them except the upstream
#: integrity veto. Written as a derivation, not a literal list, so a gate
#: added to ``GATE_ORDER`` later is covered automatically -- and so
#: ``INTEGRITY`` can never be silently added to a bypass set.
_HARD_FORCE_BYPASS: FrozenSet[Gate] = frozenset(GATE_ORDER) - UPSTREAM_GATES

#: Soft max_gap, per scope. Under LEGACY_ALL_GATES it matches the hard force
#: (that IS the bug #794 names); under MOTION_ONLY it is motion alone.
_SOFT_MAX_GAP_BYPASS: Mapping[MaxGapForceScope, FrozenSet[Gate]] = {
    MaxGapForceScope.LEGACY_ALL_GATES: _HARD_FORCE_BYPASS,
    MaxGapForceScope.MOTION_ONLY: frozenset({Gate.MOTION}),
}


def bypass_permissions(
    trigger: ForcePath,
    scope: MaxGapForceScope = MaxGapForceScope.LEGACY_ALL_GATES,
) -> FrozenSet[Gate]:
    """Gates *trigger* is permitted to bypass under *scope*.

    ``Gate.INTEGRITY`` is never a member of any returned set -- that invariant
    is asserted here rather than being left to callers.
    """
    if trigger is ForcePath.HARD_MAX_GAP_CEILING:
        allowed = _HARD_FORCE_BYPASS
    elif trigger is ForcePath.SOFT_MAX_GAP:
        allowed = _SOFT_MAX_GAP_BYPASS[scope]
    else:
        allowed = _STARVATION_BYPASS[trigger]
    assert Gate.INTEGRITY not in allowed, (
        "precedence contract violated: INTEGRITY is not bypassable by "
        f"{trigger.value}"
    )
    return allowed


def may_bypass(
    trigger: ForcePath,
    gate: Gate,
    scope: MaxGapForceScope = MaxGapForceScope.LEGACY_ALL_GATES,
) -> bool:
    """True iff *trigger* may skip *gate* under *scope*."""
    return gate in bypass_permissions(trigger, scope)


def resolve_max_gap_trigger(
    gap: int, max_gap: int, hard_mult: float
) -> ForcePath:
    """Classify a frame-index *gap* into a max_gap candidate trigger."""
    if max_gap < 1:
        max_gap = 1
    if gap >= max_gap * hard_mult:
        return ForcePath.HARD_MAX_GAP_CEILING
    if gap >= max_gap:
        return ForcePath.SOFT_MAX_GAP
    return ForcePath.NORMAL


class SoftBackstopMode(str, Enum):
    """How the SOFT starvation-escalation threshold is resolved (issue #946).

    Before #946 there was only one resolved quantity -- ``max_gap`` frames --
    and the hard starvation ceiling was *derived from it*
    (``max_gap * max_gap_hard_mult``). Raising the soft threshold from 10 to
    15/20 therefore also moved the last-resort safety ceiling from 40 to
    60/80, making any experiment on the soft backstop inseparable from a
    change in the unconditional safety bound.

    This enum names the resolution rule for the soft threshold ONLY. The hard
    ceiling is resolved by a separate, independent rule (see
    :func:`resolve_backstop_thresholds`); the multiplicative coupling survives
    **only** inside :attr:`LEGACY_FRAMES`, as that mode's explicit definition.
    """

    #: DEFAULT. Historical semantics, reproduced verbatim:
    #: ``soft = max_gap`` frames, ``hard = max_gap * max_gap_hard_mult``
    #: frames. Bit-identical to pre-#946 admission on every existing config.
    LEGACY_FRAMES = "legacy_frames"
    #: Source-time semantics: the soft threshold is a multiple of the
    #: velocity controller's own target admit interval, and the hard ceiling
    #: is resolved independently. Opt-in; NOT validated on GPU (issue #947).
    TARGET_TIME = "target_time"


def coerce_soft_backstop_mode(
    value: object,
) -> "tuple[SoftBackstopMode, Optional[str]]":
    """Parse a CLI/config value into a :class:`SoftBackstopMode`.

    Returns ``(mode, warning_or_None)``. An unrecognised value falls back to
    :attr:`SoftBackstopMode.LEGACY_FRAMES` -- the behaviour-PRESERVING mode --
    and returns a warning string the caller must surface. This is the same
    direction as :func:`coerce_scope`: a typo can never silently *enable*
    production semantics that change which frames are admitted, and the
    fallback is never silent (issue #946 validation family 9).
    """
    if isinstance(value, SoftBackstopMode):
        return value, None
    try:
        return SoftBackstopMode(str(value)), None
    except ValueError:
        return (
            SoftBackstopMode.LEGACY_FRAMES,
            f"soft_backstop_mode={value!r} is not a recognised mode "
            f"({[m.value for m in SoftBackstopMode]}); falling back to "
            f"'{SoftBackstopMode.LEGACY_FRAMES.value}' (behaviour-preserving). "
            f"The requested mode did NOT take effect.",
        )


@dataclass(frozen=True)
class ResolvedBackstop:
    """The two INDEPENDENTLY resolved starvation thresholds (issue #946).

    Each threshold carries its own authoritative domain (``"frames"`` or
    ``"seconds"``) plus a derived value in the other domain for telemetry.
    Comparison always happens in the authoritative domain -- see
    :func:`resolve_backstop_trigger` -- so a derived value can never become
    load-bearing by accident.

    The key invariant this type exists to make *provable*: nothing in the hard
    ceiling's resolution reads ``soft_multiplier`` or ``soft_threshold_*``, so
    varying the soft configuration cannot move ``hard_threshold_*``. That is
    asserted directly against this function in the #946 test suite, with no
    admitter or scene state involved.
    """

    mode: str
    #: Authoritative domain of each threshold: ``"frames"`` or ``"seconds"``.
    soft_domain: str
    hard_domain: str
    soft_threshold_frames: Optional[float]
    soft_threshold_s: Optional[float]
    hard_threshold_frames: Optional[float]
    hard_threshold_s: Optional[float]
    #: ``1 / target_admit_rate_fps`` when the velocity controller is active,
    #: else the explicit non-velocity-adaptive fallback (see
    #: ``controller_target_interval_source``). ``None`` in legacy mode, which
    #: does not consume a controller target at all.
    controller_target_interval_s: Optional[float]
    controller_target_interval_source: str
    #: The multiplier ACTUALLY applied (post-normalisation).
    soft_multiplier: float
    soft_multiplier_clamped: bool
    warnings: tuple
    #: Issue #794 (gap closure, 2026-08-23). True iff the SOFT backstop is
    #: structurally capable of firing before the controller cadence it exists
    #: to backstop -- i.e. the #1044 non-preemption floor either declined to
    #: apply (``legacy_frames``, ``target_interval_frames >= hard_frames``) or
    #: the hard ceiling is not strictly beyond the soft threshold
    #: (``hard_frames <= soft_frames``). In BOTH cases the soft escalation
    #: stage can be pre-empted by (or collapsed into) the hard ceiling before
    #: the motion criterion is ever consulted -- the exact failure #794 names.
    #: Always ``False`` under ``target_time``: that mode raises
    #: :class:`BackstopConfigError` outright on the equivalent contradiction,
    #: so a resolved ``ResolvedBackstop`` in that mode can never carry it.
    soft_backstop_preemptable: bool = False

    def as_telemetry(self) -> Dict[str, object]:
        """Flat, JSON-safe view for the admission sidecar."""
        return {
            "soft_backstop_mode": self.mode,
            "soft_backstop_domain": self.soft_domain,
            "hard_backstop_domain": self.hard_domain,
            "soft_backstop_threshold_frames": self.soft_threshold_frames,
            "soft_backstop_threshold_s": self.soft_threshold_s,
            "hard_backstop_threshold_frames": self.hard_threshold_frames,
            "hard_backstop_threshold_s": self.hard_threshold_s,
            "soft_backstop_preemptable": self.soft_backstop_preemptable,
        }


class BackstopConfigError(ValueError):
    """Raised for a contradictory #946 backstop configuration.

    Deliberately raised ONLY from configuration-resolution time (admitter
    construction), never from a per-candidate decision path, and only for
    combinations that no historical spec can reach -- i.e. inside
    ``target_time``, which is opt-in and brand new. Legacy-mode
    contradictions warn instead of raising, matching this module's standing
    "never retroactively reject an existing spec" convention
    (:func:`coerce_scope`, :func:`validate_backstop_config`).
    """


def resolve_backstop_thresholds(
    *,
    mode: object = SoftBackstopMode.LEGACY_FRAMES,
    max_gap: int = 10,
    max_gap_hard_mult: float = 4.0,
    input_fps: float = 30.0,
    velocity_adaptive: bool = False,
    target_admit_rate_fps: float = 2.0,
    soft_multiplier: float = 1.25,
    hard_backstop_s: float = 0.0,
    hard_backstop_frames: float = 40.0,
    nonpreemption_floor: bool = True,
    require_reachable_motion_gate: bool = False,
) -> ResolvedBackstop:
    """Resolve the soft escalation threshold and the hard starvation ceiling
    as two INDEPENDENT quantities (issue #946).

    Soft resolution
    ---------------
    ``legacy_frames`` -- ``soft = max_gap`` source frames. Verbatim historical
    semantics.

    ``target_time`` -- ``soft = multiplier * controller_target_interval_s``
    seconds, with ``controller_target_interval_s = 1 / target_admit_rate_fps``
    when the velocity controller is active. For a NON-velocity-adaptive run
    there is no controller cadence to backstop, so rather than borrowing an
    unrelated target the fallback is the explicit source-time image of the
    legacy soft threshold, ``max_gap / input_fps`` seconds
    (``controller_target_interval_source == "legacy_max_gap_frames"``).

    Non-preemption invariant (#946 acceptance criterion 2): the soft backstop
    must be structurally incapable of firing earlier than the controller
    cadence it exists to backstop. Enforced twice, belt and braces --
    ``multiplier`` below 1 is normalised UP to 1 with a warning, and the
    resolved interval is then floored at ``controller_target_interval_s``.

    Hard resolution
    ---------------
    ``legacy_frames`` -- ``hard = max_gap * max_gap_hard_mult`` source frames.
    This multiplicative coupling is retained ONLY here, as this mode's
    explicit definition.

    ``target_time`` -- ``hard_backstop_s`` seconds when positive, else
    ``hard_backstop_frames`` source frames. **Neither reads any soft input.**
    The frame default (40) is the resolved *current effective* ceiling of the
    production profile (``max_gap=10 * max_gap_hard_mult=4``), made explicit
    so a future soft change of 15/18/22 cannot move it.

    Raises:
        BackstopConfigError: in ``target_time`` unconditionally, when the
            resolved hard ceiling is not strictly later than the resolved
            soft threshold (a hard ceiling at or before the soft escalation
            makes the soft stage unreachable, silently restoring the
            unconditional force #794 exists to prevent). In ``legacy_frames``,
            ONLY when *require_reachable_motion_gate* is True, for the two
            structurally identical contradictions that mode otherwise only
            warns about: the #1044 non-preemption floor declining to apply
            (``target_interval_frames >= hard_frames``), and
            ``hard_frames <= soft_frames``. Both raise-vs-warn cases are the
            same as :attr:`ResolvedBackstop.soft_backstop_preemptable` being
            True; *require_reachable_motion_gate* just turns that recorded
            state into a hard failure at construction time instead of a
            printed warning. Default False -- must change no existing spec's
            behaviour; opt in explicitly.
    """
    warnings: List[str] = []
    resolved_mode, mode_warning = coerce_soft_backstop_mode(mode)
    if mode_warning:
        warnings.append(mode_warning)

    fps = float(input_fps) if float(input_fps) > 0.0 else 30.0
    if float(input_fps) <= 0.0:
        warnings.append(
            f"input_fps={input_fps} is not positive; using 30.0 for "
            f"frame<->second conversion in the backstop telemetry."
        )
    gap_frames = max(int(max_gap), 1)
    hard_mult = float(max_gap_hard_mult)

    if resolved_mode is SoftBackstopMode.LEGACY_FRAMES:
        soft_frames = float(gap_frames)
        hard_frames = float(gap_frames) * hard_mult
        # --- #1044 non-preemption floor for legacy_frames --------------------
        # The `target_time` branch below enforces a non-preemption invariant
        # (#946 acceptance criterion 2): the soft backstop must be structurally
        # incapable of firing EARLIER than the controller cadence it exists to
        # back up. `legacy_frames` never had that floor, and the production
        # profile violates it: with velocity_adaptive on, the controller's
        # motion criterion is
        #
        #     effective_min_trans = velocity_ema * (1 / target_admit_rate_fps)
        #
        # i.e. it fires after the camera has travelled for one target interval
        # (input_fps / target_admit_rate_fps = 30/2 = 15 source frames), while
        # the soft backstop fires at max_gap = 10 frames. Both scale linearly
        # with scene speed, so the ratio is a scene-INDEPENDENT 15/10 = 1.5 and
        # the TRANSLATION criterion can never OWN an admission -- not merely
        # unlikely. Note `moved` is an OR of two terms and only this one is
        # velocity-scaled; the static `min_rotation_deg` (5 deg) does NOT scale,
        # so it survives and admits on rotation-heavy segments. That is why the
        # backstop owns ~78% of admissions rather than 100% (matching #792's 78%
        # and #946's 72.7%), and why `hybrid_keyframe` degenerates toward a fixed
        # every-max_gap-th decimator rather than exactly becoming one.
        #
        # Measured on a real fr3 run over the FULL 2525-frame trajectory
        # (2026-08-22, i1044_A_prefix): of 251 admissions, 197 were max_gap-owned
        # and all 54 motion-owned ones were ROTATION-driven -- the translation
        # criterion owned exactly 0. 111 candidates did satisfy
        # `t_m >= effective_min_trans`, yet none produced a translation-owned
        # admission because max_gap reached 10 frames first every time. With the
        # floor applied the same recipe yields 134 motion-owned admissions, 28 of
        # them translation-driven. admission_preflight.py has WARNED about
        # exactly this since #792 without resolving it.
        #
        # Raise the SOFT threshold only, to the controller's nominal target
        # interval. The hard ceiling stays anchored to max_gap * hard_mult per
        # #946's independence requirement — a soft change must never move the
        # last-resort starvation bound.
        #
        # The floor is applied ONLY when it lands STRICTLY BETWEEN the current
        # soft threshold and the hard ceiling. Raising `soft` at or past `hard`
        # would make the soft escalation stage unreachable and hand every
        # starvation decision to the hard ceiling -- the exact state the
        # `target_time` branch below treats as fatal (`BackstopConfigError`:
        # "makes the soft stage unreachable and restores the unconditional
        # force #794 removed"). `legacy_frames` only WARNS about that state, so
        # without this guard a default-on floor would silently produce it for
        # any config with `max_gap * hard_mult < fps / target_admit_rate_fps`
        # (e.g. max_gap=1 or 2 at the default 30/2 = 15-frame target interval).
        # Measured: at max_gap=1, hard=4 -- an unguarded floor to 15 dropped
        # forced admissions from 59/60 to 14/60, i.e. the 4-frame hard ceiling
        # took over. Production fr3 (max_gap=10, hard=40) is unaffected: the
        # floor still resolves 15 / 40, byte-identical to this PR's arm B.
        nonpreempt_frames: Optional[float] = None
        preemptable = False
        if nonpreemption_floor and velocity_adaptive:
            target_interval_frames = fps / max(float(target_admit_rate_fps), 0.01)
            if soft_frames < target_interval_frames < hard_frames:
                nonpreempt_frames = float(target_interval_frames)
                warnings.append(
                    f"legacy_frames non-preemption floor (#1044): soft backstop "
                    f"raised {soft_frames:g} -> {target_interval_frames:g} frames so "
                    f"the velocity-adaptive motion criterion (target interval "
                    f"{fps:g}/{float(target_admit_rate_fps):g} = "
                    f"{target_interval_frames:g} frames) can own an admission "
                    f"instead of always being pre-empted by max_gap. Hard ceiling "
                    f"unchanged at {hard_frames:g} frames."
                )
                soft_frames = float(target_interval_frames)
            elif target_interval_frames > soft_frames:
                # Issue #794 Gap 1: the floor is declined because it would
                # overshoot (or reach) the hard ceiling -- the soft escalation
                # stage stays reachable, but the motion criterion remains
                # pre-emptable by max_gap exactly as before #1044.
                preemptable = True
                _msg = (
                    f"legacy_frames non-preemption floor (#1044) NOT applied: the "
                    f"controller target interval ({target_interval_frames:g} frames) "
                    f"is not below the hard starvation ceiling ({hard_frames:g} "
                    f"frames = max_gap {gap_frames} * hard_mult {hard_mult:g}), so "
                    f"raising the soft threshold to it would make the soft "
                    f"escalation stage unreachable. Soft threshold left at "
                    f"{soft_frames:g} frames; the motion criterion remains "
                    f"pre-emptable here. Raise max_gap/max_gap_hard_mult, or lower "
                    f"target_admit_rate_fps, to get the floor."
                )
                if require_reachable_motion_gate:
                    raise BackstopConfigError(
                        _msg + " require_reachable_motion_gate=True refuses this "
                        "configuration instead of warning; pass "
                        "--streaming_keyframe_require_reachable_motion_gate=False "
                        "(the default) to restore the warn-only behaviour."
                    )
                warnings.append(_msg)
        if hard_frames <= soft_frames:
            # Issue #794 Gap 1 (second form): the soft escalation stage is
            # unreachable outright -- the hard ceiling fires at or before it.
            preemptable = True
            _msg = (
                f"legacy_frames: hard ceiling ({hard_frames:g} frames = "
                f"max_gap {gap_frames} * hard_mult {hard_mult:g}) is not "
                f"beyond the soft threshold ({soft_frames:g} frames); the "
                f"soft escalation stage is unreachable."
            )
            if require_reachable_motion_gate:
                raise BackstopConfigError(
                    _msg + " require_reachable_motion_gate=True refuses this "
                    "configuration instead of warning; pass "
                    "--streaming_keyframe_require_reachable_motion_gate=False "
                    "(the default) to restore the warn-only behaviour."
                )
            warnings.append(_msg)
        return ResolvedBackstop(
            mode=resolved_mode.value,
            soft_domain="frames",
            hard_domain="frames",
            soft_threshold_frames=soft_frames,
            soft_threshold_s=soft_frames / fps,
            hard_threshold_frames=hard_frames,
            hard_threshold_s=hard_frames / fps,
            controller_target_interval_s=(
                (1.0 / max(float(target_admit_rate_fps), 0.01))
                if nonpreempt_frames is not None else None
            ),
            controller_target_interval_source=(
                "target_admit_rate_fps_nonpreemption_floor"
                if nonpreempt_frames is not None else "not_applicable"
            ),
            soft_multiplier=1.0,
            soft_multiplier_clamped=False,
            warnings=tuple(warnings),
            soft_backstop_preemptable=preemptable,
        )

    # --- target_time ---------------------------------------------------
    if velocity_adaptive:
        target_interval_s = 1.0 / max(float(target_admit_rate_fps), 0.01)
        target_source = "target_admit_rate_fps"
    else:
        target_interval_s = float(gap_frames) / fps
        target_source = "legacy_max_gap_frames"

    mult = float(soft_multiplier)
    clamped = False
    if not (mult >= 1.0) or not math.isfinite(mult):
        warnings.append(
            f"soft_backstop_multiplier={soft_multiplier!r} violates the #946 "
            f"non-preemption invariant (multiplier >= 1): the soft backstop "
            f"would fire BEFORE the controller cadence it backstops. "
            f"NORMALISED to 1.0."
        )
        mult = 1.0
        clamped = True

    soft_s = max(mult * target_interval_s, target_interval_s)

    if float(hard_backstop_s) > 0.0:
        hard_domain = "seconds"
        hard_s = float(hard_backstop_s)
        hard_frames = hard_s * fps
    else:
        hard_domain = "frames"
        hard_frames = float(hard_backstop_frames)
        hard_s = hard_frames / fps

    if hard_s <= soft_s:
        raise BackstopConfigError(
            f"#946 backstop configuration is contradictory: the hard "
            f"starvation ceiling ({hard_s:.4f} s / {hard_frames:g} frames) "
            f"is not beyond the soft escalation threshold ({soft_s:.4f} s / "
            f"{soft_s * fps:g} frames). A hard ceiling at or before the soft "
            f"threshold makes the soft stage unreachable and restores the "
            f"unconditional force #794 removed. Raise "
            f"hard_backstop_s/hard_backstop_frames or lower "
            f"soft_backstop_multiplier/target_admit_rate_fps."
        )

    return ResolvedBackstop(
        mode=resolved_mode.value,
        soft_domain="seconds",
        hard_domain=hard_domain,
        soft_threshold_frames=soft_s * fps,
        soft_threshold_s=soft_s,
        hard_threshold_frames=hard_frames,
        hard_threshold_s=hard_s,
        controller_target_interval_s=target_interval_s,
        controller_target_interval_source=target_source,
        soft_multiplier=mult,
        soft_multiplier_clamped=clamped,
        warnings=tuple(warnings),
    )


def backstop_crossings(
    resolved: ResolvedBackstop,
    starvation_age_frames: int,
    starvation_age_s: Optional[float],
) -> "tuple[bool, bool]":
    """``(soft_crossed, hard_crossed)`` -- the two thresholds evaluated
    INDEPENDENTLY, each in its own authoritative domain.

    Deliberately separate from :func:`resolve_backstop_trigger`, which
    collapses the pair into a single ``ForcePath``. Deriving "soft crossed"
    from that enum as ``trigger is not NORMAL`` would be WRONG in the general
    case and would quietly reintroduce the coupling #946 exists to remove:
    the two thresholds can live in DIFFERENT domains (the default
    ``target_time`` config has a seconds-domain soft threshold and a
    frames-domain hard ceiling), so "hard fired" does not imply "soft fired".
    Concretely, with an unusable clock (``input_fps <= 0`` => no seconds-domain
    age at all) the seconds-domain soft threshold can never fire while the
    frames-domain hard ceiling still does -- and a row claiming
    ``soft_backstop_triggered`` next to a null ``starvation_age_source_s``
    is exactly the kind of self-contradictory telemetry #947's analysis
    cannot afford.

    A threshold whose domain has no usable measurement simply does not fire;
    it is never fabricated from the other domain.
    """

    def _fires(domain: str, thr_frames: Optional[float], thr_s: Optional[float]) -> bool:
        if domain == "seconds":
            if thr_s is None or starvation_age_s is None:
                return False
            return float(starvation_age_s) >= float(thr_s)
        if thr_frames is None:
            return False
        return float(starvation_age_frames) >= float(thr_frames)

    soft = _fires(
        resolved.soft_domain, resolved.soft_threshold_frames, resolved.soft_threshold_s
    )
    hard = _fires(
        resolved.hard_domain, resolved.hard_threshold_frames, resolved.hard_threshold_s
    )
    return soft, hard


def resolve_backstop_trigger(
    resolved: ResolvedBackstop,
    starvation_age_frames: int,
    starvation_age_s: Optional[float],
) -> ForcePath:
    """Classify a starvation age against a :class:`ResolvedBackstop`.

    Totality/precedence: the HARD ceiling wins outright whenever both fire.
    This is not incidental -- the hard ceiling grants a strictly wider bypass
    set (everything but ``INTEGRITY``), so a candidate that qualifies for both
    must be reported and treated as the hard case.

    This function is LOSSY by design: it answers "which force path", not
    "which thresholds were crossed". Callers that need the latter must use
    :func:`backstop_crossings` -- see its docstring for why re-deriving the
    pair from this enum is unsound.

    In ``legacy_frames`` this is exactly :func:`resolve_max_gap_trigger` with
    ``(max_gap, max_gap * hard_mult)``; that equivalence is asserted by the
    #946 legacy-reproduction test.
    """
    soft, hard = backstop_crossings(resolved, starvation_age_frames, starvation_age_s)
    if hard:
        return ForcePath.HARD_MAX_GAP_CEILING
    if soft:
        return ForcePath.SOFT_MAX_GAP
    return ForcePath.NORMAL


def coerce_scope(value: object) -> MaxGapForceScope:
    """Best-effort parse of a CLI/config value into a scope.

    Unknown values fall back to the legacy (behaviour-preserving) scope rather
    than raising, so a typo in a spec can never silently *change* admission --
    the worst case is that the opt-in does not take effect.
    """
    if isinstance(value, MaxGapForceScope):
        return value
    try:
        return MaxGapForceScope(str(value))
    except ValueError:
        return MaxGapForceScope.LEGACY_ALL_GATES


@dataclass
class PrecedenceTrace:
    """Reason chain for one admission decision (issue #794 "Decision trace
    semantics"). Key names match the issue body verbatim so this is directly
    consumable by the #802 prefilter states and the realized-policy telemetry.

    ``bypassed_gates`` counts only gates that were **enabled** for this run --
    recording "exposure bypassed" when the exposure gate is off would inflate
    every count and make the telemetry unreadable for exactly the question
    #794 asks. Gates that were permitted-but-disabled land in
    ``bypassed_gates_inactive`` instead.
    """

    candidate_trigger: str = ForcePath.NORMAL.value
    scope: str = MaxGapForceScope.LEGACY_ALL_GATES.value
    bypassed_gates: List[str] = field(default_factory=list)
    bypassed_gates_inactive: List[str] = field(default_factory=list)
    evaluated_gates: List[str] = field(default_factory=list)
    final_owner: str = "unknown"
    hard_force: bool = False
    # Issue #928: the two failure modes of a would-be-forced (gap >= max_gap)
    # candidate's ``max_gap_require_novelty`` check were byte-identical on
    # every prior artifact -- "the gate ran and found the frame novel" and
    # "the gate never ran / abstained" both left ``forced=True,
    # reason='max_gap'`` with nothing recording which happened. These three
    # fields make that abstention explicit.
    #
    # ``novelty_gate_outcome`` -- one of:
    #   "not_applicable"     -- this candidate was never a would-be max_gap
    #                            force (gap < max_gap); the gate does not
    #                            apply, and the two fields below stay None.
    #   "not_evaluated"       -- the candidate WAS a would-be force, but the
    #                            gate could not be evaluated for it -- see
    #                            ``novelty_gate_unavailable_reason``.
    #   "evaluated_novel"     -- the gate ran and the frame passed
    #                            (new-voxel-frac >= min_new_voxel_frac);
    #                            forcing stands.
    #   "evaluated_redundant" -- the gate ran and the frame failed; forcing
    #                            was withdrawn (PR #826) and the candidate
    #                            re-entered the ordinary admission path.
    novelty_gate_outcome: str = "not_applicable"
    # Populated only when novelty_gate_outcome == "not_evaluated". One of:
    # "gate_disabled" (max_gap_require_novelty or coverage_gain_gate is
    # False), "no_occupancy_fn", "no_depth_loader" (the coverage machinery
    # is unwired), "hard_ceiling" (gap already past max_gap*max_gap_hard_mult
    # -- the unconditional starvation backstop, never gated), or
    # "frac_unavailable" (the gate ran but _coverage_new_voxel_frac()
    # returned None -- no depth, too few valid points, or any error).
    novelty_gate_unavailable_reason: Optional[str] = None
    # When the coverage call itself returned ``None``, preserve the concrete
    # reason rather than collapsing a missing depth sample and an internal
    # exception into the same silent force pass.  This is additive to the
    # established unavailable reason ``frac_unavailable``.
    novelty_gate_none_cause: Optional[str] = None
    # The measured new-voxel fraction, only when novelty_gate_outcome is
    # "evaluated_novel" or "evaluated_redundant"; None otherwise.
    novelty_gate_fraction: Optional[float] = None
    # Issue #934: the novelty-DRIVEN admission trigger (ForcePath.NOVELTY_FORCE).
    #
    # Deliberately SEPARATE fields from the three ``novelty_gate_*`` ones above,
    # which are #928's telemetry for the novelty *veto* on would-be-max_gap-
    # forced candidates. That field's documented denominator is "would-be-forced
    # candidates only"; widening it in place to also cover this trigger would
    # silently change the meaning of every historical and future
    # ``novelty_gate.outcome_counts`` reading. New mechanism, new fields.
    #
    # ``novelty_force_outcome`` -- one of:
    #   "not_evaluated"   -- the trigger could not fire for this candidate; see
    #                        ``novelty_force_unavailable_reason``.
    #   "below_threshold" -- the new-voxel fraction was measured and did NOT
    #                        reach ``novelty_force_frac``.
    #   "triggered"       -- the fraction reached the threshold; the motion
    #                        criterion was treated as satisfied and the
    #                        candidate continued through every remaining gate.
    novelty_force_outcome: str = "not_evaluated"
    # Populated only when novelty_force_outcome == "not_evaluated". One of:
    # "disabled" (novelty_force_admit off), "not_normal_trigger" (this is a
    # max_gap candidate -- that trigger owns it), "motion_already_ok" (motion
    # would admit anyway, so the trigger is moot and the coverage cost is
    # skipped), "no_baseline" (pre-first-keyframe), "machinery_unwired"
    # (no occupancy_fn / load_depth_m), "stale_pose" (the fresh-pose guard
    # refused -- no positive velocity sample), or "frac_unavailable".
    novelty_force_unavailable_reason: Optional[str] = None
    # Concrete cause when the coverage call ran but yielded no usable fraction
    # (``no_depth`` / ``too_few_valid_points`` / ``exception``).  Kept apart
    # from the established coarse ``frac_unavailable`` reason.
    novelty_force_none_cause: Optional[str] = None
    # The measured new-voxel fraction, only when the trigger was evaluated
    # ("below_threshold" or "triggered"); None otherwise.
    novelty_force_fraction: Optional[float] = None
    # True iff the trigger fired AND was the actual owner of the admission
    # (final_owner == "novelty_admit"). Stamped at trace-finalize time.
    # Deliberately narrower than "triggered": a candidate that would have been
    # admitted anyway by the render-overlap ``novel`` term is attributed to
    # "new_content", not to this trigger. The GAP between the triggered count
    # and the decisive count is the diagnostic for "the trigger fires but a
    # downstream AND-term eats it" -- i.e. the veto-only trap recurring one
    # gate lower.
    novelty_force_decisive: bool = False
    # Every consumer of _coverage_new_voxel_frac records whether it obtained a
    # value, skipped on a precondition, or made a call that returned None.  A
    # single per-decision map avoids a parallel telemetry channel while keeping
    # the source of an unavailable coverage value auditable.
    coverage_consumers: Dict[str, Dict[str, object]] = field(default_factory=dict)
    # Bounded report aggregation consumes these evaluated scalar observations;
    # the per-decision JSONL remains the complete record.
    observation_values: Dict[str, float] = field(default_factory=dict)

    def record_bypass(self, gate: Gate, gate_enabled: bool) -> None:
        target = self.bypassed_gates if gate_enabled else self.bypassed_gates_inactive
        if gate.value not in target:
            target.append(gate.value)

    def record_evaluated(self, gate: Gate) -> None:
        if gate.value not in self.evaluated_gates:
            self.evaluated_gates.append(gate.value)

    def record_novelty_gate_outcome(
        self,
        outcome: str,
        unavailable_reason: Optional[str] = None,
        fraction: Optional[float] = None,
        none_cause: Optional[str] = None,
    ) -> None:
        """Issue #928: stamp the tri-state novelty-gate outcome for a
        would-be-forced candidate. Called at most once per decision."""
        self.novelty_gate_outcome = outcome
        self.novelty_gate_unavailable_reason = unavailable_reason
        self.novelty_gate_fraction = fraction
        self.novelty_gate_none_cause = none_cause

    def record_observation(self, name: str, value: Optional[float]) -> None:
        """Keep one finite, evaluated scalar under its canonical gate name."""
        if value is None:
            return
        try:
            scalar = float(value)
        except (TypeError, ValueError):
            return
        if math.isfinite(scalar):
            self.observation_values[name] = scalar

    def record_novelty_force_outcome(
        self,
        outcome: str,
        unavailable_reason: Optional[str] = None,
        fraction: Optional[float] = None,
        none_cause: Optional[str] = None,
    ) -> None:
        """Issue #934: stamp the novelty-DRIVEN trigger's outcome. Called at
        most once per decision, on every candidate (unlike #928's veto
        telemetry, which only applies to would-be-forced candidates)."""
        self.novelty_force_outcome = outcome
        self.novelty_force_unavailable_reason = unavailable_reason
        self.novelty_force_fraction = fraction
        self.novelty_force_none_cause = none_cause

    def record_coverage_consumer(
        self,
        consumer: str,
        *,
        status: str,
        value: Optional[float] = None,
        precondition_skip_reason: Optional[str] = None,
        none_cause: Optional[str] = None,
    ) -> None:
        """Stamp one existing coverage consumer's measured/skip outcome."""
        self.coverage_consumers[consumer] = {
            "status": status,
            "value": value,
            "precondition_skip_reason": precondition_skip_reason,
            "none_cause": none_cause,
        }

    def to_dict(self) -> Dict[str, object]:
        return {
            "candidate_trigger": self.candidate_trigger,
            "scope": self.scope,
            "bypassed_gates": list(self.bypassed_gates),
            "bypassed_gates_inactive": list(self.bypassed_gates_inactive),
            "evaluated_gates": list(self.evaluated_gates),
            "final_owner": self.final_owner,
            "hard_force": self.hard_force,
            "novelty_gate_outcome": self.novelty_gate_outcome,
            "novelty_gate_unavailable_reason": self.novelty_gate_unavailable_reason,
            "novelty_gate_none_cause": self.novelty_gate_none_cause,
            "novelty_gate_fraction": self.novelty_gate_fraction,
            "novelty_force_outcome": self.novelty_force_outcome,
            "novelty_force_unavailable_reason": self.novelty_force_unavailable_reason,
            "novelty_force_none_cause": self.novelty_force_none_cause,
            "novelty_force_fraction": self.novelty_force_fraction,
            "novelty_force_decisive": self.novelty_force_decisive,
            "coverage_consumers": dict(self.coverage_consumers),
            "observation_values": dict(self.observation_values),
        }


def validate_backstop_config(
    *,
    max_gap: int,
    velocity_adaptive: bool,
    input_fps: float,
    target_admit_rate_fps: float,
    scope: MaxGapForceScope,
    max_gap_require_novelty: bool,
    coverage_gain_gate: bool,
) -> List[str]:
    """Detect contradictory controller/backstop combinations.

    Returns a list of human-readable warning strings (empty = consistent).
    **Warns, never refuses** -- a hard error here would retroactively reject
    existing specs, which this project's Run Contracts treat as a serious
    hazard. ``input_fps``/``target_admit_rate_fps`` are only consumed by the
    velocity-adaptive controller, so the rate check is scoped to it rather
    than firing on fields that are inert for the run.
    """
    warnings: List[str] = []
    if velocity_adaptive and target_admit_rate_fps > 0 and input_fps > 0:
        nominal_target_interval = input_fps / target_admit_rate_fps
        if max_gap < nominal_target_interval:
            warnings.append(
                f"max_gap={max_gap} is SHORTER than the velocity-adaptive "
                f"controller's nominal target interval "
                f"({input_fps}/{target_admit_rate_fps} = "
                f"{nominal_target_interval:.1f} frames): the starvation "
                f"backstop will fire before the controller ever admits on "
                f"motion, so max_gap becomes the dominant owner of admission."
            )
    if scope is MaxGapForceScope.MOTION_ONLY and not coverage_gain_gate:
        warnings.append(
            "max_gap_force_scope=motion_only without coverage_gain_gate: a "
            "soft max_gap force will be judged by the render-overlap gate "
            "alone (the coverage/novelty gate is unwired)."
        )
    if max_gap_require_novelty and not coverage_gain_gate:
        warnings.append(
            "max_gap_require_novelty=True has no effect without "
            "coverage_gain_gate (the novelty check needs occupancy + depth "
            "wiring); soft max_gap forcing is unchanged."
        )
    return warnings


def validate_soft_backstop_config(
    resolved: ResolvedBackstop,
    *,
    velocity_adaptive: bool,
    max_gap: int,
    max_gap_hard_mult: float,
) -> List[str]:
    """Consistency warnings for a resolved #946 backstop (never raises).

    The hard-refusal cases live in :func:`resolve_backstop_thresholds` itself
    (it raises :class:`BackstopConfigError` for an unreachable soft stage
    under ``target_time``). Everything here is advisory: combinations that are
    legal but almost certainly not what the spec author meant.
    """
    out: List[str] = list(resolved.warnings)
    if resolved.mode == SoftBackstopMode.TARGET_TIME.value:
        if not velocity_adaptive:
            out.append(
                "soft_backstop_mode=target_time with velocity_adaptive=False: "
                "there is no controller cadence to backstop, so the soft "
                "threshold falls back to the explicit source-time image of "
                f"the legacy max_gap ({max_gap} frames = "
                f"{resolved.controller_target_interval_s:.4f} s) rather than "
                "borrowing target_admit_rate_fps."
            )
        _legacy_hard_frames = float(max(int(max_gap), 1)) * float(max_gap_hard_mult)
        if (
            resolved.hard_domain == "frames"
            and resolved.hard_threshold_frames is not None
            and abs(float(resolved.hard_threshold_frames) - _legacy_hard_frames) > 1e-9
        ):
            out.append(
                "soft_backstop_mode=target_time resolves a hard starvation "
                f"ceiling of {resolved.hard_threshold_frames:g} source frames, "
                f"which DIFFERS from the legacy effective ceiling "
                f"({_legacy_hard_frames:g} = max_gap {max_gap} * hard_mult "
                f"{max_gap_hard_mult:g}). This is a deliberate change to the "
                "last-resort safety bound, not a soft-timing change."
            )
    return out


__all__ = [
    "Gate",
    "GATE_ORDER",
    "UPSTREAM_GATES",
    "ForcePath",
    "MaxGapForceScope",
    "bypass_permissions",
    "may_bypass",
    "resolve_max_gap_trigger",
    "coerce_scope",
    "SoftBackstopMode",
    "coerce_soft_backstop_mode",
    "ResolvedBackstop",
    "BackstopConfigError",
    "resolve_backstop_thresholds",
    "resolve_backstop_trigger",
    "backstop_crossings",
    "PrecedenceTrace",
    "validate_backstop_config",
    "validate_soft_backstop_config",
]
