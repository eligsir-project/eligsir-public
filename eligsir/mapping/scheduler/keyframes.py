"""Centralized keyframe admission (Milestone 4, Section 5.C).

Replaces the inline :func:`_hybrid_keyframe_admission` closure in
``train_streaming.py`` with a typed, testable object that returns a
:class:`KeyframeDecision` and tracks the current keyframe list so the
scheduler / sampler can prioritise keyframes.

The two-stage admission pattern (cheap pose-delta gate, expensive
render-overlap gate) is preserved. The render call uses a small,
pluggable callable so unit tests can pass a stub that returns a
precomputed overlap without touching CUDA.

The :class:`KeyframeAdmitter` also maintains a list of recently
admitted keyframe indices that the
:class:`eligsir.mapping.scheduler.OnlineController` reads to populate
``keyframe_indices`` on the :class:`MappingDecision`.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, List, Optional, Sequence, Tuple

import numpy as np

from .keyframe_scoring import (
    KeyframeScoringConfig,
    KeyframeScore,
    score_keyframe,
)
from eligsir.mapping.scheduler.precedence import (
    GATE_ORDER,
    ForcePath,
    Gate,
    MaxGapForceScope,
    PrecedenceTrace,
    bypass_permissions,
    coerce_scope,
    validate_backstop_config,
    SoftBackstopMode,
    coerce_soft_backstop_mode,
    resolve_backstop_thresholds,
    resolve_backstop_trigger,
    backstop_crossings,
    validate_soft_backstop_config,
)


def _coerce_timestamp(ts: Any) -> Optional[float]:
    """Finite float or ``None``. Shared by the velocity controller's clock
    and the #946 starvation clock so both reject the same junk."""
    try:
        return float(ts) if ts is not None and math.isfinite(float(ts)) else None
    except (TypeError, ValueError):
        return None


_TIMEBASE_WARMUP_SAMPLES = 5
"""Consecutive-candidate timestamp samples collected before the velocity
controller commits (stickily) to trusting or rejecting the source clock.
See :meth:`KeyframeAdmitter._resolve_timebase`."""


RenderFnType = Callable[..., Any]
"""Render function signature.

The admitter invokes ``self._render_fn(cam, render_depth=...)`` and
expects a dict-like package with an ``"alpha"`` torch tensor. The
admitter itself owns ``gaussians / pipe / background`` and binds them
into the render call so callers (production or test) only need to
provide a small, testable render callable that performs the actual
shading given the current camera.
"""


@dataclass
class KeyframeDecision:
    """Outcome of one keyframe admission call.

    The velocity-controller telemetry fields (issue #793/#795) all carry
    EXPLICIT units in their names. They are populated on every call once
    ``velocity_adaptive`` is on and a velocity sample exists; they are
    recording-only and never feed any decision other than
    ``effective_min_translation_m``.

    Attributes:
        admitted: Whether the frame should be ingested as a keyframe.
        reason: Human-readable reason (``"pose_delta"``, ``"max_gap"``,
            ``"new_content"``, ``"novelty_admit"``, ``"low_overlap"``,
            ``"redundant"``, ``"quality_reject"``, ``"render_failed"``,
            ``"exposure_reject"``). ``"new_content"`` and ``"novelty_admit"``
            are DIFFERENT quantities and must not be conflated when reading
            admission ownership: ``new_content`` is render-overlap novelty
            (``overlap <= max_overlap``), ``novelty_admit`` is #934's
            depth/voxel coverage novelty trigger.
        score: Combined score (0 = reject, 1 = strong keyframe).
        cheap_score: Score from the cheap pose-delta gate only.
        render_overlap: Render-based overlap (0 = no overlap with the
            current map, 1 = full overlap), or None if not computed.
        forced: True if the frame was force-admitted by ANY force path --
            hard/soft max_gap or low-overlap starvation relief (#932). Read
            ``reason`` (or ``trace["candidate_trigger"]``) to tell which; do
            NOT read ``forced`` as implying ``reason == "max_gap"``.
        moved: True if the pose delta exceeded the threshold.
        novel: True if the overlap is below the max-overlap threshold.
        quality: Optional quality score (0 = unusable, 1 = perfect).
        keyframe_index: Train-scene index of the admitted keyframe,
            or -1 when the frame was rejected.
        pose_delta_m: Pose translation delta in metres (cheap-gate input).
        pose_delta_deg: Pose rotation delta in degrees (cheap-gate input).
        gap: Frame-index gap since the last train frame.
        translation_velocity_m_s: Instantaneous translation speed between this
            candidate and the PREVIOUS CANDIDATE (admitted or not), m/s.
        rotation_velocity_deg_s: Same for rotation, deg/s. Telemetry only --
            the rotation criterion is the static ``min_rotation_deg``.
        translation_velocity_ema_m_s: EMA of the above, m/s. This is the
            quantity the adaptive threshold is derived from.
        rotation_velocity_ema_deg_s: EMA of rotation speed, deg/s.
        source_dt_s: Physical seconds between this candidate and the previous
            one, per the resolved timebase.
        source_frame_delta: Source-frame index delta between this candidate and
            the previous one.
        timebase_source: Which clock produced ``source_dt_s`` -- one of
            ``"source_timestamp"``, ``"frame_index_nominal_fps"``,
            ``"frame_index_nominal_fps_warmup"``, ``"unavailable"``.
        implied_input_fps: Median source rate implied by the timestamps
            (frame_delta / timestamp_delta), or None when not measurable. The
            trust decision compares this against ``input_fps``.
        effective_min_translation_m: The translation threshold actually applied
            this call, in metres (post-clamp).
        controller_target_interval_s: ``1 / target_admit_rate_fps``, seconds.
        controller_target_interval_source_frames: The same configured target
            interval in original source-frame units, using the trusted
            timestamp-implied input rate when available and configured
            ``input_fps`` otherwise.
        threshold_clamp: ``"none"`` / ``"floor"`` / ``"ceil"`` / ``"n/a"``
            (``"n/a"`` = the adaptive law did not run this call).
        trace: Issue #794 precedence reason chain
            (``candidate_trigger`` / ``bypassed_gates`` / ``evaluated_gates`` /
            ``final_owner`` / ``hard_force``), see
            :mod:`eligsir.mapping.scheduler.precedence`. ``None`` only if
            a decision was constructed outside :meth:`KeyframeAdmitter.admit`.
    """

    admitted: bool
    reason: str = "unknown"
    score: float = 0.0
    cheap_score: float = 0.0
    render_overlap: Optional[float] = None
    forced: bool = False
    moved: bool = False
    novel: bool = False
    # Issue #934: the coverage/novelty trigger fired for this candidate (its
    # measured new-voxel fraction reached novelty_force_frac, so the MOTION
    # criterion was treated as satisfied). DELIBERATELY NOT folded into
    # `forced`: force_hard_count/force_soft_count -- and therefore
    # `force_share`, itself a #934 Stage 1a pass/fail gate -- are derived from
    # `forced`, and novelty force is the SELECTOR DOING ITS JOB, not starvation
    # relief. Folding it in would make a *successful* Stage 1a run fail its own
    # operating-point check. #932 was the mirror image of this (a real
    # starvation relief under-reporting itself as forced=False); do not
    # "restore symmetry" here.
    novelty_forced: bool = False
    quality: Optional[float] = None
    keyframe_index: int = -1
    pose_delta_m: float = 0.0
    pose_delta_deg: float = 0.0
    gap: int = 0
    # --- Velocity-controller telemetry (#793/#795), recording-only ---
    translation_velocity_m_s: Optional[float] = None
    rotation_velocity_deg_s: Optional[float] = None
    translation_velocity_ema_m_s: Optional[float] = None
    rotation_velocity_ema_deg_s: Optional[float] = None
    source_dt_s: Optional[float] = None
    source_frame_delta: Optional[int] = None
    timebase_source: str = "unavailable"
    implied_input_fps: Optional[float] = None
    effective_min_translation_m: float = 0.0
    controller_target_interval_s: Optional[float] = None
    controller_target_interval_source_frames: Optional[float] = None
    threshold_clamp: str = "n/a"
    # --- Issue #946 backstop telemetry (recording-only) -------------------
    #: Physical source seconds since the last admitted keyframe (or, pre-
    #: baseline, since the #933 starvation origin). ``None`` when it could
    #: not be resolved at all.
    starvation_age_source_s: Optional[float] = None
    #: The same quantity in source frames -- i.e. ``gap``, restated under the
    #: name #946 uses so the two domains sit side by side in the sidecar.
    starvation_age_source_frames: Optional[int] = None
    #: Which clock produced ``starvation_age_source_s``. DELIBERATELY a
    #: separate field from ``timebase_source``, which describes the VELOCITY
    #: CONTROLLER's clock: the two legitimately disagree (most obviously
    #: pre-baseline, where there is no last-train timestamp to diff against
    #: even when the source clock is trusted). One of ``"source_timestamp"``,
    #: ``"frame_index_nominal_fps"``, ``"unavailable"``.
    starvation_age_basis: str = "unavailable"
    soft_backstop_mode: str = SoftBackstopMode.LEGACY_FRAMES.value
    #: Which domain each threshold was COMPARED in -- ``"frames"`` or
    #: ``"seconds"``. Without these, a sidecar row carrying both a
    #: ``*_threshold_s`` and a ``*_threshold_frames`` gives no way to tell
    #: which one was authoritative (the other is a derived restatement), and
    #: the two can differ between the soft and hard halves.
    soft_backstop_domain: str = "frames"
    hard_backstop_domain: str = "frames"
    soft_backstop_threshold_s: Optional[float] = None
    soft_backstop_threshold_frames: Optional[float] = None
    hard_backstop_threshold_s: Optional[float] = None
    hard_backstop_threshold_frames: Optional[float] = None
    #: Issue #794 (gap closure). True iff the resolved soft backstop is
    #: structurally capable of firing before the motion criterion it exists
    #: to backstop can ever own an admission -- see
    #: ``ResolvedBackstop.soft_backstop_preemptable``. Identical on every row
    #: of a run (config, not per-candidate), restated here for the same
    #: self-describing-sidecar reason as the other ``*_backstop_*`` fields.
    soft_backstop_preemptable: bool = False
    #: RAW threshold crossings, recorded BEFORE any downstream withdrawal
    #: (e.g. #826's novelty veto pulling a soft force back to NORMAL). Read
    #: ``trace["candidate_trigger"]`` for the post-withdrawal classification;
    #: the pair of them is what makes a withdrawal visible.
    soft_backstop_triggered: bool = False
    hard_backstop_triggered: bool = False
    #: Which half of the motion criterion actually fired:
    #: ``"translation"`` / ``"rotation"`` / ``"both"`` / ``"none"``.
    #: Computed against the POST-clamp ``effective_min_translation_m`` this
    #: decision was judged under. Invariant: ``(motion_trigger != "none")``
    #: iff ``moved``.
    motion_trigger: str = "none"
    trace: Optional[dict] = None


@dataclass
class KeyframeAdmitterConfig:
    """Tunables for :class:`KeyframeAdmitter`.

    All thresholds default to the corresponding ``StreamingParams``
    values, so the wrapper is a thin configuration object the training
    loop instantiates once and feeds into the closure it would
    otherwise build inline.
    """

    min_translation_m: float = 0.05
    min_rotation_deg: float = 5.0
    min_overlap: float = 0.25
    max_overlap: float = 0.90
    max_gap: int = 10
    coverage_alpha: float = 0.3
    quality_enabled: bool = False
    # Admission-rate knob for the scored path (cfg.quality_enabled=True): a
    # single scalar on the weighted score (pose/rotation/overlap/new_surface/
    # render_residual/depth_disagreement/uncertainty, see keyframe_scoring.py)
    # rather than tuning min_overlap/max_gap/min_new_voxel_frac independently.
    final_threshold: float = 0.45
    quality_min_blur: float = 8.0
    quality_max_clip_ratio: float = 0.25
    quality_min_depth_valid: float = 0.10
    depth_min: float = 0.1
    depth_max: float = 8.0
    max_history: int = 1024
    # When True, frames that have NOT moved past the keyframe motion thresholds
    # (and are not force-admitted by max_gap) are rejected WITHOUT running the
    # expensive render-overlap gate. Admission-preserving for the no-motion case
    # (a non-moved frame has ~1.0 overlap and the render gate rejects it anyway),
    # and it removes the ~2.6 ms admission render from every slow-motion frame on
    # the live path (the dominant during-ingestion GPU cost). Default on (2026-07):
    # matches the CLI default (arguments/__init__.py streaming_keyframe_motion_pregate)
    # so the dataclass default no longer silently diverges from every real run.
    motion_pregate: bool = True
    # Velocity-adaptive admission: dynamically adjust the *effective*
    # min_translation_m so the admit rate stays near target_admit_rate_fps
    # regardless of scene speed. Ported from feat/offline-gap-and-floater-consensus.
    # Disabled by default; with velocity_adaptive=False the effective threshold
    # is exactly ``min_translation_m`` so the admission path is byte-identical.
    # NOTE: floor/ceil have no CLI args (they match the offline config defaults).
    #
    # ISSUE #793 (2026-08-16) -- ARITHMETIC CORRECTED. The pre-#793 law fed the
    # EMA with ``t_m``, the CUMULATIVE displacement since the last ADMITTED
    # keyframe, then multiplied it by ``frames_between = input_fps /
    # target_admit_rate_fps``. That quantity is metres-per-admission-interval,
    # not a velocity: it grew monotonically during a rejection streak and reset
    # on every admit, so ``[m/interval] x [frames/interval]`` was not a length
    # and the criterion ``v*g >= F*v*g`` could not hold for F > 1.
    #
    # The controller now samples INSTANTANEOUS velocity between CONSECUTIVE
    # CANDIDATE frames (independent of whether they were admitted), in physical
    # units, and derives the threshold as
    #
    #     effective_min_trans [m] = translation_velocity_ema [m/s]
    #                               * target_admit_interval [s]
    #
    # with ``target_admit_interval = 1 / target_admit_rate_fps``. That is
    # dimensionally a length, and crossing it means "the camera has travelled
    # about as far as it travels in one target admit interval".
    #
    # The adaptive law applies to TRANSLATION ONLY. The rotation criterion stays
    # the static ``min_rotation_deg`` (issue #793 test-contract item 7); the
    # rotation velocity IS estimated and emitted as telemetry, but it does not
    # move any threshold. Changing that would alter which branch wins, which is
    # issue #794's territory.
    velocity_adaptive: bool = False
    target_admit_rate_fps: float = 2.0   # desired admitted frames/s
    input_fps: float = 30.0              # nominal source frame rate (Hz). Used
                                         # BOTH as the fallback clock (dt =
                                         # frame_delta / input_fps) and as the
                                         # plausibility reference that decides
                                         # whether source timestamps are trusted.
    velocity_ema_alpha: float = 0.1      # EMA smoothing (lower = slower adaptation)
    velocity_adaptive_min_trans_floor: float = 0.01   # never go below this (m)
    velocity_adaptive_min_trans_ceil: float = 0.50    # never go above this (m)
    # Timebase trust band. Source timestamps are used as the controller clock
    # only when the implied source rate (median frame_delta / timestamp_delta
    # over the warmup window) lands inside
    # [input_fps / tol, input_fps * tol]; otherwise the controller falls back to
    # ``frame_delta / input_fps`` and records why. This one rule rejects every
    # broken timebase found in the audit for #793 (see the module docstring of
    # this block's implementation): index-valued "timestamps" (implied ~1 Hz vs
    # a configured 30 Hz), microsecond/millisecond/nanosecond scaling, and
    # non-monotonic junk.
    velocity_timebase_fps_tolerance: float = 2.0
    # Track A: depth-coverage gain gate (ported from feat/offline-gap). Reject
    # frames whose strided depth backprojection lands a fraction < min_new_voxel_frac
    # of points in previously-UNOCCUPIED voxels. Cheaper than the render gate; runs
    # between the cheap motion gate and the render-overlap gate. Only active when
    # coverage_gain_gate=True AND an occupancy_fn + load_depth_m are supplied; off
    # (default) the block is skipped entirely (byte-identical to before).
    coverage_gain_gate: bool = False
    min_new_voxel_frac: float = 0.05     # admit if >= this fraction of depth pts land in new voxels
    coverage_gain_depth_stride: int = 8  # coarse backprojection stride (pixels)
    # 5A: parallax-aware covisibility gate (golden-rule admission). Pure forward
    # motion passes the pose-delta gate (large translation) AND the coverage
    # gate (steadily reveals new far-corridor voxels) while carrying near-zero
    # triangulation information: every shared surface point is re-seen from
    # almost the same direction. This gate rejects a candidate when BOTH
    #   covis_frac   >= parallax_covis_max  (mostly re-seeing known voxels)
    #   med_parallax <  parallax_min_deg    (…from nearly the same rays)
    # unless the frame is forced (max_gap). covis_frac = 1 - new_voxel_frac
    # from the same backprojection as the coverage gate; med_parallax = median
    # angle at the backprojected points between the ray from this camera and
    # the ray from the last admitted keyframe. Needs occupancy_fn+load_depth_m.
    parallax_gate: bool = False
    parallax_min_deg: float = 3.0        # reject only when median parallax below this
    parallax_covis_max: float = 0.6      # ...and covisibility (occupied frac) at/above this
    # Exposure gate: reject keyframe candidates whose mean luminance has shifted
    # too far from the last admitted frame (proxy for auto-exposure jumps that
    # poison photometric supervision). Force-admit after N starvation iters so a
    # sustained lighting change cannot deadlock admission. Default off.
    exposure_gate_enabled: bool = False
    exposure_max_delta: float = 0.15         # reject if |Δluma|/luma_ref > this
    exposure_force_admit_iters: int = 200    # admit unconditionally after this many starved iters
    # Blur gate: hard-reject frames below a Laplacian-variance blur threshold
    # (motion-blurred RGB + its paired noisy depth). Cheap (Laplacian var on a
    # downsampled greyscale). Force-admit after N starvation iters. Default off.
    blur_gate_enabled: bool = False
    blur_gate_min_laplacian: float = 20.0    # reject if var(Laplacian) < this
    blur_gate_downsample: int = 4            # downsample factor before Laplacian (speed)
    blur_gate_force_admit_iters: int = 200   # admit unconditionally after this many starved iters
    # Low-overlap starvation relief. The render-overlap gate rejects a candidate
    # as "low_overlap" when alpha-coverage over the EXISTING map is below
    # min_overlap — i.e. the frame sees mostly unreconstructed space. This is
    # exactly the frontier-expansion case a keyframe selector should welcome,
    # but on small/fast-panning scenes (confirmed on TUM fr1_desk: 596-frame
    # desk-orbit, hybrid_keyframe+coverage_gain_gate landed at 7.0% admission,
    # 56% of all candidates rejected as low_overlap) the map lags the frontier
    # and the gate can starve admission near-permanently: no new content is
    # ever admitted to grow the map into the region that would raise its own
    # overlap score, so the same frames keep failing. Mirrors the existing
    # exposure/blur force-admit pattern: once starved for this many iters,
    # a would-be low_overlap rejection is admitted instead. 0 = disabled
    # (byte-identical to previous behaviour); this gate has no separate
    # "_enabled" flag because low_overlap is core admit() logic, not an
    # optional add-on gate.
    low_overlap_force_admit_iters: int = 0
    # Novelty-gated max_gap (Component 2). The plain max_gap force-admits every
    # Nth frame UNCONDITIONALLY and dominates admits on long forward walks (~80%
    # of admits on floor3), inflating the training-view count and starving the
    # per-view iteration budget. When True, a would-be forced (gap>=max_gap) frame
    # must ALSO pass the depth-coverage novelty check (new-voxel-frac >=
    # min_new_voxel_frac) to be admitted; a redundant forced frame is deferred
    # instead. A hard ceiling gap >= max_gap*max_gap_hard_mult still force-admits
    # unconditionally so admission can never fully starve. Requires the coverage
    # machinery (coverage_gain_gate + occupancy_fn + load_depth_m); with it off, or
    # when unevaluable, forcing is unchanged. Dataclass default is False (library-
    # safe/byte-identical, matching this file's motion_pregate/velocity_adaptive/
    # coverage_gain_gate convention) but the REAL CLI default (arguments/__init__.py
    # streaming_keyframe_max_gap_require_novelty) is True as of 2026-08-15
    # (issue #792/#794) -- every real training run has this gate ON by default.
    max_gap_require_novelty: bool = False
    max_gap_hard_mult: float = 4.0           # hard-ceiling multiple of max_gap for unconditional force
    # Issue #794 precedence contract: what a SOFT max_gap force
    # (max_gap <= gap < max_gap*max_gap_hard_mult) is permitted to bypass. See
    # eligsir/mapping/admission/precedence.py for the full table.
    #   "legacy_all_gates" (DEFAULT, behaviour-preserving): a soft force
    #       short-circuits straight to admission, skipping the parallax /
    #       exposure / blur / render-overlap / quality gates. This is exactly
    #       what main did before #794; the ONLY change at this setting is that
    #       the bypass is now recorded in KeyframeDecision.trace.
    #   "motion_only": the #794 contract -- a soft force relaxes the motion
    #       criterion only (motion is treated as SATISFIED, not skipped, so the
    #       backstop still works), and every information/quality gate binds.
    # The hard ceiling (gap >= max_gap*max_gap_hard_mult) is unaffected by this
    # setting: it may always bypass everything except the upstream integrity
    # veto, which no path may ever bypass.
    max_gap_force_scope: str = MaxGapForceScope.LEGACY_ALL_GATES.value
    # Issue #934: novelty-DRIVEN admission. Until this existed, the coverage
    # signal could only ever VETO -- it withdrew a max_gap force (#826) or
    # rejected below min_new_voxel_frac -- and the motion pregate returned
    # "no_motion" before it was even consulted for a non-moving candidate. Two
    # fr3 GPU runs measured the consequence: ~96% of admissions owned by the
    # motion gate, 4-5 ever owned by novelty, at BOTH ends of the threshold
    # sweep. Lowering min_new_voxel_frac only widens what the veto lets
    # through; it can never transfer ownership.
    #
    # When True, a candidate whose measured new-voxel fraction reaches
    # novelty_force_frac has the MOTION criterion treated as SATISFIED
    # (ForcePath.NOVELTY_FORCE, bypassing Gate.MOTION and nothing else) and
    # then continues through every remaining gate -- coverage, parallax,
    # exposure, blur, render-overlap, quality all still bind. It is a trigger,
    # not a short-circuit: nothing is admitted unconditionally.
    #
    # Fires ONLY on ForcePath.NORMAL candidates (a max_gap candidate keeps its
    # own trigger, so the trigger buckets stay mutually exclusive), only
    # post-baseline, and only when occupancy_fn + load_depth_m are wired.
    # Default False => behaviour-identical to before (artifacts gain additive
    # keys regardless: 4 new `precedence` trace keys, the `novelty_forced`
    # sidecar column, and report.json's admission_fingerprint["novelty_force"]).
    novelty_force_admit: bool = False
    # Threshold for the trigger. MUST exceed min_new_voxel_frac -- that
    # invariant is what guarantees a novelty-forced candidate cannot then be
    # rejected by the coverage gate's own "< min_new_voxel_frac" branch below.
    # __post_init__ clamps (never raises) if a spec inverts them.
    novelty_force_frac: float = 0.6
    # Stale-pose guard. This trigger bypasses the motion pregate, which is a
    # standing safety net against stale-pose frames under a sparse GT-pose
    # stream -- and the hazard is concrete: a fresh image carrying a STALE pose
    # backprojects its depth at the wrong pose, lands in genuinely-unobserved
    # voxels, and therefore reads as HIGH novelty. Uncorrected, this trigger
    # would preferentially re-admit exactly the frames the pregate exists to
    # exclude. When True the trigger fires only if this candidate's own
    # instantaneous translation-velocity sample exists and is strictly
    # positive (i.e. the pose really advanced since the previous candidate).
    # With velocity_adaptive=False no such sample exists and the trigger
    # refuses to fire -- fail-closed, never fall through unguarded.
    novelty_force_require_fresh_pose: bool = True
    # --- Issue #946: decoupled soft escalation / hard starvation ceiling ---
    # Before #946 there was ONE resolved quantity (max_gap frames) and the
    # hard ceiling was derived from it, so raising the soft threshold moved
    # the last-resort safety bound too. These four fields make the two
    # independent. See precedence.resolve_backstop_thresholds for the rules.
    #
    # "legacy_frames" (DEFAULT, behaviour-preserving): soft = max_gap frames,
    #     hard = max_gap * max_gap_hard_mult frames. Bit-identical admit/reject
    #     decisions to pre-#946 on every existing config, spec and CLI
    #     invocation -- asserted by tests/test_issue946_backstop_decoupling.py.
    # "target_time": soft = soft_backstop_multiplier * the velocity
    #     controller's own target admit interval, in SOURCE SECONDS; hard is
    #     resolved from hard_backstop_s / hard_backstop_frames and reads no
    #     soft input at all. OPT-IN and NOT GPU-validated (issue #947 owns the
    #     first experiment).
    soft_backstop_mode: str = SoftBackstopMode.LEGACY_FRAMES.value
    # Must be >= 1: the soft backstop exists to catch a controller that has
    # gone quiet, so it must not be able to fire BEFORE the controller's own
    # cadence. A value below 1 is normalised up to 1 with a loud warning.
    soft_backstop_multiplier: float = 1.25
    # Independent hard ceiling for target_time. 0 => resolve from
    # hard_backstop_frames instead. Never consulted in legacy_frames mode.
    hard_backstop_s: float = 0.0
    # Frame-domain hard ceiling for target_time. The default is the CURRENT
    # EFFECTIVE production ceiling made explicit: CLI max_gap=10 *
    # max_gap_hard_mult=4.0 = 40 source frames. Stated as a literal so a
    # future soft-timing change of 15/18/22 cannot move it.
    hard_backstop_frames: float = 40.0
    # #1044: apply the #946 non-preemption invariant to `legacy_frames` too --
    # raise the SOFT backstop to the velocity controller's nominal target
    # interval whenever max_gap would otherwise pre-empt the motion criterion
    # outright (measured: 0/109 motion-owned decisions on a real fr3 run).
    # Hard ceiling is unaffected. Set False to restore pre-#1044 semantics.
    backstop_nonpreemption_floor: bool = True
    # Issue #794 (gap closure, 2026-08-23). legacy_frames mode WARNS (never
    # raises) on the two structurally contradictory configurations named
    # above -- the #1044 floor declining to apply, and hard_frames <=
    # soft_frames -- whereas target_time mode raises BackstopConfigError
    # unconditionally on the equivalent contradiction. When True, legacy_frames
    # raises too, in exactly those same two cases. Default False: must not
    # change behaviour for ANY existing spec by default -- opt in explicitly.
    require_reachable_motion_gate: bool = False

    def __post_init__(self) -> None:
        # ``max_gap`` is used for both the forced-admit check and the
        # cheap-score normalisation.  Clamp to ``>= 1`` so a CLI value
        # of 0 cannot degenerate the cheap-score math or force-admit
        # every frame.
        if self.max_gap < 1:
            self.max_gap = 1
        if self.max_history < 1:
            self.max_history = 1
        # #934: novelty_force_frac MUST exceed min_new_voxel_frac, or a
        # novelty-forced candidate could be admitted past the motion gate and
        # then immediately rejected by the coverage gate's own
        # "< min_new_voxel_frac" branch -- a trigger that fires and admits
        # nothing, i.e. the exact failure this mechanism exists to fix.
        # Clamp upward and RECORD it; never raise (a hard error here would
        # retroactively reject existing specs, which this project's Run
        # Contracts treat as a serious hazard -- same convention as
        # precedence.coerce_scope and validate_backstop_config).
        self.config_warnings: List[str] = []
        if self.novelty_force_admit and self.novelty_force_frac <= self.min_new_voxel_frac:
            _requested = float(self.novelty_force_frac)
            self.novelty_force_frac = float(self.min_new_voxel_frac) + 1e-6
            self.config_warnings.append(
                f"novelty_force_frac={_requested} must EXCEED "
                f"min_new_voxel_frac={self.min_new_voxel_frac} (a novelty force "
                f"below the coverage-gate floor would be admitted past the "
                f"motion gate and then rejected by the coverage gate); clamped "
                f"to {self.novelty_force_frac}."
            )


def keyframe_admitter_config_from_args(args: Any) -> KeyframeAdmitterConfig:
    """Build the one canonical production admission configuration."""

    get = lambda name, default: getattr(args, name, default)
    return KeyframeAdmitterConfig(
        min_translation_m=float(get("streaming_keyframe_min_translation", 0.05)),
        min_rotation_deg=float(get("streaming_keyframe_min_rotation_deg", 5.0)),
        min_overlap=float(get("streaming_keyframe_min_overlap", 0.25)),
        max_overlap=float(get("streaming_keyframe_max_overlap", 0.90)),
        max_gap=int(get("streaming_keyframe_max_gap", 10)),
        coverage_alpha=float(get("streaming_keyframe_coverage_alpha", 0.3)),
        quality_enabled=bool(get("streaming_keyframe_quality_enabled", False)),
        quality_min_blur=float(get("streaming_keyframe_quality_min_blur", 8.0)),
        quality_max_clip_ratio=float(get("streaming_keyframe_quality_max_clip_ratio", 0.25)),
        quality_min_depth_valid=float(get("streaming_keyframe_quality_min_depth_valid", 0.10)),
        final_threshold=float(get("streaming_keyframe_final_threshold", 0.45)),
        depth_min=float(get("streaming_min_depth", 0.1)),
        depth_max=float(get("streaming_max_depth", 8.0)),
        max_history=int(get("streaming_keyframe_max_history", 1024)),
        motion_pregate=bool(get("streaming_keyframe_motion_pregate", True)),
        velocity_adaptive=bool(get("streaming_keyframe_velocity_adaptive", True)),
        target_admit_rate_fps=float(get("streaming_keyframe_target_admit_fps", 2.0)),
        input_fps=float(get("streaming_keyframe_input_fps", 30.0)),
        velocity_ema_alpha=float(get("streaming_keyframe_velocity_ema_alpha", 0.1)),
        coverage_gain_gate=bool(get("streaming_keyframe_coverage_gain_gate", True)),
        min_new_voxel_frac=float(get("streaming_keyframe_min_new_voxel_frac", 0.05)),
        coverage_gain_depth_stride=int(get("streaming_keyframe_coverage_gain_depth_stride", 8)),
        parallax_gate=bool(get("streaming_kf_parallax_gate", False)),
        parallax_min_deg=float(get("streaming_kf_parallax_deg", 3.0)),
        parallax_covis_max=float(get("streaming_kf_covis_thresh", 0.6)),
        max_gap_require_novelty=bool(get("streaming_keyframe_max_gap_require_novelty", True)),
        max_gap_hard_mult=float(get("streaming_keyframe_max_gap_hard_mult", 4.0)),
        exposure_gate_enabled=bool(get("streaming_keyframe_exposure_gate", False)),
        exposure_max_delta=float(get("streaming_keyframe_exposure_max_delta", 0.15)),
        exposure_force_admit_iters=int(get("streaming_keyframe_exposure_force_admit_iters", 200)),
        blur_gate_enabled=bool(get("streaming_keyframe_blur_gate", False)),
        blur_gate_min_laplacian=float(get("streaming_keyframe_blur_min_laplacian", 20.0)),
        blur_gate_downsample=int(get("streaming_keyframe_blur_downsample", 4)),
        blur_gate_force_admit_iters=int(get("streaming_keyframe_blur_force_admit_iters", 200)),
        low_overlap_force_admit_iters=int(get("streaming_keyframe_low_overlap_force_admit_iters", 0)),
        max_gap_force_scope=str(get("streaming_keyframe_max_gap_force_scope", "legacy_all_gates")),
        novelty_force_admit=bool(get("streaming_keyframe_novelty_force_admit", False)),
        novelty_force_frac=float(get("streaming_keyframe_novelty_force_frac", 0.6)),
        novelty_force_require_fresh_pose=bool(get("streaming_keyframe_novelty_force_require_fresh_pose", True)),
        soft_backstop_mode=str(get("streaming_keyframe_soft_backstop_mode", "legacy_frames")),
        soft_backstop_multiplier=float(get("streaming_keyframe_soft_backstop_multiplier", 1.25)),
        hard_backstop_s=float(get("streaming_keyframe_hard_backstop_s", 0.0)),
        hard_backstop_frames=float(get("streaming_keyframe_hard_backstop_frames", 40.0)),
        backstop_nonpreemption_floor=bool(get("streaming_keyframe_backstop_nonpreemption_floor", True)),
        require_reachable_motion_gate=bool(get("streaming_keyframe_require_reachable_motion_gate", False)),
    )


@dataclass
class FrameInfo:
    """Minimum info needed for a cheap keyframe decision.

    The :class:`KeyframeAdmitter` reads ``.c2w`` (4x4 np.ndarray) and
    ``.index`` (int) from the source frame, plus an OPTIONAL ``.timestamp``
    (float seconds) used by the velocity controller's clock (#793).
    Production passes :class:`~eligsir.io.sources.decode_cache.StreamingRGBDFrame`,
    which always carries ``timestamp``; ``None`` here selects the documented
    ``frame_delta / input_fps`` fallback.
    """

    c2w: np.ndarray
    index: int
    timestamp: Optional[float] = None


class KeyframeAdmitter:
    """Two-stage keyframe admission.

    Cheap gate: pose delta, rotation delta, frame-index gap.
    Expensive gate: render overlap, optional quality score.
    """

    def __init__(
        self,
        config: Optional[KeyframeAdmitterConfig] = None,
        render_fn: Optional[RenderFnType] = None,
        gaussians: Any = None,
        pipe: Any = None,
        background: Any = None,
        evaluate_quality: Optional[Callable[[np.ndarray, np.ndarray, Any], Any]] = None,
        load_depth_m: Optional[Callable[[Any], np.ndarray]] = None,
        load_rgb: Optional[Callable[[Any], np.ndarray]] = None,
        occupancy_fn: Optional[Callable] = None,
        observation_db: Optional[Any] = None,
        evaluate_luma: Optional[Callable[[Any], Optional[float]]] = None,
        evaluate_blur: Optional[Callable[[Any], Optional[float]]] = None,
    ) -> None:
        self._config = config or KeyframeAdmitterConfig()
        self._render_fn = render_fn
        self._gaussians = gaussians
        self._pipe = pipe
        self._background = background
        self._evaluate_quality = evaluate_quality
        self._load_depth_m = load_depth_m
        self._load_rgb = load_rgb
        self._evaluate_luma = evaluate_luma
        self._evaluate_blur = evaluate_blur
        # Track A: callable(pts_world: np.ndarray, voxel_size: float) -> bool mask
        # of which points fall in already-occupied voxels.
        self._occupancy_fn = occupancy_fn
        # Task 5: optional FrameObservationDB for database-driven diverse-revisit
        # admission scoring. None (default) => byte-identical to pre-task-5 behavior.
        self._observation_db = observation_db

        # Tracking state
        self._keyframes_admitted: int = 0
        self._keyframes_rejected: int = 0
        # Track A: count of frames rejected specifically by the coverage gate.
        self._keyframes_rejected_coverage: int = 0
        self._last_train_index: int = -1
        self._last_train_c2w: Optional[np.ndarray] = None
        # Issue #933: source-frame origin the starvation clock counts from
        # while no keyframe has ever been admitted. See
        # _pose_delta_from_last_train for the full rationale.
        self._starvation_origin_index: Optional[int] = None
        # #946: source timestamps of the two possible starvation references
        # (the last admitted keyframe post-baseline, the #933 starvation
        # origin pre-baseline). Used ONLY to resolve starvation age in
        # seconds; never feeds a pose delta.
        self._last_train_ts: Optional[float] = None
        self._starvation_origin_ts: Optional[float] = None
        self._keyframe_indices: List[int] = []
        # History cap is now configurable (was previously hard-coded).
        self._max_history: int = self._config.max_history
        # --- Velocity controller state (#793) ---------------------------------
        # EMAs of INSTANTANEOUS speed between consecutive CANDIDATE frames, in
        # physical units. NOT displacement-per-admission-interval (the pre-#793
        # bug). Updated on every candidate with a valid dt, admitted or not.
        self._translation_velocity_ema_m_s: Optional[float] = None
        self._rotation_velocity_ema_deg_s: Optional[float] = None
        # Previous CANDIDATE (not previous admitted keyframe) reference.
        self._prev_candidate_c2w: Optional[np.ndarray] = None
        self._prev_candidate_index: Optional[int] = None
        self._prev_candidate_ts: Optional[float] = None
        # Timebase resolution state (sticky once decided).
        self._timebase_source: str = "frame_index_nominal_fps_warmup"
        self._timebase_decided: bool = False
        self._timebase_fps_samples: List[float] = []
        self._implied_input_fps: Optional[float] = None
        # Clamp/eval accounting for telemetry (#795).
        self._threshold_evals: int = 0
        self._threshold_floor_hits: int = 0
        self._threshold_ceil_hits: int = 0
        # Real elapsed source seconds observed across candidates (#795 realized
        # rate denominator; nominal index/input_fps is kept alongside it).
        self._source_elapsed_s: float = 0.0

        # Exposure gate state (mean-luminance reference + starvation counter).
        self._last_admitted_luma: Optional[float] = None
        self._last_admitted_iter: int = 0
        self._exposure_iter: int = 0
        self._pending_luma: Optional[float] = None
        # Per-call controller telemetry parked by _admit_impl for admit().
        self._pending_controller_telemetry: Optional[dict] = None
        # #946: per-call backstop telemetry, parked by the SAME mechanism and
        # stamped in the same place. _admit_impl has ~10 return sites and
        # rejected candidates are exactly where starvation age is the
        # interesting number, so stamping at construction sites would leave
        # holes on precisely the rows that matter.
        self._pending_backstop_telemetry: Optional[dict] = None
        # Blur gate state (starvation counter).
        self._blur_iter: int = 0
        self._last_admitted_iter_blur: int = 0

        # Issue #794 precedence telemetry: every bypass that ACTUALLY occurred,
        # and how many candidates entered on each trigger. Enabled-gate bypasses
        # are counted separately from permitted-but-disabled ones so the numbers
        # answer "did a force skip a live gate?" rather than being inflated by
        # gates that were off for the whole run.
        self._precedence_bypass_counts: dict = {}
        self._precedence_bypass_inactive_counts: dict = {}
        self._precedence_trigger_counts: dict = {}
        # Issue #928: per-run tally of the novelty-gate tri-state outcome
        # (see PrecedenceTrace.record_novelty_gate_outcome) for every
        # would-be-forced candidate, so a run can be audited for "how often
        # did the gate actually run vs. abstain" without replaying the
        # per-frame keyframe_admission_log.jsonl.
        self._novelty_gate_outcome_counts: dict = {}
        self._novelty_gate_unavailable_reason_counts: dict = {}
        # Set on every coverage call. It is telemetry only and never read by
        # admission control; the trace carries it to the JSONL/report surface.
        self._last_coverage_none_cause: Optional[str] = None
        # Issue #934: novelty-DRIVEN trigger tallies. `triggered` counts
        # candidates the trigger fired for; `decisive` counts the subset where
        # it was the ACTUAL owner of the resulting admission. The GAP between
        # them is the diagnostic for "the trigger fires but a downstream
        # AND-term eats it" -- i.e. the veto-only trap recurring one gate
        # lower. Keep both; a single counter cannot answer that question.
        self._novelty_force_outcome_counts: dict = {}
        self._novelty_force_unavailable_reason_counts: dict = {}
        self._novelty_force_triggered: int = 0
        self._novelty_force_decisive: int = 0
        self._precedence_config_warnings: List[str] = validate_backstop_config(
            max_gap=self._config.max_gap,
            velocity_adaptive=self._config.velocity_adaptive,
            input_fps=self._config.input_fps,
            target_admit_rate_fps=self._config.target_admit_rate_fps,
            scope=coerce_scope(self._config.max_gap_force_scope),
            max_gap_require_novelty=self._config.max_gap_require_novelty,
            coverage_gain_gate=self._config.coverage_gain_gate,
        )
        for _w in self._precedence_config_warnings:
            print(f"[KeyframeAdmitter] WARNING (#794 precedence): {_w}")
        # #934: the config-level clamps recorded by
        # KeyframeAdmitterConfig.__post_init__ must be just as loud. A silently
        # clamped threshold is exactly the class of "the spec said X, the run
        # did Y" hazard this project keeps re-discovering -- and
        # `precedence_config_warnings` alone would not surface it, because
        # nothing in production reads that property today.
        for _w in getattr(self._config, "config_warnings", ()):
            print(f"[KeyframeAdmitter] WARNING (#934 config clamp): {_w}")

        # --- Issue #946: resolve the two backstops ONCE, at construction ---
        # Config resolution is the only place that may raise
        # (BackstopConfigError, target_time only -- no historical spec can
        # reach it). A per-candidate raise would crash training mid-run, so
        # nothing below the decision path validates anything.
        self._backstop = resolve_backstop_thresholds(
            mode=self._config.soft_backstop_mode,
            max_gap=self._config.max_gap,
            max_gap_hard_mult=self._config.max_gap_hard_mult,
            input_fps=self._config.input_fps,
            velocity_adaptive=self._config.velocity_adaptive,
            target_admit_rate_fps=self._config.target_admit_rate_fps,
            soft_multiplier=self._config.soft_backstop_multiplier,
            hard_backstop_s=self._config.hard_backstop_s,
            hard_backstop_frames=self._config.hard_backstop_frames,
            nonpreemption_floor=self._config.backstop_nonpreemption_floor,
            require_reachable_motion_gate=self._config.require_reachable_motion_gate,
        )
        self._backstop_config_warnings: List[str] = validate_soft_backstop_config(
            self._backstop,
            velocity_adaptive=self._config.velocity_adaptive,
            max_gap=self._config.max_gap,
            max_gap_hard_mult=self._config.max_gap_hard_mult,
        )
        for _w in self._backstop_config_warnings:
            print(f"[KeyframeAdmitter] WARNING (#946 backstop): {_w}")
        # Static half of the per-decision telemetry -- the thresholds do not
        # vary per candidate, only the age does.
        self._backstop_static_telemetry: dict = self._backstop.as_telemetry()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Reset all per-run state. Useful at bootstrap and between
        trajectory restarts.

        #933: this clears ``_starvation_origin_index`` too, which restarts the
        pre-baseline starvation countdown. That is deliberate -- a trajectory
        restart is a new run -- but it means a caller that invoked this
        repeatedly *while already starving* could keep pushing the hard ceiling
        out of reach. No production code calls this mid-run today (it is
        bootstrap-only); anything that starts to must decide explicitly whether
        the origin should survive the reset.
        """
        self._keyframes_admitted = 0
        self._keyframes_rejected = 0
        self._last_train_index = -1
        self._last_train_c2w = None
        self._starvation_origin_index = None
        self._last_train_ts = None
        self._starvation_origin_ts = None
        self._keyframe_indices = []
        self._translation_velocity_ema_m_s = None
        self._rotation_velocity_ema_deg_s = None
        self._prev_candidate_c2w = None
        self._prev_candidate_index = None
        self._prev_candidate_ts = None
        self._timebase_source = "frame_index_nominal_fps_warmup"
        self._timebase_decided = False
        self._timebase_fps_samples = []
        self._implied_input_fps = None
        self._threshold_evals = 0
        self._threshold_floor_hits = 0
        self._threshold_ceil_hits = 0
        self._source_elapsed_s = 0.0
        self._keyframes_rejected_coverage = 0
        self._last_admitted_luma = None
        self._last_admitted_iter = 0
        self._exposure_iter = 0
        self._pending_luma = None
        self._blur_iter = 0
        self._last_admitted_iter_blur = 0
        self._precedence_bypass_counts = {}
        self._precedence_bypass_inactive_counts = {}
        self._precedence_trigger_counts = {}
        self._novelty_gate_outcome_counts = {}
        self._novelty_gate_unavailable_reason_counts = {}
        self._novelty_force_outcome_counts = {}
        self._novelty_force_unavailable_reason_counts = {}
        self._novelty_force_triggered = 0
        self._novelty_force_decisive = 0

    _RESUME_SCHEMA = "hybrid_keyframe_admitter_v1"
    _RESUME_KEYS = {
        "owner_schema", "config", "last_train_index", "last_train_c2w",
        "starvation_origin_index", "last_train_ts", "starvation_origin_ts",
        "keyframe_indices", "translation_velocity_ema_m_s",
        "rotation_velocity_ema_deg_s", "prev_candidate_c2w",
        "prev_candidate_index", "prev_candidate_ts", "timebase_source",
        "timebase_decided", "timebase_fps_samples", "implied_input_fps",
        "last_admitted_luma", "last_admitted_iter", "exposure_iter",
        "blur_iter", "last_admitted_iter_blur",
        "keyframes_admitted", "keyframes_rejected",
        "keyframes_rejected_coverage", "source_elapsed_s",
    }

    @staticmethod
    def _portable_pose(value: Optional[np.ndarray]) -> Optional[list[list[float]]]:
        if value is None:
            return None
        pose = np.asarray(value, dtype=np.float32)
        if pose.shape != (4, 4) or not np.isfinite(pose).all():
            raise ValueError("hybrid keyframe exact-resume pose must be finite 4x4")
        return pose.tolist()

    @classmethod
    def _restore_pose(cls, value: Any) -> Optional[np.ndarray]:
        if value is None:
            return None
        return np.asarray(cls._portable_pose(np.asarray(value)), dtype=np.float32)

    def resume_state_dict(self) -> dict[str, Any]:
        """Serialize only state that can change a future hybrid decision."""
        if any(
            value is not None
            for value in (
                self._pending_luma,
                self._pending_controller_telemetry,
                self._pending_backstop_telemetry,
            )
        ):
            raise ValueError("hybrid keyframe admission has pending per-call scratch")
        return {
            "owner_schema": self._RESUME_SCHEMA,
            "config": asdict(self._config),
            "last_train_index": self._last_train_index,
            "last_train_c2w": self._portable_pose(self._last_train_c2w),
            "starvation_origin_index": self._starvation_origin_index,
            "last_train_ts": self._last_train_ts,
            "starvation_origin_ts": self._starvation_origin_ts,
            "keyframe_indices": list(self._keyframe_indices),
            "translation_velocity_ema_m_s": self._translation_velocity_ema_m_s,
            "rotation_velocity_ema_deg_s": self._rotation_velocity_ema_deg_s,
            "prev_candidate_c2w": self._portable_pose(self._prev_candidate_c2w),
            "prev_candidate_index": self._prev_candidate_index,
            "prev_candidate_ts": self._prev_candidate_ts,
            "timebase_source": self._timebase_source,
            "timebase_decided": self._timebase_decided,
            "timebase_fps_samples": list(self._timebase_fps_samples),
            "implied_input_fps": self._implied_input_fps,
            "last_admitted_luma": self._last_admitted_luma,
            "last_admitted_iter": self._last_admitted_iter,
            "exposure_iter": self._exposure_iter,
            "blur_iter": self._blur_iter,
            "last_admitted_iter_blur": self._last_admitted_iter_blur,
            # Observational cumulative counters do not affect decisions, but
            # restoring them prevents an exact tail report from resetting at K.
            "keyframes_admitted": self._keyframes_admitted,
            "keyframes_rejected": self._keyframes_rejected,
            "keyframes_rejected_coverage": self._keyframes_rejected_coverage,
            "source_elapsed_s": self._source_elapsed_s,
        }

    def load_resume_state_dict(self, payload: Any, resolver=None) -> None:
        del resolver
        if not isinstance(payload, dict) or set(payload) != self._RESUME_KEYS:
            raise ValueError("hybrid keyframe exact-resume state keys mismatch")
        if payload["owner_schema"] != self._RESUME_SCHEMA:
            raise ValueError("unsupported hybrid keyframe exact-resume state")
        if payload["config"] != asdict(self._config):
            raise ValueError("hybrid keyframe configuration mismatch")
        last_train_index = payload["last_train_index"]
        if isinstance(last_train_index, bool) or not isinstance(last_train_index, int) \
                or last_train_index < -1:
            raise ValueError("hybrid keyframe last_train_index is invalid")
        nonnegative_int_fields = (
            "last_admitted_iter", "exposure_iter",
            "blur_iter", "last_admitted_iter_blur",
            "keyframes_admitted", "keyframes_rejected",
            "keyframes_rejected_coverage",
        )
        for key in nonnegative_int_fields:
            value = payload[key]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"hybrid keyframe {key} is invalid")
        optional_nonnegative_ints = (
            "starvation_origin_index", "prev_candidate_index",
        )
        for key in optional_nonnegative_ints:
            value = payload[key]
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise ValueError(f"hybrid keyframe {key} is invalid")
        indices = payload["keyframe_indices"]
        if not isinstance(indices, list) or len(indices) > self._max_history or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in indices
        ):
            raise ValueError("hybrid keyframe retained indices are invalid")
        optional_floats = (
            "last_train_ts", "starvation_origin_ts", "translation_velocity_ema_m_s",
            "rotation_velocity_ema_deg_s", "prev_candidate_ts", "implied_input_fps",
            "last_admitted_luma",
        )
        for key in optional_floats:
            value = payload[key]
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise ValueError(f"hybrid keyframe {key} is invalid")
        source_elapsed = payload["source_elapsed_s"]
        if isinstance(source_elapsed, bool) or not isinstance(source_elapsed, (int, float)) \
                or not math.isfinite(float(source_elapsed)) or source_elapsed < 0:
            raise ValueError("hybrid keyframe source elapsed time is invalid")
        if payload["keyframes_rejected_coverage"] > payload["keyframes_rejected"]:
            raise ValueError("hybrid keyframe rejection counters are inconsistent")
        samples = payload["timebase_fps_samples"]
        if not isinstance(samples, list) or len(samples) > _TIMEBASE_WARMUP_SAMPLES or any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(float(value)) or float(value) <= 0
            for value in samples
        ):
            raise ValueError("hybrid keyframe timebase samples are invalid")
        if payload["timebase_source"] not in {
            "frame_index_nominal_fps_warmup", "frame_index_nominal_fps",
            "source_timestamp",
        } or type(payload["timebase_decided"]) is not bool:
            raise ValueError("hybrid keyframe timebase state is invalid")
        last_pose = self._restore_pose(payload["last_train_c2w"])
        previous_pose = self._restore_pose(payload["prev_candidate_c2w"])
        if (last_pose is None) != (payload["last_train_index"] == -1):
            raise ValueError("hybrid keyframe last-train identity is inconsistent")
        self._last_train_index = payload["last_train_index"]
        self._last_train_c2w = last_pose
        self._starvation_origin_index = payload["starvation_origin_index"]
        self._last_train_ts = payload["last_train_ts"]
        self._starvation_origin_ts = payload["starvation_origin_ts"]
        self._keyframe_indices = list(indices)
        self._translation_velocity_ema_m_s = payload["translation_velocity_ema_m_s"]
        self._rotation_velocity_ema_deg_s = payload["rotation_velocity_ema_deg_s"]
        self._prev_candidate_c2w = previous_pose
        self._prev_candidate_index = payload["prev_candidate_index"]
        self._prev_candidate_ts = payload["prev_candidate_ts"]
        self._timebase_source = payload["timebase_source"]
        self._timebase_decided = payload["timebase_decided"]
        self._timebase_fps_samples = [float(value) for value in samples]
        self._implied_input_fps = payload["implied_input_fps"]
        self._last_admitted_luma = payload["last_admitted_luma"]
        self._last_admitted_iter = payload["last_admitted_iter"]
        self._exposure_iter = payload["exposure_iter"]
        self._blur_iter = payload["blur_iter"]
        self._last_admitted_iter_blur = payload["last_admitted_iter_blur"]
        self._keyframes_admitted = payload["keyframes_admitted"]
        self._keyframes_rejected = payload["keyframes_rejected"]
        self._keyframes_rejected_coverage = payload["keyframes_rejected_coverage"]
        self._source_elapsed_s = float(source_elapsed)

    # -- Issue #794 precedence contract -----------------------------------
    @property
    def force_scope(self) -> MaxGapForceScope:
        """Resolved soft-max_gap bypass scope for this admitter."""
        return coerce_scope(self._config.max_gap_force_scope)

    @property
    def precedence_bypass_counts(self) -> dict:
        """``{gate_name: n}`` -- bypasses of gates that were ENABLED."""
        return dict(self._precedence_bypass_counts)

    @property
    def precedence_bypass_inactive_counts(self) -> dict:
        """``{gate_name: n}`` -- bypasses permitted by the contract but of
        gates that were disabled for this run (informational, never evidence
        that a real quality check was skipped)."""
        return dict(self._precedence_bypass_inactive_counts)

    @property
    def precedence_trigger_counts(self) -> dict:
        """``{candidate_trigger: n}`` over every :meth:`admit` call."""
        return dict(self._precedence_trigger_counts)

    @property
    def precedence_config_warnings(self) -> List[str]:
        """Contradictory controller/backstop combinations found at construction
        (warn-only; see :func:`...precedence.validate_backstop_config`).

        Also carries the config-level clamps recorded by
        :meth:`KeyframeAdmitterConfig.__post_init__` (#934's
        ``novelty_force_frac <= min_new_voxel_frac`` inversion), so a spec that
        silently got clamped is visible in the same place as every other
        contradictory-config warning rather than only in source."""
        return list(self._precedence_config_warnings) + list(
            getattr(self._config, "config_warnings", [])
        )

    @property
    def novelty_force_counts(self) -> dict:
        """Issue #934: realized behaviour of the novelty-DRIVEN trigger.

        ``triggered`` = candidates the trigger fired for (motion criterion
        treated as satisfied). ``decisive`` = the subset where it was the real
        owner of the admission (``reason == "novelty_admit"``).

        **Read the two together.** ``triggered`` high with ``decisive`` ~0
        means the trigger fires but a downstream AND-term (most plausibly the
        render-overlap floor) is eating the admissions -- the veto-only trap
        this mechanism exists to fix, recurring one gate lower. That is a
        design signal, not noise; see
        docs/architecture/2026-08-16-issue934-novelty-driven-admission.md.
        """
        return {
            "triggered": self._novelty_force_triggered,
            "decisive": self._novelty_force_decisive,
            "outcome_counts": dict(self._novelty_force_outcome_counts),
            "unavailable_reason_counts": dict(
                self._novelty_force_unavailable_reason_counts
            ),
        }

    def _record_novelty_force_outcome(
        self,
        trace: PrecedenceTrace,
        outcome: str,
        unavailable_reason: Optional[str] = None,
        fraction: Optional[float] = None,
        none_cause: Optional[str] = None,
    ) -> None:
        """Issue #934: stamp *trace* with the novelty-force outcome and tally
        it. Called at most once per :meth:`_admit_impl` call."""
        trace.record_novelty_force_outcome(
            outcome, unavailable_reason, fraction, none_cause
        )
        self._novelty_force_outcome_counts[outcome] = (
            self._novelty_force_outcome_counts.get(outcome, 0) + 1
        )
        if unavailable_reason is not None:
            self._novelty_force_unavailable_reason_counts[unavailable_reason] = (
                self._novelty_force_unavailable_reason_counts.get(unavailable_reason, 0) + 1
            )
        if outcome == "triggered":
            self._novelty_force_triggered += 1

    @property
    def novelty_gate_outcome_counts(self) -> dict:
        """Issue #928: ``{outcome: n}`` over every candidate that was at least
        a would-be max_gap force (``gap >= max_gap``), keyed by
        ``PrecedenceTrace.novelty_gate_outcome`` ("not_evaluated" /
        "evaluated_novel" / "evaluated_redundant"). Candidates that were
        never a would-be force ("not_applicable") are not counted here."""
        return dict(self._novelty_gate_outcome_counts)

    @property
    def novelty_gate_unavailable_reason_counts(self) -> dict:
        """Issue #928: ``{reason: n}`` breakdown of every "not_evaluated"
        outcome above, keyed by
        ``PrecedenceTrace.novelty_gate_unavailable_reason`` ("gate_disabled" /
        "no_occupancy_fn" / "no_depth_loader" / "hard_ceiling" /
        "frac_unavailable")."""
        return dict(self._novelty_gate_unavailable_reason_counts)

    def _record_novelty_gate_outcome(
        self,
        trace: PrecedenceTrace,
        outcome: str,
        unavailable_reason: Optional[str] = None,
        fraction: Optional[float] = None,
        none_cause: Optional[str] = None,
    ) -> None:
        """Issue #928: stamp *trace* with the tri-state novelty-gate outcome
        for the current candidate and update the per-run tallies. Called at
        most once per :meth:`_admit_impl` call, only for candidates that are
        at least a would-be max_gap force."""
        trace.record_novelty_gate_outcome(
            outcome, unavailable_reason, fraction, none_cause
        )
        self._novelty_gate_outcome_counts[outcome] = (
            self._novelty_gate_outcome_counts.get(outcome, 0) + 1
        )
        if unavailable_reason is not None:
            self._novelty_gate_unavailable_reason_counts[unavailable_reason] = (
                self._novelty_gate_unavailable_reason_counts.get(unavailable_reason, 0) + 1
            )

    def _gate_enabled(self, gate: Gate) -> bool:
        """Is *gate* actually live for this admitter/run?

        Used only to classify a recorded bypass as real vs. inert -- it never
        decides anything.
        """
        cfg = self._config
        if gate is Gate.INTEGRITY:
            # Enforced upstream in StreamingScene.ingest_next_frame (#806/#817);
            # from admit()'s perspective it is always on and never bypassable.
            return True
        if gate is Gate.MOTION:
            return True
        if gate is Gate.COVERAGE_NOVELTY:
            return bool(
                cfg.coverage_gain_gate
                and self._occupancy_fn is not None
                and self._load_depth_m is not None
            )
        if gate is Gate.PARALLAX:
            return bool(cfg.parallax_gate)
        if gate is Gate.EXPOSURE:
            return bool(cfg.exposure_gate_enabled)
        if gate is Gate.BLUR:
            return bool(cfg.blur_gate_enabled)
        if gate is Gate.RENDER_OVERLAP:
            return self._render_fn is not None
        if gate is Gate.QUALITY_SCORE:
            return bool(cfg.quality_enabled and self._evaluate_quality is not None)
        return False

    def _record_bypasses(self, trace: PrecedenceTrace, trigger: ForcePath) -> None:
        """Record every gate *trigger* is skipping, split enabled vs. inert.

        Iterated in precedence order (not set order) so a trace is byte-stable
        across runs and diffable between two jobs.
        """
        permitted = bypass_permissions(trigger, self.force_scope)
        for gate in [g for g in GATE_ORDER if g in permitted]:
            enabled = self._gate_enabled(gate)
            trace.record_bypass(gate, enabled)
            bucket = (
                self._precedence_bypass_counts
                if enabled
                else self._precedence_bypass_inactive_counts
            )
            bucket[gate.value] = bucket.get(gate.value, 0) + 1

    def _finalize_trace(
        self, decision: KeyframeDecision, trace: PrecedenceTrace
    ) -> KeyframeDecision:
        """Attach the reason chain to *decision*. ``final_owner`` is the reason
        string of whichever stage actually decided -- a gate name for a
        rejection, the admitting mechanism for an admission."""
        trace.final_owner = decision.reason
        # #934: "decisive" means the novelty trigger was the ACTUAL owner, not
        # merely that it fired. A candidate the trigger fired for that would
        # have been admitted anyway via the render-overlap `novel` term is
        # attributed to "new_content"; over-claiming ownership is exactly what
        # would make the Stage 1 ownership re-measurement unreadable.
        if decision.admitted and decision.reason == "novelty_admit":
            trace.novelty_force_decisive = True
            self._novelty_force_decisive += 1
        decision.trace = trace.to_dict()
        return decision

    @property
    def keyframe_indices(self) -> List[int]:
        """Most recently admitted keyframe indices (newest last)."""
        return list(self._keyframe_indices)

    @property
    def keyframes_admitted(self) -> int:
        return self._keyframes_admitted

    @property
    def keyframes_rejected(self) -> int:
        return self._keyframes_rejected

    @property
    def starvation_clock_source(self) -> str:
        """Basis the starvation ``gap`` is currently counted on (#933).

        ``"last_keyframe"`` once a keyframe has been admitted (gap = source
        frames since that keyframe), ``"starvation_origin"`` while none has
        (gap = source frames since the first candidate ever evaluated).
        A timestamp-based basis is future work -- see issue #933.
        """
        return "last_keyframe" if self._last_train_c2w is not None else "starvation_origin"

    @property
    def starvation_origin_index(self) -> Optional[int]:
        """Source-frame index the pre-baseline starvation clock counts from,
        or ``None`` if no candidate has ever been evaluated pre-baseline.
        See :meth:`_pose_delta_from_last_train`."""
        return self._starvation_origin_index

    def note_train_frame(self, frame: FrameInfo) -> None:
        """Update the running reference used by the cheap gate.

        Args:
            frame: The most recently *ingested* train frame.
        """
        self._last_train_index = int(frame.index)
        self._last_train_c2w = np.asarray(frame.c2w, dtype=np.float32).copy()
        # #946: remember the reference frame's source timestamp so the
        # starvation age can be expressed in physical seconds. Never used for
        # any pose delta.
        self._last_train_ts = _coerce_timestamp(getattr(frame, "timestamp", None))

    def _coverage_new_voxel_frac(self, cam: Any, frame: Any) -> Optional[float]:
        """Fraction of this frame's backprojected depth points that land in
        previously-UNOCCUPIED voxels (1.0 = all new geometry, 0.0 = fully
        redundant). Returns ``None`` when it cannot be evaluated (no depth, too
        few valid points, or any error) so callers can fall back. Shared by the
        coverage gate and the novelty-gated max_gap check. Assumes the caller
        already verified ``occupancy_fn`` and ``load_depth_m`` are wired.
        """
        cfg = self._config
        self._last_coverage_none_cause = None
        try:
            _depth_m = self._load_depth_m(frame)
            if _depth_m is None:
                self._last_coverage_none_cause = "no_depth"
                return None
            if not hasattr(cam, "world_view_transform"):
                self._last_coverage_none_cause = "camera_geometry_unavailable"
                return None
            import numpy as _np
            _stride = cfg.coverage_gain_depth_stride
            _d = _depth_m[::_stride, ::_stride]
            _H, _W = _d.shape
            _ys, _xs = _np.meshgrid(
                _np.arange(_H) * _stride,
                _np.arange(_W) * _stride,
                indexing="ij",
            )
            _valid = (_d > 0.05) & (_d < cfg.depth_max)
            _xs = _xs[_valid].astype(_np.float32)
            _ys = _ys[_valid].astype(_np.float32)
            _zs = _d[_valid].astype(_np.float32)
            if len(_zs) <= 10:
                self._last_coverage_none_cause = "too_few_valid_points"
                return None
            _fx = float(cam.fx) if hasattr(cam, "fx") else float(cam.FoVx)
            _fy = float(cam.fy) if hasattr(cam, "fy") else float(cam.FoVy)
            _cx = float(cam.cx) if hasattr(cam, "cx") else _W * _stride / 2.0
            _cy = float(cam.cy) if hasattr(cam, "cy") else _H * _stride / 2.0
            _xc = (_xs - _cx) / _fx * _zs
            _yc = (_ys - _cy) / _fy * _zs
            _pts_cam = _np.stack([_xc, _yc, _zs, _np.ones_like(_zs)], axis=1)  # Nx4
            _c2w = (
                _np.array(cam.c2w)
                if hasattr(cam, "c2w")
                else _np.linalg.inv(
                    _np.array(cam.world_view_transform.cpu().numpy()).T
                )
            )
            _pts_world = _pts_cam @ _c2w[:3, :].T  # Nx3
            _occupied = self._occupancy_fn(_pts_world, 0.02)  # bool mask
            if len(_occupied) == 0:
                return 1.0
            # Stash for the parallax gate so it can reuse this backprojection
            # (points + candidate camera center) without a second pass.
            self._last_backproj = (_pts_world, _c2w[:3, 3].astype(_np.float32), _occupied)
            return float(1.0 - _np.mean(_occupied))
        except Exception:
            self._last_coverage_none_cause = "exception"
            return None

    def _db_diversity_score(self, cam: Any) -> Optional[Tuple[float, float]]:
        """Query the observation DB for this candidate's best-overlapping
        past frame and the viewing-angle delta to it.

        Returns (overlap_frac, angle_delta_deg), or None if the DB is
        unavailable, empty, or the candidate has no projected visibility
        (e.g. gaussians not wired with gaussian_uid yet).
        """
        if self._observation_db is None:
            return None
        candidate_uids = getattr(cam, "_observed_gaussian_uids", None)
        if candidate_uids is None or candidate_uids.numel() == 0:
            return None
        frame_ids, counts = self._observation_db.overlap_query(candidate_uids)
        if len(frame_ids) == 0:
            return None
        best_frame = self._observation_db.get_frame(int(frame_ids[0]))
        if best_frame is None:
            return None
        overlap_frac = float(counts[0]) / float(candidate_uids.numel())
        cand_c2w = np.asarray(getattr(cam, "c2w", np.eye(4, dtype=np.float32)), dtype=np.float32)
        old_c2w = best_frame.pose_c2w
        # Viewing direction = -z axis of the camera in world space (c2w's
        # third column, negated -- matches the optical-frame convention
        # used throughout this codebase, e.g. eligsir/contracts/conventions.py).
        cand_dir = -cand_c2w[:3, 2]
        old_dir = -old_c2w[:3, 2]
        cos_angle = float(np.clip(np.dot(cand_dir, old_dir), -1.0, 1.0))
        angle_deg = float(np.degrees(np.arccos(cos_angle)))
        return overlap_frac, angle_deg

    def _median_parallax_deg(self) -> Optional[float]:
        """Median parallax angle (deg) at the last backprojected points between
        the candidate camera ray and the last admitted keyframe's ray.

        Uses the backprojection stashed by :meth:`_coverage_new_voxel_frac`
        (call it first). Pure forward motion → rays nearly parallel → ~0°.
        Returns ``None`` when unavailable.
        """
        bp = getattr(self, "_last_backproj", None)
        if bp is None or self._last_train_c2w is None:
            return None
        try:
            import numpy as _np
            pts, c_f, _ = bp
            c_k = self._last_train_c2w[:3, 3].astype(_np.float32)
            v_f = pts - c_f[None, :]
            v_k = pts - c_k[None, :]
            n_f = _np.linalg.norm(v_f, axis=1)
            n_k = _np.linalg.norm(v_k, axis=1)
            ok = (n_f > 1e-6) & (n_k > 1e-6)
            if not ok.any():
                return None
            cos = _np.sum(v_f[ok] * v_k[ok], axis=1) / (n_f[ok] * n_k[ok])
            ang = _np.degrees(_np.arccos(_np.clip(cos, -1.0, 1.0)))
            return float(_np.median(ang))
        except Exception:
            return None

    def pregate(self, frame: Any) -> Optional[KeyframeDecision]:
        """Cheap pre-admission check callable before a GPU camera is built.

        Mirrors *only* the ``motion_pregate`` no-motion short-circuit inside
        :meth:`admit` (its step 1b), using nothing but ``frame.c2w`` — no
        ``cam``, no render, no GPU upload. Returns a rejecting
        :class:`KeyframeDecision` in the exact case :meth:`admit` would
        reject with ``reason="no_motion"``, so a caller can skip
        camera/visibility/admission work for that frame. Returns ``None``
        in every other case (moved, forced, no baseline yet, velocity-
        adaptive threshold in effect, coverage_gain_gate enabled, or the
        pregate disabled) to mean "not decided here — call :meth:`admit`
        as normal", since those cases either need ``cam`` (forced
        admission can still be downgraded by the novelty-gated max_gap
        check, which requires an occupancy lookup against ``cam``),
        mutate EMA state that must only be updated once, inside
        :meth:`admit`, to stay byte-identical, or (coverage_gain_gate)
        depend on a caller-side ``record_fn``/``visibility_fn`` having run
        for every candidate frame -- see the ``coverage_gain_gate`` note
        below for why that one is NOT optional the way the others' safety
        margin is.

        Callers must still invoke :meth:`admit` for any frame this
        returns ``None`` for; this method does not call
        ``_record_admission`` or increment any counters itself, matching
        :meth:`admit`'s own no_motion branch (which also does not).
        """
        cfg = self._config
        if not cfg.motion_pregate:
            return None
        if self._backstop.mode != SoftBackstopMode.LEGACY_FRAMES.value:
            # #946: this method's cheap `gap >= max_gap` check below is the
            # frame-domain soft threshold. Under any other resolution mode the
            # soft threshold lives in a different domain (source seconds), so
            # re-deriving it here would be a SECOND, divergent implementation
            # of the trigger -- exactly what #946 exists to prevent. Defer to
            # admit(), which reads the single resolved value.
            return None
        if cfg.novelty_force_admit:
            # #934: a no-motion candidate can now be ADMITTED by the novelty
            # trigger inside _admit_impl, which needs `cam` for the occupancy
            # backprojection. Pre-empting it here would make the trigger
            # unreachable for exactly the population it targets -- the same
            # class of mistake as pregating a forced frame (see the max_gap
            # branch below). Defer unconditionally whenever it is enabled.
            #
            # (In practice the coverage_gain_gate branch immediately below
            # already defers on every production hybrid_keyframe spec, since
            # the trigger needs that machinery anyway; this guard is here so
            # the invariant does not depend on that coincidence.)
            return None
        if cfg.coverage_gain_gate:
            # A caller that wires pregate() to skip cam/visibility_fn/record_fn
            # for the frame this returns a rejection for (see
            # scene/streaming_scene.py's ingest_next_frame `pregate_fn`) also
            # skips that frame's record_frame_observation() call -- so it never
            # enters self._observation_db. admit()'s own coverage_gain_gate
            # branch calls self._db_diversity_score(cam), which queries
            # self._observation_db.overlap_query/get_frame to find the best
            # prior observation of a would-be-redundant candidate's Gaussians.
            # A missing no-motion frame there can change which prior frame is
            # found "best" and therefore change a LATER, DIFFERENT frame's
            # admission decision (db_diverse_revisit vs rej_redundant_angle) --
            # unlike the other guards below, this is not just "this frame's
            # decision might differ", it is "some other frame's decision might
            # differ", which is a materially bigger claim than admission-
            # preserving and has not been validated end-to-end (needs a real
            # GPU run comparing admitted-frame sets with pregate on vs off on a
            # coverage_gain_gate spec -- see docs/2026-07-26-ws1-pregate-
            # coverage-gate-status.md). Defer to admit() unconditionally
            # whenever this gate is enabled, which today is nearly every
            # hybrid_keyframe production spec (grep specs/*.json).
            return None
        if self._last_train_c2w is None:
            # No baseline yet -- let the first frame fall through to admit()
            # rather than pre-empting it here.
            #
            # CORRECTED 2026-08-16 (#933): this comment used to assert that
            # admit() "admits it via the render gate and records the
            # baseline". That is NOT guaranteed. The render gate computes
            # `admitted = overlap_ok and (motion_ok or novel)`, and
            # pre-baseline `motion_ok` is False, so the frame must land inside
            # [min_overlap, max_overlap] to be admitted -- a coverage-redundant
            # first candidate sits ABOVE max_overlap (novel=False) and is
            # rejected "redundant". Deferring here is still correct (the
            # pregate must not be the thing that rejects a baseline-less
            # frame), but do not rely on a downstream rescue: what actually
            # guarantees admission can never starve pre-baseline is the
            # source-frame starvation clock feeding the hard max_gap ceiling
            # (see _pose_delta_from_last_train).
            return None
        if cfg.velocity_adaptive:
            # effective_min_trans depends on the velocity EMA, which admit()
            # updates exactly once per candidate (#793). Computing
            # it here too would either double-update the EMA (if we mutated
            # it) or use a stale value (if we peeked) that could disagree
            # with admit()'s own "moved" decision. Defer entirely.
            return None

        t_m, r_deg, gap = self._pose_delta_from_last_train(frame)
        if gap >= cfg.max_gap:
            # A forced frame's admission can still be downgraded by the
            # novelty-gated max_gap check inside admit() (needs cam for an
            # occupancy lookup) -- never pre-empt a forced frame here.
            #
            # #794 precedence: this is the ONE place the pregate must stay
            # consistent with the table -- both the soft and the hard max_gap
            # trigger are permitted to bypass Gate.MOTION, and this pregate IS
            # the motion gate's cheap half. Deferring to admit() is what lets
            # the bypass be recorded (and, under scope=motion_only, lets the
            # remaining gates still bind).
            return None
        moved = t_m >= cfg.min_translation_m or r_deg >= cfg.min_rotation_deg
        if moved:
            return None

        cheap_score = min(
            1.0,
            0.4 * (t_m / max(cfg.min_translation_m, 1e-6))
            + 0.4 * (r_deg / max(cfg.min_rotation_deg, 1e-6))
            + 0.2 * min(1.0, gap / cfg.max_gap),
        )
        _trace = PrecedenceTrace(
            candidate_trigger=ForcePath.NORMAL.value,
            scope=self.force_scope.value,
        )
        _trace.record_evaluated(Gate.MOTION)
        _pregate_decision = KeyframeDecision(
            admitted=False,
            reason="no_motion",
            score=cheap_score,
            cheap_score=cheap_score,
            forced=False,
            moved=False,
            pose_delta_m=t_m,
            pose_delta_deg=r_deg,
            gap=gap,
        )
        # #946: a caller that acts on this rejection never calls admit(), so
        # this decision is the only one the sidecar will ever see for this
        # candidate. Stamp the backstop telemetry here too, or those rows go
        # out blank. Reaching this line already implies legacy_frames mode
        # (guarded at the top) and `gap < max_gap` (guarded above), so neither
        # threshold fired and `moved` is False -- motion_trigger is "none".
        _pg_age_s, _pg_basis = self._starvation_age(frame, gap)
        for _k, _v in self._backstop_static_telemetry.items():
            if hasattr(_pregate_decision, _k):
                setattr(_pregate_decision, _k, _v)
        _pregate_decision.starvation_age_source_s = _pg_age_s
        _pregate_decision.starvation_age_source_frames = int(gap)
        _pregate_decision.starvation_age_basis = _pg_basis
        _pregate_decision.effective_min_translation_m = float(cfg.min_translation_m)
        return self._finalize_trace(_pregate_decision, _trace)

    def admit(
        self,
        cam: Any,
        frame: Any,
    ) -> Tuple[KeyframeDecision, Optional[Any]]:
        """Decide whether to admit *frame* as a keyframe.

        Thin wrapper over :meth:`_admit_impl` that owns two recording-only
        concerns, neither of which can change the decision:

        * the issue-#794 precedence trace -- creates the
          :class:`PrecedenceTrace`, lets the decision path fill it in, then
          stamps ``final_owner`` and attaches it to the returned
          :class:`KeyframeDecision` (``decision.trace``). See
          :mod:`eligsir.mapping.scheduler.precedence` for the contract.
        * the issue-#793/#795 velocity-controller telemetry -- stamps the
          pending per-call velocity fields onto the returned decision.

        The decision itself is produced entirely by ``_admit_impl``.
        """
        trace = PrecedenceTrace(scope=self.force_scope.value)
        decision, render_pkg = self._admit_impl(cam, frame, trace)
        _tel = getattr(self, "_pending_controller_telemetry", None)
        if _tel:
            for _k, _v in _tel.items():
                setattr(decision, _k, _v)
        self._pending_controller_telemetry = None
        # #946 backstop telemetry: the static resolved thresholds (identical
        # for every candidate) plus this call's parked per-candidate age.
        for _k, _v in self._backstop_static_telemetry.items():
            if hasattr(decision, _k):
                setattr(decision, _k, _v)
        _bs = getattr(self, "_pending_backstop_telemetry", None)
        if _bs:
            for _k, _v in _bs.items():
                setattr(decision, _k, _v)
        self._pending_backstop_telemetry = None
        self._precedence_trigger_counts[trace.candidate_trigger] = (
            self._precedence_trigger_counts.get(trace.candidate_trigger, 0) + 1
        )
        self._finalize_trace(decision, trace)
        return decision, render_pkg

    def _admit_impl(
        self,
        cam: Any,
        frame: Any,
        trace: PrecedenceTrace,
    ) -> Tuple[KeyframeDecision, Optional[Any]]:
        """Decide whether to admit *frame* as a keyframe. Call :meth:`admit`.

        The admitter updates its internal train-frame reference
        (formerly :meth:`note_train_frame`) atomically on admission, so
        callers do not need to call any follow-up — the API is single-
        shot.

        Args:
            cam: Camera wrapper passed to the render function.
            frame: Source frame exposing ``.c2w`` and ``.index``.

        Returns:
            ``(decision, render_pkg)`` — ``decision`` is the typed
            :class:`KeyframeDecision` and ``render_pkg`` is the dict
            returned by the (successful) render call. Callers can
            cache it on the camera as
            ``cam._streaming_admission_render_pkg`` to skip a second
            render in the depth-consistency / training path.
        """
        cfg = self._config
        t_m, r_deg, gap = self._pose_delta_from_last_train(frame)

        # #794: classify the candidate trigger up front. This is the single
        # place the soft/hard max_gap split is decided; everything downstream
        # reads `trigger`, never re-derives the thresholds.
        #
        # #946: the thresholds themselves were resolved ONCE at construction,
        # as two INDEPENDENT quantities (see
        # precedence.resolve_backstop_thresholds). Under the default
        # `legacy_frames` mode this is exactly the pre-#946 expression
        # `resolve_max_gap_trigger(gap, max_gap, max_gap_hard_mult)`, and
        # `forced` is exactly `gap >= max_gap` -- asserted bit-for-bit by
        # tests/test_issue946_backstop_decoupling.py's legacy-reproduction
        # family.
        _starv_age_s, _starv_basis = self._starvation_age(frame, gap)
        # The two crossings are read INDEPENDENTLY, not re-derived from
        # `trigger` -- that enum is lossy (HARD wins outright), and with a
        # mixed-domain configuration "hard fired" does not imply "soft fired".
        # See precedence.backstop_crossings for the concrete case.
        _soft_triggered, _hard_triggered = backstop_crossings(
            self._backstop, gap, _starv_age_s
        )
        trigger = resolve_backstop_trigger(self._backstop, gap, _starv_age_s)
        forced = trigger is not ForcePath.NORMAL
        trace.candidate_trigger = trigger.value
        trace.hard_force = trigger is ForcePath.HARD_MAX_GAP_CEILING
        scope = self.force_scope

        # Novelty-gated max_gap (Component 2): a would-be forced frame that reveals
        # no NEW geometry is deferred rather than force-admitted, so a long forward
        # walk over already-mapped space stops inflating the training-view count.
        # Only downgrades forcing below the hard ceiling and only when the coverage
        # machinery is wired and evaluable; otherwise forcing is unchanged. A hard
        # ceiling (gap >= max_gap*max_gap_hard_mult) always forces so admission can
        # never fully starve. Dataclass-default off => this block is skipped, but
        # the real CLI default is on (see max_gap_require_novelty's own comment
        # above) -- this block runs on every real training run since 2026-08-15.
        #
        # Issue #928: before this fix, `forced` stayed True under TWO byte-
        # identical-looking conditions with nothing downstream able to tell
        # them apart: (1) the gate ran and the frame passed
        # (_fnv >= min_new_voxel_frac), or (2) the gate never ran / abstained
        # (coverage_gain_gate off, occupancy_fn/load_depth_m unwired, gap past
        # the hard ceiling, or _coverage_new_voxel_frac() returning None).
        # `_record_novelty_gate_outcome` below makes the abstention explicit
        # on every would-be-forced candidate (`forced` at this point in the
        # function, before any downstream reclassification) -- purely
        # additive telemetry that never changes `forced`/`trigger` from what
        # the equivalent nested-if would have produced.
        if forced:
            if not cfg.max_gap_require_novelty or not cfg.coverage_gain_gate:
                self._record_novelty_gate_outcome(trace, "not_evaluated", "gate_disabled")
            elif self._occupancy_fn is None:
                self._record_novelty_gate_outcome(trace, "not_evaluated", "no_occupancy_fn")
            elif self._load_depth_m is None:
                self._record_novelty_gate_outcome(trace, "not_evaluated", "no_depth_loader")
            elif _hard_triggered:
                # #946: reads the resolved hard ceiling rather than
                # re-deriving `max_gap * max_gap_hard_mult` inline. Identical
                # predicate under the default legacy_frames mode.
                self._record_novelty_gate_outcome(trace, "not_evaluated", "hard_ceiling")
            else:
                _fnv = self._coverage_new_voxel_frac(cam, frame)
                trace.record_evaluated(Gate.COVERAGE_NOVELTY)
                trace.record_coverage_consumer(
                    "max_gap_novelty_veto",
                    status="evaluated" if _fnv is not None else "unavailable",
                    value=_fnv,
                    none_cause=self._last_coverage_none_cause,
                )
                if _fnv is None:
                    self._record_novelty_gate_outcome(
                        trace,
                        "not_evaluated",
                        "frac_unavailable",
                        none_cause=self._last_coverage_none_cause,
                    )
                elif _fnv < cfg.min_new_voxel_frac:
                    trace.record_observation("coverage_new_voxel_fraction", _fnv)
                    self._record_novelty_gate_outcome(trace, "evaluated_redundant", fraction=_fnv)
                    # PR #826: the force is withdrawn -- this candidate re-enters
                    # the ordinary path and is judged by every gate.
                    forced = False
                    trigger = ForcePath.NORMAL
                    trace.candidate_trigger = trigger.value
                else:
                    trace.record_observation("coverage_new_voxel_fraction", _fnv)
                    self._record_novelty_gate_outcome(trace, "evaluated_novel", fraction=_fnv)

        # Velocity-adaptive threshold (#793-corrected). Sample the INSTANTANEOUS
        # camera velocity between this candidate and the previous CANDIDATE (in
        # m/s, from real source timestamps where they are trustworthy and from
        # frame_delta/input_fps otherwise), then derive
        #
        #     effective_min_trans [m] = velocity_ema [m/s] * (1 / target_admit_rate_fps) [s]
        #
        # so a frame is admitted by motion once the camera has travelled about
        # as far as it travels in one target admit interval -> the admit RATE
        # tracks target_admit_rate_fps regardless of scene speed. Faster motion
        # -> larger velocity_ema -> HIGHER effective threshold (rate held
        # constant, NOT count).
        #
        # PLACEMENT IS LOAD-BEARING: this block sits before every early return
        # in admit() (forced, pregate, coverage, render gates), so the estimator
        # sees every candidate and reads the same velocity whether frames are
        # being admitted or rejected. Do not move it into a branch, and do not
        # advance the previous-candidate reference from _record_admission.
        #
        # With velocity_adaptive=False nothing below runs and the effective
        # threshold is exactly cfg.min_translation_m (byte-identical path).
        effective_min_trans = cfg.min_translation_m
        _vel: dict = {}
        _threshold_clamp = "n/a"
        _target_interval_s: Optional[float] = None
        _target_interval_source_frames: Optional[float] = None
        if cfg.velocity_adaptive:
            _vel = self._update_velocity_estimate(frame)
            _target_interval_s = 1.0 / max(float(cfg.target_admit_rate_fps), 0.01)
            _source_fps = float(cfg.input_fps)
            _implied_fps = _vel.get("implied_input_fps")
            if (
                _vel.get("timebase_source") == "source_timestamp"
                and _implied_fps is not None
                and math.isfinite(float(_implied_fps))
                and float(_implied_fps) > 0.0
            ):
                _source_fps = float(_implied_fps)
            _target_interval_source_frames = _source_fps * _target_interval_s
            _v_ema = self._translation_velocity_ema_m_s
            if _v_ema is not None:
                _raw = float(_v_ema) * _target_interval_s
                _lo = float(cfg.velocity_adaptive_min_trans_floor)
                _hi = float(cfg.velocity_adaptive_min_trans_ceil)
                effective_min_trans = float(np.clip(_raw, _lo, _hi))
                self._threshold_evals += 1
                if _raw < _lo:
                    _threshold_clamp = "floor"
                    self._threshold_floor_hits += 1
                elif _raw > _hi:
                    _threshold_clamp = "ceil"
                    self._threshold_ceil_hits += 1
                else:
                    _threshold_clamp = "none"

        # Park this call's controller state for the public admit() wrapper to
        # stamp onto whichever decision _admit_impl returns. Doing it in the
        # wrapper (rather than at each of the ~10 return sites in this method)
        # keeps the telemetry strictly additive and impossible to forget on a
        # branch -- and guarantees it cannot perturb decision ordering.
        self._pending_controller_telemetry = {
            "translation_velocity_m_s": _vel.get("translation_velocity_m_s"),
            "rotation_velocity_deg_s": _vel.get("rotation_velocity_deg_s"),
            "translation_velocity_ema_m_s": self._translation_velocity_ema_m_s,
            "rotation_velocity_ema_deg_s": self._rotation_velocity_ema_deg_s,
            "source_dt_s": _vel.get("source_dt_s"),
            "source_frame_delta": _vel.get("source_frame_delta"),
            "timebase_source": _vel.get("timebase_source", "unavailable"),
            "implied_input_fps": _vel.get("implied_input_fps"),
            "effective_min_translation_m": float(effective_min_trans),
            "controller_target_interval_s": _target_interval_s,
            "controller_target_interval_source_frames": _target_interval_source_frames,
            "threshold_clamp": _threshold_clamp,
        }

        _trans_ok = t_m >= effective_min_trans
        _rot_ok = r_deg >= cfg.min_rotation_deg
        moved = _trans_ok or _rot_ok

        # #946 per-decision backstop telemetry. Parked here -- after the
        # post-clamp `effective_min_trans` and `moved` exist, and BEFORE every
        # return site in this method -- and stamped in the `admit()` wrapper,
        # exactly like `_pending_controller_telemetry` above. Doing it at the
        # ~10 KeyframeDecision construction sites instead would leave holes on
        # the early `reason="max_gap"` return and the quality-scorer exit, and
        # blank rows on rejections, which is precisely where starvation age is
        # the interesting number.
        #
        # `soft/hard_backstop_triggered` record the RAW threshold crossings,
        # deliberately NOT re-derived from `trigger` (which #826's novelty veto
        # may withdraw to NORMAL further up, and #934's novelty force may
        # overwrite). Comparing this pair against
        # `trace["candidate_trigger"]` is what makes such a withdrawal
        # visible in the sidecar instead of silent.
        self._pending_backstop_telemetry = {
            "starvation_age_source_s": _starv_age_s,
            "starvation_age_source_frames": int(gap),
            "starvation_age_basis": _starv_basis,
            "soft_backstop_triggered": bool(_soft_triggered),
            "hard_backstop_triggered": bool(_hard_triggered),
            "motion_trigger": (
                "both" if (_trans_ok and _rot_ok)
                else "translation" if _trans_ok
                else "rotation" if _rot_ok
                else "none"
            ),
        }

        # #794 soft-starvation ESCALATION (opt-in, scope=motion_only).
        #
        # A soft max_gap force means "cheap motion filtering may no longer
        # suppress this candidate" -- NOT "this observation is novel, sharp and
        # useful". Under this scope we therefore withdraw the unconditional
        # force below and instead mark motion as SATISFIED, letting the
        # candidate continue through the parallax/exposure/blur/render-overlap/
        # quality gates. Satisfied, not skipped: a forced-but-stationary frame
        # has moved=False and novel=False, so simply "falling through" would
        # hit `overlap_ok and (moved or novel)` and be rejected as redundant --
        # silently reinstating the starvation the backstop exists to prevent.
        # `motion_ok` is what the downstream conjunction reads from here on.
        #
        # The hard ceiling is untouched: it keeps the unconditional short
        # circuit under every scope.
        motion_forced = False
        if (
            forced
            and trigger is ForcePath.SOFT_MAX_GAP
            and scope is MaxGapForceScope.MOTION_ONLY
        ):
            motion_forced = True
            forced = False
            self._record_bypasses(trace, ForcePath.SOFT_MAX_GAP)
        motion_ok = bool(moved or motion_forced)

        # #934 NOVELTY-DRIVEN ADMISSION (opt-in, novelty_force_admit).
        #
        # Everything above this line is why the coverage/novelty gate was
        # architecturally VETO-ONLY: its two existing call sites both only ever
        # REJECT (withdraw a max_gap force at the top of this method; reject
        # below min_new_voxel_frac at step 1c), and step 1b's motion pregate
        # returns "no_motion" BEFORE step 1c ever runs -- so for a non-moving
        # candidate the coverage machinery was not merely non-decisive, it was
        # never consulted at all. Measured on two fr3 GPU runs: ~96% of
        # admissions owned by the motion gate, 4-5 ever owned by novelty.
        #
        # This block gives the signal a way to CAUSE an admission. Direct voxel
        # evidence of unmapped geometry treats the MOTION criterion as
        # SATISFIED (not skipped -- same reasoning as the #794 soft-escalation
        # block above: a "skipped" motion gate would leave moved=False and hit
        # `overlap_ok and (moved or novel)` and be rejected as redundant).
        # Every remaining gate still binds; this is a trigger, not a
        # short-circuit.
        #
        # PLACEMENT: deliberately after the velocity block (the stale-pose
        # guard reads its per-candidate sample) and before step 1b's pregate
        # short-circuit -- hoisting it any lower would make it unreachable for
        # exactly the population it targets.
        novelty_forced = False
        # Set only when this block actually measures the fraction, so step 1c
        # below can reuse it instead of backprojecting the same candidate twice.
        _novelty_force_frac_measured: Optional[float] = None
        # Stale-pose guard input, read once (see the `stale_pose` branch).
        _vel_trans = _vel.get("translation_velocity_m_s")
        try:
            _vel_trans = float(_vel_trans) if _vel_trans is not None else None
        except (TypeError, ValueError):
            _vel_trans = None
        if not cfg.novelty_force_admit:
            self._record_novelty_force_outcome(trace, "not_evaluated", "disabled")
        elif trigger is not ForcePath.NORMAL:
            # A max_gap candidate keeps its own trigger: candidate_trigger stays
            # single-valued and the precedence buckets stay mutually exclusive
            # (the #932 double-counting class).
            self._record_novelty_force_outcome(trace, "not_evaluated", "not_normal_trigger")
        elif motion_ok:
            # Motion would admit anyway -- the trigger is moot, and skipping it
            # here is also what keeps the hoisted coverage cost off the
            # already-admitting candidates.
            self._record_novelty_force_outcome(trace, "not_evaluated", "motion_already_ok")
        elif self._last_train_c2w is None:
            # Pre-baseline: pose deltas have no reference and the starvation
            # clock (#933) owns admission. Never fire here.
            self._record_novelty_force_outcome(trace, "not_evaluated", "no_baseline")
        elif self._occupancy_fn is None or self._load_depth_m is None:
            self._record_novelty_force_outcome(trace, "not_evaluated", "machinery_unwired")
        elif cfg.novelty_force_require_fresh_pose and not (
            _vel_trans is not None and _vel_trans > 0.0
        ):
            # Stale-pose guard (fail-closed). See novelty_force_require_fresh_pose's
            # config comment: a fresh image carrying a STALE pose backprojects
            # its depth at the wrong pose, lands in genuinely-unobserved voxels,
            # and reads as HIGH novelty -- so an unguarded trigger would
            # preferentially re-admit exactly the frames the motion pregate
            # exists to exclude.
            #
            # This reads _update_velocity_estimate's RETURNED sample, not
            # `frame.c2w` vs `self._prev_candidate_c2w`: that reference is
            # advanced UNCONDITIONALLY inside _update_velocity_estimate, which
            # already ran above, so the naive comparison would compare a value
            # against itself and be silently inert.
            #
            # `None` (velocity_adaptive off, first candidate, or no usable dt)
            # lands here too, by design: refuse to fire rather than fall
            # through unguarded.
            self._record_novelty_force_outcome(trace, "not_evaluated", "stale_pose")
        else:
            _nf_frac = self._coverage_new_voxel_frac(cam, frame)
            trace.record_evaluated(Gate.COVERAGE_NOVELTY)
            trace.record_coverage_consumer(
                "novelty_force",
                status="evaluated" if _nf_frac is not None else "unavailable",
                value=_nf_frac,
                none_cause=self._last_coverage_none_cause,
            )
            if _nf_frac is None:
                self._record_novelty_force_outcome(
                    trace,
                    "not_evaluated",
                    "frac_unavailable",
                    none_cause=self._last_coverage_none_cause,
                )
            elif _nf_frac < cfg.novelty_force_frac:
                trace.record_observation("coverage_new_voxel_fraction", _nf_frac)
                self._record_novelty_force_outcome(
                    trace, "below_threshold", fraction=_nf_frac
                )
            else:
                trace.record_observation("coverage_new_voxel_fraction", _nf_frac)
                self._record_novelty_force_outcome(trace, "triggered", fraction=_nf_frac)
                novelty_forced = True
                motion_ok = True
                trigger = ForcePath.NOVELTY_FORCE
                trace.candidate_trigger = trigger.value
                # Records the MOTION bypass (and nothing else -- NOVELTY_FORCE's
                # bypass set is {MOTION} under every scope).
                self._record_bypasses(trace, ForcePath.NOVELTY_FORCE)
            # Reuse this backprojection at step 1c rather than paying for a
            # second one on the same candidate. `None` means "not measured
            # here", which is exactly what step 1c's own guard already handles.
            _novelty_force_frac_measured = _nf_frac

        # Cheap score: gap dominates, then pose delta.
        cheap_score = 0.0
        if forced:
            cheap_score = 1.0
        else:
            cheap_score = min(1.0, 0.4 * (t_m / max(effective_min_trans, 1e-6))
                              + 0.4 * (r_deg / max(cfg.min_rotation_deg, 1e-6))
                              + 0.2 * min(1.0, gap / cfg.max_gap))

        # 1. Cheap fast-path: forced (max_gap) → admit unconditionally.
        #
        # #794: this is the unconditional-force path. It is reached by the hard
        # starvation ceiling under every scope, and by a soft max_gap force
        # under the legacy (default) scope only -- the latter being precisely
        # the bypass #794 names. Every gate skipped here is now recorded on the
        # decision's trace and in the admitter's bypass counters, so the bypass
        # is visible even where the default preserves it.
        if forced:
            self._record_bypasses(trace, trigger)
            decision = KeyframeDecision(
                admitted=True,
                reason="max_gap",
                score=1.0,
                cheap_score=1.0,
                forced=True,
                moved=moved,
                pose_delta_m=t_m,
                pose_delta_deg=r_deg,
                gap=gap,
            )
            self._record_admission(decision, frame)
            return decision, None

        # 1b. Cheap motion pre-gate (optional): if the camera did not move past
        # the keyframe motion thresholds and we are not force-admitting, reject
        # WITHOUT rendering. A non-moved frame has ~1.0 render overlap, so the
        # expensive overlap gate below would reject it anyway — this is
        # admission-preserving for the no-motion case and removes the ~2.6 ms
        # admission render from every slow-motion frame (the dominant live
        # during-ingestion GPU cost; see docs/2026-06-30-live-ingestion-gil-contention.md).
        if cfg.motion_pregate and self._last_train_c2w is not None and not motion_ok:
            # Only short-circuit once a baseline keyframe exists. With no baseline
            # yet, _pose_delta_from_last_train returns (0,0,0) so `moved` is falsely
            # False — short-circuiting here would reject the very first frame, the
            # baseline would never initialize, and EVERY subsequent frame would read
            # delta 0 (permanent no_motion deadlock → map collapses to bootstrap).
            # Let the first frame fall through to the render gate, which admits it
            # and records the baseline; thereafter the pregate works normally.
            decision = KeyframeDecision(
                admitted=False,
                reason="no_motion",
                score=cheap_score,
                cheap_score=cheap_score,
                forced=False,
                moved=False,
                pose_delta_m=t_m,
                pose_delta_deg=r_deg,
                gap=gap,
            )
            # Do NOT _record_admission here: this frame is REJECTED. Recording
            # would advance the pose baseline (_last_train_c2w via note_train_frame)
            # to every rejected frame, so the next frame's delta is measured against
            # the immediately-previous frame (~2-3cm at 30fps) instead of the last
            # admitted keyframe — motion would never accumulate past the threshold
            # and EVERY frame would read no_motion, collapsing the map to the
            # bootstrap view. The baseline must only move on actual admission (as the
            # render-gate reject path below also does: it does not record either).
            return decision, None

        # 1c. Depth-coverage gain gate (Track A, ported from feat/offline-gap):
        # backproject coarse strided depth into world space, query the scene
        # occupancy hash, and reject frames whose NEW-voxel fraction falls below
        # min_new_voxel_frac before paying the ~67ms render cost. Only active when
        # coverage_gain_gate=True AND occupancy_fn + load_depth_m are wired — off
        # (default) this whole block is skipped, so the path is byte-identical.
        # NOTE (unverified runtime assumption): the cam.fx/cx vs FoVx fallback and
        # cam.c2w vs world_view_transform inversion below only execute when the gate
        # is ON with a real GPU camera; not exercised by the CPU unit tests.
        _new_voxel_frac_for_parallax: Optional[float] = None
        if cfg.coverage_gain_gate and (
            self._occupancy_fn is None or self._load_depth_m is None
        ):
            trace.record_coverage_consumer(
                "ordinary_coverage_gate",
                status="skipped",
                precondition_skip_reason="machinery_unwired",
            )
        if (
            (cfg.coverage_gain_gate or cfg.parallax_gate)
            and self._occupancy_fn is not None
            and self._load_depth_m is not None
        ):
            # #934: reuse the fraction the novelty-force block above already
            # measured for THIS candidate rather than backprojecting it twice.
            # Same function, same inputs, no intervening state change, so this
            # is value-identical; it only removes duplicated work. `None` means
            # that block did not measure it (flag off, or it short-circuited on
            # one of its guards), so fall back to computing it here as before.
            _new_voxel_frac = (
                _novelty_force_frac_measured
                if _novelty_force_frac_measured is not None
                else self._coverage_new_voxel_frac(cam, frame)
            )
            _new_voxel_frac_for_parallax = _new_voxel_frac
            if cfg.coverage_gain_gate:
                trace.record_evaluated(Gate.COVERAGE_NOVELTY)
                trace.record_coverage_consumer(
                    "ordinary_coverage_gate",
                    status=("evaluated" if _new_voxel_frac is not None else "unavailable"),
                    value=_new_voxel_frac,
                    none_cause=self._last_coverage_none_cause,
                )
            if cfg.parallax_gate:
                trace.record_coverage_consumer(
                    "parallax_gate",
                    status=("evaluated" if _new_voxel_frac is not None else "unavailable"),
                    value=_new_voxel_frac,
                    none_cause=self._last_coverage_none_cause,
                )
            if _new_voxel_frac is not None:
                trace.record_observation("coverage_new_voxel_fraction", _new_voxel_frac)
            if cfg.coverage_gain_gate and _new_voxel_frac is not None and _new_voxel_frac < cfg.min_new_voxel_frac:
                _db_result = self._db_diversity_score(cam)
                if _db_result is not None:
                    _overlap_frac, _angle_deg = _db_result
                    _ANGLE_DIVERSITY_THRESHOLD_DEG = 20.0
                    # #937: _overlap_frac is an overlap/covisibility fraction from
                    # self._db_diversity_score(cam) -- the semantically correct
                    # threshold for it is cfg.min_overlap (used identically at
                    # keyframes.py's other overlap comparisons), not
                    # cfg.min_new_voxel_frac (a *novelty* threshold, already
                    # consumed by the outer `< cfg.min_new_voxel_frac` test above).
                    if _overlap_frac >= cfg.min_overlap and _angle_deg >= _ANGLE_DIVERSITY_THRESHOLD_DEG:
                        decision = KeyframeDecision(
                            admitted=True,
                            reason="db_diverse_revisit",
                            score=_overlap_frac,
                            cheap_score=cheap_score,
                            # #794: a soft escalation that reaches this exit is
                            # still a force-triggered candidate -- carry the
                            # flag so #795's forced_soft attribution is exact on
                            # every admitting exit, not just the primary one.
                            forced=bool(motion_forced),
                            # #934: unreachable for a novelty-forced candidate
                            # today (this branch requires new_voxel_frac <
                            # min_new_voxel_frac, while the trigger requires
                            # >= novelty_force_frac > min_new_voxel_frac -- the
                            # invariant __post_init__ enforces). Carried anyway
                            # so the flag cannot silently go missing if either
                            # threshold relationship is ever revisited.
                            novelty_forced=novelty_forced,
                            moved=moved,
                            pose_delta_m=t_m,
                            pose_delta_deg=r_deg,
                            gap=gap,
                        )
                        self._record_admission(decision, frame)
                        return decision, None
                    decision = KeyframeDecision(
                        admitted=False,
                        reason="rej_redundant_angle",
                        score=_overlap_frac,
                        cheap_score=cheap_score,
                        forced=False,
                        moved=moved,
                        pose_delta_m=t_m,
                        pose_delta_deg=r_deg,
                        gap=gap,
                    )
                    self._keyframes_rejected += 1
                    self._keyframes_rejected_coverage += 1
                    return decision, None
                decision = KeyframeDecision(
                    admitted=False,
                    reason="rej_coverage",
                    score=_new_voxel_frac,
                    cheap_score=cheap_score,
                    forced=False,
                    moved=moved,
                    pose_delta_m=t_m,
                    pose_delta_deg=r_deg,
                    gap=gap,
                )
                self._keyframes_rejected += 1
                self._keyframes_rejected_coverage += 1
                return decision, None

        # 1c2. Parallax-aware covisibility gate (5A, golden-rule admission):
        # reject frames that mostly re-see known voxels from nearly the same
        # rays (forward-walk flood). Never rejects a forced frame — the
        # max_gap backstop guarantees admission can't starve.
        if (
            cfg.parallax_gate
            and not forced
            and _new_voxel_frac_for_parallax is not None
        ):
            trace.record_evaluated(Gate.PARALLAX)
            _covis_frac = 1.0 - _new_voxel_frac_for_parallax
            if _covis_frac >= cfg.parallax_covis_max:
                _med_par = self._median_parallax_deg()
                if _med_par is not None and _med_par < cfg.parallax_min_deg:
                    decision = KeyframeDecision(
                        admitted=False,
                        reason="rej_parallax",
                        score=_med_par,
                        cheap_score=cheap_score,
                        forced=False,
                        # #934: a triggered candidate CAN reach this exit (and
                        # the exposure/blur exits below). These rejected-but-
                        # triggered rows are exactly the population the
                        # `triggered`-vs-`decisive` gap is meant to expose, so
                        # the flag must survive here -- otherwise the sidecar's
                        # `novelty_forced` column reads 0 on a row whose own
                        # `precedence.novelty_force_outcome` says "triggered".
                        novelty_forced=novelty_forced,
                        moved=moved,
                        pose_delta_m=t_m,
                        pose_delta_deg=r_deg,
                        gap=gap,
                    )
                    self._keyframes_rejected += 1
                    self._keyframes_rejected_parallax = getattr(self, "_keyframes_rejected_parallax", 0) + 1
                    return decision, None

        # 1d. Exposure gate (cheap): reject frames whose mean luminance has shifted
        # too far from the last admitted frame — proxy for auto-exposure jumps that
        # poison photometric supervision. Force-admit after a starvation window so a
        # sustained lighting change cannot deadlock admission. Default off.
        self._exposure_iter += 1
        _current_luma: Optional[float] = None
        if cfg.exposure_gate_enabled:
            try:
                from eligsir.mapping.scheduler.frame_quality import (
                    mean_luma as _mean_luma,
                    exposure_reject as _exposure_reject,
                )
                if self._evaluate_luma is not None:
                    _current_luma = self._evaluate_luma(frame)
                else:
                    _rgb = self._load_frame_rgb(frame)
                    if _rgb is not None:
                        _current_luma = _mean_luma(_rgb)
            except Exception:
                _current_luma = None

            if _current_luma is not None and self._last_admitted_luma is not None:
                trace.record_evaluated(Gate.EXPOSURE)
                trace.record_observation(
                    "exposure_delta",
                    abs(_current_luma - self._last_admitted_luma)
                    / max(abs(self._last_admitted_luma), 1e-6),
                )
                _iters_since_admit = self._exposure_iter - self._last_admitted_iter
                # Single evaluation, reused by both the starvation-relief
                # bookkeeping and the rejection test (was one call; keep it one).
                _exposure_bad = _exposure_reject(
                    _current_luma, self._last_admitted_luma, cfg.exposure_max_delta
                )
                if _exposure_bad and _iters_since_admit >= cfg.exposure_force_admit_iters:
                    # Exposure starvation relief: the ONLY gate this trigger
                    # may bypass is the exposure gate itself (#794 table).
                    self._record_bypasses(trace, ForcePath.EXPOSURE_STARVATION)
                if _exposure_bad and _iters_since_admit < cfg.exposure_force_admit_iters:
                    decision = KeyframeDecision(
                        admitted=False,
                        reason="exposure_reject",
                        score=0.0,
                        cheap_score=cheap_score,
                        # #934: see the rej_parallax exit above.
                        novelty_forced=novelty_forced,
                        moved=moved,
                        pose_delta_m=t_m,
                        pose_delta_deg=r_deg,
                        gap=gap,
                    )
                    self._keyframes_rejected += 1
                    return decision, None
        # Store luma tentatively; _record_admission updates the reference on success.
        self._pending_luma = _current_luma

        # 1d. Blur gate (cheap): hard-reject frames below a Laplacian-variance
        # threshold (motion-blurred RGB + its paired noisy depth). Runs on a
        # downsampled greyscale crop. Force-admit after a starvation window. Off by default.
        self._blur_iter += 1
        if cfg.blur_gate_enabled:
            _blur_score: Optional[float] = None
            try:
                from eligsir.mapping.scheduler.frame_quality import (
                    blur_laplacian_score as _blur_laplacian_score,
                )
                if self._evaluate_blur is not None:
                    _blur_score = self._evaluate_blur(frame)
                else:
                    _rgb_blur = self._load_frame_rgb(frame)
                    if _rgb_blur is not None:
                        _blur_score = _blur_laplacian_score(
                            np.asarray(_rgb_blur), cfg.blur_gate_downsample
                        )
            except Exception:
                _blur_score = None

            if _blur_score is not None:
                trace.record_evaluated(Gate.BLUR)
                trace.record_observation("blur_laplacian", _blur_score)
            if _blur_score is not None and _blur_score < cfg.blur_gate_min_laplacian:
                _iters_since_blur_admit = self._blur_iter - self._last_admitted_iter_blur
                if _iters_since_blur_admit >= cfg.blur_gate_force_admit_iters:
                    # Blur starvation relief: bypasses the blur gate only.
                    self._record_bypasses(trace, ForcePath.BLUR_STARVATION)
                if _iters_since_blur_admit < cfg.blur_gate_force_admit_iters:
                    decision = KeyframeDecision(
                        admitted=False,
                        reason="blur_reject",
                        score=0.0,
                        cheap_score=cheap_score,
                        # #934: see the rej_parallax exit above.
                        novelty_forced=novelty_forced,
                        moved=moved,
                        pose_delta_m=t_m,
                        pose_delta_deg=r_deg,
                        gap=gap,
                    )
                    self._keyframes_rejected += 1
                    return decision, None

        # 2. Expensive gate: render-based overlap.
        overlap: Optional[float] = None
        render_pkg: Optional[Any] = None
        render_failed = False
        if self._render_fn is not None:
            try:
                # torch is REQUIRED in production (the real renderer is
                # gsplat), and no_grad() is load-bearing there: without it this
                # render would build an autograd graph on the training model
                # every candidate frame. But the gate itself only needs *some*
                # object exposing the alpha contract, so a torch-free render_fn
                # (as used by the #794 precedence tests) must not be silently
                # skipped just because torch is absent -- the bare `except
                # Exception` below would swallow the ImportError and quietly
                # disable the whole render-overlap gate. Degrade the context
                # manager only; when torch IS importable this is exactly the
                # previous behaviour.
                try:
                    import torch
                    _no_grad = torch.no_grad()
                except ImportError:
                    from contextlib import nullcontext
                    _no_grad = nullcontext()
                with _no_grad:
                    render_pkg = self._render_fn(
                        cam,
                        self._gaussians,
                        self._pipe,
                        self._background,
                        render_depth=True,
                    )
                if render_pkg is None:
                    render_failed = True
                else:
                    alpha = render_pkg.get("alpha") if hasattr(render_pkg, "get") else None
                    if alpha is not None:
                        overlap = float((alpha.detach() > cfg.coverage_alpha).float().mean().item())
            except Exception:
                render_failed = True
                render_pkg = None
        else:
            render_failed = True

        if render_failed:
            decision = KeyframeDecision(
                admitted=bool(motion_ok),
                # #934: motion_ok can now be satisfied by the novelty trigger
                # rather than by real motion, so this exit must not label such
                # an admission "pose_delta" -- that would silently credit the
                # motion gate for an admission it did not cause, which is the
                # exact attribution error this whole change exists to fix.
                reason=(
                    "render_failed" if not motion_ok
                    else ("novelty_admit" if (novelty_forced and not moved) else "pose_delta")
                ),
                score=cheap_score,
                cheap_score=cheap_score,
                forced=False,
                novelty_forced=novelty_forced,
                moved=moved,
                render_overlap=None,
                pose_delta_m=t_m,
                pose_delta_deg=r_deg,
                gap=gap,
            )
            if decision.admitted:
                self._record_admission(decision, frame)
            else:
                self._keyframes_rejected += 1
            return decision, None

        trace.record_evaluated(Gate.RENDER_OVERLAP)
        overlap_ok = (overlap is not None) and overlap >= cfg.min_overlap
        novel = (overlap is not None) and overlap <= cfg.max_overlap
        # `motion_ok` (not `moved`) so a soft max_gap escalation under
        # scope=motion_only counts as motion-SATISFIED here -- see the #794
        # escalation block above. Under the default scope motion_ok == moved.
        admitted = bool(overlap_ok and (motion_ok or novel))

        # Low-overlap starvation relief (see KeyframeAdmitterConfig.low_overlap_
        # force_admit_iters docstring): a candidate that failed only the
        # overlap_ok floor (i.e. would otherwise be rejected "low_overlap") is
        # admitted anyway once starved for the configured number of iters, so a
        # frontier region the map hasn't grown into yet cannot permanently
        # block its own admission. Does not touch moved/forced rejections.
        low_overlap_forced = False
        if (
            not admitted
            and not forced
            and not overlap_ok
            and cfg.low_overlap_force_admit_iters > 0
        ):
            _iters_since_admit_lo = self._exposure_iter - self._last_admitted_iter
            if _iters_since_admit_lo >= cfg.low_overlap_force_admit_iters:
                admitted = True
                low_overlap_forced = True
                # Low-overlap starvation relief bypasses the render-overlap
                # gate only (#794 table) -- the quality gate below still binds.
                self._record_bypasses(trace, ForcePath.LOW_OVERLAP_STARVATION)

        # 3. Optional quality scoring.
        #
        # FIXED 2026-07-31 (bootstrap deadlock): must NOT run before a
        # baseline keyframe exists. score_keyframe's composite score blends
        # translation/rotation/overlap/quality terms; at the very first
        # candidate frame t_m=r_deg=0 (no baseline to measure a delta
        # against, same as pregate()'s "no baseline yet" case above), which
        # can drag the composite score below final_threshold and reject the
        # frame even with a perfect quality_weight=1.0 -- confirmed via a
        # deterministic repro (quality_enabled=True flips frame 0 from
        # admitted="new_content" to rejected="redundant" with an identical
        # scene/render setup, identical to quality_enabled=False otherwise).
        # If the very first admittable frame is rejected this way,
        # `_record_admission` never fires, `_last_train_c2w` stays None
        # forever, `_pose_delta_from_last_train` returns (0,0,0) for every
        # subsequent frame, `forced`/`moved` can never become True, and
        # admission deadlocks permanently (observed live: floor2 bag replay,
        # 0/1991 admitted with --streaming_keyframe_quality_enabled, vs
        # ~186/1991 with it off). Exempting the pre-baseline case mirrors
        # pregate()'s own "no baseline yet -- let it fall through and admit
        # via render gate" rule above.
        quality = None
        if (
            self._evaluate_quality is not None
            and admitted
            and self._last_train_c2w is not None
            and self._load_depth_m is not None
            and self._load_rgb is not None
        ):
            try:
                rgb_np = np.asarray(self._load_rgb(frame))
                depth_np = self._load_depth_m(frame)
                quality = self._evaluate_quality(rgb_np, depth_np, frame)
                trace.record_observation(
                    "quality", getattr(quality, "quality_weight", None)
                )
                if cfg.quality_enabled:
                    trace.record_evaluated(Gate.QUALITY_SCORE)
                    # new_surface_ratio: reuse the coverage-gate backprojection
                    # when it's already wired (occupancy_fn + load_depth_m);
                    # falls back to 0.0 (no signal) when unavailable.
                    new_surface_ratio = 0.0
                    if self._occupancy_fn is not None and self._load_depth_m is not None:
                        _nsr = self._coverage_new_voxel_frac(cam, frame)
                        trace.record_coverage_consumer(
                            "quality_new_surface_ratio",
                            status="evaluated" if _nsr is not None else "unavailable",
                            value=_nsr,
                            none_cause=self._last_coverage_none_cause,
                        )
                        if _nsr is not None:
                            new_surface_ratio = _nsr
                    else:
                        trace.record_coverage_consumer(
                            "quality_new_surface_ratio",
                            status="skipped",
                            precondition_skip_reason="machinery_unwired",
                        )

                    # render_residual / depth_disagreement: pure reads off the
                    # admission render already computed above for the overlap
                    # gate (render_pkg["render"], render_pkg["rendered_depth"]) —
                    # no extra GPU work.
                    render_residual = 0.0
                    depth_disagreement = 0.0
                    try:
                        import torch as _torch
                        _rend_rgb = render_pkg.get("render") if hasattr(render_pkg, "get") else None
                        if _rend_rgb is not None:
                            _gt_rgb = _torch.as_tensor(
                                rgb_np, dtype=_rend_rgb.dtype, device=_rend_rgb.device
                            )
                            if _gt_rgb.max() > 1.5:
                                _gt_rgb = _gt_rgb / 255.0
                            if _gt_rgb.dim() == 3 and _gt_rgb.shape[-1] in (1, 3):
                                _gt_rgb = _gt_rgb.permute(2, 0, 1)
                            if _gt_rgb.shape == _rend_rgb.shape:
                                render_residual = float(
                                    (_rend_rgb.detach() - _gt_rgb).abs().mean().item()
                                )
                        _rend_depth = render_pkg.get("rendered_depth") if hasattr(render_pkg, "get") else None
                        if _rend_depth is not None and depth_np is not None:
                            _gt_depth = _torch.as_tensor(
                                depth_np, dtype=_rend_depth.dtype, device=_rend_depth.device
                            ).reshape(_rend_depth.shape[-2:])
                            _valid = (_gt_depth > 0.05) & (_gt_depth < cfg.depth_max)
                            if _valid.any():
                                _rd = _rend_depth.detach().reshape(_gt_depth.shape)
                                depth_disagreement = float(
                                    ((_rd[_valid] - _gt_depth[_valid]).abs() / _gt_depth[_valid].clamp_min(1e-3)).mean().item()
                                )
                    except Exception:
                        render_residual = 0.0
                        depth_disagreement = 0.0

                    scored = score_keyframe(
                        translation_m=t_m,
                        rotation_deg=r_deg,
                        gap=gap,
                        overlap=float(overlap) if overlap is not None else 0.0,
                        quality=quality,
                        new_surface_ratio=new_surface_ratio,
                        render_residual=render_residual,
                        depth_disagreement=depth_disagreement,
                        config=KeyframeScoringConfig(
                            min_translation_m=cfg.min_translation_m,
                            min_rotation_deg=cfg.min_rotation_deg,
                            min_overlap=cfg.min_overlap,
                            max_overlap=cfg.max_overlap,
                            max_gap=cfg.max_gap,
                            final_threshold=cfg.final_threshold,
                        ),
                    )
                    admitted = bool(scored.decision)
                    decision = KeyframeDecision(
                        admitted=admitted,
                        reason=scored.reason,
                        score=float(scored.final_score),
                        cheap_score=cheap_score,
                        render_overlap=overlap,
                        # #794: see the db_diverse_revisit exit above -- keep the
                        # soft-escalation trigger visible on this exit too.
                        # #932: low-overlap starvation relief must also stay
                        # visible on this exit -- `admitted` was already set
                        # True by that block above, so the quality scorer can
                        # run and admit here without the relief ever showing
                        # up as `forced`.
                        forced=bool(motion_forced or low_overlap_forced),
                        # #934: the quality scorer owns `reason` on this exit
                        # (for every mechanism, not just this one), so the
                        # novelty trigger's participation would otherwise be
                        # invisible here. Carry the flag.
                        novelty_forced=novelty_forced,
                        moved=moved,
                        novel=novel,
                        quality=float(getattr(quality, "quality_weight", 0.0)),
                        pose_delta_m=t_m,
                        pose_delta_deg=r_deg,
                        gap=gap,
                    )
                    if admitted:
                        self._record_admission(decision, frame)
                    else:
                        self._keyframes_rejected += 1
                    return decision, render_pkg
            except Exception:
                quality = None  # fall through to the pose/overlap decision

        if forced:
            reason = "max_gap"
        elif low_overlap_forced:
            reason = "low_overlap_force_admit"
        elif not overlap_ok:
            reason = "low_overlap"
        elif moved:
            reason = "pose_delta"
        elif novel:
            # NOTE (#934): this is RENDER-OVERLAP novelty (overlap <=
            # max_overlap), a different quantity from the depth/voxel coverage
            # novelty that owns "novelty_admit" below. It is deliberately
            # ranked ABOVE the novelty trigger: a candidate that would have
            # been admitted anyway by this term was not CAUSED by the trigger,
            # and letting the trigger claim it would inflate exactly the
            # ownership share #934 Stage 1 is trying to measure.
            reason = "new_content"
        elif novelty_forced:
            # #934: direct coverage evidence relaxed the motion criterion and
            # the candidate then PASSED every remaining gate. Its own reason
            # string (not "pose_delta", which would credit the motion gate;
            # not "max_gap"/"soft_max_gap_admit", which are how #795's
            # AdmissionFingerprint recognises starvation forcing).
            reason = "novelty_admit"
        elif motion_forced:
            # #794 scope=motion_only: the candidate was escalated past the
            # motion criterion and then PASSED every remaining gate. Deliberately
            # NOT "max_gap": that string is how train_streaming's #795
            # AdmissionFingerprint recognises a HARD force, and a gate-passing
            # escalation must not be counted as one.
            reason = "soft_max_gap_admit"
        else:
            reason = "redundant"

        decision = KeyframeDecision(
            admitted=admitted,
            reason=reason,
            score=cheap_score if admitted else 0.0,
            cheap_score=cheap_score,
            render_overlap=overlap,
            # #932: `low_overlap_forced` was previously computed but never
            # folded into the returned `forced` flag, so a
            # low_overlap_force_admit admission under-reported itself as
            # forced=False.
            forced=bool(forced or motion_forced or low_overlap_forced),
            # #934: NOT folded into `forced` -- see KeyframeDecision.novelty_forced.
            novelty_forced=novelty_forced,
            moved=moved,
            novel=novel,
            quality=quality,
            pose_delta_m=t_m,
            pose_delta_deg=r_deg,
            gap=gap,
        )
        if admitted:
            self._record_admission(decision, frame)
        else:
            self._keyframes_rejected += 1
        return decision, render_pkg

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _pose_delta_from_last_train(self, frame: Any) -> Tuple[float, float, int]:
        if self._last_train_c2w is None:
            # Issue #933 -- pre-baseline starvation clock.
            #
            # This used to return a flat ``(0.0, 0.0, 0)``. The pose deltas
            # genuinely have no reference before the first admitted keyframe
            # (there is no previous keyframe pose to diff against), and they
            # still return 0.0 here -- do NOT synthesise motion, that would
            # weaken the motion pregate, which is a safety net and not a bug.
            #
            # But ``gap`` is NOT pose-derived: it is a count of source frames,
            # and a source-frame count is perfectly well-defined with no
            # baseline. Pinning it to 0 made the hard starvation ceiling
            # (``gap >= max_gap * max_gap_hard_mult``, the last-resort
            # unconditional admission backstop) *structurally unreachable*
            # whenever an information gate -- coverage, parallax, blur,
            # render-overlap -- rejected the first candidate: no admission
            # -> no baseline -> gap stays 0 -> backstop cannot fire -> still
            # no admission. Measured on the ``i792novthresh_fr2desk`` run:
            # 2005 candidates evaluated, 0 admitted, every observed gap 0,
            # 9.84 dB.
            #
            # Counting from a starvation origin instead makes the existing
            # hard ceiling reachable from the pre-baseline state, so the
            # deadlock resolves through the backstop that already exists
            # rather than through a new force path (#933 explicitly forbids
            # adding a second one; #794 stays the owner of what a force may
            # bypass).
            #
            # Origin semantics (#933 asks for these to be explicit): bootstrap
            # initialisation COUNTS as forward progress -- it established the
            # map -- so the clock starts at the first candidate this admitter
            # ever evaluates, not at source-frame 0. Reset alongside
            # ``_last_train_index`` in ``reset()``.
            #
            # NOT byte-identical, but inert in the healthy case: if the first
            # candidate is admitted normally the baseline exists and this
            # branch is never taken again. It can only fire after
            # ``max_gap * max_gap_hard_mult`` candidates with zero admissions
            # -- i.e. only in the already-broken regime.
            try:
                _idx = int(frame.index)
            except (AttributeError, TypeError, ValueError):
                return 0.0, 0.0, 0
            if self._starvation_origin_index is None:
                self._starvation_origin_index = _idx
                # #946: anchor the seconds-domain starvation clock at the same
                # origin, so a pre-baseline age has a real physical reading
                # whenever the source clock is trustworthy.
                self._starvation_origin_ts = _coerce_timestamp(
                    getattr(frame, "timestamp", None)
                )
            return 0.0, 0.0, max(0, _idx - self._starvation_origin_index)
        frame_c2w = np.asarray(frame.c2w, dtype=np.float32)
        t_m = float(np.linalg.norm(frame_c2w[:3, 3] - self._last_train_c2w[:3, 3]))
        r_rel = self._last_train_c2w[:3, :3].T @ frame_c2w[:3, :3]
        cos_angle = np.clip((np.trace(r_rel) - 1.0) * 0.5, -1.0, 1.0)
        r_deg = float(np.degrees(np.arccos(cos_angle)))
        gap = int(frame.index) - int(self._last_train_index)
        return t_m, r_deg, gap

    # ------------------------------------------------------------------
    # Velocity controller (#793)
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Starvation clock (#946, consuming the #793/#933 source-time basis)
    # ------------------------------------------------------------------
    def _starvation_age(
        self, frame: Any, gap: int
    ) -> Tuple[Optional[float], str]:
        """Age since the starvation reference, in physical source seconds.

        Reference: the last ADMITTED keyframe post-baseline; the #933
        starvation origin pre-baseline. Returns ``(age_s, basis)``.

        Clock ownership (#946 "do not implement a third clock"): the trusted
        path is gated on the SAME sticky decision the velocity controller
        made in :meth:`_resolve_timebase` -- ``_timebase_source ==
        "source_timestamp"`` -- so a source whose "timestamps" are really
        frame indices, or are in microseconds, is rejected here for exactly
        the reason it was rejected there. The documented fallback is
        ``gap / input_fps``, the same nominal-period rule
        :meth:`_update_velocity_estimate` falls back to.

        The returned *basis* is deliberately NOT ``timebase_source``. That
        field describes which clock the velocity EMA used; this one describes
        which clock produced THIS number, and they legitimately differ --
        most obviously pre-baseline before an origin timestamp exists, and
        during the controller's 5-sample timebase warmup.
        """
        cfg = self._config
        if gap <= 0:
            return 0.0, "frame_index_nominal_fps"
        ref_ts = (
            self._last_train_ts
            if self._last_train_c2w is not None
            else self._starvation_origin_ts
        )
        if self._timebase_source == "source_timestamp" and ref_ts is not None:
            now_ts = _coerce_timestamp(getattr(frame, "timestamp", None))
            if now_ts is not None:
                d = float(now_ts) - float(ref_ts)
                if math.isfinite(d) and d >= 0.0:
                    return d, "source_timestamp"
        fps = float(cfg.input_fps)
        if not (fps > 0.0):
            return None, "unavailable"
        return float(gap) / fps, "frame_index_nominal_fps"

    @property
    def backstop(self):
        """The resolved #946 soft/hard thresholds (read-only)."""
        return self._backstop

    @property
    def backstop_config_warnings(self) -> List[str]:
        """Advisory warnings raised while resolving the #946 backstops."""
        return list(self._backstop_config_warnings)

    def _resolve_timebase(self, frame_delta: int, ts: Optional[float]) -> None:
        """Accumulate implied-source-rate samples and decide, once, whether
        source timestamps may be used as the controller clock.

        Trust rule (single comparison, deliberately): the implied source rate
        ``frame_delta / (ts - prev_ts)`` must land inside
        ``[input_fps / tol, input_fps * tol]``. This rejects, with one test:

        * index-valued "timestamps" (``timestamp=float(i)`` in the hypersim /
          orbbec_export / realityscan sources and the ``rgbd_sequence`` fallback):
          implied ~1 Hz against a configured 30 Hz;
        * wrong-scale timestamps (ScanNet ``.sens`` stores ``ts_color`` in raw
          microseconds): implied ~3e4 Hz;
        * non-monotonic or zero-delta junk (never produces a sample at all).

        Verified-good sources that pass and are therefore used as real clocks:
        TUM (``rgb_t`` seconds), ROS2 live / bag replay (``SyncedRGBD``
        seconds), RealSense rosbag (ns -> s), Replica (``idx/30.0``).

        Both the trusted path and the fallback are anchored on ``input_fps``
        (the fallback IS ``frame_delta / input_fps``), so a wrong ``input_fps``
        does not make the trusted path worse than the fallback -- it makes both
        wrong together, which is why this is a safe single-knob rule.
        """
        if self._timebase_decided:
            return
        cfg = self._config
        if ts is not None and self._prev_candidate_ts is not None and frame_delta > 0:
            dts = float(ts) - float(self._prev_candidate_ts)
            if math.isfinite(dts) and dts > 0.0:
                self._timebase_fps_samples.append(float(frame_delta) / dts)
        if len(self._timebase_fps_samples) >= _TIMEBASE_WARMUP_SAMPLES:
            implied = float(np.median(np.asarray(self._timebase_fps_samples, dtype=np.float64)))
            self._implied_input_fps = implied
            tol = max(float(cfg.velocity_timebase_fps_tolerance), 1.0)
            nominal = max(float(cfg.input_fps), 1e-6)
            if (nominal / tol) <= implied <= (nominal * tol):
                self._timebase_source = "source_timestamp"
            else:
                self._timebase_source = "frame_index_nominal_fps"
            self._timebase_decided = True

    def _update_velocity_estimate(self, frame: Any) -> dict:
        """Sample instantaneous velocity against the PREVIOUS CANDIDATE frame,
        update the EMAs, and advance the previous-candidate reference.

        This is the #793 correctness fix. Three properties matter and are
        asserted by ``tests/test_keyframe_velocity_controller_units.py``:

        1. the sample is a physical velocity in m/s (and deg/s), not a
           displacement, so it does not grow with the rejection-streak length;
        2. it is taken against the previous CANDIDATE, so it is identical
           whether or not frames are being admitted;
        3. the previous-candidate reference advances UNCONDITIONALLY here --
           not in ``_record_admission`` (admit-only) and not inside any
           decision branch -- which is what makes (2) true.

        The EMA is gated on ``dt > 0``, NOT on displacement ``> 0``: a
        stationary camera genuinely has zero velocity and that sample must
        enter the estimator, otherwise the threshold holds a stale high value
        across a pause and over-thresholds on resume.

        Returns a telemetry dict; never raises.
        """
        cfg = self._config
        out = {
            "translation_velocity_m_s": None,
            "rotation_velocity_deg_s": None,
            "source_dt_s": None,
            "source_frame_delta": None,
            "timebase_source": self._timebase_source,
            "implied_input_fps": self._implied_input_fps,
        }
        c2w = np.asarray(frame.c2w, dtype=np.float32)
        idx = int(frame.index)
        ts = getattr(frame, "timestamp", None)
        try:
            ts = float(ts) if ts is not None and math.isfinite(float(ts)) else None
        except (TypeError, ValueError):
            ts = None

        if self._prev_candidate_c2w is None or self._prev_candidate_index is None:
            # First candidate: seed the reference, no velocity sample yet.
            self._prev_candidate_c2w = c2w.copy()
            self._prev_candidate_index = idx
            self._prev_candidate_ts = ts
            return out

        frame_delta = idx - int(self._prev_candidate_index)
        self._resolve_timebase(frame_delta, ts)
        out["timebase_source"] = self._timebase_source
        out["implied_input_fps"] = self._implied_input_fps
        out["source_frame_delta"] = frame_delta

        dt_s: Optional[float] = None
        if self._timebase_source == "source_timestamp" and ts is not None and self._prev_candidate_ts is not None:
            _d = float(ts) - float(self._prev_candidate_ts)
            if math.isfinite(_d) and _d > 0.0:
                dt_s = _d
        if dt_s is None and frame_delta > 0:
            # Documented fallback: nominal period x source-frame delta. This is
            # also automatically correct under source striding (StridedFrameSource
            # yields the underlying frames unchanged, so a stride-3 view has
            # frame_delta 3 and the same m/s comes out) -- do NOT divide
            # input_fps by the stride.
            dt_s = float(frame_delta) / max(float(cfg.input_fps), 1e-6)

        if dt_s is not None and math.isfinite(dt_s) and dt_s > 0.0:
            t_step = float(np.linalg.norm(c2w[:3, 3] - self._prev_candidate_c2w[:3, 3]))
            r_rel = self._prev_candidate_c2w[:3, :3].T @ c2w[:3, :3]
            cos_a = np.clip((np.trace(r_rel) - 1.0) * 0.5, -1.0, 1.0)
            r_step = float(np.degrees(np.arccos(cos_a)))
            v_t = t_step / dt_s
            v_r = r_step / dt_s
            alpha = float(np.clip(cfg.velocity_ema_alpha, 0.0, 1.0))
            self._translation_velocity_ema_m_s = (
                v_t if self._translation_velocity_ema_m_s is None
                else alpha * v_t + (1.0 - alpha) * self._translation_velocity_ema_m_s
            )
            self._rotation_velocity_ema_deg_s = (
                v_r if self._rotation_velocity_ema_deg_s is None
                else alpha * v_r + (1.0 - alpha) * self._rotation_velocity_ema_deg_s
            )
            self._source_elapsed_s += dt_s
            out["translation_velocity_m_s"] = v_t
            out["rotation_velocity_deg_s"] = v_r
            out["source_dt_s"] = dt_s

        # Advance the reference UNCONDITIONALLY (property 3 above).
        self._prev_candidate_c2w = c2w.copy()
        self._prev_candidate_index = idx
        self._prev_candidate_ts = ts
        return out

    @property
    def controller_telemetry(self) -> dict:
        """Realized velocity-controller state, for the #795 admission
        fingerprint. Recording-only; reading it has no side effects."""
        cfg = self._config
        return {
            "velocity_adaptive": bool(cfg.velocity_adaptive),
            "translation_velocity_ema_m_s": self._translation_velocity_ema_m_s,
            "rotation_velocity_ema_deg_s": self._rotation_velocity_ema_deg_s,
            "timebase_source": self._timebase_source,
            "implied_input_fps": self._implied_input_fps,
            "configured_input_fps": float(cfg.input_fps),
            "controller_target_interval_s": (
                1.0 / max(float(cfg.target_admit_rate_fps), 0.01)
                if cfg.velocity_adaptive else None
            ),
            "threshold_evals": self._threshold_evals,
            "threshold_floor_hits": self._threshold_floor_hits,
            "threshold_ceil_hits": self._threshold_ceil_hits,
            "source_elapsed_s": self._source_elapsed_s,
        }

    def _load_frame_rgb(self, frame: Any) -> Optional[np.ndarray]:
        """Best-effort RGB fetch for the cheap input-quality gates.

        Prefers the injected ``load_rgb`` callback; falls back to the streaming
        frame loader. Returns ``None`` on any failure so the gate is skipped
        (fail-open) rather than crashing admission.
        """
        if self._load_rgb is not None:
            try:
                return np.asarray(self._load_rgb(frame))
            except Exception:
                return None
        try:
            from eligsir.io.sources.decode_cache import load_frame_rgb as _load_frame_rgb
            return np.asarray(_load_frame_rgb(frame))
        except Exception:
            return None

    def _record_admission(self, decision: KeyframeDecision, frame: Any) -> None:
        idx = int(frame.index)
        decision.keyframe_index = idx
        self._keyframe_indices.append(idx)
        if len(self._keyframe_indices) > self._max_history:
            # Keep the most recent half so the list never grows without
            # bound in long-running sessions.
            self._keyframe_indices = self._keyframe_indices[self._max_history // 2 :]
        self._keyframes_admitted += 1
        # Update exposure-gate luminance reference on admission (if computed this call).
        _pending = getattr(self, "_pending_luma", None)
        if _pending is not None:
            self._last_admitted_luma = _pending
            self._last_admitted_iter = self._exposure_iter
        self._pending_luma = None
        # Reset blur-gate starvation counter on admission.
        self._last_admitted_iter_blur = self._blur_iter
        # Self-maintaining reference: the admitter updates its own
        # "last train frame" pointer on every admission so subsequent
        # cheap-gate calls see the right baseline. Callers therefore do
        # *not* need to call ``note_train_frame`` (it's still exposed
        # for bootstrap / reset paths but is otherwise an internal
        # implementation detail).
        # NOTE (#793): this deliberately does NOT touch the velocity
        # controller's previous-CANDIDATE reference. That reference must
        # advance on every candidate (see _update_velocity_estimate), not only
        # on admission -- advancing it here is exactly the admit/reject
        # dependence the #793 fix removes.
        self.note_train_frame(
            FrameInfo(
                c2w=frame.c2w,
                index=frame.index,
                timestamp=getattr(frame, "timestamp", None),
            )
        )
