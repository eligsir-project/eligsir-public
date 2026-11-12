"""Detection of the max_gap < frames_between_admits mismatch (issue #792).

This module provides a pure function to detect when hybrid_keyframe admission
with velocity-adaptive motion is configured such that max_gap preempts the
adaptive motion gate—specifically when the forced-admit cadence (every
max_gap frames) is tighter than the nominal admission target
(input_fps / target_admit_fps).

The detection is behavior-neutral: it only reports the mismatch. It does NOT
change admission decisions or any default.

Scope note: the sole production consumer (``train_streaming.py``) currently
just PRINTS ``warning_text`` at startup -- the result is not persisted to
report.json. An earlier version of this docstring claimed it was; that was
never true. This matters because a printed warning is only recoverable by
grepping train.log, whereas this repo's Run Contracts treat a machine-readable
report.json field as the real artifact (cf. ``map_reinit_invalidated``,
``seed_fade_indeterminate``, ``ingestion.mode``/``reason``, each added for
exactly that reason). Persisting this block is a tracked follow-up on #792; it
is deliberately out of scope for the detection-only slice.

Reference: issue #792 (the umbrella), #793/#794 (the actual semantics fixes).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class AdmissionPreflightResult:
    """Structured result of the preflight check.

    Attributes:
        frames_between_admits: Computed nominal inter-admission frame spacing
            (input_fps / target_admit_fps).
        max_gap: The configured max_gap value.
        mismatch: True iff conditions for the mismatch ALL hold:
            - admission mode is hybrid_keyframe,
            - velocity_adaptive is enabled,
            - max_gap < frames_between_admits.
        verdict: A descriptive string summarizing the state:
            "OK", "MAX_GAP_PREEMPTS_ADAPTIVE_MOTION", "NOT_APPLICABLE_MODE",
            "NOT_APPLICABLE_ADAPTIVE_OFF", "NOT_APPLICABLE_NOVELTY_GATED",
            or "DIVIDE_BY_ZERO" (target_admit_fps too close to zero).
        max_gap_require_novelty: The value of the novelty-gating flag; if True,
            the consequence is softened (forced admits are novelty-gated).
        coverage_gain_gate: The value of --streaming_keyframe_coverage_gain_gate,
            when the caller supplies it (``None`` if not). Issue #928: without
            this, ``max_gap_require_novelty=True`` alone cannot actually
            evaluate the novelty gate (``keyframes.py``'s
            ``max_gap_require_novelty`` docstring / the "gate_disabled" reason
            in ``PrecedenceTrace.novelty_gate_unavailable_reason``) --
            ``coverage_gain_gate`` distinguishes "definitely inert" (``False``)
            from "requested, but whether it actually evaluates for a given
            candidate additionally depends on occupancy_fn/load_depth_m
            wiring this preflight check cannot see" (``True``).
        warning_text: Human-readable warning message, or None if verdict is OK.
    """
    frames_between_admits: float
    max_gap: int
    mismatch: bool
    verdict: str
    max_gap_require_novelty: bool
    coverage_gain_gate: Optional[bool] = None
    warning_text: Optional[str] = None


def check_admission_preflight(
    admission_mode: str,
    velocity_adaptive: bool,
    input_fps: float,
    target_admit_fps: float,
    max_gap: int,
    max_gap_require_novelty: bool = False,
    coverage_gain_gate: Optional[bool] = None,
) -> AdmissionPreflightResult:
    """Detect max_gap < frames_between_admits mismatch.

    Args:
        admission_mode: Value of --streaming_frame_admission (e.g., "all",
            "hybrid_keyframe").
        velocity_adaptive: Value of --streaming_keyframe_velocity_adaptive.
        input_fps: Value of --streaming_keyframe_input_fps.
        target_admit_fps: Value of --streaming_keyframe_target_admit_fps.
        max_gap: Value of --streaming_keyframe_max_gap.
        max_gap_require_novelty: Value of --streaming_keyframe_max_gap_require_novelty.
        coverage_gain_gate: Value of --streaming_keyframe_coverage_gain_gate,
            OPTIONAL and defaulting to ``None`` ("caller didn't say"). Issue
            #928: passing this lets the warning distinguish a statically
            KNOWN-inert novelty gate (``coverage_gain_gate=False`` --
            max_gap_require_novelty has no effect at all, see the
            "gate_disabled" reason) from a merely REQUESTED one
            (``coverage_gain_gate=True`` -- whether it actually evaluates for
            a given candidate also needs occupancy_fn/load_depth_m wiring
            this preflight check, which runs before ``KeyframeAdmitter``
            construction, cannot see). Leaving it ``None`` reproduces the
            exact pre-#928 wording byte-for-byte, so existing callers that
            don't pass it see no behavior change.

    Returns:
        AdmissionPreflightResult with computed values and verdict.

    The mismatch condition holds when ALL of these are true:
      - admission_mode == "hybrid_keyframe"
      - velocity_adaptive is True
      - max_gap < input_fps / max(target_admit_fps, 0.01)

    If the mismatch holds and max_gap_require_novelty is False, or
    ``coverage_gain_gate`` is explicitly known to be False, the warning
    describes the consequence as unmitigated: the adaptive motion criterion
    is preempted by the forced max_gap stride and no novelty check can run.
    Otherwise, with max_gap_require_novelty True, the warning is downgraded --
    but (issue #928) only ever to a REQUEST claim, not a confirmed mitigation,
    unless the caller can positively rule out coverage_gain_gate.
    """
    # Compute frames_between_admits (same formula as keyframes.py:591).
    # Clamp target_admit_fps to avoid division by zero.
    clamped_target = max(target_admit_fps, 0.01)
    frames_between_admits = input_fps / clamped_target

    # Check each mismatch condition.
    mode_is_hybrid = admission_mode == "hybrid_keyframe"
    adaptive_is_on = velocity_adaptive

    # The mismatch boolean: all three conditions must hold.
    mismatch = mode_is_hybrid and adaptive_is_on and (max_gap < frames_between_admits)

    # Determine verdict and warning text.
    if not mode_is_hybrid:
        verdict = "NOT_APPLICABLE_MODE"
        return AdmissionPreflightResult(
            frames_between_admits=frames_between_admits,
            max_gap=max_gap,
            mismatch=False,
            verdict=verdict,
            max_gap_require_novelty=max_gap_require_novelty,
            coverage_gain_gate=coverage_gain_gate,
            warning_text=None,
        )

    if not adaptive_is_on:
        verdict = "NOT_APPLICABLE_ADAPTIVE_OFF"
        return AdmissionPreflightResult(
            frames_between_admits=frames_between_admits,
            max_gap=max_gap,
            mismatch=False,
            verdict=verdict,
            max_gap_require_novelty=max_gap_require_novelty,
            coverage_gain_gate=coverage_gain_gate,
            warning_text=None,
        )

    # Handle the case where target_admit_fps was clamped to 0.01.
    if target_admit_fps < 0.01 and target_admit_fps > 0:
        verdict = "DIVIDE_BY_ZERO"
        warning_text = (
            f"[admission] WARNING: target_admit_fps={target_admit_fps} is too close to zero; "
            f"clamped to 0.01 for frames_between_admits calculation. This is a configuration error."
        )
        return AdmissionPreflightResult(
            frames_between_admits=frames_between_admits,
            max_gap=max_gap,
            mismatch=False,
            verdict=verdict,
            max_gap_require_novelty=max_gap_require_novelty,
            coverage_gain_gate=coverage_gain_gate,
            warning_text=warning_text,
        )

    # We are in hybrid_keyframe + velocity_adaptive. Check for the gap mismatch.
    if mismatch:
        if max_gap_require_novelty and coverage_gain_gate is False:
            # Issue #928: statically knowable at preflight time -- the
            # novelty gate has NO effect without coverage_gain_gate (see
            # keyframes.py's max_gap_require_novelty docstring / the
            # "gate_disabled" reason), so this is exactly as unmitigated as
            # the plain max_gap-preempts-adaptive-motion case. This is a new
            # branch (no prior caller passes coverage_gain_gate=False), so it
            # cannot regress any existing behavior.
            verdict = "MAX_GAP_PREEMPTS_ADAPTIVE_MOTION"
            warning_text = (
                f"[admission] WARNING: hybrid_keyframe with velocity_adaptive will "
                f"degenerate to a fixed ~{max_gap}-frame stride. Computed "
                f"frames_between_admits={frames_between_admits:.1f} (input_fps={input_fps} / "
                f"target_admit_fps={target_admit_fps}), but max_gap={max_gap} forces "
                f"unconditional admits more frequently, preempting the motion criterion. "
                f"--streaming_keyframe_max_gap_require_novelty is True but "
                f"--streaming_keyframe_coverage_gain_gate is False, so the novelty gate "
                f"CANNOT evaluate (issue #928 'gate_disabled' reason) -- it does NOT "
                f"mitigate this mismatch. See issue #792/#928."
            )
        elif max_gap_require_novelty:
            verdict = "NOT_APPLICABLE_NOVELTY_GATED"
            if coverage_gain_gate is True:
                # Issue #928: honest, conditional wording -- this preflight
                # check runs before KeyframeAdmitter is constructed, so it
                # cannot see whether occupancy_fn/load_depth_m end up wired,
                # which the novelty gate ALSO needs to actually evaluate (see
                # PrecedenceTrace.novelty_gate_unavailable_reason's
                # "no_occupancy_fn"/"no_depth_loader"). Claiming "mitigated"
                # here would be exactly the unsupportable claim issue #928
                # was filed against.
                warning_text = (
                    f"[admission] INFO: max_gap < frames_between_admits "
                    f"({max_gap} < {frames_between_admits:.1f}). "
                    f"--streaming_keyframe_max_gap_require_novelty and "
                    f"--streaming_keyframe_coverage_gain_gate are both True, so forced "
                    f"admits are REQUESTED to be novelty-gated -- but whether the gate "
                    f"actually evaluates for a given candidate additionally depends on "
                    f"occupancy_fn/load_depth_m wiring at KeyframeAdmitter construction, "
                    f"which happens after this preflight check and is not visible to it. "
                    f"This is a REQUEST, not a confirmed mitigation. After the run, check "
                    f"report.json's admission_fingerprint.novelty_gate (or "
                    f"keyframe_admission_log.jsonl's per-row novelty_gate_outcome) for the "
                    f"REALIZED evaluated-vs-not-evaluated split (issue #928)."
                )
            else:
                # coverage_gain_gate not supplied (None) -- an older call
                # site that predates issue #928. Preserve the exact pre-#928
                # wording byte-for-byte so no existing caller/test observes a
                # behavior change from adding this parameter.
                warning_text = (
                    f"[admission] INFO: max_gap < frames_between_admits "
                    f"({max_gap} < {frames_between_admits:.1f}), but "
                    f"--streaming_keyframe_max_gap_require_novelty is True, so forced admits "
                    f"are gated by novelty. The mismatch is mitigated."
                )
        else:
            verdict = "MAX_GAP_PREEMPTS_ADAPTIVE_MOTION"
            warning_text = (
                f"[admission] WARNING: hybrid_keyframe with velocity_adaptive will "
                f"degenerate to a fixed ~{max_gap}-frame stride. Computed "
                f"frames_between_admits={frames_between_admits:.1f} (input_fps={input_fps} / "
                f"target_admit_fps={target_admit_fps}), but max_gap={max_gap} forces "
                f"unconditional admits more frequently, preempting the motion criterion. "
                f"See issue #792."
            )
    else:
        verdict = "OK"
        warning_text = None

    return AdmissionPreflightResult(
        frames_between_admits=frames_between_admits,
        max_gap=max_gap,
        mismatch=mismatch,
        verdict=verdict,
        max_gap_require_novelty=max_gap_require_novelty,
        coverage_gain_gate=coverage_gain_gate,
        warning_text=warning_text,
    )
