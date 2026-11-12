"""Modular keyframe scoring for streaming admission."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Optional

from .frame_quality import FrameQuality


@dataclass(frozen=True)
class KeyframeScore:
    """Result of keyframe scoring and admission decision.

    Carries individual component scores (normalized [0,1]) for different
    criteria, a combined final score, and the admission decision with reason.
    """
    pose_delta_score: float
    rotation_delta_score: float
    overlap_score: float
    new_surface_score: float
    render_residual_score: float
    depth_disagreement_score: float
    quality_score: float
    uncertainty_score: float
    final_score: float
    decision: bool
    reason: str
    terms: dict[str, float] = field(default_factory=dict)

    @property
    def admitted(self) -> bool:
        """Whether this frame was admitted as a keyframe."""
        return self.decision


@dataclass(frozen=True)
class KeyframeScoringConfig:
    """Configuration parameters for keyframe admission scoring.

    Controls thresholds and weights for pose-delta-based and content-based
    admission decisions, as well as quality-based rejection.
    """
    min_translation_m: float = 0.05
    min_rotation_deg: float = 5.0
    min_overlap: float = 0.25
    max_overlap: float = 0.90
    max_gap: int = 10
    final_threshold: float = 0.45
    reject_bad_quality: bool = True


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def score_keyframe(
    *,
    translation_m: float,
    rotation_deg: float,
    gap: int,
    overlap: float,
    quality: Optional[FrameQuality] = None,
    render_residual: float = 0.0,
    depth_disagreement: float = 0.0,
    uncertainty: float = 0.0,
    new_surface_ratio: float = 0.0,
    config: Optional[KeyframeScoringConfig] = None,
) -> KeyframeScore:
    """Score and decide keyframe admission from normalized modular terms."""

    cfg = config or KeyframeScoringConfig()
    quality_score = quality.quality_weight if quality is not None else 1.0
    if cfg.reject_bad_quality and quality is not None and not quality.accepted:
        return KeyframeScore(
            pose_delta_score=0.0,
            rotation_delta_score=0.0,
            overlap_score=0.0,
            new_surface_score=0.0,
            render_residual_score=0.0,
            depth_disagreement_score=0.0,
            quality_score=quality_score,
            uncertainty_score=0.0,
            final_score=0.0,
            decision=False,
            reason=quality.reject_reason or "quality_reject",
            terms={"quality": quality_score},
        )

    forced = int(gap) >= max(1, int(cfg.max_gap))
    pose_score = _clamp01(translation_m / max(cfg.min_translation_m, 1e-6))
    rot_score = _clamp01(rotation_deg / max(cfg.min_rotation_deg, 1e-6))
    if overlap < cfg.min_overlap:
        overlap_score = 0.0
    else:
        overlap_score = _clamp01((cfg.max_overlap - overlap) / max(cfg.max_overlap - cfg.min_overlap, 1e-6))
    new_surface_score = _clamp01(new_surface_ratio)
    render_residual_score = _clamp01(render_residual)
    depth_disagreement_score = _clamp01(depth_disagreement)
    uncertainty_score = _clamp01(uncertainty)

    terms = {
        "pose_delta": pose_score,
        "rotation_delta": rot_score,
        "overlap": overlap_score,
        "new_surface": new_surface_score,
        "render_residual": render_residual_score,
        "depth_disagreement": depth_disagreement_score,
        "quality": quality_score,
        "uncertainty": uncertainty_score,
    }
    final = quality_score * (
        0.24 * pose_score
        + 0.16 * rot_score
        + 0.20 * overlap_score
        + 0.14 * new_surface_score
        + 0.10 * render_residual_score
        + 0.10 * depth_disagreement_score
        + 0.06 * uncertainty_score
    )

    moved = translation_m >= cfg.min_translation_m or rotation_deg >= cfg.min_rotation_deg
    novel = cfg.min_overlap <= overlap <= cfg.max_overlap
    decision = forced or final >= cfg.final_threshold or (novel and moved)
    if forced:
        reason = "max_gap"
    elif decision and moved:
        reason = "pose_delta"
    elif decision and novel:
        reason = "new_content"
    elif decision:
        reason = "score"
    elif overlap < cfg.min_overlap:
        reason = "low_overlap"
    else:
        reason = "redundant"

    return KeyframeScore(
        pose_delta_score=pose_score,
        rotation_delta_score=rot_score,
        overlap_score=overlap_score,
        new_surface_score=new_surface_score,
        render_residual_score=render_residual_score,
        depth_disagreement_score=depth_disagreement_score,
        quality_score=quality_score,
        uncertainty_score=uncertainty_score,
        final_score=float(final),
        decision=bool(decision),
        reason=reason,
        terms=terms,
    )


def score_to_stats(score: KeyframeScore, extra: Optional[Mapping[str, float]] = None) -> dict:
    """Convert a KeyframeScore to a telemetry statistics dictionary.

    Args:
        score: The scoring result to convert.
        extra: Optional additional metrics to include in the output.

    Returns:
        A dictionary containing the admission decision, reason, final score,
        individual term scores, and any extra metrics.
    """
    stats = {
        "admitted": int(score.admitted),
        "reason": score.reason,
        "final_score": score.final_score,
    }
    stats.update(score.terms)
    if extra:
        stats.update(dict(extra))
    return stats
