"""Cheap eligibility-only prefilter for the prepared-candidate pipeline.

The prefilter may reject invalid/unreliable observations or escalate a frame
for downstream consideration. It never commits a keyframe and never owns a
hard admission budget; those decisions belong to the cohort/scheduler layer.

See `docs/architecture/2026-09-06-issue821-legacy-hybrid-keyframe-v1-reference-mapping.md`
for the full legacy hybrid-keyframe decision mapping and known gaps (issue #821 criterion 9).
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np

from eligsir.mapping.candidates.contracts import PreparedFrameCandidate, TrackedFramePacket, TrackingState
from eligsir.mapping.candidates.preparation import coarse_pose_novelty
from eligsir.mapping.evidence.adapters import from_candidate_input_evidence
from eligsir.mapping.evidence.canonical import (
    EvidenceIdentity,
    FrameEvidence,
    ScalarEvidence,
    SourceReliabilityEvidence,
)


class PrefilterDisposition(str, Enum):
    ELIGIBLE = "eligible"
    INELIGIBLE = "ineligible"
    ESCALATE = "escalate"


@dataclass(frozen=True)
class PrefilterConfig:
    min_pose_novelty: float = 0.02
    novelty_translation_norm_m: float = 1.0
    novelty_rotation_norm_deg: float = 30.0
    min_quality_weight: float = 0.0
    reject_on_tracking_lost: bool = True
    soft_starvation_gap: int = 0


@dataclass(frozen=True)
class CandidatePrefilterResult:
    disposition: PrefilterDisposition
    evidence: FrameEvidence
    reason: Optional[str] = None

    @property
    def eligible(self) -> bool:
        return self.disposition is not PrefilterDisposition.INELIGIBLE


@dataclass(frozen=True)
class PrefilterDecision:
    """Compatibility shape for older callers; not a commit decision."""
    accept: bool
    reason: Optional[str] = None


def should_skip_decode(
    tracked: TrackedFramePacket,
    previous_c2w: Optional[np.ndarray],
    config: Optional[PrefilterConfig] = None,
) -> PrefilterDecision:
    """Conservative pre-decode eligibility check; no durable admission."""
    cfg = config or PrefilterConfig()
    if cfg.reject_on_tracking_lost and tracked.tracking_state in (TrackingState.LOST, TrackingState.INITIALIZING):
        return PrefilterDecision(False, f"tracking_{tracked.tracking_state.value}")
    novelty = coarse_pose_novelty(
        previous_c2w,
        tracked.world_T_camera,
        translation_norm_m=cfg.novelty_translation_norm_m,
        rotation_norm_deg=cfg.novelty_rotation_norm_deg,
    )
    if novelty < cfg.min_pose_novelty:
        return PrefilterDecision(False, "low_pose_novelty")
    return PrefilterDecision(True)


def evaluate(
    candidate: PreparedFrameCandidate,
    config: Optional[PrefilterConfig] = None,
    *,
    candidate_gap: int = 0,
) -> CandidatePrefilterResult:
    """Produce source-quality evidence and ELIGIBLE/INELIGIBLE/ESCALATE.

    A starvation/max-gap condition is an escalation, never an implicit proof
    that the frame is scientifically useful.  Final COMMIT/DEFER/DROP remains
    downstream under a hard budget.
    """
    cfg = config or PrefilterConfig()
    inp = candidate.input_evidence

    # Integrity ordering: hard reject gates all downstream signals
    hard: list[str] = []
    soft: list[str] = []
    escalations: list[str] = []

    if inp.reject_reason is not None:
        hard.append(str(inp.reject_reason))

    # Only evaluate cheap signals if no hard reject (integrity ordering contract)
    if not hard:
        if candidate.coarse_novelty < cfg.min_pose_novelty:
            soft.append("low_pose_novelty")
        if inp.quality_weight < cfg.min_quality_weight:
            soft.append("low_quality_weight")

    if cfg.soft_starvation_gap > 0 and int(candidate_gap) >= int(cfg.soft_starvation_gap):
        escalations.append("soft_starvation_escalation")

    # Convert legacy InputEvidence to canonical FrameEvidence via adapter (issue #822).
    # The adapter handles field mapping and availability tracking.
    evidence = from_candidate_input_evidence(
        inp,
        source_index=candidate.source_index,
        timestamp_ns=candidate.timestamp_ns,
        source_id="prepared-candidate",
    )

    # Override evidence with prefilter-specific rejections and escalations.
    # rgb_eligible: false if hard rejects or soft rejects
    # depth_eligible: false if hard rejects, soft rejects, or no valid depth
    evidence = FrameEvidence(
        identity=evidence.identity,
        source=evidence.source,
        probe=evidence.probe,
        rgb_eligible=not bool(hard) and not bool(soft),
        depth_eligible=(not bool(hard) and not bool(soft) and evidence.depth_eligible),
        hard_reject_reasons=tuple(hard),
        soft_reject_reasons=tuple(soft),
        escalations=tuple(escalations),
    )

    if hard:
        return CandidatePrefilterResult(PrefilterDisposition.INELIGIBLE, evidence, hard[0])
    if escalations:
        return CandidatePrefilterResult(PrefilterDisposition.ESCALATE, evidence, escalations[0])
    if soft:
        return CandidatePrefilterResult(PrefilterDisposition.INELIGIBLE, evidence, soft[0])
    return CandidatePrefilterResult(PrefilterDisposition.ELIGIBLE, evidence)


def decide(candidate: PreparedFrameCandidate, config: Optional[PrefilterConfig] = None) -> PrefilterDecision:
    """Compatibility shim over :func:`evaluate`; ``accept`` means eligible."""
    result = evaluate(candidate, config)
    return PrefilterDecision(accept=result.eligible, reason=result.reason)


def legacy_hybrid_keyframe_v1() -> PrefilterConfig:
    """Issue #821 criterion 9: historical hybrid-keyframe reproducibility preset.

    Returns a PrefilterConfig that exactly reproduces the legacy hybrid-keyframe motion
    gate through the canonical prefilter dispositions (ELIGIBLE/INELIGIBLE/ESCALATE).

    The preset uses historical defaults extracted from KeyframeAdmitterConfig
    (keyframes.py line 241, 242, 245, 387, 458; issue #793, #794, #946):

    - min_translation_m: 0.05 (KeyframeAdmitterConfig line 241)
    - min_rotation_deg: 5.0 (KeyframeAdmitterConfig line 242)
    - max_gap: 10 (KeyframeAdmitterConfig line 245)
    - max_gap_hard_mult: 4.0 (KeyframeAdmitterConfig line 387)
    - soft_backstop_mode: LEGACY_FRAMES (KeyframeAdmitterConfig line 458)

    ## Exact motion gate equivalence

    The legacy motion pregate (keyframes.py line 1450) uses OR logic:
        `moved = (d_t >= min_translation_m) OR (d_r >= min_rotation_deg)`
        `moved = (d_t >= 0.05m) OR (d_r >= 5.0°)`

    The prefilter's coarse_pose_novelty() computes (preparation.py line 124):
        `novelty = max(d_t / translation_norm_m, d_r / rotation_norm_deg)`

    By setting the norms equal to the legacy thresholds, the scalar MAX gate
    becomes mathematically equivalent to the legacy OR gate:

        novelty = max(d_t / 0.05, d_r / 5.0) >= 1.0
        ⟺ (d_t / 0.05 >= 1.0) OR (d_r / 5.0 >= 1.0)
        ⟺ (d_t >= 0.05) OR (d_r >= 5.0)

    This is verified by test_legacy_decision_motion_gate_exact_reproduction(),
    which tests all 16 combinations of d_t ∈ {0, 0.04, 0.05, 0.2}m and
    d_r ∈ {0, 3, 5, 20}° — including the boundary case (0.04m, 3°) that both
    legacy and prefilter correctly REJECT.

    ## Configuration

    - `min_pose_novelty`: 1.0 (scalar threshold for max(d_t/norm_t, d_r/norm_r))
    - `novelty_translation_norm_m`: 0.05 (legacy min_translation_m, from keyframes.py line 241)
    - `novelty_rotation_norm_deg`: 5.0 (legacy min_rotation_deg, from keyframes.py line 242)
    - `min_quality_weight`: 0.0 (legacy default)
    - `reject_on_tracking_lost`: True (legacy pregate)
    - `soft_starvation_gap`: 10 (legacy max_gap, issue #946 / keyframes.py line 245)

    ## Known gaps (non-motion gates; documented in architecture doc)

    - Coverage/parallax gates unreproducible (GPU-side occupancy lookups)
    - Velocity-adaptive threshold unreproducible (no per-candidate state)
    - Render-overlap novelty unreproducible (GPU-side render overlap)

    See: eligsir/mapping/scheduler/keyframes.py::KeyframeAdmitterConfig
    """
    return PrefilterConfig(
        min_pose_novelty=1.0,  # scalar gate: max(d_t/norm_t, d_r/norm_r) >= 1.0
        novelty_translation_norm_m=0.05,  # min_translation_m from keyframes.py line 241
        novelty_rotation_norm_deg=5.0,  # min_rotation_deg from keyframes.py line 242
        min_quality_weight=0.0,
        reject_on_tracking_lost=True,
        soft_starvation_gap=10,  # max_gap from keyframes.py line 245
    )
