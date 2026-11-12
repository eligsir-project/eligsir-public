"""Active, evidence-driven admission rollout policy (Wave 3, PR W3-07,
plan section 12, Subagents B + C).

``AdmissionRolloutPolicy`` is the single object that turns a
:class:`~eligsir.mapping.admission.utility_contracts.FrameUtilityScore`
(W3-06's shadow scorer -- consumed here, NOT reimplemented) into a final,
typed :class:`~eligsir.mapping.admission.decision_contracts.AdmissionDecision`
under one of the four graduated rollout modes
(:class:`~eligsir.mapping.admission.decision_contracts.RolloutMode`).

**The single most important property of this module**: hard safety gates
(tracking validity, map-ID consistency, timestamp integrity, source
validity) are checked and applied by the CALLER before
:meth:`AdmissionRolloutPolicy.decide` is ever invoked (see
``train_streaming.py``'s ``_keyframe_admission_closure``, which returns on
``_reinit_tracker.vetoing`` before constructing a camera or calling this
policy at all). This module additionally enforces the same invariant
DEFENSIVELY on its own: if a caller passes ``hard_gate_blocked=True``,
:meth:`decide` returns a rejected decision unconditionally, in every
rollout mode including ``ACTIVE`` -- see ``test_active_admission.py``'s
``test_hard_gate_blocks_every_rollout_mode`` for the adversarial test this
docstring promises (a hard gate can NEVER be talked out of rejecting a
frame by a high utility score, in any mode).

**Deliverable 1** ("current policy remains available as fallback"): the
baseline admission decision (``KeyframeAdmitter.admit()``'s result) is
always passed in as ``baseline_admitted`` and is what every mode falls
back to when there's no usable utility score, and is the ONLY thing
``SHADOW`` mode ever returns.
"""

from __future__ import annotations

from typing import Optional

from eligsir.mapping.admission.budget_controller import MappingBudgetController
from eligsir.mapping.admission.decision_contracts import (
    REASON_ACTIVE_ADMIT,
    REASON_ACTIVE_REJECT,
    REASON_ADVISORY_REJECT,
    REASON_BASELINE_PASSTHROUGH,
    REASON_BUDGET_EXHAUSTED,
    REASON_HARD_SAFETY_VETO,
    REASON_NO_UTILITY_SCORE,
    AdmissionDecision,
    MappingBudgetRequest,
    RolloutMode,
)
from eligsir.mapping.admission.utility_contracts import FrameUtilityScore
from eligsir.mapping.evidence.contracts import EvidenceMeta


class AdmissionRolloutPolicy:
    """Applies a :class:`RolloutMode` on top of a baseline admission
    decision and a (possibly absent) utility score.

    Construct one per training run, mode fixed at construction time
    (mirrors how ``ShadowUtilityPolicy`` fixes ``enabled`` at
    construction) -- rollout mode is a deliberate, reviewed run
    configuration, not something that should silently vary mid-run.
    """

    def __init__(
        self,
        mode: RolloutMode = RolloutMode.SHADOW,
        utility_admit_threshold: float = 0.0,
        budget_controller: Optional[MappingBudgetController] = None,
    ) -> None:
        self.mode = RolloutMode(mode)
        self.utility_admit_threshold = float(utility_admit_threshold)
        self.budget_controller = budget_controller
        if self.mode == RolloutMode.ACTIVE_WITH_BUDGET and budget_controller is None:
            # A budget-controlled mode with no controller would silently
            # behave like plain ACTIVE -- fail loudly instead of shipping
            # an unbudgeted "budgeted" run.
            raise ValueError("RolloutMode.ACTIVE_WITH_BUDGET requires a budget_controller")

    # ------------------------------------------------------------------
    def decide(
        self,
        *,
        frame_id: int,
        baseline_admitted: bool,
        hard_gate_blocked: bool,
        utility_score: Optional[FrameUtilityScore] = None,
        predicted_cost: float = 0.0,
        requested_burst_steps: int = 0,
        replay_weight: float = 0.0,
        meta: Optional[EvidenceMeta] = None,
    ) -> AdmissionDecision:
        """Compute the final admission decision.

        Args:
            frame_id: Source-frame index.
            baseline_admitted: What ``KeyframeAdmitter.admit()`` (or the
                admission=all path) already decided.
            hard_gate_blocked: True iff a hard safety gate has already
                rejected this frame. When True, this call is a no-op
                pass-through to a rejected decision regardless of mode.
            utility_score: W3-06 utility score for this frame, or
                ``None`` if scoring wasn't run/available.
            predicted_cost: Normalized GPU-cost proxy (see
                ``cost_model.predict_gpu_cost``), used by
                ``ACTIVE_WITH_BUDGET`` and carried through for logging.
            requested_burst_steps / replay_weight: Only consulted in
                ``ACTIVE_WITH_BUDGET`` mode to build a
                :class:`MappingBudgetRequest`.
            meta: Evidence metadata to attach to the decision.
        """
        meta = meta or EvidenceMeta(frame_id=frame_id)

        # Hard safety gates are non-negotiable in EVERY rollout mode,
        # including ACTIVE / ACTIVE_WITH_BUDGET. This check runs first,
        # before any mode-specific branching, precisely so no later branch
        # can accidentally skip it.
        if hard_gate_blocked:
            return AdmissionDecision(
                frame_id=frame_id,
                rollout_mode=self.mode,
                baseline_admitted=baseline_admitted,
                admitted=False,
                overridden=False,
                hard_gate_blocked=True,
                reason_code=REASON_HARD_SAFETY_VETO,
                reason_detail="hard safety gate (tracking/map-id/timestamp/source validity) rejected this frame",
                utility=utility_score.utility if utility_score is not None else None,
                predicted_cost=predicted_cost,
                budget_request=None,
                meta=meta,
            )

        if self.mode == RolloutMode.SHADOW:
            return AdmissionDecision(
                frame_id=frame_id,
                rollout_mode=self.mode,
                baseline_admitted=baseline_admitted,
                admitted=baseline_admitted,
                overridden=False,
                hard_gate_blocked=False,
                reason_code=REASON_BASELINE_PASSTHROUGH,
                reason_detail="shadow mode: utility policy observes only, baseline decision is authoritative",
                utility=utility_score.utility if utility_score is not None else None,
                predicted_cost=predicted_cost,
                budget_request=None,
                meta=meta,
            )

        if utility_score is None:
            # No usable score (scorer disabled/failed this call) -- every
            # non-shadow mode falls back to the baseline rather than
            # guessing, so a scoring outage never silently changes
            # behavior in a direction nobody reviewed.
            return AdmissionDecision(
                frame_id=frame_id,
                rollout_mode=self.mode,
                baseline_admitted=baseline_admitted,
                admitted=baseline_admitted,
                overridden=False,
                hard_gate_blocked=False,
                reason_code=REASON_NO_UTILITY_SCORE,
                reason_detail="no utility score available this call; falling back to baseline admission",
                utility=None,
                predicted_cost=predicted_cost,
                budget_request=None,
                meta=meta,
            )

        utility = float(utility_score.utility)
        policy_wants_admit = utility >= self.utility_admit_threshold

        if self.mode == RolloutMode.ADVISORY:
            # May only shrink the admitted set: reject a baseline
            # admission the utility policy disagrees with, but never
            # admit a frame the baseline rejected.
            if baseline_admitted and not policy_wants_admit:
                return AdmissionDecision(
                    frame_id=frame_id,
                    rollout_mode=self.mode,
                    baseline_admitted=baseline_admitted,
                    admitted=False,
                    overridden=True,
                    hard_gate_blocked=False,
                    reason_code=REASON_ADVISORY_REJECT,
                    reason_detail=f"advisory policy rejected a baseline admission (utility={utility:.4f} < threshold={self.utility_admit_threshold:.4f})",
                    utility=utility,
                    predicted_cost=predicted_cost,
                    budget_request=None,
                    meta=meta,
                )
            return AdmissionDecision(
                frame_id=frame_id,
                rollout_mode=self.mode,
                baseline_admitted=baseline_admitted,
                admitted=baseline_admitted,
                overridden=False,
                hard_gate_blocked=False,
                reason_code=REASON_BASELINE_PASSTHROUGH,
                reason_detail="advisory policy did not disagree with the baseline decision",
                utility=utility,
                predicted_cost=predicted_cost,
                budget_request=None,
                meta=meta,
            )

        # ACTIVE / ACTIVE_WITH_BUDGET: the utility policy owns the
        # decision outright, in both directions.
        final_admit = policy_wants_admit
        reason_code = REASON_ACTIVE_ADMIT if final_admit else REASON_ACTIVE_REJECT
        reason_detail = (
            f"active policy {'admitted' if final_admit else 'rejected'} "
            f"(utility={utility:.4f}, threshold={self.utility_admit_threshold:.4f})"
        )
        budget_request: Optional[MappingBudgetRequest] = None

        if self.mode == RolloutMode.ACTIVE_WITH_BUDGET and final_admit:
            assert self.budget_controller is not None  # enforced at __init__
            if not self.budget_controller.try_admit(predicted_cost):
                final_admit = False
                reason_code = REASON_BUDGET_EXHAUSTED
                reason_detail = (
                    f"active_with_budget policy wanted to admit (utility={utility:.4f}) "
                    "but the mapping budget for this window is exhausted"
                )
            else:
                raw_request = MappingBudgetRequest(
                    frame_id=frame_id,
                    local_burst_steps=int(requested_burst_steps),
                    replay_weight=float(replay_weight),
                )
                budget_request = self.budget_controller.request_budget(raw_request)

        return AdmissionDecision(
            frame_id=frame_id,
            rollout_mode=self.mode,
            baseline_admitted=baseline_admitted,
            admitted=final_admit,
            overridden=final_admit != baseline_admitted,
            hard_gate_blocked=False,
            reason_code=reason_code,
            reason_detail=reason_detail,
            utility=utility,
            predicted_cost=predicted_cost,
            budget_request=budget_request,
            meta=meta,
        )
