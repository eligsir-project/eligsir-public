"""Tests for the W3-07 active, evidence-driven admission rollout policy.

Pure CPU/Python, no GPU/container dependency -- mirrors
tests/test_frame_utility.py's style for the substrate this module
consumes (W3-06's FrameUtilityScore).
"""

from __future__ import annotations

import pytest

from eligsir.mapping.admission.budget_controller import (
    MappingBudgetController,
    MappingBudgetControllerConfig,
)
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
from eligsir.mapping.admission.rollout_policy import AdmissionRolloutPolicy
from eligsir.mapping.admission.utility_contracts import (
    FrameUtilityComponents,
    FrameUtilityScore,
    UtilityWeights,
)


def _score(utility: float, frame_id: int = 1) -> FrameUtilityScore:
    """Builds a well-formed FrameUtilityScore with the given final utility
    (components/weights are irrelevant to the rollout policy -- it only
    reads ``.utility``)."""
    components = FrameUtilityComponents(
        pose_quality=1.0,
        image_quality=1.0,
        new_surface=0.0,
        persistent_depth_supported_residual=0.0,
        under_supported_mass=0.0,
        expected_gradient_value=0.0,
        useful_parallax=0.0,
        redundancy=0.0,
        specular_risk=0.0,
        predicted_gpu_cost=0.0,
    )
    return FrameUtilityScore(frame_id=frame_id, utility=utility, components=components, weights=UtilityWeights())


# ----------------------------------------------------------------------
# AdmissionDecision contract invariants
# ----------------------------------------------------------------------
def test_admission_decision_rejects_bad_reason_code():
    with pytest.raises(ValueError):
        AdmissionDecision(
            frame_id=1,
            rollout_mode=RolloutMode.SHADOW,
            baseline_admitted=True,
            admitted=True,
            overridden=False,
            hard_gate_blocked=False,
            reason_code="not_a_real_reason_code",
        )


def test_admission_decision_hard_gate_blocked_cannot_be_admitted():
    with pytest.raises(ValueError):
        AdmissionDecision(
            frame_id=1,
            rollout_mode=RolloutMode.ACTIVE,
            baseline_admitted=True,
            admitted=True,  # invalid: hard_gate_blocked forces admitted=False
            overridden=False,
            hard_gate_blocked=True,
            reason_code=REASON_HARD_SAFETY_VETO,
        )


def test_admission_decision_hard_gate_blocked_cannot_be_overridden_flag():
    with pytest.raises(ValueError):
        AdmissionDecision(
            frame_id=1,
            rollout_mode=RolloutMode.ACTIVE,
            baseline_admitted=True,
            admitted=False,
            overridden=True,  # invalid alongside hard_gate_blocked
            hard_gate_blocked=True,
            reason_code=REASON_HARD_SAFETY_VETO,
        )


# ----------------------------------------------------------------------
# Deliverable: hard safety gates are NEVER overridden, in ANY rollout mode
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    "mode",
    [RolloutMode.SHADOW, RolloutMode.ADVISORY, RolloutMode.ACTIVE, RolloutMode.ACTIVE_WITH_BUDGET],
)
def test_hard_gate_blocks_every_rollout_mode_even_with_maximal_utility(mode):
    """Adversarial test: construct the utility policy's most favorable
    possible input (utility far above threshold, requesting a full-size
    optimization burst) and confirm a hard-safety-gate rejection still
    wins in every single rollout mode, including ACTIVE_WITH_BUDGET with
    an unlimited budget."""
    budget_controller = None
    if mode == RolloutMode.ACTIVE_WITH_BUDGET:
        budget_controller = MappingBudgetController(MappingBudgetControllerConfig())  # unlimited
    policy = AdmissionRolloutPolicy(mode=mode, utility_admit_threshold=-999.0, budget_controller=budget_controller)
    decision = policy.decide(
        frame_id=42,
        baseline_admitted=False,  # even if baseline itself already rejected
        hard_gate_blocked=True,
        utility_score=_score(utility=9999.0, frame_id=42),
        predicted_cost=0.0,
        requested_burst_steps=1000,
    )
    assert decision.admitted is False
    assert decision.overridden is False
    assert decision.hard_gate_blocked is True
    assert decision.reason_code == REASON_HARD_SAFETY_VETO
    assert decision.budget_request is None


def test_hard_gate_blocks_even_when_baseline_admitted():
    """A baseline that already admitted the frame must still be
    overturned by a hard gate -- this can happen when the hard-gate
    condition (e.g. a mid-call tracking re-init) is detected after the
    baseline admitter already ran but before the rollout policy is
    consulted."""
    policy = AdmissionRolloutPolicy(mode=RolloutMode.ACTIVE, utility_admit_threshold=0.0)
    decision = policy.decide(
        frame_id=1,
        baseline_admitted=True,
        hard_gate_blocked=True,
        utility_score=_score(utility=100.0),
    )
    assert decision.admitted is False
    assert decision.hard_gate_blocked is True


# ----------------------------------------------------------------------
# SHADOW mode: deliverable 1, no decision changes ever
# ----------------------------------------------------------------------
@pytest.mark.parametrize("baseline", [True, False])
@pytest.mark.parametrize("utility", [-100.0, 0.0, 100.0])
def test_shadow_mode_never_changes_the_baseline_decision(baseline, utility):
    policy = AdmissionRolloutPolicy(mode=RolloutMode.SHADOW, utility_admit_threshold=0.0)
    decision = policy.decide(
        frame_id=1,
        baseline_admitted=baseline,
        hard_gate_blocked=False,
        utility_score=_score(utility=utility),
    )
    assert decision.admitted == baseline
    assert decision.overridden is False
    assert decision.reason_code == REASON_BASELINE_PASSTHROUGH


def test_shadow_mode_never_changes_baseline_even_with_no_score():
    policy = AdmissionRolloutPolicy(mode=RolloutMode.SHADOW)
    decision = policy.decide(frame_id=1, baseline_admitted=True, hard_gate_blocked=False, utility_score=None)
    assert decision.admitted is True
    assert decision.overridden is False


# ----------------------------------------------------------------------
# ADVISORY mode: may only shrink the admitted set
# ----------------------------------------------------------------------
def test_advisory_mode_can_reject_a_baseline_admission():
    policy = AdmissionRolloutPolicy(mode=RolloutMode.ADVISORY, utility_admit_threshold=0.0)
    decision = policy.decide(
        frame_id=1, baseline_admitted=True, hard_gate_blocked=False, utility_score=_score(utility=-1.0)
    )
    assert decision.admitted is False
    assert decision.overridden is True
    assert decision.reason_code == REASON_ADVISORY_REJECT


def test_advisory_mode_cannot_admit_a_baseline_rejection():
    """The core advisory-mode invariant: even a very high utility score
    must not turn a baseline rejection into an admission."""
    policy = AdmissionRolloutPolicy(mode=RolloutMode.ADVISORY, utility_admit_threshold=0.0)
    decision = policy.decide(
        frame_id=1, baseline_admitted=False, hard_gate_blocked=False, utility_score=_score(utility=999.0)
    )
    assert decision.admitted is False
    assert decision.overridden is False
    assert decision.reason_code == REASON_BASELINE_PASSTHROUGH


def test_advisory_mode_agrees_with_baseline_admission_when_utility_high():
    policy = AdmissionRolloutPolicy(mode=RolloutMode.ADVISORY, utility_admit_threshold=0.0)
    decision = policy.decide(
        frame_id=1, baseline_admitted=True, hard_gate_blocked=False, utility_score=_score(utility=1.0)
    )
    assert decision.admitted is True
    assert decision.overridden is False


# ----------------------------------------------------------------------
# ACTIVE mode: owns the decision in both directions
# ----------------------------------------------------------------------
def test_active_mode_can_admit_a_baseline_rejection():
    policy = AdmissionRolloutPolicy(mode=RolloutMode.ACTIVE, utility_admit_threshold=0.0)
    decision = policy.decide(
        frame_id=1, baseline_admitted=False, hard_gate_blocked=False, utility_score=_score(utility=5.0)
    )
    assert decision.admitted is True
    assert decision.overridden is True
    assert decision.reason_code == REASON_ACTIVE_ADMIT


def test_active_mode_can_reject_a_baseline_admission():
    policy = AdmissionRolloutPolicy(mode=RolloutMode.ACTIVE, utility_admit_threshold=0.0)
    decision = policy.decide(
        frame_id=1, baseline_admitted=True, hard_gate_blocked=False, utility_score=_score(utility=-5.0)
    )
    assert decision.admitted is False
    assert decision.overridden is True
    assert decision.reason_code == REASON_ACTIVE_REJECT


def test_active_mode_falls_back_to_baseline_when_no_score():
    policy = AdmissionRolloutPolicy(mode=RolloutMode.ACTIVE)
    decision = policy.decide(frame_id=1, baseline_admitted=True, hard_gate_blocked=False, utility_score=None)
    assert decision.admitted is True
    assert decision.overridden is False
    assert decision.reason_code == REASON_NO_UTILITY_SCORE


# ----------------------------------------------------------------------
# ACTIVE_WITH_BUDGET mode: deliverable 3, explicit frame-count/compute budgets
# ----------------------------------------------------------------------
def test_active_with_budget_requires_a_budget_controller():
    with pytest.raises(ValueError):
        AdmissionRolloutPolicy(mode=RolloutMode.ACTIVE_WITH_BUDGET, budget_controller=None)


def test_active_with_budget_admits_within_budget_and_grants_burst():
    controller = MappingBudgetController(
        MappingBudgetControllerConfig(max_frames_per_window=10, max_compute_per_window=100.0, max_burst_steps_per_frame=50)
    )
    policy = AdmissionRolloutPolicy(mode=RolloutMode.ACTIVE_WITH_BUDGET, utility_admit_threshold=0.0, budget_controller=controller)
    decision = policy.decide(
        frame_id=1,
        baseline_admitted=False,
        hard_gate_blocked=False,
        utility_score=_score(utility=1.0),
        predicted_cost=1.0,
        requested_burst_steps=30,
    )
    assert decision.admitted is True
    assert decision.budget_request is not None
    assert decision.budget_request.granted_burst_steps == 30


def test_active_with_budget_rejects_once_frame_count_budget_exhausted():
    controller = MappingBudgetController(MappingBudgetControllerConfig(max_frames_per_window=2))
    policy = AdmissionRolloutPolicy(mode=RolloutMode.ACTIVE_WITH_BUDGET, utility_admit_threshold=0.0, budget_controller=controller)
    decisions = [
        policy.decide(frame_id=i, baseline_admitted=False, hard_gate_blocked=False, utility_score=_score(utility=1.0))
        for i in range(4)
    ]
    admitted = [d.admitted for d in decisions]
    assert admitted == [True, True, False, False]
    assert decisions[2].reason_code == REASON_BUDGET_EXHAUSTED
    # baseline_admitted=False for every candidate here, so a budget-denied
    # admission (admitted=False) matches the baseline and is NOT an override
    # -- overridden only reflects a *behavior change* vs. the baseline.
    assert decisions[2].overridden is False


def test_active_with_budget_rejects_once_compute_budget_exhausted():
    controller = MappingBudgetController(MappingBudgetControllerConfig(max_compute_per_window=1.5))
    policy = AdmissionRolloutPolicy(mode=RolloutMode.ACTIVE_WITH_BUDGET, utility_admit_threshold=0.0, budget_controller=controller)
    d1 = policy.decide(frame_id=1, baseline_admitted=False, hard_gate_blocked=False, utility_score=_score(utility=1.0), predicted_cost=1.0)
    d2 = policy.decide(frame_id=2, baseline_admitted=False, hard_gate_blocked=False, utility_score=_score(utility=1.0), predicted_cost=1.0)
    assert d1.admitted is True
    assert d2.admitted is False
    assert d2.reason_code == REASON_BUDGET_EXHAUSTED


def test_active_with_budget_utility_rejection_never_touches_budget():
    """A frame the utility policy itself rejects must not consume any
    budget -- only frames the policy WANTS to admit draw from the budget
    (confirmed via the controller's own accounting)."""
    controller = MappingBudgetController(MappingBudgetControllerConfig(max_frames_per_window=1))
    policy = AdmissionRolloutPolicy(mode=RolloutMode.ACTIVE_WITH_BUDGET, utility_admit_threshold=0.0, budget_controller=controller)
    decision = policy.decide(frame_id=1, baseline_admitted=False, hard_gate_blocked=False, utility_score=_score(utility=-1.0))
    assert decision.admitted is False
    assert decision.reason_code == REASON_ACTIVE_REJECT
    assert controller.summary()["total_frames_admitted"] == 0


# ----------------------------------------------------------------------
# MappingBudgetController standalone behavior
# ----------------------------------------------------------------------
def test_budget_controller_reset_window_restores_full_budget():
    controller = MappingBudgetController(MappingBudgetControllerConfig(max_frames_per_window=1))
    assert controller.try_admit(0.0) is True
    assert controller.try_admit(0.0) is False
    controller.reset_window()
    assert controller.try_admit(0.0) is True


def test_budget_controller_unlimited_when_zero():
    controller = MappingBudgetController(MappingBudgetControllerConfig())
    for _ in range(1000):
        assert controller.try_admit(1_000_000.0) is True


def test_budget_controller_remaining_budget_reporting():
    controller = MappingBudgetController(MappingBudgetControllerConfig(max_frames_per_window=5, max_compute_per_window=10.0))
    controller.try_admit(3.0)
    assert controller.remaining_frame_budget() == 4
    assert controller.remaining_compute_budget() == 7.0


def test_budget_controller_remaining_budget_none_when_disabled():
    controller = MappingBudgetController(MappingBudgetControllerConfig())
    assert controller.remaining_frame_budget() is None
    assert controller.remaining_compute_budget() is None


def test_budget_request_clamped_to_per_frame_cap():
    controller = MappingBudgetController(MappingBudgetControllerConfig(max_burst_steps_per_frame=10))
    granted = controller.request_budget(MappingBudgetRequest(frame_id=1, local_burst_steps=100))
    assert granted.granted_burst_steps == 10
    assert granted.reason_code == "clamped_to_per_frame_cap"


def test_budget_request_zero_steps_is_trivially_granted():
    controller = MappingBudgetController(MappingBudgetControllerConfig())
    granted = controller.request_budget(MappingBudgetRequest(frame_id=1, local_burst_steps=0))
    assert granted.granted_burst_steps == 0
    assert granted.reason_code == "granted_in_full"
