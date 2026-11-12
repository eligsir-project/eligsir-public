"""Typed contracts for active, evidence-driven admission and mapping
budget control (Wave 3, PR W3-07, plan section 12).

This module deliberately holds ONLY dataclasses/enums -- no policy logic.
:mod:`eligsir.mapping.admission.rollout_policy` is where the graduated
rollout (``shadow`` -> ``advisory`` -> ``active`` -> ``active_with_budget``)
lives; :mod:`eligsir.mapping.admission.budget_controller` is where the
frame-count/compute budget accounting lives. Splitting the typed result
objects out mirrors ``utility_contracts.py``'s split from
``scoring.py``/``shadow_policy.py`` in W3-06 -- a log/test consumer should
never have to import policy code just to read a result's shape.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from eligsir.mapping.evidence.contracts import EvidenceMeta


class RolloutMode(str, Enum):
    """Graduated rollout modes for the active-admission policy (plan
    section 12.3). ``str`` subclass so CLI args / JSON logs serialize as
    plain strings without extra glue.

    - ``SHADOW``: no decision changes -- the utility policy only observes
      and logs (identical to W3-06's ``ShadowUtilityPolicy`` behavior).
      **This is the wired-in default for every real run** (deliverable 6:
      no active promotion without cross-scene evidence, which does not
      exist yet for this scorer -- see W3-06's inherited coverage gap).
    - ``ADVISORY``: the utility policy may REJECT a frame the current
      (baseline) admission policy admitted, but may never ADMIT a frame
      the baseline rejected -- i.e. it can only shrink the admitted set,
      never grow it.
    - ``ACTIVE``: the utility policy owns the admit/reject decision
      outright, in both directions, but ONLY within whatever the hard
      safety gates already allow (see ``hard_gate_blocked`` on
      :class:`AdmissionDecision` -- a hard gate rejection is checked
      BEFORE the rollout policy ever runs and can never be reversed by
      it, in any mode, by construction).
    - ``ACTIVE_WITH_BUDGET``: same authority as ``ACTIVE``, plus an
      explicit frame-count/compute budget (see
      :mod:`eligsir.mapping.admission.budget_controller`) that can
      additionally reject an admission the utility policy itself wanted,
      once the declared budget for the current window is exhausted.
    """

    SHADOW = "shadow"
    ADVISORY = "advisory"
    ACTIVE = "active"
    ACTIVE_WITH_BUDGET = "active_with_budget"


# Machine-readable reason codes (deliverable 2: "all decisions include
# machine-readable reasons"). Every :class:`AdmissionDecision` populates
# ``reason_code`` with exactly one of these -- never a free-text-only
# reason -- so a log consumer can bucket/aggregate decisions without
# string-matching human prose. ``reason_detail`` carries the human-
# readable elaboration.
REASON_HARD_SAFETY_VETO = "hard_safety_veto"
REASON_BASELINE_PASSTHROUGH = "baseline_passthrough"
REASON_ADVISORY_REJECT = "advisory_reject"
REASON_ACTIVE_ADMIT = "active_admit"
REASON_ACTIVE_REJECT = "active_reject"
REASON_NO_UTILITY_SCORE = "no_utility_score_fallback_to_baseline"
REASON_BUDGET_EXHAUSTED = "budget_exhausted"

ADMISSION_REASON_CODES = frozenset(
    {
        REASON_HARD_SAFETY_VETO,
        REASON_BASELINE_PASSTHROUGH,
        REASON_ADVISORY_REJECT,
        REASON_ACTIVE_ADMIT,
        REASON_ACTIVE_REJECT,
        REASON_NO_UTILITY_SCORE,
        REASON_BUDGET_EXHAUSTED,
    }
)


@dataclass(frozen=True)
class AdmissionDecision:
    """Typed outcome of one W3-07 rollout-policy admission call.

    Replaces the ad-hoc ``(bool, dict)`` pattern with a self-describing
    record: every field a log/test consumer needs to answer "what was
    decided, why, and did the utility policy actually change anything"
    is present, not reconstructed from a loosely-typed stats dict.

    Attributes:
        frame_id: Source-frame index this decision covers.
        rollout_mode: Which mode produced this decision.
        baseline_admitted: What the pre-existing admission policy
            (``KeyframeAdmitter.admit`` / the hard-veto short-circuit)
            decided, independent of the utility policy.
        admitted: The FINAL admission outcome after the rollout policy
            has been applied. Always equals ``baseline_admitted`` in
            ``SHADOW`` mode and whenever ``hard_gate_blocked`` is True.
        overridden: True iff ``admitted != baseline_admitted``.
        hard_gate_blocked: True iff a hard safety gate (tracking
            validity / map-ID consistency / timestamp integrity / source
            validity) forced ``admitted = False`` BEFORE the utility
            policy was ever consulted. When True, ``admitted`` is always
            False and ``overridden`` is always False, regardless of
            rollout mode -- see :class:`RolloutMode`'s docstring.
        reason_code: One of :data:`ADMISSION_REASON_CODES`.
        reason_detail: Human-readable elaboration of ``reason_code``.
        utility: The utility score consulted, or ``None`` when no
            utility score was available for this candidate (e.g. shadow
            scoring not enabled, or the scorer raised and was caught).
        predicted_cost: Normalized ``[0, ~1.8]`` GPU-cost proxy from
            ``cost_model.predict_gpu_cost`` for this frame, carried here
            so ``ACTIVE_WITH_BUDGET`` decisions are auditable without a
            second lookup.
        budget_request: The optimization/replay budget this decision
            would like to draw, or ``None`` outside
            ``ACTIVE_WITH_BUDGET`` mode.
        meta: Standard evidence metadata (frame/map/topology version).
    """

    frame_id: int
    rollout_mode: RolloutMode
    baseline_admitted: bool
    admitted: bool
    overridden: bool
    hard_gate_blocked: bool
    reason_code: str
    reason_detail: str = ""
    utility: Optional[float] = None
    predicted_cost: float = 0.0
    budget_request: Optional["MappingBudgetRequest"] = None
    meta: EvidenceMeta = field(default_factory=EvidenceMeta)

    def __post_init__(self) -> None:
        if self.reason_code not in ADMISSION_REASON_CODES:
            raise ValueError(
                f"AdmissionDecision.reason_code must be one of {sorted(ADMISSION_REASON_CODES)}, "
                f"got {self.reason_code!r}"
            )
        if self.hard_gate_blocked and (self.admitted or self.overridden):
            raise ValueError(
                "AdmissionDecision invariant violated: hard_gate_blocked=True must imply "
                "admitted=False and overridden=False (hard safety gates are never overridden)"
            )


@dataclass(frozen=True)
class MappingBudgetRequest:
    """How much optimization/replay attention an admitted frame should
    receive -- deliberately separate from the admit/reject decision
    itself (plan section 12.4: "a frame may deserve storage yet not
    immediate heavy refinement, or it may deserve a short local burst
    because it exposes a weak region").

    Only ever produced in ``ACTIVE_WITH_BUDGET`` mode; ``None`` in every
    other mode (nothing consumes a budget that was never requested).

    Attributes:
        frame_id: Source-frame index this request covers.
        local_burst_steps: Extra local optimization iterations requested
            beyond the normal per-frame allotment (0 = no burst).
        replay_weight: Relative weight ``[0, 1]`` this frame should carry
            in future replay-buffer sampling (0 = never replay again,
            1 = maximum priority).
        granted_burst_steps: What the budget controller actually granted
            (<= ``local_burst_steps``; 0 if the request was denied
            outright by the budget controller).
        reason_code: Why the grant differs from the request (e.g.
            ``"granted_in_full"``, ``"budget_exhausted"``,
            ``"clamped_to_remaining_budget"``).
    """

    frame_id: int
    local_burst_steps: int = 0
    replay_weight: float = 0.0
    granted_burst_steps: int = 0
    reason_code: str = "granted_in_full"
