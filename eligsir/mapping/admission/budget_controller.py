"""Mapping budget controller (Wave 3, PR W3-07, plan section 12.4,
Subagent D).

Keeps two independent, explicit budgets bounded over a rolling window so
``active_with_budget`` mode has a hard, auditable ceiling rather than an
implicit "however much the utility policy wants":

- **Frame-count budget**: how many frames may be admitted per window.
- **Compute budget**: total normalized GPU-cost (``cost_model.predict_gpu_cost``
  units, see that module's docstring for what "normalized" means here) that
  may be admitted per window.

Deliberately NOT a rate limiter with implicit global state -- every method
is a plain call on an explicit object the caller owns, and
:meth:`MappingBudgetController.reset_window` is the caller's own choice of
when a window ends (e.g. every N iterations, or every N admitted frames).
This mirrors the bounded-queue philosophy from W3-03: a budget that isn't
enforced by an explicit, inspectable counter isn't a budget, just a hope.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from eligsir.mapping.admission.decision_contracts import MappingBudgetRequest


@dataclass
class MappingBudgetControllerConfig:
    """Tunables for :class:`MappingBudgetController`.

    Attributes:
        max_frames_per_window: Hard ceiling on admitted-frame count per
            window. ``0`` disables the frame-count budget (unlimited).
        max_compute_per_window: Hard ceiling on cumulative normalized
            GPU-cost admitted per window. ``0.0`` disables the compute
            budget (unlimited).
        max_burst_steps_per_frame: Ceiling on
            ``MappingBudgetRequest.local_burst_steps`` a single request
            may be granted, independent of the window budgets above (a
            per-frame fairness cap so one frame can't consume an entire
            window's compute budget on its own).
    """

    max_frames_per_window: int = 0
    max_compute_per_window: float = 0.0
    max_burst_steps_per_frame: int = 200


@dataclass
class MappingBudgetController:
    """Rolling-window admission-budget accountant.

    Deliverable 3 ("active policy operates under explicit frame-count and
    compute budgets") and deliverable 5 ("no performance regression beyond
    the declared budget") both cash out here: every admission the rollout
    policy wants to grant in ``active_with_budget`` mode must pass
    :meth:`try_admit` first, and every optimization-burst request must
    pass :meth:`request_budget` -- there is no other path to consuming
    budget.
    """

    config: MappingBudgetControllerConfig = field(default_factory=MappingBudgetControllerConfig)
    _frames_admitted_this_window: int = field(default=0, init=False, repr=False)
    _compute_admitted_this_window: float = field(default=0.0, init=False, repr=False)
    _windows_closed: int = field(default=0, init=False, repr=False)
    _total_frames_admitted: int = field(default=0, init=False, repr=False)
    _total_frames_denied: int = field(default=0, init=False, repr=False)

    # ------------------------------------------------------------------
    def try_admit(self, predicted_cost: float) -> bool:
        """Returns True and books ``predicted_cost`` against the current
        window iff BOTH the frame-count and compute budgets have room.
        Never mutates state on a denial (idempotent to retry after
        :meth:`reset_window`).
        """
        predicted_cost = max(0.0, float(predicted_cost))
        frame_budget_ok = (
            self.config.max_frames_per_window <= 0
            or self._frames_admitted_this_window < self.config.max_frames_per_window
        )
        compute_budget_ok = (
            self.config.max_compute_per_window <= 0.0
            or self._compute_admitted_this_window + predicted_cost <= self.config.max_compute_per_window
        )
        if frame_budget_ok and compute_budget_ok:
            self._frames_admitted_this_window += 1
            self._compute_admitted_this_window += predicted_cost
            self._total_frames_admitted += 1
            return True
        self._total_frames_denied += 1
        return False

    def request_budget(self, request: MappingBudgetRequest) -> MappingBudgetRequest:
        """Grants (a possibly-clamped amount of) the requested local
        optimization burst. Never raises; a fully-denied request still
        returns a valid :class:`MappingBudgetRequest` with
        ``granted_burst_steps=0``.
        """
        requested = max(0, int(request.local_burst_steps))
        per_frame_cap = max(0, int(self.config.max_burst_steps_per_frame))
        granted = min(requested, per_frame_cap) if per_frame_cap else requested
        reason = "granted_in_full"
        if granted < requested:
            reason = "clamped_to_per_frame_cap"
        if granted <= 0 and requested > 0:
            reason = "budget_exhausted"
        return MappingBudgetRequest(
            frame_id=request.frame_id,
            local_burst_steps=request.local_burst_steps,
            replay_weight=request.replay_weight,
            granted_burst_steps=granted,
            reason_code=reason,
        )

    # ------------------------------------------------------------------
    def reset_window(self) -> None:
        """Ends the current accounting window; counts roll over to
        ``_total_*`` for lifetime telemetry, and the per-window counters
        return to zero so the next window starts with a full budget."""
        self._windows_closed += 1
        self._frames_admitted_this_window = 0
        self._compute_admitted_this_window = 0.0

    def remaining_frame_budget(self) -> Optional[int]:
        """``None`` when the frame-count budget is disabled, else the
        number of additional frames the current window can still admit."""
        if self.config.max_frames_per_window <= 0:
            return None
        return max(0, self.config.max_frames_per_window - self._frames_admitted_this_window)

    def remaining_compute_budget(self) -> Optional[float]:
        """``None`` when the compute budget is disabled, else the
        remaining normalized-cost headroom in the current window."""
        if self.config.max_compute_per_window <= 0.0:
            return None
        return max(0.0, self.config.max_compute_per_window - self._compute_admitted_this_window)

    def summary(self) -> dict:
        """Machine-readable snapshot for logging/reporting (mirrors
        ``ShadowUtilityPolicy.cost_overhead_ms_summary``'s convention)."""
        return {
            "windows_closed": self._windows_closed,
            "frames_admitted_this_window": self._frames_admitted_this_window,
            "compute_admitted_this_window": self._compute_admitted_this_window,
            "total_frames_admitted": self._total_frames_admitted,
            "total_frames_denied": self._total_frames_denied,
            "max_frames_per_window": self.config.max_frames_per_window,
            "max_compute_per_window": self.config.max_compute_per_window,
        }
