"""Surface-cell budget bookkeeping (Wave 3, PR W3-09, plan section 14.7
"predicted affected cells").

Caps how many reseed/growth requests may target the same surface cell in a
single proposal round, so a burst of high-utility candidates in one small
region can't dominate the whole plan. Pure bookkeeping over
:class:`~eligsir.optimization.gaussian.strategies.contracts.SurfaceSeedRequest`
objects -- never touches gaussian model tensors.
"""

from __future__ import annotations

from dataclasses import dataclass

from eligsir.optimization.gaussian.strategies.contracts import SurfaceSeedRequest


@dataclass
class CellBudgetTracker:
    budget_per_cell: int = 4
    max_total_requests: int = 2000

    def admit(
        self, requests: tuple[SurfaceSeedRequest, ...]
    ) -> tuple[tuple[SurfaceSeedRequest, ...], dict[str, int]]:
        """Rank ``requests`` by ``utility`` (descending), then admit up to
        ``budget_per_cell`` per ``cell_id`` and ``max_total_requests``
        overall. Returns ``(admitted, rejection_reason_counts)``.
        """
        rejections: dict[str, int] = {}
        ranked = sorted(requests, key=lambda r: r.utility, reverse=True)
        per_cell_count: dict[int, int] = {}
        admitted: list[SurfaceSeedRequest] = []
        for req in ranked:
            if len(admitted) >= self.max_total_requests:
                rejections["cell_budget:max_total_requests"] = rejections.get(
                    "cell_budget:max_total_requests", 0
                ) + 1
                continue
            count = per_cell_count.get(req.cell_id, 0)
            if count >= self.budget_per_cell:
                rejections["cell_budget:per_cell_cap"] = rejections.get("cell_budget:per_cell_cap", 0) + 1
                continue
            per_cell_count[req.cell_id] = count + 1
            admitted.append(req)
        return tuple(admitted), rejections

    @staticmethod
    def affected_cell_count(requests: tuple[SurfaceSeedRequest, ...]) -> int:
        return len({r.cell_id for r in requests})


@dataclass
class TieredCellBudget:
    """Per-cell budget split into minimum/normal/temporary/reserve tiers
    (Wave 3, PR W3-10, plan section 15.3 Phase D): "Allocate minimum,
    normal, temporary, and reserve budgets per surface/spatial cell."

    Only active when ``ActivationPhaseConfig.phase_d_cell_budgets`` is
    True (see ``live/backend.py``) -- with that flag off, callers keep
    using the flat ``CellBudgetTracker`` above (W3-09 behavior, unchanged).

    **Scope limitation, documented honestly (not hidden):** tiers here are
    capacity buckets only, filled strictly by descending ``utility``
    within each cell -- a request is not yet allowed to *declare* which
    tier it needs (e.g. "this cell has zero live coverage, it needs the
    guaranteed minimum tier regardless of utility rank"). A fuller,
    need-based routing design (matching under-observed cells to the
    ``minimum`` tier specifically) is deferred; this PR only builds and
    tests the tier *accounting*, not need-aware routing.
    """

    minimum: int = 1
    normal: int = 3
    temporary: int = 2
    reserve: int = 1
    max_total_requests: int = 2000

    @property
    def total_per_cell(self) -> int:
        return self.minimum + self.normal + self.temporary + self.reserve

    def admit(
        self, requests: tuple[SurfaceSeedRequest, ...]
    ) -> tuple[tuple[SurfaceSeedRequest, ...], dict[str, int]]:
        """Rank ``requests`` by ``utility`` (descending), then admit up to
        ``total_per_cell`` per ``cell_id`` (filling minimum, then normal,
        then temporary, then reserve, in that order) and
        ``max_total_requests`` overall. Returns ``(admitted,
        rejection_reason_counts)`` -- same shape as
        ``CellBudgetTracker.admit`` so callers can swap between the two
        without touching downstream code."""
        rejections: dict[str, int] = {}
        ranked = sorted(requests, key=lambda r: r.utility, reverse=True)
        per_cell_tier_counts: dict[int, dict[str, int]] = {}
        admitted: list[SurfaceSeedRequest] = []
        for req in ranked:
            if len(admitted) >= self.max_total_requests:
                rejections["cell_budget:max_total_requests"] = rejections.get(
                    "cell_budget:max_total_requests", 0
                ) + 1
                continue
            tiers = per_cell_tier_counts.setdefault(
                req.cell_id, {"minimum": 0, "normal": 0, "temporary": 0, "reserve": 0}
            )
            if tiers["minimum"] < self.minimum:
                tiers["minimum"] += 1
            elif tiers["normal"] < self.normal:
                tiers["normal"] += 1
            elif tiers["temporary"] < self.temporary:
                tiers["temporary"] += 1
            elif tiers["reserve"] < self.reserve:
                tiers["reserve"] += 1
            else:
                rejections["cell_budget:tiered_per_cell_cap"] = rejections.get(
                    "cell_budget:tiered_per_cell_cap", 0
                ) + 1
                continue
            admitted.append(req)
        return tuple(admitted), rejections
