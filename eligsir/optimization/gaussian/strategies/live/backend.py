"""Backend-applied mutation-plan execution (Wave 3, PR W3-10, plan section 15).

**This is the first module in Wave 3 that actually mutates a live
Gaussian model's tensors.** Every PR through W3-09 was strictly
observation/proposal-only (see ``strategies/contracts.py`` and
``strategies/live/strategy.py``'s docstrings). This module is the single,
visible crossing point: it is the only place that constructs a
:class:`~eligsir.optimization.gaussian.strategies.contracts.PopulationMutationPlan`-adjacent
mutation with ``observe_only=False`` semantics (via ``AlignedBufferRegistry``,
never by touching the plan the strategy itself returns -- that plan stays
``observe_only=True`` always).

Treat every one of the five activation phases below as its own real safety
boundary, independently gated and independently testable (plan section
15.3), not a single big-bang switch:

* **Phase A -- lifecycle updates only.** Already happens unconditionally
  inside ``EliGSiRLiveGaussianStrategy.propose()`` (writes only to
  ``LivePopulationState``, never a model tensor -- see that module).
  ``ActivationPhaseConfig.phase_a_lifecycle`` exists purely so a caller
  driving plans through this registry can independently confirm/log that
  this phase ran.
* **Phase B -- retirement only.** ``AlignedBufferRegistry.apply_retire``:
  forces ``retire_indices``' opacity to ``RETIRE_OPACITY_LOGIT_FLOOR`` in
  place (a real, bounded, reversible-in-principle tensor write). Retired
  rows are **not** removed and **not** reused by this phase alone -- "not
  reused" per plan section 15.3 literally.
* **Phase C -- explicit reseeding.** ``AlignedBufferRegistry.apply_reseed``:
  reuses RETIRED slots' *population capacity* for admitted
  ``SurfaceSeedRequest``\\ s by pruning up to ``len(reseed_requests)``
  currently-RETIRED rows and appending equally many freshly-initialised
  rows at the requested positions, via the model's own
  ``prune_points``/``add_points_as_gaussians`` (never hand-rolled --
  those two methods are already the single place ~25 aligned per-Gaussian
  buffers on ``GsplatGaussianModel`` are kept in sync; this module does
  not duplicate that logic, it wraps it). New rows get zero-initialised
  optimizer momentum for free (``add_points_as_gaussians``'s own
  documented behavior) -- this **is** the "explicit optimizer-state reset
  semantics" deliverable, not a separate mechanism bolted on top.
* **Phase D -- cell budgets.** Tiered admission via
  ``live/cell_budget.py``'s ``TieredCellBudget`` (minimum/normal/
  temporary/reserve), used in place of the flat ``CellBudgetTracker``
  when this flag is on. See that class's docstring for the documented
  scope limitation (tiers are capacity buckets, not yet need-routed).
* **Phase E -- constrained movement.** ``constrain_position_delta``: a
  standalone, unit-tested policy function clamping a proposed position
  delta for non-PROVISIONAL Gaussians by lifecycle/confidence and
  (optionally) a tangent-plane/normal decomposition. **Not wired into the
  actual MCMC noise-injection call site** (``utils/energy_mcmc.py``) by
  this PR -- built and tested in isolation only. See the W3-10 completion
  doc's honest "not met" list for why.

Every flag in :class:`ActivationPhaseConfig` defaults to ``False``. With
all five off, :meth:`AlignedBufferRegistry.apply` is a strict no-op --
this is the load-bearing test
(``tests/test_w3_10_active_lifecycle_backend.py::test_all_phases_off_is_bit_identical_noop``)
that protects every prior Wave-3 PR's "nothing was ever mutated" guarantee
for the default, out-of-the-box case.

**Deliverable 6 (safety-fuse vetoes are never counted as strategy wins):**
:class:`ApplyReport` keeps ``safety_fuse_triggers`` (copied verbatim from
``plan.reasons.safety_fuse_triggers``, i.e. what got vetoed/capped before
this module ever saw the plan) in a field entirely separate from
``retired_count``/``reseeded_count`` (what was actually, successfully
applied). A caller reading ``retired_count`` never sees vetoed rows folded
in -- they were already excluded from ``plan.retire_indices`` upstream by
``SafetyFuses.apply()`` (see ``mutation_plan.py``); this module additionally
never re-derives a "would have retired N" number from
``safety_fuse_triggers`` and adds it back into ``retired_count``, which is
exactly the miscounting bug class deliverable 6 exists to prevent -- see
the regression test ``test_vetoed_retirements_are_not_counted_as_applied``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import torch

from eligsir.optimization.gaussian.strategies.contracts import (
    GaussianLifecycleState,
    PopulationMutationPlan,
)
from eligsir.optimization.gaussian.strategies.live.state import LivePopulationState, state_to_code

RETIRE_OPACITY_LOGIT_FLOOR = -12.0
"""``sigmoid(-12) ~= 6.1e-6`` -- effectively invisible in any render, but
finite (unlike ``-inf``), so it stays a valid, differentiable optimizer
parameter rather than a value that could poison a gradient step if the
row is ever touched again before being reseeded/pruned."""


@dataclass(frozen=True)
class ActivationPhaseConfig:
    """Independent per-phase activation switches (plan section 15.3).
    Every flag defaults to ``False``. See module docstring for what each
    phase does."""

    phase_a_lifecycle: bool = False
    phase_b_retirement: bool = False
    phase_c_reseeding: bool = False
    phase_d_cell_budgets: bool = False
    phase_e_constrained_movement: bool = False

    def any_active(self) -> bool:
        return any(
            (
                self.phase_a_lifecycle,
                self.phase_b_retirement,
                self.phase_c_reseeding,
                self.phase_d_cell_budgets,
                self.phase_e_constrained_movement,
            )
        )

    def as_dict(self) -> dict:
        return {
            "phase_a_lifecycle": self.phase_a_lifecycle,
            "phase_b_retirement": self.phase_b_retirement,
            "phase_c_reseeding": self.phase_c_reseeding,
            "phase_d_cell_budgets": self.phase_d_cell_budgets,
            "phase_e_constrained_movement": self.phase_e_constrained_movement,
        }


@dataclass
class ApplyReport:
    """What :meth:`AlignedBufferRegistry.apply` actually did this call.
    See module docstring, deliverable 6, for why ``safety_fuse_triggers``
    is a field distinct from the applied-count fields."""

    phases_active: dict = field(default_factory=dict)
    retired_count: int = 0
    reseeded_count: int = 0
    pruned_for_reseed_count: int = 0
    safety_fuse_triggers: dict = field(default_factory=dict)
    topology_version: int = -1

    def as_dict(self) -> dict:
        return {
            "phases_active": dict(self.phases_active),
            "retired_count": self.retired_count,
            "reseeded_count": self.reseeded_count,
            "pruned_for_reseed_count": self.pruned_for_reseed_count,
            "safety_fuse_triggers": dict(self.safety_fuse_triggers),
            "topology_version": self.topology_version,
        }


class AlignedBufferRegistry:
    """Single backend-owned choke point for every real (non-observe-only)
    mutation this package performs (deliverable 1: "every aligned
    per-Gaussian buffer is updated through one backend-owned registry").

    Wraps exactly one ``model`` (a ``GsplatGaussianModel``-shaped object
    exposing ``params`` (``nn.ParameterDict`` with keys ``means``, ``sh0``,
    ``shN``, ``opacities``, ``scales``, ``quats``), ``prune_points(mask)``,
    and ``add_points_as_gaussians(...)``) and its paired
    ``LivePopulationState``. Every structural change
    (:meth:`apply_reseed`'s prune+add) touches both through this one
    object in the same call, so there is no window in which the model's
    row count and the state's row count can disagree -- the two calls
    (``model.prune_points`` / ``state.prune_rows``, then
    ``model.add_points_as_gaussians`` / ``state.append_rows``) are made
    back-to-back with no intervening read of either side.

    **``topology_version`` ownership contract (found in review, stated
    explicitly rather than left implicit):** :meth:`apply_reseed` bumps
    ``self.live_state.topology_version`` directly -- this registry is the
    sole authority for that counter's value after any call to
    :meth:`apply`. A caller integrating this registry into a real
    training loop (not done by this PR -- see the W3-10 completion doc)
    **must** read ``registry.live_state.topology_version`` back after
    every :meth:`apply` call and use that as its own subsequent
    ``plan.topology_version``/``sched.topology_version`` going forward,
    rather than incrementing an independently-tracked counter. If a
    caller's own counter and this registry's ever diverge, every
    ``LivePopulationState`` buffer cold-starts back to PROVISIONAL on the
    next ``observe()`` (``state.align()``'s existing, deliberately
    conservative "topology change -> cold start" policy from W3-09,
    unchanged by this PR) -- for the *entire* population, not just the
    rows a reseed actually touched, silently discarding every
    accumulated EMA/support-count/lifecycle-state value. This is why
    Phase C is not yet safe to combine with Phase B in a real run without
    that integration being done correctly first: a reseed's own
    ``topology_version`` bump, if not properly threaded back to whatever
    external state re-drives ``observe()``/``propose()``, would make the
    very state Phase B depends on evaporate on the next observed step.
    """

    def __init__(
        self,
        model: Any,
        live_state: LivePopulationState,
        *,
        max_unreclaimed_retired: int = 0,
    ) -> None:
        self.model = model
        self.live_state = live_state
        # W3-45: Phase B retirement is a one-way capacity leak today --
        # Phase C reseeding has no real candidate producer anywhere in
        # production code (SurfaceSeedRequest is only ever constructed in
        # tests; the sole real call site,
        # utils/strategies/mcmc/live_wrapped.py's step_post_backward, never
        # passes reseed_candidates, so build_mutation_plan always sees an
        # empty tuple). Confirmed on real fr3 GPU runs (docs/architecture/
        # 2026-08-09-w3-44-matched-psnr-comparison.md): retirement executes
        # unconditionally whenever phase_b_retirement is on, regardless of
        # whether anything is actually reclaiming the capacity, producing a
        # durable ~1.3dB test-PSNR cost and permanently dead Gaussians
        # pinned at the opacity floor. This cap makes that leak impossible
        # by construction, independent of the phase_c_reseeding *flag*
        # (which today cannot make a real reseed happen no matter its
        # value): new retirements are refused once the outstanding
        # unreclaimed count (cumulative retired minus cumulative reseeded)
        # would exceed ``max_unreclaimed_retired``. Default 0 means
        # retirement is inert until a real reseed candidate producer exists
        # and starts reclaiming -- see docs/architecture/
        # 2026-08-09-w3-45-phase-c-reseed-root-cause.md for the full trace
        # and why building that producer is out of scope for this fix.
        self._max_unreclaimed_retired = max_unreclaimed_retired
        self._cumulative_retired = 0
        self._cumulative_reseeded = 0
        self._warned_suppressed_retirement = False

    def apply(
        self,
        plan: PopulationMutationPlan,
        *,
        phases: ActivationPhaseConfig,
        iteration: int,
    ) -> ApplyReport:
        """Apply ``plan`` against ``self.model``/``self.live_state``,
        respecting ``phases``. Requires ``plan.topology_version`` to match
        ``self.live_state.topology_version`` (mirrors
        ``build_mutation_plan``'s own ``state.require_valid`` check) --
        applying a plan built against stale state is refused, not
        silently reinterpreted against whatever the current state happens
        to be.
        """
        self.live_state.require_valid(plan.topology_version)

        report = ApplyReport(
            phases_active=phases.as_dict(),
            safety_fuse_triggers=dict(plan.reasons.safety_fuse_triggers),
            topology_version=self.live_state.topology_version,
        )

        if not phases.any_active():
            return report

        if phases.phase_b_retirement and plan.retire_indices.numel() > 0:
            # Clamp at 0: cumulative_reseeded can exceed cumulative_retired
            # for this registry specifically when a row reaches RETIRED via
            # the lifecycle FSM directly (not through this registry's own
            # apply_retire) and is then reseeded -- apply_reseed reuses any
            # currently-RETIRED row, not only ones this registry retired
            # (see test_phase_c_reseed_reuses_retired_capacity_with_zero_momentum,
            # which sets geometry_lifecycle to RETIRED directly). Without
            # this clamp, that case would make ``outstanding`` negative and
            # open unearned retirement headroom.
            outstanding = max(0, self._cumulative_retired - self._cumulative_reseeded)
            room = max(0, self._max_unreclaimed_retired - outstanding)
            n_requested = int(plan.retire_indices.numel())
            n_to_apply = min(n_requested, room)
            if n_to_apply < n_requested:
                n_suppressed = n_requested - n_to_apply
                report.safety_fuse_triggers["safety_fuse:unreclaimed_retirement_cap"] = (
                    report.safety_fuse_triggers.get("safety_fuse:unreclaimed_retirement_cap", 0)
                    + n_suppressed
                )
                if not self._warned_suppressed_retirement:
                    print(
                        f"[AlignedBufferRegistry] WARNING: phase_b_retirement suppressed "
                        f"{n_suppressed} retirement(s) at iteration {iteration} -- Phase C "
                        f"reseeding has no real candidate producer in this codebase today "
                        f"(SurfaceSeedRequest is only ever constructed in tests), so retired "
                        f"capacity is never reclaimed. Retirement is capped at "
                        f"max_unreclaimed_retired={self._max_unreclaimed_retired} outstanding "
                        f"unreclaimed row(s) to prevent the ~1.3dB test-PSNR leak found in "
                        f"docs/architecture/2026-08-09-w3-44-matched-psnr-comparison.md. See "
                        f"docs/architecture/2026-08-09-w3-45-phase-c-reseed-root-cause.md. "
                        f"(This message prints once per run.)"
                    )
                    self._warned_suppressed_retirement = True
            if n_to_apply > 0:
                # W3-55: this prefix-slice keeps the HIGHEST-PRIORITY
                # candidates, not the lowest-physical-index ones, because
                # `build_mutation_plan` (mutation_plan.py) now sorts
                # retire_indices by descending retire_score before handing
                # it off (PopulationMutationPlan carries no separate scores
                # field to sort by here -- see that sort's own comment for
                # why the ordering guarantee has to be established
                # upstream). Currently unreachable in practice:
                # max_unreclaimed_retired defaults to 0, so `room` above is
                # always 0 and n_to_apply == n_requested (no truncation
                # ever occurs) until a real Phase C candidate producer
                # exists (task #47) and this cap is deliberately raised
                # above 0.
                retire_indices = plan.retire_indices[:n_to_apply]
                self.apply_retire(retire_indices)
                report.retired_count = n_to_apply
                self._cumulative_retired += n_to_apply

        if phases.phase_c_reseeding and len(plan.reseed_requests) > 0:
            reseeded, pruned = self.apply_reseed(plan.reseed_requests, iteration=iteration)
            report.reseeded_count = reseeded
            report.pruned_for_reseed_count = pruned
            report.topology_version = self.live_state.topology_version
            self._cumulative_reseeded += reseeded

        return report

    # ------------------------------------------------------------------
    # Phase B
    # ------------------------------------------------------------------
    def apply_retire(self, retire_indices: torch.Tensor) -> None:
        """Force ``retire_indices``' opacity to
        ``RETIRE_OPACITY_LOGIT_FLOOR`` in place, AND zero that row's Adam
        momentum (``exp_avg``/``exp_avg_sq``) on the ``opacities``
        optimizer. Does not change row count/order -- no buffer beyond
        ``opacities`` (param + its own optimizer state) is touched, and no
        ``topology_version`` bump is needed (every index still means the
        same Gaussian it did before this call).

        **Why the momentum zeroing is not optional (found in review):**
        without it, retirement is not durable. Adam moves a parameter from
        *accumulated* momentum even when the current-step gradient is
        zero, and a retired row is still a live tensor row -- still
        rasterized, still reachable by ``opacity_reg`` -- so the next few
        real ``optimizer.step()`` calls would walk a freshly-retired
        opacity back up off the floor, silently un-retiring it. Zeroing
        momentum here (mirroring what ``add_points_as_gaussians`` already
        does for freshly-reseeded rows) is what makes "retired rows become
        inactive" (plan section 15.3) actually hold across subsequent
        training steps, not just at the instant this call returns -- see
        ``test_phase_b_retirement_survives_subsequent_optimizer_steps``.
        """
        if retire_indices.numel() == 0:
            return
        opacities = self.model.params["opacities"]
        idx = retire_indices.to(device=opacities.device, dtype=torch.int64)
        with torch.no_grad():
            opacities.data[idx] = RETIRE_OPACITY_LOGIT_FLOOR
            optimizer = self.model.optimizers.get("opacities") if hasattr(self.model, "optimizers") else None
            if optimizer is not None:
                stored_state = optimizer.state.get(opacities, None)
                if stored_state is not None:
                    for key in ("exp_avg", "exp_avg_sq"):
                        if key in stored_state and isinstance(stored_state[key], torch.Tensor):
                            stored_state[key][idx] = 0.0

    # ------------------------------------------------------------------
    # Phase C
    # ------------------------------------------------------------------
    def apply_reseed(self, reseed_requests: tuple, *, iteration: int) -> tuple[int, int]:
        """Reuse up to ``len(reseed_requests)`` currently-RETIRED rows'
        population capacity. Returns ``(n_added, n_pruned)`` -- these are
        equal by construction (this call always prunes exactly as many
        RETIRED rows as it adds new ones), kept as two return values so a
        caller doesn't have to assume that invariant holds forever.

        **Explicit slot inheritance/reset rule (deliverable 2):** a
        reseeded row does not inherit ANY value from the RETIRED row whose
        capacity it consumed -- the RETIRED row is physically pruned
        first, then a wholly new row is appended via
        ``add_points_as_gaussians`` (fresh optimizer momentum, fresh
        ``LivePopulationState`` fields via ``append_rows``, fresh
        PROVISIONAL lifecycle). There is no code path here that copies an
        old row's optimizer moment, support count, or lifecycle streak
        into a new one.
        """
        if not reseed_requests:
            return 0, 0
        retired_code = state_to_code(GaussianLifecycleState.RETIRED)
        retired_mask = self.live_state.geometry_lifecycle.to(torch.int64) == retired_code
        retired_indices = torch.nonzero(retired_mask, as_tuple=False).flatten()
        n_available = int(retired_indices.numel())
        n_to_reuse = min(len(reseed_requests), n_available)
        if n_to_reuse == 0:
            return 0, 0

        reuse_indices = retired_indices[:n_to_reuse]
        prune_mask = torch.zeros(self.live_state.n, dtype=torch.bool, device=self.live_state.device)
        prune_mask[reuse_indices] = True

        # Prune first: model and state pruned by the identical mask, back
        # to back, so both sides agree on row count/order the instant
        # either one finishes.
        self.model.prune_points(prune_mask)
        self.live_state.prune_rows(~prune_mask)

        requests = reseed_requests[:n_to_reuse]
        device = self.model.params["means"].device
        points = torch.stack(
            [
                r.position.to(device=device, dtype=torch.float32)
                if r.position is not None
                else torch.zeros(3, device=device)
                for r in requests
            ]
        )
        colors = torch.full((n_to_reuse, 3), 0.5, device=device)
        added = self.model.add_points_as_gaussians(
            points, colors, is_provisional=True, birth_frame=int(iteration)
        )
        self.live_state.append_rows(added, iteration=iteration)
        self.live_state.topology_version = self.live_state.topology_version + 1

        return added, n_to_reuse


# ----------------------------------------------------------------------
# Phase E
# ----------------------------------------------------------------------
def constrain_position_delta(
    *,
    delta: torch.Tensor,
    geometry_lifecycle: torch.Tensor,
    normal: Optional[torch.Tensor] = None,
    max_tangent_delta: float = 0.01,
    max_normal_delta: float = 0.002,
) -> torch.Tensor:
    """Clamp a proposed per-Gaussian position delta ``[N, 3]`` by
    lifecycle state (plan section 15.3 Phase E: "Limit supported geometry
    motion according to depth uncertainty and tangent-plane confidence.
    Unrestricted MCMC remains limited to provisional geometry.").

    ``PROVISIONAL`` rows pass through completely unrestricted (matches
    "unrestricted MCMC remains limited to provisional geometry" literally
    -- provisional geometry is exactly where unrestricted movement stays
    allowed). Every other lifecycle state is clamped:

    * with a per-row ``normal`` given, ``delta`` is decomposed into a
      tangent-plane component and a normal component and each is clamped
      independently (``max_normal_delta`` tighter than
      ``max_tangent_delta`` by default -- depth-sensor uncertainty is
      typically larger along the viewing/normal axis than laterally
      across a surface, matching this project's existing depth-noise
      handling elsewhere, e.g. ``streaming_camera_profile``'s per-axis
      free-space handling);
    * without a ``normal``, ``delta`` is clamped isotropically to
      ``max_tangent_delta``.

    Pure function, no model/state mutation -- the caller (a future
    integration into the MCMC noise-injection step, not built by this PR,
    see module docstring) is responsible for actually writing the
    clamped delta back into a position tensor.
    """
    if delta.shape[0] == 0:
        return delta
    provisional_code = state_to_code(GaussianLifecycleState.PROVISIONAL)
    is_provisional = geometry_lifecycle.to(torch.int64) == provisional_code
    restricted = ~is_provisional
    out = delta.clone()
    if not bool(restricted.any()):
        return out

    if normal is not None:
        n = torch.nn.functional.normalize(normal, dim=-1)
        normal_component = (delta * n).sum(-1, keepdim=True) * n
        tangent_component = delta - normal_component
        tangent_norm = tangent_component.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        tangent_scale = (max_tangent_delta / tangent_norm).clamp(max=1.0)
        normal_norm = normal_component.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        normal_scale = (max_normal_delta / normal_norm).clamp(max=1.0)
        clamped = tangent_component * tangent_scale + normal_component * normal_scale
    else:
        delta_norm = delta.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        scale = (max_tangent_delta / delta_norm).clamp(max=1.0)
        clamped = delta * scale

    out[restricted] = clamped[restricted]
    return out
