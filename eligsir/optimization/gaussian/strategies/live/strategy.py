"""``EliGSiRLiveGaussianStrategy`` (Wave 3, PR W3-09, plan section 14).

Observe-only / dry-run mode: this class **proposes** a typed
:class:`~eligsir.optimization.gaussian.strategies.contracts.PopulationMutationPlan`
and never mutates a Gaussian model's tensors. Grep-checkable guarantee (see
``tests/test_w3_09_live_strategy_observe_only.py``): no method on this
class, nor anything it calls in ``eligsir/optimization/gaussian/strategies/live/``,
references ``relocate_gs``, ``add_new_gs``, ``prune``, or any in-place
tensor-mutating assignment onto a passed-in ``gaussians`` object.

It exposes the same ``step_pre_backward``/``step_post_backward`` call
surface as the existing gsplat MCMC strategies (see
``utils/strategies/mcmc/baseline.py``) so it is structurally *pluggable* at
the same call site.

**Updated by W3-17 (2026-08), not reverted:** this class itself is
unchanged (still strictly observe-only -- see the class docstring below and
``tests/test_w3_09_live_strategy_observe_only.py``). What changed is that it
is now *reachable* from a real training run: W3-17 added
``"gsplat_live_strategy"`` to ``eligsir.mapping.strategies.registry.STRATEGIES``
and to ``utils.strategies.mcmc.registry.make_mcmc_strategy``'s dispatch,
via a new wrapper (``utils/strategies/mcmc/live_wrapped.py``,
``LiveWrappedMCMCStrategy``) that runs this class's ``observe()``/
``propose()`` as a shadow observer *alongside* a real, fixed
``GsplatEnergyMCMCStrategy`` instance -- this class still never mutates a
Gaussian model on its own; the wrapper's real mutation is 100% delegated
to that fixed base strategy in the default (all activation phases off)
case. ``train.py``/``train_streaming.py``/``engine.py`` are still not
directly touched by this class's own symbols (see
``tests/test_w3_09_live_strategy_observe_only.py``'s
``test_engine_and_train_call_sites_are_untouched_by_this_pr``, whose scope
this indirection intentionally narrows -- see that test's updated
docstring) -- the sole integration point is the wrapper module above.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import torch

from eligsir.optimization.gaussian.strategies.contracts import (
    DryRunReport,
    PopulationMutationPlan,
    SurfaceSeedRequest,
)
from eligsir.optimization.gaussian.strategies.gsplat_mcmc_adapter import overlap_report, shadow_dead_mask
from eligsir.optimization.gaussian.strategies.live.cell_budget import CellBudgetTracker
from eligsir.optimization.gaussian.strategies.live.lifecycle import LifecycleThresholds
from eligsir.optimization.gaussian.strategies.live.mutation_plan import SafetyFuses, build_mutation_plan
from eligsir.optimization.gaussian.strategies.live.state import LivePopulationState


@dataclass
class LiveStrategyConfig:
    thresholds: LifecycleThresholds = field(default_factory=LifecycleThresholds)
    fuses: SafetyFuses = field(default_factory=SafetyFuses)
    cell_budget: CellBudgetTracker = field(default_factory=CellBudgetTracker)
    ema_decay: float = 0.9
    free_space_min_views: int = 3


class EliGSiRLiveGaussianStrategy:
    """Observe-only live population strategy. See module docstring.

    ``observe_only`` is not a constructor parameter -- there is nothing to
    toggle. Every plan this class produces is built through
    :class:`~eligsir.optimization.gaussian.strategies.contracts.PopulationMutationPlan`,
    whose ``__post_init__`` refuses ``observe_only=False`` outright.
    """

    def __init__(self, config: Optional[LiveStrategyConfig] = None) -> None:
        self.config = config or LiveStrategyConfig()
        self.state = LivePopulationState()
        self._last_plan: Optional[PopulationMutationPlan] = None
        self._last_report: Optional[DryRunReport] = None

    # ------------------------------------------------------------------
    # Core observe-only API
    # ------------------------------------------------------------------
    def observe(
        self,
        *,
        n_gaussians: int,
        iteration: int,
        topology_version: int,
        visible_mask: torch.Tensor,
        device: Optional[torch.device] = None,
        contribution: Optional[torch.Tensor] = None,
        rgb_residual: Optional[torch.Tensor] = None,
        depth_residual: Optional[torch.Tensor] = None,
        pos_grad: Optional[torch.Tensor] = None,
        opacity_grad: Optional[torch.Tensor] = None,
        free_space_conflict: Optional[torch.Tensor] = None,
        geometry_confidence: Optional[torch.Tensor] = None,
        pure_append_growth: bool = False,
    ) -> None:
        """Fold one step's per-Gaussian observation into internal EMA state.
        Never touches the passed-in gaussians/model -- writes only to
        ``self.state`` (this strategy's own bookkeeping).

        ``pure_append_growth`` (W3-39): pass ``True`` when the caller can
        assert the only structural change since the previous ``observe()``
        call was a pure append (rows added at the end, nothing pruned or
        reindexed) -- see ``LivePopulationState.align``'s ``pure_append``
        kwarg, which this forwards to verbatim. Defaults to ``False``,
        reproducing the pre-W3-39 cold-start-on-any-change behavior for
        any caller that doesn't (or can't) distinguish pure append from a
        structural change.
        """
        self.state.align(
            n_gaussians, topology_version, iteration=iteration, device=device, pure_append=pure_append_growth
        )
        self.state.update_from_observation(
            visible_mask=visible_mask,
            contribution=contribution,
            rgb_residual=rgb_residual,
            depth_residual=depth_residual,
            pos_grad=pos_grad,
            opacity_grad=opacity_grad,
            free_space_conflict=free_space_conflict,
            geometry_confidence=geometry_confidence,
            iteration=iteration,
            decay=self.config.ema_decay,
        )

    def propose(
        self,
        *,
        iteration: int,
        topology_version: int,
        reseed_candidates: tuple[SurfaceSeedRequest, ...] = (),
        growth_candidates: tuple[SurfaceSeedRequest, ...] = (),
        shadow_gaussians: Any = None,
        shadow_args: Any = None,
    ) -> tuple[PopulationMutationPlan, DryRunReport]:
        """Build ``(plan, report)`` from the current observed evidence, then
        persist the strategy's *own* lifecycle bookkeeping (never a
        gaussian model tensor) so it actually advances across steps:
        ``self.state.geometry_lifecycle``/``appearance_lifecycle`` get the
        proposed transitions written in, and the streak counters feeding
        "persistent"/"sustained" evidence rules advance. This is the one
        place this class writes to ``self.state`` -- see
        ``live/state.py``'s ``apply_lifecycle_batch``/``persist_streaks``
        docstrings for why this is not a map mutation.
        """
        plan, streaks = build_mutation_plan(
            state=self.state,
            thresholds=self.config.thresholds,
            fuses=self.config.fuses,
            iteration=iteration,
            topology_version=topology_version,
            reseed_candidates=reseed_candidates,
            growth_candidates=growth_candidates,
            cell_budget=self.config.cell_budget,
        )
        self.state.apply_lifecycle_batch(plan.lifecycle_updates)
        self.state.persist_streaks(
            geometry_suspect_streak=streaks["geometry_suspect_streak"],
            geometry_recover_streak=streaks["geometry_recover_streak"],
            geometry_stable_streak=streaks["geometry_stable_streak"],
        )

        overlap: dict = {}
        if shadow_gaussians is not None and shadow_args is not None:
            shadow_mask = shadow_dead_mask(shadow_gaussians, shadow_args)
            all_proposed = torch.cat([plan.retire_indices, plan.prune_indices]) if (
                plan.retire_indices.numel() or plan.prune_indices.numel()
            ) else torch.empty(0, dtype=torch.int64)
            overlap = overlap_report(all_proposed, shadow_mask)

        report = DryRunReport(
            proposed_retire_count=int(plan.retire_indices.numel()),
            proposed_prune_count=int(plan.prune_indices.numel()),
            proposed_reseed_count=len(plan.reseed_requests),
            proposed_growth_count=len(plan.growth_requests),
            reasons=plan.reasons,
            overlap_with_current_mcmc=overlap,
            predicted_affected_cells=CellBudgetTracker.affected_cell_count(
                plan.reseed_requests + plan.growth_requests
            ),
            expected_contribution_change=0.0,
            safety_fuse_trigger_counts=dict(plan.reasons.safety_fuse_triggers),
        )
        self._last_plan = plan
        self._last_report = report
        return plan, report

    # ------------------------------------------------------------------
    # gsplat-compatible call surface (not wired into any call site by this
    # PR -- see module docstring). Inert: logs only, never mutates.
    # ------------------------------------------------------------------
    def step_pre_backward(self, **_kwargs) -> None:
        return None

    def step_post_backward(
        self,
        gaussians,
        args,
        sched,
        iteration,
        utility=None,
        temperature=1.0,
        use_energy_mcmc=False,
        tb_writer=None,
        should_log_strategy=None,
        **_kwargs,
    ) -> None:
        """Inert observation hook: computes and logs a
        :class:`PopulationMutationPlan`/:class:`DryRunReport` but performs
        **no** action against ``gaussians`` -- no relocation, no growth, no
        pruning, no noise injection. Every branch below is either a read of
        ``gaussians`` (for shape/opacity, to size/align ``self.state``) or a
        ``tb_writer.add_scalar`` call.
        """
        try:
            n = int(gaussians.get_xyz.shape[0])
        except Exception:
            return
        topology_version = getattr(sched, "topology_version", 0) if sched is not None else 0
        visible_mask = torch.ones(n, dtype=torch.bool, device=gaussians.get_xyz.device)
        self.observe(
            n_gaussians=n,
            iteration=int(iteration),
            topology_version=int(topology_version),
            visible_mask=visible_mask,
            device=gaussians.get_xyz.device,
        )
        plan, report = self.propose(
            iteration=int(iteration),
            topology_version=int(topology_version),
            shadow_gaussians=gaussians,
            shadow_args=args,
        )
        should_log = should_log_strategy or (lambda _iteration: False)
        if tb_writer is not None and should_log(iteration):
            prefix = "live_strategy_observe"
            tb_writer.add_scalar(f"{prefix}/proposed_retire_count", report.proposed_retire_count, iteration)
            tb_writer.add_scalar(f"{prefix}/proposed_prune_count", report.proposed_prune_count, iteration)
            tb_writer.add_scalar(f"{prefix}/proposed_reseed_count", report.proposed_reseed_count, iteration)
            tb_writer.add_scalar(f"{prefix}/proposed_growth_count", report.proposed_growth_count, iteration)
            tb_writer.add_scalar(
                f"{prefix}/overlap_with_current_mcmc",
                report.overlap_with_current_mcmc.get("overlap_count", 0),
                iteration,
            )
        return None
