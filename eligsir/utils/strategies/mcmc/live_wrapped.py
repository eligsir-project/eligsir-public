"""``gsplat_live_strategy`` -- W3-17 wiring of ``EliGSiRLiveGaussianStrategy``
(Wave 3, W3-09/W3-10, merged) into the real ``--densification_strategy``
dispatch table.

W3-09/W3-10 built ``EliGSiRLiveGaussianStrategy`` as a *shadow observer*:
it never mutates a Gaussian model on its own (see
``eligsir/optimization/gaussian/strategies/live/strategy.py``'s module
docstring) and its ``DryRunReport``/overlap telemetry is only meaningful
*next to* a real, currently-running strategy (see
``eligsir/optimization/gaussian/strategies/gsplat_mcmc_adapter.py``'s
``shadow_dead_mask``). There is therefore no such thing as running the live
strategy "instead of" a real strategy -- ``LiveWrappedMCMCStrategy`` below
wraps a fixed, real base strategy (``GsplatEnergyMCMCStrategy``, the
project default -- not configurable in this PR, wiring-only per the W3-17
brief) and runs the live strategy's ``observe()``/``propose()`` alongside
it, purely as additional telemetry, on every ``step_post_backward`` call.

**Observe-only (all ``ActivationPhaseConfig`` phases off, the only
combination this PR's own verification run exercises):** the live strategy
never touches the model. The wrapped base strategy performs 100% of the
real growth/relocation/noise mutation, identically to selecting
``--densification_strategy gsplat_energy_mcmc`` directly -- this is what
makes ``gsplat_live_strategy`` "behavior-neutral" per the brief's
deliverable 3. Achieving that neutrality required adding
``"gsplat_live_strategy"`` alongside ``"gsplat_energy_mcmc"`` at every
``densification_strategy in {...}`` gate in ``train.py``/
``train_streaming.py`` that flips ``use_energy_mcmc``/noise-injection/
growth-aware-compile behavior -- these gates dispatch on the *name string*,
not on the strategy object, so wrapping the object alone is not sufficient
for neutrality. See ``docs/architecture/2026-08-09-w3-17-live-strategy-wiring.md``
for the exact list, the paired-verification evidence, and an open finding
about ``topology_version`` (see this module's docstring further down)
that bears on how task #22's population-policy ablation should read this
strategy's retirement/reseed telemetry.

**With one or more phases on:** ``AlignedBufferRegistry.apply()``
(``eligsir/optimization/gaussian/strategies/live/backend.py``, W3-10)
is additionally invoked with the live strategy's own proposed plan, using
the live strategy's own ``LivePopulationState`` as the backing store for
lifecycle bookkeeping. This is real, but **not verified by this PR beyond
unit-test scope** -- W3-10's own docstring already flags that Phase C
(reseeding) bumps ``live_state.topology_version`` independently of the
rest of the engine's own topology tracking (``map_backend.topology_version``
elsewhere in ``train_streaming.py``), and this PR does not thread that
back. Per the W3-17 brief (scope item 4), no phase is enabled by default
and no phase's real-training behavior is exercised by this PR's
verification -- that is explicitly deferred to task #22.

**W3-39 (2026-08-09) closed the ``topology_version``-pinned-at-0 gap flagged
above.** ``step_post_backward`` now accepts an optional ``map_backend``
kwarg; when the streaming call site (``train_streaming.py``) passes the
real ``GsplatMapBackend``, this module reads its real, live
``topology_version`` (no longer permanently 0) and additionally derives a
``pure_append_growth`` signal from ``map_backend.cumulative_inserted``/
``cumulative_pruned`` deltas since the previous call, forwarded into
``EliGSiRLiveGaussianStrategy.observe()`` -> ``LivePopulationState.align(
..., pure_append=...)``. When true, ``LivePopulationState`` extends its
buffers for the newly-appended rows only (``grow_append()``) instead of
cold-starting the entire population -- this is what lets retirement/
recovery evidence survive an ordinary Phase-2 depth-insertion event instead
of restarting every ~50 iterations (W3-41's measured churn pattern). The
offline (``train.py``) call site has no ``map_backend`` at all
(``map_backend=None`` default) and falls back to a weaker, ``n``-only
growth signal -- see ``docs/architecture/2026-08-09-w3-39-topology-version-growth-fix.md``
for the full design, the offline-path safety argument, and honest
remaining gaps.
"""

from __future__ import annotations

from typing import Any, Optional

import torch

from .energy import GsplatEnergyMCMCStrategy

name = "gsplat_live_strategy"


class LiveWrappedMCMCStrategy:
    """Delegates all real mutation to a fixed ``GsplatEnergyMCMCStrategy``
    instance; additionally drives ``EliGSiRLiveGaussianStrategy``'s
    observe/propose telemetry (and, only if a phase is active, its
    ``AlignedBufferRegistry``-mediated mutation) on every
    ``step_post_backward`` call. See module docstring.
    """

    name = "gsplat_live_strategy"

    def __init__(
        self,
        phases: Optional[Any] = None,
        *,
        semantic_cadence: bool = False,
    ) -> None:
        # Local imports: keep this file's import surface off the hot path
        # for every other strategy selection, and keep the live-strategy
        # package import confined to this one wrapper module (see
        # tests/test_w3_09_live_strategy_registry.py's
        # test_utils_strategies_does_not_import_live_package -- that test
        # only forbids the import from eligsir.utils.strategies/utils.strategies.mcmc_strategy,
        # not from this dispatch-only wrapper module).
        from eligsir.optimization.gaussian.strategies.live.backend import (
            ActivationPhaseConfig,
            AlignedBufferRegistry,
        )
        from eligsir.optimization.gaussian.strategies.live.strategy import (
            EliGSiRLiveGaussianStrategy,
        )

        self._base = GsplatEnergyMCMCStrategy(
            semantic_cadence=semantic_cadence
        )
        self._live = EliGSiRLiveGaussianStrategy()
        self._phases = phases if phases is not None else ActivationPhaseConfig()
        self._registry_cls = AlignedBufferRegistry
        self._registry: Optional[Any] = None
        # W3-41: AlignedBufferRegistry.apply() computes retired_count/reseeded_count/
        # pruned_for_reseed_count fresh on EVERY call (every training iteration when a
        # phase is active), but this wrapper previously only wrote that value to
        # TensorBoard when should_log(iteration) was true (every strategy_log_interval
        # iterations, default 500) -- an instantaneous per-step snapshot sampled at a
        # sparse interval, not an accumulator. Real, nonzero retirement/reseed events
        # that don't happen to land exactly on a logged iteration were silently never
        # recorded (confirmed 2026-08-09: 1742/4000 real per-step calls had a nonzero
        # retired_count, summing to 571,636 total retirements over the run, yet every
        # one of the 8 iterations that are exact multiples of 500 read exactly zero --
        # see docs/architecture/2026-08-09-w3-41-apply-mutation-audit.md). Fix: accumulate
        # every call's counts here and log the accumulated total for the just-elapsed
        # logging window, then reset -- this changes only what gets logged/how often,
        # never apply()'s own mutation behavior.
        self._apply_accum_retired = 0
        self._apply_accum_reseeded = 0
        self._apply_accum_pruned_for_reseed = 0

        # W3-39: distinguish "pure append growth happened since the last
        # call" (safe to preserve LivePopulationState's existing evidence
        # via LivePopulationState.grow_append) from "a structural change
        # (prune/reseed) happened" (must still cold-start) from "nothing
        # changed". See this module's docstring further down and
        # docs/architecture/2026-08-09-w3-39-topology-version-growth-fix.md.
        # None sentinels mean "no prior call yet" -- the very first
        # observe() always cold-starts regardless (LivePopulationState has
        # no buffers yet), so pure_append_growth=False on that call is
        # correct, not a missed case.
        self._w3_39_last_cum_inserted: Optional[int] = None
        self._w3_39_last_cum_pruned: Optional[int] = None
        self._w3_39_last_seen_n: Optional[int] = None

    # ------------------------------------------------------------------
    # Pass-through surface -- identical to GsplatEnergyMCMCStrategy's own,
    # so `mcmc_strategy = make_mcmc_strategy("gsplat_live_strategy", ...)`
    # is a drop-in replacement at every existing `mcmc_strategy.<method>(...)`
    # call site in train.py/train_streaming.py.
    # ------------------------------------------------------------------
    def initialize_state(self, gaussians, args, **kwargs):
        return self._base.initialize_state(gaussians, args, **kwargs)

    def step_pre_backward(self, **kwargs):
        self._base.step_pre_backward(**kwargs)

    def inject_noise(self, **kwargs):
        self._base.inject_noise(**kwargs)

    def step_post_backward(
        self,
        gaussians,
        args,
        sched,
        iteration,
        utility=None,
        temperature=1.0,
        use_energy_mcmc=True,
        tb_writer=None,
        should_log_strategy=None,
        render_pkg=None,
        lr=None,
        map_backend=None,
        viewpoint_camera=None,
        gt_depth=None,
        **kwargs,
    ):
        # 1. The real strategy: 100% identical to selecting gsplat_energy_mcmc
        #    directly. Runs first so the live strategy's shadow observation
        #    below reflects the same post-mutation state a plain
        #    gsplat_energy_mcmc run would have produced by this point.
        self._base.step_post_backward(
            gaussians,
            args,
            sched,
            iteration,
            utility=utility,
            temperature=temperature,
            use_energy_mcmc=use_energy_mcmc,
            tb_writer=tb_writer,
            should_log_strategy=should_log_strategy,
            render_pkg=render_pkg,
            lr=lr,
        )

        # 2. Shadow observe/propose -- read-only against `gaussians` (see
        #    live/strategy.py's own guarantee), writes only to the live
        #    strategy's own LivePopulationState/telemetry.
        try:
            n = int(gaussians.get_xyz.shape[0])
        except Exception:
            return

        # W3-39: real topology_version (was permanently pinned at 0 via
        # getattr(sched, "topology_version", 0) -- sched never had that
        # attribute) plus the pure-append-growth signal, computed below.
        if map_backend is not None:
            topology_version = int(map_backend.topology_version)
            cur_inserted = int(getattr(map_backend, "cumulative_inserted", 0))
            cur_pruned = int(getattr(map_backend, "cumulative_pruned", 0))
            pure_append_growth = False
            if self._w3_39_last_cum_inserted is not None:
                inserted_since = cur_inserted - self._w3_39_last_cum_inserted
                pruned_since = cur_pruned - self._w3_39_last_cum_pruned
                # Real inserted/pruned split (approach (a) from the W3-39
                # brief): pure append iff something was inserted AND
                # nothing was pruned since the last call. A prune-then-
                # larger-insert window (net growth but a real structural
                # change) correctly falls through to cold-start here,
                # because pruned_since > 0.
                pure_append_growth = inserted_since > 0 and pruned_since == 0
            self._w3_39_last_cum_inserted = cur_inserted
            self._w3_39_last_cum_pruned = cur_pruned
        else:
            # Offline (train.py) fallback: no map_backend/TopologyChange
            # split available at all. Weaker signal -- "n grew since the
            # last call" is treated as a pure-append candidate. Verified
            # (see the doc referenced above) that GsplatEnergyMCMCStrategy
            # (the only base strategy this wrapper ever runs) never shrinks
            # the population in step_post_backward -- relocation reuses
            # dead rows in place (no prune_points call), growth only
            # appends -- so the "prune then larger insert" misclassification
            # this fallback cannot rule out on its own does not arise in
            # practice for this wrapper's only real caller. Still strictly
            # safer than the old code, which cold-started on every n change.
            topology_version = getattr(sched, "topology_version", 0) if sched is not None else 0
            pure_append_growth = self._w3_39_last_seen_n is not None and n > self._w3_39_last_seen_n
        self._w3_39_last_seen_n = n

        visible_mask = torch.ones(n, dtype=torch.bool, device=gaussians.get_xyz.device)
        self._live.observe(
            n_gaussians=n,
            iteration=int(iteration),
            topology_version=int(topology_version),
            visible_mask=visible_mask,
            device=gaussians.get_xyz.device,
            pure_append_growth=pure_append_growth,
        )

        # W3-47 (2026-08-10): real Phase C reseed candidates, derived from
        # this view's depth/alpha (see reseed_producer.py's module
        # docstring for the full design/scope). Only computed when Phase C
        # is actually enabled AND the caller supplied both
        # viewpoint_camera/gt_depth -- a no-op (empty tuple, matching the
        # pre-W3-47 default) otherwise, so this never costs anything for a
        # run that isn't using this phase or hasn't threaded the optional
        # kwargs through (e.g. train.py's offline call site).
        reseed_candidates = ()
        if self._phases.phase_c_reseeding and viewpoint_camera is not None and gt_depth is not None:
            try:
                from eligsir.optimization.gaussian.strategies.live.reseed_producer import (
                    derive_reseed_candidates,
                )

                alpha = render_pkg.get("alpha") if render_pkg is not None else None
                c2w = viewpoint_camera.world_view_transform.inverse().transpose(0, 1)
                reseed_candidates = derive_reseed_candidates(
                    gt_depth=gt_depth,
                    alpha=alpha,
                    fx=float(viewpoint_camera.fx),
                    fy=float(viewpoint_camera.fy),
                    cx=float(viewpoint_camera.cx),
                    cy=float(viewpoint_camera.cy),
                    c2w=c2w,
                )
            except Exception:
                # Never let a reseed-candidate derivation failure interrupt
                # the real strategy step above (which has already run) --
                # this is additional telemetry/mutation input, not core
                # training. Falls back to the pre-W3-47 empty-tuple default.
                reseed_candidates = ()

        plan, report = self._live.propose(
            iteration=int(iteration),
            topology_version=int(topology_version),
            shadow_gaussians=gaussians,
            shadow_args=args,
            reseed_candidates=reseed_candidates,
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

        # 3. Activation phases (all default off -- see arguments/__init__.py's
        #    streaming_live_strategy_phase_* flags). Not exercised by this
        #    PR's verification run; see module docstring's caveat.
        if self._phases.any_active():
            if self._registry is None:
                self._registry = self._registry_cls(model=gaussians, live_state=self._live.state)
            apply_report = self._registry.apply(plan, phases=self._phases, iteration=int(iteration))
            # W3-41: accumulate every real per-call count, regardless of TB logging
            # cadence -- see the accumulator fields' docstring in __init__ for why an
            # instantaneous per-call snapshot at should_log(iteration)'s sparse interval
            # (default every 500 iterations) silently missed real, nonzero events.
            self._apply_accum_retired += apply_report.retired_count
            self._apply_accum_reseeded += apply_report.reseeded_count
            self._apply_accum_pruned_for_reseed += apply_report.pruned_for_reseed_count
            if tb_writer is not None and should_log(iteration):
                prefix = "live_strategy_apply"
                # Log the accumulated total for the just-elapsed logging window (since
                # the previous should_log(iteration) hit, or since the start of the run
                # for the first one), not just this single call's instantaneous count --
                # this is what makes the value trustworthy evidence of what actually
                # happened between samples.
                tb_writer.add_scalar(f"{prefix}/retired_count", self._apply_accum_retired, iteration)
                tb_writer.add_scalar(f"{prefix}/reseeded_count", self._apply_accum_reseeded, iteration)
                tb_writer.add_scalar(
                    f"{prefix}/pruned_for_reseed_count", self._apply_accum_pruned_for_reseed, iteration
                )
                self._apply_accum_retired = 0
                self._apply_accum_reseeded = 0
                self._apply_accum_pruned_for_reseed = 0


def make_live_wrapped_strategy(
    gaussians=None,
    args=None,
    *,
    semantic_cadence: bool = False,
) -> LiveWrappedMCMCStrategy:
    """Factory used by ``utils.strategies.mcmc.registry.make_mcmc_strategy``.
    ``gaussians`` is accepted (unused) only to keep the same call signature
    every other strategy-name branch in that dispatcher uses.
    """
    from eligsir.optimization.gaussian.strategies.live.backend import ActivationPhaseConfig

    phases = ActivationPhaseConfig(
        phase_a_lifecycle=bool(getattr(args, "streaming_live_strategy_phase_a_lifecycle", False)),
        phase_b_retirement=bool(getattr(args, "streaming_live_strategy_phase_b_retirement", False)),
        phase_c_reseeding=bool(getattr(args, "streaming_live_strategy_phase_c_reseeding", False)),
        phase_d_cell_budgets=bool(getattr(args, "streaming_live_strategy_phase_d_cell_budgets", False)),
        phase_e_constrained_movement=bool(
            getattr(args, "streaming_live_strategy_phase_e_constrained_movement", False)
        ),
    )
    return LiveWrappedMCMCStrategy(
        phases=phases,
        semantic_cadence=semantic_cadence,
    )
