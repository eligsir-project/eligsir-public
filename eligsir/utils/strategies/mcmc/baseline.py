"""Plain (non energy-guided) gsplat densification strategies:
``GsplatMCMCBaselineStrategy`` (official-like MCMC baseline, legacy-layout
relocation reused via ``ScheduledMCMCStrategy``), ``UpstreamGsplatMCMCStrategy``
(thin wrapper around gsplat's own ``MCMCStrategy``), and ``GsplatDefaultStrategy``
(gradient-based clone/split/prune densification via gsplat's ``DefaultStrategy``).

Moved verbatim out of ``utils/strategies/mcmc_strategy.py`` (Wave 2 Task 4,
2026-08-05) — no behavior change, only module location.
"""

import inspect

import torch

from .base import _effective_cap_max, _exclude_locked_from_mask, _augment_dead_mask_knn_isolation
from .legacy import ScheduledMCMCStrategy


class GsplatMCMCBaselineStrategy(ScheduledMCMCStrategy):
    """Official-like MCMC baseline using opacity relocation, fixed growth, and noise."""

    def __init__(self):
        super().__init__("gsplat_mcmc")

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
        # W3-39: absorb map_backend (and any other future caller-side kwarg)
        # without breaking every non-live strategy's call signature.
        **_w3_39_extra_kwargs,
    ):
        should_log = should_log_strategy or (lambda _iteration: False)
        refine_start = int(getattr(args, "densify_from_iter", 500))
        refine_stop = int(getattr(args, "mcmc_stop_growth_iter", 12_000))
        refine_every = max(1, int(getattr(args, "densification_interval", 100)))
        if not (refine_start <= iteration < refine_stop) or iteration % refine_every != 0:
            return

        with torch.no_grad():
            # No getattr fallback: mcmc_dead_opacity_end is an unconditional
            # OptimizationParams default, always present on the real parsed
            # args (issue #368 -- old getattr(..., 0.01) fallback drifted
            # from the real default of 0.05; removed rather than re-stated).
            min_opacity = float(args.mcmc_dead_opacity_end)
            dead_mask = (gaussians.get_opacity <= min_opacity).squeeze(-1)
            dead_mask = _exclude_locked_from_mask(dead_mask, gaussians, args)
            dead_mask = _augment_dead_mask_knn_isolation(dead_mask, gaussians, args, current_iter=iteration)
            dead_count = int(dead_mask.sum().item())
            gaussians.relocate_gs(dead_mask=dead_mask)
            self._log_reloc(
                tb_writer,
                should_log,
                iteration,
                dead_count,
                {
                    "dead_opacity_threshold": min_opacity,
                    "relocate_interval": refine_every,
                    "rho": sched.get("rho", 0.0),
                },
                gaussians.get_xyz.shape[0],
            )

            before = gaussians.get_xyz.shape[0]
            added = gaussians.add_new_gs(
                cap_max=_effective_cap_max(args),
                growth_factor=1.05,
            )
            self._log_growth(
                tb_writer,
                should_log,
                iteration,
                before,
                gaussians.get_xyz.shape[0],
                added,
                {
                    "growth_factor": 1.05,
                    "grow_interval": refine_every,
                    "rho": sched.get("rho", 0.0),
                },
            )


class UpstreamGsplatMCMCStrategy:
    name = "gsplat_mcmc"

    def __init__(self, *, semantic_cadence: bool = False):
        if type(semantic_cadence) is not bool:
            raise TypeError("semantic_cadence must be exact bool")
        self.semantic_cadence = semantic_cadence
        self.strategy = None
        self.state = None

    def initialize_state(self, gaussians, args, **_kwargs):
        from gsplat.strategy import MCMCStrategy

        # Issue #1259 follow-up (2026-08-31): the canonical/causal_mapping
        # runtime does not drive strategies by a train.py-style fixed
        # iteration count -- `iteration`/`step` here is actually
        # `RuntimeWorkClockSnapshot.optimizer_updates`, a small monotonic
        # counter of optimizer transactions (e.g. 727 total across an entire
        # fr3 mapping phase), not the ~30000-iteration budget these defaults
        # (refine_start_iter=500, refine_every=100) were tuned against.
        # `GsplatEnergyMCMCStrategy` avoids this exact trap via its own
        # `semantic_cadence` flag: when True it disarms its OWN internal
        # iteration gating (refine_start=0, refine_stop=near-infinite,
        # refine_every=1) and defers entirely to `MCMCCadenceOwner`'s
        # pixel-exposure-driven `grow_due`/`relocate_due` schedule, which
        # `production_application.py::_mutate_after_step` already computes
        # and uses to decide whether to invoke this strategy's
        # `step_post_backward` at all. This class was missing that same
        # `semantic_cadence` plumbing entirely -- the registry dispatch for
        # `gsplat_mcmc` never even accepted the kwarg -- so it kept applying
        # its OWN independent iteration-count gate on top of the cadence
        # owner's already-applied gate, and that second gate almost never
        # fired at this runtime's real optimizer-update scale. Measured
        # real-run consequence: ~1,978 Gaussians grown vs. ~459,607 for
        # `gsplat_energy_mcmc` on an identical fr3 recipe -- a growth-
        # starvation artifact of this wiring gap, not a real algorithmic
        # difference between plain MCMC and energy-guided MCMC. Honoring
        # `semantic_cadence` here (mirroring `GsplatEnergyMCMCStrategy`
        # verbatim) makes the two strategies grow under the same external
        # cadence, so a comparison between them actually measures the value
        # of energy guidance rather than an accidental cadence mismatch.
        # Behavior is unchanged for any non-canonical caller (e.g. legacy
        # `train.py`/`train_streaming.py`, which never pass
        # `semantic_cadence=True`): the fixed-iteration defaults below are
        # untouched in that case.
        semantic_cadence = self.semantic_cadence
        requested = {
            "cap_max": int(args.cap_max),
            "noise_lr": float(getattr(args, "noise_lr", 5e5)),
            "refine_start_iter": (
                0 if semantic_cadence else int(getattr(args, "densify_from_iter", 500))
            ),
            # Issue #368's "no getattr fallback, both are unconditional
            # OptimizationParams defaults, always present on the real parsed
            # args" was true for train.py/train_streaming.py's fully-parsed
            # Namespace but is FALSE for the canonical/causal_mapping runtime:
            # its `CanonicalOptimizationPolicy` (eligsir/runtime/
            # config_boundary.py's `mapping_runtime_config_from_args`)
            # deliberately pops mcmc_stop_growth_iter (and mcmc_noise_stop_iter,
            # densify_from_iter/until_iter, etc.) as part of
            # `legacy_iteration_cadence` -- the canonical runtime is
            # iteration-index-agnostic by design (gsplat_energy_mcmc uses
            # pixel/exposure-based cadence instead). Found 2026-08-31 (issue
            # #1259) via a real crash the very first tick of a gsplat_mcmc
            # canonical-runtime run: `AttributeError: mcmc_stop_growth_iter`.
            # Restored getattr fallbacks matching the real argparse defaults
            # (12_000/0.05, same literals issue #368 pinned) so this
            # strategy degrades gracefully under either execution context
            # instead of hard-requiring a field only one of them supplies.
            "refine_stop_iter": (
                2**63 - 1
                if semantic_cadence
                else int(getattr(args, "mcmc_stop_growth_iter", 12_000))
            ),
            "refine_every": (
                1 if semantic_cadence else int(getattr(args, "densification_interval", 100))
            ),
            "min_opacity": float(getattr(args, "mcmc_dead_opacity_end", 0.05)),
            "verbose": bool(getattr(args, "mcmc_strategy_verbose", False)),
            "noise_injection_stop_iter": int(
                getattr(args, "mcmc_noise_stop_iter", getattr(args, "iterations", 30_000))
            ),
        }
        supported = inspect.signature(MCMCStrategy).parameters
        kwargs = {key: value for key, value in requested.items() if key in supported}
        self.strategy = MCMCStrategy(**kwargs)
        self.strategy.check_sanity(gaussians.params, gaussians.optimizers)
        self.state = self.strategy.initialize_state()
        return self.state

    def step_pre_backward(self, **kwargs):
        if self.strategy is not None:
            self.strategy.step_pre_backward(**kwargs)

    def inject_noise(self, **_kwargs):
        # Upstream MCMCStrategy injects noise in step_post_backward.
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
        render_pkg=None,
        lr=None,
        # W3-39: absorb map_backend (and any other future caller-side kwarg)
        # without breaking every non-live strategy's call signature.
        **_w3_39_extra_kwargs,
    ):
        if self.strategy is None or self.state is None:
            self.initialize_state(gaussians=gaussians, args=args)
        if self.semantic_cadence and not bool((sched or {}).get("allow_growth", False)):
            # Issue #1259 follow-up (2026-08-31): a real GPU run of the
            # semantic_cadence fix above hit a SECOND, distinct bug --
            # `ValueError: applied growth had no due exposure debt` from
            # `MCMCCadenceOwner.record_structural_mutation`. Root cause:
            # gsplat's own upstream `MCMCStrategy.step_post_backward` bundles
            # relocation and growth behind ONE combined cadence check
            # (`refine_start_iter`/`refine_stop_iter`/`refine_every`), unlike
            # `GsplatEnergyMCMCStrategy`'s own hand-written step_post_backward,
            # which reads `sched["grow_due"]`/`sched["relocate_due"]`
            # independently and only performs each action when ITS OWN flag
            # says so. Disarming that combined gate (semantic_cadence=True,
            # refine_every=1) to defer to `MCMCCadenceOwner` means EVERY call
            # now runs both relocate AND grow -- so a tick where only
            # relocation was due (schedule["allow_growth"] False,
            # schedule["allow_relocation"] True; the caller invokes this
            # method whenever `allow_growth or allow_relocation`) still grew
            # the population, violating the cadence owner's exposure-debt
            # invariant for growth. Since gsplat's raw strategy offers no way
            # to run relocation without also running growth once its own gate
            # is disarmed, the only correct fix without reaching into
            # upstream gsplat internals is to defer the WHOLE delegated call
            # until growth is genuinely due -- meaning this strategy
            # opportunistically relocates only on ticks that coincide with a
            # due growth event, rather than on every relocation-due tick
            # `GsplatEnergyMCMCStrategy` would act on. This is a real,
            # narrower relocation cadence than the energy-guided strategy's
            # own -- an honest limitation of delegating fully to gsplat's
            # bundled implementation, not something to paper over.
            return
        before = gaussians.get_xyz.shape[0]
        info = render_pkg.get("meta", {}) if render_pkg is not None else {}
        self.strategy.step_post_backward(
            params=gaussians.params,
            optimizers=gaussians.optimizers,
            state=self.state,
            step=iteration,
            info=info,
            lr=float(lr if lr is not None else 0.0),
        )
        after = gaussians.get_xyz.shape[0]
        if tb_writer:
            tb_writer.add_scalar("mcmc/growth_delta_N", after - before, iteration)
        should_log = should_log_strategy or (lambda _iteration: False)
        if should_log(iteration):
            print(
                f"[gsplat-mcmc] iter={iteration} N={before}->{after} "
                f"lr={float(lr if lr is not None else 0.0):.6g}",
                flush=True,
            )


class GsplatDefaultStrategy:
    """Gradient-based densification using gsplat's DefaultStrategy (clone/split/prune)."""

    name = "gsplat_default"

    def __init__(self):
        self.strategy = None
        self.state = None

    def initialize_state(self, gaussians, args, scene_scale=1.0, **_kwargs):
        from gsplat.strategy import DefaultStrategy

        requested = {
            # NOTE (issue #368): unlike this same flag pair's other call
            # sites, this one intentionally keeps a getattr fallback --
            # tests/test_gsplat_default_strategy_reset_every.py's
            # test_reset_every_falls_back_to_3000_when_arg_missing exercises
            # this exact function with a totally empty args object and
            # expects it to degrade gracefully (this strategy variant is
            # constructed via the population-strategy registry, not
            # guaranteed to always see a fully-parsed Namespace the way
            # train.py/train_streaming.py's own strategy call sites are).
            # Literal corrected to match the real default (was 0.005/15_000,
            # real is 0.05/12_000) since updating a *reachable* fallback's
            # literal is the right fix here, not removing it. grow_grad2d/
            # grow_scale3d have no corresponding argparse flag at all
            # (verified against arguments/__init__.py) -- their fallback
            # literal is the only value these ever take; left as-is (#368 triage).
            "prune_opa": float(getattr(args, "mcmc_dead_opacity_end", 0.05)),
            "grow_grad2d": float(getattr(args, "default_grow_grad2d", 0.0002)),
            "grow_scale3d": float(getattr(args, "default_grow_scale3d", 0.01)),
            "refine_start_iter": int(getattr(args, "densify_from_iter", 500)),
            "refine_stop_iter": int(getattr(args, "mcmc_stop_growth_iter", 12_000)),
            "refine_every": int(getattr(args, "densification_interval", 100)),
            # Reuses the existing legacy-3DGS `opacity_reset_interval` flag (default
            # 3000, matching gsplat's own DefaultStrategy default) rather than a new
            # flag, since it's exactly this strategy's own reset-cadence parameter under
            # a different name (`reset_every` in gsplat's API). DefaultStrategy has NO
            # max-Gaussian-count parameter at all (unlike MCMCStrategy's cap_max) --
            # periodic opacity reset is its only self-limiting mechanism (forces weak
            # Gaussians' opacity low so the next prune_opa pass catches them). Before
            # this fix, `reset_every` was hardcoded to 999_999_999 (effectively
            # disabled), removing that limiter entirely -- confirmed live (2026-08-12)
            # resuming a 97862-Gaussian checkpoint with resets off grew unbounded past
            # 400k within the first 11k iterations.
            "reset_every": int(getattr(args, "opacity_reset_interval", 3000)),
            "verbose": bool(getattr(args, "mcmc_strategy_verbose", False)),
        }
        supported = inspect.signature(DefaultStrategy).parameters
        kwargs = {k: v for k, v in requested.items() if k in supported}
        self.strategy = DefaultStrategy(**kwargs)
        self.state = self.strategy.initialize_state(scene_scale=scene_scale)
        return self.state

    def _reset_accum_if_n_changed(self, gaussians):
        """Reset accumulated grad stats when Gaussian count changes (e.g. after depth insertion)."""
        if self.state is None:
            return
        n = gaussians.get_xyz.shape[0]
        for key in ("grad2d", "count", "radii"):
            v = self.state.get(key)
            if v is not None and v.shape[0] != n:
                self.state[key] = None

    def step_pre_backward(self, gaussians=None, args=None, iteration=None, render_pkg=None, **_kwargs):
        if self.strategy is None or self.state is None:
            return
        self._reset_accum_if_n_changed(gaussians)
        info = render_pkg.get("meta", {}) if render_pkg is not None else {}
        self.strategy.step_pre_backward(
            params=gaussians.params,
            optimizers=gaussians.optimizers,
            state=self.state,
            step=iteration,
            info=info,
        )

    def inject_noise(self, **_kwargs):
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
        render_pkg=None,
        lr=None,
        # W3-39: absorb map_backend (and any other future caller-side kwarg)
        # without breaking every non-live strategy's call signature.
        **_w3_39_extra_kwargs,
    ):
        if self.strategy is None or self.state is None:
            self.initialize_state(gaussians=gaussians, args=args)
        self._reset_accum_if_n_changed(gaussians)
        before = gaussians.get_xyz.shape[0]
        info = render_pkg.get("meta", {}) if render_pkg is not None else {}
        self.strategy.step_post_backward(
            params=gaussians.params,
            optimizers=gaussians.optimizers,
            state=self.state,
            step=iteration,
            info=info,
            packed=True,
        )
        after = gaussians.get_xyz.shape[0]
        if tb_writer:
            tb_writer.add_scalar("mcmc/growth_delta_N", after - before, iteration)
        should_log = should_log_strategy or (lambda _iteration: False)
        if should_log(iteration):
            print(f"[gsplat-default] iter={iteration} N={before}->{after}", flush=True)
