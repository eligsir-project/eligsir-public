"""``GsplatEnergyMCMCStrategy`` — energy-guided MCMC using gsplat's
ParameterDict and optimizer-dict contract. The project default strategy.

Moved out of ``utils/strategies/mcmc_strategy.py`` (Wave 2 Task 4,
2026-08-05). The birth/relocation proposal generation previously implemented
as instance methods on this class now lives in ``.proposal`` as free
functions (see that module's docstring) and is called directly here; no
behavior change.
"""

import inspect

import torch

from eligsir.checkpointing.streaming_resume import _clone_state

from .base import _bubble_active_mask, _effective_cap_max, _exclude_locked_from_mask, _gate_dead_mask_to_visible, _locked_depth_mask, _augment_dead_mask_knn_isolation
from .proposal import (
    add_energy_guided,
    append_running_state,
    energy_weights,
    relocate_energy_guided,
    reset_relocated_state,
    sample_legacy_clamped_noise,
)


def _resume_counter(value, name, *, nullable):
    if value is None and nullable:
        return None
    if type(value) is not int or value < 0:
        suffix = " or None" if nullable else ""
        raise ValueError(f"{name} must be a nonnegative integer{suffix}")
    return value


def _noise_stop_from_policy(args, *, semantic_cadence: bool) -> int:
    """Resolve the noise horizon on the clock selected at composition.

    Canonical live mapping passes an exact typed strategy-construction mode and
    supplies an optimizer-update horizon. The legacy iteration option remains
    available only to the offline-compatible strategy boundary.
    """

    if type(semantic_cadence) is not bool:
        raise TypeError("semantic_cadence must be exact bool")
    if semantic_cadence:
        value = args.mcmc_noise_stop_optimizer_updates
        name = "mcmc_noise_stop_optimizer_updates"
    else:
        value = getattr(
            args,
            "mcmc_noise_stop_iter",
            getattr(args, "iterations", 30_000),
        )
        name = "mcmc_noise_stop_iter"
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative exact integer")
    return value


def _position_noise_allowed(
    sched,
    *,
    optimizer_updates: int,
    noise_stop_iter: int | None,
) -> bool:
    """Resolve phase authority before the optimizer-update noise horizon.

    Canonical live composition supplies ``allow_position_noise``.  Direct and
    offline callers that do not have an explicit mapping/refinement phase keep
    the historical optimizer-update horizon behavior.
    """

    phase_allowed = sched.get("allow_position_noise", True)
    if type(phase_allowed) is not bool:
        raise TypeError("allow_position_noise must be exact bool")
    horizon_allowed = (
        noise_stop_iter is None or int(optimizer_updates) <= noise_stop_iter
    )
    return phase_allowed and horizon_allowed


def _compatible_resume_state(saved, initialized, path="MCMC state"):
    """Validate saved state against initialized structure before replacing it.

    Tensor shape and dtype must match. Saved tensors are copied onto the
    initialized tensor's device, which is the safe equivalent of requiring
    device equality after ``torch.load`` remapping.
    """
    if isinstance(initialized, torch.Tensor):
        if not isinstance(saved, torch.Tensor):
            raise ValueError(f"{path} tensor is missing")
        if (
            saved.shape != initialized.shape
            or saved.dtype != initialized.dtype
            or saved.layout != initialized.layout
        ):
            raise ValueError(f"{path} tensor shape, dtype, or layout mismatch")
        return saved.detach().to(device=initialized.device).clone()
    if isinstance(initialized, dict):
        if not isinstance(saved, dict) or set(saved) != set(initialized):
            raise ValueError(f"{path} key mismatch")
        return {
            key: _compatible_resume_state(saved[key], initialized[key], f"{path}.{key}")
            for key in initialized
        }
    if isinstance(initialized, list):
        if not isinstance(saved, list) or len(saved) != len(initialized):
            raise ValueError(f"{path} list mismatch")
        return [
            _compatible_resume_state(saved_item, current_item, f"{path}[{index}]")
            for index, (saved_item, current_item) in enumerate(zip(saved, initialized))
        ]
    if isinstance(initialized, tuple):
        if not isinstance(saved, tuple) or len(saved) != len(initialized):
            raise ValueError(f"{path} tuple mismatch")
        return tuple(
            _compatible_resume_state(saved_item, current_item, f"{path}[{index}]")
            for index, (saved_item, current_item) in enumerate(zip(saved, initialized))
        )
    if type(saved) is not type(initialized):
        raise ValueError(f"{path} scalar type mismatch")
    return _clone_state(saved)


class GsplatEnergyMCMCStrategy:
    """Energy-guided MCMC using gsplat's ParameterDict and optimizer-dict contract."""

    name = "gsplat_energy_mcmc"

    def __init__(self, *, semantic_cadence: bool = False):
        if type(semantic_cadence) is not bool:
            raise TypeError("semantic_cadence must be exact bool")
        self.semantic_cadence = semantic_cadence
        self.strategy = None
        self.state = None
        self.noise_stop_iter = None
        self._next_reloc_iter = None
        # issue #1259 C1: settling-fence counter for
        # streaming_post_mapping_mature_mobility. Deliberately a plain
        # instance attribute, NOT a key in self.state -- self.state is the
        # gsplat MCMCStrategy state dict, which is checkpointed and validated
        # on resume by _compatible_resume_state (tensor shape/dtype checks);
        # an unexpected non-tensor key there would either break that
        # validation or be silently dropped. Losing this counter across a
        # resume only means C1 re-applies its settling fence once more after
        # a resume, which is safe (conservative, not incorrect).
        self._post_mapping_updates_since_source_complete = 0
        self._next_grow_iter = None

    def initialize_state(self, gaussians, args, **_kwargs):
        from gsplat.strategy import MCMCStrategy

        semantic_cadence = self.semantic_cadence
        requested = {
            "cap_max": int(args.cap_max),
            "noise_lr": float(getattr(args, "noise_lr", 5e5)),
            "refine_start_iter": (
                0 if semantic_cadence else int(getattr(args, "densify_from_iter", 500))
            ),
            "refine_stop_iter": (
                2**63 - 1
                if semantic_cadence
                else int(args.mcmc_stop_growth_iter)
            ),
            "refine_every": (
                1
                if semantic_cadence
                else int(getattr(args, "densification_interval", 100))
            ),
            "min_opacity": float(args.mcmc_dead_opacity_end),
            "verbose": bool(getattr(args, "mcmc_strategy_verbose", False)),
            "noise_injection_stop_iter": _noise_stop_from_policy(
                args, semantic_cadence=semantic_cadence
            ),
        }
        supported = inspect.signature(MCMCStrategy).parameters
        kwargs = {key: value for key, value in requested.items() if key in supported}
        self.strategy = MCMCStrategy(**kwargs)
        self.noise_stop_iter = requested["noise_injection_stop_iter"]
        self.strategy.check_sanity(gaussians.params, gaussians.optimizers)
        self.state = self.strategy.initialize_state()
        return self.state

    def resume_state_dict(self):
        if self.strategy is None or self.state is None:
            raise ValueError("energy MCMC state is not initialized")
        return {
            "owner_schema": "gsplat_energy_mcmc_v1",
            "state": _clone_state(self.state),
            "noise_stop_iter": _resume_counter(
                self.noise_stop_iter, "noise_stop_iter", nullable=False
            ),
            "next_reloc_iter": _resume_counter(
                self._next_reloc_iter, "next_reloc_iter", nullable=True
            ),
            "next_grow_iter": _resume_counter(
                self._next_grow_iter, "next_grow_iter", nullable=True
            ),
        }

    def load_resume_state_dict(self, payload, resolver=None):
        del resolver
        if self.strategy is None:
            raise ValueError("energy MCMC strategy must be initialized before restore")
        if not isinstance(payload, dict) or payload.get("owner_schema") != "gsplat_energy_mcmc_v1":
            raise ValueError("incompatible energy MCMC resume state")
        required = {
            "owner_schema", "state", "noise_stop_iter",
            "next_reloc_iter", "next_grow_iter",
        }
        if set(payload) != required:
            raise ValueError("incomplete energy MCMC resume state")
        if not isinstance(self.state, dict):
            raise ValueError("initialized MCMC state is unavailable")

        restored_state = _compatible_resume_state(payload["state"], self.state)
        noise_stop_iter = _resume_counter(
            payload["noise_stop_iter"], "noise_stop_iter", nullable=False
        )
        next_reloc_iter = _resume_counter(
            payload["next_reloc_iter"], "next_reloc_iter", nullable=True
        )
        next_grow_iter = _resume_counter(
            payload["next_grow_iter"], "next_grow_iter", nullable=True
        )

        self.state = restored_state
        self.noise_stop_iter = noise_stop_iter
        self._next_reloc_iter = next_reloc_iter
        self._next_grow_iter = next_grow_iter

    def step_pre_backward(self, **kwargs):
        if self.strategy is not None and hasattr(self.strategy, "step_pre_backward"):
            self.strategy.step_pre_backward(**kwargs)

    def inject_noise(self, **_kwargs):
        # Like upstream MCMCStrategy, this strategy injects noise in step_post_backward.
        return None

    def _due(self, name, iteration, interval, start_iter):
        attr = f"_next_{name}_iter"
        next_iter = getattr(self, attr, None)
        if next_iter is None:
            next_iter = int(start_iter) + max(1, int(interval))
            setattr(self, attr, next_iter)
        if int(iteration) < next_iter:
            return False
        setattr(self, attr, int(iteration) + max(1, int(interval)))
        return True

    def _mutation_due(
        self,
        name,
        sched,
        *,
        optimizer_updates,
        interval,
        start_update,
    ):
        semantic_key = "relocate_due" if name == "reloc" else f"{name}_due"
        if semantic_key in sched:
            due = sched[semantic_key]
            if type(due) is not bool:
                raise TypeError(f"{semantic_key} must be bool")
            return due
        return self._due(name, optimizer_updates, interval, start_update)

    def step_post_backward(
        self,
        gaussians,
        args,
        sched,
        iteration,
        optimizer_updates=None,
        utility=None,
        temperature=1.0,
        use_energy_mcmc=True,
        tb_writer=None,
        should_log_strategy=None,
        render_pkg=None,
        lr=None,
        # W3-39: absorb map_backend (and any other future caller-side kwarg)
        # without breaking every non-live strategy's call signature.
        **_w3_39_extra_kwargs,
    ):
        optimizer_updates = (
            int(iteration) if optimizer_updates is None else int(optimizer_updates)
        )
        if self.strategy is None or self.state is None:
            self.initialize_state(gaussians=gaussians, args=args)

        from eligsir.utils.seed_lineage_freeze import assert_frozen_seed_integrity
        assert_frozen_seed_integrity(
            gaussians,
            args,
            iteration=optimizer_updates,
            context="mcmc_pre_mutation",
        )

        params = gaussians.params
        optimizers = gaussians.optimizers
        self.state["binoms"] = self.state["binoms"].to(params["means"].device)
        binoms = self.state["binoms"]
        before = params["means"].shape[0]
        n_relocated = 0
        n_added = 0
        position_noise_allowed = _position_noise_allowed(
            sched,
            optimizer_updates=optimizer_updates,
            noise_stop_iter=self.noise_stop_iter,
        )
        # issue #1259 C1 arm: decouple lifecycle CLASSIFICATION (confidence/
        # stability, unchanged) from MCMC MOBILITY POLICY (whether MCMC may
        # perturb a Gaussian's position) once mapping has genuinely ended.
        # MATURE remains a real, protective state during live mapping
        # (relocation-target exclusion stays untouched -- see the comment at
        # that site below, this is deliberately NOT C1's concern) but C1
        # tests whether MATURE geometry should regain ordinary MCMC position
        # noise during the *unconstrained refinement* phase that
        # source_complete is already supposed to open (per production_
        # application.py's own "the finite source boundary explicitly opens
        # the unconstrained refinement phase" comment) -- currently it does
        # not, because the noise-revert gate below is keyed on lifecycle_state
        # alone, independent of source_complete.
        #
        # sched["allow_position_noise"] is set boolean-identical to
        # source_complete by the canonical live composer
        # (ProductionApplication._compose_mcmc_schedule) -- reusing it here
        # avoids a second source_complete plumbing path.
        #
        # Sequencing: PR #1313 (relaxed replay-depth insertion post-mapping)
        # must be allowed to finish placing its depth anchors before MCMC
        # starts moving MATURE geometry, or MCMC would immediately be
        # re-disrupted by a topology change it hasn't seen yet. Rather than a
        # second cross-module dependency on #1313's own completion signal,
        # this uses an EXPOSURE-based settling fence: count successful
        # optimizer updates since source_complete first became true (tracked
        # in a plain instance attribute, NOT self.state (see __init__'s
        # comment on _post_mapping_updates_since_source_complete for why))
        # and only relax MATURE noise-reversion once that count exceeds
        # streaming_post_mapping_mature_settle_updates (default 16, within
        # the specified 8-32 range) -- comfortably past the point where any
        # pending post-mapping depth-insertion work has already drained,
        # since draining happens before/at the same source_complete
        # transition, not after it.
        #
        # Off by default (flag defaults False): byte-identical behavior when
        # unset. C1 touches ONLY position-noise reversion (see the
        # noise-revert site below) -- relocation-target eligibility is
        # untouched, and no C2-style relocation relaxation is implemented by
        # this flag.
        _m1_c1_enabled = bool(
            getattr(args, "streaming_post_mapping_mature_mobility", False)
        )
        _source_complete = bool(sched.get("allow_position_noise", False))
        if _source_complete:
            _updates_since_source_complete = (
                int(self._post_mapping_updates_since_source_complete) + 1
            )
        else:
            _updates_since_source_complete = 0
        self._post_mapping_updates_since_source_complete = _updates_since_source_complete
        _settle_updates = max(
            0, int(getattr(args, "streaming_post_mapping_mature_settle_updates", 16))
        )
        _m1_relax_mature = (
            _m1_c1_enabled
            and _source_complete
            and _updates_since_source_complete > _settle_updates
        )

        with torch.no_grad():
            if sched["allow_relocation"] and self._mutation_due(
                "reloc",
                sched,
                optimizer_updates=optimizer_updates,
                interval=sched["relocate_interval"],
                start_update=(
                    0
                    if "relocate_due" in sched
                    else getattr(args, "densify_from_iter", 500)
                ),
            ):
                if use_energy_mcmc and utility is not None:
                    from eligsir.utils.gaussian_energy_policy import GaussianEnergyPolicy, energy_config_from_args
                    # issue #385: was a hand-listed 9-field subset that pinned five
                    # of the nine fields compute_legacy_dead_mask() actually reads
                    # (support_threshold, scale_moderate_opacity_mult,
                    # scale_dead_support_mult, utility_quantile,
                    # use_utility_quantile) at their dataclass defaults. alpha_dead
                    # stays schedule-driven -- NOT args.energy_alpha_dead.
                    _pcfg = energy_config_from_args(
                        args, alpha_dead=sched["dead_opacity_threshold"]
                    )
                    _policy = GaussianEnergyPolicy(_pcfg)
                    # Phase 4: energy-based dead-mask routing
                    _energy_dead_mode = getattr(args, "energy_dead_mode", "legacy_clauses")
                    if _energy_dead_mode in ("energy_quantile", "energy_sigmoid"):
                        _dead_terms = _policy.compute_terms(
                            gaussians, render_pkg or {}, optimizer_updates
                        )
                        _pop_energy = _policy.compute_population_energy(_dead_terms)
                        # issue #1262 §4 item 2: unexposed rows must leave the
                        # QUANTILE STATISTIC POOL as well, not just the dead mask --
                        # otherwise they drag the threshold and change which EXPOSED
                        # Gaussians die. Same precedent as the bubble's active_mask.
                        from eligsir.utils.resolution_exposure import unexposed_statistic_pool_mask
                        dead_mask = _policy.compute_dead_mask_from_energy(
                            _pop_energy, gaussians,
                            active_mask=unexposed_statistic_pool_mask(
                                gaussians, args, iteration=optimizer_updates,
                                base=_bubble_active_mask(gaussians.get_xyz, args),
                            ),
                        )
                        if tb_writer:
                            tb_writer.add_scalar("energy/dead_threshold", float(torch.quantile(_pop_energy.total, _pcfg.energy_dead_quantile)) if _energy_dead_mode == "energy_quantile" else _pop_energy.population_mean, iteration)
                            tb_writer.add_scalar("energy/dead_energy_mean", float(_pop_energy.total[dead_mask].mean()) if dead_mask.any() else 0.0, iteration)
                            tb_writer.add_scalar("energy/alive_energy_mean", float(_pop_energy.total[~dead_mask].mean()) if (~dead_mask).any() else 0.0, iteration)
                    else:
                        from eligsir.utils.resolution_exposure import unexposed_statistic_pool_mask
                        _young_prot = int(args.streaming_young_prot_iters)
                        _dead_result = _policy.compute_legacy_dead_mask(
                            gaussians, utility=utility,
                            current_iter=optimizer_updates,
                            young_prot_iters=_young_prot,
                            active_mask=unexposed_statistic_pool_mask(
                                gaussians, args, iteration=optimizer_updates,
                                base=(_bubble_active_mask(gaussians.get_xyz, args)
                                      if getattr(args, "streaming_local_stats", False) else None),
                            ),
                        )
                        dead_mask = _dead_result.mask
                        if tb_writer:
                            for clause, cnt in _dead_result.stats.items():
                                tb_writer.add_scalar(f"mcmc/dead_{clause}", cnt, iteration)
                else:
                    dead_mask = (
                        torch.sigmoid(params["opacities"].flatten())
                        <= sched["dead_opacity_threshold"]
                    )
                # issue #1365 (formerly #1372 Phase 7, superseded): purely observational candidate-survival
                # funnel snapshot -- captured BEFORE any of the exclusion gates
                # below run, so it can report what each gate blocks/keeps.
                # None (and zero extra compute) unless the feature flag is on
                # AND this optimizer update falls on the sample interval.
                from eligsir.utils.mcmc_candidate_funnel import should_sample as _funnel_should_sample
                _funnel_raw_dead_mask = (
                    dead_mask.clone() if _funnel_should_sample(args, optimizer_updates) else None
                )
                dead_mask = _exclude_locked_from_mask(dead_mask, gaussians, args)
                # issue #1262 category (a). Deliberately applied BEFORE the KNN
                # isolation augment below, which ADDS kills: isolation- and
                # depth-conflict-based kills are "other real evidence" and must NOT be
                # suppressed by an absence-of-evidence veto. No-op when the flag is off.
                from eligsir.utils.resolution_exposure import exclude_unexposed_mask
                dead_mask = exclude_unexposed_mask(
                    gaussians, args, dead_mask,
                    iteration=optimizer_updates, role="relocation_target",
                )
                dead_mask = _augment_dead_mask_knn_isolation(
                    dead_mask,
                    gaussians,
                    args,
                    current_iter=optimizer_updates,
                )
                dead_mask = _gate_dead_mask_to_visible(dead_mask, render_pkg, args)

                # Per-Gaussian MCMC quiescence (closed local budget): never relocate
                # MATURE/FROZEN geometry. Once a region has matured (frame-age +
                # utility), teleporting it away is the "vanish at source / floater
                # outward" half of the oscillation limit cycle. Gate by the lifecycle
                # so settled geometry is preserved while young geometry still churns.
                # issue #1259: relocation-target eligibility is DELIBERATELY left
                # untouched by the C1 arm below -- MATURE/FROZEN geometry remains a
                # non-target/non-donor even after source_complete (the anti-forgetting
                # mechanism this exclusion protects is real; only the noise-reversion
                # gate, further below, is relaxed by C1). A future C2 arm would gate
                # relocation eligibility on real evidence (residual/conflict +
                # exposure), never on this blanket flag alone -- not implemented here.
                _reloc_mature_mask = None
                if args.streaming_lifecycle_enabled and hasattr(gaussians, "lifecycle_state"):
                    _lc = gaussians.lifecycle_state
                    if _lc.shape[0] == dead_mask.shape[0]:
                        _reloc_mature_mask = _lc >= 2  # MATURE or FROZEN
                        dead_mask = dead_mask & (_lc < 2)  # keep only PROVISIONAL/YOUNG

                # Spatial active bubble (Component 1): never relocate OUT-OF-BUBBLE
                # geometry. As the robot travels away, out-of-bubble Gaussians get
                # zero supervision, their opacity drifts down, they are flagged dead
                # and relocated AWAY — the travel-induced "forgetting after ~iter
                # 6000". Freezing them out of relocation preserves the map until the
                # robot returns. No-op when the bubble is disabled. FREEZE
                # machinery: keyed on streaming_spatial_bubble_enabled ONLY
                # (stats-only runs must not gate relocation of dead mass).
                _bubble = (
                    _bubble_active_mask(gaussians.get_xyz, args)
                    if getattr(args, "streaming_spatial_bubble_enabled", False) else None
                )
                if _bubble is not None and _bubble.shape[0] == dead_mask.shape[0]:
                    dead_mask = dead_mask & _bubble

                # Phase 10: soft lifecycle modulation of parent scores for relocation
                _reloc_utility = utility if use_energy_mcmc else None
                if getattr(args, "soft_lifecycle_enabled", False) and use_energy_mcmc and utility is not None:
                    from eligsir.utils.gaussian_energy_policy import GaussianEnergyPolicy, energy_config_from_args
                    # issue #385: soft_lifecycle_support_k / soft_lifecycle_age_k
                    # (the only two fields compute_maturity_logit reads) were
                    # pinned here; they now come from args. alpha_dead stays
                    # schedule-driven.
                    _lc_policy = GaussianEnergyPolicy(energy_config_from_args(
                        args, alpha_dead=sched["dead_opacity_threshold"],
                    ))
                    _lc_terms = _lc_policy.compute_terms(
                        gaussians, render_pkg or {}, optimizer_updates
                    )
                    _maturity_prob = _lc_policy.compute_maturity_logit(_lc_terms).sigmoid().detach()
                    _reloc_utility = utility * _maturity_prob

                n_relocated = relocate_energy_guided(
                    gaussians=gaussians,
                    args=args,
                    dead_mask=dead_mask,
                    utility=_reloc_utility,
                    temperature=temperature,
                    binoms=binoms,
                    min_opacity=max(float(sched["dead_opacity_threshold"]), self.strategy.min_opacity),
                    iteration=optimizer_updates,
                )

                if _funnel_raw_dead_mask is not None:
                    from eligsir.utils.mcmc_candidate_funnel import (
                        compute_relocation_funnel as _compute_reloc_funnel,
                        log_to_tensorboard as _funnel_log_tb,
                        record_sample as _funnel_record,
                    )
                    _reloc_funnel = _compute_reloc_funnel(
                        gaussians, args, optimizer_updates,
                        raw_candidate_mask=_funnel_raw_dead_mask,
                        render_pkg=render_pkg,
                    )
                    _funnel_log_tb(tb_writer, iteration, _reloc_funnel, None)
                    _funnel_record(_reloc_funnel, None)

            if sched["allow_growth"] and self._mutation_due(
                "grow",
                sched,
                optimizer_updates=optimizer_updates,
                interval=sched["grow_interval"],
                start_update=(
                    0
                    if "grow_due" in sched
                    else getattr(args, "densify_from_iter", 500)
                ),
            ):
                # Phase 10: soft lifecycle modulation of parent scores for growth
                _grow_utility = utility if use_energy_mcmc else None
                if getattr(args, "soft_lifecycle_enabled", False) and use_energy_mcmc and utility is not None:
                    from eligsir.utils.gaussian_energy_policy import GaussianEnergyPolicy, energy_config_from_args
                    # issue #385: see the relocation-path sibling above.
                    _grow_lc_policy = GaussianEnergyPolicy(energy_config_from_args(
                        args, alpha_dead=sched["dead_opacity_threshold"],
                    ))
                    _grow_lc_terms = _grow_lc_policy.compute_terms(
                        gaussians, render_pkg or {}, optimizer_updates
                    )
                    _grow_maturity = _grow_lc_policy.compute_maturity_logit(_grow_lc_terms).sigmoid().detach()
                    _grow_utility = utility * _grow_maturity
                # issue #1365 (formerly #1372 Phase 7, superseded): see the relocation-path sibling above.
                # MUST run BEFORE add_energy_guided grows the population --
                # every read-only gate predicate below sizes itself against
                # gaussians.get_xyz.shape[0], which add_energy_guided extends.
                from eligsir.utils.mcmc_candidate_funnel import should_sample as _funnel_should_sample_grow
                if _funnel_should_sample_grow(args, optimizer_updates):
                    from eligsir.utils.mcmc_candidate_funnel import (
                        compute_growth_funnel as _compute_growth_funnel,
                        log_to_tensorboard as _funnel_log_tb_grow,
                        record_sample as _funnel_record_grow,
                    )
                    _growth_funnel = _compute_growth_funnel(
                        gaussians, args, optimizer_updates,
                        current_n=int(params["means"].shape[0]),
                    )
                    _funnel_log_tb_grow(tb_writer, iteration, None, _growth_funnel)
                    _funnel_record_grow(None, _growth_funnel)
                n_added = add_energy_guided(
                    gaussians=gaussians,
                    args=args,
                    utility=_grow_utility,
                    temperature=temperature,
                    binoms=binoms,
                    min_opacity=self.strategy.min_opacity,
                    cap_max=_effective_cap_max(args),
                    growth_factor=sched["growth_factor"],
                    iteration=optimizer_updates,
                )

        if n_relocated > 0 or n_added > 0:
            # WS-D5: this used to call torch.cuda.empty_cache() unconditionally on
            # every growth/relocation event. That risks allocator churn/fragmentation
            # (returning freed blocks to the driver only to immediately re-request
            # similarly-sized blocks next event) with no benchmark in this codebase
            # showing it prevents OOM/fragmentation here -- growth/relocation
            # intervals (sched["grow_interval"], etc.) can fire frequently early in
            # training. No existing cadence-gating signal in this class was a clean
            # fit to reuse (unlike e.g. streaming_global_maintenance_interval, which
            # is a StreamingScene-level concept this MCMC strategy doesn't see), so
            # per review guidance the safe default is: don't call it here at all: PyTorch's
            # caching allocator already reuses freed blocks for same-sized future
            # allocations, which is exactly what growth/relocation needs. If a real
            # OOM/fragmentation problem is observed, catch torch.cuda.OutOfMemoryError
            # at the call site and empty_cache() there -- a narrow, justified trigger --
            # rather than reintroducing this unconditional call.
            pass

        if position_noise_allowed:
            # Per-Gaussian MCMC quiescence: settled (MATURE/FROZEN) geometry receives
            # NO position noise. Capture mature positions before injection and restore
            # after, so all three noise paths are covered without per-path edits. The
            # population count is stable here (relocation/growth already ran above).
            #
            # issue #1259 C1: this lifecycle-driven freeze is the specific
            # mechanism under test -- it is independent of source_complete, so
            # MATURE geometry stays frozen through the whole refinement tail
            # today regardless of the "unconstrained refinement after mapping"
            # framing. When _m1_relax_mature is true (flag set, source_complete,
            # and the exposure-based settling fence above has elapsed), MATURE
            # is excluded from the freeze mask here so it receives ordinary
            # MCMC position noise again. Relocation-target eligibility (above)
            # is NOT touched by this -- MATURE/FROZEN geometry still cannot be
            # relocated, only its noise-quiescence is relaxed.
            _q_lifecycle_mature_mask = None
            _q_mature_mask = None
            _q_mature_means = None
            if args.streaming_lifecycle_enabled and hasattr(gaussians, "lifecycle_state"):
                _q_lc = gaussians.lifecycle_state
                if _q_lc.shape[0] == params["means"].shape[0]:
                    _q_lifecycle_mature_mask = _q_lc >= 2  # MATURE or FROZEN
                    if not _m1_relax_mature:
                        _q_mature_mask = _q_lifecycle_mature_mask
            if _q_lifecycle_mature_mask is not None:
                # Direct "how much are we blocking" telemetry: how many MATURE
                # rows would have had their MCMC noise reverted this call,
                # whether or not C1 is currently relaxing that reversion.
                #
                # issue #1259: the canonical causal_mapping production path
                # (ProductionApplication._compose_mcmc_schedule's caller)
                # invokes this strategy with tb_writer=None and
                # should_log_strategy=lambda _iteration: False unconditionally
                # (eligsir/runtime/production_application.py) -- a
                # pre-existing gap that already makes every other tb_writer
                # scalar in this function (mcmc/dead_count, mcmc/noise_step_*,
                # etc.) inert on that path, not something newly introduced
                # here. Fixing that wiring is out of scope for this change.
                # To make THIS telemetry observable on the path this A/B
                # actually exercises, emit an unconditional, cadence-gated
                # plain print in addition to the tb_writer scalar (which still
                # fires correctly on train.py's legacy, non-canonical path).
                _mature_frac = float(_q_lifecycle_mature_mask.float().mean().item())
                _mature_count = int(_q_lifecycle_mature_mask.sum().item())
                if tb_writer is not None:
                    tb_writer.add_scalar("mcmc/mature_would_freeze_count", _mature_count, iteration)
                    tb_writer.add_scalar("mcmc/mature_fraction", _mature_frac, iteration)
                    tb_writer.add_scalar(
                        "mcmc/post_mapping_mature_mobility_active", int(_m1_relax_mature), iteration
                    )
                # issue #1259: print unconditionally, gated only on a per-call
                # counter -- NOT on `optimizer_updates % scalar_log_interval`.
                # This block only executes when the caller's own
                # allow_growth/allow_relocation gate is due (mcmc_grow/
                # relocate_interval, ~every 50-2000 updates), so an
                # iteration-modulo gate can silently alias to zero prints for
                # an entire run depending on phase (the exact W3-41 class of
                # bug CLAUDE.md documents for TB-interval sampling). A
                # per-call counter cannot alias this way.
                self._c1_log_calls = getattr(self, "_c1_log_calls", 0) + 1
                print(
                        f"[issue1259-c1] iter={iteration} mature_frac={_mature_frac:.4f} "
                        f"mature_would_freeze_count={_mature_count} "
                        f"source_complete={_source_complete} "
                        f"post_mapping_mature_mobility_active={int(_m1_relax_mature)} "
                        f"updates_since_source_complete={_updates_since_source_complete}",
                        flush=True,
                    )
            # Spatial active bubble (Component 1): out-of-bubble geometry is also
            # frozen from position noise this window (union with the mature mask),
            # so the origin can't drift/vanish while the robot works elsewhere.
            # FREEZE machinery: keyed on streaming_spatial_bubble_enabled ONLY.
            _q_bubble = (
                _bubble_active_mask(params["means"], args)
                if getattr(args, "streaming_spatial_bubble_enabled", False) else None
            )
            if _q_bubble is not None and _q_bubble.shape[0] == params["means"].shape[0]:
                _q_out_of_bubble = ~_q_bubble
                _q_mature_mask = (
                    _q_out_of_bubble if _q_mature_mask is None
                    else (_q_mature_mask | _q_out_of_bubble)
                )
            if _q_mature_mask is not None:
                if bool(_q_mature_mask.any()):
                    _q_mature_means = params["means"].data[_q_mature_mask].clone()
                else:
                    _q_mature_mask = None
            _m1_proposed_noise = None
            _proposal_mode = getattr(args, "mcmc_proposal_mode", "legacy_clamped")
            if _proposal_mode == "trust_region" and lr is not None:
                # Phase 6: trust-region MCMC proposal for gsplat_energy_mcmc strategy
                from eligsir.utils.gaussian_energy_policy import GaussianEnergyPolicy, energy_config_from_args
                # issue #385: alpha_dead is unread by sample_trust_region_noise
                # (it reads only the three mcmc_* fields), but the factory requires
                # it explicitly so no site can silently pick the wrong source --
                # this is an args-driven site, so args.energy_alpha_dead it is.
                # mcmc_proposal_mode is forced because this branch is already
                # gated on args.mcmc_proposal_mode == "trust_region".
                _tr_cfg = energy_config_from_args(
                    args,
                    alpha_dead=float(args.energy_alpha_dead),
                    mcmc_proposal_mode="trust_region",
                )
                _tr_policy = GaussianEnergyPolicy(_tr_cfg)
                _visible_idx = torch.arange(params["means"].shape[0], device=params["means"].device)
                _all_noise = _tr_policy.sample_trust_region_noise(
                    gaussians,
                    xyz_lr=float(lr),
                    noise_lr=float(args.noise_lr),
                    visible=_visible_idx,
                )
                from eligsir.utils.seed_lineage_freeze import zero_frozen_seed_position_noise
                _all_noise = zero_frozen_seed_position_noise(
                    gaussians, args, _all_noise, iteration=optimizer_updates
                )
                with torch.no_grad():
                    params["means"].data.add_(_all_noise)
                if tb_writer and optimizer_updates % max(1, int(getattr(args, "scalar_log_interval", 100))) == 0:
                    _step_norms = _all_noise.norm(dim=-1)
                    tb_writer.add_scalar("mcmc/noise_step_mean", float(_step_norms.mean()), iteration)
                    tb_writer.add_scalar("mcmc/noise_step_p95", float(torch.quantile(_step_norms, 0.95)), iteration)
                    tb_writer.add_scalar("mcmc/noise_step_max", float(_step_norms.max()), iteration)
            elif _proposal_mode == "surface_manifold" and lr is not None:
                # Finding C: route surface-manifold proposals through the
                # gsplat_energy_mcmc path too (previously only ScheduledMCMCStrategy
                # dispatched it, so the default strategy silently ignored it).
                from eligsir.utils.strategies.surface_mcmc_strategy import sample_surface_manifold_noise
                _noise_idx = torch.arange(params["means"].shape[0], device=params["means"].device)
                _all_noise = sample_surface_manifold_noise(
                    gaussians,
                    noise_idx=_noise_idx,
                    xyz_lr=float(lr),
                    noise_lr=float(args.noise_lr),
                    normal_scale=float(getattr(args, "surface_mcmc_normal_scale", 0.15)),
                    tangent_scale=float(getattr(args, "surface_mcmc_tangent_scale", 1.0)),
                    provisional_normal_scale=float(getattr(args, "surface_mcmc_provisional_normal_scale", 0.75)),
                )
                from eligsir.utils.seed_lineage_freeze import zero_frozen_seed_position_noise
                _all_noise = zero_frozen_seed_position_noise(
                    gaussians, args, _all_noise, iteration=optimizer_updates
                )
                with torch.no_grad():
                    params["means"].data.add_(_all_noise)
                if tb_writer and optimizer_updates % max(1, int(getattr(args, "scalar_log_interval", 100))) == 0:
                    _step_norms = _all_noise.norm(dim=-1)
                    tb_writer.add_scalar("mcmc/surface_noise_step_mean", float(_step_norms.mean()), iteration)
                    tb_writer.add_scalar("mcmc/surface_noise_step_p95", float(torch.quantile(_step_norms, 0.95)), iteration)
            else:
                # Default ("legacy_clamped") proposal path.
                #
                # issue: this used to call gsplat's stock, UNCLAMPED
                # gsplat.strategy.ops.inject_noise_to_position() directly --
                # despite the "clamped" name of this proposal mode, that
                # upstream function applies neither a per-Gaussian scale cap
                # nor an absolute step-size cap before scaling per-Gaussian
                # noise by `noise_lr * xyz_lr * covariance`. The sibling
                # legacy/mcmc strategy (utils/strategies/mcmc/legacy.py)
                # already has exactly this protection via its own
                # noise_max_scale/noise_max_step args, but it was never
                # connected here, on the actual project-default strategy
                # (gsplat_energy_mcmc) -- which gets its OWN separate
                # gsplat_mcmc_noise_max_scale/gsplat_mcmc_noise_max_step args
                # below (not a reuse of the legacy ones -- a real GPU A/B
                # found the legacy strategy's 0.01m step cap over-constrains
                # this strategy's much larger noise_lr, see
                # .proposal.sample_legacy_clamped_noise's docstring). A
                # newly-grown/relocated Gaussian
                # transiently below the op_sigmoid near-zero-opacity gate could
                # therefore draw an unbounded single-step position jump (e.g. a
                # rare large-tail randn draw combined with an uncapped per-
                # Gaussian scale), flinging it tens-to-hundreds of metres outside
                # the scene with no recovery path once outside every camera's
                # view frustum (root-caused via a real Replica room0 run:
                # scene_bbox_volume_ratio=173x trajectory bbox, outliers traced
                # via birth_iter/birth_method to MCMC growth/relocation clones
                # a few hundred iterations old, NOT to scene init).
                #
                # This mirrors utils/strategies/mcmc/legacy.py's own clamped
                # noise path exactly (same op_sigmoid gate, same noise_lr
                # scaling, same two clamps) rather than gsplat's unclamped op.
                # See .proposal.sample_legacy_clamped_noise's docstring for
                # the full root-cause writeup.
                noise = sample_legacy_clamped_noise(
                    gaussians, params, args, lr, self.strategy.noise_lr,
                )
                from eligsir.utils.seed_lineage_freeze import zero_frozen_seed_position_noise
                noise = zero_frozen_seed_position_noise(
                    gaussians, args, noise, iteration=optimizer_updates
                )

                with torch.no_grad():
                    params["means"].data.add_(noise)
                _m1_proposed_noise = noise
                if tb_writer and optimizer_updates % max(1, int(getattr(args, "scalar_log_interval", 100))) == 0:
                    _step_norms = noise.norm(dim=-1)
                    tb_writer.add_scalar("mcmc/noise_step_mean", float(_step_norms.mean()), iteration)
                    tb_writer.add_scalar("mcmc/noise_step_p95", float(torch.quantile(_step_norms, 0.95)), iteration)
                    tb_writer.add_scalar("mcmc/noise_step_max", float(_step_norms.max()), iteration)

            # Restore settled (MATURE/FROZEN) positions — quiescence for the closed budget.
            if _q_mature_means is not None and _q_mature_mask is not None:
                if _q_mature_mask.shape[0] == params["means"].shape[0]:
                    with torch.no_grad():
                        params["means"].data[_q_mature_mask] = _q_mature_means

            # issue #1259 telemetry: the "how much are we blocking" / "is MCMC
            # actually doing anything" numbers requested for the C1 A/B.
            # Best-effort -- only the legacy_clamped proposal path (the
            # project default, and the only one exercised by any campaign
            # recipe this issue's runs use) reports the per-Gaussian noise
            # tensor here; trust_region/surface_manifold report their own
            # noise_step_* scalars above but are not wired into
            # f_effective_mcmc.
            #
            # As above: emit an unconditional, cadence-gated plain print in
            # addition to tb_writer (which is hardcoded None on the canonical
            # causal_mapping path -- see the comment at the mature_fraction
            # site above), so this is observable on the path the A/B for this
            # issue actually runs.
            if (
                _m1_proposed_noise is not None
                and _m1_proposed_noise.shape[0] == params["means"].shape[0]
            ):
                _eff_noise = _m1_proposed_noise
                if _q_mature_mask is not None and _q_mature_mask.shape[0] == _eff_noise.shape[0]:
                    _eff_noise = _eff_noise.clone()
                    _eff_noise[_q_mature_mask] = 0.0
                _eff_mag = _eff_noise.norm(dim=-1)
                _eps = 1e-6
                _moved = _eff_mag > _eps
                _f_effective_mcmc = float(_moved.float().mean().item())
                _mov_young = _mov_mature = None
                if _q_lifecycle_mature_mask is not None and _q_lifecycle_mature_mask.shape[0] == _eff_mag.shape[0]:
                    _young_mask = ~_q_lifecycle_mature_mask
                    if bool(_young_mask.any()):
                        _mov_young = float(_eff_mag[_young_mask].mean().item())
                    if bool(_q_lifecycle_mature_mask.any()):
                        _mov_mature = float(_eff_mag[_q_lifecycle_mature_mask].mean().item())
                if tb_writer is not None:
                    tb_writer.add_scalar("mcmc/f_effective_mcmc", _f_effective_mcmc, iteration)
                    if _mov_young is not None:
                        tb_writer.add_scalar("mcmc/movement_mag_mean_young", _mov_young, iteration)
                    if _mov_mature is not None:
                        tb_writer.add_scalar("mcmc/movement_mag_mean_mature", _mov_mature, iteration)
                # issue #1259: unconditional per-call print, same reasoning as
                # the mature_fraction print above -- not gated on
                # optimizer_updates modulo scalar_log_interval.
                print(
                    f"[issue1259-c1] iter={iteration} f_effective_mcmc={_f_effective_mcmc:.4f} "
                    f"movement_mag_mean_young={_mov_young} movement_mag_mean_mature={_mov_mature}",
                    flush=True,
                )

        after = params["means"].shape[0]
        from eligsir.utils.gaussian_lineage import assert_lineage_consistent
        assert_lineage_consistent(
            gaussians, context=f"step_post_backward@update={optimizer_updates}"
        )
        assert_frozen_seed_integrity(
            gaussians,
            args,
            iteration=optimizer_updates,
            context="mcmc_post_mutation",
        )
        if tb_writer:
            tb_writer.add_scalar("mcmc/dead_count", n_relocated, iteration)
            tb_writer.add_scalar("mcmc/added_count", n_added, iteration)
            tb_writer.add_scalar("mcmc/growth_delta_N", after - before, iteration)
            tb_writer.add_scalar("mcmc/energy_guided", int(use_energy_mcmc and utility is not None), iteration)

        should_log = should_log_strategy or (lambda _iteration: False)
        if should_log(iteration):
            mode = "energy" if use_energy_mcmc and utility is not None else "opacity"
            print(
                f"[gsplat-energy-mcmc] iter={iteration} mode={mode} "
                f"relocated={n_relocated} added={n_added} "
                f"factor={sched['growth_factor']:.4f} grow_int={sched['grow_interval']} "
                f"N={before}->{after} lr={float(lr if lr is not None else 0.0):.6g}",
                flush=True,
            )
        return {
            "relocated": int(n_relocated),
            "added": int(n_added),
            "content_changed": bool(
                n_relocated > 0
                or n_added > 0
                or position_noise_allowed
            ),
        }

    # -- Backward-compatible instance-method aliases -------------------------
    # These were the original implementations before Wave 2 Task 4 moved the
    # actual logic to free functions in `.proposal` (they read/wrote no state
    # on `self` beyond their explicit arguments, so the move is behavior
    # identical). Kept as thin delegating methods because they are part of
    # this class's tested public surface (e.g.
    # tests/test_mcmc_strategy_gaussian_uid_growth.py calls
    # `strategy._append_running_state(...)` directly).

    def _energy_weights(self, fallback_weights, utility, indices, temperature):
        return energy_weights(fallback_weights, utility, indices, temperature)

    def _relocate_energy_guided(self, gaussians, args, dead_mask, utility, temperature, binoms, min_opacity, iteration=0):
        return relocate_energy_guided(gaussians, args, dead_mask, utility, temperature, binoms, min_opacity, iteration)

    def _add_energy_guided(self, gaussians, args, utility, temperature, binoms, min_opacity, cap_max, growth_factor, iteration=0):
        return add_energy_guided(gaussians, args, utility, temperature, binoms, min_opacity, cap_max, growth_factor, iteration)

    def _reset_relocated_state(self, gaussians, dead_indices, sampled_idxs, iteration=0):
        return reset_relocated_state(gaussians, dead_indices, sampled_idxs, iteration)

    def _append_running_state(self, gaussians, old_count, added_count, parent_idxs=None, iteration=0):
        return append_running_state(gaussians, old_count, added_count, iteration, parent_idxs=parent_idxs)
