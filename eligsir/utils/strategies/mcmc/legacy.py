"""``ScheduledMCMCStrategy`` — the pre-gsplat-native scheduled MCMC adapter,
still used for ``--densification_strategy mcmc``/``hybrid`` on the legacy
model layout, and subclassed by the gsplat-native baseline strategy.

Moved verbatim out of ``utils/strategies/mcmc_strategy.py`` (Wave 2 Task 4,
2026-08-05) — no behavior change, only module location.
"""

import torch

from eligsir.utils.general_utils import build_scaling_rotation

from .base import (
    _augment_dead_mask_knn_isolation,
    _bubble_active_mask,
    _effective_cap_max,
    _exclude_locked_from_mask,
    _gate_dead_mask_to_visible,
    _locked_depth_mask,
)


def _resume_counter(value, name):
    if value is None:
        return None
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer or None")
    return value


class ScheduledMCMCStrategy:
    """Upstream-shaped adapter around this repo's scheduled Energy-MCMC logic."""

    def __init__(self, name):
        self.name = name
        self._next_reloc_iter = None
        self._next_grow_iter = None

    def initialize_state(self, **_kwargs):
        return {}

    def resume_state_dict(self):
        return {
            "owner_schema": "scheduled_mcmc_v1",
            "strategy_name": self.name,
            "next_reloc_iter": _resume_counter(self._next_reloc_iter, "next_reloc_iter"),
            "next_grow_iter": _resume_counter(self._next_grow_iter, "next_grow_iter"),
        }

    def load_resume_state_dict(self, payload, resolver=None):
        del resolver
        if not isinstance(payload, dict) or payload.get("owner_schema") != "scheduled_mcmc_v1":
            raise ValueError("incompatible scheduled MCMC resume state")
        required = {"owner_schema", "strategy_name", "next_reloc_iter", "next_grow_iter"}
        if set(payload) != required:
            raise ValueError("incomplete scheduled MCMC resume state")
        if payload["strategy_name"] != self.name:
            raise ValueError("MCMC strategy identity mismatch")
        next_reloc_iter = _resume_counter(payload["next_reloc_iter"], "next_reloc_iter")
        next_grow_iter = _resume_counter(payload["next_grow_iter"], "next_grow_iter")
        self._next_reloc_iter = next_reloc_iter
        self._next_grow_iter = next_grow_iter

    def step_pre_backward(self, **_kwargs):
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

    def inject_noise(self, gaussians, args, xyz_lr, visible=None, sparse_active_set=False, iteration=0, tb_writer=None):
        if iteration > int(getattr(args, "mcmc_noise_stop_iter", getattr(args, "iterations", 30_000))):
            return
        with torch.no_grad():
            if sparse_active_set and visible is not None:
                noise_idx = visible.nonzero(as_tuple=True)[0]
            else:
                noise_idx = torch.arange(gaussians.get_xyz.shape[0], device=gaussians.get_xyz.device)

            # Phase 3: Age-based MCMC filter for online mode
            if getattr(args, "online_mode", False) and hasattr(gaussians, "birth_frame") and hasattr(gaussians, "current_frame_idx"):
                _age_frames = gaussians.current_frame_idx - gaussians.birth_frame[noise_idx]
                _young_mask = _age_frames < 200  # only add noise to young Gaussians in online mode
                noise_idx = noise_idx[_young_mask]

            locked = _locked_depth_mask(gaussians, args)
            if locked is not None:
                noise_idx = noise_idx[~locked[noise_idx]]
            if noise_idx.numel() == 0:
                return

            # Phase 6: trust-region MCMC proposal routing
            _proposal_mode = getattr(args, "mcmc_proposal_mode", "legacy_clamped")
            if _proposal_mode == "trust_region":
                from eligsir.utils.gaussian_energy_policy import GaussianEnergyPolicy, energy_config_from_args
                # issue #385: see the identical gsplat-layout site in energy.py --
                # alpha_dead is unread here but stated explicitly by contract.
                _tr_cfg = energy_config_from_args(
                    args,
                    alpha_dead=float(args.energy_alpha_dead),
                    mcmc_proposal_mode="trust_region",
                )
                _tr_policy = GaussianEnergyPolicy(_tr_cfg)
                # sample_trust_region_noise returns [N_total, 3]; we apply to noise_idx subset
                _all_noise = _tr_policy.sample_trust_region_noise(
                    gaussians,
                    xyz_lr=xyz_lr,
                    noise_lr=float(args.noise_lr),
                    visible=noise_idx,
                )
                # _all_noise already has zeros outside noise_idx (see sample_trust_region_noise)
                gaussians._xyz.add_(_all_noise)
                if tb_writer and iteration % max(1, int(getattr(args, "scalar_log_interval", 100))) == 0:
                    _applied_noise = _all_noise[noise_idx]
                    _step_norms = _applied_noise.norm(dim=-1)
                    tb_writer.add_scalar("mcmc/noise_step_mean", float(_step_norms.mean()), iteration)
                    tb_writer.add_scalar("mcmc/noise_step_p95", float(torch.quantile(_step_norms, 0.95)), iteration)
                    tb_writer.add_scalar("mcmc/noise_step_max", float(_step_norms.max()), iteration)
                return
            if _proposal_mode == "surface_manifold":
                from eligsir.utils.strategies.surface_mcmc_strategy import sample_surface_manifold_noise
                _all_noise = sample_surface_manifold_noise(
                    gaussians,
                    noise_idx=noise_idx,
                    xyz_lr=xyz_lr,
                    noise_lr=float(args.noise_lr),
                    normal_scale=float(getattr(args, "surface_mcmc_normal_scale", 0.15)),
                    tangent_scale=float(getattr(args, "surface_mcmc_tangent_scale", 1.0)),
                    provisional_normal_scale=float(getattr(args, "surface_mcmc_provisional_normal_scale", 0.75)),
                )
                gaussians._xyz.add_(_all_noise)
                if tb_writer and iteration % max(1, int(getattr(args, "scalar_log_interval", 100))) == 0:
                    _applied_noise = _all_noise[noise_idx]
                    _step_norms = _applied_noise.norm(dim=-1)
                    tb_writer.add_scalar("mcmc/surface_noise_step_mean", float(_step_norms.mean()), iteration)
                    tb_writer.add_scalar("mcmc/surface_noise_step_p95", float(torch.quantile(_step_norms, 0.95)), iteration)
                return

            # Legacy clamped path (default, unchanged)
            # Cap per-axis scale before building noise covariance.
            # Without this, needles (e.g. scale=(0.3, 0.02, 0.02)) have a 225×
            # eigenvalue ratio, so the noise direction is overwhelmingly biased
            # along the major axis — causing persistent axis-aligned drift.
            noise_max_scale = float(getattr(args, "noise_max_scale", 0.05))
            scales_for_noise = gaussians.get_scaling[noise_idx]
            if noise_max_scale > 0:
                scales_for_noise = scales_for_noise.clamp(max=noise_max_scale)
            L = build_scaling_rotation(
                scales_for_noise,
                gaussians.get_rotation[noise_idx],
            )
            actual_covariance = L @ L.transpose(1, 2)

            def op_sigmoid(x, k=100, x0=0.995):
                return 1 / (1 + torch.exp(-k * (x - x0)))

            noise = torch.randn_like(gaussians._xyz[noise_idx]) * (
                op_sigmoid(1 - gaussians.get_opacity[noise_idx])
            ) * args.noise_lr * xyz_lr
            noise = torch.bmm(actual_covariance, noise.unsqueeze(-1)).squeeze(-1)
            max_step = float(getattr(args, "noise_max_step", 0.01))
            if max_step > 0:
                # Clamp per-Gaussian L2 displacement to max_step metres
                step_norm = noise.norm(dim=-1, keepdim=True).clamp_min(max_step)
                noise = noise * (max_step / step_norm)
            gaussians._xyz[noise_idx].add_(noise)

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
        # W3-39: absorb map_backend (and any other future caller-side kwarg)
        # without breaking every non-live strategy's call signature.
        **_w3_39_extra_kwargs,
    ):
        if self.name not in {"mcmc", "hybrid"}:
            return

        should_log = should_log_strategy or (lambda _iteration: False)
        if use_energy_mcmc:
            with torch.no_grad():
                if sched["allow_relocation"] and self._due(
                    "reloc", iteration, sched["relocate_interval"], getattr(args, "densify_from_iter", 500)
                ):
                    from eligsir.utils.gaussian_energy_policy import GaussianEnergyPolicy, energy_config_from_args
                    # issue #385: legacy-layout twin of energy.py's _pcfg -- same
                    # five pinned dead-mask fields, same fix. alpha_dead stays
                    # schedule-driven.
                    _policy_cfg = energy_config_from_args(
                        args, alpha_dead=sched["dead_opacity_threshold"]
                    )
                    _policy = GaussianEnergyPolicy(_policy_cfg)
                    # Phase 4: energy-based dead-mask routing
                    _energy_dead_mode = getattr(args, "energy_dead_mode", "legacy_clauses")
                    if _energy_dead_mode in ("energy_quantile", "energy_sigmoid"):
                        _dead_terms = _policy.compute_terms(gaussians, render_pkg or {}, iteration)
                        _pop_energy = _policy.compute_population_energy(_dead_terms)
                        dead_mask = _policy.compute_dead_mask_from_energy(
                            _pop_energy, gaussians,
                            active_mask=_bubble_active_mask(gaussians.get_xyz, args),
                        )
                        if tb_writer:
                            tb_writer.add_scalar("energy/dead_threshold", float(torch.quantile(_pop_energy.total, _policy_cfg.energy_dead_quantile)) if _energy_dead_mode == "energy_quantile" else _pop_energy.population_mean, iteration)
                            tb_writer.add_scalar("energy/dead_energy_mean", float(_pop_energy.total[dead_mask].mean()) if dead_mask.any() else 0.0, iteration)
                            tb_writer.add_scalar("energy/alive_energy_mean", float(_pop_energy.total[~dead_mask].mean()) if (~dead_mask).any() else 0.0, iteration)
                    else:
                        _young_prot = int(args.streaming_young_prot_iters)
                        _dead_result = _policy.compute_legacy_dead_mask(
                            gaussians, utility=utility,
                            current_iter=iteration, young_prot_iters=_young_prot,
                            active_mask=(_bubble_active_mask(gaussians.get_xyz, args)
                                         if getattr(args, "streaming_local_stats", False) else None),
                        )
                        dead_mask = _dead_result.mask
                        if tb_writer:
                            for clause, cnt in _dead_result.stats.items():
                                tb_writer.add_scalar(f"mcmc/dead_{clause}", cnt, iteration)
                    dead_mask = _exclude_locked_from_mask(dead_mask, gaussians, args)
                    dead_mask = _augment_dead_mask_knn_isolation(dead_mask, gaussians, args, current_iter=iteration)
                    dead_mask = _gate_dead_mask_to_visible(dead_mask, render_pkg, args)
                    dead_count = int(dead_mask.sum().item())
                    # Phase 10: soft lifecycle modulation of parent scores
                    _parent_scores = utility
                    if getattr(args, "soft_lifecycle_enabled", False) and utility is not None:
                        _lc_terms = _policy.compute_terms(gaussians, render_pkg or {}, iteration)
                        _maturity_prob = _policy.compute_maturity_logit(_lc_terms).sigmoid().detach()
                        _parent_scores = utility * _maturity_prob
                    gaussians.relocate_gs_energy_guided(
                        dead_mask=dead_mask,
                        parent_scores=_parent_scores,
                        temperature=temperature,
                        exclude_parent_mask=_locked_depth_mask(gaussians, args),
                    )
                    self._log_reloc(
                        tb_writer, should_log, iteration, dead_count, sched,
                        gaussians.get_xyz.shape[0],
                    )

                if self.name == "mcmc" and sched["allow_growth"] and self._due(
                    "grow", iteration, sched["grow_interval"], getattr(args, "densify_from_iter", 500)
                ):
                    before = gaussians.get_xyz.shape[0]
                    # Phase 10: soft lifecycle modulation for growth
                    _grow_parent_scores = utility
                    if getattr(args, "soft_lifecycle_enabled", False) and utility is not None:
                        from eligsir.utils.gaussian_energy_policy import GaussianEnergyPolicy, energy_config_from_args
                        # issue #385: soft-lifecycle sharpness now reaches
                        # compute_maturity_logit from args.
                        _grow_policy = GaussianEnergyPolicy(energy_config_from_args(
                            args, alpha_dead=sched["dead_opacity_threshold"],
                        ))
                        _grow_terms = _grow_policy.compute_terms(gaussians, render_pkg or {}, iteration)
                        _grow_maturity = _grow_policy.compute_maturity_logit(_grow_terms).sigmoid().detach()
                        _grow_parent_scores = utility * _grow_maturity
                    added = gaussians.add_new_gs_energy_guided(
                        cap_max=_effective_cap_max(args),
                        growth_factor=sched["growth_factor"],
                        parent_scores=_grow_parent_scores,
                        temperature=temperature,
                        exclude_parent_mask=_locked_depth_mask(gaussians, args),
                    )
                    self._log_growth(tb_writer, should_log, iteration, before, gaussians.get_xyz.shape[0], added, sched)
            return

        with torch.no_grad():
            if sched["allow_relocation"] and self._due(
                "reloc", iteration, sched["relocate_interval"], getattr(args, "densify_from_iter", 500)
            ):
                dead_mask = (gaussians.get_opacity <= sched["dead_opacity_threshold"]).squeeze(-1)
                dead_mask = _exclude_locked_from_mask(dead_mask, gaussians, args)
                dead_mask = _augment_dead_mask_knn_isolation(dead_mask, gaussians, args, current_iter=iteration)
                dead_mask = _gate_dead_mask_to_visible(dead_mask, render_pkg, args)
                dead_count = int(dead_mask.sum().item())
                gaussians.relocate_gs(dead_mask=dead_mask)
                self._log_reloc(tb_writer, should_log, iteration, dead_count, sched, gaussians.get_xyz.shape[0])

            if self.name == "mcmc" and sched["allow_growth"] and self._due(
                "grow", iteration, sched["grow_interval"], getattr(args, "densify_from_iter", 500)
            ):
                before = gaussians.get_xyz.shape[0]
                added = gaussians.add_new_gs(
                    cap_max=_effective_cap_max(args),
                    growth_factor=sched["growth_factor"],
                )
                self._log_growth(tb_writer, should_log, iteration, before, gaussians.get_xyz.shape[0], added, sched)

    def _log_reloc(self, tb_writer, should_log, iteration, dead_count, sched, count):
        if tb_writer:
            tb_writer.add_scalar("mcmc/dead_count", dead_count, iteration)
        if should_log(iteration):
            print(
                f"[mcmc-reloc] iter={iteration} "
                f"dead={dead_count} "
                f"thr={sched['dead_opacity_threshold']:.5f} "
                f"reloc_int={sched['relocate_interval']} "
                f"rho={sched['rho']:.3f} "
                f"N={count}",
                flush=True,
            )

    def _log_growth(self, tb_writer, should_log, iteration, before, after, added, sched):
        if tb_writer:
            tb_writer.add_scalar("mcmc/added_count", added, iteration)
            tb_writer.add_scalar("mcmc/growth_delta_N", after - before, iteration)
        if should_log(iteration):
            print(
                f"[mcmc-grow] iter={iteration} "
                f"added={added} "
                f"N={before}->{after} "
                f"factor={sched['growth_factor']:.4f} "
                f"grow_int={sched['grow_interval']} "
                f"rho={sched['rho']:.3f}",
                flush=True,
            )
