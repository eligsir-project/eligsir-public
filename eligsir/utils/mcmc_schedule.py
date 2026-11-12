import math
from dataclasses import dataclass


@dataclass
class MCMCScheduleConfig:
    start_iter: int = 500
    stop_growth_iter: int = 12_000
    stop_reloc_iter: int = 25_000

    relocate_interval_min: int = 50
    relocate_interval_max: int = 500
    relocate_tau: float = 0.65

    grow_interval_min: int = 100
    grow_interval_max: int = 2000
    grow_tau: float = 0.35

    growth_factor_start: float = 1.05
    growth_factor_min: float = 1.002
    growth_factor_tau: float = 0.35

    # Optional post-ingestion growth boost (streaming only): once the caller
    # signals ingestion_complete=True to get_mcmc_schedule(), this value
    # replaces growth_factor_start for the remainder of the run instead of
    # the normal value -- normal growth strength while frames are still
    # arriving, then a stronger target once all source frames are admitted.
    # 0.0 (default) disables this and preserves growth_factor_start
    # unconditionally, i.e. today's behavior when unset.
    post_ingestion_growth_factor_start: float = 0.0

    cap_growth_power: float = 2.0
    cap_interval_strength: float = 4.0
    cap_interval_power: float = 2.0
    cap_stop_ratio: float = 0.98

    # 2026-07-06: raised from 0.003/0.010 to match arguments/__init__.py's
    # mcmc_dead_opacity_start/end -- see that file's comment for the root
    # cause (dead-mask threshold was 5x stricter than the 0.05 opacity
    # threshold this codebase's own geometry reporting uses to call a
    # Gaussian "dead-looking", so real floater mass was never pruned).
    dead_opacity_start: float = 0.015
    dead_opacity_end: float = 0.050
    dead_opacity_power: float = 1.5

    # Merge A (CapacityPolicy): soft TARGET-N for the growth/deficit controller.
    # When > 0 it overrides target_splat_end as the count the deficit aims for,
    # while cap_max stays the HARD relocation/insertion ceiling (rho/cap_decay/
    # cap_stop keep reading cap_max). Default 0 is behaviour-neutral.
    target_capacity: int = 0

    # Target-deficit growth controller (optional, selectable)
    use_target_deficit: bool = False
    target_splat_end: int = 150_000
    target_q_start: float = 0.05
    target_q_end: float = 0.85
    target_tau: float = 0.45
    growth_factor_eps: float = 1e-4


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def progress(iteration, start, stop):
    return clamp((iteration - start) / float(max(1, stop - start)), 0.0, 1.0)


def exp_interp(u, start, end, tau):
    tau = max(tau, 1e-6)
    denom = 1.0 - math.exp(-1.0 / tau)
    alpha = (1.0 - math.exp(-u / tau)) / denom
    return start + (end - start) * alpha


def get_mcmc_schedule(
    iteration,
    current_n,
    cap_max,
    cfg: MCMCScheduleConfig,
    ingestion_complete: bool = False,
    *,
    grow_due: bool | None = None,
    relocate_due: bool | None = None,
    target_deficit_enabled: bool | None = None,
):
    if (grow_due is None) != (relocate_due is None):
        raise ValueError("semantic MCMC due flags must be supplied together")
    if grow_due is not None and (
        type(grow_due) is not bool or type(relocate_due) is not bool
    ):
        raise TypeError("semantic MCMC due flags must be bool")
    if target_deficit_enabled is not None and type(target_deficit_enabled) is not bool:
        raise TypeError("target_deficit_enabled must be bool when supplied")
    # The policy remains opt-in.  Callers can additionally defer its costly
    # target-chasing behaviour until a phase boundary, without making a
    # disabled policy active.
    use_target_deficit = bool(
        cfg.use_target_deficit
        and (target_deficit_enabled is None or target_deficit_enabled)
    )
    _growth_factor_start = cfg.growth_factor_start
    if ingestion_complete and cfg.post_ingestion_growth_factor_start > 0:
        _growth_factor_start = cfg.post_ingestion_growth_factor_start

    rho = 1.0
    if cap_max is not None and cap_max > 0:
        rho = clamp(current_n / float(cap_max), 0.0, 1.0)

    if grow_due is not None:
        # Canonical semantic path. Every value used by the mutator is derived
        # from current population/configuration or an explicit due flag. No
        # legacy iteration horizon, interpolation, or interval is evaluated.
        cap_decay = max(0.0, 1.0 - rho) ** cfg.cap_growth_power
        if use_target_deficit:
            target = (
                cfg.target_capacity
                if cfg.target_capacity > 0
                else cfg.target_splat_end
            )
            deficit = max(0.0, (target - current_n) / max(float(target), 1.0))
            semantic_growth_factor = 1.0 + (
                (_growth_factor_start - 1.0)
                * (deficit ** cfg.cap_growth_power)
            )
        else:
            semantic_growth_factor = 1.0 + (
                (_growth_factor_start - 1.0) * cap_decay
            )
        if semantic_growth_factor < 1.0 + cfg.growth_factor_eps:
            semantic_growth_factor = 1.0
        else:
            semantic_growth_factor = max(
                cfg.growth_factor_min, semantic_growth_factor
            )
        return {
            "allow_relocation": relocate_due,
            "allow_growth": bool(
                grow_due
                and rho < cfg.cap_stop_ratio
                and semantic_growth_factor > 1.0
            ),
            "relocate_interval": 1,
            "grow_interval": 1,
            "growth_factor": semantic_growth_factor,
            "semantic_growth_factor": semantic_growth_factor,
            "dead_opacity_threshold": cfg.dead_opacity_start,
            "rho": rho,
            "u_growth": 0.0,
            "u_reloc": 0.0,
            "grow_due": grow_due,
            "relocate_due": relocate_due,
        }

    u_growth = progress(iteration, cfg.start_iter, cfg.stop_growth_iter)
    u_reloc = progress(iteration, cfg.start_iter, cfg.stop_reloc_iter)

    relocate_interval = exp_interp(
        u_reloc,
        cfg.relocate_interval_min,
        cfg.relocate_interval_max,
        cfg.relocate_tau,
    )

    base_grow_interval = exp_interp(
        u_growth,
        cfg.grow_interval_min,
        cfg.grow_interval_max,
        cfg.grow_tau,
    )

    grow_interval = base_grow_interval * (
        1.0 + cfg.cap_interval_strength * (rho ** cfg.cap_interval_power)
    )

    time_decay = math.exp(-u_growth / max(cfg.growth_factor_tau, 1e-6))
    cap_decay = max(0.0, 1.0 - rho) ** cfg.cap_growth_power

    if use_target_deficit:
        # Target-deficit growth controller. Reuse the growth progress instead
        # of assuming a fixed 30k-iteration run.
        q = exp_interp(u_growth, cfg.target_q_start, cfg.target_q_end, cfg.target_tau)
        # Merge A: prefer the soft target_capacity over target_splat_end when set.
        _target_n_scale = cfg.target_capacity if cfg.target_capacity > 0 else cfg.target_splat_end
        N_target = _target_n_scale * q
        deficit = max(0.0, (N_target - current_n) / max(N_target, 1.0))
        growth_factor = 1.0 + (
            (_growth_factor_start - 1.0) * time_decay * (deficit ** cfg.cap_growth_power)
        )
    else:
        growth_factor = 1.0 + (
            (_growth_factor_start - 1.0) * time_decay * cap_decay
        )

    if growth_factor < 1.0 + cfg.growth_factor_eps:
        growth_factor = 1.0
    else:
        growth_factor = max(cfg.growth_factor_min, growth_factor)

    dead_threshold = cfg.dead_opacity_start + (
        cfg.dead_opacity_end - cfg.dead_opacity_start
    ) * (u_reloc ** cfg.dead_opacity_power)

    allow_relocation = cfg.start_iter < iteration < cfg.stop_reloc_iter
    allow_growth = (
        cfg.start_iter < iteration < cfg.stop_growth_iter
        and rho < cfg.cap_stop_ratio
        and growth_factor > 1.0
    )

    schedule = {
        "allow_relocation": allow_relocation,
        "allow_growth": allow_growth,
        "relocate_interval": max(1, int(round(relocate_interval))),
        "grow_interval": max(1, int(round(grow_interval))),
        "growth_factor": growth_factor,
        "dead_opacity_threshold": dead_threshold,
        "rho": rho,
        "u_growth": u_growth,
        "u_reloc": u_reloc,
    }
    return schedule
