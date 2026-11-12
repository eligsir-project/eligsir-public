"""Birth/relocation proposal generation for the energy-guided gsplat MCMC
strategy: energy-weighted sampling, relocation, growth, and the associated
running-state (per-Gaussian buffer) bookkeeping.

Moved out of ``GsplatEnergyMCMCStrategy`` in ``utils/strategies/mcmc_strategy.py``
(Wave 2 Task 4, 2026-08-05). These were previously instance methods
(``self._energy_weights`` etc.) that did not read or write any state on
``self`` beyond their explicit arguments — converting them to free functions
here is behavior-preserving; ``GsplatEnergyMCMCStrategy.step_post_backward``
now calls them as plain functions instead of ``self.<method>``.
"""

import math

import torch

from .base import _bubble_active_mask, _locked_depth_mask
from eligsir.utils.gaussian_belief import growth_pad_belief_buffers, relocate_belief_buffers


def temperature_from_optimizer_updates(args, *, optimizer_updates: int) -> float:
    """Evaluate energy temperature on the explicit optimizer-update clock."""

    if type(optimizer_updates) is not int:
        raise TypeError("optimizer_updates must be an exact integer")
    if optimizer_updates < 0:
        raise ValueError("optimizer_updates must be non-negative")
    horizon = args.mcmc_temperature_horizon_optimizer_updates
    if type(horizon) is not int or horizon <= 0:
        raise ValueError(
            "mcmc_temperature_horizon_optimizer_updates must be a positive integer"
        )
    progress = min(optimizer_updates / float(horizon), 1.0)
    tau = max(float(args.energy_temp_tau), 1e-6)
    denominator = 1.0 - math.exp(-1.0 / tau)
    return float(
        args.energy_temp_min
        + (args.energy_temp_start - args.energy_temp_min)
        * (1.0 - (1.0 - math.exp(-progress / tau)) / denominator)
    )


@torch.no_grad()
def sample_legacy_clamped_noise(gaussians, params, args, lr, noise_lr):
    """Default ("legacy_clamped") MCMC position-noise proposal for
    ``GsplatEnergyMCMCStrategy`` -- gsplat-layout equivalent of
    ``utils/strategies/mcmc/legacy.py``'s own clamped noise path (same
    op_sigmoid opacity gate, same ``noise_lr`` scaling, same two clamps).

    Extracted 2026-08-22 while root-causing a real Replica room0 "escape
    plume" (``scene_bbox_volume_ratio``=173x the camera trajectory's own
    bbox, individual live Gaussians up to ~200m from the scene). Before this
    fix, ``GsplatEnergyMCMCStrategy``'s default noise path called gsplat's
    stock, UNCLAMPED ``gsplat.strategy.ops.inject_noise_to_position`` --
    despite this proposal mode being named "legacy_clamped", it applied
    neither a per-Gaussian scale cap nor an absolute step-size cap, unlike
    the sibling legacy/mcmc strategy which already wires
    ``args.noise_max_scale``/``args.noise_max_step`` (non-zero defaults
    0.05/0.01) into exactly this computation. A newly-grown/relocated
    Gaussian transiently below the near-zero-opacity gate could draw an
    unbounded single-step position jump and be flung tens-to-hundreds of
    metres outside the scene with no recovery path once outside every
    camera's view frustum. Root-caused on a real GPU run via
    ``birth_iter``/``birth_method`` tracing the escaped Gaussians to recent
    MCMC growth/relocation clones, not to scene initialization.

    Uses ``args.gsplat_mcmc_noise_max_step``/``args.gsplat_mcmc_noise_max_scale``
    (defaults 0.1/0.05) -- deliberately SEPARATE knobs from the legacy
    strategy's own ``noise_max_step``/``noise_max_scale`` (0.01/0.05), not a
    reuse of them: a real GPU A/B on Replica room0 found the legacy
    strategy's 0.01m step cap over-constrains this strategy's much larger
    ``noise_lr`` (5e5 default), costing ~2.0dB test PSNR vs. fully unclamped
    at a matched 6000-iteration budget (22.32 vs 24.37dB) despite also
    fixing the escape plume; 0.1m recovers most of that quality (22.92dB)
    while still cutting ``scene_bbox_volume_ratio`` from 1338x (unclamped)
    to 25.5x. See ``arguments/__init__.py``'s field comment for the full
    tradeoff table and issue #1039 for the raw run data.

    Args:
        gaussians: Gaussian model (gsplat layout; used for get_scaling/get_rotation).
        params: ``gaussians.params`` (used for the raw ``means``/``opacities``).
        args: Full args namespace (reads ``gsplat_mcmc_noise_max_scale``/
            ``gsplat_mcmc_noise_max_step``).
        lr: Current position learning rate.
        noise_lr: ``args.noise_lr`` (or the strategy's own noise_lr).

    Returns:
        ``[N, 3]`` noise tensor, NOT yet added to ``params["means"]`` --
        callers apply it under ``torch.no_grad()``.
    """
    from eligsir.utils.general_utils import build_scaling_rotation

    noise_max_scale = float(getattr(args, "gsplat_mcmc_noise_max_scale", 0.05))
    scales_for_noise = gaussians.get_scaling
    if noise_max_scale > 0:
        scales_for_noise = scales_for_noise.clamp(max=noise_max_scale)
    L = build_scaling_rotation(scales_for_noise, gaussians.get_rotation)
    covars = L @ L.transpose(1, 2)

    def op_sigmoid(x, k=100, x0=0.995):
        return 1 / (1 + torch.exp(-k * (x - x0)))

    opacities = torch.sigmoid(params["opacities"].flatten())
    scaler = float(lr if lr is not None else 0.0) * float(noise_lr)
    noise = torch.randn_like(params["means"]) * op_sigmoid(1 - opacities).unsqueeze(-1) * scaler
    noise = torch.bmm(covars, noise.unsqueeze(-1)).squeeze(-1)

    max_step = float(getattr(args, "gsplat_mcmc_noise_max_step", 0.1))
    if max_step > 0:
        step_norm = noise.norm(dim=-1, keepdim=True).clamp_min(max_step)
        noise = noise * (max_step / step_norm)

    return noise


def energy_weights(fallback_weights, utility, indices, temperature):
    weights = fallback_weights.detach().flatten().clamp_min(0.0)
    if utility is None:
        return weights
    if utility.shape[0] < int(indices.max().item()) + 1:
        return weights

    scores = utility.detach().float()[indices]
    scores = torch.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    scores = torch.log1p(torch.clamp(scores, min=0.0))
    if scores.numel() == 0:
        return weights

    median = scores.median()
    mad = (scores - median).abs().median().clamp_min(1e-6)
    scores = ((scores - median) / mad).clamp(-5.0, 5.0)
    probs = torch.softmax(scores / max(float(temperature), 1e-6), dim=0)
    if not torch.isfinite(probs).all() or probs.sum() <= 0:
        return weights
    return probs


def relocate_energy_guided(gaussians, args, dead_mask, utility, temperature, binoms, min_opacity, iteration):
    if dead_mask is None or dead_mask.sum() == 0:
        return 0

    from gsplat.strategy.ops import _multinomial_sample, _update_param_with_optimizer, compute_relocation

    params = gaussians.params
    optimizers = gaussians.optimizers
    opacities = torch.sigmoid(params["opacities"].flatten())
    locked = _locked_depth_mask(gaussians, args)
    if locked is not None and locked.shape[0] == dead_mask.shape[0]:
        dead_mask = dead_mask & ~locked
    from eligsir.utils.seed_lineage_freeze import exclude_frozen_seed_mask
    relocation_candidates = dead_mask
    dead_mask = exclude_frozen_seed_mask(
        gaussians, args, relocation_candidates,
        iteration=iteration, role="relocation_target",
    )
    # Relocation revives/reseeds the target slot from a donor. Record that
    # same exclusion under both causal roles without changing the mask twice.
    exclude_frozen_seed_mask(
        gaussians, args, relocation_candidates,
        iteration=iteration, role="reseed",
    )
    # issue #1262 category (a): an insufficiently-exposed Gaussian has no evidence
    # either way against it -- never relocate it away. No-op when the flag is off.
    from eligsir.utils.resolution_exposure import exclude_unexposed_mask
    dead_mask = exclude_unexposed_mask(
        gaussians, args, dead_mask, iteration=iteration, role="relocation_target",
    )
    exclude_unexposed_mask(
        gaussians, args, relocation_candidates, iteration=iteration, role="reseed",
    )
    candidate_dead_indices = relocation_candidates.nonzero(as_tuple=True)[0]
    dead_indices = dead_mask.nonzero(as_tuple=True)[0]
    alive_indices = (~dead_mask).nonzero(as_tuple=True)[0]
    if locked is not None and locked.shape[0] == opacities.shape[0]:
        alive_indices = alive_indices[~locked[alive_indices]]
    # P0.6: relocation destination parents also get mutated in place by
    # param_fn (opacity/scale split) — restrict them to the active bubble
    # so frozen out-of-bubble rows can't be corrupted.
    _bubble = _bubble_active_mask(params["means"], args)
    if _bubble is not None and _bubble.shape[0] == opacities.shape[0]:
        alive_indices = alive_indices[_bubble[alive_indices]]
    baseline_alive_indices = alive_indices
    from eligsir.utils.seed_lineage_freeze import filter_frozen_seed_indices
    alive_indices = filter_frozen_seed_indices(
        gaussians, args, alive_indices,
        iteration=iteration, role="relocation_donor",
    )
    from eligsir.utils.resolution_exposure import filter_unexposed_indices
    alive_indices = filter_unexposed_indices(
        gaussians, args, alive_indices,
        iteration=iteration, role="relocation_donor",
    )
    n_draw = int(candidate_dead_indices.numel())
    n = int(dead_indices.numel())
    if n_draw == 0:
        return 0
    if alive_indices.numel() == 0:
        if baseline_alive_indices.numel() > 0:
            baseline_weights = energy_weights(
                opacities[baseline_alive_indices], utility, baseline_alive_indices, temperature
            )
            _multinomial_sample(baseline_weights, n_draw, replacement=True)
        return 0

    weights = energy_weights(opacities[alive_indices], utility, alive_indices, temperature)
    # Preserve the baseline relocation proposal count even when frozen seed
    # targets are excluded. Draw once per original target, then discard the
    # donor proposals paired with excluded targets.
    sampled_local = _multinomial_sample(weights, n_draw, replacement=True)
    sampled_all = alive_indices[sampled_local]
    sampled_idxs = sampled_all[dead_mask[candidate_dead_indices]]
    if n == 0:
        return 0
    ratios = torch.bincount(sampled_idxs, minlength=opacities.shape[0])[sampled_idxs] + 1

    eps = torch.finfo(torch.float32).eps
    new_opacities, new_scales = compute_relocation(
        opacities=opacities[sampled_idxs],
        scales=torch.exp(params["scales"])[sampled_idxs],
        ratios=ratios,
        binoms=binoms,
    )
    new_opacities = torch.clamp(new_opacities, max=1.0 - eps, min=min_opacity)

    def param_fn(name, p):
        if name == "opacities":
            p[sampled_idxs] = torch.logit(new_opacities)
        elif name == "scales":
            p[sampled_idxs] = torch.log(new_scales)
        p[dead_indices] = p[sampled_idxs]
        return torch.nn.Parameter(p, requires_grad=p.requires_grad)

    def optimizer_fn(_key, v):
        v[sampled_idxs] = 0
        return v

    _update_param_with_optimizer(param_fn, optimizer_fn, params, optimizers)
    reset_relocated_state(gaussians, dead_indices, sampled_idxs, iteration)
    return n


def add_energy_guided(gaussians, args, utility, temperature, binoms, min_opacity, cap_max, growth_factor, iteration):
    from gsplat.strategy.ops import _multinomial_sample, _update_param_with_optimizer, compute_relocation

    params = gaussians.params
    optimizers = gaussians.optimizers
    current_n = params["means"].shape[0]
    target_n = min(int(cap_max), int(float(growth_factor) * current_n))
    n = max(0, target_n - current_n)
    if n <= 0:
        return 0

    opacities = torch.sigmoid(params["opacities"].flatten())
    all_indices = torch.arange(current_n, device=opacities.device)
    locked = _locked_depth_mask(gaussians, args)
    if locked is not None and locked.shape[0] == current_n:
        all_indices = all_indices[~locked]
    # P0.6: growth parents must come from the active bubble population.
    # param_fn below mutates parent rows (opacity/scale) IN PLACE, outside
    # the training loop's snapshot/restore window — sampling an out-of-bubble
    # (frozen) parent would silently corrupt frozen geometry.
    _bubble = _bubble_active_mask(params["means"], args)
    if _bubble is not None and _bubble.shape[0] == current_n:
        all_indices = all_indices[_bubble[all_indices]]
    # Escape-plume fix: growth parents must be RECENTLY RENDERED. Trapped
    # opaque mass behind walls has high opacity/utility but zero recent
    # visibility — cloning it multiplied the plume 0.8%->55% of live mass.
    if getattr(args, "streaming_growth_parents_visible_only", False):
        _vema = getattr(gaussians, "visibility_ema", None)
        if _vema is not None and _vema.shape[0] == current_n:
            _vis_ok = _vema.squeeze(-1) > 0.01
            all_indices = all_indices[_vis_ok[all_indices]]
    from eligsir.utils.seed_lineage_freeze import filter_frozen_seed_indices
    pre_freeze_indices = all_indices
    all_indices = filter_frozen_seed_indices(
        gaussians, args, all_indices,
        iteration=iteration, role="growth_parent",
    )
    from eligsir.utils.resolution_exposure import filter_unexposed_indices
    all_indices = filter_unexposed_indices(
        gaussians, args, all_indices,
        iteration=iteration, role="growth_parent",
    )
    if all_indices.numel() == 0:
        if pre_freeze_indices.numel() > 0:
            baseline_weights = energy_weights(
                opacities[pre_freeze_indices],
                utility,
                pre_freeze_indices,
                temperature,
            )
            _multinomial_sample(baseline_weights, n, replacement=True)
        return 0
    weights = energy_weights(opacities[all_indices], utility, all_indices, temperature)
    sampled_local = _multinomial_sample(weights, n, replacement=True)
    sampled_idxs = all_indices[sampled_local]
    ratios = torch.bincount(sampled_idxs, minlength=current_n)[sampled_idxs] + 1

    eps = torch.finfo(torch.float32).eps
    new_opacities, new_scales = compute_relocation(
        opacities=opacities[sampled_idxs],
        scales=torch.exp(params["scales"])[sampled_idxs],
        ratios=ratios,
        binoms=binoms,
    )
    new_opacities = torch.clamp(new_opacities, max=1.0 - eps, min=min_opacity)

    def param_fn(name, p):
        if name == "opacities":
            p[sampled_idxs] = torch.logit(new_opacities)
        elif name == "scales":
            p[sampled_idxs] = torch.log(new_scales)
        p_new = torch.cat([p, p[sampled_idxs]])
        return torch.nn.Parameter(p_new, requires_grad=p.requires_grad)

    def optimizer_fn(_key, v):
        v_new = torch.zeros((len(sampled_idxs), *v.shape[1:]), device=v.device)
        return torch.cat([v, v_new])

    _update_param_with_optimizer(param_fn, optimizer_fn, params, optimizers)
    append_running_state(gaussians, current_n, n, iteration, parent_idxs=sampled_idxs)
    return n


def reset_relocated_state(gaussians, dead_indices, sampled_idxs, iteration):
    visibility_ema = getattr(gaussians, "visibility_ema", None)
    if visibility_ema is not None and visibility_ema.numel() > 0:
        if visibility_ema.shape[0] == gaussians.get_xyz.shape[0]:
            visibility_ema[dead_indices] = 0.0
            visibility_ema[sampled_idxs] = visibility_ema[sampled_idxs].clamp_max(0.5)

    provisional = getattr(gaussians, "provisional", None)
    if provisional is not None and provisional.shape[0] == gaussians.get_xyz.shape[0]:
        gaussians.provisional[dead_indices] = gaussians.provisional[sampled_idxs]
        gaussians.birth_frame[dead_indices] = gaussians.birth_frame[sampled_idxs]
        gaussians.support_count[dead_indices] = gaussians.support_count[sampled_idxs]
    for name in ("depth_conflict_count", "depth_last_support_uid", "depth_last_conflict_uid"):
        tensor = getattr(gaussians, name, None)
        if tensor is not None and tensor.shape[0] == gaussians.get_xyz.shape[0]:
            tensor[dead_indices] = tensor[sampled_idxs]
    # Finding D (Q2): a revived dead slot inherits the parent's geometry/anchor
    # priors instead of keeping the dead Gaussian's stale anchors.
    for name in ("anchor_xyz", "anchor_normal", "anchor_confidence", "geometry_source",
                 "surface_state", "anchor_scale_log", "anchor_opacity_logit",
                 "lifecycle_state", "utility_ema", "anchor_iter"):
        tensor = getattr(gaussians, name, None)
        if tensor is not None and tensor.shape[0] == gaussians.get_xyz.shape[0]:
            tensor[dead_indices] = tensor[sampled_idxs]
    # Lineage: a revived dead slot keeps its own gaussian_uid (capacity-slot
    # identity never changes on reseed) but adopts the donor's lineage —
    # parent_uid/root_uid/generation/birth_method/last_reseed_iter, plus a
    # refreshed birth_iter, so age/lineage tracking reflects the new donor
    # geometry rather than stale bookkeeping from whichever Gaussian died in
    # this slot.
    from eligsir.utils.gaussian_lineage import reseed_lineage_update
    _n = gaussians.get_xyz.shape[0]
    # gaussian_uid is fetched defensively (getattr + length check) exactly like
    # the growth path's guard in append_running_state below -- it is indexed by
    # sampled_idxs inside reseed_lineage_update, so a missing or stale-length
    # buffer must skip the lineage update, not raise.
    _existing_uid = getattr(gaussians, "gaussian_uid", None)
    _parent_uid = getattr(gaussians, "parent_uid", None)
    _root_uid = getattr(gaussians, "root_uid", None)
    _generation = getattr(gaussians, "generation", None)
    if (_existing_uid is not None and _parent_uid is not None and _root_uid is not None
            and _generation is not None
            and _existing_uid.shape[0] == _n and _parent_uid.shape[0] == _n):
        updated = reseed_lineage_update(
            _existing_uid, _root_uid, _generation, dead_indices, sampled_idxs, iteration,
        )
        gaussians.parent_uid[dead_indices] = updated["parent_uid"]
        gaussians.root_uid[dead_indices] = updated["root_uid"]
        gaussians.generation[dead_indices] = updated["generation"]
        _birth_method = getattr(gaussians, "birth_method", None)
        if _birth_method is not None and _birth_method.shape[0] == gaussians.get_xyz.shape[0]:
            gaussians.birth_method[dead_indices] = updated["birth_method"]
        _last_reseed = getattr(gaussians, "last_reseed_iter", None)
        if _last_reseed is not None and _last_reseed.shape[0] == gaussians.get_xyz.shape[0]:
            gaussians.last_reseed_iter[dead_indices] = updated["last_reseed_iter"]
    _birth_iter = getattr(gaussians, "birth_iter", None)
    if _birth_iter is not None and _birth_iter.shape[0] == gaussians.get_xyz.shape[0]:
        gaussians.birth_iter[dead_indices] = int(iteration)
    # Part C: a relocated (revived) slot is fresh geometry — never inherit the
    # dead Gaussian's (or a parent's) multi-view consensus history, or it could
    # spuriously trigger / suppress a floater prune. Reset to a clean slate.
    try:
        from eligsir.utils.streaming_depth_lock import reset_consensus_counts
        reset_consensus_counts(gaussians, dead_indices)
    except Exception:
        pass
    # issue #1262: a revived slot is new geometry at a NEW position and has had
    # zero exposure there -- reset, never inherit the donor's exposure history.
    from eligsir.utils.resolution_exposure import reset_exposure_rows
    reset_exposure_rows(gaussians, dead_indices)
    # GaussianBelief (issue #1365 Phase 1).
    relocate_belief_buffers(gaussians, dead_indices, sampled_idxs)


def append_running_state(gaussians, old_count, added_count, iteration, parent_idxs=None):
    # gaussian_uid: MCMC growth clones existing Gaussians via a direct
    # params/optimizer torch.cat (see add_energy_guided above), bypassing
    # add_points_as_gaussians/_build_new_buffer_rows entirely -- the only
    # other place gaussian_uid rows get created. Without this, gaussian_uid
    # desyncs from get_xyz on the FIRST growth event of every run (frozen at
    # its pre-growth size while get_xyz keeps changing), and later a
    # prune_points call whose mask no longer matches gaussian_uid's stale
    # shape silently skips pruning it too (its own shape==mask.shape guard),
    # leaving it permanently wrong. Found live 2026-07-25: crashed
    # project_visible_gaussian_uids's boolean-mask index with "shape of the
    # mask [10405] ... does not match ... tensor [13283]" on the very first
    # real smoke test of this feature. A clone is a genuinely new Gaussian
    # (a new physical entity in the map even though seeded from a parent's
    # values), so it gets a FRESH uid here, matching every other insertion
    # site -- never the parent's uid, never a zero/placeholder pad (which
    # would collide many clones onto the same fake uid=0).
    from eligsir.utils.gaussian_lineage import growth_lineage_rows
    # existing_gaussian_uid must be each CURRENT row's own identity
    # (gaussians.gaussian_uid), NOT gaussians.parent_uid -- growth_lineage_rows
    # indexes it by parent_idxs to find the sampled parent's own UID, which
    # becomes the new clone's parent_uid. Passing gaussians.parent_uid here
    # would make every clone a sibling of its parent instead of a child.
    _existing_uid = getattr(gaussians, "gaussian_uid", None)
    _parent_uid = getattr(gaussians, "parent_uid", None)
    _root_uid = getattr(gaussians, "root_uid", None)
    _generation = getattr(gaussians, "generation", None)
    _birth_method = getattr(gaussians, "birth_method", None)
    _last_reseed = getattr(gaussians, "last_reseed_iter", None)
    if (parent_idxs is not None and _existing_uid is not None and _parent_uid is not None
            and _root_uid is not None and _generation is not None
            and _existing_uid.shape[0] == old_count and _parent_uid.shape[0] == old_count):
        new_lineage = growth_lineage_rows(_existing_uid, _root_uid, _generation, parent_idxs)
        gaussians.parent_uid = torch.cat([_parent_uid, new_lineage["parent_uid"]], dim=0)
        gaussians.root_uid = torch.cat([_root_uid, new_lineage["root_uid"]], dim=0)
        gaussians.generation = torch.cat([_generation, new_lineage["generation"]], dim=0)
        if _birth_method is not None and _birth_method.shape[0] == old_count:
            gaussians.birth_method = torch.cat([_birth_method, new_lineage["birth_method"]], dim=0)
        if _last_reseed is not None and _last_reseed.shape[0] == old_count:
            gaussians.last_reseed_iter = torch.cat([_last_reseed, new_lineage["last_reseed_iter"]], dim=0)

    guid = getattr(gaussians, "gaussian_uid", None)
    if guid is not None and guid.shape[0] == old_count:
        next_uid = int(getattr(gaussians, "_next_gaussian_uid", old_count))
        new_uids = torch.arange(
            next_uid, next_uid + added_count, dtype=torch.int64, device=guid.device
        )
        gaussians._next_gaussian_uid = next_uid + added_count
        gaussians.gaussian_uid = torch.cat([guid, new_uids], dim=0)
    for name in ("visibility_ema", "xyz_gradient_accum", "denom", "birth_frame", "support_count", "provisional",
                 "anchor_iter", "depth_conflict_count", "utility_ema",
                 "seen_count", "consensus_support_count", "consensus_conflict_count",
                 # issue #1262: a grown clone has had zero exposure of its own. This is
                 # the site GAP-4 misses for last_supervised_iter/submap_id -- missing it
                 # desyncs the buffer and every downstream shape[0]==N guard then
                 # silently disables the veto.
                 "exposure_count"):
        tensor = getattr(gaussians, name, None)
        if tensor is not None and tensor.shape[0] == old_count:
            pad = torch.zeros((added_count, *tensor.shape[1:]), device=tensor.device, dtype=tensor.dtype)
            setattr(gaussians, name, torch.cat([tensor, pad], dim=0))
    # lifecycle_state is NOT a zero-default buffer like the ones above: 0 means
    # PROVISIONAL in the lifecycle_state enum (0=PROVISIONAL, 1=YOUNG, 2=MATURE,
    # 3=FROZEN). MCMC growth clones are non-provisional (the parallel
    # `provisional` bool buffer padded above is correctly False for them), so
    # padding lifecycle_state with 0 here disagreed with `provisional` on every
    # newly grown Gaussian until a periodic repair pass fixed it up later.
    # New growth clones start at YOUNG (1), matching `provisional=False`.
    lifecycle_state_buf = getattr(gaussians, "lifecycle_state", None)
    if lifecycle_state_buf is not None and lifecycle_state_buf.shape[0] == old_count:
        pad = torch.ones(
            (added_count, *lifecycle_state_buf.shape[1:]),
            device=lifecycle_state_buf.device,
            dtype=lifecycle_state_buf.dtype,
        )
        gaussians.lifecycle_state = torch.cat([lifecycle_state_buf, pad], dim=0)
    birth_iter_buf = getattr(gaussians, "birth_iter", None)
    if birth_iter_buf is not None and birth_iter_buf.shape[0] == old_count:
        pad = torch.full((added_count,), int(iteration), device=birth_iter_buf.device, dtype=birth_iter_buf.dtype)
        gaussians.birth_iter = torch.cat([birth_iter_buf, pad], dim=0)
    for name in ("depth_last_support_uid", "depth_last_conflict_uid", "consensus_last_seen_uid",
                 "exposure_last_view_uid", "supervised_tier_high_water"):
        tensor = getattr(gaussians, name, None)
        if tensor is not None and tensor.shape[0] == old_count:
            pad = torch.full(
                (added_count,),
                -1,
                device=tensor.device,
                dtype=tensor.dtype,
            )
            setattr(gaussians, name, torch.cat([tensor, pad], dim=0))
    # Finding D: geometry-prior buffers must grow with the model or every
    # `shape[0]==N` guard downstream silently disables surface features after
    # the first growth. Clones inherit the parent's priors when known.
    for name in ("anchor_normal", "anchor_confidence", "geometry_source", "surface_state"):
        tensor = getattr(gaussians, name, None)
        if tensor is None or tensor.shape[0] != old_count:
            continue
        if parent_idxs is not None and parent_idxs.shape[0] == added_count:
            pad = tensor[parent_idxs].clone()
        else:
            pad = torch.zeros((added_count, *tensor.shape[1:]), device=tensor.device, dtype=tensor.dtype)
            if name == "anchor_normal":
                pad[:, 2] = 1.0  # default +Z so the normal is unit-length, not zero
        setattr(gaussians, name, torch.cat([tensor, pad], dim=0))
    # anchor_xyz needs its own path: pad with the new Gaussian positions so anchor=current pos at birth
    anchor_xyz = getattr(gaussians, "anchor_xyz", None)
    if anchor_xyz is not None and anchor_xyz.shape[0] == old_count:
        new_means = gaussians.get_xyz[-added_count:].detach().clone()
        gaussians.anchor_xyz = torch.cat([anchor_xyz, new_means], dim=0)
    # Part D1: voter buffer ([N,3]) grows with NaN (grown clones have no voter
    # yet) — never inherit the parent's voter center.
    voter = getattr(gaussians, "consensus_last_voter_center", None)
    if voter is not None and voter.shape[0] == old_count:
        vpad = torch.full((added_count, 3), float("nan"), device=voter.device, dtype=voter.dtype)
        gaussians.consensus_last_voter_center = torch.cat([voter, vpad], dim=0)
    # anchor_scale_log / anchor_opacity_logit: pad with current values
    anchor_scale_log = getattr(gaussians, "anchor_scale_log", None)
    if anchor_scale_log is not None and anchor_scale_log.shape[0] == old_count:
        new_scales = gaussians.params["scales"][-added_count:].detach().clone() if hasattr(gaussians, "params") else torch.zeros((added_count, 3), device=anchor_scale_log.device)
        gaussians.anchor_scale_log = torch.cat([anchor_scale_log, new_scales], dim=0)
    anchor_opacity_logit = getattr(gaussians, "anchor_opacity_logit", None)
    if anchor_opacity_logit is not None and anchor_opacity_logit.shape[0] == old_count:
        new_ops = gaussians.params["opacities"][-added_count:].detach().clone() if hasattr(gaussians, "params") else torch.zeros(added_count, device=anchor_opacity_logit.device)
        gaussians.anchor_opacity_logit = torch.cat([anchor_opacity_logit, new_ops], dim=0)
    # GaussianBelief (issue #1365 Phase 1).
    growth_pad_belief_buffers(gaussians, old_count, added_count, parent_idxs=parent_idxs)
    radii = getattr(gaussians, "max_radii2D", None)
    if radii is not None and radii.numel() > 0 and radii.shape[0] == old_count:
        pad = torch.zeros((added_count,), device=radii.device, dtype=radii.dtype)
        gaussians.max_radii2D = torch.cat([radii, pad], dim=0)
