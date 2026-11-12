"""Observational candidate-survival funnel telemetry for MCMC relocation-target
and growth-parent selection.

Originally scoped as issue #1372 Phase 7 ("Consolidate Gaussian lifecycle
into a single evidence model"); #1372 is superseded by issue #1365 ("[P1
ARCH/METHOD] Probabilistic Gaussian belief + single population policy +
adaptive depth"), whose §17.1 "Ownership/conflict shadow telemetry" formally
specifies almost exactly this module (per-gate blocked counts plus the
MATURE/young_protected/underexposed/depth_conflicted intersections) as the
migration plan's required "measure before deleting" evidence (§18 Phase 0-1).
This module targets #1365 going forward.

This module is **purely observational**: it never changes a gating decision,
never mutates a Gaussian, and never writes to any production accumulator
(``utils.energy_mcmc._pruning_stats``, ``ResolutionExposureVeto.excluded_counts``,
``SeedLineageFreeze._excluded_counts``, etc.). Every predicate it computes is
either:

  1. a read-only re-derivation of a gate the real gating pipeline already
     applies, using the exact same input buffers (``birth_iter``,
     ``lifecycle_state``, ``exposure_count``, ``depth_conflict_count``,
     ``visibility_ema``) and the exact same *pure* helper functions the real
     pipeline calls (``_locked_depth_mask``, ``_bubble_active_mask``,
     ``_or_exposure_young``, ``SeedLineageFreeze.current_mask``); or
  2. a snapshot of a mask the real pipeline already computed, handed in by
     the caller.

Two gates are deliberately re-derived from raw buffers rather than reusing
their production entry point, because that entry point has a side effect on
a counter another part of the system reads and reports:

  - The resolution-exposure "protected" predicate is normally read via
    ``ResolutionExposureVeto.current_mask()``, which also updates
    ``bound_ceiling_lapses`` (feeds ``exposure_veto_telemetry()``). This
    module instead recomputes the same ``exposure_count < min_exposures``
    formula directly (see ``_underexposed_mask``), *ignoring* the
    bounded-immunity-ceiling lapse refinement -- a documented approximation,
    acceptable for telemetry, that keeps this module's re-derivation
    completely free of side effects on that counter.
  - ``SeedLineageFreeze.current_mask()`` has no such side effect (only
    ``exclude_mask``/``filter_indices`` bump counters), so the seed-freeze
    predicate here calls it directly.

The young-protection formula (age-since-birth OR exposure-insufficiency) is
already implemented three times in production
(``utils.energy_mcmc.compute_dead_mask``,
``utils.strategies.mcmc.energy_policy.GaussianEnergyPolicy.compute_legacy_dead_mask``,
``utils.strategies.mcmc.base._augment_dead_mask_knn_isolation``); this module
is a fourth, read-only copy for observability, in keeping with that existing
pattern.

Enable with ``--streaming_mcmc_candidate_funnel_telemetry`` (default off,
byte-identical when unset) and tune sampling cost with
``--streaming_mcmc_candidate_funnel_interval`` (optimizer-update stride,
default 200).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Optional

import torch


# The history is a diagnostic sink, not a second unbounded runtime log.  Keep
# enough samples for a bounded report while preventing long jobs from growing
# process memory when TensorBoard/report export is not invoked frequently.
MAX_FUNNEL_HISTORY = 1024


# ---------------------------------------------------------------------------
# Enablement / sampling
# ---------------------------------------------------------------------------


def funnel_enabled(args) -> bool:
    return bool(getattr(args, "streaming_mcmc_candidate_funnel_telemetry", False))


def should_sample(args, iteration: int) -> bool:
    """Whether the funnel should be computed on this optimizer-update tick.

    Always False when the feature flag is off (default) -- guarantees zero
    behavior change AND zero extra compute on the canonical profile.
    """
    if not funnel_enabled(args):
        return False
    interval = max(1, int(getattr(args, "streaming_mcmc_candidate_funnel_interval", 200)))
    return int(iteration) % interval == 0


# ---------------------------------------------------------------------------
# Read-only gate predicates (each returns a ``[N]`` bool tensor or ``None``
# when the corresponding feature is disabled / its buffers are unavailable)
# ---------------------------------------------------------------------------


def young_protected_mask(gaussians, args, iteration: int) -> Optional[torch.Tensor]:
    """True = protected by birth-grace age OR insufficient exposure.

    Mirrors ``utils.energy_mcmc.compute_dead_mask``'s young-protection clause
    exactly (age gate composed with the exposure-currency gate via
    ``_or_exposure_young``), without touching any pruning-stats accumulator.
    """
    n = int(gaussians.get_xyz.shape[0])
    device = gaussians.get_xyz.device
    birth_iter = getattr(gaussians, "birth_iter", None)
    # Fallback literal matches arguments/__init__.py's real default (400), per
    # tests/test_getattr_fallback_freshness.py's drift policy -- kept as a
    # getattr (not a bare attribute access) only because this function is
    # also exercised by unit tests against minimal synthetic Args objects.
    young_prot_iters = int(getattr(args, "streaming_young_prot_iters", 400) or 0)
    if birth_iter is None or birth_iter.shape[0] != n or young_prot_iters <= 0:
        young = torch.zeros(n, dtype=torch.bool, device=device)
    else:
        young = (int(iteration) - birth_iter.to(dtype=torch.int64)) < young_prot_iters
    from eligsir.utils.energy_mcmc import _or_exposure_young

    return _or_exposure_young(gaussians, young)


def underexposed_mask(gaussians, args) -> Optional[torch.Tensor]:
    """True = resolution-exposure veto would treat this Gaussian as
    insufficiently exposed. ``None`` when the feature is off/unavailable.

    Deliberately bypasses ``ResolutionExposureVeto.current_mask`` to avoid its
    ``bound_ceiling_lapses`` side effect -- see module docstring.
    """
    from eligsir.utils.resolution_exposure import get_resolution_exposure_veto

    veto = get_resolution_exposure_veto(gaussians, args)
    if veto is None or not veto.active():
        return None
    n = int(gaussians.get_xyz.shape[0])
    counts = getattr(gaussians, "exposure_count", None)
    if counts is None or counts.shape[0] != n:
        return None
    return counts.to(torch.int64) < int(veto.config.min_exposures)


def lifecycle_mature_mask(gaussians, args) -> Optional[torch.Tensor]:
    """True = MATURE or FROZEN (lifecycle_state >= 2). ``None`` when the
    lifecycle feature is off/unavailable."""
    # Fallback literal matches arguments/__init__.py's real default (True),
    # per tests/test_getattr_fallback_freshness.py's drift policy.
    if not getattr(args, "streaming_lifecycle_enabled", True):
        return None
    lc = getattr(gaussians, "lifecycle_state", None)
    n = int(gaussians.get_xyz.shape[0])
    if lc is None or lc.shape[0] != n:
        return None
    return lc >= 2


def depth_conflicted_mask(gaussians) -> Optional[torch.Tensor]:
    """True = recorded multiview depth conflict. ``None`` when unavailable."""
    n = int(gaussians.get_xyz.shape[0])
    dcc = getattr(gaussians, "depth_conflict_count", None)
    if dcc is None or dcc.shape[0] != n:
        return None
    return dcc > 0


def depth_locked_mask_readonly(gaussians, args) -> Optional[torch.Tensor]:
    """True = depth-lock candidate. Reuses the same pure helper the real
    gating pipeline calls (``_locked_depth_mask``), already invoked multiple
    times per real MCMC epoch in production, so a read-only extra call here
    introduces no new risk class."""
    from eligsir.utils.strategies.mcmc.base import _locked_depth_mask

    return _locked_depth_mask(gaussians, args)


def bubble_survive_mask_readonly(gaussians, args) -> Optional[torch.Tensor]:
    """True = inside the active spatial bubble (survives). ``None`` when the
    bubble feature is disabled."""
    from eligsir.utils.strategies.mcmc.base import _bubble_active_mask

    return _bubble_active_mask(gaussians.get_xyz, args)


def seed_freeze_mask_readonly(gaussians, args, iteration: int) -> Optional[torch.Tensor]:
    """True = frozen seed lineage. Uses ``SeedLineageFreeze.current_mask``
    directly -- a read-only query with no counter side effects (only
    ``exclude_mask``/``filter_indices`` bump ``_excluded_counts``)."""
    from eligsir.utils.seed_lineage_freeze import get_seed_lineage_freeze

    controller = get_seed_lineage_freeze(gaussians, args)
    if controller is None:
        return None
    mask = controller.current_mask(iteration=iteration)
    if mask is None or not bool(mask.any()):
        return None
    return mask


# ---------------------------------------------------------------------------
# Funnel assembly
# ---------------------------------------------------------------------------


def _count(mask: Optional[torch.Tensor]) -> Optional[int]:
    if mask is None:
        return None
    return int(mask.sum().item())


def _intersect_count(a: Optional[torch.Tensor], b: Optional[torch.Tensor]) -> Optional[int]:
    if a is None or b is None or a.shape != b.shape:
        return None
    return int((a & b).sum().item())


def _apply_gate(survivors: torch.Tensor, exclude_mask: Optional[torch.Tensor]):
    """Return ``(new_survivors, n_blocked_by_this_gate)``. ``exclude_mask``
    ``None`` means the gate is disabled: no-op, 0 blocked."""
    if exclude_mask is None or exclude_mask.shape != survivors.shape:
        return survivors, 0
    blocked = int((survivors & exclude_mask).sum().item())
    return survivors & ~exclude_mask, blocked


def compute_relocation_funnel(
    gaussians,
    args,
    iteration: int,
    *,
    raw_candidate_mask: torch.Tensor,
    final_eligible_mask: Optional[torch.Tensor] = None,
    render_pkg=None,
) -> dict:
    """Assemble the relocation-target candidate-survival funnel (issue #1372
    Sec. 8): lifecycle -> young-protection -> exposure -> depth-lock ->
    seed-freeze -> bubble -> visibility/locality -> final eligible.

    ``raw_candidate_mask`` is the death-candidate mask BEFORE any of those
    gates are applied (i.e. the opacity/support/scale/anisotropy/utility
    clause union). ``final_eligible_mask``, when supplied, is the mask
    actually used to pick relocation targets in production this call --
    used only to report ``final_eligible``/the survival fraction against the
    real number, not to drive the funnel's own stage-by-stage arithmetic
    (which is computed independently here so gate order/intersections are
    self-consistent even when a production mask is unavailable).

    Returns a plain dict of the headline metrics + per-gate blocked counts +
    the four gate intersections issue #1372 calls out explicitly.
    """
    n_total = int(gaussians.get_xyz.shape[0])
    survivors = raw_candidate_mask.clone()
    raw_candidates = _count(survivors) or 0

    lifecycle_mask = lifecycle_mature_mask(gaussians, args)
    young_mask = young_protected_mask(gaussians, args, iteration)
    exposure_mask = underexposed_mask(gaussians, args)
    depth_lock_mask = depth_locked_mask_readonly(gaussians, args)
    seed_freeze_mask = seed_freeze_mask_readonly(gaussians, args, iteration)
    bubble_mask = bubble_survive_mask_readonly(gaussians, args)
    bubble_exclude = (~bubble_mask) if bubble_mask is not None else None
    depth_conflict_mask = depth_conflicted_mask(gaussians)

    visible_exclude = None
    if getattr(args, "streaming_reloc_local_only", False) and render_pkg is not None:
        vis = render_pkg.get("visibility_filter") if hasattr(render_pkg, "get") else None
        if vis is not None and vis.shape == survivors.shape:
            visible_exclude = ~vis.to(survivors.device)

    stage_order = [
        ("lifecycle", lifecycle_mask),
        ("young", young_mask),
        ("exposure", exposure_mask),
        ("depth_lock", depth_lock_mask),
        ("seed_freeze", seed_freeze_mask),
        ("bubble", bubble_exclude),
        ("visibility", visible_exclude),
    ]

    survivor_counts = {"raw_candidates": raw_candidates}
    blocked_by = {}
    prev_count = raw_candidates
    for name, exclude_mask in stage_order:
        survivors, n_blocked = _apply_gate(survivors, exclude_mask)
        cur_count = _count(survivors) or 0
        # Internal consistency invariant (also asserted by the unit tests):
        # each stage's survivor count never exceeds the previous stage's.
        assert cur_count <= prev_count, f"funnel stage {name} increased survivors"
        survivor_counts[name] = cur_count
        blocked_by[f"blocked_by_{name}"] = n_blocked
        prev_count = cur_count

    final_eligible = _count(survivors)
    if final_eligible_mask is not None:
        final_eligible = _count(final_eligible_mask)

    survival_fraction = (
        float(final_eligible) / float(raw_candidates) if raw_candidates and final_eligible is not None else 0.0
    )

    intersections = {
        "mature_and_young_protected": _intersect_count(lifecycle_mask, young_mask),
        "mature_and_underexposed": _intersect_count(lifecycle_mask, exposure_mask),
        "young_protected_and_underexposed": _intersect_count(young_mask, exposure_mask),
        # issue #1365 (the canonical belief/population-control migration plan
        # that formally specifies this telemetry as its §17.1 "ownership/
        # conflict shadow telemetry", superseding issue #1372) renames MATURE
        # to the derived-diagnostic-only label ESTABLISHED for this specific
        # intersection. `lifecycle_mask` itself is still literally sourced
        # from `gaussians.lifecycle_state >= 2` (today's MATURE/FROZEN
        # predicate, unchanged) -- only the reported label is forward-facing.
        "established_and_depth_conflicted": _intersect_count(lifecycle_mask, depth_conflict_mask),
    }

    return {
        "iteration": int(iteration),
        "n_total": n_total,
        "survivor_counts": survivor_counts,
        **blocked_by,
        "final_eligible": final_eligible,
        "mcmc_relocation_candidate_survival_fraction": survival_fraction,
        "intersections": intersections,
    }


def compute_growth_funnel(
    gaussians,
    args,
    iteration: int,
    *,
    current_n: Optional[int] = None,
    final_eligible_mask: Optional[torch.Tensor] = None,
) -> dict:
    """Assemble the growth-parent candidate-survival funnel (issue #1372
    Sec. 8): depth-lock -> bubble -> recent-visibility -> seed-freeze ->
    exposure -> final growth parents.
    """
    n_total = int(current_n if current_n is not None else gaussians.get_xyz.shape[0])
    device = gaussians.get_xyz.device
    survivors = torch.ones(n_total, dtype=torch.bool, device=device)

    depth_lock_mask = depth_locked_mask_readonly(gaussians, args)
    bubble_mask = bubble_survive_mask_readonly(gaussians, args)
    bubble_exclude = (~bubble_mask) if bubble_mask is not None else None
    seed_freeze_mask = seed_freeze_mask_readonly(gaussians, args, iteration)
    exposure_mask = underexposed_mask(gaussians, args)

    visibility_exclude = None
    if getattr(args, "streaming_growth_parents_visible_only", False):
        vema = getattr(gaussians, "visibility_ema", None)
        if vema is not None and vema.shape[0] == n_total:
            visibility_exclude = ~(vema.squeeze(-1) > 0.01)

    stage_order = [
        ("depth_lock", depth_lock_mask),
        ("bubble", bubble_exclude),
        ("visibility", visibility_exclude),
        ("seed_freeze", seed_freeze_mask),
        ("exposure", exposure_mask),
    ]

    survivor_counts = {"n_total": n_total}
    blocked_by = {}
    prev_count = n_total
    for name, exclude_mask in stage_order:
        survivors, n_blocked = _apply_gate(survivors, exclude_mask)
        cur_count = _count(survivors) or 0
        assert cur_count <= prev_count, f"funnel stage {name} increased survivors"
        survivor_counts[name] = cur_count
        blocked_by[f"blocked_by_{name}"] = n_blocked
        prev_count = cur_count

    final_eligible = _count(survivors)
    if final_eligible_mask is not None:
        final_eligible = _count(final_eligible_mask)

    survival_fraction = (
        float(final_eligible) / float(n_total) if n_total and final_eligible is not None else 0.0
    )

    return {
        "iteration": int(iteration),
        "n_total": n_total,
        "survivor_counts": survivor_counts,
        **blocked_by,
        "final_eligible": final_eligible,
        "mcmc_growth_parent_survival_fraction": survival_fraction,
    }


# ---------------------------------------------------------------------------
# TensorBoard logging + cumulative history (for report.json / a dedicated
# JSON artifact) -- module-level accumulator mirroring the existing
# ``utils.energy_mcmc._pruning_stats`` pattern.
# ---------------------------------------------------------------------------

_funnel_history: list[dict] = []


def reset_funnel_history() -> None:
    _funnel_history.clear()


def get_funnel_history() -> list[dict]:
    return list(_funnel_history)


def _native_host_int(value: Any, *, field: str) -> int:
    """Accept only an actual Python ``int`` at the async host boundary.

    ``int(torch_scalar)`` and ``int(numpy_scalar)`` can synchronously copy or
    synchronize a device value.  Proxy objects are equally unsafe because
    they can hide that operation behind ``__int__``.  Device reducers must
    publish native host counters before calling this diagnostic sink.
    """
    if type(value) is not int:  # deliberate: reject bool, tensors, numpy, proxies
        raise TypeError(f"{field} must be a native host integer")
    return value


def _validated_compact_result(result: Mapping[str, Any]) -> dict:
    """Validate/normalize counters emitted by an asynchronous device reducer.

    This is the preferred ingestion boundary for production telemetry.  The
    reducer owns any device-to-host cardinality transfer; this function only
    handles already-compact scalar counters and therefore never touches a
    tensor or calls ``.item()``.
    """
    if not isinstance(result, Mapping):
        raise TypeError("compact funnel result must be a mapping")
    try:
        n_total = _native_host_int(result["n_total"], field="n_total")
        stages = dict(result["survivor_counts"])
        final = _native_host_int(result["final_eligible"], field="final_eligible")
    except KeyError as exc:
        raise ValueError("compact funnel result is missing scalar counters") from exc
    if n_total < 0 or final < 0:
        raise ValueError("compact funnel counts must be non-negative")
    values = []
    for name, value in stages.items():
        count = _native_host_int(value, field=f"survivor_counts[{name!r}]")
        if count < 0:
            raise ValueError("compact funnel counts must be non-negative")
        if count > n_total:
            raise ValueError(f"survivor stage {name!r} exceeds population")
        values.append(count)
    if any(cur > prev for prev, cur in zip(values, values[1:])):
        raise ValueError("compact funnel survivor counts must be monotonic")
    if "raw_candidates" in stages and stages["raw_candidates"] > n_total:
        raise ValueError("raw_candidates exceeds population")
    if "n_total" in stages and stages["n_total"] != n_total:
        raise ValueError("growth n_total stage must equal population")
    if final > (values[-1] if values else n_total):
        raise ValueError("final_eligible exceeds the final funnel stage")
    if n_total == 0 and final:
        raise ValueError("final_eligible cannot exceed an empty population")

    normalized = dict(result)
    normalized["n_total"] = n_total
    normalized["survivor_counts"] = {str(k): int(v) for k, v in stages.items()}
    normalized["final_eligible"] = final
    raw = normalized["survivor_counts"].get(
        "raw_candidates", normalized["survivor_counts"].get("n_total", n_total)
    )
    if "raw_candidates" in normalized["survivor_counts"]:
        normalized.setdefault(
            "mcmc_relocation_candidate_survival_fraction",
            float(final) / float(raw) if raw else 0.0,
        )
    elif "mcmc_growth_parent_survival_fraction" not in normalized:
        normalized["mcmc_growth_parent_survival_fraction"] = (
            float(final) / float(n_total) if n_total else 0.0
        )
    if "intersections" in normalized:
        intersections = {}
        for key, value in dict(normalized["intersections"]).items():
            if value is None:
                intersections[str(key)] = None
                continue
            intersection = _native_host_int(value, field=f"intersections[{key!r}]")
            if intersection < 0 or intersection > n_total:
                raise ValueError(f"intersection {key!r} is outside population bounds")
            intersections[str(key)] = intersection
        normalized["intersections"] = intersections
    for key, value in normalized.items():
        if key.startswith("blocked_by_"):
            blocked = _native_host_int(value, field=key)
            if blocked < 0 or blocked > n_total:
                raise ValueError(f"{key} is outside population bounds")
            normalized[key] = blocked
    return normalized


def _append_history(sample: dict) -> None:
    _funnel_history.append(sample)
    if len(_funnel_history) > MAX_FUNNEL_HISTORY:
        del _funnel_history[: len(_funnel_history) - MAX_FUNNEL_HISTORY]


def record_compact_sample(
    relocation: Optional[Mapping[str, Any]],
    growth: Optional[Mapping[str, Any]],
) -> None:
    """Record bounded counters produced outside the normal GPU path.

    ``relocation`` and ``growth`` are intentionally mappings of scalar
    counters, not masks.  A runtime can pass an asynchronously completed
    debug/telemetry payload here without re-running candidate gates or
    introducing host synchronization into the MCMC transaction.
    """
    if relocation is None and growth is None:
        return
    _append_history({
        "relocation": None if relocation is None else _validated_compact_result(relocation),
        "growth": None if growth is None else _validated_compact_result(growth),
    })


def record_sample(relocation: Optional[dict], growth: Optional[dict]) -> None:
    if relocation is None and growth is None:
        return
    _append_history({"relocation": relocation, "growth": growth})


def log_to_tensorboard(tb_writer, iteration: int, relocation: Optional[dict], growth: Optional[dict]) -> None:
    if tb_writer is None:
        return
    if relocation is not None:
        tb_writer.add_scalar(
            "streaming/mcmc_funnel/relocation_candidate_survival_fraction",
            relocation["mcmc_relocation_candidate_survival_fraction"],
            iteration,
        )
        tb_writer.add_scalar("streaming/mcmc_funnel/relocation_final_eligible", relocation["final_eligible"] or 0, iteration)
        for key, value in relocation.items():
            if key.startswith("blocked_by_"):
                tb_writer.add_scalar(f"streaming/mcmc_funnel/reloc_{key}", value, iteration)
        for key, value in relocation["intersections"].items():
            if value is not None:
                tb_writer.add_scalar(f"streaming/mcmc_funnel/reloc_intersect_{key}", value, iteration)
    if growth is not None:
        tb_writer.add_scalar(
            "streaming/mcmc_funnel/growth_parent_survival_fraction",
            growth["mcmc_growth_parent_survival_fraction"],
            iteration,
        )
        tb_writer.add_scalar("streaming/mcmc_funnel/growth_final_eligible", growth["final_eligible"] or 0, iteration)
        for key, value in growth.items():
            if key.startswith("blocked_by_"):
                tb_writer.add_scalar(f"streaming/mcmc_funnel/growth_{key}", value, iteration)


def summarize_history() -> Optional[dict]:
    """Cheap last-sample + mean summary for report.json. ``None`` if empty."""
    if not _funnel_history:
        return None
    last = _funnel_history[-1]
    reloc_fracs = [
        s["relocation"]["mcmc_relocation_candidate_survival_fraction"]
        for s in _funnel_history
        if s["relocation"] is not None
    ]
    growth_fracs = [
        s["growth"]["mcmc_growth_parent_survival_fraction"]
        for s in _funnel_history
        if s["growth"] is not None
    ]
    return {
        "n_samples": len(_funnel_history),
        "last_sample": last,
        "mean_relocation_candidate_survival_fraction": (
            sum(reloc_fracs) / len(reloc_fracs) if reloc_fracs else None
        ),
        "mean_growth_parent_survival_fraction": (
            sum(growth_fracs) / len(growth_fracs) if growth_fracs else None
        ),
    }
