"""W3-11 deliverable 5: peak memory + mutation latency, fixed-capacity vs.
dynamic torch.cat baseline.

Standalone GPU microbenchmark -- NOT a full training/champion-scene run.
Drives a representative sequence of insert/retire mutation batches at
streaming-realistic sizes (matching --streaming_max_new_gaussians_per_frame's
default of 2000 rows/insert, per CLAUDE.md) against two isolated
implementations of the exact same growth mechanism GsplatGaussianModel
uses today:

  * "dynamic": one torch.cat + new nn.Parameter + optimizer-state cat per
    insert batch, mirroring GsplatGaussianModel.cat_tensors_to_optimizer
    (scene/gsplat_model.py) exactly (same zero-momentum-extension logic).
  * "fixed_capacity": eligsir.optimization.gaussian.storage
    .capacity_registry.CapacityBufferRegistry, pre-sized to the scenario's
    peak N so inserts land in free capacity after the first growth event.

W3-49 (2026-08-10, two-part) added three more benchmark arms:

  * "optimizer_step_scaling": holds n_active fixed and sweeps capacity,
    timing repeated real optimizer.step() calls. Against the ORIGINAL
    (pre-redesign) registry this measured a real 6.92x mean step-time
    regression at a 20x capacity/active ratio -- see
    docs/architecture/2026-08-10-w3-49-capacity-scaling-baseline.md. THIS
    FILE now targets the REDESIGNED (packed-contiguous-active-prefix)
    registry, whose optimizer-visible Parameter is always exactly
    ``epoch_capacity``-sized regardless of ``capacity`` -- the acceptance
    gate is ratio < 1.05, mirroring upstream gsplat issue #444's own
    ~1.01x bar for the equivalent fix.
  * "active_view_vs_slice": compares CapacityBufferRegistry.active_view()
    against a plain contiguous slice. Against the original registry this
    measured a real 43-74x gather cost (active_view() did a boolean-mask
    gather). The redesigned active_view() IS a plain slice internally now
    (``buffer[:epoch_capacity]``) -- this arm demonstrates the two are
    equal (both cheap, both real slices), not that one is now faster than
    the other by some multiplier.
  * "compaction_cost": NEW in the redesign -- retires arbitrary fractions
    (0.1%/1%/5%/10%/25%) of a realistic 300k-Gaussian map (capacity
    500k) via the REAL CapacityBufferRegistry.retire() call (not a manual
    replication of its internals -- see the W3-52 correction below),
    reporting total mutation-barrier latency, a decomposition of that cost
    into "cold" (momentum synced from the optimizer for the first time)
    vs. "warm" (momentum backing already aliased, sync short-circuits) on
    a second retire() at the same fraction, and confirms cost scales with
    retirement batch size (rows actually moved), not with capacity or
    total live count -- including an explicit capacity-independence sweep
    (fixed N/fixed fraction, capacity swept 300k-1M).

    **Correction (W3-52, post-#78-audit)**: the original version of this
    arm did NOT call ``CapacityBufferRegistry.retire()`` at all -- it
    called the lower-level ``SlotAllocator.retire()`` directly and then
    manually replicated buffer/momentum movement, skipping the
    ``sync_momentum_from_optimizer()`` step ``reg.retire()`` performs
    before compaction (the exact step W3-49's own TDD process found was
    necessary -- see that PR's "bug found and fixed" writeup). Because
    ``_momentum_backing`` was never populated, ``apply_compaction()``'s
    loop over it was a silent no-op, so the originally-published
    "buffer+momentum movement stays under ~0.15ms" number did not include
    real momentum-sync cost. Fixed to call ``reg.retire()`` directly; see
    docs/architecture/2026-08-10-w3-52-p0-readiness-fixes.md.

Run inside the container (as a MODULE, not a plain script -- see the note
in docs/architecture/2026-08-10-w3-49-capacity-scaling-baseline.md: running
this file directly puts its own directory on sys.path, not the repo root,
so `import eligsir` fails):
    docker compose run --rm train python -m scripts.bench_capacity_registry
"""

from __future__ import annotations

import json
import time

import torch
from torch import nn

from eligsir.optimization.gaussian.storage.capacity_registry import (
    CapacityBufferRegistry,
    SlotAllocator,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FEATURE_DIM = 3  # stand-in for "means"-shaped rows


def _adam_factory(lr: float = 0.01):
    return lambda params: torch.optim.Adam(params, lr=lr, eps=1e-15)


# Scenario: bootstrap at 30k, insert 2000-row batches ("streaming replay",
# --streaming_max_new_gaussians_per_frame default) until 150k, retiring a
# ~10% skimming batch every 5th insert to also exercise slot reuse.
BOOTSTRAP_N = 30_000
INSERT_BATCH = 2_000
N_INSERTS = 60  # 30k + 60*2000 = 150k peak
RETIRE_EVERY = 5
RETIRE_FRACTION = 0.10


def _sync():
    if DEVICE == "cuda":
        torch.cuda.synchronize()


def _reset_mem():
    if DEVICE == "cuda":
        torch.cuda.reset_peak_memory_stats()


def _peak_mem_mb():
    if DEVICE != "cuda":
        return 0.0
    return torch.cuda.max_memory_allocated() / (1024 ** 2)


# ---------------------------------------------------------------------------
# Dynamic (torch.cat) baseline -- mirrors GsplatGaussianModel.cat_tensors_to_optimizer
# ---------------------------------------------------------------------------


def _cat_tensors_to_optimizer(param: nn.Parameter, optimizer, ext: torch.Tensor) -> nn.Parameter:
    group = optimizer.param_groups[0]
    old_param = group["params"][0]
    stored_state = optimizer.state.get(old_param, None)
    n_new = ext.shape[0]
    new_param = nn.Parameter(torch.cat([old_param.detach(), ext.detach()], dim=0).requires_grad_(True))
    if stored_state is not None:
        new_state = {}
        for k, v in stored_state.items():
            if isinstance(v, torch.Tensor) and v.dim() == ext.dim():
                zeros = torch.zeros(n_new, *ext.shape[1:], dtype=v.dtype, device=v.device)
                new_state[k] = torch.cat([v, zeros], dim=0)
            else:
                new_state[k] = v
        del optimizer.state[old_param]
        optimizer.state[new_param] = new_state
    group["params"][0] = new_param
    return new_param


def run_dynamic_baseline() -> dict:
    torch.manual_seed(0)
    _reset_mem()
    means = nn.Parameter(torch.zeros(BOOTSTRAP_N, FEATURE_DIM, device=DEVICE))
    optimizer = torch.optim.Adam([means], lr=0.01, eps=1e-15)
    # warm the optimizer state so extension has real momentum to carry.
    means.grad = torch.ones_like(means)
    optimizer.step()

    insert_times_ms = []
    live_n = BOOTSTRAP_N
    for i in range(N_INSERTS):
        ext = torch.ones(INSERT_BATCH, FEATURE_DIM, device=DEVICE)
        _sync()
        t0 = time.perf_counter()
        means = _cat_tensors_to_optimizer(means, optimizer, ext)
        _sync()
        insert_times_ms.append((time.perf_counter() - t0) * 1e3)
        live_n += INSERT_BATCH

        if (i + 1) % RETIRE_EVERY == 0:
            n_retire = int(live_n * RETIRE_FRACTION)
            keep_mask = torch.ones(live_n, dtype=torch.bool, device=DEVICE)
            keep_mask[:n_retire] = False  # "prune" the oldest rows, dynamic path's only tool
            param = optimizer.param_groups[0]["params"][0]
            stored_state = optimizer.state.get(param, None)
            new_param = nn.Parameter(param[keep_mask].detach().requires_grad_(True))
            if stored_state is not None:
                new_state = {}
                for k, v in stored_state.items():
                    if isinstance(v, torch.Tensor) and v.dim() > 0:
                        new_state[k] = v[keep_mask]
                    else:
                        new_state[k] = v
                del optimizer.state[param]
                optimizer.state[new_param] = new_state
            optimizer.param_groups[0]["params"][0] = new_param
            means = new_param
            live_n = int(keep_mask.sum().item())

    return {
        "path": "dynamic_torch_cat",
        "final_n": live_n,
        "peak_mem_mb": _peak_mem_mb(),
        "mean_insert_ms": sum(insert_times_ms) / len(insert_times_ms),
        "max_insert_ms": max(insert_times_ms),
        "total_insert_ms": sum(insert_times_ms),
    }


# ---------------------------------------------------------------------------
# Fixed-capacity path (redesigned packed-prefix registry)
# ---------------------------------------------------------------------------


def run_fixed_capacity() -> dict:
    torch.manual_seed(0)
    _reset_mem()
    # Pre-size capacity to the scenario peak so, after one initial growth to
    # get there, every subsequent insert lands in free capacity -- this is
    # the realistic operating point for a fixed-capacity allocator (sized
    # against a known cap_max), not "always start tiny and grow every time."
    peak_n = BOOTSTRAP_N + N_INSERTS * INSERT_BATCH
    alloc = SlotAllocator(peak_n, device=DEVICE, debug_checks=False)
    reg = CapacityBufferRegistry(alloc)
    means_backing = torch.zeros(peak_n, FEATURE_DIM, device=DEVICE)
    optimizer = reg.register_param("means", means_backing, _adam_factory())

    reg.insert({"means": torch.zeros(BOOTSTRAP_N, FEATURE_DIM, device=DEVICE)})
    param = reg.param_for("means")
    param.grad = torch.ones_like(param)
    optimizer.step()

    insert_times_ms = []
    for i in range(N_INSERTS):
        ext = torch.ones(INSERT_BATCH, FEATURE_DIM, device=DEVICE)
        _sync()
        t0 = time.perf_counter()
        reg.insert({"means": ext})
        _sync()
        insert_times_ms.append((time.perf_counter() - t0) * 1e3)

        if (i + 1) % RETIRE_EVERY == 0:
            active_idx = alloc.active_physical_indices()
            n_retire = int(active_idx.numel() * RETIRE_FRACTION)
            reg.retire(active_idx[:n_retire])

    return {
        "path": "fixed_capacity_registry",
        "final_n": alloc.n_active,
        "capacity": alloc.capacity,
        "growth_events": reg.growth_events,
        "peak_mem_mb": _peak_mem_mb(),
        "mean_insert_ms": sum(insert_times_ms) / len(insert_times_ms),
        "max_insert_ms": max(insert_times_ms),
        "total_insert_ms": sum(insert_times_ms),
    }


# ---------------------------------------------------------------------------
# W3-49 acceptance gate 1: optimizer-step-vs-capacity scaling
# ---------------------------------------------------------------------------

OPT_SCALING_N_ACTIVE = 50_000
OPT_SCALING_CAPACITIES = [50_000, 100_000, 250_000, 500_000, 1_000_000]
OPT_SCALING_N_STEPS = 300


def run_optimizer_step_scaling() -> dict:
    """Hold n_active fixed, sweep capacity, time repeated real optimizer.step()
    calls against the REDESIGNED (packed-active-prefix) registry. The
    optimizer-visible Parameter is always exactly ``epoch_capacity``-sized
    (== n_active by this PR's default policy) regardless of ``capacity`` --
    acceptance gate is ratio < 1.05 (mirrors upstream gsplat issue #444's
    own ~1.01x bar). Pre-redesign measurement was 6.92x at the same 20x
    capacity/active ratio -- see
    docs/architecture/2026-08-10-w3-49-capacity-scaling-baseline.md.
    """
    results = []
    for capacity in OPT_SCALING_CAPACITIES:
        torch.manual_seed(0)
        alloc = SlotAllocator(capacity, device=DEVICE, debug_checks=False)
        reg = CapacityBufferRegistry(alloc)
        means_backing = torch.zeros(capacity, FEATURE_DIM, device=DEVICE)
        optimizer = reg.register_param("means", means_backing, _adam_factory())
        reg.insert({"means": torch.randn(OPT_SCALING_N_ACTIVE, FEATURE_DIM, device=DEVICE)})

        param = reg.param_for("means")
        # Grad set once, reused every step -- this benchmark times
        # optimizer.step() specifically, not gradient population.
        param.grad = torch.ones_like(param)
        optimizer.step()  # warmup: give Adam real exp_avg/exp_avg_sq state
        _sync()

        step_times_ms = []
        for _ in range(OPT_SCALING_N_STEPS):
            _sync()
            t0 = time.perf_counter()
            optimizer.step()
            _sync()
            step_times_ms.append((time.perf_counter() - t0) * 1e3)

        results.append({
            "capacity": capacity,
            "n_active": OPT_SCALING_N_ACTIVE,
            "optimizer_visible_shape": int(reg.param_for("means").shape[0]),
            "mean_step_ms": sum(step_times_ms) / len(step_times_ms),
            "median_step_ms": sorted(step_times_ms)[len(step_times_ms) // 2],
            "max_step_ms": max(step_times_ms),
        })

    baseline_ms = results[0]["mean_step_ms"]
    for r in results:
        r["ratio_vs_min_capacity"] = (
            r["mean_step_ms"] / baseline_ms if baseline_ms > 0 else None
        )

    return {"n_steps": OPT_SCALING_N_STEPS, "n_active_held_fixed": OPT_SCALING_N_ACTIVE, "results": results}


# ---------------------------------------------------------------------------
# W3-49 acceptance gate 2: active_view() vs. a plain contiguous slice
# ---------------------------------------------------------------------------

GATHER_CAPACITY = 600_000
GATHER_RATIOS = [1.0, 0.833, 0.5, 0.167]
GATHER_N_REPS = 50


def run_active_view_vs_slice() -> dict:
    """At several active/capacity ratios, compare CapacityBufferRegistry
    .active_view() against a plain [:n] contiguous slice. Against the
    ORIGINAL registry this measured a real 43-74x gather cost (active_view()
    did a boolean-mask gather, a materializing op). The REDESIGNED
    active_view() IS `buffer[:epoch_capacity]` internally now -- this arm
    demonstrates the two costs converge (both are real slices), i.e. the
    hot-path gather this task's filing note flagged is gone, not that one
    got some multiplier faster than the other.
    """
    results = []
    for ratio in GATHER_RATIOS:
        n_active = int(GATHER_CAPACITY * ratio)
        alloc = SlotAllocator(GATHER_CAPACITY, device=DEVICE, debug_checks=False)
        reg = CapacityBufferRegistry(alloc)
        buf = torch.randn(GATHER_CAPACITY, FEATURE_DIM, device=DEVICE)
        reg.register_buffer("means", buf)
        alloc.allocate(n_active)

        _sync()
        t0 = time.perf_counter()
        for _ in range(GATHER_N_REPS):
            _ = reg.active_view("means")
        _sync()
        active_view_ms = (time.perf_counter() - t0) * 1e3 / GATHER_N_REPS

        _sync()
        t0 = time.perf_counter()
        for _ in range(GATHER_N_REPS):
            _ = buf[:n_active]
        _sync()
        slice_ms = (time.perf_counter() - t0) * 1e3 / GATHER_N_REPS

        results.append({
            "capacity": GATHER_CAPACITY,
            "n_active": n_active,
            "ratio": ratio,
            "active_view_ms": active_view_ms,
            "slice_ms": slice_ms,
        })

    return {"n_reps": GATHER_N_REPS, "results": results}


# ---------------------------------------------------------------------------
# W3-49 acceptance gate 3: compaction cost at various retirement fractions
# ---------------------------------------------------------------------------

COMPACTION_CAPACITY = 500_000
COMPACTION_N_ACTIVE = 300_000
COMPACTION_RETIRE_FRACTIONS = [0.001, 0.01, 0.05, 0.10, 0.25]
COMPACTION_CAPACITY_SWEEP = [300_000, 500_000, 750_000, 1_000_000]
COMPACTION_SWEEP_N_ACTIVE = 200_000
COMPACTION_SWEEP_FRACTION = 0.05


def _scattered_indices(n_active: int, n_retire: int) -> torch.Tensor:
    """Evenly-scattered (worst-case-ish, not tail-clustered) physical
    indices across the active prefix -- maximizes donor-move count for a
    given retirement size (a tail-only retirement needs zero donors, which
    would understate real cost).
    """
    step = max(1, n_active // n_retire)
    return torch.arange(0, min(n_retire * step, n_active), step, device=DEVICE)[:n_retire]


def _setup_compaction_registry(capacity: int, n_active: int):
    torch.manual_seed(0)
    alloc = SlotAllocator(capacity, device=DEVICE, debug_checks=False)
    reg = CapacityBufferRegistry(alloc)
    means_backing = torch.zeros(capacity, FEATURE_DIM, device=DEVICE)
    optimizer = reg.register_param("means", means_backing, _adam_factory())
    reg.insert({"means": torch.randn(n_active, FEATURE_DIM, device=DEVICE)})
    param = reg.param_for("means")
    param.grad = torch.ones_like(param)
    optimizer.step()  # give every row real momentum, so compaction has real state to move
    return alloc, reg, optimizer


def _timed_retire(reg: CapacityBufferRegistry, retire_idx: torch.Tensor):
    _sync()
    t0 = time.perf_counter()
    plan = reg.retire(retire_idx)
    _sync()
    return plan, (time.perf_counter() - t0) * 1e3


def run_compaction_cost() -> dict:
    """Retire arbitrary (evenly-scattered, worst-case-ish -- not
    tail-clustered) physical rows out of a realistic 300k-live/500k-capacity
    map at several retirement fractions, via the REAL
    ``CapacityBufferRegistry.retire()`` call (not a manual replication of
    its internals -- see this module's docstring "Correction" note). Cost
    should scale with the number of ROWS RETIRED (specifically, the
    ``dst_indices`` count -- holes below the new prefix boundary that need
    a donor), not with capacity or n_active.

    Each fraction is measured on the SAME index set, decomposed into two
    directly-timed phases rather than compared across two different retire()
    calls (an earlier version of this benchmark compared a "cold" retire
    against a "warm" retire on a smaller, disjoint index set -- the
    resulting sizes differed, so ``cold_ms - warm_ms`` mixed the sync cost
    with a genuinely different workload; that comparison is not published
    here for that reason):

    1. ``sync_momentum_from_optimizer()`` timed IN ISOLATION -- the real
       cost of aliasing the optimizer's live momentum state into the
       capacity-sized backing for the first time (this pays for itself once
       per param per "cold start"; a data_ptr() equality check makes every
       later call on the same param a no-op).
    2. ``reg.retire(retire_idx)`` timed immediately after -- since step 1
       already synced, retire()'s own internal sync call short-circuits, so
       this measures allocator bookkeeping + buffer/momentum move + rebind
       only, with no sync cost mixed in.

    ``total_first_retire_ms = sync_ms + retire_ms`` is what the very first
    retire() after a step() actually costs in production (retire() would
    have performed the same sync internally); reporting the two phases
    separately, on the SAME index set, is the honest decomposition -- see
    docs/architecture/2026-08-10-w3-52-p0-readiness-fixes.md.
    """
    results = []
    for fraction in COMPACTION_RETIRE_FRACTIONS:
        alloc, reg, optimizer = _setup_compaction_registry(COMPACTION_CAPACITY, COMPACTION_N_ACTIVE)
        n_retire = max(1, int(COMPACTION_N_ACTIVE * fraction))
        retire_idx = _scattered_indices(alloc.n_active, n_retire)

        # Track one specific Gaussian across the retire to verify its
        # momentum followed it to its new physical row, not just that
        # SOME tensor got moved somewhere.
        tracked_logical_id = int(alloc.logical_ids[alloc.n_active - 1].item())
        tracked_momentum_before = optimizer.state[reg.param_for("means")]["exp_avg"][
            alloc.n_active - 1
        ].clone()

        binding = reg.bindings["means"]
        _sync()
        t_sync0 = time.perf_counter()
        binding.sync_momentum_from_optimizer()
        _sync()
        sync_ms = (time.perf_counter() - t_sync0) * 1e3

        plan, retire_ms = _timed_retire(reg, retire_idx)

        tracked_physical_after = alloc.logical_to_physical[tracked_logical_id]
        tracked_momentum_after = optimizer.state[reg.param_for("means")]["exp_avg"][
            tracked_physical_after
        ]
        momentum_followed_gaussian = bool(
            torch.equal(tracked_momentum_before, tracked_momentum_after)
        )

        results.append({
            "retire_fraction": fraction,
            "n_retired": n_retire,
            "n_donors_moved": int(plan.dst_indices.numel()),
            "momentum_sync_ms": sync_ms,
            "retire_after_sync_ms": retire_ms,
            "total_first_retire_ms": sync_ms + retire_ms,
            "momentum_followed_gaussian_through_compaction": momentum_followed_gaussian,
        })

    capacity_sweep_results = []
    for capacity in COMPACTION_CAPACITY_SWEEP:
        alloc, reg, optimizer = _setup_compaction_registry(capacity, COMPACTION_SWEEP_N_ACTIVE)
        n_retire = max(1, int(COMPACTION_SWEEP_N_ACTIVE * COMPACTION_SWEEP_FRACTION))
        retire_idx = _scattered_indices(alloc.n_active, n_retire)
        plan, retire_ms = _timed_retire(reg, retire_idx)
        capacity_sweep_results.append({
            "capacity": capacity,
            "n_active_before_retire": COMPACTION_SWEEP_N_ACTIVE,
            "n_retired": n_retire,
            "n_donors_moved": int(plan.dst_indices.numel()),
            "retire_ms": retire_ms,
        })

    return {
        "capacity": COMPACTION_CAPACITY,
        "n_active_before_retire": COMPACTION_N_ACTIVE,
        "index_pattern": "evenly-scattered across the active prefix (worst-case-ish donor count)",
        "path": "reg.retire() -- the real production path, not a manual replication",
        "results": results,
        "capacity_independence_sweep": {
            "n_active": COMPACTION_SWEEP_N_ACTIVE,
            "retire_fraction": COMPACTION_SWEEP_FRACTION,
            "note": "N and retirement fraction held fixed, capacity swept -- cost should stay flat",
            "results": capacity_sweep_results,
        },
    }


if __name__ == "__main__":
    if DEVICE != "cuda":
        print("[bench] WARNING: no CUDA/ROCm device found -- results are CPU-only and not representative.")
    dynamic = run_dynamic_baseline()
    fixed = run_fixed_capacity()
    optimizer_scaling = run_optimizer_step_scaling()
    active_view_vs_slice = run_active_view_vs_slice()
    compaction_cost = run_compaction_cost()
    report = {"device": DEVICE, "scenario": {
        "bootstrap_n": BOOTSTRAP_N, "insert_batch": INSERT_BATCH, "n_inserts": N_INSERTS,
        "retire_every": RETIRE_EVERY, "retire_fraction": RETIRE_FRACTION,
    }, "dynamic": dynamic, "fixed_capacity": fixed,
        "optimizer_step_scaling": optimizer_scaling,
        "active_view_vs_slice": active_view_vs_slice,
        "compaction_cost": compaction_cost}
    print(json.dumps(report, indent=2))
