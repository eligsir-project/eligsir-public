"""Lightweight per-phase ingest profiler (env-gated; SPLAT_PROFILE_PHASES=1).

Extracted (2026-08-13, issue #274) from train_streaming.py into its own leaf
module -- no torch/gsplat/viewer dependency chain -- so it can be:
  (a) imported directly by scene/streaming_scene.py's
      StreamingScene.accumulate_observed_surface() sub-stage timers without a
      circular import onto the much heavier train_streaming module, and
  (b) unit-tested on CPU without importing train_streaming.py at all (which
      pulls in torch/torchvision/gsplat and, on at least one host toolchain
      seen during this work, a broken torch/torchvision ABI mismatch that has
      nothing to do with this profiler).

`torch` itself is still imported, but lazily inside `_timed`'s hot path (only
when SPLAT_PROFILE_PHASES is set) and only for `torch.cuda.is_available()` /
`torch.cuda.synchronize()` -- no torchvision, no gsplat.

Two parallel stores are kept, deliberately:
  _PHASE_OPS          -- "windowed" accumulator, cleared every ~100
                          iterations by train_streaming.PhaseTimer's
                          report_window() for the rolling [timing-window] log
                          line. Cheap, but NOT a full-run total -- do not
                          treat it as one.
  _PHASE_OPS_LIFETIME  -- accumulates for the WHOLE run (sum/count/max plus a
                          bounded recent-sample deque per name, for
                          p50/p95/p99). Never cleared by report_window().
                          This is the authoritative store for
                          _dump_phase_ops()'s final summary and for the JSON
                          artifact written at the end of a profiled run.

Before this fix (issue #274), _dump_phase_ops() read straight from
_PHASE_OPS, so a run's "final" printed numbers were silently only whatever
had fired since the LAST report_window() call (every 100 iterations), not
the whole run -- a real trustworthiness gap for anyone reading that final
line as a run-total.

_PHASE_COUNTS mirrors this for workload counters (points, valid samples,
unique voxel keys, ...) recorded via record_count() alongside the timers so
latency can be normalized by workload size, not just read as raw ms.
"""

from __future__ import annotations

import collections
import json
import os
import time

_PROFILE_PHASES = bool(os.environ.get("SPLAT_PROFILE_PHASES"))
_TORCH_RANGES = bool(os.environ.get("SPLAT_TORCH_PROFILE_TICKS"))

# --- Duty-cycle timeline hook (issue #562) ---------------------------------
# A `utils.duty_cycle.DutyCycleTimeline` (or None). Injected by
# set_duty_cycle_recorder() so this module stays a leaf: it never imports
# duty_cycle, and duty_cycle never imports this.
#
# When no recorder is installed (the default), the ONLY cost `_timed` pays for
# #562 is one module-global load and one `is not None` branch per region
# enter/exit -- no clock read, no dict write, no call. See `_timed.__enter__`
# for why this is kept separate from `_PROFILE_PHASES` rather than fused into a
# single cached flag. tests/test_issue562_duty_cycle_telemetry.py pins the
# "no work when off" property and measures the enabled cost.
_DUTY = None
_DUTY_FAILED = False


def set_duty_cycle_recorder(recorder) -> None:
    """Install (or clear, with None) the duty-cycle timeline `_timed` feeds."""
    global _DUTY, _DUTY_FAILED
    _DUTY = recorder
    _DUTY_FAILED = False


def duty_cycle_recorder():
    """The installed recorder, or None."""
    return _DUTY


def _duty_event(is_enter: bool, name: str) -> None:
    """Forward one region boundary to the timeline, never raising.

    A telemetry failure must not mask the training exception that is already
    propagating through `_timed.__exit__` -- same swallow-and-warn-once
    convention as AsyncSaveWorker and #399's snapshot failure path.
    """
    global _DUTY_FAILED
    rec = _DUTY
    if rec is None:
        return
    try:
        if is_enter:
            rec.enter(name)
        else:
            rec.exit(name)
    except Exception as e:  # pragma: no cover - defensive
        if not _DUTY_FAILED:
            _DUTY_FAILED = True
            print(f"[duty-cycle] WARNING: disabled after error in region {name!r}: {e}", flush=True)
        set_duty_cycle_recorder(None)
_PHASE_OPS: dict[str, list] = {}  # name -> [sum_s, count, max_s]
_PHASE_SAMPLE_CAP = 4096  # bounded memory even across an hours-long run
_PHASE_OPS_LIFETIME: dict[str, list] = {}  # name -> [sum_s, count, max_s, deque[float]]
_PHASE_COUNTS: dict[str, list] = {}  # name -> [sum, count]

# Declared parent/child nesting between _timed() regions. Some regions are
# measured INSIDE another region, so their times are already counted in the
# parent's total and the flat table deliberately sums past 100% of wall-time.
# Left implicit, a downstream consumer (issue #275's matrix in particular) has
# no way to detect that from the flat, sum-sorted output and will double-count.
# Declared here rather than inferred from the "/" in a name so that adding a
# name with a slash cannot silently and wrongly imply containment.
#   key   = name prefix identifying the children
#   value = the region whose _timed() block encloses them
# ingest_accumulate_surface (scene/streaming_scene.py:2212,2235) wraps every
# surface/* region (streaming_scene.py:1401-1468).
#
# Issue #279 follow-up (2026-08-31): production_application.py's
# CanonicalMappingRuntime.tick() wraps the whole causal_mapping GPU
# transaction (execute_fixed_topology_transaction(), which dispatches into
# engine.py's run_step()/run_batched_step()) in "m_optimization_transaction".
# engine.py's own render/loss/backward/optimizer/noise brackets, plus the
# CPU-side per-camera request-assembly/prefetch brackets added in
# eligsir/mapping/optimization_runtime.py, all fire INSIDE that call and
# are therefore children of it, not siblings of the other m_tick_* regions
# (camera materialize/prefetch/prepare, envelope build, post-step
# bookkeeping, mutation) that run outside it in the same tick.
#
# Issue #279 outer-loop attribution (eligsir/runtime/causal_replay.py):
# `m_outer_tick_total` wraps the ENTIRE `application.tick()` call, which is
# exactly the same call production_application.py's own `m_tick_*`/
# `m_optimization_transaction` regions (from PR #1335) are measured inside of
# -- so every `m_tick_*` region is doubly nested: inside its own declared
# parent (see the m_optimization_transaction nesting above) AND inside
# `m_outer_tick_total`. Declaring the prefix here lets a reader of
# dump_phase_ops()'s flat table know `m_outer_tick_total` already includes
# all of `m_tick_*`/`m_optimization_transaction` without re-deriving that
# from the causal_replay.py source.
#
# Issue #279 follow-up: `CausalReplayDriver.run()` (the plain, unpaced
# driver -- what `--streaming_ingestion_mode iter_based` runs actually
# exercise) additionally classifies every `m_outer_tick_total` interval,
# via `record_timed()` AFTER the tick has already run, as EITHER
# `m_outer_tick_real` (completed >=1 optimizer step) OR `m_outer_tick_drain`
# (a no-op tick -- did zero work but still cost real wall-time). Both are
# therefore also nested inside `m_outer_tick_total`, and are mutually
# exclusive with each other (every tick is exactly one or the other), unlike
# the `m_tick_*`/`surface/` entries above which nest AND overlap.
_PHASE_NESTING: dict[str, str] = {
    "surface/": "ingest_accumulate_surface",
    "m_render": "m_optimization_transaction",
    "m_loss_assembly": "m_optimization_transaction",
    "m_backward": "m_optimization_transaction",
    "m_grad_policies": "m_optimization_transaction",
    "m_optstep": "m_optimization_transaction",
    "m_post_render": "m_optimization_transaction",
    "m_noise": "m_optimization_transaction",
    "m_build_step": "m_optimization_transaction",
    "m_prefetch_selected_cameras": "m_optimization_transaction",
    "m_publish_assimilation_support": "m_optimization_transaction",
    "m_record_digest": "m_optimization_transaction",
    "m_commit_step": "m_optimization_transaction",
    # Issue #279 follow-up (2026-08-31): fine-grained sub-brackets inside
    # CanonicalMappingRuntime.tick()'s m_tick_prologue and
    # m_tick_post_optimization_tail spans. Declared BEFORE the generic
    # "m_tick_" -> "m_outer_tick_total" entry below so the more specific
    # prefix wins (dict iteration is insertion-ordered and _parent_of()
    # returns on first match) -- otherwise these would be misreported as
    # direct children of m_outer_tick_total instead of their true immediate
    # parent, one level further in.
    "m_tick_prologue_": "m_tick_prologue",
    "m_tick_post_optimization_tail_": "m_tick_post_optimization_tail",
    "m_tick_": "m_outer_tick_total",
    "m_optimization_transaction": "m_outer_tick_total",
    "m_outer_tick_real": "m_outer_tick_total",
    "m_outer_tick_drain": "m_outer_tick_total",
}


def _parent_of(name: str) -> str | None:
    """Enclosing region for `name`, or None if it is top-level.

    A region never parents itself (guards against a prefix that also matches
    the declared parent's own name).
    """
    for prefix, parent in _PHASE_NESTING.items():
        if name.startswith(prefix) and name != parent:
            return parent
    return None


def reset_all() -> None:
    """Clear every store. Test-only convenience (production never needs this
    within a single process's lifetime -- report_window() only clears the
    windowed store, by design)."""
    _PHASE_OPS.clear()
    _PHASE_OPS_LIFETIME.clear()
    _PHASE_COUNTS.clear()


def record_count(name: str, value: float) -> None:
    """Record a workload counter (points, valid samples, voxel keys, ...)
    alongside the _timed() latency regions, so per-call cost can be
    normalized by workload size. No-op unless SPLAT_PROFILE_PHASES is set."""
    if not _PROFILE_PHASES:
        return
    rec = _PHASE_COUNTS.setdefault(name, [0.0, 0])
    rec[0] += float(value)
    rec[1] += 1


def record_timed(name: str, dt_s: float) -> None:
    """Manually record one already-measured duration under `name`.

    For the rare case where a region's *name* depends on its own outcome and
    so cannot be decided until after the timed work has already run (issue
    #279's real-vs-drain tick split: whether a `CausalReplayDriver` tick did
    any work is only known once `application.tick()` has returned). Callers
    measure the interval themselves (typically because they already need the
    raw `perf_counter()` delta for an always-on, non-gated accumulator too)
    and hand it here for classification -- this does the same
    `_PHASE_OPS`/`_PHASE_OPS_LIFETIME` bookkeeping `_timed.__exit__` does,
    minus the CUDA sync (there is nothing left to sync against: the region
    already finished before this is called). No-op unless
    SPLAT_PROFILE_PHASES is set, and pays no cost at all when unset.
    """
    if not _PROFILE_PHASES:
        return
    dt = float(dt_s)
    rec = _PHASE_OPS.setdefault(name, [0.0, 0, 0.0])
    rec[0] += dt
    rec[1] += 1
    rec[2] = max(rec[2], dt)
    lt = _PHASE_OPS_LIFETIME.setdefault(
        name, [0.0, 0, 0.0, collections.deque(maxlen=_PHASE_SAMPLE_CAP)])
    lt[0] += dt
    lt[1] += 1
    lt[2] = max(lt[2], dt)
    lt[3].append(dt)


def _cuda_sync() -> None:
    """Drain in-flight GPU work so a _timed region's interval is trustworthy.

    torch is imported lazily AND tolerantly: this module is meant to be a
    torch-free leaf (that is what breaks the circular import motivating the
    extraction), but the sync itself obviously needs torch when there is a GPU
    to sync. Before this guard, merely ENABLING profiling made the module hard-
    depend on torch -- `_timed.__enter__` did a bare `import torch`, so the
    torch-free CPU image raised ModuleNotFoundError on any timed region and the
    leaf-module claim was false in exactly the environment meant to prove it.

    No torch (or no CUDA/ROCm device) simply means no sync is needed: the
    recorded interval is then pure CPU wall-time, which is correct.
    """
    try:
        import torch as _t
    except ImportError:
        return
    if _t.cuda.is_available():
        _t.cuda.synchronize()


def _percentiles(samples, ps=(50, 95, 99)) -> dict:
    """Nearest-rank percentiles over `samples` (assumed already a sequence).
    Deterministic, dependency-free (no numpy requirement at this call site)."""
    xs = sorted(samples)
    n = len(xs)
    if n == 0:
        return {f"p{p}": 0.0 for p in ps}
    out = {}
    for p in ps:
        idx = min(n - 1, max(0, int(round(p / 100.0 * n)) - 1))
        out[f"p{p}"] = xs[idx]
    return out


class _timed:
    """Context manager that records wall-time of a named region when profiling.

    The CUDA sync at enter/exit makes the recorded interval trustworthy for
    regions that themselves launch GPU work (their async kernels are forced
    to complete before the timer stops), at the cost of stalling on whatever
    GPU work was already in flight from an EARLIER, unsynced region. Regions
    that are pure-CPU/numpy (e.g. most of
    StreamingScene.accumulate_observed_surface) pay this stall too if a
    prior region left work in flight -- in practice this is usually a no-op
    here because every _timed region syncs on its own exit, so entry finds
    the queue already drained. Only enabled when SPLAT_PROFILE_PHASES=1;
    zero overhead (no sync, no dict writes) when unset.

    Issue #562: the same region boundaries ALSO feed the duty-cycle timeline
    (``utils/duty_cycle.py``) when one is installed via
    :func:`set_duty_cycle_recorder` -- deliberately reusing this call-site
    vocabulary instead of adding a second, divergent set of hooks. That path
    never synchronizes; the two consumers are independent, and both being off
    (the default) leaves this class exactly as cheap as it was before #562.
    """

    def __init__(self, name: str):
        self.name = name

    def __enter__(self):
        # Two INDEPENDENT guards, deliberately not fused into one cached flag:
        # a single cached "either is on" flag would silently stop tracking a
        # test (or any caller) that flips `_PROFILE_PHASES` directly, which is
        # how this module's own test suite drives it. Each consumer pays only
        # its own module-global load + branch, and when both are off (the
        # default) nothing else executes at all.
        if _DUTY is not None:
            _duty_event(True, self.name)
        if _TORCH_RANGES:
            try:
                import torch as _t

                self._range = _t.profiler.record_function(self.name)
                self._range.__enter__()
            except Exception:
                self._range = None
        if _PROFILE_PHASES:
            _cuda_sync()
            self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        if _TORCH_RANGES:
            try:
                _range = getattr(self, "_range", None)
                if _range is not None:
                    _range.__exit__(None, None, None)
            except Exception:
                pass
        if _PROFILE_PHASES:
            _cuda_sync()
            dt = time.perf_counter() - self._t0
            rec = _PHASE_OPS.setdefault(self.name, [0.0, 0, 0.0])
            rec[0] += dt
            rec[1] += 1
            rec[2] = max(rec[2], dt)
            lt = _PHASE_OPS_LIFETIME.setdefault(
                self.name, [0.0, 0, 0.0, collections.deque(maxlen=_PHASE_SAMPLE_CAP)])
            lt[0] += dt
            lt[1] += 1
            lt[2] = max(lt[2], dt)
            lt[3].append(dt)
        # Duty-cycle exit last: the phase-profiler's own _cuda_sync() above must
        # be billed to THIS region, not leak into the enclosing one.
        if _DUTY is not None:
            _duty_event(False, self.name)
        return False


class PhaseSegments:
    """Sequential (non-nested) sub-brackets for a long function body.

    ``begin(name)`` closes the previous segment and opens ``name``;
    ``close()`` ends the current one. Use where wrapping hundreds of lines
    in ``with`` blocks would mean re-indenting them (``assemble_loss``).
    Costs one attribute check per call when neither profiler is armed.
    """

    __slots__ = ("_current",)

    def __init__(self) -> None:
        self._current = None

    def begin(self, name: str) -> None:
        self.close()
        if _PROFILE_PHASES or _TORCH_RANGES:
            region = _timed(name)
            region.__enter__()
            self._current = region

    def close(self) -> None:
        region = self._current
        if region is not None:
            self._current = None
            region.__exit__(None, None, None)


def dump_phase_ops(model_path: str | None = None) -> dict:
    """Final, whole-run per-region timing + workload-counter summary.

    Reads _PHASE_OPS_LIFETIME (not the windowed _PHASE_OPS, which
    report_window() clears every ~100 iterations -- see the module docstring)
    so the numbers printed/written here genuinely cover the full run, not
    just whatever fired since the last windowed report.

    Returns the summary dict (also printed, and written to
    ``<model_path>/ingest_phase_profile.json`` when model_path is given).
    """
    summary: dict = {
        "phases": {},
        "counts": {},
        # Schema self-description, so a consumer never has to infer either of
        # these from column names or from the "/" in a region name.
        "schema": {
            # sum_s/count/mean_ms/max_ms are lifetime; p50/p95/p99 are NOT.
            # They come from a bounded deque of the most recent
            # _PHASE_SAMPLE_CAP samples, so on any run longer than that they
            # describe the run's TAIL, not the full run.
            "lifetime_fields": ["sum_s", "count", "mean_ms", "max_ms"],
            "windowed_fields": ["p50", "p95", "p99"],
            "percentile_scope": "recent_window",
            "percentile_window_samples": _PHASE_SAMPLE_CAP,
            # Some regions nest inside others (see `parent` per phase below),
            # so phase sums deliberately exceed 100% of wall-time.
            "phases_are_nested": True,
            "nesting": dict(_PHASE_NESTING),
        },
    }
    if not _PHASE_OPS_LIFETIME:
        return summary
    print(
        "[profile] per-region wall-time (sum_s / count / mean_ms LIFETIME; "
        f"p50/p95/p99 over the most recent {_PHASE_SAMPLE_CAP} samples ONLY, not the full run):",
        flush=True,
    )
    print(
        "[profile] '>' marks a region measured INSIDE its parent -- these are "
        "double-counted by design, so the column does not sum to wall-time.",
        flush=True,
    )
    for name, (s, c, mx, samples) in sorted(_PHASE_OPS_LIFETIME.items(), key=lambda kv: -kv[1][0]):
        pct = _percentiles([v * 1000.0 for v in samples])
        parent = _parent_of(name)
        windowed = len(samples) < c  # deque wrapped: percentiles cover a tail
        marker = ">" if parent else " "
        print(
            f"[profile] {marker} {name:28s} {s:8.2f}s  n={c:<6d} mean={1000*s/max(c,1):8.1f}ms "
            f"p50w={pct['p50']:8.1f}ms p95w={pct['p95']:8.1f}ms p99w={pct['p99']:8.1f}ms "
            f"max={1000*mx:8.1f}ms" + (f"  [in {parent}]" if parent else ""),
            flush=True,
        )
        summary["phases"][name] = {
            "sum_s": round(s, 4),
            "count": c,
            "mean_ms": round(1000 * s / max(c, 1), 3),
            "max_ms": round(1000 * mx, 3),
            "sample_count": len(samples),
            "sample_cap": _PHASE_SAMPLE_CAP,
            # True once the deque has wrapped, i.e. the percentiles below
            # genuinely describe only the tail of this run.
            "percentiles_windowed": windowed,
            # None for a top-level region; otherwise the enclosing region
            # whose sum_s ALREADY INCLUDES this one.
            "parent": parent,
            **{k: round(v, 3) for k, v in pct.items()},
        }
    if _PHASE_COUNTS:
        print("[profile] per-region workload counters (sum / count / mean):", flush=True)
        for name, (s, c) in sorted(_PHASE_COUNTS.items(), key=lambda kv: -kv[1][0]):
            print(f"[profile]   {name:28s} sum={s:.0f} n={c} mean={s/max(c,1):.2f}", flush=True)
            summary["counts"][name] = {"sum": s, "count": c, "mean": s / max(c, 1)}
    if model_path:
        out = os.path.join(model_path, "ingest_phase_profile.json")
        try:
            with open(out, "w") as f:
                json.dump(summary, f, indent=2)
            print(f"[profile] full-run phase summary written to {out}", flush=True)
        except OSError as e:
            print(f"[profile] WARNING: failed to write {out}: {e}", flush=True)
    return summary
