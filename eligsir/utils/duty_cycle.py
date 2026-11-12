"""CPU/GPU duty-cycle timeline and stall taxonomy for streaming runs (issue #562).

Why this exists
---------------
Issue #561 (P0 master, "low accelerator duty cycle on ScanNet++ and high-quality
streaming runs") is argued today from screenshots of a GPU monitor and from
tqdm's ``it/s`` average. Nothing in a run's own artifacts answers *"where did the
wall-clock actually go?"*. This module answers exactly that question and nothing
else. **It is telemetry only**: it never moves a tensor, never synchronizes,
never changes training math, and is OFF by default.

Measuring it, and the trap
--------------------------
The tempting way to get an honest "GPU busy %" is a per-iteration
``torch.cuda.synchronize()``. **This module adds none, anywhere.** That is not a
stylistic preference:

* ``eligsir/runtime/profiling/events.py``'s module docstring already records
  the finding (W3-01 review) that a per-span synchronize serializes the very
  overlap it is meant to measure -- the profiler would be measuring a workload
  it imposed.
* CLAUDE.md's standing invariant is that live observability never measurably
  backpressures training. This project has been bitten twice: the rvizsplat
  publish path's GIL contention (>1 s/call stalls on the *training* thread) and
  ``accumulate_observed_surface`` at ~64 ms/frame, which had to be auto-disabled
  in ``online_mode``.

A CUDA-event pair (``start.elapsed_time(end)``) does not rescue this either, and
it is worth writing down why so nobody re-derives it: CUDA events measure
*positions on the stream timeline*, not busy time. When the queue is empty --
exactly the starved regime #561 is about -- the GPU reaches each event the
instant it is enqueued, so the "span" converges to the CPU wall-clock of the
region and carries no information. It is a useless bound precisely where the
bound is needed. (``GpuSpanRegistry``'s deferred-resolve design is still the
right substrate for *comparing GPU stage costs* in a saturated regime; it is not
the right instrument for a starvation duty cycle. This module deliberately does
not use it, and does not touch ``SPLAT_PROFILE_RUNTIME``.)

What is delivered instead: an exact partition of host wall-clock
--------------------------------------------------------------
Every microsecond of the training loop is billed to exactly one taxonomy
category (:data:`CATEGORIES`), by construction:

* a single monotonic cursor plus a stack of open instrumented regions;
* on every region enter/exit, the interval since the last event is billed to the
  **innermost currently-open region**'s category, or to ``unattributed`` when no
  region is open;
* therefore ``sum(category totals) == wall time`` exactly -- there is no hiding
  place, and time nobody instrumented shows up as ``unattributed`` rather than
  silently vanishing. :meth:`DutyCycleTimeline.summary` asserts this invariant
  and reports the residual.

**This is a different convention from ``ingest_phase_profile.json``**, which is a
flat table of *nested* regions whose sums deliberately exceed 100% of wall-time
(see ``utils/ingest_phase_profiler.py``'s ``_PHASE_NESTING``). Both are real;
a consumer that compares the two without knowing this will conclude one is
broken. Here, innermost-wins, and the parts sum to the whole.

The headline metrics, and exactly what they bound
-------------------------------------------------
``optimizer_region_frac = frac(optimization)`` -- the fraction of wall-clock the
host spent inside the render/loss/backward/optimizer region. It is an upper
bound on **the optimizer's** GPU time (kernels inside the region may themselves
leave the device idle, so the optimizer's true device time is only ever lower).

**It is NOT an upper bound on total GPU-busy time, and must never be read as
one.** At least five other categories also launch device work, which is exactly
why host-side timing alone cannot bound device idle:

* ``depth_insertion`` -- Phase-2 insertion is preceded by a full gsplat
  rasterization of the arriving frame (``train_streaming.py``'s
  ``render(new_cam, ..., render_depth=True)``). Note that render sits *before*
  ``_t_insert_start``, so it is outside the ``depth_insertion`` bracket and its
  time lands in ``unattributed``.
* ``admission`` -- ``ingest_admission_fn`` performs the admission render.
* ``mutation`` -- MCMC noise/relocation are tensor ops on the live model.
* ``telemetry_report`` -- ``mid_training_report_snapshot`` renders test views.
* ``save_checkpoint`` -- ``save_capture`` does device->host copies.

``host_only_frac = frac(ingest_decode) + frac(host_wait)`` is offered as the
complementary, deliberately conservative signal: the categories with no *known*
device launch, i.e. a rough **lower** bound on GPU-idle wall-clock. Even this
under-claims in one direction -- ``ingest_frame_to_camera`` may perform H2D
copies -- so treat it as indicative, not exact.

A true device-side busy% would need either a synchronize (refused, above) or a
background device-utilization sampler (``rocm-smi``/``amd-smi``); the latter is
deliberately NOT built here -- it reports whole-device, all-process utilization
(``utils/gpu_coresidency.py`` exists because other processes on this box
contaminate exactly that reading) and is named as follow-up in #563.

Reading the numbers honestly
----------------------------
* **Coverage of the taxonomy is itself unmeasured.** The instrumented regions are
  the pre-existing ``_timed()`` vocabulary; if a real run comes back with a large
  ``unattributed`` fraction, that is a *finding about instrumentation coverage*,
  not a measurement of idle time. Read ``unattributed_frac`` first.
* **Only the owner thread is billed.** ``save_enqueue``/``save_capture`` and the
  frame-preparation regions can fire from worker threads; a worker's events would
  otherwise steal attribution from the main thread mid-region. Foreign-thread
  events are counted in ``foreign_thread_events`` and dropped, so the partition
  invariant holds by construction.
* ``unclosed_at_iteration_boundary`` / ``out_of_order_exits`` / ``unmatched_exits``
  are anomaly counters, not errors: a nonzero value means some region bracket did
  not nest cleanly and its time was billed to its enclosing region instead.
* This module has **never been exercised on a real starved GPU run** as of the PR
  that introduced it. See #563.
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections import deque
from typing import Any, Callable, Deque, Dict, List, Optional

#: Bumped whenever the category set or the region->category mapping changes in a
#: way that makes two runs' numbers non-comparable.
TAXONOMY_VERSION = 1

#: The stall taxonomy. Ordered, exhaustive-by-construction over wall-clock: every
#: interval is billed to exactly one of these, and the last two exist precisely so
#: that "we did not instrument this" is VISIBLE rather than folded into a named
#: bucket.
#:
#: The first three names are taken verbatim from
#: ``eligsir.runtime.profiling.gpu_timing.StageTimerRegistry``'s default stage
#: names (``optimization``/``admission``/``mutation``) rather than re-spelled, so
#: the two artifacts in this repo that name pipeline stages agree.
CATEGORIES: tuple[str, ...] = (
    # Host-side source assimilation: reading/decoding a source frame, building a
    # camera from it, accumulating observed surface, SLAM tracking.
    "ingest_decode",
    # Deciding whether a frame is admitted (keyframe gate, pregate, controller).
    "admission",
    # Phase-2 incremental depth insertion: backprojection, dedup, append.
    "depth_insertion",
    # Render + loss + backward + optimizer step. See `optimizer_region_frac`
    # -- and the module docstring's list of the OTHER categories that also
    # launch GPU work, which is why no host-side number here bounds device idle.
    "optimization",
    # MCMC noise/relocation/growth, lifecycle bookkeeping, param sanitize.
    "mutation",
    # Occupancy rebuild, bbox prune, loop closure, submap maintenance.
    "maintenance",
    # TensorBoard writes, mid-training reports, evaluation.
    "telemetry_report",
    # PLY/checkpoint capture and enqueue on the training thread.
    "save_checkpoint",
    # Explicit host waits (pacing sleeps, queue waits). No producer today --
    # reserved so that a future `wait/*` region lands somewhere meaningful
    # instead of in `other_instrumented`.
    "host_wait",
    # An instrumented region with no mapping. Visible on purpose: a new
    # `_timed()` name shows up here instead of being silently miscounted.
    "other_instrumented",
    # No instrumented region was open. This is the residual that makes the
    # partition exhaustive.
    "unattributed",
)

#: Bridge from this module's categories to the label vocabulary #562's body asks
#: for. Kept as an explicit mapping rather than renaming the categories, because
#: the ``GPU_*`` names presuppose a device-side busy/idle split this module
#: deliberately does not claim to measure (see the module docstring).
STALL_CLASS: Dict[str, str] = {
    "ingest_decode": "GPU_IDLE_WAIT_SOURCE_IO",
    "admission": "GPU_IDLE_WAIT_CPU_PREP",
    "depth_insertion": "GPU_IDLE_WAIT_TOPOLOGY_COMMIT",
    "optimization": "GPU_BUSY_OPTIMIZER",
    "mutation": "GPU_BUSY_MUTATION",
    "maintenance": "GPU_IDLE_WAIT_CPU_PREP",
    "telemetry_report": "GPU_IDLE_WAIT_EVAL",
    "save_checkpoint": "GPU_IDLE_WAIT_SAVE_SNAPSHOT",
    "host_wait": "GPU_IDLE_WAIT_CPU_PREP",
    "other_instrumented": "GPU_IDLE_UNKNOWN",
    "unattributed": "GPU_IDLE_UNKNOWN",
}

#: Exact ``_timed()`` region name -> category. Every name here exists in the tree
#: today (``grep -rho '_timed("[a-z_/]*"'``); an unknown name falls through to
#: ``other_instrumented`` rather than being guessed at.
REGION_CATEGORY: Dict[str, str] = {
    # -- ingest / decode --------------------------------------------------
    "m_ingest": "ingest_decode",
    "ingest_get_source_frame": "ingest_decode",
    "ingest_frame_to_camera": "ingest_decode",
    "ingest_accumulate_surface": "ingest_decode",
    "online_track": "ingest_decode",
    # -- admission ---------------------------------------------------------
    "ingest_admission_fn": "admission",
    "ingest_pregate": "admission",
    "online_decide": "admission",
    "online_controller_tick": "admission",
    # -- depth insertion ---------------------------------------------------
    "depth_insertion": "depth_insertion",
    # -- optimization ------------------------------------------------------
    "m_optimization_engine": "optimization",
    "m_render": "optimization",
    "m_post_render": "optimization",
    "m_loss_assembly": "optimization",
    "m_loss_item": "optimization",
    "m_backward": "optimization",
    "m_grad_masking": "optimization",
    "m_optstep": "optimization",
    # -- mutation ----------------------------------------------------------
    "m_noise": "mutation",
    "m_sanitize": "mutation",
    "after_optstep": "mutation",
    "after_mcmc_noise": "mutation",
    "step_post_backward": "mutation",
    # -- maintenance -------------------------------------------------------
    "occupancy_rebuild": "maintenance",
    "bbox_prune": "maintenance",
    "loop_closure_optimize": "maintenance",
    "online_submap": "maintenance",
    # -- telemetry / report ------------------------------------------------
    "online_tb": "telemetry_report",
    "mid_training_report_snapshot": "telemetry_report",
}

#: Prefix fallbacks, consulted only after :data:`REGION_CATEGORY` misses. Ordered;
#: first match wins. ``ingest_`` is deliberately LAST of the ingest-ish rules so
#: that ``ingest_admission_fn``/``ingest_pregate`` (exact entries above) keep
#: their ``admission`` classification.
REGION_PREFIX_CATEGORY: tuple[tuple[str, str], ...] = (
    ("surface/", "ingest_decode"),
    ("save_", "save_checkpoint"),
    ("wait/", "host_wait"),
    ("ingest_", "ingest_decode"),
)


def classify_region(name: str) -> str:
    """Category for an instrumented region name (never raises)."""
    cat = REGION_CATEGORY.get(name)
    if cat is not None:
        return cat
    for prefix, prefix_cat in REGION_PREFIX_CATEGORY:
        if name.startswith(prefix):
            return prefix_cat
    return "other_instrumented"


class DutyCycleTimeline:
    """Exact partition of host wall-clock across :data:`CATEGORIES`.

    Constructed only when telemetry is enabled, so the disabled path costs
    nothing at all (see ``utils/ingest_phase_profiler.set_duty_cycle_recorder``).

    Thread model: events are billed only when they arrive on the thread that
    constructed the timeline (or ``owner_thread`` if given). Everything else is
    counted in ``foreign_thread_events`` and dropped -- a worker thread's
    enter/exit must not mutate the main thread's stack.
    """

    def __init__(
        self,
        *,
        window_iters: int = 100,
        max_windows: int = 4096,
        stall_ms: float = 250.0,
        max_stalls: int = 64,
        clock: Callable[[], float] = time.perf_counter,
        owner_thread: Optional[int] = None,
    ) -> None:
        self.window_iters = max(1, int(window_iters))
        self.max_windows = max(1, int(max_windows))
        self.stall_s = max(0.0, float(stall_ms) / 1000.0)
        self.max_stalls = max(0, int(max_stalls))
        self._clock = clock
        self._owner = int(owner_thread) if owner_thread is not None else threading.get_ident()

        now = self._clock()
        self._t0 = now
        self._cursor = now
        self._stack: List[str] = []

        self._totals: Dict[str, float] = {c: 0.0 for c in CATEGORIES}
        self._window: Dict[str, float] = {c: 0.0 for c in CATEGORIES}
        self._window_t0 = now
        self._window_iters_seen = 0
        self._window_index = 0
        self._windows: Deque[Dict[str, Any]] = deque(maxlen=self.max_windows)
        self._windows_dropped = 0
        self._stalls: List[Dict[str, Any]] = []

        self._iteration = 0
        self.total_iterations = 0
        self.enter_events = 0
        self.exit_events = 0
        self.foreign_thread_events = 0
        self.out_of_order_exits = 0
        self.unmatched_exits = 0
        self.unclosed_at_iteration_boundary = 0

    # -- hot path ----------------------------------------------------------

    def _bill(self) -> None:
        """Bill the interval since the last event to the innermost open region."""
        now = self._clock()
        dt = now - self._cursor
        self._cursor = now
        if dt <= 0.0:
            return
        cat = classify_region(self._stack[-1]) if self._stack else "unattributed"
        self._totals[cat] += dt
        self._window[cat] += dt
        if self.stall_s and dt >= self.stall_s and self.max_stalls:
            self._record_stall(cat, dt, now)

    def enter(self, name: str) -> None:
        if threading.get_ident() != self._owner:
            self.foreign_thread_events += 1
            return
        self._bill()
        self._stack.append(name)
        self.enter_events += 1

    def exit(self, name: str) -> None:
        if threading.get_ident() != self._owner:
            self.foreign_thread_events += 1
            return
        self._bill()
        self.exit_events += 1
        if self._stack and self._stack[-1] == name:
            self._stack.pop()
            return
        # Non-LIFO close (manual __enter__/__exit__ brackets exist in
        # train_streaming.py). Remove the innermost matching entry and count it,
        # rather than corrupting every subsequent attribution.
        for i in range(len(self._stack) - 1, -1, -1):
            if self._stack[i] == name:
                del self._stack[i]
                self.out_of_order_exits += 1
                return
        self.unmatched_exits += 1

    def iteration_boundary(self, iteration: int) -> Optional[Dict[str, Any]]:
        """Call once per training iteration. Returns a window row when one closed.

        Also self-heals a leaked region: an unpaired ``__enter__`` (an early
        ``break``/``continue`` out of a manual bracket) would otherwise mis-bill
        the entire rest of the run to that region. The stack is unwound here and
        the leak counted.
        """
        if threading.get_ident() != self._owner:
            self.foreign_thread_events += 1
            return None
        self._bill()
        if self._stack:
            self.unclosed_at_iteration_boundary += len(self._stack)
            self._stack.clear()
        self._iteration = int(iteration)
        self.total_iterations += 1
        self._window_iters_seen += 1
        if self._window_iters_seen >= self.window_iters:
            return self._close_window()
        return None

    # -- windows -----------------------------------------------------------

    def _close_window(self) -> Dict[str, Any]:
        wall = max(1e-12, self._cursor - self._window_t0)
        row: Dict[str, Any] = {
            "window": self._window_index,
            "iteration": self._iteration,
            "iterations": self._window_iters_seen,
            "wall_s": round(wall, 6),
            "it_per_s": round(self._window_iters_seen / wall, 4),
            "frac": {c: round(self._window[c] / wall, 6) for c in CATEGORIES},
            "optimizer_region_frac": round(self._window["optimization"] / wall, 6),
            "host_only_frac": round(
                (self._window["ingest_decode"] + self._window["host_wait"]) / wall, 6
            ),
        }
        self._window_index += 1
        if len(self._windows) == self._windows.maxlen:
            self._windows_dropped += 1
        self._windows.append(row)
        for c in CATEGORIES:
            self._window[c] = 0.0
        self._window_t0 = self._cursor
        self._window_iters_seen = 0
        return row

    def _record_stall(self, category: str, dt: float, end_time: float) -> None:
        region = self._stack[-1] if self._stack else None
        entry = {
            "category": category,
            "stall_class": STALL_CLASS[category],
            "region": region,
            "duration_ms": round(dt * 1000.0, 3),
            "iteration": self._iteration,
            "t_end_s": round(end_time - self._t0, 6),
        }
        self._stalls.append(entry)
        if len(self._stalls) > self.max_stalls:
            self._stalls.sort(key=lambda e: -e["duration_ms"])
            del self._stalls[self.max_stalls:]

    # -- readout -----------------------------------------------------------

    def tb_scalars(self) -> Dict[str, float]:
        """Scalars for the most recently closed window (empty before the first)."""
        if not self._windows:
            return {}
        row = self._windows[-1]
        out = {f"duty_cycle/{c}_frac": float(row["frac"][c]) for c in CATEGORIES}
        out["duty_cycle/optimizer_region_frac"] = float(row["optimizer_region_frac"])
        out["duty_cycle/host_only_frac"] = float(row["host_only_frac"])
        out["duty_cycle/window_it_per_s"] = float(row["it_per_s"])
        out["duty_cycle/window_wall_s"] = float(row["wall_s"])
        return out

    def top_stalls(self) -> List[Dict[str, Any]]:
        return sorted(self._stalls, key=lambda e: -e["duration_ms"])

    def summary(self) -> Dict[str, Any]:
        """Whole-run summary. Bills the open interval first, so it is exact."""
        if threading.get_ident() == self._owner:
            self._bill()
        wall = max(1e-12, self._cursor - self._t0)
        attributed = sum(self._totals.values())
        totals = {c: round(self._totals[c], 6) for c in CATEGORIES}
        frac = {c: round(self._totals[c] / wall, 6) for c in CATEGORIES}
        return {
            "taxonomy_version": TAXONOMY_VERSION,
            "categories": list(CATEGORIES),
            "stall_class": dict(STALL_CLASS),
            "wall_s": round(wall, 6),
            "total_iterations": self.total_iterations,
            "it_per_s": round(self.total_iterations / wall, 4),
            "totals_s": totals,
            "frac": frac,
            # Fraction of wall-clock the HOST spent inside the optimizer
            # region. An upper bound on the OPTIMIZER's GPU time -- NOT on
            # total GPU-busy time, and not a utilization measurement. See the
            # module docstring for the five other categories that also launch
            # device work.
            "optimizer_region_frac": frac["optimization"],
            # Categories with no KNOWN device launch: a rough lower bound on
            # GPU-idle wall-clock (indicative, not exact -- see the docstring).
            "host_only_frac": round(frac["ingest_decode"] + frac["host_wait"], 6),
            "unattributed_frac": frac["unattributed"],
            # Exhaustiveness proof: the partition sums to wall time. `residual_s`
            # is float-noise only, and is reported rather than assumed.
            "partition_residual_s": round(wall - attributed, 9),
            "partition_exhaustive": abs(wall - attributed) < 1e-6,
            "n_windows": self._window_index,
            "windows_retained": len(self._windows),
            "windows_dropped": self._windows_dropped,
            "window_iters": self.window_iters,
            "stall_threshold_ms": round(self.stall_s * 1000.0, 3),
            "n_stalls": len(self._stalls),
            "anomalies": {
                "enter_events": self.enter_events,
                "exit_events": self.exit_events,
                "foreign_thread_events": self.foreign_thread_events,
                "out_of_order_exits": self.out_of_order_exits,
                "unmatched_exits": self.unmatched_exits,
                "unclosed_at_iteration_boundary": self.unclosed_at_iteration_boundary,
                "open_regions_at_summary": len(self._stack),
            },
        }

    def report_block(self) -> Dict[str, Any]:
        """``report.json["duty_cycle"]`` payload.

        Tri-state, mirroring ``utils/gpu_memory_telemetry.disabled_report_block``
        and the #282 NO_SAMPLES lesson: an ABSENT key means the report predates
        #562; ``enabled: false`` means telemetry was off; ``n_windows: 0`` means
        it was on but never closed a window.
        """
        block = self.summary()
        block["enabled"] = True
        block["top_stalls"] = self.top_stalls()[:10]
        return block

    # -- artifacts ---------------------------------------------------------

    def write_artifacts(self, model_path: str) -> Optional[str]:
        """Write ``<model_path>/performance_trace/`` (once, at finalization).

        Deliberately NOT flushed periodically during training: this repo has a
        documented incident where the TensorBoard writer's small periodic writes
        stalled the main thread on NFS latency. Everything is accumulated in
        bounded memory and written once here.
        """
        if not model_path:
            return None
        out_dir = os.path.join(model_path, "performance_trace")
        try:
            os.makedirs(out_dir, exist_ok=True)
            summary = self.summary()
            with open(os.path.join(out_dir, "summary.json"), "w") as f:
                json.dump(summary, f, indent=2)
            with open(os.path.join(out_dir, "duty_cycle.jsonl"), "w") as f:
                for row in self._windows:
                    f.write(json.dumps(row) + "\n")
            with open(os.path.join(out_dir, "stall_top.json"), "w") as f:
                json.dump(
                    {
                        "threshold_ms": summary["stall_threshold_ms"],
                        "n_recorded": summary["n_stalls"],
                        "stalls": self.top_stalls(),
                    },
                    f,
                    indent=2,
                )
        except OSError as e:  # pragma: no cover - filesystem failure path
            print(f"[duty-cycle] WARNING: failed to write {out_dir}: {e}", flush=True)
            return None
        return out_dir

    def format_line(self) -> str:
        """One-line human summary for train.log."""
        s = self.summary()
        parts = " ".join(
            f"{c}={s['frac'][c] * 100:.1f}%" for c in CATEGORIES if s["frac"][c] >= 0.001
        )
        return (
            f"[duty-cycle] wall={s['wall_s']:.1f}s iters={s['total_iterations']} "
            f"it/s={s['it_per_s']:.2f} optimizer_region={s['optimizer_region_frac'] * 100:.1f}% "
            f"| {parts}"
        )


def disabled_report_block(reason: str = "telemetry disabled") -> Dict[str, Any]:
    """``report.json["duty_cycle"]`` for a run where the timeline never ran."""
    return {
        "enabled": False,
        "reason": reason,
        "taxonomy_version": TAXONOMY_VERSION,
        "categories": list(CATEGORIES),
        "n_windows": 0,
        "total_iterations": 0,
        "frac": {},
        "optimizer_region_frac": None,
        "host_only_frac": None,
        "unattributed_frac": None,
        "top_stalls": [],
    }
