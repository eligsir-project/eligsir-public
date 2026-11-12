"""Online-mode schedule-horizon helpers.

Despite the module name, nothing here decides when training STOPS. The two
termination decisions in this codebase live at their own layers and are
deliberately not unified (see
``docs/architecture/2026-08-23-issue1075-stop-policy-reconcile.md``):

- **Frame source** (``eligsir/io/sources/ros2_live.py``): bounds the input
  stream on frame count / wall-clock duration / idle timeout, and emits a
  LABELED ``stream_end_reason`` -- ``bag_complete`` vs.
  ``idle_timeout_no_completion_signal`` is the stall-vs-genuine-end
  discriminator the run contracts treat as ``SUSPECTED_TRUNCATED``-equivalent.
- **Training loop** (``train_streaming.py``): the 2026-07-04 unified
  rolling-budget rule -- stop once ``iteration - _last_ingest_iter >=
  streaming_refinement_iterations`` (gated on ``online_mode``), with
  ``--streaming_max_iterations`` as an optional hard ceiling that logs when it
  truncates real ingestion work.

A pair of ``OfflineStopPolicy``/``OnlineStopPolicy`` dataclasses once sat here,
specified by ``docs/plan_ros2_online_cleanup_2026-06-17.md`` but never wired to
anything. They predated the rolling-budget rule and could not express it: they
CONSUMED a termination verdict (``input_done: bool``) rather than computing one,
so they had no parameter able to carry ``_last_ingest_iter``. Deleted 2026-08-23
(issue #1075) rather than left as misleading scaffolding. Do not reintroduce a
stop-policy abstraction here without first reading the doc above -- the two
layers named above already work, are tested, and carry semantics (the
``online_mode`` gate, the labeled end reasons, the truncation diagnostic) that a
bare ``bool`` return cannot represent.

What DOES live here: the online-mode schedule-horizon overrides
(``apply_online_schedule_overrides`` and the deliberately-decoupled clocks it
sets), the growth hard-stop overlay, the fixed-topology contract validator, and
the adaptive-mapping-end trigger.
"""

from __future__ import annotations

from dataclasses import dataclass


ONLINE_UNBOUNDED_ITERATIONS = 2_147_483_647


@dataclass(frozen=True)
class AnnealClock:
    """Anneal-progress clock decoupled from the training loop bound.

    Merge B (structural cleanup): the anneal *horizon* (how fast schedules ramp)
    is conceptually separate from the *loop bound* (when training stops). In
    online mode the loop bound is unbounded (2^31), which — when reused as the
    anneal horizon — pins every ``u`` near 0 and freezes the schedule. An
    AnnealClock carries its own horizon so callers can anneal over a finite
    window while the loop runs unbounded.

    ``progress`` is the VERBATIM clamp used by both the count-loss
    (``min(iteration/max(1,horizon), 1.0)``) and the schedule
    (``utils.mcmc_schedule.progress`` with start=0). With ``start`` defaulting to
    0 the two are identical, so this is a faithful wrapper.
    """

    horizon: int

    def progress(self, iteration: int, start: int = 0) -> float:
        """Compute fractional progress through the anneal window.

        Args:
            iteration: Current iteration (0-indexed).
            start: Start iteration of the anneal window (default 0).

        Returns:
            Progress fraction clamped to [0.0, 1.0], where 0 is the start
            and 1.0 is at or past the horizon.
        """
        span = float(max(1, int(self.horizon) - int(start)))
        u = (int(iteration) - int(start)) / span
        if u < 0.0:
            return 0.0
        if u > 1.0:
            return 1.0
        return u


def apply_online_schedule_overrides(
    opt,
    *,
    horizon: int = ONLINE_UNBOUNDED_ITERATIONS,
    schedule_horizon: int | None = None,
    anneal_horizon: int | None = None,
    noise_stop_horizon: int | None = None,
    growth_stop_horizon: int | None = None,
) -> int:
    """Neutralize offline iteration-stop schedules for online mode.

    Five deliberately-decoupled horizons:

    - ``horizon`` is the *loop* bound (``opt.iterations``); stays unbounded so the
      online loop runs until the stop policy / signal ends it.
    - ``schedule_horizon`` (A6) is the *full MCMC schedule* horizon — growth/relocate
      intervals, dead-opacity ramp, etc. A finite value anneals the WHOLE schedule
      including the dead-opacity ramp, which OVER-PRUNES short runs (scene vanish);
      use with care. 0/None = legacy unbounded.
    - ``anneal_horizon`` (Merge B) is the *count-loss anneal clock only*. A finite
      value ramps the effective-count N_target toward the capacity target WITHOUT
      touching the dead-opacity ramp (no over-prune) — the gentle N-bound lever.
      Defaults to the schedule horizon, so behaviour-neutral unless opted into.
    - ``noise_stop_horizon`` (2026-07-06, extended 2026-07-06b) is the *position-noise*
      clock: past this iteration, MCMC xyz-noise injection stops, giving positions a
      clean, jitter-free convergence tail.
    - ``growth_stop_horizon`` (2026-07-18, split out from ``noise_stop_horizon``) is
      the *growth/insertion* clock: past this iteration, MCMC growth/relocation-driven
      insertion (``mcmc_stop_growth_iter``) AND the streaming pipeline's own depth-based
      Gaussian insertion (gated separately in ``train_streaming.py`` via the same value)
      stop. **Deliberately independent of ``noise_stop_horizon``** — real-GPU testing on
      floor3 (2026-07-18) found that coupling them to the SAME (early, mapping-end-ish)
      horizon starves the model of capacity: MCMC growth is an ongoing densification
      mechanism that keeps subdividing/relocating existing Gaussians to reach adequate
      per-region resolution long after new-frame ingestion ends, entirely independent of
      frame arrival — stopping it early (e.g. at the point mapping/ingestion naturally
      completes) cut a floor3 run's Gaussian count from a healthy ~230k (unbounded
      growth) down to ~13k (frozen at the coupled horizon), craterring test PSNR from
      ~22dB to ~19dB. Defaults to ``noise_stop_horizon`` when unset (preserves the prior
      coupled behaviour exactly for any caller not yet passing this explicitly) — i.e.
      0/unset on both is still fully behaviour-neutral.

    Stashes ``opt._online_schedule_horizon`` (schedule clock) and ``opt._anneal_clock``
    (count-loss clock) for the downstream schedule / loss reads.
    """

    opt.iterations = int(horizon)
    sched = (
        int(schedule_horizon)
        if schedule_horizon is not None and int(schedule_horizon) > 0
        else int(horizon)
    )
    opt.densify_until_iter = sched
    _refinement_horizon = (
        int(noise_stop_horizon)
        if noise_stop_horizon is not None and int(noise_stop_horizon) > 0
        else None
    )
    opt.mcmc_noise_stop_iter = _refinement_horizon if _refinement_horizon is not None else sched
    # Growth-stop horizon: independent of noise-stop (see docstring, 2026-07-18
    # split). Defaults to the noise-stop horizon when unset, so any caller not
    # yet passing growth_stop_horizon explicitly keeps the prior coupled
    # behaviour (and 0/unset on both remains fully behaviour-neutral).
    _growth_horizon = (
        int(growth_stop_horizon)
        if growth_stop_horizon is not None and int(growth_stop_horizon) > 0
        else _refinement_horizon
    )
    opt.mcmc_stop_growth_iter = _growth_horizon if _growth_horizon is not None else sched
    # Streaming-pipeline depth-based insertion reads this same value directly in
    # train_streaming.py's per-iteration insertion gate (not part of the opt/schedule
    # machinery here, but stashed for discoverability/consistency).
    opt._streaming_insertion_stop_iter = _growth_horizon
    # 2026-07-18 correction: mcmc_stop_growth_iter conflated two roles --
    # (1) the HARD GATE checked at the actual growth-trigger call site
    # (utils/strategies/mcmc_strategy.py's `refine_stop`), which is what
    # growth_stop_horizon above correctly controls, and (2) the
    # NORMALIZATION DENOMINATOR for get_mcmc_schedule's u_growth progress
    # fraction (utils/mcmc_schedule.py:68), which drives how fast the
    # growth INTERVAL/RATE itself ramps from aggressive (early) to sparse
    # (late). Real-GPU testing on floor3 found that feeding a near-term
    # growth_stop_horizon (e.g. 55000 out of a 60000-iteration run) into (2)
    # makes u_growth ramp toward 1.0 far faster than the unbounded baseline
    # that actually worked (u_growth~=0 the whole run), throttling growth
    # RATE for nearly the entire run, not just after the intended late stop
    # point -- final N landed at ~30k instead of a healthy ~220k, regardless
    # of the hard gate (1) being correctly set. Role (2) must stay pinned to
    # the unbounded loop horizon always, independent of growth_stop_horizon,
    # so a late hard-stop doesn't retroactively slow down early growth.
    opt._mcmc_growth_schedule_horizon = int(horizon)
    opt.position_lr_max_steps = sched
    opt._online_schedule_horizon = sched
    opt._anneal_clock = AnnealClock(
        int(anneal_horizon)
        if anneal_horizon is not None and int(anneal_horizon) > 0
        else int(sched)
    )
    return sched


def enforce_growth_stop(schedule: dict, *, iteration: int, stop_growth_iter: int | None) -> dict:
    """Apply the authoritative MCMC growth hard-stop after all policy overlays.

    Streaming maintenance and controller overlays may re-enable growth after
    :func:`get_mcmc_schedule` has evaluated its stop horizon.  The stop flag is
    a safety/correctness boundary, so it must be the final say while leaving
    relocation (which has a separate horizon) untouched.
    """
    if stop_growth_iter is not None and int(stop_growth_iter) > 0:
        if int(iteration) >= int(stop_growth_iter):
            schedule["allow_growth"] = False
    return schedule


def validate_fixed_topology_contract(
    *,
    online_mode: bool,
    requested_growth_stop: int | None,
    explicit_growth_stop: int | None,
    resolved_growth_stop: int | None,
    insertion_enabled: bool,
    explicit_insertion_stop: int | None,
) -> list[str]:
    """Validate the explicit horizons required by a fixed-topology run.

    Online-mode schedule normalization intentionally ignores the raw
    ``mcmc_stop_growth_iter`` flag unless its dedicated growth horizon is
    supplied.  A fixed-topology experiment must therefore declare both
    horizons and verify their resolved values before training begins.
    """
    errors: list[str] = []
    growth = int(explicit_growth_stop or 0)
    insertion = int(explicit_insertion_stop or 0)
    requested = int(requested_growth_stop or 0)
    resolved = int(resolved_growth_stop or 0)
    if requested <= 0:
        errors.append("mcmc_stop_growth_iter must be a positive fixed-topology horizon")
    if growth <= 0:
        errors.append("streaming_growth_stop_horizon must be explicit and positive")
    if insertion <= 0:
        errors.append("streaming_insert_stop_iter must be explicit and positive")
    if growth > 0 and resolved != growth:
        errors.append(
            f"resolved mcmc_stop_growth_iter={resolved} does not match "
            f"streaming_growth_stop_horizon={growth}"
        )
    if growth > 0 and insertion > 0 and insertion != growth:
        errors.append(
            f"streaming_insert_stop_iter={insertion} does not match "
            f"streaming_growth_stop_horizon={growth}"
        )
    if insertion_enabled:
        errors.append("streaming_insert_from_depth must be disabled for fixed topology")
    if online_mode and requested > 0 and growth <= 0:
        errors.append(
            "online_mode overwrites raw mcmc_stop_growth_iter; set "
            "streaming_growth_stop_horizon explicitly"
        )
    return errors


def should_apply_adaptive_schedule_horizons(
    adaptive_on: bool,
    already_applied: bool,
    last_ingest_iter: int,
    first_iter: int,
    iteration: int,
    threshold: int,
) -> bool:
    """Pure trigger decision for the adaptive-schedule-horizons feature
    (2026-07-18): fires exactly once, the first iteration at which no new
    frame has been ingested for ``threshold`` iterations.

    ``last_ingest_iter > first_iter`` guards against firing before mapping
    has ever produced a single ingested frame (e.g. a source that's slow to
    start) -- ``last_ingest_iter`` only advances once real ingestion happens,
    so this is equivalent to "at least one frame has been ingested".
    """
    return (
        adaptive_on
        and not already_applied
        and last_ingest_iter > first_iter
        and (iteration - last_ingest_iter) >= threshold
    )
