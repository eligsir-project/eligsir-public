"""``MappingCoordinator`` — the real per-frame/per-mode integration entry
point, per Wave 1 consolidation plan step 10 (see
the mapping coordinator ownership boundary
and ``docs/architecture/mapping_coordinator_plan.md`` for the full design
history, including two corrections below).

**Scope correction 1 (2026-08-05):** the plan's original design assumed a
single ``coordinator.step(frame) -> MappingCycleResult`` method could
uniformly drive all three of ``train_streaming.py``'s integration modes.
Tracing the real call sites found this is not how the code is actually
shaped: ``_run_rolling_seed``/``_run_submap_stitch`` are each a complete,
self-contained alternate training loop selected once at mode-dispatch
time, not per-frame cycle calls the way ``insert_gaussians_from_frame``
is. This class exposes mode-specific entry points instead of forcing one
signature.

**Scope correction 2 (2026-08-05, same day):** an earlier version of this
class delegated to three separate wrapper modules
(``rgbd_integrator.py``/``bootstrap_integrator.py``/
``submap_integrator.py``), each itself a one-line lazy-import pass-through
back into ``train_streaming.py``. That produced a three-hop
call chain (``train_streaming`` -> ``MappingCoordinator.step`` ->
``RGBDIntegrator.integrate`` -> ``train_streaming.insert_gaussians_from_frame``)
that added indirection without adding ownership -- exactly the
boilerplate the consolidation task explicitly forbids. Collapsed: this
class now holds the three lazy imports directly. The
:class:`eligsir.mapping.integration.contracts` Protocols
(``RGBDIntegrator``, ``BootstrapIntegrator``, ``SubmapIntegrator``) are
satisfied by this class's own methods now, not by separate wrapper
classes.

**Wave 2 Task 2 update (2026-08-05):** ``insert_gaussians_from_frame``/
``_run_rolling_seed``/``_run_submap_stitch`` and their full helper
cluster have since physically moved (verbatim, no logic change) to
:mod:`eligsir.mapping.integration.rgbd_integrator_impl`. This class
now imports and calls them directly (module-level import, no more lazy
``from train_streaming import ...`` inside the methods) — the circular
import that previously required the lazy pattern no longer exists
because the functions no longer live in ``train_streaming.py``.
``train_streaming.py`` itself now re-exports the same names from that
module so every existing call site (including tests that still do
``from train_streaming import insert_gaussians_from_frame``) keeps
working unchanged.

This is real, called code: ``train_streaming.py``'s ``streaming_training()``
constructs one ``_coordinator`` instance per run (a module-level singleton
until 2026-08-06 -- that mixed telemetry across runs sharing a process and
could keep a finished run's model GPU-reachable) and calls it from all
three of its integration call sites (the per-frame depth-insertion call in
the main loop, and the ``rolling_seed``/``submap_stitch`` mode-dispatch
branches).
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Optional

from eligsir.optimization.gaussian.engine import GaussianOptimizationEngine
from eligsir.mapping.candidates.contracts import (
    PreparedFrameCandidate,
    TrackedFramePacket,
)
from eligsir.mapping.commit_queue import CommitRequest, PendingCommitQueue
from eligsir.runtime.queues.bounded_queue import BoundedQueue, QueueOfferResult, QueuePollResult
from eligsir.runtime.workers.frame_preparation_worker import (
    FramePreparationWorker,
    FramePreparationWorkerConfig,
)


@dataclass(frozen=True)
class MappingCycleResult:
    """Typed summary of one normal-mode :meth:`MappingCoordinator.step`
    call. As of 2026-08-06 this is what ``step()`` itself returns
    directly (it used to return the raw ``(candidates_or_n, stats)``
    tuple, with call sites building this type themselves afterward).

    ``raw`` still carries the original ``candidates_or_n`` value verbatim
    (an ``int`` count in the normal path, or a candidates ``dict`` under
    the ``_collect_only`` batching mode) — call sites that need the batch
    dict for later flushing read it from here instead of getting a second
    return value.

    Only covers the ``step()`` (normal, per-frame) path — the
    ``run_rolling_seed_mode()``/``run_submap_stitch_mode()`` paths run a
    complete internal training loop and return whatever
    ``_run_rolling_seed``/``_run_submap_stitch`` return today (``None``),
    not a ``MappingCycleResult``; unifying that is future work once those
    two functions are decomposed into real per-cycle steps.
    """

    source_frame_id: int
    integrated: bool
    n_inserted: int
    stats: dict
    raw: Any = None

    @classmethod
    def from_step_result(cls, source_frame_id: int, cands_or_n: Any, stats: dict) -> "MappingCycleResult":
        n_inserted = cands_or_n if isinstance(cands_or_n, int) else 0
        # `integrated` reflects whether this step actually committed new
        # geometry. Under `_collect_only` batching, cands_or_n is a
        # candidates dict (not yet inserted -- flushed later), so
        # n_inserted stays 0 and integrated is correctly False for that
        # case too.
        return cls(
            source_frame_id=source_frame_id,
            integrated=n_inserted > 0,
            n_inserted=n_inserted,
            stats=stats,
            raw=cands_or_n,
        )


@dataclass(frozen=True)
class FourClocksSummary:
    """The "four clocks" telemetry for a streaming run, per Wave 1 plan §2
    item 2 (deferred to Wave 2 Task 8): a live-SLAM system has (at least)
    four independent rates in flight at once -- how fast frames *arrive*,
    how fast they get *admitted* as keyframes, how fast the map actually
    *commits* new geometry, and whether the map is falling behind the
    input -- and conflating them (e.g. reading ``it/s`` as "mapping speed")
    has repeatedly caused wrong conclusions in this project's own eval
    history (see CLAUDE.md's streaming-eval-metric-unreliable memory).
    This type makes the four rates explicit and separately inspectable.

    All rate fields are computed over a bounded recent window (see
    :class:`FourClocksAggregator`), not since process start, so they
    reflect *current* behavior rather than an all-run average that a slow
    start or a long refinement tail would wash out.

    Rates decay toward zero if a stream stalls (F1 fix, 2026-08-06): only
    events within the last :attr:`FourClocksAggregator.RATE_WINDOW_S` wall
    seconds count toward a rate, so a source that stops entirely shows a
    correctly-decaying/zero rate rather than staying frozen at whatever it
    was between the last ``window`` buffered events.

    Fields:
        source_frame_rate_hz: wall-clock rate at which source frames are
            ingested (``StreamingScene.ingest_next_frame`` returning
            non-``None``), regardless of whether they're admitted as a
            keyframe.
        admitted_keyframe_rate_hz: wall-clock rate at which ingested
            frames are actually admitted as training keyframes
            (``is_train`` from the ingest result) -- this is the
            "mapFPS" the progress bar already shows inline, now also
            persisted.
        map_commit_rate_hz: wall-clock rate at which the map actually
            gains new geometry (``MappingCoordinator.step()`` producing
            ``n_inserted > 0``) -- distinct from admission rate because
            an admitted keyframe can still contribute zero new Gaussians
            (e.g. depth insertion is capped, gated off past the
            insertion-stop horizon, or the frame adds nothing new).
        map_age_ms: **deprecated alias** for
            ``last_observation_integration_age_ms`` (F3 fix, 2026-08-06) --
            wall-clock time since the last positive Gaussian *insertion*,
            in milliseconds. Kept for existing ``report.json`` consumers;
            prefer the explicitly-named fields below, since this one is
            easy to misread as "map freshness" when it really only reflects
            insertion activity (see those two fields' docstrings).
        last_observation_integration_age_ms: wall-clock time since the last
            positive Gaussian *insertion* (``MappingCoordinator.step()``
            producing ``n_inserted > 0``) -- ignores optimization steps,
            pruning, relocation, and loop-closure corrections, all of which
            also change the map but are NOT "new observation integrated."
            ``None`` if no insertion has happened yet this run. Identical
            value to ``map_age_ms``.
        last_topology_change_age_ms: wall-clock time since the map's
            ``topology_version`` counter (see
            ``eligsir.map.contracts.GaussianMapBackend``) last advanced
            -- i.e. since the last structural change (insertion, prune,
            relocation) as opposed to a same-shape parameter update.
            ``None`` if no topology change has been observed yet.
        last_parameter_update_age_ms: wall-clock time since the map's
            ``map_version`` counter last advanced at all -- the broadest
            freshness clock, covering every committed mutation
            (``begin_mutation``/``commit_mutation``), including
            same-shape optimization/loop-closure-correction updates that
            don't touch ``topology_version``. ``None`` if no mutation has
            been observed yet.
        source_queue_depth: current count of source frames that have
            already arrived at the frame source but have not yet been
            consumed by ``ingest_next_frame`` -- the closest thing to a
            real queue depth this aggregator has visibility into (derived
            from the frame source's own ``buffered_count()``, not
            duplicated here). ``None`` when the source doesn't expose a
            buffered count (e.g. a fully-materialized offline dataset has
            no queue at all) or none has been reported yet.
        frames_rejected_total: cumulative count of source frames that
            reached a real candidate ``cam`` but were rejected by
            admission/pregate logic (``is_train=False`` and not a holdout).
        holdout_frames_total: cumulative count of source frames
            deterministically reserved as real test holdouts (see
            CLAUDE.md's "Real test holdouts" contract) -- never trained,
            replayed, or depth-inserted by design, not a pipeline fault.
        frames_unprocessed: cumulative source frames ingested minus
            (admitted + rejected + holdout) -- frames whose fate hasn't
            been recorded yet through this aggregator's own event streams
            (e.g. still in flight through pregate/admission at summary
            time). Distinct from ``input_backlog_frames`` below, which
            conflates holdouts and rejections into one number.
        input_backlog_frames: **deprecated, kept for existing consumers**
            (F2 fix, 2026-08-06) -- cumulative source frames ingested minus
            cumulative frames admitted as keyframes. This number is NOT a
            real queue depth: it silently includes both intentional
            holdouts (see ``holdout_frames_total``) and genuinely rejected
            frames (see ``frames_rejected_total``), and is structurally
            non-zero even under ``admission=all`` because holdouts are
            never admitted by design. Prefer ``source_queue_depth`` for
            actual backlog/queue monitoring, and
            ``frames_rejected_total``/``holdout_frames_total`` to see the
            two conflated components of this field separately.
        realtime_factor: ratio of source-dataset time elapsed to
            wall-clock time elapsed over the same recent window of
            ingested frames (``frame.timestamp`` deltas /
            ``time.perf_counter()`` deltas). ``1.0`` means the pipeline
            is exactly keeping pace with the dataset's own recording
            rate; ``< 1.0`` means it is falling behind (mapping slower
            than the sensor produced the data); ``> 1.0`` means it is
            running ahead (faster than real time, e.g. deterministic
            replay with no wallclock pacing). ``None`` when source frame
            timestamps aren't available (e.g. a live ROS2 topic with no
            per-frame dataset clock) or not enough samples have been
            recorded yet.
    """

    source_frame_rate_hz: float
    admitted_keyframe_rate_hz: float
    map_commit_rate_hz: float
    map_age_ms: Optional[float]
    last_observation_integration_age_ms: Optional[float]
    last_topology_change_age_ms: Optional[float]
    last_parameter_update_age_ms: Optional[float]
    source_queue_depth: Optional[int]
    frames_rejected_total: int
    holdout_frames_total: int
    frames_unprocessed: int
    input_backlog_frames: int
    realtime_factor: Optional[float]


class FourClocksAggregator:
    """Turns the raw event streams (source-frame ingest, keyframe
    admission, map commit, map-version changes) into a
    :class:`FourClocksSummary`.

    Rates are computed over a bounded **wall-time** window
    (:attr:`RATE_WINDOW_S`, default 5.0s) of the buffered events (``maxlen``
    events per stream, default 64, just bounds memory) rather than a
    whole-run or fixed-event-count average (F1 fix, 2026-08-06) -- so a
    slow warmup or a long post-coverage refinement tail (where no new
    frames arrive at all) doesn't wash out what the pipeline is doing
    *right now*, AND a source that stalls entirely correctly decays to a
    zero rate instead of staying frozen at whatever it was between the
    last ``window`` events (the previous event-count-only windowing could
    hold a frozen nonzero rate indefinitely across an arbitrarily long
    stall). Cumulative counters (used for ``input_backlog_frames`` and the
    new F2 backlog-breakdown fields) are tracked separately and never
    trimmed, since backlog is a running total, not a rate.

    This class holds no reference to any training-loop state; call sites
    (currently ``train_streaming.py``'s main loop and
    :class:`MappingCoordinator`) feed it events directly at the point each
    one actually happens.
    """

    #: Wall-time window (seconds) a rate is computed over -- see class
    #: docstring's F1 fix note. Events older than this relative to
    #: ``summary(now=...)``'s ``now`` are excluded from the rate
    #: computation entirely (not just capped at ``window`` count), which is
    #: what makes a stalled stream's rate correctly decay to 0.0 rather
    #: than stay frozen.
    RATE_WINDOW_S = 5.0

    def __init__(self, window: int = 64, rate_window_s: float = RATE_WINDOW_S) -> None:
        self._rate_window_s = rate_window_s
        self._source_events: "deque[tuple[float, Optional[float]]]" = deque(maxlen=window)
        self._admitted_events: "deque[float]" = deque(maxlen=window)
        self._commit_events: "deque[float]" = deque(maxlen=window)
        self._n_source_total = 0
        self._n_admitted_total = 0
        self._n_rejected_total = 0
        self._n_holdout_total = 0
        self._last_commit_wall_ts: Optional[float] = None
        # F3: broader freshness clocks, driven by the backend's own
        # map_version/topology_version counters (see record_map_version).
        self._last_seen_map_version: Optional[int] = None
        self._last_seen_topology_version: Optional[int] = None
        self._last_topology_change_wall_ts: Optional[float] = None
        self._last_parameter_update_wall_ts: Optional[float] = None
        # F2: latest known queue depth, pushed by the caller (derived from
        # the frame source's own buffered_count(), not duplicated here).
        self._source_queue_depth: Optional[int] = None

    _RESUME_SCHEMA = "four_clocks_v1"

    @staticmethod
    def _resume_number(value: Any, name: str, *, nonnegative: bool = False) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"four clocks {name} must be numeric")
        result = float(value)
        if not math.isfinite(result) or (nonnegative and result < 0.0):
            raise ValueError(f"four clocks {name} is invalid")
        return result

    def resume_state_dict(self) -> dict[str, Any]:
        now = time.perf_counter()
        age = lambda timestamp: None if timestamp is None else now - timestamp
        return {
            "owner_schema": self._RESUME_SCHEMA,
            "window": int(self._source_events.maxlen),
            "rate_window_s": float(self._rate_window_s),
            "source_events": [
                (age(wall_ts), frame_timestamp)
                for wall_ts, frame_timestamp in self._source_events
            ],
            "admitted_event_ages": [age(ts) for ts in self._admitted_events],
            "commit_event_ages": [age(ts) for ts in self._commit_events],
            "n_source_total": int(self._n_source_total),
            "n_admitted_total": int(self._n_admitted_total),
            "n_rejected_total": int(self._n_rejected_total),
            "n_holdout_total": int(self._n_holdout_total),
            "last_commit_age": age(self._last_commit_wall_ts),
            "last_seen_map_version": self._last_seen_map_version,
            "last_seen_topology_version": self._last_seen_topology_version,
            "last_topology_change_age": age(self._last_topology_change_wall_ts),
            "last_parameter_update_age": age(self._last_parameter_update_wall_ts),
            "source_queue_depth": self._source_queue_depth,
        }

    def load_resume_state_dict(self, payload, resolver=None) -> None:
        expected = set(self.resume_state_dict())
        if (
            not isinstance(payload, dict)
            or set(payload) != expected
            or payload.get("owner_schema") != self._RESUME_SCHEMA
        ):
            raise ValueError("unsupported four clocks checkpoint state")
        window = payload["window"]
        if isinstance(window, bool) or not isinstance(window, int) or window <= 0:
            raise ValueError("four clocks window must be positive")
        rate_window_s = self._resume_number(
            payload["rate_window_s"], "rate_window_s", nonnegative=True
        )
        if rate_window_s <= 0.0:
            raise ValueError("four clocks rate_window_s must be positive")

        def ages(values, name, *, source=False):
            if not isinstance(values, list) or len(values) > window:
                raise ValueError(f"four clocks {name} is invalid")
            out = []
            for value in values:
                if source:
                    if not isinstance(value, (list, tuple)) or len(value) != 2:
                        raise ValueError("four clocks source_events is invalid")
                    event_age = self._resume_number(value[0], name, nonnegative=True)
                    frame_ts = value[1]
                    if frame_ts is not None:
                        frame_ts = self._resume_number(frame_ts, "frame_timestamp")
                    out.append((event_age, frame_ts))
                else:
                    out.append(self._resume_number(value, name, nonnegative=True))
            return out

        source_events = ages(payload["source_events"], "source_events", source=True)
        admitted_ages = ages(payload["admitted_event_ages"], "admitted_event_ages")
        commit_ages = ages(payload["commit_event_ages"], "commit_event_ages")

        counters = {}
        for name in (
            "n_source_total", "n_admitted_total", "n_rejected_total", "n_holdout_total"
        ):
            value = payload[name]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"four clocks {name} must be a non-negative integer")
            counters[name] = value
        if any(counters[name] > counters["n_source_total"] for name in (
            "n_admitted_total", "n_rejected_total", "n_holdout_total"
        )):
            raise ValueError("four clocks cumulative counters are inconsistent")

        def optional_age(name):
            value = payload[name]
            return None if value is None else self._resume_number(value, name, nonnegative=True)

        def optional_nonnegative_int(name):
            value = payload[name]
            if value is None:
                return None
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"four clocks {name} is invalid")
            return value

        last_commit_age = optional_age("last_commit_age")
        last_topology_age = optional_age("last_topology_change_age")
        last_parameter_age = optional_age("last_parameter_update_age")
        last_map_version = optional_nonnegative_int("last_seen_map_version")
        last_topology_version = optional_nonnegative_int("last_seen_topology_version")
        source_queue_depth = optional_nonnegative_int("source_queue_depth")

        now = time.perf_counter()
        self._rate_window_s = rate_window_s
        self._source_events = deque(
            ((now - event_age, frame_ts) for event_age, frame_ts in source_events),
            maxlen=window,
        )
        self._admitted_events = deque((now - value for value in admitted_ages), maxlen=window)
        self._commit_events = deque((now - value for value in commit_ages), maxlen=window)
        self._n_source_total = counters["n_source_total"]
        self._n_admitted_total = counters["n_admitted_total"]
        self._n_rejected_total = counters["n_rejected_total"]
        self._n_holdout_total = counters["n_holdout_total"]
        self._last_commit_wall_ts = None if last_commit_age is None else now - last_commit_age
        self._last_seen_map_version = last_map_version
        self._last_seen_topology_version = last_topology_version
        self._last_topology_change_wall_ts = (
            None if last_topology_age is None else now - last_topology_age
        )
        self._last_parameter_update_wall_ts = (
            None if last_parameter_age is None else now - last_parameter_age
        )
        self._source_queue_depth = source_queue_depth

    def _rate_hz(self, events, now: float) -> float:
        """Rate over the last :attr:`RATE_WINDOW_S` wall seconds of
        ``events`` relative to ``now``. Events outside that window are
        excluded entirely -- a stalled stream (no recent events) reports
        0.0 instead of an average of its last ``window``-sized buffer
        (which could be arbitrarily stale)."""
        recent = [t for t in events if (now - t) <= self._rate_window_s]
        if len(recent) < 2:
            return 0.0
        span = recent[-1] - recent[0]
        if span <= 0:
            return 0.0
        return (len(recent) - 1) / span

    def record_source_frame(self, wall_ts: Optional[float] = None, frame_timestamp: Optional[float] = None) -> None:
        """Call once per source frame ingested (regardless of admission)."""
        wall_ts = time.perf_counter() if wall_ts is None else wall_ts
        self._source_events.append((wall_ts, frame_timestamp))
        self._n_source_total += 1

    def record_admitted_keyframe(self, wall_ts: Optional[float] = None) -> None:
        """Call once per ingested frame that is admitted as a training keyframe."""
        wall_ts = time.perf_counter() if wall_ts is None else wall_ts
        self._admitted_events.append(wall_ts)
        self._n_admitted_total += 1

    def record_frame_rejected(self) -> None:
        """Call once per candidate frame rejected by admission/pregate logic
        (F2 fix) -- a genuine backlog component, distinct from a holdout."""
        self._n_rejected_total += 1

    def record_frame_holdout(self) -> None:
        """Call once per candidate frame reserved as a deterministic real
        test holdout (F2 fix) -- intentionally never admitted, not a
        pipeline fault; tracked separately from rejected frames."""
        self._n_holdout_total += 1

    def record_source_queue_depth(self, depth: Optional[int]) -> None:
        """Push the latest known pending-frame count at the frame source
        (F2 fix). ``None`` means the source doesn't expose one (e.g. a
        fully-materialized offline dataset)."""
        self._source_queue_depth = depth

    def record_map_commit(self, wall_ts: Optional[float] = None, n_inserted: int = 0) -> None:
        """Call once per mapping cycle that ran (``MappingCoordinator.step``),
        regardless of whether it actually inserted anything -- ``n_inserted``
        is accepted for future use (e.g. weighting commit rate by geometry
        volume) but is not currently part of the summary, which only asks
        "did the map change," not "by how much" (that's ``n_inserted``
        itself, already a separate TensorBoard scalar).

        This drives ``map_age_ms``/``last_observation_integration_age_ms``
        specifically -- INSERTION only (F3 fix: see
        :meth:`record_map_version` for the broader clocks that also cover
        pruning/relocation/loop-closure).
        """
        if n_inserted <= 0:
            return
        wall_ts = time.perf_counter() if wall_ts is None else wall_ts
        self._commit_events.append(wall_ts)
        self._last_commit_wall_ts = wall_ts

    def record_map_version(
        self, map_version: int, topology_version: int, wall_ts: Optional[float] = None
    ) -> None:
        """Feed the backend's current ``map_version``/``topology_version``
        counters (F3 fix, 2026-08-06). A version-number CHANGE since the
        last call is the signal -- this catches ANY committed mutation
        (insertion, pruning, relocation, loop-closure correction,
        same-shape optimization updates), not just positive insertions, by
        construction of ``GaussianMapBackend.begin_mutation``/
        ``commit_mutation`` (every committed transaction bumps
        ``map_version``; ``topology_version`` bumps additionally only when
        the mutation actually changed Gaussian count/structure).

        Cheap to call every iteration: callers should read the version
        counters directly off the backend (plain ints) rather than via
        ``backend.read_view()``, which materializes full Gaussian tensors.
        The first call only establishes a baseline (no age recorded) so a
        restored checkpoint's nonzero starting versions don't look like an
        instantaneous change at process start.
        """
        wall_ts = time.perf_counter() if wall_ts is None else wall_ts
        if self._last_seen_map_version is None:
            self._last_seen_map_version = map_version
            self._last_seen_topology_version = topology_version
            return
        if map_version != self._last_seen_map_version:
            self._last_seen_map_version = map_version
            self._last_parameter_update_wall_ts = wall_ts
        if topology_version != self._last_seen_topology_version:
            self._last_seen_topology_version = topology_version
            self._last_topology_change_wall_ts = wall_ts

    def summary(self, now: Optional[float] = None) -> FourClocksSummary:
        now = time.perf_counter() if now is None else now
        source_rate = self._rate_hz([t for t, _ in self._source_events], now)
        admitted_rate = self._rate_hz(self._admitted_events, now)
        commit_rate = self._rate_hz(self._commit_events, now)
        map_age_ms = (
            (now - self._last_commit_wall_ts) * 1000.0
            if self._last_commit_wall_ts is not None
            else None
        )
        topology_age_ms = (
            (now - self._last_topology_change_wall_ts) * 1000.0
            if self._last_topology_change_wall_ts is not None
            else None
        )
        param_age_ms = (
            (now - self._last_parameter_update_wall_ts) * 1000.0
            if self._last_parameter_update_wall_ts is not None
            else None
        )
        backlog = max(0, self._n_source_total - self._n_admitted_total)
        frames_unprocessed = max(
            0,
            self._n_source_total
            - self._n_admitted_total
            - self._n_rejected_total
            - self._n_holdout_total,
        )

        realtime_factor: Optional[float] = None
        _ts_events = [(t, fts) for t, fts in self._source_events if fts is not None]
        if len(_ts_events) >= 2:
            wall_span = _ts_events[-1][0] - _ts_events[0][0]
            source_span = _ts_events[-1][1] - _ts_events[0][1]
            if wall_span > 0 and source_span >= 0:
                realtime_factor = source_span / wall_span

        return FourClocksSummary(
            source_frame_rate_hz=source_rate,
            admitted_keyframe_rate_hz=admitted_rate,
            map_commit_rate_hz=commit_rate,
            map_age_ms=map_age_ms,
            last_observation_integration_age_ms=map_age_ms,
            last_topology_change_age_ms=topology_age_ms,
            last_parameter_update_age_ms=param_age_ms,
            source_queue_depth=self._source_queue_depth,
            frames_rejected_total=self._n_rejected_total,
            holdout_frames_total=self._n_holdout_total,
            frames_unprocessed=frames_unprocessed,
            input_backlog_frames=backlog,
            realtime_factor=realtime_factor,
        )


class BoundedCandidatePipeline:
    """The bounded-queue prepared-candidate pipeline, wired together (plan
    section 8.6, Subagent D: "coordinator integration"). Owned by
    :class:`MappingCoordinator`, constructed only when a caller opts in via
    :meth:`MappingCoordinator.enable_bounded_queue_pipeline` -- this is the
    feature-flagged path (deliverable #6: the default/current path is
    completely unaffected unless this is explicitly enabled).

    Three bounded queues, all real ``BoundedQueue``/``PendingCommitQueue``
    instances (deliverable #1 -- every runtime queue in this path is
    bounded, none are `queue.Queue()`/`list`/`deque` with unbounded
    growth):

    - ``input_queue``: raw ``TrackedFramePacket``s, producer = whatever
      feeds the SLAM-tracked frame stream, consumer = the
      ``FramePreparationWorker`` thread.
    - ``output_queue``: prepared candidates, producer = the worker thread,
      consumer = the optimizer thread via :meth:`poll_prepared_candidate`
      (non-blocking -- deliverable #2).
    - ``commit_queue``: accepted candidates awaiting a batched map
      mutation, drained at an explicit barrier by the optimizer thread.

    All three queue depths feed :attr:`MappingCoordinator.four_clocks`
    directly from queue metrics (deliverable #5 -- backlog is read off the
    real queue, never inferred from a separate counter).
    """

    def __init__(
        self,
        *,
        input_capacity: int = 4,
        output_capacity: int = 4,
        commit_capacity: int = 8,
        worker_config: Optional[FramePreparationWorkerConfig] = None,
    ) -> None:
        self.input_queue: BoundedQueue[TrackedFramePacket] = BoundedQueue(capacity=input_capacity)
        self.output_queue: BoundedQueue[PreparedFrameCandidate] = BoundedQueue(capacity=output_capacity)
        self.commit_queue: PendingCommitQueue[CommitRequest] = PendingCommitQueue(capacity=commit_capacity)
        self.worker = FramePreparationWorker(
            self.input_queue, self.output_queue, config=worker_config,
        ).start()

    def submit_raw_frame(self, tracked: TrackedFramePacket) -> QueueOfferResult:
        """Non-blocking offer of a newly SLAM-tracked frame. Never waits --
        under backpressure this returns ``DROPPED_QUEUE_FULL`` rather than
        blocking the caller (which, in the live path, is the same thread
        driving frame arrival)."""
        return self.input_queue.try_offer(tracked)

    def poll_prepared_candidate(self) -> QueuePollResult[PreparedFrameCandidate]:
        """Non-blocking poll for the optimizer thread. Never waits for the
        preparation worker -- an ``EMPTY`` result means "keep training on
        buffered data this iteration," exactly the existing
        ``IngestionWorker.poll()`` idiom this mirrors."""
        return self.output_queue.try_poll()

    def submit_commit(self, request: CommitRequest) -> QueueOfferResult:
        return self.commit_queue.submit(request)

    def drain_commits(self, *, max_items: Optional[int] = None) -> list:
        """Barrier drain: pop everything (or up to ``max_items``) queued
        for commit right now, for the caller to apply as one batch."""
        return self.commit_queue.drain(max_items=max_items)

    def queue_depths(self) -> dict:
        """Real, current depths of all three queues -- the source of truth
        for :meth:`MappingCoordinator.sync_source_backlog_from_pipeline`."""
        return {
            "input": self.input_queue.metrics().depth,
            "output": self.output_queue.metrics().depth,
            "commit": self.commit_queue.metrics().depth,
        }

    def shutdown(self) -> None:
        self.worker.stop()
        self.input_queue.close()
        self.output_queue.close()
        self.commit_queue.close()


class MappingCoordinator:
    """Real entry point for all three ``train_streaming.py`` integration
    modes, satisfying the ``RGBDIntegrator``/``BootstrapIntegrator``/
    ``SubmapIntegrator`` Protocols (:mod:`eligsir.mapping.integration.contracts`)
    directly rather than through separate wrapper classes.

    Each method calls its target directly, imported at module load time
    from :mod:`eligsir.mapping.integration.rgbd_integrator_impl`
    (Wave 2 Task 2 — see the module docstring's "Wave 2 Task 2 update").
    ``insert_gaussians_from_frame``/``_run_rolling_seed``/
    ``_run_submap_stitch`` and their full helper cluster now physically
    live in that module, not in ``train_streaming.py``. Behavior-neutrality
    of the earlier delegation was proven by
    ``tests/test_integration_services_delegate.py`` (real-GPU parity
    against calling the ``train_streaming.py`` functions directly) and by
    real training runs on ``rgbd_dataset_freiburg1_xyz`` through this
    exact call path; the physical move preserves the same call contract
    (verbatim body relocation, no logic change) so that parity result
    still holds.

    ``FramePreparationService``/``IntegrationPlanner`` are not implemented
    here — they remain Protocol-only because ``insert_gaussians_from_frame``
    has no existing split point to give them real work. The six
    ``StreamingScene`` narrow stores (step 9) are likewise not composed
    here yet; ``StreamingScene`` itself is still passed through to
    ``step()`` as-is, per that step's own deferred physical-extraction
    scope.
    """

    def __init__(self) -> None:
        self._backend: Any = None
        self._backend_gaussians_id: int | None = None
        self._optimization_engine: "GaussianOptimizationEngine | None" = None
        self._optimization_engine_gaussians_id: int | None = None
        # Four-clocks telemetry (Wave 2 Task 8): a stateful aggregator, not
        # a per-call computation, because the rates are only meaningful over
        # a recent rolling window -- see FourClocksAggregator's docstring.
        self.four_clocks = FourClocksAggregator()
        # Bounded-queue prepared-candidate pipeline (PR W3-03): None until
        # explicitly enabled via enable_bounded_queue_pipeline(). Every
        # call site below is a no-op / raises clearly when it's None, so a
        # coordinator built without opting in behaves exactly as before
        # (deliverable #6 -- feature flag allows immediate fallback).
        self.bounded_pipeline: "BoundedCandidatePipeline | None" = None

    def backend_for(self, gaussians: Any) -> Any:
        """Return the :class:`~eligsir.map.contracts.GaussianMapBackend`
        for ``gaussians``, building (and caching) it on first use.

        This is the **sole construction site** for a run's
        ``GaussianMapBackend`` (Wave 2 ownership-consolidation pass,
        2026-08-06). Every consumer that needs backend-mediated access to
        the live Gaussian model -- checkpointing, viewer reads, structural
        mutation, optimization -- should obtain it from here (directly, or
        indirectly via :meth:`optimization_engine_for`, which now delegates
        to this method rather than maintaining an independent cache) so
        that exactly one backend instance, with one authoritative
        ``map_version``/``topology_version`` pair, exists per run. Two
        independently-cached backend instances for the same model would
        silently disagree on those counters.

        ``gaussians`` is the same live model instance for an entire
        training run (constructed once, never reassigned -- confirmed by
        inspection of ``train_streaming.py``), so the backend is built once
        and reused; rebuilding it would reset ``map_version``/
        ``topology_version`` back to 0, discarding real history.
        """
        if self._backend is None or self._backend_gaussians_id != id(gaussians):
            from scene.gsplat_model import GsplatGaussianModel

            if isinstance(gaussians, GsplatGaussianModel):
                from eligsir.map.adapters.gsplat import GsplatMapBackend

                self._backend = GsplatMapBackend(gaussians)
            else:
                from eligsir.map.adapters.legacy_graphdeco import LegacyGraphdecoMapBackend

                self._backend = LegacyGraphdecoMapBackend(gaussians)
            self._backend_gaussians_id = id(gaussians)
        return self._backend

    def optimization_engine_for(self, gaussians: Any) -> GaussianOptimizationEngine:
        """Return the :class:`GaussianOptimizationEngine` for ``gaussians``,
        building (and caching) it on first use per Wave 2 step 1's
        "coordinator composes the real services" pattern (matching how
        ``_coordinator`` is already a module-level singleton in
        ``train_streaming.py`` that outlives any single frame/iteration).

        Wraps whatever backend :meth:`backend_for` returns -- see that
        method's docstring for why this must not build its own independent
        backend instance/cache.
        """
        if self._optimization_engine is None or self._optimization_engine_gaussians_id != id(gaussians):
            backend = self.backend_for(gaussians)
            self._optimization_engine = GaussianOptimizationEngine(backend)
            self._optimization_engine_gaussians_id = id(gaussians)
        return self._optimization_engine

    def poll_map_activity(self) -> None:
        """Cheap, tensor-free check for whether the map's version counters
        have advanced since the last poll, driving the F3
        ``last_topology_change_age_ms``/``last_parameter_update_age_ms``
        clocks (see ``FourClocksAggregator.record_map_version``'s
        docstring for why version-number diffing is the mechanism, not
        instrumenting every individual mutation call site).

        Reads the backend's own ``_map_version``/``_topology_version``
        attributes directly rather than through ``backend_for(...)
        .read_view()``, which would materialize full Gaussian tensors --
        safe to call every training iteration. No-op if no backend has
        been constructed yet (``backend_for`` never called this run).
        """
        backend = self._backend
        if backend is None:
            return
        map_version = getattr(backend, "map_content_version", None)
        if map_version is None:
            map_version = getattr(backend, "_map_version", None)
        if map_version is None:
            return
        topology_version = getattr(backend, "_topology_version", 0)
        self.four_clocks.record_map_version(map_version, topology_version)

    def step(
        self,
        gaussians: Any,
        frame: Any,
        args: Any,
        streaming_scene: Any,
        *,
        current_frame_idx: int = 0,
        **kwargs: Any,
    ) -> MappingCycleResult:
        """Normal-mode per-frame integration step.

        Calls ``insert_gaussians_from_frame`` and wraps its raw
        ``(candidates_or_n, stats)`` tuple into a typed
        :class:`MappingCycleResult`. The original ``candidates_or_n``
        value (an ``int`` count, or a candidates ``dict`` under the
        ``_collect_only`` batching mode) is preserved verbatim on
        ``result.raw`` for call sites that need it (e.g. batch flushing).
        """
        # Historical experiment/replay path. Keep this compatibility surface
        # out of module-import reachability: canonical live mapping never calls
        # ``step`` and imports its typed device producer directly.
        from eligsir.mapping.integration.rgbd_integrator_impl import (
            insert_gaussians_from_frame,
        )

        cands_or_n, stats = insert_gaussians_from_frame(
            gaussians, frame, args, streaming_scene,
            current_frame_idx=current_frame_idx, **kwargs,
        )
        n_inserted = cands_or_n if isinstance(cands_or_n, int) else 0
        self.four_clocks.record_map_commit(n_inserted=n_inserted)
        # Ensure a backend exists to poll versions from (cheap: pure-wrapper
        # construction, no tensor copies -- see backend_for's docstring).
        self.backend_for(gaussians)
        self.poll_map_activity()
        return MappingCycleResult.from_step_result(current_frame_idx, cands_or_n, stats)

    def enable_bounded_queue_pipeline(
        self,
        *,
        input_capacity: int = 4,
        output_capacity: int = 4,
        commit_capacity: int = 8,
        worker_config: Optional[FramePreparationWorkerConfig] = None,
    ) -> "BoundedCandidatePipeline":
        """Opt into the bounded-queue prepared-candidate pipeline. Only
        called by ``train_streaming.py`` when
        ``--streaming_bounded_queue_pipeline`` is set (default off) --
        see that flag's docstring in ``arguments/__init__.py``.
        Idempotent: returns the existing pipeline if already enabled
        rather than leaking a second worker thread.
        """
        if self.bounded_pipeline is None:
            self.bounded_pipeline = BoundedCandidatePipeline(
                input_capacity=input_capacity,
                output_capacity=output_capacity,
                commit_capacity=commit_capacity,
                worker_config=worker_config,
            )
        return self.bounded_pipeline

    def sync_source_backlog_from_pipeline(self) -> None:
        """Feed the bounded pipeline's real input-queue depth into
        ``four_clocks`` (deliverable #5 -- source backlog measured from an
        actual queue, not an inferred/emergent counter). No-op if the
        pipeline was never enabled."""
        if self.bounded_pipeline is None:
            return
        self.four_clocks.record_source_queue_depth(self.bounded_pipeline.queue_depths()["input"])

    def shutdown_bounded_queue_pipeline(self) -> None:
        if self.bounded_pipeline is not None:
            self.bounded_pipeline.shutdown()
            self.bounded_pipeline = None

    def run_rolling_seed_mode(self, *args: Any, **kwargs: Any) -> Any:
        """Bootstrap/rolling-seed alternate training-loop mode."""
        from eligsir.mapping.integration.rgbd_integrator_impl import (
            _run_rolling_seed,
        )

        return _run_rolling_seed(*args, **kwargs)

    def run_submap_stitch_mode(self, *args: Any, **kwargs: Any) -> Any:
        """Submap-stitching alternate training-loop mode."""
        from eligsir.mapping.integration.rgbd_integrator_impl import (
            _run_submap_stitch,
        )

        return _run_submap_stitch(*args, **kwargs)
