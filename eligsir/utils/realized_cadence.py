"""Realized-cadence accounting for streaming runs (issue #289).

Why this exists
---------------
Seed fade / streaming-quality failures are **rate-and-capacity** phenomena: how
much new data arrived, and how many optimizer updates the map got to assimilate
it. Every causal story told about a streaming PSNR curve therefore rests on an
implicit claim about the run's clocks -- and this repo has now been burned three
separate ways by that claim being false:

  * #278 (confirmed from primary artifacts, 2026-08-14): a run advertising
    ``effective_spf=150`` released its whole 2525-frame trajectory by optimizer
    step ~1250, because ``--online_mode`` resolves ingestion to ``dataset_fps``
    where ``FrameScheduler.should_release()`` never reads ``steps_per_frame`` at
    all. Realized release cadence: ~0.5 optimizer steps per frame, i.e. ~300x
    faster than advertised. See ``resolve_ingestion_mode_with_reason`` and #302.
  * W3-38 (``docs/architecture/2026-08-09-w3-38-admission-gap.md``): the
    opposite direction -- ``MappingScheduler``'s ``min_steps_per_frame`` floor
    (default 50) silently clamps an explicit lower ``--streaming_steps_per_frame``
    *upward* on registry-registered scenes.
  * The same doc's second defect: ``_coverage_n_frames`` resolves against the
    FULL registered scene length rather than the ``--streaming_max_frames``
    subset, making the clamp fire more readily still.

None of those three are visible in a report that records only *requested*
cadence. This module records the **realized** cadence, from the run's own
counters, and persists it so no future experiment has to reconstruct it
forensically from a progress bar.

The authoritative clocks (issue #289 task 1)
--------------------------------------------
Seven event streams, each with exactly one definition and one increment site:

``optimizer_step``
    One real ``gaussians.optimizer.step()`` call. **This is the denominator for
    everything else**, deliberately not ``iteration``: the two diverge under
    ``--streaming_optimization_batch_size > 1`` (gradient accumulation) and
    under the ``placement_only``/``colors_only`` diagnostic training modes. Both
    counts are recorded so the ratio is visible rather than assumed.
``source_released``
    ``FrameScheduler`` let one source frame through the release gate. Counted on
    the *scheduler* side, so it includes frames released and then dropped by
    ``BurstBudget`` -- release and ingest are different clocks and conflating
    them is precisely the ``F`` vs ``V`` confusion #278 had to disentangle.
``source_ingested``
    A released frame that actually produced a camera (``ingest_next_frame``
    returned non-None). ``source_released - source_ingested`` is the drop count.
``candidate_evaluated``
    An admission decision function was actually invoked for a candidate frame.
    Counted at the decision site rather than derived, because under
    ``--streaming_frame_admission all`` (the default) no gate runs at all and a
    derived value could not tell "no gate" from "instrumentation broken".

    **Conditional as of issue #602 -- do not read the zero as unconditional.**
    "``all`` implies no gate" holds for the *historical* ``all``, which is
    what issue #600 renames ``force_all_stress``. It does NOT hold for #600's
    ``soft_all``: there, every source frame is still *observed*, but mapping
    admission is budgeted and a real decision is made per frame, so
    ``candidate_evaluated`` is legitimately nonzero while the requested policy
    still reads ``all``. Resolve the run's ``admission_target_resolved``
    (``report.json["admission_cadence"]["resolved_semantics"]``) before
    concluding anything from this clock; a zero means "no gate ran" only when
    the resolved target is ``force_all_stress`` (or the run predates #600 and
    the target is unknown).
``keyframe_admitted``
    A frame admitted as a training keyframe (``is_train``). Under the
    historical ``admission=all`` / ``force_all_stress`` this equals
    ``source_ingested``; under ``hybrid_keyframe`` it is the gate's output.

    **Also conditional as of #602.** Under ``soft_all`` this equality is
    expected to BREAK -- observation and admission diverging under pressure is
    the entire point of soft-all, not a symptom of lost frames. Use
    ``source_ingested - keyframe_admitted`` as a coverage-loss signal only
    when the resolved target is not ``soft_all``; under ``soft_all`` the
    declined frames are accounted for by disposition in
    ``report.json["admission_cadence"]["dispositions"]``.
``insertion_commit``
    A depth-insertion flush that actually committed to the map, carrying the
    number of Gaussians inserted. Includes the end-of-run flush.
``supervision_view``
    One rendered/supervised view -- the real photometric-supervision clock,
    which under batch accumulation ticks faster than ``optimizer_step``.
``mcmc_mutation``
    One MCMC growth/relocation mutation event (kept distinct from
    ``insertion_commit``: both are topology mutations but they have different
    causes and different rates).

Cost
----
Every ``record_*`` method is plain Python integer arithmetic on ints already in
scope at its call site. No tensor reads, no ``.item()``, no device syncs, no new
allocations in the per-iteration path (the windowing appends one small tuple per
``window_steps`` optimizer steps and is bounded -- see ``_close_window``). This
keeps the standing repo invariant that observability must not measurably slow
training.

Honesty about partial coverage
------------------------------
The tracker is **in-memory only and deliberately not checkpointed**. A resumed
run therefore describes only its post-resume segment, and says so:
``summary()["coverage"]["partial_coverage"] is True`` with a reason. This repo
has shipped three report blocks that silently described part of a run as if it
were the whole (#282's ``fade_start_iter``, #290's seed-fade resume gap, #346)
-- this one declares it instead.

Relationship to ``scripts/verify_realized_cadence.py``
------------------------------------------------------
That script reconstructs the same quantities post-hoc from ``train.log`` /
``keyframe_admission_log.jsonl``, and is the only option for the entire
pre-2026-08-14 corpus. It prefers this block when a run carries one. The two
numbers are **expected to differ slightly, in known ways**, and are deliberately
reported side by side rather than merged:

  * **Bootstrap frames are not counted here.** The tracker is constructed after
    ``--streaming_initial_frames`` have already been ingested, so its
    ``source_released`` starts from the post-bootstrap stream.
  * **Different populations.** The script's release count comes from admission
    -log rows (candidates that reached the gate); this clock counts scheduler
    releases, which additionally include frames the ``BurstBudget`` dropped and
    frames withheld as eval holdouts.

A gap of that shape is the instrumentation agreeing. A gap in the *ratio*
(realized steps per release) is not, and is worth investigating.

Issue #725 -- the manipulation-check verdict measured the wrong quantity
----------------------------------------------------------------------
``reconcile()``'s divergence verdict used to compare the governing requested
SPF against ``realized_steps_per_release()`` (``optimizer_steps /
source_released``), a whole-run average over the FULL recorded span --
including any post-ingest refinement tail. For a run that releases all its
frames early and then trains a long tail, that ratio collapses to
``total_budget / n_frames`` and carries no information about the realized
inter-release gap. A real quartet (``issue289cadv2_{spf5,spf10,spf20_ref,
spf30}``, 80k-step budget, 2520 releases each) demonstrated this is wrong in
BOTH directions: every arm reported the identical ``31.75`` (=
``80000/2520``), producing a false ``diverged`` on the two arms with a
faster-than-average requested cadence (spf5, spf10) and an *accidental* ``ok``
on the two with a slower one (spf20_ref, spf30) purely because ``31.75``
happened to land inside tolerance of 20/30 but not 5/10.

The fix: the verdict is now derived from ``gaps_in_optimizer_steps
["between_releases"]`` -- the mean number of optimizer steps between
*consecutive* releases, already accumulated by ``_accum_release_gap()`` for
every release and unaffected by an idle tail (a tail with no releases
contributes no gap samples at all). On the same quartet this mean is
4.9996 / 9.9996 / 19.9996 / 29.9996 -- matching the true, independently
verified manipulation to within jitter, and all four verdict ``ok``.

``realized_steps_per_release()`` / the ``realized_steps_per_release`` field
are KEPT, unchanged in meaning, for backward-compatible reading of reports
written before this fix and because some consumers (the
``streaming/cadence/realized_steps_per_release`` TB scalar,
``scripts/verify_realized_cadence.py``) still want the whole-run average as a
descriptive number. They are simply no longer used to derive the verdict --
``reconcile()``'s output carries an explicit
``realized_steps_per_release_caveat`` string saying so, so a reader of an old
OR new report cannot mistake the budget-wide ratio for the gap metric.

Issue #727 -- insertion and MCMC-mutation clocks read structurally zero
------------------------------------------------------------------------
Two of the eight clocks (``insertion_commit``/``inserted_gaussians`` and
``mcmc_mutation``) were found reading exactly zero on every arm of the same
quartet despite ~13k real depth insertions independently visible in each
run's own TensorBoard scalars. Root causes (both in ``train_streaming.py``,
not in this module):

  * ``insertion_commit``/``inserted_gaussians`` were only ever recorded on the
    BATCHED insertion path (``--streaming_insertion_batch_frames>1`` or
    ``--streaming_insertion_batch_steps>0``). Both flags default to their
    off/1 values, so a default streaming run takes the non-batch branch,
    where the real per-frame insertion commit was computed (``added =
    cands_or_n``) but never handed to ``record_insertion_commit()`` -- a
    plain missing call, not a designed gate. Fixed by adding the call to that
    branch.
  * ``mcmc_mutation`` was gated on ``map_backend.topology_version`` changing.
    #543 found STATICALLY that no MCMC growth/relocation path (
    ``add_energy_guided``, ``relocate_energy_guided``, and their legacy/
    baseline equivalents) ever calls ``map_backend.commit_mutation()`` /
    ``record_external_mutation()`` -- those code paths mutate ``gaussians``'
    tensors directly, entirely outside the map_backend transaction API that
    ``topology_version`` tracks. So the gate could never fire for ordinary
    MCMC mutation, regardless of how much real growth/relocation happened.
    Partially fixed: the gate now ALSO fires on a population-size change
    (catches every growth/add event, which always changes the count). It
    still cannot distinguish a pure relocation-only mutation (no population-
    size change) from a no-op iteration -- closing that gap means making MCMC
    relocation participate in map_backend's mutation-tracking system, a
    larger, separate change tracked under #326 and NOT attempted here.

A zero must not be silently indistinguishable from "not measured" (the same
lesson the seed-cohort and seed-fade report blocks already apply via an
explicit ``reason``/``indeterminate`` field rather than a bare falsy value).
``summary()`` therefore carries ``clock_coverage``, documenting per-clock
measurement completeness rather than pretending every clock in the block
covers the same ground.

Relationship to ``mapping_clocks``
----------------------------------
``MappingCoordinator.four_clocks`` (``report.json["geometry"]["mapping_clocks"]``)
measures **wall-clock Hz** -- source frames/s, admitted keyframes/s, map
commits/s, realtime factor. That is the right clock for "can this keep up with a
live sensor". It is the *wrong* clock for #289: a Hz rate moves when the GPU
moves and says nothing about how many gradient steps the map got per unit of new
data. This module measures **per-optimizer-step** cadence. The two are
complementary and neither subsumes the other; do not add a third.
"""

from __future__ import annotations

from typing import Optional

# Field order for the compact per-window records. Kept as a module constant so
# the tuple layout and the summary field names cannot drift apart.
_WINDOW_FIELDS = (
    "source_released",
    "source_ingested",
    "candidate_evaluated",
    "keyframe_admitted",
    "insertion_commit",
    "inserted_gaussians",
    "supervision_view",
    "mcmc_mutation",
)

# Default divergence tolerance for the requested-vs-realized reconciliation:
# realized steps-per-release must land within a factor of 2 of the governing
# requested value. Deliberately loose -- the failure modes this exists to catch
# are 3x (W3-38's 20 -> 50 clamp) to 300x (#278), not 10%, and a tight tolerance
# on a legitimately jittery quantity would train readers to ignore the warning.
DEFAULT_TOLERANCE_FACTOR = 2.0

# Minimum evidence before any divergence verdict is issued at all. Under it the
# verdict is "insufficient_evidence", never "ok" -- a run that released three
# frames has not demonstrated its cadence.
MIN_STEPS_FOR_VERDICT = 200
MIN_RELEASES_FOR_VERDICT = 10

# Issue #727: per-clock measurement-completeness notes, persisted verbatim
# into summary()["clock_coverage"] so a zero in one of these clocks is never
# silently indistinguishable from "not measured" (mirrors the seed-cohort /
# seed-fade report blocks' explicit reason-string convention rather than a
# bare falsy value). Clocks not listed here have no known coverage caveat.
MCMC_MUTATION_COVERAGE_NOTE = (
    "Counts an MCMC growth (add) event via a population-size change. Does "
    "NOT yet count a pure relocation-only mutation (no population-size "
    "change) as a mutation, because no current MCMC growth/relocation path "
    "advances map_backend.topology_version or calls "
    "record_external_mutation() (confirmed statically, issue #543) -- "
    "closing that gap is tracked separately under issue #326. A zero here "
    "can still mean 'relocation happened but wasn't counted'; it does NOT "
    "mean no MCMC mutation occurred. See issue #727."
)
CLOCK_COVERAGE = {
    "insertion_commit": {
        "complete": True,
        "note": (
            "Fixed in issue #727: previously recorded only on the batched "
            "insertion path (--streaming_insertion_batch_frames>1 or "
            "--streaming_insertion_batch_steps>0); the non-batch path a "
            "default run actually takes never called record_insertion_"
            "commit(). Both paths now record every real insertion commit."
        ),
    },
    "inserted_gaussians": {
        "complete": True,
        "note": "Same fix and caveat as insertion_commit above.",
    },
    "mcmc_mutation": {
        "complete": False,
        "note": MCMC_MUTATION_COVERAGE_NOTE,
    },
}


def _stats(n: int, total: int, lo: Optional[int], hi: Optional[int]) -> dict:
    """Summarize a gap accumulator without retaining the samples."""
    return {
        "n": int(n),
        "mean": (float(total) / n) if n > 0 else None,
        "min": None if lo is None else int(lo),
        "max": None if hi is None else int(hi),
    }


def _ratio(num: float, den: float) -> Optional[float]:
    """Divide, or None when the denominator carries no information."""
    if den is None or den <= 0:
        return None
    return float(num) / float(den)


class RealizedCadenceTracker:
    """Accumulates the seven authoritative clocks and derives realized rates.

    Args:
        first_iteration: The run's starting iteration (``first_iter``), so the
            report can state the span this block actually covers.
        window_steps: Initial fixed window length, in optimizer steps.
        max_windows: Hard cap on retained windows. On reaching it, adjacent
            windows are merged pairwise and ``window_steps`` doubles, so memory
            is bounded regardless of run length while the whole run stays
            covered at a coarser resolution.
        resumed: True when this run resumed from a checkpoint, which makes the
            block's coverage partial by construction (the tracker is not
            checkpointed on purpose -- see the module docstring).
    """

    def __init__(
        self,
        *,
        first_iteration: int = 0,
        window_steps: int = 500,
        max_windows: int = 256,
        resumed: bool = False,
    ) -> None:
        self.first_iteration = int(first_iteration)
        self.last_iteration = int(first_iteration)
        self.resumed = bool(resumed)
        # max_windows must stay even so the pairwise merge is exact.
        self.max_windows = max(2, int(max_windows) + (int(max_windows) % 2))
        self.window_steps = max(1, int(window_steps))

        # --- clocks (cumulative) -------------------------------------------
        self.optimizer_steps = 0
        self.source_released = 0
        self.source_ingested = 0
        self.candidate_evaluated = 0
        self.keyframe_admitted = 0
        self.insertion_commit = 0
        self.inserted_gaussians = 0
        self.supervision_view = 0
        self.mcmc_mutation = 0

        # --- gap accumulators (optimizer steps between events) --------------
        self._last_admit_step: Optional[int] = None
        self._admit_gap_n = 0
        self._admit_gap_sum = 0
        self._admit_gap_min: Optional[int] = None
        self._admit_gap_max: Optional[int] = None

        self._last_insert_step: Optional[int] = None
        self._insert_gap_n = 0
        self._insert_gap_sum = 0
        self._insert_gap_min: Optional[int] = None
        self._insert_gap_max: Optional[int] = None

        self._last_release_step: Optional[int] = None
        self._release_gap_n = 0
        self._release_gap_sum = 0
        self._release_gap_min: Optional[int] = None
        self._release_gap_max: Optional[int] = None

        # --- backlog / queue age -------------------------------------------
        self._backlog_n = 0
        self._backlog_sum = 0
        self._backlog_max: Optional[int] = None
        self._queue_age_n = 0
        self._queue_age_sum = 0.0
        self._queue_age_max: Optional[float] = None

        # --- windowing ------------------------------------------------------
        self._win = [0] * len(_WINDOW_FIELDS)
        self._win_steps = 0
        self._windows: list[tuple[int, ...]] = []

        # --- divergence warning latch --------------------------------------
        self._divergence_warned = False

    def resume_state_dict(self) -> dict:
        """Capture whole-run cadence without changing report semantics."""
        return {
            "owner_schema": "realized_cadence_v1",
            "first_iteration": self.first_iteration,
            "last_iteration": self.last_iteration,
            "resumed": self.resumed,
            "max_windows": self.max_windows,
            "window_steps": self.window_steps,
            "optimizer_steps": self.optimizer_steps,
            "source_released": self.source_released,
            "source_ingested": self.source_ingested,
            "candidate_evaluated": self.candidate_evaluated,
            "keyframe_admitted": self.keyframe_admitted,
            "insertion_commit": self.insertion_commit,
            "inserted_gaussians": self.inserted_gaussians,
            "supervision_view": self.supervision_view,
            "mcmc_mutation": self.mcmc_mutation,
            "last_admit_step": self._last_admit_step,
            "admit_gap_n": self._admit_gap_n,
            "admit_gap_sum": self._admit_gap_sum,
            "admit_gap_min": self._admit_gap_min,
            "admit_gap_max": self._admit_gap_max,
            "last_insert_step": self._last_insert_step,
            "insert_gap_n": self._insert_gap_n,
            "insert_gap_sum": self._insert_gap_sum,
            "insert_gap_min": self._insert_gap_min,
            "insert_gap_max": self._insert_gap_max,
            "last_release_step": self._last_release_step,
            "release_gap_n": self._release_gap_n,
            "release_gap_sum": self._release_gap_sum,
            "release_gap_min": self._release_gap_min,
            "release_gap_max": self._release_gap_max,
            "backlog_n": self._backlog_n,
            "backlog_sum": self._backlog_sum,
            "backlog_max": self._backlog_max,
            "queue_age_n": self._queue_age_n,
            "queue_age_sum": self._queue_age_sum,
            "queue_age_max": self._queue_age_max,
            "win": list(self._win),
            "win_steps": self._win_steps,
            "windows": [list(window) for window in self._windows],
            "divergence_warned": self._divergence_warned,
        }

    def load_resume_state_dict(self, payload: dict, resolver=None) -> None:
        """Validate into a temporary tracker, then atomically replace state."""
        if not isinstance(payload, dict) or payload.get("owner_schema") != "realized_cadence_v1":
            raise ValueError("unsupported realized cadence checkpoint state")
        expected = set(self.resume_state_dict())
        if set(payload) != expected:
            raise ValueError("realized cadence checkpoint keys mismatch")

        def _nonnegative_int(name: str) -> int:
            value = payload[name]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"realized cadence {name} must be a non-negative integer")
            return value

        first_iteration = _nonnegative_int("first_iteration")
        max_windows = _nonnegative_int("max_windows")
        window_steps = _nonnegative_int("window_steps")
        if max_windows < 2 or max_windows % 2 or window_steps < 1:
            raise ValueError("invalid realized cadence window configuration")
        temp = RealizedCadenceTracker(
            first_iteration=first_iteration,
            window_steps=window_steps,
            max_windows=max_windows,
            resumed=False,
        )
        int_fields = (
            "last_iteration", "optimizer_steps", "source_released",
            "source_ingested", "candidate_evaluated", "keyframe_admitted",
            "insertion_commit", "inserted_gaussians", "supervision_view",
            "mcmc_mutation", "admit_gap_n", "admit_gap_sum", "insert_gap_n",
            "insert_gap_sum", "release_gap_n", "release_gap_sum", "backlog_n",
            "backlog_sum", "queue_age_n", "win_steps",
        )
        values = {name: _nonnegative_int(name) for name in int_fields}
        optional_int_fields = (
            "last_admit_step", "admit_gap_min", "admit_gap_max",
            "last_insert_step", "insert_gap_min", "insert_gap_max",
            "last_release_step", "release_gap_min", "release_gap_max", "backlog_max",
        )
        for name in optional_int_fields:
            value = payload[name]
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise ValueError(f"invalid realized cadence {name}")
            values[name] = value
        for name in ("queue_age_sum", "queue_age_max"):
            value = payload[name]
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0
            ):
                raise ValueError(f"invalid realized cadence {name}")
            values[name] = None if value is None else float(value)
        for name in ("resumed", "divergence_warned"):
            if not isinstance(payload[name], bool):
                raise ValueError(f"invalid realized cadence {name}")
        win = payload["win"]
        windows = payload["windows"]
        if (
            not isinstance(win, list) or len(win) != len(_WINDOW_FIELDS)
            or any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in win)
            or not isinstance(windows, list)
            or any(
                not isinstance(window, list) or len(window) != len(_WINDOW_FIELDS)
                or any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in window)
                for window in windows
            )
        ):
            raise ValueError("invalid realized cadence windows")

        temp.last_iteration = values["last_iteration"]
        temp.resumed = payload["resumed"]
        for public in (
            "optimizer_steps", "source_released", "source_ingested",
            "candidate_evaluated", "keyframe_admitted", "insertion_commit",
            "inserted_gaussians", "supervision_view", "mcmc_mutation",
        ):
            setattr(temp, public, values[public])
        for saved, attr in (
            ("last_admit_step", "_last_admit_step"), ("admit_gap_n", "_admit_gap_n"),
            ("admit_gap_sum", "_admit_gap_sum"), ("admit_gap_min", "_admit_gap_min"),
            ("admit_gap_max", "_admit_gap_max"), ("last_insert_step", "_last_insert_step"),
            ("insert_gap_n", "_insert_gap_n"), ("insert_gap_sum", "_insert_gap_sum"),
            ("insert_gap_min", "_insert_gap_min"), ("insert_gap_max", "_insert_gap_max"),
            ("last_release_step", "_last_release_step"), ("release_gap_n", "_release_gap_n"),
            ("release_gap_sum", "_release_gap_sum"), ("release_gap_min", "_release_gap_min"),
            ("release_gap_max", "_release_gap_max"), ("backlog_n", "_backlog_n"),
            ("backlog_sum", "_backlog_sum"), ("backlog_max", "_backlog_max"),
            ("queue_age_n", "_queue_age_n"), ("queue_age_sum", "_queue_age_sum"),
            ("queue_age_max", "_queue_age_max"), ("win_steps", "_win_steps"),
        ):
            setattr(temp, attr, values[saved])
        temp._win = list(win)
        temp._windows = [tuple(window) for window in windows]
        temp._divergence_warned = payload["divergence_warned"]
        self.__dict__.clear()
        self.__dict__.update(temp.__dict__)

    # ------------------------------------------------------------------ #
    # Recording. Every method here is O(1) integer work.                  #
    # ------------------------------------------------------------------ #

    def record_optimizer_step(self, iteration: Optional[int] = None) -> None:
        """One real ``optimizer.step()``. The denominator clock."""
        self.optimizer_steps += 1
        if iteration is not None:
            self.last_iteration = int(iteration)
        self._win_steps += 1
        if self._win_steps >= self.window_steps:
            self._close_window()

    def record_source_released(self) -> None:
        """``FrameScheduler`` let one source frame through the release gate."""
        self.source_released += 1
        self._win[0] += 1
        self._accum_release_gap()

    def record_source_ingested(self) -> None:
        """A released frame that actually produced a camera."""
        self.source_ingested += 1
        self._win[1] += 1

    def record_candidate_evaluated(self) -> None:
        """An admission decision function was invoked for a candidate."""
        self.candidate_evaluated += 1
        self._win[2] += 1

    def record_keyframe_admitted(self) -> None:
        """A frame was admitted as a training keyframe."""
        self.keyframe_admitted += 1
        self._win[3] += 1
        self._accum_admit_gap()

    def record_insertion_commit(self, n_inserted: int = 0) -> None:
        """A depth-insertion flush committed ``n_inserted`` Gaussians."""
        self.insertion_commit += 1
        self.inserted_gaussians += max(0, int(n_inserted))
        self._win[4] += 1
        self._win[5] += max(0, int(n_inserted))
        self._accum_insert_gap()

    def record_supervision_view(self, n: int = 1) -> None:
        """``n`` rendered/supervised views."""
        n = max(0, int(n))
        self.supervision_view += n
        self._win[6] += n

    def record_mcmc_mutation(self) -> None:
        """One MCMC growth/relocation mutation event."""
        self.mcmc_mutation += 1
        self._win[7] += 1

    def record_backlog(self, depth: Optional[int]) -> None:
        """Source queue depth, in frames. ``None`` (no backlog concept for this
        source -- e.g. a fully materialized offline dataset) is ignored, so the
        sample count distinguishes "never had a backlog" from "never asked"."""
        if depth is None:
            return
        depth = int(depth)
        self._backlog_n += 1
        self._backlog_sum += depth
        if self._backlog_max is None or depth > self._backlog_max:
            self._backlog_max = depth

    def record_queue_age(self, age_ms: Optional[float]) -> None:
        """Age of the oldest queued source frame, in milliseconds."""
        if age_ms is None:
            return
        age = float(age_ms)
        self._queue_age_n += 1
        self._queue_age_sum += age
        if self._queue_age_max is None or age > self._queue_age_max:
            self._queue_age_max = age

    # ------------------------------------------------------------------ #
    # Internals                                                           #
    # ------------------------------------------------------------------ #

    def _accum_admit_gap(self) -> None:
        prev = self._last_admit_step
        cur = self.optimizer_steps
        if prev is not None:
            gap = cur - prev
            self._admit_gap_n += 1
            self._admit_gap_sum += gap
            if self._admit_gap_min is None or gap < self._admit_gap_min:
                self._admit_gap_min = gap
            if self._admit_gap_max is None or gap > self._admit_gap_max:
                self._admit_gap_max = gap
        self._last_admit_step = cur

    def _accum_insert_gap(self) -> None:
        prev = self._last_insert_step
        cur = self.optimizer_steps
        if prev is not None:
            gap = cur - prev
            self._insert_gap_n += 1
            self._insert_gap_sum += gap
            if self._insert_gap_min is None or gap < self._insert_gap_min:
                self._insert_gap_min = gap
            if self._insert_gap_max is None or gap > self._insert_gap_max:
                self._insert_gap_max = gap
        self._last_insert_step = cur

    def _accum_release_gap(self) -> None:
        prev = self._last_release_step
        cur = self.optimizer_steps
        if prev is not None:
            gap = cur - prev
            self._release_gap_n += 1
            self._release_gap_sum += gap
            if self._release_gap_min is None or gap < self._release_gap_min:
                self._release_gap_min = gap
            if self._release_gap_max is None or gap > self._release_gap_max:
                self._release_gap_max = gap
        self._last_release_step = cur

    def _close_window(self) -> None:
        self._windows.append(tuple(self._win))
        self._win = [0] * len(_WINDOW_FIELDS)
        self._win_steps = 0
        if len(self._windows) >= self.max_windows:
            merged = [
                tuple(a + b for a, b in zip(self._windows[i], self._windows[i + 1]))
                for i in range(0, len(self._windows), 2)
            ]
            self._windows = merged
            self.window_steps *= 2

    # ------------------------------------------------------------------ #
    # Reporting                                                           #
    # ------------------------------------------------------------------ #

    def window_series(self) -> list[dict]:
        """Closed windows as dicts, oldest first. The still-open partial window
        is deliberately excluded: a partial window's counts are not comparable
        to the closed ones and would read as a spurious final dip."""
        out = []
        for i, w in enumerate(self._windows):
            rec = {"window_index": i,
                   "optimizer_step_start": i * self.window_steps}
            rec.update({name: int(v) for name, v in zip(_WINDOW_FIELDS, w)})
            out.append(rec)
        return out

    def realized_steps_per_release(self) -> Optional[float]:
        """Optimizer steps per released source frame -- the realized analogue of
        ``--streaming_steps_per_frame`` / ``effective_spf``."""
        return _ratio(self.optimizer_steps, self.source_released)

    def summary(self, *, include_windows: bool = True) -> dict:
        """A JSON-serializable realized-cadence summary. Pure derivation."""
        steps = self.optimizer_steps
        admitted = self.keyframe_admitted
        out: dict = {
            # Bumped 1 -> 2 (issues #725/#727): the reconciliation verdict is
            # now gap-derived, not budget-ratio-derived, and this block
            # additionally carries "clock_coverage". Both are purely additive
            # -- no existing field changed meaning or was removed -- so a
            # reader that only knows schema 1 still parses everything it
            # previously relied on; schema_version exists precisely so a
            # careful reader can tell "budget-ratio verdict" (schema 1)
            # apart from "gap-derived verdict" (schema >= 2) reports without
            # guessing from the run's date.
            "schema_version": 2,
            "clock_coverage": CLOCK_COVERAGE,
            "clocks": {
                "optimizer_step": steps,
                "source_released": self.source_released,
                "source_ingested": self.source_ingested,
                "source_released_not_ingested": max(
                    0, self.source_released - self.source_ingested),
                "candidate_evaluated": self.candidate_evaluated,
                "keyframe_admitted": admitted,
                "insertion_commit": self.insertion_commit,
                "inserted_gaussians": self.inserted_gaussians,
                "supervision_view": self.supervision_view,
                "mcmc_mutation": self.mcmc_mutation,
            },
            "per_optimizer_step": {
                "source_releases": _ratio(self.source_released, steps),
                "admissions": _ratio(admitted, steps),
                "inserted_gaussians": _ratio(self.inserted_gaussians, steps),
                "supervision_views": _ratio(self.supervision_view, steps),
                "mcmc_mutations": _ratio(self.mcmc_mutation, steps),
            },
            "per_admitted_frame": {
                "inserted_gaussians": _ratio(self.inserted_gaussians, admitted),
                "supervision_views": _ratio(self.supervision_view, admitted),
                "optimizer_steps": _ratio(steps, admitted),
            },
            "realized_steps_per_release": self.realized_steps_per_release(),
            "realized_steps_per_admission": _ratio(steps, admitted),
            "gaps_in_optimizer_steps": {
                "between_releases": _stats(
                    self._release_gap_n, self._release_gap_sum,
                    self._release_gap_min, self._release_gap_max),
                "between_admissions": _stats(
                    self._admit_gap_n, self._admit_gap_sum,
                    self._admit_gap_min, self._admit_gap_max),
                "between_insertion_commits": _stats(
                    self._insert_gap_n, self._insert_gap_sum,
                    self._insert_gap_min, self._insert_gap_max),
            },
            "backlog": {
                "samples": self._backlog_n,
                "mean_frames": _ratio(self._backlog_sum, self._backlog_n),
                "max_frames": self._backlog_max,
                "queue_age_samples": self._queue_age_n,
                "mean_queue_age_ms": (
                    self._queue_age_sum / self._queue_age_n
                    if self._queue_age_n > 0 else None),
                "max_queue_age_ms": self._queue_age_max,
            },
            "coverage": {
                "first_iteration": self.first_iteration,
                "last_iteration": self.last_iteration,
                "partial_coverage": bool(self.resumed),
                "partial_coverage_reason": (
                    "run resumed from a checkpoint; the realized-cadence tracker "
                    "is in-memory only, so this block describes the post-resume "
                    "segment only"
                ) if self.resumed else None,
            },
            "notes": (
                "Denominator is optimizer_step, NOT iteration -- they diverge "
                "under gradient accumulation and the placement_only/colors_only "
                "training modes. Wall-clock Hz rates live in "
                "report.json['geometry']['mapping_clocks'] (four_clocks) and are "
                "a different, complementary clock family."
            ),
        }
        if include_windows:
            out["windows"] = {
                "window_optimizer_steps": self.window_steps,
                "n_windows": len(self._windows),
                "open_window_optimizer_steps": self._win_steps,
                "series": self.window_series(),
            }
        return out

    # ------------------------------------------------------------------ #
    # Requested-vs-realized reconciliation (issue #289 task 3)            #
    # ------------------------------------------------------------------ #

    def reconcile(
        self,
        *,
        requested_steps_per_frame: Optional[float],
        frame_scheduler_spf: Optional[float],
        ingestion_mode: Optional[str],
        mapping_scheduler_effective_spf: Optional[float] = None,
        frame_admission: Optional[str] = None,
        iterations_elapsed: Optional[int] = None,
        tolerance_factor: float = DEFAULT_TOLERANCE_FACTOR,
    ) -> dict:
        """Compare the run's requested release cadence against the realized one.

        ``governing_requested`` is the value the release gate would actually have
        obeyed, which is **not** always what the user typed:

          * ``iter_based`` -- ``FrameScheduler`` is driven by
            ``MappingScheduler.effective_spf`` when the scene resolves in the
            ``eval/scenes.json`` registry (W3-38), so the value the scheduler
            was actually constructed with governs and the CLI value is only a
            request. A gap between the two is itself reported
            (``requested_overridden_by_scheduler_spf``).

        ``frame_scheduler_spf`` is deliberately the value ``FrameScheduler`` was
        CONSTRUCTED with, which is **not** always
        ``MappingScheduler.effective_spf`` -- under ``--online_mode`` the latter
        is never handed to the scheduler at all. Both are recorded, as separate
        fields, precisely because they coincide by accident on many scenes
        (CLAUDE.md: "the two '150's in a run's startup log are different
        variables that coincide by accident"), and a reader diffing the
        persisted number against ``train.log``'s ``[MappingScheduler]
        effective_spf=N`` line must not be silently misled.
          * ``dataset_fps`` / ``wallclock_strict`` -- ``should_release()`` never
            reads ``steps_per_frame`` at all; it paces against a (simulated or
            real) clock. Any comparison against a requested SPF is therefore
            category-confused, and the verdict says so explicitly rather than
            reporting a huge, meaningless divergence factor. This is exactly
            #278's situation.

        Returns a verdict dict; never raises.
        """
        # Issue #725: the verdict is derived from the REALIZED INTER-RELEASE
        # GAP (mean optimizer steps between consecutive releases), not from
        # realized_steps_per_release()'s whole-run budget ratio. The budget
        # ratio collapses to total_budget/n_frames for any run with a
        # post-ingest refinement tail and carries no information about actual
        # pacing -- see the module docstring's "Issue #725" section for the
        # real quartet that demonstrated this (identical 31.75 ratio on four
        # arms with true realized gaps of 5/10/20/30, producing false
        # `diverged` on two of them and an accidental `ok` on the other two).
        # gap_stats is exactly gaps_in_optimizer_steps["between_releases"] in
        # summary() -- same accumulator, not a second computation that could
        # drift from it.
        gap_stats = _stats(
            self._release_gap_n, self._release_gap_sum,
            self._release_gap_min, self._release_gap_max)
        realized_gap_mean = gap_stats["mean"]
        realized = self.realized_steps_per_release()  # kept: see caveat below
        mode = (ingestion_mode or "").strip() or None
        iteration_paced = mode in (None, "iter_based")

        governing = None
        governing_source = None
        if frame_scheduler_spf is not None and float(frame_scheduler_spf) > 0:
            governing = float(frame_scheduler_spf)
            governing_source = "frame_scheduler_spf"
        elif requested_steps_per_frame is not None and float(requested_steps_per_frame) > 0:
            governing = float(requested_steps_per_frame)
            governing_source = "streaming_steps_per_frame"

        requested_overridden = bool(
            requested_steps_per_frame is not None
            and frame_scheduler_spf is not None
            and float(requested_steps_per_frame) > 0
            and float(frame_scheduler_spf) > 0
            and abs(float(frame_scheduler_spf) - float(requested_steps_per_frame)) > 1e-9
        )

        verdict: dict
        divergence = None
        if not iteration_paced:
            verdict = "not_iteration_paced"
        elif realized_gap_mean is None or governing is None:
            verdict = "insufficient_evidence"
        elif (self.optimizer_steps < MIN_STEPS_FOR_VERDICT
              or self.source_released < MIN_RELEASES_FOR_VERDICT):
            verdict = "insufficient_evidence"
        else:
            divergence = max(realized_gap_mean / governing, governing / realized_gap_mean)
            verdict = "ok" if divergence <= max(1.0, float(tolerance_factor)) else "diverged"

        out = {
            "verdict": verdict,
            "ingestion_mode": mode,
            "iteration_paced": iteration_paced,
            "frame_admission": frame_admission,
            "requested_steps_per_frame": (
                None if requested_steps_per_frame is None else float(requested_steps_per_frame)),
            # The value FrameScheduler was CONSTRUCTED with -- see the docstring.
            "frame_scheduler_spf": (
                None if frame_scheduler_spf is None else float(frame_scheduler_spf)),
            # MappingScheduler's own effective_spf, recorded as a SEPARATE field
            # because it is not necessarily what the release gate obeyed (it is
            # not handed to FrameScheduler at all under --online_mode). This is
            # the field that lines up with train.log's [MappingScheduler] line.
            "mapping_scheduler_effective_spf": (
                None if mapping_scheduler_effective_spf is None
                else float(mapping_scheduler_effective_spf)),
            "governing_requested_steps_per_frame": governing,
            "governing_source": governing_source,
            "requested_overridden_by_scheduler_spf": requested_overridden,
            # Issue #725: THIS is the field the verdict/divergence_factor are
            # now derived from -- mean optimizer steps between CONSECUTIVE
            # releases, immune to dilution by a post-ingest refinement tail.
            # Same stats dict as summary()["gaps_in_optimizer_steps"]
            # ["between_releases"], surfaced here under an explicit name next
            # to the field it replaces for verdict purposes.
            "realized_release_gap_steps": gap_stats,
            # DEPRECATED FOR VERDICT PURPOSES (issue #725) -- kept, unchanged
            # in meaning, for backward-compatible reading of reports written
            # before this fix and for consumers that want the descriptive
            # whole-run average. optimizer_steps / source_released over the
            # ENTIRE recorded span (including any idle refinement tail) --
            # collapses to total_budget/n_frames and does NOT measure the
            # realized inter-release gap. Do not use this field to judge
            # whether a cadence manipulation landed; use
            # realized_release_gap_steps above.
            "realized_steps_per_release": realized,
            "realized_steps_per_release_caveat": (
                "Budget-wide average (optimizer_steps / source_released over "
                "the whole recorded span), NOT the realized inter-release "
                "gap -- superseded for verdict purposes by "
                "realized_release_gap_steps (issue #725). Kept only for "
                "backward-compatible reading of pre-fix reports and as a "
                "descriptive whole-run number."
            ),
            "realized_steps_per_admission": _ratio(self.optimizer_steps, self.keyframe_admitted),
            "divergence_factor": divergence,
            "tolerance_factor": float(tolerance_factor),
            "optimizer_steps": self.optimizer_steps,
            "iterations_elapsed": (
                None if iterations_elapsed is None else int(iterations_elapsed)),
            "source_released": self.source_released,
            "keyframes_admitted": self.keyframe_admitted,
            "message": None,
        }
        out["message"] = _verdict_message(out)
        return out

    def maybe_warn(self, **reconcile_kwargs) -> Optional[str]:
        """One-shot in-run divergence warning.

        Returns the warning string the first time the reconciliation lands on a
        non-``ok``, non-``insufficient_evidence`` verdict, then ``None`` forever
        after. Intended to be called on the existing TensorBoard logging cadence
        (never per-iteration) so it costs nothing measurable.
        """
        if self._divergence_warned:
            return None
        v = self.reconcile(**reconcile_kwargs)
        if v["verdict"] not in ("diverged", "not_iteration_paced"):
            return None
        self._divergence_warned = True
        return (
            "\n[realized-cadence] WARNING (issue #289): requested and realized "
            "release cadence disagree.\n  " + (v["message"] or "") +
            "\n  Do not cite nominal steps_per_frame for this run; read "
            "report.json['realized_cadence'] instead.\n"
        )


def _verdict_message(v: dict) -> str:
    """Human-readable one-liner for a reconciliation verdict.

    Issue #725: quotes the realized inter-release GAP mean
    (``realized_release_gap_steps["mean"]``), not the deprecated whole-run
    budget ratio (``realized_steps_per_release``) -- the message must agree
    with what the verdict itself was actually computed from.
    """
    gap = (v.get("realized_release_gap_steps") or {}).get("mean")
    gap_s = "unknown" if gap is None else f"{gap:.2f}"
    gov = v.get("governing_requested_steps_per_frame")
    gov_s = "unknown" if gov is None else f"{gov:g}"
    verdict = v.get("verdict")
    if verdict == "not_iteration_paced":
        return (
            f"ingestion_mode={v.get('ingestion_mode')} is NOT iteration-paced: "
            "FrameScheduler.should_release() never reads steps_per_frame in this "
            f"mode, so the requested {gov_s} steps/frame was not honored by "
            f"construction. Realized inter-release gap: {gap_s} optimizer steps. "
            "This is issue #278's mechanism (see #302)."
        )
    if verdict == "insufficient_evidence":
        return (
            "not enough evidence for a cadence verdict "
            f"(optimizer_steps={v.get('optimizer_steps')}, "
            f"source_released={v.get('source_released')}); "
            f"realized gap mean={gap_s}, requested={gov_s}."
        )
    if verdict == "diverged":
        div = v.get("divergence_factor")
        div_s = "unknown" if div is None else f"{div:.1f}x"
        extra = ""
        if v.get("requested_overridden_by_scheduler_spf"):
            extra = (
                " Note --streaming_steps_per_frame="
                f"{v.get('requested_steps_per_frame'):g} was already overridden: "
                f"FrameScheduler was constructed with "
                f"{v.get('frame_scheduler_spf'):g} (W3-38's min_steps_per_frame "
                "clamp)."
            )
        return (
            f"realized inter-release gap {gap_s} optimizer steps vs a "
            f"governing request of {gov_s} ({v.get('governing_source')}) -- "
            f"{div_s} divergence, beyond the {v.get('tolerance_factor')}x "
            f"tolerance.{extra}"
        )
    return (
        f"realized inter-release gap {gap_s} optimizer steps matches the "
        f"governing request of {gov_s} within the "
        f"{v.get('tolerance_factor')}x tolerance."
    )


def realized_cadence_report_block(
    tracker: Optional["RealizedCadenceTracker"],
    *,
    requested_steps_per_frame: Optional[float] = None,
    frame_scheduler_spf: Optional[float] = None,
    ingestion_mode: Optional[str] = None,
    mapping_scheduler_effective_spf: Optional[float] = None,
    frame_admission: Optional[str] = None,
    iterations_elapsed: Optional[int] = None,
    tolerance_factor: float = DEFAULT_TOLERANCE_FACTOR,
    include_windows: bool = True,
) -> Optional[dict]:
    """Build the ``report.json["realized_cadence"]`` block, or None.

    ``None`` when there is no tracker at all (the offline ``train.py`` path has
    no streaming clocks), so omitting it reproduces the pre-#289 report exactly.
    """
    if tracker is None:
        return None
    block = tracker.summary(include_windows=include_windows)
    block["reconciliation"] = tracker.reconcile(
        requested_steps_per_frame=requested_steps_per_frame,
        frame_scheduler_spf=frame_scheduler_spf,
        mapping_scheduler_effective_spf=mapping_scheduler_effective_spf,
        ingestion_mode=ingestion_mode,
        frame_admission=frame_admission,
        iterations_elapsed=iterations_elapsed,
        tolerance_factor=tolerance_factor,
    )
    return block
