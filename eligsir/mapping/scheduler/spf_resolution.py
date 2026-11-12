"""One authoritative record of how ``steps_per_frame`` (SPF) was resolved.

Issue #596. A real 2026-08-14 run (``issue290xsv2_sb20_obs``) emitted these
four startup lines, in this order::

    [streaming] full-coverage arithmetic (informational only): steps_per_frame 150
        would need to shrink to 2 for 7040 frames x 2 steps = 14080 iters <= 15000.
        Not enforced -- the rolling idle-based termination rule (see main loop)
        governs real coverage instead of this steps_per_frame arithmetic.
    [MappingScheduler] Full-coverage enforcement active: effective_spf=2 ...
    [streaming] ingestion_mode=iter_based (requested=auto, reason=offline_default)
    [streaming] Starting: ... steps_per_frame=2 ...

The first line says the computed ``2`` is *not enforced*; the run then trained
with exactly that ``2`` governing frame release. The claim was structurally
incapable of being correct: it is printed ~495 lines before
``train_streaming.py`` computes ``_use_enforced_spf`` / ``_frame_scheduler_spf``,
i.e. before the fact it asserts exists. It was true when written (2026-07-04,
when SPF really had been demoted to a statistic) and was falsified by the
2026-07-26 re-enablement for non-``--online_mode`` runs, without the message
being revisited.

Realized optimizer budget per released frame is a first-class experimental
variable for every cadence / seed-fade experiment in this repo (#289, #290,
#346), so ambiguity here is not cosmetic.

Three distinct states
---------------------
Whether SPF *reaches* ``FrameScheduler`` and whether ``FrameScheduler``
*reads* it are two independent conditions:

===========================  ==============================  ==============
run shape                    value handed to FrameScheduler  governs release
===========================  ==============================  ==============
``--online_mode``            raw request (bypasses           no -- online_mode
                             ``effective_spf``)              defaults to
                                                             ``dataset_fps``,
                                                             whose
                                                             ``should_release``
                                                             never reads SPF
non-online + ``iter_based``  ``MappingScheduler                 YES
                             .effective_spf``
non-online + explicit        ``MappingScheduler               no -- clock-paced
``dataset_fps`` /            .effective_spf``                 release gate
``wallclock_strict``
===========================  ==============================  ==============

Vocabulary
----------
``requested_spf``
    What the run asked for: ``--streaming_steps_per_frame`` (argparse default
    ``150``), possibly overridden by a ``--streaming_camera_profile``.
``coverage_required_spf``
    What full-coverage arithmetic computed:
    ``min(requested, floor(iterations * front_load_ratio / n_frames))``,
    clamped to >= 1. ``None`` when no coverage arithmetic ran (registry miss,
    or ``--no-streaming_enforce_full_coverage``).
``resolved_spf``
    The value ``FrameScheduler`` was actually constructed with -- recorded
    verbatim from ``train_streaming.py``'s ``_frame_scheduler_spf``, never
    recomputed here.
``resolved_by``
    Which mechanism produced ``resolved_spf`` (see :data:`RESOLVED_BY`).
``realized`` SPF
    NOT measured here. ``utils/realized_cadence.py`` already emits it
    (``report.json["realized_cadence"]["realized_steps_per_release"]``, plus
    ``requested_steps_per_frame`` / ``frame_scheduler_spf`` /
    ``mapping_scheduler_effective_spf``). Both blocks are sourced from the
    same ``_frame_scheduler_spf`` variable so they cannot drift.

This module records; it never decides. Nothing in it feeds back into the
release gate, and no consumer may derive an SPF from it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional

from eligsir.mapping.scheduler.frame_scheduler import (
    INGESTION_MODE_DATASET_FPS,
    INGESTION_MODE_WALLCLOCK_STRICT,
)

# Why ``resolved_spf`` holds the value it does.
RESOLVED_BY_REQUEST = "request"
"""The requested SPF survived unchanged (no coverage clamp bound it)."""

RESOLVED_BY_COVERAGE = "full_coverage_scheduler"
"""Full-coverage arithmetic reduced the request so every frame fits the budget."""

RESOLVED_BY_MIN_SPF_FLOOR = "min_spf_floor"
"""``--streaming_min_steps_per_frame`` raised the coverage result back up.

Note this floor can raise SPF ABOVE an explicitly requested lower value --
the W3-38 clamp (``docs/architecture/2026-08-09-w3-38-admission-gap.md``).
Recording it is exactly the point; #596 deliberately does not change it.
"""

RESOLVED_BY_ONLINE_BYPASS = "request_online_mode_bypass"
"""``--online_mode``: ``effective_spf`` is not handed to ``FrameScheduler`` at all.

**Corrected 2026-08-22 (#1044).** This label used to be applied to EVERY
``--online_mode`` run, on the module docstring's three-state assumption that
``online_mode ⇒ dataset_fps ⇒ the release gate never reads SPF``. An explicit
``--streaming_ingestion_mode iter_based`` — which is the documented #302
mitigation, so it appears on real production recipes — breaks that assumption:
the gate *does* read SPF. A real fr3 run therefore printed
``resolved_by=request_online_mode_bypass`` and ``governs_release=True`` and
``IS ENFORCED`` in the same block, which cannot all be true. This label is now
used only when the value genuinely does not reach the release gate.
"""

RESOLVED_BY_ONLINE_PLANNING_BUDGET = "full_coverage_online_planning_budget"
"""``--online_mode`` + ``iter_based``: coverage ran against a finite planning budget.

``opt.iterations`` is the ``ONLINE_UNBOUNDED_ITERATIONS`` sentinel (2^31-1) for
an online run, so ``min(requested, floor(budget / n_frames))`` was structurally
incapable of binding — the clamp silently evaluated to ``requested`` on every
online run, and the startup line then announced "covers 100%. All frames
covered." against a budget nobody will ever execute. See
:func:`resolve_coverage_planning_budget`.
"""

RESOLVED_BY = (
    RESOLVED_BY_REQUEST,
    RESOLVED_BY_COVERAGE,
    RESOLVED_BY_MIN_SPF_FLOOR,
    RESOLVED_BY_ONLINE_BYPASS,
    RESOLVED_BY_ONLINE_PLANNING_BUDGET,
)

# How the coverage planning budget was resolved (#1044).
BUDGET_LOOP = "loop_iterations"
BUDGET_MAX_ITERATIONS = "streaming_max_iterations"
BUDGET_OPERATOR_ITERATIONS = "operator_iterations"
BUDGET_UNBOUNDED = "unbounded_no_finite_budget"


def resolve_coverage_planning_budget(
    *,
    loop_iterations: int,
    streaming_max_iterations: int,
    operator_iterations: int,
    online_mode: bool,
    unbounded_sentinel: int = 2_147_483_647,
) -> tuple[int | None, str]:
    """Resolve the iteration budget full-coverage arithmetic should plan against.

    Issue #1044. Coverage arithmetic divides an iteration budget by the frame
    count to decide how many optimizer steps each released frame may have. For
    a non-online run the loop bound *is* that budget. For an ``--online_mode``
    run ``opt.iterations`` has already been overwritten with the unbounded
    sentinel by ``apply_online_schedule_overrides``, so dividing by it yields a
    clamp that can never bind — the defect that let a real fr3 run keep
    ``spf=150`` and reach only 130 of 2525 source frames (5.1%) while its own
    startup line claimed full coverage.

    The loop bound and the *planning* budget are different quantities: an
    online run legitimately has no fixed loop bound (rolling-idle termination),
    but it still has a budget the operator can be held to, and planning ingest
    cadence against 2^31 is never meaningful.

    Precedence for an online run: the explicit hard ceiling
    (``--streaming_max_iterations``), else the operator's own ``--iterations``
    (which survives on ``args`` because the sentinel is written to the separate
    ``opt`` object), else ``None`` — genuinely unbounded, in which case the
    caller must NOT claim coverage.

    Returns:
        ``(budget, source)``. ``budget`` is ``None`` only for the genuinely
        unbounded case; ``source`` is one of the ``BUDGET_*`` constants.
    """
    if not online_mode:
        return int(loop_iterations), BUDGET_LOOP
    if int(streaming_max_iterations) > 0:
        return int(streaming_max_iterations), BUDGET_MAX_ITERATIONS
    op = int(operator_iterations)
    if 0 < op < int(unbounded_sentinel):
        return op, BUDGET_OPERATOR_ITERATIONS
    return None, BUDGET_UNBOUNDED

# What ends the run.
TERMINATION_ROLLING_IDLE = "rolling_idle"
TERMINATION_FIXED_BUDGET = "fixed_iteration_budget"


@dataclass(frozen=True)
class SpfResolution:
    """Immutable record of the SPF lifecycle for one run.

    Every field is *recorded*, not derived from policy: the constructing call
    site passes the values its own already-executed code produced.

    Attributes:
        requested_spf: ``--streaming_steps_per_frame`` as seen by
            ``MappingScheduler``'s constructor.
        coverage_required_spf: Coverage arithmetic's result before the
            ``min_steps_per_frame`` floor; ``None`` if it never ran.
        min_spf_floor: ``--streaming_min_steps_per_frame`` (0 = no floor).
        resolved_spf: The value ``FrameScheduler`` was constructed with.
        resolved_by: One of :data:`RESOLVED_BY`.
        governs_release: Whether the release gate actually reads
            ``resolved_spf`` (true only under ``iter_based`` ingestion).
        ingestion_mode: Resolved ``INGESTION_MODE_*``.
        termination_policy: :data:`TERMINATION_ROLLING_IDLE` or
            :data:`TERMINATION_FIXED_BUDGET`. Derived from ``use_enforced_spf``
            (``not online_mode``), because the training loop's own
            ``_use_rolling_idle_stop = bool(online_mode)`` is computed ~1570
            lines later and cannot be passed backwards. **These two must remain
            exact complements**; if either predicate ever stops being
            ``online_mode``, this field silently starts lying.
        coverage_budget: Iteration budget the coverage arithmetic divided.
        coverage_n_frames: Frame count it divided by.
        registry_hit: Whether ``eval/scenes.json`` resolved the scene. A miss
            forces coverage enforcement off, so the request wins by default.
        full_coverage_enforced: Whether ``MappingScheduler`` ran the arithmetic.
        n_frames_capped_diagnostic: ``MappingScheduler.n_frames_capped`` --
            DIAGNOSTIC ONLY, no frame-source striding or capping is applied.

    Deliberately ABSENT: a ``predicted_steps_for_all_frames`` field, even
    though #596's suggested shape names one. ``train_streaming.py`` already
    prints that prediction (``[streaming] Dataset: N frames. Optimization
    steps to cover all frames: M ...``) from ``_min_iters_all``, which adds a
    bootstrap term (``streaming_initial_frames * spf``) and counts frames with
    ``streaming_scene._frame_count`` rather than ``coverage_n_frames``. A
    second prediction here would differ from it (14080 vs 14090 on the run
    that motivated this issue, and by more under ``--streaming_max_frames``)
    -- i.e. it would reintroduce exactly the two-startup-lines-disagreeing
    defect #596 exists to remove. That line owns the number; this record owns
    SPF ownership.
    """

    requested_spf: int
    coverage_required_spf: Optional[int]
    min_spf_floor: int
    resolved_spf: int
    resolved_by: str
    governs_release: bool
    ingestion_mode: str
    termination_policy: str
    coverage_budget: int
    coverage_n_frames: int
    registry_hit: bool
    full_coverage_enforced: bool
    n_frames_capped_diagnostic: Optional[int]
    #: #1044. Which quantity ``coverage_budget`` came from (``BUDGET_*``).
    #: ``unbounded_no_finite_budget`` means no coverage claim may be made.
    coverage_budget_source: str = BUDGET_LOOP
    #: #1044. Fraction of ``coverage_n_frames`` the run can actually reach at
    #: ``resolved_spf`` within ``coverage_budget``. ``None`` when the budget is
    #: genuinely unbounded. This is the number the old startup line got wrong.
    projected_coverage_frac: Optional[float] = None

    def as_dict(self) -> dict:
        """JSON-serializable form for ``report.json["spf_resolution"]``."""
        return asdict(self)

    def format_log(self) -> str:
        """The single authoritative startup summary.

        The log text and :meth:`as_dict` are formatted from the same fields,
        so console output and run metadata cannot disagree by construction.
        """
        lines = [
            "[spf] steps-per-frame resolution (authoritative; all overrides applied):",
            f"  requested_spf={self.requested_spf}"
            f"  coverage_required_spf={_fmt(self.coverage_required_spf)}"
            f"  min_spf_floor={self.min_spf_floor}",
            f"  resolved_spf={self.resolved_spf}  resolved_by={self.resolved_by}",
            f"  governs_release={self.governs_release}"
            f"  ingestion_mode={self.ingestion_mode}"
            f"  termination_policy={self.termination_policy}",
            f"  coverage_budget={self.coverage_budget}"
            f"  coverage_n_frames={self.coverage_n_frames}"
            f"  registry_hit={self.registry_hit}"
            f"  full_coverage_enforced={self.full_coverage_enforced}"
            f"  budget_source={self.coverage_budget_source}",
        ]
        # #1044: an honest coverage projection. The old
        # "covers 100%. All frames covered." line divided by the 2^31 online
        # sentinel and was therefore true of no real run.
        if self.coverage_budget_source == BUDGET_UNBOUNDED:
            _need = self.resolved_spf * max(1, self.coverage_n_frames)
            lines.append(
                f"  -> COVERAGE NOT CLAIMED: no finite planning budget "
                f"(--online_mode with neither --streaming_max_iterations nor a "
                f"finite --iterations). At spf={self.resolved_spf} this scene needs "
                f"~{_need} iterations to release all {self.coverage_n_frames} frames; "
                f"pass --streaming_max_iterations (or a finite --iterations) so "
                f"ingest cadence can be planned against a budget you will run."
            )
        elif self.projected_coverage_frac is not None:
            _pct = 100.0 * self.projected_coverage_frac
            _need = self.resolved_spf * max(1, self.coverage_n_frames)
            if self.projected_coverage_frac >= 0.999:
                lines.append(
                    f"  -> projected coverage {_pct:.1f}% of "
                    f"{self.coverage_n_frames} frames "
                    f"(needs ~{_need} iters at spf={self.resolved_spf}, "
                    f"budget={self.coverage_budget})."
                )
            else:
                lines.append(
                    f"  -> WARNING (#1044): projected coverage only {_pct:.1f}% of "
                    f"{self.coverage_n_frames} frames. At spf={self.resolved_spf} full "
                    f"release needs ~{_need} iterations but the planning budget is "
                    f"{self.coverage_budget} ({self.coverage_budget_source}). The "
                    f"trajectory will be truncated -- lower "
                    f"--streaming_steps_per_frame / --streaming_min_steps_per_frame, "
                    f"or raise the budget."
                )
        if self.governs_release:
            lines.append(
                f"  -> resolved_spf={self.resolved_spf} IS ENFORCED: FrameScheduler's "
                "iter_based release gate reads it directly."
            )
        elif self.ingestion_mode in (
            INGESTION_MODE_DATASET_FPS,
            INGESTION_MODE_WALLCLOCK_STRICT,
        ):
            lines.append(
                f"  -> resolved_spf={self.resolved_spf} is NOT read by the release gate: "
                f"ingestion_mode={self.ingestion_mode} paces release by a clock "
                "(should_release never consults steps_per_frame). Cadence for this "
                "run is set by the clock, not by this number."
            )
        else:
            lines.append(
                f"  -> resolved_spf={self.resolved_spf} is NOT read by the release gate "
                f"under ingestion_mode={self.ingestion_mode}."
            )
        if self.resolved_by == RESOLVED_BY_MIN_SPF_FLOOR:
            lines.append(
                f"  -> min_spf_floor={self.min_spf_floor} RAISED the coverage result "
                f"({_fmt(self.coverage_required_spf)}) back up; see W3-38 "
                "(docs/architecture/2026-08-09-w3-38-admission-gap.md)."
            )
        if self.n_frames_capped_diagnostic is not None:
            lines.append(
                f"  -> diagnostic only: the naive spf x n_frames <= iterations equation "
                f"would cap admitted frames to {self.n_frames_capped_diagnostic}/"
                f"{self.coverage_n_frames}. No frame-source striding or capping is "
                "applied -- this number changes nothing about the run."
            )
        if not self.registry_hit:
            lines.append(
                "  -> scene missed the eval/scenes.json registry: full-coverage "
                "enforcement is forced off, so the requested value wins by default."
            )
        return "\n".join(lines)


def _fmt(value: Optional[int]) -> str:
    return "n/a" if value is None else str(value)


def build_spf_resolution(
    *,
    requested_spf: int,
    resolved_spf: int,
    scheduler_effective_spf: int,
    scheduler_resolved_by: str,
    coverage_required_spf: Optional[int],
    min_spf_floor: int,
    use_enforced_spf: bool,
    governs_release: bool,
    ingestion_mode: str,
    coverage_budget: int,
    coverage_n_frames: int,
    registry_hit: bool,
    full_coverage_enforced: bool,
    n_frames_capped_diagnostic: Optional[int],
    online_mode: bool = False,
    coverage_budget_source: str = BUDGET_LOOP,
    projected_coverage_frac: Optional[float] = None,
) -> SpfResolution:
    """Assemble an :class:`SpfResolution` from already-computed values.

    This function performs NO scheduling decision. ``resolved_spf`` must be
    passed verbatim as the value handed to ``FrameScheduler``; the only thing
    derived here is the human-readable *attribution* of that value.

    Args:
        requested_spf: ``--streaming_steps_per_frame``.
        resolved_spf: ``train_streaming.py``'s ``_frame_scheduler_spf``.
        scheduler_effective_spf: ``MappingScheduler.effective_spf``.
        scheduler_resolved_by: ``MappingScheduler.spf_resolved_by``.
        coverage_required_spf: ``MappingScheduler.coverage_required_spf``.
        min_spf_floor: ``--streaming_min_steps_per_frame``.
        use_enforced_spf: ``train_streaming.py``'s ``_use_enforced_spf``
            (False exactly when ``--online_mode`` bypasses the scheduler).
        governs_release: Whether the release gate reads SPF -- pass
            ``_ingestion_report["steps_per_frame_honored"]``, do not re-derive.
        ingestion_mode: Resolved ``INGESTION_MODE_*``.
        coverage_budget: ``opt.iterations``.
        coverage_n_frames: ``_coverage_n_frames``.
        registry_hit: ``_is_finite_replay``.
        full_coverage_enforced: Whether the scheduler ran coverage arithmetic.
        n_frames_capped_diagnostic: ``MappingScheduler.n_frames_capped``.

    Returns:
        The immutable record.
    """
    if not use_enforced_spf:
        resolved_by = RESOLVED_BY_ONLINE_BYPASS
    elif int(resolved_spf) != int(scheduler_effective_spf):
        # Should be unreachable: under use_enforced_spf the call site passes
        # max(1, effective_spf). Recorded honestly rather than mislabelled.
        resolved_by = "unattributed"
    else:
        resolved_by = scheduler_resolved_by

    return SpfResolution(
        requested_spf=int(requested_spf),
        coverage_required_spf=(
            None if coverage_required_spf is None else int(coverage_required_spf)
        ),
        min_spf_floor=int(min_spf_floor),
        resolved_spf=int(resolved_spf),
        resolved_by=resolved_by,
        governs_release=bool(governs_release),
        ingestion_mode=str(ingestion_mode),
        # #1044: derived from `online_mode` DIRECTLY, not from
        # `use_enforced_spf`. The two stopped being complements the moment an
        # online run could also have its SPF coverage-enforced (against a
        # finite planning budget) -- and this field's own docstring warned that
        # it "silently starts lying" if that ever happened.
        termination_policy=(
            TERMINATION_ROLLING_IDLE if online_mode
            else TERMINATION_FIXED_BUDGET
        ),
        coverage_budget_source=str(coverage_budget_source),
        projected_coverage_frac=(
            None if projected_coverage_frac is None
            else float(projected_coverage_frac)
        ),
        coverage_budget=int(coverage_budget),
        coverage_n_frames=int(coverage_n_frames),
        registry_hit=bool(registry_hit),
        full_coverage_enforced=bool(full_coverage_enforced),
        n_frames_capped_diagnostic=(
            None if n_frames_capped_diagnostic is None
            else int(n_frames_capped_diagnostic)
        ),
    )
