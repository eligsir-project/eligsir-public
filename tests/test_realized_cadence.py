"""CPU unit tests for the realized-cadence accounting (issue #289).

These cover the accounting and the requested-vs-realized reconciliation, plus the
wiring assertions that the seven clocks are actually recorded from
``train_streaming.py`` and that ``report.json["realized_cadence"]`` is optional /
additive in ``write_post_training_report``.

No GPU, no training, no torch import.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from eligsir.utils.realized_cadence import (
    CLOCK_COVERAGE,
    DEFAULT_TOLERANCE_FACTOR,
    MCMC_MUTATION_COVERAGE_NOTE,
    MIN_RELEASES_FOR_VERDICT,
    MIN_STEPS_FOR_VERDICT,
    RealizedCadenceTracker,
    realized_cadence_report_block,
)

REPO = Path(__file__).resolve().parents[1]
TRAIN_STREAMING = REPO / "train_streaming.py"


def _simulate(*, steps: int, spf: int, admit_every: int = 1,
              insert_every_admission: int = 1, inserted_per_commit: int = 100,
              views_per_step: int = 1) -> RealizedCadenceTracker:
    """Drive the tracker like an iteration-paced streaming loop would."""
    t = RealizedCadenceTracker(window_steps=100)
    n_admitted = 0
    for i in range(1, steps + 1):
        if spf > 0 and i % spf == 0:
            t.record_source_released()
            t.record_source_ingested()
            t.record_candidate_evaluated()
            n_admitted += 1
            if n_admitted % admit_every == 0:
                t.record_keyframe_admitted()
                if insert_every_admission and n_admitted % insert_every_admission == 0:
                    t.record_insertion_commit(inserted_per_commit)
        for _ in range(views_per_step):
            t.record_supervision_view()
        t.record_optimizer_step(i)
    return t


def _simulate_release_then_idle_tail(
        *, spf: int, n_releases: int, tail_steps: int) -> RealizedCadenceTracker:
    """Reproduce the issue #289cadv2 quartet's shape: releases paced exactly
    ``spf`` optimizer steps apart, then a long idle refinement tail with NO
    further releases. This is exactly what collapses the whole-run budget
    ratio (optimizer_steps / source_released) toward total_budget/n_frames
    while leaving the release-to-release GAP untouched at ``spf``."""
    t = RealizedCadenceTracker(window_steps=100)
    step = 0
    for _ in range(n_releases):
        for _ in range(spf):
            step += 1
            t.record_optimizer_step(step)
        t.record_source_released()
        t.record_source_ingested()
        t.record_keyframe_admitted()
    for _ in range(tail_steps):
        step += 1
        t.record_optimizer_step(step)
    return t


# ---------------------------------------------------------------- clocks ----

def test_clocks_count_independently():
    t = RealizedCadenceTracker()
    t.record_optimizer_step(1)
    t.record_source_released()
    t.record_source_ingested()
    t.record_candidate_evaluated()
    t.record_keyframe_admitted()
    t.record_insertion_commit(42)
    t.record_supervision_view()
    t.record_mcmc_mutation()
    c = t.summary()["clocks"]
    assert c == {
        "optimizer_step": 1,
        "source_released": 1,
        "source_ingested": 1,
        "source_released_not_ingested": 0,
        "candidate_evaluated": 1,
        "keyframe_admitted": 1,
        "insertion_commit": 1,
        "inserted_gaussians": 42,
        "supervision_view": 1,
        "mcmc_mutation": 1,
    }


def test_release_is_not_ingest():
    """A frame released and then dropped (BurstBudget) still counts as
    released. Conflating the two clocks is #278's F-vs-V confusion."""
    t = RealizedCadenceTracker()
    t.record_source_released()           # dropped by BurstBudget
    t.record_source_released()
    t.record_source_ingested()           # only this one produced a camera
    c = t.summary()["clocks"]
    assert c["source_released"] == 2
    assert c["source_ingested"] == 1
    assert c["source_released_not_ingested"] == 1


def test_optimizer_step_is_the_denominator_not_iteration():
    """Under gradient accumulation the supervision clock outruns the optimizer
    clock; the per-step rates must use the optimizer clock."""
    t = _simulate(steps=100, spf=10, views_per_step=4)
    s = t.summary()
    assert s["clocks"]["optimizer_step"] == 100
    assert s["clocks"]["supervision_view"] == 400
    assert s["per_optimizer_step"]["supervision_views"] == pytest.approx(4.0)


def test_realized_steps_per_release_matches_a_paced_loop():
    t = _simulate(steps=1000, spf=50)
    assert t.summary()["clocks"]["source_released"] == 20
    assert t.realized_steps_per_release() == pytest.approx(50.0)


def test_per_admitted_frame_rates():
    t = _simulate(steps=1000, spf=50, inserted_per_commit=250)
    s = t.summary()
    assert s["clocks"]["keyframe_admitted"] == 20
    assert s["per_admitted_frame"]["inserted_gaussians"] == pytest.approx(250.0)
    assert s["per_admitted_frame"]["optimizer_steps"] == pytest.approx(50.0)
    assert s["per_admitted_frame"]["supervision_views"] == pytest.approx(50.0)


def test_gaps_between_admissions_and_insertions():
    t = _simulate(steps=1000, spf=50)
    g = t.summary()["gaps_in_optimizer_steps"]
    # 20 events -> 19 measured gaps, each exactly spf.
    assert g["between_admissions"]["n"] == 19
    assert g["between_admissions"]["mean"] == pytest.approx(50.0)
    assert g["between_admissions"]["min"] == 50
    assert g["between_admissions"]["max"] == 50
    assert g["between_insertion_commits"]["n"] == 19
    assert g["between_releases"]["mean"] == pytest.approx(50.0)


def test_zero_denominators_are_none_not_crashes():
    t = RealizedCadenceTracker()
    s = t.summary()
    assert s["realized_steps_per_release"] is None
    assert s["per_optimizer_step"]["source_releases"] is None
    assert s["per_admitted_frame"]["inserted_gaussians"] is None
    assert s["gaps_in_optimizer_steps"]["between_admissions"]["mean"] is None


def test_backlog_and_queue_age_ignore_none():
    t = RealizedCadenceTracker()
    t.record_backlog(None)
    t.record_backlog(3)
    t.record_backlog(7)
    t.record_queue_age(None)
    t.record_queue_age(120.0)
    b = t.summary()["backlog"]
    assert b["samples"] == 2
    assert b["mean_frames"] == pytest.approx(5.0)
    assert b["max_frames"] == 7
    assert b["queue_age_samples"] == 1
    assert b["max_queue_age_ms"] == pytest.approx(120.0)


# --------------------------------------------------------------- windows ----

def test_windows_are_bounded_and_cover_the_whole_run():
    t = RealizedCadenceTracker(window_steps=10, max_windows=8)
    for i in range(1, 1001):
        if i % 10 == 0:
            t.record_source_released()
        t.record_optimizer_step(i)
    w = t.summary()["windows"]
    assert w["n_windows"] <= 8
    assert w["window_optimizer_steps"] > 10          # doubled on merge
    # Merging loses no event: the closed windows account for exactly the
    # optimizer steps they cover, and the still-open partial window accounts for
    # the rest (it is excluded from the series on purpose -- see
    # test_open_window_is_excluded).
    closed_steps = w["n_windows"] * w["window_optimizer_steps"]
    assert closed_steps + w["open_window_optimizer_steps"] == 1000
    assert sum(r["source_released"] for r in w["series"]) == closed_steps // 10


def test_open_window_is_excluded():
    t = RealizedCadenceTracker(window_steps=100)
    for i in range(1, 150):
        t.record_optimizer_step(i)
    w = t.summary()["windows"]
    assert w["n_windows"] == 1
    assert w["open_window_optimizer_steps"] == 49


# --------------------------------------------------------- reconciliation ----

def _recon(t, **kw):
    base = dict(requested_steps_per_frame=50.0, frame_scheduler_spf=50.0,
                ingestion_mode="iter_based")
    base.update(kw)
    return t.reconcile(**base)


def test_reconcile_ok_when_paced_as_requested():
    t = _simulate(steps=2000, spf=50)
    v = _recon(t)
    assert v["verdict"] == "ok"
    assert v["divergence_factor"] == pytest.approx(1.0)
    assert v["governing_source"] == "frame_scheduler_spf"


def test_reconcile_flags_w3_38_upward_clamp():
    """W3-38: an explicit --streaming_steps_per_frame 20 silently clamped to 50
    by min_steps_per_frame. The scheduler obeys 50, so the realized cadence
    matches effective_spf -- the finding is the override, not a divergence."""
    t = _simulate(steps=2000, spf=50)
    v = _recon(t, requested_steps_per_frame=20.0, frame_scheduler_spf=50.0)
    assert v["requested_overridden_by_scheduler_spf"] is True
    assert v["governing_requested_steps_per_frame"] == 50.0
    assert v["verdict"] == "ok"          # realized == what the gate was given


def test_reconcile_detects_divergence_beyond_tolerance():
    # Released far faster than the governing request: 2000 steps, 1000 releases.
    t = _simulate(steps=2000, spf=2)
    v = _recon(t)
    assert v["verdict"] == "diverged"
    assert v["divergence_factor"] == pytest.approx(25.0)
    assert "diverg" in v["message"]


def test_reconcile_dataset_fps_is_category_confused_not_diverged():
    """#278: under dataset_fps, should_release() never reads steps_per_frame.
    The verdict must say the mode is not iteration-paced rather than reporting a
    meaningless ~300x divergence factor against a number nothing consulted."""
    t = _simulate(steps=2000, spf=2)
    v = _recon(t, ingestion_mode="dataset_fps")
    assert v["verdict"] == "not_iteration_paced"
    assert v["iteration_paced"] is False
    assert v["divergence_factor"] is None
    assert "#278" in v["message"]


def test_reconcile_refuses_a_verdict_without_evidence():
    t = _simulate(steps=MIN_STEPS_FOR_VERDICT - 1, spf=1)
    assert _recon(t)["verdict"] == "insufficient_evidence"
    # Enough steps, too few releases.
    t2 = _simulate(steps=MIN_STEPS_FOR_VERDICT * 4, spf=MIN_STEPS_FOR_VERDICT)
    assert t2.source_released < MIN_RELEASES_FOR_VERDICT
    assert _recon(t2)["verdict"] == "insufficient_evidence"


def test_maybe_warn_is_one_shot():
    t = _simulate(steps=2000, spf=2)
    kw = dict(requested_steps_per_frame=50.0, frame_scheduler_spf=50.0,
              ingestion_mode="iter_based")
    first = t.maybe_warn(**kw)
    assert first is not None and "#289" in first
    assert t.maybe_warn(**kw) is None


def test_maybe_warn_silent_when_ok_or_unproven():
    ok = _simulate(steps=2000, spf=50)
    assert ok.maybe_warn(requested_steps_per_frame=50.0, frame_scheduler_spf=50.0,
                         ingestion_mode="iter_based") is None
    empty = RealizedCadenceTracker()
    assert empty.maybe_warn(requested_steps_per_frame=50.0, frame_scheduler_spf=50.0,
                            ingestion_mode="iter_based") is None


def test_default_tolerance_admits_a_factor_two_wobble():
    t = _simulate(steps=2000, spf=50)
    v = _recon(t, frame_scheduler_spf=90.0)   # 1.8x -- inside the 2x tolerance
    assert v["divergence_factor"] == pytest.approx(1.8)
    assert v["verdict"] == "ok"
    assert DEFAULT_TOLERANCE_FACTOR == 2.0


# ------------------------------------------------------- issue #725 fix ----

def test_budget_wide_ratio_collapses_but_gap_mean_does_not():
    """The exact failure shape from the issue #289cadv2 quartet: a real
    inter-release gap of 5 steps, diluted by a long idle refinement tail down
    to a whole-run ratio of ~31.75. The old verdict logic used the ratio and
    falsely diverged; the gap mean must still read ~5."""
    t = _simulate_release_then_idle_tail(spf=5, n_releases=2520, tail_steps=67400)
    s = t.summary()
    # Budget-wide ratio: collapses toward total_steps / n_releases, NOT 5.
    assert s["clocks"]["optimizer_step"] == 2520 * 5 + 67400
    assert t.realized_steps_per_release() > 30.0
    # The realized inter-release gap, unaffected by the idle tail.
    gap = s["gaps_in_optimizer_steps"]["between_releases"]
    assert gap["mean"] == pytest.approx(5.0, abs=0.01)


def test_reconcile_quartet_issue289cadv2_all_verdict_ok():
    """Direct reproduction of the four issue289cadv2_{spf5,spf10,spf20_ref,
    spf30} arms' real persisted shape (2520 releases, spf-paced, then an idle
    refinement tail out to an 80000-step budget) -- issue #725's acceptance
    criterion (a). All four must verdict 'ok' under the repaired metric; the
    pre-fix logic gave 'diverged' on spf5/spf10 and only an accidental 'ok'
    on spf20_ref/spf30 (see the module docstring's issue #725 section)."""
    for spf in (5, 10, 20, 30):
        n_releases = 2520
        tail = 80000 - n_releases * spf
        assert tail > 0
        t = _simulate_release_then_idle_tail(spf=spf, n_releases=n_releases, tail_steps=tail)
        v = t.reconcile(requested_steps_per_frame=float(spf), frame_scheduler_spf=float(spf),
                         ingestion_mode="iter_based")
        assert v["verdict"] == "ok", f"spf={spf}: {v['message']}"
        assert v["realized_release_gap_steps"]["mean"] == pytest.approx(float(spf), abs=0.01)


def test_synthetic_early_dump_verdicts_non_ok():
    """Issue #725 acceptance criterion (b): a run that dumps its whole
    trajectory early (all releases packed into the first slice of the budget,
    far faster than the governing request) must verdict non-'ok' under the
    repaired metric -- the exact failure mode a budget-wide ratio can hide by
    averaging a fast dump against a long, release-free tail."""
    # Governing request: 150 steps/release (a realistic streaming default).
    # Reality: 1000 releases packed 5 steps apart, then a long idle tail --
    # the whole trajectory dumped in the first 5% of a 100000-step budget.
    t = _simulate_release_then_idle_tail(spf=5, n_releases=1000, tail_steps=95000)
    v = t.reconcile(requested_steps_per_frame=150.0, frame_scheduler_spf=150.0,
                     ingestion_mode="iter_based")
    assert v["verdict"] == "diverged"
    assert v["realized_release_gap_steps"]["mean"] == pytest.approx(5.0, abs=0.01)
    # And the OLD (deprecated-for-verdict) ratio would have told a different,
    # equally wrong story in the other direction -- worth pinning so a future
    # change cannot silently start using it again without this test noticing.
    assert v["realized_steps_per_release"] == pytest.approx(100000 / 1000, abs=0.5)


def test_realized_steps_per_release_kept_but_labeled_deprecated_for_verdict():
    """Issue #725 acceptance criterion (c): the old field must stay readable
    (not repurposed, not removed) but must carry an explicit caveat so a
    reader cannot mistake it for the gap metric that now drives the verdict."""
    t = _simulate(steps=2000, spf=50)
    v = _recon(t)
    assert v["realized_steps_per_release"] == pytest.approx(t.realized_steps_per_release())
    assert "realized_release_gap_steps" in v["realized_steps_per_release_caveat"]
    assert "NOT the realized inter-release" in v["realized_steps_per_release_caveat"]
    assert v["realized_release_gap_steps"]["mean"] == pytest.approx(50.0)


def test_verdict_message_quotes_the_gap_not_the_old_ratio():
    t = _simulate_release_then_idle_tail(spf=5, n_releases=2520, tail_steps=67400)
    v = t.reconcile(requested_steps_per_frame=5.0, frame_scheduler_spf=5.0,
                     ingestion_mode="iter_based")
    assert v["verdict"] == "ok"
    assert "5.0" in v["message"] or "5.00" in v["message"]
    assert "31.7" not in v["message"]


# --------------------------------------------------------------- coverage ----

def test_resumed_run_declares_partial_coverage():
    t = RealizedCadenceTracker(first_iteration=5000, resumed=True)
    t.record_optimizer_step(5001)
    cov = t.summary()["coverage"]
    assert cov["partial_coverage"] is True
    assert "resumed" in cov["partial_coverage_reason"]
    assert cov["first_iteration"] == 5000
    assert cov["last_iteration"] == 5001


def test_fresh_run_declares_full_coverage():
    t = _simulate(steps=10, spf=5)
    cov = t.summary()["coverage"]
    assert cov["partial_coverage"] is False
    assert cov["partial_coverage_reason"] is None


# ------------------------------------------------------------ report block ----

def test_report_block_none_without_tracker():
    assert realized_cadence_report_block(None) is None


def test_report_block_is_json_serializable_and_carries_reconciliation():
    import json

    t = _simulate(steps=2000, spf=50)
    block = realized_cadence_report_block(
        t, requested_steps_per_frame=50.0, frame_scheduler_spf=50.0,
        ingestion_mode="iter_based", frame_admission="all",
        iterations_elapsed=2000)
    assert block["reconciliation"]["verdict"] == "ok"
    assert block["reconciliation"]["frame_admission"] == "all"
    # Bumped 1 -> 2 (issues #725/#727): gap-derived verdict + clock_coverage.
    assert block["schema_version"] == 2
    json.dumps(block)   # must not raise


def test_admission_all_reads_as_no_gate_not_broken_instrumentation():
    """Historical (ungated) `all`: no gate runs, so candidate_evaluated is 0 by
    design, and frame_admission is recorded so the zero is readable.

    Scoped by issue #602: this asserts the behavior of the HISTORICAL ungated
    `all` -- what #600 renames `force_all_stress`. It is NOT a claim that
    `candidate_evaluated == 0` for every run whose requested policy reads
    `all`; see the soft-all case below.
    """
    t = RealizedCadenceTracker()
    for i in range(1, 501):
        if i % 50 == 0:
            t.record_source_released()
            t.record_source_ingested()
            t.record_keyframe_admitted()
        t.record_optimizer_step(i)
    block = realized_cadence_report_block(
        t, requested_steps_per_frame=50.0, frame_scheduler_spf=50.0,
        ingestion_mode="iter_based", frame_admission="all")
    assert block["clocks"]["candidate_evaluated"] == 0
    assert block["clocks"]["keyframe_admitted"] == 10
    assert block["reconciliation"]["frame_admission"] == "all"


def test_soft_all_may_gate_and_diverge_from_source_ingested():
    """Issue #602: soft-all falsifies two invariants this module documented.

    Under #600's `soft_all`, every source frame is still observed but mapping
    admission is budgeted, so a real decision runs per frame. The module used
    to document -- and this file used to only ever exercise -- the
    unconditional forms "under `all` no gate runs" and "`keyframe_admitted`
    equals `source_ingested`". Both are false here, and neither the tracker
    nor the report block may treat that as broken instrumentation.
    """
    t = RealizedCadenceTracker()
    for i in range(1, 501):
        if i % 10 == 0:
            t.record_source_released()
            t.record_source_ingested()
            # A budgeted decision really runs for every observed frame...
            t.record_candidate_evaluated()
            # ...and declines some of them under pressure.
            if i % 20 == 0:
                t.record_keyframe_admitted()
        t.record_optimizer_step(i)
    block = realized_cadence_report_block(
        t, requested_steps_per_frame=10.0, frame_scheduler_spf=10.0,
        ingestion_mode="iter_based", frame_admission="all")

    # A gate ran despite the requested policy reading "all".
    assert block["clocks"]["candidate_evaluated"] == 50
    # Observation and admission legitimately diverge -- this is soft-all
    # working, not coverage loss.
    assert block["clocks"]["source_ingested"] == 50
    assert block["clocks"]["keyframe_admitted"] == 25
    assert (block["clocks"]["keyframe_admitted"]
            < block["clocks"]["source_ingested"])
    # The tracker records it without any verdict claiming corruption.
    assert block["reconciliation"]["verdict"] in (
        "ok", "diverged", "not_iteration_paced", "insufficient_evidence")


# ----------------------------------------------------------------- wiring ----

def test_scheduler_spf_and_mapping_effective_spf_are_separate_fields():
    """CLAUDE.md: the two "150"s in a startup log are different variables that
    coincide by accident. The block must not persist one under the other's name
    -- a reader diffing against train.log's [MappingScheduler] line would be
    silently misled on any run where they differ (every --online_mode run, where
    effective_spf is never handed to FrameScheduler at all)."""
    t = _simulate(steps=2000, spf=50)
    v = t.reconcile(requested_steps_per_frame=50.0, frame_scheduler_spf=50.0,
                    mapping_scheduler_effective_spf=150.0,
                    ingestion_mode="dataset_fps")
    assert v["frame_scheduler_spf"] == 50.0
    assert v["mapping_scheduler_effective_spf"] == 150.0
    assert v["governing_source"] == "frame_scheduler_spf"


def test_write_post_training_report_accepts_realized_cadence():
    pytest.importorskip("torch")   # CPU-smoke image has no torch (see conftest)
    from eligsir.utils.comparison_report import write_post_training_report

    sig = inspect.signature(write_post_training_report)
    p = sig.parameters["realized_cadence"]
    assert p.default is None, "must be optional/additive (absent when omitted)"
    assert p.kind is inspect.Parameter.KEYWORD_ONLY


def test_report_writer_persists_and_mirrors_into_ingestion():
    """The persistence branch is exercised directly (the full report writer needs
    a GPU and real cameras): same dict logic, no rendering."""
    pytest.importorskip("torch")   # CPU-smoke image has no torch (see conftest)
    from eligsir.utils.comparison_report import write_post_training_report

    src = inspect.getsource(write_post_training_report)
    assert 'summary["realized_cadence"] = dict(realized_cadence)' in src
    assert 'summary["ingestion"]["cadence_verdict"]' in src
    # Guarded so omitting the kwarg reproduces the pre-#289 report exactly.
    assert "if realized_cadence:" in src


def test_train_streaming_records_all_seven_clocks():
    src = TRAIN_STREAMING.read_text()
    for call in (
        "_cadence.record_optimizer_step(",
        "_cadence.record_source_released()",
        "_cadence.record_source_ingested()",
        "_cadence.record_candidate_evaluated()",
        "_cadence.record_keyframe_admitted()",
        "_cadence.record_insertion_commit(",
        "_cadence.record_supervision_view()",
        "_cadence.record_mcmc_mutation()",
        "_cadence.record_backlog(",
    ):
        assert call in src, f"missing cadence call site: {call}"
    # Release is recorded at every mark_released site -- see
    # test_release_is_not_ingest. Three today: the normal ingest path, the
    # BurstBudget-drop path, and (issue #601) the bounded backpressure
    # owner's skip path. A skipped observation is still RELEASED on the
    # #289 release clock; release and ingest are different clocks.
    assert src.count("_cadence.record_source_released()") == 3
    # Three optimizer-step call sites, and all three are mutually exclusive --
    # exactly one records per iteration, so the denominator cannot double-count:
    #   * the inline arm's two `optimizer.step()` branches (selective_adam vs
    #     dense), which are an if/else of each other; and
    #   * the engine arm (issue #809), which takes its step inside
    #     run_batched_step() and reports its own real count.
    # The engine arm is the `if` and the whole inline arm is the `else` of the
    # same statement, and the inline arm's `if iteration < opt.iterations:`
    # block -- which owns the other two call sites -- is nested inside that
    # `else`. So an engine-arm iteration never reaches them.
    assert src.count("_cadence.record_optimizer_step(") == 3
    # Five insertion-commit call sites: the batched in-loop flush, the
    # mapping-endpoint quiescence flush, the exact causal-fork boundary flush,
    # the end-of-run batched flush, and the non-batch per-frame commit (issue
    # #727 -- the non-batch path is what a default streaming run actually
    # takes; it was silently missing this call before the fix).
    assert src.count("_cadence.record_insertion_commit(") == 5
    # And the block reaches report.json. Asserted as "the builder is called
    # AND the report key is emitted" rather than as one literal call
    # expression: issue #602 hoisted the builder call to a local so the same
    # block can also be embedded in report.json["admission_cadence"] without
    # being built twice (two independently-built copies of one block in a
    # single report is how they silently desync). Pinning the inline call text
    # would forbid that hoist for no safety gain.
    # #1075 task P3: the report-kwargs builder that owns this block moved out
    # of train_streaming.py's closure into eligsir/runtime/report_kwargs.py.
    # The build-once-then-reference property is exactly what moved, so it is
    # asserted against the file that now holds it.
    builder_src = (REPO / "eligsir" / "runtime" / "report_kwargs.py").read_text()
    assert "realized_cadence_report_block(" in builder_src
    assert '"realized_cadence":' in builder_src
    # Built exactly once, then referenced -- not rebuilt per consumer.
    assert builder_src.count("realized_cadence_report_block(\n") == 1


def _branch_path(parents, node):
    """Ancestor chain of (If-node, "body"|"orelse") pairs enclosing `node`."""
    path = []
    cur = node
    while cur in parents:
        parent, field = parents[cur]
        if isinstance(parent, ast.If) and field in ("body", "orelse"):
            path.append((parent, field))
        cur = parent
    return path


def test_optimizer_step_call_sites_are_mutually_exclusive():
    """No iteration may record two optimizer steps (issue #289 denominator).

    A bare count assertion cannot catch the regression that actually matters:
    adding a call site on a path that ALSO runs the inline arm's step would
    silently double the denominator and halve every cadence ratio. This checks
    the real invariant structurally -- every pair of `record_optimizer_step`
    call sites must sit on opposite branches of some shared `if`.
    """
    tree = ast.parse(TRAIN_STREAMING.read_text())
    parents = {}
    for parent in ast.walk(tree):
        for field, value in ast.iter_fields(parent):
            for child in (value if isinstance(value, list) else [value]):
                if isinstance(child, ast.AST):
                    parents[child] = (parent, field)

    sites = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "record_optimizer_step"
    ]
    assert len(sites) >= 2, f"expected multiple call sites, found {len(sites)}"

    paths = {id(s): _branch_path(parents, s) for s in sites}
    for i, a in enumerate(sites):
        for b in sites[i + 1:]:
            branches_a = {id(n): f for n, f in paths[id(a)]}
            exclusive = any(
                id(n) in branches_a and branches_a[id(n)] != f
                for n, f in paths[id(b)]
            )
            assert exclusive, (
                f"record_optimizer_step call sites at lines {a.lineno} and "
                f"{b.lineno} are NOT on mutually exclusive branches -- an "
                f"iteration reaching both would double-count the #289 "
                f"optimizer-step denominator."
            )


# ------------------------------------------------------- issue #727 fix ----

def test_clock_coverage_documents_insertion_as_complete():
    t = RealizedCadenceTracker()
    cov = t.summary()["clock_coverage"]
    assert cov["insertion_commit"]["complete"] is True
    assert cov["inserted_gaussians"]["complete"] is True
    assert "#727" in cov["insertion_commit"]["note"]


def test_clock_coverage_documents_mcmc_mutation_as_partial_not_a_bare_zero():
    """Issue #727's second acceptance criterion: a zero in mcmc_mutation must
    be distinguishable from 'not measured'. It is still partial (relocation-
    only mutations aren't caught -- see #326), so it must say so explicitly
    rather than presenting the same unqualified 'complete' shape insertion_
    commit now gets."""
    t = RealizedCadenceTracker()
    cov = t.summary()["clock_coverage"]
    assert cov["mcmc_mutation"]["complete"] is False
    assert "#326" in cov["mcmc_mutation"]["note"]
    assert cov["mcmc_mutation"]["note"] == MCMC_MUTATION_COVERAGE_NOTE
    assert CLOCK_COVERAGE["mcmc_mutation"]["complete"] is False


def test_non_batch_insertion_path_now_records_the_commit():
    """Root-cause regression pin for #727: the non-batch branch (what a
    default streaming run -- streaming_insertion_batch_frames=1,
    streaming_insertion_batch_steps=0 -- actually executes) must call
    record_insertion_commit, not just the batched branch. Source-inspection
    test (no GPU/training needed) mirroring the existing
    test_train_streaming_records_all_seven_clocks convention."""
    src = TRAIN_STREAMING.read_text()
    idx = src.index("else:\n                        added = cands_or_n")
    # The non-batch branch body, up to the next top-level statement after it
    # (total_inserted += added), must contain the cadence call.
    end = src.index("total_inserted += added", idx)
    branch_body = src[idx:end]
    assert "_cadence.record_insertion_commit(added)" in branch_body


def test_mcmc_mutation_gate_also_checks_population_delta():
    """Root-cause regression pin for #727: the mcmc_mutation clock must not be
    gated on map_backend.topology_version ALONE -- #543 found statically that
    no MCMC growth/relocation path ever advances it, which made the clock
    structurally zero on every run. The gate must also fire on a population-
    size change (catches growth/add events)."""
    src = TRAIN_STREAMING.read_text()
    idx = src.index("_mcmc_n_after = int(gaussians.get_xyz.shape[0])")
    gate = src[idx:idx + 400]
    assert "map_backend.topology_version != _w3_02_mcmc_topo_before" in gate
    assert "_mcmc_n_after != int(_w3_02_mcmc_n_before)" in gate
    assert "_cadence.record_mcmc_mutation()" in gate


def test_no_new_per_iteration_device_sync_in_recording():
    """Standing repo invariant: observability must not measurably slow training.
    Every record_* body must be plain Python -- no torch, no .item(), no .cpu()."""
    import eligsir.utils.realized_cadence as rc

    src = inspect.getsource(rc)
    assert "import torch" not in src
    for method in (
        rc.RealizedCadenceTracker.record_optimizer_step,
        rc.RealizedCadenceTracker.record_source_released,
        rc.RealizedCadenceTracker.record_keyframe_admitted,
        rc.RealizedCadenceTracker.record_insertion_commit,
        rc.RealizedCadenceTracker.record_supervision_view,
    ):
        body = inspect.getsource(method)
        assert ".item()" not in body
        assert ".cpu()" not in body
        assert "synchronize" not in body
