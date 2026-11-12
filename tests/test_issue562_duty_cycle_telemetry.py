"""Issue #562: CPU/GPU duty-cycle timeline + stall taxonomy.

DELIBERATELY TORCH-FREE. ``utils/duty_cycle.py`` is a leaf module (json/os/
threading/time only) and ``utils/ingest_phase_profiler.py`` imports torch only
lazily inside ``_cuda_sync``, so this file collects and executes on BOTH the
CPU-CI image (``Dockerfile.ci-cpu``) and the training image -- see
``docs/architecture/2026-08-14-issue552-test-surface-contract.md``. A single
``import torch`` here would silently remove the whole file from CPU CI's
``collect_ignore`` (#536/#649), so do not add one.

Time is injected via a fake clock everywhere a duration is asserted, so nothing
here is wall-clock-flaky. The one genuinely timing-based test (the overhead
microbenchmark) only prints its number and asserts a very loose ceiling.
"""

from __future__ import annotations

import json
import os
import threading

import pytest

from eligsir.utils import duty_cycle as dc
from eligsir.utils import ingest_phase_profiler as ipp


class FakeClock:
    """Monotonic clock the test advances explicitly."""

    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def timeline(clock):
    return dc.DutyCycleTimeline(window_iters=2, stall_ms=1000.0, clock=clock)


@pytest.fixture(autouse=True)
def _detach_recorder():
    """No test may leave a recorder latched into the shared module global."""
    yield
    ipp.set_duty_cycle_recorder(None)


# ---------------------------------------------------------------------------
# Taxonomy
# ---------------------------------------------------------------------------


def test_every_existing_timed_region_name_classifies_into_a_real_category():
    """The taxonomy must cover the region vocabulary that exists TODAY.

    A name that fell through to `other_instrumented` would be a silent coverage
    hole, which is the exact failure mode the `unattributed` bucket exists to
    make visible.
    """
    # Every name produced by `grep -rho '_timed("[a-z_/]*"' --include=*.py .`
    # at the commit this test was written, minus the two `region_x`/`region_y`
    # fixtures that only exist inside tests.
    real_regions = [
        "after_mcmc_noise", "after_optstep", "bbox_prune", "depth_insertion",
        "ingest_accumulate_surface", "ingest_admission_fn",
        "ingest_frame_to_camera", "ingest_get_source_frame", "ingest_pregate",
        "loop_closure_optimize", "m_backward", "m_grad_masking", "m_ingest",
        "m_loss_assembly", "m_loss_item", "m_noise", "m_optimization_engine",
        "m_optstep", "m_post_render", "m_render", "m_sanitize",
        "mid_training_report_snapshot", "occupancy_rebuild",
        "online_controller_tick", "online_decide", "online_submap",
        "online_tb", "online_track", "save_capture", "save_enqueue",
        "save_snapshot", "step_post_backward", "surface/backproject",
        "surface/mask", "surface/materialize", "surface/quality_sensor_noise",
        "surface/quality_thickness", "surface/voxelize",
    ]
    unmapped = [r for r in real_regions if dc.classify_region(r) == "other_instrumented"]
    assert unmapped == [], f"unclassified _timed regions: {unmapped}"
    for r in real_regions:
        assert dc.classify_region(r) in dc.CATEGORIES


def test_unknown_region_is_visible_not_guessed():
    assert dc.classify_region("brand_new_region_nobody_mapped") == "other_instrumented"
    assert "other_instrumented" in dc.CATEGORIES


def test_exact_mapping_beats_prefix_mapping():
    # `ingest_` is a prefix rule pointing at ingest_decode; these two must keep
    # their exact `admission` classification.
    assert dc.classify_region("ingest_admission_fn") == "admission"
    assert dc.classify_region("ingest_pregate") == "admission"
    assert dc.classify_region("ingest_get_source_frame") == "ingest_decode"


def test_every_category_has_a_stall_class():
    assert set(dc.STALL_CLASS) == set(dc.CATEGORIES)


# ---------------------------------------------------------------------------
# The partition is exhaustive by construction
# ---------------------------------------------------------------------------


def test_partition_sums_to_wall_time_and_names_the_unattributed_residual(timeline, clock):
    clock.advance(0.10)              # outside any region
    timeline.enter("m_render")
    clock.advance(0.30)
    timeline.exit("m_render")
    clock.advance(0.05)              # outside any region
    timeline.enter("m_ingest")
    clock.advance(0.55)
    timeline.exit("m_ingest")

    s = timeline.summary()
    assert s["wall_s"] == pytest.approx(1.0)
    assert s["partition_exhaustive"] is True
    assert s["partition_residual_s"] == pytest.approx(0.0, abs=1e-9)
    assert sum(s["frac"].values()) == pytest.approx(1.0)
    assert s["frac"]["optimization"] == pytest.approx(0.30)
    assert s["frac"]["ingest_decode"] == pytest.approx(0.55)
    # The 0.10 + 0.05 nobody instrumented is VISIBLE, not folded into a
    # neighbouring bucket.
    assert s["frac"]["unattributed"] == pytest.approx(0.15)


def test_nested_regions_bill_innermost_only(timeline, clock):
    """Unlike ingest_phase_profile.json's deliberately >100% nested sums."""
    timeline.enter("m_ingest")                 # ingest_decode
    clock.advance(0.2)
    timeline.enter("ingest_admission_fn")      # admission, nested
    clock.advance(0.5)
    timeline.exit("ingest_admission_fn")
    clock.advance(0.3)
    timeline.exit("m_ingest")

    s = timeline.summary()
    assert s["frac"]["ingest_decode"] == pytest.approx(0.5)   # 0.2 + 0.3
    assert s["frac"]["admission"] == pytest.approx(0.5)
    assert sum(s["frac"].values()) == pytest.approx(1.0)


def test_optimizer_region_frac_is_the_optimization_fraction(timeline, clock):
    timeline.enter("m_render")
    clock.advance(0.25)
    timeline.exit("m_render")
    clock.advance(0.75)
    s = timeline.summary()
    assert s["optimizer_region_frac"] == pytest.approx(0.25)
    assert s["optimizer_region_frac"] == s["frac"]["optimization"]


# ---------------------------------------------------------------------------
# Robustness: the ways a real call site can misbehave
# ---------------------------------------------------------------------------


def test_leaked_bracket_is_unwound_at_the_iteration_boundary(timeline, clock):
    """An unpaired __enter__ must not mis-bill the entire rest of the run."""
    timeline.enter("depth_insertion")
    clock.advance(0.4)
    timeline.iteration_boundary(1)   # bracket never exited
    clock.advance(0.6)               # must NOT be billed to depth_insertion
    s = timeline.summary()
    assert timeline.unclosed_at_iteration_boundary == 1
    assert s["frac"]["depth_insertion"] == pytest.approx(0.4)
    assert s["frac"]["unattributed"] == pytest.approx(0.6)


def test_out_of_order_exit_is_counted_not_silently_corrupting(timeline, clock):
    timeline.enter("m_render")
    timeline.enter("m_loss_item")
    clock.advance(0.5)
    timeline.exit("m_render")        # non-LIFO close
    clock.advance(0.5)
    timeline.exit("m_loss_item")
    s = timeline.summary()
    assert s["anomalies"]["out_of_order_exits"] == 1
    assert s["partition_exhaustive"] is True


def test_unmatched_exit_is_counted(timeline, clock):
    clock.advance(0.1)
    timeline.exit("never_entered")
    s = timeline.summary()
    assert s["anomalies"]["unmatched_exits"] == 1
    assert s["partition_exhaustive"] is True


def test_worker_thread_events_are_dropped_and_counted(timeline, clock):
    """A save-worker thread's save_enqueue must not steal the main thread's
    attribution mid-region (AsyncSaveWorker really does emit these)."""
    timeline.enter("m_optstep")

    def worker():
        timeline.enter("save_enqueue")
        timeline.exit("save_enqueue")

    t = threading.Thread(target=worker)
    t.start()
    t.join()

    clock.advance(1.0)
    timeline.exit("m_optstep")
    s = timeline.summary()
    assert s["anomalies"]["foreign_thread_events"] == 2
    assert s["frac"]["optimization"] == pytest.approx(1.0)
    assert s["frac"]["save_checkpoint"] == pytest.approx(0.0)
    assert s["partition_exhaustive"] is True


# ---------------------------------------------------------------------------
# Stall taxonomy: the classification is evidence-based
# ---------------------------------------------------------------------------


def test_synthetic_stalls_classify_by_the_region_that_was_open(clock):
    tl = dc.DutyCycleTimeline(window_iters=10, stall_ms=100.0, clock=clock)

    tl.enter("ingest_get_source_frame")   # source read sleep -> source I/O
    clock.advance(0.5)
    tl.exit("ingest_get_source_frame")

    tl.enter("save_snapshot")             # snapshot sleep -> save
    clock.advance(0.4)
    tl.exit("save_snapshot")

    clock.advance(0.3)                    # bare CPU sleep, no region -> unknown
    tl.enter("m_render")
    clock.advance(0.01)                   # below threshold, not a stall
    tl.exit("m_render")

    classes = {s["category"]: s["stall_class"] for s in tl.top_stalls()}
    assert classes["ingest_decode"] == "GPU_IDLE_WAIT_SOURCE_IO"
    assert classes["save_checkpoint"] == "GPU_IDLE_WAIT_SAVE_SNAPSHOT"
    assert classes["unattributed"] == "GPU_IDLE_UNKNOWN"
    assert "optimization" not in classes          # 10 ms is under the threshold
    assert tl.top_stalls()[0]["duration_ms"] == pytest.approx(500.0)


def test_optimizer_only_loop_shows_high_feed_efficiency(clock):
    tl = dc.DutyCycleTimeline(window_iters=1000, stall_ms=0.0, clock=clock)
    for i in range(100):
        tl.enter("m_render")
        clock.advance(0.0099)
        tl.exit("m_render")
        clock.advance(0.0001)
        tl.iteration_boundary(i)
    s = tl.summary()
    assert s["optimizer_region_frac"] > 0.98
    assert s["unattributed_frac"] < 0.02


def test_stall_list_is_bounded(clock):
    tl = dc.DutyCycleTimeline(window_iters=10, stall_ms=1.0, max_stalls=5, clock=clock)
    for _ in range(50):
        tl.enter("m_ingest")
        clock.advance(0.01)
        tl.exit("m_ingest")
    assert len(tl.top_stalls()) == 5


# ---------------------------------------------------------------------------
# Windows, artifacts, report block
# ---------------------------------------------------------------------------


def test_windows_close_on_cadence_and_carry_fracs(timeline, clock):
    # window_iters=2
    timeline.enter("m_render")
    clock.advance(1.0)
    timeline.exit("m_render")
    assert timeline.iteration_boundary(1) is None
    clock.advance(1.0)
    row = timeline.iteration_boundary(2)
    assert row is not None
    assert row["iterations"] == 2
    assert row["wall_s"] == pytest.approx(2.0)
    assert row["frac"]["optimization"] == pytest.approx(0.5)
    assert row["optimizer_region_frac"] == pytest.approx(0.5)
    assert sum(row["frac"].values()) == pytest.approx(1.0)
    scalars = timeline.tb_scalars()
    assert scalars["duty_cycle/optimizer_region_frac"] == pytest.approx(0.5)
    assert scalars["duty_cycle/window_it_per_s"] == pytest.approx(1.0)


def test_window_ring_is_bounded_and_drops_are_reported(clock):
    tl = dc.DutyCycleTimeline(window_iters=1, max_windows=3, clock=clock)
    for i in range(10):
        clock.advance(0.01)
        tl.iteration_boundary(i)
    s = tl.summary()
    assert s["n_windows"] == 10
    assert s["windows_retained"] == 3
    assert s["windows_dropped"] == 7


def test_report_block_is_tri_state_distinguishable(timeline, clock):
    clock.advance(1.0)
    on = timeline.report_block()
    assert on["enabled"] is True
    assert on["n_windows"] == 0          # on, but never closed a window
    off = dc.disabled_report_block("telemetry disabled (flag not set)")
    assert off["enabled"] is False
    assert "flag not set" in off["reason"]
    assert off["optimizer_region_frac"] is None
    # ...and both are JSON-serializable, since they land in report.json.
    json.dumps(on)
    json.dumps(off)


def test_write_artifacts_emits_the_performance_trace_directory(tmp_path, clock):
    tl = dc.DutyCycleTimeline(window_iters=1, stall_ms=10.0, clock=clock)
    tl.enter("m_render")
    clock.advance(0.5)
    tl.exit("m_render")
    tl.iteration_boundary(1)

    out = tl.write_artifacts(str(tmp_path))
    assert out == os.path.join(str(tmp_path), "performance_trace")
    summary = json.loads(open(os.path.join(out, "summary.json")).read())
    assert summary["taxonomy_version"] == dc.TAXONOMY_VERSION
    assert summary["partition_exhaustive"] is True
    rows = [json.loads(ln) for ln in open(os.path.join(out, "duty_cycle.jsonl")) if ln.strip()]
    assert len(rows) == 1 and rows[0]["window"] == 0
    stalls = json.loads(open(os.path.join(out, "stall_top.json")).read())
    assert stalls["stalls"][0]["category"] == "optimization"


def test_format_line_is_a_single_line(timeline, clock):
    clock.advance(1.0)
    line = timeline.format_line()
    assert line.startswith("[duty-cycle]") and "\n" not in line


# ---------------------------------------------------------------------------
# The disabled path adds NO per-iteration work (the #480 invariant)
# ---------------------------------------------------------------------------


def test_disabled_timed_does_no_work_at_all(monkeypatch):
    """With both consumers off, `_timed` must not read the clock, must not
    touch any store, and must not call into any recorder."""
    ipp.set_duty_cycle_recorder(None)
    assert ipp._DUTY is None

    if ipp._PROFILE_PHASES:
        pytest.skip("SPLAT_PROFILE_PHASES is set in this environment")

    calls = {"clock": 0, "sync": 0}

    class CountingTime:
        @staticmethod
        def perf_counter():
            calls["clock"] += 1
            return 0.0

    monkeypatch.setattr(ipp, "time", CountingTime)
    monkeypatch.setattr(ipp, "_cuda_sync", lambda: calls.__setitem__("sync", calls["sync"] + 1))
    ipp.reset_all()

    for _ in range(1000):
        with ipp._timed("m_render"):
            pass

    assert calls == {"clock": 0, "sync": 0}
    assert ipp._PHASE_OPS == {}
    assert ipp._PHASE_OPS_LIFETIME == {}


def test_recorder_uninstall_fully_detaches(clock):
    """A recorder left latched in the shared module global would make every
    later region in the process keep feeding a dead timeline."""
    tl = dc.DutyCycleTimeline(clock=clock)
    ipp.set_duty_cycle_recorder(tl)
    assert ipp.duty_cycle_recorder() is tl
    ipp.set_duty_cycle_recorder(None)
    assert ipp.duty_cycle_recorder() is None
    before = tl.enter_events
    with ipp._timed("m_render"):
        clock.advance(1.0)
    assert tl.enter_events == before


def test_profile_phases_alone_still_drives_timed(monkeypatch):
    """The two guards are independent: flipping `_PROFILE_PHASES` alone (which
    is how this module's own suite drives it) must still record."""
    ipp.set_duty_cycle_recorder(None)
    monkeypatch.setattr(ipp, "_PROFILE_PHASES", True)
    monkeypatch.setattr(ipp, "_cuda_sync", lambda: None)
    ipp.reset_all()
    with ipp._timed("m_render"):
        pass
    assert "m_render" in ipp._PHASE_OPS_LIFETIME
    ipp.reset_all()


def test_timed_feeds_the_installed_timeline(clock):
    """The reuse claim: existing `_timed()` call sites, not a second hook set."""
    tl = dc.DutyCycleTimeline(window_iters=1, clock=clock)
    ipp.set_duty_cycle_recorder(tl)
    with ipp._timed("m_backward"):
        clock.advance(0.75)
    clock.advance(0.25)
    s = tl.summary()
    assert s["frac"]["optimization"] == pytest.approx(0.75)
    assert s["frac"]["unattributed"] == pytest.approx(0.25)


def test_a_raising_recorder_is_disabled_and_never_masks_the_real_exception(capsys):
    """Telemetry failing inside __exit__ must not replace the training error."""

    class Exploding:
        def enter(self, name):
            raise RuntimeError("boom-enter")

        def exit(self, name):
            raise RuntimeError("boom-exit")

    ipp.set_duty_cycle_recorder(Exploding())
    with pytest.raises(ValueError, match="real training error"):
        with ipp._timed("m_render"):
            raise ValueError("real training error")
    # It self-disabled rather than raising again on the next region.
    assert ipp.duty_cycle_recorder() is None
    with ipp._timed("m_render"):
        pass
    assert "[duty-cycle] WARNING" in capsys.readouterr().out


def test_enabled_per_event_overhead_is_microseconds():
    """Overhead argument, measured rather than asserted in prose.

    Loose ceiling on purpose -- this runs on shared CI hardware. The number is
    printed so the PR body can quote a real measurement.
    """
    import time as _time

    tl = dc.DutyCycleTimeline(window_iters=10 ** 9)
    ipp.set_duty_cycle_recorder(tl)
    n = 20000
    t0 = _time.perf_counter()
    for _ in range(n):
        with ipp._timed("m_render"):
            pass
    dt = _time.perf_counter() - t0
    per_region_us = dt / n * 1e6
    print(f"\n[issue562] enabled _timed region (enter+exit): {per_region_us:.2f} us")
    assert per_region_us < 100.0


# ---------------------------------------------------------------------------
# Flags
# ---------------------------------------------------------------------------


def test_train_streaming_wiring_is_reachable():
    """Source-text reachability check for the `train_streaming.py` wiring.

    No test here can RUN the streaming loop (it needs a GPU, a dataset and
    hours), so the wiring is pinned the same way this repo already pins other
    unrunnable invariants (`test_issue333_*_reachability.py`,
    `test_getattr_fallback_freshness.py`): against the source text. Without
    this, every assertion in this file could pass while the timeline is wired
    to nothing.
    """
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src = open(os.path.join(repo_root, "train_streaming.py")).read()

    # constructed only under the flag, and installed into the shared hook
    assert 'getattr(args, "streaming_duty_cycle_telemetry", False)' in src
    assert "DutyCycleTimeline(" in src
    assert "set_duty_cycle_recorder(_duty)" in src
    # per-iteration boundary + window scalars
    assert "_duty.iteration_boundary(iteration)" in src
    assert "_duty.tb_scalars()" in src
    # the one new region bracket, entered and exited exactly once
    assert src.count('_timed("depth_insertion")') == 1
    assert src.count("_duty_insert_bracket.__enter__()") == 1
    assert src.count("_duty_insert_bracket.__exit__(") == 1
    # report block reaches the report kwargs dict, which is splatted into
    # write_post_training_report
    # #1075 task P3: the report-kwargs builder body now lives in
    # eligsir/runtime/report_kwargs.py, where `_duty` is the explicit
    # `duty` parameter. The wiring is unchanged -- the block still reaches the
    # kwargs dict that train_streaming.py splats into the report writer.
    builder_src = open(
        os.path.join(repo_root, "eligsir", "runtime", "report_kwargs.py")
    ).read()
    assert '_extra["duty_cycle"] = duty.report_block()' in builder_src
    assert "duty=_duty," in src
    assert "**_streaming_report_kwargs(final=True)" in src
    # finalization: detach, then write artifacts once
    assert "_duty.write_artifacts(args.model_path)" in src


def test_report_writer_accepts_and_emits_the_duty_cycle_kwarg():
    """The kwarg must exist on `write_post_training_report`; a key that lands in
    the report-kwargs dict but has no parameter would be a TypeError at report
    time, and one that is accepted but never written is silently absent.

    Checked by AST over the source rather than by importing
    `utils.comparison_report` -- importing it would make this whole file
    torch-needing and silently drop it out of CPU CI (#536/#649).
    """
    import ast

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(repo_root, "utils", "comparison_report.py")
    src = open(path).read()
    tree = ast.parse(src)
    fn = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "write_post_training_report"
    )
    params = {a.arg for a in fn.args.args} | {a.arg for a in fn.args.kwonlyargs}
    assert "duty_cycle" in params
    assert 'summary["duty_cycle"] = dict(duty_cycle)' in src


def test_argparse_defaults_and_toggle():
    import argparse

    from arguments import StreamingParams

    parser = argparse.ArgumentParser()
    sp = StreamingParams(parser)
    filled = sp.extract(parser.parse_args([]))
    assert filled.streaming_duty_cycle_telemetry is False   # default OFF
    assert filled.streaming_duty_cycle_window == 100
    assert filled.streaming_duty_cycle_max_windows == 4096
    assert filled.streaming_duty_cycle_stall_ms == 250.0
    assert filled.streaming_duty_cycle_max_stalls == 64

    on = sp.extract(parser.parse_args(["--streaming_duty_cycle_telemetry"]))
    assert on.streaming_duty_cycle_telemetry is True
