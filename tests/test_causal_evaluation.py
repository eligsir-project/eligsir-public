from __future__ import annotations

from types import SimpleNamespace
import json

import numpy as np
import pytest
import eligsir.utils as utils


def _frame(index: int, *, tx: float = 0.0):
    c2w = np.eye(4, dtype=np.float32)
    c2w[0, 3] = tx
    return SimpleNamespace(index=index, source_ordinal=index, c2w=c2w)


def _args(**overrides):
    values = {
        "streaming_eval_hold": 4,
        "streaming_initial_frames": 5,
        "streaming_eval_guard": 1,
        "streaming_eval_min_baseline_m": 0.03,
        "streaming_eval_min_angle_deg": 2.0,
        "source_path": "/data/fr3",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_causal_holdout_plan_is_precomputed_and_excludes_bootstrap_and_guards():
    from eligsir.runtime.causal_evaluation import causal_holdout_plan

    frames = tuple(_frame(i, tx=0.1 * i) for i in range(20))
    plan = causal_holdout_plan(frames, _args())

    assert plan.holdout_ordinals == frozenset({8, 12, 16})
    assert plan.guard_ordinals == frozenset({7, 9, 11, 13, 15, 17})
    assert plan.role_for_ordinal(8) == "holdout"
    assert plan.role_for_ordinal(9) == "guard"
    assert plan.role_for_ordinal(10) == "train"
    assert plan.eval_metadata["holdout_selection"] == "precomputed"
    assert plan.eval_metadata["n_holdout"] == 3
    assert plan.eval_metadata["holdout_hash"]


def test_evwb_reference_requires_exact_canonical_training_source_without_mutating_catalogue():
    from eligsir.runtime.causal_evaluation import (
        causal_holdout_plan,
        validate_train_evwb_reference_source,
    )
    from eligsir.runtime.replay import streaming_frame_source_identity

    frames = tuple(_frame(i, tx=0.1 * i) for i in range(20))
    for frame in frames:
        frame.timestamp = float(frame.index)
    args = _args(
        streaming_train_evwb_enabled=True,
        streaming_train_evwb_reference_source=streaming_frame_source_identity(
            frames[10], source_id="test-source"
        ).fingerprint(),
    )
    plan = causal_holdout_plan(frames, args)
    before = tuple((frame.index, frame.c2w.copy()) for frame in frames)

    validate_train_evwb_reference_source(
        frames, plan, args, source_id="test-source"
    )

    assert plan.role_for_ordinal(10) == "train"
    assert all(
        frame.index == index and np.array_equal(frame.c2w, pose)
        for frame, (index, pose) in zip(frames, before, strict=True)
    )


@pytest.mark.parametrize("ordinal, expected_role", [(8, "holdout"), (7, "guard")])
def test_evwb_reference_rejects_nontraining_canonical_source(ordinal, expected_role):
    from eligsir.runtime.causal_evaluation import (
        causal_holdout_plan,
        validate_train_evwb_reference_source,
    )
    from eligsir.runtime.replay import streaming_frame_source_identity

    frames = tuple(_frame(i, tx=0.1 * i) for i in range(20))
    for frame in frames:
        frame.timestamp = float(frame.index)
    args = _args(
        streaming_train_evwb_enabled=True,
        streaming_train_evwb_reference_source=streaming_frame_source_identity(
            frames[ordinal], source_id="test-source"
        ).fingerprint(),
    )
    plan = causal_holdout_plan(frames, args)

    with pytest.raises(ValueError, match=expected_role):
        validate_train_evwb_reference_source(
            frames, plan, args, source_id="test-source"
        )


def test_evwb_reference_rejects_unknown_canonical_source():
    from eligsir.runtime.causal_evaluation import (
        causal_holdout_plan,
        validate_train_evwb_reference_source,
    )

    frames = tuple(_frame(i, tx=0.1 * i) for i in range(20))
    for frame in frames:
        frame.timestamp = float(frame.index)
    args = _args(
        streaming_train_evwb_enabled=True,
        streaming_train_evwb_reference_source="unknown-source-fingerprint",
    )
    plan = causal_holdout_plan(frames, args)

    with pytest.raises(ValueError, match="does not resolve"):
        validate_train_evwb_reference_source(
            frames, plan, args, source_id="test-source"
        )


def test_causal_holdout_plan_pose_filter_falls_back_without_losing_test_set():
    from eligsir.runtime.causal_evaluation import causal_holdout_plan

    frames = tuple(_frame(i, tx=1e-5 * i) for i in range(20))
    plan = causal_holdout_plan(
        frames,
        _args(streaming_initial_frames=0, streaming_eval_guard=0),
    )

    assert plan.holdout_ordinals == frozenset({0, 4, 8, 12, 16})
    assert not plan.guard_ordinals


def test_causal_holdout_plan_flags_severe_collapse_without_losing_evidence(capsys):
    """issue #1337: on a smooth/high-fps trajectory, a too-permissive
    --streaming_eval_guard can make nearly every reserved candidate fail the
    pose-gap check against its nearest neighbor, collapsing the realized
    holdout set to a tiny fraction of the reserved count (confirmed as low
    as 1 of ~243 on Replica) while still reporting a populated holdout_hash.
    Unlike the total-collapse-to-zero case (the test above), this is a
    *partial* collapse the existing candidates-but-no-holdouts fallback does
    not catch -- it must be flagged, not silently accepted as a valid eval."""
    from eligsir.runtime.causal_evaluation import causal_holdout_plan

    # A smooth trajectory (tiny per-frame translation) with one isolated
    # large pose jump exactly at ordinal 24 -- the only candidate whose pose
    # is far from BOTH its nearest neighbors, so it is the only one of 10
    # reserved candidates (eval_hold=3 over 30 frames) that clears the gate.
    frames = tuple(
        _frame(i, tx=5.0 if i == 24 else 1e-6 * i) for i in range(30)
    )
    plan = causal_holdout_plan(
        frames,
        _args(streaming_eval_hold=3, streaming_initial_frames=0, streaming_eval_guard=1),
    )

    assert plan.holdout_ordinals == frozenset({24})
    assert plan.eval_metadata["n_holdout"] == 1
    assert plan.eval_metadata["n_holdout_candidates"] == 10
    assert plan.eval_metadata["holdout_severely_collapsed"] is True
    assert plan.eval_metadata["holdout_hash"]  # still populated -- not silently invalid-looking
    assert "collapsed to 1 of 10" in capsys.readouterr().out


def test_causal_holdout_plan_reserves_trajectory_spanning_control_probe():
    from eligsir.runtime.causal_evaluation import causal_holdout_plan
    from eligsir.utils.partition_eval import partition_count, partition_id

    frames = tuple(_frame(i, tx=0.1 * i) for i in range(2_520))
    plan = causal_holdout_plan(
        frames,
        _args(
            streaming_partition_focus_weight=0.5,
            streaming_partition_focus_signal_source="control_probe",
            streaming_partition_focus_control_probe_n=32,
        ),
    )

    control_ordinals = tuple(record.source_ordinal for record in plan.control_probe_records)
    assert len(control_ordinals) == 32
    assert set(control_ordinals).isdisjoint(plan.holdout_ordinals)
    assert all(plan.role_for_ordinal(ordinal) == "train" for ordinal in control_ordinals)
    assert {
        partition_id(ordinal, len(frames) - 1, k=partition_count(len(frames)))
        for ordinal in control_ordinals
    } == set(range(partition_count(len(frames))))


def test_causal_holdout_hash_changes_with_realized_source_identity():
    from eligsir.runtime.causal_evaluation import causal_holdout_plan

    frames = tuple(_frame(i, tx=0.1 * i) for i in range(20))
    a = causal_holdout_plan(frames, _args(source_path="/data/fr3"))
    b = causal_holdout_plan(frames, _args(source_path="/data/fr1"))

    assert a.eval_metadata["holdout_hash"] != b.eval_metadata["holdout_hash"]


def test_causal_holdout_camera_keeps_stable_source_identity_and_lazy_original_rgb():
    pytest.importorskip("torch")
    from eligsir.runtime.causal_evaluation import (
        build_causal_holdout_cameras,
        causal_holdout_plan,
    )

    frames = []
    for index in range(8):
        frame = _frame(index, tx=0.1 * index)
        frame.fx = frame.fy = 100.0
        frame.cx = frame.cy = 2.0
        frame.width = frame.height = 4
        frame._rgb_np = np.full((4, 4, 3), index, dtype=np.uint8)
        frame._rgb_bytes = None
        frame.rgb_path = None
        frames.append(frame)
    plan = causal_holdout_plan(
        frames,
        _args(streaming_initial_frames=1, streaming_eval_guard=0),
    )

    cameras = build_causal_holdout_cameras(plan, data_device="cpu")

    assert [camera._causal_holdout_record.index for camera in cameras] == [4]
    assert cameras[0].image_name == "000004"
    assert cameras[0].has_pending_image
    cameras[0].materialize_pending_image()
    assert tuple(cameras[0].original_image.shape) == (3, 4, 4)


def test_report_camera_rebuild_resolves_packet_identity_without_ordinal_alias():
    pytest.importorskip("torch")
    from eligsir.runtime.causal_evaluation import build_causal_holdout_cameras
    from eligsir.runtime.replay import streaming_frame_source_identity

    frames = []
    for epoch in (2, 9):
        frame = _frame(4, tx=float(epoch))
        frame.timestamp = 1.5
        frame.color_ts_ns = epoch * 10_000 + 4
        frame.source_ordinal = 4
        frame.source_id = "packet-source"
        frame.stream_epoch = epoch
        frame.fx = frame.fy = 100.0
        frame.cx = frame.cy = 2.0
        frame.width = frame.height = 4
        frame._rgb_np = np.zeros((4, 4, 3), dtype=np.uint8)
        frame._rgb_bytes = None
        frame.rgb_path = None
        frames.append(frame)

    cameras = build_causal_holdout_cameras(
        frames,
        data_device="cpu",
        report_source_id="packet-source",
        report_stream_epoch=9,
    )
    assert [camera._report_source_identity for camera in cameras] == [
        streaming_frame_source_identity(frames[0], source_id="packet-source", stream_epoch=9),
        streaming_frame_source_identity(frames[1], source_id="packet-source", stream_epoch=9),
    ]
    assert cameras[0]._report_source_identity != cameras[1]._report_source_identity
    assert all(not hasattr(camera, "_source_identity") for camera in cameras)


def test_periodic_holdout_evaluation_uses_a_bounded_evenly_spaced_subset():
    from eligsir.runtime.causal_evaluation import bounded_evaluation_cameras

    cameras = list(range(315))

    selected = bounded_evaluation_cameras(cameras, max_items=64)

    assert len(selected) == 64
    assert selected[0] == 0
    assert selected[-1] == 314
    assert selected == bounded_evaluation_cameras(cameras, max_items=64)
    assert bounded_evaluation_cameras(cameras, max_items=0) == []


def test_primary_and_frozen_holdout_curves_are_plotted_together(tmp_path):
    from eligsir.runtime.causal_evaluation import (
        append_primary_training_psnr,
        plot_paired_psnr,
    )

    append_primary_training_psnr(
        tmp_path, iteration=100, psnr_db=21.0, source_id="frame-7"
    )
    append_primary_training_psnr(
        tmp_path, iteration=200, psnr_db=24.0, source_id="frame-9"
    )
    (tmp_path / "psnr_history.jsonl").write_text(
        "\n".join(
            json.dumps({"iteration": iteration, "mean_test_psnr": value})
            for iteration, value in ((100, 18.0), (200, 20.0))
        )
        + "\n"
    )

    output = plot_paired_psnr(tmp_path, last_ingest_iteration=150)

    assert output == tmp_path / "psnr_paired_over_time.svg"
    assert output.is_file()
    rows = [
        json.loads(line)
        for line in (tmp_path / "primary_training_psnr.jsonl").read_text().splitlines()
    ]
    assert rows[-1] == {
        "iteration": 200,
        "primary_training_psnr_db": 24.0,
        "score_scope": "diagnostic_realized_supervision_grid_pre_update",
        "source_id": "frame-9",
    }


def test_primary_training_psnr_records_cross_batch_work_clocks(tmp_path):
    from eligsir.runtime.causal_evaluation import append_primary_training_psnr

    append_primary_training_psnr(
        tmp_path,
        iteration=7,
        psnr_db=23.5,
        source_id="frame-3",
        mapping_active_s=1.25,
        supervision_views=112,
    )

    row = json.loads((tmp_path / "primary_training_psnr.jsonl").read_text())
    assert row["mapping_active_s"] == 1.25
    assert row["supervision_views"] == 112


def test_causal_observer_uses_driver_clock_for_primary_psnr(tmp_path):
    from eligsir.runtime.causal_evaluation import CausalEvaluationObserver
    from eligsir.runtime.causal_replay import CausalReplayClock

    observer = object.__new__(CausalEvaluationObserver)
    observer.dataset = SimpleNamespace(model_path=tmp_path)
    observer._started_s = None
    observer._supervision_views = 0
    observer._source_complete = False
    observer._mapping_active_s = None
    observer._elapsed_s = None

    observer.observe_tick(
        SimpleNamespace(
            iteration=9,
            supervision_views=8,
            primary_training_psnr_db=22.0,
            primary_training_source_id="frame-4",
        ),
        CausalReplayClock(
            optimizer_steps=9,
            supervision_views=72,
            mapping_active_s=2.5,
            elapsed_s=3.0,
            source_frame_index=4,
        ),
    )

    row = json.loads((tmp_path / "primary_training_psnr.jsonl").read_text())
    assert row["mapping_active_s"] == 2.5
    assert row["supervision_views"] == 72
    assert observer._source_frame_index == 4

    # observe_tick must also append a causal_throughput.jsonl row for the
    # same tick -- exercised here without any __init__ having run (this
    # observer, like the one above, was constructed via object.__new__), so
    # the throughput sampler's getattr-defaulted gating/prev-state must not
    # require attributes __init__ would normally set.
    throughput_rows = [
        json.loads(line)
        for line in (tmp_path / "causal_throughput.jsonl").read_text().splitlines()
    ]
    assert len(throughput_rows) == 1
    assert throughput_rows[0]["elapsed_s"] == 3.0
    assert throughput_rows[0]["optimizer_steps"] == 9
    assert throughput_rows[0]["supervision_views"] == 72


def test_causal_observer_throughput_trace_is_gated_by_progress_interval(tmp_path):
    """A second tick inside the same causal_progress_interval_s window must
    not append a second row -- otherwise a long run would write one row per
    tick instead of a bounded periodic sample."""
    from eligsir.runtime.causal_evaluation import CausalEvaluationObserver
    from eligsir.runtime.causal_replay import CausalReplayClock

    observer = object.__new__(CausalEvaluationObserver)
    observer.dataset = SimpleNamespace(model_path=tmp_path)
    observer.args = SimpleNamespace(causal_progress_interval_s=10.0)
    observer._started_s = None
    observer._supervision_views = 0
    observer._source_complete = False
    observer._mapping_active_s = None
    observer._elapsed_s = None

    def _tick(iteration):
        return SimpleNamespace(
            iteration=iteration,
            supervision_views=0,
            primary_training_psnr_db=None,
            primary_training_source_id=None,
            primary_training_spatial=None,
        )

    observer.observe_tick(
        _tick(1),
        CausalReplayClock(
            optimizer_steps=10, supervision_views=40, mapping_active_s=1.0, elapsed_s=1.0
        ),
    )
    observer.observe_tick(
        _tick(2),
        CausalReplayClock(
            optimizer_steps=20, supervision_views=80, mapping_active_s=2.0, elapsed_s=2.0
        ),
    )
    trace = tmp_path / "causal_throughput.jsonl"
    assert len(trace.read_text().splitlines()) == 1

    # A third tick past the interval boundary (elapsed_s >= 10.0) must
    # append a second row, whose instantaneous rate is the delta since the
    # FIRST recorded row (elapsed_s=1.0), not since the un-recorded second tick.
    observer.observe_tick(
        _tick(3),
        CausalReplayClock(
            optimizer_steps=110, supervision_views=440, mapping_active_s=11.0, elapsed_s=11.0
        ),
    )
    rows = [json.loads(line) for line in trace.read_text().splitlines()]
    assert len(rows) == 2
    # delta_steps=(110-10)=100 over delta_active_s=(11.0-1.0)=10.0 -> 10/s.
    assert rows[1]["optimizer_steps_per_s"] == 10.0


def test_causal_evaluation_finalize_wires_throughput_and_tier_plots(
    tmp_path, monkeypatch
):
    """finalize() must produce the throughput/resolution-tier report
    artifacts alongside spatial_replay_focus, from real (non-monkeypatched)
    plot functions, given real trace files on disk."""
    from eligsir.runtime import causal_evaluation

    (tmp_path / "spatial_replay_focus.jsonl").write_text("")
    (tmp_path / "causal_throughput.jsonl").write_text(
        json.dumps(
            {
                "elapsed_s": 10.0,
                "mapping_active_s": 10.0,
                "optimizer_steps": 100,
                "supervision_views": 400,
                "optimizer_steps_per_s": 10.0,
                "supervision_views_per_s": 40.0,
                "optimizer_steps_per_s_cumulative": 10.0,
                "supervision_views_per_s_cumulative": 40.0,
            }
        )
        + "\n"
    )
    (tmp_path / "resolution_tier_usage.jsonl").write_text(
        "".join(
            json.dumps(
                {"iteration": i, "resolution_tier": "native", "transaction_class": None}
            )
            + "\n"
            for i in range(0, 200, 20)
        )
    )

    observer = object.__new__(causal_evaluation.CausalEvaluationObserver)
    observer.dataset = SimpleNamespace(model_path=str(tmp_path), source_path="/data/fr3")
    observer.plan = SimpleNamespace(holdout_ordinals=frozenset())
    observer._mapping_active_s = None
    observer._elapsed_s = None

    observer.finalize(
        SimpleNamespace(mapping_active_s=1.0, elapsed_s=2.0)
    )

    assert (tmp_path / "causal_throughput_over_time.svg").is_file()
    assert (tmp_path / "resolution_tier_over_time.svg").is_file()


def test_paired_psnr_work_plot_uses_active_time_and_supervised_views(tmp_path):
    from eligsir.runtime.causal_evaluation import (
        _load_paired_psnr_axes,
        plot_paired_psnr,
    )

    (tmp_path / "evaluation_ticks.jsonl").write_text(
        json.dumps(
            {
                "iteration": 5,
                "mean_test_psnr_db": 19.0,
                "accumulated_iteration_wall_s": 1.5,
                "supervision_views": 40,
            }
        )
        + "\n"
    )
    append = {
        "iteration": 5,
        "primary_training_psnr_db": 24.0,
        "mapping_active_s": 1.5,
        "supervision_views": 40,
    }
    (tmp_path / "primary_training_psnr.jsonl").write_text(
        json.dumps(append) + "\n"
    )

    axes = _load_paired_psnr_axes(tmp_path)
    assert axes["holdout"] == [
        {
            "iteration": 5.0,
            "mapping_active_s": 1.5,
            "supervision_views": 40.0,
            "psnr_db": 19.0,
        }
    ]
    assert axes["training"][0]["psnr_db"] == 24.0

    plot_paired_psnr(tmp_path, last_ingest_iteration=5)
    assert (tmp_path / "psnr_paired_over_work.svg").is_file()


def test_paired_psnr_work_plot_suppresses_invalid_wall_time_panel(
    tmp_path, monkeypatch
):
    from matplotlib.figure import Figure

    from eligsir.runtime.causal_evaluation import plot_paired_psnr

    (tmp_path / "evaluation_ticks.jsonl").write_text(
        json.dumps(
            {
                "iteration": 5,
                "mean_test_psnr_db": 19.0,
                "accumulated_iteration_wall_s": 1.5,
                "supervision_views": 40,
                "throughput_evidence_valid": False,
                "throughput_evidence_status": (
                    "invalid_mapping_holdout_observer_pause"
                ),
            }
        )
        + "\n"
    )
    (tmp_path / "primary_training_psnr.jsonl").write_text(
        json.dumps(
            {
                "iteration": 5,
                "primary_training_psnr_db": 24.0,
                "mapping_active_s": 1.5,
                "supervision_views": 40,
            }
        )
        + "\n"
    )
    figures = []
    real_savefig = Figure.savefig

    def capture_figure(figure, *args, **kwargs):
        figures.append(figure)
        return real_savefig(figure, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", capture_figure)

    plot_paired_psnr(tmp_path, last_ingest_iteration=5)

    work_figure = figures[-1]
    wall_time_axis, supervision_axis = work_figure.axes
    assert not wall_time_axis.lines
    assert any("UNAVAILABLE" in text.get_text() for text in wall_time_axis.texts)
    assert len(supervision_axis.lines) == 2
    assert (tmp_path / "psnr_paired_over_work.svg").is_file()


def test_mapping_training_psnr_publishes_partition_focus_without_holdout_feedback(tmp_path):
    from eligsir.contracts.runtime import SourceIdentity
    from eligsir.mapping.scheduler.partition_replay_focus import (
        PartitionReplayFocusController,
        TrainingPartitionPsnrAccumulator,
    )
    from eligsir.runtime.causal_evaluation import CausalEvaluationObserver

    sources = (
        SourceIdentity("camera", 0, 10, 10),
        SourceIdentity("camera", 0, 90, 90),
    )
    published = {}
    runtime = SimpleNamespace(
        replay_candidate_identities=lambda: sources,
        source_identity_for_fingerprint=lambda fingerprint: next(
            source for source in sources if source.fingerprint() == fingerprint
        ),
        publish_replay_focus_weights=lambda weights: published.update(weights),
    )
    observer = object.__new__(CausalEvaluationObserver)
    observer.runtime = runtime
    observer.dataset = SimpleNamespace(model_path=str(tmp_path))
    observer._source_complete = False
    observer._optimizer_updates = 1
    observer._mapping_active_s = 0.0
    observer._supervision_views = 0
    observer._partition_replay_focus = PartitionReplayFocusController(
        min_history=1,
        min_views_per_partition=2,
    )
    observer._partition_replay_focus_accumulator = TrainingPartitionPsnrAccumulator(
        n_source_frames=100,
        window_per_partition=4,
    )
    for source, psnr in (
        (sources[0], 15.0),
        (sources[0], 15.0),
        (sources[1], 25.0),
        (sources[1], 25.0),
    ):
        observer._observe_tick(
            SimpleNamespace(
                iteration=1000,
                primary_training_psnr_db=psnr,
                primary_training_source_id=source.fingerprint(),
                primary_training_spatial=None,
            )
        )

    assert published[sources[0]] > 1.0
    assert sources[1] not in published
    row = json.loads(
        (tmp_path / "partition_replay_focus.jsonl").read_text().splitlines()[-1]
    )
    assert row["iteration"] == 1000
    assert row["active"] is True
    assert row["replay_population"] == 2
    assert row["actuation_evidence"] == "training_primary_psnr"
    assert row["source_ordinal"] == sources[1].source_ordinal
    assert row["source_partition_id"] == 5
    assert row["partition_n"] == [2, 0, 0, 0, 0, 2]
    assert row["partition_supervision_n"] == [0, 0, 0, 0, 0, 0]
    assert row["accumulator_samples_total"] == 4
    assert row["controller_history_samples"] == 1


def test_completed_b4_views_publish_temporal_coverage_focus_before_psnr(tmp_path):
    """Tail replay must become preferred before sparse PSNR can observe it."""
    from eligsir.contracts.runtime import SourceIdentity
    from eligsir.mapping.scheduler.partition_replay_focus import (
        PartitionReplayFocusController,
        TrainingPartitionPsnrAccumulator,
    )
    from eligsir.runtime.causal_evaluation import CausalEvaluationObserver

    early = SourceIdentity("camera", 0, 10, 10)
    tail = SourceIdentity("camera", 0, 390, 390)
    published = []
    runtime = SimpleNamespace(
        replay_candidate_identities=lambda: (early, tail),
        publish_replay_focus_weights=lambda weights: published.append(dict(weights)),
    )
    observer = object.__new__(CausalEvaluationObserver)
    observer.runtime = runtime
    observer.dataset = SimpleNamespace(model_path=str(tmp_path))
    observer._source_complete = False
    observer._optimizer_updates = 1
    observer._mapping_active_s = 0.0
    observer._supervision_views = 0
    observer._mapping_holdout_curve_enabled = False
    observer._partition_replay_focus = PartitionReplayFocusController(min_history=3)
    observer._partition_replay_focus_accumulator = TrainingPartitionPsnrAccumulator(
        n_source_frames=400
    )
    observer._partition_replay_focus_updates = {}
    observer._spatial_replay_focus = None
    observer._spatial_replay_focus_updates = {}
    observer._next_partition_control_optimizer_update = None
    observer._next_partition_coverage_optimizer_update = 1
    observer._next_evaluation_optimizer_update = None

    observer._observe_tick(
        SimpleNamespace(
            iteration=1,
            primary_training_psnr_db=None,
            primary_training_spatial=None,
            completed_supervision_sources=(early, early, early, early),
        )
    )

    assert published[-1][tail] > 1.0
    assert early not in published[-1]


def test_partition_focus_caller_publishes_at_most_fixed_identity_bound(tmp_path):
    """The low-cadence evaluator must not hand full history to runtime control."""

    from eligsir.contracts.runtime import SourceIdentity
    from eligsir.runtime.causal_evaluation import (
        CausalEvaluationObserver,
        MAX_REPLAY_IDENTITY_FOCUS_UPDATES,
    )

    sources = tuple(
        SourceIdentity("camera", 0, ordinal, ordinal)
        for ordinal in range(5_000)
    )
    processed_counts = []
    published = {}

    def publish(weights):
        processed_counts.append(len(weights))
        assert len(weights) <= MAX_REPLAY_IDENTITY_FOCUS_UPDATES
        published.update(weights)

    runtime = SimpleNamespace(
        replay_candidate_identities=lambda: sources,
        source_identity_for_fingerprint=lambda _fingerprint: sources[0],
        publish_replay_focus_weights=publish,
    )
    decision = SimpleNamespace(
        active=True,
        reason="large-history-probe",
        history_samples=5_000,
        partition_weights=(),
        weights=tuple(
            (source, float(index + 2))
            for index, source in enumerate(sources)
        ),
    )
    observer = object.__new__(CausalEvaluationObserver)
    observer.runtime = runtime
    observer.dataset = SimpleNamespace(model_path=str(tmp_path))
    observer._partition_replay_focus = SimpleNamespace(
        update=lambda _record, _sources: decision
    )
    observer._partition_replay_focus_accumulator = SimpleNamespace(
        observe=lambda _source, _psnr: {"score_scope": "test"}
    )

    observed = observer._observe_training_partition_replay_focus(
        SimpleNamespace(
            iteration=10,
            primary_training_psnr_db=12.0,
            primary_training_source_id=sources[0].fingerprint(),
        )
    )
    assert observed is not None
    updates, eligible_sources = observed
    observer._replace_replay_focus_component(
        "_partition_replay_focus_updates", updates, eligible_sources
    )
    observer._publish_composed_replay_focus_weights()

    assert processed_counts == [MAX_REPLAY_IDENTITY_FOCUS_UPDATES]
    assert set(published) == set(sources[-MAX_REPLAY_IDENTITY_FOCUS_UPDATES:])


def test_adaptive_constellation_telemetry_uses_exact_post_transaction_work():
    from eligsir.contracts.control import SupervisionConstellation
    from eligsir.runtime.causal_evaluation import _tick_constellation_telemetry

    constellation = SupervisionConstellation(
        requested_batch_size=256,
        requested_camera_count=64,
        realized_batch_size=64,
        realized_camera_count=64,
        unique_eligible_count=64,
        fill_ratio=0.25,
        microbatch_count=1,
        mode="true_camera_batch",
        reason="eligible_underfill",
        hysteresis_state="stable_hold",
        cumulative_realized_supervised_views=120,
        cumulative_optimizer_updates=30,
    )
    telemetry = _tick_constellation_telemetry(
        SimpleNamespace(
            optimizer_steps=2,
            supervision_views=63,
            supervised_pixels=4096,
            unique_supervision_views=63,
            microbatch_sizes=(32, 31),
            cumulative_optimizer_updates=32,
            cumulative_supervision_views=183,
            work_plan=SimpleNamespace(supervision_constellation=constellation),
        )
    )

    assert telemetry["requested_batch_size"] == 256
    assert telemetry["realized_batch_size"] == 64
    assert telemetry["optimizer_updates"] == 2
    assert telemetry["supervised_views"] == 63
    assert telemetry["unique_supervision_views"] == 63
    assert telemetry["supervised_pixels"] == 4096
    assert telemetry["microbatch_sizes"] == [32, 31]
    assert telemetry["cumulative_realized_supervised_views"] == 183
    assert telemetry["cumulative_optimizer_updates"] == 32
    assert telemetry["reason"] == "eligible_underfill"


def test_causal_observer_uses_post_tick_work_clocks_not_iteration():
    from eligsir.runtime.causal_evaluation import CausalEvaluationObserver

    observer = object.__new__(CausalEvaluationObserver)
    observer._started_s = 1.0
    observer._optimizer_updates = 0
    observer._supervision_views = 0
    observer._source_complete = False
    observer.dataset = SimpleNamespace(model_path="unused")
    tick = SimpleNamespace(
        iteration=900,
        optimizer_steps=2,
        supervision_views=63,
        cumulative_optimizer_updates=32,
        cumulative_supervision_views=183,
        primary_training_psnr_db=None,
    )

    observer(tick)

    assert observer._optimizer_updates == 32
    assert observer._supervision_views == 183
    assert observer._latest_tick is tick
    assert observer._latest_tick_wall_clock_s is not None


def test_mapping_holdout_curve_large_jump_endpoint_and_refinement_are_phase_aware(
    tmp_path, monkeypatch
):
    """One large work-clock jump samples once, then endpoint and refinement persist."""
    pytest.importorskip("torch")
    import gaussian_renderer
    import eligsir.runtime.causal_evaluation as causal_evaluation
    from eligsir.runtime.causal_replay import CausalReplayClock
    import eligsir.utils.eval_tick
    import eligsir.utils.psnr_history

    psnr_contexts = []
    eval_rows = []

    def record_psnr(model_path, iteration, *_args, **kwargs):
        context = kwargs["metric_context"]
        psnr_contexts.append(context)
        utils.psnr_history.append_psnr_history(
            model_path,
            iteration,
            20.0,
            metric_context=context,
        )
        return object()

    monkeypatch.setattr(
        utils.psnr_history,
        "record_and_plot",
        record_psnr,
    )
    monkeypatch.setattr(
        utils.eval_tick,
        "record_eval_tick",
        lambda **kwargs: eval_rows.append(kwargs) or {},
    )
    monkeypatch.setattr(gaussian_renderer, "render", object())
    monkeypatch.setattr(
        causal_evaluation, "plot_paired_psnr", lambda *_args, **_kwargs: None
    )

    observer = object.__new__(causal_evaluation.CausalEvaluationObserver)
    observer._source_complete = False
    observer._mapping_holdout_curve_enabled = True
    observer._next_evaluation_optimizer_update = 100
    observer.interval = 100
    observer.dataset = SimpleNamespace(model_path=str(tmp_path), source_path="/data/fr3")
    observer.runtime = SimpleNamespace(
        gaussians=object(),
        pipe=object(),
        _background=object(),
        frames_seen=20,
        scene=SimpleNamespace(train_cameras=[object()] * 8),
        _admitted_observations_total=8,
    )
    observer.plan = SimpleNamespace(holdout_ordinals=frozenset({7}))
    observer.eval_cameras = [SimpleNamespace(image_name="bounded")]
    observer.test_cameras = [
        SimpleNamespace(image_name="bounded"),
        SimpleNamespace(image_name="not-rendered"),
    ]
    observer._mapping_holdout_curve_metadata = {
        "enabled": True,
        "cadence_clock": "cumulative_optimizer_updates",
        "cadence_interval": 100,
        "max_scored_holdout_views": 1,
        "observer_execution": "synchronous_mapping_thread_pause",
        "mapping_execution_paused": True,
        "throughput_evidence_valid": False,
        "throughput_evidence_status": "invalid_mapping_holdout_observer_pause",
    }
    observer._mapping_active_s = 0.0
    observer._elapsed_s = 0.0
    observer._started_s = None
    observer._last_evaluated_iteration = None
    observer._last_evaluated_phase_clock = None
    observer._next_iteration = None
    observer.last_ingest_iteration = None
    observer._ensure_cameras = lambda: None

    terminal_tick = SimpleNamespace(
        iteration=77,
        snapshot_id=44,
        control_snapshot=SimpleNamespace(control_tick_epoch=12),
        work_plan=None,
        primary_training_psnr_db=None,
        primary_training_spatial=None,
    )
    observer.observe_tick(
        terminal_tick,
        CausalReplayClock(
            optimizer_steps=350,
            supervision_views=1_400,
            mapping_active_s=3.5,
            elapsed_s=4.0,
        ),
    )
    # No catch-up burst for the crossed 100/200/300 thresholds.
    assert [row["evaluation_phase"] for row in eval_rows] == ["mapping"]

    # Same raw iteration still needs one durable semantic endpoint row.
    observer.source_complete(terminal_tick)
    observer.source_complete(terminal_tick)
    assert [row["evaluation_phase"] for row in eval_rows] == [
        "mapping",
        "mapping_endpoint",
    ]

    refinement_tick = SimpleNamespace(
        iteration=78,
        snapshot_id=45,
        control_snapshot=SimpleNamespace(control_tick_epoch=13),
        work_plan=None,
        primary_training_psnr_db=None,
        primary_training_spatial=None,
    )
    observer.observe_tick(
        refinement_tick,
        CausalReplayClock(
            optimizer_steps=450,
            supervision_views=1_800,
            mapping_active_s=4.5,
            elapsed_s=5.0,
        ),
    )
    assert [row["evaluation_phase"] for row in eval_rows] == [
        "mapping",
        "mapping_endpoint",
        "refinement",
    ]
    assert [ctx["evaluation_phase"] for ctx in psnr_contexts] == [
        "mapping",
        "mapping_endpoint",
        "refinement",
    ]
    durable_rows = [
        __import__("json").loads(line)
        for line in (tmp_path / "psnr_history.jsonl").read_text().splitlines()
    ]
    assert sum(
        row["evaluation_phase"] == "mapping_endpoint" for row in durable_rows
    ) == 1
    assert all(row["test_cams"] == observer.eval_cameras for row in eval_rows)


def test_mapping_holdout_curve_default_remains_endpoint_only():
    from eligsir.runtime.causal_evaluation import CausalEvaluationObserver
    from eligsir.runtime.causal_replay import CausalReplayClock

    observer = object.__new__(CausalEvaluationObserver)
    observer._source_complete = False
    observer._mapping_holdout_curve_enabled = False
    observer._next_evaluation_optimizer_update = None
    observer.interval = 100
    observer.dataset = SimpleNamespace(model_path="unused")
    observer._mapping_active_s = 0.0
    observer._elapsed_s = 0.0
    observer.last_ingest_iteration = None
    observed = []
    observer._evaluate = lambda tick, **_kwargs: observed.append(int(tick.iteration))
    observer._ensure_cameras = lambda: None

    tick = SimpleNamespace(
        iteration=9_999,
        primary_training_psnr_db=None,
        primary_training_spatial=None,
    )
    observer.observe_tick(
        tick,
        CausalReplayClock(
            optimizer_steps=1_000,
            supervision_views=4_000,
            mapping_active_s=10.0,
            elapsed_s=10.0,
        ),
    )
    assert observed == []

    observer.source_complete(tick)
    assert observed == [9_999]


def test_mapping_holdout_curve_metadata_invalidates_throughput_evidence():
    from eligsir.runtime.causal_evaluation import mapping_holdout_curve_metadata

    metadata = mapping_holdout_curve_metadata(
        _args(
            streaming_mapping_holdout_curve=True,
            streaming_psnr_chart_interval=250,
            streaming_report_test_metrics_max=32,
        )
    )

    assert metadata == {
        "enabled": True,
        "async_mode": False,
        "async_requested": False,
        "cadence_clock": "cumulative_optimizer_updates",
        "cadence_interval": 250,
        "max_scored_holdout_views": 32,
        "observer_execution": "synchronous_mapping_thread_pause",
        "mapping_execution_paused": True,
        "throughput_evidence_valid": False,
        "throughput_evidence_status": "invalid_mapping_holdout_observer_pause",
        "async_snapshot_copy_stall_threshold_ms": 5.0,
    }


def test_holdout_rasterizer_context_uses_completed_async_events_only():
    """No completed event means an explicit null, never a made-up duration."""
    from eligsir.runtime.causal_evaluation import _holdout_rasterizer_context

    class _Registry:
        def __init__(self, stats):
            self.stats = stats
            self.resolves = 0

        def resolve_pending(self):
            self.resolves += 1
            return 1

        def snapshot(self):
            return self.stats

    completed = _Registry(
        {"causal_holdout.rasterizer": {"count": 2, "total_ms": 7.5}}
    )
    assert _holdout_rasterizer_context(completed) == {
        "rasterizer_gpu_ms": 7.5,
        "rasterizer_timing_status": "available_async_gpu_events_total_tick",
    }
    assert completed.resolves == 1

    unavailable = _Registry({})
    assert _holdout_rasterizer_context(unavailable) == {
        "rasterizer_gpu_ms": None,
        "rasterizer_timing_status": "unavailable_stage_gpu_events",
    }


def test_holdout_rasterizer_context_resolves_deferred_events_at_evaluation_boundary():
    """A completed evaluation must flush its pending rasterizer event once."""
    from eligsir.runtime.causal_evaluation import _holdout_rasterizer_context

    class _Registry:
        def __init__(self):
            self.resolves = 0

        def resolve_pending(self):
            self.resolves += 1
            return 1

        def snapshot(self):
            return {"causal_holdout.rasterizer": {"count": 1, "total_ms": 3.25}}

    registry = _Registry()
    assert _holdout_rasterizer_context(registry) == {
        "rasterizer_gpu_ms": 3.25,
        "rasterizer_timing_status": "available_async_gpu_events_total_tick",
    }
    assert registry.resolves == 1


def test_causal_final_report_serializes_device_global_gtt_block(monkeypatch, tmp_path):
    """Causal FR3 reports retain GTT's shared-device semantics and nulls."""
    pytest.importorskip("torch")
    import gaussian_renderer
    import eligsir.runtime.causal_evaluation as causal_evaluation
    import eligsir.utils.comparison_report
    import eligsir.utils.execution_semantics
    import eligsir.utils.experiment_origin
    import eligsir.utils.partition_eval
    import eligsir.runtime.transfers

    monkeypatch.setenv(
        "SCENE_SOURCE",
        "/data/datasets/TUM/"
        "rgbd_dataset_freiburg3_long_office_household",
    )
    captured = {}
    monkeypatch.setattr(causal_evaluation, "plot_spatial_replay_focus", lambda *_: None)
    monkeypatch.setattr(
        causal_evaluation,
        "_terminal_evaluation_tick_clock_snapshot",
        lambda _observer: {"optimizer_step": 7},
    )
    monkeypatch.setattr(causal_evaluation, "mapping_clocks_for_report", lambda clocks, _meta: clocks)
    monkeypatch.setattr(gaussian_renderer, "render", object())
    monkeypatch.setattr(utils.comparison_report, "write_post_training_report", lambda **kwargs: captured.update(kwargs) or kwargs)
    monkeypatch.setattr(utils.comparison_report, "determinism_from_args", lambda _args: {})
    monkeypatch.setattr(utils.comparison_report, "git_provenance_from_args", lambda _args: {})
    monkeypatch.setattr(utils.execution_semantics, "build_execution_semantics_block", lambda _args: {})
    monkeypatch.setattr(utils.experiment_origin, "experiment_origin_from_args", lambda _args: {})
    monkeypatch.setattr(utils.partition_eval, "registry_n_source_frames", lambda _path: 10)
    monkeypatch.setattr(eligsir.runtime.transfers, "transfer_pool_report_block", lambda _owner: None)

    class _Tracker:
        def __init__(self):
            self.samples = []

        def snapshot(self, **kwargs):
            self.samples.append(kwargs)

        def report_block(self):
            return {
                "enabled": True,
                "available": True,
                "peak_gtt_used_bytes": None,
                "gtt_memory_semantics": (
                    "device_global_amd_drm_gtt_shared_system_memory_not_per_process"
                ),
                "final": {
                    "gtt_used_bytes": None,
                    "gtt_total_bytes": None,
                    "gtt_available": False,
                    "gtt_memory_semantics": (
                        "device_global_amd_drm_gtt_shared_system_memory_not_per_process"
                    ),
                },
            }

    observer = object.__new__(causal_evaluation.CausalEvaluationObserver)
    observer.dataset = SimpleNamespace(model_path=str(tmp_path), source_path="/data/fr3")
    observer._source_records = ()
    observer._packet_source_id = None
    observer._packet_stream_epoch = 0
    observer.args = SimpleNamespace(
        streaming_report_train_metrics_max=8,
        streaming_report_test_metrics_max=8,
        streaming_report_trajectory_max_frames=8,
        baseline_run=None,
        run_intent={},
        experiment_context={},
        experiment_validity_class=None,
    )
    observer.plan = SimpleNamespace(holdout_ordinals=frozenset({1}))
    observer.runtime = SimpleNamespace(
        _services=SimpleNamespace(four_clocks=SimpleNamespace(summary=lambda: {})),
        frames_seen=9,
        bootstrap_frames=1,
        _admitted_observations_total=8,
        iteration=7,
        gaussians=object(),
        scene=SimpleNamespace(train_cameras=[object()] * 8),
        pipe=object(),
        _background=object(),
        _transfer_owner=None,
    )
    observer.eval_cameras = [object()]
    observer.last_ingest_iteration = 7
    observer._mapping_active_s = 1.0
    observer._elapsed_s = 2.0
    observer._mapping_holdout_curve_metadata = {
        "throughput_evidence_valid": True,
        "throughput_evidence_status": "valid_no_mapping_holdout_observer_pause",
        "cadence_clock": "cumulative_optimizer_updates",
    }
    observer._partition_control_runtime = None
    observer._partition_control_roles = None
    observer._ensure_cameras = lambda: None
    observer._report_eval_metadata = lambda: {}
    observer._experiment_validity_source_indices = lambda: ()
    observer._gpu_memory_tracker = _Tracker()

    observer.finalize(
        SimpleNamespace(
            mapping_active_s=1.0,
            elapsed_s=2.0,
            source_offered=10,
            submitted=10,
            holdout_reserved=1,
            guard_withheld=0,
            source_ingress=None,
            coverage_ledger={
                "coverage_ledger_version": 2,
                "resolved_source_frames": 10,
                "raw_registry_source_frames": None,
                "source_identity": "/data/fr3",
                "effective_stride": 1,
                "source_denominator_provenance": "causal_source_catalog",
                "source_frames_ingested_total": 10,
                "bootstrap_frames_consumed": 1,
                "reserved_holdout_frames": 1,
                "reserved_guard_frames": 0,
                "rejected_invalid_frames": 0,
                "eligible_mapping_frames": 8,
                "frames_admitted": 2,
                "mapping_skipped_frames": 6,
                "unprocessed_frames": 0,
                "causal_source_accounting": {
                    "mode": "terminal_disposition",
                    "unique_source_count": 10,
                },
            },
        )
    )

    assert captured["coverage_counters"]["frames_admitted"] == 2
    assert captured["frames_admitted_override"] == 2
    assert captured["gpu_memory"]["gtt_memory_semantics"] == (
        "device_global_amd_drm_gtt_shared_system_memory_not_per_process"
    )
    assert captured["gpu_memory"]["final"]["gtt_used_bytes"] is None
    assert captured["gpu_memory"]["final"]["gtt_total_bytes"] is None
    assert captured["allocator_phase_tracker"] is observer._gpu_memory_tracker
    assert captured["scene_source"] == (
        "/data/datasets/TUM/"
        "rgbd_dataset_freiburg3_long_office_household"
    )
    captured.clear()
    observer._gpu_memory_tracker.samples.clear()
    observer.finalize(
        SimpleNamespace(
            mapping_active_s=1.0,
            elapsed_s=2.0,
            source_offered=10,
            submitted=10,
            holdout_reserved=1,
            guard_withheld=0,
            source_ingress=None,
        )
    )
    assert captured["coverage_counters"] is None
    assert captured["frames_admitted_override"] is None
    captured.clear()
    observer._gpu_memory_tracker.samples.clear()
    partial_ledger = {
        "coverage_ledger_version": 2,
        "resolved_source_frames": 10,
        "raw_registry_source_frames": None,
        "source_identity": "/data/fr3",
        "effective_stride": 1,
        "source_denominator_provenance": "causal_source_catalog",
        "source_frames_ingested_total": 9,
        "bootstrap_frames_consumed": 1,
        "reserved_holdout_frames": 1,
        "reserved_guard_frames": 0,
        "rejected_invalid_frames": 0,
        "eligible_mapping_frames": 7,
        "frames_admitted": 2,
        "mapping_skipped_frames": 5,
        "unprocessed_frames": 1,
        "causal_source_accounting": {
            "mode": "terminal_disposition",
            "unique_source_count": 9,
        },
    }
    observer.finalize(
        SimpleNamespace(
            mapping_active_s=1.0,
            elapsed_s=2.0,
            source_offered=10,
            submitted=9,
            holdout_reserved=1,
            guard_withheld=0,
            source_ingress={"transport_dropped": 1},
            coverage_ledger=partial_ledger,
        )
    )
    assert captured["coverage_counters"] is None
    assert captured["frames_admitted_override"] is None
    assert observer._gpu_memory_tracker.samples == [
        {
            "iteration": 7,
            "frames_ingested": 9,
            "gaussians": observer.runtime.gaussians,
            "streaming_scene": observer.runtime.scene,
            "observation_db": None,
        }
    ]


def test_gpu_span_registry_poll_completed_never_waits_for_an_event():
    """The holdout path can resolve ready events without a GPU-wide sync."""
    from eligsir.runtime.profiling import events

    class _Start:
        def elapsed_time(self, _end):
            return 3.25

    class _End:
        ready = False

        def query(self):
            return self.ready

    prior = events.PROFILING_ENABLED
    events.set_profiling_enabled(True)
    try:
        registry = events.GpuSpanRegistry()
        end = _End()
        registry.record_pending("causal_holdout.rasterizer", _Start(), end)
        assert registry.poll_completed() == 0
        assert registry.snapshot() == {}
        end.ready = True
        assert registry.poll_completed() == 1
        assert registry.snapshot()["causal_holdout.rasterizer"]["total_ms"] == 3.25
    finally:
        events.set_profiling_enabled(prior)


def test_mapping_holdout_curve_async_mode_defers_throughput_verdict():
    """Async mode (issue #1314) never blocks mapping and never hardcodes
    throughput_evidence_valid -- the real verdict is filled in later, once
    measured stall data exists (see _resolve_async_throughput_evidence)."""
    from eligsir.runtime.causal_evaluation import mapping_holdout_curve_metadata

    metadata = mapping_holdout_curve_metadata(
        _args(
            streaming_mapping_holdout_curve=True,
            streaming_mapping_holdout_curve_async=True,
            streaming_psnr_chart_interval=250,
            streaming_report_test_metrics_max=32,
        )
    )

    assert metadata["async_mode"] is True
    assert metadata["observer_execution"] == "async_snapshot_stream"
    assert metadata["mapping_execution_paused"] is False
    # Provisional until real measurements exist -- never a hardcoded False.
    assert metadata["throughput_evidence_valid"] is True
    assert metadata["throughput_evidence_status"] == (
        "pending_measured_async_snapshot_stall"
    )


def test_mapping_holdout_curve_async_requires_curve_enabled():
    from eligsir.runtime.causal_evaluation import mapping_holdout_curve_metadata

    with pytest.raises(ValueError, match="streaming_mapping_holdout_curve_async"):
        mapping_holdout_curve_metadata(
            _args(
                streaming_mapping_holdout_curve=False,
                streaming_mapping_holdout_curve_async=True,
            )
        )


def test_mapping_holdout_curve_async_yields_to_partition_control():
    """Issue #1314 scopes async mode to the holdout-curve render only --
    partition-control focus keeps the (unaffected, pre-existing) synchronous
    path when both are requested together."""
    from eligsir.runtime.causal_evaluation import mapping_holdout_curve_metadata

    metadata = mapping_holdout_curve_metadata(
        _args(
            streaming_mapping_holdout_curve=True,
            streaming_mapping_holdout_curve_async=True,
            streaming_psnr_chart_interval=250,
            streaming_report_test_metrics_max=32,
            streaming_partition_focus_weight=1.0,
        )
    )

    assert metadata["async_mode"] is False
    assert metadata["mapping_execution_paused"] is True
    assert metadata["observer_execution"] == "synchronous_mapping_thread_pause"


def test_mapping_holdout_curve_misconfiguration_fails_closed():
    from eligsir.runtime.causal_evaluation import mapping_holdout_curve_metadata

    with pytest.raises(ValueError, match="streaming_psnr_chart_interval > 0"):
        mapping_holdout_curve_metadata(
            _args(
                streaming_mapping_holdout_curve=True,
                streaming_psnr_chart_interval=0,
                streaming_report_test_metrics_max=32,
            )
        )
    with pytest.raises(
        ValueError, match="streaming_report_test_metrics_max > 0"
    ):
        mapping_holdout_curve_metadata(
            _args(
                streaming_mapping_holdout_curve=True,
                streaming_psnr_chart_interval=100,
                streaming_report_test_metrics_max=0,
            )
        )


def test_mapping_holdout_curve_report_metadata_is_machine_readable():
    from eligsir.runtime.causal_evaluation import CausalEvaluationObserver

    observer = object.__new__(CausalEvaluationObserver)
    observer.plan = SimpleNamespace(eval_metadata={"holdout_hash": "abc"})
    observer._mapping_holdout_curve_metadata = {
        "enabled": True,
        "cadence_clock": "cumulative_optimizer_updates",
        "cadence_interval": 250,
        "max_scored_holdout_views": 32,
        "observer_execution": "synchronous_mapping_thread_pause",
        "mapping_execution_paused": True,
        "throughput_evidence_valid": False,
        "throughput_evidence_status": (
            "invalid_mapping_holdout_observer_pause"
        ),
    }

    metadata = observer._report_eval_metadata()

    assert metadata["holdout_hash"] == "abc"
    assert metadata["mapping_holdout_curve"]["throughput_evidence_valid"] is False
    assert metadata["mapping_holdout_curve"]["mapping_execution_paused"] is True


def test_mapping_holdout_curve_invalidates_report_mapping_clocks():
    from eligsir.runtime.causal_evaluation import mapping_clocks_for_report

    clocks = mapping_clocks_for_report(
        {"source_frame_rate_hz": 30.0, "map_commit_rate_hz": 8.0},
        {
            "throughput_evidence_valid": False,
            "throughput_evidence_status": "invalid_mapping_holdout_observer_pause",
        },
    )

    assert clocks["throughput_evidence_valid"] is False
    assert clocks["throughput_evidence_status"] == (
        "invalid_mapping_holdout_observer_pause"
    )


def test_mapping_holdout_curve_flag_is_explicit_and_default_off():
    from argparse import ArgumentParser

    from arguments import StreamingParams

    parser = ArgumentParser()
    StreamingParams(parser)

    assert parser.parse_args([]).streaming_mapping_holdout_curve is False
    assert parser.parse_args(
        ["--streaming_mapping_holdout_curve"]
    ).streaming_mapping_holdout_curve is True


def test_mapping_holdout_curve_async_flag_is_explicit_and_default_off():
    from argparse import ArgumentParser

    from arguments import StreamingParams

    parser = ArgumentParser()
    StreamingParams(parser)

    defaults = parser.parse_args([])
    assert defaults.streaming_mapping_holdout_curve_async is False
    assert defaults.streaming_mapping_holdout_curve_async_stall_threshold_ms == 5.0
    assert parser.parse_args(
        ["--streaming_mapping_holdout_curve_async"]
    ).streaming_mapping_holdout_curve_async is True


def test_evaluation_tick_joins_post_work_clocks_to_immutable_control_keys(
    tmp_path, monkeypatch
):
    pytest.importorskip("torch")
    import gaussian_renderer
    import eligsir.runtime.causal_evaluation as causal_evaluation
    import eligsir.utils.eval_tick
    import eligsir.utils.psnr_history
    from eligsir.contracts.runtime import SourceIdentity

    monkeypatch.setenv(
        "SCENE_SOURCE",
        "/data/datasets/TUM/"
        "rgbd_dataset_freiburg3_long_office_household",
    )
    captured = {}
    captured_psnr = {}
    monkeypatch.setattr(
        utils.psnr_history,
        "record_and_plot",
        lambda *_args, **kwargs: captured_psnr.update(kwargs) or object(),
    )
    monkeypatch.setattr(
        utils.eval_tick,
        "record_eval_tick",
        lambda **kwargs: captured.update(kwargs) or {},
    )
    monkeypatch.setattr(gaussian_renderer, "render", object())
    monkeypatch.setattr(causal_evaluation, "plot_paired_psnr", lambda *_args, **_kwargs: None)

    observer = object.__new__(causal_evaluation.CausalEvaluationObserver)
    observer.dataset = SimpleNamespace(
        model_path=str(tmp_path), source_path="/data/fr3"
    )
    observer.runtime = SimpleNamespace(
        gaussians=object(),
        pipe=object(),
        _background=object(),
        frames_seen=20,
        scene=SimpleNamespace(train_cameras=[object()] * 8),
        _admitted_observations_total=8,
        successful_supervision_uses_by_identity=lambda: {
            SourceIdentity("packet", 0, 3, 3): 2,
            SourceIdentity("packet", 0, 4, 4): 1,
            SourceIdentity("packet", 0, 5, 5): 0,
        },
    )
    observer.plan = SimpleNamespace(holdout_ordinals=frozenset({7}))
    observer.eval_cameras = [object()]
    observer.test_cameras = [object()]
    observer._optimizer_updates = 32
    observer._supervision_views = 183
    observer._source_complete = False
    observer._mapping_holdout_curve_metadata = {
        "enabled": True,
        "cadence_clock": "cumulative_optimizer_updates",
        "cadence_interval": 100,
        "max_scored_holdout_views": 1,
        "observer_execution": "synchronous_mapping_thread_pause",
        "mapping_execution_paused": True,
        "throughput_evidence_valid": False,
        "throughput_evidence_status": (
            "invalid_mapping_holdout_observer_pause"
        ),
    }
    observer._started_s = None
    observer._last_evaluated_iteration = None
    observer._next_iteration = None
    observer.last_ingest_iteration = 700
    observer.interval = 100
    tick = SimpleNamespace(
        iteration=900,
        snapshot_id=44,
        control_snapshot=SimpleNamespace(control_tick_epoch=12),
        work_plan=None,
    )

    observer._evaluate(tick)

    assert captured["iteration"] == 900
    assert captured["optimizer_step"] == 32
    assert captured["supervision_views"] == 183
    assert captured["frames_trained"] == 2
    assert captured["mapping_snapshot_id"] == 44
    assert captured["control_tick_epoch"] == 12
    assert captured["scene_source"] == (
        "/data/datasets/TUM/"
        "rgbd_dataset_freiburg3_long_office_household"
    )
    assert captured_psnr["metric_context"]["evaluation_phase"] == "mapping"
    assert captured_psnr["metric_context"]["mapping_execution_paused"] is True
    assert captured_psnr["metric_context"]["throughput_evidence_valid"] is False


def test_async_evaluations_keep_trigger_time_source_frontiers_when_completed_in_reverse(
    tmp_path, monkeypatch
):
    import gaussian_renderer
    import eligsir.runtime.causal_evaluation as causal_evaluation
    import eligsir.utils.eval_tick
    import eligsir.utils.partition_eval
    import eligsir.utils.psnr_history
    import torch
    from eligsir.utils.psnr_history import PsnrTickResult

    completions = []

    class _DeferredEvaluator:
        def submit(self, *, gaussians, evaluate_fn):
            del gaussians
            completions.append(evaluate_fn)

    psnr_tick = PsnrTickResult(
        cams=[SimpleNamespace(uid=900, image_name="000900")],
        psnr_tensor=torch.tensor([20.0]),
        mean_psnr=20.0,
    )
    monkeypatch.setattr(
        utils.psnr_history, "record_and_plot", lambda *_args, **_kwargs: psnr_tick
    )
    monkeypatch.setattr(utils.eval_tick, "registry_n_source_frames", lambda _s: 1000)
    monkeypatch.setattr(utils.partition_eval, "registry_n_source_frames", lambda _s: 1000)
    monkeypatch.setattr(
        utils.eval_tick, "plot_partition_psnr_over_time", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(gaussian_renderer, "render", object())
    monkeypatch.setattr(
        causal_evaluation, "plot_paired_psnr", lambda *_args, **_kwargs: None
    )

    observer = object.__new__(causal_evaluation.CausalEvaluationObserver)
    observer.dataset = SimpleNamespace(model_path=str(tmp_path), source_path="/data/fr3")
    observer.runtime = SimpleNamespace(
        gaussians=object(),
        pipe=object(),
        _background=object(),
        frames_seen=20,
        scene=SimpleNamespace(train_cameras=[]),
        _admitted_observations_total=8,
    )
    observer.plan = SimpleNamespace(holdout_ordinals=frozenset({900}))
    observer.eval_cameras = [object()]
    observer.test_cameras = [object()]
    observer._source_complete = False
    observer._mapping_holdout_curve_metadata = {
        "enabled": True,
        "async_mode": True,
        "observer_execution": "async_frozen_snapshot",
        "mapping_execution_paused": False,
        "throughput_evidence_valid": True,
        "throughput_evidence_status": "valid",
    }
    observer._started_s = None
    observer._last_evaluated_phase_clock = None
    observer.last_ingest_iteration = 50
    observer.interval = 1
    observer._async_holdout_evaluator = _DeferredEvaluator()

    observer._optimizer_updates = 10
    observer._supervision_views = 40
    observer._mapping_active_s = 1.0
    observer._elapsed_s = 2.0
    observer._source_frame_index = 100
    observer._evaluate(
        SimpleNamespace(
            iteration=100,
            snapshot_id=1,
            control_snapshot=SimpleNamespace(control_tick_epoch=1),
            work_plan=None,
        )
    )

    observer._optimizer_updates = 20
    observer._supervision_views = 80
    observer._mapping_active_s = 3.0
    observer._elapsed_s = 4.0
    observer._source_frame_index = 400
    observer._evaluate(
        SimpleNamespace(
            iteration=200,
            snapshot_id=2,
            control_snapshot=SimpleNamespace(control_tick_epoch=2),
            work_plan=None,
        )
    )

    observer._source_frame_index = 900
    completions[1](object())
    completions[0](object())

    rows = utils.eval_tick.read_eval_ticks(str(tmp_path))
    assert [(row["iteration"], row["source_frame_index"]) for row in rows] == [
        (100, 100),
        (200, 400),
    ]
    assert [row["source_progress"] for row in rows] == [
        pytest.approx(0.1),
        pytest.approx(0.4),
    ]
    assert all(row["source_frame_index"] != 900 for row in rows)


def test_terminal_report_clock_snapshot_uses_latest_exact_tick_not_iteration():
    from eligsir.contracts.control import SupervisionConstellation
    from eligsir.contracts.runtime import SourceIdentity
    from eligsir.runtime.causal_evaluation import (
        CausalEvaluationObserver,
        _terminal_evaluation_tick_clock_snapshot,
    )

    constellation = SupervisionConstellation(
        requested_batch_size=256,
        requested_camera_count=4,
        realized_batch_size=7,
        realized_camera_count=4,
        unique_eligible_count=7,
        fill_ratio=7 / 256,
        microbatch_count=2,
        mode="true_camera_microbatch_accumulation",
        reason="eligible_underfill",
        hysteresis_state="stable_hold",
        cumulative_realized_supervised_views=176,
        cumulative_optimizer_updates=31,
    )
    tick = SimpleNamespace(
        iteration=900,
        optimizer_steps=1,
        supervision_views=7,
        supervised_pixels=28672,
        unique_supervision_views=7,
        microbatch_sizes=(4, 3),
        cumulative_optimizer_updates=32,
        cumulative_supervision_views=183,
        snapshot_id=44,
        control_snapshot=SimpleNamespace(control_tick_epoch=12),
        work_plan=SimpleNamespace(supervision_constellation=constellation),
    )
    observer = object.__new__(CausalEvaluationObserver)
    observer._latest_tick = tick
    observer._latest_tick_wall_clock_s = 123.5
    observer._elapsed_s = 123.5
    observer._mapping_active_s = 87.25
    observer._source_frame_index = 400
    observer._optimizer_updates = 32
    observer._supervision_views = 183
    observer.last_ingest_iteration = 700
    observer.plan = SimpleNamespace(holdout_ordinals=frozenset({90}))
    observer.runtime = SimpleNamespace(
        frames_seen=91,
        scene=SimpleNamespace(train_cameras=[object()] * 73),
        _admitted_observations_total=44,
        successful_supervision_uses_by_identity=lambda: {
            SourceIdentity("packet", 0, 3, 3): 2,
            SourceIdentity("packet", 0, 4, 4): 1,
            SourceIdentity("packet", 0, 5, 5): 0,
        },
    )

    snapshot = _terminal_evaluation_tick_clock_snapshot(observer)

    assert snapshot["optimizer_step"] == 32
    assert snapshot["optimizer_step"] != tick.iteration
    assert snapshot["supervision_views"] == 183
    assert snapshot["frames_trained"] == 2
    assert snapshot["mapping_snapshot_id"] == 44
    assert snapshot["control_tick_epoch"] == 12
    assert snapshot["wall_clock_s"] == 123.5
    assert snapshot["accumulated_iteration_wall_s"] == 87.25
    assert snapshot["supervision_constellation"]["microbatch_sizes"] == [4, 3]
    assert snapshot["supervision_constellation"]["unique_supervision_views"] == 7
    assert snapshot["supervision_constellation"]["supervised_pixels"] == 28672
    assert snapshot["source_frame_index"] == 400

    observer.runtime.successful_supervision_uses_by_identity = lambda: {}
    assert _terminal_evaluation_tick_clock_snapshot(observer)["frames_trained"] == 0
    del observer.runtime.successful_supervision_uses_by_identity
    assert _terminal_evaluation_tick_clock_snapshot(observer)["frames_trained"] == 73
    del observer._source_frame_index
    assert _terminal_evaluation_tick_clock_snapshot(observer)["source_frame_index"] is None


def test_terminal_report_clock_snapshot_fails_closed_without_exact_tick():
    from eligsir.runtime.causal_evaluation import (
        CausalEvaluationObserver,
        _terminal_evaluation_tick_clock_snapshot,
    )

    observer = object.__new__(CausalEvaluationObserver)
    observer._latest_tick = None

    with pytest.raises(RuntimeError, match="exact terminal mapping tick"):
        _terminal_evaluation_tick_clock_snapshot(observer)


def test_causal_validity_source_indices_cover_commit_and_defer_catalog():
    """Bounded resident cameras must not define causal trajectory validity."""
    from eligsir.contracts.runtime import SourceIdentity
    from eligsir.runtime.causal_evaluation import CausalEvaluationObserver

    retained = tuple(
        SourceIdentity("camera", 0, ordinal, ordinal * 1_000_000)
        for ordinal in (1, 2, 7, 8, 15)
    )
    observer = object.__new__(CausalEvaluationObserver)
    observer.runtime = SimpleNamespace(
        replay_candidate_identities=lambda: retained,
        scene=SimpleNamespace(train_cameras=[object()]),
    )

    assert observer._experiment_validity_source_indices() == (1, 2, 7, 8, 15)


def test_online_training_spatial_psnr_publishes_world_cell_focus(tmp_path):
    from eligsir.contracts.runtime import SourceIdentity
    from eligsir.mapping.scheduler.partition_replay_focus import (
        SpatialCellQuality,
        SpatialMicrocellQualitySample,
        SpatialMicrocellReplayFocusController,
    )
    from eligsir.runtime.causal_evaluation import CausalEvaluationObserver

    sources = (
        SourceIdentity("camera", 0, 10, 10),
        SourceIdentity("camera", 0, 90, 90),
    )
    published = {}
    growth_evidence = []
    runtime = SimpleNamespace(
        replay_candidate_identities=lambda: sources,
        publish_replay_focus_weights=lambda weights: published.update(weights),
        publish_growth_quality_evidence=lambda **evidence: growth_evidence.append(
            evidence
        ),
    )
    observer = object.__new__(CausalEvaluationObserver)
    observer.runtime = runtime
    observer.dataset = SimpleNamespace(model_path=str(tmp_path))
    observer._source_complete = False
    observer._spatial_replay_focus = SpatialMicrocellReplayFocusController(
        min_history=1,
        minimum_cell_samples=1,
    )

    for iteration, source, key, psnr, population in (
        (1000, sources[0], 101, 15.0, 1_234),
        (1025, sources[1], 202, 25.0, 2_345),
    ):
        observer._observe_tick(
            SimpleNamespace(
                iteration=iteration,
                primary_training_gaussian_count=population,
                primary_training_psnr_db=None,
                primary_training_spatial=SpatialMicrocellQualitySample(
                    source=source,
                    cell_size_m=0.5,
                    cells=(
                        SpatialCellQuality(
                            cell_key=key, psnr_db=psnr, n_samples=300
                        ),
                    ),
                ),
            )
        )

    assert published[sources[0]] > 1.0
    assert sources[1] not in published
    row = json.loads(
        (tmp_path / "spatial_replay_focus.jsonl").read_text().splitlines()[-1]
    )
    assert row["iteration"] == 1025
    assert row["active"] is True
    assert row["replay_population"] == 2
    assert row["actuation_evidence"] == "training_primary_spatial_psnr"
    assert row["score_scope"] == "training_primary_spatial_pre_update"
    assert {cell["cell_key"] for cell in row["cells"]} == {101, 202}
    assert all(len(cell["center_xyz_m"]) == 3 for cell in row["cells"])
    assert growth_evidence[-1] == {
        "iteration": 1025,
        "aggregate_psnr_db": 20.0,
        "gaussian_count": 2_345,
        "worst_drop_from_peak_db": 0.0,
    }


def test_partition_and_spatial_focus_compose_without_erasing_weak_partition(
    tmp_path,
):
    """A neutral spatial result must not erase active partition debt."""
    from eligsir.contracts.runtime import SourceIdentity
    from eligsir.runtime.causal_evaluation import CausalEvaluationObserver

    weak = SourceIdentity("camera", 0, 10, 10)
    spatial = SourceIdentity("camera", 0, 90, 90)
    published = []
    runtime = SimpleNamespace(
        replay_candidate_identities=lambda: (weak, spatial),
        source_identity_for_fingerprint=lambda _fingerprint: weak,
        publish_replay_focus_weights=lambda weights: published.append(dict(weights)),
    )
    observer = object.__new__(CausalEvaluationObserver)
    observer.runtime = runtime
    observer.dataset = SimpleNamespace(model_path=str(tmp_path))
    observer._source_complete = True
    observer._mapping_active_s = 0.0
    observer._supervision_views = 0
    observer._partition_replay_focus_accumulator = SimpleNamespace(
        observe=lambda _source, _psnr: {"score_scope": "test"}
    )
    observer._partition_replay_focus = SimpleNamespace(
        update=lambda _record, _sources: SimpleNamespace(
            active=True,
            reason="partition_psnr_focus",
            history_samples=3,
            partition_weights=(4.0, 1.0),
            weights=((weak, 4.0), (spatial, 1.0)),
        )
    )
    observer._spatial_replay_focus = SimpleNamespace(
        update=lambda _sample, _sources: SimpleNamespace(
            active=False,
            reason="spatial_microcell_balanced",
            actuation_mode="cell_debt",
            score_scope="test",
            history_samples=3,
            cell_drop_from_peak_db=(),
            cell_psnr_db=(),
            cell_cumulative_samples=(),
            cell_gain_since_first_observed_db=(),
            weights=((weak, 1.0), (spatial, 1.0)),
        )
    )
    observer._next_partition_control_optimizer_update = None
    observer._next_evaluation_optimizer_update = None

    observer._observe_tick(
        SimpleNamespace(
            iteration=100,
            primary_training_psnr_db=10.0,
            primary_training_source_id=weak.fingerprint(),
            primary_training_spatial=SimpleNamespace(cell_size_m=0.5, source=weak),
            primary_training_gaussian_count=10,
        )
    )

    assert published == [{weak: 4.0}]

    observer._observe_tick(
        SimpleNamespace(
            iteration=101,
            primary_training_psnr_db=None,
            primary_training_source_id="",
            primary_training_spatial=SimpleNamespace(cell_size_m=0.5, source=weak),
            primary_training_gaussian_count=10,
        )
    )

    assert published == [{weak: 4.0}, {weak: 4.0}]


def test_spatial_replay_focus_plot_tracks_quantiles_and_physical_cells(tmp_path):
    from eligsir.runtime.causal_evaluation import plot_spatial_replay_focus

    rows = []
    for iteration in (100, 200, 300):
        rows.append(
            {
                "iteration": iteration,
                "cells": [
                    {
                        "cell_key": key,
                        "center_xyz_m": [float(key), 0.0, 0.0],
                        "psnr_db": 10.0 + key + iteration / 100.0,
                        "drop_from_peak_db": 0.0,
                    }
                    for key in range(1, 5)
                ],
            }
        )
    (tmp_path / "spatial_replay_focus.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )

    output = plot_spatial_replay_focus(tmp_path)

    assert output == tmp_path / "spatial_replay_focus_over_time.svg"
    assert output.is_file()


def test_spatial_replay_focus_plot_orders_heatmap_rows_by_first_seen_iteration(
    tmp_path,
):
    """Heatmap rows should reflect trajectory-admission order, not cell_key."""
    from eligsir.runtime.causal_evaluation import (
        _SPATIAL_FOCUS_HEATMAP_MAX_ROWS,
        plot_spatial_replay_focus,
    )

    # cell_key 99 is numerically largest but appears FIRST; cell_key 1
    # appears only later. A hash/key-ordered heatmap would put 1 above 99;
    # first-seen-iteration ordering must put 99 above 1.
    rows = [
        {
            "iteration": 100,
            "cells": [
                {
                    "cell_key": 99,
                    "center_xyz_m": [1.0, 2.0, 3.0],
                    "psnr_db": 15.0,
                    "drop_from_peak_db": 0.0,
                }
            ],
        },
        {
            "iteration": 200,
            "cells": [
                {
                    "cell_key": 99,
                    "center_xyz_m": [1.0, 2.0, 3.0],
                    "psnr_db": 18.0,
                    "drop_from_peak_db": 0.0,
                },
                {
                    "cell_key": 1,
                    "center_xyz_m": [4.0, 5.0, 6.0],
                    "psnr_db": 20.0,
                    "drop_from_peak_db": 0.0,
                },
            ],
        },
    ]
    (tmp_path / "spatial_replay_focus.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )

    output = plot_spatial_replay_focus(tmp_path)

    assert output is not None and output.is_file()
    # No production caller reads the figure back apart from this test, so
    # exercise the module constant directly to confirm the row cap exists
    # and is a sane, positive bound.
    assert _SPATIAL_FOCUS_HEATMAP_MAX_ROWS > 0


def test_spatial_replay_focus_plot_caps_rows_for_large_cell_populations(tmp_path):
    from eligsir.runtime.causal_evaluation import (
        _SPATIAL_FOCUS_HEATMAP_MAX_ROWS,
        plot_spatial_replay_focus,
    )

    n_cells = _SPATIAL_FOCUS_HEATMAP_MAX_ROWS * 3
    rows = []
    for iteration in (100, 200, 300):
        rows.append(
            {
                "iteration": iteration,
                "cells": [
                    {
                        "cell_key": key,
                        "center_xyz_m": [float(key), 0.0, 0.0],
                        "psnr_db": 10.0 + (key % 20),
                        "drop_from_peak_db": 0.0,
                    }
                    for key in range(n_cells)
                ],
            }
        )
    (tmp_path / "spatial_replay_focus.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )

    # Should not raise despite far exceeding the row cap, and should
    # produce a normal-sized rendered file.
    output = plot_spatial_replay_focus(tmp_path)

    assert output is not None and output.is_file()
    assert output.stat().st_size > 0


def test_report_call_wires_train_metrics_max_not_hardcoded_zero():
    """Regression: `write_post_training_report` was called with a hardcoded
    `train_metrics_max=0` (landed in 60ad0c21, "Fence causal holdouts and
    report refinement quality"), unconditionally skipping train-view PSNR
    computation on causal_mapping's every mapping-endpoint/final report --
    `mean_train_psnr`/`min_train_psnr`/`max_train_psnr` silently read `None`
    on every real run, even though the sibling call sites in
    rolling_seed_application.py/submap_stitch_application.py already wired
    `streaming_report_train_metrics_max` correctly. Guards against the
    hardcoded literal reappearing."""
    pytest.importorskip("torch")
    import inspect
    import eligsir.runtime.causal_evaluation as causal_evaluation

    source = inspect.getsource(causal_evaluation)
    assert "train_metrics_max=0," not in source
    assert (
        'train_metrics_max=int(\n                getattr(self.args, '
        '"streaming_report_train_metrics_max", 128)\n            ),'
        in source
    )


def test_partition_control_publication_survives_spatial_republish(tmp_path):
    """Issue #1723 wiring: the spatial controller's per-tick composed republish
    must not erase a partition-control (debt allocator) publication.

    In the real runs every spatial tick (~7 iterations) reset the debt
    allocator's 20 sources to weight 1.0, so its actuation had a ~0.7% duty
    cycle and realized sample share never moved.
    """
    from eligsir.contracts.runtime import SourceIdentity
    from eligsir.runtime.causal_evaluation import CausalEvaluationObserver

    focus = SourceIdentity("camera", 0, 10, 10)
    other = SourceIdentity("camera", 0, 90, 90)
    published = []
    runtime = SimpleNamespace(
        replay_candidate_identities=lambda: (focus, other),
        source_identity_for_fingerprint=lambda _fingerprint: other,
        publish_replay_focus_weights=lambda weights: published.append(dict(weights)),
    )
    observer = object.__new__(CausalEvaluationObserver)
    observer.runtime = runtime
    observer.dataset = SimpleNamespace(model_path=str(tmp_path))
    observer._source_complete = True
    observer._mapping_active_s = 0.0
    observer._supervision_views = 0
    observer._optimizer_updates = 0
    observer._partition_replay_focus_accumulator = None
    observer._partition_replay_focus = None
    observer._spatial_replay_focus = SimpleNamespace(
        update=lambda _sample, _sources: SimpleNamespace(
            active=True,
            reason="spatial_microcell_focus",
            actuation_mode="frame_averaged",
            score_scope="test",
            history_samples=3,
            cell_drop_from_peak_db=(),
            cell_psnr_db=(),
            cell_cumulative_samples=(),
            cell_gain_since_first_observed_db=(),
            weights=((focus, 1.0), (other, 3.0)),
        )
    )
    observer._partition_control_runtime = SimpleNamespace(
        recorder=SimpleNamespace(record=lambda **_kwargs: None)
    )
    observer._next_partition_control_optimizer_update = None
    observer._next_evaluation_optimizer_update = None

    measurement = SimpleNamespace(
        published={focus: 50.0}, roles=None, decision=None, telemetry={}
    )
    observer._publish_partition_control(SimpleNamespace(iteration=1000), measurement)
    assert published[-1] == {focus: 50.0}

    observer._observe_tick(
        SimpleNamespace(
            iteration=1007,
            primary_training_psnr_db=None,
            primary_training_source_id="",
            primary_training_spatial=SimpleNamespace(cell_size_m=0.5, source=other),
            primary_training_gaussian_count=10,
        )
    )

    assert published[-1][focus] == 50.0
    assert published[-1][other] == 3.0
