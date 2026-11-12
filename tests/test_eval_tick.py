"""Unit tests for utils/eval_tick.py (issue #698).

CPU-only: no GPU, no real rendering, no training. The renderer is the same
monkeypatched stand-in tests/test_psnr_history.py already uses, so the
"no extra render" test exercises the REAL ``record_and_plot`` -> ``record_eval_tick``
sequence the training loop runs, not a mock of it.
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
import torch

import eligsir.utils.comparison_report as comparison_report
import eligsir.utils.eval_tick as eval_tick
import eligsir.utils.image_utils as image_utils
import eligsir.utils.partition_eval as partition_eval
import eligsir.utils.psnr_history as psnr_history
from eligsir.utils.eval_tick import (
    CANONICAL_PARTITION_SCHEMA,
    DIAGNOSTIC_PARTITION_SCHEMA,
    EVAL_TICK_FILENAME,
    EVAL_TICK_SCHEMA_VERSION,
    append_eval_tick,
    build_eval_tick_record,
    derive_phase,
    partition_summary,
    plot_partition_psnr_over_time,
    _partition_plot_work_axis,
    _partition_plot_xlim,
    psnr_history_from_eval_ticks,
    read_eval_ticks,
    record_final_report_eval_tick,
    record_eval_tick,
    reduce_partition_psnr,
    reduce_partition_psnr_cpu,
    resolve_source_progress,
)
from eligsir.utils.partition_eval import PARTITIONS
from eligsir.utils.psnr_history import PsnrTickResult, record_and_plot


class FakeCam:
    """Minimal camera: `uid` is what `camera_source_index` reads."""

    def __init__(self, uid, psnr_value=0.0, image_name=None):
        self.uid = uid
        self.psnr_value = psnr_value
        self.image_name = image_name if image_name is not None else f"{uid:06d}"


def _record_exact_terminal_constellation(tmp_path, constellation, **clock_overrides):
    cams = [FakeCam(10, 20.0)]
    clocks = {
        "source_frame_index": 90,
        "frames_ingested": 91,
        "frames_trained": 73,
        "keyframes_admitted": 44,
        "optimizer_step": 4998,
        "supervision_views": 9996,
        "mapping_snapshot_id": 44,
        "control_tick_epoch": 12,
        "supervision_constellation": constellation,
        "last_ingest_iter": 4700,
        "mapping_idle_threshold_iters": 50000,
    }
    clocks.update(clock_overrides)
    return record_final_report_eval_tick(
        model_path=str(tmp_path),
        iteration=5000,
        report_iteration=5000,
        view_psnr_db=[20.0],
        scored_cams=cams,
        test_cams=cams,
        reported_mean_test_psnr_db=20.0,
        clock_snapshot=clocks,
    )


def _valid_adaptive_terminal_constellation():
    return {
        "requested_batch_size": 256,
        "realized_batch_size": 7,
        "requested_camera_count": 4,
        "realized_camera_count": 4,
        "unique_eligible_count": 7,
        "fill_ratio": 7 / 256,
        "microbatch_count": 2,
        "microbatch_sizes": [4, 3],
        "optimizer_updates": 1,
        "supervised_views": 7,
        "unique_supervision_views": 7,
        "supervised_pixels": 28672,
        "cumulative_realized_supervised_views": 9996,
        "cumulative_optimizer_updates": 4998,
        "mode": "true_camera_microbatch_accumulation",
        "reason": "eligible_underfill",
        "hysteresis_state": "stable_hold",
    }


def _patch_render_pipeline(monkeypatch, render_counter=None):
    """Same stand-in render path as tests/test_psnr_history.py."""

    def _fake_render_image(cam, gaussians, render_fn, pipe, background):
        if render_counter is not None:
            render_counter["renders"] += 1
        return torch.full((3, 4, 4), cam.psnr_value / 100.0)

    def _fake_gt_image(cam):
        from eligsir.utils.score_contract import declare_canonical_score_tensor

        gt, _ = declare_canonical_score_tensor(
            torch.zeros(3, 4, 4), context="test_eval_tick_gt"
        )
        return gt

    def _psnr_fn(render_img, gt, **_kwargs):
        class _Scalar:
            def mean(self_inner):
                return render_img.reshape(-1)[0] * 100.0

        return _Scalar()

    monkeypatch.setattr(comparison_report, "_render_image", _fake_render_image)
    monkeypatch.setattr(comparison_report, "_gt_image", _fake_gt_image)
    monkeypatch.setattr(image_utils, "psnr", _psnr_fn)
    monkeypatch.setattr(psnr_history, "scientific_psnr", _psnr_fn)


# ---------------------------------------------------------------------------
# reduction
# ---------------------------------------------------------------------------
def test_reduction_gpu_cpu_parity():
    """scatter_add_ reduction == plain-Python group-by, exactly.

    Fails if the scatter reduction is neutered (e.g. means computed over the
    whole tensor instead of per partition).
    """
    rng = np.random.default_rng(0)
    n, k, max_src = 37, 6, 119
    src = [int(i) for i in rng.integers(0, max_src + 1, size=n)]
    vals = [float(v) for v in rng.uniform(10.0, 35.0, size=n)]
    cams = [FakeCam(s, v) for s, v in zip(src, vals)]
    tensor = torch.tensor(vals, dtype=torch.float32)

    gpu = reduce_partition_psnr(tensor, cams, max_src, k)
    cpu = reduce_partition_psnr_cpu(vals, src, max_src, k)

    assert gpu.counts == cpu.counts
    assert gpu.mean_db == cpu.mean_db
    assert gpu.view_source_indices == cpu.view_source_indices
    assert gpu.view_psnr_db == pytest.approx(cpu.view_psnr_db, abs=1e-3)
    # And the reference itself is a real group-by, not a global mean:
    assert len(set(v for v in cpu.mean_db if v is not None)) > 1


def test_reduction_is_a_single_host_sync(monkeypatch):
    """Exactly one Tensor.cpu() call per tick, and no per-view .item().

    Fails if someone re-introduces a per-view sync loop.
    """
    counts = {"cpu": 0, "item": 0}
    orig_cpu, orig_item = torch.Tensor.cpu, torch.Tensor.item

    def counted_cpu(self, *a, **kw):
        counts["cpu"] += 1
        return orig_cpu(self, *a, **kw)

    def counted_item(self, *a, **kw):
        counts["item"] += 1
        return orig_item(self, *a, **kw)

    monkeypatch.setattr(torch.Tensor, "cpu", counted_cpu)
    monkeypatch.setattr(torch.Tensor, "item", counted_item)
    cams = [FakeCam(i * 7) for i in range(20)]
    reduce_partition_psnr(torch.arange(20, dtype=torch.float32), cams, 133, 6)
    assert counts["cpu"] == 1
    assert counts["item"] == 0


def test_reduction_refuses_misaligned_inputs():
    with pytest.raises(ValueError):
        reduce_partition_psnr(torch.zeros(3), [FakeCam(0), FakeCam(1)], 10, 6)


def test_partition_assignment_is_deterministic_and_matches_partition_eval():
    """Reduction bins are exactly partition_eval.partition_id's bins."""
    max_src, k = 100, 5
    src = [0, 10, 25, 40, 60, 80, 99, 100]
    cams = [FakeCam(s, float(s)) for s in src]
    red = reduce_partition_psnr(torch.tensor([float(s) for s in src]), cams, max_src, k)
    expected = [0] * k
    for s in src:
        expected[partition_eval.partition_id(s, max_src, k=k)] += 1
    assert red.counts == expected
    # deterministic across repeated calls
    red2 = reduce_partition_psnr(torch.tensor([float(s) for s in src]), cams, max_src, k)
    assert red2.counts == red.counts and red2.mean_db == red.mean_db


def test_unresolvable_source_index_is_counted_not_hidden():
    cam_bad = FakeCam(0)
    cam_bad.uid = None
    cam_bad.colmap_id = None
    cam_bad.image_name = "no-digits-here"
    cams = [FakeCam(50, 20.0), cam_bad]
    red = reduce_partition_psnr(torch.tensor([20.0, 30.0]), cams, 100, 4)
    assert red.n_views_unresolved_source_index == 1
    assert red.view_source_indices[1] is None


# ---------------------------------------------------------------------------
# sparse / missing partitions
# ---------------------------------------------------------------------------
def test_empty_partitions_are_null_not_zero():
    """A partition with no holdout view must be None, never 0.0 dB.

    Fails if the mean is computed as sums / counts.clamp_min(1) without the
    zero-count guard.
    """
    cams = [FakeCam(0, 20.0), FakeCam(1, 22.0)]  # both land in partition 0
    red = reduce_partition_psnr(torch.tensor([20.0, 22.0]), cams, 100, 6)
    assert red.mean_db[0] == pytest.approx(21.0)
    assert all(v is None for v in red.mean_db[1:])
    assert red.counts[1:] == [0] * 5
    summ = partition_summary(red.mean_db, red.counts)
    assert summ["n_partitions_observed"] == 1
    assert summ["n_partitions_empty"] == 5
    assert summ["partition_min_db"] == pytest.approx(21.0)
    assert summ["worst_partition_id"] == 0


def test_partition_summary_matches_numpy_percentiles():
    means = [10.0, 20.0, 30.0, 40.0, None, 50.0]
    counts = [1, 1, 1, 1, 0, 1]
    summ = partition_summary(means, counts)
    observed = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    assert summ["partition_median_db"] == pytest.approx(float(np.percentile(observed, 50)), abs=1e-4)
    assert summ["partition_p10_db"] == pytest.approx(float(np.percentile(observed, 10)), abs=1e-4)
    assert summ["partition_p90_db"] == pytest.approx(float(np.percentile(observed, 90)), abs=1e-4)
    assert summ["worst_partition_id"] == 0
    assert summ["worst_partition_psnr_db"] == pytest.approx(10.0)


def test_partition_summary_all_empty():
    summ = partition_summary([None] * 6, [0] * 6)
    assert summ["partition_median_db"] is None
    assert summ["worst_partition_id"] is None
    assert summ["n_partitions_observed"] == 0


# ---------------------------------------------------------------------------
# clocks / denominator semantics
# ---------------------------------------------------------------------------
def test_source_progress_uses_declared_denominator_not_realized_prefix():
    """A 5%-prefix run must report progress 0.05, not 1.0.

    This is the regression guard for the "never renormalize from the realized
    truncated prefix" rule. Fails if the denominator is derived from the
    cameras actually seen.
    """
    p, reason = resolve_source_progress(126, 2525)
    assert reason is None
    assert p == pytest.approx(126 / 2525)
    assert p < 0.06


def test_source_progress_null_on_registry_miss():
    p, reason = resolve_source_progress(126, None)
    assert p is None
    assert reason and "no_declared_denominator" in reason


def test_source_progress_refuses_a_denominator_mismatch_instead_of_clamping():
    """A numerator past the declared length is a RULER mismatch, not progress
    1.0. Fails if the value is clamped (which would report 'trajectory
    finished' for two quantities measured against different rulers)."""
    p, reason = resolve_source_progress(9999, 100)
    assert p is None
    assert reason and reason.startswith("progress_denominator_mismatch")
    p2, reason2 = resolve_source_progress(-5, 100)
    assert p2 is None and reason2.startswith("progress_denominator_mismatch")
    # the boundary itself is legal (whole trajectory consumed)
    assert resolve_source_progress(100, 100) == (1.0, None)


def test_hash_null_reasons_distinguish_withheld_from_not_supplied():
    """A withheld hash (camera lacks a stable identifier) must not be reported
    as 'not supplied by caller'."""
    red = reduce_partition_psnr_cpu([20.0], [1], 100, 4)
    no_pool = build_eval_tick_record(iteration=1, mean_test_psnr_db=20.0, reduction=red)
    assert no_pool["field_provenance"]["realized_holdout_hash"] == "no_camera_pool_supplied"
    assert (no_pool["field_provenance"]["scored_holdout_hash"]
            == "hash_withheld_camera_lacks_stable_identifier")
    withheld = build_eval_tick_record(
        iteration=1, mean_test_psnr_db=20.0, reduction=red, n_test_views_attempted=3)
    assert (withheld["field_provenance"]["realized_holdout_hash"]
            == "hash_withheld_camera_lacks_stable_identifier")


def test_derive_phase_semantics():
    assert derive_phase(1000, 900, 500)[0] == "mapping"
    assert derive_phase(2000, 900, 500)[0] == "post_ingest_idle"
    phase, source, reason = derive_phase(1000, None, 500)
    assert phase is None and source == "unavailable" and reason == "no_last_ingest_iter"
    phase2, _s, reason2 = derive_phase(1000, 900, None)
    assert phase2 is None and reason2 == "no_idle_threshold"


def test_record_joins_clocks_verbatim_at_the_tick():
    """Every clock passed in appears in the row unchanged (exact tick join)."""
    red = reduce_partition_psnr_cpu([20.0, 30.0], [10, 90], 100, 4)
    rec = build_eval_tick_record(
        iteration=17000,
        mean_test_psnr_db=25.0,
        reduction=red,
        declared_n_source_frames=1000,
        source_frame_index=250,
        frames_ingested=240,
        frames_trained=230,
        keyframes_admitted=77,
        optimizer_step=16999,
        supervision_views=34000,
        wall_clock_s=612.5,
        accumulated_iteration_wall_s=600.25,
        last_ingest_iter=16800,
        mapping_idle_threshold_iters=50000,
    )
    assert rec["schema_version"] == EVAL_TICK_SCHEMA_VERSION
    assert rec["iteration"] == 17000
    assert rec["optimizer_step"] == 16999
    assert rec["supervision_views"] == 34000
    assert rec["frames_ingested"] == 240
    assert rec["frames_trained"] == 230
    assert rec["keyframes_admitted"] == 77
    assert rec["wall_clock_s"] == pytest.approx(612.5)
    assert rec["accumulated_iteration_wall_s"] == pytest.approx(600.25)
    assert rec["last_ingest_iter"] == 16800
    assert rec["source_progress"] == pytest.approx(0.25)
    assert rec["phase"] == "mapping"
    assert rec["mean_test_psnr_db"] == pytest.approx(25.0)
    assert rec["n_test_views"] == 2
    # #428 axis-name compatibility: constructible as a TickObservation coord set
    for axis in ("source_progress", "wall_clock_s", "supervision_views", "iteration"):
        assert axis in rec


def test_record_joins_adaptive_constellation_to_the_exact_mapping_tick():
    red = reduce_partition_psnr_cpu([20.0], [10], 100, 4)
    constellation = {
        "requested_batch_size": 256,
        "realized_batch_size": 64,
        "requested_camera_count": 4,
        "realized_camera_count": 64,
        "unique_eligible_count": 64,
        "fill_ratio": 0.25,
        "microbatch_count": 1,
        "supervised_views": 63,
        "cumulative_realized_supervised_views": 120,
        "cumulative_optimizer_updates": 30,
        "mode": "true_camera_batch",
        "reason": "eligible_underfill",
        "hysteresis_state": "stable_hold",
    }

    rec = build_eval_tick_record(
        iteration=17000,
        mean_test_psnr_db=20.0,
        reduction=red,
        mapping_snapshot_id=42,
        control_tick_epoch=17,
        supervision_constellation=constellation,
    )

    assert rec["mapping_snapshot_id"] == 42
    assert rec["control_tick_epoch"] == 17
    assert rec["supervision_constellation"] == constellation


def test_record_axis_fields_construct_a_tick_observation_without_renaming():
    """#428 join: the row's clock keys feed TickObservation's axis coords
    directly. Fails if a field is renamed or leaves the (0, 1] progress range."""
    from eligsir.utils.causal_quality import TickObservation

    red = reduce_partition_psnr_cpu([20.0, 30.0], [10, 90], 100, 4)
    rec = build_eval_tick_record(
        iteration=17000, mean_test_psnr_db=25.0, reduction=red,
        declared_n_source_frames=1000, source_frame_index=250,
        supervision_views=34000, wall_clock_s=612.5,
    )
    obs = TickObservation(
        cell_keys=np.zeros(0, dtype=np.int64),
        sq_err=np.zeros(0), n_samples=np.zeros(0), n_views=np.zeros(0),
        **{k: rec[k] for k in ("source_progress", "wall_clock_s",
                               "supervision_views", "iteration")},
    )
    assert obs.source_progress == pytest.approx(0.25)
    assert obs.wall_clock_s == pytest.approx(612.5)
    assert obs.supervision_views == 34000
    assert obs.iteration == 17000


def test_missing_fields_are_null_with_an_explicit_reason():
    red = reduce_partition_psnr_cpu([20.0], [1], 100, 4)
    rec = build_eval_tick_record(
        iteration=100, mean_test_psnr_db=20.0, reduction=red,
        declared_n_source_frames=None, source_frame_index=5,
    )
    assert rec["source_progress"] is None
    assert "source_progress" in rec["field_provenance"]
    assert rec["phase"] is None
    assert "phase" in rec["field_provenance"]
    assert rec["supervision_views"] is None
    assert rec["field_provenance"]["supervision_views"] == "not_supplied_by_caller"


# ---------------------------------------------------------------------------
# persistence + legacy compatibility
# ---------------------------------------------------------------------------
def test_append_and_read_roundtrip_sorted(tmp_path):
    red = reduce_partition_psnr_cpu([20.0], [1], 100, 4)
    for it in (3000, 1000, 2000):
        append_eval_tick(str(tmp_path), build_eval_tick_record(
            iteration=it, mean_test_psnr_db=float(it) / 100.0, reduction=red))
    rows = read_eval_ticks(str(tmp_path))
    assert [r["iteration"] for r in rows] == [1000, 2000, 3000]
    assert os.path.exists(os.path.join(str(tmp_path), EVAL_TICK_FILENAME))


def test_read_eval_ticks_skips_a_torn_final_line(tmp_path):
    red = reduce_partition_psnr_cpu([20.0], [1], 100, 4)
    append_eval_tick(str(tmp_path), build_eval_tick_record(
        iteration=1000, mean_test_psnr_db=20.0, reduction=red))
    with open(os.path.join(str(tmp_path), EVAL_TICK_FILENAME), "a") as f:
        f.write('{"iteration": 2000, "mean_test')  # killed mid-write
    rows = read_eval_ticks(str(tmp_path))
    assert [r["iteration"] for r in rows] == [1000]


def test_legacy_psnr_history_is_reproducible_from_the_record(tmp_path, monkeypatch):
    """Acceptance criterion 7: the legacy (iters, psnr) series is recoverable
    from evaluation_ticks.jsonl and matches psnr_history.jsonl exactly."""
    from eligsir.utils.psnr_history_io import _read_psnr_history

    _patch_render_pipeline(monkeypatch)
    cams = [FakeCam(10, 20.0), FakeCam(90, 30.0)]
    for it in (1000, 2000):
        tick = record_and_plot(str(tmp_path), it, None, cams, None, None, None)
        record_eval_tick(model_path=str(tmp_path), iteration=it, psnr_tick=tick,
                         test_cams=cams, plot=False)
    legacy_iters, legacy_vals = _read_psnr_history(str(tmp_path))
    new_iters, new_vals = psnr_history_from_eval_ticks(str(tmp_path))
    assert new_iters == legacy_iters
    assert new_vals == pytest.approx(legacy_vals)
    assert new_vals == pytest.approx([25.0, 25.0])


# ---------------------------------------------------------------------------
# integration with the real tick: no extra render, identical mean PSNR
# ---------------------------------------------------------------------------
def test_record_eval_tick_adds_no_render(tmp_path, monkeypatch):
    """Acceptance criterion 6. Fails if the module ever renders anything."""
    counter = {"renders": 0}
    _patch_render_pipeline(monkeypatch, render_counter=counter)
    cams = [FakeCam(10, 20.0), FakeCam(50, 24.0), FakeCam(95, 28.0)]

    tick = record_and_plot(str(tmp_path), 1000, None, cams, None, None, None)
    renders_after_tick = counter["renders"]
    assert renders_after_tick == len(cams)

    rec = record_eval_tick(model_path=str(tmp_path), iteration=1000, psnr_tick=tick,
                           test_cams=cams, plot=False)
    assert rec is not None
    assert counter["renders"] == renders_after_tick  # zero additional renders
    assert rec["metric_provenance"]["extra_renders"] == 0


def test_canonical_eval_tick_records_observer_phase_and_bounded_camera_pool(tmp_path):
    bounded = [FakeCam(10, 20.0, "bounded-a"), FakeCam(20, 22.0, "bounded-b")]
    tick = PsnrTickResult(
        cams=bounded,
        psnr_tensor=torch.tensor([20.0, 22.0]),
        mean_psnr=21.0,
    )

    rec = record_eval_tick(
        model_path=str(tmp_path),
        iteration=100,
        psnr_tick=tick,
        test_cams=bounded,
        evaluation_phase="mapping",
        evaluation_observer_execution="synchronous_mapping_thread_pause",
        mapping_execution_paused=True,
        throughput_evidence_valid=False,
        throughput_evidence_status="invalid_mapping_holdout_observer_pause",
        evaluation_cadence_clock="cumulative_optimizer_updates",
        plot=False,
    )

    assert rec["evaluation_phase"] == "mapping"
    assert rec["evaluation_observer_execution"] == (
        "synchronous_mapping_thread_pause"
    )
    assert rec["mapping_execution_paused"] is True
    assert rec["throughput_evidence_valid"] is False
    assert rec["throughput_evidence_status"] == (
        "invalid_mapping_holdout_observer_pause"
    )
    assert rec["evaluation_cadence_clock"] == "cumulative_optimizer_updates"
    assert rec["psnr_reduction"] == "rgb_per_view_v2"
    assert rec["n_test_views_attempted"] == 2
    assert rec["realized_holdout_hash"] == rec["scored_holdout_hash"]


def test_final_report_eval_tick_preserves_observer_validity_context(tmp_path):
    constellation = _valid_adaptive_terminal_constellation()
    rec = _record_exact_terminal_constellation(
        tmp_path,
        constellation,
        evaluation_phase="post_mapping_report",
        evaluation_observer_execution="post_source_completion",
        mapping_execution_paused=False,
        throughput_evidence_valid=False,
        throughput_evidence_status="invalid_mapping_holdout_observer_pause",
        evaluation_cadence_clock="cumulative_optimizer_updates",
    )

    row = read_eval_ticks(str(tmp_path))[0]
    assert row["evaluation_phase"] == "post_mapping_report"
    assert row["mapping_execution_paused"] is False
    assert row["throughput_evidence_valid"] is False
    assert row["throughput_evidence_status"] == (
        "invalid_mapping_holdout_observer_pause"
    )
    assert row["evaluation_cadence_clock"] == "cumulative_optimizer_updates"
    assert rec["status"] == "bound"


def test_canonical_eval_tick_carries_antialiasing_execution_provenance(tmp_path):
    cams = [FakeCam(10, 20.0)]
    tick = PsnrTickResult(
        cams=cams,
        psnr_tensor=torch.tensor([20.0]),
        mean_psnr=20.0,
        mean_edge_psnr=18.0,
        rasterize_mode="antialiased",
        gaussian_count=12345,
        training_resolution_tiers=("half",),
        evaluation_resolutions=((640, 480),),
    )

    rec = record_eval_tick(
        model_path=str(tmp_path), iteration=1000, psnr_tick=tick, test_cams=cams
    )

    aa = rec["antialiasing_metrics"]
    assert aa["rasterize_mode"] == "antialiased"
    assert aa["gaussian_count"] == 12345
    assert aa["training_resolution_tiers"] == ["half"]
    assert aa["evaluation_resolutions"] == [[640, 480]]


def test_canonical_eval_tick_never_serializes_nonfinite_antialias_metrics(tmp_path):
    cams = [FakeCam(10, 20.0)]
    tick = PsnrTickResult(
        cams=cams,
        psnr_tensor=torch.tensor([20.0]),
        mean_psnr=20.0,
        mean_edge_psnr=float("inf"),
        temporal_flicker_rms=float("nan"),
        temporal_flicker_psnr_db=float("inf"),
        temporal_flicker_status="valid_motion_compensated_perfect_match",
    )

    rec = record_eval_tick(
        model_path=str(tmp_path), iteration=1000, psnr_tick=tick, test_cams=cams
    )

    aa = rec["antialiasing_metrics"]
    assert aa["mean_edge_psnr_db"] is None
    assert aa["temporal_flicker_rms"] is None
    assert aa["temporal_flicker_psnr_db"] is None
    # Strict JSON encoding is the final guard against NaN/Infinity output.
    json.dumps(rec, allow_nan=False)


def test_record_eval_tick_hot_path_is_json_only_and_measures_its_named_scope(
    tmp_path, monkeypatch
):
    """The live path must neither plot nor claim end-to-end tick latency.

    The recorded scope starts immediately before the same-render partition
    reduction and ends before JSON encoding/append, which avoids a
    self-referential rewrite while still including the one D2H sync.
    """
    _patch_render_pipeline(monkeypatch)
    cams = [FakeCam(10, 20.0), FakeCam(90, 30.0)]
    tick = record_and_plot(str(tmp_path), 1000, None, cams, None, None, None)
    plot_calls = []
    monkeypatch.setattr(
        eval_tick,
        "plot_partition_psnr_over_time",
        lambda *_a, **_k: plot_calls.append(True),
    )
    times = iter((10.0, 10.025))
    monkeypatch.setattr(eval_tick.time, "perf_counter", lambda: next(times))

    rec = record_eval_tick(
        model_path=str(tmp_path), iteration=1000, psnr_tick=tick, test_cams=cams
    )

    assert plot_calls == []
    cost = rec["record_hot_path"]
    assert cost["elapsed_scope"] == eval_tick.RECORD_HOT_PATH_ELAPSED_SCOPE
    assert cost["elapsed_ms"] == pytest.approx(25.0)
    assert "total_tick_elapsed_ms" not in rec
    raw = (tmp_path / EVAL_TICK_FILENAME).read_bytes()
    assert cost["serialized_bytes"] == len(raw)
    assert raw.endswith(b"\n")
    assert json.loads(raw)["record_hot_path"] == cost


def test_same_render_partition_weighted_mean_equals_global_and_adds_no_render(
    tmp_path, monkeypatch
):
    counter = {"renders": 0}
    _patch_render_pipeline(monkeypatch, render_counter=counter)
    cams = [FakeCam(5, 18.0), FakeCam(40, 24.0), FakeCam(95, 30.0)]
    tick = record_and_plot(str(tmp_path), 1000, None, cams, None, None, None)
    renders_after_tick = counter["renders"]

    rec = record_eval_tick(
        model_path=str(tmp_path),
        iteration=1000,
        psnr_tick=tick,
        test_cams=cams,
        partition_k=3,
    )

    weighted = sum(
        mean * count
        for mean, count in zip(rec["partition_psnr_db"], rec["partition_n"])
        if mean is not None
    ) / sum(rec["partition_n"])
    assert weighted == pytest.approx(rec["mean_test_psnr_db"], abs=1e-4)
    assert counter["renders"] == renders_after_tick


def test_mean_psnr_is_unchanged_by_the_record(tmp_path, monkeypatch):
    """The record reports exactly the tick's own mean; the legacy file's value
    is untouched. Fails if the record re-derives PSNR by another route."""
    _patch_render_pipeline(monkeypatch)
    cams = [FakeCam(10, 20.0), FakeCam(50, 24.0), FakeCam(95, 28.0)]
    tick = record_and_plot(str(tmp_path), 1000, None, cams, None, None, None)
    rec = record_eval_tick(model_path=str(tmp_path), iteration=1000, psnr_tick=tick,
                           test_cams=cams, plot=False)
    assert rec["mean_test_psnr_db"] == pytest.approx(float(tick.psnr_tensor.mean()))
    with open(os.path.join(str(tmp_path), "psnr_history.jsonl")) as f:
        legacy = json.loads(f.readline())
    assert legacy["mean_test_psnr"] == pytest.approx(rec["mean_test_psnr_db"], abs=1e-4)


def test_record_eval_tick_declared_denominator_wins_over_realized(tmp_path, monkeypatch):
    """With a registry hit, partitions use the DECLARED trajectory length --
    a prefix-only run must not stretch its few cameras across all partitions.

    Fails if `resolve_max_source_index` is replaced by max-over-cameras.
    """
    _patch_render_pipeline(monkeypatch)
    monkeypatch.setattr(eval_tick, "registry_n_source_frames", lambda s: 2525)
    monkeypatch.setattr(partition_eval, "registry_n_source_frames", lambda s: 2525)
    cams = [FakeCam(i, 20.0 + i) for i in (10, 40, 80, 120)]  # a 5% prefix
    tick = record_and_plot(str(tmp_path), 1000, None, cams, None, None, None)
    rec = record_eval_tick(model_path=str(tmp_path), iteration=1000, psnr_tick=tick,
                           test_cams=cams, source_frame_index=126, plot=False)
    assert rec["partition_denominator"] == 2524
    assert rec["partition_denominator_source"] == "registry"
    assert rec["declared_n_source_frames"] == 2525
    assert rec["source_progress"] == pytest.approx(126 / 2525)
    # every camera is in the FIRST partition; the rest are unmeasured (null)
    assert rec["partition_n"][0] == 4
    assert sum(rec["partition_n"][1:]) == 0
    assert all(v is None for v in rec["partition_psnr_db"][1:])


def test_record_eval_tick_uses_canonical_report_partitions_by_default(
    tmp_path, monkeypatch
):
    """Long scenes must not silently switch the longitudinal report to 21 bins."""
    _patch_render_pipeline(monkeypatch)
    monkeypatch.setattr(eval_tick, "registry_n_source_frames", lambda _s: 2525)
    monkeypatch.setattr(partition_eval, "registry_n_source_frames", lambda _s: 2525)
    cams = [FakeCam(i, 20.0 + i / 100.0) for i in (10, 400, 900, 1600, 2400)]
    tick = record_and_plot(str(tmp_path), 1000, None, cams, None, None, None)

    rec = record_eval_tick(
        model_path=str(tmp_path), iteration=1000, psnr_tick=tick,
        test_cams=cams, scene_source="long-scene", plot=False,
    )

    assert rec["partition_k"] == PARTITIONS == 6
    assert rec["partition_schema"] == CANONICAL_PARTITION_SCHEMA
    assert len(rec["partition_n"]) == PARTITIONS


def test_record_eval_tick_marks_noncanonical_partition_as_explicit_diagnostic(
    tmp_path, monkeypatch
):
    _patch_render_pipeline(monkeypatch)
    cams = [FakeCam(10, 20.0), FakeCam(90, 30.0)]
    tick = record_and_plot(str(tmp_path), 1000, None, cams, None, None, None)

    rec = record_eval_tick(
        model_path=str(tmp_path), iteration=1000, psnr_tick=tick,
        test_cams=cams, partition_k=3, plot=False,
    )

    assert rec["partition_k"] == 3
    assert rec["partition_schema"] == DIAGNOSTIC_PARTITION_SCHEMA


def test_final_tick_vector_counts_and_min_match_canonical_report_scheme(
    tmp_path, monkeypatch
):
    """The final tick and report use one six-way reduction of the same views."""
    from eligsir.utils.partition_eval import compute_partition_stats, partition_id

    monkeypatch.setattr(eval_tick, "registry_n_source_frames", lambda _s: 2525)
    monkeypatch.setattr(partition_eval, "registry_n_source_frames", lambda _s: 2525)
    cams = [
        FakeCam(source, value)
        for source, value in (
            (10, 20.0), (400, 24.0), (900, 19.0), (1600, 28.0), (2400, 17.0)
        )
    ]
    values = [cam.psnr_value for cam in cams]
    binding = eval_tick.record_final_report_eval_tick(
        model_path=str(tmp_path), iteration=5000, report_iteration=5000,
        view_psnr_db=values, scored_cams=cams, test_cams=cams,
        reported_mean_test_psnr_db=sum(values) / len(values),
        scene_source="long-scene",
    )

    row = read_eval_ticks(str(tmp_path))[-1]
    per_view = [
        {"partition": partition_id(cam.uid, 2524), "psnr": value}
        for cam, value in zip(cams, values)
    ]
    report = compute_partition_stats(per_view)
    expected_means = [report["partition_psnr"][f"p{i}"] for i in range(PARTITIONS)]
    expected_counts = [report["partition_n"][f"p{i}"] for i in range(PARTITIONS)]

    assert binding["status"] == "bound"
    assert row["partition_schema"] == CANONICAL_PARTITION_SCHEMA
    assert row["partition_k"] == PARTITIONS
    assert row["partition_psnr_db"] == pytest.approx(expected_means, nan_ok=True)
    assert row["partition_n"] == expected_counts
    assert row["partition_summary"]["partition_min_db"] == pytest.approx(
        report["partition_psnr_min"]
    )


def test_partition_plot_labels_explicit_diagnostic_count_and_min_semantics(
    tmp_path, monkeypatch
):
    diagnostic = reduce_partition_psnr_cpu([21.0, 25.0], [10, 90], 100, 3)
    append_eval_tick(
        str(tmp_path),
        build_eval_tick_record(
            iteration=1, mean_test_psnr_db=23.0, reduction=diagnostic,
            partition_schema=DIAGNOSTIC_PARTITION_SCHEMA,
        ),
    )
    import matplotlib.pyplot as plt

    captured = {}
    real_subplots = plt.subplots

    def _spy_subplots(*args, **kwargs):
        fig, ax = real_subplots(*args, **kwargs)
        captured["ax"] = ax
        return fig, ax

    monkeypatch.setattr(plt, "subplots", _spy_subplots)
    out = plot_partition_psnr_over_time(str(tmp_path), out_name="diagnostic.png")

    assert out and os.path.exists(out)
    title = captured["ax"].get_title(loc="left")
    assert "DIAGNOSTIC 3-way" in title
    assert "single-view bins possible" in title
    assert captured["ax"].get_legend().get_texts()[-1].get_text() == (
        "worst diagnostic bin mean"
    )


def test_partition_plot_keeps_empty_canonical_history_gradeable_as_sparse(
    tmp_path,
):
    reduction = reduce_partition_psnr_cpu([], [], 100, PARTITIONS)
    append_eval_tick(
        str(tmp_path),
        build_eval_tick_record(
            iteration=1, mean_test_psnr_db=None, reduction=reduction,
            partition_schema=CANONICAL_PARTITION_SCHEMA,
        ),
    )

    out = plot_partition_psnr_over_time(str(tmp_path), out_name="sparse.png")

    assert out and os.path.exists(out)


def test_partition_plot_rejects_malformed_canonical_schema_metadata(tmp_path, monkeypatch):
    reduction = reduce_partition_psnr_cpu([20.0], [10], 100, 3)
    append_eval_tick(
        str(tmp_path),
        build_eval_tick_record(
            iteration=1, mean_test_psnr_db=20.0, reduction=reduction,
            partition_schema=CANONICAL_PARTITION_SCHEMA,
        ),
    )
    import matplotlib.pyplot as plt

    captured = {}
    real_subplots = plt.subplots

    def _spy_subplots(*args, **kwargs):
        fig, ax = real_subplots(*args, **kwargs)
        captured["ax"] = ax
        return fig, ax

    monkeypatch.setattr(plt, "subplots", _spy_subplots)
    out = plot_partition_psnr_over_time(str(tmp_path), out_name="invalid.png")

    assert out and os.path.exists(out)
    assert "INVALID PARTITION SCHEMA" in captured["ax"].get_title(loc="left")


def test_partition_plot_labels_mixed_partition_schemas_as_not_comparable(
    tmp_path, monkeypatch
):
    canonical = reduce_partition_psnr_cpu([20.0, 24.0], [10, 90], 100, PARTITIONS)
    diagnostic = reduce_partition_psnr_cpu([21.0, 25.0], [10, 90], 100, 3)
    append_eval_tick(
        str(tmp_path),
        build_eval_tick_record(
            iteration=1, mean_test_psnr_db=22.0, reduction=canonical,
            partition_schema=CANONICAL_PARTITION_SCHEMA,
        ),
    )
    append_eval_tick(
        str(tmp_path),
        build_eval_tick_record(
            iteration=2, mean_test_psnr_db=23.0, reduction=diagnostic,
            partition_schema=DIAGNOSTIC_PARTITION_SCHEMA,
        ),
    )

    import matplotlib.pyplot as plt

    captured = {}
    real_subplots = plt.subplots

    def _spy_subplots(*args, **kwargs):
        fig, ax = real_subplots(*args, **kwargs)
        captured["ax"] = ax
        return fig, ax

    monkeypatch.setattr(plt, "subplots", _spy_subplots)
    out = plot_partition_psnr_over_time(str(tmp_path), out_name="mixed.png")

    assert out and os.path.exists(out)
    title = captured["ax"].get_title(loc="left")
    assert "MIXED PARTITION SCHEMAS" in title
    assert "NOT COMPARABLE" in title


def test_record_eval_tick_registry_miss_is_honest(tmp_path, monkeypatch):
    _patch_render_pipeline(monkeypatch)
    monkeypatch.setattr(eval_tick, "registry_n_source_frames", lambda s: None)
    monkeypatch.setattr(partition_eval, "registry_n_source_frames", lambda s: None)
    cams = [FakeCam(i, 20.0) for i in (10, 40, 80, 120)]
    tick = record_and_plot(str(tmp_path), 1000, None, cams, None, None, None)
    rec = record_eval_tick(model_path=str(tmp_path), iteration=1000, psnr_tick=tick,
                           test_cams=cams, source_frame_index=126, plot=False)
    assert rec["source_progress"] is None
    assert rec["partition_denominator_source"] == "fallback_max_cams"
    assert "source_progress" in rec["field_provenance"]


def test_record_eval_tick_returns_none_without_a_tensor(tmp_path):
    assert record_eval_tick(model_path=str(tmp_path), iteration=1, psnr_tick=None) is None
    empty = PsnrTickResult(cams=[], psnr_tensor=None, mean_psnr=None)
    assert record_eval_tick(model_path=str(tmp_path), iteration=1, psnr_tick=empty) is None
    assert not os.path.exists(os.path.join(str(tmp_path), EVAL_TICK_FILENAME))


def test_final_report_tick_is_bound_to_the_report_metric_pass(tmp_path):
    """The final report gets a canonical tick without a second render."""
    cams = [FakeCam(10, 20.0), FakeCam(90, 30.0)]
    binding = record_final_report_eval_tick(
        model_path=str(tmp_path),
        iteration=5000,
        report_iteration=2_147_483_647,
        view_psnr_db=[20.0, 30.0],
        scored_cams=cams,
        test_cams=cams,
        reported_mean_test_psnr_db=25.0,
    )

    assert binding["status"] == "bound"
    assert binding["evaluation_tick_iteration"] == 5000
    assert binding["report_iteration"] == 2_147_483_647
    assert len(binding["sha256"]) == 64
    row = read_eval_ticks(str(tmp_path))[-1]
    assert row["origin"] == "final_report"
    assert row["mean_test_psnr_db"] == pytest.approx(25.0)
    assert row["final_report_iteration"] == 2_147_483_647
    assert not (tmp_path / "partition_psnr_over_time.svg").exists()


def test_final_report_tick_joins_exact_terminal_streaming_clocks(tmp_path):
    cams = [FakeCam(10, 20.0), FakeCam(90, 30.0)]
    constellation = {
        "requested_batch_size": 256,
        "realized_batch_size": 7,
        "requested_camera_count": 4,
        "realized_camera_count": 4,
        "unique_eligible_count": 7,
        "fill_ratio": 7 / 256,
        "microbatch_count": 2,
        "microbatch_sizes": [4, 3],
        "optimizer_updates": 1,
        "supervised_views": 7,
        "unique_supervision_views": 7,
        "supervised_pixels": 28672,
        "cumulative_realized_supervised_views": 9996,
        "cumulative_optimizer_updates": 4998,
        "mode": "true_camera_microbatch_accumulation",
        "reason": "eligible_underfill",
        "hysteresis_state": "stable_hold",
    }
    record_final_report_eval_tick(
        model_path=str(tmp_path),
        iteration=5000,
        report_iteration=2_147_483_647,
        view_psnr_db=[20.0, 30.0],
        scored_cams=cams,
        test_cams=cams,
        scene_source="scene",
        reported_mean_test_psnr_db=25.0,
        clock_snapshot={
            "source_frame_index": 90,
            "frames_ingested": 91,
            "frames_trained": 73,
            "keyframes_admitted": 44,
            "optimizer_step": 4998,
            "supervision_views": 9996,
            "mapping_snapshot_id": 44,
            "control_tick_epoch": 12,
            "supervision_constellation": constellation,
            "wall_clock_s": 123.5,
            "accumulated_iteration_wall_s": 118.25,
            "last_ingest_iter": 4700,
            "mapping_idle_threshold_iters": 50000,
        },
    )
    row = read_eval_ticks(str(tmp_path))[-1]
    assert row["source_frame_index"] == 90
    assert row["frames_ingested"] == 91
    assert row["frames_trained"] == 73
    assert row["keyframes_admitted"] == 44
    assert row["optimizer_step"] == 4998
    assert row["iteration"] == 5000
    assert row["optimizer_step"] != row["iteration"]
    assert row["supervision_views"] == 9996
    assert row["mapping_snapshot_id"] == 44
    assert row["control_tick_epoch"] == 12
    assert row["supervision_constellation"] == constellation
    assert row["wall_clock_s"] == pytest.approx(123.5)
    assert row["accumulated_iteration_wall_s"] == pytest.approx(118.25)
    assert row["last_ingest_iter"] == 4700
    assert row["phase"] == "mapping"
    assert row["clock_snapshot_status"] == "caller_supplied_exact_terminal_snapshot"


def test_final_report_tick_refuses_incomplete_terminal_streaming_join(tmp_path):
    cams = [FakeCam(10, 20.0)]

    with pytest.raises(ValueError, match="exact terminal clock snapshot missing"):
        record_final_report_eval_tick(
            model_path=str(tmp_path),
            iteration=900,
            report_iteration=900,
            view_psnr_db=[20.0],
            scored_cams=cams,
            test_cams=cams,
            reported_mean_test_psnr_db=20.0,
            clock_snapshot={
                "optimizer_step": 32,
                "supervision_views": 183,
            },
        )


def test_final_report_tick_rejects_reviewed_impossible_exact_snapshot(tmp_path):
    impossible = {
        "requested_batch_size": 7,
        "realized_batch_size": 7,
        "requested_camera_count": 256,
        "realized_camera_count": 256,
        "unique_eligible_count": 7,
        "fill_ratio": 0.5,
        "microbatch_count": 2,
        "microbatch_sizes": [7],
        "optimizer_updates": 99,
        "supervised_views": 123,
        "unique_supervision_views": 999,
        "supervised_pixels": -1,
        "cumulative_realized_supervised_views": 9996,
        "cumulative_optimizer_updates": 4998,
        "mode": "true_camera_microbatch_accumulation",
        "reason": "eligible_underfill",
        "hysteresis_state": "stable_hold",
    }

    with pytest.raises(ValueError, match="exact terminal supervision constellation"):
        _record_exact_terminal_constellation(tmp_path, impossible)


@pytest.mark.parametrize(
    "constellation",
    [
        _valid_adaptive_terminal_constellation(),
        {
            "requested_batch_size": 256,
            "realized_batch_size": 256,
            "requested_camera_count": 64,
            "realized_camera_count": 64,
            "unique_eligible_count": 256,
            "fill_ratio": 1.0,
            "microbatch_count": 4,
            "microbatch_sizes": [64, 64, 64, 64],
            "optimizer_updates": 1,
            "supervised_views": 256,
            "unique_supervision_views": 256,
            "supervised_pixels": 1048576,
            "cumulative_realized_supervised_views": 9996,
            "cumulative_optimizer_updates": 4998,
            "mode": "true_camera_microbatch_accumulation",
            "reason": "configured_qualified_target",
            "hysteresis_state": "stable_hold",
        },
        {
            "requested_batch_size": 256,
            "realized_batch_size": 7,
            "requested_camera_count": 64,
            "realized_camera_count": 7,
            "unique_eligible_count": 7,
            "fill_ratio": 7 / 256,
            "microbatch_count": 1,
            "microbatch_sizes": [7],
            "optimizer_updates": 1,
            "supervised_views": 7,
            "unique_supervision_views": 7,
            "supervised_pixels": 28672,
            "cumulative_realized_supervised_views": 9996,
            "cumulative_optimizer_updates": 4998,
            "mode": "true_camera_batch",
            "reason": "eligible_underfill",
            "hysteresis_state": "stable_hold",
        },
        {
            "requested_batch_size": 1,
            "realized_batch_size": 1,
            "requested_camera_count": 1,
            "realized_camera_count": 1,
            "unique_eligible_count": 1,
            "fill_ratio": 1.0,
            "microbatch_count": 2,
            "microbatch_sizes": [1, 1],
            "optimizer_updates": 2,
            "supervised_views": 2,
            "unique_supervision_views": 1,
            "supervised_pixels": 8192,
            "cumulative_realized_supervised_views": 9996,
            "cumulative_optimizer_updates": 4998,
            "mode": "single_camera_immediate",
            "reason": "configured_qualified_target",
            "hysteresis_state": "stable_hold",
        },
        {
            "requested_batch_size": 256,
            "realized_batch_size": 0,
            "requested_camera_count": 64,
            "realized_camera_count": 0,
            "unique_eligible_count": 0,
            "fill_ratio": 0.0,
            "microbatch_count": 0,
            "microbatch_sizes": [],
            "optimizer_updates": 0,
            "supervised_views": 0,
            "unique_supervision_views": 0,
            "supervised_pixels": 0,
            "cumulative_realized_supervised_views": 9996,
            "cumulative_optimizer_updates": 4998,
            "mode": "empty",
            "reason": "eligible_underfill",
            "hysteresis_state": "stable_hold",
        },
    ],
    ids=(
        "b_gt_c_underfill",
        "b256_c64_accumulation",
        "high_c_underfill",
        "legacy_b1_multi_update",
        "empty",
    ),
)
def test_final_report_tick_accepts_valid_exact_constellation_semantics(
    tmp_path, constellation
):
    _record_exact_terminal_constellation(tmp_path, constellation)

    assert (
        read_eval_ticks(str(tmp_path))[-1]["supervision_constellation"]
        == constellation
    )


def test_final_report_tick_accepts_explicitly_unknown_unique_view_count(tmp_path):
    constellation = _valid_adaptive_terminal_constellation()
    constellation["unique_supervision_views"] = None

    _record_exact_terminal_constellation(tmp_path, constellation)

    assert (
        read_eval_ticks(str(tmp_path))[-1]["supervision_constellation"][
            "unique_supervision_views"
        ]
        is None
    )


@pytest.mark.parametrize("known_unique_views", (0, 3))
def test_final_report_tick_rejects_known_unique_view_undercount(
    tmp_path, known_unique_views
):
    constellation = _valid_adaptive_terminal_constellation()
    constellation["unique_supervision_views"] = known_unique_views

    with pytest.raises(ValueError, match="known unique supervised views"):
        _record_exact_terminal_constellation(tmp_path, constellation)


def test_final_report_tick_rejects_pixels_without_supervised_work(tmp_path):
    constellation = _valid_adaptive_terminal_constellation()
    constellation.update(
        realized_batch_size=0,
        realized_camera_count=0,
        unique_eligible_count=0,
        fill_ratio=0.0,
        microbatch_count=0,
        microbatch_sizes=[],
        optimizer_updates=0,
        supervised_views=0,
        unique_supervision_views=0,
        supervised_pixels=4096,
        mode="empty",
    )

    with pytest.raises(ValueError, match="pixels without supervised work"):
        _record_exact_terminal_constellation(tmp_path, constellation)


def test_final_report_tick_rejects_supervised_work_without_pixels(tmp_path):
    constellation = _valid_adaptive_terminal_constellation()
    constellation["supervised_pixels"] = 0

    with pytest.raises(
        ValueError, match="supervised work without participating pixels"
    ):
        _record_exact_terminal_constellation(tmp_path, constellation)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("fill_ratio", 0.5),
        ("microbatch_count", 1),
        ("microbatch_sizes", [4, 0, 3]),
        ("microbatch_sizes", [5, 2]),
        ("microbatch_sizes", [4, 2]),
        ("optimizer_updates", 2),
        ("supervised_views", 6),
        ("unique_supervision_views", -1),
        ("unique_supervision_views", 8),
        ("supervised_pixels", -1),
        ("realized_batch_size", 8),
        ("realized_camera_count", 3),
    ],
)
def test_final_report_tick_rejects_cross_field_constellation_mismatch(
    tmp_path, field, value
):
    constellation = _valid_adaptive_terminal_constellation()
    constellation[field] = value

    with pytest.raises(ValueError, match="exact terminal supervision constellation"):
        _record_exact_terminal_constellation(tmp_path, constellation)


@pytest.mark.parametrize(
    "field",
    [
        "requested_batch_size",
        "realized_batch_size",
        "requested_camera_count",
        "realized_camera_count",
        "unique_eligible_count",
        "fill_ratio",
        "microbatch_count",
        "optimizer_updates",
        "supervised_views",
        "unique_supervision_views",
        "supervised_pixels",
        "cumulative_realized_supervised_views",
        "cumulative_optimizer_updates",
    ],
)
def test_final_report_tick_rejects_boolean_constellation_numeric(
    tmp_path, field
):
    constellation = _valid_adaptive_terminal_constellation()
    constellation[field] = True

    with pytest.raises(ValueError, match="exact terminal supervision constellation"):
        _record_exact_terminal_constellation(tmp_path, constellation)


@pytest.mark.parametrize(
    "field",
    (
        "source_frame_index",
        "frames_ingested",
        "frames_trained",
        "keyframes_admitted",
        "optimizer_step",
        "supervision_views",
        "mapping_snapshot_id",
        "control_tick_epoch",
        "last_ingest_iter",
        "mapping_idle_threshold_iters",
    ),
)
def test_final_report_tick_rejects_boolean_exact_clock_integer(tmp_path, field):
    with pytest.raises(ValueError, match="exact terminal clock field"):
        _record_exact_terminal_constellation(
            tmp_path,
            _valid_adaptive_terminal_constellation(),
            **{field: True},
        )


def test_final_report_tick_rejects_boolean_microbatch_size(tmp_path):
    constellation = _valid_adaptive_terminal_constellation()
    constellation["microbatch_sizes"] = [4, True, 2]
    constellation["microbatch_count"] = 3

    with pytest.raises(ValueError, match="exact terminal supervision constellation"):
        _record_exact_terminal_constellation(tmp_path, constellation)


def test_final_report_tick_nonstreaming_clocks_stay_explicitly_null(tmp_path):
    cams = [FakeCam(10, 20.0)]
    record_final_report_eval_tick(
        model_path=str(tmp_path),
        iteration=5000,
        report_iteration=5000,
        view_psnr_db=[20.0],
        scored_cams=cams,
        test_cams=cams,
        reported_mean_test_psnr_db=20.0,
    )
    row = read_eval_ticks(str(tmp_path))[-1]
    for field in (
        "source_frame_index",
        "frames_ingested",
        "frames_trained",
        "keyframes_admitted",
        "optimizer_step",
        "supervision_views",
        "wall_clock_s",
        "accumulated_iteration_wall_s",
        "last_ingest_iter",
        "phase",
    ):
        assert row[field] is None
        assert field in row["field_provenance"]
    assert (
        row["clock_snapshot_status"]
        == "not_supplied_nonstreaming_or_legacy_caller"
    )


def test_final_report_tick_refuses_a_mismatched_report_mean(tmp_path):
    """A report whose aggregate disagrees with its per-view tick is unusable."""
    cams = [FakeCam(10), FakeCam(90)]
    with pytest.raises(ValueError, match="mean_test_psnr mismatch"):
        record_final_report_eval_tick(
            model_path=str(tmp_path),
            iteration=5000,
            report_iteration=5000,
            view_psnr_db=[20.0, 30.0],
            scored_cams=cams,
            test_cams=cams,
            reported_mean_test_psnr_db=24.0,
        )
    assert read_eval_ticks(str(tmp_path)) == []


def test_final_report_tick_refuses_a_nonfinite_report_mean(tmp_path):
    cams = [FakeCam(10)]
    with pytest.raises(ValueError, match="not finite"):
        record_final_report_eval_tick(
            model_path=str(tmp_path),
            iteration=5000,
            report_iteration=5000,
            view_psnr_db=[float("inf")],
            scored_cams=cams,
            test_cams=cams,
            reported_mean_test_psnr_db=float("inf"),
        )
    assert read_eval_ticks(str(tmp_path)) == []


def test_final_report_persists_the_exact_tick_binding(tmp_path, monkeypatch):
    """The final report must name the exact #698 row it just measured."""
    # This verifies evaluation-tick/report binding rather than packaged LPIPS
    # weights. The lean CPU-Torch simulation image intentionally has no
    # scientific evaluation assets.
    import lpipsPyTorch

    monkeypatch.setattr(
        lpipsPyTorch,
        "lpips",
        lambda _render, _gt, net_type="vgg": torch.zeros(1),
    )

    class Gaussian:
        def get_xyz(self):
            return torch.zeros(0, 3)

    class Pipe:
        pass

    cams = [FakeCam(10, 20.0), FakeCam(90, 30.0)]
    from eligsir.utils.score_contract import declare_canonical_score_tensor

    for cam in cams:
        cam.original_image, _ = declare_canonical_score_tensor(
            torch.zeros(3, 64, 64), context="test_eval_tick_gt"
        )
    monkeypatch.setattr(comparison_report, "_gt_image", lambda cam: cam.original_image)

    def _render(*_args):
        image, _ = declare_canonical_score_tensor(
            torch.full((3, 64, 64), 0.5), context="test_eval_tick_render"
        )
        return {"render": image}

    constellation = {
        "requested_batch_size": 64,
        "realized_batch_size": 4,
        "requested_camera_count": 8,
        "realized_camera_count": 4,
        "unique_eligible_count": 4,
        "fill_ratio": 4 / 64,
        "microbatch_count": 1,
        "microbatch_sizes": [4],
        "optimizer_updates": 1,
        "supervised_views": 4,
        "unique_supervision_views": 4,
        "supervised_pixels": 16384,
        "cumulative_realized_supervised_views": 9996,
        "cumulative_optimizer_updates": 4998,
        "mode": "true_camera_microbatch_accumulation",
        "reason": "eligible_underfill",
        "hysteresis_state": "stable_hold",
    }
    summary = comparison_report.write_post_training_report(
        model_path=str(tmp_path),
        iteration=2_147_483_647,
        true_final_iteration=5000,
        experiment_validity_is_final=True,
        gaussians=Gaussian(),
        train_cams=[],
        test_cams=cams,
        render_fn=_render,
        pipe=Pipe(),
        background=None,
        evaluation_tick_clock_snapshot={
            "source_frame_index": 90,
            "frames_ingested": 91,
            "frames_trained": 73,
            "keyframes_admitted": 44,
            "optimizer_step": 4998,
            "supervision_views": 9996,
            "mapping_snapshot_id": 44,
            "control_tick_epoch": 12,
            "supervision_constellation": constellation,
            "wall_clock_s": 123.5,
            "accumulated_iteration_wall_s": 118.25,
            "last_ingest_iter": 4700,
            "mapping_idle_threshold_iters": 50000,
        },
    )

    binding = summary["evaluation_tick_binding"]
    assert binding["status"] == "bound"
    assert binding["evaluation_tick_iteration"] == 5000
    with open(summary["output_dir"] + "/report.json") as f:
        assert json.load(f)["evaluation_tick_binding"] == binding
    rows = read_eval_ticks(str(tmp_path))
    assert len(rows) == 1
    assert binding["sha256"] == eval_tick._record_sha256(rows[0])
    assert rows[0]["mean_test_psnr_db"] == pytest.approx(summary["mean_test_psnr"])
    assert rows[0]["optimizer_step"] == 4998
    assert rows[0]["mapping_snapshot_id"] == 44
    assert rows[0]["control_tick_epoch"] == 12
    assert rows[0]["supervision_constellation"] == constellation
    telemetry = summary["evaluation_tick_telemetry"]
    assert telemetry["count"] == 1
    assert telemetry["timed_count"] == 1
    assert telemetry["elapsed_scope"] == eval_tick.RECORD_HOT_PATH_ELAPSED_SCOPE
    assert telemetry["elapsed_ms_p50"] == rows[0]["record_hot_path"]["elapsed_ms"]
    assert telemetry["elapsed_ms_p95"] == rows[0]["record_hot_path"]["elapsed_ms"]
    assert telemetry["serialized_bytes_total"] == rows[0]["record_hot_path"]["serialized_bytes"]
    assert telemetry["plot_status"] == "written"
    assert (tmp_path / "partition_psnr_over_time.svg").exists()


def test_final_report_without_scored_holdout_is_explicitly_ungradeable(tmp_path):
    class Gaussian:
        def get_xyz(self):
            return torch.zeros(0, 3)

    summary = comparison_report.write_post_training_report(
        model_path=str(tmp_path),
        iteration=5000,
        experiment_validity_is_final=True,
        gaussians=Gaussian(),
        train_cams=[],
        test_cams=[],
        render_fn=lambda *_args: {"render": torch.zeros(3, 64, 64)},
        pipe=object(),
        background=None,
    )
    assert summary["evaluation_tick_binding"] == {
        "status": "unavailable",
        "gradeable": False,
        "reason": "final_report_has_no_valid_scored_holdout_metrics",
        "evaluation_tick_iteration": 5000,
        "report_iteration": 5000,
    }
    assert read_eval_ticks(str(tmp_path)) == []


def test_render_failure_rejects_the_entire_tick(tmp_path, monkeypatch):
    """A failed view must not create a failure-selected partial score."""
    def _fake_render_image(cam, gaussians, render_fn, pipe, background):
        if cam.uid == 50:
            raise RuntimeError("render failed")
        return torch.full((3, 4, 4), cam.psnr_value / 100.0)

    _patch_render_pipeline(monkeypatch)
    monkeypatch.setattr(comparison_report, "_render_image", _fake_render_image)
    cams = [FakeCam(10, 20.0), FakeCam(50, 24.0), FakeCam(95, 28.0)]
    tick = record_and_plot(str(tmp_path), 1000, None, cams, None, None, None)
    rec = record_eval_tick(model_path=str(tmp_path), iteration=1000, psnr_tick=tick,
                           test_cams=cams, plot=False)
    assert tick is None
    assert rec is None
    assert read_eval_ticks(str(tmp_path)) == []
    history = json.loads((tmp_path / "psnr_history.jsonl").read_text())
    assert history["eval_valid"] is False
    assert history["n_attempted"] == 3
    assert "render failed" in history["error"]


# ---------------------------------------------------------------------------
# plot
# ---------------------------------------------------------------------------
def _seed_history(tmp_path, n=5):
    for i in range(n):
        red = reduce_partition_psnr_cpu(
            [20.0 + i, 24.0 + i, 28.0 + i], [10, 50, 95], 100, 6)
        append_eval_tick(str(tmp_path), build_eval_tick_record(
            iteration=1000 * (i + 1), mean_test_psnr_db=24.0 + i, reduction=red,
            last_ingest_iter=3000, mapping_idle_threshold_iters=5000))


def test_partition_plot_prefers_cumulative_supervised_views_for_batched_runs():
    rows = [
        {"iteration": 250, "optimizer_step": 250, "supervision_views": 1000,
         "last_ingest_iter": None},
        {"iteration": 500, "optimizer_step": 500, "supervision_views": 2000,
         "last_ingest_iter": 500},
        {"iteration": 750, "optimizer_step": 750, "supervision_views": 3000,
         "last_ingest_iter": 500},
    ]

    axis = _partition_plot_work_axis(rows)

    assert axis.values == [1000, 2000, 3000]
    assert axis.label == "cumulative supervised views"
    assert axis.last_ingest_value == 2000
    assert axis.last_ingest_label == "last ingest (2,000 views)"


def test_partition_plot_falls_back_to_optimizer_updates_not_iterations():
    rows = [
        {"iteration": 250, "optimizer_step": 250, "supervision_views": None,
         "last_ingest_iter": None},
        {"iteration": 500, "optimizer_step": 500, "supervision_views": None,
         "last_ingest_iter": 500},
    ]

    axis = _partition_plot_work_axis(rows)

    assert axis.values == [250, 500]
    assert axis.label == "optimizer updates"
    assert axis.last_ingest_value == 500
    assert axis.last_ingest_label == "last ingest (update 500)"


def test_partition_plot_xlim_matches_data_bounds():
    assert _partition_plot_xlim([1000, 2000, 3000]) == (1000.0, 3000.0)
    assert _partition_plot_xlim([500]) == (500.0, 500.0)
    assert _partition_plot_xlim([]) is None


def test_partition_plot_xlim_widens_to_include_last_ingest_marker():
    # last_ingest_value sits before the first post-mapping evaluation tick
    # (mapping-end supervision count < first tick's) -- the tight bound must
    # widen to keep the marker on-axis, WITH padding so it doesn't render
    # flush against the new axis spine (still invisible in that case).
    lo, hi = _partition_plot_xlim([7000, 8000, 9000], include=6032)
    assert lo < 6032.0 < hi == 9000.0
    # a marker already inside the data range must not narrow the bound
    assert _partition_plot_xlim([1000, 2000, 3000], include=1500) == (1000.0, 3000.0)
    # a marker exactly at the first tick (the common case -- the
    # mapping-endpoint tick is itself usually the first post-mapping tick)
    # STILL needs a small margin: sitting exactly on the axis spine renders
    # it flush against, and indistinguishable from, the solid border.
    lo, hi = _partition_plot_xlim([1000, 2000, 3000], include=1000)
    assert lo < 1000.0 and hi == 3000.0
    # no ticks at all: bound collapses around the marker with a margin
    lo, hi = _partition_plot_xlim([], include=500)
    assert lo < 500.0 < hi


def test_plot_partition_psnr_over_time_has_no_prepadding_before_first_tick(
    tmp_path, monkeypatch
):
    """Regression for the fr1_desk_8060s_..._classic_v1 finding: every tick
    in a real run can legitimately sit at/after last_ingest_iter (the
    evaluation observer defaults to running only ``post_source_completion`` --
    see ``mapping_holdout_curve_metadata``), so matplotlib's default ~5%
    autoscale margin pushed the left axis edge below the first real data
    point and drew an empty region that looked like dropped mapping-phase
    data. The axis must stay close to the real data instead -- NOT
    pixel-exact any more (2026-08-30 follow-up): the last-ingest marker
    coincides with this tick and needs a small, deliberate margin of its own
    to stay visible against the axis spine (see ``_partition_plot_xlim``'s
    ``include`` docstring) -- a few-percent margin drawn for a real, labeled
    marker is not the large unlabeled empty region this regression guards
    against."""

    for i, (iteration, views) in enumerate([(327, 1308), (527, 2108), (727, 2908)]):
        red = reduce_partition_psnr_cpu(
            [20.0 + i, 24.0 + i, 28.0 + i], [10, 50, 95], 100, 6
        )
        append_eval_tick(
            str(tmp_path),
            build_eval_tick_record(
                iteration=iteration,
                mean_test_psnr_db=24.0 + i,
                reduction=red,
                supervision_views=views,
                last_ingest_iter=327,
            ),
        )

    import matplotlib.pyplot as plt

    captured = {}
    real_subplots = plt.subplots

    def _spy_subplots(*args, **kwargs):
        fig, ax = real_subplots(*args, **kwargs)
        captured["ax"] = ax
        return fig, ax

    monkeypatch.setattr(plt, "subplots", _spy_subplots)

    out = plot_partition_psnr_over_time(str(tmp_path))

    assert out is not None and os.path.exists(out)
    xlim = captured["ax"].get_xlim()
    # Left edge gets a small deliberate margin for the last-ingest marker
    # (~3% of the span, per _partition_plot_xlim) -- NOT matplotlib's ~5%
    # default autoscale margin the original incident was about, and nowhere
    # near it: assert an upper bound well short of that to keep this a real
    # regression guard against the large-empty-region failure mode.
    span = 2908.0 - 1308.0
    assert xlim[0] < 1308.0
    assert xlim[0] > 1308.0 - 0.10 * span
    assert xlim[1] == pytest.approx(2908.0)


def test_plot_is_generated_and_deterministic_across_processes(tmp_path):
    """Regenerating from the same JSONL in a FRESH interpreter yields identical
    bytes.

    Same-process byte equality would pass even without the SVG metadata and
    stable element-id configuration; a subprocess run exercises both.
    """
    import subprocess

    _seed_history(tmp_path)
    out1 = plot_partition_psnr_over_time(str(tmp_path))
    assert out1 and os.path.exists(out1)
    with open(out1, "rb") as f:
        first = f.read()
    repo = os.path.join(os.path.dirname(__file__), "..")
    subprocess.run(
        [sys.executable, "-c",
         "import sys; sys.path.insert(0, sys.argv[1]);"
         "from eligsir.utils.eval_tick import plot_partition_psnr_over_time as p;"
         "assert p(sys.argv[2])",
         os.path.abspath(repo), str(tmp_path)],
        check=True,
    )
    with open(out1, "rb") as f:
        second = f.read()
    assert first == second


def test_plot_returns_none_without_history(tmp_path):
    assert plot_partition_psnr_over_time(str(tmp_path)) is None
