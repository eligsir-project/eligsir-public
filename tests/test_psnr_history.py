"""Unit tests for utils/psnr_history.py's GPU-resident per-view PSNR path
(Partition-Focus v2 Part 2). CPU-only torch, no real rendering -- monkeypatches
_render_image/_gt_image/psnr so the render loop runs against tiny tensors.
"""
import json
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
import torch

import metrics
import eligsir.utils.comparison_report as comparison_report
import eligsir.utils.image_utils as image_utils
import eligsir.utils.psnr_history as psnr_history
import eligsir.utils.psnr_history_io as psnr_history_io
from eligsir.utils.score_contract import declare_canonical_score_tensor
from eligsir.utils.psnr_history import (
    PsnrTickResult,
    append_psnr_history,
    compute_mean_test_psnr,
    compute_per_view_test_psnr,
    compute_per_view_test_psnr_gpu,
)


class FakeCam:
    def __init__(self, uid, psnr_value):
        self.uid = uid
        self.psnr_value = psnr_value


def _patch_render_pipeline(monkeypatch, sync_counter=None):
    def _fake_render_image(cam, gaussians, render_fn, pipe, background):
        return torch.zeros(3, 4, 4)

    def _fake_gt_image(cam):
        gt, _ = declare_canonical_score_tensor(
            torch.zeros(3, 4, 4), context="test_psnr_history_gt"
        )
        return gt

    def _fake_psnr(render_img, gt):
        val = torch.tensor(render_img.psnr_value if hasattr(render_img, "psnr_value") else 0.0)
        return val

    # psnr(...).mean() is what's stacked; return a 0-d tensor per cam encoding
    # a known value via the render_img we control per-call.
    def _fake_render_image_with_value(cam, gaussians, render_fn, pipe, background):
        return torch.full((3, 4, 4), cam.psnr_value / 100.0)

    monkeypatch.setattr(comparison_report, "_render_image", _fake_render_image_with_value)
    monkeypatch.setattr(comparison_report, "_gt_image", _fake_gt_image)

    def _psnr_fn(render_img, gt, **_kwargs):
        class _Scalar:
            def mean(self_inner):
                v = render_img.reshape(-1)[0] * 100.0
                if sync_counter is not None:
                    orig_item = v.item
                    def _counted_item():
                        sync_counter["item_calls"] += 1
                        return orig_item()
                    v.item = _counted_item
                return v
        return _Scalar()

    monkeypatch.setattr(image_utils, "psnr", _psnr_fn)
    # The canonical history path imports the validated score entry point
    # directly. Patch that owned boundary as well so this unit fixture keeps
    # testing aggregation without bypassing production score validation.
    monkeypatch.setattr(psnr_history, "scientific_psnr", _psnr_fn)


def test_compute_per_view_test_psnr_gpu_returns_stacked_tensor(monkeypatch):
    _patch_render_pipeline(monkeypatch)
    cams = [FakeCam(0, 20.0), FakeCam(1, 25.0), FakeCam(2, 15.0)]
    got_cams, tensor = compute_per_view_test_psnr_gpu(None, cams, None, None, None)
    assert got_cams == cams
    assert tensor.tolist() == pytest.approx([20.0, 25.0, 15.0])


def test_compute_per_view_test_psnr_gpu_no_per_view_item_calls(monkeypatch):
    sync_counter = {"item_calls": 0}
    _patch_render_pipeline(monkeypatch, sync_counter=sync_counter)
    cams = [FakeCam(i, float(i)) for i in range(5)]
    compute_per_view_test_psnr_gpu(None, cams, None, None, None)
    assert sync_counter["item_calls"] == 0


def test_history_and_exact_report_use_the_same_rgb_psnr_for_same_cohort(monkeypatch):
    """The history producer and persisted-pair reporter must agree per view."""
    gt = torch.full((3, 2, 2), 0.5)
    channel_error = torch.tensor([0.05, 0.05, 0.0576285]).view(3, 1, 1)
    raw_pairs = {
        "view_a": (gt - channel_error, gt),
        "view_b": (gt - 2.0 * channel_error, gt),
    }
    cams = [SimpleNamespace(uid=name, image_name=name) for name in raw_pairs]

    def _render(cam, *_args):
        return raw_pairs[cam.image_name][0]

    def _gt(cam):
        tensor, _ = declare_canonical_score_tensor(
            raw_pairs[cam.image_name][1], context=f"history_gt:{cam.image_name}"
        )
        return tensor

    monkeypatch.setattr(comparison_report, "_render_image", _render)
    monkeypatch.setattr(comparison_report, "_gt_image", _gt)
    _cams, history = compute_per_view_test_psnr_gpu(None, cams, None, None, None)
    _provenance, endpoint, _clamp = metrics.score_exact_pairs(
        [
            {"artifact": name, "view_name": name}
            for name in raw_pairs
        ],
        raw_pairs,
        device="cpu",
        lpips_fn=lambda render, gt: (render - gt).abs().mean(),
    )

    assert float(history.mean()) == pytest.approx(
        endpoint["aggregate"]["psnr"], abs=1e-6
    )


def test_history_rejects_appending_rgb_psnr_to_legacy_chw_curve(tmp_path):
    (tmp_path / "psnr_history.jsonl").write_text(
        json.dumps({"iteration": 1, "mean_test_psnr": 20.0}) + "\n"
    )

    with pytest.raises(ValueError, match="legacy_chw_channel_mean_v1"):
        append_psnr_history(str(tmp_path), 2, 21.0)

    assert json.loads((tmp_path / "psnr_history.jsonl").read_text()) == {
        "iteration": 1,
        "mean_test_psnr": 20.0,
    }


def test_history_append_uses_bounded_tail_admission_without_rescanning_rows(
    monkeypatch, tmp_path
):
    append_psnr_history(str(tmp_path), 1, 20.0)

    def _full_scan_is_not_an_append_cost(*_args, **_kwargs):
        raise AssertionError("append must not scan the full history")

    monkeypatch.setattr(psnr_history_io, "_score_rows", _full_scan_is_not_an_append_cost)
    append_psnr_history(str(tmp_path), 2, 21.0)

    rows = [
        json.loads(line)
        for line in (tmp_path / "psnr_history.jsonl").read_text().splitlines()
    ]
    assert [row["mean_test_psnr"] for row in rows] == [20.0, 21.0]


def test_legacy_history_append_records_an_invalid_observation_without_blending(
    monkeypatch, tmp_path
):
    _patch_render_pipeline(monkeypatch)
    (tmp_path / "psnr_history.jsonl").write_text(
        json.dumps({"iteration": 1, "mean_test_psnr": 20.0}) + "\n"
    )

    result = psnr_history.record_and_plot(
        str(tmp_path), 2, None, [FakeCam(0, 20.0)], None,
        SimpleNamespace(rasterize_mode="classic"), None,
    )

    rows = [
        json.loads(line)
        for line in (tmp_path / "psnr_history.jsonl").read_text().splitlines()
    ]
    assert result is None
    assert rows[0] == {"iteration": 1, "mean_test_psnr": 20.0}
    assert rows[1]["eval_valid"] is False
    assert "fresh model/output path" in rows[1]["error"]


def test_control_scorer_retains_legacy_value_consumed_by_partition_control(monkeypatch):
    from eligsir.utils.partition_control import PartitionControlRuntime
    from eligsir.utils.score_contract import scientific_psnr

    gt = torch.full((3, 2, 2), 0.5)
    error = torch.tensor([0.05, 0.05, 0.0576285]).view(3, 1, 1)
    cam = SimpleNamespace(uid=1, image_name="control")
    gt_score, _ = declare_canonical_score_tensor(gt, context="control_gt")
    monkeypatch.setattr(comparison_report, "_render_image", lambda *_args: gt - error)
    monkeypatch.setattr(comparison_report, "_gt_image", lambda _cam: gt_score)

    _cams, scores = compute_per_view_test_psnr_gpu(
        None, [cam], None, None, None, score_fn=scientific_psnr
    )

    assert float(scores.item()) == pytest.approx(25.609484, abs=1e-6)
    tick = PsnrTickResult(cams=[cam], psnr_tensor=scores, mean_psnr=float(scores.item()))
    assert PartitionControlRuntime.observations_from_tick(tick)[1] == pytest.approx(
        25.609484, abs=1e-6
    )


def test_compute_per_view_test_psnr_clamps_scoring_and_hook_copies(monkeypatch):
    cam = FakeCam(0, 20.0)
    raw = torch.zeros(3, 4, 4)
    raw[0, 0, 0] = 1.1
    gt, _ = declare_canonical_score_tensor(
        torch.zeros_like(raw), context="test_psnr_history_overshoot_gt"
    )
    seen = {}

    monkeypatch.setattr(comparison_report, "_render_image", lambda *_args: raw)
    monkeypatch.setattr(comparison_report, "_gt_image", lambda _cam: gt)

    def _scientific_psnr(render_score, gt_score, **_kwargs):
        seen["score_max"] = float(render_score.max().item())
        return torch.tensor(20.0)

    def _hook(_cam, hook_render, _hook_gt):
        seen["hook_max"] = float(hook_render.max().item())

    monkeypatch.setattr(psnr_history, "scientific_psnr", _scientific_psnr)
    records = []
    cams, scores = compute_per_view_test_psnr_gpu(
        None, [cam], None, None, None,
        per_view_hook=_hook,
        clamp_records_out=records,
    )

    assert cams == [cam]
    assert scores.tolist() == [20.0]
    assert seen == {"score_max": 1.0, "hook_max": 1.0}
    assert records[0]["render"]["samples_changed"] == 1


def test_compute_per_view_test_psnr_wrapper_returns_floats(monkeypatch):
    _patch_render_pipeline(monkeypatch)
    cams = [FakeCam(0, 20.0), FakeCam(1, 25.0)]
    out = compute_per_view_test_psnr(None, cams, None, None, None)
    assert out == [(cams[0], 20.0), (cams[1], 25.0)]


def test_compute_mean_test_psnr(monkeypatch):
    _patch_render_pipeline(monkeypatch)
    cams = [FakeCam(0, 10.0), FakeCam(1, 20.0), FakeCam(2, 30.0)]
    mean = compute_mean_test_psnr(None, cams, None, None, None)
    assert mean == 20.0


def test_record_and_plot_returns_psnr_tick_result(monkeypatch, tmp_path):
    _patch_render_pipeline(monkeypatch)
    from eligsir.utils.psnr_history import record_and_plot

    cams = [FakeCam(0, 10.0), FakeCam(1, 30.0)]
    result = record_and_plot(
        str(tmp_path),
        100,
        None,
        cams,
        None,
        SimpleNamespace(rasterize_mode="antialiased"),
        None,
        metric_context={
            "training_resolution_tiers": ["half"],
            "optimizer_updates": 50,
            "gaussian_count": 1234,
        },
    )
    assert isinstance(result, PsnrTickResult)
    assert result.cams == cams
    assert result.psnr_tensor.tolist() == pytest.approx([10.0, 30.0])
    assert result.mean_psnr == 20.0
    assert result.mean_edge_psnr is not None
    assert os.path.exists(os.path.join(tmp_path, "psnr_history.jsonl"))
    with open(os.path.join(tmp_path, "psnr_history.jsonl")) as f:
        row = json.loads(f.readline())
    assert row["score_domain_clamp"]["render"]["samples_total"] == 96
    assert row["score_domain_clamp"]["render"]["samples_changed"] == 0
    assert row["psnr_reduction"] == "rgb_per_view_v2"
    assert row["rasterize_mode"] == "antialiased"
    assert row["training_resolution_tiers"] == ["half"]
    assert row["optimizer_updates"] == 50
    assert row["gaussian_count"] == 1234
    assert row["mean_edge_psnr"] == pytest.approx(result.mean_edge_psnr)
    assert row["edge_score_scope"] == "sobel_gradient_same_evaluation_resolution"
    assert row["temporal_flicker_status"] == "unavailable_pose_depth_overlap"
    assert row["temporal_flicker_valid_pairs"] == 0


def test_mapping_observer_pause_metadata_is_persisted_in_psnr_row(
    monkeypatch, tmp_path
):
    _patch_render_pipeline(monkeypatch)
    from eligsir.utils.psnr_history import record_and_plot

    result = record_and_plot(
        str(tmp_path),
        100,
        None,
        [FakeCam(0, 20.0)],
        None,
        SimpleNamespace(rasterize_mode="classic"),
        None,
        metric_context={
            "evaluation_phase": "mapping",
            "evaluation_observer_execution": "synchronous_mapping_thread_pause",
            "mapping_execution_paused": True,
            "throughput_evidence_valid": False,
            "throughput_evidence_status": (
                "invalid_mapping_holdout_observer_pause"
            ),
            "evaluation_cadence_clock": "cumulative_optimizer_updates",
        },
    )

    assert isinstance(result, PsnrTickResult)
    row = json.loads((tmp_path / "psnr_history.jsonl").read_text().splitlines()[0])
    assert row["evaluation_phase"] == "mapping"
    assert row["mapping_execution_paused"] is True
    assert row["throughput_evidence_valid"] is False
    assert row["throughput_evidence_status"] == (
        "invalid_mapping_holdout_observer_pause"
    )


def test_temporal_flicker_aggregates_sse_and_samples_not_pair_means(
    monkeypatch, tmp_path
):
    _patch_render_pipeline(monkeypatch)
    import eligsir.utils.live_antialiasing_metrics as aa_metrics

    results = iter(
        (
            aa_metrics.TemporalFlickerResult.from_sufficient_statistics(
                squared_error_sum=torch.tensor(3.0), valid_pixels=torch.tensor(1)
            ),
            aa_metrics.TemporalFlickerResult.from_sufficient_statistics(
                squared_error_sum=torch.tensor(0.0), valid_pixels=torch.tensor(9)
            ),
        )
    )
    monkeypatch.setattr(
        aa_metrics, "camera_pair_temporal_flicker", lambda *_args: next(results)
    )
    cams = [FakeCam(i, 20.0 + i) for i in range(3)]

    result = psnr_history.record_and_plot(
        str(tmp_path), 100, None, cams, None,
        SimpleNamespace(rasterize_mode="antialiased"), None,
    )

    # Three channels per valid pixel: global MSE = 3 / (3 * 10) = 0.1.
    assert result.temporal_flicker_rms == pytest.approx(0.1**0.5)
    assert result.temporal_flicker_psnr_db == pytest.approx(10.0)
    assert result.temporal_flicker_valid_pairs == 2


def test_record_and_plot_rejects_entire_tick_on_nonfinite_view(monkeypatch, tmp_path):
    cams = [FakeCam(0, 20.0), FakeCam(1, 30.0)]

    def _render(cam, *_args):
        image = torch.zeros(3, 4, 4)
        if cam.uid == 1:
            image[0, 0, 0] = float("inf")
        return image

    monkeypatch.setattr(comparison_report, "_render_image", _render)
    def _gt(_cam):
        gt, _ = declare_canonical_score_tensor(
            torch.zeros(3, 4, 4), context="test_psnr_history_nonfinite_gt"
        )
        return gt

    monkeypatch.setattr(comparison_report, "_gt_image", _gt)

    result = psnr_history.record_and_plot(
        str(tmp_path), 100, None, cams, None, None, None
    )

    assert result is None
    with open(tmp_path / "psnr_history.jsonl") as f:
        row = json.loads(f.readline())
    assert row["eval_valid"] is False
    assert row["n_attempted"] == 2
    assert row["mean_test_psnr"] is None


def test_empty_test_cams_returns_none_or_empty(monkeypatch):
    _patch_render_pipeline(monkeypatch)
    cams, tensor = compute_per_view_test_psnr_gpu(None, [], None, None, None)
    assert cams == []
    assert tensor is None
    assert compute_per_view_test_psnr(None, [], None, None, None) == []
    assert compute_mean_test_psnr(None, [], None, None, None) is None


def test_check_psnr_plateau_false_before_enough_history(tmp_path):
    from eligsir.utils.psnr_history import append_psnr_history, check_psnr_plateau

    for i, v in enumerate([10.0, 15.0, 20.0]):
        append_psnr_history(str(tmp_path), i * 100, v)
    assert check_psnr_plateau(str(tmp_path), patience_ticks=5, min_delta_db=0.1) is False


def test_check_psnr_plateau_true_when_flat(tmp_path):
    from eligsir.utils.psnr_history import append_psnr_history, check_psnr_plateau

    for i, v in enumerate([10.0, 24.95, 24.97, 25.00, 24.96, 24.98]):
        append_psnr_history(str(tmp_path), i * 100, v)
    assert check_psnr_plateau(str(tmp_path), patience_ticks=5, min_delta_db=0.1) is True


def test_check_psnr_plateau_false_when_still_climbing(tmp_path):
    from eligsir.utils.psnr_history import append_psnr_history, check_psnr_plateau

    for i, v in enumerate([10.0, 15.0, 19.0, 22.0, 24.0, 25.5]):
        append_psnr_history(str(tmp_path), i * 100, v)
    assert check_psnr_plateau(str(tmp_path), patience_ticks=5, min_delta_db=0.1) is False


def test_check_psnr_plateau_true_on_noisy_but_flat_real_run_tail(tmp_path):
    """Real psnr_history tail (iter 190000-194750) from a 200k-iteration
    rtx3090 run: genuinely flat/no-trend, but noisy enough tick-to-tick that
    a raw max-min range over the window (0.18dB) exceeds a 0.1dB threshold
    -- the exact case that motivated switching to a half-window-mean
    comparison. The smoothed check must fire here even though a raw range
    check would not."""
    from eligsir.utils.psnr_history import append_psnr_history, check_psnr_plateau

    real_tail = [
        21.2696533203125, 21.282833099365234, 21.320693969726562,
        21.277873992919922, 21.253644943237305, 21.238107681274414,
        21.226512908935547, 21.22001075744629, 21.259618759155273,
        21.256855010986328, 21.216341018676758, 21.144428253173828,
        21.193246841430664, 21.229921340942383, 21.242961883544922,
        21.258405685424805, 21.192110061645508, 21.26154327392578,
        21.258146286010742, 21.285139083862305,
    ]
    assert (max(real_tail) - min(real_tail)) > 0.1, "fixture must reproduce the noisy-range case"
    for i, v in enumerate(real_tail):
        append_psnr_history(str(tmp_path), i * 250, v)
    assert check_psnr_plateau(str(tmp_path), patience_ticks=20, min_delta_db=0.1) is True


def test_check_psnr_plateau_disabled_when_patience_zero(tmp_path):
    from eligsir.utils.psnr_history import append_psnr_history, check_psnr_plateau

    append_psnr_history(str(tmp_path), 0, 25.0)
    assert check_psnr_plateau(str(tmp_path), patience_ticks=0, min_delta_db=0.1) is False


def test_scatter_reduce_partition_mean_matches_numpy():
    """Fixture mirroring train_streaming.py's partition-focus scatter-add
    path: known per-view PSNR + partition assignment, GPU segment-mean must
    match a manual numpy groupby-mean."""
    psnr_tensor = torch.tensor([10.0, 20.0, 30.0, 5.0, 15.0])
    partition_index = torch.tensor([0, 0, 1, 1, 2], dtype=torch.long)
    k = 3
    counts_t = torch.zeros(k)
    counts_t.scatter_add_(0, partition_index, torch.ones_like(psnr_tensor))
    sums_t = torch.zeros(k)
    sums_t.scatter_add_(0, partition_index, psnr_tensor)
    means = (sums_t / counts_t.clamp_min(1.0)).tolist()

    np_psnr = np.array([10.0, 20.0, 30.0, 5.0, 15.0])
    np_part = np.array([0, 0, 1, 1, 2])
    expected = [np_psnr[np_part == p].mean() for p in range(k)]
    assert means == expected
