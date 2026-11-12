"""Contracts for optional report presentation without metric deferral."""
from __future__ import annotations

import sys
import json
from unittest.mock import MagicMock

import pytest
import torch

from eligsir.utils.score_contract import declare_canonical_score_tensor


@pytest.fixture(autouse=True)
def _mock_lpips_import():
    original = sys.modules.get("lpipsPyTorch")
    sys.modules["lpipsPyTorch"] = MagicMock()
    try:
        yield
    finally:
        if original is None:
            sys.modules.pop("lpipsPyTorch", None)
        else:
            sys.modules["lpipsPyTorch"] = original


def test_deferred_presentation_keeps_metric_rows_but_marks_artifacts_incomplete(
    tmp_path, monkeypatch
):
    import eligsir.utils.comparison_report as report

    class _Cam:
        def __init__(self, uid: int):
            self.uid = uid
            self.image_name = f"view-{uid}"
            self.original_image, _ = declare_canonical_score_tensor(
                torch.zeros(3, 4, 4),
                context="test_report_presentation_boundary:gt",
            )

    class _Gaussians:
        def get_xyz(self):
            return torch.zeros(0, 3)

    cams = [_Cam(1), _Cam(2)]
    render_calls = []

    def render_fn(cam, *_args, **kwargs):
        render_calls.append((cam.uid, dict(kwargs)))
        image, _ = declare_canonical_score_tensor(
            torch.full((3, 4, 4), 0.5),
            context="test_report_presentation_boundary:render",
        )
        return {"render": image}

    monkeypatch.setenv("SPLAT_REPORT_PRESENTATION", "0")
    monkeypatch.setenv("SPLAT_REPORT_OVERVIEW", "1")
    monkeypatch.setenv("SPLAT_REPORT_VIDEOS", "1")
    monkeypatch.setattr(report, "_gt_image", lambda camera: camera.original_image)

    summary = report.write_post_training_report(
        model_path=str(tmp_path),
        iteration=7,
        gaussians=_Gaussians(),
        train_cams=[],
        test_cams=cams,
        render_fn=render_fn,
        pipe=object(),
        background=None,
    )

    assert summary["n_test_metrics"] == 2
    presentation = summary["presentation_artifacts"]
    assert presentation["status"] == "deferred"
    assert presentation["reason"] == "SPLAT_REPORT_PRESENTATION=0"
    assert presentation["regeneration"] == {
        "status": "pending",
        "reason": "no_frozen_presentation_snapshot_or_live_writer",
    }
    assert presentation["inputs"].items() >= {
        "report_iteration": 7,
        "n_train": 0,
        "n_test": 2,
        "scored_source_indices": [1, 2],
    }.items()
    assert presentation["inputs"]["scored_holdout_hash"]
    assert not (tmp_path / "comparison" / "iter_7" / "test_contact_sheet.png").exists()
    assert not list((tmp_path / "comparison" / "iter_7" / "test").glob("*.png"))
    # Deferring presentation does not defer metrics or add another metric pass.
    assert [uid for uid, _kwargs in render_calls] == [1, 2]

    monkeypatch.setenv("SPLAT_REPORT_PRESENTATION", "1")
    presented = report.write_post_training_report(
        model_path=str(tmp_path),
        iteration=7,
        subdir="presented",
        gaussians=_Gaussians(),
        train_cams=[],
        test_cams=cams,
        render_fn=render_fn,
        pipe=object(),
        background=None,
    )
    for key in ("mean_test_psnr", "mean_test_ssim", "mean_test_lpips", "test_per_view"):
        assert summary[key] == presented[key]
    assert [row["source_index"] for row in summary["test_per_view"]] == [1, 2]


def test_final_report_records_all_three_report_phase_boundaries(tmp_path, monkeypatch):
    import eligsir.utils.comparison_report as report

    class _Cam:
        def __init__(self, uid: int):
            self.uid = uid
            self.image_name = f"view-{uid}"
            self.original_image, _ = declare_canonical_score_tensor(
                torch.zeros(3, 4, 4), context="test_report_phase_boundaries:gt"
            )

    class _Gaussians:
        def get_xyz(self):
            return torch.zeros(0, 3)

    cams = [_Cam(10), _Cam(90)]
    with open(tmp_path / "trajectory_full.jsonl", "w") as f:
        for cam in cams:
            f.write(json.dumps({
                "source_ordinal": cam.uid,
                "c2w": [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
            }) + "\n")

    def render_fn(*_args, **_kwargs):
        image, _ = declare_canonical_score_tensor(
            torch.full((3, 4, 4), 0.5),
            context="test_report_phase_boundaries:render",
        )
        return {"render": image}

    monkeypatch.setenv("SPLAT_REPORT_PRESENTATION", "0")
    monkeypatch.setattr(report, "_gt_image", lambda camera: camera.original_image)
    summary = report.write_post_training_report(
        model_path=str(tmp_path),
        iteration=2_147_483_647,
        true_final_iteration=5000,
        experiment_validity_is_final=True,
        gaussians=_Gaussians(),
        train_cams=[],
        test_cams=cams,
        render_fn=render_fn,
        pipe=object(),
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
            "supervision_constellation": {
                "requested_batch_size": 64, "realized_batch_size": 2,
                "requested_camera_count": 2, "realized_camera_count": 2,
                "unique_eligible_count": 2, "fill_ratio": 2 / 64,
                "microbatch_count": 1, "microbatch_sizes": [2],
                "optimizer_updates": 1, "supervised_views": 2,
                "unique_supervision_views": 2, "supervised_pixels": 32,
                "cumulative_realized_supervised_views": 9996,
                "cumulative_optimizer_updates": 4998,
                "mode": "true_camera_microbatch_accumulation",
                "reason": "full", "hysteresis_state": "stable_hold",
            },
            "wall_clock_s": 123.5, "accumulated_iteration_wall_s": 118.25,
            "last_ingest_iter": 4700, "mapping_idle_threshold_iters": 50000,
        },
    )

    assert list(summary["report_phase_timings"]) == [
        "record_final_report_eval_tick",
        "finalize_eval_tick_history",
        "write_trajectory_quality_profile",
    ]
    assert all(
        phase["status"] == "completed" and phase["duration_s"] >= 0
        for phase in summary["report_phase_timings"].values()
    )


def test_opt_in_allocator_samples_cover_report_boundaries_without_a_global_peak_probe(
    tmp_path, monkeypatch
):
    """The real report entry must label bounded current samples, not reset/read peaks.

    Removing any boundary, probing while the option is off, or reporting a
    current sample as a global allocator peak must break this contract.
    """
    import eligsir.utils.comparison_report as report
    from eligsir.utils.gpu_memory_telemetry import GPUMemoryResidencyTracker

    class _Cam:
        uid = 1
        image_name = "view-1"

        def __init__(self):
            self.original_image, _ = declare_canonical_score_tensor(
                torch.zeros(3, 4, 4), context="allocator_phase:gt"
            )

    class _Gaussians:
        def get_xyz(self):
            return torch.zeros(0, 3)

    tracker = GPUMemoryResidencyTracker(
        device="cpu",
        total_bytes_fn=lambda: 100,
        reserved_bytes_fn=lambda: 160,
        peak_bytes_fn=lambda: (_ for _ in ()).throw(AssertionError("no peak probe")),
    )
    monkeypatch.setenv("SPLAT_REPORT_PRESENTATION", "0")
    monkeypatch.setenv("SPLAT_REPORT_ALLOCATOR_PHASE_SAMPLES", "1")
    monkeypatch.setattr(report, "_gt_image", lambda camera: camera.original_image)

    summary = report.write_post_training_report(
        model_path=str(tmp_path),
        iteration=7,
        gaussians=_Gaussians(),
        train_cams=[],
        test_cams=[_Cam()],
        render_fn=lambda *_args, **_kwargs: {"render": torch.full((3, 4, 4), 0.5)},
        pipe=object(),
        background=None,
        allocator_phase_tracker=tracker,
    )

    phases = summary["report_allocator_phases"]
    assert phases["enabled"] is True
    assert list(phases["samples"]) == [
        "report_entry",
        "scored_metrics_complete",
        "presentation_complete",
        "presentation_cache_released",
    ]
    assert all(
        sample["current_allocated_bytes"] == 100
        and sample["current_reserved_bytes"] == 160
        and sample["torch_global_peak_allocated_bytes"] is None
        and sample["torch_global_peak_reason"] == "not_read_by_phase_sample"
        for sample in phases["samples"].values()
    )


def test_cache_release_sample_precedes_geometry_allocation(tmp_path, monkeypatch):
    """The cache-release label must not include later geometry allocations."""
    import eligsir.utils.comparison_report as report
    import eligsir.utils.scene_geometry_metrics as geometry

    # The synthetic tracker starts at the post-clear state. Geometry then
    # changes it; the named boundary must retain the pre-geometry value.
    state = {"bytes": 100}

    class _Tracker:
        def phase_snapshot(self, phase, **_kwargs):
            return {"phase": phase, "status": "measured", "current_allocated_bytes": state["bytes"]}

        def report_block(self):
            return {}

    class _Gaussians:
        def get_xyz(self):
            return torch.zeros(0, 3)

    monkeypatch.setenv("SPLAT_REPORT_PRESENTATION", "0")
    monkeypatch.setenv("SPLAT_REPORT_ALLOCATOR_PHASE_SAMPLES", "1")
    monkeypatch.setattr(geometry, "shape_stats_from_model", lambda _model: state.update(bytes=700) or {})
    monkeypatch.setattr(geometry, "lifecycle_stats_from_model", lambda _model: {})
    monkeypatch.setattr(geometry, "gaussian_belief_summary_from_model", lambda _model: {})

    summary = report.write_post_training_report(
        model_path=str(tmp_path), iteration=7, gaussians=_Gaussians(),
        train_cams=[], test_cams=[], render_fn=lambda *_args, **_kwargs: {},
        pipe=object(), background=None, allocator_phase_tracker=_Tracker(),
    )

    assert summary["report_allocator_phases"]["samples"]["presentation_cache_released"]["current_allocated_bytes"] == 100


def test_allocator_phase_snapshot_failure_is_not_reported_as_available(tmp_path, monkeypatch):
    import eligsir.utils.comparison_report as report

    class _Tracker:
        def phase_snapshot(self, *_args, **_kwargs):
            raise RuntimeError("snapshot unavailable")

        def report_block(self):
            return {}

    class _Gaussians:
        def get_xyz(self):
            return torch.zeros(0, 3)

    monkeypatch.setenv("SPLAT_REPORT_PRESENTATION", "0")
    monkeypatch.setenv("SPLAT_REPORT_ALLOCATOR_PHASE_SAMPLES", "1")
    summary = report.write_post_training_report(
        model_path=str(tmp_path), iteration=7, gaussians=_Gaussians(),
        train_cams=[], test_cams=[], render_fn=lambda *_args, **_kwargs: {},
        pipe=object(), background=None, allocator_phase_tracker=_Tracker(),
    )

    phases = summary["report_allocator_phases"]
    assert phases["available"] is False
    assert phases["reason"] == "RuntimeError: snapshot unavailable"
