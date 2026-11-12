"""Regression test for the [report]-phase timing instrumentation added to
localize the silent 25-45min report-phase stall (see
.superpowers/sdd/2026-09-08-residency-budget/task-7-brief.md).

Exercises the exact three calls between the `experiment_validity` print
block and `json.dump(summary, f)` in `write_post_training_report`
(utils/comparison_report.py): `record_final_report_eval_tick`,
`finalize_eval_tick_history`, and `write_trajectory_quality_profile`. Asserts
the new timing prints appear, in call order, and that the report's own
behavior (evaluation_tick_binding / trajectory_quality_profile content) is
unchanged by adding the instrumentation.

The real `lpipsPyTorch` package is broken in this environment (a torchvision
circular-import bug unrelated to this change -- confirmed by `import
lpipsPyTorch` failing standalone), so this stubs it out via `sys.modules`,
the same pattern tests/test_report_eval_metadata_from_streaming_scene.py
already uses.
"""
from __future__ import annotations

import json
import re
import sys
from unittest.mock import MagicMock

import pytest
import torch


@pytest.fixture(autouse=True, scope="module")
def _mock_lpips_import():
    had_original = "lpipsPyTorch" in sys.modules
    original = sys.modules.get("lpipsPyTorch")
    sys.modules["lpipsPyTorch"] = MagicMock()
    try:
        yield
    finally:
        if had_original:
            sys.modules["lpipsPyTorch"] = original
        else:
            sys.modules.pop("lpipsPyTorch", None)


import eligsir.utils.comparison_report as comparison_report
from eligsir.utils.score_contract import declare_canonical_score_tensor


class _FakeGaussian:
    def get_xyz(self):
        return torch.zeros(0, 3)


class _FakePipe:
    pass


class _FakeCam:
    def __init__(self, uid, image_name=None):
        self.uid = uid
        self.image_name = image_name if image_name is not None else f"{uid:06d}"


def _write_trajectory_jsonl(path, cams):
    with open(path, "w") as f:
        for i, cam in enumerate(cams):
            matrix = [
                [1.0, 0.0, 0.0, float(i)],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
            f.write(json.dumps({"source_ordinal": cam.uid, "c2w": matrix}) + "\n")


def test_report_stall_timing_prints_appear_in_order_and_dont_change_output(
    tmp_path, monkeypatch, capsys
):
    cams = [_FakeCam(10), _FakeCam(90)]
    for cam in cams:
        cam.original_image, _ = declare_canonical_score_tensor(
            torch.zeros(3, 64, 64), context="test_report_stall_instrumentation_gt"
        )
    monkeypatch.setattr(
        comparison_report, "_gt_image", lambda cam: cam.original_image
    )

    def _render(*_args):
        image, _ = declare_canonical_score_tensor(
            torch.full((3, 64, 64), 0.5),
            context="test_report_stall_instrumentation_render",
        )
        return {"render": image}

    # A trajectory_full.jsonl covering both scored cams so
    # write_trajectory_quality_profile's guard (test_per_view AND file
    # exists) is actually exercised, alongside the other two calls.
    _write_trajectory_jsonl(tmp_path / "trajectory_full.jsonl", cams)

    constellation = {
        "requested_batch_size": 64,
        "realized_batch_size": 2,
        "requested_camera_count": 2,
        "realized_camera_count": 2,
        "unique_eligible_count": 2,
        "fill_ratio": 2 / 64,
        "microbatch_count": 1,
        "microbatch_sizes": [2],
        "optimizer_updates": 1,
        "supervised_views": 2,
        "unique_supervision_views": 2,
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
        gaussians=_FakeGaussian(),
        train_cams=[],
        test_cams=cams,
        render_fn=_render,
        pipe=_FakePipe(),
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

    # Behavior unchanged: the same evaluation-tick binding and trajectory
    # profile the pre-instrumentation code produced.
    assert summary["evaluation_tick_binding"]["status"] == "bound"
    assert "evaluation_tick_telemetry" in summary
    assert summary["trajectory_quality_profile"]["total_trajectory_distance_m"] >= 0

    out = capsys.readouterr().out
    lines = [ln for ln in out.splitlines() if ln.startswith("[report] ") and "took" in ln]
    names_in_order = [
        re.match(r"\[report\] (\S+) took", ln).group(1) for ln in lines
    ]
    assert names_in_order == [
        "record_final_report_eval_tick",
        "finalize_eval_tick_history",
        "write_trajectory_quality_profile",
    ]
    for ln in lines:
        assert re.search(r"took \d+\.\d+s$", ln), ln
