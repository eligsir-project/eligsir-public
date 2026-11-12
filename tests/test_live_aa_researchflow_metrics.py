import json
from pathlib import Path

import pytest


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def _make_run(
    tmp_path: Path,
    *,
    mode: str = "antialiased",
    samples: int = 4,
    mapping_steps: int = 1200,
    refinement_steps: int = 200,
    throughput_evidence_valid: bool | None = None,
) -> Path:
    model = tmp_path / "run"
    report = {
        "eval_valid": True,
        "n_test": 19,
        "mean_test_psnr": 22.5,
        "eval_metadata": {"holdout_hash": "holdout-123"},
        "git_provenance": {"git_commit": "abc123", "git_dirty": False},
        "experiment_origin": {
            "researchflow_run_id": "rf-run",
            "researchflow_plan_sha256": "plan-sha",
        },
    }
    _write_json(model / "comparison/mapping_endpoint/report.json", report)
    summary = {
        "optimizer_steps": mapping_steps,
        "supervision_views": 4800,
        "optimizer_steps_per_active_s": 101.5,
        "supervision_views_per_active_s": 406.0,
        "mapping_active_s": 11.82,
        "requested_supervision_constellations": {"B4_C4": mapping_steps},
        "realized_supervision_constellations": {"B4_C4": mapping_steps},
        "supervision_constellation_modes": {"true_camera_batch": mapping_steps},
        "supervision_constellation_reasons": {
            "fixed_constellation_ablation": mapping_steps
        },
    }
    if throughput_evidence_valid is not None:
        summary["throughput_evidence_valid"] = throughput_evidence_valid
        summary["throughput_evidence_status"] = (
            "valid" if throughput_evidence_valid
            else "invalid_mapping_holdout_observer_pause"
        )
    _write_json(model / "causal_replay_summary.json", summary)
    _write_json(
        model / "streaming_phase_metadata.json",
        {
            "mapping_endpoint": {"iteration": 1200},
            "mapping_optimizer_steps": mapping_steps,
            "refinement_optimizer_steps": refinement_steps,
            "refinement_optimizer_steps_requested": refinement_steps,
        },
    )
    rows = []
    for iteration in (400, 800, 1200, 1400)[:samples]:
        rows.append(
            {
                "iteration": iteration,
                "mean_test_psnr": 20.0 + iteration / 1000.0,
                "mean_edge_psnr": 18.0 + iteration / 1000.0,
                "temporal_flicker_rms": 0.02,
                "temporal_flicker_psnr_db": 33.97,
                "temporal_flicker_valid_pairs": 18,
                "temporal_flicker_status": "ok",
                "gaussian_count": 20_000 + iteration,
                "rasterize_mode": mode,
                "training_resolution_tiers": ["native"],
                "evaluation_resolutions": [[640, 480]],
                "rasterizer_gpu_ms": None,
                "rasterizer_timing_status": "requires_stage_gpu_events",
                "mapping_active_s": iteration / 100.0,
                "optimizer_updates": iteration,
                "supervision_views": iteration * 4,
            }
        )
    (model / "psnr_history.jsonl").parent.mkdir(parents=True, exist_ok=True)
    (model / "psnr_history.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    return model


def test_assembles_mapping_endpoint_metrics_and_preserves_full_curve(tmp_path):
    from scripts.assemble_live_aa_researchflow_metrics import assemble_metrics

    model = _make_run(tmp_path)
    metrics = assemble_metrics(
        model,
        expected_rasterize_mode="antialiased",
        expected_git_sha="abc123",
        expected_batch_size=4,
        expected_camera_count=4,
    )

    assert metrics["eval_valid"] is True
    assert metrics["mean_test_psnr"] == 22.5
    assert metrics["psnr_history"]["mapping_endpoint_iteration"] == 1200
    assert metrics["psnr_history"]["mapping_sample_iteration"] == 1200
    assert metrics["psnr_history"]["mapping_sample_gap_updates"] == 0
    assert metrics["psnr_history"]["mean_edge_psnr"] == pytest.approx(19.2)
    assert metrics["psnr_history"]["gaussian_count"] == 21_200
    assert metrics["psnr_history"]["rasterize_mode"] == "antialiased"
    assert metrics["psnr_history"]["rasterizer_timing_status"] == "requires_stage_gpu_events"
    assert len(metrics["psnr_history"]["samples"]) == 4
    assert metrics["psnr_history"]["samples"][-1]["iteration"] == 1400
    assert metrics["throughput"]["optimizer_updates_per_s"] == 101.5
    assert metrics["throughput"]["supervised_views_per_s"] == 406.0
    assert metrics["phase"]["refinement_optimizer_steps"] == 200
    assert metrics["supervision_constellations"]["realized"] == {
        "B4_C4": 1200
    }


def test_rejects_throughput_contaminated_causal_summary(tmp_path):
    from scripts.assemble_live_aa_researchflow_metrics import assemble_metrics

    model = _make_run(tmp_path, throughput_evidence_valid=False)

    with pytest.raises(ValueError, match="throughput evidence is invalid"):
        assemble_metrics(
            model,
            expected_rasterize_mode="antialiased",
            expected_git_sha="abc123",
            expected_batch_size=4,
            expected_camera_count=4,
        )


def test_rejects_too_few_psnr_samples(tmp_path):
    from scripts.assemble_live_aa_researchflow_metrics import assemble_metrics

    model = _make_run(tmp_path, samples=2)
    with pytest.raises(ValueError, match="at least 3 PSNR samples"):
        assemble_metrics(
            model,
            expected_rasterize_mode="antialiased",
            expected_git_sha="abc123",
            expected_batch_size=4,
            expected_camera_count=4,
        )


def test_requires_at_least_1000_realized_updates_across_both_phases(tmp_path):
    from scripts.assemble_live_aa_researchflow_metrics import assemble_metrics

    model = _make_run(tmp_path, mapping_steps=700, refinement_steps=200)
    with pytest.raises(ValueError, match="at least 1000 total optimizer updates"):
        assemble_metrics(
            model,
            expected_rasterize_mode="antialiased",
            expected_git_sha="abc123",
            expected_batch_size=4,
            expected_camera_count=4,
        )


def test_rejects_wrong_raster_mode_or_git_sha(tmp_path):
    from scripts.assemble_live_aa_researchflow_metrics import assemble_metrics

    model = _make_run(tmp_path, mode="classic")
    with pytest.raises(ValueError, match="rasterize mode"):
        assemble_metrics(
            model,
            expected_rasterize_mode="antialiased",
            expected_git_sha="abc123",
            expected_batch_size=4,
            expected_camera_count=4,
        )
    with pytest.raises(ValueError, match="git commit"):
        assemble_metrics(
            model,
            expected_rasterize_mode="classic",
            expected_git_sha="different",
            expected_batch_size=4,
            expected_camera_count=4,
        )


def test_rejects_nominal_b4c4_run_that_realized_underfilled_transactions(tmp_path):
    from scripts.assemble_live_aa_researchflow_metrics import assemble_metrics

    model = _make_run(tmp_path)
    summary_path = model / "causal_replay_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["realized_supervision_constellations"] = {"B1_C1": 1200}
    summary["supervision_constellation_modes"] = {
        "single_camera_immediate": 1200
    }
    _write_json(summary_path, summary)

    with pytest.raises(ValueError, match="realized supervision constellation"):
        assemble_metrics(
            model,
            expected_rasterize_mode="antialiased",
            expected_git_sha="abc123",
            expected_batch_size=4,
            expected_camera_count=4,
        )
