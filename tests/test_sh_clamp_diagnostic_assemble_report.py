import json
import os
import tempfile

from scripts.sh_clamp_diagnostic.assemble_report import assemble_report, verdict_for_delta_psnr


def test_verdict_thresholds():
    assert verdict_for_delta_psnr(0.05) == "benign"
    assert verdict_for_delta_psnr(0.5) == "renderer-specific-but-legitimate"
    assert verdict_for_delta_psnr(2.0) == "serious portability concern"


def test_assemble_report_handles_missing_phase1_and_phase2():
    with tempfile.TemporaryDirectory() as tmp:
        phase0_path = os.path.join(tmp, "phase0.json")
        phase3_path = os.path.join(tmp, "phase3.json")
        with open(phase0_path, "w") as f:
            json.dump({
                "model_path": "/data/ckpt", "n_cameras": 10,
                "raw_frac_gt_1_0": 0.1, "weighted_frac_gt_1_0": 0.1,
                "raw_frac_gt_1_5": 0.0, "weighted_frac_gt_1_5": 0.0,
            }, f)
        with open(phase3_path, "w") as f:
            json.dump({
                "psnr_export_vs_current": 99.0,
                "psnr_export_vs_lower": 40.0,
                "mean_frac_pixels_gt_1_0": 0.1,
                "note": "test note",
            }, f)

        report = assemble_report(
            [phase0_path], [], None, phase3_path,
        )

    assert "PENDING" in report
    assert "## Phase 1" in report
    assert "## Phase 2" in report
    assert "indeterminate" in report.lower()
    assert "benign" not in report.lower()
    assert "portability concern" not in report.lower()
