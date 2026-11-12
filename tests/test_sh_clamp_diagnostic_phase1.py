import torch
from scripts.sh_clamp_diagnostic.phase1_renderer_ablation import summarize_metrics


def test_summarize_metrics_computes_delta_and_means():
    per_camera = [
        {"psnr_current_vs_gt": 30.0, "psnr_lower_vs_gt": 29.5, "ssim_current_vs_gt": 0.9,
         "ssim_lower_vs_gt": 0.89, "lpips_current_vs_gt": 0.05, "lpips_lower_vs_gt": 0.06,
         "psnr_current_vs_lower": 40.0, "frac_lower_pixels_gt_1_0": 0.01},
        {"psnr_current_vs_gt": 28.0, "psnr_lower_vs_gt": 28.2, "ssim_current_vs_gt": 0.85,
         "ssim_lower_vs_gt": 0.86, "lpips_current_vs_gt": 0.07, "lpips_lower_vs_gt": 0.065,
         "psnr_current_vs_lower": 38.0, "frac_lower_pixels_gt_1_0": 0.02},
    ]
    summary = summarize_metrics(per_camera)
    assert summary["mean_psnr_current_vs_gt"] == 29.0
    assert summary["mean_psnr_lower_vs_gt"] == 28.85
    assert abs(summary["mean_delta_psnr"] - (29.0 - 28.85)) < 1e-9
    assert summary["n_cameras"] == 2


def test_summarize_metrics_includes_native_variant_keys():
    per_camera = [
        {"psnr_current_vs_gt": 30.0, "psnr_lower_vs_gt": 29.5, "ssim_current_vs_gt": 0.9,
         "ssim_lower_vs_gt": 0.89, "lpips_current_vs_gt": 0.05, "lpips_lower_vs_gt": 0.06,
         "psnr_current_vs_lower": 40.0, "frac_lower_pixels_gt_1_0": 0.01,
         "psnr_native_vs_gt": 29.8, "psnr_native_vs_current": 41.0, "psnr_native_vs_lower": 39.0,
         "ssim_native_vs_gt": 0.895, "lpips_native_vs_gt": 0.055},
    ]
    summary = summarize_metrics(per_camera)
    assert "mean_psnr_native_vs_gt" in summary
    assert "mean_psnr_native_vs_current" in summary
    assert "mean_psnr_native_vs_lower" in summary
    assert "mean_ssim_native_vs_gt" in summary
    assert "mean_lpips_native_vs_gt" in summary
    # frac_lower_pixels_gt_1_0 must still be excluded from the generic mean_* sweep
    # (it has its own dedicated summary key) while native keys are NOT excluded.
    assert "mean_frac_lower_pixels_gt_1_0" in summary
    assert summary["mean_psnr_native_vs_gt"] == 29.8
    assert summary["mean_psnr_native_vs_current"] == 41.0
    assert summary["n_cameras"] == 1
