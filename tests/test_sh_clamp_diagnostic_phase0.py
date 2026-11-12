import torch
from scripts.sh_clamp_diagnostic.phase0_overshoot_stats import compute_overshoot_stats_from_raw


def test_overshoot_stats_thresholds_and_weighting():
    # 4 "pixel contributions": raw max-channel values and their w_i = T_i * alpha_i weights
    raw_max_per_sample = torch.tensor([0.5, 1.05, 1.6, 2.5])
    weights = torch.tensor([10.0, 1.0, 1.0, 1.0])  # sample 0 (no overshoot) dominates the weighted sum

    stats = compute_overshoot_stats_from_raw(raw_max_per_sample, weights)

    assert stats["raw_frac_gt_1_0"] == 0.75  # 3 of 4 samples exceed 1.0
    assert stats["raw_frac_gt_1_1"] == 0.5   # samples at 1.6, 2.5 exceed 1.1
    assert stats["raw_frac_gt_1_5"] == 0.5   # samples at 1.6, 2.5
    assert stats["raw_frac_gt_2_0"] == 0.25  # sample at 2.5 only

    total_w = 13.0
    expected_weighted_gt_1_0 = (1.0 + 1.0 + 1.0) / total_w
    assert abs(stats["weighted_frac_gt_1_0"] - expected_weighted_gt_1_0) < 1e-6
