from scripts.sh_clamp_diagnostic.phase2_eps2d_isolation import isolate_effects


def test_isolate_effects_computes_both_deltas():
    variant_summaries = {
        "variant_A": {"mean_psnr_vs_gt": 27.95},
        "variant_B": {"mean_psnr_vs_gt": 27.80},
        "variant_C": {"mean_psnr_vs_gt": 27.50},
    }
    deltas = isolate_effects(variant_summaries)
    assert abs(deltas["A_to_B_delta_psnr"] - (27.95 - 27.80)) < 1e-9
    assert abs(deltas["B_to_C_delta_psnr"] - (27.80 - 27.50)) < 1e-9
