import argparse
import json


def verdict_for_delta_psnr(delta_abs: float) -> str:
    delta_abs = abs(delta_abs)
    if delta_abs < 0.1:
        return "benign"
    if delta_abs <= 1.0:
        return "renderer-specific-but-legitimate"
    return "serious portability concern"


FOLLOW_UP_ITEMS = [
    "Backward-gradient parity testing (custom vs native gsplat SH: dL/dSH, dL/ddirection) "
    "-- needs a training-like autograd graph; motivates but does not itself justify a default change.",
    "Gradient-graph audit during real training iterations (means.grad/sh0.grad/shN.grad "
    "layout/norm under native vs custom SH) -- requires running training, out of scope here.",
    "Multi-scale throughput benchmarking (N=100k-5M, visibility 1-100%) comparing Python eager, "
    "compiled Python, and native gsplat -- a performance project independent of this parity question.",
    "Decompose sh_backend into independent sh_impl/sh_activation/sh_dir_grad/sh_update_interval axes, "
    "and consider native gsplat as default (CUDA) or capability-gated default (ROCm) -- needs its own "
    "brainstorming pass and design spec before implementation.",
    "Fix PLY active_sh_degree serialization gap (see Task 9's read-only confirmation) -- touches the "
    "checkpoint/export format, training-adjacent, not fixed by this diagnostic.",
    "Fix eligsir/eval/_render.py's standalone pipe SimpleNamespace missing an explicit sh_backend "
    "field (silently defaults to \"python\") -- a real bug, flagged here as a recommendation only.",
]


def assemble_report(phase0_paths, phase1_paths, phase2_path, phase3_path, native_parity_path=None, ply_degree_path=None) -> str:
    lines = ["# SH-Clamp Renderer-Parity Diagnostic — Findings\n"]

    lines.append("## Phase 0 — Overshoot statistics\n")
    for p in phase0_paths:
        with open(p) as f:
            d = json.load(f)
        lines.append(f"- `{d['model_path']}` ({d['n_cameras']} cameras): "
                      f"raw>1.0={d['raw_frac_gt_1_0']:.4f}, weighted>1.0={d['weighted_frac_gt_1_0']:.4f}, "
                      f"raw>1.5={d['raw_frac_gt_1_5']:.4f}, weighted>1.5={d['weighted_frac_gt_1_5']:.4f}\n")

    overall_deltas = []
    if phase1_paths:
        lines.append("\n## Phase 1 — Frozen-checkpoint renderer ablation\n")
        lines.append("| scene | PSNR current | PSNR lower | delta PSNR | SSIM current | SSIM lower | LPIPS current | LPIPS lower | verdict |\n")
        lines.append("|---|---|---|---|---|---|---|---|---|\n")
        for p in phase1_paths:
            with open(p) as f:
                d = json.load(f)
            verdict = verdict_for_delta_psnr(d["mean_delta_psnr"])
            overall_deltas.append(d["mean_delta_psnr"])
            lines.append(f"| `{d['model_path']}` | {d['mean_psnr_current_vs_gt']:.3f} | {d['mean_psnr_lower_vs_gt']:.3f} | "
                          f"{d['mean_delta_psnr']:.3f} | {d['mean_ssim_current_vs_gt']:.4f} | {d['mean_ssim_lower_vs_gt']:.4f} | "
                          f"{d['mean_lpips_current_vs_gt']:.4f} | {d['mean_lpips_lower_vs_gt']:.4f} | {verdict} |\n")
    else:
        lines.append("\n## Phase 1 — Frozen-checkpoint renderer ablation — **PENDING**\n")
        lines.append("No `phase1_metrics.json` file(s) were supplied. This phase renders every test "
                      "camera twice (current `clamped` vs `lower_only`) plus per-camera SSIM/LPIPS, which "
                      "is a genuinely multi-minute (potentially much longer) job due to per-camera LPIPS "
                      "cost, and frequently has not finished by report-assembly time. The verdict/delta-PSNR "
                      "columns below are intentionally left blank rather than filled with placeholder or "
                      "estimated values.\n")
        lines.append("| scene | PSNR current | PSNR lower | delta PSNR | SSIM current | SSIM lower | LPIPS current | LPIPS lower | verdict |\n")
        lines.append("|---|---|---|---|---|---|---|---|---|\n")
        lines.append("| PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |\n")

    if phase2_path:
        lines.append("\n## Phase 2 — eps2d isolation (A/B/C)\n")
        with open(phase2_path) as f:
            d2 = json.load(f)
        lines.append(f"- A (clamp+eps0.3): PSNR={d2['variant_A']['mean_psnr_vs_gt']:.3f}\n")
        lines.append(f"- B (lower_only+eps0.3): PSNR={d2['variant_B']['mean_psnr_vs_gt']:.3f}\n")
        lines.append(f"- C (lower_only+eps0.0): PSNR={d2['variant_C']['mean_psnr_vs_gt']:.3f}\n")
        lines.append(f"- A→B delta (appearance-nonlinearity effect): {d2['A_to_B_delta_psnr']:.3f} dB\n")
        lines.append(f"- B→C delta (footprint-regularization effect): {d2['B_to_C_delta_psnr']:.3f} dB\n")
    else:
        lines.append("\n## Phase 2 — eps2d isolation (A/B/C) — **PENDING**\n")
        lines.append("No `phase2_metrics.json` was supplied. This phase was not available at "
                      "report-assembly time. Variant A/B/C PSNR values and the A→B / B→C deltas are "
                      "not reported here rather than estimated.\n")

    lines.append("\n## Phase 3 — Export-parity approximation\n")
    with open(phase3_path) as f:
        d3 = json.load(f)
    lines.append(f"- PSNR(export-approx, current-render) = {d3['psnr_export_vs_current']:.3f} "
                  "(expected to be Infinity by construction -- see note below)\n")
    lines.append(f"- PSNR(export-approx, lower_only-render) = {d3['psnr_export_vs_lower']:.3f}\n")
    lines.append(f"- Fraction of PLY-reevaluated pixels exceeding 1.0: {d3['mean_frac_pixels_gt_1_0']:.4f}\n")
    lines.append(f"- Note: {d3['note']}\n")

    if native_parity_path:
        with open(native_parity_path) as f:
            dn = json.load(f)
        lines.append("\n## Native-gsplat forward parity (Test A)\n")
        lines.append(f"- {json.dumps(dn)}\n")

    if ply_degree_path:
        with open(ply_degree_path) as f:
            dd = json.load(f)
        lines.append("\n## PLY active_sh_degree serialization check\n")
        lines.append(f"- trained_active_sh_degree={dd['trained_active_sh_degree']}, "
                      f"reloaded_active_sh_degree={dd['reloaded_active_sh_degree']}, "
                      f"mismatch={dd['mismatch']}\n")

    lines.append("\n## Recommended follow-up work (not actioned by this diagnostic)\n")
    for item in FOLLOW_UP_ITEMS:
        lines.append(f"- {item}\n")

    lines.append("\n## Recommendation\n")
    if overall_deltas:
        mean_overall_delta = sum(overall_deltas) / len(overall_deltas)
        overall_verdict = verdict_for_delta_psnr(mean_overall_delta)
        lines.append(f"Mean ΔPSNR across scenes: {mean_overall_delta:.3f} dB -> **{overall_verdict}**.\n")
        if overall_verdict == "benign":
            lines.append("A training-time retraining ablation is not warranted based on this diagnostic alone.\n")
        else:
            lines.append("A training-time retraining ablation (clamp vs clamp_min, measuring both PSNR trajectory "
                          "and gradient-dynamics effects separately) is warranted before any default change.\n")
    else:
        lines.append("**Overall verdict: indeterminate / deferred pending Phase 1 data.** This diagnostic's "
                      "decision thresholds are keyed on Phase 1's ΔPSNR between the `clamped` and "
                      "`lower_only` renderer variants, which is not available here (Phase 1 is pending). "
                      "None of this diagnostic's named verdict categories can be computed without it, and "
                      "none is asserted. A final recommendation on whether a training-time retraining "
                      "ablation is warranted should be deferred until Phase 1 (and ideally Phase 2) "
                      "complete.\n")

    return "".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Assemble the SH-clamp diagnostic findings report")
    parser.add_argument("--phase0_json", nargs="+", required=True)
    parser.add_argument("--phase1_json", nargs="+", default=None)
    parser.add_argument("--phase2_json", default=None)
    parser.add_argument("--phase3_json", required=True)
    parser.add_argument("--native_parity_json", default=None)
    parser.add_argument("--ply_degree_json", default=None)
    parser.add_argument("--output_md", required=True)
    args = parser.parse_args()

    report = assemble_report(
        args.phase0_json, args.phase1_json or [], args.phase2_json, args.phase3_json,
        native_parity_path=args.native_parity_json, ply_degree_path=args.ply_degree_json,
    )
    with open(args.output_md, "w") as f:
        f.write(report)
    print(f"Wrote report to {args.output_md}")


if __name__ == "__main__":
    main()
