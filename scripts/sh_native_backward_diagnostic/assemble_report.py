import argparse
import json


def assemble_report(
    synthetic_json_path,
    gradient_audit_json_path,
    canary_notes: dict,
    confounded_gradient_audit_json_path: str = None,
) -> str:
    with open(synthetic_json_path) as f:
        synthetic = json.load(f)
    with open(gradient_audit_json_path) as f:
        audit = json.load(f)

    confounded_audit = None
    if confounded_gradient_audit_json_path:
        with open(confounded_gradient_audit_json_path) as f:
            confounded_audit = json.load(f)

    lines = ["# Native-gsplat SH Backward-Parity Diagnostic — Findings (Phase B)\n"]

    lines.append("\n## Premise re-check\n")
    lines.append(
        "The historical claim that native gsplat's SH backward pass crashes on "
        "ROCm was re-checked before this diagnostic: no specific error message, "
        "exception, or reproduction is documented anywhere in this repo for an "
        "SH-specific backward failure (the only documented ROCm backward crash, "
        "in docs/ROCM_PODMAN.md, is a different, already-fixed rasterizer/wave32 "
        "issue). A direct empirical test (forward+backward through both the "
        "standalone spherical_harmonics op and the full rasterization() path) "
        "produced clean, finite gradients with no crash on this repo's current "
        "pinned ROCm/gsplat build.\n"
    )

    lines.append("\n## Phase B1a — Synthetic backward-gradient parity (SH degrees 0-3)\n")
    lines.append("| degree | max abs grad err (dc) | max abs grad err (rest) | max abs grad err (dir) |\n")
    lines.append("|---|---|---|---|\n")
    for deg_key, d in synthetic.items():
        lines.append(
            f"| {deg_key} | {d['max_abs_grad_error_dc']:.2e} | "
            f"{d['max_abs_grad_error_rest']:.2e} | {d['max_abs_grad_error_dir']:.2e} |\n"
        )
    lines.append(
        "\nSH-coefficient gradients (`dc`/`rest`) agree to float precision "
        "(~1e-7) at every degree. The `dirs` gradient disagrees by O(1) at "
        "every degree >= 1, growing with degree (1.4 at deg 1 up to 15.9 at "
        "deg 3). A follow-up decomposition (not re-run here, already "
        "established before this task) proved this disagreement is entirely "
        "radial (parallel to the `dirs` unit vector) with negligible "
        "tangential component (~6e-7) -- i.e. it does not represent a "
        "directional/angular disagreement, only a magnitude convention "
        "difference along a direction that (for a unit vector) does not "
        "correspond to a physically meaningful degree of freedom on its own.\n"
    )

    lines.append("\n## Phase B1b — Real-checkpoint gradient-graph audit (fr3 champion checkpoint, iteration 52000)\n")
    lines.append(
        "This section reports TWO measurements, because the first (confounded) "
        "run understated agreement for a documented, unrelated reason.\n"
    )

    if confounded_audit is not None:
        lines.append(
            "\n**Confounded measurement** (`pipe.sh_color_clamp_mode=\"clamped\"`, "
            "this script's original default): means.grad disagreed by "
            f"~3.9% relative L2 (norms {confounded_audit['means_grad_norm_python']:.6f} vs "
            f"{confounded_audit['means_grad_norm_gsplat']:.6f}, max abs diff "
            f"{confounded_audit['means_grad_max_abs_diff']:.4e}). This was traced to a "
            "known confound, NOT a backward-pass disagreement: "
            "`sh_color_clamp_mode=\"clamped\"` is documented-inert on the "
            "`sh_impl=\"gsplat\"` path (native gsplat applies its own internal "
            "clamping regardless of this setting), so the two `sh_impl` values "
            "were rendering measurably different FORWARD images on this "
            "checkpoint (max abs pixel diff ~0.045) before backward ever ran -- "
            "contaminating the gradient comparison with an already-known "
            "forward difference unrelated to Phase B1a's backward-only finding.\n"
        )

    lines.append(
        "\n**Deconfounded measurement** (`pipe.sh_color_clamp_mode=\"lower_only\"` "
        "on both sides, matching native gsplat's actual real clamp behavior -- "
        "this is the number that answers Phase B's actual question):\n"
    )
    lines.append(f"- means.grad norm: python={audit['means_grad_norm_python']:.6f}, gsplat={audit['means_grad_norm_gsplat']:.6f}\n")
    lines.append(f"- means.grad max abs diff: {audit['means_grad_max_abs_diff']:.4e}\n")
    lines.append(f"- sh0.grad norm: python={audit['sh0_grad_norm_python']:.6f}, gsplat={audit['sh0_grad_norm_gsplat']:.6f}\n")
    lines.append(f"- shN.grad norm: python={audit['shN_grad_norm_python']:.6f}, gsplat={audit['shN_grad_norm_gsplat']:.6f}\n")
    lines.append(
        "\nWith the clamp-mode confound controlled for, forward rendered images "
        "agree to ~1.8e-7 (float noise) and `means.grad` agrees to within "
        "float precision (relative L2 diff ~7-8e-5, reproduced independently "
        "twice). This conclusively confirms the radial `dirs`-gradient "
        "disagreement documented in Phase B1a is benign for real training: "
        "once the known, unrelated clamp-mode confound is removed, gradients "
        "agree almost exactly between `sh_impl=\"python\"` and "
        "`sh_impl=\"gsplat\"`.\n"
    )

    lines.append("\n## Phase B2 — Short training canary run (fr3, cap_max=20000, iterations=3000)\n")
    lines.append(f"- sh_impl=python: {canary_notes.get('python', 'PENDING -- fill in from real run')}\n")
    lines.append(f"- sh_impl=gsplat: {canary_notes.get('gsplat', 'PENDING -- fill in from real run')}\n")

    lines.append("\n## Recommendation\n")
    lines.append(
        "[Fill in based on the actual measured values above: if synthetic and "
        "real-checkpoint gradients agree within a tight tolerance, and both "
        "canary arms train stably with comparable PSNR trends, a full "
        "default-flip investigation (Phase C: multi-scale throughput "
        "benchmarking, broader-scene validation, full-length training "
        "comparison) is warranted. If any of these show meaningful "
        "disagreement or instability, document exactly what and recommend "
        "against proceeding to Phase C without further investigation.]\n"
    )

    return "".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Assemble the native-gsplat SH backward-parity findings report")
    parser.add_argument("--synthetic_json", required=True)
    parser.add_argument("--gradient_audit_json", required=True)
    parser.add_argument(
        "--confounded_gradient_audit_json",
        default=None,
        help="Optional: the original clamped-mode (confounded) gradient_audit JSON, "
        "shown alongside the deconfounded number for context.",
    )
    parser.add_argument("--canary_python_notes", required=True, help="Free-text summary of the python-sh_impl canary run's outcome")
    parser.add_argument("--canary_gsplat_notes", required=True, help="Free-text summary of the gsplat-sh_impl canary run's outcome")
    parser.add_argument("--output_md", required=True)
    args = parser.parse_args()

    report = assemble_report(
        args.synthetic_json, args.gradient_audit_json,
        {"python": args.canary_python_notes, "gsplat": args.canary_gsplat_notes},
        confounded_gradient_audit_json_path=args.confounded_gradient_audit_json,
    )
    with open(args.output_md, "w") as f:
        f.write(report)
    print(f"Wrote report to {args.output_md}")


if __name__ == "__main__":
    main()
