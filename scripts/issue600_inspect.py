"""Read-only inspector for #600 validation artifacts."""
import json
import glob
import re
import sys


def final_report(run_dir):
    cands = [x for x in glob.glob(f"{run_dir}/comparison/*/report.json")
             if "bootstrap" not in x]
    if not cands:
        return None, None
    p = max(cands, key=lambda x: int(re.search(r"iter_(\d+)", x).group(1)))
    return p, json.load(open(p))


def tc_of(d):
    """trajectory_coverage lives under geometry, or top-level on some paths."""
    return (d.get("geometry", {}) or {}).get("trajectory_coverage") \
        or d.get("trajectory_coverage") or {}


if __name__ == "__main__":
    for run in sys.argv[1:]:
        p, d = final_report(run)
        print("=" * 64)
        print(run, "->", p)
        if d is None:
            print("  NO REPORT")
            continue
        tc = tc_of(d)
        print("  TC keys present:", bool(tc))
        for k in ("verdict", "frame_admission", "admission_target_resolved",
                  "coverage_fraction", "frames_admitted", "n_source_frames",
                  "scene_registered"):
            print(f"    {k} = {tc.get(k)}")
        print("  soft_all:", json.dumps(d.get("soft_all_admission")))
        rs = (d.get("admission_cadence", {}) or {}).get("resolved_semantics", {})
        print("  declared target:", rs.get("admission_target_resolved"),
              "| owner:", rs.get("backpressure_owner"),
              "| stress:", rs.get("stress_run_declared"))
        print("  psnr:", d.get("mean_test_psnr"), "| eval_valid:", d.get("eval_valid"),
              "| n_test_metrics_failed:", d.get("n_test_metrics_failed"))
