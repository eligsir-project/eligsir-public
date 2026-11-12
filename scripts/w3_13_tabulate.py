#!/usr/bin/env python3
"""W3-13: tabulate + mechanically exclude invalid runs from an ablation matrix.

Per CLAUDE.md's Run Contracts and the W3-13 brief's deliverable #2 ("invalid
runs are excluded automatically"), this script refuses to report a number for
any run directory whose report.json fails any of:
  - eval_valid == True and n_test > 0
  - trajectory_coverage.verdict not in {"SUSPECTED_TRUNCATED"}
  - trajectory_coverage.map_reinit_invalidated is not True
  - seed_fade.detected is False, OR a detected fade is
    flagged but not silently dropped -- printed as a warning, not excluded,
    per CLAUDE.md ("must not be quoted... without first checking ... and
    deciding"), since an investigated+recovered fade is not automatically
    disqualifying.

It also refuses to compare runs whose eval_metadata.holdout_hash differs
from the first valid run's hash (per CLAUDE.md's "Compared runs must share
the same test cameras" rule) -- these are printed as INCOMPARABLE, not
silently included with a mismatched baseline.

Usage:
    python scripts/w3_13_tabulate.py output/w3_13_fr3_G0 output/w3_13_fr3_G1 ...
"""
import json
import sys
from pathlib import Path


def load_report(model_dir: Path):
    """Load a run's FINAL report.json.

    Delegates to `utils.coverage_access.find_latest_report_path` (issue
    #685's authoritative accessor, itself a thin re-export of
    `utils.baseline_comparison._find_latest_report_json`) rather than
    sorting `comparison/iter_*` here. The local sort this replaced used
    `key=lambda d: d.name` -- LEXICAL -- so `iter_2147483647` (the
    online_mode sentinel) sorted BEFORE `iter_30000` ("2" < "3") and
    `iters[-1]` silently picked the wrong report for every online_mode run.
    That is the exact hazard CLAUDE.md's Run Contracts name, and the shared
    helper already gets it right (numeric ordering, `_bootstrap_views`
    skipped, no bogus `<run_dir>/report.json` root fallback).

    Imported from `utils.coverage_access` (torch-free, #685) so this
    tabulator keeps working in a torch-free environment -- same fix as
    #449's/#435's.
    """
    from eligsir.utils.coverage_access import find_latest_report_path

    if not (model_dir / "comparison").exists():
        return None, "no comparison/ dir"
    report_path = find_latest_report_path(str(model_dir))
    if not report_path:
        return None, "no iter_N/report.json under comparison/"
    with open(report_path) as f:
        return json.load(f), None


def validate(report: dict) -> list[str]:
    """Return a list of exclusion reasons; empty list == valid."""
    from eligsir.utils.partition_eval import get_trajectory_coverage

    reasons = []
    if not report.get("eval_valid", False):
        reasons.append("eval_valid=False")
    if report.get("n_test", 0) in (0, None):
        reasons.append("n_test=0")
    # Issue #504: was report["geometry"]["trajectory_coverage"], a path no
    # writer has ever produced -- so these two exclusion checks (truncation,
    # map re-init) silently never fired on any run.
    tc = get_trajectory_coverage(report) or {}
    if tc.get("verdict") == "SUSPECTED_TRUNCATED":
        reasons.append("trajectory_coverage.verdict=SUSPECTED_TRUNCATED")
    if tc.get("map_reinit_invalidated") is True:
        reasons.append("map_reinit_invalidated=True")
    return reasons


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 1
    rows = []
    baseline_hash = None
    for raw in argv[1:]:
        d = Path(raw)
        name = d.name
        report, err = load_report(d)
        if report is None:
            print(f"[EXCLUDED] {name}: {err}")
            continue
        reasons = validate(report)
        h = report.get("eval_metadata", {}).get("holdout_hash")
        if baseline_hash is None and not reasons:
            baseline_hash = h
        if h != baseline_hash and not reasons:
            reasons.append(f"holdout_hash mismatch ({h} != {baseline_hash})")
        seed_fade = report.get("seed_fade", {}) or {}
        fade_note = ""
        if seed_fade.get("detected"):
            # Issue #346: the graded severity/confounded verdict, when the
            # report has it (written by write_post_training_report since the
            # grading landed). Prefixed onto whichever detail line follows.
            sev = seed_fade.get("severity")
            sev_txt = f"severity={sev} " if sev else ""
            conf_txt = " CONFOUNDED" if seed_fade.get("confounded") else ""
            shape_txt = (
                f" shape={seed_fade.get('curve_shape')}"
                if seed_fade.get("curve_shape") else ""
            )
            # issue #282: `recovered` is a sticky latch ("some episode once
            # ended"), NOT "the run ended healthy" -- both audited fr3 runs
            # ended inside an unrecovered fade while reporting recovered:true.
            # Prefer recovered_final / deepest_dip_db / episode_count when the
            # report has them (runs from 2026-08-13 on); fall back to the
            # legacy keys for older reports, labelled as legacy.
            if "recovered_final" in seed_fade:
                fade_note = (
                    f" [SEED-FADE{conf_txt}: {sev_txt}status={seed_fade.get('status')}"
                    f" deepest_dip_db={seed_fade.get('deepest_dip_db')}"
                    f"@iter{seed_fade.get('trough_iter')}"
                    f" episodes={seed_fade.get('episode_count')}"
                    f" recovered_final={seed_fade.get('recovered_final')}"
                    f" in_fade_frac={seed_fade.get('fade_sample_fraction')}"
                    f"{shape_txt}"
                    f" -- not auto-excluded, review before quoting]"
                )
            else:
                fade_note = (
                    f" [SEED-FADE{conf_txt}: {sev_txt}detected (legacy block):"
                    f" max_dip_db={seed_fade.get('max_dip_db')}"
                    f" recovered={seed_fade.get('recovered')} (STICKY LATCH, not 'ended healthy')"
                    f"{shape_txt}"
                    f" -- not auto-excluded, review before quoting]"
                )
        if reasons:
            print(f"[EXCLUDED] {name}: {', '.join(reasons)}")
            continue
        g = report.get("geometry", {})
        shape = g.get("shape", {})
        offs = g.get("offsurface", {})
        rows.append({
            "arm": name,
            "mean_test_psnr": report.get("mean_test_psnr"),
            "mean_test_lpips": report.get("mean_test_lpips"),
            "n_test": report.get("n_test"),
            "n_live": shape.get("n_live") or shape.get("n_gaussians"),
            "needle_frac_lt_0.3": shape.get("needle_frac_lt_0.3"),
            "offsurface_mass_frac": offs.get("offsurface_mass_frac"),
            "holdout_hash": h,
            "fade_note": fade_note,
        })

    print()
    print(f"{'arm':<20}{'test_psnr':>11}{'lpips':>9}{'n_test':>8}{'n_live':>9}{'needle':>9}{'offsurf':>9}")
    for r in rows:
        print(
            f"{r['arm']:<20}{r['mean_test_psnr']:>11.3f}{r['mean_test_lpips']:>9.4f}"
            f"{r['n_test']:>8}{r['n_live']:>9}{r['needle_frac_lt_0.3']:>9.4f}"
            f"{r['offsurface_mass_frac']:>9.4f}{r['fade_note']}"
        )
    if len({r["holdout_hash"] for r in rows}) > 1:
        print("\nWARNING: multiple holdout_hash values among reported rows -- should not happen, investigate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
