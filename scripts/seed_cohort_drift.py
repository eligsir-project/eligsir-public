#!/usr/bin/env python3
"""CLI for the issue #287 INTRINSIC arm: per-checkpoint, per-cohort parameter drift.

CPU-only, no rendering, no GPU. Reads ``chkpntN.pth`` files already on disk and
reports what happened to the bootstrap/seed cohort's own parameters over a run,
cross-referenced against the run's ``optimization/seed_psnr`` curve.

This does NOT answer #287's occlusion-vs-intrinsic discriminator — that needs the
lineage-aware render decomposition (items 1-4) and the parameter-freeze A/Bs.
The coarse bbox-based render decomposition already in
``scripts/seed_render_decomp.py`` is the (GPU-requiring) starting point for that
half; this script is the complement, not a replacement.

Checkpoints whose lineage labels cannot be trusted are REFUSED loudly rather than
turned into numbers — see ``utils/seed_cohort_drift.lineage_trust_report``.

COHORT DEFINITION (changed 2026-08-14, issue #416 step 4): the seed cohort is now
``utils.gaussian_lineage.seed_cohort_mask`` — the EXACT one — not this module's
former local stand-in. A checkpoint written before #373 carries no
``seed_uid_range`` key and is therefore REFUSED with that verbatim reason instead
of being approximated. PR #407's original substrate
(``output/w3_106_matchedview_B1_20260812``) is one such: its drift number is not
reproducible under the exact definition and must be re-measured on a
``trustworthy`` run (see ``docs/architecture/lineage_coverage_manifest.json``).

Usage:
    python scripts/seed_cohort_drift.py output/<run> [--json out.json]
    python scripts/seed_cohort_drift.py --survey output/          # gate-only sweep
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eligsir.utils.seed_cohort_drift import (  # noqa: E402
    analyze_run,
    lineage_trust_report,
    list_checkpoints,
    load_checkpoint,
)


def _fmt_q(q):
    if not q:
        return "-"
    return f"mean={q['mean']:+.4g} p10={q['p10']:+.4g} med={q['median']:+.4g} p90={q['p90']:+.4g}"


def print_report(res):
    print(f"run: {res['run_dir']}")
    print(f"checkpoints: {res['n_checkpoints']}  trustworthy: {res['n_trustworthy_checkpoints']}")
    sp = res.get("seed_psnr") or {}
    if "error" in sp:
        print(f"seed_psnr: UNAVAILABLE ({sp['error']})")
    elif sp:
        print(f"seed_psnr: tag={sp['tag']}"
              + ("  [LEGACY TAG]" if sp.get("legacy_tag_used") else "")
              + f"  peak={sp['peak']['value']:.2f}dB@{sp['peak']['step']}"
              + f"  final={sp['final']['value']:.2f}dB@{sp['final']['step']}"
              + f"  max_drop={sp['max_drop_from_peak_db']:.2f}dB")
    print()

    for c in res["checkpoints"]:
        it = c.get("iteration")
        head = f"--- iter {it} (N={c.get('n')})"
        if c.get("seed_psnr_near") is not None:
            head += f"  seed_psnr~{c['seed_psnr_near']:.2f}dB"
        print(head)
        if c.get("lineage_untrustworthy"):
            print("    LINEAGE UNTRUSTWORTHY — refusing to report cohort numbers:")
            for r in c.get("reasons", []):
                print(f"      * {r}")
            continue
        for name, st in c["cohorts"].items():
            if not st.get("count"):
                continue
            sh = st.get("shape", {})
            print(f"    {name:<22} n={st['count']:<7d} "
                  f"alive={st['opacity_alive_frac']:.3f} "
                  f"op_med={st['opacity']['median']:.4f} "
                  f"s2/s1_med={sh.get('needle_s2_s1_median', float('nan')):.3f} "
                  f"s3/s2_med={sh.get('disk_s3_s2_median', float('nan')):.3f} "
                  f"smax_med={sh.get('max_scale_p50', float('nan')):.4f} "
                  f"luma_med={st['dc_luma']['median']:.3f}")
    print()
    for p in res["pairs"]:
        j = p["uid_join"]
        print(f"--- pair {j['prev_iteration']} -> {j['cur_iteration']}: "
              f"survived={j['n_survived']}/{j['n_prev']} new={j['n_new']} pruned={j['n_pruned']}")
        if not j["ok"]:
            for r in j.get("reasons", []):
                print(f"      * {r}")
            continue
        for name, d in p["drift"].items():
            if not d["n_prev"]:
                continue
            print(f"      [{name}] matched={d['n_matched']}/{d['n_prev']} "
                  f"survival={d['survival_frac']:.4f}")
            if not d["n_matched"]:
                continue
            print(f"        position drift (m): {_fmt_q(d.get('position_drift_m'))}")
            print(f"        opacity delta:      {_fmt_q(d.get('opacity_delta'))}")
            print(f"        log-scale delta:    {_fmt_q(d.get('log_scale_delta'))}")
            print(f"        DC-RGB L2 delta:    {_fmt_q(d.get('dc_rgb_l2_delta'))}")
    print()
    print("DISCRIMINATOR NOT RESOLVED: " + res["discriminator_note"])


def survey(root, all_checkpoints=False):
    """Gate-only sweep: which runs on disk have ANY trustworthy checkpoint?

    By default only the FIRST and LAST checkpoint of each run are gated (loading
    a full ``chkpntN.pth`` pulls the optimizer state too, so gating all 130
    checkpoints of a dense run costs minutes for no extra information — lineage
    presence is a property of the build, not of the iteration). ``--survey-all``
    gates every checkpoint.
    """
    rows = []
    for name in sorted(os.listdir(root)):
        d = os.path.join(root, name)
        if not os.path.isdir(d):
            continue
        paths = list_checkpoints(d)
        if not paths:
            continue
        probe = paths if all_checkpoints else sorted(set([paths[0], paths[-1]]),
                                                     key=paths.index)
        ok, bad, first_reason = 0, 0, None
        for p in probe:
            try:
                rec = load_checkpoint(p)
                t = lineage_trust_report(rec)
            except Exception as exc:
                bad += 1
                first_reason = first_reason or f"unreadable: {exc}"
                continue
            if t["trustworthy"]:
                ok += 1
            else:
                bad += 1
                first_reason = first_reason or t["reasons"][0].split(":")[0]
        rows.append((name, len(paths), ok, bad, first_reason or ""))
    print(f"{'run':<60} {'ckpts':>6} {'trust':>6} {'refuse':>7}  first_refusal_reason "
          f"({'all' if all_checkpoints else 'first+last'} probed)")
    for name, n, ok, bad, reason in rows:
        print(f"{name:<60} {n:>6} {ok:>6} {bad:>7}  {reason}")
    print()
    print(f"runs with >=1 trustworthy checkpoint: {sum(1 for r in rows if r[2] > 0)}/{len(rows)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", help="run directory (or output root with --survey)")
    ap.add_argument("--survey", action="store_true",
                    help="gate-only sweep over every run dir under PATH")
    ap.add_argument("--survey-all", action="store_true",
                    help="with --survey, gate every checkpoint (default: first+last only)")
    ap.add_argument("--json", help="write the full analysis JSON here")
    ap.add_argument("--cohort-edges", default="2000,6000",
                    help="comma-separated birth_iter bin upper bounds")
    ap.add_argument("--no-tb", action="store_true", help="skip the seed_psnr cross-reference")
    args = ap.parse_args()

    if args.survey:
        survey(args.path, all_checkpoints=args.survey_all)
        return

    edges = tuple(int(x) for x in args.cohort_edges.split(",") if x.strip())
    res = analyze_run(args.path, cohort_edges=edges, read_tb=not args.no_tb)
    print_report(res)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(res, f, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
