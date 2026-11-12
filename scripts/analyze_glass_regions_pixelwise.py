"""Pixel-level glass/specular vs. non-glass PSNR split (plan section 13.7
deliverable 5: "Glass/specular evaluation uses pixel masks, not only
trajectory index ranges").

Companion to ``scripts/analyze_glass_regions.py`` (which buckets whole test
*views* by a hand-picked ``source_index`` range). This script instead reads
the per-view ``[render | gt | diff]`` PNG panels a report already saved
under ``comparison/iter_N/test/*.png`` (see ``utils/comparison_report.py``),
derives a per-pixel glass/specular-risk mask straight from the ground-truth
panel content, and reports masked-vs-unmasked PSNR per view and in
aggregate. No rendering, no training, no GPU — pure post-hoc analysis of
artifacts a normal run already writes.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eligsir.mapping.geometry.eval_masks import (  # noqa: E402
    glass_specular_pixel_mask,
    masked_psnr,
    split_render_gt_diff_panel,
)


def _load_png(path: str) -> np.ndarray:
    import imageio.v2 as imageio
    arr = imageio.imread(path)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    return arr


def analyze_test_dir(test_dir: str, gap_px: int = 4) -> dict:
    per_view = []
    glass_diffs = []
    nonglass_diffs = []
    for path in sorted(glob.glob(os.path.join(test_dir, "*.png"))):
        name = os.path.basename(path)
        if name.endswith("_normals.png") or name.endswith("_centers.png") or name.endswith("_rings.png"):
            continue
        panel = _load_png(path)
        try:
            render, gt, _diff = split_render_gt_diff_panel(panel, gap_px=gap_px)
        except Exception:
            continue
        mask = glass_specular_pixel_mask(gt)
        glass_psnr = masked_psnr(render, gt, mask)
        nonglass_psnr = masked_psnr(render, gt, ~mask)
        per_view.append({
            "name": name,
            "glass_pixel_frac": float(mask.mean()),
            "glass_psnr": glass_psnr,
            "nonglass_psnr": nonglass_psnr,
        })
        if glass_psnr is not None:
            glass_diffs.append(glass_psnr)
        if nonglass_psnr is not None:
            nonglass_diffs.append(nonglass_psnr)

    return {
        "n_views": len(per_view),
        "per_view": per_view,
        "glass_psnr_mean": float(np.mean(glass_diffs)) if glass_diffs else None,
        "nonglass_psnr_mean": float(np.mean(nonglass_diffs)) if nonglass_diffs else None,
        "mean_glass_pixel_frac": (
            float(np.mean([v["glass_pixel_frac"] for v in per_view])) if per_view else None
        ),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("test_dir", help="Path to a comparison/iter_N/test/ directory")
    parser.add_argument("--out", default=None,
                         help="Output path (default: glass_vs_nonglass_pixelwise_psnr.json "
                              "next to test_dir)")
    args = parser.parse_args(argv)

    if not os.path.isdir(args.test_dir):
        print(f"error: not a directory: {args.test_dir}", file=sys.stderr)
        return 2

    result = analyze_test_dir(args.test_dir)
    out_path = args.out or os.path.join(
        os.path.dirname(os.path.normpath(args.test_dir)) or ".",
        "glass_vs_nonglass_pixelwise_psnr.json",
    )
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    def _fmt(v):
        return f"{v:.2f}dB" if v is not None else "--"

    print(
        f"[pixelwise] n_views={result['n_views']} "
        f"glass_psnr_mean={_fmt(result['glass_psnr_mean'])} "
        f"nonglass_psnr_mean={_fmt(result['nonglass_psnr_mean'])} "
        f"mean_glass_pixel_frac={result['mean_glass_pixel_frac']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
