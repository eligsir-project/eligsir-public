"""Drive Level A (utils/local_quality.py) over ANY completed run's saved
holdout artifacts, producing a real per-cell quality-under-construction
history without retraining and without a GPU (issue #358).

Generalisation of ``scripts/local_quality_offline_demo.py``, which hardcoded
one fr3 run, its sentinel iteration and its tick filter. This version derives
all three from the run itself and refuses loudly rather than silently
measuring nothing.

Inputs required from ``<run_dir>`` (all written by a normal streaming run):

  test_cameras.json            FIXED holdout poses + intrinsics
  holdout_gt/<id>_depth.png    FIXED sensor depth, uint16 mm  <-- the scarce one
  comparison/iter_N/test/*.png saved (render | GT | diff) triptychs

What this is NOT: a live in-training tick. A live tick reduces the render the
trainer just produced, over EVERY holdout camera, at the training tick cadence.
This replays the small ``_representative_cams`` subset that the report engine
happened to save, from uint8-quantised PNG panels, at checkpoint cadence. The
mechanism exercised is identical (pixel -> fixed sensor depth -> fixed pose ->
immutable microcell -> per-cell PSNR history); the sampling is coarser and the
absolute dB differ slightly from a live tick. Both differences are printed.

Usage::

    python scripts/local_quality_from_run.py <run_dir> [-o <out_dir>]
                                             [--cell-size 0.5]
"""
import argparse
import json
import os
import re
import sys
from collections import Counter

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from eligsir.plot_style import svg_output_path  # noqa: E402
from eligsir.io.holdout_gt_depth import HOLDOUT_GT_DEPTH_SCALE  # noqa: E402
from eligsir.io.holdout_gt_exact import load_holdout_depth  # noqa: E402
from eligsir.utils.local_quality import (  # noqa: E402
    LocalQualityTracker,
    EstablishmentPolicy,
    append_local_quality_history,
)

# online_mode writes this in place of a real iteration; see the
# [[online-mode-report-iteration-sentinel-bug]] memory.
ONLINE_SENTINEL = 2147483647
PANEL_GAP = 4


class RunCam:
    """Holdout camera reconstructed from the run's own test_cameras.json.

    ``position``/``rotation`` are the c2w centre and c2w rotation (see
    utils.camera_utils.camera_to_JSON), so c2w = [[R, t], [0, 1]] and
    world_view_transform = inv(c2w).T -- the row-vector w2c this repo stores.
    """

    def __init__(self, entry, run_dir, depth_scale):
        c2w = torch.eye(4, dtype=torch.float32)
        c2w[:3, :3] = torch.tensor(entry["rotation"], dtype=torch.float32)
        c2w[:3, 3] = torch.tensor(entry["position"], dtype=torch.float32)
        self.world_view_transform = torch.linalg.inv(c2w).transpose(0, 1).contiguous()
        self.image_width, self.image_height = int(entry["width"]), int(entry["height"])
        self.fx, self.fy = entry["fx"], entry["fy"]
        self.cx, self.cy = entry["cx"], entry["cy"]
        self.FoVx = self.FoVy = 1.0
        self.uid = int(entry["id"])
        self.image_name = entry["img_name"]
        self._depth_scale = depth_scale
        self._gt_dir = os.path.join(run_dir, "holdout_gt")
        self._depth_path = os.path.join(self._gt_dir, f"{int(entry['id'])}_depth.png")
        self._depth_npy_path = os.path.join(self._gt_dir, f"{int(entry['id'])}_depth.npy")
        self.depth_provenance = None  # populated by sensor_depth(); see load_holdout_depth

    def has_depth(self):
        return os.path.exists(self._depth_npy_path) or os.path.exists(self._depth_path)

    def sensor_depth(self):
        """Fixed sensor depth, preferring the exact float32 .npy (issue #642).

        Only falls back to the quantized uint16-mm legacy PNG when no exact
        .npy is present, and always records which happened in
        ``self.depth_provenance`` -- never a silent quantization.
        """
        if not self.has_depth():
            return None
        if os.path.exists(self._depth_npy_path):
            arr, prov = load_holdout_depth(self._gt_dir, self.uid)
            self.depth_provenance = prov
            if arr is not None:
                return torch.from_numpy(arr)
        # No exact .npy (pre-#639 run, or npy load failed): fall back to the
        # quantized uint16-mm legacy PNG, explicitly marked as such.
        self.depth_provenance = {
            "cam_id": self.uid,
            "format": "png_uint16_mm_legacy",
            "bit_exact": False,
            "lossy": True,
            "warning": (f"cam {self.uid}: no exact holdout_gt depth .npy present; "
                        f"using QUANTIZED LEGACY uint16-mm PNG (1mm quantization, "
                        f"65.535m clip) instead"),
        }
        raw = np.array(Image.open(self._depth_path)).astype(np.float32)
        return torch.from_numpy(raw / self._depth_scale)


def _cfg(run_dir, key, default=None):
    """Read one value out of the run's own cfg_args snapshot."""
    path = os.path.join(run_dir, "cfg_args")
    if not os.path.exists(path):
        return default
    txt = open(path, errors="ignore").read()
    m = re.search(rf"\b{re.escape(key)}=([^,)\s]+)", txt)
    if not m:
        return default
    v = m.group(1).strip().strip("'\"")
    return v


def resolve_final_iteration(run_dir):
    """Real final optimisation step, from train.log's own banner.

    Never trust report.json["iteration"] for an online_mode run -- it is the
    ONLINE_SENTINEL, not an iteration.
    """
    log = os.path.join(run_dir, "train.log")
    if not os.path.exists(log):
        return None
    best = None
    for line in open(log, errors="ignore"):
        m = re.search(r"\[FINAL (?:STEP|ITER) (\d+)\]", line)
        if m:
            best = int(m.group(1))
    return best


def discover_ticks(run_dir):
    """(ticks, dropped, views) with a NUMERIC sort and a consistent view set.

    A tick whose saved view set differs from the majority set would change the
    aggregation membership mid-history, which is exactly the confound Level A
    exists to avoid, so those ticks are dropped and reported rather than mixed.
    """
    comp = os.path.join(run_dir, "comparison")
    sets = {}
    for d in sorted(os.listdir(comp)):
        m = re.fullmatch(r"iter_(\d+)", d)
        if not m:
            continue
        tdir = os.path.join(comp, d, "test")
        if not os.path.isdir(tdir):
            continue
        v = frozenset(f[:-4] for f in os.listdir(tdir)
                      if f.endswith(".png") and "_" not in f)
        if v:
            sets[int(m.group(1))] = v
    if not sets:
        return [], [], frozenset()
    majority = Counter(sets.values()).most_common(1)[0][0]
    final_iter = resolve_final_iteration(run_dir)
    ticks, dropped = [], []
    for n, v in sets.items():
        real = n
        if n == ONLINE_SENTINEL:
            if final_iter is None:
                dropped.append((n, "sentinel iteration and no [FINAL STEP N] banner"))
                continue
            real = final_iter
        (ticks if v == majority else dropped).append(
            (real, f"iter_{n}") if v == majority else (n, f"view set differs from majority ({len(v)} views)")
        )
    ticks.sort()  # NUMERIC -- iter_9000 sorts before iter_52000 here, unlike a string sort
    return ticks, sorted(dropped), majority


def read_panels(path, width, height):
    """Split a saved (render | GT | diff) triptych into render, GT."""
    a = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    if a.shape[0] != height:
        raise ValueError(f"panel height {a.shape[0]} != camera height {height}: {path}")
    render = a[:, 0:width]
    gt = a[:, width + PANEL_GAP: 2 * width + PANEL_GAP]
    if gt.shape[1] != width:
        raise ValueError(f"panel too narrow for a render|GT pair: {path}")
    to_chw = lambda x: torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))
    return to_chw(render), to_chw(gt)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("-o", "--out-dir", default=None)
    ap.add_argument("--cell-size", type=float, default=0.5)
    ap.add_argument("--max-depth", type=float, default=None,
                    help="default: the run's own streaming_max_depth")
    ap.add_argument("--depth-scale", type=float, default=None,
                    help="default: the run's own depth_scale (holdout depth PNGs "
                         "are written as uint16 mm by _save_holdout_gt, i.e. 1000)")
    a = ap.parse_args()

    run = os.path.abspath(a.run_dir)
    out_dir = a.out_dir or os.path.join(os.getcwd(), "local_quality_" + os.path.basename(run))
    os.makedirs(out_dir, exist_ok=True)

    # holdout_gt depth PNGs are written by _save_holdout_gt as uint16 MILLIMETRES
    # regardless of the dataset's own depth_scale, so 1000.0 is the correct
    # divisor here -- NOT the dataset depth_scale (TUM's 5000.0).
    depth_scale = a.depth_scale if a.depth_scale is not None else HOLDOUT_GT_DEPTH_SCALE
    max_depth = a.max_depth
    if max_depth is None:
        try:
            max_depth = float(_cfg(run, "streaming_max_depth", 8.0))
        except (TypeError, ValueError):
            max_depth = 8.0

    cams_path = os.path.join(run, "test_cameras.json")
    if not os.path.exists(cams_path):
        sys.exit(f"NOT MEASURED: no test_cameras.json under {run}")
    entries = json.load(open(cams_path))
    by_name = {e["img_name"]: RunCam(e, run, depth_scale) for e in entries}
    n_with_depth = sum(1 for c in by_name.values() if c.has_depth())
    print(f"[levelA] run={run}")
    print(f"[levelA] holdout cameras={len(by_name)}  with sensor depth={n_with_depth}")
    if n_with_depth == 0:
        sys.exit(
            "NOT MEASURED: holdout_gt/ carries no *_depth.png. Level A requires a "
            "FIXED sensor depth per holdout camera and deliberately refuses the "
            "rendered-depth fallback (it would measure the map with itself). See "
            "the producer gap noted in docs/architecture for ros2://live runs."
        )

    ticks, dropped, views = discover_ticks(run)
    if not ticks:
        sys.exit(f"NOT MEASURED: no comparison/iter_*/test/ renders under {run}")
    print(f"[levelA] ticks={len(ticks)} iters={[n for n, _ in ticks][:4]}"
          f" ... {[n for n, _ in ticks][-2:]}  views/tick={len(views)}")
    for n, why in dropped:
        print(f"[levelA] DROPPED tick iter={n}: {why}")
    print(f"[levelA] cell_size={a.cell_size}m max_depth={max_depth}m "
          f"depth_scale={depth_scale}")
    print("[levelA] NOTE: replayed from uint8 PNG panels over the saved "
          f"{len(views)}-view subset at CHECKPOINT cadence -- not a live tick.")

    tracker = LocalQualityTracker(
        cell_size=a.cell_size, max_depth_m=max_depth,
        policy=EstablishmentPolicy(min_eval_samples=300, min_views=2, min_ticks=3),
    )
    hist_path = os.path.join(out_dir, "local_quality_history.jsonl")
    if os.path.exists(hist_path):
        os.remove(hist_path)

    for n, d in ticks:
        tdir = os.path.join(run, "comparison", d, "test")
        names = sorted(f[:-4] for f in os.listdir(tdir)
                       if f.endswith(".png") and "_" not in f)
        cams = [by_name[v] for v in names if v in by_name and by_name[v].has_depth()]
        if not cams:
            print(f"[levelA] iter={n}: no camera with sensor depth, skipping tick")
            continue
        tracker.prepare(cams, sensor_depth_fn=lambda c, h, w: c.sensor_depth())
        acc = tracker.new_accumulator()
        for cam in cams:
            r, g = read_panels(os.path.join(tdir, f"{cam.image_name}.png"),
                               cam.image_width, cam.image_height)
            tracker.observe(cam, r, g, acc)
        rec = tracker.record_tick(n, acc)
        append_local_quality_history(out_dir, rec)
        flag = " MEASUREMENT_INCOMPLETE" if rec.get("measurement_incomplete") else ""
        print(f"iter={n:>7} views={len(cams)} cells={rec['n_cells_observed']:>4} "
              f"est={rec['n_cells_established']:>4} "
              f"worst_drop={rec['worst_drop_from_peak_db']:.2f}dB{flag}")

    n_legacy = sum(1 for c in by_name.values()
                   if c.depth_provenance is not None
                   and c.depth_provenance.get("format") == "png_uint16_mm_legacy")
    n_exact = sum(1 for c in by_name.values()
                  if c.depth_provenance is not None
                  and c.depth_provenance.get("bit_exact"))
    if n_legacy:
        print(f"[levelA] WARNING: {n_legacy}/{n_exact + n_legacy} sampled holdout depth reads "
              f"fell back to the QUANTIZED LEGACY uint16-mm PNG (no exact holdout_gt/*_depth.npy "
              f"present -- see issue #642). Treat this run's Level A geometry as measured against "
              f"quantized (1mm, 65.535m-clip) sensor depth, not exact float32.")
    print(f"[levelA] history written: {hist_path}")
    write_envelope_plot(run, out_dir, hist_path)
    return hist_path


def write_envelope_plot(run_dir, out_dir, hist_path):
    """Global test curve + the spatial envelope of per-cell drop over it.

    Issue #358 lists "global curve + spatial envelope" as deferred. Both sides
    are test-holdout PSNR reduced identically (drop from each series' OWN
    running peak), so the comparison is like-for-like -- not global-train vs
    local-test. The global series is the run's own ``psnr_history.jsonl``,
    untouched by Level A.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    recs = [json.loads(l) for l in open(hist_path)]
    if not recs:
        return None
    gpath = os.path.join(run_dir, "psnr_history.jsonl")
    g_drop = {}
    if os.path.exists(gpath):
        peak = -1e9
        for r in sorted((json.loads(l) for l in open(gpath)),
                        key=lambda d: d["iteration"]):
            v = r.get("mean_test_psnr")
            if v is None:
                continue
            peak = max(peak, v)
            g_drop[r["iteration"]] = peak - v

    its, p50, p90, worst, n_est = [], [], [], [], []
    for r in recs:
        cells = [c for c in r["cells"].values() if c["established"]]
        if not cells:
            continue
        d = np.array([c["drop_from_peak_db"] for c in cells])
        its.append(r["iteration"])
        p50.append(float(np.median(d)))
        p90.append(float(np.percentile(d, 90)))
        worst.append(float(r["worst_drop_from_peak_db"]))
        n_est.append(len(cells))
    if not its:
        return None

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.fill_between(its, p50, p90, alpha=0.25, color="tab:red",
                    label="per-cell drop, p50-p90 envelope")
    ax.plot(its, worst, color="tab:red", lw=1.4, label="worst single cell")
    ax.plot(its, p50, color="tab:red", lw=1.0, ls="--", label="median cell")
    gi = [i for i in its if i in g_drop]
    if gi:
        ax.plot(gi, [g_drop[i] for i in gi], color="k", lw=2.0,
                label="GLOBAL mean test PSNR (same reduction)")
    ax.set_xlabel("iteration")
    ax.set_ylabel("dB below own running peak")
    ax.set_title(
        "Spatial envelope of quality-under-construction vs. the global curve\n"
        f"{os.path.basename(run_dir)[:48]}  |  {n_est[-1]}/"
        f"{len(recs[-1].get('cells', {}))} cells established, {len(its)} ticks",
        fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, "local_quality_envelope.svg")
    fig.savefig(svg_output_path(path), dpi=110)
    plt.close(fig)
    print(f"[levelA] envelope plot written: {path}")
    return path


if __name__ == "__main__":
    main()
