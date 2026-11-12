"""GPU microbenchmark closing issue #358's named fp64-scatter_add_ gap.

The Level A per-cell accumulator (utils/local_quality.CellTickAccumulator)
accumulates in float64 because a large cell sums millions of squared
residuals. Its only cost measurement so far is a CPU-torch microbenchmark
(22.9 ms/tick, 2.29 ms/view at 480x640 with 4000 cells), and the PR author
flagged explicitly that fp64 scatter_add_ may be EMULATED on this project's
gfx1100/gfx1151 targets, so that number may not transfer.

This measures the real thing on the real GPU:
  * whole-add_view wall time (the number that actually matters per tick),
    fp64 vs an fp32-accumulation variant, with warmup + explicit syncs;
  * a component breakdown (squared-diff, dtype cast, boolean-mask gathers,
    each scatter) so a bad number can be attributed;
  * a CONTENTION sweep over slots-per-tick at fixed element count, because
    the discriminating axis for atomic scatter_add_ is threads-per-address,
    not total elements -- a benign 4000-cell-only measurement could be a
    false negative;
  * a precision check: max per-cell |dPSNR| between fp64 and fp32
    accumulation on the same real residuals, which is the measured
    precision bound the fallback would need to state.

Run inside the container:
    python scripts/bench_local_quality_scatter.py
"""
from __future__ import annotations

import json
import sys
import time

import torch


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _time(fn, iters=20, warmup=5):
    """Median-of-iters wall time in ms, with warmup and explicit syncs."""
    for _ in range(warmup):
        fn()
    _sync()
    ts = []
    for _ in range(iters):
        _sync()
        t0 = time.perf_counter()
        fn()
        _sync()
        ts.append((time.perf_counter() - t0) * 1e3)
    ts.sort()
    return ts[len(ts) // 2]


def make_view(h, w, n_slots, device, invalid_frac=0.15, seed=0):
    """A realistic frozen cell map + one render/GT pair.

    slot_index is -1 at invalid-depth pixels (exactly as build_camera_cell_map
    produces) and otherwise a slot drawn so every slot is hit -- i.e. the
    average slot receives (h*w*(1-invalid_frac))/n_slots pixels, which is the
    contention knob.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    n = h * w
    slot = torch.randint(0, n_slots, (n,), generator=g, dtype=torch.long)
    invalid = torch.rand(n, generator=g) < invalid_frac
    slot[invalid] = -1
    slot = slot.to(device)
    mask = slot >= 0
    render = torch.rand((3, h, w), generator=g).to(device)
    gt = (render.cpu() + 0.05 * torch.randn((3, h, w), generator=g)).clamp(0, 1).to(device)
    return slot, mask, render, gt


def add_view_ops(render, gt, slot, mask, sq_err, n_samples, n_views, n_slots, acc_dtype):
    """Exactly what CellTickAccumulator.add_view does, dtype-parameterised."""
    h, w = render.shape[-2], render.shape[-1]
    se = ((render.to(torch.float32) - gt.to(torch.float32)) ** 2)
    se = se.reshape(3, h * w).sum(dim=0).to(acc_dtype)
    idx_v = slot[mask]
    sq_err.scatter_add_(0, idx_v, se[mask])
    n_samples.scatter_add_(0, idx_v, torch.full_like(idx_v, 3.0, dtype=acc_dtype))
    present = torch.zeros(n_slots, dtype=acc_dtype, device=render.device)
    present.scatter_(0, idx_v, 1.0)
    n_views += present


def add_view_hoisted(render, gt, valid_px, valid_slots, uniq, uniq_counts,
                     uniq_ones, sq_err, n_samples, n_views, acc_dtype):
    """The POST-hoist add_view: one residual scatter_add_ + two index_add_
    over the camera's frozen per-cell histogram (utils/local_quality.py)."""
    h, w = render.shape[-2], render.shape[-1]
    se = ((render.to(torch.float32) - gt.to(torch.float32)) ** 2)
    se = se.reshape(3, h * w).sum(dim=0)
    se_v = se.index_select(0, valid_px).to(acc_dtype)
    sq_err.scatter_add_(0, valid_slots, se_v)
    n_samples.index_add_(0, uniq, uniq_counts)
    n_views.index_add_(0, uniq, uniq_ones)


def bench_config(h, w, n_slots, device, n_views_per_tick=10):
    slot, mask, render, gt = make_view(h, w, n_slots, device)
    out = {
        "height": h, "width": w, "n_slots": n_slots,
        "n_valid_px": int(mask.sum().item()),
        "px_per_slot": round(float(mask.sum().item()) / n_slots, 1),
    }
    valid_px = torch.nonzero(mask, as_tuple=True)[0]
    valid_slots = slot.index_select(0, valid_px)
    uniq, cnt = torch.unique(valid_slots, return_counts=True)
    for name, dt in (("fp64", torch.float64), ("fp32", torch.float32)):
        z = lambda: torch.zeros(n_slots, dtype=dt, device=device)
        a, b, c = z(), z(), z()
        ms = _time(lambda: add_view_ops(render, gt, slot, mask, a, b, c, n_slots, dt))
        out[f"add_view_ms_{name}"] = round(ms, 4)
        out[f"tick_ms_{name}"] = round(ms * n_views_per_tick, 3)
        a2, b2, c2 = z(), z(), z()
        uc, uo = (cnt * 3).to(dt), torch.ones_like(uniq, dtype=dt)
        ms2 = _time(lambda: add_view_hoisted(
            render, gt, valid_px, valid_slots, uniq, uc, uo, a2, b2, c2, dt))
        out[f"hoisted_add_view_ms_{name}"] = round(ms2, 4)
        out[f"hoisted_tick_ms_{name}"] = round(ms2 * n_views_per_tick, 3)
    out["fp64_over_fp32"] = round(
        out["add_view_ms_fp64"] / max(out["add_view_ms_fp32"], 1e-9), 2)
    out["hoisted_fp64_over_fp32"] = round(
        out["hoisted_add_view_ms_fp64"] / max(out["hoisted_add_view_ms_fp32"], 1e-9), 2)
    out["hoist_speedup_fp64"] = round(
        out["add_view_ms_fp64"] / max(out["hoisted_add_view_ms_fp64"], 1e-9), 2)
    return out


def bench_components(h, w, n_slots, device):
    """Attribute the fp64 cost to individual ops."""
    slot, mask, render, gt = make_view(h, w, n_slots, device)
    res = {}
    for name, dt in (("fp64", torch.float64), ("fp32", torch.float32)):
        acc = torch.zeros(n_slots, dtype=dt, device=device)
        se32 = ((render - gt) ** 2).reshape(3, h * w).sum(dim=0)
        se = se32.to(dt)
        idx_v = slot[mask]
        se_v = se[mask]
        fill = torch.full_like(idx_v, 3.0, dtype=dt)
        res[f"sqdiff_fp32_ms"] = round(_time(
            lambda: ((render - gt) ** 2).reshape(3, h * w).sum(dim=0)), 4)
        res[f"cast_to_{name}_ms"] = round(_time(lambda: se32.to(dt)), 4)
        res[f"bool_mask_gather_{name}_ms"] = round(_time(lambda: (slot[mask], se[mask])), 4)
        res[f"scatter_add_sqerr_{name}_ms"] = round(_time(
            lambda: acc.scatter_add_(0, idx_v, se_v)), 4)
        res[f"scatter_add_counts_{name}_ms"] = round(_time(
            lambda: acc.scatter_add_(0, idx_v, fill)), 4)
        res[f"present_alloc_scatter_{name}_ms"] = round(_time(
            lambda: torch.zeros(n_slots, dtype=dt, device=device).scatter_(0, idx_v, 1.0)), 4)
    return res


def bench_precision(h, w, n_slots, device, n_views=10):
    """Measured fp32-vs-fp64 accumulation error, expressed in dB of pooled PSNR.

    This is the number an fp32 fallback would have to state as its bound.
    Also reports run-to-run nondeterminism of fp32 (atomic scatter_add_ has
    no defined summation order, so fp32 results are not bit-reproducible).
    """
    import numpy as np
    a64 = torch.zeros(n_slots, dtype=torch.float64, device=device)
    a32 = torch.zeros(n_slots, dtype=torch.float32, device=device)
    a32b = torch.zeros(n_slots, dtype=torch.float32, device=device)
    n64 = torch.zeros(n_slots, dtype=torch.float64, device=device)
    for v in range(n_views):
        slot, mask, render, gt = make_view(h, w, n_slots, device, seed=100 + v)
        se = ((render - gt) ** 2).reshape(3, h * w).sum(dim=0)
        idx_v = slot[mask]
        a64.scatter_add_(0, idx_v, se[mask].to(torch.float64))
        a32.scatter_add_(0, idx_v, se[mask].to(torch.float32))
        a32b.scatter_add_(0, idx_v, se[mask].to(torch.float32))
        n64.scatter_add_(0, idx_v, torch.full_like(idx_v, 3.0, dtype=torch.float64))
    n = n64.cpu().numpy()
    ok = n > 0
    def db(x):
        mse = np.maximum(np.asarray(x, dtype=np.float64)[ok] / n[ok], 1e-12)
        return -10.0 * np.log10(mse)
    d64, d32, d32b = db(a64.cpu().numpy()), db(a32.cpu().numpy()), db(a32b.cpu().numpy())
    rel = np.abs(a32.cpu().numpy()[ok] - a64.cpu().numpy()[ok]) / np.maximum(
        np.abs(a64.cpu().numpy()[ok]), 1e-30)
    return {
        "n_views": n_views,
        "n_cells_with_samples": int(ok.sum()),
        "max_channel_samples_in_one_cell": float(n[ok].max()),
        "total_channel_samples": float(n.sum()),
        "max_abs_dPSNR_db_fp32_vs_fp64": float(np.abs(d32 - d64).max()),
        "mean_abs_dPSNR_db_fp32_vs_fp64": float(np.abs(d32 - d64).mean()),
        "max_rel_err_sum_sq": float(rel.max()),
        "fp32_run_to_run_max_dPSNR_db": float(np.abs(d32 - d32b).max()),
        "fp32_exact_int_limit_2pow24": 16777216,
    }


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    info = {
        "device": dev,
        "torch": torch.__version__,
        "hip": getattr(torch.version, "hip", None),
        "cuda": torch.version.cuda,
    }
    if dev == "cuda":
        p = torch.cuda.get_device_properties(0)
        info["gpu_name"] = p.name
        info["gcn_arch"] = getattr(p, "gcnArchName", None)
    print("== environment ==")
    print(json.dumps(info, indent=2))

    H, W = 480, 640
    print("\n== whole add_view, fp64 vs fp32 (contention sweep at fixed 480x640) ==")
    # 70 / 212 / 691 are the MEASURED real fr3 operating points at
    # --streaming_local_quality_cell_size 2.0 / 1.0 / 0.5 (the default),
    # computed by scripts/local_quality_cell_count.py from the dataset's own
    # sensor depth + groundtruth poses over a 10-view holdout SPREAD ACROSS
    # THE WHOLE TRAJECTORY (per CLAUDE.md's real-holdout contract -- a
    # clustered sample of ten adjacent frames sees one corner of the room and
    # undercounts cells roughly 2x). The PR's own microbenchmark assumed 4000
    # cells, ~6x more than the default configuration actually reaches.
    rows = []
    for n_slots in (10, 70, 212, 691, 1000, 4000, 20000, 100000, 261120):
        r = bench_config(H, W, n_slots, dev)
        rows.append(r)
        print(json.dumps(r))

    print("\n== component breakdown @ 691 slots (real fr3 default operating point) ==")
    print(json.dumps(bench_components(H, W, 691, dev), indent=2))

    print("\n== precision: measured fp32-vs-fp64 bound @ 691 slots, 10 views ==")
    prec = bench_precision(H, W, 691, dev)
    print(json.dumps(prec, indent=2))

    print("\n== summary ==")
    worst = max(rows, key=lambda r: r["fp64_over_fp32"])
    print(f"worst fp64/fp32 ratio: {worst['fp64_over_fp32']}x at n_slots={worst['n_slots']} "
          f"({worst['px_per_slot']} px/slot)")
    for label, ns in (("PR-assumed 4000 cells", 4000),
                      ("REAL fr3 default, cell_size=0.5m", 691),
                      ("REAL fr3, cell_size=1.0m", 212),
                      ("REAL fr3, cell_size=2.0m", 70)):
        r = [x for x in rows if x["n_slots"] == ns][0]
        print(f"{label} (n_slots={ns}): PRE-hoist fp64={r['add_view_ms_fp64']}ms "
              f"fp32={r['add_view_ms_fp32']}ms ratio={r['fp64_over_fp32']}x "
              f"| POST-hoist fp64={r['hoisted_add_view_ms_fp64']}ms "
              f"fp32={r['hoisted_add_view_ms_fp32']}ms "
              f"ratio={r['hoisted_fp64_over_fp32']}x "
              f"(hoist speedup {r['hoist_speedup_fp64']}x) "
              f"-> 10-view tick fp64 {r['tick_ms_fp64']}ms -> "
              f"{r['hoisted_tick_ms_fp64']}ms")
    print("(CPU-torch reference quoted in PR #366: 22.9 ms/tick, 2.29 ms/view)")
    print(f"measured fp32 precision bound: max |dPSNR| = "
          f"{prec['max_abs_dPSNR_db_fp32_vs_fp64']:.3e} dB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
