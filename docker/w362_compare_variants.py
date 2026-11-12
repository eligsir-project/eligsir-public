"""W3-62: compare two docker/w362_synthetic_probe.py .pt outputs (strict vs.
fast-math ROCm gsplat builds) and print per-state, per-tensor divergence
stats. Does NOT need gsplat installed -- only torch, so it can run in any
environment (host python, a plain torch container, etc.).

Usage:
    python3 docker/w362_compare_variants.py probe_strict.pt probe_fastmath.pt
"""
import argparse
import json
import sys

import torch


def compare_tensor(a: torch.Tensor, b: torch.Tensor):
    a = a.double()
    b = b.double()
    diff = (a - b)
    nonfinite_a = int((~torch.isfinite(a)).sum().item())
    nonfinite_b = int((~torch.isfinite(b)).sum().item())
    finite_mask = torch.isfinite(a) & torch.isfinite(b)
    if finite_mask.sum() == 0:
        return {
            "nonfinite_a": nonfinite_a,
            "nonfinite_b": nonfinite_b,
            "all_nonfinite": True,
        }
    af = a[finite_mask]
    bf = b[finite_mask]
    df = diff[finite_mask]
    max_abs_a = af.abs().max().item()
    max_abs_b = bf.abs().max().item()
    p99_abs_a = torch.quantile(af.abs(), 0.99).item() if af.numel() > 1 else af.abs().max().item()
    p99_abs_b = torch.quantile(bf.abs(), 0.99).item() if bf.numel() > 1 else bf.abs().max().item()
    max_abs_diff = df.abs().max().item()
    rel_l2 = (df.norm() / (af.norm() + 1e-12)).item()
    denom = (af.norm() * bf.norm())
    cos_sim = (af.flatten() @ bf.flatten() / denom).item() if denom.item() > 0 else float("nan")
    return {
        "nonfinite_a": nonfinite_a,
        "nonfinite_b": nonfinite_b,
        "all_nonfinite": False,
        "max_abs_a": max_abs_a,
        "max_abs_b": max_abs_b,
        "p99_abs_a": p99_abs_a,
        "p99_abs_b": p99_abs_b,
        "max_abs_diff": max_abs_diff,
        "rel_l2_error": rel_l2,
        "cosine_similarity": cos_sim,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("probe_a")
    ap.add_argument("probe_b")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    a = torch.load(args.probe_a, weights_only=False)
    b = torch.load(args.probe_b, weights_only=False)

    print(f"A: variant={a['variant']} gsplat={a['gsplat_version']} torch={a['torch_version']} device={a['device_name']}")
    print(f"B: variant={b['variant']} gsplat={b['gsplat_version']} torch={b['torch_version']} device={b['device_name']}")

    if a["input_hashes"] != b["input_hashes"]:
        print("!!! INPUT HASHES DIFFER -- A and B were NOT fed identical inputs. Results below are NOT a valid comparison.")
        sys.exit(2)
    else:
        print("Input hashes match across all states: A and B received byte-identical inputs.")

    report = {}
    for name in a["results"]:
        ra = a["results"][name]
        rb = b["results"].get(name)
        entry = {}
        if ra.get("crash") or (rb is not None and rb.get("crash")):
            entry["crash_a"] = bool(ra.get("crash"))
            entry["crash_b"] = bool(rb.get("crash")) if rb else None
            report[name] = entry
            print(f"\n=== {name} === crash_a={entry['crash_a']} crash_b={entry['crash_b']}")
            continue

        entry["loss_a"] = ra["loss"]
        entry["loss_b"] = rb["loss"]
        entry["loss_abs_diff"] = abs(ra["loss"] - rb["loss"])

        print(f"\n=== {name} === loss_a={ra['loss']:.8f} loss_b={rb['loss']:.8f} diff={entry['loss_abs_diff']:.3e}")

        for field in ("rgb", "depth", "alpha"):
            stat = compare_tensor(ra[field], rb[field])
            entry[field] = stat
            print(f"  {field:8s}: {stat}")

        entry["grads"] = {}
        for pname in ra["grads"]:
            ga = ra["grads"][pname]
            gb = rb["grads"].get(pname)
            if ga is None or gb is None:
                entry["grads"][pname] = {"grad_a_none": ga is None, "grad_b_none": gb is None}
                print(f"  grad[{pname:10s}]: a_none={ga is None} b_none={gb is None}")
                continue
            stat = compare_tensor(ga, gb)
            entry["grads"][pname] = stat
            print(
                f"  grad[{pname:10s}]: nonfin(a/b)={stat.get('nonfinite_a')}/{stat.get('nonfinite_b')} "
                f"max|diff|={stat.get('max_abs_diff', float('nan')):.3e} "
                f"rel_l2={stat.get('rel_l2_error', float('nan')):.3e} "
                f"cos_sim={stat.get('cosine_similarity', float('nan')):.8f} "
                f"max|a|={stat.get('max_abs_a', float('nan')):.3e} "
                f"p99|a|={stat.get('p99_abs_a', float('nan')):.3e}"
            )
        report[name] = entry

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nWrote full JSON report to {args.json_out}")


if __name__ == "__main__":
    main()
