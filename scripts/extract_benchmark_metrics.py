#!/usr/bin/env python3
"""Extract key metrics from EliGSiR benchmark logs and output a summary table."""

import re
import sys
import json
from pathlib import Path


def extract_metrics(log_path: Path) -> dict:
    """Parse a training log and extract key benchmark metrics."""
    if not log_path.exists():
        return {"error": f"Log not found: {log_path}"}

    text = log_path.read_text()
    m = {}

    # Training duration
    t = re.search(r'\[timing\] training done: ([\d.]+)s', text)
    if t:
        m["training_time_s"] = float(t.group(1))

    # Final PSNR (test split from last [report] block)
    for line in reversed(text.splitlines()):
        p = re.search(r'\[report\] iter=\d+ test PSNR mean=([\d.]+)dB', line)
        if p:
            m["test_psnr_db"] = float(p.group(1))
            break
    for line in reversed(text.splitlines()):
        p = re.search(r'\[report\] iter=\d+ test LPIPS mean=([\d.]+)', line)
        if p:
            m["test_lpips"] = float(p.group(1))
            break

    # ATE from trajectory eval
    a = re.search(r'trajectory_eval.*ATE=([\d.e+\-]+)', text)
    if a:
        m["ate_rmse_m"] = float(a.group(1))

    # Gaussian count at end
    for line in reversed(text.splitlines()):
        n = re.search(r'N=([\d.]+e?\+?\d*),', line)
        if n:
            try:
                val = n.group(1)
                m["final_gaussian_count"] = int(float(val))
                break
            except Exception:
                pass

    # Total frames ingested
    f = re.search(r'frame=(\d+) .*Training complete', text)
    if not f:
        # try last frame= mention
        for line in reversed(text.splitlines()):
            ff = re.search(r'frame=(\d+)', line)
            if ff:
                m["frames_ingested"] = int(ff.group(1))
                break
    else:
        m["frames_ingested"] = int(f.group(1))

    # Average FPS from progress bar
    fps_matches = re.findall(r'(\d+\.\d+)it/s', text[-5000:])
    if fps_matches:
        m["avg_training_fps"] = float(fps_matches[-1])

    # Total inserted Gaussians
    ins = re.findall(r'total_inserted=(\d+)', text)
    if ins:
        m["total_depth_inserted"] = int(ins[-1])

    return m


def print_table(results: dict):
    """Print a markdown-style comparison table."""
    keys = [
        "test_psnr_db", "test_lpips", "ate_rmse_m",
        "final_gaussian_count", "frames_ingested",
        "total_depth_inserted", "training_time_s", "avg_training_fps"
    ]
    labels = {
        "test_psnr_db": "Test PSNR (dB)",
        "test_lpips": "Test LPIPS",
        "ate_rmse_m": "ATE RMSE (m)",
        "final_gaussian_count": "Gaussians (final)",
        "frames_ingested": "Frames ingested",
        "total_depth_inserted": "Depth-inserted",
        "training_time_s": "Train time (s)",
        "avg_training_fps": "Avg FPS",
    }

    scenes = sorted(results.keys())
    print(f"\n{'Metric':<25}", end="")
    for s in scenes:
        print(f"  {s:<20}", end="")
    print()
    print("-" * (25 + 22 * len(scenes)))

    for key in keys:
        print(f"{labels.get(key, key):<25}", end="")
        for s in scenes:
            val = results[s].get(key, "-")
            if isinstance(val, float):
                print(f"  {val:<20.4f}", end="")
            elif isinstance(val, int):
                print(f"  {val:<20,}", end="")
            else:
                print(f"  {str(val):<20}", end="")
        print()


if __name__ == "__main__":
    base = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("output/bench")
    results = {}
    for log in sorted(base.glob("*_run.log")):
        scene = log.stem.replace("_run", "")
        results[scene] = extract_metrics(log)
        print(f"\n=== {scene} ===")
        for k, v in results[scene].items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")

    print_table(results)

    # Write JSON summary
    summary_path = base / "benchmark_summary.json"
    summary_path.write_text(json.dumps(results, indent=2))
    print(f"\nSummary written to {summary_path}")
