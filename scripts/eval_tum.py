#!/usr/bin/env python3
"""TUM RGB-D trajectory eval: /slam/pose (raw) and optional /slam/path (loop-
closure corrected) vs the TUM mocap groundtruth.txt, in a single pass.

TUM groundtruth is a separate text file (mocap), NOT a baked /camera_pose topic:
  # timestamp tx ty tz qx qy qz qw
Stamps in the ros2-converted bags are the original TUM Unix times, so they
associate directly. ATE is computed after Umeyama SE(3) alignment (RGB-D is
metric -> no scale), matching the TUM benchmark.

Usage: eval_tum.py <bag> --gt groundtruth.txt [--est /slam/pose]
                    [--path /slam/path] [--max-dt 0.02] [--json out.json]
"""
import argparse
import glob
import json
import sys
import numpy as np

from eligsir.utils.trajectory_eval import associate_by_timestamp, trajectory_metrics, _quat_to_rot
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


def _T_from_pose(pos, quat_wxyz):
    M = np.eye(4)
    M[:3, :3] = _quat_to_rot(quat_wxyz)
    M[:3, 3] = pos
    return M


def _T_msg(pose):
    p, q = pose.position, pose.orientation
    return _T_from_pose([p.x, p.y, p.z], np.array([q.w, q.x, q.y, q.z]))


def _stamp(h):
    return h.stamp.sec + h.stamp.nanosec * 1e-9


def load_gt(path):
    ts, Ts = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            v = [float(x) for x in line.split()]
            if len(v) < 8:
                continue
            t, tx, ty, tz, qx, qy, qz, qw = v[:8]
            ts.append(t)
            Ts.append(_T_from_pose([tx, ty, tz], np.array([qw, qx, qy, qz])))
    return np.asarray(ts), (np.stack(Ts) if Ts else np.empty((0, 4, 4)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bag")
    ap.add_argument("--gt", required=True)
    ap.add_argument("--est", default="/slam/pose")
    ap.add_argument("--path", default="/slam/path")
    ap.add_argument("--max-dt", type=float, default=0.02)
    ap.add_argument("--json", default="")
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    ref_ts, ref_T = load_gt(args.gt)
    est_ts, est_T = [], []
    last_path = None

    def route(name, blob, tmap):
        nonlocal last_path
        if name == args.est and name in tmap:
            m = deserialize_message(blob, get_message(tmap[name]))
            est_ts.append(_stamp(m.header)); est_T.append(_T_msg(m.pose))
        elif name == args.path and name in tmap:
            last_path = deserialize_message(blob, get_message(tmap[name]))

    db3 = glob.glob(f"{args.bag}/*.db3")
    if db3:
        import sqlite3
        con = sqlite3.connect(f"file:{db3[0]}?mode=ro", uri=True)
        tmap = {n: t for (n, t) in con.execute("SELECT name, type FROM topics")}
        tid2name = {i: n for (i, n) in con.execute("SELECT id, name FROM topics")}
        for tid, blob in con.execute("SELECT topic_id, data FROM messages ORDER BY timestamp"):
            route(tid2name.get(tid), bytes(blob), tmap)
        con.close()
    else:
        from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
        reader = SequentialReader()
        reader.open(StorageOptions(uri=args.bag, storage_id="mcap"), ConverterOptions("cdr", "cdr"))
        tmap = {t.name: t.type for t in reader.get_all_topics_and_types()}
        while reader.has_next():
            name, blob, _ = reader.read_next()
            route(name, blob, tmap)

    def evaluate(ts, Ts, label):
        ts = np.asarray(ts); Ts = np.stack(Ts) if len(Ts) else np.empty((0, 4, 4))
        if not len(ts) or not len(ref_ts):
            return {"label": label, "n_est": int(len(ts)), "ate_rmse": None}
        _, em, rm, info = associate_by_timestamp(ts, Ts, ref_ts, ref_T, max_dt=args.max_dt)
        mt = trajectory_metrics(em, rm) if len(em) else {"ate_rmse": None, "rpe_trans_rmse": None}
        return {"label": label, "n_est": int(len(ts)), "n_matches": info.get("n_matches", 0),
                "ate_rmse": mt["ate_rmse"], "rpe_trans_rmse": mt.get("rpe_trans_rmse")}

    raw = evaluate(est_ts, est_T, "raw")
    if last_path is not None and last_path.poses:
        kf_ts = np.asarray([_stamp(ps.header) for ps in last_path.poses])
        kf_T = [_T_msg(ps.pose) for ps in last_path.poses]
        pth = evaluate(kf_ts, kf_T, "corrected")
        if len(est_ts):
            est_ts_a = np.asarray(est_ts)
            sel = [int(np.argmin(np.abs(est_ts_a - t))) for t in kf_ts]
            raw_kf = evaluate(kf_ts, [est_T[i] for i in sel], "raw_at_kf")
        else:
            raw_kf = {"label": "raw_at_kf", "n_est": 0, "ate_rmse": None}
    else:
        pth = {"label": "corrected", "n_est": 0, "ate_rmse": None}
        raw_kf = {"label": "raw_at_kf", "n_est": 0, "ate_rmse": None}

    out = {"label": args.label, "bag": args.bag, "gt": args.gt, "raw": raw,
           "raw_at_kf": raw_kf, "corrected": pth, "n_ref": int(len(ref_ts))}
    print(json.dumps(out))
    for r in (raw, raw_kf, pth):
        a = r["ate_rmse"]
        print(f"[eval] {r['label']:9s} ATE={'%.4fm' % a if a is not None else 'n/a':>9s}"
              f"  n_est={r['n_est']} matches={r.get('n_matches', 0)}"
              f"  RPE_t={r.get('rpe_trans_rmse')}", file=sys.stderr)
    base = raw_kf if raw_kf.get("ate_rmse") else raw
    if base["ate_rmse"] and pth["ate_rmse"]:
        print(f"[eval] drift reduction ({base['label']}): {base['ate_rmse']*1000:.0f}mm -> "
              f"{pth['ate_rmse']*1000:.0f}mm "
              f"({100*(1-pth['ate_rmse']/base['ate_rmse']):.0f}% lower)", file=sys.stderr)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
