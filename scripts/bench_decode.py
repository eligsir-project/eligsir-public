#!/usr/bin/env python3
"""Benchmark & correctness harness for the ROS2 live-ingestion CompressedImage
decode backends (``--ros2_live_rgbd_decode_backend {cv2,kornia}``, see
``eligsir/io/sources/ros2_live.py``).

Built for GitHub issue #123. Deliberately self-contained: it does NOT stand
up rclpy/ROS2, zenoh, or ORB-SLAM3 -- it only needs real JPEG-compressed
image bytes, which it can get two ways:

1. ``--jpeg-dir DIR``: a directory of real ``.jpg`` files (already-compressed
   frames from any source).
2. ``--bag PATH --topic /camera/rgb/image_color``: pulls real
   ``sensor_msgs/msg/Image`` (RAW, not CompressedImage) messages straight out
   of a ROS2 ``.db3`` sqlite3 file via a small embedded CDR deserializer (no
   rclpy needed -- this repo's CLAUDE.md flags "read db3 via python sqlite3,
   not mcap, no rclpy standup needed" as the established pattern for
   offline bag inspection), then re-JPEG-encodes each real frame at
   ``--jpeg-quality`` (default 90, matching typical image_transport
   defaults) to produce realistic compressed payloads. This is the path
   used for this issue's actual benchmark run because no bag with a real
   CompressedImage topic was available locally -- see
   docs/architecture/2026-08-12-kornia-decode-benchmark.md for the full
   accounting of that limitation. Either way the *pixel content* driving
   the benchmark is real camera data, not synthetic noise (noise doesn't
   compress/decode like a real photographic image and would bias timing).

Reports, for each available backend: throughput (frames/sec), per-frame
latency distribution (p50/p95/p99/max, after a warmup period is discarded),
and an explicit correctness check against cv2.imdecode (candidate decoders
must produce pixel-identical output -- a fast decoder that silently returns
different pixels is a data-corruption bug, not a win).

Usage:
    python scripts/bench_decode.py --jpeg-dir /path/to/jpegs
    python scripts/bench_decode.py --bag /path/to/bag_dir_or_file.db3 \\
        --topic /camera/rgb/image_color --n-frames 400 --output report.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
import struct
import sys
import time
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from eligsir.io.sources.ros2_live import (
    _decode_compressed_image_cv2,
    _decode_compressed_image_kornia,
)

try:
    import kornia_rs as _kornia_rs  # noqa: F401
    KORNIA_RS_AVAILABLE = True
except ImportError:
    KORNIA_RS_AVAILABLE = False


class _FakeCompressedImageMsg:
    """Minimal stand-in for sensor_msgs/msg/CompressedImage -- the decode
    helpers only touch .data and .format, so no rclpy/message type needed."""

    def __init__(self, data: bytes, fmt: str) -> None:
        self.data = data
        self.format = fmt


# ---------------------------------------------------------------------------
# Minimal CDR deserializer for sensor_msgs/msg/Image, no rclpy required.
# Standard CDR-LE encapsulation: 4-byte header, then fields aligned to their
# own size relative to the start of the encapsulated payload -- matches
# rclcpp/FastCDR's wire format for this message's field layout.
# ---------------------------------------------------------------------------
class _CdrReader:
    def __init__(self, buf: bytes):
        self.buf = buf
        self.off = 4
        self.base = 4

    def _align(self, size):
        rel = self.off - self.base
        pad = (-rel) % size
        self.off += pad

    def u8(self):
        v = self.buf[self.off]
        self.off += 1
        return v

    def u32(self):
        self._align(4)
        v = struct.unpack_from("<I", self.buf, self.off)[0]
        self.off += 4
        return v

    def i32(self):
        self._align(4)
        v = struct.unpack_from("<i", self.buf, self.off)[0]
        self.off += 4
        return v

    def string(self):
        n = self.u32()
        s = self.buf[self.off:self.off + n - 1].decode("utf-8")
        self.off += n
        return s

    def bytes_array(self):
        n = self.u32()
        b = self.buf[self.off:self.off + n]
        self.off += n
        return b


def _parse_raw_image(blob: bytes) -> dict:
    r = _CdrReader(blob)
    r.i32()  # header.stamp.sec (unused)
    r.u32()  # header.stamp.nanosec (unused)
    r.string()  # header.frame_id (unused)
    height = r.u32()
    width = r.u32()
    encoding = r.string()
    r.u8()  # is_bigendian (unused; this repo's sources are all little-endian)
    step = r.u32()
    data = r.bytes_array()
    return {"height": height, "width": width, "encoding": encoding, "step": step, "data": data}


def _load_frames_from_bag(bag_path: str, topic: str, n_frames: int, jpeg_quality: int) -> List[bytes]:
    """Real RAW frames -> real JPEG-compressed payloads (see module docstring)."""
    import sqlite3

    db3_candidates = [bag_path] if bag_path.endswith(".db3") else glob.glob(os.path.join(bag_path, "*.db3"))
    if not db3_candidates:
        raise FileNotFoundError(f"no .db3 file found at/under {bag_path!r}")
    conn = sqlite3.connect(f"file:{db3_candidates[0]}?mode=ro", uri=True)
    cur = conn.cursor()
    cur.execute("SELECT id, type FROM topics WHERE name = ?", (topic,))
    row = cur.fetchone()
    if row is None:
        cur.execute("SELECT name FROM topics")
        available = [r[0] for r in cur.fetchall()]
        raise ValueError(f"topic {topic!r} not found in {db3_candidates[0]}; available: {available}")
    topic_id, topic_type = row
    if "CompressedImage" in topic_type:
        cur.execute("SELECT data FROM messages WHERE topic_id = ? ORDER BY timestamp", (topic_id,))
        rows = cur.fetchall()
        if not rows:
            raise ValueError(f"no messages on {topic!r}")
        idxs = np.linspace(0, len(rows) - 1, min(n_frames, len(rows))).astype(int)
        return [bytes(rows[i][0]) for i in idxs]

    if "sensor_msgs/msg/Image" not in topic_type:
        raise ValueError(f"topic {topic!r} has unsupported type {topic_type!r} for this bench")

    cur.execute("SELECT data FROM messages WHERE topic_id = ? ORDER BY timestamp", (topic_id,))
    rows = cur.fetchall()
    if not rows:
        raise ValueError(f"no messages on {topic!r}")
    idxs = np.linspace(0, len(rows) - 1, min(n_frames, len(rows))).astype(int)

    jpeg_bytes = []
    for i in idxs:
        img = _parse_raw_image(bytes(rows[i][0]))
        arr = np.frombuffer(img["data"], dtype=np.uint8)
        if img["encoding"] in ("rgb8", "bgr8"):
            arr = arr.reshape(img["height"], img["width"], 3)
            if img["encoding"] == "rgb8":
                arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        elif img["encoding"] == "mono8":
            arr = arr.reshape(img["height"], img["width"])
        else:
            raise ValueError(f"unsupported raw encoding {img['encoding']!r} for JPEG re-encode")
        ok, buf = cv2.imencode(".jpg", arr, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        if not ok:
            raise RuntimeError("cv2.imencode failed while preparing benchmark payload")
        jpeg_bytes.append(buf.tobytes())
    return jpeg_bytes


def _load_frames_from_dir(jpeg_dir: str, n_frames: int) -> List[bytes]:
    paths = sorted(glob.glob(os.path.join(jpeg_dir, "*.jpg")) + glob.glob(os.path.join(jpeg_dir, "*.jpeg")))
    if not paths:
        raise FileNotFoundError(f"no .jpg/.jpeg files found in {jpeg_dir!r}")
    paths = paths[:n_frames]
    out = []
    for p in paths:
        with open(p, "rb") as f:
            out.append(f.read())
    return out


def _percentile(values: List[float], p: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    k = (len(s) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


def _time_backend_once(decode_fn, msgs) -> tuple:
    """One timed pass over msgs (no warmup here -- caller handles that
    once, up front, shared across repeats). Returns (throughput_fps,
    per-frame latencies in ms)."""
    latencies_ms = []
    t_start = time.perf_counter()
    for msg in msgs:
        t0 = time.perf_counter()
        decode_fn(msg)
        latencies_ms.append((time.perf_counter() - t0) * 1e3)
    total_s = time.perf_counter() - t_start
    throughput_fps = len(msgs) / total_s if total_s > 0 else float("nan")
    return throughput_fps, latencies_ms


def _time_backend(decode_fn, msgs, warmup: int, repeats: int = 5) -> dict:
    """Times `decode_fn` over `msgs`, `repeats` independent passes, after a
    single shared warmup pass. A lone timing pass on a shared, busy
    development box is not trustworthy on its own -- host-run variance of
    up to ~35% between otherwise-identical passes was observed during this
    investigation (see docs/architecture/2026-08-12-kornia-decode-benchmark.md),
    almost certainly from other CPU load on the machine (this project runs
    concurrent GPU ablations that still consume host CPU for orchestration/
    logging). Reporting the median throughput across independent repeats
    (rather than a single pass, or a single global total) is far less
    sensitive to one repeat getting unlucky with a scheduling hiccup."""
    for msg in msgs[:warmup]:
        decode_fn(msg)

    timed = msgs[warmup:] if len(msgs) > warmup else msgs
    throughputs = []
    all_latencies_ms = []
    for _ in range(repeats):
        fps, latencies_ms = _time_backend_once(decode_fn, timed)
        throughputs.append(fps)
        all_latencies_ms.extend(latencies_ms)

    return {
        "n_frames_timed": len(timed),
        "repeats": repeats,
        "throughput_fps_per_repeat": throughputs,
        "throughput_fps_median": statistics.median(throughputs),
        "throughput_fps_min": min(throughputs),
        "throughput_fps_max": max(throughputs),
        # Kept for backward-compat with earlier report.json consumers/docs.
        "throughput_fps": statistics.median(throughputs),
        "latency_ms": {
            "mean": statistics.mean(all_latencies_ms) if all_latencies_ms else float("nan"),
            "p50": _percentile(all_latencies_ms, 50),
            "p95": _percentile(all_latencies_ms, 95),
            "p99": _percentile(all_latencies_ms, 99),
            "max": max(all_latencies_ms) if all_latencies_ms else float("nan"),
        },
    }


def _check_correctness(msgs) -> dict:
    """Decode every frame with both backends and require pixel-identical
    output -- a silent mismatch here is a data-corruption bug, not noise."""
    mismatches = 0
    max_abs_diff = 0
    shape_mismatches = 0
    for msg in msgs:
        arr_cv2, enc_cv2 = _decode_compressed_image_cv2(msg)
        arr_kornia, enc_kornia = _decode_compressed_image_kornia(msg)
        if arr_cv2.shape != arr_kornia.shape or enc_cv2 != enc_kornia:
            shape_mismatches += 1
            continue
        if not np.array_equal(arr_cv2, arr_kornia):
            mismatches += 1
            diff = int(np.abs(arr_cv2.astype(np.int16) - arr_kornia.astype(np.int16)).max())
            max_abs_diff = max(max_abs_diff, diff)
    return {
        "n_checked": len(msgs),
        "shape_or_label_mismatches": shape_mismatches,
        "pixel_mismatches": mismatches,
        "max_abs_pixel_diff": max_abs_diff,
        "bit_exact": mismatches == 0 and shape_mismatches == 0,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jpeg-dir", type=str, default=None, help="Directory of real .jpg frames.")
    ap.add_argument("--bag", type=str, default=None, help="Path to a ROS2 bag (dir or .db3 file).")
    ap.add_argument("--topic", type=str, default="/camera/rgb/image_color",
                     help="Topic to pull frames from (CompressedImage or raw sensor_msgs/Image).")
    ap.add_argument("--n-frames", type=int, default=400, help="Number of frames to sample (evenly across the bag).")
    ap.add_argument("--jpeg-quality", type=int, default=90, help="Re-encode quality when source topic is raw Image.")
    ap.add_argument("--warmup", type=int, default=20, help="Frames to decode (and discard) before timing.")
    ap.add_argument("--repeats", type=int, default=5, help="Independent timed passes per backend; throughput is reported as the median across repeats.")
    ap.add_argument("--fmt", type=str, default="bgr8; jpeg compressed bgr8",
                     help="CompressedImage .format string to attach to synthesized messages.")
    ap.add_argument("--output", type=str, default=None, help="Optional path to write the JSON report.")
    args = ap.parse_args()

    if not args.jpeg_dir and not args.bag:
        ap.error("one of --jpeg-dir or --bag is required")

    if args.jpeg_dir:
        raw_jpeg_bytes = _load_frames_from_dir(args.jpeg_dir, args.n_frames)
        source_desc = {"kind": "jpeg_dir", "path": args.jpeg_dir}
    else:
        raw_jpeg_bytes = _load_frames_from_bag(args.bag, args.topic, args.n_frames, args.jpeg_quality)
        source_desc = {"kind": "bag", "path": args.bag, "topic": args.topic, "jpeg_quality": args.jpeg_quality}

    msgs = [_FakeCompressedImageMsg(data, args.fmt) for data in raw_jpeg_bytes]
    print(f"[bench_decode] loaded {len(msgs)} real JPEG frames from {source_desc}")

    report = {
        "source": source_desc,
        "n_frames": len(msgs),
        "warmup_frames": args.warmup,
        "kornia_rs_available": KORNIA_RS_AVAILABLE,
        "backends": {},
    }

    print("\n[bench_decode] timing cv2 backend...")
    cv2_decode = lambda msg: _decode_compressed_image_cv2(msg)
    report["backends"]["cv2"] = _time_backend(cv2_decode, msgs, args.warmup, args.repeats)

    if KORNIA_RS_AVAILABLE:
        print("[bench_decode] timing kornia backend...")
        kornia_decode = lambda msg: _decode_compressed_image_kornia(msg)
        report["backends"]["kornia"] = _time_backend(kornia_decode, msgs, args.warmup, args.repeats)

        print("[bench_decode] checking correctness (kornia vs cv2, pixel-exact)...")
        report["correctness"] = _check_correctness(msgs)
    else:
        print("[bench_decode] kornia_rs not importable in this environment -- "
              "skipping kornia timing/correctness (this is a legitimate, "
              "reportable finding: see docs/architecture/2026-08-12-kornia-decode-benchmark.md).")
        report["correctness"] = None

    print("\n=== bench_decode.py summary ===")
    for name, stats in report["backends"].items():
        lat = stats["latency_ms"]
        print(f"  {name:8s}  throughput(median)={stats['throughput_fps_median']:8.1f} fps  "
              f"(min={stats['throughput_fps_min']:.1f} max={stats['throughput_fps_max']:.1f} over {stats['repeats']} repeats)  "
              f"p50={lat['p50']:6.3f}ms  p95={lat['p95']:6.3f}ms  p99={lat['p99']:6.3f}ms  max={lat['max']:7.3f}ms")
    if "cv2" in report["backends"] and "kornia" in report["backends"]:
        speedup = (report["backends"]["kornia"]["throughput_fps_median"]
                   / report["backends"]["cv2"]["throughput_fps_median"])
        print(f"  kornia/cv2 throughput ratio: {speedup:.3f}x")
        report["kornia_vs_cv2_throughput_ratio"] = speedup
    if report["correctness"] is not None:
        c = report["correctness"]
        verdict = "BIT-EXACT" if c["bit_exact"] else "MISMATCH DETECTED"
        print(f"  correctness: {verdict}  ({c['n_checked']} frames checked, "
              f"{c['pixel_mismatches']} pixel mismatches, {c['shape_or_label_mismatches']} shape/label mismatches, "
              f"max_abs_pixel_diff={c['max_abs_pixel_diff']})")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\n[bench_decode] wrote {args.output}")


if __name__ == "__main__":
    main()
