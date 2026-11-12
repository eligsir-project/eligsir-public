#!/usr/bin/env python3
"""Materialize a ScanNet++ (v1) scene's iPhone RGB-D stream into this
project's generic RGB-D sequence format (frames.jsonl + intrinsics.json +
rgb/*.png + depth/*.png), auto-detected by scene/__init__.py.

ScanNet++ ships iPhone RGB as an mkv video (needs frame extraction) and
depth as depth.bin: a stream of per-frame length-prefixed compressed
chunks (lz4, falling back to zlib), NOT a fixed-stride raw array -- this
is the official scannetpp toolkit's own iphone/prepare_iphone_data.py
extract_depth() format, confirmed by fetching that script directly and
cross-checked against this scene's depth.bin's actual leading bytes
(first 4 bytes as little-endian uint32 decode to a plausible ~64KB
per-frame compressed-chunk size, not the ~3-8KB a raw-fixed-stride
192x256 uint16 frame would occupy).

Pose convention: pose_intrinsic_imu.json's "pose"/"aligned_pose" is a
4x4 camera-to-world matrix in OpenCV/COLMAP camera axes (X-right,
Y-down, Z-forward) -- confirmed by comparing a frame's aligned_pose
against the toolkit's own nerfstudio/transforms.json export for the
same frame: the two differ by exactly the standard ARKit/OpenCV ->
OpenGL/nerfstudio axis-flip (row-swap + sign-flip of the Y/Z columns
and translation Z), which is nerfstudio's own well-documented ARKit
ingestion convention. So aligned_pose is already the plain OpenCV-style
c2w this project's utils.rgbd_frames.c2w_to_camera_rt expects -- no
extra axis conversion needed. Uses "aligned_pose" (registered to the
scene's global mesh/DSLR alignment) rather than raw "pose".

Depth (192x256) is upsampled (nearest, to preserve edges) to match the
RGB frame's native resolution before writing, so intrinsics.json's
single fx/fy/cx/cy (already at RGB resolution in the source JSON) is
valid for both streams -- readRGBDSequenceSceneInfo's CameraInfo FoV
computation and rgbd_pointcloud_from_records' backprojection both
assume one shared intrinsics set at the depth-pixel-grid resolution
in the current codebase; upsampling depth to RGB res sidesteps that
without touching core reader code.

Usage:
    python scripts/prepare_scannetpp.py \
        --scannetpp-root /data/public/ScanNet++/data/data \
        --scene-id 8b5caf3398 \
        --out /tmp/scannetpp_cache/8b5caf3398 \
        [--stride 1] [--max-frames 0]
"""
from __future__ import annotations

import argparse
import io
import json
import os
import struct
import zlib

import cv2
import numpy as np

try:
    import lz4.block as lz4_block
except ImportError:
    lz4_block = None

from eligsir.utils.rgbd_frames import CameraIntrinsics, write_intrinsics, write_sequence_frame
from eligsir.utils.rgbd_frames import RGBDFrame

DEPTH_H, DEPTH_W = 192, 256


def iter_depth_frames(depth_bin_path):
    """Yield successive (height, width) float32-meters depth arrays from a
    ScanNet++ iphone/depth.bin, per-frame length-prefixed lz4/zlib chunks
    (scannetpp toolkit's own extract_depth() fallback branch -- confirmed
    to be this project's actual format, not the alternate whole-file-zlib
    branch, since this file's size doesn't correspond to any fixed-stride
    raw array)."""
    with open(depth_bin_path, "rb") as infile:
        while True:
            size_bytes = infile.read(4)
            if len(size_bytes) < 4:
                break
            (size,) = struct.unpack("<I", size_bytes)
            data = infile.read(size)
            if len(data) < size:
                break
            try:
                if lz4_block is None:
                    raise RuntimeError("lz4 not installed")
                raw = lz4_block.decompress(data, uncompressed_size=DEPTH_H * DEPTH_W * 2)
                depth = np.frombuffer(raw, dtype=np.uint16).reshape(DEPTH_H, DEPTH_W)
                depth_m = depth.astype(np.float32) / 1000.0
            except Exception:
                raw = zlib.decompress(data, wbits=-zlib.MAX_WBITS)
                depth = np.frombuffer(raw, dtype=np.float32).reshape(DEPTH_H, DEPTH_W)
                depth_m = depth
            yield depth_m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scannetpp-root", required=True)
    ap.add_argument("--scene-id", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stride", type=int, default=1, help="Keep every Nth frame (1 = all).")
    ap.add_argument("--max-frames", type=int, default=0, help="0 = no limit.")
    args = ap.parse_args()

    scene_dir = os.path.join(args.scannetpp_root, args.scene_id)
    iphone_dir = os.path.join(scene_dir, "iphone")
    pose_path = os.path.join(iphone_dir, "pose_intrinsic_imu.json")
    rgb_video_path = os.path.join(iphone_dir, "rgb.mkv")
    depth_bin_path = os.path.join(iphone_dir, "depth.bin")

    with open(pose_path) as f:
        pose_data = json.load(f)
    n_frames_total = len(pose_data)
    print(f"[scannetpp] scene={args.scene_id} n_frames(pose json)={n_frames_total}")

    cap = cv2.VideoCapture(rgb_video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {rgb_video_path} (opencv video backend)")

    os.makedirs(args.out, exist_ok=True)
    intr_written = False
    depth_gen = iter_depth_frames(depth_bin_path)

    n_written = 0
    frame_id = 0
    while True:
        ok, bgr = cap.read()
        depth_m = next(depth_gen, None)
        if not ok or depth_m is None:
            break
        frame_key = f"frame_{frame_id:06d}"
        finfo = pose_data.get(frame_key)
        if finfo is None:
            frame_id += 1
            continue

        if args.stride > 1 and (frame_id % args.stride != 0):
            frame_id += 1
            continue
        if args.max_frames and n_written >= args.max_frames:
            break

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h_rgb, w_rgb = rgb.shape[:2]

        if not intr_written:
            fx = float(finfo["intrinsic"][0][0])
            fy = float(finfo["intrinsic"][1][1])
            cx = float(finfo["intrinsic"][0][2])
            cy = float(finfo["intrinsic"][1][2])
            write_intrinsics(
                os.path.join(args.out, "intrinsics.json"),
                CameraIntrinsics(
                    width=w_rgb,
                    height=h_rgb,
                    fx=fx,
                    fy=fy,
                    cx=cx,
                    cy=cy,
                    depth_scale=1000.0,
                    camera_frame="iphone_lidar",
                    world_frame="scannetpp_aligned",
                    intrinsics_source="scannetpp_pose_intrinsic_imu",
                ),
            )
            intr_written = True

        depth_up = cv2.resize(depth_m, (w_rgb, h_rgb), interpolation=cv2.INTER_NEAREST)
        c2w = np.asarray(finfo["aligned_pose"], dtype=np.float32)

        frame = RGBDFrame(
            index=n_written,
            timestamp=float(finfo.get("timestamp", frame_id)),
            rgb=rgb,
            depth=depth_up,
            intrinsics=None,
            c2w=c2w,
        )
        write_sequence_frame(args.out, frame, depth_scale=1000.0)
        n_written += 1
        frame_id += 1

        if n_written % 500 == 0:
            print(f"[scannetpp] {args.scene_id}: {n_written} frames written...")

    cap.release()
    print(f"[scannetpp] {args.scene_id}: done, {n_written} frames written to {args.out}")
    print(f"[scannetpp] real n_source_frames for eval/scenes.json = {n_written}")


if __name__ == "__main__":
    main()
