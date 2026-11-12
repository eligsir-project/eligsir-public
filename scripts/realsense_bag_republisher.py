#!/usr/bin/env python3
"""Republish a native RealSense-Viewer .bag recording onto standard ROS2
realsense-ros topics, so a ROS2 SLAM frontend (ORB-SLAM3) can consume it.

Context (2026-08-01): the RealSense bags under
/data/realsense_handheld/ are ROS1
bagfiles recorded directly by the RealSense Viewer app's native librealsense
recorder -- NOT standard ROS2 bags, and NOT using the realsense-ros topic
convention our ingestion pipeline otherwise expects. utils/streaming_frames.py's
RealsenseRosBagFrameSource already reads these files directly via pyrealsense2's
native bag playback (rs.config.enable_device_from_file) for OFFLINE dataset-mode
training (--streaming_replay) -- that path needs no adapter and already works
(verified live 2026-08-01: 20/20 frames read from the hallway bag, first-frame
Open3D odometry fitness=0.86).

What that offline path does NOT provide is a live ROS2 topic feed for
ORB-SLAM3's ros2_orb_slam3 frontend to subscribe to -- ORB-SLAM3 needs a real
ROS2 node publishing on standard topics, not an in-process Python iterator.
This script bridges that gap: it opens the same bag via the same pyrealsense2
playback + align mechanism RealsenseRosBagFrameSource already uses (proven
working), and republishes each frame as sensor_msgs/Image (color + depth,
depth already aligned to color) and sensor_msgs/CameraInfo on the standard
realsense-ros ROS2 topics, at roughly the bag's native frame rate.

This is playback-only tooling for validating the tracking-confidence /
blur-exposure keyframe-quality mechanisms against genuinely degraded-tracking
data (see eligsir/mapping/scheduler/tracking_confidence.py's module
docstring) -- NOT part of the live-camera production path, and explicitly not
claimed to be. It has no bearing on the CLAUDE.md hard constraint that the
live path must use hardware-synced Orbbec ingestion; this only feeds an
offline validation scene through the same downstream ORB-SLAM3 frontend the
live path also uses, unmodified.

Usage:
    python3 realsense_bag_republisher.py <path/to/bag> [--rate 1.0] [--loop]

Publishes:
    /camera/color/image_raw            sensor_msgs/Image (rgb8)
    /camera/aligned_depth_to_color/image_raw   sensor_msgs/Image (16UC1, mm)
    /camera/color/camera_info          sensor_msgs/CameraInfo
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, CameraInfo


def build_camera_info(intr, frame_id: str) -> CameraInfo:
    msg = CameraInfo()
    msg.header.frame_id = frame_id
    msg.width = intr.width
    msg.height = intr.height
    msg.k = [intr.fx, 0.0, intr.ppx, 0.0, intr.fy, intr.ppy, 0.0, 0.0, 1.0]
    msg.p = [intr.fx, 0.0, intr.ppx, 0.0, 0.0, intr.fy, intr.ppy, 0.0, 0.0, 0.0, 1.0, 0.0]
    msg.d = list(intr.coeffs) if intr.coeffs else [0.0, 0.0, 0.0, 0.0, 0.0]
    msg.distortion_model = "plumb_bob"
    return msg


def numpy_to_image_msg(arr: np.ndarray, encoding: str, frame_id: str, stamp) -> Image:
    msg = Image()
    msg.header.frame_id = frame_id
    msg.header.stamp = stamp
    msg.height, msg.width = arr.shape[:2]
    msg.encoding = encoding
    msg.is_bigendian = 0
    msg.step = arr.strides[0]
    msg.data = arr.tobytes()
    return msg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("bag_path")
    ap.add_argument("--rate", type=float, default=1.0, help="Playback speed multiplier")
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--max-frames", type=int, default=0)
    args = ap.parse_args()

    try:
        import pyrealsense2 as rs
    except ImportError:
        print("ERROR: pyrealsense2 not available in this environment", file=sys.stderr)
        return 1

    rclpy.init()
    node = Node("realsense_bag_republisher")
    qos = QoSProfile(
        reliability=ReliabilityPolicy.RELIABLE,
        history=HistoryPolicy.KEEP_LAST,
        depth=10,
    )
    color_pub = node.create_publisher(Image, "/camera/color/image_raw", qos)
    depth_pub = node.create_publisher(Image, "/camera/aligned_depth_to_color/image_raw", qos)
    info_pub = node.create_publisher(CameraInfo, "/camera/color/camera_info", qos)

    n_published = 0
    try:
        while True:
            pipeline = rs.pipeline()
            cfg = rs.config()
            cfg.enable_device_from_file(str(args.bag_path), repeat_playback=False)
            cfg.enable_all_streams()
            profile = pipeline.start(cfg)
            playback = profile.get_device().as_playback()
            playback.set_real_time(False)

            color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
            intr = color_stream.get_intrinsics()
            need_bgr_to_rgb = color_stream.format() == rs.format.bgr8
            align = rs.align(rs.stream.color)

            info_msg = build_camera_info(intr, "camera_color_optical_frame")

            last_ts_ms = None
            while rclpy.ok():
                try:
                    frames = pipeline.wait_for_frames(timeout_ms=2000)
                except RuntimeError:
                    break  # end of bag
                frames = align.process(frames)
                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame()
                if not color_frame or not depth_frame:
                    continue

                ts_ms = frames.get_timestamp()
                if last_ts_ms is not None and args.rate > 0:
                    dt_s = max(0.0, (ts_ms - last_ts_ms) / 1000.0) / args.rate
                    if dt_s > 0:
                        time.sleep(min(dt_s, 1.0))
                last_ts_ms = ts_ms

                color_np = np.asanyarray(color_frame.get_data())
                if need_bgr_to_rgb:
                    color_np = color_np[:, :, ::-1]
                depth_np = np.asanyarray(depth_frame.get_data()).astype(np.uint16)
                # Depth validity filter (2026-08-02, root-caused a real ORB-SLAM3
                # crash): rs.align() can leave saturated/sentinel depth values
                # (observed: pixel value 65535 = the uint16 max, i.e. 65.5m at
                # this bag's 1mm depth units) in occluded/out-of-range/dropout
                # regions -- unfiltered, these become wildly wrong 3D points
                # once triangulated (DepthMapFactor=1000 in the ORB-SLAM3
                # config), degenerate the map geometry, and were the actual
                # cause of a "New Map created with N points" -> immediate
                # Sophus::SO3::exp NaN crash reproduced live on this exact bag
                # (traced by sampling raw aligned depth: 0% saturated at frame
                # 0, dominating by frame 250 -- correlates with the sensor's
                # own active-stereo dropout worsening over the trajectory, not
                # a bug in ORB-SLAM3 itself). Clamp anything at or above the
                # sensor's max representable range to 0 (ORB-SLAM3's own
                # invalid-depth convention), matching what the RealSense SDK's
                # own depth-frame validity semantics already imply but does
                # not enforce on raw get_data() output.
                depth_np[depth_np >= 65535] = 0
                # Also drop anything beyond a physically-sane indoor range for
                # this scene (D435i's own useful range tops out well under
                # 10m); keeps other, smaller-magnitude align artifacts (not
                # just the 65535 sentinel) from also corrupting triangulation.
                depth_np[depth_np > 10000] = 0

                stamp = node.get_clock().now().to_msg()
                color_msg = numpy_to_image_msg(color_np, "rgb8", "camera_color_optical_frame", stamp)
                depth_msg = numpy_to_image_msg(depth_np, "16UC1", "camera_color_optical_frame", stamp)
                info_msg.header.stamp = stamp

                color_pub.publish(color_msg)
                depth_pub.publish(depth_msg)
                info_pub.publish(info_msg)
                n_published += 1
                if n_published % 100 == 0:
                    node.get_logger().info(f"published {n_published} frames")
                if args.max_frames and n_published >= args.max_frames:
                    break

            pipeline.stop()
            node.get_logger().info(f"bag finished, {n_published} frames published")
            if not args.loop or (args.max_frames and n_published >= args.max_frames):
                break
    finally:
        node.destroy_node()
        rclpy.shutdown()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
