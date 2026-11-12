"""Offline (no live ROS2) reader for RGB/intrinsics topics in an archived ROS2
bag. Mirrors the sqlite3(.db3)/rosbag2_py(.mcap) dual-path pattern already used
by scripts/eval_tum.py -- deterministic, no SLAM, no live pub/sub.
"""
import glob
import os

import cv2
import numpy as np


def _iter_bag_messages(bag_path):
    """Yield (topic_name, raw_blob, type_map) for every message, in file order."""
    db3 = glob.glob(os.path.join(bag_path, "*.db3"))
    if db3:
        import sqlite3
        con = sqlite3.connect(f"file:{db3[0]}?mode=ro", uri=True)
        type_map = {n: t for (n, t) in con.execute("SELECT name, type FROM topics")}
        tid2name = {i: n for (i, n) in con.execute("SELECT id, name FROM topics")}
        for tid, blob in con.execute("SELECT topic_id, data FROM messages ORDER BY timestamp"):
            name = tid2name.get(tid)
            if name is not None:
                yield name, bytes(blob), type_map
        con.close()
    else:
        from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
        reader = SequentialReader()
        reader.open(StorageOptions(uri=bag_path, storage_id="mcap"), ConverterOptions("cdr", "cdr"))
        type_map = {t.name: t.type for t in reader.get_all_topics_and_types()}
        while reader.has_next():
            name, blob, _ = reader.read_next()
            yield name, blob, type_map


def _deserialize(topic_type_name, blob):
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
    return deserialize_message(blob, get_message(topic_type_name))


def _stamp(header):
    return header.stamp.sec + header.stamp.nanosec * 1e-9


def read_camera_info(bag_path, topic="/camera/color/camera_info"):
    for name, blob, type_map in _iter_bag_messages(bag_path):
        if topic not in type_map:
            raise ValueError(f"topic {topic!r} not found in bag {bag_path}")
        if name != topic:
            continue
        msg = _deserialize(type_map[topic], blob)
        return {
            "width": int(msg.width),
            "height": int(msg.height),
            "fx": float(msg.k[0]),
            "fy": float(msg.k[4]),
            "cx": float(msg.k[2]),
            "cy": float(msg.k[5]),
        }
    raise ValueError(f"no {topic} message found in {bag_path}")


def read_rgb_frames_near_timestamps(bag_path, timestamps, topic="/camera/color/image_raw/compressed",
                                     max_dt=0.02):
    remaining = {float(t): None for t in timestamps}
    best_dt = {t: float("inf") for t in remaining}

    for name, blob, type_map in _iter_bag_messages(bag_path):
        if topic not in type_map:
            raise ValueError(f"topic {topic!r} not found in bag {bag_path}")
        if name != topic or not remaining:
            continue
        msg = _deserialize(type_map[topic], blob)
        msg_ts = _stamp(msg.header)
        for t in remaining:
            dt = abs(msg_ts - t)
            if dt <= max_dt and dt < best_dt[t]:
                buf = np.frombuffer(bytes(msg.data), dtype=np.uint8)
                bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                remaining[t] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                best_dt[t] = dt

    missing = [t for t, frame in remaining.items() if frame is None]
    if missing:
        raise ValueError(
            f"no match within {max_dt}s of timestamp(s) {missing} on topic {topic} in {bag_path}"
        )
    return remaining
