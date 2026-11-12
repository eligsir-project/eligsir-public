import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
from PIL import Image

# Issue #1074: this module no longer defines its own CameraIntrinsics. The
# canonical record lives in eligsir.contracts.types and this is a plain
# re-export so the serializers below (and every historical importer) keep
# working against the one definition. New code should import it from
# eligsir.contracts directly.
from eligsir.contracts.types import CameraIntrinsics  # noqa: F401


@dataclass
class RGBDFrame:
    index: int
    timestamp: float
    rgb: np.ndarray
    depth: np.ndarray
    intrinsics: CameraIntrinsics
    c2w: np.ndarray
    rgb_path: Optional[str] = None
    depth_path: Optional[str] = None
    # Issue #745 round 2 (2026-08-15): a monotonic counter assigned by
    # ROS2LiveRGBDSource.__iter__'s on_rgb callback (eligsir/io/sources/
    # ros2_live.py) at the moment a raw color message is first accepted from
    # ROS2 -- BEFORE any of that module's own drop/reject decisions
    # (queue_overflow_dropped_oldest, reject_depth_stale_checks, reject_pose_stale_checks,
    # etc.). A message that is later dropped or rejected simply consumes its
    # counter value and produces no frame; every message that *does* become
    # a frame carries the counter value assigned at its own acceptance time,
    # so no drop anywhere downstream of assignment can renumber a later
    # frame's value. None for any frame source that doesn't populate it
    # (e.g. offline/bag-parsed sources, or a synthetic test harness that
    # bypasses on_rgb) -- consumers must treat None as "not provided" and
    # fall back accordingly (see eligsir/io/sources/datasets/
    # ros2_live.py's `_pump_loop`, which falls back to timestamp
    # quantization when this is None).
    received_seq: Optional[int] = None


class RGBDFrameSource:
    def __iter__(self) -> Iterator[RGBDFrame]:
        raise NotImplementedError


def read_intrinsics(path):
    """Read an ``intrinsics.json`` into the canonical CameraIntrinsics.

    Issue #1074: the canonical record carries separate depth-sensor intrinsics
    (``depth_fx``/.../``depth_height``), which the pre-merge serializer type did
    not have. ``write_intrinsics`` emits them via ``asdict``, so this reader
    must consume them too -- otherwise a source with genuinely distinct depth
    intrinsics (ScanNet: 640x480 depth vs 1296x968 colour) would write them
    successfully and silently lose them on read, falling back to the colour
    values. Absent keys still default to the colour intrinsics, so files
    written before this change read back unchanged.
    """
    with open(path, "r") as f:
        data = json.load(f)
    distortion = data.get("D", [])

    def _opt(key, cast):
        value = data.get(key)
        return None if value is None else cast(value)

    return CameraIntrinsics(
        width=int(data["width"]),
        height=int(data["height"]),
        fx=float(data["fx"]),
        fy=float(data["fy"]),
        cx=float(data["cx"]),
        cy=float(data["cy"]),
        depth_fx=_opt("depth_fx", float),
        depth_fy=_opt("depth_fy", float),
        depth_cx=_opt("depth_cx", float),
        depth_cy=_opt("depth_cy", float),
        depth_width=_opt("depth_width", int),
        depth_height=_opt("depth_height", int),
        depth_scale=float(data.get("depth_scale", 1000.0)),
        camera_frame=str(data.get("camera_frame", "")),
        world_frame=str(data.get("world_frame", "")),
        distortion_model=str(data.get("distortion_model", "")),
        D=tuple(distortion) if distortion is not None else (),
        intrinsics_source=str(data.get("intrinsics_source", "K")),
    )


def write_intrinsics(path, intrinsics):
    data = asdict(intrinsics)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def read_metadata(path):
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        return json.load(f)


def write_metadata(path, metadata):
    with open(path, "w") as f:
        json.dump(metadata, f, indent=2)
        f.write("\n")


def read_frame_records(path):
    records = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def append_frame_record(path, record):
    with open(path, "a") as f:
        f.write(json.dumps(record, separators=(",", ":")))
        f.write("\n")


def ensure_sequence_dirs(path):
    root = Path(path)
    (root / "rgb").mkdir(parents=True, exist_ok=True)
    (root / "depth").mkdir(parents=True, exist_ok=True)
    return root


def write_sequence_frame(root, frame, depth_scale=None):
    from PIL import Image

    root = ensure_sequence_dirs(root)
    depth_scale = float(depth_scale if depth_scale is not None else frame.intrinsics.depth_scale)
    rgb_rel = f"rgb/{frame.index:06d}.png"
    depth_rel = f"depth/{frame.index:06d}.png"

    Image.fromarray(np.asarray(frame.rgb, dtype=np.uint8)).save(root / rgb_rel)
    depth_arr = np.asarray(frame.depth)
    if depth_arr.dtype == np.uint16:
        # frame.depth can already be raw uint16 counts (e.g. a rosbag 16UC1/
        # mono16 topic decoded by decode_depth_image without conversion to
        # metres -- see eligsir/io/sources/rosbag.py). Those counts are
        # in frame.intrinsics.depth_scale's domain, not necessarily this
        # call's depth_scale (which defaults to the same value but can be
        # overridden). Say so explicitly (issue #642) instead of silently
        # trusting a pass-through.
        depth_png = depth_to_uint16_png(
            depth_arr, depth_scale, uint16_source_scale=frame.intrinsics.depth_scale
        )
    else:
        depth_png = depth_to_uint16_png(depth_arr, depth_scale)
    Image.fromarray(depth_png).save(root / depth_rel)

    record = {
        "id": int(frame.index),
        "timestamp": float(frame.timestamp),
        "rgb": rgb_rel,
        "depth": depth_rel,
        "c2w": np.asarray(frame.c2w, dtype=np.float32).tolist(),
    }
    append_frame_record(root / "frames.jsonl", record)
    return record


def decode_color_image(data, encoding, encoding_override=None):
    import cv2

    enc = str(encoding).lower()
    if encoding_override:
        enc = str(encoding_override).lower()
    arr = np.asarray(data)
    if str(encoding).lower() == "8uc3" and not encoding_override:
        raise ValueError(
            "Ambiguous RGB image encoding 8UC3. Pass --rgb-encoding-override rgb8 or bgr8."
        )
    if enc == "rgb8":
        rgb = arr
    elif enc == "bgr8":
        rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    elif enc == "rgba8":
        rgb = arr[..., :3]
    elif enc == "bgra8":
        rgb = cv2.cvtColor(arr, cv2.COLOR_BGRA2RGB)
    elif enc in {"mono8", "8uc1"}:
        rgb = np.repeat(arr[..., None], 3, axis=2)
    else:
        raise ValueError(f"Unsupported RGB image encoding: {encoding}")
    return np.ascontiguousarray(rgb.astype(np.uint8))


def decode_depth_image(data, encoding):
    enc = str(encoding).lower()
    arr = np.asarray(data)
    if enc in {"16uc1", "mono16", "uint16"}:
        return arr.astype(np.uint16, copy=False)
    if enc in {"32fc1", "float32"}:
        return arr.astype(np.float32, copy=False)
    raise ValueError(f"Unsupported depth image encoding: {encoding}")


def backproject_depth_pixels(
    z_v: "np.ndarray",
    xs_v: "np.ndarray",
    ys_v: "np.ndarray",
    d_fx: float,
    d_fy: float,
    d_cx: float,
    d_cy: float,
    c2w: "np.ndarray",
    rgb: "np.ndarray",
    depth_hw: tuple,
) -> tuple:
    """
    Project a set of valid depth pixels into world-space points and sample RGB.

    Parameters
    ----------
    z_v, xs_v, ys_v : 1-D float32 arrays of valid pixel depth, column, row.
    d_fx, d_fy, d_cx, d_cy : depth camera intrinsics.
    c2w : (4, 4) float32 camera-to-world transform.
    rgb : (H_rgb, W_rgb, 3) float32 [0, 1] color image.
    depth_hw : (H_depth, W_depth) — used to map depth pixel coords to rgb coords.

    Returns
    -------
    pts_world : (N, 3) float32 world-space points.
    cols      : (N, 3) float32 RGB colors sampled from rgb.
    """
    import numpy as np

    x_c = (xs_v - d_cx) / d_fx * z_v
    y_c = (ys_v - d_cy) / d_fy * z_v
    pts_cam = np.stack([x_c, y_c, z_v], axis=1)
    pts_world = (c2w[:3, :3] @ pts_cam.T).T + c2w[:3, 3]

    h, w = depth_hw
    rgb_h, rgb_w = rgb.shape[:2]
    rx = np.clip((xs_v / max(w - 1, 1) * (rgb_w - 1)).round().astype(np.int32), 0, rgb_w - 1)
    ry = np.clip((ys_v / max(h - 1, 1) * (rgb_h - 1)).round().astype(np.int32), 0, rgb_h - 1)
    cols = rgb[ry, rx].astype(np.float32)
    return pts_world.astype(np.float32), cols


def backproject_depth_pixels_torch(
    z_v: "torch.Tensor",
    xs_v: "torch.Tensor",
    ys_v: "torch.Tensor",
    d_fx: float,
    d_fy: float,
    d_cx: float,
    d_cy: float,
    c2w: "torch.Tensor",
    rgb: "torch.Tensor",
    depth_hw: tuple,
) -> tuple:
    """Torch (GPU-resident) mirror of :func:`backproject_depth_pixels`.

    Identical algebra to the numpy path so the two agree to float32 matmul
    reassociation tolerance:
        x = (xs - cx) / fx * z
        y = (ys - cy) / fy * z
        pts_cam = stack([x, y, z])
        pts_world = (c2w[:3,:3] @ pts_cam.T).T + c2w[:3,3]
    and the same sensor->rgb nearest mapping for color sampling.

    Parameters
    ----------
    z_v, xs_v, ys_v : 1-D float tensors of valid pixel depth, column, row.
    d_fx, d_fy, d_cx, d_cy : depth camera intrinsics.
    c2w : (4, 4) float tensor, camera-to-world transform (same device as inputs).
    rgb : (H_rgb, W_rgb, 3) float [0, 1] color image tensor.
    depth_hw : (H_depth, W_depth) — maps depth pixel coords to rgb coords.

    Returns
    -------
    pts_world : (N, 3) float32 tensor world-space points.
    cols      : (N, 3) float32 tensor RGB colors sampled from rgb.
    """
    import torch

    device = z_v.device
    z_v = z_v.to(torch.float32)
    xs_v = xs_v.to(torch.float32)
    ys_v = ys_v.to(torch.float32)
    c2w = c2w.to(device=device, dtype=torch.float32)

    x_c = (xs_v - float(d_cx)) / float(d_fx) * z_v
    y_c = (ys_v - float(d_cy)) / float(d_fy) * z_v
    pts_cam = torch.stack([x_c, y_c, z_v], dim=1)               # [N, 3]
    pts_world = (c2w[:3, :3] @ pts_cam.T).T + c2w[:3, 3]         # [N, 3]

    h, w = depth_hw
    rgb_h, rgb_w = rgb.shape[:2]
    rx = torch.clamp(
        torch.round(xs_v / max(int(w) - 1, 1) * (rgb_w - 1)).to(torch.int64), 0, rgb_w - 1
    )
    ry = torch.clamp(
        torch.round(ys_v / max(int(h) - 1, 1) * (rgb_h - 1)).to(torch.int64), 0, rgb_h - 1
    )
    cols = rgb[ry, rx].to(torch.float32)
    return pts_world.to(torch.float32), cols


def depth_to_meters(depth, depth_scale):
    arr = np.asarray(depth)
    if arr.dtype == np.float32 or arr.dtype == np.float64:
        return arr.astype(np.float32)
    return arr.astype(np.float32) / float(depth_scale)


def depth_to_uint16_png(depth, depth_scale, *, uint16_source_scale=None):
    """Metric/quantized depth -> uint16 counts at ``depth_scale`` (issue #642).

    ``depth_scale`` names the OUTPUT PNG's counts-per-metre scale.

    Float32/float64 ``depth`` is treated as exact metres and quantized to
    uint16 counts at ``depth_scale``: NaN/inf/negative clip to 0, values above
    the uint16 range clip to 65535. This branch is unconditionally lossy
    (that is the whole point of a uint16-mm-style export) -- never silent
    about *which* conversion happened, only about the (expected,
    already-documented) precision loss itself.

    uint16 ``depth`` is ALREADY-quantized counts, potentially in a DIFFERENT
    scale than ``depth_scale``. Silently returning it unconverted whenever the
    two scales happen to differ is exactly the latent bug issue #642 closes:
    it is indistinguishable, downstream, from data that really is already at
    ``depth_scale`` -- e.g. raw TUM-scale-5000 counts declared at
    depth_scale=1000 would read back 5x too near, the same defect class that
    forced issue #358's public retraction (5x-shrunk fr3 microcell lattice).
    So a uint16 input REQUIRES the caller to say explicitly, via
    ``uint16_source_scale``, what scale the input counts are already in:

      - ``uint16_source_scale == depth_scale``: identity pass-through,
        bit-exact, NOT lossy (the input already IS the target
        representation).
      - ``uint16_source_scale != depth_scale``: an explicit rescale through
        metres (lossy: two independent quantizations).
      - ``uint16_source_scale is None`` (the default): RAISES ValueError.
        There is no safe default for an already-quantized array; guessing it
        matches ``depth_scale`` is exactly the bug this closes.
    """
    arr = np.asarray(depth)
    if arr.dtype == np.uint16:
        if uint16_source_scale is None:
            raise ValueError(
                "depth_to_uint16_png: refusing to convert an already-uint16 depth "
                "array without an explicit uint16_source_scale (issue #642). "
                f"Guessing it already matches the target depth_scale ({depth_scale!r}) "
                "is exactly the latent-bug class that silently shrank issue #358's fr3 "
                "microcell lattice 5x. Pass uint16_source_scale=<the counts-per-metre "
                "scale this uint16 array is already in> -- pass the same value as "
                "depth_scale for a true identity pass-through."
            )
        uint16_source_scale = float(uint16_source_scale)
        if uint16_source_scale == float(depth_scale):
            return arr
        meters = arr.astype(np.float32) / uint16_source_scale
        scaled = np.nan_to_num(meters, nan=0.0, posinf=0.0, neginf=0.0) * float(depth_scale)
        return np.clip(np.rint(scaled), 0, np.iinfo(np.uint16).max).astype(np.uint16)
    meters = depth_to_meters(arr, depth_scale=1.0)
    scaled = np.nan_to_num(meters, nan=0.0, posinf=0.0, neginf=0.0) * float(depth_scale)
    return np.clip(np.rint(scaled), 0, np.iinfo(np.uint16).max).astype(np.uint16)


def c2w_to_camera_rt(c2w):
    if c2w is None:
        # Identity pose for frames awaiting a SLAM pose (online-SLAM bootstrap).
        # Only used during initialisation; all training frames must have a valid
        # pose before they enter the mapping pipeline.
        return np.eye(3, dtype=np.float32).T, np.zeros(3, dtype=np.float32)
    c2w = np.asarray(c2w, dtype=np.float32)
    if c2w.shape != (4, 4):
        raise ValueError("Expected c2w to have shape 4x4")
    if not np.isfinite(c2w).all():
        raise ValueError("Expected c2w to contain only finite values")
    w2c = np.linalg.inv(c2w)
    return w2c[:3, :3].T, w2c[:3, 3]


def reproject_depth_to_color(
    depth,
    depth_intrinsics,
    color_intrinsics,
    depth_to_color,
    min_depth=0.1,
    max_depth=8.0,
):
    """Align a depth image into the color optical frame with nearest-pixel z-buffering."""
    z = depth_to_meters(depth, depth_intrinsics.depth_scale)
    h, w = z.shape
    ys, xs = np.mgrid[0:h, 0:w]

    valid = np.isfinite(z) & (z > float(min_depth)) & (z < float(max_depth))
    if valid.sum() == 0:
        return np.zeros((color_intrinsics.height, color_intrinsics.width), dtype=np.float32)

    z_d = z[valid]
    x_d = (xs[valid].astype(np.float32) - depth_intrinsics.cx) / depth_intrinsics.fx * z_d
    y_d = (ys[valid].astype(np.float32) - depth_intrinsics.cy) / depth_intrinsics.fy * z_d
    pts_d = np.stack([x_d, y_d, z_d], axis=1)

    transform = np.asarray(depth_to_color, dtype=np.float32)
    if transform.shape != (4, 4):
        raise ValueError("Expected depth_to_color to have shape 4x4")
    pts_c = (transform[:3, :3] @ pts_d.T).T + transform[:3, 3]
    z_c = pts_c[:, 2]

    valid_c = np.isfinite(z_c) & (z_c > float(min_depth)) & (z_c < float(max_depth))
    if valid_c.sum() == 0:
        return np.zeros((color_intrinsics.height, color_intrinsics.width), dtype=np.float32)

    pts_c = pts_c[valid_c]
    z_c = z_c[valid_c]
    u = np.rint(color_intrinsics.fx * pts_c[:, 0] / z_c + color_intrinsics.cx).astype(np.int32)
    v = np.rint(color_intrinsics.fy * pts_c[:, 1] / z_c + color_intrinsics.cy).astype(np.int32)

    inside = (
        (u >= 0)
        & (u < color_intrinsics.width)
        & (v >= 0)
        & (v < color_intrinsics.height)
    )
    aligned = np.zeros((color_intrinsics.height, color_intrinsics.width), dtype=np.float32)
    if inside.sum() == 0:
        return aligned

    u = u[inside]
    v = v[inside]
    z_c = z_c[inside]
    flat_idx = v * color_intrinsics.width + u
    order = np.lexsort((z_c, flat_idx))
    flat_sorted = flat_idx[order]
    z_sorted = z_c[order]
    unique_first = np.unique(flat_sorted, return_index=True)[1]
    flat = aligned.reshape(-1)
    flat[flat_sorted[unique_first]] = z_sorted[unique_first]
    return aligned


def depth_sanity(depth, intrinsics, min_depth=0.1, max_depth=8.0):
    z = depth_to_meters(depth, intrinsics.depth_scale)
    finite = np.isfinite(z)
    valid = finite & (z > float(min_depth)) & (z < float(max_depth))
    if valid.any():
        vals = z[valid]
        return {
            "valid_ratio": float(valid.mean()),
            "median_m": float(np.median(vals)),
            "min_m": float(vals.min()),
            "max_m": float(vals.max()),
        }
    return {
        "valid_ratio": 0.0,
        "median_m": 0.0,
        "min_m": 0.0,
        "max_m": 0.0,
    }


def pointcloud_sanity(points, records):
    points = np.asarray(points, dtype=np.float32)
    if points.size == 0:
        return {
            "num_points": 0,
            "bbox_min": [0.0, 0.0, 0.0],
            "bbox_max": [0.0, 0.0, 0.0],
            "camera_bbox_min": [0.0, 0.0, 0.0],
            "camera_bbox_max": [0.0, 0.0, 0.0],
            "median_camera_distance": 0.0,
            "behind_camera_ratio": 1.0,
        }

    cams = []
    behind = []
    dists = []
    sample = points
    if points.shape[0] > 200000:
        rng = np.random.default_rng(7)
        sample = points[rng.choice(points.shape[0], size=200000, replace=False)]

    for record in records:
        c2w = np.asarray(record.get("c2w"), dtype=np.float32)
        if c2w.shape != (4, 4) or not np.isfinite(c2w).all():
            continue
        center = c2w[:3, 3]
        cams.append(center)
        w2c = np.linalg.inv(c2w)
        pts_cam = (w2c[:3, :3] @ sample.T).T + w2c[:3, 3]
        behind.append(pts_cam[:, 2] <= 0.0)
        dists.append(np.linalg.norm(sample - center[None, :], axis=1))

    cam_arr = np.asarray(cams, dtype=np.float32) if cams else np.zeros((0, 3), dtype=np.float32)
    behind_ratio = float(np.concatenate(behind).mean()) if behind else 0.0
    median_dist = float(np.median(np.concatenate(dists))) if dists else 0.0
    return {
        "num_points": int(points.shape[0]),
        "bbox_min": points.min(axis=0).astype(float).tolist(),
        "bbox_max": points.max(axis=0).astype(float).tolist(),
        "camera_bbox_min": (cam_arr.min(axis=0).astype(float).tolist() if len(cam_arr) else [0.0, 0.0, 0.0]),
        "camera_bbox_max": (cam_arr.max(axis=0).astype(float).tolist() if len(cam_arr) else [0.0, 0.0, 0.0]),
        "median_camera_distance": median_dist,
        "behind_camera_ratio": behind_ratio,
    }


def rgbd_pointcloud_from_records(
    root,
    records,
    intrinsics,
    depth_stride=4,
    init_frames=300,
    max_points=250000,
    min_depth=0.1,
    max_depth=8.0,
):
    points_all = []
    colors_all = []
    selected = records[:init_frames] if init_frames and init_frames > 0 else records

    for record in selected:
        c2w = np.asarray(record["c2w"], dtype=np.float32)
        depth_path = Path(root) / record["depth"]
        rgb_path = Path(root) / record["rgb"]
        if not depth_path.exists() or not rgb_path.exists():
            continue

        depth = np.array(Image.open(depth_path))
        rgb = np.array(Image.open(rgb_path).convert("RGB")).astype(np.float32) / 255.0
        z = depth_to_meters(depth, intrinsics.depth_scale)
        h, w = z.shape
        ys, xs = np.mgrid[0:h:depth_stride, 0:w:depth_stride]
        z_v = z[ys, xs]
        valid = np.isfinite(z_v) & (z_v > float(min_depth)) & (z_v < float(max_depth))
        if valid.sum() == 0:
            continue

        xs_v = xs[valid].astype(np.float32)
        ys_v = ys[valid].astype(np.float32)
        z_v = z_v[valid].astype(np.float32)
        x = (xs_v - intrinsics.cx) / intrinsics.fx * z_v
        y = (ys_v - intrinsics.cy) / intrinsics.fy * z_v
        pts_cam = np.stack([x, y, z_v], axis=1)
        pts_world = (c2w[:3, :3] @ pts_cam.T).T + c2w[:3, 3]

        rgb_h, rgb_w = rgb.shape[:2]
        rgb_x = np.clip((xs_v / max(w - 1, 1) * (rgb_w - 1)).round().astype(np.int32), 0, rgb_w - 1)
        rgb_y = np.clip((ys_v / max(h - 1, 1) * (rgb_h - 1)).round().astype(np.int32), 0, rgb_h - 1)
        rgb_v = rgb[rgb_y, rgb_x]
        points_all.append(pts_world.astype(np.float32))
        colors_all.append(rgb_v.astype(np.float32))

    if not points_all:
        raise RuntimeError("Could not initialize RGB-D point cloud from sequence frames.")

    points = np.concatenate(points_all, axis=0)
    colors = np.concatenate(colors_all, axis=0)
    if max_points and points.shape[0] > max_points:
        rng = np.random.default_rng(42)
        idx = rng.choice(points.shape[0], size=max_points, replace=False)
        points = points[idx]
        colors = colors[idx]
    normals = np.zeros_like(points, dtype=np.float32)
    return points, colors, normals
