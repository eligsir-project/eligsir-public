"""Debug-visualization helpers for RGB-D frame sources.

Extracted from eligsir.utils/streaming_frames.py (Wave 2 split, see
docs/architecture/streaming_frames_wave2_scope.md). Confirmed live (not
dead code) before moving: _rot_to_quat/_quat_to_rot/_slerp are used by
OrbbecRosBagFrameSource's pose interpolation, and
_write_depth_color_alignment_debug (plus its helpers) is exercised
directly by tests/test_realsense_alignment.py and by
OrbbecRosBagFrameSource's realsense-alignment debug path.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any


def _rot_to_quat(R):
    import numpy as np
    tr = float(np.trace(R))
    if tr > 0.0:
        s = np.sqrt(tr + 1.0) * 2.0
        return np.array([
            0.25 * s,
            (R[2, 1] - R[1, 2]) / s,
            (R[0, 2] - R[2, 0]) / s,
            (R[1, 0] - R[0, 1]) / s,
        ], dtype=np.float64)
    i = int(np.argmax(np.diag(R)))
    if i == 0:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        q = [((R[2, 1] - R[1, 2]) / s), 0.25 * s, ((R[0, 1] + R[1, 0]) / s), ((R[0, 2] + R[2, 0]) / s)]
    elif i == 1:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        q = [((R[0, 2] - R[2, 0]) / s), ((R[0, 1] + R[1, 0]) / s), 0.25 * s, ((R[1, 2] + R[2, 1]) / s)]
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        q = [((R[1, 0] - R[0, 1]) / s), ((R[0, 2] + R[2, 0]) / s), ((R[1, 2] + R[2, 1]) / s), 0.25 * s]
    q = np.asarray(q, dtype=np.float64)
    return q / max(np.linalg.norm(q), 1e-12)

def _quat_to_rot(q):
    import numpy as np
    q = q / max(np.linalg.norm(q), 1e-12)
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)

def _slerp(q0, q1, t):
    import numpy as np
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        out = q0 + t * (q1 - q0)
        return out / max(np.linalg.norm(out), 1e-12)
    theta_0 = np.arccos(np.clip(dot, -1.0, 1.0))
    theta = theta_0 * t
    return (
        np.sin(theta_0 - theta) * q0 + np.sin(theta) * q1
    ) / np.sin(theta_0)


def _depth_to_vis_rgb(depth_meters):
    import numpy as np

    z = np.asarray(depth_meters, dtype=np.float32)
    valid = np.isfinite(z) & (z > 0)
    out = np.zeros((*z.shape[:2], 3), dtype=np.uint8)
    if not valid.any():
        return out
    vals = z[valid]
    lo = float(np.percentile(vals, 2))
    hi = float(np.percentile(vals, 98))
    if hi <= lo:
        hi = lo + 1e-3
    n = np.clip((z - lo) / (hi - lo), 0.0, 1.0)
    out[..., 0] = np.where(valid, np.clip(255.0 * n, 0, 255), 0).astype(np.uint8)
    out[..., 1] = np.where(valid, np.clip(255.0 * (1.0 - np.abs(n - 0.5) * 2.0), 0, 255), 0).astype(np.uint8)
    out[..., 2] = np.where(valid, np.clip(255.0 * (1.0 - n), 0, 255), 0).astype(np.uint8)
    return out


def _depth_edge_mask(depth_meters):
    import numpy as np

    z = np.asarray(depth_meters, dtype=np.float32)
    valid = np.isfinite(z) & (z > 0)
    if not valid.any():
        return np.zeros(z.shape[:2], dtype=bool)
    dzx = np.zeros_like(z, dtype=np.float32)
    dzy = np.zeros_like(z, dtype=np.float32)
    pair_x = valid[:, 1:] & valid[:, :-1]
    pair_y = valid[1:, :] & valid[:-1, :]
    dzx[:, 1:][pair_x] = np.abs(z[:, 1:][pair_x] - z[:, :-1][pair_x])
    dzy[1:, :][pair_y] = np.abs(z[1:, :][pair_y] - z[:-1, :][pair_y])
    vals = z[valid]
    thresh = max(0.025, float(np.percentile(vals, 95) - np.percentile(vals, 5)) * 0.015)
    boundary = np.zeros_like(valid, dtype=bool)
    boundary[:, 1:] |= valid[:, 1:] & ~valid[:, :-1]
    boundary[:, :-1] |= valid[:, :-1] & ~valid[:, 1:]
    boundary[1:, :] |= valid[1:, :] & ~valid[:-1, :]
    boundary[:-1, :] |= valid[:-1, :] & ~valid[1:, :]
    return (valid & ((dzx > thresh) | (dzy > thresh))) | boundary


def _resize_mask_nearest(mask, size_wh):
    from PIL import Image as _Img
    import numpy as np

    im = _Img.fromarray(np.asarray(mask, dtype=np.uint8) * 255, mode="L")
    im = im.resize(size_wh, _Img.Resampling.NEAREST)
    return np.asarray(im) > 0


def _overlay_depth_edges(rgb, depth_meters, edge_color=(255, 32, 32), resize_depth: bool = False):
    import numpy as np

    out = np.asarray(rgb, dtype=np.uint8).copy()
    mask = _depth_edge_mask(depth_meters)
    if resize_depth and mask.shape[:2] != out.shape[:2]:
        mask = _resize_mask_nearest(mask, (out.shape[1], out.shape[0]))
    elif mask.shape[:2] != out.shape[:2]:
        return out
    out[mask] = np.asarray(edge_color, dtype=np.uint8)
    return out


def _add_label_bar(image, label: str):
    from PIL import Image as _Img, ImageDraw

    im = _Img.fromarray(image)
    bar_h = 24
    out = _Img.new("RGB", (im.width, im.height + bar_h), (20, 20, 20))
    out.paste(im, (0, bar_h))
    draw = ImageDraw.Draw(out)
    draw.text((8, 5), label, fill=(235, 235, 235))
    return out


def _write_depth_color_alignment_debug(
    debug_dir: str,
    frame_index: int,
    rgb,
    raw_depth_m,
    saved_depth_m,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    import json
    import numpy as np
    from PIL import Image as _Img

    root = Path(debug_dir)
    root.mkdir(parents=True, exist_ok=True)
    stem = f"frame_{frame_index:06d}"
    rgb = np.asarray(rgb, dtype=np.uint8)
    raw_vis = _depth_to_vis_rgb(raw_depth_m)
    saved_vis = _depth_to_vis_rgb(saved_depth_m)
    if raw_vis.shape[:2] != rgb.shape[:2]:
        raw_vis_contact = np.asarray(_Img.fromarray(raw_vis).resize((rgb.shape[1], rgb.shape[0]), _Img.Resampling.NEAREST))
    else:
        raw_vis_contact = raw_vis
    if saved_vis.shape[:2] != rgb.shape[:2]:
        saved_vis_contact = np.asarray(_Img.fromarray(saved_vis).resize((rgb.shape[1], rgb.shape[0]), _Img.Resampling.NEAREST))
    else:
        saved_vis_contact = saved_vis

    rgb_path = root / f"{stem}_color.png"
    raw_path = root / f"{stem}_raw_depth_vis.png"
    aligned_path = root / f"{stem}_aligned_depth_vis.png"
    raw_overlay_path = root / f"{stem}_raw_depth_edges_on_color.png"
    aligned_overlay_path = root / f"{stem}_aligned_depth_edges_on_color.png"
    contact_path = root / f"{stem}_depth_color_contact.png"

    raw_overlay = _overlay_depth_edges(rgb, raw_depth_m, edge_color=(255, 160, 0), resize_depth=True)
    aligned_overlay = _overlay_depth_edges(rgb, saved_depth_m, edge_color=(255, 32, 32), resize_depth=True)
    _Img.fromarray(rgb, mode="RGB").save(rgb_path)
    _Img.fromarray(raw_vis, mode="RGB").save(raw_path)
    _Img.fromarray(saved_vis, mode="RGB").save(aligned_path)
    _Img.fromarray(raw_overlay, mode="RGB").save(raw_overlay_path)
    _Img.fromarray(aligned_overlay, mode="RGB").save(aligned_overlay_path)

    panels = [
        _add_label_bar(rgb, "color"),
        _add_label_bar(raw_vis_contact, "raw depth (resized for view)"),
        _add_label_bar(raw_overlay, "raw depth edges on color"),
        _add_label_bar(aligned_overlay, "aligned depth edges on color"),
    ]
    contact = _Img.new("RGB", (rgb.shape[1] * 2, panels[0].height * 2), (0, 0, 0))
    contact.paste(panels[0], (0, 0))
    contact.paste(panels[1], (rgb.shape[1], 0))
    contact.paste(panels[2], (0, panels[0].height))
    contact.paste(panels[3], (rgb.shape[1], panels[0].height))
    contact.save(contact_path)

    record = dict(metadata)
    record.update({
        "frame_index": int(frame_index),
        "raw_depth_shape": [int(raw_depth_m.shape[0]), int(raw_depth_m.shape[1])],
        "saved_depth_shape": [int(saved_depth_m.shape[0]), int(saved_depth_m.shape[1])],
        "raw_depth_valid": int((np.asarray(raw_depth_m) > 0).sum()),
        "saved_depth_valid": int((np.asarray(saved_depth_m) > 0).sum()),
        "files": {
            "color": str(rgb_path),
            "raw_depth_vis": str(raw_path),
            "aligned_depth_vis": str(aligned_path),
            "raw_depth_edges_on_color": str(raw_overlay_path),
            "aligned_depth_edges_on_color": str(aligned_overlay_path),
            "contact_sheet": str(contact_path),
        },
    })
    meta_path = root / f"{stem}_metadata.json"
    meta_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    record["metadata_file"] = str(meta_path)
    return record
