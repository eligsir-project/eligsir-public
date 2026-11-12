"""RGBDSequenceFrameSource -- generic frames.jsonl + intrinsics.json RGB-D sequence source.

Extracted from eligsir.utils/streaming_frames.py (Wave 2 split, see
docs/architecture/streaming_frames_wave2_scope.md).
"""
from __future__ import annotations

import os
from typing import Iterator, List


from eligsir.io.sources.decode_cache import (
    StreamingRGBDFrame,
)


class RGBDSequenceFrameSource:
    """
    Ordered frames from a generic RGB-D sequence (frames.jsonl + intrinsics.json).
    Reuses read_frame_records / read_intrinsics from eligsir.utils.rgbd_frames.
    """

    def __init__(self, path: str, max_frames: int = 0, frame_stride: int = 1,
                 streaming_resolution: int = 1, full_res_indices: "set | None" = None):
        import numpy as np
        from eligsir.utils.rgbd_frames import read_frame_records, read_intrinsics

        intr = read_intrinsics(os.path.join(path, "intrinsics.json"))
        records = read_frame_records(os.path.join(path, "frames.jsonl"))
        if frame_stride > 1:
            records = records[::frame_stride]
        if max_frames > 0:
            records = records[:max_frames]

        # Metadata-only downscale (2026-07-27): scales width/height/fx/fy/cx/cy
        # here, upstream of any pixel load. scene/streaming_scene.py's frame->
        # camera conversion resizes the loaded RGB to (frame.width, frame.height)
        # whenever they mismatch the source image, and
        # geometry/sensor_depth.load_frame_depth_meters resizes depth to
        # (cam.image_height, cam.image_width) the same way -- so scaling these
        # fields is sufficient to downscale both without eagerly decoding pixels.
        # Unlike args.resolution (which scene/streaming_scene.py's StreamingScene
        # unconditionally pins to 1 for streaming runs, see its __init__ comment),
        # this is the actual per-frame resolution this loader honors.
        _R = max(1, int(streaming_resolution))
        _full_res = full_res_indices or set()

        def _scaled_intr(_id: int):
            if _R <= 1 or _id in _full_res:
                return intr.fx, intr.fy, intr.cx, intr.cy, intr.width, intr.height
            new_w = max(1, intr.width // _R)
            new_h = max(1, intr.height // _R)
            sx, sy = new_w / float(intr.width), new_h / float(intr.height)
            new_cx = (intr.cx + 0.5) * sx - 0.5
            new_cy = (intr.cy + 0.5) * sy - 0.5
            return intr.fx * sx, intr.fy * sy, new_cx, new_cy, new_w, new_h

        self._frames: List[StreamingRGBDFrame] = []
        for rec in records:
            rgb_path = os.path.join(path, rec["rgb"])
            if not os.path.exists(rgb_path):
                continue
            c2w = np.asarray(rec["c2w"], dtype=np.float32)
            if c2w.shape != (4, 4) or not np.isfinite(c2w).all():
                continue
            depth_rel = rec.get("depth")
            depth_path = os.path.join(path, depth_rel) if depth_rel else None
            _id = int(rec.get("id", len(self._frames)))
            _fx, _fy, _cx, _cy, _w, _h = _scaled_intr(_id)
            self._frames.append(StreamingRGBDFrame(
                index=_id,
                timestamp=float(rec.get("timestamp", len(self._frames))),
                rgb_path=rgb_path,
                depth_path=depth_path,
                c2w=c2w,
                fx=_fx, fy=_fy, cx=_cx, cy=_cy,
                width=_w, height=_h,
                depth_scale=intr.depth_scale,
            ))
        _res_note = f", streaming_resolution={_R}" if _R > 1 else ""
        _fr_note = f", {len(_full_res)} full-res override(s)" if _full_res else ""
        print(f"[streaming] RGBDSequence: {len(self._frames)} frames loaded from {path}{_res_note}{_fr_note}")

    def __len__(self) -> int:
        return len(self._frames)

    def __iter__(self) -> Iterator[StreamingRGBDFrame]:
        yield from self._frames

    def get_all(self) -> List[StreamingRGBDFrame]:
        return self._frames
