"""Shared frame record + decode cache for streaming RGB-D frame sources.

Extracted from eligsir.utils/streaming_frames.py (Wave 2 split, see
docs/architecture/streaming_frames_wave2_scope.md). Every dataset-specific
FrameSource depends on StreamingRGBDFrame + the decode-cache helpers, so
this module has no dependency on any dataset source and must not import
from eligsir.io.sources.datasets.*.
"""
from __future__ import annotations

import io
import os
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from eligsir.utils.rgbd_frames import CameraIntrinsics
from eligsir.io.sources.rosbag_sync import SyncedRGBD


@dataclass
class StreamingRGBDFrame:
    """Lightweight per-frame record: paths + pose + intrinsics, no pixel data.

    fx/fy/cx/cy + width/height describe the *color* camera used for rendering.
    depth_fx/depth_fy/depth_cx/depth_cy + depth_width/depth_height describe the
    *depth* sensor (may differ, e.g. ScanNet 640×480 vs 1296×968 color).
    When depth_fx is None the color intrinsics are used for backprojection too.

    _rgb_bytes / _depth_bytes / _sens_header are used by ScanNetSensFrameSource
    to store images in-memory instead of on disk (file paths remain empty).
    """
    index: int
    timestamp: float
    rgb_path: str
    depth_path: Optional[str] = None
    c2w: Optional["np.ndarray"] = None  # (4, 4) float32 camera-to-world; None until SLAM pose arrives
    fx: float = 0.0
    fy: float = 0.0
    cx: float = 0.0
    cy: float = 0.0
    width: int = 0
    height: int = 0
    depth_scale: float = 1000.0
    # Optional separate depth-sensor intrinsics (populated by ScanNetFrameSource)
    depth_fx: Optional[float] = None
    depth_fy: Optional[float] = None
    depth_cx: Optional[float] = None
    depth_cy: Optional[float] = None
    depth_width: Optional[int] = None
    depth_height: Optional[int] = None
    # In-memory image data (populated by ScanNetSensFrameSource; None → use paths)
    _rgb_bytes: Optional[bytes] = field(default=None, repr=False, compare=False)
    _depth_bytes: Optional[bytes] = field(default=None, repr=False, compare=False)
    _sens_header: Optional[dict] = field(default=None, repr=False, compare=False)
    # Raw numpy arrays — faster than bytes when the source already has decoded data.
    # load_frame_rgb / load_frame_depth_np use these first if present.
    _rgb_np: Optional[Any] = field(default=None, repr=False, compare=False)
    _depth_np: Optional[Any] = field(default=None, repr=False, compare=False)
    # Optional rosbag sync audit timestamps/deltas (nanoseconds).
    color_ts_ns: Optional[int] = None
    depth_ts_ns: Optional[int] = None
    pose_ts_ns: Optional[int] = None
    rgb_depth_dt_ns: Optional[int] = None
    rgb_pose_dt_ns: Optional[int] = None
    color_bag_ns: Optional[int] = None
    depth_bag_ns: Optional[int] = None
    # Drop-invariant source ordinal (issue #745, 2026-08-15). ``index`` above
    # is a RESIDENT-BUFFER position -- for ROS2LiveStreamingFrameSource it is
    # assigned as ``len(src._frames)`` at the moment a frame is finally
    # accepted into the bounded resident buffer (see
    # eligsir/io/sources/datasets/ros2_live.py), so a message silently
    # dropped upstream (the *inner* ROS2LiveRGBDSource's per-topic queue
    # overflow, see that module's ``frames_dropped_by_inner_queue``) shifts
    # every subsequent frame's ``index`` by one -- reshuffling any
    # every-Nth-by-index selection rule (e.g. the streaming eval holdout)
    # between two runs that happened to drop different frames. Only
    # ROS2LiveStreamingFrameSource populates this field today (timestamp-
    # quantized against the source's nominal frame period -- see that
    # module's ``_pump_loop`` for the exact derivation and its documented
    # caveats); every other frame source leaves it ``None`` and consumers
    # must fall back to ``index`` for those (safe there: none of the other
    # sources' ``index`` values are drop-sensitive in the way described
    # above).
    source_ordinal: Optional[int] = None


def load_frame_rgb(frame: "StreamingRGBDFrame"):
    """Return a PIL RGB image for this frame, from numpy array, bytes, or disk.

    Routes through the active FrameDecodeCache (offline --streaming_prefetch) when
    set: a cache hit avoids re-decoding (the same frame is decoded for camera prep,
    surface accumulation, and depth insertion within one iteration), and a
    background thread can pre-decode upcoming frames. No behavior change when off.
    """
    from PIL import Image
    # Cache key is the SOURCE POSITION (stamped on the frame by _get_source_frame /
    # the prefetch worker), not frame.index — frame.index is only guaranteed unique
    # for the shipped sources, but RGBDSequence trusts an external "id". Position is
    # always unique. Only stamped frames (those that went through _get_source_frame)
    # use the cache; everything else decodes normally.
    c = _DECODE_CACHE
    key = getattr(frame, "_cache_pos", None)
    if c is not None and key is not None:
        v = c.peek_rgb(key)
        if v is not None:
            return v
    if frame._rgb_np is not None:
        img = Image.fromarray(frame._rgb_np)
    elif frame._rgb_bytes is not None:
        img = Image.open(io.BytesIO(frame._rgb_bytes)).convert("RGB")
    else:
        img = Image.open(frame.rgb_path).convert("RGB")
    if c is not None and key is not None:
        c.put_rgb(key, img)
    return img


def frame_has_pose(frame: "StreamingRGBDFrame") -> bool:
    """Return True if the frame has a valid, finite camera-to-world pose."""
    if frame.c2w is None:
        return False
    c2w = np.asarray(frame.c2w, dtype=np.float32)
    return c2w.shape == (4, 4) and np.isfinite(c2w).all()


def load_frame_depth_np(frame: "StreamingRGBDFrame"):
    """Return raw depth numpy array (uint16 or float32), or None on failure."""
    import numpy as np
    c = _DECODE_CACHE
    key = getattr(frame, "_cache_pos", None)
    if c is not None and key is not None:
        v = c.peek_depth(key)
        if v is not None:
            return v
    if frame._depth_np is not None:
        out = frame._depth_np
    elif frame._depth_bytes is not None:
        if frame._sens_header is not None:
            from scene.readers.scannet import _decode_sens_depth
            out = _decode_sens_depth(frame._depth_bytes, frame._sens_header)
        else:
            from PIL import Image
            out = np.array(Image.open(io.BytesIO(frame._depth_bytes)))
    elif frame.depth_path is None or not os.path.exists(frame.depth_path):
        return None
    else:
        from PIL import Image
        out = np.array(Image.open(frame.depth_path))
    if c is not None and key is not None and out is not None:
        c.put_depth(key, out)
    return out


# ---------------------------------------------------------------------------
# Frame decode cache + optional background read-ahead (offline replay).
# Offline --streaming_replay is synchronous (single thread); CPU PIL decode of
# each frame stalls the GPU mapper. This bounded LRU cache (1) removes the
# redundant decode of the same frame within an iteration (camera prep + surface
# accumulation + depth insertion all decode it today), and (2) optionally
# pre-decodes upcoming frames on a daemon thread so decode overlaps GPU work.
# Deterministic: keyed by frame.index; returns identical data in identical order.
# ---------------------------------------------------------------------------
_DECODE_CACHE = None  # set via set_decode_cache()


def set_decode_cache(cache) -> None:
    """Set the global frame decode cache instance.

    Args:
        cache: A FrameDecodeCache instance or None to disable caching.
    """
    global _DECODE_CACHE
    _DECODE_CACHE = cache


def get_decode_cache():
    """Get the current global frame decode cache instance.

    Returns:
        The currently active FrameDecodeCache, or None if caching is disabled.
    """
    return _DECODE_CACHE


class FrameDecodeCache:
    """Bounded LRU cache of decoded RGB (PIL) + depth (np) keyed by frame.index,
    with an optional background read-ahead thread for offline replay sources."""

    def __init__(self, maxsize: int = 8):
        """Initialize the frame decode cache.

        Args:
            maxsize: Maximum number of frames to cache (default 8).
        """
        import threading
        from collections import OrderedDict
        self._rgb = OrderedDict()
        self._depth = OrderedDict()
        self._max = max(2, int(maxsize))
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        # prefetch state
        self._fetch_raw = None
        self._n = 0
        self._ahead = 0
        self._cursor = -1
        self._stop = False
        self._thread = None

    @staticmethod
    def _evict(od, cap):
        while len(od) > cap:
            od.popitem(last=False)

    def peek_rgb(self, idx):
        """Retrieve a cached RGB image without removing it.

        Args:
            idx: Frame index/key.

        Returns:
            The cached PIL RGB image, or None if not in cache.
        """
        with self._lock:
            v = self._rgb.get(idx)
            if v is not None:
                self._rgb.move_to_end(idx)
            return v

    def put_rgb(self, idx, val):
        """Add or update a cached RGB image.

        Args:
            idx: Frame index/key.
            val: The PIL RGB image to cache.
        """
        with self._lock:
            self._rgb[idx] = val
            self._rgb.move_to_end(idx)
            self._evict(self._rgb, self._max)

    def peek_depth(self, idx):
        """Retrieve a cached depth array without removing it.

        Args:
            idx: Frame index/key.

        Returns:
            The cached depth numpy array, or None if not in cache.
        """
        with self._lock:
            v = self._depth.get(idx)
            if v is not None:
                self._depth.move_to_end(idx)
            return v

    def put_depth(self, idx, val):
        """Add or update a cached depth array.

        Args:
            idx: Frame index/key.
            val: The depth numpy array to cache.
        """
        with self._lock:
            self._depth[idx] = val
            self._depth.move_to_end(idx)
            self._evict(self._depth, self._max)

    # -- background read-ahead -------------------------------------------------
    def start_prefetch(self, fetch_raw, n_frames: int, ahead: int, n_workers: int = 1):
        """Start ``n_workers`` background decode threads racing over the
        look-ahead window instead of a single decoder. Real parallelism, not
        just GIL-serialized threads: Pillow's JPEG decode (libjpeg-turbo,
        the codec used by most RGB sources in this project) releases the
        GIL during the actual decode loop, so multiple threads genuinely
        decode concurrently rather than trading off one core. n_workers=1
        (default) preserves the original single-thread behavior exactly."""
        import threading
        self._fetch_raw = fetch_raw
        self._n = int(n_frames)
        self._ahead = max(1, int(ahead))
        # hold the look-ahead window + the in-use frame + a little slack
        self._max = max(self._max, self._ahead + 3)
        self._inflight: set = set()
        self._threads = [
            threading.Thread(target=self._run, name=f"frame-prefetch-{i}", daemon=True)
            for i in range(max(1, int(n_workers)))
        ]
        self._thread = self._threads[0]  # back-compat: .stop()/tests reference this
        for t in self._threads:
            t.start()

    def note_consumed(self, idx: int):
        # Called by the main thread as it consumes each source index; advances the
        # read-ahead cursor (eviction is handled by LRU on put).
        with self._cv:
            if idx > self._cursor:
                self._cursor = idx
                self._cv.notify_all()

    def _next_locked(self):
        # Cache is keyed by source position; a position needs decoding if either its
        # RGB or its depth is absent (the two dicts evict independently), AND no
        # other worker thread has already claimed it (see _inflight -- without this,
        # concurrent workers would race to redundantly decode the same position
        # before either finishes, since it isn't in self._rgb/_depth until done).
        if self._fetch_raw is None or self._cursor < 0:
            return None
        end = min(self._cursor + 1 + self._ahead, self._n)
        for pos in range(self._cursor + 1, end):
            if pos not in self._rgb or pos not in self._depth:
                if pos not in self._inflight:
                    return pos
        return None

    def _run(self):
        while True:
            with self._cv:
                while not self._stop and self._next_locked() is None:
                    self._cv.wait()
                if self._stop:
                    return
                nxt = self._next_locked()
                if nxt is not None:
                    self._inflight.add(nxt)
            if nxt is None:
                continue
            try:
                fr = self._fetch_raw(nxt)
                if fr is not None:
                    # Stamp the source position so the loaders cache under the same
                    # key the main thread will use, then populate via the loaders
                    # (peek miss -> put).
                    fr._cache_pos = nxt
                    load_frame_rgb(fr)
                    load_frame_depth_np(fr)
            except Exception:
                pass  # decode errors surface on the main thread's own load
            finally:
                with self._cv:
                    self._inflight.discard(nxt)
                    self._cv.notify_all()

    def stop(self):
        with self._cv:
            self._stop = True
            self._cv.notify_all()


def streaming_frame_from_synced_rgbd(
    rec: SyncedRGBD,
    index: int,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    width: int,
    height: int,
    c2w_override=None,
    depth_scale: float = 1000.0,
) -> StreamingRGBDFrame:
    """Build a StreamingRGBDFrame while preserving rosbag sync audit fields."""
    c2w = c2w_override if c2w_override is not None else (rec.pose.payload if rec.pose is not None else None)
    return StreamingRGBDFrame(
        index=index,
        timestamp=float(rec.color.header_ns) * 1e-9,
        rgb_path="",
        depth_path=None,
        c2w=c2w,
        fx=fx, fy=fy, cx=cx, cy=cy,
        width=width, height=height,
        depth_scale=depth_scale,
        _rgb_bytes=rec.color.payload,
        _depth_bytes=rec.depth.payload,
        color_ts_ns=rec.color.header_ns,
        depth_ts_ns=rec.depth.header_ns,
        pose_ts_ns=rec.pose.header_ns if rec.pose is not None else None,
        rgb_depth_dt_ns=rec.rgb_depth_dt_ns,
        rgb_pose_dt_ns=rec.rgb_pose_dt_ns,
        color_bag_ns=rec.color.bag_ns,
        depth_bag_ns=rec.depth.bag_ns,
    )


def resize_rgbd_frame(
    rgb: np.ndarray,
    depth: np.ndarray,
    intrinsics: CameraIntrinsics,
    divisor: int,
) -> tuple:
    """Resize an RGB-D frame + intrinsics by an integer divisor.

    RGB uses area-averaging (best for downscale quality). Depth uses
    nearest-neighbor so depth values are NEVER blended across
    discontinuities -- interpolating depth produces false intermediate
    values at object boundaries. divisor=1 is a no-op passthrough.
    """
    _R = max(1, int(divisor))
    if _R <= 1:
        return rgb, depth, intrinsics
    import cv2 as _cv2
    width, height = int(intrinsics.width), int(intrinsics.height)
    new_w = max(1, width // _R)
    new_h = max(1, height // _R)
    out_rgb = _cv2.resize(rgb, (new_w, new_h), interpolation=_cv2.INTER_AREA)
    out_depth = _cv2.resize(depth, (new_w, new_h), interpolation=_cv2.INTER_NEAREST)
    sx, sy = new_w / float(width), new_h / float(height)
    new_cx = (intrinsics.cx + 0.5) * sx - 0.5
    new_cy = (intrinsics.cy + 0.5) * sy - 0.5
    from dataclasses import replace
    out_intrinsics = replace(
        intrinsics,
        width=new_w, height=new_h,
        fx=intrinsics.fx * sx, fy=intrinsics.fy * sy,
        cx=new_cx, cy=new_cy,
    )
    return out_rgb, out_depth, out_intrinsics
