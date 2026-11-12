"""RealityScanFrameSource -- depth-less RGB + pose streaming source for
RealityScan Mobile scenes (RGB + poses + sparse SfM cloud, NO depth data).

See docs/superpowers/sdd/2026-08-12-realityscan-streaming-support/ for the
originating plan. This is the streaming-mode twin of the existing offline
reader `scene/readers/realityscan.py` -- that reader loads all cameras into
memory at once for `train.py`; this source releases them one at a time, in
capture order, for `train_streaming.py`.
"""
from __future__ import annotations

import os
from typing import Iterator, List, Optional, Tuple

import numpy as np

from eligsir.io.sources.decode_cache import StreamingRGBDFrame

# EXIF tag ids (see PIL.ExifTags / TIFF/EXIF spec):
#   36867 = DateTimeOriginal (preferred: when the photo was actually taken)
#   306   = DateTime (fallback: file/metadata modify time)
_EXIF_TAG_DATETIME_ORIGINAL = 36867
_EXIF_TAG_DATETIME = 306


def _read_exif_datetime(image_path) -> Optional[str]:
    """Return the raw EXIF DateTimeOriginal (or DateTime fallback) string for
    an image, or None if neither tag is present / the file can't be read."""
    try:
        from PIL import Image
        with Image.open(image_path) as img:
            exif = img.getexif()
            if not exif:
                return None
            value = exif.get(_EXIF_TAG_DATETIME_ORIGINAL) or exif.get(_EXIF_TAG_DATETIME)
            if value:
                return str(value)
    except Exception:
        return None
    return None


class RealityScanFrameSource:
    """Ordered frames from a RealityScan Mobile project (sfm.txt + *.xmp +
    RGB captures + prior/refined poses; sparse SfM point cloud, NO depth).

    Frames are yielded in capture order, recovered from each image's EXIF
    DateTimeOriginal where available. RealityScan Mobile exports use random
    UUID filenames (not a usable proxy for capture order) and, in practice,
    strip EXIF entirely -- confirmed on real exported photos, both the naive
    and IFD-correct DateTimeOriginal lookups return nothing.

    For cameras without usable EXIF, order falls back to each image file's
    filesystem mtime when the mtimes actually discriminate (i.e. every
    camera in the group has one and they are not all identical) -- real
    RealityScan Mobile exports preserve a per-photo mtime that tracks
    capture order to within a few seconds (verified 2026-08-12 against a
    hallway-walk capture whose `sfm.txt` registration order turned out to
    be uncorrelated with true sequence: Spearman rho of dominant-axis
    position vs. `sfm.txt` order index was 0.03, essentially zero, over
    201 frames -- see docs/architecture/2026-08-12-realityscan-trajectory-
    distance-bug.md). Only when mtimes are unusable (missing, or all equal
    -- e.g. a bulk-copied/re-exported folder where every file shares one
    mtime) does order fall further back to `sfm.txt`'s own camera sequence
    (as loaded by `realityscan_importer.load_project`, which does not
    re-sort it) -- sfm.txt's registration order reflects RealityScan's
    internal reconstruction/bundle-adjustment order, not capture time, and
    is NOT a reliable proxy for capture-adjacency (this was the prior
    fallback and, on the real capture above, was empirically no better
    than a random permutation). Never falls back to filename/UUID sort,
    which carries no ordering information at all for these exports.
    """

    def __init__(
        self,
        path: str,
        max_frames: int = 0,
        frame_stride: int = 1,
        streaming_resolution: int = 1,
        auto_downscale_max_pixels: int = 0,
    ):
        try:
            from realityscan_importer import load_project
        except ImportError as exc:
            raise ImportError(
                "RealityScanFrameSource requires the 'realityscan_importer' package, "
                "which is not installed. Install the separately licensed "
                "realityscan-importer package and make it available on "
                "PYTHONPATH."
            ) from exc

        rs_scene = load_project(path, undistort=True)
        cameras = list(rs_scene.cameras)

        with_exif = []
        without_exif = []
        for cam in cameras:
            dt = _read_exif_datetime(cam.image_path)
            if dt is not None:
                with_exif.append((dt, cam))
            else:
                without_exif.append(cam)

        with_exif.sort(key=lambda pair: pair[0])

        # For cameras with no EXIF timestamp, try file mtime next -- gated so a
        # missing/degenerate mtime signal (e.g. every file sharing one mtime
        # from a bulk copy/re-export) can never silently produce a fake order.
        without_exif_mtimes = []
        for cam in without_exif:
            try:
                without_exif_mtimes.append(os.path.getmtime(cam.image_path))
            except OSError:
                without_exif_mtimes.append(None)
        mtimes_usable = (
            len(without_exif) > 1
            and all(m is not None for m in without_exif_mtimes)
            and (max(without_exif_mtimes) - min(without_exif_mtimes)) > 0
        )

        # sfm.txt's own registration order (as returned by load_project,
        # unmodified) -- reconstruction order, not chronological; last resort.
        sfm_order_index = {id(cam): i for i, cam in enumerate(cameras)}

        if mtimes_usable:
            without_exif = [
                cam for _mt, cam in sorted(zip(without_exif_mtimes, without_exif), key=lambda pair: pair[0])
            ]
            fallback_label = "file mtime"
        else:
            without_exif.sort(key=lambda cam: sfm_order_index[id(cam)])
            fallback_label = "sfm.txt registration order"

        ordered_cameras = [cam for _dt, cam in with_exif] + without_exif
        print(
            f"[streaming] RealityScan: {len(with_exif)} frame(s) ordered by EXIF "
            f"DateTimeOriginal, {len(without_exif)} by {fallback_label} "
            f"(of {len(cameras)} total)"
        )

        if frame_stride > 1:
            ordered_cameras = ordered_cameras[::frame_stride]
        if max_frames > 0:
            ordered_cameras = ordered_cameras[:max_frames]

        # No intrinsics.json exists for RealityScan sources (sfm.txt + *.xmp
        # instead), so train_streaming.py's --streaming_auto_downscale_max_pixels
        # pre-flight gate (which only checks that file) silently never fires here
        # -- confirmed to cause a real GPU hang (12MP native photos rendered
        # unbounded, ~6x the intended budget). Self-contained equivalent: compute
        # the same divisor directly from the first camera's real native size.
        divisor = max(1, int(streaming_resolution))
        if divisor == 1 and auto_downscale_max_pixels > 0 and ordered_cameras:
            native_w, native_h = int(ordered_cameras[0].width), int(ordered_cameras[0].height)
            if native_w > 0 and native_h > 0 and native_w * native_h > auto_downscale_max_pixels:
                while (native_w // (divisor + 1)) * (native_h // (divisor + 1)) > auto_downscale_max_pixels:
                    divisor += 1
                divisor += 1
                print(
                    f"[streaming] RealityScan auto-downscale gate: native {native_w}x{native_h} "
                    f"({native_w * native_h} px) exceeds --streaming_auto_downscale_max_pixels="
                    f"{auto_downscale_max_pixels}; forcing divisor={divisor} "
                    f"({native_w // divisor}x{native_h // divisor}).",
                    flush=True,
                )

        # Issue #595: read-only provenance for report.json["resolution"]. This
        # source applies its divisor itself and never touches
        # args.streaming_resolution, and it has no intrinsics.json for
        # train_streaming.py's gate to read -- so without these two attributes
        # a RealityScan run's report could only say "native unknown". Recorded,
        # not consulted: nothing above reads them back.
        self.native_resolution: Optional[Tuple[int, int]] = (
            (int(ordered_cameras[0].width), int(ordered_cameras[0].height))
            if ordered_cameras else None
        )
        self.applied_divisor: int = int(divisor)

        self._frames: List[StreamingRGBDFrame] = []
        for i, cam in enumerate(ordered_cameras):
            c2w = np.asarray(cam.world_from_camera, dtype=np.float32)
            width, height = int(cam.width), int(cam.height)
            fx, fy, cx, cy = float(cam.fx), float(cam.fy), float(cam.cx), float(cam.cy)
            if divisor > 1:
                new_w, new_h = max(1, width // divisor), max(1, height // divisor)
                sx, sy = new_w / float(width), new_h / float(height)
                fx, fy = fx * sx, fy * sy
                cx, cy = (cx + 0.5) * sx - 0.5, (cy + 0.5) * sy - 0.5
                width, height = new_w, new_h
            self._frames.append(StreamingRGBDFrame(
                index=i,
                timestamp=float(i),  # synthetic: deterministic replay releases by
                                      # iteration count, not wall-clock (see brief).
                rgb_path=str(cam.image_path),
                depth_path=None,
                c2w=c2w,
                fx=fx, fy=fy, cx=cx, cy=cy,
                width=width, height=height,
            ))

        pc = rs_scene.point_cloud
        if pc is not None:
            self.seed_points: Optional[np.ndarray] = np.asarray(pc.xyz, dtype=np.float32)
            if pc.rgb is not None:
                rgb = np.asarray(pc.rgb, dtype=np.float32)
                if rgb.size and rgb.max() > 1.0:
                    rgb = rgb / 255.0
                self.seed_colors: Optional[np.ndarray] = rgb.astype(np.float32)
            else:
                self.seed_colors = None
        else:
            self.seed_points = None
            self.seed_colors = None

        print(f"[streaming] RealityScan: {len(self._frames)} frames loaded from {path}")

    def __len__(self) -> int:
        return len(self._frames)

    def __iter__(self) -> Iterator[StreamingRGBDFrame]:
        yield from self._frames

    def get_all(self) -> List[StreamingRGBDFrame]:
        return self._frames
