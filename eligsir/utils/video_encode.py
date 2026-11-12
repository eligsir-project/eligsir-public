"""Small CPU-only streaming ffmpeg seam for post-processing videos."""

from __future__ import annotations

import os
import subprocess
from collections.abc import Iterable
from collections import deque
from pathlib import Path
from typing import Optional
from itertools import chain
from threading import Thread

import numpy as np
from PIL import Image

from eligsir.utils.trajectory_video import scale_dims_for_video


def resize_frame_for_video_canvas(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize an RGB frame with PIL, avoiding the project's cv2 loader issue."""

    return np.asarray(Image.fromarray(frame).resize((int(width), int(height)), Image.BILINEAR))


def write_mp4_cpu_stream(
    path: str | Path,
    frames: Iterable[np.ndarray],
    *,
    fps: int = 30,
    min_width: Optional[int] = None,
    max_width: Optional[int] = None,
    threads: int = 1,
) -> Optional[str]:
    """Encode an iterable without materializing all frames in memory."""

    iterator = iter(frames)
    try:
        first = np.asarray(next(iterator), dtype=np.uint8)
    except StopIteration:
        return "no frames written"
    if first.ndim != 3 or first.shape[2] != 3:
        return f"expected HWC RGB frames, got {first.shape}"
    height, width = first.shape[:2]
    out_width, out_height = scale_dims_for_video(width, height, min_width=min_width, max_width=max_width)
    output = str(path)
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    try:
        proc = subprocess.Popen(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-f", "rawvideo", "-pix_fmt", "rgb24",
                "-s", f"{out_width}x{out_height}", "-r", str(int(fps)), "-i", "pipe:0",
                "-c:v", "libx264", "-threads", str(max(1, int(threads))),
                "-preset", "ultrafast", "-crf", "23", "-pix_fmt", "yuv420p",
                "-movflags", "+faststart", output,
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
    except Exception as exc:
        return f"ffmpeg Popen failed: {exc}"

    stderr_tail: deque[bytes] = deque(maxlen=32)

    def drain_stderr() -> None:
        if proc.stderr is None:
            return
        for line in iter(proc.stderr.readline, b""):
            stderr_tail.append(line)

    stderr_thread = Thread(target=drain_stderr, daemon=True)
    stderr_thread.start()
    count = 0
    try:
        for frame in chain((first,), iterator):
            frame = np.asarray(frame, dtype=np.uint8)
            if frame.ndim != 3 or frame.shape[2] != 3:
                raise ValueError(f"expected HWC RGB frame, got {frame.shape}")
            if frame.shape[:2] != (out_height, out_width):
                frame = resize_frame_for_video_canvas(frame, out_width, out_height)
            else:
                frame = frame[:out_height, :out_width]
            proc.stdin.write(frame.tobytes())
            count += 1
        proc.stdin.close()
        proc.wait()
        stderr_thread.join(timeout=1)
    except Exception as exc:
        try:
            if proc.stdin is not None:
                proc.stdin.close()
        except Exception:
            pass
        try:
            proc.kill()
        except Exception:
            pass
        try:
            proc.wait(timeout=10)
        except Exception:
            pass
        stderr_thread.join(timeout=1)
        tail = b"".join(stderr_tail).decode(errors="replace")[-500:]
        return f"ffmpeg pipe error: {exc}; {tail}".rstrip("; ")
    if count == 0:
        return "no frames written"
    if proc.returncode != 0:
        stderr_thread.join(timeout=1)
        tail = b"".join(stderr_tail).decode(errors="replace")[-500:]
        return f"ffmpeg exited with code {proc.returncode}: {tail}"
    return None
