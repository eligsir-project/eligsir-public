"""Deterministic synthetic RGB-D frame source + fake SLAM sidecar for benchmarking.

A drop-in ``RGBDFrameSource`` (see ``utils/rgbd_frames.py``) that emits fully
synthetic, seeded RGB-D frames with no camera/bag/ORB-SLAM3 dependency, so the
live-ingestion pipeline (queueing, admission, GPU-timed spans -- see
``eligsir/runtime/profiling/``) can be benchmarked in isolation and
reproducibly. Pairs with :class:`FakeSlamSidecar` to inject artificial SLAM
latency/failure, and with :class:`RejectionPolicy` to model a configurable
admission acceptance rate (e.g. "30 FPS in, ~2 accepted keyframes/s").
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field
from typing import Iterator, Optional

import numpy as np

from eligsir.utils.rgbd_frames import CameraIntrinsics, RGBDFrame, RGBDFrameSource


def frozen_artifact_hashes(*, seed: int, num_frames: int) -> dict[str, str]:
    """Hash the exact generated source and its exact pose stream.

    Scenario admission, delay, fault, and burst policy deliberately do not
    participate: every benchmark arm consumes this one frozen source.
    """
    from eligsir.experiments.hashing import content_hash

    source = SyntheticRGBDFrameSource(SyntheticSourceConfig(
        seed=seed, num_frames=num_frames, realtime=False,
    ))
    source_frames = []
    poses = []
    for frame in source:
        intrinsics = frame.intrinsics
        source_frames.append({
            "index": frame.index,
            "timestamp": frame.timestamp,
            "intrinsics": {
                "width": intrinsics.width, "height": intrinsics.height,
                "fx": intrinsics.fx, "fy": intrinsics.fy,
                "cx": intrinsics.cx, "cy": intrinsics.cy,
                "depth_scale": intrinsics.depth_scale,
            },
            "rgb": frame.rgb,
            "depth": frame.depth,
            "c2w": frame.c2w,
        })
        poses.append({"index": frame.index, "timestamp": frame.timestamp, "c2w": frame.c2w})
    return {
        "source_artifact_hash": content_hash(source_frames),
        "pose_hash": content_hash(poses),
    }

#: Default intrinsics for synthetic frames -- arbitrary but fixed, so hashes
#: of emitted content are stable across runs/machines.
DEFAULT_WIDTH = 64
DEFAULT_HEIGHT = 48
DEFAULT_FPS = 30.0


def _make_checkerboard(width: int, height: int, seed: int, tile: int = 8) -> np.ndarray:
    """Deterministic checkerboard RGB pattern, tinted by ``seed`` so distinct
    frame indices produce visually (and byte-) distinct content while staying
    fully reproducible for a fixed seed."""
    yy, xx = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    parity = ((xx // tile) + (yy // tile)) % 2
    rng = np.random.default_rng(seed)
    color_a = rng.integers(0, 256, size=3, dtype=np.uint8)
    color_b = rng.integers(0, 256, size=3, dtype=np.uint8)
    img = np.where(parity[..., None].astype(bool), color_a, color_b)
    return img.astype(np.uint8)


def _make_depth_ramp(width: int, height: int, seed: int, near: float = 0.5, far: float = 4.0) -> np.ndarray:
    """Deterministic depth ramp (meters), varying smoothly across columns and
    offset per-seed so successive synthetic frames aren't bit-identical."""
    rng = np.random.default_rng(seed + 1_000_000)
    offset = float(rng.uniform(-0.1, 0.1))
    ramp = np.linspace(near, far, width, dtype=np.float32)
    depth = np.tile(ramp, (height, 1)) + offset
    return np.clip(depth, near, far).astype(np.float32)


@dataclass
class SyntheticSourceConfig:
    """Configuration for :class:`SyntheticRGBDFrameSource`.

    Args:
        seed: Fixed seed -- same seed always produces byte-identical frame
            content and identical inter-arrival timing.
        fps: Steady-state emission rate (frames/sec) outside of bursts.
        width, height: Synthetic frame resolution.
        num_frames: If set, emit exactly this many frames then stop
            (finite mode). ``None`` means emit indefinitely (long-running
            mode) until the caller stops iterating.
        burst_size: If > 0, emit this many frames back-to-back with no
            pacing delay at the start of the stream (and after every
            ``burst_period_frames`` steady-state frames, if set), then
            resume steady-state pacing.
        burst_period_frames: Steady-state frame count between bursts.
            ``None`` (default) means only one burst, at the very start.
        realtime: If True, actually sleep between frames to match ``fps``
            (used for real wall-clock benchmarking). If False (default),
            timestamps advance deterministically but no real sleep happens
            (used for fast, deterministic unit tests).
    """

    seed: int = 0
    fps: float = DEFAULT_FPS
    width: int = DEFAULT_WIDTH
    height: int = DEFAULT_HEIGHT
    num_frames: Optional[int] = None
    burst_size: int = 0
    burst_period_frames: Optional[int] = None
    realtime: bool = True


class SyntheticRGBDFrameSource(RGBDFrameSource):
    """Deterministic, seeded synthetic RGB-D source implementing the
    ``RGBDFrameSource`` protocol (``__iter__`` yielding ``RGBDFrame``).

    Frame content (checkerboard RGB + depth ramp) and inter-arrival timing
    are both fully determined by ``config.seed`` -- two sources built with
    the same config produce byte-identical frame sequences (verified in
    ``tests/test_synthetic_rgbd_source.py``).
    """

    def __init__(self, config: Optional[SyntheticSourceConfig] = None):
        self.config = config or SyntheticSourceConfig()
        self.intrinsics = CameraIntrinsics(
            width=self.config.width,
            height=self.config.height,
            fx=float(self.config.width),
            fy=float(self.config.width),
            cx=self.config.width / 2.0,
            cy=self.config.height / 2.0,
            depth_scale=1000.0,
            camera_frame="synthetic_camera",
            world_frame="synthetic_world",
        )
        #: Wall-clock start time of the first emitted frame, set on first
        #: iteration so repeated ``__iter__()`` calls each restart cleanly.
        self._t0: Optional[float] = None

    def __len__(self) -> int:
        if self.config.num_frames is None:
            raise TypeError("SyntheticRGBDFrameSource has no fixed length in long-running mode")
        return self.config.num_frames

    def _frame_indices(self) -> Iterator[int]:
        if self.config.num_frames is not None:
            return iter(range(self.config.num_frames))
        return itertools.count()

    def _is_burst_index(self, idx: int) -> bool:
        cfg = self.config
        if cfg.burst_size <= 0:
            return False
        if idx < cfg.burst_size:
            return True
        if cfg.burst_period_frames is None:
            return False
        # After the initial burst, a further burst fires every
        # (burst_period_frames + burst_size) frames.
        cycle = cfg.burst_period_frames + cfg.burst_size
        pos_in_cycle = (idx - cfg.burst_size) % cycle
        return pos_in_cycle >= cfg.burst_period_frames

    def get_frame(self, index: int) -> RGBDFrame:
        """Build the (deterministic) frame at ``index`` without any pacing
        delay -- used by tests and by callers that want random access rather
        than the paced ``__iter__`` stream."""
        cfg = self.config
        rgb = _make_checkerboard(cfg.width, cfg.height, seed=cfg.seed + index)
        depth = _make_depth_ramp(cfg.width, cfg.height, seed=cfg.seed + index)
        timestamp = index / cfg.fps
        c2w = np.eye(4, dtype=np.float32)
        c2w[0, 3] = 0.1 * index  # deterministic synthetic translation drift
        return RGBDFrame(
            index=index,
            timestamp=timestamp,
            rgb=rgb,
            depth=depth,
            intrinsics=self.intrinsics,
            c2w=c2w,
        )

    def lazy_frames(self) -> Iterator[RGBDFrame]:
        """Alias for ``__iter__`` matching the ``lazy_frames``-style accessor
        used by some existing frame sources in this codebase."""
        return self.__iter__()

    def __iter__(self) -> Iterator[RGBDFrame]:
        cfg = self.config
        self._t0 = time.monotonic()
        period = 1.0 / cfg.fps if cfg.fps > 0 else 0.0
        # ``next_time`` tracks the next scheduled steady-state emission time,
        # anchored to wall-clock "now" rather than a fixed idx*period offset
        # from t0 -- this way a burst (which skips pacing) doesn't leave a
        # large backlog that the pacer then tries to "catch up" on by
        # sleeping a long stretch immediately after the burst ends.
        next_time = self._t0
        first = True
        prev_was_burst = False
        for idx in self._frame_indices():
            is_burst = self._is_burst_index(idx)
            if cfg.realtime and not is_burst:
                if first or prev_was_burst:
                    # Just left burst mode (or this is the very first
                    # steady-state frame): anchor pacing fresh from "now"
                    # rather than the stale next_time left over from before
                    # the burst -- otherwise the pacer thinks it's far
                    # behind schedule and free-runs in "catch-up" mode for
                    # the first several post-burst frames instead of
                    # resuming cleanly at the configured FPS.
                    next_time = time.monotonic() + period
                else:
                    remaining = next_time - time.monotonic()
                    if remaining > 0:
                        time.sleep(remaining)
                    next_time += period
            first = False
            prev_was_burst = is_burst
            yield self.get_frame(idx)


@dataclass
class RejectionPolicy:
    """Trivial deterministic accept/reject policy pairing with the synthetic
    source for the "30 FPS in, ~N accepted/s out" benchmark scenario.

    Not a real admission policy (see ``eligsir/mapping/scheduler`` for
    that) -- this only exists to make the synthetic source pace-able at a
    controlled acceptance rate for benchmarking the queue/consumer side of
    the pipeline in isolation.

    Args:
        accept_ratio: Fraction of frames to accept, in [0.0, 1.0].
            ``0.0`` -> reject everything (the "all_rejected" scenario).
            ``1.0`` -> accept everything.
    """

    accept_ratio: float
    _counter: float = field(default=0.0, init=False, repr=False)

    def __post_init__(self) -> None:
        if not (0.0 <= self.accept_ratio <= 1.0):
            raise ValueError(f"accept_ratio must be in [0, 1], got {self.accept_ratio}")

    def accepts(self, frame: RGBDFrame) -> bool:
        """Deterministic accumulator-based acceptance test: accepts frames at
        (as close as integer accumulation allows to) exactly ``accept_ratio``
        of calls, evenly spread rather than front- or back-loaded."""
        self._counter += self.accept_ratio
        if self._counter >= 1.0:
            self._counter -= 1.0
            return True
        return False

    @classmethod
    def for_target_rate(cls, source_fps: float, target_accepted_fps: float) -> "RejectionPolicy":
        """Convenience constructor for the "30 FPS source, ~2 accepted
        keyframes/s" style scenario spec."""
        if source_fps <= 0:
            raise ValueError("source_fps must be positive")
        ratio = max(0.0, min(1.0, target_accepted_fps / source_fps))
        return cls(accept_ratio=ratio)


class SlamFailure(RuntimeError):
    """Raised by :class:`FakeSlamSidecar` once its configured failure point
    is reached, simulating a SLAM frontend dying mid-stream."""


@dataclass
class FakeSlamSidecar:
    """Fake SLAM sidecar returning a synthetic pose for each frame, after an
    optional artificial delay, with optional injected failure after N calls.

    Mirrors the ``fail_after`` convention already established in
    ``tests/test_live_ingestion_buffer_stability.py``'s
    ``_install_synthetic_pump`` for consistency with existing prior art in
    this repo.

    Args:
        delay_seconds: Sleep this long before returning each pose (models
            SLAM processing latency, e.g. 0.5 for the "500ms SLAM delay"
            scenario). 0.0 (default) = no delay.
        fail_after: If set, the (fail_after + 1)-th call raises
            :class:`SlamFailure` instead of returning a pose (0-indexed:
            ``fail_after=20`` means calls 0..19 succeed, call 20 raises).
        realtime: If False, skip the actual ``time.sleep`` (records the
            delay that *would* have happened via ``last_delay_applied`` so
            tests can assert on it without paying real wall-clock time).
    """

    delay_seconds: float = 0.0
    fail_after: Optional[int] = None
    realtime: bool = True
    _call_count: int = field(default=0, init=False, repr=False)
    last_delay_applied: float = field(default=0.0, init=False, repr=False)

    def track(self, frame: RGBDFrame) -> np.ndarray:
        """Given a frame, sleep ``delay_seconds`` (if ``realtime``) and
        return a fake camera-to-world pose, or raise :class:`SlamFailure` if
        this call index has reached the configured ``fail_after`` point."""
        idx = self._call_count
        self._call_count += 1
        if self.fail_after is not None and idx >= self.fail_after:
            raise SlamFailure(f"synthetic SLAM frontend failure at call {idx}")
        self.last_delay_applied = self.delay_seconds
        if self.delay_seconds > 0 and self.realtime:
            time.sleep(self.delay_seconds)
        pose = np.eye(4, dtype=np.float32)
        pose[:3, 3] = frame.c2w[:3, 3]
        return pose

    def reset(self) -> None:
        self._call_count = 0
        self.last_delay_applied = 0.0
