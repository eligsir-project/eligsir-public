"""Real-GPU VRAM repro/verification for issue #398.

`tests/test_issue398_holdout_image_residency.py` proves the *mechanism*
(residency COUNT is O(1) rather than O(n_holdouts)) using a fake-device tensor,
because that suite runs in CPU-only CI. This script proves the same claim with
REAL CUDA/ROCm tensors and the real allocator, closing the "no GPU available"
gap noted in the issue's own PR/comment history (acceptance criteria 1, 2, 4
are VRAM measurements and were explicitly left unverified there).

Run inside the training container (real GPU passthrough required):

    docker compose run --rm train python scripts/issue398_gpu_vram_repro.py

It ingests a synthetic stream of frames through the REAL
`StreamingScene.ingest_next_frame` / `_frame_to_camera` / eviction machinery
(same harness shape as the unit test's `_FakeIngestSelf`, but with real
960x720 float32 image tensors uploaded to the real GPU by the real,
unstubbed `prepare_camera_for_render`), at a holdout count matched to the
issue's own motivating ScanNet++ #290 shape (993 holdouts, 960x720), and
reports:

  - `gpu_resident_test_image_count()` / `gpu_resident_image_bytes()` (the
    production accounting helpers) after full ingestion, with the fix ON.
  - The delta in `torch.cuda.memory_allocated()` across ingestion, with the
    fix ON vs. OFF (`--no-streaming_offload_camera_images`), to show the O(n)
    growth this issue describes is real, and that the fix actually removes it
    from the real allocator, not just from the accounting helper.

Read-only wrt the repo; writes nothing to disk. Not part of the pytest suite
(needs a real GPU) -- run by hand or from an agent doing GPU verification.
"""

from __future__ import annotations

import gc
from types import SimpleNamespace

import numpy as np
import torch

from scene.cameras import Camera, gpu_resident_image_bytes
from scene.streaming_scene import StreamingScene


class _RealFrame:
    def __init__(self, index: int) -> None:
        self.index = index
        self._odom_valid = True
        self.timestamp = float(index)


def _make_real_camera(index: int, h: int, w: int) -> Camera:
    """Build a REAL `scene.cameras.Camera` (not a stand-in), with a real
    CPU-resident image tensor -- exactly the state `_frame_to_camera(...,
    load_image=True)` leaves a freshly-decoded camera in before the ingest
    tail's `prepare_camera_for_render(cam, device="cuda")` uploads it."""
    R = np.eye(3, dtype=np.float64)
    T = np.zeros(3, dtype=np.float64)
    image = torch.rand(3, h, w, dtype=torch.float32, device="cpu")
    cam = Camera(
        colmap_id=index, R=R, T=T, FoVx=1.0, FoVy=1.0,
        image=image, gt_alpha_mask=None,
        image_name=f"{index:06d}", uid=index,
        data_device="cpu",
        fx=500.0, fy=500.0, cx=w / 2.0, cy=h / 2.0,
        width=w, height=h,
    )
    return cam


class _RealIngestSelf:
    """Same harness shape as tests/test_issue398_holdout_image_residency.py's
    `_FakeIngestSelf`, but produces real CPU tensors that the real (unstubbed)
    `prepare_camera_for_render` genuinely uploads to the GPU."""

    def __init__(self, frames, holdout_indices, offload: bool, img_hw):
        self.args = SimpleNamespace(
            streaming_eval_hold=4,
            streaming_eval_min_baseline_m=0.03,
            streaming_eval_min_angle_deg=2.0,
            streaming_eval_guard=0,
            streaming_offload_camera_images=offload,
            streaming_replay_buffer=512,
            streaming_frame_admission="all",
            streaming_global_reservoir_stride=0,
            streaming_keyframe_window=8,
            streaming_track_scene_complexity=False,
            streaming_inactive_cache_size=0,
        )
        self._img_hw = img_hw
        self._active_source_indices = set()
        self._active_cam_by_idx = {}
        self._last_admitted_source_idx = None
        self._frames = frames
        self._source_idx = 0
        self._keyframes_rejected = 0
        self._keyframes_admitted = 0
        self._holdout_indices = set(holdout_indices)
        self._guard_indices = set()
        self._test_cameras = []
        self.train_cameras = []
        self._replay_buffer = []
        self._n_train_frames_ingested = 0
        self._global_reservoir = []
        self._steps_since_new_frame = 0
        self._pending_extent_update = False
        self.current_camera = None
        self._dense_trajectory = []
        self._dense_trajectory_flushed = 0

    def has_next_frame(self):
        return self._source_idx < len(self._frames)

    def _get_source_frame(self, index):
        if index >= len(self._frames):
            return None
        return self._frames[index]

    def _frame_to_camera(self, frame, load_image=True):
        h, w = self._img_hw
        cam = _make_real_camera(frame.index, h=h, w=w)
        cam._streaming_frame = frame
        return cam

    def accumulate_observed_surface(self, frame):
        pass

    def _cache_inactive_frame(self, frame):
        pass

    def _release_source_frames_upto_cursor(self):
        pass

    def _append_dense_trajectory(self, cam, frame, is_train, is_holdout):
        self._dense_trajectory.append(
            {"index": frame.index, "is_holdout": bool(is_holdout),
             "admitted": bool(is_train and not is_holdout)}
        )

    def flush_deferred_extent_update(self):
        pass

    def update_cameras_extent(self):
        pass

    # Bound from the real StreamingScene -- never stubbed. Note
    # `prepare_camera_for_render` is intentionally NOT stubbed here (unlike
    # the CPU unit test): we want the real CUDA/ROCm upload.
    offload_test_camera_images = StreamingScene.offload_test_camera_images
    gpu_resident_test_image_count = StreamingScene.gpu_resident_test_image_count
    _offload_camera_hot_window = StreamingScene._offload_camera_hot_window
    _offload_evicted_camera = StreamingScene._offload_evicted_camera


def _ingest_all(n_frames: int, holdout_every: int, offload: bool, img_hw):
    frames = [_RealFrame(i) for i in range(n_frames)]
    holdouts = {i for i in range(n_frames) if i % holdout_every == 0}
    fake_self = _RealIngestSelf(frames, holdouts, offload=offload, img_hw=img_hw)
    for _ in range(n_frames):
        StreamingScene.ingest_next_frame(fake_self)
    return fake_self


def _mib(nbytes: int) -> float:
    return nbytes / (1024.0 * 1024.0)


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit(
            "No CUDA/ROCm device available -- this script must run inside the "
            "training container with GPU device passthrough (docker-compose's "
            "`train` service). Refusing to fabricate a result."
        )

    device_name = torch.cuda.get_device_name(0)
    print(f"[issue398 GPU repro] device = {device_name}")

    # Matched to the issue's own motivating shape: #290 ScanNet++, 993
    # holdouts at 960x720. holdout_every=1 -> every ingested frame is a
    # holdout, so n_frames == n_holdouts exactly.
    n_frames = 993
    img_hw = (720, 960)

    for label, offload in (("BEFORE (offload disabled, reproduces #398)", False),
                            ("AFTER (offload enabled, the shipped fix)", True)):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        start_alloc = torch.cuda.memory_allocated()

        scene = _ingest_all(n_frames=n_frames, holdout_every=1, offload=offload, img_hw=img_hw)

        end_alloc = torch.cuda.memory_allocated()
        peak_alloc = torch.cuda.max_memory_allocated()
        resident_count = scene.gpu_resident_test_image_count()
        resident_bytes = gpu_resident_image_bytes(scene._test_cameras)

        print(f"\n=== {label} ===")
        print(f"  n_holdouts                     = {len(scene._test_cameras)}")
        print(f"  gpu_resident_test_image_count() = {resident_count}")
        print(f"  gpu_resident_image_bytes()      = {resident_bytes} ({_mib(resident_bytes):.1f} MiB)")
        print(f"  torch.cuda.memory_allocated() delta = {_mib(end_alloc - start_alloc):.1f} MiB")
        print(f"  torch.cuda.max_memory_allocated()   = {_mib(peak_alloc):.1f} MiB")

        # Release this arm's real GPU tensors before the next arm.
        del scene
        gc.collect()
        torch.cuda.empty_cache()

    print(
        "\n[issue398 GPU repro] Expected result: BEFORE shows resident_count == "
        f"n_holdouts ({n_frames}) and an allocator delta of several GiB "
        f"(~{_mib(n_frames * 3 * img_hw[0] * img_hw[1] * 4):.0f} MiB at this "
        "shape); AFTER shows resident_count == 0 and an allocator delta close "
        "to zero (residual allocator overhead only)."
    )

    # Acceptance criterion 2 ("VRAM does not scale linearly with total
    # holdout count") at two very different holdout counts, fix ON both
    # times -- a real-allocator counterpart to the CPU unit test
    # `test_residency_does_not_scale_with_holdout_count`.
    print("\n[issue398 GPU repro] Criterion 2: resident bytes at two holdout counts (fix ON):")
    for n in (50, 993):
        gc.collect()
        torch.cuda.empty_cache()
        start_alloc = torch.cuda.memory_allocated()
        scene = _ingest_all(n_frames=n, holdout_every=1, offload=True, img_hw=img_hw)
        end_alloc = torch.cuda.memory_allocated()
        resident_count = scene.gpu_resident_test_image_count()
        resident_bytes = gpu_resident_image_bytes(scene._test_cameras)
        print(
            f"  n_holdouts={n:4d}  resident_count={resident_count}  "
            f"resident_bytes={resident_bytes}  allocator_delta={_mib(end_alloc - start_alloc):.1f} MiB"
        )
        del scene
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
