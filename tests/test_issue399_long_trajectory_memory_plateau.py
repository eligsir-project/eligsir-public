"""Issue #399 acceptance: long-trajectory GPU memory PLATEAU regression.

What this file is, and how it differs from its two siblings
-----------------------------------------------------------
Issue #399 has two halves. The first -- per-subsystem residency *telemetry* --
landed in PRs #513/#578 and is covered by
``tests/test_gpu_memory_telemetry.py``. The second is this issue's one-line
body: *"add a regression test that verifies memory stays bounded as frame
history grows."* That is this file.

Three tests now touch this area; they are NOT duplicates:

* ``test_gpu_memory_telemetry.py::test_attribution_stays_bounded_as_history_grows``
  -- 4000 **mocked** frames, fake tensors, injected allocator readings. Pins
  the tracker's *bookkeeping arithmetic*: that already-bounded buckets read as
  bounded and an injected leak is detected.
* ``test_sensor_depth_cache_bound.py`` -- the sensor-depth LRU
  (``_enforce_sensor_depth_cache_bound``) **in isolation**, CPU, no allocator.
* **This file** -- the *composition* of every real residency mechanism at once,
  driven over a >1000-frame trajectory, asserted against the **real CUDA
  caching allocator's own numbers** (``torch.cuda.memory_allocated()`` via the
  tracker's default readers). Nothing here mocks a byte count.

The production mechanisms actually exercised (imported, not reimplemented):

* ``scene.cameras.offload_camera_image`` / ``prepare_camera_for_render``
  -- hot-window page-out and on-demand page-in.
* The admission-path **eviction cursor**, reproduced exactly as
  ``StreamingScene.ingest_next_frame`` performs it:
  ``evict_at = len(train_cameras) - hot_window - 1``, one index per admission,
  never revisited. (A *sweep* here would make the test pass for the wrong
  reason and prove nothing about production -- see #396.)
* ``utils.camera_residency.CameraResidencyLedger`` -- the #396 fix: replay
  page-ins released at the top of the next iteration.
* ``scene.cameras.offload_camera_images`` -- the #398 holdout rule.
* ``eligsir.mapping.integration.depth_helpers`` -- the #399 depth-cache
  LRU (``set_sensor_depth_cache_bound`` + ``_record_sensor_depth_cache_load``,
  i.e. the auto-enforcing production wiring, not the bare helper).
* ``eligsir.mapping.observation_db.FrameObservationDB`` -- the #333
  multires resolution-cache bounded LRU.
* ``utils.gpu_memory_telemetry.GPUMemoryResidencyTracker`` -- the measurement.

HONESTY FENCE -- what a PASS here does and does not mean
--------------------------------------------------------
A pass proves: **when the bounds are configured**, the memory owned by camera
images, holdout images, sensor-depth caches and the multires cache plateaus
instead of growing with trajectory length, measured on a real GPU allocator
over >1000 ingested frames at fixed Gaussian count and batch sizes 1/2/4.

A pass does NOT prove issue #399's acceptance criterion for a *default* run.
``--streaming_sensor_depth_cache_max_frames`` defaults to **0 = unbounded**
(``eligsir/cli/parser.py``), and that bucket is the measured dominant
owner on ScanNet++ (2.76 MB per ingested frame, ~19.4 GiB of a ~33 GiB peak --
see #399's step-6 comment). PR #782 shipped the lever; nothing has pulled it,
and flipping the default needs ResearchFlow-managed evidence, which the
existing fr3 verification (UNMANAGED dev run) is not. So: this test pins the
*mechanism*; the default remains unbounded, and #399 stays open.

Assertions are **per bucket**, never on total ``allocated_bytes``. That is
deliberate: ``prepare_camera_for_render`` caches ``cam.gsplat``
(viewmat/K/camera_center) and nothing ever clears it, so ``camera_gsplat``
grows with the number of distinct cameras ever rendered -- a real, separate,
much smaller linear term that would either fail this test for the wrong reason
or force a tolerance so loose the test became vacuous. It is measured and
reported instead (see ``test_gsplat_cache_growth_is_measured_not_hidden``).

Gating: the byte-level tests need CUDA/ROCm, because the tracker's residency
predicate is "tensor is on the measured device" -- with ``device="cpu"`` an
*offloaded* image still reads as resident and residency becomes inexpressible.
Combined with #536 (this module imports torch at module scope, so ``ci-cpu``
never collects it), these run only in the training image on a GPU box. The
``test_control_flow_*`` tests below are therefore deliberately device-free:
they guard the harness's own control flow and the occupancy invariants
(``resolution_cache_occupancy()``, depth-LRU length, ledger page-out counts)
from bit-rot wherever torch is importable at all.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from scene.cameras import (
    gpu_resident_image_count,
    offload_camera_image,
    offload_camera_images,
    prepare_camera_for_render,
)
from eligsir.mapping.integration import depth_helpers
from eligsir.mapping.observation_db import (
    BufferStage,
    FrameObservationDB,
    FrameRecord,
)
from eligsir.utils.camera_residency import CameraResidencyLedger
from eligsir.utils.gpu_memory_telemetry import GPUMemoryResidencyTracker

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="measures the real CUDA allocator"
)

# Trajectory shape. >1000 ingested frames is issue #399's own floor.
N_FRAMES = 1200
N_HOLDOUT = 150
HOT_WINDOW = 40
DEPTH_CACHE_BOUND = 64
MULTIRES_BOUND = 32
IMG_H, IMG_W = 120, 160
SNAP_EVERY = 25
#: Snapshots taken before this frame are warmup: the bounded pools are still
#: filling, so memory legitimately rises. The plateau claim is about what
#: happens *after* every working-set bound is reached.
WARMUP_FRAMES = 400
REPLAY_EVERY = 3          # how often an OLD camera is drawn instead of a recent one
HOLDOUT_EVAL_EVERY = 100  # periodic "evaluation" that pages holdouts in

_IMG_BYTES = 3 * IMG_H * IMG_W * 4
_DEPTH_BYTES = 1 * IMG_H * IMG_W * 4
_MULTIRES_BYTES = _IMG_BYTES + _DEPTH_BYTES
#: Everything one ingested frame would add if nothing were ever bounded.
PER_FRAME_PAYLOAD_BYTES = _IMG_BYTES + _DEPTH_BYTES + _MULTIRES_BYTES

#: The buckets the configured bounds are supposed to bound. `camera_gsplat` is
#: deliberately absent -- see the module docstring.
BOUNDED_BUCKETS = (
    "camera_hot_images",
    "camera_cold_images",
    "camera_holdout_images",
    "camera_depth_cache",
    "multires_cache",
)


# --------------------------------------------------------------- harness ----


class _Frame:
    """``cam._streaming_frame`` -- the tracker reads only ``.index``."""

    __slots__ = ("index",)

    def __init__(self, index: int):
        self.index = index


class _Cam:
    """Duck-typed camera carrying only what the production residency functions
    touch. ``prepare_camera_for_render`` reads ``image_width``/``image_height``/
    ``world_view_transform``/``FoVx``/``FoVy`` (to build the gsplat tensors on
    first render, exactly as production does -- lazily, not at construction)
    and ``original_image``; the depth LRU reads ``uid`` and
    ``_sensor_depth_cache``."""

    def __init__(self, uid: int, device: str, frame_index: int):
        self.uid = uid
        self.image_name = f"cam{uid}"
        self.image_width = IMG_W
        self.image_height = IMG_H
        self.FoVx = 1.0
        self.FoVy = 0.8
        self.world_view_transform = torch.eye(4, device=device)
        self.camera_center = torch.zeros(3, device=device)
        self.original_image = torch.zeros(3, IMG_H, IMG_W, device=device)
        self._sensor_depth_cache = None
        self._streaming_frame = _Frame(frame_index)


class _Args:
    def __init__(self, hot_window: int, sampling_window: int):
        self.streaming_keyframe_window = hot_window
        self.streaming_offload_camera_images = True
        self._sampling_window = sampling_window


class _Scene:
    """The subset of ``StreamingScene`` the tracker's walk and the residency
    ledger read. ``_effective_keyframe_window`` is deliberately much larger
    than the offload bound (the real ``streaming_keyframe_coverage=0.6``
    situation) so a regression that keys residency on the *sampling* window
    is caught rather than hidden -- see #396 and the tracker's
    ``_collect_cameras`` docstring."""

    def __init__(self, hot_window: int, db):
        self.args = _Args(hot_window, int(0.6 * N_FRAMES))
        self.train_cameras = []
        self._test_cameras = []
        self._replay_buffer = []
        self._global_reservoir = []
        self.observation_db = db
        self._hot_window = hot_window

    def _effective_keyframe_window(self):
        return self.args._sampling_window

    def hot_camera_ids(self, current=None):
        """Mirrors ``StreamingScene.hot_camera_ids``: the tail of the RAW
        keyframe window, plus the camera of the current step."""
        hot = {id(c) for c in self.train_cameras[-self._hot_window:]}
        if current is not None:
            hot.add(id(current))
        return hot


def _load_depth(cam, device):
    """Stands in for ``_get_sensor_depth``'s cache-miss branch: materialize the
    depth tensor and register it with the REAL LRU, which is what enforces the
    configured bound on every load."""
    if cam._sensor_depth_cache is None:
        cam._sensor_depth_cache = torch.zeros(1, IMG_H, IMG_W, device=device)
    depth_helpers._record_sensor_depth_cache_load(cam)


class _Handle:
    """Stand-in for ``CacheHandle``: the tracker's multires walk reads only
    ``.rgb``/``.depth``. Deliberately a module-level class with INSTANCE
    attributes -- an inline ``type("H", (), {...})()`` would park the tensors
    on a dynamically created class object, which lands in a gc cycle and keeps
    evicted frames' pixels alive until a collection runs, making the harness's
    own totals lie about what the eviction policy actually released."""

    __slots__ = ("rgb", "depth")

    def __init__(self, device: str):
        self.rgb = torch.zeros(3, IMG_H, IMG_W, device=device)
        self.depth = torch.zeros(1, IMG_H, IMG_W, device=device)


def _make_record(frame_id: int, device: str) -> FrameRecord:
    handle = _Handle(device)
    return FrameRecord(
        frame_id=frame_id,
        pose_c2w=np.eye(4),
        timestamp=float(frame_id),
        submap_id=0,
        buffer_stage=BufferStage.MAPPING_KEYFRAME,
        visible_gaussian_uids=torch.zeros(0, dtype=torch.int64),
        resolution_cache={"high": handle},
    )


class _Gaussians:
    """FIXED population -- issue #399 asks for the plateau at fixed/nearly-fixed
    N so that any growth is unambiguously residency, not the model."""

    def __init__(self, n: int, device: str):
        self.params = torch.nn.ParameterDict(
            {
                "means": torch.nn.Parameter(torch.zeros(n, 3, device=device)),
                "opacities": torch.nn.Parameter(torch.zeros(n, device=device)),
            }
        )
        self.optimizer = type(
            "Opt",
            (),
            {
                "state": {
                    p: {"exp_avg": torch.zeros_like(p), "exp_avg_sq": torch.zeros_like(p)}
                    for p in self.params.values()
                }
            },
        )()


def run_trajectory(
    *,
    device: str,
    batch_size: int,
    depth_bound: int = DEPTH_CACHE_BOUND,
    multires_bound: int = MULTIRES_BOUND,
    ledger_enabled: bool = True,
    offload_enabled: bool = True,
    n_frames: int = N_FRAMES,
):
    """Drive a long streaming trajectory through the real residency machinery.

    Returns ``(snapshots, meta)``. Each snapshot is a real
    ``GPUMemoryResidencyTracker.snapshot()`` result; on CUDA its
    ``allocated_bytes`` comes from ``torch.cuda.memory_allocated()``.
    """
    # The depth LRU is module-global process state; reset it so tests do not
    # inherit each other's cameras (and restore the bound at the end).
    depth_helpers._SENSOR_DEPTH_CACHE_LRU.clear()
    depth_helpers.set_sensor_depth_cache_bound(depth_bound)

    db = FrameObservationDB(device=device, max_resolution_cache_frames=multires_bound)
    scene = _Scene(HOT_WINDOW, db)
    scene.args.streaming_offload_camera_images = offload_enabled
    gaussians = _Gaussians(n=4096, device=device)
    ledger = CameraResidencyLedger(enabled=ledger_enabled)
    tracker = GPUMemoryResidencyTracker(device=device)

    # Holdouts exist from the start and are CPU-resident by policy (#398).
    for i in range(N_HOLDOUT):
        cam = _Cam(uid=1_000_000 + i, device=device, frame_index=-1)
        offload_camera_image(cam)
        scene._test_cameras.append(cam)

    rng = np.random.default_rng(0)
    snapshots = []
    evicted_indices: list = []

    for f in range(n_frames):
        # --- top of iteration: the #396 ledger releases last step's page-ins.
        current = scene.train_cameras[-1] if scene.train_cameras else None
        ledger.release(scene.hot_camera_ids(current))

        # --- admission
        cam = _Cam(uid=f, device=device, frame_index=f)
        scene.train_cameras.append(cam)
        db.add_frame(_make_record(f, device))

        # --- admission-path eviction cursor, EXACTLY as production does it:
        # one index per admission, never revisited (a sweep would prove nothing).
        if offload_enabled:
            evict_at = len(scene.train_cameras) - HOT_WINDOW - 1
            if evict_at >= 0:
                offload_camera_image(scene.train_cameras[evict_at])
                evicted_indices.append(evict_at)

        # --- one optimizer step over B supervision views
        views = [cam]
        while len(views) < batch_size:
            if f > HOT_WINDOW and (f % REPLAY_EVERY == 0):
                # global/history replay: an OLD camera, far past the offload bound
                idx = int(rng.integers(0, max(1, len(scene.train_cameras) - HOT_WINDOW)))
            else:
                idx = len(scene.train_cameras) - 1 - int(
                    rng.integers(0, min(HOT_WINDOW, len(scene.train_cameras)))
                )
            views.append(scene.train_cameras[idx])
        for v in views:
            prepare_camera_for_render(v, device=device)
            ledger.note_paged_in(v)
            _load_depth(v, device)

        # --- periodic holdout evaluation, returned to bound afterwards (#398)
        if f and f % HOLDOUT_EVAL_EVERY == 0:
            for t in scene._test_cameras:
                prepare_camera_for_render(t, device=device)
            offload_camera_images(scene._test_cameras)

        if f % SNAP_EVERY == 0:
            snapshots.append(
                tracker.snapshot(
                    iteration=f,
                    frames_ingested=f + 1,
                    gaussians=gaussians,
                    streaming_scene=scene,
                )
            )

    meta = {
        "tracker": tracker,
        "scene": scene,
        "db": db,
        "ledger": ledger,
        "depth_lru_len": len(depth_helpers._SENSOR_DEPTH_CACHE_LRU),
        "resolution_cache_occupancy": db.resolution_cache_occupancy,
        "resolution_cache_evictions": db.resolution_cache_evictions,
        "holdout_gpu_resident": gpu_resident_image_count(scene._test_cameras),
        "evicted_indices": evicted_indices,
    }
    depth_helpers.set_sensor_depth_cache_bound(0)
    depth_helpers._SENSOR_DEPTH_CACHE_LRU.clear()
    return snapshots, meta


# ------------------------------------------------------------- analysis -----


def _slope_per_frame(snapshots, key_fn) -> float:
    """Least-squares bytes-per-ingested-frame, the same statistic
    ``GPUMemoryResidencyTracker.growth_stats()`` reports."""
    xs = [float(s["frames_ingested"]) for s in snapshots]
    ys = [float(key_fn(s)) for s in snapshots]
    n = float(len(xs))
    mx = sum(xs) / n
    denom = sum((x - mx) ** 2 for x in xs)
    if denom <= 0:
        return 0.0
    my = sum(ys) / n
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denom


def _post_warmup(snapshots):
    out = [s for s in snapshots if s["frames_ingested"] >= WARMUP_FRAMES]
    assert len(out) >= 8, "not enough post-warmup samples to judge a plateau"
    return out


# ------------------------------------------------------ the real GPU test ---


@requires_cuda
@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_bounded_residency_plateaus_over_a_long_trajectory(batch_size, capsys):
    """#399's acceptance mechanism: with the bounds configured, every bounded
    bucket's bytes plateau instead of scaling with total history length.

    Measured against ``torch.cuda.memory_allocated()``-backed attribution over
    1200 ingested frames at fixed N, with replay of old frames, 150 holdouts,
    a 32-frame multires bound and a 64-frame depth-cache bound.
    """
    snaps, meta = run_trajectory(device="cuda", batch_size=batch_size)
    tail = _post_warmup(snaps)

    report = {}
    for bucket in BOUNDED_BUCKETS:
        slope = _slope_per_frame(tail, lambda s, b=bucket: s["buckets"][b])
        report[bucket] = slope
        # "Bounded" means the per-frame slope is a negligible fraction of what
        # one unbounded frame would have added.
        assert abs(slope) < 0.02 * PER_FRAME_PAYLOAD_BYTES, (
            f"bucket {bucket!r} grows at {slope:,.0f} B per ingested frame "
            f"(>{0.02 * PER_FRAME_PAYLOAD_BYTES:,.0f}); residency is not bounded"
        )

    # A slope near zero is necessary but not sufficient (a sawtooth averages
    # out): the tail's peak must also not exceed its own median materially.
    for bucket in BOUNDED_BUCKETS:
        vals = sorted(s["buckets"][bucket] for s in tail)
        med = vals[len(vals) // 2]
        if med:
            assert max(vals) <= 1.10 * med, (
                f"bucket {bucket!r} peaks {max(vals) / med:.2f}x its own tail median"
            )

    # The occupancy invariants the bounds exist to enforce.
    for s in tail:
        assert s["multires_cached_frames"] <= MULTIRES_BOUND
        assert s["holdout_resident_count"] == 0, "#398: holdouts must not stay resident"
        # #396: past the offload bound, only the current batch may be resident.
        assert s["resident_beyond_offload_bound_count"] <= batch_size
        assert s["buckets"]["camera_depth_cache"] <= (DEPTH_CACHE_BOUND + batch_size) * _DEPTH_BYTES
    assert meta["depth_lru_len"] <= DEPTH_CACHE_BOUND + batch_size
    assert meta["resolution_cache_occupancy"] <= MULTIRES_BOUND
    assert meta["holdout_gpu_resident"] == 0

    # The TOTAL is not flat, and the residual is understood rather than
    # tolerated: it is per-CAMERA (identical at B=1/2/4), not per-pixel --
    # the cached gsplat tensors plus the caching allocator's minimum block
    # size rounding up four tiny per-camera tensors (measured 2,048 B/frame
    # against 112 B of actual storage). It must stay orders of magnitude
    # below one frame's pixel payload; if it ever approaches it, a real
    # image-sized owner has appeared outside every named bucket.
    total_slope = _slope_per_frame(tail, lambda s: s["allocated_bytes"])
    assert 0 <= total_slope < 0.02 * PER_FRAME_PAYLOAD_BYTES, (
        f"total allocated grows {total_slope:,.0f} B/frame -- an image-sized "
        "owner exists outside the attributed buckets"
    )

    gsplat_slope = _slope_per_frame(tail, lambda s: s["buckets"]["camera_gsplat"])
    with capsys.disabled():
        print(f"\n[#399 B={batch_size}] post-warmup bytes/frame by bucket:")
        for k, v in report.items():
            print(f"    {k:<24} {v:12,.1f}")
        print(f"    {'camera_gsplat (unbounded)':<24} {gsplat_slope:12,.1f}")
        print(
            f"    {'unattributed':<24} "
            f"{_slope_per_frame(tail, lambda s: s['buckets']['unattributed']):12,.1f}"
        )
        print(
            f"    allocated_bytes tail: "
            f"{tail[0]['allocated_bytes'] / 2**20:.1f} MiB -> "
            f"{tail[-1]['allocated_bytes'] / 2**20:.1f} MiB "
            f"({_slope_per_frame(tail, lambda s: s['allocated_bytes']):,.0f} B/frame)"
        )
        print(f"    ledger: {meta['ledger'].stats()}")
        print(f"    multires evictions: {meta['resolution_cache_evictions']}")


@requires_cuda
def test_unbounded_depth_cache_is_detected_as_growing(capsys):
    """POSITIVE CONTROL. With the depth bound off (``max_frames=0`` -- today's
    DEFAULT), the same harness must show `camera_depth_cache` growing at
    ~one depth tensor per ingested frame. If this ever passes as "bounded",
    the test above has lost its teeth."""
    snaps, _ = run_trajectory(
        device="cuda", batch_size=2, depth_bound=0, n_frames=800
    )
    tail = _post_warmup(snaps)
    slope = _slope_per_frame(tail, lambda s: s["buckets"]["camera_depth_cache"])
    with capsys.disabled():
        print(
            f"\n[#399 control: depth bound OFF] camera_depth_cache "
            f"{slope:,.0f} B/frame (one depth tensor = {_DEPTH_BYTES:,} B)"
        )
    assert slope > 0.5 * _DEPTH_BYTES, (
        "the unbounded depth cache must be DETECTED as growing; measured "
        f"{slope:,.0f} B/frame"
    )
    # ...and the growth must land in that specific bucket, not merely somewhere.
    assert abs(_slope_per_frame(tail, lambda s: s["buckets"]["multires_cache"])) < (
        0.02 * PER_FRAME_PAYLOAD_BYTES
    )


@requires_cuda
def test_disabled_residency_ledger_is_detected_as_growing(capsys):
    """POSITIVE CONTROL for the #396 fix specifically. ``ledger_enabled=False``
    is the real ``--no-streaming_offload_camera_images`` escape hatch: replayed
    historical cameras are paged in and never paged out, so
    `camera_cold_images` -- and only it -- must grow."""
    snaps, meta = run_trajectory(
        device="cuda", batch_size=4, ledger_enabled=False, n_frames=800
    )
    tail = _post_warmup(snaps)
    cold = _slope_per_frame(tail, lambda s: s["buckets"]["camera_cold_images"])
    resident = _slope_per_frame(
        tail, lambda s: s["resident_beyond_offload_bound_count"]
    )
    with capsys.disabled():
        print(
            f"\n[#399 control: residency ledger OFF] camera_cold_images "
            f"{cold:,.0f} B/frame, resident-past-bound {resident:.3f} cams/frame"
        )
    assert cold > 0.1 * _IMG_BYTES, (
        f"a leaked replay page-in must be visible; measured {cold:,.0f} B/frame"
    )
    assert resident > 0.0
    # The depth bound is still on here, so that bucket must stay flat --
    # pinning the leak to the ledger rather than to "something grew".
    assert abs(_slope_per_frame(tail, lambda s: s["buckets"]["camera_depth_cache"])) < (
        0.02 * PER_FRAME_PAYLOAD_BYTES
    )


@requires_cuda
def test_gsplat_cache_growth_is_measured_not_hidden(capsys):
    """``prepare_camera_for_render`` caches viewmat/K/camera_center per camera
    and nothing clears them, so `camera_gsplat` grows with the number of
    DISTINCT cameras ever rendered. That is a real (small) unbounded term this
    file deliberately does not fold into the plateau claim; it is asserted to
    be small-but-nonzero so the number stays honest and visible.

    It is absent from #399's ScanNet++ owner table, so quantifying it here is
    new information, not a re-measurement."""
    snaps, _ = run_trajectory(device="cuda", batch_size=2, n_frames=800)
    tail = _post_warmup(snaps)
    slope = _slope_per_frame(tail, lambda s: s["buckets"]["camera_gsplat"])
    per_cam = 16 * 4 + 9 * 4 + 3 * 4  # viewmat + K + camera_center, float32
    with capsys.disabled():
        print(
            f"\n[#399] camera_gsplat {slope:,.1f} B/frame "
            f"(one camera's gsplat tensors = {per_cam} B)"
        )
    assert slope > 0.0, "gsplat caching is expected to grow per rendered camera"
    # Three orders of magnitude below an image: real, but not the VRAM story.
    assert slope < 0.01 * _IMG_BYTES


# ------------------------------------------- device-free control-flow guard --
#
# These do not measure bytes and do not need a GPU (see the module docstring's
# gating note): they keep the harness above -- and the occupancy invariants it
# leans on -- from silently rotting on a machine that never runs the CUDA
# tests.


def test_control_flow_bounds_occupancy_without_a_gpu():
    snaps, meta = run_trajectory(device="cpu", batch_size=2, n_frames=300)
    assert len(snaps) == 12
    assert meta["resolution_cache_occupancy"] <= MULTIRES_BOUND
    assert meta["resolution_cache_evictions"] >= 300 - MULTIRES_BOUND
    assert meta["depth_lru_len"] <= DEPTH_CACHE_BOUND + 2
    # The ledger really ran: page-ins happened and non-hot ones were released.
    stats = meta["ledger"].stats()
    assert stats["paged_in"] > 0 and stats["paged_out"] > 0


def test_control_flow_unbounded_config_really_is_unbounded():
    """The controls' premise: with bounds off, occupancy is NOT capped."""
    _, meta = run_trajectory(
        device="cpu", batch_size=2, depth_bound=0, multires_bound=0, n_frames=300
    )
    assert meta["resolution_cache_occupancy"] == 300
    assert meta["resolution_cache_evictions"] == 0
    assert meta["depth_lru_len"] > DEPTH_CACHE_BOUND


def test_control_flow_eviction_cursor_matches_production():
    """The harness must reproduce ``ingest_next_frame``'s single-index cursor,
    not a sweep. After N admissions with no replay (B=1), exactly the last
    HOT_WINDOW cameras plus the current one have ever escaped eviction."""
    _, meta = run_trajectory(device="cpu", batch_size=1, n_frames=200)
    assert len(meta["scene"].train_cameras) == 200
    # Each index is visited exactly once, in order, and never revisited --
    # which is precisely the property that made #396 possible and that a
    # (wrong, O(N)) sweep-based harness would silently paper over.
    assert meta["evicted_indices"] == list(range(200 - HOT_WINDOW))
