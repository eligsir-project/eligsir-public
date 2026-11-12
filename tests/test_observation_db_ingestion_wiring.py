"""Tests for the per-frame observation-DB recording helper (task 4)."""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest
import torch

from eligsir.mapping.observation_db import BufferStage, FrameObservationDB, FrameRecord
import eligsir.mapping.resolution_cache as resolution_cache
from eligsir.mapping.resolution_cache import build_resolution_cache
from train_streaming import bind_resolution_tier, record_frame_observation
from eligsir.utils.streaming_frames import StreamingRGBDFrame


class _FakeModel:
    def __init__(self, xyz: torch.Tensor, uids: torch.Tensor) -> None:
        self._xyz = xyz
        self.gaussian_uid = uids

    @property
    def get_xyz(self) -> torch.Tensor:
        return self._xyz


class _FakeCam:
    def __init__(self) -> None:
        self.R = np.eye(3, dtype=np.float32)
        self.T = np.zeros(3, dtype=np.float32)
        self.fx = self.fy = 500.0
        self.cx, self.cy = 320.0, 240.0
        self.image_width, self.image_height = 640, 480
        self.c2w = np.eye(4, dtype=np.float32)


def test_record_frame_observation_adds_a_record_for_every_candidate():
    db = FrameObservationDB(device="cpu")
    cam = _FakeCam()
    visible_uids = torch.tensor([10, 11], dtype=torch.int64)
    record_frame_observation(
        db, cam=cam, visible_uids=visible_uids, frame_id=0, submap_id=0,
        is_train=True, is_holdout=False,
    )
    assert len(db) == 1
    rec = db.get_frame(0)
    assert rec.buffer_stage == BufferStage.MAPPING_KEYFRAME
    assert set(rec.visible_gaussian_uids.tolist()) == {10, 11}


def test_record_frame_observation_handles_none_visible_uids():
    """Regression test for the 2026-07-26 crash: bootstrap frames have no
    Gaussians to project visibility against yet, so _bootstrap_record_fn
    passes visible_uids=None -- add_frame must not crash on it."""
    db = FrameObservationDB(device="cpu")
    cam = _FakeCam()
    record_frame_observation(
        db, cam=cam, visible_uids=None, frame_id=0, submap_id=0,
        is_train=True, is_holdout=False,
    )
    rec = db.get_frame(0)
    assert rec is not None
    assert rec.visible_gaussian_uids.numel() == 0


def test_record_frame_observation_tags_holdout_frames():
    db = FrameObservationDB(device="cpu")
    cam = _FakeCam()
    visible_uids = torch.zeros(0, dtype=torch.int64)
    record_frame_observation(
        db, cam=cam, visible_uids=visible_uids, frame_id=1, submap_id=0,
        is_train=False, is_holdout=True,
    )
    assert db.get_frame(1).buffer_stage == BufferStage.HOLDOUT_TEST


def test_record_frame_observation_tags_non_admitted_non_holdout_as_evicted():
    db = FrameObservationDB(device="cpu")
    cam = _FakeCam()
    visible_uids = torch.zeros(0, dtype=torch.int64)
    record_frame_observation(
        db, cam=cam, visible_uids=visible_uids, frame_id=2, submap_id=0,
        is_train=False, is_holdout=False,
    )
    # A rejected-by-admission candidate is not a mapping keyframe and not a
    # holdout test frame -- closest existing stage is EVICTED_REDUNDANT
    # (never was in the active buffer to begin with, but "not currently
    # contributing to mapping" is the field's actual meaning here; there is
    # no eviction *policy* yet per this plan's non-goals, this is just the
    # correct static tag).
    assert db.get_frame(2).buffer_stage == BufferStage.EVICTED_REDUNDANT


# ---------------------------------------------------------------------------
# I1: real-orchestration ordering test.
#
# C1/C2 slipped through review because every prior test exercised
# `admit()`/`record_frame_observation()` in isolation with hand-set fixture
# state, never the real `StreamingScene.ingest_next_frame` orchestration.
# This test calls the REAL, unmodified `ingest_next_frame` method (bound onto
# a minimal fake `self`) so a regression in call ORDER inside that method
# fails this test, not just a unit test of the helpers it calls.
# ---------------------------------------------------------------------------

from types import SimpleNamespace

from scene.streaming_scene import StreamingScene


class _FakeFrame:
    def __init__(self, index: int) -> None:
        self.index = index
        self._odom_valid = True


class _FakeIngestSelf:
    """Minimal stand-in for a StreamingScene instance.

    Only implements the attributes/methods `ingest_next_frame` actually
    touches for the `admission_mode == "all"`, non-holdout, non-batched
    code path -- everything else is a plain no-op so the test stays fast
    and CPU-only while still running the REAL `ingest_next_frame` body.
    """

    def __init__(self, frames):
        self.args = SimpleNamespace(
            streaming_eval_hold=0,
            streaming_offload_camera_images=False,
            streaming_replay_buffer=512,
            streaming_frame_admission="all",
            streaming_global_reservoir_stride=0,
            streaming_keyframe_window=8,
            streaming_track_scene_complexity=False,
        )
        self._frames = frames
        self._source_idx = 0
        self._keyframes_rejected = 0
        self._keyframes_admitted = 0
        self._holdout_indices = None
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

    def _append_dense_trajectory(self, cam, frame, is_train, is_holdout):
        # Not under test here (see test_streaming_scene_dense_trajectory.py) --
        # ingest_next_frame calls this unconditionally, so the fake needs a
        # no-op stand-in to avoid an AttributeError.
        pass

    def _cache_inactive_frame(self, frame):
        # Not under test here (see test_partition_focus_sampling.py /
        # test_streaming_scene_lazy_camera_image.py, which exercise the
        # real tiered inactive-frame cache) -- ingest_next_frame calls this
        # on every pregate/admission rejection, so the fake needs a no-op
        # stand-in to avoid an AttributeError.
        pass

    def has_next_frame(self):
        return self._source_idx < len(self._frames)

    def _get_source_frame(self, index):
        if index >= len(self._frames):
            return None
        return self._frames[index]

    def accumulate_observed_surface(self, frame):
        return None

    def _frame_to_camera(self, frame, load_image=True):
        cam = SimpleNamespace()
        cam.c2w = np.eye(4, dtype=np.float32)
        cam._streaming_frame = frame
        cam.materialize_pending_image = lambda: None
        return cam

    def update_cameras_extent(self):
        return None

    def _release_source_frames_upto_cursor(self):
        # WS-D1's live-frame-buffer eviction hook; irrelevant to this
        # test's admission-wiring checks, no-op like the other stubs.
        return None

    # Issue #396: the hot-window eviction cursor, extracted out of
    # ingest_next_frame's inline body into its own method. Bound from the REAL
    # StreamingScene (not stubbed) so this test keeps exercising the real
    # admission-path body; it self-disables on
    # streaming_offload_camera_images=False, which this fake sets.
    _offload_camera_hot_window = StreamingScene._offload_camera_hot_window
    _offload_evicted_camera = StreamingScene._offload_evicted_camera
    # Issue #1115: bound from the REAL StreamingScene -- #1084 extracted
    # `ingest_next_frame`'s inline holdout/guard computation into this
    # predicate verbatim, and the holdout test below flips `_holdout_indices`
    # to exercise that branch, so it must be the production rule.
    holdout_status = StreamingScene.holdout_status


def test_ingest_next_frame_calls_visibility_before_admission(monkeypatch):
    monkeypatch.setattr(
        "scene.cameras.prepare_camera_for_render", lambda cam, device="cuda": None
    )
    fake_self = _FakeIngestSelf([_FakeFrame(0)])

    calls: list[str] = []

    def visibility_fn(cam, frame):
        calls.append("visibility")
        cam._observed_gaussian_uids = torch.tensor([1, 2, 3])

    def admission_fn(cam, frame):
        calls.append("admission")
        # C1 regression check: by the time admission_fn runs, visibility_fn
        # must already have stashed its result on `cam`.
        assert getattr(cam, "_observed_gaussian_uids", None) is not None
        return True, {"admitted": True}, None

    result = StreamingScene.ingest_next_frame(
        fake_self, admission_fn=admission_fn, visibility_fn=visibility_fn
    )

    assert result is not None
    assert calls == ["visibility", "admission"]


def test_ingest_next_frame_records_admission_rejected_frames(monkeypatch):
    monkeypatch.setattr(
        "scene.cameras.prepare_camera_for_render", lambda cam, device="cuda": None
    )
    fake_self = _FakeIngestSelf([_FakeFrame(0)])

    recorded: list[tuple] = []

    def visibility_fn(cam, frame):
        cam._observed_gaussian_uids = torch.tensor([1, 2, 3])

    def admission_fn(cam, frame):
        return False, {"admitted": False}, None

    def record_fn(cam, frame, is_train, is_holdout):
        recorded.append((frame.index, is_train, is_holdout))

    result = StreamingScene.ingest_next_frame(
        fake_self, admission_fn=admission_fn, visibility_fn=visibility_fn,
        record_fn=record_fn,
    )

    # C2 regression check: `ingest_next_frame` returns cam=None for a
    # rejected candidate, but record_fn must still have fired for it.
    assert result == (None, fake_self._frames[0], False)
    assert recorded == [(0, False, False)]


def test_ingest_next_frame_records_admitted_frames():
    fake_self = _FakeIngestSelf([_FakeFrame(0)])
    recorded: list[tuple] = []

    def admission_fn(cam, frame):
        return True, {"admitted": True}, None

    def record_fn(cam, frame, is_train, is_holdout):
        recorded.append((frame.index, is_train, is_holdout))

    import scene.cameras as _cam_mod
    orig = _cam_mod.prepare_camera_for_render
    _cam_mod.prepare_camera_for_render = lambda cam, device="cuda": None
    try:
        result = StreamingScene.ingest_next_frame(
            fake_self, admission_fn=admission_fn, record_fn=record_fn
        )
    finally:
        _cam_mod.prepare_camera_for_render = orig

    assert result is not None
    cam, frame, is_train = result
    assert is_train is True
    assert recorded == [(0, True, False)]


def test_ingest_next_frame_pregate_skips_camera_and_visibility(monkeypatch):
    """A pregate_fn rejection must skip _frame_to_camera/visibility_fn/
    record_fn entirely (WS1's accepted observation-DB tradeoff) while still
    counting the rejection and returning the None-cam contract."""
    monkeypatch.setattr(
        "scene.cameras.prepare_camera_for_render", lambda cam, device="cuda": None
    )
    fake_self = _FakeIngestSelf([_FakeFrame(0)])

    calls: list[str] = []

    def _frame_to_camera_spy(frame):
        calls.append("frame_to_camera")
        raise AssertionError("must not be called for a pregate-rejected frame")

    fake_self._frame_to_camera = _frame_to_camera_spy

    def visibility_fn(cam, frame):
        calls.append("visibility")

    def admission_fn(cam, frame):
        calls.append("admission")
        return True, {"admitted": True}, None

    def record_fn(cam, frame, is_train, is_holdout):
        calls.append("record")

    def pregate_fn(frame):
        return {"admitted": 0, "reason": "no_motion"}

    result = StreamingScene.ingest_next_frame(
        fake_self, admission_fn=admission_fn, visibility_fn=visibility_fn,
        record_fn=record_fn, pregate_fn=pregate_fn,
    )

    assert result == (None, fake_self._frames[0], False)
    assert calls == []  # frame_to_camera/visibility/admission/record all skipped
    assert fake_self._keyframes_rejected == 1
    assert fake_self._frames[0]._streaming_admission_stats == {"admitted": 0, "reason": "no_motion"}


def test_ingest_next_frame_pregate_none_falls_through_to_admission(monkeypatch):
    """pregate_fn returning None must proceed exactly as if pregate_fn were
    absent: camera built, visibility_fn then admission_fn invoked."""
    monkeypatch.setattr(
        "scene.cameras.prepare_camera_for_render", lambda cam, device="cuda": None
    )
    fake_self = _FakeIngestSelf([_FakeFrame(0)])

    calls: list[str] = []

    def visibility_fn(cam, frame):
        calls.append("visibility")

    def admission_fn(cam, frame):
        calls.append("admission")
        return True, {"admitted": True}, None

    def pregate_fn(frame):
        calls.append("pregate")
        return None

    result = StreamingScene.ingest_next_frame(
        fake_self, admission_fn=admission_fn, visibility_fn=visibility_fn,
        pregate_fn=pregate_fn,
    )

    assert result is not None
    assert calls == ["pregate", "visibility", "admission"]


def test_ingest_next_frame_pregate_not_applied_to_holdout(monkeypatch):
    """A holdout frame must always get a real camera regardless of
    pregate_fn -- holdout selection is admission-independent and must
    never be skipped for motion reasons."""
    monkeypatch.setattr(
        "scene.cameras.prepare_camera_for_render", lambda cam, device="cuda": None
    )
    fake_self = _FakeIngestSelf([_FakeFrame(0)])
    fake_self._holdout_indices = {0}
    fake_self._guard_indices = set()

    calls: list[str] = []

    def pregate_fn(frame):
        calls.append("pregate_called")
        return {"admitted": 0, "reason": "no_motion"}  # would reject if consulted

    result = StreamingScene.ingest_next_frame(fake_self, pregate_fn=pregate_fn)

    assert calls == []  # pregate_fn must not even be called for a holdout frame
    assert result is not None
    cam, frame, is_train = result
    assert is_train is False
    assert fake_self._test_cameras == [cam]


def _make_fake_streaming_frame(index: int = 0) -> StreamingRGBDFrame:
    rgb = (np.random.rand(64, 64, 3) * 255).astype(np.uint8)
    depth = np.full((64, 64), 1000, dtype=np.uint16)  # depth_scale=1000 -> 1.0m
    return StreamingRGBDFrame(
        index=index, timestamp=float(index), rgb_path="", depth_path=None,
        c2w=np.eye(4, dtype=np.float32),
        fx=50.0, fy=50.0, cx=32.0, cy=32.0, width=64, height=64,
        depth_scale=1000.0,
        _rgb_np=rgb, _depth_np=depth,
    )


def test_record_frame_observation_populates_resolution_cache_for_train_frames():
    db = FrameObservationDB(device="cpu")
    cam = _FakeCam()
    frame = _make_fake_streaming_frame(index=5)
    record_frame_observation(
        db, cam=cam, visible_uids=torch.zeros(0, dtype=torch.int64), frame_id=5,
        submap_id=0, is_train=True, is_holdout=False, frame=frame, low_res_divisor=4,
    )
    rec = db.get_frame(5)
    assert set(rec.resolution_cache.keys()) == {"low", "high"}
    assert rec.resolution_cache["high"].rgb.shape == (64, 64, 3)
    assert rec.resolution_cache["low"].rgb.shape == (16, 16, 3)
    # depth is metres (converted via depth_scale before caching), never blended
    assert torch.allclose(rec.resolution_cache["high"].depth.cpu(), torch.ones(64, 64), atol=1e-4)


@pytest.mark.parametrize(
    ("depth_kwargs", "expected_depth"),
    [
        pytest.param({}, 2.0, id="default-nearest"),
        pytest.param({"depth_downsample": "valid_area"}, 3.0, id="valid-area"),
    ],
)
def test_legacy_cache_depth_downsample_reaches_the_bound_camera(
    depth_kwargs, expected_depth
):
    """The legacy cache must store its configured depth reduction.

    The two valid samples in the first 4x4 low-tier footprint are 2m and 4m.
    The default nearest cache exposes 2m after binding; validity-aware area
    reduction exposes their 3m mean to the actual training camera.
    """
    rgb = np.zeros((8, 16, 3), dtype=np.uint8)
    depth = np.zeros((8, 16), dtype=np.uint16)
    depth[0, 0] = 2000
    depth[1, 1] = 4000
    frame = StreamingRGBDFrame(
        index=11,
        timestamp=11.0,
        rgb_path="",
        depth_path=None,
        c2w=np.eye(4, dtype=np.float32),
        fx=80.0,
        fy=40.0,
        cx=7.5,
        cy=3.5,
        width=16,
        height=8,
        depth_scale=1000.0,
        _rgb_np=rgb,
        _depth_np=depth,
    )
    db = FrameObservationDB(device="cpu")
    record_frame_observation(
        db,
        cam=_FakeCam(),
        visible_uids=torch.zeros(0, dtype=torch.int64),
        frame_id=11,
        submap_id=0,
        is_train=True,
        is_holdout=False,
        frame=frame,
        low_res_divisor=4,
        device="cpu",
        **depth_kwargs,
    )

    cam = type("RenderCam", (), {})()
    cam.original_image = torch.zeros(3, 8, 16)
    cam.image_height, cam.image_width = 8, 16
    assert bind_resolution_tier(cam, db.get_frame(11), "low")
    assert tuple(cam._sensor_depth_cache.shape) == (1, 2, 4)
    assert float(cam._sensor_depth_cache[0, 0, 0]) == pytest.approx(expected_depth)
    assert float(db.get_frame(11).resolution_cache["high"].depth[0, 0]) == 2.0


def test_live_legacy_record_callback_forwards_configured_depth_downsample():
    """Normal admitted frames use the nested production callback, not bootstrap."""
    source = Path(__file__).resolve().parents[1] / "train_streaming.py"
    tree = ast.parse(source.read_text())
    training = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "streaming_training"
    )
    record = next(
        node for node in ast.walk(training)
        if isinstance(node, ast.FunctionDef) and node.name == "_record_fn"
    )
    call = next(
        node for node in ast.walk(record)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "record_frame_observation"
    )
    keyword = next(
        keyword for keyword in call.keywords
        if keyword.arg == "depth_downsample"
    )
    assert isinstance(keyword.value, ast.Name)
    assert keyword.value.id == "_multires_depth_downsample"


def test_valid_area_cache_skips_discarded_nearest_low_depth_resize(monkeypatch):
    """Validity-aware pooling derives low depth from native, once."""
    from eligsir.utils.rgbd_frames import CameraIntrinsics

    original = resolution_cache.resize_rgbd_frame
    divisors = []

    def tracked_resize(*args, **kwargs):
        divisors.append(kwargs["divisor"])
        return original(*args, **kwargs)

    monkeypatch.setattr(resolution_cache, "resize_rgbd_frame", tracked_resize)
    build_resolution_cache(
        np.zeros((8, 16, 3), dtype=np.uint8),
        np.ones((8, 16), dtype=np.float32),
        CameraIntrinsics(
            width=16, height=8, fx=80.0, fy=40.0,
            cx=7.5, cy=3.5, depth_scale=1.0,
        ),
        low_res_divisor=4,
        depth_downsample="valid_area",
        device="cpu",
    )

    assert divisors == [1]

def test_record_frame_observation_skips_resolution_cache_for_non_train_frames():
    db = FrameObservationDB(device="cpu")
    cam = _FakeCam()
    frame = _make_fake_streaming_frame(index=6)
    record_frame_observation(
        db, cam=cam, visible_uids=torch.zeros(0, dtype=torch.int64), frame_id=6,
        submap_id=0, is_train=False, is_holdout=True, frame=frame, low_res_divisor=4,
    )
    assert db.get_frame(6).resolution_cache == {}


# ---------------------------------------------------------------------------
# 2026-08-12 fix: depth-less streaming sources (RealityScan Mobile -- RGB +
# poses + sparse SfM cloud, NO per-frame depth; depth_path=None AND
# _depth_np=None AND _depth_bytes=None, see
# eligsir/io/sources/datasets/realityscan.py). Before this fix,
# resolution_cache stayed permanently empty for such frames -- confirmed via
# a real 30000-iteration run's own summary line
# (`bind stats: {'low_ok': 0, 'low_fail': 0, 'high_ok': 0, 'high_fail': 30000}`).
# ---------------------------------------------------------------------------


def _make_fake_depthless_streaming_frame(index: int = 0) -> StreamingRGBDFrame:
    rgb = (np.random.rand(64, 64, 3) * 255).astype(np.uint8)
    return StreamingRGBDFrame(
        index=index, timestamp=float(index), rgb_path="", depth_path=None,
        c2w=np.eye(4, dtype=np.float32),
        fx=50.0, fy=50.0, cx=32.0, cy=32.0, width=64, height=64,
        depth_scale=1000.0,
        _rgb_np=rgb, _depth_np=None,
    )


def test_record_frame_observation_populates_rgb_only_cache_for_depthless_train_frames():
    """The fix: a depth-less source's resolution_cache is no longer
    permanently empty -- it gets real rgb tiers (depth stays None)."""
    db = FrameObservationDB(device="cpu")
    cam = _FakeCam()
    frame = _make_fake_depthless_streaming_frame(index=7)
    record_frame_observation(
        db, cam=cam, visible_uids=torch.zeros(0, dtype=torch.int64), frame_id=7,
        submap_id=0, is_train=True, is_holdout=False, frame=frame, low_res_divisor=4,
    )
    rec = db.get_frame(7)
    assert set(rec.resolution_cache.keys()) == {"low", "high"}
    assert rec.resolution_cache["high"].rgb.shape == (64, 64, 3)
    assert rec.resolution_cache["low"].rgb.shape == (16, 16, 3)
    assert rec.resolution_cache["high"].depth is None
    assert rec.resolution_cache["low"].depth is None


def test_record_frame_observation_keeps_empty_cache_for_transient_depth_load_failure():
    """Advisor-review regression test: a DEPTH-HAVING source (depth_path is
    set, not structurally depth-less) whose depth file happens to be
    missing/corrupt for one frame must NOT get the new rgb-only-cache
    treatment -- that would let bind_resolution_tier rebind
    cam.image_height/width/fx/fy/cx/cy to a tier while _get_sensor_depth's
    own independent lazy-load later hands back native-resolution depth for
    the same camera, a resolution/intrinsics mismatch that never existed
    pre-fix. Only a STRUCTURALLY depth-less frame (depth_path AND
    _depth_np AND _depth_bytes all None) may take the rgb-only path."""
    db = FrameObservationDB(device="cpu")
    cam = _FakeCam()
    rgb = (np.random.rand(64, 64, 3) * 255).astype(np.uint8)
    frame = StreamingRGBDFrame(
        index=9, timestamp=9.0, rgb_path="", depth_path="/nonexistent/depth009.png",
        c2w=np.eye(4, dtype=np.float32),
        fx=50.0, fy=50.0, cx=32.0, cy=32.0, width=64, height=64,
        depth_scale=1000.0,
        _rgb_np=rgb, _depth_np=None,
    )
    record_frame_observation(
        db, cam=cam, visible_uids=torch.zeros(0, dtype=torch.int64), frame_id=9,
        submap_id=0, is_train=True, is_holdout=False, frame=frame, low_res_divisor=4,
        depthless_rgb_cache=True,  # even with the fix enabled...
    )
    # ...the cache must still be empty, since this source is not
    # structurally depth-less (only this one frame's depth failed to load).
    assert db.get_frame(9).resolution_cache == {}


def test_record_frame_observation_reconciles_native_rgb_against_declared_frame_size():
    """Advisor-review regression test: RealityScanFrameSource's
    --streaming_auto_downscale_max_pixels gate rewrites frame.width/height/
    fx/fy/cx/cy to a divided-down size while frame.rgb_path/_rgb_np still
    holds the NATIVE-resolution photo (eligsir/io/sources/datasets/
    realityscan.py's divisor block). Without reconciling the decoded
    image's actual size against frame.width/height before building the
    cache, the "high" tier's rgb pixel dimensions would come from the
    native size while its CacheHandle.intrinsics claims the declared
    (smaller) size -- bind_resolution_tier would then set
    cam.image_height/width from one and cam.fx/fy/cx/cy from the other."""
    db = FrameObservationDB(device="cpu")
    cam = _FakeCam()
    native_rgb = (np.random.rand(256, 256, 3) * 255).astype(np.uint8)  # native photo
    frame = StreamingRGBDFrame(
        index=10, timestamp=10.0, rgb_path="", depth_path=None,
        c2w=np.eye(4, dtype=np.float32),
        # Declared (already-divided-down) size + intrinsics, per the
        # RealityScan auto-downscale gate -- deliberately NOT 256x256.
        fx=12.5, fy=12.5, cx=32.0, cy=32.0, width=64, height=64,
        depth_scale=1000.0,
        _rgb_np=native_rgb, _depth_np=None,
    )
    record_frame_observation(
        db, cam=cam, visible_uids=torch.zeros(0, dtype=torch.int64), frame_id=10,
        submap_id=0, is_train=True, is_holdout=False, frame=frame, low_res_divisor=4,
    )
    rec = db.get_frame(10)
    high = rec.resolution_cache["high"]
    low = rec.resolution_cache["low"]
    # "high" tier must match the DECLARED (64x64) size, not the native
    # (256x256) rgb array -- and its intrinsics must describe that same size.
    assert high.rgb.shape[:2] == (64, 64)
    assert high.intrinsics.width == 64 and high.intrinsics.height == 64
    assert low.rgb.shape[:2] == (16, 16)
    assert low.intrinsics.width == 16 and low.intrinsics.height == 16


def test_record_frame_observation_depthless_opt_out_restores_old_empty_cache_behavior():
    """--no-streaming_multires_depthless_rgb_cache (depthless_rgb_cache=False)
    must reproduce the exact pre-fix behavior: resolution_cache stays empty."""
    db = FrameObservationDB(device="cpu")
    cam = _FakeCam()
    frame = _make_fake_depthless_streaming_frame(index=8)
    record_frame_observation(
        db, cam=cam, visible_uids=torch.zeros(0, dtype=torch.int64), frame_id=8,
        submap_id=0, is_train=True, is_holdout=False, frame=frame, low_res_divisor=4,
        depthless_rgb_cache=False,
    )
    assert db.get_frame(8).resolution_cache == {}


def test_bind_resolution_tier_rebinds_rgb_but_leaves_depth_cache_untouched_for_depthless_source():
    """A depth-less tier's CacheHandle.depth is None -- bind_resolution_tier
    must rebind rgb/image dims/intrinsics as usual but must NOT clobber
    cam._sensor_depth_cache, since depth-supervision loss code relies on
    that attribute's pre-existing None/False sentinel state to know this
    source has no depth at all (not just no depth for this tier)."""
    rgb = (np.random.rand(64, 64, 3) * 255).astype(np.uint8)
    from eligsir.utils.rgbd_frames import CameraIntrinsics
    intr = CameraIntrinsics(width=64, height=64, fx=50.0, fy=50.0, cx=32.0, cy=32.0, depth_scale=1.0)
    cache = build_resolution_cache(rgb, None, intr, low_res_divisor=4, device="cpu")
    record = FrameRecord(
        frame_id=0, pose_c2w=np.eye(4, dtype=np.float32), timestamp=0.0, submap_id=0,
        buffer_stage=BufferStage.MAPPING_KEYFRAME, visible_gaussian_uids=torch.zeros(0, dtype=torch.int64),
        resolution_cache=cache,
    )

    class _RenderCam:
        pass

    cam = _RenderCam()
    cam.original_image = torch.zeros(3, 64, 64)
    cam.image_height = 64
    cam.image_width = 64
    _sentinel = object()
    cam._sensor_depth_cache = _sentinel

    bound = bind_resolution_tier(cam, record, "low")
    assert bound is True
    assert cam.image_height == 16 and cam.image_width == 16
    assert cam.original_image.shape == (3, 16, 16)
    # rgb/intrinsics rebind happened, but depth cache must be untouched.
    assert cam._sensor_depth_cache is _sentinel
    assert cam.fx == pytest.approx(50.0 / 4)


def test_bind_resolution_tier_rebinds_low_tier_tensors_onto_camera():
    rgb = (np.random.rand(64, 64, 3) * 255).astype(np.uint8)
    depth = np.full((64, 64), 1000, dtype=np.uint16)
    from eligsir.utils.rgbd_frames import CameraIntrinsics
    intr = CameraIntrinsics(width=64, height=64, fx=50.0, fy=50.0, cx=32.0, cy=32.0, depth_scale=1.0)
    depth_m = depth.astype(np.float32) / 1000.0
    cache = build_resolution_cache(rgb, depth_m, intr, low_res_divisor=4, device="cpu")
    record = FrameRecord(
        frame_id=0, pose_c2w=np.eye(4, dtype=np.float32), timestamp=0.0, submap_id=0,
        buffer_stage=BufferStage.MAPPING_KEYFRAME, visible_gaussian_uids=torch.zeros(0, dtype=torch.int64),
        resolution_cache=cache,
    )

    class _RenderCam:
        pass

    cam = _RenderCam()
    cam.original_image = torch.zeros(3, 64, 64)
    cam.image_height = 64
    cam.image_width = 64

    bound = bind_resolution_tier(cam, record, "low")
    assert bound is True
    assert cam.image_height == 16 and cam.image_width == 16
    assert cam.original_image.shape == (3, 16, 16)
    assert cam._sensor_depth_cache.shape == (1, 16, 16)
    assert torch.allclose(cam._sensor_depth_cache, torch.ones(1, 16, 16), atol=1e-4)
    # Regression test for whole-branch-review finding C-1: the low tier's
    # focal length/principal point must scale down with the image, not stay
    # at the native (64x64) values -- otherwise the renderer effectively
    # zooms into a crop rather than rendering the tier's own downsampled FoV.
    assert cam.fx == pytest.approx(50.0 / 4)
    assert cam.fy == pytest.approx(50.0 / 4)
    assert cam.cx == pytest.approx((32.0 + 0.5) / 4 - 0.5)
    assert cam.cy == pytest.approx((32.0 + 0.5) / 4 - 0.5)

    bound_high = bind_resolution_tier(cam, record, "high")
    assert bound_high is True
    assert cam.image_height == 64 and cam.image_width == 64
    assert cam.fx == pytest.approx(50.0)
    assert cam.cx == pytest.approx(32.0)


def test_bind_resolution_tier_returns_false_when_no_cache_available():
    record = FrameRecord(
        frame_id=0, pose_c2w=np.eye(4, dtype=np.float32), timestamp=0.0, submap_id=0,
        buffer_stage=BufferStage.MAPPING_KEYFRAME, visible_gaussian_uids=torch.zeros(0, dtype=torch.int64),
        resolution_cache={},
    )

    class _RenderCam:
        pass

    cam = _RenderCam()
    assert bind_resolution_tier(cam, record, "low") is False
    assert bind_resolution_tier(None, record, "low") is False


def test_initialize_from_frames_calls_record_fn_for_every_bootstrap_frame():
    """Regression test for the 2026-07-26 bug: bootstrap frames bypass
    ingest_next_frame entirely, so without wiring record_fn into
    initialize_from_frames too, every bootstrap camera's observation-DB
    record was missing -- confirmed via real-run validation to cause a
    100% bind_resolution_tier miss rate whenever sampling stayed within
    the bootstrap window (which streaming_scene deliberately keeps "hot"
    early in training)."""
    import types

    from scene.streaming_scene import StreamingScene

    frames = [
        StreamingRGBDFrame(
            index=i, timestamp=float(i), rgb_path="", depth_path=None,
            c2w=np.eye(4, dtype=np.float32), fx=500.0, fy=500.0, cx=32.0, cy=32.0,
            width=64, height=64,
        )
        for i in range(3)
    ]

    scene = object.__new__(StreamingScene)
    # init_scale_mode is a BARE read at scene/streaming_scene.py:971 (its
    # getattr fallback was removed by #394's commit 4e139d2), so this real
    # StreamingScene must declare it or initialize_from_frames raises
    # AttributeError. The value must be the PARSER default "knn"
    # (arguments/__init__.py:68) -- NOT the removed literal "fixed", which
    # would exercise a bootstrap scale-init branch production never takes.
    scene.args = types.SimpleNamespace(init_scale_mode="knn")
    from eligsir.mapping.stores.keyframe_store import KeyframeStore
    scene._keyframe_store = KeyframeStore()
    scene.train_cameras = []
    from eligsir.mapping.stores.replay_view_store import ReplayViewStore
    scene._replay_view_store = ReplayViewStore()
    from eligsir.mapping.stores.occupancy_index import OccupancyIndex
    scene._occupancy_index = OccupancyIndex(voxel_size=0.02)
    scene._replay_buffer = []
    scene._frame_count = len(frames)
    scene._get_source_frame = lambda i: frames[i]
    scene._frame_to_camera = lambda frame: _FakeCam()
    scene._estimate_cameras_extent = lambda fr: 1.0

    class _FakePcd:
        points = np.zeros((0, 3), dtype=np.float32)

    scene._build_pcd_from_frames = lambda fr: _FakePcd()
    scene._export_initial_seed_pointcloud = lambda pcd: None

    class _FakeGaussians:
        def create_from_pcd(self, *a, **kw):
            pass

        @property
        def get_xyz(self):
            return torch.zeros(0, 3)

    scene.gaussians = _FakeGaussians()
    scene.args.streaming_gpu_occupancy = False

    import scene.cameras as cameras_module

    def _fake_prepare(cam, device="cuda"):
        pass

    orig_prepare = cameras_module.prepare_camera_for_render
    cameras_module.prepare_camera_for_render = _fake_prepare
    try:
        seen = []

        def record_fn(cam, frame, is_train, is_holdout):
            seen.append((frame.index, is_train, is_holdout))

        scene.initialize_from_frames(3, record_fn=record_fn)
    finally:
        cameras_module.prepare_camera_for_render = orig_prepare

    assert seen == [(0, True, False), (1, True, False), (2, True, False)]


# ---------------------------------------------------------------------------
# Issue #464: resolution-cache device derivation + swallowed-failure visibility
#
# `record_frame_observation` used to call `build_resolution_cache(...,
# device="cuda")` with the device HARDCODED. On a GPU-masked/CPU host that
# raises `RuntimeError: No HIP GPUs are available`, which the surrounding broad
# `except Exception` swallows -- leaving `resolution_cache = {}` for that frame
# and (because the warning was ONE-SHOT) for every subsequent frame with no
# further diagnostic trail. The three cache tests above are the direct victims;
# these cover the fix itself.
#
# NOTE the counters are module-global process state (matching this module's
# existing `_warned_*` function-attribute convention), so every test that
# asserts on a count MUST reset first -- otherwise it silently picks up
# occurrences from earlier tests in the same process. `-p no:randomly` only
# masks that, it does not fix it.
# ---------------------------------------------------------------------------

import pytest

from train_streaming import (
    _resolution_cache_device,
    _should_print_multires_warning,
    multires_cache_report_block,
    reset_multires_cache_stats,
)


@pytest.fixture
def fresh_multires_stats():
    reset_multires_cache_stats()
    yield
    reset_multires_cache_stats()


def test_resolution_cache_device_prefers_explicit_over_autodetect():
    assert _resolution_cache_device("cpu") == "cpu"
    assert _resolution_cache_device("cuda") == "cuda"


def test_resolution_cache_device_autodetects_and_never_raises():
    """The whole point of #464: the derivation must resolve to something usable
    on ANY host rather than asserting a GPU exists. On a GPU host this is
    "cuda" (production behavior unchanged); on a masked/CPU host it is "cpu"."""
    resolved = _resolution_cache_device(None)
    assert resolved in {"cuda", "cpu"}
    assert resolved == ("cuda" if torch.cuda.is_available() else "cpu")


def test_record_frame_observation_honors_explicit_cpu_device(fresh_multires_stats):
    db = FrameObservationDB(device="cpu")
    frame = _make_fake_streaming_frame(index=41)
    record_frame_observation(
        db, cam=_FakeCam(), visible_uids=torch.zeros(0, dtype=torch.int64), frame_id=41,
        submap_id=0, is_train=True, is_holdout=False, frame=frame, low_res_divisor=4,
        device="cpu",
    )
    rec = db.get_frame(41)
    assert set(rec.resolution_cache.keys()) == {"low", "high"}
    assert rec.resolution_cache["high"].rgb.device.type == "cpu"
    assert multires_cache_report_block()["device"] == "cpu"


def test_multires_cache_report_block_counts_successes(fresh_multires_stats):
    db = FrameObservationDB(device="cpu")
    for i in (50, 51):
        record_frame_observation(
            db, cam=_FakeCam(), visible_uids=torch.zeros(0, dtype=torch.int64), frame_id=i,
            submap_id=0, is_train=True, is_holdout=False,
            frame=_make_fake_streaming_frame(index=i), low_res_divisor=4,
        )
    block = multires_cache_report_block()
    assert block["frames_attempted"] == 2
    assert block["frames_cached"] == 2
    assert block["frames_failed"] == 0
    assert block["all_failed"] is False
    assert block["cache_hit_fraction"] == 1.0
    # `failures` only appears when something actually failed.
    assert "failures" not in block


def test_multires_cache_failures_are_counted_not_one_shot(fresh_multires_stats, monkeypatch):
    """The #464 regression guard: a build that fails on EVERY frame must be
    fully counted and reported, not represented by a single print.

    The run must also stay ALIVE -- the broad `except` is a deliberate
    live-camera-path property (CLAUDE.md hard constraint) and this fix does not
    change it; the frame is still recorded, just with an empty cache.
    """
    import eligsir.mapping.resolution_cache as rc_mod

    def _boom(*a, **kw):
        raise RuntimeError("No HIP GPUs are available")

    monkeypatch.setattr(rc_mod, "build_resolution_cache", _boom)

    db = FrameObservationDB(device="cpu")
    for i in (60, 61, 62):
        record_frame_observation(
            db, cam=_FakeCam(), visible_uids=torch.zeros(0, dtype=torch.int64), frame_id=i,
            submap_id=0, is_train=True, is_holdout=False,
            frame=_make_fake_streaming_frame(index=i), low_res_divisor=4, device="cpu",
        )
        # run-alive property: the frame is still recorded despite the failure
        assert db.get_frame(i) is not None
        assert db.get_frame(i).resolution_cache == {}

    block = multires_cache_report_block()
    assert block["frames_attempted"] == 3
    assert block["frames_failed"] == 3, "one-shot accounting would under-report this"
    assert block["frames_cached"] == 0
    assert block["all_failed"] is True
    assert block["cache_hit_fraction"] == 0.0
    failures = block["failures"]
    assert failures["n_failed"] == 3
    assert failures["error_types"] == {"RuntimeError": 3}
    assert "No HIP GPUs are available" in failures["example_error"]
    assert failures["first_failure_frame_id"] == 60
    assert failures["last_failure_frame_id"] == 62


def test_multires_cache_invariant_holds(fresh_multires_stats):
    """frames_cached + frames_failed + frames_empty_no_depth == frames_attempted
    (the #437 `n_*_metrics` accounting convention this mirrors)."""
    db = FrameObservationDB(device="cpu")
    # one healthy frame
    record_frame_observation(
        db, cam=_FakeCam(), visible_uids=torch.zeros(0, dtype=torch.int64), frame_id=70,
        submap_id=0, is_train=True, is_holdout=False,
        frame=_make_fake_streaming_frame(index=70), low_res_divisor=4, device="cpu",
    )
    # one frame whose depth fails to load on a depth-HAVING source -> empty cache
    bad = _make_fake_streaming_frame(index=71)
    bad._depth_np = None
    bad.depth_path = "/nonexistent/depth.png"
    record_frame_observation(
        db, cam=_FakeCam(), visible_uids=torch.zeros(0, dtype=torch.int64), frame_id=71,
        submap_id=0, is_train=True, is_holdout=False, frame=bad, low_res_divisor=4,
        device="cpu",
    )
    b = multires_cache_report_block()
    assert b["frames_cached"] + b["frames_failed"] + b["frames_empty_no_depth"] == b["frames_attempted"]
    assert b["frames_attempted"] == 2


def test_non_train_frames_are_not_counted_as_attempts(fresh_multires_stats):
    db = FrameObservationDB(device="cpu")
    record_frame_observation(
        db, cam=_FakeCam(), visible_uids=torch.zeros(0, dtype=torch.int64), frame_id=80,
        submap_id=0, is_train=False, is_holdout=False,
        frame=_make_fake_streaming_frame(index=80), low_res_divisor=4, device="cpu",
    )
    assert multires_cache_report_block()["frames_attempted"] == 0


def test_warning_cadence_is_escalating_not_one_shot_and_not_every_frame():
    printed = [n for n in range(1, 1001) if _should_print_multires_warning(n)]
    assert printed == [1, 10, 100, 1000]
    assert _should_print_multires_warning(0) is False


def test_multires_cache_block_reaches_report_json_on_disk():
    """The deliverable is visibility in report.json, not in a returned dict.

    Mirrors tests/test_report_render_failure_visibility.py::
    test_fields_reach_report_json_on_disk (the #437 convention this follows).
    Asserts TOP-LEVEL placement specifically: the block is deliberately NOT
    written inside write_post_training_report's big `geometry` try/except,
    because a block whose whole job is to report a swallowed failure must not
    itself be erasable by a *different* swallowed failure.
    """
    import json
    import os
    import sys
    import tempfile
    from unittest.mock import MagicMock

    _had = "lpipsPyTorch" in sys.modules
    _orig = sys.modules.get("lpipsPyTorch")
    sys.modules["lpipsPyTorch"] = MagicMock()
    try:
        from eligsir.utils.comparison_report import write_post_training_report

        block = {
            "frames_attempted": 3, "frames_cached": 0, "frames_failed": 3,
            "frames_empty_no_depth": 0, "device": "cpu", "all_failed": True,
            "cache_hit_fraction": 0.0,
            "failures": {
                "n_failed": 3,
                "error_types": {"RuntimeError": 3},
                "example_error": "RuntimeError: No HIP GPUs are available",
                "first_failure_frame_id": 60,
                "last_failure_frame_id": 62,
            },
        }

        class _FakeGaussian:
            def get_xyz(self):
                return torch.zeros(0, 3)

        with tempfile.TemporaryDirectory() as tmpdir:
            summary = write_post_training_report(
                model_path=tmpdir, iteration=5000, gaussians=_FakeGaussian(),
                train_cams=[], test_cams=[], render_fn=None,
                pipe=object(), background=None, multires_cache=block,
            )
            with open(os.path.join(summary["output_dir"], "report.json")) as f:
                on_disk = json.load(f)
    finally:
        if _had:
            sys.modules["lpipsPyTorch"] = _orig
        else:
            sys.modules.pop("lpipsPyTorch", None)

    assert on_disk["multires_cache"]["all_failed"] is True
    assert on_disk["multires_cache"]["frames_failed"] == 3
    assert on_disk["multires_cache"]["failures"]["error_types"] == {"RuntimeError": 3}
    assert summary["multires_cache"]["device"] == "cpu"


def test_multires_cache_key_absent_when_not_passed():
    """Absence must stay meaningful: it signals "report predates the fix, or is
    from an offline train.py run", never "the cache was healthy"."""
    import json
    import os
    import sys
    import tempfile
    from unittest.mock import MagicMock

    _had = "lpipsPyTorch" in sys.modules
    _orig = sys.modules.get("lpipsPyTorch")
    sys.modules["lpipsPyTorch"] = MagicMock()
    try:
        from eligsir.utils.comparison_report import write_post_training_report

        class _FakeGaussian:
            def get_xyz(self):
                return torch.zeros(0, 3)

        with tempfile.TemporaryDirectory() as tmpdir:
            summary = write_post_training_report(
                model_path=tmpdir, iteration=5000, gaussians=_FakeGaussian(),
                train_cams=[], test_cams=[], render_fn=None,
                pipe=object(), background=None,
            )
            with open(os.path.join(summary["output_dir"], "report.json")) as f:
                on_disk = json.load(f)
    finally:
        if _had:
            sys.modules["lpipsPyTorch"] = _orig
        else:
            sys.modules.pop("lpipsPyTorch", None)

    assert "multires_cache" not in on_disk
    assert "multires_cache" not in summary
