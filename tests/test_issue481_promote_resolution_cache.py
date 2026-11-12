"""Regression tests for issue #481(b): a promoted frame's multires
``resolution_cache`` is never built.

``record_frame_observation`` only builds a ``resolution_cache`` for
``is_train`` frames, so a frame first seen as a REJECTED candidate is recorded
with an empty one (``resolution_cache_absent_reason == "not_train_frame"``).
``StreamingScene.promote_from_cache`` later splices exactly such a frame back
into ``train_cameras`` -- it becomes a real training camera, but nothing ever
built its cache, so ``bind_resolution_tier`` fails ``cache_empty`` for it for
the rest of the run and every depth-supervision read for that camera falls
through to ``depth_helpers._get_sensor_depth``'s lazy path.

The fix is a second cache producer that does NOT go through
``FrameObservationDB.add_frame``: ``add_frame`` re-appends the record's
visible-uid row to the incremental CSR buffer and replaces
``visible_gaussian_uids`` wholesale, so re-recording a promoted frame would
double-count its observations and destroy the visibility captured at ingest.
``attach_resolution_cache`` (narrow in-place mutator, ``set_buffer_stage``
idiom) + ``promote_from_cache(..., resolution_cache_fn=...)`` are that
producer. The hazard `add_frame` poses is pinned here too, so a future change
cannot quietly "simplify" this back into a re-record.
"""
from __future__ import annotations

from collections import OrderedDict
from types import SimpleNamespace

import numpy as np
import pytest

# Same self-contained guard as tests/test_issue384_depth_loss_live_guard.py:
# this module needs the real torch and the modules under test import it
# transitively, but a module-scope `import torch` would put this whole file in
# tests/conftest.py's collect_ignore list in the torch-free CPU smoke image --
# i.e. silently invisible, which is exactly what issue #536 exists to prevent.
torch = pytest.importorskip("torch")

from scene.streaming_scene import StreamingScene  # noqa: E402
from eligsir.mapping.observation_db import (  # noqa: E402
    BufferStage,
    FrameObservationDB,
    FrameRecord,
)


class _FakeCam:
    def __init__(self, index: int, frame=None) -> None:
        self.image_name = f"frame_{index}"
        self._streaming_frame = frame
        self.original_image = torch.zeros(3, 2, 2)


class _FakeFrame:
    def __init__(self, index: int) -> None:
        self.index = index
        self._rgb_np = None
        self._depth_np = None
        self._rgb_bytes = None
        self._depth_bytes = None


def _index_of(cam) -> int:
    return int(cam.image_name.split("_")[1])


class _Scene:
    """Minimal StreamingScene stand-in exposing only the promotion surface."""

    def __init__(self, db=None) -> None:
        self.args = SimpleNamespace(
            streaming_keyframe_window=8,
            streaming_offload_camera_images=False,
            streaming_inactive_cache_size=64,
            streaming_icecold_cache_size=64,
        )
        self.train_cameras: list = []
        self._inactive_frame_cache: "OrderedDict[int, object]" = OrderedDict()
        self._icecold_frame_cache: "OrderedDict[int, object]" = OrderedDict()
        self._active_cam_by_idx: dict = {}
        self._active_source_indices: set = set()
        self.observation_db = db

    def _frame_to_camera(self, frame, load_image=True):
        return _FakeCam(frame.index, frame=frame)

    promote_from_cache = StreamingScene.promote_from_cache
    _attach_promoted_resolution_cache = StreamingScene._attach_promoted_resolution_cache
    _predecode_frames_parallel = StreamingScene._predecode_frames_parallel
    _enforce_hot_window = StreamingScene._enforce_hot_window
    _offload_camera_hot_window = StreamingScene._offload_camera_hot_window
    _mark_buffer_stage = StreamingScene._mark_buffer_stage


def _rejected_record(frame_id: int, uids=(1, 2, 3)) -> FrameRecord:
    """A FrameRecord exactly as record_frame_observation writes one for a
    REJECTED candidate: no resolution_cache, reason "not_train_frame"."""
    return FrameRecord(
        frame_id=frame_id,
        pose_c2w=np.eye(4, dtype=np.float32),
        timestamp=float(frame_id),
        submap_id=0,
        buffer_stage=BufferStage.EVICTED_REDUNDANT,
        visible_gaussian_uids=torch.tensor(list(uids), dtype=torch.int64),
        resolution_cache={},
        resolution_cache_absent_reason="not_train_frame",
    )


def _db_with_rejected(frame_id: int = 99, uids=(1, 2, 3)) -> FrameObservationDB:
    db = FrameObservationDB(device="cpu")
    db.add_frame(_rejected_record(frame_id, uids))
    return db


# --------------------------------------------------------------------------
# FrameObservationDB.attach_resolution_cache
# --------------------------------------------------------------------------

def test_attach_resolution_cache_populates_record_and_clears_reason():
    db = _db_with_rejected()
    assert db.get_frame(99).resolution_cache == {}
    assert db.get_frame(99).resolution_cache_absent_reason == "not_train_frame"

    assert db.attach_resolution_cache(99, {"low": object(), "high": object()}) is True

    rec = db.get_frame(99)
    assert set(rec.resolution_cache) == {"low", "high"}
    assert rec.resolution_cache_absent_reason is None
    assert db.resolution_cache_occupancy == 1


def test_attach_resolution_cache_preserves_observation_history():
    """The whole reason this is a mutator and not a second add_frame."""
    db = _db_with_rejected(uids=(4, 5))
    db.attach_resolution_cache(99, {"low": object()})
    assert db.get_frame(99).visible_gaussian_uids.tolist() == [4, 5]
    assert db.get_frame(99).buffer_stage is BufferStage.EVICTED_REDUNDANT


def test_re_adding_a_frame_would_destroy_visibility():
    """Pins the hazard attach_resolution_cache exists to avoid -- do not
    'simplify' the promotion path back into a second add_frame() call."""
    db = _db_with_rejected(uids=(4, 5))
    rec = _rejected_record(99, uids=())
    rec.visible_gaussian_uids = torch.zeros(0, dtype=torch.int64)
    db.add_frame(rec)
    assert db.get_frame(99).visible_gaussian_uids.numel() == 0


def test_attach_resolution_cache_is_a_noop_for_unknown_or_empty():
    db = _db_with_rejected()
    assert db.attach_resolution_cache(12345, {"low": object()}) is False
    assert db.attach_resolution_cache(99, {}) is False
    assert db.get_frame(99).resolution_cache == {}
    assert db.get_frame(99).resolution_cache_absent_reason == "not_train_frame"


def test_attach_resolution_cache_clears_a_stale_evicted_mark():
    db = _db_with_rejected()
    db.get_frame(99).resolution_cache_evicted = True
    db.attach_resolution_cache(99, {"low": object()})
    assert db.get_frame(99).resolution_cache_evicted is False


# --------------------------------------------------------------------------
# StreamingScene.promote_from_cache
# --------------------------------------------------------------------------

def test_promotion_builds_the_missing_resolution_cache():
    db = _db_with_rejected()
    scene = _Scene(db)
    scene._inactive_frame_cache[99] = _FakeFrame(99)

    calls: list = []

    def _builder(frame):
        calls.append(frame.index)
        return {"low": object(), "high": object()}

    assert scene.promote_from_cache({99}, resolution_cache_fn=_builder) == 1
    assert calls == [99]
    rec = db.get_frame(99)
    assert set(rec.resolution_cache) == {"low", "high"}
    assert rec.resolution_cache_absent_reason is None
    # ... and the promotion itself still did everything it did before.
    assert 99 in scene._active_source_indices
    assert rec.buffer_stage is BufferStage.MAPPING_KEYFRAME


def test_promotion_without_a_builder_is_pre_fix_behaviour():
    """The parameter is opt-in: omitting it must change nothing at all."""
    db = _db_with_rejected()
    scene = _Scene(db)
    scene._icecold_frame_cache[99] = _FakeFrame(99)

    assert scene.promote_from_cache({99}) == 1
    assert db.get_frame(99).resolution_cache == {}
    assert db.get_frame(99).resolution_cache_absent_reason == "not_train_frame"


def test_promotion_does_not_rebuild_an_already_resident_cache():
    db = _db_with_rejected()
    resident = {"low": object()}
    db.attach_resolution_cache(99, resident)
    scene = _Scene(db)
    scene._inactive_frame_cache[99] = _FakeFrame(99)

    calls: list = []
    assert scene.promote_from_cache(
        {99}, resolution_cache_fn=lambda f: calls.append(f.index) or {"low": object()}
    ) == 1
    assert calls == [], "a resident cache must not be rebuilt on re-promotion"
    assert db.get_frame(99).resolution_cache["low"] is resident["low"]


def test_a_failing_builder_never_blocks_the_promotion():
    """Live-camera hard constraint: a cache/telemetry path must not take a run
    down (nor silently drop the camera it was promoting)."""
    db = _db_with_rejected()
    scene = _Scene(db)
    scene._inactive_frame_cache[99] = _FakeFrame(99)

    def _boom(frame):
        raise RuntimeError("decode exploded")

    assert scene.promote_from_cache({99}, resolution_cache_fn=_boom) == 1
    assert 99 in scene._active_source_indices
    assert db.get_frame(99).resolution_cache == {}


def test_an_empty_builder_result_leaves_the_reason_intact():
    db = _db_with_rejected()
    scene = _Scene(db)
    scene._inactive_frame_cache[99] = _FakeFrame(99)
    assert scene.promote_from_cache({99}, resolution_cache_fn=lambda f: {}) == 1
    assert db.get_frame(99).resolution_cache_absent_reason == "not_train_frame"


def test_promotion_with_a_builder_but_no_db_does_not_crash():
    scene = _Scene(db=None)
    scene._inactive_frame_cache[99] = _FakeFrame(99)
    assert scene.promote_from_cache({99}, resolution_cache_fn=lambda f: {"low": 1}) == 1
    assert 99 in scene._active_source_indices
