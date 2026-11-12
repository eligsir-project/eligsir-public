"""D2 (Wave 2 consolidation) regression coverage: ingest_next_frame must
build a candidate camera lazily (load_image=False) for any frame that still
has to pass admission_fn, and must only materialize its RGB
(cam.materialize_pending_image()) once that frame is actually admitted.
Rejected candidates must never have their image materialized -- that is
the whole point of the deferral (avoid the H2D upload cost for a frame
that gets thrown away).

This test does not touch CUDA or the real scene.cameras.Camera (both
require a GPU in this environment); it fakes StreamingScene._frame_to_camera
with a lightweight recorder so it can assert on load_image= and
materialize_pending_image() call counts directly -- the actual Camera/
loadCam lazy-loading mechanics are covered by
tests/test_camera_lazy_image_gpu.py (GPU-gated).
"""
from __future__ import annotations

from types import SimpleNamespace

from scene.streaming_scene import StreamingScene


class _FakeFrame:
    def __init__(self, index: int) -> None:
        self.index = index
        self._odom_valid = True
        self.timestamp = float(index)


class _FakeCam:
    """Minimal cam recording whether/when its image was materialized."""

    def __init__(self, index: int, load_image: bool) -> None:
        self.R = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        self.T = [0.0, 0.0, 0.0]
        self.image_name = f"frame_{index}"
        self.width = 64
        self.height = 64
        self.fx = self.fy = 50.0
        self.cx = self.cy = 32.0
        self.c2w = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        self.load_image = load_image
        self.materialized = bool(load_image)
        self._materialize_calls = 0

    def materialize_pending_image(self):
        self._materialize_calls += 1
        self.materialized = True


class _FakeIngestSelf:
    """Mirrors tests/test_streaming_scene_dense_trajectory.py's harness,
    with a _frame_to_camera spy that records the `load_image` argument it
    was called with, per frame."""

    def __init__(self, frames, admission_decisions):
        self.args = SimpleNamespace(
            streaming_eval_hold=0,
            streaming_offload_camera_images=False,
            streaming_replay_buffer=512,
            streaming_frame_admission="hybrid_keyframe",
            streaming_global_reservoir_stride=0,
            streaming_keyframe_window=8,
            streaming_track_scene_complexity=False,
            streaming_inactive_cache_size=0,
        )
        self._active_source_indices = set()
        self._active_cam_by_idx = {}
        self._last_admitted_source_idx = None
        self._frames = frames
        self._admission_decisions = admission_decisions
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
        self.load_image_calls: list[tuple[int, bool]] = []

    def has_next_frame(self):
        return self._source_idx < len(self._frames)

    def _get_source_frame(self, index):
        if index >= len(self._frames):
            return None
        return self._frames[index]

    def _frame_to_camera(self, frame, load_image=True):
        self.load_image_calls.append((frame.index, load_image))
        return _FakeCam(frame.index, load_image)

    def accumulate_observed_surface(self, frame):
        pass

    def _cache_inactive_frame(self, frame):
        pass

    def _release_source_frames_upto_cursor(self):
        # WS-D1's live-frame-buffer eviction hook; irrelevant to this
        # test's lazy-camera-image control flow, no-op like the other
        # bookkeeping stubs above.
        pass

    # Issue #396: the hot-window eviction cursor, extracted out of
    # ingest_next_frame's inline body into its own method.
    _offload_camera_hot_window = StreamingScene._offload_camera_hot_window
    _offload_evicted_camera = StreamingScene._offload_evicted_camera
    # Issue #1115: bound from the REAL StreamingScene -- #1084 extracted
    # `ingest_next_frame`'s inline holdout/guard computation into this
    # predicate verbatim; the lazy-image control flow under test branches on
    # its result (holdout/guard frames stay eager), so it must be the
    # production rule.
    holdout_status = StreamingScene.holdout_status

    def _append_dense_trajectory(self, cam, frame, is_train, is_holdout):
        pass

    def flush_deferred_extent_update(self):
        pass

    def update_cameras_extent(self):
        pass


def _run(admission_decisions, monkeypatch):
    # ingest_next_frame calls scene.cameras.prepare_camera_for_render(cam,
    # device="cuda") on every admitted frame; that needs a real GPU-shaped
    # Camera. Stub it out (mirrors tests/test_streaming_scene_dense_
    # trajectory.py's identical monkeypatch) -- irrelevant to what this
    # test is checking (the load_image=/materialize_pending_image()
    # control flow, not gsplat tensor caching).
    monkeypatch.setattr(
        "scene.cameras.prepare_camera_for_render", lambda cam, device="cuda": None
    )
    frames = [_FakeFrame(i) for i in range(len(admission_decisions))]
    fake_self = _FakeIngestSelf(frames, admission_decisions)

    admitted_cams = []
    rejected_cams = []

    def admission_fn(cam, frame):
        admit = admission_decisions[frame.index]
        if admit:
            admitted_cams.append(cam)
        else:
            rejected_cams.append(cam)
        return admit, {"reason": "admit" if admit else "reject"}, None

    results = []
    for _ in range(len(admission_decisions)):
        results.append(
            StreamingScene.ingest_next_frame(fake_self, admission_fn=admission_fn)
        )
    return fake_self, results, admitted_cams, rejected_cams


def test_candidate_camera_built_lazily_for_admission_path(monkeypatch):
    """Every non-holdout/non-guard candidate must be built with
    load_image=False -- the whole point of D2 is that admission is decided
    before the expensive image upload happens."""
    fake_self, _, _, _ = _run([True, False, True, False, True], monkeypatch)
    assert fake_self.load_image_calls == [
        (0, False), (1, False), (2, False), (3, False), (4, False)
    ]


def test_rejected_candidate_image_never_materialized(monkeypatch):
    fake_self, _, admitted_cams, rejected_cams = _run([True, False, True, False], monkeypatch)
    for cam in rejected_cams:
        assert cam._materialize_calls == 0
        assert cam.materialized is False
    for cam in admitted_cams:
        assert cam._materialize_calls == 1
        assert cam.materialized is True


def test_admitted_camera_materialized_exactly_once(monkeypatch):
    fake_self, _, admitted_cams, _ = _run([True, True, True], monkeypatch)
    assert len(admitted_cams) == 3
    for cam in admitted_cams:
        assert cam._materialize_calls == 1


def test_admission_set_unaffected_by_lazy_camera_build(monkeypatch):
    """The admit/reject decision sequence itself (what D3's admission_fn
    returns) must be identical to what it would be fed a plain flag --
    this test does not exercise a real admission_fn, but pins that
    ingest_next_frame's control flow (guard clauses, holdout handling,
    train_cameras/_keyframes_admitted bookkeeping) is untouched by the
    load_image= plumbing: the same decisions list drives the same
    admitted/rejected partition regardless."""
    decisions = [True, False, False, True, True, False]
    fake_self, results, admitted_cams, rejected_cams = _run(decisions, monkeypatch)
    assert len(admitted_cams) == sum(decisions)
    assert len(rejected_cams) == len(decisions) - sum(decisions)
    assert fake_self._keyframes_admitted == sum(decisions)
    assert fake_self._keyframes_rejected == len(decisions) - sum(decisions)
    assert len(fake_self.train_cameras) == sum(decisions)
