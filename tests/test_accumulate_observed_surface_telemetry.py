"""Tests for issue #274 -- trustworthy accumulate_observed_surface telemetry.

Verifies:
  1. utils/ingest_phase_profiler.py's percentile/lifetime-accumulator
     mechanics in isolation (no StreamingScene dependency).
  2. The lifetime-vs-windowed-store bug this issue's audit found: a naive
     "final dump reads the windowed store" implementation silently reports
     only whatever fired since the last periodic report_window() clear, not
     the whole run. The lifetime store must survive a windowed clear.
  3. StreamingScene.accumulate_observed_surface() actually emits the
     documented sub-stage timer names ("surface/materialize", "surface/mask",
     "surface/backproject", "surface/voxelize", "surface/quality_thickness",
     "surface/quality_sensor_noise") and workload counters when
     SPLAT_PROFILE_PHASES is on, through the REAL bound method (not a
     reimplementation) called on a minimal fake self -- i.e. the profiler is
     wired through the real ingestion code path, not only exercised via a
     standalone unit call to the profiler helpers.
  4. Zero behavior change / zero overhead source when SPLAT_PROFILE_PHASES is
     unset: no timer/counter entries are recorded at all.
"""

from __future__ import annotations

import importlib.util
from types import SimpleNamespace

import numpy as np
import pytest

import eligsir.utils.ingest_phase_profiler as prof

# The `smoke` CI job (.github/workflows/ci-cpu.yml, Dockerfile.ci-cpu) runs
# `python -m pytest -q tests/` in a deliberately torch-free image, and pytest
# aborts the WHOLE run on a single collection error -- so an unguarded
# `from scene.streaming_scene import StreamingScene` here (torch via
# scene/__init__ -> dataset_readers -> gaussian_model) takes every other test
# in the repo down with it. Same bug class as #449/#450/#452.
#
# Guarded at IMPORT level rather than with a module-wide
# `pytest.importorskip("torch")` on purpose: groups 1+2 below test
# utils/ingest_phase_profiler.py, which is a genuinely torch-free leaf module
# -- that leaf-ness is the whole point of this PR's extraction, and the
# torch-free CPU image is exactly where it should be proven. Skipping the
# whole file would have silently given that coverage up.
_HAS_TORCH = importlib.util.find_spec("torch") is not None
if _HAS_TORCH:
    from scene.streaming_scene import StreamingScene

requires_streaming_scene = pytest.mark.skipif(
    not _HAS_TORCH,
    reason="StreamingScene needs torch; groups 1+2 still run torch-free",
)


# ---------------------------------------------------------------------------
# 1 + 2: leaf-module mechanics
# ---------------------------------------------------------------------------

def test_percentiles_known_values():
    samples_ms = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
    pct = prof._percentiles(samples_ms)
    assert pct["p50"] == 50.0
    assert pct["p95"] == 100.0
    assert pct["p99"] == 100.0


def test_percentiles_empty_is_zero():
    assert prof._percentiles([]) == {"p50": 0.0, "p95": 0.0, "p99": 0.0}


def test_timer_noop_when_profiling_disabled(monkeypatch):
    monkeypatch.setattr(prof, "_PROFILE_PHASES", False)
    prof.reset_all()
    with prof._timed("region_x"):
        pass
    prof.record_count("counter_x", 5)
    assert prof._PHASE_OPS == {}
    assert prof._PHASE_OPS_LIFETIME == {}
    assert prof._PHASE_COUNTS == {}


def test_lifetime_store_survives_windowed_clear(monkeypatch):
    """Regression test for the exact bug this issue's audit found: before the
    fix, the end-of-run summary read straight from the windowed store that
    train_streaming.PhaseTimer.report_window() clears every ~100 iterations,
    so a "final" dump only reflected the last partial window, not the whole
    run. dump_phase_ops() must read the LIFETIME store, which is never
    cleared by a windowed reset."""
    monkeypatch.setattr(prof, "_PROFILE_PHASES", True)
    prof.reset_all()

    with prof._timed("ingest_accumulate_surface"):
        pass
    with prof._timed("ingest_accumulate_surface"):
        pass
    assert prof._PHASE_OPS["ingest_accumulate_surface"][1] == 2
    assert prof._PHASE_OPS_LIFETIME["ingest_accumulate_surface"][1] == 2

    # Simulate PhaseTimer.report_window()'s periodic clear of the windowed store.
    prof._PHASE_OPS.clear()
    assert prof._PHASE_OPS == {}
    # The lifetime store must be untouched.
    assert prof._PHASE_OPS_LIFETIME["ingest_accumulate_surface"][1] == 2

    with prof._timed("ingest_accumulate_surface"):
        pass
    summary = prof.dump_phase_ops(None)
    # A third call after the windowed clear must still be counted in the
    # final summary -- i.e. the summary reflects all 3 calls, not just the 1
    # that happened after the clear.
    assert summary["phases"]["ingest_accumulate_surface"]["count"] == 3


def test_dump_phase_ops_writes_json_artifact(tmp_path, monkeypatch):
    monkeypatch.setattr(prof, "_PROFILE_PHASES", True)
    prof.reset_all()
    with prof._timed("region_y"):
        pass
    prof.record_count("workload_y", 7)
    summary = prof.dump_phase_ops(str(tmp_path))
    out = tmp_path / "ingest_phase_profile.json"
    assert out.exists()
    import json
    on_disk = json.loads(out.read_text())
    assert on_disk == summary
    assert "region_y" in summary["phases"]
    assert summary["counts"]["workload_y"]["sum"] == 7


# ---------------------------------------------------------------------------
# 3 + 4: wired through the real StreamingScene.accumulate_observed_surface
# ---------------------------------------------------------------------------

class _FakeSurfaceSelf:
    """Minimal StreamingScene stand-in exposing exactly the attributes
    accumulate_observed_surface() (and the _accumulate_thickness /
    _accumulate_sensor_noise helpers it calls) read/write, with the real
    bound implementations from StreamingScene -- not a reimplementation."""

    def __init__(self, args):
        self.args = args
        self._obs_scales = (1, 2, 4)
        self._obs_voxel = 0.05
        self._obs_keys = {m: set() for m in self._obs_scales}
        self._obs_vox = {m: set() for m in self._obs_scales}
        self._obs_n_points = 0
        self._surface_cell_index = SimpleNamespace(normal_bins=8)
        self._scene_complexity_cache = None
        self._iq_frame_counter = 0
        self._iq_sensor_stride = 8
        self._iq_vox: dict = {}
        self._iq_vox_cap = 120000
        self._iq_sensor_sum = 0.0
        self._iq_sensor_sumsq = 0.0
        self._input_quality_cache = None


# Bound OUTSIDE the class body so the class itself can still be defined in a
# torch-free image (a class-body reference to StreamingScene would raise
# NameError at import time, reintroducing the collection abort this file's
# import guard exists to prevent). Every test that touches these attributes
# carries @requires_streaming_scene.
if _HAS_TORCH:
    _FakeSurfaceSelf.accumulate_observed_surface = StreamingScene.accumulate_observed_surface
    _FakeSurfaceSelf._accumulate_thickness = StreamingScene._accumulate_thickness
    _FakeSurfaceSelf._accumulate_sensor_noise = StreamingScene._accumulate_sensor_noise
    _FakeSurfaceSelf._stable_depth_samples = staticmethod(StreamingScene._stable_depth_samples)
    _FakeSurfaceSelf._pos_voxel_keys = staticmethod(StreamingScene._pos_voxel_keys)


def _make_frame(h=32, w=32, depth_value=1.5, index=0):
    depth = np.full((h, w), depth_value, dtype=np.float32)
    # A bit of structure so finite-difference normals aren't degenerate and
    # the stable-depth filter (disabled below) would have something to chew
    # on if it were enabled.
    depth[h // 2 :, :] += 0.3
    return SimpleNamespace(
        index=index,
        _cache_pos=None,
        _depth_np=depth,
        _depth_bytes=None,
        depth_path=None,
        depth_scale=1.0,
        c2w=np.eye(4, dtype=np.float32),
        depth_fx=None, depth_fy=None, depth_cx=None, depth_cy=None,
        fx=40.0, fy=40.0, cx=w / 2.0, cy=h / 2.0,
    )


def _make_args(**overrides):
    base = dict(
        streaming_track_scene_complexity=True,
        streaming_track_input_quality=True,
        streaming_min_depth=0.1,
        streaming_max_depth=8.0,
        streaming_depth_stride=4,
        streaming_depth_filter_enabled=False,  # keep the synthetic mask simple
    )
    base.update(overrides)
    return SimpleNamespace(**base)


@requires_streaming_scene
def test_accumulate_observed_surface_noop_when_profiling_disabled(monkeypatch):
    monkeypatch.setattr(prof, "_PROFILE_PHASES", False)
    prof.reset_all()
    fake = _FakeSurfaceSelf(_make_args())
    fake.accumulate_observed_surface(_make_frame())
    assert prof._PHASE_OPS == {}
    assert prof._PHASE_COUNTS == {}
    # The method itself still ran its real logic (not gated by profiling).
    assert fake._obs_n_points > 0


@requires_streaming_scene
def test_accumulate_observed_surface_emits_documented_substage_timers(monkeypatch):
    monkeypatch.setattr(prof, "_PROFILE_PHASES", True)
    prof.reset_all()
    fake = _FakeSurfaceSelf(_make_args())
    fake.accumulate_observed_surface(_make_frame())

    expected = {
        "surface/materialize",
        "surface/mask",
        "surface/backproject",
        "surface/voxelize",
        "surface/quality_thickness",
        # quality_sensor_noise only fires every _iq_sensor_stride-th frame;
        # first frame (_iq_frame_counter becomes 1) does not hit stride=8.
    }
    assert expected.issubset(prof._PHASE_OPS.keys())
    for name in expected:
        assert prof._PHASE_OPS[name][1] == 1  # called exactly once

    # Workload counters recorded alongside the timers.
    assert prof._PHASE_COUNTS["surface/input_pixels"][0] == 32 * 32
    assert prof._PHASE_COUNTS["surface/world_points"][0] > 0
    assert prof._PHASE_COUNTS["surface/voxel_keys_total_scale1"][0] > 0


@requires_streaming_scene
def test_quality_sensor_noise_timer_fires_on_stride(monkeypatch):
    """The one GPU-touching sub-region (estimate_depth_normals_pca) must be
    separately attributable -- confirm it fires on the configured stride and
    not on off-stride frames, so its cost is never silently blended into the
    surrounding pure-CPU stages."""
    monkeypatch.setattr(prof, "_PROFILE_PHASES", True)
    prof.reset_all()
    args = _make_args()
    fake = _FakeSurfaceSelf(args)
    fake._iq_sensor_stride = 2

    fake.accumulate_observed_surface(_make_frame(index=0))
    assert "surface/quality_sensor_noise" not in prof._PHASE_OPS

    fake.accumulate_observed_surface(_make_frame(index=1))
    assert prof._PHASE_OPS["surface/quality_sensor_noise"][1] == 1


@requires_streaming_scene
def test_accumulate_observed_surface_respects_track_scene_complexity_off(monkeypatch):
    monkeypatch.setattr(prof, "_PROFILE_PHASES", True)
    prof.reset_all()
    fake = _FakeSurfaceSelf(_make_args(streaming_track_scene_complexity=False))
    fake.accumulate_observed_surface(_make_frame())
    # Early-return before any _timed region or counter is touched.
    assert prof._PHASE_OPS == {}
    assert prof._PHASE_COUNTS == {}
    assert fake._obs_n_points == 0


class _FakeFrame806:
    """Minimal StreamingRGBDFrame-like stand-in with real depth so the
    bound `accumulate_observed_surface` does real work when reached via
    `ingest_next_frame`, not just a no-op stub."""

    def __init__(self, index: int = 0):
        self.index = index
        self._odom_valid = True
        self._cache_pos = None
        self._depth_np = np.full((16, 16), 1.5, dtype=np.float32)
        self._depth_bytes = None
        self.depth_path = None
        self.depth_scale = 1.0
        self.c2w = np.eye(4, dtype=np.float32)
        self.depth_fx = self.depth_fy = self.depth_cx = self.depth_cy = None
        self.fx = self.fy = 20.0
        self.cx = self.cy = 8.0


class _FakeIngestSelfForTelemetry:
    """Minimal StreamingScene stand-in exposing exactly what
    `ingest_next_frame`'s admission_mode="all", non-holdout, non-batched
    path touches -- mirrors
    tests/test_observation_db_ingestion_wiring.py's `_FakeIngestSelf` --
    PLUS the real `accumulate_observed_surface` dependencies from
    `_FakeSurfaceSelf` above, so this exercises the REAL
    `with _timed("ingest_accumulate_surface"): self.accumulate_observed_surface(frame)`
    call sites inside `ingest_next_frame` end-to-end, not just a direct call
    to `accumulate_observed_surface` bypassing that outer wrapper (issue
    #274's "wired through the real ingestion path, not only direct unit
    calls" task)."""

    def __init__(self, frames, args):
        self.args = args
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
        # accumulate_observed_surface's own dependencies (from _FakeSurfaceSelf).
        self._obs_scales = (1, 2, 4)
        self._obs_voxel = 0.05
        self._obs_keys = {m: set() for m in self._obs_scales}
        self._obs_vox = {m: set() for m in self._obs_scales}
        self._obs_n_points = 0
        self._surface_cell_index = SimpleNamespace(normal_bins=8)
        self._scene_complexity_cache = None
        self._iq_frame_counter = 0
        self._iq_sensor_stride = 8
        self._iq_vox: dict = {}
        self._iq_vox_cap = 120000
        self._iq_sensor_sum = 0.0
        self._iq_sensor_sumsq = 0.0
        self._input_quality_cache = None

    def _append_dense_trajectory(self, cam, frame, is_train, is_holdout):
        pass

    def _cache_inactive_frame(self, frame):
        pass

    def has_next_frame(self):
        return self._source_idx < len(self._frames)

    def _get_source_frame(self, index):
        if index >= len(self._frames):
            return None
        return self._frames[index]

    def _frame_to_camera(self, frame, load_image=True):
        cam = SimpleNamespace()
        cam.c2w = np.eye(4, dtype=np.float32)
        cam._streaming_frame = frame
        cam.materialize_pending_image = lambda: None
        return cam

    def update_cameras_extent(self):
        return None

    def _release_source_frames_upto_cursor(self):
        return None


if _HAS_TORCH:
    _FakeIngestSelfForTelemetry.accumulate_observed_surface = (
        StreamingScene.accumulate_observed_surface
    )
    _FakeIngestSelfForTelemetry._accumulate_thickness = StreamingScene._accumulate_thickness
    _FakeIngestSelfForTelemetry._accumulate_sensor_noise = StreamingScene._accumulate_sensor_noise
    _FakeIngestSelfForTelemetry._stable_depth_samples = staticmethod(
        StreamingScene._stable_depth_samples
    )
    _FakeIngestSelfForTelemetry._pos_voxel_keys = staticmethod(StreamingScene._pos_voxel_keys)
    _FakeIngestSelfForTelemetry._offload_camera_hot_window = (
        StreamingScene._offload_camera_hot_window
    )
    _FakeIngestSelfForTelemetry._offload_evicted_camera = StreamingScene._offload_evicted_camera
    # Issue #1115: bound from the REAL StreamingScene -- #1084 extracted
    # `ingest_next_frame`'s inline holdout/guard computation into this
    # predicate verbatim. This fake runs the non-holdout path
    # (`_holdout_indices = None` with `streaming_eval_hold = 0`, which the
    # real rule maps to `(False, False)`), so binding it keeps that branch
    # honest instead of asserting it into existence with a stub.
    _FakeIngestSelfForTelemetry.holdout_status = StreamingScene.holdout_status


@requires_streaming_scene
def test_ingest_next_frame_admitted_path_times_accumulate_surface(monkeypatch):
    """The outer `_timed("ingest_accumulate_surface")` wrapper around
    `self.accumulate_observed_surface(frame)` inside the REAL, unmodified
    `StreamingScene.ingest_next_frame` -- not a reimplementation, called
    exactly as `test_observation_db_ingestion_wiring.py`'s admission-ordering
    tests call it -- must fire for the ordinary admitted-frame path."""
    monkeypatch.setattr(
        "scene.cameras.prepare_camera_for_render", lambda cam, device="cuda": None
    )
    monkeypatch.setattr(prof, "_PROFILE_PHASES", True)
    prof.reset_all()
    args = _make_args(
        streaming_eval_hold=0,
        streaming_offload_camera_images=False,
        streaming_replay_buffer=512,
        streaming_frame_admission="all",
        streaming_global_reservoir_stride=0,
        streaming_keyframe_window=8,
    )
    fake_self = _FakeIngestSelfForTelemetry([_FakeFrame806(0)], args)

    def admission_fn(cam, frame):
        return True, {"admitted": True}, None

    result = StreamingScene.ingest_next_frame(fake_self, admission_fn=admission_fn)

    assert result is not None
    assert prof._PHASE_OPS["ingest_accumulate_surface"][1] == 1
    # And the sub-stage timers inside accumulate_observed_surface fired too,
    # confirming the real (non-stubbed) method ran, not a no-op.
    assert "surface/materialize" in prof._PHASE_OPS
    assert fake_self._obs_n_points > 0


@requires_streaming_scene
def test_ingest_next_frame_pregate_rejected_path_times_accumulate_surface(monkeypatch):
    """`accumulate_observed_surface` still runs (and is still timed) for a
    frame rejected by `pregate_fn`, per `ingest_next_frame`'s own docstring
    ("accumulate_observed_surface still runs regardless, since it only
    needs frame") -- verified through the real orchestration, not asserted
    only from reading the docstring."""
    monkeypatch.setattr(prof, "_PROFILE_PHASES", True)
    prof.reset_all()
    args = _make_args(
        streaming_eval_hold=0,
        streaming_offload_camera_images=False,
        streaming_replay_buffer=512,
        streaming_frame_admission="all",
        streaming_global_reservoir_stride=0,
        streaming_keyframe_window=8,
    )
    fake_self = _FakeIngestSelfForTelemetry([_FakeFrame806(0)], args)

    def pregate_fn(frame):
        return {"rejected": "no_motion"}

    result = StreamingScene.ingest_next_frame(fake_self, pregate_fn=pregate_fn)

    assert result == (None, fake_self._frames[0], False)
    assert prof._PHASE_OPS["ingest_accumulate_surface"][1] == 1
    assert fake_self._obs_n_points > 0


@requires_streaming_scene
def test_instrumentation_overhead_is_small_on_cpu(monkeypatch):
    """Quantify the *instrumentation* overhead this issue's tasks require
    (timer bookkeeping + dict writes), NOT GPU-sync overhead -- there is no
    GPU in this CPU-only test environment, and torch.cuda.is_available() is
    False here, so this measures the pure-Python cost of the _timed context
    managers and record_count() calls, an honest lower bound. GPU-sync
    overhead of the real _timed() cuda.synchronize() calls must be measured
    in a real GPU run (issue #275) -- not verified here."""
    import time

    frame = _make_frame(h=48, w=48)
    args = _make_args()

    monkeypatch.setattr(prof, "_PROFILE_PHASES", False)
    fake_off = _FakeSurfaceSelf(args)
    t0 = time.perf_counter()
    for _ in range(50):
        fake_off._scene_complexity_cache = None
        fake_off.accumulate_observed_surface(frame)
    t_off = time.perf_counter() - t0

    monkeypatch.setattr(prof, "_PROFILE_PHASES", True)
    prof.reset_all()
    fake_on = _FakeSurfaceSelf(args)
    t0 = time.perf_counter()
    for _ in range(50):
        fake_on._scene_complexity_cache = None
        fake_on.accumulate_observed_surface(frame)
    t_on = time.perf_counter() - t0

    # Generous bound (5x) -- this is a smoke check that instrumentation
    # doesn't add a gross constant-factor tax on CPU, not a tight perf
    # assertion (CI hardware variance makes a tight bound flaky).
    assert t_on < max(t_off * 5.0, t_off + 0.5)
