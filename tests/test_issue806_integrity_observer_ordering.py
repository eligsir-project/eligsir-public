"""Regression tests for issue #806 -- mandatory integrity observation must
run on EVERY materialized frame, before any discretionary prefilter
(the motion pregate) can remove it from the decision pipeline.

Two groups of tests:

1. `StreamingScene.ingest_next_frame` orchestration tests (via a minimal
   fake `self`, mirroring `tests/test_observation_db_ingestion_wiring.py`'s
   `_FakeIngestSelf` harness) -- exercise the REAL, unmodified
   `ingest_next_frame` method body, so a regression in call ORDER inside
   that method fails these tests, not just a unit test of helpers it calls.

   Deliberately does NOT import `train_streaming` at module scope (unlike
   `test_observation_db_ingestion_wiring.py`, which needs
   `bind_resolution_tier`/`record_frame_observation` from it): this host's
   torch/torchvision install is mismatched (torch 2.12.0 vs. torchvision
   0.25.0+rocm7.1 -- `RuntimeError: operator torchvision::nms does not
   exist`, verified pre-existing and unrelated to this change), and
   `train_streaming.py` imports `torchvision.utils.save_image` at module
   level, so any test file importing it fails to even COLLECT on this host.
   `scene.streaming_scene` itself has no such import, so these tests import
   only that module directly.

2. Static/AST-based checks against `train_streaming.py`'s source text
   (mirrors `tests/test_issue795_wiring.py`'s pattern for the identical
   reason -- reads the file as text/AST rather than importing it) proving
   the real wiring: `_integrity_observer_closure` exists, is passed as
   `integrity_observer_fn=` unconditionally (not gated behind the
   hybrid_keyframe-only `admission_fn`/`pregate_fn` branch), and
   `MapReinitTracker`/`OrbslamLogReinitWatcher` behave correctly
   independent of each other.

3. A static AST guard (`ingest_next_frame`'s own source) that the
   `integrity_observer_fn` call always precedes `pregate_fn`/`admission_fn`/
   camera construction -- so a future prefilter (#802) accidentally
   inserted ahead of it would fail this test.
"""

from __future__ import annotations

import ast
import inspect
import os
import textwrap
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scene.streaming_scene import StreamingScene

REPO_ROOT = Path(__file__).parent.parent
TRAIN_STREAMING_PATH = REPO_ROOT / "train_streaming.py"


# ---------------------------------------------------------------------------
# Group 1: StreamingScene.ingest_next_frame orchestration
# ---------------------------------------------------------------------------


class _FakeFrame:
    def __init__(self, index: int) -> None:
        self.index = index
        self._odom_valid = True


class _FakeIngestSelf:
    """Minimal stand-in for a StreamingScene instance -- copied from
    `tests/test_observation_db_ingestion_wiring.py`'s `_FakeIngestSelf`
    (kept in sync deliberately; duplicated here rather than imported so
    this file avoids that module's `train_streaming` import at collection
    time -- see module docstring)."""

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
        # Issue #1135: counter the real ingest_next_frame body now increments
        # directly (self._stale_pose_holdout_rejected += 1) for a reserved
        # holdout dropped by the _odom_valid-is-False guard.
        self._stale_pose_holdout_rejected = 0
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
        pass

    def _cache_inactive_frame(self, frame):
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
        return None

    _offload_camera_hot_window = StreamingScene._offload_camera_hot_window
    _offload_evicted_camera = StreamingScene._offload_evicted_camera
    # Issue #1115: bound from the REAL StreamingScene -- #1084 extracted
    # `ingest_next_frame`'s inline holdout/guard computation into this
    # predicate verbatim, and several tests here flip `_holdout_indices` to
    # exercise the holdout branch, so it must be the production rule.
    holdout_status = StreamingScene.holdout_status


_REINIT_REJECT_STATS = {
    "mode": "hybrid_keyframe",
    "admitted": 0,
    "reason": "tracking_reinit",
    "score": 0.0,
    "cheap_score": 0.0,
    "pose_delta_m": 0.0,
    "pose_delta_deg": 0.0,
    "gap": 0,
    "overlap": 0.0,
    "forced": 0,
    "moved": 0,
    "novel": 0,
}


def test_integrity_observer_called_before_pregate(monkeypatch):
    """Test #6 (ordering half): the integrity observer must fire before
    pregate_fn is even invoked, for an ordinary (no-event) frame."""
    monkeypatch.setattr(
        "scene.cameras.prepare_camera_for_render", lambda cam, device="cuda": None
    )
    fake_self = _FakeIngestSelf([_FakeFrame(0)])

    calls: list[str] = []

    def integrity_observer_fn(frame):
        calls.append("integrity")
        return None

    def pregate_fn(frame):
        calls.append("pregate")
        return None

    def admission_fn(cam, frame):
        calls.append("admission")
        return True, {"admitted": True}, None

    result = StreamingScene.ingest_next_frame(
        fake_self, admission_fn=admission_fn, pregate_fn=pregate_fn,
        integrity_observer_fn=integrity_observer_fn,
    )

    assert result is not None
    assert calls == ["integrity", "pregate", "admission"]


def test_integrity_veto_takes_precedence_over_pregate_rejection(monkeypatch):
    """Test #1 (core precedence): a frame that would ALSO be rejected by the
    motion pregate (reason="no_motion") must be rejected with the integrity
    reason instead, and pregate_fn must never even be called -- an ordinary
    no-motion rejection must never mask a map-reinit veto."""
    monkeypatch.setattr(
        "scene.cameras.prepare_camera_for_render", lambda cam, device="cuda": None
    )
    fake_self = _FakeIngestSelf([_FakeFrame(0)])

    pregate_called = []

    def integrity_observer_fn(frame):
        return dict(_REINIT_REJECT_STATS)

    def pregate_fn(frame):
        pregate_called.append(frame.index)
        return {"admitted": 0, "reason": "no_motion"}  # would reject if reached

    def admission_fn(cam, frame):
        raise AssertionError("admission_fn must not be called for a vetoed frame")

    result = StreamingScene.ingest_next_frame(
        fake_self, admission_fn=admission_fn, pregate_fn=pregate_fn,
        integrity_observer_fn=integrity_observer_fn,
    )

    assert result == (None, fake_self._frames[0], False)
    assert pregate_called == []  # #806: pregate_fn must never even be called
    assert fake_self._frames[0]._streaming_admission_stats["reason"] == "tracking_reinit"
    assert fake_self._keyframes_rejected == 1


def test_integrity_veto_applies_under_admission_all_too(monkeypatch):
    """The invariant is "every ingested frame", not "every hybrid_keyframe
    frame" -- admission=all (no pregate_fn/admission_fn wired at all) must
    still short-circuit on an integrity veto."""
    monkeypatch.setattr(
        "scene.cameras.prepare_camera_for_render", lambda cam, device="cuda": None
    )
    fake_self = _FakeIngestSelf([_FakeFrame(0)])

    def integrity_observer_fn(frame):
        return dict(_REINIT_REJECT_STATS)

    result = StreamingScene.ingest_next_frame(
        fake_self, integrity_observer_fn=integrity_observer_fn,
    )

    assert result == (None, fake_self._frames[0], False)
    assert fake_self._frames[0]._streaming_admission_stats["reason"] == "tracking_reinit"


def test_integrity_observer_runs_without_camera_or_render(monkeypatch):
    """Test #2: the observer must fire, and be able to veto, without a
    Camera ever being built (and therefore without any render)."""
    fake_self = _FakeIngestSelf([_FakeFrame(0)])

    def _frame_to_camera_spy(frame, load_image=True):
        raise AssertionError("must not build a Camera for a vetoed frame")

    fake_self._frame_to_camera = _frame_to_camera_spy

    def integrity_observer_fn(frame):
        assert not hasattr(frame, "c2w")  # never touched a camera-shaped object
        return dict(_REINIT_REJECT_STATS)

    result = StreamingScene.ingest_next_frame(
        fake_self, integrity_observer_fn=integrity_observer_fn,
    )

    assert result == (None, fake_self._frames[0], False)


def test_integrity_observer_still_runs_for_holdout_frames(monkeypatch):
    """The observer must be called for holdout/guard candidates too (the
    mandatory-observation invariant is NOT scoped to admitted candidates),
    even though -- like the pregate -- a veto from it does not block a
    holdout frame from becoming a real eval camera (holdout selection is
    admission-independent by design; see the existing pregate/holdout
    contract this mirrors)."""
    monkeypatch.setattr(
        "scene.cameras.prepare_camera_for_render", lambda cam, device="cuda": None
    )
    fake_self = _FakeIngestSelf([_FakeFrame(0)])
    fake_self._holdout_indices = {0}
    fake_self._guard_indices = set()

    calls: list[int] = []

    def integrity_observer_fn(frame):
        calls.append(frame.index)
        return dict(_REINIT_REJECT_STATS)  # would reject an ordinary candidate

    result = StreamingScene.ingest_next_frame(
        fake_self, integrity_observer_fn=integrity_observer_fn,
    )

    assert calls == [0]  # observer WAS called for the holdout frame
    assert result is not None
    cam, frame, is_train = result
    assert is_train is False
    assert fake_self._test_cameras == [cam]  # still became a real eval camera


def test_ordinary_frame_no_integrity_event_keeps_pregate_reason_unchanged(monkeypatch):
    """Test #6 (behavior half): with no integrity event, an ordinary
    no-motion pregate rejection is byte-identical to pre-#806 behavior
    (same reason string, same stats dict, same short-circuit contract)."""
    monkeypatch.setattr(
        "scene.cameras.prepare_camera_for_render", lambda cam, device="cuda": None
    )
    fake_self = _FakeIngestSelf([_FakeFrame(0)])

    def integrity_observer_fn(frame):
        return None  # no event

    def pregate_fn(frame):
        return {"admitted": 0, "reason": "no_motion"}

    result = StreamingScene.ingest_next_frame(
        fake_self, pregate_fn=pregate_fn, integrity_observer_fn=integrity_observer_fn,
    )

    assert result == (None, fake_self._frames[0], False)
    assert fake_self._frames[0]._streaming_admission_stats == {"admitted": 0, "reason": "no_motion"}


def test_no_integrity_observer_fn_is_pre_806_behavior(monkeypatch):
    """Omitting integrity_observer_fn entirely (the default, None) must
    reproduce exactly the pre-#806 code path -- no new behavior forced on
    any caller that hasn't opted in."""
    monkeypatch.setattr(
        "scene.cameras.prepare_camera_for_render", lambda cam, device="cuda": None
    )
    fake_self = _FakeIngestSelf([_FakeFrame(0)])

    def admission_fn(cam, frame):
        return True, {"admitted": True}, None

    result = StreamingScene.ingest_next_frame(fake_self, admission_fn=admission_fn)
    assert result is not None


def test_integrity_observer_exception_is_non_fatal(monkeypatch, capsys):
    """A crash inside integrity_observer_fn must not take down real
    training -- the frame proceeds exactly as if no integrity event fired
    (this only guards a *crash in observation*, never swallows a
    successfully-returned detection -- see the docstring in
    StreamingScene.ingest_next_frame)."""
    monkeypatch.setattr(
        "scene.cameras.prepare_camera_for_render", lambda cam, device="cuda": None
    )
    fake_self = _FakeIngestSelf([_FakeFrame(0)])

    def integrity_observer_fn(frame):
        raise RuntimeError("boom")

    def admission_fn(cam, frame):
        return True, {"admitted": True}, None

    result = StreamingScene.ingest_next_frame(
        fake_self, admission_fn=admission_fn, integrity_observer_fn=integrity_observer_fn,
    )

    assert result is not None  # did not crash / did not silently reject
    err = capsys.readouterr().out
    assert "integrity-observer" in err
    assert "boom" in err


# ---------------------------------------------------------------------------
# Group 2: static wiring checks against train_streaming.py's source
# (mirrors tests/test_issue795_wiring.py's ast/text-based approach --
# importing train_streaming triggers a pre-existing, unrelated host
# torch/torchvision mismatch; see module docstring)
# ---------------------------------------------------------------------------


def _train_streaming_source() -> str:
    assert TRAIN_STREAMING_PATH.exists(), f"{TRAIN_STREAMING_PATH} not found"
    return TRAIN_STREAMING_PATH.read_text()


def test_integrity_observer_closure_defined():
    src = _train_streaming_source()
    assert "def _integrity_observer_closure(frame):" in src


def test_integrity_observer_wired_into_ingest_call():
    src = _train_streaming_source()
    assert "integrity_observer_fn=_integrity_observer_closure" in src


def test_integrity_observer_wiring_is_unconditional_on_admission_mode():
    """Test #7 -- the observer wiring must not sit inside the
    `if args.streaming_frame_admission == "hybrid_keyframe":` block that
    gates `admission_fn`/`pregate_fn` (coverage-gain-gate / velocity-
    adaptive / any other admitter config only ever affects THAT block).
    Verified via AST parent-walk, not string position, so it survives
    incidental reformatting."""
    src = _train_streaming_source()
    tree = ast.parse(src)

    parents: dict[int, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[id(child)] = node

    target_call = None
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "ingest_next_frame"
        ):
            if any(kw.arg == "integrity_observer_fn" for kw in node.keywords):
                target_call = node
                break

    assert target_call is not None, "ingest_next_frame(...) call with integrity_observer_fn= not found"

    # Walk up parents; none may be an `if` node whose test mentions
    # streaming_frame_admission (the hybrid_keyframe-only gate).
    cur = target_call
    while id(cur) in parents:
        cur = parents[id(cur)]
        if isinstance(cur, ast.If):
            test_src = ast.unparse(cur.test) if hasattr(ast, "unparse") else ""
            assert "streaming_frame_admission" not in test_src, (
                "integrity_observer_fn wiring must not be gated behind the "
                "hybrid_keyframe admission-mode check"
            )


def test_admission_fingerprint_reject_reason_preserved():
    """The '#795' AdmissionFingerprint contract (see
    tests/test_issue795_wiring.py::test_tracking_reinit_rejection_recorded)
    must still hold after the veto moved to _integrity_observer_closure."""
    src = _train_streaming_source()
    assert '_admission_fingerprint.record_reject("tracking_reinit")' in src


def test_keyframe_admission_closure_no_longer_duplicates_the_veto_check():
    """Deliberate single-call-site decision (see
    _integrity_observer_closure's docstring): _keyframe_admission_closure
    must not re-run `_check_tracking_reinit` -- proves the removal was
    actually applied, not just documented."""
    src = _train_streaming_source()
    tree = ast.parse(src)
    func = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "_keyframe_admission_closure"
    )
    calls = [
        n.func.id for n in ast.walk(func)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
    ]
    assert "_check_tracking_reinit" not in calls, (
        "_keyframe_admission_closure must not call _check_tracking_reinit "
        "directly -- see _integrity_observer_closure's single-call-site "
        "docstring"
    )


# ---------------------------------------------------------------------------
# Group 3: MapReinitTracker / OrbslamLogReinitWatcher independence
# (tests #3, #4, #5 -- these classes are already unit-tested in
# tests/test_tracking_confidence.py for their scoring behavior; these tests
# specifically target the multi-frame persistence / independence properties
# issue #806 calls out)
# ---------------------------------------------------------------------------


def test_permanent_veto_persists_across_later_ok_frames():
    """Test #3: once vetoing, later OK-tracking frames on the (new,
    unrelated-coordinate-frame) map must remain vetoed forever."""
    from eligsir.mapping.scheduler.tracking_confidence import (
        MapReinitTracker,
        TrackingStatus,
        TRACKING_STATE_NOT_INITIALIZED,
        TRACKING_STATE_OK,
    )

    tracker = MapReinitTracker(settle_frames=5)
    assert tracker.observe(TrackingStatus(TRACKING_STATE_OK, 100), 0) is None
    assert tracker.vetoing is False
    assert tracker.observe(TrackingStatus(TRACKING_STATE_NOT_INITIALIZED, 0), 1) == "reinit_started"
    assert tracker.vetoing is True

    # Many subsequent clean OK frames on the new map: still vetoed.
    for i in range(2, 50):
        result = tracker.observe(TrackingStatus(TRACKING_STATE_OK, 120), i)
        assert result is None
        assert tracker.vetoing is True, f"veto cleared at frame {i} -- must be permanent"


def test_tracking_status_detector_works_without_log_watcher():
    """Test #4: the tracking-status-based detector (MapReinitTracker) fires
    correctly on its own, with no OrbslamLogReinitWatcher involved at all
    (not even constructed) -- proves it doesn't secretly depend on the
    fallback."""
    from eligsir.mapping.scheduler.tracking_confidence import (
        MapReinitTracker,
        TrackingStatus,
        TRACKING_STATE_NOT_INITIALIZED,
        TRACKING_STATE_OK,
    )

    tracker = MapReinitTracker()
    tracker.observe(TrackingStatus(TRACKING_STATE_OK, 50), 0)
    event = tracker.observe(TrackingStatus(TRACKING_STATE_NOT_INITIALIZED, 0), 10)
    assert event == "reinit_started"
    assert tracker.vetoing is True


def test_log_watcher_is_an_independent_fallback(tmp_path):
    """Test #5: OrbslamLogReinitWatcher detects a re-init purely from the
    log file, with no TrackingStatus/MapReinitTracker.observe() call ever
    made -- an independent detection path, not a dependent one."""
    from eligsir.mapping.scheduler.tracking_confidence import OrbslamLogReinitWatcher

    log_path = tmp_path / "orbslam_live.log"
    log_path.write_text("Creation of new map with id: 0\nsome other line\n")

    watcher = OrbslamLogReinitWatcher(str(log_path))
    assert watcher.check() is False  # exactly one map so far -- normal startup

    with open(log_path, "a") as f:
        f.write("TRACK_REF_KF: Less than 15 matches!!\n")
        f.write("Creation of new map with id: 1\n")

    assert watcher.check() is True  # second map -- real re-init, fires exactly once
    assert watcher.check() is False  # subsequent calls: no new event


def test_log_watcher_missing_file_is_a_safe_noop():
    from eligsir.mapping.scheduler.tracking_confidence import OrbslamLogReinitWatcher

    watcher = OrbslamLogReinitWatcher("/nonexistent/path/orbslam_live.log")
    for _ in range(5):
        assert watcher.check() is False


# ---------------------------------------------------------------------------
# Group 4 (test #8): static AST guard on StreamingScene.ingest_next_frame
# itself -- a future prefilter (#802) accidentally inserted ahead of the
# mandatory integrity observation must fail this test.
# ---------------------------------------------------------------------------


def test_ast_guard_integrity_observer_precedes_pregate_and_admission_calls():
    """#806 acceptance criterion / test #8: statically prove
    `integrity_observer_fn(frame)` is invoked, in source order, before
    `pregate_fn(frame)`, before `admission_fn(cam, frame)`, and before
    `self._frame_to_camera(...)` inside `ingest_next_frame`'s body. A
    future change that introduces a new discretionary prefilter ahead of
    the integrity call (#802's stronger source-rate stage) will shift that
    prefilter's own call earlier than `integrity_observer_fn`'s -- this
    test does not know about any FUTURE call by name, but it does pin
    `integrity_observer_fn` as strictly earliest among the calls that
    exist today, which is the concrete, checkable form of "nothing may run
    ahead of mandatory integrity observation" until a new call is
    deliberately added earlier still (at which point this test's own
    failure is the intended guard rail forcing that decision to be
    explicit, not silent).
    """
    src = textwrap.dedent(inspect.getsource(StreamingScene.ingest_next_frame))
    tree = ast.parse(src)
    func = tree.body[0]
    assert isinstance(func, ast.FunctionDef)

    calls_by_line: dict[str, int] = {}
    for node in ast.walk(func):
        if not isinstance(node, ast.Call):
            continue
        target = None
        if isinstance(node.func, ast.Name) and node.func.id in (
            "integrity_observer_fn",
            "pregate_fn",
            "admission_fn",
        ):
            target = node.func.id
        elif (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "_frame_to_camera"
        ):
            target = "_frame_to_camera"
        if target is not None and target not in calls_by_line:
            calls_by_line[target] = node.lineno

    assert "integrity_observer_fn" in calls_by_line, (
        "integrity_observer_fn(...) call not found in ingest_next_frame -- "
        "the mandatory integrity observation stage appears to have been removed"
    )
    for other in ("pregate_fn", "admission_fn", "_frame_to_camera"):
        assert other in calls_by_line, f"{other}(...) call not found in ingest_next_frame"
        assert calls_by_line["integrity_observer_fn"] < calls_by_line[other], (
            f"integrity_observer_fn(...) (line {calls_by_line['integrity_observer_fn']}) "
            f"must precede {other}(...) (line {calls_by_line[other]}) -- a discretionary "
            f"prefilter/admission stage must never run ahead of mandatory integrity "
            f"observation (issue #806)"
        )
