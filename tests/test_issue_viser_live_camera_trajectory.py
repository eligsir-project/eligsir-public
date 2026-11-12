"""Live camera frustum + trajectory trail wiring (follow-up to #1385/#1394/#1400).

Covers the three layers touched to stop ``pose_c2w``/trajectory from being
silently dropped between training and the Viser browser sidecar:

1. ``CanonicalViewerAdapter``: ``log_camera`` now forwards ``pose`` into
   ``RawInputPreview.pose_c2w`` (previously discarded in ``_submit_raw``),
   and ``log_trajectory`` is no longer a no-op -- it publishes a bounded
   ``TrajectoryPreview``.
2. ``ViserPreviewSink``: ``on_raw_input`` now forwards ``pose_c2w`` in its
   wire payload (previously dropped there even after (1) started carrying
   it), and a new ``on_trajectory`` hook forwards the bounded pose array.
3. ``viser_server.py``: a persistent ``/camera/live`` frustum and
   ``/camera/trajectory`` line-segments scene node, updated in place from
   those two wire payloads.

Also confirms the existing "viewer off is a no-op" contract still holds for
both new/changed hooks.
"""
from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from eligsir.contracts.types import Pose, RGBDFrame, CameraIntrinsics
from eligsir.viewers.canonical import CanonicalViewerConfig
from eligsir.viewers.canonical_adapter import CanonicalViewerAdapter, _MAX_TRAJECTORY_POSES
from eligsir.viewers.events import RawInputPreview, TrajectoryPreview
from eligsir.viewers.sinks.base import PreviewSink


def _wait_for(pred, timeout_s=2.0):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(0.01)
    return False


def _frame(step: int) -> RGBDFrame:
    intr = CameraIntrinsics(
        fx=100.0, fy=100.0, cx=8.0, cy=8.0, width=16, height=16,
        depth_fx=100.0, depth_fy=100.0, depth_cx=8.0, depth_cy=8.0,
    )
    return RGBDFrame(
        index=step,
        timestamp=float(step),
        rgb=np.full((16, 16, 3), 0.5, dtype=np.float32),
        depth=np.ones((16, 16), dtype=np.float32),
        intrinsics=intr,
        pose=None,
    )


def _pose(tx: float) -> Pose:
    m = np.eye(4, dtype=np.float32)
    m[0, 3] = tx
    return Pose(m)


class _RecordingSink(PreviewSink):
    def __init__(self):
        self.raw_inputs = []
        self.trajectories = []
        self._lock = threading.Lock()

    def on_raw_input(self, envelope):
        with self._lock:
            self.raw_inputs.append(envelope.event)

    def on_trajectory(self, envelope):
        with self._lock:
            self.trajectories.append(envelope.event)


# ---------------------------------------------------------------------------
# CanonicalViewerAdapter: log_camera forwards pose_c2w
# ---------------------------------------------------------------------------
def test_log_camera_forwards_pose_c2w_to_raw_input_preview():
    sink = _RecordingSink()
    adapter = CanonicalViewerAdapter(sinks=[sink], drain_poll_interval_s=0.01)
    try:
        pose = _pose(3.0)
        adapter.log_camera(_frame(10), pose, 10)
        assert _wait_for(lambda: len(sink.raw_inputs) == 1)
        event: RawInputPreview = sink.raw_inputs[0]
        assert event.pose_c2w is not None
        assert event.pose_c2w.shape == (4, 4)
        assert np.allclose(event.pose_c2w, pose.matrix)
    finally:
        adapter.close()


def test_log_camera_pose_is_copied_not_aliased():
    # The caller's Pose must remain mutable/reusable after the call --
    # RawInputPreview freezes what it's given, it doesn't copy (see events.py's
    # array-ownership rule), so the adapter itself must copy.
    sink = _RecordingSink()
    adapter = CanonicalViewerAdapter(sinks=[sink], drain_poll_interval_s=0.01)
    try:
        pose = _pose(1.0)
        adapter.log_camera(_frame(1), pose, 1)
        assert _wait_for(lambda: len(sink.raw_inputs) == 1)
        pose.matrix[0, 3] = 999.0  # mutate the caller's own pose afterward
        assert sink.raw_inputs[0].pose_c2w[0, 3] == pytest.approx(1.0)
    finally:
        adapter.close()


def test_log_rgb_still_carries_no_pose():
    # log_rgb has no pose argument at all; must not regress to requiring one.
    sink = _RecordingSink()
    adapter = CanonicalViewerAdapter(sinks=[sink], drain_poll_interval_s=0.01)
    try:
        adapter.log_rgb(_frame(1), 1)
        assert _wait_for(lambda: len(sink.raw_inputs) == 1)
        assert sink.raw_inputs[0].pose_c2w is None
    finally:
        adapter.close()


# ---------------------------------------------------------------------------
# CanonicalViewerAdapter: log_trajectory is no longer a no-op
# ---------------------------------------------------------------------------
def test_log_trajectory_publishes_bounded_pose_array():
    sink = _RecordingSink()
    adapter = CanonicalViewerAdapter(
        sinks=[sink],
        drain_poll_interval_s=0.01,
        config=CanonicalViewerConfig(trajectory_max_hz=0.0),  # unlimited for the test
    )
    try:
        poses = [_pose(float(i)) for i in range(5)]
        adapter.log_trajectory(poses, None, 100)
        assert _wait_for(lambda: len(sink.trajectories) == 1)
        event: TrajectoryPreview = sink.trajectories[0]
        assert event.poses_est.shape == (5, 4, 4)
        assert event.poses_gt is None
        assert np.allclose(event.poses_est[2, 0, 3], 2.0)
    finally:
        adapter.close()


def test_log_trajectory_caps_at_max_trajectory_poses():
    sink = _RecordingSink()
    adapter = CanonicalViewerAdapter(
        sinks=[sink],
        drain_poll_interval_s=0.01,
        config=CanonicalViewerConfig(trajectory_max_hz=0.0),
    )
    try:
        n = _MAX_TRAJECTORY_POSES + 500
        poses = [_pose(float(i)) for i in range(n)]
        adapter.log_trajectory(poses, None, 1)
        assert _wait_for(lambda: len(sink.trajectories) == 1)
        event: TrajectoryPreview = sink.trajectories[0]
        assert event.poses_est.shape[0] == _MAX_TRAJECTORY_POSES
        # The tail (most recent poses), not the head, must be kept.
        assert np.allclose(event.poses_est[-1, 0, 3], float(n - 1))
    finally:
        adapter.close()


def test_log_trajectory_with_empty_list_is_a_noop():
    sink = _RecordingSink()
    adapter = CanonicalViewerAdapter(sinks=[sink], drain_poll_interval_s=0.01)
    try:
        adapter.log_trajectory([], None, 1)
        time.sleep(0.05)
        assert sink.trajectories == []
    finally:
        adapter.close()


def test_log_trajectory_with_gt_poses():
    sink = _RecordingSink()
    adapter = CanonicalViewerAdapter(
        sinks=[sink],
        drain_poll_interval_s=0.01,
        config=CanonicalViewerConfig(trajectory_max_hz=0.0),
    )
    try:
        est = [_pose(float(i)) for i in range(3)]
        gt = [_pose(float(i) + 0.1) for i in range(3)]
        adapter.log_trajectory(est, gt, 1)
        assert _wait_for(lambda: len(sink.trajectories) == 1)
        event = sink.trajectories[0]
        assert event.poses_gt is not None
        assert event.poses_gt.shape == (3, 4, 4)
    finally:
        adapter.close()


# ---------------------------------------------------------------------------
# viewer-off no-op contract (mirrors the existing pattern for other hooks)
# ---------------------------------------------------------------------------
def test_disabled_viewer_log_camera_and_log_trajectory_are_noops():
    sink = _RecordingSink()
    adapter = CanonicalViewerAdapter(
        sinks=[sink],
        drain_poll_interval_s=0.01,
        config=CanonicalViewerConfig(enabled=False),
    )
    try:
        adapter.log_camera(_frame(1), _pose(1.0), 1)
        adapter.log_trajectory([_pose(1.0), _pose(2.0)], None, 1)
        time.sleep(0.05)
        assert sink.raw_inputs == []
        assert sink.trajectories == []
    finally:
        adapter.close()


def test_closed_adapter_log_camera_and_log_trajectory_are_noops():
    sink = _RecordingSink()
    adapter = CanonicalViewerAdapter(sinks=[sink], drain_poll_interval_s=0.01)
    adapter.close()
    # Must not raise after close(), and must not submit anything.
    adapter.log_camera(_frame(1), _pose(1.0), 1)
    adapter.log_trajectory([_pose(1.0), _pose(2.0)], None, 1)
    time.sleep(0.05)
    assert sink.raw_inputs == []
    assert sink.trajectories == []


# ---------------------------------------------------------------------------
# ViserPreviewSink: real sidecar process, real forwarding (mirrors
# test_viser_sink_e2e.py's pattern -- skipped when viser isn't installed).
# ---------------------------------------------------------------------------
pytest.importorskip("viser")

import socket
import urllib.request

from eligsir.viewers.bus import PreviewBus
from eligsir.viewers.events import PreviewChannel
from eligsir.viewers.sinks.viser_sink import ViserPreviewSink


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture()
def viser_sink_fixture():
    port = _free_port()
    s = ViserPreviewSink(port=port, host="127.0.0.1", start_timeout_s=20.0)
    yield s
    s.close()


def test_on_raw_input_forwards_pose_without_dropping(viser_sink_fixture):
    sink = viser_sink_fixture
    rgb = np.full((16, 16, 3), 128, dtype=np.uint8)
    pose = np.eye(4, dtype=np.float32)
    pose[0, 3] = 5.0
    event = RawInputPreview(
        iteration=1,
        wall_time_s=time.monotonic(),
        source_frame_id=1,
        rgb=rgb,
        pose_c2w=pose,
    )
    bus = PreviewBus(name="test-raw-pose")
    envelope = bus.publish(event, channel=PreviewChannel.RAW_INPUT)

    sink.on_raw_input(envelope)
    time.sleep(0.5)

    assert sink.sent == 1
    assert sink.dropped == 0
    url = f"http://127.0.0.1:{sink.port}/"
    with urllib.request.urlopen(url, timeout=5.0) as resp:
        assert resp.status == 200


def test_on_trajectory_forwards_pose_array_without_dropping(viser_sink_fixture):
    sink = viser_sink_fixture
    poses = np.stack([np.eye(4, dtype=np.float32) for _ in range(5)])
    for i in range(5):
        poses[i, 0, 3] = float(i)
    event = TrajectoryPreview(
        iteration=1,
        wall_time_s=time.monotonic(),
        poses_est=poses,
    )
    bus = PreviewBus(name="test-trajectory")
    envelope = bus.publish(event, channel=PreviewChannel.TRAJECTORY)

    sink.on_trajectory(envelope)
    time.sleep(0.5)

    assert sink.sent == 1
    assert sink.dropped == 0
    url = f"http://127.0.0.1:{sink.port}/"
    with urllib.request.urlopen(url, timeout=5.0) as resp:
        assert resp.status == 200


# ---------------------------------------------------------------------------
# viser_server.py: real ViserServer, no subprocess (mirrors
# test_viser_server_map_mode_toggle.py's pattern).
# ---------------------------------------------------------------------------
from eligsir.viewers.sinks.viser_server import (
    _apply,
    _build_ui,
    _rotmat_to_quat_wxyz,
    _quat_wxyz_to_rotmat,
    _MAX_TRAJECTORY_TRAIL_POINTS,
)


@pytest.fixture()
def server():
    import viser

    port = _free_port()
    s = viser.ViserServer(host="127.0.0.1", port=port)
    yield s
    s.stop()


def test_live_camera_and_trajectory_nodes_start_invisible(server):
    handles, texts, _dropdown, _cb = _build_ui(server)
    assert "live_camera" in handles
    assert "trajectory" in handles
    assert handles["live_camera"].visible is False
    assert handles["trajectory"].visible is False


def test_raw_input_with_pose_updates_live_camera_frustum(server):
    handles, texts, _dropdown, _cb = _build_ui(server)
    pose = np.eye(4, dtype=np.float32)
    pose[:3, 3] = [1.0, 2.0, 3.0]
    cmd = {
        "kind": "raw_input",
        "rgb": np.zeros((20, 40, 3), dtype=np.uint8),  # aspect = 40/20 = 2.0
        "source_frame_id": 7,
        "camera_id": None,
        "timestamp_s": 0.5,
        "pose_c2w": pose,
    }
    _apply(cmd, handles, texts)
    live = handles["live_camera"]
    assert live.visible is True
    assert np.allclose(live.position, [1.0, 2.0, 3.0])
    assert live.aspect == pytest.approx(2.0)
    # identity rotation -> identity quaternion (w=1, x=y=z=0)
    assert np.allclose(live.wxyz, [1.0, 0.0, 0.0, 0.0], atol=1e-5)


def test_raw_input_without_pose_leaves_live_camera_untouched(server):
    handles, texts, _dropdown, _cb = _build_ui(server)
    cmd = {
        "kind": "raw_input",
        "rgb": np.zeros((16, 16, 3), dtype=np.uint8),
        "source_frame_id": 1,
        "camera_id": None,
        "timestamp_s": 0.0,
        "pose_c2w": None,
    }
    _apply(cmd, handles, texts)
    assert handles["live_camera"].visible is False


def test_trajectory_updates_line_segments_from_consecutive_poses(server):
    handles, texts, _dropdown, _cb = _build_ui(server)
    n = 4
    poses = np.stack([np.eye(4, dtype=np.float32) for _ in range(n)])
    for i in range(n):
        poses[i, 0, 3] = float(i)  # translations: (0,0,0),(1,0,0),(2,0,0),(3,0,0)
    cmd = {"kind": "trajectory", "poses_est": poses, "poses_gt": None, "iteration": 10}
    _apply(cmd, handles, texts)
    traj = handles["trajectory"]
    assert traj.visible is True
    assert traj.points.shape == (n - 1, 2, 3)
    assert np.allclose(traj.points[0, 0], [0.0, 0.0, 0.0])
    assert np.allclose(traj.points[0, 1], [1.0, 0.0, 0.0])
    assert np.allclose(traj.points[-1, 1], [3.0, 0.0, 0.0])


def test_trajectory_with_fewer_than_two_poses_is_a_noop(server):
    handles, texts, _dropdown, _cb = _build_ui(server)
    poses = np.eye(4, dtype=np.float32)[None, :, :]  # a single pose
    cmd = {"kind": "trajectory", "poses_est": poses, "poses_gt": None, "iteration": 1}
    _apply(cmd, handles, texts)  # must not raise
    assert handles["trajectory"].visible is False


def test_trajectory_trail_is_bounded_even_if_upstream_cap_is_skipped(server):
    handles, texts, _dropdown, _cb = _build_ui(server)
    n = _MAX_TRAJECTORY_TRAIL_POINTS + 200
    poses = np.stack([np.eye(4, dtype=np.float32) for _ in range(n)])
    for i in range(n):
        poses[i, 0, 3] = float(i)
    cmd = {"kind": "trajectory", "poses_est": poses, "poses_gt": None, "iteration": 1}
    _apply(cmd, handles, texts)
    traj = handles["trajectory"]
    assert traj.points.shape[0] == _MAX_TRAJECTORY_TRAIL_POINTS - 1
    # Kept the most recent tail, not the head.
    assert np.allclose(traj.points[-1, 1], [float(n - 1), 0.0, 0.0])


def test_rotmat_to_quat_wxyz_round_trips_through_existing_inverse(server):
    # Round-trip through this module's own pre-existing wxyz->rotmat helper
    # (used for map-preview covariances) to confirm the new rotmat->wxyz
    # helper agrees with it, for a handful of non-trivial rotations.
    rng = np.random.default_rng(0)
    for _ in range(20):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        R = _quat_wxyz_to_rotmat(q[None, :])[0]
        q_back = _rotmat_to_quat_wxyz(R)
        # Quaternion double-cover: q and -q represent the same rotation.
        assert np.allclose(q_back, q, atol=1e-6) or np.allclose(q_back, -q, atol=1e-6)


def test_trajectory_only_also_drives_the_live_camera_frustum(server):
    # The causal_mapping production path (eligsir/runtime/causal_entry.py)
    # only ever emits "trajectory" -- it has no per-frame image at the point
    # it observes a pose, so "raw_input" with a pose_c2w never fires on that
    # path. The live-camera frustum must still appear/move from "trajectory"
    # alone, driven by the trail's own most recent pose.
    handles, texts, _dropdown, _cb = _build_ui(server)
    assert handles["live_camera"].visible is False
    n = 3
    poses = np.stack([np.eye(4, dtype=np.float32) for _ in range(n)])
    for i in range(n):
        poses[i, 0, 3] = float(i) * 2.0
    cmd = {"kind": "trajectory", "poses_est": poses, "poses_gt": None, "iteration": 1}
    _apply(cmd, handles, texts)
    live = handles["live_camera"]
    assert live.visible is True
    assert np.allclose(live.position, [4.0, 0.0, 0.0])  # last pose's translation


def test_apply_full_end_to_end_raw_input_then_trajectory(server):
    # Both new code paths together, through the real _apply dispatch used by
    # the sidecar's drain loop -- not just each function in isolation.
    handles, texts, _dropdown, _cb = _build_ui(server)
    pose = np.eye(4, dtype=np.float32)
    pose[:3, 3] = [10.0, 0.0, 0.0]
    _apply(
        {
            "kind": "raw_input",
            "rgb": np.zeros((16, 16, 3), dtype=np.uint8),
            "source_frame_id": 1,
            "camera_id": None,
            "timestamp_s": 0.0,
            "pose_c2w": pose,
        },
        handles,
        texts,
    )
    poses = np.stack([np.eye(4, dtype=np.float32) for _ in range(3)])
    for i in range(3):
        poses[i, 0, 3] = float(i) * 5.0
    _apply(
        {"kind": "trajectory", "poses_est": poses, "poses_gt": None, "iteration": 5},
        handles,
        texts,
    )
    assert handles["live_camera"].visible is True
    assert handles["trajectory"].visible is True
    assert np.allclose(handles["live_camera"].position, [10.0, 0.0, 0.0])
    assert handles["trajectory"].points.shape == (2, 2, 3)


# ---------------------------------------------------------------------------
# submit_causal_viewer_tick: the ACTUAL causal_mapping production path.
#
# Issue #1407's original premise ("CanonicalViewerAdapter.log_camera receives
# a live pose but discards it") was true only of train_streaming.py's legacy
# call site. The live path a real `--viewer viser --execution_trainer
# causal_mapping` run takes is eligsir/runtime/causal_entry.py's
# submit_causal_viewer_tick, which -- before this change -- called only
# log_metrics and the map-output lane; it never called log_camera or
# log_trajectory at all, so no amount of fixing canonical_adapter.py alone
# could have made a live frustum/trail appear on that path.
# ---------------------------------------------------------------------------
from collections import deque

from eligsir.contracts.types import Pose as _Pose
from eligsir.runtime.causal_entry import submit_causal_viewer_tick


class _TrajFakeViewer:
    def __init__(self):
        self.metrics_calls = []
        self.trajectory_calls = []

    def log_metrics(self, metrics, iteration):
        self.metrics_calls.append((dict(metrics), int(iteration)))

    def log_trajectory(self, poses_est, poses_gt, step):
        self.trajectory_calls.append((list(poses_est), poses_gt, int(step)))

    def notify_map_output(self):
        pass


class _TrajFakeBackend:
    population_count = 1


class _TrajFakeTick:
    def __init__(self, iteration):
        self.iteration = iteration
        self.primary_training_psnr_db = None
        self.parameter_version = iteration


class _TrajFakeRuntime:
    def __init__(self, pose_matrix=None):
        self._last_admitted_pose = pose_matrix


def test_submit_causal_viewer_tick_ignores_trajectory_without_runtime_kwarg():
    # Existing callers (pre-#1407) that omit runtime/trajectory_trail must be
    # byte-identical: no trajectory attempt, no AttributeError.
    viewer = _TrajFakeViewer()
    submit_causal_viewer_tick(
        viewer, None, 0.0, _TrajFakeBackend(), _TrajFakeTick(1),
        last_map_wall_s_box=[0.0],
    )
    assert viewer.trajectory_calls == []


def test_submit_causal_viewer_tick_publishes_trajectory_from_last_admitted_pose():
    viewer = _TrajFakeViewer()
    pose = np.eye(4, dtype=np.float32)
    pose[0, 3] = 7.0
    runtime = _TrajFakeRuntime(pose_matrix=pose)
    trail = deque(maxlen=100)

    submit_causal_viewer_tick(
        viewer, None, 0.0, _TrajFakeBackend(), _TrajFakeTick(50),
        last_map_wall_s_box=[0.0], runtime=runtime, trajectory_trail=trail,
    )

    assert len(trail) == 1
    assert isinstance(trail[0], _Pose)
    assert np.allclose(trail[0].matrix, pose)
    assert len(viewer.trajectory_calls) == 1
    poses_est, poses_gt, step = viewer.trajectory_calls[0]
    assert len(poses_est) == 1
    assert poses_gt is None
    assert step == 50


def test_submit_causal_viewer_tick_skips_unchanged_pose():
    viewer = _TrajFakeViewer()
    pose = np.eye(4, dtype=np.float32)
    pose[0, 3] = 3.0
    runtime = _TrajFakeRuntime(pose_matrix=pose)
    trail = deque(maxlen=100)

    for it in (1, 2, 3):
        submit_causal_viewer_tick(
            viewer, None, 0.0, _TrajFakeBackend(), _TrajFakeTick(it),
            last_map_wall_s_box=[0.0], runtime=runtime, trajectory_trail=trail,
        )

    # Same _last_admitted_pose every tick (admission-driven, not tick-driven)
    # -> only the FIRST tick actually appends/publishes.
    assert len(trail) == 1
    assert len(viewer.trajectory_calls) == 1


def test_submit_causal_viewer_tick_appends_on_real_pose_change():
    viewer = _TrajFakeViewer()
    pose_a = np.eye(4, dtype=np.float32)
    pose_a[0, 3] = 1.0
    pose_b = np.eye(4, dtype=np.float32)
    pose_b[0, 3] = 2.0
    runtime = _TrajFakeRuntime(pose_matrix=pose_a)
    trail = deque(maxlen=100)

    submit_causal_viewer_tick(
        viewer, None, 0.0, _TrajFakeBackend(), _TrajFakeTick(1),
        last_map_wall_s_box=[0.0], runtime=runtime, trajectory_trail=trail,
    )
    runtime._last_admitted_pose = pose_b
    submit_causal_viewer_tick(
        viewer, None, 0.0, _TrajFakeBackend(), _TrajFakeTick(2),
        last_map_wall_s_box=[0.0], runtime=runtime, trajectory_trail=trail,
    )

    assert len(trail) == 2
    assert len(viewer.trajectory_calls) == 2
    assert viewer.trajectory_calls[-1][0][-1].matrix[0, 3] == pytest.approx(2.0)


def test_submit_causal_viewer_tick_trajectory_trail_respects_its_own_maxlen():
    viewer = _TrajFakeViewer()
    trail = deque(maxlen=3)
    runtime = _TrajFakeRuntime()
    for i in range(5):
        pose = np.eye(4, dtype=np.float32)
        pose[0, 3] = float(i)
        runtime._last_admitted_pose = pose
        submit_causal_viewer_tick(
            viewer, None, 0.0, _TrajFakeBackend(), _TrajFakeTick(i),
            last_map_wall_s_box=[0.0], runtime=runtime, trajectory_trail=trail,
        )
    assert len(trail) == 3
    assert trail[-1].matrix[0, 3] == pytest.approx(4.0)
    assert trail[0].matrix[0, 3] == pytest.approx(2.0)  # oldest two evicted


def test_submit_causal_viewer_tick_trajectory_never_raises_on_viewer_exception():
    class _RaisingViewer(_TrajFakeViewer):
        def log_trajectory(self, poses_est, poses_gt, step):
            raise RuntimeError("boom")

    viewer = _RaisingViewer()
    pose = np.eye(4, dtype=np.float32)
    runtime = _TrajFakeRuntime(pose_matrix=pose)
    trail = deque(maxlen=10)

    # Must not raise, and log_metrics must still have run beforehand.
    submit_causal_viewer_tick(
        viewer, None, 0.0, _TrajFakeBackend(), _TrajFakeTick(1),
        last_map_wall_s_box=[0.0], runtime=runtime, trajectory_trail=trail,
    )
    assert viewer.metrics_calls != []


def test_submit_causal_viewer_tick_no_pose_yet_does_nothing():
    viewer = _TrajFakeViewer()
    runtime = _TrajFakeRuntime(pose_matrix=None)  # pre-admission state
    trail = deque(maxlen=10)

    submit_causal_viewer_tick(
        viewer, None, 0.0, _TrajFakeBackend(), _TrajFakeTick(1),
        last_map_wall_s_box=[0.0], runtime=runtime, trajectory_trail=trail,
    )
    assert trail == deque()
    assert viewer.trajectory_calls == []
