"""Tests for eligsir/viewers/canonical_adapter.py (issue #570 AC1 migration, #572).

Exercises the FULL real call path a training run takes when
``--viewer viser`` (or any sink) is selected: adapter -> staging ring ->
drain thread -> CanonicalViewer.submit -> drain_to_sinks -> sink hook. Uses a
recording ``PreviewSink`` (not a mock of internals) so the assertions are
about observable behaviour, not implementation.
"""
from __future__ import annotations

import threading
import time

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from eligsir.viewers.canonical_adapter import CanonicalViewerAdapter
from eligsir.viewers.events import MappingPhase
from eligsir.viewers.sinks.base import PreviewSink


class _RecordingSink(PreviewSink):
    def __init__(self):
        self.pairs = []
        self.metrics = []
        self.map_snapshots = []
        self._lock = threading.Lock()

    def on_training_pair(self, envelope):
        with self._lock:
            self.pairs.append(envelope.event)

    def on_metrics(self, envelope):
        with self._lock:
            self.metrics.append(envelope.event)

    def on_map_snapshot(self, envelope):
        with self._lock:
            self.map_snapshots.append(envelope.event)


def _wait_for(pred, timeout_s=2.0):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(0.01)
    return False


@pytest.fixture()
def adapter_and_sink():
    sink = _RecordingSink()
    adapter = CanonicalViewerAdapter(sinks=[sink], drain_poll_interval_s=0.01)
    yield adapter, sink
    adapter.close()


def test_training_pair_tensors_reach_the_sink_matched_and_unblocked(adapter_and_sink):
    adapter, sink = adapter_and_sink
    target = torch.full((8, 8, 3), 0.1, dtype=torch.float32)
    render = torch.full((8, 8, 3), 0.9, dtype=torch.float32)

    start = time.monotonic()
    adapter.log_training_pair_tensors(
        target=target,
        render=render,
        camera_id=42,
        iteration=1000,
        source_frame_id=17,
        camera_role="NEW",
        n_gaussians=5000,
        psnr_db=24.1,
    )
    elapsed = start - time.monotonic()
    assert abs(elapsed) < 0.2  # must return immediately, no sink work on caller thread

    assert _wait_for(lambda: len(sink.pairs) == 1)
    event = sink.pairs[0]
    assert event.camera_id == 42
    assert event.render.camera_id == 42
    assert event.target.camera_id == 42
    assert event.source_frame_id == 17
    assert event.camera_role.value == "NEW"
    assert event.n_gaussians == 5000
    assert event.psnr_db == pytest.approx(24.1)
    assert np.allclose(event.target.array, 0.1)
    assert np.allclose(event.render.array, 0.9)


def test_log_metrics_reaches_the_sink():
    sink = _RecordingSink()
    adapter = CanonicalViewerAdapter(sinks=[sink], drain_poll_interval_s=0.01)
    try:
        adapter.log_metrics({"psnr": 20.0, "l1_loss": 0.05}, 500)
        assert _wait_for(lambda: len(sink.metrics) == 1)
        assert sink.metrics[0].as_dict()["psnr"] == 20.0
    finally:
        adapter.close()


def test_full_staging_ring_drops_a_pair_without_raising_or_blocking():
    sink = _RecordingSink()
    # Tiny ring + no drain notify wait -> force a drop deterministically by
    # submitting faster than the (slow) poll interval can drain.
    adapter = CanonicalViewerAdapter(sinks=[sink], staging_slots=2, drain_poll_interval_s=5.0)
    try:
        for i in range(10):
            adapter.log_training_pair_tensors(
                target=torch.zeros((4, 4, 3)),
                render=torch.zeros((4, 4, 3)),
                camera_id=1,
                iteration=i,
            )
        # No exception raised above is the primary assertion (never blocks/crashes).
        stats = adapter._ring.stats()
        assert stats["staging_staged"] + stats["staging_dropped"] == 10
    finally:
        adapter.close()


def test_versioned_map_output_lane_reaches_sink_off_thread(adapter_and_sink):
    from types import SimpleNamespace

    from eligsir.map.views import GaussianMapOutputSample
    from eligsir.runtime.transfers import DeviceTransferOwner
    from eligsir.viewers.map_output_lane import VersionedMapOutputLane

    adapter, sink = adapter_and_sink
    n = 10
    view = SimpleNamespace(
        means=torch.zeros((n, 3)),
        scales=torch.full((n, 3), 0.03),
        quaternions=torch.nn.functional.one_hot(torch.zeros(n, dtype=torch.long), 4).float(),
        opacities=torch.full((n, 1), 0.7),
        sh0=torch.zeros((n, 1, 3)),
        active_count=n,
        map_version=3,
        topology_version=2,
    )
    class Backend:
        map_content_version = map_version = 3
        topology_version = 2
        def sample_output(self, *, max_gaussians, parameter_version):
            selected = slice(0, max_gaussians)
            return GaussianMapOutputSample(
                view.means[selected].clone(),
                torch.full((max_gaussians, 3), 0.5),
                view.scales[selected].clone(), view.quaternions[selected].clone(),
                view.opacities[selected].clone(), n, 3, 2, parameter_version,
            )
    owner = DeviceTransferOwner("cpu", max_inflight=2, max_pinned_bytes=1_000_000)
    lane = VersionedMapOutputLane(owner, max_gaussians=4)
    adapter.bind_map_output_lane(lane)

    start = time.monotonic()
    assert lane.try_submit(Backend(), parameter_version=7, iteration=2000)
    adapter.notify_map_output()
    assert time.monotonic() - start < 0.2

    assert _wait_for(lambda: len(sink.map_snapshots) == 1)
    event = sink.map_snapshots[0]
    assert event.n_sampled == 4
    assert event.n_total_gaussians == 10
    assert event.map_version == 3
    assert event.scales.shape == (4, 3)
    assert event.quats.shape == (4, 4)
    assert event.opacities.shape == (4, 1)
    assert owner.accounting().d2h_ops == 1


def test_map_output_lane_can_only_be_bound_once(adapter_and_sink):
    from eligsir.runtime.transfers import DeviceTransferOwner
    from eligsir.viewers.map_output_lane import VersionedMapOutputLane

    adapter, _sink = adapter_and_sink
    owner = DeviceTransferOwner("cpu", max_inflight=2, max_pinned_bytes=1_000_000)
    first = VersionedMapOutputLane(owner, max_gaussians=4)
    adapter.bind_map_output_lane(first)
    adapter.bind_map_output_lane(first)
    with pytest.raises(RuntimeError, match="already has"):
        adapter.bind_map_output_lane(VersionedMapOutputLane(owner, max_gaussians=4))


def test_close_is_idempotent_and_stops_the_drain_thread():
    sink = _RecordingSink()
    adapter = CanonicalViewerAdapter(sinks=[sink])
    adapter.close()
    adapter.close()  # must not raise
    assert adapter._drain._thread is None
