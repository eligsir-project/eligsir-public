"""Tests for issue #1382: wiring the canonical viewer into causal_mapping.

Gap: `train.py`'s causal_mapping branch returns from
`eligsir.runtime.causal_entry.run_causal_mapping` before `train.py` ever
reaches its own `make_viewer()` call, so `--viewer viser` had zero effect on
this execution path. `eligsir/runtime/causal_entry.py`'s
`build_causal_viewer`/`submit_causal_viewer_tick` close that gap, gated
behind `--viewer`/`--viewer_sinks` with zero behavior change when unset.

Two contracts exercised here, CPU-only (no real GPU/viser server, no
torch.cuda):
  (a) viewer disabled (the default) is a true no-op at both call sites --
      the regression guard for "byte-identical when the flag is absent".
  (b) viewer enabled fires the expected hooks with the expected data shapes,
      using a recording PreviewSink fake instead of a real Viser server.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from eligsir.runtime.causal_entry import build_causal_viewer, submit_causal_viewer_tick
from eligsir.viewers.canonical_adapter import CanonicalViewerAdapter
from eligsir.viewers.sinks.base import PreviewSink


def _gsplat_model(n: int, device: str = "cpu"):
    x = torch.arange(n, dtype=torch.float32, device=device)
    return SimpleNamespace(
        params={
            "means": torch.stack((x, x + 100, x + 200), 1),
            "scales": torch.full((n, 3), -2.0, device=device),
            "quats": torch.nn.functional.one_hot(
                torch.zeros(n, dtype=torch.long, device=device), 4
            ).float(),
            "opacities": torch.full((n,), 1.0, device=device),
            "sh0": torch.zeros((n, 1, 3), device=device),
            "shN": torch.zeros((n, 0, 3), device=device),
        }
    )


class _FakeBackend:
    """Duck-types eligsir.map.contracts.GaussianMapBackend."""

    def __init__(self, n: int = 6, device: str = "cpu") -> None:
        self.model = _gsplat_model(n, device=device)
        self.content = 1
        self.topology = 1
        self.population_count = n
        self.sample_calls = 0

    def read_view(self):
        # A real GaussianMapReadView needs a full GsplatGaussianModel-shaped
        # object (get_xyz/get_scaling/... accessor properties); this test
        # double only exercises the ``.means.device`` read build_causal_viewer
        # actually performs, so a minimal stand-in is sufficient and avoids
        # reimplementing the real model's activation accessors here.
        return SimpleNamespace(means=self.model.params["means"])

    def sample_output(self, *, max_gaussians, parameter_version):
        from eligsir.map.views import sample_output_from_model

        self.sample_calls += 1
        return sample_output_from_model(
            self.model,
            max_gaussians=max_gaussians,
            map_content_version=self.content,
            topology_version=self.topology,
            parameter_version=parameter_version,
        )

    @property
    def map_content_version(self):
        return self.content

    @property
    def map_version(self):
        return self.content

    @property
    def topology_version(self):
        return self.topology


class _FakeTick:
    def __init__(self, iteration: int, psnr_db: float | None, parameter_version: int) -> None:
        self.executed = True
        self.iteration = iteration
        self.primary_training_psnr_db = psnr_db
        self.parameter_version = parameter_version
        self.supervision_views = 1


class _RecordingSink(PreviewSink):
    def __init__(self) -> None:
        self.metrics = []
        self.map_snapshots = []

    def on_metrics(self, envelope):
        self.metrics.append(envelope.event)

    def on_map_snapshot(self, envelope):
        self.map_snapshots.append(envelope.event)


class _Args:
    """Minimal args namespace exercising only what build_causal_viewer reads."""

    def __init__(self, viewer: str = "none", viewer_sinks: str = "") -> None:
        self.viewer = viewer
        self.viewer_sinks = viewer_sinks
        self.viewer_viser_port = 8080
        self.viewer_viser_host = "0.0.0.0"


# -- (a) disabled is a true no-op --------------------------------------------


def test_build_causal_viewer_disabled_returns_none_triple():
    args = _Args(viewer="none", viewer_sinks="")
    runtime = SimpleNamespace(backend=_FakeBackend())
    viewer, lane, map_max_hz = build_causal_viewer(args, runtime)
    assert viewer is None
    assert lane is None
    assert map_max_hz == 0.0


def test_submit_causal_viewer_tick_noop_when_viewer_none():
    # Must not raise and must not touch backend/tick at all when viewer is None.
    backend = _FakeBackend()
    tick = _FakeTick(iteration=100, psnr_db=25.0, parameter_version=3)
    submit_causal_viewer_tick(
        None, None, 0.0, backend, tick, last_map_wall_s_box=[0.0]
    )
    assert backend.sample_calls == 0


# -- (b) enabled fires the expected hooks with expected shapes ---------------


def test_build_causal_viewer_viser_wires_adapter_and_map_output_lane(monkeypatch):
    sink = _RecordingSink()

    def _fake_make_viewer(name, *, viewer_sinks=None, **kwargs):
        assert name == "viser"
        # Real construction would build a ViserPreviewSink (imports the viser
        # package and binds a real server socket) -- substitute a recording
        # fake sink so this test never touches a real network port.
        return CanonicalViewerAdapter(sinks=[sink], drain_poll_interval_s=0.01)

    # build_causal_viewer does `from eligsir.viewers import make_viewer`
    # lazily inside its own body -- patching the attribute on the source
    # module is what a fresh `from ... import` picks up at call time.
    import eligsir.viewers as viewers_pkg

    monkeypatch.setattr(viewers_pkg, "make_viewer", _fake_make_viewer, raising=True)

    args = _Args(viewer="viser", viewer_sinks="")
    backend = _FakeBackend(n=6)
    runtime = SimpleNamespace(backend=backend)
    viewer, lane, map_max_hz = build_causal_viewer(args, runtime)
    try:
        assert viewer is not None
        assert lane is not None
        assert map_max_hz > 0.0
    finally:
        viewer.close()


def test_submit_causal_viewer_tick_publishes_metrics_and_map_snapshot():
    sink = _RecordingSink()
    viewer = CanonicalViewerAdapter(sinks=[sink], drain_poll_interval_s=0.01)
    backend = _FakeBackend(n=6)
    try:
        from eligsir.runtime.transfers import get_device_transfer_owner
        from eligsir.viewers.map_output_lane import VersionedMapOutputLane

        lane = VersionedMapOutputLane(
            get_device_transfer_owner(backend.read_view().means.device),
            max_gaussians=100,
        )
        viewer.bind_map_output_lane(lane)

        tick = _FakeTick(iteration=200, psnr_db=24.5, parameter_version=7)
        submit_causal_viewer_tick(
            viewer, lane, 0.0, backend, tick, last_map_wall_s_box=[0.0]
        )

        deadline_iterations = 200
        for _ in range(deadline_iterations):
            if sink.metrics and sink.map_snapshots:
                break
            import time

            time.sleep(0.01)

        assert backend.sample_calls == 1  # exactly one bounded sample, no extra work
        assert len(sink.metrics) == 1
        metrics_event = sink.metrics[0]
        metrics_dict = dict(metrics_event.metrics)
        assert metrics_event.iteration == 200
        assert metrics_dict["psnr"] == pytest.approx(24.5)
        assert metrics_dict["gaussian_count"] == pytest.approx(6.0)

        assert len(sink.map_snapshots) == 1
    finally:
        viewer.close()
