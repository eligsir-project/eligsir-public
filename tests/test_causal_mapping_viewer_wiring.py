"""Issue #1383: wire live Viser viewing into the causal_mapping execution path.

(a) proves viewer-off is a true no-op: ``build_causal_viewer`` returns
    ``(None, None, 0.0)`` without ever importing/constructing a real viewer or
    touching the map backend -- the same guarantee ``run_causal_mapping``'s
    own gating (``if _viewer is not None: ...``) relies on to leave
    causal_mapping's iteration count/PSNR/any numeric output byte-identical
    when ``--viewer`` is left at its default.
(b) proves the per-tick hook fires with the expected data shapes when a
    viewer IS configured, using fakes for the viewer/sink/backend -- no real
    viser server, no GPU, no torch tensor needed.
"""

from __future__ import annotations

import sys
from types import ModuleType

import pytest

from eligsir.runtime.causal_entry import build_causal_viewer, submit_causal_viewer_tick


def _install_transfer_owner_fake(monkeypatch, factory) -> None:
    """Provide the GPU-only transfer module without importing Torch in smoke."""
    module = ModuleType("eligsir.runtime.transfers")
    module.get_device_transfer_owner = factory
    monkeypatch.setitem(sys.modules, "eligsir.runtime.transfers", module)


class _Args:
    def __init__(self, **overrides):
        self.viewer = "none"
        self.viewer_sinks = ""
        self.viewer_viser_port = 8080
        self.viewer_viser_host = "0.0.0.0"
        for key, value in overrides.items():
            setattr(self, key, value)


class _FakeRuntime:
    """A runtime whose .backend must never be touched when the viewer is off."""

    def __init__(self):
        self.backend = _ExplodingBackend()


class _ExplodingBackend:
    def read_view(self):  # pragma: no cover - must never be called
        raise AssertionError("backend.read_view() called with the viewer disabled")


# --- (a) viewer-off is a true no-op --------------------------------------


def test_build_causal_viewer_returns_none_triple_when_viewer_is_none():
    viewer, lane, hz = build_causal_viewer(_Args(viewer="none"), _FakeRuntime())
    assert (viewer, lane, hz) == (None, None, 0.0)


def test_build_causal_viewer_returns_none_triple_when_no_flags_set_at_all():
    """Default-constructed args (no --viewer, no --viewer_sinks) changes nothing."""
    viewer, lane, hz = build_causal_viewer(_Args(), _FakeRuntime())
    assert (viewer, lane, hz) == (None, None, 0.0)


def test_build_causal_viewer_never_touches_backend_when_disabled():
    # _ExplodingBackend.read_view() raises if called; a clean return proves
    # the disabled path never reaches into the map backend at all.
    runtime = _FakeRuntime()
    build_causal_viewer(_Args(viewer="none"), runtime)


def test_submit_causal_viewer_tick_is_a_true_no_op_when_viewer_is_none():
    calls = []

    class _ExplodingLane:
        def try_submit(self, *a, **k):  # pragma: no cover
            calls.append(("try_submit", a, k))
            raise AssertionError("map_output_lane touched with viewer disabled")

    submit_causal_viewer_tick(
        None, _ExplodingLane(), 0.5, object(), object(), last_map_wall_s_box=[0.0]
    )
    assert calls == []


# --- (b) hooks fire with expected shapes when a viewer is configured -----


class _FakeCanonicalConfig:
    map_max_gaussians = 100_000
    map_max_hz = 0.5


class _FakeCanonicalViewer:
    config = _FakeCanonicalConfig()


class _FakeViewer:
    """Stands in for CanonicalViewerAdapter -- records every call it receives."""

    def __init__(self):
        self.canonical_viewer = _FakeCanonicalViewer()
        self.metrics_calls = []
        self.map_output_lane = None
        self.notify_calls = 0
        self.closed = False

    def bind_map_output_lane(self, lane):
        self.map_output_lane = lane

    def log_metrics(self, metrics, iteration):
        self.metrics_calls.append((dict(metrics), int(iteration)))

    def notify_map_output(self):
        self.notify_calls += 1

    def close(self):
        self.closed = True


class _FakeMapOutputLane:
    def __init__(self):
        self.submissions = []

    def try_submit(self, backend, *, parameter_version, iteration):
        self.submissions.append((backend, parameter_version, iteration))
        return True


class _FakeBackend:
    population_count = 4242


class _FakeTick:
    def __init__(self, iteration, psnr_db=None, parameter_version=None):
        self.iteration = iteration
        self.primary_training_psnr_db = psnr_db
        self.parameter_version = parameter_version if parameter_version is not None else iteration


def test_submit_causal_viewer_tick_logs_metrics_with_expected_shape():
    viewer = _FakeViewer()
    backend = _FakeBackend()
    tick = _FakeTick(iteration=17, psnr_db=24.5, parameter_version=17)

    submit_causal_viewer_tick(
        viewer, None, 0.0, backend, tick, last_map_wall_s_box=[0.0]
    )

    assert viewer.metrics_calls == [
        ({"gaussian_count": 4242.0, "psnr": 24.5}, 17)
    ]


def test_submit_causal_viewer_tick_omits_psnr_key_when_tick_has_none():
    viewer = _FakeViewer()
    backend = _FakeBackend()
    tick = _FakeTick(iteration=3, psnr_db=None)

    submit_causal_viewer_tick(
        viewer, None, 0.0, backend, tick, last_map_wall_s_box=[0.0]
    )

    metrics, iteration = viewer.metrics_calls[0]
    assert iteration == 3
    assert metrics == {"gaussian_count": 4242.0}


def test_submit_causal_viewer_tick_submits_map_output_and_notifies():
    viewer = _FakeViewer()
    lane = _FakeMapOutputLane()
    backend = _FakeBackend()
    tick = _FakeTick(iteration=100, parameter_version=100)

    submit_causal_viewer_tick(
        viewer, lane, 0.0, backend, tick, last_map_wall_s_box=[0.0]
    )

    assert lane.submissions == [(backend, 100, 100)]
    assert viewer.notify_calls == 1


def test_submit_causal_viewer_tick_respects_map_max_hz_rate_limit():
    import time

    viewer = _FakeViewer()
    lane = _FakeMapOutputLane()
    backend = _FakeBackend()
    # "Last submitted" just now -- well inside the 2s window (1/0.5Hz), so
    # this call must be rate-limited and skip the map submission.
    box = [time.monotonic()]

    submit_causal_viewer_tick(
        viewer, lane, 0.5, backend, _FakeTick(iteration=1), last_map_wall_s_box=box
    )

    assert lane.submissions == []
    assert viewer.notify_calls == 0


def test_submit_causal_viewer_tick_never_raises_on_viewer_exception():
    class _RaisingViewer(_FakeViewer):
        def log_metrics(self, metrics, iteration):
            raise RuntimeError("boom")

    viewer = _RaisingViewer()
    lane = _FakeMapOutputLane()
    submit_causal_viewer_tick(
        viewer, lane, 0.0, _FakeBackend(), _FakeTick(iteration=1),
        last_map_wall_s_box=[0.0],
    )
    # Reaching here (no exception propagated past log_metrics's failure) is
    # the assertion; the map submission after it must still have run.
    assert len(lane.submissions) == 1


def test_build_causal_viewer_binds_map_output_lane_for_canonical_viewer(monkeypatch):
    """--viewer viser (mocked make_viewer/VersionedMapOutputLane) binds a lane."""
    import eligsir.runtime.causal_entry as causal_entry

    fake_viewer = _FakeViewer()

    def _fake_make_viewer(name, *, viewer_sinks=None, **kwargs):
        assert name == "viser"
        assert kwargs.get("port") == 9999
        assert kwargs.get("host") == "127.0.0.1"
        return fake_viewer

    fake_lane_instances = []

    class _FakeVersionedMapOutputLane:
        def __init__(self, transfer_owner, *, max_gaussians):
            self.transfer_owner = transfer_owner
            self.max_gaussians = max_gaussians
            fake_lane_instances.append(self)

    fake_device = object()

    class _FakeReadView:
        class _Means:
            device = fake_device

        means = _Means()

    class _FakeBackendWithView:
        def read_view(self):
            return _FakeReadView()

    class _FakeRuntimeWithBackend:
        backend = _FakeBackendWithView()

    monkeypatch.setattr(
        "eligsir.viewers.make_viewer", _fake_make_viewer, raising=False
    )
    monkeypatch.setattr(
        "eligsir.viewers.map_output_lane.VersionedMapOutputLane",
        _FakeVersionedMapOutputLane,
        raising=False,
    )
    _install_transfer_owner_fake(monkeypatch, lambda device, **kw: device)
    monkeypatch.setattr(causal_entry, "_resolve_viewer_device", lambda *_: fake_device)

    args = _Args(viewer="viser", viewer_viser_port=9999, viewer_viser_host="127.0.0.1")
    viewer, lane, hz = build_causal_viewer(args, _FakeRuntimeWithBackend())

    assert viewer is fake_viewer
    assert lane is not None
    assert lane is fake_viewer.map_output_lane
    assert hz == 0.5


# --- regression: issue #1393, pre-bootstrap model must not crash ----------


def test_build_causal_viewer_does_not_crash_on_prebootstrap_model(monkeypatch):
    """A freshly constructed CanonicalMappingRuntime has zero Gaussians until
    the first admitted/ticked frame bootstraps the model -- ``params`` is an
    empty ``nn.ParameterDict`` with no ``means``/``sh0`` keys yet.
    ``build_causal_viewer`` must not call ``backend.read_view()`` (which
    raises ``AttributeError: 'ParameterDict' object has no attribute
    'sh0'``) to resolve a device at construction time; it must fall back to
    the configured ``--data_device`` instead.
    """
    torch = pytest.importorskip("torch")

    fake_viewer = _FakeViewer()

    def _fake_make_viewer(name, *, viewer_sinks=None, **kwargs):
        return fake_viewer

    class _FakeVersionedMapOutputLane:
        def __init__(self, transfer_owner, *, max_gaussians):
            self.transfer_owner = transfer_owner
            self.max_gaussians = max_gaussians

    monkeypatch.setattr(
        "eligsir.viewers.make_viewer", _fake_make_viewer, raising=False
    )
    monkeypatch.setattr(
        "eligsir.viewers.map_output_lane.VersionedMapOutputLane",
        _FakeVersionedMapOutputLane,
        raising=False,
    )
    monkeypatch.setattr(
        "eligsir.runtime.transfers.get_device_transfer_owner",
        lambda device, **kw: device,
        raising=False,
    )

    class _PreBootstrapModel:
        """Mirrors GsplatGaussianModel.__init__'s state: an empty
        nn.ParameterDict with no keys at all until create_from_pcd bootstraps
        it on the first ticked frame."""

        def __init__(self):
            self.params = torch.nn.ParameterDict()

    class _PreBootstrapBackend:
        """Mirrors GsplatMapBackend: owns ``_model`` and a ``read_view()``
        that raises exactly like the real one does pre-bootstrap."""

        def __init__(self):
            self._model = _PreBootstrapModel()

        def read_view(self):  # pragma: no cover - must never be called
            raise AttributeError("'ParameterDict' object has no attribute 'sh0'")

    class _PreBootstrapRuntime:
        backend = _PreBootstrapBackend()

    args = _Args(viewer="viser", data_device="cpu")

    # Must not raise.
    viewer, lane, hz = build_causal_viewer(args, _PreBootstrapRuntime())

    assert viewer is fake_viewer
    assert lane is not None
    assert lane.transfer_owner == torch.device("cpu")


# --- regression: follow-up found 2026-08-31, CUDA fallback must be indexed --


def test_resolve_viewer_device_normalizes_bare_cuda_to_indexed_device(monkeypatch):
    """A bare ``torch.device("cuda")`` fallback silently breaks the map
    preview lane: ``VersionedMapOutputLane._start_request`` later compares
    this device for exact equality against a REAL sampled tensor's
    ``.device`` -- and PyTorch device equality is index-sensitive
    (``torch.device("cuda") != torch.device("cuda", 0)`` even though both
    name the same physical GPU). A real GPU run (2026-08-31) confirmed this:
    ``try_submit`` reported success on every call, but
    ``VersionedMapOutputLane.poll_completed()`` never returned a snapshot --
    Gaussians never reached the viewer. ``_resolve_viewer_device`` must
    normalize its CUDA fallback to the concrete current-device index so this
    equality check can ever pass once the model bootstraps.
    """
    torch = pytest.importorskip("torch")

    from eligsir.runtime.causal_entry import _resolve_viewer_device

    class _PreBootstrapModel:
        def __init__(self):
            self.params = torch.nn.ParameterDict()

    class _PreBootstrapBackend:
        def __init__(self):
            self._model = _PreBootstrapModel()

    class _PreBootstrapRuntime:
        backend = _PreBootstrapBackend()

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True, raising=False)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0, raising=False)

    args = _Args(viewer="viser", data_device="cuda")
    device = _resolve_viewer_device(args, _PreBootstrapRuntime())

    assert device == torch.device("cuda", 0)
    assert device.index is not None
    # The exact regression: a bare device never equals an indexed one, even
    # though they name the same GPU.
    assert torch.device("cuda") != torch.device("cuda", 0)


def test_resolve_viewer_device_cpu_fallback_stays_unindexed(monkeypatch):
    """A CPU fallback must not be perturbed by the CUDA-only normalization
    (``torch.device("cpu")`` has no index concept at all)."""
    torch = pytest.importorskip("torch")

    from eligsir.runtime.causal_entry import _resolve_viewer_device

    class _PreBootstrapModel:
        def __init__(self):
            self.params = torch.nn.ParameterDict()

    class _PreBootstrapBackend:
        def __init__(self):
            self._model = _PreBootstrapModel()

    class _PreBootstrapRuntime:
        backend = _PreBootstrapBackend()

    args = _Args(viewer="viser", data_device="cpu")
    device = _resolve_viewer_device(args, _PreBootstrapRuntime())

    assert device == torch.device("cpu")


# --- regression: build_causal_viewer must match CanonicalMappingRuntime's --
# --- later get_device_transfer_owner byte-limit request (found 2026-08-31) -


def test_build_causal_viewer_passes_streaming_max_transfer_bytes_to_owner(monkeypatch):
    """``get_device_transfer_owner`` is a process-wide singleton registry
    keyed by device string (``eligsir/runtime/transfers.py``): the FIRST
    caller for a device fixes its ``max_pinned_bytes`` for every later
    caller, which raises ``RuntimeError`` on a mismatch. ``build_causal_viewer``
    runs before ``CanonicalMappingRuntime``'s own lazy ``_initialize()``
    (``production_application.py``), which requests the same device with
    ``args.streaming_max_transfer_bytes`` -- so build_causal_viewer MUST pass
    that identical value through, not silently take the function's own
    unrelated default. A real GPU run hit exactly this after #1393's device
    fix made both callers agree on the same device key for the first time:
    ``RuntimeError: DeviceTransferOwner already configured with a different
    pinned byte limit: existing=2147483648 requested=671088640``.
    """
    import eligsir.runtime.causal_entry as causal_entry

    captured_kwargs = {}

    def _fake_get_device_transfer_owner(device, **kwargs):
        captured_kwargs.update(kwargs)
        return device

    fake_viewer = _FakeViewer()

    def _fake_make_viewer(name, *, viewer_sinks=None, **kwargs):
        return fake_viewer

    class _FakeVersionedMapOutputLane:
        def __init__(self, transfer_owner, *, max_gaussians):
            self.transfer_owner = transfer_owner
            self.max_gaussians = max_gaussians

    monkeypatch.setattr(
        "eligsir.viewers.make_viewer", _fake_make_viewer, raising=False
    )
    monkeypatch.setattr(
        "eligsir.viewers.map_output_lane.VersionedMapOutputLane",
        _FakeVersionedMapOutputLane,
        raising=False,
    )
    _install_transfer_owner_fake(monkeypatch, _fake_get_device_transfer_owner)
    monkeypatch.setattr(causal_entry, "_resolve_viewer_device", lambda *_: "cpu")

    class _FakeBackend:
        def read_view(self):  # pragma: no cover
            raise AssertionError("must not be called")

    class _FakeRuntime:
        backend = _FakeBackend()

    args = _Args(viewer="viser", data_device="cpu", streaming_max_transfer_bytes=671088640)
    causal_entry.build_causal_viewer(args, _FakeRuntime())

    assert captured_kwargs.get("max_pinned_bytes") == 671088640


# --- regression: runtime telemetry panels were never wired at all --------


class _FakeViewerWithTelemetry(_FakeViewer):
    """Extends _FakeViewer with the runtime-telemetry hook (found 2026-08-31:
    CanonicalViewerAdapter.log_runtime_telemetry_snapshot had ZERO production
    callers anywhere in the repo -- neither causal_entry.py nor
    train_streaming.py ever called it, so the viewer's
    Decision/Freshness/GPU-Transfer/Map-Residency panels stayed permanently
    on their "waiting for..." placeholder regardless of --viewer)."""

    def __init__(self):
        super().__init__()
        self.telemetry_calls = []

    def log_runtime_telemetry_snapshot(self, snapshot, *, iteration, wall_time_s=None):
        self.telemetry_calls.append((snapshot, int(iteration)))


def test_submit_causal_viewer_tick_forwards_telemetry_when_present():
    viewer = _FakeViewerWithTelemetry()
    backend = _FakeBackend()
    sentinel_snapshot = object()
    tick = _FakeTick(iteration=42)
    tick.telemetry = sentinel_snapshot

    submit_causal_viewer_tick(
        viewer, None, 0.0, backend, tick, last_map_wall_s_box=[0.0]
    )

    assert viewer.telemetry_calls == [(sentinel_snapshot, 42)]


def test_submit_causal_viewer_tick_skips_telemetry_when_tick_has_none():
    """A tick with no telemetry attribute at all (or telemetry=None) must not
    call the hook -- and must not raise either."""
    viewer = _FakeViewerWithTelemetry()
    backend = _FakeBackend()
    tick = _FakeTick(iteration=7)  # no .telemetry attribute set

    submit_causal_viewer_tick(
        viewer, None, 0.0, backend, tick, last_map_wall_s_box=[0.0]
    )

    assert viewer.telemetry_calls == []


def test_submit_causal_viewer_tick_skips_telemetry_when_viewer_lacks_hook():
    """A viewer/adapter without log_runtime_telemetry_snapshot at all (e.g. an
    older or minimal viewer stub) must not raise via getattr/hasattr."""
    viewer = _FakeViewer()  # base fake has no log_runtime_telemetry_snapshot
    backend = _FakeBackend()
    tick = _FakeTick(iteration=7)
    tick.telemetry = object()

    # Must not raise.
    submit_causal_viewer_tick(
        viewer, None, 0.0, backend, tick, last_map_wall_s_box=[0.0]
    )


def test_submit_causal_viewer_tick_never_raises_on_telemetry_exception():
    class _RaisingTelemetryViewer(_FakeViewerWithTelemetry):
        def log_runtime_telemetry_snapshot(self, snapshot, *, iteration, wall_time_s=None):
            raise RuntimeError("boom")

    viewer = _RaisingTelemetryViewer()
    backend = _FakeBackend()
    tick = _FakeTick(iteration=7)
    tick.telemetry = object()

    # Must not raise, and metrics must still have been logged before this.
    submit_causal_viewer_tick(
        viewer, None, 0.0, backend, tick, last_map_wall_s_box=[0.0]
    )
    assert len(viewer.metrics_calls) == 1
