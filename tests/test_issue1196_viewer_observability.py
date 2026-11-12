from __future__ import annotations

from queue import Empty, Full
import multiprocessing as mp
from types import SimpleNamespace

from eligsir.viewers.sidecar_ipc import LatestWinsSidecarIPC
from eligsir.viewers.telemetry_panels import assemble_viewer_telemetry
from eligsir.viewers.events import PreviewChannel, RuntimeTelemetryPreview, channel_for_event
from eligsir.viewers.sinks.viser_server import _apply


def _ns(**kwargs):
    return SimpleNamespace(**kwargs)


def _snapshot(*, completed=True):
    source = _ns(source_id="cam", stream_epoch=2, source_ordinal=17, timestamp_ns=99)
    disposition = _ns(
        source=source,
        kind=_ns(value="admit"),
        observe=True,
        mapping_candidate=True,
        admit_supervision=True,
        allow_geometry_insertion=True,
        residency=_ns(value="hot"),
        reason="newest frontier",
    )
    plan = _ns(
        plan_id="plan-41",
        control_version=7,
        supervision_sources=(source,),
        replay_sources=(),
        resolution_tier="full",
        optimizer_updates=2,
        insertion=_ns(max_seeds=64),
        allow_growth=True,
        allow_relocation=False,
        mutation_budget=_ns(max_total=12),
    )
    gpu = _ns(
        transaction_id=8,
        snapshot_id=40,
        transaction_type=_ns(value="optimization"),
        duration_ms=3.25,
        batch_size=2,
        pixels=8192,
        rasterizer_calls=1,
        optimizer_updates=2,
        mutation_type="insert",
        inserted=5,
        pruned=1,
        relocated=0,
        corrected=0,
    )
    return _ns(
        publication_id=13,
        snapshot_id=41,
        freshness=_ns(value="healthy"),
        source_clock_ns=9_000_000,
        admitted_clock_ns=7_000_000,
        map_clock_ns=4_000_000,
        rates=_ns(source_hz=30.0, admitted_hz=12.0, commit_hz=10.0),
        latencies=_ns(
            source_to_first_supervision=_ns(samples=5, p50_ms=2.0, p95_ms=4.0, max_ms=6.0),
            source_to_geometry_commit=_ns(samples=4, p50_ms=3.0, p95_ms=7.0, max_ms=9.0),
        ),
        queues=_ns(**{
            name: _ns(depth=i, oldest_age_ms=float(i))
            for i, name in enumerate(("source", "candidate", "commit", "transfer", "gpu"), 1)
        }),
        observation=disposition,
        work_plan=plan,
        gpu_transaction=gpu if completed else None,
        transfers=_ns(
            h2d_ops=3, h2d_bytes=4096, d2h_ops=1, d2h_bytes=64,
            pool_waits=2, pool_drops=0, pinned_bytes=8192,
            peak_pinned_bytes=16384, inflight_limit=4,
            shape_buckets=2, shape_bucket_limit=8,
        ) if completed else None,
        map_state=_ns(gaussian_count=500, map_content_version=5, topology_version=3, parameter_version=9),
        residency=_ns(active=2, replay=1, history=8, hot_gpu=3),
    )


def test_panels_are_a_truthful_projection_of_the_completed_runtime_snapshot():
    panels = assemble_viewer_telemetry(_snapshot())

    assert panels.decision.snapshot_id == 41
    assert panels.decision.control_version == 7
    assert panels.decision.disposition == "admit"
    assert panels.decision.reason == "newest frontier"
    assert panels.decision.supervision_sources == ("cam:2:17@99",)
    assert panels.decision.optimizer_updates == 2
    assert panels.decision.insertion_seeds == 64
    assert panels.freshness.source_hz == 30.0
    assert panels.freshness.source_to_map_age_ms == 5.0
    assert panels.freshness.queue_depths == (("source", 1), ("candidate", 2), ("commit", 3), ("transfer", 4), ("gpu", 5))
    assert panels.gpu.transaction_id == 8
    assert panels.gpu.h2d_bytes == 4096
    assert panels.gpu.batch_mode is None
    assert panels.gpu.fallback_reason is None
    assert panels.gpu.memory_bytes is None
    assert panels.residency.gaussian_count == 500
    assert panels.residency.hot_gpu == 3


def test_unobserved_gpu_facts_remain_unknown_instead_of_becoming_zero():
    panels = assemble_viewer_telemetry(_snapshot(completed=False))

    assert panels.gpu.transaction_id is None
    assert panels.gpu.duration_ms is None
    assert panels.gpu.h2d_ops is None
    assert panels.gpu.pool_waits is None
    assert panels.gpu.unexpected_d2h is None


class _DeterministicQueue:
    def __init__(self):
        self.value = None

    def put_nowait(self, value):
        if self.value is not None:
            raise Full
        self.value = value

    def get_nowait(self):
        if self.value is None:
            raise Empty
        value, self.value = self.value, None
        return value


def test_sidecar_ipc_replaces_stale_value_per_channel_without_cross_eviction():
    queues = {name: _DeterministicQueue() for name in ("metrics", "runtime_telemetry")}
    ipc = LatestWinsSidecarIPC(queues)

    ipc.publish("metrics", {"seq": 1})
    ipc.publish("runtime_telemetry", {"seq": 10})
    ipc.publish("metrics", {"seq": 2})

    assert ipc.read_latest("metrics") == {"seq": 2}
    assert ipc.read_latest("runtime_telemetry") == {"seq": 10}
    assert ipc.replaced == 1
    assert ipc.replaced_by_channel == {"metrics": 1}


def test_sidecar_ipc_rejects_unknown_channels():
    ipc = LatestWinsSidecarIPC({"metrics": _DeterministicQueue()})

    try:
        ipc.publish("raw_input", {})
    except KeyError as exc:
        assert "raw_input" in str(exc)
    else:
        raise AssertionError("unknown channel was silently accepted")


def test_process_safe_queue_burst_retains_the_newest_value():
    with mp.Manager() as manager:
        ipc = LatestWinsSidecarIPC({"metrics": manager.Queue(maxsize=1)})
        for seq in range(100):
            assert ipc.publish("metrics", {"seq": seq})

        assert ipc.read_latest("metrics") == {"seq": 99}
        assert ipc.replaced == 99
        assert ipc.failed == 0


def test_runtime_telemetry_has_an_independent_canonical_preview_channel():
    event = RuntimeTelemetryPreview(
        iteration=12,
        wall_time_s=1.0,
        panels=assemble_viewer_telemetry(_snapshot()),
    )

    assert channel_for_event(event) is PreviewChannel.RUNTIME_TELEMETRY


def test_viser_updates_four_distinct_runtime_panels():
    class _Text:
        content = ""

    texts = {key: _Text() for key in ("decision", "freshness", "gpu", "residency")}
    panels = assemble_viewer_telemetry(_snapshot()).as_wire()

    _apply({"kind": "runtime_telemetry", "panels": panels}, {}, texts)

    assert "newest frontier" in texts["decision"].content
    assert "source hz" in texts["freshness"].content
    assert "h2d bytes" in texts["gpu"].content
    assert "gaussian count" in texts["residency"].content
