from __future__ import annotations

import sys
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from eligsir.contracts.runtime import (
    BufferOwnership,
    BufferRef,
    SensorPacket,
    SourceIdentity,
)
from eligsir.contracts.types import CameraIntrinsics
from eligsir.runtime.application import (
    MappingApplicationContractError,
    MappingApplicationOptions,
    create_mapping_application,
    validate_mapping_application,
)
from eligsir.tracking.providers import observation_from_pose_matrix


def test_production_factory_forwards_terminal_capture_before_provider_reuses_invalid_packet(
    monkeypatch,
):
    """The default composition retains an invalid packet before source release."""
    from eligsir.runtime import production_application

    source = SourceIdentity("live", 0, 1, 1_000)
    rgb = np.full((2, 2, 3), 17, dtype=np.uint8)
    depth = np.full((2, 2), 1.25, dtype=np.float32)
    packet = SensorPacket(
        source=source,
        rgb=BufferRef.from_numpy(
            rgb, encoding="rgb8", ownership=BufferOwnership.BORROWED
        ),
        depth=BufferRef.from_numpy(
            depth,
            encoding="32FC1",
            units="m",
            ownership=BufferOwnership.BORROWED,
        ),
        intrinsics=CameraIntrinsics(fx=1.0, fy=1.0, cx=1.0, cy=1.0, width=2, height=2),
    )
    retained = []

    def capture(value):
        return replace(
            value,
            rgb=BufferRef.from_numpy(
                value.rgb.data.copy(), encoding="rgb8", ownership=BufferOwnership.OWNED
            ),
            depth=BufferRef.from_numpy(
                value.depth.data.copy(),
                encoding="32FC1",
                units="m",
                ownership=BufferOwnership.OWNED,
            ),
        )

    def release(_value):
        rgb.fill(0)
        depth.fill(0)

    binding = SimpleNamespace(
        mapper_sink=lambda *_args: None,
        optimizer_tick=lambda: None,
        source_complete_hook=None,
        flush_hook=None,
        close_hook=None,
        observation_owner=None,
        control_snapshot_provider=None,
        disposition_sink=None,
        packet_capture=capture,
        terminal_packet_sink=lambda value, outcome: retained.append((value, outcome)),
        source_consumed_observer=release,
        materialization_policy=None,
        materialization_backend=None,
    )
    monkeypatch.setattr(
        production_application, "_resolve_parameters", lambda *_args, **_kwargs: (object(), object(), object(), object())
    )
    monkeypatch.setattr(production_application, "mapping_runtime_config_from_args", lambda _args: object())
    monkeypatch.setattr(production_application, "MappingBackendCompatibility", lambda **_kwargs: object())
    monkeypatch.setattr(production_application, "CanonicalMappingRuntime", lambda *_args: object())
    monkeypatch.setattr(production_application, "runtime_binding_from_runtime", lambda *_args, **_kwargs: binding)

    application = production_application.create_production_mapping_application()
    outcome = application.session.submit(packet)

    assert not outcome.accepted
    assert rgb.max() == 0 and depth.max() == 0
    assert len(retained) == 1
    assert retained[0][0].source == source
    assert retained[0][0].rgb.data[0, 0, 0] == 17
    assert retained[0][0].depth.data[0, 0] == pytest.approx(1.25)


def test_config_dependencies_factory_forwards_terminal_capture_before_borrowed_buffer_reuse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The config/dependencies factory is a real supported terminal route."""
    from eligsir.runtime import application as application_module
    from eligsir.runtime.config import MappingRuntimeDependencies, RuntimeBinding
    from eligsir.runtime.session import IngressTelemetrySnapshot
    from eligsir.tracking.providers import ExternalTrackingProvider

    source = SourceIdentity("live", 0, 7, 7_000)
    rgb = np.full((2, 2, 3), 29, dtype=np.uint8)
    depth = np.full((2, 2), 2.5, dtype=np.float32)
    packet = SensorPacket(
        source=source,
        rgb=BufferRef.from_numpy(rgb, encoding="rgb8", ownership=BufferOwnership.BORROWED),
        depth=BufferRef.from_numpy(
            depth, encoding="32FC1", units="m", ownership=BufferOwnership.BORROWED
        ),
        intrinsics=CameraIntrinsics(fx=1.0, fy=1.0, cx=1.0, cy=1.0, width=2, height=2),
    )
    retained = []

    def capture(value):
        return replace(
            value,
            rgb=BufferRef.from_numpy(value.rgb.data.copy(), encoding="rgb8", ownership=BufferOwnership.OWNED),
            depth=BufferRef.from_numpy(
                value.depth.data.copy(), encoding="32FC1", units="m", ownership=BufferOwnership.OWNED
            ),
        )

    binding = RuntimeBinding(
        mapper_sink=lambda *_args: None,
        optimizer_tick=lambda: None,
        packet_capture=capture,
        terminal_packet_sink=lambda value, outcome: retained.append((value, outcome)),
        source_consumed_observer=lambda _value: (rgb.fill(0), depth.fill(0)),
    )
    dependencies = MappingRuntimeDependencies(
        runtime_factory=lambda _config: binding,
        tracking_provider=ExternalTrackingProvider(),
        ingress_telemetry_provider=lambda *_args: IngressTelemetrySnapshot(),
    )
    monkeypatch.setattr(application_module, "MappingRuntimeConfig", object)

    app = create_mapping_application(config=object(), dependencies=dependencies)
    outcome = app.session.submit(packet)

    assert not outcome.accepted
    assert rgb.max() == 0 and depth.max() == 0
    assert len(retained) == 1
    assert retained[0][0].source == source
    assert retained[0][0].rgb.data[0, 0, 0] == 29
    assert retained[0][0].depth.data[0, 0] == pytest.approx(2.5)


def _packet(ordinal: int) -> SensorPacket:
    source = SourceIdentity(
        source_id="cam",
        stream_epoch=0,
        source_ordinal=ordinal,
        timestamp_ns=1_000_000_000 + ordinal,
    )
    return SensorPacket(
        source=source,
        rgb=BufferRef.from_numpy(np.zeros((4, 4, 3), dtype=np.uint8), encoding="rgb8"),
        depth=BufferRef.from_numpy(
            np.ones((4, 4), dtype=np.float32), encoding="32FC1", units="m"
        ),
        intrinsics=CameraIntrinsics(
            fx=1.0, fy=1.0, cx=2.0, cy=2.0, width=4, height=4
        ),
        tracking=observation_from_pose_matrix(source, np.eye(4), provider="external"),
    )


def _components():
    mapped: list[object] = []
    ticks: list[int] = []
    flushed: list[int] = []
    closed: list[int] = []
    source_completed: list[int] = []

    def mapper(frame, _observation):
        mapped.append(frame)
        return SimpleNamespace(admitted=True)

    options = MappingApplicationOptions(
        mapper_sink=mapper,
        optimizer_tick=lambda: ticks.append(len(ticks)),
        source_complete_hook=lambda: source_completed.append(1),
        flush_hook=lambda: flushed.append(1),
        close_hook=lambda: closed.append(1),
    )
    return options, SimpleNamespace(
        mapped=mapped,
        ticks=ticks,
        flushed=flushed,
        closed=closed,
        source_completed=source_completed,
    )


def test_default_factory_delegates_to_the_single_production_composition(monkeypatch):
    options, _ = _components()
    expected = create_mapping_application(options)
    fake_module = SimpleNamespace(
        create_production_mapping_application=lambda **kwargs: expected
    )
    monkeypatch.setitem(
        sys.modules, "eligsir.runtime.production_application", fake_module
    )

    application = create_mapping_application(profile="live/canonical_mapping")

    assert application is expected


def test_factory_returns_valid_application_with_real_mapping_sink():
    options, _ = _components()
    application = create_mapping_application(options)

    assert validate_mapping_application(application) is application
    assert application.session.has_mapper
    assert not application.session.ingress_only


def test_options_and_production_kwargs_are_mutually_exclusive():
    options, _ = _components()
    with pytest.raises(MappingApplicationContractError, match="mutually exclusive"):
        create_mapping_application(options, profile="live/canonical_mapping")


def test_submit_materializes_and_integrates_each_packet_exactly_once():
    options, evidence = _components()
    application = create_mapping_application(options)

    first = application.session.submit(_packet(0))
    second = application.session.submit(_packet(1))

    assert first.frame is not None and second.frame is not None
    assert [frame.source.source_ordinal for frame in evidence.mapped] == [0, 1]
    assert application.session.accepted_count == 2
    assert application.session.mapped_count == 2


def test_tick_advances_optimizer_once_when_no_frame_was_submitted():
    options, evidence = _components()
    application = create_mapping_application(options)

    result = application.tick()

    assert evidence.ticks == [0]
    assert application.tick_count == 1
    assert result is None


def test_source_completion_crosses_application_boundary_exactly_once():
    options, evidence = _components()
    application = create_mapping_application(options)

    application.mark_source_complete()
    application.mark_source_complete()

    assert evidence.source_completed == [1]
    assert application.source_complete


def test_source_completion_after_close_fails_closed():
    options, _ = _components()
    application = create_mapping_application(options)
    application.close()

    with pytest.raises(RuntimeError, match="source completion after close"):
        application.mark_source_complete()


def test_close_flushes_and_releases_every_owned_component_exactly_once():
    options, evidence = _components()
    application = create_mapping_application(options)
    application.session.submit(_packet(0))

    application.close()
    application.close()

    assert evidence.flushed == [1]
    assert evidence.closed == [1]
    assert application.session.closed


def test_tick_after_close_fails_closed():
    options, _ = _components()
    application = create_mapping_application(options)
    application.close()

    with pytest.raises(RuntimeError, match=r"tick\(\) after close"):
        application.tick()


def test_cleanup_releases_all_owners_even_when_flush_raises():
    options, evidence = _components()

    def fail_flush():
        evidence.flushed.append(1)
        raise RuntimeError("flush failed")

    application = create_mapping_application(replace(options, flush_hook=fail_flush))

    with pytest.raises(RuntimeError, match="flush failed"):
        application.close()

    assert evidence.flushed == [1]
    assert evidence.closed == [1]
    assert application.closed
    assert application.session.closed
