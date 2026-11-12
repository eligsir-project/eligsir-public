"""Transport-free application composition for continual mapping.

``MappingSession`` owns ordered frame materialization. The application owns the
independent optimizer clock. Transports compose this interface; they never own
Gaussian state, admission policy, or optimizer semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Protocol, runtime_checkable

from eligsir.contracts.materialize import MaterializationPolicy, TensorBackend
from eligsir.contracts.runtime import MappingFrame, TrackingObservation
from eligsir.contracts.tracking import TrackingProvider
from eligsir.mapping.admission.observation_owner import ObservationOwner

from .config import MappingRuntimeConfig, MappingRuntimeDependencies
from .session import ControlSnapshotProvider, DispositionSink, MappingSession
from .prepared_ingress import PreparedSourceIngressHandoff


class MappingApplicationContractError(TypeError):
    """Raised when an object cannot serve as the canonical application."""


MapperSink = Callable[[MappingFrame, TrackingObservation], Any]


@runtime_checkable
class MappingApplication(Protocol):
    @property
    def session(self) -> MappingSession:
        """Return the canonical mapping session."""
        ...

    def mark_source_complete(self) -> Any:
        """Mark the source as complete."""
        ...

    def tick(self) -> Any:
        """Perform one optimization tick."""
        ...

    def flush(self) -> Any:
        """Flush pending mapped frames."""
        ...

    def close(self) -> Any:
        """Close the application and release resources."""
        ...


@dataclass(frozen=True, slots=True)
class MappingApplicationOptions:
    mapper_sink: Optional[MapperSink]
    optimizer_tick: Optional[Callable[[], Any]] = None
    tracking_provider: Optional[TrackingProvider] = None
    require_tracking: bool = True
    max_events: int = 4096
    materialization_policy: Optional[MaterializationPolicy] = None
    materialization_backend: Optional[TensorBackend] = None
    source_complete_hook: Optional[Callable[[], Any]] = None
    flush_hook: Optional[Callable[[], Any]] = None
    close_hook: Optional[Callable[[], Any]] = None
    observation_owner: Optional[ObservationOwner] = None
    control_snapshot_provider: Optional[ControlSnapshotProvider] = None
    disposition_sink: Optional[DispositionSink] = None
    ingress_telemetry_provider: Optional[Callable[[Any, TrackingObservation], Any]] = None
    packet_capture: Optional[Callable[[Any], Any]] = None
    terminal_packet_sink: Optional[Callable[[Any, Any], Any]] = None
    source_consumed_observer: Optional[Callable[[Any], Any]] = None
    runtime: Optional[Any] = None
    prepared_mode: bool = False
    prepared_ingress_handoff: Optional[PreparedSourceIngressHandoff] = None
    prepared_runtime_bridge: Optional[Any] = None
    prepared_tick: Optional[Callable[[], Any]] = None
    prepared_close: Optional[Callable[[], Any]] = None
    prepared_progress_waiter: Optional[Callable[..., bool]] = None


class ComposedMappingApplication:
    """Join one mapper owner with one optimizer-clock owner."""

    def __init__(self, options: MappingApplicationOptions) -> None:
        from eligsir.tracking.providers import ExternalTrackingProvider

        if options.prepared_mode:
            if options.mapper_sink is not None:
                raise MappingApplicationContractError(
                    "prepared_mode=True cannot bind the legacy mapper_sink"
                )
            if not callable(options.prepared_tick):
                raise MappingApplicationContractError(
                    "prepared_mode=True requires prepared_tick"
                )
        else:
            if not callable(options.mapper_sink):
                raise MappingApplicationContractError(
                    "MappingApplicationOptions.mapper_sink must be callable"
                )
            if not callable(options.optimizer_tick):
                raise MappingApplicationContractError(
                    "MappingApplicationOptions.optimizer_tick must be callable"
                )
            if options.prepared_tick is not None:
                raise MappingApplicationContractError(
                    "prepared_tick requires prepared_mode=True"
                )

        bridge = options.prepared_runtime_bridge
        if bridge is not None:
            from .prepared_runtime_bridge import PreparedRuntimeBridge

            if not isinstance(bridge, PreparedRuntimeBridge):
                raise MappingApplicationContractError(
                    "prepared_runtime_bridge must be PreparedRuntimeBridge"
                )
            if not options.prepared_mode:
                raise MappingApplicationContractError(
                    "prepared_runtime_bridge requires prepared_mode=True"
                )
            if options.prepared_ingress_handoff is not bridge.handoff:
                raise MappingApplicationContractError(
                    "prepared runtime bridge must own the session handoff"
                )
        elif options.prepared_mode:
            raise MappingApplicationContractError(
                "prepared_mode=True requires prepared_runtime_bridge"
            )
        if options.prepared_close is not None and not callable(options.prepared_close):
            raise MappingApplicationContractError(
                "prepared_close must be callable when provided"
            )
        if options.prepared_close is not None and not options.prepared_mode:
            raise MappingApplicationContractError(
                "prepared_close requires prepared_mode=True"
            )
        for name in (
            "source_complete_hook",
            "flush_hook",
            "close_hook",
            "packet_capture",
            "terminal_packet_sink",
            "source_consumed_observer",
        ):
            value = getattr(options, name)
            if value is not None and not callable(value):
                raise MappingApplicationContractError(
                    f"MappingApplicationOptions.{name} must be callable when provided"
                )

        self._options = options
        self._tick_count = 0
        self._closed = False
        self._source_complete = False
        self._source_complete_result = None
        self._flush_required = True
        self._prepared_retry_pending = False
        self._session = MappingSession(
            options.tracking_provider or ExternalTrackingProvider(),
            mapper=options.mapper_sink,
            require_tracking=options.require_tracking,
            max_events=options.max_events,
            materialization_policy=options.materialization_policy,
            materialization_backend=options.materialization_backend,
            observation_owner=options.observation_owner,
            control_snapshot_provider=options.control_snapshot_provider,
            disposition_sink=options.disposition_sink,
            ingress_telemetry_provider=options.ingress_telemetry_provider,
            packet_capture=options.packet_capture,
            terminal_packet_sink=options.terminal_packet_sink,
            source_consumed_observer=options.source_consumed_observer,
            prepared_mode=options.prepared_mode,
            prepared_ingress_handoff=options.prepared_ingress_handoff,
        )

    @property
    def session(self) -> MappingSession:
        """Return the canonical mapping session."""
        return self._session

    @property
    def runtime(self) -> Optional[Any]:
        """Return the configured runtime context."""
        return self._options.runtime

    @property
    def materialization_policy(self) -> Optional[MaterializationPolicy]:
        """Return the configured materialization policy."""
        return self._options.materialization_policy

    @property
    def materialization_backend(self) -> Optional[TensorBackend]:
        """Return the configured tensor backend."""
        return self._options.materialization_backend

    @property
    def tick_count(self) -> int:
        """Return the number of completed optimization ticks."""
        return self._tick_count

    @property
    def closed(self) -> bool:
        """Return whether the application has been closed."""
        return self._closed

    @property
    def source_complete(self) -> bool:
        """Return whether the source has been marked complete."""
        return self._source_complete

    def mark_source_complete(self) -> Any:
        """Mark the source as complete and invoke the completion hook."""
        if self._closed:
            raise RuntimeError("source completion after close()")
        if self._source_complete:
            return self._source_complete_result
        result = None
        if self._options.source_complete_hook is not None:
            result = self._options.source_complete_hook()
        if (
            self._options.prepared_mode
            and getattr(getattr(result, "state", None), "value", None)
            == "bootstrap_pending"
        ):
            return result
        self._source_complete = True
        self._source_complete_result = result
        return result

    def tick(self) -> Any:
        """Perform one optimization tick."""
        if self._closed:
            raise RuntimeError("tick() after close()")
        result = (
            self._options.prepared_tick()
            if self._options.prepared_mode
            else self._options.optimizer_tick()
        )
        self._prepared_retry_pending = bool(getattr(result, "retryable", False))
        self._tick_count += 1
        self._flush_required = True
        return result

    def wait_for_prepared_progress(self, *, timeout_s: float) -> bool:
        """Wait for prepared progress with timeout."""
        waiter = self._options.prepared_progress_waiter
        return bool(self._prepared_retry_pending and waiter and waiter(timeout_s=timeout_s))

    def abort_unaccepted_refinement(self) -> None:
        """Prevent close() from flushing a candidate that could not be restored."""

        if not self._closed:
            self._flush_required = False

    def flush(self) -> int:
        """Flush pending mapped frames and invoke flush hook."""
        if self._closed:
            raise RuntimeError("flush() after close()")
        if not self._flush_required:
            return self._session.mapped_count
        if self._options.prepared_mode and not self._source_complete:
            self.mark_source_complete()
            if not self._source_complete:
                raise RuntimeError(
                    "prepared source completion is still pending; refusing refinement/flush"
                )
        self._flush_required = False
        mapped_count = self._session.flush()
        if self._options.flush_hook is not None:
            self._options.flush_hook()
        return mapped_count

    def close(self) -> None:
        """Close the application and release all resources."""
        if self._closed:
            return
        first_error: Optional[BaseException] = None
        bridge = self._options.prepared_runtime_bridge
        for cleanup in (
            self.flush,
            self._options.prepared_close,
            None if bridge is None else bridge.close,
            self._options.close_hook,
            self._session.close,
        ):
            if cleanup is None:
                continue
            try:
                cleanup()
            except BaseException as exc:
                first_error = first_error or exc
        self._closed = True
        if first_error is not None:
            raise first_error


def create_mapping_application(
    options: Optional[MappingApplicationOptions] = None,
    *,
    config: Optional[MappingRuntimeConfig] = None,
    dependencies: Optional[MappingRuntimeDependencies] = None,
    **production_kwargs: Any,
) -> MappingApplication:
    """Create a validated mapping application from options, config, or production kwargs."""
    if config is not None or dependencies is not None:
        if config is None or dependencies is None:
            raise MappingApplicationContractError(
                "config and dependencies must be supplied together"
            )
        if options is not None or production_kwargs:
            raise MappingApplicationContractError(
                "config/dependencies cannot be combined with options or production kwargs"
            )
        if not isinstance(config, MappingRuntimeConfig):
            raise MappingApplicationContractError("config must be MappingRuntimeConfig")
        if not isinstance(dependencies, MappingRuntimeDependencies):
            raise MappingApplicationContractError(
                "dependencies must be MappingRuntimeDependencies"
            )
        binding = dependencies.runtime_factory(config)
        from .config import RuntimeBinding

        if not isinstance(binding, RuntimeBinding):
            raise MappingApplicationContractError("runtime_factory must return RuntimeBinding")
        return validate_mapping_application(
            ComposedMappingApplication(
                MappingApplicationOptions(
                    mapper_sink=binding.mapper_sink,
                    optimizer_tick=binding.optimizer_tick,
                    tracking_provider=dependencies.tracking_provider,
                    materialization_policy=binding.materialization_policy,
                    materialization_backend=binding.materialization_backend,
                    source_complete_hook=binding.source_complete_hook,
                    flush_hook=binding.flush_hook,
                    close_hook=binding.close_hook,
                    observation_owner=binding.observation_owner,
                    control_snapshot_provider=binding.control_snapshot_provider,
                    disposition_sink=binding.disposition_sink,
                    packet_capture=binding.packet_capture,
                    terminal_packet_sink=binding.terminal_packet_sink,
                    source_consumed_observer=binding.source_consumed_observer,
                    ingress_telemetry_provider=dependencies.ingress_telemetry_provider,
                )
            )
        )

    if options is not None:
        if production_kwargs:
            raise MappingApplicationContractError(
                "MappingApplicationOptions and production construction kwargs are mutually exclusive"
            )
        if not isinstance(options, MappingApplicationOptions):
            raise MappingApplicationContractError(
                "options must be MappingApplicationOptions when provided"
            )
        return validate_mapping_application(ComposedMappingApplication(options))

    from .production_application import create_production_mapping_application

    return validate_mapping_application(
        create_production_mapping_application(**production_kwargs)
    )


def validate_mapping_application(application: Any) -> MappingApplication:
    """Validate that an application implements the MappingApplication protocol."""
    missing = [
        name
        for name in ("mark_source_complete", "tick", "flush", "close")
        if not callable(getattr(application, name, None))
    ]
    if missing:
        raise MappingApplicationContractError(
            "mapping application is missing callables: " + ", ".join(missing)
        )
    session = getattr(application, "session", None)
    if not isinstance(session, MappingSession):
        raise MappingApplicationContractError(
            "mapping application must expose its canonical MappingSession as .session"
        )
    if session.ingress_only or (not session.has_mapper and not session.prepared_mode):
        raise MappingApplicationContractError(
            "production mapping application cannot use an ingress-only/mapperless session"
        )
    return application


__all__ = [
    "ComposedMappingApplication",
    "MapperSink",
    "MappingApplication",
    "MappingApplicationContractError",
    "MappingApplicationOptions",
    "create_mapping_application",
    "validate_mapping_application",
]
