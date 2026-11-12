"""Runtime utilities and the canonical mapping runtime for EliGSiR."""

from .application import (
    ComposedMappingApplication,
    MappingApplication,
    MappingApplicationContractError,
    MappingApplicationOptions,
    create_mapping_application,
    validate_mapping_application,
)
from .save_worker import AsyncSaveWorker
from .session import MappingSession, MappingSessionClosed, SubmitOutcome

_TELEMETRY_EXPORTS = {
    "AsyncGpuEventTelemetryPublisher",
    "GpuTelemetryDescriptor",
    "RuntimeTelemetryAdapter",
    "RuntimeTelemetryInputs",
    "RuntimeTelemetrySnapshot",
    "RuntimeTelemetryStore",
}


def __getattr__(name):
    """Keep lightweight runtime imports from eagerly importing PyTorch."""

    if name in _TELEMETRY_EXPORTS:
        from . import telemetry

        value = getattr(telemetry, name)
        globals()[name] = value
        return value
    raise AttributeError(name)

__all__ = [
    "AsyncSaveWorker",
    "AsyncGpuEventTelemetryPublisher",
    "ComposedMappingApplication",
    "MappingApplication",
    "MappingApplicationContractError",
    "MappingApplicationOptions",
    "MappingSession",
    "MappingSessionClosed",
    "GpuTelemetryDescriptor",
    "RuntimeTelemetryAdapter",
    "RuntimeTelemetryInputs",
    "RuntimeTelemetrySnapshot",
    "RuntimeTelemetryStore",
    "SubmitOutcome",
    "create_mapping_application",
    "validate_mapping_application",
]
