"""Issue #1601 slice 2: the real optimization step entry points must forward
the transaction's topology version into the assimilation support publisher.

Production (production_application.py) passes ``topology_version=self.backend
.topology_version`` into both entry points; if the publisher call drops it,
every support snapshot latches topology 0 and the staleness rule in
``active_control_decision`` can never fire.
"""
from __future__ import annotations

import pytest

from eligsir.mapping.optimization_runtime import (
    run_streaming_optimization_step,
    run_streaming_optimization_transaction,
)


class _Published(Exception):
    """Short-circuits the entry point right after the publish call."""


class _RecordingRuntime:
    def __init__(self) -> None:
        self.calls: list[tuple[int, dict]] = []

    def publish_assimilation_support(self, iteration: int, **provenance) -> None:
        self.calls.append((int(iteration), dict(provenance)))
        raise _Published


@pytest.mark.parametrize("topology", [7, 12345])
def test_b1_step_forwards_topology_version_to_publisher(topology: int) -> None:
    runtime = _RecordingRuntime()
    with pytest.raises(_Published):
        run_streaming_optimization_step(
            runtime, engine=None, camera=object(), iteration=41, topology_version=topology
        )
    (iteration, provenance), = runtime.calls
    assert iteration == 41
    assert provenance["topology_version"] == topology


def test_adaptive_transaction_forwards_topology_version_to_publisher() -> None:
    runtime = _RecordingRuntime()
    with pytest.raises(_Published):
        run_streaming_optimization_transaction(
            runtime,
            engine=None,
            cameras=(object(),),
            requested_batch_size=1,
            requested_camera_count=1,
            iteration=9,
            topology_version=99,
        )
    (iteration, provenance), = runtime.calls
    assert iteration == 9
    assert provenance["topology_version"] == 99
