"""``GaussianMapBackend`` protocol — the sole live Gaussian-state boundary."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from eligsir.map.mutation import GaussianMutationTransaction
    from eligsir.map.snapshots import MapSnapshot, SnapshotRequest
    from eligsir.map.views import GaussianMapOutputSample, GaussianMapReadView


@dataclass(frozen=True)
class CheckpointBlob:
    """Opaque backend checkpoint plus orthogonal runtime version counters."""

    layout: str
    payload: Any
    # ``map_version`` remains the serialized compatibility name for
    # map_content_version; new code should use the explicit backend property.
    map_version: int = 0
    topology_version: int = 0
    coordinate_frame_version: int = 0
    schema_version: int = 0

    @property
    def map_content_version(self) -> int:
        """Return the map content version."""
        return self.map_version


class GaussianOptimizationSession(Protocol):
    """Fixed-shape numerical-refinement capability."""

    def zero_grad(self) -> None:
        """Clear all gradients."""
        ...

    def step(self, visibility=None) -> None:
        """Apply optimizer step."""
        ...

    def update_learning_rate(self, iteration: int) -> float | None:
        """Update learning rate for iteration."""
        ...

    @property
    def model(self) -> Any:
        """Return the underlying model."""
        ...

    @property
    def parameter_version(self) -> int:
        """Advances on optimizer parameter-value updates in this session."""
        ...


class GaussianMapBackend(Protocol):
    """Single state owner for the live Gaussian map.

    Version dimensions intentionally have one meaning each:

    ``map_content_version``
        Any externally visible raw map-state mutation outside the optimizer.
    ``topology_version``
        Gaussian identity/count/order changes (insert/prune), not relocation.
    ``coordinate_frame_version``
        Bulk pose/submap coordinate corrections.
    ``schema_version``
        Buffer/layout/schema migrations.
    ``parameter_version``
        Lives on an optimization session and tracks numerical optimizer steps.

    ``map_version`` is retained as a read-only compatibility alias for
    ``map_content_version`` while provenance/read-view formats migrate.
    """

    @property
    def population_count(self) -> int:
        """Return the current Gaussian row count in O(1).

        Capacity arbitration needs only the population size, not the map
        contents. Exposing it directly means callers never materialize a
        whole ``GaussianMapReadView`` (and its full set of parameter tensor
        references) just to read one integer off ``means.shape[0]``.
        """
        ...

    def read_view(self) -> "GaussianMapReadView":
        """Return a read-only view of the map."""
        ...

    def sample_output(
        self, *, max_gaussians: int, parameter_version: int
    ) -> "GaussianMapOutputSample":
        """Sample map output with capacity constraint."""
        ...

    @property
    def map_content_version(self) -> int:
        """Return the map content version."""
        ...

    @property
    def map_version(self) -> int:
        """Return the map version (compatibility alias)."""
        ...

    @property
    def topology_version(self) -> int:
        """Return the topology version."""
        ...

    @property
    def coordinate_frame_version(self) -> int:
        """Return the coordinate frame version."""
        ...

    @property
    def schema_version(self) -> int:
        """Return the schema version."""
        ...

    def create_snapshot(self, request: "SnapshotRequest") -> "MapSnapshot":
        """Create a snapshot of the map."""
        ...

    def begin_optimization(self) -> GaussianOptimizationSession:
        """Begin an optimization session."""
        ...

    def begin_mutation(self) -> "GaussianMutationTransaction":
        """Begin a mutation transaction."""
        ...

    def checkpoint_state(self) -> CheckpointBlob:
        """Return the current checkpoint state."""
        ...

    def restore_checkpoint(self, blob: CheckpointBlob, training_args: Any) -> None:
        """Restore state from a checkpoint."""
        ...
