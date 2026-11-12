"""Protocol declarations for the integration services.

``RGBDIntegrator``, ``BootstrapIntegrator``, and ``SubmapIntegrator`` are
satisfied directly by :class:`eligsir.mapping.coordinator.MappingCoordinator`
(its ``step``/``run_rolling_seed_mode``/``run_submap_stitch_mode`` methods) —
there are no separate wrapper classes for these three; an earlier version
of this package had one-line pass-through wrapper modules per Protocol,
which added indirection without adding ownership and were collapsed into
the coordinator directly (see that module's docstring). ``FramePreparationService``
and ``IntegrationPlanner`` remain Protocol-only: ``insert_gaussians_from_frame``
does frame-prep, candidate-planning, and commit in one function, so there
is no existing split point to delegate a "prep-only" or "plan-only" call to
without first decomposing that function's body — a separate, larger
follow-up."""

from __future__ import annotations

from typing import Any, Protocol


class FramePreparationService(Protocol):
    """RGB-D conversion, masks, normals, backprojection for one frame.
    Corresponds to the frame-preparation portion of
    ``insert_gaussians_from_frame`` in ``train_streaming.py``."""

    def prepare(self, frame: Any, args: Any) -> Any:
        """Prepare a raw frame for integration.

        Args:
            frame: The input frame to prepare.
            args: Training/integration configuration.

        Returns:
            Prepared frame data ready for planning and insertion.
        """
        ...


class IntegrationPlanner(Protocol):
    """Coverage filtering, dedup, and candidate selection over prepared
    frame data — the voxel-downsample/occupancy-filter portion of
    ``insert_gaussians_from_frame``."""

    def plan(self, prepared: Any, streaming_scene: Any, args: Any) -> Any:
        """Plan Gaussian insertions from prepared frame data.

        Args:
            prepared: Output from FramePreparationService.prepare().
            streaming_scene: The current scene/map state.
            args: Training/integration configuration.

        Returns:
            Integration plan (e.g., candidate Gaussians to insert).
        """
        ...


class RGBDIntegrator(Protocol):
    """Normal (non-bootstrap, non-submap) keyframe geometry integration —
    the steady-state path through ``insert_gaussians_from_frame``. Real
    implementation: :meth:`eligsir.mapping.coordinator.MappingCoordinator.step`."""

    def step(self, gaussians: Any, frame: Any, args: Any, streaming_scene: Any, **kwargs: Any) -> Any:
        """Execute steady-state frame integration.

        Args:
            gaussians: The Gaussian model to update.
            frame: The frame being integrated.
            args: Training/integration configuration.
            streaming_scene: The current scene/map state.
            **kwargs: Additional integration parameters.

        Returns:
            Integration result or metrics.
        """
        ...


class BootstrapIntegrator(Protocol):
    """Initial/rolling-seed integration behavior — corresponds to
    ``_run_rolling_seed`` in ``train_streaming.py``. Real implementation:
    :meth:`eligsir.mapping.coordinator.MappingCoordinator.run_rolling_seed_mode`."""

    def run_rolling_seed_mode(self, *args: Any, **kwargs: Any) -> Any:
        """Run rolling-seed (initial) integration mode.

        Args:
            *args: Integration arguments.
            **kwargs: Additional integration parameters.

        Returns:
            Integration result or metrics.
        """
        ...


class SubmapIntegrator(Protocol):
    """Submap-specific integration policy — corresponds to
    ``_run_submap_stitch`` in ``train_streaming.py``. Real implementation:
    :meth:`eligsir.mapping.coordinator.MappingCoordinator.run_submap_stitch_mode`."""

    def run_submap_stitch_mode(self, *args: Any, **kwargs: Any) -> Any:
        """Run submap stitching integration mode.

        Args:
            *args: Integration arguments.
            **kwargs: Additional integration parameters.

        Returns:
            Integration result or metrics.
        """
        ...
