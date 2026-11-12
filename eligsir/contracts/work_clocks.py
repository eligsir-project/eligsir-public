"""Semantic cumulative work clocks for the canonical mapping runtime."""

from __future__ import annotations

from dataclasses import dataclass


def _non_negative_integer(value: object, name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an exact integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


@dataclass(frozen=True, slots=True)
class RuntimeWorkClockSnapshot:
    """Immutable cumulative semantic work completed by one mapping runtime."""

    control_tick_epoch: int = 0
    optimizer_updates: int = 0
    supervised_rgb_views: int = 0
    supervised_rgb_pixels: int = 0
    depth_valid_samples: int = 0
    source_frames_seen: int = 0
    admitted_observations: int = 0
    depth_mutation_epochs: int = 0
    mcmc_mutation_epochs: int = 0
    mapping_active_ns: int = 0
    source_complete: bool = False

    def __post_init__(self) -> None:
        for name in (
            "control_tick_epoch",
            "optimizer_updates",
            "supervised_rgb_views",
            "supervised_rgb_pixels",
            "depth_valid_samples",
            "source_frames_seen",
            "admitted_observations",
            "depth_mutation_epochs",
            "mcmc_mutation_epochs",
            "mapping_active_ns",
        ):
            _non_negative_integer(getattr(self, name), name)
        if type(self.source_complete) is not bool:
            raise TypeError("source_complete must be a bool")


class RuntimeWorkClockOwner:
    """Sole mutable owner of one runtime's cumulative semantic work clocks."""

    __slots__ = (
        "_control_tick_epoch",
        "_optimizer_updates",
        "_supervised_rgb_views",
        "_supervised_rgb_pixels",
        "_depth_valid_samples",
        "_source_frames_seen",
        "_admitted_observations",
        "_depth_mutation_epochs",
        "_mcmc_mutation_epochs",
        "_mapping_active_ns",
        "_source_complete",
    )

    def __init__(self) -> None:
        self._control_tick_epoch = 0
        self._optimizer_updates = 0
        self._supervised_rgb_views = 0
        self._supervised_rgb_pixels = 0
        self._depth_valid_samples = 0
        self._source_frames_seen = 0
        self._admitted_observations = 0
        self._depth_mutation_epochs = 0
        self._mcmc_mutation_epochs = 0
        self._mapping_active_ns = 0
        self._source_complete = False

    def snapshot(self) -> RuntimeWorkClockSnapshot:
        return RuntimeWorkClockSnapshot(
            control_tick_epoch=self._control_tick_epoch,
            optimizer_updates=self._optimizer_updates,
            supervised_rgb_views=self._supervised_rgb_views,
            supervised_rgb_pixels=self._supervised_rgb_pixels,
            depth_valid_samples=self._depth_valid_samples,
            source_frames_seen=self._source_frames_seen,
            admitted_observations=self._admitted_observations,
            depth_mutation_epochs=self._depth_mutation_epochs,
            mcmc_mutation_epochs=self._mcmc_mutation_epochs,
            mapping_active_ns=self._mapping_active_ns,
            source_complete=self._source_complete,
        )

    def restore_branch_state(self, state: RuntimeWorkClockSnapshot) -> None:
        """Restore a captured numerical work-clock boundary exactly."""

        if not isinstance(state, RuntimeWorkClockSnapshot):
            raise TypeError("work-clock branch state must be RuntimeWorkClockSnapshot")
        self._control_tick_epoch = state.control_tick_epoch
        self._optimizer_updates = state.optimizer_updates
        self._supervised_rgb_views = state.supervised_rgb_views
        self._supervised_rgb_pixels = state.supervised_rgb_pixels
        self._depth_valid_samples = state.depth_valid_samples
        self._source_frames_seen = state.source_frames_seen
        self._admitted_observations = state.admitted_observations
        self._depth_mutation_epochs = state.depth_mutation_epochs
        self._mcmc_mutation_epochs = state.mcmc_mutation_epochs
        self._mapping_active_ns = state.mapping_active_ns
        self._source_complete = state.source_complete

    def record_control_tick(self) -> RuntimeWorkClockSnapshot:
        self._control_tick_epoch += 1
        return self.snapshot()

    def record_source_frame(self) -> RuntimeWorkClockSnapshot:
        self._source_frames_seen += 1
        return self.snapshot()

    def record_admission(self) -> RuntimeWorkClockSnapshot:
        self._admitted_observations += 1
        return self.snapshot()

    def record_optimization(
        self,
        optimizer_updates: int,
        supervised_views: int,
        supervised_pixels: int,
        mapping_active_ns: int,
    ) -> RuntimeWorkClockSnapshot:
        optimizer_updates = _non_negative_integer(
            optimizer_updates, "optimizer_updates"
        )
        supervised_views = _non_negative_integer(
            supervised_views, "supervised_views"
        )
        supervised_pixels = _non_negative_integer(
            supervised_pixels, "supervised_pixels"
        )
        mapping_active_ns = _non_negative_integer(
            mapping_active_ns, "mapping_active_ns"
        )
        if supervised_views > 0 and supervised_pixels == 0:
            raise ValueError("positive supervised_views require positive supervised_pixels")
        if supervised_pixels > 0 and supervised_views == 0:
            raise ValueError("positive supervised_pixels require positive supervised_views")
        if optimizer_updates == 0 and (
            supervised_views > 0
            or supervised_pixels > 0
            or mapping_active_ns > 0
        ):
            raise ValueError("positive optimization work requires optimizer_updates")

        self._optimizer_updates += optimizer_updates
        self._supervised_rgb_views += supervised_views
        self._supervised_rgb_pixels += supervised_pixels
        self._mapping_active_ns += mapping_active_ns
        return self.snapshot()

    def record_depth_supervision(
        self, depth_valid_samples: int
    ) -> RuntimeWorkClockSnapshot:
        depth_valid_samples = _non_negative_integer(
            depth_valid_samples, "depth_valid_samples"
        )
        self._depth_valid_samples += depth_valid_samples
        return self.snapshot()

    def record_depth_mutation(self) -> RuntimeWorkClockSnapshot:
        self._depth_mutation_epochs += 1
        return self.snapshot()

    def record_mapping_active(
        self, mapping_active_ns: int
    ) -> RuntimeWorkClockSnapshot:
        mapping_active_ns = _non_negative_integer(
            mapping_active_ns, "mapping_active_ns"
        )
        self._mapping_active_ns += mapping_active_ns
        return self.snapshot()

    def record_mcmc_mutation(self) -> RuntimeWorkClockSnapshot:
        self._mcmc_mutation_epochs += 1
        return self.snapshot()

    def mark_source_complete(self) -> RuntimeWorkClockSnapshot:
        self._source_complete = True
        return self.snapshot()


__all__ = ["RuntimeWorkClockOwner", "RuntimeWorkClockSnapshot"]
