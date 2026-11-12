"""Torch-free contracts shared by population scheduling and policy."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MutationWorkToken:
    """Neutral scheduler authorization for one mutation epoch.

    This belongs at the scheduler/policy boundary, not in the device-resident
    policy implementation: cadence must remain executable in the torch-free
    CPU smoke contract.
    """

    epoch: int
    units: int
    geometry_opportunity: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.epoch, bool) or not isinstance(self.epoch, int) or self.epoch < 0:
            raise ValueError("epoch must be a nonnegative integer")
        if isinstance(self.units, bool) or not isinstance(self.units, int) or self.units < 0:
            raise ValueError("units must be a nonnegative integer")
        if type(self.geometry_opportunity) is not bool:
            raise TypeError("geometry_opportunity must be bool")


__all__ = ["MutationWorkToken"]
