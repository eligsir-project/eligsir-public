"""Pure contracts for row-aligned Gaussian population state.

This module deliberately describes topology changes without applying them.
The mutation owner supplies one :class:`PopulationRowTransform`, and every
registered row-state family must explicitly accept that same transform.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import torch


@dataclass(frozen=True)
class RowStateDescriptor:
    """Metadata and lifecycle policy for one population-aligned field."""

    name: str
    family: str
    owner: str
    shape_suffix: tuple[int, ...]
    dtype: Any
    checkpoint: Any
    default: Any
    create: Any
    grow: Any
    relocate: Any
    retire: Any

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("row-state name must be non-empty")
        if not isinstance(self.family, str) or not self.family:
            raise ValueError("row-state family must be non-empty")
        if not isinstance(self.owner, str) or not self.owner:
            raise ValueError("row-state owner must be non-empty")
        if not isinstance(self.shape_suffix, tuple) or any(
            not isinstance(dim, int) or dim < 0 for dim in self.shape_suffix
        ):
            raise ValueError("shape_suffix must be a tuple of non-negative integers")
        if any(policy is None for policy in (self.create, self.grow, self.relocate, self.retire)):
            raise ValueError("every row-state field must declare lifecycle semantics")


@dataclass(frozen=True)
class PopulationRowCreation:
    """One newly created output row and its required new hypothesis identity."""

    row_index: int
    hypothesis_uid: Any
    parent_row_index: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.row_index, int) or self.row_index < 0:
            raise ValueError("created row_index must be a non-negative integer")
        if self.hypothesis_uid is None or self.hypothesis_uid == "":
            raise ValueError("CREATE requires a fresh hypothesis_uid")


@dataclass(frozen=True)
class PopulationRowRelocation:
    """A storage-slot reuse that ends one hypothesis and starts another."""

    row_index: int
    old_hypothesis_uid: Any
    new_hypothesis_uid: Any
    donor_row_index: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.row_index, int) or self.row_index < 0:
            raise ValueError("relocated row_index must be a non-negative integer")
        if self.old_hypothesis_uid is None or self.old_hypothesis_uid == "":
            raise ValueError("RELOCATE requires an old hypothesis_uid")
        if self.new_hypothesis_uid is None or self.new_hypothesis_uid == "":
            raise ValueError("RELOCATE requires a fresh hypothesis_uid")
        if self.new_hypothesis_uid == self.old_hypothesis_uid:
            raise ValueError("RELOCATE requires a fresh hypothesis_uid")


@dataclass(frozen=True)
class PopulationRowTransform:
    """Immutable, exact row mapping for one structural population mutation.

    Survivor rows appear in output order. Created rows occupy the remaining
    output rows; relocation targets reference output rows and never preserve a
    hypothesis identity.
    """

    old_population: int
    new_population: int
    survivor_old_indices: torch.Tensor
    retired_old_indices: torch.Tensor
    created_rows: tuple[PopulationRowCreation, ...]
    relocated_rows: tuple[PopulationRowRelocation, ...]
    covered_fields: frozenset[str]

    def __post_init__(self) -> None:
        if self.old_population < 0 or self.new_population < 0:
            raise ValueError("population counts must be non-negative")
        if not isinstance(self.survivor_old_indices, torch.Tensor) or not isinstance(
            self.retired_old_indices, torch.Tensor
        ):
            raise TypeError("row indices must be torch tensors")
        if self.survivor_old_indices.ndim != 1 or self.retired_old_indices.ndim != 1:
            raise ValueError("row indices must be one-dimensional")
        if self.survivor_old_indices.dtype not in _INTEGER_DTYPES or self.retired_old_indices.dtype not in _INTEGER_DTYPES:
            raise TypeError("row indices must have an integer dtype")
        if not isinstance(self.covered_fields, frozenset) or not all(
            isinstance(name, str) and name for name in self.covered_fields
        ):
            raise ValueError("covered_fields must be a frozenset of non-empty names")

    def validate(self, *, survivor_hypothesis_uids: Iterable[Any]) -> None:
        """Validate row counts and the exact old-to-new mapping."""
        survivor = self.survivor_old_indices.tolist()
        retired = self.retired_old_indices.tolist()
        _validate_old_indices(survivor, self.old_population, "survivor_old_indices")
        _validate_old_indices(retired, self.old_population, "retired_old_indices")
        if set(survivor).intersection(retired):
            raise ValueError("survivor and retired indices must be disjoint")
        if set(survivor).union(retired) != set(range(self.old_population)):
            raise ValueError("survivor and retired indices must cover every old row")

        relocated_indices = [row.row_index for row in self.relocated_rows]
        if len(set(relocated_indices)) != len(relocated_indices):
            raise ValueError("relocated row indices must be unique")
        if any(index >= self.new_population for index in relocated_indices):
            raise ValueError("relocated row index is outside the new population")
        survivor_output_indices = set(range(len(survivor)))
        if set(relocated_indices).intersection(survivor_output_indices):
            raise ValueError("relocated rows must be disjoint from survivor output rows")

        created_indices = [row.row_index for row in self.created_rows]
        if len(set(created_indices)) != len(created_indices):
            raise ValueError("created row indices must be unique")
        if set(relocated_indices).intersection(created_indices):
            raise ValueError("a row cannot be both created and relocated")

        expected_created = self.new_population - len(survivor) - len(relocated_indices)
        if expected_created < 0 or len(self.created_rows) != expected_created:
            raise ValueError("created and relocated rows must exactly cover the new population tail")
        expected_created_indices = set(range(len(survivor), self.new_population)) - set(relocated_indices)
        if set(created_indices) != expected_created_indices:
            raise ValueError("created and relocated rows must cover the new population tail")

        minted_uids = [row.hypothesis_uid for row in self.created_rows]
        minted_uids.extend(row.new_hypothesis_uid for row in self.relocated_rows)
        minted_uid_set = _uid_set(minted_uids, "minted hypothesis UIDs")
        if len(minted_uid_set) != len(minted_uids):
            raise ValueError("minted hypothesis UIDs must be globally unique")
        if minted_uid_set.intersection(
            _uid_set(survivor_hypothesis_uids, "survivor hypothesis UIDs")
        ):
            raise ValueError("minted hypothesis UID collides with a surviving hypothesis")


class GaussianRowStateRegistry:
    """Registry that makes complete row-state transform coverage mandatory."""

    def __init__(self, descriptors: Iterable[RowStateDescriptor] = ()) -> None:
        self._descriptors: dict[str, RowStateDescriptor] = {}
        for descriptor in descriptors:
            self.register(descriptor)

    @property
    def descriptors(self) -> tuple[RowStateDescriptor, ...]:
        return tuple(self._descriptors.values())

    @property
    def names(self) -> frozenset[str]:
        return frozenset(self._descriptors)

    def register(self, descriptor: RowStateDescriptor) -> None:
        if not isinstance(descriptor, RowStateDescriptor):
            raise TypeError("registry entries must be RowStateDescriptor instances")
        if descriptor.name in self._descriptors:
            raise ValueError(f"row-state field {descriptor.name!r} is already registered")
        self._descriptors[descriptor.name] = descriptor

    def validate_transform(
        self,
        transform: PopulationRowTransform,
        *,
        survivor_hypothesis_uids: Iterable[Any],
    ) -> None:
        if not isinstance(transform, PopulationRowTransform):
            raise TypeError("transform must be a PopulationRowTransform")
        transform.validate(survivor_hypothesis_uids=survivor_hypothesis_uids)
        if transform.covered_fields != self.names:
            missing = sorted(self.names - transform.covered_fields)
            extra = sorted(transform.covered_fields - self.names)
            raise ValueError(f"transform must cover exactly the registered fields; missing={missing}, extra={extra}")


_INTEGER_DTYPES = {torch.int8, torch.uint8, torch.int16, torch.int32, torch.int64}


def _validate_old_indices(indices: list[int], population: int, name: str) -> None:
    if len(set(indices)) != len(indices):
        raise ValueError(f"{name} must be unique")
    if any(index < 0 or index >= population for index in indices):
        raise ValueError(f"{name} contains an index outside the old population")


def _uid_set(uids: Iterable[Any], name: str) -> set[Any]:
    try:
        return set(uids)
    except TypeError as exc:
        raise TypeError(f"{name} must contain hashable UIDs") from exc
