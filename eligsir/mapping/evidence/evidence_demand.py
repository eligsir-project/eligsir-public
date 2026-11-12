"""Pure, bounded contracts for policy-generated evidence requests.

This module deliberately does not resolve a demand to a frame or view.  That
choice belongs to the replay/supervision scheduler.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from numbers import Real
from typing import Any, Optional


class EvidenceDemandKind(str, Enum):
    """Evidence the population policy is asking the scheduler to obtain."""

    RGB_DETAIL = "rgb_detail"
    DEPTH_SURFACE = "depth_surface"
    DEPTH_FREE_SPACE = "depth_free_space"
    INDEPENDENT_VIEW = "independent_view"
    GENERAL_REOBSERVATION = "general_reobservation"


class EvidenceDemandModality(str, Enum):
    """Allowed supervision modalities for an evidence request."""

    RGB = "rgb"
    DEPTH = "depth"
    RGB_DEPTH = "rgb_depth"
    GENERAL = "general"


def _version(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} version must be a nonnegative integer")
    return value


def _optional_id(name: str, value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string when provided")
    return value


@dataclass(frozen=True)
class EvidenceDemand:
    """An immutable, versioned request for missing evidence.

    At least one of ``target_ids``, ``region_id`` or ``cell_id`` identifies
    the subject.  Snapshot versions are intentionally all required: a
    scheduler must be able to reject a demand that was generated from stale
    state without guessing which version dimension was omitted.
    """

    kind: EvidenceDemandKind
    requested_modality: EvidenceDemandModality
    target_ids: tuple[str, ...]
    map_version: int
    topology_version: int
    coordinate_version: int
    schema_version: int
    optimizer_version: int
    surface_version: int
    policy_version: int
    priority: Real
    expiry_epoch: Real
    region_id: Optional[str] = None
    cell_id: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, EvidenceDemandKind):
            raise TypeError("kind must be an EvidenceDemandKind")
        if not isinstance(self.requested_modality, EvidenceDemandModality):
            raise TypeError("requested_modality must be an EvidenceDemandModality")

        if not isinstance(self.target_ids, tuple):
            raise TypeError("target_ids must be a tuple of non-empty strings")
        ids = self.target_ids
        if any(not isinstance(item, str) or not item.strip() for item in ids):
            raise ValueError("target_ids must contain only non-empty strings")
        object.__setattr__(self, "target_ids", ids)
        region_id = _optional_id("region_id", self.region_id)
        cell_id = _optional_id("cell_id", self.cell_id)
        object.__setattr__(self, "region_id", region_id)
        object.__setattr__(self, "cell_id", cell_id)
        if not ids and region_id is None and cell_id is None:
            raise ValueError("demand requires a target ID, region_id, or cell_id")

        for name in (
            "map_version",
            "topology_version",
            "coordinate_version",
            "schema_version",
            "optimizer_version",
            "surface_version",
            "policy_version",
        ):
            _version(name, getattr(self, name))
        if (
            isinstance(self.priority, bool)
            or not isinstance(self.priority, Real)
            or not math.isfinite(self.priority)
            or self.priority <= 0
        ):
            raise ValueError("priority must be finite and positive")
        if (
            isinstance(self.expiry_epoch, bool)
            or not isinstance(self.expiry_epoch, Real)
            or not math.isfinite(self.expiry_epoch)
            or self.expiry_epoch <= 0
        ):
            raise ValueError("expiry_epoch must be finite and positive")

    @property
    def snapshot_versions(self) -> dict[str, int]:
        return {
            "map": self.map_version,
            "topology": self.topology_version,
            "coordinate": self.coordinate_version,
            "schema": self.schema_version,
            "optimizer": self.optimizer_version,
            "surface": self.surface_version,
            "policy": self.policy_version,
        }

    @property
    def key(self) -> tuple[Any, ...]:
        """Stable identity used to reject duplicate outstanding demands."""

        return (
            self.kind,
            self.requested_modality,
            self.target_ids,
            self.region_id,
            self.cell_id,
        )

    def describe(self) -> dict[str, Any]:
        """Return scheduler metadata without selecting any frame or view."""

        return {
            "kind": self.kind.value,
            "requested_modality": self.requested_modality.value,
            "target_ids": self.target_ids,
            "region_id": self.region_id,
            "cell_id": self.cell_id,
            "priority": self.priority,
            "expiry_epoch": self.expiry_epoch,
            "snapshot_versions": self.snapshot_versions,
        }


@dataclass(frozen=True)
class EvidenceDemandBatch:
    """A finite collection of distinct demands for one scheduler handoff."""

    demands: tuple[EvidenceDemand, ...]
    max_size: int = 64

    def __post_init__(self) -> None:
        if isinstance(self.max_size, bool) or not isinstance(self.max_size, int) or self.max_size <= 0:
            raise ValueError("max_size must be a positive integer bound")
        demands = tuple(self.demands)
        if len(demands) > self.max_size:
            raise ValueError("evidence demand batch exceeds its bound")
        if any(not isinstance(demand, EvidenceDemand) for demand in demands):
            raise TypeError("batch entries must be EvidenceDemand instances")
        keys = [demand.key for demand in demands]
        if len(keys) != len(set(keys)):
            raise ValueError("evidence demand batch contains a duplicate key")
        object.__setattr__(self, "demands", demands)

    def __len__(self) -> int:
        return len(self.demands)

    def __iter__(self):
        return iter(self.demands)
