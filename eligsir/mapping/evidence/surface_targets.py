"""Bounded, device-resident surface evidence targets.

This is an observation contract for the shadow depth/surface path.  It does
not own map rows and intentionally exposes no mutation operation.  A target
is a geometric hypothesis produced by evidence reduction; population policy
and the mutation arbiter decide what to do with it later.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import torch


class SurfaceEvidenceModality(Enum):
    RGB = "rgb"
    DEPTH = "depth"
    RGB_DEPTH = "rgb_depth"
    GEOMETRY = "geometry"


@dataclass(frozen=True)
class SurfaceTargetSnapshot:
    """Versions of the map snapshot from which targets were derived."""

    map_content: int
    topology: int
    coordinate_frame: int
    schema: int
    surface_evidence: int

    def __post_init__(self) -> None:
        for name in (
            "map_content", "topology", "coordinate_frame", "schema", "surface_evidence"
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} version must be a non-negative integer")


@dataclass(frozen=True)
class SurfaceTargetBatch:
    """A bounded batch of GPU surface hypotheses.

    All tensor members are borrowed and remain on one device.  The two view
    counters are deliberately separate: replay exposure is not independent
    physical-view evidence.
    """

    snapshot: SurfaceTargetSnapshot
    positions: torch.Tensor
    normals: torch.Tensor
    uncertainty: torch.Tensor
    support_evidence: torch.Tensor
    conflict_evidence: torch.Tensor
    independent_view_count: torch.Tensor
    repeated_exposure_count: torch.Tensor
    target_ids: torch.Tensor
    # Integer modality codes use the same device as the geometric tensors:
    # 1=RGB, 2=DEPTH, 3=RGB_DEPTH, 4=GEOMETRY.
    source_modality: torch.Tensor
    max_targets: int

    def __post_init__(self) -> None:
        if not isinstance(self.snapshot, SurfaceTargetSnapshot):
            raise TypeError("snapshot must be a SurfaceTargetSnapshot")
        if type(self.max_targets) is not int or self.max_targets <= 0:
            raise ValueError("max_targets must be a positive integer")
        tensors = (
            self.positions,
            self.normals,
            self.uncertainty,
            self.support_evidence,
            self.conflict_evidence,
            self.independent_view_count,
            self.repeated_exposure_count,
            self.target_ids,
            self.source_modality,
        )
        if any(not isinstance(value, torch.Tensor) for value in tensors):
            raise TypeError("all target fields must be torch.Tensor values")
        devices = {value.device for value in tensors}
        if len(devices) != 1:
            raise ValueError("all target tensors must share one device")
        count = self.positions.shape[0] if self.positions.ndim else -1
        if self.positions.ndim != 2 or self.positions.shape[1] != 3:
            raise ValueError("positions must have shape [K, 3]")
        if self.normals.ndim != 2 or self.normals.shape != self.positions.shape:
            raise ValueError("normals shape must match positions")
        if self.uncertainty.ndim != 2 or self.uncertainty.shape != self.positions.shape:
            raise ValueError("uncertainty shape must match positions")
        for name in (
            "support_evidence", "conflict_evidence", "independent_view_count",
            "repeated_exposure_count",
        ):
            value = getattr(self, name)
            if value.ndim != 1 or value.shape[0] != count:
                raise ValueError(f"{name} shape must be [K]")
        if count > self.max_targets:
            raise ValueError("target batch exceeds configured bound")
        if (self.source_modality.ndim != 1 or self.source_modality.shape[0] != count):
            raise ValueError("source_modality shape must be [K]")
        if self.target_ids.ndim != 1 or self.target_ids.shape[0] != count:
            raise ValueError("target_ids shape must be [K]")
        if self.target_ids.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
            raise TypeError("target_ids must use an integer dtype")
        if self.source_modality.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
            raise TypeError("source_modality must use an integer dtype")
        # Do not reduce device tensors to Python bools here.  This constructor
        # is used on the normal GPU path; content validation belongs to the
        # explicit debug validator below.

    @property
    def count(self) -> int:
        return self.positions.shape[0]

    @property
    def device(self) -> torch.device:
        return self.positions.device

    def is_current(self, current: SurfaceTargetSnapshot) -> bool:
        return self.snapshot == current

    def require_current(self, current: SurfaceTargetSnapshot) -> None:
        if not self.is_current(current):
            raise ValueError("surface target batch is stale for the current snapshot")

    def validate_debug(self) -> None:
        """Synchronously validate tensor contents for tests/diagnostics only."""

        for name in (
            "positions", "normals", "uncertainty", "support_evidence",
            "conflict_evidence",
        ):
            if not bool(torch.isfinite(getattr(self, name)).all()):
                raise ValueError(f"{name} must contain finite values")
        for name in (
            "uncertainty", "support_evidence", "conflict_evidence",
            "independent_view_count", "repeated_exposure_count",
        ):
            if bool((getattr(self, name) < 0).any()):
                raise ValueError(f"{name} must be non-negative")
        if bool(((self.source_modality < 1) | (self.source_modality > 4)).any()):
            raise ValueError("source_modality contains an unknown modality")
        if bool(torch.unique(self.target_ids).numel() != self.target_ids.numel()):
            raise ValueError("target_ids must be unique")

    @classmethod
    def select_topk(
        cls, candidates: "SurfaceTargetBatch", scores: torch.Tensor, capacity: int
    ) -> "SurfaceTargetBatch":
        """Select a bounded device-side top-k without changing evidence state."""

        if not isinstance(candidates, cls):
            raise TypeError("candidates must be a SurfaceTargetBatch")
        if type(capacity) is not int or capacity <= 0:
            raise ValueError("capacity must be a positive integer")
        if (
            not isinstance(scores, torch.Tensor)
            or scores.ndim != 1
            or scores.shape[0] != candidates.count
        ):
            raise ValueError("scores must be a one-dimensional tensor matching candidates")
        if scores.device != candidates.device:
            raise ValueError("scores must share the candidate device")
        if not scores.is_floating_point():
            raise TypeError("scores must use a floating-point dtype")
        count = min(capacity, candidates.count)
        # NaN scores cannot cross the host boundary for validation.  Treat
        # them as the least useful candidates deterministically on-device.
        safe_scores = torch.nan_to_num(scores, nan=float("-inf"))
        indices = torch.topk(safe_scores, k=count, sorted=True).indices
        return cls(
            snapshot=candidates.snapshot,
            positions=candidates.positions.index_select(0, indices),
            normals=candidates.normals.index_select(0, indices),
            uncertainty=candidates.uncertainty.index_select(0, indices),
            support_evidence=candidates.support_evidence.index_select(0, indices),
            conflict_evidence=candidates.conflict_evidence.index_select(0, indices),
            independent_view_count=candidates.independent_view_count.index_select(0, indices),
            repeated_exposure_count=candidates.repeated_exposure_count.index_select(0, indices),
            target_ids=candidates.target_ids.index_select(0, indices),
            source_modality=candidates.source_modality.index_select(0, indices),
            max_targets=count,
        )

    @staticmethod
    def validate_scores_debug(scores: torch.Tensor) -> None:
        """Validate score contents with an explicitly synchronous debug check."""

        if not bool(torch.isfinite(scores).all()):
            raise ValueError("scores must be finite")


__all__ = ["SurfaceEvidenceModality", "SurfaceTargetBatch", "SurfaceTargetSnapshot"]
