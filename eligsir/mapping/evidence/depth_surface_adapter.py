"""Adapt device depth candidates into the bounded surface-target contract.

This module is deliberately an observation boundary.  It accepts values that
an existing depth/surface producer already computed and packages them for the
shadow population path.  It does not render, transfer, synchronize, mutate
Gaussian rows, or choose an action.  In particular, replay exposure and
independent physical-view support are carried as separate counters.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from eligsir.mapping.evidence.surface_targets import (
    SurfaceTargetBatch,
    SurfaceTargetSnapshot,
)


@dataclass(frozen=True)
class DepthSurfaceEvidenceInput:
    """Already-reduced device tensors from one depth evidence transaction.

    The producer-validity precondition is explicit: the producer must have
    sanitized these tensors at the evidence boundary (finite geometry and
    evidence, non-negative evidence/counts) before constructing this value.
    Shape/device/dtype checks below are hot-path safe; use
    :meth:`validate_debug` at a test or diagnostics boundary for synchronous
    content validation.
    """

    positions: torch.Tensor
    normals: torch.Tensor
    uncertainty: torch.Tensor
    support_evidence: torch.Tensor
    conflict_evidence: torch.Tensor
    independent_view_count: torch.Tensor
    repeated_exposure_count: torch.Tensor
    target_ids: torch.Tensor

    def __post_init__(self) -> None:
        fields = (
            self.positions,
            self.normals,
            self.uncertainty,
            self.support_evidence,
            self.conflict_evidence,
            self.independent_view_count,
            self.repeated_exposure_count,
            self.target_ids,
        )
        if any(not isinstance(value, torch.Tensor) for value in fields):
            raise TypeError("depth evidence fields must be tensors")
        devices = {value.device for value in fields}
        if len(devices) != 1:
            raise ValueError("depth evidence fields must share one device")
        if self.positions.ndim != 2 or self.positions.shape[1] != 3:
            raise ValueError("positions must have shape [K, 3]")
        count = self.positions.shape[0]
        if self.normals.shape != self.positions.shape or self.uncertainty.shape != self.positions.shape:
            raise ValueError("normals and uncertainty must have shape [K, 3]")
        for name in (
            "support_evidence",
            "conflict_evidence",
            "independent_view_count",
            "repeated_exposure_count",
        ):
            value = getattr(self, name)
            if value.ndim != 1 or value.shape[0] != count:
                raise ValueError(f"{name} must have shape [K]")
        for name in ("positions", "normals", "uncertainty", "support_evidence", "conflict_evidence"):
            if not getattr(self, name).dtype.is_floating_point:
                raise TypeError(f"{name} must use a floating-point dtype")
        for name in ("independent_view_count", "repeated_exposure_count"):
            if getattr(self, name).dtype not in (
                torch.int8, torch.int16, torch.int32, torch.int64
            ):
                raise TypeError(f"{name} must use an integer dtype")
        if self.target_ids.ndim != 1 or self.target_ids.shape[0] != count:
            raise ValueError("target_ids must have shape [K]")
        if self.target_ids.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
            raise TypeError("target_ids must use an integer dtype")

    @property
    def count(self) -> int:
        return self.positions.shape[0]

    @property
    def device(self) -> torch.device:
        return self.positions.device

    def validate_debug(self) -> None:
        """Synchronously validate producer content for tests/diagnostics.

        This method is intentionally separate from construction so normal GPU
        evidence conversion never performs a reduction or host synchronization.
        """

        for name in (
            "positions",
            "normals",
            "uncertainty",
            "support_evidence",
            "conflict_evidence",
        ):
            if not bool(torch.isfinite(getattr(self, name)).all()):
                raise ValueError(f"{name} must contain finite values")
        for name in (
            "uncertainty",
            "support_evidence",
            "conflict_evidence",
            "independent_view_count",
            "repeated_exposure_count",
        ):
            if bool((getattr(self, name) < 0).any()):
                raise ValueError(f"{name} must be non-negative")


def adapt_depth_surface_evidence(
    evidence: DepthSurfaceEvidenceInput,
    *,
    snapshot: SurfaceTargetSnapshot,
    max_targets: int,
    candidate_mask: torch.Tensor | None = None,
) -> SurfaceTargetBatch:
    """Create a bounded, versioned target batch from existing device output.

    ``candidate_mask`` is an optional device boolean mask supplied by the
    producer.  Selection is performed with device indexing; no tensor value is
    converted to a Python scalar.  If the producer supplies more candidates
    than the bound, the first bounded prefix is retained deterministically.
    This is a compatibility adapter, not a ranking or population decision.
    """

    if not isinstance(evidence, DepthSurfaceEvidenceInput):
        raise TypeError("evidence must be DepthSurfaceEvidenceInput")
    if not isinstance(snapshot, SurfaceTargetSnapshot):
        raise ValueError("snapshot must be SurfaceTargetSnapshot")
    if type(max_targets) is not int or max_targets <= 0:
        raise ValueError("max_targets must be a positive integer")
    if candidate_mask is not None:
        if not isinstance(candidate_mask, torch.Tensor):
            raise TypeError("candidate_mask must be a tensor")
        if candidate_mask.dtype is not torch.bool:
            raise ValueError("candidate_mask must be a boolean tensor")
        if candidate_mask.ndim != 1 or candidate_mask.shape[0] != evidence.count:
            raise ValueError("candidate_mask must have shape [K]")
        if candidate_mask.device != evidence.device:
            raise ValueError("candidate_mask must share the evidence device")
        selected = torch.where(candidate_mask)[0]
    else:
        selected = torch.arange(evidence.count, device=evidence.device, dtype=torch.int64)
    selected = selected[:max_targets]

    def pick(value: torch.Tensor) -> torch.Tensor:
        return value.detach().index_select(0, selected)

    modality = torch.full(
        (selected.shape[0],), 4, dtype=torch.int8, device=evidence.device
    )
    return SurfaceTargetBatch(
        snapshot=snapshot,
        positions=pick(evidence.positions),
        normals=pick(evidence.normals),
        uncertainty=pick(evidence.uncertainty),
        support_evidence=pick(evidence.support_evidence),
        conflict_evidence=pick(evidence.conflict_evidence),
        independent_view_count=pick(evidence.independent_view_count),
        repeated_exposure_count=pick(evidence.repeated_exposure_count),
        target_ids=pick(evidence.target_ids),
        source_modality=modality,
        max_targets=max_targets,
    )


__all__ = ["DepthSurfaceEvidenceInput", "adapt_depth_surface_evidence"]
