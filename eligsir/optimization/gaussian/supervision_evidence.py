"""Device-resident supervision evidence contracts.

This module intentionally contains only immutable container contracts.  Tensor
members are borrowed, zero-copy device values: callers retain ownership and
must not mutate them while the container is in use.  PyTorch has no safe
read-only tensor interface, so this layer neither clones nor wraps tensors.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable

import torch

from eligsir.contracts.control import ModalityEligibility
from eligsir.contracts.runtime import SourceIdentity


class RenderRequirement(Enum):
    """The output mode required from one homogeneous render group."""

    RGB = "rgb"
    RGB_DEPTH = "rgb_depth"


@dataclass(frozen=True)
class SupervisionModality:
    """Independent supervision switches for one view."""

    rgb_loss_enabled: bool = True
    depth_loss_enabled: bool = False
    rgb_evidence_enabled: bool = True
    depth_evidence_enabled: bool = False
    geometry_evidence_enabled: bool = False

    def __post_init__(self) -> None:
        for name in (
            "rgb_loss_enabled",
            "depth_loss_enabled",
            "rgb_evidence_enabled",
            "depth_evidence_enabled",
            "geometry_evidence_enabled",
        ):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be bool")

    @property
    def requests_depth(self) -> bool:
        return (
            self.depth_loss_enabled
            or self.depth_evidence_enabled
            or self.geometry_evidence_enabled
        )


def group_render_requirement(
    modalities: Iterable[SupervisionModality],
) -> RenderRequirement:
    """Return the shared render mode while retaining each view's switches."""

    return (
        RenderRequirement.RGB_DEPTH
        if any(modality.requests_depth for modality in modalities)
        else RenderRequirement.RGB
    )


@dataclass(frozen=True)
class EvidenceSnapshotToken:
    """Exact non-negative versions identifying the source snapshot."""

    map_content: int
    topology: int
    coordinate_frame: int
    schema: int
    optimizer_update: int
    surface_evidence: int
    policy: int
    parameter: int

    def __post_init__(self) -> None:
        for name in (
            "map_content",
            "topology",
            "coordinate_frame",
            "schema",
            "optimizer_update",
            "surface_evidence",
            "policy",
            "parameter",
        ):
            value = getattr(self, name)
            if type(value) is not int:
                raise TypeError(f"{name} version must be int")
            if value < 0:
                raise ValueError(f"{name} version must be non-negative")


_PER_VIEW_FIELDS = (
    "rgb_residual_summary",
    "depth_residual_summary",
    "rgb_valid_pixels",
    "depth_valid_pixels",
    "alpha_support_summary",
    "intrinsic_reliability",
)
_PER_GAUSSIAN_FIELDS = (
    "support_evidence",
    "contradiction_evidence",
    "render_contribution",
    "gradient_utility_evidence",
    "meaningful_measurement_opportunity",
)


@dataclass(frozen=True)
class SupervisionEvidenceBatch:
    """Bounded compact evidence tensors produced by one optimization step.

    Tensor fields are optional because evidence availability is explicit at
    the field level.  Every supplied tensor remains device-resident and must
    share one device.  Per-view tensors have leading dimension ``B`` and
    per-Gaussian tensors leading dimension ``G``; no reduction or host
    conversion is performed here.
    """

    snapshot_token: EvidenceSnapshotToken | None
    transaction_id: int | str
    modalities: tuple[SupervisionModality, ...]
    # Exact per-view identities make a B>1 geometry lease auditable.  This is
    # optional for compatibility-only evidence consumers; M6 target binding
    # deliberately fails closed when it is unavailable.
    source_identities: tuple[SourceIdentity, ...] = ()
    modality_eligibility: tuple[ModalityEligibility, ...] = ()
    rgb_residual_summary: torch.Tensor | None = None
    depth_residual_summary: torch.Tensor | None = None
    rgb_valid_pixels: torch.Tensor | None = None
    depth_valid_pixels: torch.Tensor | None = None
    alpha_support_summary: torch.Tensor | None = None
    intrinsic_reliability: torch.Tensor | None = None
    gaussian_ids: torch.Tensor | None = None
    view_incidence: torch.Tensor | None = None
    meaningful_measurement_opportunity: torch.Tensor | None = None
    support_evidence: torch.Tensor | None = None
    contradiction_evidence: torch.Tensor | None = None
    render_contribution: torch.Tensor | None = None
    gradient_utility_evidence: torch.Tensor | None = None
    # Compatibility handoff for a pre-existing optimizer-only evidence value
    # (for example the energy-gradient payload).  It is intentionally opaque:
    # this batch neither interprets nor owns that policy-specific datum.
    optimization_payload: object | None = None
    # Short-lived all-view geometry render products used only by the immediate
    # insertion-preparation compatibility bridge.  They are not persistent
    # evidence maps and must never be reconstructed from a primary-only hook.
    geometry_render_leases: tuple[object, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.snapshot_token, EvidenceSnapshotToken):
            raise ValueError("snapshot token is required")
        if not isinstance(self.modalities, tuple) or not all(
            isinstance(modality, SupervisionModality) for modality in self.modalities
        ):
            raise TypeError("modalities must be a tuple of SupervisionModality")
        view_count = len(self.modalities)
        if not isinstance(self.source_identities, tuple):
            raise TypeError("source_identities must be a tuple")
        if self.source_identities:
            if (
                len(self.source_identities) != view_count
                or not all(
                    isinstance(source, SourceIdentity)
                    for source in self.source_identities
                )
                or len(set(self.source_identities)) != view_count
            ):
                raise ValueError(
                    "source_identities must be unique SourceIdentity values matching modalities"
                )
        if not self.modality_eligibility and view_count:
            object.__setattr__(
                self,
                "modality_eligibility",
                tuple(
                    ModalityEligibility(
                        rgb_supervision=modality.rgb_loss_enabled,
                        depth_supervision=modality.depth_loss_enabled,
                        geometry_population_target=modality.geometry_evidence_enabled,
                    )
                    for modality in self.modalities
                ),
            )
        if len(self.modality_eligibility) != view_count or not all(
            isinstance(value, ModalityEligibility) for value in self.modality_eligibility
        ):
            raise ValueError("modality_eligibility must match modalities")
        if not isinstance(self.geometry_render_leases, tuple):
            raise TypeError("geometry_render_leases must be a tuple")
        # Import locally to keep the standalone evidence contract independent
        # of mapping integration at import time.
        if self.geometry_render_leases:
            from eligsir.mapping.integration.depth_render_reuse import GeometryRenderLease

            if not all(
                isinstance(item, GeometryRenderLease)
                for item in self.geometry_render_leases
            ):
                raise TypeError("geometry_render_leases must contain GeometryRenderLease")
            if any(
                item.snapshot_token != self.snapshot_token
                for item in self.geometry_render_leases
            ):
                raise ValueError("geometry render leases must match evidence snapshot")
        tensors = {
            name: getattr(self, name)
            for name in (*_PER_VIEW_FIELDS, *_PER_GAUSSIAN_FIELDS, "gaussian_ids", "view_incidence")
            if getattr(self, name) is not None
        }
        for name, tensor in tensors.items():
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor")
            if tensor.ndim == 0:
                raise ValueError(f"{name} must have a compact leading dimension")

        devices = {tensor.device for tensor in tensors.values()}
        if len(devices) > 1:
            raise ValueError("all evidence tensors must share one device")

        for name in _PER_VIEW_FIELDS:
            tensor = getattr(self, name)
            if tensor is not None and tensor.shape[0] != view_count:
                raise ValueError(f"{name} has mismatched per-view shape")

        gaussian_ids = self.gaussian_ids
        gaussian_candidates = [
            tensor.shape[0]
            for name, tensor in tensors.items()
            if name in (*_PER_GAUSSIAN_FIELDS, "gaussian_ids", "view_incidence")
        ]
        gaussian_count = gaussian_candidates[0] if gaussian_candidates else 0
        if any(count != gaussian_count for count in gaussian_candidates):
            raise ValueError("mismatched per-Gaussian shape")
        if gaussian_ids is not None:
            if gaussian_ids.ndim != 1:
                raise ValueError("gaussian_ids must be one-dimensional")
            if gaussian_ids.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
                raise TypeError("gaussian_ids must use an integer dtype")
        for name in _PER_GAUSSIAN_FIELDS:
            tensor = getattr(self, name)
            if tensor is not None and tensor.shape[0] != gaussian_count:
                raise ValueError(f"{name} has mismatched per-Gaussian shape")
        if self.view_incidence is not None:
            if self.view_incidence.ndim != 2 or self.view_incidence.shape[1] != view_count:
                raise ValueError("view_incidence must have shape [G, B]")
            if self.view_incidence.shape[0] != gaussian_count:
                raise ValueError("view_incidence has mismatched per-Gaussian shape")

    @property
    def gaussian_count(self) -> int:
        """Number of compact Gaussian rows, inferred without tensor reads."""

        for name in (*_PER_GAUSSIAN_FIELDS, "gaussian_ids", "view_incidence"):
            tensor = getattr(self, name)
            if tensor is not None:
                return tensor.shape[0]
        return 0

    @property
    def payload(self) -> object | None:
        """Compatibility projection for existing optimizer evidence consumers."""

        return self.optimization_payload


__all__ = [
    "EvidenceSnapshotToken",
    "RenderRequirement",
    "SupervisionEvidenceBatch",
    "SupervisionModality",
    "group_render_requirement",
]
