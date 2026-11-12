"""Strict, short-lived reuse of normal RGB-D supervision render products.

This carries no population decision and never performs a render, transfer,
synchronization, or mutation.  Direct depth insertion and B4 population-target
production are distinct consumers; each must satisfy its own explicit
eligibility while sharing the exact source/snapshot/native/device checks.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import torch

from eligsir.contracts import ModalityEligibility, SourceIdentity
from eligsir.optimization.gaussian.supervision_evidence import EvidenceSnapshotToken
from eligsir.runtime.prepared_source import PreparedSourceLease


class DepthInsertionReuseReason(str, Enum):
    """Attribution for reuse or the explicit compatibility fallback."""

    REUSED = "reused"
    MISSING_RENDER_LEASE = "missing_render_lease"
    STALE_SNAPSHOT = "stale_snapshot"
    SOURCE_MISMATCH = "source_mismatch"
    NON_NATIVE_TIER = "non_native_tier"
    GEOMETRY_INELIGIBLE = "geometry_ineligible"
    MISSING_SOURCE_LEASE = "missing_source_lease"
    SOURCE_LEASE_SOURCE_MISMATCH = "source_lease_source_mismatch"
    SOURCE_LEASE_GEOMETRY_INELIGIBLE = "source_lease_geometry_ineligible"
    SOURCE_LEASE_TIER_MISSING = "source_lease_tier_missing"
    SOURCE_LEASE_PAYLOAD_MISSING = "source_lease_payload_missing"
    DEVICE_MISMATCH = "device_mismatch"
    NATIVE_LATTICE_MISMATCH = "native_lattice_mismatch"


@dataclass(frozen=True, slots=True)
class GeometryRenderLease:
    """One geometry-eligible view from the normal supervision render.

    ``snapshot_token`` describes the pre-Adam render state.  The tensors are
    borrowed and valid only through the immediately following insertion prep
    boundary; no caller may retain them as durable surface evidence.
    """

    source_identity: SourceIdentity
    snapshot_token: EvidenceSnapshotToken
    resolution_tier: str
    modality_eligibility: ModalityEligibility
    alpha: torch.Tensor
    rendered_depth: torch.Tensor

    def __post_init__(self) -> None:
        if not isinstance(self.source_identity, SourceIdentity):
            raise TypeError("source_identity must be SourceIdentity")
        if not isinstance(self.snapshot_token, EvidenceSnapshotToken):
            raise TypeError("snapshot_token must be EvidenceSnapshotToken")
        if not isinstance(self.resolution_tier, str) or not self.resolution_tier:
            raise ValueError("resolution_tier must be non-empty")
        if not isinstance(self.modality_eligibility, ModalityEligibility):
            raise TypeError("modality_eligibility must be ModalityEligibility")
        if not isinstance(self.alpha, torch.Tensor) or not isinstance(
            self.rendered_depth, torch.Tensor
        ):
            raise TypeError("alpha and rendered_depth must be tensors")
        if self.alpha.device != self.rendered_depth.device:
            raise ValueError("alpha and rendered_depth must share one device")
        if self.alpha.ndim < 2 or self.rendered_depth.ndim < 2:
            raise ValueError("alpha and rendered_depth must be image tensors")


@dataclass(frozen=True, slots=True)
class DepthInsertionRenderReuse:
    """All borrowed device values needed by an exact native geometry consumer."""

    alpha: torch.Tensor
    rendered_depth: torch.Tensor
    depth_meters: torch.Tensor
    rgb: torch.Tensor
    c2w: torch.Tensor
    camera_intrinsics: torch.Tensor | None = None
    source_h2d_ops: int = 0
    extra_rasterizer_calls: int = 0

    def __post_init__(self) -> None:
        values = (
            self.alpha,
            self.rendered_depth,
            self.depth_meters,
            self.rgb,
            self.c2w,
        )
        if any(not isinstance(value, torch.Tensor) for value in values):
            raise TypeError("render/source reuse values must be tensors")
        if len({value.device for value in values}) != 1:
            raise ValueError("render/source reuse values must share one device")
        if self.depth_meters.ndim != 2:
            raise ValueError("reused depth_meters must have shape [H, W]")
        if self.rgb.ndim != 3 or self.rgb.shape[:2] != self.depth_meters.shape:
            raise ValueError("reused rgb must have shape [H, W, C] matching depth")
        if self.c2w.shape != (4, 4):
            raise ValueError("reused c2w must have shape [4, 4]")
        if self.camera_intrinsics is not None and (
            not isinstance(self.camera_intrinsics, torch.Tensor)
            or self.camera_intrinsics.shape != (3, 3)
            or self.camera_intrinsics.device != self.depth_meters.device
        ):
            raise ValueError(
                "reused camera_intrinsics must have shape [3, 3] on the source device"
            )
        if self.source_h2d_ops != 0 or self.extra_rasterizer_calls != 0:
            raise ValueError("reused render/source payload cannot charge upload or render work")


def _bind_geometry_render_reuse(
    *,
    render_lease: GeometryRenderLease | None,
    source_lease: PreparedSourceLease | None,
    source: SourceIdentity,
    expected_snapshot: EvidenceSnapshotToken,
    expected_resolution_tier: str,
    population_target: bool,
) -> tuple[DepthInsertionRenderReuse | None, DepthInsertionReuseReason]:
    """Bind only an exact native geometry view; otherwise fail closed.

    The returned object is intentionally a tiny compatibility input for the
    existing device insertion producer.  All fallback decisions are explicit
    and do not perform work, allowing the production caller to account for the
    legacy render/upload path accurately.
    """

    if not isinstance(source, SourceIdentity):
        raise TypeError("source must be SourceIdentity")
    if not isinstance(expected_snapshot, EvidenceSnapshotToken):
        raise TypeError("expected_snapshot must be EvidenceSnapshotToken")
    if not isinstance(expected_resolution_tier, str) or not expected_resolution_tier:
        raise ValueError("expected_resolution_tier must be non-empty")
    if render_lease is None:
        return None, DepthInsertionReuseReason.MISSING_RENDER_LEASE
    if not isinstance(render_lease, GeometryRenderLease):
        raise TypeError("render_lease must be GeometryRenderLease or None")
    if render_lease.snapshot_token != expected_snapshot:
        return None, DepthInsertionReuseReason.STALE_SNAPSHOT
    if render_lease.source_identity != source:
        return None, DepthInsertionReuseReason.SOURCE_MISMATCH
    # Native is deliberately the only initial activation: occupancy semantics
    # are not resolution invariant and legacy insertion samples native depth.
    if (
        expected_resolution_tier != "native"
        or render_lease.resolution_tier != "native"
    ):
        return None, DepthInsertionReuseReason.NON_NATIVE_TIER
    eligible = (
        render_lease.modality_eligibility.geometry_population_target
        if population_target
        else render_lease.modality_eligibility.depth_insertion
    )
    if not eligible:
        return None, DepthInsertionReuseReason.GEOMETRY_INELIGIBLE
    if source_lease is None:
        return None, DepthInsertionReuseReason.MISSING_SOURCE_LEASE
    if not isinstance(source_lease, PreparedSourceLease):
        raise TypeError("source_lease must be PreparedSourceLease or None")
    if source_lease.source_identity != source:
        return None, DepthInsertionReuseReason.SOURCE_LEASE_SOURCE_MISMATCH
    source_eligible = (
        source_lease.modality_eligibility.geometry_population_target
        if population_target
        else source_lease.modality_eligibility.depth_insertion
    )
    if not source_eligible:
        return None, DepthInsertionReuseReason.SOURCE_LEASE_GEOMETRY_INELIGIBLE
    if "native" not in source_lease.available_tiers:
        return None, DepthInsertionReuseReason.SOURCE_LEASE_TIER_MISSING
    payloads = source_lease.payloads
    try:
        depth_meters = payloads["depth_meters"].data
        rgb = payloads["rgb"].data
        c2w = payloads["c2w"].data
    except KeyError:
        return None, DepthInsertionReuseReason.SOURCE_LEASE_PAYLOAD_MISSING
    if any(
        not isinstance(value, torch.Tensor)
        for value in (depth_meters, rgb, c2w)
    ):
        return None, DepthInsertionReuseReason.SOURCE_LEASE_PAYLOAD_MISSING
    camera_intrinsics_ref = payloads.get("camera_intrinsics")
    camera_intrinsics = (
        None
        if camera_intrinsics_ref is None
        else getattr(camera_intrinsics_ref, "data", None)
    )
    if camera_intrinsics is not None and not isinstance(
        camera_intrinsics, torch.Tensor
    ):
        return None, DepthInsertionReuseReason.SOURCE_LEASE_PAYLOAD_MISSING
    if str(render_lease.alpha.device) != source_lease.device:
        return None, DepthInsertionReuseReason.DEVICE_MISMATCH
    native_hw = tuple(int(value) for value in depth_meters.shape)

    def matches_native_lattice(image: torch.Tensor) -> bool:
        """Accept CHW/HWC/HW image layouts only when their HxW is native."""

        return image.ndim >= 2 and (
            tuple(int(value) for value in image.shape[-2:]) == native_hw
            or tuple(int(value) for value in image.shape[:2]) == native_hw
        )

    if not (
        matches_native_lattice(render_lease.alpha)
        and matches_native_lattice(render_lease.rendered_depth)
    ):
        return None, DepthInsertionReuseReason.NATIVE_LATTICE_MISMATCH
    try:
        return (
            DepthInsertionRenderReuse(
                alpha=render_lease.alpha,
                rendered_depth=render_lease.rendered_depth,
                depth_meters=depth_meters,
                rgb=rgb,
                c2w=c2w,
                camera_intrinsics=camera_intrinsics,
            ),
            DepthInsertionReuseReason.REUSED,
        )
    except ValueError:
        return None, DepthInsertionReuseReason.DEVICE_MISMATCH


def bind_depth_insertion_render_reuse(
    *,
    render_lease: GeometryRenderLease | None,
    source_lease: PreparedSourceLease | None,
    source: SourceIdentity,
    expected_snapshot: EvidenceSnapshotToken,
    expected_resolution_tier: str,
) -> tuple[DepthInsertionRenderReuse | None, DepthInsertionReuseReason]:
    """Bind geometry for the legacy direct depth-insertion owner only."""

    return _bind_geometry_render_reuse(
        render_lease=render_lease,
        source_lease=source_lease,
        source=source,
        expected_snapshot=expected_snapshot,
        expected_resolution_tier=expected_resolution_tier,
        population_target=False,
    )


def bind_population_target_render_reuse(
    *,
    render_lease: GeometryRenderLease | None,
    source_lease: PreparedSourceLease | None,
    source: SourceIdentity,
    expected_snapshot: EvidenceSnapshotToken,
    expected_resolution_tier: str,
) -> tuple[DepthInsertionRenderReuse | None, DepthInsertionReuseReason]:
    """Bind geometry for the population target producer, never insertion."""

    return _bind_geometry_render_reuse(
        render_lease=render_lease,
        source_lease=source_lease,
        source=source,
        expected_snapshot=expected_snapshot,
        expected_resolution_tier=expected_resolution_tier,
        population_target=True,
    )


__all__ = [
    "DepthInsertionRenderReuse",
    "DepthInsertionReuseReason",
    "GeometryRenderLease",
    "bind_population_target_render_reuse",
    "bind_depth_insertion_render_reuse",
]
