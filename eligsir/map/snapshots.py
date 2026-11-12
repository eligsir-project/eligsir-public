"""Owned, host-facing snapshot payload and request contract.

Wave 1 consolidation plan, step 2 (see
the snapshot boundary.

``MapSnapshot`` is deliberately distinct from
:class:`eligsir.map.views.GaussianMapReadView`: a snapshot is an owned,
immutable payload meant for viewers, persistence, export, and asynchronous
consumers that outlive a single map version — not the hot render/optimize
path. The request/provider indirection exists specifically so that a
producer (backend or export adapter) keeps full control over how it packs
and transfers data.

This matters concretely: ``eligsir/viewers/rvizsplat_export.py``
already packs positions, scales, rotations, opacities, and SH into one
contiguous GPU tensor followed by a *single* device-to-host transfer. A
naive "snapshot = one dataclass with N separate tensor fields, each
transferred independently" design would silently regress that into N
transfers. Routing every snapshot consumer through a ``SnapshotRequest`` ->
``SnapshotProvider.create()`` call keeps that specialized packing path
inside the backend/export adapter, and never hands the mutable live model
to a viewer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Optional, Protocol

if TYPE_CHECKING:
    import numpy as np


class SnapshotPurpose(Enum):
    """Why a snapshot is being requested — lets the provider pick its
    packing/transfer strategy without the caller needing to know it."""

    RVIZ_SPLAT = "rviz_splat"
    RERUN_VIEWER = "rerun_viewer"
    PLY_EXPORT = "ply_export"
    CHECKPOINT_EXPORT = "checkpoint_export"
    DEBUG_INSPECTION = "debug_inspection"


@dataclass(frozen=True)
class SnapshotRequest:
    """Parameters governing how a :class:`MapSnapshot` should be produced.

    ``coordinate_basis`` is left as an opaque string/enum-like value here
    (rather than importing a concrete convention type) so this module has
    no dependency on any particular viewer's coordinate-frame contract;
    concrete adapters (e.g. the RViz export path) interpret it.
    """

    purpose: SnapshotPurpose
    coordinate_basis: Optional[str] = None
    max_gaussians: Optional[int] = None
    include_sh: bool = True


@dataclass(frozen=True)
class MapSnapshot:
    """An owned, immutable, host-facing payload describing the live map at
    a point in time. Producers are free to choose their own internal
    packing strategy (e.g. one contiguous packed buffer for a single D2H
    transfer) — this dataclass only fixes the *logical* fields a consumer
    can rely on, not how they were physically transferred.
    """

    positions: "np.ndarray"
    scales: "np.ndarray"
    quaternions: "np.ndarray"
    opacities: "np.ndarray"
    spherical_harmonics: "np.ndarray"

    map_version: int
    purpose: SnapshotPurpose

    # Producer-specific extra fields (e.g. packed transfer buffers, per-purpose
    # metadata) that don't belong in the logical contract above but a
    # specific adapter may still want to carry alongside the snapshot.
    extra: dict = field(default_factory=dict)


class SnapshotProvider(Protocol):
    """Implemented by a map backend or a dedicated export adapter. The
    caller never touches the live model directly — only this method."""

    def create(self, request: SnapshotRequest) -> MapSnapshot:
        ...
