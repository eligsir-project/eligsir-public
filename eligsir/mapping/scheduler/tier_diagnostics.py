"""Per-transaction resolution diagnostics -- EXPLANATORY DATA, NEVER AN INPUT.

Everything in this module exists to *explain* an observed per-tier cost after
the fact.  **No scheduler, controller or allocator may read it.**  In
particular ``isects_per_tile`` and the tile-list distribution below are here
because a ScanNet++ sweep found a U-shaped cost curve whose ascending limb
correlates with ``isects_per_tile`` crossing roughly 500-750 -- and that number
is a property of one workload, one gsplat version and one GPU.  Encoding it as
a threshold anywhere in production would bake a measurement artifact into
control flow.

The correct handling of a cost inversion is already in place and needs nothing
from this module: ``WorkBudgetOwner`` selects tiers by *measured* transaction
cost (PR #1251), so if a tier genuinely becomes slow -- for this reason or any
other -- the estimator simply stops selecting it.  This module's only job is to
let a human answer "why did that happen?" afterwards.

If you are about to wire a threshold off a field in here: don't.  Measure the
cost instead and let ``TierCostEstimator`` react to it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Optional, Tuple

#: Bounded ring size.  Diagnostics must never grow with run length.
DEFAULT_DIAGNOSTIC_CAPACITY: int = 256


@dataclass(frozen=True, slots=True)
class RasterTileDiagnostics:
    """Tile-density explanation for one rasterization, from gsplat meta."""

    tile_count: int
    visible_nnz: int
    intersection_count: int
    isects_per_tile: float
    tile_list_p50: float
    tile_list_p95: float
    tile_list_max: int
    tile_imbalance_p95_p50: float


@dataclass(frozen=True, slots=True)
class TierTransactionDiagnostic:
    """One completed transaction, annotated for post-hoc explanation only."""

    resolution_tier: str
    gaussian_count: int
    camera_count: int
    transaction_ms: float
    prepare_ms: float
    residency: str
    tile_count: Optional[int] = None
    visible_nnz: Optional[int] = None
    intersection_count: Optional[int] = None
    isects_per_tile: Optional[float] = None
    tile_list_p50: Optional[float] = None
    tile_list_p95: Optional[float] = None
    tile_list_max: Optional[int] = None
    tile_imbalance_p95_p50: Optional[float] = None

    def as_dict(self) -> dict:
        return asdict(self)


def _percentile(ordered: list[int], fraction: float) -> float:
    if not ordered:
        return 0.0
    import math

    index = int(math.ceil(fraction * len(ordered))) - 1
    return float(ordered[max(0, min(len(ordered) - 1, index))])


def tile_diagnostics_from_meta(meta: Mapping[str, Any]) -> Optional[RasterTileDiagnostics]:
    """Derive tile-density diagnostics from gsplat packed-mode metadata.

    Best-effort and fail-soft: any missing or unexpected field yields ``None``
    rather than raising, because this is explanatory telemetry and must never
    be able to fail a training transaction.  ``isect_offsets`` is a prefix-sum
    over tiles, so per-tile intersection-list lengths are its consecutive
    differences.
    """

    if not isinstance(meta, Mapping):
        return None
    offsets = meta.get("isect_offsets")
    flatten_ids = meta.get("flatten_ids")
    if offsets is None or flatten_ids is None:
        return None
    try:
        flat = offsets.reshape(-1)
        total = int(flatten_ids.shape[0])
        boundaries = flat.tolist()
        if len(boundaries) < 1:
            return None
        boundaries = [int(value) for value in boundaries]
        lengths = [
            max(0, second - first)
            for first, second in zip(boundaries, boundaries[1:] + [total])
        ]
    except Exception:
        return None
    tile_count = len(lengths)
    if tile_count <= 0:
        return None
    ordered = sorted(lengths)
    p50 = _percentile(ordered, 0.50)
    p95 = _percentile(ordered, 0.95)
    visible = meta.get("gaussian_ids")
    try:
        visible_nnz = int(visible.shape[0]) if visible is not None else 0
    except Exception:
        visible_nnz = 0
    return RasterTileDiagnostics(
        tile_count=tile_count,
        visible_nnz=visible_nnz,
        intersection_count=total,
        isects_per_tile=float(total) / float(tile_count),
        tile_list_p50=p50,
        tile_list_p95=p95,
        tile_list_max=int(ordered[-1]),
        tile_imbalance_p95_p50=(p95 / p50) if p50 > 0.0 else 0.0,
    )


class TierTransactionDiagnostics:
    """Bounded ring of per-transaction explanatory records."""

    def __init__(self, capacity: int = DEFAULT_DIAGNOSTIC_CAPACITY) -> None:
        if isinstance(capacity, bool) or int(capacity) < 1:
            raise ValueError("capacity must be a positive integer")
        self._capacity = int(capacity)
        self._records: list[TierTransactionDiagnostic] = []
        self.observed = 0

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def records(self) -> Tuple[TierTransactionDiagnostic, ...]:
        return tuple(self._records)

    def record(self, diagnostic: TierTransactionDiagnostic) -> None:
        if not isinstance(diagnostic, TierTransactionDiagnostic):
            raise TypeError("diagnostic must be TierTransactionDiagnostic")
        self.observed += 1
        self._records.append(diagnostic)
        if len(self._records) > self._capacity:
            del self._records[0]

    def snapshot(self) -> dict:
        """Report-shaped view, explicitly self-labelled as non-control data."""

        return {
            "schema_note": (
                "explanatory diagnostics only; no scheduler, controller or "
                "allocator reads these fields, and no threshold may be wired "
                "off them -- tier choice is made from measured transaction "
                "cost (see scheduler/tier_cost_model.py)."
            ),
            "observed": int(self.observed),
            "retained": len(self._records),
            "capacity": self._capacity,
            "transactions": [record.as_dict() for record in self._records],
        }


__all__ = [
    "DEFAULT_DIAGNOSTIC_CAPACITY",
    "RasterTileDiagnostics",
    "TierTransactionDiagnostic",
    "TierTransactionDiagnostics",
    "tile_diagnostics_from_meta",
]
