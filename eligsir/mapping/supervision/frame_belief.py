"""Bounded, non-actuating frame-level supervision belief.

This module is a shadow projection of :class:`SupervisionEvidenceBatch`.  It
keeps compact facts keyed by the canonical source, modality, and resolution
tier.  It deliberately has no scheduler, renderer, or tensor-device policy;
callers decide when an already-produced evidence batch may be reduced.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import math
from typing import Iterable

from eligsir.contracts.control import ResolutionTier
from eligsir.contracts.runtime import SourceIdentity
from eligsir.optimization.gaussian.supervision_evidence import (
    SupervisionEvidenceBatch,
    SupervisionModality,
)


def _bounded(value: float, *, upper: float = 1.0) -> float:
    value = float(value)
    if not math.isfinite(value):
        return 0.0
    return max(0.0, min(upper, value))


def _modality_name(value: str | SupervisionModality) -> str:
    if isinstance(value, str):
        if value not in {"rgb", "depth"}:
            raise ValueError("modality must be 'rgb' or 'depth'")
        return value
    if not isinstance(value, SupervisionModality):
        raise TypeError("modality must be a string or SupervisionModality")
    if value.depth_loss_enabled or value.depth_evidence_enabled or value.geometry_evidence_enabled:
        return "depth"
    return "rgb"


@dataclass(frozen=True, slots=True)
class FrameSupervisionFact:
    """One compact observation fact, with all identity axes attached."""

    source: SourceIdentity
    modality: str | SupervisionModality
    tier: ResolutionTier
    intrinsic_reliability: float = 0.0
    rgb_debt: float = 0.0
    depth_debt: float = 0.0
    realized_effectiveness: float = 0.0
    complementarity: float = 0.0
    materialization_pixel_cost: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.source, SourceIdentity):
            raise TypeError("source must be SourceIdentity")
        object.__setattr__(self, "modality", _modality_name(self.modality))
        if not isinstance(self.tier, (str, ResolutionTier)):
            raise TypeError("tier must be ResolutionTier")
        object.__setattr__(self, "tier", ResolutionTier(self.tier))
        for name in ("intrinsic_reliability", "rgb_debt", "depth_debt", "realized_effectiveness", "complementarity"):
            object.__setattr__(self, name, _bounded(getattr(self, name)))
        object.__setattr__(self, "materialization_pixel_cost", _bounded(self.materialization_pixel_cost, upper=1e12))


@dataclass(frozen=True, slots=True)
class FrameModalityState:
    """Bounded state for one source/modality/tier identity."""

    source: SourceIdentity
    modality: str
    tier: ResolutionTier
    intrinsic_reliability: float
    rgb_debt: float
    depth_debt: float
    realized_effectiveness: float
    complementarity: float
    materialization_pixel_cost: float
    optimization_exposure_count: int = 1
    independent_physical_view_count: int = 1


class FrameSupervisionBelief:
    """Bounded facts only; no scheduling or mutation decisions are produced."""

    def __init__(self, *, max_frames: int = 256, max_exposures_per_frame: int = 32) -> None:
        if type(max_frames) is not int or max_frames < 1:
            raise ValueError("max_frames must be a positive integer")
        if type(max_exposures_per_frame) is not int or max_exposures_per_frame < 1:
            raise ValueError("max_exposures_per_frame must be a positive integer")
        self.max_frames = max_frames
        self.max_exposures_per_frame = max_exposures_per_frame
        self._states: OrderedDict[tuple[SourceIdentity, str, ResolutionTier], FrameModalityState] = OrderedDict()
        self._seen_sources: set[SourceIdentity] = set()

    @property
    def scheduling_decision(self):
        """Always ``None``: this component is observational and non-actuating."""

        return None

    @property
    def state_count(self) -> int:
        return len(self._states)

    @property
    def independent_physical_view_count(self) -> int:
        return len(self._seen_sources)

    def state(self, source: SourceIdentity, modality: str | SupervisionModality, tier: ResolutionTier) -> FrameModalityState | None:
        return self._states.get((source, _modality_name(modality), ResolutionTier(tier)))

    def update(self, fact: FrameSupervisionFact) -> None:
        if not isinstance(fact, FrameSupervisionFact):
            raise TypeError("fact must be FrameSupervisionFact")
        key = (fact.source, fact.modality, fact.tier)
        old = self._states.get(key)
        self._seen_sources.add(fact.source)
        if old is None:
            self._states[key] = FrameModalityState(
                source=fact.source, modality=fact.modality, tier=fact.tier,
                intrinsic_reliability=fact.intrinsic_reliability, rgb_debt=fact.rgb_debt,
                depth_debt=fact.depth_debt, realized_effectiveness=fact.realized_effectiveness,
                complementarity=fact.complementarity,
                materialization_pixel_cost=fact.materialization_pixel_cost,
            )
            self._trim()
            return
        # EMA-like averaging remains bounded and prevents replay from counting
        # as a new physical view. Exposure is deliberately capped.
        def avg(name: str) -> float:
            return _bounded((getattr(old, name) + getattr(fact, name)) / 2.0, upper=1e12 if name == "materialization_pixel_cost" else 1.0)
        self._states[key] = FrameModalityState(
            source=old.source, modality=old.modality, tier=old.tier,
            intrinsic_reliability=avg("intrinsic_reliability"), rgb_debt=avg("rgb_debt"),
            depth_debt=avg("depth_debt"), realized_effectiveness=avg("realized_effectiveness"),
            complementarity=avg("complementarity"), materialization_pixel_cost=avg("materialization_pixel_cost"),
            optimization_exposure_count=min(self.max_exposures_per_frame, old.optimization_exposure_count + 1),
            independent_physical_view_count=1,
        )

    def update_batch(
        self,
        batch: SupervisionEvidenceBatch,
        *,
        sources: Iterable[SourceIdentity],
        tiers: Iterable[ResolutionTier],
        materialization_pixel_costs: Iterable[float] | None = None,
        complementarities: Iterable[float] | None = None,
    ) -> None:
        if not isinstance(batch, SupervisionEvidenceBatch):
            raise TypeError("batch must be SupervisionEvidenceBatch")
        sources, tiers = tuple(sources), tuple(tiers)
        pixel_costs = tuple(materialization_pixel_costs) if materialization_pixel_costs is not None else (0.0,) * len(sources)
        complements = tuple(complementarities) if complementarities is not None else (0.0,) * len(sources)
        if len(sources) != len(batch.modalities) or len(tiers) != len(sources) or len(pixel_costs) != len(sources) or len(complements) != len(sources):
            raise ValueError("sources and tiers must match batch modalities")
        for i, (source, tier, modality) in enumerate(zip(sources, tiers, batch.modalities)):
            def row(name: str, default: float = 0.0) -> float:
                value = getattr(batch, name)
                if value is None:
                    return default
                # The evidence contract is device-resident.  This adapter is
                # intentionally fed only compact CPU facts; refusing a device
                # tensor keeps it from introducing an implicit synchronisation.
                if value.device.type != "cpu":
                    raise ValueError("FrameSupervisionBelief requires CPU compact facts")
                return default if value[i].numel() == 0 else float(value[i].float().mean().item())
            self.update(FrameSupervisionFact(
                source=source, modality=modality, tier=tier,
                intrinsic_reliability=row("intrinsic_reliability"), rgb_debt=row("rgb_residual_summary"),
                depth_debt=row("depth_residual_summary"), realized_effectiveness=row("alpha_support_summary"),
                complementarity=complements[i], materialization_pixel_cost=pixel_costs[i],
            ))

    def _trim(self) -> None:
        while len(self._states) > self.max_frames:
            self._states.popitem(last=False)
        self._seen_sources.intersection_update({key[0] for key in self._states})


__all__ = ["FrameModalityState", "FrameSupervisionBelief", "FrameSupervisionFact"]
