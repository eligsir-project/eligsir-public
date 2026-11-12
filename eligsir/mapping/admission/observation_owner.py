"""Terminal CPU-side observation disposition guards for issue #1190.

The immutable control records live in :mod:`eligsir.contracts.control`.
This module deliberately adds no policy/scoring model: it adapts the existing
bounded backpressure proposal into that record and makes terminal decisions
enforceable at the materialization boundary.
"""

from __future__ import annotations

from collections import OrderedDict
import copy
from dataclasses import dataclass, replace
import math
from typing import Any, Callable, Optional, TypeVar

from eligsir.contracts.control import (
    ControlSnapshot,
    DispositionKind,
    EvidenceFact,
    HistoricalEligibility,
    ModalityEligibility,
    ObservationDisposition,
    ResidencyIntent,
    SemanticAction,
)
from eligsir.contracts.runtime import SourceIdentity
from eligsir.mapping.scheduler.backpressure import (
    BackpressureAction,
    BackpressureOwner,
    BackpressureDecision,
    PendingObservation,
)


class DispositionConflict(RuntimeError):
    """Raised when a terminal disposition is changed for one source identity."""


@dataclass(frozen=True, slots=True)
class MaterializationPermit:
    """Capability derived only from an admitted terminal disposition."""

    source: SourceIdentity
    snapshot_id: int


@dataclass(frozen=True, slots=True)
class ColdRetentionPermit:
    """Capability to retain CPU pixels or a reload token, never GPU work."""

    source: SourceIdentity
    snapshot_id: int


@dataclass(frozen=True, slots=True)
class HistoricalPromotionPermit:
    """A current-snapshot capability to materialize one deferred source later."""

    source: SourceIdentity
    disposition_snapshot_id: int
    promotion_snapshot_id: int
    control_version: int
    topology_version: int


class DispositionLedger:
    """Keep terminal dispositions immutable across downstream consumers."""

    def __init__(self, *, recent_capacity: int = 4096) -> None:
        if recent_capacity <= 0:
            raise ValueError("recent_capacity must be positive")
        # This is an idempotency cache, not historical storage. Durable
        # disposition provenance belongs to the runtime event/observation store;
        # an indefinitely-running camera must not retain every identity here.
        self._recent_capacity = int(recent_capacity)
        self._records: OrderedDict[SourceIdentity, ObservationDisposition] = OrderedDict()

    def record(self, disposition: ObservationDisposition) -> ObservationDisposition:
        prior = self._records.get(disposition.source)
        if prior is None:
            self._records[disposition.source] = disposition
            if len(self._records) > self._recent_capacity:
                self._records.popitem(last=False)
            return disposition
        if prior != disposition:
            raise DispositionConflict(
                "terminal disposition cannot be changed for source "
                f"{disposition.source.fingerprint()}"
            )
        return prior

    @property
    def recent_count(self) -> int:
        """Return the bounded count of retry-visible dispositions."""

        return len(self._records)

    def disposition_for(self, source: SourceIdentity) -> Optional[ObservationDisposition]:
        """Return the recorded terminal disposition for one full identity."""

        disposition = self._records.get(source)
        if disposition is not None:
            self._records.move_to_end(source)
        return disposition

    def permit(self, disposition: ObservationDisposition) -> Optional[MaterializationPermit]:
        recorded = self.record(disposition)
        if not recorded.materialization_permitted:
            return None
        return MaterializationPermit(recorded.source, recorded.snapshot_id)

    def cold_retention_permit(
        self, disposition: ObservationDisposition
    ) -> Optional[ColdRetentionPermit]:
        """Mint a CPU/reload-token retention capability for a recorded DEFER."""

        recorded = self.record(disposition)
        if not recorded.cold_retention_permitted:
            return None
        return ColdRetentionPermit(recorded.source, recorded.snapshot_id)


_KIND_BY_ACTION = {
    BackpressureAction.ADMIT: DispositionKind.ADMIT,
    BackpressureAction.METADATA_ONLY: DispositionKind.METADATA_ONLY,
    BackpressureAction.COALESCE_INTO: DispositionKind.COALESCE,
    BackpressureAction.SKIP_REDUNDANT: DispositionKind.SKIP_REDUNDANT,
    BackpressureAction.SKIP_CAPACITY: DispositionKind.SKIP_CAPACITY,
}

_COLD_DEFER_REASONS = frozenset(
    {
        "no_motion",
        "redundant",
        "skip_redundant",
        "rej_redundant_angle",
        "rej_coverage",
        "rej_parallax",
    }
)


def modality_eligibility_from_evidence(
    snapshot: ControlSnapshot, decision: BackpressureDecision
) -> ModalityEligibility:
    """Resolve source modality authority from immutable ingress evidence.

    Older proposal providers remain allowed to omit modality facts while they
    are being migrated; that compatibility is deliberately confined here, at
    the one terminal owner, rather than copied into allocators or executors.
    """

    facts = {fact.name: fact.value for fact in snapshot.evidence}
    admitted = bool(decision.admit_supervision)
    rgb = admitted and bool(facts.get("rgb_supervision_eligible", True))
    depth = admitted and bool(facts.get("depth_supervision_eligible", True))
    insertion = (
        bool(decision.allow_geometry_insertion)
        and bool(facts.get("depth_insertion_eligible", depth))
        and depth
    )
    population_target = depth and bool(
        facts.get("geometry_population_target_eligible", depth)
    )
    return ModalityEligibility(
        rgb_supervision=rgb,
        rgb_quality=str(facts.get("rgb_quality", "qualified" if rgb else "declined")),
        depth_supervision=depth,
        depth_quality=str(facts.get("depth_quality", "qualified" if depth else "declined")),
        depth_insertion=insertion,
        geometry_population_target=population_target,
    )


def _historical_modality_eligibility(
    snapshot: ControlSnapshot,
) -> ModalityEligibility:
    """Resolve modalities worth retaining without authorizing work now."""

    facts = {fact.name: fact.value for fact in snapshot.evidence}
    rgb = bool(facts.get("rgb_supervision_eligible", True))
    depth = bool(facts.get("depth_supervision_eligible", True))
    insertion = depth and bool(facts.get("depth_insertion_eligible", True))
    population_target = depth and bool(
        facts.get("geometry_population_target_eligible", True)
    )
    return ModalityEligibility(
        rgb_supervision=rgb,
        rgb_quality=str(facts.get("rgb_quality", "qualified" if rgb else "declined")),
        depth_supervision=depth,
        depth_quality=str(
            facts.get("depth_quality", "qualified" if depth else "declined")
        ),
        depth_insertion=insertion,
        geometry_population_target=population_target,
    )


def disposition_from_backpressure(
    snapshot: ControlSnapshot,
    decision: BackpressureDecision,
    *,
    representative: Optional[SourceIdentity] = None,
) -> ObservationDisposition:
    """Adapt #601's cheap proposal to #1190's one terminal record.

    The caller supplies the full representative identity because the legacy
    decision carries only an integer frame id. That prevents a cross-epoch
    collision from being silently converted into a coalescing relation.
    """

    source = snapshot.source
    if source is None:
        raise ValueError("a source-specific disposition requires snapshot.source")
    if decision.frame_id != source.source_ordinal:
        raise ValueError("backpressure decision frame_id must match snapshot source ordinal")
    kind = _KIND_BY_ACTION[decision.action]
    if kind is DispositionKind.COALESCE:
        if representative is None:
            raise ValueError("coalesced backpressure decisions require a full representative")
        if representative.source_ordinal != decision.representative_frame_id:
            raise ValueError("representative identity must match backpressure representative_frame_id")
    elif representative is not None:
        raise ValueError("only coalesced backpressure decisions may name a representative")

    trace = [
        EvidenceFact("backpressure.action", decision.action.value),
        EvidenceFact("backpressure.pressure_state", decision.pressure_state.value),
        EvidenceFact("backpressure.pending_depth", decision.pending_depth),
        EvidenceFact(
            "backpressure.pending_age_control_ticks",
            decision.pending_age_control_ticks,
        ),
        EvidenceFact(
            "backpressure.pending_age_source_s",
            decision.pending_age_source_s,
        ),
        EvidenceFact(
            "backpressure.decided_control_tick",
            decision.decided_control_tick,
        ),
    ]
    trace.extend(
        EvidenceFact(f"backpressure.signal.{name}", value)
        for name, value in sorted(decision.signals.items())
    )
    residency = (
        ResidencyIntent.HOT
        if kind is DispositionKind.ADMIT
        else ResidencyIntent.METADATA_ONLY
    )
    modalities = modality_eligibility_from_evidence(snapshot, decision)
    admit_supervision = decision.admit_supervision
    allow_geometry_insertion = modalities.depth_insertion
    mapping_candidate = decision.mapping_candidate
    reason = decision.reason
    if kind is DispositionKind.ADMIT and not (
        modalities.rgb_supervision
        or modalities.depth_supervision
        or modalities.depth_insertion
    ):
        kind = DispositionKind.SKIP_QUALITY
        residency = ResidencyIntent.METADATA_ONLY
        representative = None
        admit_supervision = False
        allow_geometry_insertion = False
        mapping_candidate = False
        reason = "modality_quality_rejected"
    return ObservationDisposition(
        source=source,
        snapshot_id=snapshot.snapshot_id,
        kind=kind,
        observe=decision.observe,
        mapping_candidate=mapping_candidate,
        admit_supervision=admit_supervision,
        allow_geometry_insertion=allow_geometry_insertion,
        residency=residency,
        reason=reason,
        representative=representative,
        evidence_trace=tuple(trace),
        modality_eligibility=modalities,
    )


BackpressureProposal = Callable[[ControlSnapshot], BackpressureDecision]


class _BackpressureProposalProvider:
    """Evaluate the legacy owner on a copy and commit only terminally agreed state."""

    def __init__(self, owner: BackpressureOwner) -> None:
        self._owner = owner
        self._staged: dict[
            tuple[int, SourceIdentity], tuple[BackpressureOwner, BackpressureDecision]
        ] = {}

    def propose(self, snapshot: ControlSnapshot) -> BackpressureDecision:
        source = snapshot.source
        if source is None:
            raise ValueError("backpressure proposal requires snapshot.source")
        facts = {fact.name: fact.value for fact in snapshot.evidence}
        pending = PendingObservation(
            frame_id=source.source_ordinal,
            source_timestamp=source.timestamp_seconds,
            release_control_tick=snapshot.control_tick_epoch,
            cheap_keyframe=bool(facts.get("cheap_keyframe", False)),
            novelty=float(facts.get("novelty", 0.0)),
            quality=float(facts.get("quality", 0.0)),
            motion=float(facts.get("motion", 0.0)),
            translation_m=float(facts.get("translation_m", 0.0)),
            rotation_deg=float(facts.get("rotation_deg", 0.0)),
        )
        candidate = copy.deepcopy(self._owner)
        decisions = candidate.submit(
            pending,
            control_tick=snapshot.control_tick_epoch,
            source_time_s=source.timestamp_seconds,
        )
        current = [item for item in decisions if item.frame_id == source.source_ordinal]
        if len(current) != 1:
            raise RuntimeError(
                "BackpressureOwner did not emit exactly one terminal decision "
                "for the current source"
            )
        decision = current[0]
        self._staged[(snapshot.snapshot_id, source)] = (candidate, decision)
        return decision

    def commit(
        self, snapshot: ControlSnapshot, disposition: ObservationDisposition
    ) -> None:
        staged = self._staged.pop((snapshot.snapshot_id, disposition.source), None)
        if staged is None:
            raise RuntimeError("backpressure proposal was not staged for terminal commit")
        candidate, proposal = staged
        proposed_admit = proposal.action is BackpressureAction.ADMIT
        finally_admitted = disposition.kind is DispositionKind.ADMIT
        if proposed_admit != finally_admitted:
            return
        self._owner.__dict__.clear()
        self._owner.__dict__.update(candidate.__dict__)


class ObservationOwner:
    """The sole injectable owner of terminal semantic frame disposition.

    It owns the full-identity ledger. Existing mechanisms, including
    :class:`BackpressureOwner`, supply a cheap proposal; they never bypass the
    resulting disposition or mint a materialization permit themselves.
    """

    def __init__(
        self,
        proposal: BackpressureProposal | object,
        *,
        ledger: Optional[DispositionLedger] = None,
        representative_lookup: Optional[
            Callable[[ControlSnapshot, int], SourceIdentity]
        ] = None,
        semantic_providers: tuple[object, ...] = (),
    ) -> None:
        propose = getattr(proposal, "propose", None)
        if callable(propose):
            self._proposal = propose
            self._proposal_provider = proposal
        elif callable(proposal):
            self._proposal = proposal
            self._proposal_provider = None
        else:
            raise TypeError("proposal must be callable or define propose(snapshot)")
        self._ledger = ledger or DispositionLedger()
        self._representative_lookup = representative_lookup
        self._semantic_providers = tuple(semantic_providers)
        if any(not callable(getattr(item, "propose", None)) for item in self._semantic_providers):
            raise TypeError("semantic providers must define propose(snapshot)")

    @classmethod
    def from_backpressure(
        cls,
        owner: BackpressureOwner,
        *,
        ledger: Optional[DispositionLedger] = None,
        representative_lookup: Optional[
            Callable[[ControlSnapshot, int], SourceIdentity]
        ] = None,
        semantic_providers: tuple[object, ...] = (),
    ) -> "ObservationOwner":
        """Wrap an existing #601 owner as a proposal source.

        A cohort that defers the current source has not emitted a terminal
        disposition. Failing closed is intentional: root integration must flush
        or otherwise resolve it before any materialization can occur.
        """

        if not isinstance(owner, BackpressureOwner):
            raise TypeError("owner must be BackpressureOwner")

        return cls(
            _BackpressureProposalProvider(owner),
            ledger=ledger,
            representative_lookup=representative_lookup,
            semantic_providers=semantic_providers,
        )

    def decide(self, snapshot: ControlSnapshot) -> ObservationDisposition:
        """Return one immutable terminal disposition for ``snapshot.source``."""

        source = snapshot.source
        if source is None:
            raise ValueError("ObservationOwner.decide requires snapshot.source")
        recorded = self._ledger.disposition_for(source)
        if recorded is not None:
            return recorded
        from eligsir.mapping.admission.keyframe_provider import (
            KeyframeAdmissionProposal,
        )

        semantic = []
        for provider in self._semantic_providers:
            item = provider.propose(snapshot)
            if not isinstance(item, KeyframeAdmissionProposal):
                raise TypeError(
                    "semantic providers must return KeyframeAdmissionProposal"
                )
            semantic.append(item)
        semantic_trace = []
        for item in semantic:
            semantic_trace.extend(
                (
                    EvidenceFact(f"keyframe.{item.provider}.admitted", item.admitted),
                    EvidenceFact(f"keyframe.{item.provider}.reason", item.reason),
                )
            )
            for name, value in item.trace:
                if isinstance(value, (bool, int, float, str, type(None))) and not (
                    isinstance(value, float) and not math.isfinite(value)
                ):
                    semantic_trace.append(
                        EvidenceFact(f"keyframe.{item.provider}.{name}", value)
                    )
        rejections = tuple(item for item in semantic if not item.admitted)
        rejection = next(
            (item for item in rejections if item.reason not in _COLD_DEFER_REASONS),
            rejections[0] if rejections else None,
        )
        if rejection is not None:
            action = (
                SemanticAction.DEFER
                if rejection.reason in _COLD_DEFER_REASONS
                else SemanticAction.REJECT
            )
            modalities = (
                ModalityEligibility()
                if action is SemanticAction.REJECT
                else _historical_modality_eligibility(snapshot)
            )
            disposition = ObservationDisposition(
                source=source,
                snapshot_id=snapshot.snapshot_id,
                kind=(
                    DispositionKind.SKIP_REDUNDANT
                    if action is SemanticAction.DEFER
                    else DispositionKind.SKIP_QUALITY
                ),
                observe=True,
                mapping_candidate=False,
                admit_supervision=False,
                allow_geometry_insertion=False,
                residency=(
                    ResidencyIntent.METADATA_ONLY
                    if action is SemanticAction.REJECT
                    else ResidencyIntent.COLD
                ),
                reason=rejection.reason,
                evidence_trace=tuple(semantic_trace),
                modality_eligibility=modalities,
                semantic_action=action,
                historical_eligibility=(
                    HistoricalEligibility.NONE
                    if action is SemanticAction.REJECT
                    else HistoricalEligibility.ELIGIBLE
                ),
            )
            self._commit_providers(snapshot, disposition, semantic=True, backpressure=False)
            return self._ledger.record(disposition)
        proposal = self._proposal(snapshot)
        representative = None
        if proposal.action is BackpressureAction.COALESCE_INTO:
            if self._representative_lookup is None:
                raise ValueError("coalesced proposal requires representative_lookup")
            representative = self._representative_lookup(
                snapshot, int(proposal.representative_frame_id)
            )
        disposition = disposition_from_backpressure(
            snapshot,
            proposal,
            representative=representative,
        )
        if semantic_trace:
            disposition = replace(
                disposition,
                reason=(semantic[0].reason if disposition.kind is DispositionKind.ADMIT else disposition.reason),
                evidence_trace=disposition.evidence_trace + tuple(semantic_trace),
            )
        self._commit_providers(snapshot, disposition, semantic=True, backpressure=True)
        return self._ledger.record(disposition)

    def _commit_providers(
        self,
        snapshot: ControlSnapshot,
        disposition: ObservationDisposition,
        *,
        semantic: bool,
        backpressure: bool,
    ) -> None:
        if semantic:
            for provider in self._semantic_providers:
                commit = getattr(provider, "commit", None)
                if callable(commit):
                    commit(snapshot, disposition)
        if backpressure and self._proposal_provider is not None:
            commit = getattr(self._proposal_provider, "commit", None)
            if callable(commit):
                commit(snapshot, disposition)

    def permit(self, disposition: ObservationDisposition) -> Optional[MaterializationPermit]:
        """Return a capability only for the identical recorded ADMIT decision."""

        recorded = self._ledger.disposition_for(disposition.source)
        if recorded != disposition:
            raise DispositionConflict(
                "materialization requires the disposition recorded by this owner"
            )
        return self._ledger.permit(recorded)

    def cold_retention_permit(
        self, disposition: ObservationDisposition
    ) -> Optional[ColdRetentionPermit]:
        """Return a capability only for the identical recorded DEFER decision."""

        recorded = self._ledger.disposition_for(disposition.source)
        if recorded != disposition:
            raise DispositionConflict(
                "cold retention requires the disposition recorded by this owner"
            )
        return self._ledger.cold_retention_permit(recorded)

    def promotion_permit(
        self,
        disposition: ObservationDisposition,
        current_snapshot: ControlSnapshot,
    ) -> Optional[HistoricalPromotionPermit]:
        """Fence later DEFER materialization to a newer immutable snapshot.

        The terminal disposition is not rewritten.  A rejected observation
        therefore cannot be resurrected, while an eligible deferred source can
        be paged only under explicitly current control/topology versions.
        """

        if not isinstance(current_snapshot, ControlSnapshot):
            raise TypeError("current_snapshot must be ControlSnapshot")
        recorded = self._ledger.disposition_for(disposition.source)
        if recorded != disposition:
            raise DispositionConflict(
                "historical promotion requires the disposition recorded by this owner"
            )
        if not recorded.historical_promotion_eligible:
            return None
        if current_snapshot.snapshot_id <= recorded.snapshot_id:
            raise ValueError("historical promotion requires a newer control snapshot")
        return HistoricalPromotionPermit(
            source=recorded.source,
            disposition_snapshot_id=recorded.snapshot_id,
            promotion_snapshot_id=current_snapshot.snapshot_id,
            control_version=current_snapshot.control_version,
            topology_version=current_snapshot.topology_version,
        )

    def promotion_permit_from_retained(
        self,
        *,
        source: SourceIdentity,
        disposition_snapshot_id: int,
        semantic_action: str,
        current_snapshot: ControlSnapshot,
    ) -> Optional[HistoricalPromotionPermit]:
        """Mint a promotion fence from durable DB-owned DEFER provenance.

        The recent ledger is intentionally bounded, so an indefinitely old
        deferred frame cannot depend on it. The observation DB supplies the
        original terminal action and snapshot; only an explicit DEFER may
        cross this boundary under a newer control/topology snapshot.
        """

        if not isinstance(current_snapshot, ControlSnapshot):
            raise TypeError("current_snapshot must be ControlSnapshot")
        if str(semantic_action) != SemanticAction.DEFER.value:
            return None
        disposition_snapshot_id = int(disposition_snapshot_id)
        if current_snapshot.snapshot_id <= disposition_snapshot_id:
            raise ValueError("historical promotion requires a newer control snapshot")
        return HistoricalPromotionPermit(
            source=source,
            disposition_snapshot_id=disposition_snapshot_id,
            promotion_snapshot_id=current_snapshot.snapshot_id,
            control_version=current_snapshot.control_version,
            topology_version=current_snapshot.topology_version,
        )


T = TypeVar("T")
Materialize = Callable[[MaterializationPermit], T]
PixelSink = Callable[[MaterializationPermit, T], Any]


def apply_disposition(
    disposition: ObservationDisposition,
    *,
    permit: Optional[MaterializationPermit],
    materialize: Materialize[T],
    transfer: PixelSink[T],
    retain_for_replay: PixelSink[T],
    insert_geometry: PixelSink[T],
) -> bool:
    """Run pixel-work sinks only with an owner-issued ADMIT capability.

    This narrow seam is intentionally dependency-injected so CPU tests can use
    poison sinks to prove a declined observation never reaches materialization,
    H2D, replay, or insertion. Root runtime wiring belongs to the integration
    owner, not this contract module.  The helper never owns a ledger: doing so
    would let a caller bypass :class:`ObservationOwner` by minting a fresh
    permit for an arbitrary disposition.
    """

    if permit is None:
        return False
    if not disposition.materialization_permitted:
        raise DispositionConflict("a declined disposition cannot materialize")
    if permit.source != disposition.source or permit.snapshot_id != disposition.snapshot_id:
        raise DispositionConflict("materialization permit does not match disposition")
    payload = materialize(permit)
    transfer(permit, payload)
    retain_for_replay(permit, payload)
    if disposition.allow_geometry_insertion:
        insert_geometry(permit, payload)
    return True


__all__ = [
    "ColdRetentionPermit",
    "DispositionConflict",
    "DispositionLedger",
    "HistoricalPromotionPermit",
    "MaterializationPermit",
    "ObservationOwner",
    "apply_disposition",
    "disposition_from_backpressure",
]
