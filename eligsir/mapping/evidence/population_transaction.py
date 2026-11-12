"""First bounded M2 evidence-to-population vertical seam.

This module composes the existing fixed-topology B4 evidence, policy, target,
and mutation-fence contracts.  It intentionally accepts an already-qualified
resident target: production target extraction from the complete native B4
geometry lease remains a follow-up integration (and must not be faked here).
There is no render, source upload, adaptive growth, depth insertion, or
lifecycle fallback in this seam.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

from eligsir.utils.gaussian_belief import GaussianBelief

from eligsir.mapping.evidence.surface_targets import SurfaceTargetBatch
from eligsir.mapping.population_plan_adapter import (
    PopulationPlanAdapterContext,
    PopulationPolicyPlanAdapter,
)
from eligsir.mapping.population_policy import (
    GaussianPopulationPolicy,
    MutationWorkToken,
    PopulationDecision,
    PopulationEvidence,
)
from eligsir.optimization.gaussian.supervision_evidence import EvidenceSnapshotToken

if TYPE_CHECKING:
    from eligsir.map.arbiter import PopulationMutationProposal


@dataclass(frozen=True, slots=True)
class PopulationEvidenceTransactionResult:
    """Result of one exact evidence/policy/mutation-fence attempt."""

    proposal: "PopulationMutationProposal | None"
    reason: str


class PopulationEvidenceTransaction:
    """Run one K=1 CREATE-or-ABSTAIN population transaction.

    ``target`` must be the complete, bounded target output of an upstream B4
    evidence transaction.  Its device-resident qualification is materialized
    only by the adapter's existing attributed mutation-cardinality fence. The
    optional ``evidence_update`` is a
    fixed-topology update hook (for example a device-native belief fold); it
    runs only after all snapshot/shape guards pass and may return the exact
    snapshot token to attest that it did not cross a topology fence.
    """

    def __init__(
        self,
        *,
        policy: GaussianPopulationPolicy | None = None,
        adapter: PopulationPolicyPlanAdapter | None = None,
    ) -> None:
        self.policy = policy or GaussianPopulationPolicy()
        self.adapter = adapter or PopulationPolicyPlanAdapter()

    def build(
        self,
        *,
        belief: GaussianBelief,
        evidence: PopulationEvidence,
        work_token: MutationWorkToken,
        snapshot: EvidenceSnapshotToken,
        target: SurfaceTargetBatch,
        context: PopulationPlanAdapterContext,
        evidence_update: Callable[[], object] | None = None,
    ) -> PopulationEvidenceTransactionResult:
        """Return one atomic CREATE proposal, or a reason-coded abstention."""

        if not isinstance(snapshot, EvidenceSnapshotToken):
            return PopulationEvidenceTransactionResult(None, "ABSTAIN_INVALID_SNAPSHOT")
        if not isinstance(context, PopulationPlanAdapterContext):
            return PopulationEvidenceTransactionResult(None, "ABSTAIN_INVALID_CONTEXT")
        if context.current_snapshot != snapshot or context.decision_snapshot != snapshot:
            return PopulationEvidenceTransactionResult(None, "ABSTAIN_STALE_SNAPSHOT")
        if (
            not isinstance(work_token, MutationWorkToken)
            or work_token.units != 1
            or context.max_actions != 1
        ):
            return PopulationEvidenceTransactionResult(None, "ABSTAIN_WORK_BOUND")
        if not isinstance(target, SurfaceTargetBatch) or target.count != 1:
            return PopulationEvidenceTransactionResult(None, "ABSTAIN_TARGET_BOUND")
        expected = self.adapter._expected_surface_snapshot(snapshot)
        if target.snapshot != expected or target.max_targets != 1:
            return PopulationEvidenceTransactionResult(None, "ABSTAIN_STALE_TARGET")
        if not isinstance(belief, GaussianBelief) or not isinstance(evidence, PopulationEvidence):
            return PopulationEvidenceTransactionResult(None, "ABSTAIN_INVALID_EVIDENCE")

        # The hook is deliberately after all pre-fence identity checks.  It is
        # fixed-topology by contract; returning a different token fails closed.
        if evidence_update is not None:
            if not callable(evidence_update):
                return PopulationEvidenceTransactionResult(None, "ABSTAIN_INVALID_UPDATE")
            updated = evidence_update()
            if updated is not None and updated != snapshot:
                return PopulationEvidenceTransactionResult(None, "ABSTAIN_UPDATE_FENCE")

        versions = {
            "map": snapshot.map_content,
            "topology": snapshot.topology,
            "coordinate": snapshot.coordinate_frame,
            "schema": snapshot.schema,
            "optimizer": snapshot.optimizer_update,
            "surface": snapshot.surface_evidence,
            "policy": snapshot.policy,
        }
        try:
            decision = self.policy.decide(
                belief,
                work_token,
                evidence,
                capacity=context.capacity,
                snapshot_versions=versions,
            )
            if not isinstance(decision, PopulationDecision):
                return PopulationEvidenceTransactionResult(None, "ABSTAIN_INVALID_DECISION")
            from eligsir.mapping.evidence.production_surface_targets import bind_policy_surface_targets

            bound = bind_policy_surface_targets(decision, target, snapshot, work_token)
            if bound is None:
                return PopulationEvidenceTransactionResult(None, "ABSTAIN_POLICY_TARGET")
            decision, target = bound
            built = self.adapter.build(decision=decision, targets=target, context=context)
        except (TypeError, ValueError, RuntimeError):
            return PopulationEvidenceTransactionResult(None, "ABSTAIN_INVALID_FENCE")
        if built.proposal is None:
            return PopulationEvidenceTransactionResult(None, "ABSTAIN_" + built.reason.value.upper())
        if built.proposal.plan.actions != (built.proposal.plan.actions[0],) or built.proposal.plan.actions[0].value != "create":
            return PopulationEvidenceTransactionResult(None, "ABSTAIN_POLICY_ACTION")
        return PopulationEvidenceTransactionResult(built.proposal, "CREATE")

__all__ = ["PopulationEvidenceTransaction", "PopulationEvidenceTransactionResult"]
