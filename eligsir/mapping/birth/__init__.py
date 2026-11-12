"""Unified Gaussian birth-owner scaffolding (issue #1365).

See ``docs/architecture/2026-08-31-unified-gaussian-birth-owner-and-adaptive-depth-evidence.md``.
This package currently contains only the evidence-only data contract
(:mod:`eligsir.mapping.birth.contracts`) -- no production insertion or
MCMC growth path constructs or consumes these types yet. Phase 1 of the
design doc's implementation sequence (letting the existing depth-insertion
planner emit targets through a compatibility adapter) and Phase 2 (routing
both source-depth and MCMC growth through one owner) are future work.
"""

from eligsir.mapping.birth.contracts import (
    BirthAction,
    GaussianBirthDecision,
    GaussianBirthProposal,
    GaussianBirthTarget,
    GeometryConfidence,
    ProposalKind,
)

__all__ = [
    "BirthAction",
    "GaussianBirthDecision",
    "GaussianBirthProposal",
    "GaussianBirthTarget",
    "GeometryConfidence",
    "ProposalKind",
]
