"""Mapping evidence substrate (Wave 3, PR W3-05).

See ``plans/2026-08-06-wave3-live-mapping-runtime-evidence-strategy-implementation-plan.md``
section 10 for the full design rationale. This package is **observation-only**:
it defines a shared evidence contract for admission, replay, population
policy, diagnostics, and future learning, and a demand-driven builder that
reads values the mapper already computes. Nothing in this package changes
any mapping decision — no consumer PR exists yet, so
:class:`eligsir.mapping.evidence.requirements.EvidenceRequirements`
defaults every flag to "not needed" and building evidence with an
all-`False` requirements object is a near-zero-cost no-op.
"""

from eligsir.mapping.evidence.contracts import (
    EvidenceMeta,
    EvidenceSummary,
    GaussianEvidence,
    InputEvidence,
    RenderEvidence,
)
from eligsir.mapping.evidence.requirements import EvidenceRequirements

__all__ = [
    "EvidenceMeta",
    "InputEvidence",
    "RenderEvidence",
    "GaussianEvidence",
    "EvidenceSummary",
    "EvidenceRequirements",
]
