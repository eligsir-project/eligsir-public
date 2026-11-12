"""Demand-driven evidence requirements (plan section 10.4, "Add EvidenceRequirements").

Necessary to avoid computing expensive diagnostic tensors when no active
policy needs them yet. As of this PR (W3-05) there is no consumer of
mapping evidence anywhere in the codebase — admission, replay, population
policy, and lifecycle decisions are all made exactly as before. Every flag
below therefore defaults to ``False``/off, and :meth:`EvidenceRequirements.any_needed`
lets :class:`eligsir.mapping.evidence.builder.EvidenceBuilder` short-circuit
to a near-zero-cost no-op (one attribute-bag boolean check, no tensor work,
no allocation) whenever nothing has opted in.

This module deliberately does *not* replace or alias the pre-existing
``EvidenceRequirements`` placeholder in
``eligsir/optimization/gaussian/contracts.py`` (see that module's
docstring: it is a stub scoped to ``OptimizationStepRequest`` with exactly
two flags, ``want_energy_grad_snapshot``/``want_loss_breakdown``, and was
explicitly documented there as "real definitions belong to later Wave 3
PRs"). This is that real definition, but it lives in its own module because
the evidence substrate is broader than one optimization step's request —
future PRs (W3-06+) are expected to have ``OptimizationStepRequest`` import
from here instead of carrying its own stub, but that migration is out of
scope for this PR (no call site changes to the optimization engine).
"""

from __future__ import annotations

from dataclasses import dataclass, fields


@dataclass(frozen=True)
class EvidenceRequirements:
    """Flags describing which evidence artifacts a caller currently wants.

    Grouped by which of the four evidence dataclasses (see ``contracts.py``)
    each flag gates production of. Field-level (not just dataclass-level)
    granularity is deliberate: some fields are cheap to read off values the
    render/optimization path already produces (e.g. ``alpha``,
    ``visibility``), while others require a reduction the builder would
    otherwise skip entirely (e.g. EMA'd gradients, which need a running
    reducer to be warm — see ``reducers.py``). A caller that only wants
    ``contribution`` should not pay for ``xyz_grad_ema``/``opacity_grad_ema``
    bookkeeping just because it asked for *some* Gaussian evidence.

    All flags default ``False`` — see module docstring for why that must
    hold for the entire lifetime of this PR.
    """

    # InputEvidence
    want_input_evidence: bool = False

    # RenderEvidence
    want_render_evidence: bool = False
    want_rgb_residual: bool = False
    want_depth_residual: bool = False
    want_alpha: bool = False
    want_rendered_depth: bool = False
    want_transmittance: bool = False

    # GaussianEvidence
    want_gaussian_evidence: bool = False
    want_visibility: bool = False
    want_projected_radius: bool = False
    want_contribution: bool = False
    want_xyz_grad_ema: bool = False
    want_opacity_grad_ema: bool = False
    want_support: bool = False
    want_geometry_confidence: bool = False
    want_free_space_conflict: bool = False
    want_age: bool = False

    # EvidenceSummary (cheap scalar reductions over the above; still gated
    # separately since even a scalar reduction over a large per-Gaussian
    # tensor is not free at N in the hundreds of thousands).
    want_summary: bool = False

    def any_needed(self) -> bool:
        """``True`` iff at least one flag is set.

        The builder's hot-path entry point calls this exactly once and
        returns immediately when it is ``False`` — the guaranteed
        near-zero-cost path this PR's deliverable 4 (overhead benchmark)
        measures against.
        """
        return any(getattr(self, f.name) for f in fields(self))

    @classmethod
    def none(cls) -> "EvidenceRequirements":
        """Explicit, self-documenting alias for "the current production
        default" — every flag off. Equivalent to ``EvidenceRequirements()``.
        """
        return cls()

    @classmethod
    def everything(cls) -> "EvidenceRequirements":
        """All flags on. Used by tests/benchmarks (deliverable 4) to measure
        the worst-case cost of turning evidence collection fully on; not
        used by any production call site in this PR.
        """
        return cls(**{f.name: True for f in fields(cls)})
