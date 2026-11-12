"""``GaussianBelief`` -- Phase 1 ("shadow GaussianBelief") of issue #1365's
canonical probabilistic evidence-model consolidation.

Canonical design docs (both landed on ``main`` 2026-08-31, superseding issue
#1372 which is being closed in favor of #1365 as the umbrella issue):
``docs/architecture/2026-08-31-probabilistic-gaussian-belief-and-population-control.md``
(schema: its §3, §6, §9.1) and
``docs/architecture/2026-08-31-pipeline-fast-path-throughput-and-gpu-feeding.md``
(pipeline/boundedness constraints: its §15-16). Field names below match §3
literally (``meaningful_exposure_count``, ``independent_support_views``,
``p_exist``, ``p_surface_consistent``, etc.) so telemetry and later #1365 work
packages can read them by name without a second translation layer.

    GaussianBelief { provenance, observability, existence, geometry, representation, temporal }
        -> GaussianPopulationPolicy (probabilistic action scores -- later work packages)
        -> GaussianMutationArbiter (already the sole writer -- unchanged)

**This module is Phase 1 only: "shadow GaussianBelief" per the canonical migration
plan's own Phase 1** ("Compute compact measurement/existence/geometry/representation
belief without actuation. No new ranking-only render. No full belief D2H."). Nothing
in the codebase reads this for any MCMC eligibility, lifecycle transition, depth-lock,
anchor, or growth/relocation DECISION yet, and nothing here changes training behavior.

Key probabilistic principle carried through this schema: existence is Beta-accumulated
evidence, not a boolean or a raw count -- ``p_exist = alpha / (alpha + beta)`` where
``alpha`` only grows from independent SUPPORTING observations and ``beta`` only grows
from actual CONTRADICTING ones (a valid depth ray landing in certified free space
where this Gaussian claims to be) -- **the absence of a measurement opportunity must
never itself decrement existence**, it only leaves uncertainty uninflated. This is why
``support_evidence``/``contradiction_evidence`` are two distinct accumulators, not one
scalar, and why ``compute_p_exist`` is a pure, standalone function nothing calls yet:
wiring real per-ray alpha/beta accumulation touches the render/depth-loss pipeline (a
genuine behavior-risk zone), out of this phase's additive-only scope.

**Geometry is a reference, not an embedding (canonical §5/§9.1 correction).** Each
Gaussian holds only ``surface_anchor_id`` -- an index into a separate, shared
``DepthSurfaceAnchor`` registry (many Gaussians on one coarse-to-fine surface patch
share one anchor record; xyz/normal/covariance/confidence live there, not duplicated
per-Gaussian). **The bounded ``DepthSurfaceAnchor`` registry itself is NOT built this
phase** -- it is real new subsystem work (coarse/fine anchor allocation, a bounded
budget, merge/demote policy for low-information duplicates, per canonical §5.3/§15.6)
that belongs to its own PR with its own paired verification, not to a shadow-belief
foundation phase. ``surface_anchor_id`` defaults to the sentinel ``-1`` ("unassigned")
on every Gaussian until that registry exists to assign into it.
``p_surface_consistent``/``surface_position_uncertainty``/``surface_normal_uncertainty``
are Gaussian-level (per-member conformance to whatever anchor it will eventually
reference), which is why they stay per-Gaussian even once anchors are shared.

**Bounded state contract (canonical §15.6 "Per-Gaussian belief": "O(N) compact device
tensors. No per-Gaussian Python objects.").** Every field below is a plain device
tensor, dtype chosen for a documented per-Gaussian byte budget -- see
``NEW_BUFFER_BYTES_PER_GAUSSIAN`` (currently 60 bytes/Gaussian across the 16 new
buffers; at a 5M ``cap_max`` that is ~300MB, alongside whatever the existing
per-Gaussian buffers already cost). No Python-level per-Gaussian object is ever
constructed anywhere in this module.

**No hot-path host synchronization (canonical §15.9).** Every function in this module
that runs on a training-loop-adjacent path (``init_belief_buffers`` through
``sync_belief_buffer_lengths``) is pure device-tensor indexing/concatenation/fill --
no ``.item()``, ``float(tensor)``, ``.cpu()``/``.numpy()``, or synchronizing
reduction anywhere in them. The one place this module does read a few scalars back to
host is ``utils/scene_geometry_metrics.py``'s ``gaussian_belief_summary_from_model``,
which runs only at report-write cadence (the existing ``lifecycle_stats_from_model``
sitting immediately next to it does the identical thing) -- this matches canonical
§15.1's "CPU telemetry plane: compact asynchronous summaries only" carve-out, not the
per-iteration hot path this constraint is aimed at.

Investigation finding (recorded per the task's request to document overlap rather
than force a parallel structure): **more than half of this schema's fields already
exist** as per-Gaussian tensor buffers on ``GsplatGaussianModel``
(``scene/gsplat_model.py``), just under different names, accumulated incrementally
across many prior issues (#283 lineage, #374 lineage-preservation, #908 lifecycle,
#1262 resolution-exposure). ``GaussianBelief.from_gaussians()`` is a thin, **zero-copy**
accessor (its tensor fields are the SAME tensor objects the model already owns) that
presents those buffers under this schema's naming. See ``FIELD_PROVENANCE`` below for
the full alias table.

**KNOWN-TEMPORARY PATTERN (per the #1365 Pass 3/4 audit,
``docs/architecture/2026-08-31-probabilistic-gaussian-control-pass3-pass4-audit.md``):**
the append/resize/checkpoint mechanism these buffers use -- allocate in
``_reset_running_state``, pad in ``append_running_state``/``_build_new_buffer_rows``,
slice in ``prune_points`` -- is the same POST-HOC lifecycle-buffer padding pattern this
codebase's existing ``birth_iter``/lineage buffers already use, and that pattern is
explicitly rejected as the future architecture: belief/lineage/anchor/exposure/optimizer
state are meant to eventually undergo the SAME ATOMIC ROW TRANSFORMATION as Gaussian
parameters during CREATE/RELOCATE/RETIRE, not be reconciled after the fact. This phase
still uses the post-hoc pattern deliberately -- it is additive/shadow-only, so there is
no correctness reason yet to hold it to the atomic-row-transform standard, and building
that transaction mechanism is real, separate, later-phase work. Do not read
``growth_pad_belief_buffers``/``relocate_belief_buffers``/etc. as the target design; they
exist to get the data model in place now and are expected to be superseded once the
atomic mutation-transaction mechanism lands.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

# --------------------------------------------------------------------------
# New buffers this module owns (16 of this schema's ~26 fields have no existing
# counterpart anywhere in the codebase -- see FIELD_PROVENANCE for the rest).
# --------------------------------------------------------------------------
#
# Each spec is (name, dtype, fill_value, vector_dim). ``vector_dim=None`` means a
# scalar-per-Gaussian buffer (shape ``[N]``); an int means ``[N, vector_dim]``.
#
# Canonical §15.6 requires a documented per-Gaussian byte budget for new state.
# Per-row byte costs, dtype chosen deliberately (float16/int32 unless a wider
# type is required, matching the canonical doc's own "compact pseudo-counts,
# log-odds, EMAs, uncertainty scalars" first-implementation guidance):
NEW_BUFFER_BYTES_PER_GAUSSIAN: dict = {
    "birth_anchor_xyz": 12,             # float32 x3
    "baseline_diversity": 2,            # float16
    "last_exposed_update": 4,           # int32
    "support_evidence": 2,              # float16
    "contradiction_evidence": 2,        # float16
    "p_exist": 2,                       # float16
    "surface_anchor_id": 8,             # int64 (registry index, not yet built)
    "surface_position_uncertainty": 2,  # float16
    "surface_normal_uncertainty": 2,    # float16
    "rgb_residual_ema": 4,              # float32
    "depth_residual_ema": 4,            # float32
    "redundancy": 4,                    # float32
    "expected_marginal_gain": 4,        # float32
    "change_probability": 2,            # float16
    "evidence_recency": 4,              # int32
    "confidence_decay": 2,              # float16
}
#: Total new bytes/Gaussian (60 as of this phase) -- assert-tested against the
#: spec table below so this docstring/dict cannot silently drift from reality.
TOTAL_NEW_BYTES_PER_GAUSSIAN: int = sum(NEW_BUFFER_BYTES_PER_GAUSSIAN.values())

#: ``relocation_policy`` (``RELOCATION_POLICY`` below, keyed by name) says what
#: ``relocate_belief_buffers`` does to a revived (MCMC-reseeded) slot:
#:   "donor" -- inherit the sampled donor's transferable geometry/representation
#:              prior. Location-specific measurement/evidence fields are reset.
#:   "reset" -- reset to the buffer's default. Matches how this codebase resets
#:              exposure/consensus counters on relocation (``reset_exposure_rows``/
#:              ``reset_consensus_counts``): these are LOCATION-specific measurement
#:              histories, and a slot that just moved to a new position has
#:              genuinely observed nothing there yet. Existence belief in particular
#:              must reset to "not yet contradicted" (``p_exist=1.0``, zero
#:              alpha/beta) rather than inherit the donor's evidence about the
#:              donor's own history.
NEW_BUFFER_SPECS: tuple = (
    # -- provenance: immutable at birth, including a relocation ("rebirth") event. --
    ("birth_anchor_xyz", torch.float32, 0.0, 3),
    # -- observability: baseline diversity remains deferred; a completed
    # canonical B4 transaction updates last_exposed_update on-device. --
    ("baseline_diversity", torch.float16, 0.0, None),
    ("last_exposed_update", torch.int32, -1, None),
    # -- existence: Beta-style evidence accumulators + derived probability. All
    #    structure only -- see the module docstring for why real accumulation is
    #    deferred. p_exist defaults to 1.0 ("not yet contradicted"), never 0.0 --
    #    a freshly born Gaussian must not start believed nonexistent.
    ("support_evidence", torch.float16, 0.0, None),
    ("contradiction_evidence", torch.float16, 0.0, None),
    ("p_exist", torch.float16, 1.0, None),
    # -- geometry: surface_anchor_id references a (not-yet-built) DepthSurfaceAnchor
    #    registry -- see the module docstring's "Geometry is a reference" section.
    #    The other two are Gaussian-level conformance/uncertainty relative to
    #    whatever anchor this Gaussian will eventually reference.
    ("surface_anchor_id", torch.int64, -1, None),
    ("surface_position_uncertainty", torch.float16, 0.0, None),
    ("surface_normal_uncertainty", torch.float16, 0.0, None),
    # -- representation: residual/redundancy remain deferred.  A completed
    # canonical B4 transaction updates expected marginal gain from its normal
    # backward gradient and renderer contribution. --
    ("rgb_residual_ema", torch.float32, 0.0, None),
    ("depth_residual_ema", torch.float32, 0.0, None),
    ("redundancy", torch.float32, 0.0, None),
    ("expected_marginal_gain", torch.float32, 0.0, None),
    # -- temporal: structure only, not yet computed this phase. --
    ("change_probability", torch.float16, 0.0, None),
    ("evidence_recency", torch.int32, -1, None),
    ("confidence_decay", torch.float16, 0.0, None),
)

assert set(spec[0] for spec in NEW_BUFFER_SPECS) == set(NEW_BUFFER_BYTES_PER_GAUSSIAN), (
    "NEW_BUFFER_SPECS and NEW_BUFFER_BYTES_PER_GAUSSIAN have drifted apart -- "
    "every new buffer needs a documented byte cost (canonical §15.6)."
)

NEW_BUFFER_NAMES: tuple = tuple(spec[0] for spec in NEW_BUFFER_SPECS)

RELOCATION_POLICY: dict = {
    "birth_anchor_xyz": "donor",
    "baseline_diversity": "reset",
    "last_exposed_update": "reset",
    "support_evidence": "reset",
    "contradiction_evidence": "reset",
    "p_exist": "reset",
    "surface_anchor_id": "reset",
    "surface_position_uncertainty": "donor",
    "surface_normal_uncertainty": "donor",
    "rgb_residual_ema": "reset",
    "depth_residual_ema": "reset",
    # Redundancy is evidence about the donor's old hypothesis/location, not a
    # transferable representation prior. A relocated slot starts as a fresh
    # hypothesis at its new target and must not retain that evidence.
    "redundancy": "reset",
    "expected_marginal_gain": "reset",
    "change_probability": "reset",
    "evidence_recency": "reset",
    "confidence_decay": "reset",
}

#: Names that are "structure only, not yet computed this phase" -- initialized and
#: correctly maintained across growth/relocation/checkpoint, but no production code
#: anywhere writes a real (non-default) value into them yet. Surfaced explicitly so
#: a reader of ``GaussianBelief`` output cannot mistake a placeholder default for a
#: real measurement.
STRUCTURE_ONLY_FIELDS: frozenset = frozenset({
    "baseline_diversity", "support_evidence",
    "contradiction_evidence", "p_exist", "surface_anchor_id",
    "surface_position_uncertainty", "surface_normal_uncertainty",
    "rgb_residual_ema", "depth_residual_ema", "redundancy",
    "change_probability", "evidence_recency",
    "confidence_decay",
})


def compute_p_exist(alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    """``p_exist = alpha / (alpha + beta)``, the canonical Beta-evidence existence
    probability (§4.1). Pure, device-native, standalone -- **not called from any
    production path this phase** (see the module docstring); provided so a later
    work package wiring real per-ray alpha/beta accumulation has a single, tested
    place to derive ``p_exist`` from, rather than reimplementing the formula.

    Returns ``1.0`` (not NaN) where ``alpha + beta == 0`` -- "no evidence yet" must
    read as "not contradicted", matching this schema's ``p_exist`` default. No
    host synchronization: the ``torch.where`` runs entirely on ``alpha``'s device.
    """
    total = alpha + beta
    safe_total = total.clamp_min(torch.finfo(torch.float32).tiny if total.dtype.is_floating_point else 1)
    ratio = alpha / safe_total
    return torch.where(total > 0, ratio, torch.ones_like(ratio))


def _empty(dtype, vector_dim, device) -> torch.Tensor:
    shape = (0, vector_dim) if vector_dim else (0,)
    return torch.empty(shape, dtype=dtype, device=device)


def init_belief_buffers(gaussians, device="cuda") -> None:
    """Allocate the new buffers at zero length (mirrors ``__init__``'s own
    per-buffer ``torch.empty(0, ...)`` allocations for every other streaming
    buffer, so the class invariant "these buffers always exist" holds from
    construction)."""
    for name, dtype, _fill, vector_dim in NEW_BUFFER_SPECS:
        setattr(gaussians, name, _empty(dtype, vector_dim, device))


def reset_belief_buffers(gaussians, count: int, device="cuda") -> None:
    """(Re)allocate all new buffers sized ``count`` (mirrors ``_reset_running_state``).

    ``birth_anchor_xyz`` is special-cased to the current position (matching how
    ``anchor_xyz``/``anchor_scale_log``/``anchor_opacity_logit`` are seeded at reset:
    "anchor = current value at birth"), never left at its generic zero fill.
    """
    for name, dtype, fill, vector_dim in NEW_BUFFER_SPECS:
        shape = (count, vector_dim) if vector_dim else (count,)
        setattr(gaussians, name, torch.full(shape, fill, dtype=dtype, device=device))
    if count > 0:
        xyz = getattr(gaussians, "get_xyz", None)
        if xyz is not None and int(xyz.shape[0]) == count:
            gaussians.birth_anchor_xyz = xyz.detach().clone()


def ensure_belief_buffers(gaussians) -> None:
    """Repair/allocate the new buffers to the model's current Gaussian count.

    Mirrors ``_ensure_geometry_streaming_buffers``'s defensive
    allocate-or-reset-if-mismatched pattern; called from ``GsplatGaussianModel``'s
    checkpoint-restore path so an old checkpoint (predating this phase) or a
    size-mismatched restore never leaves these buffers desynced from ``get_xyz``.
    """
    n = int(gaussians.get_xyz.shape[0])
    device = gaussians.get_xyz.device
    for name, dtype, fill, vector_dim in NEW_BUFFER_SPECS:
        buf = getattr(gaussians, name, None)
        if buf is not None and buf.shape[0] == n:
            continue
        shape = (n, vector_dim) if vector_dim else (n,)
        default = torch.full(shape, fill, dtype=dtype, device=device)
        if name == "birth_anchor_xyz" and n > 0:
            default = gaussians.get_xyz.detach().clone()
        setattr(gaussians, name, default)


def new_row_belief_buffers(N: int, new_means: torch.Tensor) -> dict:
    """Build the new-row tensors for an ``N``-Gaussian depth-insertion append
    (mirrors ``_build_new_buffer_rows``'s per-buffer construction). ``birth_anchor_xyz``
    is the newly-inserted Gaussians' own position, matching "anchor = position at birth".
    """
    device = new_means.device
    rows = {}
    for name, dtype, fill, vector_dim in NEW_BUFFER_SPECS:
        if name == "birth_anchor_xyz":
            rows[name] = new_means.detach().clone()
            continue
        shape = (N, vector_dim) if vector_dim else (N,)
        rows[name] = torch.full(shape, fill, dtype=dtype, device=device)
    return rows


def growth_pad_belief_buffers(gaussians, old_count: int, added_count: int, parent_idxs=None) -> None:
    """Extend all new buffers by ``added_count`` rows for an MCMC growth clone event
    (mirrors ``append_running_state``'s per-buffer-group padding loops).

    Only touches a buffer whose current length already equals ``old_count`` (the
    same defensive guard every other buffer in ``append_running_state`` uses), so a
    buffer that is missing or already desynced is left alone rather than corrupted
    further. ``birth_anchor_xyz`` inherits the sampled parent's anchor (a clone is a
    literal rebirth at the parent's position) when ``parent_idxs`` is available,
    matching the donor-inherit convention every other geometry-prior buffer in this
    same function already uses; otherwise it falls back to the clone's own current
    position (the same fallback ``anchor_xyz``'s own padding site uses).
    """
    for name, dtype, fill, vector_dim in NEW_BUFFER_SPECS:
        tensor = getattr(gaussians, name, None)
        if tensor is None or tensor.shape[0] != old_count:
            continue
        if name == "birth_anchor_xyz":
            if parent_idxs is not None and parent_idxs.shape[0] == added_count:
                pad = tensor[parent_idxs].clone()
            else:
                pad = gaussians.get_xyz[-added_count:].detach().clone()
        else:
            shape = (added_count, vector_dim) if vector_dim else (added_count,)
            pad = torch.full(shape, fill, dtype=dtype, device=tensor.device)
        setattr(gaussians, name, torch.cat([tensor, pad], dim=0))


def prune_belief_buffers(gaussians, mask: torch.Tensor, valid_points_mask: torch.Tensor) -> None:
    """Slice all new buffers by a prune mask (mirrors ``prune_points``'s per-buffer
    ``if tensor.shape[0] == mask.shape[0]: tensor = tensor[valid_points_mask]`` guard)."""
    for name, _dtype, _fill, _vector_dim in NEW_BUFFER_SPECS:
        tensor = getattr(gaussians, name, None)
        if tensor is not None and tensor.shape[0] == mask.shape[0]:
            setattr(gaussians, name, tensor[valid_points_mask])


def relocate_belief_buffers(gaussians, dead_indices: torch.Tensor, sampled_idxs: torch.Tensor) -> None:
    """Apply each buffer's relocation policy to a revived (MCMC-reseeded) slot
    (mirrors ``reset_relocated_state``'s "Finding D" donor-inherit loop and the
    exposure/consensus reset-on-reseed precedent).
    """
    n = int(gaussians.get_xyz.shape[0])
    for name, dtype, fill, vector_dim in NEW_BUFFER_SPECS:
        tensor = getattr(gaussians, name, None)
        if tensor is None or tensor.shape[0] != n:
            continue
        policy = RELOCATION_POLICY[name]
        if policy == "donor":
            tensor[dead_indices] = tensor[sampled_idxs]
        else:  # "reset": fresh belief for a slot that just moved to a new position
            shape = (dead_indices.shape[0], vector_dim) if vector_dim else (dead_indices.shape[0],)
            tensor[dead_indices] = torch.full(shape, fill, dtype=dtype, device=tensor.device)

    # ``utility_ema`` predates GaussianBelief but is the legacy alias for
    # gradient/utility evidence. ``reset_relocated_state`` copies it from the
    # donor before calling this helper, so clear that stale evidence here too.
    utility_ema = getattr(gaussians, "utility_ema", None)
    if utility_ema is not None and utility_ema.shape[0] == n:
        utility_ema[dead_indices] = torch.zeros(
            dead_indices.shape[0], dtype=utility_ema.dtype, device=utility_ema.device
        )


def sync_belief_buffer_lengths(gaussians) -> None:
    """Repair all new buffers to the model's current Gaussian count with simple
    truncate/pad-with-default semantics (mirrors
    ``utils/streaming_state_sync.py``'s ``_sync_streaming_state_lengths``, the
    third growth-sync site alongside ``append_running_state``/``prune_points``).
    """
    count = int(gaussians.get_xyz.shape[0])
    for name, dtype, fill, vector_dim in NEW_BUFFER_SPECS:
        tensor = getattr(gaussians, name, None)
        if tensor is None:
            continue
        if tensor.shape[0] == count:
            continue
        if tensor.shape[0] > count:
            setattr(gaussians, name, tensor[:count].contiguous())
            continue
        deficit = count - tensor.shape[0]
        if name == "birth_anchor_xyz":
            pad = gaussians.get_xyz[-deficit:].detach().clone()
        else:
            shape = (deficit, vector_dim) if vector_dim else (deficit,)
            pad = torch.full(shape, fill, dtype=dtype, device=tensor.device)
        setattr(gaussians, name, torch.cat([tensor, pad], dim=0))


# --------------------------------------------------------------------------
# GaussianBelief: the unified read accessor over this schema's six field groups.
# --------------------------------------------------------------------------


@dataclass
class ProvenanceBelief:
    """Immutable per-Gaussian identity, set once at birth (including a relocation
    "rebirth" event -- see ``RELOCATION_POLICY``)."""

    gaussian_uid: torch.Tensor
    birth_update: torch.Tensor
    birth_method: torch.Tensor
    parent_uid: torch.Tensor
    root_uid: torch.Tensor
    generation: torch.Tensor
    birth_anchor_xyz: torch.Tensor


@dataclass
class ObservabilityBelief:
    """How much and how diverse the measurement of this Gaussian has been --
    replaces age-based young-protection with real observation counts (canonical
    §4.2). Optional fields are ``None`` when the owning flag is off, mirroring how
    ``utils/resolution_exposure.py`` already behaves (flag-gated allocation, not
    always-on)."""

    meaningful_exposure_count: Optional[torch.Tensor]
    independent_support_views: torch.Tensor
    baseline_diversity: torch.Tensor
    highest_supervised_tier: Optional[torch.Tensor]
    last_exposed_update: torch.Tensor
    last_supported_update: Optional[torch.Tensor]


@dataclass
class ExistenceBelief:
    """Beta-style existence evidence (canonical §4.1). ``support_evidence``/
    ``contradiction_evidence`` are the alpha/beta accumulators (see the module
    docstring for why an absent measurement must inflate uncertainty, never
    decrement existence); ``p_exist`` is the derived probability
    (``compute_p_exist``, not yet wired to real accumulation this phase)."""

    support_evidence: torch.Tensor
    contradiction_evidence: torch.Tensor
    p_exist: torch.Tensor


@dataclass
class GeometryBelief:
    """Surface belief as a REFERENCE, not an embedding (canonical §5/§9.1):
    ``surface_anchor_id`` indexes a shared, bounded ``DepthSurfaceAnchor`` record
    (not built this phase -- sentinel ``-1`` means unassigned) that owns the actual
    xyz/normal/covariance/confidence. The two uncertainty fields and
    ``p_surface_consistent`` are per-Gaussian conformance to whatever anchor this
    Gaussian references, not a duplicate of the anchor's own state.
    ``p_surface_consistent`` reuses the existing ``anchor_confidence`` buffer
    verbatim -- that buffer already carries the same surface-confidence semantics
    this schema names probabilistically."""

    surface_anchor_id: torch.Tensor
    p_surface_consistent: torch.Tensor
    surface_position_uncertainty: torch.Tensor
    surface_normal_uncertainty: torch.Tensor


@dataclass
class RepresentationBelief:
    """Rendering-quality / redundancy / marginal-value evidence for the
    population policy's action scores (later work packages). ``render_contribution``
    reuses the existing ``visibility_ema`` buffer; ``gradient_utility_evidence``
    reuses the existing ``utility_ema`` buffer -- both already carry these exact
    semantics under pre-canonical names."""

    render_contribution: torch.Tensor
    rgb_residual_ema: torch.Tensor
    depth_residual_ema: torch.Tensor
    gradient_utility_evidence: torch.Tensor
    redundancy: torch.Tensor
    expected_marginal_gain: torch.Tensor


@dataclass
class TemporalBelief:
    """Evidence recency / staleness (canonical §3 temporal group). Distinct from
    ``ObservabilityBelief.last_supported_update`` (a raw last-support timestamp):
    these are derived/decaying quantities about how stale that timestamp has
    become, kept as their own new placeholders rather than reusing the same
    buffer under two schema fields."""

    evidence_recency: torch.Tensor
    change_probability: torch.Tensor
    confidence_decay: torch.Tensor


@dataclass
class GaussianBelief:
    """The unified per-Gaussian probabilistic belief model (issue #1365, canonical
    schema), Phase 1 ("shadow GaussianBelief").

    Every tensor field is a **reference** to the same tensor object
    ``GsplatGaussianModel`` already owns -- this is a read accessor, not a second
    copy of per-Gaussian state, and building one is O(1) (no tensor is copied or
    indexed, no host synchronization). It exists so later work packages and any
    new telemetry can be written against one coherent naming scheme instead of the
    model's grab-bag of historically-named buffers.
    """

    provenance: ProvenanceBelief
    observability: ObservabilityBelief
    existence: ExistenceBelief
    geometry: GeometryBelief
    representation: RepresentationBelief
    temporal: TemporalBelief
    n: int

    @classmethod
    def from_gaussians(cls, gaussians) -> "GaussianBelief":
        ensure_belief_buffers(gaussians)
        n = int(gaussians.get_xyz.shape[0])

        def _opt(name):
            buf = getattr(gaussians, name, None)
            return buf if (buf is not None and buf.shape[0] == n) else None

        provenance = ProvenanceBelief(
            gaussian_uid=gaussians.gaussian_uid,
            birth_update=gaussians.birth_iter,
            birth_method=gaussians.birth_method,
            parent_uid=gaussians.parent_uid,
            root_uid=gaussians.root_uid,
            generation=gaussians.generation,
            birth_anchor_xyz=gaussians.birth_anchor_xyz,
        )
        observability = ObservabilityBelief(
            meaningful_exposure_count=_opt("exposure_count"),
            independent_support_views=gaussians.support_count,
            baseline_diversity=gaussians.baseline_diversity,
            highest_supervised_tier=_opt("supervised_tier_high_water"),
            last_exposed_update=gaussians.last_exposed_update,
            last_supported_update=_opt("last_supervised_iter"),
        )
        existence = ExistenceBelief(
            support_evidence=gaussians.support_evidence,
            contradiction_evidence=gaussians.contradiction_evidence,
            p_exist=gaussians.p_exist,
        )
        geometry = GeometryBelief(
            surface_anchor_id=gaussians.surface_anchor_id,
            p_surface_consistent=gaussians.anchor_confidence,
            surface_position_uncertainty=gaussians.surface_position_uncertainty,
            surface_normal_uncertainty=gaussians.surface_normal_uncertainty,
        )
        representation = RepresentationBelief(
            render_contribution=gaussians.visibility_ema,
            rgb_residual_ema=gaussians.rgb_residual_ema,
            depth_residual_ema=gaussians.depth_residual_ema,
            gradient_utility_evidence=gaussians.utility_ema,
            redundancy=gaussians.redundancy,
            expected_marginal_gain=gaussians.expected_marginal_gain,
        )
        temporal = TemporalBelief(
            evidence_recency=gaussians.evidence_recency,
            change_probability=gaussians.change_probability,
            confidence_decay=gaussians.confidence_decay,
        )
        return cls(
            provenance=provenance, observability=observability, existence=existence,
            geometry=geometry, representation=representation, temporal=temporal, n=n,
        )


#: Documents, for every field in this schema, whether ``GaussianBelief`` reuses an
#: EXISTING model buffer (and which one) or introduces a NEW buffer this phase.
#: Kept as data (not just docstring prose) so this mapping is queryable/testable.
#:
#: ``status`` is one of:
#:   "alias"               -- reuses an existing buffer verbatim under this schema's name.
#:   "new_wired"            -- a new buffer this phase, and it IS populated with real
#:                             signal (currently only ``birth_anchor_xyz``, set at every
#:                             birth/rebirth site).
#:   "new_structure_only"   -- a new buffer this phase, correctly threaded through
#:                             growth/relocation/checkpoint, but left at its documented
#:                             default -- no production code computes a real value into
#:                             it yet (future work packages' job; see the module
#:                             docstring for why real Beta-evidence/residual
#:                             accumulation and the DepthSurfaceAnchor registry are
#:                             deferred).
FIELD_PROVENANCE: dict = {
    # provenance
    "gaussian_uid": {"status": "alias", "existing_buffer": "gaussian_uid"},
    "birth_update": {"status": "alias", "existing_buffer": "birth_iter"},
    "birth_method": {"status": "alias", "existing_buffer": "birth_method"},
    "parent_uid": {"status": "alias", "existing_buffer": "parent_uid"},
    "root_uid": {"status": "alias", "existing_buffer": "root_uid"},
    "generation": {"status": "alias", "existing_buffer": "generation"},
    "birth_anchor_xyz": {"status": "new_wired"},
    # observability
    "meaningful_exposure_count": {
        "status": "alias", "existing_buffer": "exposure_count",
        "note": "flag-gated (issue #1262); None when --streaming_resolution_exposure_veto/_telemetry is off",
    },
    "independent_support_views": {"status": "alias", "existing_buffer": "support_count"},
    "baseline_diversity": {"status": "new_structure_only"},
    "highest_supervised_tier": {
        "status": "alias", "existing_buffer": "supervised_tier_high_water",
        "note": "flag-gated (issue #1262)",
    },
    "last_exposed_update": {
        "status": "new_wired",
        "note": "updated only from a completed, current canonical B4 E0 transaction",
    },
    "last_supported_update": {
        "status": "alias", "existing_buffer": "last_supervised_iter",
        "note": "flag-gated (--streaming_recency_reg_horizon > 0)",
    },
    # existence
    "support_evidence": {"status": "new_structure_only", "note": "Beta alpha accumulator"},
    "contradiction_evidence": {"status": "new_structure_only", "note": "Beta beta accumulator"},
    "p_exist": {
        "status": "new_structure_only",
        "note": "compute_p_exist(alpha, beta) exists and is unit-tested but not yet "
                "called from any production path -- real accumulation needs render/"
                "depth-loss pipeline wiring, out of this additive-only phase's scope",
    },
    # geometry
    "surface_anchor_id": {
        "status": "new_structure_only",
        "note": "references a DepthSurfaceAnchor registry not built this phase "
                "(canonical §5/§9.1); sentinel -1 = unassigned on every Gaussian",
    },
    "p_surface_consistent": {
        "status": "alias", "existing_buffer": "anchor_confidence",
        "note": "existing buffer already carries surface-confidence semantics under a pre-canonical name",
    },
    "surface_position_uncertainty": {"status": "new_structure_only"},
    "surface_normal_uncertainty": {"status": "new_structure_only"},
    # representation
    "render_contribution": {"status": "alias", "existing_buffer": "visibility_ema"},
    "rgb_residual_ema": {"status": "new_structure_only"},
    "depth_residual_ema": {"status": "new_structure_only"},
    "gradient_utility_evidence": {"status": "alias", "existing_buffer": "utility_ema"},
    "redundancy": {"status": "new_structure_only"},
    "expected_marginal_gain": {
        "status": "new_wired",
        "note": "bounded device-side B4 renderer contribution × backward-gradient utility",
    },
    # temporal
    "evidence_recency": {"status": "new_structure_only"},
    "change_probability": {"status": "new_structure_only"},
    "confidence_decay": {"status": "new_structure_only"},
}


__all__ = [
    "NEW_BUFFER_SPECS",
    "NEW_BUFFER_NAMES",
    "NEW_BUFFER_BYTES_PER_GAUSSIAN",
    "TOTAL_NEW_BYTES_PER_GAUSSIAN",
    "RELOCATION_POLICY",
    "STRUCTURE_ONLY_FIELDS",
    "FIELD_PROVENANCE",
    "compute_p_exist",
    "init_belief_buffers",
    "reset_belief_buffers",
    "ensure_belief_buffers",
    "new_row_belief_buffers",
    "growth_pad_belief_buffers",
    "prune_belief_buffers",
    "relocate_belief_buffers",
    "sync_belief_buffer_lengths",
    "ProvenanceBelief",
    "ObservabilityBelief",
    "ExistenceBelief",
    "GeometryBelief",
    "RepresentationBelief",
    "TemporalBelief",
    "GaussianBelief",
]
