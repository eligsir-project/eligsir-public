"""Contracts and result types for the Gaussian optimization-step engine.

Wave 3, PR W3-02 (see
``plans/2026-08-06-wave3-live-mapping-runtime-evidence-strategy-implementation-plan.md``
section 7), Subagent A scope: **types only** — no logic is moved here. The
purpose of this module is to give ``GaussianOptimizationEngine.run_step()``
(built by later subagents in this same PR) a real, typed request/result pair
(section 7.3 of the plan, reproduced verbatim below) and to resolve the one
open design question the prior author (see ``engine.py``'s own docstring,
lines 73-121) explicitly punted on: **what does the post-backward gradient-
masking cascade actually take as input?**

## The ``GradientPolicyContext`` design decision

The prior author's blocker was real: the post-backward masking cascade in
``train_streaming.py`` (~lines 4532-4700 as of this PR) reads and writes a
few dozen loop-local names, and there was no contract between "one huge flat
parameter list" (itself a design smell — a function that takes 30 unrelated
scalars is really several functions wearing a trenchcoat) and "pass the
entire loop namespace" (not a real interface at all — it makes every future
caller implicitly coupled to ``streaming_training()``'s internals).

The resolution here: **read what the cascade actually touches, group it by
which sub-concept it belongs to, and make each sub-concept a small typed
sub-dataclass.** The cascade turns out to decompose cleanly into five
independent gradient policies, each with a narrow, self-contained state
requirement:

1. **Active-set masking** (sparse selective-Adam invisible-row zeroing) —
   needs only the current visibility/provisional union mask.
2. **Stable-geometry freeze** (H7 old-geometry freeze + lifecycle-frozen +
   spatial-bubble out-of-bounds freeze) — needs birth-frame age state,
   lifecycle state, and bubble geometry (center/radius derived from
   ``gaussians.get_xyz`` + ``args`` at call time, not passed as raw
   scalars — the bubble mask function already exists at
   ``utils.strategies.mcmc_strategy._bubble_active_mask`` and takes
   ``(xyz, args)``, so ``BubbleState`` only needs to carry whether bubble
   masking is enabled plus a slot for the pre-step freeze snapshot, not a
   duplicated center/radius representation).
3. **Bootstrap-anchor masking** (birth_frame==0 mean-grad suppression,
   linear ramp) — needs the anchor weight/decay-steps schedule.
4. **Insertion-anchor masking** (birth_frame>0 mean-grad suppression,
   per-Gaussian age ramp keyed off ``anchor_iter``) — needs its own
   weight/decay-steps schedule, structurally identical in shape to (3) but
   keyed on a different birth-frame condition and a different clock
   (``anchor_iter`` vs. iteration 0), so it is kept as a distinct
   sub-dataclass rather than collapsed into one — collapsing them would
   force a policy to branch on "which anchor am I" internally, which is
   exactly the kind of hidden coupling this contract exists to avoid.
5. **Depth-lock masking** (``depth_locked_mask``/``zero_depth_locked_mean_grads``
   from ``utils.strategies.mcmc_strategy``) — needs only whether depth-lock
   is active for this run; the mask itself is recomputed from
   ``gaussians``/``args`` inside the policy, matching how the inline code
   already does it (``depth_locked_mask(gaussians, args)`` takes no other
   state).

Two names — ``gaussians`` and ``args`` — are read by every single policy
(and by the cascade as a whole). Rather than duplicate them into every
sub-dataclass, they live once on the top-level ``GradientPolicyContext``
alongside ``iteration``/``n_frames_ingested`` (the two clocks every policy's
age arithmetic is keyed against). Everything else is grouped into the five
sub-states above. This keeps ``GradientPolicyContext`` a single object with
a bounded, enumerable field list (not 30 flat scalars, not the loop
namespace) that every ``GradientPolicy.apply(context)`` receives — exactly
the shape the plan's section 7.4 sketches (``GradientPolicy.apply(context)
-> GradientPolicyResult``).

Concretely, a policy that only cares about, say, bootstrap anchoring reads
``context.gaussians``, ``context.iteration``, and
``context.anchor.bootstrap`` — and ignores the rest. Nothing forces every
policy to consume every field; the point of grouping is discoverability and
testability (each sub-dataclass can be constructed and asserted on in
isolation — see ``tests/test_contracts.py``), not that every policy must use
every group.

## Everything else in this module

``OptimizationStepRequest``/``OptimizationStepResult`` are reproduced
verbatim from plan section 7.3. The types they reference
(``OptimizationView``, ``EvidenceRequirements``, ``LossBreakdown``,
``RenderOutputs``, ``OptimizationEvidence``) do not exist anywhere in the
codebase yet (confirmed via a full-repo grep before writing this module) —
real definitions belong to later Wave 3 PRs (evidence/render-port work,
W3-05/W3-06 per the plan's own numbering), so they are stubbed here as
minimal placeholder dataclasses purely so this module imports and the
request/result types are constructible in tests. Do not treat these stubs
as the real design — they intentionally carry no fields beyond what makes
them constructible.

``RandomStateToken`` has no existing counterpart in ``utils/`` either
(grepped: no RNG-state-capturing utility exists in this repo today), so it
is a minimal stub wrapping raw ``torch.get_rng_state()``-shaped bytes.
"""

from __future__ import annotations

import abc
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator, Protocol, runtime_checkable

from eligsir.optimization.gaussian.supervision_evidence import (
    SupervisionEvidenceBatch,
    SupervisionModality,
)
from eligsir.contracts.control import ModalityEligibility
from eligsir.contracts.runtime import SourceIdentity


# ---------------------------------------------------------------------------
# Placeholder types referenced by the plan's section 7.3 target API but not
# yet defined anywhere in the codebase. Real definitions are out of scope
# for this PR (W3-02 Subagent A) — see module docstring.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OptimizationView:
    """The render inputs that vary per fixed-topology optimization step."""

    camera: Any = None
    background: Any = None
    update_sh_rest: bool = True
    render_depth: bool = False
    # Per-view objective/evidence intent.  ``None`` preserves the historical
    # coupling to ``render_depth`` for callers not yet modality-aware.
    supervision_modality: SupervisionModality | None = None
    modality_eligibility: ModalityEligibility | None = None
    resolution_bucket: str = "default"
    # Transitional compatibility for callers that already hold an opaque
    # renderer view. New callers use the typed fields above.
    payload: Any = None
    # Stable streaming source-frame identity. This is intentionally separate
    # from the opaque camera object because a B>1 batch can contain duplicate
    # camera draws and a non-finite middle view must not be inferred from
    # aggregate work counters after the fact.
    source_index: int | None = None
    # The full immutable physical source identity is required for short-lived
    # geometry render reuse.  ``source_index`` remains a legacy accounting
    # coordinate and is deliberately insufficient for reuse fencing.
    source_identity: SourceIdentity | None = None


def resolve_supervision_modality(view: OptimizationView) -> SupervisionModality:
    """Intersect optional per-view intent with its admission-owned eligibility."""
    eligibility = view.modality_eligibility
    modality = view.supervision_modality
    if modality is None:
        depth = bool(view.render_depth)
        modality = SupervisionModality(
            depth_loss_enabled=depth,
            depth_evidence_enabled=depth,
            geometry_evidence_enabled=bool(
                eligibility and eligibility.geometry_population_target
            ),
        )
    if eligibility is None:
        return modality
    return SupervisionModality(
        rgb_loss_enabled=modality.rgb_loss_enabled and eligibility.rgb_supervision,
        depth_loss_enabled=modality.depth_loss_enabled and eligibility.depth_supervision,
        rgb_evidence_enabled=modality.rgb_evidence_enabled and eligibility.rgb_supervision,
        depth_evidence_enabled=modality.depth_evidence_enabled and eligibility.depth_supervision,
        geometry_evidence_enabled=(
            modality.geometry_evidence_enabled
            and eligibility.geometry_population_target
        ),
    )


@dataclass(frozen=True)
class EvidenceRequirements:
    """Placeholder. TODO(W3-05/W3-06): real definition.

    Stands in for a typed description of which evidence artifacts a given
    optimization step is asked to produce (e.g. energy-gradient snapshots,
    per-Gaussian utility scores). Currently just a flag bag.
    """

    want_energy_grad_snapshot: bool = False
    want_loss_breakdown: bool = True
    # Opt-in handoff of device-resident facts already produced by the normal
    # supervision transaction.  This does not request a probe or rerender.
    want_supervision_evidence: bool = False


@dataclass
class RandomStateToken:
    """A CPU/CUDA RNG restore-and-return token.

    A fixed-topology step can consume randomness (notably fixed-shape MCMC
    noise). ``scoped()`` isolates a private stream; ``continuous()`` leaves
    the consumed state active so caller-owned stochastic work can continue
    the same stream.
    """

    torch_rng_state: bytes | None = None
    cuda_rng_state: bytes | None = None

    @classmethod
    def capture(cls) -> "RandomStateToken":
        """Capture current CPU and CUDA random number generator states.

        Returns:
            A new token holding byte-serialized snapshots of both generators,
            suitable for later restoration via :meth:`restore` or
            contextualization via :meth:`scoped` or :meth:`continuous`.
        """
        import torch

        cpu = torch.get_rng_state().cpu().numpy().tobytes()
        cuda = None
        if torch.cuda.is_available():
            cuda = torch.cuda.get_rng_state().cpu().numpy().tobytes()
        return cls(torch_rng_state=cpu, cuda_rng_state=cuda)

    def restore(self) -> None:
        """Restore CPU and CUDA RNG states from this token.

        Sets both the CPU torch.random and CUDA random generators to the
        states held in this token, making subsequent stochastic work (MCMC
        noise, stochastic gradients, etc.) reproducible.
        """
        import torch

        # PERF (W3-14, root-caused via microbenchmark, see
        # docs/architecture/w3_02_throughput_regression_rootcause.md):
        # ``torch.tensor(list(state_bytes), dtype=torch.uint8)`` iterates
        # every byte of the RNG state into a Python int and rebuilds a
        # tensor element-by-element -- ~0.19ms/call for the ~5KB CPU
        # Mersenne-Twister state alone, pure Python/C-API overhead, not a
        # CUDA/HIP sync. ``torch.frombuffer(bytearray(state_bytes), ...)``
        # does one bulk memcpy instead (measured ~0.0009ms/call, ~200x
        # faster) and was verified byte-for-byte/bit-identical against the
        # old path (both CPU and CUDA generator state, same subsequent
        # random draws). ``bytearray(...)`` copies the immutable ``bytes``
        # into a mutable buffer first so ``frombuffer`` does not alias the
        # original (possibly interned/reused) ``bytes`` object.
        if self.torch_rng_state is not None:
            torch.set_rng_state(
                torch.frombuffer(bytearray(self.torch_rng_state), dtype=torch.uint8)
            )
        if self.cuda_rng_state is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state(
                torch.frombuffer(bytearray(self.cuda_rng_state), dtype=torch.uint8)
            )

    @contextmanager
    def scoped(self) -> Iterator["RandomStateScope"]:
        """Context manager: restore this state, run code, then restore outer.

        Saves the current RNG state, restores this token's state for the
        context body, and restores the saved outer state when exiting.
        Useful for a deterministic sub-computation that must not affect
        later stochastic work.

        Yields:
            A RandomStateScope to which the consumed RNG state can be
            written for inspection by the caller.
        """
        outer = RandomStateToken.capture()
        self.restore()
        scope = RandomStateScope()
        try:
            yield scope
        finally:
            scope.returned = RandomStateToken.capture()
            outer.restore()

    @contextmanager
    def continuous(self) -> Iterator["RandomStateScope"]:
        """Continue this stream in process-global state after the scope.

        The streaming composition root captures the token after stochastic
        background selection.  Leaving the consumed state active lets the
        shared topology/MCMC tail continue the exact same sequence.
        """
        self.restore()
        scope = RandomStateScope()
        try:
            yield scope
        finally:
            scope.returned = RandomStateToken.capture()


@dataclass
class RandomStateScope:
    """Result holder yielded by :meth:`RandomStateToken.scoped`."""

    returned: RandomStateToken | None = None


from eligsir.optimization.gaussian.loss_breakdown import (  # noqa: E402
    LossBreakdown,
)

# NOTE(W3-02 Subagent B): ``LossBreakdown`` used to be defined inline here as
# a placeholder (``total: float`` + ``components: dict[str, float]``). It has
# been superseded by the real definition in ``loss_breakdown.py`` (adds
# ``total_tensor: Any`` — a plain ``float`` cannot carry an autograd graph
# through ``.backward()`` — and, per PR #39 review fix P1, changes
# ``components`` to ``dict[str, Any]`` holding detached ``torch.Tensor``
# values rather than pre-materialized floats, to avoid a host sync on every
# component every step; see ``loss_breakdown.py``'s docstring and
# ``LossBreakdown.materialized_components()`` for the full rationale) and
# re-exported here so every existing ``from .contracts import
# LossBreakdown`` import keeps working unchanged.


@dataclass
class RenderOutputs:
    """Placeholder. TODO(W3-05/W3-06): real definition.

    Stands in for the render package (``render_pkg``) returned by
    ``gaussian_renderer.render``/``eligsir.mapping.renderer``. Carries
    an opaque payload so this module does not need to know the renderer's
    real return shape.
    """

    payload: Any = None


@dataclass
class OptimizationEvidence:
    """Placeholder. TODO(W3-05/W3-06): real definition.

    Stands in for whatever typed evidence (energy-gradient snapshots,
    per-Gaussian utility scores, etc.) a step produces when
    ``EvidenceRequirements`` asks for it. ``None`` at the top level of
    ``OptimizationStepResult`` already covers "no evidence requested"; this
    stub covers the "some evidence requested" shape.
    """

    payload: Any = None


# ---------------------------------------------------------------------------
# Plan section 7.3 target API, reproduced verbatim (field-for-field).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OptimizationStepRequest:
    """Request to execute a single fixed-topology optimization step.

    Encapsulates all inputs required for a complete backward + optimizer step
    (and optional MCMC noise/mutation), with immutable representation to ensure
    reproducibility and cache safety.

    Attributes:
        view: Render view configuration (camera, background, SH update, etc.).
        iteration: Current training iteration (1-based), used as a training clock.
        map_version: Gaussian map generation/mutation counter, incremented on
            every structural change (birth/death).
        topology_version: Mutation event counter, incremented whenever
            relocation occurs, independent of map_version.
        resolution_bucket: Render resolution category (e.g., "default", "half",
            "quarter") for this step.
        evidence_requirements: Specification of optional diagnostics to produce
            (energy gradients, per-cell utility scores, etc.).
        random_state_token: Saved CPU/CUDA RNG state for reproducible
            stochastic work (MCMC noise, stochastic masking, etc.).
    """

    view: OptimizationView
    iteration: int
    map_version: int
    topology_version: int
    resolution_bucket: str
    evidence_requirements: EvidenceRequirements
    random_state_token: RandomStateToken


@dataclass
class OptimizationRuntimeContext:
    """Compatibility boundary between the engine and training composition.

    The streaming loop builds this once from its current configuration and
    policy state. The engine accepts no raw positional training arguments.
    ``gaussians`` remains a narrowly documented bridge while sessions are
    promoted across both model layouts; production callers should provide
    ``session`` and let it expose the fixed-shape model.
    """

    session: Any = None
    gaussians: Any = None
    args: Any = None
    opt: Any = None
    pipe: Any = None
    mcmc_strategy: Any = None
    gradient_policy_context: Any = None
    n_frames_ingested: int = 0
    use_energy_mcmc: bool = False
    sparse_active_set: bool = False
    optimizer_type: str = "adam"
    step_mask: Any = None
    fixed_shape_mcmc: bool = False
    mcmc_local: bool = False
    placement_only: bool = False
    tb_writer: Any = None
    # Streaming-owned state updates derived from a fixed-topology render
    # (visibility/support/recency) must run before objective assembly, but
    # are not numerical optimization themselves.
    post_render_hook: Any = None
    post_optimizer_hook: Any = None
    post_mcmc_hook: Any = None
    timing_scope: Any = None
    loss_assembly_kwargs: dict[str, Any] = field(default_factory=dict)
    # Optional loop-owned Adam ceiling.  The mapping-to-refinement boundary
    # extends optimization past opt.iterations without changing the mapping
    # schedules derived from that author-facing budget.
    optimizer_iteration_limit: int | None = None
    # Canonical online behavior clock.  ``None`` preserves the shared legacy
    # trainer boundary, where ``iteration`` still owns historical schedules.
    optimizer_update_id: int | None = None
    # Explicit phase/work-plan permission for canonical Adam work. ``None``
    # selects the legacy offline iteration-ceiling compatibility branch.
    optimizer_update_permitted: bool | None = None
    # Exact transaction/snapshot versions for optional supervision evidence.
    # ``None`` lets the executor use the session's existing version where it
    # has one. If neither owner supplies an exact version, evidence is
    # unavailable rather than stamped with an invented value.
    evidence_transaction_id: int | str | None = None
    coordinate_frame_version: int | None = None
    schema_version: int | None = None
    surface_evidence_version: int | None = None
    policy_version: int | None = None

    def __post_init__(self) -> None:
        validate_optimizer_clock_contract(self)


def validate_optimizer_clock_contract(runtime: OptimizationRuntimeContext) -> None:
    """Reject mixed canonical/legacy optimizer-clock semantics."""

    if (runtime.optimizer_update_id is None) != (
        runtime.optimizer_update_permitted is None
    ):
        raise ValueError(
            "optimizer_update_id and optimizer_update_permitted must either "
            "both be None or both be set"
        )


@dataclass
class OptimizationStepResult:
    """Result of a completed fixed-topology optimization step.

    Reports loss values, timing, version tracking, and optional diagnostics
    from a single backward + optimizer step cycle. Used to drive post-step
    telemetry, bookkeeping, and iteration-level decision-making in the
    training loop.

    Attributes:
        loss_breakdown: Detailed loss components (L1, SSIM, regularizers, etc.),
            with both materialized floats and autograd-tracked tensors.
        render_outputs: Opaque render output payload (rasterizer result).
        evidence: Optional diagnostic artifacts (energy snapshots, utility
            scores) produced if requested via EvidenceRequirements.
        optimization_gpu_ms: Wall time in milliseconds for the entire step
            (render + backward + optimizer + MCMC), or None if timing is
            unavailable.
        map_version_before: Gaussian map version before this step.
        map_version_after: Gaussian map version after this step (may equal
            `_before` if no mutation occurred).
        topology_version: Mutation event counter at step time, incremented
            only on relocation (independent of map_version).
        random_state_token: Saved RNG state after stochastic work if
            continuity is required by a later consumer; None if state was
            discarded or not captured.
    """

    loss_breakdown: LossBreakdown
    render_outputs: RenderOutputs
    evidence: OptimizationEvidence | SupervisionEvidenceBatch | None
    optimization_gpu_ms: float | None
    map_version_before: int
    map_version_after: int
    topology_version: int
    random_state_token: RandomStateToken | None = None
    parameter_version_before: int = 0
    parameter_version_after: int = 0
    xyz_lr: float | None = None
    # True iff the fixed-topology optimizer update was allowed. The terminal
    # iteration intentionally has step_allowed=False with backward_ran=True:
    # legacy semantics compute gradients there but skip Adam.
    step_allowed: bool = True
    skip_reason: str | None = None
    backward_ran: bool = False
    optimizer_step_ran: bool = False
    mcmc_noise_ran: bool = False
    # Issue #106 batched multi-keyframe supervision clocks. `batch_size` is
    # the REQUESTED number of supervision views for this optimizer step;
    # `supervision_view_count` is how many actually rendered AND backwarded
    # (a placement-only or non-finite-primary step reports 0), and
    # `skipped_view_count` how many requested views were NOT supervised --
    # secondary views dropped for a non-finite loss, PLUS any the caller
    # dropped before dispatch and declared via `run_batched_step`'s
    # `skipped_view_count=` (review of PR #528; before that, a caller-dropped
    # view was invisible here AND silently shrank `batch_size`, so this field
    # widened rather than changed meaning). Then
    # `batch_size == supervision_view_count + skipped_view_count` holds for
    # every step whose PRIMARY view backwarded; a placement-only or
    # non-finite-primary step reports `supervision_view_count == 0` with no
    # matching skip bump, exactly as it did pre-#528. All three are
    # 1/1/0-shaped for an ordinary unbatched step, so existing readers see no
    # change.
    batch_size: int = 1
    supervision_view_count: int = 0
    skipped_view_count: int = 0
    batch_loss_scale: float = 1.0
    # Issue #703 supervision-work accounting. One increment per actual training
    # rasterizer invocation; do NOT count diagnostic/evaluation renders.
    rasterizer_call_count: int = 0
    # Pixels actually entering the supervised objective for this step's training
    # supervision views. For B views, each successfully supervised, sum their
    # W*H. If a view is masked/cropped and fewer pixels participate, keep this
    # separate from rendered_pixel_count (not done today; both currently equal
    # for full-image renders).
    supervised_pixel_count: int = 0
    # Issue #809: number of real `optimizer.step()` calls the engine made for
    # this step. Authoritative — counted at the actual step call site, never
    # derived from `step_allowed` (a permission, not proof a step ran).
    optimizer_step_count: int = 0
    # Exact valid-depth pixels from successful depth-enabled views whose
    # objective reached this result's accepted optimizer update.
    depth_valid_sample_count: int = 0
    # Source indices of the views whose loss actually reached backward(), in
    # execution order. Duplicates are deliberate: each sampled/backwarded draw
    # is one exposure. Absent source identity is omitted rather than guessed.
    successful_supervision_source_indices: tuple[int, ...] = ()
    # Issue #703 execution-shape counters: distinguish nominal camera batching
    # from real vectorization. Track the actual render call groupings, loss
    # assembly pattern, and backward structure to detect sequential work hidden
    # inside a nominal TRUE_CAMERA_BATCH flag.
    renderer_camera_batch_sizes: tuple[int, ...] = ()
    # One entry per rasterizer call: the number of views batched in that call.
    # For SEQUENTIAL_GRADIENT_ACCUMULATION, all entries are 1. For
    # TRUE_CAMERA_BATCH_RENDER_ONLY with microbatch_size=2, e.g., [2, 2].
    loss_objective_call_count: int = 0
    # Total number of assemble_loss() invocations in this step.
    loss_views_per_call: tuple[int, ...] = ()
    # One entry per assemble_loss() call: the number of views supervised in
    # that call. For a per-view loss loop (current main), all entries are 1.
    # For a vectorized loss that processes multiple views in one call, entries
    # would be > 1.
    backward_call_count: int = 0
    # Total number of .backward() calls made during this step.


def finalize_optimization_work_accounting(
    result: OptimizationStepResult,
) -> OptimizationStepResult:
    """Validate the exact work committed by one completed optimizer result.

    Render/backward work that does not reach an optimizer update remains visible
    through the diagnostic flags and rasterizer counter, but it is not canonical
    supervision exposure.  Successful exposure must carry its exact pixel count.
    """

    if not isinstance(result, OptimizationStepResult):
        raise TypeError("result must be OptimizationStepResult")
    for name in (
        "optimizer_step_count",
        "supervision_view_count",
        "supervised_pixel_count",
        "depth_valid_sample_count",
    ):
        value = getattr(result, name)
        if type(value) is not int:
            raise TypeError(f"{name} must be an exact integer")
        if value < 0:
            raise ValueError(f"{name} must be non-negative")

    if result.optimizer_step_count == 0:
        result.supervision_view_count = 0
        result.supervised_pixel_count = 0
        result.depth_valid_sample_count = 0
        result.successful_supervision_source_indices = ()
        return result
    if result.supervision_view_count <= 0:
        raise ValueError("a completed optimizer update requires supervised views")
    if result.supervised_pixel_count <= 0:
        raise ValueError(
            "successful supervision requires positive supervised_pixel_count"
        )
    identities = tuple(result.successful_supervision_source_indices)
    if len(identities) > result.supervision_view_count:
        raise ValueError("successful supervision identities exceed realized views")
    result.successful_supervision_source_indices = identities
    return result


# ---------------------------------------------------------------------------
# Gradient-policy contract. See module docstring for the full rationale
# behind this specific grouping of fields.
# ---------------------------------------------------------------------------


@dataclass
class ActiveSetState:
    """State needed by the sparse active-set (selective-Adam) invisible-row
    zeroing policy. Mirrors ``train_streaming.py``'s
    ``visible | gaussians.provisional.detach()`` step mask construction.
    """

    sparse_active_set: bool = False
    zero_invisible_grads: bool = True
    visible_mask: Any = None  # torch.BoolTensor, shape (N,)


@dataclass
class BubbleState:
    """Spatial active-bubble state. The bubble mask itself is recomputed
    from ``gaussians.get_xyz``/``args`` inside the policy (mirrors the
    inline code's own ``_bubble_active_mask(gaussians.get_xyz, args)``
    calls) — this sub-dataclass only carries whether bubble freezing is
    enabled for this run plus mutable slots for the pre-step freeze
    snapshot a policy stashes and a later restore step consumes.
    """

    enabled: bool = False
    freeze_snapshot: Any = None  # list[tuple[Parameter, Tensor]] | None
    freeze_mask: Any = None  # torch.BoolTensor | None


@dataclass
class StableGeometryState:
    """H7 old-geometry freeze + lifecycle-frozen state. ``bubble`` is
    nested here because both freezes combine into a single
    ``_freeze_grad_mask`` in the inline cascade (frozen | out-of-bubble).
    """

    freeze_old_geometry: bool = False
    freeze_new_frame_steps: int = 0
    iter_since_new_frame: int = 0
    # Canonical semantic age.  None retains the legacy mutable call-count seam.
    successful_updates_since_admission: int | None = None
    young_age_frames: int = 0
    lifecycle_enabled: bool = False
    bubble: BubbleState = field(default_factory=BubbleState)


@dataclass
class AnchorScheduleState:
    """One anchor's weight/decay-steps schedule. Shared shape for both the
    bootstrap anchor (birth_frame==0, decays from iteration 0) and the
    insertion anchor (birth_frame>0, decays per-Gaussian from its own
    ``anchor_iter``) — kept as two separate instances (see
    ``AnchorState`` below), not one shared object, because the two anchors
    are keyed against different clocks and a policy for one must never
    silently read the other's schedule.
    """

    weight: float = 0.0
    decay_steps: int = 1
    full_n_draw: bool = True


@dataclass
class AnchorState:
    """Bootstrap- and insertion-anchor gradient-masking state, grouped
    because both mask ``gaussians.means_param.grad`` via the same
    stochastic-retain-ramp mechanism but are otherwise independent (see
    ``AnchorScheduleState`` docstring for why they are not merged).
    """

    bootstrap: AnchorScheduleState = field(default_factory=AnchorScheduleState)
    insertion: AnchorScheduleState = field(default_factory=AnchorScheduleState)


@dataclass
class DepthLockState:
    """Depth-lock masking state. The lock mask itself
    (``depth_locked_mask(gaussians, args)``) is recomputed inside the
    policy from ``gaussians``/``args``, matching the inline code.
    """

    active: bool = False
    locked_step_mask: Any = None  # torch.BoolTensor | None, set by the policy


@dataclass
class GradientPolicyContext:
    """Single object every :class:`GradientPolicy` receives.

    Resolves the "huge flat parameter list vs. whole loop namespace"
    tension the prior author left open (see ``engine.py`` docstring,
    lines 73-121, and this module's own docstring for the full design
    rationale). ``gaussians``/``args``/``iteration``/``n_frames_ingested``
    are read by essentially every policy and live at the top level;
    everything else is grouped by which of the five gradient policies it
    belongs to (active-set masking, stable-geometry freeze, bootstrap
    anchor, insertion anchor, depth lock).
    """

    gaussians: Any
    args: Any
    iteration: int
    n_frames_ingested: int
    # ID of the successful optimizer update this attempt is trying to commit.
    # It remains unchanged when an attempt is skipped and retried.
    optimizer_update_id: int | None = None
    active_set: ActiveSetState = field(default_factory=ActiveSetState)
    stable_geometry: StableGeometryState = field(default_factory=StableGeometryState)
    anchor: AnchorState = field(default_factory=AnchorState)
    depth_lock: DepthLockState = field(default_factory=DepthLockState)


@dataclass
class GradientPolicyResult:
    """What a single :meth:`GradientPolicy.apply` call reports back.

    Deliberately minimal — just enough for later ordering tests
    (``tests/test_gradient_policy_order.py`` per plan section 7.5) to
    assert which policies ran, in what order, and whether they actually
    touched anything on a given call (some policies are no-ops depending
    on run config, e.g. bootstrap anchor once past its decay window).
    """

    policy_name: str
    applied: bool
    notes: str = ""


@runtime_checkable
class GradientPolicy(Protocol):
    """Protocol every gradient-masking policy implements.

    ``apply`` takes the single :class:`GradientPolicyContext` bundle and
    returns a :class:`GradientPolicyResult`. Implementations mutate
    ``.grad``/``.data`` on ``context.gaussians``' parameters in place
    (matching the inline cascade's own in-place ``torch.no_grad()``
    mutation style) rather than returning new tensors — the result object
    is a report, not the mutated state itself.
    """

    def apply(self, context: GradientPolicyContext) -> GradientPolicyResult:
        """Apply this policy's gradient masking to the given context.

        Inspects context state and mutates ``context.gaussians``'
        parameters' gradients in-place, zeroing or scaling selected rows
        according to this policy's logic (e.g., masking invisible Gaussians,
        freezing old geometry, ramping anchor weights, etc.).

        Args:
            context: Unified state bundle holding the current Gaussian model,
                training config, iteration count, and all per-policy state.

        Returns:
            A GradientPolicyResult describing which policy ran, whether it
            applied any mutations, and optional notes for debugging/telemetry.
        """
        ...


class GradientPolicyBase(abc.ABC):
    """Optional ABC base for policies that want shared boilerplate
    (name resolution, etc.) instead of implementing the bare
    :class:`GradientPolicy` protocol directly. Concrete gradient policies
    (stable-geometry, anchors, depth-lock, active-mask — see plan section
    7.5's ``gradient_policies/`` package) are built by a later subagent in
    this PR; this class exists now purely so that package has a stable
    base to import once it lands.
    """

    name: str = "unnamed_gradient_policy"

    @abc.abstractmethod
    def apply(self, context: GradientPolicyContext) -> GradientPolicyResult:
        """Apply this policy's gradient masking logic.

        Subclasses must implement this method to inspect context state and
        mutate ``context.gaussians``' gradient tensors in-place according to
        their specific masking strategy, then return a result describing
        whether the policy applied any changes.

        Args:
            context: Unified policy state bundle (see :class:`GradientPolicyContext`).

        Returns:
            A GradientPolicyResult indicating policy name, whether mutations
            were applied, and any diagnostic notes.
        """
        raise NotImplementedError
