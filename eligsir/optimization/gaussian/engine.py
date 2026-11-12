"""``GaussianOptimizationEngine`` — the Gaussian-optimization concept's home.

Wave 1 consolidation plan step 7 (see
the Gaussian optimization ownership boundary.

Scope, per the plan's explicit correction to the first draft (which had
prune scheduling living inside the optimization engine): render -> loss/
regularizer assembly -> backward -> Adam/selective-Adam step -> learning-
rate update, on a **fixed** parameter shape. This class does not, and must
never, insert/grow/relocate/prune Gaussians or resize optimizer state or
change topology — those are :class:`eligsir.map.mutation.GaussianMutationTransaction`
operations, reached through :meth:`eligsir.map.contracts.GaussianMapBackend.begin_mutation`,
never through this class.

**Scope note for this wave:** ``train_streaming.py``'s actual render/loss/
backward/step sequence (deep inside the ~5200-line ``streaming_training()``
loop body) is not extracted into this class's methods yet — that is a
substantially larger, higher-risk change than the loss-function relocation
this class currently wraps, and isn't verifiable without a live GPU
training run in this environment. What *is* done this step: the loss/
regularizer functions (:mod:`eligsir.optimization.gaussian.losses`)
have a real module home, and this class gives the "optimization session"
concept (already built in step 3/4 as ``GaussianOptimizationSession``) a
named entry point that a future ``MappingCoordinator`` (step 10) can hold
instead of driving the raw model/optimizer directly. Wiring
``streaming_training()``'s actual loop body to construct and call through
this class is deferred to step 10, alongside the ``MappingCoordinator``
build — attempting it in isolation here would mean maintaining two parallel
optimization code paths (the still-inline one and this one) with no way to
verify they stay in sync without GPU access.
"""

from __future__ import annotations

import math
import time
from contextlib import contextmanager, nullcontext
from typing import TYPE_CHECKING, Any, Sequence

import torch

from eligsir.utils.geometry_metrics import update_visibility_ema

from eligsir.mapping.evidence.canonical import EvidenceIdentity
from eligsir.optimization.gaussian.candidate_probe import (
    CANDIDATE_PROBE_EVIDENCE_SCHEMA_VERSION,
    CandidateProbeAccounting,
    CandidateProbeBatchResult,
    CandidateProbeRequest,
    ProbeStateMutationError,
    ProbeUnsafeContextError,
    build_pairwise_evidence,
    capture_state_fingerprint,
    diff_state_fingerprints,
    probe_gaussian_row_count,
    probe_single_view,
)
from eligsir.optimization.gaussian.contracts import (
    GradientPolicyContext,
    LossBreakdown,
    OptimizationRuntimeContext,
    OptimizationStepRequest,
    OptimizationStepResult,
    RandomStateToken,
    RenderOutputs,
    finalize_optimization_work_accounting,
    resolve_supervision_modality,
    validate_optimizer_clock_contract,
)
from eligsir.optimization.gaussian.gradient_policies.active_mask import (
    ActiveMaskGradientPolicy,
)
from eligsir.optimization.gaussian.gradient_policies.anchors import (
    BootstrapAnchorGradientPolicy,
    InsertionAnchorGradientPolicy,
)
from eligsir.optimization.gaussian.gradient_policies.depth_lock import (
    DepthLockGradientPolicy,
)
from eligsir.optimization.gaussian.gradient_policies.stable_geometry import (
    StableGeometryGradientPolicy,
)
from eligsir.optimization.gaussian.losses import (
    _snapshot_energy_grads,
    _streaming_young_isotropy_reg,
    _streaming_young_opacity_reg,
)
from eligsir.optimization.gaussian import objective as _objective
from eligsir.optimization.gaussian.objective import LossAssemblyInputs
from eligsir.optimization.gaussian.render_port import invoke_render
from eligsir.utils.streaming_depth_lock import restore_depth_locked_centers


def _clear_evwb_state_grads(states: dict[int, torch.Tensor] | None) -> None:
    if states:
        for state in states.values():
            state.grad = None


@contextmanager
def _evwb_gradient_transaction(states: dict[int, torch.Tensor] | None):
    """Release transient EV/WB gradients after one optimizer transaction."""
    try:
        yield
    finally:
        _clear_evwb_state_grads(states)


def _commit_evwb_states(states: dict[int, torch.Tensor], args: Any) -> None:
    """Apply one bounded plain-SGD update per accepted train-frame state."""
    if not states:
        return
    learning_rate = float(getattr(args, "streaming_train_evwb_learning_rate", 0.01))
    max_ev = float(getattr(args, "streaming_train_evwb_max_ev", 2.0))
    max_wb = float(getattr(args, "streaming_train_evwb_max_log_wb", 1.0))
    if not all(math.isfinite(value) for value in (learning_rate, max_ev, max_wb)) or min(learning_rate, max_ev, max_wb) <= 0.0:
        raise ValueError("invalid streaming train EV/WB optimizer configuration")
    with torch.no_grad():
        for state in states.values():
            gradient = state.grad
            if gradient is None:
                continue
            finite = torch.isfinite(state).all() & torch.isfinite(gradient).all()
            candidate = state - learning_rate * torch.nan_to_num(gradient)
            ev = candidate[0].clamp(-max_ev, max_ev)
            wb = candidate[1:]
            wb3 = torch.stack((wb[0], wb[1], -wb.sum()))
            scale = (wb3.abs().max() / max_wb).clamp_min(1.0)
            candidate = torch.stack((ev, wb[0] / scale, wb[1] / scale))
            finite = finite & torch.isfinite(candidate).all()
            state.copy_(torch.where(finite, candidate, state))

# PR #39 review fix (P0): `objective.py` used to import `_get_sensor_depth`/
# `_get_surface_prior` straight from `train_streaming` (the root entry
# point), and `train_streaming` imports `MappingCoordinator`, which imports
# THIS module (`engine.py`) -- a real circular import if this module also
# imported `objective` eagerly. Those two helpers now live in
# `eligsir/mapping/integration/depth_helpers.py` (no dependency on
# `train_streaming` or this module), which breaks that cycle -- `objective`
# no longer transitively imports `train_streaming` at module scope, so the
# import above is safe as a normal top-level import.
#
# Note: the module itself is imported (`import ... as _objective`), not
# `assemble_loss` by name -- `run_step()` calls `_objective.assemble_loss(...)`
# so tests can monkeypatch `objective.assemble_loss` (see
# `test_non_finite_loss_skip.py`) and have this module observe the patched
# version; `from ... import assemble_loss` would bind a stale local
# reference at import time that a later monkeypatch on the `objective`
# module wouldn't reach.

if TYPE_CHECKING:
    from eligsir.map.contracts import GaussianMapBackend, GaussianOptimizationSession

# Documented, fixed ordering for the post-gradient-masking cascade (plan
# section 7.4 "Centralize optimizer sequencing", cross-checked against the
# real inline code in train_streaming.py's streaming_training() as of this
# PR, ~lines 4669-4744). Kept as a module-level constant purely so the
# order is discoverable/testable in one place
# (see tests/test_gradient_policy_order.py).
GRADIENT_POLICY_ORDER = (
    "active_mask",
    "stable_geometry",
    "bootstrap_anchor",
    "insertion_anchor",
    "depth_lock",
)


def record_completed_supervision(
    gaussians: Any, visible: Any, *, runtime: OptimizationRuntimeContext
) -> None:
    """Record committed supervision from the complete executed visibility mask."""

    optimizer_update_id = runtime.optimizer_update_id
    if optimizer_update_id is None or visible is None:
        return
    visible = visible.detach().to(dtype=torch.bool).contiguous()
    current = getattr(gaussians, "last_supervised_iter", None)
    if current is None or current.shape[0] != visible.shape[0]:
        current = torch.zeros(
            visible.shape[0], dtype=torch.int32, device=visible.device
        )
        gaussians.last_supervised_iter = current
    current[visible] = int(optimizer_update_id)
    if runtime.fixed_shape_mcmc:
        update_visibility_ema(gaussians, visible)


class GaussianOptimizationEngine:
    """Composes a :class:`GaussianOptimizationSession` (from a
    :class:`eligsir.map.contracts.GaussianMapBackend`) with the canonical
    renderer, loss assembly, backward pass, gradient policies and optimizer
    sequencing. Its public numerical entry point is :meth:`run_step`; all
    work in that method has fixed topology and is bound to the session model.
    """

    def __init__(self, backend: "GaussianMapBackend"):
        self._backend = backend
        self._session: "GaussianOptimizationSession" = backend.begin_optimization()
        # Issue #809: monotonic call counter for optimizer.step() invocations.
        # Used to compute the delta for optimizer_step_count in each batched step.
        self._optimizer_step_calls = 0
        # Issue #803: re-entrancy marker for the optimizer transaction. A
        # candidate probe is only meaningful against a settled map; probing
        # from inside a half-completed step (e.g. from a post_render_hook)
        # would read parameters after render but before the update, and the
        # returned evidence would silently describe a state that never existed
        # as a snapshot. Raising is the whole point -- see `probe_candidates`.
        self._in_optimizer_transaction = False
        # Issue #803 accounting: probe work summed over this engine's lifetime,
        # kept strictly separate from issue #703's supervision counters.
        self._probe_accounting = CandidateProbeAccounting()

    @property
    def session(self) -> "GaussianOptimizationSession":
        """The fixed-shape backend session bound to this engine instance."""
        return self._session

    def young_isotropy_reg(self, gaussians, args, n_frames_ingested):
        return _streaming_young_isotropy_reg(gaussians, args, n_frames_ingested)

    def young_opacity_reg(
        self, gaussians, args, iteration, active_mask, young_exempt_iters: int, maturity_ramp: bool,
    ):
        return _streaming_young_opacity_reg(
            gaussians, args, iteration, active_mask, young_exempt_iters, maturity_ramp,
        )

    def snapshot_energy_grads(self, gaussians):
        return _snapshot_energy_grads(gaussians)

    def render_and_backward(self, gaussians, args, iteration, mcmc_strategy, render_pkg, loss):
        """Own the backward half of the per-iteration render->loss->backward
        sequence in ``train_streaming.py``'s main training loop.

        **Scope note (Wave 2 step 1, deviation from the brief's literal
        wording):** the brief asked for this method to "own the
        render->loss-assembly->backward sequence." Reading the real call
        site (``streaming_training()``, normal-mode loop, around the
        historical ``loss.backward()`` at former line 6234) found the loss
        assembly itself is ~250 lines of sequentially-gated regularizer
        blocks (depth, free-space, mature-anchor, soft-anchor, surface,
        KNN-smoothness, ...), each reading & writing dozens of loop-local
        scalars that are then closed over by ~150 further lines of
        post-backward gradient-masking logic (freeze/bubble/bootstrap-
        anchor/insertion-anchor/depth-lock masks) before
        ``GaussianOptimizationSession.step()`` is ever reached. Physically
        relocating that entire span verbatim — matching
        ``MappingCoordinator``'s own documented precedent of deferring
        ``insert_gaussians_from_frame``'s body move as "separate, larger,
        higher-risk follow-up work" — is out of scope for this step given
        it cannot be decomposed without either (a) inventing a huge
        multi-dozen-field parameter/return contract that is itself a
        design change, not a relocation, or (b) passing the loop's entire
        local namespace, which is not a real interface.

        What this method DOES own for real, and is exercised on every
        iteration of the real hot path once step 1.4 repoints the call
        site: the backward call and, for the retained legacy owner only,
        its optional pre-backward hook. The live population owner passes no
        legacy strategy here, so it proceeds directly to backward:

            mcmc_strategy.step_pre_backward(
                gaussians=gaussians, args=args, iteration=iteration,
                render_pkg=render_pkg, loss=loss,
            )
            loss.backward()

        Loss assembly stays inline in ``train_streaming.py`` (caller-owned,
        passed in as the already-summed ``loss`` tensor); the post-backward
        gradient-masking cascade and ``optimizer.step()`` also stay inline
        (post-backward masking still to be moved in this task's residual
        follow-up work; ``step()``/``zero_grad()``/``update_learning_rate()``
        already route through :class:`GaussianOptimizationSession`).

        Returns ``None`` — this call is a pure side effect (populates
        ``.grad`` on every learnable parameter), exactly like the inline
        ``loss.backward()`` it replaces.
        """
        if mcmc_strategy is not None:
            mcmc_strategy.step_pre_backward(
                gaussians=gaussians, args=args, iteration=iteration,
                render_pkg=render_pkg, loss=loss,
            )
        loss.backward()

    def zero_grad(self) -> None:
        self._session.zero_grad()

    def step(self, visibility=None) -> None:
        self._session.step(visibility)

    def update_learning_rate(self, iteration: int) -> None:
        self._session.update_learning_rate(iteration)

    def _resolve_runtime(self, runtime: OptimizationRuntimeContext) -> tuple[Any, Any]:
        """Return the single session/model pair permitted for a step.

        ``run_optimizer_step`` is deliberately bound to this engine's session.
        Accepting a different runtime session here would let render/backward
        mutate model A while Adam steps model B. The raw model field is only a
        temporary composition-root bridge for test and legacy callers and must
        agree with the session whenever that session exposes a model.
        """
        if runtime.session is not None and runtime.session is not self._session:
            raise ValueError("OptimizationRuntimeContext.session must be the engine session")
        session = self._session
        session_model = getattr(session, "model", None)
        gaussians = runtime.gaussians or session_model
        if gaussians is None:
            raise ValueError("OptimizationRuntimeContext requires the engine session model")
        if session_model is not None and gaussians is not session_model:
            raise ValueError("OptimizationRuntimeContext.gaussians must match session.model")
        return session, gaussians

    # ------------------------------------------------------------------
    # Wave 3, PR W3-02, Subagent D scope: route step/LR-update/normalize/
    # zeroing through the optimization session, and own the post-backward
    # gradient-policy cascade + bubble/depth-lock restore ordering, matching
    # the exact real inline sequence in train_streaming.py's
    # streaming_training() (verified against ~lines 4669-4744 of this PR).
    #
    # LR is updated inside ``run_step`` immediately before render, matching
    # the former streaming-loop ordering. Granular helpers deliberately do
    # not update it, so they cannot accidentally schedule it after Adam.
    # ------------------------------------------------------------------

    def run_gradient_policies(self, context: GradientPolicyContext) -> list:
        """Run the five gradient-masking policies in the documented order
        (``GRADIENT_POLICY_ORDER``): active_mask, stable_geometry,
        bootstrap_anchor, insertion_anchor, depth_lock. Must be called
        AFTER ``render_and_backward`` (i.e. after ``.grad`` is populated)
        and BEFORE ``run_optimizer_step``.

        ``StableGeometryGradientPolicy.apply`` also populates
        ``context.stable_geometry.bubble.freeze_snapshot``/``freeze_mask``
        (the pre-step bubble snapshot) as a side effect — consumed by
        ``run_optimizer_step`` immediately after ``optimizer.step()``.

        Returns the list of ``GradientPolicyResult`` objects, one per
        policy, in call order.
        """
        results = [
            ActiveMaskGradientPolicy().apply(context),
            StableGeometryGradientPolicy().apply(context),
            BootstrapAnchorGradientPolicy().apply(context),
            InsertionAnchorGradientPolicy().apply(context),
            DepthLockGradientPolicy().apply(context),
        ]
        assert [r.policy_name for r in results] == list(GRADIENT_POLICY_ORDER)
        return results

    def run_optimizer_step(
        self,
        gaussians: Any,
        args: Any,
        pipe: Any,
        context: GradientPolicyContext,
        *,
        optimizer_type: str = "adam",
        sparse_active_set: bool = False,
        step_mask: Any = None,
        visible: Any = None,
        use_energy_mcmc: bool = False,
        timing_scope: Any = None,
    ) -> Any:
        """Optimizer step -> bubble-freeze restore -> rotation normalize ->
        depth-lock center restore -> energy-gradient snapshot -> zero_grad,
        in that exact order — the complete post-gradient-policy sequence,
        verified against train_streaming.py's real inline code (as of this
        PR, ~lines 4702-4744):

            if optimizer_type == "selective_adam":
                gaussians.prepare_selective_adam_step(...)
                gaussians.optimizer.step(visibility=step_mask if sparse_active_set else visible)
            else:
                gaussians.optimizer.step()
            # restore bubble-frozen rows bit-for-bit (Adam momentum still
            # moves a zero-grad param, so grad-zeroing alone is insufficient)
            if _bubble_freeze_snap is not None: ...
            if pipe.gsplat_sparse_grad:
                gaussians.normalize_rotation_params(mask=step_mask if sparse_active_set else None)
            if _depth_lock_active:
                restore_depth_locked_centers(gaussians, args, mask=_depth_locked_step_mask)
            _energy_grad_snapshot = _snapshot_energy_grads(gaussians) if use_energy_mcmc else None
            gaussians.optimizer.zero_grad(set_to_none=True)

        Must be called AFTER ``run_gradient_policies`` (relies on
        ``context.stable_geometry.bubble.freeze_snapshot``/``freeze_mask``
        and ``context.depth_lock.locked_step_mask`` populated there) and
        replaces the standalone ``step()``/``zero_grad()`` wrappers above
        for any caller that needs the full sequence, not just the bare
        optimizer step.

        Returns the energy-gradient snapshot (``None`` when
        ``use_energy_mcmc`` is False), mirroring the inline
        ``_energy_grad_snapshot`` local.
        """
        timed = timing_scope or (lambda _name: nullcontext())
        from eligsir.utils.seed_lineage_freeze import seed_lineage_optimizer_guard
        with seed_lineage_optimizer_guard(
            gaussians,
            args,
            backend=self._backend,
            iteration=(
                context.iteration
                if context.optimizer_update_id is None
                else context.optimizer_update_id
            ),
        ):
            with timed("m_optstep"):
                if optimizer_type == "selective_adam":
                    gaussians.prepare_selective_adam_step(
                        allow_dense_grads=getattr(args, "selective_adam_allow_dense_grads", False)
                    )
                    self._session.step(step_mask if sparse_active_set else visible)
                else:
                    self._session.step()
            # Issue #809: count this real optimizer.step() call.
            self._optimizer_step_calls += 1

            # Restore bubble-frozen (out-of-bubble) rows bit-for-bit, BEFORE
            # rotation normalization / depth-lock restore, matching the inline
            # ordering exactly.
            bubble = context.stable_geometry.bubble
            if bubble.freeze_snapshot is not None:
                with torch.no_grad():
                    for p, saved in bubble.freeze_snapshot:
                        if p.data.shape[0] == bubble.freeze_mask.shape[0]:
                            p.data[bubble.freeze_mask] = saved

            if getattr(pipe, "gsplat_sparse_grad", False):
                gaussians.normalize_rotation_params(
                    mask=step_mask if sparse_active_set else None
                )

            if context.depth_lock.active:
                restore_depth_locked_centers(
                    gaussians, args, mask=context.depth_lock.locked_step_mask,
                )

        energy_grad_snapshot = (
            self.snapshot_energy_grads(gaussians) if use_energy_mcmc else None
        )

        self.zero_grad()

        return energy_grad_snapshot

    def run_post_backward_step(
        self,
        gaussians: Any,
        args: Any,
        pipe: Any,
        context: GradientPolicyContext,
        *,
        optimizer_type: str = "adam",
        sparse_active_set: bool = False,
        step_mask: Any = None,
        visible: Any = None,
        use_energy_mcmc: bool = False,
        loss_breakdown: LossBreakdown | None = None,
        render_outputs: RenderOutputs | None = None,
        map_version: int = 0,
        topology_version: int = 0,
        timing_scope: Any = None,
    ) -> OptimizationStepResult:
        """Complete post-``backward()`` sequence: gradient policies ->
        optimizer step (+ bubble/depth-lock restores, rotation normalize,
        energy snapshot, zero_grad). Composes ``run_gradient_policies`` and
        ``run_optimizer_step`` (see both for the exact ordering each owns)
        into the single entry point plan section 7.3's target API implies.

        Topology mutation (insert/prune/grow/relocate/reseed) is never
        reachable from this method or anything it calls — this engine owns
        a fixed-topology numerical update only (plan section 7.4,
        "Explicitly exclude topology mutation"); ``map_version``/
        ``topology_version`` are therefore reported unchanged
        (before == after) rather than recomputed, since no mutation
        happens here to bump them.

        Review fix (PR #39, P0): if ``loss_breakdown.loss_finite`` is
        False, this method does nothing except return a
        ``step_allowed=False`` result — no gradient policies, no optimizer
        step. ``objective.assemble_loss()`` already called
        ``gaussians.optimizer.zero_grad()`` defensively for this case
        (mirrors ``train_streaming.py``'s original non-finite-loss guard),
        so there is nothing further to clean up here. This check exists
        even though ``run_step()`` also checks it before calling
        ``render_and_backward()`` — a caller may invoke this method
        directly (it's one of the "granular methods" this engine
        deliberately keeps available, per ``run_step()``'s own docstring),
        so the guard must hold here too, not just in the composed
        end-to-end wrapper.
        """
        if loss_breakdown is not None and not loss_breakdown.loss_finite:
            return OptimizationStepResult(
                loss_breakdown=loss_breakdown,
                render_outputs=render_outputs,
                evidence=None,
                optimization_gpu_ms=None,
                map_version_before=map_version,
                map_version_after=map_version,
                topology_version=topology_version,
                step_allowed=False,
                skip_reason="non_finite_loss",
            )

        # W3-14C round-3 bisection instrumentation (same SPLAT_PROFILE_PHASES
        # gating as every other `timed(...)` bracket in this file/module; zero
        # overhead when unset). Isolates the five-policy dispatch chain's own
        # cost (object construction + method-call chain across
        # ActiveMaskGradientPolicy/StableGeometryGradientPolicy/
        # BootstrapAnchorGradientPolicy/InsertionAnchorGradientPolicy/
        # DepthLockGradientPolicy) from render/loss/backward/optimizer cost --
        # this dispatch chain has no counterpart at all in the inline OFF path
        # (see docs/architecture/w3_02_completion.md's 2026-08-09 round-3
        # section).
        timed = timing_scope or (lambda _name: nullcontext())
        with timed("m_grad_policies"):
            self.run_gradient_policies(context)
        energy_grad_snapshot = self.run_optimizer_step(
            gaussians, args, pipe, context,
            optimizer_type=optimizer_type,
            sparse_active_set=sparse_active_set,
            step_mask=step_mask,
            visible=visible,
            use_energy_mcmc=use_energy_mcmc,
            timing_scope=timing_scope,
        )

        from eligsir.optimization.gaussian.contracts import OptimizationEvidence

        evidence = (
            OptimizationEvidence(payload=energy_grad_snapshot)
            if use_energy_mcmc and energy_grad_snapshot is not None
            else None
        )

        return OptimizationStepResult(
            loss_breakdown=loss_breakdown,
            render_outputs=render_outputs,
            evidence=evidence,
            optimization_gpu_ms=None,
            map_version_before=map_version,
            map_version_after=map_version,
            topology_version=topology_version,
        )

    def run_step(
        self, request: OptimizationStepRequest, runtime: OptimizationRuntimeContext,
    ) -> OptimizationStepResult:
        """Convenience end-to-end entry point chaining
        render -> loss assembly -> backward -> post-backward step, per plan
        section 7.3's target API. The granular methods exist solely for
        focused parity tests and narrowly-scoped legacy adapters; production
        callers must use this typed entry point.

        Issue #106: this is now a thin ``B=1`` delegation to
        :meth:`run_batched_step`. Delegating (rather than keeping a second,
        parallel single-view implementation) is what makes "``batch_size=1``
        is behaviorally identical to the unbatched engine path" true *by
        construction* instead of by test coverage — there is exactly one
        code path, and ``B=1`` takes the same branches through it that the
        pre-#106 ``run_step`` body took (no loss scaling is applied at
        ``B == 1``, and the visibility union / batch-only bookkeeping below
        is explicitly skipped).
        """
        return self.run_batched_step((request,), runtime)

    def run_batched_step(
        self,
        requests: "Sequence[OptimizationStepRequest]",
        runtime: OptimizationRuntimeContext,
        *,
        skipped_view_count: int = 0,
    ) -> OptimizationStepResult:
        """Multi-keyframe supervision of ONE fixed Gaussian state producing
        ONE optimizer update (issue #106, "Phase 1: accumulate mode").

        ``requests[0]`` is the **primary** view; ``requests[1:]`` are the
        additional supervision views for this optimizer step. All requests
        must agree on ``iteration``/``map_version``/``topology_version`` —
        topology is fixed for the entire supervision batch, which is this
        engine's standing invariant (see the class docstring) and also
        issue #106's explicit requirement.

        Sequence, for ``B = len(requests)``:

        1. ``update_learning_rate`` **once** (from the primary request).
        2. For each view *i*: render -> ``assemble_loss`` (the FULL
           objective, not just the photometric term) -> ``backward`` on
           ``L_i / B``.
        3. Gradient policies -> optimizer step -> MCMC noise, **once**,
           against the **union** of every view's visibility filter.

        Design decisions that are NOT defaults and are load-bearing:

        * **Union visibility mask.** ``selective_adam`` and sparse gradients
          step only the rows in the supplied visibility set. Stepping with
          only the last (or only the primary) view's mask would silently
          drop the ``B-1`` other views' gradients for every Gaussian visible
          in an earlier view but not that one — gradients that were really
          computed and really accumulated into ``.grad``. The union is
          therefore required for correctness, not an optimization. This is
          also why the engine path can support ``B > 1`` together with
          ``selective_adam``/``gsplat_sparse_grad`` at all, where
          ``train_streaming.py``'s already-merged inline accumulation
          (`--streaming_optimization_batch_size`, PR #119) has to refuse
          that combination outright.
        * **``post_render_hook`` fires for the primary view only.** That
          hook does per-*optimizer-step* bookkeeping (recency-of-supervision
          stamping, provisional-support update, visibility EMA, and an
          every-100-iterations promote/stale/conflict-prune sweep). Firing
          it once per *view* would run those cadences ``B x`` faster and
          silently change scheduling rather than only batching supervision.
        * **Full objective per view, not photometric-only.** Every view goes
          through ``objective.assemble_loss`` exactly as the primary view
          does. (The already-merged inline implementation deliberately
          narrows extra views to the photometric term; the engine does not
          need that narrowing because it has one loss-assembly entry point.)
        * **Non-finite policy.** A non-finite PRIMARY loss aborts the whole
          batch before any secondary view is rendered — identical to the
          pre-#106 single-view behavior. A non-finite SECONDARY loss skips
          only that view's backward and the batch continues; the ``1/B``
          divisor is *not* renormalized, because earlier views' backwards
          have already been applied and cannot be rescaled after the fact.
          Skipped views are reported in ``skipped_view_count``.
        * **Views the CALLER already dropped count toward the divisor too**
          (``skipped_view_count=``, review of PR #528). A caller that refuses
          a sampled view *before* dispatch -- e.g. issue #333's tier-homogeneity
          skip, or a sampler with no camera left to give -- must declare it, or
          ``B`` would silently be the *realized* batch size and the same
          non-renormalization policy stated one bullet up would be violated
          from the outside: at ``B=2`` with the one extra view dropped the
          batch collapses to a single request and the loss gets divided by
          nothing (a 2x effective-LR step). Declaring it keeps the divisor at
          the REQUESTED ``B``, matching ``train_streaming.py``'s inline
          accumulation path (whose primary-loss divide is gated on
          ``_opt_batch_size > 1`` alone -- the requested B, whatever the
          reason a slot went unfilled). ``batch_size ==
          supervision_view_count + skipped_view_count`` then holds for every
          step whose primary view backwarded; a placement-only or
          non-finite-PRIMARY step still reports ``supervision_view_count ==
          0`` against a nonzero ``batch_size``, unchanged from before. See
          ``ExtraViewBatchCollector`` in
          ``eligsir/mapping/tier_bind.py`` for the producer.
        """
        if not requests:
            raise ValueError("run_batched_step requires at least one request")
        skipped_view_count = int(skipped_view_count)
        if skipped_view_count < 0:
            raise ValueError("skipped_view_count must be >= 0")
        validate_optimizer_clock_contract(runtime)
        request = requests[0]
        batch_size = len(requests) + skipped_view_count
        for _extra in requests[1:]:
            if (
                _extra.iteration != request.iteration
                or _extra.map_version != request.map_version
                or _extra.topology_version != request.topology_version
            ):
                raise ValueError(
                    "run_batched_step requires a fixed topology and a single "
                    "iteration for the whole supervision batch: every request "
                    "must share iteration/map_version/topology_version"
                )
        session, gaussians = self._resolve_runtime(runtime)
        view = request.view
        context = runtime.gradient_policy_context
        if context is None:
            context = GradientPolicyContext(
                gaussians=gaussians, args=runtime.args, iteration=request.iteration,
                n_frames_ingested=runtime.n_frames_ingested,
                optimizer_update_id=runtime.optimizer_update_id,
            )
        elif context.gaussians is not gaussians:
            raise ValueError("GradientPolicyContext.gaussians must match session.model")
        elif context.optimizer_update_id != runtime.optimizer_update_id:
            raise ValueError(
                "GradientPolicyContext.optimizer_update_id must match "
                "OptimizationRuntimeContext.optimizer_update_id"
            )
        parameter_before = int(getattr(session, "parameter_version", 0) or 0)
        modality = resolve_supervision_modality(view)
        timed = runtime.timing_scope or (lambda _name: nullcontext())
        # #703 counters must be initialized BEFORE the first render below --
        # the primary view's `invoke_render` increments `_rasterizer_calls`
        # well above the loss-assembly block where the rest of this step's
        # counters are seeded. Initializing them down there raised
        # UnboundLocalError on the very first render of every batched step
        # (i.e. every step, since `run_step` delegates here). Not caught by
        # the cpu-ci surface, which has no torch and skips this whole file.
        _rasterizer_calls = 0
        _supervised_pixels = 0
        _depth_valid_samples = 0
        _successful_supervision_source_indices: list[int] = []
        _renderer_camera_batch_sizes: list[int] = []
        _loss_objective_call_count = 0
        _loss_views_per_call: list[int] = []
        _backward_call_count = 0
        _evwb_states: dict[int, torch.Tensor] = {}
        # Issue #809: snapshot the optimizer step call counter BEFORE this step
        # (in case any optimizer step happens inside run_post_backward_step).
        _opt_steps_before = self._optimizer_step_calls
        with (
            self._optimizer_transaction(_evwb_states),
            request.random_state_token.continuous() as rng_scope,
        ):
            # The original streaming loop updates LR before rendering. Keep
            # that ordering inside the one public fixed-topology step.
            optimizer_update_id = runtime.optimizer_update_id
            behavior_update_id = (
                request.iteration
                if optimizer_update_id is None
                else optimizer_update_id
            )
            canonical_update_denied = runtime.optimizer_update_permitted is False
            # Legacy ordering is retained offline. Canonical LR state is not
            # mutated until this attempt is known to be eligible for Adam.
            xyz_lr = (
                session.update_learning_rate(request.iteration)
                if optimizer_update_id is None
                else None
            )
            with timed("m_render"):
                render_outputs = invoke_render(
                    gaussians, view.camera, runtime.pipe, view.background,
                    update_sh_rest=view.update_sh_rest, render_depth=view.render_depth,
                )
                _rasterizer_calls += 1
                _renderer_camera_batch_sizes.append(1)  # Primary is always C=1
            # W3-14C round-3 bisection instrumentation: the streaming loop's
            # post_render_hook (train_streaming.py's _streaming_post_render_hook)
            # does real per-step work (state-length sync, depth-lock buffer
            # ensure, recency-of-supervision bookkeeping, provisional-support
            # update, visibility EMA, the every-100-iters
            # promote/stale/conflict-prune sweep) that the inline OFF path
            # performs at the SAME logical point (immediately after render,
            # before its loss-assembly duplicate) but un-bracketed together
            # with this engine path's "m_loss_assembly" region -- without this
            # separate "m_post_render" bracket, comparing the two arms'
            # "m_loss_assembly" numbers directly would silently fold this
            # hook's cost into only one side. See
            # docs/architecture/w3_02_completion.md's 2026-08-09 round-3
            # section (advisor-caught asymmetry, fixed before the real A/B).
            with timed("m_post_render"):
                if (
                    not canonical_update_denied
                    and runtime.post_render_hook is not None
                ):
                    runtime.post_render_hook(render_outputs)
            image = render_outputs.payload["render"]

            loss_inputs = LossAssemblyInputs(
                gaussians=gaussians, viewpoint_cam=view.camera, args=runtime.args, opt=runtime.opt,
                pipe=runtime.pipe, render_outputs=render_outputs, image=image,
                iteration=request.iteration, n_frames_ingested=runtime.n_frames_ingested,
                optimizer_update_id=optimizer_update_id,
                sparse_active_set=runtime.sparse_active_set,
                rgb_loss_enabled=modality.rgb_loss_enabled,
                use_depth_loss=modality.depth_loss_enabled, use_energy_mcmc=runtime.use_energy_mcmc,
                background=view.background,
                **runtime.loss_assembly_kwargs,
            )
            # W3-14C round-3 bisection instrumentation (SPLAT_PROFILE_PHASES-gated
            # via the injected `timed`/`runtime.timing_scope` callable, zero
            # overhead otherwise): directly comparable to the inline OFF path's
            # "m_loss_assembly" bracket around its separately-maintained loss
            # duplicate -- see train_streaming.py's non-engine branch and
            # docs/architecture/w3_02_completion.md's 2026-08-09 round-3 section.
            with timed("m_loss_assembly"):
                loss_breakdown = _objective.assemble_loss(loss_inputs)
            _loss_objective_call_count += 1
            _loss_views_per_call.append(1)  # Primary view is 1 view per call

            # Non-finite and placement-only iterations skip backward. The
            # terminal iteration deliberately does not: the legacy loop runs
            # pre-backward/backward there, then skips only Adam.
            if runtime.optimizer_update_permitted is None:
                iteration_limit = (
                    runtime.optimizer_iteration_limit
                    if runtime.optimizer_iteration_limit is not None
                    else getattr(runtime.opt, "iterations", float("inf"))
                )
                optimizer_update_denied = request.iteration >= iteration_limit
                endpoint_reason = "iteration_limit"
            else:
                optimizer_update_denied = not runtime.optimizer_update_permitted
                endpoint_reason = "work_plan_endpoint"
            # At B == 1 no scaling op is applied to the loss tensor at all —
            # not even a multiply by 1.0 — so the single-view path is
            # bit-identical to the pre-#106 code.
            _batch_scale = 1.0 if batch_size == 1 else 1.0 / float(batch_size)
            _supervision_views = 0
            # Caller-side drops are already-known skips: seeding the counter
            # with them is what keeps batch_size == supervision + skipped.
            _skipped_views = skipped_view_count
            # (_rasterizer_calls / _supervised_pixels are seeded above, before
            # the primary render that already increments the first of them.)
            _do_backward = (
                not runtime.placement_only
                and loss_breakdown.loss_finite
                and not canonical_update_denied
            )
            if _do_backward:
                if (state := getattr(loss_breakdown, "photometric_evwb_state", None)) is not None:
                    _evwb_states[id(state)] = state
                with timed("m_backward"):
                    self.render_and_backward(
                        gaussians, runtime.args, behavior_update_id, runtime.mcmc_strategy,
                        render_outputs.payload,
                        loss_breakdown.total_tensor
                        if batch_size == 1
                        else loss_breakdown.total_tensor * _batch_scale,
                    )
                _backward_call_count += 1
                _supervision_views = 1
                # Issue #703: count the supervised pixels from this render.
                # image is [3, H, W], so pixel count is H*W.
                _img_h, _img_w = image.shape[-2:]
                _supervised_pixels += _img_h * _img_w
                if view.render_depth:
                    _depth_valid_samples += loss_breakdown.depth_valid_sample_count
                if view.source_index is not None:
                    _successful_supervision_source_indices.append(int(view.source_index))

            visible = render_outputs.payload.get("visibility_filter")

            # ---- Additional supervision views (issue #106, B > 1) --------
            # Only reached when the primary view actually backwarded: a
            # non-finite or placement-only primary aborts the batch before
            # any extra view is rendered, matching the single-view guard.
            # Gated on the REQUEST count, not `batch_size`: a batch that is
            # only nominally >1 (every extra view dropped by the caller) has
            # no second visibility filter to union, and must keep the
            # single-view path's untouched `visible_mask` exactly.
            if len(requests) > 1 and _do_backward:
                _visible_union = (
                    visible.detach().to(dtype=torch.bool).clone()
                    if visible is not None
                    else None
                )
                for _extra_request in requests[1:]:
                    _extra_view = _extra_request.view
                    _extra_modality = resolve_supervision_modality(_extra_view)
                    with timed("m_render"):
                        _extra_outputs = invoke_render(
                            gaussians, _extra_view.camera, runtime.pipe,
                            _extra_view.background,
                            update_sh_rest=_extra_view.update_sh_rest,
                            render_depth=_extra_view.render_depth,
                        )
                        _rasterizer_calls += 1
                        _renderer_camera_batch_sizes.append(1)  # Extra views are also C=1
                    # NB: post_render_hook is deliberately NOT fired here —
                    # see this method's docstring ("primary view only").
                    _extra_image = _extra_outputs.payload["render"]
                    _extra_inputs = LossAssemblyInputs(
                        gaussians=gaussians, viewpoint_cam=_extra_view.camera,
                        args=runtime.args, opt=runtime.opt, pipe=runtime.pipe,
                        render_outputs=_extra_outputs,
                        image=_extra_image,
                        iteration=request.iteration,
                        n_frames_ingested=runtime.n_frames_ingested,
                        optimizer_update_id=optimizer_update_id,
                        sparse_active_set=runtime.sparse_active_set,
                        rgb_loss_enabled=_extra_modality.rgb_loss_enabled,
                        use_depth_loss=_extra_modality.depth_loss_enabled,
                        use_energy_mcmc=runtime.use_energy_mcmc,
                        background=_extra_view.background,
                        **runtime.loss_assembly_kwargs,
                    )
                    with timed("m_loss_assembly"):
                        _extra_breakdown = _objective.assemble_loss(_extra_inputs)
                    _loss_objective_call_count += 1
                    _loss_views_per_call.append(1)  # Extra views are 1 view per call
                    if not _extra_breakdown.loss_finite:
                        # Skip this view only; earlier views' gradients are
                        # already in .grad and stay there (the 1/B divisor is
                        # deliberately not renormalized — see docstring).
                        _skipped_views += 1
                        continue
                    if (state := getattr(_extra_breakdown, "photometric_evwb_state", None)) is not None:
                        _evwb_states[id(state)] = state
                    with timed("m_backward"):
                        self.render_and_backward(
                            gaussians, runtime.args, behavior_update_id,
                            runtime.mcmc_strategy, _extra_outputs.payload,
                            _extra_breakdown.total_tensor * _batch_scale,
                        )
                    _backward_call_count += 1
                    _supervision_views += 1
                    # Issue #703: count the supervised pixels from this extra render.
                    _extra_img_h, _extra_img_w = _extra_image.shape[-2:]
                    _supervised_pixels += _extra_img_h * _extra_img_w
                    if _extra_view.render_depth:
                        _depth_valid_samples += (
                            _extra_breakdown.depth_valid_sample_count
                        )
                    if _extra_view.source_index is not None:
                        _successful_supervision_source_indices.append(
                            int(_extra_view.source_index)
                        )
                    _extra_visible = _extra_outputs.payload.get("visibility_filter")
                    if _extra_visible is not None:
                        _extra_visible = _extra_visible.detach().to(dtype=torch.bool)
                        _visible_union = (
                            _extra_visible.clone()
                            if _visible_union is None
                            else (_visible_union | _extra_visible)
                        )
                if _visible_union is not None:
                    visible = _visible_union
                    # The gradient-policy context's visible_mask was bound to
                    # the PRIMARY view's filter by the caller's post-render
                    # hook (see optimization_runtime.run_streaming_optimization_step).
                    # Every active-set/selective-Adam decision below must see
                    # the union instead, or B-1 views' gradients get masked
                    # away on rows they alone made visible.
                    context.active_set.visible_mask = _visible_union.contiguous()

            step_mask = runtime.step_mask
            if step_mask is None and visible is not None:
                step_mask = visible | gaussians.provisional.detach()

            if runtime.placement_only:
                result = OptimizationStepResult(
                    loss_breakdown=loss_breakdown,
                    render_outputs=render_outputs,
                    evidence=None,
                    optimization_gpu_ms=None,
                    map_version_before=request.map_version,
                    map_version_after=request.map_version,
                    topology_version=request.topology_version,
                    step_allowed=False,
                    skip_reason="placement_only",
                )
            elif loss_breakdown.loss_finite and optimizer_update_denied:
                # Legacy terminal semantics: backward and gradient policies
                # run, while Adam and fixed-shape MCMC noise do not.
                if runtime.optimizer_update_permitted is None:
                    self.run_gradient_policies(context)
                result = OptimizationStepResult(
                    loss_breakdown=loss_breakdown,
                    render_outputs=render_outputs,
                    evidence=None,
                    optimization_gpu_ms=None,
                    map_version_before=request.map_version,
                    map_version_after=request.map_version,
                    topology_version=request.topology_version,
                    step_allowed=False,
                    skip_reason=endpoint_reason,
                    backward_ran=(runtime.optimizer_update_permitted is None),
                )
            else:
                if optimizer_update_id is not None and _do_backward:
                    xyz_lr = session.update_learning_rate(optimizer_update_id)
                result = self.run_post_backward_step(
                    gaussians, runtime.args, runtime.pipe, context,
                    optimizer_type=runtime.optimizer_type,
                    sparse_active_set=runtime.sparse_active_set,
                    step_mask=step_mask, visible=visible,
                    use_energy_mcmc=runtime.use_energy_mcmc,
                    loss_breakdown=loss_breakdown, render_outputs=render_outputs,
                    map_version=request.map_version, topology_version=request.topology_version,
                    timing_scope=runtime.timing_scope,
                )
                result.backward_ran = result.step_allowed
                result.optimizer_step_ran = result.step_allowed
                if result.step_allowed:
                    _commit_evwb_states(_evwb_states, runtime.args)
            if result.step_allowed and runtime.post_optimizer_hook is not None:
                runtime.post_optimizer_hook(result)
            if result.step_allowed and runtime.fixed_shape_mcmc:
                if runtime.mcmc_strategy is None:
                    raise ValueError("fixed_shape_mcmc requires an MCMC strategy")
                with timed("m_noise"):
                    runtime.mcmc_strategy.inject_noise(
                        gaussians=gaussians,
                        args=runtime.args,
                        xyz_lr=xyz_lr,
                        visible=visible if runtime.mcmc_local and runtime.sparse_active_set else None,
                        sparse_active_set=runtime.mcmc_local and runtime.sparse_active_set,
                        iteration=behavior_update_id,
                        tb_writer=runtime.tb_writer,
                    )
                result.mcmc_noise_ran = True
            if result.step_allowed and runtime.post_mcmc_hook is not None:
                runtime.post_mcmc_hook(result)
        result.random_state_token = rng_scope.returned
        result.xyz_lr = xyz_lr
        result.parameter_version_before = parameter_before
        result.parameter_version_after = int(
            getattr(session, "parameter_version", parameter_before) or parameter_before
        )
        # Issue #106 "explicitly separate the clocks": one optimizer step can
        # now consume more than one supervision view, so a run's iteration
        # count is no longer interchangeable with its rendered/supervised-view
        # count. Report both rather than letting callers infer one from the
        # other.
        result.batch_size = batch_size
        result.supervision_view_count = _supervision_views
        result.skipped_view_count = _skipped_views
        result.batch_loss_scale = _batch_scale
        # Issue #703 supervision-work accounting.
        result.rasterizer_call_count = _rasterizer_calls
        result.supervised_pixel_count = _supervised_pixels
        result.renderer_camera_batch_sizes = tuple(_renderer_camera_batch_sizes)
        result.loss_objective_call_count = _loss_objective_call_count
        result.loss_views_per_call = tuple(_loss_views_per_call)
        result.backward_call_count = _backward_call_count
        # Issue #809: real optimizer.step() calls made during THIS step,
        # measured as a delta on the engine's own monotonic call counter.
        result.optimizer_step_count = self._optimizer_step_calls - _opt_steps_before
        if result.optimizer_step_count > 0:
            record_completed_supervision(
                gaussians,
                visible,
                runtime=runtime,
            )
        result.depth_valid_sample_count = (
            _depth_valid_samples if result.optimizer_step_count > 0 else 0
        )
        result.successful_supervision_source_indices = tuple(
            _successful_supervision_source_indices
        )
        return finalize_optimization_work_accounting(result)

    # ------------------------------------------------------------------
    # Issue #803: side-effect-free batched candidate probing.
    # ------------------------------------------------------------------

    @contextmanager
    def _optimizer_transaction(self, evwb_states: dict[int, torch.Tensor] | None = None):
        """Mark the span in which an optimizer transaction is in flight.

        Only :meth:`probe_candidates` reads this. Nesting is not expected and
        is not supported: an engine driving two overlapping optimizer
        transactions is already a bug this flag would only obscure.
        """

        self._in_optimizer_transaction = True
        try:
            yield
        finally:
            _clear_evwb_state_grads(evwb_states)
            self._in_optimizer_transaction = False

    @property
    def probe_accounting(self) -> CandidateProbeAccounting:
        """Cumulative candidate-probe work performed by this engine.

        Deliberately NOT merged into any supervision counter: probe views
        render real pixels and participate in no optimizer update, and issue
        #703's supervision numbers are only trustworthy while that separation
        holds.

        Monotonic for this engine's whole lifetime, with no reset: a consumer
        wanting per-phase attribution should snapshot it and take deltas, the
        same way ``run_batched_step`` derives ``optimizer_step_count``.
        """

        return self._probe_accounting

    def probe_candidates(
        self,
        request: CandidateProbeRequest,
        runtime: OptimizationRuntimeContext,
    ) -> CandidateProbeBatchResult:
        """Render a candidate cohort read-only and return per-view evidence.

        This is the canonical engine-hosted candidate-probe entry point
        (issue #803). It renders every view in ``request`` against exactly one
        Gaussian/topology snapshot -- the CURRENT one, whose versions are read
        here and recorded in the result, so a topology change between queueing
        and probing is reported rather than papered over -- and returns
        unreduced per-view evidence plus (optionally) cheap pairwise
        complementarity evidence.

        It makes **no** admission decision, computes **no** combined quality
        scalar, and mutates **nothing**:

        * no ``optimizer.step``, ``zero_grad`` or LR-scheduler advance;
        * no mutation/topology change and no parameter update;
        * no ``.grad`` accumulation (forward-only mode runs under
          ``no_grad``; gradient mode uses ``torch.autograd.grad``, which
          returns gradients rather than accumulating them);
        * no RNG-stream consumption (the whole batch runs inside
          :meth:`RandomStateToken.scoped`);
        * no replay/tier/admission state change -- including the camera's own
          resolution binding, which is restored per view.

        Every one of those is *verified*, not merely intended: a state
        fingerprint is taken before and after and any difference raises
        :class:`ProbeStateMutationError`. Probing from inside an open optimizer
        transaction raises :class:`ProbeUnsafeContextError`.

        Args:
            request: The cohort, probe resolution tier, and evidence channels.
            runtime: The same runtime context an optimization step would use;
                only ``session``/``gaussians``/``pipe``/``opt`` are read.

        Returns:
            A :class:`CandidateProbeBatchResult` whose ``per_view`` tuple is
            index-aligned with ``request.views`` (a skipped candidate keeps its
            slot and carries its own reason) and whose ``accounting`` reports
            this batch's probe work under the ``candidate_probe_*`` names.
        """

        if self._in_optimizer_transaction:
            raise ProbeUnsafeContextError(
                "probe_candidates() was called from inside an optimizer "
                "transaction; a probe must observe a settled map, not a "
                "half-completed step"
            )
        session, gaussians = self._resolve_runtime(runtime)
        backend = self._backend
        map_version = int(getattr(backend, "map_content_version", 0) or 0)
        topology_version = int(getattr(backend, "topology_version", 0) or 0)
        coordinate_frame_version = int(
            getattr(backend, "coordinate_frame_version", 0) or 0
        )
        lambda_dssim = request.lambda_dssim
        if lambda_dssim is None:
            lambda_dssim = float(
                getattr(runtime.opt, "lambda_dssim", 0.2)
                if runtime.opt is not None
                else 0.2
            )
        gaussian_count = probe_gaussian_row_count(gaussians)

        before = capture_state_fingerprint(
            backend=backend,
            session=session,
            gaussians=gaussians,
            extra={"engine.optimizer_step_calls": self._optimizer_step_calls},
        )

        base_identity = EvidenceIdentity(
            source_id="",
            source_ordinal=-1,
            timestamp_ns=0,
            map_version=map_version,
            topology_version=topology_version,
            coordinate_frame_version=coordinate_frame_version,
            optimization_step=int(request.optimization_step),
            objective_version=str(request.objective_version),
            resolution_bucket=request.resolution_tier.value,
        )

        synchronize = bool(request.synchronize_device) and torch.cuda.is_available()
        if synchronize:
            torch.cuda.synchronize()
        batch_start = time.perf_counter()
        results: list = []
        # `scoped()` (not `continuous()`): a probe must leave the process-global
        # RNG stream exactly where it found it, so the caller's MCMC/noise
        # sequence is bit-identical to a run in which no probe ever happened.
        with RandomStateToken.capture().scoped():
            for view_index, view in enumerate(request.views):
                results.append(
                    probe_single_view(
                        gaussians=gaussians,
                        view=view,
                        request=request,
                        pipe=runtime.pipe,
                        identity=base_identity,
                        view_index=view_index,
                        gaussian_count=gaussian_count,
                        lambda_dssim=float(lambda_dssim),
                    )
                )
        if synchronize:
            torch.cuda.synchronize()
        batch_wall_ms = (time.perf_counter() - batch_start) * 1000.0

        after = capture_state_fingerprint(
            backend=backend,
            session=session,
            gaussians=gaussians,
            extra={"engine.optimizer_step_calls": self._optimizer_step_calls},
        )
        problems = diff_state_fingerprints(before, after)
        if problems:
            raise ProbeStateMutationError(
                "candidate probe mutated optimization state: " + "; ".join(problems)
            )

        pairwise = (
            build_pairwise_evidence(results) if request.compute_pairwise else ()
        )
        skipped = sum(1 for result in results if result.skipped)
        accounting = CandidateProbeAccounting(
            candidate_probe_batch_count=1,
            candidate_probe_view_count=len(results) - skipped,
            candidate_probe_rasterizer_call_count=sum(
                result.rasterizer_call_count for result in results
            ),
            candidate_probe_pixel_count=sum(
                result.probe_pixel_count for result in results
            ),
            candidate_probe_wall_ms=batch_wall_ms,
            candidate_probe_gpu_ms=batch_wall_ms if synchronize else 0.0,
            candidate_probe_gradient_view_count=sum(
                1 for result in results if result.gradient is not None
            ),
            candidate_probe_skipped_view_count=skipped,
        )
        self._probe_accounting = self._probe_accounting.merged(accounting)
        return CandidateProbeBatchResult(
            probe_batch_id=str(request.probe_batch_id),
            per_view=tuple(results),
            pairwise=pairwise,
            map_version=map_version,
            topology_version=topology_version,
            coordinate_frame_version=coordinate_frame_version,
            optimization_step=int(request.optimization_step),
            resolution_tier=request.resolution_tier,
            resolution_bucket=request.resolution_tier.value,
            objective_version=str(request.objective_version),
            evidence_schema_version=CANDIDATE_PROBE_EVIDENCE_SCHEMA_VERSION,
            accounting=accounting,
            gaussian_count=gaussian_count,
            evidence_channels=request.evidence_channels,
        )
