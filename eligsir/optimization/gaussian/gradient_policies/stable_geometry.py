"""H7 old-geometry freeze + lifecycle/spatial-bubble freeze + bubble snapshot.

Extracted VERBATIM from ``train_streaming.py`` (as of this PR,
~lines 4542-4700), covering three sub-blocks of the inline cascade that all
read/write the same ``StableGeometryState``/``BubbleState`` grouping (see
``..contracts`` module docstring, item #2 rationale — "``bubble`` is nested
here because both freezes combine into a single ``_freeze_grad_mask`` in
the inline cascade"):

1. **H7 freeze** (item #2a) — zero geometry-param grads for confirmed-old,
   non-provisional Gaussians.
2. **Lifecycle-frozen + spatial-bubble out-of-bounds freeze** (item #3) —
   zero ALL learnable-param grads for lifecycle-frozen or out-of-bubble
   Gaussians (``_freeze_grad_mask = lifecycle_frozen | out_of_bubble``).
3. **Spatial-bubble pre-step freeze snapshot** (item #7) — snapshot the
   out-of-bubble rows of every learnable param BEFORE the optimizer step so
   a later restore step (Subagent D's scope, NOT this policy) can put them
   back bit-for-bit after ``optimizer.step()`` (gradient zeroing alone is
   insufficient because Adam momentum still moves a zero-grad param).

Inline source (verbatim, for reference):

.. code-block:: python

    _iter_since_new_frame += 1
    if _freeze_old and hasattr(gaussians, "birth_frame") and hasattr(gaussians, "provisional"):
        with torch.no_grad():
            _frame_age = n_frames_ingested - gaussians.birth_frame.long()
            _strict_new_frame_phase = (_freeze_new_frame_steps > 0 and
                                       _iter_since_new_frame < _freeze_new_frame_steps)
            if _strict_new_frame_phase:
                _young = gaussians.provisional
            else:
                _young = gaussians.provisional | (_frame_age <= _young_age_frames)
            _old = ~_young
            for _param in gaussians.geometry_params:
                if (_param.grad is not None
                        and getattr(_param.grad, "layout", torch.strided) == torch.strided):
                    _param.grad[_old] = 0.0

    _freeze_grad_mask = None
    if _lifecycle_enabled and hasattr(gaussians, "lifecycle_state"):
        _freeze_grad_mask = gaussians.lifecycle_state == 3
    if getattr(args, "streaming_spatial_bubble_enabled", False):
        from eligsir.utils.strategies.mcmc_strategy import _bubble_active_mask as _bam
        _bub = _bam(gaussians.get_xyz, args)
        if _bub is not None:
            _oob = ~_bub
            _freeze_grad_mask = _oob if _freeze_grad_mask is None else (_freeze_grad_mask | _oob)
    if _freeze_grad_mask is not None:
        with torch.no_grad():
            for _param in gaussians.all_learnable_params:
                if (_param.grad is not None
                        and getattr(_param.grad, "layout", torch.strided) == torch.strided):
                    _param.grad[_freeze_grad_mask] = 0.0

    # ... (bootstrap/insertion anchor + depth-lock blocks are other policies) ...

    _bubble_freeze_snap = None
    _bubble_freeze_mask = None
    if getattr(args, "streaming_spatial_bubble_enabled", False):
        from eligsir.utils.strategies.mcmc_strategy import _bubble_active_mask as _bam2
        _bub2 = _bam2(gaussians.get_xyz, args)
        if _bub2 is not None and bool((~_bub2).any()):
            _bubble_freeze_mask = ~_bub2
            with torch.no_grad():
                _bubble_freeze_snap = [
                    (p, p.data[_bubble_freeze_mask].clone())
                    for p in gaussians.all_learnable_params
                    if p is not None and p.data.shape[0] == _bubble_freeze_mask.shape[0]
                ]

Note this policy's ``apply`` folds in the item #7 snapshot capture too
(stored onto ``context.stable_geometry.bubble.freeze_snapshot``/
``freeze_mask``), matching the grouping the caller (this PR's plan text)
requested — the RESTORE-after-``optimizer.step()`` consumer of that
snapshot is Subagent D's scope, not this file's.
"""

from __future__ import annotations

import torch

from ..contracts import GradientPolicyBase, GradientPolicyContext, GradientPolicyResult


class StableGeometryGradientPolicy(GradientPolicyBase):
    """H7 freeze + lifecycle/bubble freeze + bubble pre-step snapshot."""

    name = "stable_geometry"

    def apply(self, context: GradientPolicyContext) -> GradientPolicyResult:
        gaussians = context.gaussians
        args = context.args
        sg = context.stable_geometry
        notes = []

        if sg.successful_updates_since_admission is None:
            # Shared legacy/offline compatibility. Canonical composition sets
            # the semantic age explicitly and never mutates it during an
            # attempt, so a failed attempt cannot age stable geometry.
            sg.iter_since_new_frame += 1
            updates_since_admission = sg.iter_since_new_frame
        else:
            updates_since_admission = sg.successful_updates_since_admission

        # --- (1) H7: freeze confirmed old geometry ---------------------
        if (
            sg.freeze_old_geometry
            and hasattr(gaussians, "birth_frame")
            and hasattr(gaussians, "provisional")
        ):
            with torch.no_grad():
                frame_age = context.n_frames_ingested - gaussians.birth_frame.long()
                strict_new_frame_phase = (
                    sg.freeze_new_frame_steps > 0
                    and updates_since_admission < sg.freeze_new_frame_steps
                )
                if strict_new_frame_phase:
                    young = gaussians.provisional
                else:
                    young = gaussians.provisional | (frame_age <= sg.young_age_frames)
                old = ~young
                for param in gaussians.geometry_params:
                    if (
                        param.grad is not None
                        and getattr(param.grad, "layout", torch.strided) == torch.strided
                    ):
                        param.grad[old] = 0.0
            notes.append("h7_freeze_applied")

        # --- (2) Lifecycle-frozen + spatial-bubble out-of-bounds freeze -
        freeze_grad_mask = None
        if sg.lifecycle_enabled and hasattr(gaussians, "lifecycle_state"):
            freeze_grad_mask = gaussians.lifecycle_state == 3
        if sg.bubble.enabled:
            from eligsir.utils.strategies.mcmc_strategy import _bubble_active_mask as bam

            bub = bam(gaussians.get_xyz, args)
            if bub is not None:
                oob = ~bub
                freeze_grad_mask = oob if freeze_grad_mask is None else (freeze_grad_mask | oob)
        if freeze_grad_mask is not None:
            with torch.no_grad():
                for param in gaussians.all_learnable_params:
                    if (
                        param.grad is not None
                        and getattr(param.grad, "layout", torch.strided) == torch.strided
                    ):
                        param.grad[freeze_grad_mask] = 0.0
            sg.bubble.freeze_mask = freeze_grad_mask
            notes.append("lifecycle_bubble_freeze_applied")

        # --- (3) Spatial-bubble pre-step freeze snapshot (item #7) ------
        bubble_freeze_snap = None
        bubble_freeze_mask = None
        if sg.bubble.enabled:
            from eligsir.utils.strategies.mcmc_strategy import _bubble_active_mask as bam2

            bub2 = bam2(gaussians.get_xyz, args)
            if bub2 is not None and bool((~bub2).any()):
                bubble_freeze_mask = ~bub2
                with torch.no_grad():
                    bubble_freeze_snap = [
                        (p, p.data[bubble_freeze_mask].clone())
                        for p in gaussians.all_learnable_params
                        if p is not None and p.data.shape[0] == bubble_freeze_mask.shape[0]
                    ]
                notes.append("bubble_snapshot_captured")

        # NOTE: assigning these onto context.stable_geometry.bubble even
        # though the inline code used loop-locals (_bubble_freeze_snap/
        # _bubble_freeze_mask) — this is the typed-context replacement for
        # those loop-locals so the (Subagent D) restore-after-step
        # consumer can read them back off the context.
        sg.bubble.freeze_snapshot = bubble_freeze_snap
        if bubble_freeze_mask is not None:
            sg.bubble.freeze_mask = bubble_freeze_mask

        return GradientPolicyResult(
            policy_name=self.name,
            applied=bool(notes),
            notes=",".join(notes),
        )
