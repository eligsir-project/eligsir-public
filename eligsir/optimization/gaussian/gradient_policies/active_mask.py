"""Sparse active-set (selective-Adam) invisible-row grad zeroing.

Extracted VERBATIM from ``train_streaming.py`` (as of this PR,
~lines 4532-4540, immediately after ``loss.backward()``/stratified-state
update):

.. code-block:: python

    if sparse_active_set and getattr(args, "selective_adam_zero_invisible_grads", True):
        # Use the same active mask as the regularizer (visible | provisional)
        # so provisional Gaussians that aren't visible keep their gradients.
        _step_mask = visible | gaussians.provisional.detach()
        for group in gaussians.optimizer.param_groups:
            p = group["params"][0]
            if p.grad is not None and getattr(p.grad, "layout", torch.strided) == torch.strided:
                p.grad[~_step_mask] = 0.0

Item #1 of the five-policy cascade (see ``..contracts`` module docstring).
"""

from __future__ import annotations

import torch

from ..contracts import GradientPolicyBase, GradientPolicyContext, GradientPolicyResult


class ActiveMaskGradientPolicy(GradientPolicyBase):
    """Zero invisible-Gaussian grad rows in strided grads (active-set safety).

    Reads ``context.active_set.sparse_active_set``,
    ``context.active_set.zero_invisible_grads``, and
    ``context.active_set.visible_mask`` (the "visible" tensor from the
    inline loop). Requires ``context.gaussians.provisional`` and
    ``context.gaussians.optimizer.param_groups`` to exist, matching the
    inline code's own (unguarded) attribute access.
    """

    name = "active_mask"

    def apply(self, context: GradientPolicyContext) -> GradientPolicyResult:
        active_set = context.active_set
        if not (active_set.sparse_active_set and active_set.zero_invisible_grads):
            return GradientPolicyResult(
                policy_name=self.name,
                applied=False,
                notes="sparse_active_set disabled or zero_invisible_grads=False",
            )

        gaussians = context.gaussians
        visible = active_set.visible_mask
        # Use the same active mask as the regularizer (visible | provisional)
        # so provisional Gaussians that aren't visible keep their gradients.
        step_mask = visible | gaussians.provisional.detach()
        with torch.no_grad():
            for group in gaussians.optimizer.param_groups:
                p = group["params"][0]
                if p.grad is not None and getattr(p.grad, "layout", torch.strided) == torch.strided:
                    p.grad[~step_mask] = 0.0

        return GradientPolicyResult(policy_name=self.name, applied=True)
