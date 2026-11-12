"""Depth-lock gradient masking for the fixed-topology optimizer hot path."""

from __future__ import annotations

import torch

from eligsir.utils.streaming_depth_lock import depth_locked_mask, zero_depth_locked_mean_grads

from ..contracts import GradientPolicyBase, GradientPolicyContext, GradientPolicyResult


class DepthLockGradientPolicy(GradientPolicyBase):
    """Zero mean gradients for depth-locked anchor-pinned Gaussians.

    Once the policy is enabled, the mask stays device-resident and masking is
    submitted unconditionally. An all-false mask is a natural device no-op;
    converting it to a Python bool would serialize CPU submission with the
    accelerator on every optimizer transaction.
    """

    name = "depth_lock"

    def apply(self, context: GradientPolicyContext) -> GradientPolicyResult:
        dl = context.depth_lock
        if not dl.active:
            return GradientPolicyResult(policy_name=self.name, applied=False)

        gaussians = context.gaussians
        args = context.args

        with torch.no_grad():
            locked_step_mask = depth_locked_mask(gaussians, args)
        dl.locked_step_mask = locked_step_mask
        zero_depth_locked_mean_grads(gaussians, args, mask=locked_step_mask)

        # "applied" means the enabled policy stage executed.  Row cardinality
        # deliberately stays on-device and is not materialized into Python.
        return GradientPolicyResult(policy_name=self.name, applied=True)
