"""Bootstrap-anchor and insertion-anchor means gradient masking.

Extracted VERBATIM from ``train_streaming.py`` (as of this PR,
~lines 4585-4667). Two separate policy classes per ``..contracts``'s own
rationale for keeping ``AnchorState.bootstrap``/``AnchorState.insertion``
distinct (different clocks: iteration 0 vs. each Gaussian's own
``anchor_iter``) — see ``..contracts`` module docstring, items #3/#4.

Bootstrap inline source (verbatim):

.. code-block:: python

    if (_bootstrap_anchor_weight > 0 and iteration < _bootstrap_anchor_decay
            and hasattr(gaussians, "birth_frame")):
        with torch.no_grad():
            _boot_m = (gaussians.birth_frame == 0)
            _retain_frac = iteration / _bootstrap_anchor_decay
            if _boot_m.shape[0] == gaussians.get_xyz.shape[0] and _retain_frac < 1.0:
                if getattr(args, "streaming_anchor_fullN_draw", True):
                    _dev = gaussians.get_xyz.device
                    _keep = torch.rand(_boot_m.shape[0], device=_dev) < _retain_frac
                    _zero_boot = _boot_m & ~_keep
                    _p_means = gaussians.means_param
                    if _p_means.grad is not None:
                        if getattr(_p_means.grad, "layout", torch.strided) != torch.strided:
                            _p_means.grad = _p_means.grad.to_dense().contiguous()
                        _p_means.grad[_zero_boot] = 0.0
                elif _boot_m.any():
                    _rand = torch.rand(_boot_m.sum(), device=gaussians.get_xyz.device)
                    _zero_boot = _boot_m.clone()
                    _zero_boot[_boot_m] = _rand >= _retain_frac
                    _p_means = gaussians.means_param
                    if _p_means.grad is not None:
                        if getattr(_p_means.grad, "layout", torch.strided) != torch.strided:
                            _p_means.grad = _p_means.grad.to_dense().contiguous()
                        _p_means.grad[_zero_boot] = 0.0

Insertion inline source (verbatim):

.. code-block:: python

    if (_insertion_anchor_weight > 0
            and hasattr(gaussians, "birth_frame")
            and hasattr(gaussians, "anchor_iter")):
        with torch.no_grad():
            _ins_m = (gaussians.birth_frame > 0)
            if _ins_m.shape[0] == gaussians.get_xyz.shape[0]:
                _age_all = (iteration - gaussians.anchor_iter.long()).clamp(min=0)
                _within = _ins_m & (_age_all < _insertion_anchor_decay)
                if getattr(args, "streaming_anchor_fullN_draw", True):
                    _dev = gaussians.get_xyz.device
                    _retain = (_age_all.float() / _insertion_anchor_decay).clamp(0.0, 1.0)
                    _keep = torch.rand(_ins_m.shape[0], device=_dev) < _retain
                    _zero_ins = _within & ~_keep
                    _p_means = gaussians.means_param
                    if _p_means.grad is not None:
                        if getattr(_p_means.grad, "layout", torch.strided) != torch.strided:
                            _p_means.grad = _p_means.grad.to_dense().contiguous()
                        _p_means.grad[_zero_ins] = 0.0
                elif _ins_m.any() and _within.any():
                    _age_young = _age_all[_within].float()
                    _retain_per = (_age_young / _insertion_anchor_decay).clamp(0.0, 1.0)
                    _rand_ins = torch.rand(_retain_per.shape[0], device=gaussians.get_xyz.device)
                    _zero_ins = _within.clone()
                    _zero_ins[_within] = _rand_ins >= _retain_per
                    _p_means = gaussians.means_param
                    if _p_means.grad is not None:
                        if getattr(_p_means.grad, "layout", torch.strided) != torch.strided:
                            _p_means.grad = _p_means.grad.to_dense().contiguous()
                        _p_means.grad[_zero_ins] = 0.0
"""

from __future__ import annotations

import torch

from ..contracts import GradientPolicyBase, GradientPolicyContext, GradientPolicyResult


class BootstrapAnchorGradientPolicy(GradientPolicyBase):
    """Suppress center drift of depth-seeded Gaussians (birth_frame==0)
    during early training via a linearly-decaying gradient-zero ramp."""

    name = "bootstrap_anchor"

    def apply(self, context: GradientPolicyContext) -> GradientPolicyResult:
        sched = context.anchor.bootstrap
        gaussians = context.gaussians
        args = context.args
        optimizer_update_id = (
            context.iteration
            if context.optimizer_update_id is None
            else context.optimizer_update_id
        )

        if not (
            sched.weight > 0
            and optimizer_update_id < sched.decay_steps
            and hasattr(gaussians, "birth_frame")
        ):
            return GradientPolicyResult(policy_name=self.name, applied=False)

        applied = False
        with torch.no_grad():
            boot_m = gaussians.birth_frame == 0
            retain_frac = optimizer_update_id / sched.decay_steps
            if boot_m.shape[0] == gaussians.get_xyz.shape[0] and retain_frac < 1.0:
                if sched.full_n_draw:
                    dev = gaussians.get_xyz.device
                    keep = torch.rand(boot_m.shape[0], device=dev) < retain_frac
                    zero_boot = boot_m & ~keep
                    p_means = gaussians.means_param
                    if p_means.grad is not None:
                        if getattr(p_means.grad, "layout", torch.strided) != torch.strided:
                            p_means.grad = p_means.grad.to_dense().contiguous()
                        p_means.grad[zero_boot] = 0.0
                        applied = True
                elif boot_m.any():
                    rand = torch.rand(boot_m.sum(), device=gaussians.get_xyz.device)
                    zero_boot = boot_m.clone()
                    zero_boot[boot_m] = rand >= retain_frac
                    p_means = gaussians.means_param
                    if p_means.grad is not None:
                        if getattr(p_means.grad, "layout", torch.strided) != torch.strided:
                            p_means.grad = p_means.grad.to_dense().contiguous()
                        p_means.grad[zero_boot] = 0.0
                        applied = True

        return GradientPolicyResult(policy_name=self.name, applied=applied)


class InsertionAnchorGradientPolicy(GradientPolicyBase):
    """Per-Gaussian stochastic suppression for depth-inserted Gaussians
    (birth_frame>0) during their first N iterations after ``anchor_iter``,
    keyed off each Gaussian's own age rather than a global clock."""

    name = "insertion_anchor"

    def apply(self, context: GradientPolicyContext) -> GradientPolicyResult:
        sched = context.anchor.insertion
        gaussians = context.gaussians
        args = context.args
        optimizer_update_id = (
            context.iteration
            if context.optimizer_update_id is None
            else context.optimizer_update_id
        )

        if not (
            sched.weight > 0
            and hasattr(gaussians, "birth_frame")
            and hasattr(gaussians, "anchor_iter")
        ):
            return GradientPolicyResult(policy_name=self.name, applied=False)

        applied = False
        with torch.no_grad():
            ins_m = gaussians.birth_frame > 0
            if ins_m.shape[0] == gaussians.get_xyz.shape[0]:
                age_all = (
                    optimizer_update_id - gaussians.anchor_iter.long()
                ).clamp(min=0)
                within = ins_m & (age_all < sched.decay_steps)
                if sched.full_n_draw:
                    dev = gaussians.get_xyz.device
                    retain = (age_all.float() / sched.decay_steps).clamp(0.0, 1.0)
                    keep = torch.rand(ins_m.shape[0], device=dev) < retain
                    zero_ins = within & ~keep
                    p_means = gaussians.means_param
                    if p_means.grad is not None:
                        if getattr(p_means.grad, "layout", torch.strided) != torch.strided:
                            p_means.grad = p_means.grad.to_dense().contiguous()
                        p_means.grad[zero_ins] = 0.0
                        applied = True
                elif ins_m.any() and within.any():
                    age_young = age_all[within].float()
                    retain_per = (age_young / sched.decay_steps).clamp(0.0, 1.0)
                    rand_ins = torch.rand(retain_per.shape[0], device=gaussians.get_xyz.device)
                    zero_ins = within.clone()
                    zero_ins[within] = rand_ins >= retain_per
                    p_means = gaussians.means_param
                    if p_means.grad is not None:
                        if getattr(p_means.grad, "layout", torch.strided) != torch.strided:
                            p_means.grad = p_means.grad.to_dense().contiguous()
                        p_means.grad[zero_ins] = 0.0
                        applied = True

        return GradientPolicyResult(policy_name=self.name, applied=applied)
