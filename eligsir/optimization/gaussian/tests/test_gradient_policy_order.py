"""Exact-order / exact-behavior tests for the extracted gradient policies.

Wave 3, PR W3-02 (plan section 7.6, Subagent C scope). Each policy is a
verbatim relocation of one block of the inline post-backward gradient-
masking cascade in ``train_streaming.py`` (~lines 4532-4700). These tests:

1. Construct a tiny (3-5 Gaussian) fake ``gaussians`` object shaped like
   ``scene/gsplat_model.py``'s ``GsplatGaussianModel`` (a ``ParameterDict``
   with ``means``/``scales``/``quats``/``opacities``/``sh0``/``shN`` plus
   the streaming-only per-Gaussian buffers each policy reads:
   ``birth_frame``, ``provisional``, ``anchor_iter``, ``lifecycle_state``).
2. Run each policy in isolation and hand-verify the resulting ``.grad``
   masking against a hand-computed expected tensor.
3. Run all five policies (six classes; anchors split in two) in the
   documented cascade order against a shared context and diff the result
   against an independent hand-transcription of the original inline logic
   — a local parity proof (a fuller GPU harness is Subagent E's job).
"""

from __future__ import annotations

import torch

from eligsir.optimization.gaussian.contracts import (
    ActiveSetState,
    AnchorScheduleState,
    AnchorState,
    BubbleState,
    DepthLockState,
    GradientPolicyContext,
    StableGeometryState,
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


class _FakeOptimizer:
    def __init__(self, param_groups):
        self.param_groups = param_groups


class _FakeGaussians:
    """Minimal stand-in for ``scene/gsplat_model.py``'s ``GsplatGaussianModel``.

    Only carries the param-dict keys and streaming buffers the five
    gradient policies actually read (checked against the real model file
    before writing this stub).
    """

    def __init__(self, n: int):
        self.params = torch.nn.ParameterDict(
            {
                "means": torch.nn.Parameter(torch.zeros(n, 3)),
                "scales": torch.nn.Parameter(torch.zeros(n, 3)),
                "quats": torch.nn.Parameter(torch.zeros(n, 4)),
                "opacities": torch.nn.Parameter(torch.zeros(n)),
                "sh0": torch.nn.Parameter(torch.zeros(n, 1, 3)),
                "shN": torch.nn.Parameter(torch.zeros(n, 0, 3)),
            }
        )
        self.birth_frame = torch.zeros(n, dtype=torch.int32)
        self.provisional = torch.zeros(n, dtype=torch.bool)
        self.anchor_iter = torch.zeros(n, dtype=torch.int32)
        self.lifecycle_state = torch.zeros(n, dtype=torch.int8)
        self.optimizer = _FakeOptimizer(
            [{"params": [p]} for p in self.params.values()]
        )

    # --- properties mirroring scene/gsplat_model.py -----------------
    @property
    def get_xyz(self):
        return self.params["means"]

    @property
    def means_param(self):
        return self.params["means"]

    @property
    def geometry_params(self):
        return [self.params["means"], self.params["scales"], self.params["quats"]]

    @property
    def all_learnable_params(self):
        return list(self.params.values())

    def set_all_grads(self, value: float = 1.0):
        for p in self.params.values():
            p.grad = torch.full_like(p.data, value)


def _make_context(gaussians, args, iteration=0, n_frames_ingested=0, **overrides):
    return GradientPolicyContext(
        gaussians=gaussians,
        args=args,
        iteration=iteration,
        n_frames_ingested=n_frames_ingested,
        **overrides,
    )


class _Args:
    """Minimal args namespace; only set attributes tests actually need."""


# ---------------------------------------------------------------------------
# 1. ActiveMaskGradientPolicy
# ---------------------------------------------------------------------------


def test_active_mask_policy_zeros_invisible_rows():
    """Verify active mask policy zeros invisible non-provisional rows."""
    g = _FakeGaussians(4)
    g.set_all_grads(1.0)
    visible = torch.tensor([True, False, True, False])
    g.provisional[3] = True  # provisional row 3 should be kept despite not visible

    context = _make_context(
        g,
        _Args(),
        active_set=ActiveSetState(
            sparse_active_set=True, zero_invisible_grads=True, visible_mask=visible
        ),
    )
    result = ActiveMaskGradientPolicy().apply(context)

    assert result.applied is True
    step_mask = visible | g.provisional  # rows 0,2,3 kept; row 1 zeroed
    for p in g.params.values():
        expected = torch.full_like(p.data, 1.0)
        expected[~step_mask] = 0.0
        assert torch.equal(p.grad, expected)


def test_active_mask_policy_noop_when_disabled():
    """Verify active mask policy is noop when disabled."""
    g = _FakeGaussians(3)
    g.set_all_grads(1.0)
    context = _make_context(
        g,
        _Args(),
        active_set=ActiveSetState(sparse_active_set=False, visible_mask=torch.zeros(3, dtype=torch.bool)),
    )
    result = ActiveMaskGradientPolicy().apply(context)
    assert result.applied is False
    for p in g.params.values():
        assert torch.equal(p.grad, torch.ones_like(p.data))


# ---------------------------------------------------------------------------
# 2. StableGeometryGradientPolicy
# ---------------------------------------------------------------------------


def test_stable_geometry_h7_freeze_zeros_old_geometry_grads():
    """Verify stable geometry policy zeros old geometry gradients."""
    g = _FakeGaussians(4)
    g.set_all_grads(1.0)
    # frame_age = n_frames_ingested - birth_frame; young if provisional or frame_age<=young_age_frames
    g.birth_frame[:] = torch.tensor([0, 0, 5, 10], dtype=torch.int32)
    # n_frames_ingested = 10 -> frame_age = [10, 10, 5, 0]
    g.provisional[0] = True  # young via provisional despite old frame_age

    context = _make_context(
        g,
        _Args(),
        iteration=100,
        n_frames_ingested=10,
        stable_geometry=StableGeometryState(
            freeze_old_geometry=True,
            freeze_new_frame_steps=0,  # disable strict-new-frame-phase branch
            iter_since_new_frame=0,
            successful_updates_since_admission=1,
            young_age_frames=5,
            lifecycle_enabled=False,
            bubble=BubbleState(enabled=False),
        ),
    )
    result = StableGeometryGradientPolicy().apply(context)
    assert result.applied is True

    # young = provisional | (frame_age <= 5) = [T, F, T, T] -> old = [F, T, F, F]
    old = torch.tensor([False, True, False, False])
    for p in [g.params["means"], g.params["scales"], g.params["quats"]]:
        expected = torch.ones_like(p.data)
        expected[old] = 0.0
        assert torch.equal(p.grad, expected)
    # non-geometry params untouched by H7 alone
    assert torch.equal(g.params["opacities"].grad, torch.ones_like(g.params["opacities"].data))
    # Policy evaluation is read-only with respect to the successful-update age.
    assert context.stable_geometry.successful_updates_since_admission == 1
    assert context.stable_geometry.iter_since_new_frame == 0


def test_stable_geometry_strict_phase_reads_successful_updates_not_attempt_id():
    """Verify stable geometry strict phase uses successful updates not attempt id."""
    g = _FakeGaussians(2)
    g.set_all_grads(1.0)
    g.birth_frame[:] = torch.tensor([0, 9], dtype=torch.int32)
    context = _make_context(
        g,
        _Args(),
        iteration=10_000,
        n_frames_ingested=10,
        stable_geometry=StableGeometryState(
            freeze_old_geometry=True,
            freeze_new_frame_steps=3,
            successful_updates_since_admission=1,
            young_age_frames=5,
        ),
    )

    StableGeometryGradientPolicy().apply(context)

    # Still in the strict phase despite the unrelated artifact attempt ID.
    assert torch.equal(g.params["means"].grad[:, 0], torch.tensor([0.0, 0.0]))
    assert context.stable_geometry.successful_updates_since_admission == 1


def test_stable_geometry_lifecycle_freeze_zeros_all_params():
    g = _FakeGaussians(3)
    g.set_all_grads(1.0)
    g.lifecycle_state[:] = torch.tensor([0, 3, 0], dtype=torch.int8)

    context = _make_context(
        g,
        _Args(),
        stable_geometry=StableGeometryState(
            freeze_old_geometry=False,
            lifecycle_enabled=True,
            bubble=BubbleState(enabled=False),
        ),
    )
    result = StableGeometryGradientPolicy().apply(context)
    assert result.applied is True

    frozen = torch.tensor([False, True, False])
    for p in g.params.values():
        expected = torch.ones_like(p.data)
        expected[frozen] = 0.0
        assert torch.equal(p.grad, expected)


def test_stable_geometry_noop_when_all_disabled():
    g = _FakeGaussians(3)
    g.set_all_grads(1.0)
    context = _make_context(
        g,
        _Args(),
        stable_geometry=StableGeometryState(
            freeze_old_geometry=False, lifecycle_enabled=False, bubble=BubbleState(enabled=False)
        ),
    )
    result = StableGeometryGradientPolicy().apply(context)
    assert result.applied is False
    for p in g.params.values():
        assert torch.equal(p.grad, torch.ones_like(p.data))


def test_insertion_anchor_age_uses_optimizer_update_id_not_artifact_iteration(monkeypatch):
    g = _FakeGaussians(1)
    g.set_all_grads(1.0)
    g.birth_frame[:] = 1
    g.anchor_iter[:] = 6
    monkeypatch.setattr(torch, "rand", lambda *args, **kwargs: torch.tensor([0.75]))
    context = _make_context(
        g,
        _Args(),
        iteration=10_000,
        optimizer_update_id=7,
        anchor=AnchorState(
            insertion=AnchorScheduleState(weight=1.0, decay_steps=4, full_n_draw=True)
        ),
    )

    InsertionAnchorGradientPolicy().apply(context)

    # Semantic age is 1, retain probability 1/4, so draw .75 is suppressed.
    assert torch.equal(g.params["means"].grad, torch.zeros_like(g.params["means"].grad))


# ---------------------------------------------------------------------------
# 3. Bootstrap / Insertion anchor policies
# ---------------------------------------------------------------------------


def test_bootstrap_anchor_zeros_all_bootstrap_grads_at_iter_zero():
    g = _FakeGaussians(4)
    g.set_all_grads(1.0)
    g.birth_frame[:] = torch.tensor([0, 0, 1, 1], dtype=torch.int32)

    torch.manual_seed(0)
    context = _make_context(
        g,
        _Args(),
        iteration=0,  # retain_frac = 0 -> keep prob 0 for all bootstrap rows
        anchor=AnchorState(
            bootstrap=AnchorScheduleState(weight=1.0, decay_steps=100, full_n_draw=True)
        ),
    )
    result = BootstrapAnchorGradientPolicy().apply(context)
    assert result.applied is True

    means_grad = g.params["means"].grad
    # birth_frame==0 rows (0,1) must be fully zeroed at retain_frac=0
    assert torch.equal(means_grad[0], torch.zeros(3))
    assert torch.equal(means_grad[1], torch.zeros(3))
    # birth_frame>0 rows untouched
    assert torch.equal(means_grad[2], torch.ones(3))
    assert torch.equal(means_grad[3], torch.ones(3))


def test_bootstrap_anchor_noop_when_weight_zero():
    g = _FakeGaussians(3)
    g.set_all_grads(1.0)
    context = _make_context(
        g,
        _Args(),
        iteration=0,
        anchor=AnchorState(bootstrap=AnchorScheduleState(weight=0.0, decay_steps=100)),
    )
    result = BootstrapAnchorGradientPolicy().apply(context)
    assert result.applied is False
    assert torch.equal(g.params["means"].grad, torch.ones_like(g.params["means"].data))


def test_insertion_anchor_zeros_young_inserted_grads_at_age_zero():
    g = _FakeGaussians(4)
    g.set_all_grads(1.0)
    g.birth_frame[:] = torch.tensor([0, 1, 1, 1], dtype=torch.int32)
    g.anchor_iter[:] = torch.tensor([0, 50, 50, 50], dtype=torch.int32)

    context = _make_context(
        g,
        _Args(),
        iteration=50,  # age_all for rows 1-3 = 0 -> retain=0 -> fully zeroed
        anchor=AnchorState(
            insertion=AnchorScheduleState(weight=1.0, decay_steps=200, full_n_draw=True)
        ),
    )
    result = InsertionAnchorGradientPolicy().apply(context)
    assert result.applied is True

    means_grad = g.params["means"].grad
    assert torch.equal(means_grad[0], torch.ones(3))  # birth_frame==0, untouched
    for i in (1, 2, 3):
        assert torch.equal(means_grad[i], torch.zeros(3))


def test_insertion_anchor_noop_when_weight_zero():
    g = _FakeGaussians(3)
    g.set_all_grads(1.0)
    g.birth_frame[:] = torch.tensor([1, 1, 1], dtype=torch.int32)
    context = _make_context(
        g,
        _Args(),
        iteration=10,
        anchor=AnchorState(insertion=AnchorScheduleState(weight=0.0, decay_steps=200)),
    )
    result = InsertionAnchorGradientPolicy().apply(context)
    assert result.applied is False
    assert torch.equal(g.params["means"].grad, torch.ones_like(g.params["means"].data))


# ---------------------------------------------------------------------------
# 4. DepthLockGradientPolicy
# ---------------------------------------------------------------------------


def test_depth_lock_policy_zeros_locked_mean_grads(monkeypatch):
    g = _FakeGaussians(3)
    g.set_all_grads(1.0)

    locked_mask = torch.tensor([False, True, False])

    def _fake_depth_locked_mask(gaussians, args):
        return locked_mask

    calls = {"zeroed": False}

    def _fake_zero_depth_locked_mean_grads(gaussians, args, mask=None):
        calls["zeroed"] = True
        gaussians.means_param.grad[mask] = 0.0
        return int(mask.sum().item())

    monkeypatch.setattr(
        "eligsir.optimization.gaussian.gradient_policies.depth_lock.depth_locked_mask",
        _fake_depth_locked_mask,
    )
    monkeypatch.setattr(
        "eligsir.optimization.gaussian.gradient_policies.depth_lock.zero_depth_locked_mean_grads",
        _fake_zero_depth_locked_mean_grads,
    )

    context = _make_context(g, _Args(), depth_lock=DepthLockState(active=True))
    result = DepthLockGradientPolicy().apply(context)

    assert result.applied is True
    assert calls["zeroed"] is True
    assert torch.equal(context.depth_lock.locked_step_mask, locked_mask)
    means_grad = g.params["means"].grad
    assert torch.equal(means_grad[1], torch.zeros(3))
    assert torch.equal(means_grad[0], torch.ones(3))
    assert torch.equal(means_grad[2], torch.ones(3))


def test_depth_lock_policy_noop_when_inactive():
    g = _FakeGaussians(3)
    g.set_all_grads(1.0)
    context = _make_context(g, _Args(), depth_lock=DepthLockState(active=False))
    result = DepthLockGradientPolicy().apply(context)
    assert result.applied is False
    assert torch.equal(g.params["means"].grad, torch.ones_like(g.params["means"].data))


# ---------------------------------------------------------------------------
# 5. Full-cascade order parity: 5 policies vs. hand-transcribed inline logic
# ---------------------------------------------------------------------------


def _build_shared_gaussians():
    g = _FakeGaussians(5)
    g.set_all_grads(1.0)
    # rows: 0 = bootstrap (birth_frame=0), 1 = young inserted, 2 = old confirmed,
    # 3 = lifecycle-frozen, 4 = depth-locked
    g.birth_frame[:] = torch.tensor([0, 1, 1, 1, 1], dtype=torch.int32)
    g.anchor_iter[:] = torch.tensor([0, 90, 0, 0, 0], dtype=torch.int32)
    g.lifecycle_state[:] = torch.tensor([0, 0, 0, 3, 0], dtype=torch.int8)
    g.provisional[:] = torch.tensor([False, False, False, False, False])
    return g


def _run_policies_in_order(g, args, iteration, n_frames_ingested):
    context = GradientPolicyContext(
        gaussians=g,
        args=args,
        iteration=iteration,
        n_frames_ingested=n_frames_ingested,
        active_set=ActiveSetState(sparse_active_set=False),
        stable_geometry=StableGeometryState(
            freeze_old_geometry=True,
            freeze_new_frame_steps=0,
            iter_since_new_frame=0,
            young_age_frames=0,
            lifecycle_enabled=True,
            bubble=BubbleState(enabled=False),
        ),
        anchor=AnchorState(
            bootstrap=AnchorScheduleState(weight=1.0, decay_steps=100, full_n_draw=True),
            insertion=AnchorScheduleState(weight=1.0, decay_steps=100, full_n_draw=True),
        ),
        depth_lock=DepthLockState(active=False),
    )
    ActiveMaskGradientPolicy().apply(context)
    StableGeometryGradientPolicy().apply(context)
    BootstrapAnchorGradientPolicy().apply(context)
    InsertionAnchorGradientPolicy().apply(context)
    DepthLockGradientPolicy().apply(context)
    return {name: p.grad.clone() for name, p in g.params.items()}


def _hand_transcribed_inline_cascade(g, iteration, n_frames_ingested):
    """Independent transcription of the original inline order, NOT calling
    the policy classes, for a local parity check."""
    with torch.no_grad():
        # H7 freeze (young_age_frames=0, freeze_new_frame_steps=0)
        frame_age = n_frames_ingested - g.birth_frame.long()
        young = g.provisional | (frame_age <= 0)
        old = ~young
        for p in [g.params["means"], g.params["scales"], g.params["quats"]]:
            p.grad[old] = 0.0

        # lifecycle freeze
        freeze_grad_mask = g.lifecycle_state == 3
        for p in g.params.values():
            p.grad[freeze_grad_mask] = 0.0

        # bootstrap anchor (iteration=0 -> retain_frac=0 -> zero all birth_frame==0 rows)
        boot_m = g.birth_frame == 0
        retain_frac = iteration / 100
        if retain_frac < 1.0:
            keep = torch.zeros(boot_m.shape[0], dtype=torch.bool)  # deterministic: rand<0 never true
            zero_boot = boot_m & ~keep
            g.params["means"].grad[zero_boot] = 0.0

        # insertion anchor (row1 age=0 -> retain=0 -> zeroed; row2-4 birth_frame>0 too)
        ins_m = g.birth_frame > 0
        age_all = (iteration - g.anchor_iter.long()).clamp(min=0)
        within = ins_m & (age_all < 100)
        retain = (age_all.float() / 100).clamp(0.0, 1.0)
        keep = torch.zeros(ins_m.shape[0], dtype=torch.bool)  # deterministic: rand<retain never true at retain=0
        zero_ins = within & ~keep
        # only rows where retain==0 are guaranteed zeroed deterministically;
        # this hand test uses iteration==anchor_iter for all birth_frame>0 rows
        g.params["means"].grad[zero_ins] = 0.0

    return {name: p.grad.clone() for name, p in g.params.items()}


def test_full_cascade_order_matches_hand_transcribed_inline_logic():
    iteration = 0
    n_frames_ingested = 0

    g_policies = _build_shared_gaussians()
    g_policies.anchor_iter[:] = torch.tensor([0, 0, 0, 0, 0], dtype=torch.int32)
    policy_grads = _run_policies_in_order(g_policies, _Args(), iteration, n_frames_ingested)

    g_inline = _build_shared_gaussians()
    g_inline.anchor_iter[:] = torch.tensor([0, 0, 0, 0, 0], dtype=torch.int32)
    inline_grads = _hand_transcribed_inline_cascade(g_inline, iteration, n_frames_ingested)

    for key in policy_grads:
        assert torch.equal(policy_grads[key], inline_grads[key]), f"mismatch in {key}"
