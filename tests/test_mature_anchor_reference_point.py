"""2026-07-07: mature-anchor reference-point bug fix + absolute-radius safety net.

Root-cause investigation (this session) found two independent, real defects
behind the visually-observed "floaters detaching from surfaces":

1. The YOUNG->MATURE transition (train_streaming.py ~6019-6026) re-snapshots
   `anchor_xyz[to_mature]` at the CURRENT position at maturity time, not the
   true birth/depth-insertion position (already tracked in `anchor_xyz` since
   insertion and never otherwise overwritten). If a Gaussian drifted during
   its YOUNG phase, this re-snapshot locks in that drift as the new "true"
   anchor, so the mature-anchor restoring-force loss can never pull it back
   to where it was actually observed. Fixed behind
   `--streaming_mature_anchor_use_birth_xyz` (default False, preserving the
   historical/buggy behavior so already-queued ablation arms using
   streaming_mature_anchor_*_weight remain unaffected).

2. The relative `scale_kill_threshold` dead-mask (utils/strategies/
   mcmc_strategy.py) explicitly excludes MATURE/FROZEN Gaussians
   (`dead_mask & (lifecycle_state < 2)`), so a >5m-radius floater that
   matures before being killed can never be removed by that mechanism.
   Added an unconditional absolute hard cap,
   `--streaming_max_gaussian_radius_m` (default 0.0 = off), applied in the
   bbox/reach-prune block regardless of lifecycle state.

No CUDA / no training. Both predicates are mirrored verbatim from the source
for isolated, fast testing.
"""
import torch


# --- Bug 1: anchor re-snapshot reference point -----------------------------
def _mature_anchor_snapshot(anchor_xyz, current_xyz, to_mature, use_birth_xyz):
    """Mirror of the anchor_xyz update at the YOUNG->MATURE transition."""
    out = anchor_xyz.clone()
    if not use_birth_xyz:
        out[to_mature] = current_xyz[to_mature].clone()
    # else: leave anchor_xyz (birth/insertion position) untouched.
    return out


def test_default_overwrites_anchor_with_drifted_position():
    # Historical (buggy) behavior preserved by default: birth position is lost.
    birth = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    drifted_now = torch.tensor([[5.0, 5.0, 5.0], [1.0, 1.0, 1.0]])
    to_mature = torch.tensor([True, False])
    out = _mature_anchor_snapshot(birth, drifted_now, to_mature, use_birth_xyz=False)
    assert torch.allclose(out[0], drifted_now[0])  # drift locked in as "true" anchor


def test_birth_xyz_flag_preserves_true_insertion_position():
    birth = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    drifted_now = torch.tensor([[5.0, 5.0, 5.0], [1.0, 1.0, 1.0]])
    to_mature = torch.tensor([True, False])
    out = _mature_anchor_snapshot(birth, drifted_now, to_mature, use_birth_xyz=True)
    assert torch.allclose(out[0], birth[0])  # birth position preserved, drift not locked in
    assert torch.allclose(out[1], birth[1])  # untouched (never matured in this call)


# --- Bug 2: absolute scale safety net vs. relative/lifecycle-gated kill ----
def _relative_scale_kill_mask(max_scale, support, scale_kill_threshold, eff_support, lifecycle_state):
    """Mirror of utils/strategies/mcmc_strategy.py's dead_mask & (lifecycle_state < 2)."""
    mask = (max_scale > scale_kill_threshold) & (support < eff_support)
    mask = mask & (lifecycle_state < 2)
    return mask


def _absolute_radius_mask(max_scale, max_radius_m):
    """Mirror of the new unconditional streaming_max_gaussian_radius_m check."""
    if max_radius_m <= 0:
        return torch.zeros_like(max_scale, dtype=torch.bool)
    return max_scale > max_radius_m


def test_relative_kill_cannot_reach_mature_outlier():
    max_scale = torch.tensor([9.5, 0.05])  # first is the >5m floater
    support = torch.tensor([0.0, 0.0])
    lifecycle_state = torch.tensor([2, 2], dtype=torch.int8)  # both MATURE
    mask = _relative_scale_kill_mask(max_scale, support, 0.20, eff_support=1.0, lifecycle_state=lifecycle_state)
    # Even though 9.5 >> 0.20 threshold, MATURE shielding zeroes it out.
    assert mask.tolist() == [False, False]


def test_absolute_radius_safety_net_reaches_mature_outlier():
    max_scale = torch.tensor([9.5, 0.05])
    mask = _absolute_radius_mask(max_scale, max_radius_m=5.0)
    assert mask.tolist() == [True, False]


def test_absolute_radius_off_by_default():
    max_scale = torch.tensor([9.5, 0.05])
    mask = _absolute_radius_mask(max_scale, max_radius_m=0.0)
    assert mask.tolist() == [False, False]
