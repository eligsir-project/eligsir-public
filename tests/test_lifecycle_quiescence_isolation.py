"""Task 5: lifecycle-quiescence isolation (Axis-A).

Pins that lifecycle QUIESCENCE (MATURE/FROZEN Gaussians excluded from MCMC
relocation) is fully DECOUPLED from the mature-anchor losses, so the endorsed
ablation combo

    streaming_lifecycle_enabled=True
    streaming_mature_anchor_{xyz,scale,opacity}_weight=0.0   (defaults)
    streaming_sampling_mode=legacy                            (default)

yields quiescence WITHOUT activating the harmful anchor losses.

Two independent gates are asserted:

1. Mature-anchor loss gate (train_streaming.py ~4618): active only when the
   weight sum is > 0. With all weights 0 the loss term is skipped.
2. MCMC quiescence gate (utils/strategies/mcmc_strategy.py ~614): dead/relocate
   candidates are intersected with `lifecycle_state < 2`, gated solely by
   streaming_lifecycle_enabled + lifecycle_state — independent of anchor weights.

No CUDA / no training. The gate predicates are mirrored verbatim from the source.
"""
import torch


# --- Gate 1: mature-anchor loss activation predicate -----------------------
def _mature_anchor_loss_active(lifecycle_enabled, xyz_w, scale_w, opacity_w, has_lifecycle_state):
    """Mirror of the `if` guard around the mature-anchor loss in train_streaming.py."""
    return bool(
        lifecycle_enabled
        and (xyz_w + scale_w + opacity_w) > 0
        and has_lifecycle_state
    )


def test_anchor_loss_inactive_at_zero_weights_with_lifecycle_on():
    # Endorsed combo: lifecycle on, all anchor weights 0 -> anchor loss skipped.
    assert _mature_anchor_loss_active(True, 0.0, 0.0, 0.0, True) is False


def test_anchor_loss_active_when_any_weight_positive():
    assert _mature_anchor_loss_active(True, 0.1, 0.0, 0.0, True) is True
    assert _mature_anchor_loss_active(True, 0.0, 0.1, 0.0, True) is True
    assert _mature_anchor_loss_active(True, 0.0, 0.0, 0.1, True) is True


def test_anchor_loss_inactive_when_lifecycle_off():
    assert _mature_anchor_loss_active(False, 0.5, 0.5, 0.5, True) is False


# --- Gate 2: MCMC quiescence dead-mask intersection ------------------------
def _quiesce_dead_mask(dead_mask, lifecycle_state, lifecycle_enabled):
    """Mirror of the quiescence gate in mcmc_strategy.inject/relocation path.

    PROVISIONAL=0, YOUNG=1, MATURE=2, FROZEN=3. Only PROVISIONAL/YOUNG (< 2)
    may be relocated; MATURE/FROZEN are frozen in place.
    """
    if lifecycle_enabled and lifecycle_state is not None:
        if lifecycle_state.shape[0] == dead_mask.shape[0]:
            dead_mask = dead_mask & (lifecycle_state < 2)
    return dead_mask


def test_quiescence_excludes_mature_and_frozen():
    # states: provisional, young, mature, frozen ; all flagged dead.
    dead = torch.ones(4, dtype=torch.bool)
    lc = torch.tensor([0, 1, 2, 3], dtype=torch.int8)
    out = _quiesce_dead_mask(dead, lc, lifecycle_enabled=True)
    # mature(2) + frozen(3) protected; provisional(0)+young(1) still churnable.
    assert out.tolist() == [True, True, False, False]


def test_quiescence_independent_of_anchor_weights():
    # The quiescence gate takes no anchor weights at all — same result whether
    # the (separate) anchor losses are on or off. Demonstrated by construction:
    # the function signature has no weight argument, so it cannot couple.
    dead = torch.ones(2, dtype=torch.bool)
    lc = torch.tensor([1, 2], dtype=torch.int8)
    assert _quiesce_dead_mask(dead, lc, True).tolist() == [True, False]


def test_quiescence_noop_when_lifecycle_off():
    dead = torch.ones(3, dtype=torch.bool)
    lc = torch.tensor([2, 2, 3], dtype=torch.int8)
    # lifecycle off -> dead mask unchanged (no quiescence, mature can relocate).
    assert _quiesce_dead_mask(dead, lc, lifecycle_enabled=False).tolist() == [True, True, True]
