"""issue #1259 C1 arm: MATURE-quiescence position-noise relaxation gate.

Mirrors (does not import, to stay CUDA/gsplat-free like the sibling
`test_lifecycle_quiescence_isolation.py`) the two predicates added to
`utils/strategies/mcmc/energy.py::GsplatEnergyMCMCStrategy.step_post_backward`:

1. `_m1_relax_mature` -- gates C1 activation on
   (flag AND source_complete AND settle-fence elapsed).
2. The noise-reversion freeze-mask construction, which excludes MATURE/FROZEN
   from the freeze mask only when (1) is true -- relocation-target eligibility
   (a separate, untouched gate) is asserted to stay excluded regardless.

No CUDA / no training / no gsplat import.
"""
import torch


def _m1_relax_mature(flag_enabled, source_complete, updates_since_source_complete, settle_updates):
    """Mirror of the predicate computed once near the top of step_post_backward."""
    return bool(
        flag_enabled
        and source_complete
        and updates_since_source_complete > settle_updates
    )


def _updates_since_source_complete_sequence(source_complete_flags):
    """Mirror of the self.state-backed settling-fence counter."""
    counter = 0
    out = []
    for sc in source_complete_flags:
        counter = counter + 1 if sc else 0
        out.append(counter)
    return out


def _noise_freeze_mask(lifecycle_state, relax_mature):
    """Mirror of the _q_mature_mask construction (lifecycle-only component,
    i.e. before the separate spatial-bubble union)."""
    lifecycle_mature = lifecycle_state >= 2  # MATURE or FROZEN
    if relax_mature:
        return None  # nothing frozen by lifecycle this step
    return lifecycle_mature


def _relocation_eligible_mask(dead_mask, lifecycle_state):
    """Mirror of the relocation dead_mask lifecycle exclusion -- deliberately
    NOT parameterized by relax_mature: C1 must never touch this."""
    return dead_mask & (lifecycle_state < 2)


# --- Byte-identical-when-off ------------------------------------------------

def test_flag_off_never_relaxes_regardless_of_source_complete():
    for sc in (False, True):
        for updates in (0, 1, 16, 17, 1000):
            assert _m1_relax_mature(False, sc, updates, 16) is False


def test_source_incomplete_never_relaxes_even_with_flag_on():
    assert _m1_relax_mature(True, False, 10_000, 16) is False


# --- Settling fence ----------------------------------------------------------

def test_settle_fence_blocks_until_exceeded():
    assert _m1_relax_mature(True, True, 16, 16) is False  # exactly at fence: not yet
    assert _m1_relax_mature(True, True, 17, 16) is True   # one past: relaxed


def test_settle_fence_counter_resets_if_source_completion_flickers():
    # Source-completion transitions are expected to be monotonic in practice,
    # but the counter itself must not accumulate across a False in between.
    seq = _updates_since_source_complete_sequence([False, True, True, False, True])
    assert seq == [0, 1, 2, 0, 1]


# --- Noise-freeze mask relaxation vs. relocation-eligibility isolation ------

def test_noise_freeze_mask_excludes_mature_only_when_relaxed():
    lc = torch.tensor([0, 1, 2, 3], dtype=torch.int8)  # PROVISIONAL/YOUNG/MATURE/FROZEN
    frozen = _noise_freeze_mask(lc, relax_mature=False)
    assert frozen.tolist() == [False, False, True, True]

    relaxed = _noise_freeze_mask(lc, relax_mature=True)
    assert relaxed is None  # nothing frozen by lifecycle this step


def test_relocation_eligibility_is_never_touched_by_c1():
    dead = torch.ones(4, dtype=torch.bool)
    lc = torch.tensor([0, 1, 2, 3], dtype=torch.int8)
    # C1 "on" conceptually (irrelevant here -- function takes no such param,
    # which is the point: relocation eligibility has no C1 hook at all).
    out = _relocation_eligible_mask(dead, lc)
    assert out.tolist() == [True, True, False, False]
