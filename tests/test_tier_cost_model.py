"""The tier allocator must choose on measured cost, never on ordinal position.

Cost is monotonic in resolution on every backend measured so far, but only
*sublinear*: ``T(R) = F_device_workload + T_resolution_dependent(R)``, where
the fixed floor ``F`` belongs to the device and the workload.  On this
project's production ROCm backend the half->quarter step buys a real saving
under forward+backward and essentially nothing under forward-only; on a CUDA
dev box it buys about nothing at all.  So the saving is what must be measured,
and a calibration is never transferable across devices or pass modes.

Several tests below feed a deliberately **synthetic, non-monotonic** cost
reading (half cheapest, quarter dearest).  That ordering has *not* been
observed in production -- an earlier report of one was traced to GPU
co-tenancy contamination and retracted (see the ongoing investigation in
PR #1249).  It is used here only because it is adversarial: an allocator that
secretly assumes "lower tier is cheaper" passes every monotonic test and fails
only on input like this.  These tests pin algorithmic robustness, not a claim
about any real accelerator.
"""

from dataclasses import replace

import pytest

from eligsir.contracts.control import MappingTransactionClass, ResolutionTier
from eligsir.mapping.scheduler.tier_cost_model import (
    FORWARD_BACKWARD,
    FORWARD_ONLY,
    LIVE_TIERS,
    TierCostContext,
    TierCostEstimator,
    TierCostSample,
)

# Synthetic adversarial cost evidence -- see this module's docstring.
SYNTHETIC_MS = {
    ResolutionTier.NATIVE: 4.0,
    ResolutionTier.HALF: 2.0,
    ResolutionTier.QUARTER: 8.0,
}

CONTEXT = TierCostContext(
    device_id="test-device",
    gaussian_count=9_500,
    camera_count=4,
    render_mode="true_camera_batch",
    pass_mode=FORWARD_BACKWARD,
)

REPLAY_B32_CONTEXT = replace(CONTEXT, gaussian_count=300_000, camera_count=32)
MIXED_B4_CONTEXT = replace(CONTEXT, gaussian_count=300_000, camera_count=4)


def _sample(tier, wall_ms, *, context=CONTEXT):
    return TierCostSample(
        context=context,
        tier=tier,
        transaction_class=MappingTransactionClass.REPLAY_ONLY,
        prepare_ms=0.4,
        optimization_ms=wall_ms,
        transaction_wall_ms=wall_ms,
        kernel_ms=wall_ms,
    )


def _synthetic_estimator(**kwargs):
    estimator = TierCostEstimator(**kwargs)
    for tier, wall_ms in SYNTHETIC_MS.items():
        estimator.record(_sample(tier, wall_ms))
    return estimator


@pytest.mark.parametrize(
    "budget_ms,expected",
    (
        (10.0, ResolutionTier.NATIVE),
        (5.0, ResolutionTier.NATIVE),
        (3.0, ResolutionTier.HALF),
        (2.5, ResolutionTier.HALF),
        (1.0, ResolutionTier.HALF),
    ),
)
def test_selection_picks_the_highest_quality_feasible_measured_tier(budget_ms, expected):
    """Measured cost, not ordinal position, decides which tier is usable.

    A 3ms budget excludes native (4.0ms) but half (2.0ms) still fits, so the
    answer is half.  At 1ms nothing fits, and the fallback is the measured
    *lowest-cost* legal tier -- which under this synthetic evidence is again
    half, not the nominally smallest tier.
    """

    estimator = _synthetic_estimator()
    selection = estimator.select(
        ceiling=ResolutionTier.NATIVE, budget_ms=budget_ms, context=CONTEXT
    )
    assert selection.tier is expected


def test_a_lower_tier_is_never_chosen_on_pixel_count_alone():
    """Quarter is nominally smallest but measured dearest under this evidence."""

    estimator = _synthetic_estimator()
    for budget_ms in (10.0, 5.0, 3.0, 2.5, 1.0, 0.1):
        selection = estimator.select(
            ceiling=ResolutionTier.NATIVE, budget_ms=budget_ms, context=CONTEXT
        )
        assert selection.tier is not ResolutionTier.QUARTER


def test_monotonic_evidence_reduces_to_the_intuitive_behaviour():
    """With the ordering actually seen on ROCm, tight budgets step down."""

    estimator = TierCostEstimator()
    for tier, wall_ms in (
        (ResolutionTier.NATIVE, 8.053),
        (ResolutionTier.HALF, 3.356),
        (ResolutionTier.QUARTER, 2.368),
    ):
        estimator.record(_sample(tier, wall_ms))

    def _pick(budget_ms):
        return estimator.select(
            ceiling=ResolutionTier.NATIVE, budget_ms=budget_ms, context=CONTEXT
        ).tier

    assert _pick(10.0) is ResolutionTier.NATIVE
    assert _pick(4.0) is ResolutionTier.HALF
    assert _pick(2.5) is ResolutionTier.QUARTER


def test_a_quality_ceiling_caps_the_choice():
    estimator = _synthetic_estimator()

    selection = estimator.select(
        ceiling=ResolutionTier.HALF, budget_ms=10.0, context=CONTEXT
    )

    assert selection.tier is ResolutionTier.HALF


def test_no_feasible_tier_falls_back_to_measured_lowest_cost_without_raising():
    """Resolution alone must not be expected to solve overload."""

    estimator = _synthetic_estimator()
    selection = estimator.select(
        ceiling=ResolutionTier.NATIVE, budget_ms=0.5, context=CONTEXT
    )
    assert selection.tier is ResolutionTier.HALF
    assert selection.feasible_tiers == ()
    assert "other work-budget actuators" in selection.reason


@pytest.mark.parametrize("allow_probe", (False, True))
@pytest.mark.parametrize(
    "costs,ceiling,expected",
        (
            ((1.0, 2.0, 3.0), ResolutionTier.QUARTER, ResolutionTier.QUARTER),
            ((1.0, 2.0, 3.0), ResolutionTier.HALF, ResolutionTier.HALF),
            ((1.0, 3.0, 2.0), ResolutionTier.HALF, ResolutionTier.QUARTER),
            ((4.0, 2.0, 8.0), ResolutionTier.QUARTER, ResolutionTier.QUARTER),
            ((4.0, 2.0, 8.0, 1.0), ResolutionTier.EIGHTH, ResolutionTier.EIGHTH),
    ),
)
def test_infeasible_fallback_chooses_cheapest_measured_tier_within_ceiling(
    costs, ceiling, expected, allow_probe
):
    estimator = TierCostEstimator()
    for tier, cost in zip(LIVE_TIERS, costs):
        estimator.record(_sample(tier, cost))

    selection = estimator.select(
        ceiling=ceiling, budget_ms=0.5, context=CONTEXT, allow_probe=allow_probe
    )

    assert selection.tier is expected
    assert selection.feasible_tiers == ()
    assert selection.measured_tiers == LIVE_TIERS[: len(costs)]
    assert selection.probe is False
    assert "other work-budget actuators" in selection.reason


def test_unmeasured_tiers_are_never_selected_while_measured_evidence_exists():
    estimator = TierCostEstimator()
    estimator.record(_sample(ResolutionTier.HALF, 2.0))

    selection = estimator.select(
        ceiling=ResolutionTier.NATIVE, budget_ms=10.0, context=CONTEXT
    )

    assert selection.tier is ResolutionTier.HALF


def test_a_calibration_never_transfers_between_devices():
    """A CUDA dev-box measurement must not inform a ROCm production decision."""

    estimator = _synthetic_estimator()
    other_device = replace(CONTEXT, device_id="other-device")

    assert estimator.state(ResolutionTier.HALF, context=CONTEXT) is not None
    assert estimator.state(ResolutionTier.HALF, context=other_device) is None
    selection = estimator.select(
        ceiling=ResolutionTier.QUARTER, budget_ms=3.0, context=other_device
    )
    assert selection.tier is ResolutionTier.QUARTER
    assert "no valid measured tier evidence" in selection.reason


@pytest.mark.parametrize(
    "overrides",
    (
        {"gaussian_count": 300_000},
        {"camera_count": 1},
        {"render_mode": "sequential_accumulation"},
        {"pass_mode": FORWARD_ONLY},
    ),
)
def test_evidence_is_keyed_by_workload_not_only_by_tier(overrides):
    """Forward-only, C, map regime and render mode each get their own slot."""

    estimator = _synthetic_estimator()

    assert estimator.state(ResolutionTier.HALF, context=replace(CONTEXT, **overrides)) is None


def test_estimates_are_bounded_and_age_out():
    estimator = TierCostEstimator(window=2, evidence_ttl=2)
    estimator.record(_sample(ResolutionTier.HALF, 100.0))
    estimator.record(_sample(ResolutionTier.HALF, 2.0))
    estimator.record(_sample(ResolutionTier.HALF, 2.0))

    state = estimator.state(ResolutionTier.HALF, context=CONTEXT)
    assert state.completed_samples == 2  # bounded window, no lifetime average
    assert state.recent_mean_ms == pytest.approx(2.0)

    for _ in range(3):
        estimator.record(_sample(ResolutionTier.NATIVE, 3.0))
    aged = estimator.select(
        ceiling=ResolutionTier.NATIVE, budget_ms=10.0, context=CONTEXT
    )
    assert ResolutionTier.HALF not in aged.measured_tiers


def test_bounded_probe_is_reproposed_until_matching_evidence_is_recorded():
    estimator = TierCostEstimator(evidence_ttl=1)
    estimator.record(
        _sample(ResolutionTier.QUARTER, 50.0, context=REPLAY_B32_CONTEXT)
    )

    first = estimator.select(
        ceiling=ResolutionTier.QUARTER,
        budget_ms=160.0,
        context=REPLAY_B32_CONTEXT,
        allow_probe=True,
    )
    assert first.probe is True
    assert first.tier is ResolutionTier.NATIVE

    second = estimator.select(
        ceiling=ResolutionTier.QUARTER,
        budget_ms=160.0,
        context=REPLAY_B32_CONTEXT,
        allow_probe=True,
    )
    assert second.probe is True
    assert second.tier is ResolutionTier.NATIVE
    assert estimator.measured_estimates(context=REPLAY_B32_CONTEXT) == {
        ResolutionTier.QUARTER: 50.0
    }

    estimator.record(
        _sample(ResolutionTier.NATIVE, 120.0, context=REPLAY_B32_CONTEXT)
    )
    settled = estimator.select(
        ceiling=ResolutionTier.NATIVE,
        budget_ms=160.0,
        context=REPLAY_B32_CONTEXT,
        allow_probe=False,
    )
    assert settled.tier is ResolutionTier.NATIVE
    assert settled.probe is False


def test_b4_native_evidence_does_not_calibrate_b32_native_selection():
    """Completed costs are visible only in their own workload bucket."""

    estimator = TierCostEstimator()
    estimator.record(
        _sample(ResolutionTier.QUARTER, 50.0, context=REPLAY_B32_CONTEXT)
    )
    proposal = estimator.select(
        ceiling=ResolutionTier.QUARTER,
        budget_ms=160.0,
        context=REPLAY_B32_CONTEXT,
        allow_probe=True,
    )
    assert proposal.tier is ResolutionTier.NATIVE
    assert proposal.probe is True

    estimator.record(
        _sample(ResolutionTier.NATIVE, 20.0, context=MIXED_B4_CONTEXT)
    )
    assert ResolutionTier.NATIVE in estimator.measured_estimates(
        context=MIXED_B4_CONTEXT
    )
    assert ResolutionTier.NATIVE not in estimator.measured_estimates(
        context=REPLAY_B32_CONTEXT
    )

    for _ in range(2):
        repeated = estimator.select(
            ceiling=ResolutionTier.QUARTER,
            budget_ms=160.0,
            context=REPLAY_B32_CONTEXT,
            allow_probe=True,
        )
        assert repeated.tier is ResolutionTier.NATIVE
        assert repeated.probe is True

    estimator.record(
        _sample(ResolutionTier.NATIVE, 120.0, context=REPLAY_B32_CONTEXT)
    )
    selection = estimator.select(
        ceiling=ResolutionTier.NATIVE,
        budget_ms=160.0,
        context=REPLAY_B32_CONTEXT,
        allow_probe=False,
    )
    assert selection.tier is ResolutionTier.NATIVE
    assert selection.probe is False


def test_no_probe_without_real_headroom():
    estimator = TierCostEstimator()
    estimator.record(_sample(ResolutionTier.HALF, 2.0))

    selection = estimator.select(
        ceiling=ResolutionTier.HALF,
        budget_ms=3.0,
        context=CONTEXT,
        allow_probe=True,
    )

    assert selection.probe is False
    assert selection.tier is ResolutionTier.HALF


def test_quality_floor_narrows_the_legal_tier_set():
    estimator = TierCostEstimator(
        legal_tiers=(ResolutionTier.NATIVE, ResolutionTier.HALF)
    )
    estimator.record(_sample(ResolutionTier.QUARTER, 0.1))
    estimator.record(_sample(ResolutionTier.HALF, 2.0))

    selection = estimator.select(
        ceiling=ResolutionTier.HALF, budget_ms=10.0, context=CONTEXT
    )

    assert selection.tier is ResolutionTier.HALF
    # The out-of-floor quarter sample is observed but never becomes evidence.
    assert ResolutionTier.QUARTER not in selection.measured_tiers


def test_cost_components_stay_separately_identifiable():
    sample = TierCostSample(
        context=CONTEXT,
        tier=ResolutionTier.HALF,
        transaction_class=MappingTransactionClass.FRESH,
        prepare_ms=1.5,
        optimization_ms=2.0,
        transaction_wall_ms=4.2,
        kernel_ms=2.0,
        cache_hits=3,
        cache_misses=1,
    )

    assert (sample.prepare_ms, sample.kernel_ms, sample.transaction_wall_ms) == (
        1.5,
        2.0,
        4.2,
    )
    assert sample.context.bucket() == (
        "test-device",
        "g<25k",
        "c2-4",
        "true_camera_batch",
        FORWARD_BACKWARD,
    )


def test_live_tier_set_includes_eighth_for_adaptive_mapping():
    assert LIVE_TIERS == (
        ResolutionTier.NATIVE,
        ResolutionTier.HALF,
        ResolutionTier.QUARTER,
        ResolutionTier.EIGHTH,
    )
    assert TierCostEstimator().legal_tiers == LIVE_TIERS


def test_measured_estimates_exposes_the_same_evidence_select_decides_on():
    """Read-only view for the work budget's closed-loop budget relaxation:
    the valid (non-aged) per-tier estimates, and nothing before evidence."""
    empty = TierCostEstimator()
    assert empty.measured_estimates(context=CONTEXT) == {}

    estimator = _synthetic_estimator()
    measured = estimator.measured_estimates(context=CONTEXT)
    assert set(measured) == set(SYNTHETIC_MS)
    assert min(measured.values()) == min(
        estimator.select(ceiling=ResolutionTier.NATIVE, budget_ms=0.1, context=CONTEXT)
        .measured_tiers
        and [measured[t] for t in measured]
    )
    # Mutating the returned dict must not touch the estimator's own state.
    measured.clear()
    assert set(estimator.measured_estimates(context=CONTEXT)) == set(SYNTHETIC_MS)
