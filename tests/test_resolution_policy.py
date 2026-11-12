from dataclasses import replace

from eligsir.contracts.control import (
    ControlLimits,
    ControlSnapshot,
    EvidenceFact,
    FreshnessMode,
    ResolutionTier,
)
from eligsir.mapping.scheduler.resolution_policy import ResolutionPolicyProvider


def _snapshot(
    mode,
    depth=0,
    *,
    admitted=0,
    value=0.5,
    service_ratio=1.0,
    frontier_age_ms=10.0,
    frontier_slope=0.0,
    gpu_cost_ms=None,
    epoch=0,
):
    return ControlSnapshot(
        1,
        1,
        None,
        None,
        (),
        None,
        mode,
        0,
        depth,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        ControlLimits(1, 0, 1, 0, 0, 0),
        control_tick_epoch=epoch,
        evidence=(
            EvidenceFact("mapping_service_ratio", service_ratio),
            EvidenceFact("frontier_age_ms", frontier_age_ms),
            EvidenceFact("frontier_age_slope_ms_per_s", frontier_slope),
            EvidenceFact("observation_value", value),
            EvidenceFact("admitted_observations_total", admitted),
            EvidenceFact("gpu_cost_ms", gpu_cost_ms),
        ),
    )


def test_resolution_policy_uses_canonical_freshness_instead_of_raw_queue_occupancy():
    provider = ResolutionPolicyProvider()
    assert provider.propose(_snapshot(FreshnessMode.HEALTHY)).value == "half"
    assert provider.propose(_snapshot(FreshnessMode.HEALTHY, 1)).value == "half"
    assert provider.propose(_snapshot(FreshnessMode.HEALTHY, 1)).value == "half"
    assert provider.propose(_snapshot(FreshnessMode.CRITICAL)).value == "eighth"


def test_resolution_policy_never_drops_below_explicit_quality_floor():
    provider = ResolutionPolicyProvider(
        supported_tiers=(
            ResolutionTier.QUARTER,
            ResolutionTier.HALF,
            ResolutionTier.NATIVE,
        )
    )

    assert provider.propose(_snapshot(FreshnessMode.CRITICAL)).value == "quarter"


def test_resolution_policy_recovers_fresh_tier_from_completed_gpu_margin_without_admission():
    provider = ResolutionPolicyProvider(
        supported_tiers=(
            ResolutionTier.QUARTER,
            ResolutionTier.HALF,
            ResolutionTier.NATIVE,
        ),
        batch_size=4,
        fresh_upscale_dwell=3,
    )
    assert provider.propose(_snapshot(FreshnessMode.CRITICAL, epoch=1)).value == "quarter"

    for epoch in (2, 3):
        proposal = provider.propose(
            _snapshot(FreshnessMode.HEALTHY, gpu_cost_ms=5.0, epoch=epoch)
        )
        assert proposal.value == "quarter"
    # Issue #1346: recovery jumps straight to the ladder top (native) rather
    # than one rung at a time, once the (now much shorter, default 1)
    # measured-margin dwell completes -- the actual per-transaction tier is
    # still clamped downstream by TierCostEstimator's own measured cost.
    proposal = provider.propose(
        _snapshot(FreshnessMode.HEALTHY, gpu_cost_ms=5.0, epoch=4)
    )
    assert proposal.value == "native"
    assert "measured fresh mapping margin" in proposal.reason
    assert "immediate ceiling recovery" in proposal.reason


def test_resolution_policy_recovers_ceiling_to_native_on_a_single_measured_sample():
    """Issue #1346 regression test.

    With the default (unset) ``fresh_upscale_dwell``, a SINGLE real measured
    ``gpu_cost_ms`` sample inside the margin band is sufficient to recover
    the ceiling all the way to native -- there is no multi-sample climb and
    no dependency on ``observation_value``. This is the mechanism that was
    missing: previously a real mapping ceiling that had been downscaled once
    (e.g. by a cold-start probe blowing its budget) could stay stuck below
    native indefinitely, because recovery required either eight consecutive
    healthy admissions (fragile under real jitter) plus a
    ``observation_value >= 0.75`` bonus condition for native specifically, or
    eight consecutive sustained fresh-margin samples.
    """

    provider = ResolutionPolicyProvider(
        supported_tiers=(
            ResolutionTier.QUARTER,
            ResolutionTier.HALF,
            ResolutionTier.NATIVE,
        ),
        batch_size=4,
    )
    assert provider.propose(_snapshot(FreshnessMode.CRITICAL, epoch=1)).value == "quarter"

    proposal = provider.propose(
        _snapshot(FreshnessMode.HEALTHY, gpu_cost_ms=5.0, value=0.0, epoch=2)
    )
    assert proposal.value == "native"
    assert "immediate ceiling recovery" in proposal.reason


def test_resolution_policy_ceiling_recovery_still_falls_immediately_under_pressure():
    """The asymmetric-safety half of issue #1346: an aggressive recovery to
    native must not weaken the existing immediate-downscale behavior under
    CRITICAL/CATCH_UP freshness pressure."""

    provider = ResolutionPolicyProvider(
        supported_tiers=(
            ResolutionTier.QUARTER,
            ResolutionTier.HALF,
            ResolutionTier.NATIVE,
        ),
        batch_size=4,
    )
    provider.propose(_snapshot(FreshnessMode.CRITICAL, epoch=1))
    assert (
        provider.propose(
            _snapshot(FreshnessMode.HEALTHY, gpu_cost_ms=5.0, value=0.0, epoch=2)
        ).value
        == "native"
    )
    assert provider.propose(_snapshot(FreshnessMode.CRITICAL, epoch=3)).value == "quarter"


def test_resolution_policy_does_not_treat_keyframe_cadence_as_mapping_pressure():
    provider = ResolutionPolicyProvider()

    for admitted in range(8):
        proposal = provider.propose(
            _snapshot(
                FreshnessMode.HEALTHY,
                1,
                admitted=admitted,
                value=1.0,
                service_ratio=0.0,
                frontier_age_ms=550.0,
                frontier_slope=1_000.0,
            )
        )

    assert proposal.value == "half"
    assert "healthy" in proposal.reason


def test_resolution_policy_uses_four_admitted_observation_dwell_before_more_downscale():
    provider = ResolutionPolicyProvider()
    provider.propose(_snapshot(FreshnessMode.CATCH_UP, 1, admitted=0))
    assert provider.propose(_snapshot(FreshnessMode.CATCH_UP, 1, admitted=0)).value == "quarter"

    for admitted in range(1, 4):
        assert provider.propose(
            _snapshot(FreshnessMode.CATCH_UP, 1, admitted=admitted)
        ).value == "quarter"
    assert provider.propose(
        _snapshot(FreshnessMode.CATCH_UP, 1, admitted=4)
    ).value == "quarter"
    assert provider.propose(
        _snapshot(FreshnessMode.CATCH_UP, 1, admitted=4)
    ).value == "eighth"


def test_resolution_policy_upscales_every_eight_healthy_admissions_and_earns_native():
    provider = ResolutionPolicyProvider()
    assert provider.propose(_snapshot(FreshnessMode.CRITICAL, admitted=0)).value == "eighth"

    for admitted in range(1, 8):
        assert provider.propose(_snapshot(FreshnessMode.HEALTHY, admitted=admitted)).value == "eighth"
    assert provider.propose(_snapshot(FreshnessMode.HEALTHY, admitted=8)).value == "quarter"
    for admitted in range(9, 16):
        assert provider.propose(_snapshot(FreshnessMode.HEALTHY, admitted=admitted)).value == "quarter"
    assert provider.propose(_snapshot(FreshnessMode.HEALTHY, admitted=16)).value == "half"
    for admitted in range(17, 24):
        assert provider.propose(
            _snapshot(FreshnessMode.HEALTHY, admitted=admitted, value=0.9)
        ).value == "half"
    assert provider.propose(
        _snapshot(FreshnessMode.HEALTHY, admitted=24, value=0.9)
    ).value == "native"


def test_resolution_policy_earns_native_on_admission_fallback_regardless_of_observation_value():
    """Issue #1346: the fallback (no gpu_cost_ms evidence) admission-count
    climb previously required observation_value >= 0.75 to reach native
    specifically, on top of the eight-admission dwell every other rung
    already needed -- an extra gate that, combined with the measured-margin
    dwell bug covered above, could leave a real mapping ceiling stuck below
    native indefinitely. It is intentionally removed: the same measured,
    budget-fitting TierCostEstimator.select() downstream still clamps the
    tier actually used, so this fallback climb no longer needs its own
    quality-based veto."""

    provider = ResolutionPolicyProvider()
    provider.propose(_snapshot(FreshnessMode.CRITICAL, admitted=0))
    proposal = None
    for admitted in range(1, 25):
        proposal = provider.propose(
            _snapshot(FreshnessMode.HEALTHY, admitted=admitted, value=0.2)
        )

    assert proposal.value == "native"


def test_resolution_policy_never_proposes_an_unmaterialized_tier():
    provider = ResolutionPolicyProvider(supported_tiers=(ResolutionTier.NATIVE,))

    proposal = provider.propose(_snapshot(FreshnessMode.CRITICAL, 1))

    assert proposal.value == ResolutionTier.NATIVE.value
    assert "constrained by materialization capability" in proposal.reason


def test_idle_replay_resolution_uses_gpu_budget_and_hysteresis():
    provider = ResolutionPolicyProvider(
        target_views_per_s=100.0,
        idle_upscale_dwell=3,
    )
    snapshot = _snapshot(FreshnessMode.CRITICAL, admitted=0)
    assert provider.propose(snapshot).value == "eighth"

    for optimizer_step in (1, 2):
        proposal = provider.propose_idle(
            snapshot,
            optimizer_steps_completed=optimizer_step,
            mapping_cost_ms=5.0,
            batch_size=4,
        )
        assert proposal.value == "eighth"
    proposal = provider.propose_idle(
        snapshot,
        optimizer_steps_completed=3,
        mapping_cost_ms=5.0,
        batch_size=4,
    )
    assert proposal.value == "quarter"
    assert "upscale" in proposal.reason

    proposal = provider.propose_idle(
        snapshot,
        optimizer_steps_completed=4,
        mapping_cost_ms=50.0,
        batch_size=4,
    )
    assert proposal.value == "eighth"
    assert "immediate downscale" in proposal.reason


def test_idle_replay_resolution_counts_each_optimizer_epoch_once():
    provider = ResolutionPolicyProvider(
        target_views_per_s=100.0,
        idle_upscale_dwell=2,
    )
    snapshot = _snapshot(FreshnessMode.CRITICAL, admitted=0)
    provider.propose(snapshot)

    first = provider.propose_idle(
        snapshot,
        optimizer_steps_completed=1,
        mapping_cost_ms=5.0,
        batch_size=4,
    )
    duplicate = provider.propose_idle(
        snapshot,
        optimizer_steps_completed=1,
        mapping_cost_ms=5.0,
        batch_size=4,
    )

    assert first.value == duplicate.value == "eighth"


def test_post_refinement_forces_native_replay_resolution():
    provider = ResolutionPolicyProvider(target_views_per_s=100.0)
    snapshot = _snapshot(FreshnessMode.CRITICAL, admitted=0)
    assert provider.propose(snapshot).value == "eighth"

    proposal = provider.propose_idle(
        snapshot,
        optimizer_steps_completed=1,
        mapping_cost_ms=500.0,
        batch_size=4,
        realtime_constraints=False,
    )

    assert proposal.value == "native"
    assert "post-refinement" in proposal.reason


def test_idle_replay_pressure_does_not_downscale_mapping_tier():
    provider = ResolutionPolicyProvider(target_views_per_s=100.0)
    critical = _snapshot(FreshnessMode.CRITICAL, admitted=0)
    assert provider.propose(critical).value == "eighth"

    mapping = None
    for admitted in range(1, 9):
        mapping = provider.propose(
            _snapshot(FreshnessMode.HEALTHY, admitted=admitted, value=1.0)
        )
        replay = provider.propose_idle(
            critical,
            optimizer_steps_completed=admitted,
            mapping_cost_ms=50.0,
            batch_size=4,
        )
        assert replay.value == "eighth"

    assert mapping is not None
    assert mapping.value == "quarter"
    assert "upscale" in mapping.reason
    assert provider.propose(
        _snapshot(FreshnessMode.HEALTHY, admitted=9, value=1.0)
    ).value == "quarter"


def test_mapping_recovery_does_not_change_idle_replay_tier():
    provider = ResolutionPolicyProvider(target_views_per_s=100.0)
    critical = _snapshot(FreshnessMode.CRITICAL, admitted=0)
    provider.propose(critical)
    assert provider.propose_idle(
        critical,
        optimizer_steps_completed=1,
        mapping_cost_ms=50.0,
        batch_size=4,
    ).value == "eighth"

    for admitted in range(1, 17):
        mapping = provider.propose(
            _snapshot(FreshnessMode.HEALTHY, admitted=admitted, value=1.0)
        )
    assert mapping.value == "half"

    replay = provider.propose_idle(
        critical,
        optimizer_steps_completed=2,
        mapping_cost_ms=50.0,
        batch_size=4,
    )
    assert replay.value == "eighth"


def test_post_refinement_native_replay_does_not_lock_mapping_state():
    provider = ResolutionPolicyProvider(target_views_per_s=100.0)
    critical = _snapshot(FreshnessMode.CRITICAL, admitted=0)
    assert provider.propose(critical).value == "eighth"
    assert provider.propose_idle(
        critical,
        optimizer_steps_completed=1,
        mapping_cost_ms=500.0,
        batch_size=4,
        realtime_constraints=False,
    ).value == "native"

    for admitted in range(1, 8):
        mapping = provider.propose(
            _snapshot(FreshnessMode.HEALTHY, admitted=admitted, value=1.0)
        )
        assert mapping.value == "eighth"
    mapping = provider.propose(
        _snapshot(FreshnessMode.HEALTHY, admitted=8, value=1.0)
    )
    assert mapping.value == "quarter"
