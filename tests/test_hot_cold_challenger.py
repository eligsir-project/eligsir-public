"""Unit tests for HotColdChallengerArena (issue #1322)."""

import math

import pytest

from eligsir.mapping.scheduler.hot_cold_challenger import (
    DEFAULT_LEASE_TICKS,
    MODALITY_DEPTH,
    MODALITY_INSERTION,
    MODALITY_RGB,
    REASON_INCUMBENT_PROTECTED,
    REASON_MARGIN_NOT_MET,
    REASON_PROMOTED,
    REASON_SAME_FRAME,
    REASON_UNTRUSTED,
    HotColdChallengerArena,
)
from eligsir.mapping.scheduler.historical_usefulness import NEUTRAL_WEIGHT


# --------------------------------------------------------------------- leases


def test_lease_protects_until_expiry():
    arena = HotColdChallengerArena(lease_ticks=100)
    arena.grant_lease(7, iteration=1000)
    assert arena.is_protected(7, iteration=1099)
    assert not arena.is_protected(7, iteration=1100)


def test_unknown_frame_never_protected():
    arena = HotColdChallengerArena()
    assert not arena.is_protected(999, iteration=0)


def test_grant_lease_refreshes_window():
    arena = HotColdChallengerArena(lease_ticks=50)
    arena.grant_lease(1, iteration=0)
    assert not arena.is_protected(1, iteration=60)
    arena.grant_lease(1, iteration=60)
    assert arena.is_protected(1, iteration=100)


def test_revoke_lease():
    arena = HotColdChallengerArena()
    arena.grant_lease(3, iteration=0)
    assert arena.revoke_lease(3) is True
    assert not arena.is_protected(3, iteration=0)
    assert arena.revoke_lease(3) is False


def test_lease_table_is_bounded():
    arena = HotColdChallengerArena(max_tracked_leases=4)
    for frame_id in range(10):
        arena.grant_lease(frame_id, iteration=0)
    assert len(arena._leases) == 4
    # Oldest-granted leases were evicted first.
    assert 0 not in arena._leases
    assert 9 in arena._leases


def test_default_lease_ticks_matches_constant():
    arena = HotColdChallengerArena()
    assert arena.lease_ticks == DEFAULT_LEASE_TICKS


# ---------------------------------------------------------- modality tracking


def test_modalities_are_independent():
    arena = HotColdChallengerArena()
    for _ in range(5):
        arena.record_modality_observation(1, MODALITY_RGB, 4.0)
        arena.record_modality_observation(1, MODALITY_DEPTH, 0.0)
    rgb_weight = arena.modality_weight(1, MODALITY_RGB)
    depth_weight = arena.modality_weight(1, MODALITY_DEPTH)
    assert rgb_weight > depth_weight
    assert arena.modality_trusted(1, MODALITY_RGB)
    assert arena.modality_trusted(1, MODALITY_DEPTH)


def test_untrusted_before_min_observations():
    arena = HotColdChallengerArena()
    arena.record_modality_observation(1, MODALITY_RGB, 4.0)
    assert not arena.modality_trusted(1, MODALITY_RGB)
    assert arena.modality_weight(1, MODALITY_RGB) == NEUTRAL_WEIGHT


def test_insertion_modality_supported():
    arena = HotColdChallengerArena()
    for _ in range(5):
        arena.record_modality_observation(2, MODALITY_INSERTION, 2.0)
    assert arena.modality_trusted(2, MODALITY_INSERTION)


def test_invalid_modality_rejected():
    arena = HotColdChallengerArena()
    with pytest.raises(ValueError):
        arena.record_modality_observation(1, "audio", 1.0)
    with pytest.raises(ValueError):
        arena.modality_weight(1, "audio")


def test_forget_modality_clears_state():
    arena = HotColdChallengerArena()
    for _ in range(5):
        arena.record_modality_observation(1, MODALITY_RGB, 4.0)
    arena.forget_modality(1, MODALITY_RGB)
    assert not arena.modality_trusted(1, MODALITY_RGB)


def test_forget_modality_all_clears_every_modality():
    arena = HotColdChallengerArena()
    for _ in range(5):
        arena.record_modality_observation(1, MODALITY_RGB, 4.0)
        arena.record_modality_observation(1, MODALITY_DEPTH, 4.0)
    arena.forget_modality(1)
    assert not arena.modality_trusted(1, MODALITY_RGB)
    assert not arena.modality_trusted(1, MODALITY_DEPTH)


# -------------------------------------------------------------- decisions


def _trust(arena, frame_id, modality, raw_weight, n=5):
    for _ in range(n):
        arena.record_modality_observation(frame_id, modality, raw_weight)


def test_promotion_requires_margin_and_trust():
    arena = HotColdChallengerArena(lease_ticks=0, promotion_margin=0.25)
    _trust(arena, 1, MODALITY_RGB, 0.0)  # low loss -> low usefulness weight
    _trust(arena, 2, MODALITY_RGB, 5.0)  # high loss -> high usefulness weight
    decision = arena.evaluate_challenge(1, 2, MODALITY_RGB, iteration=1000)
    assert decision.promote is True
    assert decision.reason == REASON_PROMOTED
    assert decision.challenger_weight > decision.incumbent_weight


def test_no_promotion_when_incumbent_protected():
    arena = HotColdChallengerArena(lease_ticks=1000, promotion_margin=0.0)
    arena.grant_lease(1, iteration=0)
    _trust(arena, 1, MODALITY_RGB, 0.0)
    _trust(arena, 2, MODALITY_RGB, 5.0)
    decision = arena.evaluate_challenge(1, 2, MODALITY_RGB, iteration=1)
    assert decision.promote is False
    assert decision.reason == REASON_INCUMBENT_PROTECTED


def test_no_promotion_without_trusted_evidence():
    arena = HotColdChallengerArena(lease_ticks=0)
    # Only one observation each -- below min_observations, untrusted.
    arena.record_modality_observation(1, MODALITY_RGB, 0.0)
    arena.record_modality_observation(2, MODALITY_RGB, 5.0)
    decision = arena.evaluate_challenge(1, 2, MODALITY_RGB, iteration=0)
    assert decision.promote is False
    assert decision.reason == REASON_UNTRUSTED


def test_no_promotion_when_margin_not_met():
    arena = HotColdChallengerArena(lease_ticks=0, promotion_margin=10.0)
    _trust(arena, 1, MODALITY_RGB, 0.0)
    _trust(arena, 2, MODALITY_RGB, 5.0)
    decision = arena.evaluate_challenge(1, 2, MODALITY_RGB, iteration=0)
    assert decision.promote is False
    assert decision.reason == REASON_MARGIN_NOT_MET


def test_same_frame_never_promotes():
    arena = HotColdChallengerArena(lease_ticks=0)
    _trust(arena, 1, MODALITY_RGB, 5.0)
    decision = arena.evaluate_challenge(1, 1, MODALITY_RGB, iteration=0)
    assert decision.promote is False
    assert decision.reason == REASON_SAME_FRAME


def test_one_noisy_observation_does_not_flip_decision():
    """Hysteresis contract: a single anomalous render must not promote."""

    arena = HotColdChallengerArena(lease_ticks=0, promotion_margin=0.25)
    _trust(arena, 1, MODALITY_RGB, 0.0, n=10)
    # Challenger has only ONE high observation so far -- not yet trusted.
    arena.record_modality_observation(2, MODALITY_RGB, 5.0)
    decision = arena.evaluate_challenge(1, 2, MODALITY_RGB, iteration=0)
    assert decision.promote is False
    assert decision.reason == REASON_UNTRUSTED


def test_negative_promotion_margin_rejected():
    with pytest.raises(ValueError):
        HotColdChallengerArena(promotion_margin=-1.0)


def test_negative_lease_ticks_rejected():
    with pytest.raises(ValueError):
        HotColdChallengerArena(lease_ticks=-1)


# --------------------------------------------------------------- telemetry


def test_snapshot_tracks_promotions_and_reasons():
    arena = HotColdChallengerArena(lease_ticks=0, promotion_margin=0.25)
    _trust(arena, 1, MODALITY_RGB, 0.0)
    _trust(arena, 2, MODALITY_RGB, 5.0)
    arena.evaluate_challenge(1, 2, MODALITY_RGB, iteration=0)
    snap = arena.snapshot()
    assert snap["evaluations"] == 1
    assert snap["promotions"] == 1
    assert snap["demotions"] == 1
    assert snap["decisions_by_reason"][REASON_PROMOTED] == 1
    assert snap["tracked_frames_by_modality"][MODALITY_RGB] == 2


def test_finite_weights_only():
    arena = HotColdChallengerArena(lease_ticks=0)
    _trust(arena, 1, MODALITY_RGB, 0.0)
    _trust(arena, 2, MODALITY_RGB, 5.0)
    decision = arena.evaluate_challenge(1, 2, MODALITY_RGB, iteration=0)
    assert math.isfinite(decision.incumbent_weight)
    assert math.isfinite(decision.challenger_weight)
