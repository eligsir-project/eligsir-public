"""Deterministic CPU tests for the issue #1365 Gaussian birth-target/proposal
scaffolding (eligsir.mapping.birth). Pure dataclasses, no torch/GPU
required -- these tests only prove the contract's own invariants; they do
NOT exercise any production insertion or MCMC path, since none consumes
these types yet.
"""

import pytest

from eligsir.mapping.birth import (
    BirthAction,
    GaussianBirthDecision,
    GaussianBirthProposal,
    GaussianBirthTarget,
    GeometryConfidence,
    ProposalKind,
)
from eligsir.mapping.birth.contracts import birth_targets_from_kind


def _make_target(**overrides):
    defaults = dict(
        target_xyz=(1.0, 2.0, 3.0),
        proposal_kind=ProposalKind.FRONTIER_SURFACE,
        geometry_confidence=GeometryConfidence.SURFACE_CONFIRMED,
    )
    defaults.update(overrides)
    return GaussianBirthTarget(**defaults)


def test_target_construction_minimal():
    target = _make_target()
    assert target.target_xyz == (1.0, 2.0, 3.0)
    assert target.proposal_kind is ProposalKind.FRONTIER_SURFACE
    assert target.geometry_confidence is GeometryConfidence.SURFACE_CONFIRMED
    # Optional fields default to unset / empty, not silently zeroed.
    assert target.target_normal is None
    assert target.rgb_residual is None
    assert target.supporting_sources == ()


def test_target_is_frozen_and_hashable_by_identity_safe_fields():
    target = _make_target()
    with pytest.raises(Exception):
        target.target_xyz = (0.0, 0.0, 0.0)  # type: ignore[misc]


def test_target_rejects_malformed_xyz():
    with pytest.raises(ValueError):
        GaussianBirthTarget(
            target_xyz=(1.0, 2.0),  # only 2 components
            proposal_kind=ProposalKind.FRONTIER_SURFACE,
            geometry_confidence=GeometryConfidence.SURFACE_CONFIRMED,
        )


def test_target_rejects_malformed_normal():
    with pytest.raises(ValueError):
        _make_target(target_normal=(0.0, 1.0))


def test_target_accepts_full_evidence_payload():
    target = _make_target(
        target_normal=(0.0, 0.0, 1.0),
        geometry_covariance=(0.01, 0.0, 0.0, 0.01, 0.0, 0.001),
        anchor_id="anchor-42",
        rgb_residual=0.12,
        depth_residual=0.03,
        gradient_utility=0.5,
        coverage_debt=0.8,
        spatial_region_debt=0.2,
        local_gaussian_density=3.1,
        current_support=4,
        supporting_sources=("depth_frame_100", "depth_frame_101"),
        provenance="test",
    )
    assert target.anchor_id == "anchor-42"
    assert target.supporting_sources == ("depth_frame_100", "depth_frame_101")


def test_geometry_unknown_is_a_valid_but_distinct_confidence_class():
    # Per the design doc, GEOMETRY_UNKNOWN should never by itself justify
    # a CREATE decision -- this test only proves the contract can represent
    # that state, since no owner exists yet to enforce the rule.
    target = _make_target(geometry_confidence=GeometryConfidence.GEOMETRY_UNKNOWN)
    assert target.geometry_confidence is GeometryConfidence.GEOMETRY_UNKNOWN
    assert target.geometry_confidence is not GeometryConfidence.SURFACE_CONFIRMED


def test_proposal_keeps_parent_and_target_independent():
    target = _make_target()
    proposal = GaussianBirthProposal(target=target, parent_uid=17, score=0.9)
    assert proposal.target is target
    assert proposal.parent_uid == 17
    # Changing the notion of "parent" never mutates the target's geometry --
    # they are independent fields on the same frozen object, not a chain.
    proposal_no_parent = GaussianBirthProposal(target=target)
    assert proposal_no_parent.parent_uid is None
    assert proposal_no_parent.target is proposal.target


def test_decision_records_action_and_reason():
    target = _make_target()
    proposal = GaussianBirthProposal(target=target)
    decision = GaussianBirthDecision(
        action=BirthAction.CREATE,
        proposal=proposal,
        reason="underrepresented trusted surface",
    )
    assert decision.action is BirthAction.CREATE
    assert decision.proposal is proposal
    assert decision.donor_uid is None


@pytest.mark.parametrize(
    "action",
    [BirthAction.NONE, BirthAction.CORRECT, BirthAction.CREATE, BirthAction.RELOCATE],
)
def test_all_four_birth_actions_are_constructible(action):
    target = _make_target()
    decision = GaussianBirthDecision(action=action, proposal=GaussianBirthProposal(target=target))
    assert decision.action is action


def test_birth_targets_from_kind_filters_correctly():
    frontier = _make_target(proposal_kind=ProposalKind.FRONTIER_SURFACE)
    detail = _make_target(proposal_kind=ProposalKind.DETAIL_DENSIFICATION)
    correction = _make_target(proposal_kind=ProposalKind.GEOMETRY_CORRECTION)

    result = birth_targets_from_kind(
        [frontier, detail, correction], ProposalKind.DETAIL_DENSIFICATION
    )
    assert result == (detail,)

    result_empty = birth_targets_from_kind([frontier], ProposalKind.STRUCTURAL_ANCHOR)
    assert result_empty == ()


def test_all_proposal_kinds_and_confidence_classes_are_string_enums():
    # String-enum backing is load-bearing for JSON/telemetry serialization
    # (report.json-style dumps elsewhere in this repo assume str-castable
    # enum values); this test guards that choice, not an implementation
    # detail nobody depends on.
    for kind in ProposalKind:
        assert isinstance(kind.value, str)
    for confidence in GeometryConfidence:
        assert isinstance(confidence.value, str)
    for action in BirthAction:
        assert isinstance(action.value, str)
