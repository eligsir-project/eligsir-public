"""Issue #837: ingress ownership contract tests.

Implements the 8 tests named in the issue body, in order. Scope is
acceptance criteria 1-7 only (criterion 8 -- #832/#833/#834/#836 feeding
their realized owner state into this contract -- is explicitly out of
scope and gated on those sibling PRs; see
``eligsir/io/ingress_ownership.py``'s module docstring).

This module is behavior-neutral (report/provenance only), so these tests
exercise the pure resolver directly -- no training loop, no GPU, no ROS.
"""

from __future__ import annotations

import pytest

from eligsir.io.ingress_ownership import (
    CONTRACT_VERSION,
    DUPLICATE_INGRESS_AUTHORITY,
    LIVE_ORBBEC_GUARANTEES,
    UNVALIDATED_ROSBAG_GUARANTEES,
    OwnerDeclaration,
    OwnerHandoff,
    OwnershipRole,
    Responsibility,
    TopologyMode,
    clear_registry,
    ingress_ownership_from_report_dict,
    register_ingress_owner,
    registered_declarations,
    resolve_ingress_ownership,
)


@pytest.fixture(autouse=True)
def _clean_registry():
    clear_registry()
    yield
    clear_registry()


# --------------------------------------------------------------------------
# Test 1: canonical bag topology resolves one sync owner
# --------------------------------------------------------------------------

def test_01_canonical_bag_topology_resolves_one_sync_owner():
    """Path A (unvalidated rosbag, plan doc section 6.1): the C++ sync
    component is the sole rgbd_temporal_pairing + rgbd_timestamp_correction
    owner, with an ORB-SLAM3 splitter as a compatibility adapter (not a
    second pairing owner -- see test 6)."""
    declarations = [
        OwnerDeclaration(
            Responsibility.RGBD_TEMPORAL_PAIRING,
            owner="eligsir_rgbd_sync",
            role=OwnershipRole.MUTATOR,
            mode="software_corrected",
        ),
        OwnerDeclaration(
            Responsibility.RGBD_TIMESTAMP_CORRECTION,
            owner="eligsir_rgbd_sync",
            role=OwnershipRole.MUTATOR,
            mode="software_corrected",
        ),
        OwnerDeclaration(
            Responsibility.TRANSPORT_DECODE,
            owner="decode_only_adapter",
            role=OwnershipRole.MUTATOR,
        ),
        OwnerDeclaration(
            Responsibility.POSE_ASSOCIATION,
            owner="orbslam3_pose_adapter",
            role=OwnershipRole.MUTATOR,
        ),
        OwnerDeclaration(
            Responsibility.SOURCE_SAMPLING,
            owner="ros2_source",
            role=OwnershipRole.MUTATOR,
            attributes={"effective_stride": 1},
        ),
        OwnerDeclaration(
            Responsibility.OVERFLOW_DROP_POLICY,
            owner="bag_replay_policy",
            role=OwnershipRole.DECISION_OWNER,
            mode="no_silent_loss",
        ),
        OwnerDeclaration(
            Responsibility.COORDINATE_INTEGRITY,
            owner="integrity_observer",
            role=OwnershipRole.ASSERTOR,
        ),
        OwnerDeclaration(
            Responsibility.CANDIDATE_PREFILTER,
            owner="candidate_prefilter",
            role=OwnershipRole.DECISION_OWNER,
        ),
        OwnerDeclaration(
            Responsibility.KEYFRAME_COMMIT,
            owner="mapping_scheduler",
            role=OwnershipRole.DECISION_OWNER,
        ),
    ]

    result = resolve_ingress_ownership(
        declarations, guarantees=UNVALIDATED_ROSBAG_GUARANTEES
    )

    assert result.valid is True
    assert result.failure_codes == []
    pairing = result.responsibilities[Responsibility.RGBD_TEMPORAL_PAIRING]
    assert pairing.resolved_owner is not None
    assert pairing.resolved_owner.owner == "eligsir_rgbd_sync"
    assert pairing.duplicate is False
    assert result.missing_required_owners == []


# --------------------------------------------------------------------------
# Test 2: canonical live Orbbec topology resolves driver sync + registration
# --------------------------------------------------------------------------

def test_02_canonical_live_orbbec_resolves_driver_sync_and_registration():
    declarations = [
        OwnerDeclaration(
            Responsibility.RGBD_TEMPORAL_PAIRING,
            owner="orbbec_driver",
            role=OwnershipRole.MUTATOR,
            mode="hardware_validated",
        ),
        OwnerDeclaration(
            Responsibility.RGBD_REGISTRATION,
            owner="orbbec_driver",
            role=OwnershipRole.MUTATOR,
            mode="registered_to_color",
        ),
        OwnerDeclaration(
            Responsibility.POSE_ASSOCIATION,
            owner="ros2_live_pose_adapter",
            role=OwnershipRole.MUTATOR,
        ),
        OwnerDeclaration(
            Responsibility.SOURCE_SAMPLING,
            owner="ros2_source",
            role=OwnershipRole.MUTATOR,
            attributes={"effective_stride": 2},
        ),
        OwnerDeclaration(
            Responsibility.OVERFLOW_DROP_POLICY,
            owner="realtime_drop_policy",
            role=OwnershipRole.DECISION_OWNER,
            mode="real_time_drop",
        ),
        OwnerDeclaration(
            Responsibility.COORDINATE_INTEGRITY,
            owner="integrity_observer",
            role=OwnershipRole.ASSERTOR,
        ),
        OwnerDeclaration(
            Responsibility.CANDIDATE_PREFILTER,
            owner="candidate_prefilter",
            role=OwnershipRole.DECISION_OWNER,
        ),
        OwnerDeclaration(
            Responsibility.KEYFRAME_COMMIT,
            owner="mapping_scheduler",
            role=OwnershipRole.DECISION_OWNER,
        ),
        # Assert-only verifier: does not claim mutating authority.
        OwnerDeclaration(
            Responsibility.RGBD_TEMPORAL_PAIRING,
            owner="assert_only_source_verifier",
            role=OwnershipRole.ASSERTOR,
        ),
    ]

    result = resolve_ingress_ownership(
        declarations, guarantees=LIVE_ORBBEC_GUARANTEES
    )

    assert result.valid is True
    assert result.failure_codes == []
    sync = result.responsibilities[Responsibility.RGBD_TEMPORAL_PAIRING]
    reg = result.responsibilities[Responsibility.RGBD_REGISTRATION]
    assert sync.resolved_owner.owner == "orbbec_driver"
    assert reg.resolved_owner.owner == "orbbec_driver"
    assert reg.resolved_owner.mode == "registered_to_color"
    # The assertor is visible but is not the resolved mutating owner.
    assert len(sync.assertor_declarations) == 1
    assert sync.assertor_declarations[0].owner == "assert_only_source_verifier"
    assert result.missing_required_owners == []


# --------------------------------------------------------------------------
# Test 3: duplicate sync mutators -> structural failure/verdict
# --------------------------------------------------------------------------

def test_03_duplicate_sync_mutators_trigger_structural_failure():
    """decompress-sync + C++ sync correction both claiming
    rgbd_temporal_pairing -- the exact canonical example from the plan
    doc's "Examples the guard should catch" list."""
    declarations = [
        OwnerDeclaration(
            Responsibility.RGBD_TEMPORAL_PAIRING,
            owner="decompress_rgbd_sync_true",
            role=OwnershipRole.MUTATOR,
        ),
        OwnerDeclaration(
            Responsibility.RGBD_TEMPORAL_PAIRING,
            owner="eligsir_rgbd_sync",
            role=OwnershipRole.MUTATOR,
        ),
    ]

    result = resolve_ingress_ownership(declarations)

    assert result.valid is False
    assert DUPLICATE_INGRESS_AUTHORITY in result.failure_codes
    pairing = result.responsibilities[Responsibility.RGBD_TEMPORAL_PAIRING]
    assert pairing.duplicate is True
    assert pairing.failure_code == DUPLICATE_INGRESS_AUTHORITY
    assert pairing.resolved_owner is None
    assert pairing.topology is TopologyMode.CANONICAL
    # Behavior-neutral: this is a report-only verdict; no exception raised
    # and no training-behavior signal is embedded in the return value.


# --------------------------------------------------------------------------
# Test 4: duplicate sampling mutators -> structural failure/verdict
# --------------------------------------------------------------------------

def test_04_duplicate_sampling_mutators_trigger_structural_failure():
    """native source stride and generic StridedFrameSource stride both
    independently applying sampling."""
    declarations = [
        OwnerDeclaration(
            Responsibility.SOURCE_SAMPLING,
            owner="native_source_stride",
            role=OwnershipRole.MUTATOR,
            attributes={"stride": 2},
        ),
        OwnerDeclaration(
            Responsibility.SOURCE_SAMPLING,
            owner="generic_strided_frame_source",
            role=OwnershipRole.MUTATOR,
            attributes={"stride": 3},
        ),
    ]

    result = resolve_ingress_ownership(declarations)

    assert result.valid is False
    assert result.failure_codes == [DUPLICATE_INGRESS_AUTHORITY]
    sampling = result.responsibilities[Responsibility.SOURCE_SAMPLING]
    assert sampling.duplicate is True
    assert {d.owner for d in sampling.mutating_declarations} == {
        "native_source_stride",
        "generic_strided_frame_source",
    }


# --------------------------------------------------------------------------
# Test 5: assertor + one mutator -> valid
# --------------------------------------------------------------------------

def test_05_assertor_plus_one_mutator_is_valid():
    declarations = [
        OwnerDeclaration(
            Responsibility.OVERFLOW_DROP_POLICY,
            owner="bag_replay_policy",
            role=OwnershipRole.DECISION_OWNER,
            mode="no_silent_loss",
        ),
        OwnerDeclaration(
            Responsibility.OVERFLOW_DROP_POLICY,
            owner="queue_pressure_monitor",
            role=OwnershipRole.ASSERTOR,
        ),
        OwnerDeclaration(
            Responsibility.OVERFLOW_DROP_POLICY,
            owner="report_writer",
            role=OwnershipRole.CONSUMER,
        ),
    ]

    result = resolve_ingress_ownership(declarations)

    assert result.valid is True
    assert result.failure_codes == []
    overflow = result.responsibilities[Responsibility.OVERFLOW_DROP_POLICY]
    assert overflow.duplicate is False
    assert overflow.resolved_owner.owner == "bag_replay_policy"
    assert len(overflow.assertor_declarations) == 1
    assert len(overflow.consumer_declarations) == 1


# --------------------------------------------------------------------------
# Test 6: compatibility splitter does not count as pairing mutator
# --------------------------------------------------------------------------

def test_06_compatibility_splitter_is_not_a_pairing_mutator():
    declarations = [
        OwnerDeclaration(
            Responsibility.RGBD_TEMPORAL_PAIRING,
            owner="eligsir_rgbd_sync",
            role=OwnershipRole.MUTATOR,
        ),
        OwnerDeclaration(
            Responsibility.RGBD_TEMPORAL_PAIRING,
            owner="orbslam3_splitter",
            role=OwnershipRole.COMPATIBILITY_ADAPTER,
        ),
    ]

    result = resolve_ingress_ownership(declarations)

    assert result.valid is True
    assert result.failure_codes == []
    pairing = result.responsibilities[Responsibility.RGBD_TEMPORAL_PAIRING]
    assert pairing.duplicate is False
    assert pairing.resolved_owner.owner == "eligsir_rgbd_sync"
    assert len(pairing.mutating_declarations) == 1
    assert len(pairing.compatibility_adapter_declarations) == 1
    assert pairing.compatibility_adapter_declarations[0].owner == "orbslam3_splitter"


# --------------------------------------------------------------------------
# Test 7: explicit legacy topology can be represented without being
# mistaken for canonical
# --------------------------------------------------------------------------

def test_07_explicit_legacy_topology_is_labeled_not_canonical():
    declarations = [
        OwnerDeclaration(
            Responsibility.SOURCE_SAMPLING,
            owner="native_source_stride",
            role=OwnershipRole.MUTATOR,
            attributes={"stride": 2},
        ),
        OwnerDeclaration(
            Responsibility.SOURCE_SAMPLING,
            owner="generic_strided_frame_source",
            role=OwnershipRole.MUTATOR,
            attributes={"stride": 3},
        ),
    ]

    legacy_result = resolve_ingress_ownership(declarations, legacy_topology=True)
    canonical_result = resolve_ingress_ownership(declarations, legacy_topology=False)

    # Legacy: duplication is representable and does NOT produce the
    # canonical structural-failure code.
    assert legacy_result.topology is TopologyMode.LEGACY
    assert legacy_result.valid is True
    assert legacy_result.failure_codes == []
    legacy_sampling = legacy_result.responsibilities[Responsibility.SOURCE_SAMPLING]
    assert legacy_sampling.duplicate is True
    assert legacy_sampling.topology is TopologyMode.LEGACY

    # Canonical: the identical declarations DO fail structurally.
    assert canonical_result.topology is TopologyMode.CANONICAL
    assert canonical_result.valid is False
    assert DUPLICATE_INGRESS_AUTHORITY in canonical_result.failure_codes

    # The two verdicts must never collapse to the same report shape.
    assert legacy_result.to_report_dict()["topology"] != canonical_result.to_report_dict()["topology"]
    assert legacy_result.to_report_dict()["valid"] != canonical_result.to_report_dict()["valid"]


# --------------------------------------------------------------------------
# Test 8: report round-trip preserves owner/mode/version
# --------------------------------------------------------------------------

def test_08_report_round_trip_preserves_owner_mode_version():
    declarations = [
        OwnerDeclaration(
            Responsibility.RGBD_REGISTRATION,
            owner="orbbec_driver",
            role=OwnershipRole.MUTATOR,
            mode="registered_to_color",
            attributes={"stream": "COLOR"},
        ),
        OwnerDeclaration(
            Responsibility.RGBD_TEMPORAL_PAIRING,
            owner="assert_only_source_verifier",
            role=OwnershipRole.ASSERTOR,
        ),
    ]
    handoffs = [
        OwnerHandoff(
            responsibility=Responsibility.RGBD_TEMPORAL_PAIRING,
            from_owner="orbbec_driver",
            to_owner="eligsir_rgbd_sync",
            reason="contract violation: pair delta exceeded tolerance",
            at="iter=12000",
        )
    ]

    result = resolve_ingress_ownership(
        declarations, guarantees=LIVE_ORBBEC_GUARANTEES, handoffs=handoffs
    )
    report_block = result.to_report_dict()

    assert report_block["version"] == CONTRACT_VERSION

    restored = ingress_ownership_from_report_dict(report_block)

    assert restored.version == result.version
    assert restored.topology == result.topology
    assert restored.guarantee_profile == result.guarantee_profile

    orig_reg = result.responsibilities[Responsibility.RGBD_REGISTRATION]
    restored_reg = restored.responsibilities[Responsibility.RGBD_REGISTRATION]
    assert restored_reg.resolved_owner.owner == orig_reg.resolved_owner.owner == "orbbec_driver"
    assert restored_reg.resolved_owner.mode == orig_reg.resolved_owner.mode == "registered_to_color"
    assert restored_reg.resolved_owner.role == orig_reg.resolved_owner.role == OwnershipRole.MUTATOR

    assert len(restored.handoffs) == 1
    assert restored.handoffs[0].from_owner == "orbbec_driver"
    assert restored.handoffs[0].to_owner == "eligsir_rgbd_sync"
    assert restored.handoffs[0].reason == handoffs[0].reason
    assert restored.handoffs[0].at == "iter=12000"

    # Full dict-level round trip is exact.
    assert ingress_ownership_from_report_dict(report_block).to_report_dict() == report_block


# --------------------------------------------------------------------------
# Additional coverage: owner handoff excludes the retired owner from the
# duplicate-authority count (issue #837 "Owner handoff": "Never run both
# corrections concurrently").
# --------------------------------------------------------------------------

def test_owner_handoff_prevents_false_duplicate_after_transfer():
    declarations = [
        OwnerDeclaration(
            Responsibility.RGBD_TEMPORAL_PAIRING,
            owner="orbbec_driver",
            role=OwnershipRole.MUTATOR,
            mode="hardware_validated",
        ),
        OwnerDeclaration(
            Responsibility.RGBD_TEMPORAL_PAIRING,
            owner="eligsir_rgbd_sync",
            role=OwnershipRole.MUTATOR,
            mode="software_corrected",
        ),
    ]
    handoffs = [
        OwnerHandoff(
            responsibility=Responsibility.RGBD_TEMPORAL_PAIRING,
            from_owner="orbbec_driver",
            to_owner="eligsir_rgbd_sync",
            reason="hardware contract violated: pair delta 120ms > 30ms tolerance",
        )
    ]

    # Without the handoff on record, this is a duplicate-authority failure.
    result_no_handoff = resolve_ingress_ownership(declarations)
    assert result_no_handoff.valid is False

    # With the handoff, the old owner is retired and exactly one owner
    # (the new one) is resolved.
    result_with_handoff = resolve_ingress_ownership(declarations, handoffs=handoffs)
    assert result_with_handoff.valid is True
    pairing = result_with_handoff.responsibilities[Responsibility.RGBD_TEMPORAL_PAIRING]
    assert pairing.duplicate is False
    assert pairing.resolved_owner.owner == "eligsir_rgbd_sync"
    assert len(result_with_handoff.handoffs) == 1


# --------------------------------------------------------------------------
# Additional coverage: declarative registration API for future sibling
# components (criterion 8 scaffolding -- not wired to real owner state by
# this PR, but the registry itself must work).
# --------------------------------------------------------------------------

def test_declarative_registration_api_is_available_for_future_owners():
    assert registered_declarations() == []

    register_ingress_owner(
        OwnerDeclaration(
            Responsibility.CANDIDATE_PREFILTER,
            owner="candidate_prefilter_v2",
            role=OwnershipRole.DECISION_OWNER,
        )
    )

    declared = registered_declarations()
    assert len(declared) == 1
    assert declared[0].owner == "candidate_prefilter_v2"

    result = resolve_ingress_ownership(registered_declarations())
    prefilter = result.responsibilities[Responsibility.CANDIDATE_PREFILTER]
    assert prefilter.resolved_owner.owner == "candidate_prefilter_v2"


# --------------------------------------------------------------------------
# Additional coverage: source-guarantee-driven resolution actually restricts
# what's "required" per source mode, not a fixed set for every source.
# --------------------------------------------------------------------------

def test_source_guarantees_select_required_responsibilities_not_all():
    # Live Orbbec doesn't require rgbd_timestamp_correction as a separately
    # resolved responsibility (the driver's pairing already covers it) --
    # unlike the unvalidated rosbag profile.
    assert (
        Responsibility.RGBD_TIMESTAMP_CORRECTION
        not in LIVE_ORBBEC_GUARANTEES.required_responsibilities
    )
    assert (
        Responsibility.RGBD_TIMESTAMP_CORRECTION
        in UNVALIDATED_ROSBAG_GUARANTEES.required_responsibilities
    )

    # An empty declaration set against the rosbag profile should report
    # every required responsibility as missing (nothing resolved it).
    result = resolve_ingress_ownership([], guarantees=UNVALIDATED_ROSBAG_GUARANTEES)
    assert set(result.missing_required_owners) == {
        r.value for r in UNVALIDATED_ROSBAG_GUARANTEES.required_responsibilities
    }
