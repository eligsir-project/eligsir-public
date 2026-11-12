"""Issue #837 criterion 8, ``#832`` half: the realized RGB-D pairing /
timestamp-correction / transport-decode owner feeds the ingress-ownership
contract.

Mirrors ``tests/test_issue837_criterion8_overflow_owner.py`` (the #836 half)
in structure. The historical stacked (duplicate-owner) topology this module
used to be able to resolve was deleted by #832's own hard-cut amendment
(#1073, 2026-08-23) -- ``human_launch_scripts/05_run_offline_rosbag.sh`` and
``05c_run_offline_rosbag_new_syncer.sh`` now refuse
``RGBD_PAIRING_OWNER=legacy_stacked`` at startup, so this module can never
observe it from a real run again. What remains here documents the SAFE
failure mode instead: an unrecognized/removed owner token (including the
literal string ``"legacy_stacked"``) resolves to an EMPTY declaration list
(absent, not fabricated), never to the old duplicate-owner topology.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from eligsir.io.ingress_ownership import (
    OwnerDeclaration,
    OwnershipRole,
    Responsibility,
    TopologyMode,
    ingress_ownership_from_report_dict,
    resolve_ingress_ownership,
)
from eligsir.io.policies.rgbd_pairing_ownership import (
    ENV_BAKED_POSES,
    ENV_LEGACY_SYNCER,
    ENV_PAIRING_OWNER,
    MODE_CPP_SYNC,
    MODE_LEGACY_SYNCER_NO_CPP_NODE,
    MODE_NOT_APPLICABLE,
    OWNER_CPP_SYNC_NODE,
    OWNER_DECOMPRESS_RGBD,
    resolve_rgbd_pairing_owner_declarations,
    resolve_rgbd_pairing_owner_mode,
    rgbd_pairing_owner_declarations_from_env,
    rgbd_pairing_owner_mode_from_env,
)

PAIRING_RESPONSIBILITIES = (
    Responsibility.RGBD_TEMPORAL_PAIRING,
    Responsibility.RGBD_TIMESTAMP_CORRECTION,
)


# --------------------------------------------------------------------------
# Mode derivation
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "baked,legacy_syncer,owner,expected",
    [
        (False, False, "cpp_sync", MODE_CPP_SYNC),
        # "legacy_stacked" is the removed token: ignored, not honored -- the
        # raw baked/legacy_syncer flags alone determine the mode now.
        (False, False, "legacy_stacked", MODE_CPP_SYNC),
        (False, True, "cpp_sync", MODE_LEGACY_SYNCER_NO_CPP_NODE),
        (False, True, "legacy_stacked", MODE_LEGACY_SYNCER_NO_CPP_NODE),
        (True, False, "cpp_sync", MODE_NOT_APPLICABLE),
        (True, True, "legacy_stacked", MODE_NOT_APPLICABLE),
    ],
)
def test_mode_derivation_matches_launcher_branch_order(baked, legacy_syncer, owner, expected):
    assert (
        resolve_rgbd_pairing_owner_mode(
            use_baked_poses=baked,
            use_legacy_syncer=legacy_syncer,
            rgbd_pairing_owner=owner,
        )
        == expected
    )


def test_absent_env_resolves_to_none_not_a_fabricated_owner():
    """Every non-bag run (TUM/ScanNet/generic replay/offline train.py) has no
    launcher-exported pairing state at all. It must leave the responsibilities
    unresolved, never invent one."""
    assert rgbd_pairing_owner_mode_from_env({}) is None
    assert rgbd_pairing_owner_declarations_from_env({}) == []


def test_env_resolution_ignores_the_removed_legacy_stacked_token():
    """A stale caller (or an old, unrebuilt image) that still sets
    RGBD_PAIRING_OWNER=legacy_stacked in the environment must resolve safely
    to the single canonical owner, never to the deleted duplicate-owner
    topology -- the launch scripts themselves refuse to even start with that
    value, but this is the defense-in-depth check on the Python side."""
    env = {
        ENV_BAKED_POSES: "0",
        ENV_LEGACY_SYNCER: "0",
        ENV_PAIRING_OWNER: "legacy_stacked",
    }
    assert rgbd_pairing_owner_mode_from_env(env) == MODE_CPP_SYNC


def test_baked_poses_env_yields_no_declarations():
    env = {ENV_BAKED_POSES: "1", ENV_LEGACY_SYNCER: "0", ENV_PAIRING_OWNER: "cpp_sync"}
    assert rgbd_pairing_owner_mode_from_env(env) == MODE_NOT_APPLICABLE
    assert rgbd_pairing_owner_declarations_from_env(env) == []


def test_unknown_owner_token_is_not_guessed_into_a_topology():
    assert resolve_rgbd_pairing_owner_declarations("something_new") == []


# --------------------------------------------------------------------------
# Declarations -> resolver verdicts
# --------------------------------------------------------------------------


def test_cpp_sync_resolves_one_owner_per_responsibility_and_is_valid():
    """NEGATIVE CONTROL for the stacked test below: the canonical topology
    must NOT trip the duplicate guard, otherwise the positive test proves
    nothing."""
    result = resolve_ingress_ownership(
        resolve_rgbd_pairing_owner_declarations(MODE_CPP_SYNC)
    )
    assert result.valid is True
    assert result.failure_codes == []
    for resp in PAIRING_RESPONSIBILITIES:
        res = result.responsibilities[resp]
        assert res.duplicate is False
        assert res.resolved_owner is not None
        assert res.resolved_owner.owner == OWNER_CPP_SYNC_NODE
    decode = result.responsibilities[Responsibility.TRANSPORT_DECODE]
    assert decode.resolved_owner is not None
    assert decode.resolved_owner.owner == OWNER_DECOMPRESS_RGBD
    assert decode.resolved_owner.mode == "decode_only"
    assert decode.resolved_owner.attributes["decompress_sync_param"] is False


def test_legacy_stacked_token_yields_no_declarations_not_the_old_duplicate_topology():
    """The removed token must never resurrect the old duplicate-owner
    topology -- it degrades to "absent", the same as any other unrecognized
    string (see test_unknown_owner_token_is_not_guessed_into_a_topology)."""
    assert resolve_rgbd_pairing_owner_declarations("legacy_stacked") == []


def test_legacy_syncer_no_cpp_node_is_single_owner_and_valid():
    """--legacy-syncer runs no C++ node at all, so decompress_rgbd really is
    the sole owner: single-owner, valid, but self-describing as non-canonical
    via the mode strings (it is not the same topology as cpp_sync)."""
    result = resolve_ingress_ownership(
        resolve_rgbd_pairing_owner_declarations(MODE_LEGACY_SYNCER_NO_CPP_NODE)
    )
    assert result.valid is True
    assert result.failure_codes == []
    for resp in PAIRING_RESPONSIBILITIES:
        res = result.responsibilities[resp]
        assert res.duplicate is False
        assert res.resolved_owner is not None
        assert res.resolved_owner.owner == OWNER_DECOMPRESS_RGBD
        assert res.resolved_owner.attributes["cpp_sync_node_running"] is False


def test_pairing_declarations_do_not_collide_with_the_other_criterion8_halves():
    """Sanity: adding the pairing declarations alongside a source_sampling and
    an overflow_drop_policy declaration must not manufacture a duplicate on
    any responsibility."""
    others = [
        OwnerDeclaration(
            Responsibility.SOURCE_SAMPLING,
            owner="native_source_stride",
            role=OwnershipRole.MUTATOR,
        ),
        OwnerDeclaration(
            Responsibility.OVERFLOW_DROP_POLICY,
            owner="ros2_live_ingress_overflow_policy",
            role=OwnershipRole.DECISION_OWNER,
        ),
    ]
    result = resolve_ingress_ownership(
        others + resolve_rgbd_pairing_owner_declarations(MODE_CPP_SYNC)
    )
    assert result.valid is True
    assert result.failure_codes == []


def test_report_round_trip_preserves_pairing_owner_and_mode():
    """#837 test 8 for this half."""
    result = resolve_ingress_ownership(
        resolve_rgbd_pairing_owner_declarations(MODE_CPP_SYNC)
    )
    block = json.loads(json.dumps(result.to_report_dict()))
    restored = ingress_ownership_from_report_dict(block)
    assert restored.topology is TopologyMode.CANONICAL
    for resp in PAIRING_RESPONSIBILITIES:
        assert restored.responsibilities[resp].resolved_owner.owner == OWNER_CPP_SYNC_NODE
        assert restored.responsibilities[resp].resolved_owner.mode == "software_corrected"


def test_train_streaming_call_site_is_wired():
    """AST-free, import-free check that the report block actually calls this
    module -- train_streaming.py cannot be imported without torch, and issue
    #1112 established reading the call site off disk as the accepted pattern.
    """
    src = (Path(__file__).parent.parent / "train_streaming.py").read_text()
    assert "rgbd_pairing_owner_declarations_from_env" in src, (
        "train_streaming.py's _ingress_ownership_report_block no longer feeds "
        "the realized RGB-D pairing owner into the ingress-ownership contract"
    )
