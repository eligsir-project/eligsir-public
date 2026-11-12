"""Issue #837 criterion 8, ``#832`` half: feed the REALIZED RGB-D
pairing/timestamp-correction/transport-decode ownership of a rosbag run into
the ``eligsir.io.ingress_ownership`` contract.

This is the third consumer of PR #856's ``register_ingress_owner`` /
``resolve_ingress_ownership`` machinery, after #834's ``source_sampling``
(PR #856) and #836's ``overflow_drop_policy`` (PR #857). It changes neither
the resolver nor any pairing component; it only reshapes the pairing
topology a bag run *already* launched with into
:class:`~eligsir.io.ingress_ownership.OwnerDeclaration` objects.

Why this module exists
----------------------
Before this, ``report.json["ingress_ownership"]`` could never name an RGB-D
pairing owner at all, so #832's whole subject -- "exactly one component may
own RGB-D temporal pairing, timestamp correction and pair-level
monotonicity for one source path" -- was unrepresentable in the very
contract #837 built to represent it. The realized topology was recorded only
in ``$MODEL_PATH/rgbd_pairing_telemetry.json``, a launch-script-local file
with its own vocabulary, written *after* the run and never joined to the
report.

The consequence that matters: the historical stacked topology
(``RGBD_PAIRING_OWNER=legacy_stacked`` -- ``decompress_rgbd sync:=true``
pairing and re-stamping, and then the C++ ``sync_node`` pairing and
re-stamping again) was *invisible* to the ``DUPLICATE_INGRESS_AUTHORITY``
guard, which is the single configuration that guard was written to catch.
With this module wired in, that mode resolves to two active ``MUTATOR``
declarations on both ``rgbd_temporal_pairing`` and
``rgbd_timestamp_correction``, so the resolver returns ``valid=False`` with
``DUPLICATE_INGRESS_AUTHORITY`` and ``write_post_training_report`` prints
its existing loud ``[report] WARNING: ingress_ownership: structural verdict
INVALID`` line. That realizes #832's own "Hard-cut amendment" ("duplicate
mutating pairing/correction authority is an invalid configuration and must
fail closed") as a machine-readable verdict, in the CLAUDE.md
``eval_valid``/``map_reinit_invalidated`` sense of fail-closed -- a verdict a
human or a downstream validity gate acts on, never an in-process abort.

Hard-cut amendment applied (2026-08-23)
----------------------------------------
The ``legacy_stacked`` mode described below is DELETED, per #832's own
"Hard-cut amendment" comment: ``human_launch_scripts/05_run_offline_rosbag.sh``
and ``05c_run_offline_rosbag_new_syncer.sh`` now refuse
``RGBD_PAIRING_OWNER=legacy_stacked`` outright at launch (fail closed at
startup, not merely a self-reporting-invalid report block), so this module
can never observe that mode from a real run again.
``MODE_LEGACY_STACKED``/the "two MUTATOR declarations" branch this
docstring used to describe are removed rather than kept as dead code --
``resolve_rgbd_pairing_owner_declarations`` now only ever resolves
``MODE_CPP_SYNC`` or ``MODE_LEGACY_SYNCER_NO_CPP_NODE`` (or returns ``[]``
for anything else, same "absent beats fabricated" rule as always).
``specs/issue832_kitchen1_pairing_owner_pair.json`` (the only thing that
still referenced the stacked mode) is deleted alongside it.

Deliberately NOT done here
--------------------------
* No per-pair telemetry (pair counts, correction magnitudes, authoritative
  pair stamps). Those are #832's own remaining acceptance items and are
  uncapturable without a subscriber on the ``rgbd_synced`` topic -- see
  ``rgbd_pairing_telemetry.json``'s ``*_unavailable_reason`` fields.
  ``pair_count`` is a partial exception: the launch script now derives an
  aggregate count from the sync node's own ``PairLedger`` log line (issue
  #1114's ``pair accounting: ... pairs_published=N``) when available, which
  is strictly coarser than a true per-pair subscriber but is real, observed
  data rather than a fabricated null.
* This does not close #832, whose acceptance criteria additionally require a
  real problem-bag validation run.

Known interaction, recorded rather than silently inherited
----------------------------------------------------------
``train_streaming.py`` passes ``legacy_topology=`` from the source-decimation
block's ``legacy_composition_active`` (two *sampling* owners composing).
``TopologyMode.LEGACY`` suppresses ``DUPLICATE_INGRESS_AUTHORITY`` for
*every* responsibility, so a run that is simultaneously composed-decimation
AND ``legacy_stacked`` would report the pairing duplicate as
``duplicate=True`` under a ``legacy`` label rather than as a failure code.
That requires both ``--streaming_frame_stride > 1`` and
``--streaming_source_stride > 1``, which no bag spec in this repo sets; it is
a pre-existing property of the resolver's single global topology flag, not
introduced here, and is left to #837 proper if per-responsibility topology is
ever wanted.
"""

from __future__ import annotations

import os
from typing import List, Mapping, Optional

from eligsir.io.ingress_ownership import (
    OwnerDeclaration,
    OwnershipRole,
    Responsibility,
)

# Environment variables exported by human_launch_scripts/05_run_offline_rosbag.sh
# onto the trainer container. These carry the three RAW launch inputs, not a
# derived mode string, so this module stays the single place the mode is
# derived (the launcher's own post-run telemetry writer derives it a second
# time in shell; tests/test_issue837_criterion8_rgbd_pairing_owner.py asserts
# the two derivations agree for every input combination).
ENV_BAKED_POSES = "ELIGSIR_BAG_USE_BAKED_POSES"
ENV_LEGACY_SYNCER = "ELIGSIR_BAG_USE_LEGACY_SYNCER"
ENV_PAIRING_OWNER = "ELIGSIR_BAG_RGBD_PAIRING_OWNER"

# Mode tokens. These are the SAME tokens the launcher writes as
# rgbd_pairing_telemetry.json's "rgbd_pairing_owner_mode", deliberately reused
# verbatim so the two artifacts join on one vocabulary (issue #832's Telemetry
# section: "Feed #805 rather than adding a parallel report vocabulary").
#
# MODE_LEGACY_STACKED ("legacy_stacked") was deleted by #832's hard-cut
# amendment (#1073) -- the launch scripts refuse that value at startup, so
# there is no longer any live producer of it. Left undefined here on purpose
# (not deprecated-but-kept) so a stray reference fails fast at import time.
MODE_CPP_SYNC = "cpp_sync"
MODE_LEGACY_SYNCER_NO_CPP_NODE = "legacy_syncer_no_cpp_node"
MODE_NOT_APPLICABLE = "n/a"

# Owner names.
OWNER_CPP_SYNC_NODE = "eligsir_rgbd_sync_node"
OWNER_DECOMPRESS_RGBD = "decompress_rgbd"


def _truthy(value: Optional[str]) -> bool:
    """Interpret a shell-exported flag. The launcher exports ``0``/``1``;
    anything unset/empty is False, and any other non-``0`` value is treated as
    True so a hand-set ``true``/``yes`` does not silently read as False."""
    if value is None:
        return False
    v = value.strip().lower()
    if v in ("", "0", "false", "no"):
        return False
    return True


def resolve_rgbd_pairing_owner_mode(
    *,
    use_baked_poses: bool,
    use_legacy_syncer: bool,
    rgbd_pairing_owner: Optional[str],
) -> str:
    """Derive the realized pairing-owner mode from the three raw launch inputs.

    Mirrors ``05_run_offline_rosbag.sh``'s own branch order exactly:
    baked-pose runs run no pairing stack at all; ``--legacy-syncer`` disables
    the C++ node entirely; otherwise the run is the canonical single owner.
    ``rgbd_pairing_owner`` is accepted as a parameter for backward
    compatibility with old callers/reports but the only value the launch
    scripts can produce post-#832-hard-cut is ``"cpp_sync"`` -- any other
    (unrecognized, or the removed ``"legacy_stacked"``) value is ignored
    rather than trusted, same "absent beats fabricated" rule as
    ``resolve_rgbd_pairing_owner_declarations`` applies below.
    """
    if use_baked_poses:
        return MODE_NOT_APPLICABLE
    if use_legacy_syncer:
        return MODE_LEGACY_SYNCER_NO_CPP_NODE
    return MODE_CPP_SYNC


def rgbd_pairing_owner_mode_from_env(env: Optional[Mapping[str, str]] = None) -> Optional[str]:
    """Resolve the mode from the launcher-exported environment.

    Returns ``None`` -- meaning "this run has no bag RGB-D pairing stack in
    front of it at all, leave the responsibilities unresolved" -- when the
    launcher's variables are absent. That is the case for every TUM/ScanNet/
    generic streaming-replay run and every offline ``train.py`` run, and it is
    the same "never fabricate an owner" discipline
    ``unwrap_frame_source_overflow_policies`` uses for non-``ros2://live``
    sources (issue #837 criterion 8 trap 2).
    """
    env = os.environ if env is None else env
    if ENV_PAIRING_OWNER not in env and ENV_BAKED_POSES not in env and ENV_LEGACY_SYNCER not in env:
        return None
    return resolve_rgbd_pairing_owner_mode(
        use_baked_poses=_truthy(env.get(ENV_BAKED_POSES)),
        use_legacy_syncer=_truthy(env.get(ENV_LEGACY_SYNCER)),
        rgbd_pairing_owner=env.get(ENV_PAIRING_OWNER),
    )


def resolve_rgbd_pairing_owner_declarations(mode: Optional[str]) -> List[OwnerDeclaration]:
    """Build the ``OwnerDeclaration`` list for a realized pairing-owner mode.

    ``None`` or :data:`MODE_NOT_APPLICABLE` yields an EMPTY list: the run has
    no RGB-D pairing/correction/decode stack whose ownership this repo can
    observe, so those responsibilities stay absent/unresolved rather than
    being fabricated.

    The historical stacked topology (``decompress_rgbd sync:=true`` pairing
    and re-stamping, followed by the C++ ``sync_node`` pairing and
    re-stamping the result again) was deleted by #832's hard-cut amendment
    and can no longer be launched, so this function now only ever resolves
    the two genuinely-single-owner topologies below.
    """
    if mode is None or mode == MODE_NOT_APPLICABLE:
        return []

    if mode == MODE_LEGACY_SYNCER_NO_CPP_NODE:
        # --legacy-syncer: the C++ sync_node is not started at all, so
        # decompress_rgbd is genuinely the sole owner of all three
        # responsibilities. Single owner each => structurally valid, and
        # still self-describing as a non-canonical topology via `mode`.
        return [
            OwnerDeclaration(
                Responsibility.TRANSPORT_DECODE,
                owner=OWNER_DECOMPRESS_RGBD,
                role=OwnershipRole.MUTATOR,
                mode="decode_and_pair",
                attributes={"pairing_owner_mode": mode, "decompress_sync_param": True},
            ),
            OwnerDeclaration(
                Responsibility.RGBD_TEMPORAL_PAIRING,
                owner=OWNER_DECOMPRESS_RGBD,
                role=OwnershipRole.MUTATOR,
                mode="approximate_time_pair",
                attributes={"pairing_owner_mode": mode, "cpp_sync_node_running": False},
            ),
            OwnerDeclaration(
                Responsibility.RGBD_TIMESTAMP_CORRECTION,
                owner=OWNER_DECOMPRESS_RGBD,
                role=OwnershipRole.MUTATOR,
                mode="restamp_depth_to_color",
                attributes={"pairing_owner_mode": mode, "cpp_sync_node_running": False},
            ),
        ]

    if mode != MODE_CPP_SYNC:
        # Unknown token (including the removed "legacy_stacked"): refuse to
        # guess a topology. Same rule as above -- absent beats fabricated.
        return []

    return [
        OwnerDeclaration(
            Responsibility.TRANSPORT_DECODE,
            owner=OWNER_DECOMPRESS_RGBD,
            role=OwnershipRole.MUTATOR,
            mode="decode_only",
            attributes={
                "pairing_owner_mode": mode,
                "decompress_sync_param": False,
            },
        ),
        OwnerDeclaration(
            Responsibility.RGBD_TEMPORAL_PAIRING,
            owner=OWNER_CPP_SYNC_NODE,
            role=OwnershipRole.MUTATOR,
            mode="software_corrected",
            attributes={"pairing_owner_mode": mode, "cpp_sync_node_running": True},
        ),
        OwnerDeclaration(
            Responsibility.RGBD_TIMESTAMP_CORRECTION,
            owner=OWNER_CPP_SYNC_NODE,
            role=OwnershipRole.MUTATOR,
            mode="software_corrected",
            attributes={
                "pairing_owner_mode": mode,
                "monotonic_pair_guard": True,
            },
        ),
    ]


def rgbd_pairing_owner_declarations_from_env(
    env: Optional[Mapping[str, str]] = None,
) -> List[OwnerDeclaration]:
    """Convenience: :func:`rgbd_pairing_owner_mode_from_env` composed with
    :func:`resolve_rgbd_pairing_owner_declarations`. This is what the
    report-writing call site in ``train_streaming.py`` calls."""
    return resolve_rgbd_pairing_owner_declarations(rgbd_pairing_owner_mode_from_env(env))


def rgbd_pairing_ownership_debug(env: Optional[Mapping[str, str]] = None) -> dict:
    """Small introspection helper (tests/diagnostics): the resolved mode and
    the raw inputs it came from, without building declarations."""
    env = os.environ if env is None else env
    return {
        "mode": rgbd_pairing_owner_mode_from_env(env),
        "raw": {
            ENV_BAKED_POSES: env.get(ENV_BAKED_POSES),
            ENV_LEGACY_SYNCER: env.get(ENV_LEGACY_SYNCER),
            ENV_PAIRING_OWNER: env.get(ENV_PAIRING_OWNER),
        },
    }


__all__ = [
    "ENV_BAKED_POSES",
    "ENV_LEGACY_SYNCER",
    "ENV_PAIRING_OWNER",
    "MODE_CPP_SYNC",
    "MODE_LEGACY_SYNCER_NO_CPP_NODE",
    "MODE_NOT_APPLICABLE",
    "OWNER_CPP_SYNC_NODE",
    "OWNER_DECOMPRESS_RGBD",
    "resolve_rgbd_pairing_owner_mode",
    "rgbd_pairing_owner_mode_from_env",
    "resolve_rgbd_pairing_owner_declarations",
    "rgbd_pairing_owner_declarations_from_env",
    "rgbd_pairing_ownership_debug",
]
