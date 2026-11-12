"""Deterministic trajectory-partition helpers shared by the post-training
comparison report (utils/comparison_report.py) and the in-training sentinel
telemetry (train_streaming.py).

Each camera is binned into one of PARTITIONS trajectory segments by its
source frame index fraction. The denominator MUST be a per-scene constant
(eval/scenes.json ``n_source_frames``) so bin edges never depend on which
frames a particular run admitted.
"""
from __future__ import annotations

import json
import os
import re
from typing import List, Optional

import numpy as np

from eligsir.utils.sync_pair_ledger import (
    attribution_trust_refusal as _sync_attribution_trust_refusal,
)
from eligsir.utils.coverage_ledger import COUNTERS as _COVERAGE_V2_COUNTERS, COVERAGE_LEDGER_VERSION

# Default/fallback number of trajectory partitions, used when no scene
# length is available to scale granularity (see partition_count()). Also
# the floor for partition_count()'s clamp -- kept as the historical default
# so existing report.json readers/comparisons over old runs are unaffected.
PARTITIONS = 6

# Clamp bounds for partition_count()'s trajectory-length-scaled granularity.
MIN_PARTITIONS = 6
MAX_PARTITIONS = 64
DEFAULT_FRAMES_PER_PARTITION = 120


def partition_count(
    n_source_frames: Optional[int],
    frames_per_partition: int = DEFAULT_FRAMES_PER_PARTITION,
    min_partitions: int = MIN_PARTITIONS,
    max_partitions: int = MAX_PARTITIONS,
) -> int:
    """Trajectory-length-scaled partition count (Partition-Focus v2, Part 3).

    A fixed PARTITIONS=6 dilutes a handful of genuinely bad views into a
    scene-sixth average on long trajectories (e.g. ScanNet++'s 7040 frames).
    Scale partitions to roughly ``frames_per_partition`` source frames each,
    clamped to [min_partitions, max_partitions] so very short scenes don't
    get 1-frame partitions and very long ones don't get thousands.

    Falls back to ``min_partitions`` (== PARTITIONS) when ``n_source_frames``
    is unknown/non-positive -- same behavior as the old fixed constant for
    unregistered/live scenes.
    """
    if not n_source_frames or n_source_frames <= 0 or frames_per_partition <= 0:
        return min_partitions
    k = int(n_source_frames) // int(frames_per_partition)
    return max(min_partitions, min(max_partitions, k))


def camera_source_index(cam) -> Optional[int]:
    """Deterministic source frame index of a camera.

    Streaming cameras carry the source frame index in both ``uid`` and
    ``colmap_id`` (see ``StreamingScene``: ``CameraInfo(uid=frame.index, ...)``
    and ``image_name=f"{frame.index:06d}"``). Fallback: parse the last digit
    group from ``image_name``.
    """
    for attr in ("uid", "colmap_id"):
        v = getattr(cam, attr, None)
        if v is None:
            continue
        try:
            iv = int(v)
        except (TypeError, ValueError):
            continue
        if iv >= 0:
            return iv
    digits = re.findall(r"\d+", str(getattr(cam, "image_name", "") or ""))
    if digits:
        return int(digits[-1])
    return None


def partition_id(source_index: int, max_source_index: int, k: int = PARTITIONS) -> int:
    """Partition id = floor(k * source_index / max_source_index), clamped to k-1.

    Deterministic given (source_index, max_source_index): the same test camera
    always lands in the same partition for a fixed per-scene denominator.
    """
    if max_source_index <= 0 or source_index <= 0:
        return 0
    return min(k - 1, int((k * int(source_index)) // int(max_source_index)))


def partition_index_tensor(cams: List, max_source_index: int, k: int = PARTITIONS, device=None):
    """Vectorization-ready partition assignment for a list of cameras.

    Reads ``camera_source_index``/``partition_id`` per camera (cheap
    Python-side attribute reads off Camera objects -- not a GPU sync) and
    returns a ``torch.LongTensor[len(cams)]`` suitable for
    ``scatter_add_``/``scatter_reduce_`` against a same-length GPU PSNR
    tensor (see utils.psnr_history.PsnrTickResult). Cameras whose source
    index can't be resolved are bucketed into partition 0 (rather than
    dropped) so the returned tensor always has the same length as ``cams``
    and stays index-aligned with a psnr_tensor built from the same cams
    list -- callers that need to exclude unresolvable cams must filter
    ``cams``/``psnr_tensor`` themselves before calling this.
    """
    import torch
    ids = []
    for cam in cams:
        src = camera_source_index(cam)
        ids.append(partition_id(src, max_source_index, k=k) if src is not None else 0)
    return torch.tensor(ids, dtype=torch.long, device=device)


def registry_n_source_frames(scene_source: Optional[str]) -> Optional[int]:
    """Look up the per-scene constant frame count in eval/scenes.json.

    The partition denominator must NOT depend on which cameras a particular
    run admitted (admission varies run-to-run and would flip bin edges), so
    scenes register a fixed ``n_source_frames``. Returns None when the scene
    is not registered (caller falls back to max-over-cameras and flags it).

    Matching is exact-path first, then falls back to a basename comparison
    (found 2026-08-09, task #43): different launch commands across this
    project’s history have bind-mounted the same dataset directory at
    different container-internal paths (e.g. an identity mount at
    an external TUM dataset path matching the registry's
    recorded path verbatim, vs. an alternate mount alias used elsewhere) —
    an exact-path-only match silently misses the second case,
    landing on the "unregistered scene" branch and disabling coverage
    enforcement even though the scene genuinely is registered. The two
    mount paths are never both visible inside a single container invocation
    (only whichever ``-v`` flag that launch used), so this can't be resolved
    via ``os.path.samefile``/inode comparison — basename comparison is the
    match that's actually invariant to container-side mount aliasing. Only
    applied when exactly one registry entry's basename matches, to stay safe
    against future basename collisions (checked at call time, not assumed
    from today's registry contents). Note this matches on leaf directory
    name alone: a registry entry whose ``path`` points at a generic
    container directory (rather than a specific dataset's own leaf, e.g.
    ``tum_bags_raw``'s ``.../ros2_raw/``) would be the collision risk to
    watch if such an entry is ever added.

    CORRECTED (issue #950 follow-up): the "today's registry has none" claim
    above was false and is exactly what let a real collision through. The
    ``len(basename_matches) == 1`` guard counts matching *registry entries*,
    not matching *real directories on disk* -- ``fr2_desk``'s registered TUM
    directory and two unregistered ROS2 bag directories
    (``ros2_raw/rgbd_dataset_freiburg2_desk``,
    ``ros2_compressed/rgbd_dataset_freiburg2_desk``) all share the leaf name
    ``rgbd_dataset_freiburg2_desk``, but only one registry entry (the TUM
    one) exists, so the guard never fires and a run pointed at either bag
    silently resolves to the TUM entry's ``n_source_frames`` -- wrong,
    because bags don't drop unassociated frames the way the TUM association
    step does. A registry entry can opt out of ever being used as a
    basename-fallback match by setting ``"basename_fallback": false``; such
    an entry still matches on exact path or scene key. Absent the field,
    behavior is unchanged (fallback allowed) -- this is the default.
    """
    if not scene_source:
        return None
    try:
        reg_path = os.path.join(os.path.dirname(__file__), "..", "eval", "scenes.json")
        with open(reg_path) as f:
            reg = json.load(f)
    except Exception:
        return None
    src = os.path.normpath(str(scene_source).rstrip("/"))
    src_basename = os.path.basename(src)
    basename_matches = []
    for name, entry in reg.items():
        if name.startswith("_") or not isinstance(entry, dict):
            continue
        p = entry.get("path")
        if not p:
            continue
        p_norm = os.path.normpath(str(p).rstrip("/"))
        if p_norm == src or name == scene_source:
            n = entry.get("n_source_frames")
            return int(n) if n else None
        if src_basename and os.path.basename(p_norm) == src_basename:
            if entry.get("basename_fallback", True) is False:
                continue
            basename_matches.append(entry)
    if len(basename_matches) == 1:
        n = basename_matches[0].get("n_source_frames")
        return int(n) if n else None
    return None


def resolve_max_source_index(scene_source: Optional[str], cams) -> tuple:
    """Resolve the partition denominator: (max_source_index, denominator_source).

    Prefers the per-scene registry constant (``denominator_source="registry"``);
    falls back to the max source index across the given cameras
    (``"fallback_max_cams"``), which is run-dependent and only for telemetry.
    """
    n_reg = registry_n_source_frames(scene_source)
    if n_reg and n_reg > 1:
        return n_reg - 1, "registry"
    idx = [camera_source_index(c) for c in (cams or [])]
    idx = [i for i in idx if i is not None]
    return (max(idx) if idx else 0), "fallback_max_cams"


def compute_partition_stats(per_view: List[dict], k: int = PARTITIONS) -> dict:
    """Aggregate per-view records ({"partition": int, "psnr": float}) into
    partition_psnr / partition_n / min / std over non-empty partitions."""
    bins: dict = {i: [] for i in range(k)}
    for rec in per_view:
        p = rec.get("partition")
        v = rec.get("psnr")
        if p is None or v is None:
            continue
        bins[int(p)].append(float(v))
    part_psnr = {f"p{i}": (float(np.mean(bins[i])) if bins[i] else None) for i in range(k)}
    part_n = {f"p{i}": len(bins[i]) for i in range(k)}
    means = [v for v in part_psnr.values() if v is not None]
    return {
        "partition_psnr": part_psnr,
        "partition_n": part_n,
        "partition_psnr_min": (float(min(means)) if means else None),
        "partition_psnr_std": (float(np.std(means)) if means else None),
    }


#: Canonical location of the coverage block inside a written ``report.json``.
#: TOP-LEVEL, not under ``geometry`` -- see get_trajectory_coverage() for the
#: measurement behind this (issue #504).
TRAJECTORY_COVERAGE_REPORT_KEY = "trajectory_coverage"


# Version 2 separates source traversal from the deliberately smaller mapping
# set.  A bootstrap frame, a real holdout, and a rejected mapping candidate
# are all source frames; treating any of them as an absent frame was the root
# cause of the false #494/#499 partial-coverage verdicts.
def _coverage_v2_state(
    counters: Optional[dict],
    n_source_frames: int,
    frames_admitted: Optional[int],
    frame_admission: Optional[str],
    admission_target_resolved: Optional[str],
) -> dict:
    """Validate the source-disposition conservation laws for coverage-v2.

    The counters are mutually exclusive source dispositions.  Therefore a
    finite source must satisfy both::

        source_ingested + unprocessed == n_source_frames
        source_ingested == bootstrap + holdout + guard + invalid + eligible
        eligible == admitted + mapping_skipped

    ``mapping_skipped`` is acceptable only for an explicitly gated/budgeted
    policy.  This is intentionally a pure validator: it never infers missing
    historical facts from a ratio or from test partition placement.
    """
    base = {
        "coverage_contract_version": 2,
        "coverage_accounting_status": "MISSING",
        "coverage_counter_errors": [],
        "source_coverage_fraction": None,
        "source_traversal_complete": None,
        "eligible_mapping_accounted": None,
        "mapping_admission_complete": None,
        "mapping_admission_intentionally_partial": None,
        "coverage_v2_valid": False,
    }
    if not isinstance(counters, dict):
        return base
    if counters.get("coverage_ledger_version") != COVERAGE_LEDGER_VERSION:
        base["coverage_counter_errors"] = ["missing_or_unsupported_ledger_version"]
        return base
    missing = [key for key in _COVERAGE_V2_COUNTERS if key not in counters]
    if missing:
        base["coverage_counter_errors"] = ["missing:" + ",".join(missing)]
        return base

    parsed: dict[str, int] = {}
    errors: list[str] = []
    for key in _COVERAGE_V2_COUNTERS:
        value = counters[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            errors.append(f"invalid:{key}")
        else:
            parsed[key] = value
    if errors:
        base["coverage_accounting_status"] = "INVALID"
        base["coverage_counter_errors"] = errors
        return base

    resolved = counters.get("resolved_source_frames")
    if isinstance(resolved, bool) or not isinstance(resolved, int) or resolved <= 1:
        base["coverage_counter_errors"] = ["missing_or_invalid_resolved_source_frames"]
        return base
    if not isinstance(counters.get("source_denominator_provenance"), str):
        base["coverage_counter_errors"] = ["missing_source_denominator_provenance"]
        return base

    source = parsed["source_frames_ingested_total"]
    bootstrap = parsed["bootstrap_frames_consumed"]
    holdout = parsed["reserved_holdout_frames"]
    guard = parsed["reserved_guard_frames"]
    invalid = parsed["rejected_invalid_frames"]
    eligible = parsed["eligible_mapping_frames"]
    admitted = parsed["frames_admitted"]
    skipped = parsed["mapping_skipped_frames"]
    unprocessed = parsed["unprocessed_frames"]
    if source + unprocessed != resolved:
        errors.append("source_plus_unprocessed_mismatch")
    if source != bootstrap + holdout + guard + invalid + eligible:
        errors.append("source_disposition_mismatch")
    if eligible != admitted + skipped:
        errors.append("eligible_mapping_mismatch")
    if frames_admitted is not None and int(frames_admitted) != admitted:
        errors.append("frames_admitted_mismatch")

    raw_mode = str(frame_admission or "").lower()
    target = str(admission_target_resolved or "").lower()
    # #600/#602: production soft-all deliberately retains the raw ``all``
    # token while bounding mapping admission. Only force-all is a hard
    # assimilation promise; all other explicitly budgeted/gated modes may
    # skip mapping after accounting every eligible source frame.
    strict_all = raw_mode == "force_all" or target == "force_all_stress"
    intentionally_partial = not strict_all and (
        target in {"soft_all", "selective"}
        or raw_mode not in {"", "all"}
    )
    source_complete = source == resolved and unprocessed == 0
    eligible_accounted = eligible == admitted + skipped
    mapping_complete = skipped == 0
    base.update(parsed)
    base.update({
        "coverage_counter_errors": errors,
        "resolved_source_frames": resolved,
        "raw_registry_source_frames": counters.get("raw_registry_source_frames"),
        "source_denominator_provenance": counters["source_denominator_provenance"],
        "source_coverage_fraction": source / resolved,
        "source_traversal_complete": source_complete,
        "eligible_mapping_accounted": eligible_accounted,
        "mapping_admission_complete": mapping_complete,
        "mapping_admission_intentionally_partial": intentionally_partial,
        "coverage_v2_valid": (
            not errors
            and source_complete
            and eligible_accounted
            and (mapping_complete or intentionally_partial)
        ),
    })
    base["coverage_accounting_status"] = "INVALID" if errors else "COMPLETE"
    return base


def get_trajectory_coverage(report: Optional[dict]) -> Optional[dict]:
    """Read a report.json's trajectory-coverage block, or ``None`` if absent.

    **Why this exists (issue #504, 2026-08-14).** The block is written by
    ``utils/comparison_report.write_post_training_report`` at the TOP LEVEL of
    ``report.json`` (``summary["trajectory_coverage"]``) and has been since it
    was introduced (commit ``1140f27``). Several consumers and docstrings
    nevertheless read ``report["geometry"]["trajectory_coverage"]``, a path no
    writer in this repo has ever produced: a scan of 400 real ``report.json``
    files under ``output/`` measured 397 with the top-level key and **zero**
    with the ``geometry`` one (the remaining 3 have neither -- they are
    ``n_test == 0`` reports, see below). Those consumers therefore degraded
    silently on every run rather than on some runs.

    The ``geometry`` fallback below is NOT backward compatibility -- no report
    has ever used that path. It exists solely so a consumer switched to this
    accessor cannot get *worse* than the wrong-path read it replaces, and so a
    future decision to relocate the block doesn't silently break readers.

    Returns the block verbatim when present, else ``None``. It NEVER
    synthesizes a block: "absent" must stay distinguishable from "measured and
    clean", because CLAUDE.md gives ``map_reinit_invalidated``/
    ``verdict == SUSPECTED_TRUNCATED`` the same standing as
    ``eval_valid == False``, and a fabricated healthy-looking block (e.g. a
    default ``stream_end_reason`` of ``"bag_complete"``) would be strictly
    worse than nothing. Callers keep their own explicit missing-block branch.
    """
    if not isinstance(report, dict):
        return None
    tc = report.get(TRAJECTORY_COVERAGE_REPORT_KEY)
    if isinstance(tc, dict):
        return tc
    geom = report.get("geometry")
    if isinstance(geom, dict):
        tc = geom.get(TRAJECTORY_COVERAGE_REPORT_KEY)
        if isinstance(tc, dict):
            return tc
    return None


def compute_bag_source_gap(
    bag_source_manifest: Optional[dict],
    ingest_counters: Optional[dict],
    frames_admitted: Optional[int],
    sync_pair_ledger: Optional[dict] = None,
) -> dict:
    """Cross-check a bag-replay run's real ingest count against the bag's own
    message index (issue #865, acceptance items 1-2).

    **Why this exists.** Before this, every coverage/completion signal in
    ``report.json`` (``stream_end_reason``, ``bag_frames_offered``,
    ``mapping_result_valid``) only ever counted frames that already reached
    the trainer -- none of them could detect a frame that never arrived at
    all. A real ORB-SLAM3 bag-replay run (kitchen1, ``i858_bagsmoke_
    20260816_004950``) lost 33 real RGB-D frames (~1.6%) at the tail of the
    bag while every one of those signals reported ``bag_complete`` +
    ``mapping_result_valid: true`` + ``frame_drop_detected: false``. See
    issue #865 for the full forensic trace.

    ``bag_source_manifest`` is a plain dict, normally read from
    ``<model_path>/bag_source_manifest.json`` -- a small JSON artifact
    written by ``human_launch_scripts/05_run_offline_rosbag.sh`` BEFORE the
    trainer launches (the bag file is static, so its own message counts,
    read straight from rosbag2's ``metadata.yaml``, are known up front and
    do not depend on anything the trainer does). Expected shape::

        {
            "bag_path": "...",
            "color_topic": "/camera/color/image_raw/compressed",
            "depth_topic": "/camera/depth/image_raw/compressed",
            "topics": {
                "/camera/color/image_raw/compressed": {"message_count": 2056},
                "/camera/depth/image_raw/compressed": {"message_count": 2055},
            },
        }

    ``None`` (the default -- every non-bag-replay run, and any bag-replay
    run launched by a script that predates this manifest) means "cannot
    verify", NOT "clean" -- mirrors ``bag_replay_geometry_fields``'s own
    None-is-not-False convention elsewhere in this module. This function
    deliberately does NOT attempt anything for a genuinely live,
    unbounded camera source: there is no post-hoc oracle for how many
    frames a live camera "should" have produced, so this mechanism is
    scoped to bag replay only, where the total is knowable from the file
    itself (see issue #865's own scoping discussion).

    The "frames arrived" side of the comparison prefers
    ``ingest_counters["rgb_received"]`` (issue #865 acceptance item 4,
    PR #867) -- the count of frames that reached ``ros2_live.py``'s
    receive callback BEFORE any admission/pose-staleness rejection, which
    is the correct apples-to-apples counterpart to the bag's raw message
    count. When ``ingest_counters`` is unavailable (an older run, or a
    source that doesn't populate it), this falls back to
    ``frames_admitted`` -- a WEAKER, more conservative basis: a frame
    rejected inside ``ros2_live.py`` (stale pose, bad depth, etc.) would
    also show up as "unaccounted" under this fallback, conflating "never
    arrived" with "arrived but rejected". ``arrival_basis`` in the return
    value always records which comparison was actually used, so a
    consumer can tell precise evidence from an approximation.

    ``sync_pair_ledger`` (issue #1114) is the C++ RGB-D sync node's own
    per-frame attribution, read from ``<model_path>/rgbd_sync_node.log`` by
    ``utils.sync_pair_ledger.load_sync_pair_ledger``. **It exists because
    ``frames_estimate = min(n_color, n_depth)`` is an upper bound on pairable
    RGB-D frames, not an exact count**: a real stream gap legitimately makes
    the true pairable count lower, and without an attribution channel such a
    frame is byte-identical, here, to one that silently vanished. On a
    genuinely clean kitchen1 replay that difference is exactly ``1`` (one
    matcher drop attributed to a 34.6 ms gap), which used to stamp every
    healthy run ``mapping_result_valid: False``.

    The subtraction is **bounded by measured evidence, never by a tolerance**
    (issue #1114 is explicit that a "allow up to N unaccounted frames" slack
    would re-create the bounded-but-unmeasured silent loss #865 exists to
    eliminate). Three independent bounds apply, and the smallest wins:

    1. ``color.arrived - color.published`` -- color frames the node saw and
       never published, each with an attributed reason.
    2. ``depth.arrived - depth.published`` -- likewise for depth. A *pair*
       needs both, so the pair-level shortfall cannot exceed either.
    3. ``abs(n_color - n_depth)`` -- the **source's own measured asymmetry**,
       i.e. how many frames the bag itself proves have no counterpart. This
       is the bound that stops a genuine mid-stream matcher failure (both
       streams arrived in full, hundreds of pairs never emitted) from being
       absorbed as "explained": those frames really are lost to the trainer,
       and only what the source justifies is subtracted.

    On top of that, the ledger is used at all only when
    ``utils.sync_pair_ledger.attribution_trust_refusal`` clears it -- most
    importantly requiring the node's own ``arrived`` counters to equal the
    bag's message counts on BOTH streams, so attribution can never excuse a
    frame lost upstream of the sync node. See that function's docstring for
    why an attributed drop reason alone is not sufficient evidence.
    ``None`` (no sync node in the pipeline -- e.g. an
    ``RGBD_PAIRING_OWNER=legacy`` run -- or an unreadable log) simply means
    no subtraction happens and the pre-#1114 behaviour stands.

    Returns a dict of purely additive fields (see the docstring of
    ``compute_trajectory_coverage`` for how they fold into
    ``mapping_result_valid``). Every field is ``None`` when
    ``bag_source_manifest`` is absent.
    """
    if not isinstance(bag_source_manifest, dict):
        return {
            "bag_source_n_messages": None,
            "bag_source_frames_estimate": None,
            "bag_source_frames_arrived": None,
            "bag_source_arrival_basis": None,
            "bag_source_frames_unaccounted": None,
            "bag_source_tail_loss_detected": None,
            "bag_source_tail_loss_reason": None,
            "bag_source_completeness_verified": None,
            "bag_source_raw_gap": None,
            "bag_source_attributed_drops": None,
            "bag_source_attribution_basis": None,
            "bag_source_attribution_trusted": None,
            "bag_source_attribution_refused_reason": None,
            "bag_source_upstream_shortfall": None,
            "bag_source_unpaired_beyond_source_asymmetry": None,
        }

    topics = bag_source_manifest.get("topics") or {}
    color_topic = bag_source_manifest.get("color_topic")
    depth_topic = bag_source_manifest.get("depth_topic")
    n_color = (topics.get(color_topic) or {}).get("message_count") if color_topic else None
    n_depth = (topics.get(depth_topic) or {}).get("message_count") if depth_topic else None
    counts = [int(c) for c in (n_color, n_depth) if isinstance(c, (int, float))]
    # A frame needs both a color and a depth message, so the conservative
    # (lower-bound) real-source-frame estimate is the min of the two topic
    # counts, not either one alone.
    frames_estimate = min(counts) if counts else None

    _ic = ingest_counters if isinstance(ingest_counters, dict) else {}
    rgb_received = _ic.get("rgb_received")
    if isinstance(rgb_received, (int, float)):
        frames_arrived = int(rgb_received)
        arrival_basis = "ingest_counters.rgb_received"
    elif frames_admitted is not None:
        frames_arrived = int(frames_admitted)
        arrival_basis = "frames_admitted_fallback"
    else:
        frames_arrived = None
        arrival_basis = None

    raw_gap = (
        (frames_estimate - frames_arrived)
        if (frames_estimate is not None and frames_arrived is not None)
        else None
    )

    # --- issue #1114: subtract only attribution the evidence justifies ----
    _refusal = _sync_attribution_trust_refusal(sync_pair_ledger, n_color, n_depth)
    attribution_basis = "sync_pair_ledger" if isinstance(sync_pair_ledger, dict) else None
    attributed_drops: Optional[int] = None
    upstream_shortfall: Optional[dict] = None
    unpaired_beyond_asymmetry: Optional[int] = None
    if isinstance(sync_pair_ledger, dict):
        _c = sync_pair_ledger.get("color") or {}
        _d = sync_pair_ledger.get("depth") or {}
        if isinstance(n_color, (int, float)) and isinstance(_c.get("arrived"), int):
            upstream_shortfall = {
                "color": int(n_color) - int(_c["arrived"]),
                "depth": (
                    int(n_depth) - int(_d["arrived"])
                    if isinstance(n_depth, (int, float)) and isinstance(_d.get("arrived"), int)
                    else None
                ),
            }
    if _refusal is None:
        _c = sync_pair_ledger["color"]
        _d = sync_pair_ledger["depth"]
        color_undelivered = int(_c["arrived"]) - int(_c["published"])
        depth_undelivered = int(_d["arrived"]) - int(_d["published"])
        source_asymmetry = abs(int(n_color) - int(n_depth))
        pair_undelivered = max(0, min(color_undelivered, depth_undelivered))
        attributed_drops = min(pair_undelivered, source_asymmetry)
        # Evidence, not a verdict: undelivered pairs the SOURCE's own
        # asymmetry does not justify. These stay inside frames_unaccounted
        # (they are real frames the trainer never saw); this field just names
        # how much of the residual the matcher, rather than transport, ate.
        unpaired_beyond_asymmetry = pair_undelivered - attributed_drops

    frames_unaccounted = (
        max(0, raw_gap - (attributed_drops or 0)) if raw_gap is not None else None
    )
    tail_loss_detected = (frames_unaccounted is not None) and (frames_unaccounted > 0)
    _attrib_note = (
        f" {attributed_drops} of the {raw_gap}-frame raw difference is attributed to "
        "the sync node's own ledger (see bag_source_attributed_drops)."
        if attributed_drops else ""
    )
    tail_loss_reason = (
        f"bag source (via {color_topic!r}/{depth_topic!r} message_count) "
        f"reports {frames_estimate} real RGB-D frame(s); the trainer's own "
        f"{arrival_basis} counter shows only {frames_arrived} -- "
        f"{frames_unaccounted} frame(s) unaccounted for.{_attrib_note} "
        "See issues #865, #1114."
        if tail_loss_detected else None
    )
    return {
        "bag_source_raw_gap": raw_gap,
        "bag_source_attributed_drops": attributed_drops,
        "bag_source_attribution_basis": attribution_basis,
        "bag_source_attribution_trusted": (
            None if attribution_basis is None else (_refusal is None)
        ),
        "bag_source_attribution_refused_reason": (
            None if attribution_basis is None else _refusal
        ),
        "bag_source_upstream_shortfall": upstream_shortfall,
        "bag_source_unpaired_beyond_source_asymmetry": unpaired_beyond_asymmetry,
        "bag_source_n_messages": {"color": n_color, "depth": n_depth},
        "bag_source_frames_estimate": frames_estimate,
        "bag_source_frames_arrived": frames_arrived,
        "bag_source_arrival_basis": arrival_basis,
        "bag_source_frames_unaccounted": frames_unaccounted,
        "bag_source_tail_loss_detected": tail_loss_detected,
        "bag_source_tail_loss_reason": tail_loss_reason,
        # None (unverifiable) unless we could actually compute a gap. False
        # only on a POSITIVE detected gap -- mirrors bag_replay_geometry_
        # fields'/frame_drop_detected's "None doesn't invalidate, only an
        # explicit False does" convention.
        "bag_source_completeness_verified": (
            None if frames_unaccounted is None else not tail_loss_detected
        ),
    }


def compute_trajectory_coverage(
    scene_source: Optional[str],
    n_reg: Optional[int],
    frames_admitted: Optional[int],
    partition_psnr: Optional[dict],
    k: int = PARTITIONS,
    frame_admission: Optional[str] = None,
    stream_end_reason: Optional[str] = None,
    tracking_reinit_events: Optional[list] = None,
    frames_dropped_by_inner_queue: int = 0,
    eval_valid: Optional[bool] = None,
    bag_replay_geometry_fields: Optional[dict] = None,
    source_ingress_geometry_fields: Optional[dict] = None,
    last_ingest_iter: Optional[int] = None,
    final_iteration: Optional[int] = None,
    mapping_phase_source: Optional[str] = None,
    ingest_counters: Optional[dict] = None,
    bag_source_manifest: Optional[dict] = None,
    admission_target_resolved: Optional[str] = None,
    sync_pair_ledger: Optional[dict] = None,
    stale_pose_holdout_rejected: int = 0,
    coverage_counters: Optional[dict] = None,
) -> dict:
    """Explicit, machine-readable verdict on whether a run covered the full
    scene trajectory, synthesized from data the reporting pipeline already
    computes (registry lookup, admitted-frame count, per-partition PSNR).

    Verdict priority: SUSPECTED_TRUNCATED > UNKNOWN_LIVE_SOURCE >
    UNKNOWN_UNREGISTERED > UNKNOWN_NO_PARTITION_DATA (issue #504, see
    ``partition_psnr`` below) > PARTIAL_BY_BUDGET (issue #602, see
    ``admission_target_resolved`` below) > PARTIAL_BY_DESIGN > PARTIAL > FULL.
    A registry miss
    or a live/unbounded source can never be reported as FULL or as a
    confirmed PARTIAL -- there is no fixed total to compare against.
    ``stream_end_reason == "idle_timeout_no_completion_signal"`` (see
    eligsir/io/sources/ros2_live.py) means the source's idle-timeout
    fired without a confirmed bag-playback-complete signal -- the run may
    have silently stopped early because an upstream producer stalled, not
    because the bag/session genuinely ended. ``stream_end_reason ==
    "producer_error"`` (2026-08-06, final review round 2) means the pump
    thread died from a real exception (SLAM crash, decode failure, etc.),
    not a clean end -- see
    ROS2LiveStreamingFrameSource.stream_end_reason's docstring. Both
    override every other verdict: a truncated run's numbers are never
    comparable regardless of what its (also truncated) coverage_fraction
    happens to say.

    ``partition_psnr=None`` (issue #504, 2026-08-14) means partition PSNR was
    never measured for this run -- e.g. a report with no test cameras at all,
    which previously got NO coverage block whatsoever because the sole call
    site sat inside comparison_report's ``if test_cams:`` branch. Such a run
    reports ``partition_data_available=False``, ``partitions_with_test_data=
    None``, ``full_coverage=None`` and (for an otherwise-registered offline
    scene) ``verdict="UNKNOWN_NO_PARTITION_DATA"`` -- never a confident
    ``PARTIAL``/``full_coverage=False``, which would read as measured evidence
    of under-coverage when nothing was measured. ``{}`` keeps its original
    meaning ("measured; no partition had test data"), so every pre-existing
    call site is bit-for-bit unchanged.

    ``frame_admission`` is the run's ``--streaming_frame_admission`` mode. Gated
    modes (anything other than "all"/None) deliberately reject non-novel or
    overlapping frames by design -- a low coverage_fraction there reflects
    healthy keyframe selection, not an under-covered or interrupted run. Such
    runs report "PARTIAL_BY_DESIGN" instead of "PARTIAL" so the label isn't
    misread as a defect.

    ``admission_target_resolved`` (issue #602) is the target the admission
    owner actually resolved to -- ``soft_all`` / ``selective`` /
    ``force_all_stress`` -- as opposed to ``frame_admission``, which is the
    raw requested flag. It exists because ``soft_all`` breaks the assumption
    the ``frame_admission`` check above encodes: a soft-all run still requests
    ``all``, so it would fall through to "PARTIAL" (reads as a defect) even
    though its shortfall is a deliberate backpressure decision, not lost
    trajectory. Such runs report "PARTIAL_BY_BUDGET".

    "PARTIAL_BY_BUDGET" is deliberately a NEW label rather than a widened
    meaning for either existing one: "PARTIAL_BY_DESIGN" specifically means a
    keyframe gate rejected non-novel frames, which is a different mechanism
    with different implications, and this verdict already feeds downstream
    analyses that branch on the existing values. Leaving this argument None
    (every pre-#602 call site) reproduces the previous behavior exactly.

    ``tracking_reinit_events`` (2026-08-06): a non-empty list means
    ``MapReinitTracker``/``OrbslamLogReinitWatcher`` detected a genuine
    ORB-SLAM3 Atlas map re-initialization mid-run (see
    ``eligsir/mapping/scheduler/tracking_confidence.py``). The veto
    those detectors drive is now PERMANENT for the rest of the run -- there
    is no "recovered" state anymore, so any non-empty list here means the
    run has an unanchored segment for its entire remaining duration. This
    is reported as a distinct ``map_reinit_invalidated`` field rather than
    folded into ``eval_valid`` (which is a holdout-split-validity flag with
    its own, unrelated meaning -- see comparison_report.py) or into
    ``verdict`` (which is specifically about frame-coverage completeness).
    Consumers must treat ``map_reinit_invalidated=True`` exactly like
    ``eval_valid=False`` / the ``SUSPECTED_TRUNCATED`` verdict: never quote
    this run's coverage or PSNR as valid.

    ``last_ingest_iter`` / ``final_iteration`` (issue #346 follow-up,
    2026-08-13): the machine-readable mapping/refinement phase boundary.
    ``last_ingest_iter`` is ``train_streaming.py``'s own ``_last_ingest_iter``
    -- the optimization step at which the most recent source frame was
    ingested, re-armed on every ``streaming_scene.ingest_next_frame()`` hit
    regardless of admission mode (see that variable's own comment block in
    ``train_streaming.py`` for the exact "admission-independent clock"
    semantics; it is tracked for every streaming run, not only
    ``--online_mode`` ones, even though only ``--online_mode`` runs use it to
    drive termination). ``final_iteration`` is the run's real last
    optimization step -- deliberately a SEPARATE parameter from this
    function's caller's own ``iteration``/``summary["iteration"]`` field,
    because that field is a known-bogus ``opt.iterations`` sentinel (2^31)
    for ``--online_mode`` runs (see CLAUDE.md's "online_mode report.json
    iteration sentinel bug" / project memory) -- callers that already know
    the real final iteration (``train_streaming.py``'s own loop variable at
    loop-exit) pass it explicitly here instead of relying on the sentinel.
    Both are ``None`` when not supplied (every call site before this
    parameter existed, and any report that isn't a real mid/final training
    report). ``mapping_phase_source`` records provenance: ``"live"`` for a
    value read directly from the running process's own state,
    ``"backfilled_exact"`` / ``"backfilled_approximate"`` for values
    recovered post-hoc from a saved ``train.log`` (see
    ``utils/mapping_phase_backfill.py``), or ``None`` when neither is
    supplied. This function does NOT itself judge whether PSNR at
    ``last_ingest_iter`` clears CLAUDE.md's phase-dependent PSNR quality
    targets -- it only makes the boundary itself machine-readable so a
    downstream consumer can.

    ``frames_dropped_by_inner_queue`` (2026-08-06, final review PR #37):
    count of frames the ROS2LiveRGBDSource's own inner emit queue silently
    overwrote (dropped oldest, unrelated to WS-D1's OUTER bounded-buffer
    eviction -- see ROS2LiveStreamingFrameSource.frames_dropped_by_inner_queue's
    docstring) because it filled while the outer pump was backpressuring.
    Any nonzero value means a genuine, previously-uncounted source frame was
    lost -- for bag replay this is a real full-coverage-contract violation
    (CLAUDE.md's "Full trajectory coverage" Run Contract), not a benign
    admission-gate rejection. Reported as ``frame_drop_detected`` with the
    same "treat like eval_valid=False" standing as ``map_reinit_invalidated``.

    ``eval_valid`` (2026-08-06, final review round 2, PR #37 finding #2):
    the caller's own holdout-split-validity flag (``summary["eval_valid"]``
    in comparison_report.py), threaded through here purely so this function
    can fold it into a single authoritative ``mapping_result_valid`` field
    -- see that field's docstring below for why a documentation-only
    "treat like eval_valid=False" convention (the previous approach for
    both ``map_reinit_invalidated`` and ``frame_drop_detected``) is not
    sufficient: a naive consumer checking only ``eval_valid``/``verdict``
    could still accept and quote a run that lost source frames or hit a
    mid-run map re-init. ``None`` (the default, for call sites that predate
    a real holdout eval, e.g. before ``eval_valid`` is computed) is treated
    as "unknown, not proven valid" -- ``mapping_result_valid`` requires it
    to be explicitly ``True``, never assumes it.

    ``bag_replay_geometry_fields`` (W3-15, 2026-08-09): the dict returned
    by ``BagReplayIntegrityReport.as_geometry_fields()``
    (``eligsir/io/policies/bag_replay.py``) when a ``BagReplayPolicy``
    was actually used for this run's ingestion path -- ``None`` (the
    default) for any run that didn't use one, which today is every
    production run (``BagReplayPolicy`` has no production call site yet;
    see that module's docstring). When present, its own
    ``mapping_result_valid`` (``False`` iff the policy's spill store was
    ever exhausted and a frame was genuinely lost -- a one-way latch) is
    AND'd into this function's ``mapping_result_valid`` exactly like
    ``frames_dropped_by_inner_queue`` already is, and its remaining
    ``bag_*`` fields are surfaced verbatim into the returned dict under
    ``bag_replay_*`` keys for visibility.

    ``ingest_counters`` (issue #865, acceptance item 4): a detached dict
    snapshot of ``eligsir/io/sources/ros2_live.py``'s accept/reject
    counters (``accepted_frames``, ``rgb_received``, ``queued_frames``, the
    per-topic buffer sizes, and every ``reject_*`` counter) -- previously
    computed on every diagnostics tick and published ONLY to a transient
    ROS2 DiagnosticArray topic, never persisted. ``None`` (the default) for
    any run whose source doesn't set this, i.e. every non-``ros2://live``
    source today -- surfaced verbatim, purely additive, and does NOT
    participate in ``mapping_result_valid``/``verdict`` computation on its
    own (see ``bag_source_manifest`` immediately below, which DOES use it).

    ``bag_source_manifest`` (issue #865, acceptance items 1-2): the bag's
    own real message counts (from rosbag2's ``metadata.yaml``, written by
    ``human_launch_scripts/05_run_offline_rosbag.sh`` to ``<model_path>/
    bag_source_manifest.json`` before the trainer launches -- see that
    script and ``compute_bag_source_gap``'s own docstring for the exact
    shape and rationale). ``None`` (the default, for any run that isn't a
    bag replay launched by that script, or predates this mechanism) means
    "cannot verify", not "clean". When present, this function calls
    ``compute_bag_source_gap(bag_source_manifest, ingest_counters,
    frames_admitted)`` and folds the result's
    ``bag_source_completeness_verified`` into ``mapping_result_valid``
    exactly like ``frames_dropped_by_inner_queue``/``map_reinit_
    invalidated`` already are: an explicit ``False`` (a real, positively
    detected gap between what the bag contains and what the trainer
    ingested) invalidates the run; ``None`` (unverifiable) does not. This
    directly addresses issue #865's finding that ``stream_end_reason ==
    "bag_complete"`` could coexist with real, silent tail loss and every
    other integrity signal reporting clean -- ``mapping_result_valid`` is
    the one field that now catches it. Deliberately scoped to bag replay
    only (where a post-hoc total is knowable from the bag file) and not to
    a genuinely live, unbounded camera source, which has no such oracle.

    ``stale_pose_holdout_rejected`` (issue #1135, 2026-08-23): count of
    RESERVED HOLDOUT frames that ``StreamingScene.ingest_next_frame`` dropped
    because ``frame._odom_valid is False`` -- i.e. a holdout whose reserved
    slot in the test set was NOT honored (unlike the #1125 map-reinit veto,
    which now always bypasses holdouts/guards -- see that method's own
    comment block for why the two cases resolve oppositely: a stale/copied
    odometry pose is wrong in an unbounded, frame-specific way with no
    run-level invalidation flag analogous to ``map_reinit_invalidated``, so
    admitting it into the test set would silently pollute
    ``mean_test_psnr`` with nothing to catch it). This does NOT by itself
    invalidate the run (a genuinely smaller-than-reserved but still correctly
    posed holdout set is a real, comparable -- if smaller -- eval; the
    existing ``holdout_hash``/``realized_holdout_hash`` verification-level
    ladder in ``utils/comparison_report.py``'s ``compare_against_baseline``
    already refuses/flags a resulting size mismatch when comparing two
    runs), so it is NOT folded into ``mapping_result_valid``. It exists
    purely so a shrunken ``n_test`` is attributable to a specific, named
    cause rather than silently mysterious -- see
    ``stale_pose_holdout_rejected_detected``/``stale_pose_holdout_reason``
    below.
    """
    # ``partition_psnr=None`` means "partition PSNR was never measured for
    # this run" (issue #504) -- distinct from ``{}`` ("measured, no partition
    # had test data"). Emitting a block at all for such a run is what makes
    # the safety fields (map_reinit_invalidated / stream_end_reason /
    # frame_drop_detected) visible for an n_test==0 run, but the
    # partition-derived fields must then read as explicitly unknown rather
    # than as a confident PARTIAL/full_coverage=False -- a not-measured run is
    # not evidence of partial coverage.
    partition_data_available = partition_psnr is not None
    partitions_with_test_data = (
        sum(1 for i in range(k) if partition_psnr.get(f"p{i}") is not None)
        if partition_data_available else None
    )
    scene_registered = bool(n_reg and n_reg > 1)

    # Coverage-v2 is the authoritative finite-source contract.  Partition PSNR
    # remains useful diagnostics, but its temporal placement says nothing about
    # whether source frames were traversed, so it cannot decide this verdict.
    _v2 = (
        _coverage_v2_state(
            coverage_counters,
            int(n_reg),
            frames_admitted,
            frame_admission,
            admission_target_resolved,
        )
        if scene_registered else None
    )

    # Determine verdict with priority: stall-suspected > live > unregistered >
    # coverage accounting > source/mapping completeness.  The legacy branch is
    # retained only for non-score-bearing callers that did not declare a real
    # holdout evaluation; score-bearing historical reports fail closed below.
    if stream_end_reason in ("idle_timeout_no_completion_signal", "producer_error"):
        verdict = "SUSPECTED_TRUNCATED"
    elif scene_source and str(scene_source).startswith("ros2://"):
        verdict = "UNKNOWN_LIVE_SOURCE"
    elif not scene_registered:
        verdict = "UNKNOWN_UNREGISTERED"
    elif _v2 is not None and _v2["coverage_accounting_status"] == "MISSING" and eval_valid is True:
        verdict = "UNKNOWN_COVERAGE_COUNTERS"
    elif _v2 is not None and _v2["coverage_accounting_status"] == "INVALID":
        verdict = "INVALID_COVERAGE_COUNTERS"
    elif _v2 is not None and _v2["coverage_accounting_status"] == "COMPLETE":
        if not _v2["source_traversal_complete"]:
            verdict = "PARTIAL"
        elif not _v2["eligible_mapping_accounted"]:
            verdict = "INVALID_COVERAGE_COUNTERS"
        elif not _v2["mapping_admission_complete"]:
            if not _v2["mapping_admission_intentionally_partial"]:
                verdict = "PARTIAL"
            elif str(admission_target_resolved or "").lower() in {"soft_all", "selective"}:
                verdict = "PARTIAL_BY_BUDGET"
            elif frame_admission is not None and str(frame_admission).lower() not in {"all", "force_all"}:
                verdict = "PARTIAL_BY_DESIGN"
            else:
                verdict = "PARTIAL"
        else:
            verdict = "FULL"
    elif not partition_data_available:
        verdict = "UNKNOWN_NO_PARTITION_DATA"
    else:
        # For registered scenes, compute coverage_fraction once for verdict decision
        # and return value (avoids duplication and potential desync).
        coverage_fraction = (
            (frames_admitted / n_reg) if frames_admitted is not None else None
        )
        _target = (str(admission_target_resolved).lower()
                   if admission_target_resolved is not None else None)
        if (
            coverage_fraction is not None
            and coverage_fraction >= 0.99
            and partitions_with_test_data == k
        ):
            verdict = "FULL"
        elif _target == "soft_all":
            # Issue #602: soft-all OBSERVES every source frame but budgets
            # mapping admission, so a coverage_fraction below 1.0 here is a
            # backpressure decision, not lost trajectory. It is neither
            # "PARTIAL" (which reads as a defect) nor "PARTIAL_BY_DESIGN"
            # (which means a keyframe gate rejected non-novel frames -- a
            # different mechanism with different implications). Given its own
            # label rather than overloading either, since this verdict feeds
            # downstream analyses that already branch on the existing two.
            # The per-frame accounting is in
            # report.json["admission_cadence"]["dispositions"].
            verdict = "PARTIAL_BY_BUDGET"
        elif _target == "force_all_stress":
            # Issue #600: force_all admits UNCONDITIONALLY -- nothing in that
            # mode is permitted to skip a frame. So a shortfall here means
            # frames were genuinely LOST, which is a real anomaly and must keep
            # the plain "PARTIAL" defect label.
            #
            # This branch is load-bearing rather than cosmetic: the generic
            # token test below asks `frame_admission != "all"`, and the literal
            # token for this mode is "force_all", which satisfies that test --
            # so without this branch a force_all run with lost frames would be
            # excused as PARTIAL_BY_DESIGN, i.e. exactly backwards. The token
            # did not exist when #602 wrote that test, so this is #600's own
            # token to account for.
            verdict = "PARTIAL"
        elif frame_admission is not None and str(frame_admission).lower() != "all":
            verdict = "PARTIAL_BY_DESIGN"
        else:
            verdict = "PARTIAL"

    # For live sources, force scene_registered and dependent fields to None/False,
    # regardless of whether n_reg was resolvable. This prevents naive readers of
    # report.json from treating coverage_fraction as meaningful.
    if verdict == "UNKNOWN_LIVE_SOURCE":
        scene_registered = False

    # V2's fraction is source traversal, not admitted mapping frames.  A
    # pre-v2, non-score-bearing caller retains the legacy display field only;
    # it cannot make mapping_result_valid true.
    coverage_fraction = (
        (_v2["source_coverage_fraction"]
         if (scene_registered and _v2 is not None
             and _v2["coverage_accounting_status"] == "COMPLETE")
         else (frames_admitted / n_reg)
         if (scene_registered and frames_admitted is not None and eval_valid is not True)
         else None)
    )

    _reinit_events = tracking_reinit_events or []
    _map_reinit_invalidated = len(_reinit_events) > 0
    # W3-15: None (no BagReplayPolicy in use this run) must NOT invalidate
    # the run -- only an explicit False (a real, observed frame loss) does.
    # Issue #1273: additive-only. The causal source buffer's arrival boundary
    # is REAL_TIME_DROP by construction, so these fields are surfaced for
    # attribution and are deliberately consulted by NO verdict and by
    # mapping_result_valid nowhere -- classifying them otherwise would
    # retroactively invalidate every causal run that ever hit buffer pressure.
    _ingress_fields = {
        k: v
        for k, v in (source_ingress_geometry_fields or {}).items()
        if k.startswith("ingress_") and k != "ingress_overflow_events"
    }
    _bag_fields = bag_replay_geometry_fields or {}
    _bag_mapping_result_valid = _bag_fields.get("mapping_result_valid", True)
    # Issue #865: same None-doesn't-invalidate convention as the bag-replay
    # fields above -- only an explicit False (a positively detected gap
    # between the bag's own message index and what the trainer ingested)
    # invalidates the run.
    _bag_gap = compute_bag_source_gap(
        bag_source_manifest, ingest_counters, frames_admitted, sync_pair_ledger
    )
    _bag_source_completeness_valid = _bag_gap["bag_source_completeness_verified"] is not False
    _v2_mapping_valid = (
        True
        if not scene_registered
        else bool(_v2 is not None and _v2["coverage_v2_valid"])
    )
    return {
        **_ingress_fields,
        "scene_registered": scene_registered,
        # Public coverage denominator is the selected replay cohort (after
        # stride/cap), never the raw registry trajectory. Preserve raw identity
        # under its own unambiguous field for partition/provenance consumers.
        "n_source_frames": (
            _v2.get("resolved_source_frames")
            if (scene_registered and _v2 is not None
                and _v2["coverage_accounting_status"] == "COMPLETE")
            else int(n_reg) if scene_registered else None
        ),
        "raw_registry_n_source_frames": int(n_reg) if scene_registered else None,
        "frames_admitted": frames_admitted,
        "coverage_fraction": coverage_fraction,
        # Versioned source-disposition ledger.  All fields are explicitly
        # unknown/missing rather than inferred for historical reports.
        **(_v2 or {
            "coverage_contract_version": 2,
            "coverage_accounting_status": "NOT_APPLICABLE",
            "coverage_counter_errors": [],
            "source_coverage_fraction": None,
            "source_traversal_complete": None,
            "eligible_mapping_accounted": None,
            "mapping_admission_complete": None,
            "mapping_admission_intentionally_partial": None,
            "coverage_v2_valid": False,
        }),
        "partitions_with_test_data": partitions_with_test_data,
        "partitions_total": k,
        # False iff partition PSNR was never measured (issue #504). A consumer
        # must not read partitions_with_test_data/full_coverage as evidence
        # when this is False -- they are None, not zero/False, in that case.
        "partition_data_available": partition_data_available,
        "full_coverage": (
            verdict == "FULL"
            if (_v2 is not None and _v2["coverage_accounting_status"] == "COMPLETE")
            else None
            if eval_valid is True
            else (verdict == "FULL") if partition_data_available else None
        ),
        "verdict": verdict,
        "frame_admission": frame_admission,
        "stream_end_reason": stream_end_reason,
        "tracking_reinit_events": _reinit_events,
        # Distinct from eval_valid/verdict (see docstring above) -- True iff
        # a genuine ORB-SLAM3 Atlas re-init was detected mid-run. The veto
        # it drives is permanent, so this is a whole-run invalidation flag,
        # not a partial-coverage signal.
        "map_reinit_invalidated": _map_reinit_invalidated,
        "map_reinit_reason": (
            "ORB-SLAM3 Atlas map re-initialized mid-run at frame(s) "
            + ", ".join(str(e.get("frame")) for e in _reinit_events)
            + " -- admission was permanently vetoed from that point forward; "
            "this run's poses/coverage/PSNR must not be treated as valid."
        ) if _map_reinit_invalidated else None,
        # Distinct from map_reinit_invalidated -- see docstring above.
        "frame_drop_detected": bool(frames_dropped_by_inner_queue > 0),
        "frames_dropped_by_inner_queue": int(frames_dropped_by_inner_queue),
        # W3-15: surfaced only when a BagReplayPolicy was actually in use
        # this run (see docstring above) -- bag_replay_mapping_result_valid
        # is None otherwise, distinguishable from a real False.
        "bag_replay_mapping_result_valid": _bag_fields.get("mapping_result_valid"),
        "bag_frames_offered": _bag_fields.get("bag_frames_offered"),
        "bag_frames_spilled": _bag_fields.get("bag_frames_spilled"),
        "bag_frames_lost": _bag_fields.get("bag_frames_lost"),
        "bag_loss_events": _bag_fields.get("bag_loss_events"),
        # Issue #865, acceptance item 4: surfaced verbatim, None for any
        # source that doesn't set it -- see this function's own docstring
        # above for exactly what's in here and why it matters.
        "ingest_counters": ingest_counters,
        # Issue #865, acceptance items 1-2: the bag-source-vs-trainer gap
        # cross-check -- see compute_bag_source_gap()'s docstring for field
        # meanings. All None when bag_source_manifest wasn't supplied
        # (cannot verify, not "clean").
        **_bag_gap,
        # Single authoritative validity signal (2026-08-06, final review
        # round 2, PR #37 finding #2): True iff eval_valid was explicitly
        # True AND none of the other invalidating conditions this function
        # tracks fired. A consumer that checks ONLY this one field can never
        # be misled by checking eval_valid/verdict alone and missing a
        # frame-drop or map-reinit invalidation reported elsewhere in this
        # dict -- replaces the prior "separate flag, same standing as
        # eval_valid=False, documented but not enforced" convention.
        # W3-15 (2026-08-09): also AND's in _bag_mapping_result_valid,
        # which defaults True when no BagReplayPolicy is in use so a run
        # without one is never falsely invalidated by this addition.
        "mapping_result_valid": bool(
            eval_valid is True
            and verdict not in {
                "SUSPECTED_TRUNCATED", "UNKNOWN_COVERAGE_COUNTERS",
                "INVALID_COVERAGE_COUNTERS",
            }
            and _v2_mapping_valid
            and not _map_reinit_invalidated
            and frames_dropped_by_inner_queue <= 0
            and _bag_mapping_result_valid
            and _bag_source_completeness_valid
        ),
        # Mapping/refinement phase boundary (issue #346 follow-up) -- see
        # this parameter's docstring above. Additive/optional: None on any
        # report that doesn't supply it, reproducing the exact dict shape
        # from before these three keys existed for every such call site.
        "last_ingest_iter": (int(last_ingest_iter) if last_ingest_iter is not None else None),
        "final_iteration": (int(final_iteration) if final_iteration is not None else None),
        "refinement_tail_iterations": (
            int(final_iteration) - int(last_ingest_iter)
            if (last_ingest_iter is not None and final_iteration is not None
                and int(final_iteration) >= int(last_ingest_iter))
            else None
        ),
        "mapping_phase_source": mapping_phase_source,
        # Issue #1135: attribution for a holdout reservation shortfall caused
        # by the stale-pose veto -- see this parameter's docstring above for
        # why this is deliberately NOT folded into mapping_result_valid.
        "stale_pose_holdout_rejected": int(stale_pose_holdout_rejected),
        "stale_pose_holdout_rejected_detected": bool(stale_pose_holdout_rejected > 0),
        "stale_pose_holdout_reason": (
            f"{int(stale_pose_holdout_rejected)} reserved holdout frame(s) were "
            "dropped instead of becoming test cameras because their pose was "
            "stale/copied (frame._odom_valid is False) -- inserting depth from "
            "such a pose would corrupt the map, and building an eval camera "
            "from it would render the wrong part of the scene and silently "
            "pollute mean_test_psnr. n_test is smaller than the holdout rule "
            "reserved as a result; this is a deliberate, attributed shortfall, "
            "not evidence the run itself is invalid."
        ) if stale_pose_holdout_rejected > 0 else None,
    }


def check_coverage_budget_feasible(
    is_finite_replay: bool,
    enforce_full_coverage: bool,
    n_frames: int,
    iterations: int,
    scene_source: str = "",
) -> None:
    """Raise RuntimeError if a finite, coverage-enforced replay's iteration
    budget cannot admit every frame even at the absolute floor of 1 step
    per frame. Called before MappingScheduler construction so an
    infeasible budget fails immediately, before any GPU work starts,
    rather than silently truncating coverage or crashing later with a
    confusing error.
    """
    if not (is_finite_replay and enforce_full_coverage):
        return
    if iterations >= n_frames:
        return
    raise RuntimeError(
        f"Coverage budget infeasible for scene '{scene_source or '(unset)'}': "
        f"--iterations={iterations} cannot admit all {n_frames} registered "
        f"frames even at the absolute floor of 1 step/frame (would need "
        f">={n_frames}). Either raise --iterations to at least {n_frames}, "
        f"or pass --no-streaming_enforce_full_coverage if this is an "
        f"intentional smoke test."
    )
