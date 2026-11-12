"""Read the C++ RGB-D sync node's own frame-attribution ledger out of the
per-run log the launch scripts already capture (issue #1114).

**Why this exists.** ``utils.partition_eval.compute_bag_source_gap`` (issue
#865) cross-checks a bag's own per-topic ``message_count`` against the
trainer's ``ingest_counters.rgb_received`` and refuses the run when the two
disagree. That comparison has no attribution channel, so a frame that was
*explained* at a known stage is byte-identical, in ``report.json``, to one
that silently vanished. On a genuinely clean kitchen1 replay the difference
is exactly ``1`` -- one real matcher drop attributed to a 34.6 ms stream gap
-- and every healthy run was therefore being stamped
``mapping_result_valid: False``. See issue #1114.

**Where the data comes from.** ``eligsir-rgbd-sync``'s ``SyncNode`` owns
a ``PairLedger`` that tracks every color/depth frame from its subscription
callback to a terminal disposition (published, dropped-with-a-reason, or
still pending in the matcher), and prints the whole accounting as one
``pair accounting: ...`` line -- periodically (``drop_summary_period_sec``,
default 10 s) and once more, prefixed ``FINAL``, from its destructor.
``human_launch_scripts/05_run_offline_rosbag.sh`` (line ~616) and
``05c_run_offline_rosbag_new_syncer.sh`` already redirect that container's
logs to ``<model_path>/rgbd_sync_node.log``, and the trainer container
bind-mounts the repo root, so the file is readable from inside the trainer
at report time with no new IPC, no new ROS2 message dependency in the train
image, and no rebuild of the deployed sync image.

**Why parse the log rather than subscribe to ``sync_drop_stats``.** The node
does publish a ``eligsir_rgbd_msgs/SyncDropStats`` message carrying the
same numbers, which would be the nicer interface -- but consuming it needs
that message package installed in the *trainer* image, and the deployed sync
image would have to be rebuilt for any change either way. The log line is
already emitted, already captured per-run, and already on disk next to
``bag_source_manifest.json``. Parsing it costs one regex and keeps this fix
inside one repo.

**Trust, not just parsing.** Everything here returns ``None`` rather than a
fabricated zero when the ledger is unavailable, mirroring this codebase's
"``None`` means cannot-verify, never clean" convention (see
``compute_bag_source_gap``'s own docstring). The *decision* about whether a
parsed ledger may be used to excuse a gap lives in ``compute_bag_source_gap``
-- see ``attribution_trust_refusal`` below for the gate.
"""

from __future__ import annotations

import os
import re
from typing import Optional

SYNC_NODE_LOG_BASENAME = "rgbd_sync_node.log"

# Matches the emitter in eligsir_rgbd_sync/src/pair_ledger.cpp
# (PairLedger::summary), as rendered through RCLCPP_INFO and podman logs:
#
#   [0m[INFO] [1787495280.72] [eligsir_rgbd_sync_node]: pair accounting: \
#     pairs_matched=59 pairs_published=59 | color: arrived=60 published=59 \
#     pending=0 dropped=1 unpaired_retired=1 balanced=yes | depth: ...
#
# The FINAL variant (SyncNode::finalizeAccounting) prefixes "FINAL " and
# appends " (stream_end_unpaired flushed=N)".
_ACCOUNTING_RE = re.compile(r"(?P<final>FINAL\s+)?pair accounting:\s*(?P<body>.*)$")
_KV_RE = re.compile(r"(?P<key>[A-Za-z_][A-Za-z0-9_]*)=(?P<val>[A-Za-z0-9_.]+)")
_FINAL_SUFFIX_RE = re.compile(r"\s*\(stream_end_unpaired flushed=\d+\).*$")

_STREAM_NAMES = ("color", "depth")
# The four DropReason spellings PairLedger::drop_reason_name can emit. Any
# key inside a stream segment that is not one of the structural fields below
# is treated as a per-reason drop count, so a NEW reason added upstream is
# still counted (and named) rather than silently ignored.
_STRUCTURAL_KEYS = frozenset({"arrived", "published", "pending", "dropped", "balanced"})


def _parse_int(value: str) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def parse_pair_accounting_line(line: str) -> Optional[dict]:
    """Parse one ``pair accounting:`` line into a plain dict.

    Returns ``None`` for any line that isn't one, or that is one but is
    missing a field this module needs -- a truncated/interleaved log line
    must degrade to "cannot verify", never to a partially-populated dict a
    caller could mistake for a complete ledger.
    """
    if not isinstance(line, str):
        return None
    match = _ACCOUNTING_RE.search(line)
    if not match:
        return None
    body = match.group("body")
    # The FINAL variant appends " (stream_end_unpaired flushed=N)" AFTER the
    # last stream segment; without stripping it, `flushed` would be scraped
    # into the depth segment's per-reason drop counts as a phantom reason.
    body = _FINAL_SUFFIX_RE.sub("", body)
    # `|` separates the header from each per-stream segment; the header
    # carries pairs_matched/pairs_published/paired_without_arrival_record.
    segments = [s.strip() for s in body.split("|")]
    if not segments:
        return None

    header_kv = {m.group("key"): m.group("val") for m in _KV_RE.finditer(segments[0])}
    out: dict = {
        "is_final": bool(match.group("final")),
        "pairs_matched": _parse_int(header_kv.get("pairs_matched", "")),
        "pairs_published": _parse_int(header_kv.get("pairs_published", "")),
        # Only printed when nonzero; absent therefore means zero, which is
        # the ONE place an absent field legitimately reads as zero here --
        # the emitter's own `if (paired_without_arrival_record_ > 0)`.
        "paired_without_arrival_record": _parse_int(
            header_kv.get("paired_without_arrival_record", "0")
        ),
    }
    if out["pairs_matched"] is None or out["pairs_published"] is None:
        return None

    for stream in _STREAM_NAMES:
        segment = next((s for s in segments[1:] if s.startswith(stream + ":")), None)
        if segment is None:
            return None
        kv = {m.group("key"): m.group("val") for m in _KV_RE.finditer(segment)}
        arrived = _parse_int(kv.get("arrived", ""))
        published = _parse_int(kv.get("published", ""))
        pending = _parse_int(kv.get("pending", ""))
        dropped = _parse_int(kv.get("dropped", ""))
        balanced_raw = kv.get("balanced")
        if None in (arrived, published, pending, dropped) or balanced_raw is None:
            return None
        reasons = {
            k: _parse_int(v)
            for k, v in kv.items()
            if k not in _STRUCTURAL_KEYS and _parse_int(v) is not None
        }
        out[stream] = {
            "arrived": arrived,
            "published": published,
            "pending": pending,
            "dropped": dropped,
            "balanced": balanced_raw == "yes",
            "drop_reasons": reasons,
        }
    return out


def parse_sync_pair_ledger_log(path: str) -> Optional[dict]:
    """Read the LAST ``pair accounting:`` line from a sync-node log.

    Prefers the ``FINAL`` line (emitted from ``SyncNode``'s destructor, after
    ``PairLedger::flush()`` has given every still-pending frame a terminal
    disposition) when one is present. In practice it usually is NOT: the
    trainer writes its report while the sync node is still running, and the
    launch script tears the node down afterwards -- so the normal case is the
    most recent *periodic* snapshot. That is fine for this module's consumer,
    because a frame still pending in the matcher is counted in ``pending``
    and is just as undelivered as one already retired; what matters is that
    every arrival has *some* disposition, which is exactly the invariant
    ``balanced`` asserts.

    ``line_kind`` records which of the two was used, and ``n_lines`` how many
    accounting lines the file held, so a consumer can see the provenance
    rather than having to trust it.
    """
    try:
        with open(path, "r", errors="replace") as fh:
            lines = fh.readlines()
    except OSError:
        return None

    parsed = [p for p in (parse_pair_accounting_line(ln) for ln in lines) if p]
    if not parsed:
        return None
    finals = [p for p in parsed if p["is_final"]]
    chosen = finals[-1] if finals else parsed[-1]
    chosen = dict(chosen)
    chosen["line_kind"] = "final" if chosen.pop("is_final") else "periodic"
    chosen["n_accounting_lines"] = len(parsed)
    chosen["source_log"] = path
    return chosen


def load_sync_pair_ledger(model_path: Optional[str]) -> Optional[dict]:
    """Load ``<model_path>/rgbd_sync_node.log``'s ledger, or ``None``.

    ``None`` for every run that isn't a bag replay through the C++ sync node
    (including ``RGBD_PAIRING_OWNER=legacy`` runs, which start no sync node
    and therefore write no such log), predates this mechanism, or whose log
    is unreadable/unparseable. Never raises: a missing or corrupt sidecar log
    must not abort report generation, matching ``_load_bag_source_manifest``.
    """
    if not model_path:
        return None
    try:
        return parse_sync_pair_ledger_log(os.path.join(model_path, SYNC_NODE_LOG_BASENAME))
    except Exception:  # pragma: no cover - defensive; parser is total
        return None


def attribution_trust_refusal(
    ledger: Optional[dict],
    n_color: Optional[int],
    n_depth: Optional[int],
) -> Optional[str]:
    """Return why this ledger may NOT be used to excuse a source gap, or
    ``None`` if it may.

    **This gate is the load-bearing half of issue #1114's fix.** An attributed
    drop reason does NOT by itself mean "the source legitimately had nothing
    to pair this with". ``UnpairedRetired`` means only "the matcher could not
    pair it" -- so if DDS lost 50 depth messages between the bag player and
    the sync node, the node would see 50 unpairable color frames, retire them
    all with an attributed reason, and a naive ``unaccounted = estimate -
    (arrived + attributed_drops)`` would launder exactly the silent loss
    issue #865 exists to catch.

    ``PairLedger``'s own ``warnIfArrivalCountsReconstructed`` names the
    discriminator: the per-stream ``arrived`` counters "are exactly what
    distinguishes 'the matcher lost these frames' from 'transport lost them
    before this node ever saw them'". So attribution is trusted only when the
    node observed *every* message the bag contains on both streams, its own
    per-stream invariant holds, and no arrival count had to be reconstructed.

    A useful side effect: a stale snapshot (one taken before the last frames
    were processed) reports ``arrived < message_count`` and fails this gate,
    so freshness needs no separate heuristic.
    """
    if not isinstance(ledger, dict):
        return "no_sync_pair_ledger"
    if n_color is None or n_depth is None:
        return "bag_topic_counts_unavailable"
    color = ledger.get("color")
    depth = ledger.get("depth")
    if not isinstance(color, dict) or not isinstance(depth, dict):
        return "ledger_missing_stream_accounts"
    if ledger.get("paired_without_arrival_record"):
        return "arrival_counts_reconstructed_not_observed"
    if not color.get("balanced") or not depth.get("balanced"):
        return "ledger_unbalanced"
    if color.get("arrived") != int(n_color):
        return (
            f"color_arrived={color.get('arrived')} != bag message_count={int(n_color)} "
            "(messages lost upstream of the sync node, or a stale snapshot)"
        )
    if depth.get("arrived") != int(n_depth):
        return (
            f"depth_arrived={depth.get('arrived')} != bag message_count={int(n_depth)} "
            "(messages lost upstream of the sync node, or a stale snapshot)"
        )
    return None
