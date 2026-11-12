"""``eligsir-session`` -- run a source through the canonical MappingSession.

The CLI half of #1075's "CLI/replay/tests/simulation/ROS integration all compose
this object". It builds ``source -> SensorPacket -> TrackingProvider ->
MappingSession`` and reports the ingress/tracking ledger as JSON.

No training, no optimizer, no GPU: this exercises identity, packet
construction, tracking association and session ownership rules, which is what a
bag/dataset ingress-determinism check needs and what
``eligsir-ros2``'s deterministic-ingress profile validates on the transport
side.

Example::

    eligsir-session --source /data/tum/rgbd_desk --max-frames 200
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import List, Optional


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="eligsir-session",
        description=(
            "Replay an RGB-D source through the canonical MappingSession and "
            "report the ingress/tracking ledger. Does not train."
        ),
    )
    parser.add_argument(
        "-s",
        "--source",
        required=True,
        help="Source path (TUM / ScanNet / RGBDSequence / rosbag), as --source_path.",
    )
    parser.add_argument(
        "--pose-source",
        default="dataset_pose",
        help=(
            "Tracking frontend name. Default 'dataset_pose' consumes the pose "
            "the frame source already established."
        ),
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Stop after this many frames (default: the whole source).",
    )
    parser.add_argument(
        "--allow-untracked",
        action="store_true",
        help=(
            "Do not refuse observations without a usable pose. Off by default: "
            "mapping an untracked frame places geometry at a fabricated pose."
        ),
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """Entry point. Returns 0 on success, 2 on a usage/resolution error."""
    args = _build_parser().parse_args(argv)

    # Imported lazily so --help costs nothing and so a missing optional
    # frontend dependency surfaces as a clear message, not an import crash.
    from eligsir.io.sources.registry import make_frame_source
    from eligsir.runtime.replay import (
        packets_from_frame_source,
        run_replay_session,
    )
    from eligsir.tracking.factory import build_tracking_provider

    try:
        provider = build_tracking_provider(args.pose_source)
    except KeyError as exc:
        print(f"[session] {exc}", file=sys.stderr)
        return 2

    try:
        frame_source = make_frame_source(args.source, argparse.Namespace())
    except Exception as exc:  # noqa: BLE001 - surfaced verbatim to the operator
        print(f"[session] cannot open source {args.source!r}: {exc}", file=sys.stderr)
        return 2

    summary = run_replay_session(
        packets_from_frame_source(
            frame_source,
            source_id=args.source,
            provider=args.pose_source,
            max_frames=args.max_frames,
        ),
        provider,
        source_id=args.source,
        require_tracking=not args.allow_untracked,
    )
    print(json.dumps(summary.as_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
