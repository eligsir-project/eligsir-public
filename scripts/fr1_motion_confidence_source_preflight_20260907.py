#!/usr/bin/env python3
"""Fail-closed FR1 source/split preflight for the motion-confidence pair."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from eligsir.io.sources.datasets.tum import TUMFrameSource
from eligsir.runtime.causal_evaluation import build_causal_holdout_cameras, causal_holdout_plan
from eligsir.utils.baseline_comparison import _camera_set_hash


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-root", type=Path, required=True)
    parser.add_argument("--mapper-source-path", default="/data/scene")
    parser.add_argument("--tum-sequence", required=True)
    parser.add_argument("--eval-guard", type=int, required=True)
    parser.add_argument("--expected-source", type=int, required=True)
    parser.add_argument("--expected-train", type=int, required=True)
    parser.add_argument("--expected-scored", type=int, required=True)
    parser.add_argument("--expected-guard", type=int, required=True)
    parser.add_argument("--expected-scored-hash", required=True)
    parser.add_argument("--expected-rule-hash", required=True)
    return parser.parse_args()


def _causal_args(cli: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        source_path=str(cli.mapper_source_path),
        streaming_eval_hold=16,
        streaming_initial_frames=5,
        streaming_eval_guard=cli.eval_guard,
        streaming_eval_min_baseline_m=0.03,
        streaming_eval_min_angle_deg=2.0,
        streaming_partition_focus_weight=0.0,
    )


def _same_raw_frame(identity, candidate) -> bool:
    return (
        identity.index == candidate.index
        and identity.rgb_path == candidate.rgb_path
        and identity.depth_path == candidate.depth_path
        and np.array_equal(identity.c2w, candidate.c2w)
    )


def main() -> None:
    cli = _args()
    identity = tuple(
        TUMFrameSource(
            str(cli.scene_root), sequence=cli.tum_sequence, motion_depth_confidence_mode="identity"
        )
    )
    motion = tuple(
        TUMFrameSource(
            str(cli.scene_root), sequence=cli.tum_sequence, motion_depth_confidence_mode="motion"
        )
    )
    if len(identity) != len(motion):
        raise ValueError("motion confidence changed frame count")
    for mode in ("identity", "motion"):
        frames = identity if mode == "identity" else motion
        if any(not _same_raw_frame(identity_frame, frame) for identity_frame, frame in zip(identity, frames)):
            raise ValueError(f"{mode} mode changed raw frame identity or nearest pose")

    plan = causal_holdout_plan(identity, _causal_args(cli))
    train = {frame.index for frame in identity if plan.role_for_ordinal(frame.index) == "train"}
    counts = (len(identity), len(train), len(plan.holdout_ordinals), len(plan.guard_ordinals))
    expected = (cli.expected_source, cli.expected_train, cli.expected_scored, cli.expected_guard)
    if counts != expected:
        raise ValueError(f"source role counts {counts!r} != expected {expected!r}")
    rule_hash = str(plan.eval_metadata.get("holdout_hash", ""))
    if rule_hash != cli.expected_rule_hash:
        raise ValueError(f"rule hash {rule_hash} != expected {cli.expected_rule_hash}")
    scored_hash = _camera_set_hash(build_causal_holdout_cameras(plan, data_device="cpu"))
    if scored_hash != cli.expected_scored_hash:
        raise ValueError(f"scored hash {scored_hash} != expected {cli.expected_scored_hash}")

    qualified = neutral = 0
    for frame in motion:
        metadata = getattr(frame, "_motion_depth_confidence_diagnostics", None)
        if not isinstance(metadata, dict) or metadata.get("motion_depth_confidence_source") != "offline_tum":
            raise ValueError("motion metadata is missing the offline TUM source")
        reason = metadata.get("motion_depth_confidence_reason")
        transform = metadata.get("motion_depth_confidence_transform")
        if transform is None:
            if not isinstance(reason, str) or not reason:
                raise ValueError("motion metadata lacks an explicit neutral reason")
            neutral += 1
            continue
        if not isinstance(transform, np.ndarray) or transform.shape != (4, 4) or not np.isfinite(transform).all():
            raise ValueError("motion metadata has an invalid capture transform")
        qualified += 1

    print(
        json.dumps(
            {
                "status": "PASS",
                "source_count": len(identity),
                "train": len(train),
                "scored": len(plan.holdout_ordinals),
                "guard": len(plan.guard_ordinals),
                "realized_scored_hash": scored_hash,
                "rule_hash": rule_hash,
                "identity_raw_equal": True,
                "motion_raw_equal": True,
                "motion_metadata_qualified": qualified,
                "motion_metadata_neutral": neutral,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
