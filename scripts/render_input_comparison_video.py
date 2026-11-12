#!/usr/bin/env python3
"""Create bounded CPU-only input and sparse matched comparison videos.

This is a post-processing command. It reads an immutable completed source
tree and existing report frames; it never calls a renderer or training hook.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterator
from pathlib import Path

import numpy as np
from PIL import Image

from eligsir.utils.input_comparison_video import (
    ComparisonRow,
    SourceFrameRef,
    build_comparison_manifest,
    compare_ground_truth_panel,
    extract_render_panel,
    iter_tum_rgb_frames,
    make_comparison_frame,
    read_canonical_tum_index,
    read_tum_rgb_index,
)
from eligsir.utils.trajectory_video import compute_playback_speed, distribute_frame_repeats
from eligsir.utils.video_encode import write_mp4_cpu_stream


def _native_fps(timestamps: list[float]) -> float | None:
    if len(timestamps) < 2:
        return None
    gaps = sorted(b - a for a, b in zip(timestamps, timestamps[1:]) if b > a)
    if not gaps:
        return None
    middle = len(gaps) // 2
    median = gaps[middle] if len(gaps) % 2 else (gaps[middle - 1] + gaps[middle]) / 2
    return round(1.0 / median, 6) if median > 0 else None


def _source_duration(timestamps: list[float]) -> float | None:
    if len(timestamps) < 2:
        return None
    span = float(timestamps[-1] - timestamps[0])
    return round(span, 6) if span > 0 else None


def _bounded_repeats(
    timestamps: list[float], fps: int, max_encoded_frames: int | None
) -> list[int]:
    repeats = distribute_frame_repeats(timestamps, fps, max_encoded_frames=None)
    if max_encoded_frames in (None, 0) or sum(repeats) <= max_encoded_frames:
        return repeats
    if max_encoded_frames < len(repeats):
        raise ValueError(
            f"encoded-frame cap {max_encoded_frames} is below one frame per match ({len(repeats)})"
        )
    target = int(max_encoded_frames)
    total = float(sum(repeats))
    scaled = [value * target / total for value in repeats]
    bounded = [max(1, int(value)) for value in scaled]
    while sum(bounded) > target:
        candidates = [i for i, value in enumerate(bounded) if value > 1]
        if not candidates:
            raise ValueError("cannot satisfy encoded-frame cap while retaining every match")
        index = max(candidates, key=lambda i: (bounded[i] - scaled[i], bounded[i], -i))
        bounded[index] -= 1
    while sum(bounded) < target:
        index = max(range(len(bounded)), key=lambda i: (scaled[i] - bounded[i], -i))
        bounded[index] += 1
    return bounded


def _select_comparison_rows(
    rows: list[ComparisonRow], *, max_matches: int, encoded_cap: int
) -> list[ComparisonRow]:
    """Choose a deterministic trajectory-spanning cohort before image reads."""

    available = sorted(
        (row for row in rows if row.status == "matched"),
        key=lambda row: row.source_ordinal,
    )
    limit = len(available)
    if max_matches > 0:
        limit = min(limit, max_matches)
    if encoded_cap > 0:
        limit = min(limit, encoded_cap)
    if len(available) <= limit:
        return available
    if limit <= 0:
        return []
    if limit == 1:
        return [available[(len(available) - 1) // 2]]
    last = len(available) - 1
    denominator = limit - 1
    return [
        available[(index * last + denominator // 2) // denominator]
        for index in range(limit)
    ]


def _repeat_stream(frames: Iterator[np.ndarray], repeats: list[int]) -> Iterator[np.ndarray]:
    for frame, count in zip(frames, repeats):
        for _ in range(int(count)):
            yield frame


def _read_triptych(row: ComparisonRow) -> np.ndarray:
    if row.render_path is None:
        raise ValueError(f"cannot read missing rendered frame {row.render_id}")
    with Image.open(row.render_path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _comparison_frames(
    rows: list[ComparisonRow], refs_by_index: dict[int, SourceFrameRef], repeats: list[int]
) -> Iterator[np.ndarray]:
    for row, repeat in zip(rows, repeats):
        ref = refs_by_index[row.source_ordinal]
        source = next(iter_tum_rgb_frames([ref]))
        rendered = extract_render_panel(_read_triptych(row), source.shape)
        frame = make_comparison_frame(source, rendered)
        for _ in range(int(repeat)):
            yield frame


def _artifact_metadata(
    refs: list[SourceFrameRef], *, fps: int, encoded_frames: int, repeats: list[int]
) -> dict:
    timestamps = [ref.timestamp_s for ref in refs]
    source_duration = _source_duration(timestamps)
    video_duration = encoded_frames / float(fps) if fps > 0 else None
    return {
        "source_frame_count": len(refs),
        "source_first_timestamp_s": timestamps[0] if timestamps else None,
        "source_last_timestamp_s": timestamps[-1] if timestamps else None,
        "source_duration_s": source_duration,
        "native_source_fps": _native_fps(timestamps),
        "encoded_frames": int(encoded_frames),
        "encoded_fps": int(fps),
        "video_duration_s": round(video_duration, 6) if video_duration is not None else None,
        "playback_speed": compute_playback_speed(source_duration, video_duration),
        "repeat_min": min(repeats) if repeats else None,
        "repeat_max": max(repeats) if repeats else None,
    }


def _trajectory_timestamps(report_path: Path) -> dict[int, float]:
    report = json.loads(report_path.read_text())
    candidates = [
        report_path.parent / "trajectory_full.jsonl",
        report_path.parent.parent / "trajectory_full.jsonl",
        report_path.parent.parent.parent / "trajectory_full.jsonl",
    ]
    profile = report.get("trajectory_quality_profile")
    if isinstance(profile, dict):
        path = profile.get("trajectory_path")
        if path:
            candidates.append(Path(path))
    for path in candidates:
        if not path.is_file():
            continue
        values: dict[int, float] = {}
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            values[int(item["id"])] = float(item["timestamp"])
        if values:
            return values
    raise ValueError(
        "cannot establish canonical source identity: report has no readable trajectory timestamp catalog"
    )


def _validate_rows(
    rows: list[ComparisonRow], refs_by_index: dict[int, SourceFrameRef], report_path: Path
) -> list[dict]:
    timestamps = _trajectory_timestamps(report_path)
    proofs: list[dict] = []
    for row in rows:
        ref = refs_by_index[row.source_ordinal]
        observed = timestamps.get(row.source_ordinal)
        if observed is None or abs(observed - ref.timestamp_s) > 1e-6:
            raise ValueError(
                f"canonical timestamp mismatch for source_index {row.source_ordinal}: "
                f"catalog={ref.timestamp_s!r}, report={observed!r}"
            )
        if row.status != "matched":
            continue
        source = next(iter_tum_rgb_frames([ref]))
        proof = compare_ground_truth_panel(_read_triptych(row), source)
        if proof["status"] != "exact":
            raise ValueError(
                f"ground-truth panel does not match source RGB for {row.render_id}: {proof}"
            )
        proofs.append({"render_id": row.render_id, **proof})
    return proofs


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--render-dir", type=Path,
                        help="existing triptych directory; omit for input-only output")
    parser.add_argument("--report", type=Path,
                        help="existing report.json; omit for input-only output")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=150,
                        help="input clip source-frame bound; 0 means all")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--comparison-max-matches", type=int, default=0,
                        help="cap matched rendered rows; 0 means all present rows")
    parser.add_argument("--comparison-max-encoded-frames", type=int, default=180,
                        help="hard cap for encoded comparison frames; 0 means uncapped")
    parser.add_argument("--cpu-threads", type=int, default=1)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.start_index < 0 or args.max_frames < 0 or args.fps <= 0:
        raise SystemExit("start-index/max-frames must be >=0 and fps must be >0")
    if args.comparison_max_matches < 0 or args.comparison_max_encoded_frames < 0:
        raise SystemExit("comparison caps must be >=0")
    if (args.render_dir is None) != (args.report is None):
        raise SystemExit("--render-dir and --report must be provided together")

    raw_refs = read_tum_rgb_index(args.source_root)
    clip_end = None if args.max_frames == 0 else args.start_index + args.max_frames
    input_refs = raw_refs[args.start_index:clip_end]
    if not input_refs:
        raise SystemExit("input clip selection is empty")

    rows: list[ComparisonRow] = []
    comparison_refs: list[SourceFrameRef] = []
    proofs: list[dict] = []
    available_rows: list[ComparisonRow] = []
    if args.report is not None and args.render_dir is not None:
        comparison_refs = read_canonical_tum_index(args.source_root, raw_refs=raw_refs)
        rows = build_comparison_manifest(comparison_refs, args.render_dir, args.report)
        available_rows = [row for row in rows if row.status == "matched"]
        present_rows = _select_comparison_rows(
            rows,
            max_matches=args.comparison_max_matches,
            encoded_cap=args.comparison_max_encoded_frames,
        )
        refs_by_index = {ref.source_ordinal: ref for ref in comparison_refs}
        proofs = _validate_rows(present_rows, refs_by_index, args.report)
    else:
        present_rows = []
        refs_by_index = {}

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    input_path = out_dir / "input_clip.mp4"
    comparison_path = out_dir / "input_vs_rendered_sparse.mp4"
    manifest_path = out_dir / "input_vs_rendered_sparse.manifest.json"
    for path in (input_path, manifest_path):
        if path.exists():
            raise SystemExit(f"refusing to overwrite existing output: {path}")
    if present_rows and comparison_path.exists():
        raise SystemExit(f"refusing to overwrite existing output: {comparison_path}")

    input_repeats = distribute_frame_repeats(
        [ref.timestamp_s for ref in input_refs], args.fps, max_encoded_frames=None
    )
    input_encoded = sum(input_repeats)
    error = write_mp4_cpu_stream(
        input_path, _repeat_stream(iter_tum_rgb_frames(input_refs), input_repeats),
        fps=args.fps, threads=args.cpu_threads,
    )
    if error is not None:
        raise SystemExit(f"input video encode failed: {error}")

    comparison_metadata: dict
    if present_rows:
        comparison_refs = [refs_by_index[row.source_ordinal] for row in present_rows]
        comparison_repeats = _bounded_repeats(
            [ref.timestamp_s for ref in comparison_refs], args.fps,
            args.comparison_max_encoded_frames,
        )
        comparison_encoded = sum(comparison_repeats)
        error = write_mp4_cpu_stream(
            comparison_path,
            _comparison_frames(present_rows, refs_by_index, comparison_repeats),
            fps=args.fps, threads=args.cpu_threads,
        )
        if error is not None:
            raise SystemExit(f"comparison video encode failed: {error}")
        comparison_metadata = {
            "path": str(comparison_path),
            "status": "written",
            "subset_semantics": "report_test_per_view rows with existing triptych PNGs, joined by canonical source_index",
            "temporal_compression": "repeated encoded frames preserve matched source order; playback_speed reports source-duration compression",
            "missing_render_ids": [row.render_id for row in rows if row.status == "missing_render"],
            "matched_render_ids": [row.render_id for row in present_rows],
            "available_matched_count": len(available_rows),
            "selected_match_count": len(present_rows),
            "selection_rule": "trajectory_spanning_even_indices",
            "selection_limits": {
                "comparison_max_matches": args.comparison_max_matches,
                "comparison_max_encoded_frames": args.comparison_max_encoded_frames,
            },
            "panel_semantics": "left=source RGB, right=first panel of render|GT|diff triptych",
            "ground_truth_proof": proofs,
            **_artifact_metadata(comparison_refs, fps=args.fps, encoded_frames=comparison_encoded, repeats=comparison_repeats),
        }
    else:
        comparison_metadata = {
            "path": None,
            "status": "not_requested" if args.report is None else "no_matches",
            "missing_render_ids": [row.render_id for row in rows if row.status == "missing_render"],
            "matched_render_ids": [],
            "available_matched_count": len(available_rows),
            "selected_match_count": 0,
        }

    with Image.open(input_refs[0].path) as first_image:
        source_shape = [first_image.height, first_image.width, 3]
    manifest = {
        "schema_version": 2,
        "purpose": "post-processing presentation artifact; not scientific evidence",
        "encoding": {"backend": "cpu_ffmpeg_libx264", "threads": max(1, args.cpu_threads)},
        "source": {
            "root": str(Path(args.source_root).resolve()),
            "index": "rgb.txt",
            "native_resolution_hw": source_shape[:2],
            "total_source_frames": len(raw_refs),
            "input_identity": "raw_presentation_index_only",
            "comparison_identity": "canonical_tum_association_index",
        },
        "input_clip": {
            "path": str(input_path),
            "start_index": args.start_index,
            "max_frames": args.max_frames,
            **_artifact_metadata(input_refs, fps=args.fps, encoded_frames=input_encoded, repeats=input_repeats),
        },
        "comparison_clip": comparison_metadata,
        "rows": [
            {
                "source_reference": {
                    "source_index": row.source_ordinal,
                    "timestamp_ns": refs_by_index[row.source_ordinal].timestamp_ns,
                    "rgb_rel": refs_by_index[row.source_ordinal].source_id,
                    "catalog": "tum_associated",
                    "raw_presentation_index": refs_by_index[row.source_ordinal].raw_presentation_index,
                },
                "render_id": row.render_id,
                "render_path": str(row.render_path) if row.render_path else None,
                "status": row.status,
                "selected": row in present_rows,
            }
            for row in rows
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"input_video": str(input_path), "comparison_video": comparison_metadata["path"], "manifest": str(manifest_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
