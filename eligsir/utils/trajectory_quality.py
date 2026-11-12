"""Metric trajectory-domain quality profiles for mapping phase reports."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from eligsir.plot_style import svg_output_path

def _trajectory_rows(path: Path) -> list[tuple[int, np.ndarray]]:
    rows: list[tuple[int, np.ndarray]] = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        ordinal = row.get("id", row.get("source_ordinal"))
        position = row.get("position")
        if position is None and row.get("c2w") is not None:
            matrix = np.asarray(row["c2w"], dtype=np.float64)
            if matrix.shape != (4, 4):
                raise ValueError(f"invalid c2w at trajectory line {line_number}")
            position = matrix[:3, 3]
        point = np.asarray(position, dtype=np.float64)
        if ordinal is None or point.shape != (3,) or not np.isfinite(point).all():
            raise ValueError(
                f"invalid trajectory identity/position at line {line_number}"
            )
        rows.append((int(ordinal), point))
    if not rows:
        raise ValueError("trajectory quality profile requires at least one pose")
    rows.sort(key=lambda item: item[0])
    if len({ordinal for ordinal, _ in rows}) != len(rows):
        raise ValueError("trajectory source identities must be unique")
    return rows


def write_trajectory_quality_profile(
    *,
    trajectory_path: str | Path,
    per_view: Iterable[dict[str, Any]],
    output_path: str | Path,
    segment_size_m: float | None = None,
) -> dict[str, Any]:
    """Plot scored holdout PSNR by cumulative dataset-camera translation.

    The alternating spans are contiguous metric path segments.  They are not
    claimed to be independent scene partitions: a camera can observe geometry
    outside its own path segment, so the metadata names them explicitly as
    trajectory spatial segments.
    """

    trajectory_path = Path(trajectory_path)
    output_path = Path(output_path)
    rows = _trajectory_rows(trajectory_path)
    ordinals = [ordinal for ordinal, _ in rows]
    positions = np.stack([point for _, point in rows])
    step_distances = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    distances = np.concatenate((np.zeros(1), np.cumsum(step_distances)))
    by_ordinal = {
        ordinal: (float(distance), positions[index])
        for index, (ordinal, distance) in enumerate(zip(ordinals, distances))
    }
    total_distance = float(distances[-1])
    if segment_size_m is None:
        segment_size_m = max(0.5, total_distance / 8.0) if total_distance else 0.5
    segment_size_m = float(segment_size_m)
    if not math.isfinite(segment_size_m) or segment_size_m <= 0.0:
        raise ValueError("segment_size_m must be finite and positive")

    scored_views = []
    for view in per_view:
        source_index = view.get("source_index")
        if source_index is None or int(source_index) not in by_ordinal:
            raise ValueError(
                f"scored view source_index={source_index!r} is absent from trajectory"
            )
        psnr_db = float(view["psnr"])
        if not math.isfinite(psnr_db):
            raise ValueError(f"non-finite PSNR for source_index={source_index}")
        distance_m, center = by_ordinal[int(source_index)]
        scored_views.append(
            {
                "source_index": int(source_index),
                "distance_m": distance_m,
                "camera_center_xyz_m": [float(value) for value in center],
                "psnr_db": psnr_db,
            }
        )
    scored_views.sort(key=lambda item: (item["distance_m"], item["source_index"]))
    if not scored_views:
        raise ValueError("trajectory quality profile requires scored views")

    segment_count = max(1, int(math.ceil(total_distance / segment_size_m)))
    segments = []
    for segment_id in range(segment_count):
        start = segment_id * segment_size_m
        end = (
            total_distance
            if segment_id == segment_count - 1
            else min(total_distance, (segment_id + 1) * segment_size_m)
        )
        members = [
            view
            for view in scored_views
            if view["distance_m"] >= start
            and (
                view["distance_m"] < end
                or (segment_id == segment_count - 1 and view["distance_m"] <= end)
            )
        ]
        midpoint = 0.5 * (start + end)
        nearest_index = int(np.argmin(np.abs(distances - midpoint)))
        segments.append(
            {
                "segment_id": segment_id,
                "path_span_m": [float(start), float(end)],
                "path_midpoint_camera_xyz_m": [
                    float(value) for value in positions[nearest_index]
                ],
                "n_scored_views": len(members),
                "mean_psnr_db": (
                    None
                    if not members
                    else float(np.mean([view["psnr_db"] for view in members]))
                ),
            }
        )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(11, 5.5))
    for segment in segments:
        start, end = segment["path_span_m"]
        axis.axvspan(
            start, end, color="C0", alpha=0.04 + 0.04 * (segment["segment_id"] % 2)
        )
        axis.text(
            0.5 * (start + end),
            0.98,
            f"S{segment['segment_id']}",
            ha="center",
            va="top",
            transform=axis.get_xaxis_transform(),
            fontsize=8,
            color="0.35",
        )
    axis.plot(
        [view["distance_m"] for view in scored_views],
        [view["psnr_db"] for view in scored_views],
        marker="o",
        linewidth=1.2,
        alpha=0.72,
        label="frozen holdout view",
    )
    segment_points = [
        segment for segment in segments if segment["mean_psnr_db"] is not None
    ]
    if segment_points:
        axis.step(
            [0.5 * sum(segment["path_span_m"]) for segment in segment_points],
            [segment["mean_psnr_db"] for segment in segment_points],
            where="mid",
            linewidth=2.2,
            color="C1",
            label="trajectory-segment mean",
        )
    axis.set_xlabel("cumulative camera trajectory traveled (m)")
    axis.set_ylabel("test PSNR (dB)")
    axis.set_title("Frozen-holdout PSNR along the metric camera trajectory")
    axis.grid(True, alpha=0.25)
    axis.legend(loc="best", fontsize=8)
    fig.tight_layout()
    output_path = svg_output_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, metadata={"Date": None})
    plt.close(fig)

    return {
        "distance_domain": "cumulative_dataset_camera_translation_m",
        "trajectory_path": str(trajectory_path),
        "plot": str(output_path),
        "total_trajectory_distance_m": total_distance,
        "segment_size_m": segment_size_m,
        "scored_views": scored_views,
        "segments": segments,
    }


__all__ = ["write_trajectory_quality_profile"]
