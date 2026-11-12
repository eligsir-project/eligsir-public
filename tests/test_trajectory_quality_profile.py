from __future__ import annotations

import json
import importlib.util


def _writer():
    assert importlib.util.find_spec("utils.trajectory_quality") is not None, (
        "trajectory-distance quality publication is not implemented"
    )
    from eligsir.utils.trajectory_quality import write_trajectory_quality_profile

    return write_trajectory_quality_profile


def test_quality_profile_uses_metric_camera_path_and_marks_segments(tmp_path):
    trajectory = tmp_path / "trajectory_full.jsonl"
    poses = (
        (0, (0.0, 0.0, 0.0)),
        (1, (1.0, 0.0, 0.0)),
        (2, (1.0, 2.0, 0.0)),
        (3, (4.0, 2.0, 0.0)),
    )
    with trajectory.open("w") as handle:
        for ordinal, xyz in poses:
            matrix = [
                [1.0, 0.0, 0.0, xyz[0]],
                [0.0, 1.0, 0.0, xyz[1]],
                [0.0, 0.0, 1.0, xyz[2]],
                [0.0, 0.0, 0.0, 1.0],
            ]
            handle.write(json.dumps({"source_ordinal": ordinal, "c2w": matrix}) + "\n")

    result = _writer()(
        trajectory_path=trajectory,
        per_view=(
            {"source_index": 0, "psnr": 20.0},
            {"source_index": 1, "psnr": 21.0},
            {"source_index": 2, "psnr": 22.0},
            {"source_index": 3, "psnr": 23.0},
        ),
        output_path=tmp_path / "trajectory_psnr_by_distance.svg",
        segment_size_m=3.0,
    )

    assert result["distance_domain"] == "cumulative_dataset_camera_translation_m"
    assert result["total_trajectory_distance_m"] == 6.0
    assert [point["distance_m"] for point in result["scored_views"]] == [
        0.0,
        1.0,
        3.0,
        6.0,
    ]
    assert [segment["path_span_m"] for segment in result["segments"]] == [
        [0.0, 3.0],
        [3.0, 6.0],
    ]
    assert result["segments"][0]["mean_psnr_db"] == 20.5
    assert result["segments"][1]["mean_psnr_db"] == 22.5
    assert (tmp_path / "trajectory_psnr_by_distance.svg").stat().st_size > 0


def test_quality_profile_rejects_unknown_scored_source_identity(tmp_path):
    trajectory = tmp_path / "trajectory_full.jsonl"
    trajectory.write_text(
        json.dumps(
            {
                "source_ordinal": 0,
                "c2w": [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
            }
        )
        + "\n"
    )

    try:
        _writer()(
            trajectory_path=trajectory,
            per_view=({"source_index": 9, "psnr": 20.0},),
            output_path=tmp_path / "plot.png",
        )
    except ValueError as exc:
        assert "source_index=9" in str(exc)
    else:
        raise AssertionError("unknown scored view identity was silently omitted")
