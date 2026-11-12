"""CPU contracts for the direct-input comparison-video exporter."""

from __future__ import annotations

import json
import io
import shutil
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from eligsir.utils.input_comparison_video import (
    build_comparison_manifest,
    compare_ground_truth_panel,
    extract_render_panel,
    iter_tum_rgb_frames,
    make_comparison_frame,
    read_canonical_tum_index,
    read_tum_rgb_index,
    _split_triptych,
)
import scripts.render_input_comparison_video as render_video
from eligsir.utils.input_comparison_video import ComparisonRow, SourceFrameRef
import eligsir.utils.video_encode as video_encode
from eligsir.utils.video_encode import write_mp4_cpu_stream


def _write_rgb(root: Path, index: int, timestamp: str, color: tuple[int, int, int]) -> None:
    rgb = root / "rgb"
    rgb.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 6), color).save(rgb / f"{timestamp}.png")


def _write_report(path: Path, rows: list[dict]) -> None:
    path.write_text(json.dumps({"test_per_view": rows}))


def test_tum_index_and_frame_iterator_preserve_source_identity(tmp_path: Path):
    _write_rgb(tmp_path, 0, "1.000000", (1, 2, 3))
    _write_rgb(tmp_path, 1, "1.033333", (4, 5, 6))
    (tmp_path / "rgb.txt").write_text(
        "# color images\n1.000000 rgb/1.000000.png\n1.033333 rgb/1.033333.png\n"
    )

    refs = read_tum_rgb_index(tmp_path)
    assert [r.source_ordinal for r in refs] == [0, 1]
    assert [r.timestamp_s for r in refs] == [1.0, 1.033333]
    assert [r.timestamp_ns for r in refs] == [1_000_000_000, 1_033_333_000]
    frames = list(iter_tum_rgb_frames(refs[:1]))
    assert frames[0].shape == (6, 8, 3)
    assert tuple(frames[0][0, 0]) == (1, 2, 3)


def test_triptych_geometry_is_validated_and_first_render_panel_is_extracted():
    source = np.zeros((6, 8, 3), dtype=np.uint8)
    render = np.full_like(source, 10)
    gt = np.full_like(source, 20)
    diff = np.full_like(source, 30)
    triptych = np.concatenate([render, np.zeros((6, 2, 3), dtype=np.uint8), gt,
                               np.zeros((6, 2, 3), dtype=np.uint8), diff], axis=1)
    assert np.array_equal(extract_render_panel(triptych, source.shape), render)
    comparison = make_comparison_frame(source, render)
    assert np.array_equal(comparison[24:30, 0:8], source)


def test_comparison_manifest_joins_by_source_index_and_preserves_missing_ids(tmp_path: Path):
    _write_rgb(tmp_path, 0, "1.000000", (1, 2, 3))
    _write_rgb(tmp_path, 1, "1.033333", (4, 5, 6))
    (tmp_path / "rgb.txt").write_text(
        "1.000000 rgb/1.000000.png\n1.033333 rgb/1.033333.png\n"
    )
    refs = read_tum_rgb_index(tmp_path)
    render_dir = tmp_path / "render"
    render_dir.mkdir()
    Image.new("RGB", (28, 6), (9, 9, 9)).save(render_dir / "000000.png")
    report = tmp_path / "report.json"
    _write_report(report, [{"name": "000000", "source_index": 0},
                           {"name": "000001", "source_index": 1}])

    rows = build_comparison_manifest(refs, render_dir, report)
    assert [row.status for row in rows] == ["matched", "missing_render"]
    assert rows[0].source_ordinal == 0
    assert rows[1].source_ordinal == 1


def test_comparison_manifest_rejects_name_source_index_mismatch(tmp_path: Path):
    _write_rgb(tmp_path, 0, "1.000000", (1, 2, 3))
    (tmp_path / "rgb.txt").write_text("1.000000 rgb/1.000000.png\n")
    refs = read_tum_rgb_index(tmp_path)
    render_dir = tmp_path / "render"
    render_dir.mkdir()
    Image.new("RGB", (28, 6), (9, 9, 9)).save(render_dir / "000001.png")
    report = tmp_path / "report.json"
    _write_report(report, [{"name": "000001", "source_index": 0}])

    try:
        build_comparison_manifest(refs, render_dir, report)
    except ValueError as exc:
        assert "source_index" in str(exc)
    else:
        raise AssertionError("mismatched render name/source index was accepted")


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="requires ffmpeg on PATH")
def test_cpu_encoder_consumes_frames_incrementally(tmp_path: Path):
    frames_seen = 0

    def frames():
        nonlocal frames_seen
        for value in range(4):
            frames_seen += 1
            yield np.full((6, 8, 3), value, dtype=np.uint8)

    output = tmp_path / "stream.mp4"
    assert write_mp4_cpu_stream(output, frames(), fps=4, threads=1) is None
    assert frames_seen == 4
    assert output.stat().st_size > 0


def test_cpu_encoder_does_not_prefetch_the_whole_iterable(monkeypatch, tmp_path: Path):
    state = {"yielded": 0, "writes": 0}

    def frames():
        for value in range(4):
            assert state["writes"] >= state["yielded"]
            state["yielded"] += 1
            yield np.full((6, 8, 3), value, dtype=np.uint8)

    class FakeStdin:
        def write(self, payload):
            state["writes"] += 1
            return len(payload)

        def close(self):
            return None

    class FakeProcess:
        stdin = FakeStdin()
        stderr = io.BytesIO()
        returncode = 0

        def wait(self):
            return 0

    monkeypatch.setattr(video_encode.subprocess, "Popen", lambda *a, **k: FakeProcess())
    assert write_mp4_cpu_stream(tmp_path / "stream.mp4", frames(), fps=4) is None
    assert state == {"yielded": 4, "writes": 4}


def test_canonical_tum_index_is_filtered_before_source_ordinal(tmp_path: Path):
    _write_rgb(tmp_path, 0, "1.000000", (1, 2, 3))
    _write_rgb(tmp_path, 1, "2.000000", (4, 5, 6))
    _write_rgb(tmp_path, 2, "3.000000", (7, 8, 9))
    (tmp_path / "rgb.txt").write_text(
        "1.000000 rgb/1.000000.png\n2.000000 rgb/2.000000.png\n3.000000 rgb/3.000000.png\n"
    )
    (tmp_path / "depth.txt").write_text(
        "1.001000 depth/1.png\n2.100000 depth/2.png\n3.001000 depth/3.png\n"
    )
    (tmp_path / "groundtruth.txt").write_text(
        "1.001000 0 0 0 0 0 0 1\n3.001000 0 0 0 0 0 0 1\n"
    )
    (tmp_path / "depth").mkdir()
    for name in ("1.png", "2.png", "3.png"):
        Image.new("I", (8, 6)).save(tmp_path / "depth" / name)

    refs = read_canonical_tum_index(tmp_path)
    assert [ref.source_ordinal for ref in refs] == [0, 1]
    assert [ref.source_id for ref in refs] == [
        "rgb/1.000000.png", "rgb/3.000000.png"
    ]


def test_ground_truth_panel_proof_is_pixel_exact():
    source = np.full((6, 8, 3), 20, dtype=np.uint8)
    triptych = np.concatenate(
        [np.zeros_like(source), np.zeros((6, 2, 3), dtype=np.uint8), source,
         np.zeros((6, 2, 3), dtype=np.uint8), np.zeros_like(source)], axis=1
    )
    proof = compare_ground_truth_panel(triptych, source)
    assert proof["status"] == "exact"
    assert proof["max_abs_diff"] == 0


def test_bounded_repeats_never_exceed_hard_cap():
    repeats = render_video._bounded_repeats([0.0, 0.1, 4.0], 30, 4)
    assert len(repeats) == 3
    assert sum(repeats) <= 4
    assert all(value >= 1 for value in repeats)


def test_bounded_repeats_uses_current_deficit_for_uniform_budget():
    repeats = render_video._bounded_repeats([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], 30, 12)
    assert repeats == [2, 2, 2, 2, 2, 2]


def test_comparison_cohort_spans_trajectory_and_respects_encoded_cap():
    rows = [
        ComparisonRow(i, Path(f"{i:06d}.png"), f"{i:06d}", "matched")
        for i in range(181)
    ]
    selected = render_video._select_comparison_rows(rows, max_matches=0, encoded_cap=180)
    assert len(selected) == 180
    assert selected[0].source_ordinal == 0
    assert selected[-1].source_ordinal == 180
    assert len({row.source_ordinal for row in selected}) == 180


def test_cli_does_not_decode_unselected_malformed_triptych(tmp_path: Path, monkeypatch):
    source_path = tmp_path / "source.png"
    source = np.full((6, 8, 3), 30, dtype=np.uint8)
    Image.fromarray(source).save(source_path)
    (tmp_path / "rgb.txt").write_text("1.000000 source.png\n")
    valid_triptych = np.concatenate(
        [np.zeros_like(source), np.zeros((6, 2, 3), dtype=np.uint8), source,
         np.zeros((6, 2, 3), dtype=np.uint8), np.zeros_like(source)], axis=1
    )
    valid_path = tmp_path / "valid.png"
    Image.fromarray(valid_triptych).save(valid_path)
    malformed_path = tmp_path / "malformed.png"
    malformed_path.write_bytes(b"not an image")
    refs = [SourceFrameRef(i, float(i + 1), (i + 1) * 1_000_000_000,
                           "source.png", source_path, 0) for i in range(181)]
    rows = [ComparisonRow(i, valid_path, f"{i:06d}", "matched") for i in range(181)]
    unselected = next(row for row in rows if row not in render_video._select_comparison_rows(
        rows, max_matches=0, encoded_cap=180
    ))
    rows[unselected.source_ordinal] = ComparisonRow(
        unselected.source_ordinal, malformed_path, unselected.render_id, "matched"
    )
    report = tmp_path / "report.json"
    report.write_text("{}")
    monkeypatch.setattr(render_video, "read_canonical_tum_index", lambda *a, **k: refs)
    monkeypatch.setattr(render_video, "build_comparison_manifest", lambda *a, **k: rows)
    monkeypatch.setattr(render_video, "_trajectory_timestamps", lambda path: {i: float(i + 1) for i in range(181)})

    def fake_encode(path, frames, **kwargs):
        list(frames)
        Path(path).write_bytes(b"test")
        return None

    monkeypatch.setattr(render_video, "write_mp4_cpu_stream", fake_encode)
    out = tmp_path / "out"
    assert render_video.main([
        "--source-root", str(tmp_path), "--render-dir", str(tmp_path),
        "--report", str(report), "--out-dir", str(out), "--max-frames", "1",
    ]) == 0
    manifest = json.loads((out / "input_vs_rendered_sparse.manifest.json").read_text())
    assert manifest["comparison_clip"]["available_matched_count"] == 181
    assert manifest["comparison_clip"]["selected_match_count"] == 180
    assert manifest["comparison_clip"]["encoded_frames"] == 180
    assert sum(1 for row in manifest["rows"] if row["selected"]) == 180


def test_triptych_split_returns_views_and_render_extraction_owns_copy():
    triptych = np.zeros((6, 28, 3), dtype=np.uint8)
    views = _split_triptych(triptych, (6, 8, 3))
    assert all(np.shares_memory(view, triptych) for view in views)
    render = extract_render_panel(triptych, (6, 8, 3))
    render[0, 0] = 255
    assert triptych[0, 0, 0] == 0


def test_input_only_cli_does_not_require_report_or_render(tmp_path: Path, monkeypatch):
    _write_rgb(tmp_path, 0, "1.000000", (1, 2, 3))
    _write_rgb(tmp_path, 1, "1.033333", (4, 5, 6))
    (tmp_path / "rgb.txt").write_text(
        "1.000000 rgb/1.000000.png\n1.033333 rgb/1.033333.png\n"
    )
    out = tmp_path / "out"

    def fake_encode(path, frames, **kwargs):
        list(frames)
        Path(path).write_bytes(b"test")
        return None

    monkeypatch.setattr(render_video, "write_mp4_cpu_stream", fake_encode)
    assert render_video.main(["--source-root", str(tmp_path), "--out-dir", str(out)]) == 0
    manifest = json.loads((out / "input_vs_rendered_sparse.manifest.json").read_text())
    assert (out / "input_clip.mp4").is_file()
    assert manifest["comparison_clip"]["status"] == "not_requested"


def test_cli_joins_filtered_canonical_ordinal_and_enforces_comparison_cap(
    tmp_path: Path, monkeypatch
):
    for index, timestamp, color in (
        (0, "1.000000", (10, 10, 10)),
        (1, "2.000000", (20, 20, 20)),
        (2, "3.000000", (30, 30, 30)),
    ):
        _write_rgb(tmp_path, index, timestamp, color)
    (tmp_path / "rgb.txt").write_text(
        "1.000000 rgb/1.000000.png\n2.000000 rgb/2.000000.png\n3.000000 rgb/3.000000.png\n"
    )
    (tmp_path / "depth").mkdir()
    (tmp_path / "depth.txt").write_text(
        "1.001000 depth/1.png\n2.100000 depth/2.png\n3.001000 depth/3.png\n"
    )
    for name in ("1.png", "2.png", "3.png"):
        Image.new("I", (8, 6)).save(tmp_path / "depth" / name)
    (tmp_path / "groundtruth.txt").write_text(
        "1.001000 0 0 0 0 0 0 1\n3.001000 0 0 0 0 0 0 1\n"
    )
    render_dir = tmp_path / "render"
    render_dir.mkdir()
    # Canonical ordinal 1 is raw RGB ordinal 2.  The GT panel proves the CLI
    # did not silently use raw rgb.txt ordinal 1.
    source = np.full((6, 8, 3), 30, dtype=np.uint8)
    triptych = np.concatenate(
        [np.zeros_like(source), np.zeros((6, 2, 3), dtype=np.uint8), source,
         np.zeros((6, 2, 3), dtype=np.uint8), np.zeros_like(source)], axis=1
    )
    Image.fromarray(triptych).save(render_dir / "000001.png")
    report_dir = tmp_path / "report" / "iter"
    report_dir.mkdir(parents=True)
    report = report_dir / "report.json"
    _write_report(report, [{"name": "000001", "source_index": 1}])
    (tmp_path / "trajectory_full.jsonl").write_text(
        '{"id":1,"timestamp":3.0}\n'
    )
    counts = {}

    def fake_encode(path, frames, **kwargs):
        counts[Path(path).name] = sum(1 for _ in frames)
        Path(path).write_bytes(b"test")
        return None

    monkeypatch.setattr(render_video, "write_mp4_cpu_stream", fake_encode)
    out = tmp_path / "out"
    assert render_video.main([
        "--source-root", str(tmp_path), "--render-dir", str(render_dir),
        "--report", str(report), "--out-dir", str(out), "--max-frames", "1",
        "--comparison-max-encoded-frames", "1",
    ]) == 0
    manifest = json.loads((out / "input_vs_rendered_sparse.manifest.json").read_text())
    assert manifest["rows"][0]["source_reference"]["raw_presentation_index"] == 2
    assert manifest["comparison_clip"]["encoded_frames"] == 1
    assert counts["input_vs_rendered_sparse.mp4"] == 1


def test_encoder_failure_cleans_up_and_reports_bounded_stderr(monkeypatch, tmp_path: Path):
    class FakeStdin:
        def write(self, payload):
            return len(payload)

        def close(self):
            return None

    class FakeProcess:
        def __init__(self):
            self.stdin = FakeStdin()
            self.stderr = io.BytesIO(b"encoder failed\n" * 1000)
            self.returncode = 7

        def wait(self, timeout=None):
            return self.returncode

        def kill(self):
            return None

    monkeypatch.setattr(video_encode.subprocess, "Popen", lambda *a, **k: FakeProcess())
    error = write_mp4_cpu_stream(
        tmp_path / "failed.mp4", [np.zeros((6, 8, 3), dtype=np.uint8)]
    )
    assert error is not None
    assert "code 7" in error
    assert len(error) < 2500
