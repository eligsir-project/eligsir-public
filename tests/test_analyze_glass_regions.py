import json
import math
import pytest

from scripts.analyze_glass_regions import classify_glass_regions, main


def _view(idx, psnr):
    return {"name": f"{idx:06d}", "source_index": idx, "partition": 0, "psnr": psnr}


def test_splits_glass_and_nonglass_by_source_index():
    per_view = [
        _view(10, 24.0),   # non-glass
        _view(20, 26.0),   # non-glass
        _view(1500, 18.0), # glass
        _view(2000, 20.0), # glass
        _view(4600, 22.0), # non-glass
    ]
    result = classify_glass_regions(per_view, glass_ranges=[(1481, 4533)])

    assert result["glass_n"] == 2
    assert result["glass_psnr_mean"] == pytest.approx(19.0)
    assert result["nonglass_n"] == 3
    assert result["nonglass_psnr_mean"] == pytest.approx((24.0 + 26.0 + 22.0) / 3)
    assert result["overall_n"] == 5


def test_excludes_views_with_missing_source_index():
    per_view = [
        {"name": "x", "source_index": None, "partition": None, "psnr": 99.0},
        _view(10, 24.0),
    ]
    result = classify_glass_regions(per_view, glass_ranges=[(1481, 4533)])

    assert result["overall_n"] == 1
    assert result["nonglass_n"] == 1
    assert result["glass_n"] == 0
    assert result["glass_psnr_mean"] is None


def test_upper_bound_uses_assumed_perfect_psnr_for_glass():
    per_view = [
        _view(10, 20.0),   # non-glass
        _view(1500, 10.0), # glass
    ]
    result = classify_glass_regions(per_view, glass_ranges=[(1481, 4533)],
                                     assumed_perfect_psnr=40.0)

    # 1 glass view assumed at 40.0, 1 non-glass view at 20.0 -> mean 30.0
    assert result["upper_bound_psnr_if_glass_fixed"] == pytest.approx(30.0)
    assert result["assumed_perfect_psnr"] == 40.0


def test_upper_bound_is_none_when_no_nonglass_views():
    per_view = [_view(1500, 10.0)]
    result = classify_glass_regions(per_view, glass_ranges=[(1481, 4533)])

    assert result["nonglass_n"] == 0
    assert result["upper_bound_psnr_if_glass_fixed"] is None


def test_multiple_glass_ranges_supported():
    per_view = [_view(100, 20.0), _view(2000, 10.0), _view(6000, 12.0)]
    result = classify_glass_regions(per_view, glass_ranges=[(1481, 4533), (5900, 6500)])

    assert result["glass_n"] == 2
    assert result["nonglass_n"] == 1


def _write_report(tmp_path, eval_valid=True, n_test=2, per_view=None):
    if per_view is None:
        per_view = [
            {"name": "a", "source_index": 10, "partition": 0, "psnr": 24.0},
            {"name": "b", "source_index": 2000, "partition": 1, "psnr": 18.0},
        ]
    report = {"eval_valid": eval_valid, "n_test": n_test, "test_per_view": per_view}
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report))
    return str(path)


def test_cli_writes_output_file_and_returns_zero(tmp_path, capsys):
    report_path = _write_report(tmp_path)
    rc = main([report_path, "--glass-range", "1481:4533"])

    assert rc == 0
    out_path = tmp_path / "glass_vs_nonglass_psnr.json"
    assert out_path.exists()
    written = json.loads(out_path.read_text())
    assert written["glass_n"] == 1
    assert written["nonglass_n"] == 1

    captured = capsys.readouterr()
    assert "glass:" in captured.out
    assert "non-glass:" in captured.out


def test_cli_refuses_invalid_eval_report(tmp_path, capsys):
    report_path = _write_report(tmp_path, eval_valid=False, n_test=0, per_view=[])
    rc = main([report_path, "--glass-range", "1481:4533"])

    assert rc == 2
    assert not (tmp_path / "glass_vs_nonglass_psnr.json").exists()
    captured = capsys.readouterr()
    assert "eval_valid" in captured.err or "not a valid eval" in captured.err


def test_cli_output_includes_holdout_identity_fields(tmp_path):
    per_view = [
        {"name": "a", "source_index": 10, "partition": 0, "psnr": 24.0},
        {"name": "b", "source_index": 2000, "partition": 1, "psnr": 18.0},
    ]
    report = {
        "eval_valid": True,
        "n_test": 2,
        "test_per_view": per_view,
        "eval_metadata": {"holdout_hash": "abc123"},
        "scene_source": "floor3",
        "partition_max_source_index": 5270,
    }
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report))

    rc = main([str(report_path), "--glass-range", "1481:4533"])
    assert rc == 0

    written = json.loads((tmp_path / "glass_vs_nonglass_psnr.json").read_text())
    assert written["holdout_hash"] == "abc123"
    assert written["scene_source"] == "floor3"
    assert written["partition_max_source_index"] == 5270


def test_cli_output_identity_fields_none_when_absent(tmp_path):
    report_path = _write_report(tmp_path)
    rc = main([report_path, "--glass-range", "1481:4533"])
    assert rc == 0

    written = json.loads((tmp_path / "glass_vs_nonglass_psnr.json").read_text())
    assert written["holdout_hash"] is None
    assert written["scene_source"] is None
    assert written["partition_max_source_index"] is None


def test_cli_supports_custom_out_path_and_multiple_ranges(tmp_path):
    report_path = _write_report(tmp_path)
    custom_out = tmp_path / "custom.json"
    rc = main([
        report_path,
        "--glass-range", "1481:4533",
        "--glass-range", "9000:9500",
        "--out", str(custom_out),
    ])

    assert rc == 0
    assert custom_out.exists()
