import json
import numpy as np
import pytest

from scripts.render_bag_checkpoint import load_test_poses, sanity_check


def _write_test_pose_files(tmp_path, rows):
    """rows: list of (name, source_index, timestamp, xyz)"""
    traj_dir = tmp_path / "trajectory_eval"
    traj_dir.mkdir()
    with open(traj_dir / "trajectory_test_tum.txt", "w") as f:
        f.write("# TUM trajectory — t tx ty tz qx qy qz qw\n")
        for _, _, ts, xyz in rows:
            f.write(f"{ts:.9f} {xyz[0]:.9f} {xyz[1]:.9f} {xyz[2]:.9f} 0.0 0.0 0.0 1.0\n")
    index = [{"name": name, "source_index": idx} for name, idx, _, _ in rows]
    (traj_dir / "trajectory_test_index.json").write_text(json.dumps(index))
    return str(tmp_path)


def test_load_test_poses_reads_matching_tum_and_index(tmp_path):
    model_path = _write_test_pose_files(tmp_path, [
        ("001504", 1504, 1001.5, [1.0, 2.0, 3.0]),
        ("003008", 3008, 1003.0, [4.0, 5.0, 6.0]),
    ])

    poses = load_test_poses(model_path)

    assert len(poses) == 2
    assert poses[0]["name"] == "001504"
    assert poses[0]["source_index"] == 1504
    assert poses[0]["timestamp"] == pytest.approx(1001.5)
    assert poses[0]["c2w"].shape == (4, 4)
    assert poses[0]["c2w"][:3, 3] == pytest.approx([1.0, 2.0, 3.0])


def test_sanity_check_passes_on_identical_matching_views():
    rendered = [{"name": "001504", "source_index": 1504, "psnr": 23.0},
                {"name": "003008", "source_index": 3008, "psnr": 20.0}]
    report = [{"name": "001504", "source_index": 1504, "psnr": 23.02},
              {"name": "003008", "source_index": 3008, "psnr": 19.95}]

    result = sanity_check(rendered, report, psnr_tol=0.1, max_mismatches=2)

    assert result["ok"] is True
    assert result["mismatches"] == []


def test_sanity_check_fails_on_missing_view():
    rendered = [{"name": "001504", "source_index": 1504, "psnr": 23.0}]
    report = [{"name": "001504", "source_index": 1504, "psnr": 23.02},
              {"name": "003008", "source_index": 3008, "psnr": 19.95}]

    result = sanity_check(rendered, report, psnr_tol=0.1, max_mismatches=2)

    assert result["ok"] is False
    assert any(m["name"] == "003008" for m in result["mismatches"])


def test_sanity_check_fails_when_too_many_psnr_mismatches():
    rendered = [{"name": "a", "source_index": 1, "psnr": 10.0},
                {"name": "b", "source_index": 2, "psnr": 10.0},
                {"name": "c", "source_index": 3, "psnr": 10.0}]
    report = [{"name": "a", "source_index": 1, "psnr": 20.0},
              {"name": "b", "source_index": 2, "psnr": 20.0},
              {"name": "c", "source_index": 3, "psnr": 20.0}]

    result = sanity_check(rendered, report, psnr_tol=0.1, max_mismatches=2)

    assert result["ok"] is False
    assert len(result["mismatches"]) == 3


def test_sanity_check_tolerates_up_to_max_mismatches():
    rendered = [{"name": "a", "source_index": 1, "psnr": 10.0},
                {"name": "b", "source_index": 2, "psnr": 20.0},
                {"name": "c", "source_index": 3, "psnr": 20.0}]
    report = [{"name": "a", "source_index": 1, "psnr": 20.0},
              {"name": "b", "source_index": 2, "psnr": 20.0},
              {"name": "c", "source_index": 3, "psnr": 20.0}]

    result = sanity_check(rendered, report, psnr_tol=0.1, max_mismatches=2)

    assert result["ok"] is True
    assert len(result["mismatches"]) == 1
