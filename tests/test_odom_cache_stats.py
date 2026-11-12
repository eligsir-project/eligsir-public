"""The Open3D odometry cache must round-trip per-frame registration stats, so the
input-quality tracking metric survives cache hits (a cached estimated-pose run must
still report tracking_available=True, not silently drop _odom_stats)."""

from __future__ import annotations

import numpy as np

from eligsir.utils.streaming_frames import OrbbecRosBagFrameSource as O


def test_odom_cache_roundtrips_per_frame_stats(tmp_path):
    poses = [np.eye(4, dtype=np.float32) for _ in range(3)]
    stats = {
        "successes": 2, "failures": 1,
        "per_frame": [
            {"valid": True, "fitness": 0.80, "inlier_rmse": 0.02, "translation_m": 0.1},
            {"valid": True, "fitness": 0.60, "inlier_rmse": 0.04, "translation_m": 0.2},
            {"valid": False, "fitness": 0.0, "inlier_rmse": None, "translation_m": 0.0},
        ],
    }
    p = tmp_path / "odom.npz"
    O._save_open3d_odom_cache(p, "key1", poses, stats)

    loaded_poses, loaded_stats = O._load_open3d_odom_cache(p, "key1", expected_frames=3)
    assert loaded_poses is not None and len(loaded_poses) == 3
    pf = loaded_stats["per_frame"]
    assert len(pf) == 3
    assert pf[0]["fitness"] == 0.80 and abs(pf[1]["inlier_rmse"] - 0.04) < 1e-9
    assert pf[2]["valid"] is False and pf[2]["inlier_rmse"] is None
    assert loaded_stats["cache_hit"] is True


def test_cache_key_mismatch_ignored(tmp_path):
    poses = [np.eye(4, dtype=np.float32) for _ in range(2)]
    p = tmp_path / "odom.npz"
    O._save_open3d_odom_cache(p, "keyA", poses, {"per_frame": [None, None]})
    lp, ls = O._load_open3d_odom_cache(p, "keyB", expected_frames=2)  # wrong key
    assert lp is None and ls is None


def test_old_format_cache_is_rejected(tmp_path):
    # A pre-versioning cache (no cache_format field) must be treated as a miss so it is
    # recomputed and rewritten with per-frame stats — otherwise tracking silently stays off.
    import json
    p = tmp_path / "old.npz"
    np.savez_compressed(
        p, cache_key=np.asarray("keyA"),
        poses=np.stack([np.eye(4, dtype=np.float32)] * 2, axis=0),
        stats_json=np.asarray(json.dumps({"successes": 1, "failures": 0})),
    )
    lp, ls = O._load_open3d_odom_cache(p, "keyA", expected_frames=2)
    assert lp is None and ls is None
