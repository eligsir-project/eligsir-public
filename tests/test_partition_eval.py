"""Unit tests for deterministic partitioned evaluation (utils/comparison_report.py).

CPU-only; no rendering. Covers the partition-id function, source-index
extraction from synthetic cameras, determinism under shuffling, clamping,
and the aggregation into partition_psnr / min / std.
"""
import random
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from eligsir.utils.comparison_report import (
    PARTITIONS,
    camera_source_index,
    compute_partition_stats,
    partition_id,
)


class FakeCam:
    def __init__(self, uid=None, colmap_id=None, image_name=""):
        if uid is not None:
            self.uid = uid
        if colmap_id is not None:
            self.colmap_id = colmap_id
        self.image_name = image_name


def test_partitions_constant():
    assert PARTITIONS == 6


def test_partition_id_basic_bins():
    max_idx = 5269  # floor3: n_source_frames=5270
    assert partition_id(0, max_idx) == 0
    assert partition_id(max_idx, max_idx) == PARTITIONS - 1
    # floor(6 * i / max) matches the contract exactly
    for i in (1, 878, 879, 1756, 2635, 4391, 5000):
        expected = min(PARTITIONS - 1, (PARTITIONS * i) // max_idx)
        assert partition_id(i, max_idx) == expected


def test_partition_id_clamped_to_k_minus_1():
    # source index beyond max (shouldn't happen, but must clamp not overflow)
    assert partition_id(10_000, 5269) == PARTITIONS - 1
    assert partition_id(5269, 5269) == PARTITIONS - 1


def test_partition_id_degenerate_denominator():
    assert partition_id(3, 0) == 0
    assert partition_id(3, -1) == 0
    assert partition_id(0, 100) == 0


def test_camera_source_index_prefers_uid():
    assert camera_source_index(FakeCam(uid=42, image_name="000007")) == 42


def test_camera_source_index_colmap_id_fallback():
    assert camera_source_index(FakeCam(colmap_id=17)) == 17


def test_camera_source_index_image_name_fallback():
    # e.g. '001504' as seen in contact sheets
    assert camera_source_index(FakeCam(image_name="001504")) == 1504
    assert camera_source_index(FakeCam(image_name="frame_0032.png")) == 32
    assert camera_source_index(FakeCam(image_name="no_digits_here")) is None


def test_determinism_across_shuffled_order():
    max_idx = 999
    cams = [FakeCam(uid=i, image_name=f"{i:06d}") for i in range(0, 1000, 7)]
    ref = {c.uid: partition_id(camera_source_index(c), max_idx) for c in cams}
    for seed in (1, 2, 3):
        shuffled = list(cams)
        random.Random(seed).shuffle(shuffled)
        got = {c.uid: partition_id(camera_source_index(c), max_idx) for c in shuffled}
        assert got == ref


def test_compute_partition_stats():
    max_idx = 59
    per_view = []
    # 10 views per partition, partition p gets PSNR 20+p
    for i in range(60):
        p = partition_id(i, max_idx)
        per_view.append({"partition": p, "psnr": 20.0 + p})
    stats = compute_partition_stats(per_view)
    assert set(stats["partition_psnr"]) == {f"p{i}" for i in range(PARTITIONS)}
    assert stats["partition_psnr_min"] == 20.0
    means = [stats["partition_psnr"][f"p{i}"] for i in range(PARTITIONS)]
    assert all(m is not None for m in means)
    assert stats["partition_psnr_std"] > 0
    assert sum(stats["partition_n"].values()) == 60


def test_compute_partition_stats_empty_bins():
    stats = compute_partition_stats([{"partition": 0, "psnr": 21.0}])
    assert stats["partition_psnr"]["p0"] == 21.0
    assert stats["partition_psnr"]["p5"] is None
    assert stats["partition_n"]["p5"] == 0
    assert stats["partition_psnr_min"] == 21.0
    assert stats["partition_psnr_std"] == 0.0
    # views with missing partition/psnr are ignored
    stats2 = compute_partition_stats([{"partition": None, "psnr": 1.0},
                                      {"partition": 2, "psnr": None}])
    assert stats2["partition_psnr_min"] is None


def test_helper_import_parity_across_call_paths():
    """The comparison report and the in-training sentinel must bin identically:
    both import THE SAME helper objects from eligsir.utils.partition_eval."""
    from eligsir.utils import partition_eval as pe
    from eligsir.utils import comparison_report as cr

    assert cr.PARTITIONS == pe.PARTITIONS
    assert cr.camera_source_index is pe.camera_source_index
    assert cr.partition_id is pe.partition_id
    assert cr.compute_partition_stats is pe.compute_partition_stats

    max_idx = 5269
    cams = [FakeCam(uid=i, image_name=f"{i:06d}") for i in range(0, 5270, 97)]
    for c in cams:
        src_cr = cr.camera_source_index(c)
        src_pe = pe.camera_source_index(c)
        assert src_cr == src_pe
        assert cr.partition_id(src_cr, max_idx) == pe.partition_id(src_pe, max_idx)


def test_resolve_max_source_index_fallback():
    from eligsir.utils.partition_eval import resolve_max_source_index

    cams = [FakeCam(uid=i) for i in (3, 99, 42)]
    max_src, source = resolve_max_source_index("/not/a/registered/scene", cams)
    assert source == "fallback_max_cams"
    assert max_src == 99
    max_src2, source2 = resolve_max_source_index(None, [])
    assert source2 == "fallback_max_cams"
    assert max_src2 == 0


def test_partition_count_scales_with_trajectory_length():
    from eligsir.utils.partition_eval import partition_count, MIN_PARTITIONS, MAX_PARTITIONS

    # Long trajectory (ScanNet++-scale): far finer than the old fixed 6.
    assert partition_count(7040) == 58

    # Short trajectory: clamped to the floor, not fewer.
    assert partition_count(500) == MIN_PARTITIONS

    # Very long trajectory: clamped to the ceiling, not unbounded.
    assert partition_count(50000) == MAX_PARTITIONS

    # Unknown/unregistered scene: same fallback as the old fixed constant.
    assert partition_count(None) == MIN_PARTITIONS
    assert partition_count(0) == MIN_PARTITIONS


def test_partition_count_respects_custom_frames_per_partition():
    from eligsir.utils.partition_eval import partition_count

    assert partition_count(1200, frames_per_partition=100) == 12
    assert partition_count(1200, frames_per_partition=600) == 6  # clamped to min


def test_partition_index_tensor_matches_per_cam_partition_id():
    import torch
    from eligsir.utils.partition_eval import partition_index_tensor

    max_idx = 999
    cams = [FakeCam(uid=i, image_name=f"{i:06d}") for i in range(0, 1000, 37)]
    k = 10
    expected = [partition_id(camera_source_index(c), max_idx, k=k) for c in cams]
    got = partition_index_tensor(cams, max_idx, k=k)
    assert got.dtype == torch.long
    assert got.tolist() == expected


def test_partition_index_tensor_unresolvable_source_falls_back_to_zero():
    from eligsir.utils.partition_eval import partition_index_tensor

    cams = [FakeCam(uid=None, image_name="no_digits_here"), FakeCam(uid=5, image_name="000005")]
    got = partition_index_tensor(cams, 999, k=6)
    assert got.tolist()[0] == 0
    assert len(got) == len(cams)
