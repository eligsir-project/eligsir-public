from eligsir.utils.partition_health import detect_outlier_partitions


def test_no_outliers_when_all_similar():
    psnr = {"p0": 22.1, "p1": 21.8, "p2": 22.5, "p3": 21.9, "p4": 22.0, "p5": 22.3}
    assert detect_outlier_partitions(psnr) == set()


def test_flags_single_low_outlier():
    psnr = {"p0": 21.5, "p1": 12.3, "p2": 23.1, "p3": 21.5, "p4": 22.0, "p5": 20.9}
    assert detect_outlier_partitions(psnr) == {1}


def test_flags_multiple_low_outliers():
    psnr = {"p0": 21.0, "p1": 8.0, "p2": 21.5, "p3": 9.0, "p4": 20.5, "p5": 21.2}
    assert detect_outlier_partitions(psnr) == {1, 3}


def test_ignores_missing_and_non_numeric_entries():
    psnr = {"p0": 21.0, "p1": "--", "p2": None, "p3": 9.0, "p4": 20.5, "p5": 21.2}
    # p1/p2 excluded from the stat entirely, not treated as 0.
    assert detect_outlier_partitions(psnr) == {3}


def test_scene_relative_not_absolute_threshold():
    """A uniformly hard scene where every partition is ~13dB must NOT flag
    anything -- there's no genuine outlier, just a globally difficult scene."""
    psnr = {"p0": 12.9, "p1": 13.1, "p2": 13.4, "p3": 12.7, "p4": 13.0}
    assert detect_outlier_partitions(psnr, k=3.0) == set()


def test_too_few_partitions_refuses_to_flag():
    psnr = {"p0": 22.0, "p1": 8.0}
    assert detect_outlier_partitions(psnr, min_partitions=3) == set()


def test_identical_values_no_spread_flags_nothing():
    psnr = {"p0": 20.0, "p1": 20.0, "p2": 20.0, "p3": 20.0}
    assert detect_outlier_partitions(psnr) == set()


def test_k_sensitivity():
    psnr = {"p0": 20.0, "p1": 14.0, "p2": 21.0, "p3": 19.5, "p4": 20.5}
    # A lenient (high) k flags only the clear dip; a strict (low) k also
    # sweeps in a borderline point (p3) that a looser k tolerates.
    assert detect_outlier_partitions(psnr, k=0.3) == {1, 3}
    assert detect_outlier_partitions(psnr, k=1.0) == {1}
    assert detect_outlier_partitions(psnr, k=5.0) == {1}


def test_integer_keys_also_supported():
    psnr = {0: 21.0, 1: 8.0, 2: 21.5, 3: 20.5, 4: 21.2}
    assert detect_outlier_partitions(psnr) == {1}
