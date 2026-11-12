"""Unit tests for `tests/conftest.py`'s report-diff helper (issue #544).

Deliberately torch-free so they run in the lean CPU CI image: the helper
itself is the thing that made two report-equality tests pass on a host and
in CPU CI while failing in the training image, so the helper needs coverage
in the surface that actually runs automatically.
"""
from __future__ import annotations

import math

from conftest import (
    REPORT_OUTPUT_DIR_PLACEHOLDER,
    normalize_report_paths,
    report_key_diff,
    report_values_equal,
)


def test_paths_into_the_reports_own_output_dir_compare_equal():
    a = {"output_dir": "/tmp/aaa", "train_contact_sheet": "/tmp/aaa/comparison/iter_100/s.png"}
    b = {"output_dir": "/tmp/bbb", "train_contact_sheet": "/tmp/bbb/comparison/iter_100/s.png"}
    assert report_key_diff(a, "/tmp/aaa", b, "/tmp/bbb") == []


def test_a_real_difference_in_the_rest_of_the_path_still_fails():
    """The point of normalizing rather than excluding the key."""
    a = {"train_contact_sheet": "/tmp/aaa/comparison/iter_100/s.png"}
    b = {"train_contact_sheet": "/tmp/bbb/comparison/iter_200/s.png"}
    assert report_key_diff(a, "/tmp/aaa", b, "/tmp/bbb") == ["train_contact_sheet"]


def test_nested_paths_are_normalized_too():
    a = {"artifacts": {"sheets": ["/tmp/aaa/x.png"], "n": 1}}
    b = {"artifacts": {"sheets": ["/tmp/bbb/x.png"], "n": 1}}
    assert report_key_diff(a, "/tmp/aaa", b, "/tmp/bbb") == []


def test_nan_compares_equal_to_itself():
    a = {"partition_psnr_std": float("nan")}
    b = {"partition_psnr_std": float("nan")}
    assert report_key_diff(a, "/tmp/aaa", b, "/tmp/bbb") == []
    assert report_values_equal(float("nan"), float("nan"))
    assert not report_values_equal(float("nan"), 1.0)


def test_diff_is_sorted_and_complete():
    a = {"z": 1, "a": 1, "m": 1}
    b = {"z": 2, "a": 2, "m": 1}
    assert report_key_diff(a, "/tmp/aaa", b, "/tmp/bbb") == ["a", "z"]


def test_a_key_present_on_only_one_side_is_reported():
    assert report_key_diff({"extra": 1}, "/tmp/aaa", {}, "/tmp/bbb") == ["extra"]


def test_ignore_set_is_honored():
    a = {"gpu_coresidency": {"t": 1}}
    b = {"gpu_coresidency": {"t": 2}}
    assert report_key_diff(a, "/tmp/aaa", b, "/tmp/bbb",
                           ignore=("gpu_coresidency",)) == []


def test_normalization_leaves_unrelated_strings_alone():
    out = normalize_report_paths({"scene": "/data/fr3", "output_dir": "/tmp/aaa"},
                                 "/tmp/aaa")
    assert out["scene"] == "/data/fr3"
    assert out["output_dir"] == REPORT_OUTPUT_DIR_PLACEHOLDER


def test_normalization_handles_non_string_leaves():
    out = normalize_report_paths({"n": 3, "x": None, "f": math.inf}, "/tmp/aaa")
    assert out == {"n": 3, "x": None, "f": math.inf}
