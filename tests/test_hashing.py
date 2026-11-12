"""Tests for eligsir.experiments.hashing's binary-content primitives.

Covers the P0 fix: content_hash() must never silently fall back to
str(value) for numpy arrays or other unsupported types -- see
eligsir/experiments/hashing.py's module docstring.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from eligsir.experiments.hashing import content_hash, hash_bytes, hash_file, hash_ndarray


class TestHashBytes:
    def test_deterministic(self):
        data = b"\x00\x01\x02frame-content"
        assert hash_bytes(data) == hash_bytes(data)

    def test_content_sensitive(self):
        assert hash_bytes(b"abc") != hash_bytes(b"abd")

    def test_is_sha256_hex(self):
        digest = hash_bytes(b"abc")
        assert len(digest) == 64
        int(digest, 16)


class TestHashFile:
    def test_matches_hash_bytes(self, tmp_path):
        data = b"some binary rgbd payload" * 1000
        path = tmp_path / "frame.bin"
        path.write_bytes(data)
        assert hash_file(path) == hash_bytes(data)

    def test_content_sensitive(self, tmp_path):
        path_a = tmp_path / "a.bin"
        path_b = tmp_path / "b.bin"
        path_a.write_bytes(b"aaaa")
        path_b.write_bytes(b"aaab")
        assert hash_file(path_a) != hash_file(path_b)

    def test_large_file_streamed(self, tmp_path):
        # Larger than the internal chunk size, to exercise the streaming loop.
        data = b"x" * (2 * 1024 * 1024 + 137)
        path = tmp_path / "big.bin"
        path.write_bytes(data)
        assert hash_file(path) == hash_bytes(data)


class TestHashNdarray:
    def test_deterministic_regardless_of_object_identity(self):
        arr1 = np.arange(100, dtype=np.uint8).reshape(10, 10)
        arr2 = np.arange(100, dtype=np.uint8).reshape(10, 10).copy()
        assert arr1 is not arr2
        assert hash_ndarray(arr1) == hash_ndarray(arr2)

    def test_content_sensitive_single_element(self):
        arr1 = np.zeros((4, 4), dtype=np.float32)
        arr2 = arr1.copy()
        arr2[2, 2] = 1.0
        assert hash_ndarray(arr1) != hash_ndarray(arr2)

    def test_shape_sensitive(self):
        arr1 = np.zeros((4, 4), dtype=np.uint8)
        arr2 = np.zeros((16,), dtype=np.uint8)
        assert hash_ndarray(arr1) != hash_ndarray(arr2)

    def test_dtype_sensitive(self):
        arr1 = np.zeros((4,), dtype=np.float32)
        arr2 = np.zeros((4,), dtype=np.float64)
        assert hash_ndarray(arr1) != hash_ndarray(arr2)

    def test_non_contiguous_array_matches_contiguous_equivalent(self):
        base = np.arange(20, dtype=np.int32).reshape(4, 5)
        non_contig = base[:, ::2]  # a view, not C-contiguous
        assert not non_contig.flags["C_CONTIGUOUS"]
        contig_copy = np.ascontiguousarray(non_contig)
        assert hash_ndarray(non_contig) == hash_ndarray(contig_copy)


class TestContentHashRoutesBinaryTypes:
    def test_content_hash_ndarray_is_byte_exact_not_repr(self):
        # A large array's repr gets abbreviated with "..." by numpy -- if
        # content_hash fell back to str(), two arrays differing only in the
        # truncated middle would hash identically. Verify that's NOT so.
        arr1 = np.zeros((2000,), dtype=np.uint8)
        arr2 = arr1.copy()
        arr2[1000] = 7  # falls inside numpy's repr truncation window
        assert content_hash({"depth": arr1}) != content_hash({"depth": arr2})

    def test_content_hash_ndarray_deterministic(self):
        arr = np.arange(50, dtype=np.float64)
        assert content_hash({"a": arr}) == content_hash({"a": arr.copy()})

    def test_content_hash_bytes_routes_through_hash_bytes(self):
        data = b"raw-frame-bytes"
        result = content_hash({"payload": data})
        assert result == content_hash({"payload": data})
        assert result != content_hash({"payload": b"different"})

    def test_content_hash_raises_on_unsupported_type(self):
        class Unsupported:
            pass

        with pytest.raises(TypeError):
            content_hash({"a": Unsupported()})

    def test_content_hash_raises_names_the_type(self):
        class Weird:
            pass

        with pytest.raises(TypeError, match="Weird"):
            content_hash(Weird())

    def test_content_hash_accepts_path_as_stable_string_identity(self):
        assert content_hash({"out": Path("/tmp/report.json")}) == content_hash(
            {"out": Path("/tmp/report.json")}
        )
        assert content_hash({"out": Path("/tmp/a.json")}) != content_hash({"out": Path("/tmp/b.json")})
