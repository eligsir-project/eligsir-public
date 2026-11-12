"""Tests for packed calibration submission in prepare_ready_calibration().

Validates that the calibration tensor submission has been optimized from
25 scalar device assignments to a single copy_ per tensor, while maintaining
bit-exact equivalence with the original implementation.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

torch = pytest.importorskip("torch")

from eligsir.runtime.transfers import DeviceTransferStream


def _reference_fill(pose, camera_intrinsics, device):
    """Original element-wise fill implementation for parity testing."""
    c2w = torch.empty((4, 4), dtype=torch.float32, device=device)
    intrinsics = torch.empty((3, 3), dtype=torch.float32, device=device)
    for row_index, row in enumerate(pose):
        for column_index, value in enumerate(row):
            c2w[row_index, column_index] = value
    for row_index, row in enumerate(camera_intrinsics):
        for column_index, value in enumerate(row):
            intrinsics[row_index, column_index] = value
    return c2w, intrinsics


def test_prepare_ready_calibration_parity_identity_matrices():
    """Verify packed submission equals element-wise fill for identity pose/intrinsics.

    Also validates dtype and shape are correct.
    """
    stream = DeviceTransferStream("cpu")

    pose = (
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )
    intrinsics = (
        (1.0, 0.0, 0.5),
        (0.0, 1.0, 0.5),
        (0.0, 0.0, 1.0),
    )

    # Packed implementation (new).
    payloads_new, _ = stream.prepare_ready_calibration(
        pose=pose,
        camera_intrinsics=intrinsics,
    )
    c2w_new, intrinsics_new = payloads_new

    # Reference element-wise fill (old).
    c2w_ref, intrinsics_ref = _reference_fill(pose, intrinsics, "cpu")

    # Parity check: bit-exact equality.
    assert torch.equal(c2w_new, c2w_ref)
    assert torch.equal(intrinsics_new, intrinsics_ref)

    # Shape and dtype validation.
    assert c2w_new.shape == (4, 4)
    assert c2w_new.dtype == torch.float32
    assert intrinsics_new.shape == (3, 3)
    assert intrinsics_new.dtype == torch.float32


def test_prepare_ready_calibration_parity_difficult_floats():
    """Verify packed submission matches element-wise fill on difficult-to-represent values.

    Tests values that are not exactly representable in float32:
    - 0.1 (repeating binary)
    - 1/3 (repeating binary)
    - 1e-8 (near denormal boundary)
    - 1e20 (large exponent)
    - negative values
    - NaN (special handling in both narrowing paths)
    """
    stream = DeviceTransferStream("cpu")

    pose = (
        (0.1, 1.0 / 3.0, 1e-8, 0.0),
        (0.0, -0.1, 1e20, 0.5),
        (1.0 / 3.0, 0.0, -1.0, 0.1),
        (0.0, 0.0, 0.0, 1.0),
    )
    intrinsics = (
        (1000.5, 0.0, 320.1),
        (0.0, 1000.7, 240.3),
        (0.0, 0.0, 1.0),
    )

    # Packed implementation (new).
    payloads_new, _ = stream.prepare_ready_calibration(
        pose=pose,
        camera_intrinsics=intrinsics,
    )
    c2w_new, intrinsics_new = payloads_new

    # Reference element-wise fill (old).
    c2w_ref, intrinsics_ref = _reference_fill(pose, intrinsics, "cpu")

    # Parity check: bit-exact equality (single narrowing during tensor construction).
    assert torch.equal(c2w_new, c2w_ref), (
        f"c2w mismatch:\nNew:\n{c2w_new}\nRef:\n{c2w_ref}\n"
        f"Diff:\n{(c2w_new - c2w_ref).abs()}"
    )
    assert torch.equal(intrinsics_new, intrinsics_ref), (
        f"intrinsics mismatch:\nNew:\n{intrinsics_new}\nRef:\n{intrinsics_ref}\n"
        f"Diff:\n{(intrinsics_new - intrinsics_ref).abs()}"
    )

    # NaN case: verify NaN bit patterns match through both narrowing paths.
    nan_pose = (
        (float("nan"), 0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )
    nan_intrinsics = (
        (1.0, 0.0, 0.5),
        (0.0, 1.0, 0.5),
        (0.0, 0.0, 1.0),
    )

    payloads_nan, _ = stream.prepare_ready_calibration(
        pose=nan_pose,
        camera_intrinsics=nan_intrinsics,
    )
    c2w_nan, _ = payloads_nan

    c2w_nan_ref, _ = _reference_fill(nan_pose, nan_intrinsics, "cpu")

    # Use torch.testing.assert_close with equal_nan=True to verify NaN propagation.
    torch.testing.assert_close(c2w_nan, c2w_nan_ref, equal_nan=True)


def test_prepare_ready_calibration_submission_count():
    """Verify exactly 2 copy_() calls and 0 __setitem__() calls on device tensors.

    This test patches torch.Tensor.copy_() and torch.Tensor.__setitem__() to count
    invocations during prepare_ready_calibration(), ensuring we've eliminated the
    25 scalar device element assignments.
    """
    stream = DeviceTransferStream("cpu")

    pose = (
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )
    intrinsics = (
        (1.0, 0.0, 0.5),
        (0.0, 1.0, 0.5),
        (0.0, 0.0, 1.0),
    )

    # Spy on copy_() and __setitem__() calls.
    copy_count = [0]
    setitem_count = [0]

    original_copy_ = torch.Tensor.copy_
    original_setitem = torch.Tensor.__setitem__

    def counting_copy_(self, src, **kwargs):
        copy_count[0] += 1
        return original_copy_(self, src, **kwargs)

    def counting_setitem(self, key, value):
        setitem_count[0] += 1
        return original_setitem(self, key, value)

    # Patch at the class level to count all calls.
    with patch.object(torch.Tensor, "copy_", counting_copy_, autospec=False):
        with patch.object(torch.Tensor, "__setitem__", counting_setitem, autospec=False):
            payloads, _ = stream.prepare_ready_calibration(
                pose=pose,
                camera_intrinsics=intrinsics,
            )

    c2w, intrinsics_out = payloads

    # Verify submission counts: exactly 2 copy_() (one per tensor), 0 __setitem__().
    assert copy_count[0] == 2, (
        f"Expected exactly 2 copy_() calls (one per tensor), got {copy_count[0]}"
    )
    assert setitem_count[0] == 0, (
        f"Expected 0 __setitem__() calls on device tensors, got {setitem_count[0]}"
    )

    # Sanity check: tensors are valid.
    assert c2w.shape == (4, 4)
    assert intrinsics_out.shape == (3, 3)


def test_prepare_ready_calibration_stats_unchanged():
    """Verify transfer stats behavior is preserved (no new h2d_ops/bytes added)."""
    stream = DeviceTransferStream("cpu")

    pose = (
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )
    intrinsics = (
        (1.0, 0.0, 0.5),
        (0.0, 1.0, 0.5),
        (0.0, 0.0, 1.0),
    )

    initial_stats = stream.stats.snapshot()

    stream.prepare_ready_calibration(
        pose=pose,
        camera_intrinsics=intrinsics,
    )

    final_stats = stream.stats.snapshot()

    # prepare_ready_calibration does not use h2d() or d2h_into(),
    # so these counters should not change.
    assert final_stats["h2d_ops"] == initial_stats["h2d_ops"]
    assert final_stats["h2d_bytes"] == initial_stats["h2d_bytes"]
    assert final_stats["d2h_ops"] == initial_stats["d2h_ops"]
    assert final_stats["d2h_bytes"] == initial_stats["d2h_bytes"]


def test_prepare_ready_calibration_validation():
    """Verify input validation still works."""
    stream = DeviceTransferStream("cpu")

    # Invalid pose shape.
    with pytest.raises(ValueError, match="4x4"):
        stream.prepare_ready_calibration(
            pose=((1.0, 0.0, 0.0),) * 4,  # 3x4 instead of 4x4
            camera_intrinsics=((1.0, 0.0, 0.5), (0.0, 1.0, 0.5), (0.0, 0.0, 1.0)),
        )

    # Invalid intrinsics shape.
    with pytest.raises(ValueError, match="3x3"):
        stream.prepare_ready_calibration(
            pose=(
                (1.0, 0.0, 0.0, 0.0),
                (0.0, 1.0, 0.0, 0.0),
                (0.0, 0.0, 1.0, 0.0),
                (0.0, 0.0, 0.0, 1.0),
            ),
            camera_intrinsics=((1.0, 0.0, 0.5), (0.0, 1.0, 0.5)),  # 2x3 instead of 3x3
        )
