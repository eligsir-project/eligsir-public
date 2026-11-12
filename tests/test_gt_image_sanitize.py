"""Regression test for the GT-image NaN-sanitize guard in train_streaming.py.

streaming_training() is a large, side-effecting monolith not directly
unit-testable; this test isolates the exact sanitize pattern used at the
"---- Loss ----" block (poisoned ground-truth frame -> permanent in-place
nan_to_num repair) rather than invoking the full training loop.
"""

import torch


class _FakeCamera:
    def __init__(self, original_image, image_name="000610"):
        self.original_image = original_image
        self.image_name = image_name


def _sanitize_gt_image_if_needed(cam):
    """Mirrors the guard added in train_streaming.py's loss block."""
    gt_image = cam.original_image
    if not bool(torch.isfinite(gt_image).all()):
        gt_image = torch.nan_to_num(gt_image, nan=0.0, posinf=1.0, neginf=0.0)
        cam.original_image = gt_image.detach()
    return gt_image


def test_poisoned_frame_is_sanitized_in_place():
    poisoned = torch.rand(3, 8, 8)
    poisoned[0, 2, 3] = float("nan")
    cam = _FakeCamera(poisoned)

    out = _sanitize_gt_image_if_needed(cam)

    assert torch.isfinite(out).all()
    assert torch.isfinite(cam.original_image).all()


def test_sanitize_is_permanent_across_resamples():
    poisoned = torch.rand(3, 4, 4)
    poisoned[0, 0, 0] = float("inf")
    cam = _FakeCamera(poisoned)

    _sanitize_gt_image_if_needed(cam)
    # Second "resample" of the same camera must see the already-clean tensor
    # and take the no-op branch (nothing left to sanitize).
    out2 = _sanitize_gt_image_if_needed(cam)

    assert torch.isfinite(out2).all()


def test_clean_frame_is_untouched():
    clean = torch.rand(3, 4, 4)
    cam = _FakeCamera(clean)

    out = _sanitize_gt_image_if_needed(cam)

    assert torch.equal(out, clean)
    assert cam.original_image is clean
