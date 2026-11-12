"""Fail-closed supervision batch bounds shared by live runtime entrypoints."""

from __future__ import annotations


BOUNDED_SUPERVISION_BATCH_SIZES = (1, 2, 3, 4, 6, 8, 16, 32, 64, 128, 256)
QUALIFIED_SUPERVISION_CAMERA_COUNTS = (1, 2, 4, 8, 16, 32, 64)


def validate_supervision_batch_size(value: int) -> int:
    """Return a bounded candidate batch size or reject unqualified values."""

    if isinstance(value, bool):
        raise ValueError("supervision batch size must be an integer")
    batch_size = int(value)
    if batch_size not in BOUNDED_SUPERVISION_BATCH_SIZES:
        allowed = ", ".join(str(item) for item in BOUNDED_SUPERVISION_BATCH_SIZES)
        raise ValueError(f"supervision batch size must be one of {allowed} (got {batch_size})")
    return batch_size


def validate_supervision_camera_count(value: int) -> int:
    """Return a qualified true-camera target or fail closed.

    Camera concurrency is deliberately narrower than the optimizer batch
    budget.  Larger B values are realized as multiple C-sized microbatches;
    accepting an arbitrary C here would silently bypass the qualified ladder.
    """

    if isinstance(value, bool):
        raise ValueError("qualified supervision camera count must be an integer")
    camera_count = int(value)
    if camera_count not in QUALIFIED_SUPERVISION_CAMERA_COUNTS:
        allowed = ", ".join(str(item) for item in QUALIFIED_SUPERVISION_CAMERA_COUNTS)
        raise ValueError(
            "qualified supervision camera count must be one of "
            f"{allowed} (got {camera_count})"
        )
    return camera_count


def qualified_camera_count_for_batch(batch_size: int) -> int:
    """Resolve the largest qualified C not exceeding requested B."""

    requested = validate_supervision_batch_size(batch_size)
    return max(
        item for item in QUALIFIED_SUPERVISION_CAMERA_COUNTS if item <= requested
    )


__all__ = [
    "BOUNDED_SUPERVISION_BATCH_SIZES",
    "QUALIFIED_SUPERVISION_CAMERA_COUNTS",
    "qualified_camera_count_for_batch",
    "validate_supervision_batch_size",
    "validate_supervision_camera_count",
]
