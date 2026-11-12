"""``PinnedByteSlabPool`` slots are identities, not values.

Regression test for a real production crash: ``release_after`` checks
``slot not in self._slots``, and Python's ``in`` falls back to ``==`` for every
element that is not the same object. ``_Slot`` was a plain ``@dataclass``, so
that generated an ``__eq__`` comparing the ``buffer`` field -- a tensor --
which raises rather than returning a bool.

The first transfer through a fresh pool always succeeded (the acquired slot is
found by identity before any other slot has a buffer to compare against), so
this was invisible in any single-transfer test. The SECOND transfer acquired a
different slot, hit a populated slot earlier in the list, and crashed with
``Boolean value of Tensor with more than one value is ambiguous`` or a shape
mismatch. The pool is module-level and per-device, so in a real streaming run
every legacy-NumPy insertion flush after the first one would have died.
"""

import pytest

torch = pytest.importorskip("torch")

from eligsir.runtime.packed_staging import PinnedByteSlabPool


class _DoneEvent:
    def query(self):
        return True

    def synchronize(self):
        return None


def test_release_after_accepts_every_slot_across_repeated_acquires():
    """Exercise more acquires than the pool has slots, so slot objects differ
    and each release must still match by identity."""
    pool = PinnedByteSlabPool(slots=4)
    for i in range(12):
        # Vary the size so any value-based comparison would ALSO hit a shape
        # mismatch, not just the ambiguous-truth error.
        slot = pool.acquire(64 + i * 8)
        pool.release_after(slot, _DoneEvent())
    assert pool.exhaustions == 0


def test_slot_membership_does_not_compare_buffers():
    pool = PinnedByteSlabPool(slots=4)
    first = pool.acquire(128)
    pool.release_after(first, _DoneEvent())
    second = pool.acquire(256)
    # Before the fix this raised instead of returning normally.
    pool.release_after(second, _DoneEvent())
    assert first is not second


def test_foreign_slot_is_still_rejected():
    """eq=False must not weaken the guard it protects."""
    pool = PinnedByteSlabPool(slots=2)
    other = PinnedByteSlabPool(slots=2)
    foreign = other.acquire(64)
    with pytest.raises(ValueError):
        pool.release_after(foreign, _DoneEvent())
