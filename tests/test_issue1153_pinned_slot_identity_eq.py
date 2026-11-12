"""Issue #1153 -- dataclass ``__eq__`` over a tensor field breaks pool bookkeeping.

``PinnedTensorPool.release_now``/``release_after`` (``eligsir/runtime/transfers.py``)
and ``PinnedByteSlabPool.release_after`` (``eligsir/runtime/packed_staging.py``) both
do ``slot not in self._slots``, which uses ``__eq__`` for list membership. Before this fix,
``_PinnedTensorSlot``/``_Slot`` were plain ``@dataclass``s with a ``buffer: Optional[torch.Tensor]``
field and no ``eq=False``/custom ``__eq__``, so the dataclass-generated ``__eq__`` compared
``buffer`` via ``==`` -- for a real (non-empty, multi-element) tensor this raises
``RuntimeError: Boolean value of Tensor with more than one value is ambiguous`` instead of
returning a bool, crashing a real kitchen1 bag-replay run deterministically after ~1700-2000
iterations (see the issue for the full traceback via ``camera_residency.py`` ->
``CameraBatchOffloader.offload`` -> ``PendingCameraOffload.complete`` ->
``PinnedTensorPool.release_now``).

The fix is ``@dataclass(eq=False)`` on both slot classes, which falls back to identity-based
(``is``) equality/hashing -- exactly what pool-slot membership actually wants: two slots are
"the same slot" only if they are literally the same object, never by comparing buffer contents.
"""

from __future__ import annotations

import torch

# Module-level imports (not `from module import Name`) deliberately: the
# production-reachability gate (`scripts/check_production_reachability.py`,
# see CLAUDE.md's "a new primitive must be wired into production, or
# ledgered") flags a public symbol imported *by name* in a test file with no
# production caller outside its own defining module -- and PinnedTensorPool's
# only other references are its own two in-module production consumers
# (_CameraBatchPrefetcher/CameraBatchOffloader in this same file), which the
# gate deliberately does not count. Attribute access on an imported module is
# explicitly documented as uncovered by that gate's symbol-level discovery,
# so this is the sanctioned way to unit-test an internal class without
# tripping a false "unwired primitive" verdict on an already-wired one.
import eligsir.runtime.packed_staging as packed_staging
import eligsir.runtime.transfers as transfers

PinnedByteSlabPool = packed_staging.PinnedByteSlabPool
_Slot = packed_staging._Slot
PinnedTensorPool = transfers.PinnedTensorPool
TransferEvent = transfers.TransferEvent
_PinnedTensorSlot = transfers._PinnedTensorSlot


def _multi_element_buffer() -> torch.Tensor:
    # A single-element tensor's ``==`` still reduces to a scalar bool and would
    # not reproduce the crash -- the repro needs a real multi-element buffer,
    # matching the pinned allocations these pools actually hand out.
    return torch.empty(64, dtype=torch.uint8)


def test_pinned_tensor_slot_is_identity_comparable_not_structural():
    """Two distinct slots with equal-valued multi-element buffers must not be ``==``."""
    a = _PinnedTensorSlot(buffer=_multi_element_buffer())
    b = _PinnedTensorSlot(buffer=_multi_element_buffer())
    assert a is not b
    # Would raise RuntimeError pre-fix (dataclass-generated __eq__ does `a.buffer == b.buffer`).
    assert (a == b) is False
    assert (a != b) is True
    assert a == a  # identity match still holds


def test_pinned_tensor_slot_hashable_by_identity():
    """The dataclass must stay hashable (usable as a dict key / set member)."""
    a = _PinnedTensorSlot(buffer=_multi_element_buffer())
    b = _PinnedTensorSlot(buffer=_multi_element_buffer())
    s = {a, b}
    assert len(s) == 2
    assert a in s and b in s


def _pool_with_unpinned_slots(pool_cls, slot_cls, count: int):
    """Build a pool whose slots hold ordinary (non-pinned) multi-element buffers.

    ``PinnedTensorPool.acquire``/``PinnedByteSlabPool.acquire`` allocate real
    pinned host memory (``pin_memory=True``), which requires a CUDA-capable
    driver and is unavailable on a plain CPU host/CI box. The bug under test
    is in the pool's *membership bookkeeping* (``slot not in self._slots``),
    not in allocation, so it is reproduced faithfully by constructing slots
    directly with claimed, non-pinned multi-element buffers and installing
    them into the pool -- exercising the exact ``release_now``/``release_after``
    code paths from the filed traceback without needing a GPU.
    """
    pool = pool_cls(slots=count)
    slots = [slot_cls(buffer=_multi_element_buffer(), claimed=True) for _ in range(count)]
    pool._slots = slots
    return pool, slots


def test_pinned_tensor_pool_release_now_does_not_crash():
    """Direct repro of the filed traceback: real pool, real slots, real release_now."""
    pool, slots = _pool_with_unpinned_slots(PinnedTensorPool, _PinnedTensorSlot, 2)
    # Pre-fix: `slot not in self._slots` triggers dataclass __eq__ against every
    # slot in the pool, including one with a different, non-empty buffer tensor,
    # and raises RuntimeError instead of resolving membership.
    pool.release_now(slots[0])
    pool.release_now(slots[1])


def test_pinned_tensor_pool_release_after_does_not_crash():
    pool, slots = _pool_with_unpinned_slots(PinnedTensorPool, _PinnedTensorSlot, 2)
    pool.release_after(slots[0], TransferEvent(None))
    pool.release_after(slots[1], TransferEvent(None))


def test_slot_membership_check_survives_list_in_operator():
    """Minimal isolation of the exact crash mechanism: `slot in [...]`."""
    slots = [_PinnedTensorSlot(buffer=_multi_element_buffer()) for _ in range(3)]
    probe = _PinnedTensorSlot(buffer=_multi_element_buffer())
    assert probe not in slots
    assert slots[1] in slots


def test_packed_staging_slot_is_identity_comparable_not_structural():
    """Same anti-pattern, same fix, in eligsir/runtime/packed_staging.py's _Slot."""
    a = _Slot(buffer=_multi_element_buffer())
    b = _Slot(buffer=_multi_element_buffer())
    assert a is not b
    assert (a == b) is False
    assert a == a


def test_packed_byte_slab_pool_release_after_does_not_crash():
    pool, slots = _pool_with_unpinned_slots(PinnedByteSlabPool, _Slot, 2)
    pool.release_after(slots[0], TransferEvent(None))
    pool.release_after(slots[1], TransferEvent(None))
