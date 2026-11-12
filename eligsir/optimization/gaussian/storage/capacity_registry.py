"""Fixed-capacity Gaussian map storage and topology epochs (W3-11, redesigned W3-49).

Implements plan §16: replace frequent ``torch.cat``-based growth and
parameter-object replacement in the core Gaussian map storage with a
capacity-backed allocator, plus an explicit mutation-barrier / topology-epoch
API that gates when the map's "structural meaning" (a monotonically
increasing ``topology_version``) is allowed to change.

W3-49 (2026-08-10) REPLACED this module's original design (arbitrary
non-contiguous active rows via a boolean mask + LIFO free-slot list) with a
packed-contiguous-active-prefix design, after W3-36's audit found the
original design gave the optimizer a full-capacity ``nn.Parameter`` (Adam
cost scaling with capacity, not live count -- upstream gsplat issue #444's
exact trap) and its ``active_view()`` read path did a boolean-mask gather
(real work) every call instead of a zero-copy slice. Both costs were
measured for real against the original design before this rewrite (6.92x
optimizer-step cost at a 20x capacity/active ratio; 0.057-0.098ms/call
gather cost) -- see
docs/architecture/2026-08-10-w3-49-capacity-scaling-baseline.md -- and the
central mechanism this redesign depends on (``nn.Parameter(backing[:n])``
aliasing the backing tensor's storage, surviving a rebind, working with both
``torch.optim.Adam`` and ``gsplat.optimizers.SelectiveAdam``) was verified
empirically before writing this file -- see
docs/architecture/2026-08-10-w3-49-parameter-slice-aliasing-validation.md.
Zero production call sites existed for the original design (confirmed by
W3-36), so this is a full in-place replacement, not a parallel/dual-path
migration -- see
docs/architecture/2026-08-10-w3-49-packed-prefix-redesign.md for the
acceptance-gate benchmark results this rewrite was measured against.

**Correction (2026-08-10, W3-52, post-#78-audit)**: the packed-prefix
rewrite above shipped with a real autograd-correctness bug, found by
external review of PRs #57-#78 before task #32 (production wiring) could
proceed: ``active_view()`` originally returned ``self.buffers[name][:E]``
unconditionally, including for param-backed names -- a fresh slice of the
PLAIN backing tensor, which never has ``requires_grad``, not the actual
live ``nn.Parameter`` in ``self._live_params[name]`` (a separate tensor
object aliasing the same storage). Any renderer using ``active_view()`` on
a param-backed buffer would silently get a non-differentiable view --
gradients would never reach the optimizer's Parameter. Fixed by making
``active_view()`` dispatch to ``active_parameter()`` (== ``param_for()``)
for param-backed names; see that method's docstring and
``test_active_view_of_param_backed_buffer_is_differentiable`` for the
regression test. The compaction acceptance-gate benchmark
(``scripts/bench_capacity_registry.py``'s ``run_compaction_cost()``) also
did not exercise the real ``CapacityBufferRegistry.retire()`` path (it
called the lower-level ``SlotAllocator.retire()`` directly and skipped the
``sync_momentum_from_optimizer()`` step retire() itself performs before
compaction), so its "buffer+momentum movement stays under ~0.15ms" number
did not include real momentum-sync cost -- fixed to benchmark the actual
``reg.retire()`` path. See
docs/architecture/2026-08-10-w3-52-p0-readiness-fixes.md for the full
writeup. Neither bug had a production consequence (zero call sites existed
for either fix), but both were real defects in already-merged code that
had to be closed before task #32 could treat this registry as ready to
wire into a differentiable training loop.

Three quantities, not two
--------------------------

The API is deliberately built around three distinct quantities, even though
this PR's default policy makes two of them always equal:

``storage_capacity`` (``C``, :attr:`SlotAllocator.capacity`)
    Total physical backing-storage rows. Grows via a single, explicit,
    logged ``torch.cat``-based reallocation (the sanctioned growth
    fallback) -- never a per-insert/per-retire cost.

``epoch_capacity`` (``E``, :attr:`SlotAllocator.epoch_capacity`)
    The optimizer- and render-visible shape for the CURRENT stable
    mutation epoch -- what every ``rebind()``/``active_view()`` call
    actually slices to. ``E >= N`` always. **This PR's policy is the
    identity function, ``E == N``** -- every call site threads
    ``epoch_capacity`` through (not ``n_active`` directly) so a future
    bucketing *policy* (e.g. round ``N`` up to a fixed compiled shape, so
    ordinary mapping mutations that keep ``N`` within the same bucket
    never change ``E`` and never invalidate a captured CUDA/HIP graph)
    avoids ANOTHER storage-layer redesign -- ``epoch_capacity_policy`` is
    already the seam a bucketing policy would plug into. That is narrower
    than "a one-line change," though: bucketing ``E > N`` means padding
    rows ``[N, E)`` become visible to both the renderer and the optimizer
    as though they were real Gaussians, and this module defines no
    contract yet for what those padding rows should look like (opacity,
    an active mask, gradient masking, ``SelectiveAdam`` visibility,
    evidence buffers, lifecycle state, rasterizer inclusion) -- task #34
    will need to design and implement that contract, not just flip
    ``epoch_capacity_policy`` to a non-identity function. Deliberately NOT
    implemented in this PR -- see the "Static epoch readiness" section of
    the design doc for why bucketing is left to the later graph/compile
    work, not mixed into this correctness fix.

``logical_live_count`` (``N``, :attr:`SlotAllocator.n_active`)
    The real number of live Gaussians. Physical rows ``[0, N)`` are always
    the contiguous, packed active prefix -- retiring an arbitrary interior
    row moves a donor row in from the current tail via
    :class:`CompactionPlan` (batched swap-delete, cost bounded by the
    retirement batch size, not ``N``) rather than leaving a hole.

Three primitives:

``SlotAllocator``
    Owns ``N``/``E``/``C`` and logical-ID bookkeeping. Physical row
    ``[0, N)`` is always the packed active prefix; ``[N, C)`` is always
    free capacity, address-order-agnostic (nothing is ever "the" free slot
    at a fixed index -- it is simply "beyond the current prefix").

``CapacityOptimizerBinding``
    Owns one registered ``nn.Parameter``-backed buffer's optimizer wiring:
    a capacity-sized backing tensor, capacity-sized Adam-momentum backing
    tensors (created lazily once the optimizer has actually stepped and
    therefore has real momentum state to preserve), and the currently-live
    ``nn.Parameter`` view onto ``backing[:epoch_capacity]``.
    :meth:`rebind` is the only place a new ``nn.Parameter`` object is ever
    constructed, and it migrates existing momentum into the capacity-sized
    backing rather than ``torch.cat``-extending Adam's own state tensors.

``CapacityBufferRegistry``
    Wraps a :class:`SlotAllocator` plus a set of aligned per-Gaussian
    tensors (plain buffers via ``register_buffer``, optimizer-tracked
    buffers via ``register_param``). Inserting into free capacity is an
    in-place write at the (already free, contiguous) tail -- no
    ``torch.cat``, no new ``nn.Parameter`` object beyond the one
    ``rebind()`` produces per mutation, no per-insert optimizer-state
    resize.

``MutationEpochManager``
    The mutation-barrier / topology-version state machine (plan §16.4),
    unchanged by this redesign. A stable epoch (zero structural changes
    recorded between ``begin_mutation_barrier`` and ``end_mutation_barrier``)
    never bumps ``topology_version``; any real insert/reseed/prune/retire
    batch does. :meth:`stable_epoch` is a debug context manager asserting
    no tracked ``nn.Parameter`` object is replaced while it is open --
    exactly the invariant :meth:`CapacityOptimizerBinding.rebind` must
    never violate mid-epoch.
"""

from __future__ import annotations

import dataclasses
from typing import Callable, Dict, Iterable, List, Optional

import torch
from torch import nn


# ---------------------------------------------------------------------------
# SlotAllocator
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class AllocationResult:
    """Result of :meth:`SlotAllocator.allocate`."""

    physical_idx: torch.Tensor  # LongTensor (n,) -- always the new tail rows
    logical_ids: torch.Tensor  # LongTensor (n,)
    grew: bool
    old_capacity: int
    new_capacity: int


@dataclasses.dataclass
class CompactionPlan:
    """Result of :meth:`SlotAllocator.retire` -- a batched swap-delete plan.

    ``dst_indices``/``src_indices`` are the SAME length: row
    ``src_indices[i]`` (a surviving row currently in the tail beyond the new
    active prefix) must be copied into ``dst_indices[i]`` (a hole below the
    new active prefix) in every aligned buffer, and any momentum-backing
    tensor, to keep physical rows ``[0, new_n_active)`` packed and
    contiguous. A hole that already lies at or beyond ``new_n_active``
    needs no donor -- it is simply excluded from the prefix by the
    ``n_active`` shrink and is not represented in ``dst_indices``/
    ``src_indices`` at all. Cost is therefore bounded by the number of
    holes that fall BELOW the new prefix, never by ``N`` or ``C``.
    """

    dst_indices: torch.Tensor  # LongTensor, holes needing a donor
    src_indices: torch.Tensor  # LongTensor, same length, donor rows from the tail
    retired_logical_ids: torch.Tensor  # LongTensor, every logical id actually freed
    old_n_active: int
    new_n_active: int


class SlotAllocator:
    """Packed-contiguous-active-prefix capacity/logical-id bookkeeping.

    Physical rows ``[0, n_active)`` are ALWAYS the live, packed prefix --
    there is no boolean active mask and no free-slot list. Free storage is
    always exactly ``[n_active, capacity)``. Retiring an arbitrary physical
    row (or set of rows) below the current prefix produces a
    :class:`CompactionPlan` describing which surviving tail rows must move
    into the vacated holes; the caller (:class:`CapacityBufferRegistry`)
    applies the identical plan to every aligned buffer and to any momentum
    backing.
    """

    def __init__(
        self,
        capacity: int,
        device: torch.device | str = "cuda",
        debug_checks: bool = True,
        epoch_capacity_policy: Optional[Callable[[int], int]] = None,
    ):
        if capacity < 0:
            raise ValueError("capacity must be >= 0")
        self.device = torch.device(device)
        self.capacity = int(capacity)
        self.n_active = 0  # N: logical live count, always == len(active prefix)
        self.logical_ids = torch.full((self.capacity,), -1, dtype=torch.int64, device=self.device)
        self._next_logical_id: int = 0
        self.logical_to_physical: Dict[int, int] = {}
        # E: optimizer/render-visible shape for the current stable epoch.
        # This PR's default policy is identity (E == N) -- see module
        # docstring's "Three quantities" section. A future bucketing policy
        # (E rounds N up to a fixed compiled shape) plugs in here without
        # touching any call site, since every call site already threads
        # `epoch_capacity`, not `n_active`, through rebind()/active_view().
        self._epoch_capacity_policy = epoch_capacity_policy or (lambda n: n)
        # assert_consistent() is cheap here (no GPU sync -- n_active is a
        # plain int, unlike the pre-W3-49 design's active.sum().item()) but
        # still gated behind this flag for parity with the pre-redesign
        # constructor signature and so benchmarks can disable it explicitly.
        self.debug_checks = debug_checks

    # -- introspection ----------------------------------------------------

    @property
    def epoch_capacity(self) -> int:
        """E: optimizer/render-visible shape for the CURRENT stable epoch.
        ``E >= N`` always; this PR's policy is ``E == N`` (identity)."""
        return self._epoch_capacity_policy(self.n_active)

    @property
    def n_free(self) -> int:
        """Return the number of unused physical slots (capacity minus active rows)."""
        return self.capacity - self.n_active

    def active_physical_indices(self) -> torch.Tensor:
        """The packed active prefix as a physical-index tensor. O(N)
        materialization (kept for API parity with callers that need real
        indices, e.g. to build a gradient mask) -- callers that only need
        the LIVE DATA should prefer a plain ``buffer[:n_active]`` slice
        instead, which is O(1) (a view, not a materializing op)."""
        return torch.arange(0, self.n_active, dtype=torch.int64, device=self.device)

    def physical_for(self, logical_ids: Iterable[int]) -> List[int]:
        """Map logical Gaussian ids to their current physical row indices."""
        return [self.logical_to_physical[int(lid)] for lid in logical_ids]

    def assert_consistent(self) -> None:
        """Debug invariant: N/C/logical-id bookkeeping agree.

        Deliverable 2 ("per-Gaussian buffers cannot silently become
        length-misaligned") lives partly here: if this invariant is ever
        violated, every aligned buffer registered against this allocator is
        suspect.
        """
        assert self.logical_ids.shape[0] == self.capacity
        assert 0 <= self.n_active <= self.capacity
        assert len(self.logical_to_physical) == self.n_active
        if self.n_active > 0:
            active_ids = self.logical_ids[: self.n_active]
            assert bool((active_ids >= 0).all().item()), "active prefix contains an unassigned logical id"
        if self.n_active < self.capacity:
            inactive_ids = self.logical_ids[self.n_active :]
            assert bool((inactive_ids == -1).all().item()), "inactive tail has a stale logical id"

    # -- mutation -----------------------------------------------------------

    def allocate(self, n: int, growth_factor: float = 1.5, min_growth: int = 64) -> AllocationResult:
        """Allocate ``n`` physical rows at the tail of the active prefix
        (``[n_active, n_active + n)``) -- always contiguous, never a hole
        lookup. Growth only occurs when ``n_active + n`` exceeds capacity.
        When it does, capacity is extended by
        ``max(shortfall, min_growth, ceil(capacity * (growth_factor - 1)))``
        so repeated small growth events don't each trigger their own
        reallocation (amortized growth, same rationale as
        ``list.append``/``vector::push_back``).
        """
        if n < 0:
            raise ValueError("n must be >= 0")
        old_capacity = self.capacity
        grew = False
        if self.n_active + n > self.capacity:
            shortfall = (self.n_active + n) - self.capacity
            growth = max(shortfall, min_growth, int(self.capacity * (growth_factor - 1)))
            self._grow(self.capacity + growth)
            grew = True

        physical_idx = torch.arange(self.n_active, self.n_active + n, dtype=torch.int64, device=self.device)
        logical = torch.arange(
            self._next_logical_id, self._next_logical_id + n, dtype=torch.int64, device=self.device
        )
        self._next_logical_id += n

        self.logical_ids[physical_idx] = logical
        for lid, pidx in zip(logical.tolist(), physical_idx.tolist()):
            self.logical_to_physical[lid] = pidx

        self.n_active += n

        if self.debug_checks:
            self.assert_consistent()
        return AllocationResult(
            physical_idx=physical_idx,
            logical_ids=logical,
            grew=grew,
            old_capacity=old_capacity,
            new_capacity=self.capacity,
        )

    def _grow(self, new_capacity: int) -> None:
        """Extend the allocator's own bookkeeping tensor. Buffers/momentum
        backing are grown separately by :class:`CapacityBufferRegistry`/
        :class:`CapacityOptimizerBinding`. This is the single, explicit,
        sanctioned reallocation event -- not a per-insert cost."""
        assert new_capacity >= self.capacity
        extra = new_capacity - self.capacity
        if extra == 0:
            return
        self.logical_ids = torch.cat(
            [self.logical_ids, torch.full((extra,), -1, dtype=torch.int64, device=self.device)]
        )
        self.capacity = new_capacity

    def retire(self, physical_idx: torch.Tensor) -> CompactionPlan:
        """Retire arbitrary physical rows within the active prefix via
        batched swap-delete: for every hole that falls BELOW the shrunk
        prefix boundary, a surviving donor row from the current tail is
        selected to move into it. A hole that already lies at or beyond the
        shrunk boundary needs no donor -- it's simply excluded by the
        ``n_active`` decrease. Cost (both here and in the caller's buffer
        application) is bounded by ``len(problematic_holes)``, which is at
        most ``len(physical_idx)`` and is typically far smaller than the
        full retirement batch when most retired rows already happen to be
        near the tail -- never O(N) or O(C).
        """
        physical_idx = physical_idx.to(device=self.device, dtype=torch.int64)
        if physical_idx.numel() == 0:
            empty = torch.empty(0, dtype=torch.int64, device=self.device)
            return CompactionPlan(
                dst_indices=empty, src_indices=empty, retired_logical_ids=empty,
                old_n_active=self.n_active, new_n_active=self.n_active,
            )

        holes_unique = torch.unique(physical_idx)
        assert holes_unique.numel() == physical_idx.numel(), "retire() called with duplicate physical indices"
        assert bool((holes_unique >= 0).all().item()) and bool((holes_unique < self.n_active).all().item()), (
            "retire() called on a physical index outside the active prefix"
        )

        retired_logical_ids = self.logical_ids[holes_unique].clone()
        k = int(holes_unique.numel())
        old_n_active = self.n_active
        new_n_active = old_n_active - k

        holes_set = set(holes_unique.tolist())
        problematic = sorted(h for h in holes_set if h < new_n_active)
        donors = sorted(r for r in range(new_n_active, old_n_active) if r not in holes_set)
        assert len(problematic) == len(donors), "compaction bookkeeping invariant violated"

        dst_indices = torch.tensor(problematic, dtype=torch.int64, device=self.device)
        src_indices = torch.tensor(donors, dtype=torch.int64, device=self.device)

        if dst_indices.numel() > 0:
            moved_logical = self.logical_ids[src_indices].clone()
            self.logical_ids[dst_indices] = moved_logical
            for lid, new_phys in zip(moved_logical.tolist(), dst_indices.tolist()):
                self.logical_to_physical[lid] = new_phys

        for lid in retired_logical_ids.tolist():
            self.logical_to_physical.pop(lid, None)

        self.logical_ids[new_n_active:old_n_active] = -1
        self.n_active = new_n_active

        if self.debug_checks:
            self.assert_consistent()

        return CompactionPlan(
            dst_indices=dst_indices,
            src_indices=src_indices,
            retired_logical_ids=retired_logical_ids,
            old_n_active=old_n_active,
            new_n_active=new_n_active,
        )

    # -- checkpoint round trip ----------------------------------------------

    def capture(self) -> dict:
        """Return the allocator state (capacity, active count, logical ids) for checkpointing."""
        return {
            "capacity": self.capacity,
            "n_active": self.n_active,
            "logical_ids": self.logical_ids.detach().cpu(),
            "next_logical_id": self._next_logical_id,
        }

    @classmethod
    def restore(cls, state: dict, device: torch.device | str = "cuda") -> "SlotAllocator":
        """Rebuild an allocator on ``device`` from a ``capture`` payload."""
        obj = cls(capacity=state["capacity"], device=device)
        obj.logical_ids = state["logical_ids"].to(device=obj.device)
        obj.n_active = int(state["n_active"])
        obj._next_logical_id = int(state["next_logical_id"])
        obj.logical_to_physical = {
            int(lid): int(pidx) for pidx, lid in enumerate(obj.logical_ids[: obj.n_active].tolist())
        }
        obj.assert_consistent()
        return obj


# ---------------------------------------------------------------------------
# CapacityOptimizerBinding
# ---------------------------------------------------------------------------


class CapacityOptimizerBinding:
    """Owns one registered ``nn.Parameter``-backed buffer's optimizer wiring.

    ``backing`` is a plain, capacity-sized tensor (NOT itself an
    ``nn.Parameter``) that never moves except on a genuine capacity-growth
    event. The optimizer only ever sees ``backing[:epoch_capacity]`` wrapped
    in an ``nn.Parameter`` -- verified empirically to alias ``backing``'s
    storage (same ``data_ptr``, in-place optimizer updates visible in
    ``backing`` with no copy-back) for both ``torch.optim.Adam`` and
    ``gsplat.optimizers.SelectiveAdam``, including across a rebind -- see
    docs/architecture/2026-08-10-w3-49-parameter-slice-aliasing-validation.md.

    Adam's own momentum state tensors (``exp_avg``/``exp_avg_sq``) are NOT
    automatically aliased -- Adam allocates them itself via
    ``torch.zeros_like(param)`` the first time it steps. This class owns
    capacity-sized momentum BACKING tensors (created lazily, once real
    momentum state exists to preserve) and migrates the optimizer's current
    momentum values into them at every :meth:`rebind`, so momentum persists
    across mutation-barrier boundaries without the optimizer's own state
    ever being ``torch.cat``-extended.
    """

    def __init__(self, name: str, backing: torch.Tensor, optimizer: torch.optim.Optimizer, capacity: int):
        self.name = name
        self.backing = backing
        self.optimizer = optimizer
        self.capacity = capacity
        self.live_param: Optional[nn.Parameter] = None
        self._momentum_backing: Dict[str, torch.Tensor] = {}

    def sync_momentum_from_optimizer(self) -> None:
        """Ensure ``self._momentum_backing`` reflects the optimizer's
        CURRENT state for ``self.live_param``, creating backing tensors and
        copying values in on first use.

        Idempotent and O(1) in steady state: after the first sync following
        a given ``live_param``'s first ``optimizer.step()``, every
        subsequent call is a cheap storage-identity check that short-
        circuits immediately, because :meth:`rebind` always hands the
        optimizer state tensors that ARE already views into
        ``self._momentum_backing`` -- Adam's own in-place update ops then
        keep the backing current for free, with no further copying. The
        ONE real O(n_active) cost happens exactly once per binding
        lifetime (the first time real momentum state exists and has not
        yet been captured into backing) -- this is the fix for a real bug
        found in review: :meth:`CapacityBufferRegistry.retire`'s
        compaction step needs momentum backing to already reflect the
        CURRENT optimizer state before it moves rows, but the original
        version of this migration logic only ran inside :meth:`rebind`,
        which retire() calls AFTER compaction -- so the very first retire()
        following a step() silently compacted stale (pre-backing) momentum
        and left the real values behind. Caught by
        ``test_momentum_follows_gaussian_through_compaction`` before merge.
        """
        if self.live_param is None:
            return
        state = self.optimizer.state.get(self.live_param)
        if not state:
            return
        n = int(self.live_param.shape[0])
        for key, value in list(state.items()):
            if not (isinstance(value, torch.Tensor) and value.dim() > 0):
                continue
            if key in self._momentum_backing and self._momentum_backing[key][:n].data_ptr() == value.data_ptr():
                continue  # already aliased -- O(1) skip
            if key not in self._momentum_backing:
                self._momentum_backing[key] = torch.zeros(
                    self.capacity, *value.shape[1:], dtype=value.dtype, device=value.device
                )
            self._momentum_backing[key][:n] = value
            # Critical: rewrite the OPTIMIZER's own state entry to alias the
            # backing view immediately, not just our local dict. Without
            # this, `state[key]` stays the original (non-aliased) tensor
            # Adam allocated -- a second sync call (e.g. rebind()'s own
            # call to this method, right after CapacityBufferRegistry
            # .retire() already synced once before compaction) would read
            # THAT stale, un-compacted tensor again and clobber whatever
            # apply_compaction() had just written into _momentum_backing,
            # since compaction only ever touches _momentum_backing, never
            # the optimizer's original tensor object directly. Aliasing
            # here makes them the same storage, so every subsequent
            # reader/writer (compaction, further syncs, Adam's own future
            # in-place updates) sees one consistent value.
            state[key] = self._momentum_backing[key][:n]

    def rebind(self, epoch_capacity: int) -> nn.Parameter:
        """Construct a new active-prefix ``nn.Parameter`` view onto
        ``backing[:epoch_capacity]`` and rewire the optimizer to it. Calls
        :meth:`sync_momentum_from_optimizer` first (idempotent, see that
        method's docstring) so momentum always migrates correctly even if
        this is the first rebind after a step.

        Only legal to call while the registry's mutation barrier is open
        (caller convention, mirroring the pre-W3-49 growth path -- not
        separately enforced here; :class:`MutationEpochManager`'s
        ``stable_epoch()`` debug guard is what actually catches a violation
        of this, by asserting no tracked parameter object changes identity
        while a stable epoch is open).
        """
        self.sync_momentum_from_optimizer()
        old_param = self.live_param
        old_state = self.optimizer.state.get(old_param) if old_param is not None else None

        new_param = nn.Parameter(self.backing[:epoch_capacity].detach().requires_grad_(True))
        new_state: Dict[str, object] = {}
        if old_state:
            for key, value in old_state.items():
                if isinstance(value, torch.Tensor) and value.dim() > 0:
                    new_state[key] = self._momentum_backing[key][:epoch_capacity]
                else:
                    new_state[key] = value

        if old_param is not None and old_param in self.optimizer.state:
            del self.optimizer.state[old_param]

        replaced = False
        for group in self.optimizer.param_groups:
            params = group["params"]
            for i, p in enumerate(params):
                if p is old_param:
                    params[i] = new_param
                    replaced = True
                    break
            if replaced:
                break
        if not replaced:
            self.optimizer.param_groups[0]["params"].append(new_param)

        if new_state:
            self.optimizer.state[new_param] = new_state

        self.live_param = new_param
        return new_param

    def apply_compaction(self, dst_indices: torch.Tensor, src_indices: torch.Tensor) -> None:
        """Apply the same donor->hole swap-delete moves a
        :class:`CompactionPlan` describes to this binding's momentum
        backing tensors, so a Gaussian's Adam state follows it through
        retirement compaction exactly like its parameter values do."""
        for tensor in self._momentum_backing.values():
            tensor.index_copy_(0, dst_indices, tensor[src_indices])

    def grow_capacity(self, new_capacity: int) -> None:
        """Extend momentum backing tensors to match a capacity growth event
        (parallels ``CapacityBufferRegistry._grow_buffers`` for the plain
        parameter backing). No-op per key until that key's momentum backing
        has been created (a freshly-created one is already capacity-sized)."""
        extra = new_capacity - self.capacity
        if extra > 0:
            for key, tensor in self._momentum_backing.items():
                pad = torch.zeros(extra, *tensor.shape[1:], dtype=tensor.dtype, device=tensor.device)
                self._momentum_backing[key] = torch.cat([tensor, pad], dim=0)
        self.capacity = max(self.capacity, new_capacity)

    def zero_momentum(self, physical_idx: torch.Tensor) -> None:
        """Zero this binding's momentum backing at the given physical rows
        (deliverable 3: reused/new slots start with zero Adam m/v). No-op
        per key until that key's momentum backing exists (freshly-created
        backing is already all-zero, so there is nothing stale to clear)."""
        for tensor in self._momentum_backing.values():
            tensor.index_fill_(0, physical_idx.to(tensor.device), 0.0)


# ---------------------------------------------------------------------------
# CapacityBufferRegistry
# ---------------------------------------------------------------------------


class CapacityBufferRegistry:
    """Aligned-buffer registry over a fixed-capacity, packed-prefix
    :class:`SlotAllocator`.

    Distinct from (and NOT a replacement for) W3-10's
    ``strategies.live.backend.AlignedBufferRegistry`` -- see module
    docstring. This registry owns the core map's capacity-backed storage;
    plain tensor buffers (``register_buffer``) and optimizer-tracked
    buffers (``register_param``) are both supported so both model layouts'
    streaming state (lifecycle/anchor/support buffers) and the learnable
    Gaussian parameters themselves can live behind one allocator.
    """

    def __init__(self, allocator: SlotAllocator):
        self.allocator = allocator
        self.buffers: Dict[str, torch.Tensor] = {}
        self.param_names: set[str] = set()
        self.optimizers: Dict[str, torch.optim.Optimizer] = {}
        self.bindings: Dict[str, CapacityOptimizerBinding] = {}
        self._live_params: Dict[str, nn.Parameter] = {}
        # Count of torch.cat-based growth events, exposed for benchmarking
        # (deliverable 5) and for tests asserting the hot path avoids cat.
        self.growth_events: int = 0

    # -- registration ---------------------------------------------------

    def register_buffer(self, name: str, tensor: torch.Tensor) -> None:
        """Register a capacity-length buffer under ``name``; rejects length mismatches."""
        if tensor.shape[0] != self.allocator.capacity:
            raise ValueError(
                f"buffer '{name}' has length {tensor.shape[0]} != allocator "
                f"capacity {self.allocator.capacity}"
            )
        self.buffers[name] = tensor

    def register_param(
        self,
        name: str,
        backing: torch.Tensor,
        optimizer_factory: Callable[[List[nn.Parameter]], torch.optim.Optimizer],
    ) -> torch.optim.Optimizer:
        """Register a capacity-sized ``backing`` tensor as an
        optimizer-tracked buffer.

        ``backing`` must be a PLAIN tensor, NOT already an ``nn.Parameter``
        bound to an optimizer -- this registry constructs and owns the
        optimizer-visible active-prefix ``nn.Parameter`` view itself, and
        the optimizer object is constructed here (via ``optimizer_factory``)
        rather than passed in already-built, because the very first
        ``nn.Parameter`` the optimizer will ever track (a possibly-empty
        ``backing[:epoch_capacity]`` slice at registration time) does not
        exist until this call constructs it -- most optimizer constructors
        (including ``torch.optim.Optimizer``) reject an empty parameter
        list, so the optimizer cannot be built before this. Returns the
        constructed optimizer for the caller to hold onto (e.g. to pass to
        a training loop's optimizer-list).

        This is a breaking API change from the pre-W3-49 design (which took
        an already-constructed, already-bound full-capacity ``nn.Parameter``
        plus a pre-built optimizer) -- safe because this class had zero
        production call sites at the time of the change, confirmed in
        docs/architecture/2026-08-10-w3-36-capacity-registry-optimizer-audit.md.
        """
        self.register_buffer(name, backing)
        self.param_names.add(name)
        initial_param = nn.Parameter(
            backing[: self.allocator.epoch_capacity].detach().requires_grad_(True)
        )
        optimizer = optimizer_factory([initial_param])
        self.optimizers[name] = optimizer
        binding = CapacityOptimizerBinding(name, backing, optimizer, self.allocator.capacity)
        binding.live_param = initial_param
        self.bindings[name] = binding
        self._live_params[name] = initial_param
        return optimizer

    def register_buffer_late(
        self, name: str, dtype: torch.dtype, shape_tail: tuple = (), fill_value=0
    ) -> torch.Tensor:
        """Register a conditionally-allocated buffer (e.g. consensus/voter
        buffers that only exist once a flag is turned on mid-run) after the
        allocator already has active rows.

        Creates a fresh capacity-sized tensor filled with ``fill_value`` for
        every row (both currently-active and currently-free), so already-live
        Gaussians get an explicit, documented default instead of silently
        reading garbage the first time this buffer is touched. Returns the
        created tensor so the caller can also stash it under its own
        attribute name (matching how the model exposes streaming buffers).
        """
        cap = self.allocator.capacity
        tensor = torch.full(
            (cap, *shape_tail), fill_value, dtype=dtype, device=self.allocator.device
        )
        self.register_buffer(name, tensor)
        return tensor

    def unregister(self, name: str) -> None:
        """Forget ``name`` and every optimizer/binding/live-parameter record attached to it."""
        self.buffers.pop(name, None)
        self.param_names.discard(name)
        self.optimizers.pop(name, None)
        self.bindings.pop(name, None)
        self._live_params.pop(name, None)

    def assert_aligned(self) -> None:
        """Deliverable 2: every registered buffer's row count must equal
        allocator capacity. Raises loudly instead of silently corrupting a
        mutation if any buffer ever drifts."""
        cap = self.allocator.capacity
        for name, buf in self.buffers.items():
            if buf.shape[0] != cap:
                raise AssertionError(
                    f"CapacityBufferRegistry buffer '{name}' has length "
                    f"{buf.shape[0]} but allocator capacity is {cap} -- "
                    "buffers are length-misaligned"
                )

    # -- mutation ---------------------------------------------------------

    def insert(self, rows: Dict[str, torch.Tensor]) -> AllocationResult:
        """Insert new Gaussians described by ``rows`` (name -> (n, ...)).

        Every registered buffer name must have an entry in ``rows`` -- a
        missing key would silently leave stale/zero data in the newly
        allocated slot, which is exactly the misalignment deliverable 2
        guards against, so this raises instead.

        New rows always land at the (already-free) tail ``[n_active,
        n_active + n)`` -- an in-place write, never a ``torch.cat``, except
        for the single sanctioned growth-fallback reallocation when
        capacity is exhausted. Every registered ``nn.Parameter``-backed
        buffer is rebound (a new active-prefix view constructed, momentum
        migrated) exactly once per call, regardless of how many buffers are
        registered against this allocator.
        """
        if not self.buffers:
            raise ValueError("insert() called with no buffers registered")
        missing = set(self.buffers) - set(rows)
        if missing:
            raise ValueError(f"insert() missing rows for registered buffers: {sorted(missing)}")
        first_name = next(iter(self.buffers))
        n = rows[first_name].shape[0]
        for name in self.buffers:
            if rows[name].shape[0] != n:
                raise ValueError(f"insert() row-count mismatch for '{name}': {rows[name].shape[0]} != {n}")

        old_capacity = self.allocator.capacity
        result = self.allocator.allocate(n)
        if result.grew:
            self._grow_buffers(old_capacity, result.new_capacity)
            for binding in self.bindings.values():
                binding.grow_capacity(result.new_capacity)
            self.growth_events += 1

        physical_idx = result.physical_idx
        for name, buf in self.buffers.items():
            new_vals = rows[name].to(device=buf.device, dtype=buf.dtype)
            buf.index_copy_(0, physical_idx, new_vals)
            if name in self.param_names:
                self.bindings[name].zero_momentum(physical_idx)

        for name in self.param_names:
            self._live_params[name] = self.bindings[name].rebind(self.allocator.epoch_capacity)

        self.assert_aligned()
        return result

    def retire(self, physical_idx: torch.Tensor) -> CompactionPlan:
        """Retire arbitrary physical rows via batched swap-delete (see
        ``SlotAllocator.retire``'s docstring for the compaction mechanism).
        Applies the identical plan to every aligned buffer AND to every
        registered param's momentum backing, then rebinds every registered
        ``nn.Parameter``-backed buffer to the new (shrunk) epoch capacity.

        Every param binding's momentum is explicitly synced from the
        optimizer's live state BEFORE compaction moves any rows -- momentum
        backing is normally kept current for free by rebind() (see
        CapacityOptimizerBinding.sync_momentum_from_optimizer's docstring),
        but compaction itself must run before this call's own rebind(), so
        without this explicit sync the very first retire() after a step()
        would compact stale (pre-backing) momentum and silently leave a
        Gaussian's real momentum behind at its old physical row.
        """
        for name in self.param_names:
            self.bindings[name].sync_momentum_from_optimizer()

        plan = self.allocator.retire(physical_idx)
        if plan.dst_indices.numel() > 0:
            for name, buf in self.buffers.items():
                buf.index_copy_(0, plan.dst_indices, buf[plan.src_indices])
                if name in self.param_names:
                    self.bindings[name].apply_compaction(plan.dst_indices, plan.src_indices)

        for name in self.param_names:
            self._live_params[name] = self.bindings[name].rebind(self.allocator.epoch_capacity)

        self.assert_aligned()
        return plan

    def _grow_buffers(self, old_capacity: int, new_capacity: int) -> None:
        extra = new_capacity - old_capacity
        for name, buf in list(self.buffers.items()):
            pad_shape = (extra,) + tuple(buf.shape[1:])
            pad = torch.zeros(pad_shape, dtype=buf.dtype, device=buf.device)
            grown = torch.cat([buf, pad], dim=0)
            self.buffers[name] = grown
            if name in self.param_names:
                # The binding's backing tensor object must track the new
                # (grown) buffer object -- torch.cat cannot grow in place,
                # so the old backing tensor is now orphaned and any future
                # rebind() must slice the NEW one.
                self.bindings[name].backing = grown

    def param_for(self, name: str) -> Optional[nn.Parameter]:
        """Return the live ``nn.Parameter`` for ``name``, or ``None`` if it is not param-backed."""
        return self._live_params.get(name)

    def active_parameter(self, name: str) -> nn.Parameter:
        """The live, optimizer-tracked ``nn.Parameter`` for a param-backed
        buffer. Raises if ``name`` was never registered via
        ``register_param`` -- use this (or ``active_view``, which
        dispatches here automatically) for anything that needs to
        participate in a differentiable computation graph, e.g. rendering.
        """
        if name not in self.param_names:
            raise KeyError(
                f"'{name}' is not a param-backed buffer (registered via "
                "register_buffer, not register_param) -- use active_view() "
                "for a read-only slice instead"
            )
        param = self._live_params.get(name)
        assert param is not None, f"'{name}' is param-backed but has no live Parameter bound"
        return param

    # -- reads --------------------------------------------------------------

    def active_view(self, name: str) -> torch.Tensor:
        """The active-prefix VIEW of buffer ``name``, safe to use in a
        differentiable computation (e.g. rendering).

        For a plain buffer (registered via ``register_buffer``): a
        zero-copy slice of the backing tensor (``buffer[:epoch_capacity]``)
        -- this is the W3-49 fix for the pre-redesign boolean-mask
        ``self.buffers[name][self.allocator.active]`` (advanced indexing,
        real work every call) -- see
        docs/architecture/2026-08-10-w3-49-capacity-scaling-baseline.md's
        measured 0.057-0.098ms/call gather cost this eliminates.

        For a param-backed buffer (registered via ``register_param``):
        dispatches to ``active_parameter(name)`` and returns the live
        ``nn.Parameter`` itself, NOT a fresh slice of ``self.buffers[name]``
        (the plain backing tensor). This distinction is load-bearing, not
        cosmetic: ``self.buffers[name]`` never requires grad (it is the
        raw backing storage), while ``self._live_params[name]`` is a
        SEPARATE tensor object (``nn.Parameter(backing[:E].detach()
        .requires_grad_(True))``) that happens to alias the same storage.
        Slicing ``self.buffers[name]`` for a param-backed name would
        silently hand a caller a `requires_grad=False` view -- gradients
        computed against it would never reach the optimizer's Parameter.
        See ``test_active_view_of_param_backed_buffer_is_differentiable``
        for the regression test (a real backward()/optimizer.step() round
        trip through this exact call, not just an identity check).
        """
        if name in self.param_names:
            return self.active_parameter(name)
        return self.buffers[name][: self.allocator.epoch_capacity]

    def active_view_all(self) -> Dict[str, torch.Tensor]:
        """Return the active (live-row) view of every registered buffer keyed by name."""
        return {name: self.active_view(name) for name in self.buffers}

    # -- checkpoint round trip ----------------------------------------------

    def capture(self) -> dict:
        """Return the registry state (allocator, buffers, param names, growth events) for checkpointing."""
        return {
            "allocator": self.allocator.capture(),
            "buffers": {name: buf.detach().cpu() for name, buf in self.buffers.items()},
            "param_names": sorted(self.param_names),
            "growth_events": self.growth_events,
        }

    def restore_buffers(self, state: dict, device: torch.device | str = "cuda") -> None:
        """Restore buffer contents (capacity-sized) after the allocator has
        already been restored via :meth:`SlotAllocator.restore` and buffers
        re-registered by the caller at the restored capacity."""
        for name, tensor in state["buffers"].items():
            if name in self.buffers:
                self.buffers[name].copy_(tensor.to(device=device))


# ---------------------------------------------------------------------------
# Mutation barrier / topology epochs
# ---------------------------------------------------------------------------


class MutationEpochManager:
    """Plan §16.4 mutation-barrier / topology-version state machine.

    Usage::

        mgr.begin_mutation_barrier()
        ... apply batched insert/reseed/prune/retire ...
        mgr.record_structural_change("insert", n_inserted)
        ... rebuild any physical-index-keyed caches here ...
        bumped = mgr.end_mutation_barrier()
        # run K stable optimization steps -- during this window no
        # parameter object replacement, no optimizer-state resize, no
        # storage reallocation, no torch.cat is permitted; mgr.stable_epoch()
        # below is a debug context manager that checks this for a given set
        # of tracked param objects.
    """

    def __init__(self):
        self.topology_version: int = 0
        self._in_epoch: bool = False
        self._structural_changes: int = 0
        self.epoch_count: int = 0

    def begin_mutation_barrier(self) -> None:
        """Open a mutation epoch; structural changes must be recorded before it is committed."""
        assert not self._in_epoch, "mutation barrier already open"
        self._in_epoch = True
        self._structural_changes = 0

    def record_structural_change(self, kind: str, count: int) -> None:
        """Record ``count`` rows affected by a structural mutation of the
        given ``kind`` (e.g. "insert", "reseed", "prune", "retire"). Must be
        called between begin/end. A count of 0 records nothing (so calling
        this defensively on a no-op batch doesn't spuriously bump the
        version)."""
        assert self._in_epoch, "record_structural_change() outside a mutation barrier"
        if count > 0:
            self._structural_changes += count

    def end_mutation_barrier(self) -> bool:
        """Close the barrier. Returns True iff topology_version was bumped."""
        assert self._in_epoch, "end_mutation_barrier() without begin_mutation_barrier()"
        bumped = self._structural_changes > 0
        if bumped:
            self.topology_version += 1
        self._in_epoch = False
        self.epoch_count += 1
        return bumped

    def stable_epoch(self, tracked_params: Dict[str, nn.Parameter]):
        """Debug context manager for the K-stable-steps window: asserts that
        none of ``tracked_params`` were replaced by a new object (i.e. no
        param-object replacement / optimizer-state resize / storage
        reallocation happened) while the context is open. This is the
        concrete, testable form of the "Static-epoch readiness" invariant:
        a :class:`CapacityOptimizerBinding` rebind (from an ``insert()`` or
        ``retire()`` call) mid-epoch changes the tracked object's identity
        and is caught here."""
        return _StableEpochGuard(tracked_params)

    def capture(self) -> dict:
        """Return the epoch state (topology version, epoch count) for checkpointing."""
        return {"topology_version": self.topology_version, "epoch_count": self.epoch_count}

    def restore(self, state: dict) -> None:
        """Restore the epoch state written by ``capture``."""
        self.topology_version = int(state["topology_version"])
        self.epoch_count = int(state.get("epoch_count", 0))


class _StableEpochGuard:
    def __init__(self, tracked_params: Dict[str, nn.Parameter]):
        self._before = {name: id(p) for name, p in tracked_params.items()}
        self._tracked = tracked_params

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is not None:
            return False
        for name, before_id in self._before.items():
            after = self._tracked[name]
            assert id(after) == before_id, (
                f"stable epoch violated: parameter '{name}' object was replaced "
                "(param-object replacement is only permitted inside a mutation barrier)"
            )
        return False
