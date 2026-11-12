"""Wave 3, PR W3-11 (redesigned W3-49): fixed-capacity, packed-active-prefix
map storage and topology epochs.

Pure-tensor unit tests for ``eligsir/optimization/gaussian/storage/
capacity_registry.py``. Deliberately CPU-only (device="cpu") -- the
allocator/registry/mutation-barrier/optimizer-binding logic has no
CUDA/gsplat dependency, so these run without a GPU, unlike most of this
repo's model-level tests. (SelectiveAdam-specific compatibility is verified
separately, on GPU, in
docs/architecture/2026-08-10-w3-49-parameter-slice-aliasing-validation.md --
not re-tested here since it needs a real gsplat CUDA/HIP kernel.)

W3-49 replaced the original design (boolean active mask + LIFO free-slot
list, arbitrary non-contiguous active rows) with a packed-contiguous-
active-prefix design: physical rows [0, n_active) are ALWAYS the live
prefix, retiring an interior row triggers a batched swap-delete
(CompactionPlan) instead of leaving a hole, and register_param's calling
convention changed (caller passes a plain backing tensor + an optimizer
FACTORY, not an already-bound nn.Parameter/optimizer pair) -- see that
file's module docstring and docs/architecture/2026-08-10-w3-49-packed-
prefix-redesign.md for the full rationale and acceptance-gate benchmarks
this rewrite was measured against.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from eligsir.optimization.gaussian.storage.capacity_registry import (  # noqa: E402
    CapacityBufferRegistry,
    CompactionPlan,
    MutationEpochManager,
    SlotAllocator,
)

DEVICE = "cpu"


def _make_registry(capacity: int) -> tuple[SlotAllocator, CapacityBufferRegistry]:
    alloc = SlotAllocator(capacity, device=DEVICE)
    reg = CapacityBufferRegistry(alloc)
    reg.register_buffer("scores", torch.zeros(capacity, dtype=torch.float32, device=DEVICE))
    reg.register_buffer("tags", torch.zeros(capacity, 3, dtype=torch.float32, device=DEVICE))
    return alloc, reg


def _adam_factory(lr: float = 0.01):
    return lambda params: torch.optim.Adam(params, lr=lr, eps=1e-15)


def _make_param_registry(capacity: int, lr: float = 0.01):
    alloc = SlotAllocator(capacity, device=DEVICE)
    reg = CapacityBufferRegistry(alloc)
    means_backing = torch.zeros(capacity, 3, dtype=torch.float32, device=DEVICE)
    optimizer = reg.register_param("means", means_backing, _adam_factory(lr))
    return alloc, reg, optimizer


# ---------------------------------------------------------------------------
# SlotAllocator: allocation always appends at the tail
# ---------------------------------------------------------------------------


def test_allocate_appends_at_contiguous_tail():
    alloc = SlotAllocator(10, device=DEVICE)
    result = alloc.allocate(4)
    assert result.physical_idx.tolist() == [0, 1, 2, 3]
    assert alloc.n_active == 4
    result2 = alloc.allocate(3)
    assert result2.physical_idx.tolist() == [4, 5, 6]
    assert alloc.n_active == 7


def test_allocate_grows_only_when_capacity_exhausted():
    alloc = SlotAllocator(4, device=DEVICE, debug_checks=False)
    r1 = alloc.allocate(4)
    assert r1.grew is False
    r2 = alloc.allocate(1)
    assert r2.grew is True
    assert alloc.capacity > 4


def test_logical_ids_never_reused():
    alloc = SlotAllocator(10, device=DEVICE)
    r1 = alloc.allocate(3)
    alloc.retire(r1.physical_idx[:2])
    r2 = alloc.allocate(2)
    assert set(r1.logical_ids.tolist()).isdisjoint(set(r2.logical_ids.tolist()))


def test_epoch_capacity_defaults_to_identity_with_n_active():
    alloc = SlotAllocator(10, device=DEVICE)
    assert alloc.epoch_capacity == 0
    alloc.allocate(5)
    assert alloc.epoch_capacity == alloc.n_active == 5


def test_allocator_consistency_invariant_holds_after_mixed_ops():
    alloc = SlotAllocator(10, device=DEVICE)
    r1 = alloc.allocate(6)
    alloc.retire(r1.physical_idx[[1, 3]])
    alloc.allocate(2)
    alloc.assert_consistent()  # raises on failure


def test_allocator_checkpoint_round_trip():
    alloc = SlotAllocator(8, device=DEVICE)
    r1 = alloc.allocate(5)
    alloc.retire(r1.physical_idx[[0, 2]])
    state = alloc.capture()
    restored = SlotAllocator.restore(state, device=DEVICE)
    assert restored.capacity == alloc.capacity
    assert restored.n_active == alloc.n_active
    assert torch.equal(restored.logical_ids, alloc.logical_ids)
    assert restored.logical_to_physical == alloc.logical_to_physical
    restored.assert_consistent()


# ---------------------------------------------------------------------------
# Compaction (batched swap-delete)
# ---------------------------------------------------------------------------


def test_retire_interior_row_moves_exactly_one_tail_donor():
    alloc = SlotAllocator(10, device=DEVICE)
    r1 = alloc.allocate(5)  # physical rows 0-4
    plan = alloc.retire(torch.tensor([1]))
    assert isinstance(plan, CompactionPlan)
    # hole at row 1; donor is the tail row (4, the last active row before shrink)
    assert plan.dst_indices.tolist() == [1]
    assert plan.src_indices.tolist() == [4]
    assert alloc.n_active == 4


def test_retire_tail_row_needs_no_donor():
    alloc = SlotAllocator(10, device=DEVICE)
    alloc.allocate(5)  # rows 0-4
    plan = alloc.retire(torch.tensor([4]))  # already the last row
    assert plan.dst_indices.numel() == 0
    assert plan.src_indices.numel() == 0
    assert alloc.n_active == 4


def test_retire_multiple_interior_rows_batched_compaction():
    alloc = SlotAllocator(20, device=DEVICE)
    alloc.allocate(10)  # rows 0-9
    plan = alloc.retire(torch.tensor([0, 3, 7]))
    assert alloc.n_active == 7
    # 3 holes total; row 7 is >= new_n_active(7) so needs no donor; rows 0,3 do.
    # Donors are the surviving old-tail rows outside the retired set: of
    # [7, 10) = {7, 8, 9}, 7 is itself retired, leaving {8, 9} as the only
    # candidates -- deterministic, not just "some permutation of something".
    assert plan.dst_indices.tolist() == [0, 3]
    assert plan.src_indices.tolist() == [8, 9]
    assert plan.dst_indices.numel() == plan.src_indices.numel() == 2
    alloc.assert_consistent()


def test_retire_everything_leaves_empty_prefix():
    alloc = SlotAllocator(10, device=DEVICE)
    r1 = alloc.allocate(5)
    plan = alloc.retire(r1.physical_idx)
    assert alloc.n_active == 0
    assert plan.new_n_active == 0
    alloc.assert_consistent()


def test_retire_rejects_duplicate_physical_indices():
    alloc = SlotAllocator(10, device=DEVICE)
    alloc.allocate(5)
    with pytest.raises(AssertionError):
        alloc.retire(torch.tensor([1, 1]))


def test_retire_rejects_index_outside_active_prefix():
    alloc = SlotAllocator(10, device=DEVICE)
    alloc.allocate(3)
    with pytest.raises(AssertionError):
        alloc.retire(torch.tensor([5]))  # beyond n_active


def test_compaction_preserves_surviving_logical_identity():
    alloc = SlotAllocator(10, device=DEVICE)
    r1 = alloc.allocate(5)
    survivor_logical = r1.logical_ids[4].item()  # row 4's logical id, will be moved to row 1
    alloc.retire(torch.tensor([1]))
    assert alloc.logical_to_physical[survivor_logical] == 1


def test_compaction_at_registry_level_moves_buffer_values_with_logical_identity():
    alloc, reg = _make_registry(10)
    reg.insert({"scores": torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0]),
                "tags": torch.arange(15, dtype=torch.float32).reshape(5, 3)})
    # retire physical row 1 (score=20) -- row 4 (score=50) should become the new row 1
    reg.retire(torch.tensor([1]))
    assert reg.active_view("scores").tolist() == [10.0, 50.0, 30.0, 40.0]
    assert reg.buffers["scores"][1].item() == 50.0


# ---------------------------------------------------------------------------
# CapacityBufferRegistry: plain-buffer insert/retire
# ---------------------------------------------------------------------------


def test_insert_into_free_capacity_does_not_reallocate_buffers():
    alloc, reg = _make_registry(10)
    reg.insert({"scores": torch.zeros(3), "tags": torch.zeros(3, 3)})
    scores_ptr = reg.buffers["scores"].data_ptr()
    reg.insert({"scores": torch.zeros(2), "tags": torch.zeros(2, 3)})
    assert reg.buffers["scores"].data_ptr() == scores_ptr
    assert reg.growth_events == 0


def test_insert_values_land_at_allocated_physical_rows():
    alloc, reg = _make_registry(10)
    reg.insert({"scores": torch.tensor([1.0, 2.0, 3.0]), "tags": torch.zeros(3, 3)})
    assert reg.buffers["scores"][:3].tolist() == [1.0, 2.0, 3.0]


def test_insert_beyond_capacity_triggers_exactly_one_growth_event():
    alloc, reg = _make_registry(4)
    reg.insert({"scores": torch.zeros(4), "tags": torch.zeros(4, 3)})
    reg.insert({"scores": torch.zeros(2), "tags": torch.zeros(2, 3)})
    assert reg.growth_events == 1
    assert alloc.capacity > 4


def test_insert_missing_buffer_key_raises():
    alloc, reg = _make_registry(10)
    with pytest.raises(ValueError):
        reg.insert({"scores": torch.zeros(2)})


def test_assert_aligned_catches_manual_misalignment():
    alloc, reg = _make_registry(10)
    reg.buffers["scores"] = torch.zeros(5)
    with pytest.raises(AssertionError):
        reg.assert_aligned()


def test_insert_ignores_extra_unregistered_keys_in_rows():
    alloc, reg = _make_registry(10)
    reg.insert({"scores": torch.zeros(2), "tags": torch.zeros(2, 3), "unrelated": torch.zeros(2)})


def test_late_buffer_registration_backfills_capacity_sized_default():
    alloc, reg = _make_registry(10)
    reg.insert({"scores": torch.zeros(3), "tags": torch.zeros(3, 3)})
    consensus = reg.register_buffer_late("consensus", dtype=torch.float32, fill_value=-1.0)
    assert consensus.shape[0] == alloc.capacity
    assert bool((consensus == -1.0).all())


def test_active_view_is_a_zero_copy_slice_not_a_gather():
    alloc, reg = _make_registry(10)
    reg.insert({"scores": torch.tensor([1.0, 2.0, 3.0]), "tags": torch.zeros(3, 3)})
    view = reg.active_view("scores")
    # A true slice shares storage with the backing buffer (same data_ptr);
    # the pre-W3-49 boolean-mask gather would NOT (it materializes a copy).
    assert view.data_ptr() == reg.buffers["scores"].data_ptr()
    assert view.shape[0] == alloc.n_active == 3


def test_active_view_of_param_backed_buffer_returns_the_live_parameter_not_a_buffer_slice():
    """W3-52 regression test: active_view() on a param-backed name must
    return the SAME object as param_for(), not a fresh slice of the plain
    backing tensor -- the two are different tensor objects sharing storage
    (the backing tensor never requires grad; the live Parameter does), and
    the pre-fix version of active_view() returned the former unconditionally.
    """
    alloc, reg, optimizer = _make_param_registry(10)
    reg.insert({"means": torch.zeros(3, 3)})
    view = reg.active_view("means")
    assert view is reg.param_for("means")
    assert view is reg.active_parameter("means")


def test_active_parameter_raises_for_non_param_backed_name():
    alloc, reg = _make_registry(10)
    with pytest.raises(KeyError):
        reg.active_parameter("scores")


def test_active_view_of_param_backed_buffer_is_differentiable():
    """The actual bug this guards against: active_view() must be usable as
    the input to a real backward()/optimizer.step() round trip, exactly as
    a renderer would use it. The pre-fix version would hand back a
    requires_grad=False tensor (a slice of the plain backing tensor), so
    .sum().backward() would either raise or silently produce no gradient
    on param_for("means") -- this test would have failed against that
    version.
    """
    alloc, reg, optimizer = _make_param_registry(10, lr=1.0)
    reg.insert({"means": torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])})

    view = reg.active_view("means")
    assert view.requires_grad

    loss = (view ** 2).sum()
    loss.backward()

    param = reg.param_for("means")
    assert param.grad is not None
    assert torch.allclose(param.grad, 2 * view.detach())

    backing_before = reg.bindings["means"].backing[:2].clone()
    optimizer.step()
    backing_after = reg.bindings["means"].backing[:2]
    assert not torch.allclose(backing_before, backing_after), (
        "optimizer.step() must mutate the backing storage that active_view() "
        "was computed from -- if it doesn't, active_view() was not actually "
        "wired to the live optimizer Parameter"
    )


# ---------------------------------------------------------------------------
# CapacityOptimizerBinding: registration, momentum lifecycle, rebind
# ---------------------------------------------------------------------------


def test_register_param_returns_optimizer_and_binds_immediately():
    alloc, reg, optimizer = _make_param_registry(10)
    assert reg.param_for("means") is not None
    assert reg.param_for("means") in optimizer.param_groups[0]["params"]


def test_optimizer_visible_shape_equals_n_active_not_capacity():
    alloc, reg, optimizer = _make_param_registry(10)
    reg.insert({"means": torch.zeros(3, 3)})
    param = reg.param_for("means")
    assert param.shape[0] == alloc.n_active == 3
    assert param.shape[0] != alloc.capacity


def test_parameter_aliases_backing_storage_after_insert():
    alloc, reg, optimizer = _make_param_registry(10)
    reg.insert({"means": torch.ones(3, 3)})
    param = reg.param_for("means")
    backing = reg.bindings["means"].backing
    assert param.data_ptr() == backing[: alloc.n_active].data_ptr()


def test_optimizer_step_mutates_backing_storage_in_place():
    alloc, reg, optimizer = _make_param_registry(10, lr=1.0)
    reg.insert({"means": torch.zeros(3, 3)})
    param = reg.param_for("means")
    param.grad = torch.ones_like(param)
    optimizer.step()
    backing = reg.bindings["means"].backing
    assert torch.allclose(backing[:3], param.detach())
    assert not torch.allclose(backing[:3], torch.zeros(3, 3))


def test_new_rows_start_with_zero_momentum():
    alloc, reg, optimizer = _make_param_registry(10, lr=1.0)
    reg.insert({"means": torch.zeros(2, 3)})
    param = reg.param_for("means")
    param.grad = torch.ones_like(param)
    optimizer.step()  # seeds real momentum for rows 0-1

    reg.insert({"means": torch.zeros(2, 3)})  # rows 2-3
    new_param = reg.param_for("means")
    state = optimizer.state[new_param]
    assert torch.allclose(state["exp_avg"][2:4], torch.zeros(2, 3))
    assert torch.allclose(state["exp_avg_sq"][2:4], torch.zeros(2, 3))


def test_rebind_preserves_existing_momentum_across_insert():
    alloc, reg, optimizer = _make_param_registry(10, lr=1.0)
    reg.insert({"means": torch.zeros(2, 3)})
    param = reg.param_for("means")
    param.grad = torch.ones_like(param)
    optimizer.step()
    momentum_before = optimizer.state[param]["exp_avg"].clone()

    reg.insert({"means": torch.zeros(1, 3)})  # triggers a rebind
    new_param = reg.param_for("means")
    momentum_after = optimizer.state[new_param]["exp_avg"][:2]
    assert torch.allclose(momentum_before, momentum_after)


def test_momentum_follows_gaussian_through_compaction():
    alloc, reg, optimizer = _make_param_registry(10, lr=1.0)
    reg.insert({"means": torch.zeros(5, 3)})
    param = reg.param_for("means")
    # distinct gradients per row so momentum differs per row
    param.grad = torch.arange(1, 16, dtype=torch.float32).reshape(5, 3)
    optimizer.step()
    row4_momentum_before = optimizer.state[param]["exp_avg"][4].clone()

    reg.retire(torch.tensor([1]))  # row 4's data/momentum should move to row 1
    new_param = reg.param_for("means")
    row1_momentum_after = optimizer.state[new_param]["exp_avg"][1]
    assert torch.allclose(row4_momentum_before, row1_momentum_after)


def test_retire_before_any_optimizer_step_then_step_produces_sane_momentum():
    """W3-52 regression test (advisor-flagged gap): retire() before the
    optimizer has ever stepped means optimizer.state is empty, so
    sync_momentum_from_optimizer() returns early and apply_compaction()'s
    loop over an empty _momentum_backing is a correct no-op -- but this was
    previously true "by accident, not by test." Pins down that a later
    step(), taken AFTER an insert()+retire() with no step() in between,
    produces zero-initialised (not garbage-carried-from-a-wrong-row)
    momentum for the surviving Gaussians.
    """
    alloc, reg, optimizer = _make_param_registry(10, lr=1.0)
    reg.insert({"means": torch.zeros(5, 3)})
    assert optimizer.state.get(reg.param_for("means"), {}) == {}  # never stepped yet

    reg.retire(torch.tensor([1]))  # interior row; row 4 donates into slot 1
    param = reg.param_for("means")
    assert param.shape[0] == 4

    param.grad = torch.ones_like(param)
    optimizer.step()
    exp_avg = optimizer.state[param]["exp_avg"]
    assert exp_avg.shape[0] == 4
    assert torch.isfinite(exp_avg).all()
    # First step's Adam update from a zero-initialised exp_avg is
    # deterministic: exp_avg = (1 - beta1) * grad for every surviving row,
    # uniformly (all grads were 1.0) -- any row that inherited garbage
    # momentum from a stale physical slot would break this.
    beta1 = optimizer.param_groups[0]["betas"][0]
    expected = torch.full_like(exp_avg, 1.0 - beta1)
    assert torch.allclose(exp_avg, expected)


def test_retired_slots_receive_no_further_optimizer_updates():
    # Retire an INTERIOR row (0) so the tail row (2) is used as a donor and
    # moved into the vacated slot. The donor's old physical slot (2) then
    # becomes the one beyond the new n_active=2 prefix, and must never be
    # touched by a later optimizer step. Retiring only the tail row (as an
    # earlier version of this test did) leaves compaction a no-op and can't
    # actually exercise the "retired slot excluded from optimizer" invariant.
    alloc, reg, optimizer = _make_param_registry(10, lr=1.0)
    reg.insert({"means": torch.zeros(3, 3)})
    param = reg.param_for("means")
    param.grad = torch.ones_like(param)
    optimizer.step()

    reg.retire(torch.tensor([0]))  # interior row retired; row 2 donates into slot 0
    new_param = reg.param_for("means")
    assert new_param.shape[0] == 2  # retired row's old data is no longer optimizer-visible
    backing = reg.bindings["means"].backing
    stale_value = backing[2].clone()  # the donor's now-vacated old slot
    new_param.grad = torch.ones_like(new_param)
    optimizer.step()
    # row 2 (now inactive, beyond n_active) must be untouched by this step
    assert torch.equal(backing[2], stale_value)


def test_capacity_growth_extends_momentum_backing():
    alloc, reg, optimizer = _make_param_registry(2, lr=1.0)
    reg.insert({"means": torch.zeros(2, 3)})
    param = reg.param_for("means")
    param.grad = torch.ones_like(param)
    optimizer.step()

    reg.insert({"means": torch.zeros(1, 3)})  # exceeds capacity=2, triggers growth
    assert reg.growth_events == 1
    binding = reg.bindings["means"]
    for tensor in binding._momentum_backing.values():
        assert tensor.shape[0] == alloc.capacity


def test_param_object_replaced_on_every_mutating_call():
    alloc, reg, optimizer = _make_param_registry(10)
    p0 = reg.param_for("means")
    reg.insert({"means": torch.zeros(1, 3)})
    p1 = reg.param_for("means")
    assert p1 is not p0
    reg.retire(torch.tensor([0]))
    p2 = reg.param_for("means")
    assert p2 is not p1


def test_optimizer_momentum_unaffected_by_unrelated_param_sharing_optimizer():
    alloc = SlotAllocator(10, device=DEVICE)
    reg = CapacityBufferRegistry(alloc)
    means_backing = torch.zeros(10, 3, device=DEVICE)
    other = nn.Parameter(torch.zeros(4, device=DEVICE))

    captured_optimizer = {}

    def factory(params):
        opt = torch.optim.Adam(params + [other], lr=1.0, eps=1e-15)
        captured_optimizer["opt"] = opt
        return opt

    optimizer = reg.register_param("means", means_backing, factory)
    other.grad = torch.ones_like(other)
    reg.insert({"means": torch.ones(2, 3)})
    param = reg.param_for("means")
    param.grad = torch.ones_like(param)
    optimizer.step()
    assert other in optimizer.state
    assert not torch.allclose(other.detach(), torch.zeros(4))


# ---------------------------------------------------------------------------
# Full checkpoint round trip
# ---------------------------------------------------------------------------


def test_full_checkpoint_round_trip_capacity_active_logicalids_version():
    alloc, reg = _make_registry(8)
    reg.insert({"scores": torch.tensor([1.0, 2.0, 3.0]), "tags": torch.arange(9, dtype=torch.float32).reshape(3, 3)})
    reg.retire(torch.tensor([0]))
    mgr = MutationEpochManager()
    mgr.begin_mutation_barrier()
    mgr.record_structural_change("retire", 1)
    mgr.end_mutation_barrier()

    state = reg.capture()
    epoch_state = mgr.capture()

    restored_alloc = SlotAllocator.restore(state["allocator"], device=DEVICE)
    restored_reg = CapacityBufferRegistry(restored_alloc)
    restored_reg.register_buffer("scores", torch.zeros(restored_alloc.capacity, dtype=torch.float32))
    restored_reg.register_buffer("tags", torch.zeros(restored_alloc.capacity, 3, dtype=torch.float32))
    restored_reg.restore_buffers(state, device=DEVICE)
    restored_mgr = MutationEpochManager()
    restored_mgr.restore(epoch_state)

    assert restored_alloc.n_active == alloc.n_active
    assert torch.equal(restored_alloc.logical_ids, alloc.logical_ids)
    assert torch.equal(restored_reg.active_view("scores"), reg.active_view("scores"))
    assert restored_mgr.topology_version == mgr.topology_version == 1


# ---------------------------------------------------------------------------
# MutationEpochManager (unchanged behavior, retained coverage)
# ---------------------------------------------------------------------------


def test_stable_epoch_with_zero_structural_changes_does_not_bump_version():
    mgr = MutationEpochManager()
    mgr.begin_mutation_barrier()
    mgr.record_structural_change("insert", 0)
    bumped = mgr.end_mutation_barrier()
    assert bumped is False
    assert mgr.topology_version == 0


def test_real_structural_change_bumps_version_by_exactly_one():
    mgr = MutationEpochManager()
    mgr.begin_mutation_barrier()
    mgr.record_structural_change("insert", 5)
    bumped = mgr.end_mutation_barrier()
    assert bumped is True
    assert mgr.topology_version == 1


def test_multiple_epochs_accumulate_version_monotonically():
    mgr = MutationEpochManager()
    for _ in range(3):
        mgr.begin_mutation_barrier()
        mgr.record_structural_change("insert", 1)
        mgr.end_mutation_barrier()
    assert mgr.topology_version == 3
    assert mgr.epoch_count == 3


def test_record_structural_change_outside_barrier_raises():
    mgr = MutationEpochManager()
    with pytest.raises(AssertionError):
        mgr.record_structural_change("insert", 1)


def test_nested_or_double_begin_raises():
    mgr = MutationEpochManager()
    mgr.begin_mutation_barrier()
    with pytest.raises(AssertionError):
        mgr.begin_mutation_barrier()


def test_stable_epoch_guard_passes_when_param_object_untouched():
    alloc, reg, optimizer = _make_param_registry(10)
    reg.insert({"means": torch.zeros(2, 3)})
    mgr = MutationEpochManager()
    with mgr.stable_epoch({"means": reg.param_for("means")}):
        pass  # no mutation


def test_stable_epoch_guard_raises_when_rebind_happens_mid_epoch():
    """W3-49 'static-epoch readiness': a mutating call (insert/retire, which
    calls CapacityOptimizerBinding.rebind and therefore replaces the tracked
    Parameter object) happening inside an open stable_epoch() context must
    be caught -- this is the concrete regression test for the invariant
    that no Parameter replacement/backing allocation/topology mutation may
    occur inside a stable optimization epoch. `tracked` is the SAME dict
    object handed to stable_epoch() -- a real integration polls
    reg.param_for(name) into this dict after every mutating call, exactly
    as this test does, so the guard's __exit__ check (which reads the dict
    by reference, not a value snapshot) actually sees the replacement."""
    alloc, reg, optimizer = _make_param_registry(10)
    reg.insert({"means": torch.zeros(2, 3)})
    mgr = MutationEpochManager()
    tracked = {"means": reg.param_for("means")}
    with pytest.raises(AssertionError):
        with mgr.stable_epoch(tracked):
            reg.insert({"means": torch.zeros(1, 3)})  # rebinds
            tracked["means"] = reg.param_for("means")  # simulates a caller noticing the rebind
