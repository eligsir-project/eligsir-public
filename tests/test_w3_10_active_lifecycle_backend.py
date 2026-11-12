"""Wave 3, PR W3-10: ``AlignedBufferRegistry`` / ``ActivationPhaseConfig``
real-mutation tests.

Requires a real GsplatGaussianModel + real optimizers (ROCm/CUDA + gsplat),
matching the existing convention in
``eligsir/optimization/gaussian/tests/test_optimizer_session_integration.py``
-- there is no CPU rendering/model path in this codebase. Run inside the
training container per CLAUDE.md:

    docker compose run --rm train python -m pytest tests/test_w3_10_active_lifecycle_backend.py -q
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

CUDA_AVAILABLE = torch.cuda.is_available()

try:
    import gsplat  # noqa: F401

    GSPLAT_AVAILABLE = True
except Exception:
    GSPLAT_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not (CUDA_AVAILABLE and GSPLAT_AVAILABLE),
    reason=(
        "test_w3_10_active_lifecycle_backend requires CUDA/ROCm + gsplat; "
        "run inside the training container per CLAUDE.md: docker compose "
        "run --rm train python -m pytest tests/test_w3_10_active_lifecycle_backend.py -q"
    ),
)

from eligsir.optimization.gaussian.strategies.contracts import (  # noqa: E402
    GaussianLifecycleState,
    LifecycleUpdateBatch,
    MutationReasonSummary,
    PopulationMutationPlan,
    SurfaceSeedRequest,
)
from eligsir.optimization.gaussian.strategies.live.backend import (  # noqa: E402
    RETIRE_OPACITY_LOGIT_FLOOR,
    ActivationPhaseConfig,
    AlignedBufferRegistry,
    constrain_position_delta,
)
from eligsir.optimization.gaussian.strategies.live.state import (  # noqa: E402
    LivePopulationState,
    state_to_code,
)


def _build_model(n: int = 24, seed: int = 0):
    import numpy as np
    from types import SimpleNamespace

    from scene.gsplat_model import GsplatGaussianModel

    rng = np.random.default_rng(seed)
    points = rng.uniform(-0.5, 0.5, size=(n, 3)).astype(np.float64)
    colors = rng.uniform(0.2, 0.8, size=(n, 3)).astype(np.float64)
    pcd = SimpleNamespace(points=points, colors=colors)

    g = GsplatGaussianModel(sh_degree=0)
    g.create_from_pcd(pcd, spatial_lr_scale=1.0, init_scale_mode="fixed", init_scale=0.05)

    from eligsir.cli.parser import build_training_parser

    bundle = build_training_parser()
    ns = bundle.parser.parse_args([])
    ns.optimizer_type = "adam"
    opt = bundle.optimization_params.extract(ns)
    for key, value in vars(ns).items():
        if not hasattr(opt, key):
            setattr(opt, key, value)
    g.training_setup(opt)
    return g


def _aligned_state(model, *, topology_version: int = 0, iteration: int = 0) -> LivePopulationState:
    n = int(model.get_xyz.shape[0])
    state = LivePopulationState()
    state.align(n, topology_version, iteration=iteration, device=model.get_xyz.device)
    return state


def _empty_lifecycle_batch(device) -> LifecycleUpdateBatch:
    empty = torch.empty(0, dtype=torch.int64, device=device)
    empty8 = torch.empty(0, dtype=torch.int8, device=device)
    return LifecycleUpdateBatch(
        geometry_indices=empty,
        geometry_old_state=empty8,
        geometry_new_state=empty8,
        geometry_reasons=(),
        appearance_indices=empty,
        appearance_old_state=empty8,
        appearance_new_state=empty8,
        appearance_reasons=(),
    )


def _plan(
    *,
    device,
    topology_version: int,
    retire_indices=None,
    reseed_requests=(),
    safety_fuse_triggers=None,
) -> PopulationMutationPlan:
    retire_indices = (
        retire_indices if retire_indices is not None else torch.empty(0, dtype=torch.int64, device=device)
    )
    prune_indices = torch.empty(0, dtype=torch.int64, device=device)
    reasons = MutationReasonSummary(safety_fuse_triggers=dict(safety_fuse_triggers or {}))
    return PopulationMutationPlan(
        retire_indices=retire_indices,
        prune_indices=prune_indices,
        reseed_requests=reseed_requests,
        growth_requests=(),
        lifecycle_updates=_empty_lifecycle_batch(device),
        reasons=reasons,
        observe_only=False,
        iteration=0,
        topology_version=topology_version,
    )


# ---------------------------------------------------------------------------
# All-phases-off no-op (protects every prior Wave-3 PR's guarantee for the
# out-of-the-box default).
# ---------------------------------------------------------------------------


def test_all_phases_off_is_bit_identical_noop():
    model = _build_model()
    state = _aligned_state(model)
    registry = AlignedBufferRegistry(model, state)

    before = {name: p.detach().clone() for name, p in model.params.items()}
    n_before = int(model.get_xyz.shape[0])

    retire_idx = torch.tensor([0, 1, 2], dtype=torch.int64, device=model.get_xyz.device)
    request = SurfaceSeedRequest(cell_id=0, utility=1.0, reason="test", position=torch.zeros(3))
    plan = _plan(
        device=model.get_xyz.device,
        topology_version=state.topology_version,
        retire_indices=retire_idx,
        reseed_requests=(request,),
    )

    report = registry.apply(plan, phases=ActivationPhaseConfig(), iteration=1)

    assert report.retired_count == 0
    assert report.reseeded_count == 0
    for name, tensor in model.params.items():
        assert torch.equal(before[name], tensor.detach()), f"{name} mutated with all phases off"
    assert int(model.get_xyz.shape[0]) == n_before


# ---------------------------------------------------------------------------
# Phase B: retirement
# ---------------------------------------------------------------------------


def test_phase_b_retirement_zeroes_opacity_only_and_no_row_removed():
    model = _build_model()
    state = _aligned_state(model)
    registry = AlignedBufferRegistry(model, state, max_unreclaimed_retired=1000)  # W3-45: bypass leak cap, testing retirement mechanics

    n_before = int(model.get_xyz.shape[0])
    means_before = model.params["means"].detach().clone()
    retire_idx = torch.tensor([2, 5, 7], dtype=torch.int64, device=model.get_xyz.device)
    plan = _plan(device=model.get_xyz.device, topology_version=state.topology_version, retire_indices=retire_idx)

    report = registry.apply(plan, phases=ActivationPhaseConfig(phase_b_retirement=True), iteration=1)

    assert report.retired_count == 3
    assert int(model.get_xyz.shape[0]) == n_before, "retirement must not change row count"
    assert torch.equal(model.params["means"].detach(), means_before), "retirement must not move positions"
    opacities = model.params["opacities"].detach()
    for i in [2, 5, 7]:
        assert opacities[i].item() == pytest.approx(RETIRE_OPACITY_LOGIT_FLOOR)
    untouched = [i for i in range(n_before) if i not in (2, 5, 7)]
    for i in untouched[:3]:
        assert opacities[i].item() != pytest.approx(RETIRE_OPACITY_LOGIT_FLOOR)


def test_phase_b_retirement_zeroes_leftover_momentum():
    """Found in review: without also zeroing Adam momentum, a retired
    row's *accumulated* exp_avg/exp_avg_sq would keep pushing its opacity
    off the floor on the next real optimizer.step() -- retirement would
    not be durable. Drives a real step BEFORE retiring (to build up
    non-zero momentum on the opacities optimizer), retires, and checks the
    optimizer's own stored state for those rows is exactly zero
    immediately after (the mechanism that makes durability possible)."""
    model = _build_model()
    state = _aligned_state(model)
    registry = AlignedBufferRegistry(model, state, max_unreclaimed_retired=1000)  # W3-45: bypass leak cap, testing retirement mechanics

    for p in model.params.values():
        p.grad = torch.ones_like(p)
    model.optimizer.step()
    model.optimizer.zero_grad()

    opt = model.optimizers["opacities"]
    p = opt.param_groups[0]["params"][0]
    st_before = opt.state.get(p, {})
    zeros2 = torch.zeros(2, device=st_before["exp_avg"].device)
    assert not torch.equal(st_before["exp_avg"][[0, 1]], zeros2), "sanity: momentum really was non-zero"

    retire_idx = torch.tensor([0, 1], dtype=torch.int64, device=model.get_xyz.device)
    plan = _plan(device=model.get_xyz.device, topology_version=state.topology_version, retire_indices=retire_idx)
    registry.apply(plan, phases=ActivationPhaseConfig(phase_b_retirement=True), iteration=1)

    st_after = opt.state.get(p, {})
    assert torch.equal(st_after["exp_avg"][[0, 1]], zeros2)
    assert torch.equal(st_after["exp_avg_sq"][[0, 1]], zeros2)


def test_phase_b_retirement_survives_one_subsequent_optimizer_step():
    """The real durability check: retire, then run ONE more real
    (gradient, optimizer.step()) cycle exactly as training would, and
    confirm the retired rows' opacity has not moved measurably off the
    floor. This is the property "retired rows become inactive" (plan
    section 15.3) actually depends on -- an opacity value alone,
    unverified across a subsequent step, would not have caught the
    leftover-momentum bug found in review.

    Honest limitation documented here (not hidden): this zeroes
    *momentum*, not the gradient itself -- a retired row still receives a
    real backward-pass gradient from whatever loss terms still touch it
    (e.g. opacity_reg pulls low opacity further toward zero, which is
    harmless; an SSIM/L1 term with residual signal on that row could in
    principle still nudge it over MANY steps). This test only asserts
    survival across one step with a synthetic ones-gradient (a
    stress case, not a real render loss) precisely because a full
    guarantee across arbitrarily many real training steps would require
    routing retired rows through gradient-masking
    (``gradient_policies/active_mask.py``), which this PR does not do --
    see the W3-10 completion doc's honest gaps.
    """
    model = _build_model()
    state = _aligned_state(model)
    registry = AlignedBufferRegistry(model, state, max_unreclaimed_retired=1000)  # W3-45: bypass leak cap, testing retirement mechanics

    retire_idx = torch.tensor([0, 1], dtype=torch.int64, device=model.get_xyz.device)
    plan = _plan(device=model.get_xyz.device, topology_version=state.topology_version, retire_indices=retire_idx)
    registry.apply(plan, phases=ActivationPhaseConfig(phase_b_retirement=True), iteration=1)

    opacities = model.params["opacities"]
    assert opacities.detach()[retire_idx].allclose(
        torch.full((2,), RETIRE_OPACITY_LOGIT_FLOOR, device=opacities.device)
    )

    for p in model.params.values():
        p.grad = torch.ones_like(p)
    model.optimizer.step()
    model.optimizer.zero_grad()

    after = opacities.detach()[retire_idx]
    # With momentum zeroed, a single step's movement is bounded by the
    # optimizer's own (small) learning rate applied to a fresh unit
    # gradient -- nowhere near enough to meaningfully un-retire the row
    # (contrast: without the momentum fix, leftover accumulated momentum
    # from many prior steps could move it by a much larger amount
    # instantly).
    assert (after - RETIRE_OPACITY_LOGIT_FLOOR).abs().max().item() < 0.5


def test_phase_b_off_leaves_retirement_unapplied_even_if_plan_proposes_it():
    model = _build_model()
    state = _aligned_state(model)
    registry = AlignedBufferRegistry(model, state)
    opacities_before = model.params["opacities"].detach().clone()

    retire_idx = torch.tensor([1, 2], dtype=torch.int64, device=model.get_xyz.device)
    plan = _plan(device=model.get_xyz.device, topology_version=state.topology_version, retire_indices=retire_idx)

    report = registry.apply(
        plan, phases=ActivationPhaseConfig(phase_c_reseeding=True), iteration=1  # B deliberately off
    )

    assert report.retired_count == 0
    assert torch.equal(model.params["opacities"].detach(), opacities_before)


# ---------------------------------------------------------------------------
# Phase C: reseeding, deliverable 2 (explicit optimizer-state reset)
# ---------------------------------------------------------------------------


def test_phase_c_reseed_reuses_retired_capacity_with_zero_momentum():
    model = _build_model(n=16)
    state = _aligned_state(model)
    registry = AlignedBufferRegistry(model, state)

    # First, retire (Phase B) two rows and drive real training steps so
    # their Adam momentum is non-zero -- proves reseeding actually resets
    # state rather than the moments happening to already be zero.
    retire_idx = torch.tensor([0, 1], dtype=torch.int64, device=model.get_xyz.device)
    for p in model.params.values():
        p.grad = torch.ones_like(p)
    model.optimizer.step()
    model.optimizer.zero_grad()
    state.geometry_lifecycle[retire_idx] = state_to_code(GaussianLifecycleState.RETIRED)

    n_before = int(model.get_xyz.shape[0])
    device = model.get_xyz.device
    requests = (
        SurfaceSeedRequest(cell_id=0, utility=2.0, reason="high_confidence_surface", position=torch.tensor([1.0, 2.0, 3.0])),
    )
    plan = _plan(device=device, topology_version=state.topology_version, reseed_requests=requests)

    report = registry.apply(plan, phases=ActivationPhaseConfig(phase_c_reseeding=True), iteration=5)

    assert report.reseeded_count == 1
    assert report.pruned_for_reseed_count == 1
    # Net row count unchanged: pruned exactly as many RETIRED rows as added.
    assert int(model.get_xyz.shape[0]) == n_before
    assert state.n == n_before
    assert state.topology_version == 1  # bumped by exactly one structural change

    for name, optimizer in model.optimizers.items():
        p = optimizer.param_groups[0]["params"][0]
        st = optimizer.state.get(p, {})
        assert "exp_avg" in st
        # The new (last) row's momentum must be exactly zero -- explicit
        # optimizer-state reset semantics (deliverable 2), not inherited
        # from the RETIRED row it replaced.
        assert torch.equal(st["exp_avg"][-1], torch.zeros_like(st["exp_avg"][-1])), name
        assert torch.equal(st["exp_avg_sq"][-1], torch.zeros_like(st["exp_avg_sq"][-1])), name

    # New row's LivePopulationState fields are fresh PROVISIONAL, not
    # inherited from the pruned RETIRED row.
    assert int(state.geometry_lifecycle[-1].item()) == state_to_code(GaussianLifecycleState.PROVISIONAL)
    assert state.support_count[-1].item() == 0.0
    assert state.birth_frame[-1].item() == 5


def test_phase_c_off_leaves_row_count_unchanged():
    model = _build_model(n=16)
    state = _aligned_state(model)
    registry = AlignedBufferRegistry(model, state)
    state.geometry_lifecycle[0] = state_to_code(GaussianLifecycleState.RETIRED)
    n_before = int(model.get_xyz.shape[0])

    requests = (SurfaceSeedRequest(cell_id=0, utility=1.0, reason="x", position=torch.zeros(3)),)
    plan = _plan(device=model.get_xyz.device, topology_version=state.topology_version, reseed_requests=requests)
    report = registry.apply(plan, phases=ActivationPhaseConfig(phase_b_retirement=True), iteration=1)  # C off

    assert report.reseeded_count == 0
    assert int(model.get_xyz.shape[0]) == n_before


def test_reseed_topology_version_bump_is_self_consistent_if_caller_reads_it_back():
    """Documents/verifies the topology_version ownership contract from
    the module docstring: if a caller correctly reads
    ``registry.live_state.topology_version`` back after ``apply()`` and
    uses THAT as its next ``align()`` call's version (rather than an
    independently-tracked counter), state is NOT cold-started -- proving
    the registry's own bookkeeping is internally consistent. This is a
    narrower guarantee than "reseed never loses state": if a caller uses
    a stale/independent version instead, `align()`'s existing (W3-09,
    unchanged) policy still cold-starts everything, as documented.
    """
    model = _build_model(n=16)
    state = _aligned_state(model)
    registry = AlignedBufferRegistry(model, state)
    state.geometry_lifecycle[0] = state_to_code(GaussianLifecycleState.RETIRED)
    state.support_count[5] = 7.0  # a survives-if-not-cold-started sentinel on an untouched row

    requests = (SurfaceSeedRequest(cell_id=0, utility=1.0, reason="x", position=torch.zeros(3)),)
    plan = _plan(device=model.get_xyz.device, topology_version=state.topology_version, reseed_requests=requests)
    registry.apply(plan, phases=ActivationPhaseConfig(phase_c_reseeding=True), iteration=1)

    new_version = state.topology_version
    assert new_version == 1
    n_after = int(model.get_xyz.shape[0])

    # Caller correctly reads the bumped version back and re-aligns with it.
    state.align(n_after, new_version, iteration=2, device=model.get_xyz.device)
    # support_count[5] survives untouched because the version matched --
    # no cold start happened.
    assert state.support_count[4].item() == 7.0  # index shifted down by 1 (row 0 was pruned)


def test_stale_topology_version_is_refused():
    model = _build_model()
    state = _aligned_state(model, topology_version=0)
    registry = AlignedBufferRegistry(model, state)
    plan = _plan(device=model.get_xyz.device, topology_version=99)  # wrong on purpose

    from eligsir.optimization.gaussian.strategies.live.state import StaleLiveStateError

    with pytest.raises(StaleLiveStateError):
        registry.apply(plan, phases=ActivationPhaseConfig(phase_b_retirement=True), iteration=1)


# ---------------------------------------------------------------------------
# Deliverable 6: safety-fuse vetoes are never counted as applied/wins.
# ---------------------------------------------------------------------------


def test_vetoed_retirements_are_not_counted_as_applied():
    """Simulates what `SafetyFuses.apply()` already does upstream (drop
    indices from `plan.retire_indices`, record the drop count under a
    `safety_fuse:*` key) and asserts the registry's applied-count field
    reflects only what actually survived to be mutated -- never the
    combination of applied + vetoed. A regression where a future edit
    folded `safety_fuse_triggers` counts back into `retired_count` (e.g.
    "it would have retired N, so let's report N as retired for
    completeness") would fail this test.
    """
    model = _build_model()
    state = _aligned_state(model)
    registry = AlignedBufferRegistry(model, state, max_unreclaimed_retired=1000)  # W3-45: bypass leak cap, testing retirement mechanics

    # Plan as SafetyFuses.apply() would leave it: only 1 index survived the
    # veto (of an original 4 candidates), and the veto is recorded
    # separately.
    surviving_retire_idx = torch.tensor([0], dtype=torch.int64, device=model.get_xyz.device)
    plan = _plan(
        device=model.get_xyz.device,
        topology_version=state.topology_version,
        retire_indices=surviving_retire_idx,
        safety_fuse_triggers={"safety_fuse:max_retire_fraction": 3},
    )

    report = registry.apply(plan, phases=ActivationPhaseConfig(phase_b_retirement=True), iteration=1)

    assert report.retired_count == 1, "only the surviving (non-vetoed) index was applied"
    assert report.safety_fuse_triggers == {"safety_fuse:max_retire_fraction": 3}
    as_dict = report.as_dict()
    assert as_dict["retired_count"] == 1, "vetoed count must never be folded into the applied-count field"


# ---------------------------------------------------------------------------
# W3-45: unreclaimed-retirement cap. Phase C reseeding has no real candidate
# producer anywhere in production code (SurfaceSeedRequest is only ever
# constructed in tests -- see docs/architecture/
# 2026-08-09-w3-45-phase-c-reseed-root-cause.md), so Phase B retirement was
# a pure one-way capacity leak in every real run (confirmed on GPU, see
# docs/architecture/2026-08-09-w3-44-matched-psnr-comparison.md: -1.3dB
# test-PSNR on fr3). These tests cover the cap that closes that leak by
# construction rather than relying on the phase_c_reseeding flag (which
# cannot make retirement safe on its own -- turning it on does not create a
# real reseed candidate producer).
# ---------------------------------------------------------------------------


def test_default_cap_refuses_all_retirement_with_no_prior_reseed():
    """Out-of-the-box default (``max_unreclaimed_retired=0``): retirement
    is refused entirely until something has actually been reseeded, since
    reclaim never happens today regardless of the phase_c_reseeding flag."""
    model = _build_model()
    state = _aligned_state(model)
    registry = AlignedBufferRegistry(model, state)  # default cap == 0

    opacities_before = model.params["opacities"].detach().clone()
    retire_idx = torch.tensor([0, 1, 2], dtype=torch.int64, device=model.get_xyz.device)
    plan = _plan(device=model.get_xyz.device, topology_version=state.topology_version, retire_indices=retire_idx)

    report = registry.apply(plan, phases=ActivationPhaseConfig(phase_b_retirement=True), iteration=1)

    assert report.retired_count == 0
    assert report.safety_fuse_triggers.get("safety_fuse:unreclaimed_retirement_cap") == 3
    assert torch.equal(model.params["opacities"].detach(), opacities_before)


def test_cap_allows_retirement_up_to_configured_headroom_then_blocks_the_rest():
    """A nonzero cap allows retirement up to that many outstanding
    (unreclaimed) rows, then blocks further retirement in the same call --
    proving this is a real cap, not a binary on/off switch."""
    model = _build_model()
    state = _aligned_state(model)
    registry = AlignedBufferRegistry(model, state, max_unreclaimed_retired=2)

    retire_idx = torch.tensor([0, 1, 2, 3], dtype=torch.int64, device=model.get_xyz.device)
    plan = _plan(device=model.get_xyz.device, topology_version=state.topology_version, retire_indices=retire_idx)

    report = registry.apply(plan, phases=ActivationPhaseConfig(phase_b_retirement=True), iteration=1)

    assert report.retired_count == 2
    assert report.safety_fuse_triggers.get("safety_fuse:unreclaimed_retirement_cap") == 2
    opacities = model.params["opacities"].detach()
    for i in [0, 1]:
        assert opacities[i].item() == pytest.approx(RETIRE_OPACITY_LOGIT_FLOOR)
    for i in [2, 3]:
        assert opacities[i].item() != pytest.approx(RETIRE_OPACITY_LOGIT_FLOOR)


def test_reseed_reopens_headroom_for_further_retirement():
    """Once outstanding (retired - reseeded) drops back under the cap via a
    real reseed, further retirement is allowed again -- this is a
    self-correcting rate limiter, not a permanent lockout, so a future real
    candidate producer for Phase C does not need this cap removed."""
    model = _build_model(n=16)
    state = _aligned_state(model)
    registry = AlignedBufferRegistry(model, state, max_unreclaimed_retired=1)

    # First call: retire 1 row, exactly filling the cap.
    retire_idx_1 = torch.tensor([0], dtype=torch.int64, device=model.get_xyz.device)
    plan_1 = _plan(device=model.get_xyz.device, topology_version=state.topology_version, retire_indices=retire_idx_1)
    report_1 = registry.apply(plan_1, phases=ActivationPhaseConfig(phase_b_retirement=True), iteration=1)
    assert report_1.retired_count == 1

    # Second call, same version: a further retirement is refused -- no
    # headroom left (1 outstanding, cap is 1).
    retire_idx_2 = torch.tensor([1], dtype=torch.int64, device=model.get_xyz.device)
    plan_2 = _plan(device=model.get_xyz.device, topology_version=state.topology_version, retire_indices=retire_idx_2)
    report_2 = registry.apply(plan_2, phases=ActivationPhaseConfig(phase_b_retirement=True), iteration=2)
    assert report_2.retired_count == 0

    # Reclaim the one outstanding retirement via a real reseed (marks row 0
    # RETIRED-consuming by construction of apply_reseed, which requires the
    # state to actually be RETIRED -- already true from the first call).
    state.geometry_lifecycle[0] = state_to_code(GaussianLifecycleState.RETIRED)
    requests = (SurfaceSeedRequest(cell_id=0, utility=1.0, reason="x", position=torch.zeros(3)),)
    plan_3 = _plan(
        device=model.get_xyz.device,
        topology_version=state.topology_version,
        reseed_requests=requests,
    )
    report_3 = registry.apply(plan_3, phases=ActivationPhaseConfig(phase_c_reseeding=True), iteration=3)
    assert report_3.reseeded_count == 1

    # Third retirement attempt: headroom reopened (1 retired, 1 reseeded,
    # outstanding back to 0).
    retire_idx_4 = torch.tensor([2], dtype=torch.int64, device=model.get_xyz.device)
    plan_4 = _plan(
        device=model.get_xyz.device,
        topology_version=state.topology_version,
        retire_indices=retire_idx_4,
    )
    report_4 = registry.apply(plan_4, phases=ActivationPhaseConfig(phase_b_retirement=True), iteration=4)
    assert report_4.retired_count == 1


def test_reseed_of_externally_retired_row_does_not_open_unearned_headroom():
    """A row can reach RETIRED via the lifecycle FSM directly (never through
    this registry's own apply_retire -- exactly how
    test_phase_c_reseed_reuses_retired_capacity_with_zero_momentum sets up
    its fixture). If that row is then reseeded, cumulative_reseeded grows
    without a matching cumulative_retired increment on this registry --
    without clamping ``outstanding`` at 0, this would make it negative and
    open retirement headroom that was never earned by an actual retirement
    this registry performed."""
    model = _build_model(n=16)
    state = _aligned_state(model)
    registry = AlignedBufferRegistry(model, state, max_unreclaimed_retired=0)

    # Row 0 reaches RETIRED without ever going through registry.apply_retire.
    state.geometry_lifecycle[0] = state_to_code(GaussianLifecycleState.RETIRED)
    requests = (SurfaceSeedRequest(cell_id=0, utility=1.0, reason="x", position=torch.zeros(3)),)
    plan_reseed = _plan(device=model.get_xyz.device, topology_version=state.topology_version, reseed_requests=requests)
    report_reseed = registry.apply(plan_reseed, phases=ActivationPhaseConfig(phase_c_reseeding=True), iteration=1)
    assert report_reseed.reseeded_count == 1
    # cumulative_reseeded is now 1, cumulative_retired is still 0 on this
    # registry -- outstanding would be -1 without the max(0, ...) clamp.

    retire_idx = torch.tensor([1], dtype=torch.int64, device=model.get_xyz.device)
    plan_retire = _plan(
        device=model.get_xyz.device, topology_version=state.topology_version, retire_indices=retire_idx
    )
    report_retire = registry.apply(plan_retire, phases=ActivationPhaseConfig(phase_b_retirement=True), iteration=2)

    assert report_retire.retired_count == 0, "cap=0 must still refuse retirement -- no unearned headroom"
    assert report_retire.safety_fuse_triggers.get("safety_fuse:unreclaimed_retirement_cap") == 1


# ---------------------------------------------------------------------------
# Phase E: constrained movement (standalone policy function)
# ---------------------------------------------------------------------------


def test_constrain_position_delta_leaves_provisional_unrestricted():
    device = torch.device("cpu")
    lifecycle = torch.tensor(
        [state_to_code(GaussianLifecycleState.PROVISIONAL), state_to_code(GaussianLifecycleState.STABLE)],
        dtype=torch.int8,
        device=device,
    )
    delta = torch.tensor([[10.0, 0.0, 0.0], [10.0, 0.0, 0.0]], device=device)
    out = constrain_position_delta(delta=delta, geometry_lifecycle=lifecycle, max_tangent_delta=0.01)
    assert torch.equal(out[0], delta[0]), "PROVISIONAL row must pass through unrestricted"
    assert out[1].norm().item() == pytest.approx(0.01, abs=1e-5), "STABLE row must be clamped"


def test_constrain_position_delta_normal_decomposition_tighter_along_normal():
    device = torch.device("cpu")
    lifecycle = torch.tensor([state_to_code(GaussianLifecycleState.STABLE)], dtype=torch.int8, device=device)
    normal = torch.tensor([[0.0, 0.0, 1.0]], device=device)
    delta = torch.tensor([[0.0, 0.0, 5.0]], device=device)  # pure normal-direction motion
    out = constrain_position_delta(
        delta=delta, geometry_lifecycle=lifecycle, normal=normal,
        max_tangent_delta=0.01, max_normal_delta=0.002,
    )
    assert out[0, 2].abs().item() == pytest.approx(0.002, abs=1e-6)
