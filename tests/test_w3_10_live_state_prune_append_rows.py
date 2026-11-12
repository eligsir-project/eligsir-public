"""Wave 3, PR W3-10: pure-CPU tests for ``LivePopulationState.prune_rows``/
``append_rows`` (the state-side half of ``AlignedBufferRegistry``'s
prune+add reseed mechanism)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from eligsir.optimization.gaussian.strategies.contracts import (  # noqa: E402
    GaussianLifecycleState,
    PopulationMutationPlan,
)
from eligsir.optimization.gaussian.strategies.live.state import (  # noqa: E402
    LivePopulationState,
    state_to_code,
)


def _state(n: int = 5) -> LivePopulationState:
    s = LivePopulationState()
    s.align(n, topology_version=0, iteration=0, device=torch.device("cpu"))
    return s


def test_prune_rows_drops_masked_rows_from_every_buffer():
    s = _state(5)
    s.support_count = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    keep = torch.tensor([True, False, True, False, True])
    s.prune_rows(keep)
    assert s.n == 3
    assert torch.equal(s.support_count, torch.tensor([1.0, 3.0, 5.0]))
    assert s.geometry_lifecycle.shape[0] == 3


def test_prune_rows_rejects_wrong_size_mask():
    s = _state(5)
    with pytest.raises(ValueError):
        s.prune_rows(torch.tensor([True, False]))


def test_append_rows_adds_fresh_provisional_rows():
    s = _state(3)
    s.support_count = torch.tensor([9.0, 9.0, 9.0])
    s.geometry_lifecycle[:] = state_to_code(GaussianLifecycleState.STABLE)
    s.append_rows(2, iteration=42)
    assert s.n == 5
    assert torch.equal(s.support_count, torch.tensor([9.0, 9.0, 9.0, 0.0, 0.0]))
    assert int(s.geometry_lifecycle[-1].item()) == state_to_code(GaussianLifecycleState.PROVISIONAL)
    assert int(s.geometry_lifecycle[-2].item()) == state_to_code(GaussianLifecycleState.PROVISIONAL)
    assert int(s.geometry_lifecycle[0].item()) == state_to_code(GaussianLifecycleState.STABLE), "old rows untouched"
    assert s.birth_frame[-1].item() == 42


def test_append_rows_zero_count_is_noop():
    s = _state(3)
    before_n = s.n
    s.append_rows(0, iteration=1)
    assert s.n == before_n


def test_population_mutation_plan_observe_only_false_is_now_constructible():
    """W3-09's hard invariant (observe_only can never be False) is
    relaxed as of W3-10 -- the plan itself no longer refuses construction.
    Real gating now lives entirely in
    ``live/backend.py::AlignedBufferRegistry``/``ActivationPhaseConfig``,
    not in this dataclass."""
    empty = torch.empty(0, dtype=torch.int64)
    empty8 = torch.empty(0, dtype=torch.int8)
    from eligsir.optimization.gaussian.strategies.contracts import (
        LifecycleUpdateBatch,
        MutationReasonSummary,
    )

    batch = LifecycleUpdateBatch(
        geometry_indices=empty, geometry_old_state=empty8, geometry_new_state=empty8, geometry_reasons=(),
        appearance_indices=empty, appearance_old_state=empty8, appearance_new_state=empty8, appearance_reasons=(),
    )
    plan = PopulationMutationPlan(
        retire_indices=empty, prune_indices=empty, reseed_requests=(), growth_requests=(),
        lifecycle_updates=batch, reasons=MutationReasonSummary(), observe_only=False,
    )
    assert plan.observe_only is False
