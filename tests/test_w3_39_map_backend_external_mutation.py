"""W3-39: ``GsplatMapBackend.cumulative_inserted``/``cumulative_pruned`` and
``record_external_mutation()``.

Two real ``train_streaming.py`` call sites mutate ``gaussians`` (the model
``GsplatMapBackend`` wraps) via a direct ``gaussians.prune_points(mask)``
call, bypassing ``begin_mutation()``/``commit_mutation()`` entirely (a
"deferred lifecycle prune" and an out-of-bounds/reach-safety prune -- see
``docs/architecture/2026-08-09-w3-39-topology-version-growth-fix.md``). Without
an explicit bookkeeping hook, those prunes would be invisible to
``cumulative_pruned``, and ``LiveWrappedMCMCStrategy``'s pure-append-growth
signal (which trusts "pruned_since_last_call == 0" to mean "safe to treat
as pure append") would misclassify a window containing one of these prunes
plus a tracked insert as pure-append growth -- silently corrupting
``LivePopulationState``'s row alignment. ``record_external_mutation()``
closes that gap; this test is a fast, CPU-only unit test of the bookkeeping
itself (not a GPU re-verification -- that already exists separately).
"""

from __future__ import annotations

from eligsir.map.adapters.gsplat import GsplatMapBackend


class _StubModel:
    """GsplatMapBackend only stores the model reference at construction
    time for these tests -- record_external_mutation() never touches it."""


def test_cumulative_counters_start_at_zero():
    backend = GsplatMapBackend(_StubModel())
    assert backend.cumulative_inserted == 0
    assert backend.cumulative_pruned == 0
    assert backend.topology_version == 0
    assert backend.map_version == 0


def test_record_external_mutation_pruned_increments_counter_and_bumps_topology_version():
    backend = GsplatMapBackend(_StubModel())
    backend.record_external_mutation(pruned=7)
    assert backend.cumulative_pruned == 7
    assert backend.cumulative_inserted == 0
    assert backend.topology_version == 1


def test_record_external_mutation_inserted_increments_counter_and_bumps_topology_version():
    backend = GsplatMapBackend(_StubModel())
    backend.record_external_mutation(inserted=4)
    assert backend.cumulative_inserted == 4
    assert backend.cumulative_pruned == 0
    assert backend.topology_version == 1


def test_record_external_mutation_zero_is_noop():
    backend = GsplatMapBackend(_StubModel())
    backend.record_external_mutation(inserted=0, pruned=0)
    assert backend.cumulative_inserted == 0
    assert backend.cumulative_pruned == 0
    assert backend.topology_version == 0
    assert backend.map_version == 0


def test_record_external_mutation_accumulates_across_multiple_calls():
    backend = GsplatMapBackend(_StubModel())
    backend.record_external_mutation(pruned=3)
    backend.record_external_mutation(pruned=2)
    backend.record_external_mutation(inserted=5)
    assert backend.cumulative_pruned == 5
    assert backend.cumulative_inserted == 5
    assert backend.topology_version == 3


def test_record_external_mutation_bumps_map_version():
    """W3-53 regression test: an external prune/insert changes real map
    content just as a transaction-mediated one does (GsplatMutationTransaction
    .commit() always bumps map_version, structural or not), so leaving
    map_version stale here would let read_view().map_version, snapshot
    versioning, or any evidence-invalidation logic keyed on map_version
    silently miss a real structural change that only went through this hook.
    """
    backend = GsplatMapBackend(_StubModel())
    assert backend.map_version == 0

    backend.record_external_mutation(pruned=7)
    assert backend.map_version == 1

    backend.record_external_mutation(inserted=4)
    assert backend.map_version == 2

    # Zero-effect calls remain a true no-op, including for map_version.
    backend.record_external_mutation(inserted=0, pruned=0)
    assert backend.map_version == 2


def test_delegated_equal_count_replacement_advances_topology_version():
    import torch

    from eligsir.map.arbiter import (
        DelegatedMutationProposal,
        GaussianMutationArbiter,
        ProposalKey,
    )

    class _Rows:
        def __init__(self):
            self.rows = torch.tensor([[1.0], [2.0], [3.0]])

        @property
        def get_xyz(self):
            return self.rows

    model = _Rows()
    backend = GsplatMapBackend(model)

    def _replace_one_row():
        model.rows = torch.tensor([[2.0], [3.0], [4.0]])

    result = GaussianMutationArbiter(backend).commit((
        DelegatedMutationProposal(
            key=ProposalKey(0, 1, 0, "mcmc:test"),
            executor=_replace_one_row,
        ),
    ))

    assert result.inserted == 0
    assert result.change.pruned == 0
    assert backend.map_version == 1
    assert backend.topology_version == 1


def test_record_external_mutation_reflects_in_wrapper_pure_append_signal():
    """End-to-end check that an externally-recorded prune, not just a
    commit_mutation()-mediated one, correctly suppresses
    LiveWrappedMCMCStrategy's pure_append_growth signal -- this is the
    exact scenario the two real train_streaming.py call sites hit."""
    import torch

    from eligsir.utils.strategies.mcmc.live_wrapped import LiveWrappedMCMCStrategy

    class _StubGaussians:
        def __init__(self, n: int):
            self._xyz = torch.zeros(n, 3)

        @property
        def get_xyz(self):
            return self._xyz

    strat = LiveWrappedMCMCStrategy()
    strat._base.step_post_backward = lambda *a, **k: None
    calls = []
    strat._live.observe = lambda **kwargs: calls.append(kwargs)

    class _FakeObserveReport:
        proposed_retire_count = 0
        proposed_prune_count = 0
        proposed_reseed_count = 0
        proposed_growth_count = 0
        overlap_with_current_mcmc = {"overlap_count": 0}

    strat._live.propose = lambda **k: (object(), _FakeObserveReport())

    backend = GsplatMapBackend(_StubModel())
    strat.step_post_backward(_StubGaussians(10), object(), object(), 1, map_backend=backend)

    # Simulate one training iteration: Phase-2 insertion goes through a real
    # transaction (cumulative_inserted += 5), AND a deferred-lifecycle-style
    # prune bypasses the transaction API but is correctly recorded via the
    # new hook (cumulative_pruned += 2). Net growth (n: 10 -> 13), but this
    # IS a structural change (indices shifted) -- must not be pure append.
    backend._cumulative_inserted += 5
    backend.record_external_mutation(pruned=2)
    strat.step_post_backward(_StubGaussians(13), object(), object(), 2, map_backend=backend)

    assert calls[1]["pure_append_growth"] is False
