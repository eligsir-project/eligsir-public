"""W3-39: ``LiveWrappedMCMCStrategy`` must derive a real ``topology_version``
and a ``pure_append_growth`` signal for ``EliGSiRLiveGaussianStrategy.observe()``
instead of the old permanently-pinned-at-0 ``getattr(sched, "topology_version", 0)``.

Two paths under test:
1. Streaming (``map_backend`` passed): the real
   ``inserted``/``pruned`` split since the previous call, via
   ``map_backend.cumulative_inserted``/``cumulative_pruned`` deltas --
   pure append iff inserted > 0 and pruned == 0 since the last call.
2. Offline (``map_backend=None``, ``train.py``'s call site): the weaker
   ``n``-only fallback -- pure append iff ``n`` grew since the last call.

Both are exercised as fast, CPU-only unit tests of
``LiveWrappedMCMCStrategy.step_post_backward`` directly (mirrors
``tests/test_w3_41_apply_telemetry_accumulation.py``'s stubbing style), not
a real GPU/training re-verification -- that is covered separately.
"""

from __future__ import annotations

import torch

from eligsir.utils.strategies.mcmc.live_wrapped import LiveWrappedMCMCStrategy


class _StubGaussians:
    def __init__(self, n: int):
        self._xyz = torch.zeros(n, 3)

    @property
    def get_xyz(self):
        return self._xyz


class _FakeMapBackend:
    def __init__(self, topology_version: int = 0, cumulative_inserted: int = 0, cumulative_pruned: int = 0):
        self.topology_version = topology_version
        self.cumulative_inserted = cumulative_inserted
        self.cumulative_pruned = cumulative_pruned


class _FakeObserveReport:
    proposed_retire_count = 0
    proposed_prune_count = 0
    proposed_reseed_count = 0
    proposed_growth_count = 0
    overlap_with_current_mcmc = {"overlap_count": 0}


def _make_strategy():
    strat = LiveWrappedMCMCStrategy()
    strat._base.step_post_backward = lambda *a, **k: None
    calls = []

    def _fake_observe(**kwargs):
        calls.append(kwargs)

    strat._live.observe = _fake_observe
    strat._live.propose = lambda **k: (object(), _FakeObserveReport())
    return strat, calls


def test_streaming_path_flags_pure_append_when_inserted_and_not_pruned():
    strat, calls = _make_strategy()
    mb = _FakeMapBackend(topology_version=1, cumulative_inserted=0, cumulative_pruned=0)

    # Call 1: first-ever call -- no prior snapshot, must not claim pure append.
    strat.step_post_backward(_StubGaussians(10), object(), object(), 1, map_backend=mb)
    assert calls[0]["pure_append_growth"] is False
    assert calls[0]["topology_version"] == 1

    # Call 2: 50 inserted, 0 pruned since call 1 -- pure append growth.
    mb.topology_version = 2
    mb.cumulative_inserted = 50
    mb.cumulative_pruned = 0
    strat.step_post_backward(_StubGaussians(60), object(), object(), 2, map_backend=mb)
    assert calls[1]["pure_append_growth"] is True
    assert calls[1]["topology_version"] == 2


def test_streaming_path_does_not_flag_pure_append_when_anything_was_pruned():
    strat, calls = _make_strategy()
    mb = _FakeMapBackend(topology_version=1, cumulative_inserted=0, cumulative_pruned=0)
    strat.step_post_backward(_StubGaussians(10), object(), object(), 1, map_backend=mb)

    # Prune-then-larger-insert in the same window: net growth (n increases)
    # but a real structural change -- must NOT be treated as pure append.
    mb.topology_version = 2
    mb.cumulative_inserted = 20
    mb.cumulative_pruned = 5
    strat.step_post_backward(_StubGaussians(25), object(), object(), 2, map_backend=mb)
    assert calls[1]["pure_append_growth"] is False


def test_streaming_path_does_not_flag_pure_append_when_pure_prune_shrink():
    strat, calls = _make_strategy()
    mb = _FakeMapBackend(topology_version=1, cumulative_inserted=0, cumulative_pruned=0)
    strat.step_post_backward(_StubGaussians(10), object(), object(), 1, map_backend=mb)

    mb.topology_version = 2
    mb.cumulative_inserted = 0
    mb.cumulative_pruned = 3
    strat.step_post_backward(_StubGaussians(7), object(), object(), 2, map_backend=mb)
    assert calls[1]["pure_append_growth"] is False


def test_streaming_path_no_change_is_not_flagged():
    strat, calls = _make_strategy()
    mb = _FakeMapBackend(topology_version=1, cumulative_inserted=10, cumulative_pruned=0)
    strat.step_post_backward(_StubGaussians(10), object(), object(), 1, map_backend=mb)
    strat.step_post_backward(_StubGaussians(10), object(), object(), 2, map_backend=mb)
    assert calls[1]["pure_append_growth"] is False
    assert calls[1]["topology_version"] == 1


def test_offline_path_no_map_backend_flags_pure_append_from_n_growth_only():
    strat, calls = _make_strategy()
    strat.step_post_backward(_StubGaussians(10), object(), object(), 1)
    assert calls[0]["pure_append_growth"] is False  # first call, no prior n

    strat.step_post_backward(_StubGaussians(15), object(), object(), 2)
    assert calls[1]["pure_append_growth"] is True

    strat.step_post_backward(_StubGaussians(15), object(), object(), 3)
    assert calls[2]["pure_append_growth"] is False  # unchanged n

    strat.step_post_backward(_StubGaussians(12), object(), object(), 4)
    assert calls[3]["pure_append_growth"] is False  # shrink


def test_offline_path_topology_version_falls_back_to_sched_getattr():
    """When map_backend is None, topology_version must still read from
    sched (today's pre-W3-39 mechanism) -- this fallback is unchanged,
    only pure_append_growth is new for the offline path."""
    strat, calls = _make_strategy()

    class _Sched:
        topology_version = 7

    strat.step_post_backward(_StubGaussians(10), object(), _Sched(), 1)
    assert calls[0]["topology_version"] == 7
