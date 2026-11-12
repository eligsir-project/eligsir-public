from __future__ import annotations

from types import SimpleNamespace

import pytest

# The single test below imports eligsir.runtime.causal_prepared_ready
# lazily inside its own body, which transitively needs torch (via
# prepared_runtime_bridge -> prepared_source_producer). That's invisible to
# tests/_torch_collect_scan.py's static scanner (it only follows module-level
# imports), so without this guard the file hard-fails with a raw
# ModuleNotFoundError in the torch-less CPU smoke lane instead of skipping.
torch = pytest.importorskip("torch")


class _Lease:
    def __init__(self, source_identity, *, ready: bool) -> None:
        self.source_identity = source_identity
        self._ready = bool(ready)

    def is_ready(self) -> bool:
        return self._ready


class _Owner:
    def __init__(self, leases) -> None:
        self.pending_leases = tuple(leases)
        self.select_calls = 0

    def select(self, _requested):
        self.select_calls += 1
        raise AssertionError("partial bootstrap readiness must not acquire leases")


def test_partial_bootstrap_readiness_is_observational_across_retries():
    """A 3/4 bootstrap probe must not enter/requeue READY ownership."""

    from eligsir.runtime.causal_prepared_ready import CausalPreparedReadyB4Path

    owner = _Owner(
        tuple(_Lease(f"source-{index}", ready=index < 3) for index in range(4))
    )
    before = owner.pending_leases
    path = object.__new__(CausalPreparedReadyB4Path)
    path.bridge = SimpleNamespace(ready_owner=owner)
    path._runtime = SimpleNamespace(ready_bootstrap_remaining=lambda: 4)
    path._prep_worker = None

    for _ in range(1_000):
        outcome = path._bootstrap_tick()
        assert outcome.state == "bootstrap_pending"
        assert outcome.selection.leases == ()
        assert outcome.selection.fresh_count == 0
        assert outcome.selection.replay_count == 0
        assert outcome.selection.starved is True

    assert owner.select_calls == 0
    assert owner.pending_leases == before
