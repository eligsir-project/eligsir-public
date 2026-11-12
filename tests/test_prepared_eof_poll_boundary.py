"""EOF polling contracts for the causal prepared path."""

from types import SimpleNamespace

import pytest

# The single test below imports eligsir.runtime.causal_prepared_ready
# lazily inside its own body, which transitively needs torch (via
# prepared_runtime_bridge -> prepared_source_producer). That's invisible to
# tests/_torch_collect_scan.py's static scanner (it only follows module-level
# imports), so without this guard the file hard-fails with a raw
# ModuleNotFoundError in the torch-less CPU smoke lane instead of skipping.
torch = pytest.importorskip("torch")


def test_source_complete_default_performs_one_bootstrap_attempt(monkeypatch):
    import eligsir.runtime.causal_prepared_ready as prepared

    path = object.__new__(prepared.CausalPreparedReadyB4Path)
    path._runtime = SimpleNamespace(initialized=False)
    path._source_complete_requested = False
    path._source_complete_phase_marked = False
    path._source_complete_phase_hook = None
    calls = []
    pending = prepared.PreparedBootstrapOutcome(
        "bootstrap_pending",
        prepared.PreparedRuntimeBridgeResult(0, 0, 0, 0),
        prepared.ReadyOnlyTickSelection((), 0, 0, False, True),
    )
    monkeypatch.setattr(path, "_bootstrap_tick", lambda: calls.append(True) or pending)

    outcome = path.mark_source_complete()

    assert calls == [True]
    assert outcome.state == "bootstrap_pending"
    assert outcome.reason == "source_complete_preparation_pending"
    assert path._source_complete_requested is True
