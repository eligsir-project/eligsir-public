from __future__ import annotations

from types import SimpleNamespace

import pytest

# Every test below imports eligsir.runtime.causal_prepared_ready lazily
# inside its own body, which transitively needs torch (via
# prepared_runtime_bridge -> prepared_source_producer). That's invisible to
# tests/_torch_collect_scan.py's static scanner (it only follows module-level
# imports), so without this guard the file hard-fails with a raw
# ModuleNotFoundError in the torch-less CPU smoke lane instead of skipping.
torch = pytest.importorskip("torch")

from eligsir.contracts import SourceIdentity


def test_mutable_candidate_does_not_authorize_fresh_eviction(monkeypatch):
    import eligsir.runtime.causal_prepared_ready as prepared

    source = SourceIdentity("candidate", 0, 1, 1_000)
    candidate_plan = SimpleNamespace(
        supervision_sources=(source,), replay_sources=()
    )
    captured = []
    runtime = SimpleNamespace(
        initialized=True,
        _latest_control_outputs=None,
        _prepared_ready_frozen_work=None,
        _prepared_ready_candidate_work=(object(), candidate_plan),
        prepared_cold_promotion_shortlist=lambda: (),
        _observation_db=SimpleNamespace(),
    )
    path = object.__new__(prepared.CausalPreparedReadyB4Path)
    path._runtime = runtime
    path._promotion_owner = SimpleNamespace(
        schedule_shortlist=lambda _candidates, **kwargs: captured.append(kwargs)
    )
    path._adapter = SimpleNamespace(tick=lambda _inputs: SimpleNamespace())
    path._step_inputs_factory = lambda: object()
    path._maybe_complete_source_phase = lambda _outcome: None

    path.tick()

    assert captured[0]["protected_sources"] == frozenset()


def test_exact_candidate_freezes_before_cold_promotion_scheduling():
    """Promotion requests must capture the exact B4 eviction fence they prepare."""

    import eligsir.runtime.causal_prepared_ready as prepared

    sources = tuple(SourceIdentity("candidate", 0, index, 2_000 + index) for index in range(4))
    candidate_plan = SimpleNamespace(
        supervision_sources=sources[:1], replay_sources=sources[1:]
    )
    candidate_work = (object(), candidate_plan)
    captured = []
    decode_protection = []
    runtime = SimpleNamespace(
        initialized=True,
        _latest_control_outputs=None,
        _prepared_ready_frozen_work=None,
        _prepared_ready_candidate_work=candidate_work,
        prepared_cold_promotion_shortlist=lambda: (),
        _observation_db=SimpleNamespace(),
        _set_admission_prefetch_protected_sources=decode_protection.append,
    )

    def freeze():
        runtime._prepared_ready_frozen_work = candidate_work
        return candidate_plan

    runtime._prepared_ready_plan_for_execution = freeze
    path = object.__new__(prepared.CausalPreparedReadyB4Path)
    path._runtime = runtime
    path._promotion_owner = SimpleNamespace(
        schedule_shortlist=lambda _candidates, **kwargs: captured.append(kwargs)
    )
    path._adapter = SimpleNamespace(tick=lambda _inputs: SimpleNamespace())
    path._step_inputs_factory = lambda: object()
    path._maybe_complete_source_phase = lambda _outcome: None

    path.tick()

    assert runtime._prepared_ready_frozen_work == candidate_work
    assert captured[0]["protected_sources"] == frozenset(sources)
    assert decode_protection == [frozenset(sources)]
