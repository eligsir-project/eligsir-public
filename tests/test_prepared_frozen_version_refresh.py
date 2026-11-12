from dataclasses import dataclass
from types import SimpleNamespace

import pytest

# Every test below imports eligsir.runtime.causal_prepared_ready lazily
# inside its own body, which transitively needs torch (via
# prepared_runtime_bridge -> prepared_source_producer). That's invisible to
# tests/_torch_collect_scan.py's static scanner (it only follows module-level
# imports), so without this guard the file hard-fails with a raw
# ModuleNotFoundError in the torch-less CPU smoke lane instead of skipping.
torch = pytest.importorskip("torch")


@dataclass(frozen=True)
class _Plan:
    map_content_version: int
    topology_version: int
    parameter_version: int


def test_frozen_plan_preserves_authorized_version_provenance():
    from eligsir.runtime.causal_prepared_ready import _step_inputs_for_runtime

    snapshot = object()
    plan = _Plan(1, 2, 3)
    frozen = (snapshot, plan)
    runtime = SimpleNamespace(
        initialized=True,
        _prepared_ready_plan_for_execution=lambda: plan,
        _prepared_ready_frozen_work=frozen,
        _prepared_ready_candidate_work=frozen,
        _latest_tick_work=None,
        _latest_report_work=None,
        _last_materialized_work=None,
        _versions=lambda: (_ for _ in ()).throw(
            AssertionError("input projection must not restamp scheduler provenance")
        ),
    )

    inputs = _step_inputs_for_runtime(runtime)

    assert inputs.snapshot is snapshot
    assert inputs.work_plan is plan
    assert runtime._prepared_ready_frozen_work is frozen
    assert runtime._prepared_ready_candidate_work is frozen
    assert (
        plan.map_content_version,
        plan.topology_version,
        plan.parameter_version,
    ) == (1, 2, 3)


def test_frozen_test_seam_without_runtime_tuple_preserves_plan():
    from eligsir.runtime.causal_prepared_ready import _step_inputs_for_runtime

    plan = _Plan(1, 2, 3)
    runtime = SimpleNamespace(
        initialized=True,
        _prepared_ready_plan_for_execution=lambda: plan,
        _prepared_ready_frozen_work=None,
        _latest_tick_work=None,
        _latest_report_work=None,
        _last_materialized_work=None,
    )

    inputs = _step_inputs_for_runtime(runtime)

    assert inputs.snapshot is None
    assert inputs.work_plan is plan
