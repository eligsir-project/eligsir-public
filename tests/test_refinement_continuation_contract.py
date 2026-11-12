"""Exact state obligations for speculative post-mapping refinement."""

from types import SimpleNamespace

import pytest

from eligsir.contracts.work_clocks import RuntimeWorkClockOwner
from eligsir.map.adapters.gsplat import _GsplatOptimizationSession


def test_branch_restore_reinstates_optimizer_parameter_version():
    session = _GsplatOptimizationSession(SimpleNamespace())
    session._parameter_version = 7

    state = session.capture_branch_state()
    session._parameter_version = 9
    session.restore_branch_state(state)

    assert session.parameter_version == 7


def test_live_branch_restore_mints_parameter_version_after_rejection():
    session = _GsplatOptimizationSession(SimpleNamespace())
    session._parameter_version = 7

    session._parameter_version = 9
    session.restore_live_branch_state(7)

    assert session.parameter_version == 10


def test_branch_restore_reinstates_semantic_work_clocks():
    clocks = RuntimeWorkClockOwner()
    clocks.record_control_tick()
    clocks.record_optimization(
        optimizer_updates=1,
        supervised_views=4,
        supervised_pixels=400,
        mapping_active_ns=1_000,
    )
    baseline = clocks.snapshot()

    clocks.record_control_tick()
    clocks.record_optimization(
        optimizer_updates=1,
        supervised_views=4,
        supervised_pixels=400,
        mapping_active_ns=1_000,
    )
    clocks.restore_branch_state(baseline)

    assert clocks.snapshot() == baseline


def test_canonical_refinement_branch_restores_only_accepted_map_state():
    from eligsir.runtime.production_application import CanonicalMappingRuntime

    class Backend:
        def __init__(self) -> None:
            self.value = 1

        def checkpoint_state(self):
            return SimpleNamespace(
                layout="gsplat",
                map_version=self.value,
                topology_version=7,
            )

        def restore_checkpoint(self, state, _training_args):
            self.value = state.map_version

    runtime = CanonicalMappingRuntime.__new__(CanonicalMappingRuntime)
    runtime.backend = Backend()
    runtime._backend_optimizer_setup = SimpleNamespace()
    runtime._source_complete = True
    runtime.iteration = 7
    runtime.optimizer_steps = 7
    runtime._supervision_views_completed = 28
    runtime._latest_admitted_optimizer_update = 7
    runtime._control_tick_epoch = 7
    runtime._sh_degree_thresholds_consumed = 2
    runtime._next_lifecycle_optimizer_update = 9
    runtime._work_clock_owner = RuntimeWorkClockOwner()
    runtime._work_clock_owner.record_optimization(
        optimizer_updates=7,
        supervised_views=28,
        supervised_pixels=2_800,
        mapping_active_ns=7_000,
    )
    session = _GsplatOptimizationSession(SimpleNamespace())
    session._parameter_version = 7
    runtime._engine = SimpleNamespace(session=session)

    branch = runtime.refinement_branch_runtime(evaluate_fn=lambda: {0: 20.0})
    baseline = branch.checkpoint_state()
    runtime.backend.value = 99
    runtime.iteration = 99
    runtime.optimizer_steps = 99
    runtime._supervision_views_completed = 396
    runtime._latest_admitted_optimizer_update = 99
    runtime._control_tick_epoch = 99
    runtime._sh_degree_thresholds_consumed = 99
    runtime._next_lifecycle_optimizer_update = 99
    session._parameter_version = 99
    runtime._work_clock_owner.record_control_tick()

    branch.restore_state(baseline)

    assert runtime.backend.value == 1
    assert runtime.iteration == 7
    assert runtime.optimizer_steps == 7
    assert runtime._supervision_views_completed == 28
    assert runtime._latest_admitted_optimizer_update == 7
    assert runtime._control_tick_epoch == 7
    assert runtime._sh_degree_thresholds_consumed == 2
    assert runtime._next_lifecycle_optimizer_update == 9
    assert session.parameter_version == 7
    assert runtime._work_clock_owner.snapshot().optimizer_updates == 7
    assert branch.evaluate_fixed_views(verify_read_only=True) == {"0": 20.0}
    try:
        branch.ingest(object())
    except RuntimeError as error:
        assert "cannot ingest" in str(error)
    else:
        raise AssertionError("post-source refinement branch accepted ingestion")


def test_canonical_refinement_branch_requires_optimizer_setup():
    from eligsir.runtime.production_application import CanonicalMappingRuntime

    runtime = CanonicalMappingRuntime.__new__(CanonicalMappingRuntime)
    runtime._source_complete = True
    runtime._backend_optimizer_setup = None
    runtime.backend = SimpleNamespace()

    with pytest.raises(RuntimeError, match="optimizer setup"):
        runtime.refinement_branch_runtime(evaluate_fn=lambda: {0: 20.0})
