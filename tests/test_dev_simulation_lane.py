from __future__ import annotations

import importlib.util

import pytest

from eligsir.simulation.dev_harness import SimulationConfig, run_dev_simulation


@pytest.mark.sim
def test_numpy_simulation_is_deterministic_and_exercises_rejects_skips_and_tail():
    config = SimulationConfig(backend="numpy")

    first = run_dev_simulation(config)
    second = run_dev_simulation(config)

    assert first.signature == second.signature
    assert first.source_frames == 24
    assert first.policy_skips == 4
    assert first.submitted == 20
    assert first.rejected == 3
    assert first.mapped == 17
    assert first.zero_frame_ticks >= config.zero_frame_tail_ticks
    assert set(first.mapped_devices) == {"cpu"}
    assert set(first.rgb_dtypes) == {"float32"}
    assert set(first.depth_dtypes) == {"float32"}
    assert first.event_actions.count("reject") == 3
    assert first.event_actions.count("deliver") == 17


@pytest.mark.sim
def test_drain_cap_changes_tick_shape_not_accepted_source_identity():
    one = run_dev_simulation(SimulationConfig(drain_cap=1))
    four = run_dev_simulation(SimulationConfig(drain_cap=4))

    assert one.mapped_ordinals == four.mapped_ordinals
    assert one.mapped == four.mapped
    assert one.rejected == four.rejected
    assert one.policy_skips == four.policy_skips
    assert one.tick_count > four.tick_count


@pytest.mark.sim
def test_simulation_can_run_without_policy_skips_or_tracking_rejections():
    result = run_dev_simulation(
        SimulationConfig(frame_count=9, skip_every=0, untracked_every=0)
    )

    assert result.policy_skips == 0
    assert result.submitted == 9
    assert result.rejected == 0
    assert result.mapped == 9
    assert result.mapped_ordinals == tuple(range(9))


@pytest.mark.sim
@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="CPU-Torch lane only")
def test_torch_cpu_backend_runs_same_semantic_trace_as_numpy():
    numpy_result = run_dev_simulation(SimulationConfig(backend="numpy"))
    torch_result = run_dev_simulation(SimulationConfig(backend="torch"))

    assert torch_result.mapped_ordinals == numpy_result.mapped_ordinals
    assert torch_result.submitted == numpy_result.submitted
    assert torch_result.mapped == numpy_result.mapped
    assert torch_result.rejected == numpy_result.rejected
    assert torch_result.policy_skips == numpy_result.policy_skips
    assert set(torch_result.mapped_devices) == {"cpu"}
    assert set(torch_result.rgb_dtypes) == {"torch.float32"}
    assert set(torch_result.depth_dtypes) == {"torch.float32"}
