"""Tests for the queue-metrics semantics of
``eligsir.runtime.profiling.benchmark_scenarios.run_scenario`` (W3-01 P0 fix).

Admission-policy rejection is a normal consumption outcome, not a queue drop:
``QueueMetrics.on_drop()`` must only be reserved for actual data loss (queue-full, staleness,
freshness-policy eviction). Before this fix, a rejected frame called ``on_drop("rejected")``
without a matching ``on_dequeue()``, so ``all_rejected``'s logical queue depth grew without
bound even though every frame was consumed immediately.
"""

from __future__ import annotations

import sys

from eligsir.runtime.profiling import events as events_module
from eligsir.runtime.profiling.benchmark_scenarios import (
    BenchmarkWorkload,
    build_scenario,
    run_scenario,
)


def test_all_rejected_scenario_has_zero_cumulative_drops_and_bounded_depth():
    spec = build_scenario("all_rejected", num_frames=25, realtime=False)
    result = run_scenario(spec)

    assert result.frames_seen == 25
    assert result.frames_accepted == 0

    # Every frame was consumed (rejected, not dropped) -- the logical queue must not have
    # grown unboundedly, and no genuine drop should ever have been recorded.
    assert result.queue["cumulative_drops"] == 0
    assert result.queue["last_drop_reason"] is None
    assert result.queue["depth"] == 0


def test_two_fps_accepted_scenario_has_zero_cumulative_drops():
    spec = build_scenario("two_fps_accepted", num_frames=25, realtime=False)
    result = run_scenario(spec)

    assert result.frames_seen == 25
    assert 0 < result.frames_accepted < result.frames_seen

    # Rejections in this arm are also policy decisions, not queue drops.
    assert result.queue["cumulative_drops"] == 0
    assert result.queue["depth"] == 0


def test_rejected_count_is_recoverable_from_frames_seen_minus_accepted():
    spec = build_scenario("all_rejected", num_frames=10, realtime=False)
    result = run_scenario(spec)

    frames_rejected = result.frames_seen - result.frames_accepted
    assert frames_rejected == 10


def test_burst_uses_an_enforced_bounded_queue_with_reason_coded_drops():
    spec = build_scenario("burst", num_frames=64, realtime=False)
    result = run_scenario(spec)

    assert result.queue["capacity"] == spec.queue_capacity
    assert result.queue["peak_depth"] <= spec.queue_capacity
    # The producer runs independently, so a synthetic initial burst has a
    # real opportunity to hit capacity rather than a sequential fake queue.
    assert result.queue["cumulative_drops"] > 0
    assert result.queue["last_drop_reason"] == "queue_full"


def test_frames_seen_is_race_free_under_thread_scheduling_pressure():
    """Regression test for issue #507: ``frames_seen == num_frames`` flaked
    intermittently in CI (23 != 25) with zero recorded queue drops, on a PR
    that could not possibly have touched this code path.

    Root-caused via stress reproduction (not guessed): with
    ``sys.setswitchinterval`` set aggressively low to widen the GIL handoff
    window, the pre-fix producer's ``queue.put_nowait`` + drop-on-``Full``
    reliably lost frames under contention (reproduced 4/1000 runs) even
    though ``all_rejected``/``two_fps_accepted`` have no burst-style
    consumer-start gate -- a full queue there just means the consumer
    thread hasn't been scheduled yet, not real overload, and the drop was
    genuine, permanent frame loss rather than an intended `queue_full`
    signal (that's `burst`'s job, and is deliberately unchanged -- see
    ``test_burst_uses_an_enforced_bounded_queue_with_reason_coded_drops``
    above). Fixed by making the non-burst producer path block-with-retry
    instead of drop-on-first-Full.

    Runs both affected scenarios repeatedly under the same widened race
    window; asserts the invariant holds on every iteration, not just on
    average.
    """
    old_interval = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    try:
        for scenario_name, num_frames in (("all_rejected", 25), ("two_fps_accepted", 25)):
            for seed in range(150):
                spec = build_scenario(scenario_name, num_frames=num_frames, realtime=False, seed=seed)
                result = run_scenario(spec)
                assert result.frames_seen == num_frames, (
                    f"{scenario_name} seed={seed}: frames_seen={result.frames_seen} "
                    f"!= {num_frames} (cumulative_drops={result.queue['cumulative_drops']})"
                )
                assert result.queue["cumulative_drops"] == 0, (
                    f"{scenario_name} seed={seed}: cumulative_drops="
                    f"{result.queue['cumulative_drops']} (expected 0 -- see #507)"
                )
    finally:
        sys.setswitchinterval(old_interval)


def test_real_workload_callbacks_are_exercised_only_for_accepted_frames():
    calls = {"admit": 0, "optimize": 0, "commit": 0}

    def admit(_frame):
        calls["admit"] += 1
        return True

    def optimize():
        calls["optimize"] += 1

    def commit(_frame):
        calls["commit"] += 1

    result = run_scenario(
        build_scenario("two_fps_accepted", num_frames=16, realtime=False),
        workload=BenchmarkWorkload(admit=admit, optimize_step=optimize, commit=commit),
    )

    assert result.production_services_used is True
    assert calls["admit"] == result.frames_seen
    assert calls["optimize"] == calls["commit"] == result.frames_accepted
