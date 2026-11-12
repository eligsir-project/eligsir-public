"""Deterministic (no-GPU) coverage for the wall-clock refinement cap.

Standing session goal: real scenes must land >25dB after last-frame-ingestion
plus a refinement phase capped at a MAXIMUM of 2 minutes WALL-CLOCK TIME. Prior
to this test, every fr3/floor3/kitchen1 comparison run used a fixed
optimizer-step budget (``--streaming_post_mapping_refinement_steps``) for the
refinement tail, never an actual wall-clock cap -- so no run honestly measured
PSNR under the stated goal. These tests verify
``--streaming_post_mapping_refinement_seconds`` actually enforces a REAL
elapsed-time cap (via a fully deterministic fake ``time.perf_counter`` clock,
never real sleeps or GPU work), independent of and in addition to the
existing step-count cap, and that leaving it unset changes nothing.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from eligsir.runtime import causal_entry
from eligsir.runtime.causal_entry import run_causal_post_mapping_refinement


def _install_test_work_clock(runtime: object) -> None:
    runtime._work_clock_owner = SimpleNamespace(
        snapshot=lambda: SimpleNamespace(
            optimizer_updates=int(runtime.iteration),
            supervised_rgb_views=int(runtime.iteration) * 4,
        )
    )


class Runtime:
    optimizer_steps = 4
    iteration = 20

    def __init__(self):
        Runtime.optimizer_steps = 4
        Runtime.iteration = 20
        self.requested_updates = []
        self.backend = SimpleNamespace(
            checkpoint_state=lambda: SimpleNamespace(
                layout="gsplat",
                payload={"means": torch.tensor([[1.0, 2.0, 3.0]])},
                map_version=11,
                topology_version=7,
                coordinate_frame_version=3,
                schema_version=2,
            )
        )
        self.refinement_endpoints = []
        _install_test_work_clock(self)

    def enter_post_mapping_refinement(self, mapping_endpoint):
        self.refinement_endpoints.append(dict(mapping_endpoint))

    def rearm_post_mapping_work(
        self, *, optimizer_updates=None, refinement_progress_fraction=None
    ):
        self.requested_updates.append(optimizer_updates)


def _write_clock_qualified_point_cloud(root: Path, runtime: object) -> None:
    from eligsir.runtime.artifact_paths import (
        point_cloud_artifact_paths,
        publish_legacy_iteration_alias,
    )

    work = runtime._work_clock_owner.snapshot()
    paths = point_cloud_artifact_paths(
        root,
        optimizer_updates=int(work.optimizer_updates),
        supervised_views=int(work.supervised_rgb_views),
    )
    paths.directory.mkdir(parents=True, exist_ok=True)
    paths.point_cloud.write_bytes(b"ply\n")
    publish_legacy_iteration_alias(paths)


class Application:
    """Every ``tick()`` call always realizes exactly one optimizer step."""

    def __init__(self, runtime, root):
        self.runtime = runtime
        self.root = root
        self.calls = 0

    def flush(self):
        _write_clock_qualified_point_cloud(self.root, self.runtime)

    def tick(self):
        self.calls += 1
        Runtime.optimizer_steps += 1
        Runtime.iteration += 1


class _FakeClock:
    """A monotonically-advancing fake clock driven entirely by test code.

    Each call to ``perf_counter()`` advances the clock by ``step_s`` (default
    0.0 -- i.e. "instantaneous" unless the test wants ticks to consume real
    wall time on the fake clock). This lets a test assert the cap fires after
    a specific, exactly-known amount of *simulated* elapsed time without any
    real ``time.sleep`` and without flakiness.
    """

    def __init__(self, step_s: float = 0.0) -> None:
        self.now = 0.0
        self.step_s = step_s
        self.calls = 0

    def __call__(self) -> float:
        self.calls += 1
        value = self.now
        self.now += self.step_s
        return value

    def advance(self, seconds: float) -> None:
        self.now += seconds


def test_time_cap_stops_refinement_before_step_cap_is_reached(tmp_path, monkeypatch):
    """A tiny wall-clock cap must stop refinement well short of a huge step cap."""

    clock = _FakeClock()
    monkeypatch.setattr(causal_entry.time, "perf_counter", clock)

    runtime = Runtime()
    app = Application(runtime, tmp_path)

    # After 3 ticks, jump the fake clock past a 10-second cap so the 4th
    # loop iteration's time-cap check (evaluated BEFORE the next tick) fires.
    real_tick = app.tick

    def tick_and_maybe_advance():
        result = real_tick()
        if app.calls == 3:
            clock.advance(999.0)
        return result

    app.tick = tick_and_maybe_advance

    metadata = run_causal_post_mapping_refinement(
        app,
        runtime,
        source_frame_count=8,
        requested_steps=1_000_000,  # would run "forever" under the step cap alone
        model_path=tmp_path,
        mapping_wall_time_s=1.25,
        requested_seconds=10.0,
    )

    assert app.calls == 3, "refinement must stop as soon as the time cap is crossed"
    assert metadata["refinement_optimizer_steps"] == 3
    assert metadata["refinement_optimizer_steps_requested"] == 1_000_000
    assert metadata["refinement_stop_reason"] == "time_cap"
    assert metadata["refinement_requested_seconds"] == 10.0
    assert metadata["phase"] == "complete"
    on_disk = json.loads((tmp_path / "streaming_phase_metadata.json").read_text())
    assert on_disk["refinement_stop_reason"] == "time_cap"


def test_step_cap_still_wins_when_time_cap_is_never_reached(tmp_path, monkeypatch):
    """A generous time cap must not truncate a run that finishes on its own."""

    clock = _FakeClock()
    monkeypatch.setattr(causal_entry.time, "perf_counter", clock)

    runtime = Runtime()
    app = Application(runtime, tmp_path)

    metadata = run_causal_post_mapping_refinement(
        app,
        runtime,
        source_frame_count=8,
        requested_steps=3,
        model_path=tmp_path,
        mapping_wall_time_s=1.25,
        requested_seconds=999.0,  # never crossed at step_s == 0.0
    )

    assert app.calls == 3
    assert metadata["refinement_optimizer_steps"] == 3
    assert metadata["refinement_stop_reason"] == "step_cap"
    assert metadata["refinement_requested_seconds"] == 999.0


def test_unset_time_cap_is_fully_backward_compatible(tmp_path):
    """Omitting the flag (default 0.0) must reproduce the pre-existing behavior."""

    runtime = Runtime()
    app = Application(runtime, tmp_path)

    metadata = run_causal_post_mapping_refinement(
        app,
        runtime,
        source_frame_count=8,
        requested_steps=3,
        model_path=tmp_path,
        mapping_wall_time_s=1.25,
        # requested_seconds intentionally omitted
    )

    assert app.calls == 3
    assert runtime.requested_updates == [3, 2, 1]
    assert metadata["refinement_optimizer_steps"] == 3
    assert metadata["refinement_stop_reason"] == "step_cap"
    assert metadata["refinement_requested_seconds"] == 0.0


def test_seconds_only_refinement_stops_at_time_cap(tmp_path, monkeypatch):
    """A time-only request is valid and must record its wall-clock bound."""

    clock = _FakeClock(step_s=30.0)
    monkeypatch.setattr(causal_entry.time, "perf_counter", clock)
    runtime = Runtime()
    app = Application(runtime, tmp_path)

    metadata = run_causal_post_mapping_refinement(
        app,
        runtime,
        source_frame_count=8,
        requested_steps=0,
        model_path=tmp_path,
        mapping_wall_time_s=0.0,
        requested_seconds=120.0,
    )

    assert app.calls > 0
    assert metadata["refinement_optimizer_steps"] == app.calls
    assert metadata["refinement_stop_reason"] == "time_cap"
    assert metadata["refinement_requested_seconds"] == 120.0


def test_negative_requested_seconds_is_refused(tmp_path):
    runtime = Runtime()
    app = Application(runtime, tmp_path)

    with pytest.raises(ValueError):
        run_causal_post_mapping_refinement(
            app,
            runtime,
            source_frame_count=8,
            requested_steps=3,
            model_path=tmp_path,
            mapping_wall_time_s=0.0,
            requested_seconds=-1.0,
        )
