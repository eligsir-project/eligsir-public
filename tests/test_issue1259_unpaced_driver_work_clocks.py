"""Regression tests for issue #1259.

`CausalReplayDriver` (the unpaced driver, selected by
``--streaming_ingestion_mode iter_based``) used to return a
`CausalReplaySummary` with `optimizer_steps` / `supervision_views` /
`mapping_active_s` / `elapsed_s` left at their dataclass defaults of zero,
while the paced driver populated them.

`run_causal_mapping` hands that summary to `run_causal_post_mapping_refinement`
as the refinement `initial_clock`, so the evaluation observer's work clock ran
exactly `mapping_optimizer_steps` behind the runtime's own cumulative counters
for the whole refinement phase. At `evaluation_observer.finalize(summary)` the
fail-closed invariant in
`causal_evaluation._terminal_evaluation_tick_clock_snapshot` then raised
``RuntimeError("terminal optimizer update clock drifted from latest tick")``
*after* refinement had completed every requested optimizer step but *before*
the refined point cloud and `comparison/iter_N/report.json` were written --
real, silent loss of the run's final artifacts.

Two real GPU runs on two different hosts died exactly this way; their
`causal_replay_summary.json` recorded `optimizer_steps: 0` against a mapping
phase that had genuinely executed 529 optimizer steps.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest

from eligsir.contracts.runtime import BufferRef, SensorPacket, SourceIdentity
from eligsir.contracts.types import CameraIntrinsics
from eligsir.runtime.causal_evaluation import (
    _terminal_evaluation_tick_clock_snapshot,
)
from eligsir.runtime.causal_replay import (
    CausalReplayClock,
    CausalReplayDriver,
    DatasetPacketAdapter,
)
from eligsir.tracking.providers import observation_from_pose_matrix

SUPERVISION_VIEWS_PER_STEP = 4


@dataclass(frozen=True)
class _Record:
    ordinal: int


class _Adapter(DatasetPacketAdapter[_Record]):
    def packet_from_record(self, record: _Record) -> SensorPacket:
        source = SourceIdentity("dataset", 0, record.ordinal, 1_000 + record.ordinal)
        return SensorPacket(
            source=source,
            rgb=BufferRef.from_numpy(
                np.zeros((2, 2, 3), dtype=np.uint8), encoding="rgb8"
            ),
            depth=None,
            intrinsics=CameraIntrinsics(
                fx=1.0, fy=1.0, cx=1.0, cy=1.0, width=2, height=2
            ),
            tracking=observation_from_pose_matrix(source, np.eye(4), provider="dataset"),
        )


class _Application:
    """Minimal mapper that reports real per-tick work, like the production one.

    Crucially, most ticks complete NO optimizer update. That is not an
    artificial worst case: the two lost `iter_based` runs recorded
    `ticks: 2526` against a true `mapping_optimizer_steps` of `529`, and the
    paced sibling recorded `ticks: 1348` / `optimizer_steps: 577` with
    `zero_frame_ticks: 405`. Under `B=4` supervision batching a tick routinely
    submits work without closing an optimizer update, so a summary that
    reported `ticks` (or any per-tick constant) instead of the sum of each
    tick's own `optimizer_steps` would still drift against the runtime's
    cumulative counter -- just by a smaller, equally fatal amount.
    """

    #: Only every third tick closes an optimizer update, and when it does it
    #: closes two -- so `ticks`, `optimizer_steps` and `supervision_views` are
    #: three mutually distinct numbers and none can stand in for another.
    WORKING_TICK_STRIDE = 3
    STEPS_PER_WORKING_TICK = 2

    def __init__(self) -> None:
        self.admitted: list[SourceIdentity] = []
        self.ticks = 0
        self.cumulative_optimizer_steps = 0
        self.cumulative_supervision_views = 0
        self.session = SimpleNamespace(submit=self._submit)

    def _submit(self, packet: SensorPacket):
        self.admitted.append(packet.source)
        return SimpleNamespace(
            event=SimpleNamespace(source=packet.source),
            disposition=SimpleNamespace(admit_supervision=True),
        )

    def tick(self):
        self.ticks += 1
        if self.admitted:
            supervision_sources = (self.admitted.pop(0),)
            replay_sources = ()
        else:
            supervision_sources = ()
            replay_sources = (_Adapter().packet_from_record(_Record(0)).source,)
        working = self.ticks % self.WORKING_TICK_STRIDE == 0
        steps = self.STEPS_PER_WORKING_TICK if working else 0
        views = steps * SUPERVISION_VIEWS_PER_STEP
        self.cumulative_optimizer_steps += steps
        self.cumulative_supervision_views += views
        return SimpleNamespace(
            executed=True,
            optimizer_steps=steps,
            supervision_views=views,
            iteration=self.ticks,
            work_plan=SimpleNamespace(
                supervision_sources=supervision_sources,
                replay_sources=replay_sources,
            ),
        )


def test_unpaced_driver_summary_reports_the_work_it_actually_executed():
    """The summary's work clocks must equal the runtime's cumulative counters.

    This is the exact equality `_terminal_evaluation_tick_clock_snapshot`
    enforces at `finalize()`; anything else drifts and fails the run closed.
    """

    application = _Application()

    summary = CausalReplayDriver(application, _Adapter()).run(
        (_Record(n) for n in range(12))
    )

    assert summary.ticks == application.ticks
    # The regression: both of these were zero, so the mapping phase reported as
    # if it had done no work at all and refinement started from a zeroed clock.
    assert summary.optimizer_steps == application.cumulative_optimizer_steps
    assert summary.supervision_views == application.cumulative_supervision_views
    assert summary.optimizer_steps > 0
    assert summary.supervision_views > 0

    # Guard the near-miss: a summary that reported the tick count, or any
    # per-tick constant, would satisfy "nonzero" while still drifting against
    # the runtime. Real runs recorded ticks=2526 vs optimizer_steps=529.
    assert summary.optimizer_steps != summary.ticks
    assert summary.supervision_views != summary.optimizer_steps
    assert application.cumulative_optimizer_steps < application.ticks

    assert summary.mapping_active_s > 0.0
    assert summary.elapsed_s > 0.0
    assert summary.elapsed_s >= summary.mapping_active_s
    assert summary.optimizer_steps_per_active_s > 0.0
    assert summary.supervision_views_per_active_s > 0.0
    assert summary.supervision_views_per_elapsed_s > 0.0


def _terminal_tick(*, cumulative_optimizer_updates: int, cumulative_views: int):
    constellation = SimpleNamespace(
        requested_batch_size=4,
        realized_batch_size=4,
        requested_camera_count=4,
        realized_camera_count=4,
        unique_eligible_count=4,
        fill_ratio=1.0,
        microbatch_count=1,
        cumulative_realized_supervised_views=cumulative_views,
        cumulative_optimizer_updates=cumulative_optimizer_updates,
        mode="fixed",
        reason="fixed",
        hysteresis_state="steady",
    )
    return SimpleNamespace(
        cumulative_optimizer_updates=cumulative_optimizer_updates,
        cumulative_supervision_views=cumulative_views,
        microbatch_sizes=(4,),
        optimizer_steps=1,
        supervision_views=SUPERVISION_VIEWS_PER_STEP,
        unique_supervision_views=SUPERVISION_VIEWS_PER_STEP,
        supervised_pixels=16,
        snapshot_id=7,
        control_snapshot=SimpleNamespace(control_tick_epoch=11),
        work_plan=SimpleNamespace(supervision_constellation=constellation),
    )


class _ObserverStub:
    """The subset of `CausalEvaluationObserver` the terminal snapshot reads."""

    def __init__(self, *, initial_clock: CausalReplayClock, refinement_steps: int):
        self.plan = SimpleNamespace(holdout_ordinals=(3, 7))
        self.runtime = SimpleNamespace(
            frames_seen=12,
            scene=SimpleNamespace(train_cameras=()),
            _admitted_observations_total=12,
        )
        self.last_ingest_iteration = 5
        # `causal_entry.run_causal_post_mapping_refinement` advances the
        # observer with exactly this clock on every refinement tick.
        self._optimizer_updates = int(initial_clock.optimizer_steps) + refinement_steps
        self._supervision_views = int(initial_clock.supervision_views) + (
            refinement_steps * SUPERVISION_VIEWS_PER_STEP
        )
        self._elapsed_s = 1.0
        self._mapping_active_s = 0.5
        self._latest_tick = None


# The mapping phase ran 529 optimizer steps and refinement ran 10000, so the
# terminal step is 10529 -- deliberately NOT on the 200-step evaluation cadence
# grid, which is the exact situation the two lost GPU runs were in.
MAPPING_STEPS = 529
REFINEMENT_STEPS = 10_000
TERMINAL_STEPS = MAPPING_STEPS + REFINEMENT_STEPS


def test_terminal_clock_snapshot_holds_when_the_summary_carries_mapping_work():
    """A correct mapping summary keeps observer and runtime clocks in step."""

    observer = _ObserverStub(
        initial_clock=CausalReplayClock(
            optimizer_steps=MAPPING_STEPS,
            supervision_views=MAPPING_STEPS * SUPERVISION_VIEWS_PER_STEP,
            mapping_active_s=0.5,
            elapsed_s=1.0,
        ),
        refinement_steps=REFINEMENT_STEPS,
    )
    observer._latest_tick = _terminal_tick(
        cumulative_optimizer_updates=TERMINAL_STEPS,
        cumulative_views=TERMINAL_STEPS * SUPERVISION_VIEWS_PER_STEP,
    )

    snapshot = _terminal_evaluation_tick_clock_snapshot(observer)

    assert snapshot["optimizer_step"] == TERMINAL_STEPS
    assert snapshot["supervision_views"] == TERMINAL_STEPS * SUPERVISION_VIEWS_PER_STEP


def test_terminal_clock_snapshot_refuses_a_zeroed_mapping_summary():
    """Guard the exact failure the zeroed unpaced summary used to produce."""

    observer = _ObserverStub(
        initial_clock=CausalReplayClock(
            optimizer_steps=0,
            supervision_views=0,
            mapping_active_s=0.0,
            elapsed_s=0.0,
        ),
        refinement_steps=REFINEMENT_STEPS,
    )
    observer._latest_tick = _terminal_tick(
        cumulative_optimizer_updates=TERMINAL_STEPS,
        cumulative_views=TERMINAL_STEPS * SUPERVISION_VIEWS_PER_STEP,
    )

    with pytest.raises(RuntimeError, match="optimizer update clock drifted"):
        _terminal_evaluation_tick_clock_snapshot(observer)
