"""Regression test for issue #1374: --streaming_post_mapping_refinement_seconds without step count.

This test verifies that refinement runs for ~N seconds when ONLY
--streaming_post_mapping_refinement_seconds is passed (no explicit step count).

Prior to the fix, the loop condition `while completed < requested` with
requested=0 would never enter, causing zero refinement with reason
"requested_steps_zero" despite a valid time cap.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch", reason="checkpoint boundary requires torch")

from eligsir.runtime.causal_entry import run_causal_post_mapping_refinement
from eligsir.runtime.artifact_paths import (
    point_cloud_artifact_paths,
    publish_legacy_iteration_alias,
)


def _write_clock_qualified_point_cloud(root: Path, runtime: object) -> None:
    work = runtime._work_clock_owner.snapshot()
    paths = point_cloud_artifact_paths(
        root,
        optimizer_updates=int(work.optimizer_updates),
        supervised_views=int(work.supervised_rgb_views),
    )
    paths.directory.mkdir(parents=True, exist_ok=True)
    paths.point_cloud.write_bytes(b"ply\n")
    publish_legacy_iteration_alias(paths)


class _Runtime:
    def __init__(self) -> None:
        self.optimizer_steps = 100
        self.iteration = 100
        self._work_clock_owner = SimpleNamespace(
            snapshot=lambda: SimpleNamespace(
                optimizer_updates=self.iteration,
                supervised_rgb_views=self.iteration,
            )
        )
        self.backend = SimpleNamespace(
            checkpoint_state=lambda: SimpleNamespace(
                layout="gsplat",
                payload={"means": torch.zeros(1, 3)},
                map_version=0,
                topology_version=0,
                coordinate_frame_version=0,
                schema_version=1,
            )
        )

    def enter_post_mapping_refinement(self, _endpoint) -> None:
        return None

    def rearm_post_mapping_work(
        self, *, optimizer_updates=None, refinement_progress_fraction=None
    ) -> None:
        return None


class _Application:
    def __init__(self, runtime: _Runtime, root: Path) -> None:
        self.runtime = runtime
        self.root = root

    def flush(self) -> None:
        _write_clock_qualified_point_cloud(self.root, self.runtime)

    def tick(self) -> None:
        self.runtime.optimizer_steps += 1
        self.runtime.iteration += 1


class _NoProgressApplication(_Application):
    """Exercise the explicit step-budget failure without changing clocks."""

    def tick(self) -> SimpleNamespace:
        return SimpleNamespace(optimizer_steps=0, supervision_views=0)


def test_refinement_runs_with_seconds_only_no_steps(tmp_path):
    """Verify refinement executes when requested_steps=0 but requested_seconds>0."""

    runtime = _Runtime()
    application = _Application(runtime, tmp_path)

    # Call with steps=0, seconds=2.0 (short cap for test speed)
    result = run_causal_post_mapping_refinement(
        application,
        runtime,
        source_frame_count=100,
        requested_steps=0,  # No step budget
        model_path=tmp_path,
        mapping_wall_time_s=1.0,
        requested_seconds=2.0,  # Only time cap
    )

    # Verify metadata was written
    metadata_path = tmp_path / "streaming_phase_metadata.json"
    assert metadata_path.exists(), "Phase metadata not written"

    metadata = json.loads(metadata_path.read_text())

    # The critical assertions: refinement should NOT skip with "requested_steps_zero"
    # because a valid seconds cap was provided
    assert metadata["refinement_optimizer_steps_requested"] == 0, \
        "requested_steps should be 0 (that's what was passed)"
    assert metadata["refinement_requested_seconds"] == 2.0, \
        "requested_seconds should be 2.0 (that's what was passed)"
    assert metadata["refinement_stop_reason"] != "requested_steps_zero", \
        (f"BUG: refinement was skipped as 'requested_steps_zero' despite "
         f"requested_seconds={metadata['refinement_requested_seconds']:.1f} > 0")

    # With the fix: stop_reason should be either "time_cap" (if it ran long enough)
    # or "step_cap" (initial fallback, but will be overridden if time cap fires).
    # The important thing is NOT "requested_steps_zero".
    assert metadata["refinement_stop_reason"] in ("time_cap", "step_cap"), \
        f"Unexpected stop reason: {metadata['refinement_stop_reason']}"


def test_refinement_skips_when_both_zero(tmp_path):
    """Verify refinement still skips when BOTH steps and seconds are zero."""

    runtime = _Runtime()
    application = _Application(runtime, tmp_path)

    result = run_causal_post_mapping_refinement(
        application,
        runtime,
        source_frame_count=100,
        requested_steps=0,  # No step budget
        model_path=tmp_path,
        mapping_wall_time_s=1.0,
        requested_seconds=0.0,  # No time cap either
    )

    metadata_path = tmp_path / "streaming_phase_metadata.json"
    metadata = json.loads(metadata_path.read_text())

    # Both are zero: refinement SHOULD skip
    assert metadata["refinement_stop_reason"] == "requested_steps_zero", \
        "Should skip with 'requested_steps_zero' when both steps and seconds are 0"
    assert metadata["refinement_optimizer_steps"] == 0, \
        "No refinement steps should have run"


def test_step_mode_still_validates_delta_gt_zero(tmp_path):
    """Verify step-requested mode still rejects delta<=0 (issue #1374 part 2).

    The validation fix allows delta==0 ONLY for time-cap-only mode (requested==0).
    When steps are explicitly requested, delta must be positive.
    """

    runtime = _Runtime()
    application = _NoProgressApplication(runtime, tmp_path)

    # This should fail: steps requested but delta would be 0
    try:
        result = run_causal_post_mapping_refinement(
            application,
            runtime,
            source_frame_count=100,
            requested_steps=1000,  # Explicit step budget
            model_path=tmp_path,
            mapping_wall_time_s=1.0,
            requested_seconds=0.0,  # No time cap
        )
        # If we get here without exception, the validation failed
        assert False, \
            "Should have raised RuntimeError for delta<=0 when steps are explicitly requested"
    except RuntimeError as e:
        # Expected to fail with the validation error
        assert "did not realize requested optimizer steps" in str(e), \
            f"Unexpected error message: {e}"
        assert "state=None" in str(e)
        assert "reason=None" in str(e)
