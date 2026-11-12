"""Tests for wiring output-staging finalization into the streaming
(`train_streaming.py`) training path.

Context: Task 4 of the output-staging-manager plan
(docs/superpowers/plans/2026-08-12-output-staging-manager.md) added
`train._finalize_output_staging()` and wired it into `train.py`'s
non-streaming `training()` on both normal completion and SIGTERM. But
`train.py`'s `training()` dispatches any `--streaming_replay` run to
`train_streaming.streaming_training()` via an early `return` *before*
reaching that call site -- so every streaming run silently lost its final
sync while still exiting 0. This file tests the fix: `train_streaming.py`
now calls the same `train._finalize_output_staging()` on its own normal
completion, on its two early-return training-mode branches
(submap_stitch/rolling_seed), and (via the new
`_streaming_emergency_finalize_output_staging` module-level helper) on its
own SIGTERM/SIGINT handlers.

`streaming_training()` itself is a multi-thousand-line function that
requires a real dataset/GPU to run end-to-end, so it is not invoked here.
Instead:
  - `_streaming_emergency_finalize_output_staging` is unit-tested directly
    (module-level, not a nested closure, specifically so it can be tested
    this way -- mirrors the mocking style already used in
    tests/test_output_staging_finalization.py for train.py's own
    `_finalize_output_staging`).
  - The presence and placement of the three additional wiring call sites
    inside `streaming_training` (normal completion, submap_stitch early
    return, rolling_seed early return) is verified via source inspection,
    honestly labeled as such -- a regression guard against the exact class
    of gap this file exists to close, not a behavioral execution test.

CPU-only, no GPU compute exercised -- but importing train.py/train_streaming.py
pulls in torch and gsplat (this project's only rasterizer backend), so these
tests must run inside the container per CLAUDE.md's Run Contracts:

    docker compose run --rm train python -m pytest tests/test_output_staging_streaming.py -q
"""
from __future__ import annotations

import inspect
import os
from argparse import Namespace

import pytest

import train
import train_streaming


class _FakeManager:
    def __init__(self, active=True):
        self.active = active


class _FakeWorker:
    def __init__(self, finalize_result):
        self._finalize_result = finalize_result
        self.finalize_calls = []
        self.shutdown_called = False

    def finalize(self, timeout=600.0):
        self.finalize_calls.append(timeout)
        return self._finalize_result

    def shutdown(self):
        self.shutdown_called = True


def _staged_args(tmp_path, finalize_result, active=True):
    local_dir = tmp_path / "local"
    persist_dir = tmp_path / "persist"
    local_dir.mkdir()
    persist_dir.mkdir()
    (local_dir / "point_cloud.ply").write_text("dummy")

    worker = _FakeWorker(finalize_result)
    args = Namespace(
        model_path=str(local_dir),
        persist_model_path=str(persist_dir),
        output_keep_local=False,
        _output_staging_manager=_FakeManager(active=active),
        _persistence_worker=worker,
    )
    return args, worker, local_dir, persist_dir


class TestStreamingEmergencyFinalizeOutputStaging:
    """Direct behavioral tests of the new module-level helper that
    train_streaming.py's SIGTERM/SIGINT handlers call."""

    def test_uses_short_60s_timeout(self, tmp_path):
        args, worker, local_dir, persist_dir = _staged_args(tmp_path, finalize_result=True)
        train_streaming._streaming_emergency_finalize_output_staging(args)
        assert worker.finalize_calls == [60.0]

    def test_writes_complete_marker_on_success(self, tmp_path):
        args, worker, local_dir, persist_dir = _staged_args(tmp_path, finalize_result=True)
        train_streaming._streaming_emergency_finalize_output_staging(args)
        assert os.path.isfile(os.path.join(persist_dir, ".COMPLETE"))

    def test_writes_failed_marker_but_does_not_raise(self, tmp_path):
        args, worker, local_dir, persist_dir = _staged_args(tmp_path, finalize_result=False)
        # Must NOT raise -- callers (SIGTERM/SIGINT handlers) need to reach
        # os._exit / re-raise KeyboardInterrupt unconditionally afterward.
        train_streaming._streaming_emergency_finalize_output_staging(args)
        assert os.path.isfile(os.path.join(persist_dir, ".FAILED"))
        assert not os.path.exists(os.path.join(persist_dir, ".COMPLETE"))

    def test_noop_when_staging_inactive(self, tmp_path):
        args, worker, local_dir, persist_dir = _staged_args(
            tmp_path, finalize_result=True, active=False
        )
        train_streaming._streaming_emergency_finalize_output_staging(args)
        assert not worker.finalize_calls
        assert not os.path.exists(os.path.join(persist_dir, ".COMPLETE"))

    def test_noop_when_manager_missing_entirely(self, tmp_path):
        args = Namespace(model_path=str(tmp_path), persist_model_path=str(tmp_path))
        # Ordinary (non-staged) streaming runs must be unaffected.
        train_streaming._streaming_emergency_finalize_output_staging(args)
        assert not os.path.exists(os.path.join(str(tmp_path), ".COMPLETE"))
        assert not os.path.exists(os.path.join(str(tmp_path), ".FAILED"))


class TestStreamingTrainingSourceWiring:
    """Source-inspection regression guards: `streaming_training`'s body
    must actually call the finalize helpers at its normal-completion path
    and both early-return training-mode branches, and its SIGTERM/SIGINT
    handlers must route through the emergency helper. This does not
    execute the function (not CPU-feasible -- it requires a real
    dataset/GPU) -- it guards against the exact silent-regression shape of
    the original bug: the call sites existing in train.py but not being
    reachable from the streaming path at all.
    """

    @pytest.fixture(scope="class")
    def source(self):
        return inspect.getsource(train_streaming.streaming_training)

    def test_normal_completion_finalizes(self, source):
        # The last real statement of the function must gate on finalize,
        # after "Mapping runtime complete." is printed.
        tail = source.rsplit("Mapping runtime complete.", 1)[-1]
        assert "_finalize_output_staging(args)" in tail

    @staticmethod
    def _alt_trainer_branch(source):
        """The shared submap_stitch/rolling_seed early-return branch body.

        Issue #1102 merged the two hand-copied branches into one dispatch
        block, so both modes are covered by the same slice. It is bounded by
        the next same-indentation statement rather than by the first literal
        ``return`` -- the branch's own explanatory comment contains that word.
        """
        _, sep, after = source.partition(
            'if _training_mode in ("submap_stitch", "rolling_seed"):'
        )
        assert sep, "alternate-trainer dispatch branch not found"
        return after.split("\n    if _training_mode != ", 1)[0]

    def test_submap_stitch_early_return_finalizes(self, source):
        # The branch must still finalize staging before returning. Issue #1075
        # P1 routed the shutdown through `finalize_run()`; issue #1102 moved
        # staging back out of it (`finalize_staging=False`) so the completion
        # evidence can be written first -- the guard is against the branch not
        # finalizing at all, so it asserts on the explicit call.
        branch = self._alt_trainer_branch(source)
        assert "run_submap_stitch_mode(" in branch
        assert "finalize_run(" in branch
        assert "_finalize_output_staging(args)" in branch

    def test_rolling_seed_early_return_finalizes(self, source):
        branch = self._alt_trainer_branch(source)
        assert "run_rolling_seed_mode(" in branch
        assert "finalize_run(" in branch
        assert "_finalize_output_staging(args)" in branch

    def test_sigterm_handler_calls_emergency_finalize(self, source):
        before, _, after = source.partition("def _emergency_checkpoint_and_exit")
        handler_body = after.split("\n    def ", 1)[0]
        assert "_emergency_finalize_output_staging()" in handler_body

    def test_sigint_handler_calls_emergency_finalize(self, source):
        before, _, after = source.partition("def _graceful_shutdown")
        handler_body = after.split("\n    _signal.signal(_signal.SIGINT, _graceful_shutdown)", 1)[0]
        assert "_emergency_finalize_output_staging()" in handler_body

    def test_emergency_closure_delegates_to_module_level_helper(self, source):
        before, _, after = source.partition("def _emergency_finalize_output_staging():")
        closure_body = after.split("\n\n", 1)[0]
        assert "_streaming_emergency_finalize_output_staging(args)" in closure_body


class TestStreamingZeroBehaviorChangeWhenStagingOff:
    """The finalize helpers must be true no-ops (no filesystem writes, no
    exceptions) whenever output staging was never activated -- this is
    the default for every existing streaming run, so this is the
    behavior-neutrality guarantee task 5 requires."""
