"""W3-41: ``live_strategy_apply/*`` telemetry must accumulate real per-call
``AlignedBufferRegistry.apply()`` counts across the whole logging window, not
report only the single, instantaneous count from the one call that happens
to land on a ``should_log(iteration)`` hit.

Root cause (see docs/architecture/2026-08-09-w3-41-apply-mutation-audit.md):
``apply()`` computes ``retired_count``/``reseeded_count``/
``pruned_for_reseed_count`` fresh on every call (every training iteration
when a phase is active), but ``LiveWrappedMCMCStrategy.step_post_backward``
previously only wrote that value to TensorBoard when ``should_log(iteration)``
was true (every ``strategy_log_interval`` iterations, default 500) -- an
instantaneous per-step snapshot at a sparse sampling interval, not an
accumulator. A real, fresh GPU run at ``--strategy_log_interval 1`` showed
1742/4000 real per-step calls had a nonzero ``retired_count`` (summing to
571,636 total retirements over the run), yet every one of the 8 iterations
that are exact multiples of 500 read exactly zero -- confirming the bug is
sampling-cadence, not "nothing happened."

This test exercises the fix directly against ``LiveWrappedMCMCStrategy``
(not a real GPU run) by stubbing out ``_base``/``_live``/the registry class,
so it runs fast and CPU-only, and asserts:
1. Telemetry logged at a ``should_log`` hit equals the SUM of every
   ``apply()`` call's count since the previous hit (not just the last call).
2. The accumulator resets after each logged window (no double-counting).
3. ``apply()`` itself is invoked exactly once per iteration regardless of
   logging cadence -- this fix must not change how often (or how) the
   registry is actually called, only what gets logged.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from eligsir.utils.strategies.mcmc.live_wrapped import LiveWrappedMCMCStrategy


class _StubGaussians:
    def __init__(self, n: int = 8, seed: int = 0):
        g = torch.Generator().manual_seed(seed)
        self._xyz = torch.rand(n, 3, generator=g)

    @property
    def get_xyz(self):
        return self._xyz


@dataclass
class _FakeApplyReport:
    retired_count: int = 0
    reseeded_count: int = 0
    pruned_for_reseed_count: int = 0


class _FakeRegistry:
    """Stands in for ``AlignedBufferRegistry``: records every ``apply()``
    call and returns a caller-supplied, per-iteration count sequence -- this
    is what lets the test reproduce "nonzero counts on iterations that never
    coincide with a should_log hit"."""

    call_log: list[int] = []

    def __init__(self, model, live_state, per_iter_counts):
        self._per_iter_counts = per_iter_counts

    def apply(self, plan, *, phases, iteration):
        _FakeRegistry.call_log.append(iteration)
        n = self._per_iter_counts.get(iteration, 0)
        return _FakeApplyReport(retired_count=n, reseeded_count=0, pruned_for_reseed_count=0)


class _FakeObserveReport:
    proposed_retire_count = 0
    proposed_prune_count = 0
    proposed_reseed_count = 0
    proposed_growth_count = 0
    overlap_with_current_mcmc = {"overlap_count": 0}


class _FakeTBWriter:
    def __init__(self):
        self.calls: list[tuple[str, float, int]] = []

    def add_scalar(self, tag, value, step):
        self.calls.append((tag, value, step))


def _make_wrapped_strategy(per_iter_counts):
    from eligsir.optimization.gaussian.strategies.live.backend import ActivationPhaseConfig

    strat = LiveWrappedMCMCStrategy(phases=ActivationPhaseConfig(phase_b_retirement=True))
    # Stub out the real base strategy and the live shadow strategy so this
    # test is a fast, CPU-only unit test of the accumulation logic itself,
    # not a re-verification of GsplatEnergyMCMCStrategy or
    # EliGSiRLiveGaussianStrategy (already covered elsewhere).
    strat._base.step_post_backward = lambda *a, **k: None
    strat._live.observe = lambda **k: None
    strat._live.propose = lambda **k: (object(), _FakeObserveReport())
    _FakeRegistry.call_log = []
    strat._registry_cls = lambda model, live_state, per_iter_counts=per_iter_counts: _FakeRegistry(
        model, live_state, per_iter_counts
    )
    return strat


def test_apply_telemetry_accumulates_across_a_logging_window_not_just_the_last_call():
    # Nonzero retirements fire on iterations 3, 7, 8 -- none of which are a
    # multiple of the logging interval (5) -- mirroring the real run's
    # finding that every exact-500-multiple sample read zero despite real
    # events happening between samples.
    per_iter_counts = {3: 10, 7: 4, 8: 6}
    strat = _make_wrapped_strategy(per_iter_counts)
    gaussians = _StubGaussians()
    args = object()
    sched = object()
    tb = _FakeTBWriter()
    should_log = lambda i: i % 5 == 0  # noqa: E731

    for it in range(1, 11):
        strat.step_post_backward(
            gaussians, args, sched, it, tb_writer=tb, should_log_strategy=should_log
        )

    retired_logs = [(step, value) for (tag, value, step) in tb.calls if tag == "live_strategy_apply/retired_count"]
    # should_log hits at 5 and 10.
    assert retired_logs == [(5, 10), (10, 10)], retired_logs
    # window [1..5]: only iter 3 contributed (10) -> logged 10 at step 5.
    # window [6..10]: iters 7 (4) and 8 (6) contributed -> logged 10 at step 10.
    # Old (buggy) behavior would have logged the INSTANTANEOUS count at
    # iterations 5 and 10 themselves, both 0 (no event on those exact
    # iterations) -- this is exactly the zero-telemetry-despite-real-events
    # bug this fix addresses.


def test_apply_is_still_called_exactly_once_per_iteration_regardless_of_logging_cadence():
    """The fix must not change how often/how apply() is invoked -- only what
    gets logged. Guards against a fix that (incorrectly) skips or batches
    apply() calls to make accumulation easier."""
    strat = _make_wrapped_strategy(per_iter_counts={})
    gaussians = _StubGaussians()
    tb = _FakeTBWriter()
    should_log = lambda i: i % 5 == 0  # noqa: E731

    for it in range(1, 11):
        strat.step_post_backward(gaussians, object(), object(), it, tb_writer=tb, should_log_strategy=should_log)

    assert _FakeRegistry.call_log == list(range(1, 11))


def test_accumulator_resets_after_each_logged_window_no_double_counting():
    per_iter_counts = {2: 5}
    strat = _make_wrapped_strategy(per_iter_counts)
    gaussians = _StubGaussians()
    tb = _FakeTBWriter()
    should_log = lambda i: i % 3 == 0  # noqa: E731

    for it in range(1, 7):
        strat.step_post_backward(gaussians, object(), object(), it, tb_writer=tb, should_log_strategy=should_log)

    retired_logs = [(step, value) for (tag, value, step) in tb.calls if tag == "live_strategy_apply/retired_count"]
    # should_log hits at 3 and 6. iter 2's count of 5 must be attributed to
    # the window ending at 3, and NOT re-reported (double-counted) in the
    # window ending at 6.
    assert retired_logs == [(3, 5), (6, 0)], retired_logs
