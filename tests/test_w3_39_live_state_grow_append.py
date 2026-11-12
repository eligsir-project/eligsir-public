"""Wave 3, W3-39: ``LivePopulationState.grow_append`` and ``align(...,
pure_append=True)`` -- the append-preserving growth path that lets
retirement/recovery evidence survive an ordinary Phase-2 depth-insertion
growth event instead of cold-starting on every count change (W3-41 measured
this churn concretely: ~26 nonzero iterations of evidence followed by ~24
zero/dormant iterations, repeating roughly every 50 iterations).

The load-bearing property under test: after a pure-append growth call,
every EXISTING row's lifecycle/EMA/streak state must be bit-identical to
what it was immediately before the call -- not just "doesn't crash".
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from eligsir.optimization.gaussian.strategies.contracts import GaussianLifecycleState  # noqa: E402
from eligsir.optimization.gaussian.strategies.live.state import (  # noqa: E402
    LivePopulationState,
    StaleLiveStateError,
    state_to_code,
)


def _seeded_state(n: int = 4, topology_version: int = 0) -> LivePopulationState:
    """A state with buffers already established and populated with
    non-default, distinguishable values in every field, so a bug that
    accidentally zeros/resets any one of them is caught."""
    s = LivePopulationState()
    s.align(n, topology_version=topology_version, iteration=0, device=torch.device("cpu"))
    s.geometry_lifecycle[:] = torch.tensor(
        [state_to_code(GaussianLifecycleState.STABLE)] * n, dtype=torch.int8
    )
    s.appearance_lifecycle[:] = torch.tensor(
        [state_to_code(GaussianLifecycleState.SUSPECT)] * n, dtype=torch.int8
    )
    s.support_count = torch.arange(1, n + 1, dtype=torch.float32) * 7.0
    s.contribution_ema = torch.arange(1, n + 1, dtype=torch.float32) * 0.11
    s.rgb_residual_ema = torch.arange(1, n + 1, dtype=torch.float32) * 0.22
    s.depth_residual_ema = torch.arange(1, n + 1, dtype=torch.float32) * 0.33
    s.pos_grad_ema = torch.arange(1, n + 1, dtype=torch.float32) * 0.44
    s.opacity_grad_ema = torch.arange(1, n + 1, dtype=torch.float32) * 0.55
    s.free_space_conflict_ema = torch.arange(1, n + 1, dtype=torch.float32) * 0.66
    s.geometry_confidence = torch.arange(1, n + 1, dtype=torch.float32) * 0.77
    s.last_visible_step = torch.arange(100, 100 + n, dtype=torch.int64)
    s.birth_frame = torch.arange(1, n + 1, dtype=torch.int64)
    s.source_surface_cell = torch.arange(200, 200 + n, dtype=torch.int64)
    s._suspect_streak = torch.arange(1, n + 1, dtype=torch.int64) * 3
    s._recover_streak = torch.arange(1, n + 1, dtype=torch.int64) * 5
    s._stable_streak = torch.arange(1, n + 1, dtype=torch.int64) * 9
    return s


_FIELDS = (
    "geometry_lifecycle",
    "appearance_lifecycle",
    "support_count",
    "contribution_ema",
    "rgb_residual_ema",
    "depth_residual_ema",
    "pos_grad_ema",
    "opacity_grad_ema",
    "free_space_conflict_ema",
    "geometry_confidence",
    "last_visible_step",
    "birth_frame",
    "source_surface_cell",
    "_suspect_streak",
    "_recover_streak",
    "_stable_streak",
)


def test_grow_append_leaves_existing_rows_bit_identical():
    """The load-bearing property: every existing row's state is untouched,
    field by field, not just 'doesn't crash'."""
    s = _seeded_state(n=4)
    before = {f: getattr(s, f).clone() for f in _FIELDS}

    s.grow_append(6, topology_version=1, iteration=50)

    assert s.n == 6
    assert s.topology_version == 1
    for f in _FIELDS:
        after = getattr(s, f)
        assert torch.equal(after[:4], before[f]), f"field {f!r} mutated existing rows"


def test_grow_append_new_rows_are_fresh_provisional():
    s = _seeded_state(n=4)
    s.grow_append(6, topology_version=1, iteration=50)

    provisional = state_to_code(GaussianLifecycleState.PROVISIONAL)
    assert int(s.geometry_lifecycle[4].item()) == provisional
    assert int(s.geometry_lifecycle[5].item()) == provisional
    assert int(s.appearance_lifecycle[4].item()) == provisional
    assert s.support_count[4].item() == 0.0
    assert s.contribution_ema[5].item() == 0.0
    assert s._suspect_streak[4].item() == 0
    assert s.birth_frame[4].item() == 50
    assert s.birth_frame[5].item() == 50
    assert s.last_visible_step[4].item() == -1
    assert s.source_surface_cell[5].item() == -1


def test_grow_append_zero_growth_is_noop_on_existing_rows_but_updates_topology_version():
    s = _seeded_state(n=4)
    before = {f: getattr(s, f).clone() for f in _FIELDS}

    s.grow_append(4, topology_version=1, iteration=50)

    assert s.n == 4
    assert s.topology_version == 1
    for f in _FIELDS:
        assert torch.equal(getattr(s, f), before[f]), f"field {f!r} mutated on zero-growth call"


def test_grow_append_rejects_shrink():
    s = _seeded_state(n=4)
    with pytest.raises(ValueError):
        s.grow_append(2, topology_version=1, iteration=50)


def test_grow_append_before_any_align_raises():
    s = LivePopulationState()
    with pytest.raises(StaleLiveStateError):
        s.grow_append(4, topology_version=0, iteration=0)


def test_align_pure_append_true_delegates_to_grow_append():
    s = _seeded_state(n=4, topology_version=0)
    before = {f: getattr(s, f).clone() for f in _FIELDS}

    s.align(6, topology_version=1, iteration=50, pure_append=True)

    assert s.n == 6
    assert s.topology_version == 1
    for f in _FIELDS:
        after = getattr(s, f)
        assert torch.equal(after[:4], before[f]), f"field {f!r} mutated via align(pure_append=True)"


def test_align_pure_append_true_still_cold_starts_on_first_call():
    """pure_append=True is a no-op override on the very first align() call
    -- there is nothing to append to yet, so the existing cold-start path
    must still run (grow_append() itself would raise)."""
    s = LivePopulationState()
    s.align(4, topology_version=0, iteration=0, pure_append=True)
    assert s.n == 4
    assert s.geometry_lifecycle is not None
    assert torch.all(s.geometry_lifecycle == state_to_code(GaussianLifecycleState.PROVISIONAL))


def test_align_pure_append_true_still_cold_starts_on_shrink():
    """A shrink can never be a pure append -- align() must fall through to
    the cold-start path rather than raising grow_append()'s ValueError,
    since a real structural change (prune) is exactly the case cold-start
    exists for."""
    s = _seeded_state(n=4)
    s.geometry_lifecycle[:] = state_to_code(GaussianLifecycleState.STABLE)

    s.align(2, topology_version=1, iteration=50, pure_append=True)

    assert s.n == 2
    assert s.topology_version == 1
    # Cold start: every row re-enters as PROVISIONAL, not the STABLE value
    # the (now-shrunk) surviving rows held before this call.
    assert torch.all(s.geometry_lifecycle == state_to_code(GaussianLifecycleState.PROVISIONAL))


def test_align_pure_append_false_is_unchanged_default_behavior():
    """Regression guard: omitting pure_append (or passing False) must
    reproduce today's cold-start-on-any-n-change behavior exactly --
    W3-39 must not change any existing caller's semantics implicitly."""
    s = _seeded_state(n=4)
    s.geometry_lifecycle[:] = state_to_code(GaussianLifecycleState.STABLE)

    s.align(6, topology_version=1, iteration=50)

    assert s.n == 6
    assert torch.all(s.geometry_lifecycle == state_to_code(GaussianLifecycleState.PROVISIONAL))
