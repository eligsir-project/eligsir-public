"""Wave 3, PR W3-09: observe-only mode is behavior-neutral (deliverable 1).

Mirrors the two-pronged approach already used by
``eligsir/optimization/gaussian/tests/test_no_topology_mutation.py``
(W3-02): a static source scan for known mutation entry points across the
whole ``strategies/`` package, plus a runtime check that a real
``gaussians``-shaped stub is bit-identical before/after the strategy runs.

**Updated by W3-10 (plan section 15), not reverted.** W3-10 is the PR that
deliberately, visibly crosses this boundary -- but only through one new
module, ``live/backend.py``, and only when a caller opts a phase in via
``ActivationPhaseConfig`` (every flag defaults off). The scan below
excludes exactly that one file, by name, with this comment as the paper
trail; every other module in ``strategies/`` (``mutation_plan.py``,
``lifecycle.py``, ``scoring.py``, ``cell_budget.py``, ``state.py``,
``strategy.py`` -- i.e. everything ``propose()``/``observe()`` touch)
remains held to the original observe-only guarantee this test enforces,
unchanged. See ``tests/test_w3_10_active_lifecycle_backend.py`` for the
backend module's own (very different) guarantee: an all-phases-off no-op,
not a total absence of mutation entry points.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import torch

_PKG_ROOT = Path(__file__).resolve().parents[1] / "eligsir" / "optimization" / "gaussian" / "strategies"

_FORBIDDEN_CALL_PATTERNS = (
    ".relocate_gs(",
    ".add_new_gs(",
    ".prune_points(",
    ".densify_and_prune(",
    "gaussians.means",
    "gaussians._xyz",
    "gaussians.params[",
)


def _non_docstring_lines(source: str) -> str:
    without_docstrings = re.sub(r'"""(?:.|\n)*?"""', "", source)
    lines = [line for line in without_docstrings.splitlines() if not line.strip().startswith("#")]
    return "\n".join(lines)


_W3_10_REAL_MUTATION_MODULE = _PKG_ROOT / "live" / "backend.py"
"""The one, deliberate, documented exception (see module docstring) --
W3-10's real-mutation entry point. Every other file in ``strategies/``
must still pass the scans below unmodified."""


def _all_py_files():
    return sorted(p for p in _PKG_ROOT.rglob("*.py") if p != _W3_10_REAL_MUTATION_MODULE)


def test_strategy_package_never_references_mutation_entry_points():
    hits: dict[str, list[str]] = {}
    for path in _all_py_files():
        source = _non_docstring_lines(path.read_text())
        found = [pat for pat in _FORBIDDEN_CALL_PATTERNS if pat in source]
        if found:
            hits[str(path.relative_to(_PKG_ROOT.parents[3]))] = found
    assert not hits, (
        f"strategies/ package references known map-mutation entry point(s): {hits!r} -- "
        "this package must remain observe-only (PopulationMutationPlan-proposal only) "
        "until W3-10 explicitly activates it."
    )


def test_w3_10_backend_module_really_does_contain_the_excluded_pattern():
    """Guards against the exclusion above silently becoming a no-op (e.g.
    if ``backend.py`` were later refactored to no longer call
    ``prune_points`` at all) -- if that ever happens, this scan should
    stop excluding the file, and this test would need updating alongside
    it rather than the exclusion quietly outliving its reason."""
    source = _non_docstring_lines(_W3_10_REAL_MUTATION_MODULE.read_text())
    found = [pat for pat in _FORBIDDEN_CALL_PATTERNS if pat in source]
    assert found, "backend.py no longer contains a real mutation entry point -- update the exclusion above"


def test_strategy_package_never_assigns_to_a_gaussians_attribute():
    """A slightly different check from the call-pattern scan above: catch
    an in-place assignment like ``gaussians.<attr> = ...`` or
    ``gaussians.get_xyz[idx] = ...``, which wouldn't match any of the
    named-call patterns above."""
    assignment_re = re.compile(r"\bgaussians\.[A-Za-z_][A-Za-z0-9_]*(\[[^\]]*\])?\s*=(?!=)")
    hits: dict[str, list[str]] = {}
    for path in _all_py_files():
        source = _non_docstring_lines(path.read_text())
        found = assignment_re.findall(source)
        if found:
            hits[str(path.relative_to(_PKG_ROOT.parents[3]))] = [m[0] or "<attr>" for m in found]
    assert not hits, f"strategies/ package assigns to a gaussians attribute in place: {hits!r}"


def test_engine_and_train_call_sites_are_untouched_by_this_pr():
    """Deliverable-1 diff-level guarantee (per the workstream brief): the
    existing training call sites that could plug a strategy into a real
    mutation path must show zero references to the new live-strategy
    package. This test only checks *this PR's introduced symbols* are not
    imported/referenced anywhere outside ``strategies/`` -- it does not
    (cannot, statically) prove train.py is byte-identical to main, but a
    real ``git diff main -- train.py train_streaming.py
    eligsir/optimization/gaussian/engine.py`` review is documented as
    part of this PR's completion doc.

    **Scope narrowed by W3-17 (2026-08), not reverted.** W3-17 is the PR
    that deliberately makes ``EliGSiRLiveGaussianStrategy`` reachable
    from a real run, via ``--densification_strategy gsplat_live_strategy``
    -- but strictly through one new indirection module,
    ``utils/strategies/mcmc/live_wrapped.py``, which is *not* in
    ``call_sites`` below and is the sole place that imports
    ``EliGSiRLiveGaussianStrategy`` directly. ``registry.py`` (both the
    metadata table and ``make_mcmc_strategy``) only ever imports
    ``live_wrapped``'s factory function, never the strategy class by name,
    so this test's literal string-scan assertions below still hold and are
    not weakened -- but its *stated purpose* (this class is entirely
    unreferenced by real training call sites) is no longer true in spirit,
    only in the narrow, literal sense this test actually checks. See
    ``live/strategy.py``'s module docstring for the up-to-date picture.
    """
    repo_root = _PKG_ROOT.parents[3]
    forbidden_names = ("EliGSiRLiveGaussianStrategy", "eligsir_live_observe_only")
    call_sites = [
        repo_root / "train.py",
        repo_root / "train_streaming.py",
        repo_root / "eligsir" / "optimization" / "gaussian" / "engine.py",
        repo_root / "utils" / "strategies" / "mcmc" / "registry.py",
        repo_root / "eligsir" / "mapping" / "strategies" / "registry.py",
    ]
    hits = {}
    for path in call_sites:
        if not path.exists():
            continue
        source = path.read_text()
        found = [name for name in forbidden_names if name in source]
        if found:
            hits[str(path)] = found
    assert not hits, f"Existing call sites reference the new live strategy: {hits!r}"


class _StubGaussians:
    def __init__(self, n: int = 8, seed: int = 0):
        g = torch.Generator().manual_seed(seed)
        self._xyz = torch.rand(n, 3, generator=g)
        self._opacity = torch.rand(n, 1, generator=g)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_opacity(self):
        return self._opacity


class _StubArgs:
    mcmc_dead_opacity_end = 0.01


def test_step_post_backward_does_not_mutate_gaussians_runtime():
    from eligsir.optimization.gaussian.strategies.live.strategy import EliGSiRLiveGaussianStrategy

    gaussians = _StubGaussians(n=12, seed=3)
    args = _StubArgs()
    xyz_before = gaussians.get_xyz.clone()
    opacity_before = gaussians.get_opacity.clone()
    n_before = gaussians.get_xyz.shape[0]

    strat = EliGSiRLiveGaussianStrategy()
    for it in range(5):
        strat.step_post_backward(gaussians, args, sched={}, iteration=it)

    assert gaussians.get_xyz.shape[0] == n_before
    assert torch.equal(gaussians.get_xyz, xyz_before)
    assert torch.equal(gaussians.get_opacity, opacity_before)


def test_build_mutation_plan_still_only_ever_produces_observe_only_true():
    """The W3-09 hard-refuse assertion on ``PopulationMutationPlan`` itself
    was relaxed by W3-10 (see ``contracts.py``'s updated docstring --
    ``observe_only=False`` is now constructible, but only intended to be
    used by ``live/backend.py``'s ``AlignedBufferRegistry``). What still
    must hold, unconditionally, is that ``build_mutation_plan`` --
    ``strategy.py``'s ``propose()`` call -- never itself produces an
    ``observe_only=False`` plan; this is what this test now checks,
    replacing the old dataclass-level hard-refuse assertion. See
    ``tests/test_w3_10_live_state_prune_append_rows.py::test_population_mutation_plan_observe_only_false_is_now_constructible``
    for the new, narrower guarantee at the dataclass level."""
    from eligsir.optimization.gaussian.strategies.live.mutation_plan import SafetyFuses, build_mutation_plan
    from eligsir.optimization.gaussian.strategies.live.lifecycle import LifecycleThresholds
    from eligsir.optimization.gaussian.strategies.live.state import LivePopulationState

    state = LivePopulationState()
    state.align(8, topology_version=0, iteration=0, device=torch.device("cpu"))
    plan, _streaks = build_mutation_plan(
        state=state, thresholds=LifecycleThresholds(), fuses=SafetyFuses(),
        iteration=0, topology_version=0,
    )
    assert plan.observe_only is True
