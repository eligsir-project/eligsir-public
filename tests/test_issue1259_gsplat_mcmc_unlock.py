"""Regression tests for issue #1259: unlock ``gsplat_mcmc`` on the canonical
mapping runtime (``CanonicalRuntimePolicy``), previously hard-gated to
``gsplat_energy_mcmc`` only with no accompanying rationale.

See ``eligsir/runtime/config.py``'s ``CanonicalRuntimePolicy.__post_init__``
for the full audit trail: three real crashes were found and fixed chasing
"gsplat_mcmc" through a real canonical-runtime run (a config-field
``AttributeError``, a ``provisional`` buffer shape mismatch, then a
``birth_iter`` shape mismatch), each covered by a dedicated test below.
"gsplat_default" (gsplat's own gradient-threshold clone/split/prune ADC --
the ORIGINAL non-MCMC 3DGS densification approach) was audited the same way
and looked equally safe by code-reading, but hit a FOURTH, deeper crash on a
real run: gsplat's ``DefaultStrategy`` needs ``.grad`` populated on the 2D
screen-space means tensor post-backward (``means2d.retain_grad()``), which
this render pipeline does not enable (only MCMC-based strategies have run on
it so far). That is a render-pipeline gap, not a bookkeeping one, and is
judged out of scope for this pass -- "gsplat_default" remains gated.
``test_gsplat_default_initialize_state_under_stripped_canonical_policy``
below documents what WAS verified for it (safe construction/init) without
overclaiming it is safe end-to-end.
"""

from types import SimpleNamespace

import pytest
import torch

from eligsir.runtime.config import CanonicalRuntimePolicy
from eligsir.utils.strategies.mcmc.registry import make_mcmc_strategy


class _GsplatSelection:
    uses_gsplat_layout = True


class _StrippedCanonicalArgs:
    """Mimics ``CanonicalOptimizationPolicy``'s exact field-availability
    contract: only an explicit allowlist of names resolves, everything else
    (including every ``legacy_iteration_cadence`` field --
    ``mcmc_stop_growth_iter``, ``mcmc_noise_stop_iter``, ``densify_from_iter``,
    etc.) raises ``AttributeError``, matching
    ``eligsir/runtime/config_boundary.py``'s real pop-set."""

    _STRIPPED = frozenset(
        {
            "mcmc_stop_growth_iter",
            "mcmc_noise_stop_iter",
            "densify_from_iter",
            "densify_until_iter",
            "densification_interval",
        }
    )

    def __init__(self, **values):
        self._values = values

    def __getattr__(self, name):
        if name in self._STRIPPED or name not in self._values:
            raise AttributeError(name)
        return self._values[name]


def _minimal_gaussians(n=8):
    """A minimal gsplat-layout-shaped model sufficient for
    ``gsplat.strategy.MCMCStrategy``/``DefaultStrategy``'s own
    ``check_sanity``/``initialize_state``."""
    means = torch.zeros(n, 3, requires_grad=True)
    scales = torch.zeros(n, 3, requires_grad=True)
    quats = torch.zeros(n, 4)
    quats[:, 0] = 1.0
    quats.requires_grad_(True)
    opacities = torch.zeros(n, requires_grad=True)
    sh0 = torch.zeros(n, 1, 3, requires_grad=True)
    shN = torch.zeros(n, 0, 3, requires_grad=True)
    params = {
        "means": means,
        "scales": scales,
        "quats": quats,
        "opacities": opacities,
        "sh0": sh0,
        "shN": shN,
    }
    optimizers = {
        key: torch.optim.Adam([value], lr=1e-3) for key, value in params.items()
    }

    class _Gaussians:
        @property
        def get_xyz(self):
            return self.params["means"]

    gaussians = _Gaussians()
    gaussians.params = params
    gaussians.optimizers = optimizers
    return gaussians


def test_gsplat_energy_mcmc_still_accepted():
    """Regression: the default strategy must be unaffected by the relaxed gate."""
    policy = CanonicalRuntimePolicy(densification_strategy="gsplat_energy_mcmc")
    assert policy.densification_strategy == "gsplat_energy_mcmc"


def test_gsplat_mcmc_now_accepted():
    policy = CanonicalRuntimePolicy(densification_strategy="gsplat_mcmc")
    assert policy.densification_strategy == "gsplat_mcmc"


@pytest.mark.parametrize("name", ["mcmc", "hybrid", "gsplat_default", "taming", "bogus"])
def test_other_strategies_still_rejected(name):
    """Legacy-layout and other strategies remain gated -- not part of this unlock."""
    with pytest.raises(ValueError, match="canonical runtime supports"):
        CanonicalRuntimePolicy(densification_strategy=name)


@pytest.mark.parametrize("name", ["gsplat_mcmc", "gsplat_default"])
def test_strategy_constructs_arbiter_owned_with_gsplat_layout(name):
    strategy = make_mcmc_strategy(
        name,
        gaussians=_GsplatSelection(),
        args=SimpleNamespace(cap_max=1000),
    )
    assert strategy.name == name
    # Every canonical strategy must expose this -- production_application.py
    # raises TypeError outright if it doesn't.
    assert callable(getattr(strategy, "delegated_post_backward_proposal", None))


@pytest.mark.parametrize("name", ["gsplat_mcmc", "gsplat_default"])
def test_strategy_proposal_absorbs_energy_kwargs_without_error(name):
    """Neither strategy has an energy-guidance utility score; the arbiter
    wrapper must still be able to pass utility=None/use_energy_mcmc=False
    through without the underlying strategy choking on them."""
    strategy = make_mcmc_strategy(
        name,
        gaussians=_GsplatSelection(),
        args=SimpleNamespace(cap_max=1000),
    )
    proposal = strategy.delegated_post_backward_proposal(
        gaussians=_GsplatSelection(),
        args=SimpleNamespace(cap_max=1000),
        sched={},
        iteration=1,
        optimizer_updates=1,
        utility=None,
        temperature=1.0,
        use_energy_mcmc=False,
        map_backend=None,
    )
    assert proposal is not None
    assert proposal.executor is not None


def test_gsplat_mcmc_initialize_state_under_stripped_canonical_policy():
    """Regression for the real crash found 2026-08-31: a canonical-runtime
    run of gsplat_mcmc failed on tick 1 with
    ``AttributeError: mcmc_stop_growth_iter`` because
    ``UpstreamGsplatMCMCStrategy.initialize_state`` had no fallback for a
    field ``CanonicalOptimizationPolicy`` never supplies. Exercises the real
    executor path (not just proposal construction) with a stand-in that
    mimics that exact stripped field set."""
    pytest.importorskip("gsplat", reason="requires the upstream gsplat MCMC strategy")
    from eligsir.utils.strategies.mcmc.baseline import UpstreamGsplatMCMCStrategy

    strategy = UpstreamGsplatMCMCStrategy()
    args = _StrippedCanonicalArgs(cap_max=1000, iterations=30_000)
    gaussians = _minimal_gaussians()
    # Must not raise AttributeError for any legacy_iteration_cadence field.
    strategy.initialize_state(gaussians=gaussians, args=args)
    assert strategy.strategy is not None
    assert strategy.state is not None


def test_gsplat_mcmc_honors_semantic_cadence_like_energy_mcmc():
    """Regression for the growth-cadence confound found 2026-08-31 (issue
    #1259 follow-up): on a real fr3 canonical-runtime comparison,
    ``gsplat_mcmc`` grew ~1,978 Gaussians vs. ``gsplat_energy_mcmc``'s
    ~459,607 on an identical recipe. Root cause was NOT an algorithmic
    difference in the strategies -- it was that ``production_application.py``
    always constructs strategies with ``semantic_cadence=True`` (deferring
    growth/relocation timing entirely to ``MCMCCadenceOwner``'s pixel-
    exposure-driven schedule), but the registry dropped that kwarg for
    ``gsplat_mcmc`` and ``UpstreamGsplatMCMCStrategy`` had no way to honor it
    even if passed -- so it kept applying gsplat's own fixed-iteration-count
    gate (refine_start_iter=500, refine_every=100) against this runtime's
    tiny optimizer_updates counter (a few hundred to low thousands per whole
    run), which almost never lines up. With semantic_cadence honored, both
    strategies now disarm their own internal iteration gate identically
    (refine_start_iter=0, refine_stop_iter=2**63-1, refine_every=1) and defer
    to the same external cadence owner, making a strategy comparison
    actually measure the strategies rather than an accidental cadence
    mismatch."""
    pytest.importorskip("gsplat", reason="requires the upstream gsplat MCMC strategy")
    from eligsir.utils.strategies.mcmc.baseline import UpstreamGsplatMCMCStrategy

    args = _StrippedCanonicalArgs(cap_max=1000, iterations=30_000)

    default_strategy = UpstreamGsplatMCMCStrategy()
    default_strategy.initialize_state(gaussians=_minimal_gaussians(), args=args)
    assert default_strategy.strategy.refine_start_iter == 500
    assert default_strategy.strategy.refine_stop_iter == 12_000
    assert default_strategy.strategy.refine_every == 100

    semantic_strategy = UpstreamGsplatMCMCStrategy(semantic_cadence=True)
    semantic_strategy.initialize_state(gaussians=_minimal_gaussians(), args=args)
    assert semantic_strategy.strategy.refine_start_iter == 0
    assert semantic_strategy.strategy.refine_stop_iter == 2**63 - 1
    assert semantic_strategy.strategy.refine_every == 1


def test_registry_forwards_semantic_cadence_to_gsplat_mcmc():
    """The registry must pass ``semantic_cadence`` through to
    ``UpstreamGsplatMCMCStrategy``, mirroring ``gsplat_energy_mcmc`` -- this
    is the exact plumbing gap that produced the growth-cadence confound
    above. ``ensure_arbiter_owned`` wraps the raw strategy in
    ``ArbiterOwnedMCMCStrategy``, whose ``__getattr__`` forwards attribute
    access to the wrapped strategy, so ``.semantic_cadence`` here reads
    straight through to the real ``UpstreamGsplatMCMCStrategy`` instance."""
    strategy = make_mcmc_strategy(
        "gsplat_mcmc",
        gaussians=_GsplatSelection(),
        args=SimpleNamespace(cap_max=1000),
        semantic_cadence=True,
    )
    assert strategy.semantic_cadence is True


def test_upstream_gsplat_mcmc_skips_delegated_call_when_growth_not_due():
    """Regression for a SECOND real GPU crash found 2026-08-31, immediately
    after fixing the semantic_cadence confound above:
    ``ValueError: applied growth had no due exposure debt`` from
    ``MCMCCadenceOwner.record_structural_mutation``. Root cause: gsplat's own
    upstream ``MCMCStrategy.step_post_backward`` bundles relocation and
    growth behind ONE combined cadence gate -- once ``semantic_cadence``
    disarms it, every invocation performs both actions, so a tick where only
    relocation was due (but not growth) still grew the population and
    violated the cadence owner's growth-specific exposure-debt invariant.
    The fix defers the WHOLE delegated call until ``sched["allow_growth"]``
    is true when ``semantic_cadence`` is set; with it False (default,
    non-canonical callers), behavior is unchanged -- the strategy always
    delegates, matching pre-existing behavior."""
    from eligsir.utils.strategies.mcmc.baseline import UpstreamGsplatMCMCStrategy

    calls = []

    class _StubUpstream:
        def step_post_backward(self, **kwargs):
            calls.append(kwargs)

    strategy = UpstreamGsplatMCMCStrategy(semantic_cadence=True)
    strategy.strategy = _StubUpstream()
    strategy.state = {}
    gaussians = _minimal_gaussians()

    # allow_relocation True but allow_growth False -- must NOT delegate.
    strategy.step_post_backward(
        gaussians=gaussians,
        args=SimpleNamespace(cap_max=1000),
        sched={"allow_growth": False, "allow_relocation": True},
        iteration=5,
    )
    assert calls == []

    # allow_growth True -- must delegate.
    strategy.step_post_backward(
        gaussians=gaussians,
        args=SimpleNamespace(cap_max=1000),
        sched={"allow_growth": True, "allow_relocation": False},
        iteration=6,
    )
    assert len(calls) == 1


def test_upstream_gsplat_mcmc_always_delegates_without_semantic_cadence():
    """Default (semantic_cadence=False) behavior is unchanged: the strategy
    always delegates to the upstream executor regardless of ``sched``
    contents, matching this class's pre-#1259-follow-up behavior for
    train.py/train_streaming.py callers that never pass semantic_cadence."""
    from eligsir.utils.strategies.mcmc.baseline import UpstreamGsplatMCMCStrategy

    calls = []

    class _StubUpstream:
        def step_post_backward(self, **kwargs):
            calls.append(kwargs)

    strategy = UpstreamGsplatMCMCStrategy()
    strategy.strategy = _StubUpstream()
    strategy.state = {}
    gaussians = _minimal_gaussians()

    strategy.step_post_backward(
        gaussians=gaussians,
        args=SimpleNamespace(cap_max=1000),
        sched={},
        iteration=1,
    )
    assert len(calls) == 1


def test_arbiter_resyncs_lifecycle_buffers_after_strategy_mutates_population():
    """Regression for the SECOND real crash found 2026-08-31: after the
    mcmc_stop_growth_iter fix above, a real gsplat_mcmc canonical-runtime run
    got one growth event further and then crashed in objective.py's
    ``_visible | _provisional`` mask assembly with a tensor-size mismatch.
    Root cause: strategies that delegate straight to gsplat's own upstream
    MCMCStrategy/DefaultStrategy mutate only ``gaussians.params``/
    ``.optimizers`` and know nothing about this repo's parallel per-Gaussian
    lifecycle buffers (``provisional`` etc.), unlike ``GsplatEnergyMCMCStrategy``
    which keeps them in sync itself. ``ArbiterOwnedMCMCStrategy`` (the single
    choke point every strategy's mutation flows through) must now call
    ``train_streaming._sync_streaming_state_lengths`` after every delegated
    strategy execution so this invariant holds regardless of which strategy
    mutated the population."""
    from eligsir.utils.strategies.mcmc.arbiter_owned import ArbiterOwnedMCMCStrategy

    class _GrowingGaussians:
        uses_gsplat_layout = True

        def __init__(self):
            self.params = {"means": torch.zeros(5, 3)}
            self.provisional = torch.zeros(5, dtype=torch.bool)
            self.birth_frame = torch.zeros(5, dtype=torch.long)
            # birth_iter caused a THIRD real crash (canonical lifecycle
            # birth_iter shape mismatch, production_application.py's
            # _advance_lifecycle) after the provisional fix above -- covered
            # by _sync_streaming_state_lengths's caller list, not that
            # helper itself, which is exactly the gap this test also checks.
            self.birth_iter = torch.zeros(5, dtype=torch.long)
            self.gaussian_uid = torch.arange(5, dtype=torch.int64)

        @property
        def get_xyz(self):
            return self.params["means"]

    class _GrowingStrategy:
        name = "gsplat_mcmc"

        def step_post_backward(self, gaussians, **_kwargs):
            # Simulate gsplat's own MCMCStrategy growing the population
            # in-place without touching any lifecycle buffer.
            gaussians.params["means"] = torch.zeros(9, 3)
            return {"added": 4}

    gaussians = _GrowingGaussians()
    owned = ArbiterOwnedMCMCStrategy(_GrowingStrategy(), gaussians, SimpleNamespace(cap_max=1000))
    proposal = owned.delegated_post_backward_proposal(
        gaussians=gaussians,
        args=SimpleNamespace(cap_max=1000),
        sched={},
        iteration=1,
    )
    proposal.executor()

    assert gaussians.provisional.shape[0] == 9
    assert gaussians.birth_frame.shape[0] == 9
    assert gaussians.birth_iter.shape[0] == 9
    assert bool((gaussians.provisional[:5] == False).all())
    # gaussian_uid must be padded with fresh, unique ids -- never a
    # zero/placeholder that would collide new rows onto an existing uid.
    assert gaussians.gaussian_uid.shape[0] == 9
    assert len(set(gaussians.gaussian_uid.tolist())) == 9
    assert list(gaussians.gaussian_uid[:5].tolist()) == [0, 1, 2, 3, 4]


def test_gsplat_default_initialize_state_under_stripped_canonical_policy():
    """gsplat_default's construction/init path is safe under the stripped
    canonical policy (it already uses getattr fallbacks throughout, per
    issue #368's documented graceful-degradation contract) -- but this does
    NOT mean gsplat_default is safe end-to-end. A real run hit a deeper,
    render-pipeline-level crash (gsplat's DefaultStrategy needs .grad on the
    2D means tensor, which this pipeline doesn't populate) that this
    construction-only test cannot exercise. gsplat_default remains gated on
    CanonicalRuntimePolicy pending that separate fix."""
    pytest.importorskip("gsplat", reason="requires the upstream gsplat DefaultStrategy")
    from eligsir.utils.strategies.mcmc.baseline import GsplatDefaultStrategy

    strategy = GsplatDefaultStrategy()
    args = _StrippedCanonicalArgs(cap_max=1000, iterations=30_000)
    gaussians = _minimal_gaussians()
    strategy.initialize_state(gaussians=gaussians, args=args, scene_scale=1.0)
    assert strategy.strategy is not None
    assert strategy.state is not None
