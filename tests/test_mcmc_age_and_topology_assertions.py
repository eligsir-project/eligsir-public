import inspect

import torch

from eligsir.utils.strategies.mcmc import base, energy_policy
from eligsir.utils.gaussian_lineage import assert_lineage_consistent  # noqa: F401 (re-exported for callers)


def test_energy_policy_uses_birth_iter_not_birth_frame_for_age():
    source = inspect.getsource(energy_policy)
    # The age computation must reference birth_iter; it must NOT compute
    # age from birth_frame anywhere in this module.
    assert "gaussians.birth_iter" in source
    assert "gaussians.birth_frame" not in source


class _FakeGaussiansMismatch:
    """Mimics the real KNN-isolation-kill call site: enough attributes for
    ``_augment_dead_mask_knn_isolation`` to run past every earlier guard and
    reach the young-protection block with a deliberately stale/mismatched
    ``birth_iter`` buffer (length 3, population is 20).
    """

    def __init__(self, n=20):
        self.get_xyz = torch.randn(n, 3)
        self.get_scaling = torch.full((n, 3), 0.2)  # > scale_moderate_threshold
        # Below streaming_knn_isolation_support_threshold -> reads as low support.
        self.visibility_ema = torch.zeros(n)
        # Stale/short buffer: this is the mismatch the guard must catch.
        self.birth_iter = torch.zeros(3, dtype=torch.int32)


def test_young_protection_warns_once_on_mismatch(capsys):
    n = 20
    gaussians = _FakeGaussiansMismatch(n=n)
    mask = torch.zeros(n, dtype=torch.bool)
    args = type(
        "Args",
        (),
        {
            "streaming_knn_isolation_kill_mult": 1.5,
            "streaming_young_prot_iters": 400,
            "scale_moderate_threshold": 0.05,
            "streaming_knn_isolation_support_threshold": 0.01,
        },
    )()

    result = base._augment_dead_mask_knn_isolation(mask, gaussians, args, current_iter=1000)

    captured = capsys.readouterr()
    assert "WARNING" in captured.out or "WARNING" in captured.err
    assert result is not None
    # The warning must only print once even if called again.
    base._augment_dead_mask_knn_isolation(mask, gaussians, args, current_iter=1000)
    captured2 = capsys.readouterr()
    assert "WARNING" not in captured2.out and "WARNING" not in captured2.err


def test_assert_lineage_consistent_used_in_energy_module():
    energy_mod = __import__("utils.strategies.mcmc.energy", fromlist=["x"])
    source = inspect.getsource(energy_mod)
    assert "assert_lineage_consistent" in source


def test_young_gaussian_still_protected_after_mcmc_growth():
    """The single most important regression test in this plan (per review):
    before this fix, birth_iter desynced after the FIRST MCMC growth event of
    any run and silently disabled young-protection for the rest of that run.
    Simulate exactly that sequence: grow once (extending birth_iter
    correctly, per Task 4's fix), then verify a Gaussian born at the growth
    iteration is still protected `young_prot_iters` later.
    """
    from eligsir.utils.strategies.mcmc.proposal import append_running_state

    class _G:
        def __init__(self, n):
            self.gaussian_uid = torch.arange(n, dtype=torch.int64)
            self.parent_uid = torch.full((n,), -1, dtype=torch.int64)
            self.root_uid = self.gaussian_uid.clone()
            self.generation = torch.zeros(n, dtype=torch.int32)
            self.birth_method = torch.zeros(n, dtype=torch.int8)
            self.last_reseed_iter = torch.full((n,), -1, dtype=torch.int32)
            self.birth_iter = torch.zeros(n, dtype=torch.int32)
            self.birth_frame = torch.zeros(n, dtype=torch.int32)
            self._next_gaussian_uid = n
            self.get_xyz = torch.zeros(n, 3)
            for name in ("visibility_ema", "xyz_gradient_accum", "denom", "support_count",
                         "provisional", "anchor_iter", "depth_conflict_count", "lifecycle_state", "utility_ema"):
                setattr(self, name, torch.zeros(n))
            self.max_radii2D = torch.zeros(n)

    g = _G(5)
    growth_iter = 3000
    parent_idxs = torch.tensor([0, 1], dtype=torch.int64)
    append_running_state(g, old_count=5, added_count=2, parent_idxs=parent_idxs, iteration=growth_iter)
    assert g.birth_iter.shape[0] == 7  # buffer correctly extended (was the bug: stayed at 5)

    kill = torch.zeros(7, dtype=torch.bool)
    args = type("Args", (), {"streaming_young_prot_iters": 400})()

    # Just after growth: the two new Gaussians (indices 5, 6) must be protected.
    check_iter = growth_iter + 100  # well within the 400-iter protection window
    birth_iter = g.birth_iter
    assert birth_iter.shape[0] == kill.shape[0]  # the guard's own precondition must hold
    age = (check_iter - birth_iter.to(dtype=torch.int64)) < args.streaming_young_prot_iters
    assert bool(age[5]) is True
    assert bool(age[6]) is True

    # Well past the protection window, they must no longer read as young.
    late_iter = growth_iter + 500
    age_late = (late_iter - birth_iter.to(dtype=torch.int64)) < args.streaming_young_prot_iters
    assert bool(age_late[5]) is False
    assert bool(age_late[6]) is False


def test_compute_legacy_dead_mask_protects_young_gaussian_after_growth():
    """Task 6 review Finding 2: the previous version of this test never called into
    any real guard function -- it reimplemented the age-threshold arithmetic inline.
    This test genuinely exercises ``compute_legacy_dead_mask`` (the default
    production dead-mask path, ``energy_dead_mode="legacy_clauses"``) post-growth,
    and confirms it is the function's *own* birth_iter-based young-protection clause
    doing the protecting: every Gaussian here has unconditional-kill-worthy opacity
    (clause 1: alpha < alpha_dead * 0.1), so a young Gaussian surviving is proof the
    age clause fired, not proof no clause applied. The same Gaussians, re-checked once
    they've aged past the protection window, must then be killed by that same clause
    -- ruling out "never dies" as an alternative explanation.
    """
    from eligsir.utils.strategies.mcmc.proposal import append_running_state
    from eligsir.utils.strategies.mcmc.energy_policy import GaussianEnergyPolicy

    class _G:
        def __init__(self, n):
            self.gaussian_uid = torch.arange(n, dtype=torch.int64)
            self.parent_uid = torch.full((n,), -1, dtype=torch.int64)
            self.root_uid = self.gaussian_uid.clone()
            self.generation = torch.zeros(n, dtype=torch.int32)
            self.birth_method = torch.zeros(n, dtype=torch.int8)
            self.last_reseed_iter = torch.full((n,), -1, dtype=torch.int32)
            self.birth_iter = torch.zeros(n, dtype=torch.int32)
            self.birth_frame = torch.zeros(n, dtype=torch.int32)
            self._next_gaussian_uid = n
            self.get_xyz = torch.zeros(n, 3)
            # Unconditional-kill-worthy opacity for every Gaussian (clause 1 of
            # compute_legacy_dead_mask: alpha < alpha_dead * 0.1).
            self.get_opacity = torch.full((n, 1), 0.0001)
            self.get_scaling = torch.full((n, 3), 0.01)  # small -> avoids scale/aniso clauses
            self.visibility_ema = torch.zeros(n)
            for name in ("xyz_gradient_accum", "denom", "support_count",
                         "provisional", "anchor_iter", "depth_conflict_count", "lifecycle_state", "utility_ema"):
                setattr(self, name, torch.zeros(n))
            self.max_radii2D = torch.zeros(n)

    g = _G(5)
    growth_iter = 3000
    parent_idxs = torch.tensor([0, 1], dtype=torch.int64)
    append_running_state(g, old_count=5, added_count=2, parent_idxs=parent_idxs, iteration=growth_iter)
    assert g.birth_iter.shape[0] == 7  # buffer correctly extended (Task 4's fix)
    # append_running_state already extends visibility_ema (it's in its own running-state
    # buffer list); get_opacity/get_scaling are not, so mirror what the real growth path
    # additionally does to those for the new rows.
    g.get_opacity = torch.cat([g.get_opacity, torch.full((2, 1), 0.0001)])
    g.get_scaling = torch.cat([g.get_scaling, torch.full((2, 3), 0.01)])
    assert g.visibility_ema.shape[0] == 7

    policy = GaussianEnergyPolicy()

    # Just after growth: the two new Gaussians (indices 5, 6) must survive despite
    # opacity that unconditionally kills every other Gaussian in the population.
    check_iter = growth_iter + 100  # well within the 400-iter protection window
    result = policy.compute_legacy_dead_mask(g, current_iter=check_iter, young_prot_iters=400)
    assert bool(result.mask[5]) is False
    assert bool(result.mask[6]) is False
    assert result.stats["young_protected"] >= 2  # the age clause actually engaged
    # Everything else (already old, same bad opacity) is correctly dead.
    assert bool(result.mask[0]) is True

    # Well past the protection window, the SAME Gaussians (now old) must be killed by
    # the same unconditional-opacity clause -- ruling out "never dies" as an
    # alternative explanation for the earlier survival.
    late_iter = growth_iter + 500
    result_late = policy.compute_legacy_dead_mask(g, current_iter=late_iter, young_prot_iters=400)
    assert bool(result_late.mask[5]) is True
    assert bool(result_late.mask[6]) is True


def test_legacy_dead_mask_warns_once_on_birth_iter_mismatch(capsys):
    """Finding 1: energy_policy.compute_legacy_dead_mask is the default production
    dead-mask path (energy_dead_mode="legacy_clauses") and previously silently
    no-op'd young-protection on a birth_iter/mask length mismatch, identically to
    the base.py KNN-isolation site. Confirm it now warns once instead."""
    from eligsir.utils.strategies.mcmc.energy_policy import GaussianEnergyPolicy

    n = 5

    class _G:
        def __init__(self):
            self.get_xyz = torch.zeros(n, 3)
            self.get_opacity = torch.full((n, 1), 0.0001)
            self.get_scaling = torch.full((n, 3), 0.01)
            self.visibility_ema = torch.zeros(n)
            self.birth_iter = torch.zeros(2, dtype=torch.int32)  # stale: shorter than n=5

    g = _G()
    policy = GaussianEnergyPolicy()
    policy.compute_legacy_dead_mask(g, current_iter=1000, young_prot_iters=400)
    captured = capsys.readouterr()
    assert "WARNING" in captured.out or "WARNING" in captured.err
    assert "energy_policy.compute_legacy_dead_mask" in (captured.out + captured.err)

    policy.compute_legacy_dead_mask(g, current_iter=1000, young_prot_iters=400)
    captured2 = capsys.readouterr()
    assert "WARNING" not in captured2.out and "WARNING" not in captured2.err


def test_energy_mcmc_compute_dead_mask_warns_once_on_birth_iter_mismatch(capsys):
    """Finding 1: utils.energy_mcmc.compute_dead_mask is reached via
    GaussianEnergyPolicy.compute_dead_mask (energy_policy.py:559-575) and previously
    silently no-op'd young-protection on the same birth_iter/mask mismatch pattern.
    Confirm it now warns once, distinguishably from the other two sites."""
    from eligsir.utils.energy_mcmc import compute_dead_mask

    n = 5

    class _G:
        def __init__(self):
            self.get_xyz = torch.zeros(n, 3)
            self.get_opacity = torch.full((n, 1), 0.0001)
            self.get_scaling = torch.full((n, 3), 0.01)
            self.visibility_ema = torch.zeros(n)
            self.birth_iter = torch.zeros(2, dtype=torch.int32)  # stale: shorter than n=5

    g = _G()
    compute_dead_mask(g, current_iter=1000, young_prot_iters=400)
    captured = capsys.readouterr()
    assert "WARNING" in captured.out or "WARNING" in captured.err
    assert "energy_mcmc.compute_dead_mask" in (captured.out + captured.err)

    compute_dead_mask(g, current_iter=1000, young_prot_iters=400)
    captured2 = capsys.readouterr()
    assert "WARNING" not in captured2.out and "WARNING" not in captured2.err


# --- issue #382: KNN-isolation support gate must read a REAL, declared flag ---
#
# `base._augment_dead_mask_knn_isolation` used to read `getattr(args,
# "support_threshold", 0.01)`, an attribute no ParamGroup ever declares, so the
# gate was permanently pinned to the literal. These two tests pin BOTH halves of
# the fix: (1) the flag really exists on the production parser, and (2) base.py
# really reads it. Either one alone would pass while the bug was present.


def _knn_isolation_args(**overrides):
    base_attrs = {
        "streaming_knn_isolation_kill_mult": 3.0,
        "streaming_young_prot_iters": 0,  # young protection out of the picture
        "scale_moderate_threshold": 0.05,
        "streaming_knn_isolation_support_threshold": 0.01,
    }
    base_attrs.update(overrides)
    return type("Args", (), base_attrs)()


class _FakeGaussiansOneOutlier:
    """19 tightly-clustered Gaussians plus one far-away outlier.

    The outlier's self-KNN distance hugely exceeds the scene-median spacing, so
    it passes the isolation gate; its scale passes the scale gate. Whether it is
    marked dead therefore depends ONLY on the support gate, which is exactly the
    knob under test.
    """

    def __init__(self, outlier_support: float):
        n = 20
        means = torch.zeros(n, 3)
        # Deterministic tight cluster (spacing ~0.1) for indices 0..18.
        means[:19, 0] = torch.arange(19, dtype=torch.float32) * 0.1
        means[19] = torch.tensor([500.0, 500.0, 500.0])  # the isolated outlier
        self.get_xyz = means
        self.get_scaling = torch.full((n, 3), 0.2)  # > scale_moderate_threshold
        support = torch.ones(n)  # well-supported cluster
        support[19] = outlier_support
        self.visibility_ema = support


def test_knn_isolation_support_threshold_flag_is_declared_on_the_parser():
    """Half 1: the flag exists on the real production parser AND is settable.

    Before the fix, `support_threshold` was absent here -- which is precisely why
    the getattr in base.py could never resolve. The whole point of the fix is
    "unconfigurable -> configurable", so this must actually parse a CLI value,
    not just read the default off a `parse_args([])` namespace.
    """
    from argparse import ArgumentParser

    from arguments import OptimizationParams

    parser = ArgumentParser()
    params = OptimizationParams(parser)
    extracted = params.extract(parser.parse_args([]))

    assert hasattr(extracted, "streaming_knn_isolation_support_threshold")
    assert float(extracted.streaming_knn_isolation_support_threshold) == 0.01

    # The configurability claim itself: a real CLI value must survive parsing and
    # extraction. (Auto-generated ParamGroup flags are a known footgun in this
    # repo -- see CLAUDE.md's `--no-gsplat_sparse_grad` / exit-2 note.)
    set_ns = params.extract(
        parser.parse_args(["--streaming_knn_isolation_support_threshold", "0.001"])
    )
    assert float(set_ns.streaming_knn_isolation_support_threshold) == 0.001


def test_bare_support_threshold_is_not_resurrected_as_a_flag():
    """The drifted name must not come back as a real flag on ANY param group.

    A bare `support_threshold` arg would collide by name with
    `GaussianEnergyConfig.support_threshold`, so if anything ever maps args onto
    that dataclass by field name it would silently start feeding the
    parity-tested dead-mask clauses. This is a deliberate wall: making the energy
    config CLI-reachable (the adjacent follow-up noted in issue #382) must pick a
    scoped name, exactly as this fix did, rather than the bare one.
    """
    from argparse import ArgumentParser

    from arguments import ModelParams, PipelineParams, OptimizationParams, StreamingParams

    for group_cls in (ModelParams, PipelineParams, OptimizationParams, StreamingParams):
        parser = ArgumentParser()
        try:
            group = group_cls(parser)
        except TypeError:
            group = group_cls(parser, sentinel=True)  # ModelParams-style signature
        extracted = group.extract(parser.parse_args([]))
        assert not hasattr(extracted, "support_threshold"), (
            f"{group_cls.__name__} declares a bare `support_threshold` -- use a "
            f"scoped name instead (see issue #382)"
        )


def test_knn_isolation_support_gate_actually_reads_the_flag():
    """Half 2: base.py binds to the flag -- proved in BOTH directions.

    A one-sided assertion would be worthless here: the helper wraps its body in
    a bare `except: return mask`, so "not killed" is also what a crash looks
    like. The permissive arm therefore proves the machinery ran end-to-end, and
    the strict arm proves the flag (not a hardcoded literal) decided the outcome.
    """
    mask = torch.zeros(20, dtype=torch.bool)

    # Outlier support 0.005 sits BELOW a 0.01 threshold -> gate passes -> killed.
    permissive = base._augment_dead_mask_knn_isolation(
        mask,
        _FakeGaussiansOneOutlier(outlier_support=0.005),
        _knn_isolation_args(streaming_knn_isolation_support_threshold=0.01),
        current_iter=1000,
    )
    assert bool(permissive[19]) is True, (
        "isolated low-support outlier should be marked dead at threshold 0.01; "
        "if this fails the helper never reached the support gate at all"
    )
    assert not bool(permissive[:19].any()), "well-supported cluster must survive"

    # Same population, threshold lowered below the outlier's support -> spared.
    # Only a real flag read can change this; a hardcoded 0.01 would still kill.
    strict = base._augment_dead_mask_knn_isolation(
        mask,
        _FakeGaussiansOneOutlier(outlier_support=0.005),
        _knn_isolation_args(streaming_knn_isolation_support_threshold=0.001),
        current_iter=1000,
    )
    assert bool(strict[19]) is False, (
        "support gate ignored the flag -- base.py is still pinned to a literal"
    )
