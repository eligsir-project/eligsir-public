"""Issue #424: sequence-level tensor parity for the sanitizer production
actually calls.

Context / why this file exists
------------------------------
Two test files already guard pieces of the W3-34B/W3-61 sanitize rework:

  - ``tests/test_w3_34b_sanitize_parity.py`` proves the *numeric core*
    (``_sanitize_gaussian_params_branch_free``) is bit-identical to the
    original gated ``_sanitize_gaussian_params`` on 32 hand-built input
    classes (clamp boundary, NaN/+Inf/-Inf in each of the 6 raw tensors,
    empty tensors, all-six-bad-at-once).
  - ``tests/test_w3_61_sanitize_wiring.py`` proves the *cadence* of
    ``_sanitize_gaussian_params_sampled`` (burn-in window, post-burn-in
    interval, rate-limited counting).

Neither covers the composition. What production calls at all four
``train_streaming.py`` call sites is ``_sanitize_gaussian_params_sampled``
-- a wrapper that, depending on the iteration number, either runs the
diagnostics-only scan *and* the branch-free core, or the core alone. The
claim that matters for correctness is therefore not "the core matches
production on one call" but:

    for ANY iteration sequence and ANY corruption pattern, the resulting
    tensor state after ``_sanitize_gaussian_params_sampled`` is
    bit-identical to the state after the original
    ``_sanitize_gaussian_params``.

That is a strictly stronger statement than either existing file makes,
and it is the one that would break if someone ever gave the
diagnostics-only half a write (today it is read-only, which is precisely
why this passes -- and precisely why it is worth pinning, since nothing
else in the suite would catch that regression).

It also covers the two iteration-dependent branches the single-call
parity test structurally cannot reach: the burn-in window (diagnostics
run every iteration) and the post-burn-in sampled ticks (diagnostics run
on some iterations and not others). If sampling could perturb tensor
values, the divergence would only appear across a *sequence*, at
whichever iteration the two paths first take different branches.

Tolerance
---------
Bit-identical (``torch.equal``), not approximate. No tolerance is claimed
or needed: both paths ultimately apply the identical ``clamp_(max=15.0)``
and ``nan_to_num(nan=0.0, posinf=1e6, neginf=-1e6)`` ops to the same
tensors in the same order, and the only structural difference (production
gates those ops behind host-synced ``int(...)`` counts) gates them on a
condition that is true exactly when the op would be a no-op anyway. There
is no floating-point reassociation anywhere in either path, so exact
equality is the correct bar.

CPU-only by construction -- ``clamp_``/``nan_to_num`` are plain
elementwise ops, and no CUDA/ROCm device or ``gsplat`` build is required.
"""

from __future__ import annotations

import random

import pytest

torch = pytest.importorskip("torch")

from eligsir.mapping.integration.loss_diagnostics import (  # noqa: E402
    _SANITIZE_DIAG_BURN_IN_ITERS,
    _sanitize_gaussian_params,
    _sanitize_gaussian_params_sampled,
)

# The 6 raw tensor attributes sanitize reads/writes, in production's own
# visiting order (loss_diagnostics.py's _SANITIZE_TENSOR_NAMES).
_TENSOR_NAMES = ("_xyz", "_opacity", "_scaling", "_rotation", "_features_dc", "_features_rest")

# Per-tensor trailing shape, so a randomized fixture can build all six with
# realistic ranks (matching the real GsplatGaussianModel layout) without
# hand-writing them per test.
_TENSOR_SHAPES = {
    "_xyz": (3,),
    "_opacity": (1,),
    "_scaling": (3,),
    "_rotation": (4,),
    "_features_dc": (1, 3),
    "_features_rest": (1, 3),
}

_ADVERSARIAL_VALUES = (
    float("nan"),
    float("inf"),
    float("-inf"),
    15.0,        # exactly at the clamp boundary -- must NOT be altered
    15.0001,     # just above -- must be clamped to exactly 15.0
    14.999,      # just below -- must NOT be altered
    90.0,        # the exp()-overflow case sanitize's docstring describes
    -90.0,
    1e6,         # collides with nan_to_num's own posinf replacement value
    -1e6,
    0.0,
)


class _FakeGaussians:
    """Minimal stand-in exposing exactly what sanitize touches: the 6 raw
    leaf tensors plus the ``get_scaling`` property production's diagnostic
    line reads. Mirrors ``tests/test_w3_34b_sanitize_parity.py``'s fixture
    deliberately, so the two files agree on what "a Gaussian model" means
    for sanitize's purposes.

    NOTE: ``get_scaling`` would raise on a ``None`` ``_scaling``. That is
    not a latent fixture bug -- production short-circuits on the
    ``_scaling is None`` branch before ever reading ``get_scaling``, so
    the property is unreachable in that case. Do not misread a fixture
    error there as an implementation divergence.
    """

    def __init__(self, values_by_name):
        for name in _TENSOR_NAMES:
            setattr(self, name, values_by_name.get(name))

    @property
    def get_scaling(self):
        return torch.exp(self._scaling)


def _random_state(rng: random.Random, n: int = 6, *, dtype=torch.float32):
    """A healthy, all-finite starting state with values in realistic
    ranges (raw ``_scaling`` well below the 15.0 clamp, as real Gaussians
    are)."""
    gen = torch.Generator().manual_seed(rng.randint(0, 2**31 - 1))
    values = {}
    for name in _TENSOR_NAMES:
        shape = (n,) + _TENSOR_SHAPES[name]
        t = torch.randn(shape, generator=gen, dtype=dtype)
        if name == "_scaling":
            t = t - 3.0  # well below the clamp threshold
        values[name] = t
    return values


def _inject(rng: random.Random, values, *, n_corruptions: int):
    """Scatter adversarial values into random positions of random
    tensors, in place, returning the mutated dict."""
    for _ in range(n_corruptions):
        name = rng.choice(_TENSOR_NAMES)
        t = values[name]
        flat = t.view(-1)
        if flat.numel() == 0:
            continue
        flat[rng.randrange(flat.numel())] = rng.choice(_ADVERSARIAL_VALUES)
    return values


def _twins(values):
    """Two independent deep copies of the same starting state -- one for
    each sanitize path -- so neither can observe the other's writes."""
    a = _FakeGaussians({k: (v.clone() if v is not None else None) for k, v in values.items()})
    b = _FakeGaussians({k: (v.clone() if v is not None else None) for k, v in values.items()})
    return a, b


def _bits(t: "torch.Tensor") -> "torch.Tensor":
    """Reinterpret a float32 tensor as int32 for a true bit-level compare.

    ``torch.equal`` is NOT usable here: it applies IEEE ``==`` semantics,
    under which ``NaN != NaN``, so two byte-for-byte identical tensors
    that both contain a NaN compare unequal. Several cases in this file
    deliberately compare states that still hold an injected NaN (the
    pre-sanitize checkpoint, and the read-only probe on the diagnostics
    half, which by design does not remove NaN). Comparing the raw bit
    patterns instead is both correct for those cases and a strictly
    stronger assertion than ``torch.equal`` elsewhere -- it additionally
    distinguishes ``+0.0`` from ``-0.0`` and separates distinct NaN
    payloads, neither of which ``==`` can see.
    """
    assert t.dtype == torch.float32, f"_bits expects float32, got {t.dtype}"
    return t.detach().contiguous().view(torch.int32)


def _assert_bit_identical(g_prod, g_sampled, case: str, iteration: int):
    for name in _TENSOR_NAMES:
        p = getattr(g_prod, name)
        s = getattr(g_sampled, name)
        if p is None or s is None:
            assert p is None and s is None, f"{case} @ iter {iteration}: '{name}' None-ness diverged"
            continue
        assert torch.equal(_bits(p), _bits(s)), (
            f"{case} @ iter {iteration}: production _sanitize_gaussian_params and "
            f"_sanitize_gaussian_params_sampled diverged on '{name}'\n"
            f"  production={p}\n  sampled   ={s}"
        )


# ---------------------------------------------------------------------------
# The core property: over a long iteration sequence crossing the burn-in
# boundary, with randomized corruption, the two paths never diverge.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", list(range(12)))
def test_sampled_wrapper_matches_production_over_iteration_sequence(seed):
    """Randomized property test. Each seed drives a different corruption
    pattern, different injection iterations, and a different diagnostic
    interval. The sequence deliberately straddles
    ``_SANITIZE_DIAG_BURN_IN_ITERS`` so both the every-iteration burn-in
    branch and the post-burn-in sampled branch are exercised within one
    run, including iterations where the two branches differ.
    """
    rng = random.Random(seed)
    diag_interval = rng.choice([1, 2, 7, 100])

    values = _random_state(rng)
    g_prod, g_sampled = _twins(values)

    # Straddle the burn-in boundary: a stretch before it, the boundary
    # itself, and a stretch after it where sampling actually kicks in.
    burn_in = _SANITIZE_DIAG_BURN_IN_ITERS
    iterations = (
        list(range(0, 5))
        + [burn_in - 1, burn_in, burn_in + 1]
        + list(range(burn_in + 2, burn_in + 40))
    )

    for iteration in iterations:
        # Randomly corrupt BOTH twins identically, mid-sequence, the way a
        # bad optimizer step or MCMC write would between two sanitize calls.
        if rng.random() < 0.35:
            corrupt = _inject(rng, {n: getattr(g_prod, n).clone() for n in _TENSOR_NAMES},
                              n_corruptions=rng.randint(1, 4))
            for name in _TENSOR_NAMES:
                getattr(g_prod, name).data.copy_(corrupt[name])
                getattr(g_sampled, name).data.copy_(corrupt[name])
            _assert_bit_identical(g_prod, g_sampled, f"seed{seed}/pre-sanitize", iteration)

        _sanitize_gaussian_params(g_prod, "seq_parity", iteration, _limit=0)
        _sanitize_gaussian_params_sampled(
            g_sampled, "seq_parity", iteration, diag_interval=diag_interval, _limit=0
        )

        _assert_bit_identical(
            g_prod, g_sampled, f"seed{seed}/interval{diag_interval}", iteration
        )


@pytest.mark.parametrize("seed", list(range(8)))
def test_sampled_wrapper_matches_production_single_call_randomized(seed):
    """Single-call randomized parity across heavily-corrupted states --
    denser corruption than the sequence test uses, to hammer the
    clamp/nan_to_num interaction (notably ``_scaling`` holding a NaN and
    an over-threshold value at once, where production's ``_over`` gate
    evaluates NaN comparisons to False)."""
    rng = random.Random(1000 + seed)
    values = _inject(rng, _random_state(rng, n=8), n_corruptions=rng.randint(5, 20))
    g_prod, g_sampled = _twins(values)

    iteration = rng.randrange(0, 5000)
    _sanitize_gaussian_params(g_prod, "single_parity", iteration, _limit=0)
    _sanitize_gaussian_params_sampled(
        g_sampled, "single_parity", iteration, diag_interval=1, _limit=0
    )
    _assert_bit_identical(g_prod, g_sampled, f"single/seed{seed}", iteration)


# ---------------------------------------------------------------------------
# Structural edge cases the wrapper's own branches can reach
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("iteration", [0, 1, _SANITIZE_DIAG_BURN_IN_ITERS, _SANITIZE_DIAG_BURN_IN_ITERS + 1, 5000])
def test_scaling_is_none_matches(iteration):
    """``_scaling`` absent entirely. Production takes its
    ``hasattr``/``is not None`` else-branch (and never reads
    ``get_scaling``); the branch-free core skips the clamp; the
    diagnostics half skips its scan. All three must agree, at every
    cadence branch."""
    rng = random.Random(77)
    values = _random_state(rng)
    values["_scaling"] = None
    g_prod, g_sampled = _twins(values)

    _sanitize_gaussian_params(g_prod, "none_scaling", iteration, _limit=0)
    _sanitize_gaussian_params_sampled(
        g_sampled, "none_scaling", iteration, diag_interval=1, _limit=0
    )
    _assert_bit_identical(g_prod, g_sampled, "scaling_is_none", iteration)


@pytest.mark.parametrize("iteration", [0, _SANITIZE_DIAG_BURN_IN_ITERS + 1])
def test_all_tensors_empty_matches(iteration):
    """Zero Gaussians -- every tensor has ``numel() == 0``. Production
    guards its diagnostic reads on ``_sc.numel()``; the branch-free core
    clamps/nan_to_nums unconditionally (a no-op on an empty tensor).
    Must not raise, must not diverge."""
    empty = {name: torch.zeros((0,) + _TENSOR_SHAPES[name]) for name in _TENSOR_NAMES}
    g_prod, g_sampled = _twins(empty)

    _sanitize_gaussian_params(g_prod, "empty", iteration, _limit=0)
    _sanitize_gaussian_params_sampled(
        g_sampled, "empty", iteration, diag_interval=1, _limit=0
    )
    _assert_bit_identical(g_prod, g_sampled, "all_empty", iteration)
    for name in _TENSOR_NAMES:
        assert getattr(g_sampled, name).numel() == 0


def test_diagnostics_half_never_writes():
    """The direct statement of *why* the parity above holds: the
    diagnostics-only half is read-only. Pinning this separately means a
    future change that gave it a write would fail here with an
    unambiguous message, rather than surfacing as a confusing
    sequence-parity failure at some arbitrary iteration.
    """
    from eligsir.mapping.integration.loss_diagnostics import (
        _sanitize_gaussian_params_diagnostics_only,
    )

    rng = random.Random(4242)
    values = _inject(rng, _random_state(rng, n=8), n_corruptions=12)
    g = _FakeGaussians({k: v.clone() for k, v in values.items()})
    before = {name: getattr(g, name).clone() for name in _TENSOR_NAMES}

    _sanitize_gaussian_params_diagnostics_only(g, "readonly_probe", 0, _limit=0)

    for name in _TENSOR_NAMES:
        assert torch.equal(_bits(getattr(g, name)), _bits(before[name])), (
            f"_sanitize_gaussian_params_diagnostics_only WROTE to '{name}' -- it is "
            "required to be read-only; the sampled wrapper's tensor-state parity with "
            "production depends on it."
        )
