"""W3-34B: exact-parity gate for the branch-free sanitize rewrite.

Non-negotiable prerequisite (per the review that redefined task #34B --
see ``docs/architecture/2026-08-10-w3-34b-sanitize-graph-roi.md``) for the
three-arm A/B/C benchmark that task ran: the branch-free rewrite
(``_sanitize_gaussian_params_branch_free``,
``eligsir/mapping/integration/loss_diagnostics.py``) must produce
BIT-IDENTICAL results to the production, branch-gated
``_sanitize_gaussian_params`` for every input class the real function's
own logic implies:

  - an ordinary finite value (no clamp/nan_to_num should fire)
  - a large positive value (finite, but nowhere near the scaling clamp)
  - a large negative value (finite, symmetric case)
  - a scale exactly below the 15.0 clamp threshold (14.999...)
  - a scale exactly at the 15.0 clamp threshold (should NOT be altered --
    clamp(max=15.0) is a no-op at the boundary)
  - a scale exactly above the 15.0 clamp threshold (should be clamped to
    exactly 15.0)
  - NaN (should become 0.0 via nan_to_num, in every one of the 6 raw
    tensors sanitize visits, not just _scaling)
  - +Inf (should become 1e6)
  - -Inf (should become -1e6)

Also asserts a real healthy-path forward/backward step (real
``GsplatGaussianModel``, real render+backward gradients, no non-finite
values anywhere) leaves parameters bit-identical whether sanitized by the
production function or the branch-free rewrite -- confirming the
rewrite is a genuine no-op on already-healthy state, not just correct on
hand-built edge cases.

All 3 real production call sites (``train_streaming.py``'s
``loop_top``/``after_optstep``/``after_mcmc_noise`` hooks) invoke the
identical ``_sanitize_gaussian_params(gaussians, where, iteration)`` with
identical bounds and identical tensor-name list -- confirmed by direct
grep (see the findings doc's "call sites" section) -- so a single parity
test against that one function covers all 3 real call sites; there is no
per-call-site divergence to separately account for.

CPU is sufficient for the tensor-value cases below (clamp_/nan_to_num
are plain elementwise ops with identical semantics on CPU and
CUDA/ROCm); only the real-forward/backward healthy-path check requires
a GPU, and is skipped without one.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from eligsir.mapping.integration.loss_diagnostics import (
    _sanitize_gaussian_params,
    _sanitize_gaussian_params_branch_free,
)

# Raw tensor attribute names sanitize visits, matching the production
# function's own tuple (loss_diagnostics.py's _SANITIZE_TENSOR_NAMES).
_TENSOR_NAMES = ("_xyz", "_opacity", "_scaling", "_rotation", "_features_dc", "_features_rest")


class _FakeGaussians:
    """A minimal stand-in for GsplatGaussianModel exposing exactly the
    6 raw-tensor attributes _sanitize_gaussian_params reads/writes, each
    as a real leaf tensor (so .data mutation semantics match production),
    plus the ``get_scaling`` property production's diagnostic line reads
    (``float(gaussians.get_scaling.max())``). Deliberately NOT the real
    GsplatGaussianModel for these cases -- the real model's raw-tensor
    properties return the underlying nn.Parameter directly (see
    scene/gsplat_model.py's `_scaling`/etc.), so a plain object exposing
    the same attribute names plus this one derived property is
    behaviourally equivalent for what sanitize touches, without needing a
    full model construction for every edge case.
    """

    def __init__(self, values_by_name):
        for name in _TENSOR_NAMES:
            setattr(self, name, values_by_name.get(name))

    @property
    def get_scaling(self):
        return torch.exp(self._scaling)


def _make_gaussians(values_by_name):
    return _FakeGaussians(values_by_name)


def _clone_gaussians(g):
    return _make_gaussians({name: getattr(g, name).clone() for name in _TENSOR_NAMES})


def _run_both(base_values):
    """Build two independent copies of the same starting state, sanitize
    one with production, the other with the branch-free rewrite, and
    return both post-sanitize states for comparison."""
    g_old = _make_gaussians({k: v.clone() for k, v in base_values.items()})
    g_new = _make_gaussians({k: v.clone() for k, v in base_values.items()})
    _sanitize_gaussian_params(g_old, "test", iteration=0)
    _sanitize_gaussian_params_branch_free(g_new)
    return g_old, g_new


def _assert_bit_identical(g_old, g_new, base_values, case_name):
    for name in _TENSOR_NAMES:
        old_t = getattr(g_old, name)
        new_t = getattr(g_new, name)
        assert torch.equal(old_t, new_t), (
            f"{case_name}: production vs branch-free diverged on '{name}': "
            f"old={old_t}, new={new_t}, input was {base_values[name]}"
        )


def _base_healthy(n=4):
    """All-finite, well-within-range starting tensors for every raw name
    sanitize visits, so a single value can be overridden per test case
    without constructing 6 tensors by hand every time."""
    return {
        "_xyz": torch.randn(n, 3),
        "_opacity": torch.randn(n, 1) * 0.1,
        "_scaling": torch.full((n, 3), -3.0),  # well below the 15.0 clamp
        "_rotation": torch.randn(n, 4),
        "_features_dc": torch.randn(n, 1, 3) * 0.1,
        "_features_rest": torch.randn(n, 1, 3) * 0.1,
    }


# ---------------------------------------------------------------------------
# Ordinary / large finite values
# ---------------------------------------------------------------------------


def test_ordinary_finite_value_is_unchanged_and_matches():
    base = _base_healthy()
    g_old, g_new = _run_both(base)
    _assert_bit_identical(g_old, g_new, base, "ordinary_finite")
    # Also confirm sanitize is genuinely a no-op on healthy input (both
    # implementations), not just mutually consistent.
    for name in _TENSOR_NAMES:
        assert torch.equal(getattr(g_old, name), base[name]), (
            f"production sanitize altered healthy '{name}' unexpectedly"
        )


def test_large_positive_finite_value_matches():
    base = _base_healthy()
    base["_xyz"] = torch.full((4, 3), 1e5)  # large but finite, no clamp path involves _xyz
    g_old, g_new = _run_both(base)
    _assert_bit_identical(g_old, g_new, base, "large_positive_finite")


def test_large_negative_finite_value_matches():
    base = _base_healthy()
    base["_xyz"] = torch.full((4, 3), -1e5)
    g_old, g_new = _run_both(base)
    _assert_bit_identical(g_old, g_new, base, "large_negative_finite")


# ---------------------------------------------------------------------------
# The 15.0 raw-_scaling clamp boundary -- exactly below / at / above
# ---------------------------------------------------------------------------


def test_scaling_exactly_below_clamp_threshold_is_unchanged():
    base = _base_healthy()
    base["_scaling"] = torch.tensor([[14.999, 14.999, 14.999]] * 4)
    g_old, g_new = _run_both(base)
    _assert_bit_identical(g_old, g_new, base, "scaling_below_threshold")
    assert torch.equal(g_old._scaling, base["_scaling"]), "clamp altered a value below its threshold"


def test_scaling_exactly_at_clamp_threshold_is_unchanged():
    """clamp(max=15.0) is a no-op exactly at the boundary (>15.0 required
    to be altered, not >=15.0) -- both implementations must agree this
    boundary value passes through unchanged."""
    base = _base_healthy()
    base["_scaling"] = torch.full((4, 3), 15.0)
    g_old, g_new = _run_both(base)
    _assert_bit_identical(g_old, g_new, base, "scaling_at_threshold")
    assert torch.equal(g_old._scaling, base["_scaling"]), "clamp altered the exact-boundary value"


def test_scaling_exactly_above_clamp_threshold_is_clamped_to_15():
    base = _base_healthy()
    base["_scaling"] = torch.tensor([[15.0001, 20.0, 90.0]] * 4)
    g_old, g_new = _run_both(base)
    _assert_bit_identical(g_old, g_new, base, "scaling_above_threshold")
    assert torch.equal(g_old._scaling, torch.full((4, 3), 15.0)), (
        f"expected production to clamp to exactly 15.0, got {g_old._scaling}"
    )


# ---------------------------------------------------------------------------
# Non-finite values: NaN, +Inf, -Inf, in EVERY raw tensor sanitize visits
# (not just _scaling) -- both implementations must land on identical
# replacement values (nan->0.0, +inf->1e6, -inf->-1e6) for each.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_value,expected", [
    (float("nan"), 0.0),
    (float("inf"), 1e6),
    (float("-inf"), -1e6),
])
@pytest.mark.parametrize("tensor_name", _TENSOR_NAMES)
def test_nonfinite_value_replaced_identically_in_every_tensor(tensor_name, bad_value, expected):
    base = _base_healthy()
    t = base[tensor_name].clone()
    t.view(-1)[0] = bad_value
    base[tensor_name] = t
    g_old, g_new = _run_both(base)
    _assert_bit_identical(g_old, g_new, base, f"nonfinite[{tensor_name}]={bad_value}")
    got = getattr(g_old, tensor_name).view(-1)[0].item()
    # _scaling is special-cased: it goes through the raw clamp(max=15.0)
    # FIRST (both implementations, same order), and +Inf > 15.0 so it is
    # clamped to exactly 15.0 there -- the nan_to_num pass that follows is
    # then a no-op on that now-finite value, so it never reaches 1e6 the
    # way every other tensor's +Inf does. This is real, intentional
    # production behavior (the docstring's "preventive... every call"
    # note), not a bug this test papers over -- both implementations must
    # still agree on it.
    if tensor_name == "_scaling" and bad_value == float("inf"):
        expected = 15.0
    assert got == pytest.approx(expected), (
        f"production sanitize of {bad_value} in '{tensor_name}' gave {got}, expected {expected}"
    )


def test_scaling_nan_combined_with_over_threshold_value_matches():
    """A single call where _scaling has BOTH a NaN and an over-threshold
    finite value -- exercises the clamp-then-nan_to_num ordering (clamp
    first, which does not touch NaN since NaN comparisons are always
    False; nan_to_num second, which then replaces the NaN) identically in
    both implementations."""
    base = _base_healthy()
    base["_scaling"] = torch.tensor([[float("nan"), 90.0, -3.0]] * 4)
    g_old, g_new = _run_both(base)
    _assert_bit_identical(g_old, g_new, base, "scaling_nan_and_over_threshold")
    row = g_old._scaling[0]
    assert row[0].item() == 0.0  # nan -> 0.0
    assert row[1].item() == 15.0  # 90.0 clamped to 15.0 (untouched by nan_to_num since finite)
    assert row[2].item() == -3.0  # unaffected


def test_all_six_tensors_simultaneously_bad_matches():
    """Every raw tensor has a distinct problem at once (mixed NaN/Inf/
    over-threshold) -- the realistic 'a bad MCMC write corrupted several
    tensors in one step' case sanitize's own docstring describes."""
    base = _base_healthy()
    base["_xyz"] = torch.tensor([[float("nan"), 0.0, 0.0]] * 4)
    base["_opacity"] = torch.tensor([[float("inf")]] * 4)
    base["_scaling"] = torch.tensor([[float("-inf"), 90.0, -3.0]] * 4)
    base["_rotation"] = torch.tensor([[float("nan"), 0.0, 0.0, 1.0]] * 4)
    base["_features_dc"] = torch.full((4, 1, 3), float("inf"))
    base["_features_rest"] = torch.zeros(4, 0, 3)  # empty -- exercises the numel()==0 path
    g_old, g_new = _run_both(base)
    _assert_bit_identical(g_old, g_new, base, "all_six_simultaneously_bad")


# ---------------------------------------------------------------------------
# Real healthy-path forward/backward: real GsplatGaussianModel, real
# gradients, no non-finite values anywhere -- confirms the rewrite is a
# genuine no-op on real production state, not just on hand-built cases.
# ---------------------------------------------------------------------------

CUDA_AVAILABLE = torch.cuda.is_available()
try:
    import gsplat  # noqa: F401
    GSPLAT_AVAILABLE = True
except Exception:  # noqa: BLE001
    GSPLAT_AVAILABLE = False


@pytest.mark.skipif(
    not (CUDA_AVAILABLE and GSPLAT_AVAILABLE),
    reason="needs CUDA/ROCm + gsplat; run inside the training container per CLAUDE.md",
)
def test_real_healthy_path_forward_backward_leaves_params_identical():
    from gaussian_renderer import render
    from eligsir.optimization.gaussian.tests.test_gpu_parity_harness import (
        _build_camera,
        _build_gaussians,
    )

    def _pipe():
        return SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)

    def _one_healthy_step(seed):
        g = _build_gaussians(seed=seed)
        cam = _build_camera(image_size=32, seed=seed + 1)
        bg = torch.tensor([0.0, 0.0, 0.0], device="cuda")
        out = render(cam, g, _pipe(), bg, update_sh_rest=True, render_depth=False)["render"]
        out.sum().backward()
        # A real healthy render+backward step should not itself introduce
        # non-finite grads/params -- sanity-check before comparing sanitize
        # implementations, so a failure here is not misread as a sanitize bug.
        for name in _TENSOR_NAMES:
            t = getattr(g, name)
            assert torch.isfinite(t).all(), f"fixture produced non-finite '{name}' before sanitize ran"
        return g

    g_old = _one_healthy_step(seed=100)
    g_new = _one_healthy_step(seed=100)  # identical seed -> identical starting state

    for name in _TENSOR_NAMES:
        assert torch.equal(getattr(g_old, name), getattr(g_new, name)), (
            f"the two identically-seeded fixtures diverged on '{name}' before sanitize even ran"
        )

    _sanitize_gaussian_params(g_old, "test_real_healthy", iteration=0)
    _sanitize_gaussian_params_branch_free(g_new)

    for name in _TENSOR_NAMES:
        old_t = getattr(g_old, name)
        new_t = getattr(g_new, name)
        assert torch.equal(old_t, new_t), (
            f"real healthy-path param '{name}' diverged between production and branch-free sanitize"
        )
