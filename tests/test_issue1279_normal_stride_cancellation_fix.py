"""Regression test for the §4.2 finite-difference normal-normalization bug
described in
``docs/architecture/2026-08-29-staged-coarse-to-fine-depth-insertion.md``.

Root cause: ``insert_gaussians_from_frame``'s three near-identical branches
(gpu_resident / use_gpu / numpy) each computed

    s = depth_stride
    denom_x = float(d_fx * (2.0 * s / depth_stride))   # == 2.0 * d_fx, ALWAYS

Because every call site passed ``s == depth_stride``, the ratio was
identically 1 and the denominator silently collapsed to a stride-independent
constant, while the finite-difference numerator (``z[x+s] - z[x-s]``) grows
in magnitude with ``s``. So the estimated normal tilt scaled with stride
while its normalizing divisor did not -- a systematic, stride-dependent bias
in the grazing-rejection test (``nz > cos(max_view_angle)``) that was
invisible only because stride was a fixed constant for the whole run.

The fix (``geometry_helpers.finite_diff_normal_denominators``) makes the
denominator scale with ``s`` too (``2 * s * fx``), so a truly planar surface's
estimated normal tilt is invariant to which stride it was sampled at -- which
is the physically correct behavior and the precondition for building any
depth-insertion density *schedule* (varying stride over training) without
introducing a confound into the grazing-rejection rate.

CPU-only, no torch/GPU dependency: this is pure arithmetic on the same
formula the three branches share, so it exercises exactly what changed
without needing to drive the full ~900-line ``insert_gaussians_from_frame``.
"""

from __future__ import annotations

import numpy as np

from eligsir.mapping.integration.geometry_helpers import (
    finite_diff_normal_denominators,
)


def _old_buggy_denominators(fx: float, fy: float, s: int, depth_stride: int):
    """The exact pre-fix formula, kept here only to prove the fix changes it."""
    return float(fx * (2.0 * s / depth_stride)), float(fy * (2.0 * s / depth_stride))


def _tilt_for_planar_ramp(fx: float, fy: float, s: int, depth_stride: int, slope: float, use_fixed: bool):
    """Reproduce the exact nx/ny/nz computation the three call sites share,
    for a synthetic depth map that is a perfect plane tilted along x with the
    given real-world slope (dz/dx_world = slope). Returns the *normalized*
    nx (the quantity the grazing test ``nz > cos(max_view_angle)`` reacts to).
    """
    # A perfectly planar surface: z(x) = z0 + slope * (x / fx) * z0-ish is not
    # quite linear in pixel space for a real projective camera, but for the
    # purpose of isolating the denominator bug we only need a depth map whose
    # per-pixel finite difference scales linearly with the sampling offset
    # `s` -- exactly true for any linear-in-pixel-coordinate depth ramp,
    # which is what a real near-planar patch approximates locally.
    dzdx = 2.0 * s * slope  # finite difference across a 2*s-pixel window

    if use_fixed:
        denom_x, denom_y = finite_diff_normal_denominators(fx, fy, s)
    else:
        denom_x, denom_y = _old_buggy_denominators(fx, fy, s, depth_stride)

    nx = -dzdx / denom_x
    ny = 0.0  # no y-slope in this synthetic ramp
    nz = 1.0
    norm = np.sqrt(nx * nx + ny * ny + nz * nz) + 1e-8
    return nx / norm, nz / norm


def test_finite_diff_normal_denominators_scales_linearly_with_s():
    fx, fy = 500.0, 480.0
    denom_x_1, denom_y_1 = finite_diff_normal_denominators(fx, fy, 4)
    denom_x_2, denom_y_2 = finite_diff_normal_denominators(fx, fy, 8)
    assert denom_x_1 == fx * 2.0 * 4
    assert denom_y_1 == fy * 2.0 * 4
    # Doubling s must double the denominator -- the fixed behavior.
    assert denom_x_2 == 2.0 * denom_x_1
    assert denom_y_2 == 2.0 * denom_y_1


def test_old_formula_collapses_to_a_stride_independent_constant():
    """Pins down the bug itself: with s == depth_stride (every real call
    site), the old formula is provably independent of both."""
    fx, fy = 500.0, 480.0
    for s in (2, 4, 8, 16):
        denom_x, denom_y = _old_buggy_denominators(fx, fy, s, depth_stride=s)
        assert denom_x == 2.0 * fx
        assert denom_y == 2.0 * fy


def test_fixed_denominator_makes_planar_normal_stride_invariant():
    """The core regression: for a genuinely planar surface, the ESTIMATED
    normal tilt must not depend on which stride (`s`) the finite difference
    was sampled at, once the fix is applied."""
    fx, fy = 500.0, 480.0
    slope = 0.3  # dz/dx_world for the synthetic planar patch

    nx_s4, _ = _tilt_for_planar_ramp(fx, fy, s=4, depth_stride=4, slope=slope, use_fixed=True)
    nx_s16, _ = _tilt_for_planar_ramp(fx, fy, s=16, depth_stride=16, slope=slope, use_fixed=True)

    assert abs(nx_s4 - nx_s16) < 1e-9, (
        f"fixed formula must be stride-invariant for a planar surface: "
        f"nx(s=4)={nx_s4!r} nx(s=16)={nx_s16!r}"
    )


def test_old_formula_was_biased_by_stride_on_the_same_planar_surface():
    """The bug this fix removes: on the SAME planar surface, the pre-fix
    formula reports a systematically different (larger-magnitude) tilt at a
    larger stride, purely as an artifact of the cancelling `/ depth_stride`
    term -- not because the surface is any more or less tilted."""
    fx, fy = 500.0, 480.0
    slope = 0.3

    nx_s4, _ = _tilt_for_planar_ramp(fx, fy, s=4, depth_stride=4, slope=slope, use_fixed=False)
    nx_s16, _ = _tilt_for_planar_ramp(fx, fy, s=16, depth_stride=16, slope=slope, use_fixed=False)

    assert abs(nx_s16) > abs(nx_s4) * 3.0, (
        "expected the old (buggy) formula to report a much larger tilt "
        f"magnitude at stride 16 than at stride 4 for an identical planar "
        f"surface (this is the bug the fix removes): nx(s=4)={nx_s4!r} "
        f"nx(s=16)={nx_s16!r}"
    )


def test_grazing_rejection_bias_is_removed_by_the_fix():
    """Concrete consequence: at a fixed max_view_angle threshold, the SAME
    planar surface must not flip from accepted to rejected purely because of
    a stride change, once the fix is applied -- it did, before the fix, for
    a moderately tilted surface near the threshold."""
    fx, fy = 500.0, 480.0
    slope = 100.0  # chosen so the pre-fix bias crosses a realistic threshold
    max_view_angle_deg = 70.0
    cos_thresh = float(np.cos(np.deg2rad(max_view_angle_deg)))

    # Old (buggy): stride swing changes the accept/reject verdict.
    _, nz_old_s4 = _tilt_for_planar_ramp(fx, fy, s=4, depth_stride=4, slope=slope, use_fixed=False)
    _, nz_old_s16 = _tilt_for_planar_ramp(fx, fy, s=16, depth_stride=16, slope=slope, use_fixed=False)
    old_verdicts = {nz_old_s4 > cos_thresh, nz_old_s16 > cos_thresh}
    assert len(old_verdicts) == 2, (
        "expected the pre-fix formula to disagree on grazing-acceptance "
        f"between stride 4 and stride 16 for the same surface: "
        f"nz(s=4)={nz_old_s4!r} nz(s=16)={nz_old_s16!r}"
    )

    # Fixed: identical verdict at any stride for the same surface.
    _, nz_new_s4 = _tilt_for_planar_ramp(fx, fy, s=4, depth_stride=4, slope=slope, use_fixed=True)
    _, nz_new_s16 = _tilt_for_planar_ramp(fx, fy, s=16, depth_stride=16, slope=slope, use_fixed=True)
    assert (nz_new_s4 > cos_thresh) == (nz_new_s16 > cos_thresh)
    assert abs(nz_new_s4 - nz_new_s16) < 1e-9
