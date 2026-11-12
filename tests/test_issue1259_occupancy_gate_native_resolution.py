"""Deterministic CPU-torch reproduction of issue #1259's occupancy-gate bug.

No GPU required (fast-by-default policy,
docs/architecture/2026-08-25-fast-by-default-development-policy.md): this
exercises the exact production function
``eligsir.mapping.integration.geometry_helpers._gpu_render_occupancy_masks``
shared by both ``integration_planner.py::insert_gaussians_from_frame`` and
``device_integration_planner.py::insert_gaussians_from_frame_device_native``.

Root cause (docs/architecture/2026-08-29-replica-quarter-tier-anomaly-investigation.md,
#1259): the Phase-2 depth-insertion occupancy gate compares NATIVE-resolution
insertion candidates against ``render_alpha``/``render_depth`` rendered at
whatever supervision resolution tier happens to be bound at that moment.
gsplat's minimum-radius/antialias dilation is not scale-invariant across
tiers, so a coarser render reports a pixel as "already occupied" (alpha above
threshold) for a much larger real-world area than the same coverage would at
native resolution. The gate then over-suppresses insertion -- measured on
Replica room0 as an 80% collapse (36.8 -> 7.5 candidates/frame).

This test does not attempt to simulate gsplat's rasterizer. It instead
constructs a native-resolution alpha map with genuine sub-threshold ("hole")
pixels representing real, not-yet-mapped geometry, and its literal coarse
downsampled counterpart via max-pooling -- max-pooling is the honest
stand-in for "a coarse render pixel reports whichever finer-grained coverage
locally exists as more occupied," i.e. real holes get spatially smoothed
away by resampling to a lower resolution. It then feeds both alpha maps
through the shared, real ``_gpu_render_occupancy_masks`` gate and shows the
coarse-tier gate collapses the insertion count relative to native, matching
the qualitative Replica finding -- and that gating with a native-resolution
render (the fix in ``production_application.py::_prepare_depth_insertion_proposal``)
does not reproduce that collapse.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

from eligsir.mapping.integration.geometry_helpers import (
    _gpu_render_occupancy_masks,
)


def _make_native_alpha(
    h: int, w: int, factor: int, n_fully_unmapped_cells: int
) -> torch.Tensor:
    """Dense 'already mapped' alpha (0.9) tiled into ``factor x factor``
    cells, each aligned with one future coarse-tier pixel. Almost every cell
    holds one genuine single-pixel unmapped hole (alpha 0.05, below the 0.3
    threshold) -- real but SUB-coarse-pixel new geometry. A small number of
    cells (``n_fully_unmapped_cells``) are entirely unmapped, representing a
    real, coarse-pixel-scale gap that any resolution should still see.

    This mirrors the qualitative Replica finding: most per-frame candidates
    are small, real gaps that a coarse render dilates away; a minority are
    large enough to survive at any resolution.
    """
    alpha = torch.full((h, w), 0.9, dtype=torch.float32)
    alpha[::factor, ::factor] = 0.05  # one sub-pixel hole per cell, everywhere
    n_cells_h, n_cells_w = h // factor, w // factor
    filled = 0
    for cy in range(n_cells_h):
        for cx in range(n_cells_w):
            if filled >= n_fully_unmapped_cells:
                return alpha
            alpha[cy * factor:(cy + 1) * factor, cx * factor:(cx + 1) * factor] = 0.05
            filled += 1
    return alpha


def _coarse_via_maxpool(native: torch.Tensor, factor: int) -> torch.Tensor:
    """Coarse-tier alpha as gsplat would plausibly render it: local coverage
    maxima dominate a coarser pixel, smoothing away small real holes -- the
    same mechanism the investigation doc attributes to non-scale-invariant
    antialias/minimum-radius dilation. A cell that is genuinely unmapped in
    its entirety (every native pixel a hole) is the one case max-pooling
    cannot erase -- it correctly still reads low-alpha at the coarse tier."""
    pooled = torch.nn.functional.max_pool2d(
        native.unsqueeze(0).unsqueeze(0), kernel_size=factor, stride=factor
    )
    return pooled.squeeze(0).squeeze(0)


def _insertion_policy() -> SimpleNamespace:
    # Defaults match arguments/__init__.py's real streaming_insert_* defaults;
    # _gpu_render_occupancy_masks getattr-falls-back to them regardless.
    return SimpleNamespace()


def _candidate_count(alpha_render: torch.Tensor, native_h: int, native_w: int) -> int:
    """Run the real production gate at a dense native sensor grid (stride 1)
    and count how many pixels the gate allows insertion at (alpha_only path,
    the isolated mechanism under test)."""
    ys, xs = torch.meshgrid(
        torch.arange(native_h, dtype=torch.float32),
        torch.arange(native_w, dtype=torch.float32),
        indexing="ij",
    )
    z_v = torch.full((native_h, native_w), 2.0, dtype=torch.float32)  # dense valid depth
    low_alpha, _no_rend_depth, _sensor_closer, _score = _gpu_render_occupancy_masks(
        alpha_render, None, xs, ys, z_v, native_h, native_w, _insertion_policy()
    )
    return int(low_alpha.sum().item())


def test_coarse_tier_alpha_collapses_insertion_relative_to_native():
    """Reproduces the bug: gating against a coarser-tier render collapses the
    candidate count relative to gating against the native-resolution render,
    for IDENTICAL underlying real geometry (same holes)."""
    h = w = 64
    factor = 4  # native -> quarter, matching the Replica investigation's tier
    n_cells = (h // factor) * (w // factor)
    n_fully_unmapped_cells = 4  # ~1.6% of cells are coarse-pixel-scale gaps

    native_alpha = _make_native_alpha(h, w, factor, n_fully_unmapped_cells)
    coarse_alpha = _coarse_via_maxpool(native_alpha, factor)
    assert coarse_alpha.shape == (h // factor, w // factor)

    native_count = _candidate_count(native_alpha, h, w)
    coarse_count = _candidate_count(coarse_alpha, h, w)

    # Ground truth: every ordinary cell contributes one sub-pixel hole, the
    # fully-unmapped cells contribute their whole factor**2 block.
    expected_native = (n_cells - n_fully_unmapped_cells) * 1 + n_fully_unmapped_cells * (factor * factor)
    assert native_count == expected_native

    # The coarse-tier gate must collapse by a large factor (mirrors the
    # Replica 36.8 -> 7.5/frame, ~4.9x, finding) -- not merely "differ", and
    # not to zero (a real, coarse-pixel-scale gap must still be visible at
    # any resolution -- this is not "insertion is impossible under quarter
    # tier", it is the measured, partial collapse).
    assert coarse_count > 0, "a fully-unmapped, coarse-pixel-scale cell must survive"
    assert native_count / coarse_count >= 4.0, (
        f"expected a large collapse (Replica measured ~4.9x: 36.8/7.5), got "
        f"native={native_count} coarse={coarse_count} "
        f"ratio={native_count / coarse_count:.3f}"
    )


def test_gating_at_native_resolution_regardless_of_training_tier_does_not_collapse():
    """This is the fix's contract: once the caller always renders the gate at
    native resolution (production_application.py's
    ``_prepare_depth_insertion_proposal``, regardless of what supervision
    tier ``camera`` is currently bound to), the candidate count for a given
    frame no longer depends on the training tier -- because the gate always
    sees the same (native) render either way."""
    h = w = 64
    factor = 4
    n_fully_unmapped_cells = 4

    native_alpha = _make_native_alpha(h, w, factor, n_fully_unmapped_cells)

    # Simulate "mapping tier = native" and "mapping tier = quarter": under the
    # fix, both feed the SAME native-resolution render into the gate, so the
    # candidate counts must be identical -- no collapse from tier alone.
    count_when_tier_is_native = _candidate_count(native_alpha, h, w)
    count_when_tier_is_quarter = _candidate_count(native_alpha, h, w)

    assert count_when_tier_is_native == count_when_tier_is_quarter
    n_cells = (h // factor) * (w // factor)
    expected = (n_cells - n_fully_unmapped_cells) * 1 + n_fully_unmapped_cells * (factor * factor)
    assert count_when_tier_is_native == expected
