"""Regression test for issue #1259's report-time resolution-tier mismatch.

A train camera that was resolution-tiered (e.g. to ``quarter``) during the
mapping phase keeps that stale ``original_image``/``score_provenance``
binding on the camera object -- ``work_budget.py`` forcing
``ResolutionTier.NATIVE`` only governs future mapping-plan decisions, it
does not re-derive cameras already bound to a tier. So by the time
``utils/comparison_report.py`` renders that same camera for its
train-metrics block, the renderer (which draws at whatever
``camera.image_width``/``image_height`` currently are) renders it AT that
same stale, downsampled resolution -- the render and the cached GT end up
the *same shape*. The failure is not a shape mismatch: the GT's
``score_provenance`` still carries the mapping-time tier's declared resize
transform, while the freshly rendered tensor's provenance has no such
step, and ``utils/score_contract.py``'s ``validate_score_pair`` correctly
refuses the resulting transform-history mismatch.

This test reproduces the exact failure at the ``validate_score_pair`` layer
(no GPU render pipeline needed) and confirms
``utils.comparison_report._restore_native_resolution_for_report`` clears it.
"""

from types import SimpleNamespace

import pytest
import torch

from eligsir.contracts.control import ResolutionTier
from eligsir.mapping.resolution_materialization import materialize_resolution_tier
from eligsir.utils.score_contract import declare_canonical_score_tensor, validate_score_pair


def _mapping_camera():
    rgb = torch.rand(3, 8, 16, dtype=torch.float32)
    rgb, provenance = declare_canonical_score_tensor(rgb, context="test:native_gt")
    camera = SimpleNamespace(
        original_image=rgb,
        score_provenance=provenance,
        _sensor_depth_cache=None,
        image_width=16,
        image_height=8,
        width=16,
        height=8,
        fx=80.0,
        fy=40.0,
        cx=7.5,
        cy=3.5,
        gsplat=object(),
        uid=1025,
    )
    return camera


def _native_render_pair(height: int, width: int):
    render = torch.rand(3, height, width, dtype=torch.float32)
    render, render_provenance = declare_canonical_score_tensor(
        render, context="test:native_render"
    )
    return render, render_provenance


def test_quarter_tiered_camera_vs_native_render_is_a_real_score_domain_mismatch():
    """Establishes the bug: this is exactly what the report hit.

    The renderer (``gaussian_renderer``'s ``render``/``render_batch``) draws
    at whatever ``camera.image_width``/``image_height`` currently are -- it
    has no idea those attributes were last set by a mapping-phase resolution
    tier, so a train camera still bound to ``quarter`` renders (and reads its
    GT) at the stale, downsampled resolution rather than at native. Both
    tensors end up the SAME shape (this is not a plain shape mismatch), but
    the GT provenance still carries the declared "resize for canonical
    mapping supervision tier quarter" transform step that the freshly
    rendered tensor's provenance has no counterpart for -- exactly the
    "Transform mismatch: GT and render use different declared score-grid
    transforms (...) vs ()" error the real report hit.
    """
    camera = _mapping_camera()
    materialize_resolution_tier(camera, ResolutionTier.QUARTER)
    assert tuple(camera.original_image.shape[-2:]) == (2, 4)

    # A fresh render at the camera's current (stale, quarter-tier) dimensions,
    # with no declared resize history of its own -- exactly what the
    # renderer's own score-provenance attachment produces.
    render, render_provenance = _native_render_pair(2, 4)

    with pytest.raises(ValueError, match="score-domain mismatch"):
        validate_score_pair(
            render,
            camera.original_image,
            render_provenance=render_provenance,
            gt_provenance=camera.score_provenance,
        )


def test_restore_native_resolution_for_report_clears_the_mismatch():
    from eligsir.utils.comparison_report import _restore_native_resolution_for_report

    camera = _mapping_camera()
    materialize_resolution_tier(camera, ResolutionTier.QUARTER)
    assert camera._canonical_resolution_tier is ResolutionTier.QUARTER

    _restore_native_resolution_for_report([camera])

    assert camera._canonical_resolution_tier is ResolutionTier.NATIVE
    assert tuple(camera.original_image.shape[-2:]) == (8, 16)
    assert (camera.image_width, camera.image_height) == (16, 8)

    render, render_provenance = _native_render_pair(8, 16)
    audit = validate_score_pair(
        render,
        camera.original_image,
        render_provenance=render_provenance,
        gt_provenance=camera.score_provenance,
    )
    assert audit.resolution == (8, 16)


def test_restore_native_resolution_for_report_is_a_noop_for_already_native_cameras():
    from eligsir.utils.comparison_report import _restore_native_resolution_for_report

    camera = _mapping_camera()
    # Never touched by materialize_resolution_tier -- `_canonical_resolution_tier`
    # is absent, exactly like a camera that never went through mapping-phase
    # tiering (e.g. a held-out test camera).
    original_image = camera.original_image

    _restore_native_resolution_for_report([camera])

    assert camera.original_image is original_image


def test_restore_native_resolution_for_report_is_idempotent_when_already_native():
    from eligsir.utils.comparison_report import _restore_native_resolution_for_report

    camera = _mapping_camera()
    materialize_resolution_tier(camera, ResolutionTier.NATIVE)
    original_image = camera.original_image

    _restore_native_resolution_for_report([camera])

    assert camera.original_image is original_image
    assert camera._canonical_resolution_tier is ResolutionTier.NATIVE
