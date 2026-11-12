"""Regression test for issue #1259: a cold-restored camera must not carry
stale `score_provenance` from a prior tier/materialization into a later
native-tier (factor==1) resolution bind.

Real-world sequence that raised this on a ScanNet++ run (see
`docs/architecture/*issue1259*` and train.log tracebacks in
output/scannetpp_diag_repro3): a camera is bound to a derived mapping tier
(e.g. EIGHTH), its live payload is later retired (`payload_residency.py`'s
`clear_live_payload`), and then cold-restored (`production_application.py`'s
`_restore_camera_payload_from_cold`) for post-mapping refinement, which runs
at NATIVE tier. If either the retire step leaves `camera.score_provenance`
behind, or the cold-restore step installs fresh native pixels without
re-declaring provenance for them, `materialize_resolution_tier`'s `factor==1`
branch reuses the stale provenance verbatim (it only rewrites the
`resolution` field when `factor != 1`) -- so the tensor that finally reaches
`require_canonical_score_tensor` disagrees with its own recorded lineage and
the canonical score contract fails closed with:

    ValueError: canonical score provenance does not match tensor
    dtype/domain/resolution (...)

This test drives the exact sequence at the `resolution_materialization.py` /
`score_contract.py` seam (the smallest reproduction of the real bug), without
needing the full `MappingSession`/`production_application` machinery.
"""

from types import SimpleNamespace

import pytest
import torch

from eligsir.contracts.control import ResolutionTier
from eligsir.mapping.resolution_materialization import materialize_resolution_tier
from eligsir.utils.score_contract import (
    declare_camera_score_provenance,
    require_canonical_score_tensor,
    score_tensor_from_camera,
)


def _native_camera(height=8, width=16):
    camera = SimpleNamespace(
        original_image=torch.rand(3, height, width, dtype=torch.float32),
        _sensor_depth_cache=torch.ones((1, height, width), dtype=torch.float32),
        image_width=width,
        image_height=height,
        width=width,
        height=height,
        fx=80.0,
        fy=40.0,
        cx=float(width) / 2 - 0.5,
        cy=float(height) / 2 - 0.5,
        gsplat=object(),
        score_provenance=None,
    )
    declare_camera_score_provenance(camera, context="test_native_materialization")
    return camera


def test_stale_score_provenance_after_cold_restore_is_rejected_by_native_state():
    """`_native_state`'s pairing guard must refuse a mismatched provenance
    rather than silently pairing it with fresh pixels (issue #1259)."""
    camera = _native_camera()

    # Mapping phase: bind to a derived tier. This correctly rewrites both the
    # tensor and its provenance's `resolution` field.
    materialize_resolution_tier(camera, ResolutionTier.EIGHTH)
    assert camera.score_provenance.canonical_score_domain.resolution == (1, 2)

    # Simulate a cold restore that reinstalls brand-new NATIVE-resolution
    # pixels (as production_application.py's _restore_camera_payload_from_cold
    # does) but -- reproducing the pre-fix bug -- forgets to refresh
    # `camera.score_provenance` alongside it.
    camera.original_image = torch.rand(3, 8, 16, dtype=torch.float32)
    camera._sensor_depth_cache = torch.ones((1, 8, 16), dtype=torch.float32)
    camera.image_width = camera.width = 16
    camera.image_height = camera.height = 8
    camera._canonical_native_resolution_state = None
    camera._canonical_resolution_binding_native_state = None
    camera._canonical_resolution_tier = None
    # camera.score_provenance is deliberately left stale (still eighth-tier).

    with pytest.raises(ValueError, match="stale provenance"):
        materialize_resolution_tier(camera, ResolutionTier.NATIVE)


def test_cold_restore_with_fresh_provenance_declaration_survives_native_rebind():
    """The actual fix: re-declaring provenance at cold-restore time (mirroring
    production_application.py's fix) makes the native-tier rebind, and the
    canonical score contract, succeed end-to-end."""
    camera = _native_camera()

    materialize_resolution_tier(camera, ResolutionTier.EIGHTH)

    # Cold restore WITH the fix: fresh pixels + fresh provenance declared
    # together, exactly as production_application.py now does.
    camera.original_image = torch.rand(3, 8, 16, dtype=torch.float32)
    camera._sensor_depth_cache = torch.ones((1, 8, 16), dtype=torch.float32)
    camera.image_width = camera.width = 16
    camera.image_height = camera.height = 8
    camera._canonical_native_resolution_state = None
    camera._canonical_resolution_binding_native_state = None
    camera._canonical_resolution_tier = None
    declare_camera_score_provenance(camera, context="cold_replay_restore")

    materialize_resolution_tier(camera, ResolutionTier.NATIVE)

    gt = score_tensor_from_camera(camera)
    # Must not raise -- this is the exact call site that raised in production.
    require_canonical_score_tensor(gt, context="canonical_primary_training_spatial:gt")
