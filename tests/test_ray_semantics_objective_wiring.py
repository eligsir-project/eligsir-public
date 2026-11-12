"""Wiring/contract tests for the tri-state ray semantics integration into
``eligsir/optimization/gaussian/objective.py`` that don't require a real
GPU render (those live in
``eligsir/optimization/gaussian/tests/`` and are skipped without
CUDA+gsplat per that package's convention — see e.g.
``test_optimizer_session_integration.py``). These check the parts that are
plain Python/CPU-checkable:

- deliverable 3 (legacy carving still exists, unmodified, as an explicit
  opt-in): the flag still exists and defaults False, independent of the new
  ray-semantics flag.
- new flags all default OFF (plan section 2.4.2).
- ``LossAssemblyInputs`` gained a ``background`` field without breaking any
  existing positional/keyword call site (engine.py passes it; tests that
  don't pass it still work because of the default).
- the classification helper fails open (returns None, doesn't raise) when
  the camera object doesn't expose the intrinsics it needs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

torch = pytest.importorskip("torch")


def _default_streaming():
    from arguments import StreamingParams
    parser = argparse.ArgumentParser()
    return StreamingParams(parser)


def test_new_flags_default_off():
    args = _default_streaming()
    assert args.streaming_ray_semantics_enabled is False
    assert args.streaming_appearance_only_enabled is False
    # Legacy carve flag is untouched by this PR -- still exists, still off.
    assert args.streaming_carve_invalid_depth is False


def test_ray_semantics_and_carve_flags_are_independent_knobs():
    args = _default_streaming()
    args.streaming_carve_invalid_depth = True
    # Deliverable 3: setting legacy carve doesn't implicitly touch/require
    # ray semantics, and vice versa -- two independently gated code paths.
    assert args.streaming_ray_semantics_enabled is False
    args2 = _default_streaming()
    args2.streaming_ray_semantics_enabled = True
    assert args2.streaming_carve_invalid_depth is False


def test_loss_assembly_inputs_background_field_defaults_none():
    from eligsir.optimization.gaussian.objective import LossAssemblyInputs
    from eligsir.optimization.gaussian.contracts import RenderOutputs

    inputs = LossAssemblyInputs(
        gaussians=None, viewpoint_cam=None, args=None, opt=None, pipe=None,
        render_outputs=RenderOutputs(payload={}), image=torch.zeros(3, 2, 2),
        iteration=0, n_frames_ingested=0,
    )
    assert inputs.background is None


def test_classify_view_rays_fails_open_without_camera_intrinsics():
    from eligsir.optimization.gaussian.objective import _classify_view_rays

    class _BareCam:
        pass

    result = _classify_view_rays(_BareCam(), torch.zeros(2, 2), _default_streaming(), None, 0)
    assert result is None


def test_classify_view_rays_fails_open_when_intrinsics_are_none():
    """Camera.__init__ (scene/cameras.py) defaults fx/fy/cx/cy=None for
    sources that only populate FoVx/FoVy (e.g. COLMAP-style). hasattr()
    alone is True for a None-valued attribute, so this must be checked
    explicitly or float(None) raises and defeats the documented
    fail-open contract."""
    from eligsir.optimization.gaussian.objective import _classify_view_rays

    class _ColmapStyleCam:
        fx = None
        fy = None
        cx = None
        cy = None
        world_view_transform = torch.eye(4)

    result = _classify_view_rays(_ColmapStyleCam(), torch.zeros(2, 2), _default_streaming(), None, 0)
    assert result is None


def test_get_ray_consensus_index_is_cached_on_streaming_scene():
    from eligsir.optimization.gaussian.objective import _get_ray_consensus_index

    class _Scene:
        pass

    scene = _Scene()
    args = _default_streaming()
    idx1 = _get_ray_consensus_index(scene, args)
    idx2 = _get_ray_consensus_index(scene, args)
    assert idx1 is idx2  # same instance reused across calls (persists per run)
