"""Geometry-prior components for streaming RGB-D Gaussian mapping."""

from __future__ import annotations

__all__ = ["GeometryPriorProvider", "SurfacePrior"]

# Lazy re-export (PEP 562): geometry.priors imports torch, but several
# sibling submodules (geometry.depth_filtering, geometry.surface_cells) are
# deliberately pure numpy so they can be unit-tested on any host without a
# torch/gsplat/ROCm stack -- see geometry/depth_filtering.py's own docstring
# ("Pure numpy, no torch/torchvision/gsplat import chain -- safe to
# unit-test on any host"). An eager `from geometry.priors import ...` here
# ran on EVERY `import geometry.<anything>` (Python always executes a
# package's __init__.py before any of its submodules), silently breaking
# that promise -- `import geometry.depth_filtering` forced a torch import
# it never needed. Found 2026-08-13 auditing issue #423's CI test-discovery
# gap: tests/test_fast_nanmedian.py and tests/test_scene_complexity.py both
# import a torch-free submodule via this package and failed with
# ModuleNotFoundError('torch') in a torch-less environment despite neither
# test itself needing torch. Deferred access preserves the exact same
# public API (`from geometry import GeometryPriorProvider` still works)
# without paying the torch import cost unless that name is actually used.
def __getattr__(name: str):
    if name in __all__:
        from geometry.priors import GeometryPriorProvider, SurfacePrior

        return {"GeometryPriorProvider": GeometryPriorProvider, "SurfacePrior": SurfacePrior}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
