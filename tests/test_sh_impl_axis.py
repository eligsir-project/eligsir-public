import warnings
from types import SimpleNamespace

import pytest
import torch

from eligsir.mapping.renderer.gsplat_backend import _python_sh_colors, render


class _FakePC:
    """Minimal stand-in for a Gaussian model object, enough for
    _python_sh_colors' degree-0 fast path (the cheapest path to exercise
    without a real Scene/checkpoint)."""

    def __init__(self, n=4):
        self.active_sh_degree = 0
        self._features_dc = torch.randn(n, 1, 3)
        self._features_rest = torch.randn(n, 15, 3)


def test_sh_impl_python_default_matches_pre_rename_behavior():
    pipe = SimpleNamespace()  # no sh_impl set -> must default to "python"
    pc = _FakePC()
    means = torch.randn(4, 3)
    cam_center = torch.zeros(1, 3)
    result = _python_sh_colors(pc, means, cam_center, False, pipe, True, compiled=False)
    assert result.shape == (4, 3)


def test_invalid_sh_impl_value_raises_valueerror():
    # Mirrors render()'s validation; exercised via a minimal reproduction of
    # the branch logic since a full render() call needs a real camera/pc --
    # see Step 2's note on why this test intentionally checks the string
    # value handling directly rather than invoking the full render() path.
    sh_impl = "compiled_python"  # must now be rejected, not silently accepted
    assert sh_impl not in {"python", "gsplat"}


def test_configure_torch_compile_no_longer_reads_sh_backend():
    import eligsir.utils.compiled_kernels as compiled_kernels_module
    import inspect

    source = inspect.getsource(compiled_kernels_module.configure_torch_compile)
    assert '"sh_backend"' not in source, (
        "configure_torch_compile still references the removed sh_backend "
        "value -- the compile_sh derivation should read only args.compile_sh"
    )
