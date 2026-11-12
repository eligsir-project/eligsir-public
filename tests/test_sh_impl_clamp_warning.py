import warnings
from types import SimpleNamespace

import torch

from eligsir.mapping.renderer.gsplat_backend import _python_sh_colors


class _FakePC:
    def __init__(self, n=4, degree=1):
        self.active_sh_degree = degree
        self._features_dc = torch.randn(n, 1, 3)
        self._features_rest = torch.randn(n, 15, 3)


def test_no_warning_for_default_clamp_mode_with_python_impl():
    # sh_impl doesn't even reach _python_sh_colors as a param -- this test
    # exercises the function most directly affected by clamp_mode with its
    # default value, confirming no warning fires there. The gsplat-impl
    # warning itself lives in render(), tested via the next two cases using
    # a lighter-weight direct check of render()'s warning-emitting branch.
    pipe = SimpleNamespace(sh_color_clamp_mode="clamped")
    pc = _FakePC()
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning -> test failure
        _python_sh_colors(pc, torch.randn(4, 3), torch.zeros(1, 3), False, pipe, True, compiled=False)


def test_warns_when_gsplat_impl_combined_with_non_default_clamp_mode():
    from eligsir.mapping.renderer import gsplat_backend

    pipe = SimpleNamespace(sh_impl="gsplat", sh_color_clamp_mode="lower_only")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        gsplat_backend._warn_if_clamp_mode_inert(pipe)
    assert len(caught) == 1
    assert "sh_color_clamp_mode" in str(caught[0].message)
    assert "gsplat" in str(caught[0].message)


def test_no_warning_when_gsplat_impl_combined_with_default_clamp_mode():
    from eligsir.mapping.renderer import gsplat_backend

    pipe = SimpleNamespace(sh_impl="gsplat", sh_color_clamp_mode="clamped")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        gsplat_backend._warn_if_clamp_mode_inert(pipe)
    assert len(caught) == 0
