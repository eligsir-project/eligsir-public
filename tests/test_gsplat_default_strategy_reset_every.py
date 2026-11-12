"""GsplatDefaultStrategy.initialize_state must thread opacity_reset_interval
into gsplat's DefaultStrategy as reset_every -- previously hardcoded to
999_999_999 (effectively disabled), which removes DefaultStrategy's only
self-limiting mechanism (it has no cap_max) and lets N grow unbounded.
Stubs gsplat.strategy so this runs without a real gsplat install.

The stub is installed through ``monkeypatch.setitem(sys.modules, ...)`` so it
is REMOVED again at teardown. It used to assign ``sys.modules["gsplat"]``
directly, which leaked the fake module into every later test in the session --
a real, observed cross-test pollution: any downstream test reading
``gsplat.__version__`` or ``gsplat.cuda._wrapper._C`` got the fake's missing
attributes instead of the real 1.5.3 install, and failed for reasons that had
nothing to do with the code under test (e.g. "gsplat version changed
(None != '1.5.3')"). Found while running the full suite for issue #106.
"""
import sys
import types
from types import SimpleNamespace


def _install_fake_gsplat(captured_kwargs, monkeypatch):
    fake_strategy_mod = types.ModuleType("gsplat.strategy")

    class FakeDefaultStrategy:
        # Explicit named params (not **kwargs) to mirror gsplat's real signature --
        # GsplatDefaultStrategy filters its requested kwargs against
        # inspect.signature(DefaultStrategy).parameters, so a **kwargs catch-all
        # here would make every key "supported" and defeat that filter's own test
        # coverage.
        def __init__(self, prune_opa=0.005, grow_grad2d=0.0002, grow_scale3d=0.01,
                     refine_start_iter=500, refine_stop_iter=15000, refine_every=100,
                     reset_every=3000, verbose=False):
            captured_kwargs.update(dict(
                prune_opa=prune_opa, grow_grad2d=grow_grad2d, grow_scale3d=grow_scale3d,
                refine_start_iter=refine_start_iter, refine_stop_iter=refine_stop_iter,
                refine_every=refine_every, reset_every=reset_every, verbose=verbose,
            ))

        def initialize_state(self, scene_scale=1.0):
            return {"scene_scale": scene_scale}

    fake_strategy_mod.DefaultStrategy = FakeDefaultStrategy
    fake_gsplat_mod = types.ModuleType("gsplat")
    fake_gsplat_mod.strategy = fake_strategy_mod
    # setitem, NOT direct assignment: pytest restores the real modules at
    # teardown. See the module docstring for what direct assignment broke.
    monkeypatch.setitem(sys.modules, "gsplat", fake_gsplat_mod)
    monkeypatch.setitem(sys.modules, "gsplat.strategy", fake_strategy_mod)
    # `utils.strategies.mcmc.baseline` may already be imported (and may have
    # bound the REAL gsplat at import time); drop it so each test re-imports
    # it against the stub, and so the restored real module is what any later
    # test sees.
    monkeypatch.delitem(sys.modules, "utils.strategies.mcmc.baseline", raising=False)


def test_reset_every_defaults_to_opacity_reset_interval(monkeypatch):
    captured = {}
    _install_fake_gsplat(captured, monkeypatch)
    from eligsir.utils.strategies.mcmc.baseline import GsplatDefaultStrategy

    strategy = GsplatDefaultStrategy()
    args = SimpleNamespace(opacity_reset_interval=3000)
    strategy.initialize_state(gaussians=None, args=args, scene_scale=1.0)
    assert captured["reset_every"] == 3000


def test_reset_every_respects_explicit_opacity_reset_interval_override(monkeypatch):
    captured = {}
    _install_fake_gsplat(captured, monkeypatch)
    from eligsir.utils.strategies.mcmc.baseline import GsplatDefaultStrategy

    strategy = GsplatDefaultStrategy()
    args = SimpleNamespace(opacity_reset_interval=1500)
    strategy.initialize_state(gaussians=None, args=args, scene_scale=1.0)
    assert captured["reset_every"] == 1500


def test_reset_every_falls_back_to_3000_when_arg_missing(monkeypatch):
    captured = {}
    _install_fake_gsplat(captured, monkeypatch)
    from eligsir.utils.strategies.mcmc.baseline import GsplatDefaultStrategy

    strategy = GsplatDefaultStrategy()
    args = SimpleNamespace()  # no opacity_reset_interval attr at all
    strategy.initialize_state(gaussians=None, args=args, scene_scale=1.0)
    assert captured["reset_every"] == 3000
