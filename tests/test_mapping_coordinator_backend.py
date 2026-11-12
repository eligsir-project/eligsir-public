"""Unit tests for ``MappingCoordinator.backend_for`` (Wave 2 backend-ownership
pass, 2026-08-06): confirms it is the single, cached construction site for a
run's ``GaussianMapBackend``, dispatching to the correct adapter by model
type and never rebuilding for the same model instance. Pure CPU/object-
identity checks -- no GPU or real model construction required, since both
adapters' ``__init__`` only stores a reference (see
``eligsir/map/adapters/{gsplat,legacy_graphdeco}.py``).
"""

from __future__ import annotations

from eligsir.mapping.coordinator import MappingCoordinator
from eligsir.map.adapters.gsplat import GsplatMapBackend
from eligsir.map.adapters.legacy_graphdeco import LegacyGraphdecoMapBackend
from scene.gsplat_model import GsplatGaussianModel


class _FakeLegacyModel:
    """Stands in for ``scene.gaussian_model.GaussianModel`` -- anything that
    is not a ``GsplatGaussianModel`` instance takes the legacy adapter path.
    """


def test_backend_for_dispatches_gsplat_vs_legacy():
    coordinator = MappingCoordinator()

    gsplat_model = GsplatGaussianModel.__new__(GsplatGaussianModel)  # no __init__ needed for isinstance dispatch
    legacy_model = _FakeLegacyModel()

    gsplat_backend = coordinator.backend_for(gsplat_model)
    assert isinstance(gsplat_backend, GsplatMapBackend)

    # A second, independent coordinator to avoid cross-model cache reuse.
    coordinator2 = MappingCoordinator()
    legacy_backend = coordinator2.backend_for(legacy_model)
    assert isinstance(legacy_backend, LegacyGraphdecoMapBackend)


def test_backend_for_caches_by_model_identity():
    coordinator = MappingCoordinator()
    model = GsplatGaussianModel.__new__(GsplatGaussianModel)

    backend_a = coordinator.backend_for(model)
    backend_b = coordinator.backend_for(model)
    assert backend_a is backend_b


def test_backend_for_rebuilds_for_a_different_model():
    coordinator = MappingCoordinator()
    model_1 = GsplatGaussianModel.__new__(GsplatGaussianModel)
    model_2 = GsplatGaussianModel.__new__(GsplatGaussianModel)

    backend_1 = coordinator.backend_for(model_1)
    backend_2 = coordinator.backend_for(model_2)
    assert backend_1 is not backend_2


def test_optimization_engine_for_reuses_backend_for_cache():
    """``optimization_engine_for`` must not maintain an independent backend
    cache -- it should wrap whatever ``backend_for`` returns, so there is
    exactly one backend (and one map_version/topology_version pair) per
    model for the life of the coordinator."""
    coordinator = MappingCoordinator()
    model = GsplatGaussianModel.__new__(GsplatGaussianModel)

    backend = coordinator.backend_for(model)
    engine = coordinator.optimization_engine_for(model)
    assert engine._backend is backend


def test_optimization_session_returns_the_model_scheduler_lr():
    """Engine-owned MCMC must consume the LR selected by the backend session."""
    from eligsir.map.adapters.gsplat import _GsplatOptimizationSession

    class Model:
        def update_learning_rate(self, iteration):
            assert iteration == 17
            return 0.03125

    assert _GsplatOptimizationSession(Model()).update_learning_rate(17) == 0.03125
